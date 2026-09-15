"""Compare pinned slime retention policies at finish_session, before any optimizer.

Default: scripted tokenizer/engine. --assets and --sglang-url opt into native
inference. Raw generation and emitted Sample fields are retained separately.
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

from prepare_adapter import verify_source
from prepare_assets import verify_assets
from verify_sglang import loopback_url, save

from rolloutcheck.case import MAX_CASE_BYTES, parse_object
from rolloutcheck.evidence import export_trace
from rolloutcheck.sample_audit import audit_samples
from rolloutcheck.sglang_response import response_ids
from rolloutcheck.slime_capture import SlimeDebugCapture, TurnContext
from rolloutcheck.trace import TraceRecorder

CONTRACT = {"version": "history-prefix/v1", "mode": "append_only", "history_policy": "preserved"}


async def scenario(
    output,
    adapter_class,
    sample_class,
    tokenizer,
    endpoint,
    token_space,
    *,
    threshold,
    clean=False,
    initial_messages=None,
):
    import aiohttp
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    observed, wires = [], []
    native = endpoint is not None
    active = None
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as upstream:

        async def generate(request):
            payload = await request.json()
            if native:
                async with upstream.post(
                    endpoint + "/generate", json=payload, allow_redirects=False
                ) as reply:
                    if reply.status != 200:
                        raise web.HTTPBadGateway(reason="Engine request failed")
                    chunks, size = [], 0
                    async for chunk in reply.content.iter_chunked(65536):
                        size += len(chunk)
                        if size > MAX_CASE_BYTES:
                            raise web.HTTPBadGateway(reason="Engine response too large")
                        chunks.append(chunk)
                    data = parse_object(b"".join(chunks))
            else:
                ids = [3, 9] if not wires else [5, 9]
                data = {
                    "meta_info": {
                        "prompt_tokens": len(payload["input_ids"]),
                        "completion_tokens": len(ids),
                        "finish_reason": {"type": "stop"},
                        "output_token_logprobs": [[-0.1, token, None] for token in ids],
                    }
                }
            ids = response_ids(data, expected_prompt_tokens=len(payload["input_ids"]))
            wires.append({"input_ids": payload["input_ids"], "output_ids": ids})
            return web.json_response(data)

        proxy = web.Application()
        proxy.router.add_post("/generate", generate)
        with TraceRecorder(
            output / "trace.jsonl",
            trace_id=output.name,
            evidence_kind="observed_rollout" if native else "synthetic",
        ) as writer:
            capture = SlimeDebugCapture(
                writer, lambda *args: active, eos_token_ids=[tokenizer.eos_token_id]
            )

            def callback(*args):
                capture(*args)
                turn = args[-1]
                observed.append(
                    {
                        "turn_id": active.turn_id,
                        "input_ids": list(turn.prompt_ids),
                        "output_ids": list(turn.output_ids),
                    }
                )

            async with TestServer(proxy) as server:
                adapter = adapter_class(
                    tokenizer=tokenizer,
                    sglang_url=str(server.make_url("")),
                    fork_threshold_tokens=threshold,
                    debug_callback=callback,
                )
                messages = (
                    [dict(message) for message in initial_messages]
                    if initial_messages
                    else [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "What is 2 + 2? Give a brief explanation."},
                    ]
                )
                sid = "public-handoff"
                async with TestClient(TestServer(adapter.app)) as client:
                    for number, limit in ((1, 512), (2, 16)):
                        active = TurnContext(
                            sid,
                            "main",
                            str(number),
                            None if number == 1 else "1",
                            token_space,
                            CONTRACT,
                        )
                        response = await client.post(
                            "/v1/chat/completions",
                            json={
                                "messages": messages,
                                "metadata": {"session_id": sid},
                                "stream": False,
                                "temperature": 0,
                                "max_tokens": limit,
                            },
                        )
                        assert response.status == 200
                        reply = await response.json()
                        if number == 1:
                            assert reply["choices"][0]["finish_reason"] == "stop"
                        messages += [
                            reply["choices"][0]["message"],
                            {"role": "user", "content": "Reply with one word: done."},
                        ]
                # Actual upstream trajectory construction, not a hand-built Sample.
                emitted = await adapter.finish_session(
                    sid, base_sample=sample_class(index=0), reward=1.0
                )
                assert await adapter.finish_session(sid, base_sample=sample_class(index=0)) == []
            capture.raise_if_failed(expected_turns=2)
    assert len(observed) == len(wires) == 2
    assert all(
        {k: t[k] for k in ("input_ids", "output_ids")} == w
        for t, w in zip(observed, wires, strict=True)
    )
    samples = [
        {
            "tokens": list(s.tokens),
            "response_length": s.response_length,
            "loss_mask": list(s.loss_mask),
        }
        for s in emitted
    ]
    audit = audit_samples(observed, samples)
    raw = {
        "snapshot_version": 1,
        "evidence_kind": "observed_rollout" if native else "synthetic",
        "session_id": sid,
        "fork_threshold_tokens": threshold,
        "token_space": token_space,
        "turns": observed,
        "samples": samples,
        "scope": "Actual pinned adapter finish_session output. No optimizer/trainer executed; "
        "log probabilities, rewards and policy versions are not audited.",
    }
    save(output / "handoff.json", raw)
    save(output / "wire.json", wires)
    save(output / "audit.json", audit)
    report = export_trace(output / "trace.jsonl", output / "evidence")
    assert audit["context_status"] == "MATCHED" and audit["duplicate_training_occurrences"] == 0
    if not native:
        assert report["status"] == ("PASS" if clean else "FAIL")
        assert audit["unaccounted_generated_tokens"] == (0 if clean or threshold == 0 else 2)
    return {"name": output.name, "history_status": report["status"], "audit": audit}


async def run(source, output, assets=None, endpoint=None):
    source_lock = verify_source(source)
    sys.path.insert(0, str(source.resolve()))
    from slime.agent.adapters.openai import OpenAIAdapter
    from slime.utils.types import Sample

    imported = Path(sys.modules[OpenAIAdapter.__module__].__file__).resolve()
    if imported != source.resolve() / "slime/agent/adapters/openai.py":
        raise RuntimeError("Imported adapter does not come from the verified source directory")

    if assets is not None:
        asset_lock = verify_assets(assets)
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            assets, local_files_only=True, trust_remote_code=False
        )
        token_space = "sha256:" + asset_lock["files"]["tokenizer.json"]
    else:
        tokenizer = None
        token_space = "synthetic-handoff"
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    results = []
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2? Give a brief explanation."},
    ]
    original_messages = json.dumps(messages, sort_keys=True)
    configs = [("default", None, False), ("fork", 0, False)]
    if not endpoint:
        configs.append(("clean", None, True))
    for name, threshold, clean in configs:

        class ScriptedTokenizer:
            eos_token_id = 9

            def __init__(self, clean):
                self.clean = clean

            def apply_chat_template(self, messages, **kwargs):
                if len(messages) == 2:
                    return [1, 2]
                return [1, 2, 3, 9, 4] if self.clean else [1, 2, 8, 4]

            def decode(self, ids, **kwargs):
                return "hello"

        results.append(
            await scenario(
                output / name,
                OpenAIAdapter,
                Sample,
                tokenizer or ScriptedTokenizer(clean),
                endpoint,
                token_space,
                threshold=threshold,
                clean=clean,
                initial_messages=messages,
            )
        )
        assert json.dumps(messages, sort_keys=True) == original_messages, (
            "Scenario mutated its input"
        )
    baseline = json.loads((output / "default/wire.json").read_text())
    alternate = json.loads((output / "fork/wire.json").read_text())
    same = baseline == alternate
    summary = {
        "source_revision": source_lock["revision"],
        "evidence_kind": "observed_rollout" if endpoint else "synthetic",
        "same_generation_ids_across_policies": same,
        "scenarios": results,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Retention-policy comparison, not a universal fix or training-quality result. "
        "Threshold zero may split samples, changing context cost and reward/loss aggregation.",
    }
    save(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    assert same, "Different sampled work: do not attribute a retention change solely to policy"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--sglang-url", type=loopback_url)
    args = parser.parse_args()
    if bool(args.assets) != bool(args.sglang_url):
        parser.error("--assets and --sglang-url must be supplied together")
    asyncio.run(run(args.source, args.output, args.assets, args.sglang_url))
