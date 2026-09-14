"""Capture a user-managed, loopback SGLang endpoint through the real slime adapter.

Does not install/start/stop SGLang, authenticate a server, or infer its model identity.
The operator must independently record server version, launch command and model digest.
"""

import argparse
import asyncio
import ipaddress
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

from prepare_adapter import verify_source
from prepare_assets import verify_assets

from rolloutcheck.case import MAX_CASE_BYTES, parse_object
from rolloutcheck.history import inspect_case
from rolloutcheck.sglang_response import response_ids
from rolloutcheck.slime_capture import SLIME_REVISION, SlimeDebugCapture, TurnContext
from rolloutcheck.trace import TraceRecorder, inspect_trace

CONTRACT = {"version": "history-prefix/v1", "mode": "append_only", "history_policy": "preserved"}


def loopback_url(value):
    parts = urlsplit(value)
    if (
        parts.scheme != "http"
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
        or parts.path not in ("", "/")
        or not parts.hostname
        or not ipaddress.ip_address(parts.hostname).is_loopback
    ):
        raise ValueError("Use an explicit loopback HTTP endpoint such as http://127.0.0.1:30000")
    if parts.port is None:
        raise ValueError("Specify the local engine port explicitly")
    return value.rstrip("/")


def save(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


async def run(
    endpoint, output, tokenizer, adapter_class, token_space, *, evidence_kind="observed_rollout"
):
    import aiohttp
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    output.mkdir(parents=True, exist_ok=False)
    wires = []
    active_context = None
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:

        async def send(payload, label):
            async with session.post(
                endpoint + "/generate", json=payload, allow_redirects=False
            ) as reply:
                if reply.status != 200:
                    raise RuntimeError(f"Engine HTTP {reply.status}; inspect its local log")
                chunks, size = [], 0
                async for chunk in reply.content.iter_chunked(65536):
                    size += len(chunk)
                    if size > MAX_CASE_BYTES:
                        raise ValueError("Engine response exceeds the bounded capture size")
                    chunks.append(chunk)
                raw = b"".join(chunks)
            data = parse_object(raw)
            wire = {"label": label, "request": payload, "response": data}
            save(output / f"wire-{len(wires):02d}.json", wire)
            wires.append(wire)
            response_ids(data, expected_prompt_tokens=len(payload["input_ids"]))
            return data

        async def forward(request):
            payload = await request.json()
            return web.json_response(await send(payload, f"adapter-turn-{len(wires) + 1}"))

        proxy = web.Application()
        proxy.router.add_post("/generate", forward)
        path = output / "adapter.trace.jsonl"
        with TraceRecorder(
            path, trace_id="sglang-adapter", evidence_kind=evidence_kind
        ) as recorder:
            capture = SlimeDebugCapture(
                recorder, lambda *args: active_context, eos_token_ids=[tokenizer.eos_token_id]
            )
            async with TestServer(proxy) as server:
                adapter = adapter_class(
                    tokenizer=tokenizer, sglang_url=str(server.make_url("")), debug_callback=capture
                )
                sid = "rolloutcheck-public-probe"
                adapter.open_session(
                    sid, sampling_defaults={"temperature": 0, "top_p": 1, "top_k": -1}
                )
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is 2 + 2? Give a brief explanation."},
                ]
                async with TestClient(TestServer(adapter.app)) as client:
                    for index, limit in enumerate((512, 16)):
                        active_context = TurnContext(
                            "public-probe",
                            "main",
                            str(index + 1),
                            str(index) if index else None,
                            token_space,
                            CONTRACT,
                        )
                        reply = await client.post(
                            "/v1/chat/completions",
                            json={
                                "model": "local-pinned-qwen",
                                "messages": messages,
                                "max_tokens": limit,
                                "temperature": 0,
                                "metadata": {"session_id": sid},
                            },
                        )
                        if reply.status != 200:
                            raise RuntimeError(
                                f"slime HTTP {reply.status}; preserve wire files and local logs"
                            )
                        data = await reply.json()
                        if (
                            index == 0
                            and wires[0]["response"]["meta_info"]["finish_reason"]["type"] != "stop"
                        ):
                            raise RuntimeError(
                                "First generation was truncated; a completed first turn is required"
                            )
                        messages += [
                            data["choices"][0]["message"],
                            {"role": "user", "content": "Reply with one word: done."},
                        ]
                await adapter.shutdown_session(sid)
                capture.raise_if_failed(expected_turns=2)
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records = [record for record in records if record["record_type"] == "generation"]
        for record, wire in zip(records, wires, strict=True):
            assert record["input_ids"] == wire["request"]["input_ids"]
            assert record["output_ids"] == response_ids(
                wire["response"], expected_prompt_tokens=len(record["input_ids"])
            )
        report, cases = inspect_trace(path)
        save(output / "adapter.report.json", report)
        save(output / "adapter.case.json", cases[0])

        # A constructed append-only control, sharing the first observed generation.
        # This is explicitly not the upstream canonical-continuation fix.
        root = records[0]
        history = root["input_ids"] + root["output_ids"]
        suffix = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Reply with one word: done."}],
            tokenize=True,
            add_generation_prompt=True,
        )
        next_input = (
            history
            + ([] if history[-1] == tokenizer.eos_token_id else [tokenizer.eos_token_id])
            + suffix
        )
        positive_payload = {
            "input_ids": next_input,
            "return_logprob": True,
            "sampling_params": {
                "max_new_tokens": 16,
                "temperature": 0,
                "top_p": 1,
                "top_k": -1,
                "skip_special_tokens": False,
                "no_stop_trim": True,
            },
        }
        positive = await send(positive_payload, "constructed-append-only-control")
        positive_ids = response_ids(positive, expected_prompt_tokens=len(next_input))
        with TraceRecorder(
            output / "control.trace.jsonl", trace_id="sglang-control", evidence_kind=evidence_kind
        ) as writer:
            for index, inputs, outputs, finish in (
                (0, root["input_ids"], root["output_ids"], root["termination"]),
                (
                    1,
                    next_input,
                    positive_ids,
                    {"upstream_finish_reason": positive["meta_info"]["finish_reason"]},
                ),
            ):
                writer.record(
                    session_id="public-control",
                    branch_id="main",
                    turn_id=str(index + 1),
                    parent_turn_id=str(index) if index else None,
                    token_space=token_space,
                    input_ids=inputs,
                    output_ids=outputs,
                    contract=CONTRACT,
                    termination=finish,
                    metadata={
                        "control": "constructed prefix; root reuses first actual adapter generation"
                    },
                )
            writer.finalize(expected_generations=2)
        control_report, _ = inspect_trace(output / "control.trace.jsonl")
        assert control_report["status"] == "PASS"
        save(output / "control.report.json", control_report)

        # Deliberate offline mutation is synthetic, never labeled as an observed model request.
        mutation = json.loads(json.dumps(cases[0]))
        mutation["evidence"] = {
            "kind": "synthetic",
            "description": "Offline one-token mutation of saved request",
        }
        mutation["next"]["input_ids"][0] = mutation["previous"]["input_ids"][0] + 1
        assert inspect_case(mutation)["status"] == "FAIL"
        save(output / "mutation.case.json", mutation)

        probes = []
        for label, overrides in (
            ("length", {"max_new_tokens": 1, "ignore_eos": True}),
            ("stop-trim", {"stop_token_ids": [root["output_ids"][0]], "no_stop_trim": False}),
            ("stop-retain", {"stop_token_ids": [root["output_ids"][0]], "no_stop_trim": True}),
        ):
            payload = {
                "input_ids": root["input_ids"],
                "return_logprob": True,
                "sampling_params": {
                    "temperature": 0,
                    "top_p": 1,
                    "top_k": -1,
                    "max_new_tokens": 8,
                    "skip_special_tokens": False,
                    **overrides,
                },
            }
            result = await send(payload, label)
            ids = response_ids(result, expected_prompt_tokens=len(root["input_ids"]))
            probes.append(
                {
                    "probe": label,
                    "returned_tokens": len(ids),
                    "last_id": ids[-1],
                    "text": result.get("text"),
                    "finish_reason": result["meta_info"]["finish_reason"],
                }
            )
        assert probes[0]["finish_reason"]["type"] == "length"
        summary = {
            "execution": "operator-supplied-native-endpoint",
            "endpoint_identity": "not authenticated by this runner",
            "evidence_kind": evidence_kind,
            "slime_revision": SLIME_REVISION,
            "requests": len(wires),
            "adapter_captured": capture.captured,
            "adapter_ids_equal_wire": True,
            "adapter_status": report["status"],
            "constructed_control_status": control_report["status"],
            "mutation_status": "FAIL",
            "first_response_terminal_eos_present": root["output_ids"][-1] == tokenizer.eos_token_id,
            "stop_probes": probes,
            "training_experiment": False,
        }
        save(output / "summary.json", summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--sglang-url", type=loopback_url, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    source_lock, assets = verify_source(source), verify_assets(args.assets)
    assert source_lock["revision"] == SLIME_REVISION
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(source))
    import transformers
    from slime.agent.adapters.openai import OpenAIAdapter

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.assets, local_files_only=True, trust_remote_code=False
    )
    summary = asyncio.run(
        run(
            args.sglang_url,
            args.output,
            tokenizer,
            OpenAIAdapter,
            "sha256:" + assets["files"]["tokenizer.json"],
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
