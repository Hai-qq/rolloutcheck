"""Drive the unmodified slime HTTP adapter using a local Transformers model service.

The service emulates the specific /generate response fields used by slime. It is
NOT SGLang: scheduler, batching, cancellation and SGLang EOS semantics are untested.
Everything listens only on loopback and shuts down after this two-turn experiment.
"""

import argparse
import asyncio
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"

from prepare_adapter import verify_source  # noqa: E402
from prepare_assets import verify_assets  # noqa: E402

from rolloutcheck.slime_capture import SLIME_REVISION, SlimeDebugCapture, TurnContext  # noqa: E402
from rolloutcheck.trace import TraceRecorder, inspect_trace  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformers"))
from prepare_model import REVISION, WEIGHT_SHA256, verify_weights  # noqa: E402

CONTRACT = {"version": "history-prefix/v1", "mode": "append_only", "history_policy": "preserved"}


async def capture(assets, source, output, device):
    import torch
    import transformers
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    source_lock = verify_source(source)
    assert source_lock["revision"] == SLIME_REVISION
    asset_lock = verify_assets(assets)
    verify_weights(assets)
    sys.path.insert(0, str(source))
    import slime.agent.adapters.common as common
    from slime.agent.adapters.openai import OpenAIAdapter

    assert Path(common.__file__).resolve() == source / "slime/agent/adapters/common.py"
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        assets, local_files_only=True, trust_remote_code=False
    )
    if device == "auto":
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
    model = (
        transformers.AutoModelForCausalLM.from_pretrained(
            assets,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        .to(device)
        .eval()
    )
    observations = []

    async def generate(request):
        body = await request.json()
        assert body["return_logprob"] is True
        submitted = list(body["input_ids"])
        input_tensor = torch.tensor([submitted], device=device)
        config = transformers.GenerationConfig(
            max_new_tokens=body["sampling_params"]["max_new_tokens"],
            do_sample=False,
            num_beams=1,
            use_cache=True,
            max_time=90,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        started = time.perf_counter()
        # Intentionally synchronous: this is a single-request local test service,
        # not a production server. The CPU/CUDA/MPS device is synchronized by .cpu().
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_tensor,
                attention_mask=torch.ones_like(input_tensor),
                generation_config=config,
                use_model_defaults=False,
            )
            full = generated.sequences[0].detach().cpu().tolist()
            assert full[: len(submitted)] == submitted
            generated_ids = full[len(submitted) :]
            scores = (
                model.compute_transition_scores(
                    generated.sequences,
                    generated.scores,
                    normalize_logits=True,
                )[0]
                .detach()
                .cpu()
                .tolist()
            )
        assert generated_ids and len(scores) == len(generated_ids)
        eos = generated_ids[-1] == tokenizer.eos_token_id
        if not observations:
            assert eos, "First response must terminate with EOS for this experiment"
            assert "</think>" in tokenizer.decode(generated_ids, skip_special_tokens=False)
        finish = (
            "stop"
            if eos
            else ("length" if len(generated_ids) == config.max_new_tokens else "abort")
        )
        if finish == "abort":
            raise RuntimeError("Generation stopped before the expected EOS/length boundary")
        observations.append(
            {
                "input_ids": submitted,
                "output_ids": generated_ids,
                "generation_seconds": time.perf_counter() - started,
                "finish_reason": finish,
                "generation_config": config.to_dict(),
            }
        )
        print(
            f"actual generation {len(observations)}: {len(generated_ids)} output tokens", flush=True
        )
        return web.json_response(
            {
                "meta_info": {
                    "output_token_logprobs": [
                        [score, token, None]
                        for score, token in zip(scores, generated_ids, strict=True)
                    ],
                    "finish_reason": {"type": finish},
                }
            }
        )

    app = web.Application()
    app.router.add_post("/generate", generate)
    token_space = "sha256:" + asset_lock["files"]["tokenizer.json"]
    active_context = None
    trace_path = output / "adapter.trace.jsonl"
    with TraceRecorder(
        trace_path, trace_id="slime-local-qwen", evidence_kind="observed_rollout"
    ) as writer:
        callback = SlimeDebugCapture(
            writer, lambda *args: active_context, eos_token_ids=[tokenizer.eos_token_id]
        )
        async with TestServer(app) as server:
            adapter = OpenAIAdapter(
                tokenizer=tokenizer, sglang_url=str(server.make_url("")), debug_callback=callback
            )
            sid = "local-experiment-session"
            adapter.open_session(sid)
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2 + 2? Give a brief explanation."},
            ]
            async with TestClient(TestServer(adapter.app)) as client:
                for index, limit in enumerate([256, 16]):
                    active_context = TurnContext(
                        "public-qwen-session",
                        "main",
                        str(index + 1),
                        str(index) if index else None,
                        token_space,
                        CONTRACT,
                    )
                    response = await client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "local-qwen",
                            "messages": messages,
                            "max_tokens": limit,
                            "metadata": {"session_id": sid},
                        },
                    )
                    assert response.status == 200, await response.text()
                    data = await response.json()
                    messages += [
                        data["choices"][0]["message"],
                        {"role": "user", "content": "Reply with one word: done."},
                    ]
            await adapter.shutdown_session(sid)
            callback.raise_if_failed(expected_turns=2)
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    records = [record for record in records if record["record_type"] == "generation"]
    assert len(records) == len(observations) == callback.captured == 2
    for record, observation in zip(records, observations, strict=True):
        assert record["input_ids"] == observation["input_ids"]
        assert record["output_ids"] == observation["output_ids"]
    report, cases = inspect_trace(trace_path)
    assert report["status"] == "FAIL", "Target re-rendering failure was not observed"
    (output / "adapter.report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "adapter.case.json").write_text(json.dumps(cases[0], indent=2) + "\n")
    (output / "engine-observations.json").write_text(json.dumps(observations, indent=2) + "\n")
    summary = {
        "execution": "real_slime_adapter_with_transformers_service",
        "slime_revision": SLIME_REVISION,
        "adapter_source": "unmodified verified snapshot",
        "model": "Qwen/Qwen3-0.6B",
        "model_revision": REVISION,
        "weight_sha256": WEIGHT_SHA256,
        "template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": device,
            "system": platform.system(),
            "machine": platform.machine(),
            "dtype": "float32",
            "attention_implementation": "eager",
        },
        "decoding": "greedy",
        "engine_generation": True,
        "sglang_execution": False,
        "status": report["status"],
        "captured": callback.captured,
        "gaps": callback.failed,
        "scope": "Real adapter over loopback HTTP plus real Transformers generation; "
        "service emulates /generate fields. Not SGLang scheduler/stop-semantics validation.",
        "logprobs": "Computed from generation scores; no training correctness claim",
        "engine_ids_equal_adapter_capture": True,
        "training_experiment": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    asyncio.run(capture(args.assets.resolve(), args.source.resolve(), args.output, args.device))


if __name__ == "__main__":
    main()
