"""Run the pinned, real slime OpenAI HTTP adapter with a scripted local model server.

The adapter and HTTP transport execute unchanged. Tokenizer outputs and model
responses are synthetic test controls, not a real SGLang or model execution.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from prepare_adapter import verify_source

from rolloutcheck.case import CaseError
from rolloutcheck.slime_capture import SLIME_REVISION, SlimeDebugCapture, TurnContext
from rolloutcheck.trace import TraceRecorder, inspect_trace

CONTRACT = {"version": "history-prefix/v1", "mode": "append_only", "history_policy": "preserved"}


async def run_scenario(output, mode, adapter_class):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    first_input = [1, 2]
    second_input = [1, 2, 88] if mode in ("drift", "unchecked-drift") else [1, 2, 3, 9, 4]
    inputs = [first_input, second_input]
    outputs = [[3, 9], [5, 9]]
    requests = []

    class ScriptedTokenizer:
        def __init__(self):
            self.index = 0

        def apply_chat_template(self, *args, **kwargs):
            ids = list(inputs[self.index])
            self.index += 1
            return ids

        def decode(self, ids, **kwargs):
            return "hello"

    async def generate(request):
        body = await request.json()
        index = len(requests)
        assert body["input_ids"] == inputs[index]
        assert body["return_logprob"] is True
        requests.append(body)
        return web.json_response(
            {
                "meta_info": {
                    "output_token_logprobs": [[-0.1, token, None] for token in outputs[index]],
                    "finish_reason": {"type": "stop"},
                }
            }
        )

    upstream = web.Application()
    upstream.router.add_post("/generate", generate)
    contexts = [
        TurnContext("public-test-session", "main", "1", None, "synthetic", CONTRACT),
        TurnContext("public-test-session", "main", "2", "1", "synthetic", CONTRACT),
    ]
    active_context = None
    path = output / f"{mode}.trace.jsonl"
    with TraceRecorder(path, trace_id=f"slime-http-{mode}", evidence_kind="synthetic") as writer:
        callback = SlimeDebugCapture(writer, lambda *args: active_context, eos_token_ids=[9])
        async with TestServer(upstream) as server:
            adapter = adapter_class(
                tokenizer=ScriptedTokenizer(),
                sglang_url=str(server.make_url("")),
                debug_callback=callback,
            )
            # This identifier intentionally resembles a private session credential.
            raw_sid = "private-session-do-not-persist"
            adapter.open_session(raw_sid)
            async with TestClient(TestServer(adapter.app)) as client:
                messages = [{"role": "user", "content": "first"}]
                for index in range(2):
                    # Controlled driver supplies ancestry before each sequential request.
                    active_context = (
                        None if mode == "missing-context" and index == 1 else contexts[index]
                    )
                    response = await client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "scripted",
                            "max_tokens": 2,
                            "messages": messages,
                            "metadata": {"session_id": raw_sid},
                        },
                    )
                    assert response.status == 200, await response.text()
                    data = await response.json()
                    messages += [data["choices"][0]["message"], {"role": "user", "content": "next"}]
            await adapter.shutdown_session(raw_sid)
            if mode in ("missing-context", "count-mismatch"):
                try:
                    callback.raise_if_failed(expected_turns=3 if mode == "count-mismatch" else 2)
                except CaseError:
                    pass
                else:
                    raise AssertionError("Missing capture must fail the collection health check")
            elif not mode.startswith("unchecked-"):
                callback.raise_if_failed(expected_turns=2)
    assert len(requests) == 2
    assert raw_sid not in path.read_text()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        if record["record_type"] == "generation":
            index = int(record["turn_id"]) - 1
            assert record["input_ids"] == requests[index]["input_ids"]
            assert record["output_ids"] == outputs[index]
    report, cases = inspect_trace(path)
    expected = {
        "clean": "PASS", "drift": "FAIL", "missing-context": "INCONCLUSIVE",
        "unchecked-clean": "INCONCLUSIVE", "unchecked-drift": "FAIL",
        "count-mismatch": "INCONCLUSIVE",
    }[mode]
    assert report["status"] == expected
    expected_completion = "complete" if mode in ("clean", "drift") else "missing"
    assert report["capture_completion"]["state"] == expected_completion
    if mode == "unchecked-clean":
        assert report["counts"] == {"PASS": 1} and "capture_gaps" not in report
    (output / f"{mode}.report.json").write_text(json.dumps(report, indent=2) + "\n")
    if cases:
        (output / f"{mode}.case.json").write_text(json.dumps(cases[0], indent=2) + "\n")
    return {
        "scenario": mode,
        "requests": len(requests),
        "captured": callback.captured,
        "gaps": callback.failed,
        "status": report["status"],
        "completion": report["capture_completion"]["state"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    lock = verify_source(source)
    assert lock["revision"] == SLIME_REVISION
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(source))
    import slime.agent.adapters.common as common
    from slime.agent.adapters.openai import OpenAIAdapter

    assert Path(common.__file__).resolve() == source / "slime/agent/adapters/common.py"
    results = [
        asyncio.run(run_scenario(args.output, mode, OpenAIAdapter))
        for mode in (
            "clean", "drift", "missing-context", "unchecked-clean", "unchecked-drift",
            "count-mismatch",
        )
    ]
    summary = {
        "execution": "controlled_adapter_http",
        "slime_revision": lock["revision"],
        "adapter_source": "unmodified verified snapshot",
        "transport": "loopback aiohttp",
        "model_server": "scripted test responses; not SGLang",
        "tokenizer": "scripted",
        "engine_generation": False,
        "scenarios": results,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
