"""Exercise paired timing with the real adapter and scripted HTTP; no GPU claim."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from benchmark_capture import run, summarize
from prepare_adapter import verify_source

from rolloutcheck.trace import inspect_trace


async def verify(source, output):
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    verify_source(source)
    sys.path.insert(0, str(source))
    from slime.agent.adapters.openai import OpenAIAdapter

    class Tokenizer:
        eos_token_id = 9

        def apply_chat_template(self, messages, **kwargs):
            return [1, 2] if len(messages) == 2 else [1, 2, 3, 9, 4]

        def decode(self, *args, **kwargs):
            return "hello"

    calls = []

    async def generate(request):
        body = await request.json()
        calls.append(body)
        return web.json_response(
            {
                "text": "hello",
                "meta_info": {
                    "prompt_tokens": len(body["input_ids"]),
                    "completion_tokens": 2,
                    "output_token_logprobs": [[-0.1, 3, None], [-0.1, 9, None]],
                    "finish_reason": {"type": "stop", "matched": 9},
                },
            }
        )

    app = web.Application()
    app.router.add_post("/generate", generate)
    async with TestServer(app) as server:
        result = await run(
            str(server.make_url("")).rstrip("/"),
            output,
            Tokenizer(),
            OpenAIAdapter,
            "synthetic-benchmark",
            pairs=2,
            warmup_pairs=1,
            evidence_kind="synthetic",
        )
    assert result["status"] == "COMPARABLE" and result["measured_requests"] == 8
    assert len(calls) == 12  # Four warmup requests + eight measured requests.
    rows = [json.loads(line) for line in (output / "trials.jsonl").read_text().splitlines()]
    assert summarize(rows) == result
    traces = list(output.glob("*.trace.jsonl"))
    assert len(traces) == 3  # Only on arms write files.
    for path in traces:
        report, _ = inspect_trace(path)
        assert report["status"] == "PASS"
        assert report["capture_completion"]["state"] == "complete"
    rows[1]["phase"] = "measured"
    try:
        summarize(rows)
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate measured arm must fail")
    print("Paired benchmark HTTP contract passed; scripted responses, no timing claim.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(verify(args.source.resolve(), args.output))
