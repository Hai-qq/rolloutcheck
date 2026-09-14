"""Exercise the live-endpoint runner with scripted HTTP controls; no live engine claim."""

import argparse
import asyncio
import sys
from pathlib import Path

from prepare_adapter import verify_source
from verify_sglang import loopback_url, run


async def verify(source, output):
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    verify_source(source)
    sys.path.insert(0, str(source))
    from slime.agent.adapters.openai import OpenAIAdapter

    class Tokenizer:
        eos_token_id = 9
        count = 0

        def apply_chat_template(self, *args, **kwargs):
            value = [[1, 2], [1, 2, 77], [4]][self.count]
            self.count += 1
            return value

        def decode(self, ids, **kwargs):
            return "hello"

    calls = []

    async def generate(request):
        body = await request.json()
        params = body["sampling_params"]
        stop_ids = params.get("stop_token_ids")
        if params.get("ignore_eos"):
            ids, reason, text = [5], {"type": "length"}, "one"
        elif stop_ids:
            ids, reason = [stop_ids[0]], {"type": "stop", "matched": stop_ids[0]}
            text = "stop" if params["no_stop_trim"] else ""
        else:
            ids, reason, text = [3, 9], {"type": "stop", "matched": 9}, "hello"
        calls.append(body)
        # Stream the JSON body in pieces to test bounded accumulation, not just one read().
        import json

        raw = json.dumps(
            {
                "text": text,
                "meta_info": {
                    "prompt_tokens": len(body["input_ids"]),
                    "completion_tokens": len(ids),
                    "output_token_logprobs": [[-0.1, x, None] for x in ids],
                    "finish_reason": reason,
                },
            }
        ).encode()
        response = web.StreamResponse(headers={"Content-Type": "application/json"})
        await response.prepare(request)
        await response.write(raw[:10])
        await asyncio.sleep(0.01)
        await response.write(raw[10:])
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/generate", generate)
    async with TestServer(app) as server:
        endpoint = loopback_url(str(server.make_url("")))
        summary = await run(
            endpoint,
            output,
            Tokenizer(),
            OpenAIAdapter,
            "synthetic-test-space",
            evidence_kind="synthetic",
        )
    assert len(calls) == 6
    assert summary["adapter_ids_equal_wire"] and summary["adapter_status"] == "FAIL"
    assert summary["constructed_control_status"] == "PASS"
    assert summary["mutation_status"] == "FAIL"
    assert summary["stop_probes"][1]["text"] == ""
    assert summary["stop_probes"][2]["text"] == "stop"
    print(
        "Six scripted HTTP responses passed the runner contract; no SGLang execution."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(verify(args.source.resolve(), args.output))
