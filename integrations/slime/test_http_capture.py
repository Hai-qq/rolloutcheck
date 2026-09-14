"""Whole local capture workflow with real slime HTTP and a separate client process."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from prepare_adapter import verify_source
from serve_capture import serve

from rolloutcheck.evidence import verify_evidence


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

        def decode(self, ids, **kwargs):
            return "hello"

    calls = []

    async def generate(request):
        payload = await request.json()
        alias = request.headers["X-SMG-Routing-Key"]
        calls.append(alias)
        # Make independent sessions complete in a different order.
        if alias == "public-session-1":
            await asyncio.sleep(0.03)
        return web.json_response(
            {
                "text": "hello",
                "meta_info": {
                    "prompt_tokens": len(payload["input_ids"]),
                    "completion_tokens": 2,
                    "output_token_logprobs": [[-0.1, 3, None], [-0.1, 9, None]],
                    "finish_reason": {"type": "stop", "matched": 9},
                },
            }
        )

    app = web.Application()
    app.router.add_post("/generate", generate)
    async with TestServer(app) as engine:
        server = asyncio.create_task(
            serve(
                str(engine.make_url("")).rstrip("/"),
                output,
                Tokenizer(),
                OpenAIAdapter,
                "synthetic-http",
                requests=4,
                port=0,
                timeout=20,
                evidence_kind="synthetic",
            )
        )
        try:
            for _ in range(100):
                if server.done():
                    await server
                    raise AssertionError("Server exited before client startup")
                if (output / "ready.json").exists():
                    break
                await asyncio.sleep(0.02)
            else:
                raise AssertionError("Server did not become ready")
            ready = json.loads((output / "ready.json").read_text())
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(Path(__file__).with_name("demo_client.py")),
                "--adapter-url",
                ready["adapter_url"],
                "--output",
                str(output / "client"),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), 20)
            assert process.returncode == 0, (stdout, stderr)
            assert await asyncio.wait_for(server, 20) == 0
        finally:
            if not server.done():
                server.cancel()
                await asyncio.gather(server, return_exceptions=True)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["received"] == receipt["succeeded"] == receipt["captured"] == 4
    assert receipt["failed"] == 0 and receipt["finalized"]
    report = verify_evidence(output / "evidence")
    assert report["status"] == "PASS" and report["counts"] == {"PASS": 2}
    assert report["capture_completion"]["state"] == "complete"
    assert len(calls) == 4
    # A client that never submits its declared workload must still leave an
    # independently verifiable incomplete artifact, even with zero generations.
    incomplete = output / "no-client"
    code = await serve(
        "http://127.0.0.1:1",
        incomplete,
        Tokenizer(),
        OpenAIAdapter,
        "synthetic-http",
        requests=1,
        port=0,
        timeout=1,
        evidence_kind="synthetic",
    )
    assert code == 3
    incomplete_report = verify_evidence(incomplete / "evidence")
    assert incomplete_report["status"] == "INCONCLUSIVE"
    assert incomplete_report["capture_completion"]["state"] == "missing"
    reasons = {gap["reason"] for gap in incomplete_report["capture_gaps"]}
    assert reasons == {"http_run_deadline_exceeded", "http_request_count_mismatch"}
    print("Independent client, concurrent identity, finalized and incomplete evidence: passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(verify(args.source.resolve(), args.output))
