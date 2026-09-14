"""Serve a bounded local OpenAI-compatible capture run using pinned slime source."""

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

from rolloutcheck.case import MAX_CASE_BYTES, CaseError, parse_object
from rolloutcheck.cli import EXIT_CODES
from rolloutcheck.evidence import export_trace_details
from rolloutcheck.sglang_response import response_ids
from rolloutcheck.slime_http import ALIAS, SlimeHTTPCapture
from rolloutcheck.text_report import render_text
from rolloutcheck.trace import MAX_TRACE_BYTES, TraceRecorder


async def serve(
    endpoint,
    output,
    tokenizer,
    adapter_class,
    token_space,
    *,
    requests,
    port=31000,
    timeout=600,
    evidence_kind="observed_rollout",
):
    import aiohttp
    from aiohttp import web

    if type(requests) is not int or not 1 <= requests <= 10000:
        raise ValueError("requests must be from 1 to 10000")
    if not 1 <= timeout <= 3600 or not 0 <= port <= 65535:
        raise ValueError("Invalid timeout or port")
    endpoint = loopback_url(endpoint)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    path = output / "trace.jsonl"
    capture_error = None
    # Wire observations are independently read from the engine response, with no text or headers.
    wire_path = output / "wire.jsonl"
    wire_stream = wire_path.open("x", encoding="utf-8")
    wire_bytes = 0
    os.chmod(wire_path, 0o600)
    try:
        with TraceRecorder(path, trace_id="http-capture", evidence_kind=evidence_kind) as recorder:
            capture = SlimeHTTPCapture(
                recorder,
                token_space=token_space,
                eos_token_ids=[tokenizer.eos_token_id],
                request_limit=requests,
            )
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as session:

                async def forward(request):
                    nonlocal wire_bytes
                    alias = request.headers.get("X-SMG-Routing-Key", "")
                    if ALIAS.fullmatch(alias) is None:
                        raise web.HTTPBadRequest(reason="Public session alias required")
                    payload = parse_object(await request.read())
                    async with session.post(
                        endpoint + "/generate",
                        json=payload,
                        headers={"X-SMG-Routing-Key": alias},
                        allow_redirects=False,
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
                    ids = response_ids(data, expected_prompt_tokens=len(payload["input_ids"]))
                    raw_wire = (
                        json.dumps(
                            {
                                "session_id": alias,
                                "input_ids": payload["input_ids"],
                                "output_ids": ids,
                                "finish_reason": data["meta_info"]["finish_reason"],
                                "sampling_params": payload["sampling_params"],
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    wire_bytes += len(raw_wire.encode("utf-8"))
                    if wire_bytes > MAX_TRACE_BYTES:
                        capture.abort("http_wire_size_limit")
                        raise web.HTTPBadGateway(reason="Wire capture size limit reached")
                    wire_stream.write(raw_wire)
                    wire_stream.flush()
                    return web.json_response(data)

                proxy = web.Application()
                proxy.router.add_post("/generate", forward)
                proxy_runner = web.AppRunner(proxy, access_log=None)
                adapter_runner = None
                try:
                    await proxy_runner.setup()
                    await web.TCPSite(proxy_runner, "127.0.0.1", 0).start()
                    proxy_port = proxy_runner.addresses[0][1]
                    adapter = adapter_class(
                        tokenizer=tokenizer, sglang_url=f"http://127.0.0.1:{proxy_port}"
                    )
                    capture.attach(adapter)
                    adapter_runner = web.AppRunner(
                        adapter.app, access_log=None, shutdown_timeout=30
                    )
                    await adapter_runner.setup()
                    await web.TCPSite(adapter_runner, "127.0.0.1", port).start()
                    actual_port = adapter_runner.addresses[0][1]
                    ready = {
                        "adapter_url": f"http://127.0.0.1:{actual_port}",
                        "expected_requests": requests,
                        "scope": "Local opt-in capture; no authentication or training claim",
                    }
                    save(output / "ready.tmp", ready)
                    (output / "ready.tmp").replace(output / "ready.json")
                    print(json.dumps(ready), flush=True)
                    try:
                        await asyncio.wait_for(capture.done.wait(), timeout)
                    except TimeoutError:
                        capture_error = "Client request deadline exceeded"
                        capture.abort("http_run_deadline_exceeded")
                    except asyncio.CancelledError:
                        capture_error = "Run interrupted"
                        capture.abort("http_run_interrupted")
                finally:
                    capture.accepting = False
                    if adapter_runner is not None:
                        await adapter_runner.cleanup()
                    await proxy_runner.cleanup()
                try:
                    await capture.finish(expected_requests=requests)
                except CaseError as exc:
                    capture_error = str(exc)
            receipt = {
                **capture.status(),
                "capture_error": capture_error,
                "token_space": token_space,
                "evidence_kind": evidence_kind,
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
    finally:
        wire_stream.close()
    save(output / "receipt.json", receipt)
    report, cases = export_trace_details(path, output / "evidence")
    (output / "diagnosis.txt").write_text(render_text(report, cases) + "\n", encoding="utf-8")
    save(output / "report.json", report)
    print(json.dumps({"status": report["status"], "capture": receipt}, indent=2), flush=True)
    return EXIT_CODES[report["status"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--sglang-url", type=loopback_url, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    verify_source(args.source)
    assets = verify_assets(args.assets)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    sys.path.insert(0, str(args.source.resolve()))
    import transformers
    from slime.agent.adapters.openai import OpenAIAdapter

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.assets, local_files_only=True, trust_remote_code=False
    )
    return asyncio.run(
        serve(
            args.sglang_url,
            args.output,
            tokenizer,
            OpenAIAdapter,
            "sha256:" + assets["files"]["tokenizer.json"],
            requests=args.requests,
            port=args.port,
            timeout=args.timeout,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
