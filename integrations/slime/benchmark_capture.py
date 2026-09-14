"""Paired collector-on/off timing through the pinned slime HTTP adapter.

Explicit loopback endpoint only. Does not launch an engine or modify its settings.
Both arms include the same in-memory wire observation proxy. No timing claim is
valid unless input IDs, sampling parameters, output IDs and finish reasons match.
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

from prepare_adapter import verify_source
from prepare_assets import verify_assets
from verify_sglang import CONTRACT, loopback_url, save

from rolloutcheck import __version__
from rolloutcheck.case import MAX_CASE_BYTES, parse_object
from rolloutcheck.sglang_response import response_ids
from rolloutcheck.slime_capture import SLIME_REVISION, SlimeDebugCapture, TurnContext
from rolloutcheck.trace import MAX_TRACE_BYTES, TraceRecorder, inspect_trace, inspect_trace_bytes


def order_plan(pairs, seed):
    if type(pairs) is not int or not 2 <= pairs <= 100 or pairs % 2:
        raise ValueError("Use an even number of pairs from 2 to 100")
    orders = [["off", "on"], ["on", "off"]] * (pairs // 2)
    random.Random(seed).shuffle(orders)
    return orders


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def validate_schedule(rows, protocol):
    plan = order_plan(protocol["pairs"], protocol["seed"])
    warmup = protocol["warmup_pairs"]
    if type(warmup) is not int or not 1 <= warmup <= 10 or protocol["order"] != plan:
        raise ValueError("Invalid saved benchmark schedule")
    expected = [
        (phase, pair, mode)
        for phase, orders in (("warmup", [["off", "on"]] * warmup), ("measured", plan))
        for pair, order in enumerate(orders)
        for mode in order
    ]
    actual = [(row["phase"], row["pair"], row["mode"]) for row in rows]
    if actual != expected:
        raise ValueError("Incomplete or reordered benchmark schedule")


def summarize(rows, *, seed=0):
    measured = [row for row in rows if row["phase"] == "measured"]
    blocks = {}
    for row in measured:
        mode = row["mode"]
        if mode not in ("on", "off"):
            raise ValueError("Unknown benchmark arm")
        block = blocks.setdefault(row["pair"], {})
        if mode in block:
            raise ValueError("Duplicate arm in a pair")
        for field in ("elapsed_seconds", "finalize_seconds"):
            value = row[field]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid timing sample")
        if row["elapsed_seconds"] <= 0:
            raise ValueError("Elapsed time must be positive")
        if len(row["turns"]) != 2 or (mode == "on" and len(row["callback_seconds"]) != 2):
            raise ValueError("Expected two turns per trial")
        if mode == "off" and (row["callback_seconds"] or row["finalize_seconds"]):
            raise ValueError("Off arm must not run the collector")
        if mode == "on" and row.get("capture_complete") is not True:
            raise ValueError("On arm must have completed capture")
        for value in row["callback_seconds"]:
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid callback timing")
        block[mode] = row
    if len(blocks) < 2 or any(set(block) != {"on", "off"} for block in blocks.values()):
        raise ValueError("At least two complete on/off pairs are required")
    mismatched = [
        pair for pair, block in blocks.items() if block["on"]["turns"] != block["off"]["turns"]
    ]
    if mismatched:
        return {
            "status": "NOT_COMPARABLE",
            "mismatched_pairs": mismatched,
            "reason": "Input IDs, sampling parameters, output IDs or finish reasons differ",
        }
    on = [block["on"]["elapsed_seconds"] for block in blocks.values()]
    off = [block["off"]["elapsed_seconds"] for block in blocks.values()]
    differences = [a - b for a, b in zip(on, off, strict=True)]
    rng = random.Random(seed)
    boot = sorted(
        statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(10000)
    )
    callback = [v for block in blocks.values() for v in block["on"]["callback_seconds"]]
    finalizes = [block["on"]["finalize_seconds"] for block in blocks.values()]
    return {
        "status": "COMPARABLE",
        "pairs": len(blocks),
        "measured_requests": len(measured) * 2,
        "identical_work_within_pairs": True,
        "elapsed_seconds": {
            "on_median": statistics.median(on),
            "off_median": statistics.median(off),
            "on_mean": statistics.mean(on),
            "off_mean": statistics.mean(off),
        },
        "paired_delta_seconds": {
            "mean": statistics.mean(differences),
            "median": statistics.median(differences),
            "mean_bootstrap_95_percent_interval": [
                percentile(boot, 0.025),
                percentile(boot, 0.975),
            ],
        },
        "paired_mean_delta_percent_of_off_mean": 100
        * statistics.mean(differences)
        / statistics.mean(off),
        "callback_seconds": {
            "median": statistics.median(callback),
            "p95": percentile(callback, 0.95),
        },
        "finalize_seconds_median": statistics.median(finalizes),
        "uncertainty": "Paired bootstrap of mean differences, 10000 resamples. Small repeated "
        "workload; interval does not cover systematic noise or generalize to training.",
        "interpretation": "A negative wall-time difference alone does not establish a speedup. "
        "Callback timings include two timer reads per invocation; no timer subtraction.",
    }


async def trial(
    endpoint,
    output,
    tokenizer,
    adapter_class,
    token_space,
    *,
    label,
    mode,
    evidence_kind,
    max_tokens,
):
    import aiohttp
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    wires, callback_seconds = [], []
    context = None
    path = output / (label + ".trace.jsonl")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:

        async def forward(request):
            payload = await request.json()
            async with session.post(
                endpoint + "/generate", json=payload, allow_redirects=False
            ) as r:
                if r.status != 200:
                    raise RuntimeError(f"Engine HTTP {r.status}")
                chunks, size = [], 0
                async for chunk in r.content.iter_chunked(65536):
                    size += len(chunk)
                    if size > MAX_CASE_BYTES:
                        raise ValueError("Engine response exceeds size limit")
                    chunks.append(chunk)
                data = parse_object(b"".join(chunks))
            wires.append((payload, data))
            return web.json_response(data)

        proxy = web.Application()
        proxy.router.add_post("/generate", forward)
        async with TestServer(proxy) as server:
            adapter = adapter_class(tokenizer=tokenizer, sglang_url=str(server.make_url("")))
            async with TestClient(TestServer(adapter.app)) as client:
                writer = capture = None
                started = time.perf_counter()
                try:
                    if mode == "on":
                        writer = TraceRecorder(path, trace_id=label, evidence_kind=evidence_kind)
                        capture = SlimeDebugCapture(
                            writer, lambda *args: context, eos_token_ids=[tokenizer.eos_token_id]
                        )

                        def callback(*args):
                            begin = time.perf_counter()
                            capture(*args)
                            callback_seconds.append(time.perf_counter() - begin)

                        adapter.debug_callback = callback
                    sid = "public-benchmark"
                    adapter.open_session(
                        sid, sampling_defaults={"temperature": 0, "top_p": 1, "top_k": -1}
                    )
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "What is 2 + 2? Give a brief explanation."},
                    ]
                    for index, limit in enumerate(max_tokens):
                        context = TurnContext(
                            sid,
                            "main",
                            str(index + 1),
                            str(index) if index else None,
                            token_space,
                            CONTRACT,
                        )
                        async with client.post(
                            "/v1/chat/completions",
                            json={
                                "model": "local-pinned-qwen",
                                "messages": messages,
                                "max_tokens": limit,
                                "temperature": 0,
                                "metadata": {"session_id": sid},
                            },
                        ) as reply:
                            if reply.status != 200:
                                raise RuntimeError(f"Adapter HTTP {reply.status}")
                            data = await reply.json()
                        messages += [
                            data["choices"][0]["message"],
                            {"role": "user", "content": "Reply with one word: done."},
                        ]
                    await adapter.shutdown_session(sid)
                    finalize_seconds = 0
                    if capture is not None:
                        before_finalize = time.perf_counter()
                        capture.raise_if_failed(expected_turns=2)
                        finalize_seconds = time.perf_counter() - before_finalize
                finally:
                    if writer is not None:
                        writer.close()
                elapsed = time.perf_counter() - started
    if len(wires) != 2:
        raise RuntimeError("Expected exactly two engine observations")
    # Validation and evidence serialization are outside the measured interval, in both arms.
    turns = []
    for payload, data in wires:
        ids = response_ids(data, expected_prompt_tokens=len(payload["input_ids"]))
        turns.append(
            {
                "input_ids": payload["input_ids"],
                "output_ids": ids,
                "sampling_params": payload["sampling_params"],
                "finish_reason": data["meta_info"]["finish_reason"],
            }
        )
    if turns[0]["finish_reason"]["type"] != "stop":
        raise RuntimeError("First generation reached its budget; no completed first turn")
    record = {
        "mode": mode,
        "elapsed_seconds": elapsed,
        "callback_seconds": callback_seconds,
        "finalize_seconds": finalize_seconds,
        "turns": turns,
    }
    if capture is not None:
        report, _ = inspect_trace(path)
        if report["capture_completion"]["state"] != "complete":
            raise RuntimeError("Capture did not complete")
        records = [json.loads(line) for line in path.read_text().splitlines()]
        generations = [record for record in records if record["record_type"] == "generation"]
        for generation, wire in zip(generations, turns, strict=True):
            if (
                generation["input_ids"] != wire["input_ids"]
                or generation["output_ids"] != wire["output_ids"]
            ):
                raise RuntimeError("Callback and wire IDs differ")
        record.update(
            trace_file=path.name,
            trace_sha256=report["trace_sha256"],
            trace_status=report["status"],
            capture_complete=True,
        )
    return record


async def run(
    endpoint,
    output,
    tokenizer,
    adapter_class,
    token_space,
    *,
    pairs=20,
    warmup_pairs=2,
    seed=0,
    evidence_kind="observed_rollout",
    max_tokens=(512, 16),
):
    plan = order_plan(pairs, seed)
    if type(warmup_pairs) is not int or not 1 <= warmup_pairs <= 10:
        raise ValueError("warmup_pairs must be from 1 to 10")
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "benchmark_version": 1,
        "rolloutcheck_version": __version__,
        "pairs": pairs,
        "warmup_pairs": warmup_pairs,
        "seed": seed,
        "order": plan,
        "evidence_kind": evidence_kind,
        "max_new_tokens": list(max_tokens),
        "slime_revision": SLIME_REVISION,
        "timing_scope": "Two sequential adapter HTTP requests, session open/drain, and (on arm) "
        "trace creation, callback, completion and close. Server/client setup and "
        "post-run validation excluded. Both arms observe wire payloads in memory.",
        "cache_policy": "No cache flush; repeated public prompts with warmup. Server settings "
        "are operator-managed and must be recorded separately.",
        "identity": "Endpoint identity and runtime are not authenticated by this runner.",
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    save(output / "protocol.json", protocol)
    rows = []
    with (output / "trials.jsonl").open("x", encoding="utf-8") as stream:
        for phase, orders in (("warmup", [["off", "on"]] * warmup_pairs), ("measured", plan)):
            for pair, order in enumerate(orders):
                for mode in order:
                    label = f"{phase}-{pair:03d}-{mode}"
                    row = await trial(
                        endpoint,
                        output,
                        tokenizer,
                        adapter_class,
                        token_space,
                        label=label,
                        mode=mode,
                        evidence_kind=evidence_kind,
                        max_tokens=max_tokens,
                    )
                    row.update(phase=phase, pair=pair)
                    rows.append(row)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                print(f"{phase} pair {pair + 1}/{len(orders)} saved", flush=True)
    validate_schedule(rows, protocol)
    result = summarize(rows, seed=seed)
    save(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--sglang-url", type=loopback_url)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--warmup-pairs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--summarize", type=Path, help="Recompute statistics offline from a saved run"
    )
    args = parser.parse_args()
    if args.summarize:
        with (args.summarize / "protocol.json").open("rb") as stream:
            protocol = parse_object(stream.read(MAX_CASE_BYTES + 1))
        with (args.summarize / "trials.jsonl").open("rb") as stream:
            raw = stream.read(MAX_TRACE_BYTES + 1)
        if len(raw) > MAX_TRACE_BYTES or not raw.endswith(b"\n"):
            raise ValueError("Oversized or incomplete benchmark samples")
        rows = [parse_object(line) for line in raw.splitlines()]
        validate_schedule(rows, protocol)
        for row in rows:
            if row["mode"] != "on":
                continue
            name = f"{row['phase']}-{row['pair']:03d}-on.trace.jsonl"
            if row["trace_file"] != name:
                raise ValueError("Unexpected capture filename")
            with (args.summarize / name).open("rb") as stream:
                trace_raw = stream.read(MAX_TRACE_BYTES + 1)
            report, cases = inspect_trace_bytes(trace_raw)
            if (
                report["trace_sha256"] != row["trace_sha256"]
                or report["capture_completion"]["state"] != "complete"
                or report["status"] != row["trace_status"]
                or len(cases) != 1
            ):
                raise ValueError("Saved trace does not match its timing record")
            records = [parse_object(line) for line in trace_raw.splitlines()]
            generations = [r for r in records if r["record_type"] == "generation"]
            for generation, turn in zip(generations, row["turns"], strict=True):
                if (
                    generation["input_ids"] != turn["input_ids"]
                    or generation["output_ids"] != turn["output_ids"]
                ):
                    raise ValueError("Saved trace and benchmark token IDs differ")
        result = summarize(rows, seed=protocol["seed"])
    else:
        if not all((args.source, args.assets, args.sglang_url, args.output)):
            parser.error("Collect requires --source, --assets, --sglang-url and --output")
        verify_source(args.source)
        assets = verify_assets(args.assets)
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        sys.path.insert(0, str(args.source.resolve()))
        import transformers
        from slime.agent.adapters.openai import OpenAIAdapter

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.assets, local_files_only=True, trust_remote_code=False
        )
        result = asyncio.run(
            run(
                args.sglang_url,
                args.output,
                tokenizer,
                OpenAIAdapter,
                "sha256:" + assets["files"]["tokenizer.json"],
                pairs=args.pairs,
                warmup_pairs=args.warmup_pairs,
                seed=args.seed,
            )
        )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "COMPARABLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
