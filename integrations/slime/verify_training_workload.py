"""Native long-context retention controls, using actual pinned adapter Samples.

Fixed public filler text is an explicit stress workload, not independent user data.
Measured token lengths, not profile names, define the tested envelope.
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
from verify_training_handoff import scenario


async def run(args):
    lock = verify_source(args.source)
    assets = verify_assets(args.assets)
    sys.path.insert(0, str(args.source.resolve()))
    from slime.agent.adapters.openai import OpenAIAdapter
    from slime.utils.types import Sample
    from transformers import AutoTokenizer

    if Path(sys.modules[OpenAIAdapter.__module__].__file__).resolve() != (
        args.source.resolve() / "slime/agent/adapters/openai.py"
    ):
        raise RuntimeError("Imported adapter differs from verified source")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    tokenizer = AutoTokenizer.from_pretrained(args.assets, local_files_only=True)
    token_space = "sha256:" + assets["files"]["tokenizer.json"]
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    results = []
    comparisons = []
    for repetitions in (80, 160):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": (
                    "Reference notes (context only):\n"
                    + "The blue notebook is on the wooden desk.\n" * repetitions
                    + "\nWhat is 2 + 2? Give a brief explanation."
                ),
            },
        ]
        for name, threshold in (("default", None), ("fork", 0)):
            result = await scenario(
                args.output / f"context-{repetitions}-{name}",
                OpenAIAdapter,
                Sample,
                tokenizer,
                args.sglang_url,
                token_space,
                threshold=threshold,
                initial_messages=messages,
            )
            result["context_repetitions"] = repetitions
            results.append(result)
        left = json.loads((args.output / f"context-{repetitions}-default/wire.json").read_text())
        right = json.loads((args.output / f"context-{repetitions}-fork/wire.json").read_text())
        comparisons.append(
            {
                "context_repetitions": repetitions,
                "same_generation_ids_across_policies": left == right,
                "retention_policy_comparison_eligible": left == right,
                "reason": "Identical work"
                if left == right
                else (
                    "Different generated work: do not attribute differences "
                    "solely to retention policy"
                ),
            }
        )
    save(
        args.output / "summary.json",
        {
            "source_revision": lock["revision"],
            "scenarios": results,
            "policy_comparisons": comparisons,
            "source_sha256": {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ("verify_training_workload.py", "verify_training_handoff.py")
            },
            "scope": "Actual sequential native inference; public repeated-context stress workload. "
            "No concurrent engine requests, optimizer, throughput or independent-use claim.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sglang-url", type=loopback_url, required=True)
    asyncio.run(run(parser.parse_args()))
