"""Repeat actual Trainer jobs with policy, batch-guard and checkpoint controls.

Each subprocess starts from the same local weights/seed. Timing includes training,
checkpoint I/O and constant observation hooks, but excludes loading/preflight.
This is a bounded fixed-rollout training experiment, not convergence evaluation.
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from train_handoff import digest, save

from rolloutcheck.case import load_case
from rolloutcheck.training_batch import audit_batch, pad_features, training_features


def run(args):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, DataCollatorForLanguageModeling

    if args.pairs < 1:
        raise ValueError("At least one measured pair is required")
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    runner = Path(__file__).with_name("train_handoff.py")
    profiles = [
        ("short", args.native / "short/results/fork/handoff.json", 2, 1),
        ("context-80", args.native / "long/results/context-80-fork/handoff.json", 1, 2),
        ("context-160", args.native / "long/results/context-160-fork/handoff.json", 2, 1),
    ]
    # A real library collator misconfiguration, not a manually edited fake label array.
    snapshot, _ = load_case(profiles[0][1])
    features, _ = training_features(snapshot["turns"], snapshot["samples"])
    tokenizer = AutoTokenizer.from_pretrained(args.assets, local_files_only=True)
    rows = features[:1]
    wrong = DataCollatorForLanguageModeling(tokenizer, mlm=False)(rows)
    wrong = {name: value.tolist() for name, value in wrong.items()}
    correct = pad_features(rows, pad_token_id=tokenizer.pad_token_id)
    before = audit_batch(rows, wrong, pad_token_id=tokenizer.pad_token_id)
    after = audit_batch(rows, correct, pad_token_id=tokenizer.pad_token_id)
    assert before["status"] == "MISMATCH" and after["status"] == "MATCHED"
    save(
        args.output / "collator-control.json",
        {
            "before": before,
            "after": after,
            "unexpected_supervised_positions_before": sum(
                expected == -100 and actual != -100
                for e, a in zip(correct["labels"], wrong["labels"], strict=True)
                for expected, actual in zip(e, a, strict=True)
            ),
            "removed_supervised_positions_before": sum(
                expected != -100 and actual == -100
                for e, a in zip(correct["labels"], wrong["labels"], strict=True)
                for expected, actual in zip(e, a, strict=True)
            ),
            "before_batch": wrong,
            "after_batch": correct,
            "scope": "Controlled HF DataCollatorForLanguageModeling(mlm=False) misuse on actual "
            "saved rollout sample. Guard rejects before optimizer; not an upstream library bug.",
        },
    )

    def job(name, handoff, batch, accumulation, *, audit=True, extra=()):
        command = [
            sys.executable,
            str(runner),
            "--assets",
            str(args.assets),
            "--handoff",
            str(handoff),
            "--output",
            str(args.output / name),
            "--device",
            args.device,
            "--steps",
            "4",
            "--batch-size",
            str(batch),
            "--accumulation",
            str(accumulation),
            "--max-length",
            "3072",
            "--audit" if audit else "--no-audit",
            *extra,
        ]
        with (args.output / f"{name}.log").open("x") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
        result, _ = load_case(args.output / name / "result.json")
        print(name, result["state"], result["elapsed_seconds"], flush=True)
        return result

    def tensors(name):
        return load_file(str(args.output / name / "checkpoint-4/adapter_model.safetensors"))

    results = []
    for profile, handoff, batch, accumulation in profiles:
        pairs = []
        for pair in range(args.pairs + 1):  # Pair zero is warmup and excluded from timing summary.
            names = {
                enabled: f"{profile}-pair-{pair}-{'on' if enabled else 'off'}"
                for enabled in (True, False)
            }
            runs = {}
            for enabled in (False, True) if pair % 2 == 0 else (True, False):
                runs[enabled] = job(names[enabled], handoff, batch, accumulation, audit=enabled)
            left, right = tensors(names[True]), tensors(names[False])
            same_parameters = left.keys() == right.keys() and all(
                torch.equal(left[key], right[key]) for key in left
            )
            same_backward = (
                runs[True]["consumed_microbatches"] == runs[False]["consumed_microbatches"]
            )
            if not same_parameters or not same_backward:
                raise RuntimeError("Guard on/off changed training; timing comparison rejected")
            pairs.append(
                {
                    "pair": pair,
                    "warmup": pair == 0,
                    "on_run": names[True],
                    "off_run": names[False],
                    "on_seconds": runs[True]["elapsed_seconds"],
                    "off_seconds": runs[False]["elapsed_seconds"],
                    "delta_seconds": runs[True]["elapsed_seconds"] - runs[False]["elapsed_seconds"],
                    "identical_backward_records": same_backward,
                    "identical_parameters": same_parameters,
                }
            )
        delta = [pair["delta_seconds"] for pair in pairs if not pair["warmup"]]
        results.append(
            {
                "profile": profile,
                "pairs": pairs,
                "measured_pairs": len(delta),
                "median_delta_seconds": statistics.median(delta),
                "delta_range_seconds": [min(delta), max(delta)],
                "conclusion": "Descriptive timings only; too few pairs for a stable "
                "end-to-end overhead or speedup claim.",
            }
        )

    default = job("default-policy", args.native / "short/results/default/handoff.json", 2, 1)
    stopped = job("stopped", profiles[0][1], 2, 1, extra=("--stop-after", "2"))
    resumed = job(
        "resumed",
        profiles[0][1],
        2,
        1,
        extra=("--resume-from", str(args.output / "stopped/checkpoint-2")),
    )
    continuous, restored = tensors("short-pair-0-on"), tensors("resumed")
    identical = all(torch.equal(continuous[k], restored[k]) for k in continuous)
    if not identical:
        raise RuntimeError("Resumed weights differ from continuous run")
    summary = {
        "experiment_version": 1,
        "device": args.device,
        "profiles": results,
        "resume": {
            "stopped_at": stopped["global_step"],
            "resumed_from": resumed["resume_from_step"],
            "finished_at": resumed["global_step"],
            "identical_to_continuous": identical,
        },
        "default_policy_supervised_tokens_per_microbatch": [
            row["supervised_tokens"] for row in default["consumed_microbatches"]
        ],
        "source_sha256": {
            "verify_training_job.py": digest(Path(__file__)),
            "train_handoff.py": digest(runner),
        },
        "scope": "Actual native SGLang saved rollouts -> slime Samples -> HF Trainer/PEFT. "
        "Fixed-rollout masked supervised training, not slime/Megatron RL. Batch guard timings "
        "do not measure total rollout capture overhead. No training-quality or adoption claim.",
    }
    save(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--native", type=Path, required=True, help="Verified extracted native archive"
    )
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
    parser.add_argument("--pairs", type=int, default=3)
    run(parser.parse_args())
