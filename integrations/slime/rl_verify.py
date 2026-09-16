"""Verify a bounded slime RL run from its actual debug dumps and weight receipts.

This validates the observed loop and Sample-to-trainer token handoff. It does not
establish convergence, policy quality, or an independent engine-wire capture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections import defaultdict
from pathlib import Path

from rl_reward import dataset, score

from rolloutcheck.sample_audit import audit_samples


def plain(value):
    """Convert tensors from local debug dumps into portable JSON values."""
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_round(rollout, train, *, rollout_id, samples_per_round, group_size=4):
    require(rollout["rollout_id"] == train["rollout_id"] == rollout_id, "Wrong rollout ID")
    generated, trained = rollout["samples"], train["samples"]
    require(len(generated) == len(trained) == samples_per_round, "Unexpected sample count")
    by_position = {row["rollout_position"]: row for row in trained}
    require(set(by_position) == set(range(len(generated))),
            "Missing or duplicated sample positions")
    require(train["raw_reward"] == [row["reward"] for row in generated],
            "Raw rewards changed at the trainer handoff")
    groups = defaultdict(list)
    for source in generated:
        require(type(source["group_index"]) is int, "Missing prompt group identity")
        groups[source["group_index"]].append(source["reward"])
    require(all(len(values) == group_size for values in groups.values()),
            "Unexpected prompt group size")
    normalized = {}
    for group, values in groups.items():
        mean = sum(values) / len(values)
        # Pinned slime uses PyTorch's sample standard deviation (correction=1).
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
        normalized[group] = (mean, std)
    audits, nonzero_advantages = [], 0
    for position, source in enumerate(generated):
        target = by_position[position]
        length = source["response_length"]
        tokens = source["tokens"]
        require(0 < length < len(tokens), "Empty or invalid prompt/response")
        require(target["tokens"] == tokens, "Trainer tokens differ from saved rollout")
        require(target["response_lengths"] == length, "Trainer response boundary differs")
        require(target["sample_index"] == source["index"], "Sample identity differs")
        require(not source["remove_sample"], "Unexpected excluded sample in this workload")
        require(source["weight_versions"] == [str(rollout_id + 1)],
                "Stale or missing engine weights")
        require(source["reward"] == score(source["response"], source["label"]), "Incorrect reward")
        require(target["loss_masks"] == [1] * length, "Unexpected mask in single-turn workload")
        for field in ("advantages", "returns", "log_probs", "rollout_log_probs"):
            values = target[field]
            require(len(values) == length, f"Wrong length for {field}")
            require(all(math.isfinite(v) for v in values), f"Nonfinite {field}")
        sampled_logprobs = source["rollout_log_probs"]
        require(len(sampled_logprobs) == length and all(map(math.isfinite, sampled_logprobs)),
                "Invalid sampled log probabilities")
        expected_logprobs = [struct.unpack("<f", struct.pack("<f", v))[0]
                            for v in sampled_logprobs]
        require(target["rollout_log_probs"] == expected_logprobs,
                "Sampled log probabilities changed during float32 handoff")
        mean, std = normalized[source["group_index"]]
        expected = (source["reward"] - mean) / (std + 1e-6)
        require(math.isclose(target["rewards"], expected, abs_tol=5e-6),
                "Incorrect normalized reward")
        for field in ("advantages", "returns"):
            require(all(math.isclose(v, expected, abs_tol=5e-6) for v in target[field]),
                    f"{field} differ from expected reference-free GRPO values")
        nonzero_advantages += sum(v != 0 for v in target["advantages"])
        audit = audit_samples(
            [{"turn_id": str(source["index"]),
              "input_ids": tokens[:-length], "output_ids": tokens[-length:]}],
            [{"tokens": target["tokens"], "response_length": length,
              "loss_mask": target["loss_masks"]}],
        )
        require(audit["context_status"] == "MATCHED", "Sample-to-trainer context mismatch")
        audits.append(audit)
    varying_groups = sum(len(set(values)) > 1 for values in groups.values())
    require(varying_groups > 0, "No within-prompt reward variation; no demonstrated GRPO signal")
    require(nonzero_advantages > 0, "No nonzero advantages")
    return {
        "rollout_id": rollout_id,
        "samples": len(generated),
        "engine_weight_version": str(rollout_id + 1),
        "reward_sum": sum(row["reward"] for row in generated),
        "groups_with_reward_variation": varying_groups,
        "nonzero_advantage_tokens": nonzero_advantages,
        "trainable_tokens": sum(a["trainable_tokens"] for a in audits),
        "sample_to_trainer_context_status": "MATCHED",
        "rollout_logprob_handoff_status": "MATCHED_FLOAT32",
    }


def changed_tensors(before, after):
    first, second = before["tensors"], after["tensors"]
    require(bool(first) and first.keys() == second.keys(), "Exported tensor names differ")
    changed = []
    for name in first:
        for key in ("shape", "dtype"):
            require(first[name][key] == second[name][key], f"Tensor {key} differs for {name}")
        if first[name]["sha256"] != second[name]["sha256"]:
            changed.append(name)
    require(bool(changed), "No exported tensor values changed")
    return changed


def verify_evaluation(before, after, *, rounds, training_prompts):
    require(before["rollout_id"] == 0 and after["rollout_id"] == rounds - 1,
            "Incorrect evaluation boundaries")
    expected_labels = {row["label"] for row in dataset(evaluation=True)}
    prompts, results = [], []
    for dump, version in ((before, 1), (after, rounds + 1)):
        rows = dump["samples"]
        require(len(rows) == len(expected_labels), "Missing evaluation samples")
        require({row["label"] for row in rows} == expected_labels,
                "Unexpected evaluation questions")
        current_prompts = set()
        for row in rows:
            length = row["response_length"]
            require(0 < length < len(row["tokens"]), "Invalid evaluation response")
            prompt = tuple(row["tokens"][:-length])
            require(prompt not in training_prompts, "Evaluation prompt leaked into training")
            current_prompts.add(prompt)
            require(row["weight_versions"] == [str(version)], "Wrong evaluation weight version")
            require(row["reward"] == score(row["response"], row["label"]),
                    "Incorrect evaluation reward")
        require(len(current_prompts) == len(rows), "Duplicated evaluation prompts")
        prompts.append(current_prompts)
        results.append({"engine_weight_version": str(version), "samples": len(rows),
                        "reward_sum": sum(row["reward"] for row in rows)})
    require(prompts[0] == prompts[1], "Evaluation prompts changed between checkpoints")
    return {"before": results[0], "after": results[1],
            "scope": "Four separate arithmetic questions; descriptive smoke evaluation only."}


def verify_optimizer_recovery(records, *, rounds, steps_per_round=2):
    require(len(records) == rounds * steps_per_round, "Missing optimizer observations")
    counters = []
    for position, row in enumerate(records):
        require((row["rollout_id"], row["step_id"]) == divmod(position, steps_per_round),
                "Missing or reordered optimizer steps")
        require(row["when"] == "before optimizer step", "Ambiguous optimizer observation")
        require(len(row["optimizers"]) == 1, "Expected one native optimizer")
        optimizer = row["optimizers"][0]
        require(optimizer["class"] ==
                "transformer_engine.pytorch.optimizers.fused_adam.FusedAdam",
                "Unexpected optimizer implementation")
        steps = optimizer["group_steps"]
        require(bool(steps) and all(step == position for step in steps),
                "Optimizer step counter reset or skipped across checkpoint recovery")
        if position:
            counts = optimizer["state_elements_by_dtype"]
            require(counts.get("exp_avg:torch.bfloat16") == 596049920 and
                    counts.get("exp_avg_sq:torch.bfloat16") == 596049920,
                    "Expected full-model BF16 Adam moment state")
        counters.append(steps)
    return {"status": "CONTINUOUS_STEP_COUNTERS", "before_step_counters": counters,
            "moment_dtype": "torch.bfloat16", "moment_elements_each": 596049920,
            "scope": "Observed counters and full-model moment storage, not an independent "
                     "bitwise comparison of an uninterrupted training run."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples-per-round", type=int, default=16)
    parser.add_argument("--portable", action="store_true",
                        help="Read exported JSON instead of trusted local PyTorch dumps")
    parser.add_argument("--output", type=Path, help="Directory for the derived verification report")
    args = parser.parse_args()
    require(args.rounds >= 2, "At least two rounds are required for online weight reuse")
    run = args.run_dir
    process = json.loads((run / "process.json").read_text())
    require(process["returncode"] == process["state"]["ExitCode"] == 0,
            "Training process did not finish successfully")
    require(process["state"]["OOMKilled"] is False, "Training container was OOM-killed")
    config = json.loads((run / "resolved-config.json").read_text())
    expected_config = {
        "advantage_estimator": "grpo", "rewards_normalization": True,
        "grpo_std_normalization": True, "normalize_advantages": False,
        "kl_coef": 0, "kl_loss_coef": 0, "use_kl_loss": False,
        "num_rollout": args.rounds, "start_rollout_id": 0,
        "eval_interval": args.rounds, "n_samples_per_eval_prompt": 1, "eval_temperature": 0,
    }
    require(all(config[key] == value for key, value in expected_config.items()),
            "Resolved configuration is outside this verification profile")
    checkpoint = run / "checkpoints" / "latest_checkpointed_iteration.txt"
    # slime saves the zero-based rollout ID, not the optimizer-step count.
    require(checkpoint.read_text().strip() == str(args.rounds - 1), "Final checkpoint is missing")

    if not args.portable:
        import torch

    def load(path):
        if args.portable:
            return json.loads(path.with_suffix(".json").read_text())
        return plain(torch.load(path, map_location="cpu", weights_only=True))

    gradient_records = (json.loads((run / "gradients.json").read_text())
                        if args.portable else {})

    summaries, training_prompts = [], set()
    for number in range(args.rounds):
        rollout = load(run / f"rollout-{number}.pt")
        training_prompts.update(tuple(row["tokens"][:-row["response_length"]])
                                for row in rollout["samples"])
        train = load(run / f"train-{number}.pt")
        result = verify_round(rollout, train, rollout_id=number,
                              samples_per_round=args.samples_per_round,
                              group_size=config["n_samples_per_prompt"])
        gradient_paths = sorted(run.glob(f"grad-{number}-*.pt"))
        expected_steps = sum(len(shard["global_batch_sizes"]) for shard in train["dp_shards"])
        require(len(train["dp_shards"]) == 1,
                "This verification profile requires one training rank")
        gradients = (gradient_records[str(number)] if args.portable else
                     [float(load(path)) for path in gradient_paths])
        require(len(gradients) == expected_steps > 0, "Missing training steps")
        gradient_records[str(number)] = gradients
        require(all(math.isfinite(v) and v >= 0 for v in gradients), "Invalid gradient norm")
        require(any(v > 0 for v in gradients), "Only zero gradients observed")
        receipts = [json.loads((run / "weights" / f"weight_v{v:06d}.receipt.json").read_text())
                    for v in (number + 1, number + 2)]
        changed = changed_tensors(*receipts)
        result.update(gradient_norms=gradients, changed_tensor_count=len(changed))
        summaries.append(result)
        # Portable source values remain distinct from the derived verdict.
        if not args.portable:
            (run / f"rollout-{number}.json").write_text(json.dumps(rollout, indent=2) + "\n")
            (run / f"train-{number}.json").write_text(json.dumps(train, indent=2) + "\n")
    if args.portable:
        evaluations = [json.loads((run / f"evaluation-{name}.json").read_text())
                       for name in ("before", "after")]
    else:
        evaluations = [load(run / f"rollout-eval_{number}.pt")
                       for number in (0, args.rounds - 1)]
    evaluation_result = verify_evaluation(*evaluations, rounds=args.rounds,
                                          training_prompts=training_prompts)
    if not args.portable:
        for name, dump in zip(("before", "after"), evaluations, strict=True):
            (run / f"evaluation-{name}.json").write_text(json.dumps(dump, indent=2) + "\n")
        (run / "gradients.json").write_text(json.dumps(gradient_records, indent=2) + "\n")
    recovery = verify_optimizer_recovery(
        [json.loads(line) for line in (run / "optimizer-state.jsonl").read_text().splitlines()],
        rounds=args.rounds,
    )
    summary = {
        "status": "VERIFIED_BOUNDED_RL_LOOP",
        "optimizer_recovery": recovery,
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rounds": summaries,
        "evaluation": evaluation_result,
        "scope": "Observed native slime loop, finite/nonzero RL signal, exported parameter "
                 "changes, engine weight versions and Sample-to-trainer token/mask handoff. "
                 "Not convergence, independent wire capture, or general production readiness.",
    }
    output = args.output or run
    output.mkdir(parents=True, exist_ok=True)
    (output / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
