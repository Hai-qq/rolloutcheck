"""Synthetic negatives for the verifier; these are not RL execution evidence."""

import copy
import math
import struct

import pytest
from rl_reward import dataset
from rl_verify import changed_tensors, verify_evaluation, verify_optimizer_recovery, verify_round


def dumps():
    source, target = [], []
    for index, (answer, reward) in enumerate([("3", 1.0), ("4", 0.0)]):
        tokens = [10, 20, 30 + index]
        advantage = (reward - 0.5) / (math.sqrt(0.5) + 1e-6)
        source.append({
            "index": index, "group_index": 0, "tokens": tokens,
            "response_length": 1, "response": answer, "label": "3",
            "reward": reward, "remove_sample": False, "weight_versions": ["1"],
            "rollout_log_probs": [-0.3],
        })
        target.append({
            "rollout_position": index, "sample_index": index, "tokens": tokens.copy(),
            "response_lengths": 1, "loss_masks": [1], "advantages": [advantage],
            "returns": [advantage], "rewards": advantage,
            "log_probs": [-0.3],
            "rollout_log_probs": [struct.unpack("<f", struct.pack("<f", -0.3))[0]],
        })
    return {"rollout_id": 0, "samples": source}, {
        "rollout_id": 0, "samples": target, "raw_reward": [1.0, 0.0],
    }


def test_observed_round_fields_and_reordered_training_samples():
    rollout, train = dumps()
    train["samples"].reverse()
    result = verify_round(rollout, train, rollout_id=0, samples_per_round=2, group_size=2)
    assert result["groups_with_reward_variation"] == 1
    assert result["trainable_tokens"] == 2


@pytest.mark.parametrize("damage", [
    "tokens", "mask", "version", "reward", "nan", "position", "advantage", "logprob",
])
def test_rejects_invalid_training_evidence(damage):
    rollout, train = dumps()
    if damage == "tokens":
        train["samples"][0]["tokens"][0] += 1
    elif damage == "mask":
        train["samples"][0]["loss_masks"] = [0]
    elif damage == "version":
        rollout["samples"][0]["weight_versions"] = ["0"]
    elif damage == "reward":
        rollout["samples"][0]["reward"] = 0.0
    elif damage == "nan":
        train["samples"][0]["advantages"] = [float("nan")]
    elif damage == "advantage":
        train["samples"][0]["advantages"] = [-0.7]
    elif damage == "logprob":
        train["samples"][0]["rollout_log_probs"] = [-0.4]
    else:
        train["samples"][0]["rollout_position"] = 1
    with pytest.raises(ValueError):
        verify_round(rollout, train, rollout_id=0, samples_per_round=2, group_size=2)


def test_file_metadata_changes_cannot_substitute_for_changed_tensor_values():
    before = {"files": [{"sha256": "file-a"}], "tensors": {
        "weight": {"shape": [1], "dtype": "torch.bfloat16", "sha256": "tensor-a"},
    }}
    after = copy.deepcopy(before)
    after["files"][0]["sha256"] = "file-b"
    with pytest.raises(ValueError, match="No exported tensor"):
        changed_tensors(before, after)
    after["tensors"]["weight"]["sha256"] = "tensor-b"
    assert changed_tensors(before, after) == ["weight"]


def test_final_evaluation_requires_new_weights_and_disjoint_prompts():
    rows = [{"label": row["label"], "response": row["label"], "reward": 1.0,
             "tokens": [10, index, 30], "response_length": 1, "weight_versions": ["1"]}
            for index, row in enumerate(dataset(evaluation=True))]
    before = {"rollout_id": 0, "samples": rows}
    after = copy.deepcopy(before)
    after["rollout_id"] = 2
    for row in after["samples"]:
        row["weight_versions"] = ["4"]
    result = verify_evaluation(before, after, rounds=3, training_prompts={(99, 99)})
    assert result["after"]["reward_sum"] == 4
    with pytest.raises(ValueError, match="leaked"):
        verify_evaluation(before, after, rounds=3, training_prompts={(10, 0)})
    after["samples"][0]["weight_versions"] = ["3"]
    with pytest.raises(ValueError, match="weight version"):
        verify_evaluation(before, after, rounds=3, training_prompts=set())


def test_native_optimizer_recovery_rejects_reset_and_ignored_precision():
    records = [{"rollout_id": i // 2, "step_id": i % 2,
                "when": "before optimizer step", "optimizers": [{
                    "class": "transformer_engine.pytorch.optimizers.fused_adam.FusedAdam",
                    "group_steps": [i, i], "state_elements_by_dtype": {
                        "exp_avg:torch.bfloat16": 596049920,
                        "exp_avg_sq:torch.bfloat16": 596049920,
                    },
                }]} for i in range(6)]
    assert verify_optimizer_recovery(records, rounds=3)["status"] == "CONTINUOUS_STEP_COUNTERS"
    bad = copy.deepcopy(records)
    bad[2]["optimizers"][0]["group_steps"] = [0, 0]
    with pytest.raises(ValueError, match="reset"):
        verify_optimizer_recovery(bad, rounds=3)
    del records[2]["optimizers"][0]["state_elements_by_dtype"]["exp_avg:torch.bfloat16"]
    with pytest.raises(ValueError, match="moment state"):
        verify_optimizer_recovery(records, rounds=3)
