import copy

import pytest

from rolloutcheck.case import CaseError
from rolloutcheck.training_batch import audit_batch, pad_features, training_features


def example():
    turns = [{"turn_id": "1", "input_ids": [1, 2], "output_ids": [3, 9]}]
    samples = [{"tokens": [1, 2, 3, 9], "response_length": 2, "loss_mask": [1, 1]}]
    return turns, samples


def test_labels_preserve_mask_and_eos_at_padding_boundary():
    features, audit = training_features(*example())
    features.append({"input_ids": [1, 9], "labels": [-100, 9]})
    batch = pad_features(features, pad_token_id=9)
    assert batch["labels"] == [[-100, -100, 3, 9], [-100, 9, -100, -100]]
    assert batch["attention_mask"] == [[1, 1, 1, 1], [1, 1, 0, 0]]
    assert audit["trainable_tokens"] == 2
    assert audit_batch(features, batch, pad_token_id=9)["status"] == "MATCHED"


@pytest.mark.parametrize("field", ["labels", "attention_mask", "input_ids"])
def test_changed_training_boundary_is_detected(field):
    features, _ = training_features(*example())
    batch = pad_features(features, pad_token_id=9)
    batch[field][0][1] = 7
    report = audit_batch(features, batch, pad_token_id=9)
    assert report["status"] == "MISMATCH"
    assert report["mismatched_fields"] == [field]


def test_reject_unmatched_duplicate_and_truncated_supervision():
    turns, samples = example()
    bad = copy.deepcopy(samples)
    bad[0]["tokens"][0] = 8
    for value in (bad, samples * 2):
        with pytest.raises(CaseError, match="unique, matched"):
            training_features(turns, value)
    with pytest.raises(CaseError, match="max_length"):
        training_features(turns, samples, max_length=3)


def test_intentional_dropping_is_reported_not_repaired():
    turns, samples = example()
    samples[0]["loss_mask"] = [0, 1]
    features, audit = training_features(turns, samples)
    assert features[0]["labels"] == [-100, -100, -100, 9]
    assert audit["unaccounted_generated_tokens"] == 1
