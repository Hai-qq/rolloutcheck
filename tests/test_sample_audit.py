import pytest

from rolloutcheck.case import CaseError
from rolloutcheck.sample_audit import audit_samples


def test_clean_and_masked_rewrite_have_different_retention_not_context_corruption():
    turns = [
        {"turn_id": "1", "input_ids": [1], "output_ids": [2, 3]},
        {"turn_id": "2", "input_ids": [1, 8, 4], "output_ids": [5]},
    ]
    # Rewritten history is context only; it must not be blamed as trained corruption.
    realigned = [{"tokens": [1, 8, 4, 5], "response_length": 3, "loss_mask": [0, 0, 1]}]
    report = audit_samples(turns, realigned)
    assert report["context_status"] == "MATCHED" and report["unaccounted_generated_tokens"] == 2
    split = [{"tokens": [1, 2, 3], "response_length": 2, "loss_mask": [1, 1]}, *realigned]
    report = audit_samples(turns, split)
    assert report["context_status"] == "MATCHED" and report["unaccounted_generated_tokens"] == 0
    realigned[0]["loss_mask"] = [1, 0, 1]
    assert audit_samples(turns, realigned)["unmatched_trainable_tokens"] == 1


def test_matching_token_with_wrong_prefix_is_unmatched():
    turns = [{"turn_id": "t", "input_ids": [1, 2], "output_ids": [3]}]
    sample = {"tokens": [1, 9, 3], "response_length": 1, "loss_mask": [1]}
    assert audit_samples(turns, [sample])["context_status"] == "UNMATCHED"


def test_identical_sources_are_ambiguous_and_duplicate_training_is_visible():
    turn = {"turn_id": "one", "input_ids": [1], "output_ids": [2]}
    sample = {"tokens": [1, 2], "response_length": 1, "loss_mask": [1]}
    assert (
        audit_samples([turn, {**turn, "turn_id": "two"}], [sample])["ambiguous_trainable_tokens"]
        == 1
    )
    assert audit_samples([turn], [sample, sample])["duplicate_training_occurrences"] == 1
    assert audit_samples([turn], [])["context_status"] == "NO_TRAINABLE_TOKENS"


@pytest.mark.parametrize("length,mask", [(2, [1]), (-1, []), (1, [True]), (1, None), (1, [2])])
def test_malformed_masks_are_not_assumed_trainable(length, mask):
    with pytest.raises(CaseError):
        audit_samples([], [{"tokens": [1, 2], "response_length": length, "loss_mask": mask}])


def test_budget_and_turn_identity_are_checked(monkeypatch):
    import rolloutcheck.sample_audit as module

    turn = {"turn_id": "t", "input_ids": [1], "output_ids": [2]}
    with pytest.raises(CaseError):
        audit_samples([turn, turn], [])
    monkeypatch.setattr(module, "MAX_TOTAL_TOKENS", 1)
    with pytest.raises(CaseError, match="input limit"):
        audit_samples([turn], [])


def test_observed_mps_handoff_recomputes_and_agrees_with_captured_ids():
    import json
    from pathlib import Path

    from rolloutcheck.evidence import verify_evidence

    root = Path(__file__).resolve().parents[1] / "cases/observed/slime-training-handoff-mps"
    summary = json.loads((root / "summary.json").read_text())
    runtime = json.loads((root / "runtime.json").read_text())
    assert summary["same_generation_ids_across_policies"]
    assert runtime["device"] == "mps" and runtime["requests"] == 4
    assert runtime["optimizer_steps"] == 0
    wires = []
    for number, policy in enumerate(("default", "fork")):
        folder = root / policy
        data = json.loads((folder / "handoff.json").read_text())
        report = audit_samples(data["turns"], data["samples"])
        assert report == json.loads((folder / "audit.json").read_text())
        assert report == summary["scenarios"][number]["audit"]
        assert report["context_status"] == "MATCHED" and report["generated_tokens"] == 192
        assert report["trainable_tokens"] == (16 if policy == "default" else 192)
        assert report["unaccounted_generated_tokens"] == (176 if policy == "default" else 0)
        assert report["duplicate_training_occurrences"] == 0
        assert sum(s["tokens"] for s in report["samples"]) == (83 if policy == "default" else 291)
        trace_report = verify_evidence(folder / "evidence")
        assert trace_report["status"] == "FAIL"
        assert trace_report["capture_completion"]["state"] == "complete"
        trace = [json.loads(line) for line in (folder / "trace.jsonl").read_text().splitlines()]
        wire = json.loads((folder / "wire.json").read_text())
        assert len(data["turns"]) == len(wire) == 2
        for turn, observed, record in zip(data["turns"], wire, trace[:2], strict=True):
            assert turn["turn_id"] == record["turn_id"]
            for key in ("input_ids", "output_ids"):
                assert turn[key] == observed[key] == record[key]
        wires.append(wire)
    assert wires[0] == wires[1]
