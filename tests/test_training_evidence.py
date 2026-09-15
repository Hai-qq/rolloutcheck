"""Offline consistency checks of public observed training receipts, not model replay."""

import hashlib
import json
from pathlib import Path

from rolloutcheck.evidence import verify_evidence
from rolloutcheck.sample_audit import audit_samples
from rolloutcheck.training_batch import audit_batch, training_features

ROOT = Path(__file__).resolve().parents[1] / "cases/observed"


def read(path):
    return json.loads(path.read_text())


def test_native_policy_evidence_has_identical_work_and_recomputed_audits():
    root = ROOT / "slime-training-native"
    for profile, prefix, total in (
        ("short", "", 161),
        ("long", "context-80-", 326),
        ("long", "context-160-", 282),
    ):
        wires = []
        for policy in ("default", "fork"):
            folder = root / profile / "results" / (prefix + policy)
            data = read(folder / "handoff.json")
            audit = audit_samples(data["turns"], data["samples"])
            assert audit == read(folder / "audit.json")
            assert audit["generated_tokens"] == total
            assert audit["trainable_tokens"] == (16 if policy == "default" else total)
            assert audit["context_status"] == "MATCHED"
            history = verify_evidence(folder / "evidence")
            assert (
                history["status"] == "FAIL" and history["capture_completion"]["state"] == "complete"
            )
            wire = read(folder / "wire.json")
            trace = [json.loads(line) for line in (folder / "trace.jsonl").read_text().splitlines()]
            for turn, raw, record in zip(data["turns"], wire, trace[:2], strict=True):
                for key in ("input_ids", "output_ids"):
                    assert turn[key] == raw[key] == record[key]
            wires.append(wire)
        assert wires[0] == wires[1]
    for profile in ("short", "long"):
        runtime = read(root / profile / "runtime.json")
        name = "verify_training_handoff.py" if profile == "short" else "verify_training_workload.py"
        assert (
            hashlib.sha256((root / "provenance" / profile / name).read_bytes()).hexdigest()
            == (runtime["runner_sha256"])
        )


def test_training_receipts_bind_input_features_and_consumption():
    native, training = ROOT / "slime-training-native", ROOT / "trainer-native-mps"
    handoffs = {}
    for path in native.rglob("handoff.json"):
        handoffs[hashlib.sha256(path.read_bytes()).hexdigest()] = read(path)
    count = actual_steps = 0
    for path in training.glob("*/result.json"):
        result = read(path)
        contract = result["contract"]
        data = handoffs[contract["handoff_sha256"]]
        features, audit = training_features(data["turns"], data["samples"], max_length=3072)
        assert audit == result["sample_audit"]
        assert features == read(training / "features" / (contract["handoff_sha256"] + ".json"))
        assert result["changed_parameter_tensors"] > 0
        assert result["trainable_parameters"] == 1146880
        steps = result["global_step"] - result["resume_from_step"]
        assert (
            len(result["consumed_microbatches"]) == steps * contract["gradient_accumulation_steps"]
        )
        for step in range(result["resume_from_step"] + 1, result["global_step"] + 1):
            used = sum(
                row["supervised_tokens"]
                for row in result["consumed_microbatches"]
                if row["optimizer_step"] == step
            )
            assert used == audit["trainable_tokens"]
        count += 1
        actual_steps += steps
    assert count == 27 and actual_steps == 104
    summary = read(training / "summary.json")
    assert summary["resume"]["identical_to_continuous"]
    for profile in summary["profiles"]:
        assert profile["measured_pairs"] == 3
        for pair in profile["pairs"]:
            left = read(training / pair["on_run"] / "result.json")
            right = read(training / pair["off_run"] / "result.json")
            assert left["consumed_microbatches"] == right["consumed_microbatches"]
            assert left["final_adapter_sha256"] == right["final_adapter_sha256"]
            assert pair["identical_backward_records"] and pair["identical_parameters"]
    control = read(training / "collator-control.json")
    short = read(native / "short/results/fork/handoff.json")
    features, _ = training_features(short["turns"], short["samples"])
    # Qwen's verified config uses pad ID 151643.
    for which in ("before", "after"):
        assert (
            audit_batch(features[:1], control[which + "_batch"], pad_token_id=151643)
            == (control[which])
        )
    assert control["unexpected_supervised_positions_before"] == 32
