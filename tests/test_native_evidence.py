"""Recheck committed native evidence offline; this test does not execute SGLang."""

import json
from pathlib import Path

from rolloutcheck.history import inspect_case
from rolloutcheck.sglang_response import response_ids
from rolloutcheck.trace import inspect_trace


def test_native_wire_capture_and_saved_reports_agree():
    root = Path(__file__).resolve().parents[1]
    directory = root / "cases/observed/slime-sglang-cuda"
    summary = json.loads((directory / "summary.json").read_text())
    wires = [json.loads(p.read_text()) for p in sorted(directory.glob("wire-*.json"))]
    assert summary["requests"] == len(wires) == 6
    for mode, status in [("adapter", "FAIL"), ("control", "PASS")]:
        report, cases = inspect_trace(directory / f"{mode}.trace.jsonl")
        assert report == json.loads((directory / f"{mode}.report.json").read_text())
        assert report["status"] == status and report["transitions"] == 1
        if mode == "adapter":
            assert cases == [json.loads((directory / "adapter.case.json").read_text())]
        records = [
            json.loads(line)
            for line in (directory / f"{mode}.trace.jsonl").read_text().splitlines()
        ]
        relevant = wires[:2] if mode == "adapter" else [wires[0], wires[2]]
        for record, wire in zip(records, relevant, strict=True):
            assert record["evidence_kind"] == "observed_rollout"
            assert record["input_ids"] == wire["request"]["input_ids"]
            assert record["output_ids"] == response_ids(
                wire["response"], expected_prompt_tokens=len(record["input_ids"])
            )
    for probe, wire in zip(summary["stop_probes"], wires[3:], strict=True):
        ids = response_ids(
            wire["response"], expected_prompt_tokens=len(wire["request"]["input_ids"])
        )
        assert probe["returned_tokens"] == len(ids)
        assert probe["last_id"] == ids[-1]
        assert probe["text"] == wire["response"]["text"]
        assert probe["finish_reason"] == wire["response"]["meta_info"]["finish_reason"]
    # The trimmed text is empty while the actual stop-token ID is still present.
    assert wires[4]["response"]["text"] == ""
    assert wires[5]["response"]["text"] == "<think>"
    assert summary["stop_probes"][1]["last_id"] == 151667
    mutation = json.loads((root / "cases/synthetic/slime-sglang-mutation.case.json").read_text())
    assert mutation["evidence"]["kind"] == "synthetic"
    assert inspect_case(mutation)["status"] == "FAIL"
