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


def test_independent_http_client_native_evidence():
    from rolloutcheck.evidence import verify_evidence

    root = Path(__file__).resolve().parents[1]
    directory = root / "cases/observed/slime-http-cuda"
    report, cases = inspect_trace(directory / "trace.jsonl")
    assert report == json.loads((directory / "report.json").read_text())
    verified = verify_evidence(directory / "evidence")
    assert verified.pop("verification")["integrity"] == "verified"
    assert verified == report
    assert (directory / "trace.jsonl").read_bytes() == (
        directory / "evidence/trace.jsonl"
    ).read_bytes()
    assert report["status"] == "FAIL" and report["counts"] == {"FAIL": 2}
    assert report["capture_completion"]["state"] == "complete"
    receipt = json.loads((directory / "receipt.json").read_text())
    assert receipt["received"] == receipt["succeeded"] == receipt["captured"] == 4
    assert receipt["failed"] == 0 and receipt["finalized"]
    runtime = json.loads((directory / "runtime.json").read_text())
    assert runtime["client_returncode"] == 0 and runtime["server_returncode"] == 1
    records = [json.loads(line) for line in (directory / "trace.jsonl").read_text().splitlines()]
    generations = [record for record in records if record["record_type"] == "generation"]
    wires = [json.loads(line) for line in (directory / "wire.jsonl").read_text().splitlines()]
    # Only within each public session is order a valid join: the demo waits for
    # the parent response. Never zip across globally interleaved completions.
    for session in ("public-session-1", "public-session-2"):
        turns = [r for r in generations if r["session_id"] == session]
        observations = [r for r in wires if r["session_id"] == session]
        assert len(turns) == len(observations) == 2
        assert [r["turn_id"] for r in turns] == ["1", "2"]
        assert [r["parent_turn_id"] for r in turns] == [None, "1"]
        for turn, wire in zip(turns, observations, strict=True):
            assert turn["input_ids"] == wire["input_ids"]
            assert turn["output_ids"] == wire["output_ids"]
            assert turn["evidence_kind"] == "observed_rollout"
            assert turn["metadata"]["collector_version"] == "0.1.0a7"
        assert [len(r["input_ids"]) for r in turns] == [32, 71]
        assert [len(r["output_ids"]) for r in turns] == [145, 16]
    assert {case["previous"]["session_id"] for case in cases} == {
        "public-session-1",
        "public-session-2",
    }
    assert all(r["first_difference"]["index"] == 32 for r in report["reports"])
