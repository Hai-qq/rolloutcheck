"""Completion boundaries, crash recovery, and portable evidence regressions."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rolloutcheck.case import CaseError
from rolloutcheck.cli import main
from rolloutcheck.evidence import export_trace, verify_evidence
from rolloutcheck.trace import TraceRecorder, inspect_trace


def generation(writer, turn=1, *, drift=False):
    writer.record(
        session_id="public-test", branch_id="main", turn_id=str(turn),
        parent_turn_id=None if turn == 1 else "1", token_space="synthetic",
        input_ids=[1] if turn == 1 else ([99] if drift else [1, 2, 3]),
        output_ids=[2],
        contract={"version": "history-prefix/v1", "mode": "append_only",
                  "history_policy": "preserved"},
        termination={},
    )


def capture(path, *, complete=True, drift=False):
    with TraceRecorder(path, trace_id="completion-test", evidence_kind="synthetic") as writer:
        generation(writer)
        generation(writer, 2, drift=drift)
        if complete:
            writer.finalize(expected_generations=2)


@pytest.mark.parametrize("complete,drift,status,exit_code", [
    (True, False, "PASS", 0), (False, False, "INCONCLUSIVE", 3),
    (True, True, "FAIL", 1), (False, True, "FAIL", 1),
])
def test_completion_survives_cli_bundle_handoff(
    tmp_path, capsys, complete, drift, status, exit_code,
):
    path = tmp_path / "trace.jsonl"
    capture(path, complete=complete, drift=drift)
    report, cases = inspect_trace(path)
    assert report["status"] == status
    assert report["records"] == 2 and report["transitions"] == len(cases) == 1
    assert report["capture_completion"]["state"] == ("complete" if complete else "missing")
    assert report["capture_completion"]["expected_generations"] == (2 if complete else None)
    destination = tmp_path / "bundle"
    assert export_trace(path, destination) == report
    destination.rename(tmp_path / "relocated")
    assert verify_evidence(tmp_path / "relocated")["status"] == status
    assert main(["verify-evidence", str(tmp_path / "relocated")]) == exit_code
    assert json.loads(capsys.readouterr().out)["capture_completion"] == report["capture_completion"]


def test_whole_line_truncation_cannot_turn_partial_capture_into_pass(tmp_path):
    path = tmp_path / "trace.jsonl"
    capture(path)
    lines = path.read_bytes().splitlines(keepends=True)
    assert json.loads(lines[-1])["record_type"] == "capture_complete"
    path.write_bytes(b"".join(lines[:-1]))
    report, _ = inspect_trace(path)
    assert report["status"] == "INCONCLUSIVE" and report["counts"] == {"PASS": 1}
    assert report["capture_completion"]["state"] == "missing"


def test_abrupt_process_exit_before_health_check_is_inconclusive(tmp_path):
    # os._exit skips context-manager cleanup, matching an abrupt collector process loss.
    # Compare versions on exactly the same two saved, otherwise passing generations.
    for version, status in [(1, "PASS"), (2, "INCONCLUSIVE")]:
        path = tmp_path / f"v{version}.jsonl"
        script = '''
import os, sys
from rolloutcheck.trace import TraceRecorder
w = TraceRecorder(sys.argv[1], trace_id="crash", evidence_kind="synthetic",
                  trace_version=int(sys.argv[2]))
for n, inputs in [(1, [1]), (2, [1, 2, 3])]:
    w.record(session_id="s", branch_id="b", turn_id=str(n),
             parent_turn_id=None if n == 1 else "1", token_space="test",
             input_ids=inputs, output_ids=[2], termination={},
             contract={"version":"history-prefix/v1", "mode":"append_only",
                       "history_policy":"preserved"})
os._exit(17)
'''
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
        child = subprocess.run(
            [sys.executable, "-c", script, str(path), str(version)],
            env=env, capture_output=True, timeout=15,
        )
        assert child.returncode == 17, child.stderr.decode()
        report, _ = inspect_trace(path)
        assert report["status"] == status and report["counts"] == {"PASS": 1}


@pytest.mark.parametrize("field,value", [
    ("expected_generations", 3), ("generations", 1), ("capture_gaps", 1),
    ("prefix_sha256", "0" * 64), ("expected_generations", True),
    ("trace_version", 1), ("sequence", 0),
])
def test_invalid_footer_fails_closed(tmp_path, field, value):
    path = tmp_path / "trace.jsonl"
    capture(path)
    lines = path.read_bytes().splitlines(keepends=True)
    footer = json.loads(lines[-1])
    footer[field] = value
    path.write_bytes(b"".join(lines[:-1]) + json.dumps(footer).encode() + b"\n")
    with pytest.raises(CaseError):
        inspect_trace(path)


def test_changed_generation_is_detected_by_footer_digest(tmp_path):
    path = tmp_path / "trace.jsonl"
    capture(path)
    path.write_bytes(path.read_bytes().replace(b'"input_ids": [1]', b'"input_ids": [9]'))
    with pytest.raises(CaseError, match="digest"):
        inspect_trace(path)


@pytest.mark.parametrize("suffix", ["generation", "duplicate-footer"])
def test_nothing_can_follow_completion(tmp_path, suffix):
    path = tmp_path / "trace.jsonl"
    capture(path)
    lines = path.read_bytes().splitlines(keepends=True)
    extra = json.loads(lines[0] if suffix == "generation" else lines[-1])
    extra["sequence"] = 3
    path.write_bytes(b"".join(lines) + json.dumps(extra).encode() + b"\n")
    with pytest.raises(CaseError, match="final record"):
        inspect_trace(path)


def test_count_mismatch_is_persistent_and_cannot_be_erased_by_retry(tmp_path):
    path = tmp_path / "trace.jsonl"
    with TraceRecorder(path, trace_id="t", evidence_kind="synthetic") as writer:
        generation(writer)
        generation(writer, 2)
        with pytest.raises(CaseError, match="count"):
            writer.finalize(expected_generations=3)
        writer.finalize(expected_generations=2)
    report, _ = inspect_trace(path)
    assert report["status"] == "INCONCLUSIVE" and report["counts"] == {"PASS": 1}
    assert report["capture_gaps"][0]["reason"] == "generation_count_mismatch"


def test_finalization_prevents_accidental_append_and_double_finalize(tmp_path):
    path = tmp_path / "trace.jsonl"
    with TraceRecorder(path, trace_id="t", evidence_kind="synthetic") as writer:
        generation(writer)
        writer.finalize(expected_generations=1)
        for action in (lambda: generation(writer, 2), lambda: writer.record_gap("late"),
                       lambda: writer.finalize(expected_generations=1)):
            with pytest.raises(CaseError, match="finalized"):
                action()
    report, _ = inspect_trace(path)
    assert report["status"] == "INCONCLUSIVE"  # No comparable pair, even with a valid footer.
    assert report["capture_completion"]["state"] == "complete"


@pytest.mark.parametrize("expected", [-1, True, 1.0, None])
def test_invalid_expected_count_does_not_write_footer(tmp_path, expected):
    path = tmp_path / "trace.jsonl"
    with TraceRecorder(path, trace_id="t", evidence_kind="synthetic") as writer:
        generation(writer)
        with pytest.raises(CaseError):
            writer.finalize(expected_generations=expected)
    assert inspect_trace(path)[0]["capture_completion"]["state"] == "missing"


def test_failed_persistence_cannot_be_retried_into_complete(tmp_path):
    path = tmp_path / "trace.jsonl"
    with TraceRecorder(path, trace_id="t", evidence_kind="synthetic") as writer:
        generation(writer)
        generation(writer, 2)
        original = writer._stream

        class FailedStream:
            closed = False

            def write(self, raw):
                raise OSError("simulated disk full")

        writer._stream = FailedStream()
        with pytest.raises(OSError):
            writer.finalize(expected_generations=2)
        writer._stream = original
        with pytest.raises(CaseError, match="persistence failure"):
            writer.finalize(expected_generations=2)
    assert inspect_trace(path)[0]["status"] == "INCONCLUSIVE"


def test_empty_completed_trace_remains_inconclusive_and_hashes_exact_prefix(tmp_path):
    path = tmp_path / "empty.jsonl"
    with TraceRecorder(path, trace_id="t", evidence_kind="synthetic") as writer:
        writer.finalize(expected_generations=0)
    assert json.loads(path.read_bytes())["prefix_sha256"] == hashlib.sha256(b"").hexdigest()
    report, _ = inspect_trace(path)
    assert report["status"] == "INCONCLUSIVE" and report["records"] == 0
