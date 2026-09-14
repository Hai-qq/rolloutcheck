import json
from pathlib import Path

import pytest

from rolloutcheck import evidence
from rolloutcheck.cli import main
from rolloutcheck.history import inspect_case
from rolloutcheck.text_report import MAX_DETAILS, render_text
from rolloutcheck.trace import inspect_trace_bytes

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "cases/observed/slime-http-cuda"


@pytest.mark.parametrize(
    "name,code,status",
    [
        ("pass", 0, "PASS"),
        ("drift", 1, "FAIL"),
        ("missing", 3, "INCONCLUSIVE"),
        ("rewrite", 4, "NOT_APPLICABLE"),
    ],
)
def test_text_and_json_preserve_status_and_exit_code(capsys, name, code, status):
    path = ROOT / "cases/synthetic" / f"{name}.json"
    assert main(["inspect", str(path), "--format", "text"]) == code
    text = capsys.readouterr().out
    assert text.startswith(f"RolloutCheck: {status}\n")
    assert "turn=unknown" in text  # Older standalone cases omit the child ID.
    assert main(["inspect", str(path)]) == code
    assert json.loads(capsys.readouterr().out)["status"] == status


def test_verified_native_sessions_have_observed_child_ids(capsys):
    assert main(["verify-evidence", str(NATIVE / "evidence"), "--format", "text"]) == 1
    text = capsys.readouterr().out
    for session in ("public-session-1", "public-session-2"):
        assert f'Next:     session="{session}" branch="main" turn="2" declared_parent="1"' in text
    assert text.count("First difference: token 32") == 2
    assert "Capture completion: complete" in text and "Bundle integrity: verified" in text
    # Legacy report bytes/fields remain exactly reproducible with new identity details.
    report, cases = evidence.verify_evidence_details(NATIVE / "evidence")
    report.pop("verification")
    assert report == json.loads((NATIVE / "report.json").read_text())
    assert all(case["next"]["turn_id"] == "2" for case in cases)


def test_fail_does_not_hide_missing_completion_or_capture_gaps():
    records = [json.loads(line) for line in (NATIVE / "trace.jsonl").read_text().splitlines()]
    records[-1] = {
        "trace_version": 2,
        "record_type": "capture_gap",
        "trace_id": "http-capture",
        "sequence": 4,
        "evidence_kind": "observed_rollout",
        "reason": "omitted_callback",
    }
    raw = ("\n".join(json.dumps(r) for r in records) + "\n").encode()
    report, cases = inspect_trace_bytes(raw, include_turn_ids=True)
    text = render_text(report, cases)
    assert text.startswith("RolloutCheck: FAIL")
    assert "Capture completion: missing" in text
    assert 'Record 4: "omitted_callback"' in text
    assert "FAIL=2" in text


def test_terminal_controls_are_escaped_and_long_fields_bounded():
    case = json.loads((ROOT / "cases/synthetic/drift.json").read_text())
    payload = "\x1b[2J\r\nFORGED-PASS\u202e"
    case["case_id"] = payload + "x" * 10000
    for side in ("previous", "next"):
        case[side]["session_id"] = payload
    text = render_text(inspect_case(case), [case])
    assert "\x1b" not in text and "\r" not in text and "\u202e" not in text
    assert "\\u001b[2J\\r\\nFORGED-PASS\\u202e" in text
    assert "[truncated]" in text and len(text) < 3000
    assert not any(line.startswith("FORGED-PASS") for line in text.splitlines())


def test_errors_never_claim_verified_integrity(capsys, tmp_path):
    bundle = tmp_path / "bundle"
    evidence.export_trace(NATIVE / "trace.jsonl", bundle)
    (bundle / "report.json").write_text("{}")
    assert main(["verify-evidence", str(bundle), "--format", "text"]) == 2
    text = capsys.readouterr().out
    assert text.startswith("RolloutCheck: ERROR")
    assert "No verified diagnostic" in text and "Bundle integrity: verified" not in text


def test_trace_text_exports_identical_case_and_bundle_bytes(capsys, tmp_path):
    source = NATIVE / "trace.jsonl"
    for mode in ("json", "text"):
        assert (
            main(
                [
                    "inspect-trace",
                    str(source),
                    "--cases-dir",
                    str(tmp_path / mode),
                    "--format",
                    mode,
                ]
            )
            == 1
        )
        capsys.readouterr()
        assert (
            main(
                [
                    "export-trace-evidence",
                    str(source),
                    str(tmp_path / (mode + "-bundle")),
                    "--format",
                    mode,
                ]
            )
            == 1
        )
        capsys.readouterr()
    for filename in (tmp_path / "json").iterdir():
        assert filename.read_bytes() == (tmp_path / "text" / filename.name).read_bytes()
    for filename in (tmp_path / "json-bundle").iterdir():
        assert filename.read_bytes() == (tmp_path / "text-bundle" / filename.name).read_bytes()


def test_verified_identity_comes_from_checked_snapshot(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    evidence.export_trace(NATIVE / "trace.jsonl", bundle)
    read = evidence._read
    source_reads = []

    def replace_after_read(path, limit, **kwargs):
        raw = read(path, limit, **kwargs)
        if path.name == "trace.jsonl":
            source_reads.append(path)
            path.write_bytes(raw.replace(b"public-session-1", b"changed-session1"))
        return raw

    monkeypatch.setattr(evidence, "_read", replace_after_read)
    report, cases = evidence.verify_evidence_details(bundle)
    text = render_text(report, cases)
    assert len(source_reads) == 1
    assert "public-session-1" in text and "changed-session1" not in text
    assert "Bundle integrity: verified" in text  # Attests the byte snapshot, not future writes.


def test_large_trace_prioritizes_failure_and_reports_omissions():
    report, cases = inspect_trace_bytes(
        (NATIVE / "trace.jsonl").read_bytes(), include_turn_ids=True
    )
    failed = report["reports"][0]
    passes = [{**failed, "case_id": f"pass-{i}", "status": "PASS"} for i in range(30)]
    report["reports"] = passes + [failed]
    report["counts"] = {"PASS": 30, "FAIL": 1}
    report["transitions"] = 31
    report["capture_gaps"] = [{"sequence": i, "reason": "missing"} for i in range(25)]
    text = render_text(report, cases)
    assert text.index('FAIL  "transition-') < text.index('PASS  "pass-0"')
    assert f"{31 - MAX_DETAILS} more transitions" in text
    assert "5 more capture gaps" in text and "FAIL=1, PASS=30" in text


def test_empty_trace_has_no_invented_completion():
    report, cases = inspect_trace_bytes(b"")
    text = render_text(report, cases)
    assert "INCONCLUSIVE" in text and "no transitions" in text
    assert "completion: not recorded" in text


def test_single_case_text_export_reports_destination_and_can_be_verified(capsys, tmp_path):
    source = ROOT / "cases/synthetic/drift.json"
    destination = tmp_path / "single-bundle"
    assert main(["export-evidence", str(source), str(destination), "--format", "text"]) == 1
    text = capsys.readouterr().out
    assert "Export: witness_only; directory=" in text
    assert main(["verify-evidence", str(destination), "--format", "text"]) == 1
    assert "Bundle integrity: verified" in capsys.readouterr().out
    assert (destination / "case.json").read_bytes() == source.read_bytes()
