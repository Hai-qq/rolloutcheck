import hashlib
import json
import os
from pathlib import Path

import pytest

from rolloutcheck import evidence
from rolloutcheck.case import CaseError
from rolloutcheck.cli import main

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "cases/observed/slime-sglang-cuda"


def command(capsys, *args):
    code = main(list(map(str, args)))
    return code, json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("mode,code", [("adapter", 1), ("control", 0)])
def test_native_trace_bundle_roundtrip_preserves_source_and_check_status(
    tmp_path, capsys, mode, code
):
    source = NATIVE / f"{mode}.trace.jsonl"
    bundle = tmp_path / "bundle"
    actual, exported = command(capsys, "export-trace-evidence", source, bundle)
    assert actual == code
    assert (bundle / "trace.jsonl").read_bytes() == source.read_bytes()
    assert json.loads((bundle / "report.json").read_text()) == json.loads(
        (NATIVE / f"{mode}.report.json").read_text()
    )
    actual, verified = command(capsys, "verify-evidence", bundle)
    assert actual == code and verified["status"] == exported["status"]
    assert verified["verification"] == {
        "integrity": "verified",
        "kind": "witness_only",
        "authenticity": "not_established",
    }
    assert set(p.name for p in bundle.iterdir()) == {
        "trace.jsonl",
        "report.json",
        "README.md",
        "manifest.json",
    }
    if os.name == "posix":
        assert bundle.stat().st_mode & 0o777 == 0o700
        assert all(p.stat().st_mode & 0o777 == 0o600 for p in bundle.iterdir())
    assert command(capsys, "export-trace-evidence", source, bundle)[0] == 2
    assert (bundle / "trace.jsonl").read_bytes() == source.read_bytes()


@pytest.mark.parametrize("name,code", [("pass", 0), ("drift", 1), ("missing", 3), ("rewrite", 4)])
def test_case_bundles_gain_verification_without_changing_exit_contract(
    tmp_path, capsys, name, code
):
    bundle = tmp_path / "bundle"
    source = ROOT / f"cases/synthetic/{name}.json"
    assert command(capsys, "export-evidence", source, bundle)[0] == code
    assert (bundle / "case.json").read_bytes() == source.read_bytes()
    assert command(capsys, "verify-evidence", bundle)[0] == code


def test_capture_gap_is_preserved_in_bundle(tmp_path, capsys):
    source = ROOT / "cases/synthetic/slime-adapter/missing-context.trace.jsonl"
    bundle = tmp_path / "bundle"
    assert command(capsys, "export-trace-evidence", source, bundle)[0] == 3
    code, report = command(capsys, "verify-evidence", bundle)
    assert code == 3 and report["capture_gaps"]
    assert (bundle / "trace.jsonl").read_bytes() == source.read_bytes()


def test_failure_and_later_gap_both_survive_export(tmp_path, capsys):
    source = tmp_path / "trace.jsonl"
    raw = (NATIVE / "adapter.trace.jsonl").read_bytes()
    first = json.loads(raw.splitlines()[0])
    gap = {key: first[key] for key in ("trace_version", "trace_id", "evidence_kind")}
    gap.update(record_type="capture_gap", sequence=2, reason="Later turn was not captured")
    source.write_bytes(raw + json.dumps(gap).encode() + b"\n")
    bundle = tmp_path / "bundle"
    assert command(capsys, "export-trace-evidence", source, bundle)[0] == 1
    code, report = command(capsys, "verify-evidence", bundle)
    assert code == 1 and report["counts"] == {"FAIL": 1}
    assert report["capture_gaps"] == [{"sequence": 2, "reason": gap["reason"]}]


@pytest.mark.parametrize("filename", ["trace.jsonl", "report.json", "README.md"])
def test_changed_bundle_file_is_rejected(tmp_path, capsys, filename):
    bundle = tmp_path / "bundle"
    evidence.export_trace(NATIVE / "adapter.trace.jsonl", bundle)
    with (bundle / filename).open("ab") as stream:
        stream.write(b" ")
    code, report = command(capsys, "verify-evidence", bundle)
    assert code == 2 and "differs from manifest" in report["reason"]


@pytest.mark.parametrize("change", [{"status": "PASS"}, {"report_version": True}])
def test_rehashed_false_report_cannot_override_actual_source(tmp_path, capsys, change):
    bundle = tmp_path / "bundle"
    evidence.export_trace(NATIVE / "adapter.trace.jsonl", bundle)
    report = json.loads((bundle / "report.json").read_text())
    report.update(change)
    raw = json.dumps(report).encode()
    (bundle / "report.json").write_bytes(raw)
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["files"]["report.json"] = {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    code, checked = command(capsys, "verify-evidence", bundle)
    assert code == 2 and "recomputed" in checked["reason"]


def test_manifest_never_selects_arbitrary_file_paths(tmp_path, capsys):
    bundle = tmp_path / "bundle"
    evidence.export_trace(NATIVE / "adapter.trace.jsonl", bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["files"]["../outside"] = manifest["files"].pop("trace.jsonl")
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    code, checked = command(capsys, "verify-evidence", bundle)
    assert code == 2 and "fixed bundle filenames" in checked["reason"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink and FIFO behavior")
@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_bundle_payload_must_be_regular_file(tmp_path, capsys, kind):
    bundle = tmp_path / "bundle"
    evidence.export_trace(NATIVE / "adapter.trace.jsonl", bundle)
    (bundle / "trace.jsonl").unlink()
    if kind == "symlink":
        (bundle / "trace.jsonl").symlink_to(NATIVE / "adapter.trace.jsonl")
    else:
        os.mkfifo(bundle / "trace.jsonl")
    assert command(capsys, "verify-evidence", bundle)[0] == 2


def test_export_inspects_the_bytes_it_copies_not_a_second_source_read(tmp_path, monkeypatch):
    source = tmp_path / "trace.jsonl"
    original = (NATIVE / "adapter.trace.jsonl").read_bytes()
    source.write_bytes(original)
    inspect = evidence.inspect_trace_bytes

    def mutate_after_snapshot(raw, **kwargs):
        source.write_bytes(b"changed while inspecting")
        return inspect(raw, **kwargs)

    monkeypatch.setattr(evidence, "inspect_trace_bytes", mutate_after_snapshot)
    bundle = tmp_path / "bundle"
    evidence.export_trace(source, bundle)
    assert (bundle / "trace.jsonl").read_bytes() == original
    assert evidence.verify_evidence(bundle)["status"] == "FAIL"


def test_bounded_manifest_fails_before_creating_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, "MAX_MANIFEST_BYTES", 10)
    with pytest.raises(CaseError, match="limit"):
        evidence.export_trace(NATIVE / "adapter.trace.jsonl", tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_partial_write_leaves_no_valid_manifest(tmp_path, monkeypatch, capsys):
    write = evidence._write

    def fail_report(path, raw):
        if path.name == "report.json":
            raise OSError("injected full disk")
        write(path, raw)

    monkeypatch.setattr(evidence, "_write", fail_report)
    bundle = tmp_path / "bundle"
    code, report = command(capsys, "export-trace-evidence", NATIVE / "adapter.trace.jsonl", bundle)
    assert code == 2 and "full disk" in report["reason"]
    assert not (bundle / "manifest.json").exists()
    assert command(capsys, "verify-evidence", bundle)[0] == 2


def test_incomplete_trace_is_not_exported(tmp_path, capsys):
    source = tmp_path / "incomplete.jsonl"
    source.write_bytes((NATIVE / "adapter.trace.jsonl").read_bytes().rstrip(b"\n"))
    bundle = tmp_path / "bundle"
    assert command(capsys, "export-trace-evidence", source, bundle)[0] == 2
    assert not bundle.exists()
