"""Bounded, local evidence bundles. Verification never executes bundle content."""

import hashlib
import json
import os
import stat
from pathlib import Path

from . import __version__
from .case import MAX_CASE_BYTES, CaseError, parse_object
from .history import inspect_case
from .trace import MAX_TRACE_BYTES, inspect_trace_bytes

MAX_MANIFEST_BYTES = 64 * 1024
MAX_README_BYTES = 64 * 1024


def _read(path, limit, *, allow_symlink=False):
    if not allow_symlink and path.is_symlink():
        raise CaseError(f"Bundle file must not be a symlink: {path.name}")
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    if not allow_symlink:
        flags |= getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise CaseError(f"Expected a regular file: {path.name}")
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise CaseError(f"{path.name} exceeds its {limit}-byte limit")
    return raw


def _json_bytes(value, limit=MAX_CASE_BYTES):
    raw = bytearray()
    for chunk in json.JSONEncoder(
        indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).iterencode(value):
        raw.extend(chunk.encode("utf-8"))
        if len(raw) >= limit:
            raise CaseError(f"Evidence JSON exceeds its {limit}-byte limit")
    raw.extend(b"\n")
    return bytes(raw)


def _digest(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _write(path, raw):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    with os.fdopen(os.open(path, flags, 0o600), "wb") as stream:
        stream.write(raw)


def export_bundle(destination, *, kind, raw, report):
    """Write a new directory, with a manifest last. Partial writes are not valid bundles."""
    if kind not in ("case", "trace"):
        raise CaseError("Unknown evidence kind")
    limit = MAX_TRACE_BYTES if kind == "trace" else MAX_CASE_BYTES
    if len(raw) > limit:
        raise CaseError("Evidence source exceeds size limit")
    source_name = "trace.jsonl" if kind == "trace" else "case.json"
    command = "inspect-trace trace.jsonl" if kind == "trace" else "inspect case.json"
    files = {
        source_name: raw,
        "report.json": _json_bytes(report),
        "README.md": (
            "# RolloutCheck evidence bundle\n\n"
            "Status: **witness_only**. This bundle rechecks recorded IDs; it does not "
            "execute the original conversion or replay model sampling.\n\n"
            f"With RolloutCheck {__version__} installed, from this directory:\n\n"
            f"```sh\nrolloutcheck verify-evidence .\nrolloutcheck {command}\n```\n\n"
            "Verification checks file hashes and recomputes the report. Exit codes preserve "
            "the recorded check status: a valid FAIL bundle still exits 1.\n\n"
            "Hashes detect inconsistency, not authenticity. Labels remain caller-supplied. "
            "Review before sharing: token IDs can reconstruct text. A trace bundle includes "
            "the complete source trace, including other sessions and capture gaps.\n"
        ).encode("utf-8"),
    }
    manifest = {
        "bundle_version": 1,
        "kind": kind,
        "evidence_mode": "witness_only",
        "producer": {"name": "rolloutcheck", "version": __version__},
        "files": {name: _digest(data) for name, data in files.items()},
    }
    manifest_raw = _json_bytes(manifest, MAX_MANIFEST_BYTES)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name, data in files.items():
        _write(destination / name, data)
    _write(destination / "manifest.json", manifest_raw)


def export_trace(source, destination):
    raw = _read(Path(source), MAX_TRACE_BYTES, allow_symlink=True)
    report, _ = inspect_trace_bytes(raw)
    export_bundle(destination, kind="trace", raw=raw, report=report)
    return report


def verify_evidence(directory):
    """Check fixed filenames only, then recompute the report from the saved source."""
    directory = Path(directory)
    manifest = parse_object(_read(directory / "manifest.json", MAX_MANIFEST_BYTES))
    if (
        set(manifest) != {"bundle_version", "kind", "evidence_mode", "producer", "files"}
        or type(manifest["bundle_version"]) is not int
        or manifest["bundle_version"] != 1
        or manifest["kind"] not in ("case", "trace")
        or manifest["evidence_mode"] != "witness_only"
    ):
        raise CaseError("Unsupported evidence manifest")
    producer = manifest["producer"]
    if (
        not isinstance(producer, dict)
        or set(producer) != {"name", "version"}
        or producer["name"] != "rolloutcheck"
        or not isinstance(producer["version"], str)
        or not producer["version"]
    ):
        raise CaseError("Invalid evidence producer metadata")
    is_trace = manifest["kind"] == "trace"
    source_name = "trace.jsonl" if is_trace else "case.json"
    limits = {
        source_name: MAX_TRACE_BYTES if is_trace else MAX_CASE_BYTES,
        "report.json": MAX_CASE_BYTES,
        "README.md": MAX_README_BYTES,
    }
    if not isinstance(manifest["files"], dict) or set(manifest["files"]) != set(limits):
        raise CaseError("Manifest must list exactly the fixed bundle filenames")
    files = {}
    for name, limit in limits.items():
        entry = manifest["files"][name]
        if (
            not isinstance(entry, dict)
            or set(entry) != {"bytes", "sha256"}
            or type(entry["bytes"]) is not int
            or not 0 <= entry["bytes"] <= limit
            or not isinstance(entry["sha256"], str)
        ):
            raise CaseError(f"Invalid manifest entry: {name}")
        raw = _read(directory / name, limit)
        if _digest(raw) != entry:
            raise CaseError(f"Evidence file differs from manifest: {name}")
        files[name] = raw
    raw = files[source_name]
    if is_trace:
        report, _ = inspect_trace_bytes(raw)
    else:
        report = dict(inspect_case(parse_object(raw)), case_sha256=hashlib.sha256(raw).hexdigest())
    saved_report = parse_object(files["report.json"])
    if _json_bytes(saved_report) != _json_bytes(report):
        raise CaseError("Saved report differs from recomputed source report")
    return {
        **report,
        "verification": {
            "integrity": "verified",
            "kind": "witness_only",
            "authenticity": "not_established",
        },
    }
