"""Opt-in raw-token recording and offline transition extraction; no model imports."""

import copy
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from .case import CaseError, parse_object
from .history import EVIDENCE_KINDS, _ids, inspect_case

MAX_TRACE_BYTES = 64 * 1024 * 1024


def validate_record(record):
    if not isinstance(record, dict) or type(record.get("trace_version")) is not int:
        raise CaseError("Trace record requires integer trace_version")
    if record["trace_version"] not in (1, 2) or record.get("record_type") not in (
        "generation",
        "capture_gap",
        "capture_complete",
    ):
        raise CaseError("Unsupported trace version or record type")
    if not isinstance(record.get("trace_id"), str) or not record["trace_id"]:
        raise CaseError("Trace trace_id must be a nonempty string")
    if record.get("evidence_kind") not in EVIDENCE_KINDS:
        raise CaseError("Trace evidence_kind is required")
    if type(record.get("sequence")) is not int or record["sequence"] < 0:
        raise CaseError("Trace sequence must be a nonnegative integer")
    if record["record_type"] == "capture_complete":
        if record["trace_version"] != 2:
            raise CaseError("Capture completion requires trace version 2")
        for key in ("expected_generations", "generations", "capture_gaps"):
            if type(record.get(key)) is not int or record[key] < 0:
                raise CaseError(f"Capture completion {key} must be a nonnegative integer")
        digest = record.get("prefix_sha256")
        if (
            not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise CaseError("Capture completion requires a SHA256 digest")
        return
    if record["record_type"] == "capture_gap":
        if not isinstance(record.get("reason"), str) or not record["reason"]:
            raise CaseError("Capture gap requires a nonempty reason")
        return
    for key in ("session_id", "branch_id", "turn_id", "token_space"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise CaseError(f"Trace {key} must be a nonempty string")
    if "parent_turn_id" not in record:
        raise CaseError("parent_turn_id must explicitly be a string or null for a root")
    parent = record["parent_turn_id"]
    if parent is not None and (not isinstance(parent, str) or not parent):
        raise CaseError("parent_turn_id must be a nonempty string or null")
    if parent == record["turn_id"]:
        raise CaseError("A turn cannot be its own parent")
    for key in ("input_ids", "output_ids"):
        _ids(record.get(key), f"trace.{key}")
    if not record["output_ids"]:
        raise CaseError("A completed generation record must contain output IDs")
    for key in ("contract", "metadata", "termination"):
        if not isinstance(record.get(key), dict):
            raise CaseError(f"Trace {key} must be an object")
    # Validate supplied contract and boundaries through the same checker used at read time.
    inspect_case(_make_case(record, {**record, "parent_turn_id": record["turn_id"]}, "validation"))


def _make_case(previous, current, case_id):
    previous_fields = (
        "session_id",
        "branch_id",
        "turn_id",
        "token_space",
        "input_ids",
        "output_ids",
    )
    next_fields = ("session_id", "branch_id", "token_space", "input_ids")
    return {
        "schema_version": 1,
        "case_id": case_id,
        "evidence": {
            "kind": current["evidence_kind"],
            "description": "Transition extracted from declared generation records; labels are "
            "caller-supplied, not proof of authenticity.",
            "trace_id": current["trace_id"],
            "previous_sequence": previous.get("sequence"),
            "next_sequence": current["sequence"],
            "previous_termination": previous.get("termination"),
            "next_termination": current["termination"],
            "previous_metadata": previous.get("metadata"),
            "next_metadata": current["metadata"],
        },
        "contract": copy.deepcopy(current["contract"]),
        "previous": {
            key: copy.deepcopy(previous[key]) for key in previous_fields if key in previous
        },
        "next": {
            **{key: copy.deepcopy(current[key]) for key in next_fields},
            "previous_turn_id": current["parent_turn_id"],
        },
        "boundaries": copy.deepcopy(current.get("boundaries", [])),
    }


class TraceRecorder:
    """Single-process, single-writer recorder. Existing files are never appended/overwritten.

    Supply IDs at the execution boundary, not from decoded/re-tokenized messages.
    Use the optional Transformers adapter for the verified generate-call path.
    """

    def __init__(self, path, *, trace_id, evidence_kind, trace_version=2):
        if not isinstance(trace_id, str) or not trace_id or evidence_kind not in EVIDENCE_KINDS:
            raise CaseError("Explicit trace identity and evidence kind are required")
        if type(trace_version) is not int or trace_version not in (1, 2):
            raise CaseError("trace_version must be 1 or 2")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb")
        self.trace_id, self.evidence_kind = trace_id, evidence_kind
        self.trace_version = trace_version
        self._seen = set()
        self._size = 0
        self._sequence = 0
        self._generations = 0
        self._gaps = 0
        self._digest = hashlib.sha256()
        self._finalized = False
        self._write_failed = False

    def record(
        self,
        *,
        session_id,
        branch_id,
        turn_id,
        parent_turn_id,
        token_space,
        input_ids,
        output_ids,
        contract,
        termination,
        metadata=None,
        boundaries=None,
    ):
        record = copy.deepcopy(
            {
                "trace_version": self.trace_version,
                "record_type": "generation",
                "trace_id": self.trace_id,
                "sequence": self._sequence,
                "evidence_kind": self.evidence_kind,
                "session_id": session_id,
                "branch_id": branch_id,
                "turn_id": turn_id,
                "parent_turn_id": parent_turn_id,
                "token_space": token_space,
                "input_ids": input_ids,
                "output_ids": output_ids,
                "contract": contract,
                "termination": termination,
                "metadata": metadata if metadata is not None else {},
                "boundaries": boundaries if boundaries is not None else [],
            }
        )
        self._write_record(record, key=(session_id, branch_id, turn_id))

    def record_gap(self, reason):
        """Persist a collection omission so an otherwise clean trace cannot report PASS."""
        self._write_record(
            {
                "trace_version": self.trace_version,
                "record_type": "capture_gap",
                "trace_id": self.trace_id,
                "sequence": self._sequence,
                "evidence_kind": self.evidence_kind,
                "reason": reason,
            }
        )

    def _write_record(self, record, key=None):
        if self._finalized or self._stream.closed or self._write_failed:
            raise CaseError("Trace is finalized, closed, or has a persistence failure")
        validate_record(record)
        if key is not None and key in self._seen:
            raise CaseError("Duplicate turn identity in trace")
        try:
            raw = json.dumps(record, ensure_ascii=False, allow_nan=False).encode() + b"\n"
        except (ValueError, TypeError) as exc:
            raise CaseError("Trace metadata must be finite JSON data") from exc
        parse_object(raw)
        if self._size + len(raw) > MAX_TRACE_BYTES:
            raise CaseError("Trace exceeds size limit; start a new trace")
        try:
            if self._stream.write(raw) != len(raw):
                raise OSError("Incomplete trace write")
            self._stream.flush()
        except (OSError, ValueError):
            self._write_failed = True
            raise
        if key is not None:
            self._seen.add(key)
        self._sequence += 1
        self._size += len(raw)
        self._digest.update(raw)
        self._generations += record["record_type"] == "generation"
        self._gaps += record["record_type"] == "capture_gap"

    def finalize(self, *, expected_generations):
        """After requests drain, check the owner's independent count and seal v2 bytes.

        Closing alone never attests completion. Counts are caller declarations,
        not proof of engine coverage. A v1 recorder cannot write this attestation.
        """
        if self.trace_version != 2:
            raise CaseError("Finalization requires trace version 2")
        if type(expected_generations) is not int or expected_generations < 0:
            raise CaseError("expected_generations must be an independent nonnegative count")
        if expected_generations != self._generations:
            self.record_gap("generation_count_mismatch")
            raise CaseError("Expected generation count does not match captured generations")
        self._write_record({
            "trace_version": 2,
            "record_type": "capture_complete",
            "trace_id": self.trace_id,
            "sequence": self._sequence,
            "evidence_kind": self.evidence_kind,
            "expected_generations": expected_generations,
            "generations": self._generations,
            "capture_gaps": self._gaps,
            "prefix_sha256": self._digest.hexdigest(),
        })
        self._finalized = True

    def close(self):
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def inspect_trace(path, *, include_turn_ids=False):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_TRACE_BYTES + 1)
    return inspect_trace_bytes(raw, include_turn_ids=include_turn_ids)


def inspect_trace_bytes(raw, *, include_turn_ids=False):
    """Inspect one captured byte snapshot; export can preserve exactly these bytes."""
    if len(raw) > MAX_TRACE_BYTES:
        raise CaseError("Trace exceeds size limit")
    if raw and not raw.endswith(b"\n"):
        raise CaseError("Incomplete trace: last record must end in a newline")
    digest = hashlib.sha256(raw).hexdigest()
    seen, cases, reports = {}, [], []
    trace_id = evidence_kind = trace_version = None
    roots = 0
    gaps = []
    record_count = 0
    completion = None
    prefix_digest = hashlib.sha256()
    for index, line in enumerate(raw.splitlines(keepends=True)):
        if completion is not None:
            raise CaseError("Capture completion must be the final record")
        record = parse_object(line)
        validate_record(record)
        if record["sequence"] != index:
            raise CaseError("Trace sequences must be contiguous and ordered from zero")
        if index == 0:
            trace_id, evidence_kind = record["trace_id"], record["evidence_kind"]
            trace_version = record["trace_version"]
        if record["trace_version"] != trace_version:
            raise CaseError("Trace version must remain consistent")
        if record["trace_id"] != trace_id or record["evidence_kind"] != evidence_kind:
            raise CaseError("Trace identity and evidence kind must remain consistent")
        if record["record_type"] == "capture_complete":
            if (
                record["generations"] != len(seen)
                or record["expected_generations"] != len(seen)
                or record["capture_gaps"] != len(gaps)
                or record["prefix_sha256"] != prefix_digest.hexdigest()
            ):
                raise CaseError("Capture completion counts or prefix digest do not match")
            completion = record
            continue
        prefix_digest.update(line)
        record_count += 1
        if record["record_type"] == "capture_gap":
            gaps.append({"sequence": index, "reason": record["reason"]})
            continue
        key = (record["session_id"], record["branch_id"], record["turn_id"])
        if key in seen:
            raise CaseError("Duplicate turn identity in trace")
        parent = record["parent_turn_id"]
        if parent is None:
            roots += 1
        else:
            prior = seen.get((record["session_id"], record["branch_id"], parent), {})
            case = _make_case(prior, record, f"transition-{index:06d}")
            case["evidence"]["trace_sha256"] = digest
            if include_turn_ids:
                case["next"]["turn_id"] = record["turn_id"]
            result = inspect_case(case)
            cases.append(case)
            reports.append(result)
        seen[key] = record
    counts = dict(Counter(report["status"] for report in reports))
    # FAIL is actionable even if another pair lacks evidence; counts preserve the latter.
    status = next(
        (s for s in ("FAIL", "INCONCLUSIVE", "NOT_APPLICABLE", "PASS") if counts.get(s)),
        "INCONCLUSIVE",
    )
    if gaps and status != "FAIL":
        status = "INCONCLUSIVE"
    if trace_version == 2 and completion is None and status != "FAIL":
        status = "INCONCLUSIVE"
    return {
        **({"capture_completion": {
            "state": "complete" if completion is not None else "missing",
            "generations": len(seen),
            "expected_generations": (
                completion["expected_generations"] if completion is not None else None
            ),
            "scope": "Caller-declared count and saved-byte integrity; not engine coverage "
            "or authenticity. Missing completion cannot aggregate to PASS.",
        }} if trace_version == 2 else {}),
        **({"capture_gaps": gaps} if gaps else {}),
        "report_version": 1,
        "status": status,
        "trace_sha256": digest,
        "trace_id": trace_id,
        "records": record_count,
        "roots": roots,
        "transitions": len(reports),
        "counts": counts,
        "reports": reports,
        "scope": "Only explicit parent links within the same session/branch are checked; "
        "no transitions is INCONCLUSIVE, not PASS.",
    }, cases
