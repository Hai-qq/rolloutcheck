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
    if record["trace_version"] != 1 or record.get("record_type") not in (
        "generation",
        "capture_gap",
    ):
        raise CaseError("Unsupported trace version or record type")
    if not isinstance(record.get("trace_id"), str) or not record["trace_id"]:
        raise CaseError("Trace trace_id must be a nonempty string")
    if record.get("evidence_kind") not in EVIDENCE_KINDS:
        raise CaseError("Trace evidence_kind is required")
    if type(record.get("sequence")) is not int or record["sequence"] < 0:
        raise CaseError("Trace sequence must be a nonnegative integer")
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

    def __init__(self, path, *, trace_id, evidence_kind):
        if not isinstance(trace_id, str) or not trace_id or evidence_kind not in EVIDENCE_KINDS:
            raise CaseError("Explicit trace identity and evidence kind are required")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb")
        self.trace_id, self.evidence_kind = trace_id, evidence_kind
        self._seen = set()
        self._size = 0
        self._sequence = 0

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
                "trace_version": 1,
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
                "trace_version": 1,
                "record_type": "capture_gap",
                "trace_id": self.trace_id,
                "sequence": self._sequence,
                "evidence_kind": self.evidence_kind,
                "reason": reason,
            }
        )

    def _write_record(self, record, key=None):
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
        self._stream.write(raw)
        self._stream.flush()
        if key is not None:
            self._seen.add(key)
        self._sequence += 1
        self._size += len(raw)

    def close(self):
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def inspect_trace(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_TRACE_BYTES + 1)
    return inspect_trace_bytes(raw)


def inspect_trace_bytes(raw):
    """Inspect one captured byte snapshot; export can preserve exactly these bytes."""
    if len(raw) > MAX_TRACE_BYTES:
        raise CaseError("Trace exceeds size limit")
    if raw and not raw.endswith(b"\n"):
        raise CaseError("Incomplete trace: last record must end in a newline")
    digest = hashlib.sha256(raw).hexdigest()
    seen, cases, reports = {}, [], []
    trace_id = evidence_kind = None
    roots = 0
    gaps = []
    record_count = 0
    for index, line in enumerate(raw.splitlines()):
        record = parse_object(line)
        validate_record(record)
        if record["sequence"] != index:
            raise CaseError("Trace sequences must be contiguous and ordered from zero")
        if index == 0:
            trace_id, evidence_kind = record["trace_id"], record["evidence_kind"]
        if record["trace_id"] != trace_id or record["evidence_kind"] != evidence_kind:
            raise CaseError("Trace identity and evidence kind must remain consistent")
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
    return {
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
