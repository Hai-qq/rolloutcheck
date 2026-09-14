"""Bounded JSON loading. Case data is never executable."""

import hashlib
import json
from pathlib import Path

MAX_CASE_BYTES = 16 * 1024 * 1024


class CaseError(ValueError):
    """An invalid or unsupported case document."""


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CaseError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise CaseError(f"Non-finite JSON value: {value}")


def load_case(path: str | Path) -> tuple[dict, str]:
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_CASE_BYTES + 1)
    return parse_object(raw), hashlib.sha256(raw).hexdigest()


def parse_object(raw: bytes) -> dict:
    """Parse one bounded object, shared by case files and individual trace lines."""
    if len(raw) > MAX_CASE_BYTES:
        raise CaseError(f"Case exceeds {MAX_CASE_BYTES} bytes")
    try:
        value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise CaseError(f"Invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise CaseError("Case must be a JSON object")
    return value
