"""Compare token IDs under an explicitly declared history-preservation contract."""

from .case import CaseError

CONTRACT_VERSION = "history-prefix/v1"
EVIDENCE_KINDS = ("synthetic", "controlled_upstream_transform", "observed_rollout")


def _object(case, key):
    value = case.get(key, {})
    if not isinstance(value, dict):
        raise CaseError(f"{key} must be an object")
    return value


def _ids(value, field):
    if not isinstance(value, list) or any(type(x) is not int or x < 0 for x in value):
        raise CaseError(f"{field} must be an array of non-negative integer token IDs")
    return value


def _difference(expected, actual):
    for index, token in enumerate(expected):
        if index >= len(actual) or actual[index] != token:
            return index
    return None


def inspect_case(case: dict) -> dict:
    """Return a JSON-safe report. PASS is limited to this token-prefix contract.

    All evidence labels and continuity metadata are supplied by the caller;
    this function does not authenticate them or infer causality from a diff.
    """
    if not isinstance(case, dict):
        raise CaseError("Case must be an object")
    if type(case.get("schema_version")) is not int or case["schema_version"] != 1:
        raise CaseError("Unsupported or missing schema_version (expected 1)")
    if not isinstance(case.get("case_id"), str) or not case["case_id"]:
        raise CaseError("case_id must be a non-empty string")
    contract = _object(case, "contract")
    evidence = _object(case, "evidence")
    previous = _object(case, "previous")
    following = _object(case, "next")
    if evidence.get("kind") not in EVIDENCE_KINDS:
        raise CaseError("evidence.kind must explicitly describe the case's provenance")
    report = {
        "report_version": 1,
        "case_id": case["case_id"],
        "check": "RC-HISTORY-DRIFT",
        "evidence_kind": evidence["kind"],
        "scope": "declared history-prefix contract only; not whole-training correctness",
    }

    def finish(status, reason, **details):
        return dict(report, status=status, reason=reason, **details)

    # Malformed supplied data is an ERROR, even if other evidence is absent.
    for label, record, fields in (
        ("previous", previous, ("input_ids", "output_ids")),
        ("next", following, ("input_ids",)),
    ):
        for field in fields:
            if field in record:
                _ids(record[field], f"{label}.{field}")
        for field in ("session_id", "branch_id", "token_space", "turn_id", "previous_turn_id"):
            if field in record and (not isinstance(record[field], str) or not record[field]):
                raise CaseError(f"{label}.{field} must be a non-empty string")
    snapshots = case.get("boundaries", [])
    if not isinstance(snapshots, list):
        raise CaseError("boundaries must be an ordered array")
    names = set()
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            raise CaseError("Each boundary must be an object")
        name = snapshot.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise CaseError("Boundary names must be unique non-empty strings")
        names.add(name)
        if snapshot.get("origin") not in ("captured", "derived"):
            raise CaseError("Boundary origin must be captured or derived")
        _ids(snapshot.get("input_ids"), f"boundary {name}.input_ids")
        if not isinstance(snapshot.get("token_space"), str) or not snapshot["token_space"]:
            raise CaseError("Boundary token_space must be a non-empty string")
    if contract.get("mode") not in (None, "unknown", "append_only", "history_rewrite"):
        raise CaseError("Unsupported contract mode")
    if contract.get("history_policy") not in (None, "unknown", "preserved", "truncated"):
        raise CaseError("Unsupported history_policy")
    if contract.get("mode") == "history_rewrite" or contract.get("history_policy") == "truncated":
        return finish("NOT_APPLICABLE", "Path explicitly permits history rewriting or truncation")
    if contract.get("version") != CONTRACT_VERSION or contract.get("mode") != "append_only":
        return finish("INCONCLUSIVE", "Missing or unsupported append-only contract")
    if contract.get("history_policy") != "preserved":
        return finish("INCONCLUSIVE", "History retention is not established")
    missing = []
    for label, record, fields in (
        (
            "previous",
            previous,
            ("session_id", "branch_id", "turn_id", "token_space", "input_ids", "output_ids"),
        ),
        (
            "next",
            following,
            ("session_id", "branch_id", "previous_turn_id", "token_space", "input_ids"),
        ),
    ):
        missing.extend(f"{label}.{field}" for field in fields if field not in record)
    if missing:
        return finish("INCONCLUSIVE", "Required evidence is missing", missing=missing)
    if any(previous[key] != following[key] for key in ("session_id", "branch_id")):
        return finish("NOT_APPLICABLE", "Turns belong to different sessions or branches")
    if previous["token_space"] != following["token_space"]:
        return finish("NOT_APPLICABLE", "Token spaces differ; integer IDs cannot be compared")
    if previous["turn_id"] != following["previous_turn_id"]:
        return finish("INCONCLUSIVE", "Direct turn continuity is not established")
    expected = previous["input_ids"] + previous["output_ids"]
    actual = following["input_ids"]
    if not expected:
        return finish("INCONCLUSIVE", "Empty history provides no preservation evidence")
    first = _difference(expected, actual)
    comparisons = []
    last_normal = "previous.input_ids + previous.output_ids"
    first_interval = None
    gaps = []
    for snapshot in snapshots:
        if snapshot["token_space"] != previous["token_space"]:
            gaps.append(f"{snapshot['name']}: incomparable token space")
            continue
        offset = _difference(expected, snapshot["input_ids"])
        comparisons.append(
            {"name": snapshot["name"], "origin": snapshot["origin"], "first_difference": offset}
        )
        if snapshot["origin"] == "derived":
            gaps.append(f"{snapshot['name']}: derived projection is not a captured boundary")
            continue
        if first_interval is None:
            if offset is None:
                last_normal = snapshot["name"]
            else:
                first_interval = {"after": last_normal, "at_or_before": snapshot["name"]}
    details = {
        "expected_history_tokens": len(expected),
        "next_input_tokens": len(actual),
        "boundary_comparisons": comparisons,
        "evidence_gaps": gaps,
    }
    if first is None:
        return finish("PASS", "Next input preserves the complete declared history", **details)
    start, end = max(0, first - 4), first + 5
    return finish(
        "FAIL",
        "Next input violates the declared history-prefix contract",
        first_difference={
            "index": first,
            "region": "previous_input" if first < len(previous["input_ids"]) else "previous_output",
            "expected_id": expected[first],
            "actual_id": actual[first] if first < len(actual) else None,
            "kind": "missing_suffix" if first >= len(actual) else "different_token",
            "window_start": start,
            "expected_window": expected[start:end],
            "actual_window": actual[start : min(end, len(expected))],
        },
        localization={
            "first_observed_interval": first_interval
            or {"after": last_normal, "at_or_before": "next.input_ids"},
            "claim": "observed interval, not a proven root cause or causal link to final drift",
        },
        **details,
    )
