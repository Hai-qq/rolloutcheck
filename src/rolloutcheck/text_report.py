"""Bounded, terminal-safe views of checked reports; JSON remains authoritative."""

import json

MAX_DETAILS = 20


def _safe(value):
    """Escape terminal controls, newlines and bidi controls in untrusted fields."""
    if value is None:
        return "unknown"
    text = str(value)
    shortened = text[:160]
    return json.dumps(shortened, ensure_ascii=True) + (" [truncated]" if len(text) > 160 else "")


def _identity(case):
    previous, following = case.get("previous", {}), case.get("next", {})
    return [
        "  Previous: session={} branch={} turn={}".format(
            *(_safe(previous.get(k)) for k in ("session_id", "branch_id", "turn_id"))
        ),
        "  Next:     session={} branch={} turn={} declared_parent={}".format(
            *(
                _safe(following.get(k))
                for k in ("session_id", "branch_id", "turn_id", "previous_turn_id")
            )
        ),
    ]


def _detail(report, case):
    lines = [f"{report['status']}  {_safe(report.get('case_id'))}"]
    if case is not None:
        lines.extend(_identity(case))
    lines.append("  Reason: " + _safe(report.get("reason")))
    if report.get("evidence_kind"):
        lines.append("  Evidence label: " + _safe(report["evidence_kind"]))
    difference = report.get("first_difference")
    if difference:
        actual = (
            str(difference["actual_id"])
            if difference["actual_id"] is not None
            else "end of next input"
        )
        lines.append(
            f"  First difference: token {difference['index']} (zero-based), "
            f"{difference['region']}; expected {difference['expected_id']}, actual {actual}"
        )
    if "expected_history_tokens" in report:
        lines.append(
            f"  Tokens: retained history {report['expected_history_tokens']}; "
            f"next input {report['next_input_tokens']}"
        )
    interval = report.get("localization", {}).get("first_observed_interval")
    if interval:
        lines.append(
            f"  Observed interval: after {_safe(interval['after'])}, "
            f"at or before {_safe(interval['at_or_before'])}"
        )
        lines.append("  This interval does not prove the cause of drift.")
    for boundary in report.get("boundary_comparisons", [])[:MAX_DETAILS]:
        offset = boundary["first_difference"]
        result = "prefix preserved" if offset is None else f"first difference at token {offset}"
        lines.append(f"  Boundary {_safe(boundary['name'])} ({boundary['origin']}): {result}")
    hidden_boundaries = len(report.get("boundary_comparisons", [])) - MAX_DETAILS
    if hidden_boundaries > 0:
        lines.append(f"  {hidden_boundaries} more boundaries; use JSON for all.")
    for key, label in (("missing", "Missing"), ("evidence_gaps", "Evidence gap")):
        for value in report.get(key, [])[:MAX_DETAILS]:
            lines.append(f"  {label}: {_safe(value)}")
        hidden = len(report.get(key, [])) - MAX_DETAILS
        if hidden > 0:
            lines.append(f"  {hidden} more {key} entries; use JSON for all.")
    return lines


def render_text(report, cases=()):
    """Render results without re-reading source files or changing diagnostic status.

    Callers must supply cases from the same checked snapshot. No text is decoded
    from token IDs. Detailed items are capped; counts and incomplete capture state
    are never hidden. Unknown child IDs stay unknown for older standalone cases.
    """
    status = report["status"]
    lines = [f"RolloutCheck: {status}"]
    if status == "ERROR":
        return "\n".join(
            lines
            + ["Reason: " + _safe(report.get("reason")), "No verified diagnostic is available."]
        )
    is_trace = "reports" in report
    if is_trace:
        lines.append("Trace: " + _safe(report.get("trace_id")))
        lines.append(
            f"Records: {report['records']}; roots: {report['roots']}; "
            f"transitions: {report['transitions']}"
        )
        lines.append(
            "Results: "
            + (
                ", ".join(
                    f"{key}={report['counts'][key]}"
                    for key in ("FAIL", "INCONCLUSIVE", "NOT_APPLICABLE", "PASS")
                    if report["counts"].get(key)
                )
                or "no transitions"
            )
        )
        completion = report.get("capture_completion")
        if completion is None:
            lines.append("Capture completion: not recorded (legacy or empty trace).")
        else:
            lines.append(
                f"Capture completion: {completion['state']}; "
                f"generations={completion['generations']}; "
                f"expected={completion['expected_generations']}"
            )
            if completion["state"] == "missing":
                lines.append("Capture is incomplete. Observed FAILs remain actionable.")
        gaps = report.get("capture_gaps", [])
        if gaps:
            lines.append(f"Capture gaps: {len(gaps)}")
            for gap in gaps[:MAX_DETAILS]:
                lines.append(f"  Record {gap['sequence']}: {_safe(gap['reason'])}")
            if len(gaps) > MAX_DETAILS:
                lines.append(f"  {len(gaps) - MAX_DETAILS} more capture gaps; use JSON for all.")
    verification = report.get("verification")
    if verification:
        lines.append("Bundle integrity: verified; authenticity: not established.")
    exported = report.get("export")
    if exported:
        lines.append("Export: witness_only; directory=" + _safe(exported["directory"]))
    digest = report.get("trace_sha256", report.get("case_sha256"))
    if digest:
        lines.append("Source SHA256: " + _safe(digest))
    results = report["reports"] if is_trace else [report]
    # Show failures even if many earlier transitions passed. Counts above retain
    # every status; omitted details are explicit and available in the JSON view.
    priority = {"FAIL": 0, "INCONCLUSIVE": 1, "NOT_APPLICABLE": 2, "PASS": 3}
    ordered = sorted(results, key=lambda r: priority[r["status"]])
    by_id = {case["case_id"]: case for case in cases}
    for result in ordered[:MAX_DETAILS]:
        lines.append("")
        lines.extend(_detail(result, by_id.get(result["case_id"])))
    if len(ordered) > MAX_DETAILS:
        lines.append(f"\n{len(ordered) - MAX_DETAILS} more transitions; use JSON for all details.")
    if status == "FAIL":
        lines.append(
            "\nNext: compare the retained IDs with captured conversion boundaries. "
            "Confirm whether history preservation is the intended contract."
        )
    elif status == "INCONCLUSIVE":
        lines.append(
            "\nNext: resolve the reported contract, continuity or capture gaps; "
            "do not interpret missing evidence as PASS."
        )
    elif status == "NOT_APPLICABLE":
        lines.append(
            "\nNext: use a check appropriate to the declared rewriting, truncation "
            "or token-space semantics."
        )
    lines.append(
        "Scope: declared token-history contract only; no model replay, "
        "root-cause proof or whole-training correctness claim."
    )
    return "\n".join(lines)
