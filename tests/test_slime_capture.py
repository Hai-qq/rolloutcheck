import json
from types import SimpleNamespace

import pytest

from rolloutcheck.case import CaseError
from rolloutcheck.slime_capture import SlimeDebugCapture, TurnContext
from rolloutcheck.trace import TraceRecorder, inspect_trace

CONTRACT = {"version": "history-prefix/v1", "mode": "append_only", "history_policy": "preserved"}


def context(turn="1", parent=None, branch="main"):
    return TurnContext("public-session", branch, turn, parent, "test-vocabulary", CONTRACT)


def snapshot(inputs=None, outputs=None):
    return SimpleNamespace(
        prompt_ids=inputs or [1, 2],
        output_ids=outputs if outputs is not None else [3, 9],
        finish_reason="stop",
        output_log_probs=[-0.1, -0.2],
    )


def emit(callback, turn):
    callback("DO-NOT-PERSIST-RAW-BEARER", [{"content": "PRIVATE-TEXT"}], None, {}, turn)


def test_raw_turn_record_ids_and_eos_are_copied_without_persisting_messages(tmp_path):
    path = tmp_path / "trace.jsonl"
    contexts = iter([context(), context("2", "1")])
    with TraceRecorder(path, trace_id="t", evidence_kind="synthetic") as writer:
        callback = SlimeDebugCapture(writer, lambda *args: next(contexts), eos_token_ids=[9])
        turn = snapshot()
        emit(callback, turn)
        turn.output_ids[0] = 999
        emit(callback, snapshot([1, 2, 3, 9, 4]))
        callback.raise_if_failed(expected_turns=2)
    report, cases = inspect_trace(path)
    assert report["status"] == "PASS" and callback.captured == 2
    assert cases[0]["previous"]["output_ids"] == [3, 9]
    assert cases[0]["evidence"]["previous_termination"]["terminal_eos_present"] is True
    assert "RAW-BEARER" not in path.read_text() and "PRIVATE-TEXT" not in path.read_text()


def test_missing_context_creates_gap_not_a_silent_success(tmp_path):
    path = tmp_path / "trace.jsonl"
    contexts = iter([context(), None, context("2", "1")])
    with TraceRecorder(path, trace_id="t", evidence_kind="synthetic") as writer:
        callback = SlimeDebugCapture(writer, lambda *args: next(contexts))
        emit(callback, snapshot())
        emit(callback, snapshot())
        emit(callback, snapshot([1, 2, 3, 9, 4]))
        with pytest.raises(CaseError, match="incomplete"):
            callback.raise_if_failed(expected_turns=3)
    report, _ = inspect_trace(path)
    assert report["status"] == "INCONCLUSIVE"
    assert report["counts"] == {"PASS": 1} and report["records"] == 3
    assert report["capture_gaps"] == [{"sequence": 1, "reason": "turn_context_missing"}]


def test_resolver_failure_does_not_leak_exception_text(tmp_path):
    path = tmp_path / "error.jsonl"

    def bad_resolver(*args):
        raise RuntimeError("PRIVATE-TEXT")

    with TraceRecorder(path, trace_id="t", evidence_kind="synthetic") as writer:
        callback = SlimeDebugCapture(writer, bad_resolver)
        emit(callback, snapshot())
        with pytest.raises(CaseError):
            callback.raise_if_failed(expected_turns=1)
    assert "PRIVATE-TEXT" not in path.read_text()
    assert inspect_trace(path)[0]["status"] == "INCONCLUSIVE"


def test_unwritable_gap_still_fails_explicit_health_check(tmp_path):
    writer = TraceRecorder(tmp_path / "closed.jsonl", trace_id="t", evidence_kind="synthetic")
    writer.close()
    callback = SlimeDebugCapture(writer, lambda *args: context())
    with pytest.raises(CaseError, match="persist"):
        emit(callback, snapshot())
    with pytest.raises(CaseError, match="persistence failed=True"):
        callback.raise_if_failed(expected_turns=1)


@pytest.mark.parametrize(
    "turn", [snapshot(outputs=[]), SimpleNamespace(), snapshot(outputs=[True])]
)
def test_invalid_turn_or_empty_budget_response_creates_gap(tmp_path, turn):
    path = tmp_path / "invalid.jsonl"
    with TraceRecorder(path, trace_id="t", evidence_kind="synthetic") as writer:
        callback = SlimeDebugCapture(writer, lambda *args: context())
        emit(callback, turn)
    report, _ = inspect_trace(path)
    assert report["status"] == "INCONCLUSIVE" and callback.failed == 1


def test_explicit_context_preserves_forks_and_missing_ancestry(tmp_path):
    path = tmp_path / "fork.jsonl"
    contexts = iter([context(), context("2", "1", "fork")])
    with TraceRecorder(path, trace_id="t", evidence_kind="synthetic") as writer:
        callback = SlimeDebugCapture(writer, lambda *args: next(contexts))
        emit(callback, snapshot())
        emit(callback, snapshot([1, 2, 3, 9, 4]))
    assert inspect_trace(path)[0]["status"] == "INCONCLUSIVE"


def test_duplicate_context_is_not_silently_overwritten(tmp_path):
    path = tmp_path / "duplicate.jsonl"
    with TraceRecorder(path, trace_id="t", evidence_kind="synthetic") as writer:
        callback = SlimeDebugCapture(writer, lambda *args: context())
        emit(callback, snapshot())
        emit(callback, snapshot())
    assert callback.failed == 1
    assert inspect_trace(path)[0]["capture_gaps"][0]["reason"] == "turn_record_capture_failed"


def test_upstream_stop_is_not_assumed_to_prove_eos_retention(tmp_path):
    path = tmp_path / "unknown.jsonl"
    with TraceRecorder(path, trace_id="t", evidence_kind="synthetic") as writer:
        callback = SlimeDebugCapture(writer, lambda *args: context())
        emit(callback, snapshot())
    termination = json.loads(path.read_text())["termination"]
    assert termination["terminal_eos_present"] is None
    assert termination["engine_stop_token_retention"] == "unverified"


def test_owner_count_detects_a_callback_that_was_never_invoked(tmp_path):
    path = tmp_path / "missing-callback.jsonl"
    contexts = iter([context(), context("2", "1")])
    with TraceRecorder(path, trace_id="t", evidence_kind="synthetic") as writer:
        callback = SlimeDebugCapture(writer, lambda *args: next(contexts))
        emit(callback, snapshot())
        emit(callback, snapshot([1, 2, 3, 9, 4]))
        # The owner knows three turns were served, including one absent callback.
        with pytest.raises(CaseError):
            callback.raise_if_failed(expected_turns=3)
    report, _ = inspect_trace(path)
    assert report["status"] == "INCONCLUSIVE" and report["counts"] == {"PASS": 1}
    assert report["capture_gaps"][-1]["reason"] == "served_turn_count_mismatch"
