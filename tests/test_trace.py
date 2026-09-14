import copy
import json
from pathlib import Path

import pytest

from rolloutcheck.case import CaseError
from rolloutcheck.cli import main
from rolloutcheck.trace import TraceRecorder, inspect_trace


def record(recorder, turn="1", parent=None, inputs=None, **kwargs):
    args = dict(
        session_id="s",
        branch_id="main",
        turn_id=turn,
        parent_turn_id=parent,
        token_space="test",
        input_ids=inputs if inputs is not None else [1, 2],
        output_ids=[3, 4],
        contract={
            "version": "history-prefix/v1",
            "mode": "append_only",
            "history_policy": "preserved",
        },
        termination={"reason": "eos", "eos_token_ids": [4], "terminal_eos_retained": True},
    )
    args.update(kwargs)
    recorder.record(**args)


def make_trace(tmp_path, next_input=(1, 2, 3, 4, 8)):
    path = tmp_path / "trace.jsonl"
    with TraceRecorder(path, trace_id="t", evidence_kind="synthetic") as writer:
        record(writer)
        record(writer, "2", "1", list(next_input))
    return path


def rewrite(path, mutate):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(records)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_capture_builds_a_case_without_retokenizing(tmp_path):
    path = make_trace(tmp_path)
    report, cases = inspect_trace(path)
    assert report["status"] == "PASS"
    assert report["records"] == 2 and report["roots"] == 1
    assert cases[0]["previous"]["output_ids"] == [3, 4]  # EOS retained
    assert cases[0]["next"]["input_ids"] == [1, 2, 3, 4, 8]
    assert cases[0]["evidence"]["trace_sha256"] == report["trace_sha256"]


def test_failed_pair_survives_extraction_and_cli_export(tmp_path, capsys):
    path = make_trace(tmp_path, [1, 2, 99])
    destination = tmp_path / "cases"
    assert main(["inspect-trace", str(path), "--cases-dir", str(destination)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["counts"] == {"FAIL": 1}
    case_file = destination / "transition-000001.json"
    assert main(["inspect", str(case_file)]) == 1
    capsys.readouterr()
    assert main(["inspect-trace", str(path), "--cases-dir", str(destination)]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "ERROR"


def test_unknown_parent_is_inconclusive_not_silently_skipped(tmp_path):
    path = make_trace(tmp_path)
    rewrite(path, lambda records: records[1].update(parent_turn_id="absent"))
    report, cases = inspect_trace(path)
    assert report["status"] == "INCONCLUSIVE"
    assert len(cases) == 1 and cases[0]["previous"] == {}


def test_parent_lookup_never_crosses_session_or_branch(tmp_path):
    path = make_trace(tmp_path)
    rewrite(path, lambda records: records[1].update(branch_id="another"))
    assert inspect_trace(path)[0]["status"] == "INCONCLUSIVE"


def test_empty_trace_and_root_only_are_not_pass(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_bytes(b"")
    assert inspect_trace(path)[0]["status"] == "INCONCLUSIVE"
    with TraceRecorder(tmp_path / "root.jsonl", trace_id="t", evidence_kind="synthetic") as writer:
        record(writer)
    assert inspect_trace(tmp_path / "root.jsonl")[0]["status"] == "INCONCLUSIVE"


def test_snapshot_is_not_changed_by_caller_mutation(tmp_path):
    path = tmp_path / "copy.jsonl"
    inputs = [1, 2]
    metadata = {"nested": {"name": "original"}}
    with TraceRecorder(path, trace_id="t", evidence_kind="synthetic") as writer:
        record(writer, inputs=inputs, metadata=metadata)
        inputs[0] = 999
        metadata["nested"]["name"] = "modified"
    captured = json.loads(path.read_text())
    assert captured["input_ids"] == [1, 2]
    assert captured["metadata"]["nested"]["name"] == "original"


def test_recorder_rejects_duplicate_turn_and_existing_path(tmp_path):
    path = tmp_path / "duplicate.jsonl"
    with TraceRecorder(path, trace_id="t", evidence_kind="synthetic") as writer:
        record(writer)
        with pytest.raises(CaseError, match="Duplicate"):
            record(writer)
    with pytest.raises(FileExistsError):
        TraceRecorder(path, trace_id="t", evidence_kind="synthetic")


@pytest.mark.parametrize(
    "change",
    [
        {"evidence_kind": []},
        {"sequence": True},
        {"sequence": 99},
        {"trace_id": "other"},
        {"output_ids": []},
        {"parent_turn_id": "2"},
        {"input_ids": [True]},
        {"contract": []},
        {"termination": None},
        {"token_space": []},
        {"trace_version": True},
    ],
)
def test_malformed_records_fail_closed(tmp_path, change):
    path = make_trace(tmp_path)
    rewrite(path, lambda records: records[1].update(change))
    with pytest.raises(CaseError):
        inspect_trace(path)


def test_duplicate_loaded_turn_rejected(tmp_path):
    path = make_trace(tmp_path)

    def duplicate(records):
        records[1] = copy.deepcopy(records[0])
        records[1]["sequence"] = 1

    rewrite(path, duplicate)
    with pytest.raises(CaseError, match="Duplicate"):
        inspect_trace(path)


def test_partial_last_line_rejected(tmp_path):
    path = make_trace(tmp_path)
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    with pytest.raises(CaseError, match="Incomplete"):
        inspect_trace(path)


def test_recorder_does_not_record_non_json_metadata(tmp_path):
    with TraceRecorder(tmp_path / "bad.jsonl", trace_id="t", evidence_kind="synthetic") as writer:
        with pytest.raises(CaseError, match="finite JSON"):
            record(writer, metadata={"value": float("nan")})
        record(writer)  # Failed writes must not consume sequence numbers.


def test_mixed_success_and_missing_evidence_do_not_aggregate_to_pass(tmp_path):
    path = make_trace(tmp_path)

    def add_missing(records):
        r = copy.deepcopy(records[-1])
        r.update(sequence=2, turn_id="3", parent_turn_id="absent")
        records.append(r)

    rewrite(path, add_missing)
    report, _ = inspect_trace(path)
    assert report["status"] == "INCONCLUSIVE"
    assert report["counts"] == {"PASS": 1, "INCONCLUSIVE": 1}


def test_committed_observed_traces_match_reports_and_cases():
    directory = Path(__file__).resolve().parents[1] / "cases/observed/qwen3-transformers"
    for mode, status in [("rerender", "FAIL"), ("canonical", "PASS")]:
        report, cases = inspect_trace(directory / f"{mode}.trace.jsonl")
        assert report == json.loads((directory / f"{mode}.report.json").read_text())
        assert cases == [json.loads((directory / f"{mode}.case.json").read_text())]
        assert report["status"] == status
        assert len(cases) == 1
        assert cases[0]["evidence"]["kind"] == "observed_rollout"
