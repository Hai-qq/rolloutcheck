import copy
import json
from pathlib import Path

import pytest

from rolloutcheck import inspect_case
from rolloutcheck.case import CaseError, load_case

CASES = Path(__file__).resolve().parents[1] / "cases" / "synthetic"


@pytest.fixture
def case():
    return load_case(CASES / "drift.json")[0]


@pytest.mark.parametrize(
    "filename,status",
    [
        ("drift", "FAIL"),
        ("pass", "PASS"),
        ("rewrite", "NOT_APPLICABLE"),
        ("missing", "INCONCLUSIVE"),
    ],
)
def test_documented_cases(filename, status):
    assert inspect_case(load_case(CASES / f"{filename}.json")[0])["status"] == status


def test_diff_has_original_coordinates_and_does_not_mutate(case):
    original = copy.deepcopy(case)
    result = inspect_case(case)
    assert result["first_difference"]["index"] == 2
    assert result["first_difference"]["region"] == "previous_output"
    assert result["first_difference"]["expected_id"] == 20
    assert case == original


def test_all_single_token_changes_and_shortened_prefixes(case):
    expected = case["previous"]["input_ids"] + case["previous"]["output_ids"]
    for index in range(len(expected)):
        changed = expected.copy()
        changed[index] += 1000
        case["next"]["input_ids"] = changed
        assert inspect_case(case)["first_difference"]["index"] == index
        case["next"]["input_ids"] = expected[:index]
        report = inspect_case(case)
        assert report["first_difference"]["index"] == index
        assert report["first_difference"]["actual_id"] is None


@pytest.mark.parametrize("tail", [[], [0], [20, 21, 99]])
def test_arbitrary_new_suffix_is_allowed(case, tail):
    case["next"]["input_ids"] = [10, 11, 20, 21] + tail
    assert inspect_case(case)["status"] == "PASS"


@pytest.mark.parametrize("field", ["session_id", "branch_id", "token_space"])
def test_incomparable_turns_are_not_failures(case, field):
    case["next"][field] = "different"
    assert inspect_case(case)["status"] == "NOT_APPLICABLE"


def test_missing_contract_and_broken_continuity_are_inconclusive(case):
    case["next"]["previous_turn_id"] = "not-the-previous-turn"
    assert inspect_case(case)["status"] == "INCONCLUSIVE"
    del case["contract"]
    assert inspect_case(case)["status"] == "INCONCLUSIVE"


def test_explicit_truncation_is_not_applicable(case):
    case["contract"]["history_policy"] = "truncated"
    assert inspect_case(case)["status"] == "NOT_APPLICABLE"


def test_empty_history_cannot_establish_pass(case):
    case["previous"]["input_ids"] = []
    case["previous"]["output_ids"] = []
    assert inspect_case(case)["status"] == "INCONCLUSIVE"


@pytest.mark.parametrize("invalid", [[True], [-1], [1.5], "1 2 3", None])
def test_malformed_ids_are_errors(case, invalid):
    case["next"]["input_ids"] = invalid
    with pytest.raises(CaseError):
        inspect_case(case)


def test_captured_boundaries_localize_only_an_observed_interval(case):
    case["boundaries"] = [
        {
            "name": "before_adapter",
            "origin": "captured",
            "token_space": "synthetic/v1",
            "input_ids": [10, 11, 20, 21],
        },
        {
            "name": "projection",
            "origin": "derived",
            "token_space": "synthetic/v1",
            "input_ids": [99],
        },
        {
            "name": "after_adapter",
            "origin": "captured",
            "token_space": "synthetic/v1",
            "input_ids": [10, 11, 99, 21],
        },
    ]
    result = inspect_case(case)
    assert result["localization"]["first_observed_interval"] == {
        "after": "before_adapter",
        "at_or_before": "after_adapter",
    }
    assert "projection" in result["evidence_gaps"][0]


def test_projection_alone_cannot_establish_captured_boundary(case):
    case["boundaries"] = [
        {
            "name": "projection",
            "origin": "derived",
            "token_space": "synthetic/v1",
            "input_ids": [99],
        }
    ]
    result = inspect_case(case)
    assert result["localization"]["first_observed_interval"]["at_or_before"] == "next.input_ids"


def test_input_problem_is_not_training_correctness(case):
    case["next"]["input_ids"][0] = 999
    report = inspect_case(case)
    assert report["first_difference"]["region"] == "previous_input"
    assert "not whole-training correctness" in report["scope"]


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"a":NaN}', "[1,2]", "{"])
def test_loader_rejects_ambiguous_or_invalid_json(tmp_path, raw):
    path = tmp_path / "case.json"
    path.write_text(raw)
    with pytest.raises(CaseError):
        load_case(path)


def test_unknown_evidence_cannot_appear_as_real(case):
    case["evidence"]["kind"] = "certified"
    with pytest.raises(CaseError):
        inspect_case(case)


def test_schema_version_is_not_boolean(case):
    case["schema_version"] = True
    with pytest.raises(CaseError):
        inspect_case(case)


def test_well_formed_data_roundtrips(tmp_path, case):
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case))
    loaded, digest = load_case(path)
    assert loaded == case
    assert len(digest) == 64


def test_loader_has_bounded_input(tmp_path, monkeypatch):
    import rolloutcheck.case as case_module

    monkeypatch.setattr(case_module, "MAX_CASE_BYTES", 8)
    path = tmp_path / "large.json"
    path.write_text('{"long": "value"}')
    with pytest.raises(CaseError, match="exceeds"):
        load_case(path)


@pytest.mark.parametrize(
    "section,field", [("evidence", "kind"), ("contract", "mode"), ("contract", "history_policy")]
)
@pytest.mark.parametrize("invalid", [[], {}, 0, True])
def test_malformed_enum_is_case_error(case, section, field, invalid):
    case[section][field] = invalid
    with pytest.raises(CaseError):
        inspect_case(case)
