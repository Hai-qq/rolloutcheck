import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/slime"))
from benchmark_capture import order_plan, summarize, validate_schedule  # noqa: E402


def samples():
    return [
        {
            "phase": "measured",
            "pair": pair,
            "mode": mode,
            "elapsed_seconds": 12 if mode == "on" else 10,
            "finalize_seconds": 0.1 if mode == "on" else 0,
            "callback_seconds": [0.2, 0.3] if mode == "on" else [],
            "capture_complete": mode == "on",
            "turns": [
                {"input_ids": [1], "output_ids": [2]},
                {"input_ids": [1, 2], "output_ids": [3]},
            ],
        }
        for pair in range(2)
        for mode in ("off", "on")
    ]


def test_constant_paired_effect_and_warmup_exclusion():
    rows = samples()
    warmup = copy.deepcopy(rows[0])
    warmup.update(phase="warmup", elapsed_seconds=1000)
    result = summarize([warmup, *rows])
    assert result["status"] == "COMPARABLE" and result["pairs"] == 2
    assert result["measured_requests"] == 8
    assert result["paired_delta_seconds"]["mean"] == 2
    assert result["paired_delta_seconds"]["mean_bootstrap_95_percent_interval"] == [2, 2]
    assert result["paired_mean_delta_percent_of_off_mean"] == 20
    assert result["callback_seconds"]["median"] == 0.25


@pytest.mark.parametrize("field", ["input_ids", "output_ids", "sampling_params", "finish_reason"])
def test_different_workload_refuses_a_performance_comparison(field):
    rows = samples()
    rows[1]["turns"][0][field] = [999]
    result = summarize(rows)
    assert result["status"] == "NOT_COMPARABLE"
    assert result["mismatched_pairs"] == [0]
    assert "paired_delta_seconds" not in result


@pytest.mark.parametrize(
    "change",
    [
        {"elapsed_seconds": float("nan")},
        {"elapsed_seconds": 0},
        {"elapsed_seconds": True},
        {"finalize_seconds": -1},
        {"callback_seconds": [float("inf"), 0.1]},
        {"capture_complete": False},
        {"callback_seconds": [0.1]},
        {"turns": []},
    ],
)
def test_invalid_measurements_are_not_accepted(change):
    rows = samples()
    rows[1].update(change)
    with pytest.raises(ValueError):
        summarize(rows)


def test_missing_or_duplicate_arms_are_rejected():
    with pytest.raises(ValueError, match="complete"):
        summarize(samples()[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        summarize([*samples(), samples()[0]])


def test_balanced_reproducible_randomized_order():
    plan = order_plan(20, 17)
    assert plan == order_plan(20, 17)
    assert plan.count(["off", "on"]) == plan.count(["on", "off"]) == 10


@pytest.mark.parametrize("pairs", [0, 1, 3, 102, True])
def test_invalid_plan_is_rejected(pairs):
    with pytest.raises(ValueError):
        order_plan(pairs, 0)


def test_saved_schedule_requires_warmup_and_recorded_random_order():
    protocol = {"pairs": 2, "seed": 0, "warmup_pairs": 1, "order": order_plan(2, 0)}
    rows = [
        {"phase": phase, "pair": pair, "mode": mode}
        for phase, plan in [("warmup", [["off", "on"]]), ("measured", protocol["order"])]
        for pair, order in enumerate(plan)
        for mode in order
    ]
    validate_schedule(rows, protocol)
    for changed in [rows[2:], list(reversed(rows)), rows[:-1]]:
        with pytest.raises(ValueError, match="schedule"):
            validate_schedule(changed, protocol)
