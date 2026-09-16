"""The reward is a real answer check, including malformed-answer negatives."""

import asyncio
from types import SimpleNamespace

from rl_reward import dataset, reward, score


def test_exact_answer_reward():
    assert score(" 387\n", "387") == 1.0
    assert score("388", "387") == 0.0
    assert score("The answer is 387", "387") == 0.0
    assert score("387 or 388", "387") == 0.0
    assert score("", "387") == 0.0
    assert score("-4", "-4") == 1.0
    assert score(r"The result is $\boxed{387}$.", "387") == 1.0
    assert score(r"Intermediate 387, final $\boxed{388}$.", "387") == 0.0
    assert score(r"\boxed{387} then corrected to \boxed{388}", "387") == 0.0
    assert score(r"\boxed{387} then corrected to \boxed{invalid}", "387") == 0.0
    assert score(r"\boxed{387} then corrected to \boxed{388", "387") == 0.0
    assert score(r"\boxed{40,943}", "40943") == 1.0
    assert score(r"\boxed{40,94}", "4094") == 0.0
    assert asyncio.run(reward(None, SimpleNamespace(response="387", label="387"))) == 1.0


def test_workload_has_unique_prompts_and_computed_labels():
    rows = dataset()
    assert len({row["prompt"][0]["content"] for row in rows}) == len(rows) == 16
    assert rows[0]["label"] == str(137 * 264 - 815)
    assert rows[-1]["label"] == str(326 * 158 - 987)
