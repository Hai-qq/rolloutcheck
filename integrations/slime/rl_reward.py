"""Deterministic arithmetic reward for the opt-in full-framework RL experiment.

No model, network service, random reward, or training-algorithm replacement is
used here. slime calls ``reward`` through its supported custom reward interface.
"""

from __future__ import annotations

import re


def score(response: str, label: str) -> float:
    """Check the final boxed integer, or an answer consisting only of an integer."""
    answer = response.strip()
    boxed = list(re.finditer(r"\\boxed\s*\{", answer))
    if boxed:
        body, closing, _ = answer[boxed[-1].end():].partition("}")
        if not closing:
            return 0.0
        answer = body.strip()
    if re.fullmatch(r"-?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)", answer) is None:
        return 0.0
    return float(int(answer.replace(",", "")) == int(label))


async def reward(args, sample, **kwargs) -> float:
    return score(sample.response, sample.label)


def dataset(*, evaluation: bool = False) -> list[dict]:
    """Small public, generated workload; not a held-out evaluation benchmark."""
    rows = []
    triples = [
        (137, 264, 815), (218, 379, 546), (93, 447, 628), (126, 358, 927),
        (147, 293, 746), (267, 134, 585), (143, 256, 719), (372, 218, 646),
        (123, 287, 829), (246, 135, 798), (157, 368, 897), (284, 129, 766),
        (237, 164, 695), (318, 279, 864), (193, 347, 758), (326, 158, 987),
    ]
    if evaluation:
        triples = [(179, 246, 835), (328, 157, 964), (213, 389, 647), (296, 183, 758)]
    for a, b, c in triples:
        expression = f"({a} * {b}) - {c}"
        rows.append({
            "prompt": [{
                "role": "user",
                "content": f"Calculate {expression}. "
                           "Put your final integer answer inside \\boxed{}.",
            }],
            "label": str(a * b - c),
        })
    return rows


if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", action="store_true")
    args = parser.parse_args()
    for row in dataset(evaluation=args.evaluation):
        sys.stdout.write(json.dumps(row, ensure_ascii=False) + "\n")
