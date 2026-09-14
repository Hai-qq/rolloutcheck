import copy

import pytest

from rolloutcheck.case import CaseError
from rolloutcheck.sglang_response import response_ids

RESPONSE = {
    "text": "",
    "meta_info": {
        "prompt_tokens": 2,
        "completion_tokens": 1,
        "output_token_logprobs": [[-0.1, 9, None]],
        "finish_reason": {"type": "stop", "matched": 9},
    },
}


def test_trimmed_text_is_not_used_to_reconstruct_token_ids():
    assert response_ids(RESPONSE, expected_prompt_tokens=2) == [9]


@pytest.mark.parametrize(
    "change",
    [
        {"completion_tokens": 0},
        {"prompt_tokens": True},
        {"prompt_tokens": 3},
        {"output_token_logprobs": []},
        {"output_token_logprobs": [[None, 9]]},
        {"output_token_logprobs": [[float("nan"), 9]]},
        {"output_token_logprobs": [[-1, True]]},
        {"finish_reason": {"type": "abort"}},
    ],
)
def test_incomplete_or_misaligned_wire_evidence_is_rejected(change):
    value = copy.deepcopy(RESPONSE)
    value["meta_info"].update(change)
    with pytest.raises(CaseError):
        response_ids(value, expected_prompt_tokens=2)


def test_conflicting_top_level_ids_are_rejected():
    with pytest.raises(CaseError):
        response_ids(dict(RESPONSE, output_ids=[10]), expected_prompt_tokens=2)
