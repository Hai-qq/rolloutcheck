"""Validate the native SGLang response fields used by the slime adapter."""

import math

from .case import CaseError
from .history import _ids


def response_ids(response, *, expected_prompt_tokens):
    if not isinstance(response, dict) or not isinstance(response.get("meta_info"), dict):
        raise CaseError("Expected a non-streaming response with meta_info")
    meta = response["meta_info"]
    pairs = meta.get("output_token_logprobs")
    if not isinstance(pairs, list) or not pairs:
        raise CaseError("Missing nonempty output_token_logprobs; do not reconstruct IDs from text")
    ids = []
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) < 2:
            raise CaseError("Malformed output logprob tuple")
        score, token = pair[:2]
        if type(score) not in (int, float) or not math.isfinite(score):
            raise CaseError("Output token logprob must be a finite number")
        ids.append(token)
    _ids(ids, "response.output_ids")
    for key, expected in (
        ("prompt_tokens", expected_prompt_tokens),
        ("completion_tokens", len(ids)),
    ):
        if type(meta.get(key)) is not int or meta[key] != expected:
            raise CaseError(f"{key} does not agree with captured token count")
    finish = meta.get("finish_reason")
    if not isinstance(finish, dict) or finish.get("type") not in ("stop", "length"):
        raise CaseError("Generation did not finish with a supported stop/length reason")
    if "output_ids" in response:
        _ids(response["output_ids"], "response.top_level_output_ids")
        if response["output_ids"] != ids:
            raise CaseError("Top-level output_ids differs from output_token_logprobs IDs")
    return ids
