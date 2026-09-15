"""Offline accounting of emitted training-token contexts against sampled contexts.

This checks token/mask placement only, not log probabilities, rewards, optimizer
behavior, policy versions, or the suitability of a framework's retention policy.
"""

from collections import Counter, defaultdict

from .case import CaseError
from .history import _ids

MAX_TOTAL_TOKENS = 131072


def audit_samples(turns, samples):
    """Audit one explicitly scoped session's captured turns and emitted samples.

    turns: unique public turn_id, input_ids, output_ids.
    samples: tokens, response_length, response-region binary loss_mask.
    Full prefixes are interned in a trie: equality is exact, never token-only or
    digest-based. Identical sampled contexts are reported as ambiguous instead
    of assigning a turn by order. Unaccounted output may be intentionally dropped.
    """
    if not isinstance(turns, list) or not isinstance(samples, list):
        raise CaseError("turns and samples must be lists")
    if len(turns) + len(samples) > 4096:
        raise CaseError("Too many turns or samples for this bounded audit")
    total, names = 0, set()
    for turn in turns:
        if not isinstance(turn, dict):
            raise CaseError("Each turn must be an object")
        name = turn.get("turn_id")
        if not isinstance(name, str) or not 1 <= len(name) <= 128 or name in names:
            raise CaseError("Each turn needs a unique public turn_id, at most 128 characters")
        names.add(name)
        for field in ("input_ids", "output_ids"):
            ids = _ids(turn.get(field), field)
            if not ids:
                raise CaseError("Captured prompt and output IDs must be nonempty")
            total += len(ids)
    for sample in samples:
        if not isinstance(sample, dict):
            raise CaseError("Each sample must be an object")
        tokens = _ids(sample.get("tokens"), "tokens")
        total += len(tokens)
        length, mask = sample.get("response_length"), sample.get("loss_mask")
        if type(length) is not int or not 0 <= length <= len(tokens):
            raise CaseError("response_length must fit inside tokens")
        if (
            not isinstance(mask, list)
            or len(mask) != length
            or any(type(bit) is not int or bit not in (0, 1) for bit in mask)
        ):
            raise CaseError("loss_mask must be explicit binary integers for the response region")
    if total > MAX_TOTAL_TOKENS:
        raise CaseError(f"Audit exceeds the {MAX_TOTAL_TOKENS}-token input limit")

    edges, sources = {}, defaultdict(list)

    def advance(node, token):
        key = (node, token)
        if key not in edges:
            edges[key] = len(edges) + 1
        return edges[key]

    for turn in turns:
        node = 0
        for token in turn["input_ids"]:
            node = advance(node, token)
        for index, token in enumerate(turn["output_ids"]):
            sources[node, token].append((turn["turn_id"], index))
            node = advance(node, token)

    matched = Counter()
    trained = unmatched = ambiguous = 0
    sample_counts = []
    for number, sample in enumerate(samples):
        node = count = 0
        start = len(sample["tokens"]) - sample["response_length"]
        for index, token in enumerate(sample["tokens"]):
            if index >= start and sample["loss_mask"][index - start]:
                count += 1
                candidates = sources.get((node, token), ())
                if len(candidates) == 1:
                    matched[candidates[0]] += 1
                elif candidates:
                    ambiguous += 1
                else:
                    unmatched += 1
            node = advance(node, token)
        trained += count
        sample_counts.append(
            {"sample_index": number, "tokens": len(sample["tokens"]), "trainable_tokens": count}
        )
    per_turn = []
    for turn in turns:
        represented = sum((turn["turn_id"], i) in matched for i in range(len(turn["output_ids"])))
        per_turn.append(
            {
                "turn_id": turn["turn_id"],
                "generated_tokens": len(turn["output_ids"]),
                "uniquely_represented_tokens": represented,
                "unaccounted_tokens": len(turn["output_ids"]) - represented,
            }
        )
    duplicates = sum(count - 1 for count in matched.values())
    return {
        "audit_version": 1,
        "context_status": (
            "UNMATCHED"
            if unmatched
            else "AMBIGUOUS"
            if ambiguous
            else "NO_TRAINABLE_TOKENS"
            if not trained
            else "MATCHED"
        ),
        "generated_tokens": sum(len(t["output_ids"]) for t in turns),
        "trainable_tokens": trained,
        "unmatched_trainable_tokens": unmatched,
        "ambiguous_trainable_tokens": ambiguous,
        "duplicate_training_occurrences": duplicates,
        "unaccounted_generated_tokens": sum(t["unaccounted_tokens"] for t in per_turn),
        "turns": per_turn,
        "samples": sample_counts,
        "scope": "Exact token and preceding-context accounting within supplied session only. "
        "MATCHED does not imply all generated tokens were retained, no duplicates, correct "
        "log probabilities/rewards, training benefit, or complete/authentic source capture. "
        "Unaccounted tokens can be intentional policy decisions or ambiguous attribution.",
    }
