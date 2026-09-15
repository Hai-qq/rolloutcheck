"""Explicit, unpacked causal-LM batch boundary; no tensor/framework dependency.

This bridge uses masked next-token cross entropy, not a slime RL objective.
Labels stay unshifted: Hugging Face causal-LM models shift them internally.
"""

from .case import CaseError
from .sample_audit import audit_samples


def training_features(turns, samples, *, max_length=4096):
    """Reject unmatched/ambiguous/duplicated supervision, preserve explicit masks.

    Intentionally unretained generation is allowed and remains in the audit.
    No truncation, packing, retokenization or automatic mask repair is performed.
    """
    if type(max_length) is not int or not 2 <= max_length <= 131072:
        raise CaseError("max_length must be an integer from 2 to 131072")
    audit = audit_samples(turns, samples)
    if audit["context_status"] != "MATCHED" or audit["duplicate_training_occurrences"]:
        raise CaseError("Training bridge requires unique, matched supervision without duplicates")
    features = []
    for sample in samples:
        tokens = sample["tokens"]
        start = len(tokens) - sample["response_length"]
        if not 1 <= start < len(tokens) <= max_length:
            raise CaseError("Each sample needs prompt and response tokens within max_length")
        if not any(sample["loss_mask"]):
            raise CaseError("Each sample needs at least one supervised response token")
        labels = [-100] * start + [
            token if bit else -100
            for token, bit in zip(tokens[start:], sample["loss_mask"], strict=True)
        ]
        features.append({"input_ids": list(tokens), "labels": labels})
    return features, audit


def pad_features(features, *, pad_token_id):
    """Right-pad already prepared features, preserving EOS labels even when EOS=PAD."""
    if type(pad_token_id) is not int or pad_token_id < 0 or not features:
        raise CaseError("A nonempty batch and a nonnegative integer pad token are required")
    width = max(len(row["input_ids"]) for row in features)
    result = {name: [] for name in ("input_ids", "attention_mask", "labels")}
    for row in features:
        ids, labels = row["input_ids"], row["labels"]
        if (
            not ids
            or len(ids) != len(labels)
            or any(type(x) is not int or x < 0 for x in ids)
            or any(
                type(y) is not int or y not in (-100, x) for x, y in zip(ids, labels, strict=True)
            )
            or labels[0] != -100
            or not any(y != -100 for y in labels)
        ):
            raise CaseError("Invalid unshifted causal-LM feature")
        padding = width - len(ids)
        result["input_ids"].append(ids + [pad_token_id] * padding)
        result["attention_mask"].append([1] * len(ids) + [0] * padding)
        result["labels"].append(labels + [-100] * padding)
    return result


def audit_batch(features, batch, *, pad_token_id):
    """Check the actual unpacked/right-padded batch against the prepared features.

    Supply CPU lists (e.g. tensor.tolist()). This intentionally rejects alternate
    packing/padding policies instead of claiming support for their semantics.
    """
    expected = pad_features(features, pad_token_id=pad_token_id)
    mismatches = [name for name in expected if batch.get(name) != expected[name]]
    return {
        "batch_audit_version": 1,
        "status": "MISMATCH" if mismatches else "MATCHED",
        "mismatched_fields": mismatches,
        "expected_supervised_tokens": sum(
            label != -100 for row in expected["labels"] for label in row
        ),
        "scope": "Exact unpacked, right-padded token/attention/label equality only; "
        "does not establish RL loss, rewards, policy version or training benefit.",
    }
