# Case format v1

This is an experimental internal interchange format, not a universal trace standard.
A case file contains one JSON object. Multi-turn JSONL generation records use
the separate [trace format and collector](trace-capture.md).
See the four complete examples in [cases/synthetic](../cases/synthetic).

## Required evidence

- `schema_version`: integer `1`.
- `case_id`: a nonempty string.
- `evidence.kind`: `synthetic`, `controlled_upstream_transform`, or `observed_rollout`.
  These are caller declarations, not authenticated certifications.
- `contract`: `version: history-prefix/v1`, `mode: append_only`,
  `history_policy: preserved` for this check to apply.
- `previous`: `session_id`, `branch_id`, `turn_id`, `token_space`, `input_ids`, `output_ids`.
- `next`: `session_id`, `branch_id`, `previous_turn_id`, `token_space`, `input_ids`.

IDs must be non-negative JSON integers; booleans are not token IDs. Identity fields
must be nonempty strings. `token_space` identifies the tokenizer vocabulary and
encoding rules (prefer an asset digest), not the chat template. Template changes
may be the bug under investigation; templates are recorded separately in provenance.
IDs from different token spaces must not be compared, even if their numbers match.

`next.previous_turn_id` must equal `previous.turn_id`. Do not infer adjacency from
timestamps alone. Session and branch IDs must agree. A pair cannot establish that
the declared identifiers were collected correctly; capture them at the actual boundary.

`contract.mode: history_rewrite` or `history_policy: truncated` means this specific
check is not applicable. Unknown/missing retention or contract evidence means
INCONCLUSIVE. Different sessions, branches, or token spaces are NOT_APPLICABLE.
Malformed supplied fields produce ERROR before applicability is evaluated.

## Capturing a case

Copy raw request and generated token IDs at the engine/adapter boundary **before**
text decoding, response normalization, or subsequent template application. The
upstream conversion experiment demonstrates direct collection in
[reproduce.py](../integrations/slime/reproduce.py). It is not a production collector.

Record whether stop/EOS IDs are returned, stripped, or appended, and whether the
backend rewrites/truncates the submitted input. Do not mark reconstruction from
decoded text as actual sampled IDs. `observed_rollout` is appropriate only when
all three token arrays come from a real sampling path. A trace with missing raw
IDs cannot be upgraded by rendering its messages later.

The optional [Transformers collector](trace-capture.md) records raw generation
tensors and extracts cases from JSONL. It does not intercept live SGLang traffic
or automatically install hooks in slime.

## Optional boundaries

`boundaries` is an array in actual execution order; each item contains a unique
`name`, `origin` (`captured` or `derived`), `token_space`, and `input_ids`.
These IDs must represent the full prefix beginning at the same history origin as
`previous.input_ids`. Arbitrary message token offsets and suffix-only arrays are
not comparable coordinates. Do not pass them here.

Each comparable boundary is checked against the independently retained original
history. Derived projections are displayed but cannot narrow the observed interval.
An earlier transient mismatch need not cause the final mismatch; the report does
not assert causality or monotonic propagation. The first differing index does not
prove that every later token was changed, deleted, or lost from training.

## Output, limits and data handling

`PASS` establishes only the declared prefix relationship. It does not validate
log probabilities, loss masks, rewards, context meaning, or whole-training correctness.
`FAIL` includes the first difference and a bounded ID window, without decoding IDs.
An empty canonical history is INCONCLUSIVE, rather than a vacuous PASS.

Input is limited to 16 MiB. Duplicate JSON keys and non-finite values are rejected.
The core does not execute Python, shell, pickle, remote code, or network requests.
Case loading and reporting are local. Token IDs and evidence bundles can still
contain recoverable sensitive content: share only data you are authorized to disclose.

`export-evidence` preserves the case bytes and labels the result `witness_only`.
It rechecks saved evidence; it does not execute the conversion that produced it.
Existing destination directories are refused. Its exit code reflects the check
status, so a successfully exported FAIL still exits with code 1.
Since v0.1.0a5, exports include a versioned manifest and support
`verify-evidence DIRECTORY`, which checks hashes and recomputes the saved report.
See [evidence bundles](evidence-bundles.md) for trace exports and CI usage.
