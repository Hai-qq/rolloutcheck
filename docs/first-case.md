# First controlled conversion case

Date: 2026-09-14. Recorded output: [summary.json](../cases/upstream/slime-qwen3/summary.json).

## What ran

- Real Qwen3-0.6B tokenizer, asset revision
  `c1899de289a04d12100db370d81485cdf75e47ca`, verified by SHA-256.
- Unmodified helper-function excerpt from slime PR #2287 commit
  `885a09e852e5272d497936370707cc604830f805`.
- Python 3.12.12, transformers 4.57.6, macOS arm64, CPU.
- A hand-authored two-turn conversation and frozen reasoning response. The
  authored response includes `<think>` and its closing marker; a sanity check
  confirms that it matches serialization before adding the new user message.

The original history is assembled independently from the previous prompt and
frozen response IDs. The failing next input is obtained by calling the upstream
renderer on the replayed structured history. The passing control uses upstream's
reasoning-preserving template and canonical-continuation function. RolloutCheck
does not generate its expected prefix by re-rendering the modified history.

## Recorded result

| Measurement | Result |
|---|---|
| Previous input | 27 tokens |
| Hand-authored response suffix | 14 tokens |
| Canonical history | 41 tokens |
| Failing next input | 43 tokens |
| Passing next input | 53 tokens |
| First mismatch | index 27, previous output |
| Expected / actual ID | 151667 / 19 |
| Failure / control check | FAIL / PASS |

The next-input lengths include the new observation and generation prompt.
Their difference is not a measurement of GPU memory, throughput, or training loss.

## Fair comparison

slime's **existing** `_assert_append_only_prompt` catches the same failure:

```text
non-append-only agent prompt: previous_tokens=41 prompt_tokens=43 common_prefix=27
```

It also accepts the passing control. RolloutCheck's additional output is structured
status, a token window, evidence kind, declared boundary interval, and an exportable
case. This small experiment does **not** establish a reduction in human debugging
time or show that an additional standalone tool is needed. There is only one
captured failing conversion boundary here; controlled multi-boundary unit tests
test the reporting behavior but are not additional real bugs.

## Evidence and limitations

- `transform_reproduced` describes executing the audited helper excerpts.
- Input provenance is `controlled_upstream_transform`, never `observed_rollout`.
- No full slime adapter/server, SGLang backend, Claude Code interaction, model
  weights, original author logs, model sampling, or RL training ran.
- The motivating issue concerns Qwen3.5 and a specific adapter. This case uses
  Qwen3-0.6B and validates the related template mechanism, not exact compatibility
  with that whole original setup.
- The fix and motivating report belong to upstream author `zy20031230`.
- There is no external-user adoption evidence yet.
- The experiment needs no GPU. Scaling to a training server is not required by
  this result and would not, by itself, establish practical tool value.

Run the commands in [README](../README.md#reproduce-the-upstream-conversion).
Saved case files can be inspected without installing transformers or fetching assets.

Sources: [issue #2288](https://github.com/THUDM/slime/issues/2288),
[PR #2287](https://github.com/THUDM/slime/pull/2287),
[pinned source](https://github.com/THUDM/slime/blob/885a09e852e5272d497936370707cc604830f805/slime/agent/adapters/common.py),
[verl's existing checks](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html).
