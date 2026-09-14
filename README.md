# RolloutCheck

[![CI](https://github.com/Hai-qq/rolloutcheck/actions/workflows/ci.yml/badge.svg)](https://github.com/Hai-qq/rolloutcheck/actions/workflows/ci.yml)

**Inspect history-token drift in multi-turn LLM rollouts.**

Compare the IDs a trajectory retained with the next request, identify the first
changed token, compare captured conversion boundaries, and export local evidence
that another developer can recheck.

**Early prototype · v0.1.0a1.** The core runs on CPU with zero runtime dependencies.
A pinned Qwen/slime conversion experiment is included. It uses a real tokenizer
with hand-authored response content; it is not a model sampling or training run.

## Why

In an explicitly append-only training path, replaying a conversation through a
chat template can change previously retained assistant tokens. The sentence may
look equivalent while the token sequence is different. RolloutCheck checks:

```python
history = previous_input_ids + previous_output_ids
next_input_ids[:len(history)] == history
```

This is **not a universal rule for conversations**. History rewriting, legitimate
truncation, and different branches/token spaces require different semantics.
Missing evidence is inconclusive, not a successful check.

## Try the offline checker

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/Hai-qq/rolloutcheck.git
cd rolloutcheck
uv sync --locked --no-dev
uv run --no-sync rolloutcheck inspect cases/synthetic/pass.json
uv run --no-sync rolloutcheck inspect cases/upstream/slime-qwen3/failure.case.json
```

The last command deliberately exits with code **1** and reports:

```json
{
  "status": "FAIL",
  "first_difference": {
    "index": 27,
    "region": "previous_output",
    "expected_id": 151667,
    "actual_id": 19
  }
}
```

This is an excerpt of the report. Indices are zero-based. In the included
controlled case, token `151667` is `<think>` and token `19` begins the visible
answer. The failing path drops historical reasoning during re-rendering.

| Exit code | Status | Meaning |
|---|---|---|
| 0 | PASS | The declared history prefix is preserved |
| 1 | FAIL | A comparable prefix violates the contract |
| 2 | ERROR | Malformed data, unsupported schema, or an I/O error |
| 3 | INCONCLUSIVE | Required contract or continuity evidence is missing |
| 4 | NOT_APPLICABLE | This pair is outside the supported contract |

PASS does not establish correct log probabilities, masks, rewards, or training.
See [the case format and capture guidance](docs/case-format.md).

## Export evidence

```sh
uv run --no-sync rolloutcheck export-evidence \
  cases/upstream/slime-qwen3/failure.case.json artifacts/evidence-bundle
```

The destination must be new. This also exits with code 1 for the intentional
failure. The bundle preserves the case bytes, report, digest, and rerun command.
It is explicitly **witness_only**: rechecking saved IDs does not rerun the original
conversion. No input-supplied code is executed and no evidence is uploaded.

## Reproduce the upstream conversion

This is a separate, reviewed Python runner in the source repository:

```sh
uv sync --locked --extra qwen
# Explicit network preparation: ~11.5 MB of tokenizer/config/license assets; no weights.
uv run --no-sync python integrations/slime/prepare_assets.py .cache/qwen3-0.6b
# The transformation itself loads only verified local assets.
uv run --no-sync python integrations/slime/reproduce.py \
  --assets .cache/qwen3-0.6b --output artifacts/qwen3-reproduction
```

The runner executes unmodified function excerpts from slime commit
`885a09e852e5272d497936370707cc604830f805` and the pinned Qwen3-0.6B tokenizer.
It checks both failure and the upstream fix, writes cases/reports, and returns 0
only when the declared fail/pass contrast succeeds. See the
[measured case and limits](docs/first-case.md) and [third-party attribution](NOTICE.md).

## What this adds today

Existing tools already detect drift. In particular, slime's assertion reports the
common-prefix length, and verl documents tokenization checks and mismatch logs.
RolloutCheck currently packages **explicit applicability states, structured
boundary evidence, provenance labels, and portable evidence export** around this
task. Its practical advantage over existing workflows has **not yet been validated**.

It does not currently provide live framework hooks, automatic shrinking, general
conversion execution, model replay, or automatic fixes. The included conversion
case is small; no artificial padding is used to manufacture a shrink-rate result.

## Development and feedback

```sh
uv sync --locked
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests integrations/slime/prepare_assets.py integrations/slime/reproduce.py
uv build
```

CI checks the core on Python 3.11–3.13 and reruns the controlled conversion on
Linux/Python 3.12. See [the roadmap](docs/roadmap.md) for remaining validation gates.

If you maintain a rollout adapter, feedback on where you can capture raw token IDs
and what makes your current reproduction workflow difficult is especially useful.
Use [Issues](https://github.com/Hai-qq/rolloutcheck/issues) with public or authorized
examples. Token IDs can expose original content; do not post private traces.

Original code: MIT. Upstream excerpts: Apache-2.0, with attribution retained.
