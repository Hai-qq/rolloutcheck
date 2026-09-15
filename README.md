# RolloutCheck

[![CI](https://github.com/Hai-qq/rolloutcheck/actions/workflows/ci.yml/badge.svg)](https://github.com/Hai-qq/rolloutcheck/actions/workflows/ci.yml)

**Inspect history-token drift in multi-turn LLM rollouts.**

Compare the IDs a trajectory retained with the next request, identify the first
changed token, compare captured conversion boundaries, and export local evidence
that another developer can recheck.

**Early alpha · v0.1.0a9.** The offline core has zero runtime dependencies.
An optional Transformers collector records actual input/output tensors. The repo
includes both a controlled tokenizer experiment and a real Qwen3-0.6B generation
trace captured locally on MPS. An opt-in slime debug callback has also been
tested through its real HTTP adapter with a local Transformers service and a
[native SGLang/CUDA experiment](docs/native-sglang.md) on an RTX 4070 SUPER.
The HTTP capture run also verifies two interleaved sessions against that engine.
The [new training bridge](docs/training-job.md) feeds native rollout samples into
actual HF Trainer/PEFT updates, with explicit limits on what this validates.

**Independent HTTP client workflow:** run the local capture service, attach explicit
turn metadata to requests, and receive a finalized trace, diagnosis and evidence
bundle automatically. [Run it or inspect the native GPU evidence](docs/http-capture.md).
Interrupted captures cannot silently become PASS; see [completion checks](docs/trace-completion.md).

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

## Read a diagnostic

Add `--format text` to any inspect, export or verify command for a terminal view:

```sh
uv run --no-sync rolloutcheck verify-evidence \
  cases/observed/slime-http-cuda/evidence --format text
```

This rechecks the saved native capture without a GPU and deliberately exits 1.
The view shows both failed sessions, their declared parent/child turns and the
first differing token, alongside capture completion and bundle integrity:

```text
RolloutCheck: FAIL
Results: FAIL=2
Capture completion: complete; generations=4; expected=4
Bundle integrity: verified; authenticity: not established.
...
  Previous: session="public-session-1" branch="main" turn="1"
  Next:     session="public-session-1" branch="main" turn="2" declared_parent="1"
  First difference: token 32 (zero-based), previous_output; expected 151667, actual 17
```

This is an excerpt. [The complete view](docs/validation/text-report-native.txt)
includes the other session and diagnostic limits. JSON remains the default for
scripts, with the same status, report fields and exit codes. Text mode prioritizes
FAILs and limits detail to 20 transitions; counts include every transition and
omissions are explicit. Capture gaps and missing completion stay visible even
when a pair already FAILs. It never decodes token IDs or infers a missing turn ID.

## Export evidence

```sh
uv run --no-sync rolloutcheck export-evidence \
  cases/upstream/slime-qwen3/failure.case.json artifacts/evidence-bundle
```

The destination must be new. This also exits with code 1 for the intentional
failure. The bundle preserves the case bytes, report, manifest and rerun command.
For a complete captured trace, including all sessions and capture gaps:

```sh
uv run --no-sync rolloutcheck export-trace-evidence \
  cases/observed/slime-sglang-cuda/adapter.trace.jsonl artifacts/native-bundle
uv run --no-sync rolloutcheck verify-evidence artifacts/native-bundle
```

Both commands exit 1 for this valid FAIL bundle. Verification checks file hashes
and recomputes the report from the saved source; a modified report cannot silently
override that result. This is **witness_only**: saved IDs are rechecked, without
rerunning sampling or the original conversion. Nothing is executed from the
bundle or uploaded. See [bundle limits and CI usage](docs/evidence-bundles.md).

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

## Inspect a real generation trace

No model download is needed to check the committed records:

```sh
uv run --no-sync rolloutcheck inspect-trace \
  cases/observed/qwen3-transformers/rerender.trace.jsonl
uv run --no-sync rolloutcheck inspect-trace \
  cases/observed/qwen3-transformers/canonical.trace.jsonl
```

The first command exits 1: the next request loses the expected prefix at token 32.
The second exits 0: all 208 retained history tokens are preserved. Both sessions
contain actual model output, including terminal EOS, and actual second-call input.
Use `--cases-dir NEW_DIRECTORY` to extract standalone cases from a trace.

To generate new evidence locally, install the optional `generation` extra,
explicitly prepare the pinned tokenizer and 1.50 GB weights, then run
`integrations/transformers/capture_qwen.py`. See [the exact commands and measured
limits](docs/observed-case.md) and [collector API / trace format](docs/trace-capture.md).

## Collect from a slime adapter

For a ready-to-run local service and separate client, start with
[the HTTP workflow](docs/http-capture.md). For a custom controller,
use `SlimeDebugCapture` with slime's existing `debug_callback`. Supply explicit
turn/parent context from your controller and check the independently counted
served turns at shutdown. Missing context or failed capture is recorded as a gap,
so it cannot silently produce an aggregate PASS.

The actual, pinned slime HTTP adapter has passed clean/drift/missing-context
contract tests, plus a real local Qwen generation experiment. See [the integration
API, evidence and exact limits](docs/slime-integration.md). An opt-in
[`verify_sglang.py`](integrations/slime/verify_sglang.py) runner captures a native
loopback endpoint, compares wire IDs with callback records, and saves stop/length
probes. Its scripted CI test does not replace a live engine experiment.
The committed [native CUDA evidence](docs/native-sglang.md) includes six real
requests, matching wire/callback IDs, and stop-token trimming probes.

To measure the collector cost on your own idle local engine, use the
[paired on/off benchmark](docs/collector-overhead.md). It retains warmups and raw
samples, checks identical work within pairs, and supports offline recomputation.
Its scripted CI test does not establish native GPU overhead.
The [separate native run](docs/collector-overhead.md#native-rtx-4070-super-run-2026-09-14)
retains 20 measured pairs / 80 requests: callback median 0.218 ms on this short
workload, with total-duration differences unresolved amid timing variability.

## Inspect the training handoff

A drift FAIL alone does not establish corrupted training data: slime can mask or
split rewritten history while constructing training samples. The new experimental
[`audit_samples` workflow](docs/training-handoff.md) compares actual emitted Sample
positions against sampled token contexts and reports retention, ambiguity and
duplicate training occurrences.

With actual Qwen3-0.6B/MPS generation, the pinned default policy marks 16 of 192
sampled output tokens trainable; threshold zero retains 192 in two samples. Both
policies have matching trainable contexts and still show history-prefix FAIL.
This is a policy/retention observation, **not a training-quality result or universal
fix**. The full RL training loop and independent practical benefit remain unverified.

## Train and check the batch boundary

The [optional training bridge](docs/training-job.md) now consumes saved native
SGLang/slime Samples in actual Hugging Face Trainer/PEFT LoRA jobs. It preserves
loss masks through batching, verifies parameter updates and binds checkpoints to
their data/configuration. It uses masked supervised cross entropy, not slime RL.
A controlled misuse of an HF language-modeling collator is detected before the
optimizer because that collator replaces the prepared loss labels.

```sh
rolloutcheck audit-samples \
  cases/observed/slime-training-native/short/results/default/handoff.json --format text
# Explicitly require every generated output token to be represented:
rolloutcheck audit-samples \
  cases/observed/slime-training-native/short/results/default/handoff.json \
  --require-all-generated --format text
```

The second command exits 1 on this policy example. The command does not infer that
intentional response dropping is a bug. See the [native capture](cases/observed/slime-training-native)
and [training run](cases/observed/trainer-native-mps) for the measured scope.
For a first external trial, [use your own authorized trajectory](docs/try-your-trajectory.md).
There is no verified external adoption yet.

## What this adds today

Existing tools already detect drift. In particular, slime's assertion reports the
common-prefix length, and verl documents tokenization checks and mismatch logs.
RolloutCheck currently packages **explicit applicability states, structured
boundary evidence, provenance labels, and portable evidence export** around this
task. Its practical advantage over existing workflows has **not yet been validated**.

The collector currently supports one unpadded decoder-only Transformers sequence.
It does not provide full slime/SGLang integration, automatic shrinking, general
conversion execution, model replay, or automatic fixes. The included conversion
case is small; no artificial padding is used to manufacture a shrink-rate result.

## Development and feedback

```sh
uv sync --locked
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests integrations/slime/prepare_assets.py integrations/slime/reproduce.py integrations/slime/prepare_adapter.py integrations/slime/verify_adapter.py integrations/slime/capture_local_model.py integrations/slime/verify_sglang.py integrations/slime/test_live_runner.py integrations/slime/benchmark_capture.py integrations/slime/test_benchmark_runner.py integrations/slime/serve_capture.py integrations/slime/demo_client.py integrations/slime/test_http_capture.py integrations/slime/verify_training_handoff.py integrations/slime/verify_training_handoff_mps.py integrations/slime/verify_training_workload.py integrations/transformers
uv build
```

CI checks the core on Linux/Python 3.11–3.13 and Windows/Python 3.12,
tests the optional collector with CPU
tensors, exercises the real slime HTTP adapter with a scripted upstream, and
reruns the controlled conversion on Linux/Python 3.12. Full model
generation is a separate opt-in local experiment. See [the roadmap](docs/roadmap.md)
for remaining validation gates.

If you maintain a rollout adapter, feedback on where you can capture raw token IDs
and what makes your current reproduction workflow difficult is especially useful.
Use [Issues](https://github.com/Hai-qq/rolloutcheck/issues) with public or authorized
examples. Token IDs can expose original content; do not post private traces.

Original code: MIT. Upstream excerpts: Apache-2.0, with attribution retained.
