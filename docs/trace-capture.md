# Raw-token trace capture

A trace is a local UTF-8 JSONL file. Each line is a completed generation or an
explicit `capture_gap` event (supported since v0.1.0a3). New recorders use v2,
which also requires a final `capture_complete` record (since v0.1.0a6).
It lets the checker extract parent/child transitions without manually building
case objects. This is an experimental format, not a universal trace standard.

## Inspect without a model

```sh
uv sync --locked --no-dev
uv run --no-sync rolloutcheck inspect-trace \
  cases/observed/qwen3-transformers/rerender.trace.jsonl \
  --cases-dir artifacts/extracted-rerender
uv run --no-sync rolloutcheck inspect artifacts/extracted-rerender/transition-000001.json
```

Add `--format text` to see failed sessions, declared parent/child IDs, token
differences and capture completion directly. The default JSON reports and
`--cases-dir` file contents remain unchanged by this display option. Text output
limits details to 20 transitions and reports omissions; JSON retains all details.

Both checks intentionally exit 1. A PASS trace exits 0; the other exit codes match
single-case inspection. The export directory must be new. It contains standalone
cases and the aggregate report, labeled `witness_only` in the command output.
It does not run conversions or model generation. Keep the original JSONL if the
recipient needs the complete trace; its SHA-256 is included in every extracted case.

## Opt-in Transformers adapter

Install the `generation` extra in a project-local environment. The adapter accepts
an already-loaded, ordinary Transformers decoder-only model and an unpadded
single-sequence integer tensor. It does not download a model or install hooks.

```python
from transformers import GenerationConfig
from rolloutcheck.trace import TraceRecorder
from rolloutcheck.transformers_capture import generate_recorded

# model, tokenizer and input_ids come from your own verified generation path.
# Use stable session/branch/turn identity and a tokenizer asset digest.
with TraceRecorder("run.jsonl", trace_id="run-001", evidence_kind="observed_rollout") as trace:
    output_ids, timing = generate_recorded(
        model, input_ids,
        recorder=trace,
        session_id="conversation-001", branch_id="main",
        turn_id="1", parent_turn_id=None,
        token_space="your-tokenizer-asset-digest",
        contract={"version": "history-prefix/v1", "mode": "append_only",
                  "history_policy": "preserved"},
        generation_config=GenerationConfig(
            max_new_tokens=256, do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        ),
        metadata={"model_revision": "your-pinned-revision"},
    )
    # Call again with the next actual input and parent_turn_id="1" to check a transition.
    # This example makes one call. After all your requests drain, use the owner's
    # independent expected count (2 if you add the second call above).
    trace.finalize(expected_generations=1)
```

The adapter snapshots the input tensor before `model.generate`, checks that the
returned sequence still begins with that exact input, and slices the generated
suffix without decoding or tokenizing it. Final EOS stays in `output_ids`.
All-ones attention masks are explicit. `use_model_defaults=False` prevents
model-specific beam/sampling defaults from replacing the checked configuration. EOS and PAD IDs must be configured; when
PAD aliases EOS, historical EOS is treated as content. The caller must supply
an unpadded input even in that case, because aliased IDs cannot distinguish
actual padding from content.

Supported: one decoder-only sequence, greedy or ordinary sampling, explicit
`max_new_tokens`, a plain output tensor, and no output padding. Unsupported modes
fail before generation: batching, beam search, multiple returned sequences,
token healing, stop strings, contrastive search, forced boundary tokens, and rich
output dictionaries/scores. `max_time` may produce `reason: other`; length-limited
output is recorded as such, never silently relabeled a completed response.
The wrapper does not validate an entire framework or any custom `generate` implementation.

`capture_seconds` includes host copies, record validation and JSONL flush.
`generation_seconds` includes the generate call and device synchronization.
These are per-call measurements, not a before/after overhead benchmark. There is
no multi-process writer, process-crash durability guarantee, or resume/append mode.

For another engine, `TraceRecorder.record(...)` accepts explicitly captured Python
ID lists with the same identity, contract, termination and metadata fields. This
low-level API trusts the caller's capture point and EOS accounting. Do not label
re-tokenized text as observed model output. Importing either module does not load
PyTorch; only invoking `generate_recorded` does.

## Records and transition semantics

Each generation record has:

- `trace_version: 2` (or legacy `1`), `record_type: generation`, a nonempty `trace_id` and
  contiguous integer `sequence` starting at zero.
- `evidence_kind`: `synthetic`, `controlled_upstream_transform` or `observed_rollout`.
  Version, identity and evidence kind must remain consistent within a file. These labels
  are declarations, not authenticated provenance.
- Nonempty `session_id`, `branch_id`, `turn_id`, `token_space`; an explicit
  `parent_turn_id` string, or `null` for a root.
- `input_ids`, nonempty `output_ids`, and object-valued `contract`, `termination`,
  `metadata`; optional `boundaries` use [the case-format rules](case-format.md).

Use the contract actually intended for the current transition. Do not declare
`append_only/preserved` for a path that intentionally rewrites or truncates history.
Termination metadata is descriptive; the checker does not independently verify
whether an engine stripped stop tokens. Prefix comparison retains every supplied ID.

A parent must be an earlier record with matching session and branch. No temporal
adjacency or cross-branch ancestry is inferred. Missing parents produce
INCONCLUSIVE transitions. Duplicate turn identities are errors. Roots are counted
but not compared; an empty or root-only trace is INCONCLUSIVE. This checks declared
links, not whether all real requests were captured or all roots were labeled correctly.

A `capture_gap` record contains the file's `trace_version`, `record_type: capture_gap`,
`trace_id`, `sequence`, `evidence_kind` and a nonempty `reason`; it has no token
arrays or inferred ancestry. `TraceRecorder.record_gap(reason)` writes one.
Gaps share the contiguous sequence counter and are listed in report `capture_gaps`.
`records` counts generation and gap lines (excluding the v2 completion footer),
while `transitions` and `counts` describe generation
pairs only. Any gap prevents aggregate PASS, even if all captured pairs pass.
An earlier reader that does not understand gap events returns ERROR.

Without capture gaps, aggregate status uses FAIL, INCONCLUSIVE, NOT_APPLICABLE,
then PASS precedence, with all per-transition status counts retained. With gaps,
the aggregate is FAIL if any pair fails, otherwise INCONCLUSIVE. A failing pair remains actionable even if
another pair lacks evidence. Malformed data anywhere makes the whole read ERROR.

For v2, missing completion also makes an otherwise non-failing trace INCONCLUSIVE.
`close()` and context-manager exit only close the file. Call
`finalize(expected_generations=...)` once after requests drain, with the owner's
independent generation count; do not derive it from recorder counters. It seals
the original record bytes and prevents further recording. A count mismatch
persists a gap and raises. Completion never removes a gap or makes a root-only
trace PASS. See [the exact footer format and migration](trace-completion.md).

Legacy v1 traces retain their historical reports, with no completion attestation.
Their PASS applies only to the saved prefix comparisons. Do not relabel them v2
or attach a footer retrospectively to imply an old run was checked at shutdown.

Limits: 64 MiB per trace, 16 MiB per record. Duplicate keys, non-finite values,
blank lines, and a missing final newline are rejected. The writer creates a new
file exclusively with owner-only permissions and flushes each completed record.
It refuses to append to or overwrite existing evidence.

Token IDs can reconstruct private content. Collection is opt-in, stays local,
and has no telemetry. Review traces and arbitrary caller metadata before sharing.
