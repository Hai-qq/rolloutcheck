# Capture completion and interrupted runs

Before v0.1.0a6, a newline-terminated trace with passing transitions could return
PASS even if the collector exited before its shutdown health check. Per-record
validation detects a partially written JSON line, but cannot distinguish a
finished run from one interrupted between complete records. This is a
RolloutCheck lifecycle limitation, not a newly discovered slime or engine bug.

New `TraceRecorder` instances default to **trace version 2**. After all requests
drain, the owner must supply an independently counted number of generations:

```python
with TraceRecorder(path, trace_id=public_run_id, evidence_kind="observed_rollout") as writer:
    # Drive your workload and record its actual generations here.
    # expected_generations comes from your controller, not writer counters.
    writer.finalize(expected_generations=controller.completed_generation_count)
```

`controller` and its count are application responsibilities, not supplied APIs.
With `SlimeDebugCapture`, the existing
`capture.raise_if_failed(expected_turns=controller.served_turn_count)` performs
this finalization after verifying the callbacks. Do not also call `finalize`.
Both operations must run once, after requests drain and while the writer is open.
Closing the file or leaving its context manager does **not** declare completion.

## Observable outcomes

| Saved evidence | Aggregate result |
|---|---|
| All comparable pairs pass, valid completion | PASS / exit 0 |
| Passing pairs, omitted completion or whole-line truncation removing it | INCONCLUSIVE / exit 3 |
| A failing pair, even without completion | FAIL / exit 1; missing completion remains visible |
| Missing context or mismatched served-generation count | INCONCLUSIVE unless another pair fails |
| Invalid footer counts/digest, records after footer, partial JSON line | ERROR / exit 2 |
| Empty/root-only capture, even with completion | INCONCLUSIVE / exit 3 |

The report's `capture_completion.state` is `complete` or `missing`. Per-transition
`counts` remain unchanged, so a missing-completion report can legitimately show
`counts: {"PASS": 1}` with aggregate `status: "INCONCLUSIVE"`. Automation must use
the aggregate status or CLI exit code, not a count of passing pairs.

`complete` describes the capture lifecycle only. It does not say the history
contract passed, or remove existing gaps. A rejected count leaves a persistent
`generation_count_mismatch` gap; retrying with a different count cannot erase it.

## Footer format and limits

The last line has `record_type: "capture_complete"`, `trace_version: 2`, the same
`trace_id` and `evidence_kind`, and the next contiguous `sequence`. It also has:

- `expected_generations`: the owner's declared count.
- `generations`: the number of preceding generation records; it must equal the
  declared count when completing the file.
- `capture_gaps`: the number of preceding gap records.
- `prefix_sha256`: SHA256 of the **exact preceding bytes**, including newlines.

The reader validates all fields, counts and the digest. The footer is not counted
in report `records`, `roots` or `transitions`. It shares the existing 64 MiB trace
limit. The writer refuses later records, a second finalization or recovery after
an I/O failure. It flushes records, but does not promise power-loss durability or
provide a resume/append protocol. Keep failed outputs and start a new file.

This is a local consistency check. An incorrect owner count, deliberately
re-authored file, undeclared request, wrong parent identity, or requests made
after the declared end cannot be authenticated by a footer. Drain and count the
actual workload; do not use a footer to claim complete engine or training coverage.

## Compatibility and evidence handoff

Existing v1 traces and v0.1.0a5 bundles retain their exact reports. They have no
completion attestation; a historical PASS checks only the saved transitions.
Do not rewrite old captures as v2 or manufacture retrospective completion.
An explicit `trace_version=1` recorder is available for legacy interchange, with
those limitations. Readers before v0.1.0a6 reject v2 rather than silently ignore
its footer.

When upgrading a custom collector, add the one finalization call. Slime users
already calling `raise_if_failed` at shutdown need no extra call. Scripts reading
every JSONL line as a generation must filter on `record_type == "generation"`.
The included Transformers, local slime and native SGLang runners are migrated.
Prior native GPU captures remain v1; this update does not relabel them or claim a
new native engine measurement.

`export-trace-evidence` preserves completion or its absence in the saved bytes;
`verify-evidence` recomputes the same result. A bundle with verified integrity can
still be INCONCLUSIVE or FAIL. Extracted single-transition cases do not represent
whole-run completion; share the whole bundle when that state matters.

## Reproduce the regression checks

```sh
uv sync --locked
uv run --no-sync pytest tests/test_trace_completion.py tests/test_slime_capture.py -q
```

The regression starts a child process, writes the same two passing generations
under v1 and v2, and exits with `os._exit` to skip cleanup. The legacy trace reports
PASS; v2 reports INCONCLUSIVE. Other tests remove a whole footer line, mutate its
counts/digest, inject a persistence failure, and recheck relocated evidence bundles.
These are controlled fault injections, not observations of a production crash.

The [slime integration runner](slime-integration.md#reproduce-the-http-contract-test)
also drives the pinned unmodified adapter through real HTTP for six scenarios:
clean, drift, missing context, skipped health check with/without drift, and a served
count mismatch. Model responses and tokenization are scripted. CI retains its
artifacts and tests the lifecycle regressions on Linux and Windows.

The [Windows receipt](validation/lifecycle-windows.json) records an additional
installed-wheel check on 2026-09-14. A fresh native Python 3.11.9 environment
installed v0.1.0a6 with `--no-index --no-deps` and verified relocated bundles
exported on macOS: completed/PASS, unfinished/INCONCLUSIVE,
unfinished-drift/FAIL, and the existing v1 native control/PASS. A Windows child
process also exited abruptly after two passing generations; the installed checker
returned INCONCLUSIVE with missing completion. This operator-run test uses
synthetic lifecycle inputs and saved legacy evidence, without a model or GPU;
it is not independent adoption.
