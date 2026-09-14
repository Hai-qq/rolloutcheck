# Portable evidence for a saved rollout

`export-trace-evidence` packages a complete raw trace and its checked report in
one command. A reviewer can run `verify-evidence` without the original checkout,
model, GPU, tokenizer, training framework or network access. RolloutCheck itself
must be installed. The package has no runtime dependencies.

```sh
rolloutcheck export-trace-evidence rollout.jsonl bundle
rolloutcheck verify-evidence bundle
```

The exporter reads one bounded snapshot, checks that snapshot and preserves those
exact bytes. It does not read a later version of a growing trace for the copy.
Finish/drain the collector before exporting: an incomplete final JSONL record is
an error, and a snapshot cannot include future records.
For new v2 traces, finalize the recorder before exporting. An absent completion
footer prevents aggregate PASS even if the last saved line is complete. The bundle
preserves and rechecks that state; export never adds a footer on the collector's
behalf. Legacy v1 PASS does not attest completion. See [capture lifecycle](trace-completion.md).

## Files and results

A trace bundle contains `trace.jsonl`, `report.json`, `README.md` and
`manifest.json`. A single-case bundle produced by `export-evidence` contains
`case.json` instead of `trace.jsonl`. The manifest records the bundle version,
producer version, fixed filenames, byte lengths and SHA256 hashes.

`verify-evidence` checks those files and recomputes the report. It checks the
report itself, not only a digest supplied beside it. A consistent bundle returns
the source's check status and adds:

```json
{"verification": {"integrity": "verified", "kind": "witness_only", "authenticity": "not_established"}}
```

Exit codes remain PASS `0`, FAIL `1`, ERROR `2`, INCONCLUSIVE `3`, and
NOT_APPLICABLE `4`. A valid FAIL bundle exits **1**, not 0. Missing/changed files,
invalid manifests, or a saved report that differs from the recomputed report
exit **2**. Capture gaps and no-transition traces remain inconclusive unless
another transition already establishes FAIL. Verification does not remove gaps.

This is consistency verification, not proof of authenticity: someone who changes
the source and generates a corresponding report can create a different valid
bundle. Evidence labels remain caller-supplied. Use the producer's recorded
RolloutCheck version when exact historical report compatibility is required.
Pre-v0.1.0a5 exports have no manifest; re-export their source to use this command.

## Data and filesystem boundaries

- The whole source trace is included, even other sessions and capture gaps.
  Review it before sharing; token IDs can reconstruct text. There is no automatic
  redaction, upload or sampling replay.
- Source limits remain 16 MiB per case / 64 MiB per trace. The report is limited
  to 16 MiB, and the manifest and README to 64 KiB each. A too-large report fails
  before creating the destination. Verification applies the same read limits.
- Manifest filenames are fixed; they cannot select arbitrary relative/absolute
  paths. Payload symlinks and non-regular files are rejected. Additional directory
  contents are ignored, never executed or treated as verified bundle content.
- Existing destination directories are refused. New bundle files use owner-only
  permissions on POSIX. Windows access remains subject to its directory ACLs.
- The manifest is written last. A disk error may leave a partial directory;
  without complete matching files and a valid manifest it cannot verify. Preserve
  or remove that partial output deliberately, then export to a new directory.

## Keep the failure visible in CI

After your job has produced and closed `artifacts/rollout.jsonl`, use a step like
this. It requires an installed RolloutCheck command in the job's PATH:

```yaml
- name: Check and package the captured trace
  run: rolloutcheck export-trace-evidence artifacts/rollout.jsonl artifacts/evidence
- name: Retain local evidence even when the check fails
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: rollout-evidence
    path: artifacts/evidence/
```

A failed contract fails the first step; `if: always()` lets the evidence be
retained for review. Do not turn all nonzero exits into success with `|| true`.
An export error can leave partial output; reviewers must run `verify-evidence`
and check its result. Upload only trajectories authorized for the artifact's
audience. GitHub artifact access/retention is separate from this local checker.

## Current validation

Regression tests export and verify the committed native SGLang adapter failure
and its constructed positive control. They also cover capture gaps, false reports
with recomputed hashes, changed payloads, incomplete traces, partial writes and
source changes after the inspected snapshot. CI checks the offline core on Linux
and Windows; it does not run a GPU generation for bundle verification.

The [2026-09-14 handoff receipt](validation/bundle-handoff-windows.json) records
an additional cross-machine check: bundles exported on macOS were transferred
to native Windows, moved to another directory, and verified using the built
v0.1.0a5 wheel in a new Python 3.11.9 environment. Installation used `--no-index
--no-deps`; installed distributions were only RolloutCheck, pip and setuptools.
The adapter bundle returned FAIL/1 and the control PASS/0, both with verified
integrity. This was an operator-run check, not independent external adoption.

The practical increment is packaging and checking a complete saved-trace handoff.
It does not yet establish adoption, debugging-time savings, collector overhead or
an advantage on another developer's own rollout problem.
