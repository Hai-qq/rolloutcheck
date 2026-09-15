# RolloutCheck evidence bundle

Status: **witness_only**. This bundle rechecks recorded IDs; it does not execute the original conversion or replay model sampling.

With RolloutCheck 0.1.0a9 installed, from this directory:

```sh
rolloutcheck verify-evidence .
rolloutcheck inspect-trace trace.jsonl
```

Verification checks file hashes and recomputes the report. Exit codes preserve the recorded check status: a valid FAIL bundle still exits 1.

Hashes detect inconsistency, not authenticity. Labels remain caller-supplied. Review before sharing: token IDs can reconstruct text. A trace bundle includes the complete source trace, including other sessions and capture gaps.
