# Observed model generation and training-sample construction

Actual Qwen3-0.6B / Transformers / MPS generation through the pinned, unmodified
slime OpenAI adapter. Each policy runs two requests; `finish_session` produces
real upstream `Sample` objects. No optimizer or full training framework ran.
This service is not SGLang. See [interpretation and reproduction](../../../docs/training-handoff.md).

- `default/` uses the upstream default fork threshold; `fork/` explicitly sets zero.
- Each `handoff.json` retains callback input/output IDs and emitted Sample token,
  response-length and loss-mask fields. No decoded messages or credentials are stored.
- Each `wire.json` independently retains the model service's input/output IDs.
- Each `audit.json` is recomputed from the handoff snapshot by the pure-core auditor.
- Each `trace.jsonl` and `evidence/` retains the original history-prefix FAIL and
  completion check. Bundle verification covers this trace, not adjacent handoff files.
- `summary.json` compares policies and verifies identical generated work.
- `runtime.json` records the MPS environment, weights, request count, zero optimizer
  steps and source hashes. These records do not authenticate execution provenance.

Both policies generate the same 192 output IDs per two-turn conversation. Default
construction marks 16 token positions trainable in one 83-token sample; explicit
forking marks 192 positions trainable in two samples containing 291 total tokens.
All trainable tokens retain a uniquely matching sampled context. This is a
retention-policy tradeoff, not proof of a bug, better learning, or a universal fix.
