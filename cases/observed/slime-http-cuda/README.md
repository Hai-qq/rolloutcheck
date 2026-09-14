# Native independent HTTP-client capture

Observed 2026-09-14 on RTX 4070 SUPER / WSL2, SGLang 0.5.9 and pinned Qwen3-0.6B.
A separate standard-library client sends two independent sessions with two turns
each through the unmodified, source-verified slime OpenAI adapter.

`receipt.json` records four successful requests and captured generations, zero
failure events and successful finalization. `report.json` records two diagnostic
FAILs at index 32. See [the workflow and interpretation](../../../docs/http-capture.md).

- `trace.jsonl`: untouched v2 capture, including checked completion footer.
- `wire.jsonl`: narrow independent observations at the engine forwarding boundary.
  Join within each session only; this client waits for each parent response.
- `evidence/`: automatically exported witness-only bundle. Run
  `rolloutcheck verify-evidence cases/observed/slime-http-cuda/evidence` from the
  repository root; expected exit 1, FAIL with verified integrity.
- `client/`: actual responses to the public arithmetic/continuation prompts.
- `runtime.json`, `sglang-launch.json`, `source-hashes.json`: actual package versions,
  engine command and source snapshot hashes; the base commit precedes this feature.
  These are supporting records, not authenticated execution provenance.
- `preflight.json`, `after.json`, and logs: idle-GPU checks and owned engine cleanup.
  The engine log also includes a preceding four-request smoke run and health checks;
  `runtime.json` timestamps identify the retained run. They are not additional
  requests within this capture receipt.

The source snapshot used alpha 7 code before documentation was finalized. Captured
bytes and their original paths/receipts are preserved. The engine served one GPU
request at a time; this is not a throughput benchmark or a training experiment.
