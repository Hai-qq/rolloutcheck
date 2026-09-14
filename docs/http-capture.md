# Capture an independent HTTP client's rollout

The v0.1.0a7 local service accepts explicit turn metadata from an ordinary HTTP
client, captures the pinned slime adapter's `TurnRecord`, drains a bounded run,
and automatically exports a diagnostic and portable evidence. The client does
not import RolloutCheck, PyTorch or slime. No upstream source patch is required.

This removes the example controller's custom context resolver, manual callback
counting, finalization and export steps. It does not supply identities for a
client that cannot declare them. Actual usefulness for another developer's own
workflow remains to be evaluated.

## Run against your local engine

First prepare the pinned slime source, tokenizer and Qwen3-0.6B engine using
[the native setup](native-sglang.md#repeat-the-native-experiment). Keep that engine
running on `127.0.0.1:30000`. The commands below use its existing environment;
they do not install or start an engine or change network settings.

In a second terminal, from the repository root:

```sh
PYTHONPATH=src .cache/sglang-venv/bin/python integrations/slime/serve_capture.py \
  --source .cache/slime-adapter --assets .cache/qwen3-0.6b \
  --sglang-url http://127.0.0.1:30000 --port 31000 \
  --requests 4 --timeout 180 --output artifacts/http-run
```

Wait for the printed `adapter_url` (also atomically written to `ready.json`).
In a third terminal, run the separate, standard-library-only client:

```sh
python3 integrations/slime/demo_client.py \
  --adapter-url http://127.0.0.1:31000 --sessions 2 \
  --output artifacts/http-client
```

Both output directories must be new. The client sends two concurrent independent
sessions, each with two sequential turns. It echoes the actual first assistant
response into the second request. The first response must finish with `stop`;
the second is deliberately limited to 16 tokens. The service closes admission
after four target requests and exits after draining them and exporting evidence.
Stop the separately managed engine when finished.

The recorded Qwen configuration produces **exit 1 / FAIL** from the service,
while the client exits 0: capture succeeded and the declared history contract
failed. Read both `receipt.json` and `report.json`. A valid diagnostic FAIL must
not be treated as a service crash or changed into exit 0.

```sh
uv run --no-sync rolloutcheck verify-evidence artifacts/http-run/evidence
uv run --no-sync rolloutcheck inspect-trace artifacts/http-run/trace.jsonl \
  --cases-dir artifacts/http-cases
```

These commands also exit 1 for the recorded drift. The extracted cases contain
the session/parent identities and exact token windows for each transition.

## Add metadata to your own client

Use `/v1/chat/completions`, `stream: false`, and an ordinary JSON Accept header.
Each request must include this metadata alongside its messages and model options:

```json
{
  "metadata": {
    "session_id": "public-session-1",
    "rolloutcheck": {
      "session_id": "public-session-1",
      "branch_id": "main",
      "turn_id": "2",
      "parent_turn_id": "1",
      "contract": {
        "version": "history-prefix/v1",
        "mode": "append_only",
        "history_policy": "preserved"
      }
    }
  }
}
```

Use `parent_turn_id: null` for a root. Only declare `append_only/preserved` when
that is the actual intended contract; an empty contract stays inconclusive.
The server supplies the verified tokenizer digest. Identity fields are public
ASCII aliases, 1–128 characters from letters, digits, `_`, `-`, and `.`.
The two session aliases must match. Identity is client-declared, not authenticated.

Omit Authorization. If an SDK requires it, its value must be exactly
`Bearer public-session-1` for the matching session: the pinned upstream adapter
uses Bearer as the session identifier before checking metadata. A different
Bearer is rejected before upstream processing. This is **not authentication**;
never send a real credential to this local demo. The server binds only loopback
and is not intended for exposure to other hosts. The demo client ignores proxies
only for its own requests; it does not modify system proxy/VPN configuration.

`SlimeHTTPCapture.attach(adapter)` installs the middleware and debug callback
before the app starts. A task-local context keeps interleaved sessions separate.
The writer and adapter must share a single event loop; multiple worker processes
must not share a trace. A parent must already have completed capture in the same
session and branch. An early child or duplicate identity is rejected, not queued
or automatically retried. Cross-branch parent links are unsupported.

For an existing application owner, use the helper in
[`slime_http.py`](../src/rolloutcheck/slime_http.py), close admission, drain your
handlers, and `await capture.finish(expected_requests=your_count)` before closing
the recorder. The complete runnable lifecycle is in
[`serve_capture.py`](../integrations/slime/serve_capture.py). The simpler
[callback API](slime-integration.md) remains available for other controllers.

## Failures and retained output

| Event | Behavior |
|---|---|
| Missing/malformed identity or contract; unsupported streaming | HTTP 422 and persistent capture gap |
| Bearer/session mismatch | HTTP 422 before upstream use; no credential persisted |
| Duplicate turn or parent not completed in the same branch | HTTP 409 and gap |
| Request limit already reached | HTTP 503; outside the admitted capture |
| Handler fails, is cancelled, or produces missing/extra callback | Gap; no completion footer |
| Expected requests never arrive | Deadline ends run; incomplete evidence, exit 3 unless an observed pair already FAILs |
| Writer cannot persist records/gaps | Sticky failure prevents successful finalization; output may be partial |

`--requests` is 1–10,000 and counts received chat requests, including rejected
ones. `--timeout` is 1–3,600 seconds for waiting for the admitted workload; shutdown
may additionally drain/cancel active handlers for up to 30 seconds. The receipt's
`failed` counts failure events, not unique failed requests. `succeeded` counts
handler 2xx responses; it is not a delivery acknowledgment from the client.
Streaming through either the body or `Accept: text/event-stream` is rejected.

The output includes raw `trace.jsonl`, narrow independent engine observations in
`wire.jsonl`, `receipt.json`, the aggregate `report.json`, and `evidence/`. Since
v0.1.0a8 it also writes `diagnosis.txt`, a bounded terminal summary with per-session
parent/child identity. The same view is available with
`rolloutcheck verify-evidence artifacts/http-run/evidence --format text`.
The JSON and bundle formats remain unchanged.
The wire file contains public session aliases, input/output IDs, sampling options
and finish metadata, with no decoded text or request headers. The trace/wire
limits are 64 MiB each; individual engine responses are bounded to 16 MiB.
The separate demo client saves its public generated responses in its own output.
Token IDs can reconstruct content; review any real user's evidence before sharing.

The evidence bundle verifies its trace/report only. The adjacent runtime receipts,
wire observations and source hashes are supporting records, not authenticated
proof of model execution. Hard termination or I/O/startup failure can prevent
export entirely. Preserve partial files and rerun into a new directory; never add
a completion footer after the fact. This service has no resume mode.

## Native validation: 2026-09-14

The committed [raw run](../cases/observed/slime-http-cuda) used the unmodified pinned
slime adapter, SGLang 0.5.9, Qwen3-0.6B and an RTX 4070 SUPER under WSL2. The service
and client were separate Python processes. Source hashes and engine launch/runtime
records are retained; capture records identify RolloutCheck 0.1.0a7.

- Two interleaved HTTP sessions / four successful requests / four callbacks and
  captured records, no gaps, checked completion footer.
- Each session has input/output lengths `32/145` then `71/16`. All captured input
  and output ID arrays equal the corresponding independent engine observations.
- Both transitions FAIL at zero-based index 32: expected ID `151667`, actual `17`.
  The expected retained history is 177 tokens; the next actual input has 71.
- The engine used `--max-running-requests 1`: this validates concurrent HTTP
  identity and lifecycle, **not batched GPU execution or training**.
- The owned engine was stopped after the experiment; its port closed and GPU
  free memory returned to 10,925 MiB. No system network configuration was changed.

This is the known history-drift class exercised through a usable external-client
path. It does not prove a new upstream bug or its cause from the trace alone.
The earlier [callback benchmark](collector-overhead.md) does not measure this new
middleware, forwarding observation, or multi-session workload.

The [Windows handoff receipt](validation/http-handoff-windows.json) checks the native
FAIL bundle and a no-client incomplete bundle after relocation, using a fresh
native Windows environment and an offline, dependency-free wheel installation.
This verifies saved evidence, without invoking a model or GPU.

## Reproduce the CPU workflow test

```sh
uv sync --locked --extra generation --extra slime-adapter
uv run --no-sync python integrations/slime/prepare_adapter.py .cache/slime-adapter
uv run --no-sync pytest tests/test_slime_http.py -q
uv run --no-sync python integrations/slime/test_http_capture.py \
  --source .cache/slime-adapter --output artifacts/http-contract
```

CI runs the real pinned adapter with a scripted tokenizer/engine and a separate
client subprocess. It exercises interleaved sessions, complete PASS evidence,
and a no-client deadline producing incomplete evidence. Unit tests force
out-of-order completion, identity/auth/stream rejection, missing callbacks,
cancellation, count mismatch and shutdown failure. The native regression rechecks
committed IDs and evidence offline; CI does not run the GPU model.
