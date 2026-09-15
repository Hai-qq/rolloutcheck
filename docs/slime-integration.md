# slime debug-callback integration

RolloutCheck can collect a slime adapter's raw `TurnRecord` through the existing
`BaseAdapter(debug_callback=...)` option. No slime source patch or global hook is
installed. The inspected source is pinned to
[`4c193f1f37509cca70f0e88807a9305b70f63f4e`](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/agent/adapters/common.py).

The callback integration has been exercised with the unmodified OpenAI adapter
and real loopback HTTP with a scripted server, a local Transformers/Qwen3-0.6B
service, and [native SGLang on CUDA](native-sglang.md). The Anthropic adapter
shares the base callback but its HTTP route has not been tested here.

## Ready-to-run HTTP workflow

Since v0.1.0a7, [`SlimeHTTPCapture` and the local service](http-capture.md) provide
a request-specific context resolver and bounded shutdown from explicit client
metadata. Two interleaved sessions have been verified against native SGLang.
Use this path when your client can supply public session/branch/turn/parent IDs.
The lower-level callback example below is for custom controllers.

## Attach explicitly

In your controller, construct a `TurnContext` for each request, using stable public
session aliases and actual branch/parent identity. The callback does not receive
an authoritative turn ID or parent ID from slime. It cannot safely invent these
from completion order, timestamps, or token-prefix equality.

```python
from rolloutcheck.trace import TraceRecorder
from rolloutcheck.slime_capture import SlimeDebugCapture, TurnContext
from slime.agent.adapters.openai import OpenAIAdapter

# Your controller owns the mapping. Return None when ancestry is unavailable.
# A concurrent controller needs a request-specific resolver; a shared "last turn"
# variable is only valid in the explicitly sequential experiment below.
def resolve_context(sid, messages, tools, response, turn):
    return controller.lookup_turn_context(sid, messages, turn)

with TraceRecorder("slime-run.jsonl", trace_id="public-run-001",
                   evidence_kind="observed_rollout") as recorder:
    capture = SlimeDebugCapture(recorder, resolve_context,
                               eos_token_ids=[tokenizer.eos_token_id])
    adapter = OpenAIAdapter(tokenizer=tokenizer, sglang_url=your_local_endpoint,
                            debug_callback=capture)
    # Start and drive the adapter in your existing lifecycle, then drain requests.
    # The controller's returned context has this shape:
    example_context = TurnContext(
        session_id="public-session-001", branch_id="main",
        turn_id="2", parent_turn_id="1", token_space="tokenizer-asset-digest",
        contract={"version": "history-prefix/v1", "mode": "append_only",
                  "history_policy": "preserved"},
    )
    # Mandatory: check before closing the recorder or accepting the trace.
    capture.raise_if_failed(expected_turns=controller.served_turn_count)
```

`controller`, its lookup and its served-turn count are integration responsibilities,
not provided slime APIs. The older `verify_adapter.py` and `verify_sglang.py`
runners demonstrate an explicit, sequential request driver. For tested interleaved
HTTP identity, use the [new bounded service](http-capture.md); general concurrent
ancestry resolution remains outside these examples.
Branches remain isolated by the offline checker; cross-branch parent links are
INCONCLUSIVE rather than silently reinterpreted.

The callback persists only IDs, caller-supplied context and narrow metadata. It
does not save raw `sid`, messages, tool schemas or response text. This matters
because slime can derive `sid` from a bearer credential. Do not put that value
into `TurnContext.session_id`; supply your own public alias. Token IDs themselves
can reconstruct content, so traces still require review before sharing.

## Training samples are a separate boundary

The prefix check applies to the caller-declared append-only contract. The pinned
slime trajectory manager may intentionally realign or split drifted history when
`finish_session` emits training samples. A prefix FAIL must not be presented as
proof of wrong training loss. See the [actual handoff comparison](training-handoff.md)
for sampled-context matching and retained trainable-token accounting.

## Capture failures and coverage

The inspected slime implementation catches and logs callback exceptions. Raising
inside a callback therefore does not stop an upstream run. RolloutCheck records
`capture_gap` events for unresolved identity, malformed/empty turn records and
resolver failures. A trace with a gap cannot aggregate to PASS. An existing FAIL
remains FAIL, and the gap is still shown.

After requests drain, **call `raise_if_failed(expected_turns=...)` while the writer
is open**, using your controller's independently counted served turns. This
checks failures and detects missing callback invocations. A mismatch is itself
recorded as a gap. Do not derive the expected count from the collector's counters.
If disk writes fail, the trace cannot reliably record its own error; the sticky
in-memory health check still rejects the run. It does not validate a count that
the controller supplied incorrectly.

Since v0.1.0a6, the default v2 recorder is finalized by a successful
`raise_if_failed(expected_turns=...)`. Call it exactly once after draining all
requests and before closing the writer. Omitting it leaves a visible missing
completion state, even when the captured pairs pass and no callback reported a
gap. A FAIL remains actionable. Late callbacks after finalization are rejected;
the footer covers the declared completed capture, not future requests. Explicit
`trace_version=1` recorders retain the legacy health check without a footer.
See [migration and fault-injection checks](trace-completion.md).

The callback runs after response flush and before trajectory-manager recording.
It covers served adapter turns, not requests rejected before generation, client
cancellations, all engine requests, or training samples. An upstream context-budget
response with zero output IDs creates a gap rather than a fabricated generation.

`TurnRecord.output_ids` comes from the token-ID fields in the upstream response's
logprob tuples. We copy these IDs unchanged. `finish_reason: stop` does not prove
that EOS was returned. Configured EOS IDs allow reporting whether a final EOS is
present; [native probes](native-sglang.md#stop-token-behavior-in-this-configuration)
record stop-token retention for one SGLang configuration. The generic callback
does not attest that behavior. No logprob, mask or training correctness is inferred from PASS.
The integration reference revision in each record is a compatibility reference,
not authentication of the installed runtime.

## Reproduce the HTTP contract test

```sh
uv sync --locked --extra generation --extra slime-adapter
# Explicitly download and verify a small, unmodified source subset; no model weights.
uv run --no-sync python integrations/slime/prepare_adapter.py .cache/slime-adapter
uv run --no-sync python integrations/slime/verify_adapter.py \
  --source .cache/slime-adapter --output artifacts/slime-adapter-check
```

The [source lock](../integrations/slime/adapter-source.lock.json) pins every source
file's size and digest. The runner refuses changed files. It imports the actual
adapter, starts it on loopback, and sends two HTTP requests for each scenario:

| Scenario | Captured turns | Gaps | Expected result |
|---|---:|---:|---|
| Clean prefix | 2 | 0 | PASS |
| Changed prefix | 2 | 0 | FAIL |
| Missing turn context | 1 | 1 | INCONCLUSIVE |
| Clean prefix, omitted health check | 2 | 0 | INCONCLUSIVE; completion missing |
| Changed prefix, omitted health check | 2 | 0 | FAIL; completion missing |
| Declared three served turns, only two callbacks | 2 | 1 | INCONCLUSIVE |

The tokenizer and model responses are scripted in this contract test. The
committed [synthetic evidence](../cases/synthetic/slime-adapter) is labeled accordingly.
CI runs this path with CPU PyTorch, without downloading model weights. The runner
returns 0 only if all six expected outcomes, completion states and ID comparisons hold.
The committed snapshots predate v2; current CI artifacts include the six-scenario
lifecycle check using newly captured v2 traces.

## Real Qwen through the actual adapter

Use the pinned Qwen assets from [the model preparation steps](observed-case.md),
then run:

```sh
uv run --no-sync python integrations/slime/capture_local_model.py \
  --source .cache/slime-adapter --assets .cache/qwen3-0.6b \
  --output artifacts/slime-local-qwen
```

This runner starts a local Transformers service that implements the specific
`/generate` response fields the adapter consumes. It uses actual generated IDs
and selected-token log probabilities computed from generation scores, not canned
model replies. Its code is intentionally a sequential experiment, not an inference
server. It shuts down both HTTP listeners after the run.

The committed [observed evidence](../cases/observed/slime-local-qwen) records a
macOS/MPS run with the pinned Qwen3-0.6B model. The first request has 32 tokens;
the model produces 176 including terminal EOS. The next actual adapter request
has 67 tokens and differs from the retained 208-token history at index 32.
Both adapter snapshots exactly match the model service's independent input/output
observations. The second call generates 16 tokens and stops at its length limit.

This next input differs from the earlier direct-Transformers experiment's 66-token
input: the unchanged slime reply path decodes and passes through the model's final
EOS in its text view. We preserve and report that behavior. The local service's
EOS-in-logprobs behavior must not be assumed to match every SGLang configuration.

The original template removes historical reasoning in this setup. This is a
local observation of the known class of history drift; no new upstream bug,
full SGLang compatibility, training benefit or external adoption is claimed.

## Capture a native SGLang endpoint

`integrations/slime/verify_sglang.py` drives an operator-managed native `/generate`
endpoint through the same pinned, unmodified slime OpenAI adapter. It accepts an
explicit loopback HTTP address only, refuses redirects, and bounds response size.
It does not install or launch SGLang, or authenticate the process behind a port.
Keep the engine environment separate from the pinned client dependencies below:

```sh
uv sync --locked --extra generation --extra slime-adapter
uv run --no-sync python integrations/slime/verify_sglang.py \
  --source .cache/slime-adapter --assets .cache/qwen3-0.6b \
  --sglang-url http://127.0.0.1:30000 --output artifacts/slime-sglang
```

Prepare the verified tokenizer and source subset as above, and serve the pinned
Qwen3-0.6B weights before running this command. The output directory must be new.
Record the actual engine/client package versions, launch command, GPU, weight
digest and model revision independently. A successful endpoint response alone is
not evidence of the server's identity.

The runner performs six requests: two adapter turns, one constructed append-only
control, and three native length/stop probes. It saves each complete wire response
before field validation, compares callback IDs with the wire fields, and requires
the first adapter generation to finish rather than exhaust its token budget.
`output_token_logprobs` must contain finite scores and integer token IDs consistent
with prompt/completion counts. If top-level output IDs are also returned, they
must agree. Missing IDs are never reconstructed from text.

The positive control reuses the first observed generation and explicitly appends
a suffix to its retained IDs; it is not the upstream canonical-continuation fix.
The one-token mutation is performed offline and labeled synthetic. The stop probes
record text, returned IDs and finish reasons with trimming enabled/disabled;
`no_stop_trim` behavior must be read from these observations, not presumed.
Traces remain sequential and do not establish general concurrent ancestry or
training correctness. Wire files contain public test prompts and generated text,
in addition to the narrower token-only callback trace.

CI runs `integrations/slime/test_live_runner.py` against six scripted HTTP responses,
including a response split across transport chunks. This validates the runner's
capture path and controls without claiming a SGLang or GPU execution.
