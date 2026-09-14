# slime debug-callback integration

RolloutCheck can collect a slime adapter's raw `TurnRecord` through the existing
`BaseAdapter(debug_callback=...)` option. No slime source patch or global hook is
installed. The inspected source is pinned to
[`4c193f1f37509cca70f0e88807a9305b70f63f4e`](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/agent/adapters/common.py).

The callback integration has been exercised with the unmodified OpenAI adapter
and real loopback HTTP in two settings: a scripted server, and a local
Transformers/Qwen3-0.6B service. Neither setting runs SGLang. The Anthropic adapter
shares the base callback but its HTTP route has not been tested here.

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
not provided slime APIs. The local runners demonstrate an explicit, sequential
request driver. They do not establish general concurrent ancestry resolution.
Branches remain isolated by the offline checker; cross-branch parent links are
INCONCLUSIVE rather than silently reinterpreted.

The callback persists only IDs, caller-supplied context and narrow metadata. It
does not save raw `sid`, messages, tool schemas or response text. This matters
because slime can derive `sid` from a bearer credential. Do not put that value
into `TurnContext.session_id`; supply your own public alias. Token IDs themselves
can reconstruct content, so traces still require review before sharing.

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

The callback runs after response flush and before trajectory-manager recording.
It covers served adapter turns, not requests rejected before generation, client
cancellations, all engine requests, or training samples. An upstream context-budget
response with zero output IDs creates a gap rather than a fabricated generation.

`TurnRecord.output_ids` comes from the token-ID fields in the upstream response's
logprob tuples. We copy these IDs unchanged. `finish_reason: stop` does not prove
that EOS was returned. Configured EOS IDs allow reporting whether a final EOS is
present; SGLang's actual stop-token retention remains unverified until a live
engine test. No logprob, mask or training correctness is inferred from PASS.
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

The tokenizer and model responses are scripted in this contract test. The
committed [synthetic evidence](../cases/synthetic/slime-adapter) is labeled accordingly.
CI runs this path with CPU PyTorch, without downloading model weights. The runner
returns 0 only if all three expected outcomes and the ID comparisons hold.

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

## Next resource gate

The remaining engine validation needs a usable NVIDIA environment and a real
SGLang process. Confirm the RTX 4070 machine's OS (native Linux or Windows/WSL2),
GPU/driver state and authorized connection method before choosing installation
commands. No rented machine, paid service or shared network setting is needed
for the completed tests above.

For that run, retain the exact engine version/launch options, tokenizer and model
revisions, raw `/generate` token fields, stop/EOS behavior and independently counted
served turns. Exercise both a successful transition and a deliberate mismatch,
then compare the captured trace with the live engine records. Engine versions,
stop trimming and concurrency cannot be certified by the current surrogate service.
