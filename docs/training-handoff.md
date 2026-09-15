# From history drift to training-sample accounting

A history-prefix FAIL does **not** establish that the eventual training sample is
corrupt. In the pinned slime adapter, `finish_session` calls `get_trajectory` to
construct loss-masked `Sample` objects. The trajectory manager can realign drifted
history as context-only tokens or split it into separate samples.

RolloutCheck's experimental `audit_samples` checks this next boundary: which
sampled tokens appear at trainable positions, and whether their full preceding
ID context matches what the generator actually saw. The audit itself does not validate
an optimizer, reward function, log probabilities, or a complete training job.
The [optional training bridge](training-job.md) separately executes actual masked
supervised Trainer/PEFT updates from the emitted Samples.

## Observed Qwen3-0.6B result

The [committed run](../cases/observed/slime-training-handoff-mps) uses actual
Transformers/MPS generation through the unmodified pinned slime OpenAI adapter,
including its real `finish_session → Sample` path. This is **not native SGLang**.
The later [native SGLang comparison](../cases/observed/slime-training-native) is
now recorded separately; its output tokens differ from this earlier MPS fixture.
Each policy makes two real model requests. The policies produce identical input
and output ID arrays, so the comparison isolates this sample-construction setting
on this particular conversation.

| Per-policy measurement | Upstream default (`None`, effective threshold 1024) | Explicit threshold `0` |
|---|---:|---:|
| Generated output tokens | 192 | 192 |
| Tokens at `loss_mask=1` positions | 16 | 192 |
| First-turn output represented at trainable positions | 0 / 176 | 176 / 176 |
| Emitted samples | 1 | 2 |
| Total token slots across emitted samples | 83 | 291 |
| Trainable token/context matches | All 16 | All 192 |
| Ambiguous / unmatched / duplicate occurrences | 0 / 0 / 0 | 0 / 0 / 0 |
| Original history-prefix diagnostic | FAIL | FAIL |
| Optimizer steps | 0 | 0 |

The second generated response has 16 tokens. Under the default policy, the first
response is replaced as context and contributes no trainable positions. This is
an explicit upstream policy, not a newly discovered bug. Threshold zero preserves
the two responses in separate samples here. It does **not** fix the underlying
chat-template drift, and it increases the sample token footprint.

Do not turn this observation into a blanket recommendation to set the threshold
to zero. Different sample splitting changes context cost and interacts with
reward assignment and loss aggregation. More retained tokens do not by themselves
establish better convergence, higher quality, saved GPU time, or correct RL loss.
If an operator expects every sampled response to train, this comparison makes the
policy discrepancy visible; if response dropping is intentional, it explains the
observed retention rather than declaring an error.

## Audit saved handoff data offline

No model/framework dependencies are needed to recompute the token accounting:

```sh
uv sync --locked --no-dev
uv run --no-sync python - <<'PY'
import json
from pathlib import Path
from rolloutcheck.sample_audit import audit_samples
root = Path("cases/observed/slime-training-handoff-mps")
for policy in ("default", "fork"):
    data = json.loads((root / policy / "handoff.json").read_text())
    result = audit_samples(data["turns"], data["samples"])
    print(policy, json.dumps(result, indent=2))
PY
```

For your own already-drained adapter, pass only the explicit session's captured
turns and the actual returned samples:

```python
from rolloutcheck.sample_audit import audit_samples

samples = await adapter.finish_session(public_session_id, base_sample=base_sample)
# observed_turns comes from the controller's callback snapshots for this session:
# [{"turn_id": "1", "input_ids": [...], "output_ids": [...]}, ...]
report = audit_samples(observed_turns, [
    {"tokens": s.tokens, "response_length": s.response_length, "loss_mask": s.loss_mask}
    for s in samples
])
```

The complete capture and adapter-lifecycle example is
[`verify_training_handoff.py`](../integrations/slime/verify_training_handoff.py).
It does not install a training hook or change any Sample. Configure retention
policy in your own training owner; this utility does not automatically reject or
rewrite training data.

## Meaning and limits of the accounting

- `context_status=MATCHED`: every trainable token has exactly one candidate among
  the supplied sampled token-and-full-prefix pairs. It is not a training PASS.
- `unaccounted_generated_tokens`: sampled outputs that could not be uniquely
  assigned to a trainable position. This can reflect intentional dropping or
  ambiguous source attribution; do not assume accidental data loss.
- `unmatched_trainable_tokens`: a token or its full preceding ID context has no
  match among the supplied observations. Investigate missing observations and
  sample construction before attributing a cause.
- `ambiguous_trainable_tokens`: repeated identical generation contexts cannot be
  assigned to one source turn. The auditor never chooses by arrival order.
- `duplicate_training_occurrences`: a uniquely identified generated token appears
  at more than one trainable position. This remains visible even when every
  context matches; repeated training may or may not be the intended policy.

Masks must be explicit binary integer lists covering the response region;
`response_length` determines that region's offset inside `tokens`. Tool/history
positions with mask zero do not count as training. No implicit all-ones mask is
invented. Full token prefixes are interned exactly in a trie, avoiding both
quadratic prefix materialization and token-only matching. Input is limited to
131,072 total ID entries and 4,096 turn/sample objects per call.

This does not establish completeness/authenticity of the supplied snapshots,
attention-mask behavior, position IDs, model/weight versions, packed-sequence
semantics, logprob consistency, correct rewards or optimizer behavior. Scope one
session explicitly. An observed token sequence from another session with identical
IDs is not identity evidence. The saved history trace has its own completion
footer; that footer does not authenticate the adjacent `handoff.json`.

## Reproduce the framework path

CPU control, with the real pinned adapter and synthetic generation:

```sh
uv sync --locked --extra generation --extra slime-adapter
uv run --no-sync python integrations/slime/prepare_adapter.py .cache/slime-adapter
uv run --no-sync python integrations/slime/verify_training_handoff.py \
  --source .cache/slime-adapter --output artifacts/handoff-control
```

This runs clean, default-drift and fork-drift scenarios. CI runs the same path
with CPU PyTorch and no model download. Deliberately corrupted masks/contexts,
ambiguous source tokens and duplicates are covered by the pure-core tests.

Actual MPS generation, after [preparing the pinned weights](observed-case.md#re-run):

```sh
uv run --no-sync python integrations/slime/verify_training_handoff_mps.py \
  --source .cache/slime-adapter --assets .cache/qwen3-0.6b \
  --output artifacts/handoff-mps
```

For an operator-managed native engine, use the
[native engine setup](native-sglang.md#repeat-the-native-experiment), then:

```sh
PYTHONPATH=src .cache/sglang-venv/bin/python integrations/slime/verify_training_handoff.py \
  --source .cache/slime-adapter --assets .cache/qwen3-0.6b \
  --sglang-url http://127.0.0.1:30000 --output artifacts/handoff-native
```

The native handoff command has now been validated against SGLang 0.5.9 on an
RTX 4070 SUPER. Short and two longer context profiles have matching generation
IDs across retention policies. Runners refuse existing output folders and compare
actual generation IDs before accepting a policy comparison. The separate
[training job](training-job.md) consumes these snapshots without claiming a full
slime/Megatron RL run.
