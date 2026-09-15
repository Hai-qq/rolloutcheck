# A real, bounded training job

The optional bridge connects **saved native SGLang rollouts → actual slime
`finish_session` Samples → Hugging Face Trainer/PEFT LoRA updates**. It supervises
the emitted tokens using their explicit masks. This is fixed-rollout, masked
causal-language-model training, **not slime/Megatron RL**, online policy-weight
synchronization, or evidence of better model quality.

The training dependencies live in the optional `training` extra. The core package
and `audit-samples` command remain free of runtime dependencies.

Observed results: [native capture](../cases/observed/slime-training-native),
[27 Trainer jobs / 104 optimizer steps](../cases/observed/trainer-native-mps),
[fresh offline core install](validation/training-offline-review.json).

## Check your samples before allocating a GPU

```sh
rolloutcheck audit-samples handoff.json --format text
# Use only if your application requires all generated output to contribute:
rolloutcheck audit-samples handoff.json --require-all-generated --format text
```

Exit codes: 0 means all trainable token contexts match; 1 means unmatched contexts
or a violated explicit all-generated retention requirement; 2 means malformed
input/read failure; 3 means ambiguous attribution or no trainable tokens. Counts
for duplicate and unaccounted positions remain visible even with exit 0.
Neither exit 0 nor `MATCHED` establishes complete capture, no duplicates, valid
RL loss, or model quality. Without the flag, intentional dropping is allowed.

The [handoff format and collection example](training-handoff.md) specify how to
save one explicitly scoped session. This command never runs file-provided code.

## Train the captured samples

Prepare the [pinned local weights](observed-case.md#re-run), then:

```sh
uv sync --locked --extra training
uv run --no-sync python integrations/transformers/train_handoff.py \
  --handoff cases/observed/slime-training-native/short/results/fork/handoff.json \
  --assets .cache/qwen3-0.6b --output artifacts/my-training-run \
  --device mps --steps 4 --batch-size 2 --max-length 3072
```

`--device cpu` and `--device cuda` are explicit alternatives; the pretrained job
is currently validated on MPS. The inference half is validated on RTX 4070 SUPER
with SGLang 0.5.9. A CPU test uses a small random model to verify framework loss
semantics, not to stand in for the pretrained experiment.

The runner verifies local model/tokenizer identities, audits the supplied Sample
contexts, and rejects unmatched, ambiguous, duplicate or empty supervision.
`response_length` locates the response suffix. Prompt and mask-zero positions
become label `-100`; labels remain **unshifted** because the causal-LM model shifts
them internally. There is no truncation, packing, retokenization or mask repair.
Right padding has attention mask zero and label `-100`, while a real EOS remains
supervised even when EOS is also used as padding.

LoRA trains `q_proj` and `v_proj`, rank 8, alpha 16, dropout zero. The base weights
stay frozen. The run uses float32, seed 7, AdamW at 0.0002, a constant learning
rate and gradient checkpointing. These settings define a reproducible integration
experiment; they are not a tuning recommendation. Four passes over two samples
do not evaluate generalization. The fixture's slime rewards and rollout logprobs
are not inputs to this supervised objective.

Each new output directory contains configuration, input digest, prepared labels,
sample audit, step losses/gradient norms, consumed microbatch counts, parameter
change checks, and native Trainer checkpoints. `result.json` distinguishes
prefetched batch records from microbatches actually consumed by `training_step`.
Use per-step losses when comparing resumed runs: Trainer's aggregate `train_loss`
can include a cumulative global-step denominator after resume.

## Catch a mask-losing collator

`DataCollatorForLanguageModeling(mlm=False)` constructs labels from input IDs.
Using it on already loss-masked rollout samples can discard the intended mask.
This is a controlled integration misuse, not a newly discovered library bug.

`training_features` prepares unshifted labels; `pad_features` preserves them;
`audit_batch` checks the actual input IDs, attention mask and labels against those
features. Integrate the audit **after your own collator**, converting tensors to
CPU lists. `MISMATCH` should trigger investigation before the optimizer. The
supplied training bridge checks its own batch and never silently changes labels
to make the audit pass. Other padding/packing conventions are unsupported.

The experiment's `collator-control.json` preserves the actual library-produced
bad batch and the corrected batch. CPU regression tests also compare the Qwen
loss with explicit shifted, masked cross entropy and check that unequal-token
microbatches with gradient accumulation match a combined batch's parameter update.

## Stop and resume safely

```sh
uv run --no-sync python integrations/transformers/train_handoff.py \
  --handoff cases/observed/slime-training-native/short/results/fork/handoff.json \
  --assets .cache/qwen3-0.6b --output artifacts/stopped \
  --device mps --steps 4 --batch-size 2 --max-length 3072 --stop-after 2
uv run --no-sync python integrations/transformers/train_handoff.py \
  --handoff cases/observed/slime-training-native/short/results/fork/handoff.json \
  --assets .cache/qwen3-0.6b --output artifacts/resumed \
  --device mps --steps 4 --batch-size 2 --max-length 3072 \
  --resume-from artifacts/stopped/checkpoint-2
```

Resume requires the same data bytes, source hashes, pinned weights, package
versions, device and job configuration. A receipt binds adapter weights,
optimizer/scheduler, RNG and Trainer state; missing or modified files are rejected
before model loading. This detects accidental checkpoint mismatch, not hostile
tampering. Resume only checkpoints you created/trust: Trainer deserializes local
optimizer/RNG files. `--stop-after` is a controlled stop at a completed checkpoint;
power loss, disk-full recovery and mid-write crash recovery are not validated.

## Reproduce the operating-envelope experiment

```sh
uv run --no-sync python integrations/transformers/verify_training_job.py \
  --native cases/observed/slime-training-native --assets .cache/qwen3-0.6b \
  --output artifacts/training-comparison --device mps --pairs 3
```

Each profile has one excluded warmup pair and three measured pairs. The guard
on/off order alternates. Both halves must have identical consumed backward-loss
records and final LoRA parameters. The timed region includes training, checkpoint
I/O and the same observation hooks; loading, preflight sample auditing and native
generation are outside it. The measured difference concerns **batch auditing**,
not total rollout collection overhead. Three pairs support descriptive timing
ranges only, not stable overhead or speedup claims.

To recapture the longer native inputs, run `integrations/slime/verify_training_workload.py`
with `--source .cache/slime-adapter --assets .cache/qwen3-0.6b
--sglang-url http://127.0.0.1:30000 --output artifacts/native-long` against your
already-managed engine. The repeated public notes are an explicit context stress
fixture. Comparisons are eligible only when actual generation IDs match. The
runner copies initial messages so one scenario cannot contaminate the next.

Framework contracts: [Trainer 4.57 documentation](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer),
[PEFT LoRA workflow](https://huggingface.co/docs/peft/main/en/quicktour).
The [pinned slime quick start](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/docs/en/get_started/quick_start.md)
describes the separate full RL setup; this bridge does not execute that job.
