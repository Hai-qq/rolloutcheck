# Observed native slime GRPO loop — 2026-09-16

Qwen3-0.6B (596,049,920 parameters) completed three online rollout/training rounds,
six Adam steps, three native Megatron checkpoints and final SGLang evaluation on
one RTX 4070 SUPER / WSL2 host. `process.json` records exit 0. The observed
[verification](verification.json) is `VERIFIED_BOUNDED_RL_LOOP`.

This is the native slime/Megatron/SGLang training loop with an explicit local
synchronous checkpoint-writer adapter. It is separate from the HF supervised
bridge. It is a short single-turn integration experiment, not convergence or a
production-readiness result. See [the recipe and limits](../../../docs/full-slime-rl.md).

| Round | Engine version | Samples | Reward sum | Trainable tokens | Changed exported tensors |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 16 | 8 | 3,603 | 247 |
| 1 | 2 | 16 | 9 | 3,620 | 243 |
| 2 | 3 | 16 | 11 | 3,602 | 244 |

Each round contains within-prompt reward variation and nonzero advantages. One
of the six steps has zero gradient because its groups have equal rewards; the
other five gradient norms are finite and nonzero. Adam's observed pre-step
counters continue from 0 through 5 across actor recreation. Full-model moment
storage remains BF16. Tensor hashes compare values, not only safetensors headers.

The final evaluation uses engine version 4. Four separate greedy arithmetic
questions score **2/4 both before and after**. Different training rounds use
different questions; their reward sums are not a learning curve.

## Recheck without a GPU or PyTorch

From the repository root, after installing the core package:

```sh
python integrations/slime/rl_verify.py cases/observed/slime-native-rl \
  --portable --output artifacts/native-rl-recheck
```

The command recomputes token/mask identity, float32 sampled-logprob preservation,
rewards, group advantages/returns, gradients, parameter changes, engine versions,
evaluation boundaries and optimizer counters. It reads exported source values;
it does not rerun generation or training. `rollout-*.json`, `train-*.json` and
evaluation JSON are direct tensor-to-list exports of the trusted native dumps.

`recipe/` preserves the exact experiment-time scripts. The post-run verifier was
extended to check optimizer recovery; its own digest is in `verification.json`.
`inputs.json` preserves the original recipe hashes. The standard repository
verifier is authoritative for the documented recheck command.

## What is retained

- `weights/*.receipt.json`: exported files and individual tensor-value hashes.
- `optimizer-state.jsonl`, `gradients.json`, `training-metrics.json`: actual
  optimizer observations, gradient norms and native per-step log metrics.
- `checkpoint-writes.jsonl`, `checkpoint-files.json`: completed synchronous writes
  and hashes/sizes of the final checkpoint files. Large checkpoints and exported
  model weights remain on the experiment host, outside Git.
- `training-excerpt.log`: selected lifecycle/metric lines from the native log.
  Node addresses are redacted. `raw-source-hashes.json` identifies the original
  unfiltered log and trusted local PyTorch dumps, retained outside Git.
- `resources.jsonl`, `run-summary.json`: sampled resources, with container IDs
  removed from resource rows. Sampling is approximately every three seconds;
  GPU usage includes the Windows desktop. This is not exact peak measurement.
- `environment/`: package/source identities, CUDA/TE preflight, verified official
  image identity and the separate synthetic checkpoint round-trip test.
- `earlier-attempts.json`: failed configurations and actual process receipts.
  The stock checkpoint writer exceeded the 14/15 GiB container memory limits.
  The successful profile used the synchronous adapter; this is not a matched
  performance A/B experiment.

The accepted run took 504 seconds including startup/shutdown. Sampled whole-GPU
memory reached 11,973 MiB; sampled container memory reached about 14.37 GiB.
This profile has little memory margin. It does not establish long-run stability,
portable hardware minima, multi-rank save support or training-quality gains.
