# Native slime RL experiment

This opt-in experiment runs upstream slime's Megatron training loop and SGLang
rollout engine. It is separate from the [HF Trainer bridge](training-job.md).
The experiment is not part of installing or running the offline checker.

**Observed on 2026-09-16:** a complete bounded loop passed verification on one
RTX 4070 SUPER / WSL2 host: 3 rounds, 48 samples, 10,825 trainable tokens,
6 optimizer steps, 3 checkpoints and final engine weight version 4.
[Source evidence and offline recheck](../cases/observed/slime-native-rl/README.md)
include real optimizer recovery and changed tensor values. Four separate
arithmetic evaluation questions score 2/4 both before and after. This verifies
the integration path, not convergence or better model quality.

## Fixed inputs

- Official image: `slimerl/slime:v0.3.2-cu129`, pinned to
  `sha256:39be6cbb00f9b6770e664ace0c7b9f5ecff2977a1a205e7926a720f906fbc62c`.
- slime source in that image: `a3f500977f33b82de6ac0414c0c43abfe656c3e0`.
- Megatron base source: `1dcf0dafa884ad52ffb243625717a3471643e087`.
  Megatron and SGLang include patches bundled by the image publisher; the image
  digest pins these too. These are not clean checkouts of their base commits.
- Qwen3-0.6B revision and tokenizer/configuration hashes are in
  [assets.lock.json](../integrations/slime/assets.lock.json). The weight file's
  SHA256 is `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.

The recipe checks model bytes before use, copies its source into each output
directory, and records installed package versions. The model is loaded offline.
Downloading the complete image requires approximately 21.2 GB compressed, with
additional space for extraction and several checkpoints/weight exports.

## Workload and resource plan

[full_rl.sh](../integrations/slime/full_rl.sh) selects a single GPU, colocated
training/rollout, native GRPO and Adam, three rollout rounds, four prompts per
round and four samples per prompt. Global batch size 8 produces two optimizer
steps per round. Responses are capped at 384 tokens with a 512-token engine
context. The deterministic reward checks the arithmetic answer; it does not call
an external model or reward service.

The trainer is released between rounds using upstream `--release-train` and
recreated from its checkpoint. The GPU precision-aware Adam path stores moments
in BF16 while retaining FP32 master-parameter precision and accumulated gradients.
Transformer Engine represents the master weights using BF16 model values plus
int16 remainders; the observed optimizer records expose that storage choice.
CPU optimizer offload is disabled: the pinned hybrid offload implementation does
not forward the configured moment dtypes to its inner optimizers. Full-layer activation
recomputation limits GPU memory. Compiler concurrency is one, BLAS/OpenMP thread
counts are bounded, and glibc uses at most two allocator arenas. These settings
are local to the experiment processes. Full
weights are synchronized through upstream disk transport. These are resource
choices for the local 12 GB GPU, not measured throughput recommendations.
Do not assume that a model fitting inference also fits this training profile.

The explicit [checkpoint adapter](../integrations/slime/rl_checkpoint.py) replaces
only the save method of the pinned Megatron strategy inside each experiment
worker. The image's asynchronous writer stages all state on CPU even when saving
synchronously. This adapter uses PyTorch's synchronous `FileSystemWriter` with
one writer and 10 MiB copy-ahead, retaining Megatron's tensor mapping, save planner,
format and native loader. Copy-ahead is not a hard process-memory limit: a tensor
can exceed it. Asynchronous and multi-rank saves are outside this adapter's scope.
The recipe therefore runs the native training loop with a local checkpoint I/O
adapter; it is not a claim that the stock image fits this host unchanged.

Test the adapter on the already configured GPU container before training:

```sh
python /work/code/rl_checkpoint.py /work/checkpoint-io-test
```

This compares GPU model/master-weight/moment tensors exactly after native reload.
It is a synthetic I/O test; the full run must additionally demonstrate real
optimizer updates and checkpoint restores. `optimizer-state.jsonl` records the
actual inner optimizer classes, step counters and state dtypes before each step.

Four separate arithmetic questions are evaluated greedily before training and
after the final update. They are a smoke evaluation, not a statistically useful
model-quality benchmark. The verifier does not require accuracy to improve.

The accepted run used a 15 GiB container memory limit, 18 GiB memory-plus-swap
limit and 8 CPU quota on WSL with approximately 15.35 GiB RAM and 4 GiB swap.
Sampled container use reached 14.37 GiB and whole-GPU use 11,973 MiB. Treat this
as a narrow observed profile, not a safe general hardware minimum. It took
504 seconds including startup and shutdown. The inference environment and
shared network settings were preserved.

## Running the recipe

Use an isolated GPU container with the pinned image. Provide these mounts:

| Container path | Contents |
| --- | --- |
| `/models/qwen3-0.6b` | Verified complete model directory, read-only |
| `/work/code` | Repository `integrations/slime` directory, read-only |
| `/work` | New writable experiment workspace |
| `/opt/rolloutcheck/src` | Repository `src` directory, read-only, for verification |

The host must already provide working CUDA device access. WSL's GPU passthrough
and Linux NVIDIA Container Toolkit are different setups; this recipe does not
install drivers, change a VPN/proxy, or configure a shared Docker daemon.
Run the two commands sequentially inside separate owned containers:

```sh
bash /work/code/full_rl.sh convert /work/converted
bash /work/code/full_rl.sh train /work/run-001
```

Output paths must not already exist. Keep failed attempts under their original
paths instead of overwriting them. The runner starts and shuts down its own local
Ray runtime; it does not use `ray stop --force` or terminate unrelated processes.

After the training container exits, retain its actual exit status and Docker
`State` in `/work/run-001/process.json`, using this structure:

```json
{"returncode": 0, "state": {"ExitCode": 0, "OOMKilled": false}}
```

Those values must come from the completed process and container inspection, not
from copying this example. The record may retain additional fields. Then run:

```sh
PYTHONPATH=/work/code:/opt/rolloutcheck/src \
  python /work/code/rl_verify.py /work/run-001
```

The verifier exports portable JSON for the sampled/training records, evaluation
and gradients. Recheck these without PyTorch by passing `--portable`; use
`--output artifacts/recheck` to write the verdict separately from the source
records. This recomputes the diagnosis from recorded values, not from a GPU rerun.

Only load the experiment's own trusted debug dumps. PyTorch is needed to read
these files; the normal RolloutCheck package still has no runtime dependencies.

## Acceptance and limits

The verifier checks actual saved rollout/training samples: token identity,
response boundaries, masks, float32 rollout-logprob handoff, recomputed rewards,
group-normalized advantages and returns. Every round must have a real nonzero
training signal, completed optimizer steps and changed exported tensor values.
Subsequent generation must identify the expected new engine weight version.
Final evaluation must use the final weight version and unchanged evaluation
prompts that were absent from training.

`VERIFIED_BOUNDED_RL_LOOP` means those checks passed for this observed experiment.
It does not establish convergence, general reward correctness, independent
wire-level capture, multi-turn history behavior during RL, distributed training,
production reliability or independent adoption. Tensor hashes and saved records
support local reproduction; they are not proof of third-party authenticity.
