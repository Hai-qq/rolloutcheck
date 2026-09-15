# Native rollout to actual Trainer updates: 2026-09-15

These receipts cover 27 actual Qwen3-0.6B Trainer/PEFT jobs on MPS, totaling 104
executed optimizer steps. Native SGLang generated the fixed inputs in the
[adjacent capture](../slime-training-native); the pinned slime adapter constructed
the Samples. This is masked supervised cross entropy with 1,146,880 trainable
LoRA parameters, not a full slime/Megatron RL training run or quality evaluation.

| Training profile | Longest sample | Batch / accumulation | Supervised tokens per update |
|---|---:|---:|---:|
| Short fork | 177 | 2 / 1 | 161 |
| 80-note fork | 1068 | 1 / 2 | 326 |
| 160-note fork | 1744 | 2 / 1 | 282 |
| Short default | 87 | 2 / 1 | 16 |

Every guard-on/off pair has identical backward records and final adapter tensors.
One warmup pair per profile is excluded, followed by three measured pairs in
alternating order. Median on-minus-off time for a four-step job was +16.63 ms
(short), -4.72 ms (80 notes) and +4.55 ms (160 notes). All three observed ranges
cross zero. These descriptive runs do not resolve stable end-to-end overhead;
they do not measure native generation, preflight audit or total collector cost.
The Mac was not an isolated benchmarking host.

The real HF `DataCollatorForLanguageModeling(mlm=False)` control added 32 unwanted
supervised prompt positions. `audit_batch` reported MISMATCH; preserving the
prepared labels produced MATCHED. This is deliberate integration misuse, not a
claim that the library is defective or that an external user's job was fixed.

Stopping after completed checkpoint 2 and resuming through step 4 produced tensors
identical to uninterrupted training. Missing actual optimizer files and a wrong
data identity were rejected before loading a model. Sudden power loss and disk
failure are not validated.

Prepared `features/` are shared by source handoff SHA256 and serialized compactly;
these are derived labels, not a replacement for the original captured bytes.
`result.json` includes per-step loss/gradient records, actual consumed microbatch
counts, configuration, data/source hashes and parameter-change receipts. Full
adapter, optimizer and RNG checkpoint files remain local; these public JSON
receipts are **observational evidence, not an independently replayed training
proof**. Offline tests recompute input/label audits and check receipt consistency.
To reproduce the parameter comparisons, run the [training experiment](../../../docs/training-job.md)
with the pinned local weights. Do not infer convergence from four repeated passes
over two samples or from reduced training loss.
