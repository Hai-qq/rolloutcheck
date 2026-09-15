# Native SGLang training handoff: 2026-09-15

Actual SGLang 0.5.9, RTX 4070 SUPER (12 GB), Qwen3-0.6B BF16, unmodified slime
adapter revision `4c193f1f37509cca70f0e88807a9305b70f63f4e`. The pinned model weight
identity is in each runtime receipt. There are 12 generation requests: two turns
for each of two retention policies on each of three context profiles. Requests
are sequential; this is not concurrent SGLang batching.

| Profile | Prompt lengths, turns 1 / 2 | Output lengths | Default trainable | Fork trainable |
|---|---:|---:|---:|---:|
| Short | 32 / 71 | 145 / 16 | 16 | 161 |
| 80 repeated notes | 758 / 818 | 310 / 16 | 16 | 326 |
| 160 repeated notes | 1478 / 1559 | 266 / 16 | 16 | 282 |

Each policy pair has exactly equal input and output IDs. All six history traces
have complete capture and diagnose history-prefix FAIL; every emitted trainable
token context matches. Threshold zero retains more generated tokens in this
fixture, without fixing history drift or establishing better training quality.
Longer prompts are explicitly constructed public stress inputs, not user field data.

`short/results` and `long/results` retain original handoffs, wire IDs, sealed
traces and offline-verifiable evidence bundles. Runtime receipts omit local
command paths. `provenance` retains exact producer scripts; the current workload
script differs only in formatting of one adjacent string literal. Source
snapshots are immutable evidence, not additional maintained entry points.

A prior long-workload attempt accidentally reused a mutable messages list across
scenarios. It was rejected/excluded; the producer here copies inputs and the CPU
control checks they remain unchanged. Both corrected comparisons have identical
work. The original failed attempt remains a local diagnostic artifact.

`after.json` confirms the owned engine stopped, the loopback port closed and GPU
free memory returned to 10,776 MiB. No VPN, proxy or routing settings were changed.

These are **generation and Sample-construction** receipts. The separate
[actual Trainer job](../trainer-native-mps) consumes these samples on MPS.
See [reproduction instructions](../../../docs/training-job.md).
