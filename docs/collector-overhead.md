# Measuring collector overhead

The opt-in benchmark compares collector **off** (`debug_callback=None`) with
collector **on** through the pinned, unmodified slime OpenAI HTTP adapter and an
operator-managed native SGLang endpoint. It does not install a server, change
engine settings, download model weights, or modify system network configuration.

## Workload and timing boundary

Each trial starts a fresh adapter session and sends the same public arithmetic
prompt followed by a continuation using the first actual response. The first
generation has a 512-token cap and must finish with a stop; the second has a
16-token cap. Sampling uses temperature 0, top-p 1 and top-k -1. These are two
sequential requests, without batching or concurrent ancestry.

Default schedule: two warmup on/off pairs, then 20 measured on/off pairs, for
**88 engine requests** in total. Measured pair order is balanced (10 on-first,
10 off-first) and shuffled using seed 0. Warmup samples are retained but excluded
from statistics. The runner does not flush the engine cache; this is a repeated
prompt, warm-cache workload, not cold-start or general training performance.

The trial timer includes session opening, both adapter HTTP requests, session
draining, and (when enabled) trace creation, callbacks, completion and file close.
Adapter/client/listener setup and subsequent evidence validation are excluded.
Both arms use the same additional loopback proxy to observe engine request and
response payloads in memory. This shared measurement instrumentation makes the
baseline an instrumented adapter path, not an uninstrumented production server.

The callback's own elapsed time and the finalization time are also recorded.
The callback measurement includes timer instrumentation with no correction.
File writes are flushed by the collector; there is no `fsync` or power-loss
durability claim. Store traces on the intended filesystem and record it with the
result: local WSL ext4, a Windows-mounted path, and network storage can differ.

After timing, the runner checks that callback IDs equal independently observed
wire IDs and that each on trace has completed capture. Within each measured
on/off pair it requires exactly matching input IDs, sampling parameters, output
IDs and finish reasons. Request UUIDs are excluded from this comparison.
If any pair differs, it saves **NOT_COMPARABLE** and returns exit 1, without an
overhead estimate. It does not drop inconvenient samples or silently replace
them. A failed request or incomplete capture stops collection; partial trial
files remain available, but there is no completed summary.

## Run a native measurement

Prepare the verified source, tokenizer and GPU service using
[the native SGLang instructions](native-sglang.md#repeat-the-native-experiment).
Use an idle GPU and record the engine version, device, package versions, launch
command and model digest independently. The endpoint alone does not authenticate
that environment. From the repo, with the same separate engine environment:

```sh
PYTHONPATH=src .cache/sglang-venv/bin/python integrations/slime/benchmark_capture.py \
  --source .cache/slime-adapter --assets .cache/qwen3-0.6b \
  --sglang-url http://127.0.0.1:30000 \
  --pairs 20 --warmup-pairs 2 --seed 0 --output artifacts/capture-overhead
```

The endpoint must be explicit loopback HTTP; redirects are refused. The output
directory must be new. Use an even number of measured pairs between 2 and 100;
2 is suitable for a smoke test, not a useful performance estimate. Warmup pairs
must be between 1 and 10. The runner contains fixed public prompts, but its traces
and raw timing records still contain generated token IDs; review before sharing.

## Recompute without a model

```sh
uv sync --locked --no-dev
uv run --no-sync python integrations/slime/benchmark_capture.py \
  --summarize artifacts/capture-overhead
```

This reads the saved `protocol.json`, `trials.jsonl` and on-arm traces. It checks
the expected warmup and randomized measurement schedule, trace hashes, completion
and recorded token IDs, then prints recomputed statistics without editing the
saved files or importing model libraries. The records remain caller-supplied
evidence; these consistency checks do not authenticate wall-clock measurements.

The summary reports on/off mean and median trial duration, paired mean/median
differences, callback median/p95 and median finalization time. A deterministic
10,000-resample paired bootstrap gives a 95% interval for the mean wall-time
difference. The reported percentage divides the mean paired difference by the
off-arm mean. A negative difference alone does not establish a speedup; an interval
spanning zero does not establish zero overhead. Small-workload bootstraps do not
account for systematic cache, thermal, desktop activity or order effects.

## Validation boundary

### Native RTX 4070 SUPER run, 2026-09-14

The [captured run](../cases/observed/slime-sglang-overhead) contains the protocol,
all 44 trial records, 22 on-arm v2 traces, recomputable statistics, engine/client
logs, GPU telemetry and runtime/source manifests. The transferred archive was
verified as SHA256 `796c170ce6c10fd87e04a3a36dcd41d17ab859e37f8afef86b7f1ecd727d006b`.
The saved summary was reproduced on macOS after transfer.

Runtime: RTX 4070 SUPER (12,282 MiB), Windows driver 610.60, WSL2/Linux,
Python 3.12.3, SGLang 0.5.9, PyTorch 2.9.1, Transformers 4.57.1 and the pinned
Qwen3-0.6B weights. Traces were written inside the local WSL filesystem (`stat`
reports `ext2/ext3`), not a Windows-mounted or network directory. The runner and
source hashes match commit `7efea8ee340559e60333f8cda491b9aa9d14c51b`.
The launch command is retained in `sglang-launch.json`.

All 20 measured on/off pairs had identical work. Every trial had 32 input / 145
output tokens in the first request and 71 input / 16 output tokens in the second.
The first generation ended with EOS; the second was length-limited. The 8 warmup
requests are excluded from the following statistics:

| Measurement | Observed result |
|---|---:|
| Measured pairs / requests | 20 / 80 |
| Collector callback median / p95 | 0.218 / 0.243 ms per call |
| Finalization median | 0.033 ms per two-turn trial |
| Two-turn duration median, off / on | 2.797 / 2.880 s |
| Two-turn duration mean, off / on | 2.736 / 2.690 s |
| Paired mean on-minus-off duration | −46.3 ms (−1.69% of off mean) |
| Paired mean difference, bootstrap 95% interval | −266.8 to +165.4 ms |

**The total duration measurement does not resolve the added end-to-end cost.**
The interval crosses zero and is much wider than the directly timed callback.
The paired median difference is +64.9 ms while the paired mean is negative.
Grouping the retained rows by order gives mean differences of −264.0 ms for the
10 on-first pairs and +171.3 ms for the 10 off-first pairs. These observations show
substantial timing variability/order dependence; they do not establish its cause,
zero overhead, or a collector-induced speedup. The directly timed callback is a
narrow local measurement, not the entire observer effect or training throughput.

Every on-arm trace is complete and still reports **FAIL** for the previously
observed history-prefix drift. Benchmark **COMPARABLE** means the workloads match
and the comparison can be computed; it does not turn that diagnostic into PASS.
This run measures the known failing adapter path, not a repaired training pipeline.

Before launch, three GPU samples reported 0% utilization and over 10,700 MiB free.
This was a Windows desktop GPU, not an exclusively reserved research server.
The owned model service was stopped after collection; the port was checked closed
and 10,915 MiB was free. See `preflight.json`, `gpu-telemetry.csv` and `after.json`.

To check these exact saved samples without a GPU:

```sh
uv run --no-sync python integrations/slime/benchmark_capture.py \
  --summarize cases/observed/slime-sglang-overhead
```

### Automated checks and remaining scope

CI tests balanced scheduling, warmup exclusion, known paired statistics, malformed
samples, incomplete schedules and mismatched workloads. Its HTTP test runs the
real pinned adapter against scripted responses, checks on-arm traces, and
recomputes the saved statistics. **CI timings are not GPU performance evidence.**

The earlier six-request native experiment established wire/callback consistency
and stop-token behavior. The separate timing run above adds a narrow callback-cost
measurement and exposes the uncertainty in total elapsed time. It cannot establish overhead for
large models, long contexts, many concurrent sessions, other filesystems or full
training jobs.
