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

CI tests balanced scheduling, warmup exclusion, known paired statistics, malformed
samples, incomplete schedules and mismatched workloads. Its HTTP test runs the
real pinned adapter against scripted responses, checks on-arm traces, and
recomputes the saved statistics. **CI timings are not GPU performance evidence.**

The earlier native experiment established wire/callback consistency and stop-token
behavior, not collector overhead. A native timing result needs a separate actual
run with this protocol and its recorded runtime. It cannot establish overhead for
large models, long contexts, many concurrent sessions, other filesystems or full
training jobs.
