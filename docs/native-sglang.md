# Native SGLang / CUDA observation

On 2026-09-14, the pinned, unmodified slime OpenAI adapter was exercised against
SGLang 0.5.9 serving Qwen3-0.6B on an RTX 4070 SUPER under Ubuntu 24.04 / WSL2.
This is a real engine observation, separate from the scripted HTTP tests and
the earlier Transformers/MPS service.

## Result and evidence

The [saved evidence](../cases/observed/slime-sglang-cuda) contains six native
request/response pairs, callback traces, reports, package versions, source hashes,
the launch manifest and the engine log. Prompts are fixed public arithmetic/test
prompts. No model weights, credentials or remote-access configuration are included.

| Observation | Result |
|---|---|
| First adapter turn | 32 input tokens, 145 output tokens, including EOS `151645` |
| Second adapter turn | 71 input tokens, 16 output tokens, length-limited |
| Declared retained history | 177 tokens; first difference at index 32: `151667` → `17` |
| Callback IDs versus native wire IDs | Both turns match exactly |
| Constructed append-only control | PASS; 192-token next input preserves all 177 history tokens |
| Offline one-token mutation | FAIL; explicitly synthetic |

The failing adapter transition is consistent with the previously demonstrated
class of historical-reasoning removal during chat-template rendering. The checker
locates an observed interval; it does not prove an engine bug or a causal training
effect. The positive control explicitly appends a suffix to retained IDs and makes
a real second generation. It is not the upstream canonical-continuation fix.
The [mutation](../cases/synthetic/slime-sglang-mutation.case.json) is a changed
saved request, not an observed request sent to SGLang.

## Stop-token behavior in this configuration

All three native probes return one token, ID `151667` (`<think>`):

| Probe | Returned text | Finish reason | Token present in logprob tuples |
|---|---|---|---|
| One-token length limit | `<think>` | length | Yes |
| Explicit stop, trimming enabled | empty | stop, matched `151667` | Yes |
| Explicit stop, trimming disabled | `<think>` | stop, matched `151667` | Yes |

Text trimming therefore must not be used to reconstruct the generated IDs in this
run. Separately, the first completed adapter generation includes terminal EOS.
These observations apply to the recorded version and parameters, not every
SGLang backend/version. The generic callback's `engine_stop_token_retention`
field remains `unverified`: that callback cannot authenticate an engine or probe
its behavior. The wire probes and operator manifest provide separate evidence.

## Recheck without a GPU

```sh
uv sync --locked --no-dev
uv run --no-sync rolloutcheck inspect-trace cases/observed/slime-sglang-cuda/adapter.trace.jsonl
# Expected exit 1: the recorded history contract fails.
uv run --no-sync rolloutcheck inspect-trace cases/observed/slime-sglang-cuda/control.trace.jsonl
# Expected exit 0: the constructed control preserves the prefix.
```

Offline regression tests also compare the saved reports, wire IDs, callback IDs,
control and mutation. This checks saved evidence; CI does not rerun a GPU model.

## Repeat the native experiment

Prerequisites: an NVIDIA-capable Linux/WSL2 environment, Python 3.12, a C++
compiler, and a CUDA 12.8 toolkit including NVCC, runtime development headers and
CCCL. PyTorch CUDA wheels alone do not provide the compiler required by this
SGLang version's RoPE JIT kernel. The recorded run used the three official NVIDIA
components listed with URLs and SHA256 hashes in
[`runtime.json`](../cases/observed/slime-sglang-cuda/runtime.json), installed in a
user-owned directory. See NVIDIA's
[installation and redistributable documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

Prepare the verified model/tokenizer and slime source using the commands in
[the model preparation guide](observed-case.md#re-run) and
[the adapter guide](slime-integration.md#reproduce-the-http-contract-test).
Then create a separate environment matching the recorded packages:

```sh
uv venv --python 3.12 .cache/sglang-venv
uv pip install --python .cache/sglang-venv/bin/python \
  -r cases/observed/slime-sglang-cuda/engine-freeze.txt
: "${CUDA_HOME:?Set CUDA_HOME to your CUDA 12.8 toolkit directory}"
export PATH="$PWD/.cache/sglang-venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  .cache/sglang-venv/bin/python -m sglang.launch_server \
  --model-path .cache/qwen3-0.6b --host 127.0.0.1 --port 30000 \
  --dtype bfloat16 --context-length 4096 --mem-fraction-static 0.5 \
  --attention-backend triton --sampling-backend pytorch \
  --disable-cuda-graph --max-running-requests 1 --random-seed 0 \
  --disable-overlap-schedule
```

Keep that process/session running. After its warmup completes and `/health`
returns 200, run in another terminal from the repository root:

```sh
PYTHONPATH=src .cache/sglang-venv/bin/python integrations/slime/verify_sglang.py \
  --source .cache/slime-adapter --assets .cache/qwen3-0.6b \
  --sglang-url http://127.0.0.1:30000 --output artifacts/slime-sglang-repeat
```

The recorded client used the engine environment's Transformers 4.57.1, not the
4.57.6 version pinned for the earlier MPS experiment. Tokenizer assets are the same
verified files. Generated text/token counts can vary with numerical configuration;
the runner checks actual wire/callback alignment and reports actual stop behavior.
The capture predates the alpha version increment and retains collector version
0.1.0a3; source hashes identify the code that ran. Captured files were not relabeled.

This closes the first native-engine resource gate for one sequential configuration.
Concurrent ancestry, other adapters/backends, collector overhead, full training
correctness and independent developer use remain unvalidated.
