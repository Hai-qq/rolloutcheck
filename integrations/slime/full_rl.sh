#!/usr/bin/env bash
# Experimental single-GPU profile for the pinned slime image. Run in its container.
set -euo pipefail

mode="${1:?Usage: full_rl.sh convert|train ABSOLUTE_OUTPUT_DIR}"
output="${2:?An unused absolute output directory is required}"
[[ "$output" == /* ]] || { echo 'Output must be absolute' >&2; exit 2; }
[[ ! -e "$output" ]] || { echo 'Refusing to overwrite an existing run' >&2; exit 2; }
[[ "$(git -C /root/slime rev-parse HEAD)" == a3f500977f33b82de6ac0414c0c43abfe656c3e0 ]]
case "$mode" in convert|train) ;; *) echo 'Expected convert or train' >&2; exit 2 ;; esac
mkdir -p "$output"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$script_dir:/root/slime:/root/Megatron-LM"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_NVLS_ENABLE=0 OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MALLOC_ARENA_MAX=2
export TORCHINDUCTOR_COMPILE_THREADS=1
export RAY_USAGE_STATS_ENABLED=0 RAY_USE_UVLOOP=0
export ROLLOUTCHECK_RL_RUN_DIR="$output"
source /root/slime/scripts/models/qwen3-0.6B.sh
cd /root/slime
python "$script_dir/rl_metadata.py" "$output" "$mode"

if [[ "$mode" == convert ]]; then
    exec torchrun --standalone --nproc-per-node=1 tools/convert_hf_to_torch_dist.py \
        "${MODEL_ARGS[@]}" --hf-checkpoint /models/qwen3-0.6b --save "$output"
fi

python "$script_dir/rl_reward.py" > "$output/arithmetic.jsonl"
python "$script_dir/rl_reward.py" --evaluation > "$output/evaluation.jsonl"
exec python "$script_dir/rl_train_entry.py" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint /models/qwen3-0.6b --load /work/converted \
    --save "$output/checkpoints" --save-interval 1 \
    --actor-num-nodes 1 --actor-num-gpus-per-node 1 --num-gpus-per-node 1 \
    --rollout-num-gpus 1 --rollout-num-gpus-per-engine 1 --colocate --release-train \
    --update-weight-transport disk --update-weight-mode full \
    --update-weight-disk-dir "$output/weights" --update-weight-disk-keep-files \
    --custom-update-weight-post-write-path rl_hooks.weights_written \
    --custom-megatron-before-train-step-hook-path rl_checkpoint.before_train_step \
    --prompt-data "$output/arithmetic.jsonl" --input-key prompt --label-key label \
    --apply-chat-template --apply-chat-template-kwargs '{"enable_thinking":false}' \
    --custom-rm-path rl_reward.reward \
    --eval-prompt-data arithmetic "$output/evaluation.jsonl" --eval-interval 3 \
    --n-samples-per-eval-prompt 1 --eval-temperature 0 --eval-max-response-len 384 \
    --start-rollout-id 0 --num-rollout 3 --rollout-batch-size 4 --n-samples-per-prompt 4 \
    --rollout-max-response-len 384 --rollout-temperature 1.0 \
    --global-batch-size 8 --micro-batch-size 1 \
    --advantage-estimator grpo --kl-coef 0 --kl-loss-coef 0 --entropy-coef 0 \
    --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0 \
    --adam-beta1 0.9 --adam-beta2 0.98 --use-precision-aware-optimizer \
    --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16 \
    --attention-dropout 0 --hidden-dropout 0 --attention-backend flash \
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
    --sglang-mem-fraction-static 0.24 --sglang-context-length 512 \
    --sglang-max-running-requests 4 --sglang-attention-backend triton \
    --sglang-disable-cuda-graph --sglang-disable-piecewise-cuda-graph \
    --save-debug-rollout-data "$output/rollout-{rollout_id}.pt" \
    --save-debug-train-data "$output/train-{rollout_id}.pt" \
    --ci-save-grad-norm "$output/grad-{rollout_id}-{step_id}.pt" \
    --seed 7
