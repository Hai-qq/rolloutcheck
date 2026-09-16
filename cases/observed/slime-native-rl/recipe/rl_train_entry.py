"""Start the upstream train loop with a bounded, owned local Ray runtime.

Run inside the pinned experiment container, not inside an existing Ray cluster.
All remaining CLI arguments belong to upstream slime. This module does not
replace its rollout, advantage estimation, loss, optimizer, or weight update.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    import ray

    run_dir = Path(os.environ["ROLLOUTCHECK_RL_RUN_DIR"]).resolve()
    if ray.is_initialized():
        raise RuntimeError("This runner requires its own local Ray runtime")
    ray.init(
        address="local",
        num_cpus=8,
        num_gpus=1,
        object_store_memory=256 * 1024 * 1024,
        include_dashboard=False,
        _temp_dir=str(run_dir / "ray"),
        _node_ip_address="127.0.0.1",
    )
    try:
        from slime.utils.arguments import parse_args
        from train import train

        args = parse_args()
        fields = (
            "advantage_estimator", "rewards_normalization", "grpo_std_normalization",
            "normalize_advantages", "kl_coef", "kl_loss_coef", "use_kl_loss",
            "optimizer", "lr", "global_batch_size", "micro_batch_size",
            "n_samples_per_prompt", "rollout_batch_size", "num_rollout", "start_rollout_id",
            "release_train", "optimizer_cpu_offload", "optimizer_offload_fraction",
            "use_precision_aware_optimizer", "exp_avg_dtype", "exp_avg_sq_dtype",
            "main_params_dtype", "main_grads_dtype",
            "recompute_granularity", "recompute_method", "recompute_num_layers",
            "eval_interval", "n_samples_per_eval_prompt", "eval_temperature",
        )
        (run_dir / "resolved-config.json").write_text(json.dumps(
            {name: getattr(args, name) for name in fields}, indent=2, default=str
        ) + "\n")
        train(args)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
