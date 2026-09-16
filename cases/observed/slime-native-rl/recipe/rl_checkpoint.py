"""Bound CPU checkpoint staging for the pinned single-GPU experiment.

The image's default asynchronous writer stages all tensors on the host even for
synchronous saves. Use PyTorch's standard synchronous filesystem writer instead,
retaining Megatron's conversion, planner, checkpoint format and native loader.
This adapter changes checkpoint I/O, not the loss, gradients or optimizer.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path


def streaming_save(self, sharded_state_dict, checkpoint_dir):
    import torch.distributed as dist
    from megatron.core.dist_checkpointing.strategies.torch import (
        MCoreSavePlanner,
        _replace_state_dict_keys_with_sharded_keys,
        mcore_to_pyt_state_dict,
    )
    from torch.distributed.checkpoint import FileSystemWriter, save_state_dict

    if dist.get_world_size() != 1:
        raise ValueError("The bounded experiment writer is validated for one rank only")
    state, _, _ = _replace_state_dict_keys_with_sharded_keys(
        sharded_state_dict, self.keep_only_main_replica
    )
    state = mcore_to_pyt_state_dict(state, False)
    started = time.monotonic()
    save_state_dict(
        state,
        storage_writer=FileSystemWriter(
            checkpoint_dir, thread_count=1, per_thread_copy_ahead=10 * 1024 * 1024,
            single_file_per_rank=True, sync_files=True,
        ),
        planner=MCoreSavePlanner(
            dedup_replicated_tensors=not self.keep_only_main_replica,
            flatten_state_dict=False,
        ),
    )
    run = os.environ.get("ROLLOUTCHECK_RL_RUN_DIR")
    if run:
        with (Path(run) / "checkpoint-writes.jsonl").open("a") as stream:
            stream.write(json.dumps({
                "checkpoint": Path(checkpoint_dir).name,
                "writer": "torch.distributed.checkpoint.FileSystemWriter",
                "mode": "synchronous", "seconds": time.monotonic() - started,
                "copy_ahead_bytes": 10 * 1024 * 1024,
                "scope": "copy-ahead target, not an absolute process-memory bound",
            }) + "\n")


def install():
    from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy

    # Process-local, scoped to this explicit opt-in experiment. Asynchronous saves
    # are rejected below; the upstream strategy/loader and source files stay intact.
    TorchDistSaveShardedStrategy.save = streaming_save


def before_train_step(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    import torch
    import torch.distributed as dist

    if args.async_save:
        raise ValueError("Bounded checkpoint adapter requires synchronous saves")
    install()
    if dist.get_rank() != 0:
        return

    def inner_optimizers(value):
        children = getattr(value, "chained_optimizers", None)
        if children is not None:
            for child in children:
                yield from inner_optimizers(child)
        else:
            yield value.optimizer

    states = []
    for inner in inner_optimizers(optimizer):
        dtypes = Counter()
        for state in inner.state.values():
            for name, value in state.items():
                if isinstance(value, torch.Tensor):
                    dtypes[f"{name}:{value.dtype}"] += value.numel()
        states.append({
            "class": f"{type(inner).__module__}.{type(inner).__name__}",
            "group_steps": [int(group.get("step", 0)) for group in inner.param_groups],
            "state_elements_by_dtype": dict(dtypes),
        })
    with (Path(os.environ["ROLLOUTCHECK_RL_RUN_DIR"]) / "optimizer-state.jsonl").open("a") as f:
        f.write(json.dumps({"rollout_id": rollout_id, "step_id": step_id,
                            "when": "before optimizer step", "optimizers": states}) + "\n")


def self_test(output):
    """Exact GPU-to-file-to-GPU round trip through the native Megatron loader."""
    import torch
    import torch.distributed as dist
    from megatron.core.dist_checkpointing import load, save
    from megatron.core.dist_checkpointing.mapping import ShardedTensor

    output.mkdir(parents=True, exist_ok=False)
    os.environ["ROLLOUTCHECK_RL_RUN_DIR"] = str(output)
    dist.init_process_group("gloo", init_method=(output / "rendezvous").as_uri(),
                            rank=0, world_size=1)
    try:
        install()
        values = {
            "model": torch.arange(4096, device="cuda", dtype=torch.bfloat16),
            "main_param": torch.linspace(-1, 1, 4096, device="cuda", dtype=torch.float32),
            "exp_avg": torch.linspace(-0.1, 0.1, 4096, device="cuda", dtype=torch.bfloat16),
            "exp_avg_sq": torch.linspace(0, 0.01, 4096, device="cuda", dtype=torch.bfloat16),
        }
        checkpoint = output / "checkpoint"
        checkpoint.mkdir()
        save({key: ShardedTensor.from_rank_offsets(key, value)
              for key, value in values.items()}, checkpoint)
        restored = load({key: ShardedTensor.from_rank_offsets(key, torch.empty_like(value))
                         for key, value in values.items()}, checkpoint)
        assert all(torch.equal(value, restored[key]) and value.dtype == restored[key].dtype
                   for key, value in values.items())
        assert (output / "checkpoint-writes.jsonl").exists(), "Adapter was not invoked"
        result = {"status": "EXACT_ROUND_TRIP", "device": torch.cuda.get_device_name(),
                  "fields": {key: str(value.dtype) for key, value in values.items()},
                  "scope": "Synthetic checkpoint I/O test; not training execution evidence"}
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check the pinned GPU checkpoint adapter")
    parser.add_argument("output", type=Path)
    self_test(parser.parse_args().output.resolve())
