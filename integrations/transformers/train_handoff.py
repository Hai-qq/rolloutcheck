"""Train masked causal-LM LoRA on saved slime Samples with Hugging Face Trainer.

This is an offline supervised bridge, NOT slime/Megatron RL or online weight sync.
Only resume checkpoints produced by this runner locally: Trainer loads optimizer
and RNG files using PyTorch serialization, so never use untrusted checkpoints.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

from prepare_model import REVISION, WEIGHT_SHA256, verify_weights

from rolloutcheck.case import CaseError, load_case
from rolloutcheck.training_batch import audit_batch, pad_features, training_features

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "slime"))
from prepare_assets import verify_assets  # noqa: E402


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def checkpoint_manifest(directory, contract):
    required = {
        "adapter_model.safetensors",
        "adapter_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
        "training_args.bin",
    }
    if not all((directory / name).is_file() for name in required):
        raise CaseError("Incomplete Trainer checkpoint; refusing resume")
    return {
        "contract": contract,
        "files": {name: digest(directory / name) for name in sorted(required)},
    }


def validate_resume(directory, contract):
    receipt, _ = load_case(directory / "rolloutcheck-checkpoint.json")
    if receipt != checkpoint_manifest(directory, contract):
        raise CaseError("Checkpoint data/configuration/file identity mismatch; refusing resume")


def run(args):
    # Validate the supplied data and identity before loading model weights or training.
    snapshot, snapshot_sha = load_case(args.handoff)
    features, audit = training_features(
        snapshot.get("turns"), snapshot.get("samples"), max_length=args.max_length
    )
    lock = verify_assets(args.assets)
    if snapshot.get("token_space") != "sha256:" + lock["files"]["tokenizer.json"]:
        raise CaseError("Snapshot tokenizer identity does not match the verified training assets")
    if args.steps < 1 or args.batch_size < 1 or args.accumulation < 1:
        raise CaseError("Steps, batch size and accumulation must be positive")
    if args.stop_after is not None and not 0 < args.stop_after < args.steps:
        raise CaseError("stop-after must be before the configured final step")
    packages = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "peft", "accelerate")
    }
    source_hashes = {
        str(path.relative_to(Path(__file__).resolve().parents[2])): digest(path)
        for path in (
            Path(__file__).resolve(),
            Path(__file__).resolve().parents[2] / "src/rolloutcheck/training_batch.py",
            Path(__file__).resolve().parents[2] / "src/rolloutcheck/sample_audit.py",
        )
    }
    contract = {
        "handoff_sha256": snapshot_sha,
        "model_revision": REVISION,
        "weight_sha256": WEIGHT_SHA256,
        "packages": packages,
        "source_sha256": source_hashes,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.accumulation,
        "max_length": args.max_length,
        "seed": 7,
        "learning_rate": 0.0002,
        "dtype": "float32",
        "lora": {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.0,
        },
        "objective": "unshifted masked causal-LM cross entropy; no rewards or RL loss",
        "device": args.device,
    }
    if args.resume_from is not None:
        validate_resume(args.resume_from, contract)
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    save(args.output / "contract.json", contract)
    save(args.output / "sample-audit.json", audit)
    save(args.output / "features.json", features)
    verify_weights(args.assets)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise CaseError("MPS requested but unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise CaseError("CUDA requested but unavailable")
    set_seed(contract["seed"])
    tokenizer = AutoTokenizer.from_pretrained(args.assets, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.assets,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, **contract["lora"]))
    initial = {
        name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad
    }
    batches = []
    consumed = []

    class ObservedTrainer(Trainer):
        def training_step(self, model, inputs, num_items_in_batch=None):
            loss = super().training_step(model, inputs, num_items_in_batch)
            consumed.append(
                {
                    "optimizer_step": self.state.global_step + 1,
                    "supervised_tokens": (inputs["labels"][:, 1:] != -100).sum().item(),
                    "rows": inputs["input_ids"].shape[0],
                    "width": inputs["input_ids"].shape[1],
                    "backward_loss": loss.detach().cpu().item(),
                }
            )
            return loss

    def collate(rows):
        batch = pad_features(rows, pad_token_id=tokenizer.pad_token_id)
        start = time.perf_counter_ns()
        report = (
            audit_batch(rows, batch, pad_token_id=tokenizer.pad_token_id) if args.audit else None
        )
        audit_ns = time.perf_counter_ns() - start if args.audit else 0
        if report is not None and report["status"] != "MATCHED":
            raise CaseError("Training batch changed supervised token/attention/label placement")
        batches.append(
            {
                "rows": len(rows),
                "width": len(batch["input_ids"][0]),
                "supervised_tokens": sum(x != -100 for row in batch["labels"] for x in row),
                "batch_sha256": hashlib.sha256(
                    json.dumps(batch, sort_keys=True).encode()
                ).hexdigest(),
                "audit_ns": audit_ns,
                "audit": report,
            }
        )
        return {name: torch.tensor(value, dtype=torch.long) for name, value in batch.items()}

    class Receipts(TrainerCallback):
        def on_step_end(self, training_args, state, control, **kwargs):
            if state.global_step == args.stop_after:
                control.should_save = True
                control.should_training_stop = True

        def on_save(self, training_args, state, control, **kwargs):
            directory = Path(training_args.output_dir) / f"checkpoint-{state.global_step}"
            save(
                directory / "rolloutcheck-checkpoint.json", checkpoint_manifest(directory, contract)
            )

    training_args = TrainingArguments(
        output_dir=str(args.output),
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation,
        learning_rate=contract["learning_rate"],
        lr_scheduler_type="constant",
        optim="adamw_torch",
        weight_decay=0.0,
        max_grad_norm=1.0,
        seed=7,
        data_seed=7,
        use_cpu=args.device == "cpu",
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        logging_steps=1,
        save_steps=1,
        save_total_limit=2,
        report_to="none",
        disable_tqdm=True,
        remove_unused_columns=False,
        label_names=["labels"],
        save_safetensors=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    if training_args.device.type != args.device:
        raise CaseError("Trainer selected a different device from the requested device")
    trainer = ObservedTrainer(
        model=model,
        args=training_args,
        data_collator=collate,
        train_dataset=features,
        callbacks=[Receipts()],
    )
    start = time.perf_counter()
    outcome = trainer.train(
        resume_from_checkpoint=str(args.resume_from) if args.resume_from else None
    )
    if args.device == "mps":
        torch.mps.synchronize()
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    final = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
    changed = sum(not torch.equal(initial[name], value) for name, value in final.items())
    max_delta = max((final[name] - value).abs().max().item() for name, value in initial.items())
    if not changed or not all(torch.isfinite(value).all().item() for value in final.values()):
        raise RuntimeError("No finite parameter update observed")
    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    if not losses or not all(math.isfinite(loss) for loss in losses):
        raise RuntimeError("Missing or nonfinite training loss")
    result = {
        "run_version": 1,
        "state": "STOPPED" if args.stop_after else "COMPLETED",
        "global_step": trainer.state.global_step,
        "resume_from_step": (
            load_case(args.resume_from / "trainer_state.json")[0]["global_step"]
            if args.resume_from
            else 0
        ),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "contract": contract,
        "batch_audit_enabled": args.audit,
        "sample_audit": audit,
        "batch_records": batches,
        "consumed_microbatches": consumed,
        "log_history": trainer.state.log_history,
        "train_metrics": outcome.metrics,
        "elapsed_seconds": elapsed,
        "trainable_parameters": sum(p.numel() for p in final.values()),
        "changed_parameter_tensors": changed,
        "max_abs_change_from_initialized_lora": max_delta,
        "final_adapter_sha256": digest(
            args.output / f"checkpoint-{trainer.state.global_step}" / "adapter_model.safetensors"
        ),
        "source_sha256": source_hashes,
        "scope": "Actual Hugging Face Trainer/PEFT parameter updates on saved generated Samples. "
        "Fixed-rollout supervised fit, not full slime RL, online policy sync or quality evidence. "
        "Batch records include data-loader prefetch; they are not optimizer consumption counts.",
    }
    save(args.output / "result.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("state", "global_step", "elapsed_seconds", "changed_parameter_tensors")
            }
        ),
        flush=True,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--audit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume-from", type=Path)
    run(parser.parse_args())
