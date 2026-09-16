"""Write the exact local inputs of an opt-in full-framework experiment."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

EXPECTED_WEIGHTS = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    output = Path(sys.argv[1])
    mode = sys.argv[2]
    model = Path("/models/qwen3-0.6b")
    weights_sha = digest(model / "model.safetensors")
    if weights_sha != EXPECTED_WEIGHTS:
        raise ValueError("Model weights differ from the pinned Qwen3-0.6B artifact")
    scripts = Path(__file__).parent
    assets = json.loads((scripts / "assets.lock.json").read_text())
    asset_digests = {name: digest(model / name) for name in assets["files"]}
    if asset_digests != assets["files"]:
        raise ValueError("Tokenizer/configuration differs from the pinned model assets")
    source_names = ["full_rl.sh", "rl_metadata.py", "rl_reward.py", "rl_hooks.py",
                    "rl_train_entry.py", "rl_verify.py", "rl_checkpoint.py", "assets.lock.json"]
    sources = {name: digest(scripts / name) for name in source_names}
    (output / "recipe").mkdir()
    for name in source_names:
        (output / "recipe" / name).write_bytes((scripts / name).read_bytes())
    versions = {}
    for name in ("torch", "sglang", "ray", "transformer-engine", "transformers",
                 "megatron-core", "torch-memory-saver"):
        versions[name] = importlib.metadata.version(name)
    commits = {}
    for name, path in (("slime", "/root/slime"), ("megatron", "/root/Megatron-LM")):
        commits[name] = subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], text=True
        ).strip()
    metadata = {
        "mode": mode,
        "weights_sha256": weights_sha,
        "model_revision": assets["revision"],
        "asset_sha256": asset_digests,
        "versions": versions,
        "source_commits": commits,
        "recipe_sha256": sources,
    }
    (output / "inputs.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
