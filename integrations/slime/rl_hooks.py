"""Read-only evidence hooks for slime's disk weight synchronization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def weights_written(args, version_dir: str, rollout_engines) -> None:
    """Record files before SGLang reloads them; reload acceptance is separate."""
    import torch
    import torch.distributed as dist
    from safetensors import safe_open

    if dist.get_rank() != 0:
        return
    root = Path(version_dir)
    files = []
    tensors = {}
    for path in sorted(root.rglob("*.safetensors")):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        files.append({
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        })
        with safe_open(path, framework="pt", device="cpu") as weights:
            for name in sorted(weights.keys()):
                if name in tensors:
                    raise ValueError(f"Duplicate exported tensor {name}")
                value = weights.get_tensor(name)
                tensors[name] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": hashlib.sha256(
                        value.contiguous().view(torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                }
    if not files:
        raise ValueError(f"No exported safetensors in {root}")
    target = root.parent / (root.name + ".receipt.json")
    target.write_text(json.dumps({
        "version": root.name,
        "files": files,
        "tensors": tensors,
        "engine_count": len(rollout_engines),
        "scope": "written files only; not engine acknowledgement",
    }, indent=2) + "\n", encoding="utf-8")
