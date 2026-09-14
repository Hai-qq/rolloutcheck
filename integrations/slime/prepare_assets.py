"""Explicit network preparation: download only pinned public tokenizer assets, never weights."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
FILES = ("tokenizer.json", "tokenizer_config.json", "config.json", "LICENSE")
LOCK = Path(__file__).with_name("assets.lock.json")


def verify_assets(directory):
    lock = json.loads(LOCK.read_text())
    for filename, expected in lock["files"].items():
        raw = (directory / filename).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Asset digest mismatch: {filename}")
    return lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    lock = json.loads(LOCK.read_text())
    for filename in FILES:
        target = args.destination / filename
        if target.exists():
            raw = target.read_bytes()
        else:
            url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{filename}"
            with urllib.request.urlopen(url, timeout=60) as response:
                raw = response.read(32 * 1024 * 1024 + 1)
        if len(raw) > 32 * 1024 * 1024:
            raise ValueError(f"Asset too large: {filename}")
        if hashlib.sha256(raw).hexdigest() != lock["files"][filename]:
            raise ValueError(
                f"Asset digest mismatch: {filename}; existing files are not overwritten"
            )
        if not target.exists():
            target.write_bytes(raw)
        print(f"Verified {filename} ({len(raw)} bytes)")


if __name__ == "__main__":
    main()
