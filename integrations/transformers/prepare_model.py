"""Explicitly download the pinned 1.50 GB Qwen3-0.6B weight file (no remote code)."""

import argparse
import hashlib
import urllib.request
from pathlib import Path

REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
WEIGHT_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
WEIGHT_SIZE = 1503300328


def verify_weights(directory):
    path = directory / "model.safetensors"
    if path.stat().st_size != WEIGHT_SIZE:
        raise ValueError("Weight file size differs from pinned model")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != WEIGHT_SHA256:
        raise ValueError("Weight file digest differs from pinned model")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="Directory with prepared tokenizer assets")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    target = args.destination / "model.safetensors"
    if target.exists():
        verify_weights(args.destination)
        print("Verified existing model weights")
        return
    temporary = args.destination / "model.safetensors.partial"
    url = f"https://huggingface.co/Qwen/Qwen3-0.6B/resolve/{REVISION}/model.safetensors"
    digest = hashlib.sha256()
    total, logged = 0, 0
    with temporary.open("xb") as stream, urllib.request.urlopen(url, timeout=60) as response:
        for chunk in iter(lambda: response.read(4 * 1024 * 1024), b""):
            total += len(chunk)
            if total > WEIGHT_SIZE:
                raise ValueError("Download exceeds pinned weight size")
            stream.write(chunk)
            digest.update(chunk)
            if total - logged >= 100 * 1024 * 1024:
                print(f"Downloaded {total / 1e6:.0f} MB", flush=True)
                logged = total
    if total != WEIGHT_SIZE or digest.hexdigest() != WEIGHT_SHA256:
        raise ValueError("Downloaded weights do not match the pinned digest")
    # Exclusive link prevents overwriting a destination created by another process.
    target.hardlink_to(temporary)
    temporary.unlink()
    print("Verified pinned model weights")


if __name__ == "__main__":
    main()
