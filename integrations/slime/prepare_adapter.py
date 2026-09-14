"""Explicitly fetch the pinned, unmodified slime adapter source subset for local tests."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

LOCK_PATH = Path(__file__).with_name("adapter-source.lock.json")


def verify_source(directory):
    lock = json.loads(LOCK_PATH.read_text())
    for name, entry in lock["files"].items():
        path = directory / name
        if path.stat().st_size != entry["size"]:
            raise ValueError(f"Pinned source size mismatch: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Pinned source digest mismatch: {name}")
    unexpected = {str(p.relative_to(directory)) for p in directory.rglob("*.py")} - set(
        lock["files"]
    )
    if unexpected:
        raise ValueError("Use a dedicated source snapshot directory without extra Python modules")
    return lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    lock = json.loads(LOCK_PATH.read_text())
    for name, entry in lock["files"].items():
        target = args.destination / name
        if target.exists():
            raw = target.read_bytes()
        else:
            url = f"https://raw.githubusercontent.com/THUDM/slime/{lock['revision']}/{name}"
            with urllib.request.urlopen(url, timeout=60) as response:
                raw = response.read(entry["size"] + 1)
        if len(raw) != entry["size"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"Refusing source with unexpected size/digest: {name}")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(raw)
    verify_source(args.destination)
    print(f"Verified {len(lock['files'])} files at slime {lock['revision']}")


if __name__ == "__main__":
    main()
