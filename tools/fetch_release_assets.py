#!/usr/bin/env python3
"""Fetch and verify immutable study archives from the GitHub release."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=ROOT / "artifacts/release_assets")
    parser.add_argument("--filename", action="append", default=[])
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/release_assets.json").read_text(encoding="utf-8"))
    selected = set(args.filename)
    assets = [item for item in config["assets"] if not selected or item["filename"] in selected]
    missing = selected.difference(item["filename"] for item in assets)
    if missing:
        raise ValueError(f"Unknown release assets: {sorted(missing)}")
    destination = args.destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    base = (
        f"https://github.com/{config['repository']}/releases/download/"
        f"{config['release_tag']}"
    )
    for item in assets:
        target = destination / item["filename"]
        if target.is_file() and sha256(target) == item["sha256"]:
            print(f"[ASSET PRESENT] {target.name}")
            continue
        temporary = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(f"{base}/{item['filename']}", temporary)
        observed = sha256(temporary)
        if observed != item["sha256"]:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"{item['filename']}: SHA-256 mismatch expected={item['sha256']} observed={observed}"
            )
        temporary.replace(target)
        print(f"[ASSET PASS] {target.name}")


if __name__ == "__main__":
    main()
