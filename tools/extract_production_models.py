#!/usr/bin/env python3
"""Verify and safely extract frozen production models from release asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

ASSET = "SKYFARE_128_DAY_FINAL_PRODUCTION_MODELS_R1.tar.gz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/models/final_production"),
    )
    parser.add_argument(
        "--release-config",
        type=Path,
        default=Path("configs/release_assets.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    archive = args.archive.resolve()
    config = json.loads(args.release_config.read_text(encoding="utf-8"))
    expected = next(
        item for item in config["assets"] if item["filename"] == ASSET
    )
    if archive.name != ASSET:
        raise RuntimeError(f"Expected {ASSET}, got {archive.name}")
    if archive.stat().st_size != int(expected["bytes"]):
        raise RuntimeError("Production archive byte count mismatch")
    digest = _sha256(archive)
    if digest != expected["sha256"]:
        raise RuntimeError("Production archive SHA-256 mismatch")

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        roots = {Path(member.name).parts[0] for member in members if member.name}
        if roots != {"skyfare_128_day_final_production_refit_r1"}:
            raise RuntimeError(f"Unexpected production archive roots: {sorted(roots)}")
        for member in members:
            destination = (output / member.name).resolve()
            if output not in destination.parents and destination != output:
                raise RuntimeError(f"Unsafe archive path: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"Links are forbidden in model archive: {member.name}")
        bundle.extractall(output, members=members)

    root = output / "skyfare_128_day_final_production_refit_r1"
    manifest = root / "production_refit_outputs/deployment/DEPLOYMENT_MANIFEST_R1.json"
    if not manifest.is_file():
        raise RuntimeError("Extracted deployment manifest missing")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise RuntimeError("Extracted deployment manifest is not PASS")
    current = {
        "status": "PASS",
        "archive": archive.name,
        "archive_sha256": digest,
        "artifact_root": str(root),
        "deployment_manifest": str(manifest),
        "training_cutoff_inclusive": "2026-08-19",
    }
    pointer = output / "current.json"
    pointer.write_text(
        json.dumps(current, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(current, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
