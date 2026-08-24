#!/usr/bin/env python3
"""Create deterministic SHA-256 inventory for repository-controlled files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "repository_manifest.json"
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if path == OUTPUT or any(
        part in EXCLUDED_PARTS
        or part.startswith(".venv")
        or part.endswith(".egg-info")
        for part in relative.parts
    ):
        return False
    return not path.is_symlink() and path.is_file()


def main() -> None:
    files = sorted(path for path in ROOT.rglob("*") if included(path))
    payload = {
        "files": {
            path.relative_to(ROOT).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in files
        },
        "status": "PASS",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[REPOSITORY MANIFEST PASS] files={len(files)}")


if __name__ == "__main__":
    main()
