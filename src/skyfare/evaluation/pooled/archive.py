#!/usr/bin/env python3
"""Build deterministic finalizer and final-report archives."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

from skyfare.core.paths import DataLayout

MODULE = Path(__file__).resolve().parent
LAYOUT = DataLayout.resolve()
ROOT = LAYOUT.root
DIST = LAYOUT.artifacts / "release_packages"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(name: str, files: list[Path]) -> None:
    DIST.mkdir(parents=True, exist_ok=True)
    archive = DIST / name
    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as tar:
                for path in sorted(files, key=lambda item: item.relative_to(ROOT).as_posix()):
                    relative = path.relative_to(ROOT).as_posix()
                    info = tar.gettarinfo(
                        str(path),
                        arcname=f"skyfare_prospective_evaluation/{relative}",
                    )
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)
    digest = sha256(archive)
    (DIST / f"{name}.sha256").write_text(f"{digest}  {name}\n", encoding="utf-8")
    manifest = {
        "archive": name,
        "archive_sha256": digest,
        "files": [{"path": path.relative_to(ROOT).as_posix(), "size": path.stat().st_size, "sha256": sha256(path)} for path in files],
        "status": "PASS",
    }
    (DIST / f"{name}.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ARCHIVE PASS] {name} files={len(files)}")


def main() -> None:
    source = [
        ROOT / "README.rst",
        ROOT / "configs/baseline_metrics.json",
        ROOT / "configs/pooled_evaluation_contract.json",
        ROOT / "requirements.txt",
        *sorted(path for path in MODULE.rglob("*.py") if "__pycache__" not in path.parts),
    ]
    output_root = LAYOUT.artifacts / "pooled_evaluation"
    outputs = sorted(path for path in output_root.iterdir() if path.is_file())
    build("skyfare_prospective_evaluation_code.tar.gz", source)
    build("skyfare_prospective_evaluation_results.tar.gz", source + outputs)


if __name__ == "__main__":
    main()
