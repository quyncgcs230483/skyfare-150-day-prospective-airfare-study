#!/usr/bin/env python3
"""Build final downloadable production model archive plus integrity evidence."""

from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

from skyfare.core.paths import DataLayout
from skyfare.production.runtime import INPUT_ROOT, OUTPUT_ROOT, ROOT, sha256, write_json_atomic


ARCHIVE_NAME = "skyfare_final_production_models.tar.gz"
RELEASE_ROOT = DataLayout.resolve().artifacts / "release_packages"


def _included_files() -> list[Path]:
    paths: list[Path] = []
    for base in (
        OUTPUT_ROOT / "jobs",
        OUTPUT_ROOT / "deployment",
        ROOT / "artifacts/evidence",
        ROOT / "src/skyfare/models",
        ROOT / "src/skyfare/calibration",
        ROOT / "src/skyfare/policy",
        ROOT / "src/skyfare/production",
        ROOT / "configs",
        ROOT / "data/schema",
    ):
        for path in sorted(base.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".log"}:
                paths.append(path)
    paths.extend(
        [
            INPUT_ROOT / "input_manifest.json",
            INPUT_ROOT / "standard_offers.parquet",
            ROOT / "README.rst",
        ]
    )
    return sorted(set(paths))


def main() -> None:
    verification = json.loads(
        (OUTPUT_ROOT / "deployment/PRODUCTION_VERIFICATION_R1.json").read_text(encoding="utf-8")
    )
    if verification.get("status") != "PASS":
        raise RuntimeError("production verification is not PASS")
    files = _included_files()
    content = {
        "status": "PASS",
        "archive": ARCHIVE_NAME,
        "files": {str(path.relative_to(ROOT)): sha256(path) for path in files},
        "file_count": len(files),
        "training_label_frames_embedded": False,
        "sequence_history_embedded": True,
    }
    content_path = OUTPUT_ROOT / "deployment/RESULT_CONTENT_MANIFEST_R1.json"
    write_json_atomic(content_path, content)
    files = _included_files()
    raw_bytes = sum(path.stat().st_size for path in files)
    free_bytes = shutil.disk_usage(ROOT.parent).free
    reserve_bytes = 1024**3
    if free_bytes < raw_bytes + reserve_bytes:
        raise RuntimeError(
            f"insufficient disk for result archive: free={free_bytes} required={raw_bytes + reserve_bytes}"
        )
    RELEASE_ROOT.mkdir(parents=True, exist_ok=True)
    archive = RELEASE_ROOT / ARCHIVE_NAME
    with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
        for path in files:
            tar.add(path, arcname=str(Path(ROOT.name) / path.relative_to(ROOT)), recursive=False)
    digest = sha256(archive)
    (archive.parent / f"{ARCHIVE_NAME}.sha256").write_text(f"{digest}  {ARCHIVE_NAME}\n", encoding="utf-8")
    archive_manifest = {
        "status": "PASS",
        "archive": ARCHIVE_NAME,
        "sha256": digest,
        "bytes": archive.stat().st_size,
        "file_count": len(files),
        "verification_sha256": sha256(OUTPUT_ROOT / "deployment/PRODUCTION_VERIFICATION_R1.json"),
    }
    write_json_atomic(archive.parent / f"{ARCHIVE_NAME}.manifest.json", archive_manifest)
    print(f"[PRODUCTION ARCHIVE PASS] {ARCHIVE_NAME} sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
