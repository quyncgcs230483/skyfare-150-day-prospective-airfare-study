#!/usr/bin/env python3
"""Build checksummed Test 2 result and compact evidence archives."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

from skyfare.core.paths import DataLayout
from skyfare.evaluation.prospective_test_two.runtime import (
    OUTPUT_ROOT,
    ROOT,
    sha256,
    write_json_atomic,
)

ARCHIVE_NAME = "skyfare_prospective_test_two_results.tar.gz"
EVIDENCE_NAME = "skyfare_prospective_test_two_evidence.tar.gz"
RELEASE_ROOT = DataLayout.resolve().artifacts / "release_packages"


def _add_tree(archive: tarfile.TarFile, source: Path, arcname: str) -> None:
    archive.add(source, arcname=arcname, recursive=True)


def _write_sha(path: Path) -> None:
    path.with_name(path.name + ".sha256").write_text(f"{sha256(path)}  {path.name}\n", encoding="ascii")


def main() -> None:
    verification = json.loads((OUTPUT_ROOT / "VERIFICATION_TEST_2.json").read_text(encoding="utf-8"))
    if verification.get("status") != "PASS":
        raise RuntimeError("refuse archive before integrity verification")
    RELEASE_ROOT.mkdir(parents=True, exist_ok=True)
    archive_path = RELEASE_ROOT / ARCHIVE_NAME
    temporary = archive_path.with_suffix(archive_path.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz", compresslevel=6) as archive:
        _add_tree(archive, OUTPUT_ROOT, "artifacts/prospective_test_two")
        _add_tree(archive, Path(__file__).resolve().parent, "src/skyfare/evaluation/prospective_test_two")
        _add_tree(archive, ROOT / "src/skyfare/models", "src/skyfare/models")
        _add_tree(archive, ROOT / "configs", "configs")
    temporary.replace(archive_path)
    manifest = {
        "status": "PASS",
        "archive": ARCHIVE_NAME,
        "archive_sha256": sha256(archive_path),
        "scientific_decision": verification["scientific_decision"],
        "test_2_access": True,
        "runtime_hotfix_used": False,
        "files": [],
    }
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if member.isfile():
                manifest["files"].append({"path": member.name, "size": member.size})
    manifest_path = RELEASE_ROOT / f"{ARCHIVE_NAME}.manifest.json"
    write_json_atomic(manifest_path, manifest)
    _write_sha(archive_path)

    evidence_path = RELEASE_ROOT / EVIDENCE_NAME
    evidence_members = [
        OUTPUT_ROOT / "VERIFICATION_TEST_2.json",
        OUTPUT_ROOT / "analysis" / "TEST_2_REPORT.json",
        OUTPUT_ROOT / "runtime" / "pipeline_status_test2.json",
        OUTPUT_ROOT / "runtime" / "preflight_test2.json",
        OUTPUT_ROOT / "runtime" / "deployable_lock_test2.json",
        OUTPUT_ROOT / "runtime" / "prediction_commit_test2.json",
        OUTPUT_ROOT / "runtime" / "scheduler_test2.json",
        manifest_path,
        RELEASE_ROOT / f"{ARCHIVE_NAME}.sha256",
        OUTPUT_ROOT / "test_two.log",
    ]
    with tarfile.open(evidence_path, "w:gz", compresslevel=9) as archive:
        for path in evidence_members:
            if path.is_file():
                archive.add(path, arcname=path.name)
    _write_sha(evidence_path)
    print(
        f"[TEST2 ARCHIVE PASS] result={archive_path.name} evidence={evidence_path.name} "
        f"decision={verification['scientific_decision']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
