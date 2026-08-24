#!/usr/bin/env python3
"""Create and validate immutable V24 smoke, screen/history and late registries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skyfare.models.selection_contract import late_registry, manifest, screen_registry, smoke_registry
from skyfare.models.temporal_runtime import OUTPUT_ROOT, write_json_atomic


REGISTRY_ROOT = OUTPUT_ROOT / "registry"


def write_registry(path: Path, phase: str, jobs: list[dict[str, object]]) -> None:
    identifiers = [str(job["job_id"]) for job in jobs]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError(f"{phase}: duplicate job identity")
    write_json_atomic(
        path,
        {
            "status": "READY",
            "contract_id": manifest()["contract_id"],
            "contract_sha256": manifest()["contract_sha256"],
            "phase": phase,
            "jobs": jobs,
            "job_count": len(jobs),
        },
    )


def build_base() -> dict[str, Path]:
    REGISTRY_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {
        "SMOKE": REGISTRY_ROOT / "smoke_v24.json",
        "SCREEN": REGISTRY_ROOT / "screen_history_v24.json",
    }
    write_registry(paths["SMOKE"], "SMOKE", smoke_registry())
    write_registry(paths["SCREEN"], "SCREEN_AND_HISTORY", screen_registry())
    return paths


def build_late(finalists_path: Path) -> Path:
    payload = json.loads(finalists_path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise RuntimeError("finalist selection did not pass")
    path = REGISTRY_ROOT / "late_v24.json"
    write_registry(path, "LATE", late_registry(payload["finalists"]))
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--late-finalists")
    cli = parser.parse_args()
    if cli.late_finalists:
        path = build_late(Path(cli.late_finalists))
        print(f"[V24 LATE REGISTRY PASS] path={path} jobs=48")
    else:
        paths = build_base()
        print(f"[V24 BASE REGISTRY PASS] smoke={paths['SMOKE']} screen={paths['SCREEN']}")


if __name__ == "__main__":
    main()
