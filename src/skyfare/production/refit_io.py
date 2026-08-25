"""I/O helpers for 128-day deployment refit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from skyfare.core.paths import DataLayout

LAYOUT = DataLayout.resolve()
ROOT = LAYOUT.root
OUTPUT_ROOT = LAYOUT.artifacts / "development_refit"
INPUT_ROOT = LAYOUT.processed


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_root(task: str, component: str) -> Path:
    return OUTPUT_ROOT / "artifacts" / task / component


def done_path(task: str, component: str) -> Path:
    return artifact_root(task, component) / "done.json"


def load_registry(path: Path, job_id: str) -> dict[str, Any]:
    registry = pd.read_csv(path)
    selected = registry[registry["job_id"].eq(job_id)]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one registry row for {job_id}; found {len(selected)}"
        )
    return selected.iloc[0].to_dict()


def hash_tree(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]
