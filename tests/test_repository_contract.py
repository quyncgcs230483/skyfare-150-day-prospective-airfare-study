from __future__ import annotations

import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def inclusive_days(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days + 1


def test_study_blocks_are_contiguous_and_total_150_days() -> None:
    study = json.loads((ROOT / "configs/study.json").read_text(encoding="utf-8"))
    development = study["development"]
    first = study["prospective_tests"]["test_1"]
    second = study["prospective_tests"]["test_2"]
    blocks = (development, first, second)
    assert all(inclusive_days(block["start"], block["end"]) == block["days"] for block in blocks)
    assert sum(block["days"] for block in blocks) == 150
    assert date.fromisoformat(development["end"]).toordinal() + 1 == date.fromisoformat(first["start"]).toordinal()
    assert date.fromisoformat(first["end"]).toordinal() + 1 == date.fromisoformat(second["start"]).toordinal()


def test_frozen_model_registry_contains_all_eight_finalists() -> None:
    payload = json.loads(
        (ROOT / "configs/model_identifier_registry.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "FROZEN_AFTER_DEVELOPMENT"
    assert len(payload["models"]) == 8
    assert {item["task"] for item in payload["models"].values()} == {
        "temporal fare-drop classification",
        "point fare regression",
        "probabilistic fare forecasting",
        "within-query learning-to-rank",
    }


def test_repository_contains_no_markdown_files() -> None:
    assert not [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not any(
            part.startswith(".venv") or part in {".pytest_cache", ".ruff_cache"}
            for part in path.relative_to(ROOT).parts
        )
        and path.suffix.lower() in {".md", ".markdown"}
    ]


def test_release_assets_are_unique_and_sha256_is_well_formed() -> None:
    payload = json.loads((ROOT / "configs/release_assets.json").read_text(encoding="utf-8"))
    names = [item["filename"] for item in payload["assets"]]
    assert len(names) == len(set(names))
    assert all(len(item["sha256"]) == 64 for item in payload["assets"])
