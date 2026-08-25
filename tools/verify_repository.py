#!/usr/bin/env python3
"""Fail closed when repository structure, provenance, or portability drifts."""

from __future__ import annotations

import ast
import hashlib
import json
import stat
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DIRECTORIES = (
    ".github/workflows",
    "artifacts/evidence",
    "artifacts/release_manifests",
    "application/skyfare_inference_demo",
    "configs",
    "data/schema",
    "data/raw/fli",
    "data/raw/google_flights_manual_9g",
    "data/raw/trip_com",
    "experiments",
    "reports/figures",
    "scripts",
    "src/skyfare/acquisition",
    "src/skyfare/preparation",
    "src/skyfare/features",
    "src/skyfare/models",
    "src/skyfare/evaluation",
    "src/skyfare/production",
    "src/skyfare/serving",
    "tests",
    "tools",
)
REQUIRED_EVIDENCE = (
    "artifacts/evidence/development_eda_128_day_summary.json",
    "artifacts/evidence/development_search/verification.json",
    "artifacts/evidence/development_selection/verification.json",
    "artifacts/evidence/prospective_test_one/verification.json",
    "artifacts/evidence/prospective_test_two/verification.json",
    "artifacts/evidence/pooled_evaluation_verification.json",
    "artifacts/evidence/production_refit/verification.json",
    "artifacts/evidence/final_deployment_decision.json",
)
FORBIDDEN_SOURCE_LABELS = ("SERP" + "API_ERA", "TRIP_DAILY" + "_ERA")
FORBIDDEN_CODE_FRAGMENTS = (
    "/work" + "space",
    "/Use" + "rs/",
    "\\Use" + "rs\\",
)
EXCLUDED_PARTS = {".git", ".pytest_cache", ".ruff_cache", "__pycache__"}
EXCLUDED_NAMES = {".DS_Store", ".env"}
EXCLUDED_SUFFIXES = (".joblib", ".keras", ".npz", ".parquet", ".pyc", ".tar.gz")
EXCLUDED_RUNTIME_PREFIXES = (
    ("artifacts", "cache"),
    ("artifacts", "jobs"),
    ("artifacts", "logs"),
    ("artifacts", "models"),
    ("artifacts", "predictions"),
    ("artifacts", "release_assets"),
    ("data", "interim"),
    ("data", "processed"),
)


def _ignored(path: Path) -> bool:
    parts = path.relative_to(ROOT).parts
    generated = any(parts[: len(prefix)] == prefix for prefix in EXCLUDED_RUNTIME_PREFIXES)
    return generated or path.name in EXCLUDED_NAMES or path.name.endswith(EXCLUDED_SUFFIXES) or any(
        part in EXCLUDED_PARTS
        or part.startswith(".venv")
        or part.endswith(".egg-info")
        for part in parts
    )


def _date(value: str) -> date:
    return date.fromisoformat(value)


def _json_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*.json")
        if not _ignored(path)
    )


def _verify_dates() -> None:
    study = json.loads((ROOT / "configs/study.json").read_text(encoding="utf-8"))
    blocks = [study["development"], *study["prospective_tests"].values()]
    for block in blocks:
        observed = (_date(block["end"]) - _date(block["start"])).days + 1
        if observed != block["days"]:
            raise RuntimeError(f"Study-day mismatch: {block}")
    if sum(block["days"] for block in blocks) != 150:
        raise RuntimeError("Development plus prospective blocks do not total 150 days")
    if _date(blocks[0]["end"]).toordinal() + 1 != _date(blocks[1]["start"]).toordinal():
        raise RuntimeError("Test 1 does not immediately follow development")
    if _date(blocks[1]["end"]).toordinal() + 1 != _date(blocks[2]["start"]).toordinal():
        raise RuntimeError("Test 2 does not immediately follow Test 1")


def _verify_evidence() -> None:
    for relative in REQUIRED_EVIDENCE:
        path = ROOT / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") not in {"PASS", "FINAL_LOCK"}:
            raise RuntimeError(f"Evidence is not PASS: {relative}")
    development_eda = json.loads(
        (ROOT / "artifacts/evidence/development_eda_128_day_summary.json").read_text(
            encoding="utf-8"
        )
    )
    scope = development_eda["scope"]
    expected_scope = {
        "raw_context_start": "2026-03-21",
        "registered_development_start": "2026-03-23",
        "development_end": "2026-07-28",
        "calendar_span_days": 130,
        "observed_history_days": 128,
        "uncollected_dates": ["2026-05-08", "2026-05-09"],
        "prospective_labels_used": False,
    }
    if scope != expected_scope or development_eda["observations"] != 923855:
        raise RuntimeError("128-day development EDA scope changed")
    if development_eda["coverage"] != {
        "routes": 20,
        "airlines": 5,
        "supported_route_airline_pairs": 72,
        "canonical_booking_windows": 11,
    }:
        raise RuntimeError("128-day development EDA coverage changed")
    agreement = development_eda["source_agreement"]
    if agreement != {
        "overlap_start": "2026-05-16",
        "overlap_end": "2026-06-03",
        "overlap_days": 19,
        "matched_schedule_minute_slots": 150538,
        "median_trip_to_fli_minimum_fare_ratio": 1.024936061381074,
        "within_5pct_fraction": 0.9361622978915622,
        "within_10pct_fraction": 0.9846351087433073,
        "claim_guard": (
            "Early-period cross-source comparability only; source and time effects remain "
            "non-identifiable and later June/July display semantics are not verified."
        ),
    }:
        raise RuntimeError("Cross-source agreement evidence changed")
    for relative in development_eda["figures"]:
        if not (ROOT / relative).is_file():
            raise RuntimeError(f"Development EDA figure missing: {relative}")


def _verify_models() -> None:
    finalists = json.loads((ROOT / "configs/finalist_models.json").read_text(encoding="utf-8"))
    registry = json.loads((ROOT / "configs/model_identifier_registry.json").read_text(encoding="utf-8"))
    if len(registry["models"]) != 8:
        raise RuntimeError("Finalist identifier registry must contain eight models")
    if set(finalists) != {
        "classification",
        "point_regression",
        "probabilistic_forecasting",
        "learning_to_rank",
        "ensemble_recipe",
        "selection_reopened_after_prospective_tests",
    }:
        raise RuntimeError("Unexpected finalist-model contract keys")
    if finalists["selection_reopened_after_prospective_tests"] is not False:
        raise RuntimeError("Post-test model selection must remain closed")


def _verify_files() -> None:
    for relative in REQUIRED_DIRECTORIES:
        if not (ROOT / relative).is_dir():
            raise RuntimeError(f"Required directory missing: {relative}")
    markdown = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and not _ignored(path)
        and path.suffix.lower() in {".md", ".markdown"}
    ]
    if markdown:
        raise RuntimeError(f"Markdown files are prohibited: {markdown}")
    documents = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*.docx")
        if path.is_file() and not _ignored(path)
    ]
    if documents:
        raise RuntimeError(f"DOCX files are prohibited: {documents}")
    for path in _json_files():
        json.loads(path.read_text(encoding="utf-8"))
    for path in ROOT.rglob("*.py"):
        if not _ignored(path):
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))
            if path.is_relative_to(ROOT / "src") and any(value in source for value in FORBIDDEN_CODE_FRAGMENTS):
                raise RuntimeError(f"Hard-coded host path in source: {path.relative_to(ROOT)}")
    for path in (ROOT / "scripts").glob("*.sh"):
        if not path.stat().st_mode & stat.S_IXUSR:
            raise RuntimeError(f"Script is not executable: {path.name}")
    oversized = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and not _ignored(path) and path.stat().st_size > 100 * 1024 * 1024
    ]
    if oversized:
        raise RuntimeError(f"Files exceed GitHub's 100 MiB object limit: {oversized}")


def _verify_internal_imports() -> None:
    source_root = ROOT / "src"
    unresolved: set[tuple[str, str]] = set()
    for path in [*source_root.rglob("*.py"), *(ROOT / "tests").rglob("*.py")]:
        if _ignored(path):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(
                    alias.name
                    for alias in node.names
                    if alias.name == "skyfare" or alias.name.startswith("skyfare.")
                )
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and (node.module == "skyfare" or node.module.startswith("skyfare."))
            ):
                modules.add(node.module)
        for module in modules:
            candidate = source_root.joinpath(*module.split("."))
            if not candidate.with_suffix(".py").is_file() and not (candidate / "__init__.py").is_file():
                unresolved.add((path.relative_to(ROOT).as_posix(), module))
    if unresolved:
        raise RuntimeError(f"Unresolved internal imports: {sorted(unresolved)}")


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _verify_repository_manifest() -> None:
    manifest_path = ROOT / "artifacts" / "repository_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path != manifest_path and path.is_file() and not path.is_symlink() and not _ignored(path)
    )
    expected = {
        path.relative_to(ROOT).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": _digest(path),
        }
        for path in files
    }
    if payload.get("status") != "PASS" or payload.get("files") != expected:
        raise RuntimeError("Repository manifest does not match publishable file inventory")


def _verify_sources() -> None:
    for path in ROOT.rglob("*"):
        if not path.is_file() or _ignored(path):
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith(("artifacts/evidence/", "artifacts/release_manifests/", "reports/")):
            continue
        if path.suffix.lower() not in {".py", ".json", ".rst", ".txt", ".sh"}:
            continue
        text = path.read_text(encoding="utf-8")
        if any(label in text for label in FORBIDDEN_SOURCE_LABELS):
            raise RuntimeError(f"Legacy source taxonomy remains in publishable source: {relative}")
    figure_source = (ROOT / "src/skyfare/reporting/results_figures.py").read_text(encoding="utf-8")
    if "Fli library" not in figure_source or "CollectionSource.FLI" not in figure_source:
        raise RuntimeError("Result figures do not use canonical Fli source taxonomy")


def _verify_raw_collection() -> None:
    path = ROOT / "data/manifests/raw_collection_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise RuntimeError("Raw collection manifest is not PASS")
    if payload.get("calendar_days") != 157 or payload.get("observed_dates") != 155:
        raise RuntimeError("Raw collection calendar changed")
    if payload.get("missing_dates") != ["2026-05-08", "2026-05-09"]:
        raise RuntimeError("Raw collection missing-date evidence changed")
    if payload.get("pre_study_collection_dates") != ["2026-03-21", "2026-03-22"]:
        raise RuntimeError("Pre-study collection boundary changed")
    if payload.get("post_freeze_serving_dates") != [
        "2026-08-20",
        "2026-08-21",
        "2026-08-22",
        "2026-08-23",
        "2026-08-24",
    ]:
        raise RuntimeError("Post-freeze serving window changed")
    expected = {
        item.relative_to(ROOT).as_posix()
        for directory in (
            ROOT / "data/raw/fli",
            ROOT / "data/raw/google_flights_manual_9g",
            ROOT / "data/raw/trip_com",
            ROOT / "data/raw/collection_issues",
        )
        for item in directory.rglob("*.csv")
    }
    if set(payload.get("files", {})) != expected:
        raise RuntimeError("Raw CSV inventory does not match manifest")
    for relative, metadata in payload["files"].items():
        item = ROOT / relative
        if item.stat().st_size != metadata["bytes"] or _digest(item) != metadata["sha256"]:
            raise RuntimeError(f"Raw CSV integrity failure: {relative}")


def main() -> None:
    _verify_files()
    _verify_internal_imports()
    _verify_repository_manifest()
    _verify_sources()
    _verify_dates()
    _verify_models()
    _verify_evidence()
    _verify_raw_collection()
    experiments = json.loads((ROOT / "experiments/registry.json").read_text(encoding="utf-8"))
    if experiments.get("status") != "COMPLETE_RESEARCH_LINEAGE" or len(experiments.get("experiments", [])) < 35:
        raise RuntimeError("Research-lineage registry is incomplete")
    print(
        f"[REPOSITORY VERIFICATION PASS] json={len(_json_files())} "
        f"experiments={len(experiments['experiments'])}"
    )


if __name__ == "__main__":
    main()
