#!/usr/bin/env python3
"""Fail closed when repository structure, provenance, or portability drifts."""

from __future__ import annotations

import ast
import hashlib
import json
import stat
import zipfile
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DIRECTORIES = (
    ".github/workflows",
    "artifacts/evidence",
    "artifacts/release_manifests",
    "configs",
    "data/schema",
    "experiments",
    "reports/figures",
    "reports/manuscript",
    "scripts",
    "src/skyfare/acquisition",
    "src/skyfare/preparation",
    "src/skyfare/features",
    "src/skyfare/models",
    "src/skyfare/evaluation",
    "src/skyfare/production",
    "tests",
    "tools",
)
REQUIRED_EVIDENCE = (
    "artifacts/evidence/development_search/verification.json",
    "artifacts/evidence/development_selection/verification.json",
    "artifacts/evidence/prospective_test_one/verification.json",
    "artifacts/evidence/prospective_test_two/verification.json",
    "artifacts/evidence/pooled_evaluation_verification.json",
    "artifacts/evidence/production_refit/verification.json",
    "artifacts/evidence/final_deployment_decision.json",
)
ALLOWED_LEGACY_SOURCE_FILES = {
    "README.rst",
    "configs/data_sources.json",
    "src/skyfare/core/sources.py",
    "tests/test_sources.py",
    "tools/verify_repository.py",
}
FORBIDDEN_CODE_FRAGMENTS = ("/workspace", "/Users/", "\\Users\\")
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
    ("data", "raw"),
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
                modules.update(alias.name for alias in node.names if alias.name.startswith("skyfare"))
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("skyfare"):
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
        if relative not in ALLOWED_LEGACY_SOURCE_FILES and (
            "SERPAPI_ERA" in text or "TRIP_DAILY_ERA" in text
        ):
            raise RuntimeError(f"Legacy source taxonomy escaped migration boundary: {relative}")
    figure_source = (ROOT / "src/skyfare/reporting/results_figures.py").read_text(encoding="utf-8")
    if "Fli library" not in figure_source or "CollectionSource.FLI" not in figure_source:
        raise RuntimeError("Result figures do not use canonical Fli source taxonomy")


def _verify_manuscript() -> None:
    documents = list((ROOT / "reports/manuscript").glob("*.docx"))
    if len(documents) != 1:
        raise RuntimeError("Exactly one final DOCX manuscript is required")
    with zipfile.ZipFile(documents[0]) as archive:
        required = {"[Content_Types].xml", "word/document.xml", "word/styles.xml"}
        if not required.issubset(archive.namelist()) or archive.testzip() is not None:
            raise RuntimeError("DOCX package integrity failure")


def main() -> None:
    _verify_files()
    _verify_internal_imports()
    _verify_repository_manifest()
    _verify_sources()
    _verify_dates()
    _verify_models()
    _verify_evidence()
    _verify_manuscript()
    experiments = json.loads((ROOT / "experiments/registry.json").read_text(encoding="utf-8"))
    if experiments.get("status") != "COMPLETE_RESEARCH_LINEAGE" or len(experiments.get("experiments", [])) < 35:
        raise RuntimeError("Research-lineage registry is incomplete")
    print(
        f"[REPOSITORY VERIFICATION PASS] json={len(_json_files())} "
        f"experiments={len(experiments['experiments'])}"
    )


if __name__ == "__main__":
    main()
