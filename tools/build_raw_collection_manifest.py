#!/usr/bin/env python3
"""Inventory every published raw CSV and the complete collection calendar."""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw"
OUTPUT = ROOT / "data" / "manifests" / "raw_collection_manifest.json"
START = date.fromisoformat("2026-03-21")
END = date.fromisoformat("2026-08-24")
STUDY_START = date.fromisoformat("2026-03-23")
STUDY_END = date.fromisoformat("2026-08-19")
SOURCE_DIRECTORIES = {
    "fli_library": RAW_ROOT / "fli",
    "google_flights_manual_9g": RAW_ROOT / "google_flights_manual_9g",
    "trip_com": RAW_ROOT / "trip_com",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def day_files() -> tuple[dict[str, dict[str, Path]], dict[str, dict[str, object]]]:
    by_source: dict[str, dict[str, Path]] = {}
    files: dict[str, dict[str, object]] = {}
    for source, directory in SOURCE_DIRECTORIES.items():
        source_files: dict[str, Path] = {}
        for path in sorted(directory.glob("????-??-??.csv")):
            observed = date.fromisoformat(path.stem)
            if not START <= observed <= END:
                raise RuntimeError(f"Raw file outside declared collection span: {path}")
            source_files[path.stem] = path
            relative = path.relative_to(ROOT).as_posix()
            files[relative] = {"bytes": path.stat().st_size, "sha256": digest(path)}
        by_source[source] = source_files
    return by_source, files


def main() -> None:
    by_source, files = day_files()
    calendar: list[dict[str, object]] = []
    observed_dates = 0
    current = START
    while current <= END:
        token = current.isoformat()
        sources = [source for source, paths in by_source.items() if token in paths]
        if sources:
            observed_dates += 1
        if current < STUDY_START:
            phase = "pre_study_collection"
        elif current <= STUDY_END:
            phase = "study"
        else:
            phase = "post_freeze_serving"
        calendar.append(
            {
                "date": token,
                "phase": phase,
                "sources": sources,
                "status": "OBSERVED" if sources else "NO_COLLECTION_FILE",
            }
        )
        current += timedelta(days=1)
    issue_files = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (RAW_ROOT / "collection_issues").rglob("*.csv")
    )
    for relative in issue_files:
        path = ROOT / relative
        files[relative] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    payload = {
        "calendar": calendar,
        "calendar_days": len(calendar),
        "files": dict(sorted(files.items())),
        "missing_dates": [item["date"] for item in calendar if item["status"] != "OBSERVED"],
        "observed_dates": observed_dates,
        "pre_study_collection_dates": [
            item["date"] for item in calendar if item["phase"] == "pre_study_collection"
        ],
        "post_freeze_serving_dates": [
            item["date"] for item in calendar if item["phase"] == "post_freeze_serving"
        ],
        "status": "PASS",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"[RAW COLLECTION MANIFEST PASS] files={len(files)} "
        f"observed_dates={observed_dates}/{len(calendar)}"
    )


if __name__ == "__main__":
    main()
