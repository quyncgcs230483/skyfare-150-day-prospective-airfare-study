#!/usr/bin/env python3
"""Verify final two-block report and write completion evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    required = [
        "TWO_BLOCK_POOLED_REPORT.json",
        "FINAL_DEPLOYMENT_DECISION.json",
        "TWO_BLOCK_PROVENANCE.json",
        "DAILY_STRATIFIED_DELTAS.csv",
        "POOLED_METRIC_SUMMARY.csv",
        "FINAL_REPORT.txt",
    ]
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"missing outputs={missing}")
    report = json.loads((output / required[0]).read_text(encoding="utf-8"))
    decision = json.loads((output / required[1]).read_text(encoding="utf-8"))
    provenance = json.loads((output / required[2]).read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("scientific_conclusion") != "TWO_BLOCK_CONFIRMED":
        raise RuntimeError("pooled scientific conclusion not PASS")
    if len(report.get("gates", {})) != 10 or not all(report["gates"].values()):
        raise RuntimeError("pooled gates incomplete")
    if decision.get("status") != "FINAL_LOCK" or decision.get("training_required_for_pooled_confirmation") is not False:
        raise RuntimeError("final deployment lock invalid")
    if provenance.get("status") != "PASS":
        raise RuntimeError("provenance not PASS")
    evidence = {
        "status": "PASS",
        "scientific_conclusion": report["scientific_conclusion"],
        "gates": report["gates"],
        "files": {name: {"sha256": sha256(output / name), "size": (output / name).stat().st_size} for name in required},
    }
    path = output / "TWO_BLOCK_VERIFICATION.json"
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[TWO BLOCK VERIFICATION PASS] files={len(required)} gates={len(report['gates'])}")


if __name__ == "__main__":
    main()
