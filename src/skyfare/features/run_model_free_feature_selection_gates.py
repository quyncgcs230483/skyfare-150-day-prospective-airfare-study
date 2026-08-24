"""Run all new-build model-free Feature Selection gates in one command."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from skyfare.features.audit_common import ROOT


HERE = Path(__file__).resolve().parent
TABLE_DIR = ROOT / "artifacts/feature_research/tables"
LOG_DIR = TABLE_DIR / "execution_logs"


def run(script: str) -> None:
    path = HERE / script
    log_path = LOG_DIR / f"{path.stem}.log"
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/private/tmp/mpl-cache")
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"{script} failed; inspect {log_path}")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run("build_error_analysis_core_figures.py")
    run("build_feature_lineage.py")

    parallel = []
    log_handles = []
    try:
        for script in [
            "identity_continuity_audit.py",
            "listed_price_semantics_audit.py",
        ]:
            handle = (LOG_DIR / f"{Path(script).stem}.log").open("w", encoding="utf-8")
            log_handles.append(handle)
            parallel.append(
                subprocess.Popen(
                    [sys.executable, str(HERE / script)],
                    cwd=ROOT,
                    env={**os.environ, "MPLCONFIGDIR": "/private/tmp/mpl-cache"},
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            )
        for process in parallel:
            if process.wait():
                raise RuntimeError("identity/price-semantics audit failed; inspect execution logs")
    finally:
        for handle in log_handles:
            handle.close()

    run("target_horizon_audit.py")
    run("feature_legality_audit.py")
    run("redundancy_and_adapter_audit.py")
    run("causal_baseline_audit.py")
    run("sequence_structure_audit.py")
    run("build_prefit_contract.py")
    run("build_post_selection_eda.py")

    required = [
        TABLE_DIR / "ea01_identity_continuity_summary.json",
        TABLE_DIR / "ea02_target_horizon_summary.json",
        TABLE_DIR / "ea03_feature_legality_matrix.csv",
        TABLE_DIR / "ea04_redundancy_summary.json",
        TABLE_DIR / "ea05_causal_baseline_summary.json",
        TABLE_DIR / "ea06_listed_price_semantics_summary.json",
        TABLE_DIR / "ea07_sequence_structure_summary.json",
        TABLE_DIR / "fs01_feature_lineage_summary.json",
        TABLE_DIR / "fs07_prefit_contract_manifest.json",
        TABLE_DIR / "branch_eda_manifest.json",
        TABLE_DIR / "feature_selection_criteria.json",
    ]
    manifest = {
        "status": "ALL_MODEL_FREE_FEATURE_SELECTION_GATES_REPRODUCED_NO_MODEL_FIT",
        "created_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
        "model_fit_called": False,
        "uses_archived_110_115_outcomes": False,
        "artifacts": [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in required
        ],
    }
    output = TABLE_DIR / "model_free_gates_execution_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[PASS] {output}")


if __name__ == "__main__":
    main()
