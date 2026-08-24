#!/usr/bin/env python3
"""Select two finalists per task using screen folds only."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from skyfare.models.selection_contract import CANDIDATES, SCREEN_FOLDS, SCREEN_SEEDS, TASKS
from skyfare.models.temporal_runtime import OUTPUT_ROOT, job_root, write_json_atomic


PRIMARY = {
    "CLASSIFICATION": ("proper_score", False),
    "POINT": ("mape", False),
    "DISTRIBUTION": ("wis", False),
    "RANKING": ("ndcg5", True),
}


def main() -> None:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for candidate in CANDIDATES:
        for fold in SCREEN_FOLDS:
            for seed in SCREEN_SEEDS:
                identifier = f"{candidate.candidate_id}__{fold}__S{seed}"
                path = job_root(identifier) / "done.json"
                if not path.is_file():
                    raise RuntimeError(f"missing screen result {identifier}")
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("status") != "COMPLETE":
                    raise RuntimeError(f"incomplete screen result {identifier}")
                grouped[candidate.candidate_id].append(payload)

    table: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        rows = grouped[candidate.candidate_id]
        metric, maximize = PRIMARY[candidate.task]
        values = np.asarray([row["metrics"][metric] for row in rows], dtype=float)
        fold_values = {
            fold: float(np.mean([row["metrics"][metric] for row in rows if row["job"]["fold"] == fold]))
            for fold in SCREEN_FOLDS
        }
        table.append(
            {
                "candidate_id": candidate.candidate_id,
                "task": candidate.task,
                "family": candidate.family,
                "metric": metric,
                "maximize": maximize,
                "mean": float(values.mean()),
                "standard_deviation": float(values.std()),
                "worst_fold": float(min(fold_values.values()) if maximize else max(fold_values.values())),
                "folds": fold_values,
                "score": float(-values.mean() if maximize else values.mean()),
            }
        )
    finalists: dict[str, list[str]] = {}
    for task in TASKS:
        ranked = sorted((row for row in table if row["task"] == task), key=lambda row: (row["score"], row["standard_deviation"], row["candidate_id"]))
        finalists[task] = [str(row["candidate_id"]) for row in ranked[:2]]
    report = {"status": "PASS", "selection_folds": list(SCREEN_FOLDS), "finalists": finalists, "ranking": table}
    destination = OUTPUT_ROOT / "analysis" / "finalists_v24.json"
    write_json_atomic(destination, report)
    print(f"[V24 FINALIST PASS] {json.dumps(finalists, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
