#!/usr/bin/env python3
"""Freeze BUY/WAIT threshold using screen folds; audit late folds and friction."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from skyfare.models.selection_contract import LATE_FOLDS, SCREEN_FOLDS
from skyfare.models.temporal_runtime import OUTPUT_ROOT, write_json_atomic, write_parquet_atomic


THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.20, 0.81, 0.05))
FRICTIONS_VND = (0, 25_000, 50_000, 100_000)
NOMINAL_FRICTION_VND = 50_000


def evaluate(frame: pd.DataFrame, threshold: float | None, friction: int, action: str | None = None) -> dict[str, float]:
    current = frame["price_vnd"].to_numpy(dtype=float)
    future = frame["target_price_vnd"].to_numpy(dtype=float)
    if action == "BUY":
        wait = np.zeros(len(frame), dtype=bool)
    elif action == "WAIT":
        wait = np.ones(len(frame), dtype=bool)
    else:
        wait = frame["V24_FROZEN"].to_numpy(dtype=float) >= float(threshold)
    oracle = np.minimum(current, future + friction)
    paid = np.where(wait, future + friction, current)
    regret = np.maximum(paid - oracle, 0.0)
    cutoff = np.quantile(regret, 0.95)
    tail = regret[regret >= cutoff]
    return {
        "mean_regret_vnd": float(regret.mean()),
        "cvar95_regret_vnd": float(tail.mean()) if len(tail) else 0.0,
        "p95_regret_vnd": float(cutoff),
        "wait_rate": float(wait.mean()),
        "oracle_savings_capture": float(1.0 - regret.sum() / max(np.maximum(current - oracle, 0.0).sum(), 1.0)),
        "rows": float(len(frame)),
    }


def main() -> None:
    path = OUTPUT_ROOT / "analysis" / "classification_final_oof_v24.parquet"
    frame = pd.read_parquet(path)
    rows: list[dict[str, object]] = []
    for threshold in THRESHOLDS:
        fold_metrics = {
            fold: evaluate(frame.loc[frame["fold"].eq(fold)], threshold, NOMINAL_FRICTION_VND)
            for fold in (*SCREEN_FOLDS, *LATE_FOLDS)
        }
        screen_mean = float(np.mean([fold_metrics[fold]["mean_regret_vnd"] for fold in SCREEN_FOLDS]))
        screen_cvar = float(np.mean([fold_metrics[fold]["cvar95_regret_vnd"] for fold in SCREEN_FOLDS]))
        screen_worst = float(max(fold_metrics[fold]["mean_regret_vnd"] for fold in SCREEN_FOLDS))
        rows.append(
            {
                "threshold": threshold,
                "screen_mean_regret_vnd": screen_mean,
                "screen_cvar95_regret_vnd": screen_cvar,
                "screen_worst_fold_regret_vnd": screen_worst,
                "selection_score": screen_mean + 0.20 * screen_cvar + 0.10 * screen_worst,
                "fold_metrics": fold_metrics,
            }
        )
    selected = min(rows, key=lambda row: (row["selection_score"], abs(float(row["threshold"]) - 0.5)))
    threshold = float(selected["threshold"])
    stress: list[dict[str, object]] = []
    for friction in FRICTIONS_VND:
        for fold_set, folds in (("SCREEN", SCREEN_FOLDS), ("LATE", LATE_FOLDS)):
            subset = frame.loc[frame["fold"].isin(folds)]
            stress.append({"friction_vnd": friction, "fold_set": fold_set, "policy": "V24_FROZEN", **evaluate(subset, threshold, friction)})
            stress.append({"friction_vnd": friction, "fold_set": fold_set, "policy": "BUY_ALL", **evaluate(subset, None, friction, "BUY")})
            stress.append({"friction_vnd": friction, "fold_set": fold_set, "policy": "WAIT_ALL", **evaluate(subset, None, friction, "WAIT")})
    stress_frame = pd.DataFrame(stress)
    write_parquet_atomic(OUTPUT_ROOT / "analysis" / "buy_wait_policy_stress_v24.parquet", stress_frame)
    report = {
        "status": "PASS",
        "policy": "WAIT when calibrated P(DROP_5PCT) >= threshold; otherwise BUY",
        "threshold": threshold,
        "nominal_friction_vnd": NOMINAL_FRICTION_VND,
        "fallback_on_missing_prediction": "BUY",
        "selection_folds": list(SCREEN_FOLDS),
        "late_folds_used_for_stability_audit_only": list(LATE_FOLDS),
        "threshold_tournament": rows,
        "friction_stress": stress,
        "test_access": False,
    }
    write_json_atomic(OUTPUT_ROOT / "analysis" / "buy_wait_policy_v24.json", report)
    print(f"[V24 POLICY PASS] threshold={threshold:.2f} fallback=BUY friction_stress={list(FRICTIONS_VND)}", flush=True)


if __name__ == "__main__":
    main()
