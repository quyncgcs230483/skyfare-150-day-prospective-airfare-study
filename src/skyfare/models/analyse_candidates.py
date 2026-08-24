#!/usr/bin/env python3
"""Screen-selected, late-gated V24 single/ensemble/incumbent tournament."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from skyfare.models.selection_contract import LATE_FOLDS, LATE_SEEDS, QUANTILES, REPORT_FOLDS, SCREEN_FOLDS, SCREEN_SEEDS
from skyfare.evaluation.metrics import classification_metrics, distribution_metrics, point_metrics, ranking_metrics, rearrange
from skyfare.models.temporal_runtime import CONTROL_ROOT, OUTPUT_ROOT, job_root, sha256, write_json_atomic, write_parquet_atomic


CONTROL_FILES = {
    "CLASSIFICATION": "v23_classification_final_oof.parquet",
    "POINT": "v23_regression_final_oof.parquet",
    "DISTRIBUTION": "v22_distribution_oof.parquet",
    "RANKING": "v22_ranking_oof.parquet",
}
PRIMARY = {
    "CLASSIFICATION": ("proper_score", False),
    "POINT": ("mape", False),
    "DISTRIBUTION": ("wis", False),
    "RANKING": ("ndcg5", True),
}


def _finalists() -> dict[str, list[str]]:
    path = OUTPUT_ROOT / "analysis" / "finalists_v24.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise RuntimeError("V24 finalist report not PASS")
    return payload["finalists"]


def _average_candidate(candidate: str, task: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for fold in REPORT_FOLDS:
        seeds = SCREEN_SEEDS if fold in SCREEN_FOLDS else LATE_SEEDS
        seed_frames = [
            pd.read_parquet(job_root(f"{candidate}__{fold}__S{seed}") / "predictions.parquet")
            .sort_values("row_key", kind="stable").reset_index(drop=True)
            for seed in seeds
        ]
        reference = seed_frames[0]["row_key"].to_numpy(dtype=np.uint64)
        if any(not np.array_equal(reference, frame["row_key"].to_numpy(dtype=np.uint64)) for frame in seed_frames[1:]):
            raise RuntimeError(f"seed row identity changed: {candidate}/{fold}")
        if task == "CLASSIFICATION":
            value_columns = ["probability"]
        elif task == "POINT":
            value_columns = ["predicted_price_vnd", "predicted_relative_log"]
        elif task == "DISTRIBUTION":
            value_columns = [f"q{int(level * 100):02d}_relative_log" for level in QUANTILES]
        else:
            value_columns = ["score"]
        base = seed_frames[0].drop(columns=[*value_columns, "seed", "phase"], errors="ignore").copy()
        for column in value_columns:
            base[column] = np.mean(np.stack([frame[column].to_numpy(dtype=float) for frame in seed_frames]), axis=0)
        base["candidate_id"] = candidate
        frames.append(base)
    return pd.concat(frames, ignore_index=True)


def _fold_metric(task: str, frame: pd.DataFrame, value: str | list[str]) -> dict[str, float]:
    if task == "CLASSIFICATION":
        return classification_metrics(frame["target"], frame[str(value)])
    if task == "POINT":
        return point_metrics(frame["target_price_vnd"], frame[str(value)])
    if task == "DISTRIBUTION":
        return distribution_metrics(frame["target_relative_log"], frame[list(value)].to_numpy(dtype=float))
    return ranking_metrics(frame, frame[str(value)].to_numpy(dtype=float))


def _metric_table(task: str, frame: pd.DataFrame, recipes: dict[str, str | list[str]]) -> list[dict[str, object]]:
    primary, maximize = PRIMARY[task]
    rows: list[dict[str, object]] = []
    for recipe, value in recipes.items():
        fold_metrics = {
            fold: _fold_metric(task, frame.loc[frame["fold"].eq(fold)], value)
            for fold in REPORT_FOLDS
        }
        screen = float(np.mean([fold_metrics[fold][primary] for fold in SCREEN_FOLDS]))
        late = float(np.mean([fold_metrics[fold][primary] for fold in LATE_FOLDS]))
        rows.append(
            {
                "recipe": recipe,
                "primary_metric": primary,
                "maximize": maximize,
                "screen": screen,
                "late": late,
                "all": float(np.mean([fold_metrics[fold][primary] for fold in REPORT_FOLDS])),
                "screen_worst_fold": float(
                    min(fold_metrics[fold][primary] for fold in SCREEN_FOLDS)
                    if maximize else max(fold_metrics[fold][primary] for fold in SCREEN_FOLDS)
                ),
                "late_worst_fold": float(
                    min(fold_metrics[fold][primary] for fold in LATE_FOLDS)
                    if maximize else max(fold_metrics[fold][primary] for fold in LATE_FOLDS)
                ),
                "fold_metrics": fold_metrics,
                "selection_score": -screen if maximize else screen,
            }
        )
    return rows


def _late_gate(task: str, challenger: dict[str, object], incumbent: dict[str, object]) -> tuple[bool, dict[str, object]]:
    maximize = bool(challenger["maximize"])
    tolerance = {"CLASSIFICATION": 0.01, "POINT": 0.02, "DISTRIBUTION": 0.03, "RANKING": 0.01}[task]
    if maximize:
        pass_mean = float(challenger["late"]) >= float(incumbent["late"]) - tolerance
        pass_worst = float(challenger["late_worst_fold"]) >= float(incumbent["late_worst_fold"]) - tolerance * 1.5
    else:
        pass_mean = float(challenger["late"]) <= float(incumbent["late"]) * (1.0 + tolerance)
        pass_worst = float(challenger["late_worst_fold"]) <= float(incumbent["late_worst_fold"]) * (1.0 + tolerance * 1.5)
    return bool(pass_mean and pass_worst), {
        "pass_mean": bool(pass_mean),
        "pass_worst_fold": bool(pass_worst),
        "tolerance": tolerance,
        "late_labels_used_only_as_stability_gate": True,
    }


def _classification(candidates: list[str]) -> tuple[pd.DataFrame, dict[str, object]]:
    left, right = (_average_candidate(candidate, "CLASSIFICATION") for candidate in candidates)
    frame = left.rename(columns={"probability": "V24_SINGLE_A"})
    frame = frame.merge(right[["row_key", "fold", "probability"]].rename(columns={"probability": "V24_SINGLE_B"}), on=["row_key", "fold"], validate="one_to_one")
    control = pd.read_parquet(CONTROL_ROOT / CONTROL_FILES["CLASSIFICATION"])[
        ["row_key", "fold", "V17_SYSTEM_BLEND", "V23_SELECTED"]
    ]
    frame = frame.merge(control, on=["row_key", "fold"], validate="one_to_one")
    frame["INCUMBENT"] = frame["V17_SYSTEM_BLEND"]
    frame["V24_MEAN"] = (frame["V24_SINGLE_A"] + frame["V24_SINGLE_B"]) / 2.0
    for weight in (0.25, 0.50, 0.75):
        frame[f"V24_BLEND_{int(weight * 100):02d}"] = weight * frame["V24_MEAN"] + (1.0 - weight) * frame["INCUMBENT"]
    recipes = {name: name for name in ("INCUMBENT", "V23_SELECTED", "V24_SINGLE_A", "V24_SINGLE_B", "V24_MEAN", "V24_BLEND_25", "V24_BLEND_50", "V24_BLEND_75")}
    return frame, _select("CLASSIFICATION", frame, recipes, candidates)


def _point(candidates: list[str]) -> tuple[pd.DataFrame, dict[str, object]]:
    left, right = (_average_candidate(candidate, "POINT") for candidate in candidates)
    frame = left.rename(columns={"predicted_price_vnd": "V24_SINGLE_A"})
    frame = frame.merge(right[["row_key", "fold", "predicted_price_vnd"]].rename(columns={"predicted_price_vnd": "V24_SINGLE_B"}), on=["row_key", "fold"], validate="one_to_one")
    control = pd.read_parquet(CONTROL_ROOT / CONTROL_FILES["POINT"])[
        ["target_offer_id", "fold", "V19_REG_ENSEMBLE", "V23_SELECTED"]
    ]
    frame = frame.merge(control, on=["target_offer_id", "fold"], validate="one_to_one")
    frame["INCUMBENT"] = frame["V19_REG_ENSEMBLE"]
    frame["V24_MEAN"] = (frame["V24_SINGLE_A"] + frame["V24_SINGLE_B"]) / 2.0
    for weight in (0.25, 0.50, 0.75):
        frame[f"V24_BLEND_{int(weight * 100):02d}"] = weight * frame["V24_MEAN"] + (1.0 - weight) * frame["INCUMBENT"]
    recipes = {name: name for name in ("INCUMBENT", "V23_SELECTED", "V24_SINGLE_A", "V24_SINGLE_B", "V24_MEAN", "V24_BLEND_25", "V24_BLEND_50", "V24_BLEND_75")}
    return frame, _select("POINT", frame, recipes, candidates)


def _distribution(candidates: list[str]) -> tuple[pd.DataFrame, dict[str, object]]:
    left, right = (_average_candidate(candidate, "DISTRIBUTION") for candidate in candidates)
    raw_columns = [f"q{int(level * 100):02d}_relative_log" for level in QUANTILES]
    left_names = [f"V24_A_{column}" for column in raw_columns]
    right_names = [f"V24_B_{column}" for column in raw_columns]
    frame = left.rename(columns=dict(zip(raw_columns, left_names)))
    right_small = right[["row_key", "fold", *raw_columns]].rename(columns=dict(zip(raw_columns, right_names)))
    frame = frame.merge(right_small, on=["row_key", "fold"], validate="one_to_one")
    control_columns = [f"v16_interval_control_q{int(level * 100):02d}" for level in QUANTILES]
    v22_columns = [f"v22_selected_q{int(level * 100):02d}" for level in QUANTILES]
    control = pd.read_parquet(CONTROL_ROOT / CONTROL_FILES["DISTRIBUTION"])[
        ["row_key", "fold", *control_columns, *v22_columns]
    ]
    frame = frame.merge(control, on=["row_key", "fold"], validate="one_to_one")
    incumbent_names, v22_names, mean_names = [], [], []
    for index, level in enumerate(QUANTILES):
        token = f"q{int(level * 100):02d}"
        incumbent, v22, mean = f"INCUMBENT_{token}", f"V22_SELECTED_{token}", f"V24_MEAN_{token}"
        frame[incumbent] = frame[control_columns[index]]
        frame[v22] = frame[v22_columns[index]]
        frame[mean] = (frame[left_names[index]] + frame[right_names[index]]) / 2.0
        incumbent_names.append(incumbent); v22_names.append(v22); mean_names.append(mean)
    recipe_columns: dict[str, list[str]] = {
        "INCUMBENT": incumbent_names,
        "V22_SELECTED": v22_names,
        "V24_SINGLE_A": left_names,
        "V24_SINGLE_B": right_names,
        "V24_MEAN": mean_names,
    }
    for weight in (0.25, 0.50, 0.75):
        names = []
        for index, level in enumerate(QUANTILES):
            name = f"V24_BLEND_{int(weight * 100):02d}_q{int(level * 100):02d}"
            frame[name] = weight * frame[mean_names[index]] + (1.0 - weight) * frame[incumbent_names[index]]
            names.append(name)
        recipe_columns[f"V24_BLEND_{int(weight * 100):02d}"] = names
    return frame, _select("DISTRIBUTION", frame, recipe_columns, candidates)


def _ranking(candidates: list[str]) -> tuple[pd.DataFrame, dict[str, object]]:
    left, right = (_average_candidate(candidate, "RANKING") for candidate in candidates)
    frame = left.rename(columns={"score": "V24_SINGLE_A"})
    frame = frame.merge(right[["row_key", "fold", "score"]].rename(columns={"score": "V24_SINGLE_B"}), on=["row_key", "fold"], validate="one_to_one")
    control = pd.read_parquet(CONTROL_ROOT / CONTROL_FILES["RANKING"])[
        ["row_key", "fold", "v22_rank_selected", "v22_rank_ensemble"]
    ]
    frame = frame.merge(control, on=["row_key", "fold"], validate="one_to_one")
    frame["INCUMBENT"] = frame["v22_rank_selected"]
    frame["V24_MEAN"] = (frame["V24_SINGLE_A"] + frame["V24_SINGLE_B"]) / 2.0
    for weight in (0.25, 0.50, 0.75):
        frame[f"V24_BLEND_{int(weight * 100):02d}"] = weight * frame["V24_MEAN"] + (1.0 - weight) * frame["INCUMBENT"]
    recipes = {name: name for name in ("INCUMBENT", "v22_rank_ensemble", "V24_SINGLE_A", "V24_SINGLE_B", "V24_MEAN", "V24_BLEND_25", "V24_BLEND_50", "V24_BLEND_75")}
    return frame, _select("RANKING", frame, recipes, candidates)


def _select(task: str, frame: pd.DataFrame, recipes: dict[str, str | list[str]], candidates: list[str]) -> dict[str, object]:
    table = _metric_table(task, frame, recipes)
    ranked = sorted(table, key=lambda row: (row["selection_score"], row["recipe"]))
    challenger = ranked[0]
    incumbent = next(row for row in table if row["recipe"] == "INCUMBENT")
    if challenger["recipe"] == "INCUMBENT":
        frozen = incumbent
        gate = {"status": "INCUMBENT_WON_SCREEN", "pass_mean": True, "pass_worst_fold": True}
    else:
        passed, gate = _late_gate(task, challenger, incumbent)
        gate["status"] = "PASS" if passed else "RETAIN_INCUMBENT"
        frozen = challenger if passed else incumbent
    return {
        "status": "PASS",
        "task": task,
        "finalists": candidates,
        "screen_winner": challenger["recipe"],
        "frozen_recipe": frozen["recipe"],
        "late_gate": gate,
        "recipes": table,
        "recipe_columns": recipes,
        "test_access": False,
    }


def main() -> None:
    finalists = _finalists()
    builders = {
        "CLASSIFICATION": _classification,
        "POINT": _point,
        "DISTRIBUTION": _distribution,
        "RANKING": _ranking,
    }
    reports: dict[str, dict[str, object]] = {}
    outputs: dict[str, str] = {}
    analysis = OUTPUT_ROOT / "analysis"
    for task, builder in builders.items():
        frame, report = builder(finalists[task])
        columns = report["recipe_columns"][report["frozen_recipe"]]
        if isinstance(columns, str):
            frame["V24_FROZEN"] = frame[columns]
        else:
            values = rearrange(frame[columns].to_numpy(dtype=float))
            for index, level in enumerate(QUANTILES):
                frame[f"V24_FROZEN_q{int(level * 100):02d}"] = values[:, index]
        path = analysis / f"{task.lower()}_final_oof_v24.parquet"
        write_parquet_atomic(path, frame)
        outputs[task] = str(path.relative_to(OUTPUT_ROOT))
        report["oof_sha256"] = sha256(path)
        reports[task] = report
        write_json_atomic(analysis / f"{task.lower()}_selection_v24.json", report)
    summary = {
        "status": "PASS",
        "test_access": False,
        "screen_folds": list(SCREEN_FOLDS),
        "late_folds": list(LATE_FOLDS),
        "outputs": outputs,
        "tasks": {task: {"finalists": reports[task]["finalists"], "frozen_recipe": reports[task]["frozen_recipe"], "late_gate": reports[task]["late_gate"]} for task in reports},
    }
    write_json_atomic(analysis / "research_report_v24.json", summary)
    print(f"[V24 ANALYSIS PASS] {json.dumps(summary['tasks'], sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
