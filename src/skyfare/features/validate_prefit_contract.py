"""Rehearse the leakage-controlled prefit contract without fitting an estimator.

This is the last structural gate before training. It resolves the registered
feature lists against the candidate frame, reconstructs multi-horizon and exact-next labels,
and audits rolling-origin train/validation membership and label-time purge.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from skyfare.features.audit_common import CUTOFF, ROOT, next_window_map
from skyfare.features.candidate_feature_contract import build_candidate_frame


TABLE_DIR = ROOT / "artifacts/feature_research/tables"
REPORT = ROOT / "artifacts/feature_research/reports/PREFIT_VALIDATION.txt"
ALLOWED_SYNTHETIC_FEATURES = {"target_dud"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def physical_entity(frame: pd.DataFrame) -> pd.Series:
    return pd.util.hash_pandas_object(
        frame[["collection_era", "schedule_slot_id", "session_label"]], index=False
    ).astype("uint64")


def build_future_pairs(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    work["entity_band"] = physical_entity(work)
    target = work[
        ["entity_band", "days_until_departure", "price_vnd", "feature_time", "session_date"]
    ].copy()
    if target.duplicated(["entity_band", "days_until_departure"]).any():
        raise RuntimeError("Exact-next target key is not unique")
    target = target.rename(
        columns={
            "days_until_departure": "target_dud",
            "price_vnd": "target_price_vnd",
            "feature_time": "label_time",
            "session_date": "label_session_date",
        }
    )
    pieces: list[pd.DataFrame] = []
    duds = sorted(work["days_until_departure"].dropna().astype(int).unique(), reverse=True)
    for current_dud in duds:
        current = work.loc[
            work["days_until_departure"].eq(current_dud),
            [
                "entity_band",
                "schedule_slot_id",
                "session_date",
                "feature_time",
                "flight_date",
                "price_vnd",
                "has_same_schedule_history",
            ],
        ].copy()
        for target_dud in [value for value in duds if value < current_dud]:
            pairs = current.merge(
                target[target["target_dud"].eq(target_dud)],
                on="entity_band",
                how="left",
                validate="one_to_one",
            )
            expected_target_date = pairs["flight_date"].dt.normalize() - pd.to_timedelta(
                target_dud, unit="D"
            )
            valid = (
                pairs["target_price_vnd"].notna()
                & pairs["label_time"].gt(pairs["feature_time"])
                & pairs["label_session_date"].eq(expected_target_date)
            )
            pairs = pairs.loc[valid].copy()
            if pairs.empty:
                continue
            pairs["current_dud"] = current_dud
            pairs["target_dud"] = target_dud
            pairs["horizon_gap_days"] = current_dud - target_dud
            pairs["is_exact_next"] = next_window_map().get(current_dud) == target_dud
            pieces.append(pairs)
    multi = pd.concat(pieces, ignore_index=True)
    exact = multi[multi["is_exact_next"]].copy()
    exact["will_drop_exact_next"] = (
        exact["target_price_vnd"] <= 0.95 * exact["price_vnd"]
    ).astype("int8")
    if exact.empty:
        raise RuntimeError("No exact-next labels survived the maturity contract")
    if not exact["label_time"].gt(exact["feature_time"]).all():
        raise RuntimeError("Exact-next labels are not strictly future")
    if not multi["label_time"].gt(multi["feature_time"]).all():
        raise RuntimeError("Multi-horizon labels are not strictly future")
    return multi, exact


def build_dud1_intraday(frame: pd.DataFrame) -> pd.DataFrame:
    """Match DUD1 AM offers to same-slot PM observations on same date."""

    keys = ["collection_era", "schedule_slot_id", "session_date", "days_until_departure"]
    current = frame[
        frame["days_until_departure"].eq(1)
        & frame["session_label"].astype("string").eq("AM")
    ][keys + ["feature_time", "price_vnd", "has_same_schedule_history"]].copy()
    current["current_row_id"] = pd.util.hash_pandas_object(
        current[keys + ["feature_time"]], index=False
    ).astype("uint64")
    target = frame[
        frame["days_until_departure"].eq(1)
        & frame["session_label"].astype("string").eq("PM")
    ][keys + ["feature_time", "price_vnd"]].rename(
        columns={"feature_time": "label_time", "price_vnd": "target_price_vnd"}
    )
    pairs = current.merge(target, on=keys, how="left", validate="one_to_many")
    pairs = pairs[
        pairs["target_price_vnd"].notna()
        & pairs["label_time"].gt(pairs["feature_time"])
    ].copy()
    pairs = (
        pairs.sort_values(["current_row_id", "label_time"])
        .drop_duplicates("current_row_id", keep="first")
        .reset_index(drop=True)
    )
    pairs["horizon_gap_hours"] = (
        pairs["label_time"] - pairs["feature_time"]
    ).dt.total_seconds() / 3600
    pairs["will_drop_intraday"] = (
        pairs["target_price_vnd"] <= 0.95 * pairs["price_vnd"]
    ).astype("int8")
    if pairs.empty or not pairs["label_time"].gt(pairs["feature_time"]).all():
        raise RuntimeError("DUD1 intraday target is empty or not strictly future")
    return pairs


def verify_registered_features(frame: pd.DataFrame, registry: pd.DataFrame) -> dict[str, object]:
    stage1 = registry[registry["stage"].eq("STAGE1_FAMILY_SCREEN")].copy()
    missing: dict[str, list[str]] = {}
    for row in stage1.itertuples(index=False):
        features = [value.strip() for value in row.resolved_features.split("|")]
        unknown = sorted(set(features) - set(frame.columns) - ALLOWED_SYNTHETIC_FEATURES)
        if unknown:
            missing[row.candidate_id] = unknown
        if len(features) != int(row.resolved_feature_count):
            raise RuntimeError(f"Resolved feature count mismatch: {row.candidate_id}")
    if missing:
        raise RuntimeError(f"Registered features are absent from candidate frame: {missing}")

    for (_, _, adapter), group in stage1.groupby(["branch", "model_family", "adapter"], observed=True):
        counts = group.set_index("feature_variant")["resolved_feature_count"].to_dict()
        ordered = [
            counts["STATIC_CURRENT_STATE"],
            counts["PLUS_MARKET_LEVEL_CONTEXT"],
            counts["PLUS_PRIOR_MARKET_DIRECTION"],
        ]
        if not ordered[0] < ordered[1] < ordered[2]:
            raise RuntimeError(f"Feature-family screen is not strictly incremental: {adapter}")

    h0_adapters = pd.read_csv(TABLE_DIR / "ea04_model_family_adapters.csv")
    h0 = h0_adapters[h0_adapters["branch"].eq("Relative h=0")]
    illegal_h0 = []
    for row in h0.itertuples(index=False):
        features = {value.strip() for value in row.features.split("|")}
        illegal = sorted(features & {"price_vnd", "current_vs_competitor_ratio"})
        if illegal:
            illegal_h0.append({"adapter": row.adapter, "illegal": illegal})
    if illegal_h0:
        raise RuntimeError(f"Current target leaks into h0 adapters: {illegal_h0}")

    return {
        "stage1_tabular_candidates": int(len(stage1)),
        "all_resolved_features_exist": True,
        "incremental_feature_counts_strict": True,
        "h0_current_target_leakage_violations": 0,
    }


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    required = [
        ROOT / "artifacts/feature_research/reports/PREDICTION_CONTRACT_V2.txt",
        TABLE_DIR / "model_free_gates_execution_manifest.json",
        TABLE_DIR / "fs07_prefit_contract_manifest.json",
        TABLE_DIR / "prefit_training_manifest.json",
        TABLE_DIR / "prefit_training_acceptance_contract.json",
        TABLE_DIR / "prefit_candidate_registry.csv",
        TABLE_DIR / "ea04_redundancy_summary.json",
        TABLE_DIR / "ea04_model_family_adapters.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Preflight inputs are missing: {missing}")

    gate = json.loads(required[1].read_text())
    prefit = json.loads(required[3].read_text())
    contract = json.loads(required[4].read_text())
    redundancy = json.loads(required[6].read_text())
    if gate["status"] != "ALL_MODEL_FREE_FEATURE_SELECTION_GATES_REPRODUCED_NO_MODEL_FIT":
        raise RuntimeError("Model-free gates are not complete")
    if gate["model_fit_called"] or gate["uses_archived_110_115_outcomes"]:
        raise RuntimeError("Model-free evidence is contaminated")
    if prefit["model_fit_called"] or prefit["calibrator_fit_called"]:
        raise RuntimeError("Pre-fit contract already reports a fit")
    if contract["status"] != "AUTHORITATIVE_V3_BEFORE_FIRST_PILOT_FIT":
        raise RuntimeError("Pilot training contract is not locked")
    if redundancy["cross_collection_era_history_violations"] != 0:
        raise RuntimeError("History crosses collection era")
    if redundancy["cross_collection_era_market_anchor_violations"] != 0:
        raise RuntimeError("Market anchor crosses collection era")

    registry = pd.read_csv(required[5])
    if registry["candidate_id"].duplicated().any():
        raise RuntimeError("Candidate registry contains duplicate ids")
    if int(contract["candidate_rows"]) != len(registry):
        raise RuntimeError("Candidate count differs from locked contract")

    frame = build_candidate_frame()
    pilot = frame[
        frame["session_date"].le(CUTOFF.normalize())
        & frame["collection_era"].astype("string").eq("TRIP_COM_BROWSER_ERA")
    ].copy()
    if pilot.empty or pilot["session_date"].max() != CUTOFF.normalize():
        raise RuntimeError("Trip pilot does not end exactly on the locked cutoff")
    if pilot["session_date"].min() != pd.Timestamp(contract["source_contract"]["primary_start"]):
        raise RuntimeError("Trip pilot does not start on the locked primary date")
    if pilot["collection_era"].astype("string").nunique() != 1:
        raise RuntimeError("Primary pilot contains more than one collection era")

    history = pilot["previous_schedule_time"].notna()
    market = pilot["temporal_market_time"].notna()
    history_time_violations = int(
        pilot.loc[history, "previous_schedule_time"].ge(pilot.loc[history, "feature_time"]).sum()
    )
    current_batch_history_violations = int(
        pilot.loc[history, "previous_schedule_session_key"]
        .eq(pilot.loc[history, "session_key"])
        .sum()
    )
    market_time_violations = int(
        pilot.loc[market, "temporal_market_time"].ge(pilot.loc[market, "feature_time"]).sum()
    )
    cross_era_history = int(
        pilot.loc[history, "previous_schedule_collection_era"]
        .ne(pilot.loc[history, "collection_era"])
        .sum()
    )
    cross_era_market = int(
        pilot.loc[market, "temporal_market_collection_era"]
        .ne(pilot.loc[market, "collection_era"])
        .sum()
    )
    if any([
        history_time_violations,
        current_batch_history_violations,
        market_time_violations,
        cross_era_history,
        cross_era_market,
    ]):
        raise RuntimeError("Point-in-time or collection-era preflight failed")

    feature_audit = verify_registered_features(pilot, registry)
    multi, exact = build_future_pairs(pilot)
    intraday = build_dud1_intraday(pilot)

    fold_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    multi_transition_rows: list[dict[str, object]] = []
    exact_transitions_by_fold: dict[str, set[tuple[int, int]]] = {}
    multi_transitions_by_fold: dict[str, set[tuple[int, int]]] = {}
    all_blocks = [
        {**spec, "phase": "SELECTION_OOF", "name": spec["fold"]}
        for spec in contract["validation"]["selection_folds"]
    ] + [
        {**spec, "phase": spec["role"], "name": spec["block"]}
        for spec in contract["validation"]["confirmation_blocks"]
    ]
    for spec in all_blocks:
        start = pd.Timestamp(spec["validation_start"])
        end = pd.Timestamp(spec["validation_end"])
        h0_train = pilot[pilot["feature_time"].lt(start)]
        h0_valid = pilot[pilot["session_date"].between(start, end)]
        multi_train = multi[multi["label_time"].lt(start)]
        multi_valid = multi[multi["session_date"].between(start, end)]
        exact_train = exact[exact["label_time"].lt(start)]
        exact_valid = exact[exact["session_date"].between(start, end)]
        intraday_train = intraday[intraday["label_time"].lt(start)]
        intraday_valid = intraday[intraday["session_date"].between(start, end)]
        if any(part.empty for part in [h0_train, h0_valid, multi_train, multi_valid, exact_train, exact_valid]):
            raise RuntimeError(f"Empty structural partition in {spec['name']}")
        if multi_train["label_time"].max() >= start:
            raise RuntimeError(f"Label-time purge failed in {spec['name']}")
        if h0_train["feature_time"].max() >= start:
            raise RuntimeError(f"Feature-time split failed in {spec['name']}")
        transition_counts = multi_train.groupby(["current_dud", "target_dud"]).size()
        supported = int((transition_counts >= contract["multi_horizon_training"]["minimum_training_rows_per_transition"]).sum())
        fold_transition_set: set[tuple[int, int]] = set()
        for (current_dud, target_dud), group in exact_valid.groupby(
            ["current_dud", "target_dud"], observed=True
        ):
            transition = (int(current_dud), int(target_dud))
            fold_transition_set.add(transition)
            transition_rows.append(
                {
                    "block": spec["name"],
                    "phase": spec["phase"],
                    "current_dud": transition[0],
                    "target_dud": transition[1],
                    "validation_rows": len(group),
                    "schedule_slots": group["schedule_slot_id"].nunique(),
                    "warm_rows": int(group["has_same_schedule_history"].sum()),
                    "cold_rows": int(group["has_same_schedule_history"].eq(0).sum()),
                    "warm_pct": 100 * group["has_same_schedule_history"].mean(),
                    "wait_prevalence_pct": 100 * group["will_drop_exact_next"].mean(),
                }
            )
        exact_transitions_by_fold[spec["name"]] = fold_transition_set
        multi_transition_set: set[tuple[int, int]] = set()
        for (current_dud, target_dud), group in multi_valid.groupby(
            ["current_dud", "target_dud"], observed=True
        ):
            transition = (int(current_dud), int(target_dud))
            multi_transition_set.add(transition)
            multi_transition_rows.append(
                {
                    "block": spec["name"],
                    "phase": spec["phase"],
                    "current_dud": transition[0],
                    "target_dud": transition[1],
                    "validation_rows": len(group),
                    "schedule_slots": group["schedule_slot_id"].nunique(),
                    "warm_rows": int(group["has_same_schedule_history"].sum()),
                    "cold_rows": int(group["has_same_schedule_history"].eq(0).sum()),
                }
            )
        multi_transitions_by_fold[spec["name"]] = multi_transition_set
        fold_rows.append(
            {
                "block": spec["name"],
                "phase": spec["phase"],
                "validation_start": start.date(),
                "validation_end": end.date(),
                "h0_train_rows": len(h0_train),
                "h0_validation_rows": len(h0_valid),
                "multi_horizon_train_rows_after_label_time_purge": len(multi_train),
                "multi_horizon_validation_rows": len(multi_valid),
                "supported_transition_cells": supported,
                "total_observed_transition_cells": int(len(transition_counts)),
                "exact_next_train_rows_after_label_time_purge": len(exact_train),
                "exact_next_validation_rows": len(exact_valid),
                "exact_next_validation_schedule_slots": exact_valid["schedule_slot_id"].nunique(),
                "exact_next_validation_warm_pct": 100
                * exact_valid["has_same_schedule_history"].mean(),
                "willdrop_validation_prevalence_pct": 100
                * exact_valid["will_drop_exact_next"].mean(),
                "dud1_intraday_train_rows_after_label_time_purge": len(intraday_train),
                "dud1_intraday_validation_rows": len(intraday_valid),
                "dud1_intraday_wait_prevalence_pct": 100
                * intraday_valid["will_drop_intraday"].mean()
                if len(intraday_valid)
                else np.nan,
                "max_training_label_time": multi_train["label_time"].max(),
                "min_validation_feature_time": multi_valid["feature_time"].min(),
                "label_time_purge_pass": bool(multi_train["label_time"].max() < start),
            }
        )
    folds = pd.DataFrame(fold_rows)
    selection_names = [spec["fold"] for spec in contract["validation"]["selection_folds"]]
    confirmation_names = [spec["block"] for spec in contract["validation"]["confirmation_blocks"]]
    selection_common = sorted(set.intersection(*(exact_transitions_by_fold[name] for name in selection_names)))
    confirmation_common = sorted(
        set.intersection(*(exact_transitions_by_fold[name] for name in confirmation_names))
    )
    if len(selection_common) != 10 or len(confirmation_common) != 6:
        raise RuntimeError(
            f"Unexpected comparable transitions: selection={selection_common}, confirmation={confirmation_common}"
        )
    selection_common_multi = sorted(
        set.intersection(*(multi_transitions_by_fold[name] for name in selection_names))
    )
    confirmation_common_multi = sorted(
        set.intersection(*(multi_transitions_by_fold[name] for name in confirmation_names))
    )

    def parse_locked(values: list[str]) -> list[tuple[int, int]]:
        return sorted(tuple(map(int, value.split("->"))) for value in values)

    locked_selection_exact = parse_locked(
        contract["validation"]["selection_common_exact_transitions"]
    )
    locked_confirmation_exact = parse_locked(
        contract["validation"]["confirmation_common_exact_transitions"]
    )
    locked_selection_multi = parse_locked(
        contract["validation"]["selection_common_multi_horizon_transitions"]
    )
    locked_confirmation_multi = parse_locked(
        contract["validation"]["confirmation_common_multi_horizon_transitions"]
    )
    if selection_common != locked_selection_exact or confirmation_common != locked_confirmation_exact:
        raise RuntimeError("Locked exact-next comparable-transition lists do not match the data")
    if selection_common_multi != locked_selection_multi or confirmation_common_multi != locked_confirmation_multi:
        raise RuntimeError(
            "Locked multi-horizon comparable-transition lists do not match the data: "
            f"selection={selection_common_multi}, confirmation={confirmation_common_multi}"
        )
    transition_frame = pd.DataFrame(transition_rows)
    transition_frame["transition_key"] = list(
        zip(transition_frame["current_dud"], transition_frame["target_dud"])
    )
    transition_frame["common_in_selection"] = transition_frame["transition_key"].isin(selection_common)
    transition_frame["common_in_confirmation"] = transition_frame["transition_key"].isin(confirmation_common)
    transition_frame = transition_frame.drop(columns="transition_key")
    folds_path = TABLE_DIR / "prefit_temporal_fold_structure.csv"
    folds.to_csv(folds_path, index=False)
    transitions_path = TABLE_DIR / "prefit_transition_structure.csv"
    transition_frame.to_csv(transitions_path, index=False)
    multi_transition_frame = pd.DataFrame(multi_transition_rows)
    multi_transition_frame["transition_key"] = list(
        zip(multi_transition_frame["current_dud"], multi_transition_frame["target_dud"])
    )
    multi_transition_frame["common_in_selection"] = multi_transition_frame[
        "transition_key"
    ].isin(selection_common_multi)
    multi_transition_frame["common_in_confirmation"] = multi_transition_frame[
        "transition_key"
    ].isin(confirmation_common_multi)
    multi_transition_frame = multi_transition_frame.drop(columns="transition_key")
    multi_transitions_path = (
        TABLE_DIR / "prefit_multi_horizon_transition_structure.csv"
    )
    multi_transition_frame.to_csv(multi_transitions_path, index=False)

    stage1_fit_rows = int(
        registry["stage"].isin(["STAGE1_FAMILY_SCREEN", "STAGE1_SEQUENCE_SCREEN"]).sum()
    )
    expected_fits = stage1_fit_rows * len(contract["validation"]["selection_folds"])
    if expected_fits != int(contract["stage1_actual_model_fits"]):
        raise RuntimeError("Locked Stage-1 fit accounting is inconsistent")

    now = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat()
    report_data = {
        "status": "LEAKAGE_CONTROLLED_PREFIT_VALIDATION_PASS",
        "created_at": now,
        "cutoff": str(CUTOFF.date()),
        "primary_source": "TRIP_COM_BROWSER_ERA",
        "pilot_rows": len(pilot),
        "pilot_batches": int(pilot["session_key"].nunique()),
        "exact_next_labeled_rows": len(exact),
        "multi_horizon_labeled_rows": len(multi),
        "dud1_intraday_labeled_rows": len(intraday),
        "selection_common_exact_next_transitions": [
            {"current_dud": current, "target_dud": target}
            for current, target in selection_common
        ],
        "confirmation_common_exact_next_transitions": [
            {"current_dud": current, "target_dud": target}
            for current, target in confirmation_common
        ],
        "selection_common_multi_horizon_transitions": [
            {"current_dud": current, "target_dud": target}
            for current, target in selection_common_multi
        ],
        "confirmation_common_multi_horizon_transitions": [
            {"current_dud": current, "target_dud": target}
            for current, target in confirmation_common_multi
        ],
        "feature_audit": feature_audit,
        "point_in_time_audit": {
            "history_time_order_violations": history_time_violations,
            "current_batch_history_violations": current_batch_history_violations,
            "market_time_violations": market_time_violations,
            "cross_collection_era_history_violations": cross_era_history,
            "cross_collection_era_market_violations": cross_era_market,
        },
        "stage1_actual_model_fits_after_authorization": expected_fits,
        "stage1_zero_shot_inference_runs_after_authorization": int(
            contract["stage1_zero_shot_inference_runs"]
        ),
        "model_fit_called": False,
        "calibrator_fit_called": False,
        "threshold_selected": False,
        "archived_110_115_outcomes_used": False,
    }
    report_path = TABLE_DIR / "prefit_validation_report.json"
    report_path.write_text(json.dumps(report_data, indent=2, default=str) + "\n", encoding="utf-8")

    REPORT.write_text(
        f"""# Leakage-controlled prefit validation

> Status: `LEAKAGE_CONTROLLED_PREFIT_VALIDATION_PASS`
>
> This rehearsal built labels and structural fold membership only. It did not
> import or fit an estimator, calibrator, threshold or ensemble.

## Structural fold counts

{folds.to_markdown(index=False)}

## Final authorization boundary

Training is still not authorized automatically. The first authorized Stage-1
run will contain **{expected_fits} model-fold fits** plus
**{contract['stage1_zero_shot_inference_runs']} zero-shot inference runs**.
Stage 2 and Stage 3 remain conditional, but one resumable Vast orchestrator may
run them after machine-readable promotion gates pass. C1 and C2 cannot reselect
model family, features, target representation, hyperparameters or ensemble.

Three checks remain non-negotiable: source-era isolation, label-time purge, and
matched-row baseline comparison. They are important, important, important.
""",
        encoding="utf-8",
    )

    outputs = [folds_path, transitions_path, multi_transitions_path, report_path, REPORT]
    manifest = {
        "status": "PREFIT_CONTRACT_READY_FOR_CANDIDATE_SCREENING",
        "created_at": now,
        "inputs": [
            {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest(path)}
            for path in required
        ],
        "outputs": [
            {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest(path)}
            for path in outputs
        ],
        "model_fit_called": False,
        "calibrator_fit_called": False,
        "threshold_selected": False,
        "uses_archived_110_115_outcomes": False,
    }
    manifest_path = TABLE_DIR / "prefit_validation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"[NO-FIT PREFLIGHT PASS] rows={len(pilot):,} exact_next={len(exact):,} "
        f"future_stage1_fits={expected_fits}"
    )


if __name__ == "__main__":
    main()
