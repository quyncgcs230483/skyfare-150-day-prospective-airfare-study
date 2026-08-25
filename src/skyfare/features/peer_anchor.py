#!/usr/bin/env python3
"""Build leakage-guarded anchor-relative and exact-event products."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from skyfare.core.paths import DataLayout
from skyfare.features.peer_anchor_contract import (
    BOOKING_WINDOWS,
    CATEGORICAL_FEATURES,
    EVENT_STATIC_FEATURES,
    EVENT_TARGETS,
    MILESTONES,
    PEER_ANCHOR_MIN_SUPPORT,
    PRIMARY_EVENT_TARGET,
    REGRESSION_STATIC_FEATURES,
    WARM_FEATURES,
    milestone_contract,
)

LAYOUT = DataLayout.resolve()
ROOT = LAYOUT.root
PRODUCT_ROOT = LAYOUT.artifacts / "peer_anchor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--milestone", choices=sorted(MILESTONES), required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def leave_one_out_median(values: np.ndarray) -> np.ndarray:
    """Return exact median after removing each corresponding value."""

    values = np.asarray(values, dtype=float)
    size = len(values)
    result = np.full(size, np.nan, dtype=float)
    if size <= 1:
        return result
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.arange(size)
    remaining = size - 1
    if remaining % 2:
        position = remaining // 2
        source = position + (position >= ranks)
        ranked_result = sorted_values[source]
    else:
        lower = remaining // 2 - 1
        upper = remaining // 2
        lower_source = lower + (lower >= ranks)
        upper_source = upper + (upper >= ranks)
        ranked_result = (sorted_values[lower_source] + sorted_values[upper_source]) / 2.0
    result[order] = ranked_result
    return result


def attach_peer_anchor(offers: pd.DataFrame) -> pd.DataFrame:
    work = offers.copy()
    key = ["session_key", "route", "flight_date"]
    peer = np.full(len(work), np.nan, dtype=float)
    support = np.zeros(len(work), dtype=np.int32)
    prices = work["price_vnd"].to_numpy(dtype=float)
    for positions in work.groupby(key, observed=True, sort=False).indices.values():
        positions = np.asarray(positions, dtype=np.int64)
        peer[positions] = leave_one_out_median(prices[positions])
        support[positions] = len(positions) - 1
    work["peer_anchor_vnd"] = peer
    work["peer_anchor_support"] = support

    strong_peer = (
        work["peer_anchor_support"].ge(PEER_ANCHOR_MIN_SUPPORT)
        & work["peer_anchor_vnd"].gt(0)
    ).fillna(False).to_numpy(dtype=bool)
    temporal_time = pd.to_datetime(work["temporal_market_time"], errors="coerce")
    historical = (
        work["temporal_market_median_price"].gt(0)
        & temporal_time.lt(work["feature_time"])
        & work["temporal_market_collection_era"].eq(work["collection_era"])
    ).fillna(False).to_numpy(dtype=bool)
    low_peer = (
        work["peer_anchor_support"].ge(1)
        & work["peer_anchor_vnd"].gt(0)
    ).fillna(False).to_numpy(dtype=bool)
    work["anchor_vnd"] = np.select(
        [strong_peer, historical, low_peer],
        [work["peer_anchor_vnd"], work["temporal_market_median_price"], work["peer_anchor_vnd"]],
        default=np.nan,
    )
    work["anchor_source"] = np.select(
        [strong_peer, historical, low_peer],
        ["CURRENT_PEER", "PRIOR_HISTORICAL", "CURRENT_PEER_LOW_SUPPORT"],
        default="UNAVAILABLE",
    )
    work["anchor_is_observed_peer"] = strong_peer.astype("int8")
    work["anchor_is_fallback"] = (
        ~strong_peer & work["anchor_vnd"].notna().to_numpy(dtype=bool)
    ).astype("int8")
    work["historical_anchor_time"] = temporal_time.where(work["anchor_source"].eq("PRIOR_HISTORICAL"))
    work["anchor_log"] = np.log(work["anchor_vnd"].where(work["anchor_vnd"].gt(0)))
    work["anchor_support_log1p"] = np.log1p(work["peer_anchor_support"].clip(lower=0))
    work["current_relative_log"] = np.log(
        work["price_vnd"].where(work["price_vnd"].gt(0)) / work["anchor_vnd"].where(work["anchor_vnd"].gt(0))
    )
    return work


def attach_relative_history(offers: pd.DataFrame) -> pd.DataFrame:
    out = offers.sort_values(
        ["collection_era", "schedule_slot_id", "feature_time", "session_key"]
    ).copy()
    keys = [out["collection_era"], out["schedule_slot_id"]]
    grouped = out.groupby(["collection_era", "schedule_slot_id"], observed=True, sort=False)
    valid = out["current_relative_log"].notna().astype(float)
    z = out["current_relative_log"].fillna(0.0)
    dud = out["days_until_departure"].astype(float)
    valid_dud = dud * valid

    prior_count = valid.groupby(keys, observed=True).cumsum() - valid
    prior_sum_z = z.groupby(keys, observed=True).cumsum() - z
    prior_sum_z2 = (z * z).groupby(keys, observed=True).cumsum() - z * z
    prior_sum_x = valid_dud.groupby(keys, observed=True).cumsum() - valid_dud
    prior_sum_x2 = (dud * dud * valid).groupby(keys, observed=True).cumsum() - dud * dud * valid
    prior_sum_xz = (dud * z).groupby(keys, observed=True).cumsum() - dud * z

    out["prior_relative_count"] = prior_count.astype("int32")
    out["previous_relative_log"] = grouped["current_relative_log"].shift(1)
    out["previous_anchor_vnd"] = grouped["anchor_vnd"].shift(1)
    out["previous_relative_time"] = grouped["feature_time"].shift(1)
    out["previous_relative_session_key"] = grouped["session_key"].shift(1)
    out["relative_lag_age_hours"] = (
        out["feature_time"] - out["previous_relative_time"]
    ).dt.total_seconds() / 3600.0
    out["market_shift_log"] = np.log(
        out["anchor_vnd"].where(out["anchor_vnd"].gt(0))
        / out["previous_anchor_vnd"].where(out["previous_anchor_vnd"].gt(0))
    )

    mean_z = prior_sum_z / prior_count.where(prior_count.gt(0))
    variance = prior_sum_z2 / prior_count.where(prior_count.gt(0)) - mean_z.pow(2)
    out["prior_relative_volatility"] = np.sqrt(variance.clip(lower=0)).where(prior_count.ge(2))
    denominator = prior_count * prior_sum_x2 - prior_sum_x.pow(2)
    numerator = prior_count * prior_sum_xz - prior_sum_x * prior_sum_z
    out["prior_relative_trend_per_dud_day"] = (numerator / denominator).where(
        prior_count.ge(2) & denominator.abs().gt(1e-12)
    )
    out["relative_history_eligible"] = prior_count.ge(3).astype("int8")

    previous_mask = out["previous_relative_time"].notna()
    if not out.loc[previous_mask, "previous_relative_time"].lt(out.loc[previous_mask, "feature_time"]).all():
        raise RuntimeError("Relative history contains non-prior timestamps")
    if out.loc[previous_mask, "previous_relative_session_key"].eq(out.loc[previous_mask, "session_key"]).any():
        raise RuntimeError("Relative history contains current-batch rows")
    return out.sort_values(["feature_time", "session_key", "schedule_slot_id"]).reset_index(drop=True)


def target_column(target_name: str) -> str:
    return f"event_{target_name.lower()}"


def attach_target_definitions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    saving = out["price_vnd"].astype(float) - out["target_price_vnd"].astype(float)
    out["saving_vnd"] = saving
    for name, definition in EVENT_TARGETS.items():
        kind = definition["kind"]
        if kind == "PCT":
            threshold = out["price_vnd"].astype(float) * float(definition["value"])
            label = saving.gt(0) if float(definition["value"]) == 0 else saving.ge(threshold)
        elif kind == "VND":
            label = saving.ge(float(definition["value"]))
        elif kind == "MIN_PCT_VND":
            threshold = np.minimum(
                out["price_vnd"].astype(float) * float(definition["fraction"]),
                float(definition["vnd"]),
            )
            label = saving.ge(threshold)
        else:
            raise KeyError(f"Unknown event target kind: {kind}")
        out[target_column(name)] = label.astype("int8")
    out["material_drop_next"] = out[target_column(PRIMARY_EVENT_TARGET)].astype("int8")
    return out


def attach_event_truth(
    offers: pd.DataFrame,
    targets: pd.DataFrame,
    source: str,
) -> pd.DataFrame:
    columns = [
        "offer_id",
        "target_offer_id",
        "current_dud",
        "target_dud",
        "horizon_gap_days",
        "label_time",
        "target_price_vnd",
        "price_change_vnd",
        "price_change_pct",
    ]
    event = targets[columns].copy()
    if "transition" in targets.columns:
        event["transition"] = targets["transition"].astype("string")
    else:
        event["transition"] = (
            targets["current_dud"].astype(int).astype(str)
            + "->"
            + targets["target_dud"].astype(int).astype(str)
        )
    frame = event.merge(offers, on="offer_id", how="inner", validate="one_to_one")
    if not pd.to_datetime(frame["label_time"]).gt(pd.to_datetime(frame["feature_time"])).all():
        raise RuntimeError(f"{source} event contains non-future label time")
    frame["event_source"] = source
    next_dud = {current: target for current, target in zip(BOOKING_WINDOWS[:-1], BOOKING_WINDOWS[1:], strict=True)}
    expected_target = frame["current_dud"].map(next_dud)
    expected_gap = frame["current_dud"] - expected_target
    frame["is_auxiliary_target"] = int(source == "AUX_NEAREST_LOWER")
    frame["aux_gap_extra_days"] = (
        frame["horizon_gap_days"].astype(float) - expected_gap.astype(float)
    ).clip(lower=0.0)
    frame = attach_target_definitions(frame)
    return frame.sort_values(["feature_time", "offer_id"]).reset_index(drop=True)


def build_auxiliary_targets(exact: pd.DataFrame, multi: pd.DataFrame) -> pd.DataFrame:
    exact_ids = set(exact["offer_id"].tolist())
    candidates = multi[
        ~multi["offer_id"].isin(exact_ids)
        & multi["target_dud"].lt(multi["current_dud"])
    ].copy()
    if candidates.empty:
        return candidates
    candidates = candidates.sort_values(
        ["offer_id", "target_dud", "label_time"],
        ascending=[True, False, True],
    ).drop_duplicates("offer_id", keep="first")
    if candidates["offer_id"].duplicated().any():
        raise RuntimeError("Auxiliary nearest-lower selection is not one-row-per-offer")
    return candidates.reset_index(drop=True)


def build_dud1_events(offers: pd.DataFrame, dud1: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "offer_id",
        "target_offer_id",
        "current_dud",
        "target_dud",
        "label_time",
        "target_price_vnd",
        "price_change_vnd",
        "price_change_pct",
    ]
    event = dud1[columns].copy()
    event["horizon_gap_days"] = 0.0
    event["horizon_gap_hours"] = dud1["horizon_gap_hours"].astype(float)
    event["transition"] = "DUD1_AM->PM"
    frame = event.merge(offers, on="offer_id", how="inner", validate="one_to_one")
    if not frame["session_label"].astype(str).eq("AM").all():
        raise RuntimeError("DUD1 intraday source is not exclusively AM")
    if not pd.to_datetime(frame["label_time"]).gt(pd.to_datetime(frame["feature_time"])).all():
        raise RuntimeError("DUD1 intraday event contains non-future label time")
    frame["event_source"] = "DUD1_INTRADAY_AM_TO_PM"
    frame["is_auxiliary_target"] = 0
    frame["aux_gap_extra_days"] = 0.0
    return attach_target_definitions(frame).sort_values(["feature_time", "offer_id"]).reset_index(drop=True)


def build_event_registry(
    offers: pd.DataFrame,
    exact_events: pd.DataFrame,
    aux_events: pd.DataFrame,
    dud1_events: pd.DataFrame,
    cutoff: str,
) -> pd.DataFrame:
    next_dud = {current: target for current, target in zip(BOOKING_WINDOWS[:-1], BOOKING_WINDOWS[1:], strict=True)}
    registry = offers[[
        "offer_id", "session_key", "session_date", "session_label", "feature_time",
        "flight_date", "days_until_departure", "route", "airline", "schedule_slot_id",
    ]].copy()
    registry["expected_target_dud"] = registry["days_until_departure"].map(next_dud)
    expected_target = registry["expected_target_dud"]
    registry["expected_label_date"] = (
        registry["flight_date"]
        - pd.to_timedelta(expected_target.fillna(0).astype("int16"), unit="D")
    ).dt.normalize().where(expected_target.notna())
    exact_ids = set(exact_events["offer_id"].tolist())
    aux_ids = set(aux_events["offer_id"].tolist())
    dud1_ids = set(dud1_events["offer_id"].tolist())
    registry["has_exact_label"] = registry["offer_id"].isin(exact_ids).astype("int8")
    registry["has_aux_nearest_lower_label"] = registry["offer_id"].isin(aux_ids).astype("int8")
    registry["has_dud1_intraday_label"] = registry["offer_id"].isin(dud1_ids).astype("int8")
    cutoff_date = pd.Timestamp(cutoff)
    is_dud1 = registry["days_until_departure"].eq(1)
    is_am = registry["session_label"].astype(str).eq("AM")
    exact_status = np.select(
        [
            registry["has_exact_label"].eq(1),
            is_dud1,
            registry["expected_label_date"].gt(cutoff_date),
        ],
        ["EXACT_MATURE", "NOT_APPLICABLE_DUD1", "RIGHT_CENSORED"],
        default="MATURED_NOT_REOBSERVED",
    )
    registry["exact_status"] = exact_status
    registry["event_label_status"] = np.select(
        [
            registry["has_exact_label"].eq(1),
            is_dud1 & is_am & registry["has_dud1_intraday_label"].eq(1),
            is_dud1 & is_am,
            is_dud1 & ~is_am,
            registry["has_aux_nearest_lower_label"].eq(1),
            registry["exact_status"].eq("RIGHT_CENSORED"),
        ],
        [
            "EXACT_MATURE",
            "DUD1_INTRADAY_MATURE",
            "DUD1_INTRADAY_MISSING",
            "DUD1_PM_TERMINAL",
            "AUX_NEAREST_AVAILABLE",
            "RIGHT_CENSORED",
        ],
        default="MATURED_NOT_REOBSERVED",
    )
    return registry.sort_values(["feature_time", "offer_id"]).reset_index(drop=True)


def correlation_audit(frame: pd.DataFrame, features: list[str], scope: str) -> pd.DataFrame:
    numeric = sorted(
        {
            feature
            for feature in features
            if feature not in CATEGORICAL_FEATURES and feature in frame.columns
        }
    )
    sample = frame[numeric].sample(min(150_000, len(frame)), random_state=17) if len(frame) else frame[numeric]
    pearson = sample.corr(method="pearson", min_periods=100)
    spearman = sample.corr(method="spearman", min_periods=100)
    rows = []
    for left_index, left in enumerate(numeric):
        for right in numeric[left_index + 1 :]:
            rows.append(
                {
                    "scope": scope,
                    "feature_left": left,
                    "feature_right": right,
                    "pearson": pearson.loc[left, right],
                    "spearman": spearman.loc[left, right],
                    "max_abs_association": max(abs(pearson.loc[left, right]), abs(spearman.loc[left, right])),
                }
            )
    return pd.DataFrame(rows).sort_values("max_abs_association", ascending=False)


def frame_summary(
    offers: pd.DataFrame,
    events: pd.DataFrame,
    auxiliary: pd.DataFrame,
    dud1: pd.DataFrame,
    registry: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "offer_rows": len(offers),
        "event_rows": len(events),
        "session_start": str(offers["session_date"].min()),
        "session_end": str(offers["session_date"].max()),
        "anchor_coverage": float(offers["anchor_vnd"].notna().mean()),
        "anchor_source_counts": offers["anchor_source"].value_counts(dropna=False).to_dict(),
        "relative_history_eligible_rate": float(offers["relative_history_eligible"].mean()),
        "event_prevalence": float(events["material_drop_next"].mean()),
        "event_transition_counts": events["transition"].value_counts().to_dict(),
        "event_target_prevalence": {
            name: float(events[target_column(name)].mean()) for name in EVENT_TARGETS
        },
        "auxiliary_nearest_lower_rows": len(auxiliary),
        "dud1_intraday_rows": len(dud1),
        "dud1_primary_event_prevalence": float(dud1["material_drop_next"].mean()) if len(dud1) else None,
        "event_label_status_counts": registry["event_label_status"].value_counts(dropna=False).to_dict(),
        "exact_status_counts": registry["exact_status"].value_counts(dropna=False).to_dict(),
    }


def main() -> None:
    args = parse_args()
    config = MILESTONES[args.milestone]
    input_root = ROOT / config["input_root"]
    offers_path = input_root / config["offers"]
    exact_path = input_root / config["exact"]
    multi_path = input_root / config["multi"]
    dud1_path = input_root / config["dud1"]
    output_root = PRODUCT_ROOT / f"pilot{args.milestone}"
    manifest_path = output_root / "product_manifest.json"
    if manifest_path.exists() and not args.force:
        print(f"[PRODUCT SKIP] {manifest_path}")
        return

    print(f"[PRODUCT START] milestone={args.milestone} cutoff={config['cutoff']}", flush=True)
    offers = pd.read_parquet(offers_path)
    exact = pd.read_parquet(exact_path)
    multi = pd.read_parquet(multi_path)
    dud1 = pd.read_parquet(dud1_path)
    offers["feature_time"] = pd.to_datetime(offers["feature_time"])
    offers["session_date"] = pd.to_datetime(offers["session_date"]).dt.normalize()
    offers["flight_date"] = pd.to_datetime(offers["flight_date"]).dt.normalize()
    exact["feature_time"] = pd.to_datetime(exact["feature_time"])
    exact["label_time"] = pd.to_datetime(exact["label_time"])
    multi["feature_time"] = pd.to_datetime(multi["feature_time"])
    multi["label_time"] = pd.to_datetime(multi["label_time"])
    dud1["feature_time"] = pd.to_datetime(dud1["feature_time"])
    dud1["label_time"] = pd.to_datetime(dud1["label_time"])
    if offers.duplicated(["session_key", "schedule_slot_id"]).any():
        raise RuntimeError("Input contains duplicate schedule slots inside a completed batch")
    if offers["session_date"].max() > pd.Timestamp(config["cutoff"]):
        raise RuntimeError("Input exceeds milestone cutoff")

    offers = attach_peer_anchor(offers)
    offers = attach_relative_history(offers)
    events = attach_event_truth(offers, exact, "EXACT_NEXT")
    auxiliary_raw = build_auxiliary_targets(exact, multi)
    auxiliary = attach_event_truth(offers, auxiliary_raw, "AUX_NEAREST_LOWER")
    dud1_events = build_dud1_events(offers, dud1)
    registry = build_event_registry(offers, events, auxiliary, dud1_events, config["cutoff"])
    audit = pd.concat(
        [
            correlation_audit(offers, [*REGRESSION_STATIC_FEATURES, *WARM_FEATURES], "REGRESSION"),
            correlation_audit(events, [*EVENT_STATIC_FEATURES, *WARM_FEATURES], "EVENT_EXACT"),
            correlation_audit(auxiliary, [*EVENT_STATIC_FEATURES, *WARM_FEATURES], "EVENT_AUX"),
        ],
        ignore_index=True,
    ).sort_values("max_abs_association", ascending=False)
    selected_high = audit[audit["max_abs_association"].ge(0.95)]

    output_root.mkdir(parents=True, exist_ok=True)
    offers_output = output_root / "anchor_relative_offers.parquet"
    events_output = output_root / "exact_material_drop_events.parquet"
    auxiliary_output = output_root / "aux_nearest_lower_events.parquet"
    dud1_output = output_root / "dud1_intraday_events.parquet"
    registry_output = output_root / "event_row_registry.parquet"
    correlation_output = output_root / "selected_feature_correlation_audit.csv"
    contract_output = output_root / "contract.json"
    offers.to_parquet(offers_output, index=False, compression="zstd")
    events.to_parquet(events_output, index=False, compression="zstd")
    auxiliary.to_parquet(auxiliary_output, index=False, compression="zstd")
    dud1_events.to_parquet(dud1_output, index=False, compression="zstd")
    registry.to_parquet(registry_output, index=False, compression="zstd")
    audit.to_csv(correlation_output, index=False)
    write_json(contract_output, milestone_contract(args.milestone))

    manifest = {
        "status": "ANCHOR_EVENT_PRODUCTS_COMPLETE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "milestone": args.milestone,
        "summary": frame_summary(offers, events, auxiliary, dud1_events, registry),
        "selected_feature_pairs_ge_0_95": selected_high.to_dict(orient="records"),
        "inputs": {
            str(offers_path.relative_to(ROOT)): sha256(offers_path),
            str(exact_path.relative_to(ROOT)): sha256(exact_path),
            str(multi_path.relative_to(ROOT)): sha256(multi_path),
            str(dud1_path.relative_to(ROOT)): sha256(dud1_path),
        },
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in [
                offers_output,
                events_output,
                auxiliary_output,
                dud1_output,
                registry_output,
                correlation_output,
                contract_output,
            ]
        },
    }
    write_json(manifest_path, manifest)
    print(
        f"[PRODUCT PASS] milestone={args.milestone} offers={len(offers):,} events={len(events):,} "
        f"aux={len(auxiliary):,} dud1={len(dud1_events):,} "
        f"anchor={offers['anchor_vnd'].notna().mean():.3%} high_corr_pairs={len(selected_high)}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[PRODUCT FAIL] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
