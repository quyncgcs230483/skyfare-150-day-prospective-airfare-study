"""EA05 causal baselines by horizon and cold/warm population.

No estimator is fitted. Historical prevalence and market summaries are built
strictly from observations available before each validation block.
"""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from skyfare.features.audit_common import BOOKING_WINDOWS, CUTOFF, ROOT, next_window_map
from skyfare.features.candidate_feature_contract import build_candidate_frame

TABLE_DIR = ROOT / "artifacts/feature_research/tables"
REPORT = ROOT / "artifacts/feature_research/reports/EA05_CAUSAL_BASELINES.txt"


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    mask = np.isfinite(y) & np.isfinite(pred)
    y = y[mask].astype(float)
    pred = pred[mask].astype(float)
    if not len(y):
        return {"eligible_rows": 0, "mae_vnd": np.nan, "mape_pct": np.nan, "r2": np.nan}
    residual = y - pred
    denominator = np.sum((y - y.mean()) ** 2)
    return {
        "eligible_rows": int(len(y)),
        "mae_vnd": float(np.mean(np.abs(residual))),
        "mape_pct": float(100 * np.mean(np.abs(residual) / y)),
        "r2": float(1 - np.sum(residual**2) / denominator) if denominator > 0 else np.nan,
    }


def exact_identity(frame: pd.DataFrame) -> pd.Series:
    return pd.util.hash_pandas_object(
        frame[["collection_era", "schedule_slot_id", "session_label"]], index=False
    ).astype("uint64")


def target_lookup(frame: pd.DataFrame) -> pd.DataFrame:
    target = frame[
        ["entity_band", "days_until_departure", "price_vnd", "feature_time", "session_date"]
    ].copy()
    if target.duplicated(["entity_band", "days_until_departure"]).any():
        raise RuntimeError("EA05 requires unique entity-band-DUD target rows")
    return target.rename(
        columns={
            "days_until_departure": "target_dud",
            "price_vnd": "target_price_vnd",
            "feature_time": "label_time",
            "session_date": "label_session_date",
        }
    )


def causal_target_market_index(frame: pd.DataFrame) -> dict[tuple[str, str, str, int], tuple[np.ndarray, np.ndarray]]:
    market = (
        frame.groupby(
            ["collection_era", "route", "airline", "days_until_departure", "feature_time"], observed=True
        )["price_vnd"]
        .median()
        .reset_index()
        .sort_values(["collection_era", "route", "airline", "days_until_departure", "feature_time"])
    )
    output = {}
    for key, group in market.groupby(
        ["collection_era", "route", "airline", "days_until_departure"], observed=True
    ):
        output[(str(key[0]), str(key[1]), str(key[2]), int(key[3]))] = (
            group["feature_time"].astype("int64").to_numpy(),
            group["price_vnd"].astype(float).to_numpy(),
        )
    return output


def attach_causal_target_market(
    pairs: pd.DataFrame,
    index: dict[tuple[str, str, str, int], tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    prediction = np.full(len(pairs), np.nan)
    grouped = pairs.groupby(
        ["collection_era", "route", "airline", "target_dud"], observed=True
    ).groups
    for key, row_index in grouped.items():
        lookup = index.get((str(key[0]), str(key[1]), str(key[2]), int(key[3])))
        if lookup is None:
            continue
        times, values = lookup
        positions = pairs.index.get_indexer(row_index)
        current = pairs.loc[row_index, "feature_time"].astype("int64").to_numpy()
        prior = np.searchsorted(times, current, side="left") - 1
        valid = prior >= 0
        prediction[positions[valid]] = values[prior[valid]]
    return prediction


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    frame = build_candidate_frame()
    frame["entity_band"] = exact_identity(frame)
    lookup = target_lookup(frame)
    market_index = causal_target_market_index(frame)

    result_rows: list[dict[str, object]] = []

    # h=0 baselines are evaluated against current price but never use it as input.
    h0 = [
        ("prior_route_airline_dud_market_median", "all", frame["temporal_market_median_price"].to_numpy(float)),
        ("current_other_airline_minimum", "all", frame["competitor_min_price_other_airlines"].to_numpy(float)),
        ("previous_same_schedule_carry_forward", "warm", frame["previous_price_same_schedule"].to_numpy(float)),
    ]
    y0 = frame["price_vnd"].to_numpy(float)
    for baseline, population, pred in h0:
        item = metrics(y0, pred)
        result_rows.append(
            {
                "task": "relative_h0",
                "current_dud": "ALL",
                "target_dud": "CURRENT",
                "population": population,
                "baseline": baseline,
                "coverage_pct": 100 * item["eligible_rows"] / len(frame),
                **item,
            }
        )

    exact_pairs: list[pd.DataFrame] = []
    for current_dud in BOOKING_WINDOWS[:-1]:
        current = frame.loc[
            frame["days_until_departure"].eq(current_dud),
            [
                "entity_band",
                "collection_era",
                "route",
                "airline",
                "flight_date",
                "feature_time",
                "session_date",
                "price_vnd",
                "competitor_min_price_other_airlines",
                "previous_price_same_schedule",
            ],
        ].copy()
        for target_dud in [value for value in BOOKING_WINDOWS if value < current_dud]:
            pairs = current.merge(
                lookup[lookup["target_dud"].eq(target_dud)],
                on="entity_band",
                how="left",
                validate="one_to_one",
            )
            expected = pairs["flight_date"].dt.normalize() - pd.to_timedelta(target_dud, unit="D")
            valid = (
                pairs["target_price_vnd"].notna()
                & pairs["label_time"].gt(pairs["feature_time"])
                & pairs["label_session_date"].eq(expected)
            )
            pairs = pairs.loc[valid].reset_index(drop=True)
            if pairs.empty:
                continue
            pairs["target_dud"] = target_dud
            pairs["causal_target_market_median"] = attach_causal_target_market(pairs, market_index)
            y = pairs["target_price_vnd"].to_numpy(float)
            baselines = [
                ("current_price_carry_forward", pairs["price_vnd"].to_numpy(float)),
                ("current_other_airline_minimum", pairs["competitor_min_price_other_airlines"].to_numpy(float)),
                ("causal_route_airline_target_dud_market", pairs["causal_target_market_median"]),
            ]
            for baseline, pred in baselines:
                item = metrics(y, np.asarray(pred, dtype=float))
                result_rows.append(
                    {
                        "task": "future_price_exact_next" if next_window_map()[current_dud] == target_dud else "future_price_multi_horizon",
                        "current_dud": current_dud,
                        "target_dud": target_dud,
                        "population": "all",
                        "baseline": baseline,
                        "coverage_pct": 100 * item["eligible_rows"] / len(pairs),
                        **item,
                    }
                )
            if next_window_map()[current_dud] == target_dud:
                pairs["will_drop_exact_next"] = (
                    pairs["target_price_vnd"] <= 0.95 * pairs["price_vnd"]
                ).astype("int8")
                exact_pairs.append(
                    pairs[["entity_band", "session_date", "feature_time", "label_time", "target_dud", "will_drop_exact_next"]]
                )

    regression = pd.DataFrame(result_rows)
    regression.to_csv(TABLE_DIR / "ea05_regression_causal_baselines.csv", index=False)

    exact = pd.concat(exact_pairs, ignore_index=True)
    folds = [
        ("PILOT_F1", pd.Timestamp("2026-06-20"), pd.Timestamp("2026-06-26")),
        ("PILOT_F2", pd.Timestamp("2026-06-27"), pd.Timestamp("2026-07-03")),
        ("PILOT_F3", pd.Timestamp("2026-07-04"), pd.Timestamp("2026-07-10")),
    ]
    cls_rows = []
    for fold, start, end in folds:
        train = exact[exact["label_time"].lt(start)]
        valid = exact[exact["session_date"].between(start, end)]
        overall = float(train["will_drop_exact_next"].mean())
        dud_rate = train.groupby("target_dud", observed=True)["will_drop_exact_next"].mean()
        for baseline in ["past_overall_prevalence", "past_target_dud_prevalence"]:
            pred = (
                np.full(len(valid), overall)
                if baseline == "past_overall_prevalence"
                else valid["target_dud"].map(dud_rate).fillna(overall).to_numpy(float)
            )
            y = valid["will_drop_exact_next"].to_numpy(int)
            ap = float(average_precision_score(y, pred)) if len(np.unique(y)) > 1 else np.nan
            cls_rows.append(
                {
                    "fold": fold,
                    "validation_start": start.date(),
                    "validation_end": end.date(),
                    "baseline": baseline,
                    "train_rows_after_label_time_purge": len(train),
                    "validation_rows": len(valid),
                    "validation_wait_prevalence": float(y.mean()) if len(y) else np.nan,
                    "pr_auc": ap,
                    "pr_auc_minus_validation_prevalence": ap - float(y.mean()) if len(y) else np.nan,
                }
            )
    classification = pd.DataFrame(cls_rows)
    classification.to_csv(TABLE_DIR / "ea05_willdrop_causal_baselines.csv", index=False)

    summary = {
        "status": "EA05_COMPLETE_MODEL_FREE",
        "created_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
        "cutoff": str(CUTOFF.date()),
        "regression_baseline_rows": len(regression),
        "exact_next_classification_rows": len(exact),
        "classification_folds": len(folds),
        "label_time_purge_applied": True,
        "model_fit_called": False,
        "required_model_claim": "beat the strongest legal baseline on the same rows, horizon and population",
    }
    (TABLE_DIR / "ea05_causal_baseline_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    h0_table = regression[regression["task"].eq("relative_h0")][
        ["baseline", "population", "coverage_pct", "mape_pct", "r2"]
    ].to_markdown(index=False)
    exact_table = (
        regression[regression["task"].eq("future_price_exact_next")]
        .groupby("baseline", observed=True)
        .agg(rows=("eligible_rows", "sum"), mean_mape_pct=("mape_pct", "mean"), mean_r2=("r2", "mean"))
        .reset_index()
        .to_markdown(index=False)
    )
    cls_table = classification.to_markdown(index=False)
    REPORT.write_text(
        f"""# EA05 - Causal Baseline Matrix

> Created before any new-build estimator fit. Every baseline uses information
> available at prediction time; classification training rows are purged by
> `label_time < validation_start`.

## Current fair-value h=0

{h0_table}

## Future-price exact next window

{exact_table}

## Will-Drop rolling baselines

{cls_table}

## Gate for later modelling

A model is not valuable merely because it beats another trained model. It must
beat the strongest legal baseline on the same rows, target horizon and
population. Carry-forward is unavailable for cold h=0 rows; this is exactly why
cold modelling remains a separate product requirement.
""",
        encoding="utf-8",
    )
    print(f"[EA05 PASS] regression_rows={len(regression)} exact_labels={len(exact):,} folds={len(folds)}")


if __name__ == "__main__":
    main()
