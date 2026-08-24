"""EA02 multi-horizon target, maturity and opportunity audit.

Snapshot: new-build 110 pilot, cutoff 2026-07-10.
Branch: model-free pre-fit gate EA02.
Input: immutable raw rows through cutoff; one latest offer per completed AM/PM batch.
Output: horizon/transition CSVs, JSON, PNG and Markdown report.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from skyfare.features.audit_common import BOOKING_WINDOWS, CUTOFF, ROOT, completed_batch_offers, input_manifest, load_new_build_raw, next_window_map


TABLE_DIR = ROOT / "artifacts/feature_research/tables"
FIG_DIR = ROOT / "artifacts/feature_research/figures"
REPORT = ROOT / "artifacts/feature_research/reports/EA02_TARGET_HORIZON_AUDIT.txt"


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(100 * numerator / denominator) if denominator else float("nan")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_new_build_raw()
    offers = completed_batch_offers(raw)
    offers = offers[offers["days_until_departure"].isin(BOOKING_WINDOWS)].copy()
    offers = offers.reset_index(drop=True)
    offers["row_id"] = np.arange(len(offers), dtype=np.int64)
    offers["entity_band"] = pd.util.hash_pandas_object(
        offers[["collection_era", "schedule_slot_id", "session_label"]], index=False
    ).astype("uint64")

    duplicate_targets = int(
        offers.duplicated(["entity_band", "days_until_departure"], keep=False).sum()
    )
    if duplicate_targets:
        raise RuntimeError(
            f"EA02 requires unique entity-band-DUD targets; found {duplicate_targets} duplicate rows"
        )

    target_lookup = offers[
        ["entity_band", "days_until_departure", "price_vnd", "session_id", "session_date"]
    ].copy()
    target_lookup.columns = [
        "entity_band",
        "target_dud",
        "target_price_vnd",
        "target_time",
        "target_session_date",
    ]

    oracle_min = np.full(len(offers), np.inf, dtype=float)
    oracle_best_dud = np.full(len(offers), -1, dtype=int)
    observed_future_count = np.zeros(len(offers), dtype=int)
    horizon_rows: list[dict[str, object]] = []

    next_map = next_window_map()
    for current_dud in BOOKING_WINDOWS[:-1]:
        future_duds = [dud for dud in BOOKING_WINDOWS if dud < current_dud]
        current = offers.loc[
            offers["days_until_departure"].eq(current_dud),
            ["row_id", "entity_band", "flight_date", "session_id", "price_vnd"],
        ].copy()
        for target_dud in future_duds:
            pairs = current.merge(
                target_lookup[target_lookup["target_dud"].eq(target_dud)],
                on="entity_band",
                how="left",
                validate="one_to_one",
            )
            expected_date = pairs["flight_date"].dt.normalize() - pd.to_timedelta(target_dud, unit="D")
            mature = expected_date.le(CUTOFF)
            observed = (
                pairs["target_price_vnd"].notna()
                & pairs["target_time"].gt(pairs["session_id"])
                & pairs["target_session_date"].eq(expected_date)
            )
            valid = pairs.loc[observed].copy()
            valid["change_vnd"] = valid["target_price_vnd"] - valid["price_vnd"]
            valid["change_pct"] = valid["target_price_vnd"] / valid["price_vnd"] - 1
            wait = valid["change_pct"].le(-0.05)
            rise = valid["change_pct"].ge(0.05)
            stable = ~(wait | rise)
            wait_savings = -valid.loc[wait, "change_vnd"]
            rise_cost = valid.loc[rise, "change_vnd"]

            ids = valid["row_id"].to_numpy(dtype=int)
            target_prices = valid["target_price_vnd"].to_numpy(dtype=float)
            improve = target_prices < oracle_min[ids]
            if len(ids):
                observed_future_count[ids] += 1
                oracle_min[ids] = np.minimum(oracle_min[ids], target_prices)
                oracle_best_dud[ids[improve]] = target_dud

            horizon_rows.append(
                {
                    "current_dud": current_dud,
                    "target_dud": target_dud,
                    "horizon_gap_days": current_dud - target_dud,
                    "is_exact_next": next_map[current_dud] == target_dud,
                    "current_rows": int(len(pairs)),
                    "mature_rows": int(mature.sum()),
                    "observed_target_rows": int(observed.sum()),
                    "coverage_of_mature_pct": pct(int(observed.sum()), int(mature.sum())),
                    "strict_time_violations": int(
                        (
                            pairs["target_price_vnd"].notna()
                            & ~pairs["target_time"].gt(pairs["session_id"])
                        ).sum()
                    ),
                    "median_change_vnd": float(valid["change_vnd"].median()) if len(valid) else np.nan,
                    "median_change_pct": float(100 * valid["change_pct"].median()) if len(valid) else np.nan,
                    "wait_prevalence_pct": pct(int(wait.sum()), len(valid)),
                    "stable_pct": pct(int(stable.sum()), len(valid)),
                    "rise_pct": pct(int(rise.sum()), len(valid)),
                    "wait_rows": int(wait.sum()),
                    "median_wait_savings_vnd": float(wait_savings.median()) if len(wait_savings) else np.nan,
                    "p90_wait_savings_vnd": float(wait_savings.quantile(0.90)) if len(wait_savings) else np.nan,
                    "rise_rows": int(rise.sum()),
                    "median_rise_cost_vnd": float(rise_cost.median()) if len(rise_cost) else np.nan,
                    "p90_rise_cost_vnd": float(rise_cost.quantile(0.90)) if len(rise_cost) else np.nan,
                }
            )

    horizons = pd.DataFrame(horizon_rows)
    horizons.to_csv(TABLE_DIR / "ea02_multi_horizon_coverage.csv", index=False)
    exact = horizons[horizons["is_exact_next"]].copy()
    exact.to_csv(TABLE_DIR / "ea02_exact_next_transition_targets.csv", index=False)

    oracle = offers[["row_id", "days_until_departure", "price_vnd"]].copy()
    oracle["observed_future_count"] = observed_future_count
    oracle["oracle_min_future_price"] = np.where(np.isfinite(oracle_min), oracle_min, np.nan)
    oracle["oracle_best_dud"] = np.where(oracle_best_dud >= 0, oracle_best_dud, np.nan)
    oracle["oracle_savings_vnd"] = oracle["price_vnd"] - oracle["oracle_min_future_price"]
    oracle["oracle_savings_pct"] = oracle["oracle_savings_vnd"] / oracle["price_vnd"] * 100
    oracle_summary = (
        oracle.groupby("days_until_departure", observed=True)
        .agg(
            current_rows=("row_id", "size"),
            rows_with_observed_future=("oracle_min_future_price", "count"),
            median_oracle_savings_vnd=("oracle_savings_vnd", "median"),
            median_oracle_savings_pct=("oracle_savings_pct", "median"),
            opportunity_rows=("oracle_savings_vnd", lambda x: x.gt(0).sum()),
        )
        .reset_index()
    )
    oracle_summary["future_coverage_pct"] = (
        100 * oracle_summary["rows_with_observed_future"] / oracle_summary["current_rows"]
    )
    oracle_summary["opportunity_pct_of_observed"] = (
        100
        * oracle_summary["opportunity_rows"]
        / oracle_summary["rows_with_observed_future"].replace(0, np.nan)
    )
    oracle_summary.to_csv(TABLE_DIR / "ea02_oracle_opportunity_ceiling.csv", index=False)

    summary = {
        "status": "EA02_COMPLETE_MODEL_FREE",
        "cutoff": str(CUTOFF.date()),
        "prediction_unit": "ONE_LATEST_OFFER_STATE_PER_COMPLETED_AM_PM_BATCH",
        "identity": "COLLECTION_ERA_PLUS_SCHEDULE_SLOT_PROXY_PLUS_SESSION_LABEL",
        "grid": BOOKING_WINDOWS,
        "multi_horizon_pairs": int(len(horizons)),
        "exact_next_transitions": int(len(exact)),
        "strict_time_violations": int(horizons["strict_time_violations"].sum()),
        "exact_next_observed_targets": int(exact["observed_target_rows"].sum()),
        "exact_next_mature_rows": int(exact["mature_rows"].sum()),
        "exact_next_wait_prevalence_pct": float(
            np.average(
                exact["wait_prevalence_pct"].fillna(0),
                weights=exact["observed_target_rows"].clip(lower=1),
            )
        ),
        "oracle_is_upper_bound_not_policy_value": True,
        "input_manifest": input_manifest(),
    }
    (TABLE_DIR / "ea02_target_horizon_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )

    sns.set_theme(style="whitegrid", palette="colorblind")
    labels = exact["current_dud"].astype(str) + "->" + exact["target_dud"].astype(str)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    sns.barplot(x=labels, y=exact["coverage_of_mature_pct"], color="#0072B2", ax=axes[0])
    axes[0].set_title("Exact-next coverage varies after maturity")
    axes[0].set_ylabel("Observed / mature rows (%)")
    sns.barplot(x=labels, y=exact["wait_prevalence_pct"], color="#009E73", ax=axes[1])
    axes[1].set_title("WAIT prevalence changes by DUD transition")
    axes[1].set_ylabel("Price drop >=5% (%)")
    sns.barplot(x=labels, y=exact["median_change_pct"], color="#D55E00", ax=axes[2])
    axes[2].axhline(0, color="#222222", linewidth=1)
    axes[2].set_title("Median next-window change is not constant")
    axes[2].set_ylabel("Median price change (%)")
    for ax in axes:
        ax.set_xlabel("DUD transition")
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("EA02. Each booking horizon has different coverage and target behavior", fontweight="bold")
    fig.tight_layout()
    figure_path = FIG_DIR / "EA02_target_horizon_audit.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    report = f"""# EA02 - Target and Horizon Audit

> New-build 110 cutoff: `{CUTOFF.date()}`. Model-free. No interpolation creates labels.

## Contract checked

- Prediction unit: latest offer state in one completed AM/PM batch.
- Identity: same schedule-slot proxy and same AM/PM band.
- Regression: every reachable lower-DUD target.
- Classification: exact-next `WAIT = 1[next_price <= 0.95 * current_price]`.

## Results

- Multi-horizon transitions audited: **{len(horizons)}**.
- Exact-next transitions: **{len(exact)}**.
- Mature exact-next rows: **{int(exact['mature_rows'].sum()):,}**.
- Observed exact-next labels: **{int(exact['observed_target_rows'].sum()):,}**.
- Strict target-time violations: **{int(horizons['strict_time_violations'].sum()):,}**.

Coverage and prevalence differ by transition. Therefore headline metrics must be
reported by horizon/DUD, and target masks must distinguish immature rows from
mature-but-missing exact targets. Oracle savings remain opportunity ceiling,
not realized policy value.
"""
    REPORT.write_text(report, encoding="utf-8")
    print(
        f"[EA02 PASS] offers={len(offers):,} exact_labels={int(exact['observed_target_rows'].sum()):,} "
        f"time_violations={int(horizons['strict_time_violations'].sum())}"
    )
    print(f"[EA02 PASS] {figure_path}")


if __name__ == "__main__":
    main()
