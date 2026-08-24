"""Shared event runtime for standard-only Pilot115 closing."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import betaln, expit, logit
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from skyfare.models.classification_event_contract import (
    DROP_RECALL_FLOOR,
    LABELS,
    NO_DROP_RECALL_FLOOR,
    UNSAFE_BUY_WEIGHT,
    WAIT_FRICTION_VND,
)


@dataclass(frozen=True)
class Paths:
    root: Path
    inputs: Path
    outputs: Path


def discover_root(value: str | None) -> Path:
    return Path(value).resolve() if value else Path(__file__).resolve().parents[1]


def bundle_paths(root: Path) -> Paths:
    return Paths(root, root / "inputs", root / "outputs")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def load_inputs(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = {
        "offers": paths.inputs / "anchor_relative_offers.parquet",
        "exact": paths.inputs / "exact_next_targets.parquet",
        "ledger": paths.inputs / "exact_observability.parquet",
        "manifest": paths.inputs / "product_manifest.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing standard closing inputs: " + ", ".join(missing))
    manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
    if manifest.get("nonstd_accessed") is not False:
        raise RuntimeError("Manifest does not prove nonstd_accessed=false")
    for item in manifest["source_manifest"]:
        if "nonstd" in item["path"].lower():
            raise RuntimeError(f"Forbidden source in manifest: {item['path']}")
    offers = pd.read_parquet(required["offers"])
    exact = pd.read_parquet(required["exact"])
    ledger = pd.read_parquet(required["ledger"])
    for frame in [offers, exact, ledger]:
        for column in [
            "feature_time",
            "label_time",
            "session_date",
            "flight_date",
            "departure_time",
            "expected_target_session_date",
        ]:
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return offers, exact, ledger, manifest


def _safe_log_ratio(left: pd.Series, right: pd.Series) -> pd.Series:
    a = pd.to_numeric(left, errors="coerce").clip(lower=1.0)
    b = pd.to_numeric(right, errors="coerce").clip(lower=1.0)
    return np.log(a / b)


def enrich_offers(offers: pd.DataFrame) -> pd.DataFrame:
    result = offers.copy()
    result["log_price_vnd"] = np.log(
        pd.to_numeric(result["price_vnd"], errors="coerce").clip(lower=1.0)
    )
    result["log_current_over_competitor_min"] = _safe_log_ratio(
        result["price_vnd"], result["competitor_min_price_other_airlines"]
    )
    result["log_same_airline_alt_over_current"] = _safe_log_ratio(
        result["same_airline_alternative_min_price"], result["price_vnd"]
    )
    result["previous_relative_log"] = _safe_log_ratio(
        result["previous_price_same_schedule"], result["temporal_market_median_price"]
    )
    current_market = (
        result.groupby(
            ["session_key", "route", "airline", "days_until_departure"],
            observed=True,
        )["price_vnd"]
        .transform("median")
    )
    result["market_shift_log"] = _safe_log_ratio(
        current_market, result["temporal_market_median_price"]
    )
    result["prior_relative_volatility"] = (
        pd.to_numeric(result["prior_price_volatility_vnd"], errors="coerce")
        / pd.to_numeric(result["temporal_market_median_price"], errors="coerce")
    )
    result["prior_relative_trend_per_dud_day"] = (
        pd.to_numeric(result["prior_price_trend_vnd_per_dud_day"], errors="coerce")
        / pd.to_numeric(result["temporal_market_median_price"], errors="coerce")
    )
    return result


def apply_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    saving = (
        pd.to_numeric(result["price_vnd"], errors="coerce")
        - pd.to_numeric(result["target_price_vnd"], errors="coerce")
    )
    result["saving_vnd"] = saving
    current = pd.to_numeric(result["price_vnd"], errors="coerce")
    for name, rule in LABELS.items():
        if rule["kind"] == "ABSOLUTE":
            threshold = np.full(len(result), float(rule["vnd"]))
        elif rule["kind"] == "PERCENT":
            threshold = current.to_numpy(dtype=float) * float(rule["pct"])
        else:
            threshold = np.maximum(
                current.to_numpy(dtype=float) * float(rule["pct"]),
                float(rule["vnd"]),
            )
        result[name] = saving.to_numpy(dtype=float) >= threshold
    return result


def event_frame(offers: pd.DataFrame, exact: pd.DataFrame) -> pd.DataFrame:
    enriched = enrich_offers(offers)
    excluded = {
        "offer_id",
        "feature_time",
        "session_date",
        "price_vnd",
        "flight_date",
    }
    columns = [column for column in enriched.columns if column not in excluded]
    result = exact.merge(
        enriched[["offer_id", *columns]],
        on="offer_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_offer"),
    )
    if "days_until_departure" not in result:
        result["days_until_departure"] = result["current_dud"]
    if "transition" not in result:
        if "horizon_type" in result and result["horizon_type"].astype(
            str
        ).str.contains("INTRADAY", case=False, na=False).any():
            result["transition"] = "DUD1_AM->PM"
        else:
            result["transition"] = (
                result["current_dud"].astype("Int64").astype(str)
                + "->"
                + result["target_dud"].astype("Int64").astype(str)
            )
    result = apply_labels(result)
    result["outcome_time"] = result["label_time"]
    required = [
        "feature_time",
        "outcome_time",
        "price_vnd",
        "target_price_vnd",
        "route",
        "airline",
        "transition",
    ]
    result = result.dropna(subset=required)
    if result["feature_time"].ge(result["outcome_time"]).any():
        raise RuntimeError("Event feature_time is not strictly before outcome_time")
    return result.sort_values(["feature_time", "offer_id"]).reset_index(drop=True)


def reobservation_frame(offers: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    mature = ledger[
        ledger["observation_state"].isin(["OBSERVED", "MATURE_NOT_REOBSERVED"])
    ].copy()
    mature["REOBSERVED"] = mature["observation_state"].eq("OBSERVED")
    mature["outcome_time"] = (
        mature["expected_target_session_date"].dt.normalize()
        + pd.Timedelta(days=1)
    )
    enriched = enrich_offers(offers)
    excluded = {
        "offer_id",
        "feature_time",
        "session_date",
        "price_vnd",
        "flight_date",
    }
    columns = [column for column in enriched.columns if column not in excluded]
    result = mature.merge(
        enriched[["offer_id", *columns]],
        on="offer_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_offer"),
    )
    result["days_until_departure"] = result["current_dud"]
    result["horizon_gap_days"] = (
        pd.to_numeric(result["current_dud"], errors="coerce")
        - pd.to_numeric(result["target_dud"], errors="coerce")
    )
    if result["feature_time"].ge(result["outcome_time"]).any():
        raise RuntimeError("Reobservation outcome maturity is not after feature_time")
    return result.sort_values(["feature_time", "offer_id"]).reset_index(drop=True)


def split_rows(
    frame: pd.DataFrame, fold: dict[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp(fold["validation_start"])
    end = pd.Timestamp(fold["validation_end"]) + pd.Timedelta(days=1)
    train = frame[
        frame["feature_time"].lt(start) & frame["outcome_time"].lt(start)
    ].copy()
    valid = frame[
        frame["feature_time"].ge(start) & frame["feature_time"].lt(end)
    ].copy()
    if train.empty or valid.empty:
        raise RuntimeError(f"Empty split for {fold['fold']}")
    if train["outcome_time"].max() >= start:
        raise RuntimeError(f"Outcome-time purge failed for {fold['fold']}")
    return train, valid


def calibration_split(
    train: pd.DataFrame, target: str, fraction: float = 0.20
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = np.array(sorted(train["feature_time"].dt.normalize().unique()))
    count = max(3, int(math.ceil(len(dates) * fraction)))
    if len(dates) <= count + 3:
        raise RuntimeError("Insufficient dates for disjoint temporal calibration")
    start = pd.Timestamp(dates[-count])
    fit = train[
        train["feature_time"].lt(start) & train["outcome_time"].lt(start)
    ].copy()
    calibration = train[train["feature_time"].ge(start)].copy()
    if fit.empty or calibration.empty:
        raise RuntimeError("Empty fit/calibration split")
    if fit["outcome_time"].max() >= calibration["feature_time"].min():
        raise RuntimeError("Internal calibration purge failed")
    if fit[target].nunique() < 2 or calibration[target].nunique() < 2:
        raise RuntimeError(f"Internal split lacks both classes for {target}")
    return fit, calibration


def _beta_binomial_objective(
    log_kappa: float, successes: np.ndarray, totals: np.ndarray, means: np.ndarray
) -> float:
    kappa = math.exp(log_kappa)
    mu = np.clip(means, 1e-6, 1 - 1e-6)
    alpha = mu * kappa
    beta = (1 - mu) * kappa
    value = betaln(successes + alpha, totals - successes + beta) - betaln(alpha, beta)
    return -float(np.sum(value))


def estimate_kappa(
    successes: np.ndarray, totals: np.ndarray, means: np.ndarray
) -> float:
    valid = totals > 0
    if int(valid.sum()) < 2:
        return 100.0
    result = minimize_scalar(
        _beta_binomial_objective,
        bounds=(math.log(0.1), math.log(100_000.0)),
        args=(successes[valid], totals[valid], means[valid]),
        method="bounded",
        options={"xatol": 1e-5},
    )
    return float(math.exp(result.x)) if result.success else 100.0


def hierarchy_probability(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target: str,
    mode: str,
) -> tuple[np.ndarray, dict[str, float]]:
    y = train[target].astype(float)
    global_rate = float((y.sum() + 0.5) / (len(y) + 1.0))
    if mode == "GLOBAL_RATE":
        return np.full(len(valid), global_rate), {
            "global_rate": global_rate,
            "transition_kappa": math.nan,
            "granular_kappa": math.nan,
        }

    transition = (
        train.groupby("transition", observed=True)[target]
        .agg(["sum", "count"])
        .astype(float)
    )
    granular = (
        train.groupby(["route", "airline", "transition"], observed=True)[target]
        .agg(["sum", "count"])
        .astype(float)
    )
    if mode == "FIXED_HIERARCHY":
        transition_kappa, granular_kappa = 3000.0, 1600.0
    elif mode == "EMPIRICAL_BAYES_HIERARCHY":
        transition_means = np.full(len(transition), global_rate)
        transition_kappa = estimate_kappa(
            transition["sum"].to_numpy(),
            transition["count"].to_numpy(),
            transition_means,
        )
        transition_parent = (
            transition["sum"] + transition_kappa * global_rate
        ) / (transition["count"] + transition_kappa)
        parent_lookup = transition_parent.to_dict()
        granular_means = np.array(
            [parent_lookup[key[2]] for key in granular.index], dtype=float
        )
        granular_kappa = estimate_kappa(
            granular["sum"].to_numpy(),
            granular["count"].to_numpy(),
            granular_means,
        )
    else:
        raise KeyError(mode)

    transition_prob = (
        transition["sum"] + transition_kappa * global_rate
    ) / (transition["count"] + transition_kappa)
    transition_lookup = transition_prob.to_dict()
    probabilities = []
    for row in valid[["route", "airline", "transition"]].itertuples(index=False):
        parent = float(transition_lookup.get(row.transition, global_rate))
        key = (row.route, row.airline, row.transition)
        if key in granular.index:
            stats = granular.loc[key]
            value = (float(stats["sum"]) + granular_kappa * parent) / (
                float(stats["count"]) + granular_kappa
            )
        else:
            value = parent
        probabilities.append(value)
    return np.clip(np.asarray(probabilities), 1e-6, 1 - 1e-6), {
        "global_rate": global_rate,
        "transition_kappa": float(transition_kappa),
        "granular_kappa": float(granular_kappa),
    }


def classification_metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    predicted = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "rows": int(len(y)),
        "prevalence": float(y.mean()),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else math.nan,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "drop_precision": float(precision_score(y, predicted, zero_division=0)),
        "drop_recall": float(recall_score(y, predicted, zero_division=0)),
        "no_drop_precision": float(
            precision_score(1 - y, ~predicted, zero_division=0)
        ),
        "no_drop_recall": float(recall_score(1 - y, ~predicted, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def realized_vnd_cost(frame: pd.DataFrame, probability: np.ndarray, threshold: float) -> float:
    wait = np.asarray(probability) >= threshold
    current = frame["price_vnd"].to_numpy(dtype=float)
    target = frame["target_price_vnd"].to_numpy(dtype=float)
    buy_loss = UNSAFE_BUY_WEIGHT * np.maximum(current - target, 0.0)
    wait_loss = np.maximum(target - current, 0.0) + WAIT_FRICTION_VND
    return float(np.mean(np.where(wait, wait_loss, buy_loss)))


def tune_threshold(
    frame: pd.DataFrame,
    target: str,
    probability: np.ndarray,
) -> dict[str, float | bool]:
    rows = []
    y = frame[target].to_numpy(dtype=int)
    for threshold in np.linspace(0.01, 0.99, 197):
        metrics = classification_metrics(y, probability, float(threshold))
        metrics["threshold"] = float(threshold)
        metrics["mean_vnd_cost"] = realized_vnd_cost(frame, probability, float(threshold))
        metrics["min_class_recall"] = min(
            metrics["drop_recall"], metrics["no_drop_recall"]
        )
        metrics["floor_pass"] = (
            metrics["drop_recall"] >= DROP_RECALL_FLOOR
            and metrics["no_drop_recall"] >= NO_DROP_RECALL_FLOOR
        )
        rows.append(metrics)
    feasible = [row for row in rows if row["floor_pass"]]
    if feasible:
        selected = min(
            feasible,
            key=lambda row: (
                row["mean_vnd_cost"],
                -row["min_class_recall"],
                abs(row["threshold"] - 0.5),
            ),
        )
    else:
        selected = max(
            rows,
            key=lambda row: (
                row["min_class_recall"],
                -row["mean_vnd_cost"],
                -abs(row["threshold"] - 0.5),
            ),
        )
    selected["tuning_feasible"] = bool(feasible)
    return selected


def probability_logit(probability: np.ndarray) -> np.ndarray:
    return logit(np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6))


def probability_expit(score: np.ndarray) -> np.ndarray:
    return expit(np.asarray(score, dtype=float))
