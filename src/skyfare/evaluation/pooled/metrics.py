"""Metrics for final two-block prospective walk-forward analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, roc_auc_score

EPSILON = 1e-7
QUANTILES = np.asarray([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], dtype=float)


def classification_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.int8)
    probability = np.clip(np.asarray(probability, dtype=float), EPSILON, 1.0 - EPSILON)
    auc = float(roc_auc_score(target, probability)) if np.unique(target).size == 2 else 0.5
    edges = np.linspace(0.0, 1.0, 16)
    groups = np.clip(np.digitize(probability, edges[1:-1]), 0, 14)
    ece = 0.0
    for group in range(15):
        mask = groups == group
        if mask.any():
            ece += mask.mean() * abs(float(target[mask].mean()) - float(probability[mask].mean()))
    loss = float(log_loss(target, probability, labels=[0, 1]))
    brier = float(np.mean(np.square(probability - target)))
    return {"log_loss": loss, "brier": brier, "ece15": float(ece), "auc": auc, "proper_score": loss + brier}


def point_metrics(target_price: np.ndarray, prediction_price: np.ndarray) -> dict[str, float]:
    target = np.maximum(np.asarray(target_price, dtype=float), 1.0)
    prediction = np.maximum(np.asarray(prediction_price, dtype=float), 1.0)
    absolute = np.abs(prediction - target)
    return {
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(mean_squared_error(target, prediction) ** 0.5),
        "mape": float(np.mean(absolute / target)),
        "median_ape": float(np.median(absolute / target)),
        "bias": float(np.mean(prediction - target)),
    }


def distribution_metrics(target: np.ndarray, quantiles: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=float)
    raw = np.asarray(quantiles, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != len(QUANTILES):
        raise RuntimeError(f"quantile shape changed: {raw.shape}")
    if not np.isfinite(raw).all():
        raise RuntimeError("quantiles contain non-finite values")
    q = np.maximum.accumulate(raw, axis=1)
    errors = target[:, None] - q
    pinball_rows = np.maximum(QUANTILES[None, :] * errors, (QUANTILES[None, :] - 1.0) * errors)
    return {
        "wis": float(2.0 * pinball_rows.mean()),
        "pinball_mean": float(pinball_rows.mean()),
        "coverage50": float(((target >= q[:, 2]) & (target <= q[:, 4])).mean()),
        "coverage80": float(((target >= q[:, 1]) & (target <= q[:, 5])).mean()),
        "coverage90": float(((target >= q[:, 0]) & (target <= q[:, 6])).mean()),
        "width50": float(np.mean(q[:, 4] - q[:, 2])),
        "width80": float(np.mean(q[:, 5] - q[:, 1])),
        "width90": float(np.mean(q[:, 6] - q[:, 0])),
        "lower_tail90": float(np.mean(target < q[:, 0])),
        "upper_tail90": float(np.mean(target > q[:, 6])),
        "crossing_rate": float(np.mean(np.any(np.diff(raw, axis=1) < 0, axis=1))),
    }


def distribution_row_loss(target: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=float)
    q = np.maximum.accumulate(np.asarray(quantiles, dtype=float), axis=1)
    errors = target[:, None] - q
    return 2.0 * np.mean(
        np.maximum(QUANTILES[None, :] * errors, (QUANTILES[None, :] - 1.0) * errors),
        axis=1,
    )


def ranking_query_metrics(frame: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    work = frame[["query_id", "target_price_vnd", "label_time", "block_id"]].copy()
    work["score"] = np.asarray(score, dtype=float)
    records: list[dict[str, object]] = []
    for query_id, group in work.groupby("query_id", sort=False):
        if len(group) < 2:
            continue
        target_rank = group["target_price_vnd"].rank(method="average", ascending=True)
        relevance = (len(group) - target_rank + 1).to_numpy(dtype=float)
        order = np.argsort(-group["score"].to_numpy(dtype=float), kind="stable")[:5]
        ideal = np.argsort(-relevance, kind="stable")[:5]
        discounts = 1.0 / np.log2(np.arange(2, len(order) + 2))
        dcg = float(np.sum((np.power(2.0, relevance[order]) - 1.0) * discounts))
        idcg = float(np.sum((np.power(2.0, relevance[ideal]) - 1.0) * discounts))
        price = group["target_price_vnd"].to_numpy(dtype=float)
        raw_score = group["score"].to_numpy(dtype=float)
        left, right = np.triu_indices(len(group), 1)
        informative = price[left] != price[right]
        concordance = 0.5
        if informative.any():
            expected = np.sign(price[right][informative] - price[left][informative])
            observed = np.sign(raw_score[left][informative] - raw_score[right][informative])
            concordance = float(np.mean(expected == observed))
        records.append({
            "query_id": str(query_id),
            "ndcg5": dcg / idcg if idcg > 0 else 0.0,
            "pairwise_concordance": concordance,
            "label_time": pd.to_datetime(group["label_time"].iloc[0]),
            "block_id": str(group["block_id"].iloc[0]),
        })
    if not records:
        raise RuntimeError("ranking validation contains no eligible query")
    return pd.DataFrame.from_records(records)


def ranking_metrics(frame: pd.DataFrame, score: np.ndarray) -> dict[str, float]:
    queries = ranking_query_metrics(frame, score)
    return {
        "ndcg5": float(queries["ndcg5"].mean()),
        "pairwise_concordance": float(queries["pairwise_concordance"].mean()),
        "queries": int(len(queries)),
    }


def policy_metrics(frame: pd.DataFrame, action: str, threshold: float, friction_vnd: float) -> tuple[dict[str, float], np.ndarray]:
    current = frame["price_vnd"].to_numpy(dtype=float)
    future = frame["target_price_vnd"].to_numpy(dtype=float)
    if action == "FROZEN":
        wait = frame["probability"].to_numpy(dtype=float) >= threshold
    elif action == "BUY_ALL":
        wait = np.zeros(len(frame), dtype=bool)
    elif action == "WAIT_ALL":
        wait = np.ones(len(frame), dtype=bool)
    else:
        raise KeyError(action)
    oracle = np.minimum(current, future + friction_vnd)
    regret = np.maximum(np.where(wait, future + friction_vnd, current) - oracle, 0.0)
    cutoff = float(np.quantile(regret, 0.95))
    tail = regret[regret >= cutoff]
    return {
        "mean_regret_vnd": float(regret.mean()),
        "cvar95_regret_vnd": float(tail.mean()) if len(tail) else 0.0,
        "p95_regret_vnd": cutoff,
        "wait_rate": float(wait.mean()),
        "rows": int(len(frame)),
    }, regret


def daily_stratified_bootstrap(
    delta: np.ndarray,
    dates: pd.Series,
    blocks: pd.Series,
    seed: int,
    draws: int = 10000,
) -> tuple[dict[str, float], pd.DataFrame]:
    work = pd.DataFrame({
        "delta": np.asarray(delta, dtype=float),
        "date": pd.to_datetime(dates).dt.normalize(),
        "block_id": blocks.astype(str).to_numpy(),
    })
    daily = work.groupby(["block_id", "date"], sort=True)["delta"].mean().reset_index()
    rng = np.random.default_rng(seed)
    sampled_sum = np.zeros(draws, dtype=float)
    sampled_days = 0
    for _, group in daily.groupby("block_id", sort=True):
        values = group["delta"].to_numpy(dtype=float)
        sampled_sum += rng.choice(values, size=(draws, len(values)), replace=True).sum(axis=1)
        sampled_days += len(values)
    samples = sampled_sum / sampled_days
    result = {
        "mean_model_minus_baseline": float(daily["delta"].mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "day_blocks": int(len(daily)),
        "strata": int(daily["block_id"].nunique()),
        "draws": int(draws),
    }
    return result, daily


def block_heterogeneity(daily: pd.DataFrame, seed: int, draws: int = 10000) -> dict[str, float]:
    groups = {name: group["delta"].to_numpy(dtype=float) for name, group in daily.groupby("block_id", sort=True)}
    if set(groups) != {"TEST_1", "TEST_2"}:
        raise RuntimeError(f"unexpected blocks: {sorted(groups)}")
    rng = np.random.default_rng(seed)
    sampled = {}
    for name, values in groups.items():
        sampled[name] = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    difference = sampled["TEST_2"] - sampled["TEST_1"]
    return {
        "test_1_mean_delta": float(groups["TEST_1"].mean()),
        "test_2_mean_delta": float(groups["TEST_2"].mean()),
        "test_2_minus_test_1": float(groups["TEST_2"].mean() - groups["TEST_1"].mean()),
        "difference_ci95_low": float(np.quantile(difference, 0.025)),
        "difference_ci95_high": float(np.quantile(difference, 0.975)),
        "draws": int(draws),
    }
