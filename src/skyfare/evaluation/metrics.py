"""Selection metrics with explicit fold and query weighting for V24."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, roc_auc_score

from skyfare.models.selection_contract import QUANTILES


EPSILON = 1e-7


def rearrange(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != len(QUANTILES):
        raise RuntimeError(f"quantile shape changed: {result.shape}")
    if not np.isfinite(result).all():
        raise RuntimeError("quantiles contain non-finite values")
    return np.maximum.accumulate(result, axis=1)


def expected_calibration_error(target: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    target = np.asarray(target, dtype=np.int8)
    probability = np.clip(np.asarray(probability, dtype=float), EPSILON, 1.0 - EPSILON)
    edges = np.linspace(0.0, 1.0, bins + 1)
    groups = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    value = 0.0
    for group in range(bins):
        mask = groups == group
        if mask.any():
            value += mask.mean() * abs(float(target[mask].mean()) - float(probability[mask].mean()))
    return float(value)


def classification_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.int8)
    probability = np.clip(np.asarray(probability, dtype=float), EPSILON, 1.0 - EPSILON)
    auc = float(roc_auc_score(target, probability)) if np.unique(target).size == 2 else 0.5
    return {
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "brier": float(np.mean(np.square(probability - target))),
        "ece15": expected_calibration_error(target, probability),
        "auc": auc,
        "proper_score": float(log_loss(target, probability, labels=[0, 1]) + np.mean(np.square(probability - target))),
    }


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


def pinball(target: np.ndarray, prediction: np.ndarray, level: float) -> float:
    error = np.asarray(target, dtype=float) - np.asarray(prediction, dtype=float)
    return float(np.mean(np.maximum(level * error, (level - 1.0) * error)))


def distribution_metrics(target: np.ndarray, quantiles: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=float)
    q = rearrange(quantiles)
    levels = np.asarray(QUANTILES, dtype=float)
    pinballs = [pinball(target, q[:, index], level) for index, level in enumerate(levels)]
    central50 = (target >= q[:, 2]) & (target <= q[:, 4])
    central80 = (target >= q[:, 1]) & (target <= q[:, 5])
    central90 = (target >= q[:, 0]) & (target <= q[:, 6])
    wis = float(np.mean(pinballs) * 2.0)
    return {
        "wis": wis,
        "pinball_mean": float(np.mean(pinballs)),
        "coverage50": float(central50.mean()),
        "coverage80": float(central80.mean()),
        "coverage90": float(central90.mean()),
        "width50": float(np.mean(q[:, 4] - q[:, 2])),
        "width80": float(np.mean(q[:, 5] - q[:, 1])),
        "width90": float(np.mean(q[:, 6] - q[:, 0])),
        "lower_tail90": float(np.mean(target < q[:, 0])),
        "upper_tail90": float(np.mean(target > q[:, 6])),
        "crossing_rate": float(np.mean(np.any(np.diff(np.asarray(quantiles), axis=1) < 0, axis=1))),
    }


def ranking_metrics(frame: pd.DataFrame, score: np.ndarray) -> dict[str, float]:
    target_column = "target_session_price_vnd" if "target_session_price_vnd" in frame else "target_price_vnd"
    work = frame[["query_id", target_column]].rename(columns={target_column: "target_price_vnd"}).copy()
    work["score"] = np.asarray(score, dtype=float)
    ndcg_values: list[float] = []
    concordance: list[float] = []
    for _, group in work.groupby("query_id", sort=False):
        if len(group) < 2:
            continue
        target_rank = group["target_price_vnd"].rank(method="average", ascending=True)
        relevance = (len(group) - target_rank + 1).to_numpy(dtype=float)
        order = np.argsort(-group["score"].to_numpy(dtype=float), kind="stable")[:5]
        ideal = np.argsort(-relevance, kind="stable")[:5]
        discounts = 1.0 / np.log2(np.arange(2, len(order) + 2))
        dcg = float(np.sum((np.power(2.0, relevance[order]) - 1.0) * discounts))
        idcg = float(np.sum((np.power(2.0, relevance[ideal]) - 1.0) * discounts))
        ndcg_values.append(dcg / idcg if idcg > 0 else 0.0)
        price = group["target_price_vnd"].to_numpy(dtype=float)
        raw_score = group["score"].to_numpy(dtype=float)
        left, right = np.triu_indices(len(group), 1)
        informative = price[left] != price[right]
        if informative.any():
            expected = np.sign(price[right][informative] - price[left][informative])
            observed = np.sign(raw_score[left][informative] - raw_score[right][informative])
            concordance.append(float(np.mean(expected == observed)))
    if not ndcg_values:
        raise RuntimeError("ranking validation contains no eligible query")
    return {
        "ndcg5": float(np.mean(ndcg_values)),
        "pairwise_concordance": float(np.mean(concordance)) if concordance else 0.5,
        "queries": float(len(ndcg_values)),
    }
