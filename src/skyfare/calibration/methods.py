"""Leakage-safe probability and quantile calibration for V24."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression

from skyfare.models.selection_contract import QUANTILES
from skyfare.evaluation.metrics import classification_metrics, distribution_metrics, rearrange


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-values))


def _beta_features(raw: np.ndarray) -> np.ndarray:
    probability = np.clip(sigmoid(raw), 1e-6, 1.0 - 1e-6)
    return np.column_stack([np.log(probability), -np.log1p(-probability)])


def _logistic_fit_predict(
    features: np.ndarray,
    target: np.ndarray,
    valid_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    valid_features = np.asarray(valid_features, dtype=np.float64)
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-6)
    train = np.column_stack([np.ones(len(features)), (features - mean) / scale])
    valid = np.column_stack([np.ones(len(valid_features)), (valid_features - mean) / scale])
    target = np.asarray(target, dtype=np.float64)

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        logits = np.sum(train * weights[None, :], axis=1)
        penalty = 5e-4 * float(np.dot(weights[1:], weights[1:]))
        loss = float(np.mean(np.logaddexp(0.0, logits) - target * logits) + penalty)
        gradient = np.sum(train * (expit(logits) - target)[:, None], axis=0) / len(target)
        gradient[1:] += 1e-3 * weights[1:]
        return loss, gradient

    initial = np.zeros(train.shape[1], dtype=np.float64)
    initial[0] = np.log((target.mean() + 1e-5) / (1.0 - target.mean() + 1e-5))
    fitted = minimize(objective, initial, method="L-BFGS-B", jac=True, options={"maxiter": 300, "ftol": 1e-12})
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError(f"logistic calibration failed: {fitted.message}")
    return (
        expit(np.sum(train * fitted.x[None, :], axis=1)),
        expit(np.sum(valid * fitted.x[None, :], axis=1)),
    )


def calibrate_probability(
    head_target: np.ndarray,
    head_raw: np.ndarray,
    valid_raw: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    target = np.asarray(head_target, dtype=np.int8)
    raw = np.clip(np.asarray(head_raw, dtype=float).reshape(-1), -20.0, 20.0)
    valid = np.clip(np.asarray(valid_raw, dtype=float).reshape(-1), -20.0, 20.0)
    if np.unique(target).size != 2:
        probability = sigmoid(valid)
        return probability, {"recipe": "SIGMOID", "fallback": "single_class_head"}
    recipes: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "SIGMOID": (sigmoid(raw), sigmoid(valid)),
    }
    recipes["PLATT"] = _logistic_fit_predict(raw.reshape(-1, 1), target, valid.reshape(-1, 1))
    recipes["BETA"] = _logistic_fit_predict(_beta_features(raw), target, _beta_features(valid))
    if len(target) >= 200 and min(np.bincount(target)) >= 30:
        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(raw, target)
        recipes["ISOTONIC"] = (isotonic.predict(raw), isotonic.predict(valid))
    scores = {name: classification_metrics(target, pair[0]) for name, pair in recipes.items()}
    order = {"SIGMOID": 0, "PLATT": 1, "BETA": 2, "ISOTONIC": 3}
    selected = min(scores, key=lambda name: (scores[name]["proper_score"] + order[name] * 1e-6, order[name]))
    return np.clip(recipes[selected][1], 1e-7, 1.0 - 1e-7), {
        "recipe": selected,
        "head_scores": scores,
        "fit_population": int(len(target)),
    }


def _group_tokens(frame: pd.DataFrame) -> pd.Series:
    dud = pd.cut(
        pd.to_numeric(frame["query_dud"], errors="coerce"),
        [-np.inf, 3, 7, 14, 30, 60, np.inf],
        labels=["D0_3", "D4_7", "D8_14", "D15_30", "D31_60", "D61P"],
    ).astype("string")
    regime = frame["regime"].astype("string").fillna("__NA__")
    return regime + "|" + dud


def _conformal_offsets(target: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    residual = np.asarray(target, dtype=float)[:, None] - rearrange(quantiles)
    levels = np.asarray(QUANTILES, dtype=float)
    return np.asarray([np.quantile(residual[:, index], level, method="higher") for index, level in enumerate(levels)])


def calibrate_quantiles(
    head: pd.DataFrame,
    head_quantiles: np.ndarray,
    valid: pd.DataFrame,
    valid_quantiles: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    target = head["target_anchor_relative_log"].to_numpy(dtype=float)
    head_raw = rearrange(head_quantiles)
    valid_raw = rearrange(valid_quantiles)
    candidates: dict[str, np.ndarray] = {"RAW": valid_raw}
    global_offsets = _conformal_offsets(target, head_raw)
    candidates["GLOBAL_CONFORMAL"] = rearrange(valid_raw + global_offsets)

    head_tokens = _group_tokens(head)
    valid_tokens = _group_tokens(valid)
    mondrian = valid_raw.copy()
    group_offsets: dict[str, list[float]] = {}
    for token in sorted(head_tokens.dropna().unique()):
        mask = head_tokens.eq(token).to_numpy()
        if mask.sum() < 100:
            continue
        offsets = _conformal_offsets(target[mask], head_raw[mask])
        group_offsets[str(token)] = offsets.tolist()
        target_mask = valid_tokens.eq(token).to_numpy()
        mondrian[target_mask] += offsets
    missing = ~valid_tokens.isin(group_offsets).to_numpy()
    mondrian[missing] += global_offsets
    candidates["MONDRIAN_CONFORMAL"] = rearrange(mondrian)

    head_scores = {
        "RAW": distribution_metrics(target, head_raw),
        "GLOBAL_CONFORMAL": distribution_metrics(target, rearrange(head_raw + global_offsets)),
    }
    head_mondrian = head_raw.copy()
    for token, offsets in group_offsets.items():
        head_mondrian[head_tokens.eq(token).to_numpy()] += np.asarray(offsets)
    head_mondrian[~head_tokens.isin(group_offsets).to_numpy()] += global_offsets
    head_scores["MONDRIAN_CONFORMAL"] = distribution_metrics(target, rearrange(head_mondrian))
    complexity = {"RAW": 0, "GLOBAL_CONFORMAL": 1, "MONDRIAN_CONFORMAL": 2}
    selected = min(
        head_scores,
        key=lambda name: (
            head_scores[name]["wis"]
            + 0.25 * abs(head_scores[name]["coverage90"] - 0.90)
            + 0.15 * abs(head_scores[name]["coverage80"] - 0.80)
            + complexity[name] * 1e-7,
            complexity[name],
        ),
    )
    return candidates[selected], {
        "recipe": selected,
        "head_scores": head_scores,
        "global_offsets": global_offsets.tolist(),
        "mondrian_groups": group_offsets,
        "fit_population": int(len(head)),
    }
