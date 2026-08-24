"""Serializable ensemble calibration fitted on temporal training head only."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression


def sigmoid(values: np.ndarray) -> np.ndarray:
    return expit(np.clip(np.asarray(values, dtype=float), -35.0, 35.0))


def _features(raw: np.ndarray, recipe: str) -> np.ndarray:
    raw = np.asarray(raw, dtype=float).reshape(-1)
    if recipe == "PLATT":
        return raw[:, None]
    probability = np.clip(sigmoid(raw), 1e-6, 1.0 - 1e-6)
    return np.column_stack([np.log(probability), -np.log1p(-probability)])


def _fit_logistic(features: np.ndarray, target: np.ndarray) -> dict[str, object]:
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-6)
    design = np.column_stack([np.ones(len(features)), (features - mean) / scale])
    target = np.asarray(target, dtype=float)

    def objective(weights):
        logits = design @ weights
        loss = np.mean(np.logaddexp(0.0, logits) - target * logits) + 5e-4 * np.dot(weights[1:], weights[1:])
        gradient = design.T @ (expit(logits) - target) / len(target)
        gradient[1:] += 1e-3 * weights[1:]
        return float(loss), gradient

    initial = np.zeros(design.shape[1], dtype=float)
    initial[0] = np.log((target.mean() + 1e-5) / (1.0 - target.mean() + 1e-5))
    fitted = minimize(objective, initial, method="L-BFGS-B", jac=True, options={"maxiter": 300, "ftol": 1e-12})
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError(f"logistic calibration failed: {fitted.message}")
    return {"mean": mean.tolist(), "scale": scale.tolist(), "weights": fitted.x.tolist()}


def apply_probability(raw: np.ndarray, state: dict[str, object]) -> np.ndarray:
    recipe = str(state["recipe"])
    raw = np.asarray(raw, dtype=float).reshape(-1)
    if recipe == "SIGMOID":
        return sigmoid(raw)
    if recipe == "ISOTONIC":
        return np.interp(raw, state["x_thresholds"], state["y_thresholds"])
    features = _features(raw, recipe)
    mean = np.asarray(state["mean"], dtype=float)
    scale = np.asarray(state["scale"], dtype=float)
    design = np.column_stack([np.ones(len(features)), (features - mean) / scale])
    return expit(design @ np.asarray(state["weights"], dtype=float))


def _proper_score(target: np.ndarray, probability: np.ndarray) -> float:
    probability = np.clip(probability, 1e-7, 1.0 - 1e-7)
    logloss = -np.mean(target * np.log(probability) + (1 - target) * np.log1p(-probability))
    brier = np.mean((probability - target) ** 2)
    return float(logloss + brier)


def fit_probability(target: np.ndarray, raw: np.ndarray) -> dict[str, object]:
    target = np.asarray(target, dtype=np.int8)
    raw = np.clip(np.asarray(raw, dtype=float).reshape(-1), -20.0, 20.0)
    states: dict[str, dict[str, object]] = {"SIGMOID": {"recipe": "SIGMOID"}}
    for recipe in ("PLATT", "BETA"):
        states[recipe] = {"recipe": recipe, **_fit_logistic(_features(raw, recipe), target)}
    if len(target) >= 200 and min(np.bincount(target)) >= 30:
        model = IsotonicRegression(out_of_bounds="clip").fit(raw, target)
        states["ISOTONIC"] = {
            "recipe": "ISOTONIC",
            "x_thresholds": model.X_thresholds_.astype(float).tolist(),
            "y_thresholds": model.y_thresholds_.astype(float).tolist(),
        }
    complexity = {"SIGMOID": 0, "PLATT": 1, "BETA": 2, "ISOTONIC": 3}
    scores = {name: _proper_score(target, apply_probability(raw, state)) for name, state in states.items()}
    selected = min(scores, key=lambda name: (scores[name] + complexity[name] * 1e-6, complexity[name]))
    return {**states[selected], "head_scores": scores, "fit_population": int(len(target))}


def _rearrange(values: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(np.asarray(values, dtype=float), axis=1)


def _wis(target: np.ndarray, quantiles: np.ndarray) -> float:
    levels = np.asarray([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    error = target[:, None] - _rearrange(quantiles)
    return float(2.0 * np.mean(np.maximum(levels * error, (levels - 1.0) * error)))


def _tokens(frame: pd.DataFrame) -> pd.Series:
    dud = pd.cut(
        pd.to_numeric(frame["query_dud"], errors="coerce"),
        [-np.inf, 3, 7, 14, 30, 60, np.inf],
        labels=["D0_3", "D4_7", "D8_14", "D15_30", "D31_60", "D61P"],
    ).astype("string")
    return frame["regime"].astype("string").fillna("__NA__") + "|" + dud


def _offsets(target: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    levels = np.asarray([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    residual = target[:, None] - _rearrange(quantiles)
    return np.asarray([np.quantile(residual[:, i], level, method="higher") for i, level in enumerate(levels)])


def apply_quantiles(frame: pd.DataFrame, raw: np.ndarray, state: dict[str, object]) -> np.ndarray:
    result = np.asarray(raw, dtype=float).copy()
    recipe = str(state["recipe"])
    if recipe == "RAW":
        return _rearrange(result)
    global_offsets = np.asarray(state["global_offsets"], dtype=float)
    if recipe == "GLOBAL_CONFORMAL":
        return _rearrange(result + global_offsets)
    tokens = _tokens(frame)
    groups = state["mondrian_groups"]
    for token in tokens.unique():
        mask = tokens.eq(token).to_numpy()
        result[mask] += np.asarray(groups.get(str(token), global_offsets), dtype=float)
    return _rearrange(result)


def fit_quantiles(frame: pd.DataFrame, raw: np.ndarray) -> dict[str, object]:
    target = frame["target_anchor_relative_log"].to_numpy(dtype=float)
    raw = _rearrange(raw)
    global_offsets = _offsets(target, raw)
    tokens = _tokens(frame)
    groups: dict[str, list[float]] = {}
    for token in sorted(tokens.dropna().unique()):
        mask = tokens.eq(token).to_numpy()
        if mask.sum() >= 100:
            groups[str(token)] = _offsets(target[mask], raw[mask]).tolist()
    states = {
        "RAW": {"recipe": "RAW", "global_offsets": global_offsets.tolist(), "mondrian_groups": groups},
        "GLOBAL_CONFORMAL": {"recipe": "GLOBAL_CONFORMAL", "global_offsets": global_offsets.tolist(), "mondrian_groups": groups},
        "MONDRIAN_CONFORMAL": {"recipe": "MONDRIAN_CONFORMAL", "global_offsets": global_offsets.tolist(), "mondrian_groups": groups},
    }
    scores = {name: _wis(target, apply_quantiles(frame, raw, state)) for name, state in states.items()}
    complexity = {"RAW": 0, "GLOBAL_CONFORMAL": 1, "MONDRIAN_CONFORMAL": 2}
    selected = min(scores, key=lambda name: (scores[name] + complexity[name] * 1e-7, complexity[name]))
    return {**states[selected], "head_wis": scores, "fit_population": int(len(frame))}
