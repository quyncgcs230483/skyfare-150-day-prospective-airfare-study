from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skyfare.evaluation.metrics import (
    classification_metrics,
    distribution_metrics,
    point_metrics,
    ranking_metrics,
    rearrange,
)


def test_classification_metrics_reward_perfect_ordering() -> None:
    result = classification_metrics(np.array([0, 0, 1, 1]), np.array([0.0, 0.0, 1.0, 1.0]))
    assert result["auc"] == 1.0
    assert result["brier"] < 1e-12
    assert result["log_loss"] < 1e-6


def test_point_metrics_include_exact_r_squared() -> None:
    result = point_metrics(np.array([100.0, 200.0, 300.0]), np.array([100.0, 200.0, 300.0]))
    assert result == {
        "mae": 0.0,
        "rmse": 0.0,
        "mape": 0.0,
        "median_ape": 0.0,
        "bias": 0.0,
        "r2": 1.0,
    }


def test_distribution_metrics_have_zero_loss_for_exact_quantiles() -> None:
    target = np.array([100.0, 200.0])
    quantiles = np.repeat(target[:, None], 7, axis=1)
    result = distribution_metrics(target, quantiles)
    assert result["wis"] == 0.0
    assert result["pinball_mean"] == 0.0
    assert result["coverage50"] == 1.0
    assert result["coverage80"] == 1.0
    assert result["coverage90"] == 1.0


def test_quantile_rearrangement_is_monotone_and_shape_checked() -> None:
    crossed = np.array([[7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
    result = rearrange(crossed)
    assert np.all(np.diff(result, axis=1) >= 0)
    with pytest.raises(RuntimeError):
        rearrange(np.ones((2, 6)))


def test_ranking_metrics_match_perfect_price_order() -> None:
    frame = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q1"],
            "target_price_vnd": [100.0, 200.0, 300.0],
        }
    )
    result = ranking_metrics(frame, np.array([3.0, 2.0, 1.0]))
    assert result["ndcg5"] == 1.0
    assert result["pairwise_concordance"] == 1.0
    assert result["queries"] == 1.0
