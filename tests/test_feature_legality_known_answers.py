from __future__ import annotations

import pandas as pd

from skyfare.features.candidate_feature_contract import (
    _causal_market_anchor,
    _same_schedule_history,
)


def _offers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "collection_era": ["FLI_LIBRARY_ERA"] * 3,
            "route": ["SGN-HAN"] * 3,
            "airline": ["VN"] * 3,
            "days_until_departure": [30] * 3,
            "session_key": [1, 2, 3],
            "schedule_slot_id": [99] * 3,
            "session_label": ["AM", "PM", "AM"],
            "feature_time": pd.to_datetime(
                ["2026-03-23 10:00", "2026-03-23 19:00", "2026-03-24 10:00"]
            ),
            "price_vnd": [1_000_000.0, 1_100_000.0, 1_200_000.0],
        }
    )


def test_same_schedule_features_use_only_prior_batches() -> None:
    result = _same_schedule_history(_offers()).sort_values("session_key")
    assert result["previous_price_same_schedule"].tolist()[1:] == [1_000_000.0, 1_100_000.0]
    assert result["prior_observation_count"].tolist() == [0.0, 1.0, 2.0]
    available = result["previous_schedule_time"].notna()
    assert result.loc[available, "previous_schedule_time"].lt(
        result.loc[available, "feature_time"]
    ).all()


def test_market_anchor_is_shifted_one_complete_batch() -> None:
    result = _causal_market_anchor(_offers()).sort_values("session_key")
    assert pd.isna(result.iloc[0]["temporal_market_median_price"])
    assert result.iloc[1]["temporal_market_median_price"] == 1_000_000.0
    assert result.iloc[2]["temporal_market_median_price"] == 1_100_000.0
    available = result["temporal_market_time"].notna()
    source = _offers().set_index("session_key")["feature_time"]
    assert all(
        row.temporal_market_time < source.loc[row.session_key]
        for row in result.loc[available].itertuples()
    )
