from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from skyfare.models.classification_runtime import build_inference_frame

ROOT = Path(__file__).resolve().parents[1]


def test_post_freeze_days_are_serving_only() -> None:
    study = json.loads((ROOT / "configs/study.json").read_text(encoding="utf-8"))
    serving = study["serving_snapshot"]
    assert serving["latest_observation_date"] == "2026-08-24"
    assert serving["model_training_cutoff_unchanged"] is True
    assert study["prospective_tests"]["test_2"]["end"] == "2026-08-19"


def test_application_defaults_to_repository_data_and_verified_snapshot() -> None:
    source = (
        ROOT
        / "application/skyfare_inference_demo/skyfare_inference_demo/service.py"
    ).read_text(encoding="utf-8")
    assert 'REPO_ROOT / "data"' in source
    assert 'REPO_ROOT / "artifacts" / "serving"' in source
    assert 'TRAINING_CUTOFF = "2026-08-19"' in source
    assert "post_cutoff_labels_used" in source


def test_classification_inference_frame_is_target_free() -> None:
    row = {
        "offer_id": 7,
        "schedule_slot_id": 11,
        "feature_time": "2026-08-24 10:00:00",
        "session_date": "2026-08-24",
        "session_key": 41_358,
        "session_label": "AM",
        "route": "SGN-HAN",
        "airline": "VN",
        "flight_date": "2026-09-23",
        "departure_time": "2026-09-23 08:00:00",
        "days_until_departure": 30,
        "price_vnd": 2_000_000,
        "collection_era": "TRIP_COM_BROWSER_ERA",
        "departure_period": "MORNING",
        "anchor_source": "STRICT_PRIOR_PEER",
        "flight_day_of_week": 2,
        "flight_month": 9,
        "is_peak_period": 0,
        "departure_time_sin": 0.5,
        "departure_time_cos": -0.5,
        "current_relative_log": 0.1,
        "anchor_support_log1p": 2.0,
        "competitor_airline_count": 4,
        "competitor_offer_count": 12,
        "competitor_min_price_other_airlines": 1_900_000,
        "same_airline_alternative_min_price": 2_100_000,
        "temporal_market_median_price": 1_950_000,
        "prior_market_change_pct_per_day": -0.2,
        "has_prior_market_change": 1,
        "relative_history_eligible": 1,
        "previous_price_same_schedule": 2_050_000,
        "relative_lag_age_hours": 24.0,
        "prior_relative_count": 4,
        "prior_price_volatility_vnd": 50_000,
        "prior_price_trend_vnd_per_dud_day": -2_000,
        "prior_relative_volatility": 0.03,
        "prior_relative_trend_per_dud_day": -0.001,
        "peer_anchor_support": 15,
        "historical_anchor_time": "2026-08-23 10:00:00",
        "anchor_is_fallback": False,
        "temporal_market_time": "2026-08-23 10:00:00",
        "temporal_market_collection_era": "TRIP_COM_BROWSER_ERA",
    }
    frame = build_inference_frame(pd.DataFrame([row]), "2026-08-24")
    assert len(frame) == 1
    assert frame.loc[0, "transition"] == "30->21"
    assert frame.loc[0, "target_observation_state"] == "UNOBSERVED_FUTURE"
    assert frame.loc[0, "target_session_key"] == str(41_358 + 18)
    assert not {
        "DROP_5PCT",
        "target_price_vnd",
        "price_change_vnd",
        "price_change_pct",
        "label_time",
    }.intersection(frame.columns)


def test_application_interpolates_between_canonical_booking_windows() -> None:
    app_root = ROOT / "application/skyfare_inference_demo"
    sys.path.insert(0, str(app_root))
    try:
        from skyfare_inference_demo.service import SkyFareService
    finally:
        sys.path.pop(0)
    service = SkyFareService()
    service._snapshot = pd.DataFrame(
        [
            {
                "session_date": pd.Timestamp("2026-08-24"),
                "session_label": "AM",
                "route": "SGN-HAN",
                "flight_date": pd.Timestamp("2026-08-29"),
                "airline": "VN",
                "departure_HHMM": 800,
                "schedule_slot_id": "5",
                "predicted_price_vnd": 1_000_000.0,
                "baseline_price_vnd": 1_100_000.0,
                "observed_price_vnd": 1_050_000.0,
                "drop_probability": 0.20,
                "ranking_score": 0.4,
            },
            {
                "session_date": pd.Timestamp("2026-08-24"),
                "session_label": "AM",
                "route": "SGN-HAN",
                "flight_date": pd.Timestamp("2026-08-31"),
                "airline": "VN",
                "departure_HHMM": 800,
                "schedule_slot_id": "7",
                "predicted_price_vnd": 1_200_000.0,
                "baseline_price_vnd": 1_300_000.0,
                "observed_price_vnd": 1_250_000.0,
                "drop_probability": 0.40,
                "ranking_score": 0.6,
            },
        ]
    )
    result = service._snapshot_query(
        "2026-08-24", "AM", "SGN-HAN", "2026-08-30", "VN"
    )
    assert result.loc[0, "predicted_price_vnd"] == 1_100_000.0
    assert abs(result.loc[0, "drop_probability"] - 0.30) < 1e-12
    assert result.loc[0, "action"] == "WAIT"
    assert result.loc[0, "dud_support_mode"] == "INTERIOR_OFF_GRID_INTERPOLATED"
