#!/usr/bin/env python3
"""Build full 128-day frames from sealed standard-only offers."""

from __future__ import annotations

import json
import shutil

import pandas as pd

from skyfare.core.integrity import sha256, write_json_atomic
from skyfare.core.paths import DataLayout
from skyfare.core.sources import normalize_source_columns
from skyfare.models import classification_runtime as exact_runtime
from skyfare.models import regression_runtime as fare_runtime
from skyfare.models.classification_contract import (
    CATEGORICAL_FEATURES as EXACT_CATEGORICAL,
)
from skyfare.models.classification_contract import (
    FORBIDDEN_PREDICTORS as EXACT_FORBIDDEN,
)
from skyfare.models.classification_contract import (
    NUMERIC_FEATURES as EXACT_NUMERIC,
)
from skyfare.models.classification_contract import (
    TARGET_NAME,
)
from skyfare.models.regression_contract import (
    CATEGORICAL_FEATURES as FARE_CATEGORICAL,
)
from skyfare.models.regression_contract import (
    FORBIDDEN_PREDICTORS as FARE_FORBIDDEN,
)
from skyfare.models.regression_contract import (
    NUMERIC_FEATURES as FARE_NUMERIC,
)
from skyfare.preparation.data_contract import (
    CANONICAL_DUDS,
    CONTRACT_VERSION,
    DATA_CUTOFF,
    EXACT_BRIDGE_POLICY,
    OBSERVED_DAYS,
    require_vast,
)

LAYOUT = DataLayout.resolve()
SOURCE = LAYOUT.standardised / "standard_offers.parquet"
DEVELOPMENT_ROOT = LAYOUT.processed / "development"
EXACT_FRAME = DEVELOPMENT_ROOT / "classification_training_frame.parquet"
FARE_FRAME = DEVELOPMENT_ROOT / "regression_training_frame.parquet"
FARE_LEDGER = DEVELOPMENT_ROOT / "candidate_coverage_ledger.parquet"
OFFERS_COPY = DEVELOPMENT_ROOT / "standard_offers.parquet"
MANIFEST = LAYOUT.artifacts / "data_preparation" / "development_frame_manifest.json"


def _times(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in (
        "feature_time",
        "label_time",
        "session_date",
        "source_session_date",
        "target_session_date",
        "flight_date",
        "departure_time",
        "previous_schedule_time",
        "previous_relative_time",
        "template_previous_time",
    ):
        if column in result:
            result[column] = pd.to_datetime(result[column], errors="coerce")
    return result


def _assert_predictor_contract(
    frame: pd.DataFrame,
    categorical: tuple[str, ...] | list[str],
    numeric: tuple[str, ...] | list[str],
    forbidden: tuple[str, ...] | list[str],
    task: str,
) -> None:
    predictors = set(categorical) | set(numeric)
    missing = sorted(predictors.difference(frame.columns))
    leaked = sorted(predictors.intersection(forbidden))
    if missing:
        raise RuntimeError(f"{task}: missing predictors: {missing}")
    if leaked:
        raise RuntimeError(f"{task}: forbidden predictors: {leaked}")
    if not frame["feature_time"].lt(frame["label_time"]).all():
        raise RuntimeError(f"{task}: point-in-time contract failed")


def _assert_offers(offers: pd.DataFrame) -> dict[str, object]:
    if offers.empty:
        raise RuntimeError("Standard 128-day offers are empty")
    dates = pd.to_datetime(offers["session_date"], errors="coerce")
    if dates.nunique() != OBSERVED_DAYS:
        raise RuntimeError(
            f"Expected {OBSERVED_DAYS} observed days; got {dates.nunique()}"
        )
    cutoff = pd.Timestamp(DATA_CUTOFF)
    if dates.max().normalize() != cutoff:
        raise RuntimeError(f"Wrong cutoff: {dates.max()}")
    latest = offers[dates.dt.normalize().eq(cutoff)].copy()
    labels = set(latest["session_label"].astype(str))
    if labels != {"AM", "PM"}:
        raise RuntimeError(f"Cutoff batches incomplete: {sorted(labels)}")
    if latest["route"].nunique() != 20:
        raise RuntimeError("Cutoff route coverage is not 20")
    if latest["airline"].nunique() != 5:
        raise RuntimeError("Cutoff airline coverage is not 5")
    duds = set(
        pd.to_numeric(
            latest["days_until_departure"], errors="coerce"
        ).dropna().astype(int)
    )
    if duds != set(CANONICAL_DUDS):
        raise RuntimeError(f"Cutoff DUD coverage mismatch: {sorted(duds)}")
    if "price_tier" in offers:
        tiers = set(offers["price_tier"].astype(str).str.upper())
        if tiers != {"STANDARD"}:
            raise RuntimeError(f"Non-standard price tier entered: {tiers}")
    if "is_standard" in offers:
        values = offers["is_standard"].fillna(False).astype(bool)
        if not values.all():
            raise RuntimeError("Non-standard offer entered sealed input")
    return {
        "rows": len(offers),
        "observed_days": int(dates.nunique()),
        "first_date": dates.min().normalize(),
        "cutoff": dates.max().normalize(),
        "cutoff_batches": sorted(labels),
        "cutoff_routes": int(latest["route"].nunique()),
        "cutoff_airlines": int(latest["airline"].nunique()),
        "cutoff_duds": sorted(duds),
        "collection_era_rows": {
            str(key): int(value)
            for key, value in offers["collection_era"]
            .astype(str)
            .value_counts()
            .items()
        },
    }


def build() -> dict[str, object]:
    require_vast()
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    DEVELOPMENT_ROOT.mkdir(parents=True, exist_ok=True)
    offers = normalize_source_columns(_times(pd.read_parquet(SOURCE)))
    offers_summary = _assert_offers(offers)

    if not OFFERS_COPY.is_file() or sha256(OFFERS_COPY) != sha256(SOURCE):
        temporary = OFFERS_COPY.with_suffix(".tmp.parquet")
        shutil.copy2(SOURCE, temporary)
        temporary.replace(OFFERS_COPY)
    if sha256(OFFERS_COPY) != sha256(SOURCE):
        raise RuntimeError("Sealed standard input changed during staging")

    exact_runtime.CUTOFF = DATA_CUTOFF
    exact = _times(exact_runtime.build_exact_frame(offers))
    if exact[TARGET_NAME].isna().any():
        raise RuntimeError("Exact target contains missing values")
    _assert_predictor_contract(
        exact,
        EXACT_CATEGORICAL,
        EXACT_NUMERIC,
        EXACT_FORBIDDEN,
        "Exact",
    )
    exact.to_parquet(EXACT_FRAME, index=False, compression="zstd")
    exact_fit = exact[
        exact["source_target_era_transition"].ne(
            "FLI_LIBRARY_ERA->TRIP_COM_BROWSER_ERA"
        )
    ]
    if EXACT_BRIDGE_POLICY != "WITHIN_ERA_ONLY":
        raise RuntimeError("Unexpected Exact bridge policy")

    fare_runtime.CUTOFF = DATA_CUTOFF
    fare_runtime.STANDARD_OFFERS_CACHE = OFFERS_COPY
    fare_runtime.RECURRENT_SEQUENCE_SOURCE = OFFERS_COPY
    fare_runtime.FRAME_CACHE = FARE_FRAME
    fare_runtime.LEDGER_CACHE = FARE_LEDGER
    fare_runtime.legacy.CUTOFF = DATA_CUTOFF
    fare_runtime.legacy.STANDARD_OFFERS_CACHE = OFFERS_COPY
    fare, ledger = fare_runtime.build_training_frame()
    fare = _times(fare)
    ledger = _times(ledger)
    _assert_predictor_contract(
        fare,
        FARE_CATEGORICAL,
        FARE_NUMERIC,
        FARE_FORBIDDEN,
        "Fare",
    )
    if fare["target_anchor_relative_log"].isna().any():
        raise RuntimeError("Fare target contains missing values")
    fare.to_parquet(FARE_FRAME, index=False, compression="zstd")
    ledger.to_parquet(FARE_LEDGER, index=False, compression="zstd")

    cutoff_end = pd.Timestamp(DATA_CUTOFF) + pd.Timedelta(days=1)
    if exact["label_time"].max() >= cutoff_end:
        raise RuntimeError("Exact post-cutoff label entered frame")
    if fare["label_time"].max() >= cutoff_end:
        raise RuntimeError("Fare post-cutoff label entered frame")

    summary = {
        "status": "DEPLOYMENT_FRAMES_PASS",
        "contract_version": CONTRACT_VERSION,
        "input_tier": "STANDARD_ONLY",
        "nonstandard_rows_used": 0,
        "source_sha256": sha256(SOURCE),
        "staged_source_sha256": sha256(OFFERS_COPY),
        "offers": offers_summary,
        "exact_rows_all_policies": len(exact),
        "exact_rows_within_era_fit": len(exact_fit),
        "exact_drop5_rate_within_era": float(
            exact_fit[TARGET_NAME].mean()
        ),
        "exact_bridge_rows_excluded": int(len(exact) - len(exact_fit)),
        "fare_rows": len(fare),
        "fare_candidate_ledger_rows": len(ledger),
        "feature_contracts_unchanged": True,
        "c3_used_for_selection": False,
        "c3_labels_included_in_post_evaluation_refit": True,
        "frame_hashes": {
            "exact": sha256(EXACT_FRAME),
            "fare": sha256(FARE_FRAME),
            "fare_ledger": sha256(FARE_LEDGER),
        },
    }
    write_json_atomic(MANIFEST, summary)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return summary


if __name__ == "__main__":
    build()
