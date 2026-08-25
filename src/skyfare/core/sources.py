"""Canonical collection-source taxonomy."""

from __future__ import annotations

from enum import Enum

import pandas as pd


class CollectionSource(str, Enum):
    FLI = "FLI_LIBRARY_ERA"
    TRIP_COM = "TRIP_COM_BROWSER_ERA"


CANONICAL_ERAS = frozenset(source.value for source in CollectionSource)


def normalize_collection_era(value: str) -> str:
    normalized = str(value)
    if normalized not in CANONICAL_ERAS:
        raise ValueError(f"Unknown collection era: {value!r}")
    return normalized


def normalize_transition(value: str) -> str:
    parts = str(value).split("->")
    if len(parts) != 2:
        raise ValueError(f"Invalid collection transition: {value!r}")
    return "->".join(normalize_collection_era(part) for part in parts)


def normalize_source_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "collection_era" in result:
        result["collection_era"] = result["collection_era"].astype("string").map(
            normalize_collection_era
        )
    if "source_target_era_transition" in result:
        result["source_target_era_transition"] = (
            result["source_target_era_transition"].astype("string").map(normalize_transition)
        )
    return result
