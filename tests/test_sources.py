from __future__ import annotations

import pandas as pd
import pytest

from skyfare.core.sources import (
    CollectionSource,
    normalize_collection_era,
    normalize_source_columns,
    normalize_transition,
)


def test_legacy_fli_alias_is_normalized() -> None:
    assert normalize_collection_era("SERPAPI_ERA") == CollectionSource.FLI.value


def test_legacy_trip_alias_is_normalized() -> None:
    assert normalize_collection_era("TRIP_DAILY_ERA") == CollectionSource.TRIP_COM.value


def test_transition_normalization_preserves_direction() -> None:
    assert normalize_transition("SERPAPI_ERA->TRIP_DAILY_ERA") == "FLI_LIBRARY_ERA->TRIP_COM_BROWSER_ERA"


def test_source_frame_values_change_without_row_or_index_change() -> None:
    source = pd.DataFrame(
        {
            "collection_era": ["SERPAPI_ERA", "TRIP_DAILY_ERA"],
            "source_target_era_transition": [
                "SERPAPI_ERA->SERPAPI_ERA",
                "SERPAPI_ERA->TRIP_DAILY_ERA",
            ],
            "value": [1, 2],
        },
        index=[4, 9],
    )
    result = normalize_source_columns(source)
    assert list(result.index) == [4, 9]
    assert result["value"].tolist() == [1, 2]
    assert result["collection_era"].tolist() == ["FLI_LIBRARY_ERA", "TRIP_COM_BROWSER_ERA"]


def test_unknown_source_fails_closed() -> None:
    with pytest.raises(ValueError):
        normalize_collection_era("UNKNOWN")
