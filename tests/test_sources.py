from __future__ import annotations

import pandas as pd
import pytest

from skyfare.core.sources import (
    CollectionSource,
    normalize_collection_era,
    normalize_source_columns,
    normalize_transition,
)


def test_fli_source_is_canonical() -> None:
    assert normalize_collection_era(CollectionSource.FLI.value) == CollectionSource.FLI.value


def test_trip_source_is_canonical() -> None:
    assert normalize_collection_era(CollectionSource.TRIP_COM.value) == CollectionSource.TRIP_COM.value


def test_transition_normalization_preserves_direction() -> None:
    transition = f"{CollectionSource.FLI.value}->{CollectionSource.TRIP_COM.value}"
    assert normalize_transition(transition) == transition


def test_source_frame_values_change_without_row_or_index_change() -> None:
    source = pd.DataFrame(
        {
            "collection_era": [CollectionSource.FLI.value, CollectionSource.TRIP_COM.value],
            "source_target_era_transition": [
                f"{CollectionSource.FLI.value}->{CollectionSource.FLI.value}",
                f"{CollectionSource.FLI.value}->{CollectionSource.TRIP_COM.value}",
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
