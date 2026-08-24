"""Canonical collection-session and UI daypart semantics.

Collection sessions follow the data-collection contract:

* AM: 06:00:00 through 17:59:59 local time.
* PM: 18:00:00 through 05:59:59 on the following calendar day.

UI dayparts are presentation-only and use the familiar 00:00/12:00 split.
They must never replace collection labels in model features or target pairing.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd


LOCAL_TIMEZONE = "Asia/Ho_Chi_Minh"


def _local_timestamp(value: datetime | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(LOCAL_TIMEZONE)
    return timestamp


def _local_values(values: pd.Series) -> pd.Series:
    if values.dt.tz is not None:
        return values.dt.tz_convert(LOCAL_TIMEZONE)
    return values


def collection_session_label(value: datetime | pd.Timestamp) -> str:
    """Return canonical AM/PM label for a collection timestamp."""
    hour = _local_timestamp(value).hour
    return "AM" if 6 <= hour < 18 else "PM"


def operational_session_date(value: datetime | pd.Timestamp) -> date:
    """Return the owning date; 00:00-05:59 belongs to the previous PM."""
    timestamp = _local_timestamp(value)
    if timestamp.hour < 6:
        timestamp -= timedelta(days=1)
    return timestamp.date()


def ui_daypart_label(value: datetime | pd.Timestamp) -> str:
    """Return presentation daypart: AM before 12:00, otherwise PM."""
    return "AM" if _local_timestamp(value).hour < 12 else "PM"


def collection_session_labels(values: pd.Series) -> pd.Series:
    """Return vectorized collection labels for parsed datetime values."""
    local = _local_values(values)
    hours = local.dt.hour
    labels = pd.Series(
        pd.array(((hours >= 6) & (hours < 18)).map({True: "AM", False: "PM"})),
        index=values.index,
        dtype="object",
    )
    return labels.mask(local.isna())


def operational_session_dates(values: pd.Series) -> pd.Series:
    """Return vectorized operational dates for parsed datetime values."""
    local = _local_values(values)
    adjusted = local - pd.to_timedelta(local.dt.hour.lt(6).astype(int), unit="D")
    return adjusted.dt.date
