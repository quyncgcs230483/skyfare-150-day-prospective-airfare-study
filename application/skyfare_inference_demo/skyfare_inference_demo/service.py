"""Portable data and inference adapter for the SkyFare demonstration UI."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parents[1]
DATA_ROOT = Path(os.environ.get("SKYFARE_DATA_ROOT", REPO_ROOT / "data")).resolve()
SERVING_ROOT = Path(
    os.environ.get("SKYFARE_SERVING_ROOT", REPO_ROOT / "artifacts" / "serving")
).resolve()
RAW_DIRECTORIES = (
    DATA_ROOT / "raw" / "fli",
    DATA_ROOT / "raw" / "google_flights_manual_9g",
    DATA_ROOT / "raw" / "trip_com",
)
CANONICAL_DUDS = frozenset({1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60})
ORDERED_DUDS = tuple(sorted(CANONICAL_DUDS))
BUY_WAIT_THRESHOLD = 0.30
TRAINING_CUTOFF = "2026-08-19"

AIRLINE_NAMES = {
    "VN": "Vietnam Airlines",
    "VJ": "VietJet Air",
    "QH": "Bamboo Airways",
    "VU": "Vietravel Airlines",
    "9G": "Sun Phu Quoc Airways",
}


class ServingError(RuntimeError):
    """Request violates serving contract or no matching offer exists."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _session_label(values: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="coerce")
    return pd.Series(
        ((timestamps.dt.hour >= 6) & (timestamps.dt.hour < 18)).map(
            {True: "AM", False: "PM"}
        ),
        index=values.index,
        dtype="string",
    )


class SkyFareService:
    """Load raw observations plus optional verified final-refit predictions."""

    def __init__(self) -> None:
        self.ready = False
        self.load_seconds = 0.0
        self._dates: list[str] = []
        self._date_files: dict[str, list[Path]] = {}
        self._catalog_frame = pd.DataFrame()
        self._snapshot = pd.DataFrame()
        self._snapshot_manifest: dict[str, Any] = {}

    @property
    def model_version(self) -> str:
        return str(self._snapshot_manifest.get("model_version", "OBSERVED_FARE_ONLY"))

    @property
    def cutoff(self) -> str:
        return self._dates[-1] if self._dates else ""

    def _discover_raw_files(self) -> dict[str, list[Path]]:
        files: dict[str, list[Path]] = {}
        for directory in RAW_DIRECTORIES:
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.csv")):
                try:
                    date = pd.Timestamp(path.stem).strftime("%Y-%m-%d")
                except ValueError:
                    continue
                files.setdefault(date, []).append(path)
        if not files:
            raise ServingError(f"No raw daily CSV files found below {DATA_ROOT}")
        return files

    @staticmethod
    def _read_files(paths: list[Path]) -> pd.DataFrame:
        columns = [
            "scraped_at",
            "session_id",
            "route",
            "days_until_departure",
            "flight_date",
            "airline",
            "airline_name",
            "flight_no",
            "departure_time",
            "price_vnd",
            "data_source",
        ]
        frames: list[pd.DataFrame] = []
        for path in paths:
            header = pd.read_csv(path, nrows=0).columns
            selected = [column for column in columns if column in header]
            frame = pd.read_csv(path, usecols=selected, low_memory=False)
            try:
                source_file = path.relative_to(REPO_ROOT).as_posix()
            except ValueError:
                source_file = path.as_posix()
            frame["source_file"] = source_file
            frames.append(frame)
        data = pd.concat(frames, ignore_index=True)
        for column in ("scraped_at", "session_id", "flight_date", "departure_time"):
            if column in data:
                data[column] = pd.to_datetime(data[column], errors="coerce")
        data["price_vnd"] = pd.to_numeric(data["price_vnd"], errors="coerce")
        data["days_until_departure"] = pd.to_numeric(
            data["days_until_departure"], errors="coerce"
        )
        session_source = data.get("session_id", data["scraped_at"])
        data["session_label"] = _session_label(session_source)
        data = data[
            data["route"].notna()
            & data["airline"].notna()
            & data["flight_date"].notna()
            & data["departure_time"].notna()
            & data["price_vnd"].gt(0)
            & data["days_until_departure"].between(1, 60)
        ].copy()
        data["departure_HHMM"] = (
            data["departure_time"].dt.hour * 100 + data["departure_time"].dt.minute
        ).astype("int16")
        data["schedule_slot_id"] = pd.util.hash_pandas_object(
            data[["route", "airline", "flight_date", "departure_HHMM"]], index=False
        ).astype("uint64").astype(str)
        return data

    def _load_verified_snapshot(self) -> None:
        current_path = SERVING_ROOT / "current.json"
        if not current_path.is_file():
            return
        current = json.loads(current_path.read_text(encoding="utf-8"))
        manifest_path = SERVING_ROOT / str(current.get("manifest", ""))
        prediction_path = SERVING_ROOT / str(current.get("predictions", ""))
        if not manifest_path.is_file() or not prediction_path.is_file():
            raise ServingError("Serving pointer references missing files")
        if _sha256(manifest_path) != str(current.get("manifest_sha256", "")):
            raise ServingError("Serving manifest hash mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "PASS":
            raise ServingError("Serving snapshot has not passed verification")
        if manifest.get("training_cutoff_inclusive") != TRAINING_CUTOFF:
            raise ServingError("Serving snapshot changed frozen training cutoff")
        if manifest.get("post_cutoff_labels_used") is not False:
            raise ServingError("Serving snapshot may contain post-cutoff labels")
        snapshot = pd.read_parquet(prediction_path)
        required = {
            "session_date",
            "session_label",
            "route",
            "flight_date",
            "airline",
            "departure_HHMM",
            "schedule_slot_id",
            "predicted_price_vnd",
            "baseline_price_vnd",
        }
        missing = sorted(required.difference(snapshot.columns))
        if missing:
            raise ServingError(f"Serving prediction columns missing: {missing}")
        for column in ("session_date", "flight_date"):
            snapshot[column] = pd.to_datetime(snapshot[column], errors="coerce")
        self._snapshot = snapshot
        self._snapshot_manifest = manifest

    def initialize(self) -> dict[str, Any]:
        if self.ready:
            return self.catalog()
        started = time.perf_counter()
        self._date_files = self._discover_raw_files()
        self._dates = sorted(self._date_files)
        latest = [self._read_files(self._date_files[date]) for date in self._dates[-7:]]
        self._catalog_frame = pd.concat(latest, ignore_index=True)
        self._load_verified_snapshot()
        self.load_seconds = time.perf_counter() - started
        self.ready = True
        return self.catalog()

    def catalog(self) -> dict[str, Any]:
        pairs = (
            self._catalog_frame[["route", "airline"]]
            .dropna()
            .astype(str)
            .drop_duplicates()
        )
        routes = sorted(pairs["route"].unique().tolist())
        airlines = sorted(pairs["airline"].unique().tolist())
        route_airlines = {
            route: sorted(group["airline"].unique().tolist())
            for route, group in pairs.groupby("route", sort=True)
        }
        return {
            "dates": self._dates,
            "routes": routes,
            "airlines": airlines,
            "route_airlines": route_airlines,
            "cutoff": self.cutoff,
            "model_version": self.model_version,
            "load_seconds": self.load_seconds,
            "data_mode": "verified_snapshot" if not self._snapshot.empty else "raw_observation",
            "snapshot_id": str(self._snapshot_manifest.get("snapshot_id", "")),
        }

    def _observed_query(
        self,
        session_date: str,
        session_label: str,
        route: str,
        flight_date: str,
        airline: str,
    ) -> pd.DataFrame:
        if session_date not in self._date_files:
            raise ServingError(f"No observation file for booking date {session_date}")
        data = self._read_files(self._date_files[session_date])
        selected = data[
            data["route"].astype(str).eq(route)
            & data["flight_date"].dt.normalize().eq(pd.Timestamp(flight_date))
            & data["session_label"].astype(str).eq(session_label)
        ].copy()
        if airline:
            selected = selected[selected["airline"].astype(str).eq(airline)].copy()
        if selected.empty:
            raise ServingError("No observed fare matches selected query")
        selected = selected.sort_values("scraped_at", kind="stable").drop_duplicates(
            ["airline", "departure_HHMM"], keep="last"
        )
        selected["predicted_price_vnd"] = selected["price_vnd"]
        selected["baseline_price_vnd"] = selected["price_vnd"]
        selected["action"] = "FARE ONLY"
        selected["drop_probability"] = pd.NA
        return selected

    def _snapshot_query(
        self,
        session_date: str,
        session_label: str,
        route: str,
        flight_date: str,
        airline: str,
    ) -> pd.DataFrame:
        session = pd.Timestamp(session_date).normalize()
        requested_flight = pd.Timestamp(flight_date).normalize()
        selected = self._snapshot[
            self._snapshot["session_date"].dt.normalize().eq(pd.Timestamp(session_date))
            & self._snapshot["session_label"].astype(str).eq(session_label)
            & self._snapshot["route"].astype(str).eq(route)
            & self._snapshot["flight_date"].dt.normalize().eq(requested_flight)
        ].copy()
        if airline:
            selected = selected[selected["airline"].astype(str).eq(airline)].copy()
        if not selected.empty:
            return selected

        dud = int((requested_flight - session).days)
        lower = max((value for value in ORDERED_DUDS if value < dud), default=None)
        upper = min((value for value in ORDERED_DUDS if value > dud), default=None)
        if lower is None or upper is None:
            raise ServingError("No canonical interval brackets requested flight date")
        base = self._snapshot[
            self._snapshot["session_date"].dt.normalize().eq(session)
            & self._snapshot["session_label"].astype(str).eq(session_label)
            & self._snapshot["route"].astype(str).eq(route)
        ].copy()
        if airline:
            base = base[base["airline"].astype(str).eq(airline)].copy()
        left = base[
            base["flight_date"].dt.normalize().eq(session + pd.Timedelta(days=lower))
        ].copy()
        right = base[
            base["flight_date"].dt.normalize().eq(session + pd.Timedelta(days=upper))
        ].copy()
        identity = ["airline", "departure_HHMM"]
        left = left.sort_values("ranking_score", ascending=False).drop_duplicates(identity)
        right = right.sort_values("ranking_score", ascending=False).drop_duplicates(identity)
        if left.empty or right.empty:
            raise ServingError("Canonical interpolation endpoints are incomplete")
        joined = left.merge(
            right,
            on=identity,
            how="inner",
            validate="one_to_one",
            suffixes=("_lower", "_upper"),
        )
        if joined.empty:
            raise ServingError("No schedule templates span interpolation endpoints")
        weight = (dud - lower) / (upper - lower)
        numeric = [
            "predicted_price_vnd",
            "baseline_price_vnd",
            "observed_price_vnd",
            "drop_probability",
            "ranking_score",
            *sorted(
                column
                for column in self._snapshot
                if column.startswith("predicted_price_q")
            ),
        ]
        result = joined[identity].copy()
        for column in numeric:
            lower_column = f"{column}_lower"
            upper_column = f"{column}_upper"
            if lower_column in joined and upper_column in joined:
                result[column] = (
                    pd.to_numeric(joined[lower_column], errors="coerce") * (1.0 - weight)
                    + pd.to_numeric(joined[upper_column], errors="coerce") * weight
                )
        result["session_date"] = session
        result["session_label"] = session_label
        result["route"] = route
        result["flight_date"] = requested_flight
        result["query_dud"] = dud
        result["dud_support_mode"] = "INTERIOR_OFF_GRID_INTERPOLATED"
        result["interpolation_interval"] = f"DUD {lower}-{upper}"
        result["interpolation_note"] = (
            "Linear interpolation between adjacent frozen canonical booking windows"
        )
        result["action"] = result["drop_probability"].ge(
            BUY_WAIT_THRESHOLD
        ).map({True: "WAIT", False: "BUY"})
        result["operational_action"] = "BUY"
        result["schedule_slot_id"] = pd.util.hash_pandas_object(
            result[["route", "airline", "flight_date", "departure_HHMM"]],
            index=False,
        ).astype("uint64").astype(str)
        return result

    def search(
        self,
        session_date: str,
        session_label: str,
        route: str,
        flight_date: str,
        airline: str = "",
    ) -> dict[str, Any]:
        if not self.ready:
            self.initialize()
        started = time.perf_counter()
        dud = (pd.Timestamp(flight_date) - pd.Timestamp(session_date)).days
        if dud < 1 or dud > 60:
            raise ServingError("Flight date must be 1-60 days after booking date")
        if self._snapshot.empty:
            frame = self._observed_query(session_date, session_label, route, flight_date, airline)
            scored = False
        else:
            frame = self._snapshot_query(session_date, session_label, route, flight_date, airline)
            scored = True
        frame = frame.sort_values(
            ["predicted_price_vnd", "departure_HHMM"], kind="stable"
        ).reset_index(drop=True)
        rows: list[dict[str, Any]] = []
        for index, row in frame.iterrows():
            predicted = float(row["predicted_price_vnd"])
            baseline = float(row["baseline_price_vnd"])
            hhmm = int(row["departure_HHMM"])
            probability = row.get("drop_probability", pd.NA)
            has_probability = pd.notna(probability)
            airline_code = str(row["airline"])
            action = str(row.get("action", "FARE ONLY"))
            rows.append(
                {
                    "rank": index + 1,
                    "is_cheapest": index == 0,
                    "airline": airline_code,
                    "airline_name": AIRLINE_NAMES.get(airline_code, airline_code),
                    "logo_path": f"/airlines/{airline_code}.png",
                    "logo_alt": f"{AIRLINE_NAMES.get(airline_code, airline_code)} logo",
                    "departure_time": f"{hhmm // 100:02d}:{hhmm % 100:02d}",
                    "price": f"{predicted:,.0f}",
                    "baseline": f"{baseline:,.0f}",
                    "baseline_display": f"{baseline:,.0f} VND",
                    "anchor_delta": f"{(predicted / baseline - 1.0) * 100:+.1f}%",
                    "anchor_down": predicted < baseline,
                    "action": action,
                    "has_action": action in {"BUY", "WAIT"},
                    "action_note": "Guarded BUY/WAIT pilot" if scored else "Observed fare",
                    "is_wait": action == "WAIT",
                    "drop_probability": f"{float(probability) * 100:.1f}%" if has_probability else "-",
                    "fare_models": "Verified final-refit ensemble" if scored else "Not scored",
                    "exact_models": "Verified calibrated classifier" if scored else "Not scored",
                    "regime_label": "Verified snapshot" if scored else "Observed offer",
                    "regime_help": "Prediction generated under frozen model contract." if scored else "Raw fare observed at selected booking session.",
                    "history_label": "Point-in-time",
                    "history_help": "Only information available by selected booking timestamp is eligible.",
                    "anchor_label": "Prior anchor" if scored else "Observed fare",
                    "anchor_help": "Strictly-prior fare anchor." if scored else "No model anchor used in observation mode.",
                    "support_label": "Direct model prediction" if scored else "Direct observation",
                    "support_help": "Frozen production model output." if scored else "No prediction or policy claim is made.",
                    "interpolation_interval": str(row.get("interpolation_interval", "")),
                    "interpolation_note": str(row.get("interpolation_note", "")),
                    "slot_id": str(row["schedule_slot_id"]),
                }
            )
        return {
            "rows": rows,
            "count": len(rows),
            "dud": dud,
            "on_grid": dud in CANONICAL_DUDS,
            "query_seconds": time.perf_counter() - started,
            "cheapest_price": rows[0]["price"],
            "cheapest_airline": rows[0]["airline_name"],
            "cheapest_logo": rows[0]["logo_path"],
            "cheapest_logo_alt": rows[0]["logo_alt"],
            "cheapest_time": rows[0]["departure_time"],
        }


service = SkyFareService()
