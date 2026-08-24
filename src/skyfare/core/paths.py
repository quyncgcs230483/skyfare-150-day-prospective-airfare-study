"""Repository-relative data and artifact paths with explicit overrides."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path


def repository_root() -> Path:
    override = os.environ.get("SKYFARE_PROJECT_ROOT")
    return Path(override).expanduser().resolve() if override else Path(__file__).resolve().parents[3]


def _resolved(env_name: str, default: Path) -> Path:
    value = os.environ.get(env_name)
    return Path(value).expanduser().resolve() if value else default.resolve()


@dataclass(frozen=True)
class DataLayout:
    root: Path
    raw_fli: Path
    raw_trip_com: Path
    collection_logs: Path
    standardised: Path
    processed: Path
    controls: Path
    artifacts: Path

    @classmethod
    def resolve(cls) -> "DataLayout":
        root = repository_root()
        data = _resolved("SKYFARE_DATA_ROOT", root / "data")
        artifacts = _resolved("SKYFARE_ARTIFACT_ROOT", root / "artifacts")
        return cls(
            root=root,
            raw_fli=_resolved("SKYFARE_FLI_OUTPUT_DIR", data / "raw" / "fli"),
            raw_trip_com=_resolved("SKYFARE_TRIP_OUTPUT_DIR", data / "raw" / "trip_com"),
            collection_logs=_resolved(
                "SKYFARE_COLLECTION_LOG_DIR", artifacts / "logs" / "collection"
            ),
            standardised=_resolved(
                "SKYFARE_STANDARDISED_ROOT", data / "interim" / "standardised"
            ),
            processed=_resolved("SKYFARE_PROCESSED_ROOT", data / "processed"),
            controls=_resolved("SKYFARE_CONTROL_ROOT", root / "configs"),
            artifacts=artifacts,
        )

    def create_runtime_directories(self) -> None:
        for path in (
            self.raw_fli,
            self.raw_trip_com,
            self.collection_logs,
            self.standardised,
            self.processed,
            self.artifacts,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def serializable(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}
