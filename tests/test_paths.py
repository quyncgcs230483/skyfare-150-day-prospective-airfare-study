from __future__ import annotations

from skyfare.core.paths import DataLayout


def test_default_layout_is_repository_relative(monkeypatch) -> None:
    for key in (
        "SKYFARE_PROJECT_ROOT",
        "SKYFARE_DATA_ROOT",
        "SKYFARE_ARTIFACT_ROOT",
        "SKYFARE_FLI_OUTPUT_DIR",
        "SKYFARE_TRIP_OUTPUT_DIR",
    ):
        monkeypatch.delenv(key, raising=False)
    layout = DataLayout.resolve()
    assert layout.raw_fli == layout.root / "data/raw/fli"
    assert layout.raw_trip_com == layout.root / "data/raw/trip_com"
    assert layout.standardised == layout.root / "data/interim/standardised"
    assert layout.processed == layout.root / "data/processed"


def test_layout_honours_explicit_root(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SKYFARE_PROJECT_ROOT", str(tmp_path))
    layout = DataLayout.resolve()
    assert layout.root == tmp_path.resolve()
    assert layout.artifacts == tmp_path.resolve() / "artifacts"
