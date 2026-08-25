#!/usr/bin/env python3
"""Regenerate report figures and copy selected outputs into Reflex assets."""

from __future__ import annotations

import shutil
from pathlib import Path

from skyfare.reporting.results_figures import main as build_figures

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "reports" / "figures"
ASSETS = ROOT / "application" / "skyfare_inference_demo" / "assets"
MAPPINGS = {
    "development_128_day_collection_coverage.png": (
        "data_analysis/collection_coverage.png"
    ),
    "development_128_day_source_agreement.png": (
        "data_analysis/source_agreement.png"
    ),
    "development_128_day_route_airline_coverage.png": (
        "data_analysis/route_airline_coverage.png"
    ),
    "development_128_day_airline_density.png": (
        "data_analysis/airline_density.png"
    ),
    "development_128_day_booking_window_fares.png": (
        "data_analysis/booking_window_fares.png"
    ),
    "airline_composition_temporal_drift.png": (
        "data_analysis/airline_composition_temporal_drift.png"
    ),
    "classification_prospective_diagnostics.png": (
        "evaluation/classification_prospective_diagnostics.png"
    ),
    "regression_prospective_diagnostics.png": (
        "evaluation/regression_prospective_diagnostics.png"
    ),
    "distribution_ranking_prospective_diagnostics.png": (
        "evaluation/distribution_ranking_prospective_diagnostics.png"
    ),
    "prospective_evaluation_decision_utility.png": (
        "evaluation/prospective_evaluation_decision_utility.png"
    ),
}


def main() -> None:
    build_figures()
    for source_name, target_name in MAPPINGS.items():
        source = REPORTS / source_name
        target = ASSETS / target_name
        if not source.is_file():
            raise FileNotFoundError(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    print(f"[APPLICATION FIGURE SYNC PASS] files={len(MAPPINGS)}")


if __name__ == "__main__":
    main()
