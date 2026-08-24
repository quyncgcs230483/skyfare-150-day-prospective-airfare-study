"""Repository-rooted entry points for complete study stages."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from skyfare.core.paths import DataLayout


def _run(module: str, *arguments: str, env: dict[str, str] | None = None) -> None:
    command = [sys.executable, "-m", module, *arguments]
    subprocess.run(command, check=True, env=env)


def prepare_development_data() -> None:
    env = {**os.environ, "SKYFARE_EXECUTION_ENV": "VAST"}
    _run("skyfare.preparation.standardize_sources", "--cutoff", "2026-07-28")
    _run("skyfare.preparation.build_training_frames", env=env)


def audit_features() -> None:
    _run("skyfare.features.run_model_free_feature_selection_gates")
    _run("skyfare.features.build_prefit_training_contract")
    _run("skyfare.features.validate_prefit_contract")


def select_models() -> None:
    layout = DataLayout.resolve()
    env = {
        **os.environ,
        "SKYFARE_EXECUTION_ENV": "VAST",
        "SKYFARE_MODEL_INPUT_ROOT": str(layout.processed / "development"),
        "SKYFARE_MODEL_OUTPUT_ROOT": str(layout.artifacts / "model_selection"),
    }
    _run("skyfare.models.preflight", "--root", str(layout.root), env=env)
    _run("skyfare.models.candidate_registry", env=env)
    deadline = str(time.time() + 48 * 3600)
    registry = layout.artifacts / "model_selection" / "registry"
    for name in ("smoke_v24.json", "screen_history_v24.json"):
        _run(
            "skyfare.models.scheduler",
            "--registry",
            str(registry / name),
            "--python",
            sys.executable,
            "--gpu-slots",
            "4",
            "--deadline-epoch",
            deadline,
            "--require-complete",
            env=env,
        )
    _run("skyfare.models.select_finalists", env=env)
    finalists = layout.artifacts / "model_selection" / "analysis" / "finalists_v24.json"
    _run("skyfare.models.candidate_registry", "--late-finalists", str(finalists), env=env)
    _run(
        "skyfare.models.scheduler",
        "--registry",
        str(registry / "late_v24.json"),
        "--python",
        sys.executable,
        "--gpu-slots",
        "4",
        "--deadline-epoch",
        deadline,
        "--require-complete",
        env=env,
    )
    _run("skyfare.models.analyse_candidates", env=env)
    _run("skyfare.policy.buy_wait", env=env)
    _run("skyfare.models.freeze_finalists", env=env)
    _run("skyfare.models.verify_selection", env=env)


def run_prospective_test_one() -> None:
    layout = DataLayout.resolve()
    development = layout.processed / "development"
    sealed = layout.processed / "prospective_test_one"
    output = layout.artifacts / "prospective_test_one"
    _run(
        "skyfare.evaluation.prospective_test_one.prepare",
        "--frozen-input-root",
        str(development),
        "--output-root",
        str(sealed),
        "--trip-raw-root",
        str(layout.raw_trip_com),
    )
    env = {
        **os.environ,
        "SKYFARE_TEST_ONE_DEVELOPMENT_INPUT_ROOT": str(development),
        "SKYFARE_TEST_ONE_SEALED_ROOT": str(sealed),
        "SKYFARE_TEST_ONE_OUTPUT_ROOT": str(output),
    }
    for module in (
        "skyfare.evaluation.prospective_test_one.scheduler",
        "skyfare.evaluation.prospective_test_one.freeze_predictions",
        "skyfare.evaluation.prospective_test_one.evaluate",
        "skyfare.evaluation.prospective_test_one.verify",
        "skyfare.evaluation.prospective_test_one.archive",
    ):
        _run(module, env=env)


def run_prospective_test_two() -> None:
    layout = DataLayout.resolve()
    development = layout.processed / "development"
    prior = layout.processed / "prospective_test_one"
    sealed = layout.processed / "prospective_test_two"
    output = layout.artifacts / "prospective_test_two"
    _run(
        "skyfare.evaluation.prospective_test_two.prepare",
        "--development-input-root",
        str(development),
        "--history-root",
        str(prior),
        "--output-root",
        str(sealed),
        "--trip-raw-root",
        str(layout.raw_trip_com),
    )
    env = {
        **os.environ,
        "SKYFARE_TEST_TWO_DEVELOPMENT_INPUT_ROOT": str(development),
        "SKYFARE_TEST_TWO_PRIOR_ROOT": str(prior),
        "SKYFARE_TEST_TWO_SEALED_ROOT": str(sealed),
        "SKYFARE_TEST_TWO_OUTPUT_ROOT": str(output),
    }
    for module in (
        "skyfare.evaluation.prospective_test_two.scheduler",
        "skyfare.evaluation.prospective_test_two.freeze_predictions",
        "skyfare.evaluation.prospective_test_two.evaluate",
        "skyfare.evaluation.prospective_test_two.verify",
        "skyfare.evaluation.prospective_test_two.archive",
    ):
        _run(module, env=env)


def run_pooled_evaluation() -> None:
    layout = DataLayout.resolve()
    contract = json.loads(
        (layout.controls / "pooled_evaluation_contract.json").read_text(
            encoding="utf-8"
        )
    )
    release_assets = layout.artifacts / "release_assets"
    output = layout.artifacts / "pooled_evaluation"
    first = release_assets / contract["expected_archives"]["test_1"]["filename"]
    second = release_assets / contract["expected_archives"]["test_2"]["filename"]
    for path in (first, second):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required prospective result archive is absent: {path}. "
                "Fetch release assets before pooled evaluation."
            )
    _run(
        "skyfare.evaluation.pooled.finalize",
        "--test1-results",
        str(first),
        "--test2-results",
        str(second),
        "--output-dir",
        str(output),
    )
    _run("skyfare.evaluation.pooled.verify", "--output-dir", str(output))
    _run("skyfare.evaluation.pooled.archive")


def run_production_refit() -> None:
    _run("skyfare.production.preflight")
    _run("skyfare.production.scheduler")


def build_result_figures() -> None:
    _run("skyfare.reporting.results_figures")
