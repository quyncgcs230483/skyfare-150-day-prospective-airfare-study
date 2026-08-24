"""Inspection and explicit stage runner for the prospective study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skyfare.core.paths import DataLayout
from skyfare import workflow


def main() -> None:
    parser = argparse.ArgumentParser(prog="skyfare")
    parser.add_argument(
        "command",
        choices=(
            "paths",
            "contract",
            "prepare-development-data",
            "audit-features",
            "select-models",
            "run-test-one",
            "run-test-two",
            "pooled-evaluation",
            "production-refit",
            "build-figures",
        ),
    )
    args = parser.parse_args()
    if args.command == "paths":
        payload = DataLayout.resolve().serializable()
    elif args.command == "contract":
        path = DataLayout.resolve().root / "configs" / "study.json"
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    else:
        stages = {
            "prepare-development-data": workflow.prepare_development_data,
            "audit-features": workflow.audit_features,
            "select-models": workflow.select_models,
            "run-test-one": workflow.run_prospective_test_one,
            "run-test-two": workflow.run_prospective_test_two,
            "pooled-evaluation": workflow.run_pooled_evaluation,
            "production-refit": workflow.run_production_refit,
            "build-figures": workflow.build_result_figures,
        }
        stages[args.command]()
        return
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
