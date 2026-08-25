#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${SKYFARE_PYTHON:-python3}"
DATE="${SKYFARE_SERVING_DATE:-2026-08-24}"
ASSETS="$ROOT/artifacts/release_assets"
MODEL_ARCHIVE="$ASSETS/SKYFARE_128_DAY_FINAL_PRODUCTION_MODELS_R1.tar.gz"
MODEL_OUTPUT="$ROOT/artifacts/models/final_production"
BUILD="$ROOT/artifacts/serving/inference_build"
SCORES="$ROOT/artifacts/serving/inference_scores"
ASSEMBLED="$ROOT/artifacts/serving/assembled_predictions.parquet"

export SKYFARE_PROJECT_ROOT="$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -s "$MODEL_ARCHIVE" ]]; then
  "$PYTHON" "$ROOT/tools/fetch_release_assets.py" \
    --destination "$ASSETS" \
    --filename "$(basename "$MODEL_ARCHIVE")"
fi

"$PYTHON" -m skyfare.preparation.standardize_sources --cutoff "$DATE"
"$PYTHON" -m skyfare.serving.prepare_inference_frames \
  --through-date "$DATE" \
  --output-root "$BUILD"
"$PYTHON" "$ROOT/tools/extract_production_models.py" \
  "$MODEL_ARCHIVE" \
  --output-root "$MODEL_OUTPUT" \
  --release-config "$ROOT/configs/release_assets.json"

ARTIFACT_ROOT="$MODEL_OUTPUT/skyfare_128_day_final_production_refit_r1"
MANIFEST="$ARTIFACT_ROOT/production_refit_outputs/deployment/DEPLOYMENT_MANIFEST_R1.json"

"$PYTHON" -m skyfare.production.inference \
  --classification-frame "$BUILD/classification_features.parquet" \
  --regression-frame "$BUILD/regression_features.parquet" \
  --sequence-source "$BUILD/standard_offers.parquet" \
  --artifact-root "$ARTIFACT_ROOT" \
  --manifest "$MANIFEST" \
  --output-root "$SCORES"
"$PYTHON" -m skyfare.serving.assemble_predictions \
  --feature-root "$BUILD" \
  --prediction-root "$SCORES" \
  --output "$ASSEMBLED"
"$PYTHON" -m skyfare.serving.build_live_snapshot \
  --predictions "$ASSEMBLED" \
  --prediction-provenance "$BUILD/prediction_provenance.json" \
  --model-manifest "$MANIFEST" \
  --through-date "$DATE" \
  --output-root "$ROOT/artifacts/serving"

echo "[SERVING SNAPSHOT PASS] booking_date=$DATE training_cutoff=2026-08-19"
