#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_ROOT="${SKYFARE_BACKUP_ROOT:-${ROOT}/artifacts/serving/backups}"
OUT="${BACKUP_ROOT}/${STAMP}"
DATABASE_URL="${SKYFARE_DATABASE_URL:-postgresql://postgres@127.0.0.1:5432/fyp_flights}"
CURRENT="${ROOT}/artifacts/serving/current.json"

mkdir -p "${OUT}"

pg_dump \
  --dbname="${DATABASE_URL}" \
  --schema=skyfare_live \
  --format=custom \
  --file="${OUT}/skyfare_live.dump"

if [[ -s "${CURRENT}" ]]; then
  cp "${CURRENT}" "${OUT}/current.json"
  MANIFEST_REL="$(
    python3 -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["manifest"])' \
      "${CURRENT}"
  )"
  PREDICTIONS_REL="$(
    python3 -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["predictions"])' \
      "${CURRENT}"
  )"
  cp "${ROOT}/artifacts/serving/${MANIFEST_REL}" "${OUT}/snapshot_manifest.json"
  cp "${ROOT}/artifacts/serving/${PREDICTIONS_REL}" "${OUT}/predictions.parquet"
fi

(
  cd "${OUT}"
  shasum -a 256 ./* > SHA256SUMS
  shasum -a 256 -c SHA256SUMS
)

echo "[SKYFARE LIVE BACKUP PASS] ${OUT}"
