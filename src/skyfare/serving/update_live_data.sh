#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${SKYFARE_PYTHON:-${ROOT}/.venv/bin/python}"
DAY="${1:?Usage: update_live_data.sh YYYY-MM-DD [FULL_DAY|AM_ONLY]}"
MODE="${2:-FULL_DAY}"
DAILY="${ROOT}/data/raw/trip_com/${DAY}.csv"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -s "${DAILY}" ]]; then
  echo "Missing daily file: ${DAILY}" >&2
  exit 1
fi

if [[ "${MODE}" == "FULL_DAY" ]]; then
  "${PYTHON}" -m skyfare.preparation.merge_daily_offers --through-date "${DAY}" --check-only
  INGEST_ARGS=()
elif [[ "${MODE}" == "AM_ONLY" ]]; then
  INGEST_ARGS=(--allow-am-only)
else
  echo "Mode must be FULL_DAY or AM_ONLY" >&2
  exit 1
fi

"${PYTHON}" \
  -m skyfare.serving.manage_live_store \
  ingest-day "${DAILY}" "${INGEST_ARGS[@]}"

"${PYTHON}" \
  -m skyfare.serving.manage_live_store status

echo "[SKYFARE POSTGRES UPDATE PASS] ${DAY} ${MODE}"
