#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

if [[ ! -x ".venv/bin/reflex" ]]; then
  echo "Missing .venv. Run ./scripts/setup_application.sh from repository root." >&2
  exit 1
fi

PACKAGE_ROOT="$(cd "$APP_DIR/../.." && pwd)"
export SKYFARE_DATA_ROOT="${SKYFARE_DATA_ROOT:-$PACKAGE_ROOT/data}"
export SKYFARE_SERVING_ROOT="${SKYFARE_SERVING_ROOT:-$PACKAGE_ROOT/artifacts/serving}"
export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${TMPDIR:-/private/tmp}"

exec .venv/bin/reflex run \
  --env dev \
  --frontend-port "${SKYFARE_FRONTEND_PORT:-3000}" \
  --backend-port "${SKYFARE_BACKEND_PORT:-8000}"
