#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="$ROOT/application/skyfare_inference_demo"
PYTHON="${SKYFARE_PYTHON:-python3}"

"$PYTHON" -m venv "$APP/.venv"
"$APP/.venv/bin/python" -m pip install --upgrade pip
"$APP/.venv/bin/python" -m pip install -r "$APP/requirements.txt"

echo "[APPLICATION ENVIRONMENT PASS] $APP/.venv"
