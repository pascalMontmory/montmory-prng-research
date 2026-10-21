#!/usr/bin/env bash
set -euo pipefail
KIT_ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$KIT_ROOT"
MONTMORY_PYTHON="${MONTMORY_PYTHON:-python3}"
MONTMORY_VENV="${MONTMORY_VENV:-.venv}"
"$MONTMORY_PYTHON" -c 'import sys; assert sys.version_info >= (3,11), "Python 3.11 ou plus recent requis"'
"$MONTMORY_PYTHON" check_manifest.py
"$MONTMORY_PYTHON" -m venv "$MONTMORY_VENV"
"$MONTMORY_VENV/bin/python" -m pip install --disable-pip-version-check -r requirements.txt
exec "$MONTMORY_VENV/bin/python" -u run_cpu.py "$@"
