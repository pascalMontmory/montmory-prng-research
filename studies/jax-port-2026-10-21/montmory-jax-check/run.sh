#!/usr/bin/env bash
set -euo pipefail
KIT_ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$KIT_ROOT"
MONTMORY_PYTHON="${MONTMORY_PYTHON:-python3}"
MONTMORY_VENV="${MONTMORY_VENV:-.venv}"
"$MONTMORY_PYTHON" -c 'import sys; assert sys.version_info >= (3,11), "Python 3.11 ou plus recent requis"'
"$MONTMORY_PYTHON" check_manifest.py
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
OUT="results/$RUN_ID"
mkdir -p "$OUT"
archive_result() {
    status=$?
    trap - EXIT
    printf '%s\n' "$status" > "$OUT/exit-status.txt"
    archive="$KIT_ROOT/montmory-jax-results-$RUN_ID.tar.gz"
    if tar -czf "$archive" "$OUT" manifest.json README.md; then
        printf '\nResultats et diagnostics : %s\n' "$archive"
    else
        printf '\nEchec de l archivage ; diagnostics conserves dans %s/%s\n' "$KIT_ROOT" "$OUT" >&2
        if [ "$status" -eq 0 ]; then status=1; fi
    fi
    exit "$status"
}
trap archive_result EXIT
main() {
    printf '1/3 Environnement Python local ; premiere execution : telechargement des dependances.\n'
    "$MONTMORY_PYTHON" -m venv "$MONTMORY_VENV"
    "$MONTMORY_VENV/bin/python" -m pip install --disable-pip-version-check -r requirements.txt
    "$MONTMORY_VENV/bin/python" -m pip freeze > "$OUT/packages.txt"
    printf '2/3 Reference C native, JAX CPU et Pallas en mode interprete.\n'
    JAX_PLATFORMS=cpu "$MONTMORY_VENV/bin/python" -u verify.py --out "$OUT" --compare reference/vps-report.json
    printf '3/3 PASS. Empreinte des sorties identique au VPS x86.\n'
    printf 'Aucun TPU natif teste ; aucune performance TPU mesuree ; aucun envoi automatique.\n'
}
main 2>&1 | tee "$OUT/run.log"
