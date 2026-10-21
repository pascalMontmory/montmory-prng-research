#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ "$(uname -s)" != Darwin ]; then
  echo "Ce test doit etre execute sur le Mac Apple Silicon."
  exit 2
fi
if pgrep -f '[/]testu01_metal([[:space:]]|$)' >/dev/null 2>&1; then
  echo "TestU01 Metal tourne encore. Relance ce benchmark apres sa fin pour mesurer les performances sans concurrence."
  exit 2
fi
lanes_trials="${1:-9}"
case "$lanes_trials" in ''|*[!0-9]*) echo "Argument attendu : nombre de repetitions (3 a 31)."; exit 2 ;; esac
if [ "$lanes_trials" -lt 3 ] || [ "$lanes_trials" -gt 31 ]; then
  echo "Nombre de repetitions attendu : 3 a 31."
  exit 2
fi
lanes_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
lanes_results="results/$lanes_id"
lanes_archive="montmory-lanes-results-$lanes_id.tar.gz"
mkdir -p build "$lanes_results"
lanes_finish() {
  lanes_status=$?
  trap - EXIT
  set +e
  echo "$lanes_status" > "$lanes_results/exit-code.txt"
  tar -czf "$lanes_archive" "$lanes_results"
  lanes_tar_status=$?
  if [ "$lanes_tar_status" -eq 0 ]; then
    echo "Resultats et diagnostics : $PWD/$lanes_archive"
  else
    echo "Archivage impossible ; fichiers conserves dans $PWD/$lanes_results"
    if [ "$lanes_status" -eq 0 ]; then lanes_status=$lanes_tar_status; fi
  fi
  exit "$lanes_status"
}
trap lanes_finish EXIT
shasum -a 256 -c SHA256SUMS > "$lanes_results/integrity.txt"
cp SHA256SUMS "$lanes_results/kit-SHA256SUMS.txt"
cp protocol.json "$lanes_results/protocol.json"
cp generated/provenance.json "$lanes_results/v3-source-provenance.json"
unset SDKROOT CPPFLAGS LDFLAGS LIBS CFLAGS CXXFLAGS CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH
unset COMPILER_PATH GCC_EXEC_PREFIX MACOSX_DEPLOYMENT_TARGET CONFIG_SITE
lanes_sdk="$(xcrun --sdk macosx --show-sdk-path)"
export SDKROOT="$lanes_sdk"
sw_vers > "$lanes_results/macos.txt"
xcrun --sdk macosx clang++ --version > "$lanes_results/compiler.txt"
sysctl -n machdep.cpu.brand_string > "$lanes_results/cpu.txt"
echo "1/3 Verification des 1600 vecteurs ARM64/x86."
xcrun --sdk macosx clang -arch arm64 -isysroot "$lanes_sdk" -std=c11 -O2 -Wall -Wextra -Werror -Isrc \
  src/conformance_dump.c -o build/conformance_dump 2>&1 | tee "$lanes_results/core-build.txt"
build/conformance_dump > "$lanes_results/golden-vectors.txt"
cmp reference/golden-vectors.txt "$lanes_results/golden-vectors.txt"
echo "Vecteurs ARM64/x86 : PASS."
echo "2/3 Compilation du benchmark de lanes."
xcrun --sdk macosx clang++ -arch arm64 -isysroot "$lanes_sdk" -std=c++17 -O3 -fobjc-arc -Wall -Wextra -Werror \
  -Isrc -Ivendor/include src/lanes.mm -framework Foundation -framework Metal -o build/lanes \
  2>&1 | tee "$lanes_results/benchmark-build.txt"
echo "3/3 Conformite native puis mesures : flux fixe et geometries alternatives."
echo "Chaque execution ecrit 8388608 mots de 64 bits. Tous sont compares a la reference CPU hors chronometrage."
build/lanes generated/all.metal "$lanes_results" "$lanes_trials" 2>&1 | tee "$lanes_results/console.txt"
echo "PASS. Aucune donnee envoyee automatiquement. Les anciens kits et tests sont inchanges."
