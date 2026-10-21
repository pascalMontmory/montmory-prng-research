#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ "$(uname -s)" != Darwin ]; then
  echo "Ce script doit tourner sur le Mac Apple Silicon."
  exit 2
fi
opt_mode="${1:-all}"
case "$opt_mode" in all|cpu|metal) ;; *) echo "Usage : bash run.sh [all|cpu|metal]"; exit 2 ;; esac
if pgrep -f '[/](testu01_metal|lanes|cpu_benchmark|metal_opt_benchmark)([[:space:]]|$)' >/dev/null 2>&1; then
  echo "Un test PRNG connu tourne encore. Attends sa fin avant ces mesures."
  exit 2
fi
opt_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
opt_results="results/$opt_id"
opt_archive="montmory-metal-opt-results-$opt_id.tar.gz"
mkdir -p build "$opt_results"
opt_finish() {
  opt_status=$?
  trap - EXIT
  set +e
  echo "$opt_status" > "$opt_results/exit-code.txt"
  tar -czf "$opt_archive" "$opt_results"
  opt_tar_status=$?
  if [ "$opt_tar_status" -eq 0 ]; then
    echo "Resultats et diagnostics : $PWD/$opt_archive"
  else
    echo "Archivage impossible ; resultats dans $PWD/$opt_results"
    if [ "$opt_status" -eq 0 ]; then opt_status=$opt_tar_status; fi
  fi
  exit "$opt_status"
}
trap opt_finish EXIT
shasum -a 256 -c SHA256SUMS > "$opt_results/integrity.txt"
cp SHA256SUMS "$opt_results/kit-SHA256SUMS.txt"
cp protocol.json "$opt_results/protocol.json"
cp generated/optimization-verification.json "$opt_results/linux-verification.json"
echo "$opt_mode" > "$opt_results/mode.txt"
unset SDKROOT CPPFLAGS LDFLAGS LIBS CFLAGS CXXFLAGS CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH
unset COMPILER_PATH GCC_EXEC_PREFIX MACOSX_DEPLOYMENT_TARGET CONFIG_SITE
opt_sdk="$(xcrun --sdk macosx --show-sdk-path)"
export SDKROOT="$opt_sdk"
sw_vers > "$opt_results/macos.txt"
xcrun --sdk macosx clang++ --version > "$opt_results/compiler.txt"
sysctl -n machdep.cpu.brand_string > "$opt_results/cpu.txt"
echo "Verification des 1600 vecteurs Montmory ARM64/x86."
xcrun --sdk macosx clang -arch arm64 -isysroot "$opt_sdk" -std=c11 -O2 -Wall -Wextra -Werror -Isrc \
  src/conformance_dump.c -o build/conformance_dump 2>&1 | tee "$opt_results/core-build.txt"
build/conformance_dump > "$opt_results/golden-vectors.txt"
cmp reference/golden-vectors.txt "$opt_results/golden-vectors.txt"
echo "Vecteurs ARM64/x86 : PASS."
if [ "$opt_mode" != metal ]; then
  echo "CPU : quatre implementations equivalentes, neuf repetitions par nombre de workers."
  xcrun --sdk macosx clang++ -arch arm64 -isysroot "$opt_sdk" -std=c++17 -O3 -pthread -Wall -Wextra -Werror \
    -Isrc -Ivendor/include src/cpu_benchmark.cpp -o build/cpu_benchmark 2>&1 | tee "$opt_results/cpu-build.txt"
  build/cpu_benchmark "$opt_results" 2>&1 | tee "$opt_results/cpu-console.txt"
fi
if [ "$opt_mode" != cpu ]; then
  echo "Metal : sept variantes, reference v3, Philox, Threefry et temoin d'ecriture."
  xcrun --sdk macosx clang++ -arch arm64 -isysroot "$opt_sdk" -std=c++17 -O3 -fobjc-arc -Wall -Wextra -Werror \
    -Isrc -Ivendor/include src/benchmark.mm -framework Foundation -framework Metal -o build/metal_opt_benchmark \
    2>&1 | tee "$opt_results/metal-build.txt"
  build/metal_opt_benchmark generated/all.metal "$opt_results" 2>&1 | tee "$opt_results/metal-console.txt"
fi
echo "PASS. Les performances sont celles de cette machine et de cette execution. Aucune transmission automatique."
