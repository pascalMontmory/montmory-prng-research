#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ "$(uname -s)" != Darwin ]; then
  echo "Ce script doit tourner sur le Mac Apple Silicon."
  exit 2
fi
if pgrep -f '[/](testu01_metal|lanes|cpu_benchmark|metal_opt_benchmark|metal_confirm_benchmark)([[:space:]]|$)' >/dev/null 2>&1; then
  echo "Un test PRNG connu tourne encore. Attends sa fin."
  exit 2
fi
confirm_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
confirm_results="results/$confirm_id"
confirm_archive="montmory-metal-confirm-results-$confirm_id.tar.gz"
mkdir -p build "$confirm_results"
confirm_finish() {
  confirm_status=$?
  trap - EXIT
  set +e
  echo "$confirm_status" > "$confirm_results/exit-code.txt"
  tar -czf "$confirm_archive" "$confirm_results"
  confirm_tar_status=$?
  if [ "$confirm_tar_status" -eq 0 ]; then
    echo "Resultats et diagnostics : $PWD/$confirm_archive"
  else
    echo "Archivage impossible : $PWD/$confirm_results"
    if [ "$confirm_status" -eq 0 ]; then confirm_status=$confirm_tar_status; fi
  fi
  exit "$confirm_status"
}
trap confirm_finish EXIT
shasum -a 256 -c SHA256SUMS > "$confirm_results/integrity.txt"
cp SHA256SUMS "$confirm_results/kit-SHA256SUMS.txt"
cp protocol.json frozen-selection.json "$confirm_results/"
unset SDKROOT CPPFLAGS LDFLAGS LIBS CFLAGS CXXFLAGS CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH
unset COMPILER_PATH GCC_EXEC_PREFIX MACOSX_DEPLOYMENT_TARGET CONFIG_SITE
confirm_sdk="$(xcrun --sdk macosx --show-sdk-path)"
export SDKROOT="$confirm_sdk"
sw_vers > "$confirm_results/macos.txt"
xcrun --sdk macosx clang++ --version > "$confirm_results/compiler.txt"
sysctl -n machdep.cpu.brand_string > "$confirm_results/cpu.txt"
echo "Verification des 1600 vecteurs ARM64/x86."
xcrun --sdk macosx clang -arch arm64 -isysroot "$confirm_sdk" -std=c11 -O2 -Wall -Wextra -Werror -Isrc \
  src/conformance_dump.c -o build/conformance_dump 2>&1 | tee "$confirm_results/core-build.txt"
build/conformance_dump > "$confirm_results/golden-vectors.txt"
cmp reference/golden-vectors.txt "$confirm_results/golden-vectors.txt"
echo "Vecteurs ARM64/x86 : PASS."
xcrun --sdk macosx clang++ -arch arm64 -isysroot "$confirm_sdk" -std=c++17 -O3 -fobjc-arc -Wall -Wextra -Werror \
  -Isrc -Ivendor/include src/confirm.mm -framework Foundation -framework Metal -o build/metal_confirm_benchmark \
  2>&1 | tee "$confirm_results/metal-build.txt"
build/metal_confirm_benchmark generated/all.metal "$confirm_results" 2>&1 | tee "$confirm_results/metal-console.txt"
