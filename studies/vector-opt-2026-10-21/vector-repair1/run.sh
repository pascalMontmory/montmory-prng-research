#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  echo "Ce correctif doit tourner dans montmory-vector-m3 sur le Mac Apple Silicon."
  exit 2
fi
if pgrep -f '[/](testu01_metal|lanes|cpu_benchmark|metal_opt_benchmark|metal_confirm_benchmark|metal_vector_benchmark|metal_vector_repair1|confirm_pool)([[:space:]]|$)' >/dev/null 2>&1; then
  echo "Un test PRNG connu tourne encore. Attends sa fin."
  exit 2
fi
repair_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
repair_results="results/vector-repair1-$repair_id"
repair_archive="montmory-vector-repair1-results-$repair_id.tar.gz"
mkdir -p build "$repair_results"
repair_finish() {
  repair_status=$?
  trap - EXIT
  set +e
  echo "$repair_status" > "$repair_results/exit-code.txt"
  tar -czf "$repair_archive" "$repair_results"
  repair_tar_status=$?
  if [ "$repair_tar_status" -eq 0 ]; then
    echo "Resultats et diagnostics : $PWD/$repair_archive"
  else
    echo "Archivage impossible : $PWD/$repair_results"
    if [ "$repair_status" -eq 0 ]; then repair_status=$repair_tar_status; fi
  fi
  exit "$repair_status"
}
trap repair_finish EXIT
shasum -a 256 -c SHA256SUMS > "$repair_results/original-kit-integrity.txt"
(
  cd vector-repair1
  shasum -a 256 -c SHA256SUMS
) > "$repair_results/repair-integrity.txt"
shasum -a 256 -c vector-repair1/parent-SHA256SUMS > "$repair_results/parent-manifest-integrity.txt"
cp SHA256SUMS "$repair_results/original-kit-SHA256SUMS.txt"
cp vector-repair1/SHA256SUMS "$repair_results/repair-SHA256SUMS.txt"
cp vector-repair1/repair.json "$repair_results/"
cp protocol.json "$repair_results/"
# Preserve the CPU results already supplied by the user, when available.
if [ -d results/20261021T170347Z-22732 ]; then
  cp -R results/20261021T170347Z-22732 "$repair_results/previous-cpu-results"
fi
unset SDKROOT CPPFLAGS LDFLAGS LIBS CFLAGS CXXFLAGS CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH
unset COMPILER_PATH GCC_EXEC_PREFIX MACOSX_DEPLOYMENT_TARGET CONFIG_SITE
repair_sdk="$(xcrun --sdk macosx --show-sdk-path)"
export SDKROOT="$repair_sdk"
sw_vers > "$repair_results/macos.txt"
xcrun --sdk macosx clang++ --version > "$repair_results/compiler.txt"
sysctl -n machdep.cpu.brand_string > "$repair_results/cpu.txt"
echo "1/3 Compilation du programme Metal corrige."
xcrun --sdk macosx clang++ -arch arm64 -isysroot "$repair_sdk" -std=c++17 -O3 -fobjc-arc -Wall -Wextra -Werror \
  -Isrc -Ivendor/include vector-repair1/vector_benchmark.mm -framework Foundation -framework Metal \
  -o build/metal_vector_repair1 2>&1 | tee "$repair_results/metal-build.txt"
echo "2/3 Verification CPU : cas aleatoires ordonnes explicitement, sans refaire les mesures CPU."
xcrun --sdk macosx clang++ -arch arm64 -isysroot "$repair_sdk" -std=c++17 -O2 -Wall -Wextra -Werror -pthread \
  -fsanitize=undefined -fno-sanitize-recover=all -Isrc vector-repair1/verify.cpp -o build/verify_repair1 \
  2>&1 | tee "$repair_results/verify-build.txt"
build/verify_repair1 | tee "$repair_results/cpu-native-ubsan.json"
echo "3/3 Metal : verification native, recherche puis confirmation."
build/metal_vector_repair1 generated/all.metal "$repair_results" | tee "$repair_results/metal-console.txt"
echo "PASS. Sorties chronometrees comparees ; mesures CPU precedentes conservees. Aucun envoi automatique."
