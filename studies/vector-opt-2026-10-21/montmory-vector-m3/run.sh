#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  echo "Ce kit doit tourner sur le Mac Apple Silicon, dans un terminal ARM64."
  exit 2
fi
if pgrep -f '[/](testu01_metal|lanes|cpu_benchmark|metal_opt_benchmark|metal_confirm_benchmark|metal_vector_benchmark|confirm_pool)([[:space:]]|$)' >/dev/null 2>&1; then
  echo "Un test PRNG connu tourne encore. Attends sa fin."
  exit 2
fi
vector_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
vector_results="results/$vector_id"
vector_archive="montmory-vector-results-$vector_id.tar.gz"
mkdir -p build "$vector_results"
vector_finish() {
  vector_status=$?
  trap - EXIT
  set +e
  echo "$vector_status" > "$vector_results/exit-code.txt"
  tar -czf "$vector_archive" "$vector_results"
  vector_tar_status=$?
  if [ "$vector_tar_status" -eq 0 ]; then
    echo "Resultats et diagnostics : $PWD/$vector_archive"
  else
    echo "Archivage impossible : $PWD/$vector_results"
    if [ "$vector_status" -eq 0 ]; then vector_status=$vector_tar_status; fi
  fi
  exit "$vector_status"
}
trap vector_finish EXIT
shasum -a 256 -c SHA256SUMS > "$vector_results/integrity.txt"
cp SHA256SUMS "$vector_results/kit-SHA256SUMS.txt"
cp protocol.json "$vector_results/"
unset SDKROOT CPPFLAGS LDFLAGS LIBS CFLAGS CXXFLAGS CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH
unset COMPILER_PATH GCC_EXEC_PREFIX MACOSX_DEPLOYMENT_TARGET CONFIG_SITE
vector_sdk="$(xcrun --sdk macosx --show-sdk-path)"
export SDKROOT="$vector_sdk"
sw_vers > "$vector_results/macos.txt"
xcrun --sdk macosx clang++ --version > "$vector_results/compiler.txt"
sysctl -n machdep.cpu.brand_string > "$vector_results/cpu.txt"
echo "1/5 Verification des 1600 vecteurs ARM64/x86."
xcrun --sdk macosx clang -arch arm64 -isysroot "$vector_sdk" -std=c11 -O2 -Wall -Wextra -Werror -Isrc \
  src/conformance_dump.c -o build/conformance_dump 2>&1 | tee "$vector_results/core-build.txt"
build/conformance_dump > "$vector_results/golden-vectors.txt"
cmp reference/golden-vectors.txt "$vector_results/golden-vectors.txt"
echo "Vecteurs ARM64/x86 : PASS."
echo "2/5 Verification SIMD/NEON, decoupages, debordements et workers reutilises, avec UBSan."
xcrun --sdk macosx clang++ -arch arm64 -isysroot "$vector_sdk" -std=c++17 -O2 -Wall -Wextra -Werror -pthread \
  -fsanitize=undefined -fno-sanitize-recover=all -Isrc src/verify.cpp -o build/verify \
  2>&1 | tee "$vector_results/verify-build.txt"
build/verify | tee "$vector_results/cpu-native-ubsan.json"
echo "3/5 Mesures CPU : petits et grands blocs, geometrie logique inchangee."
xcrun --sdk macosx clang++ -arch arm64 -isysroot "$vector_sdk" -std=c++17 -O3 -Wall -Wextra -Werror -pthread \
  -Isrc src/cpu_benchmark.cpp -o build/cpu_benchmark 2>&1 | tee "$vector_results/cpu-build.txt"
build/cpu_benchmark "$vector_results" | tee "$vector_results/cpu-console.txt"
echo "4/5 Confirmation separee des workers reutilises : sept nouvelles seeds."
xcrun --sdk macosx clang++ -arch arm64 -isysroot "$vector_sdk" -std=c++17 -O3 -Wall -Wextra -Werror -pthread \
  -Isrc src/confirm_pool.cpp -o build/confirm_pool 2>&1 | tee "$vector_results/pool-build.txt"
build/confirm_pool "$vector_results" | tee "$vector_results/pool-console.txt"
if command -v python3 >/dev/null 2>&1; then
  python3 tools/analyze_cpu.py "$vector_results" | tee "$vector_results/cpu-analysis.txt"
fi
echo "5/5 Metal : conformite, recherche puis confirmation sur de nouvelles seeds."
xcrun --sdk macosx clang++ -arch arm64 -isysroot "$vector_sdk" -std=c++17 -O3 -fobjc-arc -Wall -Wextra -Werror \
  -Isrc -Ivendor/include src/vector_benchmark.mm -framework Foundation -framework Metal -o build/metal_vector_benchmark \
  2>&1 | tee "$vector_results/metal-build.txt"
build/metal_vector_benchmark generated/all.metal "$vector_results" | tee "$vector_results/metal-console.txt"
echo "PASS. Chaque sortie chronometree a ete comparee. Aucune transmission automatique."
