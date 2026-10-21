#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ "$(uname -s)" != Darwin ]; then echo "Ce script demande le Mac Apple Silicon."; exit 2; fi
test_engine="${1:-montmory_u32_tiled}"
test_battery="${2:-small}"
case "$test_battery" in small|crush|big|all) ;; *) echo "Batterie attendue: small, crush, big ou all."; exit 2 ;; esac
test_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
test_results="results/testu01-$test_id"
mkdir -p "$test_results" build
test_finish() {
  test_status=$?; trap - EXIT
  echo "$test_status" > "$test_results/exit-code.txt"
  tar -czf "montmory-metal-v3-testu01-$test_id.tar.gz" "$test_results"
  echo "Archive: $PWD/montmory-metal-v3-testu01-$test_id.tar.gz"
  exit "$test_status"
}
trap test_finish EXIT
shasum -a 256 -c SHA256SUMS > "$test_results/integrity.txt"
cp SHA256SUMS "$test_results/kit-SHA256SUMS.txt"
cp generated/provenance.json "$test_results/source-provenance.json"
xcrun clang++ --version > "$test_results/compiler.txt"
sw_vers > "$test_results/macos.txt"
test_prefix="$PWD/build/testu01-install"
if [ ! -f "$test_prefix/lib/libtestu01.a" ]; then
  echo "Compilation locale de TestU01 1.2.3; aucune installation systeme."
  unzip -q -o vendor/TestU01.zip -d build
  (
    cd build/TestU01-1.2.3
    CC="$(xcrun -f clang)" CFLAGS="-O2 -arch arm64" ./configure --disable-shared --prefix="$test_prefix"
    make -j2
    make install
  ) > "$test_results/testu01-build.log" 2>&1
fi
xcrun clang++ -arch arm64 -std=c++17 -O3 -fobjc-arc -Wall -Wextra -Werror \
  -Isrc -Ivendor/include -I"$test_prefix/include" src/testu01_metal.mm \
  -L"$test_prefix/lib" -ltestu01 -lprobdist -lmylib -lm -framework Foundation -framework Metal \
  -o build/testu01_metal 2>&1 | tee "$test_results/wrapper-build.log"
test_batteries="$test_battery"
if [ "$test_battery" = all ]; then test_batteries="small crush big"; fi
echo "Crush et BigCrush peuvent durer longtemps. Chaque batterie repart de la seed 42."
for test_current in $test_batteries; do
  mkdir -p "$test_results/$test_current"
  build/testu01_metal "$test_engine" "$test_current" "$test_results/$test_current" \
    2>&1 | tee "$test_results/$test_current/console.log"
  if ! grep -Fq "All tests were passed" "$test_results/$test_current/console.log"; then
    echo "TestU01 n'annonce pas un succes integral; examiner le resume de $test_current."
    exit 3
  fi
done
