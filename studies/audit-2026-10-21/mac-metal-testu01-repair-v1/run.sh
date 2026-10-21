#!/bin/bash
# Companion for the frozen v3 kit: build environment and diagnostics only.
set -euo pipefail
repair_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$repair_dir/.."
if [ "$(uname -s)" != Darwin ]; then echo "Ce lanceur demande le Mac Apple Silicon."; exit 2; fi
if [ ! -f src/testu01_metal.mm ] || [ ! -f vendor/TestU01.zip ]; then
  echo "Extraire testu01-repair-v1 dans le dossier montmory-metal-m3-v3."
  exit 2
fi
test_engine="${1:-montmory_mulhi_pair}"
test_battery="${2:-all}"
case "$test_battery" in small|crush|big|all) ;; *) echo "Batterie attendue: small, crush, big ou all."; exit 2 ;; esac
test_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
test_results="results/testu01-repair1-$test_id"
test_work="$PWD/build/testu01-repair1-$test_id"
test_source="$test_work/TestU01-1.2.3"
test_prefix="$test_work/install"
test_archive="montmory-metal-v3-testu01-repair1-$test_id.tar.gz"
test_stage="integrite"
mkdir -p "$test_results" "$test_work"
test_finish() {
  test_status=$?; trap - EXIT; set +e
  echo "$test_status" > "$test_results/exit-code.txt"
  echo "$test_stage" > "$test_results/last-stage.txt"
  if [ -f "$test_source/config.log" ]; then cp "$test_source/config.log" "$test_results/config.log"; fi
  if [ "$test_status" -ne 0 ]; then
    echo "ECHEC pendant: $test_stage (code $test_status). Les batteries suivantes ne sont pas lancees."
    if [ -f "$test_results/testu01-build.log" ]; then tail -n 60 "$test_results/testu01-build.log"; fi
    if [ -f "$test_results/config.log" ] && [ "$test_stage" = configuration ]; then
      echo "Detail du premier essai de liaison dans config.log :"
      grep -A 45 'checking for C compiler default output file name' "$test_results/config.log"
    fi
  fi
  tar -czf "$test_archive" "$test_results"
  test_tar_status=$?
  if [ "$test_tar_status" -eq 0 ]; then echo "Archive: $PWD/$test_archive"
  else
    echo "Archivage impossible; journaux conserves dans $PWD/$test_results"
    if [ "$test_status" -eq 0 ]; then test_status=$test_tar_status; fi
  fi
  exit "$test_status"
}
trap test_finish EXIT
shasum -a 256 -c SHA256SUMS > "$test_results/integrity-original.txt"
(cd "$repair_dir" && shasum -a 256 -c SHA256SUMS) > "$test_results/integrity-repair.txt"
cp SHA256SUMS "$test_results/kit-SHA256SUMS.txt"
cp "$repair_dir/SHA256SUMS" "$test_results/repair-SHA256SUMS.txt"
cp generated/provenance.json "$test_results/source-provenance.json"
if [ -f build/TestU01-1.2.3/config.log ]; then cp build/TestU01-1.2.3/config.log "$test_results/previous-config.log"; fi

test_stage="selection SDK macOS"
# These changes affect only this script and its children, never the user's shell.
# Keep DEVELOPER_DIR if selected by the user; resolve the SDK from that Xcode.
unset SDKROOT CPPFLAGS LDFLAGS LIBS CFLAGS CXXFLAGS CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH
unset COMPILER_PATH GCC_EXEC_PREFIX MACOSX_DEPLOYMENT_TARGET CONFIG_SITE
SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
export SDKROOT
test -d "$SDKROOT"
CC="$(xcrun --sdk macosx --find clang)"
CXX="$(xcrun --sdk macosx --find clang++)"
AR="$(xcrun --sdk macosx --find ar)"
RANLIB="$(xcrun --sdk macosx --find ranlib)"
export CC CXX AR RANLIB
export CFLAGS="-O2 -arch arm64"
export CXXFLAGS="-O2 -arch arm64"
sw_vers > "$test_results/macos.txt"
"$CC" --version > "$test_results/compiler.txt"
{
  echo "SDKROOT=$SDKROOT"
  echo "CC=$CC"
  echo "CXX=$CXX"
  echo "CFLAGS=$CFLAGS"
  echo "CXXFLAGS=$CXXFLAGS"
  echo "Source officielle et code Montmory v3 inchanges."
} > "$test_results/build-settings.txt"

test_stage="verification du compilateur ARM64"
cat > "$test_work/compiler-check.c" <<'EOF'
#include <stdint.h>
#include <stdio.h>
#if !defined(__aarch64__) && !defined(__arm64__)
#error ARM64 compiler required
#endif
int main(void) {
    if (sizeof(void*) != 8 || sizeof(uint64_t) != 8) return 1;
    puts("Compilation et execution C ARM64 : PASS.");
    return 0;
}
EOF
"$CC" -O2 -arch arm64 "$test_work/compiler-check.c" -o "$test_work/compiler-check" \
  2>&1 | tee "$test_results/compiler-check-build.log"
"$test_work/compiler-check" | tee "$test_results/compiler-check-run.log"

echo "Construction locale de TestU01 avec le SDK macOS explicite."
test_stage="extraction TestU01"
unzip -q vendor/TestU01.zip -d "$test_work"
test_stage="configuration"
(cd "$test_source" && ./configure --disable-shared --prefix="$test_prefix") \
  > "$test_results/testu01-build.log" 2>&1
test_stage="compilation TestU01"
(cd "$test_source" && make -j2) >> "$test_results/testu01-build.log" 2>&1
test_stage="installation locale TestU01"
(cd "$test_source" && make install) >> "$test_results/testu01-build.log" 2>&1

test_stage="compilation wrapper Metal"
xcrun --sdk macosx clang++ -arch arm64 -std=c++17 -O3 -fobjc-arc -Wall -Wextra -Werror \
  -Isrc -Ivendor/include -I"$test_prefix/include" src/testu01_metal.mm \
  -L"$test_prefix/lib" -ltestu01 -lprobdist -lmylib -lm -framework Foundation -framework Metal \
  -o "$test_work/testu01_metal" 2>&1 | tee "$test_results/wrapper-build.log"
test_batteries="$test_battery"
if [ "$test_battery" = all ]; then test_batteries="small crush big"; fi
echo "Chaque batterie repart de la seed 42. Crush et BigCrush peuvent durer longtemps."
for test_current in $test_batteries; do
  test_stage="batterie $test_current"
  mkdir -p "$test_results/$test_current"
  "$test_work/testu01_metal" "$test_engine" "$test_current" "$test_results/$test_current" \
    2>&1 | tee "$test_results/$test_current/console.log"
  if ! grep -Fq "All tests were passed" "$test_results/$test_current/console.log"; then
    echo "TestU01 n'annonce pas un succes integral; examiner le resume de $test_current."
    exit 3
  fi
done
test_stage="termine"
echo "PASS : toutes les batteries demandees annoncent leur succes."
