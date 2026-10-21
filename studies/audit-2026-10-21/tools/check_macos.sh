#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
project="snapshot/montmory_gpu_test/montmory_gpu_test"
mkdir -p build/macos evidence/macos
xcrun clang -std=c11 -O2 -Wall -Wextra -Werror -I"$project/src" \
  tools/conformance_dump.c -o build/macos/conformance_dump
build/macos/conformance_dump > evidence/macos/golden-vectors.txt
cmp evidence/golden-vectors.txt evidence/macos/golden-vectors.txt
shasum -a 256 evidence/macos/golden-vectors.txt
xcrun -sdk macosx metal -c "$project/metal/mmc_kernel.metal" -o build/macos/mmc_kernel.air
xcrun -sdk macosx metallib build/macos/mmc_kernel.air -o build/macos/mmc_kernel.metallib
xcrun clang++ -std=c++17 -O2 -fobjc-arc -I"$project/src" \
  tools/check_metal.mm -framework Foundation -framework Metal -o build/macos/check_metal
build/macos/check_metal build/macos/mmc_kernel.metallib > evidence/macos/metal-check.json
cat evidence/macos/metal-check.json
