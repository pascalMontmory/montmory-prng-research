#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

if [ "$(uname -s)" != Darwin ]; then
  echo "Ce script doit etre lance sur le Mac Apple Silicon."
  exit 2
fi
if ! xcrun --find clang++ >/dev/null 2>&1; then
  echo "Outils Apple absents. Lance xcode-select --install, termine l'installation, puis relance ce script."
  exit 2
fi
mac_bench_words="${1:-4194304}"
mac_bench_trials="${2:-11}"
case "$mac_bench_words:$mac_bench_trials" in
  *[!0-9:]*|:*|*:) echo "Arguments attendus: nombre de mots, repetitions."; exit 2 ;;
esac
mac_bench_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
mac_bench_results="results/$mac_bench_id"
mkdir -p build "$mac_bench_results"
mac_bench_archive="montmory-metal-v2-results-$mac_bench_id.tar.gz"
mac_bench_finish() {
  mac_bench_status=$?
  trap - EXIT
  echo "$mac_bench_status" > "$mac_bench_results/exit-code.txt"
  tar -czf "$mac_bench_archive" "$mac_bench_results"
  echo
  echo "Resultats et diagnostics: $PWD/$mac_bench_archive"
  exit "$mac_bench_status"
}
trap mac_bench_finish EXIT
shasum -a 256 -c SHA256SUMS > "$mac_bench_results/integrity.txt"
cp SHA256SUMS "$mac_bench_results/kit-SHA256SUMS.txt"
cp generated/provenance.json "$mac_bench_results/source-provenance.json"
sw_vers > "$mac_bench_results/macos.txt"
xcrun clang++ --version > "$mac_bench_results/compiler.txt"
sysctl -n machdep.cpu.brand_string > "$mac_bench_results/cpu.txt"

echo "1/3 Compilation ARM64 et comparaison des 1600 vecteurs Montmory avec le VPS x86."
xcrun clang -arch arm64 -std=c11 -O2 -Wall -Wextra -Werror -Isrc \
  src/conformance_dump.c -o build/conformance_dump 2>&1 | tee "$mac_bench_results/core-build.txt"
build/conformance_dump > "$mac_bench_results/golden-vectors.txt"
cmp reference/golden-vectors.txt "$mac_bench_results/golden-vectors.txt"
echo "Vecteurs ARM64/x86 : PASS."

echo "2/3 Compilation du programme de test."
xcrun clang++ -arch arm64 -std=c++17 -O3 -fobjc-arc -Wall -Wextra -Werror \
  -Isrc -Ivendor/include src/benchmark.mm -framework Foundation -framework Metal \
  -o build/benchmark 2>&1 | tee "$mac_bench_results/benchmark-build.txt"

echo "3/3 Verification sur Metal, puis mesures GPU."
echo "Volume: $mac_bench_words mots de 64 bits; $mac_bench_trials repetitions."
build/benchmark generated/all.metal "$mac_bench_results" "$mac_bench_words" "$mac_bench_trials" \
  2>&1 | tee "$mac_bench_results/console.txt"

echo "Le tableau final compare les debits sur ton GPU. Aucun resultat n'a ete envoye automatiquement."
