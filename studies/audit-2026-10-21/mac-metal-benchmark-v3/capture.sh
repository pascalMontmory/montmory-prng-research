#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
# Separate profiling run; do not merge its timings with ordinary benchmark runs.
export MTL_CAPTURE_ENABLED=1
export MONTMORY_CAPTURE="${1:-montmory_spec_dual}"
bash run.sh 4194304 3
