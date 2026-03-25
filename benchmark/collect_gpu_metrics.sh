#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-benchmark/results/gpu_metrics.csv}"
mkdir -p "$(dirname "$OUT")"

nvidia-smi \
  --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits \
  -l 1 > "$OUT"

