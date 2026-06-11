#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
PY="${PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"
SEQ="${SEQ:-splits/scannet_v6_probe5.txt}"
LOG_DIR="${LOG_DIR:-outputs/audit/v19_logs}"
mkdir -p "$LOG_DIR"

for VARIANT in M0 M1 M2 M3; do
  PY="$PY" \
  SEQ="$SEQ" \
  VARIANT="$VARIANT" \
  LOG_DIR="$LOG_DIR" \
  bash scripts/reproduce_v19_materialization_variant_probe5.sh
done
