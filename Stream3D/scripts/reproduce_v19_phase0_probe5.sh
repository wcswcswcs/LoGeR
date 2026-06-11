#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
PY="${PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"
SEQ="${SEQ:-splits/scannet_v6_probe5.txt}"
LOG_DIR="${LOG_DIR:-outputs/audit/v19_logs}"
mkdir -p "$LOG_DIR"

"$PY" -m tools.summarize_v18_unified_eval_matrix \
  --root . \
  --seq-list "$SEQ" \
  --output-prefix outputs/audit/v18_phase0/unified_eval_matrix_probe5 \
  2>&1 | tee "$LOG_DIR/phase0_v18_unified_eval_matrix.log"

"$PY" -m tools.summarize_v19_phase0 \
  --root . \
  --output-prefix outputs/audit/v19_phase0/phase0_reproduction_probe5 \
  2>&1 | tee "$LOG_DIR/phase0_v19_reproduction_summary.log"
