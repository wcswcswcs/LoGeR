#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
PY="${PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"
SEQ="${SEQ:-splits/scannet_v6_probe5.txt}"
BANK_ROOT="${BANK_ROOT:-outputs/v14_measurement_bank_bank16_cropformer}"
GRAPH_ROOT="${GRAPH_ROOT:-outputs/audit/v18_phase1}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-outputs/audit/v18_phase3}"
VARIANT="${VARIANT:-E5_full_signed}"
LOG_DIR="${LOG_DIR:-outputs/audit/v18_logs}"
mkdir -p "$LOG_DIR"

"$PY" -m tools.export_v18_signed_graph_partition \
  --bank-root "$BANK_ROOT" \
  --graph-root "$GRAPH_ROOT" \
  --evidence-root "$EVIDENCE_ROOT" \
  --seq-list "$SEQ" \
  --variant "$VARIANT" \
  --output-config stream4d_v18_gcore_probe5 \
  --export-mode G_core \
  2>&1 | tee "$LOG_DIR/phase4_gcore_${VARIANT}.log"

"$PY" -m tools.export_v18_signed_graph_partition \
  --bank-root "$BANK_ROOT" \
  --graph-root "$GRAPH_ROOT" \
  --evidence-root "$EVIDENCE_ROOT" \
  --seq-list "$SEQ" \
  --variant "$VARIANT" \
  --output-config stream4d_v18_gregion_fill_probe5 \
  --export-mode G_region_fill \
  2>&1 | tee "$LOG_DIR/phase4_gregion_fill_${VARIANT}.log"
