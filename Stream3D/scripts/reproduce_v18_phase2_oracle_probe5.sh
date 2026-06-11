#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
PY="${PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"
SEQ="${SEQ:-splits/scannet_v6_probe5.txt}"
BANK_ROOT="${BANK_ROOT:-outputs/v14_measurement_bank_bank16_cropformer}"
GRAPH_ROOT="${GRAPH_ROOT:-outputs/audit/v18_phase1}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-outputs/audit/v18_phase2/edge_oracle_probe5}"
ORACLE_CONFIG="${ORACLE_CONFIG:-stream4d_v18_edge_oracle_probe5}"
NN_RADIUS="${NN_RADIUS:-0.05}"
MAX_GT_LABEL_FRAMES="${MAX_GT_LABEL_FRAMES:-16}"
ORACLE_EXPORT_CORE_NN_RADIUS="${ORACLE_EXPORT_CORE_NN_RADIUS:-0.05}"
ORACLE_EXPORT_FRINGE_NN_RADIUS="${ORACLE_EXPORT_FRINGE_NN_RADIUS:-0.05}"
ORACLE_EXPORT_FRINGE_RADIUS="${ORACLE_EXPORT_FRINGE_RADIUS:-0.0}"
ORACLE_EXPORT_FRINGE_MAX_RATIO="${ORACLE_EXPORT_FRINGE_MAX_RATIO:-0.35}"
LOG_DIR="${LOG_DIR:-outputs/audit/v18_logs}"
mkdir -p "$LOG_DIR"

"$PY" -m tools.diagnose_v18_edge_boundary_quality \
  --bank-root "$BANK_ROOT" \
  --graph-root "$GRAPH_ROOT" \
  --seq-list "$SEQ" \
  --mode oracle \
  --output-prefix "$OUTPUT_PREFIX" \
  --oracle-output-config "$ORACLE_CONFIG" \
  --nn-radius "$NN_RADIUS" \
  --max-gt-label-frames "$MAX_GT_LABEL_FRAMES" \
  --oracle-export-core-nn-radius "$ORACLE_EXPORT_CORE_NN_RADIUS" \
  --oracle-export-fringe-nn-radius "$ORACLE_EXPORT_FRINGE_NN_RADIUS" \
  --oracle-export-fringe-radius "$ORACLE_EXPORT_FRINGE_RADIUS" \
  --oracle-export-fringe-max-ratio "$ORACLE_EXPORT_FRINGE_MAX_RATIO" \
  2>&1 | tee "$LOG_DIR/phase2_oracle.log"
