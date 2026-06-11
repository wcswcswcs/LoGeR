#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
PY="${PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"
SEQ="${SEQ:-splits/scannet_v6_probe5.txt}"
BANK_ROOT="${BANK_ROOT:-outputs/v14_measurement_bank_bank16_cropformer}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/audit/v18_phase1}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-$OUTPUT_ROOT/signed_surfel_graph_probe5}"
KNN_K="${KNN_K:-8}"
KNN_MAX_FRAMES="${KNN_MAX_FRAMES:-16}"
CROSS_FRAME_NEIGHBORS="${CROSS_FRAME_NEIGHBORS:-4}"
PRECUT_MASK_DISAGREEMENT_RATIO="${PRECUT_MASK_DISAGREEMENT_RATIO:-0.25}"
PRECUT_SOURCE_RGB_DISCONTINUITY="${PRECUT_SOURCE_RGB_DISCONTINUITY:-0.45}"
PRECUT_UV_DISCONTINUITY="${PRECUT_UV_DISCONTINUITY:-0.06}"
LOG_DIR="${LOG_DIR:-outputs/audit/v18_logs}"
mkdir -p "$LOG_DIR"

"$PY" -m tools.build_v18_signed_surfel_graph \
  --bank-root "$BANK_ROOT" \
  --seq-list "$SEQ" \
  --output-root "$OUTPUT_ROOT" \
  --output-prefix "$OUTPUT_PREFIX" \
  --knn-k "$KNN_K" \
  --knn-max-frames "$KNN_MAX_FRAMES" \
  --cross-frame-neighbors "$CROSS_FRAME_NEIGHBORS" \
  --precut-mask-disagreement-ratio "$PRECUT_MASK_DISAGREEMENT_RATIO" \
  --precut-source-rgb-discontinuity "$PRECUT_SOURCE_RGB_DISCONTINUITY" \
  --precut-uv-discontinuity "$PRECUT_UV_DISCONTINUITY" \
  2>&1 | tee "$LOG_DIR/phase1_graph_${KNN_K}.log"
