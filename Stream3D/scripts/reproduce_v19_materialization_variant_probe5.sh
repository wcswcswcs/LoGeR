#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
PY="${PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"
SEQ="${SEQ:-splits/scannet_v6_probe5.txt}"
VARIANT="${VARIANT:-M1}"
BANK_ROOT="${BANK_ROOT:-outputs/v14_measurement_bank_bank16_cropformer}"
GRAPH_ROOT="${GRAPH_ROOT:-outputs/audit/v18_phase1_repair_precut_k16_d015}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/audit/v19_phase2a_${VARIANT}}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-$OUTPUT_ROOT/materialization_probe5}"
OUTPUT_CONFIG_PREFIX="${OUTPUT_CONFIG_PREFIX:-stream4d_v19}"
NN_RADIUS="${NN_RADIUS:-0.05}"
MAX_FRAMES="${MAX_FRAMES:-16}"
M2_DILATION_RADIUS="${M2_DILATION_RADIUS:-0.03}"
SKIP_EVAL="${SKIP_EVAL:-0}"
LOG_DIR="${LOG_DIR:-outputs/audit/v19_logs}"
mkdir -p "$LOG_DIR"

EXTRA_ARGS=()
if [[ "$SKIP_EVAL" == "1" ]]; then
  EXTRA_ARGS+=(--skip-eval)
fi

"$PY" -m tools.diagnose_v19_materialization \
  --variant "$VARIANT" \
  --bank-root "$BANK_ROOT" \
  --graph-root "$GRAPH_ROOT" \
  --seq-list "$SEQ" \
  --output-prefix "$OUTPUT_PREFIX" \
  --output-config-prefix "$OUTPUT_CONFIG_PREFIX" \
  --nn-radius "$NN_RADIUS" \
  --max-frames "$MAX_FRAMES" \
  --m2-dilation-radius "$M2_DILATION_RADIUS" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG_DIR/phase2a_${VARIANT}.log"
