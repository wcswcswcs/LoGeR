#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
PY="${PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"
SEQ="${SEQ:-splits/scannet_v6_probe5.txt}"
BANK_ROOT="${BANK_ROOT:-outputs/v14_measurement_bank_bank16_cropformer}"
GRAPH_ROOT="${GRAPH_ROOT:-outputs/audit/v18_phase1}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-outputs/audit/v18_phase3}"
BOUNDARY_SAFE_PX="${BOUNDARY_SAFE_PX:-3.0}"
CUT_LAMBDA="${CUT_LAMBDA:-1.0}"
MERGE_LAMBDA="${MERGE_LAMBDA:-0.65}"
LOG_DIR="${LOG_DIR:-outputs/audit/v18_logs}"
mkdir -p "$LOG_DIR"

variants=(
  E0_mask_co_membership_baseline
  E1_mask_signed
  E2_mask_signed_boundary_safe
  E3_mask_signed_depth_normal
  E4_mask_signed_d4rt_temporal
  E5_full_signed
  E6_shuffle_d4rt
  E7_no_temporal
)

for variant in "${variants[@]}"; do
  "$PY" -m tools.build_v18_signed_boundary_evidence \
    --bank-root "$BANK_ROOT" \
    --graph-root "$GRAPH_ROOT" \
    --seq-list "$SEQ" \
    --variant "$variant" \
    --output-root "$EVIDENCE_ROOT" \
    --boundary-safe-px "$BOUNDARY_SAFE_PX" \
    --cut-lambda "$CUT_LAMBDA" \
    --merge-lambda "$MERGE_LAMBDA" \
    2>&1 | tee "$LOG_DIR/phase3_build_${variant}.log"

  "$PY" -m tools.diagnose_v18_edge_boundary_quality \
    --bank-root "$BANK_ROOT" \
    --graph-root "$GRAPH_ROOT" \
    --evidence-root "$EVIDENCE_ROOT" \
    --seq-list "$SEQ" \
    --mode evidence \
    --variant "$variant" \
    --output-prefix "$EVIDENCE_ROOT/$variant/edge_boundary_quality_probe5" \
    2>&1 | tee "$LOG_DIR/phase3_quality_${variant}.log"
done
