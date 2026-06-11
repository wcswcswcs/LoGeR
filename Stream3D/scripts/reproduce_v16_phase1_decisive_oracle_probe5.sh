#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
PY="${PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"

mkdir -p outputs/audit/v16_logs outputs/audit/v16_phase1

"$PY" -m tools.diagnose_v15_union_oracle \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --pred-config stream4d_v13_c_mask_unsup_probe5 \
  --pre-points-config stream4d_v13_c_mask_unsup_probe5 \
  --output-config-prefix stream4d_v16_oracle_c_mask_union_probe5 \
  --summary-prefix outputs/audit/v16_phase1/c_mask_union_oracle_probe5 \
  --algorithm-name v16_union_oracle \
  --k 2 --k 4 --k 8 \
  --max-candidates-per-gt 512 \
  > outputs/audit/v16_logs/v16_phase1_c_mask_union_oracle.log 2>&1

"$PY" -m tools.diagnose_v15_union_oracle \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --pred-config stream4d_v13_c_hybrid_unsup_probe5 \
  --pre-points-config stream4d_v13_c_hybrid_unsup_probe5 \
  --output-config-prefix stream4d_v16_oracle_c_hybrid_union_probe5 \
  --summary-prefix outputs/audit/v16_phase1/c_hybrid_union_oracle_probe5 \
  --algorithm-name v16_union_oracle \
  --k 2 --k 4 --k 8 \
  --max-candidates-per-gt 512 \
  > outputs/audit/v16_logs/v16_phase1_c_hybrid_union_oracle.log 2>&1

"$PY" -m tools.diagnose_v15_union_oracle \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --pred-config stream4d_v13_c_regionlet_unsup_probe5 \
  --pre-points-config stream4d_v13_c_regionlet_unsup_probe5 \
  --output-config-prefix stream4d_v16_oracle_c_regionlet_union_probe5 \
  --summary-prefix outputs/audit/v16_phase1/c_regionlet_union_oracle_probe5 \
  --algorithm-name v16_union_oracle \
  --k 2 --k 4 --k 8 \
  --max-candidates-per-gt 512 \
  > outputs/audit/v16_logs/v16_phase1_c_regionlet_union_oracle.log 2>&1

"$PY" -m tools.diagnose_v15_union_oracle \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --pred-config stream4d_v13_c_hybrid_unsup_probe5 \
  --pre-points-config stream4d_v13_c_hybrid_unsup_probe5 \
  --output-config-prefix stream4d_v16_oracle_c_hybrid_union_stress_probe5 \
  --summary-prefix outputs/audit/v16_phase1/c_hybrid_union_stress_probe5 \
  --algorithm-name v16_union_oracle_stress \
  --k 16 --k 32 \
  --max-candidates-per-gt 768 \
  > outputs/audit/v16_logs/v16_phase1_c_hybrid_union_stress.log 2>&1

"$PY" -m tools.diagnose_v15_union_oracle \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --pred-config stream4d_v13_c_hybrid_unsup_probe5 \
  --pre-points-config stream4d_v13_c_hybrid_unsup_probe5 \
  --output-config-prefix stream4d_v16_oracle_c_hybrid_union_min50_probe5 \
  --summary-prefix outputs/audit/v16_phase1/c_hybrid_union_min50_probe5 \
  --algorithm-name v16_union_oracle_min50 \
  --k 2 --k 4 --k 8 \
  --min-region-size 50 \
  --max-candidates-per-gt 768 \
  > outputs/audit/v16_logs/v16_phase1_c_hybrid_union_min50.log 2>&1

"$PY" -m tools.summarize_v16_decisive_diagnostics \
  --root . \
  --output-prefix outputs/audit/v16_phase1/three_layer_oracle_matrix_probe5 \
  > outputs/audit/v16_logs/v16_phase1_decisive_summary.log 2>&1
