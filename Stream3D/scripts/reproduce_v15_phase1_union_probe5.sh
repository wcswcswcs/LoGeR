#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
PY="${PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"

mkdir -p outputs/audit/v15_logs outputs/audit/v15_phase1

"$PY" -m tools.diagnose_v15_union_oracle \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --pred-config stream4d_v14_a3_bank16_target_atom_candidate_probe5 \
  --pre-points-config stream4d_v14_a3_bank16_target_atom_candidate_probe5 \
  --output-config-prefix stream4d_v15_oracle_a3t16_union_probe5 \
  --summary-prefix outputs/audit/v15_phase1/a3t16_union_oracle_probe5 \
  --eval-support candidate \
  > outputs/audit/v15_logs/v15_phase1_a3t16_union_oracle.log 2>&1

"$PY" -m tools.diagnose_v15_union_oracle \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --pred-config stream4d_v14_a4_bank16_target_atom_candidate_probe5 \
  --pre-points-config stream4d_v14_a4_bank16_target_atom_candidate_probe5 \
  --output-config-prefix stream4d_v15_oracle_a4t16_union_probe5 \
  --summary-prefix outputs/audit/v15_phase1/a4t16_union_oracle_probe5 \
  --eval-support candidate \
  > outputs/audit/v15_logs/v15_phase1_a4t16_union_oracle.log 2>&1

"$PY" -m tools.diagnose_v15_union_oracle \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --pred-config stream4d_v14_a4_bank16_target_minpts5_atom_candidate_probe5 \
  --pre-points-config stream4d_v14_a4_bank16_target_minpts5_atom_candidate_probe5 \
  --output-config-prefix stream4d_v15_oracle_a4t16mp5_union_probe5 \
  --summary-prefix outputs/audit/v15_phase1/a4t16mp5_union_oracle_probe5 \
  --eval-support candidate \
  > outputs/audit/v15_logs/v15_phase1_a4t16mp5_union_oracle.log 2>&1
