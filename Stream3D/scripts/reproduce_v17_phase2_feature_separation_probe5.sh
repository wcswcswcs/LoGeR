#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"
mkdir -p outputs/audit/v17_logs outputs/audit/v17_phase2

"$PY" -m tools.diagnose_v17_oracle_feature_separation \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --pred-config stream4d_v13_c_hybrid_unsup_probe5 \
  --regionlet-config stream4d_v13_c_regionlet_unsup_probe5 \
  --mask-config stream4d_v13_c_mask_unsup_probe5 \
  --surfel-config stream4d_v13_c_surfel_unsup_probe5 \
  --oracle-json outputs/audit/v16_phase1/c_hybrid_union_oracle_probe5.json \
  --k 8 \
  --output-prefix outputs/audit/v17_phase2/c_hybrid_oracle_feature_separation_probe5 \
  > outputs/audit/v17_logs/v17_phase2_c_hybrid_feature_separation.log 2>&1
