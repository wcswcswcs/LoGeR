#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"

mkdir -p outputs/audit/v14_logs outputs/audit/v14_failure_decomposition

"$PY" -m tools.diagnose_v14_failure_decomposition \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --source B1:stream4d_v8_b1_surfacelet_singlemask_probe5 \
  --source O38:stream4d_v9_o38_o22_memory_c055_newpts_logarea_probe5:stream4d_v11_oracle_c2_o38_memory_probe5 \
  --source C_mask:stream4d_v13_c_mask_unsup_probe5:stream4d_v13_oracle_c_mask_probe5 \
  --source C_regionlet:stream4d_v13_c_regionlet_unsup_probe5:stream4d_v13_oracle_c_regionlet_probe5 \
  --source C_surfel:stream4d_v13_c_surfel_unsup_probe5:stream4d_v13_oracle_c_surfel_probe5 \
  --source C_masklet:stream4d_v13_c3_masklet_candidate_probe5:stream4d_v13_oracle_c_masklet_probe5 \
  --source C_hybrid:stream4d_v13_c_hybrid_unsup_probe5:stream4d_v13_oracle_c_hybrid_probe5 \
  --source M13c:stream4d_v13_m13c_mdl_c3_fullmask_probe5 \
  --source M13d:stream4d_v13_m13d_mdl_c3_posterior_wta_probe5 \
  --method M13c:stream4d_v13_m13c_mdl_c3_fullmask_probe5 \
  --method M13d:stream4d_v13_m13d_mdl_c3_posterior_wta_probe5 \
  --output-prefix outputs/audit/v14_failure_decomposition/failure_decomposition_probe5 \
  --visual-dir outputs/audit/v14_failure_decomposition/visuals \
  --visual-limit 30 \
  > outputs/audit/v14_logs/v14_failure_decomposition_reproduce.log 2>&1
