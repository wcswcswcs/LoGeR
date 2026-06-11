#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
PY="${PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"

mkdir -p outputs/audit/v16_logs outputs/audit/v16_phase6

"$PY" -m tools.d4rt_geometry_diagnostic \
  --debug-root outputs/v10_d4rt_grid_surfel_field/stream4d_v10_g1_grid32m002_probe5_16f_stride1_fresh_gpu67 \
  --seq-list splits/scannet_v6_probe5.txt \
  --output-prefix outputs/audit/v16_phase6/d4rt_sim3_residual_probe5 \
  --scannet-root data/scannet/processed \
  --backbone Cropformer \
  --max-anchors-per-window 2000 \
  --min-visibility 0.0 \
  --min-confidence 0.0 \
  > outputs/audit/v16_logs/v16_phase6_d4rt_sim3_residual.log 2>&1
