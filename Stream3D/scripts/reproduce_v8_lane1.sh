#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
D4RT="$ROOT/Open-d4rt"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"

"$PY" -m tools.compare_d4rt_adapter_official_v8 \
  --d4rt-root "$D4RT" \
  --d4rt-config "$D4RT/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml" \
  --d4rt-ckpt "$D4RT/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt" \
  --seq-name scene0050_00 \
  --backbone Cropformer \
  --frame-stride 10 \
  --max-frames 16 \
  --grid-size 4 \
  --grid-margin-ratio 0.02 \
  --query-chunk-size 1024 \
  --output-prefix outputs/audit/v8_adapter_vs_official_scene0050_grid4_loger

"$PY" -m tools.export_d4rt_grid_surfel_field_v8 \
  --d4rt-root "$D4RT" \
  --d4rt-config "$D4RT/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml" \
  --d4rt-ckpt "$D4RT/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt" \
  --seq-list splits/scannet_v6_probe5.txt \
  --backbone Cropformer \
  --frame-stride 1 \
  --max-frames 16 \
  --window-size 16 \
  --window-stride 16 \
  --grid-size 32 \
  --grid-margin-ratio 0.02 \
  --allow-missing-masks \
  --cycle-max-tracks 128 \
  --query-chunk-size 4096 \
  --save-overlays \
  --run-name stream4d_v8_g1_grid32m002_probe5_16f_stride1_loger

"$PY" -m tools.d4rt_geometry_diagnostic \
  --debug-root outputs/v8_d4rt_grid_surfel_field/stream4d_v8_g1_grid32m002_probe5_16f_stride1_loger \
  --seq-list splits/scannet_v6_probe5.txt \
  --output-prefix outputs/audit/v8_g1_grid32m002_probe5_16f_stride1_loger_geometry

"$PY" -m tools.diagnose_v8_mask_measurement_coverage \
  --debug-root outputs/v8_d4rt_grid_surfel_field/stream4d_v8_g1_grid32m002_probe5_16f_stride1_loger \
  --seq-list splits/scannet_v6_probe5.txt \
  --backbone Cropformer \
  --rho-min 0.35 \
  --output-prefix outputs/audit/v8_mask_measurement_coverage_probe5_stride1_loger
