#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/v10_d4rt_grid_surfel_field

RUN_NAME="stream4d_v10_g1_grid32m002_probe5_16f_stride1_fresh_gpu67"

"$PY" -m tools.export_d4rt_grid_surfel_field_v8 \
  --d4rt-root "$ROOT/Open-d4rt" \
  --d4rt-config "$ROOT/Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml" \
  --d4rt-ckpt "$ROOT/Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt" \
  --device cuda:0 \
  --seq-list splits/scannet_v6_probe5.txt \
  --frame-stride 1 \
  --max-frames 16 \
  --window-size 16 \
  --window-stride 16 \
  --grid-size 32 \
  --grid-margin-ratio 0.02 \
  --visible-min-visibility 0.5 \
  --visible-min-confidence 0.5 \
  --query-chunk-size 2048 \
  --cycle-max-tracks 256 \
  --cycle-source-local 0 \
  --cycle-target-local -1 \
  --output-root outputs/v10_d4rt_grid_surfel_field \
  --run-name "$RUN_NAME" \
  --allow-missing-masks \
  --continue-on-error

echo "fresh D4RT carriers written to outputs/v10_d4rt_grid_surfel_field/${RUN_NAME}"
