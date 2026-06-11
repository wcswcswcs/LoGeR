#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/audit/v13_geometry_diagnostic

run_logged() {
  local log_path="$1"
  shift
  "$@" > "$log_path" 2>&1
}

run_logged logs/stream4d_v13_geometry_py_compile.log \
  "$PY" -m py_compile \
    tools/d4rt_geometry_diagnostic.py \
    tools/summarize_v13_geometry_diagnostic.py

run_logged logs/stream4d_v13_geometry_sim3_residual.log \
  "$PY" -m tools.d4rt_geometry_diagnostic \
    --debug-root outputs/v10_d4rt_grid_surfel_field/stream4d_v10_g1_grid32m002_probe5_16f_stride1_fresh_gpu67 \
    --seq-list splits/scannet_v6_probe5.txt \
    --output-prefix outputs/audit/v13_geometry_diagnostic/d4rt_sim3_residual_probe5 \
    --scannet-root data/scannet/processed \
    --backbone Cropformer \
    --max-anchors-per-window 2000 \
    --min-visibility 0.5 \
    --min-confidence 0.5

run_logged logs/stream4d_v13_geometry_summary.log \
  "$PY" -m tools.summarize_v13_geometry_diagnostic \
    --root . \
    --output-prefix outputs/audit/v13_geometry_diagnostic/geometry_diagnostic_probe5 \
    --sim3-diagnostic-json outputs/audit/v13_geometry_diagnostic/d4rt_sim3_residual_probe5.json

echo "v13 geometry diagnostic probe5 done"
