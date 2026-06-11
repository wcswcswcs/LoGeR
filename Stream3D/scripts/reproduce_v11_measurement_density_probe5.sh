#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/audit/v11_measurement_density

DEBUG_ROOT="outputs/v10_d4rt_grid_surfel_field/stream4d_v10_g1_grid32m002_probe5_16f_stride1_fresh_gpu67"

run_logged() {
  local log_path="$1"
  shift
  "$@" > "$log_path" 2>&1
}

run_logged logs/stream4d_v11_measurement_density_py_compile.log \
  "$PY" -m py_compile tools/v11_measurement_density_diagnostic.py

run_logged logs/stream4d_v11_measurement_density_probe5.log \
  "$PY" -m tools.v11_measurement_density_diagnostic \
    --debug-root "$DEBUG_ROOT" \
    --seq-list splits/scannet_v6_probe5.txt \
    --scannet-root data/scannet/processed \
    --backbone Cropformer \
    --min-visibility 0.5 \
    --min-confidence 0.5 \
    --seed 11 \
    --output-prefix outputs/audit/v11_measurement_density/measurement_density_probe5

echo "v11 measurement density probe5 done"
