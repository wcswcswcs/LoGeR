#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/v13_masklet_measurements outputs/audit/v13_masklet_density

run_logged() {
  local log_path="$1"
  shift
  "$@" > "$log_path" 2>&1
}

run_logged logs/stream4d_v13_masklet_py_compile.log \
  "$PY" -m py_compile \
    stream4d/video_masklet.py \
    tools/build_v13_video_masklet_measurements.py

run_logged logs/stream4d_v13_masklet_density_probe5.log \
  "$PY" -m tools.build_v13_video_masklet_measurements \
    --bank-root outputs/v12_measurement_bank \
    --seq-list splits/scannet_v6_probe5.txt \
    --output-root outputs/v13_masklet_measurements \
    --output-prefix outputs/audit/v13_masklet_density/masklet_density_probe5 \
    --modes C1,C2,C3 \
    --min-birth-surfels 12 \
    --min-frame-surfels 6 \
    --boundary-safe-px 3.0 \
    --c2-min-available-mask-agreement 0.30 \
    --c3-min-available-mask-agreement 0.40 \
    --c3-min-boundary-safe-ratio 0.35 \
    --c3-min-confidence 0.45 \
    --c3-max-negative-visible-outside-ratio 0.75

echo "v13 masklet density probe5 done"
