#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/v12_object_explanation_repair outputs/audit/v12_object_explanation_repair data/evaluation/scannet

"$PY" -m tools.summarize_v10_unified_eval \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --matrix-json scripts/v12_object_explanation_repair_matrix_probe5.json \
  --output-prefix outputs/audit/v12_object_explanation_repair/object_explanation_repair_matrix_probe5 \
  --plot-dir outputs/audit/v12_object_explanation_repair \
  --dataset scannet \
  --stream3d-config scannet > logs/stream4d_v12_object_repair_matrix.log 2>&1

echo "v12 object explanation repair matrix done"
