#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/v12_measurement_bank outputs/audit/v12_measurement_bank

DEBUG_ROOT="outputs/v10_d4rt_grid_surfel_field/stream4d_v10_g1_grid32m002_probe5_16f_stride1_fresh_gpu67"

"$PY" -m py_compile \
  stream4d/measurement_bank.py \
  tools/build_v12_measurement_bank.py \
  tools/diagnose_v12_measurement_bank.py > logs/stream4d_v12_measurement_bank_py_compile.log 2>&1

"$PY" -m tools.build_v12_measurement_bank \
  --debug-root "$DEBUG_ROOT" \
  --seq-list splits/scannet_v6_probe5.txt \
  --scannet-root data/scannet/processed \
  --backbone Cropformer \
  --output-root outputs/v12_measurement_bank \
  --audit-prefix outputs/audit/v12_measurement_bank/measurement_bank_probe5 \
  --min-visibility 0.5 \
  --min-confidence 0.5 \
  --boundary-safe-px 3.0 > logs/stream4d_v12_measurement_bank_build.log 2>&1

"$PY" -m tools.diagnose_v12_measurement_bank \
  --bank-root outputs/v12_measurement_bank \
  --seq-list splits/scannet_v6_probe5.txt \
  --output-prefix outputs/audit/v12_measurement_bank/measurement_bank_probe5 \
  --boundary-safe-px 3.0 > logs/stream4d_v12_measurement_bank_diagnose.log 2>&1

echo "v12 measurement bank probe5 done"
