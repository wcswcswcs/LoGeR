#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PATH=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
CFG=stream4d_v8_b1_surfacelet_singlemask_probe5
DEBUG_ROOT=outputs/v8_d4rt_grid_surfel_field/stream4d_v8_g1_grid32m002_probe5_16f_stride1_loger

"$PY" -m py_compile tools/export_v8_surfel_object_field.py

"$PY" -m tools.export_v8_surfel_object_field \
  --debug-root "$DEBUG_ROOT" \
  --seq-list splits/scannet_v6_probe5.txt \
  --output-config "$CFG" \
  --prototype-direction B_surfacelet_singlemask \
  --min-observations 1 \
  --max-observations 1 \
  --min-carriers 16 \
  --min-owned-masks 1 \
  --max-masks-per-object 1 \
  --export-mask-sample-stride 2 \
  --export-mask-max-pixels 50000 \
  --min-points-per-object 20 \
  --summary-root outputs/v8_surfel_object_field

"$PY" -m evaluation.evaluate \
  --pred_path "data/prediction/${CFG}_class_agnostic" \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --no_class \
  --output_file "data/evaluation/scannet/${CFG}_class_agnostic.txt" \
  --require-manifest

"$PY" -m tools.scan_reportable_configs \
  --configs "$CFG" \
  --require-manifest \
  --output outputs/audit/v8_reportable_config_scan_b1_surfacelet_singlemask_probe5.md

"$PY" -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root . \
  --current-root . \
  --configs "$CFG" \
  --seq-list splits/scannet_v6_probe5.txt \
  --output outputs/audit/v8_metric_integrity_b1_surfacelet_singlemask_probe5.md \
  --backbone Cropformer \
  --require-manifest
