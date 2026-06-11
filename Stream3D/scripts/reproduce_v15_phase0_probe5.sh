#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
PY="${PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"

mkdir -p outputs/audit/v15_logs outputs/audit/v15_phase0

"$PY" -m py_compile stream4d/*.py tools/*.py evaluation/*.py tests/*.py \
  > outputs/audit/v15_logs/py_compile_final.log 2>&1
"$PY" -m unittest discover tests \
  > outputs/audit/v15_logs/unittest_all_final.log 2>&1

find data/prediction -maxdepth 1 -type d -name 'stream4d_v15*_class_agnostic' -printf '%f\n' \
  | sed 's/_class_agnostic$//' \
  | sort \
  > outputs/audit/v15_phase0_configs.txt

CONFIGS="$(paste -sd, outputs/audit/v15_phase0_configs.txt)"
"$PY" -m tools.scan_reportable_configs \
  --root . \
  --configs "$CONFIGS" \
  --output outputs/audit/v15_phase0/reportable_config_scan_v15_probe5.md \
  --require-manifest \
  --require-eval-policy \
  > outputs/audit/v15_logs/v15_reportable_scan.log 2>&1

"$PY" -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root . \
  --current-root . \
  --configs "$CONFIGS" \
  --seq-list splits/scannet_v6_probe5.txt \
  --output outputs/audit/v15_phase0/metric_integrity_v15_probe5.md \
  --require-manifest \
  > outputs/audit/v15_logs/v15_metric_integrity.log 2>&1
