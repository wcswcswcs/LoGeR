#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"

mkdir -p outputs/audit/v14_final outputs/audit/v14_logs

"$PY" -m py_compile stream4d/*.py tools/*.py evaluation/*.py tests/*.py \
  > outputs/audit/v14_logs/py_compile_final.log 2>&1
"$PY" -m unittest discover tests -p '*pure*.py' \
  > outputs/audit/v14_logs/unittest_pure_final.log 2>&1
"$PY" -m unittest discover tests \
  > outputs/audit/v14_logs/unittest_all_final.log 2>&1

CONFIGS=$(paste -sd, outputs/audit/v14_final/configs_v14_final.txt)

"$PY" -m tools.scan_reportable_configs \
  --root . \
  --configs "$CONFIGS" \
  --output outputs/audit/v14_final/reportable_config_scan_v14_final.md \
  --require-manifest \
  --require-eval-policy \
  > outputs/audit/v14_logs/v14_reportable_scan_final.log 2>&1

"$PY" -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root . \
  --current-root . \
  --configs "$CONFIGS" \
  --seq-list splits/scannet_v6_probe5.txt \
  --output outputs/audit/v14_final/metric_integrity_v14_final.md \
  --require-manifest \
  > outputs/audit/v14_logs/v14_metric_integrity_final.log 2>&1
