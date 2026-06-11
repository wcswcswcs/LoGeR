#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"
mkdir -p outputs/audit/v17_logs outputs/audit/v17_phase0

CONFIGS="${CONFIGS:-stream4d_v17_m17_real_probe5,stream4d_v17_m17_shuffle_probe5,stream4d_v17_m17_no_temporal_probe5,stream4d_v17_m17_no_negative_probe5,stream4d_v17_m17_area_only_probe5,stream4d_v17_m17_random_same_count_probe5}"

"$PY" -m py_compile \
  evaluation/*.py \
  stream4d/*.py \
  tools/*.py \
  tests/*.py \
  > outputs/audit/v17_logs/v17_phase0_py_compile.log 2>&1

"$PY" -m unittest discover tests \
  > outputs/audit/v17_logs/v17_phase0_unittest.log 2>&1

"$PY" -m tools.scan_reportable_configs \
  --root . \
  --configs "$CONFIGS" \
  --require-manifest \
  --require-eval-policy \
  --output outputs/audit/v17_phase0/reportable_config_scan_v17_probe5.md \
  > outputs/audit/v17_logs/v17_phase0_reportable_scan.log 2>&1

"$PY" -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root . \
  --current-root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --configs "$CONFIGS" \
  --output outputs/audit/v17_phase0/metric_integrity_v17_probe5.md \
  --require-manifest \
  > outputs/audit/v17_logs/v17_phase0_metric_integrity.log 2>&1
