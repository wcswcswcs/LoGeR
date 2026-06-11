#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"

mkdir -p outputs/audit/v14_logs outputs/audit/v14_phase0

"$PY" -m py_compile stream4d/*.py tools/*.py evaluation/*.py tests/*.py \
  > outputs/audit/v14_logs/py_compile_reproduce_phase0.log 2>&1
"$PY" -m unittest discover tests -p '*pure*.py' \
  > outputs/audit/v14_logs/unittest_pure_reproduce_phase0.log 2>&1
"$PY" -m unittest discover tests \
  > outputs/audit/v14_logs/unittest_all_reproduce_phase0.log 2>&1

"$PY" -m tools.summarize_v9_unified_eval \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --matrix-json scripts/v14_phase0_baseline_matrix_probe5.json \
  --output-prefix outputs/audit/v14_phase0/baseline_matrix_probe5 \
  --stream3d-config scannet

CONFIGS=$(paste -sd, outputs/audit/v14_final/configs_v14_final.txt 2>/dev/null || true)
if [[ -z "${CONFIGS}" ]]; then
  CONFIGS=$(python3 - <<'PY'
import json
rows = json.load(open('outputs/audit/v14_phase0/baseline_matrix_probe5.json'))['rows']
print(','.join(row['output_config'] for row in rows))
PY
)
fi

"$PY" -m tools.scan_reportable_configs \
  --root . \
  --configs "$CONFIGS" \
  --output outputs/audit/v14_phase0/reportable_config_scan_v14_phase0.md \
  --require-manifest \
  --require-eval-policy

"$PY" -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root . \
  --current-root . \
  --configs "$CONFIGS" \
  --seq-list splits/scannet_v6_probe5.txt \
  --output outputs/audit/v14_phase0/metric_integrity_v14_phase0.md \
  --require-manifest
