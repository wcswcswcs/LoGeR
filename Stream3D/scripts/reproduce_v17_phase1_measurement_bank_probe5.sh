#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"
mkdir -p outputs/audit/v17_logs outputs/audit/v17_phase1

"$PY" -m tools.diagnose_measurement_bank_v17 \
  --bank-root outputs/v14_measurement_bank_bank16_cropformer \
  --seq-list splits/scannet_v6_probe5.txt \
  --output-prefix outputs/audit/v17_phase1/measurement_bank_fixed_probe5 \
  > outputs/audit/v17_logs/v17_phase1_measurement_bank_fixed.log 2>&1
