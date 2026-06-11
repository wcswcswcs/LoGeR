#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"

mkdir -p outputs/audit/v16_logs outputs/audit/v16_phase2

"$PY" -m tools.diagnose_v12_measurement_bank \
  --bank-root outputs/v14_measurement_bank_bank16_cropformer \
  --seq-list splits/scannet_v6_probe5.txt \
  --output-prefix outputs/audit/v16_phase2/measurement_bank_bank16_probe5 \
  > outputs/audit/v16_logs/v16_phase2_measurement_bank_bank16.log 2>&1
