#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
PY="${PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"
LOG_DIR="${LOG_DIR:-outputs/audit/v19_logs}"
SEQ="${SEQ:-splits/scannet_v6_probe5.txt}"
ORIG_STREAM3D_ROOT="${ORIG_STREAM3D_ROOT:-.}"
CONFIGS="${CONFIGS:-stream4d_v19_m0_oracle_a_probe5,stream4d_v19_m0_oracle_b_probe5,stream4d_v19_m0_oracle_c_probe5,stream4d_v19_m1_oracle_a_probe5,stream4d_v19_m1_oracle_b_probe5,stream4d_v19_m1_oracle_c_probe5,stream4d_v19_m2_oracle_a_probe5,stream4d_v19_m2_oracle_b_probe5,stream4d_v19_m2_oracle_c_probe5,stream4d_v19_m3_oracle_a_probe5,stream4d_v19_m3_oracle_b_probe5,stream4d_v19_m3_oracle_c_probe5,stream4d_v19_nn008_m1_oracle_a_probe5,stream4d_v19_nn008_m1_oracle_b_probe5,stream4d_v19_nn008_m1_oracle_c_probe5,stream4d_v19_grid48_m1_oracle_a_probe5,stream4d_v19_grid48_m1_oracle_b_probe5,stream4d_v19_grid48_m1_oracle_c_probe5,stream4d_v19_m2r008_m2_oracle_a_probe5,stream4d_v19_m2r008_m2_oracle_b_probe5,stream4d_v19_m2r008_m2_oracle_c_probe5}"
mkdir -p "$LOG_DIR"

"$PY" -m py_compile evaluation/*.py stream4d/*.py tools/*.py tests/*.py \
  2>&1 | tee "$LOG_DIR/audit_py_compile.log"

"$PY" -m unittest discover tests \
  2>&1 | tee "$LOG_DIR/audit_unittest_discover.log"

"$PY" -m tools.scan_reportable_configs \
  --configs "$CONFIGS" \
  --output outputs/audit/v19_final/oracle_reportable_scan.md \
  --require-manifest \
  --require-eval-policy \
  2>&1 | tee "$LOG_DIR/audit_scan_reportable_configs.log"

"$PY" -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root "$ORIG_STREAM3D_ROOT" \
  --current-root . \
  --configs "$CONFIGS" \
  --seq-list "$SEQ" \
  --output outputs/audit/v19_final/metric_integrity.md \
  --require-manifest \
  2>&1 | tee "$LOG_DIR/audit_metric_integrity.log"
