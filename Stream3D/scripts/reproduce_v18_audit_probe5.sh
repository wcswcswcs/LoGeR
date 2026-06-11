#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
PY="${PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"
LOG_DIR="${LOG_DIR:-outputs/audit/v18_logs}"
mkdir -p "$LOG_DIR"

"$PY" -m py_compile \
  stream4d/signed_surfel_graph.py \
  stream4d/signed_boundary_evidence.py \
  stream4d/signed_graph_partition.py \
  tools/build_v18_signed_surfel_graph.py \
  tools/build_v18_signed_boundary_evidence.py \
  tools/diagnose_v18_edge_boundary_quality.py \
  tools/diagnose_v18_partition_quality.py \
  tools/export_v18_signed_graph_partition.py \
  tools/summarize_v18_unified_eval_matrix.py \
  tests/test_v18_signed_boundary_graph.py \
  2>&1 | tee "$LOG_DIR/audit_py_compile.log"

"$PY" -m unittest tests.test_v18_signed_boundary_graph \
  2>&1 | tee "$LOG_DIR/audit_unittest_v18.log"
