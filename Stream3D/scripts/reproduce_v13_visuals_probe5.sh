#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/audit/v13_visuals

"$PY" -m py_compile tools/make_v13_failure_visuals.py > logs/stream4d_v13_visuals_py_compile.log 2>&1
"$PY" -m tools.make_v13_failure_visuals \
  --matrix-json outputs/audit/v13_candidate_attribution/candidate_unsup_matrix_probe5.json \
  --matrix-json outputs/audit/v13_candidate_attribution/candidate_oracle_matrix_probe5.json \
  --matrix-json outputs/audit/v13_object_explanation_mdl/object_mdl_matrix_probe5.json \
  --output-dir outputs/audit/v13_visuals \
  --limit 20 > logs/stream4d_v13_visuals.log 2>&1

echo "v13 visuals probe5 done"
