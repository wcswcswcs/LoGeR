#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/audit/v11_failure_attribution

run_logged() {
  local log_path="$1"
  shift
  "$@" > "$log_path" 2>&1
}

run_attr() {
  local label="$1"
  local pool_name="$2"
  local pool_configs="$3"
  local method_config="$4"
  run_logged "logs/stream4d_v11_failure_attribution_${label}.log" \
    "$PY" -m tools.v11_gt_failure_attribution \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --pool-name "$pool_name" \
      --pool-configs "$pool_configs" \
      --method-config "$method_config" \
      --low-iou 0.25 \
      --high-iou 0.50 \
      --output-prefix "outputs/audit/v11_failure_attribution/${label}"
}

run_logged logs/stream4d_v11_failure_attribution_py_compile.log \
  "$PY" -m py_compile tools/v11_gt_failure_attribution.py

C5_POOL="stream4d_v8_b1_surfacelet_singlemask_probe5,stream4d_v9_o1_b1_core_only_probe5,stream4d_v9_o38_o22_memory_c055_newpts_logarea_probe5,stream4d_v10_r0_fullmask_probe5,stream4d_v10_r1_maskcore_probe5,stream4d_v10_r2_depthsplit_probe5,stream4d_v10_r3_d4rtseed_probe5,stream4d_v10_r4_combined_probe5"
C4_POOL="stream4d_v10_r0b_fullmask_32f_wta_probe5,stream4d_v10_r1b_maskcore_32f_wta_probe5,stream4d_v10_r4b_combined_32f_wta_probe5"

run_attr b1_vs_c5 C5_c1_c2_c3_union "$C5_POOL" stream4d_v8_b1_surfacelet_singlemask_probe5
run_attr o38_vs_c5 C5_c1_c2_c3_union "$C5_POOL" stream4d_v9_o38_o22_memory_c055_newpts_logarea_probe5
run_attr r1b_vs_c4 C4_regionlet_repair "$C4_POOL" stream4d_v10_r1b_maskcore_32f_wta_probe5
run_attr s3_vs_c5 C5_c1_c2_c3_union "$C5_POOL" stream4d_v11_s3_d4rt_maskcount_probe5

echo "v11 failure attribution probe5 done"
