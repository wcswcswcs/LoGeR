#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/audit/v12_phase0 data/evaluation/scannet

run_logged() {
  local log_path="$1"
  shift
  "$@" > "$log_path" 2>&1
}

cross_eval() {
  local pred_config="$1"
  local pre_points_config="$2"
  local output_config="$3"
  run_logged "logs/stream4d_v12_phase0_${output_config}.log" \
    "$PY" -m tools.evaluate_cross_prepoints \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --pred-config "$pred_config" \
      --pre-points-config "$pre_points_config" \
      --output-config "$output_config" \
      --dataset scannet \
      --no-class \
      --output-file "data/evaluation/scannet/${output_config}_class_agnostic.txt" \
      --audit-root outputs/audit/v12_phase0 \
      --require-manifest \
      --allow-diagnostic-eval \
      --eval-policy cross_fixed_support
}

run_logged logs/stream4d_v12_phase0_py_compile.log \
  "$PY" -m py_compile tools/evaluate_cross_prepoints.py tools/summarize_v10_unified_eval.py evaluation/evaluate.py

run_logged logs/stream4d_v12_phase0_update_v6_manifest.log \
  "$PY" -m tools.update_config_manifest_fields \
    --root . \
    --config stream4d_v6_e4_probe5_objcomp_m650_g101_scoreunique_preserve \
    --eval-policy own_recompute_paper_style \
    --support-source own \
    --geometry-source rgbd_eval_bridge \
    --uses-gt-for-prediction false \
    --uses-gt-for-diagnostic false \
    --is-method-result true \
    --is-diagnostic-only false \
    --reason "v12 Phase0 protocol completion for pre-existing v6 compact artifact"

preds=(
  "p0_stream3d:scannet"
  "p1_v6compact:stream4d_v6_e4_probe5_objcomp_m650_g101_scoreunique_preserve"
  "p2_b1:stream4d_v8_b1_surfacelet_singlemask_probe5"
  "p3_o1:stream4d_v9_o1_b1_core_only_probe5"
  "p4_o38:stream4d_v9_o38_o22_memory_c055_newpts_logarea_probe5"
  "p5_r1b:stream4d_v10_r1b_maskcore_32f_wta_probe5"
)
supports=(
  "s0:scannet"
  "s1:stream4d_32f_self_probe5"
  "s2:stream4d_v8_b1_surfacelet_singlemask_probe5"
  "s3:stream4d_v9_o1_b1_core_only_probe5"
  "s4:stream4d_v9_o38_o22_memory_c055_newpts_logarea_probe5"
  "s5:stream4d_v6_e4_probe5_objcomp_m650_g101_scoreunique_preserve"
)

for pred_item in "${preds[@]}"; do
  pred_label="${pred_item%%:*}"
  pred_config="${pred_item#*:}"
  for support_item in "${supports[@]}"; do
    support_label="${support_item%%:*}"
    support_config="${support_item#*:}"
    cross_eval "$pred_config" "$support_config" "stream4d_v12_phase0_${pred_label}_on_${support_label}_probe5"
  done
done

run_logged logs/stream4d_v12_phase0_matrix.log \
  "$PY" -m tools.summarize_v10_unified_eval \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --matrix-json scripts/v12_phase0_matrix_probe5.json \
    --output-prefix outputs/audit/v12_phase0/unified_eval_matrix_probe5 \
    --plot-dir outputs/audit/v12_phase0 \
    --dataset scannet \
    --stream3d-config scannet

echo "v12 phase0 probe5 done"
