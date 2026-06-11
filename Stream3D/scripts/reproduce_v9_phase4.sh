#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/audit/v9_phase4 outputs/v9_core_fringe data/evaluation/scannet

run_logged() {
  local log_path="$1"
  shift
  "$@" > "$log_path" 2>&1
}

eval_own() {
  local config="$1"
  run_logged "logs/${config}_eval.log" \
    "$PY" -m evaluation.evaluate \
      --pred_path "data/prediction/${config}_class_agnostic" \
      --gt_path data/scannet/gt \
      --dataset scannet \
      --output_file "data/evaluation/scannet/${config}_class_agnostic.txt" \
      --tmp_root data/TMP \
      --tmp_config "$config" \
      --no_class \
      --require-manifest
}

cross_eval() {
  local pred_config="$1"
  local pre_points_config="$2"
  local output_config="$3"
  local eval_policy="$4"
  run_logged "logs/${output_config}.log" \
    "$PY" -m tools.evaluate_cross_prepoints \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --pred-config "$pred_config" \
      --pre-points-config "$pre_points_config" \
      --output-config "$output_config" \
      --dataset scannet \
      --no-class \
      --output-file "data/evaluation/scannet/${output_config}_class_agnostic.txt" \
      --audit-root outputs/audit/v9_phase4 \
      --require-manifest \
      --allow-diagnostic-eval \
      --eval-policy "$eval_policy"
}

run_logged logs/stream4d_v9_phase4_py_compile.log \
  "$PY" -m py_compile tools/split_core_fringe_prediction.py tools/summarize_v9_unified_eval.py

run_logged logs/stream4d_v9_o1_core_only_export.log \
  "$PY" -m tools.split_core_fringe_prediction \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --input-config stream4d_v8_b1_surfacelet_singlemask_probe5 \
    --support-config stream4d_v8_b1_surfacelet_singlemask_probe5 \
    --output-config stream4d_v9_o1_b1_core_only_probe5 \
    --assignment-mode low_conflict \
    --max-core-owners 1 \
    --low-mode none \
    --min-core-points 10 \
    --min-core-ratio 0.05 \
    --tmp-policy recompute \
    --eval-policy own_recompute_core_only \
    --summary-root outputs/v9_core_fringe

run_logged logs/stream4d_v9_o2_core_radius_fringe_export.log \
  "$PY" -m tools.split_core_fringe_prediction \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --input-config stream4d_v8_b1_surfacelet_singlemask_probe5 \
    --support-config stream4d_v8_b1_surfacelet_singlemask_probe5 \
    --output-config stream4d_v9_o2_b1_core_radius_fringe_probe5 \
    --assignment-mode low_conflict \
    --max-core-owners 1 \
    --low-mode none \
    --growth-mode radius \
    --growth-candidate-mode full \
    --growth-radius 0.05 \
    --growth-max-owners 1 \
    --min-core-points 10 \
    --min-core-ratio 0.05 \
    --tmp-policy recompute \
    --eval-policy own_recompute_core_radius_fringe \
    --summary-root outputs/v9_core_fringe

run_logged logs/stream4d_v9_o3_wta_negative_fringe_export.log \
  "$PY" -m tools.split_core_fringe_prediction \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --input-config stream4d_v8_b1_surfacelet_singlemask_probe5 \
    --support-config stream4d_v8_b1_surfacelet_singlemask_probe5 \
    --output-config stream4d_v9_o3_b1_wta_negative_fringe_probe5 \
    --assignment-mode wta \
    --wta-priority score_over_sqrt_area \
    --low-mode none \
    --growth-mode radius \
    --growth-candidate-mode full \
    --growth-radius 0.05 \
    --growth-max-owners 1 \
    --min-core-points 10 \
    --min-core-ratio 0.05 \
    --tmp-policy recompute \
    --eval-policy own_recompute_wta_negative_fringe \
    --summary-root outputs/v9_core_fringe

for config in \
  stream4d_v9_o1_b1_core_only_probe5 \
  stream4d_v9_o2_b1_core_radius_fringe_probe5 \
  stream4d_v9_o3_b1_wta_negative_fringe_probe5; do
  eval_own "$config"
done

cross_eval scannet stream4d_v9_o1_b1_core_only_probe5 stream4d_v9_p0_on_o1_core_support_probe5 stream3d_on_o1_support
cross_eval stream4d_v9_o1_b1_core_only_probe5 scannet stream4d_v9_o1_core_on_s0_scannet_probe5 method_on_stream3d_support
cross_eval stream4d_v9_o1_b1_core_only_probe5 stream4d_32f_self_probe5 stream4d_v9_o1_core_on_s1_32f_probe5 method_on_32f_support

cross_eval scannet stream4d_v9_o2_b1_core_radius_fringe_probe5 stream4d_v9_p0_on_o2_fringe_support_probe5 stream3d_on_o2_support
cross_eval stream4d_v9_o2_b1_core_radius_fringe_probe5 scannet stream4d_v9_o2_fringe_on_s0_scannet_probe5 method_on_stream3d_support
cross_eval stream4d_v9_o2_b1_core_radius_fringe_probe5 stream4d_32f_self_probe5 stream4d_v9_o2_fringe_on_s1_32f_probe5 method_on_32f_support

cross_eval scannet stream4d_v9_o3_b1_wta_negative_fringe_probe5 stream4d_v9_p0_on_o3_negative_support_probe5 stream3d_on_o3_support
cross_eval stream4d_v9_o3_b1_wta_negative_fringe_probe5 scannet stream4d_v9_o3_negative_on_s0_scannet_probe5 method_on_stream3d_support
cross_eval stream4d_v9_o3_b1_wta_negative_fringe_probe5 stream4d_32f_self_probe5 stream4d_v9_o3_negative_on_s1_32f_probe5 method_on_32f_support

run_logged logs/stream4d_v9_phase4_matrix_probe5.log \
  "$PY" -m tools.summarize_v9_unified_eval \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --matrix-json scripts/v9_phase4_matrix_probe5.json \
    --output-prefix outputs/audit/v9_phase4/phase4_matrix_probe5 \
    --dataset scannet \
    --stream3d-config scannet

run_logged logs/stream4d_v9_phase4_reportable_scan.log \
  "$PY" -m tools.scan_reportable_configs \
    --root . \
    --configs stream4d_v9_o1_b1_core_only_probe5,stream4d_v9_o2_b1_core_radius_fringe_probe5,stream4d_v9_o3_b1_wta_negative_fringe_probe5 \
    --output outputs/audit/v9_phase4/reportable_config_scan_phase4_probe5.md \
    --require-manifest \
    --require-eval-policy

echo "v9 Phase4 done"
