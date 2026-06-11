#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/audit/v9_o32_o31_score_calibration outputs/v9_score_calibration data/evaluation/scannet

run_logged() {
  local log_path="$1"
  shift
  "$@" > "$log_path" 2>&1
}

eval_own() {
  local config="$1"
  run_logged "logs/stream4d_v9_o32_${config}_eval.log" \
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
  run_logged "logs/stream4d_v9_o32_${output_config}.log" \
    "$PY" -m tools.evaluate_cross_prepoints \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --pred-config "$pred_config" \
      --pre-points-config "$pre_points_config" \
      --output-config "$output_config" \
      --dataset scannet \
      --no-class \
      --output-file "data/evaluation/scannet/${output_config}_class_agnostic.txt" \
      --audit-root outputs/audit/v9_o32_o31_score_calibration \
      --require-manifest \
      --allow-diagnostic-eval \
      --eval-policy "$eval_policy"
}

rescore_eval() {
  local output_config="$1"
  local feature="$2"
  local policy="$3"
  run_logged "logs/stream4d_v9_o32_${output_config}_rescore.log" \
    "$PY" -m tools.rescore_prediction_scores \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --input-config stream4d_v9_o31_o22_memory_c065_noexcl_probe5 \
      --output-config "$output_config" \
      --score-feature "$feature" \
      --base-score-mode constant \
      --constant-score 1.0 \
      --tiebreaker-weight 1.0 \
      --summary-root outputs/v9_score_calibration \
      --eval-policy "$policy"
  eval_own "$output_config"
  cross_eval "$output_config" scannet "${output_config}_on_s0_probe5" method_on_stream3d_support
  cross_eval "$output_config" stream4d_32f_self_probe5 "${output_config}_on_s1_probe5" method_on_32f_support
}

run_logged logs/stream4d_v9_o32_o31_score_calibration_py_compile.log \
  "$PY" -m py_compile \
    tools/rescore_prediction_scores.py \
    tools/summarize_v9_unified_eval.py \
    tools/evaluate_cross_prepoints.py \
    tools/scan_reportable_configs.py \
    tools/verify_stream4d_metric_integrity.py

rescore_eval stream4d_v9_o32_o31_noexcl_logarea_probe5 log_area own_score_calibration_o31_noexcl_logarea
rescore_eval stream4d_v9_o32_o31_noexcl_invlogarea_probe5 inverse_log_area own_score_calibration_o31_noexcl_invlogarea

run_logged logs/stream4d_v9_o32_o31_score_calibration_matrix_probe5.log \
  "$PY" -m tools.summarize_v9_unified_eval \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --matrix-json scripts/v9_o32_o31_score_calibration_matrix_probe5.json \
    --output-prefix outputs/audit/v9_o32_o31_score_calibration/o32_o31_score_calibration_matrix_probe5 \
    --dataset scannet \
    --stream3d-config scannet

run_logged logs/stream4d_v9_o32_o31_score_calibration_reportable_scan.log \
  "$PY" -m tools.scan_reportable_configs \
    --root . \
    --configs stream4d_v9_o32_o31_noexcl_logarea_probe5,stream4d_v9_o32_o31_noexcl_invlogarea_probe5 \
    --output outputs/audit/v9_o32_o31_score_calibration/reportable_config_scan_o32_probe5.md \
    --require-manifest \
    --require-eval-policy

run_logged logs/stream4d_v9_o32_o31_score_calibration_metric_integrity.log \
  "$PY" -m tools.verify_stream4d_metric_integrity \
    --orig-stream3d-root . \
    --current-root . \
    --configs stream4d_v9_o32_o31_noexcl_logarea_probe5,stream4d_v9_o32_o31_noexcl_invlogarea_probe5 \
    --seq-list splits/scannet_v6_probe5.txt \
    --output outputs/audit/v9_o32_o31_score_calibration/metric_integrity_o32_probe5.md \
    --require-manifest

echo "v9 O32 O31-score-calibration cross-support done"
