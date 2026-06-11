#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/audit/v9_o23_o22_merge_threshold outputs/v9_mask_merge outputs/v9_score_calibration data/evaluation/scannet

INPUT="stream4d_v9_o22_b1_fullspan_grid8_ws25_probe5"

run_logged() {
  local log_path="$1"
  shift
  "$@" > "$log_path" 2>&1
}

eval_own() {
  local config="$1"
  run_logged "logs/stream4d_v9_o23_${config}_eval.log" \
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
  run_logged "logs/stream4d_v9_o23_${output_config}.log" \
    "$PY" -m tools.evaluate_cross_prepoints \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --pred-config "$pred_config" \
      --pre-points-config "$pre_points_config" \
      --output-config "$output_config" \
      --dataset scannet \
      --no-class \
      --output-file "data/evaluation/scannet/${output_config}_class_agnostic.txt" \
      --audit-root outputs/audit/v9_o23_o22_merge_threshold \
      --require-manifest \
      --allow-diagnostic-eval \
      --eval-policy "$eval_policy"
}

run_one() {
  local tag="$1"
  local threshold="$2"
  local merge_config="stream4d_v9_o23_o22_ws25_merge_${tag}_probe5"
  local score_config="stream4d_v9_o23_o22_ws25_merge_${tag}_logarea_probe5"

  run_logged "logs/stream4d_v9_o23_o22_ws25_merge_${tag}_export.log" \
    "$PY" -m tools.merge_overlapping_prediction_masks \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --input-config "$INPUT" \
      --output-config "$merge_config" \
      --min-ioc "$threshold" \
      --min-area 20 \
      --min-output-area 20 \
      --summary-root outputs/v9_mask_merge \
      --eval-policy "own_recompute_o23_ws25_merge_${tag}"

  run_logged "logs/stream4d_v9_o23_o22_ws25_merge_${tag}_logarea_rescore.log" \
    "$PY" -m tools.rescore_prediction_scores \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --input-config "$merge_config" \
      --output-config "$score_config" \
      --score-feature log_area \
      --base-score-mode constant \
      --constant-score 1.0 \
      --tiebreaker-weight 1.0 \
      --summary-root outputs/v9_score_calibration \
      --eval-policy "own_recompute_score_calibration_o23_ws25_${tag}_logarea"

  eval_own "$score_config"
  cross_eval "$score_config" scannet "stream4d_v9_o23_o22_ws25_merge_${tag}_logarea_on_s0_probe5" method_on_stream3d_support
  cross_eval "$score_config" stream4d_32f_self_probe5 "stream4d_v9_o23_o22_ws25_merge_${tag}_logarea_on_s1_probe5" method_on_32f_support
}

run_logged logs/stream4d_v9_o23_o22_merge_threshold_py_compile.log \
  "$PY" -m py_compile \
    tools/merge_overlapping_prediction_masks.py \
    tools/rescore_prediction_scores.py \
    tools/summarize_v9_unified_eval.py \
    tools/evaluate_cross_prepoints.py \
    tools/scan_reportable_configs.py \
    tools/verify_stream4d_metric_integrity.py

run_one mioc030 0.30
run_one mioc040 0.40
run_one mioc060 0.60

run_logged logs/stream4d_v9_o23_o22_merge_threshold_matrix_probe5.log \
  "$PY" -m tools.summarize_v9_unified_eval \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --matrix-json scripts/v9_o23_o22_merge_threshold_matrix_probe5.json \
    --output-prefix outputs/audit/v9_o23_o22_merge_threshold/o23_o22_merge_threshold_matrix_probe5 \
    --dataset scannet \
    --stream3d-config scannet

run_logged logs/stream4d_v9_o23_o22_merge_threshold_reportable_scan.log \
  "$PY" -m tools.scan_reportable_configs \
    --root . \
    --configs stream4d_v9_o23_o22_ws25_merge_mioc030_logarea_probe5,stream4d_v9_o23_o22_ws25_merge_mioc040_logarea_probe5,stream4d_v9_o23_o22_ws25_merge_mioc060_logarea_probe5 \
    --output outputs/audit/v9_o23_o22_merge_threshold/reportable_config_scan_o23_probe5.md \
    --require-manifest \
    --require-eval-policy

run_logged logs/stream4d_v9_o23_o22_merge_threshold_metric_integrity.log \
  "$PY" -m tools.verify_stream4d_metric_integrity \
    --orig-stream3d-root . \
    --current-root . \
    --configs stream4d_v9_o23_o22_ws25_merge_mioc030_logarea_probe5,stream4d_v9_o23_o22_ws25_merge_mioc040_logarea_probe5,stream4d_v9_o23_o22_ws25_merge_mioc060_logarea_probe5 \
    --seq-list splits/scannet_v6_probe5.txt \
    --output outputs/audit/v9_o23_o22_merge_threshold/metric_integrity_o23_probe5.md \
    --require-manifest

echo "v9 O23 O22 merge-threshold cross-support done"
