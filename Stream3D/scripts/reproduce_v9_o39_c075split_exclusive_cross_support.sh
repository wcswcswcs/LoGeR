#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/audit/v9_o39_c075split_exclusive outputs/v9_scene_object_memory outputs/v9_score_calibration data/evaluation/scannet

run_logged() {
  local log_path="$1"
  shift
  "$@" > "$log_path" 2>&1
}

eval_own() {
  local config="$1"
  run_logged "logs/stream4d_v9_o39_${config}_eval.log" \
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
  run_logged "logs/stream4d_v9_o39_${output_config}.log" \
    "$PY" -m tools.evaluate_cross_prepoints \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --pred-config "$pred_config" \
      --pre-points-config "$pre_points_config" \
      --output-config "$output_config" \
      --dataset scannet \
      --no-class \
      --output-file "data/evaluation/scannet/${output_config}_class_agnostic.txt" \
      --audit-root outputs/audit/v9_o39_c075split_exclusive \
      --require-manifest \
      --allow-diagnostic-eval \
      --eval-policy "$eval_policy"
}

RAW="stream4d_v9_o39_o22_memory_c075split_smallarea_probe5"
SCORED="stream4d_v9_o39_o22_memory_c075split_smallarea_logarea_probe5"

run_logged logs/stream4d_v9_o39_c075split_exclusive_py_compile.log \
  "$PY" -m py_compile \
    tools/scene_object_memory_from_predictions.py \
    tools/rescore_prediction_scores.py \
    tools/summarize_v9_unified_eval.py \
    tools/evaluate_cross_prepoints.py \
    tools/scan_reportable_configs.py \
    tools/verify_stream4d_metric_integrity.py

run_logged "logs/stream4d_v9_o39_${RAW}_export.log" \
  "$PY" -m tools.scene_object_memory_from_predictions \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --input-config stream4d_v9_o22_b1_fullspan_grid8_ws25_probe5 \
    --output-config "$RAW" \
    --order-mode score_area \
    --attach-candidate-ioc 0.75 \
    --attach-slot-ioc 0.55 \
    --attach-iou 0.25 \
    --birth-max-overlap 0.30 \
    --ambiguous-candidate-ioc 0.70 \
    --ambiguous-slot-ioc 0.70 \
    --ambiguous-iou 0.40 \
    --reject-ambiguous \
    --update-mode new_points_only \
    --exclusive-mode small_area \
    --summary-root outputs/v9_scene_object_memory \
    --eval-policy own_memory_c075split_smallarea_raw

run_logged "logs/stream4d_v9_o39_${SCORED}_rescore.log" \
  "$PY" -m tools.rescore_prediction_scores \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --input-config "$RAW" \
    --output-config "$SCORED" \
    --score-feature log_area \
    --base-score-mode constant \
    --constant-score 1.0 \
    --tiebreaker-weight 1.0 \
    --summary-root outputs/v9_score_calibration \
    --eval-policy own_memory_c075split_smallarea_logarea

eval_own "$SCORED"
cross_eval "$SCORED" scannet "${SCORED}_on_s0_probe5" method_on_stream3d_support
cross_eval "$SCORED" stream4d_32f_self_probe5 "${SCORED}_on_s1_probe5" method_on_32f_support

run_logged logs/stream4d_v9_o39_c075split_exclusive_matrix_probe5.log \
  "$PY" -m tools.summarize_v9_unified_eval \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --matrix-json scripts/v9_o39_c075split_exclusive_matrix_probe5.json \
    --output-prefix outputs/audit/v9_o39_c075split_exclusive/o39_c075split_exclusive_matrix_probe5 \
    --dataset scannet \
    --stream3d-config scannet

run_logged logs/stream4d_v9_o39_c075split_exclusive_reportable_scan.log \
  "$PY" -m tools.scan_reportable_configs \
    --root . \
    --configs "$SCORED" \
    --output outputs/audit/v9_o39_c075split_exclusive/reportable_config_scan_o39_probe5.md \
    --require-manifest \
    --require-eval-policy

run_logged logs/stream4d_v9_o39_c075split_exclusive_metric_integrity.log \
  "$PY" -m tools.verify_stream4d_metric_integrity \
    --orig-stream3d-root . \
    --current-root . \
    --configs "$SCORED" \
    --seq-list splits/scannet_v6_probe5.txt \
    --output outputs/audit/v9_o39_c075split_exclusive/metric_integrity_o39_probe5.md \
    --require-manifest

echo "v9 O39 c075split-exclusive cross-support done"
