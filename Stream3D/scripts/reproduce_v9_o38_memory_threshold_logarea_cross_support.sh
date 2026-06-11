#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/audit/v9_o38_memory_threshold_logarea outputs/v9_scene_object_memory outputs/v9_score_calibration data/evaluation/scannet

run_logged() {
  local log_path="$1"
  shift
  "$@" > "$log_path" 2>&1
}

eval_own() {
  local config="$1"
  run_logged "logs/stream4d_v9_o38_${config}_eval.log" \
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
  run_logged "logs/stream4d_v9_o38_${output_config}.log" \
    "$PY" -m tools.evaluate_cross_prepoints \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --pred-config "$pred_config" \
      --pre-points-config "$pre_points_config" \
      --output-config "$output_config" \
      --dataset scannet \
      --no-class \
      --output-file "data/evaluation/scannet/${output_config}_class_agnostic.txt" \
      --audit-root outputs/audit/v9_o38_memory_threshold_logarea \
      --require-manifest \
      --allow-diagnostic-eval \
      --eval-policy "$eval_policy"
}

memory_then_logarea_eval() {
  local raw_config="$1"
  local scored_config="$2"
  local policy="$3"
  local attach_cioc="$4"
  local attach_sioc="$5"
  local attach_iou="$6"
  local birth_overlap="$7"
  local amb_cioc="$8"
  local amb_sioc="$9"
  local amb_iou="${10}"

  run_logged "logs/stream4d_v9_o38_${raw_config}_export.log" \
    "$PY" -m tools.scene_object_memory_from_predictions \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --input-config stream4d_v9_o22_b1_fullspan_grid8_ws25_probe5 \
      --output-config "$raw_config" \
      --order-mode score_area \
      --attach-candidate-ioc "$attach_cioc" \
      --attach-slot-ioc "$attach_sioc" \
      --attach-iou "$attach_iou" \
      --birth-max-overlap "$birth_overlap" \
      --ambiguous-candidate-ioc "$amb_cioc" \
      --ambiguous-slot-ioc "$amb_sioc" \
      --ambiguous-iou "$amb_iou" \
      --reject-ambiguous \
      --update-mode new_points_only \
      --exclusive-mode none \
      --summary-root outputs/v9_scene_object_memory \
      --eval-policy "${policy}_raw"

  run_logged "logs/stream4d_v9_o38_${scored_config}_rescore.log" \
    "$PY" -m tools.rescore_prediction_scores \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --input-config "$raw_config" \
      --output-config "$scored_config" \
      --score-feature log_area \
      --base-score-mode constant \
      --constant-score 1.0 \
      --tiebreaker-weight 1.0 \
      --summary-root outputs/v9_score_calibration \
      --eval-policy "$policy"

  eval_own "$scored_config"
  cross_eval "$scored_config" scannet "${scored_config}_on_s0_probe5" method_on_stream3d_support
  cross_eval "$scored_config" stream4d_32f_self_probe5 "${scored_config}_on_s1_probe5" method_on_32f_support
}

run_logged logs/stream4d_v9_o38_memory_threshold_logarea_py_compile.log \
  "$PY" -m py_compile \
    tools/scene_object_memory_from_predictions.py \
    tools/rescore_prediction_scores.py \
    tools/summarize_v9_unified_eval.py \
    tools/evaluate_cross_prepoints.py \
    tools/scan_reportable_configs.py \
    tools/verify_stream4d_metric_integrity.py

memory_then_logarea_eval \
  stream4d_v9_o38_o22_memory_c055_newpts_probe5 \
  stream4d_v9_o38_o22_memory_c055_newpts_logarea_probe5 \
  own_memory_threshold_c055_newpts_logarea \
  0.55 0.35 0.14 0.14 0.50 0.50 0.25

memory_then_logarea_eval \
  stream4d_v9_o38_o22_memory_c075split_newpts_probe5 \
  stream4d_v9_o38_o22_memory_c075split_newpts_logarea_probe5 \
  own_memory_threshold_c075split_newpts_logarea \
  0.75 0.55 0.25 0.30 0.70 0.70 0.40

run_logged logs/stream4d_v9_o38_memory_threshold_logarea_matrix_probe5.log \
  "$PY" -m tools.summarize_v9_unified_eval \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --matrix-json scripts/v9_o38_memory_threshold_logarea_matrix_probe5.json \
    --output-prefix outputs/audit/v9_o38_memory_threshold_logarea/o38_memory_threshold_logarea_matrix_probe5 \
    --dataset scannet \
    --stream3d-config scannet

run_logged logs/stream4d_v9_o38_memory_threshold_logarea_reportable_scan.log \
  "$PY" -m tools.scan_reportable_configs \
    --root . \
    --configs stream4d_v9_o38_o22_memory_c055_newpts_logarea_probe5,stream4d_v9_o38_o22_memory_c075split_newpts_logarea_probe5 \
    --output outputs/audit/v9_o38_memory_threshold_logarea/reportable_config_scan_o38_probe5.md \
    --require-manifest \
    --require-eval-policy

run_logged logs/stream4d_v9_o38_memory_threshold_logarea_metric_integrity.log \
  "$PY" -m tools.verify_stream4d_metric_integrity \
    --orig-stream3d-root . \
    --current-root . \
    --configs stream4d_v9_o38_o22_memory_c055_newpts_logarea_probe5,stream4d_v9_o38_o22_memory_c075split_newpts_logarea_probe5 \
    --seq-list splits/scannet_v6_probe5.txt \
    --output outputs/audit/v9_o38_memory_threshold_logarea/metric_integrity_o38_probe5.md \
    --require-manifest

echo "v9 O38 memory threshold logarea cross-support done"
