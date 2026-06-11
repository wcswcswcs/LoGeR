#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/audit/v9_o31_memory_exclusive outputs/v9_scene_object_memory data/evaluation/scannet

run_logged() {
  local log_path="$1"
  shift
  "$@" > "$log_path" 2>&1
}

eval_own() {
  local config="$1"
  run_logged "logs/stream4d_v9_o31_${config}_eval.log" \
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
  run_logged "logs/stream4d_v9_o31_${output_config}.log" \
    "$PY" -m tools.evaluate_cross_prepoints \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --pred-config "$pred_config" \
      --pre-points-config "$pre_points_config" \
      --output-config "$output_config" \
      --dataset scannet \
      --no-class \
      --output-file "data/evaluation/scannet/${output_config}_class_agnostic.txt" \
      --audit-root outputs/audit/v9_o31_memory_exclusive \
      --require-manifest \
      --allow-diagnostic-eval \
      --eval-policy "$eval_policy"
}

memory_eval() {
  local output_config="$1"
  local exclusive_mode="$2"
  local policy="$3"
  run_logged "logs/stream4d_v9_o31_${output_config}_export.log" \
    "$PY" -m tools.scene_object_memory_from_predictions \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --input-config stream4d_v9_o22_b1_fullspan_grid8_ws25_probe5 \
      --output-config "$output_config" \
      --order-mode score_area \
      --attach-candidate-ioc 0.65 \
      --attach-slot-ioc 0.45 \
      --attach-iou 0.18 \
      --birth-max-overlap 0.18 \
      --ambiguous-candidate-ioc 0.60 \
      --ambiguous-slot-ioc 0.60 \
      --ambiguous-iou 0.30 \
      --reject-ambiguous \
      --exclusive-mode "$exclusive_mode" \
      --summary-root outputs/v9_scene_object_memory \
      --eval-policy "$policy"
  eval_own "$output_config"
  cross_eval "$output_config" scannet "${output_config}_on_s0_probe5" method_on_stream3d_support
  cross_eval "$output_config" stream4d_32f_self_probe5 "${output_config}_on_s1_probe5" method_on_32f_support
}

run_logged logs/stream4d_v9_o31_memory_exclusive_py_compile.log \
  "$PY" -m py_compile \
    tools/scene_object_memory_from_predictions.py \
    tools/summarize_v9_unified_eval.py \
    tools/evaluate_cross_prepoints.py \
    tools/scan_reportable_configs.py \
    tools/verify_stream4d_metric_integrity.py

memory_eval stream4d_v9_o31_o22_memory_c065_noexcl_probe5 none own_scene_object_memory_c065_noexclusive
memory_eval stream4d_v9_o31_o22_memory_c065_smallarea_probe5 small_area own_scene_object_memory_c065_smallarea_exclusive

run_logged logs/stream4d_v9_o31_memory_exclusive_matrix_probe5.log \
  "$PY" -m tools.summarize_v9_unified_eval \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --matrix-json scripts/v9_o31_memory_exclusive_matrix_probe5.json \
    --output-prefix outputs/audit/v9_o31_memory_exclusive/o31_memory_exclusive_matrix_probe5 \
    --dataset scannet \
    --stream3d-config scannet

run_logged logs/stream4d_v9_o31_memory_exclusive_reportable_scan.log \
  "$PY" -m tools.scan_reportable_configs \
    --root . \
    --configs stream4d_v9_o31_o22_memory_c065_noexcl_probe5,stream4d_v9_o31_o22_memory_c065_smallarea_probe5 \
    --output outputs/audit/v9_o31_memory_exclusive/reportable_config_scan_o31_probe5.md \
    --require-manifest \
    --require-eval-policy

run_logged logs/stream4d_v9_o31_memory_exclusive_metric_integrity.log \
  "$PY" -m tools.verify_stream4d_metric_integrity \
    --orig-stream3d-root . \
    --current-root . \
    --configs stream4d_v9_o31_o22_memory_c065_noexcl_probe5,stream4d_v9_o31_o22_memory_c065_smallarea_probe5 \
    --seq-list splits/scannet_v6_probe5.txt \
    --output outputs/audit/v9_o31_memory_exclusive/metric_integrity_o31_probe5.md \
    --require-manifest

echo "v9 O31 memory-exclusive cross-support done"
