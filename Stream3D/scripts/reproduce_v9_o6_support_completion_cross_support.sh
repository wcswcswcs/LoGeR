#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/audit/v9_o6_support_completion outputs/v9_support_completion data/evaluation/scannet

run_logged() {
  local log_path="$1"
  shift
  "$@" > "$log_path" 2>&1
}

eval_own() {
  local config="$1"
  run_logged "logs/stream4d_v9_o6_${config}_eval.log" \
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
  run_logged "logs/stream4d_v9_o6_${output_config}.log" \
    "$PY" -m tools.evaluate_cross_prepoints \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --pred-config "$pred_config" \
      --pre-points-config "$pre_points_config" \
      --output-config "$output_config" \
      --dataset scannet \
      --no-class \
      --output-file "data/evaluation/scannet/${output_config}_class_agnostic.txt" \
      --audit-root outputs/audit/v9_o6_support_completion \
      --require-manifest \
      --allow-diagnostic-eval \
      --eval-policy "$eval_policy"
}

export_o6() {
  local tag="$1"
  local radius="$2"
  local policy="$3"
  run_logged "logs/stream4d_v9_o6_o1_complete_s0_${tag}_export.log" \
    "$PY" -m tools.complete_prediction_to_support \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --input-config stream4d_v9_o1_b1_core_only_probe5 \
      --target-support-config scannet \
      --output-config "stream4d_v9_o6_o1_complete_s0_${tag}_probe5" \
      --max-radius "$radius" \
      --min-core-points 20 \
      --min-points-per-object 20 \
      --keep-core-points \
      --summary-root outputs/v9_support_completion \
      --eval-policy "$policy"
}

run_logged logs/stream4d_v9_o6_support_completion_py_compile.log \
  "$PY" -m py_compile \
    tools/complete_prediction_to_support.py \
    tools/summarize_v9_unified_eval.py \
    tools/evaluate_cross_prepoints.py \
    tools/scan_reportable_configs.py \
    tools/verify_stream4d_metric_integrity.py

export_o6 r002 0.02 own_recompute_support_completion_s0_r002
export_o6 r005 0.05 own_recompute_support_completion_s0_r005
export_o6 r010 0.10 own_recompute_support_completion_s0_r010
export_o6 all -1.0 own_recompute_support_completion_s0_all

for config in \
  stream4d_v9_o6_o1_complete_s0_r002_probe5 \
  stream4d_v9_o6_o1_complete_s0_r005_probe5 \
  stream4d_v9_o6_o1_complete_s0_r010_probe5 \
  stream4d_v9_o6_o1_complete_s0_all_probe5; do
  eval_own "$config"
done

cross_eval scannet stream4d_v9_o6_o1_complete_s0_r002_probe5 stream4d_v9_p0_on_o6_o1_complete_s0_r002_probe5 stream3d_on_o6_support_completion_s0_r002
cross_eval stream4d_v9_o6_o1_complete_s0_r002_probe5 scannet stream4d_v9_o6_o1_complete_s0_r002_on_s0_probe5 method_on_stream3d_support
cross_eval stream4d_v9_o6_o1_complete_s0_r002_probe5 stream4d_32f_self_probe5 stream4d_v9_o6_o1_complete_s0_r002_on_s1_probe5 method_on_32f_support

cross_eval scannet stream4d_v9_o6_o1_complete_s0_r005_probe5 stream4d_v9_p0_on_o6_o1_complete_s0_r005_probe5 stream3d_on_o6_support_completion_s0_r005
cross_eval stream4d_v9_o6_o1_complete_s0_r005_probe5 scannet stream4d_v9_o6_o1_complete_s0_r005_on_s0_probe5 method_on_stream3d_support
cross_eval stream4d_v9_o6_o1_complete_s0_r005_probe5 stream4d_32f_self_probe5 stream4d_v9_o6_o1_complete_s0_r005_on_s1_probe5 method_on_32f_support

cross_eval scannet stream4d_v9_o6_o1_complete_s0_r010_probe5 stream4d_v9_p0_on_o6_o1_complete_s0_r010_probe5 stream3d_on_o6_support_completion_s0_r010
cross_eval stream4d_v9_o6_o1_complete_s0_r010_probe5 scannet stream4d_v9_o6_o1_complete_s0_r010_on_s0_probe5 method_on_stream3d_support
cross_eval stream4d_v9_o6_o1_complete_s0_r010_probe5 stream4d_32f_self_probe5 stream4d_v9_o6_o1_complete_s0_r010_on_s1_probe5 method_on_32f_support

cross_eval scannet stream4d_v9_o6_o1_complete_s0_all_probe5 stream4d_v9_p0_on_o6_o1_complete_s0_all_probe5 stream3d_on_o6_support_completion_s0_all
cross_eval stream4d_v9_o6_o1_complete_s0_all_probe5 scannet stream4d_v9_o6_o1_complete_s0_all_on_s0_probe5 method_on_stream3d_support
cross_eval stream4d_v9_o6_o1_complete_s0_all_probe5 stream4d_32f_self_probe5 stream4d_v9_o6_o1_complete_s0_all_on_s1_probe5 method_on_32f_support

run_logged logs/stream4d_v9_o6_support_completion_matrix_probe5.log \
  "$PY" -m tools.summarize_v9_unified_eval \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --matrix-json scripts/v9_o6_support_completion_matrix_probe5.json \
    --output-prefix outputs/audit/v9_o6_support_completion/o6_support_completion_matrix_probe5 \
    --dataset scannet \
    --stream3d-config scannet

run_logged logs/stream4d_v9_o6_support_completion_reportable_scan.log \
  "$PY" -m tools.scan_reportable_configs \
    --root . \
    --configs stream4d_v9_o6_o1_complete_s0_r002_probe5,stream4d_v9_o6_o1_complete_s0_r005_probe5,stream4d_v9_o6_o1_complete_s0_r010_probe5,stream4d_v9_o6_o1_complete_s0_all_probe5 \
    --output outputs/audit/v9_o6_support_completion/reportable_config_scan_o6_probe5.md \
    --require-manifest \
    --require-eval-policy

run_logged logs/stream4d_v9_o6_support_completion_metric_integrity.log \
  "$PY" -m tools.verify_stream4d_metric_integrity \
    --orig-stream3d-root . \
    --current-root . \
    --configs stream4d_v9_o6_o1_complete_s0_r002_probe5,stream4d_v9_o6_o1_complete_s0_r005_probe5,stream4d_v9_o6_o1_complete_s0_r010_probe5,stream4d_v9_o6_o1_complete_s0_all_probe5 \
    --seq-list splits/scannet_v6_probe5.txt \
    --output outputs/audit/v9_o6_support_completion/metric_integrity_o6_probe5.md \
    --require-manifest

echo "v9 O6 support-completion cross-support done"
