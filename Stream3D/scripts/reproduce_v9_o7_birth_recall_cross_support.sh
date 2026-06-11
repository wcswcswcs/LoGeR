#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/audit/v9_o7_birth_recall outputs/v9_birth_recall data/evaluation/scannet

DEBUG_ROOT="outputs/v8_d4rt_grid_surfel_field/stream4d_v8_g1_grid32m002_probe5_16f_stride1_loger"

run_logged() {
  local log_path="$1"
  shift
  "$@" > "$log_path" 2>&1
}

eval_own() {
  local config="$1"
  run_logged "logs/stream4d_v9_o7_${config}_eval.log" \
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
  run_logged "logs/stream4d_v9_o7_${output_config}.log" \
    "$PY" -m tools.evaluate_cross_prepoints \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --pred-config "$pred_config" \
      --pre-points-config "$pre_points_config" \
      --output-config "$output_config" \
      --dataset scannet \
      --no-class \
      --output-file "data/evaluation/scannet/${output_config}_class_agnostic.txt" \
      --audit-root outputs/audit/v9_o7_birth_recall \
      --require-manifest \
      --allow-diagnostic-eval \
      --eval-policy "$eval_policy"
}

export_o7() {
  local tag="$1"
  local min_carriers="$2"
  local policy="$3"
  run_logged "logs/stream4d_v9_o7_b1_recall_${tag}_export.log" \
    "$PY" -m tools.export_v8_surfel_object_field \
      --debug-root "$DEBUG_ROOT" \
      --seq-list splits/scannet_v6_probe5.txt \
      --output-config "stream4d_v9_o7_b1_recall_${tag}_probe5" \
      --prototype-direction B_surfacelet_singlemask \
      --min-observations 1 \
      --max-observations 1 \
      --min-carriers "$min_carriers" \
      --min-owned-masks 1 \
      --max-masks-per-object 1 \
      --export-mask-sample-stride 2 \
      --export-mask-max-pixels 50000 \
      --min-points-per-object 20 \
      --summary-root outputs/v9_birth_recall \
      --eval-policy "$policy"
}

run_logged logs/stream4d_v9_o7_birth_recall_py_compile.log \
  "$PY" -m py_compile \
    tools/export_v8_surfel_object_field.py \
    tools/summarize_v9_unified_eval.py \
    tools/evaluate_cross_prepoints.py \
    tools/scan_reportable_configs.py \
    tools/verify_stream4d_metric_integrity.py

export_o7 mc08 8 own_recompute_birth_recall_mc08
export_o7 mc04 4 own_recompute_birth_recall_mc04
export_o7 mc02 2 own_recompute_birth_recall_mc02

for config in \
  stream4d_v9_o7_b1_recall_mc08_probe5 \
  stream4d_v9_o7_b1_recall_mc04_probe5 \
  stream4d_v9_o7_b1_recall_mc02_probe5; do
  eval_own "$config"
done

cross_eval scannet stream4d_v9_o7_b1_recall_mc08_probe5 stream4d_v9_p0_on_o7_b1_recall_mc08_probe5 stream3d_on_birth_recall_mc08_support
cross_eval stream4d_v9_o7_b1_recall_mc08_probe5 scannet stream4d_v9_o7_b1_recall_mc08_on_s0_probe5 method_on_stream3d_support
cross_eval stream4d_v9_o7_b1_recall_mc08_probe5 stream4d_32f_self_probe5 stream4d_v9_o7_b1_recall_mc08_on_s1_probe5 method_on_32f_support

cross_eval scannet stream4d_v9_o7_b1_recall_mc04_probe5 stream4d_v9_p0_on_o7_b1_recall_mc04_probe5 stream3d_on_birth_recall_mc04_support
cross_eval stream4d_v9_o7_b1_recall_mc04_probe5 scannet stream4d_v9_o7_b1_recall_mc04_on_s0_probe5 method_on_stream3d_support
cross_eval stream4d_v9_o7_b1_recall_mc04_probe5 stream4d_32f_self_probe5 stream4d_v9_o7_b1_recall_mc04_on_s1_probe5 method_on_32f_support

cross_eval scannet stream4d_v9_o7_b1_recall_mc02_probe5 stream4d_v9_p0_on_o7_b1_recall_mc02_probe5 stream3d_on_birth_recall_mc02_support
cross_eval stream4d_v9_o7_b1_recall_mc02_probe5 scannet stream4d_v9_o7_b1_recall_mc02_on_s0_probe5 method_on_stream3d_support
cross_eval stream4d_v9_o7_b1_recall_mc02_probe5 stream4d_32f_self_probe5 stream4d_v9_o7_b1_recall_mc02_on_s1_probe5 method_on_32f_support

run_logged logs/stream4d_v9_o7_birth_recall_matrix_probe5.log \
  "$PY" -m tools.summarize_v9_unified_eval \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --matrix-json scripts/v9_o7_birth_recall_matrix_probe5.json \
    --output-prefix outputs/audit/v9_o7_birth_recall/o7_birth_recall_matrix_probe5 \
    --dataset scannet \
    --stream3d-config scannet

run_logged logs/stream4d_v9_o7_birth_recall_reportable_scan.log \
  "$PY" -m tools.scan_reportable_configs \
    --root . \
    --configs stream4d_v9_o7_b1_recall_mc08_probe5,stream4d_v9_o7_b1_recall_mc04_probe5,stream4d_v9_o7_b1_recall_mc02_probe5 \
    --output outputs/audit/v9_o7_birth_recall/reportable_config_scan_o7_probe5.md \
    --require-manifest \
    --require-eval-policy

run_logged logs/stream4d_v9_o7_birth_recall_metric_integrity.log \
  "$PY" -m tools.verify_stream4d_metric_integrity \
    --orig-stream3d-root . \
    --current-root . \
    --configs stream4d_v9_o7_b1_recall_mc08_probe5,stream4d_v9_o7_b1_recall_mc04_probe5,stream4d_v9_o7_b1_recall_mc02_probe5 \
    --seq-list splits/scannet_v6_probe5.txt \
    --output outputs/audit/v9_o7_birth_recall/metric_integrity_o7_probe5.md \
    --require-manifest

echo "v9 O7 birth-recall cross-support done"
