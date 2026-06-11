#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/audit/v9_day0 outputs/v9_b1_controls data/evaluation/scannet

run_logged() {
  local log_path="$1"
  shift
  "$@" > "$log_path" 2>&1
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
      --audit-root outputs/audit/v9_day0 \
      --require-manifest \
      --allow-diagnostic-eval \
      --eval-policy "$eval_policy"
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

export_control() {
  local mode="$1"
  local output_config="$2"
  local seed="$3"
  run_logged "logs/${output_config}_export.log" \
    "$PY" -m tools.export_v9_b1_controls \
      --debug-root outputs/v8_d4rt_grid_surfel_field/stream4d_v8_g1_grid32m002_probe5_16f_stride1_loger \
      --seq-list splits/scannet_v6_probe5.txt \
      --output-config "$output_config" \
      --control-mode "$mode" \
      --seed "$seed" \
      --export-mask-sample-stride 2 \
      --export-mask-max-pixels 50000 \
      --min-points-per-object 20 \
      --summary-root outputs/v9_b1_controls
}

run_logged logs/stream4d_v9_py_compile.log \
  "$PY" -m py_compile \
    tools/evaluate_cross_prepoints.py \
    tools/export_v9_b1_controls.py \
    tools/summarize_v9_unified_eval.py \
    tools/scan_reportable_configs.py

run_logged logs/stream4d_v9_import_smoke.log \
  "$PY" -c "import importlib; mods=['tools.evaluate_cross_prepoints','tools.export_v9_b1_controls','tools.summarize_v9_unified_eval','tools.scan_reportable_configs']; [importlib.import_module(m) for m in mods]; print('v9 import smoke OK')"

run_logged logs/stream4d_v9_unittest.log \
  "$PY" -m unittest discover tests

eval_own stream4d_v8_b1_surfacelet_singlemask_probe5

cross_eval scannet scannet stream4d_v9_p0_on_s0_scannet_probe5 stream3d_self_probe5
cross_eval scannet stream4d_32f_self_probe5 stream4d_v9_p0_on_s1_32f_probe5 cross_fixed_32f_support
cross_eval scannet stream4d_v8_b1_surfacelet_singlemask_probe5 stream4d_v9_p0_on_s2_b1_probe5 cross_fixed_b1_support
cross_eval stream4d_v8_b1_surfacelet_singlemask_probe5 scannet stream4d_v9_p2_b1_on_s0_scannet_probe5 method_on_stream3d_support
cross_eval stream4d_v8_b1_surfacelet_singlemask_probe5 stream4d_32f_self_probe5 stream4d_v9_p2_b1_on_s1_32f_probe5 method_on_32f_support
cross_eval stream4d_v6_e4_probe5_objcomp_m650_g101_compact_only_preserve stream4d_32f_self_probe5 stream4d_v9_p1_v6compact_on_s1_32f_probe5 v6_compact_on_32f_support

export_control no_track stream4d_v9_b1_no_track_probe5 0
export_control shuffle stream4d_v9_b1_shuffle_probe5 0
export_control random_same_count stream4d_v9_b1_random_same_count_s0_probe5 0
export_control area_same_count stream4d_v9_b1_area_same_count_probe5 0
export_control maskcount_same_count stream4d_v9_b1_maskcount_same_count_probe5 0

for config in \
  stream4d_v9_b1_no_track_probe5 \
  stream4d_v9_b1_shuffle_probe5 \
  stream4d_v9_b1_random_same_count_s0_probe5 \
  stream4d_v9_b1_area_same_count_probe5 \
  stream4d_v9_b1_maskcount_same_count_probe5; do
  eval_own "$config"
done

cross_eval scannet stream4d_v9_b1_no_track_probe5 stream4d_v9_p0_on_b1_no_track_support_probe5 stream3d_on_control_support
cross_eval stream4d_v9_b1_no_track_probe5 scannet stream4d_v9_b1_no_track_on_s0_scannet_probe5 control_on_stream3d_support
cross_eval scannet stream4d_v9_b1_shuffle_probe5 stream4d_v9_p0_on_b1_shuffle_support_probe5 stream3d_on_control_support
cross_eval stream4d_v9_b1_shuffle_probe5 scannet stream4d_v9_b1_shuffle_on_s0_scannet_probe5 control_on_stream3d_support
cross_eval scannet stream4d_v9_b1_random_same_count_s0_probe5 stream4d_v9_p0_on_b1_random_support_probe5 stream3d_on_control_support
cross_eval stream4d_v9_b1_random_same_count_s0_probe5 scannet stream4d_v9_b1_random_on_s0_scannet_probe5 control_on_stream3d_support
cross_eval scannet stream4d_v9_b1_area_same_count_probe5 stream4d_v9_p0_on_b1_area_support_probe5 stream3d_on_control_support
cross_eval stream4d_v9_b1_area_same_count_probe5 scannet stream4d_v9_b1_area_on_s0_scannet_probe5 control_on_stream3d_support
cross_eval scannet stream4d_v9_b1_maskcount_same_count_probe5 stream4d_v9_p0_on_b1_maskcount_support_probe5 stream3d_on_control_support
cross_eval stream4d_v9_b1_maskcount_same_count_probe5 scannet stream4d_v9_b1_maskcount_on_s0_scannet_probe5 control_on_stream3d_support

run_logged logs/stream4d_v9_unified_eval_matrix_probe5.log \
  "$PY" -m tools.summarize_v9_unified_eval \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --matrix-json scripts/v9_day0_matrix_probe5.json \
    --output-prefix outputs/audit/v9_day0/unified_eval_matrix_probe5 \
    --dataset scannet \
    --stream3d-config scannet

run_logged logs/stream4d_v9_reportable_scan_controls_probe5.log \
  "$PY" -m tools.scan_reportable_configs \
    --root . \
    --configs stream4d_v8_b1_surfacelet_singlemask_probe5,stream4d_v9_b1_no_track_probe5,stream4d_v9_b1_shuffle_probe5,stream4d_v9_b1_random_same_count_s0_probe5,stream4d_v9_b1_area_same_count_probe5,stream4d_v9_b1_maskcount_same_count_probe5 \
    --output outputs/audit/v9_day0/reportable_config_scan_controls_probe5.md \
    --require-manifest

run_logged logs/stream4d_v9_metric_integrity_controls_probe5.log \
  "$PY" -m tools.verify_stream4d_metric_integrity \
    --orig-stream3d-root . \
    --current-root . \
    --configs stream4d_v8_b1_surfacelet_singlemask_probe5,stream4d_v9_b1_no_track_probe5,stream4d_v9_b1_shuffle_probe5,stream4d_v9_b1_random_same_count_s0_probe5,stream4d_v9_b1_area_same_count_probe5,stream4d_v9_b1_maskcount_same_count_probe5 \
    --seq-list splits/scannet_v6_probe5.txt \
    --output outputs/audit/v9_day0/metric_integrity_controls_probe5.md \
    --require-manifest

echo "v9 Day0 done"
