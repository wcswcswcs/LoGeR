#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/audit/v11_d4rt_geometry outputs/v11_d4rt_geometry outputs/v11_d4rt_stream3d_geometry_adapter data/scannet_d4rt_aligned data/evaluation/scannet

DEBUG_ROOT="outputs/v10_d4rt_grid_surfel_field/stream4d_v10_g1_grid32m002_probe5_16f_stride1_fresh_gpu67"

run_logged() {
  local log_path="$1"
  shift
  "$@" > "$log_path" 2>&1
}

adapter_export() {
  local mode="$1"
  local name="$2"
  run_logged "logs/stream4d_v11_d4rt_geometry_adapter_${name}.log" \
    "$PY" -m tools.run_stream3d_with_d4rt_geometry \
      --debug-root "$DEBUG_ROOT" \
      --seq-list splits/scannet_v6_probe5.txt \
      --output-name "$name" \
      --mode "$mode" \
      --backbone Cropformer \
      --output-root outputs/v11_d4rt_stream3d_geometry_adapter \
      --summary-root outputs/audit/v11_d4rt_geometry \
      --min-visibility 0.5 \
      --min-confidence 0.5 \
      --max-anchors 8000 \
      --robust-trim-percentile 90
}

eval_diagnostic_own() {
  local config="$1"
  run_logged "logs/stream4d_v11_d4rt_geometry_${config}_eval.log" \
    "$PY" -m evaluation.evaluate \
      --pred_path "data/prediction/${config}_class_agnostic" \
      --gt_path data/scannet/gt \
      --dataset scannet \
      --output_file "data/evaluation/scannet/${config}_class_agnostic.txt" \
      --tmp_root data/TMP \
      --tmp_config "$config" \
      --no_class \
      --require-manifest \
      --allow-oracle-eval
}

cross_eval() {
  local pred_config="$1"
  local pre_points_config="$2"
  local output_config="$3"
  local uses_gt_diag="$4"
  run_logged "logs/stream4d_v11_d4rt_geometry_${output_config}.log" \
    "$PY" -m tools.evaluate_cross_prepoints \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --pred-config "$pred_config" \
      --pre-points-config "$pre_points_config" \
      --output-config "$output_config" \
      --dataset scannet \
      --no-class \
      --output-file "data/evaluation/scannet/${output_config}_class_agnostic.txt" \
      --audit-root outputs/audit/v11_d4rt_geometry \
      --require-manifest \
      --allow-diagnostic-eval \
      --eval-policy cross_fixed_support
  run_logged "logs/stream4d_v11_d4rt_geometry_${output_config}_patch.log" \
    "$PY" -m tools.update_config_manifest_fields \
      --root . \
      --config "$output_config" \
      --eval-policy cross_fixed_support \
      --prediction-config "$pred_config" \
      --pre-points-config "$pre_points_config" \
      --uses-gt-for-prediction false \
      --uses-gt-for-diagnostic "$uses_gt_diag" \
      --is-method-result false \
      --is-diagnostic-only true \
      --reason "v11 D4RT geometry cross-support diagnostic metadata"
}

export_geometry() {
  local mode="$1"
  local config="$2"
  local eval_policy="$3"
  run_logged "logs/stream4d_v11_d4rt_geometry_${config}_export.log" \
    "$PY" -m tools.materialize_d4rt_aligned_geometry_for_stream3d \
      --debug-root "$DEBUG_ROOT" \
      --seq-list splits/scannet_v6_probe5.txt \
      --output-config "$config" \
      --mode "$mode" \
      --summary-root outputs/v11_d4rt_geometry \
      --output-geometry-root "data/scannet_d4rt_aligned/${config}" \
      --min-visibility 0.5 \
      --min-confidence 0.5 \
      --max-anchors 8000 \
      --robust-trim-percentile 90 \
      --nn-radius 0.05 \
      --density-alpha 2.0 \
      --min-points-per-object 1 \
      --eval-policy "$eval_policy"
  eval_diagnostic_own "$config"
}

run_logged logs/stream4d_v11_d4rt_geometry_py_compile.log \
  "$PY" -m py_compile \
    stream4d/d4rt_stream3d_geometry_adapter.py \
    tools/run_stream3d_with_d4rt_geometry.py \
    tools/materialize_d4rt_aligned_geometry_for_stream3d.py \
    tools/evaluate_cross_prepoints.py \
    tools/summarize_v10_unified_eval.py \
    evaluation/evaluate.py

adapter_export raw stream4d_v11_adapter_raw_probe5
adapter_export scene_sim3 stream4d_v11_adapter_scene_sim3_probe5
adapter_export window_sim3 stream4d_v11_adapter_window_sim3_probe5

export_geometry raw stream4d_v11_g1_d4rt_raw_probe5 d4rt_raw_geometry_diagnostic
export_geometry scene_sim3 stream4d_v11_g2_d4rt_scene_sim3_probe5 d4rt_scene_sim3_geometry_diagnostic
export_geometry window_sim3 stream4d_v11_g3_d4rt_window_sim3_probe5 d4rt_window_sim3_geometry_diagnostic
export_geometry scene_sim3_density stream4d_v11_g4_d4rt_scene_sim3_density_probe5 d4rt_scene_sim3_density_geometry_diagnostic
export_geometry window_sim3_density stream4d_v11_g5_d4rt_window_sim3_density_probe5 d4rt_window_sim3_density_geometry_diagnostic

cross_eval scannet stream4d_v11_g1_d4rt_raw_probe5 stream4d_v11_p0_on_g1_probe5 true
cross_eval stream4d_v11_g1_d4rt_raw_probe5 scannet stream4d_v11_g1_on_s0_probe5 true
cross_eval stream4d_v11_g1_d4rt_raw_probe5 stream4d_32f_self_probe5 stream4d_v11_g1_on_s1_probe5 true

cross_eval scannet stream4d_v11_g2_d4rt_scene_sim3_probe5 stream4d_v11_p0_on_g2_probe5 true
cross_eval stream4d_v11_g2_d4rt_scene_sim3_probe5 scannet stream4d_v11_g2_on_s0_probe5 true
cross_eval stream4d_v11_g2_d4rt_scene_sim3_probe5 stream4d_32f_self_probe5 stream4d_v11_g2_on_s1_probe5 true

cross_eval scannet stream4d_v11_g3_d4rt_window_sim3_probe5 stream4d_v11_p0_on_g3_probe5 true
cross_eval stream4d_v11_g3_d4rt_window_sim3_probe5 scannet stream4d_v11_g3_on_s0_probe5 true
cross_eval stream4d_v11_g3_d4rt_window_sim3_probe5 stream4d_32f_self_probe5 stream4d_v11_g3_on_s1_probe5 true

cross_eval stream4d_v11_g4_d4rt_scene_sim3_density_probe5 scannet stream4d_v11_g4_on_s0_probe5 true
cross_eval stream4d_v11_g4_d4rt_scene_sim3_density_probe5 stream4d_32f_self_probe5 stream4d_v11_g4_on_s1_probe5 true

cross_eval stream4d_v11_g5_d4rt_window_sim3_density_probe5 scannet stream4d_v11_g5_on_s0_probe5 true
cross_eval stream4d_v11_g5_d4rt_window_sim3_density_probe5 stream4d_32f_self_probe5 stream4d_v11_g5_on_s1_probe5 true

run_logged logs/stream4d_v11_d4rt_geometry_matrix.log \
  "$PY" -m tools.summarize_v10_unified_eval \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --matrix-json scripts/v11_d4rt_geometry_matrix_probe5.json \
    --output-prefix outputs/audit/v11_d4rt_geometry/d4rt_geometry_matrix_probe5 \
    --plot-dir outputs/audit/v11_d4rt_geometry \
    --dataset scannet \
    --stream3d-config scannet

run_logged logs/stream4d_v11_d4rt_geometry_reportable_scan.log \
  "$PY" -m tools.scan_reportable_configs \
    --root . \
    --configs stream4d_v11_g1_d4rt_raw_probe5,stream4d_v11_g2_d4rt_scene_sim3_probe5,stream4d_v11_g3_d4rt_window_sim3_probe5,stream4d_v11_g4_d4rt_scene_sim3_density_probe5,stream4d_v11_g5_d4rt_window_sim3_density_probe5 \
    --output outputs/audit/v11_d4rt_geometry/reportable_config_scan_d4rt_geometry_probe5.md \
    --require-manifest \
    --require-eval-policy

echo "v11 D4RT geometry probe5 done"
