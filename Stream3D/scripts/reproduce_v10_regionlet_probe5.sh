#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/audit/v10_regionlet outputs/v10_regionlet_birth outputs/visualization/v10_regionlet_birth data/evaluation/scannet

DEBUG_ROOT="outputs/v10_d4rt_grid_surfel_field/stream4d_v10_g1_grid32m002_probe5_16f_stride1_fresh_gpu67"

run_logged() {
  local log_path="$1"
  shift
  "$@" > "$log_path" 2>&1
}

eval_own() {
  local config="$1"
  run_logged "logs/stream4d_v10_regionlet_${config}_eval.log" \
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
  local support_source="$4"
  local geometry_source="$5"
  run_logged "logs/stream4d_v10_regionlet_${output_config}.log" \
    "$PY" -m tools.evaluate_cross_prepoints \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --pred-config "$pred_config" \
      --pre-points-config "$pre_points_config" \
      --output-config "$output_config" \
      --dataset scannet \
      --no-class \
      --output-file "data/evaluation/scannet/${output_config}_class_agnostic.txt" \
      --audit-root outputs/audit/v10_regionlet \
      --require-manifest \
      --allow-diagnostic-eval \
      --eval-policy cross_fixed_support
  run_logged "logs/stream4d_v10_regionlet_${output_config}_patch.log" \
    "$PY" -m tools.update_config_manifest_fields \
      --root . \
      --config "$output_config" \
      --eval-policy cross_fixed_support \
      --support-source "$support_source" \
      --geometry-source "$geometry_source" \
      --prediction-config "$pred_config" \
      --pre-points-config "$pre_points_config" \
      --uses-gt-for-prediction false \
      --uses-gt-for-diagnostic false \
      --is-method-result false \
      --is-diagnostic-only true \
      --reason "v10 regionlet cross-support diagnostic metadata"
}

export_regionlet() {
  local variant="$1"
  local config="$2"
  run_logged "logs/stream4d_v10_regionlet_${config}_export.log" \
    "$PY" -m tools.export_v10_regionlet_birth \
      --seq-list splits/scannet_v6_probe5.txt \
      --debug-root "$DEBUG_ROOT" \
      --output-config "$config" \
      --variant "$variant" \
      --summary-root outputs/v10_regionlet_birth \
      --min-area-2d 64 \
      --boundary-px 4 \
      --depth-bin-m 0.20 \
      --seed-radius-px 16 \
      --max-pixels-per-regionlet 8000 \
      --nn-radius 0.05 \
      --min-points-per-object 20 \
      --eval-policy own_recompute_paper_style \
      --write-visualizations
  eval_own "$config"
}

run_logged logs/stream4d_v10_regionlet_py_compile.log \
  "$PY" -m py_compile \
    tools/export_v10_regionlet_birth.py \
    tools/summarize_v10_unified_eval.py \
    tools/evaluate_cross_prepoints.py \
    tools/update_config_manifest_fields.py

cross_eval scannet scannet stream4d_v10_p0_on_s0_probe5 stream3d_s0 rgbd_eval_bridge
cross_eval scannet stream4d_32f_self_probe5 stream4d_v10_p0_on_s1_probe5 stream4d_s1 rgbd_eval_bridge

export_regionlet R0_full_mask stream4d_v10_r0_fullmask_probe5
export_regionlet R1_mask_core stream4d_v10_r1_maskcore_probe5
export_regionlet R2_depth_split stream4d_v10_r2_depthsplit_probe5
export_regionlet R3_d4rt_seeded stream4d_v10_r3_d4rtseed_probe5
export_regionlet R4_combined stream4d_v10_r4_combined_probe5

cross_eval scannet stream4d_v10_r0_fullmask_probe5 stream4d_v10_p0_on_r0_probe5 named_config rgbd_eval_bridge
cross_eval stream4d_v10_r0_fullmask_probe5 scannet stream4d_v10_r0_on_s0_probe5 stream3d_s0 rgbd_eval_bridge
cross_eval stream4d_v10_r0_fullmask_probe5 stream4d_32f_self_probe5 stream4d_v10_r0_on_s1_probe5 stream4d_s1 rgbd_eval_bridge

cross_eval scannet stream4d_v10_r1_maskcore_probe5 stream4d_v10_p0_on_r1_probe5 named_config rgbd_eval_bridge
cross_eval stream4d_v10_r1_maskcore_probe5 scannet stream4d_v10_r1_on_s0_probe5 stream3d_s0 rgbd_eval_bridge
cross_eval stream4d_v10_r1_maskcore_probe5 stream4d_32f_self_probe5 stream4d_v10_r1_on_s1_probe5 stream4d_s1 rgbd_eval_bridge

cross_eval scannet stream4d_v10_r2_depthsplit_probe5 stream4d_v10_p0_on_r2_probe5 named_config rgbd_eval_bridge
cross_eval stream4d_v10_r2_depthsplit_probe5 scannet stream4d_v10_r2_on_s0_probe5 stream3d_s0 rgbd_eval_bridge
cross_eval stream4d_v10_r2_depthsplit_probe5 stream4d_32f_self_probe5 stream4d_v10_r2_on_s1_probe5 stream4d_s1 rgbd_eval_bridge

cross_eval scannet stream4d_v10_r3_d4rtseed_probe5 stream4d_v10_p0_on_r3_probe5 named_config rgbd_eval_bridge
cross_eval stream4d_v10_r3_d4rtseed_probe5 scannet stream4d_v10_r3_on_s0_probe5 stream3d_s0 mixed
cross_eval stream4d_v10_r3_d4rtseed_probe5 stream4d_32f_self_probe5 stream4d_v10_r3_on_s1_probe5 stream4d_s1 mixed

cross_eval scannet stream4d_v10_r4_combined_probe5 stream4d_v10_p0_on_r4_probe5 named_config rgbd_eval_bridge
cross_eval stream4d_v10_r4_combined_probe5 scannet stream4d_v10_r4_on_s0_probe5 stream3d_s0 mixed
cross_eval stream4d_v10_r4_combined_probe5 stream4d_32f_self_probe5 stream4d_v10_r4_on_s1_probe5 stream4d_s1 mixed

run_logged logs/stream4d_v10_regionlet_matrix.log \
  "$PY" -m tools.summarize_v10_unified_eval \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --matrix-json scripts/v10_regionlet_matrix_probe5.json \
    --output-prefix outputs/audit/v10_regionlet/regionlet_matrix_probe5 \
    --plot-dir outputs/audit/v10_regionlet \
    --dataset scannet \
    --stream3d-config scannet

run_logged logs/stream4d_v10_regionlet_reportable_scan.log \
  "$PY" -m tools.scan_reportable_configs \
    --root . \
    --configs stream4d_v10_r0_fullmask_probe5,stream4d_v10_r1_maskcore_probe5,stream4d_v10_r2_depthsplit_probe5,stream4d_v10_r3_d4rtseed_probe5,stream4d_v10_r4_combined_probe5 \
    --output outputs/audit/v10_regionlet/reportable_config_scan_regionlet_probe5.md \
    --require-manifest \
    --require-eval-policy

run_logged logs/stream4d_v10_regionlet_metric_integrity.log \
  "$PY" -m tools.verify_stream4d_metric_integrity \
    --orig-stream3d-root . \
    --current-root . \
    --configs stream4d_v10_r0_fullmask_probe5,stream4d_v10_r1_maskcore_probe5,stream4d_v10_r2_depthsplit_probe5,stream4d_v10_r3_d4rtseed_probe5,stream4d_v10_r4_combined_probe5 \
    --seq-list splits/scannet_v6_probe5.txt \
    --output outputs/audit/v10_regionlet/metric_integrity_regionlet_probe5.md \
    --require-manifest

echo "v10 regionlet probe5 done"
