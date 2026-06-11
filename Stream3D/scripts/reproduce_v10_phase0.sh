#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/audit/v10_phase0 data/evaluation/scannet

run_logged() {
  local log_path="$1"
  shift
  "$@" > "$log_path" 2>&1
}

patch_manifest() {
  local config="$1"
  local eval_policy="$2"
  local support_source="$3"
  local geometry_source="$4"
  run_logged "logs/stream4d_v10_patch_${config}.log" \
    "$PY" -m tools.update_config_manifest_fields \
      --root . \
      --config "$config" \
      --eval-policy "$eval_policy" \
      --support-source "$support_source" \
      --geometry-source "$geometry_source" \
      --uses-gt-for-prediction false \
      --uses-gt-for-diagnostic false \
      --is-method-result true \
      --is-diagnostic-only false \
      --reason "v10 Phase0 protocol completion for pre-existing artifact"
}

eval_own() {
  local config="$1"
  run_logged "logs/stream4d_v10_phase0_${config}_eval.log" \
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
  local support_source="$5"
  local geometry_source="$6"
  run_logged "logs/stream4d_v10_phase0_${output_config}.log" \
    "$PY" -m tools.evaluate_cross_prepoints \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --pred-config "$pred_config" \
      --pre-points-config "$pre_points_config" \
      --output-config "$output_config" \
      --dataset scannet \
      --no-class \
      --output-file "data/evaluation/scannet/${output_config}_class_agnostic.txt" \
      --audit-root outputs/audit/v10_phase0 \
      --require-manifest \
      --allow-diagnostic-eval \
      --eval-policy "$eval_policy"
  run_logged "logs/stream4d_v10_phase0_${output_config}_patch.log" \
    "$PY" -m tools.update_config_manifest_fields \
      --root . \
      --config "$output_config" \
      --eval-policy "$eval_policy" \
      --support-source "$support_source" \
      --geometry-source "$geometry_source" \
      --prediction-config "$pred_config" \
      --pre-points-config "$pre_points_config" \
      --uses-gt-for-prediction false \
      --uses-gt-for-diagnostic false \
      --is-method-result false \
      --is-diagnostic-only true \
      --reason "v10 Phase0 cross-support diagnostic metadata"
}

run_logged logs/stream4d_v10_phase0_py_compile.log \
  "$PY" -m py_compile \
    tools/evaluate_cross_prepoints.py \
    tools/summarize_v10_unified_eval.py \
    tools/update_config_manifest_fields.py \
    tools/scan_reportable_configs.py \
    tools/verify_stream4d_metric_integrity.py

patch_manifest scannet own_recompute_paper_style stream3d_s0 rgbd_eval_bridge
patch_manifest stream4d_32f_self_probe5 own_recompute_paper_style stream4d_s1 rgbd_eval_bridge
patch_manifest stream4d_v6_e4_probe5_objcomp_m650_g101_compact_only_preserve own_recompute_paper_style own mixed
patch_manifest stream4d_v8_b1_surfacelet_singlemask_probe5 own_recompute_paper_style own mixed
patch_manifest stream4d_v9_o1_b1_core_only_probe5 own_recompute_paper_style own mixed
patch_manifest stream4d_v9_o38_o22_memory_c055_newpts_logarea_probe5 own_recompute_paper_style own mixed

eval_own stream4d_v6_e4_probe5_objcomp_m650_g101_compact_only_preserve
eval_own stream4d_v8_b1_surfacelet_singlemask_probe5
eval_own stream4d_v9_o1_b1_core_only_probe5
eval_own stream4d_v9_o38_o22_memory_c055_newpts_logarea_probe5

cross_eval scannet scannet stream4d_v10_p0_on_s0_probe5 own_recompute_paper_style stream3d_s0 rgbd_eval_bridge
cross_eval scannet stream4d_32f_self_probe5 stream4d_v10_p0_on_s1_probe5 cross_fixed_support stream4d_s1 rgbd_eval_bridge
cross_eval scannet stream4d_v8_b1_surfacelet_singlemask_probe5 stream4d_v10_p0_on_s2_b1_probe5 cross_fixed_support named_config rgbd_eval_bridge
cross_eval scannet stream4d_v6_e4_probe5_objcomp_m650_g101_compact_only_preserve stream4d_v10_p0_on_s5_v6compact_probe5 cross_fixed_support named_config rgbd_eval_bridge
cross_eval scannet stream4d_v9_o1_b1_core_only_probe5 stream4d_v10_p0_on_o1_probe5 cross_fixed_support named_config rgbd_eval_bridge
cross_eval scannet stream4d_v9_o38_o22_memory_c055_newpts_logarea_probe5 stream4d_v10_p0_on_o38_c055_probe5 cross_fixed_support named_config rgbd_eval_bridge
cross_eval stream4d_v8_b1_surfacelet_singlemask_probe5 scannet stream4d_v10_b1_on_s0_probe5 cross_fixed_support stream3d_s0 mixed
cross_eval stream4d_v8_b1_surfacelet_singlemask_probe5 stream4d_32f_self_probe5 stream4d_v10_b1_on_s1_probe5 cross_fixed_support stream4d_s1 mixed
cross_eval stream4d_v6_e4_probe5_objcomp_m650_g101_compact_only_preserve scannet stream4d_v10_v6compact_on_s0_probe5 cross_fixed_support stream3d_s0 mixed
cross_eval stream4d_v6_e4_probe5_objcomp_m650_g101_compact_only_preserve stream4d_32f_self_probe5 stream4d_v10_v6compact_on_s1_probe5 cross_fixed_support stream4d_s1 mixed
cross_eval stream4d_v9_o1_b1_core_only_probe5 scannet stream4d_v10_o1_on_s0_probe5 cross_fixed_support stream3d_s0 mixed
cross_eval stream4d_v9_o1_b1_core_only_probe5 stream4d_32f_self_probe5 stream4d_v10_o1_on_s1_probe5 cross_fixed_support stream4d_s1 mixed
cross_eval stream4d_v9_o38_o22_memory_c055_newpts_logarea_probe5 scannet stream4d_v10_o38_c055_on_s0_probe5 cross_fixed_support stream3d_s0 mixed
cross_eval stream4d_v9_o38_o22_memory_c055_newpts_logarea_probe5 stream4d_32f_self_probe5 stream4d_v10_o38_c055_on_s1_probe5 cross_fixed_support stream4d_s1 mixed

run_logged logs/stream4d_v10_phase0_matrix.log \
  "$PY" -m tools.summarize_v10_unified_eval \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --matrix-json scripts/v10_phase0_matrix_probe5.json \
    --output-prefix outputs/audit/v10_phase0/unified_eval_matrix_probe5 \
    --plot-dir outputs/audit/v10_phase0 \
    --dataset scannet \
    --stream3d-config scannet

run_logged logs/stream4d_v10_phase0_reportable_scan.log \
  "$PY" -m tools.scan_reportable_configs \
    --root . \
    --configs scannet,stream4d_32f_self_probe5,stream4d_v6_e4_probe5_objcomp_m650_g101_compact_only_preserve,stream4d_v8_b1_surfacelet_singlemask_probe5,stream4d_v9_o1_b1_core_only_probe5,stream4d_v9_o38_o22_memory_c055_newpts_logarea_probe5 \
    --output outputs/audit/v10_phase0/reportable_config_scan_phase0_methods.md \
    --require-manifest \
    --require-eval-policy

run_logged logs/stream4d_v10_phase0_metric_integrity.log \
  "$PY" -m tools.verify_stream4d_metric_integrity \
    --orig-stream3d-root . \
    --current-root . \
    --configs scannet,stream4d_32f_self_probe5,stream4d_v6_e4_probe5_objcomp_m650_g101_compact_only_preserve,stream4d_v8_b1_surfacelet_singlemask_probe5,stream4d_v9_o1_b1_core_only_probe5,stream4d_v9_o38_o22_memory_c055_newpts_logarea_probe5 \
    --seq-list splits/scannet_v6_probe5.txt \
    --output outputs/audit/v10_phase0/metric_integrity_phase0_methods.md \
    --require-manifest

echo "v10 Phase0 done"
