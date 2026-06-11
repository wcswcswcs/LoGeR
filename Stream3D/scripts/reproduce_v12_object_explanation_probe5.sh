#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/v12_object_explanation outputs/audit/v12_object_explanation data/evaluation/scannet

run_logged() {
  local log_path="$1"
  shift
  "$@" > "$log_path" 2>&1
}

eval_own() {
  local config="$1"
  run_logged "logs/stream4d_v12_object_${config}_eval.log" \
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
  run_logged "logs/stream4d_v12_object_${output_config}.log" \
    "$PY" -m tools.evaluate_cross_prepoints \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --pred-config "$pred_config" \
      --pre-points-config "$pre_points_config" \
      --output-config "$output_config" \
      --dataset scannet \
      --no-class \
      --output-file "data/evaluation/scannet/${output_config}_class_agnostic.txt" \
      --audit-root outputs/audit/v12_object_explanation \
      --require-manifest \
      --allow-diagnostic-eval \
      --eval-policy cross_fixed_support
}

export_object() {
  local mode="$1"
  local config="$2"
  run_logged "logs/stream4d_v12_object_${config}_export.log" \
    "$PY" -m tools.export_v12_object_explanation \
      --bank-root outputs/v12_measurement_bank \
      --seq-list splits/scannet_v6_probe5.txt \
      --output-config "$config" \
      --mode "$mode" \
      --summary-root outputs/v12_object_explanation \
      --seed 12 \
      --birth-min-surfels 16 \
      --birth-min-boundary-safe-ratio 0.65 \
      --birth-max-ambiguous-ratio 0.25 \
      --core-posterior-threshold 0.70 \
      --fringe-posterior-threshold 0.45 \
      --reject-negative-threshold 0.40 \
      --visible-outside-negative-weight 1.0 \
      --boundary-risk-weight 0.5 \
      --appearance-weight 0.3 \
      --d4rt-temporal-weight 0.5 \
      --max-slots-per-frame-mask 3 \
      --min-core-surfels-per-object 12 \
      --min-export-points-per-object 100 \
      --measurement-min-surfels 4 \
      --measurement-min-core-ratio 0.08 \
      --export-nn-radius 0.05 \
      --export-mask-sample-stride 2 \
      --export-mask-max-pixels 50000 \
      --export-score-mode reliability
}

run_logged logs/stream4d_v12_object_py_compile.log \
  "$PY" -m py_compile \
    stream4d/measurement_bank.py \
    stream4d/evidence_terms.py \
    stream4d/object_slot.py \
    stream4d/object_explanation.py \
    tools/export_v12_object_explanation.py \
    tools/diagnose_v12_object_explanation.py \
    tools/evaluate_cross_prepoints.py \
    tools/summarize_v10_unified_eval.py \
    evaluation/evaluate.py

export_object no_negative stream4d_v12_m4_no_negative_probe5
export_object with_negative stream4d_v12_m5_with_negative_probe5
export_object shuffled_d4rt stream4d_v12_m6_shuffled_d4rt_probe5
export_object no_d4rt_temporal stream4d_v12_m7_no_d4rt_temporal_probe5

for config in \
  stream4d_v12_m4_no_negative_probe5 \
  stream4d_v12_m5_with_negative_probe5 \
  stream4d_v12_m6_shuffled_d4rt_probe5 \
  stream4d_v12_m7_no_d4rt_temporal_probe5
do
  eval_own "$config"
done

cross_eval stream4d_v8_b1_surfacelet_singlemask_probe5 stream4d_v8_b1_surfacelet_singlemask_probe5 stream4d_v12_m0_b1_own_probe5
cross_eval scannet stream4d_v8_b1_surfacelet_singlemask_probe5 stream4d_v12_p0_on_m0_b1_probe5
cross_eval stream4d_v8_b1_surfacelet_singlemask_probe5 scannet stream4d_v12_m0_b1_on_s0_probe5
cross_eval stream4d_v8_b1_surfacelet_singlemask_probe5 stream4d_32f_self_probe5 stream4d_v12_m0_b1_on_s1_probe5

cross_eval stream4d_v11_s2_area_same_count_probe5 stream4d_v11_s2_area_same_count_probe5 stream4d_v12_m1_mask_area_own_probe5
cross_eval scannet stream4d_v11_s2_area_same_count_probe5 stream4d_v12_p0_on_m1_mask_area_probe5
cross_eval stream4d_v11_s2_area_same_count_probe5 scannet stream4d_v12_m1_mask_area_on_s0_probe5
cross_eval stream4d_v11_s2_area_same_count_probe5 stream4d_32f_self_probe5 stream4d_v12_m1_mask_area_on_s1_probe5

cross_eval stream4d_v10_r1b_maskcore_32f_wta_probe5 stream4d_v10_r1b_maskcore_32f_wta_probe5 stream4d_v12_m2_regionlet_repair_own_probe5
cross_eval scannet stream4d_v10_r1b_maskcore_32f_wta_probe5 stream4d_v12_p0_on_m2_regionlet_repair_probe5
cross_eval stream4d_v10_r1b_maskcore_32f_wta_probe5 scannet stream4d_v12_m2_regionlet_repair_on_s0_probe5
cross_eval stream4d_v10_r1b_maskcore_32f_wta_probe5 stream4d_32f_self_probe5 stream4d_v12_m2_regionlet_repair_on_s1_probe5

cross_eval stream4d_v11_s5_no_track_probe5 stream4d_v11_s5_no_track_probe5 stream4d_v12_m3_no_track_own_probe5
cross_eval scannet stream4d_v11_s5_no_track_probe5 stream4d_v12_p0_on_m3_no_track_probe5
cross_eval stream4d_v11_s5_no_track_probe5 scannet stream4d_v12_m3_no_track_on_s0_probe5
cross_eval stream4d_v11_s5_no_track_probe5 stream4d_32f_self_probe5 stream4d_v12_m3_no_track_on_s1_probe5

for config in \
  stream4d_v12_m4_no_negative_probe5 \
  stream4d_v12_m5_with_negative_probe5 \
  stream4d_v12_m6_shuffled_d4rt_probe5 \
  stream4d_v12_m7_no_d4rt_temporal_probe5
do
  short="${config#stream4d_v12_}"
  short="${short%_probe5}"
  cross_eval scannet "$config" "stream4d_v12_p0_on_${short}_probe5"
  cross_eval "$config" scannet "stream4d_v12_${short}_on_s0_probe5"
  cross_eval "$config" stream4d_32f_self_probe5 "stream4d_v12_${short}_on_s1_probe5"
done

run_logged logs/stream4d_v12_object_internal_diagnostic.log \
  "$PY" -m tools.diagnose_v12_object_explanation \
    --summary-root outputs/v12_object_explanation \
    --configs stream4d_v12_m4_no_negative_probe5,stream4d_v12_m5_with_negative_probe5,stream4d_v12_m6_shuffled_d4rt_probe5,stream4d_v12_m7_no_d4rt_temporal_probe5 \
    --output-prefix outputs/audit/v12_object_explanation/object_explanation_internal_probe5

run_logged logs/stream4d_v12_object_matrix.log \
  "$PY" -m tools.summarize_v10_unified_eval \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --matrix-json scripts/v12_object_explanation_matrix_probe5.json \
    --output-prefix outputs/audit/v12_object_explanation/object_explanation_matrix_probe5 \
    --plot-dir outputs/audit/v12_object_explanation \
    --dataset scannet \
    --stream3d-config scannet

run_logged logs/stream4d_v12_object_reportable_scan.log \
  "$PY" -m tools.scan_reportable_configs \
    --root . \
    --configs stream4d_v12_m4_no_negative_probe5,stream4d_v12_m5_with_negative_probe5,stream4d_v12_m6_shuffled_d4rt_probe5,stream4d_v12_m7_no_d4rt_temporal_probe5 \
    --output outputs/audit/v12_object_explanation/reportable_config_scan_v12_object_probe5.md \
    --require-manifest \
    --require-eval-policy

echo "v12 object explanation probe5 done"
