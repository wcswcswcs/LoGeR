#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/v11_posterior_proxy outputs/audit/v11_posterior_proxy data/evaluation/scannet

DEBUG_ROOT="outputs/v10_d4rt_grid_surfel_field/stream4d_v10_g1_grid32m002_probe5_16f_stride1_fresh_gpu67"

run_logged() {
  local log_path="$1"
  shift
  "$@" > "$log_path" 2>&1
}

export_control() {
  local mode="$1"
  local config="$2"
  run_logged "logs/stream4d_v11_posterior_proxy_${config}_export.log" \
    "$PY" -m tools.export_v9_b1_controls \
      --debug-root "$DEBUG_ROOT" \
      --seq-list splits/scannet_v6_probe5.txt \
      --output-config "$config" \
      --control-mode "$mode" \
      --summary-root outputs/v11_posterior_proxy \
      --match-count-config stream4d_v8_b1_surfacelet_singlemask_probe5 \
      --match-count-summary-root outputs/v8_surfel_object_field \
      --fallback-target-count 16 \
      --seed 11 \
      --min-visibility 0.5 \
      --min-confidence 0.5 \
      --export-nn-radius 0.05 \
      --export-mask-sample-stride 2 \
      --export-mask-max-pixels 50000 \
      --min-points-per-object 20 \
      --export-score-mode reliability
}

eval_own() {
  local config="$1"
  run_logged "logs/stream4d_v11_posterior_proxy_${config}_eval.log" \
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
  run_logged "logs/stream4d_v11_posterior_proxy_${output_config}.log" \
    "$PY" -m tools.evaluate_cross_prepoints \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --pred-config "$pred_config" \
      --pre-points-config "$pre_points_config" \
      --output-config "$output_config" \
      --dataset scannet \
      --no-class \
      --output-file "data/evaluation/scannet/${output_config}_class_agnostic.txt" \
      --audit-root outputs/audit/v11_posterior_proxy \
      --require-manifest \
      --allow-diagnostic-eval \
      --eval-policy cross_fixed_support
}

run_logged logs/stream4d_v11_posterior_proxy_py_compile.log \
  "$PY" -m py_compile \
    tools/export_v9_b1_controls.py \
    tools/evaluate_cross_prepoints.py \
    tools/summarize_v10_unified_eval.py \
    evaluation/evaluate.py

export_control area_same_count stream4d_v11_s2_area_same_count_probe5
export_control maskcount_same_count stream4d_v11_s3_d4rt_maskcount_probe5
export_control shuffle stream4d_v11_s4_shuffle_maskcount_probe5
export_control no_track stream4d_v11_s5_no_track_probe5

for config in \
  stream4d_v11_s2_area_same_count_probe5 \
  stream4d_v11_s3_d4rt_maskcount_probe5 \
  stream4d_v11_s4_shuffle_maskcount_probe5 \
  stream4d_v11_s5_no_track_probe5
do
  eval_own "$config"
done

cross_eval scannet stream4d_v11_s2_area_same_count_probe5 stream4d_v11_p0_on_s2_area_same_count_probe5
cross_eval stream4d_v11_s2_area_same_count_probe5 scannet stream4d_v11_s2_area_on_s0_probe5
cross_eval stream4d_v11_s2_area_same_count_probe5 stream4d_32f_self_probe5 stream4d_v11_s2_area_on_s1_probe5

cross_eval scannet stream4d_v11_s3_d4rt_maskcount_probe5 stream4d_v11_p0_on_s3_d4rt_maskcount_probe5
cross_eval stream4d_v11_s3_d4rt_maskcount_probe5 scannet stream4d_v11_s3_d4rt_on_s0_probe5
cross_eval stream4d_v11_s3_d4rt_maskcount_probe5 stream4d_32f_self_probe5 stream4d_v11_s3_d4rt_on_s1_probe5

cross_eval scannet stream4d_v11_s4_shuffle_maskcount_probe5 stream4d_v11_p0_on_s4_shuffle_maskcount_probe5
cross_eval stream4d_v11_s4_shuffle_maskcount_probe5 scannet stream4d_v11_s4_shuffle_on_s0_probe5
cross_eval stream4d_v11_s4_shuffle_maskcount_probe5 stream4d_32f_self_probe5 stream4d_v11_s4_shuffle_on_s1_probe5

cross_eval scannet stream4d_v11_s5_no_track_probe5 stream4d_v11_p0_on_s5_no_track_probe5
cross_eval stream4d_v11_s5_no_track_probe5 scannet stream4d_v11_s5_no_track_on_s0_probe5
cross_eval stream4d_v11_s5_no_track_probe5 stream4d_32f_self_probe5 stream4d_v11_s5_no_track_on_s1_probe5

run_logged logs/stream4d_v11_posterior_proxy_matrix.log \
  "$PY" -m tools.summarize_v10_unified_eval \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --matrix-json scripts/v11_posterior_proxy_matrix_probe5.json \
    --output-prefix outputs/audit/v11_posterior_proxy/posterior_proxy_matrix_probe5 \
    --plot-dir outputs/audit/v11_posterior_proxy \
    --dataset scannet \
    --stream3d-config scannet

echo "v11 posterior proxy probe5 done"
