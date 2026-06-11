#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
PY="${PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"
mkdir -p logs outputs/audit/v17_logs outputs/audit/v17_phase3 outputs/audit/v17_phase4 outputs/v17_object_explanation_solver data/evaluation/scannet

run_logged() {
  local log_path="$1"
  shift
  "$@" > "$log_path" 2>&1
}

eval_own() {
  local config="$1"
  run_logged "outputs/audit/v17_logs/${config}_own_eval.log" \
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
  run_logged "outputs/audit/v17_logs/${output_config}.log" \
    "$PY" -m tools.evaluate_cross_prepoints \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --pred-config "$pred_config" \
      --pre-points-config "$pre_points_config" \
      --output-config "$output_config" \
      --dataset scannet \
      --no-class \
      --output-file "data/evaluation/scannet/${output_config}_class_agnostic.txt" \
      --audit-root outputs/audit/v17_phase3 \
      --require-manifest \
      --allow-diagnostic-eval \
      --eval-policy cross_fixed_support
}

solver() {
  local variant="$1"
  local output_config="$2"
  shift 2
  run_logged "outputs/audit/v17_logs/${output_config}_solver.log" \
    "$PY" -m tools.export_v17_object_explanation_solver \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --output-config "$output_config" \
      --variant "$variant" \
      --use-surfel-seeds \
      "$@"
}

solver real stream4d_v17_m17_real_probe5
solver shuffle stream4d_v17_m17_shuffle_probe5 --random-seed 1702
solver no_temporal stream4d_v17_m17_no_temporal_probe5
solver no_negative stream4d_v17_m17_no_negative_probe5
solver area_only stream4d_v17_m17_area_only_probe5
solver random_same_count stream4d_v17_m17_random_same_count_probe5 \
  --same-count-from-config stream4d_v17_m17_real_probe5 \
  --random-seed 1703

for config in \
  stream4d_v17_m17_real_probe5 \
  stream4d_v17_m17_shuffle_probe5 \
  stream4d_v17_m17_no_temporal_probe5 \
  stream4d_v17_m17_no_negative_probe5 \
  stream4d_v17_m17_area_only_probe5 \
  stream4d_v17_m17_random_same_count_probe5
do
  eval_own "$config"
  short="${config#stream4d_v17_}"
  short="${short%_probe5}"
  cross_eval scannet "$config" "stream4d_v17_p0_on_${short}_probe5"
  cross_eval "$config" scannet "stream4d_v17_${short}_on_s0_probe5"
  cross_eval "$config" stream4d_32f_self_probe5 "stream4d_v17_${short}_on_s1_probe5"
  cross_eval "$config" stream4d_v13_c_hybrid_unsup_probe5 "stream4d_v17_${short}_inherit_c_hybrid_probe5"
done

run_logged outputs/audit/v17_logs/v17_phase3_unified_matrix.log \
  "$PY" -m tools.summarize_v10_unified_eval \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --matrix-json scripts/v17_object_explanation_matrix_probe5.json \
    --output-prefix outputs/audit/v17_phase3/object_explanation_matrix_probe5 \
    --plot-dir outputs/audit/v17_phase3 \
    --dataset scannet \
    --stream3d-config scannet

run_logged outputs/audit/v17_logs/v17_phase4_object_explanation_summary.log \
  "$PY" -m tools.summarize_v17_object_explanation \
    --root . \
    --matrix-json outputs/audit/v17_phase3/object_explanation_matrix_probe5.json \
    --output-prefix outputs/audit/v17_phase4/object_explanation_summary_probe5
