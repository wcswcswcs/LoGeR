#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/audit/v9_o27_silhouette_score outputs/v9_silhouette_score data/evaluation/scannet

run_logged() {
  local log_path="$1"
  shift
  "$@" > "$log_path" 2>&1
}

eval_own() {
  local config="$1"
  run_logged "logs/stream4d_v9_o27_${config}_eval.log" \
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
  run_logged "logs/stream4d_v9_o27_${output_config}.log" \
    "$PY" -m tools.evaluate_cross_prepoints \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --pred-config "$pred_config" \
      --pre-points-config "$pre_points_config" \
      --output-config "$output_config" \
      --dataset scannet \
      --no-class \
      --output-file "data/evaluation/scannet/${output_config}_class_agnostic.txt" \
      --audit-root outputs/audit/v9_o27_silhouette_score \
      --require-manifest \
      --allow-diagnostic-eval \
      --eval-policy "$eval_policy"
}

score_eval() {
  local output_config="$1"
  local quality_mode="$2"
  local score_weight="$3"
  local silhouette_weight="$4"
  local policy="$5"
  run_logged "logs/stream4d_v9_o27_${output_config}_score.log" \
    "$PY" -m tools.self_discovered_silhouette_score \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --input-config stream4d_v9_o25_o23_refine_inside050_probe5 \
      --output-config "$output_config" \
      --quality-mode "$quality_mode" \
      --score-weight "$score_weight" \
      --silhouette-weight "$silhouette_weight" \
      --frame-stride 10 \
      --max-frames 60 \
      --max-observations 12 \
      --discovery-max-points 800 \
      --score-max-points 1600 \
      --depth-tolerance 0.08 \
      --boundary-margin-px 2.0 \
      --min-visible-points 5 \
      --min-dominant-points 5 \
      --min-dominant-ratio 0.35 \
      --visible-saturation 200 \
      --observation-saturation 4 \
      --summary-root outputs/v9_silhouette_score \
      --eval-policy "$policy"
  eval_own "$output_config"
  cross_eval "$output_config" scannet "${output_config}_on_s0_probe5" method_on_stream3d_support
  cross_eval "$output_config" stream4d_32f_self_probe5 "${output_config}_on_s1_probe5" method_on_32f_support
}

run_logged logs/stream4d_v9_o27_silhouette_score_py_compile.log \
  "$PY" -m py_compile \
    tools/self_discovered_silhouette_score.py \
    tools/self_discovered_boundary_refine.py \
    tools/summarize_v9_unified_eval.py \
    tools/evaluate_cross_prepoints.py \
    tools/scan_reportable_configs.py \
    tools/verify_stream4d_metric_integrity.py

score_eval stream4d_v9_o27_o25_inside050_score_silhouette_area_probe5 score_self_silhouette_area 0.35 0.50 own_self_discovered_silhouette_score_o27
score_eval stream4d_v9_o27_o25_inside050_silhouette_area_probe5 self_silhouette_area 0.0 0.70 own_self_discovered_silhouette_area_o27

run_logged logs/stream4d_v9_o27_silhouette_score_matrix_probe5.log \
  "$PY" -m tools.summarize_v9_unified_eval \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --matrix-json scripts/v9_o27_silhouette_score_matrix_probe5.json \
    --output-prefix outputs/audit/v9_o27_silhouette_score/o27_silhouette_score_matrix_probe5 \
    --dataset scannet \
    --stream3d-config scannet

run_logged logs/stream4d_v9_o27_silhouette_score_reportable_scan.log \
  "$PY" -m tools.scan_reportable_configs \
    --root . \
    --configs stream4d_v9_o27_o25_inside050_score_silhouette_area_probe5,stream4d_v9_o27_o25_inside050_silhouette_area_probe5 \
    --output outputs/audit/v9_o27_silhouette_score/reportable_config_scan_o27_probe5.md \
    --require-manifest \
    --require-eval-policy

run_logged logs/stream4d_v9_o27_silhouette_score_metric_integrity.log \
  "$PY" -m tools.verify_stream4d_metric_integrity \
    --orig-stream3d-root . \
    --current-root . \
    --configs stream4d_v9_o27_o25_inside050_score_silhouette_area_probe5,stream4d_v9_o27_o25_inside050_silhouette_area_probe5 \
    --seq-list splits/scannet_v6_probe5.txt \
    --output outputs/audit/v9_o27_silhouette_score/metric_integrity_o27_probe5.md \
    --require-manifest

echo "v9 O27 silhouette-score cross-support done"
