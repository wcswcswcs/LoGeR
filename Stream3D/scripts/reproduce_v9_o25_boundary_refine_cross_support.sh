#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/audit/v9_o25_boundary_refine outputs/v9_boundary_refine data/evaluation/scannet

INPUT="stream4d_v9_o23_o22_ws25_merge_mioc040_logarea_probe5"

run_logged() {
  local log_path="$1"
  shift
  "$@" > "$log_path" 2>&1
}

eval_own() {
  local config="$1"
  run_logged "logs/stream4d_v9_o25_${config}_eval.log" \
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
  run_logged "logs/stream4d_v9_o25_${output_config}.log" \
    "$PY" -m tools.evaluate_cross_prepoints \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --pred-config "$pred_config" \
      --pre-points-config "$pre_points_config" \
      --output-config "$output_config" \
      --dataset scannet \
      --no-class \
      --output-file "data/evaluation/scannet/${output_config}_class_agnostic.txt" \
      --audit-root outputs/audit/v9_o25_boundary_refine \
      --require-manifest \
      --allow-diagnostic-eval \
      --eval-policy "$eval_policy"
}

run_refine() {
  local config="$1"
  local inside="$2"
  local interior="$3"
  local policy="$4"
  run_logged "logs/stream4d_v9_o25_${config}_refine.log" \
    "$PY" -m tools.self_discovered_boundary_refine \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --input-config "$INPUT" \
      --output-config "$config" \
      --frame-stride 10 \
      --max-frames 5000 \
      --max-observations 12 \
      --discovery-max-points 1500 \
      --depth-tolerance 0.08 \
      --boundary-margin-px 2.0 \
      --min-visible-points 8 \
      --min-dominant-points 5 \
      --min-dominant-ratio 0.35 \
      --min-point-visible-views 1 \
      --min-point-inside-ratio "$inside" \
      --min-point-interior-ratio "$interior" \
      --unobserved-policy keep \
      --min-points-before-refine 20 \
      --min-points-after-refine 10 \
      --drop-empty \
      --tmp-policy recompute \
      --summary-root outputs/v9_boundary_refine \
      --eval-policy "$policy"
  eval_own "$config"
  cross_eval "$config" scannet "${config}_on_s0_probe5" method_on_stream3d_support
  cross_eval "$config" stream4d_32f_self_probe5 "${config}_on_s1_probe5" method_on_32f_support
}

run_logged logs/stream4d_v9_o25_boundary_refine_py_compile.log \
  "$PY" -m py_compile \
    tools/self_discovered_boundary_refine.py \
    tools/summarize_v9_unified_eval.py \
    tools/evaluate_cross_prepoints.py \
    tools/scan_reportable_configs.py \
    tools/verify_stream4d_metric_integrity.py

run_refine stream4d_v9_o25_o23_refine_inside050_probe5 0.50 0.0 own_recompute_boundary_refine_o23_inside050
run_refine stream4d_v9_o25_o23_refine_inside070int010_probe5 0.70 0.10 own_recompute_boundary_refine_o23_inside070int010

run_logged logs/stream4d_v9_o25_boundary_refine_matrix_probe5.log \
  "$PY" -m tools.summarize_v9_unified_eval \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --matrix-json scripts/v9_o25_boundary_refine_matrix_probe5.json \
    --output-prefix outputs/audit/v9_o25_boundary_refine/o25_boundary_refine_matrix_probe5 \
    --dataset scannet \
    --stream3d-config scannet

run_logged logs/stream4d_v9_o25_boundary_refine_reportable_scan.log \
  "$PY" -m tools.scan_reportable_configs \
    --root . \
    --configs stream4d_v9_o25_o23_refine_inside050_probe5,stream4d_v9_o25_o23_refine_inside070int010_probe5 \
    --output outputs/audit/v9_o25_boundary_refine/reportable_config_scan_o25_probe5.md \
    --require-manifest \
    --require-eval-policy

run_logged logs/stream4d_v9_o25_boundary_refine_metric_integrity.log \
  "$PY" -m tools.verify_stream4d_metric_integrity \
    --orig-stream3d-root . \
    --current-root . \
    --configs stream4d_v9_o25_o23_refine_inside050_probe5,stream4d_v9_o25_o23_refine_inside070int010_probe5 \
    --seq-list splits/scannet_v6_probe5.txt \
    --output outputs/audit/v9_o25_boundary_refine/metric_integrity_o25_probe5.md \
    --require-manifest

echo "v9 O25 boundary-refine cross-support done"
