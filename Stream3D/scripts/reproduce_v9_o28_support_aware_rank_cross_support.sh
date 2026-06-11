#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/audit/v9_o28_support_aware_rank outputs/v9_support_aware_rank data/evaluation/scannet

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
  run_logged "logs/stream4d_v9_o28_${output_config}.log" \
    "$PY" -m tools.evaluate_cross_prepoints \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --pred-config "$pred_config" \
      --pre-points-config "$pre_points_config" \
      --output-config "$output_config" \
      --dataset scannet \
      --no-class \
      --output-file "data/evaluation/scannet/${output_config}_class_agnostic.txt" \
      --audit-root outputs/audit/v9_o28_support_aware_rank \
      --require-manifest \
      --allow-diagnostic-eval \
      --eval-policy "$eval_policy"
}

rank_config() {
  local output_config="$1"
  local score_support="$2"
  local policy="$3"
  run_logged "logs/stream4d_v9_o28_${output_config}_rank.log" \
    "$PY" -m tools.support_aware_object_rank \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --input-config stream4d_v9_o25_o23_refine_inside050_probe5 \
      --output-config "$output_config" \
      --score-pre-points-config "$score_support" \
      --quality-mode support_area_conflict_penalty \
      --overlap-threshold 0.40 \
      --overlap-mode min_ioc \
      --min-support-area 100 \
      --max-instances 128 \
      --tmp-policy recompute \
      --summary-root outputs/v9_support_aware_rank \
      --eval-policy "$policy" \
      --diagnostic-only
}

run_logged logs/stream4d_v9_o28_support_aware_rank_py_compile.log \
  "$PY" -m py_compile \
    tools/support_aware_object_rank.py \
    tools/summarize_v9_unified_eval.py \
    tools/evaluate_cross_prepoints.py \
    tools/scan_reportable_configs.py \
    tools/verify_stream4d_metric_integrity.py

rank_config stream4d_v9_o28_o25_s0aware_rank_g040_probe5 stream4d_v9_p0_on_s0_scannet_probe5 diagnostic_target_support_aware_rank_s0
cross_eval stream4d_v9_o28_o25_s0aware_rank_g040_probe5 scannet stream4d_v9_o28_o25_s0aware_rank_g040_probe5_on_s0_probe5 diagnostic_target_support_aware_rank_on_s0
cross_eval stream4d_v9_o28_o25_s0aware_rank_g040_probe5 stream4d_32f_self_probe5 stream4d_v9_o28_o25_s0aware_rank_g040_probe5_on_s1_probe5 diagnostic_target_support_aware_rank_on_s1

rank_config stream4d_v9_o28_o25_s1aware_rank_g040_probe5 stream4d_32f_self_probe5 diagnostic_target_support_aware_rank_s1
cross_eval stream4d_v9_o28_o25_s1aware_rank_g040_probe5 stream4d_32f_self_probe5 stream4d_v9_o28_o25_s1aware_rank_g040_probe5_on_s1_probe5 diagnostic_target_support_aware_rank_on_s1
cross_eval stream4d_v9_o28_o25_s1aware_rank_g040_probe5 scannet stream4d_v9_o28_o25_s1aware_rank_g040_probe5_on_s0_probe5 diagnostic_target_support_aware_rank_on_s0

run_logged logs/stream4d_v9_o28_support_aware_rank_matrix_probe5.log \
  "$PY" -m tools.summarize_v9_unified_eval \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --matrix-json scripts/v9_o28_support_aware_rank_matrix_probe5.json \
    --output-prefix outputs/audit/v9_o28_support_aware_rank/o28_support_aware_rank_matrix_probe5 \
    --dataset scannet \
    --stream3d-config scannet

run_logged logs/stream4d_v9_o28_support_aware_rank_reportable_scan.log \
  "$PY" -m tools.scan_reportable_configs \
    --root . \
    --configs stream4d_v9_o28_o25_s0aware_rank_g040_probe5,stream4d_v9_o28_o25_s1aware_rank_g040_probe5 \
    --output outputs/audit/v9_o28_support_aware_rank/reportable_config_scan_o28_probe5.md \
    --require-manifest \
    --require-eval-policy

echo "v9 O28 support-aware rank diagnostic done"
