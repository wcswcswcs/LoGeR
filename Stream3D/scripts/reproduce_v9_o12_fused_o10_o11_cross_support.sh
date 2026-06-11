#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/audit/v9_o12_fused_o10_o11 outputs/v9_fusion outputs/v9_overlap_suppression data/evaluation/scannet

O10="stream4d_v9_o10_o9_overlap_mioc050_probe5"
O11="stream4d_v9_o11_obsbank_overlap_mioc050_probe5"
UNION="stream4d_v9_o12_o10_o11_union_probe5"
RANKED="stream4d_v9_o12_o10_o11_union_overlap_mioc050_probe5"

run_logged() {
  local log_path="$1"
  shift
  "$@" > "$log_path" 2>&1
}

eval_own() {
  local config="$1"
  run_logged "logs/stream4d_v9_o12_${config}_eval.log" \
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
  run_logged "logs/stream4d_v9_o12_${output_config}.log" \
    "$PY" -m tools.evaluate_cross_prepoints \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --pred-config "$pred_config" \
      --pre-points-config "$pre_points_config" \
      --output-config "$output_config" \
      --dataset scannet \
      --no-class \
      --output-file "data/evaluation/scannet/${output_config}_class_agnostic.txt" \
      --audit-root outputs/audit/v9_o12_fused_o10_o11 \
      --require-manifest \
      --allow-diagnostic-eval \
      --eval-policy "$eval_policy"
}

run_logged logs/stream4d_v9_o12_fused_py_compile.log \
  "$PY" -m py_compile \
    tools/fuse_prediction_configs.py \
    tools/support_aware_object_rank.py \
    tools/summarize_v9_unified_eval.py \
    tools/evaluate_cross_prepoints.py \
    tools/scan_reportable_configs.py \
    tools/verify_stream4d_metric_integrity.py

run_logged logs/stream4d_v9_o12_o10_o11_union_export.log \
  "$PY" -m tools.fuse_prediction_configs \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --primary-config "$O10" \
    --secondary-config "$O11" \
    --output-config "$UNION" \
    --preserve-primary-score \
    --preserve-secondary-score \
    --fusion-mode concatenate \
    --summary-root outputs/v9_fusion \
    --eval-policy own_recompute_o10_o11_union

run_logged logs/stream4d_v9_o12_o10_o11_union_overlap_mioc050_export.log \
  "$PY" -m tools.support_aware_object_rank \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --input-config "$UNION" \
    --output-config "$RANKED" \
    --score-pre-points-config "$UNION" \
    --quality-mode score_support_area_conflict_penalty \
    --score-weight 0.25 \
    --overlap-threshold 0.50 \
    --overlap-mode min_ioc \
    --min-support-area 20 \
    --tmp-policy recompute \
    --summary-root outputs/v9_overlap_suppression \
    --eval-policy own_recompute_o10_o11_union_overlap_mioc050

eval_own "$UNION"
eval_own "$RANKED"

cross_eval scannet "$RANKED" stream4d_v9_p0_on_o12_o10_o11_union_overlap_mioc050_probe5 stream3d_on_o12_fused_overlap_support
cross_eval "$RANKED" scannet stream4d_v9_o12_o10_o11_union_overlap_mioc050_on_s0_probe5 method_on_stream3d_support
cross_eval "$RANKED" stream4d_32f_self_probe5 stream4d_v9_o12_o10_o11_union_overlap_mioc050_on_s1_probe5 method_on_32f_support

run_logged logs/stream4d_v9_o12_fused_matrix_probe5.log \
  "$PY" -m tools.summarize_v9_unified_eval \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --matrix-json scripts/v9_o12_fused_o10_o11_matrix_probe5.json \
    --output-prefix outputs/audit/v9_o12_fused_o10_o11/o12_fused_o10_o11_matrix_probe5 \
    --dataset scannet \
    --stream3d-config scannet

run_logged logs/stream4d_v9_o12_fused_reportable_scan.log \
  "$PY" -m tools.scan_reportable_configs \
    --root . \
    --configs "$UNION","$RANKED" \
    --output outputs/audit/v9_o12_fused_o10_o11/reportable_config_scan_o12_probe5.md \
    --require-manifest \
    --require-eval-policy

run_logged logs/stream4d_v9_o12_fused_metric_integrity.log \
  "$PY" -m tools.verify_stream4d_metric_integrity \
    --orig-stream3d-root . \
    --current-root . \
    --configs "$UNION","$RANKED" \
    --seq-list splits/scannet_v6_probe5.txt \
    --output outputs/audit/v9_o12_fused_o10_o11/metric_integrity_o12_probe5.md \
    --require-manifest

echo "v9 O12 fused O10/O11 cross-support done"
