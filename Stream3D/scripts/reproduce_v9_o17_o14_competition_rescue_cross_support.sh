#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/audit/v9_o17_o14_competition_rescue outputs/v9_object_competition data/evaluation/scannet

INPUT="stream4d_v9_o14_b1_fullspan_grid8_ws100_probe5"

run_logged() {
  local log_path="$1"
  shift
  "$@" > "$log_path" 2>&1
}

eval_own() {
  local config="$1"
  run_logged "logs/stream4d_v9_o17_${config}_eval.log" \
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
  run_logged "logs/stream4d_v9_o17_${output_config}.log" \
    "$PY" -m tools.evaluate_cross_prepoints \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --pred-config "$pred_config" \
      --pre-points-config "$pre_points_config" \
      --output-config "$output_config" \
      --dataset scannet \
      --no-class \
      --output-file "data/evaluation/scannet/${output_config}_class_agnostic.txt" \
      --audit-root outputs/audit/v9_o17_o14_competition_rescue \
      --require-manifest \
      --allow-diagnostic-eval \
      --eval-policy "$eval_policy"
}

export_rank() {
  local config="$1"
  local quality_mode="$2"
  local group_threshold="$3"
  local max_instances="$4"
  local rescue_reserve="$5"
  local eval_policy="$6"
  run_logged "logs/stream4d_v9_o17_${config}_export.log" \
    "$PY" -m tools.object_competition_rank \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --input-config "$INPUT" \
      --output-config "$config" \
      --score-pre-points-config "$INPUT" \
      --quality-mode "$quality_mode" \
      --group-overlap-mode min_ioc \
      --group-overlap-threshold "$group_threshold" \
      --min-support-area 20 \
      --max-instances "$max_instances" \
      --tmp-policy recompute \
      --eval-policy "$eval_policy" \
      --small-rescue-reserve "$rescue_reserve" \
      --small-rescue-min-support-area 20 \
      --small-rescue-min-novel-points 60 \
      --small-rescue-overlap-threshold 0.50 \
      --small-rescue-overlap-mode min_ioc \
      --summary-root outputs/v9_object_competition
  eval_own "$config"
}

run_logged logs/stream4d_v9_o17_competition_rescue_py_compile.log \
  "$PY" -m py_compile \
    tools/object_competition_rank.py \
    tools/summarize_v9_unified_eval.py \
    tools/evaluate_cross_prepoints.py \
    tools/scan_reportable_configs.py \
    tools/verify_stream4d_metric_integrity.py

export_rank stream4d_v9_o17_o14_rank_area_g030_probe5 area_unique 0.30 0 0 own_recompute_o14_competition_area_g030
export_rank stream4d_v9_o17_o14_rank_area_g030_rescue_probe5 area_unique 0.30 160 60 own_recompute_o14_competition_area_g030_rescue
export_rank stream4d_v9_o17_o14_rank_uc_g050_rescue_probe5 unique_compact_area 0.50 180 60 own_recompute_o14_competition_uc_g050_rescue

cross_eval stream4d_v9_o17_o14_rank_area_g030_probe5 scannet stream4d_v9_o17_o14_rank_area_g030_on_s0_probe5 method_on_stream3d_support
cross_eval stream4d_v9_o17_o14_rank_area_g030_probe5 stream4d_32f_self_probe5 stream4d_v9_o17_o14_rank_area_g030_on_s1_probe5 method_on_32f_support

cross_eval stream4d_v9_o17_o14_rank_area_g030_rescue_probe5 scannet stream4d_v9_o17_o14_rank_area_g030_rescue_on_s0_probe5 method_on_stream3d_support
cross_eval stream4d_v9_o17_o14_rank_area_g030_rescue_probe5 stream4d_32f_self_probe5 stream4d_v9_o17_o14_rank_area_g030_rescue_on_s1_probe5 method_on_32f_support

cross_eval stream4d_v9_o17_o14_rank_uc_g050_rescue_probe5 scannet stream4d_v9_o17_o14_rank_uc_g050_rescue_on_s0_probe5 method_on_stream3d_support
cross_eval stream4d_v9_o17_o14_rank_uc_g050_rescue_probe5 stream4d_32f_self_probe5 stream4d_v9_o17_o14_rank_uc_g050_rescue_on_s1_probe5 method_on_32f_support

run_logged logs/stream4d_v9_o17_competition_rescue_matrix_probe5.log \
  "$PY" -m tools.summarize_v9_unified_eval \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --matrix-json scripts/v9_o17_o14_competition_rescue_matrix_probe5.json \
    --output-prefix outputs/audit/v9_o17_o14_competition_rescue/o17_o14_competition_rescue_matrix_probe5 \
    --dataset scannet \
    --stream3d-config scannet

run_logged logs/stream4d_v9_o17_competition_rescue_reportable_scan.log \
  "$PY" -m tools.scan_reportable_configs \
    --root . \
    --configs stream4d_v9_o17_o14_rank_area_g030_probe5,stream4d_v9_o17_o14_rank_area_g030_rescue_probe5,stream4d_v9_o17_o14_rank_uc_g050_rescue_probe5 \
    --output outputs/audit/v9_o17_o14_competition_rescue/reportable_config_scan_o17_probe5.md \
    --require-manifest \
    --require-eval-policy

run_logged logs/stream4d_v9_o17_competition_rescue_metric_integrity.log \
  "$PY" -m tools.verify_stream4d_metric_integrity \
    --orig-stream3d-root . \
    --current-root . \
    --configs stream4d_v9_o17_o14_rank_area_g030_probe5,stream4d_v9_o17_o14_rank_area_g030_rescue_probe5,stream4d_v9_o17_o14_rank_uc_g050_rescue_probe5 \
    --seq-list splits/scannet_v6_probe5.txt \
    --output outputs/audit/v9_o17_o14_competition_rescue/metric_integrity_o17_probe5.md \
    --require-manifest

echo "v9 O17 O14 competition-rescue cross-support done"
