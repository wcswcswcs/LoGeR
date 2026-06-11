#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/audit/v9_o14_fullspan_grid8 outputs/v9_fullspan outputs/v9_overlap_suppression data/evaluation/scannet

D4RT_ROOT="../Open-d4rt"
D4RT_CONFIG="../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml"
D4RT_CKPT="../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt"
G1_RUN="stream4d_v9_g1_grid8m002_probe5_fullspan_ws100_loger"
RAW_CONFIG="stream4d_v9_o14_b1_fullspan_grid8_ws100_probe5"
RANKED_CONFIG="stream4d_v9_o14_b1_fullspan_grid8_ws100_overlap_mioc050_probe5"

run_logged() {
  local log_path="$1"
  shift
  "$@" > "$log_path" 2>&1
}

eval_own() {
  local config="$1"
  run_logged "logs/stream4d_v9_o14_${config}_eval.log" \
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
  run_logged "logs/stream4d_v9_o14_${output_config}.log" \
    "$PY" -m tools.evaluate_cross_prepoints \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --pred-config "$pred_config" \
      --pre-points-config "$pre_points_config" \
      --output-config "$output_config" \
      --dataset scannet \
      --no-class \
      --output-file "data/evaluation/scannet/${output_config}_class_agnostic.txt" \
      --audit-root outputs/audit/v9_o14_fullspan_grid8 \
      --require-manifest \
      --allow-diagnostic-eval \
      --eval-policy "$eval_policy"
}

run_logged logs/stream4d_v9_o14_fullspan_grid8_py_compile.log \
  "$PY" -m py_compile \
    tools/export_d4rt_grid_surfel_field_v8.py \
    tools/export_v8_surfel_object_field.py \
    tools/support_aware_object_rank.py \
    tools/summarize_v9_unified_eval.py \
    tools/evaluate_cross_prepoints.py \
    tools/scan_reportable_configs.py \
    tools/verify_stream4d_metric_integrity.py

run_logged logs/stream4d_v9_o14_g1_grid8m002_probe5_fullspan_ws100.log \
  "$PY" -m tools.export_d4rt_grid_surfel_field_v8 \
    --d4rt-root "$D4RT_ROOT" \
    --d4rt-config "$D4RT_CONFIG" \
    --d4rt-ckpt "$D4RT_CKPT" \
    --device cuda \
    --seq-list splits/scannet_v6_probe5.txt \
    --frame-stride 1 \
    --max-frames 5000 \
    --window-size 16 \
    --window-stride 100 \
    --grid-size 8 \
    --grid-margin-ratio 0.02 \
    --visible-min-visibility 0.5 \
    --visible-min-confidence 0.5 \
    --query-chunk-size 4096 \
    --cycle-max-tracks 0 \
    --output-root outputs/v8_d4rt_grid_surfel_field \
    --run-name "$G1_RUN" \
    --allow-missing-masks \
    --continue-on-error

run_logged logs/stream4d_v9_o14_fullspan_grid8_export.log \
  "$PY" -m tools.export_v8_surfel_object_field \
    --debug-root "outputs/v8_d4rt_grid_surfel_field/${G1_RUN}" \
    --seq-list splits/scannet_v6_probe5.txt \
    --output-config "$RAW_CONFIG" \
    --prototype-direction B_surfacelet_singlemask \
    --min-observations 1 \
    --max-observations 1 \
    --min-carriers 4 \
    --min-owned-masks 1 \
    --max-masks-per-object 1 \
    --export-mask-sample-stride 2 \
    --export-mask-max-pixels 50000 \
    --min-points-per-object 20 \
    --summary-root outputs/v9_fullspan \
    --eval-policy own_recompute_d4rt_fullspan_grid8_ws100

run_logged logs/stream4d_v9_o14_fullspan_grid8_overlap_mioc050_export.log \
  "$PY" -m tools.support_aware_object_rank \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --input-config "$RAW_CONFIG" \
    --output-config "$RANKED_CONFIG" \
    --score-pre-points-config "$RAW_CONFIG" \
    --quality-mode score_support_area_conflict_penalty \
    --score-weight 0.25 \
    --overlap-threshold 0.50 \
    --overlap-mode min_ioc \
    --min-support-area 20 \
    --tmp-policy recompute \
    --summary-root outputs/v9_overlap_suppression \
    --eval-policy own_recompute_d4rt_fullspan_grid8_ws100_overlap_mioc050

eval_own "$RAW_CONFIG"
eval_own "$RANKED_CONFIG"

cross_eval scannet "$RANKED_CONFIG" stream4d_v9_p0_on_o14_fullspan_grid8_overlap_mioc050_probe5 stream3d_on_o14_fullspan_grid8_overlap_support
cross_eval "$RANKED_CONFIG" scannet stream4d_v9_o14_fullspan_grid8_overlap_mioc050_on_s0_probe5 method_on_stream3d_support
cross_eval "$RANKED_CONFIG" stream4d_32f_self_probe5 stream4d_v9_o14_fullspan_grid8_overlap_mioc050_on_s1_probe5 method_on_32f_support

run_logged logs/stream4d_v9_o14_fullspan_grid8_matrix_probe5.log \
  "$PY" -m tools.summarize_v9_unified_eval \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --matrix-json scripts/v9_o14_fullspan_grid8_matrix_probe5.json \
    --output-prefix outputs/audit/v9_o14_fullspan_grid8/o14_fullspan_grid8_matrix_probe5 \
    --dataset scannet \
    --stream3d-config scannet

run_logged logs/stream4d_v9_o14_fullspan_grid8_reportable_scan.log \
  "$PY" -m tools.scan_reportable_configs \
    --root . \
    --configs "$RAW_CONFIG","$RANKED_CONFIG" \
    --output outputs/audit/v9_o14_fullspan_grid8/reportable_config_scan_o14_probe5.md \
    --require-manifest \
    --require-eval-policy

run_logged logs/stream4d_v9_o14_fullspan_grid8_metric_integrity.log \
  "$PY" -m tools.verify_stream4d_metric_integrity \
    --orig-stream3d-root . \
    --current-root . \
    --configs "$RAW_CONFIG","$RANKED_CONFIG" \
    --seq-list splits/scannet_v6_probe5.txt \
    --output outputs/audit/v9_o14_fullspan_grid8/metric_integrity_o14_probe5.md \
    --require-manifest

echo "v9 O14 full-span grid8 cross-support done"
