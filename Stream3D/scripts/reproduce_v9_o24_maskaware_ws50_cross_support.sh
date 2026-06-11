#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/audit/v9_o24_maskaware_ws50 outputs/v9_fullspan outputs/v9_mask_merge outputs/v9_score_calibration data/evaluation/scannet

D4RT_ROOT="../Open-d4rt"
D4RT_CONFIG="../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml"
D4RT_CKPT="../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt"
G1_RUN="stream4d_v9_g1_grid8m002_probe5_fullspan_ws50_mam8_loger"
RAW_CONFIG="stream4d_v9_o24_b1_fullspan_grid8_ws50_mam8_probe5"

run_logged() {
  local log_path="$1"
  shift
  "$@" > "$log_path" 2>&1
}

eval_own() {
  local config="$1"
  run_logged "logs/stream4d_v9_o24_${config}_eval.log" \
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
  run_logged "logs/stream4d_v9_o24_${output_config}.log" \
    "$PY" -m tools.evaluate_cross_prepoints \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --pred-config "$pred_config" \
      --pre-points-config "$pre_points_config" \
      --output-config "$output_config" \
      --dataset scannet \
      --no-class \
      --output-file "data/evaluation/scannet/${output_config}_class_agnostic.txt" \
      --audit-root outputs/audit/v9_o24_maskaware_ws50 \
      --require-manifest \
      --allow-diagnostic-eval \
      --eval-policy "$eval_policy"
}

merge_score_eval() {
  local tag="$1"
  local threshold="$2"
  local merge_config="stream4d_v9_o24_ws50_mam8_merge_${tag}_probe5"
  local score_config="stream4d_v9_o24_ws50_mam8_merge_${tag}_logarea_probe5"

  run_logged "logs/stream4d_v9_o24_ws50_mam8_merge_${tag}_export.log" \
    "$PY" -m tools.merge_overlapping_prediction_masks \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --input-config "$RAW_CONFIG" \
      --output-config "$merge_config" \
      --min-ioc "$threshold" \
      --min-area 20 \
      --min-output-area 20 \
      --summary-root outputs/v9_mask_merge \
      --eval-policy "own_recompute_o24_ws50_mam8_merge_${tag}"

  run_logged "logs/stream4d_v9_o24_ws50_mam8_merge_${tag}_logarea_rescore.log" \
    "$PY" -m tools.rescore_prediction_scores \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --input-config "$merge_config" \
      --output-config "$score_config" \
      --score-feature log_area \
      --base-score-mode constant \
      --constant-score 1.0 \
      --tiebreaker-weight 1.0 \
      --summary-root outputs/v9_score_calibration \
      --eval-policy "own_recompute_score_calibration_o24_ws50_mam8_${tag}_logarea"

  eval_own "$score_config"
  cross_eval "$score_config" scannet "stream4d_v9_o24_ws50_mam8_merge_${tag}_logarea_on_s0_probe5" method_on_stream3d_support
  cross_eval "$score_config" stream4d_32f_self_probe5 "stream4d_v9_o24_ws50_mam8_merge_${tag}_logarea_on_s1_probe5" method_on_32f_support
}

run_logged logs/stream4d_v9_o24_maskaware_ws50_py_compile.log \
  "$PY" -m py_compile \
    tools/export_d4rt_grid_surfel_field_v8.py \
    tools/export_v8_surfel_object_field.py \
    tools/merge_overlapping_prediction_masks.py \
    tools/rescore_prediction_scores.py \
    tools/summarize_v9_unified_eval.py \
    tools/evaluate_cross_prepoints.py \
    tools/scan_reportable_configs.py \
    tools/verify_stream4d_metric_integrity.py

run_logged logs/stream4d_v9_o24_g1_grid8m002_probe5_fullspan_ws50_mam8.log \
  "$PY" -m tools.export_d4rt_grid_surfel_field_v8 \
    --d4rt-root "$D4RT_ROOT" \
    --d4rt-config "$D4RT_CONFIG" \
    --d4rt-ckpt "$D4RT_CKPT" \
    --device cuda \
    --seq-list splits/scannet_v6_probe5.txt \
    --frame-stride 1 \
    --max-frames 5000 \
    --window-size 16 \
    --window-stride 50 \
    --grid-size 8 \
    --grid-margin-ratio 0.02 \
    --mask-aware-min-points-per-mask 8 \
    --min-mask-area 8 \
    --visible-min-visibility 0.5 \
    --visible-min-confidence 0.5 \
    --query-chunk-size 4096 \
    --cycle-max-tracks 0 \
    --output-root outputs/v8_d4rt_grid_surfel_field \
    --run-name "$G1_RUN" \
    --allow-missing-masks \
    --continue-on-error

run_logged logs/stream4d_v9_o24_fullspan_grid8_ws50_mam8_export.log \
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
    --eval-policy own_recompute_d4rt_fullspan_grid8_ws50_mam8

eval_own "$RAW_CONFIG"
merge_score_eval mioc040 0.40
merge_score_eval mioc050 0.50

run_logged logs/stream4d_v9_o24_maskaware_ws50_matrix_probe5.log \
  "$PY" -m tools.summarize_v9_unified_eval \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --matrix-json scripts/v9_o24_maskaware_ws50_matrix_probe5.json \
    --output-prefix outputs/audit/v9_o24_maskaware_ws50/o24_maskaware_ws50_matrix_probe5 \
    --dataset scannet \
    --stream3d-config scannet

run_logged logs/stream4d_v9_o24_maskaware_ws50_reportable_scan.log \
  "$PY" -m tools.scan_reportable_configs \
    --root . \
    --configs "$RAW_CONFIG",stream4d_v9_o24_ws50_mam8_merge_mioc040_logarea_probe5,stream4d_v9_o24_ws50_mam8_merge_mioc050_logarea_probe5 \
    --output outputs/audit/v9_o24_maskaware_ws50/reportable_config_scan_o24_probe5.md \
    --require-manifest \
    --require-eval-policy

run_logged logs/stream4d_v9_o24_maskaware_ws50_metric_integrity.log \
  "$PY" -m tools.verify_stream4d_metric_integrity \
    --orig-stream3d-root . \
    --current-root . \
    --configs "$RAW_CONFIG",stream4d_v9_o24_ws50_mam8_merge_mioc040_logarea_probe5,stream4d_v9_o24_ws50_mam8_merge_mioc050_logarea_probe5 \
    --seq-list splits/scannet_v6_probe5.txt \
    --output outputs/audit/v9_o24_maskaware_ws50/metric_integrity_o24_probe5.md \
    --require-manifest

echo "v9 O24 mask-aware ws50 cross-support done"
