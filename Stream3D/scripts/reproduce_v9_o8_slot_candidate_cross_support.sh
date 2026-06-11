#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/audit/v9_o8_slot_candidate outputs/v9_slot_candidate data/evaluation/scannet

run_logged() {
  local log_path="$1"
  shift
  "$@" > "$log_path" 2>&1
}

eval_own() {
  local config="$1"
  run_logged "logs/stream4d_v9_o8_${config}_eval.log" \
    "$PY" -m evaluation.evaluate \
      --pred_path "data/prediction/${config}_class_agnostic" \
      --gt_path data/scannet/gt \
      --dataset scannet \
      --output_file "data/evaluation/scannet/${config}_class_agnostic.txt" \
      --tmp_root data/TMP \
      --tmp_config "$config" \
      --no_class \
      --require-manifest \
      --allow-oracle-eval
}

cross_eval() {
  local pred_config="$1"
  local pre_points_config="$2"
  local output_config="$3"
  local eval_policy="$4"
  run_logged "logs/stream4d_v9_o8_${output_config}.log" \
    "$PY" -m tools.evaluate_cross_prepoints \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --pred-config "$pred_config" \
      --pre-points-config "$pre_points_config" \
      --output-config "$output_config" \
      --dataset scannet \
      --no-class \
      --output-file "data/evaluation/scannet/${output_config}_class_agnostic.txt" \
      --audit-root outputs/audit/v9_o8_slot_candidate \
      --require-manifest \
      --allow-diagnostic-eval \
      --eval-policy "$eval_policy"
}

run_logged logs/stream4d_v9_o8_slot_candidate_py_compile.log \
  "$PY" -m py_compile \
    tools/slotwise_candidate_select.py \
    tools/summarize_v9_unified_eval.py \
    tools/evaluate_cross_prepoints.py \
    tools/scan_reportable_configs.py \
    tools/verify_stream4d_metric_integrity.py

run_logged logs/stream4d_v9_o8_scannet_slot_upper_export.log \
  "$PY" -m tools.slotwise_candidate_select \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --slot-config stream4d_v8_b1_surfacelet_singlemask_probe5 \
    --candidate-config scannet \
    --score-pre-points-config stream4d_v8_b1_surfacelet_singlemask_probe5 \
    --output-config stream4d_v9_o8_scannet_slot_upper_probe5 \
    --copy-tmp-from candidate \
    --min-slot-area 20 \
    --min-candidate-area 20 \
    --min-slot-ioc 0.10 \
    --min-candidate-ioc 0.01 \
    --min-iou 0.01 \
    --min-area-ratio 0.02 \
    --max-area-ratio 80.0 \
    --score-weight 0.25 \
    --conflict-weight 0.20 \
    --summary-root outputs/v9_slot_candidate \
    --eval-policy diagnostic_scannet_candidate_slot_select_s0 \
    --diagnostic-only

run_logged logs/stream4d_v9_o8_obsbank_slot_only_export.log \
  "$PY" -m tools.slotwise_candidate_select \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --slot-config stream4d_v8_b1_surfacelet_singlemask_probe5 \
    --candidate-config stream4d_v5_obs_bank_96f_probe5_ioc075 \
    --score-pre-points-config stream4d_v8_b1_surfacelet_singlemask_probe5 \
    --output-config stream4d_v9_o8_obsbank_slot_only_probe5 \
    --copy-tmp-from candidate \
    --min-slot-area 20 \
    --min-candidate-area 20 \
    --min-slot-ioc 0.10 \
    --min-candidate-ioc 0.01 \
    --min-iou 0.005 \
    --min-area-ratio 0.02 \
    --max-area-ratio 80.0 \
    --score-weight 0.25 \
    --conflict-weight 0.20 \
    --summary-root outputs/v9_slot_candidate \
    --eval-policy own_obsbank_slot_only_candidate_select

run_logged logs/stream4d_v9_o8_obsbank_slot_top80_export.log \
  "$PY" -m tools.slotwise_candidate_select \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --slot-config stream4d_v8_b1_surfacelet_singlemask_probe5 \
    --candidate-config stream4d_v5_obs_bank_96f_probe5_ioc075 \
    --score-pre-points-config stream4d_v8_b1_surfacelet_singlemask_probe5 \
    --output-config stream4d_v9_o8_obsbank_slot_top80_probe5 \
    --copy-tmp-from candidate \
    --min-slot-area 20 \
    --min-candidate-area 20 \
    --min-slot-ioc 0.10 \
    --min-candidate-ioc 0.01 \
    --min-iou 0.005 \
    --min-area-ratio 0.02 \
    --max-area-ratio 80.0 \
    --score-weight 0.25 \
    --conflict-weight 0.20 \
    --add-unmatched-top-k 80 \
    --unmatched-score 0.01 \
    --summary-root outputs/v9_slot_candidate \
    --eval-policy own_obsbank_slot_top80_candidate_select

for config in \
  stream4d_v9_o8_scannet_slot_upper_probe5 \
  stream4d_v9_o8_obsbank_slot_only_probe5 \
  stream4d_v9_o8_obsbank_slot_top80_probe5; do
  eval_own "$config"
done

cross_eval stream4d_v9_o8_scannet_slot_upper_probe5 stream4d_32f_self_probe5 stream4d_v9_o8_scannet_slot_upper_on_s1_probe5 diagnostic_scannet_candidate_slot_select_on_s1

cross_eval scannet stream4d_v9_o8_obsbank_slot_only_probe5 stream4d_v9_p0_on_o8_obsbank_slot_only_probe5 stream3d_on_obsbank_slot_only_support
cross_eval stream4d_v9_o8_obsbank_slot_only_probe5 scannet stream4d_v9_o8_obsbank_slot_only_on_s0_probe5 method_on_stream3d_support
cross_eval stream4d_v9_o8_obsbank_slot_only_probe5 stream4d_32f_self_probe5 stream4d_v9_o8_obsbank_slot_only_on_s1_probe5 method_on_32f_support

cross_eval scannet stream4d_v9_o8_obsbank_slot_top80_probe5 stream4d_v9_p0_on_o8_obsbank_slot_top80_probe5 stream3d_on_obsbank_slot_top80_support
cross_eval stream4d_v9_o8_obsbank_slot_top80_probe5 scannet stream4d_v9_o8_obsbank_slot_top80_on_s0_probe5 method_on_stream3d_support
cross_eval stream4d_v9_o8_obsbank_slot_top80_probe5 stream4d_32f_self_probe5 stream4d_v9_o8_obsbank_slot_top80_on_s1_probe5 method_on_32f_support

run_logged logs/stream4d_v9_o8_slot_candidate_matrix_probe5.log \
  "$PY" -m tools.summarize_v9_unified_eval \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --matrix-json scripts/v9_o8_slot_candidate_matrix_probe5.json \
    --output-prefix outputs/audit/v9_o8_slot_candidate/o8_slot_candidate_matrix_probe5 \
    --dataset scannet \
    --stream3d-config scannet

run_logged logs/stream4d_v9_o8_slot_candidate_reportable_scan.log \
  "$PY" -m tools.scan_reportable_configs \
    --root . \
    --configs stream4d_v9_o8_scannet_slot_upper_probe5,stream4d_v9_o8_obsbank_slot_only_probe5,stream4d_v9_o8_obsbank_slot_top80_probe5 \
    --output outputs/audit/v9_o8_slot_candidate/reportable_config_scan_o8_probe5.md \
    --require-manifest \
    --require-eval-policy

run_logged logs/stream4d_v9_o8_slot_candidate_metric_integrity.log \
  "$PY" -m tools.verify_stream4d_metric_integrity \
    --orig-stream3d-root . \
    --current-root . \
    --configs stream4d_v9_o8_scannet_slot_upper_probe5,stream4d_v9_o8_obsbank_slot_only_probe5,stream4d_v9_o8_obsbank_slot_top80_probe5 \
    --seq-list splits/scannet_v6_probe5.txt \
    --output outputs/audit/v9_o8_slot_candidate/metric_integrity_o8_probe5.md \
    --require-manifest

echo "v9 O8 slot-candidate cross-support done"
