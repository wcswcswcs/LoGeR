#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
PY="${PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"
mkdir -p outputs/audit/v17_logs outputs/audit/v17_repairs outputs/v17_object_explanation_solver data/evaluation/scannet

eval_own() {
  local config="$1"
  "$PY" -m evaluation.evaluate \
    --pred_path "data/prediction/${config}_class_agnostic" \
    --gt_path data/scannet/gt \
    --dataset scannet \
    --output_file "data/evaluation/scannet/${config}_class_agnostic.txt" \
    --tmp_root data/TMP \
    --tmp_config "$config" \
    --no_class \
    --require-manifest \
    > "outputs/audit/v17_logs/${config}_own_eval.log" 2>&1
}

cross_eval() {
  local pred_config="$1"
  local pre_points_config="$2"
  local output_config="$3"
  local policy="${4:-cross_fixed_support}"
  "$PY" -m tools.evaluate_cross_prepoints \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --pred-config "$pred_config" \
    --pre-points-config "$pre_points_config" \
    --output-config "$output_config" \
    --dataset scannet \
    --no-class \
    --output-file "data/evaluation/scannet/${output_config}_class_agnostic.txt" \
    --audit-root outputs/audit/v17_repairs \
    --require-manifest \
    --allow-diagnostic-eval \
    --eval-policy "$policy" \
    > "outputs/audit/v17_logs/${output_config}.log" 2>&1
}

eval_repair_five_rows() {
  local config="$1"
  local short="$2"
  local parent="$3"
  local parent_label="$4"
  eval_own "$config"
  cross_eval scannet "$config" "stream4d_v17_p0_on_${short}_probe5"
  cross_eval "$config" scannet "stream4d_v17_${short}_on_s0_probe5"
  cross_eval "$config" stream4d_32f_self_probe5 "stream4d_v17_${short}_on_s1_probe5"
  cross_eval "$config" "$parent" "stream4d_v17_${short}_inherit_${parent_label}_probe5" inherit_parent_support
}

"$PY" -m tools.export_v17_object_explanation_solver \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --output-config stream4d_v17_m17_repair_conflict_probe5 \
  --variant real \
  --use-surfel-seeds \
  --max-measurements-per-slot 1 \
  --target-union-ratio 0.30 \
  --packing-max-min-ioc 0.55 \
  --fallback-max-min-ioc 0.65 \
  --w-anchor 0.25 \
  --w-score 0.15 \
  --w-area 0.35 \
  --w-conflict 1.60 \
  --w-negative 1.00 \
  --w-boundary 1.60 \
  --summary-root outputs/v17_object_explanation_solver \
  > outputs/audit/v17_logs/stream4d_v17_m17_repair_conflict_probe5_solver.log 2>&1
eval_repair_five_rows stream4d_v17_m17_repair_conflict_probe5 m17_repair_conflict stream4d_v13_c_hybrid_unsup_probe5 c_hybrid

"$PY" -m tools.export_v17_object_explanation_solver \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --output-config stream4d_v17_m17_repair_strict_probe5 \
  --variant real \
  --use-surfel-seeds \
  --max-measurements-per-slot 1 \
  --target-union-ratio 0.25 \
  --max-slots 48 \
  --packing-max-min-ioc 0.42 \
  --fallback-max-min-ioc 0.48 \
  --fallback-min-new-points 650 \
  --max-fallback-slots 32 \
  --seed-min-ioc 0.25 \
  --anchor-seed-min-score 0.20 \
  --w-anchor 0.15 \
  --w-score 0.10 \
  --w-area 0.30 \
  --w-conflict 2.50 \
  --w-negative 1.20 \
  --w-boundary 2.50 \
  --summary-root outputs/v17_object_explanation_solver \
  > outputs/audit/v17_logs/stream4d_v17_m17_repair_strict_probe5_solver.log 2>&1
eval_repair_five_rows stream4d_v17_m17_repair_strict_probe5 m17_repair_strict stream4d_v13_c_hybrid_unsup_probe5 c_hybrid

"$PY" -m tools.export_v17_object_explanation_solver \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --pred-config stream4d_v13_c_mask_unsup_probe5 \
  --regionlet-config stream4d_v13_c_regionlet_unsup_probe5 \
  --mask-config stream4d_v13_c_mask_unsup_probe5 \
  --surfel-config stream4d_v13_c_surfel_unsup_probe5 \
  --output-config stream4d_v17_m17_repair_cmask_probe5 \
  --variant real \
  --use-surfel-seeds \
  --max-measurements-per-slot 1 \
  --target-union-ratio 0.30 \
  --packing-max-min-ioc 0.60 \
  --fallback-max-min-ioc 0.70 \
  --fallback-min-new-points 400 \
  --w-anchor 0.30 \
  --w-score 0.20 \
  --w-area 0.30 \
  --w-conflict 1.20 \
  --w-negative 1.00 \
  --w-boundary 1.20 \
  --summary-root outputs/v17_object_explanation_solver \
  > outputs/audit/v17_logs/stream4d_v17_m17_repair_cmask_probe5_solver.log 2>&1
eval_repair_five_rows stream4d_v17_m17_repair_cmask_probe5 m17_repair_cmask stream4d_v13_c_mask_unsup_probe5 c_mask

"$PY" -m tools.select_v13_unsupervised_candidate_pool \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --pool-name v17_cmask_repair_plus_surfel_anchor \
  --pool-configs stream4d_v17_m17_repair_cmask_probe5,stream4d_v13_c_surfel_unsup_probe5 \
  --output-config stream4d_v17_m17_repair_cmask_surfel_probe5 \
  --summary-root outputs/v17_object_explanation_solver \
  --min-candidate-points 100 \
  --dedup-threshold 0.92 \
  --dedup-overlap-mode min_ioc \
  > outputs/audit/v17_logs/stream4d_v17_m17_repair_cmask_surfel_probe5_pool.log 2>&1
"$PY" -m tools.update_config_manifest_fields \
  --root . \
  --config stream4d_v17_m17_repair_cmask_surfel_probe5 \
  --pred-suffix class_agnostic \
  --eval-policy own_recompute_paper_style \
  --support-source own \
  --geometry-source candidate_union \
  --prediction-config stream4d_v17_m17_repair_cmask_surfel_probe5 \
  --pre-points-config stream4d_v17_m17_repair_cmask_surfel_probe5 \
  --algorithm-name v17_cmask_repair_plus_surfel_anchor \
  --algorithm v17_cmask_repair_plus_surfel_anchor \
  --uses-gt-for-prediction false \
  --uses-gt-for-diagnostic false \
  --is-method-result true \
  --is-diagnostic-only false \
  --forbidden-for-method-table false \
  --gt-selected-output false \
  --reason "v17 repair manifest protocol completion" \
  > outputs/audit/v17_logs/stream4d_v17_m17_repair_cmask_surfel_probe5_manifest_update_r2.log 2>&1
eval_repair_five_rows stream4d_v17_m17_repair_cmask_surfel_probe5 m17_repair_cmask_surfel stream4d_v13_c_mask_unsup_probe5 c_mask

"$PY" -m tools.export_v17_object_explanation_solver \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --pred-config stream4d_v13_c_mask_unsup_probe5 \
  --regionlet-config stream4d_v13_c_regionlet_unsup_probe5 \
  --mask-config stream4d_v13_c_mask_unsup_probe5 \
  --surfel-config stream4d_v13_c_surfel_unsup_probe5 \
  --output-config stream4d_v17_m17_repair_cmask_strict_probe5 \
  --variant real \
  --use-surfel-seeds \
  --max-measurements-per-slot 1 \
  --target-union-ratio 0.28 \
  --max-slots 28 \
  --packing-max-min-ioc 0.30 \
  --fallback-max-min-ioc 0.36 \
  --fallback-min-new-points 700 \
  --max-fallback-slots 20 \
  --w-anchor 0.15 \
  --w-score 0.10 \
  --w-area 0.25 \
  --w-conflict 2.50 \
  --w-negative 1.20 \
  --w-boundary 2.50 \
  --summary-root outputs/v17_object_explanation_solver \
  > outputs/audit/v17_logs/stream4d_v17_m17_repair_cmask_strict_probe5_solver.log 2>&1
eval_repair_five_rows stream4d_v17_m17_repair_cmask_strict_probe5 m17_repair_cmask_strict stream4d_v13_c_mask_unsup_probe5 c_mask
