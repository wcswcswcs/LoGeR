#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/v13_candidate_unsupervised outputs/v13_masklet_candidates outputs/audit/v13_candidate_attribution data/evaluation/scannet

run_logged() {
  local log_path="$1"
  shift
  "$@" > "$log_path" 2>&1
}

eval_own() {
  local config="$1"
  run_logged "logs/stream4d_v13_candidate_${config}_eval.log" \
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
  run_logged "logs/stream4d_v13_candidate_${output_config}.log" \
    "$PY" -m tools.evaluate_cross_prepoints \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --pred-config "$pred_config" \
      --pre-points-config "$pre_points_config" \
      --output-config "$output_config" \
      --dataset scannet \
      --no-class \
      --output-file "data/evaluation/scannet/${output_config}_class_agnostic.txt" \
      --audit-root outputs/audit/v13_candidate_attribution \
      --require-manifest \
      --allow-diagnostic-eval \
      --eval-policy cross_fixed_support
}

unsup_pool() {
  local pool_name="$1"
  local pool_configs="$2"
  local output_config="$3"
  run_logged "logs/stream4d_v13_candidate_${output_config}_select.log" \
    "$PY" -m tools.select_v13_unsupervised_candidate_pool \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --pool-name "$pool_name" \
      --pool-configs "$pool_configs" \
      --output-config "$output_config" \
      --summary-root outputs/v13_candidate_unsupervised \
      --min-candidate-points 100 \
      --dedup-threshold 0.95 \
      --dedup-overlap-mode min_ioc
}

oracle_pool() {
  local pool_name="$1"
  local pool_configs="$2"
  local output_config="$3"
  run_logged "logs/stream4d_v13_candidate_${output_config}_select.log" \
    "$PY" -m tools.v11_candidate_pool_oracle \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --pool-name "$pool_name" \
      --pool-configs "$pool_configs" \
      --output-config "$output_config" \
      --support-mode union \
      --summary-root outputs/audit/v13_candidate_attribution \
      --min-candidate-points 100 \
      --min-select-iou 0.25 \
      --dedup-threshold 0.95 \
      --dedup-overlap-mode min_ioc
  run_logged "logs/stream4d_v13_candidate_${output_config}_eval.log" \
    "$PY" -m evaluation.evaluate \
      --pred_path "data/prediction/${output_config}_class_agnostic" \
      --gt_path data/scannet/gt \
      --dataset scannet \
      --output_file "data/evaluation/scannet/${output_config}_class_agnostic.txt" \
      --tmp_root data/TMP \
      --tmp_config "$output_config" \
      --no_class \
      --require-manifest \
      --allow-oracle-eval
}

run_logged logs/stream4d_v13_candidate_py_compile.log \
  "$PY" -m py_compile \
    stream4d/video_masklet.py \
    tools/select_v13_unsupervised_candidate_pool.py \
    tools/export_v13_masklet_candidates.py \
    tools/v11_candidate_pool_oracle.py \
    tools/summarize_v10_unified_eval.py \
    evaluation/evaluate.py

if [ ! -f outputs/v13_masklet_measurements/C3/scene0050_00/masklets.npz ]; then
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash scripts/reproduce_v13_masklet_density_probe5.sh
fi

run_logged logs/stream4d_v13_candidate_c3_masklet_export.log \
  "$PY" -m tools.export_v13_masklet_candidates \
    --bank-root outputs/v12_measurement_bank \
    --masklet-root outputs/v13_masklet_measurements \
    --masklet-mode C3 \
    --seq-list splits/scannet_v6_probe5.txt \
    --output-config stream4d_v13_c3_masklet_candidate_probe5 \
    --summary-root outputs/v13_masklet_candidates \
    --min-rows-per-candidate 2 \
    --min-core-surfels 8 \
    --min-export-points-per-object 60 \
    --export-nn-radius 0.05 \
    --export-core-nn-radius 0.05 \
    --export-fringe-nn-radius 0.05 \
    --export-fringe-radius 0.05 \
    --export-fringe-max-ratio 0.35 \
    --export-score-mode reliability

unsup_pool \
  "C_mask_unsup_b1_o1_o38" \
  "stream4d_v8_b1_surfacelet_singlemask_probe5,stream4d_v9_o1_b1_core_only_probe5,stream4d_v9_o38_o22_memory_c055_newpts_logarea_probe5" \
  "stream4d_v13_c_mask_unsup_probe5"

unsup_pool \
  "C_regionlet_unsup_v10" \
  "stream4d_v10_r0_fullmask_probe5,stream4d_v10_r1_maskcore_probe5,stream4d_v10_r2_depthsplit_probe5,stream4d_v10_r3_d4rtseed_probe5,stream4d_v10_r4_combined_probe5,stream4d_v10_r1b_maskcore_32f_wta_probe5" \
  "stream4d_v13_c_regionlet_unsup_probe5"

unsup_pool \
  "C_surfel_unsup_v12" \
  "stream4d_v12_c_surfel_cluster_candidate_probe5" \
  "stream4d_v13_c_surfel_unsup_probe5"

unsup_pool \
  "C_hybrid_unsup_mask_regionlet_surfel_masklet" \
  "stream4d_v8_b1_surfacelet_singlemask_probe5,stream4d_v9_o1_b1_core_only_probe5,stream4d_v9_o38_o22_memory_c055_newpts_logarea_probe5,stream4d_v10_r0_fullmask_probe5,stream4d_v10_r1_maskcore_probe5,stream4d_v10_r2_depthsplit_probe5,stream4d_v10_r3_d4rtseed_probe5,stream4d_v10_r4_combined_probe5,stream4d_v10_r1b_maskcore_32f_wta_probe5,stream4d_v12_c_surfel_cluster_candidate_probe5,stream4d_v13_c3_masklet_candidate_probe5" \
  "stream4d_v13_c_hybrid_unsup_probe5"

for config in \
  stream4d_v13_c_mask_unsup_probe5 \
  stream4d_v13_c_regionlet_unsup_probe5 \
  stream4d_v13_c_surfel_unsup_probe5 \
  stream4d_v13_c3_masklet_candidate_probe5 \
  stream4d_v13_c_hybrid_unsup_probe5
do
  eval_own "$config"
  short="${config#stream4d_v13_}"
  short="${short%_probe5}"
  cross_eval scannet "$config" "stream4d_v13_p0_on_${short}_probe5"
  cross_eval "$config" scannet "stream4d_v13_${short}_on_s0_probe5"
  cross_eval "$config" stream4d_32f_self_probe5 "stream4d_v13_${short}_on_s1_probe5"
done

oracle_pool \
  "C_mask_b1_o1_o38" \
  "stream4d_v8_b1_surfacelet_singlemask_probe5,stream4d_v9_o1_b1_core_only_probe5,stream4d_v9_o38_o22_memory_c055_newpts_logarea_probe5" \
  "stream4d_v13_oracle_c_mask_probe5"

oracle_pool \
  "C_regionlet_v10" \
  "stream4d_v10_r0_fullmask_probe5,stream4d_v10_r1_maskcore_probe5,stream4d_v10_r2_depthsplit_probe5,stream4d_v10_r3_d4rtseed_probe5,stream4d_v10_r4_combined_probe5,stream4d_v10_r1b_maskcore_32f_wta_probe5" \
  "stream4d_v13_oracle_c_regionlet_probe5"

oracle_pool \
  "C_surfel_v12" \
  "stream4d_v12_c_surfel_cluster_candidate_probe5" \
  "stream4d_v13_oracle_c_surfel_probe5"

oracle_pool \
  "C_masklet_v13_c3" \
  "stream4d_v13_c3_masklet_candidate_probe5" \
  "stream4d_v13_oracle_c_masklet_probe5"

oracle_pool \
  "C_hybrid_mask_regionlet_surfel_masklet" \
  "stream4d_v8_b1_surfacelet_singlemask_probe5,stream4d_v9_o1_b1_core_only_probe5,stream4d_v9_o38_o22_memory_c055_newpts_logarea_probe5,stream4d_v10_r0_fullmask_probe5,stream4d_v10_r1_maskcore_probe5,stream4d_v10_r2_depthsplit_probe5,stream4d_v10_r3_d4rtseed_probe5,stream4d_v10_r4_combined_probe5,stream4d_v10_r1b_maskcore_32f_wta_probe5,stream4d_v12_c_surfel_cluster_candidate_probe5,stream4d_v13_c3_masklet_candidate_probe5" \
  "stream4d_v13_oracle_c_hybrid_probe5"

run_logged logs/stream4d_v13_candidate_unsup_matrix.log \
  "$PY" -m tools.summarize_v10_unified_eval \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --matrix-json scripts/v13_candidate_unsup_matrix_probe5.json \
    --output-prefix outputs/audit/v13_candidate_attribution/candidate_unsup_matrix_probe5 \
    --plot-dir outputs/audit/v13_candidate_attribution \
    --dataset scannet \
    --stream3d-config scannet

run_logged logs/stream4d_v13_candidate_oracle_matrix.log \
  "$PY" -m tools.summarize_v10_unified_eval \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --matrix-json scripts/v13_candidate_oracle_matrix_probe5.json \
    --output-prefix outputs/audit/v13_candidate_attribution/candidate_oracle_matrix_probe5 \
    --plot-dir outputs/audit/v13_candidate_attribution \
    --dataset scannet \
    --stream3d-config scannet

echo "v13 candidate attribution probe5 done"
