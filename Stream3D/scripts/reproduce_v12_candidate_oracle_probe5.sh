#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/v12_candidate_oracle outputs/audit/v12_candidate_oracle data/evaluation/scannet

run_logged() {
  local log_path="$1"
  shift
  "$@" > "$log_path" 2>&1
}

export_surfel_candidate() {
  run_logged logs/stream4d_v12_candidate_surfel_cluster_export.log \
    "$PY" -m tools.export_v12_object_explanation \
      --bank-root outputs/v12_measurement_bank \
      --seq-list splits/scannet_v6_probe5.txt \
      --output-config stream4d_v12_c_surfel_cluster_candidate_probe5 \
      --mode surfel_cluster_candidate \
      --summary-root outputs/v12_candidate_oracle \
      --seed 12 \
      --birth-min-surfels 16 \
      --birth-min-boundary-safe-ratio 0.0 \
      --birth-max-ambiguous-ratio 1.0 \
      --core-posterior-threshold 0.0 \
      --fringe-posterior-threshold 0.0 \
      --reject-negative-threshold 1.0 \
      --max-slots-per-frame-mask 3 \
      --min-core-surfels-per-object 12 \
      --min-export-points-per-object 100 \
      --measurement-min-surfels 4 \
      --measurement-min-core-ratio 0.08 \
      --export-nn-radius 0.05 \
      --export-mask-sample-stride 2 \
      --export-mask-max-pixels 50000 \
      --export-score-mode reliability \
      --diagnostic-candidate-only
}

oracle_pool() {
  local pool_name="$1"
  local pool_configs="$2"
  local output_config="$3"
  run_logged "logs/stream4d_v12_candidate_${output_config}_select.log" \
    "$PY" -m tools.v11_candidate_pool_oracle \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --pool-name "$pool_name" \
      --pool-configs "$pool_configs" \
      --output-config "$output_config" \
      --support-mode union \
      --summary-root outputs/audit/v12_candidate_oracle \
      --min-candidate-points 100 \
      --min-select-iou 0.25 \
      --dedup-threshold 0.95 \
      --dedup-overlap-mode min_ioc
  run_logged "logs/stream4d_v12_candidate_${output_config}_eval.log" \
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

run_logged logs/stream4d_v12_candidate_oracle_py_compile.log \
  "$PY" -m py_compile \
    tools/v11_candidate_pool_oracle.py \
    tools/export_v12_object_explanation.py \
    tools/summarize_v10_unified_eval.py \
    evaluation/evaluate.py

export_surfel_candidate

oracle_pool \
  "C_mask_b1_o1_o38" \
  "stream4d_v8_b1_surfacelet_singlemask_probe5,stream4d_v9_o1_b1_core_only_probe5,stream4d_v9_o38_o22_memory_c055_newpts_logarea_probe5" \
  "stream4d_v12_oracle_c_mask_probe5"

oracle_pool \
  "C_regionlet_v10_birth_repair" \
  "stream4d_v10_r0_fullmask_probe5,stream4d_v10_r1_maskcore_probe5,stream4d_v10_r2_depthsplit_probe5,stream4d_v10_r3_d4rtseed_probe5,stream4d_v10_r4_combined_probe5,stream4d_v10_r1b_maskcore_32f_wta_probe5" \
  "stream4d_v12_oracle_c_regionlet_probe5"

oracle_pool \
  "C_surfel_cluster_candidate" \
  "stream4d_v12_c_surfel_cluster_candidate_probe5" \
  "stream4d_v12_oracle_c_surfel_cluster_probe5"

oracle_pool \
  "C_hybrid_mask_regionlet_surfel" \
  "stream4d_v8_b1_surfacelet_singlemask_probe5,stream4d_v9_o1_b1_core_only_probe5,stream4d_v9_o38_o22_memory_c055_newpts_logarea_probe5,stream4d_v10_r0_fullmask_probe5,stream4d_v10_r1_maskcore_probe5,stream4d_v10_r2_depthsplit_probe5,stream4d_v10_r3_d4rtseed_probe5,stream4d_v10_r4_combined_probe5,stream4d_v10_r1b_maskcore_32f_wta_probe5,stream4d_v12_c_surfel_cluster_candidate_probe5" \
  "stream4d_v12_oracle_c_hybrid_probe5"

run_logged logs/stream4d_v12_candidate_oracle_matrix.log \
  "$PY" -m tools.summarize_v10_unified_eval \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --matrix-json scripts/v12_candidate_oracle_matrix_probe5.json \
    --output-prefix outputs/audit/v12_candidate_oracle/candidate_oracle_matrix_probe5 \
    --plot-dir outputs/audit/v12_candidate_oracle \
    --dataset scannet \
    --stream3d-config scannet

echo "v12 candidate oracle probe5 done"
