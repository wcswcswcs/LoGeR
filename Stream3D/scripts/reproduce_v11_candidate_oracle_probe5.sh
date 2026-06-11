#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/audit/v11_candidate_oracle data/evaluation/scannet

run_logged() {
  local log_path="$1"
  shift
  "$@" > "$log_path" 2>&1
}

export_oracle_pool() {
  local pool_name="$1"
  local pool_configs="$2"
  local output_config="$3"
  run_logged "logs/stream4d_v11_candidate_oracle_${output_config}_export.log" \
    "$PY" -m tools.v11_candidate_pool_oracle \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --pool-name "$pool_name" \
      --pool-configs "$pool_configs" \
      --output-config "$output_config" \
      --support-mode union \
      --summary-root outputs/audit/v11_candidate_oracle \
      --min-candidate-points 100 \
      --min-select-iou 0.25 \
      --dedup-threshold 0.95 \
      --dedup-overlap-mode min_ioc
}

eval_oracle_pool() {
  local output_config="$1"
  run_logged "logs/stream4d_v11_candidate_oracle_${output_config}_eval.log" \
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

run_logged logs/stream4d_v11_candidate_oracle_py_compile.log \
  "$PY" -m py_compile \
    tools/v11_candidate_pool_oracle.py \
    tools/summarize_v10_unified_eval.py \
    evaluation/evaluate.py

export_oracle_pool \
  C0_Stream3D \
  scannet \
  stream4d_v11_oracle_c0_stream3d_probe5

export_oracle_pool \
  C1_tiny_clean \
  stream4d_v8_b1_surfacelet_singlemask_probe5,stream4d_v9_o1_b1_core_only_probe5 \
  stream4d_v11_oracle_c1_tiny_clean_probe5

export_oracle_pool \
  C2_o38_memory \
  stream4d_v9_o38_o22_memory_c055_newpts_logarea_probe5 \
  stream4d_v11_oracle_c2_o38_memory_probe5

export_oracle_pool \
  C3_regionlet_birth \
  stream4d_v10_r0_fullmask_probe5,stream4d_v10_r1_maskcore_probe5,stream4d_v10_r2_depthsplit_probe5,stream4d_v10_r3_d4rtseed_probe5,stream4d_v10_r4_combined_probe5 \
  stream4d_v11_oracle_c3_regionlet_birth_probe5

export_oracle_pool \
  C4_regionlet_repair \
  stream4d_v10_r0b_fullmask_32f_wta_probe5,stream4d_v10_r1b_maskcore_32f_wta_probe5,stream4d_v10_r4b_combined_32f_wta_probe5 \
  stream4d_v11_oracle_c4_regionlet_repair_probe5

export_oracle_pool \
  C5_c1_c2_c3_union \
  stream4d_v8_b1_surfacelet_singlemask_probe5,stream4d_v9_o1_b1_core_only_probe5,stream4d_v9_o38_o22_memory_c055_newpts_logarea_probe5,stream4d_v10_r0_fullmask_probe5,stream4d_v10_r1_maskcore_probe5,stream4d_v10_r2_depthsplit_probe5,stream4d_v10_r3_d4rtseed_probe5,stream4d_v10_r4_combined_probe5 \
  stream4d_v11_oracle_c5_c1_c2_c3_union_probe5

for config in \
  stream4d_v11_oracle_c0_stream3d_probe5 \
  stream4d_v11_oracle_c1_tiny_clean_probe5 \
  stream4d_v11_oracle_c2_o38_memory_probe5 \
  stream4d_v11_oracle_c3_regionlet_birth_probe5 \
  stream4d_v11_oracle_c4_regionlet_repair_probe5 \
  stream4d_v11_oracle_c5_c1_c2_c3_union_probe5
do
  eval_oracle_pool "$config"
done

run_logged logs/stream4d_v11_candidate_oracle_matrix.log \
  "$PY" -m tools.summarize_v10_unified_eval \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --matrix-json scripts/v11_candidate_oracle_matrix_probe5.json \
    --output-prefix outputs/audit/v11_candidate_oracle/candidate_oracle_matrix_probe5 \
    --plot-dir outputs/audit/v11_candidate_oracle \
    --dataset scannet \
    --stream3d-config scannet

echo "v11 candidate oracle probe5 done"
