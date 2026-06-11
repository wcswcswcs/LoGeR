#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
PY="${PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"

mkdir -p outputs/audit/v15_logs outputs/audit/v15_phase2

run_region() {
  local output_config="$1"
  local summary_prefix="$2"
  local visual_dir="$3"
  local mode="$4"
  shift 4
  "$PY" -m tools.build_v15_mask_region_measurements \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --bank-root outputs/v14_measurement_bank_bank16_cropformer \
    --output-config "$output_config" \
    --region-root outputs/v15_mask_region_measurements \
    --summary-prefix "$summary_prefix" \
    --visual-dir "$visual_dir" \
    --mode "$mode" \
    --min-surfels 5 \
    --min-region-pixels 80 \
    --max-regions-per-scene 600 \
    --pixel-stride 3 \
    --max-pixels-per-region 12000 \
    --min-export-points 20 \
    --gt-diagnostic \
    "$@"
}

run_region stream4d_v15_r0_component_region_probe5 \
  outputs/audit/v15_phase2/r0_component_region_probe5 \
  outputs/audit/v15_phase2/visuals_r0 \
  component \
  --export-nn-radius 0.05 \
  > outputs/audit/v15_logs/v15_phase2_r0_component_region.log 2>&1

run_region stream4d_v15_r0b_component_region_r010_probe5 \
  outputs/audit/v15_phase2/r0b_component_region_r010_probe5 \
  outputs/audit/v15_phase2/visuals_r0b \
  component \
  --export-nn-radius 0.10 \
  > outputs/audit/v15_logs/v15_phase2_r0b_component_region_r010.log 2>&1

run_region stream4d_v15_r1_seed_voronoi_region_probe5 \
  outputs/audit/v15_phase2/r1_seed_voronoi_region_probe5 \
  outputs/audit/v15_phase2/visuals_r1 \
  seed_voronoi \
  --split-grid 2 \
  --export-nn-radius 0.05 \
  > outputs/audit/v15_logs/v15_phase2_r1_seed_voronoi_region.log 2>&1

run_region stream4d_v15_r2_boundary_core_region_probe5 \
  outputs/audit/v15_phase2/r2_boundary_core_region_probe5 \
  outputs/audit/v15_phase2/visuals_r2 \
  boundary_core \
  --erode-px 2 \
  --export-nn-radius 0.05 \
  > outputs/audit/v15_logs/v15_phase2_r2_boundary_core_region.log 2>&1

for spec in \
  "r0_component_region:stream4d_v15_r0_component_region_probe5:outputs/audit/v15_phase2/r0_component_region_union_oracle_probe5:stream4d_v15_oracle_r0_component_region_union_probe5" \
  "r1_seed_voronoi_region:stream4d_v15_r1_seed_voronoi_region_probe5:outputs/audit/v15_phase2/r1_seed_voronoi_region_union_oracle_probe5:stream4d_v15_oracle_r1_seed_voronoi_region_union_probe5" \
  "r2_boundary_core_region:stream4d_v15_r2_boundary_core_region_probe5:outputs/audit/v15_phase2/r2_boundary_core_region_union_oracle_probe5:stream4d_v15_oracle_r2_boundary_core_region_union_probe5"; do
  IFS=: read -r name pred summary oracle_prefix <<< "$spec"
  "$PY" -m tools.diagnose_v15_union_oracle \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --pred-config "$pred" \
    --pre-points-config "$pred" \
    --output-config-prefix "$oracle_prefix" \
    --summary-prefix "$summary" \
    --eval-support candidate \
    --max-candidates-per-gt 256 \
    > "outputs/audit/v15_logs/v15_phase2_${name}_union_oracle.log" 2>&1
done
