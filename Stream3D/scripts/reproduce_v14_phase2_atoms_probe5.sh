#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"

mkdir -p outputs/audit/v14_logs

for V in A0 A1 A2 A3 A4; do
  L=$(printf '%s' "$V" | tr 'A-Z' 'a-z')
  "$PY" -m tools.build_v14_surfel_atom_bank \
    --seq-list splits/scannet_v6_probe5.txt \
    --variant "$V" \
    --output-config "stream4d_v14_${L}_atom_candidate_probe5" \
    --atom-root outputs/v14_surfel_atom_bank \
    --summary-root outputs/v14_surfel_atom_bank \
    --min-surfels 4 \
    --min-export-surfels 4 \
    --min-export-points-per-object 20 \
    --export-enable-wta \
    > "outputs/audit/v14_logs/v14_atom_${V}_reproduce.log" 2>&1
done

"$PY" -m tools.diagnose_v14_atom_oracle \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --variant A0:stream4d_v14_a0_atom_candidate_probe5 \
  --variant A1:stream4d_v14_a1_atom_candidate_probe5 \
  --variant A2:stream4d_v14_a2_atom_candidate_probe5 \
  --variant A3:stream4d_v14_a3_atom_candidate_probe5 \
  --variant A4:stream4d_v14_a4_atom_candidate_probe5 \
  --summary-root outputs/audit/v14_atom_oracle \
  --atom-summary-root outputs/v14_surfel_atom_bank \
  > outputs/audit/v14_logs/v14_atom_oracle_reproduce.log 2>&1

for V in A0 A1 A2 A3 A4; do
  L=$(printf '%s' "$V" | tr 'A-Z' 'a-z')
  "$PY" -m tools.evaluate_cross_prepoints \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --pred-config scannet \
    --source-pre-points-config scannet \
    --pre-points-config "stream4d_v14_${L}_atom_candidate_probe5" \
    --output-config "stream4d_v14_p0_on_${L}_atom_candidate_probe5" \
    --dataset scannet \
    --gt-root data/scannet/gt \
    --no-class \
    --output-file "data/evaluation/scannet/stream4d_v14_p0_on_${L}_atom_candidate_probe5_class_agnostic.txt" \
    --audit-root outputs/audit/v14_atom_support \
    --require-manifest \
    --allow-diagnostic-eval \
    --eval-policy stream3d_on_v14_atom_support
done

# Bank16 repair assumes the 70 generated masks listed in
# outputs/audit/v14_cropformer_bank16_missing_copied_filelist.txt are present.
"$PY" -m tools.build_v12_measurement_bank \
  --debug-root outputs/v10_d4rt_grid_surfel_field/stream4d_v10_g1_grid32m002_probe5_16f_stride1_fresh_gpu67 \
  --seq-list splits/scannet_v6_probe5.txt \
  --scannet-root data/scannet/processed \
  --backbone Cropformer \
  --output-root outputs/v14_measurement_bank_bank16_cropformer \
  --audit-prefix outputs/audit/v14_measurement_bank_bank16/measurement_bank_probe5

for SPEC in \
  "A0 source stream4d_v14_a0_bank16_atom_candidate_probe5" \
  "A3 source stream4d_v14_a3_bank16_atom_candidate_probe5" \
  "A4 source stream4d_v14_a4_bank16_atom_candidate_probe5" \
  "A3 target_dominant stream4d_v14_a3_bank16_target_atom_candidate_probe5" \
  "A4 target_dominant stream4d_v14_a4_bank16_target_atom_candidate_probe5" \
  "A4 target_dominant stream4d_v14_a4_bank16_target_minpts5_atom_candidate_probe5"; do
  set -- $SPEC
  V="$1"; BASE="$2"; CFG="$3"
  EXTRA=()
  if [[ "$CFG" == *minpts5* ]]; then
    EXTRA=(--min-export-points-per-object 5)
  else
    EXTRA=(--min-export-points-per-object 20)
  fi
  "$PY" -m tools.build_v14_surfel_atom_bank \
    --bank-root outputs/v14_measurement_bank_bank16_cropformer \
    --seq-list splits/scannet_v6_probe5.txt \
    --variant "$V" \
    --base-mode "$BASE" \
    --output-config "$CFG" \
    --atom-root outputs/v14_surfel_atom_bank_bank16_target \
    --summary-root outputs/v14_surfel_atom_bank_bank16_target \
    --min-surfels 4 \
    --min-export-surfels 4 \
    "${EXTRA[@]}" \
    --export-enable-wta
done
