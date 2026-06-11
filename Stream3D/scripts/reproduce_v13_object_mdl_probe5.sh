#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/users/chengshun.wang/pjs/LoGeR"
STREAM3D="$ROOT/Stream3D"
PY="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache}"
export PATH="/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH"

cd "$STREAM3D"
mkdir -p logs outputs/v13_object_explanation_mdl outputs/audit/v13_object_explanation_mdl data/evaluation/scannet

run_logged() {
  local log_path="$1"
  shift
  "$@" > "$log_path" 2>&1
}

eval_own() {
  local config="$1"
  run_logged "logs/stream4d_v13_mdl_${config}_eval.log" \
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
  run_logged "logs/stream4d_v13_mdl_${output_config}.log" \
    "$PY" -m tools.evaluate_cross_prepoints \
      --root . \
      --seq-list splits/scannet_v6_probe5.txt \
      --pred-config "$pred_config" \
      --pre-points-config "$pre_points_config" \
      --output-config "$output_config" \
      --dataset scannet \
      --no-class \
      --output-file "data/evaluation/scannet/${output_config}_class_agnostic.txt" \
      --audit-root outputs/audit/v13_object_explanation_mdl \
      --require-manifest \
      --allow-diagnostic-eval \
      --eval-policy cross_fixed_support
}

export_mdl() {
  local config="$1"
  local support_mode="$2"
  shift 2
  run_logged "logs/stream4d_v13_mdl_${config}_export.log" \
    "$PY" -m tools.export_v13_object_explanation_mdl \
      --bank-root outputs/v12_measurement_bank \
      --masklet-root outputs/v13_masklet_measurements \
      --masklet-mode C3 \
      --seq-list splits/scannet_v6_probe5.txt \
      --output-config "$config" \
      --summary-root outputs/v13_object_explanation_mdl \
      --export-support-mode "$support_mode" \
      --export-nn-radius 0.05 \
      --export-core-nn-radius 0.05 \
      --export-fringe-nn-radius 0.05 \
      --export-fringe-radius 0.05 \
      --export-fringe-max-ratio 0.35 \
      --min-export-points-per-object 60 \
      "$@"
}

run_logged logs/stream4d_v13_mdl_py_compile.log \
  "$PY" -m py_compile \
    stream4d/object_explanation_mdl.py \
    stream4d/video_masklet.py \
    stream4d/export_scannet.py \
    tools/export_v13_object_explanation_mdl.py \
    tools/evaluate_cross_prepoints.py \
    tools/summarize_v10_unified_eval.py \
    evaluation/evaluate.py

if [ ! -f outputs/v13_masklet_measurements/C3/scene0050_00/masklets.npz ]; then
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash scripts/reproduce_v13_masklet_density_probe5.sh
fi

export_mdl stream4d_v13_m13a_mdl_c3_posterior_probe5 posterior_support \
  --birth-min-surfels 12 \
  --birth-min-boundary-safe-ratio 0.50 \
  --birth-max-ambiguous-ratio 0.60 \
  --core-posterior-threshold 0.62 \
  --fringe-posterior-threshold 0.40 \
  --reject-negative-threshold 0.45 \
  --visible-outside-negative-weight 1.0 \
  --boundary-risk-weight 0.45 \
  --appearance-weight 0.25 \
  --d4rt-temporal-weight 0.60 \
  --min-core-surfels-per-object 8 \
  --measurement-min-surfels 3 \
  --measurement-min-core-ratio 0.05 \
  --max-core-overlap-ratio 0.20

export_mdl stream4d_v13_m13b_mdl_c3_strict_probe5 posterior_support \
  --birth-min-surfels 16 \
  --birth-min-boundary-safe-ratio 0.65 \
  --birth-max-ambiguous-ratio 0.30 \
  --core-posterior-threshold 0.72 \
  --fringe-posterior-threshold 0.50 \
  --reject-negative-threshold 0.35 \
  --visible-outside-negative-weight 1.25 \
  --boundary-risk-weight 0.65 \
  --appearance-weight 0.35 \
  --d4rt-temporal-weight 0.70 \
  --min-core-surfels-per-object 10 \
  --measurement-min-surfels 4 \
  --measurement-min-core-ratio 0.08 \
  --max-core-overlap-ratio 0.12

export_mdl stream4d_v13_m13c_mdl_c3_fullmask_probe5 mask_backproject \
  --birth-min-surfels 12 \
  --birth-min-boundary-safe-ratio 0.50 \
  --birth-max-ambiguous-ratio 0.60 \
  --core-posterior-threshold 0.62 \
  --fringe-posterior-threshold 0.40 \
  --reject-negative-threshold 0.45 \
  --visible-outside-negative-weight 1.0 \
  --boundary-risk-weight 0.45 \
  --appearance-weight 0.25 \
  --d4rt-temporal-weight 0.60 \
  --min-core-surfels-per-object 8 \
  --measurement-min-surfels 3 \
  --measurement-min-core-ratio 0.05 \
  --max-core-overlap-ratio 0.20

for config in \
  stream4d_v13_m13a_mdl_c3_posterior_probe5 \
  stream4d_v13_m13b_mdl_c3_strict_probe5 \
  stream4d_v13_m13c_mdl_c3_fullmask_probe5
do
  eval_own "$config"
  short="${config#stream4d_v13_}"
  short="${short%_probe5}"
  cross_eval scannet "$config" "stream4d_v13_p0_on_${short}_probe5"
  cross_eval "$config" scannet "stream4d_v13_${short}_on_s0_probe5"
  cross_eval "$config" stream4d_32f_self_probe5 "stream4d_v13_${short}_on_s1_probe5"
done

run_logged logs/stream4d_v13_mdl_matrix.log \
  "$PY" -m tools.summarize_v10_unified_eval \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --matrix-json scripts/v13_object_mdl_matrix_probe5.json \
    --output-prefix outputs/audit/v13_object_explanation_mdl/object_mdl_matrix_probe5 \
    --plot-dir outputs/audit/v13_object_explanation_mdl \
    --dataset scannet \
    --stream3d-config scannet

run_logged logs/stream4d_v13_mdl_reportable_scan.log \
  "$PY" -m tools.scan_reportable_configs \
    --root . \
    --configs stream4d_v13_m13a_mdl_c3_posterior_probe5,stream4d_v13_m13b_mdl_c3_strict_probe5,stream4d_v13_m13c_mdl_c3_fullmask_probe5 \
    --output outputs/audit/v13_object_explanation_mdl/reportable_config_scan_v13_mdl_probe5.md \
    --require-manifest \
    --require-eval-policy

echo "v13 MDL object explanation probe5 done"
