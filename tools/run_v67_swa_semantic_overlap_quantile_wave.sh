#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 q80|q90|q95 [alpha]" >&2
  exit 2
fi

QUANTILE="$1"
ALPHA="${2:-0.50}"
case "$QUANTILE" in
  q80) QUANTILE_VALUE="0.80";;
  q90) QUANTILE_VALUE="0.90";;
  q95) QUANTILE_VALUE="0.95";;
  *)
    echo "Unsupported quantile: $QUANTILE (expected q80, q90, or q95)" >&2
    exit 2
    ;;
esac

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
ALPHA_TAG="${ALPHA_TAG:-$(awk -v a="$ALPHA" 'BEGIN { printf "a%03d", int(a * 100 + 0.5) }')}"
SWA_REPLACE_LAYER_MODE="${SWA_REPLACE_LAYER_MODE:-last}"
case "$SWA_REPLACE_LAYER_MODE" in
  all|first|last|single) ;;
  *)
    echo "Unsupported SWA_REPLACE_LAYER_MODE: $SWA_REPLACE_LAYER_MODE" >&2
    exit 2
    ;;
esac
if [ "$SWA_REPLACE_LAYER_MODE" = "last" ]; then
  LAYER_TAG=""
else
  LAYER_TAG="_${SWA_REPLACE_LAYER_MODE}"
fi

BASE_REL="results/kitti01_hmc_v2/acl2_v67_dense_semantic_reconstruction/phase2_swa_semantic_overlap_h10_structure_highd_${QUANTILE}_replace_${ALPHA_TAG}${LAYER_TAG}_gated/rollouts"
BASE="$ROOT/$BASE_REL"
LOG_DIR="$BASE/_launcher_logs"
mkdir -p "$LOG_DIR"

export ATTN_CUE_BASE="$BASE_REL"
export START_FRAME=0
export END_FRAME=612
unset GLOBAL_CHUNK_OFFSET
export STAGE_C_MODE=reference
export STAGE_C_CACHE_DIR=results/kitti_preprocess/01/stage_c_cache_semantic_chunks
export STAGE_C_CACHE_MODE=read
export STAGE_C_CACHE_REQUIRE_HIT=1
export STAGE_C_CACHE_VALIDATE=0
export SEMANTIC_PRIOR_MODE=spg_v2
export HMC_IGNORE_SEMANTIC_PRIOR=0
export READ_PATH=frame
export READ_LAYER_MODE=early
export BETA_SWA=4.75
export SEMANTIC_ACTION_INACTIVE_READ_CUE_SOURCE=acl2.gg.qq.low.g2_3.past_only.headmean.robustq
export ENABLE_SWA_OVERLAP_SOURCE_REPLACE=1
export SWA_OVERLAP_SOURCE_REPLACE_ALPHA="$ALPHA"
export SWA_OVERLAP_SOURCE_REPLACE_TARGET=kv
export SWA_OVERLAP_SOURCE_REPLACE_LAYER_MODE="$SWA_REPLACE_LAYER_MODE"
export SEMANTIC_ROLE_HIGHD_QUANTILE="$QUANTILE_VALUE"

launch_one() {
  local gpu="$1"
  local chunk_label="$2"
  local chunk_value="$3"
  local suffix="$4"
  local mode="$5"
  local run_name="V67_SWASEM_GATED_H10_${chunk_label}_STRUCT_HIGHD${QUANTILE^^}_REPL_${ALPHA_TAG^^}${suffix}"
  local out_dir="$BASE/$run_name"
  if [ -d "$out_dir" ] && [ "${ALLOW_EXISTING:-0}" != "1" ]; then
    echo "Refusing to overwrite existing run dir: $out_dir" >&2
    echo "Set ALLOW_EXISTING=1 only if you intentionally want to rerun in place." >&2
    return 3
  fi
  (
    export SEMANTIC_ACTION_ACTIVE_CHUNKS="$chunk_value"
    export SWA_OVERLAP_SOURCE_REPLACE_MODE="$mode"
    bash "$ROOT/tools/run_attention_cue_experiment.sh" \
      "$gpu" "$run_name" readonly \
      acl2.gg.qq.low.g2_3.past_only.headmean.robustq \
      4.75 stage_d_x_dg_inv_sqrt
  ) > "$LOG_DIR/$run_name.launch.log" 2>&1 &
}

launch_one 0 CH06 6 "" "semantic_structure_highd_${QUANTILE}"
launch_one 1 CH06 6 "_RAND" "semantic_structure_highd_${QUANTILE}_random_same_mass"
launch_one 2 CH10 10 "" "semantic_structure_highd_${QUANTILE}"
launch_one 3 CH10 10 "_RAND" "semantic_structure_highd_${QUANTILE}_random_same_mass"
launch_one 4 CH12 12 "" "semantic_structure_highd_${QUANTILE}"
launch_one 5 CH12 12 "_RAND" "semantic_structure_highd_${QUANTILE}_random_same_mass"
wait
