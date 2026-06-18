#!/usr/bin/env bash
set -euo pipefail

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
cd "$ROOT"

BASE="${ATTN_CUE_BASE:-results/kitti01_hmc_v2/acl2_v66b_dense_semantic_scale/phase9_parallel_continuation/rollouts}"
mkdir -p "$BASE"
END_FRAME_VALUE="${END_FRAME:-704}"
RUN_PREFIX="${RUN_PREFIX:-V66B_P9_704_TTT_ANCHOR_FLOOR}"

COMMON_ENV=(
  ATTN_CUE_BASE="$BASE"
  END_FRAME="$END_FRAME_VALUE"
  STAGE_C_MODE=reference
  STAGE_C_CACHE_DIR=results/kitti_preprocess/01/stage_c_cache_semantic_chunks
  STAGE_C_CACHE_MODE=read
  STAGE_C_CACHE_REQUIRE_HIT=1
  STAGE_C_CACHE_VALIDATE=0
  STAGE_C_INLINE_WHEN_IGNORED=0
  SEMANTIC_PRIOR_MODE=spg_v2
  HMC_IGNORE_SEMANTIC_PRIOR=0
  SEMANTIC_ROLE_POLICY=none
  SEMANTIC_MEMORY_PATHS=
  ENABLE_SEMANTIC_ANCHOR_TTT_FLOOR=1
  SEMANTIC_ANCHOR_MIN_RATIO=0.02
  SEMANTIC_ANCHOR_MAX_RATIO=0.20
  SEMANTIC_ANCHOR_MIN_SCORE=0.02
  SEMANTIC_ANCHOR_GRID_ROWS=4
  SEMANTIC_ANCHOR_GRID_COLS=4
  READ_PATH=frame
  FRAME_BIAS_MODE=pair
  BETA_SWA=4.75
  RESET_EVERY=5
  TTT_WRITE_POST_ZP_SUMMARY=1
  CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS=1
  CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES=256
)

run_job() {
  local gpu="$1"
  local suffix="$2"
  local anchor_mode="$3"
  local ratio="$4"
  (
    set -x
    env "${COMMON_ENV[@]}" \
      SEMANTIC_ANCHOR_MODE="$anchor_mode" \
      SEMANTIC_ANCHOR_TARGET_RATIO="$ratio" \
      bash tools/run_attention_cue_experiment.sh "$gpu" "${RUN_PREFIX}_${suffix}" hybrid dyn 4.75 stage_d
  ) &
}

# Stable-anchor positive write floor.  R012 is the default anchor density;
# R006 is a more conservative spatially-diverse anchor set.  Each density has
# semantic, random-same-mass, and shuffled-semantic controls.
run_job 0 SEM_R012 semantic 0.12
run_job 1 RANDOM_R012 random_same_mass 0.12
run_job 2 SHUFFLE_R012 shuffled_semantic 0.12
run_job 3 SEM_R006 semantic 0.06
run_job 4 RANDOM_R006 random_same_mass 0.06
run_job 5 SHUFFLE_R006 shuffled_semantic 0.06

wait

echo "[$(date '+%F %T')] all v66b anchor-floor jobs finished"
