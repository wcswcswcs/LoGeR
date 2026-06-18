#!/usr/bin/env bash
set -euo pipefail

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
cd "$ROOT"

BASE="results/kitti01_hmc_v2/acl2_v66b_dense_semantic_scale/phase9_parallel_continuation/rollouts"
mkdir -p "$BASE"

COMMON_ENV=(
  ATTN_CUE_BASE="$BASE"
  END_FRAME=96
  STAGE_C_MODE=reference
  STAGE_C_CACHE_DIR=results/kitti_preprocess/01/stage_c_cache_semantic_chunks
  STAGE_C_CACHE_MODE=read
  STAGE_C_CACHE_REQUIRE_HIT=1
  STAGE_C_CACHE_VALIDATE=0
  STAGE_C_INLINE_WHEN_IGNORED=0
  SEMANTIC_PRIOR_MODE=spg_v2
  HMC_IGNORE_SEMANTIC_PRIOR=0
  SEMANTIC_ROLE_POLICY=causal_fg_semantic_risk_skip
  READ_PATH=frame
  FRAME_BIAS_MODE=pair
  BETA_SWA=4.75
  RESET_EVERY=5
  CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS=1
  CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES=256
)

run_job() {
  local gpu="$1"
  local run_name="$2"
  local mode="$3"
  shift 3
  (
    set -x
    env "${COMMON_ENV[@]}" "$@" \
      bash tools/run_attention_cue_experiment.sh "$gpu" "$run_name" "$mode" dyn 4.75 stage_d
  ) &
}

# READ R4/R6 variants: strengthen the source intervention that previously had
# action fidelity but lost to random same-mass, and test a group-normalized risk
# variant instead of only the raw negative-role mask.
run_job 0 V66B_P9_READ_NEG_SOFT_RHO050_96F readonly \
  SEMANTIC_MEMORY_PATHS=frame,global \
  ENABLE_CONTEXT_SOURCE_SKIP=1 \
  CONTEXT_SOURCE_SKIP_MASK=semantic_role_negative \
  CONTEXT_SOURCE_SKIP_MODE=soft \
  CONTEXT_SOURCE_SKIP_IMPL=bias \
  CONTEXT_SOURCE_SKIP_SCOPE=frame \
  CONTEXT_SOURCE_SKIP_LAYER_MODE=early \
  CONTEXT_SOURCE_SKIP_SOFT_RHO=0.50 \
  CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP=0.25

run_job 1 V66B_P9_READ_NEG_SOFT_RHO075_ALL_96F readonly \
  SEMANTIC_MEMORY_PATHS=frame,global \
  ENABLE_CONTEXT_SOURCE_SKIP=1 \
  CONTEXT_SOURCE_SKIP_MASK=semantic_role_negative \
  CONTEXT_SOURCE_SKIP_MODE=soft \
  CONTEXT_SOURCE_SKIP_IMPL=bias \
  CONTEXT_SOURCE_SKIP_SCOPE=frame \
  CONTEXT_SOURCE_SKIP_LAYER_MODE=all \
  CONTEXT_SOURCE_SKIP_SOFT_RHO=0.75 \
  CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP=0.20

run_job 2 V66B_P9_READ_SEM_Z_DG_SOFT_RHO050_96F readonly \
  SEMANTIC_MEMORY_PATHS=frame,global \
  ENABLE_CONTEXT_SOURCE_SKIP=1 \
  CONTEXT_SOURCE_SKIP_MASK=sem_z_dg_soft_resid \
  CONTEXT_SOURCE_SKIP_MODE=soft \
  CONTEXT_SOURCE_SKIP_IMPL=bias \
  CONTEXT_SOURCE_SKIP_SCOPE=frame \
  CONTEXT_SOURCE_SKIP_LAYER_MODE=early \
  CONTEXT_SOURCE_SKIP_SOFT_RHO=0.50 \
  CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP=0.25

run_job 3 V66B_P9_READ_RANDOM_SAMEMASS_RHO050_96F readonly \
  SEMANTIC_MEMORY_PATHS=frame,global \
  ENABLE_CONTEXT_SOURCE_SKIP=1 \
  CONTEXT_SOURCE_SKIP_MASK=random_same_mass_semantic_role_negative \
  CONTEXT_SOURCE_SKIP_MODE=soft \
  CONTEXT_SOURCE_SKIP_IMPL=bias \
  CONTEXT_SOURCE_SKIP_SCOPE=frame \
  CONTEXT_SOURCE_SKIP_LAYER_MODE=early \
  CONTEXT_SOURCE_SKIP_SOFT_RHO=0.50 \
  CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP=0.25

# TTT variants: the previous 32F run altered write prior by only about 8.9%
# and changed no output.  These test whether the issue is too-weak role scaling
# or missing semantic anchor floor, with output and post-zp hashes audited.
run_job 4 V66B_P9_TTT_ROLE_STRONG_96F hybrid \
  SEMANTIC_MEMORY_PATHS=ttt \
  SEMANTIC_ROLE_POSITIVE_SCALE=1.35 \
  SEMANTIC_ROLE_NEUTRAL_SCALE=0.70 \
  SEMANTIC_ROLE_NEGATIVE_SCALE=0.20 \
  TTT_WRITE_POST_ZP_SUMMARY=1

run_job 5 V66B_P9_TTT_ANCHOR_FLOOR_96F hybrid \
  SEMANTIC_MEMORY_PATHS=ttt \
  ENABLE_SEMANTIC_ANCHOR_TTT_FLOOR=1 \
  SEMANTIC_ANCHOR_MODE=semantic \
  SEMANTIC_ANCHOR_TARGET_RATIO=0.18 \
  SEMANTIC_ANCHOR_MIN_RATIO=0.04 \
  SEMANTIC_ANCHOR_MAX_RATIO=0.35 \
  SEMANTIC_ROLE_POSITIVE_SCALE=1.20 \
  SEMANTIC_ROLE_NEUTRAL_SCALE=0.80 \
  SEMANTIC_ROLE_NEGATIVE_SCALE=0.40 \
  TTT_WRITE_POST_ZP_SUMMARY=1

wait

echo "[$(date '+%F %T')] all v66b phase9 continuation jobs finished"
