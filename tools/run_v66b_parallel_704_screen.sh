#!/usr/bin/env bash
set -euo pipefail

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
cd "$ROOT"

BASE="results/kitti01_hmc_v2/acl2_v66b_dense_semantic_scale/phase9_parallel_continuation/rollouts"
mkdir -p "$BASE"

COMMON_ENV=(
  ATTN_CUE_BASE="$BASE"
  END_FRAME=704
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

# 704F screen for the only TTT variant that reached the 20% write-action gate
# at 96F, plus its dense-cache ignore-semantic baseline.
run_job 0 V66B_P9_704_TTT_BASE_DENSE_IGNORE hybrid \
  HMC_IGNORE_SEMANTIC_PRIOR=1 \
  STAGE_C_INLINE_WHEN_IGNORED=1 \
  TTT_WRITE_POST_ZP_SUMMARY=1

run_job 1 V66B_P9_704_TTT_ROLE_EXTREME hybrid \
  SEMANTIC_MEMORY_PATHS=ttt \
  SEMANTIC_ROLE_POSITIVE_SCALE=1.50 \
  SEMANTIC_ROLE_NEUTRAL_SCALE=0.50 \
  SEMANTIC_ROLE_NEGATIVE_SCALE=0.00 \
  TTT_WRITE_POST_ZP_SUMMARY=1

# READ 704F screen: base, strongest 96F control, semantic anchor boost, and
# shuffled-anchor control to distinguish semantic value from perturbation value.
run_job 2 V66B_P9_704_READ_BASE_DENSE_IGNORE readonly \
  HMC_IGNORE_SEMANTIC_PRIOR=1 \
  STAGE_C_INLINE_WHEN_IGNORED=1

run_job 3 V66B_P9_704_READ_RANDOM_SAMEMASS_RHO050 readonly \
  SEMANTIC_MEMORY_PATHS=frame,global \
  ENABLE_CONTEXT_SOURCE_SKIP=1 \
  CONTEXT_SOURCE_SKIP_MASK=random_same_mass_semantic_role_negative \
  CONTEXT_SOURCE_SKIP_MODE=soft \
  CONTEXT_SOURCE_SKIP_IMPL=bias \
  CONTEXT_SOURCE_SKIP_SCOPE=frame \
  CONTEXT_SOURCE_SKIP_LAYER_MODE=early \
  CONTEXT_SOURCE_SKIP_SOFT_RHO=0.50 \
  CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP=0.25

run_job 4 V66B_P9_704_READ_ANCHOR_BOOST_RHO025 readonly \
  SEMANTIC_MEMORY_PATHS=frame,global \
  ENABLE_CONTEXT_SOURCE_SKIP=1 \
  CONTEXT_SOURCE_SKIP_MASK=semantic_anchor \
  CONTEXT_SOURCE_SKIP_MODE=boost \
  CONTEXT_SOURCE_SKIP_IMPL=bias_boost \
  CONTEXT_SOURCE_SKIP_SCOPE=frame \
  CONTEXT_SOURCE_SKIP_LAYER_MODE=early \
  CONTEXT_SOURCE_SKIP_SOFT_RHO=0.25 \
  SEMANTIC_ANCHOR_MODE=semantic

run_job 5 V66B_P9_704_READ_ANCHOR_BOOST_SHUFFLED readonly \
  SEMANTIC_MEMORY_PATHS=frame,global \
  ENABLE_CONTEXT_SOURCE_SKIP=1 \
  CONTEXT_SOURCE_SKIP_MASK=semantic_anchor \
  CONTEXT_SOURCE_SKIP_MODE=boost \
  CONTEXT_SOURCE_SKIP_IMPL=bias_boost \
  CONTEXT_SOURCE_SKIP_SCOPE=frame \
  CONTEXT_SOURCE_SKIP_LAYER_MODE=early \
  CONTEXT_SOURCE_SKIP_SOFT_RHO=0.25 \
  SEMANTIC_ANCHOR_MODE=shuffled_semantic

wait

echo "[$(date '+%F %T')] all v66b phase9 704F screen jobs finished"
