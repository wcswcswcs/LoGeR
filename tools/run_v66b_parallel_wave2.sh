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

# R1 anchor rescue / boost with semantic, random-same-mass, and shuffled
# anchor controls. This tests whether stable anchors help READ when harmful
# suppression keeps losing to random controls.
run_job 0 V66B_P9_READ_ANCHOR_BOOST_RHO025_96F readonly \
  SEMANTIC_MEMORY_PATHS=frame,global \
  ENABLE_CONTEXT_SOURCE_SKIP=1 \
  CONTEXT_SOURCE_SKIP_MASK=semantic_anchor \
  CONTEXT_SOURCE_SKIP_MODE=boost \
  CONTEXT_SOURCE_SKIP_IMPL=bias_boost \
  CONTEXT_SOURCE_SKIP_SCOPE=frame \
  CONTEXT_SOURCE_SKIP_LAYER_MODE=early \
  CONTEXT_SOURCE_SKIP_SOFT_RHO=0.25 \
  SEMANTIC_ANCHOR_MODE=semantic

run_job 1 V66B_P9_READ_ANCHOR_BOOST_RANDOM_96F readonly \
  SEMANTIC_MEMORY_PATHS=frame,global \
  ENABLE_CONTEXT_SOURCE_SKIP=1 \
  CONTEXT_SOURCE_SKIP_MASK=semantic_anchor \
  CONTEXT_SOURCE_SKIP_MODE=boost \
  CONTEXT_SOURCE_SKIP_IMPL=bias_boost \
  CONTEXT_SOURCE_SKIP_SCOPE=frame \
  CONTEXT_SOURCE_SKIP_LAYER_MODE=early \
  CONTEXT_SOURCE_SKIP_SOFT_RHO=0.25 \
  SEMANTIC_ANCHOR_MODE=random_same_mass

run_job 2 V66B_P9_READ_ANCHOR_BOOST_SHUFFLED_96F readonly \
  SEMANTIC_MEMORY_PATHS=frame,global \
  ENABLE_CONTEXT_SOURCE_SKIP=1 \
  CONTEXT_SOURCE_SKIP_MASK=semantic_anchor \
  CONTEXT_SOURCE_SKIP_MODE=boost \
  CONTEXT_SOURCE_SKIP_IMPL=bias_boost \
  CONTEXT_SOURCE_SKIP_SCOPE=frame \
  CONTEXT_SOURCE_SKIP_LAYER_MODE=early \
  CONTEXT_SOURCE_SKIP_SOFT_RHO=0.25 \
  SEMANTIC_ANCHOR_MODE=shuffled_semantic

# TTT self-check after the first wave: one stronger role-scale run to try to
# exceed the 20% write-prior action gate, plus random/shuffled anchor-floor
# controls for the semantic anchor floor path.
run_job 3 V66B_P9_TTT_ROLE_EXTREME_96F hybrid \
  SEMANTIC_MEMORY_PATHS=ttt \
  SEMANTIC_ROLE_POSITIVE_SCALE=1.50 \
  SEMANTIC_ROLE_NEUTRAL_SCALE=0.50 \
  SEMANTIC_ROLE_NEGATIVE_SCALE=0.00 \
  TTT_WRITE_POST_ZP_SUMMARY=1

run_job 4 V66B_P9_TTT_ANCHOR_FLOOR_RANDOM_96F hybrid \
  SEMANTIC_MEMORY_PATHS=ttt \
  ENABLE_SEMANTIC_ANCHOR_TTT_FLOOR=1 \
  SEMANTIC_ANCHOR_MODE=random_same_mass \
  SEMANTIC_ANCHOR_TARGET_RATIO=0.18 \
  SEMANTIC_ANCHOR_MIN_RATIO=0.04 \
  SEMANTIC_ANCHOR_MAX_RATIO=0.35 \
  SEMANTIC_ROLE_POSITIVE_SCALE=1.20 \
  SEMANTIC_ROLE_NEUTRAL_SCALE=0.80 \
  SEMANTIC_ROLE_NEGATIVE_SCALE=0.40 \
  TTT_WRITE_POST_ZP_SUMMARY=1

run_job 5 V66B_P9_TTT_ANCHOR_FLOOR_SHUFFLED_96F hybrid \
  SEMANTIC_MEMORY_PATHS=ttt \
  ENABLE_SEMANTIC_ANCHOR_TTT_FLOOR=1 \
  SEMANTIC_ANCHOR_MODE=shuffled_semantic \
  SEMANTIC_ANCHOR_TARGET_RATIO=0.18 \
  SEMANTIC_ANCHOR_MIN_RATIO=0.04 \
  SEMANTIC_ANCHOR_MAX_RATIO=0.35 \
  SEMANTIC_ROLE_POSITIVE_SCALE=1.20 \
  SEMANTIC_ROLE_NEUTRAL_SCALE=0.80 \
  SEMANTIC_ROLE_NEGATIVE_SCALE=0.40 \
  TTT_WRITE_POST_ZP_SUMMARY=1

wait

echo "[$(date '+%F %T')] all v66b phase9 wave2 jobs finished"
