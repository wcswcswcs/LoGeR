#!/usr/bin/env bash
set -euo pipefail

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
cd "$ROOT"

BASE="${ATTN_CUE_BASE:-results/kitti01_hmc_v2/acl2_v66b_dense_semantic_scale/phase9_parallel_continuation/rollouts}"
mkdir -p "$BASE"
END_FRAME_VALUE="${END_FRAME:-704}"

ACTIVE_BALANCED="2,3,6,8,10,17,18,21,23"
ACTIVE_FUTURE="2,3,6,8,10,15,17,18,21,23,24"

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
  SEMANTIC_ROLE_POLICY=causal_fg_semantic_risk_skip
  SEMANTIC_MEMORY_PATHS=ttt
  SEMANTIC_ROLE_POSITIVE_SCALE=1.50
  SEMANTIC_ROLE_NEUTRAL_SCALE=0.50
  SEMANTIC_ROLE_NEGATIVE_SCALE=0.00
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
  local run_name="$2"
  local active_chunks="$3"
  local control_mode="$4"
  local control_seed="$5"
  (
    set -x
    env "${COMMON_ENV[@]}" \
      SEMANTIC_ACTION_ACTIVE_CHUNKS="$active_chunks" \
      SEMANTIC_ROLE_CONTROL_MODE="$control_mode" \
      SEMANTIC_ROLE_CONTROL_SEED="$control_seed" \
      bash tools/run_attention_cue_experiment.sh "$gpu" "$run_name" hybrid dyn 4.75 stage_d
  ) &
}

# Posthoc-guided targeted TTT semantic action. ACTIVE_BALANCED keeps chunks
# where role-extreme improved future-after-overlap and at least one local/
# scale/head-tail proxy. ACTIVE_FUTURE keeps every chunk with a positive
# future-after-overlap proxy, including chunks whose local/scale proxies were
# mixed. Both have same-path random/shuffled controls.
run_job 0 V66B_P9_704_TTT_ROLE_EXTREME_ACTIVEBALANCED "$ACTIVE_BALANCED" none 12345
run_job 1 V66B_P9_704_TTT_ROLE_EXTREME_ACTIVEBALANCED_RANDOMCTRL_S12345 "$ACTIVE_BALANCED" random_same_mass 12345
run_job 2 V66B_P9_704_TTT_ROLE_EXTREME_ACTIVEBALANCED_SHUFFLECTRL_S12345 "$ACTIVE_BALANCED" shuffled_semantic 12345
run_job 3 V66B_P9_704_TTT_ROLE_EXTREME_ACTIVEFUTURE "$ACTIVE_FUTURE" none 12345
run_job 4 V66B_P9_704_TTT_ROLE_EXTREME_ACTIVEFUTURE_RANDOMCTRL_S12345 "$ACTIVE_FUTURE" random_same_mass 12345
run_job 5 V66B_P9_704_TTT_ROLE_EXTREME_ACTIVEFUTURE_SHUFFLECTRL_S12345 "$ACTIVE_FUTURE" shuffled_semantic 12345

wait

echo "[$(date '+%F %T')] all v66b phase9 TTT active-chunk jobs finished"
