#!/usr/bin/env bash
set -euo pipefail

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
cd "$ROOT"

BASE="${ATTN_CUE_BASE:-results/kitti01_hmc_v2/acl2_v66b_dense_semantic_scale/phase9_parallel_continuation/rollouts}"
mkdir -p "$BASE"
END_FRAME_VALUE="${END_FRAME:-704}"
RUN_PREFIX="${RUN_PREFIX:-V66B_P15_704_TTT_CAUSAL_SCALE_BROADCAST}"
SELECTOR_CHUNKS="${SELECTOR_CHUNKS:-3,4,5,6,7,16,17,20,21,22,23,26,27,33,36}"

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
  SEMANTIC_ROLE_POLICY=causal_ttt_full_role_tree
  SEMANTIC_MEMORY_PATHS=ttt
  SEMANTIC_ACTION_ACTIVE_CHUNKS="$SELECTOR_CHUNKS"
  SEMANTIC_CONDITION_SCALE_LEVEL=chunk_broadcast
  SEMANTIC_CONDITION_SCALE_SOURCE=v66b_phase14_selector_active_chunk_broadcast
  SEMANTIC_CONDITION_SCALE_VALUE=1.0
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
  local control_mode="$3"
  local pos="$4"
  local neu="$5"
  local neg="$6"
  (
    set -x
    env "${COMMON_ENV[@]}" \
      SEMANTIC_ROLE_CONTROL_MODE="$control_mode" \
      SEMANTIC_ROLE_CONTROL_SEED=12345 \
      SEMANTIC_ROLE_POSITIVE_SCALE="$pos" \
      SEMANTIC_ROLE_NEUTRAL_SCALE="$neu" \
      SEMANTIC_ROLE_NEGATIVE_SCALE="$neg" \
      bash tools/run_attention_cue_experiment.sh "$gpu" "$run_name" hybrid dyn 4.75 stage_d
  ) &
}

# Uses the existing causal_ttt_full_role_tree, but supplies an explicit
# chunk_broadcast scale-risk signal on Phase14 selector chunks.  This is not
# token-exact; the provenance fields must keep it diagnostic/audit-visible.
run_job 0 "${RUN_PREFIX}_SOFT_SEM" none 1.25 0.75 0.25
run_job 1 "${RUN_PREFIX}_SOFT_RANDOMCTRL_S12345" random_same_mass 1.25 0.75 0.25
run_job 2 "${RUN_PREFIX}_MID130_SEM" none 1.30 0.70 0.15
run_job 3 "${RUN_PREFIX}_MID130_RANDOMCTRL_S12345" random_same_mass 1.30 0.70 0.15
run_job 4 "${RUN_PREFIX}_MID140_SEM" none 1.40 0.60 0.05
run_job 5 "${RUN_PREFIX}_MID140_RANDOMCTRL_S12345" random_same_mass 1.40 0.60 0.05

wait

echo "[$(date '+%F %T')] all v66b causal-scale-broadcast TTT jobs finished"
