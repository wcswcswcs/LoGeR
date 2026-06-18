#!/usr/bin/env bash
set -euo pipefail

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
cd "$ROOT"

BASE="${ATTN_CUE_BASE:-results/kitti01_hmc_v2/acl2_v66b_dense_semantic_scale/phase9_parallel_continuation/rollouts}"
mkdir -p "$BASE"
END_FRAME_VALUE="${END_FRAME:-704}"
RUN_PREFIX="${RUN_PREFIX:-V66B_P16_704_TTT_CAUSAL_SCALE_SUBSET_DIAG}"

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
  SEMANTIC_CONDITION_SCALE_LEVEL=chunk_broadcast
  SEMANTIC_CONDITION_SCALE_SOURCE=v66b_phase16_phase15_posthoc_subset_chunk_broadcast
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
  local chunks="$3"
  local control_mode="$4"
  local pos="$5"
  local neu="$6"
  local neg="$7"
  (
    set -x
    env "${COMMON_ENV[@]}" \
      SEMANTIC_ACTION_ACTIVE_CHUNKS="$chunks" \
      SEMANTIC_ROLE_CONTROL_MODE="$control_mode" \
      SEMANTIC_ROLE_CONTROL_SEED=12345 \
      SEMANTIC_ROLE_POSITIVE_SCALE="$pos" \
      SEMANTIC_ROLE_NEUTRAL_SCALE="$neu" \
      SEMANTIC_ROLE_NEGATIVE_SCALE="$neg" \
      bash tools/run_attention_cue_experiment.sh "$gpu" "$run_name" hybrid dyn 4.75 stage_d
  ) &
}

# Diagnostic-only subsets selected from Phase15 posthoc chunk-effect analysis.
# These are not eligible for promotion without an independent rule/holdout.
run_job 0 "${RUN_PREFIX}_ROBUST41723_MID130_SEM" "4,17,23" none 1.30 0.70 0.15
run_job 1 "${RUN_PREFIX}_ROBUST41723_MID130_RANDOMCTRL_S12345" "4,17,23" random_same_mass 1.30 0.70 0.15
run_job 2 "${RUN_PREFIX}_MID130KEEP4172223_SEM" "4,17,22,23" none 1.30 0.70 0.15
run_job 3 "${RUN_PREFIX}_MID130KEEP4172223_RANDOMCTRL_S12345" "4,17,22,23" random_same_mass 1.30 0.70 0.15
run_job 4 "${RUN_PREFIX}_MID140KEEP523_SEM" "5,23" none 1.40 0.60 0.05
run_job 5 "${RUN_PREFIX}_MID140KEEP523_RANDOMCTRL_S12345" "5,23" random_same_mass 1.40 0.60 0.05

wait

echo "[$(date '+%F %T')] all v66b causal-scale subset diagnostic TTT jobs finished"
