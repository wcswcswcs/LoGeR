#!/usr/bin/env bash
set -euo pipefail

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
cd "$ROOT"

BASE="results/kitti01_hmc_v2/acl2_v66b_dense_semantic_scale/phase9_parallel_continuation/rollouts"
mkdir -p "$BASE"

COMMON_ENV=(
  ATTN_CUE_BASE="$BASE"
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
  local end_frame="$3"
  local control_mode="$4"
  local control_seed="$5"
  (
    set -x
    env "${COMMON_ENV[@]}" \
      END_FRAME="$end_frame" \
      SEMANTIC_ROLE_CONTROL_MODE="$control_mode" \
      SEMANTIC_ROLE_CONTROL_SEED="$control_seed" \
      bash tools/run_attention_cue_experiment.sh "$gpu" "$run_name" hybrid dyn 4.75 stage_d
  ) &
}

# Same-path controls for the Phase9 TTT role-extreme signal. 96F catches wiring
# bugs quickly; 704F tests the actual promotion-screen scope. Random controls
# keep the exact role mass, shuffled controls keep the role sequence but offset
# it away from its semantic patch positions.
run_job 0 V66B_P9_TTT_ROLE_EXTREME_RANDOMCTRL_S12345_96F 96 random_same_mass 12345
run_job 1 V66B_P9_TTT_ROLE_EXTREME_SHUFFLECTRL_S12345_96F 96 shuffled_semantic 12345
run_job 2 V66B_P9_704_TTT_ROLE_EXTREME_RANDOMCTRL_S12345 704 random_same_mass 12345
run_job 3 V66B_P9_704_TTT_ROLE_EXTREME_SHUFFLECTRL_S12345 704 shuffled_semantic 12345
run_job 4 V66B_P9_704_TTT_ROLE_EXTREME_RANDOMCTRL_S22345 704 random_same_mass 22345
run_job 5 V66B_P9_704_TTT_ROLE_EXTREME_SHUFFLECTRL_S22345 704 shuffled_semantic 22345

wait

echo "[$(date '+%F %T')] all v66b phase9 TTT role-control jobs finished"
