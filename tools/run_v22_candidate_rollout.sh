#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 GPU CANDIDATE_ID CHUNK_ID HORIZON" >&2
  exit 2
fi

GPU="$1"
CANDIDATE_ID="$2"
CHUNK_ID="$3"
HORIZON="$4"

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
V16_ROOT="${V16_ROOT:-results/kitti01_hmc_v2/acl2_v16_ttt_causalfork_candidatebank_target25}"
V22_ROOT="${V22_ROOT:-results/kitti01_hmc_v2/acl2_v22_durable_contextskip_semanticallmemory_ttt_target25}"
PHASE1="$V16_ROOT/phase1_causalfork"
ROLLOUT_BASE="$V22_ROOT/rollouts"
STAGE_C_CACHE_DEFAULT="results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full"

case "$HORIZON" in
  3|5|8|10|15) ;;
  *) echo "Unsupported HORIZON: $HORIZON" >&2; exit 2 ;;
esac

case "$CHUNK_ID" in
  5) START_FRAME=145; SNAP="005" ;;
  6) START_FRAME=174; SNAP="006" ;;
  9) START_FRAME=261; SNAP="009" ;;
  10) START_FRAME=290; SNAP="010" ;;
  12) START_FRAME=348; SNAP="012" ;;
  16) START_FRAME=464; SNAP="016" ;;
  *) echo "Unsupported CHUNK_ID: $CHUNK_ID" >&2; exit 2 ;;
esac

END_FRAME=$((START_FRAME + 32 + HORIZON * 29))

BASE_GAMMAS="5:0.005,6:0.005,7:0.005,8:0.005,9:0.005,10:0.003,11:0.003,12:0.003,13:0.003,14:0.003,15:0.003,16:0.0003"
BASE_TRI_PARAMS="5:0.35/0.12/0.85,6:0.35/0.12/0.85,7:0.35/0.12/0.85,8:0.35/0.12/0.85,9:0.35/0.12/0.85,10:0.35/0.12/0.85,11:0.35/0.12/0.85,12:0.35/0.12/0.85,13:0.35/0.12/0.85,14:0.35/0.12/0.85,15:0.35/0.12/0.85,16:0.35/0.08/0.85"
CHUNK_GAMMAS="$BASE_GAMMAS"
GR_RISK_SOURCE="update_conflict_energy"
READ_CUE="acl2.gg.qq.low.g2_3.past_only.headmean.robustq"
BETA_VALUE="4.75"
READ_PATH_VALUE="${READ_PATH:-frame}"
USES_CONTEXT_SKIP=false
USES_SEMANTIC_CACHE=false
USES_TRUE_COMPACTION=false
USES_EXACT_SEMANTIC_GROUP=false
RUN_MODE="hybrid"
V22_FAMILY="unassigned"
MEMORY_ROLE_POLICY="baseline"
WRITE_SCORE_VALUE="${WRITE_SCORE_VALUE:-stage_d_x_dg_inv_sqrt}"
ACTIVE_SCALE_CHUNKS="${TTT_WRITE_SCALE_STATE_CHUNKS:-$CHUNK_ID-$((CHUNK_ID + HORIZON))}"

CONTEXT_SOURCE_SKIP_ENABLE=0
CONTEXT_SOURCE_SKIP_IMPL="bias"
CONTEXT_SOURCE_SKIP_SCOPE="frame"
CONTEXT_SOURCE_SKIP_MODE="hard"
CONTEXT_SOURCE_SKIP_MASK="dg_q90"
CONTEXT_SOURCE_SKIP_LAYER_MODE="early"
CONTEXT_SOURCE_SKIP_SOFT_RHO="0.5"
CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP="0.5"

STAGE_C_MODE_VALUE="${STAGE_C_MODE:-none}"
STAGE_C_CACHE_DIR_VALUE="${STAGE_C_CACHE_DIR:-}"
STAGE_C_CACHE_MODE_VALUE="${STAGE_C_CACHE_MODE:-off}"
STAGE_C_CACHE_REQUIRE_HIT_VALUE="${STAGE_C_CACHE_REQUIRE_HIT:-0}"
STAGE_C_CACHE_VALIDATE_VALUE="${STAGE_C_CACHE_VALIDATE:-0}"
SEMANTIC_PRIOR_MODE_VALUE="${SEMANTIC_PRIOR_MODE:-spg_v2}"

enable_exact_semantic_cache() {
  USES_SEMANTIC_CACHE=true
  USES_EXACT_SEMANTIC_GROUP=true
  STAGE_C_MODE_VALUE="${STAGE_C_MODE_VALUE:-reference}"
  if [ "$STAGE_C_MODE_VALUE" = "none" ]; then
    STAGE_C_MODE_VALUE="reference"
  fi
  STAGE_C_CACHE_DIR_VALUE="${STAGE_C_CACHE_DIR_VALUE:-$STAGE_C_CACHE_DEFAULT}"
  if [ -z "$STAGE_C_CACHE_MODE_VALUE" ] || [ "$STAGE_C_CACHE_MODE_VALUE" = "off" ]; then
    STAGE_C_CACHE_MODE_VALUE="read"
  fi
  if [ -z "$STAGE_C_CACHE_REQUIRE_HIT_VALUE" ] || [ "$STAGE_C_CACHE_REQUIRE_HIT_VALUE" = "0" ]; then
    STAGE_C_CACHE_REQUIRE_HIT_VALUE="1"
  fi
  STAGE_C_CACHE_VALIDATE_VALUE="${STAGE_C_CACHE_VALIDATE_VALUE:-0}"
}

enable_compact_skip() {
  USES_CONTEXT_SKIP=true
  USES_TRUE_COMPACTION=true
  CONTEXT_SOURCE_SKIP_ENABLE=1
  CONTEXT_SOURCE_SKIP_IMPL="compact_kv"
  CONTEXT_SOURCE_SKIP_SCOPE="$1"
  CONTEXT_SOURCE_SKIP_MASK="$2"
  CONTEXT_SOURCE_SKIP_LAYER_MODE="${3:-early}"
  BETA_VALUE="${KVS_SKIP_ONLY_BETA:-0.0}"
}

enable_bias_skip() {
  USES_CONTEXT_SKIP=true
  CONTEXT_SOURCE_SKIP_ENABLE=1
  CONTEXT_SOURCE_SKIP_IMPL="bias"
  CONTEXT_SOURCE_SKIP_SCOPE="$1"
  CONTEXT_SOURCE_SKIP_MASK="$2"
  CONTEXT_SOURCE_SKIP_LAYER_MODE="${3:-early}"
  BETA_VALUE="${KVS_SKIP_ONLY_BETA:-0.0}"
}

enable_scale_state_commit() {
  export TTT_WRITE_SCALE_STATE_MODE=projection_risk
  export TTT_WRITE_SCALE_STATE_PROXY=pose_step_ema
  export TTT_WRITE_SCALE_STATE_CARRIER=structure_lowdg
  export TTT_WRITE_SCALE_STATE_ALPHA="${TTT_WRITE_SCALE_STATE_ALPHA:-0.25}"
  export TTT_WRITE_SCALE_STATE_BRANCH_MASK="${TTT_WRITE_SCALE_STATE_BRANCH_MASK:-0}"
  export TTT_WRITE_SCALE_STATE_CHUNKS="$ACTIVE_SCALE_CHUNKS"
  export TTT_WRITE_NATIVE_DELTA_GATE_MODE=orthogonal_suppress
  export TTT_WRITE_NATIVE_DELTA_GATE_BRANCH_MASK=0
  GR_RISK_SOURCE="v19_scale_state"
}

set_support_cue() {
  case "$1" in
    locked|past_only)
      READ_CUE="acl2.gg.qq.low.g2_3.past_only.headmean.robustq"
      ;;
    full_true|full_chunk_true)
      READ_CUE="acl2.gg.qq.low.g2_3.full_chunk_true.headmean.robustq"
      ;;
    no_overlap|full_chunk_no_overlap)
      READ_CUE="acl2.gg.qq.low.g2_3.full_chunk_no_overlap.headmean.robustq"
      ;;
    near_future12|past_plus_near_future12)
      READ_CUE="acl2.gg.qq.low.g2_3.past_plus_near_future12.headmean.robustq"
      ;;
    future_light|past_plus_future_light_real)
      READ_CUE="acl2.gg.qq.low.g2_3.past_plus_future_light_real.headmean.robustq"
      ;;
    *)
      echo "Unsupported support cue alias: $1" >&2
      exit 2
      ;;
  esac
}

enable_skip_neutral_commit_filter() {
  export TTT_WRITE_COMMIT_FILTER_MODE=old_decay_by_risk
  export TTT_WRITE_COMMIT_FILTER_RISK_SOURCE=d_tok
  export TTT_WRITE_COMMIT_FILTER_SCOPE=both_overlap
  export TTT_WRITE_COMMIT_FILTER_STAT=q90
  export TTT_WRITE_COMMIT_FILTER_BASE="${TTT_WRITE_COMMIT_FILTER_BASE:-1.0}"
  export TTT_WRITE_COMMIT_FILTER_GAIN="${TTT_WRITE_COMMIT_FILTER_GAIN:-0.75}"
  export TTT_WRITE_COMMIT_FILTER_MIN="${TTT_WRITE_COMMIT_FILTER_MIN:-0.20}"
  export TTT_WRITE_COMMIT_FILTER_MAX="${TTT_WRITE_COMMIT_FILTER_MAX:-1.0}"
  export TTT_WRITE_COMMIT_FILTER_BRANCH_MASK="${TTT_WRITE_COMMIT_FILTER_BRANCH_MASK:-0}"
  export TTT_WRITE_COMMIT_FILTER_CHUNKS="$ACTIVE_SCALE_CHUNKS"
}

enable_swa_downweight_skipped() {
  export ENABLE_SWA_WRITE_CONTROL=1
  export SWA_WRITE_MODE="${SWA_WRITE_MODE:-kv}"
  export SWA_WRITE_RHO="${SWA_WRITE_RHO:-0.75}"
  export SWA_WRITE_MIN_GATE="${SWA_WRITE_MIN_GATE:-0.25}"
  export SWA_WRITE_SCOPE="${SWA_WRITE_SCOPE:-both_overlap}"
  export SWA_WRITE_KEEP_SCOPE="${SWA_WRITE_KEEP_SCOPE:-all}"
  export SWA_WRITE_SCORE_SOURCE="${SWA_WRITE_SCORE_SOURCE:-read}"
  export SWA_WRITE_LAYER_MODE="${SWA_WRITE_LAYER_MODE:-last}"
}

enable_swa_compact_overlap_history() {
  export ENABLE_SWA_WRITE_CONTROL=1
  export SWA_WRITE_MODE="${SWA_WRITE_MODE:-kv}"
  export SWA_WRITE_RHO="${SWA_WRITE_RHO:-0.65}"
  export SWA_WRITE_MIN_GATE="${SWA_WRITE_MIN_GATE:-0.20}"
  export SWA_WRITE_SCOPE="${SWA_WRITE_SCOPE:-both_overlap}"
  export SWA_WRITE_KEEP_SCOPE="${SWA_WRITE_KEEP_SCOPE:-exclude_both_overlap}"
  export SWA_WRITE_SCORE_SOURCE="${SWA_WRITE_SCORE_SOURCE:-read}"
  export SWA_WRITE_LAYER_MODE="${SWA_WRITE_LAYER_MODE:-last}"
}

enable_semantic_structure_write() {
  enable_exact_semantic_cache
  WRITE_SCORE_VALUE="stage_d_x_semantic_x_dg_inv_sqrt"
  export SPG_VALUE_STRUCTURE="${SPG_VALUE_STRUCTURE:-1.20}"
  export SPG_VALUE_BACKGROUND="${SPG_VALUE_BACKGROUND:-0.50}"
  export SPG_VALUE_DISTRACTOR="${SPG_VALUE_DISTRACTOR:-0.30}"
  export SPG_VALUE_MOVABLE="${SPG_VALUE_MOVABLE:-0.05}"
  export SPG_VALUE_UNCERTAIN="${SPG_VALUE_UNCERTAIN:-0.30}"
}

case "$CANDIDATE_ID" in
  K1_H9|S0_C23_PAST_LOCKED|SUP_LOCKED_A)
    set_support_cue locked
    V22_FAMILY="support_setting_A"
    ;;
  S1_C23_FULL_CHUNK_TRUE|SUP_FULL_TRUE_A)
    set_support_cue full_true
    V22_FAMILY="support_setting_A"
    ;;
  S2_C23_FULL_CHUNK_NO_OVERLAP_TRUE|SUP_NO_OVERLAP_A)
    set_support_cue no_overlap
    V22_FAMILY="support_setting_A"
    ;;
  S3_C23_PAST_PLUS_NEAR_FUTURE12|SUP_PAST_NEAR_FUTURE12_A)
    set_support_cue near_future12
    V22_FAMILY="support_setting_A"
    ;;
  S4_C23_PAST_PLUS_FUTURE_LIGHT_REAL)
    set_support_cue future_light
    V22_FAMILY="support_setting_A"
    ;;
  SUP_LOCKED_B_DGQ80_COMPACT)
    set_support_cue locked
    enable_compact_skip frame dg_q80 early
    V22_FAMILY="support_setting_B_compact"
    ;;
  SUP_FULL_TRUE_B_DGQ80_COMPACT)
    set_support_cue full_true
    enable_compact_skip frame dg_q80 early
    V22_FAMILY="support_setting_B_compact"
    ;;
  SUP_NO_OVERLAP_B_DGQ80_COMPACT)
    set_support_cue no_overlap
    enable_compact_skip frame dg_q80 early
    V22_FAMILY="support_setting_B_compact"
    ;;
  SUP_PAST_NEAR_FUTURE12_B_DGQ80_COMPACT)
    set_support_cue near_future12
    enable_compact_skip frame dg_q80 early
    V22_FAMILY="support_setting_B_compact"
    ;;
  SUP_LOCKED_C_STRUCTURE_RESCUE_COMPACT)
    set_support_cue locked
    enable_exact_semantic_cache
    enable_compact_skip frame sem_structure_rescue_dg_q80 early
    V22_FAMILY="support_setting_C_semantic_rescue"
    ;;
  SUP_FULL_TRUE_C_STRUCTURE_RESCUE_COMPACT)
    set_support_cue full_true
    enable_exact_semantic_cache
    enable_compact_skip frame sem_structure_rescue_dg_q80 early
    V22_FAMILY="support_setting_C_semantic_rescue"
    ;;
  SUP_NO_OVERLAP_C_STRUCTURE_RESCUE_COMPACT)
    set_support_cue no_overlap
    enable_exact_semantic_cache
    enable_compact_skip frame sem_structure_rescue_dg_q80 early
    V22_FAMILY="support_setting_C_semantic_rescue"
    ;;
  SUP_PAST_NEAR_FUTURE12_C_STRUCTURE_RESCUE_COMPACT)
    set_support_cue near_future12
    enable_exact_semantic_cache
    enable_compact_skip frame sem_structure_rescue_dg_q80 early
    V22_FAMILY="support_setting_C_semantic_rescue"
    ;;
  S5_C23_PAST_PLUS_STATIC_FUTURE_ONLY)
    echo "S5_C23_PAST_PLUS_STATIC_FUTURE_ONLY is intentionally unsupported until static future support is implemented." >&2
    exit 2
    ;;
  KVC_01_FRAME_EARLY_DG_Q80_COMPACT|KVC_READ_01)
    RUN_MODE="readonly"
    enable_compact_skip frame dg_q80 early
    V22_FAMILY="read_only_compact_kv"
    ;;
  KVC_02_FRAME_EARLY_DG_Q90_COMPACT|KVC_READ_02)
    RUN_MODE="readonly"
    enable_compact_skip frame dg_q90 early
    V22_FAMILY="read_only_compact_kv"
    ;;
  KVC_03_FRAME_EARLY_LOWSTUFF_HIGHD_COMPACT|SEM_ROLE_02_LOWSTUFF_HIGHD_SKIP)
    enable_exact_semantic_cache
    enable_compact_skip frame sem_lowstuff_highd early
    V22_FAMILY="semantic_coarse_role"
    MEMORY_ROLE_POLICY="lowstuff_highD_source_skip"
    ;;
  KVC_04_GLOBAL_EARLY_DG_Q80_COMPACT|KVC_READ_03)
    RUN_MODE="readonly"
    READ_PATH_VALUE="chunk"
    enable_compact_skip chunk dg_q80 early
    V22_FAMILY="read_only_compact_kv"
    ;;
  KVC_05_FRAME_GLOBAL_EARLY_DG_Q80_COMPACT|KVC_READ_04)
    RUN_MODE="readonly"
    enable_compact_skip both dg_q80 early
    V22_FAMILY="read_only_compact_kv"
    ;;
  KVC_06_FRAME_EARLY_DG_Q80_BIAS_REPEAT)
    enable_bias_skip frame dg_q80 early
    ;;
  KVC_08_FRAME_EARLY_DG_Q80_COMPACT_WITH_STATIC_RESCUE|SEM_ROLE_01_STRUCTURE_RESCUE)
    enable_exact_semantic_cache
    enable_compact_skip frame sem_structure_rescue_dg_q80 early
    V22_FAMILY="semantic_coarse_role"
    MEMORY_ROLE_POLICY="structure_rescue_read_source"
    ;;
  SEMFA_04_LOWSTUFF_HIGHD_FRAME_EARLY_COMPACT)
    enable_exact_semantic_cache
    enable_compact_skip frame sem_lowstuff_highd early
    ;;
  SEMFA_05_STRUCTURE_RESCUE_DGQ80_FRAME_EARLY_COMPACT)
    enable_exact_semantic_cache
    enable_compact_skip frame sem_structure_rescue_dg_q80 early
    ;;
  SEM_ROLE_04_STRUCTURE_POSITIVE_TTT)
    enable_semantic_structure_write
    enable_compact_skip frame sem_structure_rescue_dg_q80 early
    enable_skip_neutral_commit_filter
    V22_FAMILY="semantic_all_memory_role"
    MEMORY_ROLE_POLICY="structure_lowD_positive_lowstuff_highD_neutral"
    ;;
  SEM_ROLE_05_ALL_MEMORY_ROLE)
    enable_semantic_structure_write
    enable_compact_skip both sem_structure_rescue_dg_q80 early
    enable_skip_neutral_commit_filter
    enable_swa_downweight_skipped
    V22_FAMILY="semantic_all_memory_role"
    MEMORY_ROLE_POLICY="frame_global_swa_ttt_role_sync"
    ;;
  TTTSSP_01_SCALECOMMIT_DGQ80_COMPACT)
    enable_compact_skip frame dg_q80 early
    enable_scale_state_commit
    V22_FAMILY="scale_state_persistence"
    ;;
  TTTSSP_02_SCALECOMMIT_DGQ80_STRUCTURE_RESCUE_COMPACT)
    enable_exact_semantic_cache
    enable_compact_skip frame sem_structure_rescue_dg_q80 early
    enable_scale_state_commit
    V22_FAMILY="scale_state_persistence"
    MEMORY_ROLE_POLICY="v21_strongest_scale_state_semantic_rescue"
    ;;
  KVC_TTT_01_NEUTRAL_COMMIT_FILTER)
    enable_compact_skip frame dg_q80 early
    enable_skip_neutral_commit_filter
    V22_FAMILY="skip_aware_ttt_write"
    MEMORY_ROLE_POLICY="highD_overlap_commit_neutralized"
    ;;
  KVC_TTT_02_WEAK_NEGATIVE)
    enable_compact_skip frame dg_q80 early
    GR_RISK_SOURCE="d_tok"
    CHUNK_GAMMAS="10:0.0015,11:0.0015,12:0.0015,13:0.0015,14:0.0015,15:0.0015,16:0.0015,17:0.0015,18:0.0015,19:0.0015,20:0.0015,21:0.001,22:0.001,23:0.001,24:0.001,25:0.001"
    export TTT_WRITE_TRI_REPLAY_NEGATIVE_FRAC=0.08
    V22_FAMILY="skip_aware_ttt_write"
    MEMORY_ROLE_POLICY="skipped_highD_weak_negative"
    ;;
  KVC_TTT_03_STRUCTURE_KEPT_BOOST)
    enable_semantic_structure_write
    enable_compact_skip frame sem_structure_rescue_dg_q80 early
    enable_skip_neutral_commit_filter
    V22_FAMILY="skip_aware_ttt_write"
    MEMORY_ROLE_POLICY="structure_kept_positive_highD_neutral"
    ;;
  KVC_TTT_04_SOURCE_KEEP_GATED_WRITE)
    enable_compact_skip frame dg_q80 early
    export TTT_WRITE_REPLAY_TOKEN_FILTER_MODE=scoped_dynamic_veto
    export TTT_WRITE_REPLAY_TOKEN_FILTER_THRESHOLD=0.35
    export TTT_WRITE_REPLAY_TOKEN_FILTER_SCOPE=both_overlap
    export TTT_WRITE_REPLAY_TOKEN_FILTER_BRANCH_MASK=0
    export TTT_WRITE_REPLAY_TOKEN_FILTER_BLEND=0.50
    export TTT_WRITE_REPLAY_TOKEN_FILTER_BLEND_MODE=ttl_aligned_dynamic
    export TTT_WRITE_TRANSIENT_DELTA_TTL=2
    V22_FAMILY="skip_aware_ttt_write"
    MEMORY_ROLE_POLICY="overlap_highD_replay_token_filter"
    ;;
  KVC_MEM_01_SWA_COMPACT_OVERLAP_HISTORY)
    enable_compact_skip frame dg_q80 early
    enable_swa_compact_overlap_history
    V22_FAMILY="skip_aware_swa_memory"
    MEMORY_ROLE_POLICY="swa_overlap_history_compacted"
    ;;
  KVC_MEM_02_SWA_DOWNWEIGHT_SKIPPED)
    enable_compact_skip frame dg_q80 early
    enable_swa_downweight_skipped
    V22_FAMILY="skip_aware_swa_memory"
    MEMORY_ROLE_POLICY="swa_highD_downweighted"
    ;;
  KVC_MEM_03_GLOBAL_CHUNK_SOURCE_SKIP)
    enable_compact_skip both dg_q80 early
    READ_PATH_VALUE="chunk"
    V22_FAMILY="skip_aware_global_source"
    MEMORY_ROLE_POLICY="frame_chunk_read_source_compact"
    ;;
  KVC_MEM_04_TTT_AND_SWA_DOWNWEIGHT)
    enable_compact_skip frame dg_q80 early
    enable_skip_neutral_commit_filter
    enable_swa_downweight_skipped
    V22_FAMILY="skip_aware_swa_memory"
    MEMORY_ROLE_POLICY="ttt_commit_filter_plus_swa_downweight"
    ;;
  TTT_DUR_01_READ_COMPACT_ONLY)
    RUN_MODE="readonly"
    enable_compact_skip frame dg_q80 early
    V22_FAMILY="ttt_durability_control"
    MEMORY_ROLE_POLICY="read_only_compact_control"
    ;;
  TTT_DUR_02_SKIP_AWARE_COMMIT_FILTER)
    enable_compact_skip frame dg_q80 early
    enable_skip_neutral_commit_filter
    V22_FAMILY="ttt_durable_commit"
    MEMORY_ROLE_POLICY="skip_aware_long_commit_filter"
    ;;
  TTT_DUR_03_NATIVE_READ_SKIP_REPLAY_ONLY)
    enable_compact_skip frame dg_q80 early
    export TTT_WRITE_REPLAY_TOKEN_FILTER_MODE=scoped_static_topk
    export TTT_WRITE_REPLAY_TOKEN_FILTER_RATIO=0.70
    export TTT_WRITE_REPLAY_TOKEN_FILTER_SCOPE=both_overlap
    export TTT_WRITE_REPLAY_TOKEN_FILTER_BRANCH_MASK=0
    export TTT_WRITE_NATIVE_MIX_SCALES="1.00,1.00,1.00"
    V22_FAMILY="ttt_durable_commit"
    MEMORY_ROLE_POLICY="native_read_with_skip_filtered_replay"
    ;;
  TTT_DUR_04_POST_ZP_SKIP_BASIS_ROUTING)
    enable_compact_skip frame dg_q80 early
    export TTT_WRITE_REPLAY_TOKEN_FILTER_MODE=scoped_dynamic_veto
    export TTT_WRITE_REPLAY_TOKEN_FILTER_THRESHOLD=0.35
    export TTT_WRITE_REPLAY_TOKEN_FILTER_SCOPE=both_overlap
    export TTT_WRITE_REPLAY_TOKEN_FILTER_BRANCH_MASK=0
    export TTT_WRITE_REPLAY_TOKEN_FILTER_BLEND=0.45
    export TTT_WRITE_REPLAY_TOKEN_FILTER_BLEND_MODE=project_anti_dynamic
    V22_FAMILY="ttt_durable_commit"
    MEMORY_ROLE_POLICY="post_zp_skip_basis_routing"
    ;;
  TTT_LIFE_01_SHORT_HIGHD_K2)
    enable_compact_skip frame dg_q80 early
    GR_RISK_SOURCE="d_tok"
    export TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_MODE=dual_lifetime
    export TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_BRANCH_MASK=0
    export TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_LONG_SCALE=0.35
    export TTT_WRITE_TRANSIENT_DELTA_TTL=2
    V22_FAMILY="ttt_lifecycle_split"
    MEMORY_ROLE_POLICY="highD_short_life_K2"
    ;;
  TTT_LIFE_02_SHORT_HIGHD_K4)
    enable_compact_skip frame dg_q80 early
    GR_RISK_SOURCE="d_tok"
    export TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_MODE=dual_lifetime
    export TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_BRANCH_MASK=0
    export TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_LONG_SCALE=0.35
    export TTT_WRITE_TRANSIENT_DELTA_TTL=4
    V22_FAMILY="ttt_lifecycle_split"
    MEMORY_ROLE_POLICY="highD_short_life_K4"
    ;;
  TTT_LIFE_03_LOWSTUFF_SHORT_STRUCTURE_LONG)
    enable_semantic_structure_write
    enable_compact_skip frame sem_lowstuff_highd early
    GR_RISK_SOURCE="d_tok"
    export TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_MODE=dual_lifetime
    export TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_BRANCH_MASK=0
    export TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_LONG_SCALE=0.45
    export TTT_WRITE_TRANSIENT_DELTA_TTL=3
    V22_FAMILY="ttt_lifecycle_split"
    MEMORY_ROLE_POLICY="lowstuff_highD_short_structure_long"
    ;;
  TTT_LIFE_04_SCALE_LONG_HIGHD_SHORT)
    enable_exact_semantic_cache
    enable_compact_skip frame sem_structure_rescue_dg_q80 early
    enable_scale_state_commit
    export TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_MODE=dual_lifetime
    export TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_BRANCH_MASK=0
    export TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_LONG_SCALE=0.60
    export TTT_WRITE_TRANSIENT_DELTA_TTL=3
    V22_FAMILY="ttt_lifecycle_split"
    MEMORY_ROLE_POLICY="scale_state_long_highD_short"
    ;;
  *)
    echo "Unsupported CANDIDATE_ID for v22 rollout: $CANDIDATE_ID" >&2
    exit 2
    ;;
esac

RUN_PREFIX="${RUN_PREFIX:-V22_A_SUPPORT_R1}"
RUN_NAME="${RUN_PREFIX}_${CANDIDATE_ID}_chunk${CHUNK_ID}_h${HORIZON}_globalgate_H9parent_SWKS3"
RUN_DIR="$ROOT/$ROLLOUT_BASE/$RUN_NAME"

if [ -d "$RUN_DIR" ]; then
  if [ "${FORCE:-0}" != "1" ] && [ -f "$RUN_DIR/01.txt" ] && grep -q "DONE $RUN_NAME" "$RUN_DIR/run_status.txt" 2>/dev/null; then
    echo "SKIP existing DONE run: $RUN_NAME"
    exit 0
  fi
  STAMP="$(date '+%Y%m%d_%H%M%S')"
  INVALID_DIR="${RUN_DIR}.INVALID_RERUN_${STAMP}"
  mv "$RUN_DIR" "$INVALID_DIR"
  echo "Moved stale/forced run directory to: $INVALID_DIR"
fi

mkdir -p "$RUN_DIR"
if [ "${V22_SAVE_ATTRIBUTION_STATES:-0}" = "1" ]; then
  ATTR_CHUNKS="$CHUNK_ID,$((CHUNK_ID + 10)),$((CHUNK_ID + HORIZON))"
  export SAVE_HMC_STATES="${SAVE_HMC_STATES:-$RUN_DIR/hmc_state_snapshots}"
  export SAVE_HMC_STATE_CHUNKS="${SAVE_HMC_STATE_CHUNKS:-$ATTR_CHUNKS}"
  export SAVE_HMC_STATE_KINDS="${SAVE_HMC_STATE_KINDS:-input,after}"
  export SAVE_MERGE_STATES="${SAVE_MERGE_STATES:-$RUN_DIR/merge_state_snapshots}"
  export SAVE_MERGE_STATE_CHUNKS="${SAVE_MERGE_STATE_CHUNKS:-$ATTR_CHUNKS}"
  export SAVE_MERGE_STATE_KINDS="${SAVE_MERGE_STATE_KINDS:-input,after}"
fi
cat > "$RUN_DIR/run_config.yaml" <<EOF
run_name: "$RUN_NAME"
candidate_id: "$CANDIDATE_ID"
chunk_id: $CHUNK_ID
horizon: $HORIZON
start_frame: $START_FRAME
end_frame: $END_FRAME
read_cue: "$READ_CUE"
read_mode: "frame pair/all"
read_path: "$READ_PATH_VALUE"
beta: $BETA_VALUE
parent: "H9_P0_V16_R2 causal fork snapshots"
diagnostic_only_short_rollout: true
counts_as_online_ttt_write_success: false
uses_gt_runtime_action: false
uses_offline_postprocess: false
uses_semantic_cache: $USES_SEMANTIC_CACHE
uses_exact_semantic_group: $USES_EXACT_SEMANTIC_GROUP
semantic_group_taxonomy: "stage_c_coarse_5_groups"
fine_sky_vegetation_available: false
uses_context_skip: $USES_CONTEXT_SKIP
uses_true_kv_compaction: $USES_TRUE_COMPACTION
context_source_skip_impl: "$CONTEXT_SOURCE_SKIP_IMPL"
context_source_skip_scope: "$CONTEXT_SOURCE_SKIP_SCOPE"
context_source_skip_mode: "$CONTEXT_SOURCE_SKIP_MODE"
context_source_skip_mask: "$CONTEXT_SOURCE_SKIP_MASK"
context_source_skip_layer_mode: "$CONTEXT_SOURCE_SKIP_LAYER_MODE"
stage_c_mode: "$STAGE_C_MODE_VALUE"
stage_c_cache_dir: "$STAGE_C_CACHE_DIR_VALUE"
stage_c_cache_mode: "$STAGE_C_CACHE_MODE_VALUE"
stage_c_cache_require_hit: "$STAGE_C_CACHE_REQUIRE_HIT_VALUE"
semantic_prior_mode: "$SEMANTIC_PRIOR_MODE_VALUE"
v22_family: "$V22_FAMILY"
memory_role_policy: "$MEMORY_ROLE_POLICY"
run_mode: "$RUN_MODE"
write_score: "$WRITE_SCORE_VALUE"
EOF

env \
  KITTI_SEQ=01 \
  ATTN_CUE_BASE="$ROLLOUT_BASE" \
  START_FRAME="$START_FRAME" \
  END_FRAME="$END_FRAME" \
  GLOBAL_CHUNK_OFFSET="$CHUNK_ID" \
  RESET_EVERY=5 \
  WRITE_ALPHA=0.125 \
  READ_PATH="$READ_PATH_VALUE" \
  READ_LAYER_MODE=all \
  STAGE_C_MODE="$STAGE_C_MODE_VALUE" \
  STAGE_C_CACHE_DIR="$STAGE_C_CACHE_DIR_VALUE" \
  STAGE_C_CACHE_MODE="$STAGE_C_CACHE_MODE_VALUE" \
  STAGE_C_CACHE_REQUIRE_HIT="$STAGE_C_CACHE_REQUIRE_HIT_VALUE" \
  STAGE_C_CACHE_VALIDATE="$STAGE_C_CACHE_VALIDATE_VALUE" \
  SEMANTIC_PRIOR_MODE="$SEMANTIC_PRIOR_MODE_VALUE" \
  ENABLE_CONTEXT_SOURCE_SKIP="$CONTEXT_SOURCE_SKIP_ENABLE" \
  CONTEXT_SOURCE_SKIP_IMPL="$CONTEXT_SOURCE_SKIP_IMPL" \
  CONTEXT_SOURCE_SKIP_SCOPE="$CONTEXT_SOURCE_SKIP_SCOPE" \
  CONTEXT_SOURCE_SKIP_MODE="$CONTEXT_SOURCE_SKIP_MODE" \
  CONTEXT_SOURCE_SKIP_MASK="$CONTEXT_SOURCE_SKIP_MASK" \
  CONTEXT_SOURCE_SKIP_LAYER_MODE="$CONTEXT_SOURCE_SKIP_LAYER_MODE" \
  CONTEXT_SOURCE_SKIP_SOFT_RHO="$CONTEXT_SOURCE_SKIP_SOFT_RHO" \
  CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP="$CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP" \
  ENABLE_SWA_OVERLAP_SOURCE_REPLACE="${ENABLE_SWA_OVERLAP_SOURCE_REPLACE:-1}" \
  SWA_OVERLAP_SOURCE_REPLACE_ALPHA="${SWA_OVERLAP_SOURCE_REPLACE_ALPHA:-0.5}" \
  SWA_OVERLAP_SOURCE_REPLACE_MODE="${SWA_OVERLAP_SOURCE_REPLACE_MODE:-source}" \
  ENABLE_SWA_WRITE_CONTROL="${ENABLE_SWA_WRITE_CONTROL:-1}" \
  SWA_WRITE_MODE="${SWA_WRITE_MODE:-none}" \
  SWA_WRITE_RHO="${SWA_WRITE_RHO:-0.0}" \
  SWA_WRITE_MIN_GATE="${SWA_WRITE_MIN_GATE:-0.0}" \
  SWA_WRITE_SCOPE="${SWA_WRITE_SCOPE:-all}" \
  SWA_WRITE_LAYER_MODE="${SWA_WRITE_LAYER_MODE:-last}" \
  SWA_WRITE_KEEP_SCOPE="${SWA_WRITE_KEEP_SCOPE:-both_overlap}" \
  SWA_WRITE_SCORE_SOURCE="${SWA_WRITE_SCORE_SOURCE:-read}" \
  TTT_WRITE_GRADIENT_REVERSAL_MODE=tri_replay \
  TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE="$GR_RISK_SOURCE" \
  TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS="$CHUNK_GAMMAS" \
  TTT_WRITE_NATIVE_MIX_SCALES="${TTT_WRITE_NATIVE_MIX_SCALES:-1.10,1.00,1.00}" \
  TTT_WRITE_TRI_REPLAY_POSITIVE_FRAC=0.35 \
  TTT_WRITE_TRI_REPLAY_NEGATIVE_FRAC=0.12 \
  TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA=0.85 \
  TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS="$BASE_TRI_PARAMS" \
  LOAD_HMC_STATE_AT_CHUNK="$PHASE1/state_snapshots/H9_P0_V16_R2/chunk_${SNAP}_input.pt" \
  LOAD_HMC_STATE_AT_CHUNK_INDEX=0 \
  LOAD_MERGE_STATE_AT_CHUNK="$PHASE1/merge_state_snapshots/H9_P0_V16_R2/chunk_${SNAP}_input.pt" \
  LOAD_MERGE_STATE_AT_CHUNK_INDEX=0 \
  "$ROOT/tools/run_attention_cue_experiment.sh" \
  "$GPU" "$RUN_NAME" "$RUN_MODE" "$READ_CUE" "$BETA_VALUE" "$WRITE_SCORE_VALUE"
