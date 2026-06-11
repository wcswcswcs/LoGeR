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
V21_ROOT="${V21_ROOT:-results/kitti01_hmc_v2/acl2_v21_contextskip_semanticallmemory_ttt_persistence_target25}"
PHASE1="$V16_ROOT/phase1_causalfork"
ROLLOUT_BASE="$V21_ROOT/rollouts"
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

case "$CANDIDATE_ID" in
  K1_H9|S0_C23_PAST_LOCKED)
    READ_CUE="acl2.gg.qq.low.g2_3.past_only.headmean.robustq"
    ;;
  S1_C23_FULL_CHUNK_TRUE)
    READ_CUE="acl2.gg.qq.low.g2_3.full_chunk_true.headmean.robustq"
    ;;
  S2_C23_FULL_CHUNK_NO_OVERLAP_TRUE)
    READ_CUE="acl2.gg.qq.low.g2_3.full_chunk_no_overlap.headmean.robustq"
    ;;
  S3_C23_PAST_PLUS_NEAR_FUTURE12)
    READ_CUE="acl2.gg.qq.low.g2_3.past_plus_near_future12.headmean.robustq"
    ;;
  S4_C23_PAST_PLUS_FUTURE_LIGHT_REAL)
    READ_CUE="acl2.gg.qq.low.g2_3.past_plus_future_light_real.headmean.robustq"
    ;;
  S5_C23_PAST_PLUS_STATIC_FUTURE_ONLY)
    echo "S5_C23_PAST_PLUS_STATIC_FUTURE_ONLY is intentionally unsupported until static future support is implemented." >&2
    exit 2
    ;;
  KVC_01_FRAME_EARLY_DG_Q80_COMPACT)
    enable_compact_skip frame dg_q80 early
    ;;
  KVC_02_FRAME_EARLY_DG_Q90_COMPACT)
    enable_compact_skip frame dg_q90 early
    ;;
  KVC_03_FRAME_EARLY_LOWSTUFF_HIGHD_COMPACT)
    enable_exact_semantic_cache
    enable_compact_skip frame sem_lowstuff_highd early
    ;;
  KVC_04_GLOBAL_EARLY_DG_Q80_COMPACT)
    READ_PATH_VALUE="chunk"
    enable_compact_skip chunk dg_q80 early
    ;;
  KVC_05_FRAME_GLOBAL_EARLY_DG_Q80_COMPACT)
    enable_compact_skip both dg_q80 early
    ;;
  KVC_06_FRAME_EARLY_DG_Q80_BIAS_REPEAT)
    enable_bias_skip frame dg_q80 early
    ;;
  KVC_08_FRAME_EARLY_DG_Q80_COMPACT_WITH_STATIC_RESCUE)
    enable_exact_semantic_cache
    enable_compact_skip frame sem_structure_rescue_dg_q80 early
    ;;
  SEMFA_04_LOWSTUFF_HIGHD_FRAME_EARLY_COMPACT)
    enable_exact_semantic_cache
    enable_compact_skip frame sem_lowstuff_highd early
    ;;
  SEMFA_05_STRUCTURE_RESCUE_DGQ80_FRAME_EARLY_COMPACT)
    enable_exact_semantic_cache
    enable_compact_skip frame sem_structure_rescue_dg_q80 early
    ;;
  TTTSSP_01_SCALECOMMIT_DGQ80_COMPACT)
    enable_compact_skip frame dg_q80 early
    enable_scale_state_commit
    ;;
  TTTSSP_02_SCALECOMMIT_DGQ80_STRUCTURE_RESCUE_COMPACT)
    enable_exact_semantic_cache
    enable_compact_skip frame sem_structure_rescue_dg_q80 early
    enable_scale_state_commit
    ;;
  *)
    echo "Unsupported CANDIDATE_ID for v21 rollout: $CANDIDATE_ID" >&2
    exit 2
    ;;
esac

RUN_PREFIX="${RUN_PREFIX:-V21_A_SUPPORT_R1}"
RUN_NAME="${RUN_PREFIX}_${CANDIDATE_ID}_chunk${CHUNK_ID}_h${HORIZON}_globalgate_H9parent_SWKS3"
RUN_DIR="$ROOT/$ROLLOUT_BASE/$RUN_NAME"

if [ "${FORCE:-0}" != "1" ] && [ -f "$RUN_DIR/01.txt" ] && grep -q "DONE $RUN_NAME" "$RUN_DIR/run_status.txt" 2>/dev/null; then
  echo "SKIP existing DONE run: $RUN_NAME"
  exit 0
fi

mkdir -p "$RUN_DIR"
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
  ENABLE_SWA_OVERLAP_SOURCE_REPLACE=1 \
  SWA_OVERLAP_SOURCE_REPLACE_ALPHA=0.5 \
  SWA_OVERLAP_SOURCE_REPLACE_MODE=source \
  ENABLE_SWA_WRITE_CONTROL=1 \
  SWA_WRITE_LAYER_MODE=last \
  SWA_WRITE_KEEP_SCOPE=both_overlap \
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
  "$GPU" "$RUN_NAME" hybrid "$READ_CUE" "$BETA_VALUE" stage_d_x_dg_inv_sqrt
