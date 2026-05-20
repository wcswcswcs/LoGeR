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
V20_ROOT="${V20_ROOT:-results/kitti01_hmc_v2/acl2_v20_ttt_contextskip_semanticmemory_target25}"
PHASE1="$V16_ROOT/phase1_causalfork"
ROLLOUT_BASE="$V20_ROOT/rollouts"

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
USES_CONTEXT_SKIP=false
USES_SEMANTIC_CACHE=false
ACTIVE_SCALE_CHUNKS="${TTT_WRITE_SCALE_STATE_CHUNKS:-$CHUNK_ID-$((CHUNK_ID + HORIZON))}"
BETA_VALUE="4.75"
READ_PATH_VALUE="${READ_PATH:-frame}"
CONTEXT_SOURCE_SKIP_ENABLE=0
CONTEXT_SOURCE_SKIP_SCOPE="frame"
CONTEXT_SOURCE_SKIP_MODE="hard"
CONTEXT_SOURCE_SKIP_MASK="dg_q90"
CONTEXT_SOURCE_SKIP_LAYER_MODE="early"
CONTEXT_SOURCE_SKIP_SOFT_RHO="0.5"
CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP="0.5"

case "$CANDIDATE_ID" in
  S1_00_C23_PAST|K1_H9)
    READ_CUE="acl2.gg.qq.low.g2_3.past_only.headmean.robustq"
    ;;
  S1_01_C23_FULL_CHUNK)
    READ_CUE="acl2.gg.qq.low.g2_3.full_chunk.headmean.robustq"
    ;;
  S1_02_C23_FULL_CHUNK_NO_OVERLAP)
    READ_CUE="acl2.gg.qq.low.g2_3.full_chunk_no_overlap.headmean.robustq"
    ;;
  S1_03_C23_PAST_PLUS_FUTURE_LIGHT)
    READ_CUE="acl2.gg.qq.low.g2_3.past_plus_future_light.headmean.robustq"
    ;;
  S1_04_C23_NEAR24)
    READ_CUE="acl2.gg.qq.low.g2_3.near24.headmean.robustq"
    ;;
  KVS_00_C23_PAST_PAIR)
    READ_CUE="acl2.gg.qq.low.g2_3.past_only.headmean.robustq"
    BETA_VALUE="4.75"
    ;;
  KVS_01_FRAME_EARLY_DG_Q80_HARD)
    READ_CUE="acl2.gg.qq.low.g2_3.past_only.headmean.robustq"
    BETA_VALUE="${KVS_SKIP_ONLY_BETA:-0.0}"
    USES_CONTEXT_SKIP=true
    CONTEXT_SOURCE_SKIP_ENABLE=1
    CONTEXT_SOURCE_SKIP_SCOPE="frame"
    CONTEXT_SOURCE_SKIP_MODE="hard"
    CONTEXT_SOURCE_SKIP_MASK="dg_q80"
    CONTEXT_SOURCE_SKIP_LAYER_MODE="early"
    ;;
  KVS_02_FRAME_EARLY_DG_Q90_HARD)
    READ_CUE="acl2.gg.qq.low.g2_3.past_only.headmean.robustq"
    BETA_VALUE="${KVS_SKIP_ONLY_BETA:-0.0}"
    USES_CONTEXT_SKIP=true
    CONTEXT_SOURCE_SKIP_ENABLE=1
    CONTEXT_SOURCE_SKIP_SCOPE="frame"
    CONTEXT_SOURCE_SKIP_MODE="hard"
    CONTEXT_SOURCE_SKIP_MASK="dg_q90"
    CONTEXT_SOURCE_SKIP_LAYER_MODE="early"
    ;;
  KVS_03_FRAME_EARLY_LOWSTUFF_HIGHD_HARD)
    READ_CUE="acl2.gg.qq.low.g2_3.past_only.headmean.robustq"
    BETA_VALUE="${KVS_SKIP_ONLY_BETA:-0.0}"
    USES_CONTEXT_SKIP=true
    CONTEXT_SOURCE_SKIP_ENABLE=1
    CONTEXT_SOURCE_SKIP_SCOPE="frame"
    CONTEXT_SOURCE_SKIP_MODE="hard"
    CONTEXT_SOURCE_SKIP_MASK="lowstuff_highd"
    CONTEXT_SOURCE_SKIP_LAYER_MODE="early"
    ;;
  KVS_06_FRAME_EARLY_DG_Q90_HARD_PLUS_PAIR)
    READ_CUE="acl2.gg.qq.low.g2_3.past_only.headmean.robustq"
    BETA_VALUE="4.75"
    USES_CONTEXT_SKIP=true
    CONTEXT_SOURCE_SKIP_ENABLE=1
    CONTEXT_SOURCE_SKIP_SCOPE="frame"
    CONTEXT_SOURCE_SKIP_MODE="hard"
    CONTEXT_SOURCE_SKIP_MASK="dg_q90"
    CONTEXT_SOURCE_SKIP_LAYER_MODE="early"
    ;;
  KVS_07_CHUNK_EARLY_DG_Q90_HARD)
    READ_CUE="acl2.gg.qq.low.g2_3.past_only.headmean.robustq"
    BETA_VALUE="${KVS_SKIP_ONLY_BETA:-0.0}"
    READ_PATH_VALUE="chunk"
    USES_CONTEXT_SKIP=true
    CONTEXT_SOURCE_SKIP_ENABLE=1
    CONTEXT_SOURCE_SKIP_SCOPE="chunk"
    CONTEXT_SOURCE_SKIP_MODE="hard"
    CONTEXT_SOURCE_SKIP_MASK="dg_q90"
    CONTEXT_SOURCE_SKIP_LAYER_MODE="early"
    ;;
  KVS_09_FRAME_EARLY_DG_Q90_SOFT_R025|KVS_09_FRAME_EARLY_DG_Q90_SOFT)
    READ_CUE="acl2.gg.qq.low.g2_3.past_only.headmean.robustq"
    BETA_VALUE="${KVS_SKIP_ONLY_BETA:-0.0}"
    USES_CONTEXT_SKIP=true
    CONTEXT_SOURCE_SKIP_ENABLE=1
    CONTEXT_SOURCE_SKIP_SCOPE="frame"
    CONTEXT_SOURCE_SKIP_MODE="soft"
    CONTEXT_SOURCE_SKIP_MASK="dg_q90"
    CONTEXT_SOURCE_SKIP_LAYER_MODE="early"
    CONTEXT_SOURCE_SKIP_SOFT_RHO="0.25"
    CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP="0.50"
    ;;
  SCALECOMMIT_01_PZBASIS_HARM_W0_G025)
    READ_CUE="acl2.gg.qq.low.g2_3.past_only.headmean.robustq"
    export TTT_WRITE_SCALE_STATE_MODE=projection_risk
    export TTT_WRITE_SCALE_STATE_PROXY=pose_step_ema
    export TTT_WRITE_SCALE_STATE_CARRIER=structure_lowdg
    export TTT_WRITE_SCALE_STATE_ALPHA=0.25
    export TTT_WRITE_SCALE_STATE_BRANCH_MASK=0
    export TTT_WRITE_SCALE_STATE_CHUNKS="$ACTIVE_SCALE_CHUNKS"
    export TTT_WRITE_NATIVE_DELTA_GATE_MODE=orthogonal_suppress
    export TTT_WRITE_NATIVE_DELTA_GATE_BRANCH_MASK=0
    GR_RISK_SOURCE="v19_scale_state"
    ;;
  TTTSS_03_SCALECOMMIT_DGQ90_HARD)
    READ_CUE="acl2.gg.qq.low.g2_3.past_only.headmean.robustq"
    BETA_VALUE="${KVS_SKIP_ONLY_BETA:-0.0}"
    USES_CONTEXT_SKIP=true
    CONTEXT_SOURCE_SKIP_ENABLE=1
    CONTEXT_SOURCE_SKIP_SCOPE="frame"
    CONTEXT_SOURCE_SKIP_MODE="hard"
    CONTEXT_SOURCE_SKIP_MASK="dg_q90"
    CONTEXT_SOURCE_SKIP_LAYER_MODE="early"
    export TTT_WRITE_SCALE_STATE_MODE=projection_risk
    export TTT_WRITE_SCALE_STATE_PROXY=pose_step_ema
    export TTT_WRITE_SCALE_STATE_CARRIER=structure_lowdg
    export TTT_WRITE_SCALE_STATE_ALPHA=0.25
    export TTT_WRITE_SCALE_STATE_BRANCH_MASK=0
    export TTT_WRITE_SCALE_STATE_CHUNKS="$ACTIVE_SCALE_CHUNKS"
    export TTT_WRITE_NATIVE_DELTA_GATE_MODE=orthogonal_suppress
    export TTT_WRITE_NATIVE_DELTA_GATE_BRANCH_MASK=0
    GR_RISK_SOURCE="v19_scale_state"
    ;;
  TTTSS_03B_SCALECOMMIT_DGQ80_HARD)
    READ_CUE="acl2.gg.qq.low.g2_3.past_only.headmean.robustq"
    BETA_VALUE="${KVS_SKIP_ONLY_BETA:-0.0}"
    USES_CONTEXT_SKIP=true
    CONTEXT_SOURCE_SKIP_ENABLE=1
    CONTEXT_SOURCE_SKIP_SCOPE="frame"
    CONTEXT_SOURCE_SKIP_MODE="hard"
    CONTEXT_SOURCE_SKIP_MASK="dg_q80"
    CONTEXT_SOURCE_SKIP_LAYER_MODE="early"
    export TTT_WRITE_SCALE_STATE_MODE=projection_risk
    export TTT_WRITE_SCALE_STATE_PROXY=pose_step_ema
    export TTT_WRITE_SCALE_STATE_CARRIER=structure_lowdg
    export TTT_WRITE_SCALE_STATE_ALPHA=0.25
    export TTT_WRITE_SCALE_STATE_BRANCH_MASK=0
    export TTT_WRITE_SCALE_STATE_CHUNKS="$ACTIVE_SCALE_CHUNKS"
    export TTT_WRITE_NATIVE_DELTA_GATE_MODE=orthogonal_suppress
    export TTT_WRITE_NATIVE_DELTA_GATE_BRANCH_MASK=0
    GR_RISK_SOURCE="v19_scale_state"
    ;;
  TTTSS_03C_SCALECOMMIT_DGQ80_HARD_PLUS_PAIR)
    READ_CUE="acl2.gg.qq.low.g2_3.past_only.headmean.robustq"
    BETA_VALUE="4.75"
    USES_CONTEXT_SKIP=true
    CONTEXT_SOURCE_SKIP_ENABLE=1
    CONTEXT_SOURCE_SKIP_SCOPE="frame"
    CONTEXT_SOURCE_SKIP_MODE="hard"
    CONTEXT_SOURCE_SKIP_MASK="dg_q80"
    CONTEXT_SOURCE_SKIP_LAYER_MODE="early"
    export TTT_WRITE_SCALE_STATE_MODE=projection_risk
    export TTT_WRITE_SCALE_STATE_PROXY=pose_step_ema
    export TTT_WRITE_SCALE_STATE_CARRIER=structure_lowdg
    export TTT_WRITE_SCALE_STATE_ALPHA=0.25
    export TTT_WRITE_SCALE_STATE_BRANCH_MASK=0
    export TTT_WRITE_SCALE_STATE_CHUNKS="$ACTIVE_SCALE_CHUNKS"
    export TTT_WRITE_NATIVE_DELTA_GATE_MODE=orthogonal_suppress
    export TTT_WRITE_NATIVE_DELTA_GATE_BRANCH_MASK=0
    GR_RISK_SOURCE="v19_scale_state"
    ;;
  *)
    echo "Unsupported CANDIDATE_ID for v20 rollout: $CANDIDATE_ID" >&2
    exit 2
    ;;
esac

RUN_PREFIX="${RUN_PREFIX:-V20_A_SUPPORT_R1}"
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
uses_context_skip: $USES_CONTEXT_SKIP
context_source_skip_scope: "$CONTEXT_SOURCE_SKIP_SCOPE"
context_source_skip_mode: "$CONTEXT_SOURCE_SKIP_MODE"
context_source_skip_mask: "$CONTEXT_SOURCE_SKIP_MASK"
context_source_skip_layer_mode: "$CONTEXT_SOURCE_SKIP_LAYER_MODE"
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
  ENABLE_CONTEXT_SOURCE_SKIP="$CONTEXT_SOURCE_SKIP_ENABLE" \
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
