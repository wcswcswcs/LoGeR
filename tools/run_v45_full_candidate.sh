#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 GPU RUN_NAME CANDIDATE_ID" >&2
  echo "CANDIDATE_ID: P0_C9_REPEAT | D0..D7 | I1..I8 | S0..S5 | A0..A4 | SEM1..SEM4" >&2
  exit 2
fi

GPU="$1"
RUN_NAME="$2"
CANDIDATE_ID="$3"

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
RESULT_ROOT="${V45_RESULT_ROOT:-$ROOT/results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay}"
case "$CANDIDATE_ID" in
  P0_*|D0_*) DEFAULT_PHASE="phase0_hard_gate" ;;
  D*) DEFAULT_PHASE="phase1_c9_clean" ;;
  I*) DEFAULT_PHASE="phase2_interaction" ;;
  S*) DEFAULT_PHASE="phase3_c23_support" ;;
  A*) DEFAULT_PHASE="phase4_adaptive_trireplay" ;;
  SEM*) DEFAULT_PHASE="phase5_semantic_minimal" ;;
  XS*) DEFAULT_PHASE="phase6_cross_sequence" ;;
  SMOKE*) DEFAULT_PHASE="phase0_smoke" ;;
  *) DEFAULT_PHASE="misc" ;;
esac
BASE="${V45_ROLLOUT_BASE:-$RESULT_ROOT/$DEFAULT_PHASE/rollouts}"
OUT="$BASE/$RUN_NAME"
C9_CUE="acl2.gg.qq.low.g2_3.past_only.headmean.robustq"

if [ -f "$OUT/run_status.txt" ] && grep -q "DONE $RUN_NAME" "$OUT/run_status.txt" && [ -s "$OUT/${KITTI_SEQ:-01}.txt" ]; then
  echo "[$(date '+%F %T')] SKIP existing DONE $RUN_NAME"
  exit 0
fi

MODE="${V45_MODE_OVERRIDE:-hybrid}"
CUE="$C9_CUE"
BETA="4.75"
WRITE_SCORE="stage_d_x_dg_inv_sqrt"

export ATTN_CUE_BASE="$BASE"
export START_FRAME="${START_FRAME:-0}"
export END_FRAME="${END_FRAME:-10000}"
export RESET_EVERY="${RESET_EVERY:-5}"
export FAST_CUE_EVAL="${FAST_CUE_EVAL:-1}"
export READ_PATH="${READ_PATH:-frame}"
export READ_LAYER_MODE="${READ_LAYER_MODE:-all}"
export BETA_SWA="${BETA_SWA:-4.75}"
export READ_BETA_FRAME_CHUNKS="${READ_BETA_FRAME_CHUNKS:-5:4.85,6:4.85,7:4.85,8:4.85,9:4.85,10:4.25,11:4.25,12:4.25,16:4.25}"
export WRITE_ALPHA="${WRITE_ALPHA:-0.1}"
export WRITE_MIN="${WRITE_MIN:-0.8}"
export WRITE_MAX="${WRITE_MAX:-1.2}"
export MP_SCORE_SOURCE="${MP_SCORE_SOURCE:-dyn}"
export HMC_COMMIT_MODE="${HMC_COMMIT_MODE:-probe_ttt_write}"
export ENABLE_SWA_WRITE_CONTROL="${ENABLE_SWA_WRITE_CONTROL:-1}"
export SWA_WRITE_MODE="${SWA_WRITE_MODE:-none}"
export SWA_WRITE_LAYER_MODE="${SWA_WRITE_LAYER_MODE:-last}"
export SWA_WRITE_KEEP_SCOPE="${SWA_WRITE_KEEP_SCOPE:-both_overlap}"
export ENABLE_SWA_OVERLAP_SOURCE_REPLACE="${ENABLE_SWA_OVERLAP_SOURCE_REPLACE:-1}"
export SWA_OVERLAP_SOURCE_REPLACE_ALPHA="${SWA_OVERLAP_SOURCE_REPLACE_ALPHA:-0.5}"
export SWA_OVERLAP_SOURCE_REPLACE_MODE="${SWA_OVERLAP_SOURCE_REPLACE_MODE:-source}"
export SWA_OVERLAP_SOURCE_REPLACE_TARGET="${SWA_OVERLAP_SOURCE_REPLACE_TARGET:-kv}"
export SWA_OVERLAP_SOURCE_REPLACE_LAYER_MODE="${SWA_OVERLAP_SOURCE_REPLACE_LAYER_MODE:-last}"
export TTT_WRITE_GRADIENT_REVERSAL_MODE="${TTT_WRITE_GRADIENT_REVERSAL_MODE:-tri_replay}"
export TTT_WRITE_GRADIENT_REVERSAL_GAMMA="${TTT_WRITE_GRADIENT_REVERSAL_GAMMA:-0.0}"
export TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK="${TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK:-0}"
export TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS="${TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS:-5:0.005,6:0.005,7:0.005,8:0.005,9:0.005,10:0.003,11:0.003,12:0.003,16:0.0003}"
export TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE="${TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE:-update_conflict_energy}"
export TTT_WRITE_TRI_REPLAY_POSITIVE_FRAC="${TTT_WRITE_TRI_REPLAY_POSITIVE_FRAC:-0.35}"
export TTT_WRITE_TRI_REPLAY_NEGATIVE_FRAC="${TTT_WRITE_TRI_REPLAY_NEGATIVE_FRAC:-0.12}"
export TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA="${TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA:-0.85}"
export TTT_WRITE_TRI_REPLAY_ROLE_MODE="${TTT_WRITE_TRI_REPLAY_ROLE_MODE:-fixed}"
export TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS="${TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS:-5:0.35/0.12/0.85,6:0.35/0.12/0.85,7:0.35/0.12/0.85,8:0.35/0.12/0.85,9:0.35/0.12/0.85,10:0.35/0.12/0.85,11:0.35/0.12/0.85,12:0.35/0.12/0.85,16:0.35/0.08/0.85}"
export TTT_WRITE_COMMIT_EMA_ALPHA="${TTT_WRITE_COMMIT_EMA_ALPHA:-0.5}"
export TTT_WRITE_COMMIT_EMA_BRANCH_MASK="${TTT_WRITE_COMMIT_EMA_BRANCH_MASK:-0}"
export TTT_WRITE_COMMIT_EMA_CHUNKS="${TTT_WRITE_COMMIT_EMA_CHUNKS:-5,6}"
export TTT_WRITE_NATIVE_MIX_SCALES="${TTT_WRITE_NATIVE_MIX_SCALES:-1.10,1.00,1.00}"
export V11_PROJECTION_TRACE_DIR="${V11_PROJECTION_TRACE_DIR:-$OUT/v11_projection_trace}"

export STAGE_C_MODE="${STAGE_C_MODE:-none}"
export STAGE_C_CACHE_DIR="${STAGE_C_CACHE_DIR:-}"
export STAGE_C_CACHE_MODE="${STAGE_C_CACHE_MODE:-off}"
export STAGE_C_CACHE_REQUIRE_HIT="${STAGE_C_CACHE_REQUIRE_HIT:-0}"
export STAGE_C_CACHE_VALIDATE="${STAGE_C_CACHE_VALIDATE:-0}"
export SEMANTIC_PRIOR_MODE="${SEMANTIC_PRIOR_MODE:-spg_v2}"
export HMC_IGNORE_SEMANTIC_PRIOR="${HMC_IGNORE_SEMANTIC_PRIOR:-0}"

export ENABLE_CONTEXT_SOURCE_SKIP=0
export CONTEXT_SOURCE_SKIP_IMPL="bias"
export CONTEXT_SOURCE_SKIP_SCOPE="frame"
export CONTEXT_SOURCE_SKIP_MODE="hard"
export CONTEXT_SOURCE_SKIP_MASK="semantic_role_negative"
export CONTEXT_SOURCE_SKIP_LAYER_MODE="early"
export CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS="${CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS:-1}"
export CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES="${CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES:-128}"
export SEMANTIC_ROLE_POLICY="none"
export SEMANTIC_MEMORY_PATHS=""
export SEMANTIC_ACTION_ACTIVE_CHUNKS=""
export SEMANTIC_ACTION_INACTIVE_READ_CUE_SOURCE=""
export READ_CALIB_MODE="${READ_CALIB_MODE:-none}"
export READ_TARGET_MASS="${READ_TARGET_MASS:-0.06}"
export READ_BLEND_LAMBDA="${READ_BLEND_LAMBDA:-0.25}"

disable_commit_ema() {
  export TTT_WRITE_COMMIT_EMA_ALPHA="1.0"
  export TTT_WRITE_COMMIT_EMA_BRANCH_MASK="all"
  export TTT_WRITE_COMMIT_EMA_CHUNKS=""
}

global_commit_ema() {
  export TTT_WRITE_COMMIT_EMA_ALPHA="$1"
  export TTT_WRITE_COMMIT_EMA_BRANCH_MASK="${2:-0}"
  export TTT_WRITE_COMMIT_EMA_CHUNKS=""
}

fixed_read_beta() {
  export READ_BETA_FRAME_CHUNKS=""
}

fixed_tri_gamma() {
  export TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS=""
  export TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS=""
  export TTT_WRITE_GRADIENT_REVERSAL_GAMMA="$1"
}

disable_tri_replay() {
  export TTT_WRITE_GRADIENT_REVERSAL_MODE="none"
  export TTT_WRITE_GRADIENT_REVERSAL_GAMMA="0.0"
  export TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS=""
  export TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS=""
}

apply_clean_parent() {
  fixed_read_beta
  fixed_tri_gamma "${V45_C9_CLEAN_TRI_GAMMA:-0.003}"
  global_commit_ema "${V45_C9_CLEAN_COMMIT_EMA_ALPHA:-1.0}" "${V45_C9_CLEAN_COMMIT_EMA_BRANCH_MASK:-all}"
}

enable_semantic_cache() {
  export STAGE_C_MODE="${STAGE_C_MODE_OVERRIDE:-reference}"
  export STAGE_C_CACHE_DIR="${STAGE_C_CACHE_DIR_OVERRIDE:-$ROOT/results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full}"
  export STAGE_C_CACHE_MODE="${STAGE_C_CACHE_MODE_OVERRIDE:-read}"
  export STAGE_C_CACHE_REQUIRE_HIT="${STAGE_C_CACHE_REQUIRE_HIT_OVERRIDE:-1}"
}

sem_resid_read_only() {
  enable_semantic_cache
  CUE="${V45_SEM_RESID_CUE_OVERRIDE:-v31.sem_resid_coarse_l025.c23past}"
  export READ_BLEND_LAMBDA="${READ_BLEND_LAMBDA_OVERRIDE:-0.25}"
}

case "${V45_PARENT:-C9}" in
  C9|c9|"") ;;
  C9_CLEAN|c9_clean|clean) apply_clean_parent ;;
  *) echo "Unsupported V45_PARENT=${V45_PARENT}" >&2; exit 2 ;;
esac

case "$CANDIDATE_ID" in
  P0_C9_REPEAT|D0_C9_REPEAT)
    ;;

  D1_FIXED_READ_BETA_ONLY)
    fixed_read_beta
    ;;
  D2_FIXED_TRI_GAMMA_003)
    fixed_tri_gamma "0.003"
    ;;
  D3_FIXED_TRI_GAMMA_004)
    fixed_tri_gamma "0.004"
    ;;
  D4_FIXED_TRI_GAMMA_005)
    fixed_tri_gamma "0.005"
    ;;
  D5_FIXED_COMMIT_EMA_OFF)
    disable_commit_ema
    ;;
  D6_FIXED_COMMIT_EMA_GLOBAL_A08)
    global_commit_ema "0.8" "0"
    ;;
  D7_C9_CLEAN_BEST_FIXED)
    apply_clean_parent
    ;;

  I1_NO_TRI_REPLAY_NO_EMA)
    disable_tri_replay
    disable_commit_ema
    ;;
  I2_NO_TRI_REPLAY_NO_SWA)
    disable_tri_replay
    export ENABLE_SWA_OVERLAP_SOURCE_REPLACE=0
    ;;
  I3_NO_TRI_REPLAY_NATIVE_MIX_OFF)
    disable_tri_replay
    export TTT_WRITE_NATIVE_MIX_SCALES="1.00,1.00,1.00"
    ;;
  I4_FIXED_TRI_GAMMA_BEST_NO_EMA)
    fixed_tri_gamma "${V45_FIXED_TRI_GAMMA_BEST:-0.003}"
    disable_commit_ema
    ;;
  I5_FIXED_TRI_GAMMA_BEST_NO_SWA)
    fixed_tri_gamma "${V45_FIXED_TRI_GAMMA_BEST:-0.003}"
    export ENABLE_SWA_OVERLAP_SOURCE_REPLACE=0
    ;;
  I6_FIXED_TRI_GAMMA_BEST_NATIVE_MIX_OFF)
    fixed_tri_gamma "${V45_FIXED_TRI_GAMMA_BEST:-0.003}"
    export TTT_WRITE_NATIVE_MIX_SCALES="1.00,1.00,1.00"
    ;;
  I7_FIXED_READ_BETA_FIXED_TRI_GAMMA_BEST)
    fixed_read_beta
    fixed_tri_gamma "${V45_FIXED_TRI_GAMMA_BEST:-0.003}"
    ;;
  I8_FIXED_READ_BETA_FIXED_TRI_GAMMA_BEST_FIXED_EMA_BEST)
    fixed_read_beta
    fixed_tri_gamma "${V45_FIXED_TRI_GAMMA_BEST:-0.003}"
    global_commit_ema "${V45_FIXED_EMA_ALPHA_BEST:-1.0}" "${V45_FIXED_EMA_BRANCH_MASK_BEST:-all}"
    ;;

  S0_C23_PAST)
    CUE="acl2.gg.qq.low.g2_3.past_only.headmean.robustq"
    ;;
  S1_C23_FULL_CHUNK)
    CUE="acl2.gg.qq.low.g2_3.full_chunk.headmean.robustq"
    ;;
  S2_C23_FULL_CHUNK_NO_OVERLAP)
    CUE="acl2.gg.qq.low.g2_3.full_chunk_no_overlap.headmean.robustq"
    ;;
  S3_C23_OFF246)
    CUE="acl2.gg.qq.low.g2_3.off246.headmean.robustq"
    ;;
  S4_C23_NEAR12)
    CUE="acl2.gg.qq.low.g2_3.near12.headmean.robustq"
    ;;
  S5_C23_PAST_PLUS_FUTURE_LIGHT)
    CUE="acl2.gg.qq.low.g2_3.past_plus_future_light_real.headmean.robustq"
    ;;

  A0_FIXED_C9_TRI_REPLAY)
    ;;
  A1_KMEANS3_TRI_REPLAY)
    export TTT_WRITE_TRI_REPLAY_ROLE_MODE="kmeans3"
    ;;
  A2_OTSU3_TRI_REPLAY)
    export TTT_WRITE_TRI_REPLAY_ROLE_MODE="otsu3"
    ;;
  A3_MAD_TRI_REPLAY)
    export TTT_WRITE_TRI_REPLAY_ROLE_MODE="mad"
    ;;
  A4_ADAPTIVE_QUANTILE_TRI_REPLAY)
    export TTT_WRITE_TRI_REPLAY_ROLE_MODE="adaptive_quantile"
    ;;

  SEM1_C23_RESID_READ_ONLY_ON_C9)
    sem_resid_read_only
    ;;
  SEM2_C23_RESID_READ_ONLY_ON_C9_CLEAN)
    apply_clean_parent
    sem_resid_read_only
    ;;
  SEM3_C23_RESID_PLUS_BEST_SUPPORT)
    sem_resid_read_only
    CUE="${V45_BEST_SUPPORT_CUE:?set V45_BEST_SUPPORT_CUE for SEM3}"
    ;;
  SEM4_C23_RESID_PLUS_ADAPTIVE_TRI_BEST)
    sem_resid_read_only
    export TTT_WRITE_TRI_REPLAY_ROLE_MODE="${V45_BEST_TRI_ROLE_MODE:?set V45_BEST_TRI_ROLE_MODE for SEM4}"
    ;;

  SMOKE_TWO_REPLAY)
    export TTT_WRITE_GRADIENT_REVERSAL_MODE="two_replay"
    export TTT_WRITE_GRADIENT_REVERSAL_GAMMA="${V45_SMOKE_TWO_REPLAY_GAMMA:-0.001}"
    export TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS=""
    export TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS=""
    ;;

  *)
    echo "Unsupported v45 CANDIDATE_ID: $CANDIDATE_ID" >&2
    exit 2
    ;;
esac

mkdir -p "$OUT"
cat > "$OUT/effective_config.yaml" <<EOF
run_name: "$RUN_NAME"
candidate_id: "$CANDIDATE_ID"
gpu: "$GPU"
kitti_seq: "${KITTI_SEQ:-01}"
start_frame: "$START_FRAME"
end_frame: "$END_FRAME"
mode: "$MODE"
cue: "$CUE"
beta_frame: "$BETA"
write_score: "$WRITE_SCORE"
read_beta_frame_chunks: "$READ_BETA_FRAME_CHUNKS"
ttt_write_gradient_reversal_mode: "$TTT_WRITE_GRADIENT_REVERSAL_MODE"
ttt_write_gradient_reversal_gamma: "$TTT_WRITE_GRADIENT_REVERSAL_GAMMA"
ttt_write_gradient_reversal_chunk_gammas: "$TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS"
ttt_write_tri_replay_positive_frac: "$TTT_WRITE_TRI_REPLAY_POSITIVE_FRAC"
ttt_write_tri_replay_negative_frac: "$TTT_WRITE_TRI_REPLAY_NEGATIVE_FRAC"
ttt_write_tri_replay_neutral_lambda: "$TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA"
ttt_write_tri_replay_role_mode: "$TTT_WRITE_TRI_REPLAY_ROLE_MODE"
ttt_write_tri_replay_chunk_params: "$TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS"
ttt_write_commit_ema_alpha: "$TTT_WRITE_COMMIT_EMA_ALPHA"
ttt_write_commit_ema_branch_mask: "$TTT_WRITE_COMMIT_EMA_BRANCH_MASK"
ttt_write_commit_ema_chunks: "$TTT_WRITE_COMMIT_EMA_CHUNKS"
ttt_write_native_mix_scales: "$TTT_WRITE_NATIVE_MIX_SCALES"
enable_swa_overlap_source_replace: "$ENABLE_SWA_OVERLAP_SOURCE_REPLACE"
swa_overlap_source_replace_alpha: "$SWA_OVERLAP_SOURCE_REPLACE_ALPHA"
stage_c_mode: "$STAGE_C_MODE"
stage_c_cache_mode: "$STAGE_C_CACHE_MODE"
stage_c_cache_dir: "$STAGE_C_CACHE_DIR"
semantic_role_policy: "$SEMANTIC_ROLE_POLICY"
semantic_memory_paths: "$SEMANTIC_MEMORY_PATHS"
v11_projection_trace_dir: "$V11_PROJECTION_TRACE_DIR"
EOF

cat > "$OUT/chunk_id_policy_audit.json" <<EOF
{
  "candidate_id": "$CANDIDATE_ID",
  "has_read_beta_frame_chunks": $([ -n "$READ_BETA_FRAME_CHUNKS" ] && echo true || echo false),
  "has_tri_gamma_chunk_map": $([ -n "$TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS" ] && echo true || echo false),
  "has_tri_replay_chunk_params": $([ -n "$TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS" ] && echo true || echo false),
  "has_commit_ema_chunks": $([ -n "$TTT_WRITE_COMMIT_EMA_CHUNKS" ] && echo true || echo false),
  "read_beta_frame_chunks": "$READ_BETA_FRAME_CHUNKS",
  "tri_gamma_chunk_map": "$TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS",
  "tri_replay_chunk_params": "$TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS",
  "commit_ema_chunks": "$TTT_WRITE_COMMIT_EMA_CHUNKS"
}
EOF

cat > "$OUT/stage_c_semantic_disabled_confirm.json" <<EOF
{
  "candidate_id": "$CANDIDATE_ID",
  "stage_c_mode": "$STAGE_C_MODE",
  "stage_c_cache_mode": "$STAGE_C_CACHE_MODE",
  "semantic_role_policy": "$SEMANTIC_ROLE_POLICY",
  "semantic_memory_paths": "$SEMANTIC_MEMORY_PATHS",
  "stage_c_disabled": $([ "$STAGE_C_MODE" = "none" ] && echo true || echo false)
}
EOF

echo "[$(date '+%F %T')] v45 launch gpu=$GPU run=$RUN_NAME candidate=$CANDIDATE_ID base=$BASE parent=${V45_PARENT:-C9} cue=$CUE beta=$BETA stage_c=$STAGE_C_MODE tri_role=$TTT_WRITE_TRI_REPLAY_ROLE_MODE"
"$ROOT/tools/run_attention_cue_experiment.sh" "$GPU" "$RUN_NAME" "$MODE" "$CUE" "$BETA" "$WRITE_SCORE"
