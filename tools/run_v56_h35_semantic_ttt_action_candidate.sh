#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 GPU ROW RUN_NAME" >&2
  echo "Rows: H35_FULL_REPEAT A1_96F A2_96F A3_96F A4_96F A1_704F ... A4_FULL B1_96F ... B4_FULL" >&2
  exit 2
fi

GPU="$1"
ROW="$2"
RUN_NAME="$3"

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
PY="${LOGER_PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"
RESULT_ROOT="${V56_RESULT_ROOT:-$ROOT/results/kitti01_hmc_v2/acl2_v56_h35_semanticboost_newtttaction_fast}"
PLAN_NOTE="${V56_PLAN_NOTE:-$ROOT/docs/ACL2_v56_H35_SemanticBoost_NewTTTAction_FastPlan.md}"

PHASE="phase_unknown"
END_FRAME="${V56_END_FRAME:-96}"
TRACK="unknown"
CANDIDATE="$ROW"
MODE="hybrid"
CUE="${V56_DG_CUE:-acl2.gg.qq.low.g2_3.past_only.headmean.robustq}"
BETA="${V56_READ_BETA:-4.75}"
WRITE_SCORE="${V56_TTT_WRITE_SCORE:-stage_d_x_dg_inv_sqrt}"
ROLE_MODE="adaptive_writer_sc_gamma_split"
RISK_SOURCE="${V56_TTT_RISK_SOURCE:-ttt_residual_x_dg}"
LAYER_GAMMAS="${V56_TTT_LAYER_GAMMAS:-0:0.0075,8:0.0075,17:0.0075}"
COMMIT_FILTER_MODE="none"
SEMANTIC_DESC="none"
TTT_ACTION_DESC="H35 clean adaptive writer"

case "$ROW" in
  H35_FULL_REPEAT)
    PHASE="${V56_PHASE:-phase0_h35_repeat}"
    END_FRAME="${V56_END_FRAME:-10000}"
    TRACK="phase0"
    CANDIDATE="H35"
    ;;
  A[1-4]_96F)
    PHASE="${V56_PHASE:-phase1_track_a_smoke}"
    END_FRAME="${V56_END_FRAME:-96}"
    TRACK="A"
    CANDIDATE="${ROW%_96F}"
    ;;
  A[1-4]_704F)
    PHASE="${V56_PHASE:-phase1_track_a_704_screen}"
    END_FRAME="${V56_END_FRAME:-704}"
    TRACK="A"
    CANDIDATE="${ROW%_704F}"
    ;;
  A[1-4]_FULL)
    PHASE="${V56_PHASE:-phase1_track_a_full}"
    END_FRAME="${V56_END_FRAME:-10000}"
    TRACK="A"
    CANDIDATE="${ROW%_FULL}"
    ;;
  B[1-4]_96F)
    PHASE="${V56_PHASE:-phase2_track_b_smoke}"
    END_FRAME="${V56_END_FRAME:-96}"
    TRACK="B"
    CANDIDATE="${ROW%_96F}"
    ;;
  B[1-4]_704F)
    PHASE="${V56_PHASE:-phase2_track_b_704_screen}"
    END_FRAME="${V56_END_FRAME:-704}"
    TRACK="B"
    CANDIDATE="${ROW%_704F}"
    ;;
  B[1-4]_FULL)
    PHASE="${V56_PHASE:-phase2_track_b_full}"
    END_FRAME="${V56_END_FRAME:-10000}"
    TRACK="B"
    CANDIDATE="${ROW%_FULL}"
    ;;
  COMBO_FULL)
    PHASE="${V56_PHASE:-phase3_combo_full}"
    END_FRAME="${V56_END_FRAME:-10000}"
    TRACK="combo"
    CANDIDATE="${V56_COMBO_CANDIDATE:-COMBO}"
    ;;
  *)
    echo "Unsupported v56 row: $ROW" >&2
    exit 2
    ;;
esac

enable_semantic_cache() {
  export STAGE_C_MODE="${V56_STAGE_C_MODE:-reference}"
  export STAGE_C_CACHE_DIR="${V56_STAGE_C_CACHE_DIR:-$ROOT/results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full}"
  export STAGE_C_CACHE_MODE="${V56_STAGE_C_CACHE_MODE:-read}"
  export STAGE_C_CACHE_REQUIRE_HIT="${V56_STAGE_C_CACHE_REQUIRE_HIT:-1}"
  export STAGE_C_CACHE_VALIDATE="${V56_STAGE_C_CACHE_VALIDATE:-0}"
  export STAGE_C_INLINE_WHEN_IGNORED="0"
}

enable_semantic_frame_skip() {
  export ENABLE_CONTEXT_SOURCE_SKIP="1"
  export CONTEXT_SOURCE_SKIP_IMPL="${V56_CONTEXT_SOURCE_SKIP_IMPL:-bias}"
  export CONTEXT_SOURCE_SKIP_SCOPE="${V56_CONTEXT_SOURCE_SKIP_SCOPE:-both}"
  export CONTEXT_SOURCE_SKIP_MODE="${V56_CONTEXT_SOURCE_SKIP_MODE:-soft}"
  export CONTEXT_SOURCE_SKIP_MASK="semantic_role_negative"
  export CONTEXT_SOURCE_SKIP_LAYER_MODE="${V56_CONTEXT_SOURCE_SKIP_LAYER_MODE:-early}"
  export CONTEXT_SOURCE_SKIP_SOFT_RHO="${V56_CONTEXT_SOURCE_SKIP_SOFT_RHO:-0.50}"
  export CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP="${V56_CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP:-0.50}"
  export CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS="${V56_CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS:-1}"
  export CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES="${V56_CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES:-128}"
  export SEMANTIC_MEMORY_PATHS="frame,global"
}

if [[ "$TRACK" == "A" || "$TRACK" == "combo" ]]; then
  enable_semantic_cache
fi

case "$CANDIDATE" in
  H35)
    ;;
  A1)
    CUE="${V56_A1_CUE:-v31.sem_resid_coarse_l050.c23past}"
    SEMANTIC_DESC="semantic C23 residual lambda=0.50 READ only"
    ;;
  A2)
    enable_semantic_frame_skip
    export SEMANTIC_ROLE_POLICY="${V56_A2_SEMANTIC_ROLE_POLICY:-causal_fg_risk_only}"
    SEMANTIC_DESC="high-influence anomaly READ filtering via semantic_role_negative source skip"
    ;;
  A3)
    enable_semantic_frame_skip
    export SEMANTIC_ROLE_POLICY="${V56_A3_SEMANTIC_ROLE_POLICY:-causal_fg_semantic_risk_skip}"
    SEMANTIC_DESC="high-influence anomaly READ filtering with static-structure protection"
    ;;
  A4)
    CUE="${V56_A4_CUE:-v31.sem_resid_coarse_l050.c23past}"
    enable_semantic_frame_skip
    export SEMANTIC_ROLE_POLICY="${V56_A4_SEMANTIC_ROLE_POLICY:-causal_fg_risk_only}"
    SEMANTIC_DESC="semantic C23 residual plus high-influence anomaly READ filtering"
    ;;
  B1)
    ROLE_MODE="adaptive_writer_binary_anchor_split"
    TTT_ACTION_DESC="Binary Stable-Anchor Replay: stable anchor long write, non-anchor no-long-write"
    ;;
  B2)
    ROLE_MODE="adaptive_writer_risk_veto_split"
    TTT_ACTION_DESC="Risk-Veto Commit: non-risk long write, risk no-long-write"
    ;;
  B3)
    ROLE_MODE="adaptive_writer_state_energy_matched_split"
    export TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_MODE="dual_lifetime"
    export TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_BRANCH_MASK="${V56_B3_TRANSIENT_BRANCH_MASK:-0}"
    export TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_LONG_SCALE="${V56_B3_LONG_SCALE:-0.35}"
    export TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_APPLY_SCALE="${V56_B3_APPLY_SCALE:-1.0}"
    TTT_ACTION_DESC="Two-Lifetime Commit using dual_lifetime transient delta"
    ;;
  B4)
    ROLE_MODE="adaptive_writer_state_energy_matched_split"
    export TTT_WRITE_NATIVE_DELTA_GATE_MODE="${V56_B4_NATIVE_DELTA_GATE_MODE:-orthogonal_suppress}"
    export TTT_WRITE_NATIVE_DELTA_GATE_FALLBACK="${V56_B4_ORTHOGONAL_RHO:-0.35}"
    export TTT_WRITE_NATIVE_DELTA_GATE_BRANCH_MASK="${V56_B4_BRANCH_MASK:-0}"
    TTT_ACTION_DESC="Projection Commit via native-delta orthogonal suppression"
    ;;
  COMBO)
    if [ -n "${V56_COMBO_READ_CUE:-}" ]; then CUE="$V56_COMBO_READ_CUE"; fi
    if [ -n "${V56_COMBO_ROLE_MODE:-}" ]; then ROLE_MODE="$V56_COMBO_ROLE_MODE"; fi
    if [ -n "${V56_COMBO_SEMANTIC_ROLE_POLICY:-}" ]; then
      enable_semantic_frame_skip
      export SEMANTIC_ROLE_POLICY="$V56_COMBO_SEMANTIC_ROLE_POLICY"
    fi
    SEMANTIC_DESC="best Track A semantic config supplied by V56_COMBO_*"
    TTT_ACTION_DESC="best Track B action supplied by V56_COMBO_*"
    ;;
esac

BASE="${V56_ROLLOUT_BASE:-$RESULT_ROOT/$PHASE/rollouts}"
OUT="$BASE/$RUN_NAME"

mkdir -p "$OUT"
if [ -f "$OUT/run_status.txt" ] && grep -q "DONE $RUN_NAME" "$OUT/run_status.txt"; then
  echo "[v56] Skip completed run: $RUN_NAME"
  exit 0
fi

export LOGER_ROOT="$ROOT"
export LOGER_PY="$PY"
export ATTN_CUE_BASE="$BASE"
export KITTI_SEQ="${KITTI_SEQ:-01}"
export END_FRAME="$END_FRAME"
export RESET_EVERY="${V56_RESET_EVERY:-5}"
export EMPTY_CUDA_CACHE_EACH_CHUNK="${V56_EMPTY_CUDA_CACHE_EACH_CHUNK:-0}"
export STAGE_C_MODE="${STAGE_C_MODE:-none}"
export STAGE_C_CACHE_DIR="${STAGE_C_CACHE_DIR:-}"
export STAGE_C_CACHE_MODE="${STAGE_C_CACHE_MODE:-off}"
export STAGE_C_CACHE_REQUIRE_HIT="${STAGE_C_CACHE_REQUIRE_HIT:-0}"
export STAGE_C_CACHE_VALIDATE="${STAGE_C_CACHE_VALIDATE:-0}"
export STAGE_C_INLINE_WHEN_IGNORED="${STAGE_C_INLINE_WHEN_IGNORED:-0}"
export SEMANTIC_ROLE_POLICY="${SEMANTIC_ROLE_POLICY:-none}"
export SEMANTIC_MEMORY_PATHS="${SEMANTIC_MEMORY_PATHS:-}"
export SEMANTIC_ACTION_ACTIVE_CHUNKS=""
export SEMANTIC_ACTION_INACTIVE_READ_CUE_SOURCE=""
export READ_PATH="frame"
export FRAME_BIAS_MODE="${V56_FRAME_BIAS_MODE:-pair}"
export READ_LAYER_MODE="${V56_READ_LAYER_MODE:-early}"
export READ_BETA_FRAME_CHUNKS=""
export READ_TOPK_FRAC="${V56_READ_TOPK_FRAC:-0.0}"
export READ_CALIB_MODE="${READ_CALIB_MODE:-none}"
export READ_BLEND_LAMBDA="${READ_BLEND_LAMBDA:-0.25}"
export ENABLE_CONTEXT_SOURCE_SKIP="${ENABLE_CONTEXT_SOURCE_SKIP:-0}"
export CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS="${CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS:-0}"
export HMC_COMMIT_MODE="probe_ttt_write"
export TTT_WRITE_GRADIENT_REVERSAL_MODE="tri_replay"
export TTT_WRITE_GRADIENT_REVERSAL_GAMMA="0.0"
export TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE="$RISK_SOURCE"
export TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK="${V56_TTT_BRANCH_MASK:-0}"
export TTT_WRITE_GRADIENT_REVERSAL_BRANCH_GAMMAS=""
export TTT_WRITE_GRADIENT_REVERSAL_LAYER_GAMMAS="$LAYER_GAMMAS"
export TTT_WRITE_GRADIENT_REVERSAL_HEAD_ROUTES=""
export TTT_WRITE_GRADIENT_REVERSAL_CHUNKS=""
export TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS=""
export TTT_WRITE_TRI_REPLAY_POSITIVE_FRAC="0.0"
export TTT_WRITE_TRI_REPLAY_NEGATIVE_FRAC="0.0"
export TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA="0.0"
export TTT_WRITE_TRI_REPLAY_ROLE_MODE="$ROLE_MODE"
export TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS=""
export TTT_WRITE_COMMIT_EMA_ALPHA="1.0"
export TTT_WRITE_COMMIT_EMA_BRANCH_MASK="all"
export TTT_WRITE_COMMIT_EMA_CHUNKS=""
export TTT_WRITE_NATIVE_MIX_SCALES="${V56_NATIVE_MIX_SCALES:-1.00,1.00,1.00}"
export TTT_WRITE_NATIVE_MIX_CHUNKS=""
export TTT_WRITE_COMMIT_FILTER_MODE="$COMMIT_FILTER_MODE"
export TTT_WRITE_COMMIT_FILTER_RISK_SOURCE="${V56_TTT_COMMIT_FILTER_RISK_SOURCE:-d_tok}"
export TTT_WRITE_COMMIT_FILTER_SCOPE="${V56_TTT_COMMIT_FILTER_SCOPE:-tail_overlap}"
export TTT_WRITE_COMMIT_FILTER_STAT="${V56_TTT_COMMIT_FILTER_STAT:-mean}"
export TTT_WRITE_COMMIT_FILTER_MIN="${V56_TTT_COMMIT_FILTER_MIN:-0.0}"
export TTT_WRITE_COMMIT_FILTER_MAX="${V56_TTT_COMMIT_FILTER_MAX:-1.0}"
export TTT_WRITE_COMMIT_FILTER_BRANCH_MASK="${V56_TTT_COMMIT_FILTER_BRANCH_MASK:-0}"
export TTT_WRITE_COMMIT_FILTER_CHUNKS=""
export TTT_WRITE_SCALE_STATE_MODE="${V56_TTT_SCALE_STATE_MODE:-none}"
export TTT_WRITE_SCALE_STATE_CHUNKS=""
export TTT_WRITE_SCALE_STATE_SAMPLE_TOKENS="${V56_TTT_SCALE_STATE_SAMPLE_TOKENS:-0}"
export ENABLE_SWA_OVERLAP_SOURCE_REPLACE="0"
export SWA_OVERLAP_SOURCE_REPLACE_ALPHA="0.0"

cat > "$OUT/effective_config.yaml" <<EOF
row: "$ROW"
run_name: "$RUN_NAME"
candidate: "$CANDIDATE"
track: "$TRACK"
plan_note: "$PLAN_NOTE"
gpu: "$GPU"
end_frame: "$END_FRAME"
mode: "$MODE"
cue: "$CUE"
beta: "$BETA"
write_score: "$WRITE_SCORE"
semantic_desc: "$SEMANTIC_DESC"
ttt_action_desc: "$TTT_ACTION_DESC"
read_path: "$READ_PATH"
read_beta_frame_chunks: "$READ_BETA_FRAME_CHUNKS"
ttt_write_gradient_reversal_mode: "$TTT_WRITE_GRADIENT_REVERSAL_MODE"
ttt_write_gradient_reversal_gamma: "$TTT_WRITE_GRADIENT_REVERSAL_GAMMA"
ttt_write_gradient_reversal_risk_source: "$TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE"
ttt_write_gradient_reversal_branch_mask: "$TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK"
ttt_write_gradient_reversal_layer_gammas: "$TTT_WRITE_GRADIENT_REVERSAL_LAYER_GAMMAS"
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
ttt_write_native_mix_chunks: "$TTT_WRITE_NATIVE_MIX_CHUNKS"
ttt_write_gradient_reversal_transient_mode: "${TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_MODE:-none}"
ttt_write_gradient_reversal_transient_long_scale: "${TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_LONG_SCALE:-0.0}"
ttt_write_gradient_reversal_transient_apply_scale: "${TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_APPLY_SCALE:-1.0}"
ttt_write_native_delta_gate_mode: "${TTT_WRITE_NATIVE_DELTA_GATE_MODE:-none}"
ttt_write_native_delta_gate_fallback: "${TTT_WRITE_NATIVE_DELTA_GATE_FALLBACK:-0.0}"
ttt_write_native_delta_gate_branch_mask: "${TTT_WRITE_NATIVE_DELTA_GATE_BRANCH_MASK:-all}"
stage_c_mode: "$STAGE_C_MODE"
stage_c_cache_dir: "$STAGE_C_CACHE_DIR"
stage_c_cache_mode: "$STAGE_C_CACHE_MODE"
stage_c_cache_require_hit: "$STAGE_C_CACHE_REQUIRE_HIT"
stage_c_inline_when_ignored: "$STAGE_C_INLINE_WHEN_IGNORED"
semantic_role_policy: "$SEMANTIC_ROLE_POLICY"
semantic_memory_paths: "$SEMANTIC_MEMORY_PATHS"
semantic_action_active_chunks: "$SEMANTIC_ACTION_ACTIVE_CHUNKS"
enable_context_source_skip: "$ENABLE_CONTEXT_SOURCE_SKIP"
context_source_skip_mask: "${CONTEXT_SOURCE_SKIP_MASK:-}"
context_source_skip_scope: "${CONTEXT_SOURCE_SKIP_SCOPE:-}"
context_source_skip_mode: "${CONTEXT_SOURCE_SKIP_MODE:-}"
context_source_skip_soft_rho: "${CONTEXT_SOURCE_SKIP_SOFT_RHO:-}"
enable_swa_overlap_source_replace: "$ENABLE_SWA_OVERLAP_SOURCE_REPLACE"
swa_overlap_source_replace_alpha: "$SWA_OVERLAP_SOURCE_REPLACE_ALPHA"
EOF
cp "$OUT/effective_config.yaml" "$OUT/v56_effective_config.yaml"

cat > "$OUT/chunk_id_policy_audit.json" <<EOF
{
  "row": "$ROW",
  "run_name": "$RUN_NAME",
  "absolute_chunk_id_policy_audit": {"pass": true},
  "has_read_beta_frame_chunks": false,
  "has_tri_gamma_chunk_map": false,
  "has_tri_replay_chunk_params": false,
  "has_commit_ema_chunks": false,
  "has_native_mix_chunks": false,
  "has_semantic_action_active_chunks": false,
  "read_beta_frame_chunks_empty": true,
  "ttt_gradient_reversal_chunk_gammas_empty": true,
  "ttt_tri_replay_chunk_params_empty": true,
  "ttt_commit_ema_chunks_empty": true,
  "native_mix_chunks_empty": true,
  "semantic_action_active_chunks_empty": true,
  "scale_state_chunks_empty": true,
  "read_beta_frame_chunks": "$READ_BETA_FRAME_CHUNKS",
  "tri_gamma_chunk_map": "$TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS",
  "tri_replay_chunk_params": "$TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS",
  "commit_ema_chunks": "$TTT_WRITE_COMMIT_EMA_CHUNKS",
  "native_mix_chunks": "$TTT_WRITE_NATIVE_MIX_CHUNKS",
  "semantic_action_active_chunks": "$SEMANTIC_ACTION_ACTIVE_CHUNKS"
}
EOF

cat > "$OUT/adaptive_ttt_audit.json" <<EOF
{
  "row": "$ROW",
  "run_name": "$RUN_NAME",
  "adaptive_ttt_writer": true,
  "manual_percentage_audit": {"pass": true},
  "no_manual_tri_replay_percentages": true,
  "manual_positive_frac": 0.0,
  "manual_negative_frac": 0.0,
  "manual_neutral_lambda": 0.0,
  "external_gamma": 0.0,
  "role_mode": "$TTT_WRITE_TRI_REPLAY_ROLE_MODE",
  "risk_source": "$TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE",
  "definition": "v56 no-chunk/no-manual H35-derived adaptive writer or semantic READ candidate"
}
EOF

cat > "$OUT/reproduce_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
V56_RESULT_ROOT="$RESULT_ROOT" \\
V56_PLAN_NOTE="$PLAN_NOTE" \\
tools/run_v56_h35_semantic_ttt_action_candidate.sh "$GPU" "$ROW" "$RUN_NAME"
EOF
chmod +x "$OUT/reproduce_command.sh"

echo "[v56] row=$ROW candidate=$CANDIDATE track=$TRACK run=$RUN_NAME gpu=$GPU end_frame=$END_FRAME phase=$PHASE"
echo "[v56] cue=$CUE role=$ROLE_MODE risk=$RISK_SOURCE layer_gammas=$LAYER_GAMMAS stage_c=$STAGE_C_CACHE_MODE"

tools/run_attention_cue_experiment.sh "$GPU" "$RUN_NAME" "$MODE" "$CUE" "$BETA" "$WRITE_SCORE"
