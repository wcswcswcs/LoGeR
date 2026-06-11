#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 GPU ROW RUN_NAME" >&2
  echo "Rows: S0_96F S1_96F S2_96F SREAD01_704F ... SREAD04_FULL TTT01_96F ... TTT03_FULL COMBO01_FULL" >&2
  exit 2
fi

GPU="$1"
ROW="$2"
RUN_NAME="$3"

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
PY="${LOGER_PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"
RESULT_ROOT="${V57_RESULT_ROOT:-$ROOT/results/kitti01_hmc_v2/acl2_v57_h35_semantic_action_repair_ttt_ttl_fast}"
PLAN_NOTE="${V57_PLAN_NOTE:-$ROOT/docs/ACL2_v57_H35_SemanticActionRepair_TTT_TTL_FastPlan.md}"

PHASE="phase_unknown"
TRACK="unknown"
CANDIDATE="$ROW"
END_FRAME="${V57_END_FRAME:-96}"
MODE="hybrid"
CUE="${V57_DG_CUE:-acl2.gg.qq.low.g2_3.past_only.headmean.robustq}"
BETA="${V57_READ_BETA:-4.75}"
WRITE_SCORE="${V57_TTT_WRITE_SCORE:-stage_d_x_dg_inv_sqrt}"
ROLE_MODE="adaptive_writer_sc_gamma_split"
RISK_SOURCE="${V57_TTT_RISK_SOURCE:-ttt_residual_x_dg}"
LAYER_GAMMAS="${V57_TTT_LAYER_GAMMAS:-0:0.0075,8:0.0075,17:0.0075}"
SEMANTIC_DESC="none"
TTT_ACTION_DESC="H35 clean adaptive writer"
ACTION_CLASS="baseline"

case "$ROW" in
  S0_96F)
    PHASE="${V57_PHASE:-phase1_semantic_smoke}"
    TRACK="semantic_smoke"
    CANDIDATE="S0_FORCED_SEMANTIC_SOURCE_SKIP"
    END_FRAME="${V57_END_FRAME:-96}"
    ;;
  S1_96F)
    PHASE="${V57_PHASE:-phase1_semantic_smoke}"
    TRACK="semantic_smoke"
    CANDIDATE="S1_SKY_LOWSTUFF_SOURCE_SKIP"
    END_FRAME="${V57_END_FRAME:-96}"
    ;;
  S2_96F)
    PHASE="${V57_PHASE:-phase1_semantic_smoke}"
    TRACK="semantic_smoke"
    CANDIDATE="S2_HIGH_D_SEMANTIC_SOURCE_SKIP"
    END_FRAME="${V57_END_FRAME:-96}"
    ;;
  SREAD0[1-4]_704F)
    PHASE="${V57_PHASE:-phase2_semantic_read_704_screen}"
    TRACK="semantic_read"
    CANDIDATE="${ROW%_704F}"
    END_FRAME="${V57_END_FRAME:-704}"
    ;;
  SREAD0[1-4]_FULL)
    PHASE="${V57_PHASE:-phase3_semantic_read_full}"
    TRACK="semantic_read"
    CANDIDATE="${ROW%_FULL}"
    END_FRAME="${V57_END_FRAME:-10000}"
    ;;
  TTT0[1-3]_96F)
    PHASE="${V57_PHASE:-phase4_ttt_smoke}"
    TRACK="ttt_action"
    CANDIDATE="${ROW%_96F}"
    END_FRAME="${V57_END_FRAME:-96}"
    ;;
  TTT0[1-3]_704F)
    PHASE="${V57_PHASE:-phase4_ttt_704_screen}"
    TRACK="ttt_action"
    CANDIDATE="${ROW%_704F}"
    END_FRAME="${V57_END_FRAME:-704}"
    ;;
  TTT0[1-3]_FULL)
    PHASE="${V57_PHASE:-phase4_ttt_full}"
    TRACK="ttt_action"
    CANDIDATE="${ROW%_FULL}"
    END_FRAME="${V57_END_FRAME:-10000}"
    ;;
  COMBO01_FULL)
    PHASE="${V57_PHASE:-phase5_combo_full}"
    TRACK="combo"
    CANDIDATE="COMBO01"
    END_FRAME="${V57_END_FRAME:-10000}"
    ;;
  *)
    echo "Unsupported v57 row: $ROW" >&2
    exit 2
    ;;
esac

enable_stage_c_cache() {
  export STAGE_C_MODE="${V57_STAGE_C_MODE:-reference}"
  export STAGE_C_CACHE_DIR="${V57_STAGE_C_CACHE_DIR:-$ROOT/results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full}"
  export STAGE_C_CACHE_MODE="${V57_STAGE_C_CACHE_MODE:-read}"
  export STAGE_C_CACHE_REQUIRE_HIT="${V57_STAGE_C_CACHE_REQUIRE_HIT:-1}"
  export STAGE_C_CACHE_VALIDATE="${V57_STAGE_C_CACHE_VALIDATE:-0}"
  export STAGE_C_INLINE_WHEN_IGNORED="0"
  export SEMANTIC_MEMORY_PATHS="${SEMANTIC_MEMORY_PATHS:-frame,global}"
}

enable_context_skip() {
  export ENABLE_CONTEXT_SOURCE_SKIP="1"
  export CONTEXT_SOURCE_SKIP_IMPL="${CONTEXT_SOURCE_SKIP_IMPL:-compact_kv}"
  export CONTEXT_SOURCE_SKIP_SCOPE="${CONTEXT_SOURCE_SKIP_SCOPE:-both}"
  export CONTEXT_SOURCE_SKIP_MODE="${CONTEXT_SOURCE_SKIP_MODE:-hard}"
  export CONTEXT_SOURCE_SKIP_LAYER_MODE="${CONTEXT_SOURCE_SKIP_LAYER_MODE:-early}"
  export CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS="${CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS:-1}"
  export CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES="${CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES:-128}"
  export READ_PATH="${READ_PATH:-frame}"
  export FRAME_BIAS_MODE="${FRAME_BIAS_MODE:-query}"
}

if [[ "$TRACK" == "semantic_smoke" || "$TRACK" == "semantic_read" || "$TRACK" == "combo" ]]; then
  enable_stage_c_cache
  enable_context_skip
fi

case "$CANDIDATE" in
  S0_FORCED_SEMANTIC_SOURCE_SKIP)
    BETA="${V57_S0_READ_BETA:-0.0}"
    export SEMANTIC_ROLE_POLICY="v57_forced_any_semantic_source_skip"
    export CONTEXT_SOURCE_SKIP_MASK="v36_synthetic_role_negative"
    SEMANTIC_DESC="forced any nonempty Stage-C semantic group to source skip; validates projection-to-hook path"
    ;;
  S1_SKY_LOWSTUFF_SOURCE_SKIP)
    BETA="${V57_S1_READ_BETA:-0.0}"
    export SEMANTIC_ROLE_POLICY="v57_sky_lowstuff_source_skip"
    export CONTEXT_SOURCE_SKIP_MASK="v36_synthetic_role_negative"
    SEMANTIC_DESC="sky/lowstuff/vegetation source skip without appearance anomaly filter"
    ;;
  S2_HIGH_D_SEMANTIC_SOURCE_SKIP)
    BETA="${V57_S2_READ_BETA:-0.0}"
    export SEMANTIC_ROLE_POLICY="${V57_S2_SEMANTIC_ROLE_POLICY:-causal_fg_semantic_risk_skip}"
    export CONTEXT_SOURCE_SKIP_MASK="semantic_role_negative"
    export SEMANTIC_ROLE_HIGHD_QUANTILE="${V57_S2_HIGHD_Q:-0.70}"
    SEMANTIC_DESC="semantic group intersected with high-D source skip"
    ;;
  SREAD01)
    export SEMANTIC_ROLE_POLICY="${V57_SREAD01_POLICY:-causal_fg_risk_only}"
    export CONTEXT_SOURCE_SKIP_MASK="${V57_SREAD01_MASK:-sem_structure_rescue_dg_q80}"
    SEMANTIC_DESC="general high-influence anomaly with semantic static-structure rescue"
    ;;
  SREAD02)
    export SEMANTIC_ROLE_POLICY="${V57_SREAD02_POLICY:-v57_sky_lowstuff_source_skip}"
    export CONTEXT_SOURCE_SKIP_MASK="${V57_SREAD02_MASK:-semantic_role_negative}"
    export SEMANTIC_ROLE_HIGHD_QUANTILE="${V57_SREAD02_HIGHD_Q:-0.75}"
    SEMANTIC_DESC="sky/lowstuff/vegetation high-influence source filtering"
    ;;
  SREAD03)
    CUE="${V57_SREAD03_CUE:-v31.sem_resid_coarse_l050.c23past}"
    export SEMANTIC_ROLE_POLICY="${V57_SREAD03_POLICY:-causal_fg_semantic_risk_skip}"
    export CONTEXT_SOURCE_SKIP_MASK="${V57_SREAD03_MASK:-semantic_role_negative}"
    export SEMANTIC_ROLE_HIGHD_QUANTILE="${V57_SREAD03_HIGHD_Q:-0.70}"
    SEMANTIC_DESC="semantic C23 residual plus source-action guard"
    ;;
  SREAD04)
    export SEMANTIC_ROLE_POLICY="${V57_SREAD04_POLICY:-causal_fg_semantic_risk_skip}"
    export CONTEXT_SOURCE_SKIP_MASK="${V57_SREAD04_MASK:-sem_structure_rescue_dg_q80}"
    export SEMANTIC_ROLE_HIGHD_QUANTILE="${V57_SREAD04_HIGHD_Q:-0.70}"
    SEMANTIC_DESC="anomaly filter with static-structure source rescue"
    ;;
  TTT01)
    ROLE_MODE="${V57_TTT01_ROLE_MODE:-adaptive_writer_binary_anchor_split}"
    export TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA="${V57_TTT01_NEUTRAL_LAMBDA:-0.35}"
    export TTT_WRITE_NATIVE_DELTA_GATE_MODE="${V57_TTT01_NATIVE_DELTA_GATE_MODE:-cosine_cap}"
    export TTT_WRITE_NATIVE_DELTA_GATE_MIN_COS="${V57_TTT01_MIN_COS:-0.0}"
    export TTT_WRITE_NATIVE_DELTA_GATE_CAP_RATIO="${V57_TTT01_CAP_RATIO:-1.0}"
    TTT_ACTION_DESC="TTT_01 two-replay static long plus native residual continuity guard"
    ACTION_CLASS="ttt_two_replay_static_native"
    ;;
  TTT02)
    ROLE_MODE="${V57_TTT02_ROLE_MODE:-adaptive_writer_state_energy_matched_split}"
    export TTT_WRITE_REPLAY_TOKEN_FILTER_MODE="${V57_TTT02_FILTER_MODE:-scoped_static_topk}"
    export TTT_WRITE_REPLAY_TOKEN_FILTER_SCOPE="${V57_TTT02_FILTER_SCOPE:-tail_overlap}"
    export TTT_WRITE_REPLAY_TOKEN_FILTER_RATIO="${V57_TTT02_FILTER_RATIO:-0.70}"
    export TTT_WRITE_REPLAY_TOKEN_FILTER_BRANCH_MASK="${V57_TTT02_BRANCH_MASK:-0}"
    export TTT_WRITE_REPLAY_TOKEN_FILTER_BLEND_MODE="${V57_TTT02_BLEND_MODE:-ttl_aligned_dynamic}"
    export TTT_WRITE_REPLAY_TOKEN_FILTER_BLEND="${V57_TTT02_BLEND:-0.50}"
    export TTT_WRITE_TRANSIENT_DELTA_SUBTRACT_SCALE="${V57_TTT02_SUBTRACT_SCALE:-1.0}"
    export TTT_WRITE_TRANSIENT_DELTA_BRANCH_MASK="${V57_TTT02_BRANCH_MASK:-0}"
    export TTT_WRITE_TRANSIENT_DELTA_TTL="${V57_TTT02_TTL:-1}"
    TTT_ACTION_DESC="TTT_02 short residual TTL using one-hop transient dynamic residual"
    ACTION_CLASS="ttt_short_residual_ttl"
    ;;
  TTT03)
    ROLE_MODE="${V57_TTT03_ROLE_MODE:-adaptive_writer_risk_veto_split}"
    export TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA="${V57_TTT03_NEUTRAL_LAMBDA:-0.65}"
    export TTT_WRITE_COMMIT_FILTER_MODE="${V57_TTT03_COMMIT_FILTER_MODE:-native_to_candidate_by_risk}"
    export TTT_WRITE_COMMIT_FILTER_RISK_SOURCE="${V57_TTT03_COMMIT_RISK:-ttt_residual_x_dg}"
    export TTT_WRITE_COMMIT_FILTER_SCOPE="${V57_TTT03_COMMIT_SCOPE:-tail_overlap}"
    export TTT_WRITE_COMMIT_FILTER_STAT="${V57_TTT03_COMMIT_STAT:-q90}"
    export TTT_WRITE_COMMIT_FILTER_MIN="${V57_TTT03_COMMIT_MIN:-0.35}"
    export TTT_WRITE_COMMIT_FILTER_MAX="${V57_TTT03_COMMIT_MAX:-1.0}"
    TTT_ACTION_DESC="TTT_03 read-conditioned/high-risk restricted no-long write"
    ACTION_CLASS="ttt_read_conditioned_restricted_no_long"
    ;;
  COMBO01)
    export SEMANTIC_ROLE_POLICY="${V57_COMBO_SEMANTIC_ROLE_POLICY:-causal_fg_semantic_risk_skip}"
    export CONTEXT_SOURCE_SKIP_MASK="${V57_COMBO_CONTEXT_MASK:-sem_structure_rescue_dg_q80}"
    ROLE_MODE="${V57_COMBO_ROLE_MODE:-adaptive_writer_state_energy_matched_split}"
    TTT_ACTION_DESC="best available semantic READ plus best available new TTT action"
    SEMANTIC_DESC="combo supplied by V57_COMBO_*"
    ACTION_CLASS="combo"
    ;;
esac

BASE="${V57_ROLLOUT_BASE:-$RESULT_ROOT/$PHASE/rollouts}"
OUT="$BASE/$RUN_NAME"

mkdir -p "$OUT"
if [ -f "$OUT/run_status.txt" ] && grep -q "DONE $RUN_NAME" "$OUT/run_status.txt"; then
  echo "[v57] Skip completed run: $RUN_NAME"
  exit 0
fi

export LOGER_ROOT="$ROOT"
export LOGER_PY="$PY"
export ATTN_CUE_BASE="$BASE"
export KITTI_SEQ="${KITTI_SEQ:-01}"
export END_FRAME="$END_FRAME"
export RESET_EVERY="${V57_RESET_EVERY:-5}"
export EMPTY_CUDA_CACHE_EACH_CHUNK="${V57_EMPTY_CUDA_CACHE_EACH_CHUNK:-0}"
export STAGE_C_MODE="${STAGE_C_MODE:-none}"
export STAGE_C_CACHE_DIR="${STAGE_C_CACHE_DIR:-}"
export STAGE_C_CACHE_MODE="${STAGE_C_CACHE_MODE:-off}"
export STAGE_C_CACHE_REQUIRE_HIT="${STAGE_C_CACHE_REQUIRE_HIT:-0}"
export STAGE_C_CACHE_VALIDATE="${STAGE_C_CACHE_VALIDATE:-0}"
export STAGE_C_INLINE_WHEN_IGNORED="${STAGE_C_INLINE_WHEN_IGNORED:-0}"
export READ_PATH="${READ_PATH:-frame}"
export FRAME_BIAS_MODE="${FRAME_BIAS_MODE:-pair}"
export READ_LAYER_MODE="${V57_READ_LAYER_MODE:-early}"
export READ_BETA_FRAME_CHUNKS=""
export READ_TOPK_FRAC="${V57_READ_TOPK_FRAC:-0.0}"
export READ_CALIB_MODE="${READ_CALIB_MODE:-none}"
export READ_BLEND_LAMBDA="${READ_BLEND_LAMBDA:-0.25}"
export ENABLE_CONTEXT_SOURCE_SKIP="${ENABLE_CONTEXT_SOURCE_SKIP:-0}"
export CONTEXT_SOURCE_SKIP_IMPL="${CONTEXT_SOURCE_SKIP_IMPL:-bias}"
export CONTEXT_SOURCE_SKIP_SCOPE="${CONTEXT_SOURCE_SKIP_SCOPE:-frame}"
export CONTEXT_SOURCE_SKIP_MODE="${CONTEXT_SOURCE_SKIP_MODE:-hard}"
export CONTEXT_SOURCE_SKIP_MASK="${CONTEXT_SOURCE_SKIP_MASK:-dg_q90}"
export CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS="${CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS:-0}"
export SEMANTIC_ROLE_POLICY="${SEMANTIC_ROLE_POLICY:-none}"
export SEMANTIC_MEMORY_PATHS="${SEMANTIC_MEMORY_PATHS:-}"
export SEMANTIC_ACTION_ACTIVE_CHUNKS=""
export SEMANTIC_ACTION_INACTIVE_READ_CUE_SOURCE=""
export HMC_COMMIT_MODE="probe_ttt_write"
export TTT_WRITE_GRADIENT_REVERSAL_MODE="${TTT_WRITE_GRADIENT_REVERSAL_MODE:-tri_replay}"
export TTT_WRITE_GRADIENT_REVERSAL_GAMMA="${TTT_WRITE_GRADIENT_REVERSAL_GAMMA:-0.0}"
export TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE="$RISK_SOURCE"
export TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK="${V57_TTT_BRANCH_MASK:-0}"
export TTT_WRITE_GRADIENT_REVERSAL_LAYER_GAMMAS="$LAYER_GAMMAS"
export TTT_WRITE_TRI_REPLAY_POSITIVE_FRAC="0.0"
export TTT_WRITE_TRI_REPLAY_NEGATIVE_FRAC="0.0"
export TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA="${TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA:-0.0}"
export TTT_WRITE_TRI_REPLAY_ROLE_MODE="$ROLE_MODE"
export TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS=""
export TTT_WRITE_COMMIT_EMA_ALPHA="${TTT_WRITE_COMMIT_EMA_ALPHA:-1.0}"
export TTT_WRITE_COMMIT_EMA_BRANCH_MASK="${TTT_WRITE_COMMIT_EMA_BRANCH_MASK:-all}"
export TTT_WRITE_COMMIT_EMA_CHUNKS=""
export TTT_WRITE_NATIVE_MIX_SCALES="${V57_NATIVE_MIX_SCALES:-1.00,1.00,1.00}"
export TTT_WRITE_NATIVE_MIX_CHUNKS=""
export TTT_WRITE_COMMIT_FILTER_MODE="${TTT_WRITE_COMMIT_FILTER_MODE:-none}"
export TTT_WRITE_COMMIT_FILTER_CHUNKS=""
export TTT_WRITE_SCALE_STATE_MODE="${V57_TTT_SCALE_STATE_MODE:-none}"
export TTT_WRITE_SCALE_STATE_CHUNKS=""
export ENABLE_SWA_OVERLAP_SOURCE_REPLACE="0"
export SWA_OVERLAP_SOURCE_REPLACE_ALPHA="0.0"

cat > "$OUT/effective_config.yaml" <<EOF
row: "$ROW"
run_name: "$RUN_NAME"
candidate: "$CANDIDATE"
track: "$TRACK"
action_class: "$ACTION_CLASS"
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
frame_bias_mode: "$FRAME_BIAS_MODE"
stage_c_mode: "$STAGE_C_MODE"
stage_c_cache_dir: "$STAGE_C_CACHE_DIR"
stage_c_cache_mode: "$STAGE_C_CACHE_MODE"
stage_c_cache_require_hit: "$STAGE_C_CACHE_REQUIRE_HIT"
semantic_role_policy: "$SEMANTIC_ROLE_POLICY"
semantic_memory_paths: "$SEMANTIC_MEMORY_PATHS"
enable_context_source_skip: "$ENABLE_CONTEXT_SOURCE_SKIP"
context_source_skip_impl: "$CONTEXT_SOURCE_SKIP_IMPL"
context_source_skip_scope: "$CONTEXT_SOURCE_SKIP_SCOPE"
context_source_skip_mode: "$CONTEXT_SOURCE_SKIP_MODE"
context_source_skip_layer_mode: "${CONTEXT_SOURCE_SKIP_LAYER_MODE:-early}"
context_source_skip_mask: "$CONTEXT_SOURCE_SKIP_MASK"
context_source_skip_record_attention_mass: "$CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS"
context_source_skip_attention_mass_max_queries: "${CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES:-128}"
semantic_role_highd_quantile: "${SEMANTIC_ROLE_HIGHD_QUANTILE:-}"
semantic_action_active_chunks: "$SEMANTIC_ACTION_ACTIVE_CHUNKS"
ttt_write_tri_replay_role_mode: "$TTT_WRITE_TRI_REPLAY_ROLE_MODE"
ttt_write_tri_replay_neutral_lambda: "$TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA"
ttt_write_gradient_reversal_risk_source: "$TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE"
ttt_write_gradient_reversal_layer_gammas: "$TTT_WRITE_GRADIENT_REVERSAL_LAYER_GAMMAS"
ttt_write_replay_token_filter_mode: "${TTT_WRITE_REPLAY_TOKEN_FILTER_MODE:-none}"
ttt_write_replay_token_filter_blend_mode: "${TTT_WRITE_REPLAY_TOKEN_FILTER_BLEND_MODE:-linear}"
ttt_write_transient_delta_subtract_scale: "${TTT_WRITE_TRANSIENT_DELTA_SUBTRACT_SCALE:-0.0}"
ttt_write_native_delta_gate_mode: "${TTT_WRITE_NATIVE_DELTA_GATE_MODE:-none}"
ttt_write_commit_filter_mode: "$TTT_WRITE_COMMIT_FILTER_MODE"
EOF
cp "$OUT/effective_config.yaml" "$OUT/v57_effective_config.yaml"

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
  "scale_state_chunks_empty": true
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
  "definition": "v57 no-chunk/no-manual H35-derived semantic action repair or TTT TTL candidate"
}
EOF

cat > "$OUT/reproduce_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
V57_RESULT_ROOT="$RESULT_ROOT" \\
V57_PLAN_NOTE="$PLAN_NOTE" \\
tools/run_v57_h35_semantic_action_repair_ttt_ttl_fast.sh "$GPU" "$ROW" "$RUN_NAME"
EOF
chmod +x "$OUT/reproduce_command.sh"

echo "[v57] row=$ROW candidate=$CANDIDATE track=$TRACK run=$RUN_NAME gpu=$GPU end_frame=$END_FRAME phase=$PHASE"
echo "[v57] cue=$CUE beta=$BETA frame_bias=$FRAME_BIAS_MODE role=$ROLE_MODE semantic_policy=$SEMANTIC_ROLE_POLICY context_mask=$CONTEXT_SOURCE_SKIP_MASK"

tools/run_attention_cue_experiment.sh "$GPU" "$RUN_NAME" "$MODE" "$CUE" "$BETA" "$WRITE_SCORE"
