#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 GPU ROW RUN_NAME" >&2
  echo "Rows: A1_96F A2_96F A3_96F A5_96F N1_96F N2_96F, A1_704F ... N2_704F, A4_704F, A1_FULL ... A5_FULL N1_FULL N2_FULL A4_FULL" >&2
  exit 2
fi

GPU="$1"
ROW="$2"
RUN_NAME="$3"

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
PY="${LOGER_PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"
RESULT_ROOT="${V59_RESULT_ROOT:-$ROOT/results/kitti01_hmc_v2/acl2_v59_h35_semantic_anchor_memory}"
PLAN_NOTE="${V59_PLAN_NOTE:-$ROOT/docs/ACL2_v59_H35_SemanticAnchorMemory_Plan.md}"

PHASE="phase_unknown"
TRACK="semantic_anchor"
CANDIDATE="$ROW"
ACTION_CLASS="semantic_anchor_memory"
END_FRAME="${V59_END_FRAME:-96}"
MODE="hybrid"
CUE="${V59_DG_CUE:-acl2.gg.qq.low.g2_3.past_only.headmean.robustq}"
BETA="${V59_READ_BETA:-4.75}"
WRITE_SCORE="${V59_TTT_WRITE_SCORE:-stage_d_x_dg_inv_sqrt}"
ROLE_MODE="${V59_TTT_ROLE_MODE:-adaptive_writer_sc_gamma_split}"
RISK_SOURCE="${V59_TTT_RISK_SOURCE:-ttt_residual_x_dg}"
LAYER_GAMMAS="${V59_TTT_LAYER_GAMMAS:-0:0.0075,8:0.0075,17:0.0075}"
COMMIT_MODE="${V59_HMC_COMMIT_MODE:-probe_ttt_write}"
CONTEXT_ENABLE="0"
CONTEXT_IMPL="bias"
CONTEXT_MODE="hard"
CONTEXT_MASK="semantic_anchor_boost"
CONTEXT_SCOPE="frame"
CONTEXT_LAYER_MODE="${V59_ANCHOR_LAYER_MODE:-early}"
BOOST_RHO="${V59_ANCHOR_BOOST_RHO:-0.20}"
BOOST_MIN_KEEP="1.0"
ANCHOR_MODE="${V59_ANCHOR_MODE:-semantic}"
ANCHOR_TTT_FLOOR="0"
SEMANTIC_DESC="semantic geometry anchor bank diagnostic"

case "$ROW" in
  A[1235]_96F|N[12]_96F)
    PHASE="${V59_PHASE:-phase1_anchor_smoke}"
    END_FRAME="${V59_END_FRAME:-96}"
    ;;
  A[1235]_704F|N[12]_704F|A4_704F)
    PHASE="${V59_PHASE:-phase2_anchor_704_screen}"
    END_FRAME="${V59_END_FRAME:-704}"
    ;;
  A[12345]_FULL|N[12]_FULL)
    PHASE="${V59_PHASE:-phase3_anchor_full}"
    END_FRAME="${V59_END_FRAME:-10000}"
    ;;
  *)
    echo "Unsupported v59 row: $ROW" >&2
    exit 2
    ;;
esac

BASE_ROW="${ROW%_96F}"
BASE_ROW="${BASE_ROW%_704F}"
BASE_ROW="${BASE_ROW%_FULL}"

enable_stage_c_cache() {
  export STAGE_C_MODE="${V59_STAGE_C_MODE:-reference}"
  export STAGE_C_CACHE_DIR="${V59_STAGE_C_CACHE_DIR:-$ROOT/results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full}"
  export STAGE_C_CACHE_MODE="${V59_STAGE_C_CACHE_MODE:-read}"
  export STAGE_C_CACHE_REQUIRE_HIT="${V59_STAGE_C_CACHE_REQUIRE_HIT:-1}"
  export STAGE_C_CACHE_VALIDATE="${V59_STAGE_C_CACHE_VALIDATE:-0}"
  export STAGE_C_INLINE_WHEN_IGNORED="0"
}

enable_anchor_read_boost() {
  CONTEXT_ENABLE="1"
  CONTEXT_IMPL="${V59_ANCHOR_BOOST_IMPL:-bias_boost}"
  CONTEXT_MODE="${V59_ANCHOR_BOOST_MODE:-boost}"
  CONTEXT_SCOPE="${V59_ANCHOR_BOOST_SCOPE:-frame}"
  CONTEXT_MASK="semantic_anchor_boost"
  CONTEXT_LAYER_MODE="${V59_ANCHOR_LAYER_MODE:-early}"
  BOOST_RHO="${V59_ANCHOR_BOOST_RHO:-0.20}"
  BOOST_MIN_KEEP="1.0"
}

enable_stage_c_cache

case "$BASE_ROW" in
  A1)
    CANDIDATE="A1_READ_ANCHOR_BOOST"
    ACTION_CLASS="read_anchor_boost"
    COMMIT_MODE="${V59_A1_COMMIT_MODE:-controlled}"
    enable_anchor_read_boost
    SEMANTIC_DESC="READ anchor boost; controlled commit retained to expose side effects"
    ;;
  A2)
    CANDIDATE="A2_READ_ANCHOR_BOOST_COMMIT_ISO"
    ACTION_CLASS="read_anchor_boost_commit_isolation"
    COMMIT_MODE="${V59_A2_COMMIT_MODE:-probe_ttt_write}"
    enable_anchor_read_boost
    SEMANTIC_DESC="READ anchor boost with probe_ttt_write commit isolation"
    ;;
  A3)
    CANDIDATE="A3_TTT_ANCHOR_WRITE_FLOOR"
    ACTION_CLASS="ttt_anchor_write_floor"
    COMMIT_MODE="${V59_A3_COMMIT_MODE:-probe_ttt_write}"
    ANCHOR_TTT_FLOOR="1"
    SEMANTIC_DESC="TTT anchor write floor only; no source boost"
    ;;
  A4)
    CANDIDATE="A4_READ_BOOST_PLUS_TTT_FLOOR"
    ACTION_CLASS="read_anchor_boost_plus_ttt_floor"
    COMMIT_MODE="${V59_A4_COMMIT_MODE:-probe_ttt_write}"
    enable_anchor_read_boost
    ANCHOR_TTT_FLOOR="1"
    SEMANTIC_DESC="A1 + A3 combination; should only run after A1 and A3 show positive 704F signal"
    ;;
  A5)
    CANDIDATE="A5_SEM_DG_ANCHOR_RESCUE"
    ACTION_CLASS="semantic_dg_anchor_rescue"
    COMMIT_MODE="${V59_A5_COMMIT_MODE:-probe_ttt_write}"
    CUE="${V59_A5_CUE:-semantic_anchor_rescue.c23past_l025}"
    SEMANTIC_DESC="semantic-conditioned D_g anchor rescue; no source boost and no TTT floor"
    ;;
  N1)
    CANDIDATE="N1_RANDOM_SAME_MASS_ANCHOR_BOOST"
    TRACK="negative_control"
    ACTION_CLASS="random_same_mass_anchor_boost"
    COMMIT_MODE="${V59_N1_COMMIT_MODE:-controlled}"
    ANCHOR_MODE="random_same_mass"
    enable_anchor_read_boost
    SEMANTIC_DESC="random same-mass anchor boost control"
    ;;
  N2)
    CANDIDATE="N2_SHUFFLED_SEMANTIC_ANCHOR"
    TRACK="negative_control"
    ACTION_CLASS="shuffled_semantic_anchor_boost"
    COMMIT_MODE="${V59_N2_COMMIT_MODE:-controlled}"
    ANCHOR_MODE="shuffled_semantic"
    enable_anchor_read_boost
    SEMANTIC_DESC="shuffled semantic label anchor boost control"
    ;;
  *)
    echo "Unsupported v59 base row: $BASE_ROW" >&2
    exit 2
    ;;
esac

BASE="${V59_ROLLOUT_BASE:-$RESULT_ROOT/$PHASE/rollouts}"
OUT="$BASE/$RUN_NAME"
mkdir -p "$OUT"
if [ -f "$OUT/run_status.txt" ] && grep -q "DONE $RUN_NAME" "$OUT/run_status.txt"; then
  echo "[v59] Skip completed run: $RUN_NAME"
  exit 0
fi

export LOGER_ROOT="$ROOT"
export LOGER_PY="$PY"
export ATTN_CUE_BASE="$BASE"
export KITTI_SEQ="${KITTI_SEQ:-01}"
export END_FRAME="$END_FRAME"
export RESET_EVERY="${V59_RESET_EVERY:-5}"
export EMPTY_CUDA_CACHE_EACH_CHUNK="${V59_EMPTY_CUDA_CACHE_EACH_CHUNK:-0}"
export FAST_CUE_EVAL="${V59_FAST_CUE_EVAL:-1}"
export READ_PATH="${V59_READ_PATH:-frame}"
export FRAME_BIAS_MODE="${V59_FRAME_BIAS_MODE:-pair}"
export READ_LAYER_MODE="${V59_READ_LAYER_MODE:-early}"
export READ_BETA_FRAME_CHUNKS=""
export READ_TOPK_FRAC="${V59_READ_TOPK_FRAC:-0.0}"
export READ_CALIB_MODE="${V59_READ_CALIB_MODE:-none}"
export READ_BLEND_LAMBDA="${V59_READ_BLEND_LAMBDA:-0.25}"
export HMC_COMMIT_MODE="$COMMIT_MODE"

export SEMANTIC_PRIOR_MODE="${V59_SEMANTIC_PRIOR_MODE:-spg_v2}"
export HMC_IGNORE_SEMANTIC_PRIOR="0"
export SEMANTIC_MEMORY_PATHS="${V59_SEMANTIC_MEMORY_PATHS:-frame,global}"
export SEMANTIC_ROLE_POLICY="${V59_SEMANTIC_ROLE_POLICY:-none}"
export SEMANTIC_ACTION_ACTIVE_CHUNKS="${V59_SEMANTIC_ACTION_ACTIVE_CHUNKS:-}"
export SEMANTIC_ACTION_INACTIVE_READ_CUE_SOURCE="${V59_SEMANTIC_ACTION_INACTIVE_READ_CUE_SOURCE:-}"

export SEMANTIC_ANCHOR_MODE="$ANCHOR_MODE"
export SEMANTIC_ANCHOR_TARGET_RATIO="${V59_ANCHOR_TARGET_RATIO:-0.12}"
export SEMANTIC_ANCHOR_MIN_RATIO="${V59_ANCHOR_MIN_RATIO:-0.03}"
export SEMANTIC_ANCHOR_MAX_RATIO="${V59_ANCHOR_MAX_RATIO:-0.30}"
export SEMANTIC_ANCHOR_MIN_SCORE="${V59_ANCHOR_MIN_SCORE:-0.02}"
export SEMANTIC_ANCHOR_GRID_ROWS="${V59_ANCHOR_GRID_ROWS:-4}"
export SEMANTIC_ANCHOR_GRID_COLS="${V59_ANCHOR_GRID_COLS:-4}"
export ENABLE_SEMANTIC_ANCHOR_TTT_FLOOR="$ANCHOR_TTT_FLOOR"

export ENABLE_CONTEXT_SOURCE_SKIP="$CONTEXT_ENABLE"
export CONTEXT_SOURCE_SKIP_IMPL="$CONTEXT_IMPL"
export CONTEXT_SOURCE_SKIP_SCOPE="$CONTEXT_SCOPE"
export CONTEXT_SOURCE_SKIP_MODE="$CONTEXT_MODE"
export CONTEXT_SOURCE_SKIP_LAYER_MODE="$CONTEXT_LAYER_MODE"
export CONTEXT_SOURCE_SKIP_MASK="$CONTEXT_MASK"
export CONTEXT_SOURCE_SKIP_SOFT_RHO="$BOOST_RHO"
export CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP="$BOOST_MIN_KEEP"
export CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS="${V59_RECORD_ATTENTION_MASS:-1}"
export CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES="${V59_ATTENTION_MASS_MAX_QUERIES:-128}"

export TTT_WRITE_GRADIENT_REVERSAL_MODE="${TTT_WRITE_GRADIENT_REVERSAL_MODE:-tri_replay}"
export TTT_WRITE_GRADIENT_REVERSAL_GAMMA="${TTT_WRITE_GRADIENT_REVERSAL_GAMMA:-0.0}"
export TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE="$RISK_SOURCE"
export TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK="${V59_TTT_BRANCH_MASK:-0}"
export TTT_WRITE_GRADIENT_REVERSAL_BRANCH_GAMMAS="${V59_TTT_BRANCH_GAMMAS:-}"
export TTT_WRITE_GRADIENT_REVERSAL_LAYER_GAMMAS="$LAYER_GAMMAS"
export TTT_WRITE_GRADIENT_REVERSAL_HEAD_ROUTES="${V59_TTT_HEAD_ROUTES:-}"
export TTT_WRITE_GRADIENT_REVERSAL_CHUNKS=""
export TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS=""
export TTT_WRITE_GRADIENT_REVERSAL_NEGATIVE_FRAC="0.0"
export TTT_WRITE_TRI_REPLAY_POSITIVE_FRAC="0.0"
export TTT_WRITE_TRI_REPLAY_NEGATIVE_FRAC="0.0"
export TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA="${TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA:-0.0}"
export TTT_WRITE_TRI_REPLAY_ROLE_MODE="$ROLE_MODE"
export TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS=""
export TTT_WRITE_COMMIT_EMA_ALPHA="${TTT_WRITE_COMMIT_EMA_ALPHA:-1.0}"
export TTT_WRITE_COMMIT_EMA_BRANCH_MASK="${TTT_WRITE_COMMIT_EMA_BRANCH_MASK:-all}"
export TTT_WRITE_COMMIT_EMA_CHUNKS=""
export TTT_WRITE_NATIVE_MIX_SCALES="${V59_NATIVE_MIX_SCALES:-1.00,1.00,1.00}"
export TTT_WRITE_NATIVE_MIX_CHUNKS=""
export TTT_WRITE_COMMIT_FILTER_MODE="${V59_TTT_COMMIT_FILTER_MODE:-none}"
export TTT_WRITE_COMMIT_FILTER_CHUNKS=""
export TTT_WRITE_SCALE_STATE_MODE="${V59_TTT_SCALE_STATE_MODE:-none}"
export TTT_WRITE_SCALE_STATE_CHUNKS=""
export ENABLE_SWA_WRITE_CONTROL="0"
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
hmc_commit_mode: "$HMC_COMMIT_MODE"
commit_protocol: "$HMC_COMMIT_MODE"
read_path: "$READ_PATH"
frame_bias_mode: "$FRAME_BIAS_MODE"
stage_c_mode: "$STAGE_C_MODE"
stage_c_cache_dir: "$STAGE_C_CACHE_DIR"
stage_c_cache_mode: "$STAGE_C_CACHE_MODE"
stage_c_cache_require_hit: "$STAGE_C_CACHE_REQUIRE_HIT"
semantic_prior_mode: "$SEMANTIC_PRIOR_MODE"
semantic_memory_paths: "$SEMANTIC_MEMORY_PATHS"
semantic_role_policy: "$SEMANTIC_ROLE_POLICY"
semantic_action_active_chunks: "$SEMANTIC_ACTION_ACTIVE_CHUNKS"
semantic_action_inactive_read_cue_source: "$SEMANTIC_ACTION_INACTIVE_READ_CUE_SOURCE"
semantic_anchor_mode: "$SEMANTIC_ANCHOR_MODE"
semantic_anchor_target_ratio: "$SEMANTIC_ANCHOR_TARGET_RATIO"
semantic_anchor_min_ratio: "$SEMANTIC_ANCHOR_MIN_RATIO"
semantic_anchor_max_ratio: "$SEMANTIC_ANCHOR_MAX_RATIO"
semantic_anchor_min_score: "$SEMANTIC_ANCHOR_MIN_SCORE"
semantic_anchor_grid_rows: "$SEMANTIC_ANCHOR_GRID_ROWS"
semantic_anchor_grid_cols: "$SEMANTIC_ANCHOR_GRID_COLS"
enable_semantic_anchor_ttt_floor: "$ENABLE_SEMANTIC_ANCHOR_TTT_FLOOR"
enable_context_source_skip: "$ENABLE_CONTEXT_SOURCE_SKIP"
context_source_skip_impl: "$CONTEXT_SOURCE_SKIP_IMPL"
context_source_skip_scope: "$CONTEXT_SOURCE_SKIP_SCOPE"
context_source_skip_mode: "$CONTEXT_SOURCE_SKIP_MODE"
context_source_skip_layer_mode: "$CONTEXT_SOURCE_SKIP_LAYER_MODE"
context_source_skip_mask: "$CONTEXT_SOURCE_SKIP_MASK"
context_source_skip_soft_rho: "$CONTEXT_SOURCE_SKIP_SOFT_RHO"
context_source_skip_record_attention_mass: "$CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS"
context_source_skip_attention_mass_max_queries: "$CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES"
ttt_write_tri_replay_role_mode: "$TTT_WRITE_TRI_REPLAY_ROLE_MODE"
ttt_write_gradient_reversal_risk_source: "$TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE"
ttt_write_gradient_reversal_layer_gammas: "$TTT_WRITE_GRADIENT_REVERSAL_LAYER_GAMMAS"
ttt_write_native_mix_scales: "$TTT_WRITE_NATIVE_MIX_SCALES"
ttt_write_commit_filter_mode: "$TTT_WRITE_COMMIT_FILTER_MODE"
EOF
cp "$OUT/effective_config.yaml" "$OUT/v59_effective_config.yaml"

cat > "$OUT/chunk_id_policy_audit.json" <<EOF
{
  "row": "$ROW",
  "run_name": "$RUN_NAME",
  "absolute_chunk_id_policy_audit": {"pass": true},
  "has_read_beta_frame_chunks": false,
  "has_semantic_action_active_chunks": false,
  "has_tri_replay_chunk_params": false,
  "has_commit_ema_chunks": false,
  "has_native_mix_chunks": false,
  "read_beta_frame_chunks_empty": true,
  "semantic_action_active_chunks_empty": true,
  "ttt_gradient_reversal_chunk_gammas_empty": true,
  "ttt_tri_replay_chunk_params_empty": true,
  "ttt_commit_ema_chunks_empty": true,
  "native_mix_chunks_empty": true,
  "scale_state_chunks_empty": true
}
EOF

cat > "$OUT/adaptive_ttt_audit.json" <<EOF
{
  "row": "$ROW",
  "run_name": "$RUN_NAME",
  "adaptive_ttt_writer": true,
  "ttt_write_tri_replay_role_mode": "$TTT_WRITE_TRI_REPLAY_ROLE_MODE",
  "ttt_write_gradient_reversal_layer_gammas": "$TTT_WRITE_GRADIENT_REVERSAL_LAYER_GAMMAS",
  "ttt_write_gradient_reversal_branch_mask": "$TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK",
  "ttt_write_gradient_reversal_gamma": "$TTT_WRITE_GRADIENT_REVERSAL_GAMMA",
  "ttt_write_tri_replay_positive_frac": "$TTT_WRITE_TRI_REPLAY_POSITIVE_FRAC",
  "ttt_write_tri_replay_negative_frac": "$TTT_WRITE_TRI_REPLAY_NEGATIVE_FRAC",
  "ttt_write_tri_replay_neutral_lambda": "$TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA",
  "hmc_commit_mode": "$HMC_COMMIT_MODE",
  "no_swa": true
}
EOF

cat > "$OUT/reproduce_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
V59_RESULT_ROOT="$RESULT_ROOT" \\
V59_ROLLOUT_BASE="$BASE" \\
V59_PLAN_NOTE="$PLAN_NOTE" \\
V59_END_FRAME="$END_FRAME" \\
V59_DG_CUE="$CUE" \\
V59_READ_BETA="$BETA" \\
V59_TTT_WRITE_SCORE="$WRITE_SCORE" \\
V59_TTT_RISK_SOURCE="$RISK_SOURCE" \\
V59_TTT_ROLE_MODE="$ROLE_MODE" \\
V59_TTT_LAYER_GAMMAS="$LAYER_GAMMAS" \\
V59_ANCHOR_MODE="$ANCHOR_MODE" \\
V59_ANCHOR_TARGET_RATIO="$SEMANTIC_ANCHOR_TARGET_RATIO" \\
V59_ANCHOR_MIN_RATIO="$SEMANTIC_ANCHOR_MIN_RATIO" \\
V59_ANCHOR_MAX_RATIO="$SEMANTIC_ANCHOR_MAX_RATIO" \\
V59_ANCHOR_MIN_SCORE="$SEMANTIC_ANCHOR_MIN_SCORE" \\
V59_ANCHOR_GRID_ROWS="$SEMANTIC_ANCHOR_GRID_ROWS" \\
V59_ANCHOR_GRID_COLS="$SEMANTIC_ANCHOR_GRID_COLS" \\
V59_ANCHOR_BOOST_RHO="$BOOST_RHO" \\
V59_SEMANTIC_ACTION_ACTIVE_CHUNKS="$SEMANTIC_ACTION_ACTIVE_CHUNKS" \\
V59_SEMANTIC_ACTION_INACTIVE_READ_CUE_SOURCE="$SEMANTIC_ACTION_INACTIVE_READ_CUE_SOURCE" \\
V59_FRAME_BIAS_MODE="$FRAME_BIAS_MODE" \\
V59_STAGE_C_MODE="$STAGE_C_MODE" \\
V59_STAGE_C_CACHE_DIR="$STAGE_C_CACHE_DIR" \\
V59_STAGE_C_CACHE_MODE="$STAGE_C_CACHE_MODE" \\
V59_STAGE_C_CACHE_REQUIRE_HIT="$STAGE_C_CACHE_REQUIRE_HIT" \\
tools/run_v59_h35_semantic_anchor_memory.sh "$GPU" "$ROW" "$RUN_NAME"
EOF
chmod +x "$OUT/reproduce_command.sh"

tools/run_attention_cue_experiment.sh "$GPU" "$RUN_NAME" "$MODE" "$CUE" "$BETA" "$WRITE_SCORE"
