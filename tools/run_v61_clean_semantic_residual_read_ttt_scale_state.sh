#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 GPU ROW RUN_NAME" >&2
  echo "Rows: A0_96F A1_96F A2_96F A3_96F A4_96F NA1_96F NA2_96F NA3_96F NA4_96F B0_96F B1_96F B2_96F B4_96F NB1_96F NB2_96F NB3_96F; also _256F, _704F, _FULL for supported rows" >&2
  exit 2
fi

GPU="$1"
ROW="$2"
RUN_NAME="$3"

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
PY="${LOGER_PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"
RESULT_ROOT="${V61_RESULT_ROOT:-$ROOT/results/kitti01_hmc_v2/acl2_v61_clean_semantic_residual_read_ttt_scale_state}"
PLAN_NOTE="${V61_PLAN_NOTE:-$ROOT/docs/ACL2_v61_CleanSemanticResidualRead_TTTWriting_ScaleState_Plan.md}"

PHASE="phase_unknown"
END_FRAME="${V61_END_FRAME:-96}"
MODE="hybrid"
TRACK="baseline"
CANDIDATE="$ROW"
ACTION_CLASS="h35_clean"
IMPLEMENTATION_NOTE="H35 clean adaptive writer/read baseline"

CUE="${V61_DG_CUE:-acl2.gg.qq.low.g2_3.past_only.headmean.robustq}"
BETA="${V61_READ_BETA:-4.75}"
WRITE_SCORE="${V61_TTT_WRITE_SCORE:-stage_d_x_dg_inv_sqrt}"
ROLE_MODE="${V61_TTT_ROLE_MODE:-adaptive_writer_sc_gamma_split}"
RISK_SOURCE="${V61_TTT_RISK_SOURCE:-ttt_residual_x_dg}"
LAYER_GAMMAS="${V61_TTT_LAYER_GAMMAS:-0:0.0075,8:0.0075,17:0.0075}"
COMMIT_MODE="${V61_HMC_COMMIT_MODE:-probe_ttt_write}"

CONTEXT_ENABLE="0"
CONTEXT_IMPL="bias"
CONTEXT_MODE="hard"
CONTEXT_MASK="dg_q90"
CONTEXT_SCOPE="frame"
CONTEXT_LAYER_MODE="${V61_READ_LAYER_MODE:-early}"
SOFT_RHO="${V61_SOFT_RHO:-0.50}"
SOFT_MIN_KEEP="${V61_SOFT_MIN_KEEP:-0.50}"
ANCHOR_MODE="${V61_ANCHOR_MODE:-semantic}"
ANCHOR_TTT_FLOOR="0"
SEMANTIC_ROLE_POLICY="none"
SEMANTIC_MEMORY_PATHS=""
SEMANTIC_ROLE_POSITIVE_SCALE="${V61_SEMANTIC_ROLE_POSITIVE_SCALE:-1.05}"
SEMANTIC_ROLE_NEUTRAL_SCALE="${V61_SEMANTIC_ROLE_NEUTRAL_SCALE:-0.85}"
SEMANTIC_ROLE_NEGATIVE_SCALE="${V61_SEMANTIC_ROLE_NEGATIVE_SCALE:-0.65}"
STAGE_C_ON="0"

case "$ROW" in
  *_96F)
    PHASE="${V61_PHASE:-phase1_smoke_96f}"
    END_FRAME="${V61_END_FRAME:-96}"
    ;;
  *_256F)
    PHASE="${V61_PHASE:-phase1_smoke_256f}"
    END_FRAME="${V61_END_FRAME:-256}"
    ;;
  *_704F)
    PHASE="${V61_PHASE:-phase2_704_screen}"
    END_FRAME="${V61_END_FRAME:-704}"
    ;;
  *_FULL)
    PHASE="${V61_PHASE:-phase4_full}"
    END_FRAME="${V61_END_FRAME:-10000}"
    ;;
  *)
    echo "Unsupported v61 row suffix: $ROW" >&2
    exit 2
    ;;
esac

BASE_ROW="${ROW%_96F}"
BASE_ROW="${BASE_ROW%_256F}"
BASE_ROW="${BASE_ROW%_704F}"
BASE_ROW="${BASE_ROW%_FULL}"

enable_stage_c_cache() {
  STAGE_C_ON="1"
  export STAGE_C_MODE="${V61_STAGE_C_MODE:-reference}"
  export STAGE_C_CACHE_DIR="${V61_STAGE_C_CACHE_DIR:-$ROOT/results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full}"
  export STAGE_C_CACHE_MODE="${V61_STAGE_C_CACHE_MODE:-read}"
  export STAGE_C_CACHE_REQUIRE_HIT="${V61_STAGE_C_CACHE_REQUIRE_HIT:-1}"
  export STAGE_C_CACHE_VALIDATE="${V61_STAGE_C_CACHE_VALIDATE:-0}"
  export STAGE_C_INLINE_WHEN_IGNORED="0"
  export SEMANTIC_PRIOR_MODE="${V61_SEMANTIC_PRIOR_MODE:-spg_v2}"
  export HMC_IGNORE_SEMANTIC_PRIOR="0"
}

enable_soft_read() {
  CONTEXT_ENABLE="1"
  CONTEXT_IMPL="${1:-bias}"
  CONTEXT_MODE="${2:-soft}"
  CONTEXT_MASK="${3:-semantic_role_negative}"
  CONTEXT_SCOPE="${4:-frame}"
  CONTEXT_LAYER_MODE="${5:-early}"
  SOFT_RHO="${6:-0.50}"
  SOFT_MIN_KEEP="${7:-0.50}"
}

enable_semantic_roles_for_ttt() {
  enable_stage_c_cache
  SEMANTIC_ROLE_POLICY="${V61_SEMANTIC_ROLE_POLICY:-causal_fg_semantic_risk_skip}"
  SEMANTIC_MEMORY_PATHS="${V61_SEMANTIC_MEMORY_PATHS:-frame,global,ttt}"
}

case "$BASE_ROW" in
  A0|B0|NB3)
    CANDIDATE="${BASE_ROW}_H35_CLEAN_REPEAT"
    TRACK="baseline_or_geometry_control"
    ACTION_CLASS="h35_clean_geometry_only"
    IMPLEMENTATION_NOTE="Clean H35 geometry-only READ and adaptive TTT writer; semantic runtime disabled."
    ;;
  A1)
    CANDIDATE="A1_SEM_RESID_C23_READ"
    TRACK="track_a_semantic_read"
    ACTION_CLASS="semantic_residual_read"
    enable_stage_c_cache
    enable_soft_read "bias" "soft" "semantic_z_dg_soft_resid" "frame" "${V61_A1_LAYER_MODE:-early}" "${V61_A1_SOFT_RHO:-0.50}" "${V61_A1_SOFT_MIN_KEEP:-0.50}"
    CUE="${V61_A1_CUE:-v31.sem_resid_coarse_l050.c23past}"
    BETA="${V61_A1_BETA:-0.0}"
    IMPLEMENTATION_NOTE="Existing semantic_z_dg_soft_resid source-control path: semantic residual risk reconditions high-D source READ softly."
    ;;
  A2)
    CANDIDATE="A2_SEM_CONDITIONED_DG_READ"
    TRACK="track_a_semantic_read"
    ACTION_CLASS="semantic_conditioned_dg_read"
    enable_stage_c_cache
    CUE="${V61_A2_CUE:-v31.sem_resid_coarse_l050.c23past}"
    BETA="${V61_A2_BETA:-4.75}"
    IMPLEMENTATION_NOTE="Existing v31 semantic residual D_g cue used as READ bias; no TTT semantic writing."
    ;;
  A3)
    CANDIDATE="A3_SEM_ANCHOR_RESCUE_READ"
    TRACK="track_a_semantic_read"
    ACTION_CLASS="semantic_anchor_rescue_read"
    enable_stage_c_cache
    enable_soft_read "bias_boost" "boost" "semantic_anchor_boost" "frame" "${V61_A3_LAYER_MODE:-early}" "${V61_A3_BOOST_RHO:-0.20}" "1.0"
    IMPLEMENTATION_NOTE="Existing semantic anchor bank boosts stable anchor source READ; no transient risk boost."
    ;;
  A4)
    CANDIDATE="A4_SEM_TRANSIENT_RISK_READ"
    TRACK="track_a_semantic_read"
    ACTION_CLASS="semantic_transient_risk_read"
    enable_stage_c_cache
    enable_soft_read "bias" "soft" "semantic_role_negative" "frame" "${V61_A4_LAYER_MODE:-early}" "${V61_A4_SOFT_RHO:-0.50}" "${V61_A4_SOFT_MIN_KEEP:-0.50}"
    SEMANTIC_ROLE_POLICY="${V61_SEMANTIC_ROLE_POLICY:-causal_fg_semantic_risk_skip}"
    SEMANTIC_MEMORY_PATHS="${V61_SEMANTIC_MEMORY_PATHS:-frame,global}"
    CUE="${V61_A4_CUE:-acl2.gg.qq.low.g2_3.past_only.headmean.robustq}"
    BETA="${V61_A4_BETA:-0.0}"
    IMPLEMENTATION_NOTE="Existing semantic_role_negative soft READ attenuation as transient-risk READ proxy."
    ;;
  NA1)
    CANDIDATE="NA1_RANDOM_SAME_MASS_READ"
    TRACK="track_a_negative_control"
    ACTION_CLASS="random_same_mass_read"
    enable_stage_c_cache
    enable_soft_read "bias" "soft" "random_same_mass_semantic_role_negative" "frame" "${V61_NA1_LAYER_MODE:-early}" "${V61_NA1_SOFT_RHO:-0.50}" "${V61_NA1_SOFT_MIN_KEEP:-0.50}"
    SEMANTIC_ROLE_POLICY="${V61_SEMANTIC_ROLE_POLICY:-causal_fg_semantic_risk_skip}"
    SEMANTIC_MEMORY_PATHS="${V61_SEMANTIC_MEMORY_PATHS:-frame,global}"
    BETA="${V61_NA1_BETA:-0.0}"
    IMPLEMENTATION_NOTE="Deterministic random source tokens matched to semantic_role_negative mass."
    ;;
  NA2)
    CANDIDATE="NA2_SHUFFLED_SEMANTIC_READ"
    TRACK="track_a_negative_control"
    ACTION_CLASS="shuffled_semantic_anchor_read"
    enable_stage_c_cache
    ANCHOR_MODE="shuffled_semantic"
    enable_soft_read "bias_boost" "boost" "semantic_anchor_boost" "frame" "${V61_NA2_LAYER_MODE:-early}" "${V61_NA2_BOOST_RHO:-0.20}" "1.0"
    IMPLEMENTATION_NOTE="Shuffled semantic labels through existing anchor-bank selector; matched to A3-style READ action."
    ;;
  NA3)
    CANDIDATE="NA3_GEOMETRY_ONLY_RESIDUAL_READ"
    TRACK="track_a_negative_control"
    ACTION_CLASS="geometry_only_residual_read"
    enable_soft_read "bias" "soft" "dg_q80" "frame" "${V61_NA3_LAYER_MODE:-early}" "${V61_NA3_SOFT_RHO:-0.50}" "${V61_NA3_SOFT_MIN_KEEP:-0.50}"
    BETA="${V61_NA3_BETA:-0.0}"
    IMPLEMENTATION_NOTE="Geometry-only high-D residual soft READ control; Stage C disabled."
    ;;
  NA4)
    CANDIDATE="NA4_SEMANTIC_ONLY_READ"
    TRACK="track_a_negative_control"
    ACTION_CLASS="semantic_only_anchor_read"
    enable_stage_c_cache
    ANCHOR_MODE="${V61_NA4_ANCHOR_MODE:-semantic}"
    enable_soft_read "bias_boost" "boost" "semantic_anchor_boost" "frame" "${V61_NA4_LAYER_MODE:-early}" "${V61_NA4_BOOST_RHO:-0.20}" "1.0"
    CUE="${V61_NA4_CUE:-acl2.gg.qq.low.g2_3.past_only.headmean.robustq}"
    IMPLEMENTATION_NOTE="Semantic anchor bank READ boost without semantic-conditioned D_g cue; used as semantic-only-ish control because pointwise label-only READ is not separately implemented."
    ;;
  B1)
    CANDIDATE="B1_SEM_ANCHOR_WRITE_FLOOR"
    TRACK="track_b_semantic_ttt"
    ACTION_CLASS="semantic_anchor_write_floor"
    enable_semantic_roles_for_ttt
    ANCHOR_TTT_FLOOR="1"
    IMPLEMENTATION_NOTE="Existing semantic anchor write floor modulates H35 TTT write eligibility."
    ;;
  B2)
    CANDIDATE="B2_SEM_TRANSIENT_RISK_BOOST"
    TRACK="track_b_semantic_ttt"
    ACTION_CLASS="semantic_transient_risk_ttt"
    enable_semantic_roles_for_ttt
    SEMANTIC_ROLE_NEGATIVE_SCALE="${V61_B2_NEGATIVE_SCALE:-0.45}"
    SEMANTIC_ROLE_POSITIVE_SCALE="${V61_B2_POSITIVE_SCALE:-1.00}"
    SEMANTIC_ROLE_NEUTRAL_SCALE="${V61_B2_NEUTRAL_SCALE:-0.90}"
    IMPLEMENTATION_NOTE="Existing semantic_role_negative TTT eligibility downweight as transient write-risk boost proxy; no hard no-long."
    ;;
  B4)
    CANDIDATE="B4_SEM_CONDITIONED_DG_TTT"
    TRACK="track_b_semantic_ttt"
    ACTION_CLASS="semantic_conditioned_dg_ttt"
    enable_semantic_roles_for_ttt
    CUE="${V61_B4_CUE:-v31.sem_resid_coarse_l050.c23past}"
    BETA="${V61_B4_BETA:-0.0}"
    IMPLEMENTATION_NOTE="Semantic-conditioned D_g cue plus semantic role modulation enters TTT risk/eligibility path."
    ;;
  NB1)
    CANDIDATE="NB1_RANDOM_SAME_MASS_TTT"
    TRACK="track_b_negative_control"
    ACTION_CLASS="random_same_mass_anchor_ttt"
    enable_semantic_roles_for_ttt
    ANCHOR_MODE="random_same_mass"
    ANCHOR_TTT_FLOOR="1"
    IMPLEMENTATION_NOTE="Random same-mass anchor write floor control matched to B1 anchor count."
    ;;
  NB2)
    CANDIDATE="NB2_SHUFFLED_SEMANTIC_TTT"
    TRACK="track_b_negative_control"
    ACTION_CLASS="shuffled_semantic_anchor_ttt"
    enable_semantic_roles_for_ttt
    ANCHOR_MODE="shuffled_semantic"
    ANCHOR_TTT_FLOOR="1"
    IMPLEMENTATION_NOTE="Shuffled semantic anchor write floor control matched to B1 action family."
    ;;
  *)
    echo "Unsupported v61 base row: $BASE_ROW" >&2
    exit 2
    ;;
esac

if [ "$STAGE_C_ON" = "0" ]; then
  export STAGE_C_MODE="none"
  export STAGE_C_CACHE_DIR=""
  export STAGE_C_CACHE_MODE="off"
  export STAGE_C_CACHE_REQUIRE_HIT="0"
  export STAGE_C_CACHE_VALIDATE="0"
  export STAGE_C_INLINE_WHEN_IGNORED="0"
  export SEMANTIC_PRIOR_MODE="spg_v2"
  export HMC_IGNORE_SEMANTIC_PRIOR="${V61_HMC_IGNORE_SEMANTIC_PRIOR:-0}"
fi

BASE="${V61_ROLLOUT_BASE:-$RESULT_ROOT/$PHASE/rollouts}"
OUT="$BASE/$RUN_NAME"
mkdir -p "$OUT"
if [ -f "$OUT/run_status.txt" ] && grep -q "DONE $RUN_NAME" "$OUT/run_status.txt"; then
  echo "[v61] Skip completed run: $RUN_NAME"
  exit 0
fi

export LOGER_ROOT="$ROOT"
export LOGER_PY="$PY"
export ATTN_CUE_BASE="$BASE"
export KITTI_SEQ="${KITTI_SEQ:-01}"
export END_FRAME="$END_FRAME"
export RESET_EVERY="${V61_RESET_EVERY:-5}"
export EMPTY_CUDA_CACHE_EACH_CHUNK="${V61_EMPTY_CUDA_CACHE_EACH_CHUNK:-0}"
export FAST_CUE_EVAL="${V61_FAST_CUE_EVAL:-1}"
export READ_PATH="${V61_READ_PATH:-frame}"
export FRAME_BIAS_MODE="${V61_FRAME_BIAS_MODE:-pair}"
export READ_LAYER_MODE="${V61_READ_LAYER_MODE:-early}"
export READ_BETA_FRAME_CHUNKS=""
export READ_TOPK_FRAC="${V61_READ_TOPK_FRAC:-0.0}"
export READ_CALIB_MODE="${V61_READ_CALIB_MODE:-none}"
export READ_BLEND_LAMBDA="${V61_READ_BLEND_LAMBDA:-0.25}"
export HMC_COMMIT_MODE="$COMMIT_MODE"

export SEMANTIC_MEMORY_PATHS="$SEMANTIC_MEMORY_PATHS"
export SEMANTIC_ROLE_POLICY="$SEMANTIC_ROLE_POLICY"
export SEMANTIC_ACTION_ACTIVE_CHUNKS=""
export SEMANTIC_ACTION_INACTIVE_READ_CUE_SOURCE=""
export SEMANTIC_ANCHOR_MODE="$ANCHOR_MODE"
export SEMANTIC_ANCHOR_TARGET_RATIO="${V61_ANCHOR_TARGET_RATIO:-0.12}"
export SEMANTIC_ANCHOR_MIN_RATIO="${V61_ANCHOR_MIN_RATIO:-0.03}"
export SEMANTIC_ANCHOR_MAX_RATIO="${V61_ANCHOR_MAX_RATIO:-0.30}"
export SEMANTIC_ANCHOR_MIN_SCORE="${V61_ANCHOR_MIN_SCORE:-0.02}"
export SEMANTIC_ANCHOR_GRID_ROWS="${V61_ANCHOR_GRID_ROWS:-4}"
export SEMANTIC_ANCHOR_GRID_COLS="${V61_ANCHOR_GRID_COLS:-4}"
export ENABLE_SEMANTIC_ANCHOR_TTT_FLOOR="$ANCHOR_TTT_FLOOR"
export SEMANTIC_ROLE_POSITIVE_SCALE="$SEMANTIC_ROLE_POSITIVE_SCALE"
export SEMANTIC_ROLE_NEUTRAL_SCALE="$SEMANTIC_ROLE_NEUTRAL_SCALE"
export SEMANTIC_ROLE_NEGATIVE_SCALE="$SEMANTIC_ROLE_NEGATIVE_SCALE"

export ENABLE_CONTEXT_SOURCE_SKIP="$CONTEXT_ENABLE"
export CONTEXT_SOURCE_SKIP_IMPL="$CONTEXT_IMPL"
export CONTEXT_SOURCE_SKIP_SCOPE="$CONTEXT_SCOPE"
export CONTEXT_SOURCE_SKIP_MODE="$CONTEXT_MODE"
export CONTEXT_SOURCE_SKIP_LAYER_MODE="$CONTEXT_LAYER_MODE"
export CONTEXT_SOURCE_SKIP_MASK="$CONTEXT_MASK"
export CONTEXT_SOURCE_SKIP_SOFT_RHO="$SOFT_RHO"
export CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP="$SOFT_MIN_KEEP"
export CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS="${V61_RECORD_ATTENTION_MASS:-1}"
export CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES="${V61_ATTENTION_MASS_MAX_QUERIES:-128}"

export TTT_WRITE_GRADIENT_REVERSAL_MODE="${TTT_WRITE_GRADIENT_REVERSAL_MODE:-tri_replay}"
export TTT_WRITE_GRADIENT_REVERSAL_GAMMA="${TTT_WRITE_GRADIENT_REVERSAL_GAMMA:-0.0}"
export TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE="$RISK_SOURCE"
export TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK="${V61_TTT_BRANCH_MASK:-0}"
export TTT_WRITE_GRADIENT_REVERSAL_BRANCH_GAMMAS="${V61_TTT_BRANCH_GAMMAS:-}"
export TTT_WRITE_GRADIENT_REVERSAL_LAYER_GAMMAS="$LAYER_GAMMAS"
export TTT_WRITE_GRADIENT_REVERSAL_HEAD_ROUTES="${V61_TTT_HEAD_ROUTES:-}"
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
export TTT_WRITE_NATIVE_MIX_SCALES="${V61_NATIVE_MIX_SCALES:-1.00,1.00,1.00}"
export TTT_WRITE_NATIVE_MIX_CHUNKS=""
export TTT_WRITE_COMMIT_FILTER_MODE="${V61_TTT_COMMIT_FILTER_MODE:-none}"
export TTT_WRITE_COMMIT_FILTER_CHUNKS=""
export TTT_WRITE_SCALE_STATE_MODE="${V61_TTT_SCALE_STATE_MODE:-none}"
export TTT_WRITE_SCALE_STATE_CHUNKS=""
export ENABLE_SWA_WRITE_CONTROL="0"
export ENABLE_SWA_OVERLAP_SOURCE_REPLACE="0"
export SWA_OVERLAP_SOURCE_REPLACE_ALPHA="0.0"

export OUTPUT_PT="$OUT/merged_outputs.pt"
export PER_CHUNK_GEOMETRY_DIR="$OUT/per_chunk_geometry"

cat > "$OUT/effective_config.yaml" <<EOF
row: "$ROW"
run_name: "$RUN_NAME"
candidate: "$CANDIDATE"
track: "$TRACK"
action_class: "$ACTION_CLASS"
implementation_note: "$IMPLEMENTATION_NOTE"
plan_note: "$PLAN_NOTE"
gpu: "$GPU"
end_frame: "$END_FRAME"
mode: "$MODE"
cue: "$CUE"
beta: "$BETA"
write_score: "$WRITE_SCORE"
hmc_commit_mode: "$HMC_COMMIT_MODE"
read_path: "$READ_PATH"
frame_bias_mode: "$FRAME_BIAS_MODE"
stage_c_mode: "$STAGE_C_MODE"
stage_c_cache_dir: "$STAGE_C_CACHE_DIR"
stage_c_cache_mode: "$STAGE_C_CACHE_MODE"
stage_c_cache_require_hit: "$STAGE_C_CACHE_REQUIRE_HIT"
semantic_prior_mode: "$SEMANTIC_PRIOR_MODE"
semantic_memory_paths: "$SEMANTIC_MEMORY_PATHS"
semantic_role_policy: "$SEMANTIC_ROLE_POLICY"
semantic_anchor_mode: "$SEMANTIC_ANCHOR_MODE"
enable_semantic_anchor_ttt_floor: "$ENABLE_SEMANTIC_ANCHOR_TTT_FLOOR"
semantic_role_positive_scale: "$SEMANTIC_ROLE_POSITIVE_SCALE"
semantic_role_neutral_scale: "$SEMANTIC_ROLE_NEUTRAL_SCALE"
semantic_role_negative_scale: "$SEMANTIC_ROLE_NEGATIVE_SCALE"
enable_context_source_skip: "$ENABLE_CONTEXT_SOURCE_SKIP"
context_source_skip_impl: "$CONTEXT_SOURCE_SKIP_IMPL"
context_source_skip_scope: "$CONTEXT_SOURCE_SKIP_SCOPE"
context_source_skip_mode: "$CONTEXT_SOURCE_SKIP_MODE"
context_source_skip_layer_mode: "$CONTEXT_SOURCE_SKIP_LAYER_MODE"
context_source_skip_mask: "$CONTEXT_SOURCE_SKIP_MASK"
context_source_skip_soft_rho: "$CONTEXT_SOURCE_SKIP_SOFT_RHO"
context_source_skip_soft_min_keep: "$CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP"
ttt_write_tri_replay_role_mode: "$TTT_WRITE_TRI_REPLAY_ROLE_MODE"
ttt_write_gradient_reversal_risk_source: "$TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE"
ttt_write_gradient_reversal_layer_gammas: "$TTT_WRITE_GRADIENT_REVERSAL_LAYER_GAMMAS"
output_pt: "$OUTPUT_PT"
per_chunk_geometry_dir: "$PER_CHUNK_GEOMETRY_DIR"
EOF
cp "$OUT/effective_config.yaml" "$OUT/v61_effective_config.yaml"

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
V61_RESULT_ROOT="$RESULT_ROOT" \\
V61_ROLLOUT_BASE="$BASE" \\
V61_PLAN_NOTE="$PLAN_NOTE" \\
V61_END_FRAME="$END_FRAME" \\
tools/run_v61_clean_semantic_residual_read_ttt_scale_state.sh "$GPU" "$ROW" "$RUN_NAME"
EOF
chmod +x "$OUT/reproduce_command.sh"

tools/run_attention_cue_experiment.sh "$GPU" "$RUN_NAME" "$MODE" "$CUE" "$BETA" "$WRITE_SCORE"

"$PY" "$ROOT/tools/v61_scale_metrics.py" \
  --run-dir "$OUT" \
  --geometry-dir "$OUT/per_chunk_geometry" \
  --out-dir "$OUT/scale_metrics" \
  > "$OUT/scale_metrics_summary.stdout.json"
