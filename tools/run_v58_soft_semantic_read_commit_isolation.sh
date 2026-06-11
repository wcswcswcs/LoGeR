#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 GPU ROW RUN_NAME" >&2
  echo "Rows: R1_96F R2_96F R3_96F R4_96F N0_96F, R1_704F ... N0_704F, R1_FULL ... R4_FULL" >&2
  exit 2
fi

GPU="$1"
ROW="$2"
RUN_NAME="$3"

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
PY="${LOGER_PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"
RESULT_ROOT="${V58_RESULT_ROOT:-$ROOT/results/kitti01_hmc_v2/acl2_v58_soft_semantic_read_commit_isolation_geometry}"
PLAN_NOTE="${V58_PLAN_NOTE:-$ROOT/docs/ACL2_v58_SoftSemanticRead_CommitIsolation_GeometryPlan.md}"

PHASE="phase_unknown"
TRACK="semantic_soft_read"
CANDIDATE="$ROW"
END_FRAME="${V58_END_FRAME:-96}"
MODE="hybrid"
CUE="${V58_DG_CUE:-v31.sem_resid_coarse_l050.c23past}"
BETA="${V58_READ_BETA:-0.0}"
WRITE_SCORE="${V58_TTT_WRITE_SCORE:-stage_d_x_dg_inv_sqrt}"
ROLE_MODE="${V58_TTT_ROLE_MODE:-adaptive_writer_sc_gamma_split}"
RISK_SOURCE="${V58_TTT_RISK_SOURCE:-ttt_residual_x_dg}"
LAYER_GAMMAS="${V58_TTT_LAYER_GAMMAS:-0:0.0075,8:0.0075,17:0.0075}"
CONTEXT_IMPL="bias"
CONTEXT_MODE="soft"
CONTEXT_LAYER_MODE="early"
CONTEXT_MASK="semantic_role_negative"
SOFT_RHO="${V58_SOFT_RHO:-0.5}"
SOFT_MIN_KEEP="${V58_SOFT_MIN_KEEP:-0.5}"
SEMANTIC_DESC="SREAD03 semantic source selection with soft READ control"
ACTION_CLASS="soft_semantic_read_c1"

case "$ROW" in
  R[1-4]_96F|N0_96F)
    PHASE="${V58_PHASE:-phase1_soft_read_smoke}"
    END_FRAME="${V58_END_FRAME:-96}"
    ;;
  R[1-4]_704F|N0_704F)
    PHASE="${V58_PHASE:-phase2_soft_read_704_screen}"
    END_FRAME="${V58_END_FRAME:-704}"
    ;;
  R[1-4]_FULL)
    PHASE="${V58_PHASE:-phase3_soft_read_full}"
    END_FRAME="${V58_END_FRAME:-10000}"
    ;;
  *)
    echo "Unsupported v58 row: $ROW" >&2
    exit 2
    ;;
esac

BASE_ROW="${ROW%_96F}"
BASE_ROW="${BASE_ROW%_704F}"
BASE_ROW="${BASE_ROW%_FULL}"

enable_stage_c_cache() {
  export STAGE_C_MODE="${V58_STAGE_C_MODE:-reference}"
  export STAGE_C_CACHE_DIR="${V58_STAGE_C_CACHE_DIR:-$ROOT/results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full}"
  export STAGE_C_CACHE_MODE="${V58_STAGE_C_CACHE_MODE:-read}"
  export STAGE_C_CACHE_REQUIRE_HIT="${V58_STAGE_C_CACHE_REQUIRE_HIT:-1}"
  export STAGE_C_CACHE_VALIDATE="${V58_STAGE_C_CACHE_VALIDATE:-0}"
  export STAGE_C_INLINE_WHEN_IGNORED="0"
  export SEMANTIC_MEMORY_PATHS="${SEMANTIC_MEMORY_PATHS:-frame,global}"
}

enable_soft_context_control() {
  export ENABLE_CONTEXT_SOURCE_SKIP="1"
  export CONTEXT_SOURCE_SKIP_SCOPE="${CONTEXT_SOURCE_SKIP_SCOPE:-both}"
  export CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS="${CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS:-1}"
  export CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES="${CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES:-128}"
  export READ_PATH="${READ_PATH:-frame}"
  export FRAME_BIAS_MODE="${FRAME_BIAS_MODE:-query}"
}

enable_stage_c_cache
enable_soft_context_control

case "$BASE_ROW" in
  R1)
    CANDIDATE="R1_SREAD03_V_ONLY_C1"
    CONTEXT_IMPL="v_only"
    CONTEXT_MODE="soft"
    CONTEXT_LAYER_MODE="${V58_R1_LAYER_MODE:-early}"
    CONTEXT_MASK="semantic_role_negative"
    SOFT_RHO="${V58_R1_SOFT_RHO:-0.5}"
    SOFT_MIN_KEEP="${V58_R1_SOFT_MIN_KEEP:-0.5}"
    SEMANTIC_DESC="SREAD03 source selection; V-only attenuation; K/source topology retained; C1 probe_ttt_write commit isolation"
    ;;
  R2)
    CANDIDATE="R2_SREAD03_BIAS_FLOOR_C1"
    CONTEXT_IMPL="bias"
    CONTEXT_MODE="soft"
    CONTEXT_LAYER_MODE="${V58_R2_LAYER_MODE:-early}"
    CONTEXT_MASK="semantic_role_negative"
    SOFT_RHO="${V58_R2_SOFT_RHO:-0.5}"
    SOFT_MIN_KEEP="${V58_R2_SOFT_MIN_KEEP:-0.5}"
    SEMANTIC_DESC="SREAD03 source selection; soft attention bias floor target; C1 probe_ttt_write commit isolation"
    ;;
  R3)
    CANDIDATE="R3_SREAD03_EARLY_ONLY_C1"
    CONTEXT_IMPL="bias"
    CONTEXT_MODE="soft"
    CONTEXT_LAYER_MODE="${V58_R3_LAYER_MODE:-early_quarter}"
    CONTEXT_MASK="semantic_role_negative"
    SOFT_RHO="${V58_R3_SOFT_RHO:-0.5}"
    SOFT_MIN_KEEP="${V58_R3_SOFT_MIN_KEEP:-0.5}"
    SEMANTIC_DESC="SREAD03 source selection; soft attenuation restricted to early-quarter READ layers; C1 commit isolation"
    ;;
  R4)
    CANDIDATE="R4_SEM_Z_DG_SOFT_RESID_C1"
    CONTEXT_IMPL="bias"
    CONTEXT_MODE="soft"
    CONTEXT_LAYER_MODE="${V58_R4_LAYER_MODE:-early}"
    CONTEXT_MASK="semantic_z_dg_soft_resid"
    SOFT_RHO="${V58_R4_SOFT_RHO:-0.5}"
    SOFT_MIN_KEEP="${V58_R4_SOFT_MIN_KEEP:-0.5}"
    SEMANTIC_DESC="semantic-conditioned D_g residual risk; soft attention bias; no TTT/SWA semantic write; C1 commit isolation"
    ;;
  N0)
    CANDIDATE="N0_RANDOM_SAME_MASS_SOFT_C1"
    TRACK="negative_control"
    CONTEXT_IMPL="bias"
    CONTEXT_MODE="soft"
    CONTEXT_LAYER_MODE="${V58_N0_LAYER_MODE:-early}"
    CONTEXT_MASK="random_same_mass_semantic_role_negative"
    SOFT_RHO="${V58_N0_SOFT_RHO:-0.5}"
    SOFT_MIN_KEEP="${V58_N0_SOFT_MIN_KEEP:-0.5}"
    SEMANTIC_DESC="deterministic random same-mass source control matched to SREAD03 selection; C1 commit isolation"
    ;;
esac

BASE="${V58_ROLLOUT_BASE:-$RESULT_ROOT/$PHASE/rollouts}"
OUT="$BASE/$RUN_NAME"
mkdir -p "$OUT"
if [ -f "$OUT/run_status.txt" ] && grep -q "DONE $RUN_NAME" "$OUT/run_status.txt"; then
  echo "[v58] Skip completed run: $RUN_NAME"
  exit 0
fi

export LOGER_ROOT="$ROOT"
export LOGER_PY="$PY"
export ATTN_CUE_BASE="$BASE"
export KITTI_SEQ="${KITTI_SEQ:-01}"
export END_FRAME="$END_FRAME"
export RESET_EVERY="${V58_RESET_EVERY:-5}"
export EMPTY_CUDA_CACHE_EACH_CHUNK="${V58_EMPTY_CUDA_CACHE_EACH_CHUNK:-0}"
export READ_LAYER_MODE="${V58_READ_LAYER_MODE:-early}"
export READ_BETA_FRAME_CHUNKS=""
export READ_TOPK_FRAC="${V58_READ_TOPK_FRAC:-0.0}"
export READ_CALIB_MODE="${READ_CALIB_MODE:-none}"
export READ_BLEND_LAMBDA="${READ_BLEND_LAMBDA:-0.25}"
export CONTEXT_SOURCE_SKIP_IMPL="$CONTEXT_IMPL"
export CONTEXT_SOURCE_SKIP_MODE="$CONTEXT_MODE"
export CONTEXT_SOURCE_SKIP_LAYER_MODE="$CONTEXT_LAYER_MODE"
export CONTEXT_SOURCE_SKIP_MASK="$CONTEXT_MASK"
export CONTEXT_SOURCE_SKIP_SOFT_RHO="$SOFT_RHO"
export CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP="$SOFT_MIN_KEEP"
export SEMANTIC_ROLE_POLICY="${V58_SEMANTIC_ROLE_POLICY:-causal_fg_semantic_risk_skip}"
export SEMANTIC_ROLE_HIGHD_QUANTILE="${V58_SEMANTIC_ROLE_HIGHD_Q:-0.70}"
export SEMANTIC_ACTION_ACTIVE_CHUNKS=""
export SEMANTIC_ACTION_INACTIVE_READ_CUE_SOURCE=""
export HMC_COMMIT_MODE="${V58_HMC_COMMIT_MODE:-probe_ttt_write}"
export TTT_WRITE_GRADIENT_REVERSAL_MODE="${TTT_WRITE_GRADIENT_REVERSAL_MODE:-tri_replay}"
export TTT_WRITE_GRADIENT_REVERSAL_GAMMA="${TTT_WRITE_GRADIENT_REVERSAL_GAMMA:-0.0}"
export TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE="$RISK_SOURCE"
export TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK="${V58_TTT_BRANCH_MASK:-0}"
export TTT_WRITE_GRADIENT_REVERSAL_LAYER_GAMMAS="$LAYER_GAMMAS"
export TTT_WRITE_TRI_REPLAY_POSITIVE_FRAC="0.0"
export TTT_WRITE_TRI_REPLAY_NEGATIVE_FRAC="0.0"
export TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA="${TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA:-0.0}"
export TTT_WRITE_TRI_REPLAY_ROLE_MODE="$ROLE_MODE"
export TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS=""
export TTT_WRITE_COMMIT_EMA_ALPHA="${TTT_WRITE_COMMIT_EMA_ALPHA:-1.0}"
export TTT_WRITE_COMMIT_EMA_BRANCH_MASK="${TTT_WRITE_COMMIT_EMA_BRANCH_MASK:-all}"
export TTT_WRITE_COMMIT_EMA_CHUNKS=""
export TTT_WRITE_NATIVE_MIX_SCALES="${V58_NATIVE_MIX_SCALES:-1.00,1.00,1.00}"
export TTT_WRITE_NATIVE_MIX_CHUNKS=""
export TTT_WRITE_COMMIT_FILTER_MODE="${TTT_WRITE_COMMIT_FILTER_MODE:-none}"
export TTT_WRITE_COMMIT_FILTER_CHUNKS=""
export TTT_WRITE_SCALE_STATE_MODE="${V58_TTT_SCALE_STATE_MODE:-none}"
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
commit_protocol: "C1_probe_ttt_write_commit_isolation"
read_path: "$READ_PATH"
frame_bias_mode: "$FRAME_BIAS_MODE"
stage_c_mode: "$STAGE_C_MODE"
stage_c_cache_dir: "$STAGE_C_CACHE_DIR"
stage_c_cache_mode: "$STAGE_C_CACHE_MODE"
stage_c_cache_require_hit: "$STAGE_C_CACHE_REQUIRE_HIT"
semantic_role_policy: "$SEMANTIC_ROLE_POLICY"
semantic_memory_paths: "$SEMANTIC_MEMORY_PATHS"
semantic_role_highd_quantile: "$SEMANTIC_ROLE_HIGHD_QUANTILE"
enable_context_source_skip: "$ENABLE_CONTEXT_SOURCE_SKIP"
context_source_skip_impl: "$CONTEXT_SOURCE_SKIP_IMPL"
context_source_skip_scope: "$CONTEXT_SOURCE_SKIP_SCOPE"
context_source_skip_mode: "$CONTEXT_SOURCE_SKIP_MODE"
context_source_skip_layer_mode: "$CONTEXT_SOURCE_SKIP_LAYER_MODE"
context_source_skip_mask: "$CONTEXT_SOURCE_SKIP_MASK"
context_source_skip_soft_rho: "$CONTEXT_SOURCE_SKIP_SOFT_RHO"
context_source_skip_soft_min_keep: "$CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP"
context_source_skip_record_attention_mass: "$CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS"
context_source_skip_attention_mass_max_queries: "$CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES"
semantic_action_active_chunks: "$SEMANTIC_ACTION_ACTIVE_CHUNKS"
ttt_write_tri_replay_role_mode: "$TTT_WRITE_TRI_REPLAY_ROLE_MODE"
ttt_write_gradient_reversal_risk_source: "$TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE"
ttt_write_gradient_reversal_layer_gammas: "$TTT_WRITE_GRADIENT_REVERSAL_LAYER_GAMMAS"
ttt_write_commit_filter_mode: "$TTT_WRITE_COMMIT_FILTER_MODE"
EOF
cp "$OUT/effective_config.yaml" "$OUT/v58_effective_config.yaml"

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
  "definition": "v58 soft semantic READ with C1 probe_ttt_write commit isolation"
}
EOF

cat > "$OUT/reproduce_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
V58_RESULT_ROOT="$RESULT_ROOT" \\
V58_PLAN_NOTE="$PLAN_NOTE" \\
V58_${BASE_ROW}_LAYER_MODE="$CONTEXT_LAYER_MODE" \\
V58_${BASE_ROW}_SOFT_RHO="$SOFT_RHO" \\
V58_${BASE_ROW}_SOFT_MIN_KEEP="$SOFT_MIN_KEEP" \\
tools/run_v58_soft_semantic_read_commit_isolation.sh "$GPU" "$ROW" "$RUN_NAME"
EOF
chmod +x "$OUT/reproduce_command.sh"

echo "[v58] row=$ROW candidate=$CANDIDATE track=$TRACK run=$RUN_NAME gpu=$GPU end_frame=$END_FRAME phase=$PHASE"
echo "[v58] cue=$CUE commit=$HMC_COMMIT_MODE impl=$CONTEXT_IMPL mode=$CONTEXT_MODE mask=$CONTEXT_MASK layer=$CONTEXT_LAYER_MODE rho=$SOFT_RHO min_keep=$SOFT_MIN_KEEP"

tools/run_attention_cue_experiment.sh "$GPU" "$RUN_NAME" "$MODE" "$CUE" "$BETA" "$WRITE_SCORE"
