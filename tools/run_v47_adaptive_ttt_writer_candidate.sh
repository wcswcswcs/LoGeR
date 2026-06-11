#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 GPU ROW RUN_NAME" >&2
  echo "Rows: AW010_ADAPTIVE_TTT_ONLY AW110_FRAME_ADAPTIVE_TTT AW111_FRAME_ADAPTIVE_TTT_SWA" >&2
  exit 2
fi

GPU="$1"
ROW="$2"
RUN_NAME="$3"

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
PY="${LOGER_PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"
RESULT_ROOT="${V47_RESULT_ROOT:-$ROOT/results/kitti01_hmc_v2/acl2_v47_adaptive_ttt_writer_nochunk}"
BASE="${V47_ROLLOUT_BASE:-$RESULT_ROOT/phase1_adaptive_writer/rollouts}"
OUT="$BASE/$RUN_NAME"
PLAN_NOTE="${V47_PLAN_NOTE:-$ROOT/docs/ACL2_v46B_ComponentAttribution_FrameTTT_FrameSWA_Addendum.md}"

mkdir -p "$OUT"
if [ -f "$OUT/run_status.txt" ] && grep -q "DONE $RUN_NAME" "$OUT/run_status.txt"; then
  echo "[v47] Skip completed run: $RUN_NAME"
  exit 0
fi

FRAME_ON=0
SWA_ON=0
case "$ROW" in
  AW010_ADAPTIVE_TTT_ONLY) ;;
  AW110_FRAME_ADAPTIVE_TTT) FRAME_ON=1 ;;
  AW111_FRAME_ADAPTIVE_TTT_SWA) FRAME_ON=1; SWA_ON=1 ;;
  *)
    echo "Unsupported v47 row: $ROW" >&2
    exit 2
    ;;
esac

# Hard no-chunk policy: this experiment is not allowed to inherit C9's
# KITTI01-specific chunk maps.
export ATTN_CUE_BASE="$BASE"
export LOGER_ROOT="$ROOT"
export LOGER_PY="$PY"
export KITTI_SEQ="${KITTI_SEQ:-01}"
export STAGE_C_MODE="none"
export STAGE_C_CACHE_MODE="off"
export STAGE_C_CACHE_REQUIRE_HIT="0"
export EMPTY_CUDA_CACHE_EACH_CHUNK="${V47_EMPTY_CUDA_CACHE_EACH_CHUNK:-0}"
export SEMANTIC_ROLE_POLICY="none"
export SEMANTIC_MEMORY_PATHS=""
export READ_BETA_FRAME_CHUNKS=""
export TTT_WRITE_GRADIENT_REVERSAL_CHUNKS=""
export TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS=""
export TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS=""
export TTT_WRITE_COMMIT_EMA_CHUNKS=""
export TTT_WRITE_NATIVE_MIX_CHUNKS=""
export READ_TOPK_FRAC="${V47_READ_TOPK_FRAC:-0.0}"
export RESET_EVERY="${V47_RESET_EVERY:-5}"
export END_FRAME="${V47_END_FRAME:-10000}"

MODE="hybrid"
CUE="${V47_DG_CUE:-acl2.gg.qq.low.g2_3.past_only.headmean.robustq}"
BETA="${V47_READ_BETA:-4.75}"
WRITE_SCORE="${V47_TTT_WRITE_SCORE:-stage_d_x_dg_inv_sqrt}"

export READ_PATH="none"
export HMC_COMMIT_MODE="probe_ttt_write"
export TTT_WRITE_GRADIENT_REVERSAL_MODE="tri_replay"
export TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE="${V47_TTT_RISK_SOURCE:-update_conflict_energy}"
export TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK="${V47_TTT_BRANCH_MASK:-0}"
export TTT_WRITE_GRADIENT_REVERSAL_BRANCH_GAMMAS="${V47_TTT_BRANCH_GAMMAS:-}"
export TTT_WRITE_GRADIENT_REVERSAL_LAYER_GAMMAS="${V47_TTT_LAYER_GAMMAS:-}"
export TTT_WRITE_GRADIENT_REVERSAL_HEAD_ROUTES="${V47_TTT_HEAD_ROUTES:-}"

# Adaptive writer contract:
# - external gamma is zero;
# - manual tri-replay positive/negative/neutral percentages are zero and unused;
# - role assignment and negative write magnitude are inferred from the online
#   risk distribution inside ttt_write_controller.py.
export TTT_WRITE_GRADIENT_REVERSAL_GAMMA="0.0"
export TTT_WRITE_GRADIENT_REVERSAL_NEGATIVE_FRAC="0.0"
export TTT_WRITE_TRI_REPLAY_POSITIVE_FRAC="0.0"
export TTT_WRITE_TRI_REPLAY_NEGATIVE_FRAC="0.0"
export TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA="0.0"
export TTT_WRITE_TRI_REPLAY_ROLE_MODE="${V47_TTT_ROLE_MODE:-adaptive_writer_fused}"
if [[ "$TTT_WRITE_TRI_REPLAY_ROLE_MODE" == *split* ]]; then
  ADAPTIVE_WRITER_DEFINITION="online risk/prior adaptive role assignment + adaptive negative gamma/lambda + split tri replay"
else
  ADAPTIVE_WRITER_DEFINITION="online risk/prior adaptive role assignment + adaptive negative gamma/lambda + fused single replay"
fi

export TTT_WRITE_COMMIT_EMA_ALPHA="1.0"
export TTT_WRITE_COMMIT_EMA_BRANCH_MASK="all"
export TTT_WRITE_NATIVE_MIX_SCALES="${V47_NATIVE_MIX_SCALES:-1.00,1.00,1.00}"
export TTT_WRITE_NATIVE_MIX_CHUNKS=""
export TTT_WRITE_COMMIT_FILTER_MODE="${V47_TTT_COMMIT_FILTER_MODE:-none}"
export TTT_WRITE_COMMIT_FILTER_RISK_SOURCE="${V47_TTT_COMMIT_FILTER_RISK_SOURCE:-d_tok}"
export TTT_WRITE_COMMIT_FILTER_SCOPE="${V47_TTT_COMMIT_FILTER_SCOPE:-tail_overlap}"
export TTT_WRITE_COMMIT_FILTER_STAT="${V47_TTT_COMMIT_FILTER_STAT:-mean}"
export TTT_WRITE_COMMIT_FILTER_BASE="${V47_TTT_COMMIT_FILTER_BASE:-0.0}"
export TTT_WRITE_COMMIT_FILTER_GAIN="${V47_TTT_COMMIT_FILTER_GAIN:-1.0}"
export TTT_WRITE_COMMIT_FILTER_MIN="${V47_TTT_COMMIT_FILTER_MIN:-0.0}"
export TTT_WRITE_COMMIT_FILTER_MAX="${V47_TTT_COMMIT_FILTER_MAX:-1.0}"
export TTT_WRITE_COMMIT_FILTER_BRANCH_MASK="${V47_TTT_COMMIT_FILTER_BRANCH_MASK:-0}"
export TTT_WRITE_COMMIT_FILTER_CHUNKS=""
export TTT_WRITE_SCALE_STATE_MODE="${V47_TTT_SCALE_STATE_MODE:-none}"
export TTT_WRITE_SCALE_STATE_PROXY="${V47_TTT_SCALE_STATE_PROXY:-pose_step_ema}"
export TTT_WRITE_SCALE_STATE_CARRIER="${V47_TTT_SCALE_STATE_CARRIER:-all}"
export TTT_WRITE_SCALE_STATE_ALPHA="${V47_TTT_SCALE_STATE_ALPHA:-0.0}"
export TTT_WRITE_SCALE_STATE_BRANCH_MASK="${V47_TTT_SCALE_STATE_BRANCH_MASK:-0}"
export TTT_WRITE_SCALE_STATE_CHUNKS=""
export TTT_WRITE_SCALE_STATE_SAMPLE_TOKENS="${V47_TTT_SCALE_STATE_SAMPLE_TOKENS:-0}"
export ONLINE_SCALE_STATE_MODE="${V47_ONLINE_SCALE_STATE_MODE:-none}"
export ONLINE_SCALE_STATE_MIN="${V47_ONLINE_SCALE_STATE_MIN:-0.80}"
export ONLINE_SCALE_STATE_MAX="${V47_ONLINE_SCALE_STATE_MAX:-1.25}"

export ENABLE_SWA_OVERLAP_SOURCE_REPLACE="0"
export SWA_OVERLAP_SOURCE_REPLACE_ALPHA="0.0"
export SWA_OVERLAP_SOURCE_REPLACE_MODE="${V47_SWA_REPLACE_MODE:-source}"
export SWA_OVERLAP_SOURCE_REPLACE_TARGET="${V47_SWA_REPLACE_TARGET:-kv}"
export SWA_OVERLAP_SOURCE_REPLACE_LAYER_MODE="${V47_SWA_REPLACE_LAYER_MODE:-last}"

if [ "$FRAME_ON" -eq 1 ]; then
  export READ_PATH="frame"
  export FRAME_BIAS_MODE="${V47_FRAME_BIAS_MODE:-pair}"
  export READ_LAYER_MODE="${V47_READ_LAYER_MODE:-early}"
fi

if [ "$SWA_ON" -eq 1 ]; then
  export ENABLE_SWA_OVERLAP_SOURCE_REPLACE="1"
  export SWA_OVERLAP_SOURCE_REPLACE_ALPHA="${V47_SWA_ALPHA:-0.5}"
fi

cat > "$OUT/effective_config.yaml" <<EOF
row: "$ROW"
run_name: "$RUN_NAME"
plan_note: "$PLAN_NOTE"
frame_attn_expected: $FRAME_ON
adaptive_ttt_expected: 1
swa_expected: $SWA_ON
mode: "$MODE"
cue: "$CUE"
beta: "$BETA"
write_score: "$WRITE_SCORE"
read_path: "$READ_PATH"
hmc_commit_mode: "$HMC_COMMIT_MODE"
read_beta_frame_chunks: "$READ_BETA_FRAME_CHUNKS"
ttt_write_gradient_reversal_mode: "$TTT_WRITE_GRADIENT_REVERSAL_MODE"
ttt_write_gradient_reversal_gamma: "$TTT_WRITE_GRADIENT_REVERSAL_GAMMA"
ttt_write_gradient_reversal_risk_source: "$TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE"
ttt_write_gradient_reversal_branch_mask: "$TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK"
ttt_write_gradient_reversal_branch_gammas: "$TTT_WRITE_GRADIENT_REVERSAL_BRANCH_GAMMAS"
ttt_write_gradient_reversal_layer_gammas: "$TTT_WRITE_GRADIENT_REVERSAL_LAYER_GAMMAS"
ttt_write_gradient_reversal_head_routes: "$TTT_WRITE_GRADIENT_REVERSAL_HEAD_ROUTES"
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
ttt_write_commit_filter_mode: "$TTT_WRITE_COMMIT_FILTER_MODE"
ttt_write_commit_filter_risk_source: "$TTT_WRITE_COMMIT_FILTER_RISK_SOURCE"
ttt_write_commit_filter_scope: "$TTT_WRITE_COMMIT_FILTER_SCOPE"
ttt_write_commit_filter_stat: "$TTT_WRITE_COMMIT_FILTER_STAT"
ttt_write_commit_filter_base: "$TTT_WRITE_COMMIT_FILTER_BASE"
ttt_write_commit_filter_gain: "$TTT_WRITE_COMMIT_FILTER_GAIN"
ttt_write_commit_filter_min: "$TTT_WRITE_COMMIT_FILTER_MIN"
ttt_write_commit_filter_max: "$TTT_WRITE_COMMIT_FILTER_MAX"
ttt_write_commit_filter_branch_mask: "$TTT_WRITE_COMMIT_FILTER_BRANCH_MASK"
ttt_write_commit_filter_chunks: "$TTT_WRITE_COMMIT_FILTER_CHUNKS"
ttt_write_scale_state_mode: "$TTT_WRITE_SCALE_STATE_MODE"
ttt_write_scale_state_proxy: "$TTT_WRITE_SCALE_STATE_PROXY"
ttt_write_scale_state_carrier: "$TTT_WRITE_SCALE_STATE_CARRIER"
ttt_write_scale_state_alpha: "$TTT_WRITE_SCALE_STATE_ALPHA"
ttt_write_scale_state_branch_mask: "$TTT_WRITE_SCALE_STATE_BRANCH_MASK"
ttt_write_scale_state_chunks: "$TTT_WRITE_SCALE_STATE_CHUNKS"
ttt_write_scale_state_sample_tokens: "$TTT_WRITE_SCALE_STATE_SAMPLE_TOKENS"
online_scale_state_mode: "$ONLINE_SCALE_STATE_MODE"
online_scale_state_min: "$ONLINE_SCALE_STATE_MIN"
online_scale_state_max: "$ONLINE_SCALE_STATE_MAX"
enable_swa_overlap_source_replace: "$ENABLE_SWA_OVERLAP_SOURCE_REPLACE"
swa_overlap_source_replace_alpha: "$SWA_OVERLAP_SOURCE_REPLACE_ALPHA"
swa_overlap_source_replace_mode: "$SWA_OVERLAP_SOURCE_REPLACE_MODE"
swa_overlap_source_replace_target: "$SWA_OVERLAP_SOURCE_REPLACE_TARGET"
swa_overlap_source_replace_layer_mode: "$SWA_OVERLAP_SOURCE_REPLACE_LAYER_MODE"
stage_c_mode: "$STAGE_C_MODE"
stage_c_cache_mode: "$STAGE_C_CACHE_MODE"
empty_cuda_cache_each_chunk: "$EMPTY_CUDA_CACHE_EACH_CHUNK"
semantic_role_policy: "$SEMANTIC_ROLE_POLICY"
semantic_memory_paths: "$SEMANTIC_MEMORY_PATHS"
EOF
cp "$OUT/effective_config.yaml" "$OUT/v47_effective_config.yaml"

cat > "$OUT/adaptive_writer_audit.json" <<EOF
{
  "row": "$ROW",
  "run_name": "$RUN_NAME",
  "adaptive_ttt_writer": true,
  "no_chunk_id_policy": true,
  "no_manual_tri_replay_percentages": true,
  "manual_tri_replay_positive_frac": "$TTT_WRITE_TRI_REPLAY_POSITIVE_FRAC",
  "manual_tri_replay_negative_frac": "$TTT_WRITE_TRI_REPLAY_NEGATIVE_FRAC",
  "manual_tri_replay_neutral_lambda": "$TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA",
  "external_gamma": "$TTT_WRITE_GRADIENT_REVERSAL_GAMMA",
  "role_mode": "$TTT_WRITE_TRI_REPLAY_ROLE_MODE",
  "definition": "$ADAPTIVE_WRITER_DEFINITION"
}
EOF
cat > "$OUT/adaptive_ttt_audit.json" <<EOF
{
  "row": "$ROW",
  "run_name": "$RUN_NAME",
  "adaptive_ttt_writer": true,
  "no_chunk_id_policy": true,
  "manual_positive_frac": 0.0,
  "manual_negative_frac": 0.0,
  "manual_neutral_lambda": 0.0,
  "no_manual_tri_replay_percentages": true,
  "external_gamma": 0.0,
  "role_mode": "$TTT_WRITE_TRI_REPLAY_ROLE_MODE",
  "commit_filter_mode": "$TTT_WRITE_COMMIT_FILTER_MODE",
  "risk_source": "$TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE",
  "scale_state_mode": "$TTT_WRITE_SCALE_STATE_MODE",
  "scale_state_chunks_empty": true,
  "scale_state_sample_tokens": "$TTT_WRITE_SCALE_STATE_SAMPLE_TOKENS",
  "native_mix_scales": "$TTT_WRITE_NATIVE_MIX_SCALES",
  "split_required": true,
  "definition": "$ADAPTIVE_WRITER_DEFINITION"
}
EOF

cat > "$OUT/chunk_id_policy_audit.json" <<EOF
{
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
  "scale_state_chunks": "$TTT_WRITE_SCALE_STATE_CHUNKS",
  "scale_state_sample_tokens": "$TTT_WRITE_SCALE_STATE_SAMPLE_TOKENS",
  "semantic_action_active_chunks": ""
}
EOF

cat > "$OUT/reproduce_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
V47_RESULT_ROOT="$RESULT_ROOT" \\
V47_ROLLOUT_BASE="$BASE" \\
V47_PLAN_NOTE="$PLAN_NOTE" \\
V47_END_FRAME="$END_FRAME" \\
V47_DG_CUE="$CUE" \\
V47_READ_BETA="$BETA" \\
V47_TTT_WRITE_SCORE="$WRITE_SCORE" \\
V47_TTT_RISK_SOURCE="$TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE" \\
V47_TTT_ROLE_MODE="$TTT_WRITE_TRI_REPLAY_ROLE_MODE" \\
V47_TTT_BRANCH_MASK="$TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK" \\
V47_TTT_BRANCH_GAMMAS="$TTT_WRITE_GRADIENT_REVERSAL_BRANCH_GAMMAS" \\
V47_TTT_LAYER_GAMMAS="$TTT_WRITE_GRADIENT_REVERSAL_LAYER_GAMMAS" \\
V47_TTT_HEAD_ROUTES="$TTT_WRITE_GRADIENT_REVERSAL_HEAD_ROUTES" \\
V47_NATIVE_MIX_SCALES="$TTT_WRITE_NATIVE_MIX_SCALES" \\
V47_TTT_COMMIT_FILTER_MODE="$TTT_WRITE_COMMIT_FILTER_MODE" \\
V47_TTT_COMMIT_FILTER_RISK_SOURCE="$TTT_WRITE_COMMIT_FILTER_RISK_SOURCE" \\
V47_TTT_COMMIT_FILTER_SCOPE="$TTT_WRITE_COMMIT_FILTER_SCOPE" \\
V47_TTT_COMMIT_FILTER_STAT="$TTT_WRITE_COMMIT_FILTER_STAT" \\
V47_TTT_COMMIT_FILTER_BASE="$TTT_WRITE_COMMIT_FILTER_BASE" \\
V47_TTT_COMMIT_FILTER_GAIN="$TTT_WRITE_COMMIT_FILTER_GAIN" \\
V47_TTT_COMMIT_FILTER_MIN="$TTT_WRITE_COMMIT_FILTER_MIN" \\
V47_TTT_COMMIT_FILTER_MAX="$TTT_WRITE_COMMIT_FILTER_MAX" \\
V47_TTT_COMMIT_FILTER_BRANCH_MASK="$TTT_WRITE_COMMIT_FILTER_BRANCH_MASK" \\
V47_TTT_SCALE_STATE_MODE="$TTT_WRITE_SCALE_STATE_MODE" \\
V47_TTT_SCALE_STATE_PROXY="$TTT_WRITE_SCALE_STATE_PROXY" \\
V47_TTT_SCALE_STATE_CARRIER="$TTT_WRITE_SCALE_STATE_CARRIER" \\
V47_TTT_SCALE_STATE_ALPHA="$TTT_WRITE_SCALE_STATE_ALPHA" \\
V47_TTT_SCALE_STATE_BRANCH_MASK="$TTT_WRITE_SCALE_STATE_BRANCH_MASK" \\
V47_TTT_SCALE_STATE_SAMPLE_TOKENS="$TTT_WRITE_SCALE_STATE_SAMPLE_TOKENS" \\
V47_ONLINE_SCALE_STATE_MODE="$ONLINE_SCALE_STATE_MODE" \\
V47_ONLINE_SCALE_STATE_MIN="$ONLINE_SCALE_STATE_MIN" \\
V47_ONLINE_SCALE_STATE_MAX="$ONLINE_SCALE_STATE_MAX" \\
V47_EMPTY_CUDA_CACHE_EACH_CHUNK="$EMPTY_CUDA_CACHE_EACH_CHUNK" \\
tools/run_v47_adaptive_ttt_writer_candidate.sh "$GPU" "$ROW" "$RUN_NAME"
EOF
chmod +x "$OUT/reproduce_command.sh"
cp "$OUT/reproduce_command.sh" "$OUT/v47_reproduce_command.sh"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $RUN_NAME row=$ROW gpu=$GPU" | tee "$OUT/run_status.txt"
tools/run_attention_cue_experiment.sh "$GPU" "$RUN_NAME" "$MODE" "$CUE" "$BETA" "$WRITE_SCORE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE $RUN_NAME row=$ROW gpu=$GPU" | tee -a "$OUT/run_status.txt"
