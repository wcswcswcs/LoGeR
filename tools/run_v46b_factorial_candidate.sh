#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 GPU ROW RUN_NAME" >&2
  echo "Rows: F000_NONE F100_ONLY_FRAME_ATTN F010_ONLY_TTT F001_ONLY_SWA F110_FRAME_ATTN_TTT F101_FRAME_ATTN_SWA F011_TTT_SWA F111_ALL_THREE" >&2
  exit 2
fi

GPU="$1"
ROW="$2"
RUN_NAME="$3"

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
PY="${LOGER_PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"
RESULT_ROOT="${V46B_RESULT_ROOT:-$ROOT/results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa}"
BASE="${V46B_ROLLOUT_BASE:-$RESULT_ROOT/phase2_factorial/rollouts}"
OUT="$BASE/$RUN_NAME"
PLAN_DOC="$ROOT/docs/ACL2_v46B_ComponentAttribution_FrameTTT_FrameSWA_Addendum.md"

mkdir -p "$OUT"
if [ -f "$OUT/run_status.txt" ] && grep -q "DONE $RUN_NAME" "$OUT/run_status.txt"; then
  echo "[v46B] Skip completed run: $RUN_NAME"
  exit 0
fi

FRAME_ON=0
TTT_ON=0
SWA_ON=0
case "$ROW" in
  F000_NONE) ;;
  F100_ONLY_FRAME_ATTN) FRAME_ON=1 ;;
  F010_ONLY_TTT) TTT_ON=1 ;;
  F001_ONLY_SWA) SWA_ON=1 ;;
  F110_FRAME_ATTN_TTT) FRAME_ON=1; TTT_ON=1 ;;
  F101_FRAME_ATTN_SWA) FRAME_ON=1; SWA_ON=1 ;;
  F011_TTT_SWA) TTT_ON=1; SWA_ON=1 ;;
  F111_ALL_THREE) FRAME_ON=1; TTT_ON=1; SWA_ON=1 ;;
  *)
    echo "Unsupported v46B row: $ROW" >&2
    exit 2
    ;;
esac

# Shared clean no-chunk policy.  These rows intentionally avoid every
# absolute KITTI01/C9 chunk-id map so the attribution can be audited.
export ATTN_CUE_BASE="$BASE"
export LOGER_ROOT="$ROOT"
export LOGER_PY="$PY"
export KITTI_SEQ="${KITTI_SEQ:-01}"
export STAGE_C_MODE="none"
export STAGE_C_CACHE_MODE="off"
export STAGE_C_CACHE_REQUIRE_HIT="0"
export SEMANTIC_ROLE_POLICY="none"
export SEMANTIC_MEMORY_PATHS=""
export READ_BETA_FRAME_CHUNKS=""
export TTT_WRITE_GRADIENT_REVERSAL_CHUNKS=""
export TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS=""
export TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS=""
export TTT_WRITE_COMMIT_EMA_CHUNKS=""
export TTT_WRITE_NATIVE_MIX_CHUNKS=""
export READ_TOPK_FRAC="${V46B_READ_TOPK_FRAC:-0.0}"
export RESET_EVERY="${V46B_RESET_EVERY:-5}"
export END_FRAME="${V46B_END_FRAME:-10000}"

# Baseline defaults: no read control, no TTT write, no SWA replace.
MODE="hybrid"
CUE="${V46B_DG_CUE:-acl2.gg.qq.low.g2_3.past_only.headmean.robustq}"
BETA="${V46B_READ_BETA:-4.75}"
WRITE_SCORE="${V46B_WRITE_SCORE:-stage_d}"
export READ_PATH="none"
export HMC_COMMIT_MODE="probe_native"
export TTT_WRITE_GRADIENT_REVERSAL_MODE="none"
export TTT_WRITE_GRADIENT_REVERSAL_GAMMA="0.0"
export TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE="${V46B_TTT_RISK_SOURCE:-update_conflict_energy}"
export TTT_WRITE_GRADIENT_REVERSAL_NEGATIVE_FRAC="0.0"
export TTT_WRITE_TRI_REPLAY_POSITIVE_FRAC="${V46B_TTT_POSITIVE_FRAC:-0.35}"
export TTT_WRITE_TRI_REPLAY_NEGATIVE_FRAC="${V46B_TTT_NEGATIVE_FRAC:-0.12}"
export TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA="${V46B_TTT_NEUTRAL_LAMBDA:-0.85}"
export TTT_WRITE_TRI_REPLAY_ROLE_MODE="${V46B_TTT_ROLE_MODE:-fixed}"
export TTT_WRITE_COMMIT_EMA_ALPHA="1.0"
export TTT_WRITE_COMMIT_EMA_BRANCH_MASK="all"
export TTT_WRITE_NATIVE_MIX_SCALES="${V46B_NATIVE_MIX_SCALES:-1.00,1.00,1.00}"
export ENABLE_SWA_OVERLAP_SOURCE_REPLACE="0"
export SWA_OVERLAP_SOURCE_REPLACE_ALPHA="0.0"
export SWA_OVERLAP_SOURCE_REPLACE_MODE="${V46B_SWA_REPLACE_MODE:-source}"
export SWA_OVERLAP_SOURCE_REPLACE_TARGET="${V46B_SWA_REPLACE_TARGET:-kv}"
export SWA_OVERLAP_SOURCE_REPLACE_LAYER_MODE="${V46B_SWA_REPLACE_LAYER_MODE:-last}"

if [ "$FRAME_ON" -eq 1 ]; then
  export READ_PATH="frame"
  export FRAME_BIAS_MODE="${V46B_FRAME_BIAS_MODE:-pair}"
  export READ_LAYER_MODE="${V46B_READ_LAYER_MODE:-early}"
fi

if [ "$TTT_ON" -eq 1 ]; then
  export HMC_COMMIT_MODE="probe_ttt_write"
  export TTT_WRITE_GRADIENT_REVERSAL_MODE="tri_replay"
  export TTT_WRITE_GRADIENT_REVERSAL_GAMMA="${V46B_TTT_GAMMA:-0.004}"
  WRITE_SCORE="${V46B_TTT_WRITE_SCORE:-stage_d_x_dg_inv_sqrt}"
fi

if [ "$SWA_ON" -eq 1 ]; then
  export ENABLE_SWA_OVERLAP_SOURCE_REPLACE="1"
  export SWA_OVERLAP_SOURCE_REPLACE_ALPHA="${V46B_SWA_ALPHA:-0.5}"
fi

cat > "$OUT/v46b_effective_config.yaml" <<EOF
row: "$ROW"
run_name: "$RUN_NAME"
plan_doc: "$PLAN_DOC"
frame_attn_expected: $FRAME_ON
ttt_expected: $TTT_ON
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
swa_overlap_source_replace_mode: "$SWA_OVERLAP_SOURCE_REPLACE_MODE"
swa_overlap_source_replace_target: "$SWA_OVERLAP_SOURCE_REPLACE_TARGET"
swa_overlap_source_replace_layer_mode: "$SWA_OVERLAP_SOURCE_REPLACE_LAYER_MODE"
stage_c_mode: "$STAGE_C_MODE"
stage_c_cache_mode: "$STAGE_C_CACHE_MODE"
semantic_role_policy: "$SEMANTIC_ROLE_POLICY"
semantic_memory_paths: "$SEMANTIC_MEMORY_PATHS"
EOF

cat > "$OUT/component_factor_audit.json" <<EOF
{
  "row": "$ROW",
  "run_name": "$RUN_NAME",
  "frame_attn_expected": $([ "$FRAME_ON" -eq 1 ] && echo true || echo false),
  "ttt_expected": $([ "$TTT_ON" -eq 1 ] && echo true || echo false),
  "swa_expected": $([ "$SWA_ON" -eq 1 ] && echo true || echo false),
  "frame_attn_definition": "READ_PATH=frame with fixed global read beta; no chunk read map",
  "ttt_definition": "HMC_COMMIT_MODE=probe_ttt_write with tri_replay and global gamma/role fractions; no chunk gamma map",
  "swa_definition": "ENABLE_SWA_OVERLAP_SOURCE_REPLACE=1 with global alpha; no TTT/read side effects by itself"
}
EOF

cat > "$OUT/chunk_id_policy_audit.json" <<EOF
{
  "has_read_beta_frame_chunks": false,
  "has_tri_gamma_chunk_map": false,
  "has_tri_replay_chunk_params": false,
  "has_commit_ema_chunks": false,
  "read_beta_frame_chunks": "$READ_BETA_FRAME_CHUNKS",
  "tri_gamma_chunk_map": "$TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS",
  "tri_replay_chunk_params": "$TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS",
  "commit_ema_chunks": "$TTT_WRITE_COMMIT_EMA_CHUNKS"
}
EOF

cat > "$OUT/v46b_reproduce_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
V46B_RESULT_ROOT="$RESULT_ROOT" tools/run_v46b_factorial_candidate.sh "$GPU" "$ROW" "$RUN_NAME"
EOF
chmod +x "$OUT/v46b_reproduce_command.sh"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $RUN_NAME row=$ROW gpu=$GPU" | tee "$OUT/run_status.txt"
tools/run_attention_cue_experiment.sh "$GPU" "$RUN_NAME" "$MODE" "$CUE" "$BETA" "$WRITE_SCORE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE $RUN_NAME row=$ROW gpu=$GPU" | tee -a "$OUT/run_status.txt"
