#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 GPU ROW RUN_NAME" >&2
  echo "Rows: H35_BASE H35_GEOM_TRI_SWA SEM_READ_L050 SEM_TRI_L050_NOSWA SEM_TRI_SWA_L050 SEM_TRI_SWA_L050_NATIVE110" >&2
  exit 2
fi

GPU="$1"
ROW="$2"
RUN_NAME="$3"

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
PY="${LOGER_PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"
BASE="${V76_H35_GLOBAL_ROLLOUT_BASE:-$ROOT/results/kitti01_hmc_v2/acl2_v76tf_c9_informed_semantic_tri_replay_memory_control/report_final/phase4_h35_global_l050_smoke/rollouts}"
OUT="$BASE/$RUN_NAME"

if [ -f "$OUT/run_status.txt" ] && grep -q "DONE $RUN_NAME" "$OUT/run_status.txt"; then
  echo "[v76-h35-global] Skip completed run: $RUN_NAME"
  exit 0
fi

mkdir -p "$OUT"

MODE="hybrid"
CUE="acl2.gg.qq.low.g2_3.past_only.headmean.robustq"
BETA="${V76_H35_GLOBAL_READ_BETA:-4.75}"
WRITE_SCORE="stage_d"
SEMANTIC_DESC="none"

export LOGER_ROOT="$ROOT"
export LOGER_PY="$PY"
export ATTN_CUE_BASE="$BASE"
export KITTI_SEQ="${KITTI_SEQ:-01}"
export START_FRAME="${START_FRAME:-0}"
export END_FRAME="${END_FRAME:-256}"
export RESET_EVERY="${RESET_EVERY:-5}"
export FAST_CUE_EVAL="${FAST_CUE_EVAL:-1}"
export READ_PATH="none"
export FRAME_BIAS_MODE="${V76_H35_GLOBAL_FRAME_BIAS_MODE:-pair}"
export READ_LAYER_MODE="${V76_H35_GLOBAL_READ_LAYER_MODE:-early}"
export READ_BETA_FRAME_CHUNKS=""
export READ_TOPK_FRAC="${V76_H35_GLOBAL_READ_TOPK_FRAC:-0.0}"
export READ_CALIB_MODE="${READ_CALIB_MODE:-none}"
export READ_BLEND_LAMBDA="${READ_BLEND_LAMBDA:-0.25}"
export BETA_SWA="$BETA"

export STAGE_C_MODE="none"
export STAGE_C_CACHE_DIR=""
export STAGE_C_CACHE_MODE="off"
export STAGE_C_CACHE_REQUIRE_HIT="0"
export STAGE_C_CACHE_VALIDATE="0"
export STAGE_C_INLINE_WHEN_IGNORED="0"
export SEMANTIC_ROLE_POLICY="none"
export SEMANTIC_MEMORY_PATHS=""
export SEMANTIC_ACTION_ACTIVE_CHUNKS=""
export SEMANTIC_ACTION_INACTIVE_READ_CUE_SOURCE=""

export HMC_COMMIT_MODE="probe_native"
export TTT_WRITE_GRADIENT_REVERSAL_MODE="none"
export TTT_WRITE_GRADIENT_REVERSAL_GAMMA="0.0"
export TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK="0"
export TTT_WRITE_GRADIENT_REVERSAL_CHUNKS=""
export TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS=""
export TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE="${V76_H35_GLOBAL_TTT_RISK_SOURCE:-update_conflict_energy}"
export TTT_WRITE_TRI_REPLAY_POSITIVE_FRAC="0.35"
export TTT_WRITE_TRI_REPLAY_NEGATIVE_FRAC="0.12"
export TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA="0.85"
export TTT_WRITE_TRI_REPLAY_ROLE_MODE="fixed"
export TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS=""
export TTT_WRITE_COMMIT_EMA_ALPHA="1.0"
export TTT_WRITE_COMMIT_EMA_BRANCH_MASK="all"
export TTT_WRITE_COMMIT_EMA_CHUNKS=""
export TTT_WRITE_NATIVE_MIX_SCALES="1.00,1.00,1.00"
export TTT_WRITE_NATIVE_MIX_CHUNKS=""

export ENABLE_SWA_WRITE_CONTROL="0"
export ENABLE_SWA_OVERLAP_SOURCE_REPLACE="0"
export SWA_OVERLAP_SOURCE_REPLACE_ALPHA="0.0"
export SWA_OVERLAP_SOURCE_REPLACE_MODE="${V76_H35_GLOBAL_SWA_REPLACE_MODE:-source}"
export SWA_OVERLAP_SOURCE_REPLACE_TARGET="${V76_H35_GLOBAL_SWA_REPLACE_TARGET:-kv}"
export SWA_OVERLAP_SOURCE_REPLACE_LAYER_MODE="${V76_H35_GLOBAL_SWA_REPLACE_LAYER_MODE:-last}"

enable_read() {
  export READ_PATH="frame"
  WRITE_SCORE="stage_d_x_dg_inv_sqrt"
}

enable_semantic_l050() {
  enable_read
  CUE="${V76_H35_GLOBAL_SEM_CUE:-v31.sem_resid_coarse_l050.c23past}"
  export STAGE_C_MODE="${V76_H35_GLOBAL_STAGE_C_MODE:-reference}"
  export STAGE_C_CACHE_DIR="${V76_H35_GLOBAL_STAGE_C_CACHE_DIR:-$ROOT/results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full}"
  export STAGE_C_CACHE_MODE="${V76_H35_GLOBAL_STAGE_C_CACHE_MODE:-read}"
  export STAGE_C_CACHE_REQUIRE_HIT="${V76_H35_GLOBAL_STAGE_C_CACHE_REQUIRE_HIT:-1}"
  SEMANTIC_DESC="semantic residual L050 read cue"
}

enable_global_tri() {
  export HMC_COMMIT_MODE="probe_ttt_write"
  export TTT_WRITE_GRADIENT_REVERSAL_MODE="tri_replay"
  export TTT_WRITE_GRADIENT_REVERSAL_GAMMA="${V76_H35_GLOBAL_TRI_GAMMA:-0.004}"
  export TTT_WRITE_TRI_REPLAY_ROLE_MODE="${V76_H35_GLOBAL_TRI_ROLE_MODE:-adaptive_quantile}"
  export TTT_WRITE_COMMIT_EMA_ALPHA="${V76_H35_GLOBAL_COMMIT_EMA_ALPHA:-0.5}"
  export TTT_WRITE_COMMIT_EMA_BRANCH_MASK="${V76_H35_GLOBAL_COMMIT_EMA_BRANCH_MASK:-0}"
}

enable_global_tri_swa() {
  enable_global_tri
  export ENABLE_SWA_OVERLAP_SOURCE_REPLACE="1"
  export SWA_OVERLAP_SOURCE_REPLACE_ALPHA="${V76_H35_GLOBAL_SWA_ALPHA:-0.5}"
}

case "$ROW" in
  H35_BASE)
    ;;
  H35_GEOM_TRI_SWA)
    enable_read
    enable_global_tri_swa
    export TTT_WRITE_COMMIT_EMA_ALPHA="1.0"
    export TTT_WRITE_COMMIT_EMA_BRANCH_MASK="all"
    SEMANTIC_DESC="geometry READ plus global adaptive tri/SWA"
    ;;
  SEM_READ_L050)
    enable_semantic_l050
    ;;
  SEM_TRI_L050_NOSWA)
    enable_semantic_l050
    enable_global_tri
    ;;
  SEM_TRI_SWA_L050)
    enable_semantic_l050
    enable_global_tri_swa
    ;;
  SEM_TRI_SWA_L050_NATIVE110)
    enable_semantic_l050
    enable_global_tri_swa
    export TTT_WRITE_NATIVE_MIX_SCALES="1.10,1.00,1.00"
    ;;
  *)
    echo "Unsupported row: $ROW" >&2
    exit 2
    ;;
esac

cat > "$OUT/effective_config.yaml" <<EOF
row: "$ROW"
run_name: "$RUN_NAME"
gpu: "$GPU"
start_frame: "$START_FRAME"
end_frame: "$END_FRAME"
mode: "$MODE"
cue: "$CUE"
beta_frame: "$BETA"
write_score: "$WRITE_SCORE"
semantic_desc: "$SEMANTIC_DESC"
read_path: "$READ_PATH"
read_layer_mode: "$READ_LAYER_MODE"
read_beta_frame_chunks: "$READ_BETA_FRAME_CHUNKS"
stage_c_mode: "$STAGE_C_MODE"
stage_c_cache_mode: "$STAGE_C_CACHE_MODE"
stage_c_cache_dir: "$STAGE_C_CACHE_DIR"
stage_c_cache_require_hit: "$STAGE_C_CACHE_REQUIRE_HIT"
hmc_commit_mode: "$HMC_COMMIT_MODE"
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
ttt_write_native_mix_chunks: "$TTT_WRITE_NATIVE_MIX_CHUNKS"
enable_swa_overlap_source_replace: "$ENABLE_SWA_OVERLAP_SOURCE_REPLACE"
swa_overlap_source_replace_alpha: "$SWA_OVERLAP_SOURCE_REPLACE_ALPHA"
semantic_role_policy: "$SEMANTIC_ROLE_POLICY"
semantic_memory_paths: "$SEMANTIC_MEMORY_PATHS"
semantic_action_active_chunks: "$SEMANTIC_ACTION_ACTIVE_CHUNKS"
EOF

cat > "$OUT/chunk_id_policy_audit.json" <<EOF
{
  "row": "$ROW",
  "run_name": "$RUN_NAME",
  "base": "H35",
  "has_read_beta_frame_chunks": false,
  "has_tri_gamma_chunk_map": false,
  "has_tri_replay_chunk_params": false,
  "has_commit_ema_chunks": false,
  "has_native_mix_chunks": false,
  "has_semantic_action_active_chunks": false,
  "read_beta_frame_chunks": "$READ_BETA_FRAME_CHUNKS",
  "tri_gamma_chunk_map": "$TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS",
  "tri_replay_chunk_params": "$TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS",
  "commit_ema_chunks": "$TTT_WRITE_COMMIT_EMA_CHUNKS",
  "native_mix_chunks": "$TTT_WRITE_NATIVE_MIX_CHUNKS",
  "semantic_action_active_chunks": "$SEMANTIC_ACTION_ACTIVE_CHUNKS"
}
EOF

cat > "$OUT/reproduce_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
V76_H35_GLOBAL_ROLLOUT_BASE="$BASE" \\
START_FRAME="$START_FRAME" \\
END_FRAME="$END_FRAME" \\
RESET_EVERY="$RESET_EVERY" \\
FAST_CUE_EVAL="$FAST_CUE_EVAL" \\
tools/run_v76_h35_global_l050_candidate.sh "$GPU" "$ROW" "$RUN_NAME"
EOF
chmod +x "$OUT/reproduce_command.sh"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $RUN_NAME row=$ROW gpu=$GPU base=H35" | tee "$OUT/run_status.txt"
"$ROOT/tools/run_attention_cue_experiment.sh" "$GPU" "$RUN_NAME" "$MODE" "$CUE" "$BETA" "$WRITE_SCORE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE $RUN_NAME row=$ROW gpu=$GPU base=H35" | tee -a "$OUT/run_status.txt"
