#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 GPU RUN_NAME MODE [CUE] [BETA] [WRITE_SCORE]" >&2
  echo "MODE: native | identity_hooks | readonly | hybrid" >&2
  exit 2
fi

GPU="$1"
RUN_NAME="$2"
MODE="$3"
CUE="${4:-dyn}"
BETA="${5:-1.0}"
WRITE_SCORE="${6:-stage_d}"

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
PY="${LOGER_PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"
SEQ="${KITTI_SEQ:-01}"
DATA="${KITTI_IMAGES:-/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/$SEQ/image_2}"
POSES="${KITTI_POSES:-/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses}"
BASE="${ATTN_CUE_BASE:-$ROOT/results/kitti01_hmc_v2/attention_cue_library_v1}"
case "$BASE" in
  /*) ;;
  *) BASE="$ROOT/$BASE" ;;
esac
OUT="$BASE/$RUN_NAME"
FRAME_BIAS="${FRAME_BIAS_MODE:-pair}"
READ_PATH="${READ_PATH:-frame}"
READ_LAYER_MODE="${READ_LAYER_MODE:-early}"
READ_TOPK_FRAC="${READ_TOPK_FRAC:-0.0}"
READ_CALIB_MODE="${READ_CALIB_MODE:-none}"
READ_TARGET_MASS="${READ_TARGET_MASS:-0.06}"
READ_CALIB_TAU="${READ_CALIB_TAU:-0.05}"
READ_BLEND_LAMBDA="${READ_BLEND_LAMBDA:-0.25}"
SWA_GATE_MIN="${SWA_GATE_MIN:-0.85}"
BETA_SWA="${BETA_SWA:-$BETA}"
END_FRAME="${END_FRAME:-10000}"
RESET_EVERY="${RESET_EVERY:-5}"
FAST_CUE_EVAL="${FAST_CUE_EVAL:-1}"
STAGE_C_MODE="${STAGE_C_MODE:-none}"
STAGE_C_CACHE_DIR="${STAGE_C_CACHE_DIR:-}"
STAGE_C_CACHE_MODE="${STAGE_C_CACHE_MODE:-off}"
STAGE_C_CACHE_REQUIRE_HIT="${STAGE_C_CACHE_REQUIRE_HIT:-0}"
STAGE_C_CACHE_VALIDATE="${STAGE_C_CACHE_VALIDATE:-0}"
STAGE_C_INLINE_WHEN_IGNORED="${STAGE_C_INLINE_WHEN_IGNORED:-0}"
STAGE_C_SAVE_VIDEO="${STAGE_C_SAVE_VIDEO:-0}"
STAGE_C_VIDEO_PATH="${STAGE_C_VIDEO_PATH:-}"
STAGE_C_VIDEO_FPS="${STAGE_C_VIDEO_FPS:-10}"
STAGE_C_VIDEO_ALPHA="${STAGE_C_VIDEO_ALPHA:-0.45}"
SEMANTIC_PRIOR_MODE="${SEMANTIC_PRIOR_MODE:-spg_v2}"
HMC_IGNORE_SEMANTIC_PRIOR="${HMC_IGNORE_SEMANTIC_PRIOR:-0}"
PRIOR_POLICY="${PRIOR_POLICY:-eta_mean_preserving}"
LAMBDA_MIN="${LAMBDA_MIN:-1.0}"
LAMBDA_MAX="${LAMBDA_MAX:-1.0}"
WRITE_ALPHA="${WRITE_ALPHA:-0.1}"
WRITE_MIN="${WRITE_MIN:-0.8}"
WRITE_MAX="${WRITE_MAX:-1.2}"
MP_SCORE_SOURCE="${MP_SCORE_SOURCE:-dyn}"
RHO_SEM="${RHO_SEM:-0.6}"
SPG_USE_G_WRITE_GEO="${SPG_USE_G_WRITE_GEO:-1}"
SPG_VALUE_STRUCTURE="${SPG_VALUE_STRUCTURE:-1.0}"
SPG_VALUE_BACKGROUND="${SPG_VALUE_BACKGROUND:-0.7}"
SPG_VALUE_DISTRACTOR="${SPG_VALUE_DISTRACTOR:-0.4}"
SPG_VALUE_MOVABLE="${SPG_VALUE_MOVABLE:-0.1}"
SPG_VALUE_UNCERTAIN="${SPG_VALUE_UNCERTAIN:-0.4}"
WRITE_SPARSE_RATIO="${WRITE_SPARSE_RATIO:-1.0}"
WRITE_SPARSE_MODE="${WRITE_SPARSE_MODE:-none}"
PRIOR_BRANCH_MASK="${PRIOR_BRANCH_MASK:-0}"
PRIOR_LAYER_MODE="${PRIOR_LAYER_MODE:-all}"
PRIOR_SINGLE_LAYER="${PRIOR_SINGLE_LAYER:-}"
PRIOR_LAYER_BRANCH_POLICY="${PRIOR_LAYER_BRANCH_POLICY:-}"
TTT_WRITE_DELTA_SCALE="${TTT_WRITE_DELTA_SCALE:-1.0}"
TTT_WRITE_DELTA_SCALES="${TTT_WRITE_DELTA_SCALES:-}"
TTT_WRITE_NATIVE_MIX_SCALES="${TTT_WRITE_NATIVE_MIX_SCALES:-}"
TTT_WRITE_PRIOR_TRANSFORM_MODE="${TTT_WRITE_PRIOR_TRANSFORM_MODE:-none}"
TTT_WRITE_PRIOR_ANTI_SCALE="${TTT_WRITE_PRIOR_ANTI_SCALE:-0.0}"
TTT_WRITE_PRIOR_GAMMA="${TTT_WRITE_PRIOR_GAMMA:-1.0}"
TTT_WRITE_SPECIAL_TOKEN_POLICY="${TTT_WRITE_SPECIAL_TOKEN_POLICY:-none}"
TTT_WRITE_SPECIAL_TOKEN_FLOOR="${TTT_WRITE_SPECIAL_TOKEN_FLOOR:-0.0}"
TTT_WRITE_SPECIAL_TOKEN_CEILING="${TTT_WRITE_SPECIAL_TOKEN_CEILING:-1.0}"
TTT_WRITE_GRADIENT_REVERSAL_MODE="${TTT_WRITE_GRADIENT_REVERSAL_MODE:-none}"
TTT_WRITE_GRADIENT_REVERSAL_GAMMA="${TTT_WRITE_GRADIENT_REVERSAL_GAMMA:-0.0}"
TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK="${TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK:-0}"
TTT_WRITE_GRADIENT_REVERSAL_BRANCH_GAMMAS="${TTT_WRITE_GRADIENT_REVERSAL_BRANCH_GAMMAS:-}"
TTT_WRITE_GRADIENT_REVERSAL_LAYER_GAMMAS="${TTT_WRITE_GRADIENT_REVERSAL_LAYER_GAMMAS:-}"
TTT_WRITE_GRADIENT_REVERSAL_HEAD_ROUTES="${TTT_WRITE_GRADIENT_REVERSAL_HEAD_ROUTES:-}"
TTT_WRITE_GRADIENT_REVERSAL_CHUNKS="${TTT_WRITE_GRADIENT_REVERSAL_CHUNKS:-}"
TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS="${TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS:-}"
TTT_WRITE_GRADIENT_REVERSAL_NEGATIVE_FRAC="${TTT_WRITE_GRADIENT_REVERSAL_NEGATIVE_FRAC:-0.0}"
TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE="${TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE:-prior}"
TTT_WRITE_TRI_REPLAY_POSITIVE_FRAC="${TTT_WRITE_TRI_REPLAY_POSITIVE_FRAC:-0.35}"
TTT_WRITE_TRI_REPLAY_NEGATIVE_FRAC="${TTT_WRITE_TRI_REPLAY_NEGATIVE_FRAC:-0.15}"
TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA="${TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA:-1.0}"
TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS="${TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS:-}"
TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_MODE="${TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_MODE:-none}"
TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_BRANCH_MASK="${TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_BRANCH_MASK:-}"
TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_LONG_SCALE="${TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_LONG_SCALE:-0.0}"
TTT_WRITE_REPLAY_FEATURE_GATE_MODE="${TTT_WRITE_REPLAY_FEATURE_GATE_MODE:-none}"
TTT_WRITE_REPLAY_FEATURE_GATE_RHO="${TTT_WRITE_REPLAY_FEATURE_GATE_RHO:-0.0}"
TTT_WRITE_REPLAY_FEATURE_GATE_MIN="${TTT_WRITE_REPLAY_FEATURE_GATE_MIN:-0.5}"
TTT_WRITE_REPLAY_FEATURE_GATE_BRANCH_MASK="${TTT_WRITE_REPLAY_FEATURE_GATE_BRANCH_MASK:-all}"
TTT_WRITE_REPLAY_TOKEN_FILTER_MODE="${TTT_WRITE_REPLAY_TOKEN_FILTER_MODE:-none}"
TTT_WRITE_REPLAY_TOKEN_FILTER_RATIO="${TTT_WRITE_REPLAY_TOKEN_FILTER_RATIO:-1.0}"
TTT_WRITE_REPLAY_TOKEN_FILTER_THRESHOLD="${TTT_WRITE_REPLAY_TOKEN_FILTER_THRESHOLD:-1.0}"
TTT_WRITE_REPLAY_TOKEN_FILTER_SCOPE="${TTT_WRITE_REPLAY_TOKEN_FILTER_SCOPE:-all}"
TTT_WRITE_REPLAY_TOKEN_FILTER_BRANCH_MASK="${TTT_WRITE_REPLAY_TOKEN_FILTER_BRANCH_MASK:-all}"
TTT_WRITE_REPLAY_TOKEN_FILTER_BLEND="${TTT_WRITE_REPLAY_TOKEN_FILTER_BLEND:-1.0}"
TTT_WRITE_REPLAY_TOKEN_FILTER_BLEND_MODE="${TTT_WRITE_REPLAY_TOKEN_FILTER_BLEND_MODE:-linear}"
TTT_WRITE_TRANSIENT_DELTA_SUBTRACT_SCALE="${TTT_WRITE_TRANSIENT_DELTA_SUBTRACT_SCALE:-0.0}"
TTT_WRITE_TRANSIENT_DELTA_BRANCH_MASK="${TTT_WRITE_TRANSIENT_DELTA_BRANCH_MASK:-0}"
TTT_WRITE_TRANSIENT_DELTA_TTL="${TTT_WRITE_TRANSIENT_DELTA_TTL:-1}"
TTT_WRITE_COMMIT_EMA_ALPHA="${TTT_WRITE_COMMIT_EMA_ALPHA:-1.0}"
TTT_WRITE_COMMIT_EMA_BRANCH_MASK="${TTT_WRITE_COMMIT_EMA_BRANCH_MASK:-all}"
TTT_WRITE_NATIVE_DELTA_GATE_MODE="${TTT_WRITE_NATIVE_DELTA_GATE_MODE:-none}"
TTT_WRITE_NATIVE_DELTA_GATE_MIN_COS="${TTT_WRITE_NATIVE_DELTA_GATE_MIN_COS:-0.0}"
TTT_WRITE_NATIVE_DELTA_GATE_FALLBACK="${TTT_WRITE_NATIVE_DELTA_GATE_FALLBACK:-0.0}"
TTT_WRITE_NATIVE_DELTA_GATE_CAP_RATIO="${TTT_WRITE_NATIVE_DELTA_GATE_CAP_RATIO:-1.0}"
TTT_WRITE_NATIVE_DELTA_GATE_BRANCH_MASK="${TTT_WRITE_NATIVE_DELTA_GATE_BRANCH_MASK:-all}"
TTT_WRITE_COMMIT_FILTER_MODE="${TTT_WRITE_COMMIT_FILTER_MODE:-none}"
TTT_WRITE_COMMIT_FILTER_RISK_SOURCE="${TTT_WRITE_COMMIT_FILTER_RISK_SOURCE:-d_tok}"
TTT_WRITE_COMMIT_FILTER_SCOPE="${TTT_WRITE_COMMIT_FILTER_SCOPE:-tail_overlap}"
TTT_WRITE_COMMIT_FILTER_STAT="${TTT_WRITE_COMMIT_FILTER_STAT:-mean}"
TTT_WRITE_COMMIT_FILTER_BASE="${TTT_WRITE_COMMIT_FILTER_BASE:-0.0}"
TTT_WRITE_COMMIT_FILTER_GAIN="${TTT_WRITE_COMMIT_FILTER_GAIN:-1.0}"
TTT_WRITE_COMMIT_FILTER_MIN="${TTT_WRITE_COMMIT_FILTER_MIN:-0.0}"
TTT_WRITE_COMMIT_FILTER_MAX="${TTT_WRITE_COMMIT_FILTER_MAX:-1.0}"
TTT_WRITE_COMMIT_FILTER_BRANCH_MASK="${TTT_WRITE_COMMIT_FILTER_BRANCH_MASK:-0}"
TTT_WRITE_COMMIT_FILTER_CHUNKS="${TTT_WRITE_COMMIT_FILTER_CHUNKS:-}"
ENABLE_SWA_WRITE_CONTROL="${ENABLE_SWA_WRITE_CONTROL:-0}"
SWA_WRITE_MODE="${SWA_WRITE_MODE:-none}"
SWA_WRITE_RHO="${SWA_WRITE_RHO:-0.0}"
SWA_WRITE_MIN_GATE="${SWA_WRITE_MIN_GATE:-0.0}"
SWA_WRITE_SPARSE_RATIO="${SWA_WRITE_SPARSE_RATIO:-1.0}"
SWA_WRITE_LAYER_MODE="${SWA_WRITE_LAYER_MODE:-all}"
SWA_WRITE_SINGLE_LAYER="${SWA_WRITE_SINGLE_LAYER:-}"
SWA_WRITE_SCOPE="${SWA_WRITE_SCOPE:-all}"
SWA_WRITE_KEEP_SCOPE="${SWA_WRITE_KEEP_SCOPE:-all}"
SWA_WRITE_SCORE_SOURCE="${SWA_WRITE_SCORE_SOURCE:-read}"
SWA_WRITE_CACHE_BLEND_ALPHA="${SWA_WRITE_CACHE_BLEND_ALPHA:-0.0}"
SWA_WRITE_CACHE_BLEND_MODE="${SWA_WRITE_CACHE_BLEND_MODE:-dynamic}"
SWA_WRITE_CACHE_BLEND_TARGET="${SWA_WRITE_CACHE_BLEND_TARGET:-v}"
ENABLE_SWA_OVERLAP_BIAS="${ENABLE_SWA_OVERLAP_BIAS:-0}"
SWA_OVERLAP_BIAS_BETA="${SWA_OVERLAP_BIAS_BETA:-0.0}"
SWA_OVERLAP_BIAS_MIN_KEEP="${SWA_OVERLAP_BIAS_MIN_KEEP:-0.0001}"
SWA_OVERLAP_BIAS_MODE="${SWA_OVERLAP_BIAS_MODE:-pair}"
SWA_OVERLAP_BIAS_LAYER_MODE="${SWA_OVERLAP_BIAS_LAYER_MODE:-last}"
SWA_OVERLAP_BIAS_SINGLE_LAYER="${SWA_OVERLAP_BIAS_SINGLE_LAYER:-}"
ENABLE_SWA_OVERLAP_SOURCE_GATE="${ENABLE_SWA_OVERLAP_SOURCE_GATE:-0}"
SWA_OVERLAP_SOURCE_GATE_RHO="${SWA_OVERLAP_SOURCE_GATE_RHO:-0.0}"
SWA_OVERLAP_SOURCE_GATE_MIN="${SWA_OVERLAP_SOURCE_GATE_MIN:-0.85}"
SWA_OVERLAP_SOURCE_GATE_MODE="${SWA_OVERLAP_SOURCE_GATE_MODE:-source}"
SWA_OVERLAP_SOURCE_GATE_TARGET="${SWA_OVERLAP_SOURCE_GATE_TARGET:-v}"
SWA_OVERLAP_SOURCE_GATE_LAYER_MODE="${SWA_OVERLAP_SOURCE_GATE_LAYER_MODE:-last}"
SWA_OVERLAP_SOURCE_GATE_SINGLE_LAYER="${SWA_OVERLAP_SOURCE_GATE_SINGLE_LAYER:-}"
ENABLE_SWA_OVERLAP_SOURCE_REPLACE="${ENABLE_SWA_OVERLAP_SOURCE_REPLACE:-0}"
SWA_OVERLAP_SOURCE_REPLACE_ALPHA="${SWA_OVERLAP_SOURCE_REPLACE_ALPHA:-0.0}"
SWA_OVERLAP_SOURCE_REPLACE_MODE="${SWA_OVERLAP_SOURCE_REPLACE_MODE:-union}"
SWA_OVERLAP_SOURCE_REPLACE_TARGET="${SWA_OVERLAP_SOURCE_REPLACE_TARGET:-kv}"
SWA_OVERLAP_SOURCE_REPLACE_LAYER_MODE="${SWA_OVERLAP_SOURCE_REPLACE_LAYER_MODE:-last}"
SWA_OVERLAP_SOURCE_REPLACE_SINGLE_LAYER="${SWA_OVERLAP_SOURCE_REPLACE_SINGLE_LAYER:-}"
TTT_WRITE_TOKEN_SCOPE="${TTT_WRITE_TOKEN_SCOPE:-all}"
TTT_WRITE_TOKEN_SCOPE_FLOOR="${TTT_WRITE_TOKEN_SCOPE_FLOOR:-0.0}"
TTT_FREEZE_CHUNKS="${TTT_FREEZE_CHUNKS:-}"
TTT_SEMANTIC_WRITE_SCALE_CHUNKS="${TTT_SEMANTIC_WRITE_SCALE_CHUNKS:-}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"

mkdir -p "$OUT"
echo "[$(date '+%F %T')] START $RUN_NAME gpu=$GPU seq=$SEQ mode=$MODE cue=$CUE beta=$BETA write=$WRITE_SCORE reset_every=$RESET_EVERY" | tee "$OUT/run_status.txt"

COMMON_ARGS=(
  --input "$DATA"
  --checkpoint "$ROOT/ckpts/LoGeR/latest.pt"
  --config "$ROOT/ckpts/LoGeR/original_config.yaml"
  --geometry_edge_rtol 0.0
  --chunk_size 32 --chunk_overlap 3
  --window_size 32 --overlap_size 3
  --reset_every "$RESET_EVERY"
  --ttt_freeze_chunks "$TTT_FREEZE_CHUNKS"
  --ttt_semantic_write_scale_chunks "$TTT_SEMANTIC_WRITE_SCALE_CHUNKS"
  --end_frame "$END_FRAME"
  --output_txt "$OUT/$SEQ.txt"
  --hybrid_debug_jsonl "$OUT/hmc_state_hash.jsonl"
)

STAGE_B_ARGS=(
  --stage_c_mode "$STAGE_C_MODE"
  --stage_c_cache_mode "$STAGE_C_CACHE_MODE"
  --stage_c_cache_require_hit "$STAGE_C_CACHE_REQUIRE_HIT"
  --stage_c_cache_validate "$STAGE_C_CACHE_VALIDATE"
  --stage_c_inline_when_ignored "$STAGE_C_INLINE_WHEN_IGNORED"
  --stage_c_save_video "$STAGE_C_SAVE_VIDEO"
  --stage_c_video_fps "$STAGE_C_VIDEO_FPS"
  --stage_c_video_alpha "$STAGE_C_VIDEO_ALPHA"
  --semantic_prior_mode "$SEMANTIC_PRIOR_MODE"
  --hmc_ignore_semantic_prior "$HMC_IGNORE_SEMANTIC_PRIOR"
  --dyn_fusion_mode calibrated_soft_or
  --implicit_weight 0.50
  --implicit_gate_floor 0.25
  --k_intra 5
  --sigma_pt 0.25
  --lambda_s 1.2 --lambda_a 0.8 --lambda_d 1.2 --lambda_o 0.3 --lambda_u 0.3
  --lambda_min "$LAMBDA_MIN" --lambda_max "$LAMBDA_MAX"
  --a_min_special 1.0 --a_token_floor 0.0
  --prior_policy "$PRIOR_POLICY"
  --mp_alpha "$WRITE_ALPHA" --mp_min "$WRITE_MIN" --mp_max "$WRITE_MAX"
  --mp_score_source "$MP_SCORE_SOURCE"
  --rho_sem "$RHO_SEM"
  --spg_use_g_write_geo "$SPG_USE_G_WRITE_GEO"
  --spg_value_structure "$SPG_VALUE_STRUCTURE"
  --spg_value_background "$SPG_VALUE_BACKGROUND"
  --spg_value_distractor "$SPG_VALUE_DISTRACTOR"
  --spg_value_movable "$SPG_VALUE_MOVABLE"
  --spg_value_uncertain "$SPG_VALUE_UNCERTAIN"
  --prior_branch_mask "$PRIOR_BRANCH_MASK"
  --prior_layer_mode "$PRIOR_LAYER_MODE"
  --prior_layer_branch_policy "$PRIOR_LAYER_BRANCH_POLICY"
  --read_path "$READ_PATH"
  --read_cue_source "$CUE"
  --frame_bias_mode "$FRAME_BIAS"
  --beta_swa "$BETA_SWA"
  --swa_gate_min "$SWA_GATE_MIN"
  --read_layer_mode "$READ_LAYER_MODE"
  --read_topk_frac "$READ_TOPK_FRAC"
  --read_calib_mode "$READ_CALIB_MODE"
  --read_target_mass "$READ_TARGET_MASS"
  --read_calib_tau "$READ_CALIB_TAU"
  --read_blend_lambda "$READ_BLEND_LAMBDA"
  --beta_frame "$BETA"
  --prior_debug_jsonl "$OUT/prior_debug.jsonl"
)

if [ -n "$STAGE_C_CACHE_DIR" ]; then
  STAGE_B_ARGS+=(--stage_c_cache_dir "$STAGE_C_CACHE_DIR")
fi
if [ -n "$STAGE_C_VIDEO_PATH" ]; then
  STAGE_B_ARGS+=(--stage_c_video_path "$STAGE_C_VIDEO_PATH")
fi

if [ -n "$PRIOR_SINGLE_LAYER" ]; then
  STAGE_B_ARGS+=(--prior_single_layer "$PRIOR_SINGLE_LAYER")
fi

if [ "$FAST_CUE_EVAL" = "1" ]; then
  STAGE_B_ARGS+=(--fast_cue_eval 1)
fi

case "$MODE" in
  native|identity_hooks)
    RUN_ARGS=(
      "${COMMON_ARGS[@]}"
      --geometry_eval_mode
      --hybrid_memory_mode "$MODE"
    )
    ;;
  readonly)
    RUN_ARGS=(
      "${COMMON_ARGS[@]}"
      "${STAGE_B_ARGS[@]}"
      --hybrid_memory_mode read_path_only
      --hmc_commit_mode probe_native
    )
    ;;
  hybrid)
    RUN_ARGS=(
      "${COMMON_ARGS[@]}"
      "${STAGE_B_ARGS[@]}"
      --hybrid_memory_mode hybrid
      --hmc_commit_mode probe_ttt_write
      --hmc_write_score_source "$WRITE_SCORE"
      --hmc_write_sparse_ratio "$WRITE_SPARSE_RATIO"
      --hmc_write_sparse_mode "$WRITE_SPARSE_MODE"
      --ttt_write_delta_scale "$TTT_WRITE_DELTA_SCALE"
      --ttt_write_delta_scales "$TTT_WRITE_DELTA_SCALES"
      --ttt_write_native_mix_scales "$TTT_WRITE_NATIVE_MIX_SCALES"
      --ttt_write_prior_transform_mode "$TTT_WRITE_PRIOR_TRANSFORM_MODE"
      --ttt_write_prior_anti_scale "$TTT_WRITE_PRIOR_ANTI_SCALE"
      --ttt_write_prior_gamma "$TTT_WRITE_PRIOR_GAMMA"
      --ttt_write_special_token_policy "$TTT_WRITE_SPECIAL_TOKEN_POLICY"
      --ttt_write_special_token_floor "$TTT_WRITE_SPECIAL_TOKEN_FLOOR"
      --ttt_write_special_token_ceiling "$TTT_WRITE_SPECIAL_TOKEN_CEILING"
      --ttt_write_gradient_reversal_mode "$TTT_WRITE_GRADIENT_REVERSAL_MODE"
      --ttt_write_gradient_reversal_gamma "$TTT_WRITE_GRADIENT_REVERSAL_GAMMA"
      --ttt_write_gradient_reversal_branch_mask "$TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK"
      --ttt_write_gradient_reversal_branch_gammas "$TTT_WRITE_GRADIENT_REVERSAL_BRANCH_GAMMAS"
      --ttt_write_gradient_reversal_layer_gammas "$TTT_WRITE_GRADIENT_REVERSAL_LAYER_GAMMAS"
      --ttt_write_gradient_reversal_head_routes "$TTT_WRITE_GRADIENT_REVERSAL_HEAD_ROUTES"
      --ttt_write_gradient_reversal_chunks "$TTT_WRITE_GRADIENT_REVERSAL_CHUNKS"
      --ttt_write_gradient_reversal_chunk_gammas "$TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS"
      --ttt_write_gradient_reversal_negative_frac "$TTT_WRITE_GRADIENT_REVERSAL_NEGATIVE_FRAC"
      --ttt_write_gradient_reversal_risk_source "$TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE"
      --ttt_write_tri_replay_positive_frac "$TTT_WRITE_TRI_REPLAY_POSITIVE_FRAC"
      --ttt_write_tri_replay_negative_frac "$TTT_WRITE_TRI_REPLAY_NEGATIVE_FRAC"
      --ttt_write_tri_replay_neutral_lambda "$TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA"
      --ttt_write_tri_replay_chunk_params "$TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS"
      --ttt_write_gradient_reversal_transient_mode "$TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_MODE"
      --ttt_write_gradient_reversal_transient_branch_mask "$TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_BRANCH_MASK"
      --ttt_write_gradient_reversal_transient_long_scale "$TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_LONG_SCALE"
      --ttt_write_replay_feature_gate_mode "$TTT_WRITE_REPLAY_FEATURE_GATE_MODE"
      --ttt_write_replay_feature_gate_rho "$TTT_WRITE_REPLAY_FEATURE_GATE_RHO"
      --ttt_write_replay_feature_gate_min "$TTT_WRITE_REPLAY_FEATURE_GATE_MIN"
      --ttt_write_replay_feature_gate_branch_mask "$TTT_WRITE_REPLAY_FEATURE_GATE_BRANCH_MASK"
      --ttt_write_replay_token_filter_mode "$TTT_WRITE_REPLAY_TOKEN_FILTER_MODE"
      --ttt_write_replay_token_filter_ratio "$TTT_WRITE_REPLAY_TOKEN_FILTER_RATIO"
      --ttt_write_replay_token_filter_threshold "$TTT_WRITE_REPLAY_TOKEN_FILTER_THRESHOLD"
      --ttt_write_replay_token_filter_scope "$TTT_WRITE_REPLAY_TOKEN_FILTER_SCOPE"
      --ttt_write_replay_token_filter_branch_mask "$TTT_WRITE_REPLAY_TOKEN_FILTER_BRANCH_MASK"
      --ttt_write_replay_token_filter_blend "$TTT_WRITE_REPLAY_TOKEN_FILTER_BLEND"
      --ttt_write_replay_token_filter_blend_mode "$TTT_WRITE_REPLAY_TOKEN_FILTER_BLEND_MODE"
      --ttt_write_transient_delta_subtract_scale "$TTT_WRITE_TRANSIENT_DELTA_SUBTRACT_SCALE"
      --ttt_write_transient_delta_branch_mask "$TTT_WRITE_TRANSIENT_DELTA_BRANCH_MASK"
      --ttt_write_transient_delta_ttl "$TTT_WRITE_TRANSIENT_DELTA_TTL"
      --ttt_write_commit_ema_alpha "$TTT_WRITE_COMMIT_EMA_ALPHA"
      --ttt_write_commit_ema_branch_mask "$TTT_WRITE_COMMIT_EMA_BRANCH_MASK"
      --ttt_write_native_delta_gate_mode "$TTT_WRITE_NATIVE_DELTA_GATE_MODE"
      --ttt_write_native_delta_gate_min_cos "$TTT_WRITE_NATIVE_DELTA_GATE_MIN_COS"
      --ttt_write_native_delta_gate_fallback "$TTT_WRITE_NATIVE_DELTA_GATE_FALLBACK"
      --ttt_write_native_delta_gate_cap_ratio "$TTT_WRITE_NATIVE_DELTA_GATE_CAP_RATIO"
      --ttt_write_native_delta_gate_branch_mask "$TTT_WRITE_NATIVE_DELTA_GATE_BRANCH_MASK"
      --ttt_write_commit_filter_mode "$TTT_WRITE_COMMIT_FILTER_MODE"
      --ttt_write_commit_filter_risk_source "$TTT_WRITE_COMMIT_FILTER_RISK_SOURCE"
      --ttt_write_commit_filter_scope "$TTT_WRITE_COMMIT_FILTER_SCOPE"
      --ttt_write_commit_filter_stat "$TTT_WRITE_COMMIT_FILTER_STAT"
      --ttt_write_commit_filter_base "$TTT_WRITE_COMMIT_FILTER_BASE"
      --ttt_write_commit_filter_gain "$TTT_WRITE_COMMIT_FILTER_GAIN"
      --ttt_write_commit_filter_min "$TTT_WRITE_COMMIT_FILTER_MIN"
      --ttt_write_commit_filter_max "$TTT_WRITE_COMMIT_FILTER_MAX"
      --ttt_write_commit_filter_branch_mask "$TTT_WRITE_COMMIT_FILTER_BRANCH_MASK"
      --ttt_write_commit_filter_chunks "$TTT_WRITE_COMMIT_FILTER_CHUNKS"
      --enable_swa_write_control "$ENABLE_SWA_WRITE_CONTROL"
      --swa_write_mode "$SWA_WRITE_MODE"
      --swa_write_rho "$SWA_WRITE_RHO"
      --swa_write_min_gate "$SWA_WRITE_MIN_GATE"
      --swa_write_sparse_ratio "$SWA_WRITE_SPARSE_RATIO"
      --swa_write_layer_mode "$SWA_WRITE_LAYER_MODE"
      --swa_write_scope "$SWA_WRITE_SCOPE"
      --swa_write_keep_scope "$SWA_WRITE_KEEP_SCOPE"
      --swa_write_score_source "$SWA_WRITE_SCORE_SOURCE"
      --swa_write_cache_blend_alpha "$SWA_WRITE_CACHE_BLEND_ALPHA"
      --swa_write_cache_blend_mode "$SWA_WRITE_CACHE_BLEND_MODE"
      --swa_write_cache_blend_target "$SWA_WRITE_CACHE_BLEND_TARGET"
      --enable_swa_overlap_bias "$ENABLE_SWA_OVERLAP_BIAS"
      --swa_overlap_bias_beta "$SWA_OVERLAP_BIAS_BETA"
      --swa_overlap_bias_min_keep "$SWA_OVERLAP_BIAS_MIN_KEEP"
      --swa_overlap_bias_mode "$SWA_OVERLAP_BIAS_MODE"
      --swa_overlap_bias_layer_mode "$SWA_OVERLAP_BIAS_LAYER_MODE"
      --enable_swa_overlap_source_gate "$ENABLE_SWA_OVERLAP_SOURCE_GATE"
      --swa_overlap_source_gate_rho "$SWA_OVERLAP_SOURCE_GATE_RHO"
      --swa_overlap_source_gate_min "$SWA_OVERLAP_SOURCE_GATE_MIN"
      --swa_overlap_source_gate_mode "$SWA_OVERLAP_SOURCE_GATE_MODE"
      --swa_overlap_source_gate_target "$SWA_OVERLAP_SOURCE_GATE_TARGET"
      --swa_overlap_source_gate_layer_mode "$SWA_OVERLAP_SOURCE_GATE_LAYER_MODE"
      --enable_swa_overlap_source_replace "$ENABLE_SWA_OVERLAP_SOURCE_REPLACE"
      --swa_overlap_source_replace_alpha "$SWA_OVERLAP_SOURCE_REPLACE_ALPHA"
      --swa_overlap_source_replace_mode "$SWA_OVERLAP_SOURCE_REPLACE_MODE"
      --swa_overlap_source_replace_target "$SWA_OVERLAP_SOURCE_REPLACE_TARGET"
      --swa_overlap_source_replace_layer_mode "$SWA_OVERLAP_SOURCE_REPLACE_LAYER_MODE"
      --ttt_write_token_scope "$TTT_WRITE_TOKEN_SCOPE"
      --ttt_write_token_scope_floor "$TTT_WRITE_TOKEN_SCOPE_FLOOR"
  )

  if [ -n "$SWA_WRITE_SINGLE_LAYER" ]; then
    RUN_ARGS+=(--swa_write_single_layer "$SWA_WRITE_SINGLE_LAYER")
  fi
  if [ -n "$SWA_OVERLAP_BIAS_SINGLE_LAYER" ]; then
    RUN_ARGS+=(--swa_overlap_bias_single_layer "$SWA_OVERLAP_BIAS_SINGLE_LAYER")
  fi
  if [ -n "$SWA_OVERLAP_SOURCE_GATE_SINGLE_LAYER" ]; then
    RUN_ARGS+=(--swa_overlap_source_gate_single_layer "$SWA_OVERLAP_SOURCE_GATE_SINGLE_LAYER")
  fi
  if [ -n "$SWA_OVERLAP_SOURCE_REPLACE_SINGLE_LAYER" ]; then
    RUN_ARGS+=(--swa_overlap_source_replace_single_layer "$SWA_OVERLAP_SOURCE_REPLACE_SINGLE_LAYER")
  fi
    ;;
  *)
    echo "Unsupported MODE: $MODE" >&2
    exit 2
    ;;
esac

if PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$ROOT/run_pipeline_abc_v2.py" "${RUN_ARGS[@]}" > "$OUT/01.log" 2>&1; then
  (cd "$ROOT/eval/long_eval_script" && ./kitti_benchmark "$POSES" "$OUT" --plot) > "$OUT/kitti_benchmark.log" 2>&1
  echo "[$(date '+%F %T')] DONE $RUN_NAME" | tee -a "$OUT/run_status.txt"
else
  rc=$?
  echo "[$(date '+%F %T')] FAIL $RUN_NAME rc=$rc" | tee -a "$OUT/run_status.txt"
  exit "$rc"
fi
