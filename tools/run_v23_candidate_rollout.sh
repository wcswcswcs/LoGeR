#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 GPU CANDIDATE_ID CHUNK_ID HORIZON" >&2
  exit 2
fi

GPU="$1"
CANDIDATE_ID="$2"
CHUNK_ID="$3"
HORIZON="$4"

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
V16_ROOT="${V16_ROOT:-results/kitti01_hmc_v2/acl2_v16_ttt_causalfork_candidatebank_target25}"
V23_ROOT="${V23_ROOT:-results/kitti01_hmc_v2/acl2_v23_semanticprior_allmemory_durable_target25}"
PHASE1="$V16_ROOT/phase1_causalfork"
ROLLOUT_BASE="$V23_ROOT/rollouts"
STAGE_C_CACHE_DEFAULT="results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full"

case "$HORIZON" in
  3|5|8|10|15) ;;
  *) echo "Unsupported HORIZON: $HORIZON" >&2; exit 2 ;;
esac

case "$CHUNK_ID" in
  5) START_FRAME=145; SNAP="005" ;;
  6) START_FRAME=174; SNAP="006" ;;
  9) START_FRAME=261; SNAP="009" ;;
  10) START_FRAME=290; SNAP="010" ;;
  12) START_FRAME=348; SNAP="012" ;;
  16) START_FRAME=464; SNAP="016" ;;
  *) echo "Unsupported CHUNK_ID: $CHUNK_ID" >&2; exit 2 ;;
esac

END_FRAME=$((START_FRAME + 32 + HORIZON * 29))
READ_CUE="${READ_CUE:-acl2.gg.qq.low.g2_3.past_only.headmean.robustq}"
READ_PATH_VALUE="${READ_PATH:-frame}"
BETA_VALUE="${BETA_VALUE:-4.75}"
WRITE_SCORE_VALUE="${WRITE_SCORE_VALUE:-stage_d_x_dg_inv_sqrt}"
RUN_MODE="${RUN_MODE:-hybrid}"
RUN_PREFIX="${RUN_PREFIX:-V23_P0_SMOKE_R1}"

STAGE_C_MODE_VALUE="${STAGE_C_MODE:-reference}"
STAGE_C_CACHE_DIR_VALUE="${STAGE_C_CACHE_DIR:-$STAGE_C_CACHE_DEFAULT}"
STAGE_C_CACHE_MODE_VALUE="${STAGE_C_CACHE_MODE:-read}"
STAGE_C_CACHE_REQUIRE_HIT_VALUE="${STAGE_C_CACHE_REQUIRE_HIT:-1}"
STAGE_C_CACHE_VALIDATE_VALUE="${STAGE_C_CACHE_VALIDATE:-0}"
SEMANTIC_PRIOR_MODE_VALUE="${SEMANTIC_PRIOR_MODE:-spg_v2}"

CONTEXT_SOURCE_SKIP_ENABLE=0
CONTEXT_SOURCE_SKIP_IMPL="bias"
CONTEXT_SOURCE_SKIP_SCOPE="frame"
CONTEXT_SOURCE_SKIP_MODE="hard"
CONTEXT_SOURCE_SKIP_MASK="dg_q90"
CONTEXT_SOURCE_SKIP_LAYER_MODE="early"
SEMANTIC_ROLE_POLICY_VALUE="${SEMANTIC_ROLE_POLICY:-none}"
SEMANTIC_MEMORY_PATHS_VALUE="${SEMANTIC_MEMORY_PATHS:-}"
SEMANTIC_ROLE_HIGHD_QUANTILE_VALUE="${SEMANTIC_ROLE_HIGHD_QUANTILE:-0.80}"
SEMANTIC_ROLE_LOW_TRUST_VALUE="${SEMANTIC_ROLE_LOW_TRUST:-0.20}"
USES_CONTEXT_SKIP=false
USES_TRUE_COMPACTION=false
USES_SEMANTIC_CACHE=true

enable_compact_role_skip() {
  CONTEXT_SOURCE_SKIP_ENABLE=1
  CONTEXT_SOURCE_SKIP_IMPL="compact_kv"
  CONTEXT_SOURCE_SKIP_SCOPE="$1"
  CONTEXT_SOURCE_SKIP_MASK="semantic_role_negative"
  CONTEXT_SOURCE_SKIP_LAYER_MODE="${2:-early}"
  USES_CONTEXT_SKIP=true
  USES_TRUE_COMPACTION=true
}

enable_semantic_role() {
  SEMANTIC_ROLE_POLICY_VALUE="$1"
  SEMANTIC_MEMORY_PATHS_VALUE="$2"
}

case "$CANDIDATE_ID" in
  K1_H9|P0_01_SEMANTIC_ROLE_NOOP_IGNORED)
    RUN_MODE="readonly"
    SEMANTIC_PRIOR_MODE_VALUE="pass_through"
    enable_semantic_role noop ""
    ;;
  P0_02_SEMANTIC_ROLE_PASS_THROUGH_CONSUMED)
    RUN_MODE="readonly"
    SEMANTIC_PRIOR_MODE_VALUE="pass_through"
    enable_semantic_role debug_only "all"
    ;;
  P0_03_SEMANTIC_ROLE_DEBUG_ONLY_ALL_MEMORY)
    RUN_MODE="readonly"
    enable_semantic_role debug_only "all"
    ;;
  FRAME_SEM_01_STRUCTURE_KEEP)
    RUN_MODE="readonly"
    enable_semantic_role structure_positive "frame"
    enable_compact_role_skip frame early
    ;;
  FRAME_SEM_02_LOWSTUFF_HIGHD_SKIP)
    RUN_MODE="readonly"
    enable_semantic_role lowstuff_highd_skip "frame"
    enable_compact_role_skip frame early
    ;;
  GLOBAL_SEM_01_STRUCTURE_KEEP)
    RUN_MODE="readonly"
    READ_PATH_VALUE="chunk"
    enable_semantic_role structure_positive "global"
    enable_compact_role_skip chunk early
    ;;
  SWA_SEM_01_STRUCTURE_LONG_KEEP)
    RUN_MODE="hybrid"
    enable_semantic_role structure_positive "swa"
    export ENABLE_SWA_WRITE_CONTROL=1
    export SWA_WRITE_MODE="${SWA_WRITE_MODE:-kv}"
    export SWA_WRITE_SCORE_SOURCE="${SWA_WRITE_SCORE_SOURCE:-read}"
    ;;
  TTT_SEM_01_STRUCTURE_POSITIVE)
    RUN_MODE="hybrid"
    enable_semantic_role structure_positive "ttt"
    ;;
  TTT_SEM_02_LOWSTUFF_HIGHD_SHORT_NEG)
    RUN_MODE="hybrid"
    enable_semantic_role lowstuff_highd_skip "ttt"
    ;;
  ALLSEM_01_FRAME_GLOBAL_STRUCTURE_KEEP)
    RUN_MODE="readonly"
    enable_semantic_role structure_positive "frame,global"
    enable_compact_role_skip both early
    ;;
  ALLSEM_02_FRAME_GLOBAL_LOWSTUFF_HIGHD_SKIP)
    RUN_MODE="readonly"
    enable_semantic_role lowstuff_highd_skip "frame,global"
    enable_compact_role_skip both early
    ;;
  ALLSEM_03_FRAME_GLOBAL_SWA_STRUCTURE_LONG_KEEP)
    RUN_MODE="hybrid"
    enable_semantic_role structure_positive "frame,global,swa"
    enable_compact_role_skip both early
    export ENABLE_SWA_WRITE_CONTROL=1
    export SWA_WRITE_MODE="${SWA_WRITE_MODE:-kv}"
    export SWA_WRITE_SCORE_SOURCE="${SWA_WRITE_SCORE_SOURCE:-read}"
    ;;
  ALLSEM_04_FRAME_GLOBAL_TTT_POSNEG)
    RUN_MODE="hybrid"
    enable_semantic_role all_memory_role "frame,global,ttt"
    enable_compact_role_skip both early
    ;;
  ALLSEM_05_FRAME_GLOBAL_SWA_TTT_ALL_ROLE)
    RUN_MODE="hybrid"
    enable_semantic_role all_memory_role "frame,global,swa,ttt"
    enable_compact_role_skip both early
    export ENABLE_SWA_WRITE_CONTROL=1
    export SWA_WRITE_MODE="${SWA_WRITE_MODE:-kv}"
    ;;
  ALLSEM_06_ALL_ROLE_LONG_SHORT)
    RUN_MODE="hybrid"
    enable_semantic_role long_short "all"
    enable_compact_role_skip both early
    export TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_MODE=dual_lifetime
    export TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_BRANCH_MASK=0
    export TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_LONG_SCALE="${TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_LONG_SCALE:-0.50}"
    export TTT_WRITE_TRANSIENT_DELTA_TTL="${TTT_WRITE_TRANSIENT_DELTA_TTL:-3}"
    ;;
  *)
    echo "Unsupported CANDIDATE_ID for v23 rollout: $CANDIDATE_ID" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_PREFIX}_${CANDIDATE_ID}_chunk${CHUNK_ID}_h${HORIZON}_globalgate_H9parent_SWKS3"
RUN_DIR="$ROOT/$ROLLOUT_BASE/$RUN_NAME"

if [ -d "$RUN_DIR" ]; then
  if [ "${FORCE:-0}" != "1" ] && [ -f "$RUN_DIR/01.txt" ] && grep -q "DONE $RUN_NAME" "$RUN_DIR/run_status.txt" 2>/dev/null; then
    echo "SKIP existing DONE run: $RUN_NAME"
    exit 0
  fi
  STAMP="$(date '+%Y%m%d_%H%M%S')"
  INVALID_DIR="${RUN_DIR}.INVALID_RERUN_${STAMP}"
  mv "$RUN_DIR" "$INVALID_DIR"
  echo "Moved stale/forced run directory to: $INVALID_DIR"
fi

mkdir -p "$RUN_DIR"
cat > "$RUN_DIR/run_config.yaml" <<EOF
run_name: "$RUN_NAME"
candidate_id: "$CANDIDATE_ID"
chunk_id: $CHUNK_ID
horizon: $HORIZON
start_frame: $START_FRAME
end_frame: $END_FRAME
parent: "H9_P0_V16_R2 causal fork snapshots"
diagnostic_only_short_rollout: true
counts_as_online_ttt_write_success: false
uses_gt_runtime_action: false
uses_semantic_cache: $USES_SEMANTIC_CACHE
uses_context_skip: $USES_CONTEXT_SKIP
uses_true_kv_compaction: $USES_TRUE_COMPACTION
semantic_role_policy: "$SEMANTIC_ROLE_POLICY_VALUE"
semantic_memory_paths: "$SEMANTIC_MEMORY_PATHS_VALUE"
context_source_skip_impl: "$CONTEXT_SOURCE_SKIP_IMPL"
context_source_skip_scope: "$CONTEXT_SOURCE_SKIP_SCOPE"
context_source_skip_mask: "$CONTEXT_SOURCE_SKIP_MASK"
stage_c_mode: "$STAGE_C_MODE_VALUE"
stage_c_cache_dir: "$STAGE_C_CACHE_DIR_VALUE"
stage_c_cache_mode: "$STAGE_C_CACHE_MODE_VALUE"
stage_c_cache_require_hit: "$STAGE_C_CACHE_REQUIRE_HIT_VALUE"
run_mode: "$RUN_MODE"
write_score: "$WRITE_SCORE_VALUE"
EOF

env \
  KITTI_SEQ=01 \
  ATTN_CUE_BASE="$ROLLOUT_BASE" \
  START_FRAME="$START_FRAME" \
  END_FRAME="$END_FRAME" \
  GLOBAL_CHUNK_OFFSET="$CHUNK_ID" \
  RESET_EVERY=5 \
  READ_PATH="$READ_PATH_VALUE" \
  READ_LAYER_MODE=all \
  STAGE_C_MODE="$STAGE_C_MODE_VALUE" \
  STAGE_C_CACHE_DIR="$STAGE_C_CACHE_DIR_VALUE" \
  STAGE_C_CACHE_MODE="$STAGE_C_CACHE_MODE_VALUE" \
  STAGE_C_CACHE_REQUIRE_HIT="$STAGE_C_CACHE_REQUIRE_HIT_VALUE" \
  STAGE_C_CACHE_VALIDATE="$STAGE_C_CACHE_VALIDATE_VALUE" \
  SEMANTIC_PRIOR_MODE="$SEMANTIC_PRIOR_MODE_VALUE" \
  ENABLE_CONTEXT_SOURCE_SKIP="$CONTEXT_SOURCE_SKIP_ENABLE" \
  CONTEXT_SOURCE_SKIP_IMPL="$CONTEXT_SOURCE_SKIP_IMPL" \
  CONTEXT_SOURCE_SKIP_SCOPE="$CONTEXT_SOURCE_SKIP_SCOPE" \
  CONTEXT_SOURCE_SKIP_MODE="$CONTEXT_SOURCE_SKIP_MODE" \
  CONTEXT_SOURCE_SKIP_MASK="$CONTEXT_SOURCE_SKIP_MASK" \
  CONTEXT_SOURCE_SKIP_LAYER_MODE="$CONTEXT_SOURCE_SKIP_LAYER_MODE" \
  SEMANTIC_ROLE_POLICY="$SEMANTIC_ROLE_POLICY_VALUE" \
  SEMANTIC_MEMORY_PATHS="$SEMANTIC_MEMORY_PATHS_VALUE" \
  SEMANTIC_ROLE_HIGHD_QUANTILE="$SEMANTIC_ROLE_HIGHD_QUANTILE_VALUE" \
  SEMANTIC_ROLE_LOW_TRUST="$SEMANTIC_ROLE_LOW_TRUST_VALUE" \
  TTT_WRITE_GRADIENT_REVERSAL_MODE="${TTT_WRITE_GRADIENT_REVERSAL_MODE:-tri_replay}" \
  TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE="${TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE:-d_tok}" \
  TTT_WRITE_TRI_REPLAY_POSITIVE_FRAC="${TTT_WRITE_TRI_REPLAY_POSITIVE_FRAC:-0.35}" \
  TTT_WRITE_TRI_REPLAY_NEGATIVE_FRAC="${TTT_WRITE_TRI_REPLAY_NEGATIVE_FRAC:-0.08}" \
  TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA="${TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA:-0.85}" \
  LOAD_HMC_STATE_AT_CHUNK="${LOAD_HMC_STATE_AT_CHUNK:-$PHASE1/state_snapshots/H9_P0_V16_R2/chunk_${SNAP}_input.pt}" \
  LOAD_HMC_STATE_AT_CHUNK_INDEX=0 \
  LOAD_MERGE_STATE_AT_CHUNK="${LOAD_MERGE_STATE_AT_CHUNK:-$PHASE1/merge_state_snapshots/H9_P0_V16_R2/chunk_${SNAP}_input.pt}" \
  LOAD_MERGE_STATE_AT_CHUNK_INDEX=0 \
  "$ROOT/tools/run_attention_cue_experiment.sh" \
  "$GPU" "$RUN_NAME" "$RUN_MODE" "$READ_CUE" "$BETA_VALUE" "$WRITE_SCORE_VALUE"
