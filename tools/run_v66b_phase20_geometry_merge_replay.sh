#!/usr/bin/env bash
set -euo pipefail

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
BASE="${ATTN_CUE_BASE:-results/kitti01_hmc_v2/acl2_v66b_dense_semantic_scale/phase20_geometry_merge_replay/rollouts}"
END_FRAME="${END_FRAME:-704}"
TRACE_ROOT="${TRACE_ROOT:-results/kitti01_hmc_v2/acl2_v66b_dense_semantic_scale/report_final/phase20_geometry_merge_traces}"
RUN_PREFIX="${RUN_PREFIX:-V66B_P20_704_GEOM_MERGE}"

run_one() {
  local gpu="$1"
  local tag="$2"
  local strategy="$3"
  local trace="$ROOT/$TRACE_ROOT/$strategy.jsonl"
  if [ ! -f "$trace" ]; then
    echo "Missing trace: $trace" >&2
    return 2
  fi
  (
    cd "$ROOT"
    LOAD_MERGE_STATE_PATH="$trace" \
    ATTN_CUE_BASE="$BASE" \
    END_FRAME="$END_FRAME" \
    STAGE_C_MODE=reference \
    STAGE_C_CACHE_DIR=results/kitti_preprocess/01/stage_c_cache_semantic_chunks \
    STAGE_C_CACHE_MODE=read \
    STAGE_C_CACHE_REQUIRE_HIT=1 \
    STAGE_C_CACHE_VALIDATE=0 \
    STAGE_C_INLINE_WHEN_IGNORED=1 \
    SEMANTIC_PRIOR_MODE=spg_v2 \
    HMC_IGNORE_SEMANTIC_PRIOR=1 \
    bash tools/run_attention_cue_experiment.sh "$gpu" "${RUN_PREFIX}_${tag}" hybrid dyn 4.75 stage_d
  )
}

run_one 0 S1_GEOMETRY S1_GEOMETRY_ONLY &
pid0=$!
run_one 1 S8_VERTICAL S8_VERTICAL_STATIC_ONLY &
pid1=$!
run_one 2 S8_SHUFFLED S8_VERTICAL_STATIC_ONLY_SHUFFLED &
pid2=$!
run_one 3 S11_SEMWEIGHT S11_SEMANTIC_GEOMETRY_WEIGHTED &
pid3=$!

wait "$pid0"
wait "$pid1"
wait "$pid2"
wait "$pid3"

echo "[$(date '+%F %T')] all v66b phase20 geometry merge replay jobs finished"
