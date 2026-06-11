#!/usr/bin/env bash
set -euo pipefail

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
GPU_LIST_TEXT="${GPU_LIST:-0,1,2,3,4,5}"
RUN_PREFIX="${RUN_PREFIX:-V21_A_SUPPORT_R1}"
PHASE_NAME="${PHASE_NAME:-phaseA_support_true}"
LOG_DIR="${LOG_DIR:-results/kitti01_hmc_v2/acl2_v21_contextskip_semanticallmemory_ttt_persistence_target25/matrix_logs/$PHASE_NAME}"
FORCE="${FORCE:-0}"

IFS=',' read -r -a GPUS <<< "$GPU_LIST_TEXT"
if [ "${#GPUS[@]}" -eq 0 ]; then
  echo "GPU_LIST is empty" >&2
  exit 2
fi

case "$PHASE_NAME" in
  phaseA_support_true)
    CANDIDATES_TEXT="${CANDIDATES:-S0_C23_PAST_LOCKED S1_C23_FULL_CHUNK_TRUE S2_C23_FULL_CHUNK_NO_OVERLAP_TRUE S3_C23_PAST_PLUS_NEAR_FUTURE12 S4_C23_PAST_PLUS_FUTURE_LIGHT_REAL}"
    CHUNKS_TEXT="${CHUNKS:-6 10 16}"
    HORIZONS_TEXT="${HORIZONS:-10 15}"
    ;;
  phaseB_kvcompact_initial)
    CANDIDATES_TEXT="${CANDIDATES:-KVC_01_FRAME_EARLY_DG_Q80_COMPACT KVC_02_FRAME_EARLY_DG_Q90_COMPACT KVC_04_GLOBAL_EARLY_DG_Q80_COMPACT KVC_05_FRAME_GLOBAL_EARLY_DG_Q80_COMPACT KVC_06_FRAME_EARLY_DG_Q80_BIAS_REPEAT}"
    CHUNKS_TEXT="${CHUNKS:-6 10}"
    HORIZONS_TEXT="${HORIZONS:-10 15}"
    ;;
  phaseB_kvcompact_semantic_static_rescue)
    CANDIDATES_TEXT="${CANDIDATES:-KVC_03_FRAME_EARLY_LOWSTUFF_HIGHD_COMPACT KVC_08_FRAME_EARLY_DG_Q80_COMPACT_WITH_STATIC_RESCUE}"
    CHUNKS_TEXT="${CHUNKS:-6 10}"
    HORIZONS_TEXT="${HORIZONS:-10 15}"
    ;;
  phaseD_semantic_coarse_role)
    CANDIDATES_TEXT="${CANDIDATES:-SEMFA_04_LOWSTUFF_HIGHD_FRAME_EARLY_COMPACT SEMFA_05_STRUCTURE_RESCUE_DGQ80_FRAME_EARLY_COMPACT}"
    CHUNKS_TEXT="${CHUNKS:-6 10}"
    HORIZONS_TEXT="${HORIZONS:-10 15}"
    ;;
  phaseE_scale_compact_persistence)
    CANDIDATES_TEXT="${CANDIDATES:-TTTSSP_01_SCALECOMMIT_DGQ80_COMPACT TTTSSP_02_SCALECOMMIT_DGQ80_STRUCTURE_RESCUE_COMPACT}"
    CHUNKS_TEXT="${CHUNKS:-10}"
    HORIZONS_TEXT="${HORIZONS:-10 15}"
    ;;
  *)
    echo "Unsupported PHASE_NAME: $PHASE_NAME" >&2
    exit 2
    ;;
esac

read -r -a CANDIDATES <<< "$CANDIDATES_TEXT"
read -r -a CHUNKS <<< "$CHUNKS_TEXT"
read -r -a HORIZONS <<< "$HORIZONS_TEXT"

TASKS=()
for candidate in "${CANDIDATES[@]}"; do
  for chunk in "${CHUNKS[@]}"; do
    for horizon in "${HORIZONS[@]}"; do
      TASKS+=("$candidate $chunk $horizon")
    done
  done
done

mkdir -p "$ROOT/$LOG_DIR"
printf '%s\n' "${TASKS[@]}" > "$ROOT/$LOG_DIR/tasks.txt"

worker() {
  local gpu="$1"
  local worker_idx="$2"
  local n_workers="$3"
  local task_idx=0
  for task in "${TASKS[@]}"; do
    if [ $((task_idx % n_workers)) -ne "$worker_idx" ]; then
      task_idx=$((task_idx + 1))
      continue
    fi
    read -r candidate chunk horizon <<< "$task"
    local log="$ROOT/$LOG_DIR/gpu${gpu}_${candidate}_chunk${chunk}_h${horizon}.log"
    echo "[$(date '+%F %T')] START gpu=$gpu candidate=$candidate chunk=$chunk h=$horizon" | tee "$log"
    if FORCE="$FORCE" RUN_PREFIX="$RUN_PREFIX" \
       "$ROOT/tools/run_v21_candidate_rollout.sh" "$gpu" "$candidate" "$chunk" "$horizon" >> "$log" 2>&1; then
      echo "[$(date '+%F %T')] DONE gpu=$gpu candidate=$candidate chunk=$chunk h=$horizon" | tee -a "$log"
    else
      rc=$?
      echo "[$(date '+%F %T')] FAIL rc=$rc gpu=$gpu candidate=$candidate chunk=$chunk h=$horizon" | tee -a "$log"
      return "$rc"
    fi
    task_idx=$((task_idx + 1))
  done
}

pids=()
for idx in "${!GPUS[@]}"; do
  worker "${GPUS[$idx]}" "$idx" "${#GPUS[@]}" &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    rc=1
  fi
done
exit "$rc"
