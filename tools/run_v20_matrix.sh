#!/usr/bin/env bash
set -euo pipefail

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
GPU_LIST_TEXT="${GPU_LIST:-0,1,2,3,4,5}"
RUN_PREFIX="${RUN_PREFIX:-V20_A_SUPPORT_R1}"
PHASE_NAME="${PHASE_NAME:-batchA_support_initial}"
LOG_DIR="${LOG_DIR:-results/kitti01_hmc_v2/acl2_v20_ttt_contextskip_semanticmemory_target25/matrix_logs/$PHASE_NAME}"
FORCE="${FORCE:-0}"

IFS=',' read -r -a GPUS <<< "$GPU_LIST_TEXT"
if [ "${#GPUS[@]}" -eq 0 ]; then
  echo "GPU_LIST is empty" >&2
  exit 2
fi

case "$PHASE_NAME" in
  batchA_support_initial)
    CANDIDATES_TEXT="${CANDIDATES:-S1_00_C23_PAST S1_01_C23_FULL_CHUNK S1_02_C23_FULL_CHUNK_NO_OVERLAP S1_03_C23_PAST_PLUS_FUTURE_LIGHT S1_04_C23_NEAR24}"
    CHUNKS_TEXT="${CHUNKS:-6 10}"
    HORIZONS_TEXT="${HORIZONS:-10 15}"
    ;;
  batchD_scale_anchor_repeat)
    CANDIDATES_TEXT="${CANDIDATES:-SCALECOMMIT_01_PZBASIS_HARM_W0_G025}"
    CHUNKS_TEXT="${CHUNKS:-10}"
    HORIZONS_TEXT="${HORIZONS:-10 15}"
    ;;
  batchB_kvskip_initial)
    CANDIDATES_TEXT="${CANDIDATES:-KVS_00_C23_PAST_PAIR KVS_01_FRAME_EARLY_DG_Q80_HARD KVS_02_FRAME_EARLY_DG_Q90_HARD KVS_03_FRAME_EARLY_LOWSTUFF_HIGHD_HARD KVS_06_FRAME_EARLY_DG_Q90_HARD_PLUS_PAIR KVS_07_CHUNK_EARLY_DG_Q90_HARD}"
    CHUNKS_TEXT="${CHUNKS:-6 10}"
    HORIZONS_TEXT="${HORIZONS:-10 15}"
    ;;
  batchB_kvskip_soft_fallback)
    CANDIDATES_TEXT="${CANDIDATES:-KVS_09_FRAME_EARLY_DG_Q90_SOFT_R025}"
    CHUNKS_TEXT="${CHUNKS:-6 10}"
    HORIZONS_TEXT="${HORIZONS:-10 15}"
    ;;
  batchD_scale_skip_combo)
    CANDIDATES_TEXT="${CANDIDATES:-TTTSS_03_SCALECOMMIT_DGQ90_HARD TTTSS_03B_SCALECOMMIT_DGQ80_HARD}"
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
       "$ROOT/tools/run_v20_candidate_rollout.sh" "$gpu" "$candidate" "$chunk" "$horizon" >> "$log" 2>&1; then
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
