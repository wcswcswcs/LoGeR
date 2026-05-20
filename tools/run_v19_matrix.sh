#!/usr/bin/env bash
set -euo pipefail

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
GPU_LIST_TEXT="${GPU_LIST:-0,1,2,3,4,5}"
RUN_PREFIX="${RUN_PREFIX:-V19_P2SCALE_R1}"
PHASE_NAME="${PHASE_NAME:-trackB1B2_scale_state_initial}"
LOG_DIR="${LOG_DIR:-results/kitti01_hmc_v2/acl2_v19_ttt_trajectory_state_write_reboot_target25/matrix_logs/$PHASE_NAME}"
FORCE="${FORCE:-0}"

IFS=',' read -r -a GPUS <<< "$GPU_LIST_TEXT"
if [ "${#GPUS[@]}" -eq 0 ]; then
  echo "GPU_LIST is empty" >&2
  exit 2
fi

case "$PHASE_NAME" in
  trackB1B2_scale_state_initial)
    CANDIDATES_TEXT="${CANDIDATES:-K1_H9 SCALETTT_01_SPECIAL_TOKEN_W0_A005 SCALETTT_02_STRUCTURE_LOWDG_W0_A005 SCALETTT_03_OVERLAP_STATIC_W0W2_A005 SCALETTT_04_STRUCTURE_LOWDG_W0_A010 SCALETTT_05_OVERLAP_STATIC_W0W2_A010 SCALECOMMIT_01_PZBASIS_HARM_W0_G025 SCALECOMMIT_02_AUXGEO_OVERLAP_W0_G025 SCALECOMMIT_03_HIST_DELTA_W0_G025}"
    CHUNKS_TEXT="${CHUNKS:-6 10}"
    HORIZONS_TEXT="${HORIZONS:-5 8 10}"
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
       "$ROOT/tools/run_v19_candidate_rollout.sh" "$gpu" "$candidate" "$chunk" "$horizon" >> "$log" 2>&1; then
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
