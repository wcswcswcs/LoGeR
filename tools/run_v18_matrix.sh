#!/usr/bin/env bash
set -euo pipefail

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
GPU_LIST_TEXT="${GPU_LIST:-0,1,2,3,4,5,6,7}"
RUN_PREFIX="${RUN_PREFIX:-V18}"
PHASE_NAME="${PHASE_NAME:-phase2_pzbasis_initial}"
LOG_DIR="${LOG_DIR:-results/kitti01_hmc_v2/acl2_v18_ttt_write_true_actionspace_target25/matrix_logs/$PHASE_NAME}"
FORCE="${FORCE:-0}"
SAVE_TRUE_ACTION_TRACE="${SAVE_TRUE_ACTION_TRACE:-0}"
V18_TRUE_ACTION_TRACE_LAYERS="${V18_TRUE_ACTION_TRACE_LAYERS:-0,6,12,17}"
V18_TRUE_ACTION_TRACE_BRANCHES="${V18_TRUE_ACTION_TRACE_BRANCHES:-w0}"

IFS=',' read -r -a GPUS <<< "$GPU_LIST_TEXT"
if [ "${#GPUS[@]}" -eq 0 ]; then
  echo "GPU_LIST is empty" >&2
  exit 2
fi

case "$PHASE_NAME" in
  phase2_pzbasis_initial)
    CANDIDATES_TEXT="${CANDIDATES:-K1_H9 PZBASIS_01_CONTINUITY_BOOST_W0 PZBASIS_02_HARM_SUPPRESS_W0 PZBASIS_03_HARM_SUPPRESS_W0W2 PZBASIS_04_SCALE_BASIS_SUPPRESS_W0 PZBASIS_05_OVERLAP_BASIS_BOOST_W0 PZBASIS_06_CONTINUITY_PLUS_HARM_W0}"
    CHUNKS_TEXT="${CHUNKS:-6 10}"
    HORIZONS_TEXT="${HORIZONS:-5 8 10}"
    ;;
  phase3_auxgeo_initial)
    CANDIDATES_TEXT="${CANDIDATES:-K1_H9 AUXGEO_TRUE_01_OVERLAP_POINTMAP_V_W0 AUXGEO_TRUE_02_OVERLAP_POINTMAP_KV_W0 AUXGEO_TRUE_03_OVERLAP_SCALE_PROXY_W0 AUXGEO_TRUE_04_OVERLAP_SCALE_PROXY_W0W2 AUXGEO_TRUE_05_STRUCTURE_ONLY_OVERLAP_W0 AUXGEO_TRUE_06_LOWD_STRUCTURE_OVERLAP_W0}"
    CHUNKS_TEXT="${CHUNKS:-6 10}"
    HORIZONS_TEXT="${HORIZONS:-5 8 10}"
    ;;
  phase4_dltrue_initial)
    CANDIDATES_TEXT="${CANDIDATES:-K1_H9 DLTRUE_01_SHORT_HARM_ONLY_W0 DLTRUE_02_SHORT_BODY_HARM_LONG_CONTINUITY DLTRUE_03_SHORT_SCALE_CORRECTION_LONG_STRUCTURE DLTRUE_04_SHORT_OVERLAP_CORRECTION_LONG_NATIVE DLTRUE_05_SHORT_DECAY_FAST_K2 DLTRUE_06_SHORT_DECAY_SLOW_K5 DLTRUE_07_RESET_BOUND_SHORT_CLEAR DLTRUE_08_EXIT_WEAK_SHORT_HANDOFF}"
    CHUNKS_TEXT="${CHUNKS:-10}"
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
    if SAVE_TRUE_ACTION_TRACE="$SAVE_TRUE_ACTION_TRACE" \
       V18_TRUE_ACTION_TRACE_LAYERS="$V18_TRUE_ACTION_TRACE_LAYERS" \
       V18_TRUE_ACTION_TRACE_BRANCHES="$V18_TRUE_ACTION_TRACE_BRANCHES" \
       FORCE="$FORCE" \
       RUN_PREFIX="$RUN_PREFIX" \
       "$ROOT/tools/run_v18_candidate_rollout.sh" "$gpu" "$candidate" "$chunk" "$horizon" >> "$log" 2>&1; then
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
