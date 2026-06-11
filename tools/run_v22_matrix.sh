#!/usr/bin/env bash
set -euo pipefail

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
GPU_LIST_TEXT="${GPU_LIST:-0,1,2,3,4,5,6,7}"
RUN_PREFIX="${RUN_PREFIX:-V22_A_SUPPORT_R1}"
PHASE_NAME="${PHASE_NAME:-phaseA_support_compact_semantic}"
LOG_DIR="${LOG_DIR:-results/kitti01_hmc_v2/acl2_v22_durable_contextskip_semanticallmemory_ttt_target25/matrix_logs/$PHASE_NAME}"
FORCE="${FORCE:-0}"

IFS=',' read -r -a GPUS <<< "$GPU_LIST_TEXT"
if [ "${#GPUS[@]}" -eq 0 ]; then
  echo "GPU_LIST is empty" >&2
  exit 2
fi

case "$PHASE_NAME" in
  phaseA_support_setting_A)
    CANDIDATES_TEXT="${CANDIDATES:-SUP_LOCKED_A SUP_FULL_TRUE_A SUP_NO_OVERLAP_A SUP_PAST_NEAR_FUTURE12_A}"
    CHUNKS_TEXT="${CHUNKS:-6 10 16}"
    HORIZONS_TEXT="${HORIZONS:-10 15}"
    ;;
  phaseA_support_setting_B)
    CANDIDATES_TEXT="${CANDIDATES:-SUP_LOCKED_B_DGQ80_COMPACT SUP_FULL_TRUE_B_DGQ80_COMPACT SUP_NO_OVERLAP_B_DGQ80_COMPACT SUP_PAST_NEAR_FUTURE12_B_DGQ80_COMPACT}"
    CHUNKS_TEXT="${CHUNKS:-6 10 16}"
    HORIZONS_TEXT="${HORIZONS:-10 15}"
    ;;
  phaseA_support_setting_C)
    CANDIDATES_TEXT="${CANDIDATES:-SUP_LOCKED_C_STRUCTURE_RESCUE_COMPACT SUP_FULL_TRUE_C_STRUCTURE_RESCUE_COMPACT SUP_NO_OVERLAP_C_STRUCTURE_RESCUE_COMPACT SUP_PAST_NEAR_FUTURE12_C_STRUCTURE_RESCUE_COMPACT}"
    CHUNKS_TEXT="${CHUNKS:-6 10 16}"
    HORIZONS_TEXT="${HORIZONS:-10 15}"
    ;;
  phaseA_support_compact_semantic)
    CANDIDATES_TEXT="${CANDIDATES:-SUP_LOCKED_B_DGQ80_COMPACT SUP_FULL_TRUE_B_DGQ80_COMPACT SUP_NO_OVERLAP_B_DGQ80_COMPACT SUP_PAST_NEAR_FUTURE12_B_DGQ80_COMPACT SUP_LOCKED_C_STRUCTURE_RESCUE_COMPACT SUP_FULL_TRUE_C_STRUCTURE_RESCUE_COMPACT SUP_NO_OVERLAP_C_STRUCTURE_RESCUE_COMPACT SUP_PAST_NEAR_FUTURE12_C_STRUCTURE_RESCUE_COMPACT}"
    CHUNKS_TEXT="${CHUNKS:-6 10 16}"
    HORIZONS_TEXT="${HORIZONS:-10 15}"
    ;;
  phaseB_read_only_compact)
    CANDIDATES_TEXT="${CANDIDATES:-KVC_READ_01 KVC_READ_02 KVC_READ_03 KVC_READ_04}"
    CHUNKS_TEXT="${CHUNKS:-6 10}"
    HORIZONS_TEXT="${HORIZONS:-10 15}"
    ;;
  phaseC_semantic_role_initial)
    CANDIDATES_TEXT="${CANDIDATES:-SEM_ROLE_01_STRUCTURE_RESCUE SEM_ROLE_02_LOWSTUFF_HIGHD_SKIP}"
    CHUNKS_TEXT="${CHUNKS:-6 10}"
    HORIZONS_TEXT="${HORIZONS:-10 15}"
    ;;
  phaseC_semantic_role_extended)
    CANDIDATES_TEXT="${CANDIDATES:-SEM_ROLE_04_STRUCTURE_POSITIVE_TTT SEM_ROLE_05_ALL_MEMORY_ROLE}"
    CHUNKS_TEXT="${CHUNKS:-6 10}"
    HORIZONS_TEXT="${HORIZONS:-10 15}"
    ;;
  phaseD_v21_strongest_attribution)
    CANDIDATES_TEXT="${CANDIDATES:-TTTSSP_02_SCALECOMMIT_DGQ80_STRUCTURE_RESCUE_COMPACT}"
    CHUNKS_TEXT="${CHUNKS:-10}"
    HORIZONS_TEXT="${HORIZONS:-15}"
    ;;
  phaseE_skip_aware_ttt)
    CANDIDATES_TEXT="${CANDIDATES:-KVC_TTT_01_NEUTRAL_COMMIT_FILTER KVC_TTT_02_WEAK_NEGATIVE KVC_TTT_03_STRUCTURE_KEPT_BOOST KVC_TTT_04_SOURCE_KEEP_GATED_WRITE}"
    CHUNKS_TEXT="${CHUNKS:-10}"
    HORIZONS_TEXT="${HORIZONS:-10 15}"
    ;;
  phaseF_skip_aware_memory)
    CANDIDATES_TEXT="${CANDIDATES:-KVC_MEM_01_SWA_COMPACT_OVERLAP_HISTORY KVC_MEM_02_SWA_DOWNWEIGHT_SKIPPED KVC_MEM_03_GLOBAL_CHUNK_SOURCE_SKIP KVC_MEM_04_TTT_AND_SWA_DOWNWEIGHT}"
    CHUNKS_TEXT="${CHUNKS:-10}"
    HORIZONS_TEXT="${HORIZONS:-10 15}"
    ;;
  phaseG_ttt_durable_commit)
    CANDIDATES_TEXT="${CANDIDATES:-TTT_DUR_01_READ_COMPACT_ONLY TTT_DUR_02_SKIP_AWARE_COMMIT_FILTER TTT_DUR_03_NATIVE_READ_SKIP_REPLAY_ONLY TTT_DUR_04_POST_ZP_SKIP_BASIS_ROUTING}"
    CHUNKS_TEXT="${CHUNKS:-10}"
    HORIZONS_TEXT="${HORIZONS:-10 15}"
    ;;
  phaseH_ttt_lifecycle)
    CANDIDATES_TEXT="${CANDIDATES:-TTT_LIFE_01_SHORT_HIGHD_K2 TTT_LIFE_02_SHORT_HIGHD_K4 TTT_LIFE_03_LOWSTUFF_SHORT_STRUCTURE_LONG TTT_LIFE_04_SCALE_LONG_HIGHD_SHORT}"
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
       "$ROOT/tools/run_v22_candidate_rollout.sh" "$gpu" "$candidate" "$chunk" "$horizon" >> "$log" 2>&1; then
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
