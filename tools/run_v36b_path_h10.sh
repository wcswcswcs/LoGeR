#!/usr/bin/env bash
set -euo pipefail

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
RESULT_ROOT="${V36B_RESULT_ROOT:-results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30}"
PHASE_NAME="${V36B_PHASE_NAME:?set V36B_PHASE_NAME, e.g. phase2_h2_swa_h10}"
RUN_PREFIX_BASE="${V36B_RUN_PREFIX_BASE:?set V36B_RUN_PREFIX_BASE, e.g. V36B_H2_H10_R1}"
CANDIDATES_CSV="${V36B_CANDIDATES:?set V36B_CANDIDATES comma-separated}"
CHUNKS_CSV="${V36B_CHUNKS:-6,10,16}"
PARENTS_CSV="${V36B_PARENTS:-H9,C9}"
HORIZON="${V36B_HORIZON:-10}"
GPUS_CSV="${V36B_GPUS:-0,1,2,3}"
FORCE="${FORCE:-0}"

IFS=',' read -r -a CANDIDATES <<< "$CANDIDATES_CSV"
IFS=',' read -r -a CHUNKS <<< "$CHUNKS_CSV"
IFS=',' read -r -a PARENTS <<< "$PARENTS_CSV"
IFS=',' read -r -a GPUS <<< "$GPUS_CSV"

if [ "${#GPUS[@]}" -lt 1 ]; then
  echo "No GPUs supplied" >&2
  exit 2
fi

ROLLOUT_PARENT="$ROOT/$RESULT_ROOT/$PHASE_NAME"
ROLLOUT_ROOT="$ROLLOUT_PARENT/rollouts"
LOG_DIR="$ROOT/$RESULT_ROOT/matrix_logs/$PHASE_NAME"
mkdir -p "$ROLLOUT_ROOT" "$LOG_DIR"

jobs_file="$(mktemp)"
fail_file="$(mktemp)"
trap 'rm -f "$jobs_file" "$fail_file"' EXIT

for parent in "${PARENTS[@]}"; do
  for chunk in "${CHUNKS[@]}"; do
    for candidate in "${CANDIDATES[@]}"; do
      echo "$parent,$chunk,$candidate" >> "$jobs_file"
    done
  done
done

run_one() {
  local gpu="$1"
  local parent="$2"
  local chunk="$3"
  local candidate="$4"
  local snap
  snap="$(printf '%03d' "$chunk")"
  local state_parent="${parent}_V36B_R1"
  local state="$ROOT/$RESULT_ROOT/phase0_parent_snapshots/state_snapshots/$state_parent/chunk_${snap}_input.pt"
  local merge="$ROOT/$RESULT_ROOT/phase0_parent_snapshots/merge_state_snapshots/$state_parent/chunk_${snap}_input.pt"
  if [ ! -f "$state" ] || [ ! -f "$merge" ]; then
    echo "missing parent snapshots parent=$parent chunk=$chunk state=$state merge=$merge" >&2
    return 3
  fi
  local prefix="${RUN_PREFIX_BASE}_${parent}"
  local log="$LOG_DIR/${prefix}_${candidate}_chunk${chunk}_h${HORIZON}.log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START parent=$parent candidate=$candidate chunk=$chunk h=$HORIZON gpu=$gpu" | tee -a "$log"
  (
    cd "$ROOT"
    env \
      LOGER_ROOT="$ROOT" \
      V24_ROOT="$RESULT_ROOT/$PHASE_NAME" \
      RUN_PREFIX="$prefix" \
      FORCE="$FORCE" \
      LOAD_HMC_STATE_AT_CHUNK="$state" \
      LOAD_MERGE_STATE_AT_CHUNK="$merge" \
      "$ROOT/tools/run_v24_candidate_rollout.sh" "$gpu" "$candidate" "$chunk" "$HORIZON"
  ) 2>&1 | tee -a "$log"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAIL parent=$parent candidate=$candidate chunk=$chunk h=$HORIZON gpu=$gpu rc=$rc" | tee -a "$log"
    return "$rc"
  fi
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] END parent=$parent candidate=$candidate chunk=$chunk h=$HORIZON gpu=$gpu" | tee -a "$log"
}

worker() {
  local gpu="$1"
  while IFS=, read -r parent chunk candidate; do
    [ -n "${parent:-}" ] || continue
    if ! run_one "$gpu" "$parent" "$chunk" "$candidate"; then
      echo "$gpu,$parent,$chunk,$candidate" >> "$fail_file"
    fi
  done
}

for i in "${!GPUS[@]}"; do
  awk -v n="${#GPUS[@]}" -v r="$i" 'NR % n == r' "$jobs_file" > "${jobs_file}.${i}"
  worker "${GPUS[$i]}" < "${jobs_file}.${i}" &
done

wait

if [ -s "$fail_file" ]; then
  echo "Failed rows:" >&2
  cat "$fail_file" >&2
  exit 1
fi
