#!/usr/bin/env bash
set -euo pipefail

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
RESULT_ROOT="${V36B_ROOT:-results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30}"
SNAP_ROOT="$RESULT_ROOT/phase0_parent_snapshots"
H1_ROOT="$RESULT_ROOT/phase1_h1_frame_global_h10"
MATRIX_LOG_DIR="$RESULT_ROOT/matrix_logs/phase1_h1_h10_R1"
PARENTS="${V36B_PARENTS:-H9,C9}"
CHUNKS="${V36B_CHUNKS:-6,10,16}"
HORIZON="${V36B_HORIZON:-10}"
CANDIDATES="${V36B_CANDIDATES:-V31_BASE_H9_REFERENCE,FG_RISK_00,FG_SEM_04}"
GPUS="${V36B_GPUS:-0,1,2,3}"
MAX_PARALLEL="${V36B_MAX_PARALLEL:-4}"
RUN_PREFIX_BASE="${V36B_H1_RUN_PREFIX_BASE:-V36B_H1_H10_R1}"

mkdir -p "$ROOT/$MATRIX_LOG_DIR"

IFS=',' read -r -a parent_arr <<< "$PARENTS"
IFS=',' read -r -a chunk_arr <<< "$CHUNKS"
IFS=',' read -r -a cand_arr <<< "$CANDIDATES"
IFS=',' read -r -a gpu_arr <<< "$GPUS"

if [ "${#gpu_arr[@]}" -lt 1 ]; then
  echo "No GPUs configured" >&2
  exit 2
fi

launch_row() {
  local gpu="$1"
  local parent="$2"
  local chunk="$3"
  local candidate="$4"
  local snap
  snap="$(printf '%03d' "$chunk")"
  local hmc="$ROOT/$SNAP_ROOT/state_snapshots/${parent}_V36B_R1/chunk_${snap}_input.pt"
  local merge="$ROOT/$SNAP_ROOT/merge_state_snapshots/${parent}_V36B_R1/chunk_${snap}_input.pt"
  if [ ! -f "$hmc" ] || [ ! -f "$merge" ]; then
    echo "Missing parent snapshot for parent=$parent chunk=$chunk hmc=$hmc merge=$merge" >&2
    return 3
  fi
  local run_prefix="${RUN_PREFIX_BASE}_${parent}"
  local log="$ROOT/$MATRIX_LOG_DIR/${run_prefix}_${candidate}_chunk${chunk}_h${HORIZON}.log"
  echo "[$(date '+%F %T')] START parent=$parent candidate=$candidate chunk=$chunk h=$HORIZON gpu=$gpu" | tee "$log"
  (
    cd "$ROOT"
    env \
      V24_ROOT="$H1_ROOT" \
      RUN_PREFIX="$run_prefix" \
      LOAD_HMC_STATE_AT_CHUNK="$hmc" \
      LOAD_MERGE_STATE_AT_CHUNK="$merge" \
      FORCE="${FORCE:-0}" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$ROOT/tools/run_v24_candidate_rollout.sh" "$gpu" "$candidate" "$chunk" "$HORIZON"
  ) 2>&1 | tee -a "$log"
  echo "[$(date '+%F %T')] END parent=$parent candidate=$candidate chunk=$chunk h=$HORIZON gpu=$gpu" | tee -a "$log"
}

JOB_FILE="$(mktemp "${TMPDIR:-/tmp}/v36b_h1_jobs.XXXXXX")"
FAIL_FILE="$(mktemp "${TMPDIR:-/tmp}/v36b_h1_fail.XXXXXX")"
trap 'rm -f "$JOB_FILE" "$FAIL_FILE"' EXIT

idx=0
for parent in "${parent_arr[@]}"; do
  parent="$(echo "$parent" | tr -d '[:space:]')"
  for chunk in "${chunk_arr[@]}"; do
    chunk="$(echo "$chunk" | tr -d '[:space:]')"
    for candidate in "${cand_arr[@]}"; do
      candidate="$(echo "$candidate" | tr -d '[:space:]')"
      printf '%s,%s,%s,%s\n' "$idx" "$parent" "$chunk" "$candidate" >> "$JOB_FILE"
      idx=$((idx + 1))
    done
  done
done

worker() {
  local worker_idx="$1"
  local gpu="$2"
  local gpu_count="$3"
  while IFS=',' read -r job_idx parent chunk candidate; do
    if [ $((job_idx % gpu_count)) -ne "$worker_idx" ]; then
      continue
    fi
    if ! launch_row "$gpu" "$parent" "$chunk" "$candidate"; then
      printf '%s,%s,%s,%s,%s\n' "$job_idx" "$gpu" "$parent" "$chunk" "$candidate" >> "$FAIL_FILE"
    fi
  done < "$JOB_FILE"
}

gpu_count="${#gpu_arr[@]}"
worker_count="$gpu_count"
if [ "$MAX_PARALLEL" -lt "$worker_count" ]; then
  worker_count="$MAX_PARALLEL"
fi

for worker_idx in $(seq 0 $((worker_count - 1))); do
  worker "$worker_idx" "${gpu_arr[$worker_idx]}" "$worker_count" &
done
wait

if [ -s "$FAIL_FILE" ]; then
  echo "FAILED H1 rows:" >&2
  cat "$FAIL_FILE" >&2
  exit 1
fi
