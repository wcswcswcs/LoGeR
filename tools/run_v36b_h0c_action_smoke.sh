#!/usr/bin/env bash
set -euo pipefail

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
RESULT_ROOT="${V36B_ROOT:-results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30}"
SNAP_ROOT="$RESULT_ROOT/phase0_parent_snapshots"
SMOKE_ROOT="$RESULT_ROOT/phase0c_action_smoke"
MATRIX_LOG_DIR="$RESULT_ROOT/matrix_logs/phase0c_action_smoke_R1"
PARENT="${V36B_H0C_PARENT:-H9}"
CHUNK="${V36B_H0C_CHUNK:-10}"
HORIZON="${V36B_H0C_HORIZON:-3}"
CANDIDATES="${V36B_H0C_CANDIDATES:-V31_BASE_H9_REFERENCE,FG_RISK_00,FG_SEM_01,FG_SEM_02,FG_SEM_03,FG_SEM_04,FG_SEM_05}"
GPUS="${V36B_GPUS:-0,1,2,3}"
MAX_PARALLEL="${V36B_MAX_PARALLEL:-4}"

mkdir -p "$ROOT/$MATRIX_LOG_DIR"
IFS=',' read -r -a cand_arr <<< "$CANDIDATES"
IFS=',' read -r -a gpu_arr <<< "$GPUS"

snap="$(printf '%03d' "$CHUNK")"
hmc="$ROOT/$SNAP_ROOT/state_snapshots/${PARENT}_V36B_R1/chunk_${snap}_input.pt"
merge="$ROOT/$SNAP_ROOT/merge_state_snapshots/${PARENT}_V36B_R1/chunk_${snap}_input.pt"
if [ ! -f "$hmc" ] || [ ! -f "$merge" ]; then
  echo "Missing parent snapshot for H0C parent=$PARENT chunk=$CHUNK hmc=$hmc merge=$merge" >&2
  exit 3
fi

launch_row() {
  local gpu="$1"
  local candidate="$2"
  local run_prefix="V36B_H0C_SMOKE_R1_${PARENT}"
  local log="$ROOT/$MATRIX_LOG_DIR/${run_prefix}_${candidate}_chunk${CHUNK}_h${HORIZON}.log"
  echo "[$(date '+%F %T')] START H0C parent=$PARENT candidate=$candidate chunk=$CHUNK h=$HORIZON gpu=$gpu" | tee "$log"
  (
    cd "$ROOT"
    env \
      V24_ROOT="$SMOKE_ROOT" \
      RUN_PREFIX="$run_prefix" \
      LOAD_HMC_STATE_AT_CHUNK="$hmc" \
      LOAD_MERGE_STATE_AT_CHUNK="$merge" \
      FORCE="${FORCE:-0}" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$ROOT/tools/run_v24_candidate_rollout.sh" "$gpu" "$candidate" "$CHUNK" "$HORIZON"
  ) 2>&1 | tee -a "$log"
  echo "[$(date '+%F %T')] END H0C parent=$PARENT candidate=$candidate chunk=$CHUNK h=$HORIZON gpu=$gpu" | tee -a "$log"
}

running=0
idx=0
for candidate in "${cand_arr[@]}"; do
  candidate="$(echo "$candidate" | tr -d '[:space:]')"
  gpu="${gpu_arr[$((idx % ${#gpu_arr[@]}))]}"
  launch_row "$gpu" "$candidate" &
  running=$((running + 1))
  idx=$((idx + 1))
  if [ "$running" -ge "$MAX_PARALLEL" ]; then
    wait -n
    running=$((running - 1))
  fi
done

while [ "$running" -gt 0 ]; do
  wait -n
  running=$((running - 1))
done
