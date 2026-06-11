#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 5 ]; then
  echo "Usage: $0 GPU PARENT CANDIDATE_ID CHUNK_ID HORIZON" >&2
  echo "PARENT: C9 | C9_CLEAN" >&2
  echo "CANDIDATE_ID: S0_C23_PAST .. S5_C23_PAST_PLUS_FUTURE_LIGHT" >&2
  exit 2
fi

GPU="$1"
PARENT="$2"
CANDIDATE_ID="$3"
CHUNK_ID="$4"
HORIZON="$5"

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
RESULT_ROOT="${V45_RESULT_ROOT:-$ROOT/results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay}"

case "$CHUNK_ID" in
  6) START_FRAME=174; SNAP="006" ;;
  10) START_FRAME=290; SNAP="010" ;;
  16) START_FRAME=464; SNAP="016" ;;
  *) echo "Unsupported v45 Phase3 short CHUNK_ID=$CHUNK_ID" >&2; exit 2 ;;
esac

case "$HORIZON" in
  10) ;;
  *) echo "Unsupported v45 Phase3 short HORIZON=$HORIZON" >&2; exit 2 ;;
esac

case "$PARENT" in
  C9|c9)
    PARENT_LABEL="C9"
    V45_PARENT_VALUE="C9"
    SNAP_PARENT="phase0_hard_gate"
    SNAP_RUN="V45_P0_C9_REPEAT"
    ;;
  C9_CLEAN|c9_clean|clean)
    PARENT_LABEL="C9CLEAN"
    V45_PARENT_VALUE="C9_CLEAN"
    SNAP_PARENT="phase1_c9_clean"
    SNAP_RUN="V45_D7_C9_CLEAN_BEST_FIXED"
    ;;
  *) echo "Unsupported PARENT=$PARENT" >&2; exit 2 ;;
esac

STATE_PATH="$RESULT_ROOT/$SNAP_PARENT/state_snapshots/$SNAP_RUN/chunk_${SNAP}_input.pt"
MERGE_PATH="$RESULT_ROOT/$SNAP_PARENT/merge_state_snapshots/$SNAP_RUN/chunk_${SNAP}_input.pt"
if [ ! -s "$STATE_PATH" ]; then
  echo "Missing HMC snapshot: $STATE_PATH" >&2
  exit 1
fi
if [ ! -s "$MERGE_PATH" ]; then
  echo "Missing merge snapshot: $MERGE_PATH" >&2
  exit 1
fi

END_FRAME=$((START_FRAME + 32 + HORIZON * 29))
RUN_NAME="V45_P3SHORT_${PARENT_LABEL}_${CANDIDATE_ID}_CH${CHUNK_ID}_H${HORIZON}_READONLY"

V45_ROLLOUT_BASE="$RESULT_ROOT/phase3_c23_support/short_rollouts" \
V45_PARENT="$V45_PARENT_VALUE" \
V45_MODE_OVERRIDE=readonly \
HMC_COMMIT_MODE=probe_native \
START_FRAME="$START_FRAME" \
END_FRAME="$END_FRAME" \
GLOBAL_CHUNK_OFFSET="$CHUNK_ID" \
LOAD_HMC_STATE_AT_CHUNK="$STATE_PATH" \
LOAD_HMC_STATE_AT_CHUNK_INDEX=0 \
LOAD_MERGE_STATE_AT_CHUNK="$MERGE_PATH" \
LOAD_MERGE_STATE_AT_CHUNK_INDEX=0 \
V45_C9_CLEAN_TRI_GAMMA="${V45_C9_CLEAN_TRI_GAMMA:-0.004}" \
V45_C9_CLEAN_COMMIT_EMA_ALPHA="${V45_C9_CLEAN_COMMIT_EMA_ALPHA:-1.0}" \
V45_C9_CLEAN_COMMIT_EMA_BRANCH_MASK="${V45_C9_CLEAN_COMMIT_EMA_BRANCH_MASK:-all}" \
"$ROOT/tools/run_v45_full_candidate.sh" "$GPU" "$RUN_NAME" "$CANDIDATE_ID"
