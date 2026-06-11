#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 GPU PHASE" >&2
  echo "PHASE: phase0 | phase2 | phase3 | phase4_smoke" >&2
  exit 2
fi

GPU="$1"
PHASE="$2"
ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
RUNNER="$ROOT/tools/run_v23_candidate_rollout.sh"

run_one() {
  local candidate="$1"
  local chunk="$2"
  local horizon="$3"
  "$RUNNER" "$GPU" "$candidate" "$chunk" "$horizon"
}

case "$PHASE" in
  phase0)
    export RUN_PREFIX="${RUN_PREFIX:-V23_P0_SMOKE_R1}"
    run_one P0_01_SEMANTIC_ROLE_NOOP_IGNORED 10 3
    run_one P0_02_SEMANTIC_ROLE_PASS_THROUGH_CONSUMED 10 3
    run_one P0_03_SEMANTIC_ROLE_DEBUG_ONLY_ALL_MEMORY 10 3
    ;;
  phase2)
    export RUN_PREFIX="${RUN_PREFIX:-V23_P2_SINGLEPATH_R1}"
    for candidate in \
      FRAME_SEM_01_STRUCTURE_KEEP \
      FRAME_SEM_02_LOWSTUFF_HIGHD_SKIP \
      GLOBAL_SEM_01_STRUCTURE_KEEP \
      SWA_SEM_01_STRUCTURE_LONG_KEEP \
      TTT_SEM_01_STRUCTURE_POSITIVE \
      TTT_SEM_02_LOWSTUFF_HIGHD_SHORT_NEG
    do
      for chunk in 6 10 16; do
        run_one "$candidate" "$chunk" 10
        run_one "$candidate" "$chunk" 15
      done
    done
    ;;
  phase3)
    export RUN_PREFIX="${RUN_PREFIX:-V23_P3_ALLMEM_R1}"
    for candidate in \
      ALLSEM_01_FRAME_GLOBAL_STRUCTURE_KEEP \
      ALLSEM_02_FRAME_GLOBAL_LOWSTUFF_HIGHD_SKIP \
      ALLSEM_03_FRAME_GLOBAL_SWA_STRUCTURE_LONG_KEEP \
      ALLSEM_04_FRAME_GLOBAL_TTT_POSNEG \
      ALLSEM_05_FRAME_GLOBAL_SWA_TTT_ALL_ROLE \
      ALLSEM_06_ALL_ROLE_LONG_SHORT
    do
      for chunk in 6 10 16; do
        run_one "$candidate" "$chunk" 10
        run_one "$candidate" "$chunk" 15
      done
    done
    ;;
  phase4_smoke)
    export RUN_PREFIX="${RUN_PREFIX:-V23_P4_ATTR_SMOKE_R1}"
    run_one ALLSEM_06_ALL_ROLE_LONG_SHORT 10 15
    ;;
  *)
    echo "Unsupported PHASE: $PHASE" >&2
    exit 2
    ;;
esac
