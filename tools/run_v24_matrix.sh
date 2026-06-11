#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 GPU PHASE" >&2
  echo "PHASE: phase0 | phase1 | phase2_initial | phase2_expand | phase3 | phase4 | phase5_attr" >&2
  exit 2
fi

GPU="$1"
PHASE="$2"
ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
RUNNER="$ROOT/tools/run_v24_candidate_rollout.sh"

run_one() {
  local candidate="$1"
  local chunk="$2"
  local horizon="$3"
  "$RUNNER" "$GPU" "$candidate" "$chunk" "$horizon"
}

phase2_candidates() {
  printf '%s\n' \
    FRAMESEM_01_STRUCTURE_KEEP_LOWSTUFF_SOFT \
    FRAMESEM_02_LOWSTUFF_HIGHD_SKIP \
    FRAMESEM_03_SKY_NEUTRAL_VEGETATION_HIGHD_SKIP \
    GLOBALSEM_01_STRUCTURE_KEEP_LOWSTUFF_SOFT \
    GLOBALSEM_02_LOWSTUFF_HIGHD_SKIP \
    FRAMEGLOBAL_01_FRAME_ONLY \
    FRAMEGLOBAL_02_GLOBAL_ONLY \
    FRAMEGLOBAL_03_FRAME_AND_GLOBAL \
    SWASEM_01_STRUCTURE_CACHE_KEEP \
    SWASEM_02_LOWSTUFF_HIGHD_CACHE_SOFTDROP \
    SWASEM_03_SKY_PROTECT_VEG_HIGHD_DROP \
    SWASEM_04_PREVIOUS_SOURCE_ONLY \
    SWASEM_05_OVERLAP_ONLY \
    SWASEM_06_CURRENT_AND_PREVIOUS_COMPARE \
    TTTSEM_01_STRUCTURE_POSITIVE_LONG \
    TTTSEM_02_LOWSTUFF_HIGHD_NEGATIVE_SHORT \
    TTTSEM_03_SKY_NEUTRAL_PROTECT \
    TTTSEM_04_SEMANTIC_PLUS_TTT_CONFLICT \
    TTTSEM_05_SEMANTIC_PLUS_DG_PLUS_CONFLICT \
    TTTSEM_06_ROLE_SPECIFIC_BRANCH_W0 \
    TTTSEM_07_ROLE_SPECIFIC_LONG_SHORT \
    CHUNKSEM_01_STRUCTURE_KEEP \
    CHUNKSEM_02_LOWSTUFF_HIGHD_SKIP \
    CHUNKSEM_03_PROTECT_SPECIAL_TOKENS
}

case "$PHASE" in
  phase0)
    export RUN_PREFIX="${RUN_PREFIX:-V24_P0_SMOKE_R1}"
    run_one K1_H9 10 3
    run_one P0_01_SEMANTIC_ROLE_NOOP_IGNORED 10 3
    run_one P0_02_SEMANTIC_ROLE_PASS_THROUGH_CONSUMED 10 3
    run_one P0_03_SEMANTIC_ROLE_DEBUG_ONLY_ALL_MEMORY 10 3
    run_one FRAMESEM_02_LOWSTUFF_HIGHD_SKIP 10 3
    run_one ALLMEM_03_FRAME_GLOBAL_SWA_TTT_PATHSPEC 10 3
    ;;
  phase1)
    export RUN_PREFIX="${RUN_PREFIX:-V24_P1_PASSIVE_R1}"
    for chunk in 6 10 16; do
      run_one K1_H9 "$chunk" 10
      run_one PASSIVE_DEBUG_ONLY "$chunk" 10
      run_one K1_H9 "$chunk" 15
      run_one PASSIVE_DEBUG_ONLY "$chunk" 15
    done
    ;;
  phase2_initial)
    export RUN_PREFIX="${RUN_PREFIX:-V24_P2_INITIAL_R1}"
    phase2_candidates | while read -r candidate; do
      run_one "$candidate" 10 10
      run_one "$candidate" 10 15
    done
    ;;
  phase2_expand)
    if [ -z "${V24_PHASE2_EXPAND_CANDIDATES:-}" ]; then
      echo "Set V24_PHASE2_EXPAND_CANDIDATES to a comma-separated candidate list." >&2
      exit 2
    fi
    export RUN_PREFIX="${RUN_PREFIX:-V24_P2_EXPAND_R1}"
    IFS=',' read -r -a candidates <<< "$V24_PHASE2_EXPAND_CANDIDATES"
    for candidate in "${candidates[@]}"; do
      for chunk in 6 16; do
        run_one "$candidate" "$chunk" 10
        run_one "$candidate" "$chunk" 15
      done
    done
    ;;
  phase3)
    export RUN_PREFIX="${RUN_PREFIX:-V24_P3_PAIRWISE_R1}"
    for candidate in \
      PAIR_FRAME_TTT_PATHSPEC \
      PAIR_FRAME_SWA_PATHSPEC \
      PAIR_SWA_TTT_PATHSPEC \
      PAIR_GLOBAL_TTT_PATHSPEC \
      PAIR_FRAME_GLOBAL_PATHSPEC \
      PAIR_FRAME_GLOBAL_SWA_TTT_PATHSPEC
    do
      run_one "$candidate" 10 10
      run_one "$candidate" 10 15
    done
    ;;
  phase4)
    export RUN_PREFIX="${RUN_PREFIX:-V24_P4_ALLMEM_R1}"
    for candidate in \
      ALLMEM_01_FRAME_TTT_PATHSPEC \
      ALLMEM_02_FRAME_SWA_TTT_PATHSPEC \
      ALLMEM_03_FRAME_GLOBAL_SWA_TTT_PATHSPEC \
      ALLMEM_04_SKY_NEUTRAL_STRUCTURE_LONG \
      ALLMEM_05_LOWSTUFF_HIGHD_SHORTNEG \
      ALLMEM_06_CONFLICT_GATED_SEMANTIC
    do
      run_one "$candidate" 10 10
      run_one "$candidate" 10 15
    done
    ;;
  phase5_attr)
    if [ -z "${V24_PHASE5_CANDIDATE:-}" ]; then
      echo "Set V24_PHASE5_CANDIDATE before running phase5_attr." >&2
      exit 2
    fi
    export RUN_PREFIX="${RUN_PREFIX:-V24_P5_ATTR_R1}"
    export V24_SAVE_ATTRIBUTION_STATES=1
    run_one "$V24_PHASE5_CANDIDATE" "${V24_PHASE5_CHUNK:-10}" 15
    ;;
  *)
    echo "Unsupported PHASE: $PHASE" >&2
    exit 2
    ;;
esac
