#!/usr/bin/env bash
set -euo pipefail

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
RESULT_ROOT="${V24_RESULT_ROOT:-$ROOT/results/kitti01_hmc_v2/acl2_v24_semanticprior_pathspecific_allmemory_parallel}"
RUNNER="$ROOT/tools/run_v24_candidate_rollout.sh"
RUN_PREFIX="${RUN_PREFIX:-V24_P2_INITIAL_R1}"
LOG_DIR="$RESULT_ROOT/matrix_logs/${V24_PHASE2_LOG_DIR_NAME:-phase2_initial_R1}"
QUEUE_FILE="$LOG_DIR/queue.tsv"
GPUS_TEXT="${V24_GPUS:-0 1 2 3 4 5}"
HORIZONS_TEXT="${V24_PHASE2_HORIZONS:-10 15}"
CANDIDATES_TEXT="${V24_PHASE2_CANDIDATES:-}"

mkdir -p "$LOG_DIR"

phase2_candidates() {
  if [ -n "$CANDIDATES_TEXT" ]; then
    printf '%s\n' $CANDIDATES_TEXT
    return
  fi
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

row_done() {
  local candidate="$1"
  local horizon="$2"
  local log="$LOG_DIR/${candidate}_h${horizon}.log"
  local legacy
  shopt -s nullglob
  for legacy in "$LOG_DIR"/*_"${candidate}_h${horizon}.log"; do
    if rg -q '^DONE ' "$legacy"; then
      shopt -u nullglob
      return 0
    fi
  done
  shopt -u nullglob
  [ -f "$log" ] && rg -q '^DONE ' "$log"
}

: > "$QUEUE_FILE"
for candidate in $(phase2_candidates); do
  for horizon in $HORIZONS_TEXT; do
    if row_done "$candidate" "$horizon"; then
      continue
    fi
    printf '%s\t%s\n' "$candidate" "$horizon" >> "$QUEUE_FILE"
  done
done

total=$(wc -l < "$QUEUE_FILE" | tr -d ' ')
echo "[$(date -Is)] phase2_initial queue rows=$total run_prefix=$RUN_PREFIX"
if [ "$total" = "0" ]; then
  exit 0
fi

worker() {
  local worker_idx="$1"
  local gpu="$2"
  local candidate horizon log
  while IFS=$'\t' read -r candidate horizon; do
    [ -n "$candidate" ] || continue
    log="$LOG_DIR/${candidate}_h${horizon}.log"
    {
      start_epoch="$(date +%s)"
      echo "START $(date -Is) gpu=$gpu candidate=$candidate horizon=$horizon"
      RUN_PREFIX="$RUN_PREFIX" FORCE=1 OUTPUT_VIDEO= bash "$RUNNER" "$gpu" "$candidate" 10 "$horizon"
      end_epoch="$(date +%s)"
      echo "DONE $(date -Is) gpu=$gpu candidate=$candidate horizon=$horizon wall_seconds=$((end_epoch - start_epoch))"
    } > "$log" 2>&1
  done < <(awk -v worker="$worker_idx" -v nworkers="$NUM_WORKERS" '(NR - 1) % nworkers == worker {print}' "$QUEUE_FILE")
}

read -r -a GPUS <<< "$GPUS_TEXT"
NUM_WORKERS="${#GPUS[@]}"
pids=()
for idx in "${!GPUS[@]}"; do
  worker "$idx" "${GPUS[$idx]}" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "[$(date -Is)] phase2_initial queue complete"
