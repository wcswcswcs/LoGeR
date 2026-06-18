#!/usr/bin/env bash
set -euo pipefail

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
RESULT_ROOT="${V64_RESULT_ROOT:-results/kitti01_hmc_v2/acl2_v64_ttt_scale_mechanism_attribution}"
BASE="$ROOT/$RESULT_ROOT/phase3_within_reset_ttt_causal_fork/rollouts"
PLAN="$ROOT/docs/ACL2_v64_TTT_Scale_Mechanism_Attribution_Plan.md"

CHUNK="${V64_PHASE3_CHUNK:-6}"
END_FRAME="${V64_PHASE3_END_FRAME:-360}"

mkdir -p "$BASE"

run_one() {
  local gpu="$1"
  local fork="$2"
  local run="$3"
  local extra_env="$4"
  (
    cd "$ROOT"
    eval "$extra_env" \
    V47_RESULT_ROOT="$RESULT_ROOT" \
    V47_ROLLOUT_BASE="$BASE" \
    V47_PLAN_NOTE="$PLAN" \
    V47_END_FRAME="$END_FRAME" \
    V47_DG_CUE="acl2.gg.qq.low.g2_3.past_only.headmean.robustq" \
    V47_READ_BETA=4.75 \
    V47_TTT_WRITE_SCORE="stage_d_x_dg_inv_sqrt" \
    V47_TTT_RISK_SOURCE="ttt_residual_x_dg" \
    V47_TTT_ROLE_MODE="adaptive_writer_sc_gamma_split" \
    V47_TTT_BRANCH_MASK=0 \
    V47_TTT_LAYER_GAMMAS="0:0.0075,8:0.0075,17:0.0075" \
    V47_NATIVE_MIX_SCALES="1.00,1.00,1.00" \
    V47_EMPTY_CUDA_CACHE_EACH_CHUNK=0 \
    PER_CHUNK_POSE_TRACE_JSONL="$BASE/$run/per_chunk_pose_trace.jsonl" \
    tools/run_v47_adaptive_ttt_writer_candidate.sh "$gpu" AW110_FRAME_ADAPTIVE_TTT "$run"
    printf '%s,%s,%s,%s\n' "$fork" "$gpu" "$run" "$extra_env" >> "$BASE/chunk${CHUNK}_hook_validation_completed.csv"
  )
}

rm -f "$BASE/chunk${CHUNK}_hook_validation_completed.csv"
printf 'fork,gpu,run_name,extra_env\n' > "$BASE/chunk${CHUNK}_hook_validation_launched.csv"

declare -a JOBS=(
  "2 F0_BASE V64_P3_C${CHUNK}_F0_BASE_E${END_FRAME} ''"
  "3 F1_NO_TTT_WRITE V64_P3_C${CHUNK}_F1_FREEZE_E${END_FRAME} 'TTT_FREEZE_CHUNKS=${CHUNK}'"
  "4 F2_NATIVE_ZERO_DELTA V64_P3_C${CHUNK}_F2_NATIVE0_E${END_FRAME} 'TTT_SEMANTIC_WRITE_SCALE_CHUNKS=${CHUNK}:0.0'"
  "5 F4_HALF_DELTA V64_P3_C${CHUNK}_F4_HALF_E${END_FRAME} 'TTT_SEMANTIC_WRITE_SCALE_CHUNKS=${CHUNK}:0.5'"
  "6 F5_DOUBLE_DELTA V64_P3_C${CHUNK}_F5_DOUBLE_E${END_FRAME} 'TTT_SEMANTIC_WRITE_SCALE_CHUNKS=${CHUNK}:2.0'"
  "7 F6_NEGATE_DELTA V64_P3_C${CHUNK}_F6_NEGATE_E${END_FRAME} 'TTT_SEMANTIC_WRITE_SCALE_CHUNKS=${CHUNK}:-1.0'"
)

pids=()
for item in "${JOBS[@]}"; do
  read -r gpu fork run extra <<<"$item"
  extra="${extra#\'}"
  extra="${extra%\'}"
  printf '%s,%s,%s,%s\n' "$fork" "$gpu" "$run" "$extra" >> "$BASE/chunk${CHUNK}_hook_validation_launched.csv"
  run_one "$gpu" "$fork" "$run" "$extra" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
