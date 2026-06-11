#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 GPU ROW RUN_NAME" >&2
  echo "Rows: E1_96F E2_96F E1_704F E2_704F E1_FULL E2_FULL" >&2
  exit 2
fi

GPU="$1"
ROW="$2"
RUN_NAME="$3"

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
RESULT_ROOT="${V55_RESULT_ROOT:-$ROOT/results/kitti01_hmc_v2/acl2_v55_c9schedule_autopsy_failforward_adaptivettt_clean}"
PLAN_NOTE="${V55_PLAN_NOTE:-$ROOT/docs/ACL2_v55_C9ScheduleAutopsy_FailForward_AdaptiveTTT_CleanPlan.md}"

END_FRAME="${V55_END_FRAME:-96}"
PHASE="phase3_smoke"
ROLE_MODE="adaptive_writer_tail_state_continuity_guard"
COMMIT_FILTER_MODE="none"
COMMIT_FILTER_MIN="${V55_COMMIT_ALPHA_MIN:-0.35}"
COMMIT_FILTER_MAX="1.0"

case "$ROW" in
  E1_96F)
    END_FRAME="${V55_END_FRAME:-96}"
    PHASE="${V55_PHASE:-phase3_smoke}"
    ROLE_MODE="adaptive_writer_tail_state_continuity_guard"
    ;;
  E2_96F)
    END_FRAME="${V55_END_FRAME:-96}"
    PHASE="${V55_PHASE:-phase3_smoke}"
    ROLE_MODE="adaptive_writer_tail_state_continuity_guard_selective_commit"
    COMMIT_FILTER_MODE="tail_state_selective_commit"
    ;;
  E1_704F)
    END_FRAME="${V55_END_FRAME:-704}"
    PHASE="${V55_PHASE:-phase3_704_screen}"
    ROLE_MODE="adaptive_writer_tail_state_continuity_guard"
    ;;
  E2_704F)
    END_FRAME="${V55_END_FRAME:-704}"
    PHASE="${V55_PHASE:-phase3_704_screen}"
    ROLE_MODE="adaptive_writer_tail_state_continuity_guard_selective_commit"
    COMMIT_FILTER_MODE="tail_state_selective_commit"
    ;;
  E1_FULL)
    END_FRAME="${V55_END_FRAME:-10000}"
    PHASE="${V55_PHASE:-phase4_full}"
    ROLE_MODE="adaptive_writer_tail_state_continuity_guard"
    ;;
  E2_FULL)
    END_FRAME="${V55_END_FRAME:-10000}"
    PHASE="${V55_PHASE:-phase4_full}"
    ROLE_MODE="adaptive_writer_tail_state_continuity_guard_selective_commit"
    COMMIT_FILTER_MODE="tail_state_selective_commit"
    ;;
  *)
    echo "Unsupported v55 row: $ROW" >&2
    exit 2
    ;;
esac

BASE="${V55_ROLLOUT_BASE:-$RESULT_ROOT/$PHASE/rollouts}"
LAYER_GAMMAS="${V55_TTT_LAYER_GAMMAS:-0:0.0075,8:0.0075,17:0.0075}"

echo "[v55] row=$ROW run=$RUN_NAME gpu=$GPU end_frame=$END_FRAME phase=$PHASE"
echo "[v55] role=$ROLE_MODE risk=ttt_residual_x_dg layer_gammas=$LAYER_GAMMAS commit=$COMMIT_FILTER_MODE"

V47_RESULT_ROOT="$RESULT_ROOT" \
V47_ROLLOUT_BASE="$BASE" \
V47_PLAN_NOTE="$PLAN_NOTE" \
V47_END_FRAME="$END_FRAME" \
V47_DG_CUE="${V55_DG_CUE:-acl2.gg.qq.low.g2_3.past_only.headmean.robustq}" \
V47_READ_BETA="${V55_READ_BETA:-4.75}" \
V47_TTT_WRITE_SCORE="${V55_TTT_WRITE_SCORE:-stage_d_x_dg_inv_sqrt}" \
V47_TTT_RISK_SOURCE="${V55_TTT_RISK_SOURCE:-ttt_residual_x_dg}" \
V47_TTT_ROLE_MODE="$ROLE_MODE" \
V47_TTT_BRANCH_MASK="${V55_TTT_BRANCH_MASK:-0}" \
V47_TTT_BRANCH_GAMMAS="${V55_TTT_BRANCH_GAMMAS:-}" \
V47_TTT_LAYER_GAMMAS="$LAYER_GAMMAS" \
V47_NATIVE_MIX_SCALES="${V55_NATIVE_MIX_SCALES:-1.00,1.00,1.00}" \
V47_TTT_COMMIT_FILTER_MODE="$COMMIT_FILTER_MODE" \
V47_TTT_COMMIT_FILTER_RISK_SOURCE="${V55_TTT_COMMIT_FILTER_RISK_SOURCE:-ttt_residual_x_dg}" \
V47_TTT_COMMIT_FILTER_MIN="$COMMIT_FILTER_MIN" \
V47_TTT_COMMIT_FILTER_MAX="$COMMIT_FILTER_MAX" \
V47_TTT_COMMIT_FILTER_BRANCH_MASK="${V55_TTT_COMMIT_FILTER_BRANCH_MASK:-0}" \
V47_TTT_COMMIT_FILTER_SCOPE="${V55_TTT_COMMIT_FILTER_SCOPE:-tail_overlap}" \
V47_TTT_COMMIT_FILTER_STAT="${V55_TTT_COMMIT_FILTER_STAT:-mean}" \
V47_EMPTY_CUDA_CACHE_EACH_CHUNK="${V55_EMPTY_CUDA_CACHE_EACH_CHUNK:-0}" \
"$ROOT/tools/run_v47_adaptive_ttt_writer_candidate.sh" "$GPU" "AW110_FRAME_ADAPTIVE_TTT" "$RUN_NAME"
