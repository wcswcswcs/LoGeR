#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 GPU ROW RUN_NAME" >&2
  echo "Rows: M1_704F M2_704F M1_FULL M2_FULL H35_704F_REF" >&2
  exit 2
fi

GPU="$1"
ROW="$2"
RUN_NAME="$3"

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
RESULT_ROOT="${V54_RESULT_ROOT:-$ROOT/results/kitti01_hmc_v2/acl2_v54_fast_state_conditioned_adaptive_ttt_clean_to_c9}"
PLAN_NOTE="${V54_PLAN_NOTE:-$ROOT/docs/ACL2_v54_Fast_StateConditioned_AdaptiveTTT_Clean_to_C9_Plan.md}"

END_FRAME="${V54_END_FRAME:-704}"
PHASE="phase3_704_screen"
ROLE_MODE="adaptive_writer_state_energy_matched_split"
COMMIT_FILTER_MODE="none"
COMMIT_FILTER_MIN="${V54_COMMIT_ALPHA_MIN:-0.25}"
COMMIT_FILTER_MAX="1.0"

case "$ROW" in
  M1_704F)
    END_FRAME="${V54_END_FRAME:-704}"
    PHASE="${V54_PHASE:-phase3_704_screen}"
    ROLE_MODE="adaptive_writer_state_energy_matched_split"
    ;;
  M2_704F)
    END_FRAME="${V54_END_FRAME:-704}"
    PHASE="${V54_PHASE:-phase3_704_screen}"
    ROLE_MODE="adaptive_writer_state_energy_directional_commit_split"
    COMMIT_FILTER_MODE="state_energy_directional_commit"
    ;;
  M1_FULL)
    END_FRAME="${V54_END_FRAME:-10000}"
    PHASE="${V54_PHASE:-phase4_full}"
    ROLE_MODE="adaptive_writer_state_energy_matched_split"
    ;;
  M2_FULL)
    END_FRAME="${V54_END_FRAME:-10000}"
    PHASE="${V54_PHASE:-phase4_full}"
    ROLE_MODE="adaptive_writer_state_energy_directional_commit_split"
    COMMIT_FILTER_MODE="state_energy_directional_commit"
    ;;
  H35_704F_REF)
    END_FRAME="${V54_END_FRAME:-704}"
    PHASE="${V54_PHASE:-phase3_704_screen}"
    ROLE_MODE="adaptive_writer_sc_gamma_split"
    ;;
  *)
    echo "Unsupported v54 row: $ROW" >&2
    exit 2
    ;;
esac

BASE="${V54_ROLLOUT_BASE:-$RESULT_ROOT/$PHASE/rollouts}"
LAYER_GAMMAS="${V54_TTT_LAYER_GAMMAS:-0:0.0075,8:0.0075,17:0.0075}"

echo "[v54] row=$ROW run=$RUN_NAME gpu=$GPU end_frame=$END_FRAME phase=$PHASE"
echo "[v54] role=$ROLE_MODE risk=ttt_residual_x_dg layer_gammas=$LAYER_GAMMAS commit=$COMMIT_FILTER_MODE"

V47_RESULT_ROOT="$RESULT_ROOT" \
V47_ROLLOUT_BASE="$BASE" \
V47_PLAN_NOTE="$PLAN_NOTE" \
V47_END_FRAME="$END_FRAME" \
V47_DG_CUE="${V54_DG_CUE:-acl2.gg.qq.low.g2_3.past_only.headmean.robustq}" \
V47_READ_BETA="${V54_READ_BETA:-4.75}" \
V47_TTT_WRITE_SCORE="${V54_TTT_WRITE_SCORE:-stage_d_x_dg_inv_sqrt}" \
V47_TTT_RISK_SOURCE="${V54_TTT_RISK_SOURCE:-ttt_residual_x_dg}" \
V47_TTT_ROLE_MODE="$ROLE_MODE" \
V47_TTT_BRANCH_MASK="${V54_TTT_BRANCH_MASK:-0}" \
V47_TTT_BRANCH_GAMMAS="${V54_TTT_BRANCH_GAMMAS:-}" \
V47_TTT_LAYER_GAMMAS="$LAYER_GAMMAS" \
V47_NATIVE_MIX_SCALES="${V54_NATIVE_MIX_SCALES:-1.00,1.00,1.00}" \
V47_TTT_COMMIT_FILTER_MODE="$COMMIT_FILTER_MODE" \
V47_TTT_COMMIT_FILTER_MIN="$COMMIT_FILTER_MIN" \
V47_TTT_COMMIT_FILTER_MAX="$COMMIT_FILTER_MAX" \
V47_TTT_COMMIT_FILTER_BRANCH_MASK="${V54_TTT_COMMIT_FILTER_BRANCH_MASK:-0}" \
V47_TTT_COMMIT_FILTER_SCOPE="${V54_TTT_COMMIT_FILTER_SCOPE:-tail_overlap}" \
V47_TTT_COMMIT_FILTER_STAT="${V54_TTT_COMMIT_FILTER_STAT:-mean}" \
V47_EMPTY_CUDA_CACHE_EACH_CHUNK="${V54_EMPTY_CUDA_CACHE_EACH_CHUNK:-0}" \
"$ROOT/tools/run_v47_adaptive_ttt_writer_candidate.sh" "$GPU" "AW110_FRAME_ADAPTIVE_TTT" "$RUN_NAME"
