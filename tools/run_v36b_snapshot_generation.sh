#!/usr/bin/env bash
set -euo pipefail

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
RESULT_ROOT="${V36B_ROOT:-results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30}"
ROLL_ROOT="$RESULT_ROOT/phase0_parent_snapshots/rollouts"
SNAP_ROOT="$RESULT_ROOT/phase0_parent_snapshots"
CHUNKS="${V36B_PARENT_CHUNKS:-6,10,16}"

mkdir -p "$ROOT/$SNAP_ROOT/state_snapshots/H9_V36B_R1" \
         "$ROOT/$SNAP_ROOT/state_snapshots/C9_V36B_R1" \
         "$ROOT/$SNAP_ROOT/merge_state_snapshots/H9_V36B_R1" \
         "$ROOT/$SNAP_ROOT/merge_state_snapshots/C9_V36B_R1" \
         "$ROOT/$RESULT_ROOT/matrix_logs/phase0_parent_snapshots_R1"

run_parent() {
  local gpu="$1"
  local parent="$2"
  local alpha="$3"
  local run_name="V36B_PHASE0_${parent}_SNAPSHOT_R1"
  local log="$ROOT/$RESULT_ROOT/matrix_logs/phase0_parent_snapshots_R1/${run_name}.log"
  echo "[$(date '+%F %T')] START $run_name gpu=$gpu alpha=$alpha chunks=$CHUNKS" | tee "$log"
  (
    cd "$ROOT"
    env \
      KITTI_SEQ=01 \
      ATTN_CUE_BASE="$ROLL_ROOT" \
      START_FRAME=0 \
      END_FRAME=10000 \
      RESET_EVERY=5 \
      READ_PATH=frame \
      READ_LAYER_MODE=all \
      READ_BETA_FRAME_CHUNKS="5:4.85,6:4.85,7:4.85,8:4.85,9:4.85,10:4.25,11:4.25,12:4.25,16:4.25" \
      WRITE_ALPHA="$alpha" \
      WRITE_MIN=0.8 \
      WRITE_MAX=1.2 \
      MP_SCORE_SOURCE=dyn \
      STAGE_C_MODE=reference \
      STAGE_C_CACHE_DIR="results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full" \
      STAGE_C_CACHE_MODE=read \
      STAGE_C_CACHE_REQUIRE_HIT=1 \
      STAGE_C_CACHE_VALIDATE=0 \
      SEMANTIC_PRIOR_MODE=spg_v2 \
      ENABLE_SWA_WRITE_CONTROL=1 \
      SWA_WRITE_MODE=kv \
      SWA_WRITE_RHO=0.65 \
      SWA_WRITE_MIN_GATE=0.20 \
      SWA_WRITE_SCOPE=both_overlap \
      SWA_WRITE_KEEP_SCOPE=all \
      SWA_WRITE_SCORE_SOURCE=read \
      SWA_WRITE_LAYER_MODE=last \
      ENABLE_SWA_OVERLAP_SOURCE_REPLACE=1 \
      SWA_OVERLAP_SOURCE_REPLACE_ALPHA=0.5 \
      SWA_OVERLAP_SOURCE_REPLACE_MODE=source \
      SWA_OVERLAP_SOURCE_REPLACE_TARGET=kv \
      SWA_OVERLAP_SOURCE_REPLACE_LAYER_MODE=last \
      TTT_WRITE_GRADIENT_REVERSAL_MODE=tri_replay \
      TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE=update_conflict_energy \
      TTT_WRITE_GRADIENT_REVERSAL_BRANCH_MASK=0 \
      TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS="5:0.005,6:0.005,7:0.005,8:0.005,9:0.005,10:0.003,11:0.003,12:0.003,16:0.0003" \
      TTT_WRITE_TRI_REPLAY_POSITIVE_FRAC=0.35 \
      TTT_WRITE_TRI_REPLAY_NEGATIVE_FRAC=0.12 \
      TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA=0.85 \
      TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS="5:0.35/0.12/0.85,6:0.35/0.12/0.85,7:0.35/0.12/0.85,8:0.35/0.12/0.85,9:0.35/0.12/0.85,10:0.35/0.12/0.85,11:0.35/0.12/0.85,12:0.35/0.12/0.85,16:0.35/0.08/0.85" \
      SAVE_HMC_STATES="$SNAP_ROOT/state_snapshots/${parent}_V36B_R1" \
      SAVE_HMC_STATE_CHUNKS="$CHUNKS" \
      SAVE_HMC_STATE_KINDS=input \
      SAVE_MERGE_STATES="$SNAP_ROOT/merge_state_snapshots/${parent}_V36B_R1" \
      SAVE_MERGE_STATE_CHUNKS="$CHUNKS" \
      SAVE_MERGE_STATE_KINDS=input \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$ROOT/tools/run_attention_cue_experiment.sh" \
        "$gpu" "$run_name" hybrid \
        "acl2.gg.qq.low.g2_3.past_only.headmean.robustq" \
        4.75 stage_d_x_dg_inv_sqrt
  ) 2>&1 | tee -a "$log"
  echo "[$(date '+%F %T')] END $run_name" | tee -a "$log"
}

run_parent "${H9_GPU:-0}" H9 0.125 &
p1=$!
run_parent "${C9_GPU:-1}" C9 0.1 &
p2=$!
wait "$p1"
wait "$p2"
