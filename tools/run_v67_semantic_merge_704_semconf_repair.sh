#!/usr/bin/env bash
set -euo pipefail

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
PHASE_REL="${V67_SEMANTIC_MERGE_PHASE_REL:-results/kitti01_hmc_v2/acl2_v67_dense_semantic_reconstruction/phase4_semantic_merge_704_semconf_fix}"
BASE="$ROOT/$PHASE_REL/rollouts"
LOG_DIR="$BASE/_launcher_logs"
mkdir -p "$LOG_DIR"
FRAME_TAG="${V67_SEMANTIC_MERGE_FRAME_TAG:-704}"
RUN_SUFFIX="${V67_SEMANTIC_MERGE_RUN_SUFFIX:-FIX}"
MERGE_SEED="${V67_SEMANTIC_MERGE_RANDOM_SEED:-123}"
MERGE_MAX_POINTS="${V67_SEMANTIC_MERGE_MAX_POINTS:-12000}"
JOB_SET="${V67_SEMANTIC_MERGE_JOB_SET:-all}"
SAVE_GEOMETRY="${V67_SEMANTIC_MERGE_SAVE_GEOMETRY:-0}"
GPU_SEM="${V67_SEMANTIC_MERGE_GPU_SEM:-0}"
GPU_RANDOM="${V67_SEMANTIC_MERGE_GPU_RANDOM:-1}"
GPU_SHUFFLED="${V67_SEMANTIC_MERGE_GPU_SHUFFLED:-2}"

export LOGER_ROOT="$ROOT"
export LOGER_PY="${LOGER_PY:-/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python}"
export V47_RESULT_ROOT="$ROOT/$PHASE_REL"
export V47_ROLLOUT_BASE="$BASE"
export V47_PLAN_NOTE="$ROOT/docs/ACL2_v67_DenseSemantic_Reconstruction_Emergency_Detailed_Plan_semgeo_ttt_revision.md"
export V47_END_FRAME="${V67_SEMANTIC_MERGE_END_FRAME:-704}"
export V47_DG_CUE=acl2.gg.qq.low.g2_3.past_only.headmean.robustq
export V47_READ_BETA=4.75
export V47_TTT_WRITE_SCORE=stage_d_x_dg_inv_sqrt
export V47_TTT_RISK_SOURCE=ttt_residual_x_dg
export V47_TTT_ROLE_MODE=adaptive_writer_sc_gamma_split
export V47_TTT_BRANCH_MASK=0
export V47_TTT_LAYER_GAMMAS=0:0.0075,8:0.0075,17:0.0075
export V47_NATIVE_MIX_SCALES=1.00,1.00,1.00
export V47_TTT_COMMIT_FILTER_MODE=none
export V47_TTT_COMMIT_FILTER_RISK_SOURCE=d_tok
export V47_TTT_COMMIT_FILTER_SCOPE=tail_overlap
export V47_TTT_COMMIT_FILTER_STAT=mean
export V47_TTT_COMMIT_FILTER_BASE=0.0
export V47_TTT_COMMIT_FILTER_GAIN=1.0
export V47_TTT_COMMIT_FILTER_MIN=0.0
export V47_TTT_COMMIT_FILTER_MAX=1.0
export V47_TTT_COMMIT_FILTER_BRANCH_MASK=0
export V47_TTT_SCALE_STATE_MODE=none
export V47_TTT_SCALE_STATE_PROXY=pose_step_ema
export V47_TTT_SCALE_STATE_CARRIER=all
export V47_TTT_SCALE_STATE_ALPHA=0.0
export V47_TTT_SCALE_STATE_BRANCH_MASK=0
export V47_TTT_SCALE_STATE_SAMPLE_TOKENS=0
export V47_ONLINE_SCALE_STATE_MODE=none
export V47_ONLINE_SCALE_STATE_MIN=0.80
export V47_ONLINE_SCALE_STATE_MAX=1.25
export V47_EMPTY_CUDA_CACHE_EACH_CHUNK=0
export TTT_WRITE_POST_ZP_SUMMARY=1

case "$JOB_SET" in
  all)
    JOBS=(
      "${GPU_SEM}|V67P4_${FRAME_TAG}_MERGE_S11_SEMCONF_${RUN_SUFFIX}|S11_SEMANTIC_GEOMETRY_WEIGHTED"
      "${GPU_RANDOM}|V67P4_${FRAME_TAG}_MERGE_S11_SEMCONF_RANDOM_${RUN_SUFFIX}|S11_SEMANTIC_GEOMETRY_WEIGHTED_RANDOM"
      "${GPU_SHUFFLED}|V67P4_${FRAME_TAG}_MERGE_S11_SEMCONF_SHUFFLED_${RUN_SUFFIX}|S11_SEMANTIC_GEOMETRY_WEIGHTED_SHUFFLED"
    )
    ;;
  sem_only)
    JOBS=(
      "${GPU_SEM}|V67P4_${FRAME_TAG}_MERGE_S11_SEMCONF_${RUN_SUFFIX}|S11_SEMANTIC_GEOMETRY_WEIGHTED"
    )
    ;;
  *)
    echo "Unsupported V67_SEMANTIC_MERGE_JOB_SET=$JOB_SET (expected all or sem_only)" >&2
    exit 2
    ;;
esac

for job in "${JOBS[@]}"; do
  IFS="|" read -r _gpu run_name _strategy <<<"$job"
  out_dir="$BASE/$run_name"
  if [ -d "$out_dir" ] && [ "${ALLOW_EXISTING:-0}" != "1" ]; then
    echo "Refusing to overwrite existing run dir: $out_dir" >&2
    echo "Set ALLOW_EXISTING=1 only if you intentionally want to rerun in place." >&2
    exit 3
  fi
done

pids=()
names=()

launch_one() {
  local gpu="$1"
  local run_name="$2"
  local strategy="$3"
  local out_dir="$BASE/$run_name"
  (
    export SEMANTIC_MERGE_MODE=current_world_to_aligned_previous
    export SEMANTIC_MERGE_STRATEGY="$strategy"
    export SEMANTIC_MERGE_MAX_POINTS="$MERGE_MAX_POINTS"
    export SEMANTIC_MERGE_RANDOM_SEED="$MERGE_SEED"
    export SEMANTIC_MERGE_CONF_MIN=0.05
    export STAGE_C_CACHE_DIR=results/kitti_preprocess/01/stage_c_cache_semantic_chunks
    export EXTRA_RUN_ARGS="--semantic_merge_use_semantic_confidence 1 --semantic_merge_semantic_conf_min 0.05 ${V67_SEMANTIC_MERGE_EXTRA_ARGS:-}"
    export SAVE_MERGE_STATES="$out_dir/merge_states"
    export SAVE_MERGE_STATE_KINDS=all
    export MERGE_STATE_TRACE_JSONL="$out_dir/merge_state_trace.jsonl"
    export PER_CHUNK_POSE_TRACE_JSONL="$out_dir/per_chunk_pose_trace.jsonl"
    if [ "$SAVE_GEOMETRY" = "1" ]; then
      export PER_CHUNK_GEOMETRY_DIR="$out_dir/per_chunk_geometry"
    fi
    export SAVE_PREMERGE_LOCAL_OUTPUT="$out_dir/premerge_local_pose.jsonl"
    export SAVE_POSTMERGE_GLOBAL_OUTPUT="$out_dir/postmerge_global_pose.jsonl"
    bash "$ROOT/tools/run_v47_adaptive_ttt_writer_candidate.sh" \
      "$gpu" AW110_FRAME_ADAPTIVE_TTT "$run_name"
  ) > "$LOG_DIR/$run_name.launch.log" 2>&1 &
  pids+=("$!")
  names+=("$run_name")
}

for job in "${JOBS[@]}"; do
  IFS="|" read -r gpu run_name strategy <<<"$job"
  launch_one "$gpu" "$run_name" "$strategy"
done

failed=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "FAILED ${names[$i]}" >&2
    failed=1
  fi
done

exit "$failed"
