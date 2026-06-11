#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 GPU CANDIDATE_ID CHUNK_ID HORIZON" >&2
  exit 2
fi

GPU="$1"
CANDIDATE_ID="$2"
CHUNK_ID="$3"
HORIZON="$4"

ROOT="${LOGER_ROOT:-/mnt/data/users/chengshun.wang/pjs/LoGeR}"
V16_ROOT="${V16_ROOT:-results/kitti01_hmc_v2/acl2_v16_ttt_causalfork_candidatebank_target25}"
V24_ROOT="${V24_ROOT:-results/kitti01_hmc_v2/acl2_v24_semanticprior_pathspecific_allmemory_parallel}"
PHASE1="$V16_ROOT/phase1_causalfork"
ROLLOUT_BASE="$V24_ROOT/rollouts"
STAGE_C_CACHE_DEFAULT="results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full"
WARM_ROOT="${LOGER_WARM_ROOT:-/tmp/loger_v23_warm}"
PROBE_CACHE_ROOT="${V24_PROBE_CACHE_ROOT:-$V24_ROOT/probe_cache}"

case "$HORIZON" in
  3|5|8|10|15) ;;
  *) echo "Unsupported HORIZON: $HORIZON" >&2; exit 2 ;;
esac

case "$CHUNK_ID" in
  5) START_FRAME=145; SNAP="005" ;;
  6) START_FRAME=174; SNAP="006" ;;
  9) START_FRAME=261; SNAP="009" ;;
  10) START_FRAME=290; SNAP="010" ;;
  12) START_FRAME=348; SNAP="012" ;;
  16) START_FRAME=464; SNAP="016" ;;
  *) echo "Unsupported CHUNK_ID: $CHUNK_ID" >&2; exit 2 ;;
esac

END_FRAME=$((START_FRAME + 32 + HORIZON * 29))
V26_BASE_GAMMAS="5:0.005,6:0.005,7:0.005,8:0.005,9:0.005,10:0.003,11:0.003,12:0.003,13:0.003,14:0.003,15:0.003,16:0.0003"
V26_BASE_TRI_PARAMS="5:0.35/0.12/0.85,6:0.35/0.12/0.85,7:0.35/0.12/0.85,8:0.35/0.12/0.85,9:0.35/0.12/0.85,10:0.35/0.12/0.85,11:0.35/0.12/0.85,12:0.35/0.12/0.85,13:0.35/0.12/0.85,14:0.35/0.12/0.85,15:0.35/0.12/0.85,16:0.35/0.08/0.85"
READ_CUE="${READ_CUE:-acl2.gg.qq.low.g2_3.past_only.headmean.robustq}"
READ_PATH_VALUE="${READ_PATH:-frame}"
BETA_VALUE="${BETA_VALUE:-4.75}"
WRITE_SCORE_VALUE="${WRITE_SCORE_VALUE:-stage_d_x_dg_inv_sqrt}"
RUN_MODE="${RUN_MODE:-hybrid}"
RUN_PREFIX="${RUN_PREFIX:-V24_P0_SMOKE_R1}"

STAGE_C_MODE_VALUE="${STAGE_C_MODE:-reference}"
STAGE_C_CACHE_DIR_VALUE="${STAGE_C_CACHE_DIR:-$STAGE_C_CACHE_DEFAULT}"
STAGE_C_CACHE_MODE_VALUE="${STAGE_C_CACHE_MODE:-read}"
STAGE_C_CACHE_REQUIRE_HIT_VALUE="${STAGE_C_CACHE_REQUIRE_HIT:-1}"
STAGE_C_CACHE_VALIDATE_VALUE="${STAGE_C_CACHE_VALIDATE:-0}"
SEMANTIC_PRIOR_MODE_VALUE="${SEMANTIC_PRIOR_MODE:-spg_v2}"

CONTEXT_SOURCE_SKIP_ENABLE=0
CONTEXT_SOURCE_SKIP_IMPL="bias"
CONTEXT_SOURCE_SKIP_SCOPE="frame"
CONTEXT_SOURCE_SKIP_MODE="hard"
CONTEXT_SOURCE_SKIP_MASK="dg_q90"
CONTEXT_SOURCE_SKIP_LAYER_MODE="early"
CONTEXT_SOURCE_SKIP_SOFT_RHO="${CONTEXT_SOURCE_SKIP_SOFT_RHO:-0.5}"
CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP="${CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP:-0.5}"

SEMANTIC_ROLE_POLICY_VALUE="${SEMANTIC_ROLE_POLICY:-none}"
SEMANTIC_MEMORY_PATHS_VALUE="${SEMANTIC_MEMORY_PATHS:-}"
SEMANTIC_ROLE_HIGHD_QUANTILE_VALUE="${SEMANTIC_ROLE_HIGHD_QUANTILE:-0.80}"
SEMANTIC_ROLE_LOW_TRUST_VALUE="${SEMANTIC_ROLE_LOW_TRUST:-0.20}"
SEMANTIC_ROLE_POSITIVE_SCALE_VALUE="${SEMANTIC_ROLE_POSITIVE_SCALE:-1.05}"
SEMANTIC_ROLE_NEUTRAL_SCALE_VALUE="${SEMANTIC_ROLE_NEUTRAL_SCALE:-0.85}"
SEMANTIC_ROLE_NEGATIVE_SCALE_VALUE="${SEMANTIC_ROLE_NEGATIVE_SCALE:-0.65}"
SEMANTIC_ROLE_SWA_NEGATIVE_SCALE_VALUE="${SEMANTIC_ROLE_SWA_NEGATIVE_SCALE:-1.0}"

USES_CONTEXT_SKIP=false
USES_TRUE_COMPACTION=false
USES_SEMANTIC_CACHE=true
V24_FAMILY="unassigned"
FINE_SPLIT_AVAILABLE=false
COARSE_FALLBACK=false

enable_compact_role_skip() {
  CONTEXT_SOURCE_SKIP_ENABLE=1
  CONTEXT_SOURCE_SKIP_IMPL="compact_kv"
  CONTEXT_SOURCE_SKIP_SCOPE="$1"
  CONTEXT_SOURCE_SKIP_MASK="semantic_role_negative"
  CONTEXT_SOURCE_SKIP_LAYER_MODE="${2:-early}"
  USES_CONTEXT_SKIP=true
  USES_TRUE_COMPACTION=true
}

enable_semantic_role() {
  SEMANTIC_ROLE_POLICY_VALUE="$1"
  SEMANTIC_MEMORY_PATHS_VALUE="$2"
}

enable_swa_policy() {
  export READ_PATH=swa
  READ_PATH_VALUE=swa
  export ENABLE_SWA_WRITE_CONTROL=1
  export SWA_WRITE_MODE="${SWA_WRITE_MODE:-kv}"
  export SWA_WRITE_RHO="${SWA_WRITE_RHO:-0.65}"
  export SWA_WRITE_MIN_GATE="${SWA_WRITE_MIN_GATE:-0.20}"
  export SWA_WRITE_SCOPE="${SWA_WRITE_SCOPE:-both_overlap}"
  export SWA_WRITE_KEEP_SCOPE="${SWA_WRITE_KEEP_SCOPE:-all}"
  export SWA_WRITE_SCORE_SOURCE="${SWA_WRITE_SCORE_SOURCE:-read}"
  export SWA_WRITE_LAYER_MODE="${SWA_WRITE_LAYER_MODE:-last}"
}

enable_v26_conflict_tri_replay() {
  export TTT_WRITE_GRADIENT_REVERSAL_MODE=tri_replay
  export TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE=update_conflict_energy
  export TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS="${TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS:-$V26_BASE_GAMMAS}"
  export TTT_WRITE_TRI_REPLAY_POSITIVE_FRAC="${TTT_WRITE_TRI_REPLAY_POSITIVE_FRAC:-0.35}"
  export TTT_WRITE_TRI_REPLAY_NEGATIVE_FRAC="${TTT_WRITE_TRI_REPLAY_NEGATIVE_FRAC:-0.12}"
  export TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA="${TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA:-0.85}"
  export TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS="${TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS:-$V26_BASE_TRI_PARAMS}"
}

enable_v26_scale_state_commit() {
  export TTT_WRITE_SCALE_STATE_MODE=projection_risk
  export TTT_WRITE_SCALE_STATE_PROXY=pose_step_ema
  export TTT_WRITE_SCALE_STATE_CARRIER=structure_lowdg
  export TTT_WRITE_SCALE_STATE_ALPHA="${TTT_WRITE_SCALE_STATE_ALPHA:-0.25}"
  export TTT_WRITE_SCALE_STATE_BRANCH_MASK="${TTT_WRITE_SCALE_STATE_BRANCH_MASK:-0}"
  export TTT_WRITE_SCALE_STATE_CHUNKS="${TTT_WRITE_SCALE_STATE_CHUNKS:-$CHUNK_ID-$((CHUNK_ID + HORIZON))}"
  export TTT_WRITE_NATIVE_DELTA_GATE_MODE=orthogonal_suppress
  export TTT_WRITE_NATIVE_DELTA_GATE_BRANCH_MASK=0
  export TTT_WRITE_GRADIENT_REVERSAL_MODE=tri_replay
  export TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE=v19_scale_state
  export TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS="${TTT_WRITE_GRADIENT_REVERSAL_CHUNK_GAMMAS:-$V26_BASE_GAMMAS}"
  export TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS="${TTT_WRITE_TRI_REPLAY_CHUNK_PARAMS:-$V26_BASE_TRI_PARAMS}"
}

enable_v26_dual_lifetime_short() {
  export TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_MODE=dual_lifetime
  export TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_BRANCH_MASK="${TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_BRANCH_MASK:-0}"
  export TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_LONG_SCALE="${TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_LONG_SCALE:-0.25}"
  export TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_APPLY_SCALE="${TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_APPLY_SCALE:-1.0}"
  export TTT_WRITE_TRANSIENT_DELTA_TTL="${TTT_WRITE_TRANSIENT_DELTA_TTL:-3}"
}

enable_v29c_masklet_intervention() {
  export V29C_MASKLET_ALIGNMENT_CSV="${V29C_MASKLET_ALIGNMENT_CSV:-$V24_ROOT/masklet_3d_alignment/masklet_alignment.csv}"
  export V29C_MASKLET_INTERVENTION_POLICY="${V29C_MASKLET_INTERVENTION_POLICY:-top_support_per_chunk}"
  export V29C_MASKLET_INTERVENTION_PATH="$1"
  export V29C_MASKLET_INTERVENTION_ACTION="$2"
  export V29C_MASKLET_INTERVENTION_PATCH_THRESHOLD="${V29C_MASKLET_INTERVENTION_PATCH_THRESHOLD:-0.20}"
  SEMANTIC_ROLE_POLICY_VALUE="v29c_masklet_override"
  SEMANTIC_MEMORY_PATHS_VALUE="$1"
  FINE_SPLIT_AVAILABLE=true
}

mark_fine_fallback() {
  COARSE_FALLBACK=true
  FINE_SPLIT_AVAILABLE=false
}

case "$CANDIDATE_ID" in
  V29C_BASE_H9_REFERENCE)
    RUN_MODE="readonly"
    SEMANTIC_PRIOR_MODE_VALUE="pass_through"
    enable_semantic_role noop ""
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v29c_causal_bank_reference"
    ;;
  V29C_CAUSAL_FRAME_SKIP_TOP)
    RUN_MODE="readonly"
    enable_v29c_masklet_intervention frame source_skip
    enable_compact_role_skip frame early
    V24_FAMILY="v29c_masklet_causal_frame"
    ;;
  V29C_CAUSAL_GLOBAL_SKIP_TOP)
    RUN_MODE="readonly"
    READ_PATH_VALUE="chunk"
    enable_v29c_masklet_intervention global source_skip
    enable_compact_role_skip chunk early
    V24_FAMILY="v29c_masklet_causal_global"
    ;;
  V29C_CAUSAL_SWA_ANCHOR_TOP)
    RUN_MODE="hybrid"
    export HMC_COMMIT_MODE=probe_native
    enable_v29c_masklet_intervention swa swa_anchor_keep
    enable_swa_policy
    V24_FAMILY="v29c_masklet_causal_swa"
    ;;
  V29C_CAUSAL_SWA_REMOVE_TOP)
    RUN_MODE="hybrid"
    export HMC_COMMIT_MODE=probe_native
    SEMANTIC_ROLE_SWA_NEGATIVE_SCALE_VALUE="${SEMANTIC_ROLE_SWA_NEGATIVE_SCALE:-0.65}"
    enable_v29c_masklet_intervention swa swa_remove
    enable_swa_policy
    V24_FAMILY="v29c_masklet_causal_swa"
    ;;
  V29C_CAUSAL_TTT_POS_TOP)
    RUN_MODE="hybrid"
    enable_v29c_masklet_intervention ttt ttt_positive
    V24_FAMILY="v29c_masklet_causal_ttt"
    ;;
  V29C_CAUSAL_TTT_NEG_TOP)
    RUN_MODE="hybrid"
    SEMANTIC_ROLE_NEGATIVE_SCALE_VALUE="${SEMANTIC_ROLE_NEGATIVE_SCALE:-0.35}"
    enable_v29c_masklet_intervention ttt ttt_negative
    V24_FAMILY="v29c_masklet_causal_ttt"
    ;;
  V30_BASE_H9_REFERENCE)
    RUN_MODE="readonly"
    SEMANTIC_PRIOR_MODE_VALUE="pass_through"
    enable_semantic_role noop ""
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v30_causal_bank_reference"
    ;;
  V30_MASKLET_FRAME_SKIP)
    RUN_MODE="readonly"
    enable_v29c_masklet_intervention frame source_skip
    enable_compact_role_skip frame early
    V24_FAMILY="v30_masklet_causal_frame"
    ;;
  V30_MASKLET_GLOBAL_SKIP)
    RUN_MODE="readonly"
    READ_PATH_VALUE="chunk"
    enable_v29c_masklet_intervention global source_skip
    enable_compact_role_skip chunk early
    V24_FAMILY="v30_masklet_causal_global"
    ;;
  V30_MASKLET_SWA_ANCHOR)
    RUN_MODE="hybrid"
    export HMC_COMMIT_MODE=probe_native
    enable_v29c_masklet_intervention swa swa_anchor_keep
    enable_swa_policy
    V24_FAMILY="v30_masklet_causal_swa"
    ;;
  V30_MASKLET_SWA_REMOVE)
    RUN_MODE="hybrid"
    export HMC_COMMIT_MODE=probe_native
    SEMANTIC_ROLE_SWA_NEGATIVE_SCALE_VALUE="${SEMANTIC_ROLE_SWA_NEGATIVE_SCALE:-1.0}"
    enable_v29c_masklet_intervention swa swa_remove
    enable_swa_policy
    V24_FAMILY="v30_masklet_causal_swa"
    ;;
  V30_MASKLET_TTT_POS)
    RUN_MODE="hybrid"
    enable_v29c_masklet_intervention ttt ttt_positive
    V24_FAMILY="v30_masklet_causal_ttt"
    ;;
  V30_MASKLET_TTT_NEG)
    RUN_MODE="hybrid"
    SEMANTIC_ROLE_NEGATIVE_SCALE_VALUE="${SEMANTIC_ROLE_NEGATIVE_SCALE:-0.35}"
    enable_v29c_masklet_intervention ttt ttt_negative
    V24_FAMILY="v30_masklet_causal_ttt"
    ;;
  V31_BASE_H9_REFERENCE)
    RUN_MODE="readonly"
    SEMANTIC_PRIOR_MODE_VALUE="pass_through"
    enable_semantic_role noop ""
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v31_h9_reference"
    ;;
  V31_A0_ORIG_C23)
    RUN_MODE="readonly"
    READ_PATH_VALUE="${READ_PATH_VALUE:-frame}"
    READ_CUE="acl2.gg.qq.low.g2_3.past_only.headmean.robustq"
    enable_semantic_role fine_path_router_debug "all"
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v31_track_a_original_c23"
    ;;
  V31_A1_SEM_Z_FINE)
    RUN_MODE="readonly"
    READ_PATH_VALUE="${READ_PATH_VALUE:-frame}"
    READ_CUE="v31.sem_z_fine.c23past"
    enable_semantic_role fine_path_router_debug "all"
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v31_track_a_semantic_z_fine"
    ;;
  V31_A1B_SEM_Z_COARSE)
    RUN_MODE="readonly"
    READ_PATH_VALUE="${READ_PATH_VALUE:-frame}"
    READ_CUE="v31.sem_z_coarse.c23past"
    enable_semantic_role fine_path_router_debug "all"
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v31_track_a_semantic_z_coarse"
    ;;
  V31_A5_SEM_RESID_FINE_L025)
    RUN_MODE="readonly"
    READ_PATH_VALUE="${READ_PATH_VALUE:-frame}"
    READ_CUE="v31.sem_resid_fine_l025.c23past"
    enable_semantic_role fine_path_router_debug "all"
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v31_track_a_semantic_residual_fine"
    ;;
  V31_A5B_SEM_RESID_COARSE_L025)
    RUN_MODE="readonly"
    READ_PATH_VALUE="${READ_PATH_VALUE:-frame}"
    READ_CUE="v31.sem_resid_coarse_l025.c23past"
    enable_semantic_role fine_path_router_debug "all"
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v31_track_a_semantic_residual_coarse"
    ;;
  V31_B0_STATIC_RESCUE_EXISTING)
    RUN_MODE="readonly"
    READ_PATH_VALUE="${READ_PATH_VALUE:-frame}"
    READ_CUE="mix.c23past_static_rescue_a025"
    enable_semantic_role fine_path_router_debug "all"
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v31_track_b_existing_static_rescue"
    ;;
  SEM_C23_01_READ_ONLY_RESID)
    RUN_MODE="readonly"
    READ_PATH_VALUE="${READ_PATH_VALUE:-frame}"
    READ_CUE="v31.sem_resid_coarse_l025.c23past"
    enable_semantic_role fine_path_router_debug "all"
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v37_track4_read_only_residual"
    ;;
  SEM_C23_02_NO_TTT)
    RUN_MODE="readonly"
    READ_PATH_VALUE="${READ_PATH_VALUE:-frame}"
    READ_CUE="v31.sem_resid_coarse_l025.c23past"
    enable_semantic_role causal_fg_semantic_risk_skip "frame,global"
    enable_compact_role_skip both early
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v37_track4_no_ttt_residual_source"
    ;;
  SEM_C23_03_NO_SWA)
    RUN_MODE="hybrid"
    READ_PATH_VALUE="${READ_PATH_VALUE:-frame}"
    READ_CUE="v31.sem_resid_coarse_l025.c23past"
    enable_semantic_role fine_path_router_debug "frame,global,ttt"
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v37_track4_no_swa_residual"
    ;;
  SEM_C23_04_FRAMEGLOBAL_COMPACT_ONLY)
    RUN_MODE="readonly"
    READ_PATH_VALUE="${READ_PATH_VALUE:-frame}"
    READ_CUE="v31.sem_resid_coarse_l025.c23past"
    enable_semantic_role causal_fg_semantic_risk_skip "frame,global"
    enable_compact_role_skip both early
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v37_track4_frameglobal_compact_only"
    ;;
  SEM_C23_05_STATIC_RESCUE_RESID)
    RUN_MODE="readonly"
    READ_PATH_VALUE="${READ_PATH_VALUE:-frame}"
    READ_CUE="mix.c23past_static_rescue_a025"
    enable_semantic_role fine_fg_structure_rescue "frame,global"
    enable_compact_role_skip both early
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v37_track4_static_rescue_residual"
    ;;

  FG_01_DYNAMIC_HIGHD_SKIP|FG_02_VEGETATION_HIGHD_SKIP|FG_03_LOWTRUST_HIGHD_SKIP|FG_04_STRUCTURE_RESCUE|FG_05_RISK_SKIP_STATIC_RESCUE|FG_06_COMPACT_KV_TRUE|FG_04_LOWTRUST_APPANOM_SKIP|FG_05_RISK_SKIP_STRUCTURE_RESCUE)
    RUN_MODE="readonly"
    case "$CANDIDATE_ID" in
      FG_01_DYNAMIC_HIGHD_SKIP) SEMANTIC_ROLE_POLICY_VALUE="causal_fg_semantic_risk_skip"; V24_FAMILY="v38_track1_dynamic_highd_skip" ;;
      FG_02_VEGETATION_HIGHD_SKIP) SEMANTIC_ROLE_POLICY_VALUE="causal_fg_vegetation_highrisk_skip"; V24_FAMILY="v38_track1_vegetation_highd_skip" ;;
      FG_03_LOWTRUST_HIGHD_SKIP) SEMANTIC_ROLE_POLICY_VALUE="causal_fg_risk_only"; V24_FAMILY="v38_track1_lowtrust_highd_skip" ;;
      FG_04_STRUCTURE_RESCUE) SEMANTIC_ROLE_POLICY_VALUE="fine_fg_structure_rescue"; V24_FAMILY="v38_track1_structure_rescue" ;;
      FG_05_RISK_SKIP_STATIC_RESCUE) SEMANTIC_ROLE_POLICY_VALUE="causal_fg_semantic_risk_skip"; V24_FAMILY="v38_track1_risk_skip_static_rescue" ;;
      FG_06_COMPACT_KV_TRUE) SEMANTIC_ROLE_POLICY_VALUE="causal_fg_semantic_risk_skip"; V24_FAMILY="v38_track1_compact_kv_true" ;;
      FG_04_LOWTRUST_APPANOM_SKIP) SEMANTIC_ROLE_POLICY_VALUE="causal_fg_risk_only"; V24_FAMILY="v39_track1_lowtrust_appanom_skip" ;;
      FG_05_RISK_SKIP_STRUCTURE_RESCUE) SEMANTIC_ROLE_POLICY_VALUE="causal_fg_semantic_risk_skip"; V24_FAMILY="v39_track1_risk_skip_structure_rescue" ;;
    esac
    SEMANTIC_MEMORY_PATHS_VALUE="frame,global"
    enable_compact_role_skip both early
    FINE_SPLIT_AVAILABLE=true
    ;;
  FG_03_SKY_APPANOM_WEAK_SKIP|FG_06_SHADOW_PROXY_SKIP)
    RUN_MODE="readonly"
    case "$CANDIDATE_ID" in
      FG_03_SKY_APPANOM_WEAK_SKIP) SEMANTIC_ROLE_POLICY_VALUE="causal_fg_soft_skip"; V24_FAMILY="v39_track1_sky_appanom_weak_skip" ;;
      FG_06_SHADOW_PROXY_SKIP) SEMANTIC_ROLE_POLICY_VALUE="causal_fg_soft_skip"; V24_FAMILY="v39_track1_shadow_proxy_skip" ;;
    esac
    SEMANTIC_MEMORY_PATHS_VALUE="frame,global"
    CONTEXT_SOURCE_SKIP_ENABLE=1
    CONTEXT_SOURCE_SKIP_IMPL="bias"
    CONTEXT_SOURCE_SKIP_SCOPE="both"
    CONTEXT_SOURCE_SKIP_MODE="soft"
    CONTEXT_SOURCE_SKIP_MASK="semantic_role_negative"
    CONTEXT_SOURCE_SKIP_LAYER_MODE="early"
    CONTEXT_SOURCE_SKIP_SOFT_RHO="${CONTEXT_SOURCE_SKIP_SOFT_RHO:-0.35}"
    CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP="${CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP:-0.75}"
    USES_CONTEXT_SKIP=true
    USES_TRUE_COMPACTION=false
    FINE_SPLIT_AVAILABLE=true
    ;;
  FG_07_BIAS_ONLY_CONTROL)
    RUN_MODE="readonly"
    enable_semantic_role causal_fg_semantic_risk_skip "frame,global"
    CONTEXT_SOURCE_SKIP_ENABLE=1
    CONTEXT_SOURCE_SKIP_IMPL="bias"
    CONTEXT_SOURCE_SKIP_SCOPE="both"
    CONTEXT_SOURCE_SKIP_MASK="semantic_role_negative"
    CONTEXT_SOURCE_SKIP_LAYER_MODE="early"
    USES_CONTEXT_SKIP=true
    USES_TRUE_COMPACTION=false
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v38_track1_bias_only_control"
    ;;

  SWA_01_NONOVERLAP_RISK_REMOVE|SWA_02_OVERLAP_K_KEEP_V_ATTEN|SWA_03_STRUCTURE_OVERLAP_PROTECT|SWA_04_DYNAMIC_NONOVERLAP_REMOVE_STRUCTURE_PROTECT|SWA_05_SKY_HORIZON_NEUTRAL|SWA_06_SOURCE_TOPOLOGY_CONTROL|SWA_01_NONOVERLAP_DYNAMIC_REMOVE|SWA_02_OVERLAP_K_KEEP_V_ATTEN_DYNAMIC|SWA_04_SKY_HORIZON_NEUTRAL|SWA_05_VEG_HIGHCONFLICT_NONOVERLAP_SKIP|SWA_06_COMBINED_LOCAL_TOPOLOGY)
    RUN_MODE="hybrid"
    export HMC_COMMIT_MODE=probe_native
    case "$CANDIDATE_ID" in
      SWA_01_NONOVERLAP_RISK_REMOVE|SWA_01_NONOVERLAP_DYNAMIC_REMOVE)
        SEMANTIC_ROLE_POLICY_VALUE="causal_swa_cache_lifecycle"
        export SWA_WRITE_KEEP_SCOPE="exclude_both_overlap"
        export SWA_WRITE_SPARSE_RATIO="${SWA_WRITE_SPARSE_RATIO:-0.50}"
        V24_FAMILY="v39_track2_nonoverlap_dynamic_remove"
        ;;
      SWA_02_OVERLAP_K_KEEP_V_ATTEN|SWA_02_OVERLAP_K_KEEP_V_ATTEN_DYNAMIC)
        SEMANTIC_ROLE_POLICY_VALUE="causal_swa_boundary_protect"
        export SWA_WRITE_KEEP_SCOPE="both_overlap"
        export ENABLE_SWA_OVERLAP_SOURCE_GATE=1
        export SWA_OVERLAP_SOURCE_GATE_RHO="${SWA_OVERLAP_SOURCE_GATE_RHO:-0.35}"
        export SWA_OVERLAP_SOURCE_GATE_MIN="${SWA_OVERLAP_SOURCE_GATE_MIN:-0.85}"
        export SWA_OVERLAP_SOURCE_GATE_TARGET="v"
        export SWA_OVERLAP_SOURCE_GATE_LAYER_MODE="${SWA_OVERLAP_SOURCE_GATE_LAYER_MODE:-last}"
        V24_FAMILY="v39_track2_overlap_k_keep_v_atten_dynamic"
        ;;
      SWA_03_STRUCTURE_OVERLAP_PROTECT)
        SEMANTIC_ROLE_POLICY_VALUE="causal_swa_overlap_structure_keep"
        export SWA_WRITE_KEEP_SCOPE="both_overlap"
        V24_FAMILY="v39_track2_structure_overlap_protect"
        ;;
      SWA_04_DYNAMIC_NONOVERLAP_REMOVE_STRUCTURE_PROTECT)
        SEMANTIC_ROLE_POLICY_VALUE="causal_swa_boundary_protect"
        export SWA_WRITE_KEEP_SCOPE="exclude_both_overlap"
        export SWA_WRITE_SPARSE_RATIO="${SWA_WRITE_SPARSE_RATIO:-0.60}"
        V24_FAMILY="v38_track2_dynamic_nonoverlap_remove_structure_protect"
        ;;
      SWA_05_SKY_HORIZON_NEUTRAL|SWA_04_SKY_HORIZON_NEUTRAL)
        SEMANTIC_ROLE_POLICY_VALUE="causal_swa_sky_partial_keep"
        export SWA_WRITE_KEEP_SCOPE="both_overlap"
        export ENABLE_SWA_OVERLAP_SOURCE_GATE=1
        export SWA_OVERLAP_SOURCE_GATE_RHO="${SWA_OVERLAP_SOURCE_GATE_RHO:-0.25}"
        export SWA_OVERLAP_SOURCE_GATE_MIN="${SWA_OVERLAP_SOURCE_GATE_MIN:-0.90}"
        export SWA_OVERLAP_SOURCE_GATE_TARGET="v"
        V24_FAMILY="v39_track2_sky_horizon_neutral"
        ;;
      SWA_05_VEG_HIGHCONFLICT_NONOVERLAP_SKIP)
        SEMANTIC_ROLE_POLICY_VALUE="causal_swa_vegetation_conditional"
        export SWA_WRITE_KEEP_SCOPE="exclude_both_overlap"
        export SWA_WRITE_SPARSE_RATIO="${SWA_WRITE_SPARSE_RATIO:-0.50}"
        V24_FAMILY="v39_track2_veg_highconflict_nonoverlap_skip"
        ;;
      SWA_06_SOURCE_TOPOLOGY_CONTROL)
        SEMANTIC_ROLE_POLICY_VALUE="causal_swa_cache_lifecycle"
        export SWA_WRITE_KEEP_SCOPE="all"
        export SWA_WRITE_SPARSE_RATIO="${SWA_WRITE_SPARSE_RATIO:-0.35}"
        export SWA_WRITE_RHO="${SWA_WRITE_RHO:-0.45}"
        export SWA_WRITE_MIN_GATE="${SWA_WRITE_MIN_GATE:-0.35}"
        V24_FAMILY="v38_track2_source_topology_control"
        ;;
      SWA_06_COMBINED_LOCAL_TOPOLOGY)
        SEMANTIC_ROLE_POLICY_VALUE="causal_swa_boundary_protect"
        export SWA_WRITE_KEEP_SCOPE="exclude_both_overlap"
        export SWA_WRITE_SPARSE_RATIO="${SWA_WRITE_SPARSE_RATIO:-0.45}"
        export ENABLE_SWA_OVERLAP_SOURCE_GATE=1
        export SWA_OVERLAP_SOURCE_GATE_RHO="${SWA_OVERLAP_SOURCE_GATE_RHO:-0.30}"
        export SWA_OVERLAP_SOURCE_GATE_MIN="${SWA_OVERLAP_SOURCE_GATE_MIN:-0.85}"
        export SWA_OVERLAP_SOURCE_GATE_TARGET="v"
        V24_FAMILY="v39_track2_combined_local_topology"
        ;;
    esac
    SEMANTIC_MEMORY_PATHS_VALUE="swa"
    enable_swa_policy
    FINE_SPLIT_AVAILABLE=true
    ;;

  TTT_01_STRUCTURE_LONG|TTT_02_STRUCTURE_LONG_DYNAMIC_NOLONG|TTT_03_VEGETATION_CONDITIONAL_SHORTNEG|TTT_04_LOWTRUST_SHORTNEG|TTT_05_SKY_NEUTRAL|TTT_06_FULL_LIFECYCLE_POLICY|TTT_01_STRUCTURE_LONG_ANCHOR|TTT_02_DYNAMIC_NO_LONG_WRITE|TTT_03_VEG_SHORT_NEGATIVE|TTT_04_SKY_NEUTRAL_NO_LONG|TTT_05_COMBINED_LIFECYCLE|TTT_06_SHADOW_LOWTRUST_NO_LONG)
    RUN_MODE="hybrid"
    case "$CANDIDATE_ID" in
      TTT_01_STRUCTURE_LONG|TTT_01_STRUCTURE_LONG_ANCHOR) SEMANTIC_ROLE_POLICY_VALUE="causal_ttt_structure_lowconflict_pos"; V24_FAMILY="v39_track3_structure_long_anchor" ;;
      TTT_02_STRUCTURE_LONG_DYNAMIC_NOLONG|TTT_02_DYNAMIC_NO_LONG_WRITE) SEMANTIC_ROLE_POLICY_VALUE="causal_ttt_block_highconflict_structure_longwrite"; V24_FAMILY="v39_track3_dynamic_no_long_write" ;;
      TTT_03_VEGETATION_CONDITIONAL_SHORTNEG|TTT_03_VEG_SHORT_NEGATIVE) SEMANTIC_ROLE_POLICY_VALUE="causal_ttt_vegetation_conditional_neg"; V24_FAMILY="v39_track3_veg_short_negative" ;;
      TTT_04_LOWTRUST_SHORTNEG) SEMANTIC_ROLE_POLICY_VALUE="causal_ttt_risk_only"; V24_FAMILY="v38_track3_lowtrust_shortneg" ;;
      TTT_05_SKY_NEUTRAL|TTT_04_SKY_NEUTRAL_NO_LONG) SEMANTIC_ROLE_POLICY_VALUE="fine_ttt_sky_neutral"; V24_FAMILY="v39_track3_sky_neutral_no_long" ;;
      TTT_06_FULL_LIFECYCLE_POLICY|TTT_05_COMBINED_LIFECYCLE) SEMANTIC_ROLE_POLICY_VALUE="causal_ttt_full_role_tree"; SEMANTIC_MEMORY_PATHS_VALUE="ttt,lifecycle"; enable_v26_dual_lifetime_short; V24_FAMILY="v39_track3_combined_lifecycle" ;;
      TTT_06_SHADOW_LOWTRUST_NO_LONG) SEMANTIC_ROLE_POLICY_VALUE="causal_ttt_lowstuff_highd_conflict_shortneg"; V24_FAMILY="v39_track3_shadow_lowtrust_no_long" ;;
    esac
    SEMANTIC_MEMORY_PATHS_VALUE="${SEMANTIC_MEMORY_PATHS_VALUE:-ttt}"
    FINE_SPLIT_AVAILABLE=true
    ;;

  C23R_01_READ_ONLY_RESID)
    RUN_MODE="readonly"
    READ_PATH_VALUE="${READ_PATH_VALUE:-frame}"
    READ_CUE="v31.sem_resid_coarse_l025.c23past"
    enable_semantic_role fine_path_router_debug "all"
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v38_track4_read_only_residual"
    ;;
  C23R_02_NO_TTT)
    RUN_MODE="readonly"
    READ_PATH_VALUE="${READ_PATH_VALUE:-frame}"
    READ_CUE="v31.sem_resid_coarse_l025.c23past"
    enable_semantic_role causal_fg_semantic_risk_skip "frame,global"
    enable_compact_role_skip both early
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v38_track4_no_ttt_residual_source"
    ;;
  C23R_03_NO_SWA)
    RUN_MODE="hybrid"
    READ_PATH_VALUE="${READ_PATH_VALUE:-frame}"
    READ_CUE="v31.sem_resid_coarse_l025.c23past"
    enable_semantic_role fine_path_router_debug "frame,global,ttt"
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v38_track4_no_swa_residual"
    ;;
  C23R_04_FRAMEGLOBAL_COMPACT_ONLY)
    RUN_MODE="readonly"
    READ_PATH_VALUE="${READ_PATH_VALUE:-frame}"
    READ_CUE="v31.sem_resid_coarse_l025.c23past"
    enable_semantic_role causal_fg_semantic_risk_skip "frame,global"
    enable_compact_role_skip both early
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v38_track4_frameglobal_compact_only"
    ;;
  C23R_05_STATIC_RESCUE_RESID)
    RUN_MODE="readonly"
    READ_PATH_VALUE="${READ_PATH_VALUE:-frame}"
    READ_CUE="mix.c23past_static_rescue_a025"
    enable_semantic_role fine_fg_structure_rescue "frame,global"
    enable_compact_role_skip both early
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v38_track4_static_rescue_residual"
    ;;
  C23R_06_C9_COMPAT_READ_ONLY)
    RUN_MODE="readonly"
    READ_PATH_VALUE="${READ_PATH_VALUE:-frame}"
    READ_CUE="v31.sem_resid_coarse_l025.c23past"
    enable_semantic_role fine_path_router_debug "frame,global"
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v38_track4_c9_compat_read_only"
    ;;
  C23R_04_FRAMEGLOBAL_ONLY)
    RUN_MODE="readonly"
    READ_PATH_VALUE="${READ_PATH_VALUE:-frame}"
    READ_CUE="v31.sem_resid_coarse_l025.c23past"
    enable_semantic_role causal_fg_semantic_risk_skip "frame,global"
    enable_compact_role_skip both early
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v39_track4_frameglobal_only_residual"
    ;;
  C23R_05_APPANOM_SEM_Z)
    RUN_MODE="readonly"
    READ_PATH_VALUE="${READ_PATH_VALUE:-frame}"
    READ_CUE="v31.sem_z_coarse.c23past"
    enable_semantic_role causal_fg_soft_skip "frame,global"
    CONTEXT_SOURCE_SKIP_ENABLE=1
    CONTEXT_SOURCE_SKIP_IMPL="bias"
    CONTEXT_SOURCE_SKIP_SCOPE="both"
    CONTEXT_SOURCE_SKIP_MODE="soft"
    CONTEXT_SOURCE_SKIP_MASK="semantic_role_negative"
    CONTEXT_SOURCE_SKIP_LAYER_MODE="early"
    CONTEXT_SOURCE_SKIP_SOFT_RHO="${CONTEXT_SOURCE_SKIP_SOFT_RHO:-0.25}"
    CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP="${CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP:-0.80}"
    USES_CONTEXT_SKIP=true
    USES_TRUE_COMPACTION=false
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v39_track4_appanom_sem_z_soft"
    ;;
  C23R_06_STATIC_RESCUE)
    RUN_MODE="readonly"
    READ_PATH_VALUE="${READ_PATH_VALUE:-frame}"
    READ_CUE="mix.c23past_static_rescue_a025"
    enable_semantic_role fine_fg_structure_rescue "frame,global"
    enable_compact_role_skip both early
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v39_track4_static_rescue"
    ;;

  P0_00_C9_REFERENCE|P0_01_HEALTH_LOGGING_ONLY|P0_02_SEMANTIC_PASSIVE_ONLY|P0_03_APPEARANCE_AUDIT_ONLY|P1_00_HEALTH_LOGGING_ONLY)
    RUN_MODE="readonly"
    SEMANTIC_PRIOR_MODE_VALUE="pass_through"
    case "$CANDIDATE_ID" in
      P0_02_SEMANTIC_PASSIVE_ONLY)
        enable_semantic_role fine_path_router_debug "all"
        FINE_SPLIT_AVAILABLE=true
        V24_FAMILY="v40_phase0_semantic_passive_only"
        ;;
      *)
        enable_semantic_role noop ""
        FINE_SPLIT_AVAILABLE=true
        V24_FAMILY="v40_phase0_noop_health_logging"
        ;;
    esac
    ;;

  READ_A1_HIGH_INFLUENCE_ANOMALY_V_ATTEN)
    RUN_MODE="readonly"
    READ_CUE="old_dyn_switch_flow_sem_veto"
    READ_CALIB_MODE="${READ_CALIB_MODE:-per_frame_quantile}"
    READ_TARGET_MASS="${READ_TARGET_MASS:-0.06}"
    enable_semantic_role causal_fg_soft_skip "frame,global"
    CONTEXT_SOURCE_SKIP_ENABLE=1
    CONTEXT_SOURCE_SKIP_IMPL="bias"
    CONTEXT_SOURCE_SKIP_SCOPE="both"
    CONTEXT_SOURCE_SKIP_MODE="soft"
    CONTEXT_SOURCE_SKIP_MASK="semantic_role_negative"
    CONTEXT_SOURCE_SKIP_LAYER_MODE="early"
    CONTEXT_SOURCE_SKIP_SOFT_RHO="${CONTEXT_SOURCE_SKIP_SOFT_RHO:-0.30}"
    CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP="${CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP:-0.80}"
    USES_CONTEXT_SKIP=true
    USES_TRUE_COMPACTION=false
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v40_read_high_influence_anomaly_v_atten"
    ;;
  READ_A2_HIGH_INFLUENCE_ANOMALY_KV_COMPACT)
    RUN_MODE="readonly"
    READ_CUE="old_dyn_switch_flow_sem_veto"
    READ_CALIB_MODE="${READ_CALIB_MODE:-per_frame_quantile}"
    READ_TARGET_MASS="${READ_TARGET_MASS:-0.06}"
    enable_semantic_role causal_fg_semantic_risk_skip "frame,global"
    enable_compact_role_skip both early
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v40_read_high_influence_anomaly_kv_compact"
    ;;
  READ_A3_DYNAMIC_VEG_SHADOW_HIGHD_SKIP_STRUCT_RESCUE)
    RUN_MODE="readonly"
    READ_CUE="old_dyn_key_static_rescue"
    enable_semantic_role causal_fg_semantic_risk_skip "frame,global"
    enable_compact_role_skip both early
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v40_read_dynamic_veg_shadow_skip_struct_rescue"
    ;;
  READ_A4_SKY_APPANOM_WEAK_ATTEN_ONLY_IF_SOURCE_MASS_HIGH)
    RUN_MODE="readonly"
    READ_CUE="old_dyn_switch_flow_sem_veto"
    READ_CALIB_MODE="${READ_CALIB_MODE:-per_frame_quantile}"
    READ_TARGET_MASS="${READ_TARGET_MASS:-0.05}"
    enable_semantic_role causal_fg_soft_skip "frame,global"
    CONTEXT_SOURCE_SKIP_ENABLE=1
    CONTEXT_SOURCE_SKIP_IMPL="bias"
    CONTEXT_SOURCE_SKIP_SCOPE="both"
    CONTEXT_SOURCE_SKIP_MODE="soft"
    CONTEXT_SOURCE_SKIP_MASK="semantic_role_negative"
    CONTEXT_SOURCE_SKIP_LAYER_MODE="early"
    CONTEXT_SOURCE_SKIP_SOFT_RHO="${CONTEXT_SOURCE_SKIP_SOFT_RHO:-0.20}"
    CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP="${CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP:-0.85}"
    USES_CONTEXT_SKIP=true
    USES_TRUE_COMPACTION=false
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v40_read_sky_appanom_weak_atten_if_source_mass_high"
    ;;
  READ_A5_STATIC_ANCHOR_RESCUE_ONLY)
    RUN_MODE="readonly"
    READ_CUE="old_dyn_key_static_rescue"
    READ_BLEND_LAMBDA="${READ_BLEND_LAMBDA:-0.25}"
    enable_semantic_role fine_fg_structure_rescue "frame,global"
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v40_read_static_anchor_rescue_only"
    ;;

  R1_READ_HIGH_INFLUENCE_ANOMALY)
    RUN_MODE="readonly"
    READ_CUE="old_dyn_switch_flow_sem_veto"
    READ_CALIB_MODE="${READ_CALIB_MODE:-per_frame_quantile}"
    READ_TARGET_MASS="${READ_TARGET_MASS:-0.06}"
    enable_semantic_role causal_fg_semantic_risk_skip "frame,global"
    enable_compact_role_skip both early
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v41_read_high_influence_anomaly_kv_compact"
    ;;
  R2_READ_SKY_APP_ANOMALY)
    RUN_MODE="readonly"
    READ_CUE="old_dyn_switch_flow_sem_veto"
    READ_CALIB_MODE="${READ_CALIB_MODE:-per_frame_quantile}"
    READ_TARGET_MASS="${READ_TARGET_MASS:-0.05}"
    enable_semantic_role causal_fg_soft_skip "frame,global"
    CONTEXT_SOURCE_SKIP_ENABLE=1
    CONTEXT_SOURCE_SKIP_IMPL="bias"
    CONTEXT_SOURCE_SKIP_SCOPE="both"
    CONTEXT_SOURCE_SKIP_MODE="soft"
    CONTEXT_SOURCE_SKIP_MASK="semantic_role_negative"
    CONTEXT_SOURCE_SKIP_LAYER_MODE="early"
    CONTEXT_SOURCE_SKIP_SOFT_RHO="${CONTEXT_SOURCE_SKIP_SOFT_RHO:-0.20}"
    CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP="${CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP:-0.85}"
    USES_CONTEXT_SKIP=true
    USES_TRUE_COMPACTION=false
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v41_read_sky_appanom_weak_atten_if_source_mass_high"
    ;;
  R3_READ_ANOMALY_PLUS_STATIC_RESCUE)
    RUN_MODE="readonly"
    READ_CUE="old_dyn_key_static_rescue"
    READ_CALIB_MODE="${READ_CALIB_MODE:-per_frame_quantile}"
    READ_TARGET_MASS="${READ_TARGET_MASS:-0.06}"
    enable_semantic_role causal_fg_semantic_risk_skip "frame,global"
    enable_compact_role_skip both early
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v41_read_anomaly_plus_static_rescue_proxy"
    ;;
  R4_NEG_CONTROL_SKY_NO_SOURCE_MASS)
    RUN_MODE="readonly"
    READ_CUE="old_dyn_switch_flow_sem_veto"
    READ_CALIB_MODE="${READ_CALIB_MODE:-none}"
    READ_TARGET_MASS="${READ_TARGET_MASS:-0.0}"
    enable_semantic_role causal_fg_soft_skip "frame,global"
    CONTEXT_SOURCE_SKIP_ENABLE=1
    CONTEXT_SOURCE_SKIP_IMPL="bias"
    CONTEXT_SOURCE_SKIP_SCOPE="both"
    CONTEXT_SOURCE_SKIP_MODE="soft"
    CONTEXT_SOURCE_SKIP_MASK="semantic_role_negative"
    CONTEXT_SOURCE_SKIP_LAYER_MODE="early"
    CONTEXT_SOURCE_SKIP_SOFT_RHO="${CONTEXT_SOURCE_SKIP_SOFT_RHO:-0.20}"
    CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP="${CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP:-0.85}"
    USES_CONTEXT_SKIP=true
    USES_TRUE_COMPACTION=false
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v41_neg_control_sky_no_source_mass_proxy"
    ;;
  R5_NEG_CONTROL_STATIC_RESCUE_ONLY)
    RUN_MODE="readonly"
    READ_CUE="old_dyn_key_static_rescue"
    READ_BLEND_LAMBDA="${READ_BLEND_LAMBDA:-0.25}"
    enable_semantic_role fine_fg_structure_rescue "frame,global"
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v41_neg_control_static_rescue_only"
    ;;

  SWA_B1_NONOVERLAP_RISKY_REMOVE_OVERLAP_KEEP|SWA_B2_OVERLAP_K_PRESERVE_V_ATTEN_RISKY|SWA_B3_STRUCTURE_OVERLAP_ANCHOR_PROTECT|SWA_B4_SKY_HORIZON_NEUTRAL_K_KEEP_V_ATTEN_IF_ANOMALOUS|SWA_B5_DYNAMIC_NONOVERLAP_REMOVE_STRUCTURE_OVERLAP_PROTECT)
    RUN_MODE="hybrid"
    export HMC_COMMIT_MODE=probe_native
    case "$CANDIDATE_ID" in
      SWA_B1_NONOVERLAP_RISKY_REMOVE_OVERLAP_KEEP)
        SEMANTIC_ROLE_POLICY_VALUE="causal_swa_cache_lifecycle"
        export SWA_WRITE_KEEP_SCOPE="exclude_both_overlap"
        export SWA_WRITE_SPARSE_RATIO="${SWA_WRITE_SPARSE_RATIO:-0.50}"
        V24_FAMILY="v40_swa_nonoverlap_risky_remove_overlap_keep"
        ;;
      SWA_B2_OVERLAP_K_PRESERVE_V_ATTEN_RISKY)
        SEMANTIC_ROLE_POLICY_VALUE="causal_swa_boundary_protect"
        export SWA_WRITE_KEEP_SCOPE="both_overlap"
        export ENABLE_SWA_OVERLAP_SOURCE_GATE=1
        export SWA_OVERLAP_SOURCE_GATE_RHO="${SWA_OVERLAP_SOURCE_GATE_RHO:-0.30}"
        export SWA_OVERLAP_SOURCE_GATE_MIN="${SWA_OVERLAP_SOURCE_GATE_MIN:-0.87}"
        export SWA_OVERLAP_SOURCE_GATE_TARGET="v"
        V24_FAMILY="v40_swa_overlap_k_preserve_v_atten_risky"
        ;;
      SWA_B3_STRUCTURE_OVERLAP_ANCHOR_PROTECT)
        SEMANTIC_ROLE_POLICY_VALUE="causal_swa_overlap_structure_keep"
        export SWA_WRITE_KEEP_SCOPE="both_overlap"
        V24_FAMILY="v40_swa_structure_overlap_anchor_protect"
        ;;
      SWA_B4_SKY_HORIZON_NEUTRAL_K_KEEP_V_ATTEN_IF_ANOMALOUS)
        SEMANTIC_ROLE_POLICY_VALUE="causal_swa_sky_partial_keep"
        export SWA_WRITE_KEEP_SCOPE="both_overlap"
        export ENABLE_SWA_OVERLAP_SOURCE_GATE=1
        export SWA_OVERLAP_SOURCE_GATE_RHO="${SWA_OVERLAP_SOURCE_GATE_RHO:-0.22}"
        export SWA_OVERLAP_SOURCE_GATE_MIN="${SWA_OVERLAP_SOURCE_GATE_MIN:-0.90}"
        export SWA_OVERLAP_SOURCE_GATE_TARGET="v"
        V24_FAMILY="v40_swa_sky_horizon_neutral_k_keep_v_atten"
        ;;
      SWA_B5_DYNAMIC_NONOVERLAP_REMOVE_STRUCTURE_OVERLAP_PROTECT)
        SEMANTIC_ROLE_POLICY_VALUE="causal_swa_boundary_protect"
        export SWA_WRITE_KEEP_SCOPE="exclude_both_overlap"
        export SWA_WRITE_SPARSE_RATIO="${SWA_WRITE_SPARSE_RATIO:-0.55}"
        V24_FAMILY="v40_swa_dynamic_nonoverlap_remove_structure_protect"
        ;;
    esac
    SEMANTIC_MEMORY_PATHS_VALUE="swa"
    enable_swa_policy
    FINE_SPLIT_AVAILABLE=true
    ;;

  TTT_C1_STRUCTURE_LOW_RISK_POSITIVE_LONG|TTT_C2_DYNAMIC_HIGH_RISK_NO_LONG_WRITE|TTT_C3_VEGETATION_HIGH_D_CONFLICT_SHORT_NEGATIVE|TTT_C4_SKY_ANOMALY_NO_POSITIVE_LONG_NEUTRAL|TTT_C5_COMBINED_LIFECYCLE|TTT_C6_FILTERED_COMMIT_ON_HEALTH_FAIL)
    RUN_MODE="hybrid"
    case "$CANDIDATE_ID" in
      TTT_C1_STRUCTURE_LOW_RISK_POSITIVE_LONG)
        SEMANTIC_ROLE_POLICY_VALUE="causal_ttt_structure_lowconflict_pos"
        V24_FAMILY="v40_ttt_structure_low_risk_positive_long"
        ;;
      TTT_C2_DYNAMIC_HIGH_RISK_NO_LONG_WRITE)
        SEMANTIC_ROLE_POLICY_VALUE="causal_ttt_block_highconflict_structure_longwrite"
        V24_FAMILY="v40_ttt_dynamic_high_risk_no_long_write"
        ;;
      TTT_C3_VEGETATION_HIGH_D_CONFLICT_SHORT_NEGATIVE)
        SEMANTIC_ROLE_POLICY_VALUE="causal_ttt_vegetation_conditional_neg"
        V24_FAMILY="v40_ttt_vegetation_highd_conflict_short_negative"
        ;;
      TTT_C4_SKY_ANOMALY_NO_POSITIVE_LONG_NEUTRAL)
        SEMANTIC_ROLE_POLICY_VALUE="fine_ttt_sky_neutral"
        V24_FAMILY="v40_ttt_sky_anomaly_no_positive_long_neutral"
        ;;
      TTT_C5_COMBINED_LIFECYCLE)
        SEMANTIC_ROLE_POLICY_VALUE="causal_ttt_full_role_tree"
        SEMANTIC_MEMORY_PATHS_VALUE="ttt,lifecycle"
        enable_v26_dual_lifetime_short
        V24_FAMILY="v40_ttt_combined_lifecycle"
        ;;
      TTT_C6_FILTERED_COMMIT_ON_HEALTH_FAIL)
        SEMANTIC_ROLE_POLICY_VALUE="fine_ttt_scale_conditioned"
        enable_v26_scale_state_commit
        export TTT_WRITE_COMMIT_FILTER_MODE=old_decay_by_risk
        export TTT_WRITE_COMMIT_FILTER_RISK_SOURCE=update_conflict_energy
        export TTT_WRITE_COMMIT_FILTER_SCOPE=tail_overlap
        export TTT_WRITE_COMMIT_FILTER_STAT=q90
        export TTT_WRITE_COMMIT_FILTER_BASE="${TTT_WRITE_COMMIT_FILTER_BASE:-0.35}"
        export TTT_WRITE_COMMIT_FILTER_GAIN="${TTT_WRITE_COMMIT_FILTER_GAIN:-0.60}"
        export TTT_WRITE_COMMIT_FILTER_MIN="${TTT_WRITE_COMMIT_FILTER_MIN:-0.15}"
        export TTT_WRITE_COMMIT_FILTER_MAX="${TTT_WRITE_COMMIT_FILTER_MAX:-1.0}"
        export TTT_WRITE_COMMIT_FILTER_BRANCH_MASK="${TTT_WRITE_COMMIT_FILTER_BRANCH_MASK:-0}"
        V24_FAMILY="v40_ttt_filtered_commit_on_health_fail"
        ;;
    esac
    SEMANTIC_MEMORY_PATHS_VALUE="${SEMANTIC_MEMORY_PATHS_VALUE:-ttt}"
    FINE_SPLIT_AVAILABLE=true
    ;;

  T0_SYN_ALL_PATCH_SKIP|T0_SYN_CENTER_BOX_SKIP|T0_SYN_RANDOM_20PCT_SKIP|T0_SYN_LEFT_HALF_SKIP|T0_SYN_ALL_DYNAMIC_ROLE|T0_SYN_ALL_STATIC_ROLE)
    RUN_MODE="readonly"
    enable_semantic_role v36_synthetic_override "frame,global"
    enable_compact_role_skip both early
    export V36_SYNTHETIC_PATH="all"
    case "$CANDIDATE_ID" in
      T0_SYN_ALL_PATCH_SKIP) export V36_SYNTHETIC_MASK="all_patch_skip"; export V36_SYNTHETIC_ACTION="source_skip"; V24_FAMILY="v38_track0_synthetic_all_patch_skip" ;;
      T0_SYN_CENTER_BOX_SKIP) export V36_SYNTHETIC_MASK="center_box_skip"; export V36_SYNTHETIC_ACTION="source_skip"; V24_FAMILY="v38_track0_synthetic_center_box_skip" ;;
      T0_SYN_RANDOM_20PCT_SKIP) export V36_SYNTHETIC_MASK="random_20pct_skip"; export V36_SYNTHETIC_ACTION="source_skip"; V24_FAMILY="v38_track0_synthetic_random_20pct_skip" ;;
      T0_SYN_LEFT_HALF_SKIP) export V36_SYNTHETIC_MASK="left_half_skip"; export V36_SYNTHETIC_ACTION="source_skip"; V24_FAMILY="v38_track0_synthetic_left_half_skip" ;;
      T0_SYN_ALL_DYNAMIC_ROLE) export V36_SYNTHETIC_MASK="all_dynamic_role"; export V36_SYNTHETIC_ACTION="source_skip"; V24_FAMILY="v38_track0_synthetic_all_dynamic_role" ;;
      T0_SYN_ALL_STATIC_ROLE) export V36_SYNTHETIC_MASK="all_static_role"; export V36_SYNTHETIC_ACTION="source_keep"; V24_FAMILY="v38_track0_synthetic_all_static_role" ;;
    esac
    FINE_SPLIT_AVAILABLE=true
    ;;
  T0_SEM_DYNAMIC_HIGHD|T0_SEM_VEGETATION_HIGHD|T0_SEM_SKY_HIGHD|T0_SEM_LOWTRUST_HIGHD|T0_SEM_STRUCTURE_LOWD|T0_SEM_STRUCTURE_LOWD_LOWCONFLICT)
    RUN_MODE="hybrid"
    case "$CANDIDATE_ID" in
      T0_SEM_DYNAMIC_HIGHD) SEMANTIC_ROLE_POLICY_VALUE="causal_fg_semantic_risk_skip"; SEMANTIC_MEMORY_PATHS_VALUE="frame,global"; enable_compact_role_skip both early; V24_FAMILY="v38_track0_sem_dynamic_highd" ;;
      T0_SEM_VEGETATION_HIGHD) SEMANTIC_ROLE_POLICY_VALUE="causal_fg_vegetation_highrisk_skip"; SEMANTIC_MEMORY_PATHS_VALUE="frame,global"; enable_compact_role_skip both early; V24_FAMILY="v38_track0_sem_vegetation_highd" ;;
      T0_SEM_SKY_HIGHD) SEMANTIC_ROLE_POLICY_VALUE="causal_fg_semantic_risk_skip"; SEMANTIC_MEMORY_PATHS_VALUE="frame,global"; enable_compact_role_skip both early; V24_FAMILY="v38_track0_sem_sky_highd" ;;
      T0_SEM_LOWTRUST_HIGHD) SEMANTIC_ROLE_POLICY_VALUE="causal_fg_risk_only"; SEMANTIC_MEMORY_PATHS_VALUE="frame,global"; enable_compact_role_skip both early; V24_FAMILY="v38_track0_sem_lowtrust_highd" ;;
      T0_SEM_STRUCTURE_LOWD) SEMANTIC_ROLE_POLICY_VALUE="fine_fg_structure_rescue"; SEMANTIC_MEMORY_PATHS_VALUE="frame,global"; enable_compact_role_skip both early; V24_FAMILY="v38_track0_sem_structure_lowd" ;;
      T0_SEM_STRUCTURE_LOWD_LOWCONFLICT) SEMANTIC_ROLE_POLICY_VALUE="causal_ttt_structure_lowconflict_pos"; SEMANTIC_MEMORY_PATHS_VALUE="ttt"; V24_FAMILY="v38_track0_sem_structure_lowd_lowconflict" ;;
    esac
    FINE_SPLIT_AVAILABLE=true
    ;;
  T0_PATH_FRAME_ONLY|T0_PATH_GLOBAL_ONLY|T0_PATH_FRAME_GLOBAL|T0_PATH_SWA_ONLY|T0_PATH_TTT_ONLY)
    RUN_MODE="readonly"
    enable_semantic_role v36_synthetic_override "all"
    export V36_SYNTHETIC_MASK="all_patch_skip"
    case "$CANDIDATE_ID" in
      T0_PATH_FRAME_ONLY)
        export V36_SYNTHETIC_PATH="frame"
        export V36_SYNTHETIC_ACTION="source_skip"
        enable_compact_role_skip frame early
        V24_FAMILY="v38_track0_path_frame_only"
        ;;
      T0_PATH_GLOBAL_ONLY)
        READ_PATH_VALUE="chunk"
        export V36_SYNTHETIC_PATH="global"
        export V36_SYNTHETIC_ACTION="source_skip"
        enable_compact_role_skip chunk early
        V24_FAMILY="v38_track0_path_global_only"
        ;;
      T0_PATH_FRAME_GLOBAL)
        export V36_SYNTHETIC_PATH="all"
        export V36_SYNTHETIC_ACTION="source_skip"
        enable_compact_role_skip both early
        V24_FAMILY="v38_track0_path_frame_global"
        ;;
      T0_PATH_SWA_ONLY)
        RUN_MODE="hybrid"
        export HMC_COMMIT_MODE=probe_native
        export V36_SYNTHETIC_PATH="swa"
        export V36_SYNTHETIC_ACTION="swa_remove"
        SEMANTIC_MEMORY_PATHS_VALUE="swa"
        enable_swa_policy
        V24_FAMILY="v38_track0_path_swa_only"
        ;;
      T0_PATH_TTT_ONLY)
        RUN_MODE="hybrid"
        export V36_SYNTHETIC_PATH="ttt"
        export V36_SYNTHETIC_ACTION="ttt_negative"
        SEMANTIC_MEMORY_PATHS_VALUE="ttt"
        V24_FAMILY="v38_track0_path_ttt_only"
        ;;
    esac
    FINE_SPLIT_AVAILABLE=true
    ;;

  K1_H9|P0_01_SEMANTIC_ROLE_NOOP_IGNORED)
    RUN_MODE="readonly"
    SEMANTIC_PRIOR_MODE_VALUE="pass_through"
    enable_semantic_role noop ""
    V24_FAMILY="phase0_noop"
    ;;
  P0_00_H9_REFERENCE)
    RUN_MODE="readonly"
    SEMANTIC_PRIOR_MODE_VALUE="pass_through"
    enable_semantic_role noop ""
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v26_phase0_reference"
    ;;
  P0_01_FINE_LABEL_LOADED_BUT_IGNORED)
    RUN_MODE="readonly"
    SEMANTIC_PRIOR_MODE_VALUE="pass_through"
    enable_semantic_role noop ""
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v26_phase0_noop"
    ;;
  P0_02_SEMANTIC_ROLE_PASS_THROUGH_CONSUMED)
    RUN_MODE="readonly"
    SEMANTIC_PRIOR_MODE_VALUE="pass_through"
    enable_semantic_role debug_only "all"
    V24_FAMILY="phase0_noop"
    ;;
  P0_02_FINE_ROLE_PASS_THROUGH_CONSUMED)
    RUN_MODE="readonly"
    SEMANTIC_PRIOR_MODE_VALUE="pass_through"
    enable_semantic_role fine_path_router_debug "all"
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v26_phase0_noop"
    ;;
  P0_03_SEMANTIC_ROLE_DEBUG_ONLY_ALL_MEMORY|PASSIVE_DEBUG_ONLY)
    RUN_MODE="readonly"
    enable_semantic_role debug_only "all"
    V24_FAMILY="passive_debug"
    ;;
  P0_03_FINE_ROLE_DEBUG_ONLY_ALL_PATHS)
    RUN_MODE="readonly"
    enable_semantic_role fine_path_router_debug "all"
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v26_phase0_debug"
    ;;
  P0_04_FRAME_SOURCE_SMOKE)
    RUN_MODE="readonly"
    enable_semantic_role fine_path_router "frame"
    enable_compact_role_skip frame early
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v26_phase0_path_smoke"
    ;;
  P0_05_SWA_CACHE_SMOKE)
    RUN_MODE="hybrid"
    enable_semantic_role fine_path_router "swa"
    enable_swa_policy
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v26_phase0_path_smoke"
    ;;
  P0_06_TTT_ROLE_SMOKE)
    RUN_MODE="hybrid"
    enable_semantic_role fine_path_router "ttt"
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v26_phase0_path_smoke"
    ;;

  V27_P0_00_H9_REFERENCE)
    RUN_MODE="readonly"
    SEMANTIC_PRIOR_MODE_VALUE="pass_through"
    enable_semantic_role noop ""
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v27_phase0_reference"
    ;;
  V27_P0_01_CAUSAL_LOADED_BUT_IGNORED)
    RUN_MODE="readonly"
    SEMANTIC_PRIOR_MODE_VALUE="pass_through"
    enable_semantic_role noop ""
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v27_phase0_noop"
    ;;
  V27_P0_02_CAUSAL_PASS_THROUGH_CONSUMED)
    RUN_MODE="readonly"
    SEMANTIC_PRIOR_MODE_VALUE="pass_through"
    enable_semantic_role causal_path_router_debug "all"
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v27_phase0_noop"
    ;;
  V27_P0_03_CAUSAL_DEBUG_ONLY_ALL_PATHS)
    RUN_MODE="readonly"
    enable_semantic_role causal_path_router_debug "all"
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v27_phase0_debug"
    ;;
  V27_P0_04_CAUSAL_FRAME_GLOBAL_SMOKE)
    RUN_MODE="readonly"
    enable_semantic_role causal_fg_structure_keep "frame,global"
    enable_compact_role_skip both early
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v27_phase0_path_smoke"
    ;;
  V27_P0_05_CAUSAL_SWA_SMOKE)
    RUN_MODE="hybrid"
    export HMC_COMMIT_MODE=probe_native
    enable_semantic_role causal_swa_overlap_structure_keep "swa"
    enable_swa_policy
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v27_phase0_path_smoke"
    ;;
  V27_P0_06_CAUSAL_TTT_SMOKE)
    RUN_MODE="hybrid"
    enable_semantic_role causal_ttt_full_role_tree "ttt"
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v27_phase0_path_smoke"
    ;;

  V28_P0_00_H9_REFERENCE)
    RUN_MODE="readonly"
    SEMANTIC_PRIOR_MODE_VALUE="pass_through"
    enable_semantic_role noop ""
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v28_phase0_reference"
    ;;
  V28_P0_01_SEM_LOADED_IGNORED)
    RUN_MODE="readonly"
    SEMANTIC_PRIOR_MODE_VALUE="pass_through"
    enable_semantic_role noop ""
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v28_phase0_noop"
    ;;
  V28_P0_02_SEM_PASS_THROUGH_CONSUMED)
    RUN_MODE="readonly"
    SEMANTIC_PRIOR_MODE_VALUE="pass_through"
    enable_semantic_role causal_path_router_debug "all"
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v28_phase0_noop"
    ;;
  V28_P0_03_TOKEN_RISK_DEBUG_ONLY)
    RUN_MODE="readonly"
    enable_semantic_role causal_path_router_debug "all"
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v28_phase0_token_risk_debug"
    ;;
  V28_P0_04_FRAME_SOURCE_SMOKE)
    RUN_MODE="readonly"
    enable_semantic_role causal_fg_structure_keep "frame"
    enable_compact_role_skip frame early
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v28_phase0_path_smoke"
    ;;
  V28_P0_05_GLOBAL_SOURCE_SMOKE)
    RUN_MODE="readonly"
    enable_semantic_role causal_fg_structure_keep "global"
    enable_compact_role_skip both early
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v28_phase0_path_smoke"
    ;;
  V28_P0_06_SWA_SOURCE_SMOKE)
    RUN_MODE="hybrid"
    export HMC_COMMIT_MODE=probe_native
    enable_semantic_role causal_swa_overlap_structure_keep "swa"
    enable_swa_policy
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v28_phase0_path_smoke"
    ;;
  V28_P0_07_TTT_WRITE_SMOKE)
    RUN_MODE="hybrid"
    enable_semantic_role causal_ttt_full_role_tree "ttt"
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v28_phase0_path_smoke"
    ;;
  V28_P0_08_ALL_PATH_DEBUG_ONLY)
    RUN_MODE="hybrid"
    export HMC_COMMIT_MODE=probe_native
    enable_semantic_role causal_path_router_debug "frame,global,swa,ttt"
    enable_swa_policy
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v28_phase0_debug"
    ;;

  FG_RISK_00)
    RUN_MODE="readonly"
    enable_semantic_role causal_fg_risk_only "frame,global"
    enable_compact_role_skip both early
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v27_phase2_fg_causal"
    ;;
  FG_SEM_01|FG_SEM_02|FG_SEM_03|FG_SEM_04|FG_SEM_05)
    RUN_MODE="readonly"
    case "$CANDIDATE_ID" in
      FG_SEM_01) SEMANTIC_ROLE_POLICY_VALUE="causal_fg_structure_keep" ;;
      FG_SEM_02) SEMANTIC_ROLE_POLICY_VALUE="causal_fg_vegetation_highrisk_skip" ;;
      FG_SEM_03) SEMANTIC_ROLE_POLICY_VALUE="causal_fg_structure_conflict_protect" ;;
      FG_SEM_04) SEMANTIC_ROLE_POLICY_VALUE="causal_fg_semantic_risk_skip" ;;
      FG_SEM_05) SEMANTIC_ROLE_POLICY_VALUE="causal_fg_soft_skip" ;;
    esac
    SEMANTIC_MEMORY_PATHS_VALUE="frame,global"
    enable_compact_role_skip both early
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v27_phase2_fg_causal"
    ;;
  SWA_SEM_01|SWA_SEM_02|SWA_SEM_03|SWA_SEM_04|SWA_SEM_05)
    RUN_MODE="hybrid"
    export HMC_COMMIT_MODE=probe_native
    case "$CANDIDATE_ID" in
      SWA_SEM_01) SEMANTIC_ROLE_POLICY_VALUE="causal_swa_overlap_structure_keep"; export SWA_WRITE_KEEP_SCOPE="both_overlap" ;;
      SWA_SEM_02) SEMANTIC_ROLE_POLICY_VALUE="causal_swa_sky_partial_keep"; export SWA_WRITE_KEEP_SCOPE="both_overlap" ;;
      SWA_SEM_03) SEMANTIC_ROLE_POLICY_VALUE="causal_swa_vegetation_conditional"; export SWA_WRITE_KEEP_SCOPE="all" ;;
      SWA_SEM_04) SEMANTIC_ROLE_POLICY_VALUE="causal_swa_boundary_protect"; export SWA_WRITE_KEEP_SCOPE="both_overlap" ;;
      SWA_SEM_05) SEMANTIC_ROLE_POLICY_VALUE="causal_swa_cache_lifecycle"; export SWA_WRITE_KEEP_SCOPE="all" ;;
    esac
    SEMANTIC_MEMORY_PATHS_VALUE="swa"
    enable_swa_policy
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v27_phase2_swa_causal"
    ;;
  TTT_ROLE_00_RISK_ONLY|TTT_ROLE_01_STRUCTURE_LOWCONFLICT_POS|TTT_ROLE_02_LOWSTUFF_HIGHD_CONFLICT_SHORTNEG|TTT_ROLE_03_VEGETATION_CONDITIONAL_NEG|TTT_ROLE_04_BLOCK_HIGHCONFLICT_STRUCTURE_LONGWRITE|TTT_ROLE_05_FULL_ROLE_TREE)
    RUN_MODE="hybrid"
    case "$CANDIDATE_ID" in
      TTT_ROLE_00_RISK_ONLY) SEMANTIC_ROLE_POLICY_VALUE="causal_ttt_risk_only" ;;
      TTT_ROLE_01_STRUCTURE_LOWCONFLICT_POS) SEMANTIC_ROLE_POLICY_VALUE="causal_ttt_structure_lowconflict_pos" ;;
      TTT_ROLE_02_LOWSTUFF_HIGHD_CONFLICT_SHORTNEG) SEMANTIC_ROLE_POLICY_VALUE="causal_ttt_lowstuff_highd_conflict_shortneg" ;;
      TTT_ROLE_03_VEGETATION_CONDITIONAL_NEG) SEMANTIC_ROLE_POLICY_VALUE="causal_ttt_vegetation_conditional_neg" ;;
      TTT_ROLE_04_BLOCK_HIGHCONFLICT_STRUCTURE_LONGWRITE) SEMANTIC_ROLE_POLICY_VALUE="causal_ttt_block_highconflict_structure_longwrite" ;;
      TTT_ROLE_05_FULL_ROLE_TREE) SEMANTIC_ROLE_POLICY_VALUE="causal_ttt_full_role_tree" ;;
    esac
    SEMANTIC_MEMORY_PATHS_VALUE="ttt"
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v27_phase2_ttt_causal"
    ;;

  FRAME_SEM_ONLY|FRAME_RISK_ONLY|FRAME_SEM_RISK)
    RUN_MODE="readonly"
    case "$CANDIDATE_ID" in
      FRAME_SEM_ONLY) SEMANTIC_ROLE_POLICY_VALUE="fine_fg_structure_keep"; V24_FAMILY="v28_frame_semantic_only" ;;
      FRAME_RISK_ONLY) SEMANTIC_ROLE_POLICY_VALUE="causal_fg_risk_only"; V24_FAMILY="v28_frame_risk_only" ;;
      FRAME_SEM_RISK) SEMANTIC_ROLE_POLICY_VALUE="causal_fg_semantic_risk_skip"; V24_FAMILY="v28_frame_semantic_risk" ;;
    esac
    SEMANTIC_MEMORY_PATHS_VALUE="frame"
    enable_compact_role_skip frame early
    FINE_SPLIT_AVAILABLE=true
    ;;
  GLOBAL_SEM_ONLY|GLOBAL_RISK_ONLY|GLOBAL_SEM_RISK)
    RUN_MODE="readonly"
    case "$CANDIDATE_ID" in
      GLOBAL_SEM_ONLY) SEMANTIC_ROLE_POLICY_VALUE="fine_fg_structure_keep"; V24_FAMILY="v28_global_semantic_only" ;;
      GLOBAL_RISK_ONLY) SEMANTIC_ROLE_POLICY_VALUE="causal_fg_risk_only"; V24_FAMILY="v28_global_risk_only" ;;
      GLOBAL_SEM_RISK) SEMANTIC_ROLE_POLICY_VALUE="causal_fg_semantic_risk_skip"; V24_FAMILY="v28_global_semantic_risk" ;;
    esac
    SEMANTIC_MEMORY_PATHS_VALUE="global"
    enable_compact_role_skip both early
    FINE_SPLIT_AVAILABLE=true
    ;;
  SWA_SEM_ONLY|SWA_RISK_ONLY|SWA_SEM_RISK)
    RUN_MODE="hybrid"
    export HMC_COMMIT_MODE=probe_native
    case "$CANDIDATE_ID" in
      SWA_SEM_ONLY) SEMANTIC_ROLE_POLICY_VALUE="fine_swa_overlap_structure_keep"; export SWA_WRITE_KEEP_SCOPE="both_overlap"; V24_FAMILY="v28_swa_semantic_only" ;;
      SWA_RISK_ONLY) SEMANTIC_ROLE_POLICY_VALUE="causal_swa_cache_lifecycle"; export SWA_WRITE_KEEP_SCOPE="all"; V24_FAMILY="v28_swa_risk_only" ;;
      SWA_SEM_RISK) SEMANTIC_ROLE_POLICY_VALUE="causal_swa_boundary_protect"; export SWA_WRITE_KEEP_SCOPE="both_overlap"; V24_FAMILY="v28_swa_semantic_risk" ;;
    esac
    SEMANTIC_MEMORY_PATHS_VALUE="swa"
    enable_swa_policy
    FINE_SPLIT_AVAILABLE=true
    ;;
  TTT_SEM_ONLY|TTT_RISK_ONLY|TTT_SEM_RISK)
    RUN_MODE="hybrid"
    case "$CANDIDATE_ID" in
      TTT_SEM_ONLY) SEMANTIC_ROLE_POLICY_VALUE="fine_ttt_structure_positive"; V24_FAMILY="v28_ttt_semantic_only" ;;
      TTT_RISK_ONLY) SEMANTIC_ROLE_POLICY_VALUE="causal_ttt_risk_only"; V24_FAMILY="v28_ttt_risk_only" ;;
      TTT_SEM_RISK) SEMANTIC_ROLE_POLICY_VALUE="causal_ttt_full_role_tree"; V24_FAMILY="v28_ttt_semantic_risk" ;;
    esac
    SEMANTIC_MEMORY_PATHS_VALUE="ttt"
    FINE_SPLIT_AVAILABLE=true
    ;;

  FG_FINE_01_STRUCTURE_KEEP|FG_FINE_03_SKY_NEUTRAL|FG_FINE_04_STRUCTURE_RESCUE|FG_FINE_05_CONFLICT_CONDITIONED)
    RUN_MODE="readonly"
    enable_semantic_role "fine_${CANDIDATE_ID#FG_FINE_}" "frame,global"
    case "$CANDIDATE_ID" in
      FG_FINE_01_STRUCTURE_KEEP) SEMANTIC_ROLE_POLICY_VALUE="fine_fg_structure_keep" ;;
      FG_FINE_03_SKY_NEUTRAL) SEMANTIC_ROLE_POLICY_VALUE="fine_fg_sky_neutral" ;;
      FG_FINE_04_STRUCTURE_RESCUE) SEMANTIC_ROLE_POLICY_VALUE="fine_fg_structure_rescue" ;;
      FG_FINE_05_CONFLICT_CONDITIONED) SEMANTIC_ROLE_POLICY_VALUE="fine_fg_conflict_conditioned" ;;
    esac
    enable_compact_role_skip both early
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v26_single_frame_global_fine"
    ;;
  FG_FINE_02_LOWSTUFF_HIGHD_SKIP)
    RUN_MODE="readonly"
    enable_semantic_role fine_fg_lowstuff_highd_skip "frame,global"
    enable_compact_role_skip both early
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v26_single_frame_global_fine"
    ;;

  SWA_FINE_01_OVERLAP_STRUCTURE_KEEP|SWA_FINE_02_SKY_PARTIAL_KEEP|SWA_FINE_03_VEGETATION_CONDITIONAL|SWA_FINE_04_BOUNDARY_PROTECT|SWA_FINE_05_CACHE_LIFECYCLE)
    RUN_MODE="hybrid"
    case "$CANDIDATE_ID" in
      SWA_FINE_01_OVERLAP_STRUCTURE_KEEP) SEMANTIC_ROLE_POLICY_VALUE="fine_swa_overlap_structure_keep"; export SWA_WRITE_KEEP_SCOPE="both_overlap" ;;
      SWA_FINE_02_SKY_PARTIAL_KEEP) SEMANTIC_ROLE_POLICY_VALUE="fine_swa_sky_partial_keep"; export SWA_WRITE_KEEP_SCOPE="both_overlap" ;;
      SWA_FINE_03_VEGETATION_CONDITIONAL) SEMANTIC_ROLE_POLICY_VALUE="fine_swa_vegetation_conditional"; export SWA_WRITE_KEEP_SCOPE="all" ;;
      SWA_FINE_04_BOUNDARY_PROTECT) SEMANTIC_ROLE_POLICY_VALUE="fine_swa_boundary_protect"; export SWA_WRITE_KEEP_SCOPE="both_overlap" ;;
      SWA_FINE_05_CACHE_LIFECYCLE) SEMANTIC_ROLE_POLICY_VALUE="fine_swa_cache_lifecycle"; export SWA_WRITE_KEEP_SCOPE="all" ;;
    esac
    SEMANTIC_MEMORY_PATHS_VALUE="swa"
    enable_swa_policy
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v26_single_swa_fine"
    ;;

  TTT_FINE_01_STRUCTURE_POSITIVE|TTT_FINE_02_SKY_NEUTRAL|TTT_FINE_03_SCALE_CONDITIONED|TTT_FINE_04_LOWSTUFF_HIGHD_SHORT|TTT_FINE_05_STRUCTURE_PROTECT)
    RUN_MODE="hybrid"
    case "$CANDIDATE_ID" in
      TTT_FINE_01_STRUCTURE_POSITIVE) SEMANTIC_ROLE_POLICY_VALUE="fine_ttt_structure_positive" ;;
      TTT_FINE_02_SKY_NEUTRAL) SEMANTIC_ROLE_POLICY_VALUE="fine_ttt_sky_neutral" ;;
      TTT_FINE_03_SCALE_CONDITIONED) SEMANTIC_ROLE_POLICY_VALUE="fine_ttt_scale_conditioned" ;;
      TTT_FINE_04_LOWSTUFF_HIGHD_SHORT) SEMANTIC_ROLE_POLICY_VALUE="fine_ttt_lowstuff_highd_short" ;;
      TTT_FINE_05_STRUCTURE_PROTECT) SEMANTIC_ROLE_POLICY_VALUE="fine_ttt_structure_protect" ;;
    esac
    SEMANTIC_MEMORY_PATHS_VALUE="ttt"
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v26_single_ttt_fine"
    ;;

  TTT_FINE_RISK_01_CONFLICT_TRI)
    RUN_MODE="hybrid"
    SEMANTIC_ROLE_POLICY_VALUE="fine_ttt_lowstuff_highd_short"
    SEMANTIC_MEMORY_PATHS_VALUE="ttt"
    enable_v26_conflict_tri_replay
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v26_ttt_fine_conflict_risk"
    ;;
  TTT_FINE_RISK_02_SCALE_STATE)
    RUN_MODE="hybrid"
    SEMANTIC_ROLE_POLICY_VALUE="fine_ttt_scale_conditioned"
    SEMANTIC_MEMORY_PATHS_VALUE="ttt"
    enable_v26_scale_state_commit
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v26_ttt_fine_scale_risk"
    ;;
  TTT_FINE_RISK_03_CONFLICT_COMMIT_FILTER)
    RUN_MODE="hybrid"
    SEMANTIC_ROLE_POLICY_VALUE="fine_ttt_structure_positive"
    SEMANTIC_MEMORY_PATHS_VALUE="ttt"
    enable_v26_conflict_tri_replay
    export TTT_WRITE_COMMIT_FILTER_MODE=old_decay_by_risk
    export TTT_WRITE_COMMIT_FILTER_RISK_SOURCE=update_conflict_energy
    export TTT_WRITE_COMMIT_FILTER_SCOPE=tail_overlap
    export TTT_WRITE_COMMIT_FILTER_STAT=q90
    export TTT_WRITE_COMMIT_FILTER_BASE="${TTT_WRITE_COMMIT_FILTER_BASE:-0.15}"
    export TTT_WRITE_COMMIT_FILTER_GAIN="${TTT_WRITE_COMMIT_FILTER_GAIN:-0.75}"
    export TTT_WRITE_COMMIT_FILTER_MIN="${TTT_WRITE_COMMIT_FILTER_MIN:-0.10}"
    export TTT_WRITE_COMMIT_FILTER_MAX="${TTT_WRITE_COMMIT_FILTER_MAX:-1.0}"
    export TTT_WRITE_COMMIT_FILTER_BRANCH_MASK="${TTT_WRITE_COMMIT_FILTER_BRANCH_MASK:-0}"
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v26_ttt_fine_conflict_commit_filter"
    ;;
  TTT_FINE_REPAIR_01_SCALE_DUAL_LIFETIME)
    RUN_MODE="hybrid"
    SEMANTIC_ROLE_POLICY_VALUE="fine_ttt_scale_conditioned"
    SEMANTIC_MEMORY_PATHS_VALUE="ttt,lifecycle"
    enable_v26_scale_state_commit
    enable_v26_dual_lifetime_short
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v26_ttt_fine_scale_dual_lifetime"
    ;;
  TTT_FINE_REPAIR_02_SCALE_CONFLICT_COMMIT_FILTER)
    RUN_MODE="hybrid"
    SEMANTIC_ROLE_POLICY_VALUE="fine_ttt_scale_conditioned"
    SEMANTIC_MEMORY_PATHS_VALUE="ttt"
    enable_v26_scale_state_commit
    export TTT_WRITE_COMMIT_FILTER_MODE=old_decay_by_risk
    export TTT_WRITE_COMMIT_FILTER_RISK_SOURCE=update_conflict_energy
    export TTT_WRITE_COMMIT_FILTER_SCOPE=tail_overlap
    export TTT_WRITE_COMMIT_FILTER_STAT=q90
    export TTT_WRITE_COMMIT_FILTER_BASE="${TTT_WRITE_COMMIT_FILTER_BASE:-0.35}"
    export TTT_WRITE_COMMIT_FILTER_GAIN="${TTT_WRITE_COMMIT_FILTER_GAIN:-0.60}"
    export TTT_WRITE_COMMIT_FILTER_MIN="${TTT_WRITE_COMMIT_FILTER_MIN:-0.15}"
    export TTT_WRITE_COMMIT_FILTER_MAX="${TTT_WRITE_COMMIT_FILTER_MAX:-1.0}"
    export TTT_WRITE_COMMIT_FILTER_BRANCH_MASK="${TTT_WRITE_COMMIT_FILTER_BRANCH_MASK:-0}"
    FINE_SPLIT_AVAILABLE=true
    V24_FAMILY="v26_ttt_fine_scale_conflict_commit_filter"
    ;;

  FRAMESEM_01_STRUCTURE_KEEP_LOWSTUFF_SOFT|FRAMEGLOBAL_01_FRAME_ONLY)
    RUN_MODE="readonly"
    enable_semantic_role structure_positive "frame"
    enable_compact_role_skip frame early
    V24_FAMILY="single_frame_source"
    ;;
  FRAMESEM_02_LOWSTUFF_HIGHD_SKIP)
    RUN_MODE="readonly"
    enable_semantic_role lowstuff_highd_skip "frame"
    enable_compact_role_skip frame early
    V24_FAMILY="single_frame_source"
    ;;
  FRAMESEM_03_SKY_NEUTRAL_VEGETATION_HIGHD_SKIP)
    RUN_MODE="readonly"
    mark_fine_fallback
    enable_semantic_role lowstuff_highd_skip "frame"
    enable_compact_role_skip frame early
    V24_FAMILY="single_frame_source_coarse_fallback"
    ;;
  GLOBALSEM_01_STRUCTURE_KEEP_LOWSTUFF_SOFT|FRAMEGLOBAL_02_GLOBAL_ONLY|CHUNKSEM_01_STRUCTURE_KEEP)
    RUN_MODE="readonly"
    READ_PATH_VALUE="chunk"
    enable_semantic_role structure_positive "global"
    enable_compact_role_skip chunk early
    V24_FAMILY="single_global_source"
    ;;
  GLOBALSEM_02_LOWSTUFF_HIGHD_SKIP|CHUNKSEM_02_LOWSTUFF_HIGHD_SKIP)
    RUN_MODE="readonly"
    READ_PATH_VALUE="chunk"
    enable_semantic_role lowstuff_highd_skip "global"
    enable_compact_role_skip chunk early
    V24_FAMILY="single_global_source"
    ;;
  CHUNKSEM_03_PROTECT_SPECIAL_TOKENS)
    RUN_MODE="readonly"
    READ_PATH_VALUE="chunk"
    enable_semantic_role structure_positive "global"
    enable_compact_role_skip chunk early
    V24_FAMILY="single_global_source"
    ;;
  FRAMEGLOBAL_03_FRAME_AND_GLOBAL)
    RUN_MODE="readonly"
    enable_semantic_role structure_positive "frame,global"
    enable_compact_role_skip both early
    V24_FAMILY="single_frame_global_source"
    ;;

  SWASEM_01_STRUCTURE_CACHE_KEEP)
    RUN_MODE="hybrid"
    enable_semantic_role structure_positive "swa"
    enable_swa_policy
    V24_FAMILY="single_swa_cache"
    ;;
  SWASEM_02_LOWSTUFF_HIGHD_CACHE_SOFTDROP)
    RUN_MODE="hybrid"
    SEMANTIC_ROLE_SWA_NEGATIVE_SCALE_VALUE="${SEMANTIC_ROLE_SWA_NEGATIVE_SCALE:-0.65}"
    enable_semantic_role lowstuff_highd_skip "swa"
    enable_swa_policy
    V24_FAMILY="single_swa_cache"
    ;;
  SWASEM_03_SKY_PROTECT_VEG_HIGHD_DROP)
    RUN_MODE="hybrid"
    mark_fine_fallback
    SEMANTIC_ROLE_SWA_NEGATIVE_SCALE_VALUE="${SEMANTIC_ROLE_SWA_NEGATIVE_SCALE:-0.65}"
    enable_semantic_role lowstuff_highd_skip "swa"
    enable_swa_policy
    V24_FAMILY="single_swa_cache_coarse_fallback"
    ;;
  SWASEM_04_PREVIOUS_SOURCE_ONLY)
    RUN_MODE="hybrid"
    export SWA_WRITE_KEEP_SCOPE="exclude_both_overlap"
    enable_semantic_role structure_positive "swa"
    enable_swa_policy
    V24_FAMILY="single_swa_cache"
    ;;
  SWASEM_05_OVERLAP_ONLY)
    RUN_MODE="hybrid"
    export SWA_WRITE_KEEP_SCOPE="both_overlap"
    enable_semantic_role structure_positive "swa"
    enable_swa_policy
    V24_FAMILY="single_swa_cache"
    ;;
  SWASEM_06_CURRENT_AND_PREVIOUS_COMPARE)
    RUN_MODE="hybrid"
    export SWA_WRITE_KEEP_SCOPE="all"
    enable_semantic_role all_memory_role "swa"
    enable_swa_policy
    V24_FAMILY="single_swa_cache"
    ;;

  TTTSEM_01_STRUCTURE_POSITIVE_LONG)
    RUN_MODE="hybrid"
    enable_semantic_role structure_positive "ttt"
    V24_FAMILY="single_ttt_write"
    ;;
  TTTSEM_02_LOWSTUFF_HIGHD_NEGATIVE_SHORT)
    RUN_MODE="hybrid"
    enable_semantic_role lowstuff_highd_skip "ttt"
    V24_FAMILY="single_ttt_write"
    ;;
  TTTSEM_03_SKY_NEUTRAL_PROTECT)
    RUN_MODE="hybrid"
    mark_fine_fallback
    enable_semantic_role structure_positive "ttt"
    V24_FAMILY="single_ttt_write_coarse_fallback"
    ;;
  TTTSEM_04_SEMANTIC_PLUS_TTT_CONFLICT|TTTSEM_05_SEMANTIC_PLUS_DG_PLUS_CONFLICT)
    RUN_MODE="hybrid"
    enable_semantic_role all_memory_role "ttt"
    V24_FAMILY="single_ttt_write"
    ;;
  TTTSEM_06_ROLE_SPECIFIC_BRANCH_W0)
    RUN_MODE="hybrid"
    export TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_BRANCH_MASK=0
    enable_semantic_role all_memory_role "ttt"
    V24_FAMILY="single_ttt_write"
    ;;
  TTTSEM_07_ROLE_SPECIFIC_LONG_SHORT)
    RUN_MODE="hybrid"
    enable_semantic_role long_short "ttt,lifecycle"
    export TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_MODE=dual_lifetime
    export TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_BRANCH_MASK=0
    export TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_LONG_SCALE="${TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_LONG_SCALE:-0.50}"
    export TTT_WRITE_TRANSIENT_DELTA_TTL="${TTT_WRITE_TRANSIENT_DELTA_TTL:-3}"
    V24_FAMILY="single_ttt_lifecycle"
    ;;

  PAIR_FRAME_TTT_PATHSPEC)
    RUN_MODE="hybrid"
    enable_semantic_role all_memory_role "frame,ttt"
    enable_compact_role_skip frame early
    V24_FAMILY="pairwise_frame_ttt"
    ;;
  PAIR_FRAME_SWA_PATHSPEC)
    RUN_MODE="hybrid"
    enable_semantic_role structure_positive "frame,swa"
    enable_compact_role_skip frame early
    enable_swa_policy
    V24_FAMILY="pairwise_frame_swa"
    ;;
  PAIR_SWA_TTT_PATHSPEC)
    RUN_MODE="hybrid"
    enable_semantic_role all_memory_role "swa,ttt"
    enable_swa_policy
    V24_FAMILY="pairwise_swa_ttt"
    ;;
  PAIR_GLOBAL_TTT_PATHSPEC)
    RUN_MODE="hybrid"
    READ_PATH_VALUE="chunk"
    enable_semantic_role all_memory_role "global,ttt"
    enable_compact_role_skip chunk early
    V24_FAMILY="pairwise_global_ttt"
    ;;
  PAIR_FRAME_GLOBAL_PATHSPEC)
    RUN_MODE="readonly"
    enable_semantic_role structure_positive "frame,global"
    enable_compact_role_skip both early
    V24_FAMILY="pairwise_frame_global"
    ;;
  PAIR_FRAME_GLOBAL_SWA_TTT_PATHSPEC)
    RUN_MODE="hybrid"
    enable_semantic_role all_memory_role "frame,global,swa,ttt"
    enable_compact_role_skip both early
    enable_swa_policy
    V24_FAMILY="pairwise_all"
    ;;

  ALLMEM_01_FRAME_TTT_PATHSPEC)
    RUN_MODE="hybrid"
    enable_semantic_role all_memory_role "frame,ttt"
    enable_compact_role_skip frame early
    V24_FAMILY="allmem_pathspec"
    ;;
  ALLMEM_02_FRAME_SWA_TTT_PATHSPEC)
    RUN_MODE="hybrid"
    enable_semantic_role all_memory_role "frame,swa,ttt"
    enable_compact_role_skip frame early
    enable_swa_policy
    V24_FAMILY="allmem_pathspec"
    ;;
  ALLMEM_03_FRAME_GLOBAL_SWA_TTT_PATHSPEC)
    RUN_MODE="hybrid"
    enable_semantic_role all_memory_role "frame,global,swa,ttt"
    enable_compact_role_skip both early
    enable_swa_policy
    V24_FAMILY="allmem_pathspec"
    ;;
  ALLMEM_04_SKY_NEUTRAL_STRUCTURE_LONG)
    RUN_MODE="hybrid"
    mark_fine_fallback
    enable_semantic_role structure_positive "frame,global,swa,ttt"
    enable_compact_role_skip both early
    enable_swa_policy
    V24_FAMILY="allmem_pathspec_coarse_fallback"
    ;;
  ALLMEM_05_LOWSTUFF_HIGHD_SHORTNEG)
    RUN_MODE="hybrid"
    enable_semantic_role lowstuff_highd_skip "frame,global,swa,ttt"
    enable_compact_role_skip both early
    enable_swa_policy
    V24_FAMILY="allmem_pathspec"
    ;;
  ALLMEM_06_CONFLICT_GATED_SEMANTIC)
    RUN_MODE="hybrid"
    enable_semantic_role all_memory_role "frame,global,swa,ttt,lifecycle"
    enable_compact_role_skip both early
    enable_swa_policy
    V24_FAMILY="allmem_pathspec"
    ;;
  *)
    echo "Unsupported CANDIDATE_ID for v24 rollout: $CANDIDATE_ID" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_PREFIX}_${CANDIDATE_ID}_chunk${CHUNK_ID}_h${HORIZON}_globalgate_H9parent_SWKS3"
RUN_DIR="$ROOT/$ROLLOUT_BASE/$RUN_NAME"

if [ -d "$RUN_DIR" ]; then
  if [ "${FORCE:-0}" != "1" ] && grep -q "DONE $RUN_NAME" "$RUN_DIR/run_status.txt" 2>/dev/null; then
    echo "SKIP existing DONE run: $RUN_NAME"
    exit 0
  fi
  STAMP="$(date '+%Y%m%d_%H%M%S')"
  INVALID_DIR="${RUN_DIR}.INVALID_RERUN_${STAMP}"
  mv "$RUN_DIR" "$INVALID_DIR"
  echo "Moved stale/forced run directory to: $INVALID_DIR"
fi

mkdir -p "$RUN_DIR"
if [ "${V24_SAVE_ATTRIBUTION_STATES:-0}" = "1" ]; then
  ATTR_CHUNKS="$CHUNK_ID,$((CHUNK_ID + 10)),$((CHUNK_ID + HORIZON))"
  export SAVE_HMC_STATES="${SAVE_HMC_STATES:-$RUN_DIR/hmc_state_snapshots}"
  export SAVE_HMC_STATE_CHUNKS="${SAVE_HMC_STATE_CHUNKS:-$ATTR_CHUNKS}"
  export SAVE_HMC_STATE_KINDS="${SAVE_HMC_STATE_KINDS:-input,after}"
  export SAVE_MERGE_STATES="${SAVE_MERGE_STATES:-$RUN_DIR/merge_state_snapshots}"
  export SAVE_MERGE_STATE_CHUNKS="${SAVE_MERGE_STATE_CHUNKS:-$ATTR_CHUNKS}"
  export SAVE_MERGE_STATE_KINDS="${SAVE_MERGE_STATE_KINDS:-input,after}"
fi
cat > "$RUN_DIR/run_config.yaml" <<EOF
run_name: "$RUN_NAME"
candidate_id: "$CANDIDATE_ID"
candidate_family: "$V24_FAMILY"
chunk_id: $CHUNK_ID
horizon: $HORIZON
start_frame: $START_FRAME
end_frame: $END_FRAME
parent: "H9_P0_V16_R2 causal fork snapshots"
diagnostic_only_short_rollout: true
counts_as_online_ttt_write_success: false
uses_gt_runtime_action: false
uses_semantic_cache: $USES_SEMANTIC_CACHE
uses_context_skip: $USES_CONTEXT_SKIP
uses_true_kv_compaction: $USES_TRUE_COMPACTION
fine_semantic_split_available: $FINE_SPLIT_AVAILABLE
candidate_uses_coarse_fallback_for_fine_roles: $COARSE_FALLBACK
semantic_role_policy: "$SEMANTIC_ROLE_POLICY_VALUE"
semantic_memory_paths: "$SEMANTIC_MEMORY_PATHS_VALUE"
read_cue: "$READ_CUE"
read_calib_mode: "${READ_CALIB_MODE:-none}"
read_target_mass: "${READ_TARGET_MASS:-0.06}"
read_quality_mass_min: "${READ_QUALITY_MASS_MIN:-0.03}"
read_quality_mass_max: "${READ_QUALITY_MASS_MAX:-0.20}"
read_quality_anchor_max: "${READ_QUALITY_ANCHOR_MAX:-0.35}"
read_quality_frag_max: "${READ_QUALITY_FRAG_MAX:-0.15}"
beta_policy: "${BETA_POLICY:-fixed}"
context_source_skip_impl: "$CONTEXT_SOURCE_SKIP_IMPL"
context_source_skip_scope: "$CONTEXT_SOURCE_SKIP_SCOPE"
context_source_skip_mask: "$CONTEXT_SOURCE_SKIP_MASK"
context_source_skip_record_attention_mass: "${CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS:-0}"
stage_c_mode: "$STAGE_C_MODE_VALUE"
stage_c_cache_dir: "$STAGE_C_CACHE_DIR_VALUE"
stage_c_cache_mode: "$STAGE_C_CACHE_MODE_VALUE"
stage_c_cache_require_hit: "$STAGE_C_CACHE_REQUIRE_HIT_VALUE"
run_mode: "$RUN_MODE"
write_score: "$WRITE_SCORE_VALUE"
EOF

DEFAULT_HMC="$PHASE1/state_snapshots/H9_P0_V16_R2/chunk_${SNAP}_input.pt"
DEFAULT_MERGE="$PHASE1/merge_state_snapshots/H9_P0_V16_R2/chunk_${SNAP}_input.pt"
if [ -f "$WARM_ROOT/snapshots/chunk_${SNAP}_input.pt" ]; then
  DEFAULT_HMC="$WARM_ROOT/snapshots/chunk_${SNAP}_input.pt"
fi
if [ -f "$WARM_ROOT/merge_snapshots/chunk_${SNAP}_input.pt" ]; then
  DEFAULT_MERGE="$WARM_ROOT/merge_snapshots/chunk_${SNAP}_input.pt"
fi

env \
  KITTI_SEQ=01 \
  ATTN_CUE_BASE="$ROLLOUT_BASE" \
  START_FRAME="$START_FRAME" \
  END_FRAME="$END_FRAME" \
  GLOBAL_CHUNK_OFFSET="$CHUNK_ID" \
  RESET_EVERY=5 \
  READ_PATH="$READ_PATH_VALUE" \
  READ_CALIB_MODE="${READ_CALIB_MODE:-none}" \
  READ_TARGET_MASS="${READ_TARGET_MASS:-0.06}" \
  READ_CALIB_TAU="${READ_CALIB_TAU:-0.05}" \
  READ_BLEND_LAMBDA="${READ_BLEND_LAMBDA:-0.25}" \
  READ_QUALITY_MASS_MIN="${READ_QUALITY_MASS_MIN:-0.03}" \
  READ_QUALITY_MASS_MAX="${READ_QUALITY_MASS_MAX:-0.20}" \
  READ_QUALITY_ANCHOR_MAX="${READ_QUALITY_ANCHOR_MAX:-0.35}" \
  READ_QUALITY_FRAG_MAX="${READ_QUALITY_FRAG_MAX:-0.15}" \
  GRAM_LAYER_GROUPS="${GRAM_LAYER_GROUPS:-shallow,middle,deep}" \
  BETA_POLICY="${BETA_POLICY:-fixed}" \
  BETA_ENERGY_TARGET="${BETA_ENERGY_TARGET:-0.0}" \
  BETA_MIN="${BETA_MIN:-0.5}" \
  BETA_MAX="${BETA_MAX:-1.5}" \
  READ_LAYER_MODE=all \
  LOGER_CHECKPOINT="${LOGER_CHECKPOINT:-$WARM_ROOT/ckpts/latest.pt}" \
  LOGER_CONFIG="${LOGER_CONFIG:-$WARM_ROOT/ckpts/original_config.yaml}" \
  STAGE_C_MODE="$STAGE_C_MODE_VALUE" \
  STAGE_C_CACHE_DIR="$STAGE_C_CACHE_DIR_VALUE" \
  STAGE_C_CACHE_MODE="$STAGE_C_CACHE_MODE_VALUE" \
  STAGE_C_CACHE_REQUIRE_HIT="$STAGE_C_CACHE_REQUIRE_HIT_VALUE" \
  STAGE_C_CACHE_VALIDATE="$STAGE_C_CACHE_VALIDATE_VALUE" \
  SEMANTIC_PRIOR_MODE="$SEMANTIC_PRIOR_MODE_VALUE" \
  ENABLE_CONTEXT_SOURCE_SKIP="$CONTEXT_SOURCE_SKIP_ENABLE" \
  CONTEXT_SOURCE_SKIP_IMPL="$CONTEXT_SOURCE_SKIP_IMPL" \
  CONTEXT_SOURCE_SKIP_SCOPE="$CONTEXT_SOURCE_SKIP_SCOPE" \
  CONTEXT_SOURCE_SKIP_MODE="$CONTEXT_SOURCE_SKIP_MODE" \
  CONTEXT_SOURCE_SKIP_MASK="$CONTEXT_SOURCE_SKIP_MASK" \
  CONTEXT_SOURCE_SKIP_LAYER_MODE="$CONTEXT_SOURCE_SKIP_LAYER_MODE" \
  CONTEXT_SOURCE_SKIP_SOFT_RHO="$CONTEXT_SOURCE_SKIP_SOFT_RHO" \
  CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP="$CONTEXT_SOURCE_SKIP_SOFT_MIN_KEEP" \
  CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS="${CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS:-0}" \
  CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES="${CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES:-512}" \
  SEMANTIC_ROLE_POLICY="$SEMANTIC_ROLE_POLICY_VALUE" \
  SEMANTIC_MEMORY_PATHS="$SEMANTIC_MEMORY_PATHS_VALUE" \
  SEMANTIC_ROLE_HIGHD_QUANTILE="$SEMANTIC_ROLE_HIGHD_QUANTILE_VALUE" \
  SEMANTIC_ROLE_LOW_TRUST="$SEMANTIC_ROLE_LOW_TRUST_VALUE" \
  SEMANTIC_ROLE_POSITIVE_SCALE="$SEMANTIC_ROLE_POSITIVE_SCALE_VALUE" \
  SEMANTIC_ROLE_NEUTRAL_SCALE="$SEMANTIC_ROLE_NEUTRAL_SCALE_VALUE" \
  SEMANTIC_ROLE_NEGATIVE_SCALE="$SEMANTIC_ROLE_NEGATIVE_SCALE_VALUE" \
  SEMANTIC_ROLE_SWA_NEGATIVE_SCALE="$SEMANTIC_ROLE_SWA_NEGATIVE_SCALE_VALUE" \
  V29C_MASKLET_ALIGNMENT_CSV="${V29C_MASKLET_ALIGNMENT_CSV:-}" \
  V29C_MASKLET_INTERVENTION_POLICY="${V29C_MASKLET_INTERVENTION_POLICY:-none}" \
  V29C_MASKLET_INTERVENTION_CHUNK="${V29C_MASKLET_INTERVENTION_CHUNK:--1}" \
  V29C_MASKLET_INTERVENTION_ID="${V29C_MASKLET_INTERVENTION_ID:--1}" \
  V29C_MASKLET_INTERVENTION_PATH="${V29C_MASKLET_INTERVENTION_PATH:-ttt}" \
  V29C_MASKLET_INTERVENTION_ACTION="${V29C_MASKLET_INTERVENTION_ACTION:-ttt_neutral}" \
  V29C_MASKLET_INTERVENTION_PATCH_THRESHOLD="${V29C_MASKLET_INTERVENTION_PATCH_THRESHOLD:-0.20}" \
  TTT_WRITE_GRADIENT_REVERSAL_MODE="${TTT_WRITE_GRADIENT_REVERSAL_MODE:-tri_replay}" \
  TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE="${TTT_WRITE_GRADIENT_REVERSAL_RISK_SOURCE:-d_tok}" \
  TTT_WRITE_TRI_REPLAY_POSITIVE_FRAC="${TTT_WRITE_TRI_REPLAY_POSITIVE_FRAC:-0.35}" \
  TTT_WRITE_TRI_REPLAY_NEGATIVE_FRAC="${TTT_WRITE_TRI_REPLAY_NEGATIVE_FRAC:-0.08}" \
  TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA="${TTT_WRITE_TRI_REPLAY_NEUTRAL_LAMBDA:-0.85}" \
  LOAD_HMC_STATE_AT_CHUNK="${LOAD_HMC_STATE_AT_CHUNK:-$DEFAULT_HMC}" \
  LOAD_HMC_STATE_AT_CHUNK_INDEX=0 \
  LOAD_MERGE_STATE_AT_CHUNK="${LOAD_MERGE_STATE_AT_CHUNK:-$DEFAULT_MERGE}" \
  LOAD_MERGE_STATE_AT_CHUNK_INDEX=0 \
  PROBE_CACHE_DIR="${PROBE_CACHE_DIR:-$ROOT/$PROBE_CACHE_ROOT/chunk_${CHUNK_ID}}" \
  PROBE_CACHE_MODE="${PROBE_CACHE_MODE:-${V24_PROBE_CACHE_MODE:-off}}" \
  PROBE_CACHE_PAYLOAD="${PROBE_CACHE_PAYLOAD:-${V24_PROBE_CACHE_PAYLOAD:-read_path_min}}" \
  PROBE_CACHE_REQUIRE_HIT="${PROBE_CACHE_REQUIRE_HIT:-${V24_PROBE_CACHE_REQUIRE_HIT:-0}}" \
  "$ROOT/tools/run_attention_cue_experiment.sh" \
  "$GPU" "$RUN_NAME" "$RUN_MODE" "$READ_CUE" "$BETA_VALUE" "$WRITE_SCORE_VALUE"
