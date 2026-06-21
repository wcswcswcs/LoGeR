#!/usr/bin/env python3
"""Evaluate ACL2 v76-TF phase outputs and write the final audit report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v76tf_common import V76_ROOT, ensure_dir, read_json, rel, write_json, write_text


def _load(path: Path) -> Dict[str, Any]:
    value = read_json(path)
    return value if isinstance(value, dict) else {}


def _status(
    phase0: Mapping[str, Any],
    phase1: Mapping[str, Any],
    phase2: Mapping[str, Any],
    phase3_6: Mapping[str, Any],
    h35_base: Mapping[str, Any],
) -> Dict[str, Any]:
    phase0_pass = bool(phase0.get("phase0_gate_pass"))
    phase1_signal = bool(phase1.get("phase1_start_a_signal_gate_pass"))
    phase1_strict = bool(phase1.get("phase1_strict_plan_gate_pass"))
    phase2_pass = bool(phase2.get("phase2_actuator_gate_pass"))
    precondition_pass = phase0_pass and phase1_strict and phase2_pass
    phase3_6_pass = bool(phase3_6.get("phase3_to_phase6_gate_pass"))
    h35_base_success = bool(h35_base.get("h35_base_strict_semantic_success_found"))
    h35_local_phase4_pass = bool(
        h35_base.get("h35_semread_l100_beta_sweep_phase4_allowed")
        or h35_base.get("h35_semread_l100_beta_sweep_h35_local_phase4_allowed")
        or h35_base.get("h35_semread_lam_beta_sweep_phase4_allowed")
        or h35_base.get("h35_semread_sweep_phase4_allowed")
        or h35_base.get("h35_global_l050_256f_phase4_allowed")
    )
    h35_official_704_tested = bool(
        h35_base.get("h35_official_aw110_l100_b525_704f_available")
        or h35_base.get("h35_official_aw110_repair_704f_available")
        or h35_base.get("h35_official_aw110_calib_repair_704f_available")
        or h35_base.get("h35_official_aw110_stable_positive_repair_704f_available")
        or h35_base.get("h35_official_aw110_low_lambda_fusion_repair_704f_available")
        or h35_base.get("h35_official_aw110_ultra_low_lambda_fusion_refine_704f_available")
        or h35_base.get("h35_official_aw110_read_layer_mode_probe_704f_available")
        or h35_base.get("h35_official_aw110_pca_single_read_layer_probe_704f_available")
        or h35_base.get("h35_official_aw110_pca_frame_read_layer_probe_704f_available")
        or h35_base.get("h35_official_aw110_pca_frame_dec00_beta_probe_704f_available")
        or h35_base.get("h35_official_aw110_pca_chunk_source_soft_read_layer_probe_704f_available")
    )
    promotion_allowed = precondition_pass and phase3_6_pass and h35_base_success
    primary_blocker = str(phase3_6.get("primary_blocker") or "one or more prerequisite gates failed")
    if h35_base and not h35_base_success:
        if h35_local_phase4_pass:
            if h35_official_704_tested:
                primary_blocker = (
                    "H35-base no-chunk semantic READ-only passed the 256F local Phase4 gate "
                    f"(best {h35_base.get('best_h35_semread_l100_beta_sweep_256f_candidate')} delta "
                    f"{h35_base.get('best_h35_semread_l100_beta_sweep_256f_candidate_minus_base_m')}m), "
                    "but official H35/v53 AW110 704F validation failed. "
                    f"L100_B525 official delta vs H35 704F = "
                    f"{h35_base.get('best_h35_official_aw110_l100_b525_704f_candidate_minus_h35_704_m')}m; "
                    f"best repair delta vs H35 704F = "
                    f"{h35_base.get('best_h35_official_aw110_repair_704f_candidate_minus_h35_704_m')}m; "
                    f"best calibration repair delta vs H35 704F = "
                    f"{h35_base.get('best_h35_official_aw110_calib_repair_704f_candidate_minus_h35_704_m')}m; "
                    f"best stable-positive repair delta vs H35 704F = "
                    f"{h35_base.get('best_h35_official_aw110_stable_positive_repair_704f_candidate_minus_h35_704_m')}m; "
                    f"best low-lambda fusion repair delta vs H35 704F = "
                    f"{h35_base.get('best_h35_official_aw110_low_lambda_fusion_repair_704f_candidate_minus_h35_704_m')}m; "
                    f"best ultra-low-lambda fusion refine delta vs H35 704F = "
                    f"{h35_base.get('best_h35_official_aw110_ultra_low_lambda_fusion_refine_704f_candidate_minus_h35_704_m')}m; "
                    f"best read-layer-mode probe delta vs H35 704F = "
                    f"{h35_base.get('best_h35_official_aw110_read_layer_mode_probe_704f_candidate_minus_h35_704_m')}m; "
                    f"best PCA-selected single-layer probe delta vs H35 704F = "
                    f"{h35_base.get('best_h35_official_aw110_pca_single_read_layer_probe_704f_candidate_minus_h35_704_m')}m; "
                    f"best PCA-selected frame-path layer probe delta vs H35 704F = "
                    f"{h35_base.get('best_h35_official_aw110_pca_frame_read_layer_probe_704f_candidate_minus_h35_704_m')}m; "
                    f"best DEC00 frame-path beta probe delta vs H35 704F = "
                    f"{h35_base.get('best_h35_official_aw110_pca_frame_dec00_beta_probe_704f_candidate_minus_h35_704_m')}m; "
                    f"best chunk source-soft layer probe delta vs H35 704F = "
                    f"{h35_base.get('best_h35_official_aw110_pca_chunk_source_soft_read_layer_probe_704f_candidate_minus_h35_704_m')}m. "
                    "C9/C9-clean/X3 evidence remains diagnostic only because chunk-wise policy is a confound."
                )
            else:
                primary_blocker = (
                    "H35-base no-chunk semantic READ-only has passed the 256F local Phase4 gate "
                    f"(best {h35_base.get('best_h35_semread_l100_beta_sweep_256f_candidate')} delta "
                    f"{h35_base.get('best_h35_semread_l100_beta_sweep_256f_candidate_minus_base_m')}m), "
                    "but 704F/full official H35 guard has not yet passed. "
                    "C9/C9-clean/X3 evidence remains diagnostic only because chunk-wise policy is a confound."
                )
        else:
            primary_blocker = (
                "H35-base no-chunk semantic candidates do not beat the clean H35/v53 reference. "
                "Current H35 semantic READ-only 256F sweeps/refinements have local gains, but Phase4 promotion remains false "
                "and all tested global tri/SWA handoff repairs worsen the baseline/read-only candidate. "
                "C9/C9-clean/X3 evidence remains diagnostic only because chunk-wise policy is a confound."
            )
    return {
        "phase0_gate_pass": phase0_pass,
        "phase1_start_a_signal_gate_pass": phase1_signal,
        "phase1_strict_plan_gate_pass": phase1_strict,
        "phase2_actuator_gate_pass": phase2_pass,
        "phase0_to_phase2_precondition_pass": precondition_pass,
        "phase3_to_phase6_gate_pass": phase3_6_pass,
        "h35_base_evidence_available": bool(h35_base.get("h35_base_evidence_available")),
        "h35_base_strict_semantic_success_found": h35_base_success,
        "h35_local_phase4_pass_found": h35_local_phase4_pass,
        "h35_official_704_tested": h35_official_704_tested,
        "h35_reference_full_ATE": h35_base.get("h35_reference_full_ATE"),
        "h35_reference_704_ATE": h35_base.get("h35_reference_704_ATE"),
        "best_h35_base_semantic_704_candidate_minus_h35_ref_m": h35_base.get(
            "best_h35_base_semantic_704_candidate_minus_h35_ref_m"
        ),
        "best_h35_base_semantic_full_candidate_minus_h35_ref_m": h35_base.get(
            "best_h35_base_semantic_full_candidate_minus_h35_ref_m"
        ),
        "h35_global_l050_256f_available": bool(h35_base.get("h35_global_l050_256f_available")),
        "h35_global_l050_256f_phase4_allowed": h35_base.get("h35_global_l050_256f_phase4_allowed"),
        "best_h35_global_l050_256f_candidate": h35_base.get("best_h35_global_l050_256f_candidate"),
        "best_h35_global_l050_256f_candidate_minus_base_m": h35_base.get(
            "best_h35_global_l050_256f_candidate_minus_base_m"
        ),
        "h35_global_l050_sem_tri_swa_minus_base_m": h35_base.get("h35_global_l050_sem_tri_swa_minus_base_m"),
        "h35_global_l050_sem_tri_swa_native110_minus_base_m": h35_base.get(
            "h35_global_l050_sem_tri_swa_native110_minus_base_m"
        ),
        "h35_global_l050_handoff_repair_256f_available": bool(
            h35_base.get("h35_global_l050_handoff_repair_256f_available")
        ),
        "h35_global_l050_handoff_repair_phase4_allowed": h35_base.get(
            "h35_global_l050_handoff_repair_phase4_allowed"
        ),
        "best_h35_global_l050_handoff_repair_candidate": h35_base.get(
            "best_h35_global_l050_handoff_repair_candidate"
        ),
        "best_h35_global_l050_handoff_repair_candidate_minus_base_m": h35_base.get(
            "best_h35_global_l050_handoff_repair_candidate_minus_base_m"
        ),
        "best_h35_global_l050_handoff_repair_candidate_minus_sem_read_m": h35_base.get(
            "best_h35_global_l050_handoff_repair_candidate_minus_sem_read_m"
        ),
        "h35_semread_sweep_256f_available": bool(h35_base.get("h35_semread_sweep_256f_available")),
        "h35_semread_sweep_phase4_allowed": h35_base.get("h35_semread_sweep_phase4_allowed"),
        "best_h35_semread_sweep_256f_candidate": h35_base.get("best_h35_semread_sweep_256f_candidate"),
        "best_h35_semread_sweep_256f_candidate_minus_base_m": h35_base.get(
            "best_h35_semread_sweep_256f_candidate_minus_base_m"
        ),
        "h35_semread_lam_beta_sweep_256f_available": bool(
            h35_base.get("h35_semread_lam_beta_sweep_256f_available")
        ),
        "h35_semread_lam_beta_sweep_phase4_allowed": h35_base.get(
            "h35_semread_lam_beta_sweep_phase4_allowed"
        ),
        "best_h35_semread_lam_beta_sweep_256f_candidate": h35_base.get(
            "best_h35_semread_lam_beta_sweep_256f_candidate"
        ),
        "best_h35_semread_lam_beta_sweep_256f_candidate_minus_base_m": h35_base.get(
            "best_h35_semread_lam_beta_sweep_256f_candidate_minus_base_m"
        ),
        "h35_semread_l100_beta_sweep_256f_available": bool(
            h35_base.get("h35_semread_l100_beta_sweep_256f_available")
        ),
        "h35_semread_l100_beta_sweep_phase4_allowed": h35_base.get(
            "h35_semread_l100_beta_sweep_phase4_allowed"
        ),
        "h35_semread_l100_beta_sweep_h35_local_phase4_allowed": h35_base.get(
            "h35_semread_l100_beta_sweep_h35_local_phase4_allowed"
        ),
        "best_h35_semread_l100_beta_sweep_256f_candidate": h35_base.get(
            "best_h35_semread_l100_beta_sweep_256f_candidate"
        ),
        "best_h35_semread_l100_beta_sweep_256f_candidate_minus_base_m": h35_base.get(
            "best_h35_semread_l100_beta_sweep_256f_candidate_minus_base_m"
        ),
        "h35_l100_b525_simplified_704f_available": bool(
            h35_base.get("h35_l100_b525_simplified_704f_available")
        ),
        "best_h35_l100_b525_simplified_704f_candidate": h35_base.get(
            "best_h35_l100_b525_simplified_704f_candidate"
        ),
        "best_h35_l100_b525_simplified_704f_candidate_minus_base_m": h35_base.get(
            "best_h35_l100_b525_simplified_704f_candidate_minus_base_m"
        ),
        "h35_official_aw110_l100_b525_704f_available": bool(
            h35_base.get("h35_official_aw110_l100_b525_704f_available")
        ),
        "best_h35_official_aw110_l100_b525_704f_candidate": h35_base.get(
            "best_h35_official_aw110_l100_b525_704f_candidate"
        ),
        "best_h35_official_aw110_l100_b525_704f_candidate_minus_h35_704_m": h35_base.get(
            "best_h35_official_aw110_l100_b525_704f_candidate_minus_h35_704_m"
        ),
        "h35_official_aw110_repair_704f_available": bool(
            h35_base.get("h35_official_aw110_repair_704f_available")
        ),
        "best_h35_official_aw110_repair_704f_candidate": h35_base.get(
            "best_h35_official_aw110_repair_704f_candidate"
        ),
        "best_h35_official_aw110_repair_704f_candidate_minus_h35_704_m": h35_base.get(
            "best_h35_official_aw110_repair_704f_candidate_minus_h35_704_m"
        ),
        "h35_official_aw110_calib_repair_704f_available": bool(
            h35_base.get("h35_official_aw110_calib_repair_704f_available")
        ),
        "best_h35_official_aw110_calib_repair_704f_candidate": h35_base.get(
            "best_h35_official_aw110_calib_repair_704f_candidate"
        ),
        "best_h35_official_aw110_calib_repair_704f_candidate_minus_h35_704_m": h35_base.get(
            "best_h35_official_aw110_calib_repair_704f_candidate_minus_h35_704_m"
        ),
        "h35_official_aw110_stable_positive_repair_704f_available": bool(
            h35_base.get("h35_official_aw110_stable_positive_repair_704f_available")
        ),
        "best_h35_official_aw110_stable_positive_repair_704f_candidate": h35_base.get(
            "best_h35_official_aw110_stable_positive_repair_704f_candidate"
        ),
        "best_h35_official_aw110_stable_positive_repair_704f_candidate_minus_h35_704_m": h35_base.get(
            "best_h35_official_aw110_stable_positive_repair_704f_candidate_minus_h35_704_m"
        ),
        "h35_official_aw110_low_lambda_fusion_repair_704f_available": bool(
            h35_base.get("h35_official_aw110_low_lambda_fusion_repair_704f_available")
        ),
        "best_h35_official_aw110_low_lambda_fusion_repair_704f_candidate": h35_base.get(
            "best_h35_official_aw110_low_lambda_fusion_repair_704f_candidate"
        ),
        "best_h35_official_aw110_low_lambda_fusion_repair_704f_candidate_minus_h35_704_m": h35_base.get(
            "best_h35_official_aw110_low_lambda_fusion_repair_704f_candidate_minus_h35_704_m"
        ),
        "h35_official_aw110_ultra_low_lambda_fusion_refine_704f_available": bool(
            h35_base.get("h35_official_aw110_ultra_low_lambda_fusion_refine_704f_available")
        ),
        "best_h35_official_aw110_ultra_low_lambda_fusion_refine_704f_candidate": h35_base.get(
            "best_h35_official_aw110_ultra_low_lambda_fusion_refine_704f_candidate"
        ),
        "best_h35_official_aw110_ultra_low_lambda_fusion_refine_704f_candidate_minus_h35_704_m": h35_base.get(
            "best_h35_official_aw110_ultra_low_lambda_fusion_refine_704f_candidate_minus_h35_704_m"
        ),
        "h35_official_aw110_read_layer_mode_probe_704f_available": bool(
            h35_base.get("h35_official_aw110_read_layer_mode_probe_704f_available")
        ),
        "best_h35_official_aw110_read_layer_mode_probe_704f_candidate": h35_base.get(
            "best_h35_official_aw110_read_layer_mode_probe_704f_candidate"
        ),
        "best_h35_official_aw110_read_layer_mode_probe_704f_candidate_minus_h35_704_m": h35_base.get(
            "best_h35_official_aw110_read_layer_mode_probe_704f_candidate_minus_h35_704_m"
        ),
        "h35_official_aw110_pca_single_read_layer_probe_704f_available": bool(
            h35_base.get("h35_official_aw110_pca_single_read_layer_probe_704f_available")
        ),
        "best_h35_official_aw110_pca_single_read_layer_probe_704f_candidate": h35_base.get(
            "best_h35_official_aw110_pca_single_read_layer_probe_704f_candidate"
        ),
        "best_h35_official_aw110_pca_single_read_layer_probe_704f_candidate_minus_h35_704_m": h35_base.get(
            "best_h35_official_aw110_pca_single_read_layer_probe_704f_candidate_minus_h35_704_m"
        ),
        "h35_official_aw110_pca_frame_read_layer_probe_704f_available": bool(
            h35_base.get("h35_official_aw110_pca_frame_read_layer_probe_704f_available")
        ),
        "best_h35_official_aw110_pca_frame_read_layer_probe_704f_candidate": h35_base.get(
            "best_h35_official_aw110_pca_frame_read_layer_probe_704f_candidate"
        ),
        "best_h35_official_aw110_pca_frame_read_layer_probe_704f_candidate_minus_h35_704_m": h35_base.get(
            "best_h35_official_aw110_pca_frame_read_layer_probe_704f_candidate_minus_h35_704_m"
        ),
        "h35_official_aw110_pca_frame_dec00_beta_probe_704f_available": bool(
            h35_base.get("h35_official_aw110_pca_frame_dec00_beta_probe_704f_available")
        ),
        "best_h35_official_aw110_pca_frame_dec00_beta_probe_704f_candidate": h35_base.get(
            "best_h35_official_aw110_pca_frame_dec00_beta_probe_704f_candidate"
        ),
        "best_h35_official_aw110_pca_frame_dec00_beta_probe_704f_candidate_minus_h35_704_m": h35_base.get(
            "best_h35_official_aw110_pca_frame_dec00_beta_probe_704f_candidate_minus_h35_704_m"
        ),
        "h35_official_aw110_pca_chunk_source_soft_read_layer_probe_704f_available": bool(
            h35_base.get("h35_official_aw110_pca_chunk_source_soft_read_layer_probe_704f_available")
        ),
        "best_h35_official_aw110_pca_chunk_source_soft_read_layer_probe_704f_candidate": h35_base.get(
            "best_h35_official_aw110_pca_chunk_source_soft_read_layer_probe_704f_candidate"
        ),
        "best_h35_official_aw110_pca_chunk_source_soft_read_layer_probe_704f_candidate_minus_h35_704_m": h35_base.get(
            "best_h35_official_aw110_pca_chunk_source_soft_read_layer_probe_704f_candidate_minus_h35_704_m"
        ),
        "c9_c9clean_dechunk_counts_as_strict_success": False,
        "phase7_to_phase8_promotion_allowed": promotion_allowed,
        "overall_status": "go_for_09_704f_full" if promotion_allowed else "no_go_after_phase3_6_semantic_control_gates",
        "primary_blocker": "none" if promotion_allowed else primary_blocker,
    }


def _answer_rows(
    phase0: Mapping[str, Any],
    phase1: Mapping[str, Any],
    phase2: Mapping[str, Any],
    phase3_6: Mapping[str, Any],
    h35_base: Mapping[str, Any],
    status: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    return [
        {
            "question": "Can C9 tri-replay / tri-gamma / commit-EMA signature be recovered?",
            "answer": "yes" if phase0.get("ttt_tri_replay_configs_recoverable") and phase2.get("c9_knockout_direction_reproduced") else "partial/no",
            "evidence": "Phase0 config/artifact lock plus Phase2 C9 knockout direction fields.",
        },
        {
            "question": "Which C9 components have clean H35 positive-only effect?",
            "answer": "READ and TTT have Start A positive-only gains; SWA alone is near zero in imported v46B.",
            "evidence": "Phase1 positive_only_factorial_table.csv and phase1 summary.",
        },
        {
            "question": "Is TTT tri-replay independently active?",
            "answer": "yes for actuator/activity in historical traces" if phase2.get("tri_replay_applied_count_gt0") else "no",
            "evidence": "Phase2 recursive trace and v46B registry cross-check.",
        },
        {
            "question": "Does semantic tri-replay beat geometry-only and shuffled/random controls?",
            "answer": "no for available v76-imported control evidence.",
            "evidence": (
                "Phase3-6 semantic control summary: SEM4/X3 are chunk-map contaminated; "
                "C9-clean smoke worsens D7; H35-base semantic evidence does not beat H35/v53 reference."
            ),
        },
        {
            "question": "Does semantic residual READ + tri-replay form a handshake?",
            "answer": "not as a deployable online gate.",
            "evidence": "Phase1 shows clean READ+TTT synergy, but Phase4 READ online smoke gate is false.",
        },
        {
            "question": "Does commit EMA bridge TTT changes to future geometry?",
            "answer": "not established as a training-free v76 bridge.",
            "evidence": str(phase3_6.get("phase5_commit_ema_bridge_reason")),
        },
        {
            "question": "Does SWA handoff help tri-replay propagation?",
            "answer": "no in available local-window smoke evidence.",
            "evidence": "Phase6 SWA online smoke gate is false and candidate harms local ATE in the imported summary.",
        },
        {
            "question": "Does a fixed semantic trigger migrate to KITTI09?",
            "answer": "not promoted to KITTI09.",
            "evidence": "Phase7 requires Phase3-6 gate pass; it failed. Historical v74 KITTI09 remains prefix-only.",
        },
        {
            "question": "Did any candidate pass 704F/full official H35 guard?",
            "answer": "no.",
            "evidence": (
                "H35-base evidence lock: strict semantic success found = "
                f"{h35_base.get('h35_base_strict_semantic_success_found')}; "
                "v76 Phase8 promotion remains false."
            ),
        },
        {
            "question": "If no, what is the blocker?",
            "answer": str(status.get("primary_blocker")),
            "evidence": "Final status JSON.",
        },
    ]


def _write_report(
    out_dir: Path,
    phase0: Mapping[str, Any],
    phase1: Mapping[str, Any],
    phase2: Mapping[str, Any],
    phase3_6: Mapping[str, Any],
    h35_base: Mapping[str, Any],
    status: Mapping[str, Any],
    answers: List[Mapping[str, Any]],
) -> None:
    lines = [
        "# ACL2 v76-TF Memory Control Final Audit",
        "",
        "This report is generated from auditable phase outputs. It does not invent missing metrics.",
        "",
        "## Phase Status",
        "",
        f"- Phase 0 C9 experience lock: `{status['phase0_gate_pass']}`",
        f"- Phase 1 Start A signal gate: `{status['phase1_start_a_signal_gate_pass']}`",
        f"- Phase 1 strict Start A + Start B plan gate: `{status['phase1_strict_plan_gate_pass']}`",
        f"- Phase 2 actuator gate: `{status['phase2_actuator_gate_pass']}`",
        f"- Phase 3-6 semantic control gate: `{status['phase3_to_phase6_gate_pass']}`",
        f"- H35-base strict semantic success: `{status['h35_base_strict_semantic_success_found']}`",
        f"- C9/C9-clean/dechunk counts as strict success: `{status['c9_c9clean_dechunk_counts_as_strict_success']}`",
        f"- Promotion to Phase 7-8: `{status['phase7_to_phase8_promotion_allowed']}`",
        f"- Overall status: `{status['overall_status']}`",
        "",
        "## Key Metrics",
        "",
        f"- C9 repeat metric available: `{phase0.get('c9_locked_repeat_metric_available')}`",
        f"- H35 clean metric available: `{phase0.get('h35_clean_metric_available')}`",
        f"- Start A best gain vs F000: `{phase1.get('start_a_best_gain_vs_F000')}`",
        f"- READ-only gain vs F000: `{phase1.get('read_only_gain_vs_F000')}`",
        f"- TTT-only gain vs F000: `{phase1.get('ttt_only_gain_vs_F000')}`",
        f"- READ+TTT gain vs F000: `{phase1.get('read_ttt_gain_vs_F000')}`",
        f"- READ+TTT synergy over best single: `{phase1.get('read_ttt_synergy_over_best_single')}`",
        f"- Tri-replay applied count registry total: `{phase2.get('tri_replay_applied_count_registry_total')}`",
        f"- Post-zp nonzero numeric count total: `{phase2.get('post_zp_nonzero_numeric_count_total')}`",
        f"- SEM4 historical positive but chunk-mapped: `{phase3_6.get('phase3_historical_sem4_positive_but_chunk_mapped')}`",
        f"- H35 reference full ATE: `{h35_base.get('h35_reference_full_ATE')}`",
        f"- H35 reference 704F ATE: `{h35_base.get('h35_reference_704_ATE')}`",
        f"- Best H35-base semantic 704F delta vs H35 ref: `{h35_base.get('best_h35_base_semantic_704_candidate_minus_h35_ref_m')}`",
        f"- Best H35-base semantic full delta vs H35 ref: `{h35_base.get('best_h35_base_semantic_full_candidate_minus_h35_ref_m')}`",
        f"- Best current H35 global-L050 256F candidate: `{h35_base.get('best_h35_global_l050_256f_candidate')}`",
        f"- Best current H35 global-L050 256F delta vs base: `{h35_base.get('best_h35_global_l050_256f_candidate_minus_base_m')}`",
        f"- Current H35 global-L050 tri/SWA delta vs base: `{h35_base.get('h35_global_l050_sem_tri_swa_minus_base_m')}`",
        f"- Current H35 global-L050 tri/SWA native110 delta vs base: `{h35_base.get('h35_global_l050_sem_tri_swa_native110_minus_base_m')}`",
        f"- Best H35 global-L050 handoff repair candidate: `{h35_base.get('best_h35_global_l050_handoff_repair_candidate')}`",
        f"- Best H35 global-L050 handoff repair delta vs base: `{h35_base.get('best_h35_global_l050_handoff_repair_candidate_minus_base_m')}`",
        f"- Best H35 global-L050 handoff repair delta vs SEM_READ: `{h35_base.get('best_h35_global_l050_handoff_repair_candidate_minus_sem_read_m')}`",
        f"- Best H35 semantic READ sweep 256F candidate: `{h35_base.get('best_h35_semread_sweep_256f_candidate')}`",
        f"- Best H35 semantic READ sweep 256F delta vs base: `{h35_base.get('best_h35_semread_sweep_256f_candidate_minus_base_m')}`",
        f"- Best H35 semantic READ lambda/beta refinement 256F candidate: `{h35_base.get('best_h35_semread_lam_beta_sweep_256f_candidate')}`",
        f"- Best H35 semantic READ lambda/beta refinement 256F delta vs base: `{h35_base.get('best_h35_semread_lam_beta_sweep_256f_candidate_minus_base_m')}`",
        f"- Best H35 semantic READ L100 beta 256F candidate: `{h35_base.get('best_h35_semread_l100_beta_sweep_256f_candidate')}`",
        f"- Best H35 semantic READ L100 beta 256F delta vs base: `{h35_base.get('best_h35_semread_l100_beta_sweep_256f_candidate_minus_base_m')}`",
        f"- Best H35 L100_B525 simplified 704F candidate: `{h35_base.get('best_h35_l100_b525_simplified_704f_candidate')}`",
        f"- Best H35 L100_B525 simplified 704F delta vs simplified base: `{h35_base.get('best_h35_l100_b525_simplified_704f_candidate_minus_base_m')}`",
        f"- Best H35 official AW110 L100_B525 704F candidate: `{h35_base.get('best_h35_official_aw110_l100_b525_704f_candidate')}`",
        f"- Best H35 official AW110 L100_B525 704F delta vs H35 704F: `{h35_base.get('best_h35_official_aw110_l100_b525_704f_candidate_minus_h35_704_m')}`",
        f"- Best H35 official AW110 repair 704F candidate: `{h35_base.get('best_h35_official_aw110_repair_704f_candidate')}`",
        f"- Best H35 official AW110 repair 704F delta vs H35 704F: `{h35_base.get('best_h35_official_aw110_repair_704f_candidate_minus_h35_704_m')}`",
        f"- Best H35 official AW110 calibration repair 704F candidate: `{h35_base.get('best_h35_official_aw110_calib_repair_704f_candidate')}`",
        f"- Best H35 official AW110 calibration repair 704F delta vs H35 704F: `{h35_base.get('best_h35_official_aw110_calib_repair_704f_candidate_minus_h35_704_m')}`",
        f"- Best H35 official AW110 stable-positive repair 704F candidate: `{h35_base.get('best_h35_official_aw110_stable_positive_repair_704f_candidate')}`",
        f"- Best H35 official AW110 stable-positive repair 704F delta vs H35 704F: `{h35_base.get('best_h35_official_aw110_stable_positive_repair_704f_candidate_minus_h35_704_m')}`",
        f"- Best H35 official AW110 low-lambda fusion repair 704F candidate: `{h35_base.get('best_h35_official_aw110_low_lambda_fusion_repair_704f_candidate')}`",
        f"- Best H35 official AW110 low-lambda fusion repair 704F delta vs H35 704F: `{h35_base.get('best_h35_official_aw110_low_lambda_fusion_repair_704f_candidate_minus_h35_704_m')}`",
        f"- Best H35 official AW110 ultra-low-lambda fusion refine 704F candidate: `{h35_base.get('best_h35_official_aw110_ultra_low_lambda_fusion_refine_704f_candidate')}`",
        f"- Best H35 official AW110 ultra-low-lambda fusion refine 704F delta vs H35 704F: `{h35_base.get('best_h35_official_aw110_ultra_low_lambda_fusion_refine_704f_candidate_minus_h35_704_m')}`",
        f"- Best H35 official AW110 read-layer-mode probe 704F candidate: `{h35_base.get('best_h35_official_aw110_read_layer_mode_probe_704f_candidate')}`",
        f"- Best H35 official AW110 read-layer-mode probe 704F delta vs H35 704F: `{h35_base.get('best_h35_official_aw110_read_layer_mode_probe_704f_candidate_minus_h35_704_m')}`",
        f"- Best H35 official AW110 PCA-selected single-layer probe 704F candidate: `{h35_base.get('best_h35_official_aw110_pca_single_read_layer_probe_704f_candidate')}`",
        f"- Best H35 official AW110 PCA-selected single-layer probe 704F delta vs H35 704F: `{h35_base.get('best_h35_official_aw110_pca_single_read_layer_probe_704f_candidate_minus_h35_704_m')}`",
        f"- Best H35 official AW110 PCA-selected frame-path layer probe 704F candidate: `{h35_base.get('best_h35_official_aw110_pca_frame_read_layer_probe_704f_candidate')}`",
        f"- Best H35 official AW110 PCA-selected frame-path layer probe 704F delta vs H35 704F: `{h35_base.get('best_h35_official_aw110_pca_frame_read_layer_probe_704f_candidate_minus_h35_704_m')}`",
        f"- Best H35 official AW110 PCA-selected DEC00 frame-path beta probe 704F candidate: `{h35_base.get('best_h35_official_aw110_pca_frame_dec00_beta_probe_704f_candidate')}`",
        f"- Best H35 official AW110 PCA-selected DEC00 frame-path beta probe 704F delta vs H35 704F: `{h35_base.get('best_h35_official_aw110_pca_frame_dec00_beta_probe_704f_candidate_minus_h35_704_m')}`",
        f"- Best H35 official AW110 PCA-selected chunk source-soft layer probe 704F candidate: `{h35_base.get('best_h35_official_aw110_pca_chunk_source_soft_read_layer_probe_704f_candidate')}`",
        f"- Best H35 official AW110 PCA-selected chunk source-soft layer probe 704F delta vs H35 704F: `{h35_base.get('best_h35_official_aw110_pca_chunk_source_soft_read_layer_probe_704f_candidate_minus_h35_704_m')}`",
        f"- H35 local Phase4 pass found: `{status.get('h35_local_phase4_pass_found')}`",
        f"- H35 official 704F tested: `{status.get('h35_official_704_tested')}`",
        f"- Phase3-6 blocker: `{phase3_6.get('primary_blocker')}`",
        "",
        "## Required Questions",
        "",
    ]
    for idx, row in enumerate(answers, start=1):
        lines.extend([
            f"{idx}. {row['question']}",
            f"   Answer: {row['answer']}",
            f"   Evidence: {row['evidence']}",
            "",
        ])
    lines.extend([
        "## Artifact Pointers",
        "",
        "- `phase0_c9_experience_lock/`",
        "- `phase1_positive_only_ablation/`",
        "- `phase2_tri_replay_actuator_audit/`",
        "- `phase3_6_semantic_control_evidence/`",
        "- `phase4_h35_base_evidence_lock/`",
        "- `phase4_c9clean_semantic_trireplay_smoke/`",
        "- `phase4_read_ttt_handshake/`",
        "- `phase5_commit_ema_bridge/`",
        "- `phase6_swa_tri_handoff/`",
        "- `v76tf_final_status.json`",
        "- `v76tf_required_questions.csv`",
        "",
    ])
    write_text(out_dir / "v76tf_final_report.md", "\n".join(lines))


def run(root: Path) -> Dict[str, Any]:
    ensure_dir(root)
    phase0 = _load(root / "phase0_c9_experience_lock/phase0_summary.json")
    phase1 = _load(root / "phase1_positive_only_ablation/phase1_positive_only_summary.json")
    phase2 = _load(root / "phase2_tri_replay_actuator_audit/tri_replay_actuator_summary.json")
    phase3_6 = _load(root / "phase3_6_semantic_control_evidence/phase3_6_semantic_control_summary.json")
    h35_base = _load(root / "phase4_h35_base_evidence_lock/h35_base_evidence_summary.json")
    status = _status(phase0, phase1, phase2, phase3_6, h35_base)
    answers = _answer_rows(phase0, phase1, phase2, phase3_6, h35_base, status)
    write_json(root / "v76tf_final_status.json", status)
    from tools.v76tf_common import write_csv

    write_csv(root / "v76tf_required_questions.csv", answers)
    _write_report(root, phase0, phase1, phase2, phase3_6, h35_base, status, answers)
    return {"out_dir": rel(root), **status}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(V76_ROOT))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    result = run(Path(args.root))
    write_json(Path(args.root) / "command_result.json", result)
    if args.strict and not result["phase7_to_phase8_promotion_allowed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
