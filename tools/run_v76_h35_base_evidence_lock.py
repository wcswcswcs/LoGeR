#!/usr/bin/env python3
"""Lock H35-base no-chunk semantic evidence for ACL2 v76 audits."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v76tf_common import (  # noqa: E402
    RESULT_ROOT,
    V46B_REGISTRY,
    V76_ROOT,
    boolish,
    ensure_dir,
    first_row,
    read_csv,
    read_json,
    rel,
    safe_float,
    safe_int,
    write_csv,
    write_json,
    write_text,
)


V56_ROOT = RESULT_ROOT / "acl2_v56_h35_semanticboost_newtttaction_fast/report_final"
V58_ROOT = RESULT_ROOT / "acl2_v58_soft_semantic_read_commit_isolation_geometry/report_final"
H35_GLOBAL_SMOKE_ROOT = V76_ROOT / "phase4_h35_global_l050_smoke/report_R1"
H35_HANDOFF_REPAIR_ROOT = V76_ROOT / "phase4_h35_global_l050_handoff_repair_smoke/report_R1"
H35_SEMREAD_SWEEP_ROOT = V76_ROOT / "phase4_h35_semread_sweep_256f/report_R1"
H35_SEMREAD_LAM_BETA_SWEEP_ROOT = V76_ROOT / "phase4_h35_semread_l075_lam_beta_sweep_256f/report_R1"
H35_SEMREAD_L100_BETA_SWEEP_ROOT = V76_ROOT / "phase4_h35_semread_l100_beta_sweep_256f/report_R1"
H35_L100_B525_704F_ROOT = V76_ROOT / "phase8_h35_l100_b525_704f/report_R1"
H35_OFFICIAL_AW110_704F_ROOT = V76_ROOT / "phase8_h35_l100_b525_official_aw110_704f/report_R1"
H35_OFFICIAL_AW110_REPAIR_704F_ROOT = (
    V76_ROOT / "phase8_h35_l100_b525_official_aw110_704f_repair/report_R1"
)
H35_OFFICIAL_AW110_CALIB_REPAIR_704F_ROOT = (
    V76_ROOT / "phase8_h35_l100_b525_official_aw110_704f_calib_repair/report_R1"
)
H35_OFFICIAL_AW110_STABLE_POSITIVE_REPAIR_704F_ROOT = (
    V76_ROOT / "phase8_h35_l100_b525_official_aw110_704f_stable_positive_repair/report_R1"
)
H35_OFFICIAL_AW110_LOW_LAMBDA_FUSION_REPAIR_704F_ROOT = (
    V76_ROOT / "phase8_h35_official_aw110_low_lambda_fusion_repair/report_R1"
)
H35_OFFICIAL_AW110_ULTRA_LOW_LAMBDA_FUSION_REFINE_704F_ROOT = (
    V76_ROOT / "phase8_h35_official_aw110_ultra_low_lambda_fusion_refine/report_R1"
)
H35_OFFICIAL_AW110_READ_LAYER_MODE_PROBE_704F_ROOT = (
    V76_ROOT / "phase8_h35_official_aw110_read_layer_mode_probe/report_R1"
)
H35_OFFICIAL_AW110_PCA_SINGLE_READ_LAYER_PROBE_704F_ROOT = (
    V76_ROOT / "phase8_h35_official_aw110_pca_selected_single_read_layer_probe/report_R1"
)
H35_OFFICIAL_AW110_PCA_FRAME_READ_LAYER_PROBE_704F_ROOT = (
    V76_ROOT / "phase8_h35_official_aw110_pca_selected_frame_read_layer_probe/report_R1"
)
H35_OFFICIAL_AW110_PCA_FRAME_DEC00_BETA_PROBE_704F_ROOT = (
    V76_ROOT / "phase8_h35_official_aw110_pca_selected_frame_dec00_beta_probe/report_R1"
)
H35_OFFICIAL_AW110_PCA_CHUNK_SOURCE_SOFT_READ_LAYER_PROBE_704F_ROOT = (
    V76_ROOT / "phase8_h35_official_aw110_pca_selected_chunk_source_soft_read_layer_probe/report_R1"
)


def _row_by_name(rows: Iterable[Mapping[str, Any]], *names: str) -> Optional[Mapping[str, Any]]:
    for name in names:
        for key in ("run_name", "row", "name"):
            row = first_row(rows, key, name)
            if row is not None:
                return row
    return None


def _ate(row: Optional[Mapping[str, Any]]) -> Optional[float]:
    if not row:
        return None
    return safe_float(row.get("ATE") or row.get("ATE_full"))


def _frames(row: Optional[Mapping[str, Any]]) -> Optional[int]:
    if not row:
        return None
    return safe_int(row.get("frames"))


def _audit_no_chunk(row: Optional[Mapping[str, Any]]) -> Optional[bool]:
    if not row:
        return None
    if "no_chunk_policy_pass" in row and str(row.get("no_chunk_policy_pass", "")).strip() != "":
        return boolish(row.get("no_chunk_policy_pass"))
    run_dir = row.get("run_dir")
    if not run_dir:
        return None
    audit = read_json(REPO_ROOT / str(run_dir) / "chunk_id_policy_audit.json")
    if not isinstance(audit, dict):
        return None
    keys = (
        "has_read_beta_frame_chunks",
        "has_tri_gamma_chunk_map",
        "has_tri_replay_chunk_params",
        "has_commit_ema_chunks",
        "has_native_mix_chunks",
        "has_semantic_action_active_chunks",
    )
    return not any(boolish(audit.get(key)) for key in keys)


def _entry(
    *,
    family: str,
    artifact: Path,
    row: Optional[Mapping[str, Any]],
    run_name: str,
    h35_ref_ate: Optional[float],
    notes: str,
) -> Dict[str, Any]:
    ate = _ate(row)
    return {
        "family": family,
        "artifact": rel(artifact),
        "run_name": run_name,
        "row": row.get("row") if row else None,
        "candidate": row.get("candidate") if row else None,
        "status": row.get("status") if row else None,
        "frames": _frames(row),
        "ATE": ate,
        "candidate_minus_h35_ref_m": ate - h35_ref_ate if ate is not None and h35_ref_ate is not None else None,
        "no_chunk_policy_pass": _audit_no_chunk(row),
        "manual_percentage_audit_pass": boolish(row.get("manual_percentage_audit_pass")) if row else None,
        "success_pass": boolish(row.get("success_pass")) if row and row.get("success_pass") != "" else None,
        "screen_decision": row.get("screen_decision") if row else None,
        "semantic_desc": row.get("semantic_desc") if row else None,
        "notes": notes,
    }


def run(out_dir: Path) -> Dict[str, Any]:
    ensure_dir(out_dir)

    v46b = read_csv(V46B_REGISTRY)
    v56_h35 = read_csv(V56_ROOT / "v56_h35_reference_registry.csv")
    v56_704 = read_csv(V56_ROOT / "v56_track_a_704f_registry.csv")
    v56_full = read_csv(V56_ROOT / "v56_track_a_full_registry.csv")
    v58_all = read_csv(V58_ROOT / "v58_all_registry.csv")
    h35_global = read_csv(H35_GLOBAL_SMOKE_ROOT / "full_online_registry.csv")
    h35_global_summary = read_json(H35_GLOBAL_SMOKE_ROOT / "v42_full_online_summary.json")
    if not isinstance(h35_global_summary, dict):
        h35_global_summary = {}
    h35_repair = read_csv(H35_HANDOFF_REPAIR_ROOT / "full_online_registry.csv")
    h35_repair_summary = read_json(H35_HANDOFF_REPAIR_ROOT / "v42_full_online_summary.json")
    if not isinstance(h35_repair_summary, dict):
        h35_repair_summary = {}
    h35_semread_sweep = read_csv(H35_SEMREAD_SWEEP_ROOT / "full_online_registry.csv")
    h35_semread_sweep_summary = read_json(H35_SEMREAD_SWEEP_ROOT / "v42_full_online_summary.json")
    if not isinstance(h35_semread_sweep_summary, dict):
        h35_semread_sweep_summary = {}
    h35_semread_lam_beta_sweep = read_csv(H35_SEMREAD_LAM_BETA_SWEEP_ROOT / "full_online_registry.csv")
    h35_semread_lam_beta_sweep_summary = read_json(
        H35_SEMREAD_LAM_BETA_SWEEP_ROOT / "v42_full_online_summary.json"
    )
    if not isinstance(h35_semread_lam_beta_sweep_summary, dict):
        h35_semread_lam_beta_sweep_summary = {}
    h35_semread_l100_beta_sweep = read_csv(H35_SEMREAD_L100_BETA_SWEEP_ROOT / "full_online_registry.csv")
    h35_semread_l100_beta_sweep_summary = read_json(
        H35_SEMREAD_L100_BETA_SWEEP_ROOT / "v42_full_online_summary.json"
    )
    if not isinstance(h35_semread_l100_beta_sweep_summary, dict):
        h35_semread_l100_beta_sweep_summary = {}
    h35_l100_b525_704f = read_csv(H35_L100_B525_704F_ROOT / "full_online_registry.csv")
    h35_l100_b525_704f_summary = read_json(H35_L100_B525_704F_ROOT / "v42_full_online_summary.json")
    if not isinstance(h35_l100_b525_704f_summary, dict):
        h35_l100_b525_704f_summary = {}
    h35_official_aw110_704f = read_csv(H35_OFFICIAL_AW110_704F_ROOT / "full_online_registry.csv")
    h35_official_aw110_704f_summary = read_json(H35_OFFICIAL_AW110_704F_ROOT / "v42_full_online_summary.json")
    if not isinstance(h35_official_aw110_704f_summary, dict):
        h35_official_aw110_704f_summary = {}
    h35_official_aw110_repair_704f = read_csv(
        H35_OFFICIAL_AW110_REPAIR_704F_ROOT / "full_online_registry.csv"
    )
    h35_official_aw110_repair_704f_summary = read_json(
        H35_OFFICIAL_AW110_REPAIR_704F_ROOT / "v42_full_online_summary.json"
    )
    if not isinstance(h35_official_aw110_repair_704f_summary, dict):
        h35_official_aw110_repair_704f_summary = {}
    h35_official_aw110_calib_repair_704f = read_csv(
        H35_OFFICIAL_AW110_CALIB_REPAIR_704F_ROOT / "full_online_registry.csv"
    )
    h35_official_aw110_calib_repair_704f_summary = read_json(
        H35_OFFICIAL_AW110_CALIB_REPAIR_704F_ROOT / "v42_full_online_summary.json"
    )
    if not isinstance(h35_official_aw110_calib_repair_704f_summary, dict):
        h35_official_aw110_calib_repair_704f_summary = {}
    h35_official_aw110_stable_positive_repair_704f = read_csv(
        H35_OFFICIAL_AW110_STABLE_POSITIVE_REPAIR_704F_ROOT / "full_online_registry.csv"
    )
    h35_official_aw110_stable_positive_repair_704f_summary = read_json(
        H35_OFFICIAL_AW110_STABLE_POSITIVE_REPAIR_704F_ROOT / "v42_full_online_summary.json"
    )
    if not isinstance(h35_official_aw110_stable_positive_repair_704f_summary, dict):
        h35_official_aw110_stable_positive_repair_704f_summary = {}
    h35_official_aw110_low_lambda_fusion_repair_704f = read_csv(
        H35_OFFICIAL_AW110_LOW_LAMBDA_FUSION_REPAIR_704F_ROOT / "full_online_registry.csv"
    )
    h35_official_aw110_low_lambda_fusion_repair_704f_summary = read_json(
        H35_OFFICIAL_AW110_LOW_LAMBDA_FUSION_REPAIR_704F_ROOT / "v42_full_online_summary.json"
    )
    if not isinstance(h35_official_aw110_low_lambda_fusion_repair_704f_summary, dict):
        h35_official_aw110_low_lambda_fusion_repair_704f_summary = {}
    h35_official_aw110_ultra_low_lambda_fusion_refine_704f = read_csv(
        H35_OFFICIAL_AW110_ULTRA_LOW_LAMBDA_FUSION_REFINE_704F_ROOT / "full_online_registry.csv"
    )
    h35_official_aw110_ultra_low_lambda_fusion_refine_704f_summary = read_json(
        H35_OFFICIAL_AW110_ULTRA_LOW_LAMBDA_FUSION_REFINE_704F_ROOT / "v42_full_online_summary.json"
    )
    if not isinstance(h35_official_aw110_ultra_low_lambda_fusion_refine_704f_summary, dict):
        h35_official_aw110_ultra_low_lambda_fusion_refine_704f_summary = {}
    h35_official_aw110_read_layer_mode_probe_704f = read_csv(
        H35_OFFICIAL_AW110_READ_LAYER_MODE_PROBE_704F_ROOT / "full_online_registry.csv"
    )
    h35_official_aw110_read_layer_mode_probe_704f_summary = read_json(
        H35_OFFICIAL_AW110_READ_LAYER_MODE_PROBE_704F_ROOT / "v42_full_online_summary.json"
    )
    if not isinstance(h35_official_aw110_read_layer_mode_probe_704f_summary, dict):
        h35_official_aw110_read_layer_mode_probe_704f_summary = {}
    h35_official_aw110_pca_single_read_layer_probe_704f = read_csv(
        H35_OFFICIAL_AW110_PCA_SINGLE_READ_LAYER_PROBE_704F_ROOT / "full_online_registry.csv"
    )
    h35_official_aw110_pca_single_read_layer_probe_704f_summary = read_json(
        H35_OFFICIAL_AW110_PCA_SINGLE_READ_LAYER_PROBE_704F_ROOT / "v42_full_online_summary.json"
    )
    if not isinstance(h35_official_aw110_pca_single_read_layer_probe_704f_summary, dict):
        h35_official_aw110_pca_single_read_layer_probe_704f_summary = {}
    h35_official_aw110_pca_frame_read_layer_probe_704f = read_csv(
        H35_OFFICIAL_AW110_PCA_FRAME_READ_LAYER_PROBE_704F_ROOT / "full_online_registry.csv"
    )
    h35_official_aw110_pca_frame_read_layer_probe_704f_summary = read_json(
        H35_OFFICIAL_AW110_PCA_FRAME_READ_LAYER_PROBE_704F_ROOT / "v42_full_online_summary.json"
    )
    if not isinstance(h35_official_aw110_pca_frame_read_layer_probe_704f_summary, dict):
        h35_official_aw110_pca_frame_read_layer_probe_704f_summary = {}
    h35_official_aw110_pca_frame_dec00_beta_probe_704f = read_csv(
        H35_OFFICIAL_AW110_PCA_FRAME_DEC00_BETA_PROBE_704F_ROOT / "full_online_registry.csv"
    )
    h35_official_aw110_pca_frame_dec00_beta_probe_704f_summary = read_json(
        H35_OFFICIAL_AW110_PCA_FRAME_DEC00_BETA_PROBE_704F_ROOT / "v42_full_online_summary.json"
    )
    if not isinstance(h35_official_aw110_pca_frame_dec00_beta_probe_704f_summary, dict):
        h35_official_aw110_pca_frame_dec00_beta_probe_704f_summary = {}
    h35_official_aw110_pca_chunk_source_soft_read_layer_probe_704f = read_csv(
        H35_OFFICIAL_AW110_PCA_CHUNK_SOURCE_SOFT_READ_LAYER_PROBE_704F_ROOT / "full_online_registry.csv"
    )
    h35_official_aw110_pca_chunk_source_soft_read_layer_probe_704f_summary = read_json(
        H35_OFFICIAL_AW110_PCA_CHUNK_SOURCE_SOFT_READ_LAYER_PROBE_704F_ROOT
        / "v42_full_online_summary.json"
    )
    if not isinstance(h35_official_aw110_pca_chunk_source_soft_read_layer_probe_704f_summary, dict):
        h35_official_aw110_pca_chunk_source_soft_read_layer_probe_704f_summary = {}

    h35_full = _row_by_name(v56_h35, "V53_PHASE7_FULL_H35_LAYERGAMMAFIX_RHO0075", "V56_PHASE0_H35_FULL_REPEAT")
    h35_704 = _row_by_name(v56_h35, "V53_PHASE7_SCREEN_H35_LAYERGAMMAFIX_RHO0075_704F")
    h35_full_ate = _ate(h35_full)
    h35_704_ate = _ate(h35_704)

    rows = [
        _entry(
            family="h35_reference_full",
            artifact=V56_ROOT / "v56_h35_reference_registry.csv",
            row=h35_full,
            run_name="V53_PHASE7_FULL_H35_LAYERGAMMAFIX_RHO0075",
            h35_ref_ate=h35_full_ate,
            notes="Clean H35/v53 full reference; no chunk policy.",
        ),
        _entry(
            family="h35_reference_704f",
            artifact=V56_ROOT / "v56_h35_reference_registry.csv",
            row=h35_704,
            run_name="V53_PHASE7_SCREEN_H35_LAYERGAMMAFIX_RHO0075_704F",
            h35_ref_ate=h35_704_ate,
            notes="Clean H35/v53 704F reference; no chunk policy.",
        ),
        _entry(
            family="h35_v46b_geometry_factorial_full",
            artifact=V46B_REGISTRY,
            row=_row_by_name(v46b, "F110_FRAME_ATTN_TTT"),
            run_name="F110_FRAME_ATTN_TTT",
            h35_ref_ate=_ate(_row_by_name(v46b, "F000_NONE")),
            notes="Clean H35 geometry READ+TTT factorial, semantic-free positive reference.",
        ),
    ]

    for name in (
        "V56_A1_SEM_C23_RESID_704F",
        "V56_A2_HIGH_INFL_ANOM_READ_704F",
        "V56_A3_ANOM_STATIC_RESCUE_704F",
        "V56_A4_SEM_C23_PLUS_ANOM_704F",
    ):
        rows.append(
            _entry(
                family="h35_v56_semantic_704f",
                artifact=V56_ROOT / "v56_track_a_704f_registry.csv",
                row=_row_by_name(v56_704, name),
                run_name=name,
                h35_ref_ate=h35_704_ate,
                notes="H35-base semantic 704F; no chunk policy; used to test correct v76 direction.",
            )
        )

    for name in ("V56_A2_HIGH_INFL_ANOM_READ_FULL", "V56_A3_ANOM_STATIC_RESCUE_FULL"):
        rows.append(
            _entry(
                family="h35_v56_semantic_full",
                artifact=V56_ROOT / "v56_track_a_full_registry.csv",
                row=_row_by_name(v56_full, name),
                run_name=name,
                h35_ref_ate=h35_full_ate,
                notes="H35-base semantic full run promoted from v56; no chunk policy.",
            )
        )

    for name in (
        "V58_N0_RANDOM_SAME_MASS_SOFT_C1_704F",
        "V58_R1_SREAD03_V_ONLY_C1_704F",
        "V58_R2_SREAD03_BIAS_FLOOR_C1_704F",
        "V58_R3_SREAD03_EARLY_ONLY_C1_704F",
    ):
        rows.append(
            _entry(
                family="h35_v58_soft_semantic_704f",
                artifact=V58_ROOT / "v58_all_registry.csv",
                row=_row_by_name(v58_all, name),
                run_name=name,
                h35_ref_ate=h35_704_ate,
                notes="H35-base soft semantic READ/commit-isolation evidence; no chunk policy.",
            )
        )

    h35_global_base = _row_by_name(h35_global, "BASE", "V76_H35G_BASE_256F")
    h35_global_base_ate = _ate(h35_global_base)
    for name in ("BASE", "GEOM", "SEM_READ", "SEM_TRI", "SEM_TRI_N110"):
        rows.append(
            _entry(
                family="h35_v76_global_l050_256f_smoke",
                artifact=H35_GLOBAL_SMOKE_ROOT / "full_online_registry.csv",
                row=_row_by_name(h35_global, name),
                run_name=name,
                h35_ref_ate=h35_global_base_ate,
                notes=(
                    "Current v76 H35-base 256F global-L050 smoke; diagnostic only, "
                    "not 704F/full strict success."
                ),
            )
        )

    h35_repair_base = _row_by_name(h35_repair, "BASE")
    h35_repair_base_ate = _ate(h35_repair_base)
    for name in (
        "NOSWA_G004",
        "NOSWA_G002",
        "NOSWA_G001",
        "NOSWA_G001_NOEMA",
        "SWA_G002_A025",
        "SWA_G001_A025",
    ):
        rows.append(
            _entry(
                family="h35_v76_global_l050_handoff_repair_256f_smoke",
                artifact=H35_HANDOFF_REPAIR_ROOT / "full_online_registry.csv",
                row=_row_by_name(h35_repair, name),
                run_name=name,
                h35_ref_ate=h35_repair_base_ate,
                notes=(
                    "Current v76 H35-base 256F handoff repair smoke; diagnostic only, "
                    "not 704F/full strict success."
                ),
            )
        )

    h35_sweep_base = _row_by_name(h35_semread_sweep, "BASE")
    h35_sweep_base_ate = _ate(h35_sweep_base)
    for name in ("L035_B475", "L045_B475", "L050_B475", "L055_B475", "L060_B475", "L075_B475", "L050_B600"):
        rows.append(
            _entry(
                family="h35_v76_semread_sweep_256f_smoke",
                artifact=H35_SEMREAD_SWEEP_ROOT / "full_online_registry.csv",
                row=_row_by_name(h35_semread_sweep, name),
                run_name=name,
                h35_ref_ate=h35_sweep_base_ate,
                notes=(
                    "Current v76 H35-base 256F semantic READ-only cue/beta sweep; diagnostic only, "
                    "not 704F/full strict success."
                ),
            )
        )

    h35_lam_beta_base = _row_by_name(h35_semread_lam_beta_sweep, "BASE")
    h35_lam_beta_base_ate = _ate(h35_lam_beta_base)
    for name in ("L075_B475", "L075_B350", "L075_B425", "L075_B525", "L075_B600", "L090_B475", "L100_B475"):
        rows.append(
            _entry(
                family="h35_v76_semread_lam_beta_sweep_256f_smoke",
                artifact=H35_SEMREAD_LAM_BETA_SWEEP_ROOT / "full_online_registry.csv",
                row=_row_by_name(h35_semread_lam_beta_sweep, name),
                run_name=name,
                h35_ref_ate=h35_lam_beta_base_ate,
                notes=(
                    "Current v76 H35-base 256F semantic READ-only L075/lambda-beta refinement; "
                    "diagnostic only, not 704F/full strict success."
                ),
            )
        )

    h35_l100_beta_base = _row_by_name(h35_semread_l100_beta_sweep, "BASE")
    h35_l100_beta_base_ate = _ate(h35_l100_beta_base)
    for name in ("L100_B350", "L100_B425", "L100_B475", "L100_B525", "L100_B600", "L100_B750", "L100_B900"):
        rows.append(
            _entry(
                family="h35_v76_semread_l100_beta_sweep_256f_smoke",
                artifact=H35_SEMREAD_L100_BETA_SWEEP_ROOT / "full_online_registry.csv",
                row=_row_by_name(h35_semread_l100_beta_sweep, name),
                run_name=name,
                h35_ref_ate=h35_l100_beta_base_ate,
                notes=(
                    "Current v76 H35-base 256F semantic READ-only L100 beta sweep; "
                    "local smoke only, not 704F/full strict success."
                ),
            )
        )

    h35_l100_b525_704f_base = _row_by_name(h35_l100_b525_704f, "BASE")
    h35_l100_b525_704f_base_ate = _ate(h35_l100_b525_704f_base)
    for name in ("BASE", "L100_B525"):
        rows.append(
            _entry(
                family="h35_v76_l100_b525_simplified_704f",
                artifact=H35_L100_B525_704F_ROOT / "full_online_registry.csv",
                row=_row_by_name(h35_l100_b525_704f, name),
                run_name=name,
                h35_ref_ate=h35_l100_b525_704f_base_ate,
                notes=(
                    "Current v76 704F simplified-runner check. Positive deltas here are baseline-mismatch "
                    "diagnostics only; strict guard must compare against official H35 704F."
                ),
            )
        )

    for name in ("L100_B525_OFFICIAL_AW110",):
        rows.append(
            _entry(
                family="h35_v76_official_aw110_semread_704f",
                artifact=H35_OFFICIAL_AW110_704F_ROOT / "full_online_registry.csv",
                row=_row_by_name(h35_official_aw110_704f, name),
                run_name=name,
                h35_ref_ate=h35_704_ate,
                notes=(
                    "Current v76 official AW110 704F semantic READ candidate; strict guard compares "
                    "against clean H35/v53 704F, not the simplified runner."
                ),
            )
        )

    for name in ("L075_B525", "L090_B475", "L100_B350", "L100_B425", "L100_B600"):
        rows.append(
            _entry(
                family="h35_v76_official_aw110_semread_repair_704f",
                artifact=H35_OFFICIAL_AW110_REPAIR_704F_ROOT / "full_online_registry.csv",
                row=_row_by_name(h35_official_aw110_repair_704f, name),
                run_name=name,
                h35_ref_ate=h35_704_ate,
                notes=(
                    "Current v76 official AW110 704F repair sweep; no chunk policy. Tested whether "
                    "lambda/beta repair preserves the 256F READ gain under official H35 AW110."
                ),
            )
        )

    for name in ("T030_BL050", "T060_BL050", "T060_BL100", "L100_T060_BL050"):
        rows.append(
            _entry(
                family="h35_v76_official_aw110_semread_calib_repair_704f",
                artifact=H35_OFFICIAL_AW110_CALIB_REPAIR_704F_ROOT / "full_online_registry.csv",
                row=_row_by_name(h35_official_aw110_calib_repair_704f, name),
                run_name=name,
                h35_ref_ate=h35_704_ate,
                notes=(
                    "Current v76 official AW110 704F calibration repair sweep; no chunk policy. "
                    "Tests per-frame quantile READ calibration/target-mass repair under official H35 AW110."
                ),
            )
        )

    for name in ("L100_BINANCHOR", "L075_BINANCHOR"):
        rows.append(
            _entry(
                family="h35_v76_official_aw110_semread_stable_positive_repair_704f",
                artifact=H35_OFFICIAL_AW110_STABLE_POSITIVE_REPAIR_704F_ROOT / "full_online_registry.csv",
                row=_row_by_name(h35_official_aw110_stable_positive_repair_704f, name),
                run_name=name,
                h35_ref_ate=h35_704_ate,
                notes=(
                    "Current v76 official AW110 704F stable-positive repair sweep; no chunk policy. "
                    "Tests adaptive_writer_binary_anchor_split, i.e. stable-anchor-only long write with no negative branch."
                ),
            )
        )

    for name in ("L025_B475", "L025_B525", "L050_B475", "L050_B525"):
        rows.append(
            _entry(
                family="h35_v76_official_aw110_semread_low_lambda_fusion_repair_704f",
                artifact=H35_OFFICIAL_AW110_LOW_LAMBDA_FUSION_REPAIR_704F_ROOT / "full_online_registry.csv",
                row=_row_by_name(h35_official_aw110_low_lambda_fusion_repair_704f, name),
                run_name=name,
                h35_ref_ate=h35_704_ate,
                notes=(
                    "Current v76 official AW110 704F low-lambda fusion repair sweep; no chunk policy. "
                    "Tests conservative v31 semantic residual fusion with official H35 D_g cue "
                    "(lambda 0.25/0.50, beta 4.75/5.25)."
                ),
            )
        )

    for name in ("L010_B475", "L010_B525", "L020_B525", "L025_B600"):
        rows.append(
            _entry(
                family="h35_v76_official_aw110_semread_ultra_low_lambda_fusion_refine_704f",
                artifact=H35_OFFICIAL_AW110_ULTRA_LOW_LAMBDA_FUSION_REFINE_704F_ROOT / "full_online_registry.csv",
                row=_row_by_name(h35_official_aw110_ultra_low_lambda_fusion_refine_704f, name),
                run_name=name,
                h35_ref_ate=h35_704_ate,
                notes=(
                    "Current v76 official AW110 704F ultra-low-lambda fusion refinement; no chunk policy. "
                    "Tests whether even weaker semantic residual fusion or beta 6.00 improves over L025_B525."
                ),
            )
        )

    for name in ("L025_B525_RLMIDDLE", "L025_B525_RLALL"):
        rows.append(
            _entry(
                family="h35_v76_official_aw110_semread_read_layer_mode_probe_704f",
                artifact=H35_OFFICIAL_AW110_READ_LAYER_MODE_PROBE_704F_ROOT / "full_online_registry.csv",
                row=_row_by_name(h35_official_aw110_read_layer_mode_probe_704f, name),
                run_name=name,
                h35_ref_ate=h35_704_ate,
                notes=(
                    "Current v76 official AW110 704F read-layer-mode probe; no chunk policy. "
                    "Keeps L025_B525 fixed and changes semantic READ placement to middle/all."
                ),
            )
        )

    for name in ("DEC09_CTRL", "DEC11_PCAK05", "DEC13_CTRL", "DEC15_PCAK07"):
        rows.append(
            _entry(
                family="h35_v76_official_aw110_semread_pca_single_read_layer_probe_704f",
                artifact=H35_OFFICIAL_AW110_PCA_SINGLE_READ_LAYER_PROBE_704F_ROOT / "full_online_registry.csv",
                row=_row_by_name(h35_official_aw110_pca_single_read_layer_probe_704f, name),
                run_name=name,
                h35_ref_ate=h35_704_ate,
                notes=(
                    "Current v76 official AW110 704F PCA-selected single read-layer probe; no chunk policy. "
                    "Uses visual layer PCA to test decoder 11/15 and adjacent controls 9/13."
                ),
            )
        )

    for name in ("DEC00_KEYL00", "DEC10_KEYL05", "DEC12_KEYL06", "DEC16_QUERYL08"):
        rows.append(
            _entry(
                family="h35_v76_official_aw110_semread_pca_frame_read_layer_probe_704f",
                artifact=H35_OFFICIAL_AW110_PCA_FRAME_READ_LAYER_PROBE_704F_ROOT / "full_online_registry.csv",
                row=_row_by_name(h35_official_aw110_pca_frame_read_layer_probe_704f, name),
                run_name=name,
                h35_ref_ate=h35_704_ate,
                notes=(
                    "Current v76 official AW110 704F PCA-selected frame-path single read-layer probe; no chunk policy. "
                    "Uses visually reviewed frame-attention PCA layers and verifies frame attention bias is active."
                ),
            )
        )

    for name in ("DEC00_KEYL00_B475", "DEC00_KEYL00_B600"):
        rows.append(
            _entry(
                family="h35_v76_official_aw110_semread_pca_frame_dec00_beta_probe_704f",
                artifact=H35_OFFICIAL_AW110_PCA_FRAME_DEC00_BETA_PROBE_704F_ROOT
                / "full_online_registry.csv",
                row=_row_by_name(h35_official_aw110_pca_frame_dec00_beta_probe_704f, name),
                run_name=name,
                h35_ref_ate=h35_704_ate,
                notes=(
                    "Current v76 official AW110 704F DEC00 frame-path beta probe; no chunk policy. "
                    "Uses the visually strongest frame-attention key L00 layer and tests beta 4.75/6.00."
                ),
            )
        )

    for name in ("DEC09_CTRL_CHUNKSS", "DEC11_PCAK05_CHUNKSS", "DEC13_CTRL_CHUNKSS", "DEC15_PCAK07_CHUNKSS"):
        rows.append(
            _entry(
                family="h35_v76_official_aw110_semread_pca_chunk_source_soft_read_layer_probe_704f",
                artifact=H35_OFFICIAL_AW110_PCA_CHUNK_SOURCE_SOFT_READ_LAYER_PROBE_704F_ROOT
                / "full_online_registry.csv",
                row=_row_by_name(h35_official_aw110_pca_chunk_source_soft_read_layer_probe_704f, name),
                run_name=name,
                h35_ref_ate=h35_704_ate,
                notes=(
                    "Current v76 official AW110 704F PCA-selected chunk/global source-soft read-layer probe; "
                    "no chunk policy. Tests repaired compact chunk attention actuator on visual K-L05/K-L07 "
                    "layers and adjacent controls."
                ),
            )
        )

    semantic_rows = [
        row
        for row in rows
        if row["family"] in {"h35_v56_semantic_704f", "h35_v56_semantic_full", "h35_v58_soft_semantic_704f"}
        and row.get("ATE") is not None
    ]
    best_704_delta = min(
        (
            row["candidate_minus_h35_ref_m"]
            for row in semantic_rows
            if row["family"] != "h35_v56_semantic_full" and row.get("candidate_minus_h35_ref_m") is not None
        ),
        default=None,
    )
    best_full_delta = min(
        (
            row["candidate_minus_h35_ref_m"]
            for row in semantic_rows
            if row["family"] == "h35_v56_semantic_full" and row.get("candidate_minus_h35_ref_m") is not None
        ),
        default=None,
    )
    smoke_rows = [
        row
        for row in rows
        if row["family"] == "h35_v76_global_l050_256f_smoke"
        and row.get("ATE") is not None
        and row.get("run_name") != "BASE"
    ]
    best_smoke = min(smoke_rows, key=lambda row: row["candidate_minus_h35_ref_m"], default=None)
    smoke_by_name = {row.get("run_name"): row for row in smoke_rows}
    repair_rows = [
        row
        for row in rows
        if row["family"] == "h35_v76_global_l050_handoff_repair_256f_smoke"
        and row.get("ATE") is not None
    ]
    best_repair = min(repair_rows, key=lambda row: row["candidate_minus_h35_ref_m"], default=None)
    sem_read_ate = smoke_by_name.get("SEM_READ", {}).get("ATE") if smoke_by_name else None
    semread_sweep_rows = [
        row
        for row in rows
        if row["family"] == "h35_v76_semread_sweep_256f_smoke"
        and row.get("ATE") is not None
    ]
    best_semread_sweep = min(
        semread_sweep_rows,
        key=lambda row: row["candidate_minus_h35_ref_m"],
        default=None,
    )
    semread_lam_beta_rows = [
        row
        for row in rows
        if row["family"] == "h35_v76_semread_lam_beta_sweep_256f_smoke"
        and row.get("ATE") is not None
    ]
    best_semread_lam_beta = min(
        semread_lam_beta_rows,
        key=lambda row: row["candidate_minus_h35_ref_m"],
        default=None,
    )
    semread_l100_beta_rows = [
        row
        for row in rows
        if row["family"] == "h35_v76_semread_l100_beta_sweep_256f_smoke"
        and row.get("ATE") is not None
    ]
    best_semread_l100_beta = min(
        semread_l100_beta_rows,
        key=lambda row: row["candidate_minus_h35_ref_m"],
        default=None,
    )
    simplified_704_rows = [
        row
        for row in rows
        if row["family"] == "h35_v76_l100_b525_simplified_704f"
        and row.get("ATE") is not None
        and row.get("run_name") != "BASE"
    ]
    best_simplified_704 = min(
        simplified_704_rows,
        key=lambda row: row["candidate_minus_h35_ref_m"],
        default=None,
    )
    official_aw110_704_rows = [
        row
        for row in rows
        if row["family"] == "h35_v76_official_aw110_semread_704f"
        and row.get("ATE") is not None
    ]
    best_official_aw110_704 = min(
        official_aw110_704_rows,
        key=lambda row: row["candidate_minus_h35_ref_m"],
        default=None,
    )
    official_aw110_repair_rows = [
        row
        for row in rows
        if row["family"] == "h35_v76_official_aw110_semread_repair_704f"
        and row.get("ATE") is not None
    ]
    best_official_aw110_repair = min(
        official_aw110_repair_rows,
        key=lambda row: row["candidate_minus_h35_ref_m"],
        default=None,
    )
    official_aw110_calib_repair_rows = [
        row
        for row in rows
        if row["family"] == "h35_v76_official_aw110_semread_calib_repair_704f"
        and row.get("ATE") is not None
    ]
    best_official_aw110_calib_repair = min(
        official_aw110_calib_repair_rows,
        key=lambda row: row["candidate_minus_h35_ref_m"],
        default=None,
    )
    official_aw110_stable_positive_repair_rows = [
        row
        for row in rows
        if row["family"] == "h35_v76_official_aw110_semread_stable_positive_repair_704f"
        and row.get("ATE") is not None
    ]
    best_official_aw110_stable_positive_repair = min(
        official_aw110_stable_positive_repair_rows,
        key=lambda row: row["candidate_minus_h35_ref_m"],
        default=None,
    )
    official_aw110_low_lambda_fusion_repair_rows = [
        row
        for row in rows
        if row["family"] == "h35_v76_official_aw110_semread_low_lambda_fusion_repair_704f"
        and row.get("ATE") is not None
    ]
    best_official_aw110_low_lambda_fusion_repair = min(
        official_aw110_low_lambda_fusion_repair_rows,
        key=lambda row: row["candidate_minus_h35_ref_m"],
        default=None,
    )
    official_aw110_ultra_low_lambda_fusion_refine_rows = [
        row
        for row in rows
        if row["family"] == "h35_v76_official_aw110_semread_ultra_low_lambda_fusion_refine_704f"
        and row.get("ATE") is not None
    ]
    best_official_aw110_ultra_low_lambda_fusion_refine = min(
        official_aw110_ultra_low_lambda_fusion_refine_rows,
        key=lambda row: row["candidate_minus_h35_ref_m"],
        default=None,
    )
    official_aw110_read_layer_mode_probe_rows = [
        row
        for row in rows
        if row["family"] == "h35_v76_official_aw110_semread_read_layer_mode_probe_704f"
        and row.get("ATE") is not None
    ]
    best_official_aw110_read_layer_mode_probe = min(
        official_aw110_read_layer_mode_probe_rows,
        key=lambda row: row["candidate_minus_h35_ref_m"],
        default=None,
    )
    official_aw110_pca_single_read_layer_probe_rows = [
        row
        for row in rows
        if row["family"] == "h35_v76_official_aw110_semread_pca_single_read_layer_probe_704f"
        and row.get("ATE") is not None
    ]
    best_official_aw110_pca_single_read_layer_probe = min(
        official_aw110_pca_single_read_layer_probe_rows,
        key=lambda row: row["candidate_minus_h35_ref_m"],
        default=None,
    )
    official_aw110_pca_frame_read_layer_probe_rows = [
        row
        for row in rows
        if row["family"] == "h35_v76_official_aw110_semread_pca_frame_read_layer_probe_704f"
        and row.get("ATE") is not None
    ]
    best_official_aw110_pca_frame_read_layer_probe = min(
        official_aw110_pca_frame_read_layer_probe_rows,
        key=lambda row: row["candidate_minus_h35_ref_m"],
        default=None,
    )
    official_aw110_pca_frame_dec00_beta_probe_rows = [
        row
        for row in rows
        if row["family"] == "h35_v76_official_aw110_semread_pca_frame_dec00_beta_probe_704f"
        and row.get("ATE") is not None
    ]
    best_official_aw110_pca_frame_dec00_beta_probe = min(
        official_aw110_pca_frame_dec00_beta_probe_rows,
        key=lambda row: row["candidate_minus_h35_ref_m"],
        default=None,
    )
    official_aw110_pca_chunk_source_soft_read_layer_probe_rows = [
        row
        for row in rows
        if row["family"] == "h35_v76_official_aw110_semread_pca_chunk_source_soft_read_layer_probe_704f"
        and row.get("ATE") is not None
    ]
    best_official_aw110_pca_chunk_source_soft_read_layer_probe = min(
        official_aw110_pca_chunk_source_soft_read_layer_probe_rows,
        key=lambda row: row["candidate_minus_h35_ref_m"],
        default=None,
    )
    strict_success_families = {
        "h35_v56_semantic_704f",
        "h35_v56_semantic_full",
        "h35_v58_soft_semantic_704f",
        "h35_v76_official_aw110_semread_704f",
        "h35_v76_official_aw110_semread_repair_704f",
        "h35_v76_official_aw110_semread_calib_repair_704f",
        "h35_v76_official_aw110_semread_stable_positive_repair_704f",
        "h35_v76_official_aw110_semread_low_lambda_fusion_repair_704f",
        "h35_v76_official_aw110_semread_ultra_low_lambda_fusion_refine_704f",
        "h35_v76_official_aw110_semread_read_layer_mode_probe_704f",
        "h35_v76_official_aw110_semread_pca_single_read_layer_probe_704f",
        "h35_v76_official_aw110_semread_pca_frame_read_layer_probe_704f",
        "h35_v76_official_aw110_semread_pca_frame_dec00_beta_probe_704f",
        "h35_v76_official_aw110_semread_pca_chunk_source_soft_read_layer_probe_704f",
    }
    strict_success = any(
        row.get("candidate_minus_h35_ref_m") is not None
        and row["candidate_minus_h35_ref_m"] < 0
        and row.get("no_chunk_policy_pass") is True
        and row.get("success_pass") is True
        and row["family"] in strict_success_families
        for row in rows
    )
    summary = {
        "h35_base_evidence_available": bool(rows),
        "h35_reference_full_ATE": h35_full_ate,
        "h35_reference_704_ATE": h35_704_ate,
        "h35_base_semantic_rows": len(semantic_rows),
        "best_h35_base_semantic_704_candidate_minus_h35_ref_m": best_704_delta,
        "best_h35_base_semantic_full_candidate_minus_h35_ref_m": best_full_delta,
        "h35_global_l050_256f_available": bool(h35_global),
        "h35_global_l050_256f_phase4_allowed": h35_global_summary.get("phase4_allowed"),
        "best_h35_global_l050_256f_candidate": best_smoke.get("run_name") if best_smoke else None,
        "best_h35_global_l050_256f_candidate_minus_base_m": (
            best_smoke.get("candidate_minus_h35_ref_m") if best_smoke else None
        ),
        "h35_global_l050_sem_tri_swa_minus_base_m": (
            smoke_by_name.get("SEM_TRI", {}).get("candidate_minus_h35_ref_m") if smoke_by_name else None
        ),
        "h35_global_l050_sem_tri_swa_native110_minus_base_m": (
            smoke_by_name.get("SEM_TRI_N110", {}).get("candidate_minus_h35_ref_m") if smoke_by_name else None
        ),
        "h35_global_l050_handoff_repair_256f_available": bool(h35_repair),
        "h35_global_l050_handoff_repair_phase4_allowed": h35_repair_summary.get("phase4_allowed"),
        "best_h35_global_l050_handoff_repair_candidate": best_repair.get("run_name") if best_repair else None,
        "best_h35_global_l050_handoff_repair_candidate_minus_base_m": (
            best_repair.get("candidate_minus_h35_ref_m") if best_repair else None
        ),
        "best_h35_global_l050_handoff_repair_candidate_minus_sem_read_m": (
            best_repair.get("ATE") - sem_read_ate
            if best_repair and isinstance(sem_read_ate, (int, float))
            else None
        ),
        "h35_semread_sweep_256f_available": bool(h35_semread_sweep),
        "h35_semread_sweep_phase4_allowed": h35_semread_sweep_summary.get("phase4_allowed"),
        "best_h35_semread_sweep_256f_candidate": best_semread_sweep.get("run_name") if best_semread_sweep else None,
        "best_h35_semread_sweep_256f_candidate_minus_base_m": (
            best_semread_sweep.get("candidate_minus_h35_ref_m") if best_semread_sweep else None
        ),
        "h35_semread_lam_beta_sweep_256f_available": bool(h35_semread_lam_beta_sweep),
        "h35_semread_lam_beta_sweep_phase4_allowed": h35_semread_lam_beta_sweep_summary.get("phase4_allowed"),
        "best_h35_semread_lam_beta_sweep_256f_candidate": (
            best_semread_lam_beta.get("run_name") if best_semread_lam_beta else None
        ),
        "best_h35_semread_lam_beta_sweep_256f_candidate_minus_base_m": (
            best_semread_lam_beta.get("candidate_minus_h35_ref_m") if best_semread_lam_beta else None
        ),
        "h35_semread_l100_beta_sweep_256f_available": bool(h35_semread_l100_beta_sweep),
        "h35_semread_l100_beta_sweep_phase4_allowed": h35_semread_l100_beta_sweep_summary.get("phase4_allowed"),
        "h35_semread_l100_beta_sweep_h35_local_phase4_allowed": h35_semread_l100_beta_sweep_summary.get(
            "h35_local_phase4_allowed"
        ),
        "best_h35_semread_l100_beta_sweep_256f_candidate": (
            best_semread_l100_beta.get("run_name") if best_semread_l100_beta else None
        ),
        "best_h35_semread_l100_beta_sweep_256f_candidate_minus_base_m": (
            best_semread_l100_beta.get("candidate_minus_h35_ref_m") if best_semread_l100_beta else None
        ),
        "h35_l100_b525_simplified_704f_available": bool(h35_l100_b525_704f),
        "h35_l100_b525_simplified_704f_phase4_allowed": h35_l100_b525_704f_summary.get("phase4_allowed"),
        "best_h35_l100_b525_simplified_704f_candidate": (
            best_simplified_704.get("run_name") if best_simplified_704 else None
        ),
        "best_h35_l100_b525_simplified_704f_candidate_minus_base_m": (
            best_simplified_704.get("candidate_minus_h35_ref_m") if best_simplified_704 else None
        ),
        "h35_official_aw110_l100_b525_704f_available": bool(h35_official_aw110_704f),
        "h35_official_aw110_l100_b525_704f_phase4_allowed": h35_official_aw110_704f_summary.get(
            "phase4_allowed"
        ),
        "best_h35_official_aw110_l100_b525_704f_candidate": (
            best_official_aw110_704.get("run_name") if best_official_aw110_704 else None
        ),
        "best_h35_official_aw110_l100_b525_704f_candidate_minus_h35_704_m": (
            best_official_aw110_704.get("candidate_minus_h35_ref_m") if best_official_aw110_704 else None
        ),
        "h35_official_aw110_repair_704f_available": bool(h35_official_aw110_repair_704f),
        "h35_official_aw110_repair_704f_phase4_allowed": h35_official_aw110_repair_704f_summary.get(
            "phase4_allowed"
        ),
        "best_h35_official_aw110_repair_704f_candidate": (
            best_official_aw110_repair.get("run_name") if best_official_aw110_repair else None
        ),
        "best_h35_official_aw110_repair_704f_candidate_minus_h35_704_m": (
            best_official_aw110_repair.get("candidate_minus_h35_ref_m") if best_official_aw110_repair else None
        ),
        "h35_official_aw110_calib_repair_704f_available": bool(h35_official_aw110_calib_repair_704f),
        "h35_official_aw110_calib_repair_704f_phase4_allowed": (
            h35_official_aw110_calib_repair_704f_summary.get("phase4_allowed")
        ),
        "best_h35_official_aw110_calib_repair_704f_candidate": (
            best_official_aw110_calib_repair.get("run_name") if best_official_aw110_calib_repair else None
        ),
        "best_h35_official_aw110_calib_repair_704f_candidate_minus_h35_704_m": (
            best_official_aw110_calib_repair.get("candidate_minus_h35_ref_m")
            if best_official_aw110_calib_repair
            else None
        ),
        "h35_official_aw110_stable_positive_repair_704f_available": bool(
            h35_official_aw110_stable_positive_repair_704f
        ),
        "h35_official_aw110_stable_positive_repair_704f_phase4_allowed": (
            h35_official_aw110_stable_positive_repair_704f_summary.get("phase4_allowed")
        ),
        "best_h35_official_aw110_stable_positive_repair_704f_candidate": (
            best_official_aw110_stable_positive_repair.get("run_name")
            if best_official_aw110_stable_positive_repair
            else None
        ),
        "best_h35_official_aw110_stable_positive_repair_704f_candidate_minus_h35_704_m": (
            best_official_aw110_stable_positive_repair.get("candidate_minus_h35_ref_m")
            if best_official_aw110_stable_positive_repair
            else None
        ),
        "h35_official_aw110_low_lambda_fusion_repair_704f_available": bool(
            h35_official_aw110_low_lambda_fusion_repair_704f
        ),
        "h35_official_aw110_low_lambda_fusion_repair_704f_phase4_allowed": (
            h35_official_aw110_low_lambda_fusion_repair_704f_summary.get("phase4_allowed")
        ),
        "best_h35_official_aw110_low_lambda_fusion_repair_704f_candidate": (
            best_official_aw110_low_lambda_fusion_repair.get("run_name")
            if best_official_aw110_low_lambda_fusion_repair
            else None
        ),
        "best_h35_official_aw110_low_lambda_fusion_repair_704f_candidate_minus_h35_704_m": (
            best_official_aw110_low_lambda_fusion_repair.get("candidate_minus_h35_ref_m")
            if best_official_aw110_low_lambda_fusion_repair
            else None
        ),
        "h35_official_aw110_ultra_low_lambda_fusion_refine_704f_available": bool(
            h35_official_aw110_ultra_low_lambda_fusion_refine_704f
        ),
        "h35_official_aw110_ultra_low_lambda_fusion_refine_704f_phase4_allowed": (
            h35_official_aw110_ultra_low_lambda_fusion_refine_704f_summary.get("phase4_allowed")
        ),
        "best_h35_official_aw110_ultra_low_lambda_fusion_refine_704f_candidate": (
            best_official_aw110_ultra_low_lambda_fusion_refine.get("run_name")
            if best_official_aw110_ultra_low_lambda_fusion_refine
            else None
        ),
        "best_h35_official_aw110_ultra_low_lambda_fusion_refine_704f_candidate_minus_h35_704_m": (
            best_official_aw110_ultra_low_lambda_fusion_refine.get("candidate_minus_h35_ref_m")
            if best_official_aw110_ultra_low_lambda_fusion_refine
            else None
        ),
        "h35_official_aw110_read_layer_mode_probe_704f_available": bool(
            h35_official_aw110_read_layer_mode_probe_704f
        ),
        "h35_official_aw110_read_layer_mode_probe_704f_phase4_allowed": (
            h35_official_aw110_read_layer_mode_probe_704f_summary.get("phase4_allowed")
        ),
        "best_h35_official_aw110_read_layer_mode_probe_704f_candidate": (
            best_official_aw110_read_layer_mode_probe.get("run_name")
            if best_official_aw110_read_layer_mode_probe
            else None
        ),
        "best_h35_official_aw110_read_layer_mode_probe_704f_candidate_minus_h35_704_m": (
            best_official_aw110_read_layer_mode_probe.get("candidate_minus_h35_ref_m")
            if best_official_aw110_read_layer_mode_probe
            else None
        ),
        "h35_official_aw110_pca_single_read_layer_probe_704f_available": bool(
            h35_official_aw110_pca_single_read_layer_probe_704f
        ),
        "h35_official_aw110_pca_single_read_layer_probe_704f_phase4_allowed": (
            h35_official_aw110_pca_single_read_layer_probe_704f_summary.get("phase4_allowed")
        ),
        "best_h35_official_aw110_pca_single_read_layer_probe_704f_candidate": (
            best_official_aw110_pca_single_read_layer_probe.get("run_name")
            if best_official_aw110_pca_single_read_layer_probe
            else None
        ),
        "best_h35_official_aw110_pca_single_read_layer_probe_704f_candidate_minus_h35_704_m": (
            best_official_aw110_pca_single_read_layer_probe.get("candidate_minus_h35_ref_m")
            if best_official_aw110_pca_single_read_layer_probe
            else None
        ),
        "h35_official_aw110_pca_frame_read_layer_probe_704f_available": bool(
            h35_official_aw110_pca_frame_read_layer_probe_704f
        ),
        "h35_official_aw110_pca_frame_read_layer_probe_704f_phase4_allowed": (
            h35_official_aw110_pca_frame_read_layer_probe_704f_summary.get("phase4_allowed")
        ),
        "best_h35_official_aw110_pca_frame_read_layer_probe_704f_candidate": (
            best_official_aw110_pca_frame_read_layer_probe.get("run_name")
            if best_official_aw110_pca_frame_read_layer_probe
            else None
        ),
        "best_h35_official_aw110_pca_frame_read_layer_probe_704f_candidate_minus_h35_704_m": (
            best_official_aw110_pca_frame_read_layer_probe.get("candidate_minus_h35_ref_m")
            if best_official_aw110_pca_frame_read_layer_probe
            else None
        ),
        "h35_official_aw110_pca_frame_dec00_beta_probe_704f_available": bool(
            h35_official_aw110_pca_frame_dec00_beta_probe_704f
        ),
        "h35_official_aw110_pca_frame_dec00_beta_probe_704f_phase4_allowed": (
            h35_official_aw110_pca_frame_dec00_beta_probe_704f_summary.get("phase4_allowed")
        ),
        "best_h35_official_aw110_pca_frame_dec00_beta_probe_704f_candidate": (
            best_official_aw110_pca_frame_dec00_beta_probe.get("run_name")
            if best_official_aw110_pca_frame_dec00_beta_probe
            else None
        ),
        "best_h35_official_aw110_pca_frame_dec00_beta_probe_704f_candidate_minus_h35_704_m": (
            best_official_aw110_pca_frame_dec00_beta_probe.get("candidate_minus_h35_ref_m")
            if best_official_aw110_pca_frame_dec00_beta_probe
            else None
        ),
        "h35_official_aw110_pca_chunk_source_soft_read_layer_probe_704f_available": bool(
            h35_official_aw110_pca_chunk_source_soft_read_layer_probe_704f
        ),
        "h35_official_aw110_pca_chunk_source_soft_read_layer_probe_704f_phase4_allowed": (
            h35_official_aw110_pca_chunk_source_soft_read_layer_probe_704f_summary.get("phase4_allowed")
        ),
        "best_h35_official_aw110_pca_chunk_source_soft_read_layer_probe_704f_candidate": (
            best_official_aw110_pca_chunk_source_soft_read_layer_probe.get("run_name")
            if best_official_aw110_pca_chunk_source_soft_read_layer_probe
            else None
        ),
        "best_h35_official_aw110_pca_chunk_source_soft_read_layer_probe_704f_candidate_minus_h35_704_m": (
            best_official_aw110_pca_chunk_source_soft_read_layer_probe.get("candidate_minus_h35_ref_m")
            if best_official_aw110_pca_chunk_source_soft_read_layer_probe
            else None
        ),
        "h35_base_strict_semantic_success_found": strict_success,
        "h35_base_conclusion": (
            "Existing H35-base no-chunk semantic evidence is available, but it does not beat the clean H35/v53 "
            "reference under the recorded success gates; C9 remains diagnostic only."
        ),
    }

    write_csv(out_dir / "h35_base_evidence_rows.csv", rows)
    write_json(out_dir / "h35_base_evidence_summary.json", summary)
    _write_report(out_dir, rows, summary)
    return {"out_dir": rel(out_dir), **summary}


def _write_report(out_dir: Path, rows: list[Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    lines = [
        "# v76 H35-Base Evidence Lock",
        "",
        "This audit locks the corrected success base: Clean H35/v53. C9/C9-clean/dechunk evidence is diagnostic only.",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| family | run | frames | ATE | candidate_minus_h35_ref_m | no_chunk_policy_pass | success_pass | notes |",
            "|---|---|---:|---:|---:|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| `{family}` | `{run_name}` | {frames} | {ate} | {delta} | {chunk} | {success} | {notes} |".format(
                family=row.get("family"),
                run_name=row.get("run_name"),
                frames=row.get("frames"),
                ate=row.get("ATE"),
                delta=row.get("candidate_minus_h35_ref_m"),
                chunk=row.get("no_chunk_policy_pass"),
                success=row.get("success_pass"),
                notes=row.get("notes"),
            )
        )
    lines.append("")
    write_text(out_dir / "h35_base_evidence_report.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(V76_ROOT / "phase4_h35_base_evidence_lock"))
    args = parser.parse_args()
    result = run(Path(args.out_dir))
    write_json(Path(args.out_dir) / "command_result.json", result)


if __name__ == "__main__":
    main()
