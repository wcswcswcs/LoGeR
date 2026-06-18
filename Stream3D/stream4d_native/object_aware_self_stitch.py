from __future__ import annotations

from typing import Any

import numpy as np

from .material_tube_roles import MaterialTubeEvidence, TubeRoleScores


def _metrics(selected: list[MaterialTubeEvidence], *, weak_residual_threshold: float = 0.12) -> dict[str, Any]:
    residuals = np.asarray([float(e.self_stitch_residual) for e in selected], dtype=np.float32)
    motions = np.asarray([float(e.motion_magnitude) for e in selected], dtype=np.float32)
    scales = np.asarray([float(e.scale_proxy) for e in selected], dtype=np.float32)
    if selected:
        residual_median = float(np.median(residuals))
        residual_p90 = float(np.percentile(residuals, 90))
        dynamic_leakage = float(np.mean(motions > 0.12))
        scale_drift = float(abs(np.mean(scales) - 1.0))
        weak_alignment = int(np.count_nonzero(residuals > float(weak_residual_threshold)))
        inlier_ratio = float(np.mean(residuals <= float(weak_residual_threshold)))
    else:
        residual_median = None
        residual_p90 = None
        dynamic_leakage = 0.0
        scale_drift = None
        weak_alignment = 0
        inlier_ratio = 0.0
    return {
        "static_anchor_count": int(len(selected)),
        "dynamic_leakage_ratio": dynamic_leakage,
        "self_sim3_residual_median": residual_median,
        "self_sim3_residual_p90": residual_p90,
        "scale_drift": scale_drift,
        "weak_alignment_count": weak_alignment,
        "inlier_ratio": inlier_ratio,
    }


def evaluate_role_aware_stitch_variants(
    evidences: list[MaterialTubeEvidence],
    roles: list[TubeRoleScores],
) -> list[dict[str, Any]]:
    role_by_id = {int(r.tube_id): r for r in roles}
    variants: list[tuple[str, list[MaterialTubeEvidence]]] = []
    variants.append(("D0_all_tubes", list(evidences)))
    variants.append(("D1_confidence_only", [e for e in evidences if e.confidence >= 0.5 and e.visibility >= 0.5]))
    variants.append(
        (
            "D2_semantic_static_role_prior",
            [e for e in evidences if e.semantic_stability >= 0.70 and e.motion_magnitude <= 0.15],
        )
    )
    variants.append(
        (
            "D3_role_posterior_robust_residual",
            [
                e
                for e in evidences
                if (role_by_id.get(int(e.tube_id)) is not None and role_by_id[int(e.tube_id)].role == "scene")
                and e.self_stitch_residual <= 0.12
            ],
        )
    )
    variants.append(
        (
            "D4_dynamic_tubes_negative_control",
            [e for e in evidences if role_by_id.get(int(e.tube_id)) is not None and role_by_id[int(e.tube_id)].role == "object"],
        )
    )
    rows = []
    for variant, selected in variants:
        rows.append({"variant": variant, **_metrics(selected)})
    return rows

