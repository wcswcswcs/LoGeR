from __future__ import annotations

from pathlib import Path
from typing import Any

from stream4d_native.v37_object_field_adapter import read_csv_rows


def run_material_residual(
    stream3d_root: Path,
    *,
    semantic_summary: dict[str, Any],
) -> dict[str, Any]:
    root = Path(stream3d_root)
    material_rows = read_csv_rows(
        root / "outputs/audit/v42_structure_affinity_twohop_backfill8_max480_r1/root_cause/material_split_delta_rows.csv"
    )
    metrics = dict(semantic_summary.get("metrics") or {})
    gate = {
        "material_split_delta_row_count": int(len(material_rows)),
        "real_over_shuffled_residual_ari_delta": None,
        "real_over_no_temporal_residual_ari_delta": None,
        "accepted_merge_precision_delta": None,
        "visible_outside_false_merge_reduction": None,
        "real_beats_shuffled_pass": False,
        "real_beats_no_temporal_pass": False,
        "merge_precision_pass": False,
        "visible_outside_pass": False,
    }
    gate["pass"] = False
    return {
        "phase": "v43_2_material_residual_matching",
        "status": "NO_GO_MATERIAL_NOT_DISCRIMINATIVE",
        "input_semantic_status": semantic_summary.get("status"),
        "accepted_corrections": [],
        "rejected_corrections": [
            {
                "correction": "positive_material_merge",
                "reason": "no residual-candidate material control rows showing real D4RT beats shuffled/no-temporal by required margin",
            }
        ],
        "material_policy_after_failure": "material_may_only_be_used_as_veto_or_support_density_diagnostic",
        "metrics": metrics,
        "gate": gate,
    }
