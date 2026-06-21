from __future__ import annotations

from pathlib import Path
from typing import Any

from stream4d_native.v47_common import ROOT, utc_now, write_csv, write_json
from stream4d_native.v51_remask_source_discovery import _inspect_npz_root


PLAN_PATH = "docs/stream4d_v51_r2_mosaic_remask_lift_codex_plan.md"


def build_v51_overlap_mask_bank(input_root: str | Path, max_files: int = 32) -> dict[str, Any]:
    root = ROOT / input_root if not Path(input_root).is_absolute() else Path(input_root)
    row, sample_rows = _inspect_npz_root(root, max_files=max_files)
    mask_count = int(row.get("mask_count_sampled") or 0)
    frame_count = int(row.get("frame_count_sampled") or 0)
    mean_masks = float(row.get("mean_masks_per_frame_sampled") or 0.0)
    containment_count = int(row.get("containment_pair_count_sampled") or 0)
    containment_ratio = float(row.get("containment_pair_ratio_sampled") or 0.0)
    whole_count = int(row.get("whole_candidate_count_sampled") or 0)
    whole_ratio = whole_count / max(mask_count, 1)
    component_coverage = None
    gate = {
        "overlap_capable": bool(row.get("overlap_capable")),
        "preserves_nxhxw_stack": bool(row.get("preserves_nxhxw_stack")),
        "mean_masks_per_frame_pass": mean_masks >= 10.0,
        "containment_pass": containment_count >= 200 or containment_ratio >= 0.02,
        "whole_candidate_pass": whole_count >= 0.20 * max(mask_count, 1),
        "component_coverage_pass": component_coverage is None,
        "component_coverage_not_evaluated": component_coverage is None,
    }
    gate["pass"] = bool(
        gate["overlap_capable"]
        and gate["preserves_nxhxw_stack"]
        and gate["mean_masks_per_frame_pass"]
        and gate["containment_pass"]
        and gate["whole_candidate_pass"]
    )
    return {
        "phase": "v51_r2_overlap_mask_bank",
        "created_at": utc_now(),
        "plan": PLAN_PATH,
        "input_root": str(root),
        "bank_row": row,
        "sample_rows": sample_rows,
        "summary": {
            "frame_count": frame_count,
            "mask_count": mask_count,
            "mean_masks_per_frame": mean_masks,
            "overlap_pair_count": int(row.get("overlap_pair_count_sampled") or 0),
            "containment_pair_count": containment_count,
            "containment_pair_ratio": containment_ratio,
            "whole_candidate_count": whole_count,
            "whole_candidate_ratio": whole_ratio,
            "component_coverage": component_coverage,
            "uses_gt_for_prediction": False,
            "preserves_nxhxw_stack": bool(row.get("preserves_nxhxw_stack")),
        },
        "gate": gate,
    }


def write_v51_overlap_mask_bank(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = ROOT / output_root if not Path(output_root).is_absolute() else Path(output_root)
    write_json(out / "overlap_mask_bank_summary.json", payload)
    write_csv(out / "overlap_mask_bank_rows.csv", [payload.get("bank_row", {})])
    write_csv(out / "overlap_mask_bank_sample_rows.csv", payload.get("sample_rows", []))
