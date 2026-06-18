from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def _region_row(rows: list[dict[str, str]], source: str) -> dict[str, str] | None:
    return next((row for row in rows if row.get("source") == source), None)


def _assignment_row(rows: list[dict[str, str]], variant: str) -> dict[str, str] | None:
    return next((row for row in rows if row.get("variant") == variant and row.get("scene") == "ALL"), None)


def _source_from_region(row: dict[str, str] | None, source: str, label: str, note: str) -> dict[str, Any]:
    if row is None:
        return {
            "source": source,
            "label": label,
            "source_available": False,
            "not_run_reason": "source metrics missing",
            "note": note,
        }
    mixed = _float(row, "mixed_region_rate")
    scene_mixed = _float(row, "scene0081_mixed_region_rate")
    out = {
        "source": source,
        "label": label,
        "source_available": True,
        "region_count": int(float(row.get("region_count") or 0)),
        "masklet_purity_proxy": None if mixed is None else float(1.0 - mixed),
        "mixed_masklet_rate_proxy": mixed,
        "masklet_completeness_proxy": _float(row, "GT_object_coverage@0.10"),
        "GT_object_coverage@0.10": _float(row, "GT_object_coverage@0.10"),
        "GT_object_coverage@0.25": _float(row, "GT_object_coverage@0.25"),
        "same_frame_cannot_link_violation": _float(row, "same_region_diff_GT_ratio"),
        "scene0081_masklet_purity_proxy": None if scene_mixed is None else float(1.0 - scene_mixed),
        "scene0081_masklet_coverage": _float(row, "scene0081_GT_object_coverage@0.10"),
        "note": note,
    }
    out["phaseC_source_gate_pass"] = bool(
        (out["masklet_purity_proxy"] or 0.0) >= 0.85
        and (out["masklet_completeness_proxy"] or 0.0) >= 0.45
        and (out["GT_object_coverage@0.10"] or 0.0) >= 0.75
        and (out["same_frame_cannot_link_violation"] or 1.0) <= 0.05
        and (out["scene0081_masklet_purity_proxy"] or 0.0) >= 0.80
    )
    return out


def _source_not_run(source: str, label: str, reason: str, note: str) -> dict[str, Any]:
    return {
        "source": source,
        "label": label,
        "source_available": False,
        "not_run_reason": reason,
        "phaseC_source_gate_pass": False,
        "note": note,
    }


def _masklet_row(row: dict[str, str] | None, label: str, note: str) -> dict[str, Any]:
    if row is None:
        return {"label": label, "source_available": False, "not_run_reason": "assignment row missing", "note": note}
    out = {
        "label": label,
        "source_available": True,
        "ARI": _float(row, "ARI"),
        "masklet_purity": _float(row, "purity"),
        "masklet_completeness": _float(row, "completeness"),
        "unknown_tube_ratio": _float(row, "unknown_tube_ratio"),
        "scene0081_masklet_ARI": _float(row, "scene0081_ARI"),
        "masklet_count": int(float(row.get("masklet_count") or 0)),
        "temporal_span_mean": _float(row, "temporal_span_mean"),
        "same_frame_cannot_link_violations": int(float(row.get("same_frame_cannot_link_violations") or 0)),
        "note": note,
    }
    out["phaseC_masklet_gate_pass"] = bool(
        (out["masklet_purity"] or 0.0) >= 0.85
        and (out["masklet_completeness"] or 0.0) >= 0.45
        and out["same_frame_cannot_link_violations"] == 0
    )
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.stream3d_root).resolve()
    source_root = root / args.source_root
    output_root = root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    region_metrics = _read_csv(source_root / "v36_region_sources/region_source_metrics.csv")
    dino_quality = _read_json(source_root / "v36_dino_features/dino_feature_quality.json")
    assignment_rows = _read_csv(source_root / "v36_masklet_graph/tube_assignment_summary.csv")
    masklet_graph_rows = _read_csv(source_root / "v36_masklet_graph/masklet_graph_summary.csv")
    v36_summary = _read_json(source_root / "v36_masklet_first_summary.json")

    source_rows = [
        _source_from_region(
            _region_row(region_metrics, "R0_current_cropformer"),
            "C0_current_prepared_masks",
            "current prepared masks",
            "v39 C0 imported from v36 R0_current_cropformer.",
        ),
        _source_from_region(
            _region_row(region_metrics, "R1_boundary_watershed"),
            "C1_watershed_split_inside_masks",
            "boundary/watershed split",
            "Plan-directed purity repair: split regions more aggressively by boundary/watershed.",
        ),
        _source_not_run(
            "C2_DINO_feature_split_inside_masks",
            "DINO feature split",
            "no full-frame DINO split artifact found; only DINO diagnostic AUC is available",
            "DINO diagnostics are recorded separately; no C2 predictions are claimed.",
        ),
        _source_not_run(
            "C3_SAM_EfficientSAM_SAM3",
            "SAM/EfficientSAM/SAM3",
            "checkpoint_missing_or_integration_failed",
            "No usable SAM/EfficientSAM/SAM3 integration artifact found in current v39 state.",
        ),
        _source_from_region(
            _region_row(region_metrics, "R4_hybrid_split"),
            "C4_hybrid_masks_dino_watershed_regions",
            "hybrid split",
            "Plan-directed repair: combine masks with stronger split variants from prior v36/v37 lineage.",
        ),
        _source_from_region(
            _region_row(region_metrics, "R6_hybrid_union"),
            "C5_current_best_hybrid_diagnostic",
            "current best hybrid diagnostic baseline",
            "Diagnostic-only imported baseline; not a v39 method success.",
        ),
    ]

    masklet_rows = [
        _masklet_row(_assignment_row(assignment_rows, "D0_greedy_temporal_R1"), "D0_greedy_temporal_R1", "watershed temporal grouping"),
        _masklet_row(_assignment_row(assignment_rows, "D1_correlation_R4"), "D1_correlation_R4", "hybrid correlation grouping"),
        _masklet_row(_assignment_row(assignment_rows, "D2_adjacent_matching_R6"), "D2_adjacent_matching_R6", "adjacent-only matching"),
        _masklet_row(_assignment_row(assignment_rows, "D3_hybrid_unknown_R6"), "D3_hybrid_unknown_R6", "hybrid unknown assignment; best prior real row"),
        _masklet_row(_assignment_row(assignment_rows, "D5_mask_only_control"), "D5_mask_only_control", "mask-only control"),
        _masklet_row(_assignment_row(assignment_rows, "D6_no_temporal_control"), "D6_no_temporal_control", "no-temporal control"),
    ]

    best_source = max(
        [row for row in source_rows if row.get("source_available")],
        key=lambda row: float(row.get("masklet_purity_proxy") or -1.0),
    )
    best_masklet = max(
        [row for row in masklet_rows if row.get("source_available")],
        key=lambda row: float(row.get("masklet_purity") or -1.0),
    )
    dino_manifest = {
        "source": "C2_DINO_feature_quality_diagnostic",
        "source_available": bool(dino_quality.get("feature_csv_exists")),
        "uses_frozen_visual_backbone": bool(dino_quality.get("uses_frozen_visual_backbone")),
        "same_GT_region_pair_AUC": dino_quality.get("same_GT_region_pair_AUC"),
        "mixed_region_AUC": dino_quality.get("mixed_region_AUC"),
        "scene0081_AUC": dino_quality.get("scene0081_AUC"),
        "phaseC_pass": bool(dino_quality.get("phaseC_pass")),
        "not_run_reason": dino_quality.get("not_run_reason"),
    }
    gate = {
        "best_source_by_purity_proxy": best_source["source"],
        "best_source_purity_proxy": best_source.get("masklet_purity_proxy"),
        "best_source_coverage_at_010": best_source.get("GT_object_coverage@0.10"),
        "best_masklet_variant_by_purity": best_masklet["label"],
        "best_masklet_purity": best_masklet.get("masklet_purity"),
        "best_masklet_completeness": best_masklet.get("masklet_completeness"),
        "best_masklet_scene0081_ARI": best_masklet.get("scene0081_masklet_ARI"),
        "dino_phaseC_pass": dino_manifest["phaseC_pass"],
    }
    gate["phaseC_gate_pass"] = bool(
        any(bool(row.get("phaseC_source_gate_pass")) for row in source_rows)
        or any(bool(row.get("phaseC_masklet_gate_pass")) for row in masklet_rows)
    )
    gate["object_birth_primitive_blocker"] = not gate["phaseC_gate_pass"]
    manifest = {
        "phase": "v39_phaseC_masklet_primitive",
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "uses_rgbd_for_prediction": False,
        "uses_pose_for_prediction": False,
        "uses_scannet_mesh_for_prediction": False,
        "uses_eval_sim3_for_prediction": False,
        "uses_d4rt_self_sim3": True,
        "uses_frozen_visual_backbone": bool(dino_quality.get("uses_frozen_visual_backbone")),
        "visual_backbone_name": "DINO diagnostic import" if dino_quality.get("uses_frozen_visual_backbone") else None,
        "mask_source": "v36_region_sources_imported_for_v39_audit",
        "object_birth_source": "2d_masklet_primitive_diagnostic_import",
        "d4rt_role": "support_attachment_or_association_diagnostic_only",
        "geometry_field": "none_for_prediction",
        "coordinate_frame": "2d_frame_region_ids",
        "alignment_source": "v36_v37_audit_artifacts",
    }
    summary = {
        **manifest,
        "source_root": str(source_root),
        "source_rows": source_rows,
        "masklet_rows": masklet_rows,
        "masklet_graph_rows_raw": masklet_graph_rows,
        "dino_diagnostic": dino_manifest,
        "v36_summary_best_real_assignment": v36_summary.get("best_real_assignment"),
        "phaseC_gate": gate,
        "notes": [
            "This is a v39 Phase C import/audit of existing masklet-first artifacts, not a new success claim.",
            "C1/C4/C5 represent plan-directed stronger split/repair attempts already present in the v36/v37 lineage.",
            "C2 and C3 are recorded as not-run when no full split/integration artifact is available.",
        ],
    }
    _write_json(output_root / "masklet_primitive_summary.json", summary)
    _write_json(output_root / "masklet_primitive_manifest.json", manifest)
    _write_csv(output_root / "masklet_primitive_source_matrix.csv", source_rows)
    _write_csv(output_root / "masklet_primitive_assignment_matrix.csv", masklet_rows)
    md = [
        "# Stream4D v39 Phase C Masklet Primitive",
        "",
        "| source | purity/proxy | completeness/coverage | scene0081 | pass | note |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in source_rows:
        md.append(
            "| {source} | {purity} | {coverage} | {scene} | {passed} | {note} |".format(
                source=row["source"],
                purity=row.get("masklet_purity_proxy"),
                coverage=row.get("GT_object_coverage@0.10"),
                scene=row.get("scene0081_masklet_purity_proxy"),
                passed=row.get("phaseC_source_gate_pass"),
                note=row.get("not_run_reason") or row.get("note"),
            )
        )
    md.extend(
        [
            "",
            "## Masklet Assignment Rows",
            "",
            "| variant | purity | completeness | ARI | scene0081 ARI | pass |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in masklet_rows:
        md.append(
            "| {label} | {purity} | {comp} | {ari} | {s81} | {passed} |".format(
                label=row["label"],
                purity=row.get("masklet_purity"),
                comp=row.get("masklet_completeness"),
                ari=row.get("ARI"),
                s81=row.get("scene0081_masklet_ARI"),
                passed=row.get("phaseC_masklet_gate_pass"),
            )
        )
    md.extend(
        [
            "",
            "## Gate",
            "",
            f"`phaseC_gate_pass={gate['phaseC_gate_pass']}`",
            "",
            f"`object_birth_primitive_blocker={gate['object_birth_primitive_blocker']}`",
        ]
    )
    (output_root / "masklet_primitive_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v39 Phase C masklet primitive import/audit.")
    parser.add_argument("--stream3d-root", default=".")
    parser.add_argument("--source-root", default="outputs/audit/v36_fallback_v35_generated_inputs")
    parser.add_argument("--output-root", default="outputs/audit/v39_masklet_primitive")
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
