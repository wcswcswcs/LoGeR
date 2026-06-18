"""Aggregate per-scene v27 pair-attribution outputs.

This helper exists because the full probe5 pair-attribution command can be
long enough to be killed by the execution environment before final file writes.
It keeps the experiment reproducible by allowing scene-level runs to be merged
after each scene has already landed its own artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from tools.run_v27_pair_attribution import _dominant_false_merge_source, _json_safe, _summarize_rows, _write_csv


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _coerce_row(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    bool_keys = {
        "same_chunk",
        "same_submap",
        "gt_labeled_pair",
        "same_gt",
        "different_gt",
        "guard_pass",
        "predicted_merge",
        "is_diagnostic_only",
    }
    int_keys = {
        "tube_i",
        "tube_j",
        "chunk_i",
        "chunk_j",
        "submap_i",
        "submap_j",
        "common_frame_count",
        "gt_i",
        "gt_j",
        "same_mask_count",
        "mask_cooccurrence_count",
        "boundary_safe_count",
        "boundary_cross_count",
        "visible_outside_count",
        "same_frame_cannot_link_count",
    }
    float_keys = {
        "distance",
        "distance_normalized",
        "merge_score",
        "cut_score",
        "appearance_similarity",
        "motion_consistency",
        "alignment_uncertainty",
    }
    for key, value in row.items():
        if value == "":
            out[key] = None
        elif key in bool_keys:
            out[key] = value.lower() == "true"
        elif key in int_keys:
            out[key] = int(float(value))
        elif key in float_keys:
            out[key] = float(value)
        else:
            out[key] = value
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate v27 per-scene pair-attribution diagnostics.")
    parser.add_argument("--scene-root", required=True, help="Directory containing one subdirectory per scene output.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--label", default="v27_measurements_by_scene_probe5")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scene_root = Path(args.scene_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    pair_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for scene_dir in sorted(p for p in scene_root.iterdir() if p.is_dir()):
        pair_files = sorted(scene_dir.glob("*_pair_rows.csv"))
        scene_files = sorted(scene_dir.glob("*_scene_rows.csv"))
        manifest_files = sorted(scene_dir.glob("*_manifest.json"))
        if not pair_files or not scene_files:
            continue
        for row in _read_csv(pair_files[0]):
            pair_rows.append(_coerce_row(row))
        scene_rows.extend(_read_csv(scene_files[0]))
        if manifest_files:
            manifests.append(json.loads(manifest_files[0].read_text(encoding="utf-8")))

    summary_rows = _summarize_rows(pair_rows)
    dominant = _dominant_false_merge_source(summary_rows)
    phase_c_boundary_negative_complete = all(
        bool(m.get("phase_c_boundary_negative_fields_complete", False)) for m in manifests
    ) and bool(manifests)
    phase_c_appearance_motion_complete = all(
        bool(m.get("phase_c_appearance_motion_fields_complete", False)) for m in manifests
    ) and bool(manifests)
    missing: list[str] = []
    for manifest in manifests:
        for field in manifest.get("missing_phase_c_fields", []):
            if field not in missing:
                missing.append(str(field))
    manifest = {
        "label": str(args.label),
        "source_scene_root": str(scene_root),
        "scene_output_count": len(manifests),
        "pair_row_count": len(pair_rows),
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "uses_d4rt_self_sim3": True,
        "alignment_source": "d4rt_self_sim3_or_same_chunk_identity",
        "coordinate_frame": "d4rt_canonical_for_cross_chunk_guarded_metric_pairs",
        "dominant_false_merge_source": dominant,
        "phase_b_complete": True,
        "phase_c_boundary_negative_fields_complete": phase_c_boundary_negative_complete,
        "phase_c_appearance_motion_fields_complete": phase_c_appearance_motion_complete,
        "phase_c_measurement_fields_complete": bool(
            phase_c_boundary_negative_complete and phase_c_appearance_motion_complete
        ),
        "missing_phase_c_fields": missing,
    }

    _write_csv(output_root / f"{args.label}_pair_rows.csv", pair_rows)
    _write_csv(output_root / f"{args.label}_category_summary.csv", summary_rows)
    _write_csv(output_root / f"{args.label}_scene_rows.csv", scene_rows)
    (output_root / f"{args.label}_category_summary.json").write_text(
        json.dumps(_json_safe(summary_rows), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / f"{args.label}_scene_rows.json").write_text(
        json.dumps(_json_safe(scene_rows), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / f"{args.label}_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe({"manifest": manifest, "summary_rows": [r for r in summary_rows if r["scene"] == "ALL"]}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
