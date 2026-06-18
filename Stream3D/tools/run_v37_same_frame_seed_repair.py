from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.run_v26_object_quality_diagnostics import _json_safe, _read_split
from tools.run_v36_external_downstream_assignment import _collect_observations, _load_masks, _load_tubes
from tools.run_v37_temporal_curriculum import (
    _aggregate_seed_rows,
    _region_diagnostics,
    _seed_metrics,
    _write_csv,
    _write_json,
)


def _support_concentration(counter: Any) -> float:
    total = sum(int(v) for v in counter.values())
    if total <= 0:
        return 0.0
    return float(max(int(v) for v in counter.values()) / total)


def run(args: argparse.Namespace) -> dict[str, Any]:
    setattr(_load_masks, "max_regions_per_scene", int(args.max_regions_per_scene))
    scenes = _read_split(Path(args.split))
    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_rows = []
    manifests = {}
    for scene in scenes:
        nodes, labels_by_frame, mask_manifest = _load_masks(Path(args.mask_root), scene, args.source, args.mode, int(args.min_region_area))
        tubes = _load_tubes(scene, args)
        support_by_region, _, _ = _collect_observations(nodes, labels_by_frame, tubes, args)
        diagnostics, gt_area = _region_diagnostics(scene, nodes, labels_by_frame, compute_rgb=False)
        support_sets = {idx: set(counter.keys()) for idx, counter in support_by_region.items()}
        concentrations = {idx: _support_concentration(counter) for idx, counter in support_by_region.items()}
        areas = np.asarray([int(node.area) for node in nodes], dtype=np.float64)
        p50 = float(np.quantile(areas, 0.50)) if areas.size else 0.0
        p75 = float(np.quantile(areas, 0.75)) if areas.size else 0.0
        variants = {
            "C5_D4RT_supported3_unknown": {
                idx for idx, tubeset in support_sets.items() if len(tubeset) >= 3
            },
            "C6_D4RT_support_concentration_060": {
                idx for idx, tubeset in support_sets.items() if len(tubeset) >= 2 and concentrations.get(idx, 0.0) >= 0.60
            },
            "C7_D4RT_supported3_concentration_060": {
                idx for idx, tubeset in support_sets.items() if len(tubeset) >= 3 and concentrations.get(idx, 0.0) >= 0.60
            },
            "C8_safe_fringe_supported1_area_le_p75": {
                idx for idx, tubeset in support_sets.items() if len(tubeset) >= 1 and int(nodes[idx].area) <= p75
            },
            "C9_tiny_safe_fringe_supported1_area_le_p50": {
                idx for idx, tubeset in support_sets.items() if len(tubeset) >= 1 and int(nodes[idx].area) <= p50
            },
        }
        for variant, active in variants.items():
            scene_rows.append(
                _seed_metrics(
                    scene,
                    nodes,
                    labels_by_frame,
                    diagnostics,
                    gt_area,
                    active,
                    variant=variant,
                    status="repair_method_filter_no_gt_prediction",
                )
            )
        manifests[scene] = {
            **mask_manifest,
            "area_p50": p50,
            "area_p75": p75,
            "region_with_support_count": int(len(support_sets)),
            "repair_filters_use_gt_for_prediction": False,
            "repair_filters": "D4RT support count, D4RT support concentration, region area percentile",
        }
    summary = _aggregate_seed_rows(scene_rows)
    _write_csv(out_dir / "same_frame_seed_repair_scene_rows.csv", scene_rows)
    _write_csv(out_dir / "same_frame_seed_repair_summary.csv", summary)
    _write_json(out_dir / "same_frame_seed_repair_summary.json", summary)
    payload = {
        "plan": "docs/stream4d_v37_temporal_curriculum_masklet_plan.md",
        "phase": "Phase C repair",
        "summary": summary,
        "manifests": manifests,
        "is_method_result": False,
        "is_diagnostic_only": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    _write_json(out_dir / "same_frame_seed_repair_manifest.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v37 Phase C same-frame seed repair filters.")
    parser.add_argument("--mask-root", default="outputs/audit/v36_external_mask_source_all")
    parser.add_argument("--source", default="watershed")
    parser.add_argument("--mode", default="all_masks")
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v21_3_occupancy_d5_warmstart64_probe5_r1")
    parser.add_argument("--output-root", default="outputs/audit/v37_same_frame_objectlets")
    parser.add_argument("--max-tubes-per-window", type=int, default=640)
    parser.add_argument("--image-width", type=int, default=1296)
    parser.add_argument("--image-height", type=int, default=968)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--min-region-area", type=int, default=64)
    parser.add_argument("--max-regions-per-scene", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(_json_safe(run(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

