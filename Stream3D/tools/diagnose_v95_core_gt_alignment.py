#!/usr/bin/env python3
"""Diagnostic-only v95 object-core alignment against ScanNet GT instances."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_v95_phase3_object_query as phase3  # noqa: E402
from tools import build_v95_phase4_core_conditioned_expansion as base  # noqa: E402
from tools import run_v65_scene_multiview_ap as v65eval  # noqa: E402


PHASE_ID = "v95_core_gt_alignment_diagnostic"
RUN_ID = "v95_core_gt_alignment_diagnostic"
OUT = ROOT / "outputs/audit/v95_phase5_core_gt_alignment_diagnostic"
PHASE1 = ROOT / "outputs/audit/v95_phase1_physical_source_registry"
PHASE2 = ROOT / "outputs/audit/v95_phase2_object_core_discovery_repair1"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(base._jsonable({key: row.get(key, "") for key in fields}))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base._jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mean(values: list[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(clean)) if clean else 0.0


def _source_meta() -> dict[tuple[str, str, int, int], dict[str, str]]:
    out: dict[tuple[str, str, int, int], dict[str, str]] = {}
    with (PHASE1 / "source_container_rows.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if base._bool(row.get("uses_gt_for_prediction")) or base._bool(row.get("uses_future")):
                continue
            out[(row["scene_id"], row["window_id"], base._int(row["frame_id"]), base._int(row["source_mask_id"]))] = dict(row)
    return out


def _dominant_gt(mask: np.ndarray, gt: np.ndarray) -> tuple[int, int, float, float]:
    pixels = int(np.count_nonzero(mask))
    if pixels <= 0:
        return 0, 0, 0.0, 0.0
    vals, counts = np.unique(gt[mask], return_counts=True)
    order = np.argsort(counts)[::-1]
    dominant_id = int(vals[order[0]]) if order.size else 0
    dominant_count = int(counts[order[0]]) if order.size else 0
    foreground = int(np.sum(counts[vals > 0])) if vals.size else 0
    return dominant_id, dominant_count, dominant_count / pixels, foreground / pixels


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = base._resolve(args.output_root)
    if out.exists() and args.clean:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    core_root = base._resolve(args.core_root)
    selected_by_source, _selected_regions, core_meta = phase3._load_selected_cores(core_root, int(args.max_sources))
    source_meta = _source_meta()
    nodes_by_source = phase3._load_region_nodes(PHASE1 / "region_node_rows.csv", set(selected_by_source))
    label_cache: dict[tuple[str, int], np.ndarray] = {}
    gt_cache: dict[tuple[str, int], np.ndarray] = {}
    core_rows: list[dict[str, Any]] = []
    object_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"dominant_gt_ids": [], "fg": [], "purity": [], "source_count": 0})
    processed_sources = 0
    for key, cores in selected_by_source.items():
        scene, window, frame_id, mask_id = key
        meta = source_meta.get(key)
        nodes = nodes_by_source.get(key, {})
        if not meta or not nodes:
            continue
        frame_key = (scene, int(frame_id))
        if frame_key not in label_cache:
            label_cache[frame_key] = base._read_label(base._resolve(meta.get("mask_path", "")))
        source_mask = label_cache[frame_key] == int(mask_id)
        if not np.any(source_mask):
            continue
        if frame_key not in gt_cache:
            gt_cache[frame_key] = v65eval._load_gt_2d(scene, int(frame_id), source_mask.shape)
        gt = gt_cache[frame_key]
        processed_sources += 1
        for core in cores:
            query_core = {int(idx) for idx in core.get("query_core_region_indices", [])}
            confirmed_core = {int(idx) for idx in core.get("selected_core_region_indices", [])}
            mask = base._node_mask(nodes, query_core or confirmed_core, source_mask)
            dom_id, dom_count, dom_frac, fg_frac = _dominant_gt(mask, gt)
            pixels = int(np.count_nonzero(mask))
            object_id = str(core["object_id"])
            row = {
                "schema_version": "stream4d_v95_core_gt_alignment_row_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "scene_id": scene,
                "window_id": window,
                "frame_id": int(frame_id),
                "source_mask_id": int(mask_id),
                "object_id": object_id,
                "core_variant_id": core.get("variant_id", ""),
                "query_core_region_count": len(query_core),
                "confirmed_core_region_count": len(confirmed_core),
                "core_pixel_count_bbox_approx": pixels,
                "dominant_gt_id": dom_id,
                "dominant_gt_pixel_count": dom_count,
                "dominant_gt_fraction": dom_frac,
                "gt_foreground_fraction": fg_frac,
                "gt_background_majority": fg_frac < 0.5,
                "diagnostic_only_uses_gt": True,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
            core_rows.append(row)
            stats = object_stats[object_id]
            stats["dominant_gt_ids"].append(dom_id)
            stats["fg"].append(fg_frac)
            stats["purity"].append(dom_frac)
            stats["source_count"] += 1
        if int(args.progress_every_sources) > 0 and processed_sources % int(args.progress_every_sources) == 0:
            print(
                json.dumps(
                    {
                        "phase": PHASE_ID,
                        "processed_sources": processed_sources,
                        "core_rows": len(core_rows),
                        "elapsed_sec": time.time() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    object_rows: list[dict[str, Any]] = []
    consistency_values: list[float] = []
    multi_gt_count = 0
    for object_id, stats in object_stats.items():
        positive_ids = [int(value) for value in stats["dominant_gt_ids"] if int(value) > 0]
        counts = Counter(positive_ids)
        mode_id, mode_count = counts.most_common(1)[0] if counts else (0, 0)
        source_count = int(stats["source_count"])
        consistency = mode_count / max(1, source_count)
        consistency_values.append(consistency)
        unique_positive = len(counts)
        multi_gt = unique_positive > 1
        multi_gt_count += int(multi_gt)
        object_rows.append(
            {
                "schema_version": "stream4d_v95_core_gt_object_alignment_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "object_id": object_id,
                "source_count": source_count,
                "positive_dominant_gt_unique_count": unique_positive,
                "dominant_gt_mode_id": mode_id,
                "dominant_gt_mode_count": mode_count,
                "dominant_gt_mode_consistency": consistency,
                "gt_foreground_fraction_mean": _mean(stats["fg"]),
                "dominant_gt_fraction_mean": _mean(stats["purity"]),
                "multi_gt_anchor": multi_gt,
                "diagnostic_only_uses_gt": True,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    fg_values = [base._num(row["gt_foreground_fraction"]) for row in core_rows]
    purity_values = [base._num(row["dominant_gt_fraction"]) for row in core_rows]
    summary = {
        "schema": "stream4d_v95_core_gt_alignment_diagnostic_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": base._created_at(),
        "decision": "DIAGNOSTIC_ONLY_V95_CORE_GT_ALIGNMENT_READY",
        "core_root": base._rel(core_root),
        "selected_confirmed_core_count": core_meta.get("selected_confirmed_core_count"),
        "processed_sources": processed_sources,
        "core_gt_rows": len(core_rows),
        "object_gt_alignment_rows": len(object_rows),
        "core_gt_foreground_fraction_mean": _mean(fg_values),
        "core_dominant_gt_fraction_mean": _mean(purity_values),
        "core_background_majority_rate": _mean([1.0 if row["gt_background_majority"] else 0.0 for row in core_rows]),
        "object_dominant_gt_mode_consistency_mean": _mean(consistency_values),
        "object_multi_gt_anchor_rate": multi_gt_count / max(1, len(object_rows)),
        "diagnostic_only_uses_gt": True,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "notes": "Uses the same bbox-limited region materialization as Phase4; GT is diagnostic-only and never feeds prediction.",
        "duration_sec": time.time() - started,
        "row_counts": {
            "core_gt_rows": len(core_rows),
            "object_gt_alignment_rows": len(object_rows),
        },
    }
    _write_csv(out / "core_gt_rows.csv", core_rows)
    _write_csv(out / "object_gt_alignment_rows.csv", object_rows)
    _write_json(out / "summary.json", summary)
    _write_json(
        out / "SHA256SUMS.json",
        {
            base._rel(out / "core_gt_rows.csv"): base._sha256(out / "core_gt_rows.csv"),
            base._rel(out / "object_gt_alignment_rows.csv"): base._sha256(out / "object_gt_alignment_rows.csv"),
            base._rel(out / "summary.json"): base._sha256(out / "summary.json"),
        },
    )
    print(json.dumps(base._jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-root", default=str(PHASE2))
    parser.add_argument("--output-root", default=str(OUT))
    parser.add_argument("--max-sources", type=int, default=0)
    parser.add_argument("--progress-every-sources", type=int, default=512)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
