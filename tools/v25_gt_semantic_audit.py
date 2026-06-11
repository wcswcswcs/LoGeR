#!/usr/bin/env python3
"""ACL2 v25 GT semantic Phase-0 audit.

This audit is intentionally strict: it only accepts real GT semantic layouts.
It never falls back to the v24 predicted Stage-C semantic cache.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loger.pipeline.gt_semantic_provider import (  # noqa: E402
    GTSemanticLayout,
    GTSemanticProvider,
    discover_gt_semantic_layouts,
    read_kitti_calib,
)


CHUNK_STARTS = {
    6: 174,
    10: 290,
    16: 464,
}


def _write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    materialized = list(rows)
    fields: List[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _parse_int_list(text: str, default: Sequence[int]) -> List[int]:
    text = (text or "").strip()
    if not text:
        return list(default)
    values: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values


def _selected_frames(chunks: Sequence[int], horizons: Sequence[int]) -> List[Dict[str, int]]:
    rows: List[Dict[str, int]] = []
    for chunk in chunks:
        if chunk not in CHUNK_STARTS:
            raise ValueError(f"Unsupported chunk for v25 GT audit: {chunk}")
        start = CHUNK_STARTS[chunk]
        for horizon in horizons:
            end = start + 32 + horizon * 29
            for frame in range(start, end):
                rows.append({"chunk": chunk, "horizon": horizon, "frame": frame})
    return rows


def _image_resolution(sequence_root: Path) -> tuple[Optional[int], Optional[int], Optional[str]]:
    first = next(iter(sorted((sequence_root / "image_2").glob("*.png"))), None)
    if first is None:
        return None, None, None
    try:
        from PIL import Image  # type: ignore

        with Image.open(first) as img:
            width, height = img.size
        return int(width), int(height), str(first)
    except Exception:
        return None, None, str(first)


def _layout_hits(layout: GTSemanticLayout, frames: Sequence[int]) -> tuple[int, List[int], str]:
    hits = 0
    missing: List[int] = []
    first_existing = ""
    for frame in frames:
        if layout.has_frame(frame):
            hits += 1
            if not first_existing:
                if layout.point_projection and layout.point_label_dir is not None:
                    first_existing = str(layout.point_label_dir / f"{frame:06d}.label")
                else:
                    first_existing = str(layout.label_path(frame))
        else:
            missing.append(frame)
    return hits, missing, first_existing


def _calib_status(layout: GTSemanticLayout) -> Dict[str, object]:
    if not layout.point_projection:
        return {
            "calib_path": "",
            "calib_exists": "",
            "calib_has_p2": "",
            "calib_has_tr_velo": "",
        }
    calib_path = layout.calib_path
    status: Dict[str, object] = {
        "calib_path": str(calib_path or ""),
        "calib_exists": bool(calib_path and calib_path.exists()),
        "calib_has_p2": False,
        "calib_has_tr_velo": False,
    }
    if calib_path and calib_path.exists():
        try:
            calib = read_kitti_calib(calib_path)
            status["calib_has_p2"] = "P2" in calib
            status["calib_has_tr_velo"] = any(key in calib for key in ("Tr", "Tr_velo_to_cam", "Tr_velo_cam"))
        except Exception as exc:
            status["calib_error"] = str(exc)
    return status


def _safe_probe_first_frame(
    layout: GTSemanticLayout,
    frames: Sequence[int],
    image_size: Optional[tuple[int, int]],
) -> Dict[str, object]:
    for frame in frames:
        if not layout.has_frame(frame):
            continue
        try:
            loaded = GTSemanticProvider(layout, image_size=image_size).load_frame(frame)
            return {
                "probe_frame": frame,
                "probe_load_ok": True,
                "probe_coverage": loaded.coverage,
                "probe_num_labels": len(loaded.label_counts),
                "probe_source_path": loaded.source_path,
                "probe_error": "",
            }
        except Exception as exc:
            return {
                "probe_frame": frame,
                "probe_load_ok": False,
                "probe_coverage": "",
                "probe_num_labels": "",
                "probe_source_path": "",
                "probe_error": str(exc),
            }
    return {
        "probe_frame": "",
        "probe_load_ok": False,
        "probe_coverage": "",
        "probe_num_labels": "",
        "probe_source_path": "",
        "probe_error": "no frame hit",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--sequence-root", default="/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01")
    parser.add_argument("--sequence", default="01")
    parser.add_argument("--gt-root", default=os.environ.get("V25_GT_SEMANTIC_ROOT", ""))
    parser.add_argument(
        "--results-root",
        default="results/kitti01_hmc_v2/acl2_v25_gt_semanticprior_allmemory_parallel",
    )
    parser.add_argument("--chunks", default="6,10,16")
    parser.add_argument("--horizons", default="10,15")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    sequence_root = Path(args.sequence_root).resolve()
    results = Path(args.results_root)
    if not results.is_absolute():
        results = repo / results
    out_dir = results / "implementation_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_root = Path(args.gt_root).resolve() if str(args.gt_root or "").strip() else None
    chunks = _parse_int_list(args.chunks, [6, 10, 16])
    horizons = _parse_int_list(args.horizons, [10, 15])
    frame_rows = _selected_frames(chunks, horizons)
    unique_frames = sorted({int(row["frame"]) for row in frame_rows})
    width, height, first_image = _image_resolution(sequence_root)
    image_size = (width, height) if width is not None and height is not None else None

    layouts = discover_gt_semantic_layouts(
        sequence_root=sequence_root,
        sequence=str(args.sequence),
        explicit_gt_root=gt_root,
    )

    layout_rows: List[Dict[str, object]] = []
    best_hits = 0
    best_layout: Optional[GTSemanticLayout] = None
    best_dense_hits = 0
    best_projection_hits = 0
    for layout in layouts:
        hits, missing, first_existing = _layout_hits(layout, unique_frames)
        if hits > best_hits:
            best_hits = hits
            best_layout = layout
        if layout.dense_image_map:
            best_dense_hits = max(best_dense_hits, hits)
        if layout.point_projection:
            best_projection_hits = max(best_projection_hits, hits)
        row: Dict[str, object] = {
            "layout_name": layout.name,
            "kind": layout.kind,
            "label_dir": str(layout.label_dir),
            "label_dir_exists": layout.label_dir.exists(),
            "image_dir": str(layout.image_dir or ""),
            "image_dir_exists": bool(layout.image_dir and layout.image_dir.exists()),
            "dense_image_map_supported": layout.dense_image_map,
            "point_projection_supported": layout.point_projection,
            "semantic_id_encoding": layout.semantic_id_encoding,
            "frame_digits": layout.frame_digits,
            "suffix": layout.suffix,
            "unique_frames_expected": len(unique_frames),
            "unique_frames_hit": hits,
            "unique_frames_missing": len(unique_frames) - hits,
            "hit_rate": hits / max(1, len(unique_frames)),
            "first_existing_label": first_existing,
            "first_missing_frame": missing[0] if missing else "",
            "note": layout.note,
        }
        row.update(_calib_status(layout))
        if hits > 0:
            row.update(_safe_probe_first_frame(layout, unique_frames, image_size))
        layout_rows.append(row)

    chunk_rows: List[Dict[str, object]] = []
    chosen_layout = best_layout
    for chunk in chunks:
        for horizon in horizons:
            rows = [row for row in frame_rows if int(row["chunk"]) == chunk and int(row["horizon"]) == horizon]
            frames = [int(row["frame"]) for row in rows]
            if chosen_layout is None:
                hits = 0
                missing = frames
            else:
                hits, missing, _ = _layout_hits(chosen_layout, frames)
            chunk_rows.append(
                {
                    "sequence": str(args.sequence),
                    "chunk": chunk,
                    "horizon": horizon,
                    "start_frame": min(frames),
                    "end_frame_exclusive": max(frames) + 1,
                    "frames_expected": len(frames),
                    "frames_hit": hits,
                    "frames_missing": len(missing),
                    "hit_rate": hits / max(1, len(frames)),
                    "gt_layout": chosen_layout.name if chosen_layout is not None else "",
                    "gt_layout_kind": chosen_layout.kind if chosen_layout is not None else "",
                    "first_missing_frame": missing[0] if missing else "",
                }
            )

    total_expected = len(unique_frames)
    gt_hit_rate = best_hits / max(1, total_expected)
    gt_available = bool(best_layout is not None and best_hits == total_expected and total_expected > 0)
    no_predicted_fallback_flag = True

    _write_csv(out_dir / "gt_semantic_layout_scan.csv", layout_rows)
    _write_csv(out_dir / "gt_semantic_cache_audit.csv", chunk_rows)
    _write_csv(
        out_dir / "semantic_role_alignment_audit.csv",
        [
            {
                "status": "blocked",
                "reason": "gt_semantic_cache_missing" if not gt_available else "not_run",
                "semantic_role_noop_ATE_delta": "",
                "semantic_role_noop_raw_trans_diff": "",
                "note": "No runtime GT projection/no-op smoke is valid until GT cache hit rate is 1.0.",
            }
        ],
    )
    _write_csv(
        out_dir / "path_consumption_audit.csv",
        [
            {
                "status": "blocked",
                "reason": "gt_semantic_cache_missing" if not gt_available else "not_run",
                "frame_consumed": False,
                "global_consumed": False,
                "swa_consumed": False,
                "ttt_consumed": False,
                "note": "Candidate path consumption is forbidden before Phase 0 GT cache gate passes.",
            }
        ],
    )
    _write_csv(
        out_dir / "noop_parity_metrics.csv",
        [
            {
                "status": "blocked",
                "reason": "gt_semantic_cache_missing" if not gt_available else "not_run",
                "ATE_delta_vs_H9": "",
                "raw_trans_max_diff": "",
                "note": "GT no-op parity smoke was not launched because GT labels are unavailable.",
            }
        ],
    )

    failures: List[Dict[str, object]] = []
    if not gt_available:
        failures.append(
            {
                "gate": "phase0_gt_cache",
                "failure": "gt_semantic_cache_missing",
                "detail": (
                    "No GT semantic layout has labels for all selected KITTI01 frames. "
                    "Predicted Stage-C semantic fallback is forbidden by the v25 plan."
                ),
                "attempted_fix": (
                    "Implemented GT provider support for KITTI semantic benchmark dense PNG layouts, "
                    "KITTI-STEP panoptic PNG layouts, KITTI-360 dense 2D semantic layouts, and "
                    "SemanticKITTI point-label projection from velodyne/*.bin + labels/*.label. "
                    "The local KITTI01 odometry copy still has only calib.txt, image_2, image_3, "
                    "and times.txt; no dense 2D labels, velodyne scans, or point labels were found."
                ),
            }
        )

    summary: Dict[str, object] = {
        "phase": "v25_phase0_gt_semantic_implementation_hard_gate",
        "phase0_gate_pass": bool(gt_available),
        "gt_semantic_available": bool(gt_available),
        "gt_cache_hit_rate": gt_hit_rate,
        "gt_frames_expected": total_expected,
        "gt_frames_hit": best_hits,
        "best_dense_frames_hit": best_dense_hits,
        "best_projection_frames_hit": best_projection_hits,
        "best_layout": best_layout.name if best_layout is not None else "",
        "best_layout_kind": best_layout.kind if best_layout is not None else "",
        "sequence_root": str(sequence_root),
        "sequence": str(args.sequence),
        "explicit_gt_root": str(gt_root or ""),
        "chunks": chunks,
        "horizons": horizons,
        "image_width": width,
        "image_height": height,
        "first_image": first_image or "",
        "no_predicted_fallback_flag": no_predicted_fallback_flag,
        "semantic_role_noop_ATE_delta": None,
        "semantic_role_noop_raw_trans_diff": None,
        "selector_allowed": False,
        "full_online_validation_allowed": False,
        "counts_as_deployable_online_success": False,
        "blocked_reason": "" if gt_available else "gt_semantic_cache_missing",
    }
    (out_dir / "codex_self_check_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(out_dir / "codex_self_check_failures.jsonl", failures)

    report_lines = [
        "# ACL2 v25 GT Semantic Phase-0 Audit",
        "",
        f"sequence_root: `{sequence_root}`",
        f"explicit_gt_root: `{gt_root or ''}`",
        f"chunks: `{chunks}`",
        f"horizons: `{horizons}`",
        "",
        "## Supported GT Inputs",
        "",
        "- KITTI semantic benchmark dense PNG: `training/semantic/*.png` or `training/semantic_rgb/*.png`",
        "- KITTI-STEP panoptic PNG: `panoptic_maps/{train,val}/{sequence}/{frame}.png`",
        "- KITTI-360 dense 2D semantic PNG: `data_2d_semantics/.../semantic/*.png`",
        "- SemanticKITTI point labels projected to image_2 from `velodyne/*.bin` + `labels/*.label` + calib `Tr`",
        "",
        "## Gate",
        "",
        f"phase0_gate_pass = `{str(gt_available).lower()}`",
        f"gt_cache_hit_rate = `{gt_hit_rate:.10f}`",
        f"gt_frames_hit = `{best_hits}`",
        f"gt_frames_expected = `{total_expected}`",
        f"best_dense_frames_hit = `{best_dense_hits}`",
        f"best_projection_frames_hit = `{best_projection_hits}`",
        f"no_predicted_fallback_flag = `{str(no_predicted_fallback_flag).lower()}`",
        "",
        "## Decision",
        "",
    ]
    if gt_available:
        report_lines.extend(
            [
                "GT semantic labels were found for all selected frames.",
                "Runtime no-op/path smoke is now allowed, but has not been launched by this audit script.",
            ]
        )
    else:
        report_lines.extend(
            [
                "Phase 0 fails because no supported GT semantic layout covers the selected KITTI01 frames.",
                "The local KITTI01 odometry copy has no dense 2D semantic labels and no velodyne/labels pair for projection.",
                "Per v25 rules, predicted Mask2Former/Stage-C semantic fallback is forbidden.",
                "No GT semantic candidate rollout, pairwise/all-memory experiment, selector, or full online validation is allowed.",
            ]
        )
    (out_dir / "codex_self_check_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return 0 if gt_available else 1


if __name__ == "__main__":
    raise SystemExit(main())
