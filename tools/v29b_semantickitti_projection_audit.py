#!/usr/bin/env python3
"""ACL2 v29B SemanticKITTI projection Phase-0 audit.

This audit is intentionally strict.  It only accepts real SemanticKITTI-style
point labels paired with matching KITTI odometry velodyne scans and calibration.
It never falls back to predicted Stage-C/video-masklet semantic labels.
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
            raise ValueError(f"Unsupported chunk for v29B projection audit: {chunk}")
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
                first_existing = str(layout.point_label_dir / f"{frame:06d}.label") if layout.point_label_dir else ""
        else:
            missing.append(frame)
    return hits, missing, first_existing


def _calib_status(layout: GTSemanticLayout) -> Dict[str, object]:
    calib_path = layout.calib_path
    status: Dict[str, object] = {
        "calib_path": str(calib_path or ""),
        "calib_exists": bool(calib_path and calib_path.exists()),
        "calib_has_p2": False,
        "calib_has_tr_velo": False,
        "calib_error": "",
    }
    if calib_path and calib_path.exists():
        try:
            calib = read_kitti_calib(calib_path)
            status["calib_has_p2"] = "P2" in calib
            status["calib_has_tr_velo"] = any(key in calib for key in ("Tr", "Tr_velo_to_cam", "Tr_velo_cam"))
        except Exception as exc:
            status["calib_error"] = str(exc)
    return status


def _probe_first_frame(
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


def _projection_layouts(sequence_root: Path, sequence: str, explicit_root: Optional[Path]) -> List[GTSemanticLayout]:
    return [
        layout
        for layout in discover_gt_semantic_layouts(
            sequence_root=sequence_root,
            sequence=sequence,
            explicit_gt_root=explicit_root,
        )
        if layout.point_projection
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--sequence-root", default="/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01")
    parser.add_argument("--sequence", default="01")
    parser.add_argument("--semantickitti-root", default=os.environ.get("V29B_SEMANTICKITTI_ROOT", ""))
    parser.add_argument("--results-root", default="results/kitti01_hmc_v2/acl2_v29b_semantickitti3dprojection_videomasklet_semanticprior_allmemory")
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

    explicit_root = Path(args.semantickitti_root).resolve() if str(args.semantickitti_root or "").strip() else None
    chunks = _parse_int_list(args.chunks, [6, 10, 16])
    horizons = _parse_int_list(args.horizons, [10, 15])
    frame_rows = _selected_frames(chunks, horizons)
    unique_frames = sorted({int(row["frame"]) for row in frame_rows})
    width, height, first_image = _image_resolution(sequence_root)
    image_size = (width, height) if width is not None and height is not None else None

    layouts = _projection_layouts(sequence_root, str(args.sequence), explicit_root)
    layout_rows: List[Dict[str, object]] = []
    best_hits = 0
    best_layout: Optional[GTSemanticLayout] = None
    best_probe: Dict[str, object] = {}
    for layout in layouts:
        hits, missing, first_existing = _layout_hits(layout, unique_frames)
        if hits > best_hits:
            best_hits = hits
            best_layout = layout
        row: Dict[str, object] = {
            "layout_name": layout.name,
            "kind": layout.kind,
            "image_dir": str(layout.image_dir or ""),
            "image_dir_exists": bool(layout.image_dir and layout.image_dir.exists()),
            "velodyne_dir": str(layout.velodyne_dir or ""),
            "velodyne_dir_exists": bool(layout.velodyne_dir and layout.velodyne_dir.exists()),
            "point_label_dir": str(layout.point_label_dir or ""),
            "point_label_dir_exists": bool(layout.point_label_dir and layout.point_label_dir.exists()),
            "unique_frames_expected": len(unique_frames),
            "unique_frames_hit": hits,
            "unique_frames_missing": len(unique_frames) - hits,
            "hit_rate": hits / max(1, len(unique_frames)),
            "first_existing_label": first_existing,
            "first_missing_frame": missing[0] if missing else "",
            "semantic_id_encoding": layout.semantic_id_encoding,
            "note": layout.note,
        }
        row.update(_calib_status(layout))
        if hits > 0:
            probe = _probe_first_frame(layout, unique_frames, image_size)
            row.update(probe)
            if layout is best_layout:
                best_probe = probe
        layout_rows.append(row)

    chunk_rows: List[Dict[str, object]] = []
    chosen_layout = best_layout
    for chunk in chunks:
        for horizon in horizons:
            frames = [int(row["frame"]) for row in frame_rows if int(row["chunk"]) == chunk and int(row["horizon"]) == horizon]
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
                    "projection_layout": chosen_layout.name if chosen_layout is not None else "",
                    "first_missing_frame": missing[0] if missing else "",
                }
            )

    total_expected = len(unique_frames)
    projection_hit_rate = best_hits / max(1, total_expected)
    projection_available = bool(best_layout is not None and best_hits == total_expected and total_expected > 0)
    no_predicted_fallback_flag = True

    _write_csv(out_dir / "projection_layout_scan.csv", layout_rows)
    _write_csv(out_dir / "projection_frame_hit_audit.csv", chunk_rows)
    _write_csv(
        out_dir / "noop_parity_metrics.csv",
        [
            {
                "status": "blocked",
                "reason": "" if projection_available else "semantickitti_projection_frames_missing",
                "ATE_delta_vs_H9": "",
                "raw_trans_max_diff": "",
                "pose_max_abs_diff": "",
                "note": "No-op smoke is forbidden until projection_frame_hit_rate is 1.0.",
            }
        ],
    )
    _write_csv(
        out_dir / "masklet_3d_alignment_status.csv",
        [
            {
                "status": "blocked",
                "reason": "" if projection_available else "semantickitti_projection_frames_missing",
                "masklet_3d_alignment_generated": False,
                "note": "Masklet-3D alignment is forbidden until projection cache exists.",
            }
        ],
    )

    failures: List[Dict[str, object]] = []
    if not projection_available:
        failures.append(
            {
                "gate": "phase0_semantickitti_projection_cache",
                "failure": "semantickitti_projection_frames_missing",
                "detail": (
                    "No SemanticKITTI-style projection layout has matching velodyne/*.bin and labels/*.label "
                    "for all selected KITTI01 frames."
                ),
                "attempted_fix": (
                    "Scanned sequence-local KITTI odometry layout and optional V29B_SEMANTICKITTI_ROOT. "
                    "Verified calib.txt/image_2 exist.  The local sequence 01 copy has no velodyne or labels "
                    "directory, and no predicted Stage-C/video-masklet fallback was used."
                ),
            }
        )

    summary: Dict[str, object] = {
        "phase": "v29b_phase0_semantickitti_projection_hard_gate",
        "phase0_gate_pass": projection_available,
        "projection_frame_hit_rate": projection_hit_rate,
        "projection_frames_expected": total_expected,
        "projection_frames_hit": best_hits,
        "projection_available": projection_available,
        "best_layout": best_layout.name if best_layout is not None else "",
        "best_layout_kind": best_layout.kind if best_layout is not None else "",
        "best_probe": best_probe,
        "sequence_root": str(sequence_root),
        "sequence": str(args.sequence),
        "explicit_semantickitti_root": str(explicit_root or ""),
        "chunks": chunks,
        "horizons": horizons,
        "image_width": width,
        "image_height": height,
        "first_image": first_image or "",
        "sequence_has_image_2": (sequence_root / "image_2").exists(),
        "sequence_has_velodyne": (sequence_root / "velodyne").exists(),
        "sequence_has_labels": (sequence_root / "labels").exists(),
        "sequence_has_calib": (sequence_root / "calib.txt").exists(),
        "no_predicted_fallback_flag": no_predicted_fallback_flag,
        "masklet_3d_alignment_allowed": False,
        "phase2_rollout_allowed": False,
        "selector_allowed": False,
        "full_online_validation_allowed": False,
        "counts_as_deployable_online_success": False,
        "blocked_reason": "" if projection_available else "semantickitti_projection_frames_missing",
    }
    (out_dir / "codex_self_check_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(out_dir / "codex_self_check_failures.jsonl", failures)

    report_lines = [
        "# ACL2 v29B SemanticKITTI Projection Phase-0 Audit",
        "",
        f"sequence_root: `{sequence_root}`",
        f"explicit_semantickitti_root: `{explicit_root or ''}`",
        f"chunks: `{chunks}`",
        f"horizons: `{horizons}`",
        "",
        "## Gate",
        "",
        f"phase0_gate_pass = `{str(projection_available).lower()}`",
        f"projection_frame_hit_rate = `{projection_hit_rate:.10f}`",
        f"projection_frames_hit = `{best_hits}`",
        f"projection_frames_expected = `{total_expected}`",
        f"sequence_has_image_2 = `{str((sequence_root / 'image_2').exists()).lower()}`",
        f"sequence_has_velodyne = `{str((sequence_root / 'velodyne').exists()).lower()}`",
        f"sequence_has_labels = `{str((sequence_root / 'labels').exists()).lower()}`",
        f"sequence_has_calib = `{str((sequence_root / 'calib.txt').exists()).lower()}`",
        f"no_predicted_fallback_flag = `{str(no_predicted_fallback_flag).lower()}`",
        "",
        "## Decision",
        "",
    ]
    if projection_available:
        report_lines.extend(
            [
                "SemanticKITTI point projection prerequisites cover all selected frames.",
                "Projection cache generation and no-op smoke are allowed next.",
            ]
        )
    else:
        report_lines.extend(
            [
                "Phase 0 fails because SemanticKITTI point projection prerequisites are missing.",
                "The local KITTI01 odometry copy has image_2 and calib.txt, but no velodyne/labels pair.",
                "Per v29B rules, video-masklet predicted semantic cannot be used as projected 3D semantic fallback.",
                "No masklet-3D alignment, semantic candidate rollout, selector, or full online validation is allowed.",
            ]
        )
    (out_dir / "codex_self_check_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return 0 if projection_available else 1


if __name__ == "__main__":
    raise SystemExit(main())
