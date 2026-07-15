#!/usr/bin/env python3
"""Merge visual review, watcher rows, and LingBot anchors for durable admission audit.

This is a shadow audit. It never mutates output masks or SAM2 memory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "Stream3D"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from Stream3D.stream4d_v108.lifecycle import DelayedAdmissionPolicy  # noqa: E402
from Stream3D.stream4d_v108.visual_review import (  # noqa: E402
    default_pending_review,
    load_visual_review_manifest,
)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def resolve_path(text: str) -> Path:
    path = Path(text)
    if path.is_absolute():
        return path
    return ROOT / path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: jsonable(row.get(key)) for key in fieldnames})


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_bool(text: Any) -> bool:
    return str(text).strip().lower() in {"1", "true", "yes", "y"}


def parse_float(text: Any, default: float = -1.0) -> float:
    try:
        return float(text)
    except Exception:
        return float(default)


def parse_int(text: Any, default: int = 0) -> int:
    try:
        return int(float(text))
    except Exception:
        return int(default)


def load_phase6_root(root: Path) -> dict[str, Any]:
    summary_path = root / "phase6_probation_watcher_shadow_summary.json"
    summary = read_json(summary_path)
    admission_rows = read_csv_rows(root / "admission_shadow_rows.csv")
    frame_rows = read_csv_rows(root / "watcher_frame_rows.csv")
    tracks = read_json(root / "track_summaries.json").get("tracks", [])
    visual_by_object = {int(row["object_id"]): row for row in tracks}
    frame_by_key = {
        (str(row["scene_id"]), int(row["object_id"]), int(row["frame_id"])): row
        for row in frame_rows
    }
    return {
        "root": root,
        "summary_path": summary_path,
        "summary": summary,
        "admission_rows": admission_rows,
        "frame_by_key": frame_by_key,
        "visual_by_object": visual_by_object,
    }


def load_anchor_roots(anchor_roots: list[Path]) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for root in anchor_roots:
        summary_path = root / "phase4_lingbot_anchor_visual_summary.json"
        case_path = root / "case_summaries.json"
        if not summary_path.exists() or not case_path.exists():
            continue
        summary = read_json(summary_path)
        cases = read_json(case_path).get("cases", [])
        for case in cases:
            scene_id = str(case.get("scene_id", summary.get("scene_id", "")))
            object_id = int(case["target_obj_id"])
            item = dict(case)
            item["anchor_root"] = rel(root)
            item["anchor_summary"] = rel(summary_path)
            item["anchor_summary_sha256"] = sha256_file(summary_path)
            item["geometry_available"] = bool(summary.get("geometry_available", True))
            item["uses_scannet_pose_or_depth_for_projection"] = bool(summary.get("uses_scannet_pose_or_depth_for_projection", True))
            out[(scene_id, object_id)] = item
    return out


def component_stats_from_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "area_px": parse_int(row.get("area_px")),
        "area_frac": parse_float(row.get("area_frac"), 0.0),
        "bbox_area_frac": parse_float(row.get("bbox_area_frac"), 0.0),
        "bbox_extent": parse_float(row.get("bbox_extent"), 0.0),
        "edge_touch_count": parse_int(row.get("edge_touch_count")),
    }


def watcher_stats_from_admission(row: dict[str, str]) -> dict[str, Any]:
    return {
        "visible_frame_count": parse_int(row.get("watcher_visible_frame_count")),
        "first_visible_frame_id": parse_int(row.get("watcher_first_visible_frame_id"), -1),
        "last_visible_frame_id": parse_int(row.get("watcher_last_visible_frame_id"), -1),
        "mean_iou_to_previous_visible": parse_float(row.get("watcher_mean_iou_to_previous_visible")),
        "min_iou_to_previous_visible": parse_float(row.get("watcher_min_iou_to_previous_visible")),
        "max_iou_to_previous_visible": parse_float(row.get("watcher_max_iou_to_previous_visible")),
    }


def physical_stats_from_anchor(anchor: dict[str, Any] | None) -> dict[str, Any] | None:
    if anchor is None:
        return None
    return {
        "geometry_available": bool(anchor.get("geometry_available", True)),
        "projected_positive_count": int(anchor.get("projected_positive_count", 0)),
        "projected_negative_count": int(anchor.get("projected_negative_count", 0)),
        "conflict_diagnostics": dict(anchor.get("conflict_diagnostics") or {}),
        "source_support": dict(anchor.get("source_support") or {}),
        "target_support": dict(anchor.get("target_support") or {}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--phase6-root", action="append", required=True)
    parser.add_argument("--anchor-root", action="append", default=[])
    parser.add_argument("--review-manifest", required=True)
    parser.add_argument("--min-visible-positive-anchors", type=int, default=1)
    parser.add_argument("--max-anchor-conflict-count", type=int, default=0)
    parser.add_argument("--max-positive-anchor-outlier-count", type=int, default=0)
    parser.add_argument("--min-anchor-depth-valid-fraction", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()
    output_root = resolve_path(str(args.output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "last_command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n",
        encoding="utf-8",
    )
    phase6_roots = [resolve_path(str(item)) for item in args.phase6_root]
    anchor_roots = [resolve_path(str(item)) for item in args.anchor_root]
    review_manifest_path = resolve_path(str(args.review_manifest))
    reviews = load_visual_review_manifest(review_manifest_path)
    anchor_by_key = load_anchor_roots(anchor_roots)
    policy = DelayedAdmissionPolicy(
        min_visible_positive_anchors=int(args.min_visible_positive_anchors),
        max_anchor_conflict_count=int(args.max_anchor_conflict_count),
        max_positive_anchor_outlier_count=int(args.max_positive_anchor_outlier_count),
        min_anchor_depth_valid_fraction=float(args.min_anchor_depth_valid_fraction),
    )

    rows: list[dict[str, Any]] = []
    user_review_rows: list[dict[str, Any]] = []
    for phase6 in [load_phase6_root(root) for root in phase6_roots]:
        for admission in phase6["admission_rows"]:
            scene_id = str(admission["scene_id"])
            object_id = int(admission["object_id"])
            frame_id = int(admission["frame_id"])
            frame_row = phase6["frame_by_key"].get((scene_id, object_id, frame_id), {})
            visual = phase6["visual_by_object"].get(object_id, {})
            evidence_paths = tuple([str(visual.get("visual_path"))] if visual.get("visual_path") else [])
            evidence_sha256 = tuple([str(visual.get("visual_sha256"))] if visual.get("visual_sha256") else [])
            review = reviews.get(
                (scene_id, object_id, frame_id),
                default_pending_review(
                    scene_id=scene_id,
                    object_id=object_id,
                    frame_id=frame_id,
                    evidence_paths=evidence_paths,
                    evidence_sha256=evidence_sha256,
                ),
            )
            anchor = anchor_by_key.get((scene_id, object_id))
            physical_stats = physical_stats_from_anchor(anchor)
            diagnostic = policy.evaluate(
                frame_id=frame_id,
                global_object_id=object_id,
                component_stats=component_stats_from_row(frame_row),
                watcher_stats=watcher_stats_from_admission(admission),
                physical_support_stats=physical_stats,
                visual_review_status=review.visual_review_status,
            )
            user_review_rows.append(review.as_dict())
            rows.append(
                {
                    "scene_id": scene_id,
                    "object_id": object_id,
                    "frame_id": frame_id,
                    "phase6_root": rel(phase6["root"]),
                    "phase6_summary": rel(phase6["summary_path"]),
                    "phase6_summary_sha256": sha256_file(phase6["summary_path"]),
                    "visual_review_status": review.visual_review_status,
                    "visual_note": review.visual_note,
                    "visual_evidence_paths": list(review.evidence_paths),
                    "visual_evidence_sha256": list(review.evidence_sha256),
                    "reviewer": review.reviewer,
                    "anchor_audit_available": anchor is not None,
                    "anchor_root": anchor.get("anchor_root") if anchor else "",
                    "anchor_visual_path": anchor.get("visual_path") if anchor else "",
                    "anchor_visual_sha256": anchor.get("visual_sha256") if anchor else "",
                    "geometry_available": physical_stats.get("geometry_available") if physical_stats else "",
                    "uses_scannet_pose_or_depth_for_projection": anchor.get("uses_scannet_pose_or_depth_for_projection") if anchor else "",
                    "projected_positive_count": physical_stats.get("projected_positive_count") if physical_stats else "",
                    "projected_negative_count": physical_stats.get("projected_negative_count") if physical_stats else "",
                    "positive_negative_conflict_count": (physical_stats.get("conflict_diagnostics") or {}).get("positive_negative_conflict_count") if physical_stats else "",
                    "positive_cluster_outlier_count": (physical_stats.get("conflict_diagnostics") or {}).get("positive_cluster_outlier_count") if physical_stats else "",
                    "output_state": str(diagnostic.output_state.value),
                    "output_allowed": bool(diagnostic.output_allowed),
                    "durable_memory_allowed": bool(diagnostic.durable_memory_allowed),
                    "durable_reject_reasons": list(diagnostic.reasons),
                    "diagnostic_only": bool(diagnostic.diagnostic_only),
                    "metrics_are_diagnostic_only": True,
                }
            )

    audit_csv = output_root / "durable_admission_audit_rows.csv"
    audit_jsonl = output_root / "durable_admission_audit_rows.jsonl"
    review_out = output_root / "user_review_manifest.json"
    write_csv(audit_csv, rows)
    audit_jsonl.write_text(
        "".join(json.dumps(jsonable(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    write_json(
        review_out,
        {
            "schema_version": "stream4d_v108_user_review_manifest_v1",
            "source_review_manifest": rel(review_manifest_path),
            "source_review_manifest_sha256": sha256_file(review_manifest_path),
            "records": user_review_rows,
            "acceptance_rule": "Only VISUALLY_ACCEPTED_FOR_DURABLE_MEMORY from explicit user review can allow durable memory.",
        },
    )
    summary_path = output_root / "phase7_visual_admission_audit_summary.json"
    summary = {
        "schema_version": "stream4d_v108_phase7_visual_admission_audit_v1",
        "created_unix_time": time.time(),
        "runtime_sec": float(time.time() - started),
        "phase6_roots": [rel(root) for root in phase6_roots],
        "anchor_roots": [rel(root) for root in anchor_roots],
        "review_manifest": rel(review_manifest_path),
        "review_manifest_sha256": sha256_file(review_manifest_path),
        "row_count": int(len(rows)),
        "durable_memory_allowed_count": int(sum(1 for row in rows if row["durable_memory_allowed"])),
        "audit_rows_csv": rel(audit_csv),
        "audit_rows_csv_sha256": sha256_file(audit_csv),
        "audit_rows_jsonl": rel(audit_jsonl),
        "audit_rows_jsonl_sha256": sha256_file(audit_jsonl),
        "user_review_manifest": rel(review_out),
        "user_review_manifest_sha256": sha256_file(review_out),
        "acceptance_rule": "Metrics are diagnostic only; durable admission requires explicit visual review acceptance plus physical/watcher checks.",
        "shadow_only": True,
    }
    write_json(summary_path, summary)
    print(json.dumps({"summary": rel(summary_path), "row_count": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
