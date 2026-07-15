#!/usr/bin/env python3
"""Shadow watcher/probation diagnostic for v108 delayed durable admission."""

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

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "Stream3D"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from Stream3D.stream4d_v108.lifecycle import DelayedAdmissionPolicy  # noqa: E402


DEFAULT_REFERENCE_ROOT = (
    ROOT
    / "Stream3D/outputs/audit/v108_phase1_reference_scene0050_90f_labelonly_20260714_1442"
    / "v106_stateful_sam2_rolling_scene_stream"
)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: jsonable(row.get(key)) for key in fields})


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(path_text: str, base: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidate = base / path
    if candidate.exists():
        return candidate
    return ROOT / path


def load_reference_records(reference_root: Path) -> dict[int, dict[str, Any]]:
    summary_path = reference_root / "summary.json"
    if not summary_path.exists():
        nested = reference_root / "v106_stateful_sam2_rolling_scene_stream" / "summary.json"
        if nested.exists():
            summary_path = nested
    summary = read_json(summary_path)
    records: dict[int, dict[str, Any]] = {}
    for row in summary.get("records", []):
        item = dict(row)
        item["label_path"] = resolve_path(str(row["label_path"]), summary_path.parent)
        records[int(row["frame_id"])] = item
    return records


def load_label(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.int32, copy=False)


def load_rgb(scene_root: Path, scene_id: str, frame_id: int) -> np.ndarray:
    path = scene_root / scene_id / "color" / f"{int(frame_id)}.jpg"
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def parse_ids(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def mask_stats(mask: np.ndarray) -> dict[str, Any]:
    mask_b = np.asarray(mask).astype(bool)
    h, w = mask_b.shape[:2]
    area = int(np.count_nonzero(mask_b))
    if area <= 0:
        return {
            "area_px": 0,
            "area_frac": 0.0,
            "bbox_xyxy": [],
            "bbox_area_frac": 0.0,
            "bbox_extent": 0.0,
            "edge_touch_count": 0,
            "visible": False,
        }
    ys, xs = np.where(mask_b)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    edge_touch = int(x0 == 0) + int(y0 == 0) + int(x1 == w - 1) + int(y1 == h - 1)
    return {
        "area_px": int(area),
        "area_frac": float(area / max(1, h * w)),
        "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "bbox_area_frac": float((bw * bh) / max(1, h * w)),
        "bbox_extent": float(area / max(1, bw * bh)),
        "edge_touch_count": int(edge_touch),
        "visible": True,
    }


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.count_nonzero(np.asarray(a).astype(bool) & np.asarray(b).astype(bool)))
    union = int(np.count_nonzero(np.asarray(a).astype(bool) | np.asarray(b).astype(bool)))
    return float(inter / max(union, 1))


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, *, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = rgb.copy()
    mask_b = np.asarray(mask).astype(bool)
    if np.any(mask_b):
        c = np.asarray(color, dtype=np.float32)
        out[mask_b] = ((1.0 - float(alpha)) * out[mask_b].astype(np.float32) + float(alpha) * c).clip(0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_b.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2, lineType=cv2.LINE_AA)
    return out


def add_header(image: np.ndarray, text: str) -> np.ndarray:
    header = 34
    out = np.zeros((image.shape[0] + header, image.shape[1], 3), dtype=np.uint8)
    out[:] = 12
    out[header:] = image
    cv2.putText(out, text[:170], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def frame_ids_in_range(records: dict[int, dict[str, Any]], start: int, end: int, stride: int) -> list[int]:
    wanted = []
    for fid in sorted(records):
        if int(fid) < int(start) or int(fid) > int(end):
            continue
        if (int(fid) - int(start)) % max(1, int(stride)) == 0:
            wanted.append(int(fid))
    return wanted


def build_visual_panel(
    *,
    scene_root: Path,
    scene_id: str,
    object_id: int,
    frame_ids: list[int],
    labels_by_frame: dict[int, np.ndarray],
    rows_by_frame: dict[int, dict[str, Any]],
    output_root: Path,
    max_visual_frames: int,
) -> Path:
    selected = frame_ids[: int(max_visual_frames)]
    panels = []
    for fid in selected:
        rgb = load_rgb(scene_root, scene_id, int(fid))
        mask = labels_by_frame[int(fid)] == int(object_id)
        panel = overlay_mask(rgb, mask, color=(40, 220, 255), alpha=0.42)
        row = rows_by_frame[int(fid)]
        text = (
            f"frame {fid} obj {object_id}; area={int(row['area_px'])}; "
            f"edge={int(row['edge_touch_count'])}; prev_iou={float(row['iou_to_previous_visible']):.3f}"
        )
        panels.append(add_header(panel, text))
    if not panels:
        raise RuntimeError("no visual frames selected")
    height = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] == height:
            padded.append(panel)
            continue
        out = np.zeros((height, panel.shape[1], 3), dtype=np.uint8)
        out[:] = 12
        out[: panel.shape[0], : panel.shape[1]] = panel
        padded.append(out)
    merged = np.concatenate(padded, axis=1)
    out_path = output_root / "visual_checks" / f"track_obj{int(object_id):04d}_{selected[0]:06d}_{selected[-1]:06d}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(merged, cv2.COLOR_RGB2BGR))
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", default="scene0050_00")
    parser.add_argument("--scene-root", default="Stream3D/data/scannet/processed")
    parser.add_argument("--reference-run-root", default=str(DEFAULT_REFERENCE_ROOT))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--object-ids", required=True)
    parser.add_argument("--start-frame-id", type=int, required=True)
    parser.add_argument("--end-frame-id", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--max-visual-frames", type=int, default=5)
    parser.add_argument("--max-durable-edge-touch-count", type=int, default=1)
    parser.add_argument("--max-durable-bbox-area-frac", type=float, default=0.35)
    parser.add_argument("--max-durable-area-frac", type=float, default=0.25)
    parser.add_argument("--min-durable-extent", type=float, default=0.25)
    parser.add_argument("--min-watcher-visible-frames", type=int, default=2)
    parser.add_argument("--min-watcher-mean-iou", type=float, default=0.40)
    return parser.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()
    scene_root = resolve_path(str(args.scene_root), ROOT)
    reference_root = resolve_path(str(args.reference_run_root), ROOT)
    output_root = resolve_path(str(args.output_root), ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "last_command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n",
        encoding="utf-8",
    )
    records = load_reference_records(reference_root)
    object_ids = parse_ids(str(args.object_ids))
    frames = frame_ids_in_range(records, int(args.start_frame_id), int(args.end_frame_id), int(args.frame_stride))
    if not frames:
        raise RuntimeError("no frames selected")

    labels_by_frame: dict[int, np.ndarray] = {
        int(fid): load_label(Path(records[int(fid)]["label_path"])) for fid in frames
    }
    policy = DelayedAdmissionPolicy(
        max_durable_edge_touch_count=int(args.max_durable_edge_touch_count),
        max_durable_bbox_area_frac=float(args.max_durable_bbox_area_frac),
        max_durable_area_frac=float(args.max_durable_area_frac),
        min_durable_extent=float(args.min_durable_extent),
        min_watcher_visible_frames=int(args.min_watcher_visible_frames),
        min_watcher_mean_iou=float(args.min_watcher_mean_iou),
    )

    frame_rows: list[dict[str, Any]] = []
    track_summaries: list[dict[str, Any]] = []
    admission_rows: list[dict[str, Any]] = []
    visual_paths: list[Path] = []
    for oid in object_ids:
        previous_visible_mask: np.ndarray | None = None
        ious: list[float] = []
        visible_frames: list[int] = []
        rows_by_frame: dict[int, dict[str, Any]] = {}
        first_component_stats: dict[str, Any] | None = None
        for fid in frames:
            mask = labels_by_frame[int(fid)] == int(oid)
            stats = mask_stats(mask)
            iou_prev = -1.0
            if previous_visible_mask is not None and bool(stats["visible"]):
                iou_prev = iou(previous_visible_mask, mask)
                ious.append(float(iou_prev))
            if bool(stats["visible"]):
                previous_visible_mask = mask
                visible_frames.append(int(fid))
            if first_component_stats is None and int(fid) == int(args.start_frame_id):
                first_component_stats = dict(stats)
            row = {
                "scene_id": str(args.scene_id),
                "object_id": int(oid),
                "frame_id": int(fid),
                "iou_to_previous_visible": float(iou_prev),
                **stats,
            }
            rows_by_frame[int(fid)] = row
            frame_rows.append(row)
        if first_component_stats is None:
            first_component_stats = dict(rows_by_frame[frames[0]])
        watcher_stats = {
            "visible_frame_count": int(len(visible_frames)),
            "first_visible_frame_id": int(visible_frames[0]) if visible_frames else -1,
            "last_visible_frame_id": int(visible_frames[-1]) if visible_frames else -1,
            "mean_iou_to_previous_visible": float(np.mean(ious)) if ious else -1.0,
            "min_iou_to_previous_visible": float(np.min(ious)) if ious else -1.0,
            "max_iou_to_previous_visible": float(np.max(ious)) if ious else -1.0,
        }
        decision = policy.evaluate(
            frame_id=int(args.start_frame_id),
            global_object_id=int(oid),
            component_stats=first_component_stats,
            watcher_stats=watcher_stats,
            visual_review_status="USER_REVIEW_PENDING",
        )
        admission = {
            "scene_id": str(args.scene_id),
            "object_id": int(oid),
            "frame_id": int(args.start_frame_id),
            "output_state": str(decision.output_state.value),
            "output_allowed": bool(decision.output_allowed),
            "durable_memory_allowed": bool(decision.durable_memory_allowed),
            "reasons": list(decision.reasons),
            "visual_review_required": bool(decision.visual_review_required),
            "diagnostic_only": bool(decision.diagnostic_only),
            **{f"watcher_{key}": val for key, val in watcher_stats.items()},
        }
        admission_rows.append(admission)
        visual_path = build_visual_panel(
            scene_root=scene_root,
            scene_id=str(args.scene_id),
            object_id=int(oid),
            frame_ids=frames,
            labels_by_frame=labels_by_frame,
            rows_by_frame=rows_by_frame,
            output_root=output_root,
            max_visual_frames=int(args.max_visual_frames),
        )
        visual_paths.append(visual_path)
        track_summaries.append(
            {
                "scene_id": str(args.scene_id),
                "object_id": int(oid),
                "frame_ids": frames,
                "watcher_stats": watcher_stats,
                "admission": admission,
                "visual_path": rel(visual_path),
                "visual_sha256": sha256_file(visual_path),
                "visual_review_required": True,
            }
        )

    frame_rows_csv = output_root / "watcher_frame_rows.csv"
    admission_csv = output_root / "admission_shadow_rows.csv"
    write_csv(frame_rows_csv, frame_rows)
    write_csv(admission_csv, admission_rows)
    track_path = output_root / "track_summaries.json"
    write_json(track_path, {"tracks": track_summaries})
    summary_path = output_root / "phase6_probation_watcher_shadow_summary.json"
    summary = {
        "schema_version": "stream4d_v108_phase6_probation_watcher_shadow_v1",
        "created_unix_time": time.time(),
        "runtime_sec": float(time.time() - started),
        "scene_id": str(args.scene_id),
        "reference_run_root": rel(reference_root),
        "object_ids": [int(v) for v in object_ids],
        "frame_ids": frames,
        "frame_row_count": int(len(frame_rows)),
        "admission_row_count": int(len(admission_rows)),
        "track_count": int(len(track_summaries)),
        "frame_rows_csv": rel(frame_rows_csv),
        "frame_rows_csv_sha256": sha256_file(frame_rows_csv),
        "admission_csv": rel(admission_csv),
        "admission_csv_sha256": sha256_file(admission_csv),
        "track_summaries": rel(track_path),
        "track_summaries_sha256": sha256_file(track_path),
        "visual_paths": [rel(path) for path in visual_paths],
        "visual_sha256": {rel(path): sha256_file(path) for path in visual_paths},
        "policy": {
            "max_durable_edge_touch_count": int(args.max_durable_edge_touch_count),
            "max_durable_bbox_area_frac": float(args.max_durable_bbox_area_frac),
            "max_durable_area_frac": float(args.max_durable_area_frac),
            "min_durable_extent": float(args.min_durable_extent),
            "min_watcher_visible_frames": int(args.min_watcher_visible_frames),
            "min_watcher_mean_iou": float(args.min_watcher_mean_iou),
        },
        "acceptance_rule": "Metrics and watcher rows are diagnostic only; quality must be judged by high-resolution visual review.",
        "shadow_only": True,
    }
    write_json(summary_path, summary)
    print(json.dumps({"summary": rel(summary_path), "track_count": len(track_summaries)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
