#!/usr/bin/env python3
"""Shadow active-growth repair/demotion casebook for v108 Phase9.

This tool diagnoses growth and demotion candidates. It does not run SAM2 repair
or mutate output/memory state.
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

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "Stream3D"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from Stream3D.stream4d_v108.growth_repair import GrowthRepairPlanner  # noqa: E402


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


def resolve_path(text: str, base: Path = ROOT) -> Path:
    path = Path(text)
    if path.is_absolute():
        return path
    candidate = base / path
    if candidate.exists():
        return candidate
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
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: jsonable(row.get(key)) for key in fields})


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
    bgr = cv2.imread(str(scene_root / scene_id / "color" / f"{int(frame_id)}.jpg"), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError((scene_root / scene_id / "color" / f"{int(frame_id)}.jpg"))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def parse_ids(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


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
    return {
        "area_px": int(area),
        "area_frac": float(area / max(1, h * w)),
        "bbox_xyxy": [x0, y0, x1, y1],
        "bbox_area_frac": float((bw * bh) / max(1, h * w)),
        "bbox_extent": float(area / max(1, bw * bh)),
        "edge_touch_count": int(x0 == 0) + int(y0 == 0) + int(x1 == w - 1) + int(y1 == h - 1),
        "visible": True,
    }


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
    cv2.putText(out, text[:165], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def choose_visual_frames(rows: list[dict[str, Any]], max_frames: int) -> list[int]:
    if len(rows) <= int(max_frames):
        return [int(row["frame_id"]) for row in rows]
    max_frames = int(max_frames)
    keep: set[int] = {int(rows[0]["frame_id"]), int(rows[-1]["frame_id"])}
    visible_rows = [row for row in rows if bool(row.get("visible", False))]
    if visible_rows:
        keep.add(int(visible_rows[0]["frame_id"]))
        keep.add(int(visible_rows[-1]["frame_id"]))
        last_visible_index = max(idx for idx, row in enumerate(rows) if bool(row.get("visible", False)))
        for row in rows[last_visible_index + 1 :]:
            if not bool(row.get("visible", False)):
                keep.add(int(row["frame_id"]))
                break
    scored = []
    previous_visible = False
    for row in rows:
        score = 0.0
        score += abs(float(row.get("area_ratio_to_previous", 1.0)) - 1.0)
        score += 0.5 * int(row.get("edge_touch_count", 0))
        score += 0.25 * float(row.get("bbox_area_frac", 0.0))
        visible = bool(row.get("visible", False))
        if not visible and previous_visible:
            score += 0.25
        if not visible and not previous_visible:
            score -= 1.0
        scored.append((score, int(row["frame_id"])))
        previous_visible = visible
    for _score, frame_id in sorted(scored, reverse=True):
        keep.add(int(frame_id))
        if len(keep) >= max_frames:
            break
    if len(keep) > max_frames:
        ranked = [frame_id for _score, frame_id in sorted(scored, reverse=True)]
        must_keep = [int(rows[0]["frame_id"]), int(rows[-1]["frame_id"])]
        selected: list[int] = []
        for frame_id in must_keep + ranked:
            if frame_id in keep and frame_id not in selected:
                selected.append(frame_id)
            if len(selected) >= max_frames:
                break
        keep = set(selected)
    return sorted(keep)


def build_panel(
    *,
    scene_root: Path,
    scene_id: str,
    object_id: int,
    frame_ids: list[int],
    labels_by_frame: dict[int, np.ndarray],
    rows_by_frame: dict[int, dict[str, Any]],
    output_root: Path,
) -> Path:
    panels = []
    for frame_id in frame_ids:
        rgb = load_rgb(scene_root, scene_id, int(frame_id))
        mask = labels_by_frame[int(frame_id)] == int(object_id)
        panel = overlay_mask(rgb, mask, color=(40, 220, 255), alpha=0.42)
        row = rows_by_frame[int(frame_id)]
        text = (
            f"frame {frame_id} obj {object_id}; area={int(row['area_px'])}; "
            f"hist_ratio={float(row['area_ratio_to_history_median']):.2f}; "
            f"edge={int(row['edge_touch_count'])}; action={str(row['suggested_action'])[:36]}"
        )
        panels.append(add_header(panel, text))
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
    out_path = output_root / "visual_checks" / f"growth_obj{int(object_id):04d}_{frame_ids[0]:06d}_{frame_ids[-1]:06d}.png"
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
    parser.add_argument("--max-visual-frames", type=int, default=6)
    parser.add_argument("--growth-ratio-alert", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()
    scene_root = resolve_path(str(args.scene_root))
    reference_root = resolve_path(str(args.reference_run_root))
    output_root = resolve_path(str(args.output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "last_command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n",
        encoding="utf-8",
    )
    records = load_reference_records(reference_root)
    frames = [
        int(fid)
        for fid in sorted(records)
        if int(args.start_frame_id) <= int(fid) <= int(args.end_frame_id)
        and (int(fid) - int(args.start_frame_id)) % max(1, int(args.frame_stride)) == 0
    ]
    if not frames:
        raise RuntimeError("no frames selected")
    labels_by_frame = {int(fid): load_label(Path(records[int(fid)]["label_path"])) for fid in frames}
    object_ids = parse_ids(str(args.object_ids))
    planner = GrowthRepairPlanner()
    all_rows: list[dict[str, Any]] = []
    alert_rows: list[dict[str, Any]] = []
    track_summaries: list[dict[str, Any]] = []
    visual_paths: list[Path] = []
    for object_id in object_ids:
        previous_area = -1
        history_areas: list[int] = []
        rows: list[dict[str, Any]] = []
        rows_by_frame: dict[int, dict[str, Any]] = {}
        for frame_id in frames:
            stats = mask_stats(labels_by_frame[int(frame_id)] == int(object_id))
            area = int(stats["area_px"])
            previous_nonzero = previous_area if previous_area > 0 else -1
            area_ratio_prev = float(area / previous_nonzero) if previous_nonzero > 0 and area > 0 else (-1.0 if area == 0 else 1.0)
            hist_vals = [v for v in history_areas if v > 0]
            hist_median = float(np.median(hist_vals)) if hist_vals else float(area if area > 0 else 1)
            area_ratio_history = float(area / max(hist_median, 1.0)) if area > 0 else -1.0
            suggestion = planner.suggest_from_shadow_stats(
                frame_id=int(frame_id),
                global_object_id=int(object_id),
                visible=bool(stats["visible"]),
                edge_touch_count=int(stats["edge_touch_count"]),
                area_ratio_to_history=float(area_ratio_history),
                bbox_area_fraction=float(stats["bbox_area_frac"]),
            )
            row = {
                "scene_id": str(args.scene_id),
                "object_id": int(object_id),
                "frame_id": int(frame_id),
                **stats,
                "area_ratio_to_previous": float(area_ratio_prev),
                "history_area_median_px": float(hist_median),
                "area_ratio_to_history_median": float(area_ratio_history),
                "growth_alert": bool(area_ratio_history >= float(args.growth_ratio_alert) and area > 0),
                "candidate_A_current_mask": "available" if area > 0 else "empty",
                "candidate_B_sam2_repair": "not_run_shadow_diagnostic",
                "suggested_action": str(suggestion.action),
                "suggested_reason": str(suggestion.reason),
                "visual_review_required": True,
                "metrics_are_diagnostic_only": True,
            }
            rows.append(row)
            rows_by_frame[int(frame_id)] = row
            all_rows.append(row)
            if bool(row["growth_alert"]) or str(suggestion.action) != "keep_output_probation_until_visual_review":
                alert_rows.append(row)
            if area > 0:
                previous_area = int(area)
                history_areas.append(int(area))
        visual_frames = choose_visual_frames(rows, int(args.max_visual_frames))
        visual_path = build_panel(
            scene_root=scene_root,
            scene_id=str(args.scene_id),
            object_id=int(object_id),
            frame_ids=visual_frames,
            labels_by_frame=labels_by_frame,
            rows_by_frame=rows_by_frame,
            output_root=output_root,
        )
        visual_paths.append(visual_path)
        track_summaries.append(
            {
                "scene_id": str(args.scene_id),
                "object_id": int(object_id),
                "frame_ids": frames,
                "visual_frame_ids": visual_frames,
                "visual_path": rel(visual_path),
                "visual_sha256": sha256_file(visual_path),
                "alert_count": int(sum(1 for row in rows if bool(row["growth_alert"]))),
                "actions": sorted({str(row["suggested_action"]) for row in rows}),
                "visual_review_required": True,
            }
        )
    rows_csv = output_root / "growth_timeline_rows.csv"
    alerts_csv = output_root / "growth_alert_rows.csv"
    track_path = output_root / "track_summaries.json"
    write_csv(rows_csv, all_rows)
    write_csv(alerts_csv, alert_rows)
    write_json(track_path, {"tracks": track_summaries})
    summary_path = output_root / "phase9_growth_repair_shadow_summary.json"
    summary = {
        "schema_version": "stream4d_v108_phase9_growth_repair_shadow_v1",
        "created_unix_time": time.time(),
        "runtime_sec": float(time.time() - started),
        "scene_id": str(args.scene_id),
        "reference_run_root": rel(reference_root),
        "object_ids": [int(v) for v in object_ids],
        "frame_ids": frames,
        "row_count": int(len(all_rows)),
        "alert_row_count": int(len(alert_rows)),
        "growth_ratio_alert": float(args.growth_ratio_alert),
        "rows_csv": rel(rows_csv),
        "rows_csv_sha256": sha256_file(rows_csv),
        "alerts_csv": rel(alerts_csv),
        "alerts_csv_sha256": sha256_file(alerts_csv),
        "track_summaries": rel(track_path),
        "track_summaries_sha256": sha256_file(track_path),
        "visual_paths": [rel(path) for path in visual_paths],
        "visual_sha256": {rel(path): sha256_file(path) for path in visual_paths},
        "candidate_A": "current mask snapshot in each visual panel",
        "candidate_B": "not_run_shadow_diagnostic",
        "acceptance_rule": "Metrics and growth rows are diagnostic only; quality must be judged by high-resolution visual review.",
        "shadow_only": True,
    }
    write_json(summary_path, summary)
    print(json.dumps({"summary": rel(summary_path), "row_count": len(all_rows), "alert_row_count": len(alert_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
