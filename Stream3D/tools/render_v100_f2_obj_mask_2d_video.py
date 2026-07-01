#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import cv2  # type: ignore
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
STREAM3D_ROOT = REPO_ROOT / "Stream3D"
SCANNET_PROCESSED_ROOT = STREAM3D_ROOT / "data" / "scannet" / "processed"
DEFAULT_PHASE2_ROOT = STREAM3D_ROOT / "outputs" / "audit" / "v100_phase2_f2_local_final"
DEFAULT_SOURCE_REGISTRY = STREAM3D_ROOT / "outputs" / "audit" / "v95_phase1_physical_source_registry" / "source_container_rows.csv"
DEFAULT_OUTPUT_ROOT = STREAM3D_ROOT / "outputs" / "audit" / "v100_f2_obj_mask_2d_video"
VARIANT_ID = "F2_v100_chunk32_surfel_maskview_thr018_p2d2_formalized"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) & 0xFFFFFFFF


def _color_from_text(text: str, *, sat: float = 0.72, val: float = 0.95) -> tuple[int, int, int]:
    seed = _stable_seed(text)
    hue = (seed % 4096) / 4096.0
    red, green, blue = colorsys.hsv_to_rgb(hue, sat, val)
    return int(red * 255), int(green * 255), int(blue * 255)


def _project_source_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in {"Stream3D", "Open-d4rt", "docs"}:
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _to_int_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").fillna(-1).round().astype(np.int64)


def _load_v100_rows(phase2_root: Path, split: str, scene: str, chunk: str) -> pd.DataFrame:
    path = phase2_root / "mv_object_frame_mask_rows.parquet"
    rows = pd.read_parquet(path)
    rows = rows[rows["variant_id"].astype(str) == VARIANT_ID].copy()
    rows = rows[rows["dataset_split"].astype(str) == split].copy()
    rows = rows[rows["scene_id"].astype(str) == scene].copy()
    if rows.empty:
        raise RuntimeError(f"no rows for split={split} scene={scene} variant={VARIANT_ID}")
    chunks = sorted(rows["chunk_id"].astype(str).unique().tolist())
    resolved_chunk = chunks[0] if chunk == "first" else str(chunk)
    rows = rows[rows["chunk_id"].astype(str) == resolved_chunk].copy()
    if rows.empty:
        raise RuntimeError(f"no rows for split={split} scene={scene} chunk={resolved_chunk}")
    rows["frame_id_i"] = _to_int_series(rows["frame_id"])
    rows["selected_mask_id_i"] = _to_int_series(rows["selected_mask_id"])
    rows["score_f"] = pd.to_numeric(rows["score"], errors="coerce").fillna(0.0).astype(float)
    rows.attrs["resolved_chunk"] = resolved_chunk
    return rows


def _load_mask_path_by_frame(source_registry: Path) -> tuple[dict[tuple[str, str, int], Path], dict[str, Any]]:
    rows = pd.read_csv(
        source_registry,
        usecols=["scene_id", "split", "frame_id", "mask_path", "uses_gt_for_prediction", "uses_future"],
    )
    out: dict[tuple[str, str, int], Path] = {}
    duplicate_keys = 0
    source_uses_future = False
    source_uses_gt = False
    for row in rows.itertuples(index=False):
        scene = str(row.scene_id)
        split = str(row.split)
        frame_id = int(row.frame_id)
        raw = str(row.mask_path or "")
        if not raw:
            continue
        key = (split, scene, frame_id)
        path = _project_source_path(raw)
        if key in out and out[key] != path:
            duplicate_keys += 1
            continue
        out.setdefault(key, path)
        source_uses_future = source_uses_future or str(row.uses_future).strip().lower() in {"1", "true", "yes", "y"}
        source_uses_gt = source_uses_gt or str(row.uses_gt_for_prediction).strip().lower() in {"1", "true", "yes", "y"}
    return out, {
        "source_registry": _rel(source_registry),
        "source_registry_sha256": _sha256(source_registry),
        "source_registry_rows": int(len(rows)),
        "mask_path_frame_count": int(len(out)),
        "duplicate_frame_mask_path_keys": int(duplicate_keys),
        "source_uses_future": bool(source_uses_future),
        "source_uses_gt_for_prediction": bool(source_uses_gt),
    }


def _frame_assignments(frame_rows: pd.DataFrame) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    assignments: dict[int, dict[str, Any]] = {}
    duplicate_mask_object_keys = 0
    duplicate_mask_row_count = 0
    for mask_id, group in frame_rows.groupby("selected_mask_id_i", sort=True):
        group = group.copy()
        if int(group["mv_object_id"].astype(str).nunique()) > 1:
            duplicate_mask_object_keys += 1
        if len(group) > 1:
            duplicate_mask_row_count += int(len(group) - 1)
        row = group.sort_values(["score_f", "mv_object_id"], ascending=[False, True]).iloc[0]
        object_id = str(row["mv_object_id"])
        rgb = _color_from_text(f"object:{object_id}")
        assignments[int(mask_id)] = {
            "mv_object_id": object_id,
            "rgb": rgb,
            "bgr": (rgb[2], rgb[1], rgb[0]),
            "score": float(row["score_f"]),
        }
    return assignments, {
        "selected_mask_count": int(len(assignments)),
        "duplicate_mask_object_keys": int(duplicate_mask_object_keys),
        "duplicate_mask_row_count": int(duplicate_mask_row_count),
    }


def _draw_mask_contours(base: np.ndarray, hit: np.ndarray, bgr: tuple[int, int, int]) -> None:
    contours, _ = cv2.findContours(hit.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(base, contours, -1, bgr, 2, lineType=cv2.LINE_AA)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def render(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    frame_dir = output_root / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frame_dir.glob("*.png"):
        old_frame.unlink()
    rows = _load_v100_rows(Path(args.phase2_root), args.split, args.scene, args.chunk)
    resolved_chunk = str(rows.attrs["resolved_chunk"])
    mask_path_by_frame, mask_source = _load_mask_path_by_frame(Path(args.source_registry))

    scene_root = SCANNET_PROCESSED_ROOT / args.scene
    frame_ids = sorted(rows["frame_id_i"].astype(int).unique().tolist())
    object_ids = sorted(rows["mv_object_id"].astype(str).unique().tolist())
    object_color_rows = []
    for object_id in object_ids:
        rgb = _color_from_text(f"object:{object_id}")
        object_color_rows.append(
            {
                "mv_object_id": object_id,
                "rgb": ",".join(str(v) for v in rgb),
                "hex": f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}",
            }
        )

    video_path = output_root / f"{args.split}_{args.scene}_{resolved_chunk}_v100_f2_obj_mask_overlay.mp4"
    if video_path.exists():
        video_path.unlink()
    writer: cv2.VideoWriter | None = None
    frame_rows_out: list[dict[str, Any]] = []
    total_mask_pixels = 0
    total_selected_rows = 0
    missing_rgb_frames: list[int] = []
    missing_mask_frames: list[int] = []
    resized_mask_frames: list[int] = []
    duplicate_mask_object_keys = 0
    duplicate_mask_row_count = 0

    for frame_index, frame_id in enumerate(frame_ids):
        rgb_path = scene_root / "color" / f"{frame_id}.jpg"
        mask_path = mask_path_by_frame.get((args.split, args.scene, frame_id))
        image_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            missing_rgb_frames.append(frame_id)
            continue
        h, w = image_bgr.shape[:2]
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*args.codec)
            writer = cv2.VideoWriter(str(video_path), fourcc, float(args.fps), (int(w), int(h)))
            if not writer.isOpened():
                raise RuntimeError(f"failed to open video writer: {video_path} codec={args.codec}")

        frame_rows = rows[rows["frame_id_i"] == int(frame_id)].copy()
        assignments, diag = _frame_assignments(frame_rows)
        duplicate_mask_object_keys += int(diag["duplicate_mask_object_keys"])
        duplicate_mask_row_count += int(diag["duplicate_mask_row_count"])
        total_selected_rows += int(len(frame_rows))
        overlay = image_bgr.copy()
        color_layer = np.zeros_like(image_bgr)
        union = np.zeros((h, w), dtype=bool)
        frame_mask_pixels = 0
        frame_missing_mask = False
        frame_resized_mask = False

        if mask_path is None or not mask_path.is_file():
            missing_mask_frames.append(frame_id)
            frame_missing_mask = True
        else:
            label = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if label is None:
                missing_mask_frames.append(frame_id)
                frame_missing_mask = True
            else:
                if label.ndim == 3:
                    label = label[..., 0]
                if tuple(label.shape[:2]) != (h, w):
                    label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
                    resized_mask_frames.append(frame_id)
                    frame_resized_mask = True
                for mask_id, payload in assignments.items():
                    hit = label == int(mask_id)
                    pixels = int(np.count_nonzero(hit))
                    if pixels <= 0:
                        continue
                    color_layer[hit] = np.asarray(payload["bgr"], dtype=np.uint8)
                    union |= hit
                    frame_mask_pixels += pixels
                    if bool(args.draw_contours):
                        _draw_mask_contours(overlay, hit, payload["bgr"])

        if np.any(union):
            blended = cv2.addWeighted(image_bgr, 1.0 - float(args.alpha), color_layer, float(args.alpha), 0.0)
            overlay[union] = blended[union]
        frame_png = frame_dir / f"{frame_index:04d}_fid{frame_id:06d}.png"
        cv2.imwrite(str(frame_png), overlay)
        writer.write(overlay)
        total_mask_pixels += frame_mask_pixels
        frame_rows_out.append(
            {
                "frame_index": frame_index,
                "frame_id": frame_id,
                "rgb_path": _rel(rgb_path),
                "mask_path": _rel(mask_path) if mask_path else "",
                "frame_png": _rel(frame_png),
                "selected_row_count": int(len(frame_rows)),
                "selected_mask_count": int(diag["selected_mask_count"]),
                "mask_pixel_count": frame_mask_pixels,
                "missing_rgb": False,
                "missing_mask": bool(frame_missing_mask),
                "mask_resized_to_rgb": bool(frame_resized_mask),
                "height": int(h),
                "width": int(w),
            }
        )

    if writer is not None:
        writer.release()
    if writer is None:
        raise RuntimeError("no frames were rendered")

    frame_csv = output_root / "frame_render_rows.csv"
    object_csv = output_root / "object_color_rows.csv"
    _write_csv(
        frame_csv,
        frame_rows_out,
        [
            "frame_index",
            "frame_id",
            "rgb_path",
            "mask_path",
            "frame_png",
            "selected_row_count",
            "selected_mask_count",
            "mask_pixel_count",
            "missing_rgb",
            "missing_mask",
            "mask_resized_to_rgb",
            "height",
            "width",
        ],
    )
    _write_csv(object_csv, object_color_rows, ["mv_object_id", "rgb", "hex"])
    status = {
        "renderer": "render_v100_f2_obj_mask_2d_video",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "variant_id": VARIANT_ID,
        "split": args.split,
        "scene": args.scene,
        "chunk_filter": args.chunk,
        "resolved_chunk": resolved_chunk,
        "phase2_root": _rel(args.phase2_root),
        "source_registry": _rel(args.source_registry),
        "output_root": _rel(output_root),
        "video_path": _rel(video_path),
        "frame_dir": _rel(frame_dir),
        "frame_rows_csv": _rel(frame_csv),
        "object_color_rows_csv": _rel(object_csv),
        "mask_source": mask_source,
        "render_params": {
            "fps": float(args.fps),
            "alpha": float(args.alpha),
            "codec": args.codec,
            "draw_contours": bool(args.draw_contours),
            "background": "input RGB view from Stream3D/data/scannet/processed/{scene}/color/{frame}.jpg",
            "mask_coloring": "same mv_object_id keeps the same stable color across frames",
        },
        "counts": {
            "frame_count_requested": int(len(frame_ids)),
            "frame_count_rendered": int(len(frame_rows_out)),
            "v100_frame_mask_rows_selected": int(len(rows)),
            "v100_frame_mask_rows_rendered": int(total_selected_rows),
            "object_count": int(len(object_ids)),
            "total_colored_mask_pixels": int(total_mask_pixels),
            "missing_rgb_frame_count": int(len(missing_rgb_frames)),
            "missing_mask_frame_count": int(len(missing_mask_frames)),
            "resized_mask_frame_count": int(len(resized_mask_frames)),
            "duplicate_mask_object_keys": int(duplicate_mask_object_keys),
            "duplicate_mask_row_count": int(duplicate_mask_row_count),
        },
        "missing_rgb_frames": missing_rgb_frames,
        "missing_mask_frames": missing_mask_frames,
        "resized_mask_frames": resized_mask_frames,
        "gate": {
            "video_exists": video_path.is_file() and video_path.stat().st_size > 0,
            "all_requested_frames_rendered": int(len(frame_rows_out)) == int(len(frame_ids)),
            "missing_rgb_frame_count_eq_0": int(len(missing_rgb_frames)) == 0,
            "missing_mask_frame_count_eq_0": int(len(missing_mask_frames)) == 0,
            "object_count_positive": int(len(object_ids)) > 0,
            "colored_pixels_positive": int(total_mask_pixels) > 0,
        },
    }
    status["gate"]["pass"] = bool(all(status["gate"].values()))
    _write_json(output_root / "render_status.json", status)
    print(json.dumps(status, indent=2, sort_keys=True, default=_json_default), flush=True)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a 2D RGB video with v100 F2 object masks colored by mv_object_id.")
    parser.add_argument("--phase2-root", default=str(DEFAULT_PHASE2_ROOT))
    parser.add_argument("--source-registry", default=str(DEFAULT_SOURCE_REGISTRY))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--split", default="dev")
    parser.add_argument("--scene", default="scene0011_00")
    parser.add_argument("--chunk", default="first")
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--draw-contours", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    render(parse_args())


if __name__ == "__main__":
    main()
