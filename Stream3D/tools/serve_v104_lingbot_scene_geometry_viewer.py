#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from geometry_provider.lingbot_map_provider import LingBotMapGeometryProvider  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
DEFAULT_SCENE = "scene0011_00"
DEFAULT_BSS_SUMMARY = AUDIT_ROOT / "v104_lingbot_map_only_phase6_full_bss_scene0011" / "summary.json"
DEFAULT_OUT = AUDIT_ROOT / "v104_lingbot_scene0011_full_scene_viser"


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return p.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _frame_ids_from_sampling(path: Path) -> tuple[list[int], list[int]]:
    payload = _read_json(path)
    source_frames = [int(v) for v in payload.get("frames", [])]
    return list(range(len(source_frames))), source_frames


def _rgb_candidates(gt_root: Path, bss_frame_id: int, source_frame_id: int | None) -> list[Path]:
    ids = [int(bss_frame_id)]
    if source_frame_id is not None and int(source_frame_id) not in ids:
        ids.append(int(source_frame_id))
    out: list[Path] = []
    for frame_id in ids:
        key = f"{frame_id:06d}"
        for root in (gt_root / "rgb", gt_root):
            for suffix in (".png", ".jpg", ".jpeg"):
                out.append(root / f"{key}{suffix}")
    return out


def _load_rgb(gt_root: Path, bss_frame_id: int, source_frame_id: int | None) -> np.ndarray | None:
    for path in _rgb_candidates(gt_root, bss_frame_id, source_frame_id):
        if not path.exists():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return None


def _frame_palette(frame_id: int) -> np.ndarray:
    seed = (int(frame_id) * 1103515245 + 12345) & 0xFFFFFFFF
    r = 60 + ((seed >> 0) & 0x7F)
    g = 60 + ((seed >> 8) & 0x7F)
    b = 60 + ((seed >> 16) & 0x7F)
    return np.asarray([r, g, b], dtype=np.uint8)


def _sample_rgb_at_xy(
    *,
    image: np.ndarray | None,
    xy: np.ndarray | None,
    image_shape: tuple[int, int] | None,
    keep: np.ndarray,
    bss_frame_id: int,
) -> tuple[np.ndarray, str]:
    if image is None or xy is None or xy.size == 0:
        color = np.broadcast_to(_frame_palette(bss_frame_id)[None, :], (keep.shape[0], 3)).copy()
        return color.astype(np.uint8), "frame_palette_fallback"
    pts = np.asarray(xy, dtype=np.float32)[keep]
    ih, iw = image.shape[:2]
    if image_shape is None:
        sh, sw = ih, iw
    else:
        sh, sw = image_shape
    scale_x = float(max(iw - 1, 1)) / float(max(sw - 1, 1))
    scale_y = float(max(ih - 1, 1)) / float(max(sh - 1, 1))
    x = np.rint(pts[:, 0] * scale_x).astype(np.int64)
    y = np.rint(pts[:, 1] * scale_y).astype(np.int64)
    valid = (x >= 0) & (x < iw) & (y >= 0) & (y < ih)
    colors = np.broadcast_to(_frame_palette(bss_frame_id)[None, :], (keep.shape[0], 3)).copy()
    colors[valid] = image[y[valid], x[valid], :3]
    return colors.astype(np.uint8), "rgb_xy"


def _select_frames(frame_ids: list[int], *, frame_step: int, max_frames: int) -> list[int]:
    selected = frame_ids[:: max(1, int(frame_step))]
    if max_frames > 0 and len(selected) > max_frames:
        idx = np.linspace(0, len(selected) - 1, int(max_frames), dtype=np.int64)
        selected = [selected[int(i)] for i in idx]
    return selected


def _build_payload(args: argparse.Namespace) -> dict[str, Any]:
    bss_summary_path = _project(args.bss_summary)
    bss_summary = _read_json(bss_summary_path)
    scene_id = str(args.scene_id or bss_summary.get("scene_id", DEFAULT_SCENE))
    lingbot_root = _project(args.lingbot_root or bss_summary["lingbot_root"])
    gt_root = _project(args.gt_root or bss_summary["gt_root"])
    sampling_json = _project(args.sampling_json or bss_summary["sampling_json"])
    all_bss_frame_ids, source_frames = _frame_ids_from_sampling(sampling_json)
    selected_bss_frame_ids = _select_frames(all_bss_frame_ids, frame_step=int(args.frame_step), max_frames=int(args.max_frames))

    provider = LingBotMapGeometryProvider(
        geometry_root=lingbot_root,
        max_points_per_frame=int(args.provider_max_points_per_frame),
        min_confidence=args.min_confidence,
        sampling_json=sampling_json,
    )
    rng = np.random.default_rng(int(args.seed))
    points_parts: list[np.ndarray] = []
    colors_parts: list[np.ndarray] = []
    frame_id_parts: list[np.ndarray] = []
    loaded_frames = 0
    empty_frames = 0
    rgb_frames = 0
    fallback_frames = 0

    for bss_frame_id in selected_bss_frame_ids:
        samples = provider.load_frame_samples(int(bss_frame_id))
        points = np.asarray(samples.points, dtype=np.float32).reshape(-1, 3)
        finite = np.isfinite(points).all(axis=1)
        valid_idx = np.flatnonzero(finite)
        if valid_idx.size == 0:
            empty_frames += 1
            continue
        if valid_idx.size > int(args.points_per_frame):
            keep = rng.choice(valid_idx, size=int(args.points_per_frame), replace=False)
            keep.sort()
        else:
            keep = valid_idx
        source_frame_id = int(source_frames[int(bss_frame_id)]) if int(bss_frame_id) < len(source_frames) else samples.source_frame_id
        rgb = _load_rgb(gt_root, int(bss_frame_id), source_frame_id)
        colors, color_source = _sample_rgb_at_xy(
            image=rgb,
            xy=samples.xy,
            image_shape=samples.image_shape,
            keep=keep,
            bss_frame_id=int(bss_frame_id),
        )
        if color_source == "rgb_xy":
            rgb_frames += 1
        else:
            fallback_frames += 1
        points_parts.append(points[keep])
        colors_parts.append(colors)
        frame_id_parts.append(np.full((keep.shape[0],), int(bss_frame_id), dtype=np.int32))
        loaded_frames += 1

    if not points_parts:
        raise RuntimeError(f"no LingBot points loaded for scene={scene_id} from {lingbot_root}")
    points = np.concatenate(points_parts, axis=0).astype(np.float32, copy=False)
    colors = np.concatenate(colors_parts, axis=0).astype(np.uint8, copy=False)
    frame_ids = np.concatenate(frame_id_parts, axis=0).astype(np.int32, copy=False)
    if points.shape[0] > int(args.max_total_points):
        keep = rng.choice(np.arange(points.shape[0]), size=int(args.max_total_points), replace=False)
        keep.sort()
        points = points[keep]
        colors = colors[keep]
        frame_ids = frame_ids[keep]

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    cache_npz = _project(args.cache_npz)
    cache_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_npz,
        points=points,
        colors=colors,
        bss_frame_ids=frame_ids,
        selected_bss_frame_ids=np.asarray(selected_bss_frame_ids, dtype=np.int32),
        source_frames=np.asarray(source_frames, dtype=np.int32),
        bbox_min=bbox_min.astype(np.float32),
        bbox_max=bbox_max.astype(np.float32),
    )
    return {
        "scene_id": scene_id,
        "bss_summary": _rel(bss_summary_path),
        "lingbot_root": _rel(lingbot_root),
        "gt_root": _rel(gt_root),
        "rgb_root": _rel(gt_root / "rgb"),
        "sampling_json": _rel(sampling_json),
        "cache_npz": _rel(cache_npz),
        "total_bss_frame_count": int(len(all_bss_frame_ids)),
        "selected_bss_frame_count": int(len(selected_bss_frame_ids)),
        "loaded_frame_count": int(loaded_frames),
        "empty_frame_count": int(empty_frames),
        "rgb_colored_frame_count": int(rgb_frames),
        "fallback_colored_frame_count": int(fallback_frames),
        "point_count": int(points.shape[0]),
        "points_per_frame_cap": int(args.points_per_frame),
        "provider_max_points_per_frame": int(args.provider_max_points_per_frame),
        "frame_step": int(args.frame_step),
        "max_frames": int(args.max_frames),
        "max_total_points": int(args.max_total_points),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "color_source": "gt/rgb sampled at LingBot projected xy; frame palette fallback only if RGB/xy missing",
        "whole_scene_not_chunk": True,
    }


def _load_cached_payload(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cache_npz = _project(args.cache_npz)
    if not cache_npz.exists():
        status = _build_payload(args)
    else:
        status_path = _project(args.status_json) if str(args.status_json).strip() else cache_npz.with_suffix(".status.json")
        status = _read_json(status_path) if status_path.exists() else {"cache_npz": _rel(cache_npz)}
    with np.load(cache_npz) as payload:
        points = np.asarray(payload["points"], dtype=np.float32)
        colors = np.asarray(payload["colors"], dtype=np.uint8)
        if "bbox_min" in payload and "bbox_max" in payload:
            status["bbox_min"] = np.asarray(payload["bbox_min"], dtype=np.float32)
            status["bbox_max"] = np.asarray(payload["bbox_max"], dtype=np.float32)
    status["point_count"] = int(points.shape[0])
    status["cache_npz"] = _rel(cache_npz)
    return points, colors, status


def serve(args: argparse.Namespace) -> dict[str, Any]:
    status: dict[str, Any] | None = None
    if bool(args.rebuild_cache) or not _project(args.cache_npz).exists():
        status = _build_payload(args)
        if str(args.status_json).strip():
            _write_json(_project(args.status_json), status)
    if bool(args.build_cache_only):
        if status is None:
            status_path = _project(args.status_json) if str(args.status_json).strip() else _project(args.cache_npz).with_suffix(".status.json")
            status = _read_json(status_path) if status_path.exists() else {"cache_npz": _rel(_project(args.cache_npz))}
        return status

    points, colors, status = _load_cached_payload(args)

    import viser  # type: ignore

    server = viser.ViserServer(host=str(args.host), port=int(args.port), verbose=True)
    server.scene.set_up_direction("+z")
    bbox_min = np.asarray(status.get("bbox_min", points.min(axis=0)), dtype=np.float32)
    bbox_max = np.asarray(status.get("bbox_max", points.max(axis=0)), dtype=np.float32)
    center = (bbox_min + bbox_max) * 0.5
    extent = np.maximum(bbox_max - bbox_min, 1e-3)
    grid_size = float(max(4.0, np.max(extent[:2]) * 1.25))
    server.scene.add_grid(
        "/v104_lingbot_scene/grid",
        width=grid_size,
        height=grid_size,
        plane="xy",
        cell_size=0.5,
        section_size=2.0,
        position=(float(center[0]), float(center[1]), float(bbox_min[2] - 0.02)),
    )
    cloud_handle = server.scene.add_point_cloud(
        f"/v104_lingbot_scene/{status.get('scene_id', args.scene_id)} full-scene LingBot RGB points",
        points=points,
        colors=colors,
        point_size=float(args.point_size),
        point_shape="circle",
        visible=True,
        precision="float32",
    )
    toggle = server.gui.add_checkbox("LingBot full-scene RGB point cloud", True)
    point_size = server.gui.add_slider("Point size", min=0.001, max=0.05, step=0.001, initial_value=float(args.point_size))

    @toggle.on_update
    def _(_: Any) -> None:
        cloud_handle.visible = bool(toggle.value)

    @point_size.on_update
    def _(_: Any) -> None:
        cloud_handle.point_size = float(point_size.value)

    status.update(
        {
            "viewer": "v104_lingbot_full_scene_rgb_geometry",
            "url": f"http://localhost:{int(args.port)}",
            "host": str(args.host),
            "port": int(args.port),
            "point_size": float(args.point_size),
            "layer": "LingBot full-scene RGB point cloud",
        }
    )
    if str(args.status_json).strip():
        _write_json(_project(args.status_json), status)
    print(json.dumps(_jsonable(status), indent=2, sort_keys=True), flush=True)

    stop = {"value": False}

    def _stop(_signum: int, _frame: Any) -> None:
        stop["value"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    while not stop["value"]:
        time.sleep(1.0)
    server.stop()
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a full-scene RGB-colored LingBot map geometry point cloud in Viser.")
    parser.add_argument("--scene-id", default=DEFAULT_SCENE)
    parser.add_argument("--bss-summary", default=str(DEFAULT_BSS_SUMMARY))
    parser.add_argument("--lingbot-root", default="")
    parser.add_argument("--gt-root", default="")
    parser.add_argument("--sampling-json", default="")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--cache-npz", default=str(DEFAULT_OUT / "scene0011_00_lingbot_full_scene_rgb_points.npz"))
    parser.add_argument("--status-json", default=str(DEFAULT_OUT / "viewer_status.json"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--points-per-frame", type=int, default=300)
    parser.add_argument("--provider-max-points-per-frame", type=int, default=5000)
    parser.add_argument("--max-total-points", type=int, default=150000)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--point-size", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=10317)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--build-cache-only", action="store_true")
    args = parser.parse_args()
    serve(args)


if __name__ == "__main__":
    main()
