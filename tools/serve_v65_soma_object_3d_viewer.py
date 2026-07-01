#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
STREAM3D_ROOT = REPO_ROOT / "Stream3D"
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from geometry_provider.d4rt_carrier_provider import D4RTCarrierProjectionProvider, _apply_fit
from stream4d.scannet_stream import ScanNetStream
from stream4d_native.v65_common import rel, sha256_file
from stream4d_native.v65_visualization_export import _id_colors, _load_gt, _load_scene_mesh


REQUIRED_LAYERS = ["GT geo", "GT sem", "D4RT geo", "SOMA sem"]


def _stable_seed(text: str) -> int:
    value = 2166136261
    for byte in text.encode("utf-8"):
        value ^= int(byte)
        value = (value * 16777619) & 0xFFFFFFFF
    return int(value)


def _sample_indices(count: int, max_count: int, seed: str) -> np.ndarray:
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    if max_count <= 0 or count <= max_count:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(_stable_seed(seed))
    return np.sort(rng.choice(count, size=int(max_count), replace=False).astype(np.int64))


def _existing_mask_frames(scene: str, backbone: str) -> set[int]:
    mask_dir = STREAM3D_ROOT / "data" / "scannet" / "processed" / scene / f"output_{backbone}" / "mask"
    return {int(path.stem) for path in mask_dir.glob("*.png") if path.stem.isdigit()}


def _sample_rgb_colors(scene: str, frame_ids: np.ndarray, uv: np.ndarray) -> np.ndarray:
    stream = ScanNetStream(seq_name=scene, root=STREAM3D_ROOT / "data" / "scannet" / "processed")
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    uv = np.asarray(uv, dtype=np.float32)
    colors = np.zeros((frame_ids.shape[0], 3), dtype=np.uint8)
    for frame_id in sorted(set(frame_ids.tolist())):
        sel = frame_ids == int(frame_id)
        rgb = stream.load_rgb(int(frame_id))
        h, w = rgb.shape[:2]
        xy = np.rint(
            np.stack(
                [
                    uv[sel, 0] * float(max(w - 1, 1)),
                    uv[sel, 1] * float(max(h - 1, 1)),
                ],
                axis=1,
            )
        ).astype(np.int64)
        xy[:, 0] = np.clip(xy[:, 0], 0, max(w - 1, 0))
        xy[:, 1] = np.clip(xy[:, 1], 0, max(h - 1, 0))
        colors[sel] = rgb[xy[:, 1], xy[:, 0], :3]
    return colors


def _load_d4rt_rgb_uv(
    *,
    scene: str,
    debug_root: Path,
    mode: str,
    frame_stride: int,
    backbone: str,
    min_visibility: float,
    min_confidence: float,
    max_anchors: int,
    stitch_uv_radius: float,
    stitch_max_matches_per_frame: int,
    stitch_fit_trim_percentile: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    debug_root = debug_root if debug_root.is_absolute() else (REPO_ROOT / debug_root).resolve()
    provider = D4RTCarrierProjectionProvider(
        debug_root=debug_root,
        mode=mode,
        nn_radius=0.05,
        min_visibility=float(min_visibility),
        min_confidence=float(min_confidence),
        max_anchors=int(max_anchors),
        robust_trim_percentile=90.0,
        density_alpha=2.0,
        overlap_policy="best_confidence",
        stitch_uv_radius=float(stitch_uv_radius),
        stitch_max_matches_per_frame=int(stitch_max_matches_per_frame),
        stitch_fit_trim_percentile=float(stitch_fit_trim_percentile),
    )
    old_cwd = Path.cwd()
    try:
        import os

        os.chdir(STREAM3D_ROOT)
        cache = provider._load_scene(scene)
    finally:
        import os

        os.chdir(old_cwd)

    available = _existing_mask_frames(scene, backbone)
    ap_frames = sorted(
        {
            int(frame)
            for window in cache["windows"]
            for frame in window.frame_ids
            if int(frame) % int(frame_stride) == 0 and int(frame) in available
        }
    )
    point_parts: list[np.ndarray] = []
    uv_parts: list[np.ndarray] = []
    frame_parts: list[np.ndarray] = []
    raw_slot_count = 0
    kept_slot_count = 0
    selected_window_count = 0
    for frame_id in ap_frames:
        for window, local_idx in provider._windows_for_frame(cache["windows"], int(frame_id)):
            selected_window_count += 1
            xyz = _apply_fit(window.xyz[local_idx], window.transform)
            uv = window.uv[local_idx]
            ok = (
                window.valid[local_idx]
                & np.isfinite(xyz).all(axis=1)
                & np.isfinite(uv).all(axis=1)
                & (uv[:, 0] >= 0.0)
                & (uv[:, 0] <= 1.0)
                & (uv[:, 1] >= 0.0)
                & (uv[:, 1] <= 1.0)
                & (window.visibility[local_idx] >= float(min_visibility))
                & (window.confidence[local_idx] >= float(min_confidence))
            )
            raw_slot_count += int(ok.shape[0])
            kept_slot_count += int(np.count_nonzero(ok))
            if not np.any(ok):
                continue
            pts = np.asarray(xyz[ok], dtype=np.float32)
            point_parts.append(pts)
            uv_parts.append(np.asarray(uv[ok], dtype=np.float32))
            frame_parts.append(np.full((pts.shape[0],), int(frame_id), dtype=np.int64))

    if point_parts:
        points = np.concatenate(point_parts, axis=0)
        uv = np.concatenate(uv_parts, axis=0)
        frame_ids = np.concatenate(frame_parts, axis=0)
    else:
        points = np.zeros((0, 3), dtype=np.float32)
        uv = np.zeros((0, 2), dtype=np.float32)
        frame_ids = np.zeros((0,), dtype=np.int64)

    idx = _sample_indices(points.shape[0], int(max_points), seed=f"{scene}:{mode}:soma_object_3d")
    points = points[idx]
    uv = uv[idx]
    frame_ids = frame_ids[idx]
    colors = _sample_rgb_colors(scene, frame_ids, uv) if points.shape[0] else np.zeros((0, 3), dtype=np.uint8)
    diag = {
        "scene": scene,
        "debug_root": str(debug_root),
        "mode": mode,
        "frame_stride": int(frame_stride),
        "existing_mask_frame_count": int(len(available)),
        "ap_frame_count": int(len(ap_frames)),
        "ap_frame_min": int(ap_frames[0]) if ap_frames else None,
        "ap_frame_max": int(ap_frames[-1]) if ap_frames else None,
        "window_count": int(len(cache.get("windows", []))),
        "selected_window_count": int(selected_window_count),
        "raw_slot_count": int(raw_slot_count),
        "kept_slot_count": int(kept_slot_count),
        "returned_point_count": int(points.shape[0]),
        "sampled": bool(idx.shape[0] < kept_slot_count),
        "max_points": int(max_points),
        "min_visibility": float(min_visibility),
        "min_confidence": float(min_confidence),
        "scene_fit": cache.get("scene_fit", {}),
        "stitch_diag": cache.get("stitch_diag", {}),
        "anchor_diag": cache.get("anchor_diag", {}),
    }
    return points, colors, frame_ids, uv, diag


def _read_support_map(path: Path, scene: str) -> tuple[dict[tuple[int, int], int], dict[str, int], dict[str, Any]]:
    object_to_idx: dict[str, int] = {}
    mask_to_object_idx: dict[tuple[int, int], int] = {}
    support_rows = 0
    duplicate_pairs = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("scene_id") != scene:
                continue
            if row.get("support_kind") != "object_declared_mask_observation_support":
                continue
            frame_text = str(row.get("frame_id") or "").strip()
            mask_text = str(row.get("observed_mask_id") or "").strip()
            object_id = str(row.get("object_id") or "").strip()
            if not frame_text or not mask_text or not object_id:
                continue
            if object_id not in object_to_idx:
                object_to_idx[object_id] = len(object_to_idx) + 1
            key = (int(frame_text), int(mask_text))
            if key in mask_to_object_idx and mask_to_object_idx[key] != object_to_idx[object_id]:
                duplicate_pairs += 1
            mask_to_object_idx[key] = object_to_idx[object_id]
            support_rows += 1
    diag = {
        "soma_support_row_count": int(support_rows),
        "soma_object_count": int(len(object_to_idx)),
        "soma_frame_mask_pair_count": int(len(mask_to_object_idx)),
        "soma_duplicate_frame_mask_pair_conflicts": int(duplicate_pairs),
    }
    return mask_to_object_idx, object_to_idx, diag


def _color_d4rt_by_soma_support(
    *,
    scene: str,
    backbone: str,
    points: np.ndarray,
    frame_ids: np.ndarray,
    uv: np.ndarray,
    support_rows: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    mask_to_object_idx, object_to_idx, diag = _read_support_map(support_rows, scene)
    stream = ScanNetStream(seq_name=scene, backbone=backbone, root=STREAM3D_ROOT / "data" / "scannet" / "processed")
    colors = np.full((points.shape[0], 3), 170, dtype=np.uint8)
    owner = np.zeros((points.shape[0],), dtype=np.int32)
    evaluated = 0
    for frame_id in sorted(set(np.asarray(frame_ids, dtype=np.int64).tolist())):
        sel = np.asarray(frame_ids) == int(frame_id)
        if not np.any(sel):
            continue
        try:
            mask = stream.load_mask(int(frame_id))
        except FileNotFoundError:
            continue
        h, w = mask.shape[:2]
        xy = np.rint(
            np.stack(
                [
                    uv[sel, 0] * float(max(w - 1, 1)),
                    uv[sel, 1] * float(max(h - 1, 1)),
                ],
                axis=1,
            )
        ).astype(np.int64)
        xy[:, 0] = np.clip(xy[:, 0], 0, max(w - 1, 0))
        xy[:, 1] = np.clip(xy[:, 1], 0, max(h - 1, 0))
        mask_ids = mask[xy[:, 1], xy[:, 0]]
        global_idx = np.flatnonzero(sel)
        evaluated += int(global_idx.shape[0])
        for local_i, mask_id in enumerate(mask_ids.tolist()):
            object_idx = mask_to_object_idx.get((int(frame_id), int(mask_id)), 0)
            if object_idx > 0:
                owner[int(global_idx[local_i])] = int(object_idx)
    assigned = owner > 0
    if np.any(assigned):
        colors[assigned] = _id_colors(owner[assigned])
    diag.update(
        {
            "soma_sem_contract": "D4RT points colored by SOMA object-bank 2D frame/mask support; no AP pred_masks used.",
            "soma_sem_point_count": int(points.shape[0]),
            "soma_sem_evaluated_point_count": int(evaluated),
            "soma_sem_assigned_point_count": int(np.count_nonzero(assigned)),
            "soma_sem_unassigned_point_count": int(points.shape[0] - np.count_nonzero(assigned)),
            "soma_sem_assigned_ratio": float(np.count_nonzero(assigned) / max(points.shape[0], 1)),
            "soma_sem_assigned_object_count": int(np.unique(owner[assigned]).shape[0]) if np.any(assigned) else 0,
            "soma_sem_unassigned_color": [170, 170, 170],
        }
    )
    return colors, diag


def export_layers(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    scene_points, scene_colors, mesh_path = _load_scene_mesh(args.scene)
    gt_labels = _load_gt(args.scene)
    if gt_labels.shape[0] != scene_points.shape[0]:
        raise ValueError(f"GT/mesh length mismatch: gt={gt_labels.shape[0]} mesh={scene_points.shape[0]}")
    gt_positive = gt_labels > 0
    d4rt_points, d4rt_colors, frame_ids, uv, d4rt_diag = _load_d4rt_rgb_uv(
        scene=args.scene,
        debug_root=Path(args.debug_root),
        mode=args.d4rt_mode,
        frame_stride=int(args.frame_stride),
        backbone=args.backbone,
        min_visibility=float(args.min_visibility),
        min_confidence=float(args.min_confidence),
        max_anchors=int(args.max_anchors),
        stitch_uv_radius=float(args.stitch_uv_radius),
        stitch_max_matches_per_frame=int(args.stitch_max_matches_per_frame),
        stitch_fit_trim_percentile=float(args.stitch_fit_trim_percentile),
        max_points=int(args.max_d4rt_points),
    )
    soma_colors, soma_diag = _color_d4rt_by_soma_support(
        scene=args.scene,
        backbone=args.backbone,
        points=d4rt_points,
        frame_ids=frame_ids,
        uv=uv,
        support_rows=Path(args.object_support_rows),
    )
    layer_npz = output_root / f"{args.scene}_soma_object_four_layers.npz"
    np.savez_compressed(
        layer_npz,
        gt_geo_points=scene_points.astype(np.float32),
        gt_geo_colors=scene_colors.astype(np.uint8),
        gt_sem_points=scene_points[gt_positive].astype(np.float32),
        gt_sem_colors=_id_colors(gt_labels[gt_positive]).astype(np.uint8),
        d4rt_geo_points=d4rt_points.astype(np.float32),
        d4rt_geo_colors=d4rt_colors.astype(np.uint8),
        soma_sem_points=d4rt_points.astype(np.float32),
        soma_sem_colors=soma_colors.astype(np.uint8),
    )
    status = {
        "phase": "v65_soma_object_3d_viewer_export",
        "scene": args.scene,
        "layers_npz": str(layer_npz),
        "layers_npz_sha256": sha256_file(layer_npz.resolve()),
        "mesh_path": rel(mesh_path),
        "mesh_path_sha256": sha256_file(mesh_path),
        "gt_path": rel(STREAM3D_ROOT / "data" / "scannet" / "gt" / f"{args.scene}.txt"),
        "gt_path_sha256": sha256_file(STREAM3D_ROOT / "data" / "scannet" / "gt" / f"{args.scene}.txt"),
        "object_support_rows": str(args.object_support_rows),
        "object_support_rows_sha256": sha256_file(Path(args.object_support_rows).resolve()),
        "gt_geo_point_count": int(scene_points.shape[0]),
        "gt_sem_point_count": int(np.count_nonzero(gt_positive)),
        "gt_sem_instance_count": int(np.unique(gt_labels[gt_labels > 0]).shape[0]),
        "d4rt_geo_point_count": int(d4rt_points.shape[0]),
        "soma_sem_point_count": int(d4rt_points.shape[0]),
        "required_layers": REQUIRED_LAYERS,
        "layer_controls_required": True,
        "d4rt_diag": d4rt_diag,
        **soma_diag,
    }
    status_path = output_root / "export_status.json"
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2, sort_keys=True), flush=True)
    return status


def load_existing_layers(args: argparse.Namespace) -> dict[str, Any]:
    layer_npz = Path(args.layers_npz).resolve()
    if not layer_npz.exists():
        raise FileNotFoundError(layer_npz)
    status: dict[str, Any] = {}
    if args.status_json and Path(args.status_json).exists():
        status = json.loads(Path(args.status_json).read_text(encoding="utf-8"))
    with np.load(layer_npz) as payload:
        required = [
            "gt_geo_points",
            "gt_geo_colors",
            "gt_sem_points",
            "gt_sem_colors",
            "d4rt_geo_points",
            "d4rt_geo_colors",
            "soma_sem_points",
            "soma_sem_colors",
        ]
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise ValueError(f"missing layer arrays in {layer_npz}: {missing}")
    status.update(
        {
            "phase": "v65_soma_object_3d_viewer_existing_npz",
            "layers_npz": str(layer_npz),
            "layers_npz_sha256": sha256_file(layer_npz),
            "required_layers": REQUIRED_LAYERS,
            "layer_controls_required": True,
        }
    )
    return status


def serve(args: argparse.Namespace) -> dict[str, Any]:
    import viser  # type: ignore

    status = load_existing_layers(args) if args.layers_npz else export_layers(args)
    with np.load(status["layers_npz"]) as payload:
        layers = {key: np.asarray(payload[key]) for key in payload.files}
    server = viser.ViserServer(host=args.host, port=int(args.port), verbose=True)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/v65_soma_object/grid", width=8.0, height=8.0, plane="xy", cell_size=0.5, section_size=2.0)
    handles: dict[str, Any] = {
        "GT geo": server.scene.add_point_cloud(
            "/v65_soma_object/GT geo",
            points=layers["gt_geo_points"],
            colors=layers["gt_geo_colors"],
            point_size=float(args.gt_point_size),
            point_shape="circle",
            visible=True,
            precision="float32",
        ),
        "GT sem": server.scene.add_point_cloud(
            "/v65_soma_object/GT sem",
            points=layers["gt_sem_points"],
            colors=layers["gt_sem_colors"],
            point_size=float(args.gt_sem_point_size),
            point_shape="circle",
            visible=True,
            precision="float32",
        ),
        "D4RT geo": server.scene.add_point_cloud(
            "/v65_soma_object/D4RT geo",
            points=layers["d4rt_geo_points"],
            colors=layers["d4rt_geo_colors"],
            point_size=float(args.d4rt_point_size),
            point_shape="circle",
            visible=True,
            precision="float32",
        ),
        "SOMA sem": server.scene.add_point_cloud(
            "/v65_soma_object/SOMA sem",
            points=layers["soma_sem_points"],
            colors=layers["soma_sem_colors"],
            point_size=float(args.soma_point_size),
            point_shape="circle",
            visible=True,
            precision="float32",
        ),
    }
    controls: dict[str, Any] = {}
    with server.gui.add_folder("v65 SOMA object layers"):
        for name in REQUIRED_LAYERS:
            controls[name] = server.gui.add_checkbox(name, initial_value=True)
            controls[name].on_update(lambda _event, name=name: setattr(handles[name], "visible", bool(controls[name].value)))
    live_status = dict(status)
    live_status.update(
        {
            "phase": "v65_soma_object_3d_live_viewer",
            "host": args.host,
            "port": int(args.port),
            "url": f"http://localhost:{int(args.port)}",
            "toggles": list(controls.keys()),
            "all_four_toggles_present": sorted(controls.keys()) == sorted(REQUIRED_LAYERS),
        }
    )
    live_status_path = Path(args.output_root) / "live_viewer_status.json"
    live_status_path.write_text(json.dumps(live_status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(live_status, indent=2, sort_keys=True), flush=True)

    stop = {"value": False}

    def _stop(_signum: int, _frame: Any) -> None:
        stop["value"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    while not stop["value"]:
        time.sleep(1.0)
    server.stop()
    return live_status


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve one-scene GT/D4RT/SOMA-object Viser viewer.")
    parser.add_argument("--scene", default="scene0050_00")
    parser.add_argument("--object-support-rows", default="Stream3D/outputs/audit/v65_soma_object_bank/soma_object_support_rows.csv")
    parser.add_argument("--debug-root", required=True)
    parser.add_argument("--d4rt-mode", default="self_stitched_eval_sim3")
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--min-visibility", type=float, default=0.0)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--max-anchors", type=int, default=120000)
    parser.add_argument("--stitch-uv-radius", type=float, default=0.002)
    parser.add_argument("--stitch-max-matches-per-frame", type=int, default=4096)
    parser.add_argument("--stitch-fit-trim-percentile", type=float, default=90.0)
    parser.add_argument("--max-d4rt-points", type=int, default=1000000)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--layers-npz", default=None)
    parser.add_argument("--status-json", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--gt-point-size", type=float, default=0.006)
    parser.add_argument("--gt-sem-point-size", type=float, default=0.012)
    parser.add_argument("--d4rt-point-size", type=float, default=0.012)
    parser.add_argument("--soma-point-size", type=float, default=0.02)
    args = parser.parse_args()
    if args.export_only:
        export_layers(args)
    else:
        serve(args)


if __name__ == "__main__":
    main()
