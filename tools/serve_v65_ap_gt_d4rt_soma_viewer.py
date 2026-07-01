#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
STREAM3D_ROOT = REPO_ROOT / "Stream3D"
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from geometry_provider.d4rt_carrier_provider import D4RTCarrierProjectionProvider, _apply_fit
from stream4d.scannet_stream import ScanNetStream
from stream4d_native.v65_common import rel, sha256_file
from stream4d_native.v65_visualization_export import (
    _id_colors,
    _load_gt,
    _load_prediction_overlay,
    _load_scene_mesh,
    _prediction_owner_ids,
)


PRED_SEM_LAYER_NAME = "AP pred sem on D4RT (diagnostic)"
REQUIRED_LAYERS = ["GT geometry", "GT sem", "D4RT stride-5 RGB geometry", PRED_SEM_LAYER_NAME]
DIAGNOSTIC_SEM_CONTRACT = (
    "diagnostic AP pred_masks assigned to D4RT points by nearest ScanNet mesh vertex; "
    "not native method-safe SOMA sem"
)
NATIVE_SOMA_SEM_BLOCKER = (
    "A9 method-safe materializer has no ScanNet AP mask join key; displayed semantic layer is "
    "diagnostic AP pred mask back-projection."
)


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


def _color_d4rt_by_final_soma_instances(
    *,
    scene: str,
    pred_config: str,
    scene_points: np.ndarray,
    d4rt_points: np.ndarray,
    semantic_nn_radius: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    pred_path = STREAM3D_ROOT / "data" / "prediction" / f"{pred_config}_class_agnostic" / f"{scene}.npz"
    colors = np.full((d4rt_points.shape[0], 3), 170, dtype=np.uint8)
    if d4rt_points.shape[0] == 0:
        return colors, {
            "soma_sem_contract": DIAGNOSTIC_SEM_CONTRACT,
            "soma_sem_assignment_error": "",
            "soma_sem_point_count": 0,
            "soma_sem_assigned_point_count": 0,
            "soma_sem_unassigned_point_count": 0,
            "soma_sem_instance_count": 0,
            "soma_sem_assignment_radius": float(semantic_nn_radius),
            "soma_sem_unassigned_color": [170, 170, 170],
        }
    if not pred_path.exists():
        return colors, {
            "soma_sem_contract": DIAGNOSTIC_SEM_CONTRACT,
            "soma_sem_assignment_error": f"missing prediction npz: {rel(pred_path)}",
            "soma_sem_point_count": int(d4rt_points.shape[0]),
            "soma_sem_assigned_point_count": 0,
            "soma_sem_unassigned_point_count": int(d4rt_points.shape[0]),
            "soma_sem_instance_count": 0,
            "soma_sem_assignment_radius": float(semantic_nn_radius),
            "soma_sem_unassigned_color": [170, 170, 170],
        }
    with np.load(pred_path) as payload:
        masks = np.asarray(payload["pred_masks"], dtype=bool)
        if "pred_score" in payload.files:
            scores = np.asarray(payload["pred_score"], dtype=np.float32)
        elif "pred_scores" in payload.files:
            scores = np.asarray(payload["pred_scores"], dtype=np.float32)
        else:
            scores = masks.sum(axis=0).astype(np.float32) if masks.ndim == 2 else np.asarray([], dtype=np.float32)
    if masks.ndim != 2 or masks.shape[0] != scene_points.shape[0]:
        return colors, {
            "soma_sem_contract": DIAGNOSTIC_SEM_CONTRACT,
            "soma_sem_assignment_error": f"unsupported pred_masks shape={tuple(masks.shape)} vertex_count={scene_points.shape[0]}",
            "soma_sem_point_count": int(d4rt_points.shape[0]),
            "soma_sem_assigned_point_count": 0,
            "soma_sem_unassigned_point_count": int(d4rt_points.shape[0]),
            "soma_sem_instance_count": int(masks.shape[1]) if masks.ndim == 2 else 0,
            "soma_sem_assignment_radius": float(semantic_nn_radius),
            "soma_sem_unassigned_color": [170, 170, 170],
        }
    owner = _prediction_owner_ids(masks, scores)
    dist, nn_idx = cKDTree(np.asarray(scene_points, dtype=np.float32)).query(np.asarray(d4rt_points, dtype=np.float32), k=1)
    assigned = np.isfinite(dist) & (dist <= float(semantic_nn_radius)) & (owner[nn_idx] >= 0)
    if np.any(assigned):
        colors[assigned] = _id_colors(owner[nn_idx[assigned]] + 1)
    assigned_instances = np.unique(owner[nn_idx[assigned]]) if np.any(assigned) else np.zeros((0,), dtype=np.int64)
    return colors, {
        "soma_sem_contract": DIAGNOSTIC_SEM_CONTRACT,
        "soma_sem_assignment_error": "",
        "soma_sem_point_count": int(d4rt_points.shape[0]),
        "soma_sem_assigned_point_count": int(np.count_nonzero(assigned)),
        "soma_sem_unassigned_point_count": int(d4rt_points.shape[0] - np.count_nonzero(assigned)),
        "soma_sem_assigned_ratio": float(np.count_nonzero(assigned) / max(d4rt_points.shape[0], 1)),
        "soma_sem_instance_count": int(assigned_instances.shape[0]),
        "soma_sem_assignment_radius": float(semantic_nn_radius),
        "soma_sem_nn_distance_median": float(np.median(dist)) if dist.size else None,
        "soma_sem_nn_distance_p90": float(np.percentile(dist, 90)) if dist.size else None,
        "soma_sem_unassigned_color": [170, 170, 170],
    }


def _load_ap_provider_d4rt_rgb(
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
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
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
        # Provider internals use ScanNetStream defaults relative to Stream3D.
        # Keep this identical to AP provider replacement execution.
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

    idx = _sample_indices(points.shape[0], int(max_points), seed=f"{scene}:{mode}:ap_provider_rgb")
    points = points[idx]
    uv = uv[idx]
    frame_ids = frame_ids[idx]
    colors = _sample_rgb_colors(scene, frame_ids, uv) if points.shape[0] else np.zeros((0, 3), dtype=np.uint8)
    diag = {
        "scene": scene,
        "debug_root": str(debug_root),
        "mode": mode,
        "frame_stride": int(frame_stride),
        "filter_existing_segmentation": True,
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
        "color_source": "ScanNet RGB sampled at AP-provider D4RT predicted uv/frame_id",
        "scene_fit": cache.get("scene_fit", {}),
        "anchor_diag": cache.get("anchor_diag", {}),
        "stitch_diag": cache.get("stitch_diag", {}),
    }
    return points, colors, diag


def export_layers(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    scene_points, scene_colors, mesh_path = _load_scene_mesh(args.scene)
    gt_labels = _load_gt(args.scene)
    if gt_labels.shape[0] != scene_points.shape[0]:
        raise ValueError(f"GT/mesh length mismatch: gt={gt_labels.shape[0]} mesh={scene_points.shape[0]}")
    gt_positive = gt_labels > 0
    pred_overlay = _load_prediction_overlay(args.scene, args.pred_config, scene_points.shape[0])
    d4rt_points, d4rt_colors, d4rt_diag = _load_ap_provider_d4rt_rgb(
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
    soma_colors, soma_diag = _color_d4rt_by_final_soma_instances(
        scene=args.scene,
        pred_config=args.pred_config,
        scene_points=scene_points,
        d4rt_points=d4rt_points,
        semantic_nn_radius=float(args.semantic_nn_radius),
    )
    final_ap_pred_count = int(np.asarray(pred_overlay["_points"], dtype=np.float32).shape[0])
    layer_npz = output_root / f"{args.scene}_four_layers.npz"
    np.savez_compressed(
        layer_npz,
        gt_geometry_points=scene_points.astype(np.float32),
        gt_geometry_colors=scene_colors.astype(np.uint8),
        gt_sem_points=scene_points[gt_positive].astype(np.float32),
        gt_sem_colors=_id_colors(gt_labels[gt_positive]).astype(np.uint8),
        d4rt_points=d4rt_points.astype(np.float32),
        d4rt_colors=d4rt_colors.astype(np.uint8),
        soma_sem_points=d4rt_points.astype(np.float32),
        soma_sem_colors=soma_colors.astype(np.uint8),
    )
    status = {
        "phase": "v65_conf02_ap_four_layer_viewer_export",
        "scene": args.scene,
        "layers_npz": str(layer_npz),
        "layers_npz_sha256": sha256_file(layer_npz.resolve()),
        "mesh_path": rel(mesh_path),
        "mesh_path_sha256": sha256_file(mesh_path),
        "gt_path": rel(STREAM3D_ROOT / "data" / "scannet" / "gt" / f"{args.scene}.txt"),
        "gt_path_sha256": sha256_file(STREAM3D_ROOT / "data" / "scannet" / "gt" / f"{args.scene}.txt"),
        "pred_config": args.pred_config,
        "pred_path": pred_overlay.get("pred_path", ""),
        "pred_path_sha256": pred_overlay.get("pred_path_sha256", ""),
        "pre_points_path": pred_overlay.get("pre_points_path", ""),
        "pre_points_path_sha256": pred_overlay.get("pre_points_path_sha256", ""),
        "pred_mask_contract": pred_overlay.get("mask_contract", ""),
        "pred_error": pred_overlay.get("error", ""),
        "gt_geometry_point_count": int(scene_points.shape[0]),
        "gt_sem_point_count": int(np.count_nonzero(gt_positive)),
        "gt_sem_instance_count": int(np.unique(gt_labels[gt_labels > 0]).shape[0]),
        "d4rt_point_count": int(d4rt_points.shape[0]),
        "soma_sem_point_count": int(d4rt_points.shape[0]),
        **soma_diag,
        "final_ap_pred_mask_point_count": final_ap_pred_count,
        "final_ap_pred_instance_count": int(pred_overlay.get("pred_instance_count", 0)),
        "final_ap_pred_mask_contract": pred_overlay.get("mask_contract", ""),
        "d4rt_diag": d4rt_diag,
        "required_layers": REQUIRED_LAYERS,
        "layer_controls_required": True,
        "native_soma_sem_available": False,
        "native_soma_sem_blocker": NATIVE_SOMA_SEM_BLOCKER,
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
    if args.status_json:
        status_path = Path(args.status_json)
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
    with np.load(layer_npz) as payload:
        required = [
            "gt_geometry_points",
            "gt_geometry_colors",
            "gt_sem_points",
            "gt_sem_colors",
            "d4rt_points",
            "d4rt_colors",
            "soma_sem_points",
            "soma_sem_colors",
        ]
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise ValueError(f"missing layer arrays in {layer_npz}: {missing}")
        counts = {
            "gt_geometry_point_count": int(np.asarray(payload["gt_geometry_points"]).shape[0]),
            "gt_sem_point_count": int(np.asarray(payload["gt_sem_points"]).shape[0]),
            "d4rt_point_count": int(np.asarray(payload["d4rt_points"]).shape[0]),
            "soma_sem_point_count": int(np.asarray(payload["soma_sem_points"]).shape[0]),
        }
    status.update(
        {
            "phase": "v65_conf02_ap_four_layer_existing_npz",
            "scene": status.get("scene", args.scene),
            "layers_npz": str(layer_npz),
            "layers_npz_sha256": sha256_file(layer_npz),
            "required_layers": REQUIRED_LAYERS,
            "layer_controls_required": True,
            "existing_layers_mode": True,
            "native_soma_sem_available": False,
            "native_soma_sem_blocker": NATIVE_SOMA_SEM_BLOCKER,
            "soma_sem_contract": DIAGNOSTIC_SEM_CONTRACT,
            **counts,
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
    server.scene.add_grid("/v65_conf02_ap/grid", width=8.0, height=8.0, plane="xy", cell_size=0.5, section_size=2.0)
    handles: dict[str, Any] = {}
    handles["GT geometry"] = server.scene.add_point_cloud(
        "/v65_conf02_ap/GT geometry",
        points=layers["gt_geometry_points"],
        colors=layers["gt_geometry_colors"],
        point_size=float(args.gt_point_size),
        point_shape="circle",
        visible=True,
        precision="float32",
    )
    handles["GT sem"] = server.scene.add_point_cloud(
        "/v65_conf02_ap/GT sem",
        points=layers["gt_sem_points"],
        colors=layers["gt_sem_colors"],
        point_size=float(args.gt_sem_point_size),
        point_shape="circle",
        visible=True,
        precision="float32",
    )
    handles["D4RT stride-5 RGB geometry"] = server.scene.add_point_cloud(
        "/v65_conf02_ap/D4RT stride-5 RGB geometry",
        points=layers["d4rt_points"],
        colors=layers["d4rt_colors"],
        point_size=float(args.d4rt_point_size),
        point_shape="circle",
        visible=True,
        precision="float32",
    )
    handles[PRED_SEM_LAYER_NAME] = server.scene.add_point_cloud(
        f"/v65_conf02_ap/{PRED_SEM_LAYER_NAME}",
        points=layers["soma_sem_points"],
        colors=layers["soma_sem_colors"],
        point_size=float(args.soma_point_size),
        point_shape="circle",
        visible=True,
        precision="float32",
    )

    controls: dict[str, Any] = {}
    with server.gui.add_folder("v65 AP four layers"):
        for name in REQUIRED_LAYERS:
            controls[name] = server.gui.add_checkbox(name, initial_value=True)
            controls[name].on_update(lambda _event, name=name: setattr(handles[name], "visible", bool(controls[name].value)))

    live_status = dict(status)
    live_status.update(
        {
            "phase": "v65_conf02_ap_four_layer_live_viewer",
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
    parser = argparse.ArgumentParser(description="Serve one-scene v65 AP GT/D4RT/SOMA four-layer Viser viewer.")
    parser.add_argument("--scene", default="scene0050_00")
    parser.add_argument("--pred-config", required=True)
    parser.add_argument("--debug-root", required=True)
    parser.add_argument("--d4rt-mode", default="self_stitched_eval_sim3")
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--min-visibility", type=float, default=0.0)
    parser.add_argument("--min-confidence", type=float, default=0.2)
    parser.add_argument("--max-anchors", type=int, default=120000)
    parser.add_argument("--stitch-uv-radius", type=float, default=0.002)
    parser.add_argument("--stitch-max-matches-per-frame", type=int, default=4096)
    parser.add_argument("--stitch-fit-trim-percentile", type=float, default=90.0)
    parser.add_argument("--max-d4rt-points", type=int, default=1000000)
    parser.add_argument("--semantic-nn-radius", type=float, default=0.05)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--layers-npz", default=None)
    parser.add_argument("--status-json", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8093)
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
