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


REPO_ROOT = Path(__file__).resolve().parents[1]
STREAM3D_ROOT = REPO_ROOT / "Stream3D"
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from stream4d.scannet_stream import ScanNetStream
from stream4d_native.v65_d4rt_geometry import D4RT_COORDINATE_MODES, load_d4rt_geometry_frames
from stream4d_native.v65_common import rel, sha256_file
from stream4d_native.v65_visualization_export import _id_colors, _load_gt, _load_scene_mesh, _prediction_owner_ids


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


def _project_stream3d(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    if path_obj.parts and path_obj.parts[0] == STREAM3D_ROOT.name:
        return REPO_ROOT / path_obj
    return STREAM3D_ROOT / path_obj


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def _load_ap_d4rt_points(
    *,
    scene: str,
    pipeline_root: Path,
    confidence_threshold: float,
    visibility_threshold: float,
    max_points: int,
    coordinate_mode: str,
    d4rt_stride_summary: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    geometry_frames, geometry_diag = load_d4rt_geometry_frames(
        pipeline_root=pipeline_root,
        scene=scene,
        stream3d_root=STREAM3D_ROOT,
        coordinate_mode=coordinate_mode,
        d4rt_stride_summary=d4rt_stride_summary or None,
    )
    point_parts: list[np.ndarray] = []
    uv_parts: list[np.ndarray] = []
    frame_parts: list[np.ndarray] = []
    raw_slot_count = 0
    kept_slot_count = 0
    frames: set[int] = set()
    carrier_count = 0
    for geometry_payload in geometry_frames:
        carrier_count += 1
        frame_ids = [int(value) for value in geometry_payload["frame_ids"]]
        xyz = np.asarray(geometry_payload["xyz"], dtype=np.float32)
        uv = np.asarray(geometry_payload["uv"], dtype=np.float32)
        valid = np.asarray(geometry_payload["valid"], dtype=bool)
        confidence = np.asarray(geometry_payload["confidence"], dtype=np.float32)
        visibility = np.asarray(geometry_payload["visibility"], dtype=np.float32)
        if len(frame_ids) != xyz.shape[0]:
            raise ValueError("frame manifest length mismatch in D4RT geometry payload")
        for local_idx, frame_id in enumerate(frame_ids):
            ok = (
                valid[local_idx]
                & np.isfinite(xyz[local_idx]).all(axis=1)
                & np.isfinite(uv[local_idx]).all(axis=1)
                & (uv[local_idx, :, 0] >= 0.0)
                & (uv[local_idx, :, 0] <= 1.0)
                & (uv[local_idx, :, 1] >= 0.0)
                & (uv[local_idx, :, 1] <= 1.0)
                & (confidence[local_idx] >= float(confidence_threshold))
                & (visibility[local_idx] >= float(visibility_threshold))
            )
            raw_slot_count += int(ok.shape[0])
            kept_slot_count += int(np.count_nonzero(ok))
            frames.add(int(frame_id))
            if not np.any(ok):
                continue
            point_parts.append(np.asarray(xyz[local_idx, ok], dtype=np.float32))
            uv_parts.append(np.asarray(uv[local_idx, ok], dtype=np.float32))
            frame_parts.append(np.full((np.count_nonzero(ok),), int(frame_id), dtype=np.int64))
    points = np.concatenate(point_parts, axis=0) if point_parts else np.zeros((0, 3), dtype=np.float32)
    uv = np.concatenate(uv_parts, axis=0) if uv_parts else np.zeros((0, 2), dtype=np.float32)
    frame_ids = np.concatenate(frame_parts, axis=0) if frame_parts else np.zeros((0,), dtype=np.int64)
    idx = _sample_indices(points.shape[0], int(max_points), seed=f"{scene}:v65_pipeline_ap_four_layer:d4rt")
    points = points[idx]
    uv = uv[idx]
    frame_ids = frame_ids[idx]
    colors = _sample_rgb_colors(scene, frame_ids, uv) if points.shape[0] else np.zeros((0, 3), dtype=np.uint8)
    return points, colors, {
        "d4rt_contract": f"exact D4RT geometry input used by run_v65_pipeline_d4rt_nn_ap.py before mesh-NN materialization: {coordinate_mode} filtered by valid/uv/confidence/visibility",
        "pipeline_root": rel(pipeline_root),
        "carrier_cache_window_count": int(carrier_count),
        **geometry_diag,
        "unique_frame_count": int(len(frames)),
        "frame_min": int(min(frames)) if frames else None,
        "frame_max": int(max(frames)) if frames else None,
        "raw_slot_count": int(raw_slot_count),
        "kept_slot_count_before_sampling": int(kept_slot_count),
        "returned_point_count": int(points.shape[0]),
        "sampled": bool(points.shape[0] < kept_slot_count),
        "confidence_threshold": float(confidence_threshold),
        "visibility_threshold": float(visibility_threshold),
    }


def _load_ap_pred_sem(scene: str, pred_config: str, scene_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pred_path = STREAM3D_ROOT / "data" / "prediction" / f"{pred_config}_class_agnostic" / f"{scene}.npz"
    pre_points_path = STREAM3D_ROOT / "data" / "TMP" / pred_config / f"{scene}_pre_points.npy"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    if not pre_points_path.exists():
        raise FileNotFoundError(pre_points_path)
    with np.load(pred_path) as payload:
        masks = np.asarray(payload["pred_masks"], dtype=bool)
        if "pred_score" in payload.files:
            scores = np.asarray(payload["pred_score"], dtype=np.float32)
        elif "pred_scores" in payload.files:
            scores = np.asarray(payload["pred_scores"], dtype=np.float32)
        else:
            scores = masks.sum(axis=0).astype(np.float32) if masks.ndim == 2 else np.zeros((0,), dtype=np.float32)
    pre_points = np.load(pre_points_path).astype(np.int64)
    if masks.ndim != 2:
        raise ValueError(f"pred_masks must be 2D, got {masks.shape}")
    if masks.shape[0] == scene_points.shape[0]:
        owner = _prediction_owner_ids(masks, scores)
        point_ids = np.flatnonzero(owner >= 0).astype(np.int64)
        owners = owner[point_ids]
        mask_contract = "full_scene_vertex_masks"
    elif masks.shape[0] == pre_points.shape[0]:
        local_owner = _prediction_owner_ids(masks, scores)
        keep = local_owner >= 0
        point_ids = pre_points[keep].astype(np.int64)
        owners = local_owner[keep]
        mask_contract = "pre_points_vertex_masks"
    else:
        raise ValueError(
            f"pred_masks first dim {masks.shape[0]} does not match mesh {scene_points.shape[0]} or pre_points {pre_points.shape[0]}"
        )
    points = scene_points[point_ids] if point_ids.shape[0] else np.zeros((0, 3), dtype=np.float32)
    colors = _id_colors(owners + 1) if point_ids.shape[0] else np.zeros((0, 3), dtype=np.uint8)
    return points.astype(np.float32), colors.astype(np.uint8), {
        "soma_sem_contract": "exact AP pred_masks/pre_points from the selected v65_pipeline_d4rt_nn output config; no 2D projection is used for this 3D layer",
        "pred_config": pred_config,
        "pred_path": rel(pred_path),
        "pred_path_sha256": sha256_file(pred_path),
        "pre_points_path": rel(pre_points_path),
        "pre_points_path_sha256": sha256_file(pre_points_path),
        "mask_contract": mask_contract,
        "pred_mask_shape": [int(v) for v in masks.shape],
        "pred_instance_count": int(masks.shape[1]),
        "pre_points_count": int(pre_points.shape[0]),
        "soma_sem_point_count": int(points.shape[0]),
        "soma_sem_instance_count": int(np.unique(owners).shape[0]) if owners.shape[0] else 0,
    }


def export_layers(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pipeline_root = _project_stream3d(args.pipeline_root)
    scene_points, scene_colors, mesh_path = _load_scene_mesh(args.scene)
    gt_labels = _load_gt(args.scene)
    if gt_labels.shape[0] != scene_points.shape[0]:
        raise ValueError(f"GT/mesh length mismatch: gt={gt_labels.shape[0]} mesh={scene_points.shape[0]}")
    gt_positive = gt_labels > 0
    d4rt_points, d4rt_colors, d4rt_diag = _load_ap_d4rt_points(
        scene=args.scene,
        pipeline_root=pipeline_root,
        confidence_threshold=float(args.confidence_threshold),
        visibility_threshold=float(args.visibility_threshold),
        max_points=int(args.max_d4rt_points),
        coordinate_mode=str(args.d4rt_coordinate_mode),
        d4rt_stride_summary=str(args.d4rt_stride_summary),
    )
    soma_points, soma_colors, soma_diag = _load_ap_pred_sem(args.scene, args.pred_config, scene_points)
    layer_npz = output_root / f"{args.scene}_{args.pred_config}_ap_four_layers.npz"
    np.savez_compressed(
        layer_npz,
        gt_geo_points=scene_points.astype(np.float32),
        gt_geo_colors=scene_colors.astype(np.uint8),
        gt_sem_points=scene_points[gt_positive].astype(np.float32),
        gt_sem_colors=_id_colors(gt_labels[gt_positive]).astype(np.uint8),
        d4rt_geo_points=d4rt_points.astype(np.float32),
        d4rt_geo_colors=d4rt_colors.astype(np.uint8),
        soma_sem_points=soma_points.astype(np.float32),
        soma_sem_colors=soma_colors.astype(np.uint8),
    )
    ap_summary_path = STREAM3D_ROOT / args.ap_summary if args.ap_summary else Path("")
    status = {
        "phase": "v65_pipeline_ap_four_layer_export",
        "scene": args.scene,
        "pred_config": args.pred_config,
        "pipeline_root": rel(pipeline_root),
        "pipeline_summary": rel(pipeline_root / "pipeline_summary.json"),
        "pipeline_summary_sha256": sha256_file(pipeline_root / "pipeline_summary.json"),
        "ap_summary": rel(ap_summary_path) if ap_summary_path.exists() else "",
        "ap_summary_sha256": sha256_file(ap_summary_path) if ap_summary_path.exists() else "",
        "layers_npz": str(layer_npz),
        "layers_npz_sha256": sha256_file(layer_npz.resolve()),
        "mesh_path": rel(mesh_path),
        "mesh_path_sha256": sha256_file(mesh_path),
        "gt_path": rel(STREAM3D_ROOT / "data" / "scannet" / "gt" / f"{args.scene}.txt"),
        "gt_path_sha256": sha256_file(STREAM3D_ROOT / "data" / "scannet" / "gt" / f"{args.scene}.txt"),
        "gt_geo_point_count": int(scene_points.shape[0]),
        "gt_sem_point_count": int(np.count_nonzero(gt_positive)),
        "gt_sem_instance_count": int(np.unique(gt_labels[gt_labels > 0]).shape[0]),
        "d4rt_geo_point_count": int(d4rt_points.shape[0]),
        **d4rt_diag,
        **soma_diag,
        "required_layers": REQUIRED_LAYERS,
        "layer_controls_required": True,
        "contract_warning": (
            "This viewer intentionally displays the exact AP adapter inputs/outputs. "
            f"D4RT geo is pipeline {args.d4rt_coordinate_mode} before mesh-NN; SOMA sem is exact AP pred_masks on evaluator mesh vertices."
        ),
    }
    status_path = output_root / "export_status.json"
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2, sort_keys=True), flush=True)
    return status


def _load_existing(args: argparse.Namespace) -> dict[str, Any]:
    status_path = Path(args.status_json)
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    layer_npz = Path(args.layers_npz)
    if not layer_npz.exists():
        raise FileNotFoundError(layer_npz)
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
            "phase": "v65_pipeline_ap_four_layer_existing_export",
            "layers_npz": str(layer_npz),
            "layers_npz_sha256": sha256_file(layer_npz.resolve()),
            "required_layers": REQUIRED_LAYERS,
            "layer_controls_required": True,
        }
    )
    return status


def serve(args: argparse.Namespace) -> dict[str, Any]:
    import viser  # type: ignore

    status = _load_existing(args) if args.layers_npz else export_layers(args)
    with np.load(status["layers_npz"]) as payload:
        layers = {key: np.asarray(payload[key]) for key in payload.files}
    server = viser.ViserServer(host=args.host, port=int(args.port), verbose=True)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/v65_pipeline_ap/grid", width=8.0, height=8.0, plane="xy", cell_size=0.5, section_size=2.0)
    handles: dict[str, Any] = {
        "GT geo": server.scene.add_point_cloud(
            "/v65_pipeline_ap/GT geo",
            points=layers["gt_geo_points"],
            colors=layers["gt_geo_colors"],
            point_size=float(args.gt_point_size),
            point_shape="circle",
            visible=True,
            precision="float32",
        ),
        "GT sem": server.scene.add_point_cloud(
            "/v65_pipeline_ap/GT sem",
            points=layers["gt_sem_points"],
            colors=layers["gt_sem_colors"],
            point_size=float(args.gt_sem_point_size),
            point_shape="circle",
            visible=True,
            precision="float32",
        ),
        "D4RT geo": server.scene.add_point_cloud(
            "/v65_pipeline_ap/D4RT geo",
            points=layers["d4rt_geo_points"],
            colors=layers["d4rt_geo_colors"],
            point_size=float(args.d4rt_point_size),
            point_shape="circle",
            visible=True,
            precision="float32",
        ),
        "SOMA sem": server.scene.add_point_cloud(
            "/v65_pipeline_ap/SOMA sem",
            points=layers["soma_sem_points"],
            colors=layers["soma_sem_colors"],
            point_size=float(args.soma_point_size),
            point_shape="circle",
            visible=True,
            precision="float32",
        ),
    }
    controls: dict[str, Any] = {}
    with server.gui.add_folder("v65 AP pipeline layers"):
        for name in REQUIRED_LAYERS:
            controls[name] = server.gui.add_checkbox(name, initial_value=True)
            controls[name].on_update(lambda _event, name=name: setattr(handles[name], "visible", bool(controls[name].value)))
    live_status = dict(status)
    live_status.update(
        {
            "phase": "v65_pipeline_ap_four_layer_live_viewer",
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
    parser = argparse.ArgumentParser(description="Serve exact v65 pipeline AP four-layer Viser viewer.")
    parser.add_argument("--scene", default="scene0050_00")
    parser.add_argument("--pipeline-root", required=True)
    parser.add_argument("--pred-config", required=True)
    parser.add_argument("--ap-summary", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.2)
    parser.add_argument("--visibility-threshold", type=float, default=0.0)
    parser.add_argument("--d4rt-coordinate-mode", choices=list(D4RT_COORDINATE_MODES), default="chunk_final_gt_sim3")
    parser.add_argument("--d4rt-stride-summary", default="")
    parser.add_argument("--max-d4rt-points", type=int, default=0)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--layers-npz", default="")
    parser.add_argument("--status-json", default="")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--gt-point-size", type=float, default=0.006)
    parser.add_argument("--gt-sem-point-size", type=float, default=0.012)
    parser.add_argument("--d4rt-point-size", type=float, default=0.012)
    parser.add_argument("--soma-point-size", type=float, default=0.025)
    args = parser.parse_args()
    if args.export_only:
        export_layers(args)
    else:
        serve(args)


if __name__ == "__main__":
    main()
