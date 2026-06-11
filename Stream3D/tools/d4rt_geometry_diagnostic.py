from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.scannet_stream import ScanNetStream
from stream4d_native.sim3 import fit_sim3_umeyama


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_seq_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _backproject_xy_world(stream: ScanNetStream, frame_id: int, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(xy, dtype=np.float32)
    if xy.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.zeros((0,), dtype=bool)
    depth = stream.load_depth(int(frame_id))
    pose = stream.load_pose(int(frame_id))
    intrinsics = stream.load_intrinsics()
    if not np.isfinite(pose).all():
        return np.empty((xy.shape[0], 3), dtype=np.float32), np.zeros((xy.shape[0],), dtype=bool)
    h, w = depth.shape
    x = np.rint(xy[:, 0]).astype(np.int64)
    y = np.rint(xy[:, 1]).astype(np.int64)
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    world = np.full((xy.shape[0], 3), np.nan, dtype=np.float32)
    if not np.any(valid):
        return world, valid
    z = depth[y[valid], x[valid]].astype(np.float32)
    depth_valid = np.isfinite(z) & (z > 0.0)
    valid_indices = np.flatnonzero(valid)
    valid[valid_indices[~depth_valid]] = False
    if not np.any(depth_valid):
        return world, valid
    x_f = x[valid_indices[depth_valid]].astype(np.float32)
    y_f = y[valid_indices[depth_valid]].astype(np.float32)
    z_f = z[depth_valid]
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    cam = np.stack([(x_f - cx) * z_f / fx, (y_f - cy) * z_f / fy, z_f, np.ones_like(z_f)], axis=1)
    pts = (pose @ cam.T).T[:, :3].astype(np.float32)
    finite = np.isfinite(pts).all(axis=1)
    keep_indices = valid_indices[depth_valid]
    valid[keep_indices[~finite]] = False
    world[keep_indices[finite]] = pts[finite]
    return world, valid


def _pick_anchors(
    stream: ScanNetStream,
    carrier_npz: Path,
    max_anchors: int,
    min_visibility: float,
    min_confidence: float,
) -> dict[str, Any]:
    with np.load(carrier_npz) as data:
        src_frame = np.asarray(data["src_frame"], dtype=np.int64)
        src_frame_global = np.asarray(data.get("src_frame_global", src_frame), dtype=np.int64)
        src_xy = np.asarray(data["src_xy"], dtype=np.float32)
        xyz_ref = np.asarray(data["xyz_ref"], dtype=np.float32)
        valid = np.asarray(data.get("valid", np.ones(xyz_ref.shape[:2], dtype=bool)), dtype=bool)
        visibility = np.asarray(data.get("visibility_prob", np.ones(xyz_ref.shape[:2], dtype=np.float32)), dtype=np.float32)
        confidence = np.asarray(data.get("confidence_prob", np.ones(xyz_ref.shape[:2], dtype=np.float32)), dtype=np.float32)
        uv_pred = np.asarray(data.get("uv_pred", np.empty((0, 0, 2), dtype=np.float32)), dtype=np.float32)

    n_carriers = int(src_frame.shape[0])
    carrier_index = np.arange(n_carriers, dtype=np.int64)
    local_ok = (src_frame >= 0) & (src_frame < xyz_ref.shape[0])
    if not np.any(local_ok):
        return {"anchors": 0, "reason": "no_valid_src_frame"}

    local_idx = src_frame[local_ok]
    carrier_idx = carrier_index[local_ok]
    d4rt = xyz_ref[local_idx, carrier_idx]
    d4rt_valid = np.isfinite(d4rt).all(axis=1)
    if valid.ndim == 2:
        d4rt_valid &= valid[local_idx, carrier_idx]
    if visibility.ndim == 2:
        vis = visibility[local_idx, carrier_idx]
        d4rt_valid &= vis >= float(min_visibility)
    else:
        vis = np.ones_like(local_idx, dtype=np.float32)
    if confidence.ndim == 2:
        conf = confidence[local_idx, carrier_idx]
        d4rt_valid &= conf >= float(min_confidence)
    else:
        conf = np.ones_like(local_idx, dtype=np.float32)

    selected_indices = np.flatnonzero(local_ok)[d4rt_valid]
    if selected_indices.size == 0:
        return {"anchors": 0, "reason": "no_finite_d4rt_anchors"}
    if selected_indices.size > max_anchors:
        pick = np.linspace(0, selected_indices.size - 1, int(max_anchors), dtype=np.int64)
        selected_indices = selected_indices[pick]

    rgbd_world = np.full((selected_indices.size, 3), np.nan, dtype=np.float32)
    rgbd_valid = np.zeros((selected_indices.size,), dtype=bool)
    for frame_id in sorted(set(int(v) for v in src_frame_global[selected_indices].tolist())):
        frame_mask = src_frame_global[selected_indices] == frame_id
        world, valid_mask = _backproject_xy_world(stream, frame_id, src_xy[selected_indices][frame_mask])
        rgbd_world[frame_mask] = world
        rgbd_valid[frame_mask] = valid_mask

    selected_local = src_frame[selected_indices]
    d4rt_points = xyz_ref[selected_local, selected_indices]
    ok = rgbd_valid & np.isfinite(d4rt_points).all(axis=1) & np.isfinite(rgbd_world).all(axis=1)
    selected_indices = selected_indices[ok]
    selected_local = selected_local[ok]
    d4rt_points = d4rt_points[ok]
    rgbd_world = rgbd_world[ok]
    if visibility.ndim == 2 and selected_indices.size:
        vis = visibility[selected_local, selected_indices]
    else:
        vis = np.ones((selected_indices.size,), dtype=np.float32)
    if confidence.ndim == 2 and selected_indices.size:
        conf = confidence[selected_local, selected_indices]
    else:
        conf = np.ones((selected_indices.size,), dtype=np.float32)

    uv_in01_rate = None
    if uv_pred.ndim == 3 and uv_pred.shape[-1] == 2 and uv_pred.size:
        uv_in01 = (uv_pred[..., 0] >= 0.0) & (uv_pred[..., 0] <= 1.0) & (uv_pred[..., 1] >= 0.0) & (uv_pred[..., 1] <= 1.0)
        uv_in01_rate = float(np.mean(uv_in01))

    return {
        "anchors": int(selected_indices.size),
        "d4rt_points": d4rt_points,
        "rgbd_world": rgbd_world,
        "visibility_mean": float(np.mean(vis)) if vis.size else None,
        "confidence_mean": float(np.mean(conf)) if conf.size else None,
        "uv_in01_rate": uv_in01_rate,
    }


def _window_report(
    stream: ScanNetStream,
    carrier_npz: Path,
    max_anchors: int,
    min_visibility: float,
    min_confidence: float,
) -> dict[str, Any]:
    started = time.time()
    anchors = _pick_anchors(stream, carrier_npz, max_anchors, min_visibility, min_confidence)
    row: dict[str, Any] = {
        "scene": stream.seq_name,
        "window": carrier_npz.stem.replace("carriers_window", ""),
        "carrier_file": str(carrier_npz),
        "sim3_anchor_count": int(anchors.get("anchors", 0)),
        "elapsed_sec": None,
    }
    if int(anchors.get("anchors", 0)) < 4:
        row.update(
            {
                "status": "failed",
                "failure_reason": anchors.get("reason", "too_few_anchors"),
                "elapsed_sec": float(time.time() - started),
            }
        )
        return row
    try:
        fit = fit_sim3_umeyama(anchors["d4rt_points"], anchors["rgbd_world"])
    except Exception as exc:
        row.update(
            {
                "status": "failed",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "elapsed_sec": float(time.time() - started),
            }
        )
        return row
    residual = fit["residual"]
    row.update(
        {
            "status": "ok",
            "failure_reason": "",
            "sim3_scale": float(fit["scale"]),
            "sim3_rotation_det": float(fit["rotation_det"]),
            "sim3_residual_mean": float(np.mean(residual)),
            "sim3_residual_median": float(np.median(residual)),
            "sim3_residual_p90": float(np.percentile(residual, 90)),
            "sim3_residual_p95": float(np.percentile(residual, 95)),
            "visibility_mean": anchors.get("visibility_mean"),
            "confidence_mean": anchors.get("confidence_mean"),
            "uv_in01_rate": anchors.get("uv_in01_rate"),
            "elapsed_sec": float(time.time() - started),
        }
    )
    return row


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    out: dict[str, Any] = {
        "num_windows": len(rows),
        "num_ok_windows": len(ok_rows),
        "num_failed_windows": len(rows) - len(ok_rows),
    }
    for key in (
        "sim3_anchor_count",
        "sim3_scale",
        "sim3_residual_mean",
        "sim3_residual_median",
        "sim3_residual_p90",
        "sim3_residual_p95",
        "visibility_mean",
        "confidence_mean",
        "uv_in01_rate",
    ):
        values = [float(row[key]) for row in ok_rows if row.get(key) is not None]
        if values:
            out[f"{key}_mean"] = float(np.mean(values))
            out[f"{key}_min"] = float(np.min(values))
            out[f"{key}_max"] = float(np.max(values))
    return out


def _write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# D4RT Geometry Diagnostic",
        "",
        "This diagnostic fits Sim3 from D4RT same-pixel carrier anchors to ScanNet RGB-D world points.",
        "It does not count as Stream3D-D4RT internal geometry segmentation AP.",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Windows",
            "",
            "| Scene | Window | Status | Anchors | Scale | Residual median | Residual p90 | Visibility | Confidence | uv in01 |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {scene} | {window} | {status} | {anchors} | {scale} | {median} | {p90} | {vis} | {conf} | {uv} |".format(
                scene=row.get("scene", ""),
                window=row.get("window", ""),
                status=row.get("status", ""),
                anchors=row.get("sim3_anchor_count", ""),
                scale="" if row.get("sim3_scale") is None else f"{row['sim3_scale']:.6g}",
                median="" if row.get("sim3_residual_median") is None else f"{row['sim3_residual_median']:.6g}",
                p90="" if row.get("sim3_residual_p90") is None else f"{row['sim3_residual_p90']:.6g}",
                vis="" if row.get("visibility_mean") is None else f"{row['visibility_mean']:.6g}",
                conf="" if row.get("confidence_mean") is None else f"{row['confidence_mean']:.6g}",
                uv="" if row.get("uv_in01_rate") is None else f"{row['uv_in01_rate']:.6g}",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug-root", required=True, help="Root containing per-scene carriers_window*.npz")
    parser.add_argument("--seq-list", required=True, help="Scene list")
    parser.add_argument("--output-prefix", required=True, help="Output prefix without extension")
    parser.add_argument("--scannet-root", default="data/scannet/processed")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--max-anchors-per-window", type=int, default=2000)
    parser.add_argument("--min-visibility", type=float, default=0.0)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    args = parser.parse_args()

    debug_root = Path(args.debug_root)
    scenes = _read_seq_list(Path(args.seq_list))
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        stream = ScanNetStream(seq_name=scene, backbone=args.backbone, root=args.scannet_root)
        errors = stream.validate(require_masks=False)
        scene_dir = debug_root / scene
        carrier_paths = sorted(scene_dir.glob("carriers_window*.npz"))
        if errors or not carrier_paths:
            rows.append(
                {
                    "scene": scene,
                    "window": "",
                    "status": "failed",
                    "failure_reason": "; ".join(errors) if errors else f"no carriers_window*.npz under {scene_dir}",
                    "sim3_anchor_count": 0,
                }
            )
            continue
        for carrier_path in carrier_paths:
            rows.append(
                _window_report(
                    stream=stream,
                    carrier_npz=carrier_path,
                    max_anchors=int(args.max_anchors_per_window),
                    min_visibility=float(args.min_visibility),
                    min_confidence=float(args.min_confidence),
                )
            )

    summary = _aggregate(rows)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".json")
    md_path = output_prefix.with_suffix(".md")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(row))
    json_path.write_text(
        json.dumps({"summary": _json_safe(summary), "rows": _json_safe(rows)}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(md_path, rows, summary)
    print(f"[d4rt-geometry] wrote {csv_path}")
    print(f"[d4rt-geometry] wrote {json_path}")
    print(f"[d4rt-geometry] wrote {md_path}")
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
