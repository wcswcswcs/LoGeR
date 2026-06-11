from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree

from geometry_provider.common import backproject_xy_world, fit_transform
from stream4d.export_scannet import ScanNetExporter
from stream4d.scannet_stream import ScanNetStream
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _read_seq_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _frame_ids_for_carrier_file(carrier_path: Path, num_frames: int) -> list[int]:
    summary_path = carrier_path.with_name(carrier_path.name.replace("carriers_", "").replace(".npz", "_summary.json"))
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            frame_ids = [int(value) for value in payload.get("frame_ids", [])]
            if len(frame_ids) == num_frames:
                return frame_ids
        except Exception:
            pass
    return list(range(num_frames))


def _backproject_xy_world(stream: ScanNetStream, frame_id: int, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return backproject_xy_world(stream, frame_id, xy)


def _load_carrier(carrier_path: Path) -> dict[str, np.ndarray | list[int]]:
    with np.load(carrier_path) as data:
        uv_pred = np.asarray(data["uv_pred"], dtype=np.float32)
        xyz_ref = np.asarray(data["xyz_ref"], dtype=np.float32)
        visibility = np.asarray(data["visibility_prob"], dtype=np.float32)
        confidence = np.asarray(data["confidence_prob"], dtype=np.float32)
        valid = np.asarray(data.get("valid", np.ones(uv_pred.shape[:2], dtype=bool)), dtype=bool)
        carrier_id = np.asarray(data.get("carrier_id", np.arange(uv_pred.shape[1])), dtype=np.int64)
    return {
        "uv_pred": uv_pred,
        "xyz_ref": xyz_ref,
        "visibility": visibility,
        "confidence": confidence,
        "valid": valid,
        "carrier_id": carrier_id,
        "frame_ids": _frame_ids_for_carrier_file(carrier_path, uv_pred.shape[0]),
    }


def _collect_anchors(
    stream: ScanNetStream,
    carrier_paths: list[Path],
    min_visibility: float,
    min_confidence: float,
    max_anchors: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    total_candidates = 0
    valid_candidates = 0
    for carrier_path in carrier_paths:
        data = _load_carrier(carrier_path)
        uv = np.asarray(data["uv_pred"], dtype=np.float32)
        xyz = np.asarray(data["xyz_ref"], dtype=np.float32)
        visibility = np.asarray(data["visibility"], dtype=np.float32)
        confidence = np.asarray(data["confidence"], dtype=np.float32)
        valid = np.asarray(data["valid"], dtype=bool)
        frame_ids = [int(value) for value in data["frame_ids"]]
        for local_idx, frame_id in enumerate(frame_ids):
            h = stream.load_depth(frame_id).shape[0]
            w = stream.load_depth(frame_id).shape[1]
            xy = np.stack(
                [
                    uv[local_idx, :, 0] * float(max(w - 1, 1)),
                    uv[local_idx, :, 1] * float(max(h - 1, 1)),
                ],
                axis=1,
            )
            ok = (
                valid[local_idx]
                & np.isfinite(xyz[local_idx]).all(axis=1)
                & np.isfinite(uv[local_idx]).all(axis=1)
                & (uv[local_idx, :, 0] >= 0.0)
                & (uv[local_idx, :, 0] <= 1.0)
                & (uv[local_idx, :, 1] >= 0.0)
                & (uv[local_idx, :, 1] <= 1.0)
                & (visibility[local_idx] >= float(min_visibility))
                & (confidence[local_idx] >= float(min_confidence))
            )
            total_candidates += int(ok.shape[0])
            if not np.any(ok):
                continue
            world, world_ok = _backproject_xy_world(stream, frame_id, xy[ok])
            src = xyz[local_idx][ok][world_ok]
            tgt = world[world_ok]
            valid_candidates += int(src.shape[0])
            source_parts.append(src)
            target_parts.append(tgt)
    if not source_parts:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32), {
            "anchor_candidates": int(total_candidates),
            "anchor_valid": 0,
        }
    source = np.concatenate(source_parts, axis=0)
    target = np.concatenate(target_parts, axis=0)
    if source.shape[0] > int(max_anchors):
        keep = np.linspace(0, source.shape[0] - 1, num=int(max_anchors), dtype=np.int64)
        source = source[keep]
        target = target[keep]
    return source, target, {"anchor_candidates": int(total_candidates), "anchor_valid": int(valid_candidates)}


def _fit_transform(source: np.ndarray, target: np.ndarray, robust_trim_percentile: float) -> dict[str, Any] | None:
    return fit_transform(source, target, robust_trim_percentile)


def _apply_fit(points: np.ndarray, fit: dict[str, Any] | None, raw: bool) -> np.ndarray:
    if raw or fit is None:
        return np.asarray(points, dtype=np.float32)
    scale = float(fit["scale"])
    rotation = np.asarray(fit["rotation"], dtype=np.float64)
    translation = np.asarray(fit["translation"], dtype=np.float64)
    return (scale * (np.asarray(points, dtype=np.float64) @ rotation.T) + translation).astype(np.float32)


def _spacing_stats(points: np.ndarray, max_points: int = 4096) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float32)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] < 2:
        return {
            "d4rt_point_density": int(points.shape[0]),
            "point_spacing_q25": None,
            "point_spacing_q50": None,
            "point_spacing_q75": None,
            "point_spacing_q90": None,
        }
    if points.shape[0] > max_points:
        keep = np.linspace(0, points.shape[0] - 1, num=max_points, dtype=np.int64)
        points = points[keep]
    tree = cKDTree(points)
    dist, _ = tree.query(points, k=2)
    nn = dist[:, 1]
    return {
        "d4rt_point_density": int(points.shape[0]),
        "point_spacing_q25": float(np.percentile(nn, 25)),
        "point_spacing_q50": float(np.percentile(nn, 50)),
        "point_spacing_q75": float(np.percentile(nn, 75)),
        "point_spacing_q90": float(np.percentile(nn, 90)),
    }


def _mask_points_from_d4rt(
    *,
    stream: ScanNetStream,
    exporter: ScanNetExporter,
    carrier_paths: list[Path],
    scene_fit: dict[str, Any] | None,
    mode: str,
    nn_radius: float,
    density_alpha: float,
    min_visibility: float,
    min_confidence: float,
    output_scene_dir: Path,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    object_points: dict[tuple[int, int], set[int]] = defaultdict(set)
    object_weights: Counter[tuple[int, int]] = Counter()
    all_aligned: list[np.ndarray] = []
    empty_masks = 0
    projected_masks = 0
    raw = mode == "raw"

    for window_idx, carrier_path in enumerate(carrier_paths):
        data = _load_carrier(carrier_path)
        window_fit = scene_fit
        if mode in {"window_sim3", "window_sim3_density"}:
            source, target, _ = _collect_anchors(
                stream,
                [carrier_path],
                min_visibility=min_visibility,
                min_confidence=min_confidence,
                max_anchors=4096,
            )
            window_fit = _fit_transform(source, target, robust_trim_percentile=90.0)
        uv = np.asarray(data["uv_pred"], dtype=np.float32)
        xyz = np.asarray(data["xyz_ref"], dtype=np.float32)
        visibility = np.asarray(data["visibility"], dtype=np.float32)
        confidence = np.asarray(data["confidence"], dtype=np.float32)
        valid = np.asarray(data["valid"], dtype=bool)
        frame_ids = [int(value) for value in data["frame_ids"]]
        transformed_all = _apply_fit(xyz.reshape(-1, 3), window_fit, raw=raw).reshape(xyz.shape)
        pointcloud_dir = output_scene_dir / "pointcloud"
        pointcloud_dir.mkdir(parents=True, exist_ok=True)
        np.save(pointcloud_dir / f"window{window_idx:03d}_aligned_points.npy", transformed_all.reshape(-1, 3))
        all_aligned.append(transformed_all.reshape(-1, 3))

        spacing = _spacing_stats(transformed_all.reshape(-1, 3))
        radius = float(nn_radius)
        if mode in {"scene_sim3_density", "window_sim3_density"} and spacing.get("point_spacing_q75") is not None:
            radius = max(radius, float(spacing["point_spacing_q75"]) * float(density_alpha))
        for local_idx, frame_id in enumerate(frame_ids):
            try:
                mask = stream.load_mask(frame_id)
            except FileNotFoundError:
                continue
            h, w = mask.shape[:2]
            x = np.rint(uv[local_idx, :, 0] * float(max(w - 1, 1))).astype(np.int64)
            y = np.rint(uv[local_idx, :, 1] * float(max(h - 1, 1))).astype(np.int64)
            ok = (
                valid[local_idx]
                & np.isfinite(transformed_all[local_idx]).all(axis=1)
                & np.isfinite(uv[local_idx]).all(axis=1)
                & (x >= 0)
                & (x < w)
                & (y >= 0)
                & (y < h)
                & (visibility[local_idx] >= float(min_visibility))
                & (confidence[local_idx] >= float(min_confidence))
            )
            if not np.any(ok):
                continue
            mask_ids = mask[y[ok], x[ok]].astype(np.int64)
            points = transformed_all[local_idx][ok]
            positive = mask_ids > 0
            if not np.any(positive):
                continue
            dist, nn_idx = exporter.tree.query(points[positive], k=1, distance_upper_bound=radius)
            hit = np.isfinite(dist) & (nn_idx < exporter.scene_points.shape[0])
            projected_masks += int(len(set((frame_id, int(v)) for v in mask_ids[positive].tolist())))
            if not np.any(hit):
                empty_masks += int(len(set((frame_id, int(v)) for v in mask_ids[positive].tolist())))
                continue
            for mask_id, point_id in zip(mask_ids[positive][hit].tolist(), nn_idx[hit].tolist()):
                key = (int(frame_id), int(mask_id))
                object_points[key].add(int(point_id))
                object_weights[key] += 1
            depth_like = np.full((h, w), np.nan, dtype=np.float32)
            depth_like[y[ok][positive], x[ok][positive]] = points[positive, 2]
            depth_dir = output_scene_dir / "depth_like"
            depth_dir.mkdir(parents=True, exist_ok=True)
            np.save(depth_dir / f"window{window_idx:03d}_frame{frame_id:06d}.npy", depth_like)

    object_dict: dict[int, dict[str, Any]] = {}
    for object_id, (key, point_ids) in enumerate(sorted(object_points.items())):
        object_dict[int(object_id)] = {
            "point_ids": np.asarray(sorted(point_ids), dtype=np.int64),
            "mask_list": [(int(key[0]), int(key[1]), float(object_weights[key]))],
            "carrier_ids": np.empty((0,), dtype=np.int64),
        }
    aligned = np.concatenate(all_aligned, axis=0) if all_aligned else np.empty((0, 3), dtype=np.float32)
    diag = {
        "num_2d_masks_with_d4rt_points": int(projected_masks),
        "num_3d_masks_after_projection": int(len(object_dict)),
        "empty_projected_mask_ratio": float(empty_masks / max(projected_masks, 1)),
        **_spacing_stats(aligned),
    }
    return object_dict, diag


def _fit_summary(fit: dict[str, Any] | None) -> dict[str, Any]:
    if fit is None:
        return {
            "anchor_count": 0,
            "sim3_scale": None,
            "sim3_rotation_det": None,
            "translation_norm": None,
            "inlier_ratio": None,
            "median_residual": None,
            "p90_residual": None,
            "p95_residual": None,
        }
    residual = np.asarray(fit["residual"], dtype=np.float64)
    return {
        "anchor_count": int(fit.get("anchor_count", residual.shape[0])),
        "sim3_scale": float(fit["scale"]),
        "sim3_rotation_det": float(fit["rotation_det"]),
        "translation_norm": float(np.linalg.norm(np.asarray(fit["translation"], dtype=np.float64))),
        "inlier_ratio": float(np.mean(residual <= np.percentile(residual, 90))) if residual.size else None,
        "median_residual": float(np.median(residual)) if residual.size else None,
        "p90_residual": float(np.percentile(residual, 90)) if residual.size else None,
        "p95_residual": float(np.percentile(residual, 95)) if residual.size else None,
    }


def _export_scene(args: argparse.Namespace, seq_name: str) -> dict[str, Any]:
    stream = ScanNetStream(seq_name=seq_name, backbone=args.backbone)
    errors = stream.validate(require_masks=True)
    if errors:
        raise RuntimeError("; ".join(errors))
    scene_debug_dir = Path(args.debug_root) / seq_name
    carrier_paths = sorted(scene_debug_dir.glob("carriers_window*.npz"))
    if not carrier_paths:
        raise FileNotFoundError(f"No carrier windows under {scene_debug_dir}")
    output_scene_dir = Path(args.output_geometry_root) / seq_name
    output_scene_dir.mkdir(parents=True, exist_ok=True)

    source, target, anchor_diag = _collect_anchors(
        stream,
        carrier_paths,
        min_visibility=float(args.min_visibility),
        min_confidence=float(args.min_confidence),
        max_anchors=int(args.max_anchors),
    )
    scene_fit = None if args.mode == "raw" else _fit_transform(source, target, robust_trim_percentile=float(args.robust_trim_percentile))
    exporter = ScanNetExporter(
        stream,
        output_config=args.output_config,
        export_support_mode="reuse_point_ids",
        export_min_points_per_object=int(args.min_points_per_object),
        export_score_mode="area",
    )
    object_dict, projection_diag = _mask_points_from_d4rt(
        stream=stream,
        exporter=exporter,
        carrier_paths=carrier_paths,
        scene_fit=scene_fit,
        mode=args.mode,
        nn_radius=float(args.nn_radius),
        density_alpha=float(args.density_alpha),
        min_visibility=float(args.min_visibility),
        min_confidence=float(args.min_confidence),
        output_scene_dir=output_scene_dir,
    )
    export_diag = exporter.export_object_dict_points(object_dict)
    summary = {
        "seq_name": seq_name,
        "mode": args.mode,
        "geometry_source": "d4rt_raw" if args.mode == "raw" else "d4rt_sim3_aligned",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic": True,
        "diagnostic_alignment_uses_scannet_depth_pose": True,
        **anchor_diag,
        **_fit_summary(scene_fit),
        **projection_diag,
        **export_diag,
    }
    manifest = {
        "scene": seq_name,
        "mode": args.mode,
        "diagnostic_alignment_uses_scannet_depth_pose": True,
        "fit": _json_safe(_fit_summary(scene_fit)),
        "projection": _json_safe(projection_diag),
        "export": _json_safe(export_diag),
    }
    (output_scene_dir / "geometry_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_scene_dir / "poses_or_reference.json").write_text(
        json.dumps(
            _json_safe(
                {
                    "reference": "D4RT xyz_ref transformed into ScanNet world with diagnostic Sim3"
                    if args.mode != "raw"
                    else "D4RT raw xyz_ref, no Sim3",
                    "mode": args.mode,
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return summary


def _write_aggregate(args: argparse.Namespace, summaries: list[dict[str, Any]]) -> None:
    out_dir = Path(args.summary_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    numeric_keys = sorted(
        {
            key
            for row in summaries
            for key, value in row.items()
            if isinstance(value, (int, float, np.generic)) and not isinstance(value, bool)
        }
    )
    aggregate = {
        "args": vars(args),
        "algorithm": "v10_d4rt_aligned_geometry_for_stream3d_minimal_projection",
        "mode": args.mode,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic": True,
        "is_method_result": False,
        "is_diagnostic_only": True,
        "num_scenes": len(summaries),
        "scenes": summaries,
        "numeric_mean": {
            key: float(np.mean([float(row[key]) for row in summaries if key in row and row[key] is not None]))
            for key in numeric_keys
            if any(key in row and row[key] is not None for row in summaries)
        },
    }
    aggregate_path = out_dir / f"{args.output_config}_summary.json"
    aggregate_path.write_text(json.dumps(_json_safe(aggregate), indent=2, sort_keys=True), encoding="utf-8")
    csv_path = out_dir / f"{args.output_config}_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seq_name"] + numeric_keys)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key) for key in ["seq_name"] + numeric_keys})
    md_path = out_dir / f"{args.output_config}_summary.md"
    lines = [
        f"# {args.output_config}",
        "",
        "This is a diagnostic-only minimal geometry replacement. It replaces 2D mask projection points with D4RT raw/Sim3-aligned points, then maps them to the ScanNet mesh for evaluator compatibility.",
        "",
        "| scene | anchors | scale | median residual | p90 residual | projected masks | empty ratio | objects | points |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {scene} | {anchors} | {scale} | {median} | {p90} | {masks} | {empty} | {objects} | {points} |".format(
                scene=row.get("seq_name"),
                anchors=int(row.get("anchor_count") or 0),
                scale="NA" if row.get("sim3_scale") is None else f"{float(row['sim3_scale']):.6g}",
                median="NA" if row.get("median_residual") is None else f"{float(row['median_residual']):.6g}",
                p90="NA" if row.get("p90_residual") is None else f"{float(row['p90_residual']):.6g}",
                masks=int(row.get("num_3d_masks_after_projection") or 0),
                empty=f"{float(row.get('empty_projected_mask_ratio') or 0.0):.3f}",
                objects=int(row.get("num_exported_objects") or 0),
                points=int(row.get("num_exported_points") or 0),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = build_prediction_manifest(
        root=".",
        output_config=args.output_config,
        is_method_result=False,
        is_diagnostic_only=True,
        uses_gt=False,
        gt_usage="scannet_depth_pose_for_sim3_geometry_diagnostic",
        source_configs=[str(args.debug_root)],
        pre_points_policy="diagnostic_recompute",
        support_policy=f"v10_d4rt_geometry:{args.mode}",
        notes=(
            "Diagnostic-only D4RT geometry attribution artifact. ScanNet RGB-D depth/pose are used as "
            "Sim3 anchors to isolate metric geometry errors; no instance labels are used for prediction. "
            "This config must not enter method result tables."
        ),
        extra={
            "algorithm": "v10_d4rt_aligned_geometry_for_stream3d_minimal_projection",
            "mode": args.mode,
            "eval_policy": args.eval_policy,
            "prediction_config": args.output_config,
            "pre_points_config": args.output_config,
            "support_source": "own",
            "geometry_source": "d4rt_raw" if args.mode == "raw" else "d4rt_sim3_aligned",
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic": True,
            "is_method_result": False,
            "is_diagnostic_only": True,
            "summary_path": str(aggregate_path),
            "seq_list": str(args.seq_list),
        },
    )
    write_prediction_manifest(args.output_config, manifest, root=".")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize diagnostic D4RT raw/Sim3-aligned geometry for Stream3D-style projection.")
    parser.add_argument("--debug-root", required=True)
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument(
        "--mode",
        choices=["raw", "scene_sim3", "window_sim3", "scene_sim3_density", "window_sim3_density"],
        required=True,
    )
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--summary-root", default="outputs/v10_d4rt_geometry")
    parser.add_argument("--output-geometry-root", default="data/scannet_d4rt_aligned")
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--max-anchors", type=int, default=8000)
    parser.add_argument("--robust-trim-percentile", type=float, default=90.0)
    parser.add_argument("--nn-radius", type=float, default=0.05)
    parser.add_argument("--density-alpha", type=float, default=2.0)
    parser.add_argument("--min-points-per-object", type=int, default=1)
    parser.add_argument("--eval-policy", default="d4rt_geometry_diagnostic")
    args = parser.parse_args()

    summaries = [_export_scene(args, seq_name) for seq_name in _read_seq_list(Path(args.seq_list))]
    _write_aggregate(args, summaries)
    print(json.dumps(_json_safe({"output_config": args.output_config, "scenes": summaries}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
