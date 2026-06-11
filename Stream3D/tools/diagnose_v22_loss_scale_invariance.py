from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

STREAM3D_ROOT = Path(__file__).resolve().parents[1]
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from geometry_provider.d4rt_carrier_provider import D4RTCarrierProjectionProvider
from stream4d.scannet_stream import ScanNetStream
from tools.diagnose_v22_ref0_trajectory_scale import _finite_values
from tools.run_v22_direct_reconstruction_benchmark import _json_safe, _read_seq_list, _sample_indices


def _d4rt_signed_log1p(xyz: np.ndarray) -> np.ndarray:
    arr = np.asarray(xyz, dtype=np.float64)
    return np.sign(arr) * np.log1p(np.abs(arr))


def _d4rt_xyz_preprocess(
    xyz: np.ndarray,
    *,
    normalize_depth: bool = True,
    transform_log: bool = True,
) -> tuple[np.ndarray, float | None]:
    """Mirror OpenD4RT xyz loss preprocessing for one sampled frame."""
    arr = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    out = arr.copy()
    scale: float | None = None
    if normalize_depth:
        depth = np.abs(out[:, 2])
        valid = np.isfinite(depth) & (depth > 1e-8)
        if not np.any(valid):
            return np.full_like(out, np.nan, dtype=np.float64), None
        scale = float(np.mean(depth[valid]))
        out = out / scale
    if transform_log:
        out = _d4rt_signed_log1p(out)
    return out, scale


def _loss_space_l1(
    pred_xyz: np.ndarray,
    gt_xyz: np.ndarray,
    *,
    normalize_depth: bool = True,
    transform_log: bool = True,
) -> float | None:
    pred = np.asarray(pred_xyz, dtype=np.float64).reshape(-1, 3)
    gt = np.asarray(gt_xyz, dtype=np.float64).reshape(-1, 3)
    ok = np.isfinite(pred).all(axis=1) & np.isfinite(gt).all(axis=1)
    if np.count_nonzero(ok) == 0:
        return None
    pred_prep, pred_scale = _d4rt_xyz_preprocess(pred[ok], normalize_depth=normalize_depth, transform_log=transform_log)
    gt_prep, gt_scale = _d4rt_xyz_preprocess(gt[ok], normalize_depth=normalize_depth, transform_log=transform_log)
    if normalize_depth and (pred_scale is None or gt_scale is None):
        return None
    finite = np.isfinite(pred_prep).all(axis=1) & np.isfinite(gt_prep).all(axis=1)
    if np.count_nonzero(finite) == 0:
        return None
    return float(np.mean(np.abs(pred_prep[finite] - gt_prep[finite])))


def _metric_l1(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> float | None:
    pred = np.asarray(pred_xyz, dtype=np.float64).reshape(-1, 3)
    gt = np.asarray(gt_xyz, dtype=np.float64).reshape(-1, 3)
    ok = np.isfinite(pred).all(axis=1) & np.isfinite(gt).all(axis=1)
    if np.count_nonzero(ok) == 0:
        return None
    return float(np.mean(np.abs(pred[ok] - gt[ok])))


def _metric_z_absrel(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> float | None:
    pred = np.asarray(pred_xyz, dtype=np.float64).reshape(-1, 3)
    gt = np.asarray(gt_xyz, dtype=np.float64).reshape(-1, 3)
    pred_z = np.abs(pred[:, 2])
    gt_z = np.abs(gt[:, 2])
    ok = np.isfinite(pred_z) & np.isfinite(gt_z) & (pred_z > 1e-8) & (gt_z > 1e-8)
    if np.count_nonzero(ok) == 0:
        return None
    return float(np.mean(np.abs(pred_z[ok] - gt_z[ok]) / gt_z[ok]))


def _sample_gt_camera_xyz(
    stream: ScanNetStream,
    intrinsics: np.ndarray,
    frame_id: int,
    uv_norm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    uv = np.asarray(uv_norm, dtype=np.float64).reshape(-1, 2)
    depth = stream.load_depth(int(frame_id))
    h, w = depth.shape[:2]
    x = np.rint(uv[:, 0] * float(max(w - 1, 1))).astype(np.int64)
    y = np.rint(uv[:, 1] * float(max(h - 1, 1))).astype(np.int64)
    ok = (
        np.isfinite(uv).all(axis=1)
        & (x >= 0)
        & (x < w)
        & (y >= 0)
        & (y < h)
    )
    z = np.full((uv.shape[0],), np.nan, dtype=np.float64)
    z[ok] = depth[y[ok], x[ok]].astype(np.float64)
    ok &= np.isfinite(z) & (z > 1e-8)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    out = np.full((uv.shape[0], 3), np.nan, dtype=np.float64)
    out[ok, 0] = (x[ok].astype(np.float64) - cx) * z[ok] / fx
    out[ok, 1] = (y[ok].astype(np.float64) - cy) * z[ok] / fy
    out[ok, 2] = z[ok]
    return out, ok


def _finite_mean(values: list[float | None]) -> float | None:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else None


def _finite_median(values: list[float | None]) -> float | None:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else None


def _range(values: list[float | None]) -> float | None:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.max(arr) - np.min(arr))


def _best_scale(sweep_rows: list[dict[str, Any]], key: str) -> float | None:
    valid = [row for row in sweep_rows if row.get(key) is not None and np.isfinite(float(row[key]))]
    if not valid:
        return None
    return float(min(valid, key=lambda row: float(row[key]))["pred_scale"])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return "NA"
        return f"{float(value):.6f}"
    if isinstance(value, bool):
        return str(value)
    return str(value)


def _write_md(
    path: Path,
    scene_rows: list[dict[str, Any]],
    scale_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    lines: list[str] = [
        "# v22.15 OpenD4RT xyz loss scale-invariance diagnostic",
        "",
        "Diagnostic-only: samples D4RT `xyz_local` at target UVs, builds GT camera-space XYZ from ScanNet depth, and sweeps uniform prediction scales.",
        "",
        "The OpenD4RT xyz loss preprocesses prediction and target independently by each sample's own mean absolute z before signed-log transform, so uniform prediction scale is expected to be invisible in loss space.",
        "",
        "## Metadata",
        "",
        "| item | value |",
        "|---|---:|",
    ]
    for key in sorted(metadata.keys()):
        lines.append(f"| {key} | {_fmt(metadata[key])} |")
    lines.extend(
        [
            "",
            "## Scale Sweep",
            "",
            "| pred scale | frames | loss L1 mean | loss L1 frame range | normalized loss L1 mean | metric L1 mean | z absrel mean |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in scale_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("pred_scale")),
                    _fmt(row.get("frame_count")),
                    _fmt(row.get("loss_l1_signed_log_mean")),
                    _fmt(row.get("loss_l1_signed_log_range")),
                    _fmt(row.get("loss_l1_normalized_mean")),
                    _fmt(row.get("metric_l1_mean")),
                    _fmt(row.get("metric_z_absrel_mean")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Per-scene",
            "",
            "| scene | status | frames | anchors mean | loss range mean | metric range mean | metric best scale median | GT/pred mean-z median |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in scene_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("scene")),
                    _fmt(row.get("status")),
                    _fmt(row.get("frame_count")),
                    _fmt(row.get("anchor_count_mean")),
                    _fmt(row.get("loss_l1_signed_log_range_mean")),
                    _fmt(row.get("metric_l1_range_mean")),
                    _fmt(row.get("metric_l1_best_scale_median")),
                    _fmt(row.get("gt_over_pred_mean_abs_z_median")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _diagnose_scene(
    scene: str,
    *,
    stream: ScanNetStream,
    provider: D4RTCarrierProjectionProvider,
    pred_scales: list[float],
    max_windows_per_scene: int | None,
    max_anchors: int,
    min_visibility: float,
    min_confidence: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cache = provider._load_scene(scene)
    windows = list(cache["windows"])
    if max_windows_per_scene is not None:
        windows = windows[: int(max_windows_per_scene)]
    intrinsics = stream.load_intrinsics()
    frame_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for window_index, window in enumerate(windows):
        with np.load(window.path) as data:
            if "xyz_local" not in data.files:
                window_rows.append(
                    {
                        "scene": scene,
                        "window_index": int(window_index),
                        "window_path": str(window.path),
                        "status": "missing_xyz_local",
                    }
                )
                continue
            xyz_local = np.asarray(data["xyz_local"], dtype=np.float64)
        per_frame_cap = max(16, int(max_anchors) // max(len(window.frame_ids), 1))
        frame_count_before = len(frame_rows)
        for local_idx, frame_id in enumerate(window.frame_ids):
            uv = np.asarray(window.uv[local_idx], dtype=np.float64)
            ok = (
                np.asarray(window.valid[local_idx], dtype=bool)
                & np.isfinite(xyz_local[local_idx]).all(axis=1)
                & np.isfinite(uv).all(axis=1)
                & (uv[:, 0] >= 0.0)
                & (uv[:, 0] <= 1.0)
                & (uv[:, 1] >= 0.0)
                & (uv[:, 1] <= 1.0)
                & (np.asarray(window.visibility[local_idx], dtype=np.float64) >= float(min_visibility))
                & (np.asarray(window.confidence[local_idx], dtype=np.float64) >= float(min_confidence))
            )
            indices = np.flatnonzero(ok)
            if indices.shape[0] < 8:
                continue
            indices = _sample_indices(indices, per_frame_cap)
            gt_xyz, gt_ok = _sample_gt_camera_xyz(stream, intrinsics, int(frame_id), uv[indices])
            if np.count_nonzero(gt_ok) < 8:
                continue
            pred_xyz = xyz_local[local_idx, indices][gt_ok]
            gt_xyz = gt_xyz[gt_ok]
            pred_mean_abs_z = float(np.mean(np.abs(pred_xyz[:, 2])))
            gt_mean_abs_z = float(np.mean(np.abs(gt_xyz[:, 2])))
            frame_sweep: list[dict[str, Any]] = []
            for pred_scale in pred_scales:
                scaled = pred_xyz * float(pred_scale)
                row = {
                    "scene": scene,
                    "window_index": int(window_index),
                    "window_path": str(window.path),
                    "ref_frame": int(window.frame_ids[0]) if window.frame_ids else -1,
                    "frame_id": int(frame_id),
                    "local_idx": int(local_idx),
                    "anchor_count": int(pred_xyz.shape[0]),
                    "pred_scale": float(pred_scale),
                    "metric_l1": _metric_l1(scaled, gt_xyz),
                    "metric_z_absrel": _metric_z_absrel(scaled, gt_xyz),
                    "loss_l1_signed_log": _loss_space_l1(scaled, gt_xyz, normalize_depth=True, transform_log=True),
                    "loss_l1_normalized": _loss_space_l1(scaled, gt_xyz, normalize_depth=True, transform_log=False),
                    "loss_l1_raw_no_norm": _loss_space_l1(scaled, gt_xyz, normalize_depth=False, transform_log=False),
                }
                frame_sweep.append(row)
                sweep_rows.append(row)
            loss_values = [row.get("loss_l1_signed_log") for row in frame_sweep]
            normalized_loss_values = [row.get("loss_l1_normalized") for row in frame_sweep]
            metric_values = [row.get("metric_l1") for row in frame_sweep]
            z_absrel_values = [row.get("metric_z_absrel") for row in frame_sweep]
            frame_rows.append(
                {
                    "scene": scene,
                    "window_index": int(window_index),
                    "window_path": str(window.path),
                    "ref_frame": int(window.frame_ids[0]) if window.frame_ids else -1,
                    "frame_id": int(frame_id),
                    "local_idx": int(local_idx),
                    "anchor_count": int(pred_xyz.shape[0]),
                    "pred_mean_abs_z": pred_mean_abs_z,
                    "gt_mean_abs_z": gt_mean_abs_z,
                    "gt_over_pred_mean_abs_z": float(gt_mean_abs_z / pred_mean_abs_z) if pred_mean_abs_z > 1e-8 else None,
                    "loss_l1_signed_log_range": _range(loss_values),
                    "loss_l1_normalized_range": _range(normalized_loss_values),
                    "metric_l1_range": _range(metric_values),
                    "metric_z_absrel_range": _range(z_absrel_values),
                    "metric_l1_best_scale": _best_scale(frame_sweep, "metric_l1"),
                    "metric_z_absrel_best_scale": _best_scale(frame_sweep, "metric_z_absrel"),
                    "loss_l1_signed_log_at_scale_1": next(
                        (row.get("loss_l1_signed_log") for row in frame_sweep if abs(float(row["pred_scale"]) - 1.0) < 1e-8),
                        None,
                    ),
                    "metric_l1_at_scale_1": next(
                        (row.get("metric_l1") for row in frame_sweep if abs(float(row["pred_scale"]) - 1.0) < 1e-8),
                        None,
                    ),
                }
            )
        window_rows.append(
            {
                "scene": scene,
                "window_index": int(window_index),
                "window_path": str(window.path),
                "status": "ok",
                "frame_row_count": int(len(frame_rows) - frame_count_before),
                "window_frame_count": int(len(window.frame_ids)),
            }
        )
    return frame_rows, sweep_rows, window_rows


def _scene_summary(scene: str, frame_rows: list[dict[str, Any]], window_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in frame_rows if row.get("scene") == scene]
    summary: dict[str, Any] = {
        "scene": scene,
        "status": "ok" if rows else "no_frame_rows",
        "frame_count": int(len(rows)),
        "window_count": int(len([row for row in window_rows if row.get("scene") == scene])),
    }
    for key in [
        "anchor_count",
        "pred_mean_abs_z",
        "gt_mean_abs_z",
        "gt_over_pred_mean_abs_z",
        "loss_l1_signed_log_range",
        "loss_l1_normalized_range",
        "metric_l1_range",
        "metric_z_absrel_range",
        "metric_l1_best_scale",
        "metric_z_absrel_best_scale",
        "loss_l1_signed_log_at_scale_1",
        "metric_l1_at_scale_1",
    ]:
        values = _finite_values(rows, key)
        summary[f"{key}_mean"] = float(np.mean(values)) if values.size else None
        summary[f"{key}_median"] = float(np.median(values)) if values.size else None
    return summary


def _scale_summary(sweep_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scales = sorted({float(row["pred_scale"]) for row in sweep_rows})
    out: list[dict[str, Any]] = []
    for scale in scales:
        rows = [row for row in sweep_rows if abs(float(row["pred_scale"]) - scale) < 1e-12]
        out.append(
            {
                "pred_scale": float(scale),
                "frame_count": int(len(rows)),
                "metric_l1_mean": _finite_mean([row.get("metric_l1") for row in rows]),
                "metric_l1_median": _finite_median([row.get("metric_l1") for row in rows]),
                "metric_z_absrel_mean": _finite_mean([row.get("metric_z_absrel") for row in rows]),
                "metric_z_absrel_median": _finite_median([row.get("metric_z_absrel") for row in rows]),
                "loss_l1_signed_log_mean": _finite_mean([row.get("loss_l1_signed_log") for row in rows]),
                "loss_l1_signed_log_median": _finite_median([row.get("loss_l1_signed_log") for row in rows]),
                "loss_l1_signed_log_range": _range([row.get("loss_l1_signed_log") for row in rows]),
                "loss_l1_normalized_mean": _finite_mean([row.get("loss_l1_normalized") for row in rows]),
                "loss_l1_normalized_range": _range([row.get("loss_l1_normalized") for row in rows]),
                "loss_l1_raw_no_norm_mean": _finite_mean([row.get("loss_l1_raw_no_norm") for row in rows]),
            }
        )
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose uniform-scale invariance of the OpenD4RT xyz loss.")
    parser.add_argument("--seq-list", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v22_local_xyz_probe5_r1")
    parser.add_argument("--audit-root", default="outputs/audit/v22_15_loss_scale_invariance_probe5")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-windows-per-scene", type=int, default=1)
    parser.add_argument("--max-anchors", type=int, default=8000)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--pred-scales", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    audit_root = Path(args.audit_root)
    audit_root.mkdir(parents=True, exist_ok=True)
    scenes = _read_seq_list(Path(args.seq_list))
    if args.max_scenes is not None:
        scenes = scenes[: int(args.max_scenes)]
    pred_scales = [float(v) for v in args.pred_scales]

    provider = D4RTCarrierProjectionProvider(
        debug_root=args.cache_root,
        mode="raw",
        max_anchors=int(args.max_anchors),
        min_visibility=float(args.min_visibility),
        min_confidence=float(args.min_confidence),
    )
    all_frame_rows: list[dict[str, Any]] = []
    all_sweep_rows: list[dict[str, Any]] = []
    all_window_rows: list[dict[str, Any]] = []
    scene_summaries: list[dict[str, Any]] = []

    for scene in scenes:
        scene_dir = Path(args.cache_root) / scene
        if not scene_dir.exists():
            scene_summaries.append({"scene": scene, "status": "missing_d4rt_cache", "frame_count": 0, "window_count": 0})
            continue
        stream = ScanNetStream(seq_name=scene, backbone=args.backbone)
        frame_rows, sweep_rows, window_rows = _diagnose_scene(
            scene,
            stream=stream,
            provider=provider,
            pred_scales=pred_scales,
            max_windows_per_scene=args.max_windows_per_scene,
            max_anchors=int(args.max_anchors),
            min_visibility=float(args.min_visibility),
            min_confidence=float(args.min_confidence),
        )
        all_frame_rows.extend(frame_rows)
        all_sweep_rows.extend(sweep_rows)
        all_window_rows.extend(window_rows)
        scene_summaries.append(_scene_summary(scene, frame_rows, window_rows))

    scale_rows = _scale_summary(all_sweep_rows)
    loss_range_values = _finite_values(all_frame_rows, "loss_l1_signed_log_range")
    normalized_loss_range_values = _finite_values(all_frame_rows, "loss_l1_normalized_range")
    metric_range_values = _finite_values(all_frame_rows, "metric_l1_range")
    metric_z_range_values = _finite_values(all_frame_rows, "metric_z_absrel_range")
    metadata = {
        "is_diagnostic_only": True,
        "uses_scannet_depth_for_gt_xyz": True,
        "uses_gt_for_prediction": False,
        "forbidden_for_method_table": True,
        "loss_reference": "OpenD4RT d4rt_loss._xyz_preprocess",
        "loss_normalize_pred_and_gt_independently": True,
        "loss_normalize_by_mean_abs_z": True,
        "loss_value_transform": "sign(x)*log1p(abs(x))",
        "scene_count_requested": int(len(scenes)),
        "frame_row_count": int(len(all_frame_rows)),
        "sweep_row_count": int(len(all_sweep_rows)),
        "loss_l1_signed_log_range_across_pred_scales_mean": float(np.mean(loss_range_values)) if loss_range_values.size else None,
        "loss_l1_signed_log_range_across_pred_scales_max": float(np.max(loss_range_values)) if loss_range_values.size else None,
        "loss_l1_normalized_range_across_pred_scales_mean": float(np.mean(normalized_loss_range_values)) if normalized_loss_range_values.size else None,
        "loss_l1_normalized_range_across_pred_scales_max": float(np.max(normalized_loss_range_values)) if normalized_loss_range_values.size else None,
        "metric_l1_range_across_pred_scales_mean": float(np.mean(metric_range_values)) if metric_range_values.size else None,
        "metric_l1_range_across_pred_scales_max": float(np.max(metric_range_values)) if metric_range_values.size else None,
        "metric_z_absrel_range_across_pred_scales_mean": float(np.mean(metric_z_range_values)) if metric_z_range_values.size else None,
        "metric_z_absrel_range_across_pred_scales_max": float(np.max(metric_z_range_values)) if metric_z_range_values.size else None,
        "pred_scales": ",".join(str(v) for v in pred_scales),
        "max_windows_per_scene": args.max_windows_per_scene,
        "max_anchors": int(args.max_anchors),
    }

    _write_csv(audit_root / "loss_scale_frame_rows.csv", all_frame_rows)
    _write_csv(audit_root / "loss_scale_sweep_rows.csv", all_sweep_rows)
    _write_csv(audit_root / "loss_scale_scene_summary.csv", scene_summaries)
    _write_csv(audit_root / "loss_scale_by_pred_scale.csv", scale_rows)
    (audit_root / "loss_scale_frame_rows.json").write_text(json.dumps(_json_safe(all_frame_rows), indent=2), encoding="utf-8")
    (audit_root / "loss_scale_sweep_rows.json").write_text(json.dumps(_json_safe(all_sweep_rows), indent=2), encoding="utf-8")
    (audit_root / "loss_scale_scene_summary.json").write_text(json.dumps(_json_safe(scene_summaries), indent=2), encoding="utf-8")
    (audit_root / "loss_scale_by_pred_scale.json").write_text(json.dumps(_json_safe(scale_rows), indent=2), encoding="utf-8")
    (audit_root / "loss_scale_metadata.json").write_text(json.dumps(_json_safe(metadata), indent=2), encoding="utf-8")
    _write_md(audit_root / "loss_scale_invariance.md", scene_summaries, scale_rows, metadata)
    print(f"Wrote v22.15 loss scale-invariance diagnostic to {audit_root}")


if __name__ == "__main__":
    main()
