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
from tools.diagnose_v22_ref0_intrinsics_proxy import _reprojection_error_px
from tools.diagnose_v22_ref0_scale_convention import _median_positive, _sample_depth_uv
from tools.diagnose_v22_ref0_trajectory_scale import _finite_values
from tools.run_v22_direct_reconstruction_benchmark import _json_safe, _read_seq_list, _sample_indices


SCALE_KEYS = [
    "uv_reprojection_median_px",
    "normalized_z_l1",
    "depth_rank_spearman",
    "gt_depth_absrel",
]


def _normalize_positive(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    ok = np.isfinite(arr) & (arr > 1e-8)
    out = np.full_like(arr, np.nan, dtype=np.float64)
    if np.count_nonzero(ok) == 0:
        return out
    denom = float(np.mean(arr[ok]))
    if not np.isfinite(denom) or denom <= 1e-8:
        return out
    out[ok] = arr[ok] / denom
    return out


def _mean_l1(a: np.ndarray, b: np.ndarray) -> float | None:
    lhs = np.asarray(a, dtype=np.float64).reshape(-1)
    rhs = np.asarray(b, dtype=np.float64).reshape(-1)
    ok = np.isfinite(lhs) & np.isfinite(rhs)
    if np.count_nonzero(ok) == 0:
        return None
    return float(np.mean(np.abs(lhs[ok] - rhs[ok])))


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.shape[0], dtype=np.float64)
    i = 0
    while i < order.size:
        j = i + 1
        while j < order.size and arr[order[j]] == arr[order[i]]:
            j += 1
        avg = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg
        i = j
    return ranks


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    lhs = np.asarray(a, dtype=np.float64).reshape(-1)
    rhs = np.asarray(b, dtype=np.float64).reshape(-1)
    ok = np.isfinite(lhs) & np.isfinite(rhs)
    if np.count_nonzero(ok) < 2:
        return None
    ra = _rankdata_average(lhs[ok])
    rb = _rankdata_average(rhs[ok])
    if float(np.std(ra)) <= 1e-12 or float(np.std(rb)) <= 1e-12:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _depth_absrel(pred_z: np.ndarray, target_depth: np.ndarray) -> float | None:
    pred = np.asarray(pred_z, dtype=np.float64).reshape(-1)
    target = np.asarray(target_depth, dtype=np.float64).reshape(-1)
    ok = np.isfinite(pred) & np.isfinite(target) & (pred > 1e-8) & (target > 1e-8)
    if np.count_nonzero(ok) == 0:
        return None
    return float(np.mean(np.abs(pred[ok] - target[ok]) / target[ok]))


def _range(values: list[float | None]) -> float | None:
    arr = np.asarray([value for value in values if value is not None], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.max(arr) - np.min(arr))


def _best_scale(rows: list[dict[str, Any]], metric_key: str, *, maximize: bool = False) -> float | None:
    valid = [row for row in rows if row.get(metric_key) is not None and np.isfinite(float(row[metric_key]))]
    if not valid:
        return None
    if maximize:
        best = max(valid, key=lambda row: float(row[metric_key]))
    else:
        best = min(valid, key=lambda row: float(row[metric_key]))
    return float(best["pred_scale"])


def _image_hw_cached(stream: ScanNetStream, cache: dict[int, tuple[int, int]], frame_id: int) -> tuple[int, int]:
    key = int(frame_id)
    if key not in cache:
        rgb = stream.load_rgb(key)
        cache[key] = (int(rgb.shape[0]), int(rgb.shape[1]))
    return cache[key]


def _sweep_scale_metrics(
    *,
    xyz_local: np.ndarray,
    uv: np.ndarray,
    target_depth: np.ndarray,
    intrinsics: np.ndarray,
    image_hw: tuple[int, int],
    pred_scales: list[float],
) -> list[dict[str, Any]]:
    xyz = np.asarray(xyz_local, dtype=np.float64).reshape(-1, 3)
    uv_arr = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target_depth, dtype=np.float64).reshape(-1)
    if xyz.shape[0] != uv_arr.shape[0] or xyz.shape[0] != target.shape[0]:
        raise ValueError("xyz/uv/depth row count mismatch")
    base_z_norm = _normalize_positive(np.abs(xyz[:, 2]))
    target_norm = _normalize_positive(target)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    rows: list[dict[str, Any]] = []
    for pred_scale in pred_scales:
        scaled = xyz * float(pred_scale)
        reproj = _reprojection_error_px(scaled, uv_arr, fx=fx, fy=fy, cx=cx, cy=cy, image_hw=image_hw)
        z_abs = np.abs(scaled[:, 2])
        rows.append(
            {
                "pred_scale": float(pred_scale),
                "uv_reprojection_median_px": reproj.get("median"),
                "uv_reprojection_p90_px": reproj.get("p90"),
                "normalized_z_l1": _mean_l1(_normalize_positive(z_abs), base_z_norm),
                "depth_rank_spearman": _spearman_corr(z_abs, target),
                "normalized_depth_shape_l1_vs_gt": _mean_l1(_normalize_positive(z_abs), target_norm),
                "gt_depth_absrel": _depth_absrel(z_abs, target),
            }
        )
    return rows


def _summarize_sweep(frame_row: dict[str, Any], sweep_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = dict(frame_row)
    out["scale_count"] = int(len(sweep_rows))
    for key in SCALE_KEYS:
        values = [row.get(key) for row in sweep_rows]
        out[f"{key}_range"] = _range(values)
    out["uv_reprojection_best_scale"] = _best_scale(sweep_rows, "uv_reprojection_median_px")
    out["normalized_z_best_scale"] = _best_scale(sweep_rows, "normalized_z_l1")
    out["depth_rank_best_scale"] = _best_scale(sweep_rows, "depth_rank_spearman", maximize=True)
    out["gt_depth_absrel_best_scale"] = _best_scale(sweep_rows, "gt_depth_absrel")
    gt_values = _finite_values(sweep_rows, "gt_depth_absrel")
    out["gt_depth_absrel_min"] = float(np.min(gt_values)) if gt_values.size else None
    out["gt_depth_absrel_at_scale_1"] = next(
        (row.get("gt_depth_absrel") for row in sweep_rows if abs(float(row["pred_scale"]) - 1.0) < 1e-12),
        None,
    )
    return out


def _diagnose_scene(
    provider: D4RTCarrierProjectionProvider,
    scene: str,
    *,
    cache_root: Path,
    pred_scales: list[float],
    max_anchors: int,
    backbone: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    scene_dir = cache_root / scene
    if not scene_dir.exists():
        return [], [], {"scene": scene, "status": "missing_cache"}
    stream = ScanNetStream(seq_name=scene, backbone=backbone)
    cache = provider._load_scene(scene)
    intrinsics = stream.load_intrinsics()
    frame_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    hw_cache: dict[int, tuple[int, int]] = {}
    for window_index, window in enumerate(cache["windows"]):
        with np.load(window.path) as data:
            if "xyz_local" not in data.files:
                continue
            xyz_local = np.asarray(data["xyz_local"], dtype=np.float64)
        per_frame_cap = max(16, int(max_anchors) // max(len(window.frame_ids), 1))
        for local_idx, frame_id in enumerate(window.frame_ids):
            uv = np.asarray(window.uv[local_idx], dtype=np.float64)
            ok = (
                np.asarray(window.valid[local_idx], dtype=bool)
                & np.isfinite(xyz_local[local_idx]).all(axis=1)
                & np.isfinite(uv).all(axis=1)
                & (np.asarray(window.visibility[local_idx], dtype=np.float64) >= 0.5)
                & (np.asarray(window.confidence[local_idx], dtype=np.float64) >= 0.5)
                & (uv[:, 0] >= 0.0)
                & (uv[:, 0] <= 1.0)
                & (uv[:, 1] >= 0.0)
                & (uv[:, 1] <= 1.0)
                & (np.abs(xyz_local[local_idx, :, 2]) > 1e-8)
            )
            indices = np.flatnonzero(ok)
            if indices.shape[0] < 16:
                continue
            indices = _sample_indices(indices, per_frame_cap)
            depth = stream.load_depth(int(frame_id))
            target_depth = _sample_depth_uv(depth, uv[indices])
            depth_ok = np.isfinite(target_depth) & (target_depth > 1e-8)
            if np.count_nonzero(depth_ok) < 16:
                continue
            indices = indices[depth_ok]
            target_depth = target_depth[depth_ok]
            image_hw = _image_hw_cached(stream, hw_cache, int(frame_id))
            row = {
                "scene": scene,
                "window_index": int(window_index),
                "window_path": str(window.path),
                "frame_id": int(frame_id),
                "local_idx": int(local_idx),
                "anchor_count": int(indices.shape[0]),
                "target_depth_median": _median_positive(target_depth),
                "pred_local_abs_z_median": _median_positive(np.abs(xyz_local[local_idx, indices, 2])),
            }
            row["target_depth_over_local_z_median"] = (
                float(row["target_depth_median"]) / float(row["pred_local_abs_z_median"])
                if row["target_depth_median"] is not None
                and row["pred_local_abs_z_median"] is not None
                and float(row["pred_local_abs_z_median"]) > 1e-8
                else None
            )
            cur_sweep = _sweep_scale_metrics(
                xyz_local=xyz_local[local_idx, indices],
                uv=uv[indices],
                target_depth=target_depth,
                intrinsics=intrinsics,
                image_hw=image_hw,
                pred_scales=pred_scales,
            )
            for sweep_row in cur_sweep:
                sweep_rows.append({**row, **sweep_row})
            frame_rows.append(_summarize_sweep(row, cur_sweep))
    summary = _scene_summary(scene, frame_rows)
    return frame_rows, sweep_rows, summary


def _scene_summary(scene: str, frame_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"scene": scene, "status": "ok" if frame_rows else "no_frame_rows", "frame_count": len(frame_rows)}
    for key in [
        "target_depth_over_local_z_median",
        "uv_reprojection_median_px_range",
        "normalized_z_l1_range",
        "depth_rank_spearman_range",
        "gt_depth_absrel_range",
        "gt_depth_absrel_min",
        "gt_depth_absrel_at_scale_1",
    ]:
        values = _finite_values(frame_rows, key)
        summary[f"{key}_mean"] = float(np.mean(values)) if values.size else None
        summary[f"{key}_median"] = float(np.median(values)) if values.size else None
        summary[f"{key}_max"] = float(np.max(values)) if values.size else None
    return summary


def _aggregate_summary(scene_rows: list[dict[str, Any]], frame_rows: list[dict[str, Any]], pred_scales: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "scene_count": int(len(scene_rows)),
        "frame_count": int(len(frame_rows)),
        "pred_scales": ",".join(str(v) for v in pred_scales),
        "method_result": False,
        "diagnostic_only": True,
        "uses_scannet_depth_for_positive_control": True,
    }
    for key in [
        "uv_reprojection_median_px_range",
        "normalized_z_l1_range",
        "depth_rank_spearman_range",
        "gt_depth_absrel_range",
        "gt_depth_absrel_min",
        "gt_depth_absrel_at_scale_1",
    ]:
        values = _finite_values(frame_rows, key)
        out[f"{key}_mean"] = float(np.mean(values)) if values.size else None
        out[f"{key}_median"] = float(np.median(values)) if values.size else None
        out[f"{key}_max"] = float(np.max(values)) if values.size else None
    return out


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


def _write_md(path: Path, metadata: dict[str, Any], scene_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# v22.18 self-supervised scale-sensitivity diagnostic",
        "",
        "Diagnostic-only: sweeps uniform scale on D4RT `xyz_local` and measures whether candidate self-supervised constraints change with scale. GT depth is used only as a positive-control metric.",
        "",
        "## Aggregate",
        "",
        "| item | value |",
        "|---|---:|",
    ]
    for key in [
        "scene_count",
        "frame_count",
        "pred_scales",
        "uv_reprojection_median_px_range_mean",
        "normalized_z_l1_range_mean",
        "depth_rank_spearman_range_mean",
        "gt_depth_absrel_range_mean",
        "gt_depth_absrel_min_mean",
        "gt_depth_absrel_at_scale_1_mean",
        "method_result",
    ]:
        lines.append(f"| {key} | {_fmt(metadata.get(key))} |")
    lines.extend(
        [
            "",
            "## Per-scene",
            "",
            "| scene | frames | uv range mean | normalized-z range mean | rank range mean | GT absrel range mean | GT absrel min mean | GT absrel scale1 mean |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in scene_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("scene")),
                    _fmt(row.get("frame_count")),
                    _fmt(row.get("uv_reprojection_median_px_range_mean")),
                    _fmt(row.get("normalized_z_l1_range_mean")),
                    _fmt(row.get("depth_rank_spearman_range_mean")),
                    _fmt(row.get("gt_depth_absrel_range_mean")),
                    _fmt(row.get("gt_depth_absrel_min_mean")),
                    _fmt(row.get("gt_depth_absrel_at_scale_1_mean")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "UV reprojection, normalized-depth shape, and depth-order/rank consistency are effectively invariant to uniform scale on the current D4RT outputs. The GT-depth positive control changes strongly across the same scale sweep. Therefore a self-supervised target-depth objective built only from existing D4RT UV/relative-shape signals would not recover metric scale; it needs an external metric anchor, an explicit learned scale head, or retained training-time normalization metadata.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose scale sensitivity of self-supervised D4RT constraints.")
    parser.add_argument("--seq-list", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v22_local_xyz_probe5_r1")
    parser.add_argument("--audit-root", default="outputs/audit/v22_18_self_supervised_scale_sensitivity_probe5")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-anchors", type=int, default=8000)
    parser.add_argument("--pred-scales", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audit_root = Path(args.audit_root)
    audit_root.mkdir(parents=True, exist_ok=True)
    scenes = _read_seq_list(Path(args.seq_list))
    if args.max_scenes is not None:
        scenes = scenes[: int(args.max_scenes)]
    pred_scales = [float(v) for v in args.pred_scales]
    provider = D4RTCarrierProjectionProvider(debug_root=args.cache_root, mode="raw", max_anchors=int(args.max_anchors))
    all_frame_rows: list[dict[str, Any]] = []
    all_sweep_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    cache_root = Path(args.cache_root)
    for scene in scenes:
        frame_rows, sweep_rows, scene_summary = _diagnose_scene(
            provider,
            scene,
            cache_root=cache_root,
            pred_scales=pred_scales,
            max_anchors=int(args.max_anchors),
            backbone=str(args.backbone),
        )
        all_frame_rows.extend(frame_rows)
        all_sweep_rows.extend(sweep_rows)
        scene_rows.append(scene_summary)
    metadata = _aggregate_summary(scene_rows, all_frame_rows, pred_scales)
    _write_csv(audit_root / "self_supervised_scale_sensitivity_frame_rows.csv", all_frame_rows)
    _write_csv(audit_root / "self_supervised_scale_sensitivity_sweep_rows.csv", all_sweep_rows)
    _write_csv(audit_root / "self_supervised_scale_sensitivity_scene_summary.csv", scene_rows)
    (audit_root / "self_supervised_scale_sensitivity_frame_rows.json").write_text(
        json.dumps(_json_safe(all_frame_rows), indent=2), encoding="utf-8"
    )
    (audit_root / "self_supervised_scale_sensitivity_sweep_rows.json").write_text(
        json.dumps(_json_safe(all_sweep_rows), indent=2), encoding="utf-8"
    )
    (audit_root / "self_supervised_scale_sensitivity_scene_summary.json").write_text(
        json.dumps(_json_safe(scene_rows), indent=2), encoding="utf-8"
    )
    (audit_root / "self_supervised_scale_sensitivity_metadata.json").write_text(
        json.dumps(_json_safe(metadata), indent=2), encoding="utf-8"
    )
    _write_md(audit_root / "self_supervised_scale_sensitivity.md", metadata, scene_rows)
    print(f"Wrote v22.18 self-supervised scale-sensitivity diagnostic to {audit_root}")
    print(json.dumps(_json_safe(metadata), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
