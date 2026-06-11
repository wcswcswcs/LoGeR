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

from stream4d.scannet_stream import ScanNetStream
from tools.run_v22_direct_reconstruction_benchmark import (
    VARIANTS,
    _camera_space_point_metrics,
    _collect_pred_points,
    _depth_metrics,
    _fmt,
    _instance_coverage,
    _json_safe,
    _load_gt_points,
    _load_windows,
    _point_metrics,
    _read_seq_list,
    _sample_rows,
)


def _scale_fit(fit: dict[str, Any] | None, multiplier: float) -> dict[str, Any] | None:
    if fit is None:
        return None
    out = dict(fit)
    out["scale"] = float(fit["scale"]) * float(multiplier)
    return out


def _relative_scale_error(multiplier: float) -> float:
    return float(abs(float(multiplier) - 1.0))


def _aggregate_numeric(rows: list[dict[str, Any]], *, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {"scene_count": int(len(rows))}
    keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float, np.generic)) and not str(key).startswith("_")
        }
    )
    for key in keys:
        values = np.asarray(
            [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))],
            dtype=np.float64,
        )
        if values.size:
            out[f"{prefix}{key}_mean"] = float(np.mean(values))
            out[f"{prefix}{key}_median"] = float(np.median(values))
            out[f"{prefix}{key}_min"] = float(np.min(values))
            out[f"{prefix}{key}_max"] = float(np.max(values))
    return out


def _find_oracle_summary(summary_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in summary_rows:
        if abs(float(row["scale_multiplier"]) - 1.0) < 1e-12:
            return row
    return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys() if not key.startswith("_")})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def _write_md(path: Path, metadata: dict[str, Any], summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# v22.19 scale-anchor tolerance diagnostic",
        "",
        "Diagnostic-only: starts from R23 `xyz_ref0 + ScanNet ref0 pose + eval-only scale`, then perturbs only the fitted scale. GT depth/pose are used to form the R23 oracle and to evaluate tolerance.",
        "",
        "## Aggregate",
        "",
        "| item | value |",
        "|---|---:|",
    ]
    for key in [
        "scene_count",
        "scale_multipliers",
        "oracle_fscore10_mean",
        "oracle_completeness20_mean",
        "oracle_depth_delta1_mean",
        "method_result",
    ]:
        lines.append(f"| {key} | {_fmt(metadata.get(key))} |")
    lines.extend(
        [
            "",
            "## Scale Sweep",
            "",
            "| multiplier | rel scale error | F@10 mean | F@10 retention | comp@20 mean | outlier@20 mean | depth delta1 mean | depth AbsRel mean |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("scale_multiplier")),
                    _fmt(row.get("relative_scale_error")),
                    _fmt(row.get("fscore@10cm_mean")),
                    _fmt(row.get("fscore@10cm_retention_vs_oracle")),
                    _fmt(row.get("completeness@20cm_mean")),
                    _fmt(row.get("outlier_rate_20cm_mean")),
                    _fmt(row.get("depth_median_delta1_mean")),
                    _fmt(row.get("depth_median_absrel_mean")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "This is not a method result. It quantifies how accurate a future non-GT scale anchor must be to preserve the R23 upper-bound geometry. If modest scale perturbations sharply reduce F-score or depth metrics, weak predictors such as v22.16 D4RT-internal regressors are insufficient even if their mean scale error looks numerically small.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose direct reconstruction tolerance to ref0 scale-anchor errors.")
    parser.add_argument("--seq-list", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--audit-root", default="outputs/audit/v22_19_scale_anchor_tolerance_probe5")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--scale-multipliers", type=float, nargs="+", default=[0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5])
    parser.add_argument("--depth-sample-stride", type=int, default=12)
    parser.add_argument("--max-gt-points-per-scene", type=int, default=60000)
    parser.add_argument("--max-pred-points-per-frame", type=int, default=1500)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--max-anchors", type=int, default=8000)
    parser.add_argument("--robust-trim-percentile", type=float, default=90.0)
    parser.add_argument("--nn-radius", type=float, default=0.05)
    parser.add_argument("--density-alpha", type=float, default=2.0)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-windows-per-scene", type=int, default=None)
    parser.add_argument("--debug-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audit_root = Path(args.audit_root)
    audit_root.mkdir(parents=True, exist_ok=True)
    scenes = _read_seq_list(Path(args.seq_list))
    if args.max_scenes is not None:
        scenes = scenes[: int(args.max_scenes)]
    r23 = next(spec for spec in VARIANTS if spec.name == "R23")
    scene_rows: list[dict[str, Any]] = []
    multipliers = [float(value) for value in args.scale_multipliers]
    for scene in scenes:
        if args.debug_progress:
            print(f"[v22-scale-tolerance] start {scene}", file=sys.stderr, flush=True)
        scene_dir = Path(r23.cache_root) / scene
        if not scene_dir.exists():
            for multiplier in multipliers:
                scene_rows.append(
                    {
                        "scene": scene,
                        "scale_multiplier": float(multiplier),
                        "relative_scale_error": _relative_scale_error(multiplier),
                        "status": "missing_cache",
                    }
                )
            continue
        try:
            stream = ScanNetStream(seq_name=scene, backbone=args.backbone)
            windows, cache = _load_windows(
                r23,
                scene,
                nn_radius=float(args.nn_radius),
                density_alpha=float(args.density_alpha),
                robust_trim_percentile=float(args.robust_trim_percentile),
                max_anchors=int(args.max_anchors),
                max_windows_per_scene=args.max_windows_per_scene,
            )
            base_fits = [None if window.transform is None else dict(window.transform) for window in windows]
            frame_ids = sorted({int(frame_id) for window in windows for frame_id in window.frame_ids})
            gt_points, _, _ = _load_gt_points(
                stream,
                frame_ids,
                depth_sample_stride=int(args.depth_sample_stride),
                max_gt_points_per_scene=int(args.max_gt_points_per_scene),
            )
            gt_points = _sample_rows(gt_points, int(args.max_gt_points_per_scene))
            alignment_diag = cache.get("alignment_diag", {})
            for multiplier in multipliers:
                for window, base_fit in zip(windows, base_fits):
                    window.transform = _scale_fit(base_fit, multiplier)
                row: dict[str, Any] = {
                    "scene": scene,
                    "variant": "R23_scale_tolerance",
                    "scale_multiplier": float(multiplier),
                    "relative_scale_error": _relative_scale_error(multiplier),
                    "provider_mode": r23.provider_mode,
                    "cache_root": r23.cache_root,
                    "diagnostic_only": True,
                    "method_result": False,
                    **alignment_diag,
                }
                pred = _collect_pred_points(
                    windows,
                    stream=stream,
                    point_mode=r23.point_mode,
                    depth_calibration_fit=None,
                    depth_calibration="none",
                    max_points_per_frame=int(args.max_pred_points_per_frame),
                    min_visibility=float(args.min_visibility),
                    min_confidence=float(args.min_confidence),
                )
                pred_points = _sample_rows(pred["points"], int(args.max_gt_points_per_scene))
                metrics = _point_metrics(pred_points, gt_points)
                if metrics.get("status") == "empty":
                    row.update({"status": "empty"})
                else:
                    metrics.pop("_pred_to_gt_dist", None)
                    row.update(metrics)
                    row.update(_depth_metrics(stream, pred, points_are_world=True))
                    row.update(
                        _camera_space_point_metrics(
                            stream,
                            pred,
                            points_are_world=True,
                            depth_sample_stride=int(args.depth_sample_stride),
                            max_gt_points_per_scene=int(args.max_gt_points_per_scene),
                        )
                    )
                    row.update(_instance_coverage(stream, pred))
                    row["status"] = "ok"
                row["num_windows"] = int(len(windows))
                scene_rows.append(row)
        except Exception as exc:
            for multiplier in multipliers:
                scene_rows.append(
                    {
                        "scene": scene,
                        "scale_multiplier": float(multiplier),
                        "relative_scale_error": _relative_scale_error(multiplier),
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    summary_rows: list[dict[str, Any]] = []
    for multiplier in multipliers:
        rows = [
            row
            for row in scene_rows
            if row.get("status") == "ok" and abs(float(row["scale_multiplier"]) - float(multiplier)) < 1e-12
        ]
        summary_rows.append(
            {
                "scale_multiplier": float(multiplier),
                "relative_scale_error": _relative_scale_error(multiplier),
                "diagnostic_only": True,
                "method_result": False,
                **_aggregate_numeric(rows),
            }
        )
    oracle = _find_oracle_summary(summary_rows)
    if oracle is not None:
        oracle_f10 = float(oracle.get("fscore@10cm_mean", 0.0) or 0.0)
        oracle_comp20 = float(oracle.get("completeness@20cm_mean", 0.0) or 0.0)
        oracle_delta1 = float(oracle.get("depth_median_delta1_mean", 0.0) or 0.0)
        for row in summary_rows:
            row["fscore@10cm_retention_vs_oracle"] = (
                float(row.get("fscore@10cm_mean", 0.0)) / oracle_f10 if oracle_f10 > 1e-12 else None
            )
            row["completeness@20cm_retention_vs_oracle"] = (
                float(row.get("completeness@20cm_mean", 0.0)) / oracle_comp20 if oracle_comp20 > 1e-12 else None
            )
            row["depth_delta1_retention_vs_oracle"] = (
                float(row.get("depth_median_delta1_mean", 0.0)) / oracle_delta1 if oracle_delta1 > 1e-12 else None
            )
    metadata = {
        "scene_count": int(len(scenes)),
        "scale_multipliers": ",".join(str(value) for value in multipliers),
        "diagnostic_only": True,
        "method_result": False,
        "uses_gt_depth_or_pose_for_oracle_scale": True,
        "oracle_fscore10_mean": oracle.get("fscore@10cm_mean") if oracle else None,
        "oracle_completeness20_mean": oracle.get("completeness@20cm_mean") if oracle else None,
        "oracle_depth_delta1_mean": oracle.get("depth_median_delta1_mean") if oracle else None,
    }
    _write_csv(audit_root / "scale_anchor_tolerance_scene_rows.csv", scene_rows)
    _write_csv(audit_root / "scale_anchor_tolerance_summary.csv", summary_rows)
    (audit_root / "scale_anchor_tolerance_scene_rows.json").write_text(
        json.dumps(_json_safe(scene_rows), indent=2, sort_keys=True), encoding="utf-8"
    )
    (audit_root / "scale_anchor_tolerance_summary.json").write_text(
        json.dumps(_json_safe(summary_rows), indent=2, sort_keys=True), encoding="utf-8"
    )
    (audit_root / "scale_anchor_tolerance_metadata.json").write_text(
        json.dumps(_json_safe(metadata), indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_md(audit_root / "scale_anchor_tolerance.md", metadata, summary_rows)
    print(f"Wrote v22.19 scale-anchor tolerance diagnostic to {audit_root}")
    print(json.dumps(_json_safe({"metadata": metadata, "summary": summary_rows}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
