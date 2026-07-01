from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from geometry_provider.d4rt_carrier_provider import D4RTCarrierProjectionProvider, _apply_fit
from stream4d.scannet_stream import ScanNetStream
from tools.v64r2_visible_support_utils import (
    frame_ids_from_debug_root,
    json_safe,
    point_metrics,
    read_seq_list,
    scene_points_from_stream,
    visible_support_point_ids,
)


VARIANTS: dict[str, dict[str, Any]] = {
    "R4_SCENE_EVAL_SIM3": {
        "debug_root": "outputs/stream4d_debug_full_32f_ioc075_fixmem",
        "mode": "eval_sim3",
        "label": "R4 cache, all windows, eval-Sim3",
        "chunk_scale_policy": "single_or_available_windows_eval_sim3_no_self_stitch",
    },
    "D5_SCALE_STITCH_EVAL_SIM3": {
        "debug_root": "outputs/stream4d_debug_v21_3_occupancy_d5_warmstart64_probe5_r1",
        "mode": "self_stitched_scale_normalized_eval_sim3",
        "label": "D5 cache, scale-normalized self-stitch, then eval-Sim3",
        "chunk_scale_policy": "chunk_scale_first_self_stitch_then_eval_sim3",
    },
    "D5_SCALE_STITCH_EVAL_SIM3_DENSITY": {
        "debug_root": "outputs/stream4d_debug_v21_3_occupancy_d5_warmstart64_probe5_r1",
        "mode": "self_stitched_scale_normalized_eval_sim3_density",
        "label": "D5 cache, scale-normalized self-stitch, then eval-Sim3 density mode",
        "chunk_scale_policy": "chunk_scale_first_self_stitch_then_eval_sim3",
    },
}


def _sample_rows(points: np.ndarray, max_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] <= int(max_points):
        return points
    keep = np.linspace(0, points.shape[0] - 1, num=int(max_points), dtype=np.int64)
    return points[keep]


def _collect_pred_scene_points(cache: dict[str, Any], *, min_visibility: float, min_confidence: float, max_points_per_frame: int) -> tuple[np.ndarray, dict[str, Any]]:
    parts: list[np.ndarray] = []
    query_slots = 0
    kept_points = 0
    for window in cache["windows"]:
        for local_idx, _frame_id in enumerate(window.frame_ids):
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
            query_slots += int(ok.shape[0])
            idx = np.flatnonzero(ok)
            if idx.size == 0:
                continue
            if idx.size > int(max_points_per_frame):
                keep = np.linspace(0, idx.size - 1, num=int(max_points_per_frame), dtype=np.int64)
                idx = idx[keep]
            pts = xyz[idx].astype(np.float32)
            kept_points += int(pts.shape[0])
            parts.append(pts)
    if not parts:
        return np.empty((0, 3), dtype=np.float32), {
            "raw_query_slots": int(query_slots),
            "sampled_pred_points_before_scene_cap": 0,
        }
    return np.concatenate(parts, axis=0).astype(np.float32), {
        "raw_query_slots": int(query_slots),
        "sampled_pred_points_before_scene_cap": int(kept_points),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scene-level D4RT geometry diagnostic against ScanNet scene mesh/support.")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--variants", default="R4_SCENE_EVAL_SIM3,D5_SCALE_STITCH_EVAL_SIM3,D5_SCALE_STITCH_EVAL_SIM3_DENSITY")
    parser.add_argument("--audit-root", default="outputs/audit/v64r2_scene_level_d4rt_geometry")
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--max-anchors", type=int, default=8000)
    parser.add_argument("--robust-trim-percentile", type=float, default=90.0)
    parser.add_argument("--nn-radius", type=float, default=0.05)
    parser.add_argument("--density-alpha", type=float, default=2.0)
    parser.add_argument("--max-pred-points-per-frame", type=int, default=1500)
    parser.add_argument("--max-points-per-scene", type=int, default=80000)
    parser.add_argument("--support-pixel-stride", type=int, default=2)
    parser.add_argument("--support-mask-positive-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    audit_root = Path(args.audit_root)
    audit_root.mkdir(parents=True, exist_ok=True)
    scenes = read_seq_list(Path(args.seq_list))
    requested = [item.strip() for item in args.variants.split(",") if item.strip()]
    scene_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for variant in requested:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant}; choices={sorted(VARIANTS)}")
        spec = VARIANTS[variant]
        provider = D4RTCarrierProjectionProvider(
            debug_root=spec["debug_root"],
            mode=spec["mode"],
            nn_radius=float(args.nn_radius),
            min_visibility=float(args.min_visibility),
            min_confidence=float(args.min_confidence),
            max_anchors=int(args.max_anchors),
            robust_trim_percentile=float(args.robust_trim_percentile),
            density_alpha=float(args.density_alpha),
        )
        ok_rows: list[dict[str, Any]] = []
        for scene in scenes:
            row: dict[str, Any] = {
                "variant": variant,
                "scene": scene,
                "label": spec["label"],
                "debug_root": spec["debug_root"],
                "provider_mode": spec["mode"],
                "chunk_scale_policy": spec["chunk_scale_policy"],
            }
            try:
                stream = ScanNetStream(seq_name=scene, backbone="Cropformer")
                scene_points = scene_points_from_stream(stream)
                cache = provider._load_scene(scene)
                pred_points, pred_diag = _collect_pred_scene_points(
                    cache,
                    min_visibility=float(args.min_visibility),
                    min_confidence=float(args.min_confidence),
                    max_points_per_frame=int(args.max_pred_points_per_frame),
                )
                pred_points = _sample_rows(pred_points, int(args.max_points_per_scene))
                frame_ids = frame_ids_from_debug_root(spec["debug_root"], scene)
                support_ids, support_diag = visible_support_point_ids(
                    stream,
                    scene_points,
                    frame_ids,
                    pixel_stride=int(args.support_pixel_stride),
                    nn_radius=float(args.nn_radius),
                    mask_positive_only=bool(args.support_mask_positive_only),
                )
                full_gt = _sample_rows(scene_points, int(args.max_points_per_scene))
                support_gt = _sample_rows(scene_points[support_ids], int(args.max_points_per_scene)) if support_ids.size else np.empty((0, 3), dtype=np.float32)
                full_metrics = point_metrics(pred_points, full_gt)
                support_metrics = point_metrics(pred_points, support_gt)
                row.update(pred_diag)
                row.update({f"support_{key}": value for key, value in support_diag.items()})
                row.update({f"full_mesh_{key}": value for key, value in full_metrics.items() if not str(key).startswith("_")})
                row.update({f"used_frame_support_{key}": value for key, value in support_metrics.items() if not str(key).startswith("_")})
                row["pred_point_count_after_scene_cap"] = int(pred_points.shape[0])
                row["status"] = "ok" if full_metrics.get("status") == "ok" or support_metrics.get("status") == "ok" else "empty"
                row.update({f"scene_fit_{key}": value for key, value in dict(cache.get("scene_fit") or {}).items()})
                row.update({f"stitch_{key}": value for key, value in dict(cache.get("stitch_diag") or {}).items()})
                row.update({f"anchor_{key}": value for key, value in dict(cache.get("anchor_diag") or {}).items()})
            except Exception as exc:
                row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            scene_rows.append(row)
            if row.get("status") == "ok":
                ok_rows.append(row)
        summary = {
            "variant": variant,
            "label": spec["label"],
            "debug_root": spec["debug_root"],
            "provider_mode": spec["mode"],
            "chunk_scale_policy": spec["chunk_scale_policy"],
            "status": "ok" if ok_rows else "missing_or_failed",
            "num_scenes": int(len(ok_rows)),
        }
        for key in sorted({key for row in ok_rows for key, value in row.items() if isinstance(value, (int, float, np.generic)) and not isinstance(value, bool)}):
            vals = [float(row[key]) for row in ok_rows if row.get(key) is not None and np.isfinite(float(row[key]))]
            if vals:
                summary[key] = float(np.mean(vals))
        summary_rows.append(summary)

    payload = {
        "args": vars(args),
        "scene_rows": scene_rows,
        "summary": summary_rows,
        "metric_note": (
            "full_mesh_* compares all predicted D4RT points to the full ScanNet mesh; "
            "used_frame_support_* compares to mesh vertices visible from the actual frames in the D4RT cache via ScanNet depth/pose/mask."
        ),
        "is_method_result": False,
        "is_diagnostic_only": True,
    }
    (audit_root / "scene_level_d4rt_geometry_summary.json").write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(audit_root / "scene_level_d4rt_geometry_scene_rows.csv", scene_rows)
    _write_csv(audit_root / "scene_level_d4rt_geometry_summary.csv", summary_rows)
    print(json.dumps(json_safe({"summary": summary_rows}), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
