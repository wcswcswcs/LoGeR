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
from tools.run_v22_direct_reconstruction_benchmark import (
    _fit_ref0_pose_scale,
    _fit_rigid_no_scale,
    _json_safe,
    _read_seq_list,
    _sample_indices,
)


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    rot = np.asarray(rotation, dtype=np.float64)
    value = (float(np.trace(rot)) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def _rotation_error_deg(pred: np.ndarray, target: np.ndarray) -> float:
    pred_rot = np.asarray(pred, dtype=np.float64)
    target_rot = np.asarray(target, dtype=np.float64)
    return _rotation_angle_deg(target_rot.T @ pred_rot)


def _translation_direction_error_deg(pred: np.ndarray, target: np.ndarray) -> float | None:
    pred_vec = np.asarray(pred, dtype=np.float64).reshape(3)
    target_vec = np.asarray(target, dtype=np.float64).reshape(3)
    pred_norm = float(np.linalg.norm(pred_vec))
    target_norm = float(np.linalg.norm(target_vec))
    if pred_norm <= 1e-8 or target_norm <= 1e-8:
        return None
    value = float(np.dot(pred_vec, target_vec) / (pred_norm * target_norm))
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def _relative_ref_to_target(pose0: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(pose, dtype=np.float64)) @ np.asarray(pose0, dtype=np.float64)


def _relative_target_to_ref(pose0: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(pose0, dtype=np.float64)) @ np.asarray(pose, dtype=np.float64)


def _finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            values.append(numeric)
    return np.asarray(values, dtype=np.float64)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[ok]
    weights = weights[ok]
    if values.size == 0:
        return None
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cutoff = 0.5 * float(cumulative[-1])
    return float(values[int(np.searchsorted(cumulative, cutoff, side="left"))])


def _candidate_scales(frame_rows: list[dict[str, Any]]) -> dict[str, float | None]:
    ratios = _finite_values(frame_rows, "trajectory_scale_ratio")
    residual = _finite_values(frame_rows, "rigid_residual_p90")
    trans_dir = _finite_values(frame_rows, "trans_dir_err_ref_to_target_deg")
    out: dict[str, float | None] = {
        "ratio_min": None,
        "ratio_q10": None,
        "ratio_q25": None,
        "ratio_median": None,
        "ratio_mean": None,
        "ratio_q75": None,
        "ratio_low_residual_median": None,
        "ratio_low_direction_median": None,
        "ratio_residual_weighted_median": None,
    }
    if ratios.size == 0:
        return out
    out.update(
        {
            "ratio_min": float(np.min(ratios)),
            "ratio_q10": float(np.percentile(ratios, 10)),
            "ratio_q25": float(np.percentile(ratios, 25)),
            "ratio_median": float(np.median(ratios)),
            "ratio_mean": float(np.mean(ratios)),
            "ratio_q75": float(np.percentile(ratios, 75)),
        }
    )
    if residual.shape == ratios.shape and residual.size > 0:
        keep = residual <= float(np.median(residual))
        if np.any(keep):
            out["ratio_low_residual_median"] = float(np.median(ratios[keep]))
        out["ratio_residual_weighted_median"] = _weighted_median(ratios, 1.0 / np.maximum(residual, 1e-6))
    if trans_dir.shape == ratios.shape and trans_dir.size > 0:
        keep = trans_dir <= float(np.median(trans_dir))
        if np.any(keep):
            out["ratio_low_direction_median"] = float(np.median(ratios[keep]))
    return out


def _diagnose_window(
    stream: ScanNetStream,
    window: Any,
    *,
    scene: str,
    window_index: int,
    max_anchors: int,
    robust_trim_percentile: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame_rows: list[dict[str, Any]] = []
    if not window.frame_ids:
        return frame_rows, {"status": "no_frames"}
    pose0 = stream.load_pose(int(window.frame_ids[0]))
    if not np.isfinite(pose0).all():
        return frame_rows, {"status": "invalid_pose0"}
    _, eval_diag = _fit_ref0_pose_scale(
        stream,
        window,
        robust_trim_percentile=float(robust_trim_percentile),
        max_anchors=int(max_anchors),
    )
    eval_scale = eval_diag.get("ref0_pose_scale")
    with np.load(window.path) as data:
        if "xyz_local" not in data.files:
            return frame_rows, {"status": "missing_xyz_local", **eval_diag}
        xyz_local = np.asarray(data["xyz_local"], dtype=np.float64)
    if xyz_local.shape != np.asarray(window.xyz).shape:
        return frame_rows, {"status": "shape_mismatch", **eval_diag}

    per_frame_cap = max(4, int(max_anchors) // max(len(window.frame_ids) - 1, 1))
    for local_idx, frame_id in enumerate(window.frame_ids):
        if local_idx == 0:
            continue
        pose = stream.load_pose(int(frame_id))
        if not np.isfinite(pose).all():
            continue
        ok = (
            np.asarray(window.valid[local_idx], dtype=bool)
            & np.isfinite(window.xyz[local_idx]).all(axis=1)
            & np.isfinite(xyz_local[local_idx]).all(axis=1)
            & (np.asarray(window.visibility[local_idx], dtype=np.float64) >= 0.5)
            & (np.asarray(window.confidence[local_idx], dtype=np.float64) >= 0.5)
            & np.isfinite(window.uv[local_idx]).all(axis=1)
            & (window.uv[local_idx, :, 0] >= 0.0)
            & (window.uv[local_idx, :, 0] <= 1.0)
            & (window.uv[local_idx, :, 1] >= 0.0)
            & (window.uv[local_idx, :, 1] <= 1.0)
        )
        indices = np.flatnonzero(ok)
        if indices.shape[0] < 4:
            continue
        indices = _sample_indices(indices, per_frame_cap)
        try:
            rot, trans, residual = _fit_rigid_no_scale(
                np.asarray(window.xyz[local_idx, indices], dtype=np.float64),
                np.asarray(xyz_local[local_idx, indices], dtype=np.float64),
            )
        except Exception:
            continue

        rel_ref_to_target = _relative_ref_to_target(pose0, pose)
        rel_target_to_ref = _relative_target_to_ref(pose0, pose)
        d4rt_len = float(np.linalg.norm(trans))
        pose_len = float(np.linalg.norm(rel_ref_to_target[:3, 3]))
        ratio = float(pose_len / d4rt_len) if d4rt_len > 1e-8 and pose_len > 1e-8 else None
        abs_rel_vs_eval = None
        if ratio is not None and eval_scale is not None and np.isfinite(float(eval_scale)) and float(eval_scale) > 1e-8:
            abs_rel_vs_eval = float(abs(ratio - float(eval_scale)) / float(eval_scale))

        frame_rows.append(
            {
                "scene": scene,
                "window_index": int(window_index),
                "window_path": str(window.path),
                "ref_frame": int(window.frame_ids[0]),
                "frame_id": int(frame_id),
                "local_idx": int(local_idx),
                "anchor_count": int(indices.shape[0]),
                "d4rt_translation_norm": d4rt_len,
                "pose_translation_norm": pose_len,
                "trajectory_scale_ratio": ratio,
                "eval_ref0_depth_scale": float(eval_scale) if eval_scale is not None and np.isfinite(float(eval_scale)) else None,
                "ratio_abs_rel_vs_eval_scale": abs_rel_vs_eval,
                "d4rt_rotation_angle_deg": _rotation_angle_deg(rot),
                "pose_ref_to_target_rotation_angle_deg": _rotation_angle_deg(rel_ref_to_target[:3, :3]),
                "rot_err_ref_to_target_deg": _rotation_error_deg(rot, rel_ref_to_target[:3, :3]),
                "rot_err_target_to_ref_deg": _rotation_error_deg(rot, rel_target_to_ref[:3, :3]),
                "trans_dir_err_ref_to_target_deg": _translation_direction_error_deg(trans, rel_ref_to_target[:3, 3]),
                "trans_dir_err_target_to_ref_deg": _translation_direction_error_deg(trans, rel_target_to_ref[:3, 3]),
                "rigid_residual_median": float(np.median(residual)),
                "rigid_residual_p90": float(np.percentile(residual, 90)),
            }
        )
    return frame_rows, {"status": "ok", **eval_diag}


def _scene_summary(scene: str, window_rows: list[dict[str, Any]], frame_rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = _candidate_scales(frame_rows)
    eval_scale_values = _finite_values(window_rows, "eval_ref0_depth_scale")
    eval_scale = float(np.median(eval_scale_values)) if eval_scale_values.size else None
    ratios = _finite_values(frame_rows, "trajectory_scale_ratio")
    residual_p90 = _finite_values(frame_rows, "rigid_residual_p90")
    rot_err = _finite_values(frame_rows, "rot_err_ref_to_target_deg")
    trans_dir = _finite_values(frame_rows, "trans_dir_err_ref_to_target_deg")
    summary: dict[str, Any] = {
        "scene": scene,
        "status": "ok" if ratios.size else "no_frame_rows",
        "window_count": int(len(window_rows)),
        "frame_count": int(ratios.size),
        "eval_ref0_depth_scale": eval_scale,
        **candidates,
        "rigid_residual_p90_median": float(np.median(residual_p90)) if residual_p90.size else None,
        "rot_err_ref_to_target_median_deg": float(np.median(rot_err)) if rot_err.size else None,
        "trans_dir_err_ref_to_target_median_deg": float(np.median(trans_dir)) if trans_dir.size else None,
    }
    for key, value in candidates.items():
        if value is None or eval_scale is None or not np.isfinite(float(eval_scale)) or float(eval_scale) <= 1e-8:
            summary[f"{key}_abs_rel_vs_eval_scale"] = None
        else:
            summary[f"{key}_abs_rel_vs_eval_scale"] = float(abs(float(value) - float(eval_scale)) / float(eval_scale))
    return summary


def _candidate_error_rows(scene_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        "ratio_min",
        "ratio_q10",
        "ratio_q25",
        "ratio_median",
        "ratio_mean",
        "ratio_q75",
        "ratio_low_residual_median",
        "ratio_low_direction_median",
        "ratio_residual_weighted_median",
    ]
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        errors = _finite_values(scene_summaries, f"{candidate}_abs_rel_vs_eval_scale")
        values = _finite_values(scene_summaries, candidate)
        if errors.size == 0:
            continue
        rows.append(
            {
                "candidate": candidate,
                "scene_count": int(errors.size),
                "mean_abs_rel_vs_eval_scale": float(np.mean(errors)),
                "median_abs_rel_vs_eval_scale": float(np.median(errors)),
                "max_abs_rel_vs_eval_scale": float(np.max(errors)),
                "mean_scale": float(np.mean(values)) if values.size else None,
            }
        )
    rows.sort(key=lambda row: (float(row["mean_abs_rel_vs_eval_scale"]), float(row["max_abs_rel_vs_eval_scale"])))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(path: Path, scene_summaries: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(float(value)):
                return "NA"
            return f"{float(value):.6f}"
        return str(value)

    lines: list[str] = [
        "# v22.10 ref0 trajectory consistency diagnostic",
        "",
        "Diagnostic-only: compares D4RT ref-to-local rigid motion against ScanNet pose trajectory and R23 eval-only ref0 depth scale. This does not define a method result.",
        "",
        "## Per-scene",
        "",
        "| scene | eval scale | median ratio | q25 | q75 | ratio std | ratio absrel | rot err med | trans dir err med | residual p90 med | frames |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scene_summaries:
        ratios = [row.get("ratio_q25"), row.get("ratio_median"), row.get("ratio_q75")]
        ratio_std = "NA"
        if all(value is not None for value in ratios):
            ratio_std = fmt(np.std(np.asarray(ratios, dtype=np.float64)))
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row.get("scene")),
                    fmt(row.get("eval_ref0_depth_scale")),
                    fmt(row.get("ratio_median")),
                    fmt(row.get("ratio_q25")),
                    fmt(row.get("ratio_q75")),
                    ratio_std,
                    fmt(row.get("ratio_median_abs_rel_vs_eval_scale")),
                    fmt(row.get("rot_err_ref_to_target_median_deg")),
                    fmt(row.get("trans_dir_err_ref_to_target_median_deg")),
                    fmt(row.get("rigid_residual_p90_median")),
                    fmt(row.get("frame_count")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Candidate Scale Error",
            "",
            "| candidate | mean absrel vs eval scale | median absrel | max absrel | scene count |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in candidate_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row.get("candidate")),
                    fmt(row.get("mean_abs_rel_vs_eval_scale")),
                    fmt(row.get("median_abs_rel_vs_eval_scale")),
                    fmt(row.get("max_abs_rel_vs_eval_scale")),
                    fmt(row.get("scene_count")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose v22 R27 ref0 trajectory scale stability.")
    parser.add_argument("--seq-list", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v22_local_xyz_probe5_r1")
    parser.add_argument("--audit-root", default="outputs/audit/v22_10_ref0_trajectory_consistency_probe5")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-windows-per-scene", type=int, default=None)
    parser.add_argument("--max-anchors", type=int, default=8000)
    parser.add_argument("--robust-trim-percentile", type=float, default=90.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    audit_root = Path(args.audit_root)
    audit_root.mkdir(parents=True, exist_ok=True)
    scenes = _read_seq_list(Path(args.seq_list))
    if args.max_scenes is not None:
        scenes = scenes[: int(args.max_scenes)]

    provider = D4RTCarrierProjectionProvider(
        debug_root=args.cache_root,
        mode="raw",
        max_anchors=int(args.max_anchors),
    )
    all_frame_rows: list[dict[str, Any]] = []
    all_window_rows: list[dict[str, Any]] = []
    scene_summaries: list[dict[str, Any]] = []
    for scene in scenes:
        scene_dir = Path(args.cache_root) / scene
        if not scene_dir.exists():
            scene_summaries.append({"scene": scene, "status": "missing_cache"})
            continue
        stream = ScanNetStream(seq_name=scene, backbone=args.backbone)
        cache = provider._load_scene(scene)
        windows = list(cache["windows"])
        if args.max_windows_per_scene is not None:
            windows = windows[: int(args.max_windows_per_scene)]
        scene_frame_rows: list[dict[str, Any]] = []
        scene_window_rows: list[dict[str, Any]] = []
        for window_index, window in enumerate(windows):
            frame_rows, window_diag = _diagnose_window(
                stream,
                window,
                scene=scene,
                window_index=window_index,
                max_anchors=int(args.max_anchors),
                robust_trim_percentile=float(args.robust_trim_percentile),
            )
            eval_scale = window_diag.get("ref0_pose_scale")
            window_row = {
                "scene": scene,
                "window_index": int(window_index),
                "window_path": str(window.path),
                "ref_frame": int(window.frame_ids[0]) if window.frame_ids else -1,
                "window_frame_count": int(len(window.frame_ids)),
                "frame_row_count": int(len(frame_rows)),
                "eval_ref0_depth_scale": float(eval_scale) if eval_scale is not None and np.isfinite(float(eval_scale)) else None,
                **window_diag,
            }
            scene_window_rows.append(window_row)
            scene_frame_rows.extend(frame_rows)
        all_window_rows.extend(scene_window_rows)
        all_frame_rows.extend(scene_frame_rows)
        scene_summaries.append(_scene_summary(scene, scene_window_rows, scene_frame_rows))

    candidate_rows = _candidate_error_rows(scene_summaries)
    _write_csv(audit_root / "ref0_trajectory_frame_rows.csv", all_frame_rows)
    _write_csv(audit_root / "ref0_trajectory_window_rows.csv", all_window_rows)
    _write_csv(audit_root / "ref0_trajectory_scene_summary.csv", scene_summaries)
    _write_csv(audit_root / "ref0_trajectory_candidate_scale_errors.csv", candidate_rows)
    (audit_root / "ref0_trajectory_frame_rows.json").write_text(json.dumps(_json_safe(all_frame_rows), indent=2), encoding="utf-8")
    (audit_root / "ref0_trajectory_window_rows.json").write_text(json.dumps(_json_safe(all_window_rows), indent=2), encoding="utf-8")
    (audit_root / "ref0_trajectory_scene_summary.json").write_text(json.dumps(_json_safe(scene_summaries), indent=2), encoding="utf-8")
    (audit_root / "ref0_trajectory_candidate_scale_errors.json").write_text(json.dumps(_json_safe(candidate_rows), indent=2), encoding="utf-8")
    _write_md(audit_root / "ref0_trajectory_consistency.md", scene_summaries, candidate_rows)
    print(f"Wrote v22.10 ref0 trajectory diagnostic to {audit_root}")


if __name__ == "__main__":
    main()
