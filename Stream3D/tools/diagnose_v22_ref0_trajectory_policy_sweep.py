from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

STREAM3D_ROOT = Path(__file__).resolve().parents[1]
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from geometry_provider.d4rt_carrier_provider import D4RTCarrierProjectionProvider
from stream4d.scannet_stream import ScanNetStream
from tools.diagnose_v22_ref0_trajectory_scale import (
    _finite_values,
    _relative_ref_to_target,
    _rotation_error_deg,
    _translation_direction_error_deg,
)
from tools.run_v22_direct_reconstruction_benchmark import (
    _fit_ref0_pose_scale,
    _fit_rigid_no_scale,
    _json_safe,
    _read_seq_list,
    _sample_indices,
)


@dataclass(frozen=True)
class AnchorPolicy:
    name: str
    min_visibility: float = 0.5
    min_confidence: float = 0.5
    source_mode: str = "all"
    residual_trim_percentile: float | None = None


POLICIES: tuple[AnchorPolicy, ...] = (
    AnchorPolicy("vc05_all"),
    AnchorPolicy("vc07_all", min_visibility=0.7, min_confidence=0.7),
    AnchorPolicy("vc09_all", min_visibility=0.9, min_confidence=0.9),
    AnchorPolicy("vc05_ref_source", source_mode="ref_source"),
    AnchorPolicy("vc05_target_source", source_mode="target_source"),
    AnchorPolicy("vc05_nonref_source", source_mode="nonref_source"),
    AnchorPolicy("vc05_trim90", residual_trim_percentile=90.0),
    AnchorPolicy("vc05_trim80", residual_trim_percentile=80.0),
)


def _apply_source_policy(base_ok: np.ndarray, src_frame: np.ndarray | None, local_idx: int, policy: AnchorPolicy) -> np.ndarray:
    ok = np.asarray(base_ok, dtype=bool).copy()
    if policy.source_mode == "all":
        return ok
    if src_frame is None:
        return np.zeros_like(ok, dtype=bool)
    src = np.asarray(src_frame).reshape(-1)
    if src.shape[0] != ok.shape[0]:
        return np.zeros_like(ok, dtype=bool)
    if policy.source_mode == "ref_source":
        ok &= src == 0
    elif policy.source_mode == "target_source":
        ok &= src == int(local_idx)
    elif policy.source_mode == "nonref_source":
        ok &= src != 0
    else:
        raise ValueError(f"unsupported source_mode: {policy.source_mode}")
    return ok


def _fit_policy_motion(source: np.ndarray, target: np.ndarray, policy: AnchorPolicy) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]:
    rot, trans, residual = _fit_rigid_no_scale(source, target)
    used_trim = False
    trim = policy.residual_trim_percentile
    if trim is not None and 0.0 < float(trim) < 100.0 and residual.size >= 8:
        keep = residual <= float(np.percentile(residual, float(trim)))
        if int(np.count_nonzero(keep)) >= 4 and int(np.count_nonzero(keep)) < residual.size:
            rot, trans, residual = _fit_rigid_no_scale(source[keep], target[keep])
            used_trim = True
            return rot, trans, residual, int(np.count_nonzero(keep)), used_trim
    return rot, trans, residual, int(source.shape[0]), used_trim


def _policy_scene_summary(scene: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = _finite_values(rows, "trajectory_scale_ratio")
    eval_scales = _finite_values(rows, "eval_ref0_depth_scale")
    anchors = _finite_values(rows, "anchor_count")
    residuals = _finite_values(rows, "rigid_residual_p90")
    rot_err = _finite_values(rows, "rot_err_ref_to_target_deg")
    trans_dir = _finite_values(rows, "trans_dir_err_ref_to_target_deg")
    eval_scale = float(np.median(eval_scales)) if eval_scales.size else None
    ratio_median = float(np.median(ratios)) if ratios.size else None
    out: dict[str, Any] = {
        "scene": scene,
        "policy": rows[0]["policy"] if rows else "NA",
        "status": "ok" if ratios.size else "no_frame_rows",
        "frame_count": int(ratios.size),
        "eval_ref0_depth_scale": eval_scale,
        "ratio_median": ratio_median,
        "ratio_mean": float(np.mean(ratios)) if ratios.size else None,
        "ratio_std": float(np.std(ratios)) if ratios.size else None,
        "ratio_min": float(np.min(ratios)) if ratios.size else None,
        "ratio_max": float(np.max(ratios)) if ratios.size else None,
        "anchor_count_median": float(np.median(anchors)) if anchors.size else None,
        "rigid_residual_p90_median": float(np.median(residuals)) if residuals.size else None,
        "rot_err_ref_to_target_median_deg": float(np.median(rot_err)) if rot_err.size else None,
        "trans_dir_err_ref_to_target_median_deg": float(np.median(trans_dir)) if trans_dir.size else None,
    }
    if eval_scale is not None and ratio_median is not None and np.isfinite(eval_scale) and eval_scale > 1e-8:
        out["ratio_median_abs_rel_vs_eval_scale"] = float(abs(ratio_median - eval_scale) / eval_scale)
    else:
        out["ratio_median_abs_rel_vs_eval_scale"] = None
    return out


def _policy_error_rows(scene_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    policies = sorted({str(row.get("policy")) for row in scene_summaries})
    out: list[dict[str, Any]] = []
    for policy in policies:
        rows = [row for row in scene_summaries if row.get("policy") == policy]
        errors = _finite_values(rows, "ratio_median_abs_rel_vs_eval_scale")
        ratios = _finite_values(rows, "ratio_median")
        frames = _finite_values(rows, "frame_count")
        anchors = _finite_values(rows, "anchor_count_median")
        if errors.size == 0:
            continue
        out.append(
            {
                "policy": policy,
                "scene_count": int(errors.size),
                "mean_abs_rel_vs_eval_scale": float(np.mean(errors)),
                "median_abs_rel_vs_eval_scale": float(np.median(errors)),
                "max_abs_rel_vs_eval_scale": float(np.max(errors)),
                "mean_ratio_median": float(np.mean(ratios)) if ratios.size else None,
                "mean_frame_count": float(np.mean(frames)) if frames.size else None,
                "mean_anchor_count_median": float(np.mean(anchors)) if anchors.size else None,
            }
        )
    out.sort(key=lambda row: (float(row["mean_abs_rel_vs_eval_scale"]), float(row["max_abs_rel_vs_eval_scale"])))
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
    if not window.frame_ids:
        return [], {"status": "no_frames"}
    pose0 = stream.load_pose(int(window.frame_ids[0]))
    if not np.isfinite(pose0).all():
        return [], {"status": "invalid_pose0"}
    _, eval_diag = _fit_ref0_pose_scale(
        stream,
        window,
        robust_trim_percentile=float(robust_trim_percentile),
        max_anchors=int(max_anchors),
    )
    eval_scale = eval_diag.get("ref0_pose_scale")
    with np.load(window.path) as data:
        if "xyz_local" not in data.files:
            return [], {"status": "missing_xyz_local", **eval_diag}
        xyz_local = np.asarray(data["xyz_local"], dtype=np.float64)
        src_frame = np.asarray(data["src_frame"], dtype=np.int64) if "src_frame" in data.files else None
    if xyz_local.shape != np.asarray(window.xyz).shape:
        return [], {"status": "shape_mismatch", **eval_diag}

    frame_rows: list[dict[str, Any]] = []
    per_frame_cap = max(4, int(max_anchors) // max(len(window.frame_ids) - 1, 1))
    for local_idx, frame_id in enumerate(window.frame_ids):
        if local_idx == 0:
            continue
        pose = stream.load_pose(int(frame_id))
        if not np.isfinite(pose).all():
            continue
        base_ok = (
            np.asarray(window.valid[local_idx], dtype=bool)
            & np.isfinite(window.xyz[local_idx]).all(axis=1)
            & np.isfinite(xyz_local[local_idx]).all(axis=1)
            & np.isfinite(window.uv[local_idx]).all(axis=1)
            & (window.uv[local_idx, :, 0] >= 0.0)
            & (window.uv[local_idx, :, 0] <= 1.0)
            & (window.uv[local_idx, :, 1] >= 0.0)
            & (window.uv[local_idx, :, 1] <= 1.0)
        )
        rel_ref_to_target = _relative_ref_to_target(pose0, pose)
        pose_len = float(np.linalg.norm(rel_ref_to_target[:3, 3]))
        for policy in POLICIES:
            ok = (
                base_ok
                & (np.asarray(window.visibility[local_idx], dtype=np.float64) >= float(policy.min_visibility))
                & (np.asarray(window.confidence[local_idx], dtype=np.float64) >= float(policy.min_confidence))
            )
            ok = _apply_source_policy(ok, src_frame, local_idx, policy)
            indices = np.flatnonzero(ok)
            if indices.shape[0] < 4:
                continue
            indices = _sample_indices(indices, per_frame_cap)
            source = np.asarray(window.xyz[local_idx, indices], dtype=np.float64)
            target = np.asarray(xyz_local[local_idx, indices], dtype=np.float64)
            try:
                rot, trans, residual, anchor_count, trim_used = _fit_policy_motion(source, target, policy)
            except Exception:
                continue
            d4rt_len = float(np.linalg.norm(trans))
            ratio = float(pose_len / d4rt_len) if d4rt_len > 1e-8 and pose_len > 1e-8 else None
            absrel = None
            if ratio is not None and eval_scale is not None and np.isfinite(float(eval_scale)) and float(eval_scale) > 1e-8:
                absrel = float(abs(ratio - float(eval_scale)) / float(eval_scale))
            frame_rows.append(
                {
                    "scene": scene,
                    "window_index": int(window_index),
                    "window_path": str(window.path),
                    "policy": policy.name,
                    "source_mode": policy.source_mode,
                    "min_visibility": float(policy.min_visibility),
                    "min_confidence": float(policy.min_confidence),
                    "residual_trim_percentile": policy.residual_trim_percentile,
                    "trim_used": bool(trim_used),
                    "ref_frame": int(window.frame_ids[0]),
                    "frame_id": int(frame_id),
                    "local_idx": int(local_idx),
                    "anchor_count": int(anchor_count),
                    "pretrim_anchor_count": int(indices.shape[0]),
                    "d4rt_translation_norm": d4rt_len,
                    "pose_translation_norm": pose_len,
                    "trajectory_scale_ratio": ratio,
                    "eval_ref0_depth_scale": float(eval_scale) if eval_scale is not None and np.isfinite(float(eval_scale)) else None,
                    "ratio_abs_rel_vs_eval_scale": absrel,
                    "rot_err_ref_to_target_deg": _rotation_error_deg(rot, rel_ref_to_target[:3, :3]),
                    "trans_dir_err_ref_to_target_deg": _translation_direction_error_deg(trans, rel_ref_to_target[:3, 3]),
                    "rigid_residual_median": float(np.median(residual)),
                    "rigid_residual_p90": float(np.percentile(residual, 90)),
                }
            )
    return frame_rows, {"status": "ok", **eval_diag}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(path: Path, scene_summaries: list[dict[str, Any]], policy_errors: list[dict[str, Any]]) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(float(value)):
                return "NA"
            return f"{float(value):.6f}"
        return str(value)

    lines: list[str] = [
        "# v22.11 ref/local trajectory anchor-policy sweep",
        "",
        "Diagnostic-only: tests whether R27 trajectory-ratio drift is explained by carrier anchor policy, confidence threshold, source-frame selection, or residual trimming.",
        "",
        "## Policy Error",
        "",
        "| policy | mean absrel vs R23 scale | median absrel | max absrel | mean frames | mean anchors |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in policy_errors:
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row.get("policy")),
                    fmt(row.get("mean_abs_rel_vs_eval_scale")),
                    fmt(row.get("median_abs_rel_vs_eval_scale")),
                    fmt(row.get("max_abs_rel_vs_eval_scale")),
                    fmt(row.get("mean_frame_count")),
                    fmt(row.get("mean_anchor_count_median")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Scene Policy Summary",
            "",
            "| scene | policy | eval scale | ratio median | absrel | ratio std | anchors med | rot err med | trans dir err med | residual p90 med | frames |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(scene_summaries, key=lambda item: (str(item.get("scene")), str(item.get("policy")))):
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row.get("scene")),
                    fmt(row.get("policy")),
                    fmt(row.get("eval_ref0_depth_scale")),
                    fmt(row.get("ratio_median")),
                    fmt(row.get("ratio_median_abs_rel_vs_eval_scale")),
                    fmt(row.get("ratio_std")),
                    fmt(row.get("anchor_count_median")),
                    fmt(row.get("rot_err_ref_to_target_median_deg")),
                    fmt(row.get("trans_dir_err_ref_to_target_median_deg")),
                    fmt(row.get("rigid_residual_p90_median")),
                    fmt(row.get("frame_count")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose v22 ref/local trajectory scale sensitivity to anchor policies.")
    parser.add_argument("--seq-list", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v22_local_xyz_probe5_r1")
    parser.add_argument("--audit-root", default="outputs/audit/v22_11_ref0_trajectory_policy_sweep_probe5")
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

    provider = D4RTCarrierProjectionProvider(debug_root=args.cache_root, mode="raw", max_anchors=int(args.max_anchors))
    frame_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    scene_summaries: list[dict[str, Any]] = []
    for scene in scenes:
        if not (Path(args.cache_root) / scene).exists():
            for policy in POLICIES:
                scene_summaries.append({"scene": scene, "policy": policy.name, "status": "missing_cache"})
            continue
        stream = ScanNetStream(seq_name=scene, backbone=args.backbone)
        cache = provider._load_scene(scene)
        windows = list(cache["windows"])
        if args.max_windows_per_scene is not None:
            windows = windows[: int(args.max_windows_per_scene)]
        scene_frame_rows: list[dict[str, Any]] = []
        for window_index, window in enumerate(windows):
            rows, diag = _diagnose_window(
                stream,
                window,
                scene=scene,
                window_index=window_index,
                max_anchors=int(args.max_anchors),
                robust_trim_percentile=float(args.robust_trim_percentile),
            )
            eval_scale = diag.get("ref0_pose_scale")
            window_rows.append(
                {
                    "scene": scene,
                    "window_index": int(window_index),
                    "window_path": str(window.path),
                    "ref_frame": int(window.frame_ids[0]) if window.frame_ids else -1,
                    "window_frame_count": int(len(window.frame_ids)),
                    "frame_row_count": int(len(rows)),
                    "eval_ref0_depth_scale": float(eval_scale) if eval_scale is not None and np.isfinite(float(eval_scale)) else None,
                    **diag,
                }
            )
            frame_rows.extend(rows)
            scene_frame_rows.extend(rows)
        for policy in POLICIES:
            policy_rows = [row for row in scene_frame_rows if row.get("policy") == policy.name]
            scene_summaries.append(_policy_scene_summary(scene, policy_rows))

    policy_errors = _policy_error_rows(scene_summaries)
    _write_csv(audit_root / "ref0_trajectory_policy_frame_rows.csv", frame_rows)
    _write_csv(audit_root / "ref0_trajectory_policy_window_rows.csv", window_rows)
    _write_csv(audit_root / "ref0_trajectory_policy_scene_summary.csv", scene_summaries)
    _write_csv(audit_root / "ref0_trajectory_policy_errors.csv", policy_errors)
    (audit_root / "ref0_trajectory_policy_frame_rows.json").write_text(json.dumps(_json_safe(frame_rows), indent=2), encoding="utf-8")
    (audit_root / "ref0_trajectory_policy_window_rows.json").write_text(json.dumps(_json_safe(window_rows), indent=2), encoding="utf-8")
    (audit_root / "ref0_trajectory_policy_scene_summary.json").write_text(json.dumps(_json_safe(scene_summaries), indent=2), encoding="utf-8")
    (audit_root / "ref0_trajectory_policy_errors.json").write_text(json.dumps(_json_safe(policy_errors), indent=2), encoding="utf-8")
    _write_md(audit_root / "ref0_trajectory_policy_sweep.md", scene_summaries, policy_errors)
    print(f"Wrote v22.11 trajectory policy sweep to {audit_root}")


if __name__ == "__main__":
    main()
