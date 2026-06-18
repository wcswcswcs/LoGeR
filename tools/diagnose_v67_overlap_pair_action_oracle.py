#!/usr/bin/env python3
"""Offline action oracle using materialized ACL2 v67 raw overlap point pairs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch

try:
    from diagnose_v67_boundary_sim3_action_oracle import (
        _apply_global_correction,
        _best_mechanism_improvement,
        _rotation_delta_deg,
        _rotation_power,
        _rmse_dist,
        _safe_improvement_ratio,
        _target_chunks,
    )
    from diagnose_v67_offline_scale_controller import (
        DEFAULT_GT,
        _load_kitti_gt,
        _load_postmerge_trajectory,
        _load_trace,
        _metric_result_row,
    )
    from kitti_trajectory_diagnostics import _umeyama_sim3
except ImportError:  # pragma: no cover
    from tools.diagnose_v67_boundary_sim3_action_oracle import (
        _apply_global_correction,
        _best_mechanism_improvement,
        _rotation_delta_deg,
        _rotation_power,
        _rmse_dist,
        _safe_improvement_ratio,
        _target_chunks,
    )
    from tools.diagnose_v67_offline_scale_controller import (
        DEFAULT_GT,
        _load_kitti_gt,
        _load_postmerge_trajectory,
        _load_trace,
        _metric_result_row,
    )
    from tools.kitti_trajectory_diagnostics import _umeyama_sim3


GROUP_LABELS: Mapping[str, Sequence[str]] = {
    "dynamic": ("person", "car", "truck", "bus", "van", "rider", "cyclist", "bicycle", "motorcycle", "animal"),
    "sky_context": ("sky", "cloud", "horizon"),
    "vegetation_farstuff": ("grass", "tree", "vegetation", "plant", "terrain", "mountain"),
    "vertical_static": (
        "building",
        "house",
        "wall",
        "handrail_or_fence",
        "fence",
        "pole",
        "traffic sign",
        "traffic light",
        "billboard_or_bulletin_board",
        "bridge",
    ),
    "ground_static": ("road", "ground", "sidewalk", "crosswalk", "floor"),
    "void_lowtrust": ("void", "unknown", "unlabeled"),
}


def _parse_source(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--source must be LABEL=RUN_DIR")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("--source label must be non-empty")
    return label, Path(path)


def _float(value: Any, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _finite_mean(values: Iterable[Any]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(xs)) if xs else None


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_pair(path: Path) -> Dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected dict payload")
    return obj


def _normalise_label_names(label_names: Any) -> Dict[int, str]:
    if isinstance(label_names, Mapping):
        return {int(k): str(v) for k, v in label_names.items()}
    return {int(i): str(v) for i, v in enumerate(label_names)}


def _load_label_to_id(path: Path) -> Dict[str, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    sem = payload.get("semantic_segmentation", payload) if isinstance(payload, dict) else {}
    if not isinstance(sem, dict):
        return {}
    label_names = _normalise_label_names(sem.get("label_names", []))
    return {name: idx for idx, name in label_names.items()}


def _ids_for(names: Iterable[str], label_to_id: Mapping[str, int]) -> Set[int]:
    return {int(label_to_id[name]) for name in names if name in label_to_id}


def _parse_filter_spec(spec: str, label_to_id: Mapping[str, int]) -> Tuple[str, Optional[Set[int]]]:
    value = str(spec).strip()
    if not value or value in {"all", "unfiltered"}:
        return "all", None
    ids: Set[int] = set()
    for part in value.split("+"):
        item = part.strip()
        if not item:
            continue
        if item in GROUP_LABELS:
            ids.update(_ids_for(GROUP_LABELS[item], label_to_id))
        elif item.startswith("label:"):
            label_name = item.split(":", 1)[1].strip()
            if label_name in label_to_id:
                ids.add(int(label_to_id[label_name]))
        elif item.startswith("ids:"):
            for raw_id in item.split(":", 1)[1].split(","):
                raw_id = raw_id.strip()
                if raw_id:
                    ids.add(int(raw_id))
        elif item in label_to_id:
            ids.add(int(label_to_id[item]))
        else:
            raise argparse.ArgumentTypeError(f"Unknown semantic fit filter item: {item}")
    if not ids:
        raise argparse.ArgumentTypeError(f"Semantic fit filter has no labels in this cache: {value}")
    return value, ids


def _safe_tag(value: str) -> str:
    out = []
    for ch in str(value):
        if ch.isalnum():
            out.append(ch)
        elif ch in {"_", "-"}:
            out.append(ch)
        elif ch == "+":
            out.append("PLUS")
        else:
            out.append("_")
    return "".join(out).strip("_") or "all"


def _select_fit_points(
    pair: Dict[str, Any],
    max_points: int,
    *,
    label_ids: Optional[Set[int]] = None,
    semantic_min_conf: float = 0.0,
    min_filter_fit_points: int = 0,
) -> Tuple[np.ndarray, np.ndarray, int, int, Optional[str]]:
    prev = pair.get("prev_overlap_points")
    curr = pair.get("curr_overlap_points")
    if not torch.is_tensor(prev) or not torch.is_tensor(curr):
        return np.empty((0, 3)), np.empty((0, 3)), 0, 0, "missing_overlap_points"
    prev = prev.detach().cpu().float()
    curr = curr.detach().cpu().float()
    valid = torch.isfinite(prev).all(dim=-1) & torch.isfinite(curr).all(dim=-1)
    prev_conf = pair.get("prev_conf")
    curr_conf = pair.get("curr_conf")
    if torch.is_tensor(prev_conf) and torch.is_tensor(curr_conf):
        score = torch.minimum(prev_conf.detach().cpu().float(), curr_conf.detach().cpu().float())
    else:
        score = torch.ones((prev.shape[0],), dtype=torch.float32)
    score = torch.where(valid, score, torch.full_like(score, -1.0))
    valid_count = int(valid.sum().item())
    fit_mask = valid.clone()
    if label_ids is not None:
        labels = pair.get("curr_semantic_labels")
        if not torch.is_tensor(labels) or labels.numel() != fit_mask.numel():
            return np.empty((0, 3)), np.empty((0, 3)), valid_count, 0, "missing_or_misaligned_semantic_labels"
        labels = labels.detach().cpu().long()
        label_mask = torch.zeros_like(fit_mask, dtype=torch.bool)
        for label_id in sorted(label_ids):
            label_mask |= labels == int(label_id)
        fit_mask &= label_mask
    if float(semantic_min_conf) > 0.0:
        sem_conf = pair.get("curr_semantic_conf")
        if not torch.is_tensor(sem_conf) or sem_conf.numel() != fit_mask.numel():
            return np.empty((0, 3)), np.empty((0, 3)), valid_count, 0, "missing_or_misaligned_semantic_conf"
        fit_mask &= sem_conf.detach().cpu().float() >= float(semantic_min_conf)
    fit_valid_count = int(fit_mask.sum().item())
    if label_ids is not None and fit_valid_count < int(min_filter_fit_points):
        return (
            np.empty((0, 3)),
            np.empty((0, 3)),
            valid_count,
            fit_valid_count,
            f"filter_support_too_small:{fit_valid_count}<{int(min_filter_fit_points)}",
        )
    if fit_valid_count < 3:
        return np.empty((0, 3)), np.empty((0, 3)), valid_count, fit_valid_count, "not_enough_fit_points"
    score = torch.where(fit_mask, score, torch.full_like(score, -1.0))
    k = min(int(max_points), fit_valid_count) if int(max_points) > 0 else fit_valid_count
    _, idx = torch.topk(score, k=k, largest=True, sorted=False)
    return prev[idx].numpy(), curr[idx].numpy(), valid_count, fit_valid_count, None


def _fit_pair_correction(
    prev_points: np.ndarray,
    curr_points: np.ndarray,
    *,
    with_scale: bool,
) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    if prev_points.shape != curr_points.shape or prev_points.shape[0] < 3:
        return None, None, None, "not_enough_points"
    try:
        scale, rot, trans = _umeyama_sim3(curr_points, prev_points, with_scale=with_scale)
    except Exception as exc:  # noqa: BLE001 - diagnostic records fit failures.
        return None, None, None, f"fit_error:{type(exc).__name__}:{exc}"
    if not np.isfinite(scale) or not np.all(np.isfinite(rot)) or not np.all(np.isfinite(trans)):
        return None, None, None, "fit_nonfinite"
    return float(scale), rot, trans, None


def _make_baseline_row(
    frames: np.ndarray,
    poses: np.ndarray,
    gt_pos: np.ndarray,
    trace: Dict[int, Dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
    head_len: int,
) -> Dict[str, Any]:
    scales = [float(trace[c]["scale"]) for c in trace]
    controller_rows = [
        {
            "chunk_idx": int(c),
            "action": "baseline",
            "source_scale": float(trace[c]["scale"]),
            "ctrl_scale": float(trace[c]["scale"]),
            "is_estimated_boundary": str(trace[c].get("row", {}).get("transform_reason", "")) == "estimated_overlap_transform",
        }
        for c in sorted(trace)
    ]
    return _metric_result_row(
        "BASELINE_SOURCE_NOOP",
        frames,
        poses,
        gt_pos,
        None,
        controller_rows,
        sum(abs(s - 1.0) > 1e-6 for s in scales),
        any(abs(s - 1.0) > 1e-6 for s in scales),
        chunk_size,
        chunk_overlap,
        head_len,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=_parse_source, required=True)
    parser.add_argument("--overlap-pairs-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--head-len", type=int, default=10)
    parser.add_argument("--max-fit-points", type=int, default=20000)
    parser.add_argument("--semantic-full-pt", type=Path, default=Path("results/kitti_preprocess/01/sparse_masklets_with_semantic.pt"))
    parser.add_argument("--fit-semantic-filter", action="append", default=None)
    parser.add_argument("--semantic-min-conf", type=float, default=0.0)
    parser.add_argument("--min-filter-fit-points", type=int, default=256)
    parser.add_argument("--max-safe-rotation-deg", type=float, default=2.0)
    parser.add_argument("--max-safe-overlap-displacement-m", type=float, default=0.5)
    parser.add_argument("--max-safe-log-scale-delta", type=float, default=0.03)
    parser.add_argument("--min-raw-overlap-improvement-ratio", type=float, default=0.20)
    parser.add_argument("--min-mechanism-improvement-ratio", type=float, default=0.10)
    parser.add_argument("--max-ate-regression-m", type=float, default=0.30)
    parser.add_argument("--damping-alpha", type=float, action="append", default=None)
    args = parser.parse_args()

    source_label, run_dir = args.source
    pairs_dir = args.overlap_pairs_dir or (run_dir / "overlap_pairs")
    pair_files = sorted(pairs_dir.glob("chunk_*_*.pt"))
    if not pair_files:
        raise FileNotFoundError(f"No overlap pair files in {pairs_dir}")
    trace = _load_trace(run_dir / "merge_state_trace.jsonl")
    frames, poses, frame_to_chunk = _load_postmerge_trajectory(run_dir / "postmerge_global_pose.jsonl")
    _, _, gt_pos = _load_kitti_gt(args.gt)
    baseline = _make_baseline_row(frames, poses, gt_pos, trace, args.chunk_size, args.chunk_overlap, args.head_len)
    damping_alphas = args.damping_alpha if args.damping_alpha is not None else [1.0, 0.5, 0.25]
    scopes = ["current", "boundary_span", "future"]
    actions = [("SE3_PAIR", False), ("SIM3_PAIR", True)]
    label_to_id = _load_label_to_id(args.semantic_full_pt)
    fit_filter_specs = args.fit_semantic_filter if args.fit_semantic_filter is not None else ["all"]
    fit_filters = [_parse_filter_spec(spec, label_to_id) for spec in fit_filter_specs]
    rows: List[Dict[str, Any]] = []
    fit_failures: List[Dict[str, Any]] = []

    for pair_file in pair_files:
        pair = _load_pair(pair_file)
        curr_chunk = int(pair.get("curr_chunk"))
        prev_chunk = int(pair.get("prev_chunk", curr_chunk - 1))
        for fit_filter_name, fit_label_ids in fit_filters:
            prev_points, curr_points, valid_count, filter_fit_point_count, filter_reason = _select_fit_points(
                pair,
                int(args.max_fit_points),
                label_ids=fit_label_ids,
                semantic_min_conf=float(args.semantic_min_conf),
                min_filter_fit_points=int(args.min_filter_fit_points),
            )
            raw_before = _rmse_dist(prev_points, curr_points)
            for action_name, with_scale in actions:
                full_scale, full_rot, full_trans, fail_reason = _fit_pair_correction(
                    prev_points,
                    curr_points,
                    with_scale=with_scale,
                )
                if filter_reason is not None:
                    fail_reason = filter_reason if fail_reason is None else f"{filter_reason};{fail_reason}"
                if fail_reason is not None or full_scale is None or full_rot is None or full_trans is None:
                    fit_failures.append({
                        "pair_file": str(pair_file),
                        "prev_chunk": int(prev_chunk),
                        "curr_chunk": int(curr_chunk),
                        "fit_semantic_filter": fit_filter_name,
                        "fit_semantic_label_ids": sorted(fit_label_ids) if fit_label_ids is not None else [],
                        "semantic_min_conf": float(args.semantic_min_conf),
                        "action_family": action_name,
                        "valid_pair_count": int(valid_count),
                        "filter_fit_point_count": int(filter_fit_point_count),
                        "fit_failure": fail_reason,
                    })
                    continue
                for damping_alpha in damping_alphas:
                    scale = float(full_scale) ** float(damping_alpha) if with_scale else 1.0
                    rot = _rotation_power(full_rot, float(damping_alpha))
                    trans = float(damping_alpha) * full_trans
                    corrected = float(scale) * (rot @ curr_points.T).T + trans
                    raw_after = _rmse_dist(prev_points, corrected)
                    raw_improvement = _safe_improvement_ratio(raw_before, raw_after)
                    overlap_displacement = _rmse_dist(curr_points, corrected)
                    rot_deg = _rotation_delta_deg(rot)
                    abs_log_scale = abs(math.log(max(float(scale), 1e-12)))
                    safe_rotation = rot_deg <= float(args.max_safe_rotation_deg)
                    safe_displacement = overlap_displacement <= float(args.max_safe_overlap_displacement_m)
                    safe_scale = abs_log_scale <= float(args.max_safe_log_scale_delta)
                    safe_correction = bool(safe_rotation and safe_displacement and (safe_scale or not with_scale))
                    alpha_tag = str(damping_alpha).replace(".", "p")
                    damped_action_name = f"{action_name}_A{alpha_tag}"
                    for scope in scopes:
                        targets = _target_chunks(trace, curr_chunk, scope)
                        candidate_poses = _apply_global_correction(
                            frames,
                            poses,
                            frame_to_chunk,
                            targets,
                            float(scale),
                            rot,
                            trans,
                        )
                        controller_rows = [
                            {
                                "chunk_idx": int(c),
                                "action": "overlap_pair_oracle" if c in targets else "baseline",
                                "source_scale": float(trace[c]["scale"]),
                                "ctrl_scale": float(trace[c]["scale"]),
                                "is_estimated_boundary": str(trace[c].get("row", {}).get("transform_reason", "")) == "estimated_overlap_transform",
                            }
                            for c in sorted(trace)
                        ]
                        metric = _metric_result_row(
                            f"{damped_action_name}_{scope}_b{curr_chunk}",
                            frames,
                            candidate_poses,
                            gt_pos,
                            baseline,
                            controller_rows,
                            int(baseline.get("source_nonunit_scale_count", 0)),
                            bool(baseline.get("source_has_scale_state", False)),
                            args.chunk_size,
                            args.chunk_overlap,
                            args.head_len,
                        )
                        if fit_filter_name != "all" or len(fit_filters) > 1:
                            metric["candidate"] = f"{metric.get('candidate')}_fit_{_safe_tag(fit_filter_name)}"
                        best_mech = _best_mechanism_improvement(metric)
                        ate_delta = _float(metric.get("delta_vs_baseline_global_ate"))
                        raw_support_pass = bool(
                            math.isfinite(raw_improvement)
                            and raw_improvement >= float(args.min_raw_overlap_improvement_ratio)
                        )
                        mechanism_pass = bool(
                            math.isfinite(best_mech)
                            and best_mech >= float(args.min_mechanism_improvement_ratio)
                        )
                        ate_guard_pass = bool(math.isfinite(ate_delta) and ate_delta <= float(args.max_ate_regression_m))
                        gate_pass = bool(raw_support_pass and mechanism_pass and ate_guard_pass and safe_correction)
                        metric.update({
                            "source_label": source_label,
                            "source_run": str(run_dir),
                            "overlap_pair_file": str(pair_file),
                            "prev_chunk": int(prev_chunk),
                            "curr_chunk": int(curr_chunk),
                            "fit_semantic_filter": fit_filter_name,
                            "fit_semantic_label_ids": sorted(fit_label_ids) if fit_label_ids is not None else [],
                            "semantic_min_conf": float(args.semantic_min_conf),
                            "filter_fit_point_count": int(filter_fit_point_count),
                            "filter_reason": filter_reason or "",
                            "action_family": action_name,
                            "damped_action_family": damped_action_name,
                            "damping_alpha": float(damping_alpha),
                            "scope": scope,
                            "target_chunk_count": int(len(targets)),
                            "target_chunks_first": int(targets[0]) if targets else None,
                            "target_chunks_last": int(targets[-1]) if targets else None,
                            "fit_point_count": int(prev_points.shape[0]),
                            "valid_pair_count": int(valid_count),
                            "raw_overlap_before_m": raw_before,
                            "raw_overlap_after_m": raw_after,
                            "raw_overlap_improvement_ratio": raw_improvement,
                            "correction_overlap_displacement_m": overlap_displacement,
                            "correction_scale": float(scale),
                            "correction_abs_log_scale_delta": abs_log_scale,
                            "correction_rotation_deg": rot_deg,
                            "correction_translation_norm_m": float(np.linalg.norm(trans)),
                            "safe_rotation_pass": safe_rotation,
                            "safe_overlap_displacement_pass": safe_displacement,
                            "safe_scale_pass": safe_scale if with_scale else True,
                            "safe_correction_pass": safe_correction,
                            "best_mechanism_improvement": best_mech,
                            "raw_support_pass": raw_support_pass,
                            "mechanism_pass": mechanism_pass,
                            "ate_guard_pass": ate_guard_pass,
                            "oracle_action_gate_pass": gate_pass,
                        })
                        rows.append(metric)

    rows.sort(key=lambda row: (
        not bool(row.get("oracle_action_gate_pass")),
        -_float(row.get("best_mechanism_improvement")),
        -_float(row.get("raw_overlap_improvement_ratio")),
        _float(row.get("delta_vs_baseline_global_ate")),
        str(row.get("candidate")),
    ))
    gate_rows = [row for row in rows if bool(row.get("oracle_action_gate_pass"))]
    summary = {
        "schema": "acl2_v67_overlap_pair_action_oracle_summary_v1",
        "source_label": source_label,
        "source_run": str(run_dir),
        "overlap_pairs_dir": str(pairs_dir),
        "pair_files": len(pair_files),
        "rows": len(rows),
        "fit_failures": fit_failures,
        "counts": {
            "safe_correction_pass": sum(bool(row.get("safe_correction_pass")) for row in rows),
            "raw_support_pass": sum(bool(row.get("raw_support_pass")) for row in rows),
            "mechanism_pass": sum(bool(row.get("mechanism_pass")) for row in rows),
            "ate_guard_pass": sum(bool(row.get("ate_guard_pass")) for row in rows),
            "oracle_action_gate_pass": len(gate_rows),
        },
        "gate_rule": {
            "semantic_full_pt": str(args.semantic_full_pt),
            "fit_semantic_filters": [name for name, _ in fit_filters],
            "semantic_min_conf": float(args.semantic_min_conf),
            "min_filter_fit_points": int(args.min_filter_fit_points),
            "min_raw_overlap_improvement_ratio": float(args.min_raw_overlap_improvement_ratio),
            "min_mechanism_improvement_ratio": float(args.min_mechanism_improvement_ratio),
            "max_ate_regression_m": float(args.max_ate_regression_m),
            "max_safe_rotation_deg": float(args.max_safe_rotation_deg),
            "max_safe_overlap_displacement_m": float(args.max_safe_overlap_displacement_m),
            "max_safe_log_scale_delta": float(args.max_safe_log_scale_delta),
            "damping_alphas": [float(x) for x in damping_alphas],
        },
        "best_row": rows[0] if rows else {},
        "mean_raw_overlap_improvement_ratio": _finite_mean(row.get("raw_overlap_improvement_ratio") for row in rows),
        "mean_best_mechanism_improvement": _finite_mean(row.get("best_mechanism_improvement") for row in rows),
        "oracle_action_gate_pass": bool(gate_rows),
        "note": "Diagnostic-only oracle using materialized raw overlap point pairs; not a semantic method result.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "overlap_pair_action_oracle_results.csv", rows)
    (args.out_dir / "overlap_pair_action_oracle_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
