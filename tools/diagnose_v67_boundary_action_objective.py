#!/usr/bin/env python3
"""Held-out boundary-action objective diagnostic for ACL2 v67 O3.

This diagnostic asks whether pre-action boundary/source context can predict
which reset-boundary hold action improves the trajectory. It intentionally keeps
method claims separate from posthoc oracle labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from diagnose_v67_boundary_predictiveness import _normalize_row
    from diagnose_v67_offline_scale_controller import (
        DEFAULT_GT,
        _apply_scale_control,
        _load_postmerge_trajectory,
        _load_trace,
    )
except ImportError:  # pragma: no cover
    from tools.diagnose_v67_boundary_predictiveness import _normalize_row
    from tools.diagnose_v67_offline_scale_controller import (
        DEFAULT_GT,
        _apply_scale_control,
        _load_postmerge_trajectory,
        _load_trace,
    )


NO_GT_FEATURES = [
    "semantic_Q_scale",
    "semantic_lowQ_hold",
    "semantic_Q_smoothed",
    "geometry_confidence",
    "source_abs_log_scale_from_first",
    "source_abs_log_scale_from_prev_boundary",
    "source_abs_log_scale_from_prev_chunk",
    "source_transform_trans_norm",
    "source_rot_trace_abs_dev_from_3",
    "baseline_raw_overlap_residual_m",
    "baseline_entry_jump_m",
    "baseline_exit_jump_m",
    "combined_source_gauge_no_gt",
    "combined_semantic_source_gauge",
]


def _float(value: Any, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _auc(scores: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    positives = [float(s) for s, label in zip(scores, labels) if label and math.isfinite(float(s))]
    negatives = [float(s) for s, label in zip(scores, labels) if not label and math.isfinite(float(s))]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0
    for ps in positives:
        for ns in negatives:
            total += 1
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return float(wins / total) if total else None


def _safe_log(value: float) -> float:
    return math.log(max(float(value), 1e-12))


def _finite_mean(values: Iterable[float]) -> Optional[float]:
    xs = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(xs)) if xs else None


def _finite_std(values: Iterable[float]) -> Optional[float]:
    xs = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.std(xs)) if xs else None


def _zscore_values(rows: List[Dict[str, Any]], key: str) -> Dict[int, float]:
    vals = [_float(row.get(key)) for row in rows]
    mean = _finite_mean(vals)
    std = _finite_std(vals)
    out: Dict[int, float] = {}
    for idx, value in enumerate(vals):
        if mean is None or std is None or std < 1e-12 or not math.isfinite(value):
            out[idx] = 0.0
        else:
            out[idx] = float((value - mean) / std)
    return out


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_premerge_windows(run_dir: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    path = run_dir / "premerge_local_pose.jsonl"
    if not path.is_file():
        return out
    for row in _read_jsonl(path):
        chunk_idx = int(row.get("chunk_idx", row.get("local_chunk_idx", len(out))))
        poses = np.asarray(row.get("camera_poses", []), dtype=np.float64)
        if poses.ndim == 3 and poses.shape[-2:] == (4, 4):
            out[chunk_idx] = {"row": row, "poses": poses}
    return out


def _load_source(
    run_dir: Path,
) -> Tuple[Dict[int, Dict[str, Any]], np.ndarray, np.ndarray, Dict[int, int], Dict[int, Dict[str, Any]]]:
    trace = _load_trace(run_dir / "merge_state_trace.jsonl")
    frames, poses, frame_to_chunk = _load_postmerge_trajectory(run_dir / "postmerge_global_pose.jsonl")
    premerge = _load_premerge_windows(run_dir)
    return trace, frames, poses, frame_to_chunk, premerge


def _step_distance(frames: np.ndarray, poses: np.ndarray, frame_a: int, frame_b: int) -> float:
    index = {int(frame): i for i, frame in enumerate(frames)}
    if frame_a not in index or frame_b not in index:
        return float("nan")
    pa = poses[index[frame_a], :3, 3]
    pb = poses[index[frame_b], :3, 3]
    return float(np.linalg.norm(pb - pa))


def _transform_local_positions(local_poses: np.ndarray, scale: float, mat: np.ndarray) -> np.ndarray:
    rot = np.asarray(mat[:3, :3], dtype=np.float64)
    trans = np.asarray(mat[:3, 3], dtype=np.float64)
    local_t = np.asarray(local_poses[:, :3, 3], dtype=np.float64)
    return (float(scale) * (rot @ local_t.T)).T + trans


def _raw_overlap_residual(
    trace: Dict[int, Dict[str, Any]],
    premerge: Dict[int, Dict[str, Any]],
    boundary: int,
    *,
    current_scale_override: Optional[float] = None,
) -> float:
    prev = int(boundary) - 1
    if prev not in trace or boundary not in trace or prev not in premerge or boundary not in premerge:
        return float("nan")
    overlap = int(trace[boundary].get("row", {}).get("overlap_size", 0) or 0)
    if overlap <= 0:
        return float("nan")
    prev_poses = premerge[prev]["poses"]
    curr_poses = premerge[boundary]["poses"]
    actual = min(overlap, prev_poses.shape[0], curr_poses.shape[0])
    if actual <= 0:
        return float("nan")
    prev_pos = _transform_local_positions(
        prev_poses[-actual:],
        float(trace[prev]["scale"]),
        np.asarray(trace[prev]["matrix"], dtype=np.float64),
    )
    curr_scale = float(current_scale_override if current_scale_override is not None else trace[boundary]["scale"])
    curr_pos = _transform_local_positions(
        curr_poses[:actual],
        curr_scale,
        np.asarray(trace[boundary]["matrix"], dtype=np.float64),
    )
    dist = np.linalg.norm(prev_pos - curr_pos, axis=1)
    return float(np.sqrt(np.mean(dist * dist)))


def _boundary_context(
    row: Dict[str, Any],
    source_cache: Dict[
        str,
        Tuple[Dict[int, Dict[str, Any]], np.ndarray, np.ndarray, Dict[int, int], Dict[int, Dict[str, Any]]],
    ],
) -> Dict[str, Any]:
    run_dir = Path(str(row.get("source_run", "")))
    run_key = str(run_dir)
    if run_key not in source_cache:
        source_cache[run_key] = _load_source(run_dir)
    trace, frames, poses, frame_to_chunk, premerge = source_cache[run_key]

    boundary = int(float(str(row["held_boundary_chunk"])))
    source_scales = {c: float(trace[c]["scale"]) for c in trace}
    first_chunk = min(trace)
    first_scale = source_scales[first_chunk]
    boundary_scale = source_scales[boundary]
    sorted_chunks = sorted(trace)
    pos = sorted_chunks.index(boundary)
    prev_chunk = sorted_chunks[pos - 1] if pos > 0 else boundary
    prev_chunk_scale = source_scales.get(prev_chunk, boundary_scale)
    estimated_boundaries = [
        c for c in sorted_chunks
        if str(trace[c].get("row", {}).get("transform_reason", "")) == "estimated_overlap_transform"
    ]
    prev_estimated = max([c for c in estimated_boundaries if c < boundary], default=first_chunk)
    prev_estimated_scale = source_scales.get(prev_estimated, first_scale)

    trace_row = trace[boundary].get("row", {})
    emitted = [int(x) for x in trace_row.get("emitted_frame_ids", [])]
    entry_frame = emitted[0] if emitted else int(trace_row.get("start_frame", -1))
    exit_frame = emitted[-1] if emitted else int(trace_row.get("end_frame", -1)) - 1
    baseline_entry_jump = _step_distance(frames, poses, entry_frame - 1, entry_frame)
    baseline_exit_jump = _step_distance(frames, poses, exit_frame, exit_frame + 1)
    baseline_raw_overlap_residual = _raw_overlap_residual(trace, premerge, boundary)

    ctrl_scales = dict(source_scales)
    ctrl_scales[boundary] = first_scale
    candidate_poses = _apply_scale_control(
        frames,
        poses,
        frame_to_chunk,
        trace,
        ctrl_scales,
        origin_mode=str(row.get("origin_mode", "transform_translation")),
    )
    candidate_entry_jump = _step_distance(frames, candidate_poses, entry_frame - 1, entry_frame)
    candidate_exit_jump = _step_distance(frames, candidate_poses, exit_frame, exit_frame + 1)
    candidate_raw_overlap_residual = _raw_overlap_residual(
        trace,
        premerge,
        boundary,
        current_scale_override=first_scale,
    )

    return {
        "source_abs_log_scale_from_first": abs(_safe_log(boundary_scale) - _safe_log(first_scale)),
        "source_abs_log_scale_from_prev_boundary": abs(_safe_log(boundary_scale) - _safe_log(prev_estimated_scale)),
        "source_abs_log_scale_from_prev_chunk": abs(_safe_log(boundary_scale) - _safe_log(prev_chunk_scale)),
        "source_transform_trans_norm": _float(trace_row.get("transform_trans_norm")),
        "source_rot_trace_abs_dev_from_3": abs(_float(trace_row.get("transform_rot_trace"), 3.0) - 3.0),
        "baseline_raw_overlap_residual_m": baseline_raw_overlap_residual,
        "baseline_entry_jump_m": baseline_entry_jump,
        "baseline_exit_jump_m": baseline_exit_jump,
        "candidate_raw_overlap_residual_m": candidate_raw_overlap_residual,
        "candidate_entry_jump_m": candidate_entry_jump,
        "candidate_exit_jump_m": candidate_exit_jump,
        "raw_overlap_residual_improvement_m": (
            float(baseline_raw_overlap_residual - candidate_raw_overlap_residual)
            if math.isfinite(baseline_raw_overlap_residual) and math.isfinite(candidate_raw_overlap_residual)
            else float("nan")
        ),
        "raw_overlap_residual_improvement_ratio": (
            float((baseline_raw_overlap_residual - candidate_raw_overlap_residual) / abs(baseline_raw_overlap_residual))
            if math.isfinite(baseline_raw_overlap_residual)
            and math.isfinite(candidate_raw_overlap_residual)
            and abs(baseline_raw_overlap_residual) > 1e-12
            else float("nan")
        ),
        "entry_jump_improvement_m": (
            float(baseline_entry_jump - candidate_entry_jump)
            if math.isfinite(baseline_entry_jump) and math.isfinite(candidate_entry_jump)
            else float("nan")
        ),
        "exit_jump_improvement_m": (
            float(baseline_exit_jump - candidate_exit_jump)
            if math.isfinite(baseline_exit_jump) and math.isfinite(candidate_exit_jump)
            else float("nan")
        ),
        "entry_frame": entry_frame,
        "exit_frame": exit_frame,
        "prev_estimated_boundary": prev_estimated,
        "transform_reason": trace_row.get("transform_reason"),
    }


def _augment_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    source_cache: Dict[
        str,
        Tuple[Dict[int, Dict[str, Any]], np.ndarray, np.ndarray, Dict[int, int], Dict[int, Dict[str, Any]]],
    ] = {}
    out: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["semantic_Q_scale"] = _float(item.get("boundary_Q_scale"))
        item["semantic_lowQ_hold"] = -_float(item.get("boundary_Q_scale"))
        item["semantic_Q_smoothed"] = _float(item.get("boundary_Q_scale_smoothed"))
        item["geometry_confidence"] = _float(item.get("boundary_geometry_confidence_mean"))
        item.update(_boundary_context(item, source_cache))
        item["ate_improves"] = _float(item.get("delta_vs_baseline_global_ate")) < 0.0
        out.append(item)

    z_abs_first = _zscore_values(out, "source_abs_log_scale_from_first")
    z_abs_prev = _zscore_values(out, "source_abs_log_scale_from_prev_boundary")
    z_trans = _zscore_values(out, "source_transform_trans_norm")
    z_entry = _zscore_values(out, "baseline_entry_jump_m")
    z_overlap = _zscore_values(out, "baseline_raw_overlap_residual_m")
    z_q = _zscore_values(out, "semantic_Q_scale")
    z_geom = _zscore_values(out, "geometry_confidence")
    for idx, item in enumerate(out):
        item["combined_source_gauge_no_gt"] = float(
            z_abs_first[idx] + z_abs_prev[idx] + z_trans[idx] + z_entry[idx] + z_overlap[idx]
        )
        item["combined_semantic_source_gauge"] = float(
            item["combined_source_gauge_no_gt"] + 0.5 * z_q[idx] + 0.5 * z_geom[idx]
        )
    return out


def _direction_from_train(rows: Sequence[Dict[str, Any]], feature: str) -> Tuple[str, Optional[float]]:
    scores = [_float(row.get(feature)) for row in rows]
    labels = [bool(row.get("ate_improves")) for row in rows]
    auc_high = _auc(scores, labels)
    auc_low = _auc([-s for s in scores], labels)
    if auc_high is None or auc_low is None:
        return "higher", None
    return ("higher", auc_high) if auc_high >= auc_low else ("lower", auc_low)


def _score(row: Dict[str, Any], feature: str, direction: str) -> float:
    value = _float(row.get(feature))
    return value if direction == "higher" else -value


def _evaluate_feature(rows: List[Dict[str, Any]], feature: str, group: str) -> Dict[str, Any]:
    scores = [_float(row.get(feature)) for row in rows]
    labels = [bool(row.get("ate_improves")) for row in rows]
    global_auc_high = _auc(scores, labels)
    global_auc_low = _auc([-s for s in scores], labels)
    if global_auc_high is None or global_auc_low is None:
        global_direction = "higher"
        global_best_auc = None
    elif global_auc_high >= global_auc_low:
        global_direction = "higher"
        global_best_auc = global_auc_high
    else:
        global_direction = "lower"
        global_best_auc = global_auc_low

    folds: List[Dict[str, Any]] = []
    for source_label in sorted({str(row.get("source_label")) for row in rows}):
        train = [row for row in rows if str(row.get("source_label")) != source_label]
        test = [row for row in rows if str(row.get("source_label")) == source_label]
        direction, train_auc = _direction_from_train(train, feature)
        test_scores = [_score(row, feature, direction) for row in test]
        test_labels = [bool(row.get("ate_improves")) for row in test]
        test_auc = _auc(test_scores, test_labels)
        finite_test = [row for row in test if math.isfinite(_score(row, feature, direction))]
        if finite_test:
            top = max(finite_test, key=lambda row: _score(row, feature, direction))
            top_positive = bool(top.get("ate_improves"))
            top_delta = _float(top.get("delta_vs_baseline_global_ate"))
            top_headtail_imp = _float(top.get("head_to_tail_transfer_ratio_mean_improvement_vs_baseline"))
            top_future_imp = _float(top.get("future_after_overlap_mean_improvement_vs_baseline"))
            top_intra_imp = _float(top.get("intra_scale_variance_mean_improvement_vs_baseline"))
            top_raw_overlap_imp = _float(top.get("raw_overlap_residual_improvement_ratio"))
            top_candidate = top.get("candidate")
            top_origin = top.get("origin_mode")
            top_boundary = top.get("held_boundary_chunk")
        else:
            top_positive = None
            top_delta = None
            top_headtail_imp = None
            top_future_imp = None
            top_intra_imp = None
            top_raw_overlap_imp = None
            top_candidate = None
            top_origin = None
            top_boundary = None
        folds.append({
            "group": group,
            "feature": feature,
            "heldout_source_label": source_label,
            "train_direction": direction,
            "train_auc": train_auc,
            "test_auc": test_auc,
            "test_n": len(test),
            "test_positive_n": sum(test_labels),
            "test_negative_n": len(test_labels) - sum(test_labels),
            "top1_positive": top_positive,
            "top1_delta_ate": top_delta,
            "top1_headtail_imp": top_headtail_imp,
            "top1_future_imp": top_future_imp,
            "top1_intra_imp": top_intra_imp,
            "top1_raw_overlap_imp": top_raw_overlap_imp,
            "top1_candidate": top_candidate,
            "top1_origin_mode": top_origin,
            "top1_boundary": top_boundary,
        })

    valid_auc = [float(row["test_auc"]) for row in folds if row.get("test_auc") is not None]
    top_rows = [row for row in folds if row.get("top1_positive") is not None]
    top_best_mech = [
        max(
            _float(row.get("top1_headtail_imp")),
            _float(row.get("top1_future_imp")),
            _float(row.get("top1_intra_imp")),
        )
        for row in top_rows
    ]
    return {
        "group": group,
        "feature": feature,
        "global_auc_higher": global_auc_high,
        "global_auc_lower": global_auc_low,
        "global_best_auc": global_best_auc,
        "global_best_direction": global_direction,
        "heldout_source_auc_mean": _finite_mean(valid_auc),
        "heldout_source_auc_folds": len(valid_auc),
        "heldout_top1_positive_rate": (
            None if not top_rows else sum(bool(row["top1_positive"]) for row in top_rows) / len(top_rows)
        ),
        "heldout_top1_mean_delta_ate": _finite_mean([
            _float(row.get("top1_delta_ate")) for row in top_rows if row.get("top1_delta_ate") is not None
        ]),
        "heldout_top1_mean_headtail_imp": _finite_mean([
            _float(row.get("top1_headtail_imp")) for row in top_rows if row.get("top1_headtail_imp") is not None
        ]),
        "heldout_top1_mean_future_imp": _finite_mean([
            _float(row.get("top1_future_imp")) for row in top_rows if row.get("top1_future_imp") is not None
        ]),
        "heldout_top1_mean_intra_imp": _finite_mean([
            _float(row.get("top1_intra_imp")) for row in top_rows if row.get("top1_intra_imp") is not None
        ]),
        "heldout_top1_mean_best_mech_imp": _finite_mean(top_best_mech),
        "heldout_top1_mean_raw_overlap_imp": _finite_mean([
            _float(row.get("top1_raw_overlap_imp")) for row in top_rows if row.get("top1_raw_overlap_imp") is not None
        ]),
        "folds": folds,
    }


def _evaluation_groups(rows: List[Dict[str, Any]]) -> List[Tuple[str, List[Dict[str, Any]]]]:
    groups: List[Tuple[str, List[Dict[str, Any]]]] = [("all", rows)]
    for origin_mode in sorted({str(row.get("origin_mode")) for row in rows}):
        groups.append((f"origin={origin_mode}", [row for row in rows if str(row.get("origin_mode")) == origin_mode]))
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-csv", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT, help="Reserved for audit symmetry; labels are read from oracle CSVs.")
    args = parser.parse_args()

    raw_rows: List[Dict[str, Any]] = []
    for path in args.oracle_csv:
        for row in _read_csv(path):
            normalized = _normalize_row(dict(row), path)
            if (
                str(normalized.get("candidate", "")).startswith("hold_only_")
                and normalized.get("source_run")
                and normalized.get("held_boundary_chunk") not in (None, "")
                and math.isfinite(_float(normalized.get("delta_vs_baseline_global_ate")))
            ):
                raw_rows.append(normalized)

    rows = _augment_rows(raw_rows)
    feature_summaries: List[Dict[str, Any]] = []
    for group, group_rows in _evaluation_groups(rows):
        for feature in NO_GT_FEATURES:
            feature_summaries.append(_evaluate_feature(group_rows, feature, group))
    flat_summary_rows = [
        {k: v for k, v in summary.items() if k != "folds"}
        for summary in feature_summaries
    ]
    fold_rows: List[Dict[str, Any]] = []
    for summary in feature_summaries:
        fold_rows.extend(summary["folds"])

    flat_summary_rows.sort(key=lambda row: (
        -1.0 if row.get("heldout_source_auc_mean") is None else -float(row["heldout_source_auc_mean"]),
        -1.0 if row.get("heldout_top1_positive_rate") is None else -float(row["heldout_top1_positive_rate"]),
        0.0 if row.get("heldout_top1_mean_delta_ate") is None else float(row["heldout_top1_mean_delta_ate"]),
        str(row.get("group")),
        str(row.get("feature")),
    ))

    best = flat_summary_rows[0] if flat_summary_rows else {}
    best_auc = best.get("heldout_source_auc_mean")
    best_group = best.get("group")
    controls_same_group = {
        row["feature"]: row for row in flat_summary_rows
        if row.get("group") == best_group
    }
    geometry_auc = controls_same_group.get("geometry_confidence", {}).get("heldout_source_auc_mean")
    semantic_auc = controls_same_group.get("semantic_Q_scale", {}).get("heldout_source_auc_mean")
    gate_pass = bool(
        best_auc is not None
        and float(best_auc) >= 0.65
        and (
            geometry_auc is None
            or str(best.get("feature")) == "geometry_confidence"
            or float(best_auc) >= float(geometry_auc) + 0.05
        )
        and (
            semantic_auc is None
            or str(best.get("feature")) == "semantic_Q_scale"
            or float(best_auc) >= float(semantic_auc) + 0.05
        )
    )
    best_top1_delta = best.get("heldout_top1_mean_delta_ate")
    best_top1_mech = best.get("heldout_top1_mean_best_mech_imp")
    o3_action_gate_pass = bool(
        gate_pass
        and best_top1_delta is not None
        and float(best_top1_delta) <= 0.3
        and best_top1_mech is not None
        and float(best_top1_mech) >= 0.10
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "boundary_action_objective_rows.csv", rows)
    _write_csv(out_dir / "heldout_feature_summary.csv", flat_summary_rows)
    _write_csv(out_dir / "heldout_feature_folds.csv", fold_rows)
    summary = {
        "oracle_csvs": [str(path) for path in args.oracle_csv],
        "usable_rows": len(rows),
        "feature_count": len(NO_GT_FEATURES),
        "groups": [group for group, _ in _evaluation_groups(rows)],
        "heldout_split": "leave_one_source_label_out within each group",
        "best_feature": best,
        "semantic_Q_scale_heldout_auc_mean": semantic_auc,
        "geometry_confidence_heldout_auc_mean": geometry_auc,
        "boundary_action_predictive_gate_pass": gate_pass,
        "o3_action_gate_pass": o3_action_gate_pass,
        "predictive_gate_rule": "best heldout AUC >=0.65 and at least +0.05 above semantic_Q_scale and geometry_confidence in the same group unless the best feature is that control itself",
        "o3_action_gate_rule": "predictive gate plus top1 mean delta_ate <= +0.3m and top1 mean best mechanism improvement >= 0.10",
        "note": (
            "Diagnostic-only. Labels come from true single-boundary hold oracle CSVs. "
            "Passing this gate would still require an O3 action-sensitive metric check before O4."
        ),
    }
    (out_dir / "boundary_action_objective_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
