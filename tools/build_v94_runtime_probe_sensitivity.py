#!/usr/bin/env python3
"""Summarize v94 Phase3R runtime merge/gauge trajectory probes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.kitti_trajectory_diagnostics import _load_kitti_gt, _rmse, _umeyama_sim3  # noqa: E402


ROOT = Path("results/acl2_v94tf_semantic_gauge_failure_localization_causal_memory_control")
DEFAULT_PROBE = ROOT / "phase3r_runtime_merge_gauge_probe"
DEFAULT_GT_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                rows.append(data)
    return rows


def f(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def pose_positions(
    seq: str,
    record: dict[str, Any],
    gt_cache: dict[str, tuple[np.ndarray, np.ndarray]],
    gt_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    frame_ids = record.get("emitted_frame_ids")
    poses = record.get("camera_poses")
    if not isinstance(frame_ids, list) or not isinstance(poses, list) or not frame_ids or not poses:
        return np.empty(0, dtype=int), np.empty((0, 3)), np.empty((0, 3)), "missing_frame_ids_or_camera_poses"
    if seq not in gt_cache:
        _, gt_poses, gt_pos = _load_kitti_gt(gt_root / f"{seq}.txt")
        gt_cache[seq] = (gt_poses, gt_pos)
    _, gt_pos = gt_cache[seq]
    frames = np.asarray([int(x) for x in frame_ids], dtype=int)
    raw = np.asarray([np.asarray(mat, dtype=float)[:3, 3] for mat in poses], dtype=float)
    if raw.ndim != 2 or raw.shape[0] != frames.shape[0] or raw.shape[1] != 3:
        return np.empty(0, dtype=int), np.empty((0, 3)), np.empty((0, 3)), "bad_pose_shape"
    valid = (frames >= 0) & (frames < gt_pos.shape[0])
    frames = frames[valid]
    raw = raw[valid]
    if frames.size < 2:
        return np.empty(0, dtype=int), np.empty((0, 3)), np.empty((0, 3)), "too_few_valid_frames"
    return frames, raw, gt_pos[frames], ""


def eval_pose_record(
    seq: str,
    record: dict[str, Any],
    gt_cache: dict[str, tuple[np.ndarray, np.ndarray]],
    gt_root: Path,
) -> dict[str, Any]:
    frames, raw, gt, reason = pose_positions(seq, record, gt_cache, gt_root)
    if reason:
        return {"pose_eval_valid": False, "pose_eval_reason": reason}
    scale, rot, trans = _umeyama_sim3(raw, gt, with_scale=True)
    aligned = (scale * (rot @ raw.T)).T + trans[None]
    errors = np.linalg.norm(aligned - gt, axis=1)
    step_raw = np.linalg.norm(np.diff(raw, axis=0), axis=1)
    step_gt = np.linalg.norm(np.diff(gt, axis=0), axis=1)
    ratios = step_raw / np.maximum(step_gt, 1e-12)
    ratios = ratios[np.isfinite(ratios)]
    return {
        "pose_eval_valid": True,
        "pose_eval_reason": "",
        "frame_start": int(frames.min()),
        "frame_end": int(frames.max()),
        "frame_count": int(frames.size),
        "postmerge_sim3_scale": float(scale),
        "postmerge_sim3_rmse": float(_rmse(errors)),
        "postmerge_error_mean": float(np.mean(errors)),
        "postmerge_error_p90": float(np.quantile(errors, 0.90)),
        "postmerge_head_tail_error": float(
            abs(np.linalg.norm(aligned[-1] - aligned[0]) - np.linalg.norm(gt[-1] - gt[0]))
        ),
        "postmerge_scale_cv": float(np.std(ratios) / max(np.mean(ratios), 1e-12)) if ratios.size else "",
    }


def eval_handoff_transfer(
    seq: str,
    prev_record: dict[str, Any],
    curr_record: dict[str, Any],
    gt_cache: dict[str, tuple[np.ndarray, np.ndarray]],
    gt_root: Path,
) -> dict[str, Any]:
    prev_frames, prev_raw, prev_gt, prev_reason = pose_positions(seq, prev_record, gt_cache, gt_root)
    curr_frames, curr_raw, curr_gt, curr_reason = pose_positions(seq, curr_record, gt_cache, gt_root)
    if prev_reason or curr_reason:
        return {
            "handoff_transfer_valid": False,
            "handoff_transfer_reason": f"prev:{prev_reason or 'ok'};curr:{curr_reason or 'ok'}",
        }
    if prev_frames.size < 3 or curr_frames.size < 3:
        return {"handoff_transfer_valid": False, "handoff_transfer_reason": "too_few_frames"}
    scale, rot, trans = _umeyama_sim3(prev_raw, prev_gt, with_scale=True)
    aligned_curr = (scale * (rot @ curr_raw.T)).T + trans[None]
    errors = np.linalg.norm(aligned_curr - curr_gt, axis=1)
    return {
        "handoff_transfer_valid": True,
        "handoff_transfer_reason": "",
        "curr_handoff_transfer_rmse": float(_rmse(errors)),
        "curr_handoff_transfer_error_mean": float(np.mean(errors)),
        "curr_handoff_transfer_error_p90": float(np.quantile(errors, 0.90)),
        "curr_handoff_transfer_head_tail_error": float(
            abs(np.linalg.norm(aligned_curr[-1] - aligned_curr[0]) - np.linalg.norm(curr_gt[-1] - curr_gt[0]))
        ),
        "prev_fit_transfer_scale": float(scale),
    }


def trace_by_chunk(path: Path) -> dict[int, dict[str, Any]]:
    rows = read_jsonl(path)
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            out[int(row.get("chunk_idx"))] = row
        except (TypeError, ValueError):
            continue
    return out


def score_row(native: dict[str, Any], probe: dict[str, Any]) -> dict[str, Any]:
    components = [
        ("handoff_transfer", "curr_handoff_transfer_rmse", 0.35),
        ("scale", "abs_log_scale_jump_runtime", 0.25),
        ("boundary", "boundary_update_norm", 0.25),
        ("residual", "merge_residual_after_abs", 0.15),
    ]
    native_score = 0.0
    probe_score = 0.0
    weight_sum = 0.0
    used: list[str] = []
    component_improvements: dict[str, Any] = {}
    for name, key, weight in components:
        n = f(native.get(key))
        p = f(probe.get(key))
        if not math.isfinite(n) or not math.isfinite(p):
            component_improvements[f"{name}_improvement_ratio"] = ""
            continue
        denom = max(abs(n), 1e-9)
        native_score += weight * (n / denom)
        probe_score += weight * (p / denom)
        weight_sum += weight
        used.append(name)
        component_improvements[f"{name}_improvement_ratio"] = float((n - p) / denom)
    if weight_sum <= 0:
        return {
            "J_handoff_runtime_proxy_native": "",
            "J_handoff_runtime_proxy_probe": "",
            "I_J_runtime_proxy": "",
            "W_good_runtime_proxy": "",
            "runtime_proxy_components_used": "",
            **component_improvements,
        }
    native_norm = native_score / weight_sum
    probe_norm = probe_score / weight_sum
    return {
        "J_handoff_runtime_proxy_native": native_norm,
        "J_handoff_runtime_proxy_probe": probe_norm,
        "I_J_runtime_proxy": float((native_norm - probe_norm) / max(abs(native_norm), 1e-9)),
        "W_good_runtime_proxy": float((probe_norm - native_norm) / max(abs(native_norm), 1e-9)),
        "runtime_proxy_components_used": ";".join(used),
        **component_improvements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--candidate-variants", default="merge_boundary_hold")
    parser.add_argument("--candidate-mode", choices=["first", "best"], default="first")
    args = parser.parse_args()
    manifest = read_json(args.probe_root / "runtime_probe_manifest.json")
    gt_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    metric_rows: list[dict[str, Any]] = []
    for job in manifest.get("jobs", []):
        out_dir = Path(job["out_dir"])
        seq = str(job["seq"]).zfill(2)
        prev = int(job["prev_chunk"])
        curr = int(job["curr_chunk"])
        pose_rows = read_jsonl(out_dir / "postmerge_global_pose.jsonl")
        trace_rows = trace_by_chunk(out_dir / "merge_state_trace.jsonl")
        eval_by_chunk: dict[int, dict[str, Any]] = {}
        for record in pose_rows:
            try:
                chunk = int(record.get("chunk_idx"))
            except (TypeError, ValueError):
                continue
            eval_by_chunk[chunk] = eval_pose_record(seq, record, gt_cache, args.gt_root)
        records_by_chunk = {
            int(record.get("chunk_idx")): record
            for record in pose_rows
            if isinstance(record, dict) and str(record.get("chunk_idx", "")).lstrip("-").isdigit()
        }
        prev_eval = eval_by_chunk.get(prev, {})
        curr_eval = eval_by_chunk.get(curr, {})
        handoff_eval = {}
        if prev in records_by_chunk and curr in records_by_chunk:
            handoff_eval = eval_handoff_transfer(seq, records_by_chunk[prev], records_by_chunk[curr], gt_cache, args.gt_root)
        prev_scale = f(prev_eval.get("postmerge_sim3_scale"))
        curr_scale = f(curr_eval.get("postmerge_sim3_scale"))
        abs_log_scale_jump = (
            abs(math.log(max(curr_scale, 1e-12)) - math.log(max(prev_scale, 1e-12)))
            if math.isfinite(prev_scale) and math.isfinite(curr_scale) and prev_scale > 0 and curr_scale > 0
            else float("nan")
        )
        trace = trace_rows.get(curr, {})
        residual_after = f(trace.get("merge_residual_after"))
        row = {
            "pair_id": job.get("pair_id"),
            "seq": seq,
            "prev_chunk": prev,
            "curr_chunk": curr,
            "case_label_offline_only": job.get("case_label_offline_only"),
            "failure_type_primary": job.get("failure_type_primary"),
            "failure_type_secondary": job.get("failure_type_secondary"),
            "probe_selection_tag": job.get("probe_selection_tag"),
            "variant": job.get("variant"),
            "returncode": job.get("returncode"),
            "duration_sec": job.get("duration_sec"),
            "out_dir": job.get("out_dir"),
            "curr_pose_eval_valid": curr_eval.get("pose_eval_valid", False),
            "curr_pose_eval_reason": curr_eval.get("pose_eval_reason", ""),
            "curr_postmerge_sim3_rmse": curr_eval.get("postmerge_sim3_rmse", ""),
            "curr_postmerge_error_p90": curr_eval.get("postmerge_error_p90", ""),
            "curr_postmerge_head_tail_error": curr_eval.get("postmerge_head_tail_error", ""),
            "handoff_transfer_valid": handoff_eval.get("handoff_transfer_valid", False),
            "handoff_transfer_reason": handoff_eval.get("handoff_transfer_reason", ""),
            "curr_handoff_transfer_rmse": handoff_eval.get("curr_handoff_transfer_rmse", ""),
            "curr_handoff_transfer_error_p90": handoff_eval.get("curr_handoff_transfer_error_p90", ""),
            "curr_handoff_transfer_head_tail_error": handoff_eval.get("curr_handoff_transfer_head_tail_error", ""),
            "prev_fit_transfer_scale": handoff_eval.get("prev_fit_transfer_scale", ""),
            "prev_postmerge_sim3_rmse": prev_eval.get("postmerge_sim3_rmse", ""),
            "prev_postmerge_sim3_scale": prev_eval.get("postmerge_sim3_scale", ""),
            "curr_postmerge_sim3_scale": curr_eval.get("postmerge_sim3_scale", ""),
            "abs_log_scale_jump_runtime": abs_log_scale_jump if math.isfinite(abs_log_scale_jump) else "",
            "boundary_update_norm": trace.get("boundary_update_norm", ""),
            "boundary_update_scale_component": trace.get("boundary_update_scale_component", ""),
            "transform_scale_value": trace.get("transform_scale_value", ""),
            "merge_residual_before": trace.get("merge_residual_before", ""),
            "merge_residual_after": trace.get("merge_residual_after", ""),
            "merge_residual_delta": trace.get("merge_residual_delta", ""),
            "merge_residual_after_abs": abs(residual_after) if math.isfinite(residual_after) else "",
            "carrier_state_hash": trace.get("state_hash", ""),
            "forced_merge_state_replay": trace.get("forced_merge_state_replay", ""),
            "online_semantic_merge_controller": trace.get("online_semantic_merge_controller", ""),
            "trace_provenance": trace.get("trace_provenance", ""),
        }
        metric_rows.append(row)

    by_key = {(row["pair_id"], row["variant"]): row for row in metric_rows}
    effect_rows: list[dict[str, Any]] = []
    for row in metric_rows:
        if row["variant"] == "native_actual":
            continue
        native = by_key.get((row["pair_id"], "native_actual"))
        if not native:
            continue
        effect = {
            "pair_id": row["pair_id"],
            "seq": row["seq"],
            "prev_chunk": row["prev_chunk"],
            "curr_chunk": row["curr_chunk"],
            "case_label_offline_only": row["case_label_offline_only"],
            "failure_type_primary": row["failure_type_primary"],
            "failure_type_secondary": row["failure_type_secondary"],
            "probe_selection_tag": row["probe_selection_tag"],
            "variant": row["variant"],
            "native_curr_postmerge_sim3_rmse": native.get("curr_postmerge_sim3_rmse"),
            "probe_curr_postmerge_sim3_rmse": row.get("curr_postmerge_sim3_rmse"),
            "native_curr_handoff_transfer_rmse": native.get("curr_handoff_transfer_rmse"),
            "probe_curr_handoff_transfer_rmse": row.get("curr_handoff_transfer_rmse"),
            "native_abs_log_scale_jump_runtime": native.get("abs_log_scale_jump_runtime"),
            "probe_abs_log_scale_jump_runtime": row.get("abs_log_scale_jump_runtime"),
            "native_boundary_update_norm": native.get("boundary_update_norm"),
            "probe_boundary_update_norm": row.get("boundary_update_norm"),
            "carrier_state_delta": f(native.get("boundary_update_norm")) - f(row.get("boundary_update_norm"))
            if math.isfinite(f(native.get("boundary_update_norm"))) and math.isfinite(f(row.get("boundary_update_norm")))
            else "",
            "native_merge_residual_after_abs": native.get("merge_residual_after_abs"),
            "probe_merge_residual_after_abs": row.get("merge_residual_after_abs"),
        }
        effect.update(score_row(native, row))
        effect_rows.append(effect)

    summary_rows: list[dict[str, Any]] = []
    effects = pd.DataFrame(effect_rows)
    if len(effects):
        numeric_i = pd.to_numeric(effects["I_J_runtime_proxy"], errors="coerce")
        numeric_w = pd.to_numeric(effects["W_good_runtime_proxy"], errors="coerce")
        effects["_I"] = numeric_i
        effects["_W"] = numeric_w
        for variant, group in effects.groupby("variant", sort=True):
            labelled_bad = group[group["case_label_offline_only"].astype(str).eq("bad")]
            labelled_good = group[group["case_label_offline_only"].astype(str).eq("good")]
            carrier_delta = pd.to_numeric(group["carrier_state_delta"], errors="coerce")
            row = {
                "variant": variant,
                "row_count": int(len(group)),
                "sequence_coverage": int(group["seq"].nunique()),
                "bad_rows": int(len(labelled_bad)),
                "good_rows": int(len(labelled_good)),
                "bad_median_I_J_runtime_proxy": float(labelled_bad["_I"].median()) if len(labelled_bad["_I"].dropna()) else "",
                "good_median_worsen_runtime_proxy": float(labelled_good["_W"].median()) if len(labelled_good["_W"].dropna()) else "",
                "good_max_worsen_runtime_proxy": float(labelled_good["_W"].max()) if len(labelled_good["_W"].dropna()) else "",
                "carrier_state_delta_nonzero_rows": int((carrier_delta.abs() > 1e-9).sum()),
                "runtime_probe_trajectory_available_rows": int(
                    pd.to_numeric(group["probe_curr_postmerge_sim3_rmse"], errors="coerce").notna().sum()
                ),
                "handoff_transfer_available_rows": int(
                    pd.to_numeric(group["probe_curr_handoff_transfer_rmse"], errors="coerce").notna().sum()
                ),
            }
            bad_i = f(row["bad_median_I_J_runtime_proxy"])
            good_med = f(row["good_median_worsen_runtime_proxy"])
            good_max = f(row["good_max_worsen_runtime_proxy"])
            row["bad_improvement_gate_ge_0p05"] = math.isfinite(bad_i) and bad_i >= 0.05
            row["good_median_worsen_gate_le_0p02"] = math.isfinite(good_med) and good_med <= 0.02
            row["good_catastrophic_worsen_absent_le_0p02"] = math.isfinite(good_max) and good_max <= 0.02
            row["sequence_coverage_ge_3"] = int(row["sequence_coverage"]) >= 3
            row["trajectory_rows_complete"] = int(row["runtime_probe_trajectory_available_rows"]) == int(row["row_count"])
            row["handoff_transfer_rows_complete"] = int(row["handoff_transfer_available_rows"]) == int(row["row_count"])
            summary_rows.append(row)

    summary_by_variant = {row["variant"]: row for row in summary_rows}
    control_variants = [row for row in summary_rows if row["variant"] in {"merge_no_refresh", "merge_robust_native_only", "native_no_semantic_merge"}]
    best_control_bad = max(
        [f(row.get("bad_median_I_J_runtime_proxy")) for row in control_variants if math.isfinite(f(row.get("bad_median_I_J_runtime_proxy")))],
        default=float("nan"),
    )
    candidate_names = [item.strip() for item in args.candidate_variants.split(",") if item.strip()]
    candidate_rows = [summary_by_variant[name] for name in candidate_names if name in summary_by_variant]
    if args.candidate_mode == "best" and candidate_rows:
        actual = max(candidate_rows, key=lambda row: f(row.get("bad_median_I_J_runtime_proxy")) if math.isfinite(f(row.get("bad_median_I_J_runtime_proxy"))) else -float("inf"))
    else:
        actual = candidate_rows[0] if candidate_rows else summary_by_variant.get("merge_boundary_hold", {})
    actual_bad = f(actual.get("bad_median_I_J_runtime_proxy"))
    beats_control = math.isfinite(actual_bad) and math.isfinite(best_control_bad) and actual_bad > best_control_bad
    actual_pass = (
        bool(actual.get("bad_improvement_gate_ge_0p05"))
        and bool(actual.get("good_median_worsen_gate_le_0p02"))
        and bool(actual.get("good_catastrophic_worsen_absent_le_0p02"))
        and bool(actual.get("sequence_coverage_ge_3"))
        and bool(actual.get("trajectory_rows_complete"))
        and bool(actual.get("handoff_transfer_rows_complete"))
        and beats_control
    )

    phase3 = read_json(ROOT / "phase3_neutral_causal_sensitivity/phase3_gate_summary.json")
    summary = {
        "phase": "Phase3R_runtime_merge_gauge_probe_sensitivity",
        "runtime_probe_executed": bool(manifest.get("all_completed")) and bool(metric_rows),
        "runtime_probe_job_count": manifest.get("job_count"),
        "runtime_probe_failed_count": manifest.get("failed_count"),
        "target_count": manifest.get("target_count"),
        "metric_row_count": len(metric_rows),
        "effect_row_count": len(effect_rows),
        "variant_count": len(summary_rows),
        "expanded_gauge_candidate_count": int(
            sum(1 for target in manifest.get("targets", []) if target.get("probe_selection_tag") in {"handoff_gauge_primary", "gauge_candidate_expanded_boundary_top"})
        ),
        "original_phase3_gate_pass": phase3.get("phase3_gate_pass"),
        "original_balanced_probe_gate_pass": (phase3.get("balanced_probe") or {}).get("balanced_probe_gate_pass"),
        "phase3r_runtime_probe_gate_pass": bool(actual_pass),
        "candidate_variants": candidate_names,
        "candidate_mode": args.candidate_mode,
        "selected_candidate_variant": actual.get("variant", ""),
        "merge_boundary_hold_beats_control": beats_control,
        "selected_candidate_beats_control": beats_control,
        "best_control_bad_median_I_J_runtime_proxy": best_control_bad if math.isfinite(best_control_bad) else None,
        "actual_merge_boundary_hold": actual,
        "selected_candidate_summary": actual,
        "blocker": ""
        if actual_pass
        else "runtime_probe_gate_failed_or_original_balanced_probe_still_failed",
        "interpretation_boundary": (
            "This is a Phase3 repair diagnostic using measured postmerge trajectories. "
            "It does not rewrite the original Phase3 balanced-probe gate unless the measured "
            "runtime probe also satisfies good protection and control beating."
        ),
        "runtime_action_allowed": False,
        "counterfactual_allowed": False,
        "ttt_allowed": False,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "runtime_probe_metric_rows.csv", metric_rows)
    write_csv(args.out_dir / "runtime_probe_effect_rows.csv", effect_rows)
    write_csv(args.out_dir / "runtime_probe_variant_summary.csv", summary_rows)
    write_json(args.out_dir / "runtime_probe_sensitivity_summary.json", summary)
    print(f"runtime_probe_executed={summary['runtime_probe_executed']}")
    print(f"phase3r_runtime_probe_gate_pass={summary['phase3r_runtime_probe_gate_pass']}")
    print(f"metric_row_count={summary['metric_row_count']}")
    print(f"effect_row_count={summary['effect_row_count']}")
    print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
