#!/usr/bin/env python3
"""Summarize ACL2 v70 RADIO SWA online smoke jobs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.kitti_trajectory_diagnostics import _load_kitti_gt, _load_tum_prediction  # noqa: E402
from tools.v18_true_action_report import _align_metrics  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v70_geometry_first_semantic_trust/"
    "report_final/phaseR5_radio_swa_online_smoke_r1"
)
DEFAULT_KITTI_GT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
CONTROL_CASES = {
    "geometry_only",
    "label_structure",
    "radio_feature_shuffle",
    "radio_risk_shuffle",
    "radio_component_shuffle",
    "same_cue_random",
}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _to_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"_json_decode_error": True, "_raw": line[:200]})
    return rows


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: _to_jsonable(value) for key, value in row.items()} for row in rows])


def _last_present(rows: List[Mapping[str, Any]], key: str) -> Any:
    for row in reversed(rows):
        if key in row and row.get(key) is not None:
            return row.get(key)
    return None


def _max_nested(rows: List[Mapping[str, Any]], path: Tuple[str, ...]) -> float:
    best = float("nan")
    for row in rows:
        cur: Any = row
        for part in path:
            if not isinstance(cur, Mapping) or part not in cur:
                cur = None
                break
            cur = cur.get(part)
        val = _to_float(cur)
        if math.isfinite(val):
            best = val if not math.isfinite(best) else max(best, val)
    return best


def _has_implemented_path(rows: List[Mapping[str, Any]], path_name: str) -> bool:
    for row in rows:
        trace = row.get("control_trace")
        if not isinstance(trace, Mapping):
            continue
        paths = trace.get("implemented_paths")
        if isinstance(paths, list) and path_name in {str(x) for x in paths}:
            return True
    return False


def _metric_row(
    path: Path,
    gt_poses: np.ndarray,
    gt_pos: np.ndarray,
    *,
    target_start: int,
    target_end: int,
) -> Dict[str, Any]:
    if not path.exists():
        return {
            "trajectory_exists": False,
            "trajectory_rows": 0,
            "ATE_horizon": float("nan"),
            "Rot_horizon": float("nan"),
            "FinalErr_horizon": float("nan"),
            "alignment_scale": float("nan"),
        }
    frames, raw_poses, _ = _load_tum_prediction(path, gt_poses.shape[0])
    aligned, metrics = _align_metrics(frames, raw_poses, gt_poses, gt_pos)
    out: Dict[str, Any] = {
        "trajectory_exists": True,
        "trajectory_rows": int(frames.shape[0]),
        "frame_min": int(frames.min()) if frames.shape[0] else None,
        "frame_max": int(frames.max()) if frames.shape[0] else None,
    }
    out.update(metrics)
    for start, end, label in [(200, 300, "intersection_200_300"), (400, 600, "intersection_400_600")]:
        mask = (frames >= start) & (frames < end)
        if int(mask.sum()) >= 3:
            err = aligned[mask, :3, 3] - gt_pos[frames[mask]]
            out[f"{label}_ATE"] = float(np.sqrt(np.nanmean(np.linalg.norm(err, axis=1) ** 2)))
            out[f"{label}_rows"] = int(mask.sum())
        else:
            out[f"{label}_ATE"] = float("nan")
            out[f"{label}_rows"] = int(mask.sum())
    target_mask = (frames >= int(target_start)) & (frames < int(target_end))
    if int(target_mask.sum()) >= 3:
        err = aligned[target_mask, :3, 3] - gt_pos[frames[target_mask]]
        out["target_chunk_ATE"] = float(np.sqrt(np.nanmean(np.linalg.norm(err, axis=1) ** 2)))
        out["target_chunk_rows"] = int(target_mask.sum())
    else:
        out["target_chunk_ATE"] = float("nan")
        out["target_chunk_rows"] = int(target_mask.sum())
    return out


def _job_row(job: Mapping[str, Any], gt_poses: np.ndarray, gt_pos: np.ndarray) -> Dict[str, Any]:
    out_dir = Path(str(job.get("out_dir", "")))
    hmc_path = Path(str(job.get("hmc_state_hash") or out_dir / "hmc_state_hash.jsonl"))
    traj_path = Path(str(job.get("trajectory") or out_dir / "01.txt"))
    hmc_rows = _read_jsonl(hmc_path)
    read_cfg = _last_present(hmc_rows, "read_path_controls_requested")
    control_trace = _last_present(hmc_rows, "control_trace")
    swa_read_calls = _max_nested(hmc_rows, ("control_trace", "hook_trace_counts", "swa_read"))
    swa_replace_calls = _max_nested(
        hmc_rows,
        ("control_trace", "hook_effect_summary", "swa_read", "num_swa_overlap_source_replace_applied"),
    )
    swa_replace_alpha = _max_nested(
        hmc_rows,
        ("control_trace", "hook_effect_summary", "swa_read", "mean_swa_overlap_source_replace_alpha"),
    )
    swa_replace_alpha_p90 = _max_nested(
        hmc_rows,
        ("control_trace", "hook_effect_summary", "swa_read", "mean_swa_overlap_source_replace_alpha_p90"),
    )
    swa_replace_score = _max_nested(
        hmc_rows,
        ("control_trace", "hook_effect_summary", "swa_read", "mean_swa_overlap_source_replace_score"),
    )
    swa_gate_calls = _max_nested(
        hmc_rows,
        ("control_trace", "hook_effect_summary", "swa_read", "num_swa_overlap_source_gate_applied"),
    )
    swa_gate_mean = _max_nested(
        hmc_rows,
        ("control_trace", "hook_effect_summary", "swa_read", "mean_swa_overlap_source_gate"),
    )
    swa_gate_delta = _max_nested(
        hmc_rows,
        ("control_trace", "hook_effect_summary", "swa_read", "mean_swa_overlap_source_gate_delta"),
    )
    swa_gate_score = _max_nested(
        hmc_rows,
        ("control_trace", "hook_effect_summary", "swa_read", "mean_swa_overlap_source_score"),
    )

    row: Dict[str, Any] = {
        "chunk": int(job.get("chunk", -1)),
        "context_start_chunk": int(job.get("context_start_chunk", job.get("chunk", -1))),
        "context_chunks": int(job.get("context_chunks", 1)),
        "start_frame": int(job.get("start_frame", -1)),
        "end_frame": int(job.get("end_frame", -1)),
        "target_start_frame": int(job.get("target_start_frame", job.get("start_frame", -1))),
        "target_end_frame": int(job.get("target_end_frame", job.get("end_frame", -1))),
        "case": str(job.get("case", "")),
        "returncode": int(job.get("returncode")) if job.get("returncode") is not None else None,
        "skipped": bool(job.get("skipped", False)),
        "duration_sec": _to_float(job.get("duration_sec")),
        "gpu": job.get("gpu"),
        "out_dir": str(out_dir),
        "run_log": str(job.get("run_log") or out_dir / "run.log"),
        "trajectory": str(traj_path),
        "hmc_state_hash": str(hmc_path),
        "read_cue_source_effective_manifest": job.get("read_cue_source_effective"),
        "swa_action_effective_manifest": job.get("swa_action_effective"),
        "swa_overlap_mode_effective_manifest": job.get("swa_overlap_mode_effective"),
        "swa_overlap_source_replace_mode_effective_manifest": job.get("swa_overlap_source_replace_mode_effective"),
        "swa_overlap_source_gate_mode_effective_manifest": job.get("swa_overlap_source_gate_mode_effective"),
        "turn_off_swa_effective_manifest": job.get("turn_off_swa_effective"),
        "hmc_rows": len(hmc_rows),
        "hmc_json_decode_errors": sum(1 for item in hmc_rows if item.get("_json_decode_error")),
        "control_trace_present": isinstance(control_trace, Mapping),
        "swa_read_hook_trace_count_max": swa_read_calls,
        "swa_overlap_source_replace_applied_count_max": swa_replace_calls,
        "swa_overlap_source_replace_alpha_mean_max": swa_replace_alpha,
        "swa_overlap_source_replace_alpha_p90_max": swa_replace_alpha_p90,
        "swa_overlap_source_replace_score_mean_max": swa_replace_score,
        "swa_overlap_source_gate_applied_count_max": swa_gate_calls,
        "swa_overlap_source_gate_mean_max": swa_gate_mean,
        "swa_overlap_source_gate_delta_mean_max": swa_gate_delta,
        "swa_overlap_source_gate_score_mean_max": swa_gate_score,
        "swa_overlap_source_replace_path_observed": _has_implemented_path(hmc_rows, "swa_overlap_source_replace"),
        "swa_overlap_source_gate_path_observed": _has_implemented_path(hmc_rows, "swa_overlap_source_gate"),
        "frame_attention_path_observed": _has_implemented_path(hmc_rows, "frame_attention"),
        "prior_read_cue_source": _last_present(hmc_rows, "prior_read_cue_source"),
        "prior_cue_source_effective": _last_present(hmc_rows, "prior_cue_source_effective"),
        "prior_read_calib_mode": _last_present(hmc_rows, "prior_read_calib_mode"),
        "prior_read_target_mass": _last_present(hmc_rows, "prior_read_target_mass"),
        "prior_read_blend_lambda": _last_present(hmc_rows, "prior_read_blend_lambda"),
        "prior_read_topk_frac": _last_present(hmc_rows, "prior_read_topk_frac"),
        "prior_mean_D_patch": _last_present(hmc_rows, "prior_mean_D_patch"),
        "prior_q90_D_patch": _last_present(hmc_rows, "prior_q90_D_patch"),
        "prior_dynamic_mass_D_gt_050": _last_present(hmc_rows, "prior_dynamic_mass_D_gt_050"),
        "prior_v70_radio_read_available": _last_present(hmc_rows, "prior_v70_radio_read_available"),
        "prior_v70_radio_read_reason": _last_present(hmc_rows, "prior_v70_radio_read_reason"),
        "prior_v70_radio_read_cue_source": _last_present(hmc_rows, "prior_v70_radio_read_cue_source"),
        "prior_v70_radio_read_mode": _last_present(hmc_rows, "prior_v70_radio_read_mode"),
        "prior_v70_radio_read_control": _last_present(hmc_rows, "prior_v70_radio_read_control"),
        "prior_v70_radio_sidecar_path": _last_present(hmc_rows, "prior_v70_radio_sidecar_path"),
        "prior_v70_radio_output_mean": _last_present(hmc_rows, "prior_v70_radio_output_mean"),
        "prior_v70_radio_output_q90": _last_present(hmc_rows, "prior_v70_radio_output_q90"),
        "prior_v70_radio_output_gt050_mass": _last_present(hmc_rows, "prior_v70_radio_output_gt050_mass"),
        "prior_v70_radio_geom_default_mean": _last_present(hmc_rows, "prior_v70_radio_geom_default_mean"),
        "prior_v70_radio_geom_default_q90": _last_present(hmc_rows, "prior_v70_radio_geom_default_q90"),
        "prior_v70_radio_corr_output_risk": _last_present(hmc_rows, "prior_v70_radio_corr_output_risk"),
        "prior_v70_radio_corr_output_interior_static": _last_present(hmc_rows, "prior_v70_radio_corr_output_interior_static"),
        "pass1_pass2_pose_t_mean": _last_present(hmc_rows, "pass1_pass2_pose_t_mean"),
        "pass1_pass2_pose_t_max": _last_present(hmc_rows, "pass1_pass2_pose_t_max"),
        "pass1_pass2_pose_r_deg_mean": _last_present(hmc_rows, "pass1_pass2_pose_r_deg_mean"),
        "pass1_pass2_pose_r_deg_max": _last_present(hmc_rows, "pass1_pass2_pose_r_deg_max"),
        "pass1_pass2_pose_matrix_abs_max": _last_present(hmc_rows, "pass1_pass2_pose_matrix_abs_max"),
    }
    if isinstance(read_cfg, Mapping):
        row.update({
            "requested_read_path": read_cfg.get("read_path"),
            "requested_frame": read_cfg.get("frame"),
            "requested_read_layer_mode": read_cfg.get("read_layer_mode"),
            "requested_swa_overlap_source_replace_alpha": read_cfg.get("swa_overlap_source_replace_alpha"),
            "requested_swa_overlap_source_replace_mode": read_cfg.get("swa_overlap_source_replace_mode"),
            "requested_swa_overlap_source_replace_target": read_cfg.get("swa_overlap_source_replace_target"),
            "requested_swa_overlap_source_replace_layer_mode": read_cfg.get("swa_overlap_source_replace_layer_mode"),
            "requested_swa_overlap_source_gate_rho": read_cfg.get("swa_overlap_source_gate_rho"),
            "requested_swa_overlap_source_gate_min": read_cfg.get("swa_overlap_source_gate_min"),
            "requested_swa_overlap_source_gate_mode": read_cfg.get("swa_overlap_source_gate_mode"),
            "requested_swa_overlap_source_gate_target": read_cfg.get("swa_overlap_source_gate_target"),
            "requested_swa_overlap_source_gate_layer_mode": read_cfg.get("swa_overlap_source_gate_layer_mode"),
            "requested_v70_radio_sidecar_dir": read_cfg.get("v70_radio_sidecar_dir"),
        })
    if isinstance(control_trace, Mapping):
        paths = control_trace.get("implemented_paths")
        row["control_trace_implemented_paths"] = ",".join(str(x) for x in paths) if isinstance(paths, list) else ""

    is_radio_case = str(row["case"]) in {
        "candidate",
        "geometry_only",
        "radio_feature_shuffle",
        "radio_risk_shuffle",
        "radio_component_shuffle",
        "same_cue_random",
    }
    leave_one_out_active = str(job.get("turn_off_swa_effective", "0")).strip().lower() in {"1", "true", "yes", "y"}
    row["hook_active"] = bool(
        row["case"] != "native_no_swa"
        and row["returncode"] == 0
        and row["hmc_rows"] > 0
        and (
            leave_one_out_active
            or
            (
                row["swa_overlap_source_replace_path_observed"]
                and math.isfinite(swa_replace_calls)
                and swa_replace_calls > 0
            )
            or (
                row["swa_overlap_source_gate_path_observed"]
                and math.isfinite(swa_gate_calls)
                and swa_gate_calls > 0
            )
        )
        and (leave_one_out_active or not is_radio_case or row["prior_v70_radio_read_available"] is True)
    )
    try:
        row.update(
            _metric_row(
                traj_path,
                gt_poses,
                gt_pos,
                target_start=int(row["target_start_frame"]),
                target_end=int(row["target_end_frame"]),
            )
        )
    except Exception as exc:
        row.update({
            "trajectory_exists": traj_path.exists(),
            "trajectory_metric_error": f"{type(exc).__name__}:{exc}",
            "ATE_horizon": float("nan"),
            "Rot_horizon": float("nan"),
            "FinalErr_horizon": float("nan"),
            "alignment_scale": float("nan"),
        })
    return row


def _metric_value(row: Mapping[str, Any], metric_name: str) -> float:
    val = _to_float(row.get(metric_name))
    if math.isfinite(val):
        return val
    return _to_float(row.get("ATE_horizon"))


def _gate(rows: List[Dict[str, Any]], min_improvement: float, min_gate_chunks: int, metric_name: str) -> Dict[str, Any]:
    by_chunk_case: Dict[Tuple[int, str], Dict[str, Any]] = {(int(r["chunk"]), str(r["case"])): r for r in rows}
    candidate_chunks: List[int] = sorted({int(r["chunk"]) for r in rows if r.get("case") == "candidate"})
    pass_chunks: List[int] = []
    chunk_details: List[Dict[str, Any]] = []
    for chunk in candidate_chunks:
        native = by_chunk_case.get((chunk, "native_no_swa"))
        cand = by_chunk_case.get((chunk, "candidate"))
        controls = [by_chunk_case[(chunk, case)] for case in sorted(CONTROL_CASES) if (chunk, case) in by_chunk_case]
        native_ate = _metric_value(native, metric_name) if native else float("nan")
        cand_ate = _metric_value(cand, metric_name) if cand else float("nan")
        improvement = native_ate - cand_ate if math.isfinite(native_ate) and math.isfinite(cand_ate) else float("nan")
        finite_controls = [row for row in controls if math.isfinite(_metric_value(row, metric_name))]
        beats_all_controls = bool(
            finite_controls
            and math.isfinite(cand_ate)
            and all(cand_ate < _metric_value(row, metric_name) for row in finite_controls)
        )
        hook_active = bool(cand and cand.get("hook_active"))
        candidate_ok = bool(
            cand
            and cand.get("returncode") == 0
            and hook_active
            and math.isfinite(improvement)
            and improvement >= min_improvement
            and beats_all_controls
        )
        if candidate_ok:
            pass_chunks.append(chunk)
        chunk_details.append({
            "chunk": chunk,
            f"native_{metric_name}": native_ate,
            f"candidate_{metric_name}": cand_ate,
            "candidate_improvement_m": improvement,
            "candidate_hook_active": hook_active,
            "control_cases": [str(row.get("case")) for row in finite_controls],
            f"min_control_{metric_name}": min((_metric_value(row, metric_name) for row in finite_controls), default=float("nan")),
            "candidate_beats_all_controls": beats_all_controls,
            "candidate_pass": candidate_ok,
        })
    failed_jobs = [r for r in rows if r.get("returncode") not in {0, None}]
    hook_active_chunks = sorted({int(r["chunk"]) for r in rows if r.get("case") == "candidate" and r.get("hook_active")})
    return {
        "phase": "ACL2 v70 RADIO SWA online smoke",
        "rows": len(rows),
        "candidate_chunks": candidate_chunks,
        "candidate_hook_active_chunks": hook_active_chunks,
        "candidate_pass_chunks": pass_chunks,
        "min_local_improvement_m": float(min_improvement),
        "min_gate_chunks": int(min_gate_chunks),
        "gate_metric": str(metric_name),
        "swa_online_gate_pass": len(pass_chunks) >= int(min_gate_chunks) and not failed_jobs,
        "failed_jobs": len(failed_jobs),
        "chunk_details": chunk_details,
        "gate_rule": (
            "bounded local-window smoke: candidate must return 0, show SWA intervention evidence, "
            f"improve local ATE vs native by >= {min_improvement} m, and beat all finite controls for the same chunk; "
            f"smoke pass requires >= {min_gate_chunks} passing chunks and no failed jobs. "
            "This smoke does not by itself satisfy the plan's full future-after-overlap gate."
        ),
    }


def _write_markdown(path: Path, rows: List[Dict[str, Any]], summary: Mapping[str, Any]) -> None:
    metric_name = str(summary.get("gate_metric") or "target_chunk_ATE")
    lines = [
        "# ACL2 v70 RADIO SWA Online Smoke",
        "",
        f"Gate pass: `{str(summary.get('swa_online_gate_pass')).lower()}`",
        "",
        f"Rule: {summary.get('gate_rule')}",
        "",
        "## Chunk Gate Details",
        "",
        f"| chunk | native {metric_name} | candidate {metric_name} | improvement m | hook | min control {metric_name} | beats controls | pass |",
        "|---:|---:|---:|---:|---|---:|---|---|",
    ]
    for item in summary.get("chunk_details", []):
        lines.append(
            "| {chunk} | {native:.6g} | {cand:.6g} | {imp:.6g} | `{hook}` | {ctrl:.6g} | `{beats}` | `{passed}` |".format(
                chunk=item.get("chunk"),
                native=_to_float(item.get(f"native_{metric_name}")),
                cand=_to_float(item.get(f"candidate_{metric_name}")),
                imp=_to_float(item.get("candidate_improvement_m")),
                hook=str(bool(item.get("candidate_hook_active"))).lower(),
                ctrl=_to_float(item.get(f"min_control_{metric_name}")),
                beats=str(bool(item.get("candidate_beats_all_controls"))).lower(),
                passed=str(bool(item.get("candidate_pass"))).lower(),
            )
        )
    lines.extend([
        "",
        "## Per-Run Rows",
        "",
        "| chunk | case | rc | hook | target ATE | ATE | Rot | FinalErr | v70 reason | action | mode | calls | strength |",
        "|---:|---|---:|---|---:|---:|---:|---:|---|---|---|---:|---:|",
    ])
    for row in sorted(rows, key=lambda r: (int(r.get("chunk", -1)), str(r.get("case", "")))):
        lines.append(
            "| {chunk} | {case} | {rc} | `{hook}` | {target:.6g} | {ate:.6g} | {rot:.6g} | {final:.6g} | {reason} | {action} | {mode} | {calls:.6g} | {strength:.6g} |".format(
                chunk=row.get("chunk"),
                case=row.get("case"),
                rc=row.get("returncode"),
                hook=str(bool(row.get("hook_active"))).lower(),
                target=_to_float(row.get("target_chunk_ATE")),
                ate=_to_float(row.get("ATE_horizon")),
                rot=_to_float(row.get("Rot_horizon")),
                final=_to_float(row.get("FinalErr_horizon")),
                reason=row.get("prior_v70_radio_read_reason"),
                action=row.get("swa_action_effective_manifest") or "",
                mode=row.get("requested_swa_overlap_source_replace_mode")
                or row.get("requested_swa_overlap_source_gate_mode")
                or row.get("swa_overlap_mode_effective_manifest"),
                calls=max(
                    _to_float(row.get("swa_overlap_source_replace_applied_count_max")),
                    _to_float(row.get("swa_overlap_source_gate_applied_count_max")),
                ),
                strength=max(
                    _to_float(row.get("swa_overlap_source_replace_alpha_mean_max")),
                    _to_float(row.get("swa_overlap_source_gate_delta_mean_max")),
                ),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--kitti-gt", type=Path, default=DEFAULT_KITTI_GT)
    parser.add_argument("--min-local-improvement", type=float, default=0.5)
    parser.add_argument("--min-gate-chunks", type=int, default=4)
    parser.add_argument("--gate-metric", default="target_chunk_ATE")
    args = parser.parse_args()

    output_root = args.output_root
    manifest_path = args.manifest or output_root / "v70_radio_swa_online_manifest.json"
    manifest = _read_json(manifest_path)
    jobs = manifest.get("jobs", [])
    if not isinstance(jobs, list) or not jobs:
        raise SystemExit(f"no jobs found in {manifest_path}")

    _, gt_poses, gt_pos = _load_kitti_gt(args.kitti_gt)
    rows = [_job_row(job, gt_poses, gt_pos) for job in jobs]
    for row in rows:
        native = next((r for r in rows if r.get("chunk") == row.get("chunk") and r.get("case") == "native_no_swa"), None)
        native_ate = _to_float(native.get("ATE_horizon") if native else None)
        row_ate = _to_float(row.get("ATE_horizon"))
        row["ATE_delta_vs_native"] = row_ate - native_ate if math.isfinite(row_ate) and math.isfinite(native_ate) else float("nan")
        row["local_window_improvement_m"] = native_ate - row_ate if math.isfinite(row_ate) and math.isfinite(native_ate) else float("nan")

    summary = _gate(rows, args.min_local_improvement, args.min_gate_chunks, args.gate_metric)
    summary["manifest"] = str(manifest_path)
    summary["output_root"] = str(output_root)
    summary["kitti_gt"] = str(args.kitti_gt)
    summary["cases"] = sorted({str(row.get("case")) for row in rows})
    summary["chunks"] = sorted({int(row.get("chunk")) for row in rows})

    _write_csv(output_root / "radio_swa_online_smoke_results.csv", rows)
    (output_root / "radio_swa_online_smoke_summary.json").write_text(
        json.dumps(_to_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(output_root / "radio_swa_online_smoke_report.md", rows, summary)
    print(json.dumps(_to_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
