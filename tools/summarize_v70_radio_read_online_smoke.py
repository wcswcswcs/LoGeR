#!/usr/bin/env python3
"""Summarize ACL2 v70 RADIO READ online smoke jobs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.kitti_trajectory_diagnostics import _load_kitti_gt, _load_tum_prediction  # noqa: E402
from tools.v18_true_action_report import _align_metrics  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v70_geometry_first_semantic_trust/"
    "report_final/phaseR5_radio_read_online_smoke_r3"
)
DEFAULT_KITTI_GT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
CONTROL_CASES = {
    "geometry_only",
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


def _has_frame_attention(rows: List[Mapping[str, Any]]) -> bool:
    max_count = _max_nested(rows, ("control_trace", "hook_trace_counts", "frame_attention"))
    if math.isfinite(max_count) and max_count > 0:
        return True
    for row in rows:
        trace = row.get("control_trace")
        if not isinstance(trace, Mapping):
            continue
        paths = trace.get("implemented_paths")
        if isinstance(paths, list) and "frame_attention" in paths:
            return True
    return False


def _metric_row(path: Path, gt_poses: np.ndarray, gt_pos: np.ndarray) -> Dict[str, Any]:
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
    return out


def _job_row(job: Mapping[str, Any], gt_poses: np.ndarray, gt_pos: np.ndarray) -> Dict[str, Any]:
    out_dir = Path(str(job.get("out_dir", "")))
    hmc_path = Path(str(job.get("hmc_state_hash") or out_dir / "hmc_state_hash.jsonl"))
    traj_path = Path(str(job.get("trajectory") or out_dir / "01.txt"))
    hmc_rows = _read_jsonl(hmc_path)
    read_cfg = _last_present(hmc_rows, "read_path_controls_requested")
    control_trace = _last_present(hmc_rows, "control_trace")
    frame_attention_calls = _max_nested(hmc_rows, ("control_trace", "hook_trace_counts", "frame_attention"))
    frame_attention_bias = _max_nested(hmc_rows, ("control_trace", "hook_effect_summary", "frame_attention", "mean_abs_bias"))

    row: Dict[str, Any] = {
        "chunk": int(job.get("chunk", -1)),
        "start_frame": int(job.get("start_frame", -1)),
        "end_frame": int(job.get("end_frame", -1)),
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
        "hmc_rows": len(hmc_rows),
        "hmc_json_decode_errors": sum(1 for item in hmc_rows if item.get("_json_decode_error")),
        "control_trace_present": isinstance(control_trace, Mapping),
        "frame_attention_hook_trace_count_max": frame_attention_calls,
        "frame_attention_mean_abs_bias_max": frame_attention_bias,
        "frame_attention_path_observed": _has_frame_attention(hmc_rows),
        "prior_read_cue_source": _last_present(hmc_rows, "prior_read_cue_source"),
        "prior_cue_source_effective": _last_present(hmc_rows, "prior_cue_source_effective"),
        "prior_frame_bias_mode": _last_present(hmc_rows, "prior_frame_bias_mode"),
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
            "requested_beta_frame": read_cfg.get("beta_frame"),
            "requested_v70_radio_sidecar_dir": read_cfg.get("v70_radio_sidecar_dir"),
        })
    if isinstance(control_trace, Mapping):
        paths = control_trace.get("implemented_paths")
        row["control_trace_implemented_paths"] = ",".join(str(x) for x in paths) if isinstance(paths, list) else ""
    row["hook_active"] = bool(
        row["case"] != "native_no_read"
        and row["returncode"] == 0
        and row["hmc_rows"] > 0
        and row["frame_attention_path_observed"]
        and (not math.isfinite(frame_attention_calls) or frame_attention_calls > 0)
        and row["prior_v70_radio_read_available"] is True
    )
    try:
        row.update(_metric_row(traj_path, gt_poses, gt_pos))
    except Exception as exc:  # keep failed jobs auditable instead of hiding them
        row.update({
            "trajectory_exists": traj_path.exists(),
            "trajectory_metric_error": f"{type(exc).__name__}:{exc}",
            "ATE_horizon": float("nan"),
            "Rot_horizon": float("nan"),
            "FinalErr_horizon": float("nan"),
            "alignment_scale": float("nan"),
        })
    return row


def _gate(rows: List[Dict[str, Any]], min_improvement: float, min_gate_chunks: int) -> Dict[str, Any]:
    by_chunk_case: Dict[Tuple[int, str], Dict[str, Any]] = {(int(r["chunk"]), str(r["case"])): r for r in rows}
    candidate_chunks: List[int] = sorted({int(r["chunk"]) for r in rows if r.get("case") == "candidate"})
    pass_chunks: List[int] = []
    chunk_details: List[Dict[str, Any]] = []
    for chunk in candidate_chunks:
        native = by_chunk_case.get((chunk, "native_no_read"))
        cand = by_chunk_case.get((chunk, "candidate"))
        controls = [by_chunk_case[(chunk, case)] for case in sorted(CONTROL_CASES) if (chunk, case) in by_chunk_case]
        native_ate = _to_float(native.get("ATE_horizon") if native else None)
        cand_ate = _to_float(cand.get("ATE_horizon") if cand else None)
        improvement = native_ate - cand_ate if math.isfinite(native_ate) and math.isfinite(cand_ate) else float("nan")
        finite_controls = [row for row in controls if math.isfinite(_to_float(row.get("ATE_horizon")))]
        beats_all_controls = bool(
            finite_controls
            and math.isfinite(cand_ate)
            and all(cand_ate < _to_float(row.get("ATE_horizon")) for row in finite_controls)
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
            "native_ATE_horizon": native_ate,
            "candidate_ATE_horizon": cand_ate,
            "candidate_improvement_m": improvement,
            "candidate_hook_active": hook_active,
            "control_cases": [str(row.get("case")) for row in finite_controls],
            "min_control_ATE_horizon": min((_to_float(row.get("ATE_horizon")) for row in finite_controls), default=float("nan")),
            "candidate_beats_all_controls": beats_all_controls,
            "candidate_pass": candidate_ok,
        })
    failed_jobs = [r for r in rows if r.get("returncode") not in {0, None}]
    hook_active_chunks = sorted({int(r["chunk"]) for r in rows if r.get("case") == "candidate" and r.get("hook_active")})
    return {
        "phase": "ACL2 v70 RADIO READ online smoke",
        "rows": len(rows),
        "candidate_chunks": candidate_chunks,
        "candidate_hook_active_chunks": hook_active_chunks,
        "candidate_pass_chunks": pass_chunks,
        "min_local_improvement_m": float(min_improvement),
        "min_gate_chunks": int(min_gate_chunks),
        "read_online_gate_pass": len(pass_chunks) >= int(min_gate_chunks) and not failed_jobs,
        "failed_jobs": len(failed_jobs),
        "chunk_details": chunk_details,
        "gate_rule": (
            "candidate must return 0, have v70 RADIO frame-attention hook evidence, improve local "
            f"ATE vs native by >= {min_improvement} m, and beat all finite controls for the same chunk; "
            f"smoke pass requires >= {min_gate_chunks} passing chunks and no failed jobs"
        ),
    }


def _write_markdown(path: Path, rows: List[Dict[str, Any]], summary: Mapping[str, Any]) -> None:
    lines = [
        "# ACL2 v70 RADIO READ Online Smoke",
        "",
        f"Gate pass: `{str(summary.get('read_online_gate_pass')).lower()}`",
        "",
        f"Rule: {summary.get('gate_rule')}",
        "",
        "## Chunk Gate Details",
        "",
        "| chunk | native ATE | candidate ATE | improvement m | hook | min control ATE | beats controls | pass |",
        "|---:|---:|---:|---:|---|---:|---|---|",
    ]
    for item in summary.get("chunk_details", []):
        lines.append(
            "| {chunk} | {native:.6g} | {cand:.6g} | {imp:.6g} | `{hook}` | {ctrl:.6g} | `{beats}` | `{passed}` |".format(
                chunk=item.get("chunk"),
                native=_to_float(item.get("native_ATE_horizon")),
                cand=_to_float(item.get("candidate_ATE_horizon")),
                imp=_to_float(item.get("candidate_improvement_m")),
                hook=str(bool(item.get("candidate_hook_active"))).lower(),
                ctrl=_to_float(item.get("min_control_ATE_horizon")),
                beats=str(bool(item.get("candidate_beats_all_controls"))).lower(),
                passed=str(bool(item.get("candidate_pass"))).lower(),
            )
        )
    lines.extend([
        "",
        "## Per-Run Rows",
        "",
        "| chunk | case | rc | hook | ATE | Rot | FinalErr | v70 reason | v70 control | frame attn calls |",
        "|---:|---|---:|---|---:|---:|---:|---|---|---:|",
    ])
    for row in sorted(rows, key=lambda r: (int(r.get("chunk", -1)), str(r.get("case", "")))):
        lines.append(
            "| {chunk} | {case} | {rc} | `{hook}` | {ate:.6g} | {rot:.6g} | {final:.6g} | {reason} | {control} | {calls:.6g} |".format(
                chunk=row.get("chunk"),
                case=row.get("case"),
                rc=row.get("returncode"),
                hook=str(bool(row.get("hook_active"))).lower(),
                ate=_to_float(row.get("ATE_horizon")),
                rot=_to_float(row.get("Rot_horizon")),
                final=_to_float(row.get("FinalErr_horizon")),
                reason=row.get("prior_v70_radio_read_reason"),
                control=row.get("prior_v70_radio_read_control"),
                calls=_to_float(row.get("frame_attention_hook_trace_count_max")),
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
    args = parser.parse_args()

    output_root = args.output_root
    manifest_path = args.manifest or output_root / "v70_radio_read_online_manifest.json"
    manifest = _read_json(manifest_path)
    jobs = manifest.get("jobs", [])
    if not isinstance(jobs, list) or not jobs:
        raise SystemExit(f"no jobs found in {manifest_path}")

    _, gt_poses, gt_pos = _load_kitti_gt(args.kitti_gt)
    rows = [_job_row(job, gt_poses, gt_pos) for job in jobs]
    for row in rows:
        native = next((r for r in rows if r.get("chunk") == row.get("chunk") and r.get("case") == "native_no_read"), None)
        native_ate = _to_float(native.get("ATE_horizon") if native else None)
        row_ate = _to_float(row.get("ATE_horizon"))
        row["ATE_delta_vs_native"] = row_ate - native_ate if math.isfinite(row_ate) and math.isfinite(native_ate) else float("nan")
        row["local_window_improvement_m"] = native_ate - row_ate if math.isfinite(row_ate) and math.isfinite(native_ate) else float("nan")

    summary = _gate(rows, args.min_local_improvement, args.min_gate_chunks)
    summary["manifest"] = str(manifest_path)
    summary["output_root"] = str(output_root)
    summary["kitti_gt"] = str(args.kitti_gt)
    summary["cases"] = sorted({str(row.get("case")) for row in rows})
    summary["chunks"] = sorted({int(row.get("chunk")) for row in rows})

    _write_csv(output_root / "radio_read_online_smoke_results.csv", rows)
    (output_root / "radio_read_online_smoke_summary.json").write_text(
        json.dumps(_to_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(output_root / "radio_read_online_smoke_report.md", rows, summary)
    print(json.dumps(_to_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
