#!/usr/bin/env python3
"""Summarize v74-TF RADIO TTT-write online local-window smoke jobs."""

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
    "results/kitti01_hmc_v2/acl2_v74tf_training_free_semantic_memory_control/"
    "phase5_harmful_no_persistent_ttt_dynamic_lowstable_top4"
)
DEFAULT_KITTI_GT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
CONTROL_CASES = {"geometry_only", "spatial_shuffle"}


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


def _last_present(rows: List[Mapping[str, Any]], key: str) -> Any:
    for row in reversed(rows):
        if key in row and row.get(key) is not None:
            return row.get(key)
    return None


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


def _metric_row(path: Path, gt_poses: np.ndarray, gt_pos: np.ndarray, *, target_start: int, target_end: int) -> Dict[str, Any]:
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
    control_trace = _last_present(hmc_rows, "control_trace")
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
        "hmc_rows": len(hmc_rows),
        "hmc_json_decode_errors": sum(1 for item in hmc_rows if item.get("_json_decode_error")),
        "control_trace_present": isinstance(control_trace, Mapping),
        "enable_v70_radio_ttt_write_prior_effective": job.get("enable_v70_radio_ttt_write_prior_effective"),
        "v70_radio_ttt_mode_effective": job.get("v70_radio_ttt_mode_effective"),
        "v70_radio_ttt_control_effective": job.get("v70_radio_ttt_control_effective"),
        "prior_v70_radio_ttt_write_prior_enabled": _last_present(hmc_rows, "prior_v70_radio_ttt_write_prior_enabled"),
        "prior_v70_radio_ttt_write_prior_applied": _last_present(hmc_rows, "prior_v70_radio_ttt_write_prior_applied"),
        "prior_v70_radio_ttt_reason": _last_present(hmc_rows, "prior_v70_radio_ttt_reason"),
        "prior_v70_radio_ttt_mode": _last_present(hmc_rows, "prior_v70_radio_ttt_mode"),
        "prior_v70_radio_ttt_control": _last_present(hmc_rows, "prior_v70_radio_ttt_control"),
        "prior_v70_radio_ttt_changed_patch_frac": _last_present(hmc_rows, "prior_v70_radio_ttt_changed_patch_frac"),
        "prior_v70_radio_ttt_dynamic_lowstable_mask_mean": _last_present(hmc_rows, "prior_v70_radio_ttt_dynamic_lowstable_mask_mean"),
        "prior_v70_radio_ttt_group_prior_relative_change": _last_present(hmc_rows, "prior_v70_radio_ttt_group_prior_relative_change"),
        "prior_v70_radio_ttt_prior_abs_change_ratio": _last_present(hmc_rows, "prior_v70_radio_ttt_prior_abs_change_ratio"),
        "probe_ttt_write_debug_available": _last_present(hmc_rows, "probe_ttt_write_debug_available"),
        "probe_ttt_write_action_delta_norm_mean": _last_present(hmc_rows, "probe_ttt_write_action_delta_norm_mean"),
        "probe_ttt_write_post_delta_norm_mean": _last_present(hmc_rows, "probe_ttt_write_post_delta_norm_mean"),
    }
    row["hook_active"] = bool(
        row["case"] != "native_no_ttt_radio"
        and row["returncode"] == 0
        and row["hmc_rows"] > 0
        and str(row.get("prior_v70_radio_ttt_write_prior_applied")).lower() == "true"
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
            "target_chunk_ATE": float("nan"),
        })
    return row


def _metric_value(row: Mapping[str, Any], metric_name: str) -> float:
    val = _to_float(row.get(metric_name))
    if math.isfinite(val):
        return val
    return _to_float(row.get("ATE_horizon"))


def _gate(rows: List[Dict[str, Any]], min_improvement: float, min_gate_chunks: int, metric_name: str) -> Dict[str, Any]:
    by_chunk_case: Dict[Tuple[int, str], Dict[str, Any]] = {(int(r["chunk"]), str(r["case"])): r for r in rows}
    candidate_chunks = sorted({int(r["chunk"]) for r in rows if r.get("case") == "candidate"})
    pass_chunks: List[int] = []
    chunk_details: List[Dict[str, Any]] = []
    for chunk in candidate_chunks:
        native = by_chunk_case.get((chunk, "native_no_ttt_radio"))
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
        "phase": "ACL2 v74-TF RADIO TTT-write online smoke",
        "rows": len(rows),
        "candidate_chunks": candidate_chunks,
        "candidate_hook_active_chunks": hook_active_chunks,
        "candidate_pass_chunks": pass_chunks,
        "min_local_improvement_m": float(min_improvement),
        "min_gate_chunks": int(min_gate_chunks),
        "gate_metric": str(metric_name),
        "ttt_write_online_gate_pass": len(pass_chunks) >= int(min_gate_chunks) and not failed_jobs,
        "failed_jobs": len(failed_jobs),
        "chunk_details": chunk_details,
        "gate_rule": (
            "bounded local-window smoke: candidate must return 0, apply v70 RADIO TTT-write prior, "
            f"improve local ATE vs native by >= {min_improvement} m, and beat all finite controls for the same chunk; "
            f"smoke pass requires >= {min_gate_chunks} passing chunks and no failed jobs."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--kitti-gt", type=Path, default=DEFAULT_KITTI_GT)
    parser.add_argument("--min-local-improvement", type=float, default=0.5)
    parser.add_argument("--min-gate-chunks", type=int, default=4)
    parser.add_argument("--gate-metric", default="target_chunk_ATE")
    args = parser.parse_args()

    manifest_path = args.manifest or (args.output_root / "v74tf_ttt_write_online_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _, gt_poses, gt_pos = _load_kitti_gt(args.kitti_gt)
    rows = [_job_row(job, gt_poses, gt_pos) for job in manifest.get("jobs", [])]
    by_native = {(int(row["chunk"]), "native_no_ttt_radio"): _metric_value(row, args.gate_metric) for row in rows if row.get("case") == "native_no_ttt_radio"}
    for row in rows:
        native = by_native.get((int(row["chunk"]), "native_no_ttt_radio"))
        current = _metric_value(row, args.gate_metric)
        row["ATE_delta_vs_native"] = current - native if native is not None and math.isfinite(native) and math.isfinite(current) else float("nan")
        row["local_window_improvement_m"] = native - current if native is not None and math.isfinite(native) and math.isfinite(current) else float("nan")
    summary = _gate(rows, args.min_local_improvement, args.min_gate_chunks, args.gate_metric)
    summary.update({
        "manifest": str(manifest_path),
        "output_root": str(args.output_root),
        "kitti_gt": str(args.kitti_gt),
        "chunks": sorted({int(row["chunk"]) for row in rows}),
        "cases": sorted({str(row["case"]) for row in rows}),
    })
    _write_csv(args.output_root / "ttt_write_online_smoke_results.csv", rows)
    (args.output_root / "ttt_write_online_smoke_summary.json").write_text(
        json.dumps(_to_jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_to_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
