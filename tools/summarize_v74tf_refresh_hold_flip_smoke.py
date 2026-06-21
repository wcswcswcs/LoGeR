#!/usr/bin/env python3
"""Summarize v74-TF refresh/hold flip online merge smoke jobs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.kitti_trajectory_diagnostics import _load_kitti_gt, _load_tum_prediction  # noqa: E402
from tools.v18_true_action_report import _align_metrics  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v74tf_training_free_semantic_memory_control/"
    "phase5_refresh_hold_flip_radio_qscale_holdalpha005_top4"
)
DEFAULT_KITTI_GT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
CONTROL_CASES = {"geometry_only", "radio_qscale_random", "radio_qscale_shuffled"}


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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
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
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: _to_jsonable(value) for key, value in row.items()} for row in rows])


def _finite(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        val = _to_float(value)
        if math.isfinite(val):
            out.append(val)
    return out


def _finite_mean(values: Iterable[Any]) -> float:
    vals = _finite(values)
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _finite_median(values: Iterable[Any]) -> float:
    vals = _finite(values)
    return float(statistics.median(vals)) if vals else float("nan")


def _finite_min(values: Iterable[Any]) -> float:
    vals = _finite(values)
    return float(min(vals)) if vals else float("nan")


def _finite_max(values: Iterable[Any]) -> float:
    vals = _finite(values)
    return float(max(vals)) if vals else float("nan")


def _metric_row(path: Path, gt_poses: np.ndarray, gt_pos: np.ndarray, *, target_start: int, target_end: int) -> dict[str, Any]:
    if not path.exists():
        return {
            "trajectory_exists": False,
            "trajectory_rows": 0,
            "ATE_horizon": float("nan"),
            "Rot_horizon": float("nan"),
            "FinalErr_horizon": float("nan"),
            "alignment_scale": float("nan"),
            "target_chunk_ATE": float("nan"),
            "target_chunk_rows": 0,
        }
    frames, raw_poses, _ = _load_tum_prediction(path, gt_poses.shape[0])
    aligned, metrics = _align_metrics(frames, raw_poses, gt_poses, gt_pos)
    out: dict[str, Any] = {
        "trajectory_exists": True,
        "trajectory_rows": int(frames.shape[0]),
        "frame_min": int(frames.min()) if frames.shape[0] else None,
        "frame_max": int(frames.max()) if frames.shape[0] else None,
    }
    out.update(metrics)
    mask = (frames >= int(target_start)) & (frames < int(target_end))
    if int(mask.sum()) >= 3:
        err = aligned[mask, :3, 3] - gt_pos[frames[mask]]
        out["target_chunk_ATE"] = float(np.sqrt(np.nanmean(np.linalg.norm(err, axis=1) ** 2)))
        out["target_chunk_rows"] = int(mask.sum())
    else:
        out["target_chunk_ATE"] = float("nan")
        out["target_chunk_rows"] = int(mask.sum())
    return out


def _trace_summary(trace_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    q_enabled = [
        row
        for row in trace_rows
        if str(row.get("semantic_merge_qscale_hold_refresh_enabled", "")).strip().lower()
        in {"1", "true", "yes"}
    ]
    return {
        "merge_trace_rows": int(len(trace_rows)),
        "merge_trace_json_decode_errors": int(sum(1 for row in trace_rows if row.get("_json_decode_error"))),
        "semantic_merge_qscale_hold_refresh_enabled_any": bool(q_enabled),
        "semantic_merge_qscale_factor_min": _finite_min(row.get("semantic_merge_qscale_factor") for row in trace_rows),
        "semantic_merge_qscale_factor_median": _finite_median(row.get("semantic_merge_qscale_factor") for row in trace_rows),
        "semantic_merge_qscale_factor_max": _finite_max(row.get("semantic_merge_qscale_factor") for row in trace_rows),
        "semantic_merge_qscale_effective_blend_alpha_min": _finite_min(
            row.get("semantic_merge_qscale_effective_blend_alpha") for row in trace_rows
        ),
        "semantic_merge_qscale_effective_blend_alpha_median": _finite_median(
            row.get("semantic_merge_qscale_effective_blend_alpha") for row in trace_rows
        ),
        "semantic_merge_qscale_effective_blend_alpha_max": _finite_max(
            row.get("semantic_merge_qscale_effective_blend_alpha") for row in trace_rows
        ),
        "semantic_merge_radio_handoff_qscale_observability_mean": _finite_mean(
            row.get("semantic_merge_radio_handoff_qscale_observability") for row in trace_rows
        ),
        "semantic_merge_radio_component_available_any": any(
            str(row.get("semantic_merge_radio_component_available", "")).strip().lower() in {"1", "true", "yes"}
            for row in trace_rows
        ),
        "semantic_merge_overlap_residual_mean": _finite_mean(row.get("semantic_merge_overlap_residual") for row in trace_rows),
        "semantic_merge_scale_median": _finite_median(row.get("semantic_merge_scale") for row in trace_rows),
        "semantic_merge_fit_reasons": ",".join(
            sorted({str(row.get("semantic_merge_fit_reason")) for row in trace_rows if row.get("semantic_merge_fit_reason")})
        ),
    }


def _target_window(job: Mapping[str, Any], chunk_size: int) -> tuple[int, int]:
    end = int(job.get("window_end", 0) or 0)
    start = max(0, end - int(chunk_size))
    return start, end


def _job_row(job: Mapping[str, Any], gt_poses: np.ndarray, gt_pos: np.ndarray, *, chunk_size: int) -> dict[str, Any]:
    out_dir = Path(str(job.get("out_dir", "")))
    traj_path = Path(str(job.get("trajectory") or out_dir / "01.txt"))
    trace_path = Path(str(job.get("merge_state_trace") or out_dir / "merge_state_trace.jsonl"))
    target_start, target_end = _target_window(job, chunk_size)
    trace_rows = _read_jsonl(trace_path)
    row: dict[str, Any] = {
        "chunk": int(job.get("chunk", -1)),
        "window_start": int(job.get("window_start", -1)),
        "window_end": int(job.get("window_end", -1)),
        "target_start_frame": int(target_start),
        "target_end_frame": int(target_end),
        "case": str(job.get("case", "")),
        "strategy": str(job.get("strategy", "")),
        "returncode": int(job.get("returncode")) if job.get("returncode") is not None else None,
        "duration_sec": _to_float(job.get("duration_sec")),
        "gpu": job.get("gpu"),
        "out_dir": str(out_dir),
        "run_log": str(job.get("run_log") or out_dir / "run.log"),
        "trajectory": str(traj_path),
        "merge_state_trace": str(trace_path),
    }
    row.update(_trace_summary(trace_rows))
    row["hook_active"] = bool(
        row["case"] == "radio_qscale"
        and row["returncode"] == 0
        and row["semantic_merge_qscale_hold_refresh_enabled_any"]
        and row["semantic_merge_radio_component_available_any"]
    )
    try:
        row.update(_metric_row(traj_path, gt_poses, gt_pos, target_start=target_start, target_end=target_end))
    except Exception as exc:
        row.update(
            {
                "trajectory_exists": traj_path.exists(),
                "trajectory_metric_error": f"{type(exc).__name__}:{exc}",
                "ATE_horizon": float("nan"),
                "Rot_horizon": float("nan"),
                "FinalErr_horizon": float("nan"),
                "alignment_scale": float("nan"),
                "target_chunk_ATE": float("nan"),
                "target_chunk_rows": 0,
            }
        )
    return row


def _metric_value(row: Mapping[str, Any] | None, metric_name: str) -> float:
    if row is None:
        return float("nan")
    val = _to_float(row.get(metric_name))
    if math.isfinite(val):
        return val
    return _to_float(row.get("ATE_horizon"))


def _gate(
    rows: list[dict[str, Any]],
    *,
    candidate_case: str,
    baseline_case: str,
    min_improvement: float,
    min_gate_chunks: int,
    metric_name: str,
) -> dict[str, Any]:
    by_chunk_case = {(int(row["chunk"]), str(row["case"])): row for row in rows}
    chunks = sorted({int(row["chunk"]) for row in rows if row.get("case") == candidate_case})
    pass_chunks: list[int] = []
    positive_chunks: list[int] = []
    details: list[dict[str, Any]] = []
    for chunk in chunks:
        native = by_chunk_case.get((chunk, baseline_case))
        cand = by_chunk_case.get((chunk, candidate_case))
        controls = [by_chunk_case[(chunk, case)] for case in sorted(CONTROL_CASES) if (chunk, case) in by_chunk_case]
        native_v = _metric_value(native, metric_name)
        cand_v = _metric_value(cand, metric_name)
        improvement = native_v - cand_v if math.isfinite(native_v) and math.isfinite(cand_v) else float("nan")
        if math.isfinite(improvement) and improvement > 0.0:
            positive_chunks.append(chunk)
        finite_controls = [row for row in controls if math.isfinite(_metric_value(row, metric_name))]
        beats_controls = bool(
            finite_controls and math.isfinite(cand_v) and all(cand_v < _metric_value(row, metric_name) for row in finite_controls)
        )
        hook_active = bool(cand and cand.get("hook_active"))
        candidate_ok = bool(
            cand
            and cand.get("returncode") == 0
            and hook_active
            and math.isfinite(improvement)
            and improvement >= min_improvement
            and beats_controls
        )
        if candidate_ok:
            pass_chunks.append(chunk)
        details.append(
            {
                "chunk": chunk,
                f"{baseline_case}_{metric_name}": native_v,
                f"{candidate_case}_{metric_name}": cand_v,
                "candidate_improvement_m": improvement,
                "candidate_hook_active": hook_active,
                "control_cases": [str(row.get("case")) for row in finite_controls],
                f"min_control_{metric_name}": min((_metric_value(row, metric_name) for row in finite_controls), default=float("nan")),
                "candidate_beats_all_controls": beats_controls,
                "candidate_pass": candidate_ok,
            }
        )
    failed_jobs = [row for row in rows if row.get("returncode") not in {0, None}]
    return {
        "phase": "ACL2 v74-TF refresh_hold_flip online merge smoke",
        "rows": len(rows),
        "candidate_case": candidate_case,
        "baseline_case": baseline_case,
        "candidate_chunks": chunks,
        "candidate_hook_active_chunks": sorted(
            {int(row["chunk"]) for row in rows if row.get("case") == candidate_case and row.get("hook_active")}
        ),
        "candidate_positive_chunks": positive_chunks,
        "candidate_pass_chunks": pass_chunks,
        "min_local_improvement_m": float(min_improvement),
        "min_gate_chunks": int(min_gate_chunks),
        "gate_metric": metric_name,
        "refresh_hold_flip_gate_pass": bool(len(pass_chunks) >= int(min_gate_chunks) and not failed_jobs),
        "failed_jobs": len(failed_jobs),
        "chunk_details": details,
        "gate_rule": (
            "candidate must return 0, show qscale hold-refresh trace evidence, improve the metric vs native, "
            "and beat geometry/random/shuffled controls. This smoke is a Phase5 intervention input, not a full "
            "Phase6 controller pass."
        ),
    }


def _write_markdown(path: Path, rows: list[dict[str, Any]], summary: Mapping[str, Any]) -> None:
    metric = str(summary.get("gate_metric") or "target_chunk_ATE")
    lines = [
        "# v74-TF refresh_hold_flip Online Merge Smoke",
        "",
        f"Gate pass: `{str(summary.get('refresh_hold_flip_gate_pass')).lower()}`",
        "",
        f"Rule: {summary.get('gate_rule')}",
        "",
        "## Chunk Gate Details",
        "",
        f"| chunk | native {metric} | radio_qscale {metric} | improvement m | hook | min control {metric} | beats controls | pass |",
        "|---:|---:|---:|---:|---|---:|---|---|",
    ]
    for item in summary.get("chunk_details", []):
        lines.append(
            "| {chunk} | {native:.6g} | {cand:.6g} | {imp:.6g} | `{hook}` | {ctrl:.6g} | `{beats}` | `{passed}` |".format(
                chunk=item.get("chunk"),
                native=_to_float(item.get(f"{summary.get('baseline_case')}_{metric}")),
                cand=_to_float(item.get(f"{summary.get('candidate_case')}_{metric}")),
                imp=_to_float(item.get("candidate_improvement_m")),
                hook=str(bool(item.get("candidate_hook_active"))).lower(),
                ctrl=_to_float(item.get(f"min_control_{metric}")),
                beats=str(bool(item.get("candidate_beats_all_controls"))).lower(),
                passed=str(bool(item.get("candidate_pass"))).lower(),
            )
        )
    lines.extend(
        [
            "",
            "## Per-Run Rows",
            "",
            "| chunk | case | rc | hook | target ATE | qscale factor med | effective alpha med | q handoff mean | scale med | residual mean |",
            "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda r: (int(r.get("chunk", -1)), str(r.get("case", "")))):
        lines.append(
            "| {chunk} | {case} | {rc} | `{hook}` | {target:.6g} | {qfac:.6g} | {alpha:.6g} | {qh:.6g} | {scale:.6g} | {res:.6g} |".format(
                chunk=row.get("chunk"),
                case=row.get("case"),
                rc=row.get("returncode"),
                hook=str(bool(row.get("hook_active"))).lower(),
                target=_to_float(row.get("target_chunk_ATE")),
                qfac=_to_float(row.get("semantic_merge_qscale_factor_median")),
                alpha=_to_float(row.get("semantic_merge_qscale_effective_blend_alpha_median")),
                qh=_to_float(row.get("semantic_merge_radio_handoff_qscale_observability_mean")),
                scale=_to_float(row.get("semantic_merge_scale_median")),
                res=_to_float(row.get("semantic_merge_overlap_residual_mean")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--kitti-gt", type=Path, default=DEFAULT_KITTI_GT)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--candidate-case", default="radio_qscale")
    parser.add_argument("--baseline-case", default="native_no_swa")
    parser.add_argument("--min-local-improvement", type=float, default=0.0)
    parser.add_argument("--min-gate-chunks", type=int, default=3)
    parser.add_argument("--gate-metric", default="target_chunk_ATE")
    args = parser.parse_args()

    output_root = args.output_root
    manifest_path = args.manifest or output_root / "phaseE_merge_run_manifest.json"
    manifest = _read_json(manifest_path)
    jobs = manifest.get("jobs", [])
    if not isinstance(jobs, list) or not jobs:
        raise SystemExit(f"no jobs found in {manifest_path}")

    _, gt_poses, gt_pos = _load_kitti_gt(args.kitti_gt)
    rows = [_job_row(job, gt_poses, gt_pos, chunk_size=args.chunk_size) for job in jobs]
    for row in rows:
        native = next((r for r in rows if r.get("chunk") == row.get("chunk") and r.get("case") == args.baseline_case), None)
        native_ate = _to_float(native.get("ATE_horizon") if native else None)
        row_ate = _to_float(row.get("ATE_horizon"))
        row["ATE_delta_vs_native"] = row_ate - native_ate if math.isfinite(row_ate) and math.isfinite(native_ate) else float("nan")
        native_target = _to_float(native.get("target_chunk_ATE") if native else None)
        row_target = _to_float(row.get("target_chunk_ATE"))
        row["target_chunk_ATE_delta_vs_native"] = (
            row_target - native_target if math.isfinite(row_target) and math.isfinite(native_target) else float("nan")
        )
        row["local_window_improvement_m"] = (
            native_target - row_target if math.isfinite(row_target) and math.isfinite(native_target) else float("nan")
        )

    summary = _gate(
        rows,
        candidate_case=args.candidate_case,
        baseline_case=args.baseline_case,
        min_improvement=args.min_local_improvement,
        min_gate_chunks=args.min_gate_chunks,
        metric_name=args.gate_metric,
    )
    summary["manifest"] = str(manifest_path)
    summary["output_root"] = str(output_root)
    summary["kitti_gt"] = str(args.kitti_gt)
    summary["cases"] = sorted({str(row.get("case")) for row in rows})
    summary["chunks"] = sorted({int(row.get("chunk")) for row in rows})
    summary["command_note"] = "Generated from run_v68_phaseE_merge_multichunk.py outputs with qscale hold-refresh enabled."

    _write_csv(output_root / "refresh_hold_flip_online_smoke_results.csv", rows)
    (output_root / "refresh_hold_flip_online_smoke_summary.json").write_text(
        json.dumps(_to_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(output_root / "refresh_hold_flip_online_smoke_report.md", rows, summary)
    print(json.dumps(_to_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
