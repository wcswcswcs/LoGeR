#!/usr/bin/env python3
"""Summarize ACL2 v74-TF mid/tail semantic-anchor boost smoke jobs."""

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
    "phase4_extra_nA_context_anchor_boost_midtail_rho010_top4"
)
DEFAULT_KITTI_GT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
CONTROL_CASES = {"random_same_mass", "shuffled_semantic"}


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


def _metric_row(path: Path, gt_poses: np.ndarray, gt_pos: np.ndarray, *, target_start: int, target_end: int) -> Dict[str, Any]:
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
    read_cfg = _last_present(hmc_rows, "read_path_controls_requested")
    control_trace = _last_present(hmc_rows, "control_trace")

    context_calls = _max_nested(
        hmc_rows,
        ("control_trace", "hook_effect_summary", "frame_attention", "num_context_source_skip_applied"),
    )
    anchor_boost_calls = _max_nested(
        hmc_rows,
        ("control_trace", "hook_effect_summary", "frame_attention", "num_semantic_anchor_boost_applied"),
    )
    boost_tokens = _max_nested(
        hmc_rows,
        ("control_trace", "hook_effect_summary", "frame_attention", "max_context_source_boost_tokens"),
    )
    boost_fraction = _max_nested(
        hmc_rows,
        ("control_trace", "hook_effect_summary", "frame_attention", "mean_context_source_keep_ratio"),
    )

    row: Dict[str, Any] = {
        "chunk": int(job.get("chunk", -1)),
        "case": str(job.get("case", "")),
        "returncode": int(job.get("returncode")) if job.get("returncode") is not None else None,
        "skipped": bool(job.get("skipped", False)),
        "duration_sec": _to_float(job.get("duration_sec")),
        "gpu": job.get("gpu"),
        "out_dir": str(out_dir),
        "run_log": str(job.get("run_log") or out_dir / "run.log"),
        "trajectory": str(traj_path),
        "hmc_state_hash": str(hmc_path),
        "start_frame": int(job.get("start_frame", -1)),
        "end_frame": int(job.get("end_frame", -1)),
        "target_start_frame": int(job.get("target_start_frame", job.get("start_frame", -1))),
        "target_end_frame": int(job.get("target_end_frame", job.get("end_frame", -1))),
        "semantic_anchor_mode_effective_manifest": job.get("semantic_anchor_mode_effective"),
        "frame_region_effective_manifest": job.get("frame_region_effective"),
        "boost_rho_effective_manifest": job.get("boost_rho_effective"),
        "hmc_rows": len(hmc_rows),
        "hmc_json_decode_errors": sum(1 for item in hmc_rows if item.get("_json_decode_error")),
        "control_trace_present": isinstance(control_trace, Mapping),
        "context_source_skip_path_observed": _has_implemented_path(hmc_rows, "context_source_skip"),
        "context_source_skip_applied_count_max": context_calls,
        "semantic_anchor_boost_applied_count_max": anchor_boost_calls,
        "context_source_boost_tokens_max": boost_tokens,
        "context_source_keep_ratio_mean_max": boost_fraction,
        "prior_semantic_anchor_bank_available": _last_present(hmc_rows, "prior_semantic_anchor_bank_available"),
        "prior_semantic_anchor_reason": _last_present(hmc_rows, "prior_semantic_anchor_reason"),
        "prior_semantic_anchor_token_count": _last_present(hmc_rows, "prior_semantic_anchor_token_count"),
        "prior_semantic_anchor_token_ratio": _last_present(hmc_rows, "prior_semantic_anchor_token_ratio"),
    }
    if isinstance(read_cfg, Mapping):
        row.update(
            {
                "requested_context_source_skip_impl": read_cfg.get("context_source_skip_impl"),
                "requested_context_source_skip_scope": read_cfg.get("context_source_skip_scope"),
                "requested_context_source_skip_mode": read_cfg.get("context_source_skip_mode"),
                "requested_context_source_skip_mask": read_cfg.get("context_source_skip_mask"),
                "requested_context_source_skip_frame_region": read_cfg.get("context_source_skip_frame_region"),
                "requested_context_source_skip_soft_rho": read_cfg.get("context_source_skip_soft_rho"),
                "requested_semantic_anchor_mode": read_cfg.get("semantic_anchor_mode"),
            }
        )
    row["hook_active"] = bool(
        row["case"] != "native_no_boost"
        and row["returncode"] == 0
        and row["hmc_rows"] > 0
        and row["context_source_skip_path_observed"]
        and math.isfinite(anchor_boost_calls)
        and anchor_boost_calls > 0
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
        row.update(
            {
                "trajectory_exists": traj_path.exists(),
                "trajectory_metric_error": f"{type(exc).__name__}:{exc}",
                "ATE_horizon": float("nan"),
                "Rot_horizon": float("nan"),
                "FinalErr_horizon": float("nan"),
                "alignment_scale": float("nan"),
                "target_chunk_ATE": float("nan"),
            }
        )
    return row


def _gate(rows: List[Dict[str, Any]], min_improvement: float, min_gate_chunks: int) -> Dict[str, Any]:
    by_chunk_case: Dict[Tuple[int, str], Dict[str, Any]] = {(int(r["chunk"]), str(r["case"])): r for r in rows}
    chunks = sorted({int(r["chunk"]) for r in rows if r.get("case") == "candidate"})
    pass_chunks: List[int] = []
    details: List[Dict[str, Any]] = []
    for chunk in chunks:
        native = by_chunk_case.get((chunk, "native_no_boost"))
        cand = by_chunk_case.get((chunk, "candidate"))
        controls = [by_chunk_case[(chunk, case)] for case in sorted(CONTROL_CASES) if (chunk, case) in by_chunk_case]
        native_ate = _to_float(native.get("target_chunk_ATE") if native else None)
        cand_ate = _to_float(cand.get("target_chunk_ATE") if cand else None)
        improvement = native_ate - cand_ate if math.isfinite(native_ate) and math.isfinite(cand_ate) else float("nan")
        finite_controls = [row for row in controls if math.isfinite(_to_float(row.get("target_chunk_ATE")))]
        beats_all_controls = bool(
            finite_controls
            and math.isfinite(cand_ate)
            and all(cand_ate < _to_float(row.get("target_chunk_ATE")) for row in finite_controls)
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
        details.append(
            {
                "chunk": chunk,
                "native_target_chunk_ATE": native_ate,
                "candidate_target_chunk_ATE": cand_ate,
                "candidate_improvement_m": improvement,
                "candidate_hook_active": hook_active,
                "control_cases": [str(row.get("case")) for row in finite_controls],
                "min_control_target_chunk_ATE": min(
                    (_to_float(row.get("target_chunk_ATE")) for row in finite_controls),
                    default=float("nan"),
                ),
                "candidate_beats_all_controls": beats_all_controls,
                "candidate_pass": candidate_ok,
            }
        )
    failed_jobs = [r for r in rows if r.get("returncode") not in {0, None}]
    return {
        "phase": "ACL2 v74-TF context semantic-anchor boost online smoke",
        "rows": len(rows),
        "candidate_chunks": chunks,
        "candidate_hook_active_chunks": sorted(
            {int(r["chunk"]) for r in rows if r.get("case") == "candidate" and r.get("hook_active")}
        ),
        "candidate_pass_chunks": pass_chunks,
        "min_local_improvement_m": float(min_improvement),
        "min_gate_chunks": int(min_gate_chunks),
        "gate_metric": "target_chunk_ATE",
        "context_anchor_boost_gate_pass": len(pass_chunks) >= int(min_gate_chunks) and not failed_jobs,
        "failed_jobs": len(failed_jobs),
        "chunk_details": details,
        "gate_rule": (
            "candidate must return 0, show context_source_skip semantic-anchor boost hook evidence, "
            f"improve target_chunk_ATE vs native by >= {min_improvement} m, and beat random_same_mass "
            f"and shuffled_semantic controls; smoke pass requires >= {min_gate_chunks} passing chunks and no failed jobs."
        ),
    }


def _write_markdown(path: Path, rows: List[Dict[str, Any]], summary: Mapping[str, Any]) -> None:
    lines = [
        "# ACL2 v74-TF Context Anchor Boost Smoke",
        "",
        f"Gate pass: `{str(summary.get('context_anchor_boost_gate_pass')).lower()}`",
        "",
        f"Rule: {summary.get('gate_rule')}",
        "",
        "## Chunk Gate Details",
        "",
        "| chunk | native target ATE | candidate target ATE | improvement m | hook | min control target ATE | beats controls | pass |",
        "|---:|---:|---:|---:|---|---:|---|---|",
    ]
    for item in summary.get("chunk_details", []):
        lines.append(
            "| {chunk} | {native:.6g} | {cand:.6g} | {imp:.6g} | `{hook}` | {ctrl:.6g} | `{beats}` | `{passed}` |".format(
                chunk=item.get("chunk"),
                native=_to_float(item.get("native_target_chunk_ATE")),
                cand=_to_float(item.get("candidate_target_chunk_ATE")),
                imp=_to_float(item.get("candidate_improvement_m")),
                hook=str(bool(item.get("candidate_hook_active"))).lower(),
                ctrl=_to_float(item.get("min_control_target_chunk_ATE")),
                beats=str(bool(item.get("candidate_beats_all_controls"))).lower(),
                passed=str(bool(item.get("candidate_pass"))).lower(),
            )
        )
    lines.extend(
        [
            "",
            "## Per-Run Rows",
            "",
            "| chunk | case | rc | hook | target ATE | mode | region | boost calls | boost tokens |",
            "|---:|---|---:|---|---:|---|---|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda r: (int(r.get("chunk", -1)), str(r.get("case", "")))):
        lines.append(
            "| {chunk} | {case} | {rc} | `{hook}` | {target:.6g} | {mode} | {region} | {calls:.6g} | {tokens:.6g} |".format(
                chunk=row.get("chunk"),
                case=row.get("case"),
                rc=row.get("returncode"),
                hook=str(bool(row.get("hook_active"))).lower(),
                target=_to_float(row.get("target_chunk_ATE")),
                mode=row.get("semantic_anchor_mode_effective_manifest") or "",
                region=row.get("frame_region_effective_manifest") or "",
                calls=_to_float(row.get("semantic_anchor_boost_applied_count_max")),
                tokens=_to_float(row.get("context_source_boost_tokens_max")),
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
    manifest_path = args.manifest or output_root / "v74tf_context_anchor_boost_manifest.json"
    manifest = _read_json(manifest_path)
    jobs = manifest.get("jobs", [])
    if not isinstance(jobs, list) or not jobs:
        raise SystemExit(f"no jobs found in {manifest_path}")

    _, gt_poses, gt_pos = _load_kitti_gt(args.kitti_gt)
    rows = [_job_row(job, gt_poses, gt_pos) for job in jobs]
    summary = _gate(rows, args.min_local_improvement, args.min_gate_chunks)
    summary["manifest"] = str(manifest_path)
    summary["output_root"] = str(output_root)
    summary["kitti_gt"] = str(args.kitti_gt)
    summary["cases"] = sorted({str(row.get("case")) for row in rows})
    summary["chunks"] = sorted({int(row.get("chunk")) for row in rows})

    _write_csv(output_root / "context_anchor_boost_smoke_results.csv", rows)
    (output_root / "context_anchor_boost_smoke_summary.json").write_text(
        json.dumps(_to_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(output_root / "context_anchor_boost_smoke_summary.md", rows, summary)
    print(json.dumps(_to_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
