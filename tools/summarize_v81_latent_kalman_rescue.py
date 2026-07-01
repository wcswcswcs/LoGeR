#!/usr/bin/env python3
"""Summarize ACL2 v81 latent Kalman-style semantic merge rescue evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


DEFAULT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v81tf_longwindow_semantic_three_memory_control/"
    "report_final/phase10_latent_kalman_rescue/latent_kalman_typeb_smoke"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse_csv_ints(text: str) -> List[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out: Dict[str, Any] = {}
            for key in fields:
                value = row.get(key)
                out[key] = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
            writer.writerow(out)


def _chunk_eval_map(eval_summary: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in eval_summary.get("chunk_decisions", []) or []:
        try:
            out[int(row["chunk"])] = row
        except Exception:
            continue
    return out


def _metric(row: Dict[str, Any], key: str, field: str) -> Any:
    return (((row.get("metrics") or {}).get(key) or {}).get(field))


def _last_trace_row(root: Path, *, chunk: int, case: str) -> Dict[str, Any]:
    rows = _read_jsonl(root / f"chunk{int(chunk):02d}" / case / "merge_state_trace.jsonl")
    for row in reversed(rows):
        if row.get("semantic_merge_latent_kalman_available") is True:
            return row
    return rows[-1] if rows else {}


def _summarize_chunk(root: Path, eval_by_chunk: Dict[int, Dict[str, Any]], *, chunk: int, case: str) -> Dict[str, Any]:
    trace = _last_trace_row(root, chunk=chunk, case=case)
    eval_row = eval_by_chunk.get(int(chunk), {})
    return {
        "chunk": int(chunk),
        "case": case,
        "trace_available": bool(trace),
        "latent_available": trace.get("semantic_merge_latent_kalman_available"),
        "latent_valid_count": trace.get("semantic_merge_latent_kalman_valid_count"),
        "latent_candidate_count": trace.get("semantic_merge_latent_kalman_candidate_count"),
        "latent_valid_ratio": trace.get("semantic_merge_latent_kalman_valid_ratio"),
        "latent_support_mean": trace.get("semantic_merge_latent_kalman_support_mean"),
        "latent_strategy_support_mean": trace.get("semantic_merge_latent_kalman_strategy_support_mean"),
        "latent_prior_uncertainty_mean": trace.get("semantic_merge_latent_kalman_prior_uncertainty_mean"),
        "latent_gain_mean": trace.get("semantic_merge_latent_kalman_gain_mean"),
        "latent_strategy_gain_mean": trace.get("semantic_merge_latent_kalman_strategy_gain_mean"),
        "latent_update_mean": trace.get("semantic_merge_latent_kalman_update_mean"),
        "latent_preserve_mean": trace.get("semantic_merge_latent_kalman_preserve_mean"),
        "latent_update_mass": trace.get("semantic_merge_latent_kalman_update_mass"),
        "latent_preserve_mass": trace.get("semantic_merge_latent_kalman_preserve_mass"),
        "latent_weighted_mass": trace.get("semantic_merge_latent_kalman_weighted_mass"),
        "latent_update_count": trace.get("semantic_merge_latent_kalman_update_count"),
        "latent_preserve_count": trace.get("semantic_merge_latent_kalman_preserve_count"),
        "native_overlap_guard_rejected": trace.get("semantic_merge_native_overlap_guard_rejected"),
        "native_overlap_residual": trace.get("semantic_merge_native_overlap_residual"),
        "final_overlap_residual": trace.get("semantic_merge_final_overlap_residual"),
        "semantic_merge_scale": trace.get("semantic_merge_scale"),
        "transform_scale_value": trace.get("transform_scale_value"),
        "fit_reason": trace.get("semantic_merge_fit_reason"),
        "head_tail_phaseE_chunk_pass": eval_row.get("head_tail_phaseE_chunk_pass"),
        "head_tail_beats_controls": eval_row.get("head_tail_beats_controls"),
        "head_tail_improvement_vs_baseline_ratio": eval_row.get("head_tail_improvement_vs_baseline_ratio"),
        "overlap_phaseE_chunk_pass": eval_row.get("overlap_phaseE_chunk_pass"),
        "overlap_beats_controls": eval_row.get("overlap_beats_controls"),
        "overlap_improvement_vs_baseline_ratio": eval_row.get("overlap_improvement_vs_baseline_ratio"),
        "head_candidate": _metric(eval_row, "head10_to_tail10_pose_sim3_rmse_m", "candidate"),
        "head_baseline": _metric(eval_row, "head10_to_tail10_pose_sim3_rmse_m", "baseline"),
        "head_best_control": _metric(eval_row, "head10_to_tail10_pose_sim3_rmse_m", "best_control"),
        "overlap_candidate": _metric(eval_row, "overlap3_to_future_pose_sim3_rmse_m", "candidate"),
        "overlap_baseline": _metric(eval_row, "overlap3_to_future_pose_sim3_rmse_m", "baseline"),
        "overlap_best_control": _metric(eval_row, "overlap3_to_future_pose_sim3_rmse_m", "best_control"),
    }


def _output_presence(root: Path, *, chunks: Sequence[int], cases: Sequence[str]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    missing_trajectories: List[str] = []
    missing_traces: List[str] = []
    for chunk in chunks:
        for case in cases:
            out_dir = root / f"chunk{int(chunk):02d}" / case
            traj = out_dir / "01.txt"
            trace = out_dir / "merge_state_trace.jsonl"
            row = {
                "chunk": int(chunk),
                "case": case,
                "trajectory": str(traj),
                "trajectory_exists": traj.is_file() and traj.stat().st_size > 0,
                "merge_state_trace": str(trace),
                "merge_state_trace_exists": trace.is_file() and trace.stat().st_size > 0,
            }
            rows.append(row)
            if not row["trajectory_exists"]:
                missing_trajectories.append(str(traj))
            if case != "native_no_swa" and not row["merge_state_trace_exists"]:
                missing_traces.append(str(trace))
    return {
        "expected_output_count": int(len(chunks) * len(cases)),
        "present_trajectory_count": int(sum(1 for row in rows if row["trajectory_exists"])),
        "present_trace_count": int(sum(1 for row in rows if row["merge_state_trace_exists"])),
        "missing_trajectories": missing_trajectories,
        "missing_traces": missing_traces,
        "rows": rows,
    }


def _render_report(summary: Dict[str, Any]) -> str:
    eval_summary = summary["eval_summary"]
    lines = [
        "# ACL2 v81 Phase10 Latent Kalman-Style Rescue",
        "",
        "## Decision",
        "",
        f"- phaseE_gate_pass: `{eval_summary.get('phaseE_gate_pass')}`",
        f"- head_tail_pass_count: `{eval_summary.get('head_tail_pass_count')}`",
        f"- overlap_pass_count: `{eval_summary.get('overlap_pass_count')}`",
        f"- head_tail_median_improvement_vs_baseline_ratio: `{eval_summary.get('head_tail_median_improvement_vs_baseline_ratio')}`",
        f"- overlap_median_improvement_vs_baseline_ratio: `{eval_summary.get('overlap_median_improvement_vs_baseline_ratio')}`",
        f"- missing_count: `{len(eval_summary.get('missing', []) or [])}`",
        f"- present_trajectory_count: `{summary.get('output_presence', {}).get('present_trajectory_count')}` / `{summary.get('output_presence', {}).get('expected_output_count')}`",
        "",
        "## Per-Chunk Evidence",
        "",
    ]
    for row in summary["latent_rows"]:
        lines.append(
            "- chunk{chunk:02d}: latent_available=`{active}`, valid_ratio=`{vr}`, "
            "gain_mean=`{gm}`, preserve_mean=`{pm}`, guard_rejected=`{gr}`, "
            "head_pass=`{hp}`, overlap_pass=`{op}`, head_improve=`{hi}`, overlap_improve=`{oi}`".format(
                chunk=int(row["chunk"]),
                active=row.get("latent_available"),
                vr=row.get("latent_valid_ratio"),
                gm=row.get("latent_gain_mean"),
                pm=row.get("latent_preserve_mean"),
                gr=row.get("native_overlap_guard_rejected"),
                hp=row.get("head_tail_phaseE_chunk_pass"),
                op=row.get("overlap_phaseE_chunk_pass"),
                hi=row.get("head_tail_improvement_vs_baseline_ratio"),
                oi=row.get("overlap_improvement_vs_baseline_ratio"),
            )
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- root: `{summary.get('root')}`",
            f"- manifest: `{summary.get('manifest_path')}`",
            f"- eval_json: `{summary.get('eval_json')}`",
            f"- trace_audit_csv: `{summary.get('trace_audit_csv')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--eval-json", type=Path, default=DEFAULT_ROOT / "latent_kalman_phaseE_summary.json")
    parser.add_argument("--chunks", default="7,8,9,10")
    parser.add_argument("--cases", default="native_no_swa,latent_kalman,geometry_only,latent_kalman_random,latent_kalman_shuffled")
    parser.add_argument("--candidate", default="latent_kalman")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-report", type=Path, default=None)
    args = parser.parse_args()

    chunks = _parse_csv_ints(args.chunks)
    cases = [part.strip() for part in args.cases.split(",") if part.strip()]
    eval_summary = _read_json(args.eval_json)
    eval_by_chunk = _chunk_eval_map(eval_summary)
    latent_rows = [_summarize_chunk(args.root, eval_by_chunk, chunk=chunk, case=args.candidate) for chunk in chunks]
    out_json = args.out_json or args.root / "latent_kalman_rescue_summary.json"
    out_csv = args.out_csv or args.root / "latent_kalman_trace_audit.csv"
    out_report = args.out_report or args.root / "latent_kalman_rescue_report.md"
    manifest_path = args.root / "phaseE_merge_run_manifest.json"
    manifest = _read_json(manifest_path)
    output_presence = _output_presence(args.root, chunks=chunks, cases=cases)
    summary = {
        "root": str(args.root),
        "manifest_path": str(manifest_path),
        "eval_json": str(args.eval_json),
        "trace_audit_csv": str(out_csv),
        "report": str(out_report),
        "chunks": chunks,
        "cases": cases,
        "candidate": args.candidate,
        "manifest_job_count": manifest.get("job_count"),
        "manifest_failed_jobs": manifest.get("failed_jobs", []),
        "output_presence": {k: v for k, v in output_presence.items() if k != "rows"},
        "eval_summary": {k: v for k, v in eval_summary.items() if k not in {"run_rows", "chunk_decisions"}},
        "latent_rows": latent_rows,
        "chunk_decisions": eval_summary.get("chunk_decisions", []),
        "output_presence_rows": output_presence["rows"],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_csv, latent_rows)
    out_report.write_text(_render_report(summary), encoding="utf-8")
    printable = {k: v for k, v in summary.items() if k not in {"latent_rows", "chunk_decisions"}}
    print(json.dumps(_jsonable(printable), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote_json={out_json}")
    print(f"wrote_csv={out_csv}")
    print(f"wrote_report={out_report}")


if __name__ == "__main__":
    main()
