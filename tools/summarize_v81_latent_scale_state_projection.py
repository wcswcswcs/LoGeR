#!/usr/bin/env python3
"""Summarize ACL2 v81 latent scale-state/projection merge rescue evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


DEFAULT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v81tf_longwindow_semantic_three_memory_control/"
    "report_final/phase10_latent_scale_state_rescue/latent_kalman_residual_projection_q0_allgated_typeb"
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
            clean: Dict[str, Any] = {}
            for key in fields:
                value = row.get(key)
                clean[key] = (
                    json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
            writer.writerow(clean)


def _eval_by_chunk(eval_summary: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in eval_summary.get("chunk_decisions", []) or []:
        try:
            out[int(row["chunk"])] = row
        except Exception:
            continue
    return out


def _metric(eval_row: Dict[str, Any], key: str, field: str) -> Any:
    return (((eval_row.get("metrics") or {}).get(key) or {}).get(field))


def _last_trace(root: Path, *, chunk: int, case: str) -> Dict[str, Any]:
    rows = _read_jsonl(root / f"chunk{int(chunk):02d}" / case / "merge_state_trace.jsonl")
    return rows[-1] if rows else {}


def _summarize_case(
    root: Path,
    eval_rows: Dict[int, Dict[str, Any]],
    *,
    chunk: int,
    case: str,
    candidate: str,
) -> Dict[str, Any]:
    run_dir = root / f"chunk{int(chunk):02d}" / case
    trace = _last_trace(root, chunk=chunk, case=case)
    eval_row = eval_rows.get(int(chunk), {}) if case == candidate else {}
    return {
        "chunk": int(chunk),
        "case": case,
        "trajectory_exists": (run_dir / "01.txt").is_file() and (run_dir / "01.txt").stat().st_size > 0,
        "trace_exists": bool(trace),
        "fit_reason": trace.get("semantic_merge_fit_reason"),
        "transform_scale_value": trace.get("transform_scale_value"),
        "latent_weighted_mass": trace.get("semantic_merge_latent_kalman_weighted_mass"),
        "latent_gain_mean": trace.get("semantic_merge_latent_kalman_strategy_gain_mean"),
        "latent_preserve_mean": trace.get("semantic_merge_latent_kalman_preserve_mean"),
        "scale_state_gate_policy": trace.get("online_scale_state_gate_policy"),
        "scale_state_gate_reason": trace.get("online_scale_state_gate_reason"),
        "scale_state_active": trace.get("online_scale_state_active"),
        "scale_state_action": trace.get("online_scale_state_action"),
        "scale_state_input_scale": trace.get("online_scale_state_input_scale"),
        "scale_state_output_scale": trace.get("online_scale_state_output_scale"),
        "projection_enabled": trace.get("semantic_merge_residual_safe_projection_enabled"),
        "projection_accepted": trace.get("semantic_merge_residual_safe_projection_accepted"),
        "projection_alpha": trace.get("semantic_merge_residual_safe_projection_alpha"),
        "projection_scale": trace.get("semantic_merge_residual_safe_projection_scale"),
        "native_overlap_guard_rejected": trace.get("semantic_merge_native_overlap_guard_rejected"),
        "native_overlap_residual": trace.get("semantic_merge_native_overlap_residual"),
        "final_overlap_residual": trace.get("semantic_merge_final_overlap_residual"),
        "semantic_fit_overlap_residual": trace.get("semantic_merge_overlap_residual"),
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


def _render_report(summary: Dict[str, Any]) -> str:
    eval_summary = summary["eval_summary"]
    candidate = summary["candidate"]
    rows = [row for row in summary["trace_rows"] if row["case"] == candidate]
    accepted = [row["chunk"] for row in rows if row.get("projection_accepted") is True]
    rejected = [row["chunk"] for row in rows if row.get("native_overlap_guard_rejected") is True]
    lines = [
        "# ACL2 v81 Latent Scale-State Residual Projection",
        "",
        "## Decision",
        "",
        f"- phaseE_gate_pass: `{eval_summary.get('phaseE_gate_pass')}`",
        f"- head_tail_pass_count: `{eval_summary.get('head_tail_pass_count')}`",
        f"- overlap_pass_count: `{eval_summary.get('overlap_pass_count')}`",
        f"- head_tail_median_improvement_vs_baseline_ratio: `{eval_summary.get('head_tail_median_improvement_vs_baseline_ratio')}`",
        f"- overlap_median_improvement_vs_baseline_ratio: `{eval_summary.get('overlap_median_improvement_vs_baseline_ratio')}`",
        f"- missing_count: `{len(eval_summary.get('missing', []) or [])}`",
        f"- manifest_failed_jobs_count: `{len(summary.get('manifest_failed_jobs', []) or [])}`",
        f"- oom_repair_logs: `{len(summary.get('oom_repair_logs', []) or [])}`",
        "",
        "## Candidate Evidence",
        "",
        f"- projection_accepted_chunks: `{accepted}`",
        f"- native_overlap_guard_rejected_chunks: `{rejected}`",
        "",
        "## Per-Chunk Candidate Metrics",
        "",
    ]
    for row in rows:
        lines.append(
            "- chunk{chunk:02d}: action=`{action}`, projection=`{proj}`, alpha=`{alpha}`, "
            "guard_rejected=`{guard}`, head_imp=`{head}`, overlap_imp=`{overlap}`, "
            "head_beats_controls=`{hb}`, overlap_beats_controls=`{ob}`".format(
                chunk=int(row["chunk"]),
                action=row.get("scale_state_action"),
                proj=row.get("projection_accepted"),
                alpha=row.get("projection_alpha"),
                guard=row.get("native_overlap_guard_rejected"),
                head=row.get("head_tail_improvement_vs_baseline_ratio"),
                overlap=row.get("overlap_improvement_vs_baseline_ratio"),
                hb=row.get("head_tail_beats_controls"),
                ob=row.get("overlap_beats_controls"),
            )
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- root: `{summary.get('root')}`",
            f"- eval_json: `{summary.get('eval_json')}`",
            f"- trace_audit_csv: `{summary.get('trace_audit_csv')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--eval-json", type=Path, default=DEFAULT_ROOT / "latent_projection_q0_allgated_phaseE_summary.json")
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
    eval_rows = _eval_by_chunk(eval_summary)
    trace_rows = [
        _summarize_case(args.root, eval_rows, chunk=chunk, case=case, candidate=args.candidate)
        for chunk in chunks
        for case in cases
    ]
    manifest = _read_json(args.root / "phaseE_merge_run_manifest.json")
    oom_logs = sorted(str(path) for path in args.root.glob("chunk*/**/run_oom*.log"))
    out_json = args.out_json or args.root / "latent_scale_projection_rescue_summary.json"
    out_csv = args.out_csv or args.root / "latent_scale_projection_trace_audit.csv"
    out_report = args.out_report or args.root / "latent_scale_projection_rescue_report.md"
    summary = {
        "root": str(args.root),
        "eval_json": str(args.eval_json),
        "trace_audit_csv": str(out_csv),
        "report": str(out_report),
        "chunks": chunks,
        "cases": cases,
        "candidate": args.candidate,
        "manifest_job_count": manifest.get("job_count"),
        "manifest_failed_jobs": manifest.get("failed_jobs", []),
        "oom_repair_logs": oom_logs,
        "eval_summary": {k: v for k, v in eval_summary.items() if k not in {"run_rows", "chunk_decisions"}},
        "chunk_decisions": eval_summary.get("chunk_decisions", []),
        "trace_rows": trace_rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_csv, trace_rows)
    out_report.write_text(_render_report(summary), encoding="utf-8")
    printable = {k: v for k, v in summary.items() if k not in {"trace_rows", "chunk_decisions"}}
    print(json.dumps(_jsonable(printable), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote_json={out_json}")
    print(f"wrote_csv={out_csv}")
    print(f"wrote_report={out_report}")


if __name__ == "__main__":
    main()
