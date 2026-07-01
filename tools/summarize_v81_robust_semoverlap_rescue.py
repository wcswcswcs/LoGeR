#!/usr/bin/env python3
"""Summarize ACL2 v81 robust semantic-overlap rescue evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


DEFAULT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v81tf_longwindow_semantic_three_memory_control/"
    "report_final/phase10_robust_semoverlap_rescue/robust_semoverlap_typeb_smoke"
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
    return rows[-1] if rows else {}


def _summarize_chunk(root: Path, eval_by_chunk: Dict[int, Dict[str, Any]], *, chunk: int, case: str) -> Dict[str, Any]:
    trace = _last_trace_row(root, chunk=chunk, case=case)
    eval_row = eval_by_chunk.get(int(chunk), {})
    return {
        "chunk": int(chunk),
        "case": case,
        "trace_available": bool(trace),
        "robust_weight_active": trace.get("semantic_merge_v68_robust_semoverlap_weight"),
        "robust_residual_mode": trace.get("semantic_merge_v68_robust_semoverlap_residual_mode"),
        "robust_low_conf_mean": trace.get("semantic_merge_v68_robust_semoverlap_low_conf_mean"),
        "robust_proxy_mean": trace.get("semantic_merge_v68_robust_semoverlap_proxy_mean"),
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


def _render_report(summary: Dict[str, Any]) -> str:
    eval_summary = summary["eval_summary"]
    lines = [
        "# ACL2 v81 Phase10 Robust Semantic-Overlap Rescue",
        "",
        "## Decision",
        "",
        f"- phaseE_gate_pass: `{eval_summary.get('phaseE_gate_pass')}`",
        f"- head_tail_pass_count: `{eval_summary.get('head_tail_pass_count')}`",
        f"- overlap_pass_count: `{eval_summary.get('overlap_pass_count')}`",
        f"- head_tail_median_improvement_vs_baseline_ratio: `{eval_summary.get('head_tail_median_improvement_vs_baseline_ratio')}`",
        f"- overlap_median_improvement_vs_baseline_ratio: `{eval_summary.get('overlap_median_improvement_vs_baseline_ratio')}`",
        f"- missing_count: `{len(eval_summary.get('missing', []) or [])}`",
        "",
        "## Per-Chunk Evidence",
        "",
    ]
    for row in summary["robust_rows"]:
        lines.append(
            "- chunk{chunk:02d}: robust_active=`{active}`, guard_rejected=`{gr}`, "
            "head_pass=`{hp}`, overlap_pass=`{op}`, head_improve=`{hi}`, overlap_improve=`{oi}`".format(
                chunk=int(row["chunk"]),
                active=row.get("robust_weight_active"),
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
    parser.add_argument("--eval-json", type=Path, default=DEFAULT_ROOT / "robust_semoverlap_phaseE_summary.json")
    parser.add_argument("--chunks", default="7,8,9,10")
    parser.add_argument("--candidate", default="robust_semoverlap")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-report", type=Path, default=None)
    args = parser.parse_args()

    chunks = _parse_csv_ints(args.chunks)
    eval_summary = _read_json(args.eval_json)
    eval_by_chunk = _chunk_eval_map(eval_summary)
    robust_rows = [_summarize_chunk(args.root, eval_by_chunk, chunk=chunk, case=args.candidate) for chunk in chunks]
    out_json = args.out_json or args.root / "robust_semoverlap_rescue_summary.json"
    out_csv = args.out_csv or args.root / "robust_semoverlap_trace_audit.csv"
    out_report = args.out_report or args.root / "robust_semoverlap_rescue_report.md"
    manifest_path = args.root / "phaseE_merge_run_manifest.json"
    manifest = _read_json(manifest_path)
    summary = {
        "root": str(args.root),
        "manifest_path": str(manifest_path),
        "eval_json": str(args.eval_json),
        "trace_audit_csv": str(out_csv),
        "report": str(out_report),
        "chunks": chunks,
        "candidate": args.candidate,
        "manifest_job_count": manifest.get("job_count"),
        "manifest_failed_jobs": manifest.get("failed_jobs", []),
        "eval_summary": {k: v for k, v in eval_summary.items() if k not in {"run_rows", "chunk_decisions"}},
        "robust_rows": robust_rows,
        "chunk_decisions": eval_summary.get("chunk_decisions", []),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_csv, robust_rows)
    out_report.write_text(_render_report(summary), encoding="utf-8")
    printable = {k: v for k, v in summary.items() if k not in {"robust_rows", "chunk_decisions"}}
    print(json.dumps(_jsonable(printable), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote_json={out_json}")
    print(f"wrote_csv={out_csv}")
    print(f"wrote_report={out_report}")


if __name__ == "__main__":
    main()
