#!/usr/bin/env python3
"""Summarize ACL2 v81 retrieval-static semantic merge rescue evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


DEFAULT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v81tf_longwindow_semantic_three_memory_control/"
    "report_final/phase10_retrieval_static_rescue/retrieval_static_typeb_smoke"
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
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
                if isinstance(value, (dict, list, tuple)):
                    out[key] = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
                else:
                    out[key] = value
            writer.writerow(out)


def _last_retrieval_row(root: Path, *, chunk: int, case: str) -> Dict[str, Any]:
    trace_path = root / f"chunk{int(chunk):02d}" / case / "merge_state_trace.jsonl"
    rows = _read_jsonl(trace_path)
    for row in reversed(rows):
        if row.get("semantic_merge_retrieval_static_available") is True:
            return row
    return rows[-1] if rows else {}


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


def _summarize_chunk(root: Path, eval_by_chunk: Dict[int, Dict[str, Any]], *, chunk: int, case: str) -> Dict[str, Any]:
    trace = _last_retrieval_row(root, chunk=chunk, case=case)
    eval_row = eval_by_chunk.get(int(chunk), {})
    return {
        "chunk": int(chunk),
        "case": case,
        "trace_available": bool(trace),
        "retrieval_available": trace.get("semantic_merge_retrieval_static_available"),
        "retrieval_valid_count": trace.get("semantic_merge_retrieval_static_valid_count"),
        "retrieval_candidate_count": trace.get("semantic_merge_retrieval_static_candidate_count"),
        "retrieval_valid_ratio": trace.get("semantic_merge_retrieval_static_valid_ratio"),
        "retrieval_support_mean": trace.get("semantic_merge_retrieval_static_strategy_support_mean"),
        "retrieval_anchor_count": trace.get("semantic_merge_retrieval_static_anchor_count"),
        "retrieval_weighted_mass": trace.get("semantic_merge_retrieval_static_weighted_mass"),
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
        "# ACL2 v81 Phase10 Retrieval-Static Rescue",
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
    for row in summary["retrieval_rows"]:
        lines.append(
            "- chunk{chunk:02d}: valid_ratio=`{vr}`, weighted_mass=`{wm}`, "
            "guard_rejected=`{gr}`, head_pass=`{hp}`, overlap_pass=`{op}`, "
            "head_improve=`{hi}`, overlap_improve=`{oi}`".format(
                chunk=int(row["chunk"]),
                vr=row.get("retrieval_valid_ratio"),
                wm=row.get("retrieval_weighted_mass"),
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
    parser.add_argument("--eval-json", type=Path, default=DEFAULT_ROOT / "retrieval_static_phaseE_summary.json")
    parser.add_argument("--chunks", default="7,8,9,10")
    parser.add_argument("--candidate", default="retrieval_static")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-report", type=Path, default=None)
    args = parser.parse_args()

    chunks = _parse_csv_ints(args.chunks)
    eval_summary = _read_json(args.eval_json)
    eval_by_chunk = _chunk_eval_map(eval_summary)
    retrieval_rows = [
        _summarize_chunk(args.root, eval_by_chunk, chunk=chunk, case=args.candidate)
        for chunk in chunks
    ]
    out_json = args.out_json or args.root / "retrieval_static_rescue_summary.json"
    out_csv = args.out_csv or args.root / "retrieval_static_trace_audit.csv"
    out_report = args.out_report or args.root / "retrieval_static_rescue_report.md"
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
        "retrieval_rows": retrieval_rows,
        "chunk_decisions": eval_summary.get("chunk_decisions", []),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_csv, retrieval_rows)
    out_report.write_text(_render_report(summary), encoding="utf-8")
    printable = {k: v for k, v in summary.items() if k not in {"retrieval_rows", "chunk_decisions"}}
    print(json.dumps(_jsonable(printable), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote_json={out_json}")
    print(f"wrote_csv={out_csv}")
    print(f"wrote_report={out_report}")


if __name__ == "__main__":
    main()
