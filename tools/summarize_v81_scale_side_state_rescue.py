#!/usr/bin/env python3
"""Summarize ACL2 v81 Phase10 scale-side-state rescue evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v81tf_longwindow_semantic_three_memory_control/"
    "report_final/phase10_scale_side_state_rescue/scale_state_pre_guard_semcand_s098_102"
)
DEFAULT_EVAL_JSON = DEFAULT_ROOT / "scale_state_phaseE_summary.json"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _parse_csv_ints(text: str) -> List[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _counter_field(rows: Iterable[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        value = row.get(key)
        if value is None or value == "":
            value = "missing"
        counts[str(value)] += 1
    return dict(sorted(counts.items()))


def _float_values(rows: Iterable[Dict[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        raw = row.get(key)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value == value and value not in {float("inf"), float("-inf")}:
            values.append(value)
    return values


def _minmax(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"min": None, "max": None}
    return {"min": float(min(values)), "max": float(max(values))}


def _summarize_trace(root: Path, *, chunk: int, case: str) -> Dict[str, Any]:
    run_dir = root / f"chunk{int(chunk):02d}" / case
    trace_path = run_dir / "merge_state_trace.jsonl"
    rows = _read_jsonl(trace_path)
    active_rows = [row for row in rows if bool(row.get("online_scale_state_active", False))]
    gate_rows = [row for row in rows if "online_scale_state_gate_pass" in row]
    guard_rows = [row for row in rows if "semantic_merge_native_overlap_guard_rejected" in row]
    input_scales = _float_values(rows, "online_scale_state_input_scale")
    output_scales = _float_values(rows, "online_scale_state_output_scale")
    transform_scales = _float_values(rows, "transform_scale_value")
    return {
        "chunk": int(chunk),
        "case": case,
        "run_dir": str(run_dir),
        "trace_path": str(trace_path),
        "trace_exists": bool(trace_path.is_file()),
        "trace_rows": int(len(rows)),
        "scale_state_active_rows": int(len(active_rows)),
        "scale_state_action_counts": _counter_field(active_rows, "online_scale_state_action"),
        "scale_state_stage_counts": _counter_field(active_rows, "online_scale_state_stage"),
        "scale_state_gate_rows": int(len(gate_rows)),
        "scale_state_gate_pass_counts": _counter_field(gate_rows, "online_scale_state_gate_pass"),
        "scale_state_gate_reason_counts": _counter_field(gate_rows, "online_scale_state_gate_reason"),
        "native_overlap_guard_rows": int(len(guard_rows)),
        "native_overlap_guard_reject_counts": _counter_field(guard_rows, "semantic_merge_native_overlap_guard_rejected"),
        "semantic_merge_reject_reason_counts": _counter_field(
            [row for row in rows if row.get("semantic_merge_rejected")],
            "semantic_merge_reject_reason",
        ),
        "scale_state_input_scale": _minmax(input_scales),
        "scale_state_output_scale": _minmax(output_scales),
        "transform_scale_value": _minmax(transform_scales),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat: Dict[str, Any] = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (dict, list, tuple)):
                    flat[key] = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
                else:
                    flat[key] = value
            writer.writerow(flat)


def _case_totals(rows: List[Dict[str, Any]], case: str) -> Dict[str, Any]:
    selected = [row for row in rows if row["case"] == case]
    action_counts: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()
    gate_reason_counts: Counter[str] = Counter()
    reject_counts: Counter[str] = Counter()
    for row in selected:
        action_counts.update(row.get("scale_state_action_counts", {}))
        stage_counts.update(row.get("scale_state_stage_counts", {}))
        gate_reason_counts.update(row.get("scale_state_gate_reason_counts", {}))
        reject_counts.update(row.get("semantic_merge_reject_reason_counts", {}))
    return {
        "case": case,
        "chunks": [int(row["chunk"]) for row in selected],
        "trace_rows": int(sum(int(row.get("trace_rows", 0)) for row in selected)),
        "scale_state_active_rows": int(sum(int(row.get("scale_state_active_rows", 0)) for row in selected)),
        "scale_state_action_counts": dict(sorted(action_counts.items())),
        "scale_state_stage_counts": dict(sorted(stage_counts.items())),
        "scale_state_gate_reason_counts": dict(sorted(gate_reason_counts.items())),
        "semantic_merge_reject_reason_counts": dict(sorted(reject_counts.items())),
    }


def _eval_chunk_map(eval_summary: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in eval_summary.get("chunk_decisions", []) or []:
        try:
            chunk = int(row.get("chunk"))
        except (TypeError, ValueError):
            continue
        out[chunk] = row
    return out


def _render_report(summary: Dict[str, Any]) -> str:
    eval_summary = summary.get("eval_summary", {})
    candidate_trace = summary.get("case_totals", {}).get(summary.get("candidate", ""), {})
    lines = [
        "# ACL2 v81 Phase10 Scale-Side-State Rescue",
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
        "## Candidate Scale-State Evidence",
        "",
        f"- scale_state_active_rows: `{candidate_trace.get('scale_state_active_rows')}`",
        f"- scale_state_action_counts: `{json.dumps(candidate_trace.get('scale_state_action_counts', {}), ensure_ascii=False, sort_keys=True)}`",
        f"- scale_state_stage_counts: `{json.dumps(candidate_trace.get('scale_state_stage_counts', {}), ensure_ascii=False, sort_keys=True)}`",
        f"- scale_state_gate_reason_counts: `{json.dumps(candidate_trace.get('scale_state_gate_reason_counts', {}), ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Per-Chunk Candidate Gates",
        "",
    ]
    chunk_map = _eval_chunk_map({"chunk_decisions": summary.get("chunk_decisions", [])})
    for chunk in summary.get("chunks", []):
        row = chunk_map.get(int(chunk), {})
        lines.append(
            "- chunk{chunk:02d}: head_tail_pass=`{head}`, overlap_pass=`{overlap}`, "
            "head_improve=`{hi}`, overlap_improve=`{oi}`".format(
                chunk=int(chunk),
                head=row.get("head_tail_phaseE_chunk_pass"),
                overlap=row.get("overlap_phaseE_chunk_pass"),
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
    parser.add_argument("--eval-json", type=Path, default=DEFAULT_EVAL_JSON)
    parser.add_argument("--chunks", default="7,8,9,10")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case name under each chunk dir; repeatable.",
    )
    parser.add_argument("--candidate", default="overlap_outlier")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-report", type=Path, default=None)
    args = parser.parse_args()

    chunks = _parse_csv_ints(args.chunks)
    cases = args.case or [
        "native_no_swa",
        "overlap_outlier",
        "geometry_only",
        "overlap_outlier_random",
        "overlap_outlier_shuffled",
    ]
    trace_rows: List[Dict[str, Any]] = []
    for chunk in chunks:
        for case in cases:
            trace_rows.append(_summarize_trace(args.root, chunk=chunk, case=case))

    eval_summary = _load_json(args.eval_json)
    manifest_path = args.root / "phaseE_merge_run_manifest.json"
    manifest = _load_json(manifest_path)
    case_totals = {case: _case_totals(trace_rows, case) for case in cases}
    out_json = args.out_json or args.root / "scale_side_state_rescue_summary.json"
    out_csv = args.out_csv or args.root / "scale_side_state_trace_audit.csv"
    out_report = args.out_report or args.root / "scale_side_state_rescue_report.md"
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
        "eval_summary": {
            key: value
            for key, value in eval_summary.items()
            if key not in {"run_rows", "chunk_decisions"}
        },
        "case_totals": case_totals,
        "trace_rows": trace_rows,
        "chunk_decisions": eval_summary.get("chunk_decisions", []),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_csv, trace_rows)
    out_report.write_text(_render_report(summary), encoding="utf-8")
    printable = {key: value for key, value in summary.items() if key not in {"trace_rows", "chunk_decisions"}}
    print(json.dumps(_jsonable(printable), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote_json={out_json}")
    print(f"wrote_csv={out_csv}")
    print(f"wrote_report={out_report}")


if __name__ == "__main__":
    main()
