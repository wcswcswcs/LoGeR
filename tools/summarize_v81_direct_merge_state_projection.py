#!/usr/bin/env python3
"""Summarize v81 direct merge-state projection continuation results."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v81tf_longwindow_semantic_three_memory_control/"
    "report_final/phase13_direct_merge_state_projection/typeb_overlap_outlier_projection_bad"
)
SUMMARY_JSON = ROOT / "direct_projection_phaseE_summary.json"
MANIFEST_JSON = ROOT / "phaseE_merge_run_manifest.json"
OUT_JSON = ROOT / "direct_merge_state_projection_summary.json"
OUT_REPORT = ROOT / "direct_merge_state_projection_report.md"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _target_semantic_rows(case: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace_path in sorted(ROOT.glob(f"chunk*/{case}/merge_state_trace.jsonl")):
        chunk_text = trace_path.parts[-3].replace("chunk", "")
        for row in _read_jsonl(trace_path):
            if not row.get("online_semantic_merge_controller"):
                continue
            if int(row.get("local_chunk_idx", -1)) != 1:
                continue
            out = dict(row)
            out["target_chunk"] = int(chunk_text)
            rows.append(out)
    return rows


def _chunk_decision_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = summary.get("chunk_decisions")
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _projection_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    guard_rejected = [
        int(row["target_chunk"])
        for row in rows
        if bool(row.get("semantic_merge_native_overlap_guard_rejected"))
    ]
    projection_accepted = [
        int(row["target_chunk"])
        for row in rows
        if bool(row.get("semantic_merge_residual_safe_projection_accepted"))
    ]
    direct_chunks = [
        int(row["target_chunk"])
        for row in rows
        if not bool(row.get("semantic_merge_native_overlap_guard_rejected"))
    ]
    scales = {
        f"chunk{int(row['target_chunk']):02d}": _finite(row.get("semantic_merge_blend_scale"))
        for row in rows
    }
    residuals = {
        f"chunk{int(row['target_chunk']):02d}": _finite(row.get("semantic_merge_overlap_residual"))
        for row in rows
    }
    native_residuals = {
        f"chunk{int(row['target_chunk']):02d}": _finite(row.get("semantic_merge_native_overlap_residual"))
        for row in rows
    }
    reasons = {
        f"chunk{int(row['target_chunk']):02d}": row.get("semantic_merge_fit_reason")
        for row in rows
    }
    return {
        "target_row_count": len(rows),
        "guard_rejected_chunks": guard_rejected,
        "direct_semantic_chunks": direct_chunks,
        "projection_accepted_chunks": projection_accepted,
        "blend_scale_by_chunk": scales,
        "semantic_overlap_residual_by_chunk": residuals,
        "native_overlap_residual_by_chunk": native_residuals,
        "fit_reason_by_chunk": reasons,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    phasee = _read_json(SUMMARY_JSON)
    manifest = _read_json(MANIFEST_JSON)
    decisions = _chunk_decision_rows(phasee)
    candidate_trace = _projection_summary(_target_semantic_rows("overlap_outlier"))
    random_trace = _projection_summary(_target_semantic_rows("overlap_outlier_random"))
    shuffled_trace = _projection_summary(_target_semantic_rows("overlap_outlier_shuffled"))

    chunk_notes: list[dict[str, Any]] = []
    for row in decisions:
        chunk = int(row.get("chunk"))
        chunk_notes.append(
            {
                "chunk": chunk,
                "head_tail_improvement_vs_baseline_ratio": row.get("head_tail_improvement_vs_baseline_ratio"),
                "overlap_improvement_vs_baseline_ratio": row.get("overlap_improvement_vs_baseline_ratio"),
                "head_tail_beats_controls": row.get("head_tail_beats_controls"),
                "overlap_beats_controls": row.get("overlap_beats_controls"),
                "head_tail_phaseE_chunk_pass": row.get("head_tail_phaseE_chunk_pass"),
                "overlap_phaseE_chunk_pass": row.get("overlap_phaseE_chunk_pass"),
                "fit_reason": candidate_trace["fit_reason_by_chunk"].get(f"chunk{chunk:02d}"),
                "projection_accepted": chunk in candidate_trace["projection_accepted_chunks"],
            }
        )
    _write_csv(ROOT / "direct_merge_state_projection_chunk_notes.csv", chunk_notes)

    failed_jobs = manifest.get("failed_jobs")
    if not isinstance(failed_jobs, list):
        failed_jobs = []
    summary = {
        "schema": "acl2_v81_direct_merge_state_projection_summary_v1",
        "root": str(ROOT),
        "decision": "No-Go_direct_merge_state_projection_failed_phaseE_controls",
        "phaseE_gate_pass": bool(phasee.get("phaseE_gate_pass")),
        "head_tail_pass_count": phasee.get("head_tail_pass_count"),
        "overlap_pass_count": phasee.get("overlap_pass_count"),
        "head_tail_median_improvement_vs_baseline_ratio": phasee.get("head_tail_median_improvement_vs_baseline_ratio"),
        "overlap_median_improvement_vs_baseline_ratio": phasee.get("overlap_median_improvement_vs_baseline_ratio"),
        "missing": phasee.get("missing", []),
        "job_count": manifest.get("job_count"),
        "failed_jobs_count": len(failed_jobs),
        "candidate_trace": candidate_trace,
        "random_trace": random_trace,
        "shuffled_trace": shuffled_trace,
        "chunk_notes_csv": str(ROOT / "direct_merge_state_projection_chunk_notes.csv"),
        "rule": (
            "Continuation tests direct merge/gauge state via semantic overlap-outlier transforms with "
            "native-overlap guard plus residual-safe projection enabled for candidate and controls."
        ),
        "blocker": (
            "No target chunk accepted residual-safe projection; chunks 7/8 fell back, chunks 9/10 "
            "changed merge state but failed PhaseE thresholds or controls."
        ),
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# ACL2 v81 Direct Merge-State Projection Continuation",
        "",
        f"Decision: `{summary['decision']}`",
        f"PhaseE gate pass: `{summary['phaseE_gate_pass']}`",
        "",
        "## Metrics",
        "",
        f"- head_tail_pass_count: `{summary['head_tail_pass_count']}`",
        f"- overlap_pass_count: `{summary['overlap_pass_count']}`",
        f"- head_tail_median_improvement_vs_baseline_ratio: `{summary['head_tail_median_improvement_vs_baseline_ratio']}`",
        f"- overlap_median_improvement_vs_baseline_ratio: `{summary['overlap_median_improvement_vs_baseline_ratio']}`",
        f"- failed_jobs_count: `{summary['failed_jobs_count']}`",
        "",
        "## Trace",
        "",
        f"- guard_rejected_chunks: `{candidate_trace['guard_rejected_chunks']}`",
        f"- direct_semantic_chunks: `{candidate_trace['direct_semantic_chunks']}`",
        f"- projection_accepted_chunks: `{candidate_trace['projection_accepted_chunks']}`",
        "",
        "## Chunk Notes",
        "",
    ]
    for row in chunk_notes:
        lines.append(
            "- chunk {chunk}: head={head_tail_improvement_vs_baseline_ratio}, "
            "overlap={overlap_improvement_vs_baseline_ratio}, head_beats_controls={head_tail_beats_controls}, "
            "overlap_beats_controls={overlap_beats_controls}, fit_reason={fit_reason}, "
            "projection_accepted={projection_accepted}".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            summary["blocker"],
            "This is not a method success and does not unlock held-out or 704F validation.",
        ]
    )
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
