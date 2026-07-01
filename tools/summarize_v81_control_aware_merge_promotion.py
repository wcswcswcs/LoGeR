#!/usr/bin/env python3
"""Summarize v81 control-aware merge promotion continuation results."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v81tf_longwindow_semantic_three_memory_control/"
    "report_final/phase13_control_aware_merge_promotion/typeb_overlap_outlier_random_qscale_gate_bad"
)
PHASEE_JSON = ROOT / "control_aware_phaseE_summary.json"
MANIFEST_JSON = ROOT / "phaseE_merge_run_manifest.json"
OUT_JSON = ROOT / "control_aware_merge_promotion_summary.json"
OUT_MD = ROOT / "control_aware_merge_promotion_report.md"
OUT_CSV = ROOT / "control_aware_merge_promotion_trace.csv"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
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


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _target_rows(case: str = "overlap_outlier") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace_path in sorted(ROOT.glob(f"chunk*/{case}/merge_state_trace.jsonl")):
        target_chunk = int(trace_path.parts[-3].replace("chunk", ""))
        for row in _read_jsonl(trace_path):
            if not row.get("online_semantic_merge_controller"):
                continue
            if int(row.get("local_chunk_idx", -1)) != 1:
                continue
            rows.append(
                {
                    "target_chunk": target_chunk,
                    "chunk_idx": row.get("chunk_idx"),
                    "fit_reason": row.get("semantic_merge_fit_reason"),
                    "promotion_gate_pass": row.get("semantic_merge_promotion_gate_pass"),
                    "promotion_gate_reason": row.get("semantic_merge_promotion_gate_reason"),
                    "candidate_qscale": _finite(row.get("semantic_merge_promotion_candidate_qscale")),
                    "random_qscale": _finite(row.get("semantic_merge_promotion_random_qscale")),
                    "random_qscale_gap": _finite(row.get("semantic_merge_promotion_random_qscale_gap")),
                    "random_valid_count": row.get("semantic_merge_promotion_random_valid_count"),
                    "native_overlap_guard_rejected": row.get("semantic_merge_native_overlap_guard_rejected"),
                    "residual_safe_projection_accepted": row.get("semantic_merge_residual_safe_projection_accepted"),
                    "semantic_overlap_residual": _finite(row.get("semantic_merge_overlap_residual")),
                    "native_overlap_residual": _finite(row.get("semantic_merge_native_overlap_residual")),
                    "blend_scale": _finite(row.get("semantic_merge_blend_scale")),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    phasee = _read_json(PHASEE_JSON)
    manifest = _read_json(MANIFEST_JSON)
    trace_rows = _target_rows()
    _write_csv(OUT_CSV, trace_rows)

    failed_jobs = manifest.get("failed_jobs")
    if not isinstance(failed_jobs, list):
        failed_jobs = []
    promotion_pass_chunks = [
        int(row["target_chunk"]) for row in trace_rows if bool(row.get("promotion_gate_pass"))
    ]
    promotion_rejected_chunks = [
        int(row["target_chunk"]) for row in trace_rows if row.get("promotion_gate_pass") is False
    ]
    random_gap_values = [
        float(row["random_qscale_gap"]) for row in trace_rows if row.get("random_qscale_gap") is not None
    ]
    guard_rejected_chunks = [
        int(row["target_chunk"]) for row in trace_rows if bool(row.get("native_overlap_guard_rejected"))
    ]
    projection_accepted_chunks = [
        int(row["target_chunk"]) for row in trace_rows if bool(row.get("residual_safe_projection_accepted"))
    ]

    summary = {
        "schema": "acl2_v81_control_aware_merge_promotion_summary_v1",
        "root": str(ROOT),
        "decision": "No-Go_control_aware_merge_promotion_failed_phaseE_and_random_qscale_gap",
        "job_count": manifest.get("job_count"),
        "failed_jobs_count": len(failed_jobs),
        "phaseE_gate_pass": bool(phasee.get("phaseE_gate_pass")),
        "head_tail_pass_count": phasee.get("head_tail_pass_count"),
        "overlap_pass_count": phasee.get("overlap_pass_count"),
        "head_tail_median_improvement_vs_baseline_ratio": phasee.get("head_tail_median_improvement_vs_baseline_ratio"),
        "overlap_median_improvement_vs_baseline_ratio": phasee.get("overlap_median_improvement_vs_baseline_ratio"),
        "missing": phasee.get("missing", []),
        "promotion_gate_policy": "random_qscale_gap",
        "promotion_random_qscale_gap_min": 0.02,
        "promotion_pass_chunks": promotion_pass_chunks,
        "promotion_rejected_chunks": promotion_rejected_chunks,
        "guard_rejected_chunks": guard_rejected_chunks,
        "projection_accepted_chunks": projection_accepted_chunks,
        "random_qscale_gap_values": random_gap_values,
        "max_random_qscale_gap": max(random_gap_values) if random_gap_values else None,
        "trace_csv": str(OUT_CSV),
        "blocker": (
            "The no-GT random-qscale promotion selector rejected the only direct semantic chunks "
            "because candidate qscale was nearly indistinguishable from deterministic random qscale."
        ),
    }
    OUT_JSON.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# ACL2 v81 Control-Aware Merge Promotion Continuation",
        "",
        f"Decision: `{summary['decision']}`",
        f"PhaseE gate pass: `{summary['phaseE_gate_pass']}`",
        "",
        "## Metrics",
        "",
        f"- job_count: `{summary['job_count']}`",
        f"- failed_jobs_count: `{summary['failed_jobs_count']}`",
        f"- head_tail_pass_count: `{summary['head_tail_pass_count']}`",
        f"- overlap_pass_count: `{summary['overlap_pass_count']}`",
        f"- head_tail_median_improvement_vs_baseline_ratio: `{summary['head_tail_median_improvement_vs_baseline_ratio']}`",
        f"- overlap_median_improvement_vs_baseline_ratio: `{summary['overlap_median_improvement_vs_baseline_ratio']}`",
        "",
        "## Promotion Gate",
        "",
        f"- promotion_pass_chunks: `{promotion_pass_chunks}`",
        f"- promotion_rejected_chunks: `{promotion_rejected_chunks}`",
        f"- guard_rejected_chunks: `{guard_rejected_chunks}`",
        f"- projection_accepted_chunks: `{projection_accepted_chunks}`",
        f"- max_random_qscale_gap: `{summary['max_random_qscale_gap']}`",
        "",
        "## Trace Rows",
        "",
    ]
    for row in trace_rows:
        lines.append(
            "- chunk {target_chunk}: reason={promotion_gate_reason}, fit={fit_reason}, "
            "candidate_qscale={candidate_qscale}, random_qscale={random_qscale}, "
            "gap={random_qscale_gap}".format(**row)
        )
    lines.extend(["", "## Interpretation", "", summary["blocker"], "This does not unlock held-out or 704F validation."])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
