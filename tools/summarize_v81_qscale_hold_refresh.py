#!/usr/bin/env python3
"""Summarize v81 qscale hold-refresh continuation results."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v81tf_longwindow_semantic_three_memory_control/"
    "report_final/phase14_qscale_hold_refresh"
)
DEFAULT_ROOT = ROOT / "typeb_overlap_outlier_qscale_hold_refresh_bad"
STRICT_ROOT = ROOT / "typeb_overlap_outlier_qscale_hold_refresh_strict_unitref_bad"
DEFAULT_PHASEE = DEFAULT_ROOT / "qscale_hold_phaseE_summary.json"
STRICT_PHASEE = STRICT_ROOT / "qscale_hold_strict_phaseE_summary.json"
OUT_JSON = ROOT / "qscale_hold_refresh_summary.json"
OUT_REPORT = ROOT / "qscale_hold_refresh_report.md"
OUT_CSV = ROOT / "qscale_hold_refresh_trace.csv"


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
            data = json.loads(raw)
            if isinstance(data, dict):
                rows.append(data)
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


def _manifest(root: Path) -> dict[str, Any]:
    return _read_json(root / "phaseE_merge_run_manifest.json")


def _failed_count(manifest: Mapping[str, Any]) -> int:
    failed = manifest.get("failed_jobs")
    return len(failed) if isinstance(failed, list) else 0


def _block(name: str, root: Path, phasee_path: Path, qscale_reference: float) -> dict[str, Any]:
    phasee = _read_json(phasee_path)
    manifest = _manifest(root)
    return {
        "name": name,
        "root": str(root),
        "phasee_json": str(phasee_path),
        "qscale_reference": qscale_reference,
        "job_count": manifest.get("job_count"),
        "failed_jobs_count": _failed_count(manifest),
        "chunks": manifest.get("chunks"),
        "cases": manifest.get("cases"),
        "phaseE_gate_pass": bool(phasee.get("phaseE_gate_pass")),
        "head_tail_pass_count": phasee.get("head_tail_pass_count"),
        "overlap_pass_count": phasee.get("overlap_pass_count"),
        "head_tail_median_improvement_vs_baseline_ratio": phasee.get(
            "head_tail_median_improvement_vs_baseline_ratio"
        ),
        "overlap_median_improvement_vs_baseline_ratio": phasee.get(
            "overlap_median_improvement_vs_baseline_ratio"
        ),
        "missing": phasee.get("missing", []),
        "chunk_decisions": phasee.get("chunk_decisions", []),
    }


def _trace_rows(root: Path, variant: str, case: str = "overlap_outlier") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace_path in sorted(root.glob(f"chunk*/{case}/merge_state_trace.jsonl")):
        target_chunk = int(trace_path.parts[-3].replace("chunk", ""))
        for row in _read_jsonl(trace_path):
            if not row.get("online_semantic_merge_controller"):
                continue
            if int(row.get("local_chunk_idx", -1)) != 1:
                continue
            rows.append(
                {
                    "variant": variant,
                    "target_chunk": target_chunk,
                    "fit_reason": row.get("semantic_merge_fit_reason"),
                    "guard_rejected": row.get("semantic_merge_native_overlap_guard_rejected"),
                    "projection_accepted": row.get("semantic_merge_residual_safe_projection_accepted"),
                    "qscale_reference": _finite(row.get("semantic_merge_qscale_reference")),
                    "qscale_min_factor": _finite(row.get("semantic_merge_qscale_min_factor")),
                    "qscale_observability": _finite(row.get("semantic_merge_qscale_observability")),
                    "qscale_factor": _finite(row.get("semantic_merge_qscale_factor")),
                    "effective_blend_alpha": _finite(row.get("semantic_merge_qscale_effective_blend_alpha")),
                    "blend_scale": _finite(row.get("semantic_merge_blend_scale")),
                    "candidate_scale": _finite(row.get("semantic_merge_candidate_scale")),
                    "native_overlap_residual": _finite(row.get("semantic_merge_native_overlap_residual")),
                    "final_overlap_residual": _finite(row.get("semantic_merge_final_overlap_residual")),
                    "semantic_overlap_residual": _finite(row.get("semantic_merge_overlap_residual")),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _qscale_values(rows: list[dict[str, Any]], variant: str, key: str) -> list[float]:
    vals = [
        float(row[key])
        for row in rows
        if row.get("variant") == variant and row.get(key) is not None
    ]
    return vals


def main() -> None:
    default = _block("default_qscale_reference_0_35", DEFAULT_ROOT, DEFAULT_PHASEE, 0.35)
    strict = _block("strict_qscale_reference_1_0", STRICT_ROOT, STRICT_PHASEE, 1.0)
    trace_rows = _trace_rows(DEFAULT_ROOT, "default") + _trace_rows(STRICT_ROOT, "strict")
    _write_csv(OUT_CSV, trace_rows)

    default_factors = _qscale_values(trace_rows, "default", "qscale_factor")
    strict_factors = _qscale_values(trace_rows, "strict", "qscale_factor")
    default_guard_rejected = [
        int(row["target_chunk"])
        for row in trace_rows
        if row.get("variant") == "default" and bool(row.get("guard_rejected"))
    ]
    strict_guard_rejected = [
        int(row["target_chunk"])
        for row in trace_rows
        if row.get("variant") == "strict" and bool(row.get("guard_rejected"))
    ]

    summary = {
        "schema": "acl2_v81_qscale_hold_refresh_summary_v1",
        "root": str(ROOT),
        "decision": "No-Go_qscale_hold_refresh_failed_phaseE",
        "phase14_qscale_hold_refresh_pass": False,
        "default": default,
        "strict": strict,
        "trace_csv": str(OUT_CSV),
        "default_qscale_factor_min": min(default_factors) if default_factors else None,
        "default_qscale_factor_max": max(default_factors) if default_factors else None,
        "strict_qscale_factor_min": min(strict_factors) if strict_factors else None,
        "strict_qscale_factor_max": max(strict_factors) if strict_factors else None,
        "default_guard_rejected_chunks": default_guard_rejected,
        "strict_guard_rejected_chunks": strict_guard_rejected,
        "blocker": (
            "The default qscale reference did not attenuate the action because all target qscale "
            "values exceeded 0.35. A stricter unit reference attenuated alpha, but PhaseE still failed "
            "and chunks 7/8 remained native-overlap rejected."
        ),
    }
    OUT_JSON.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# ACL2 v81 Qscale Hold-Refresh Continuation",
        "",
        f"Decision: `{summary['decision']}`",
        f"Gate pass: `{summary['phase14_qscale_hold_refresh_pass']}`",
        "",
        "## Runs",
        "",
    ]
    for block in [default, strict]:
        lines.extend(
            [
                f"### {block['name']}",
                "",
                f"- qscale_reference: `{block['qscale_reference']}`",
                f"- job_count: `{block['job_count']}`",
                f"- failed_jobs_count: `{block['failed_jobs_count']}`",
                f"- phaseE_gate_pass: `{block['phaseE_gate_pass']}`",
                f"- head_tail_pass_count: `{block['head_tail_pass_count']}`",
                f"- overlap_pass_count: `{block['overlap_pass_count']}`",
                f"- head_tail_median_improvement_vs_baseline_ratio: `{block['head_tail_median_improvement_vs_baseline_ratio']}`",
                f"- overlap_median_improvement_vs_baseline_ratio: `{block['overlap_median_improvement_vs_baseline_ratio']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Trace",
            "",
            f"- default_qscale_factor_min/max: `{summary['default_qscale_factor_min']}` / `{summary['default_qscale_factor_max']}`",
            f"- strict_qscale_factor_min/max: `{summary['strict_qscale_factor_min']}` / `{summary['strict_qscale_factor_max']}`",
            f"- default_guard_rejected_chunks: `{default_guard_rejected}`",
            f"- strict_guard_rejected_chunks: `{strict_guard_rejected}`",
            "",
            "## Interpretation",
            "",
            summary["blocker"],
            "This is not a method success and does not unlock held-out or 704F validation.",
        ]
    )
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
