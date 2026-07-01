#!/usr/bin/env python3
"""Summarize v81 seq05 Type-B good-protection coverage results."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v81tf_longwindow_semantic_three_memory_control/"
    "report_final/phase12_typeb_coverage_seq05_goodprotect"
)
PLAIN_ROOT = ROOT / "seq05_abs_support_overlap_outlier_chunks009_024_083"
RANDOMGAP_ROOT = ROOT / "seq05_abs_support_overlap_outlier_randomgap_chunks009_024_083"
PLAIN_PHASEE = PLAIN_ROOT / "seq05_typeb_goodprotect_phaseE_summary.json"
RANDOMGAP_PHASEE = RANDOMGAP_ROOT / "seq05_typeb_goodprotect_randomgap_phaseE_summary.json"
OUT_JSON = ROOT / "seq05_typeb_goodprotect_summary.json"
OUT_REPORT = ROOT / "seq05_typeb_goodprotect_report.md"
OUT_CSV = ROOT / "seq05_typeb_goodprotect_trace.csv"


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


def _phasee_block(name: str, root: Path, phasee_path: Path) -> dict[str, Any]:
    phasee = _read_json(phasee_path)
    manifest = _manifest(root)
    return {
        "name": name,
        "root": str(root),
        "phasee_json": str(phasee_path),
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


def _target_trace_rows(root: Path, variant: str, case: str = "overlap_outlier") -> list[dict[str, Any]]:
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
                    "support_available": row.get("semantic_merge_overlap_support_available"),
                    "support_path": row.get("semantic_merge_overlap_support_path"),
                    "support_mean": _finite(row.get("semantic_merge_overlap_support_mean")),
                    "support_q90": _finite(row.get("semantic_merge_overlap_support_q90")),
                    "support_weight_mean": _finite(row.get("semantic_merge_overlap_support_weight_mean")),
                    "support_weighted_mass": _finite(row.get("semantic_merge_overlap_support_weighted_mass")),
                    "remaining_valid_ratio": _finite(row.get("semantic_merge_remaining_valid_ratio")),
                    "blend_scale": _finite(row.get("semantic_merge_blend_scale")),
                    "candidate_scale": _finite(row.get("semantic_merge_candidate_scale")),
                    "native_overlap_residual": _finite(row.get("semantic_merge_native_overlap_residual")),
                    "semantic_overlap_residual": _finite(row.get("semantic_merge_overlap_residual")),
                    "final_overlap_residual": _finite(row.get("semantic_merge_final_overlap_residual")),
                    "promotion_gate_pass": row.get("semantic_merge_promotion_gate_pass"),
                    "promotion_gate_reason": row.get("semantic_merge_promotion_gate_reason"),
                    "random_qscale_gap": _finite(row.get("semantic_merge_promotion_random_qscale_gap")),
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


def _chunk_notes(block: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = block.get("chunk_decisions")
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def main() -> None:
    plain = _phasee_block("seq05_abs_support_overlap_outlier", PLAIN_ROOT, PLAIN_PHASEE)
    randomgap = _phasee_block("seq05_abs_support_overlap_outlier_randomgap", RANDOMGAP_ROOT, RANDOMGAP_PHASEE)
    trace_rows = _target_trace_rows(PLAIN_ROOT, "plain") + _target_trace_rows(RANDOMGAP_ROOT, "randomgap")
    _write_csv(OUT_CSV, trace_rows)

    support_masses = [
        float(row["support_weighted_mass"])
        for row in trace_rows
        if row.get("support_weighted_mass") is not None
    ]
    plain_chunks_with_negative = [
        int(row.get("chunk"))
        for row in _chunk_notes(plain)
        if _finite(row.get("head_tail_improvement_vs_baseline_ratio")) is not None
        and float(row.get("head_tail_improvement_vs_baseline_ratio")) < 0
    ]

    summary = {
        "schema": "acl2_v81_seq05_goodprotect_summary_v1",
        "root": str(ROOT),
        "decision": "No-Go_seq05_goodprotect_failed_phaseE_and_controls",
        "phase12_typeb_seq05_goodprotect_pass": False,
        "plain": plain,
        "randomgap": randomgap,
        "trace_csv": str(OUT_CSV),
        "support_map_count": len(list((ROOT / "seq05_support_maps_abs_error").glob("*.pt"))),
        "trace_target_row_count": len(trace_rows),
        "support_weighted_mass_min": min(support_masses) if support_masses else None,
        "support_weighted_mass_max": max(support_masses) if support_masses else None,
        "plain_negative_head_tail_chunks": plain_chunks_with_negative,
        "blocker": (
            "Seq05 support coverage exists, but the semantic overlap-outlier candidate does not pass "
            "PhaseE or controls; random-gap promotion collapses to native/no-improvement rows."
        ),
    }
    OUT_JSON.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# ACL2 v81 Seq05 Type-B Good-Protection Coverage",
        "",
        f"Decision: `{summary['decision']}`",
        f"Gate pass: `{summary['phase12_typeb_seq05_goodprotect_pass']}`",
        "",
        "## Runs",
        "",
    ]
    for block in [plain, randomgap]:
        lines.extend(
            [
                f"### {block['name']}",
                "",
                f"- job_count: `{block['job_count']}`",
                f"- failed_jobs_count: `{block['failed_jobs_count']}`",
                f"- chunks: `{block['chunks']}`",
                f"- phaseE_gate_pass: `{block['phaseE_gate_pass']}`",
                f"- head_tail_pass_count: `{block['head_tail_pass_count']}`",
                f"- overlap_pass_count: `{block['overlap_pass_count']}`",
                f"- head_tail_median_improvement_vs_baseline_ratio: `{block['head_tail_median_improvement_vs_baseline_ratio']}`",
                f"- overlap_median_improvement_vs_baseline_ratio: `{block['overlap_median_improvement_vs_baseline_ratio']}`",
                "",
            ]
        )
        for row in _chunk_notes(block):
            lines.append(
                "- chunk {chunk}: head={head_tail_improvement_vs_baseline_ratio}, overlap={overlap_improvement_vs_baseline_ratio}, "
                "head_beats_controls={head_tail_beats_controls}, overlap_beats_controls={overlap_beats_controls}".format(
                    **row
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Support Trace",
            "",
            f"- support_map_count: `{summary['support_map_count']}`",
            f"- trace_target_row_count: `{summary['trace_target_row_count']}`",
            f"- support_weighted_mass_min: `{summary['support_weighted_mass_min']}`",
            f"- support_weighted_mass_max: `{summary['support_weighted_mass_max']}`",
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
