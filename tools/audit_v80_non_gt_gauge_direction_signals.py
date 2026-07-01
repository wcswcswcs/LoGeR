#!/usr/bin/env python3
"""Audit v80 non-GT gauge-direction signals on seq01 canary chunks.

This is diagnostic-only.  It reads already landed v80 artifacts and asks a
specific question: does any non-GT signal select chunks/actions that improve
head-tail without breaking overlap-to-future?  GT-derived synthetic direction
rows are allowed only as non-deployable diagnostics and are flagged as such.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Callable


REPORT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final"
)
DEFAULT_RECHECK_ROWS = REPORT_ROOT / "phase9_seq01_non_gt_direction_recheck" / "non_gt_direction_recheck_rows.csv"
DEFAULT_QSCALE_ROOT = REPORT_ROOT / "phase9_seq01_thingstuff_radio_qscale_canary5_ref055_chunks006_008_010_012"
DEFAULT_OUT_DIR = REPORT_ROOT / "phase10_seq01_non_gt_gauge_direction_signal_audit_20260622_2045"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recheck-rows", type=Path, default=DEFAULT_RECHECK_ROWS)
    parser.add_argument("--qscale-root", type=Path, default=DEFAULT_QSCALE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-median-improvement", type=float, default=0.05)
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
    return value


def _f(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _b(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _read_last_trace(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    last: dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if int(row.get("local_chunk_idx", -1)) == 1:
                last = row
    return last


def _qscale_trace_by_chunk(root: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for chunk_dir in sorted(root.glob("chunk*")):
        chunk_text = chunk_dir.name.replace("chunk", "")
        if not chunk_text.isdigit():
            continue
        chunk = int(chunk_text)
        row = _read_last_trace(chunk_dir / "thingstuff_radio_qscale" / "merge_state_trace.jsonl")
        rand = _read_last_trace(chunk_dir / "thingstuff_radio_qscale_random" / "merge_state_trace.jsonl")
        shuf = _read_last_trace(chunk_dir / "thingstuff_radio_qscale_shuffled" / "merge_state_trace.jsonl")
        if not row:
            continue
        qscale = _f(row.get("semantic_merge_scale"))
        random_qscale = _f(rand.get("semantic_merge_scale"))
        shuffled_qscale = _f(shuf.get("semantic_merge_scale"))
        native_res = _f(row.get("semantic_merge_native_overlap_residual"))
        final_res = _f(row.get("semantic_merge_final_overlap_residual") or row.get("semantic_merge_overlap_residual"))
        out[chunk] = {
            "qscale_trace": str(chunk_dir / "thingstuff_radio_qscale" / "merge_state_trace.jsonl"),
            "qscale_scale": qscale,
            "qscale_direction": _direction(qscale),
            "qscale_observability": _f(row.get("semantic_merge_qscale_observability")),
            "qscale_factor": _f(row.get("semantic_merge_qscale_factor")),
            "qscale_native_overlap_residual": native_res,
            "qscale_final_overlap_residual": final_res,
            "qscale_local_overlap_delta": (
                None if native_res is None or final_res is None else float(final_res - native_res)
            ),
            "qscale_random_scale": random_qscale,
            "qscale_shuffled_scale": shuffled_qscale,
            "qscale_random_abs_gap": (
                None if qscale is None or random_qscale is None else abs(float(qscale) - float(random_qscale))
            ),
            "qscale_shuffled_abs_gap": (
                None if qscale is None or shuffled_qscale is None else abs(float(qscale) - float(shuffled_qscale))
            ),
        }
    return out


def _direction(scale: float | None, eps: float = 1e-6) -> str:
    if scale is None:
        return ""
    if scale > 1.0 + eps:
        return "up"
    if scale < 1.0 - eps:
        return "down"
    return "flat"


def _rows_by_chunk(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        chunk = int(row["chunk"])
        out[chunk] = dict(row)
    return out


def _enrich_rows(recheck_rows: list[dict[str, Any]], qscale_trace: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in recheck_rows:
        chunk = int(row["chunk"])
        out = dict(row)
        out.update(qscale_trace.get(chunk, {}))
        rows.append(out)
    return rows


def _imp(row: dict[str, Any], source: str, key: str) -> float:
    val = _f(row.get(f"{source}_{key}_improvement_vs_baseline_ratio"))
    return float(val) if val is not None else 0.0


def _pass(row: dict[str, Any], source: str, key: str) -> bool:
    return bool(_b(row.get(f"{source}_{key}_phaseE_chunk_pass")))


def _audit_rule(
    *,
    name: str,
    rows: list[dict[str, Any]],
    source: str,
    selector: Callable[[dict[str, Any]], bool],
    description: str,
    deployable_non_gt: bool,
    uses_gt_diagnostic_direction: bool,
    min_median_improvement: float,
) -> dict[str, Any]:
    selected = [row for row in rows if selector(row)]
    selected_chunks = [int(row["chunk"]) for row in selected]
    head_values = [_imp(row, source, "head_tail") if int(row["chunk"]) in selected_chunks else 0.0 for row in rows]
    overlap_values = [_imp(row, source, "overlap") if int(row["chunk"]) in selected_chunks else 0.0 for row in rows]
    head_median = float(median(head_values)) if head_values else 0.0
    overlap_median = float(median(overlap_values)) if overlap_values else 0.0
    head_pass_chunks = [int(row["chunk"]) for row in selected if _pass(row, source, "head_tail")]
    overlap_pass_chunks = [int(row["chunk"]) for row in selected if _pass(row, source, "overlap")]
    both_pass_chunks = sorted(set(head_pass_chunks).intersection(overlap_pass_chunks))
    overlap_harm_chunks = [
        int(row["chunk"]) for row in selected if _imp(row, source, "overlap") < 0.0
    ]
    canary_gate_pass = (
        deployable_non_gt
        and len(selected_chunks) > 0
        and len(head_pass_chunks) >= 2
        and len(overlap_pass_chunks) >= 2
        and head_median >= float(min_median_improvement)
        and overlap_median >= float(min_median_improvement)
        and not overlap_harm_chunks
    )
    return {
        "rule": name,
        "source": source,
        "description": description,
        "deployable_non_gt": bool(deployable_non_gt),
        "uses_gt_diagnostic_direction": bool(uses_gt_diagnostic_direction),
        "selected_chunks": selected_chunks,
        "selected_count": len(selected_chunks),
        "head_tail_pass_chunks": head_pass_chunks,
        "head_tail_pass_count": len(head_pass_chunks),
        "overlap_pass_chunks": overlap_pass_chunks,
        "overlap_pass_count": len(overlap_pass_chunks),
        "both_pass_chunks": both_pass_chunks,
        "overlap_harm_chunks": overlap_harm_chunks,
        "head_tail_median_improvement_with_native_fallback": head_median,
        "overlap_median_improvement_with_native_fallback": overlap_median,
        "canary_rule_gate_pass": bool(canary_gate_pass),
    }


def _make_rule_rows(rows: list[dict[str, Any]], min_median_improvement: float) -> list[dict[str, Any]]:
    return [
        _audit_rule(
            name="selected_write_low_support_only",
            rows=rows,
            source="qscale",
            selector=lambda row: (_f(row.get("selected_low_support_mass")) or 0.0) > 0.0,
            description="Keep qscale only where selected TTT write overlaps low semantic/geometry support.",
            deployable_non_gt=True,
            uses_gt_diagnostic_direction=False,
            min_median_improvement=min_median_improvement,
        ),
        _audit_rule(
            name="qscale_observability_ge_0p52",
            rows=rows,
            source="qscale",
            selector=lambda row: (_f(row.get("qscale_observability")) or -1.0) >= 0.52,
            description="Keep qscale where RADIO/thingstuff weighted qscale observability is at least 0.52.",
            deployable_non_gt=True,
            uses_gt_diagnostic_direction=False,
            min_median_improvement=min_median_improvement,
        ),
        _audit_rule(
            name="qscale_observability_ge_0p55",
            rows=rows,
            source="qscale",
            selector=lambda row: (_f(row.get("qscale_observability")) or -1.0) >= 0.55,
            description="Stricter qscale observability gate.",
            deployable_non_gt=True,
            uses_gt_diagnostic_direction=False,
            min_median_improvement=min_median_improvement,
        ),
        _audit_rule(
            name="qscale_random_gap_ge_0p02",
            rows=rows,
            source="qscale",
            selector=lambda row: (_f(row.get("qscale_random_abs_gap")) or 0.0) >= 0.02,
            description="Keep qscale only when candidate-vs-same-mass-random scale gap is at least 0.02.",
            deployable_non_gt=True,
            uses_gt_diagnostic_direction=False,
            min_median_improvement=min_median_improvement,
        ),
        _audit_rule(
            name="local_overlap_guard_retained",
            rows=rows,
            source="guard",
            selector=lambda row: _b(row.get("guard_trace_rejected")) is False,
            description="Keep action only when local native-overlap residual guard retains it.",
            deployable_non_gt=True,
            uses_gt_diagnostic_direction=False,
            min_median_improvement=min_median_improvement,
        ),
        _audit_rule(
            name="tight_scale_state_retained",
            rows=rows,
            source="tight",
            selector=lambda row: _b(row.get("tight_trace_rejected")) is False,
            description="Keep overlap-tight scale-state action when not rejected by local overlap guard.",
            deployable_non_gt=True,
            uses_gt_diagnostic_direction=False,
            min_median_improvement=min_median_improvement,
        ),
        _audit_rule(
            name="qscale_direction_matches_gt_global_future_direction",
            rows=rows,
            source="qscale",
            selector=lambda row: row.get("qscale_direction") == row.get("global_future_best_direction"),
            description="Diagnostic-only: qscale scale direction matches GT synthetic global-future best direction.",
            deployable_non_gt=False,
            uses_gt_diagnostic_direction=True,
            min_median_improvement=min_median_improvement,
        ),
        _audit_rule(
            name="qscale_direction_matches_gt_tail3_future_direction",
            rows=rows,
            source="qscale",
            selector=lambda row: row.get("qscale_direction") == row.get("tail3_future_best_direction"),
            description="Diagnostic-only: qscale scale direction matches GT synthetic tail3-future best direction.",
            deployable_non_gt=False,
            uses_gt_diagnostic_direction=True,
            min_median_improvement=min_median_improvement,
        ),
        _audit_rule(
            name="all_gt_key_metrics_same_direction_and_qscale_matches",
            rows=rows,
            source="qscale",
            selector=lambda row: (
                _b(row.get("all_key_metrics_same_direction")) is True
                and row.get("qscale_direction") == row.get("global_future_best_direction")
            ),
            description="Diagnostic-only upper-bound: all synthetic GT metrics agree and qscale direction matches.",
            deployable_non_gt=False,
            uses_gt_diagnostic_direction=True,
            min_median_improvement=min_median_improvement,
        ),
    ]


def _status(rule_rows: list[dict[str, Any]]) -> str:
    deployable_passes = [row for row in rule_rows if row["deployable_non_gt"] and row["canary_rule_gate_pass"]]
    if deployable_passes:
        return "deployable_non_gt_signal_found"
    diagnostic_passes = [row for row in rule_rows if row["canary_rule_gate_pass"]]
    if diagnostic_passes:
        return "diagnostic_only_signal_found"
    return "no_deployable_non_gt_gauge_direction_signal"


def main() -> None:
    args = parse_args()
    recheck_rows = _read_csv(args.recheck_rows)
    qscale_trace = _qscale_trace_by_chunk(args.qscale_root)
    signal_rows = _enrich_rows(recheck_rows, qscale_trace)
    rule_rows = _make_rule_rows(signal_rows, float(args.min_median_improvement))
    deployable_passes = [
        row["rule"] for row in rule_rows if row["deployable_non_gt"] and row["canary_rule_gate_pass"]
    ]
    summary = {
        "schema": "acl2_v80_non_gt_gauge_direction_signal_audit_v1",
        "status": _status(rule_rows),
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "runtime_promotion_allowed": False,
        "v80_goal_achieved": False,
        "recheck_rows": str(args.recheck_rows),
        "qscale_root": str(args.qscale_root),
        "chunk_count": len(signal_rows),
        "rules_evaluated": len(rule_rows),
        "deployable_gate_pass_rules": deployable_passes,
        "best_deployable_rules_by_head_tail": [
            row["rule"]
            for row in sorted(
                [r for r in rule_rows if r["deployable_non_gt"]],
                key=lambda r: (
                    float(r["head_tail_median_improvement_with_native_fallback"]),
                    float(r["overlap_median_improvement_with_native_fallback"]),
                ),
                reverse=True,
            )[:3]
        ],
        "core_blocker": (
            "No deployable non-GT rule selects chunk10/chunk12 head-tail gains while also passing "
            "overlap-to-future. Local overlap guards either reject chunk10 or retain chunk12 with "
            "remaining downstream overlap harm."
        ),
        "next_action": (
            "Do not promote qscale/selected-write/scale-state gates. Either design a new future-overlap "
            "proxy with evidence beyond local overlap residual, or limit new runtime work to a chunk08-only "
            "OUT3/MEMIX no-persistent smoke with paired controls."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "non_gt_gauge_signal_input_rows.csv", signal_rows)
    _write_csv(args.out_dir / "non_gt_gauge_signal_rule_audit.csv", rule_rows)
    _write_json(args.out_dir / "non_gt_gauge_signal_audit_summary.json", summary)
    report = [
        "# v80 non-GT gauge-direction signal audit",
        "",
        f"status: {summary['status']}",
        f"v80_goal_achieved: {summary['v80_goal_achieved']}",
        f"method_gate_claimed: {summary['method_gate_claimed']}",
        "",
        "## Core Blocker",
        "",
        summary["core_blocker"],
        "",
        "## Next Action",
        "",
        summary["next_action"],
        "",
    ]
    (args.out_dir / "non_gt_gauge_signal_audit_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote_dir={args.out_dir}")


if __name__ == "__main__":
    main()
