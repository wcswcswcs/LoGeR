#!/usr/bin/env python3
"""Apply v41 READ h10/h15 continuation gates to landed durability effects."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


BASE_CANDIDATE = "V31_BASE_H9_REFERENCE"


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _f(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _read_context_empty(run_dir: Path) -> int:
    total = 0
    path = run_dir / "context_skip_summary.jsonl"
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += int(_f(row.get("num_context_empty_source_events"), 0.0))
    return total


def _static_anchor_loss_upper_bound(row: Dict[str, str]) -> float:
    ratios = []
    for key in (
        "frame_attention_mean_context_source_keep_ratio",
        "chunk_attention_mean_context_source_keep_ratio",
    ):
        val = _f(row.get(key), float("nan"))
        if math.isfinite(val):
            ratios.append(val)
    if not ratios:
        return 0.0
    return max(0.0, 1.0 - min(ratios))


def _signal_pass(row: Dict[str, Any]) -> Tuple[bool, str]:
    ate = _f(row.get("ATE_delta_vs_base"))
    roll100 = _f(row.get("rolling_100f_best_delta_vs_base"))
    stress = _f(row.get("intersection_200_300_delta_vs_base"))
    downstream = _f(row.get("intersection_400_600_delta_vs_base"))
    checks = []
    if math.isfinite(ate) and ate <= -1.5:
        checks.append("ATE")
    if math.isfinite(roll100) and roll100 <= -3.0:
        checks.append("rolling100")
    if math.isfinite(stress) and stress <= -5.0 and (not math.isfinite(downstream) or downstream <= 1.0):
        checks.append("stress_window_with_downstream")
    return bool(checks), ",".join(checks)


def _durability(row: Dict[str, Any], h10: Dict[str, Any] | None) -> Tuple[float, str]:
    if not h10:
        return float("nan"), "missing_h10_reference"
    ratios = []
    names = []
    for name, key in (
        ("ATE", "ATE_delta_vs_base"),
        ("rolling100", "rolling_100f_best_delta_vs_base"),
        ("stress", "intersection_200_300_delta_vs_base"),
    ):
        h10_val = _f(h10.get(key))
        h15_val = _f(row.get(key))
        if math.isfinite(h10_val) and math.isfinite(h15_val) and h10_val < 0 and h15_val < 0:
            ratios.append(abs(h15_val) / (abs(h10_val) + 1e-9))
            names.append(name)
    if not ratios:
        return float("nan"), "no_negative_matching_signal"
    best_idx = max(range(len(ratios)), key=lambda i: ratios[i])
    return ratios[best_idx], names[best_idx]


def _boundary_rows(path: Path | None) -> Dict[str, Dict[str, float]]:
    if path is None or not path.exists():
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for row in _read_csv(path):
        run_dir = row.get("run_dir") or row.get("candidate_run") or ""
        if not run_dir:
            continue
        out[run_dir] = {
            "boundary_10f_delta": _f(row.get("mean_boundary_10f_delta_vs_H9")),
            "boundary_20f_delta": _f(row.get("mean_boundary_20f_delta_vs_H9")),
        }
    return out


def _key(row: Dict[str, Any]) -> Tuple[str, int, str]:
    return str(row.get("parent")), int(_f(row.get("chunk"), -1)), str(row.get("candidate"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effects", type=Path, required=True)
    parser.add_argument("--mode", choices=("h10", "h15"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--report-prefix", required=True)
    parser.add_argument("--h10-effects", type=Path)
    parser.add_argument("--boundary-summary-csv", type=Path)
    args = parser.parse_args()

    effects = [r for r in _read_csv(args.effects) if r.get("candidate") != BASE_CANDIDATE]
    h10_lookup: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    if args.h10_effects:
        for row in _read_csv(args.h10_effects):
            if row.get("candidate") != BASE_CANDIDATE:
                h10_lookup[_key(row)] = row
    boundary = _boundary_rows(args.boundary_summary_csv)

    rows: List[Dict[str, Any]] = []
    for row in effects:
        signal, signal_reason = _signal_pass(row)
        context_empty = _read_context_empty(Path(row.get("run_dir", "")))
        static_loss = _static_anchor_loss_upper_bound(row)
        static_control = "STATIC_RESCUE" in str(row.get("candidate", ""))
        static_pass = static_control or static_loss <= 0.15
        downstream = _f(row.get("intersection_400_600_delta_vs_base"))
        downstream_pass = not math.isfinite(downstream) or downstream <= 1.0
        row_out: Dict[str, Any] = dict(row)
        row_out.update({
            "v41_signal_pass": signal,
            "v41_signal_reason": signal_reason,
            "context_empty_source_events": context_empty,
            "context_empty_pass": context_empty == 0,
            "static_anchor_mass_loss_upper_bound_proxy": static_loss,
            "static_anchor_loss_evidence": "upper_bound_from_total_context_source_keep_ratio_not_per_label",
            "static_anchor_pass": static_pass,
            "downstream_400_600_pass": downstream_pass,
        })
        if args.mode == "h15":
            h10 = h10_lookup.get(_key(row))
            durability, durability_metric = _durability(row, h10)
            b = boundary.get(str(row.get("run_dir", "")), {})
            b10 = b.get("boundary_10f_delta", float("nan"))
            b20 = b.get("boundary_20f_delta", float("nan"))
            boundary_available = math.isfinite(b10) or math.isfinite(b20)
            boundary_pass = boundary_available and (not math.isfinite(b10) or b10 <= 0.25) and (not math.isfinite(b20) or b20 <= 0.25)
            row_out.update({
                "durability_ratio": durability,
                "durability_metric": durability_metric,
                "durability_pass": math.isfinite(durability) and durability >= 0.45,
                "boundary_10f_delta": b10,
                "boundary_20f_delta": b20,
                "boundary_evidence_available": boundary_available,
                "boundary_pass": boundary_pass,
            })
            gate = bool(signal and context_empty == 0 and downstream_pass and row_out["durability_pass"] and boundary_pass)
        else:
            gate = bool(signal and context_empty == 0 and static_pass)
        row_out["v41_gate_pass"] = gate
        rows.append(row_out)

    _write_csv(args.out_dir / f"{args.report_prefix}_v41_gate_rows.csv", rows)
    pass_rows = [r for r in rows if bool(r.get("v41_gate_pass"))]
    best_ate = min(rows, key=lambda r: _f(r.get("ATE_delta_vs_base"), float("inf")), default=None)
    best_roll = min(rows, key=lambda r: _f(r.get("rolling_100f_best_delta_vs_base"), float("inf")), default=None)
    best_stress = min(rows, key=lambda r: _f(r.get("intersection_200_300_delta_vs_base"), float("inf")), default=None)
    summary = {
        "mode": args.mode,
        "rows": len(rows),
        "gate_pass": bool(pass_rows),
        "gate_pass_rows": [
            {"parent": r.get("parent"), "chunk": int(_f(r.get("chunk"), -1)), "candidate": r.get("candidate"), "reason": r.get("v41_signal_reason")}
            for r in pass_rows
        ],
        "best_ATE_candidate": best_ate.get("candidate") if best_ate else None,
        "best_ATE_parent": best_ate.get("parent") if best_ate else None,
        "best_ATE_chunk": int(_f(best_ate.get("chunk"), -1)) if best_ate else None,
        "best_ATE_delta_vs_base": _f(best_ate.get("ATE_delta_vs_base")) if best_ate else None,
        "best_rolling_100f_candidate": best_roll.get("candidate") if best_roll else None,
        "best_rolling_100f_parent": best_roll.get("parent") if best_roll else None,
        "best_rolling_100f_chunk": int(_f(best_roll.get("chunk"), -1)) if best_roll else None,
        "best_rolling_100f_best_delta": _f(best_roll.get("rolling_100f_best_delta_vs_base")) if best_roll else None,
        "best_stress_candidate": best_stress.get("candidate") if best_stress else None,
        "best_stress_parent": best_stress.get("parent") if best_stress else None,
        "best_stress_chunk": int(_f(best_stress.get("chunk"), -1)) if best_stress else None,
        "best_stress_delta": _f(best_stress.get("intersection_200_300_delta_vs_base")) if best_stress else None,
        "thresholds": {
            "ATE_delta": -1.5,
            "rolling100_delta": -3.0,
            "stress_delta": -5.0,
            "downstream_400_600_max": 1.0,
            "static_anchor_mass_loss_upper_bound_max": 0.15,
            "h15_durability_ratio_min": 0.45,
            "h15_boundary_delta_max": 0.25,
        },
    }
    _write_json(args.out_dir / f"{args.report_prefix}_v41_gate_summary.json", summary)

    lines = [
        f"# {args.report_prefix} v41 READ Gate Report",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "| Parent | Chunk | Candidate | ATE delta | rolling100 best | stress delta | downstream | signal | static/context | gate |",
        "|---|---:|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row.get('parent')}` | {int(_f(row.get('chunk'), -1))} | `{row.get('candidate')}` | "
            f"{_f(row.get('ATE_delta_vs_base')):.10f} | {_f(row.get('rolling_100f_best_delta_vs_base')):.10f} | "
            f"{_f(row.get('intersection_200_300_delta_vs_base')):.10f} | {_f(row.get('intersection_400_600_delta_vs_base')):.10f} | "
            f"`{row.get('v41_signal_reason')}` | `ctx={row.get('context_empty_source_events')}, static_ub={_f(row.get('static_anchor_mass_loss_upper_bound_proxy')):.4f}` | "
            f"`{bool(row.get('v41_gate_pass'))}` |"
        )
    lines.append("")
    (args.out_dir / f"{args.report_prefix}_v41_gate_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
