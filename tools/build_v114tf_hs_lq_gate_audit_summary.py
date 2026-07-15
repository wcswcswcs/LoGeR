#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


DEFAULT_RESULTS = Path("results/acl2_v114tf_semantic_internal_evidence_quality_memory_influence_control")
NUMERIC_KEYS = [
    "num_local_rows",
    "spatial_patch_count",
    "gate_mean",
    "gate_std",
    "gate_min",
    "gate_max",
    "gate_row_mean_mean",
    "gate_row_mean_std",
    "changed_token_count_abs_gt_1e_4",
    "semantic_risk_mean",
    "semantic_stable_mean",
    "internal_q_mean",
    "internal_q_std",
    "internal_mismatch_mean",
    "internal_mismatch_active_fraction",
]


def finite_float(value: str) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in {float("inf"), float("-inf")}:
        return None
    return out


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": float(min(values)),
        "median": float(statistics.median(values)),
        "mean": float(statistics.fmean(values)),
        "max": float(max(values)),
    }


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    out: dict[str, Any] = {"row_count": len(rows)}
    actions = sorted({row.get("action", "") for row in rows})
    controls = sorted({row.get("control", "") for row in rows})
    rowmean_values = sorted({row.get("rowmean_neutral", "") for row in rows})
    semint_values = sorted({row.get("semantic_internal_coupled", "") for row in rows})
    out["actions"] = actions
    out["controls"] = controls
    out["rowmean_neutral_values"] = rowmean_values
    out["semantic_internal_coupled_values"] = semint_values
    for key in NUMERIC_KEYS:
        vals = [v for row in rows if (v := finite_float(row.get(key, ""))) is not None]
        stats = summarize_values(vals)
        for stat_key, stat_value in stats.items():
            out[f"{key}_{stat_key}"] = stat_value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize v114 HS-LQ action gate audit CSV files.")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS))
    parser.add_argument("--case-glob", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    root = Path(args.results_root).resolve()
    diagnostics = root / "diagnostics"
    case_dirs = sorted(p for p in diagnostics.glob(args.case_glob) if p.is_dir())
    case_summaries: dict[str, Any] = {}
    all_rows: list[dict[str, str]] = []
    missing: list[str] = []
    for case_dir in case_dirs:
        csv_path = case_dir / "hs_lq_action_gate_rows.csv"
        if not csv_path.exists():
            missing.append(str(csv_path))
            continue
        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        case_summaries[case_dir.name] = summarize_rows(rows)
        all_rows.extend(rows)

    summary = {
        "schema": "acl2_v114tf_hs_lq_gate_audit_summary_v1",
        "results_root": str(root),
        "case_glob": args.case_glob,
        "case_count": len(case_summaries),
        "missing_csv": missing,
        "aggregate": summarize_rows(all_rows),
        "cases": case_summaries,
    }
    out_path = diagnostics / f"{args.output_prefix}_gate_audit_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out_path), "case_count": len(case_summaries), "row_count": len(all_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
