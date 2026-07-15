#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


NUMERIC_KEYS = [
    "metric_readout_feature_norm",
    "predicted_metric_scale",
    "predicted_metric_scale_delta",
    "chunk_semantic_risk",
    "chunk_stable_mass",
]


def finite(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if out != out or out in {float("inf"), float("-inf")}:
        return None
    return out


def stats(values: list[float]) -> dict[str, float | int | None]:
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
    out: dict[str, Any] = {"row_count": len(rows), "seqs": sorted({row.get("seq", "") for row in rows})}
    for key in NUMERIC_KEYS:
        vals = [v for row in rows if (v := finite(row.get(key))) is not None]
        key_stats = stats(vals)
        for stat_key, stat_value in key_stats.items():
            out[f"{key}_{stat_key}"] = stat_value
        out[f"{key}_missing_count"] = len(rows) - len(vals)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize v114 HorizonStream MRT trace rows.")
    parser.add_argument("--results-root", default="results/acl2_v114tf_semantic_internal_evidence_quality_memory_influence_control")
    parser.add_argument("--case-glob", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    root = Path(args.results_root).resolve()
    diagnostics = root / "diagnostics"
    case_dirs = sorted(p for p in diagnostics.glob(args.case_glob) if p.is_dir())
    all_rows: list[dict[str, str]] = []
    cases: dict[str, Any] = {}
    missing: list[str] = []
    for case_dir in case_dirs:
        csv_path = case_dir / "hs_mrt_readout_probe_rows.csv"
        if not csv_path.exists():
            missing.append(str(csv_path))
            continue
        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        cases[case_dir.name] = summarize_rows(rows)
        all_rows.extend(rows)

    summary = {
        "schema": "acl2_v114tf_hs_mrt_trace_summary_v1",
        "results_root": str(root),
        "case_glob": args.case_glob,
        "case_count": len(cases),
        "missing_csv": missing,
        "aggregate": summarize_rows(all_rows),
        "cases": cases,
        "note": "predicted_metric_scale_delta is present only if runtime trace fills it; missing counts are reported explicitly.",
    }
    out_path = diagnostics / f"{args.output_prefix}_mrt_trace_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out_path), "case_count": len(cases), "row_count": len(all_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
