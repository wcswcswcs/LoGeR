#!/usr/bin/env python3
"""Retarget v67 point-pair oracle rows to alternative future/tail objectives.

The existing overlap-pair action oracle gate is intentionally conservative and
primarily counts a row when intra-scale variance improves enough, raw overlap is
supported, and safety/ATE guards pass. This diagnostic does not change that
gate. It asks a narrower audit question: if we retarget the already materialized
oracle rows to the plan's future/tail or ATE objectives, do we get independent
positive boundaries beyond the known chunk-8 collapse?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


TARGETS = {
    "intra_scale": {
        "column": "intra_scale_variance_mean_improvement_vs_baseline",
        "threshold": 0.10,
        "note": "Current mechanism-style target; positive means lower intra-scale variance.",
    },
    "future_after_overlap": {
        "column": "future_after_overlap_mean_improvement_vs_baseline",
        "threshold": 0.10,
        "note": "Plan B/O6 target; positive >=0.10 would be a 10% future/overlap improvement proxy.",
    },
    "head_to_tail": {
        "column": "head_to_tail_transfer_ratio_mean_improvement_vs_baseline",
        "threshold": 0.10,
        "note": "Plan B/O6 target; positive >=0.10 would be a 10% head-to-tail transfer improvement proxy.",
    },
    "global_ate": {
        "column": "ate_improvement_m",
        "threshold": 0.50,
        "note": "Positive is -delta_vs_baseline_global_ate; threshold follows the 0.5m method-signal scale.",
    },
}


def _parse_source(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--source must be LABEL=oracle_results.csv")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("source label is empty")
    return label, Path(path)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _int_chunk(row: Dict[str, Any], key: str) -> int:
    value = _float(row.get(key))
    return int(value) if math.isfinite(value) else -1


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
        writer.writerows(rows)


def _safe_row(row: Dict[str, Any]) -> bool:
    return (
        _bool(row.get("safe_correction_pass"))
        and _bool(row.get("ate_guard_pass"))
        and _bool(row.get("raw_support_pass"))
    )


def _base_item(label: str, csv_path: Path, row: Dict[str, Any]) -> Dict[str, Any]:
    delta_ate = _float(row.get("delta_vs_baseline_global_ate"))
    return {
        "source_label": label,
        "oracle_results_csv": str(csv_path),
        "candidate": row.get("candidate", ""),
        "prev_chunk": _int_chunk(row, "prev_chunk"),
        "curr_chunk": _int_chunk(row, "curr_chunk"),
        "scope": row.get("scope", ""),
        "damped_action_family": row.get("damped_action_family", ""),
        "fit_semantic_filter": row.get("fit_semantic_filter", "all"),
        "oracle_action_gate_pass": _bool(row.get("oracle_action_gate_pass")),
        "safe_correction_pass": _bool(row.get("safe_correction_pass")),
        "ate_guard_pass": _bool(row.get("ate_guard_pass")),
        "raw_support_pass": _bool(row.get("raw_support_pass")),
        "raw_overlap_improvement_ratio": _float(row.get("raw_overlap_improvement_ratio")),
        "intra_scale_variance_mean_improvement_vs_baseline": _float(
            row.get("intra_scale_variance_mean_improvement_vs_baseline")
        ),
        "future_after_overlap_mean_improvement_vs_baseline": _float(
            row.get("future_after_overlap_mean_improvement_vs_baseline")
        ),
        "head_to_tail_transfer_ratio_mean_improvement_vs_baseline": _float(
            row.get("head_to_tail_transfer_ratio_mean_improvement_vs_baseline")
        ),
        "delta_vs_baseline_global_ate": delta_ate,
        "ate_improvement_m": -delta_ate if math.isfinite(delta_ate) else float("nan"),
        "correction_rotation_deg": _float(row.get("correction_rotation_deg")),
        "correction_abs_log_scale_delta": _float(row.get("correction_abs_log_scale_delta")),
        "correction_overlap_displacement_m": _float(row.get("correction_overlap_displacement_m")),
        "valid_pair_count": _float(row.get("valid_pair_count")),
    }


def _load_sources(sources: Sequence[Tuple[str, Path]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for label, path in sources:
        for row in _read_csv(path):
            rows.append(_base_item(label, path, row))
    return rows


def _top_rows(rows: Iterable[Dict[str, Any]], column: str, limit: int) -> List[Dict[str, Any]]:
    finite = [row for row in rows if math.isfinite(_float(row.get(column)))]
    finite.sort(key=lambda row: _float(row.get(column)), reverse=True)
    return finite[: int(limit)]


def _evaluate_target(
    rows: Sequence[Dict[str, Any]],
    *,
    name: str,
    column: str,
    threshold: float,
    min_raw_overlap_improvement: float,
    min_distinct_positive_chunks: int,
    top_k: int,
) -> Dict[str, Any]:
    safe_rows = [row for row in rows if _safe_row(row)]
    pass_rows = [
        row for row in safe_rows
        if _float(row.get(column)) >= float(threshold)
        and _float(row.get("raw_overlap_improvement_ratio")) >= float(min_raw_overlap_improvement)
    ]
    positive_chunks = sorted({int(row["curr_chunk"]) for row in pass_rows if int(row["curr_chunk"]) >= 0})
    pass_rows_by_source: Dict[str, int] = {}
    pass_chunks_by_source: Dict[str, List[int]] = {}
    for row in pass_rows:
        source = str(row.get("source_label", ""))
        pass_rows_by_source[source] = pass_rows_by_source.get(source, 0) + 1
        pass_chunks_by_source.setdefault(source, [])
        chunk = int(row["curr_chunk"])
        if chunk >= 0 and chunk not in pass_chunks_by_source[source]:
            pass_chunks_by_source[source].append(chunk)
    pass_chunks_by_source = {key: sorted(value) for key, value in sorted(pass_chunks_by_source.items())}
    return {
        "target": name,
        "target_column": column,
        "target_threshold": float(threshold),
        "min_raw_overlap_improvement": float(min_raw_overlap_improvement),
        "rows": len(rows),
        "safe_rows": len(safe_rows),
        "pass_rows": len(pass_rows),
        "positive_chunks": positive_chunks,
        "distinct_positive_chunk_count": len(positive_chunks),
        "pass_rows_by_source": dict(sorted(pass_rows_by_source.items())),
        "pass_chunks_by_source": pass_chunks_by_source,
        "target_gate_pass": bool(len(positive_chunks) >= int(min_distinct_positive_chunks)),
        "top_safe_rows": _top_rows(safe_rows, column, top_k),
        "top_pass_rows": _top_rows(pass_rows, column, top_k),
        "pass_rows_detail": pass_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=_parse_source, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-raw-overlap-improvement", type=float, default=0.20)
    parser.add_argument("--min-distinct-positive-chunks", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    rows = _load_sources(args.source)
    if not rows:
        raise ValueError("no oracle rows loaded")

    target_summaries = []
    for name, spec in TARGETS.items():
        target_summaries.append(
            _evaluate_target(
                rows,
                name=name,
                column=str(spec["column"]),
                threshold=float(spec["threshold"]),
                min_raw_overlap_improvement=float(args.min_raw_overlap_improvement),
                min_distinct_positive_chunks=int(args.min_distinct_positive_chunks),
                top_k=int(args.top_k),
            )
        )

    existing_gate_rows = [row for row in rows if _bool(row.get("oracle_action_gate_pass"))]
    existing_gate_chunks = sorted({int(row["curr_chunk"]) for row in existing_gate_rows if int(row["curr_chunk"]) >= 0})
    mechanism_targets = {"intra_scale", "future_after_overlap", "head_to_tail"}
    summary = {
        "schema": "acl2_v67_oracle_target_retarget_summary_v1",
        "sources": [{"label": label, "oracle_results_csv": str(path)} for label, path in args.source],
        "rows": len(rows),
        "safe_rows": sum(1 for row in rows if _safe_row(row)),
        "existing_oracle_gate_rows": len(existing_gate_rows),
        "existing_oracle_gate_chunks": existing_gate_chunks,
        "existing_oracle_distinct_positive_chunk_count": len(existing_gate_chunks),
        "min_raw_overlap_improvement": float(args.min_raw_overlap_improvement),
        "min_distinct_positive_chunks": int(args.min_distinct_positive_chunks),
        "target_notes": {name: spec["note"] for name, spec in TARGETS.items()},
        "target_summaries": [
            {k: v for k, v in item.items() if k not in {"top_safe_rows", "top_pass_rows", "pass_rows_detail"}}
            for item in target_summaries
        ],
        "target_gate_pass_any": any(bool(item["target_gate_pass"]) for item in target_summaries),
        "mechanism_target_gate_pass_any": any(
            bool(item["target_gate_pass"]) for item in target_summaries if item["target"] in mechanism_targets
        ),
        "global_ate_target_gate_pass": any(
            bool(item["target_gate_pass"]) for item in target_summaries if item["target"] == "global_ate"
        ),
        "note": (
            "Diagnostic only. Retargeting existing oracle rows to future/tail/ATE objectives "
            "does not validate a deployable method unless independent chunks and controls pass."
        ),
    }

    detail_rows: List[Dict[str, Any]] = []
    pass_detail_rows: List[Dict[str, Any]] = []
    for item in target_summaries:
        for group_name in ("top_safe_rows", "top_pass_rows"):
            for rank, row in enumerate(item[group_name], start=1):
                detail = dict(row)
                detail["target"] = item["target"]
                detail["target_column"] = item["target_column"]
                detail["target_threshold"] = item["target_threshold"]
                detail["row_group"] = group_name
                detail["rank"] = rank
                detail_rows.append(detail)
        for row in item["pass_rows_detail"]:
            detail = dict(row)
            detail["target"] = item["target"]
            detail["target_column"] = item["target_column"]
            detail["target_threshold"] = item["target_threshold"]
            pass_detail_rows.append(detail)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "oracle_target_retarget_top_rows.csv", detail_rows)
    _write_csv(args.out_dir / "oracle_target_retarget_pass_rows.csv", pass_detail_rows)
    (args.out_dir / "oracle_target_retarget_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
