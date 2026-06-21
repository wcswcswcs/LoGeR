#!/usr/bin/env python3
"""Summarize v73 Phase 5 single-path candidates for Phase 9 eligibility.

This audit is intentionally conservative: Phase 9 integration is allowed only
when a full 11-chunk single-path summary has already passed its Phase E gate.
Canaries are reported for context but never promoted to integration evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_PHASE5_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v73_semantic_memory_control/"
    "report_final/phase5_mid_term"
)
DEFAULT_OUT_DIR = Path(
    "results/kitti01_hmc_v2/acl2_v73_semantic_memory_control/"
    "report_final/phase9_integration/best_single_path_audit"
)
SUMMARY_PATTERNS = (
    "phaseE_multichunk_summary_full11.json",
    "phaseE_multichunk_summary_geometry_only_candidate_full11.json",
    "phaseE_multichunk_summary_canary.json",
)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _safe_bool(value: Any) -> bool:
    return bool(value) if value is not None else False


def _discover(root: Path) -> List[Path]:
    paths: List[Path] = []
    for pattern in SUMMARY_PATTERNS:
        paths.extend(root.glob(f"**/{pattern}"))
    return sorted(set(paths))


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _row_from_summary(path: Path, data: Dict[str, Any], phase5_root: Path) -> Dict[str, Any]:
    chunks = data.get("chunks") or []
    missing = data.get("missing") or []
    chunk_count = len(chunks) if isinstance(chunks, list) else 0
    missing_count = len(missing) if isinstance(missing, list) else 0
    nominal_file_scope = "canary" if path.name.endswith("_canary.json") else "full11"
    evaluated_scope = "full11" if chunk_count == 11 else "canary"
    is_full11 = evaluated_scope == "full11"
    phase_e_gate_pass = _safe_bool(data.get("phaseE_gate_pass"))
    integration_eligible = bool(is_full11 and phase_e_gate_pass and missing_count == 0)
    head_pass_count = int(data.get("head_tail_pass_count") or 0)
    overlap_pass_count = int(data.get("overlap_pass_count") or 0)
    head_median = _safe_float(data.get("head_tail_median_improvement_vs_baseline_ratio"))
    overlap_median = _safe_float(data.get("overlap_median_improvement_vs_baseline_ratio"))
    best_pass_count = max(head_pass_count, overlap_pass_count)
    best_median = max(
        [v for v in (head_median, overlap_median) if v is not None],
        default=None,
    )
    if head_pass_count > overlap_pass_count:
        best_metric = "head_tail"
    elif overlap_pass_count > head_pass_count:
        best_metric = "overlap"
    elif (head_median or -999.0) >= (overlap_median or -999.0):
        best_metric = "head_tail"
    else:
        best_metric = "overlap"
    try:
        rel = str(path.relative_to(phase5_root))
    except ValueError:
        rel = str(path)
    return {
        "experiment": rel.rsplit("/", 1)[0],
        "summary_path": str(path),
        "summary_file": path.name,
        "candidate": data.get("candidate"),
        "baseline": data.get("baseline"),
        "controls": _json_text(data.get("controls") or []),
        "chunks": _json_text(chunks),
        "chunk_count": chunk_count,
        "missing_count": missing_count,
        "nominal_file_scope": nominal_file_scope,
        "evaluated_scope": evaluated_scope,
        "is_full11": is_full11,
        "phaseE_gate_pass": phase_e_gate_pass,
        "phaseE_head_tail_pass": _safe_bool(data.get("phaseE_head_tail_pass")),
        "phaseE_overlap_pass": _safe_bool(data.get("phaseE_overlap_pass")),
        "head_tail_pass_count": head_pass_count,
        "overlap_pass_count": overlap_pass_count,
        "head_tail_median_improvement_vs_baseline_ratio": head_median,
        "overlap_median_improvement_vs_baseline_ratio": overlap_median,
        "best_metric": best_metric,
        "best_pass_count": best_pass_count,
        "best_median_improvement_vs_baseline_ratio": best_median,
        "integration_eligible": integration_eligible,
    }


def _load_rows(paths: Iterable[Path], phase5_root: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(_row_from_summary(path, data, phase5_root))
    return rows


def _sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(row: Dict[str, Any]) -> Any:
        best_median = row.get("best_median_improvement_vs_baseline_ratio")
        return (
            bool(row.get("integration_eligible")),
            bool(row.get("is_full11")),
            int(row.get("best_pass_count") or 0),
            float(best_median) if best_median is not None else -999.0,
            int(row.get("head_tail_pass_count") or 0),
            int(row.get("overlap_pass_count") or 0),
            str(row.get("experiment") or ""),
        )

    return sorted(rows, key=key, reverse=True)


def _best(rows: List[Dict[str, Any]], metric_prefix: str) -> Optional[Dict[str, Any]]:
    candidates = [row for row in rows if row.get("is_full11")]
    if not candidates:
        return None
    pass_key = f"{metric_prefix}_pass_count"
    median_key = f"{metric_prefix}_median_improvement_vs_baseline_ratio"
    return max(
        candidates,
        key=lambda row: (
            int(row.get(pass_key) or 0),
            float(row.get(median_key)) if row.get(median_key) is not None else -999.0,
            str(row.get("experiment") or ""),
        ),
    )


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
            writer.writerow(row)


def _compact_row(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    keys = [
        "experiment",
        "summary_file",
        "candidate",
        "baseline",
        "chunk_count",
        "missing_count",
        "phaseE_gate_pass",
        "head_tail_pass_count",
        "overlap_pass_count",
        "head_tail_median_improvement_vs_baseline_ratio",
        "overlap_median_improvement_vs_baseline_ratio",
        "integration_eligible",
    ]
    return {key: row.get(key) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase5-root", type=Path, default=DEFAULT_PHASE5_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    paths = _discover(args.phase5_root)
    rows = _sort_rows(_load_rows(paths, args.phase5_root))
    full11_rows = [row for row in rows if row.get("is_full11")]
    canary_rows = [row for row in rows if not row.get("is_full11")]
    eligible_rows = [row for row in rows if row.get("integration_eligible")]
    best_head = _best(rows, "head_tail")
    best_overlap = _best(rows, "overlap")
    best_any = full11_rows[0] if full11_rows else None

    summary = {
        "phase5_root": str(args.phase5_root),
        "out_dir": str(args.out_dir),
        "summary_count": len(rows),
        "full11_count": len(full11_rows),
        "canary_count": len(canary_rows),
        "eligible_full11_count": len(eligible_rows),
        "integration_allowed": bool(eligible_rows),
        "phase9_blocked_reason": (
            None
            if eligible_rows
            else "No full11 single-path candidate passed Phase E; plan requires a passed single-path gate before Phase 9 integration."
        ),
        "best_full11_overall": _compact_row(best_any),
        "best_full11_head_tail": _compact_row(best_head),
        "best_full11_overlap": _compact_row(best_overlap),
        "eligible_rows": [_compact_row(row) for row in eligible_rows],
        "rows": [_compact_row(row) for row in rows],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "best_single_path_rows.csv"
    out_json = args.out_dir / "best_single_path_summary.json"
    _write_csv(out_csv, rows)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    printable = dict(summary)
    printable["rows"] = printable["rows"][:8]
    print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote_csv={out_csv}")
    print(f"wrote_json={out_json}")


if __name__ == "__main__":
    main()
