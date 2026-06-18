#!/usr/bin/env python3
"""Summarize v66B Phase19 geometry diagnostics.

This script is intentionally read-only with respect to experiment outputs. It
parses the completed H35 no-op geometry trace and the Phase0-4 offline
diagnostic tables, then writes a compact audit summary.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


ROOT = Path("results/kitti01_hmc_v2/acl2_v66b_dense_semantic_scale")
REPORT_ROOT = ROOT / "report_final"
PHASE19_DIR = REPORT_ROOT / "phase19_full704_geometry_diagnostic"
TRACE_DIR = (
    Path("results/kitti01_hmc_v2/acl2_v66b_artifacts/H35_FULL/rollouts")
    / "V66B_P19_704_GEOMETRY_TRACE_H35_NOOP"
)
GEOMETRY_DIR = Path("results/kitti01_hmc_v2/acl2_v66b_artifacts/H35_FULL/per_chunk_geometry")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int(value: Any) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _by_strategy(rows: Iterable[Mapping[str, str]]) -> Dict[str, Mapping[str, str]]:
    return {str(row.get("strategy")): row for row in rows}


def _strategy_record(
    phase: str,
    strategy: str,
    row: Optional[Mapping[str, str]],
    *,
    primary_metric: str,
    baseline: Optional[Mapping[str, str]] = None,
    random_row: Optional[Mapping[str, str]] = None,
    shuffled_row: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    row = row or {}
    rec: Dict[str, Any] = {
        "phase": phase,
        "strategy": strategy,
        "chunk_count": _int(row.get("chunk_count")),
        "fit_success_count": _int(row.get("fit_success_count")),
        primary_metric: _float(row.get(primary_metric)),
        "positive_proxy_count": _int(row.get("positive_proxy_count")),
    }
    for key in (
        "median_intra_scale_variance",
        "median_intra_scale_variance_improvement_vs_S1",
        "median_head_to_tail_transfer_error",
        "median_overlap_residual",
        "median_future_after_overlap_error",
    ):
        if key in row:
            rec[key] = _float(row.get(key))
    if baseline:
        base_value = _float(baseline.get(primary_metric))
        cur_value = _float(row.get(primary_metric))
        rec[f"{primary_metric}_delta_vs_baseline"] = (
            cur_value - base_value if cur_value is not None and base_value is not None else None
        )
    if random_row:
        random_value = _float(random_row.get(primary_metric))
        cur_value = _float(row.get(primary_metric))
        rec[f"{primary_metric}_delta_vs_random"] = (
            cur_value - random_value if cur_value is not None and random_value is not None else None
        )
    if shuffled_row:
        shuffled_value = _float(shuffled_row.get(primary_metric))
        cur_value = _float(row.get(primary_metric))
        rec[f"{primary_metric}_delta_vs_shuffled"] = (
            cur_value - shuffled_value if cur_value is not None and shuffled_value is not None else None
        )
    return rec


def _count_lines(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    with path.open("rb") as f:
        return sum(1 for _ in f)


def _read_ate(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    out: Dict[str, Any] = {"exists": True, "raw_lines": lines}
    for line in lines:
        if line.startswith("Average:"):
            parts = line.split()
            if len(parts) >= 3:
                out["ate_rmse"] = _float(parts[1])
                out["rot_rmse"] = _float(parts[2])
    return out


def main() -> None:
    phase0_4 = _load_json(PHASE19_DIR / "v66b_summary.json")
    phase3_summary = _load_json(PHASE19_DIR / "phase3_intrachunk_scale/phase3_summary.json")
    phase4_summary = _load_json(PHASE19_DIR / "phase4_overlap_merge_anchor/phase4_summary.json")
    phase3_rows = _read_csv(PHASE19_DIR / "phase3_intrachunk_scale/strategy_summary.csv")
    phase4_rows = _read_csv(PHASE19_DIR / "phase4_overlap_merge_anchor/overlap_strategy_summary.csv")
    p3 = _by_strategy(phase3_rows)
    p4 = _by_strategy(phase4_rows)

    geometry_files = sorted(GEOMETRY_DIR.glob("chunk_*.pt"))
    trace = {
        "run_dir": str(TRACE_DIR),
        "geometry_dir": str(GEOMETRY_DIR),
        "geometry_file_count": len(geometry_files),
        "trajectory_rows": _count_lines(TRACE_DIR / "01.txt"),
        "hmc_state_rows": _count_lines(TRACE_DIR / "hmc_state_hash.jsonl"),
        "run_status_tail": (TRACE_DIR / "run_status.txt").read_text().splitlines()[-2:]
        if (TRACE_DIR / "run_status.txt").exists()
        else [],
        "ate": _read_ate(TRACE_DIR / "results_sim3/results_ate.txt"),
    }

    selected: List[Dict[str, Any]] = []
    selected.extend(
        [
            _strategy_record(
                "phase3_intrachunk_scale",
                "S8_VERTICAL_STATIC_ONLY",
                p3.get("S8_VERTICAL_STATIC_ONLY"),
                primary_metric="median_head_to_tail_improvement_vs_S1",
                baseline=p3.get("S1_GEOMETRY_ONLY"),
                random_row=p3.get("S8_VERTICAL_STATIC_ONLY_RANDOM"),
                shuffled_row=p3.get("S8_VERTICAL_STATIC_ONLY_SHUFFLED"),
            ),
            _strategy_record(
                "phase3_intrachunk_scale",
                "S8_VERTICAL_STATIC_ONLY_SHUFFLED",
                p3.get("S8_VERTICAL_STATIC_ONLY_SHUFFLED"),
                primary_metric="median_head_to_tail_improvement_vs_S1",
                baseline=p3.get("S1_GEOMETRY_ONLY"),
            ),
            _strategy_record(
                "phase3_intrachunk_scale",
                "S11_SEMANTIC_GEOMETRY_WEIGHTED",
                p3.get("S11_SEMANTIC_GEOMETRY_WEIGHTED"),
                primary_metric="median_head_to_tail_improvement_vs_S1",
                baseline=p3.get("S1_GEOMETRY_ONLY"),
            ),
            _strategy_record(
                "phase4_overlap_merge_anchor",
                "S8_VERTICAL_STATIC_ONLY",
                p4.get("S8_VERTICAL_STATIC_ONLY"),
                primary_metric="median_future_improvement_vs_S1",
                baseline=p4.get("S1_GEOMETRY_ONLY"),
                random_row=p4.get("S8_VERTICAL_STATIC_ONLY_RANDOM"),
                shuffled_row=p4.get("S8_VERTICAL_STATIC_ONLY_SHUFFLED"),
            ),
            _strategy_record(
                "phase4_overlap_merge_anchor",
                "S11_SEMANTIC_GEOMETRY_WEIGHTED",
                p4.get("S11_SEMANTIC_GEOMETRY_WEIGHTED"),
                primary_metric="median_future_improvement_vs_S1",
                baseline=p4.get("S1_GEOMETRY_ONLY"),
            ),
            _strategy_record(
                "phase4_overlap_merge_anchor",
                "S10_VERTICAL_PLUS_ROAD_BOUNDARY",
                p4.get("S10_VERTICAL_PLUS_ROAD_BOUNDARY"),
                primary_metric="median_future_improvement_vs_S1",
                baseline=p4.get("S1_GEOMETRY_ONLY"),
            ),
        ]
    )

    best_p3 = phase3_summary.get("best_by_median_head_to_tail_improvement") or {}
    best_p4 = phase4_summary.get("best_by_median_future_improvement") or {}
    p3_best_strategy = str(best_p3.get("strategy", ""))
    p4_best_future = _float(best_p4.get("median_future_improvement_vs_S1")) or 0.0
    p4_positive = _int(best_p4.get("positive_proxy_count")) or 0
    p4_chunks = _int(best_p4.get("chunk_count")) or 0

    no_go_reasons = []
    if p3_best_strategy.endswith("_SHUFFLED") or p3_best_strategy.endswith("_RANDOM"):
        no_go_reasons.append("phase3_best_is_control_not_semantic_specific")
    if p4_best_future < 0.01:
        no_go_reasons.append("phase4_best_future_improvement_below_1_percent")
    if p4_chunks and (p4_positive / p4_chunks) < 0.10:
        no_go_reasons.append("phase4_positive_proxy_count_below_10_percent")
    s11_p4 = p4.get("S11_SEMANTIC_GEOMETRY_WEIGHTED", {})
    if (_int(s11_p4.get("positive_proxy_count")) or 0) == 0:
        no_go_reasons.append("phase4_semantic_geometry_weighted_has_zero_positive_proxy_chunks")

    summary = {
        "status": "no_go_after_phase19_geometry_diagnostic",
        "trace": trace,
        "phase0": phase0_4.get("phase0", {}),
        "phase1": phase0_4.get("phase1", {}),
        "phase2": phase0_4.get("phase2", {}),
        "phase3": phase3_summary,
        "phase4": phase4_summary,
        "selected_strategy_comparisons": selected,
        "no_go_reasons": no_go_reasons,
        "interpretation": (
            "The missing per-chunk geometry blocker is repaired, but the 704F offline proxy does not show a "
            "solid semantic-specific scale or overlap-to-future repair. Phase3 is led by a shuffled control, "
            "and Phase4's best real semantic future proxy is tiny and sparse."
        ),
    }
    _write_json(PHASE19_DIR / "phase19_geometry_diagnostic_summary.json", summary)
    _write_csv(PHASE19_DIR / "phase19_selected_strategy_comparisons.csv", selected)

    top_summary_path = REPORT_ROOT / "v66b_summary.json"
    top = _load_json(top_summary_path)
    top["status"] = "no_go_after_phase19_geometry_diagnostic"
    top["phase19_geometry_diagnostic"] = {
        "summary_path": str(PHASE19_DIR / "phase19_geometry_diagnostic_summary.json"),
        "selected_strategy_comparisons_path": str(PHASE19_DIR / "phase19_selected_strategy_comparisons.csv"),
        "status": summary["status"],
        "no_go_reasons": no_go_reasons,
        "trace": trace,
        "phase3_best": best_p3,
        "phase4_best": best_p4,
    }
    _write_json(top_summary_path, top)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
