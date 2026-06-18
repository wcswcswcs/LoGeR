#!/usr/bin/env python3
"""Posthoc Phase3/4 selector audit for ACL2 v66B.

This is diagnostic-only.  It reads already generated Phase3/4 CSVs and asks:

* Do any fixed semantic strategies pass the documented high-risk + random
  control gates?
* If an oracle could pick the best semantic strategy per chunk using the
  measured proxy, is there enough headroom to justify another implementation
  attempt?

No GT-derived or proxy-selected oracle row is deployable evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_REPORT = Path("results/kitti01_hmc_v2/acl2_v66b_dense_semantic_scale/report_final")

PHASE3_SEMANTIC = [
    "S2_SUPPRESS_DYNAMIC",
    "S3_SUPPRESS_SKY",
    "S4_SUPPRESS_VEGETATION",
    "S5_SUPPRESS_DYNAMIC_SKY",
    "S6_SUPPRESS_DYNAMIC_SKY_VEGETATION",
    "S7_STATIC_ANCHOR_ONLY",
    "S8_VERTICAL_STATIC_ONLY",
    "S9_ROAD_GROUND_ONLY",
    "S10_VERTICAL_PLUS_ROAD_BOUNDARY",
    "S11_SEMANTIC_GEOMETRY_WEIGHTED",
]

PHASE4_SEMANTIC = [
    "S5_SUPPRESS_DYNAMIC_SKY",
    "S6_SUPPRESS_DYNAMIC_SKY_VEGETATION",
    "S7_STATIC_ANCHOR_ONLY",
    "S8_VERTICAL_STATIC_ONLY",
    "S10_VERTICAL_PLUS_ROAD_BOUNDARY",
    "S11_SEMANTIC_GEOMETRY_WEIGHTED",
]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _f(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except Exception:
        return math.nan


def _i(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _median(values: Iterable[Any]) -> Optional[float]:
    vals = [_f(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return None
    return float(median(vals))


def _frac(num: int, den: int) -> Optional[float]:
    if den <= 0:
        return None
    return float(num) / float(den)


def _suffix_control(strategy: str, suffix: str) -> str:
    return f"{strategy}_{suffix}"


def _phase3_score(row: Mapping[str, Any]) -> float:
    return max(
        _f(row.get("head_to_tail_improvement_vs_S1")),
        _f(row.get("intra_scale_variance_improvement_vs_S1")),
    )


def _phase4_score(row: Mapping[str, Any]) -> float:
    future = _f(row.get("future_improvement_vs_S1"))
    overlap = _f(row.get("overlap_residual_change_vs_S1"))
    if not math.isfinite(future):
        return math.nan
    if math.isfinite(overlap) and overlap > 0.20:
        return math.nan
    return future


def _rows_by_chunk_strategy(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[int, str], Mapping[str, Any]]:
    out: Dict[Tuple[int, str], Mapping[str, Any]] = {}
    for row in rows:
        cid = _i(row.get("chunk_id"))
        if cid is None:
            continue
        out[(cid, str(row.get("strategy", "")))] = row
    return out


def _phase3_high_risk(rows: Sequence[Mapping[str, Any]]) -> List[int]:
    base = [r for r in rows if r.get("strategy") == "S1_GEOMETRY_ONLY"]
    h2t_med = _median(r.get("head_to_tail_transfer_error") for r in base)
    var_med = _median(r.get("intra_scale_variance") for r in base)
    high: List[int] = []
    for row in base:
        cid = _i(row.get("chunk_id"))
        if cid is None:
            continue
        h2t = _f(row.get("head_to_tail_transfer_error"))
        var = _f(row.get("intra_scale_variance"))
        if (
            h2t_med is not None
            and math.isfinite(h2t)
            and h2t >= h2t_med
        ) or (
            var_med is not None
            and math.isfinite(var)
            and var >= var_med
        ):
            high.append(cid)
    return sorted(set(high))


def _phase4_high_risk(rows: Sequence[Mapping[str, Any]]) -> List[int]:
    base = [r for r in rows if r.get("strategy") == "S1_GEOMETRY_ONLY"]
    future_med = _median(r.get("future_after_overlap_error") for r in base)
    high: List[int] = []
    for row in base:
        cid = _i(row.get("chunk_id"))
        if cid is None:
            continue
        future = _f(row.get("future_after_overlap_error"))
        if future_med is not None and math.isfinite(future) and future >= future_med:
            high.append(cid)
    return sorted(set(high))


def _control_scores(
    by_cs: Mapping[Tuple[int, str], Mapping[str, Any]],
    strategy: str,
    chunks: Sequence[int],
    scorer,
) -> Dict[str, Optional[float]]:
    vals: Dict[str, List[float]] = {"random": [], "shuffled": []}
    for suffix, key in [("RANDOM", "random"), ("SHUFFLED", "shuffled")]:
        cname = _suffix_control(strategy, suffix)
        for cid in chunks:
            row = by_cs.get((cid, cname))
            if row is None:
                continue
            score = scorer(row)
            if math.isfinite(score):
                vals[key].append(score)
    return {
        "random_median_score": float(median(vals["random"])) if vals["random"] else None,
        "shuffled_median_score": float(median(vals["shuffled"])) if vals["shuffled"] else None,
        "control_count": len(vals["random"]) + len(vals["shuffled"]),
    }


def _strategy_audit(
    rows: Sequence[Mapping[str, Any]],
    strategies: Sequence[str],
    high_chunks: Sequence[int],
    scorer,
    *,
    phase: str,
) -> List[Dict[str, Any]]:
    by_cs = _rows_by_chunk_strategy(rows)
    all_chunks = sorted({cid for cid, strategy in by_cs.keys() if strategy == "S1_GEOMETRY_ONLY"})
    out: List[Dict[str, Any]] = []
    for strategy in strategies:
        for scope, chunks in [("all", all_chunks), ("high_risk", list(high_chunks))]:
            scores: List[float] = []
            positive = 0
            valid = 0
            for cid in chunks:
                row = by_cs.get((cid, strategy))
                if row is None:
                    continue
                score = scorer(row)
                if not math.isfinite(score):
                    continue
                valid += 1
                scores.append(score)
                if score >= 0.10:
                    positive += 1
            controls = _control_scores(by_cs, strategy, chunks, scorer)
            control_meds = [
                x
                for x in [controls["random_median_score"], controls["shuffled_median_score"]]
                if x is not None and math.isfinite(float(x))
            ]
            control_best = max(control_meds) if control_meds else None
            med = float(median(scores)) if scores else None
            margin = (med - control_best) if med is not None and control_best is not None else None
            gate_pass = bool(
                scope == "high_risk"
                and valid > 0
                and _frac(positive, valid) is not None
                and _frac(positive, valid) >= 0.60
                and med is not None
                and med >= 0.10
                and margin is not None
                and margin >= 0.05
            )
            out.append(
                {
                    "phase": phase,
                    "scope": scope,
                    "strategy": strategy,
                    "chunk_count": len(chunks),
                    "valid_score_count": valid,
                    "positive_count_score_ge_10pct": positive,
                    "positive_fraction": _frac(positive, valid),
                    "median_score": med,
                    "random_median_score": controls["random_median_score"],
                    "shuffled_median_score": controls["shuffled_median_score"],
                    "control_count": controls["control_count"],
                    "median_margin_vs_best_control": margin,
                    "fixed_strategy_gate_pass": gate_pass,
                }
            )
    return out


def _oracle_rows(
    rows: Sequence[Mapping[str, Any]],
    strategies: Sequence[str],
    high_chunks: Sequence[int],
    scorer,
    *,
    phase: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_cs = _rows_by_chunk_strategy(rows)
    control_strategies = [
        str(r.get("strategy"))
        for r in rows
        if str(r.get("strategy", "")).endswith("_RANDOM") or str(r.get("strategy", "")).endswith("_SHUFFLED")
    ]
    control_strategies = sorted(set(control_strategies))
    out: List[Dict[str, Any]] = []
    for cid in high_chunks:
        best_sem = None
        best_ctrl = None
        for strategy in strategies:
            row = by_cs.get((cid, strategy))
            if row is None:
                continue
            score = scorer(row)
            if not math.isfinite(score):
                continue
            if best_sem is None or score > best_sem[1]:
                best_sem = (strategy, score)
        for strategy in control_strategies:
            row = by_cs.get((cid, strategy))
            if row is None:
                continue
            score = scorer(row)
            if not math.isfinite(score):
                continue
            if best_ctrl is None or score > best_ctrl[1]:
                best_ctrl = (strategy, score)
        out.append(
            {
                "phase": phase,
                "chunk_id": cid,
                "best_semantic_strategy": best_sem[0] if best_sem else None,
                "best_semantic_score": best_sem[1] if best_sem else None,
                "best_control_strategy": best_ctrl[0] if best_ctrl else None,
                "best_control_score": best_ctrl[1] if best_ctrl else None,
                "semantic_minus_control": (
                    best_sem[1] - best_ctrl[1] if best_sem is not None and best_ctrl is not None else None
                ),
                "oracle_positive_ge_10pct": bool(best_sem is not None and best_sem[1] >= 0.10),
                "oracle_beats_control_by_5pct": bool(
                    best_sem is not None and best_ctrl is not None and (best_sem[1] - best_ctrl[1]) >= 0.05
                ),
            }
        )
    scores = [_f(r.get("best_semantic_score")) for r in out]
    scores = [s for s in scores if math.isfinite(s)]
    margins = [_f(r.get("semantic_minus_control")) for r in out]
    margins = [m for m in margins if math.isfinite(m)]
    summary = {
        "phase": phase,
        "high_risk_chunk_count": len(high_chunks),
        "oracle_positive_count": sum(1 for r in out if r.get("oracle_positive_ge_10pct")),
        "oracle_positive_fraction": _frac(sum(1 for r in out if r.get("oracle_positive_ge_10pct")), len(out)),
        "oracle_beats_control_by_5pct_count": sum(1 for r in out if r.get("oracle_beats_control_by_5pct")),
        "oracle_beats_control_by_5pct_fraction": _frac(sum(1 for r in out if r.get("oracle_beats_control_by_5pct")), len(out)),
        "median_best_semantic_score": float(median(scores)) if scores else None,
        "median_semantic_minus_control": float(median(margins)) if margins else None,
        "diagnostic_only": True,
    }
    return out, summary


def _write_report(out_dir: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# v66B Phase3/4 Causal Selector Posthoc",
        "",
        "This is diagnostic-only. It uses measured proxy outcomes to audit whether",
        "semantic strategies have enough headroom to justify more online action.",
        "Oracle rows are not deployable and must not be counted as success.",
        "",
        "## Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Phase3 high-risk chunks: `{summary['phase3']['high_risk_chunk_count']}`",
        f"- Phase3 fixed gate pass strategies: `{summary['phase3']['fixed_gate_pass_strategies']}`",
        f"- Phase3 oracle median best semantic score: `{summary['phase3']['oracle']['median_best_semantic_score']}`",
        f"- Phase3 oracle median semantic-control margin: `{summary['phase3']['oracle']['median_semantic_minus_control']}`",
        f"- Phase4 high-risk chunks: `{summary['phase4']['high_risk_chunk_count']}`",
        f"- Phase4 fixed gate pass strategies: `{summary['phase4']['fixed_gate_pass_strategies']}`",
        f"- Phase4 oracle median best semantic score: `{summary['phase4']['oracle']['median_best_semantic_score']}`",
        f"- Phase4 oracle median semantic-control margin: `{summary['phase4']['oracle']['median_semantic_minus_control']}`",
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
        "",
        "Artifacts:",
        "",
        "- `phase3_fixed_strategy_audit.csv`",
        "- `phase3_oracle_by_chunk.csv`",
        "- `phase4_fixed_strategy_audit.csv`",
        "- `phase4_oracle_by_chunk.csv`",
        "- `phase34_causal_selector_summary.json`",
    ]
    (out_dir / "phase34_causal_selector_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report_final", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    report = args.report_final
    out_dir = args.out_dir or (report / "phase14_phase34_causal_selector")
    out_dir.mkdir(parents=True, exist_ok=True)

    p3_rows = _read_csv(report / "phase3_intrachunk_scale/strategy_metrics_by_chunk.csv")
    p4_rows = _read_csv(report / "phase4_overlap_merge_anchor/overlap_strategy_results.csv")

    p3_high = _phase3_high_risk(p3_rows)
    p4_high = _phase4_high_risk(p4_rows)
    p3_audit = _strategy_audit(p3_rows, PHASE3_SEMANTIC, p3_high, _phase3_score, phase="phase3")
    p4_audit = _strategy_audit(p4_rows, PHASE4_SEMANTIC, p4_high, _phase4_score, phase="phase4")
    p3_oracle_rows, p3_oracle = _oracle_rows(p3_rows, PHASE3_SEMANTIC, p3_high, _phase3_score, phase="phase3")
    p4_oracle_rows, p4_oracle = _oracle_rows(p4_rows, PHASE4_SEMANTIC, p4_high, _phase4_score, phase="phase4")

    p3_pass = [r["strategy"] for r in p3_audit if r["scope"] == "high_risk" and r["fixed_strategy_gate_pass"]]
    p4_pass = [r["strategy"] for r in p4_audit if r["scope"] == "high_risk" and r["fixed_strategy_gate_pass"]]
    status = "no_deployable_phase34_semantic_selector"
    if p3_pass or p4_pass:
        status = "fixed_strategy_gate_pass_requires_manual_review"
    interpretation = (
        "No fixed Phase3/4 semantic strategy passed the documented high-risk, 10% median, "
        "and random/shuffled margin gate. Diagnostic oracle rows may show per-chunk proxy "
        "headroom, but because the selection uses measured outcomes, they are not deployable "
        "success and should only guide future causal-mask design."
    )

    summary = {
        "status": status,
        "phase3": {
            "high_risk_chunks": p3_high,
            "high_risk_chunk_count": len(p3_high),
            "fixed_gate_pass_strategies": p3_pass,
            "oracle": p3_oracle,
        },
        "phase4": {
            "high_risk_chunks": p4_high,
            "high_risk_chunk_count": len(p4_high),
            "fixed_gate_pass_strategies": p4_pass,
            "oracle": p4_oracle,
        },
        "interpretation": interpretation,
        "diagnostic_only": True,
    }

    _write_csv(out_dir / "phase3_fixed_strategy_audit.csv", p3_audit)
    _write_csv(out_dir / "phase3_oracle_by_chunk.csv", p3_oracle_rows)
    _write_csv(out_dir / "phase4_fixed_strategy_audit.csv", p4_audit)
    _write_csv(out_dir / "phase4_oracle_by_chunk.csv", p4_oracle_rows)
    (out_dir / "phase34_causal_selector_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(out_dir, summary)
    print(json.dumps({"out_dir": str(out_dir), "status": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
