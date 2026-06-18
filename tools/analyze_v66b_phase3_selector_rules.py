#!/usr/bin/env python3
"""Audit whether Phase3 oracle-positive chunks are predictable pre-action.

This reads Phase3 semantic fitting artifacts and the Phase14 oracle-by-chunk
file.  It searches only tiny threshold rules over chunk-level semantic/geometry
composition features.  The target label is diagnostic oracle success:

    best semantic score >= 10% and semantic-control margin >= 5%.

The output is not a deployable method by itself; it only answers whether there
is a simple, auditable selector worth turning into an online action.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_REPORT = Path("results/kitti01_hmc_v2/acl2_v66b_dense_semantic_scale/report_final")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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
    if value in (None, ""):
        return math.nan
    try:
        return float(value)
    except Exception:
        return math.nan


def _safe_ratio(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or den == 0:
        return math.nan
    return num / den


def _median(values: Iterable[float]) -> Optional[float]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return None
    return float(median(vals))


def _metric(y_true: Sequence[bool], y_pred: Sequence[bool]) -> Dict[str, float]:
    tp = sum(1 for y, p in zip(y_true, y_pred) if y and p)
    fp = sum(1 for y, p in zip(y_true, y_pred) if (not y) and p)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y and (not p))
    tn = sum(1 for y, p in zip(y_true, y_pred) if (not y) and (not p))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def _rows_by_chunk_strategy(rows: Sequence[Mapping[str, str]]) -> Dict[Tuple[int, str], Mapping[str, str]]:
    out: Dict[Tuple[int, str], Mapping[str, str]] = {}
    for row in rows:
        try:
            cid = int(row["chunk_id"])
        except Exception:
            continue
        out[(cid, str(row.get("strategy", "")))] = row
    return out


def _feature_rows(report: Path) -> List[Dict[str, Any]]:
    phase3 = _read_csv(report / "phase3_intrachunk_scale/strategy_metrics_by_chunk.csv")
    oracle = _read_csv(report / "phase14_phase34_causal_selector/phase3_oracle_by_chunk.csv")
    by_cs = _rows_by_chunk_strategy(phase3)
    out: List[Dict[str, Any]] = []
    for row in oracle:
        cid = int(row["chunk_id"])
        s1 = by_cs.get((cid, "S1_GEOMETRY_ONLY"), {})
        s8 = by_cs.get((cid, "S8_VERTICAL_STATIC_ONLY"), {})
        s9 = by_cs.get((cid, "S9_ROAD_GROUND_ONLY"), {})
        s10 = by_cs.get((cid, "S10_VERTICAL_PLUS_ROAD_BOUNDARY"), {})
        base_mass = _f(s1.get("valid_weight_sum"))
        best_score = _f(row.get("best_semantic_score"))
        margin = _f(row.get("semantic_minus_control"))
        rec: Dict[str, Any] = {
            "chunk_id": cid,
            "target_oracle_success": bool(best_score >= 0.10 and margin >= 0.05),
            "best_semantic_strategy": row.get("best_semantic_strategy"),
            "best_semantic_score": best_score,
            "semantic_minus_control": margin,
            "base_mass": base_mass,
            "base_grid_coverage": _f(s1.get("grid_coverage_ratio")),
            "base_condition": _f(s1.get("sim3_condition_score")),
            "dynamic_mass_ratio": _safe_ratio(_f(s1.get("removed_dynamic_mass")), base_mass),
            "sky_mass_ratio": _safe_ratio(_f(s1.get("removed_sky_mass")), base_mass),
            "vegetation_mass_ratio": _safe_ratio(_f(s1.get("removed_vegetation_mass")), base_mass),
            "vertical_mass_ratio": _safe_ratio(_f(s1.get("kept_vertical_static_mass")), base_mass),
            "ground_mass_ratio": _safe_ratio(_f(s1.get("kept_ground_mass")), base_mass),
            "vertical_only_remaining_ratio": _f(s8.get("remaining_valid_ratio")),
            "ground_only_remaining_ratio": _f(s9.get("remaining_valid_ratio")),
            "vertical_plus_boundary_remaining_ratio": _f(s10.get("remaining_valid_ratio")),
            "vertical_grid_coverage": _f(s8.get("grid_coverage_ratio")),
            "ground_grid_coverage": _f(s9.get("grid_coverage_ratio")),
            "vertical_condition": _f(s8.get("sim3_condition_score")),
            "ground_condition": _f(s9.get("sim3_condition_score")),
            "vertical_plus_boundary_condition": _f(s10.get("sim3_condition_score")),
        }
        out.append(rec)
    return out


def _candidate_literals(rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]) -> List[Tuple[str, str, float]]:
    lits: List[Tuple[str, str, float]] = []
    for name in feature_names:
        vals = sorted({_f(r.get(name)) for r in rows if math.isfinite(_f(r.get(name)))})
        if len(vals) < 2:
            continue
        qs = [0.2, 0.4, 0.5, 0.6, 0.8]
        thresholds = sorted({vals[min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))] for q in qs})
        for thr in thresholds:
            lits.append((name, ">=", thr))
            lits.append((name, "<=", thr))
    return lits


def _eval_rule(row: Mapping[str, Any], rule: Sequence[Tuple[str, str, float]]) -> bool:
    for name, op, thr in rule:
        val = _f(row.get(name))
        if not math.isfinite(val):
            return False
        if op == ">=" and not (val >= thr):
            return False
        if op == "<=" and not (val <= thr):
            return False
    return True


def _rule_text(rule: Sequence[Tuple[str, str, float]]) -> str:
    return " AND ".join(f"{name} {op} {thr:.6g}" for name, op, thr in rule)


def _search_rules(rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]) -> List[Dict[str, Any]]:
    y = [bool(r["target_oracle_success"]) for r in rows]
    literals = _candidate_literals(rows, feature_names)
    rules: List[Tuple[Tuple[str, str, float], ...]] = [(lit,) for lit in literals]
    # Two-literal conjunctions are allowed but still deliberately tiny.
    rules.extend(tuple(pair) for pair in itertools.combinations(literals, 2) if pair[0][0] != pair[1][0])
    scored: List[Dict[str, Any]] = []
    for rule in rules:
        pred = [_eval_rule(r, rule) for r in rows]
        m = _metric(y, pred)
        support = sum(pred)
        if support == 0:
            continue
        scored.append({"rule": _rule_text(rule), "support": support, **m})
    scored.sort(key=lambda r: (r["f1"], r["precision"], r["recall"], -r["fp"]), reverse=True)
    return scored[:50]


def _loocv(rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    preds: List[Dict[str, Any]] = []
    for i, held in enumerate(rows):
        train = [r for j, r in enumerate(rows) if j != i]
        rules = _search_rules(train, feature_names)
        best = rules[0] if rules else {"rule": "NO_RULE"}
        pred = False if best["rule"] == "NO_RULE" else _eval_rule(held, _parse_rule(best["rule"]))
        preds.append(
            {
                "chunk_id": held["chunk_id"],
                "target_oracle_success": held["target_oracle_success"],
                "predicted_success": pred,
                "selected_rule": best["rule"],
                "best_semantic_strategy": held["best_semantic_strategy"],
                "best_semantic_score": held["best_semantic_score"],
                "semantic_minus_control": held["semantic_minus_control"],
            }
        )
    metrics = _metric([bool(r["target_oracle_success"]) for r in preds], [bool(r["predicted_success"]) for r in preds])
    return preds, metrics


def _parse_rule(text: str) -> Tuple[Tuple[str, str, float], ...]:
    if text == "NO_RULE":
        return tuple()
    parts = [p.strip() for p in text.split(" AND ")]
    out = []
    for part in parts:
        bits = part.split()
        out.append((bits[0], bits[1], float(bits[2])))
    return tuple(out)


def _write_report(out_dir: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# v66B Phase3 Selector Rule Audit",
        "",
        "This is a diagnostic check for whether Phase3 oracle-positive chunks",
        "can be predicted from pre-action semantic/geometry composition features.",
        "",
        f"- Status: `{summary['status']}`",
        f"- Target positives: `{summary['target_positive_count']}/{summary['chunk_count']}`",
        f"- Best in-sample rule: `{summary['best_in_sample_rule']}`",
        f"- Best in-sample F1: `{summary['best_in_sample_f1']}`",
        f"- LOOCV precision/recall/F1: `{summary['loocv_metrics']['precision']}` / `{summary['loocv_metrics']['recall']}` / `{summary['loocv_metrics']['f1']}`",
        "",
        "Interpretation:",
        "",
        summary["interpretation"],
    ]
    (out_dir / "phase3_selector_rule_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report_final", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()
    report = args.report_final
    out_dir = args.out_dir or (report / "phase14_phase34_causal_selector")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _feature_rows(report)
    feature_names = [
        k
        for k in rows[0].keys()
        if k
        not in {
            "chunk_id",
            "target_oracle_success",
            "best_semantic_strategy",
            "best_semantic_score",
            "semantic_minus_control",
        }
    ]
    rules = _search_rules(rows, feature_names)
    loocv_rows, loocv_metrics = _loocv(rows, feature_names)
    best = rules[0] if rules else {"rule": "NO_RULE", "f1": 0.0}
    status = "no_stable_preaction_selector"
    if loocv_metrics["precision"] >= 0.70 and loocv_metrics["recall"] >= 0.50 and loocv_metrics["f1"] >= 0.60:
        status = "candidate_preaction_selector_requires_online_validation"
    interpretation = (
        "The selector is considered usable only if leave-one-chunk-out precision >=0.70, "
        "recall >=0.50, and F1 >=0.60.  Otherwise the Phase3 oracle headroom is treated "
        "as outcome-selected and not stable enough to justify another online semantic action."
    )
    summary = {
        "status": status,
        "chunk_count": len(rows),
        "target_positive_count": sum(1 for r in rows if r["target_oracle_success"]),
        "feature_names": feature_names,
        "best_in_sample_rule": best["rule"],
        "best_in_sample_f1": best["f1"],
        "loocv_metrics": loocv_metrics,
        "interpretation": interpretation,
        "diagnostic_only": True,
    }

    _write_csv(out_dir / "phase3_selector_feature_rows.csv", rows)
    _write_csv(out_dir / "phase3_selector_rule_candidates.csv", rules)
    _write_csv(out_dir / "phase3_selector_loocv_predictions.csv", loocv_rows)
    (out_dir / "phase3_selector_rule_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(out_dir, summary)
    print(json.dumps({"out_dir": str(out_dir), "status": status, "loocv": loocv_metrics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
