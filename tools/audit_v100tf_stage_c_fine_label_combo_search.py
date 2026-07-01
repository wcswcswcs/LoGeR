#!/usr/bin/env python3
"""Pairwise selector search for v100 fine-label Stage-C current support.

This is a diagnostic-only threshold search over already-materialized case rows.
It does not run or authorize runtime memory-control actions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_v100tf_same_space_semantic_anchor_latent_state_multiroute_memory_control import (
    f,
    pearson,
    write_json,
    write_rows,
)


ROOT = Path("results/acl2_v100tf_same_space_semantic_anchor_latent_state_memory_control")
L2_DIR = ROOT / "trackL2_anchor_scale_observability"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def condition(row: dict[str, Any], field: str, direction: str, threshold: float) -> bool:
    value = f(row.get(field))
    if not math.isfinite(value):
        return False
    return value <= threshold if direction == "lower_bad" else value >= threshold


def evaluate(rows: list[dict[str, Any]], selected: list[bool]) -> dict[str, Any]:
    bad_idx = [i for i, row in enumerate(rows) if row.get("case_label") == "non_good"]
    good_idx = [i for i, row in enumerate(rows) if row.get("case_label") == "good"]
    tp = [rows[i] for i in bad_idx if selected[i]]
    fp = [rows[i] for i in good_idx if selected[i]]
    recall = len(tp) / len(bad_idx) if bad_idx else math.nan
    fpr = len(fp) / len(good_idx) if good_idx else math.nan
    ba = (recall + (1.0 - fpr)) / 2.0 if math.isfinite(recall) and math.isfinite(fpr) else math.nan
    seq_counts = Counter(str(row.get("seq", "")) for row in tp if row.get("seq"))
    max_frac = max(seq_counts.values()) / len(tp) if tp and seq_counts else math.nan
    corr = pearson([1.0 if value else 0.0 for value in selected], [row.get("L3_handoff_transfer_penalty_proxy") for row in rows])
    return {
        "balanced_accuracy": ba,
        "bad_recall": recall,
        "good_FPR": fpr,
        "selected_case_count": sum(1 for value in selected if value),
        "true_positive_cases": ";".join(str(row.get("case_id")) for row in tp),
        "false_positive_cases": ";".join(str(row.get("case_id")) for row in fp),
        "missed_positive_cases": ";".join(str(rows[i].get("case_id")) for i in bad_idx if not selected[i]),
        "selected_positive_sequence_coverage": len(seq_counts),
        "selected_positive_sequence_max_frac": max_frac,
        "selector_corr_L3": corr,
        "selector_abs_corr_L3": abs(corr) if math.isfinite(corr) else math.nan,
    }


def thresholds(rows: list[dict[str, Any]], field: str) -> list[float]:
    values = sorted({f(row.get(field)) for row in rows if math.isfinite(f(row.get(field)))})
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--l2-rows", type=Path, default=L2_DIR / "rows.csv")
    parser.add_argument("--stage-c-rows", type=Path, default=L2_DIR / "stage_c_fine_label_current_support_rows.csv")
    parser.add_argument("--out-dir", type=Path, default=L2_DIR)
    args = parser.parse_args()

    l2_rows = {str(row.get("case_id", "")): row for row in read_rows(args.l2_rows)}
    stage_rows = {str(row.get("case_id", "")): row for row in read_rows(args.stage_c_rows)}
    rows: list[dict[str, Any]] = []
    for case_id in sorted(set(l2_rows) & set(stage_rows)):
        row = {**l2_rows[case_id], **{f"stagec_{k}": v for k, v in stage_rows[case_id].items() if k not in {"case_id", "seq", "case_label", "L3_handoff_transfer_penalty_proxy"}}}
        rows.append(row)

    specs = [
        ("geometry_parallax_x_current_support", "lower_bad"),
        ("geometry_parallax_depth_x_current_support", "lower_bad"),
        ("geometry_depth_ratio_risk", "higher_bad"),
        ("geometry_depth_ratio_risk", "lower_bad"),
        ("current_support", "lower_bad"),
        ("low_observability_risk", "higher_bad"),
        ("low_geometry_parallax_risk", "higher_bad"),
        ("low_geometry_depth_spread_risk", "higher_bad"),
        ("same_space_inconsistency_risk_proxy", "higher_bad"),
        ("scale_observability_score", "lower_bad"),
        ("no_scale_evidence_proxy", "higher_bad"),
        ("stale_or_unstable_role_frac", "higher_bad"),
        ("R_same_mean", "higher_bad"),
        ("anchor_current_feature_residual_max", "higher_bad"),
        ("stagec_fine_semantic_parallax_current", "lower_bad"),
        ("stagec_fine_semantic_parallax_depth_current", "lower_bad"),
        ("stagec_fine_semantic_current_support", "lower_bad"),
        ("stagec_fine_low_semantic_current_support_risk", "higher_bad"),
        ("stagec_group_semantic_parallax_current", "lower_bad"),
        ("stagec_group_semantic_parallax_depth_current", "lower_bad"),
        ("stagec_group_semantic_current_support", "lower_bad"),
        ("stagec_group_low_semantic_current_support_risk", "higher_bad"),
    ]
    specs = [(field, direction) for field, direction in specs if thresholds(rows, field)]

    candidates: list[dict[str, Any]] = []
    for idx_a, (field_a, dir_a) in enumerate(specs):
        for field_b, dir_b in specs[idx_a + 1:]:
            thrs_a = thresholds(rows, field_a)
            thrs_b = thresholds(rows, field_b)
            for thr_a in thrs_a:
                cond_a = [condition(row, field_a, dir_a, thr_a) for row in rows]
                for thr_b in thrs_b:
                    cond_b = [condition(row, field_b, dir_b, thr_b) for row in rows]
                    for op in ("AND", "OR"):
                        if op == "AND":
                            selected = [a and b for a, b in zip(cond_a, cond_b)]
                        else:
                            selected = [a or b for a, b in zip(cond_a, cond_b)]
                        metric = evaluate(rows, selected)
                        metric.update({
                            "op": op,
                            "field_a": field_a,
                            "direction_a": dir_a,
                            "threshold_a": thr_a,
                            "field_b": field_b,
                            "direction_b": dir_b,
                            "threshold_b": thr_b,
                        })
                        metric["gate_like"] = bool(
                            f(metric.get("bad_recall")) >= 0.65
                            and f(metric.get("good_FPR"), 1.0) <= 0.25
                            and f(metric.get("selector_abs_corr_L3")) >= 0.50
                            and f(metric.get("selector_corr_L3")) > 0.0
                            and int(f(metric.get("selected_positive_sequence_coverage"), 0)) >= 4
                            and f(metric.get("selected_positive_sequence_max_frac"), 1.0) <= 0.60
                        )
                        candidates.append(metric)

    candidates.sort(
        key=lambda row: (
            bool(row.get("gate_like")),
            f(row.get("balanced_accuracy")),
            f(row.get("bad_recall")),
            -f(row.get("good_FPR"), 1.0),
            f(row.get("selector_abs_corr_L3")),
        ),
        reverse=True,
    )
    top_rows = candidates[:500]
    gate_like = [row for row in candidates if row.get("gate_like")]
    summary = {
        "case_count": len(rows),
        "spec_count": len(specs),
        "candidate_count": len(candidates),
        "gate_like_count": len(gate_like),
        "best": candidates[0] if candidates else {},
        "best_gate_like": gate_like[0] if gate_like else {},
        "note": "Pairwise AND/OR diagnostic over existing v100 L2 rows and fine-label Stage-C current-support rows; no runtime action or true rerun controls.",
    }
    out = args.out_dir
    write_rows(out / "stage_c_fine_label_combo_search_top_metrics.csv", top_rows)
    write_json(out / "stage_c_fine_label_combo_search_summary.json", summary)
    report = [
        "# Stage-C Fine-Label Combo Search",
        "",
        "Pairwise AND/OR threshold search over existing L2 case rows and repaired Stage-C fine-label current support rows.",
        "",
        f"- case_count: `{len(rows)}`",
        f"- spec_count: `{len(specs)}`",
        f"- candidate_count: `{len(candidates)}`",
        f"- gate_like_count: `{len(gate_like)}`",
        "",
        "## Best Overall",
        "",
        "```json",
        json.dumps(clean(candidates[0] if candidates else {}), ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Best Gate-Like",
        "",
        "```json",
        json.dumps(clean(gate_like[0] if gate_like else {}), ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "Gate-like is not full diagnostic gate pass because true anchor-id/semantic-label/query-head rerun controls are absent.",
    ]
    (out / "stage_c_fine_label_combo_search_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(clean(summary), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
