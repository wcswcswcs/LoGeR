#!/usr/bin/env python3
"""Pairwise combo search for v100 anchor/head identity-edge diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.audit_v100tf_stage_c_fine_label_combo_search import clean, condition, evaluate, read_rows, thresholds
from tools.build_v100tf_same_space_semantic_anchor_latent_state_multiroute_memory_control import f, write_json, write_rows


ROOT = Path("results/acl2_v100tf_same_space_semantic_anchor_latent_state_memory_control")
L2_DIR = ROOT / "trackL2_anchor_scale_observability"
N2_DIR = ROOT / "trackN2_anchor_identity_graph"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--l2-rows", type=Path, default=L2_DIR / "rows.csv")
    parser.add_argument("--anchor-head-rows", type=Path, default=N2_DIR / "semantic_trace_anchor_head_case_rows.csv")
    parser.add_argument("--out-dir", type=Path, default=N2_DIR)
    args = parser.parse_args()

    l2_rows = {str(row.get("case_id", "")): row for row in read_rows(args.l2_rows)}
    ah_rows = {str(row.get("case_id", "")): row for row in read_rows(args.anchor_head_rows)}
    rows = []
    for case_id in sorted(set(l2_rows) & set(ah_rows)):
        extra = {
            f"ah_{key}": value
            for key, value in ah_rows[case_id].items()
            if key not in {"case_id", "seq", "case_label", "L3_handoff_transfer_penalty_proxy"}
        }
        rows.append({**l2_rows[case_id], **extra})

    specs = [
        ("geometry_parallax_x_current_support", "lower_bad"),
        ("geometry_parallax_depth_x_current_support", "lower_bad"),
        ("geometry_depth_ratio_risk", "higher_bad"),
        ("geometry_depth_ratio_risk", "lower_bad"),
        ("current_support", "lower_bad"),
        ("low_observability_risk", "higher_bad"),
        ("same_space_inconsistency_risk_proxy", "higher_bad"),
        ("scale_observability_score", "lower_bad"),
        ("no_scale_evidence_proxy", "higher_bad"),
        ("R_same_mean", "higher_bad"),
        ("ah_anchor_head_same_fine_frac_mean", "lower_bad"),
        ("ah_anchor_head_same_fine_frac_min", "lower_bad"),
        ("ah_anchor_head_low_fine_support_risk_max", "higher_bad"),
        ("ah_anchor_head_low_fine_support_risk_top3_mean", "higher_bad"),
        ("ah_anchor_head_low_fine_support_risk_top5_mean", "higher_bad"),
        ("ah_anchor_head_top_anchor_hit_frac", "higher_bad"),
        ("ah_anchor_head_hit_count_top3_mean", "higher_bad"),
        ("ah_anchor_head_high_hit_low_fine_frac", "higher_bad"),
        ("ah_anchor_head_label1_low_fine_risk_top3_mean", "higher_bad"),
        ("ah_anchor_head_label23_low_fine_risk_top3_mean", "higher_bad"),
        ("ah_anchor_head_label15_low_fine_risk_top3_mean", "higher_bad"),
        ("ah_anchor_head_label22_low_fine_risk_top3_mean", "higher_bad"),
    ]
    specs = [(field, direction) for field, direction in specs if thresholds(rows, field)]

    candidates = []
    for idx_a, (field_a, dir_a) in enumerate(specs):
        for field_b, dir_b in specs[idx_a + 1:]:
            thrs_a = thresholds(rows, field_a)
            thrs_b = thresholds(rows, field_b)
            for thr_a in thrs_a:
                cond_a = [condition(row, field_a, dir_a, thr_a) for row in rows]
                for thr_b in thrs_b:
                    cond_b = [condition(row, field_b, dir_b, thr_b) for row in rows]
                    for op in ("AND", "OR"):
                        selected = [a and b for a, b in zip(cond_a, cond_b)] if op == "AND" else [a or b for a, b in zip(cond_a, cond_b)]
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
    gate_like = [row for row in candidates if row.get("gate_like")]
    summary = {
        "case_count": len(rows),
        "spec_count": len(specs),
        "candidate_count": len(candidates),
        "gate_like_count": len(gate_like),
        "best": candidates[0] if candidates else {},
        "best_gate_like": gate_like[0] if gate_like else {},
        "note": "Pairwise AND/OR diagnostic over L2 case rows plus anchor/head semantic identity-edge rows.",
    }
    out = args.out_dir
    write_rows(out / "semantic_trace_anchor_head_combo_search_top_metrics.csv", candidates[:500])
    write_json(out / "semantic_trace_anchor_head_combo_search_summary.json", summary)
    report = [
        "# Semantic Trace Anchor/Head Combo Search",
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
        "No runtime action is allowed from this diagnostic-only search.",
    ]
    (out / "semantic_trace_anchor_head_combo_search_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(clean(summary), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
