#!/usr/bin/env python3
"""Pairwise combo search for v100 source-target semantic relation cues.

This tests whether D3-C source-target relations only become useful after a
chunk/geometry or head/layer constraint is added.  Diagnostic-only.
"""

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
D4_DIR = ROOT / "trackD4_read_current_support_provider"
N2_DIR = ROOT / "trackN2_anchor_identity_graph"


def top_source_target_specs(metric_rows: list[dict[str, str]], limit: int) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    rows = sorted(
        metric_rows,
        key=lambda row: (
            f(row.get("balanced_accuracy")),
            f(row.get("bad_recall")),
            -f(row.get("good_FPR"), 1.0),
            f(row.get("abs_corr_L3")),
            f(row.get("pairwise_add_BA_over_source_marginal")),
        ),
        reverse=True,
    )
    for row in rows:
        field = str(row.get("field", ""))
        direction = str(row.get("direction", ""))
        if not field.startswith("src_tgt_") or direction not in {"higher_bad", "lower_bad"}:
            continue
        key = (field, direction)
        if key in seen:
            continue
        seen.add(key)
        specs.append(key)
        if len(specs) >= int(limit):
            break
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--l2-rows", type=Path, default=L2_DIR / "rows.csv")
    parser.add_argument("--head-layer-rows", type=Path, default=D4_DIR / "semantic_trace_head_layer_case_rows.csv")
    parser.add_argument("--anchor-head-rows", type=Path, default=N2_DIR / "semantic_trace_anchor_head_case_rows.csv")
    parser.add_argument("--source-target-rows", type=Path, default=D4_DIR / "semantic_trace_source_target_case_rows.csv")
    parser.add_argument("--source-target-metrics", type=Path, default=D4_DIR / "semantic_trace_source_target_pair_metrics.csv")
    parser.add_argument("--out-dir", type=Path, default=D4_DIR)
    parser.add_argument("--top-source-target-specs", type=int, default=20)
    args = parser.parse_args()

    l2_rows = {str(row.get("case_id", "")): row for row in read_rows(args.l2_rows)}
    head_rows = {str(row.get("case_id", "")): row for row in read_rows(args.head_layer_rows)}
    anchor_rows = {str(row.get("case_id", "")): row for row in read_rows(args.anchor_head_rows)}
    source_rows = {str(row.get("case_id", "")): row for row in read_rows(args.source_target_rows)}
    rows = []
    for case_id in sorted(set(l2_rows) & set(head_rows) & set(anchor_rows) & set(source_rows)):
        head_extra = {
            f"headlayer_{key}": value
            for key, value in head_rows[case_id].items()
            if key not in {"case_id", "seq", "case_label", "L3_handoff_transfer_penalty_proxy"}
        }
        anchor_extra = {
            f"anchorhead_{key}": value
            for key, value in anchor_rows[case_id].items()
            if key not in {"case_id", "seq", "case_label", "L3_handoff_transfer_penalty_proxy"}
        }
        source_extra = {
            f"sourcetarget_{key}": value
            for key, value in source_rows[case_id].items()
            if key not in {"case_id", "seq", "case_label", "L3_handoff_transfer_penalty_proxy"}
        }
        rows.append({**l2_rows[case_id], **head_extra, **anchor_extra, **source_extra})

    metric_rows = read_rows(args.source_target_metrics)
    source_specs = [
        (f"sourcetarget_{field}", direction)
        for field, direction in top_source_target_specs(metric_rows, args.top_source_target_specs)
    ]
    fixed_specs = [
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
        ("headlayer_stable_anchor_topk_hit_frac_top3_mean", "lower_bad"),
        ("headlayer_stable_anchor_topk_hit_frac_top3_mean", "higher_bad"),
        ("headlayer_head_low_fine_current_support_risk_top3_mean", "lower_bad"),
        ("headlayer_head_low_group_current_support_risk_top3_mean", "lower_bad"),
        ("headlayer_layer3_low_fine_risk_max", "lower_bad"),
        ("anchorhead_anchor_head_same_fine_frac_mean", "lower_bad"),
        ("anchorhead_anchor_head_same_group_frac_mean", "lower_bad"),
        ("anchorhead_anchor_head_low_fine_support_risk_top3_mean", "higher_bad"),
        ("anchorhead_anchor_head_low_group_support_risk_top3_mean", "higher_bad"),
        ("anchorhead_anchor_head_label22_low_fine_risk_top3_mean", "higher_bad"),
        ("anchorhead_anchor_head_label23_low_fine_risk_top3_mean", "higher_bad"),
        ("anchorhead_anchor_head_high_hit_low_fine_frac", "higher_bad"),
    ]
    specs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for field, direction in fixed_specs + source_specs:
        key = (field, direction)
        if key not in seen and thresholds(rows, field):
            specs.append(key)
            seen.add(key)

    candidates = []
    for idx_a, (field_a, dir_a) in enumerate(specs):
        thrs_a = thresholds(rows, field_a)
        for field_b, dir_b in specs[idx_a + 1:]:
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
        "source_target_spec_count": len(source_specs),
        "source_target_specs": [{"field": field, "direction": direction} for field, direction in source_specs],
        "candidate_count": len(candidates),
        "gate_like_count": len(gate_like),
        "best": candidates[0] if candidates else {},
        "best_gate_like": gate_like[0] if gate_like else {},
        "note": "Pairwise AND/OR diagnostic over L2, head/layer, anchor/head identity, and top source-target semantic relation case rows.",
    }
    out = args.out_dir
    write_rows(out / "semantic_trace_source_target_combo_search_top_metrics.csv", candidates[:500])
    write_json(out / "semantic_trace_source_target_combo_search_summary.json", clean(summary))
    report = [
        "# Semantic Trace Source-Target Combo Search",
        "",
        "Diagnostic-only pairwise AND/OR threshold search. It does not authorize runtime action.",
        "",
        f"- case_count: `{len(rows)}`",
        f"- spec_count: `{len(specs)}`",
        f"- source_target_spec_count: `{len(source_specs)}`",
        f"- candidate_count: `{len(candidates)}`",
        f"- gate_like_count: `{len(gate_like)}`",
        "",
        "## Source-Target Specs",
        "",
        "```json",
        json.dumps(clean(summary["source_target_specs"]), ensure_ascii=False, indent=2, sort_keys=True),
        "```",
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
    ]
    (out / "semantic_trace_source_target_combo_search_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(clean(summary), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
