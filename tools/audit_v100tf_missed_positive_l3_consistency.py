#!/usr/bin/env python3
"""Missed-positive and L3-gate consistency audit for ACL2 v100.

Several v100 diagnostics classify non_good/good cases well but fail the
selector-L3 correlation gate.  This audit checks whether that is a candidate
bug in the selector, or a deeper inconsistency between the binary case labels
and the current L3_handoff_transfer_penalty_proxy used by the gate.

Diagnostic-only; no runtime action is authorized.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.audit_v100tf_stage_c_fine_label_combo_search import condition, evaluate, read_rows
from tools.build_v100tf_same_space_semantic_anchor_latent_state_multiroute_memory_control import (
    f,
    pearson,
    quantile,
    write_json,
    write_rows,
)


ROOT = Path("results/acl2_v100tf_same_space_semantic_anchor_latent_state_memory_control")
OUT_DIR = ROOT / "trackD4_read_current_support_provider"


def clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def merge_case_rows() -> list[dict[str, Any]]:
    roots = {
        "l2": ROOT / "trackL2_anchor_scale_observability/rows.csv",
        "R": ROOT / "trackR_edge_head_control_audit/case_rows.csv",
        "headlayer": ROOT / "trackD4_read_current_support_provider/semantic_trace_head_layer_case_rows.csv",
        "anchorhead": ROOT / "trackN2_anchor_identity_graph/semantic_trace_anchor_head_case_rows.csv",
        "sourcetarget": ROOT / "trackD4_read_current_support_provider/semantic_trace_source_target_case_rows.csv",
        "semcontrol": ROOT / "trackL2_anchor_scale_observability/anchor_head_semantic_control_case_rows.csv",
        "stagec": ROOT / "trackL2_anchor_scale_observability/stage_c_fine_label_current_support_rows.csv",
        "Q": ROOT / "trackQ_chunk_update_admission/rows.csv",
    }
    tables: dict[str, dict[str, dict[str, str]]] = {}
    for name, path in roots.items():
        if path.is_file():
            tables[name] = {str(row.get("case_id", "")): row for row in read_rows(path)}
    case_ids = set(tables.get("l2", {}))
    for table in tables.values():
        case_ids &= set(table)
    rows: list[dict[str, Any]] = []
    skip = {"case_id", "seq", "case_label", "L3_handoff_transfer_penalty_proxy"}
    for case_id in sorted(case_ids):
        base = dict(tables["l2"][case_id])
        for name, table in tables.items():
            if name == "l2":
                continue
            for key, value in table[case_id].items():
                if key not in skip:
                    base[f"{name}_{key}"] = value
        rows.append(base)
    return rows


def seq_of(row: dict[str, Any]) -> str:
    return str(row.get("seq", ""))


def rule_selected(rows: list[dict[str, Any]], rule: dict[str, Any]) -> list[bool]:
    if "children" in rule:
        child_masks = [rule_selected(rows, child) for child in rule["children"]]
        if str(rule.get("op")) == "AND":
            return [all(mask[i] for mask in child_masks) for i in range(len(rows))]
        return [any(mask[i] for mask in child_masks) for i in range(len(rows))]
    return [condition(row, str(rule["field"]), str(rule["direction"]), float(rule["threshold"])) for row in rows]


def metric_for_mask(rows: list[dict[str, Any]], name: str, mask: list[bool]) -> dict[str, Any]:
    metric = evaluate(rows, mask)
    metric["cue_name"] = name
    metric["gate_like"] = bool(
        f(metric.get("bad_recall")) >= 0.65
        and f(metric.get("good_FPR"), 1.0) <= 0.25
        and f(metric.get("selector_abs_corr_L3")) >= 0.50
        and f(metric.get("selector_corr_L3")) > 0.0
        and int(f(metric.get("selected_positive_sequence_coverage"), 0)) >= 4
        and f(metric.get("selected_positive_sequence_max_frac"), 1.0) <= 0.60
    )
    bad_rows = [row for row in rows if row.get("case_label") == "non_good"]
    selected_bad = [row for row, selected in zip(rows, mask) if selected and row.get("case_label") == "non_good"]
    seq_counts = Counter(seq_of(row) for row in selected_bad)
    metric["selected_positive_sequence_counts"] = dict(sorted(seq_counts.items()))
    metric["missed_positive_case_count"] = len([row for row in bad_rows if row.get("case_id") not in set(str(r.get("case_id")) for r in selected_bad)])
    return metric


def rule_name(rule: dict[str, Any]) -> str:
    if "children" in rule:
        return f"{rule.get('op')}(" + ",".join(rule_name(child) for child in rule["children"]) + ")"
    return f"{rule['field']} {rule['direction']} {rule['threshold']}"


def build_rule_pool(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manual_rules = [
        {
            "name": "combo_headlayer_low_stable_x_structure_to_movable_low",
            "op": "AND",
            "children": [
                {"field": "headlayer_stable_anchor_topk_hit_frac_top3_mean", "direction": "lower_bad", "threshold": 0.5299479166666666},
                {
                    "field": "sourcetarget_src_tgt_alltopk_q0_structure_anchor_s2_movable_thing_frac_top3_mean",
                    "direction": "lower_bad",
                    "threshold": 0.15364583333333334,
                },
            ],
        },
        {"name": "R_head_top1_stable_hit_max", "field": "R_head_top1_stable_hit_max", "direction": "higher_bad", "threshold": 0.8125},
        {"name": "R_head_stable_query_hit_max", "field": "R_head_stable_query_hit_max", "direction": "higher_bad", "threshold": 0.984375},
        {"name": "headlayer_stable_anchor_topk_hit_frac_top3_mean", "field": "headlayer_stable_anchor_topk_hit_frac_top3_mean", "direction": "lower_bad", "threshold": 0.5299479166666666},
        {"name": "anchorhead_same_fine_frac_mean", "field": "anchorhead_anchor_head_same_fine_frac_mean", "direction": "lower_bad", "threshold": 0.3300505949556347},
        {"name": "sourcetarget_lowstuff_uncertain", "field": "sourcetarget_src_tgt_alltopk_q3_low_value_stuff_s4_uncertain_region_frac_mean", "direction": "higher_bad", "threshold": 0.0146942138671875},
        {"name": "sourcetarget_source_marginal_nearframe", "field": "sourcetarget_src_marginal_stablehit_nearframe_frac_mean", "direction": "higher_bad", "threshold": 0.08895464468794649},
        {"name": "sem22_low_parallax_current", "field": "semcontrol_sem_22_low_parallax_current_top3_mean", "direction": "higher_bad", "threshold": 0.6717082815125877},
        {"name": "stagec_group_semantic_parallax_depth_current", "field": "stagec_group_semantic_parallax_depth_current", "direction": "lower_bad", "threshold": 0.18058157115799645},
        {"name": "Q_stale_or_fresh_proxy", "op": "OR", "children": [
            {"field": "Q_stale_anchor_score_proxy", "direction": "higher_bad", "threshold": 0.06769266373524647},
            {"field": "Q_fresh_supported_score_proxy", "direction": "lower_bad", "threshold": 0.34875828286576405},
        ]},
    ]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rule in manual_rules:
        try:
            mask = rule_selected(rows, rule)
        except Exception:
            continue
        if not any(mask):
            continue
        name = str(rule.get("name", rule_name(rule)))
        if name not in seen:
            out.append({**rule, "name": name})
            seen.add(name)
    return out


def oracle_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y = [f(row.get("L3_handoff_transfer_penalty_proxy")) for row in rows]
    bad_mask = [1.0 if row.get("case_label") == "non_good" else 0.0 for row in rows]
    good_mask = [1.0 if row.get("case_label") == "good" else 0.0 for row in rows]
    bad_l3 = [f(row.get("L3_handoff_transfer_penalty_proxy")) for row in rows if row.get("case_label") == "non_good"]
    good_l3 = [f(row.get("L3_handoff_transfer_penalty_proxy")) for row in rows if row.get("case_label") == "good"]
    bad_median = quantile(bad_l3, 0.50)
    good_median = quantile(good_l3, 0.50)
    table = []
    for row in sorted(rows, key=lambda item: f(item.get("L3_handoff_transfer_penalty_proxy")), reverse=True):
        l3 = f(row.get("L3_handoff_transfer_penalty_proxy"))
        label = str(row.get("case_label", ""))
        conflict = ""
        if label == "good" and math.isfinite(l3) and math.isfinite(bad_median) and l3 >= bad_median:
            conflict = "good_high_l3"
        elif label == "non_good" and math.isfinite(l3) and math.isfinite(good_median) and l3 <= good_median:
            conflict = "bad_low_l3"
        table.append({
            "case_id": row.get("case_id", ""),
            "seq": row.get("seq", ""),
            "case_label": label,
            "failure_type": row.get("failure_type", ""),
            "L3_handoff_transfer_penalty_proxy": l3,
            "label_l3_conflict": conflict,
        })
    summary = {
        "case_count": len(rows),
        "non_good_count": int(sum(bad_mask)),
        "good_count": int(sum(good_mask)),
        "bad_L3_median": bad_median,
        "good_L3_median": good_median,
        "oracle_bad_selector_corr_L3": pearson(bad_mask, y),
        "oracle_bad_selector_abs_corr_L3": abs(pearson(bad_mask, y)),
        "oracle_good_selector_corr_L3": pearson(good_mask, y),
        "oracle_good_selector_abs_corr_L3": abs(pearson(good_mask, y)),
        "good_high_l3_cases": ";".join(row["case_id"] for row in table if row["label_l3_conflict"] == "good_high_l3"),
        "bad_low_l3_cases": ";".join(row["case_id"] for row in table if row["label_l3_conflict"] == "bad_low_l3"),
    }
    return summary, table


def composite_search(rows: list[dict[str, Any]], rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    masks = {str(rule["name"]): rule_selected(rows, rule) for rule in rules}
    candidates: list[dict[str, Any]] = []
    for size in (1, 2, 3):
        for combo in itertools.combinations([str(rule["name"]) for rule in rules], size):
            mask = [any(masks[name][i] for name in combo) for i in range(len(rows))]
            metric = metric_for_mask(rows, "OR(" + ",".join(combo) + ")", mask)
            metric["rule_count"] = size
            metric["rules"] = ";".join(combo)
            candidates.append(metric)
    candidates.sort(
        key=lambda row: (
            bool(row.get("gate_like")),
            f(row.get("bad_recall")),
            -f(row.get("good_FPR"), 1.0),
            f(row.get("selector_abs_corr_L3")),
            f(row.get("balanced_accuracy")),
        ),
        reverse=True,
    )
    return candidates


def median(values: list[float]) -> float:
    return quantile(values, 0.50)


def failure_type_rows(
    rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    selected_mask: list[bool],
) -> list[dict[str, Any]]:
    conflict_by_case = {str(row.get("case_id", "")): str(row.get("label_l3_conflict", "")) for row in label_rows}
    groups: dict[tuple[str, str], list[tuple[dict[str, Any], bool]]] = {}
    for row, selected in zip(rows, selected_mask):
        key = (str(row.get("failure_type", "")), str(row.get("case_label", "")))
        groups.setdefault(key, []).append((row, selected))
    out: list[dict[str, Any]] = []
    for (failure_type, label), parts in sorted(groups.items()):
        l3_vals = [f(row.get("L3_handoff_transfer_penalty_proxy")) for row, _ in parts]
        conflicts = [
            str(row.get("case_id", ""))
            for row, _ in parts
            if conflict_by_case.get(str(row.get("case_id", "")))
        ]
        out.append({
            "failure_type": failure_type,
            "case_label": label,
            "case_count": len(parts),
            "selected_by_best_count": sum(1 for _, selected in parts if selected),
            "L3_mean": sum(v for v in l3_vals if math.isfinite(v)) / max(len([v for v in l3_vals if math.isfinite(v)]), 1),
            "L3_median": median(l3_vals),
            "L3_min": min([v for v in l3_vals if math.isfinite(v)], default=math.nan),
            "L3_max": max([v for v in l3_vals if math.isfinite(v)], default=math.nan),
            "case_ids": ";".join(str(row.get("case_id", "")) for row, _ in parts),
            "label_l3_conflict_cases": ";".join(conflicts),
        })
    return out


def subset_metrics(rows: list[dict[str, Any]], label_rows: list[dict[str, Any]], selected_mask: list[bool]) -> list[dict[str, Any]]:
    conflict_by_case = {str(row.get("case_id", "")): str(row.get("label_l3_conflict", "")) for row in label_rows}
    subsets = {
        "all": lambda row: True,
        "without_good_high_l3": lambda row: conflict_by_case.get(str(row.get("case_id", ""))) != "good_high_l3",
        "without_bad_low_l3": lambda row: conflict_by_case.get(str(row.get("case_id", ""))) != "bad_low_l3",
        "without_label_l3_conflicts": lambda row: not conflict_by_case.get(str(row.get("case_id", ""))),
        "only_label_l3_conflicts": lambda row: bool(conflict_by_case.get(str(row.get("case_id", "")))),
    }
    out: list[dict[str, Any]] = []
    for name, pred in subsets.items():
        idxs = [i for i, row in enumerate(rows) if pred(row)]
        sub_rows = [rows[i] for i in idxs]
        sub_mask = [selected_mask[i] for i in idxs]
        if len(sub_rows) < 2:
            continue
        metric = metric_for_mask(sub_rows, f"best_composite_on_{name}", sub_mask)
        y = [f(row.get("L3_handoff_transfer_penalty_proxy")) for row in sub_rows]
        bad_mask = [1.0 if row.get("case_label") == "non_good" else 0.0 for row in sub_rows]
        metric["subset"] = name
        metric["subset_case_count"] = len(sub_rows)
        metric["subset_oracle_bad_selector_corr_L3"] = pearson(bad_mask, y)
        metric["subset_oracle_bad_selector_abs_corr_L3"] = abs(pearson(bad_mask, y))
        out.append(metric)
    return out


def write_report(path: Path, summary: dict[str, Any], label_rows: list[dict[str, Any]], rules: list[dict[str, Any]], top: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Missed-Positive and L3 Consistency Audit",
        "",
        "Diagnostic-only. This report does not authorize M3, E4, runtime action, or full validation.",
        "",
        "## Oracle Label-vs-L3 Check",
        "",
        "```json",
        json.dumps(clean(summary.get("oracle", {})), ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Rule Pool",
        "",
        "```json",
        json.dumps(clean([{k: v for k, v in rule.items() if k != "children"} for rule in rules]), ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Best Composite Candidates",
        "",
        "```json",
        json.dumps(clean(top[:10]), ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Subset Metrics For Best Composite",
        "",
        "```json",
        json.dumps(clean(summary.get("subset_metrics", [])), ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## L3-Sorted Case Labels",
        "",
    ]
    for row in label_rows:
        conflict = f" conflict={row['label_l3_conflict']}" if row.get("label_l3_conflict") else ""
        lines.append(
            f"- {row['case_id']} {row['case_label']} L3={row['L3_handoff_transfer_penalty_proxy']} "
            f"failure={row.get('failure_type','')}{conflict}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    rows = merge_case_rows()
    oracle, label_rows = oracle_rows(rows)
    rules = build_rule_pool(rows)
    composites = composite_search(rows, rules)
    gate_like = [row for row in composites if row.get("gate_like")]
    rule_masks = {str(rule["name"]): rule_selected(rows, rule) for rule in rules}
    best_rules = str(composites[0].get("rules", "") if composites else "").split(";")
    best_mask = [
        any(rule_masks.get(name, [False] * len(rows))[i] for name in best_rules if name)
        for i in range(len(rows))
    ] if best_rules else [False] * len(rows)
    selected_by_case = {
        str(row.get("case_id", "")): bool(selected)
        for row, selected in zip(rows, best_mask)
    }
    for label_row in label_rows:
        label_row["selected_by_best_composite"] = selected_by_case.get(str(label_row.get("case_id", "")), False)
    ftype_rows = failure_type_rows(rows, label_rows, best_mask)
    subset_rows = subset_metrics(rows, label_rows, best_mask)
    summary = {
        "case_count": len(rows),
        "oracle": oracle,
        "rule_count": len(rules),
        "candidate_count": len(composites),
        "gate_like_count": len(gate_like),
        "best": composites[0] if composites else {},
        "best_gate_like": gate_like[0] if gate_like else {},
        "subset_metrics": subset_rows,
        "runtime_action_allowed": False,
        "note": "Checks whether binary case labels and L3 proxy make selector-L3 gate feasible; diagnostic only.",
    }
    out = args.out_dir
    write_rows(out / "missed_positive_l3_case_rows.csv", label_rows)
    write_rows(out / "missed_positive_l3_failure_type_rows.csv", ftype_rows)
    write_rows(out / "missed_positive_l3_subset_metrics.csv", subset_rows)
    write_rows(out / "missed_positive_l3_rule_pool.csv", [{k: v for k, v in rule.items() if k != "children"} for rule in rules])
    write_rows(out / "missed_positive_l3_composite_search_top_metrics.csv", composites[:500])
    write_json(out / "missed_positive_l3_consistency_summary.json", clean(summary))
    write_report(out / "missed_positive_l3_consistency_report.md", summary, label_rows, rules, composites)
    print(json.dumps(clean(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
