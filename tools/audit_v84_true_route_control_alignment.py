#!/usr/bin/env python3
"""Align landed true-route attention-mass controls with v84 support rows.

This is an audit bridge, not a runtime method.  It reuses v82 per-head
attention-mass/control rows and joins them to the v84 Memory Ruler support bank
by seq/curr_chunk, keeping the limitation explicit: the landed route masks are
stable-agreement or semantic-samegroup selections, not v84 RULER_ANCHOR masks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping


DEFAULT_V84_AUDIT = Path(
    "results/acl2_v84tf_memory_ruler_audit/phase10_support_expansion_audit/"
    "support_expansion_audit_by_pair.csv"
)
DEFAULT_ROUTE_ROWS = Path(
    "results/acl2_v82tf_swa_carrier_semantic_scale_handoff/"
    "phase12_route_control_rule_refinement/route_control_rule_rows_joined.csv"
)
DEFAULT_OUT_DIR = Path("results/acl2_v84tf_memory_ruler_audit/phase13_true_route_control_alignment")

ROUTE_GROUPS_FOR_SEMANTIC_SPECIFICITY = {"semantic_samegroup_all_head", "semantic_samegroup_head15"}
MARGIN_GATE = 0.05
BAD_RECALL_GATE = 0.60
GOOD_FPR_GATE = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v84-audit", type=Path, default=DEFAULT_V84_AUDIT)
    parser.add_argument("--route-rows", type=Path, default=DEFAULT_ROUTE_ROWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize(row.get(key, "")) for key in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def serialize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.12g}"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def safe_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def safe_int_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def seq_norm(value: Any) -> str:
    text = str(value or "").strip()
    return text.zfill(2) if text else ""


def pair_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return seq_norm(row.get("seq")), safe_int_text(row.get("curr_chunk") or row.get("chunk"))


def median_or_none(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(median(vals)) if vals else None


def gate_metrics(rows: list[dict[str, Any]], *, route_group: str, control_kind: str, margin_field: str) -> dict[str, Any]:
    subset = [
        row
        for row in rows
        if row.get("route_group") == route_group
        and row.get("control_kind") == control_kind
        and row.get("pair_complete") is True
    ]
    bad = [row for row in subset if row.get("base_case_type") == "bad"]
    good = [row for row in subset if row.get("base_case_type") == "good"]
    bad_pos = [
        row for row in bad if (safe_float(row.get(margin_field)) or -1e9) >= MARGIN_GATE
    ]
    good_pos = [
        row for row in good if (safe_float(row.get(margin_field)) or -1e9) >= MARGIN_GATE
    ]
    margins = [safe_float(row.get(margin_field)) for row in subset]
    margins = [v for v in margins if v is not None]
    return {
        "route_group": route_group,
        "control_kind": control_kind,
        "margin_field": margin_field,
        "rows": len(subset),
        "bad_rows": len(bad),
        "good_rows": len(good),
        "median_margin": median_or_none(margins),
        "max_margin": max(margins) if margins else None,
        "bad_positive_count_margin_ge_0_05": len(bad_pos),
        "good_positive_count_margin_ge_0_05": len(good_pos),
        "bad_recall_margin_ge_0_05": len(bad_pos) / max(len(bad), 1),
        "good_fpr_margin_ge_0_05": len(good_pos) / max(len(good), 1),
        "gate_pass": (
            len(bad) > 0
            and len(good) > 0
            and len(bad_pos) / max(len(bad), 1) >= BAD_RECALL_GATE
            and len(good_pos) / max(len(good), 1) <= GOOD_FPR_GATE
            and (median_or_none(margins) or -1e9) >= MARGIN_GATE
        ),
    }


def main() -> None:
    args = parse_args()
    v84_rows = read_csv(args.v84_audit)
    route_rows = read_csv(args.route_rows)

    route_by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for route in route_rows:
        route_by_key[(seq_norm(route.get("seq_base") or route.get("seq")), safe_int_text(route.get("chunk")))].append(route)

    joined: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    for v84 in v84_rows:
        key = pair_key(v84)
        routes = route_by_key.get(key, [])
        if not routes:
            unmatched.append(
                {
                    "seq": key[0],
                    "curr_chunk": key[1],
                    "case_type": v84.get("case_type"),
                    "base_case_type": v84.get("base_case_type"),
                    "reason": "no_landed_route_control_row_for_seq_curr_chunk",
                }
            )
            continue
        for route in routes:
            pair_complete = str(route.get("pair_complete", "")).lower() == "true"
            actual_attention = str(route.get("actual_attention_mass_available", "")).lower() == "true"
            control_attention = str(route.get("control_attention_mass_available", "")).lower() == "true"
            row = {
                "seq": key[0],
                "prev_chunk": safe_int_text(v84.get("prev_chunk")),
                "curr_chunk": key[1],
                "case_type": v84.get("case_type"),
                "base_case_type": v84.get("base_case_type"),
                "support_expansion_label_scope": v84.get("support_expansion_label_scope"),
                "ruler_anchor_count": safe_float(v84.get("ruler_anchor_count")) or 0.0,
                "ruler_risk_count": safe_float(v84.get("ruler_risk_count")) or 0.0,
                "ruler_degenerate_count": safe_float(v84.get("ruler_degenerate_count")) or 0.0,
                "failure_flags": v84.get("failure_flags"),
                "route_group": route.get("route_group"),
                "control_kind": route.get("control_kind"),
                "pair_complete": pair_complete,
                "actual_attention_mass_available": actual_attention,
                "control_attention_mass_available": control_attention,
                "actual_is_semantic_candidate": str(route.get("actual_is_semantic_candidate", "")).lower() == "true",
                "actual_case": route.get("actual_case"),
                "control_case": route.get("control_case"),
                "actual_minus_control_selected_lift": safe_float(
                    route.get("actual_minus_control_mean_swa_overlap_attention_mass_selected_lift")
                ),
                "actual_minus_control_source_lift": safe_float(
                    route.get("actual_minus_control_mean_swa_overlap_attention_mass_source_lift")
                ),
                "actual_minus_control_selected_head_max_lift": safe_float(
                    route.get("actual_minus_control_mean_swa_overlap_attention_mass_selected_head_max_lift")
                ),
                "actual_selected_top_head_by_lift_mode": route.get("actual_selected_top_head_by_lift_mode"),
                "control_selected_top_head_by_lift_mode": route.get("control_selected_top_head_by_lift_mode"),
                "route_mask_scope": (
                    "v82_stable_or_semantic_samegroup_proxy;not_v84_ruler_anchor_mask"
                ),
            }
            joined.append(row)

    labelled = [row for row in joined if row.get("base_case_type") in {"bad", "good"}]
    labelled_pairs = {
        (row.get("seq"), row.get("curr_chunk"))
        for row in v84_rows
        if row.get("base_case_type") in {"bad", "good"}
    }
    covered_labelled_pairs = {
        (row.get("seq"), row.get("curr_chunk"))
        for row in labelled
        if row.get("pair_complete") and row.get("actual_attention_mass_available") and row.get("control_attention_mass_available")
    }
    all_pairs = {(row.get("seq"), safe_int_text(row.get("curr_chunk"))) for row in v84_rows}
    covered_all_pairs = {
        (row.get("seq"), row.get("curr_chunk"))
        for row in joined
        if row.get("pair_complete") and row.get("actual_attention_mass_available") and row.get("control_attention_mass_available")
    }

    groups = sorted({str(row.get("route_group")) for row in joined if row.get("route_group")})
    controls = sorted({str(row.get("control_kind")) for row in joined if row.get("control_kind")})
    margin_rows: list[dict[str, Any]] = []
    for group in groups:
        for control in controls:
            for field in [
                "actual_minus_control_selected_lift",
                "actual_minus_control_source_lift",
                "actual_minus_control_selected_head_max_lift",
            ]:
                margin_rows.append(gate_metrics(joined, route_group=group, control_kind=control, margin_field=field))

    semantic_specificity_rows = [
        row
        for row in margin_rows
        if row["route_group"] in ROUTE_GROUPS_FOR_SEMANTIC_SPECIFICITY
        and row["control_kind"] == "shuffled_semantic"
    ]
    same_mass_rows = [row for row in margin_rows if row["control_kind"] == "same_mass_random"]
    best_semantic = max(
        semantic_specificity_rows,
        key=lambda row: safe_float(row.get("median_margin")) or -1e9,
        default=None,
    )
    best_same_mass = max(
        same_mass_rows,
        key=lambda row: safe_float(row.get("median_margin")) or -1e9,
        default=None,
    )
    gate_pass = any(row.get("gate_pass") for row in margin_rows)
    route_group_counts = Counter(str(row.get("route_group")) for row in joined)
    control_counts = Counter(str(row.get("control_kind")) for row in joined)
    summary = {
        "schema": "acl2_v84_true_route_control_alignment_v1",
        "alignment_gate_pass": bool(gate_pass),
        "v84_support_rows": len(v84_rows),
        "v84_support_pairs": len(all_pairs),
        "joined_rows": len(joined),
        "unmatched_v84_rows": len(unmatched),
        "covered_all_pairs_with_true_route_controls": len(covered_all_pairs),
        "covered_all_pair_ratio": len(covered_all_pairs) / max(len(all_pairs), 1),
        "labelled_pairs": len(labelled_pairs),
        "covered_labelled_pairs_with_true_route_controls": len(covered_labelled_pairs),
        "covered_labelled_pair_ratio": len(covered_labelled_pairs) / max(len(labelled_pairs), 1),
        "route_group_counts": dict(sorted(route_group_counts.items())),
        "control_kind_counts": dict(sorted(control_counts.items())),
        "semantic_shuffle_control_available": bool(semantic_specificity_rows),
        "same_mass_control_available": bool(same_mass_rows),
        "best_semantic_shuffle_margin_row": best_semantic,
        "best_same_mass_margin_row": best_same_mass,
        "margin_gate_threshold": MARGIN_GATE,
        "bad_recall_gate": BAD_RECALL_GATE,
        "good_fpr_gate": GOOD_FPR_GATE,
        "limitations": [
            "Landed route rows are true attention-mass controls, but their selected masks are v82 stable-agreement or semantic-samegroup masks.",
            "This audit does not prove v84 RULER_ANCHOR-specific route use.",
            "Rows added by v84 Phase10 support expansion without v82 route-control jobs remain unmatched.",
        ],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "true_route_control_alignment_rows.csv", joined)
    write_csv(args.out_dir / "true_route_control_margin_summary.csv", margin_rows)
    write_csv(args.out_dir / "true_route_control_unmatched_v84_rows.csv", unmatched)
    write_json(args.out_dir / "true_route_control_alignment_summary.json", summary)
    report = [
        "# v84 True Route Control Alignment",
        "",
        f"- Alignment gate pass: `{summary['alignment_gate_pass']}`",
        f"- Covered labelled pairs: {summary['covered_labelled_pairs_with_true_route_controls']} / {summary['labelled_pairs']}",
        f"- Covered all v84 support pairs: {summary['covered_all_pairs_with_true_route_controls']} / {summary['v84_support_pairs']}",
        f"- Semantic-shuffle available: `{summary['semantic_shuffle_control_available']}`",
        f"- Same-mass available: `{summary['same_mass_control_available']}`",
        "",
        "## Limitation",
        "",
        "This uses landed v82 true attention-mass controls. The selected route masks are not v84 RULER_ANCHOR masks, so any failure is a route-control warning, not a direct v84 anchor-route measurement.",
        "",
    ]
    (args.out_dir / "true_route_control_alignment_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
