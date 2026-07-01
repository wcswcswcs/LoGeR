#!/usr/bin/env python3
"""Audit ACL2 v83 Phase3 carrier alignment.

The audit is diagnostic-only. It does not run a runtime action and does not
promote any carrier unless bad/good separation, actual-vs-random evidence, and
semantic-shuffle specificity are all satisfied.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_MATRIX = Path(
    "results/acl2_v83tf_clue_sufficiency_vs_action_misuse/"
    "phase1_unified_clue_matrix/unified_clue_matrix.csv"
)
DEFAULT_OUT_DIR = Path(
    "results/acl2_v83tf_clue_sufficiency_vs_action_misuse/"
    "phase3_carrier_alignment"
)
DEFAULT_VISUAL_AUDIT = DEFAULT_OUT_DIR / "visual_integrity_audit.json"
DEFAULT_V82_ROUTE_DECOMP = Path(
    "results/acl2_v82tf_swa_carrier_semantic_scale_handoff/"
    "phase12_route_gate_failure_decomp/route_gate_failure_decomp_summary.json"
)
DEFAULT_V82_ROUTE_GROUPS = Path(
    "results/acl2_v82tf_swa_carrier_semantic_scale_handoff/"
    "phase12_route_gate_failure_decomp/route_gate_failure_by_route_group.csv"
)
DEFAULT_V82_PER_HEAD_SUMMARY = Path(
    "results/acl2_v82tf_swa_carrier_semantic_scale_handoff/"
    "phase12_per_head_route_localization/per_head_route_localization_summary.json"
)
DEFAULT_V82_LEDGER_DECISION = Path(
    "results/acl2_v82tf_swa_carrier_semantic_scale_handoff/"
    "phase4_swa_carrier_ledger/swa_carrier_ledger_decision.json"
)


FAMILY_FIELDS = {
    "READ": [
        "READ_used_stable_mass",
        "READ_used_harm_mass",
        "QK_pair_compatibility",
        "query_risk_mass",
        "read_entropy",
    ],
    "SWA": [
        "current_Q_alignment",
        "cache_K_alignment",
        "cache_V_alignment",
        "K_risk_delta",
        "V_protect_delta",
        "route_mass",
        "head_layer_sensitivity",
        "actual_vs_random_route_delta",
    ],
    "merge_gauge": [
        "boundary_transform_residual",
        "merge_raw_overlap_residual",
        "postmerge_pose_sensitivity",
        "reset_relative_position",
        "gauge_hold_signal",
    ],
    "TTT": [
        "selected_low_support_ratio",
        "continuous_low_support_cluster_len",
        "update_conflict",
        "post_zp_delta",
        "write_mass_stable",
        "write_mass_harm",
        "write_mass_context",
    ],
}

LOW_IS_RISK = {
    "READ_used_stable_mass",
    "QK_pair_compatibility",
    "current_Q_alignment",
    "cache_K_alignment",
    "cache_V_alignment",
    "V_protect_delta",
    "route_mass",
    "actual_vs_random_route_delta",
    "gauge_hold_signal",
    "write_mass_stable",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--visual-audit", type=Path, default=DEFAULT_VISUAL_AUDIT)
    parser.add_argument("--v82-route-decomp", type=Path, default=DEFAULT_V82_ROUTE_DECOMP)
    parser.add_argument("--v82-route-groups", type=Path, default=DEFAULT_V82_ROUTE_GROUPS)
    parser.add_argument("--v82-per-head-summary", type=Path, default=DEFAULT_V82_PER_HEAD_SUMMARY)
    parser.add_argument("--v82-ledger-decision", type=Path, default=DEFAULT_V82_LEDGER_DECISION)
    parser.add_argument("--selection-quantile", type=float, default=0.60)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize(row.get(key, "")) for key in keys})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def serialize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def safe_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = max(0.0, min(1.0, q)) * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def auc_score(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    pairs = [(int(y), float(s)) for y, s in zip(labels, scores) if math.isfinite(float(s))]
    pos = [s for y, s in pairs if y == 1]
    neg = [s for y, s in pairs if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    total = float(len(pos) * len(neg))
    for ps in pos:
        for ns in neg:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / total


def normalize_feature(rows: list[dict[str, str]], field: str) -> dict[str, float]:
    values: list[tuple[str, float]] = []
    for row in rows:
        value = safe_float(row.get(field))
        if value is not None:
            values.append((row["row_id"], value))
    if not values:
        return {}
    vals = [value for _, value in values]
    lo = min(vals)
    hi = max(vals)
    if math.isclose(lo, hi):
        return {}
    out: dict[str, float] = {}
    for row_id, value in values:
        norm = (value - lo) / (hi - lo)
        out[row_id] = 1.0 - norm if field in LOW_IS_RISK else norm
    return out


def score_rows(rows: list[dict[str, str]], fields: Sequence[str]) -> tuple[dict[str, float], dict[str, Any]]:
    normalized = {field: normalize_feature(rows, field) for field in fields}
    used_fields = [field for field, values in normalized.items() if values]
    scores: dict[str, float] = {}
    feature_counts: dict[str, int] = {}
    for row in rows:
        vals = [normalized[field][row["row_id"]] for field in used_fields if row["row_id"] in normalized[field]]
        if vals:
            scores[row["row_id"]] = mean(vals)
            feature_counts[row["row_id"]] = len(vals)
    return scores, {
        "requested_fields": list(fields),
        "used_fields": used_fields,
        "dropped_constant_or_empty_fields": [field for field in fields if field not in used_fields],
        "feature_count_min": min(feature_counts.values()) if feature_counts else 0,
        "feature_count_max": max(feature_counts.values()) if feature_counts else 0,
        "feature_count_mean": mean(feature_counts.values()) if feature_counts else 0.0,
    }


def metrics_for_scores(
    rows: list[dict[str, str]],
    scores: Mapping[str, float],
    selection_quantile: float,
) -> dict[str, Any]:
    scored = [row for row in rows if row["row_id"] in scores]
    threshold = percentile([scores[row["row_id"]] for row in scored], selection_quantile)
    if threshold is None:
        return {
            "scored_rows": 0,
            "bad_rows": 0,
            "good_rows": 0,
            "threshold": None,
            "positive_count": 0,
            "bad_recall": None,
            "good_false_positive_rate": None,
            "balanced_accuracy": None,
            "auc": None,
            "sequence_coverage": [],
            "selected_sequence_coverage": [],
        }
    selected = [row for row in scored if scores[row["row_id"]] >= threshold]
    bad = [row for row in scored if safe_int(row.get("target_label")) == 1]
    good = [row for row in scored if safe_int(row.get("target_label")) == 0]
    selected_bad = [row for row in selected if safe_int(row.get("target_label")) == 1]
    selected_good = [row for row in selected if safe_int(row.get("target_label")) == 0]
    bad_recall = len(selected_bad) / len(bad) if bad else None
    good_fpr = len(selected_good) / len(good) if good else None
    tnr = 1.0 - good_fpr if good_fpr is not None else None
    return {
        "scored_rows": len(scored),
        "bad_rows": len(bad),
        "good_rows": len(good),
        "threshold": threshold,
        "positive_count": len(selected),
        "selected_bad": len(selected_bad),
        "selected_good": len(selected_good),
        "max_possible_bad_recall_at_positive_count": (min(len(selected), len(bad)) / len(bad)) if bad else None,
        "bad_recall": bad_recall,
        "good_false_positive_rate": good_fpr,
        "balanced_accuracy": ((bad_recall + tnr) / 2.0) if bad_recall is not None and tnr is not None else None,
        "auc": auc_score(
            [safe_int(row.get("target_label")) for row in scored],
            [scores[row["row_id"]] for row in scored],
        ),
        "sequence_coverage": sorted({row.get("seq", "") for row in scored if row.get("seq", "")}),
        "selected_sequence_coverage": sorted({row.get("seq", "") for row in selected if row.get("seq", "")}),
    }


def field_means(rows: list[dict[str, str]], fields: Sequence[str]) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    bad_means: dict[str, float] = {}
    good_means: dict[str, float] = {}
    counts: dict[str, int] = {}
    for field in fields:
        bad_vals = [safe_float(row.get(field)) for row in rows if safe_int(row.get("target_label")) == 1]
        good_vals = [safe_float(row.get(field)) for row in rows if safe_int(row.get("target_label")) == 0]
        bad = [value for value in bad_vals if value is not None]
        good = [value for value in good_vals if value is not None]
        if bad:
            bad_means[field] = mean(bad)
        if good:
            good_means[field] = mean(good)
        counts[field] = len(bad) + len(good)
    return bad_means, good_means, counts


def route_specificity(args: argparse.Namespace) -> dict[str, Any]:
    decomp = read_json_if_exists(args.v82_route_decomp)
    group_rows = read_csv(args.v82_route_groups) if args.v82_route_groups.is_file() else []
    per_head = read_json_if_exists(args.v82_per_head_summary)
    ledger = read_json_if_exists(args.v82_ledger_decision)

    semantic_available_rows = sum(safe_int(row.get("semantic_shuffle_available_rows")) for row in group_rows)
    semantic_pass_rows = sum(safe_int(row.get("semantic_shuffle_pass_rows")) for row in group_rows)
    same_mass_rows = sum(safe_int(row.get("same_mass_rule_rows")) for row in group_rows)
    same_mass_within_gate_rows = sum(safe_int(row.get("same_mass_within_gate_rows")) for row in group_rows)
    group_summary = per_head.get("group_summary", {}) if isinstance(per_head.get("group_summary"), dict) else {}
    shuffled_pairs = sum(safe_int(item.get("shuffled_semantic_pairs")) for item in group_summary.values() if isinstance(item, dict))
    same_mass_pairs = sum(safe_int(item.get("same_mass_random_pairs")) for item in group_summary.values() if isinstance(item, dict))
    top_near = decomp.get("top_near_misses", [])
    return {
        "v82_route_decomp": str(args.v82_route_decomp),
        "v82_route_groups": str(args.v82_route_groups),
        "v82_per_head_summary": str(args.v82_per_head_summary),
        "v82_ledger_decision": str(args.v82_ledger_decision),
        "fully_passing_rule_rows": safe_int(decomp.get("fully_passing_rule_rows")),
        "same_mass_rule_rows": same_mass_rows or safe_int(decomp.get("same_mass_rule_rows")),
        "same_mass_within_gate_rows": same_mass_within_gate_rows,
        "same_mass_near_without_full_gate_rows": safe_int(decomp.get("same_mass_near_without_full_gate_rows")),
        "near_with_semantic_shuffle_available": safe_int(decomp.get("near_with_semantic_shuffle_available")),
        "near_without_semantic_shuffle_available": safe_int(decomp.get("near_without_semantic_shuffle_available")),
        "semantic_shuffle_available_rows": semantic_available_rows,
        "semantic_shuffle_pass_rows": semantic_pass_rows,
        "per_head_observability_gate_pass": bool(per_head.get("per_head_observability_gate_pass")),
        "same_mass_control_gate_pass": bool(per_head.get("same_mass_control_gate_pass")),
        "same_mass_control_pairs_complete": safe_int(per_head.get("same_mass_control_pairs_complete")) or same_mass_pairs,
        "shuffled_semantic_pairs": shuffled_pairs,
        "v82_ledger_phase4_gate_pass": bool(ledger.get("gate", {}).get("phase4_gate_pass")),
        "v82_ledger_blocker": ledger.get("blocker", ""),
        "top_near_misses": top_near[:5] if isinstance(top_near, list) else [],
    }


def build_rows(
    matrix_rows: list[dict[str, str]],
    args: argparse.Namespace,
    visual_audit: dict[str, Any],
    specificity: dict[str, Any],
) -> list[dict[str, Any]]:
    visual_gate = bool(visual_audit.get("gate", {}).get("visual_audit_gate_pass"))
    visual_actual_random_rows = safe_int(visual_audit.get("actual_vs_random_difference_rows"))
    rows: list[dict[str, Any]] = []
    for family, fields in FAMILY_FIELDS.items():
        scores, meta = score_rows(matrix_rows, fields)
        metrics = metrics_for_scores(matrix_rows, scores, args.selection_quantile)
        bad_means, good_means, counts = field_means(matrix_rows, fields)
        coverage_gate = len(metrics.get("sequence_coverage") or []) >= 3
        good_fp = metrics.get("good_false_positive_rate")
        bad_recall = metrics.get("bad_recall")
        auc = metrics.get("auc")
        separation_gate = (
            bad_recall is not None
            and bad_recall >= 0.60
            and good_fp is not None
            and good_fp <= 0.25
            and auc is not None
            and auc > 0.50
        )

        if family == "SWA":
            actual_vs_random_available = visual_actual_random_rows >= 24 or specificity["same_mass_control_pairs_complete"] >= 24
            actual_vs_random_pass = actual_vs_random_available and (
                specificity["same_mass_within_gate_rows"] > 0 or visual_actual_random_rows >= 24
            )
            semantic_shuffle_available = specificity["semantic_shuffle_available_rows"] > 0 or specificity["shuffled_semantic_pairs"] > 0
            semantic_shuffle_specificity_pass = (
                specificity["semantic_shuffle_pass_rows"] > 0 and specificity["fully_passing_rule_rows"] > 0
            )
            blocker = (
                "semantic_shuffle_specificity_failed"
                if semantic_shuffle_available and not semantic_shuffle_specificity_pass
                else "missing_semantic_shuffle_control"
                if not semantic_shuffle_available
                else ""
            )
        elif family == "READ":
            actual_vs_random_available = False
            actual_vs_random_pass = False
            semantic_shuffle_available = False
            semantic_shuffle_specificity_pass = False
            blocker = "missing_read_same_pair_random_and_semantic_shuffle_qk_controls"
        elif family == "merge_gauge":
            actual_vs_random_available = False
            actual_vs_random_pass = False
            semantic_shuffle_available = False
            semantic_shuffle_specificity_pass = False
            blocker = "missing_merge_same_overlap_random_and_semantic_shuffle_weighting_controls"
        else:
            actual_vs_random_available = False
            actual_vs_random_pass = False
            semantic_shuffle_available = False
            semantic_shuffle_specificity_pass = False
            blocker = "ttt_not_eligible_without_confirmed_swa_or_merge_carrier"

        carrier_gate = (
            family != "TTT"
            and coverage_gate
            and visual_gate
            and separation_gate
            and actual_vs_random_pass
            and semantic_shuffle_specificity_pass
        )
        if carrier_gate:
            conclusion = "carrier_aligned"
            blocker = ""
        elif family == "TTT":
            conclusion = "not_eligible"
        else:
            conclusion = "not_aligned_or_not_specific"
        rows.append(
            {
                "carrier_body": family,
                **meta,
                **metrics,
                "field_nonempty_counts": counts,
                "field_mean_bad": bad_means,
                "field_mean_good": good_means,
                "coverage_gate_pass": coverage_gate,
                "bad_good_separation_gate_pass": separation_gate,
                "visual_audit_gate_pass": visual_gate,
                "actual_vs_random_available": actual_vs_random_available,
                "actual_vs_random_gate_pass": actual_vs_random_pass,
                "semantic_shuffle_available": semantic_shuffle_available,
                "semantic_shuffle_specificity_gate_pass": semantic_shuffle_specificity_pass,
                "good_case_false_positive_gate_pass": good_fp is not None and good_fp <= 0.25,
                "carrier_gate_pass": carrier_gate,
                "conclusion": conclusion,
                "blocker": blocker,
            }
        )
    return rows


def write_report(path: Path, rows: list[dict[str, Any]], summary: Mapping[str, Any]) -> None:
    lines = [
        "# ACL2 v83 Phase3 Carrier Alignment Report",
        "",
        f"phase3_gate_pass: `{summary['phase3_gate_pass']}`",
        f"decision: `{summary['decision']}`",
        f"passing_carriers: `{summary['passing_carriers']}`",
        "",
        "| Carrier | Gate | AUC | Bad Recall | Good FPR | Actual/Random | Semantic Shuffle | Blocker |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {carrier} | {gate} | {auc} | {bad} | {fpr} | {ar} | {ss} | {blocker} |".format(
                carrier=row["carrier_body"],
                gate=row["carrier_gate_pass"],
                auc=format_float(row.get("auc")),
                bad=format_float(row.get("bad_recall")),
                fpr=format_float(row.get("good_false_positive_rate")),
                ar=row["actual_vs_random_gate_pass"],
                ss=row["semantic_shuffle_specificity_gate_pass"],
                blocker=row.get("blocker", ""),
            )
        )
    lines.extend(
        [
            "",
            "Gate requires coverage >=3 sequences, visual audit pass, bad_recall >=0.60, good_false_positive_rate <=0.25, actual-vs-random evidence, and semantic-shuffle specificity.",
            "No runtime action was executed by this diagnostic.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def format_float(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.6f}"


def main() -> None:
    args = parse_args()
    rows = read_csv(args.matrix)
    visual_audit = read_json_if_exists(args.visual_audit)
    specificity = route_specificity(args)
    carrier_rows = build_rows(rows, args, visual_audit, specificity)
    passing = [row["carrier_body"] for row in carrier_rows if row.get("carrier_gate_pass")]
    summary = {
        "schema": "acl2_v83_phase3_carrier_alignment_summary_v1",
        "matrix": str(args.matrix),
        "out_dir": str(args.out_dir),
        "rows": len(rows),
        "row_counts_by_scope": dict(Counter(row.get("row_scope", "") for row in rows)),
        "case_counts": dict(Counter(row.get("case_type", "") for row in rows)),
        "selection_quantile": args.selection_quantile,
        "visual_audit_gate_pass": bool(visual_audit.get("gate", {}).get("visual_audit_gate_pass")),
        "visual_audit": str(args.visual_audit),
        "route_specificity_evidence": specificity,
        "passing_carriers": passing,
        "phase3_gate_pass": bool(passing),
        "runtime_action_allowed": bool(passing),
        "rediscovery_required": not bool(passing),
        "rediscovery_actions": [
            "add missing Q/K/V head/layer dumps",
            "add actual-vs-same-mass random panels",
            "add semantic-shuffle panels",
            "run visual review",
            "if still fails, mark carrier_not_localized",
        ]
        if not passing
        else [],
        "decision": "carrier_aligned_continue_to_phase4" if passing else "carrier_not_localized_no_runtime_action",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "carrier_alignment_rows.csv", carrier_rows)
    write_json(args.out_dir / "carrier_alignment_summary.json", summary)
    write_report(args.out_dir / "carrier_alignment_report.md", carrier_rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
