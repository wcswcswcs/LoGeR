#!/usr/bin/env python3
"""Audit whether v94/v95 cases can legally extend the v101 Track T target set.

This script is intentionally read-only with respect to source artifacts.  It
does not relabel cases for runtime use; it writes diagnostic evidence under the
v101 Track T result directory.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
TRACK_T = ROOT / "trackT_drift_target_relabel"
V100_ROOT = Path("results/acl2_v100tf_same_space_semantic_anchor_latent_state_memory_control")
V95_ROOT = Path("results/acl2_v95tf_multiroute_semantic_memory_evidence_control")
V94_ROOT = Path("results/acl2_v94tf_semantic_gauge_failure_localization_causal_memory_control")

V100_CASE_ROWS = V100_ROOT / "trackD4_read_current_support_provider/label_l3_hygiene_provenance_rows.csv"
V100_SAME_SPACE_ROWS = V100_ROOT / "trackS_same_space_latent_state/same_space_anchor_rows.csv"
V95_CANONICAL_CASES = V95_ROOT / "trackA_base_case_bank/canonical_case_rows.csv"
V95_GOOD_CONTROLS = V95_ROOT / "trackA_base_case_bank/good_controls.csv"
V94_BOUNDARY_ROWS = V94_ROOT / "phase1_boundary_failure_atlas/boundary_failure_rows.csv"
V100_Q_ROWS = V100_ROOT / "trackQ_chunk_update_admission/rows.csv"


def f(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def b(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    if not keys:
        keys = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            clean: dict[str, Any] = {}
            for key in keys:
                value = row.get(key, "")
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, ensure_ascii=False, sort_keys=True)
                clean[key] = value
            writer.writerow(clean)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def median(values: list[Any]) -> float:
    vals = sorted(f(v) for v in values if math.isfinite(f(v)))
    return statistics.median(vals) if vals else math.nan


def quantile(values: list[Any], q: float) -> float:
    vals = sorted(f(v) for v in values if math.isfinite(f(v)))
    if not vals:
        return math.nan
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def mad(values: list[Any]) -> float:
    vals = [f(v) for v in values if math.isfinite(f(v))]
    if not vals:
        return math.nan
    med = statistics.median(vals)
    return statistics.median(abs(v - med) for v in vals)


def case_seq(case_id: str) -> str:
    return case_id.split("_")[0]


def pearson(xs: list[Any], ys: list[Any]) -> float:
    pairs: list[tuple[float, float]] = []
    for x, y in zip(xs, ys):
        fx = f(x)
        fy = f(y)
        if math.isfinite(fx) and math.isfinite(fy):
            pairs.append((fx, fy))
    if len(pairs) < 2:
        return math.nan
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 1.0e-12 or vy <= 1.0e-12:
        return math.nan
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def selected_metrics(rows: list[dict[str, Any]], selected: set[str], positives: set[str], negatives: set[str]) -> dict[str, Any]:
    cases = sorted((positives | negatives) & {str(row.get("case_id", "")) for row in rows})
    pos = positives & set(cases)
    neg = negatives & set(cases)
    tp = selected & pos
    fp = selected & neg
    missed = pos - selected
    recall = len(tp) / len(pos) if pos else math.nan
    fpr = len(fp) / len(neg) if neg else math.nan
    seq_counts = Counter(case_seq(case) for case in tp)
    corr = pearson(
        [1.0 if case in selected else 0.0 for case in cases],
        [f(next((row.get("L3_handoff_transfer_penalty_proxy") for row in rows if row.get("case_id") == case), math.nan)) for case in cases],
    )
    return {
        "available_case_count": len(cases),
        "positive_case_count": len(pos),
        "negative_case_count": len(neg),
        "selected_case_count": len(selected & set(cases)),
        "bad_recall": recall,
        "good_FPR": fpr,
        "balanced_accuracy": ((recall + (1.0 - fpr)) / 2.0) if math.isfinite(recall) and math.isfinite(fpr) else math.nan,
        "corr_L3": corr,
        "abs_corr_L3": abs(corr) if math.isfinite(corr) else math.nan,
        "selected_positive_sequence_coverage": len(seq_counts),
        "true_positive_cases": ";".join(sorted(tp)),
        "false_positive_cases": ";".join(sorted(fp)),
        "missed_positive_cases": ";".join(sorted(missed)),
    }


def normalize_label(raw_label: str) -> str:
    label = str(raw_label or "").strip().lower()
    if label == "good":
        return "good"
    if label in {"bad", "non_good", "nongood"}:
        return "non_good"
    if label == "unlabelled_support":
        return "unlabelled_support"
    if not label:
        return "missing"
    return label


def classify_v101(row: dict[str, Any], high: float, low: float) -> str:
    label = str(row.get("case_label", ""))
    failure = str(row.get("failure_type", "")).upper()
    l3 = f(row.get("L3_handoff_transfer_penalty_proxy"))
    hygiene_excluded = b(row.get("v98_hygiene_excluded_good_control"))
    l3_high = math.isfinite(l3) and math.isfinite(high) and l3 >= high
    l3_low = math.isfinite(l3) and math.isfinite(low) and l3 <= low
    lowobs_or_multimode = ("LOW_OBSERVABILITY" in failure) or ("MULTIMODE_CONFLICT" in failure)
    handoff_failure = ("HANDOFF_SCALE" in failure) or ("HANDOFF_GAUGE" in failure)
    if label == "good" and (hygiene_excluded or l3_high):
        return "GOOD_HIGH_L3_CONTAMINATED"
    if lowobs_or_multimode and l3_high:
        return "MULTIMODE_LOWOBS_ABSTAIN"
    if label != "good" and handoff_failure and l3_high:
        return "HANDOFF_SCALE_GAUGE_TARGET"
    if label != "good" and l3_low:
        return "LOCAL_BAD_NOT_HANDOFF"
    if label == "good" and l3_low and not hygiene_excluded:
        return "SAFE_GOOD"
    return "AMBIGUOUS_SUPPORT"


def build_threshold_variants(v100_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    l3_all = [f(row.get("L3_handoff_transfer_penalty_proxy")) for row in v100_rows]
    good_l3 = [f(row.get("L3_handoff_transfer_penalty_proxy")) for row in v100_rows if row.get("case_label") == "good"]
    base = {
        "median_good_L3": median(good_l3),
        "MAD_good_L3": mad(good_l3),
        "Q40_all_L3": quantile(l3_all, 0.40),
        "Q50_good_L3": quantile(good_l3, 0.50),
        "Q60_all_L3": quantile(l3_all, 0.60),
        "Q75_all_L3": quantile(l3_all, 0.75),
    }
    variants = [
        {
            "variant": "primary_v101_plan",
            "L3_high": max(base["median_good_L3"] + 2.0 * base["MAD_good_L3"], base["Q75_all_L3"]),
            "L3_low": base["Q40_all_L3"],
            "valid_for_gate": True,
            "note": "plan threshold",
        },
        {
            "variant": "relaxed_mad1_or_q60",
            "L3_high": max(base["median_good_L3"] + base["MAD_good_L3"], base["Q60_all_L3"]),
            "L3_low": base["Q40_all_L3"],
            "valid_for_gate": False,
            "note": "diagnostic relaxed high threshold",
        },
        {
            "variant": "q60_high_q40_low",
            "L3_high": base["Q60_all_L3"],
            "L3_low": base["Q40_all_L3"],
            "valid_for_gate": False,
            "note": "diagnostic quantile threshold",
        },
        {
            "variant": "q50good_high_q40_low",
            "L3_high": base["Q50_good_L3"],
            "L3_low": base["Q40_all_L3"],
            "valid_for_gate": False,
            "note": "aggressive diagnostic threshold; can contaminate good controls",
        },
        {
            "variant": "any_handoff_no_l3_high",
            "L3_high": -math.inf,
            "L3_low": base["Q40_all_L3"],
            "valid_for_gate": False,
            "note": "invalid for gate; tests effect of forcing low-L3 handoff failures into target",
        },
    ]
    rows: list[dict[str, Any]] = []
    for variant in variants:
        classes = [classify_v101(row, variant["L3_high"], variant["L3_low"]) for row in v100_rows]
        counts = Counter(classes)
        target_cases = [
            row.get("case_id", "")
            for row, taxonomy in zip(v100_rows, classes)
            if taxonomy == "HANDOFF_SCALE_GAUGE_TARGET"
        ]
        safe_good = [
            row.get("case_id", "")
            for row, taxonomy in zip(v100_rows, classes)
            if taxonomy == "SAFE_GOOD"
        ]
        contaminated = [
            row.get("case_id", "")
            for row, taxonomy in zip(v100_rows, classes)
            if taxonomy == "GOOD_HIGH_L3_CONTAMINATED"
        ]
        seq_cov = len({case_seq(case_id) for case_id in target_cases})
        raw_gate = len(v100_rows) == 28 and len(safe_good) >= 6 and len(target_cases) >= 8 and seq_cov >= 3
        strict_gate = raw_gate and bool(variant["valid_for_gate"])
        rows.append(
            {
                **variant,
                "safe_good_count": len(safe_good),
                "handoff_scale_gauge_target_count": len(target_cases),
                "handoff_target_sequence_coverage": seq_cov,
                "good_high_l3_contaminated_count": len(contaminated),
                "ambiguous_case_count": counts.get("AMBIGUOUS_SUPPORT", 0),
                "raw_count_gate_pass": raw_gate,
                "strict_v101_gate_pass": strict_gate,
                "target_counts": dict(counts),
                "handoff_target_cases": ";".join(sorted(target_cases)),
                "safe_good_cases": ";".join(sorted(safe_good)),
                "contaminated_good_cases": ";".join(sorted(contaminated)),
            }
        )
    summary = {
        "schema": "acl2_v101_trackT_threshold_variant_summary_v1",
        "base_threshold_terms": base,
        "variant_count": len(rows),
        "strict_gate_pass_variants": [row["variant"] for row in rows if row["strict_v101_gate_pass"]],
        "raw_count_gate_pass_variants": [row["variant"] for row in rows if row["raw_count_gate_pass"]],
        "claim": "Threshold variants are diagnostic only; invalid variants cannot authorize downstream action.",
    }
    return rows, summary


def collect_old_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_name, source_path, case_key, label_key, primary_key, secondary_key in [
        (
            "v95_trackA_base_case_bank",
            V95_CANONICAL_CASES,
            "case_id",
            "case_label_offline_only",
            "failure_type_primary",
            "failure_type_secondary",
        ),
        (
            "v94_phase1_boundary_failure_atlas",
            V94_BOUNDARY_ROWS,
            "pair_id",
            "case_label_offline_only",
            "failure_type_primary",
            "failure_type_secondary",
        ),
    ]:
        for raw in read_rows(source_path):
            case_id = str(raw.get(case_key, ""))
            primary = str(raw.get(primary_key, ""))
            secondary = str(raw.get(secondary_key, ""))
            failure = primary if not secondary else f"{primary};{secondary}"
            rows.append(
                {
                    "source_name": source_name,
                    "source_path": str(source_path),
                    "case_id": case_id,
                    "seq": raw.get("seq", case_seq(case_id)),
                    "raw_label": raw.get(label_key, ""),
                    "v101_label_norm": normalize_label(raw.get(label_key, "")),
                    "failure_type": failure,
                    "failure_type_primary": primary,
                    "failure_type_secondary": secondary,
                    "L3_handoff_transfer_penalty_proxy": raw.get("L3_handoff_transfer_penalty_proxy", raw.get("future_after_overlap", "")),
                    "semantic_evidence_type": raw.get("semantic_evidence_type", raw.get("semantic_evidence_type_majority", "")),
                    "trace_path": raw.get("trace_path", ""),
                    "trace_provenance": raw.get("trace_provenance", ""),
                    "offline_audit_label_only": raw.get("offline_audit_label_only", ""),
                    "no_gt_runtime_feature": raw.get("no_gt_runtime_feature", ""),
                }
            )
    return rows


def audit_extension(v100_rows: list[dict[str, str]], old_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    v100_cases = {row.get("case_id", "") for row in v100_rows}
    same_space_cases = {
        row.get("case_id", "")
        for row in read_rows(V100_SAME_SPACE_ROWS)
        if row.get("canonical_space_name") == "S-B_preprojection_hidden"
    }
    l3_all = [f(row.get("L3_handoff_transfer_penalty_proxy")) for row in v100_rows]
    good_l3 = [f(row.get("L3_handoff_transfer_penalty_proxy")) for row in v100_rows if row.get("case_label") == "good"]
    l3_high = max(median(good_l3) + 2.0 * mad(good_l3), quantile(l3_all, 0.75))
    l3_low = quantile(l3_all, 0.40)
    out_rows: list[dict[str, Any]] = []
    for row in old_rows:
        case_id = str(row.get("case_id", ""))
        label = str(row.get("v101_label_norm", ""))
        failure = str(row.get("failure_type", "")).upper()
        l3 = f(row.get("L3_handoff_transfer_penalty_proxy"))
        handoff = ("HANDOFF_SCALE" in failure) or ("HANDOFF_GAUGE" in failure)
        lowobs_or_multimode = ("LOW_OBSERVABILITY" in failure) or ("MULTIMODE_CONFLICT" in failure)
        l3_is_high = math.isfinite(l3) and l3 >= l3_high
        l3_is_low = math.isfinite(l3) and l3 <= l3_low
        clean_handoff_target = label == "non_good" and handoff and l3_is_high and not lowobs_or_multimode
        diagnostic_handoff_if_unlabelled_allowed = label in {"non_good", "unlabelled_support"} and handoff and l3_is_high
        clean_safe_good = label == "good" and l3_is_low
        has_v100_trace = case_id in same_space_cases
        already = case_id in v100_cases
        extension_usable = (not already) and has_v100_trace and (clean_handoff_target or clean_safe_good)
        reason: list[str] = []
        if already:
            reason.append("already_in_v100_28_case_universe")
        if not has_v100_trace:
            reason.append("missing_v100_same_space_anchor_trace")
        if label == "unlabelled_support":
            reason.append("offline_unlabelled_support_not_clean_non_good_label")
        if lowobs_or_multimode and handoff:
            reason.append("handoff_candidate_has_lowobs_or_multimode_abstain_risk")
        if not (clean_handoff_target or clean_safe_good):
            reason.append("does_not_meet_clean_v101_target_or_safe_good_rule")
        out_rows.append(
            {
                **row,
                "already_in_v100_28_case_universe": already,
                "has_v100_same_space_anchor_trace": has_v100_trace,
                "handoff_failure_mode": handoff,
                "lowobs_or_multimode": lowobs_or_multimode,
                "L3_high_threshold": l3_high,
                "L3_low_threshold": l3_low,
                "is_L3_high": l3_is_high,
                "is_L3_low": l3_is_low,
                "clean_handoff_target_under_v101_rules": clean_handoff_target,
                "diagnostic_handoff_if_unlabelled_allowed": diagnostic_handoff_if_unlabelled_allowed,
                "clean_safe_good_under_v101_rules": clean_safe_good,
                "usable_for_v101_extension": extension_usable,
                "exclusion_reason": ";".join(reason) if reason else "",
            }
        )
    usable = [row for row in out_rows if row["usable_for_v101_extension"]]
    new_without_trace = [row for row in out_rows if (not row["already_in_v100_28_case_universe"]) and (not row["has_v100_same_space_anchor_trace"])]
    new_clean_missing_trace = [
        row
        for row in new_without_trace
        if row["clean_handoff_target_under_v101_rules"] or row["clean_safe_good_under_v101_rules"]
    ]
    diagnostic_handoff = [row for row in out_rows if row["diagnostic_handoff_if_unlabelled_allowed"]]
    summary = {
        "schema": "acl2_v101_trackT_target_extension_feasibility_v1",
        "source_files": [str(V95_CANONICAL_CASES), str(V94_BOUNDARY_ROWS)],
        "old_source_row_count": len(out_rows),
        "unique_old_case_count": len({row["case_id"] for row in out_rows}),
        "v100_case_count": len(v100_cases),
        "v100_same_space_case_count": len(same_space_cases),
        "new_old_case_count": len({row["case_id"] for row in out_rows if not row["already_in_v100_28_case_universe"]}),
        "new_old_case_with_v100_same_space_trace_count": len(
            {row["case_id"] for row in out_rows if (not row["already_in_v100_28_case_universe"]) and row["has_v100_same_space_anchor_trace"]}
        ),
        "usable_extension_case_count": len({row["case_id"] for row in usable}),
        "usable_extension_cases": ";".join(sorted({row["case_id"] for row in usable})),
        "diagnostic_handoff_if_unlabelled_allowed_unique_count": len({row["case_id"] for row in diagnostic_handoff}),
        "diagnostic_handoff_if_unlabelled_allowed_cases": ";".join(sorted({row["case_id"] for row in diagnostic_handoff})),
        "new_case_missing_v100_same_space_trace_count": len({row["case_id"] for row in new_without_trace}),
        "new_case_missing_v100_same_space_trace_cases": ";".join(sorted({row["case_id"] for row in new_without_trace})),
        "new_clean_candidate_missing_v100_trace_count": len({row["case_id"] for row in new_clean_missing_trace}),
        "new_clean_candidate_missing_v100_trace_cases": ";".join(sorted({row["case_id"] for row in new_clean_missing_trace})),
        "trace_extension_run_recommended": bool(new_clean_missing_trace),
        "extension_gate_pass": bool(usable),
        "runtime_action_allowed": False,
        "claim": "No v94/v95 case is promoted into v101 action without v100 same-space trace and clean v101 target/safe-good eligibility.",
    }
    return out_rows, summary


def audit_good_control_extension(v100_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    v100_cases = {row.get("case_id", "") for row in v100_rows}
    same_space_cases = {
        row.get("case_id", "")
        for row in read_rows(V100_SAME_SPACE_ROWS)
        if row.get("canonical_space_name") == "S-B_preprojection_hidden"
    }
    l3_all = [f(row.get("L3_handoff_transfer_penalty_proxy")) for row in v100_rows]
    good_l3 = [f(row.get("L3_handoff_transfer_penalty_proxy")) for row in v100_rows if row.get("case_label") == "good"]
    l3_high = max(median(good_l3) + 2.0 * mad(good_l3), quantile(l3_all, 0.75))
    l3_low = quantile(l3_all, 0.40)
    raw_good = read_rows(V95_GOOD_CONTROLS)
    rolling_values: list[float] = []
    for row in raw_good:
        growth = f(row.get("L4_propagation_growth_rate"))
        future = max(
            [
                f(row.get("L4_future_error_1chunk")),
                f(row.get("L4_future_error_3chunk")),
                f(row.get("L4_future_error_5chunk")),
            ]
        )
        proxy = growth if math.isfinite(growth) else future
        if math.isfinite(proxy):
            rolling_values.append(proxy)
    rolling_low = quantile(rolling_values, 0.40)
    rows: list[dict[str, Any]] = []
    for row in raw_good:
        case_id = str(row.get("case_id", ""))
        l3 = f(row.get("L3_handoff_transfer_penalty_proxy"))
        growth = f(row.get("L4_propagation_growth_rate"))
        future_values = [
            f(row.get("L4_future_error_1chunk")),
            f(row.get("L4_future_error_3chunk")),
            f(row.get("L4_future_error_5chunk")),
        ]
        future = max(future_values)
        rolling_proxy = growth if math.isfinite(growth) else future
        already = case_id in v100_cases
        has_trace = case_id in same_space_cases
        l3_low_pass = math.isfinite(l3) and l3 <= l3_low
        rolling_low_pass = math.isfinite(rolling_proxy) and math.isfinite(rolling_low) and rolling_proxy <= rolling_low
        contaminated = math.isfinite(l3) and l3 >= l3_high
        eligible = l3_low_pass and rolling_low_pass and not contaminated
        usable_existing = eligible and has_trace
        trace_only_candidate = eligible and not has_trace
        reason: list[str] = []
        if not l3_low_pass:
            reason.append("not_low_L3_under_v101_threshold")
        if not rolling_low_pass:
            reason.append("not_low_rolling_worse_proxy")
        if contaminated:
            reason.append("high_L3_contaminated_good")
        if already:
            reason.append("already_in_v100_28_case_universe")
        if not has_trace:
            reason.append("missing_v100_same_space_anchor_trace")
        rows.append(
            {
                "source_name": "v95_trackA_good_controls",
                "source_path": str(V95_GOOD_CONTROLS),
                "case_id": case_id,
                "seq": row.get("seq", case_seq(case_id)),
                "prev_chunk": row.get("prev_chunk", ""),
                "curr_chunk": row.get("curr_chunk", ""),
                "failure_type": ";".join(x for x in [row.get("failure_type_primary", ""), row.get("failure_type_secondary", "")] if x),
                "L3_handoff_transfer_penalty_proxy": row.get("L3_handoff_transfer_penalty_proxy", ""),
                "rolling_worse_proxy": rolling_proxy,
                "L3_low_threshold": l3_low,
                "rolling_low_threshold_q40_v95_good": rolling_low,
                "L3_high_contamination_threshold": l3_high,
                "already_in_v100_28_case_universe": already,
                "has_v100_same_space_anchor_trace": has_trace,
                "low_L3_pass": l3_low_pass,
                "low_rolling_worse_pass": rolling_low_pass,
                "high_L3_contaminated_good": contaminated,
                "eligible_low_L3_low_rolling_good": eligible,
                "usable_existing_safe_good_extension": usable_existing,
                "trace_only_safe_good_candidate": trace_only_candidate,
                "exclusion_reason": ";".join(reason),
            }
        )
    usable = [row for row in rows if row["usable_existing_safe_good_extension"]]
    trace_only = [row for row in rows if row["trace_only_safe_good_candidate"]]
    new_usable = [row for row in usable if not row["already_in_v100_28_case_universe"]]
    summary = {
        "schema": "acl2_v101_trackT_v95_good_control_extension_v1",
        "source_file": str(V95_GOOD_CONTROLS),
        "v95_good_control_row_count": len(rows),
        "v101_L3_low_threshold": l3_low,
        "rolling_low_threshold_q40_v95_good": rolling_low,
        "eligible_low_L3_low_rolling_good_count": len([row for row in rows if row["eligible_low_L3_low_rolling_good"]]),
        "eligible_low_L3_low_rolling_good_cases": ";".join(sorted(row["case_id"] for row in rows if row["eligible_low_L3_low_rolling_good"])),
        "usable_existing_safe_good_extension_count": len(usable),
        "usable_existing_safe_good_extension_cases": ";".join(sorted(row["case_id"] for row in usable)),
        "new_usable_safe_good_extension_count": len(new_usable),
        "new_usable_safe_good_extension_cases": ";".join(sorted(row["case_id"] for row in new_usable)),
        "trace_only_safe_good_candidate_count": len(trace_only),
        "trace_only_safe_good_candidate_cases": ";".join(sorted(row["case_id"] for row in trace_only)),
        "high_L3_contaminated_good_count": len([row for row in rows if row["high_L3_contaminated_good"]]),
        "high_L3_contaminated_good_cases": ";".join(sorted(row["case_id"] for row in rows if row["high_L3_contaminated_good"])),
        "safe_good_extension_gate_pass": bool(new_usable or trace_only),
        "runtime_action_allowed": False,
        "claim": "V95 good controls were audited for low-L3/low-rolling SAFE_GOOD extension; no runtime claim is made.",
    }
    return rows, summary


def build_l3_metric_target_split(v100_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    q_by_case = {row.get("case_id", ""): row for row in read_rows(V100_Q_ROWS)}
    l3_all = [f(row.get("L3_handoff_transfer_penalty_proxy")) for row in v100_rows]
    good_l3 = [f(row.get("L3_handoff_transfer_penalty_proxy")) for row in v100_rows if row.get("case_label") == "good"]
    l3_high = max(median(good_l3) + 2.0 * mad(good_l3), quantile(l3_all, 0.75))
    l3_low = quantile(l3_all, 0.40)
    rows: list[dict[str, Any]] = []
    positives: set[str] = set()
    negatives: set[str] = set()
    selected: set[str] = set()
    for raw in v100_rows:
        case_id = str(raw.get("case_id", ""))
        label = str(raw.get("case_label", ""))
        failure = str(raw.get("failure_type", "")).upper()
        l3 = f(raw.get("L3_handoff_transfer_penalty_proxy"))
        high = math.isfinite(l3) and l3 >= l3_high
        low = math.isfinite(l3) and l3 <= l3_low
        good = label == "good"
        hygiene_excluded = b(raw.get("v98_hygiene_excluded_good_control"))
        handoff = ("HANDOFF_SCALE" in failure) or ("HANDOFF_GAUGE" in failure)
        lowobs_or_multimode = ("LOW_OBSERVABILITY" in failure) or ("MULTIMODE_CONFLICT" in failure)
        q = q_by_case.get(case_id, {})
        q_decision = str(q.get("admission_decision_proxy", ""))
        q_selected = (
            b(q.get("delay_or_no_scale_proxy"))
            or f(q.get("q_composite_delay_or_no_scale_proxy"), 0.0) > 0.0
            or q_decision in {"DELAY_UPDATE", "NO_SCALE_EVIDENCE"}
        )
        target_kind = "EXCLUDED_MID_L3"
        if high:
            target_kind = "L3_HIGH_HARM_TARGET"
            positives.add(case_id)
        elif good and low and not hygiene_excluded:
            target_kind = "SAFE_GOOD_LOW_L3_CONTROL"
            negatives.add(case_id)
        if q_selected:
            selected.add(case_id)
        rows.append(
            {
                **raw,
                "split_target_kind": target_kind,
                "L3_high_threshold": l3_high,
                "L3_low_threshold": l3_low,
                "is_L3_high": high,
                "is_L3_low": low,
                "binary_good_high_L3_conflict": good and high,
                "handoff_failure_mode": handoff,
                "lowobs_or_multimode": lowobs_or_multimode,
                "action_eligible_handoff_target": (not good) and high and handoff and not lowobs_or_multimode,
                "safe_good_low_l3_control": good and low and not hygiene_excluded,
                "q_proxy_selected_delay_or_no_scale": q_selected,
                "q_admission_decision_proxy": q_decision,
                "q_composite_delay_or_no_scale_proxy": q.get("q_composite_delay_or_no_scale_proxy", ""),
                "claim_level": "diagnostic_l3_metric_target_only",
            }
        )
    metrics = selected_metrics(rows, selected, positives, negatives)
    high_rows = [row for row in rows if row["split_target_kind"] == "L3_HIGH_HARM_TARGET"]
    safe_rows = [row for row in rows if row["split_target_kind"] == "SAFE_GOOD_LOW_L3_CONTROL"]
    action_eligible = [row for row in high_rows if row["action_eligible_handoff_target"]]
    good_conflict = [row for row in high_rows if row["binary_good_high_L3_conflict"]]
    lowobs_abstain = [row for row in high_rows if row["lowobs_or_multimode"]]
    diagnostic_ready = len(high_rows) >= 6 and len({row["seq"] for row in high_rows}) >= 3 and len(safe_rows) >= 5
    summary = {
        "schema": "acl2_v101_trackT_l3_metric_target_split_v1",
        "case_count": len(rows),
        "L3_high_threshold": l3_high,
        "L3_low_threshold": l3_low,
        "l3_high_harm_target_count": len(high_rows),
        "l3_high_harm_target_cases": ";".join(sorted(row["case_id"] for row in high_rows)),
        "l3_high_harm_sequence_coverage": len({row["seq"] for row in high_rows}),
        "safe_good_low_l3_control_count": len(safe_rows),
        "safe_good_low_l3_control_cases": ";".join(sorted(row["case_id"] for row in safe_rows)),
        "binary_good_high_l3_conflict_count": len(good_conflict),
        "binary_good_high_l3_conflict_cases": ";".join(sorted(row["case_id"] for row in good_conflict)),
        "lowobs_or_multimode_high_l3_count": len(lowobs_abstain),
        "lowobs_or_multimode_high_l3_cases": ";".join(sorted(row["case_id"] for row in lowobs_abstain)),
        "action_eligible_handoff_target_count": len(action_eligible),
        "action_eligible_handoff_target_cases": ";".join(sorted(row["case_id"] for row in action_eligible)),
        "q_proxy_metrics_on_l3_metric_split": metrics,
        "diagnostic_l3_metric_split_ready": diagnostic_ready,
        "action_gate_pass": False,
        "runtime_action_allowed": False,
        "claim": "L3 metric target split is diagnostic only because high-L3 harm includes good-label conflicts and lowobs/multimode cases.",
    }
    return rows, summary


def main() -> None:
    v100_rows = read_rows(V100_CASE_ROWS)
    old_rows = collect_old_candidates()
    threshold_rows, threshold_summary = build_threshold_variants(v100_rows)
    extension_rows, extension_summary = audit_extension(v100_rows, old_rows)
    good_extension_rows, good_extension_summary = audit_good_control_extension(v100_rows)
    split_rows, split_summary = build_l3_metric_target_split(v100_rows)
    write_rows(TRACK_T / "threshold_variant_audit.csv", threshold_rows)
    write_json(TRACK_T / "threshold_variant_summary.json", threshold_summary)
    write_rows(TRACK_T / "target_extension_candidates.csv", extension_rows)
    write_json(TRACK_T / "target_extension_feasibility_summary.json", extension_summary)
    write_rows(TRACK_T / "v95_good_control_extension_rows.csv", good_extension_rows)
    write_json(TRACK_T / "v95_good_control_extension_summary.json", good_extension_summary)
    write_rows(TRACK_T / "target_split_l3_metric_rows.csv", split_rows)
    write_json(TRACK_T / "target_split_l3_metric_summary.json", split_summary)
    write_text(
        TRACK_T / "target_extension_search_report.md",
        "# Track T Target Extension Search Report\n\n"
        f"- Source rows scanned: {extension_summary['old_source_row_count']}\n"
        f"- Unique old cases: {extension_summary['unique_old_case_count']}\n"
        f"- v100 cases: {extension_summary['v100_case_count']}\n"
        f"- v100 same-space cases: {extension_summary['v100_same_space_case_count']}\n"
        f"- New old cases outside v100: {extension_summary['new_old_case_count']}\n"
        f"- New old cases with v100 same-space trace: {extension_summary['new_old_case_with_v100_same_space_trace_count']}\n"
        f"- New clean candidates missing v100 trace: {extension_summary['new_clean_candidate_missing_v100_trace_count']}\n"
        f"- Usable extension cases: {extension_summary['usable_extension_case_count']}\n"
        f"- Diagnostic handoff cases if unlabelled support were allowed: {extension_summary['diagnostic_handoff_if_unlabelled_allowed_unique_count']}\n\n"
        "Conclusion: v94/v95 contain broader offline boundary candidates, but no new case can be promoted into v101 Track T unless it has v100 S-B same-space anchor rows and clean v101 target/safe-good eligibility. This audit found no usable extension case and no clean new candidate for which a trace-only rerun would repair eligibility.\n",
    )
    write_text(
        TRACK_T / "target_split_l3_metric_report.md",
        "# Track T L3 Metric Target Split Report\n\n"
        f"- L3 high harm targets: {split_summary['l3_high_harm_target_count']}\n"
        f"- L3 high sequence coverage: {split_summary['l3_high_harm_sequence_coverage']}\n"
        f"- Safe good low-L3 controls: {split_summary['safe_good_low_l3_control_count']}\n"
        f"- Binary good/high-L3 conflicts: {split_summary['binary_good_high_l3_conflict_count']}\n"
        f"- Lowobs or multimode high-L3 cases: {split_summary['lowobs_or_multimode_high_l3_count']}\n"
        f"- Action-eligible handoff targets after split: {split_summary['action_eligible_handoff_target_count']}\n"
        f"- Q proxy metrics on split target: `{json.dumps(split_summary['q_proxy_metrics_on_l3_metric_split'], sort_keys=True)}`\n\n"
        "Conclusion: splitting the L3 metric target exposes a diagnostic high-harm set, but it does not repair the action gate because most high-L3 harm is label-conflicted, low-observability/multimode, or not a clean handoff target.\n",
    )
    write_text(
        TRACK_T / "v95_good_control_extension_report.md",
        "# Track T V95 Good-Control Extension Report\n\n"
        f"- V95 good-control rows: {good_extension_summary['v95_good_control_row_count']}\n"
        f"- Eligible low-L3/low-rolling good controls: {good_extension_summary['eligible_low_L3_low_rolling_good_count']}\n"
        f"- Usable existing safe-good extension count: {good_extension_summary['usable_existing_safe_good_extension_count']}\n"
        f"- New usable safe-good extension count: {good_extension_summary['new_usable_safe_good_extension_count']}\n"
        f"- Trace-only safe-good candidates: {good_extension_summary['trace_only_safe_good_candidate_count']}\n"
        f"- High-L3 contaminated good controls: {good_extension_summary['high_L3_contaminated_good_count']}\n\n"
        "Conclusion: the only low-L3/low-rolling V95 good controls are already represented in the v100 28-case universe; no new SAFE_GOOD extension is available.\n",
    )
    write_text(
        TRACK_T / "target_insufficient_report.md",
        "# Track T Target Insufficient Report\n\n"
        "The primary v101 taxonomy remains insufficient. The extension search scanned v94/v95 offline candidates and found no new usable case with v100 same-space anchor trace. Threshold variants were also audited; any variant that forces low-L3 or unlabelled-support handoff cases into target is diagnostic-only and cannot authorize Q2/M4/runtime. V95 good controls were audited for low-L3/low-rolling extension but provided no new SAFE_GOOD case. The L3 metric split exposes a diagnostic harm set but still does not produce enough clean action-eligible handoff targets.\n",
    )
    print(json.dumps(extension_summary, sort_keys=True))


if __name__ == "__main__":
    main()
