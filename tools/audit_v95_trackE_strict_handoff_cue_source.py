#!/usr/bin/env python3
"""Backtrack Track E cue/source evidence against strict L3 handoff labels.

The v95 Track E action surface produced mixed proxy changes but failed the
component-strict handoff gate. This audit follows the plan's fail-forward
direction: pause action, return to cue/source, and check whether any fixed
training-free cue actually explains L3 handoff improvement rather than boundary
or no-op proxy movement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any


V95_ROOT = Path("results/acl2_v95tf_multiroute_semantic_memory_evidence_control")
V94_ROOT = Path("results/acl2_v94tf_semantic_gauge_failure_localization_causal_memory_control")
DEFAULT_EFFECT_ROWS = V95_ROOT / "trackE_alpha04_runtime_probe/runtime_probe_effect_rows.csv"
DEFAULT_SELECTED_ROWS = V94_ROOT / "phase5_object_source_extension/selected_policy_rows.csv"
DEFAULT_PHASE1_ROWS = V94_ROOT / "phase1_boundary_failure_atlas/boundary_failure_rows.csv"
DEFAULT_OUT_DIR = V95_ROOT / "trackE_strict_handoff_cue_source_backtrack"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def f(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def median_or_none(values: list[float]) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value))
    return float(median(finite)) if finite else None


def max_or_none(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(max(finite)) if finite else None


def quantile(values: list[float], q: float) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return float("nan")
    pos = (len(finite) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(finite[lo])
    frac = pos - lo
    return float(finite[lo] * (1.0 - frac) + finite[hi] * frac)


def bool_text(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def stable_key(*parts: Any) -> float:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def rotate(mask: list[bool], amount: int) -> list[bool]:
    if not mask:
        return []
    amount %= len(mask)
    return mask[amount:] + mask[:amount]


def regime_rotate(rows: list[dict[str, Any]], mask: list[bool], amount: int) -> list[bool]:
    out = [False] * len(mask)
    by_seq: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        by_seq.setdefault(str(row.get("seq")), []).append(idx)
    for indices in by_seq.values():
        values = rotate([mask[idx] for idx in indices], amount)
        for idx, value in zip(indices, values):
            out[idx] = value
    return out


def object_order_rotate(rows: list[dict[str, Any]], mask: list[bool], amount: int) -> list[bool]:
    order = sorted(range(len(rows)), key=lambda idx: (f(rows[idx].get("object_boundary_ratio")), idx))
    values = rotate([mask[idx] for idx in order], amount)
    out = [False] * len(mask)
    for idx, value in zip(order, values):
        out[idx] = value
    return out


def same_count_random(rows: list[dict[str, Any]], mask: list[bool]) -> list[bool]:
    out = [False] * len(mask)
    groups: dict[tuple[str, str], list[int]] = {}
    for idx, row in enumerate(rows):
        groups.setdefault((str(row.get("seq")), str(row.get("case_label_offline_only"))), []).append(idx)
    for (seq, label), indices in groups.items():
        count = sum(1 for idx in indices if mask[idx])
        ordered = sorted(indices, key=lambda idx: stable_key("v95_strict_handoff_same_count", seq, label, rows[idx].get("pair_id")))
        for idx in ordered[: min(count, len(ordered))]:
            out[idx] = True
    return out


def not_mask(mask: list[bool]) -> list[bool]:
    return [not value for value in mask]


def and_mask(*masks: list[bool]) -> list[bool]:
    return [all(values) for values in zip(*masks)]


def or_mask(*masks: list[bool]) -> list[bool]:
    return [any(values) for values in zip(*masks)]


def build_pair_rows(
    effect_rows: list[dict[str, str]],
    selected_rows: list[dict[str, str]],
    phase1_rows: list[dict[str, str]],
    *,
    bad_threshold: float,
    boundary_proxy_threshold: float,
    boundary_component_threshold: float,
) -> list[dict[str, Any]]:
    selected_by_pair = {row.get("pair_id"): row for row in selected_rows}
    phase1_by_pair = {row.get("pair_id"): row for row in phase1_rows}
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in effect_rows:
        grouped.setdefault(str(row.get("pair_id")), []).append(row)

    out: list[dict[str, Any]] = []
    for pair_id, rows in sorted(grouped.items()):
        first = rows[0]
        label = str(first.get("case_label_offline_only"))
        handoff_values = [(str(row.get("variant")), f(row.get("handoff_transfer_improvement_ratio"))) for row in rows]
        boundary_values = [(str(row.get("variant")), f(row.get("boundary_improvement_ratio"))) for row in rows]
        proxy_values = [(str(row.get("variant")), f(row.get("I_J_runtime_proxy"))) for row in rows]
        best_handoff_variant, max_handoff = max(
            handoff_values,
            key=lambda item: item[1] if math.isfinite(item[1]) else -float("inf"),
        )
        best_boundary_variant, max_boundary = max(
            boundary_values,
            key=lambda item: item[1] if math.isfinite(item[1]) else -float("inf"),
        )
        best_proxy_variant, max_proxy = max(
            proxy_values,
            key=lambda item: item[1] if math.isfinite(item[1]) else -float("inf"),
        )
        good_worsen_values = [-value for _, value in handoff_values if math.isfinite(value)]
        row: dict[str, Any] = {
            "pair_id": pair_id,
            "seq": first.get("seq"),
            "prev_chunk": first.get("prev_chunk"),
            "curr_chunk": first.get("curr_chunk"),
            "case_label_offline_only": label,
            "failure_type_primary": first.get("failure_type_primary"),
            "failure_type_secondary": first.get("failure_type_secondary"),
            "probe_selection_tag": first.get("probe_selection_tag"),
            "max_handoff_improvement": max_handoff,
            "best_handoff_variant": best_handoff_variant,
            "max_boundary_improvement": max_boundary,
            "best_boundary_variant": best_boundary_variant,
            "max_proxy_improvement": max_proxy,
            "best_proxy_variant": best_proxy_variant,
            "strict_handoff_positive": label == "bad" and math.isfinite(max_handoff) and max_handoff >= bad_threshold,
            "handoff_near_miss_ge_0p01": label == "bad" and math.isfinite(max_handoff) and max_handoff >= 0.01,
            "boundary_only_warning_pair": (
                math.isfinite(max_proxy)
                and math.isfinite(max_boundary)
                and math.isfinite(max_handoff)
                and max_proxy >= boundary_proxy_threshold
                and max_boundary >= boundary_component_threshold
                and max_handoff < bad_threshold
            ),
            "good_handoff_worst_worsen": max_or_none(good_worsen_values) if label == "good" else None,
        }
        for meta in (selected_by_pair.get(pair_id, {}), phase1_by_pair.get(pair_id, {})):
            for key, value in meta.items():
                row.setdefault(key, value)
        # Keep selected-row values authoritative for cue atoms when both tables share a name.
        for key, value in selected_by_pair.get(pair_id, {}).items():
            row[key] = value
        out.append(row)
    return out


def build_atoms(rows: list[dict[str, Any]]) -> tuple[dict[str, list[bool]], dict[str, list[bool]], dict[str, float]]:
    threshold_columns = [
        "boundary_global_cross_ratio",
        "boundary_new_id_ratio",
        "object_boundary_ratio",
        "radio_boundary_mean",
        "component_boundary_ratio",
        "cross_component_ratio",
        "dynamic_or_transient_ratio",
        "semantic_mode_entropy",
        "observability_score",
        "J_handoff",
        "future_after_overlap",
        "boundary_jump",
        "adjacent_log_scale_jump_offline",
        "carrier_error_merge_residual_after_abs",
        "carrier_error_abs_log_scale_jump_runtime",
    ]
    thresholds: dict[str, float] = {}
    for column in threshold_columns:
        values = [f(row.get(column)) for row in rows]
        for q in [0.50, 0.60, 0.75, 0.85, 0.90]:
            thresholds[f"{column}_q{int(q * 100)}"] = quantile(values, q)
        if column == "observability_score":
            for q in [0.10, 0.25, 0.40]:
                thresholds[f"{column}_q{int(q * 100)}"] = quantile(values, q)

    semantic_roles = sorted({str(row.get("semantic_evidence_type")) for row in rows if row.get("semantic_evidence_type")})
    method_atoms: dict[str, list[bool]] = {
        "OBJECT_SOURCE_POLICY": [bool_text(row.get("selected_policy_positive")) for row in rows],
        "SAME_GLOBAL_OBJECT_FALSE": [str(row.get("same_global_object_id")).lower() == "false" for row in rows],
    }
    for role in semantic_roles:
        method_atoms[role] = [str(row.get("semantic_evidence_type")) == role for row in rows]
    for column in [
        "boundary_global_cross_ratio",
        "boundary_new_id_ratio",
        "object_boundary_ratio",
        "radio_boundary_mean",
        "component_boundary_ratio",
        "cross_component_ratio",
        "dynamic_or_transient_ratio",
        "semantic_mode_entropy",
    ]:
        for q in [50, 60, 75, 85, 90]:
            thresh = thresholds[f"{column}_q{q}"]
            method_atoms[f"{column.upper()}_GE_Q{q}"] = [f(row.get(column)) >= thresh for row in rows]
    for q in [10, 25, 40]:
        thresh = thresholds[f"observability_score_q{q}"]
        method_atoms[f"OBSERVABILITY_SCORE_LE_Q{q}"] = [f(row.get("observability_score")) <= thresh for row in rows]

    diagnostic_atoms = dict(method_atoms)
    for column in [
        "J_handoff",
        "future_after_overlap",
        "boundary_jump",
        "adjacent_log_scale_jump_offline",
        "carrier_error_merge_residual_after_abs",
        "carrier_error_abs_log_scale_jump_runtime",
    ]:
        for q in [50, 60, 75, 85, 90]:
            thresh = thresholds[f"{column}_q{q}"]
            diagnostic_atoms[f"{column.upper()}_GE_Q{q}"] = [f(row.get(column)) >= thresh for row in rows]
    for failure in sorted({str(row.get("failure_type_primary")) for row in rows if row.get("failure_type_primary")}):
        diagnostic_atoms[f"FAILURE_{failure}"] = [str(row.get("failure_type_primary")) == failure for row in rows]
    diagnostic_atoms["LABEL_BAD_UPPER_BOUND"] = [row.get("case_label_offline_only") == "bad" for row in rows]
    diagnostic_atoms["BOUNDARY_ONLY_WARNING_OBSERVED"] = [bool(row.get("boundary_only_warning_pair")) for row in rows]
    diagnostic_atoms["HANDOFF_NEAR_MISS_GE_0P01"] = [bool(row.get("handoff_near_miss_ge_0p01")) for row in rows]
    return method_atoms, diagnostic_atoms, thresholds


def candidate_masks(atoms: dict[str, list[bool]], max_or_terms: int = 2) -> dict[str, list[bool]]:
    candidates: dict[str, list[bool]] = {}
    names = sorted(atoms)
    for name in names:
        candidates[name] = atoms[name]
        if "SEM_MULTIMODE_UNSAFE" in atoms and name != "SEM_MULTIMODE_UNSAFE":
            candidates[f"{name}__AND_NOT_SEM_MULTIMODE"] = and_mask(atoms[name], not_mask(atoms["SEM_MULTIMODE_UNSAFE"]))
        if "SEM_LOWOBS_ABSTAIN" in atoms and name != "SEM_LOWOBS_ABSTAIN":
            candidates[f"{name}__AND_NOT_SEM_LOWOBS"] = and_mask(atoms[name], not_mask(atoms["SEM_LOWOBS_ABSTAIN"]))
    for term_count in range(2, max_or_terms + 1):
        for combo in combinations(names, term_count):
            candidates["_OR_".join(combo)] = or_mask(*(atoms[item] for item in combo))
    return {name: mask for name, mask in candidates.items() if any(mask)}


def cue_metric(rows: list[dict[str, Any]], mask: list[bool], policy: str, scope: str) -> dict[str, Any]:
    selected = [idx for idx, value in enumerate(mask) if value]
    bad_total = sum(1 for row in rows if row.get("case_label_offline_only") == "bad")
    good_total = sum(1 for row in rows if row.get("case_label_offline_only") == "good")
    positive_total = sum(1 for row in rows if bool(row.get("strict_handoff_positive")))
    boundary_total = sum(1 for row in rows if bool(row.get("boundary_only_warning_pair")))
    selected_bad = [idx for idx in selected if rows[idx].get("case_label_offline_only") == "bad"]
    selected_good = [idx for idx in selected if rows[idx].get("case_label_offline_only") == "good"]
    selected_positive = [idx for idx in selected if bool(rows[idx].get("strict_handoff_positive"))]
    selected_boundary = [idx for idx in selected if bool(rows[idx].get("boundary_only_warning_pair"))]
    selected_bad_handoff = [f(rows[idx].get("max_handoff_improvement")) for idx in selected_bad]
    return {
        "policy": policy,
        "scope": scope,
        "selected_pair_count": len(selected),
        "selected_bad_pairs": len(selected_bad),
        "selected_good_pairs": len(selected_good),
        "strict_handoff_positive_hits": len(selected_positive),
        "strict_handoff_positive_total": positive_total,
        "strict_handoff_recall": (len(selected_positive) / positive_total) if positive_total else None,
        "bad_recall_on_labelled": len(selected_bad) / max(bad_total, 1),
        "good_FPR_on_labelled": len(selected_good) / max(good_total, 1),
        "boundary_only_hits": len(selected_boundary),
        "boundary_only_total": boundary_total,
        "boundary_only_recall": (len(selected_boundary) / boundary_total) if boundary_total else None,
        "selected_bad_median_max_handoff_improvement": median_or_none(selected_bad_handoff),
        "selected_bad_max_handoff_improvement": max_or_none(selected_bad_handoff),
        "sequence_coverage": len({rows[idx].get("seq") for idx in selected}),
        "selected_pairs": ",".join(str(rows[idx].get("pair_id")) for idx in selected),
    }


def evaluate_candidate(rows: list[dict[str, Any]], mask: list[bool], policy: str, scope: str) -> dict[str, Any]:
    actual = cue_metric(rows, mask, policy, scope)
    controls = [
        cue_metric(rows, same_count_random(rows, mask), f"{policy}__CONTROL_same_count_random", scope),
        cue_metric(rows, object_order_rotate(rows, mask, 5), f"{policy}__CONTROL_object_order_rot5", scope),
        cue_metric(rows, regime_rotate(rows, mask, 1), f"{policy}__CONTROL_regime_rot1", scope),
        cue_metric(rows, regime_rotate(rows, mask, 2), f"{policy}__CONTROL_regime_rot2", scope),
        cue_metric(rows, rotate(mask, 7), f"{policy}__CONTROL_rot7", scope),
    ]
    best_control = max(
        [
            row["selected_bad_median_max_handoff_improvement"]
            for row in controls
            if row["selected_bad_median_max_handoff_improvement"] is not None
        ],
        default=None,
    )
    actual_bad = actual.get("selected_bad_median_max_handoff_improvement")
    actual_minus_best = (
        float(actual_bad - best_control) if actual_bad is not None and best_control is not None else None
    )
    positive_total = int(actual.get("strict_handoff_positive_total") or 0)
    recall = actual.get("strict_handoff_recall")
    good_fpr = float(actual.get("good_FPR_on_labelled") or 0.0)
    median_bad = actual.get("selected_bad_median_max_handoff_improvement")
    gates = {
        "strict_positive_exists_gate": positive_total > 0,
        "strict_handoff_recall_ge_0p60": recall is not None and recall >= 0.60,
        "good_FPR_le_0p25": good_fpr <= 0.25,
        "selected_bad_handoff_median_ge_0p05": median_bad is not None and median_bad >= 0.05,
        "sequence_coverage_ge_3": int(actual.get("sequence_coverage") or 0) >= 3,
        "beats_same_count_controls_ge_0p05": actual_minus_best is not None and actual_minus_best >= 0.05,
    }
    return {
        **actual,
        "best_control_bad_median_max_handoff_improvement": best_control,
        "actual_minus_best_control": actual_minus_best,
        **gates,
        "candidate_gate_pass": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effect-rows", type=Path, default=DEFAULT_EFFECT_ROWS)
    parser.add_argument("--selected-rows", type=Path, default=DEFAULT_SELECTED_ROWS)
    parser.add_argument("--phase1-rows", type=Path, default=DEFAULT_PHASE1_ROWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bad-threshold", type=float, default=0.05)
    parser.add_argument("--boundary-proxy-threshold", type=float, default=0.05)
    parser.add_argument("--boundary-component-threshold", type=float, default=0.25)
    args = parser.parse_args()

    pair_rows = build_pair_rows(
        read_csv(args.effect_rows),
        read_csv(args.selected_rows),
        read_csv(args.phase1_rows),
        bad_threshold=args.bad_threshold,
        boundary_proxy_threshold=args.boundary_proxy_threshold,
        boundary_component_threshold=args.boundary_component_threshold,
    )
    method_atoms, diagnostic_atoms, thresholds = build_atoms(pair_rows)
    method_candidates = candidate_masks(method_atoms)
    diagnostic_candidates = candidate_masks(diagnostic_atoms)

    metrics: list[dict[str, Any]] = []
    for name, mask in method_candidates.items():
        metrics.append(evaluate_candidate(pair_rows, mask, name, "method_safe"))
    for name, mask in diagnostic_candidates.items():
        metrics.append(evaluate_candidate(pair_rows, mask, name, "diagnostic_only"))
    metrics.sort(
        key=lambda row: (
            bool(row.get("candidate_gate_pass")),
            f(row.get("actual_minus_best_control")),
            f(row.get("selected_bad_median_max_handoff_improvement")),
            f(row.get("boundary_only_recall"),),
            -f(row.get("good_FPR_on_labelled")),
        ),
        reverse=True,
    )
    method_rows = [row for row in metrics if row.get("scope") == "method_safe"]
    diagnostic_rows = [row for row in metrics if row.get("scope") == "diagnostic_only"]
    method_passing = [row for row in method_rows if row.get("candidate_gate_pass")]
    diagnostic_passing = [row for row in diagnostic_rows if row.get("candidate_gate_pass")]
    best_method = method_rows[0] if method_rows else {}
    best_diag = diagnostic_rows[0] if diagnostic_rows else {}

    strict_positive_pairs = [row for row in pair_rows if row.get("strict_handoff_positive")]
    near_miss_pairs = [row for row in pair_rows if row.get("handoff_near_miss_ge_0p01")]
    boundary_only_pairs = [row for row in pair_rows if row.get("boundary_only_warning_pair")]
    best_pair = max(pair_rows, key=lambda row: f(row.get("max_handoff_improvement")), default={})
    summary = {
        "phase": "v95_trackE_strict_handoff_cue_source_backtrack",
        "effect_rows": str(args.effect_rows),
        "selected_rows": str(args.selected_rows),
        "phase1_rows": str(args.phase1_rows),
        "pair_count": len(pair_rows),
        "bad_pair_count": sum(1 for row in pair_rows if row.get("case_label_offline_only") == "bad"),
        "good_pair_count": sum(1 for row in pair_rows if row.get("case_label_offline_only") == "good"),
        "strict_handoff_positive_pair_count": len(strict_positive_pairs),
        "handoff_near_miss_ge_0p01_pair_count": len(near_miss_pairs),
        "boundary_only_warning_pair_count": len(boundary_only_pairs),
        "best_pair": best_pair,
        "method_candidate_count": len(method_rows),
        "diagnostic_candidate_count": len(diagnostic_rows),
        "method_safe_passing_count": len(method_passing),
        "diagnostic_only_passing_count": len(diagnostic_passing),
        "best_method_safe": best_method,
        "best_diagnostic_only": best_diag,
        "thresholds": {
            "bad_handoff_threshold": args.bad_threshold,
            "boundary_proxy_threshold": args.boundary_proxy_threshold,
            "boundary_component_threshold": args.boundary_component_threshold,
            **thresholds,
        },
        "runtime_action_allowed": False,
        "blocker": (
            "no_bad_pair_reaches_5pct_handoff_improvement_under_any_variant"
            if not strict_positive_pairs
            else "no_method_safe_cue_selects_strict_handoff_positive_pairs_with_good_protection"
        ),
        "recommended_next_route": (
            "pause_trackE_action_return_to_trackG_or_trackC_swa_cue_source"
            if not method_passing
            else "manual_review_before_any_runtime_action"
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "pair_handoff_source_rows.csv", pair_rows)
    write_csv(args.out_dir / "candidate_cue_source_metrics.csv", metrics)
    write_csv(args.out_dir / "method_safe_candidate_metrics.csv", method_rows)
    write_csv(args.out_dir / "diagnostic_only_candidate_metrics.csv", diagnostic_rows)
    write_json(args.out_dir / "summary.json", summary)
    write_text(
        args.out_dir / "analysis.md",
        f"""
# Track E Strict Handoff Cue/Source Backtrack

- pair_count: `{len(pair_rows)}`
- bad_pair_count: `{summary['bad_pair_count']}`
- good_pair_count: `{summary['good_pair_count']}`
- strict_handoff_positive_pair_count: `{len(strict_positive_pairs)}`
- handoff_near_miss_ge_0p01_pair_count: `{len(near_miss_pairs)}`
- boundary_only_warning_pair_count: `{len(boundary_only_pairs)}`
- best_pair: `{best_pair.get('pair_id')}`
- best_pair_max_handoff_improvement: `{best_pair.get('max_handoff_improvement')}`
- best_pair_variant: `{best_pair.get('best_handoff_variant')}`
- method_candidate_count: `{len(method_rows)}`
- diagnostic_candidate_count: `{len(diagnostic_rows)}`
- method_safe_passing_count: `{len(method_passing)}`
- diagnostic_only_passing_count: `{len(diagnostic_passing)}`
- blocker: `{summary['blocker']}`

Interpretation: this audit is deliberately stricter than the earlier mixed-proxy
cue refinement. A cue cannot promote Track E action unless at least one bad pair
actually reaches the plan's 5% L3 handoff improvement threshold under the measured
action surface. Boundary-only proxy movement is reported separately as diagnostic
evidence, not as runtime action permission.
""",
    )
    write_text(
        args.out_dir / "next_route_recommendation.md",
        f"""
# Next Route Recommendation

Recommended route: `{summary['recommended_next_route']}`.

Reason: `{summary['blocker']}`.

Do not continue direct Track E alpha/off/max-points sweeps until a Track G/C-SWA
cue/source can explain actual L3 handoff improvement rather than boundary-only
proxy movement.
""",
    )

    print(f"pair_count={summary['pair_count']}")
    print(f"strict_handoff_positive_pair_count={summary['strict_handoff_positive_pair_count']}")
    print(f"handoff_near_miss_ge_0p01_pair_count={summary['handoff_near_miss_ge_0p01_pair_count']}")
    print(f"boundary_only_warning_pair_count={summary['boundary_only_warning_pair_count']}")
    print(f"method_candidate_count={summary['method_candidate_count']}")
    print(f"diagnostic_candidate_count={summary['diagnostic_candidate_count']}")
    print(f"method_safe_passing_count={summary['method_safe_passing_count']}")
    print(f"diagnostic_only_passing_count={summary['diagnostic_only_passing_count']}")
    print(f"best_pair={best_pair.get('pair_id')}")
    print(f"best_pair_max_handoff_improvement={best_pair.get('max_handoff_improvement')}")
    print(f"blocker={summary['blocker']}")
    print(f"recommended_next_route={summary['recommended_next_route']}")


if __name__ == "__main__":
    main()
