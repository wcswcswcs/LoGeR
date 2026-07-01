#!/usr/bin/env python3
"""Retrospective rich selector screen for v101 Outcome-D merge/gauge re-entry.

This script is deliberately read-only over existing v94 measured action rows.
It searches a fixed whitelist of pre-action geometry / merge-gauge /
observability fields, but the screen is still retrospective because the policy
family is evaluated after v94 outcomes are known.  It therefore cannot
authorize runtime action.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from statistics import median
from typing import Any


V101_ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
V94_ROOT = Path("results/acl2_v94tf_semantic_gauge_failure_localization_causal_memory_control")
OUT = V101_ROOT / "final_decision"

ACTION_ROWS = V94_ROOT / "phase6_object_source_action_surface" / "action_surface_effect_rows.csv"
CARRIER_ROWS = V94_ROOT / "phase5_semantic_carrier_alignment" / "semantic_carrier_alignment_rows.csv"
BOUNDARY_ROWS = V94_ROOT / "phase1_boundary_failure_atlas" / "boundary_failure_rows.csv"

CARRIER_FEATURES = [
    "carrier_error_boundary_update_norm",
    "carrier_error_merge_residual_after_abs",
    "carrier_error_abs_log_scale_jump_runtime",
    "carrier_error_composite_z",
]

BOUNDARY_FEATURES = [
    "overlap_scale_residual",
    "raw_overlap_residual",
    "confidence_weighted_overlap_residual",
    "adjacent_log_scale_jump_offline",
    "adjacent_gauge_jump_proxy",
    "baseline_proxy",
    "median_depth_proxy",
    "baseline_over_depth",
    "verified_match_count",
    "raw_overlap_inlier_count",
    "local_shape_mode_entropy",
    "local_shape_mode_mad",
    "observability_score",
]

FEATURES = [
    "native_curr_postmerge_sim3_rmse",
    "native_curr_handoff_transfer_rmse",
    "native_abs_log_scale_jump_runtime",
    "native_boundary_update_norm",
    "carrier_state_delta",
    "native_merge_residual_after_abs",
    *CARRIER_FEATURES,
    *[f"boundary_{name}" for name in BOUNDARY_FEATURES],
]

QUANTILES = [0.25, 0.40, 0.50, 0.60, 0.75, 0.90]
RANDOM_REPEATS = 512
RANDOM_SEED = 10194
PROMOTION_MIN_BA = 0.65
PROMOTION_MIN_MARGIN = 0.005
RANDOM_CONTROL_CACHE: dict[tuple[tuple[str, ...], int], dict[str, Any]] = {}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fnum(value: Any) -> float | None:
    try:
        out = float(str(value))
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def quantile(values: list[float], q: float) -> float:
    vals = sorted(values)
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def join_rows() -> list[dict[str, Any]]:
    carrier_by_pair = {row["pair_id"]: row for row in read_csv(CARRIER_ROWS)}
    boundary_by_pair = {row["pair_id"]: row for row in read_csv(BOUNDARY_ROWS)}
    rows: list[dict[str, Any]] = []
    for row in read_csv(ACTION_ROWS):
        out: dict[str, Any] = dict(row)
        carrier = carrier_by_pair.get(row["pair_id"], {})
        boundary = boundary_by_pair.get(row["pair_id"], {})
        for key in CARRIER_FEATURES:
            out[key] = carrier.get(key, "")
        for key in BOUNDARY_FEATURES:
            out[f"boundary_{key}"] = boundary.get(key, "")
        rows.append(out)
    return rows


def valid_features(rows: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for feature in FEATURES:
        vals = [fnum(row.get(feature)) for row in rows]
        finite = [val for val in vals if val is not None]
        if len(finite) >= 12 and len(set(finite)) >= 3:
            out.append(feature)
    return out


def random_controls(rows: list[dict[str, Any]], selected_count: int) -> dict[str, Any]:
    cache_key = (tuple(row.get("pair_id", "") for row in rows), selected_count)
    if cache_key in RANDOM_CONTROL_CACHE:
        return RANDOM_CONTROL_CACHE[cache_key]
    if selected_count <= 0:
        out = {
            "same_count_random_repeats": RANDOM_REPEATS,
            "same_count_random_bad_median_I_J_p50": "",
            "same_count_random_bad_median_I_J_p95": "",
            "same_count_random_BA_p95": "",
        }
        RANDOM_CONTROL_CACHE[cache_key] = out
        return out
    rng = random.Random(RANDOM_SEED + selected_count)
    bad_total = sum(1 for row in rows if row.get("case_label_offline_only") == "bad")
    good_total = sum(1 for row in rows if row.get("case_label_offline_only") == "good")
    bad_medians: list[float] = []
    bas: list[float] = []
    for _ in range(RANDOM_REPEATS):
        sample = rng.sample(rows, min(selected_count, len(rows)))
        bad = [row for row in sample if row.get("case_label_offline_only") == "bad"]
        good = [row for row in sample if row.get("case_label_offline_only") == "good"]
        vals = [fnum(row.get("I_J_runtime_proxy")) for row in bad]
        vals = [val for val in vals if val is not None]
        if vals:
            bad_medians.append(median(vals))
        bas.append(((len(bad) / bad_total if bad_total else 0.0) + (1.0 - (len(good) / good_total if good_total else 0.0))) / 2.0)
    out = {
        "same_count_random_repeats": RANDOM_REPEATS,
        "same_count_random_bad_median_I_J_p50": quantile(bad_medians, 0.50) if bad_medians else "",
        "same_count_random_bad_median_I_J_p95": quantile(bad_medians, 0.95) if bad_medians else "",
        "same_count_random_BA_p95": quantile(bas, 0.95) if bas else "",
    }
    RANDOM_CONTROL_CACHE[cache_key] = out
    return out


def selected_stats(rows: list[dict[str, Any]], selected: list[dict[str, Any]]) -> dict[str, Any]:
    bad_total = sum(1 for row in rows if row.get("case_label_offline_only") == "bad")
    good_total = sum(1 for row in rows if row.get("case_label_offline_only") == "good")
    bad_rows = [row for row in selected if row.get("case_label_offline_only") == "bad"]
    good_rows = [row for row in selected if row.get("case_label_offline_only") == "good"]
    bad_ij = [fnum(row.get("I_J_runtime_proxy")) for row in bad_rows]
    bad_ij = [val for val in bad_ij if val is not None]
    good_w = [fnum(row.get("W_good_runtime_proxy")) for row in good_rows]
    good_w = [val for val in good_w if val is not None]
    bad_median = median(bad_ij) if bad_ij else None
    controls = random_controls(rows, len(selected))
    rand_p95 = controls["same_count_random_bad_median_I_J_p95"]
    margin = ""
    if bad_median is not None and rand_p95 != "":
        margin = float(bad_median) - float(rand_p95)
    good_max = max(good_w) if good_w else None
    good_gate = good_max is None or float(good_max) <= 0.02
    bad_negative = sum(1 for val in bad_ij if val < 0.0)
    stats: dict[str, Any] = {
        "selected_row_count": len(selected),
        "bad_rows": len(bad_rows),
        "good_rows": len(good_rows),
        "bad_recall": len(bad_rows) / bad_total if bad_total else 0.0,
        "good_FPR": len(good_rows) / good_total if good_total else 0.0,
        "balanced_accuracy": ((len(bad_rows) / bad_total if bad_total else 0.0) + (1.0 - (len(good_rows) / good_total if good_total else 0.0))) / 2.0,
        "bad_sequence_coverage": len({row.get("seq", "") for row in bad_rows}),
        "selected_sequence_coverage": len({row.get("seq", "") for row in selected}),
        "bad_median_I_J_runtime_proxy": bad_median if bad_median is not None else "",
        "bad_min_I_J_runtime_proxy": min(bad_ij) if bad_ij else "",
        "bad_negative_improvement_rows": bad_negative,
        "good_max_worsen_runtime_proxy": good_max if good_max is not None else "",
        "good_worsen_gt_0p02_rows": sum(1 for val in good_w if val > 0.02),
        "bad_hits": ",".join(row.get("pair_id", "") for row in bad_rows),
        "good_hits": ",".join(row.get("pair_id", "") for row in good_rows),
        "selected_signature": ",".join(sorted(row.get("pair_id", "") for row in selected)),
        **controls,
        "bad_median_margin_vs_same_count_random_p95": margin,
    }
    pass_gate = (
        stats["bad_rows"] >= 3
        and stats["bad_sequence_coverage"] >= 3
        and stats["good_FPR"] <= 0.25
        and bad_median is not None
        and float(bad_median) >= 0.05
        and bad_negative == 0
        and good_gate
        and margin != ""
        and float(margin) > 0.0
    )
    stats["retrospective_selector_gate_pass"] = pass_gate
    promotion_floor_pass = (
        pass_gate
        and float(stats["balanced_accuracy"]) >= PROMOTION_MIN_BA
        and margin != ""
        and float(margin) >= PROMOTION_MIN_MARGIN
    )
    stats["promotion_floor_pass"] = promotion_floor_pass
    stats["action_authorized"] = False
    stats["action_block_reason"] = "retrospective_existing_rows_no_fresh_predeclared_holdout"
    if pass_gate and not promotion_floor_pass:
        stats["action_block_reason"] = "weak_retrospective_pass_low_BA_or_tiny_margin_requires_replication"
    stats["claim_level"] = "retrospective_selector_screen_no_runtime"
    return stats


def mask_for(row: dict[str, Any], feature: str, direction: str, threshold: float) -> bool:
    value = fnum(row.get(feature))
    if value is None:
        return False
    if direction == "ge":
        return value >= threshold
    return value <= threshold


def evaluate(
    rows: list[dict[str, Any]],
    *,
    family: str,
    feature1: str,
    direction1: str,
    threshold1: float,
    quantile1: float,
    feature2: str = "",
    direction2: str = "",
    threshold2: float | None = None,
    quantile2: float | None = None,
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        keep = mask_for(row, feature1, direction1, threshold1)
        if keep and feature2:
            assert threshold2 is not None
            keep = mask_for(row, feature2, direction2, threshold2)
        if keep:
            selected.append(row)
    policy_id = f"{feature1}_{direction1}_q{int(quantile1 * 100):02d}"
    if feature2:
        policy_id += f"__AND__{feature2}_{direction2}_q{int((quantile2 or 0.0) * 100):02d}"
    return {
        "policy_id": policy_id,
        "family": family,
        "feature1": feature1,
        "direction1": direction1,
        "threshold1": threshold1,
        "quantile1": quantile1,
        "feature2": feature2,
        "direction2": direction2,
        "threshold2": threshold2 if threshold2 is not None else "",
        "quantile2": quantile2 if quantile2 is not None else "",
        **selected_stats(rows, selected),
    }


def build_candidates(rows: list[dict[str, Any]], features: list[str]) -> tuple[list[dict[str, Any]], int]:
    atoms: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for feature in features:
        vals = [fnum(row.get(feature)) for row in rows]
        vals = [val for val in vals if val is not None]
        for q in QUANTILES:
            threshold = quantile(vals, q)
            for direction in ("ge", "le"):
                row = evaluate(
                    rows,
                    family="1d",
                    feature1=feature,
                    direction1=direction,
                    threshold1=threshold,
                    quantile1=q,
                )
                atoms.append(row)
                candidates.append(row)
    for i, left in enumerate(atoms):
        for right in atoms[i + 1 :]:
            if left["feature1"] == right["feature1"]:
                continue
            candidates.append(
                evaluate(
                    rows,
                    family="2d_and",
                    feature1=left["feature1"],
                    direction1=left["direction1"],
                    threshold1=float(left["threshold1"]),
                    quantile1=float(left["quantile1"]),
                    feature2=right["feature1"],
                    direction2=right["direction1"],
                    threshold2=float(right["threshold1"]),
                    quantile2=float(right["quantile1"]),
                )
            )
    candidates.sort(
        key=lambda row: (
            not bool(row["retrospective_selector_gate_pass"]),
            -float(row["balanced_accuracy"]),
            -float(row["bad_rows"]),
            float(row["good_rows"]),
            -float(row["bad_median_margin_vs_same_count_random_p95"] or -1e99),
            -float(row["bad_median_I_J_runtime_proxy"] or -1e99),
        )
    )
    return candidates, len(atoms)


def selected_rows_for_policy(rows: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        keep = mask_for(row, policy["feature1"], policy["direction1"], float(policy["threshold1"]))
        if keep and policy.get("feature2"):
            keep = mask_for(row, policy["feature2"], policy["direction2"], float(policy["threshold2"]))
        if keep:
            selected.append(
                {
                    "policy_id": policy["policy_id"],
                    "pair_id": row.get("pair_id", ""),
                    "seq": row.get("seq", ""),
                    "label": row.get("case_label_offline_only", ""),
                    "failure_type_primary": row.get("failure_type_primary", ""),
                    "feature1": policy["feature1"],
                    "feature1_value": row.get(policy["feature1"], ""),
                    "feature2": policy.get("feature2", ""),
                    "feature2_value": row.get(policy.get("feature2", ""), "") if policy.get("feature2") else "",
                    "I_J_runtime_proxy": row.get("I_J_runtime_proxy", ""),
                    "W_good_runtime_proxy": row.get("W_good_runtime_proxy", ""),
                }
            )
    return selected


def selected_source_rows_for_policy(rows: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        keep = mask_for(row, policy["feature1"], policy["direction1"], float(policy["threshold1"]))
        if keep and policy.get("feature2"):
            keep = mask_for(row, policy["feature2"], policy["direction2"], float(policy["threshold2"]))
        if keep:
            selected.append(row)
    return selected


def build_sequence_stability_rows(rows: list[dict[str, Any]], policy: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not policy:
        return [], {}
    selected = selected_source_rows_for_policy(rows, policy)
    selected_pair_ids = {row.get("pair_id", "") for row in selected}
    all_sequences = sorted({row.get("seq", "") for row in rows})
    selected_sequences = sorted({row.get("seq", "") for row in selected})
    out: list[dict[str, Any]] = []
    for seq in all_sequences:
        seq_rows = [row for row in rows if row.get("seq", "") == seq]
        seq_selected = [row for row in selected if row.get("seq", "") == seq]
        bad_vals = [
            fnum(row.get("I_J_runtime_proxy"))
            for row in seq_selected
            if row.get("case_label_offline_only") == "bad"
        ]
        bad_vals = [val for val in bad_vals if val is not None]
        good_vals = [
            fnum(row.get("W_good_runtime_proxy"))
            for row in seq_selected
            if row.get("case_label_offline_only") == "good"
        ]
        good_vals = [val for val in good_vals if val is not None]
        out.append(
            {
                "row_kind": "per_sequence_best_policy",
                "policy_id": policy.get("policy_id", ""),
                "sequence": seq,
                "sequence_total_rows": len(seq_rows),
                "sequence_bad_rows": sum(1 for row in seq_rows if row.get("case_label_offline_only") == "bad"),
                "sequence_good_rows": sum(1 for row in seq_rows if row.get("case_label_offline_only") == "good"),
                "selected_row_count": len(seq_selected),
                "selected_bad_rows": sum(
                    1 for row in seq_selected if row.get("case_label_offline_only") == "bad"
                ),
                "selected_good_rows": sum(
                    1 for row in seq_selected if row.get("case_label_offline_only") == "good"
                ),
                "selected_bad_median_I_J": median(bad_vals) if bad_vals else "",
                "selected_good_max_worsen": max(good_vals) if good_vals else "",
                "selected_pair_ids": ",".join(row.get("pair_id", "") for row in seq_selected),
                "claim_level": "sequence_stability_no_runtime",
            }
        )
    leaveout_pass_count = 0
    min_remaining_bad_seq = ""
    remaining_bad_seq_values: list[int] = []
    for seq in selected_sequences:
        remaining = [row for row in selected if row.get("seq", "") != seq]
        stats = selected_stats(rows, remaining)
        leaveout_pass_count += 1 if stats["retrospective_selector_gate_pass"] else 0
        remaining_bad_seq_values.append(int(stats["bad_sequence_coverage"]))
        out.append(
            {
                "row_kind": "leave_one_selected_sequence_out",
                "policy_id": policy.get("policy_id", ""),
                "sequence": seq,
                "remaining_selected_row_count": stats["selected_row_count"],
                "remaining_bad_rows": stats["bad_rows"],
                "remaining_good_rows": stats["good_rows"],
                "remaining_bad_sequence_coverage": stats["bad_sequence_coverage"],
                "remaining_balanced_accuracy": stats["balanced_accuracy"],
                "remaining_bad_median_I_J": stats["bad_median_I_J_runtime_proxy"],
                "remaining_margin_vs_random_p95": stats["bad_median_margin_vs_same_count_random_p95"],
                "leaveout_gate_pass": stats["retrospective_selector_gate_pass"],
                "claim_level": "sequence_stability_no_runtime",
            }
        )
    if remaining_bad_seq_values:
        min_remaining_bad_seq = min(remaining_bad_seq_values)
    summary = {
        "best_selected_sequence_ids": selected_sequences,
        "best_missing_sequence_ids": [seq for seq in all_sequences if seq not in selected_sequences],
        "best_selected_pair_id_count": len(selected_pair_ids),
        "best_leave_one_selected_sequence_out_trial_count": len(selected_sequences),
        "best_leave_one_selected_sequence_out_pass_count": leaveout_pass_count,
        "best_min_remaining_bad_sequence_coverage_after_drop": min_remaining_bad_seq,
        "best_signal_stability_status": (
            "unstable_selected_sequence_dependent"
            if selected_sequences and leaveout_pass_count == 0
            else "not_a_sequence_drop_failure"
        ),
    }
    return out, summary


def promotion_readiness_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    passing = [row for row in candidates if row["retrospective_selector_gate_pass"]]
    out: list[dict[str, Any]] = []
    for row in passing:
        out.append(
            {
                "policy_id": row["policy_id"],
                "family": row["family"],
                "selected_signature": row["selected_signature"],
                "selected_row_count": row["selected_row_count"],
                "bad_rows": row["bad_rows"],
                "good_rows": row["good_rows"],
                "bad_recall": row["bad_recall"],
                "good_FPR": row["good_FPR"],
                "balanced_accuracy": row["balanced_accuracy"],
                "bad_sequence_coverage": row["bad_sequence_coverage"],
                "bad_median_I_J_runtime_proxy": row["bad_median_I_J_runtime_proxy"],
                "bad_median_margin_vs_same_count_random_p95": row[
                    "bad_median_margin_vs_same_count_random_p95"
                ],
                "retrospective_selector_gate_pass": row["retrospective_selector_gate_pass"],
                "promotion_floor_pass": row["promotion_floor_pass"],
                "action_authorized": False,
                "readiness_blocker": (
                    "retrospective_pass_but_low_BA_or_tiny_margin"
                    if not row["promotion_floor_pass"]
                    else "retrospective_only_requires_fresh_predeclared_rerun"
                ),
                "claim_level": "promotion_readiness_no_runtime",
            }
        )
    return out


def holdout_supportive(stats: dict[str, Any]) -> bool:
    good_max = stats.get("good_max_worsen_runtime_proxy")
    good_gate = good_max == "" or good_max is None or float(good_max) <= 0.02
    bad_median = stats.get("bad_median_I_J_runtime_proxy")
    return bool(
        stats.get("bad_rows", 0) >= 1
        and float(stats.get("good_FPR", 1.0)) <= 0.25
        and bad_median != ""
        and bad_median is not None
        and float(bad_median) >= 0.05
        and stats.get("bad_negative_improvement_rows", 1) == 0
        and good_gate
    )


def build_loso_holdout_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out: list[dict[str, Any]] = []
    sequences = sorted({row.get("seq", "") for row in rows})
    train_split_with_retrospective_pass = 0
    train_split_with_promotion_floor_pass = 0
    train_best_holdout_supportive_count = 0
    train_passing_holdout_supportive_count = 0
    for heldout_seq in sequences:
        train_rows = [row for row in rows if row.get("seq", "") != heldout_seq]
        heldout_rows = [row for row in rows if row.get("seq", "") == heldout_seq]
        train_features = valid_features(train_rows)
        train_candidates, train_atom_count = build_candidates(train_rows, train_features)
        train_passing = [row for row in train_candidates if row["retrospective_selector_gate_pass"]]
        train_promotion = [row for row in train_candidates if row["promotion_floor_pass"]]
        train_split_with_retrospective_pass += 1 if train_passing else 0
        train_split_with_promotion_floor_pass += 1 if train_promotion else 0

        train_best = train_candidates[0] if train_candidates else {}
        best_stats = selected_stats(heldout_rows, selected_source_rows_for_policy(heldout_rows, train_best)) if train_best else {}
        best_supportive = holdout_supportive(best_stats) if best_stats else False
        train_best_holdout_supportive_count += 1 if best_supportive else 0

        heldout_supportive_from_train_passing = 0
        best_train_passing_policy = ""
        best_train_passing_supportive_stats: dict[str, Any] = {}
        for candidate in train_passing:
            heldout_stats = selected_stats(heldout_rows, selected_source_rows_for_policy(heldout_rows, candidate))
            supportive = holdout_supportive(heldout_stats)
            heldout_supportive_from_train_passing += 1 if supportive else 0
            if not best_train_passing_supportive_stats or (
                bool(supportive),
                float(heldout_stats.get("balanced_accuracy", 0.0)),
                int(heldout_stats.get("bad_rows", 0)),
                -int(heldout_stats.get("good_rows", 0)),
            ) > (
                bool(best_train_passing_supportive_stats.get("heldout_supportive", False)),
                float(best_train_passing_supportive_stats.get("balanced_accuracy", 0.0)),
                int(best_train_passing_supportive_stats.get("bad_rows", 0)),
                -int(best_train_passing_supportive_stats.get("good_rows", 0)),
            ):
                best_train_passing_policy = candidate["policy_id"]
                best_train_passing_supportive_stats = {
                    **heldout_stats,
                    "heldout_supportive": supportive,
                }
        train_passing_holdout_supportive_count += 1 if heldout_supportive_from_train_passing else 0

        out.append(
            {
                "heldout_seq": heldout_seq,
                "train_row_count": len(train_rows),
                "heldout_row_count": len(heldout_rows),
                "train_feature_count": len(train_features),
                "train_threshold_atom_count": train_atom_count,
                "train_candidate_policy_count": len(train_candidates),
                "train_retrospective_passing_candidate_count": len(train_passing),
                "train_promotion_floor_passing_candidate_count": len(train_promotion),
                "train_best_policy_id": train_best.get("policy_id", ""),
                "train_best_retrospective_gate_pass": train_best.get("retrospective_selector_gate_pass", ""),
                "train_best_promotion_floor_pass": train_best.get("promotion_floor_pass", ""),
                "train_best_heldout_selected_row_count": best_stats.get("selected_row_count", ""),
                "train_best_heldout_bad_rows": best_stats.get("bad_rows", ""),
                "train_best_heldout_good_rows": best_stats.get("good_rows", ""),
                "train_best_heldout_bad_recall": best_stats.get("bad_recall", ""),
                "train_best_heldout_good_FPR": best_stats.get("good_FPR", ""),
                "train_best_heldout_BA": best_stats.get("balanced_accuracy", ""),
                "train_best_heldout_bad_median_I_J": best_stats.get("bad_median_I_J_runtime_proxy", ""),
                "train_best_heldout_bad_negative_rows": best_stats.get("bad_negative_improvement_rows", ""),
                "train_best_heldout_margin_vs_random_p95": best_stats.get(
                    "bad_median_margin_vs_same_count_random_p95", ""
                ),
                "train_best_heldout_supportive": best_supportive,
                "train_passing_heldout_supportive_candidate_count": heldout_supportive_from_train_passing,
                "best_train_passing_policy_id_on_heldout": best_train_passing_policy,
                "best_train_passing_heldout_supportive": best_train_passing_supportive_stats.get(
                    "heldout_supportive", ""
                ),
                "claim_level": "loso_holdout_diagnostic_no_runtime",
            }
        )
    summary = {
        "loso_holdout_sequence_count": len(sequences),
        "loso_train_split_with_retrospective_pass_count": train_split_with_retrospective_pass,
        "loso_train_split_with_promotion_floor_pass_count": train_split_with_promotion_floor_pass,
        "loso_train_best_heldout_supportive_split_count": train_best_holdout_supportive_count,
        "loso_train_passing_heldout_supportive_split_count": train_passing_holdout_supportive_count,
        "loso_holdout_status": (
            "no_train_split_finds_retrospective_selector"
            if train_split_with_retrospective_pass == 0
            else "train_retrospective_selectors_require_manual_review"
        ),
    }
    return out, summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = join_rows()
    features = valid_features(rows)
    candidates, atom_count = build_candidates(rows, features)
    passing = [row for row in candidates if row["retrospective_selector_gate_pass"]]
    best = candidates[0] if candidates else {}
    signature_count = len({row["selected_signature"] for row in passing})

    metric_fields = [
        "policy_id",
        "family",
        "feature1",
        "direction1",
        "threshold1",
        "quantile1",
        "feature2",
        "direction2",
        "threshold2",
        "quantile2",
        "selected_row_count",
        "bad_rows",
        "good_rows",
        "bad_recall",
        "good_FPR",
        "balanced_accuracy",
        "bad_sequence_coverage",
        "selected_sequence_coverage",
        "bad_median_I_J_runtime_proxy",
        "bad_min_I_J_runtime_proxy",
        "bad_negative_improvement_rows",
        "good_max_worsen_runtime_proxy",
        "good_worsen_gt_0p02_rows",
        "same_count_random_repeats",
        "same_count_random_bad_median_I_J_p50",
        "same_count_random_bad_median_I_J_p95",
        "same_count_random_BA_p95",
        "bad_median_margin_vs_same_count_random_p95",
        "retrospective_selector_gate_pass",
        "promotion_floor_pass",
        "action_authorized",
        "action_block_reason",
        "selected_signature",
        "claim_level",
    ]
    passing_fields = metric_fields + ["bad_hits", "good_hits"]
    write_csv(OUT / "merge_gauge_rich_selector_reentry_candidate_metrics.csv", candidates, metric_fields)
    write_csv(OUT / "merge_gauge_rich_selector_reentry_passing_candidates.csv", passing, passing_fields)
    readiness_rows = promotion_readiness_rows(candidates)
    write_csv(
        OUT / "merge_gauge_rich_selector_reentry_promotion_readiness.csv",
        readiness_rows,
        [
            "policy_id",
            "family",
            "selected_signature",
            "selected_row_count",
            "bad_rows",
            "good_rows",
            "bad_recall",
            "good_FPR",
            "balanced_accuracy",
            "bad_sequence_coverage",
            "bad_median_I_J_runtime_proxy",
            "bad_median_margin_vs_same_count_random_p95",
            "retrospective_selector_gate_pass",
            "promotion_floor_pass",
            "action_authorized",
            "readiness_blocker",
            "claim_level",
        ],
    )
    write_csv(
        OUT / "merge_gauge_rich_selector_reentry_best_selected_rows.csv",
        selected_rows_for_policy(rows, best) if best else [],
        [
            "policy_id",
            "pair_id",
            "seq",
            "label",
            "failure_type_primary",
            "feature1",
            "feature1_value",
            "feature2",
            "feature2_value",
            "I_J_runtime_proxy",
            "W_good_runtime_proxy",
        ],
    )
    stability_rows, stability_summary = build_sequence_stability_rows(rows, best)
    write_csv(
        OUT / "merge_gauge_rich_selector_reentry_sequence_stability.csv",
        stability_rows,
        [
            "row_kind",
            "policy_id",
            "sequence",
            "sequence_total_rows",
            "sequence_bad_rows",
            "sequence_good_rows",
            "selected_row_count",
            "selected_bad_rows",
            "selected_good_rows",
            "selected_bad_median_I_J",
            "selected_good_max_worsen",
            "selected_pair_ids",
            "remaining_selected_row_count",
            "remaining_bad_rows",
            "remaining_good_rows",
            "remaining_bad_sequence_coverage",
            "remaining_balanced_accuracy",
            "remaining_bad_median_I_J",
            "remaining_margin_vs_random_p95",
            "leaveout_gate_pass",
            "claim_level",
        ],
    )
    loso_rows, loso_summary = build_loso_holdout_rows(rows)
    write_csv(
        OUT / "merge_gauge_rich_selector_reentry_loso_holdout.csv",
        loso_rows,
        [
            "heldout_seq",
            "train_row_count",
            "heldout_row_count",
            "train_feature_count",
            "train_threshold_atom_count",
            "train_candidate_policy_count",
            "train_retrospective_passing_candidate_count",
            "train_promotion_floor_passing_candidate_count",
            "train_best_policy_id",
            "train_best_retrospective_gate_pass",
            "train_best_promotion_floor_pass",
            "train_best_heldout_selected_row_count",
            "train_best_heldout_bad_rows",
            "train_best_heldout_good_rows",
            "train_best_heldout_bad_recall",
            "train_best_heldout_good_FPR",
            "train_best_heldout_BA",
            "train_best_heldout_bad_median_I_J",
            "train_best_heldout_bad_negative_rows",
            "train_best_heldout_margin_vs_random_p95",
            "train_best_heldout_supportive",
            "train_passing_heldout_supportive_candidate_count",
            "best_train_passing_policy_id_on_heldout",
            "best_train_passing_heldout_supportive",
            "claim_level",
        ],
    )

    summary = {
        "schema": "acl2_v101_merge_gauge_rich_selector_reentry_v1",
        "source_action_rows": str(ACTION_ROWS),
        "source_carrier_rows": str(CARRIER_ROWS),
        "source_boundary_rows": str(BOUNDARY_ROWS),
        "feature_policy": "pre_action_geometry_merge_gauge_observability_whitelist_no_semantic_no_future_outcomes",
        "feature_count": len(features),
        "feature_names": features,
        "labelled_row_count": len(rows),
        "bad_label_count": sum(1 for row in rows if row.get("case_label_offline_only") == "bad"),
        "good_label_count": sum(1 for row in rows if row.get("case_label_offline_only") == "good"),
        "threshold_atom_count": atom_count,
        "candidate_policy_count": len(candidates),
        "retrospective_passing_candidate_count": len(passing),
        "promotion_floor_passing_candidate_count": sum(
            1 for row in candidates if row.get("promotion_floor_pass")
        ),
        "action_authorized_candidate_count": 0,
        "passing_selected_signature_count": signature_count,
        "best_policy_id": best.get("policy_id", ""),
        "best_family": best.get("family", ""),
        "best_retrospective_selector_gate_pass": bool(best.get("retrospective_selector_gate_pass", False)),
        "best_promotion_floor_pass": bool(best.get("promotion_floor_pass", False)),
        "best_action_authorized": False,
        "best_action_block_reason": best.get("action_block_reason", ""),
        "best_balanced_accuracy": best.get("balanced_accuracy", ""),
        "best_bad_rows": best.get("bad_rows", ""),
        "best_good_rows": best.get("good_rows", ""),
        "best_bad_recall": best.get("bad_recall", ""),
        "best_good_FPR": best.get("good_FPR", ""),
        "best_bad_sequence_coverage": best.get("bad_sequence_coverage", ""),
        "best_selected_sequence_coverage": best.get("selected_sequence_coverage", ""),
        "best_bad_median_I_J_runtime_proxy": best.get("bad_median_I_J_runtime_proxy", ""),
        "best_bad_min_I_J_runtime_proxy": best.get("bad_min_I_J_runtime_proxy", ""),
        "best_bad_negative_improvement_rows": best.get("bad_negative_improvement_rows", ""),
        "best_good_max_worsen_runtime_proxy": best.get("good_max_worsen_runtime_proxy", ""),
        "best_random_p95_bad_median_I_J": best.get("same_count_random_bad_median_I_J_p95", ""),
        "best_bad_median_margin_vs_random_p95": best.get("bad_median_margin_vs_same_count_random_p95", ""),
        **stability_summary,
        **loso_summary,
        "runtime_action_allowed": False,
        "full_validation_allowed": False,
        "v101_goal_achieved": False,
        "claim": "Existing-row retrospective selector screen only; no runtime or full-validation authorization.",
        "limitations": [
            "policy family was screened after v94 measured outcomes existed",
            "best retrospective pass has low balanced accuracy or tiny margin",
            "requires a fresh predeclared holdout or measured-control rerun before promotion",
        ],
    }
    write_json(OUT / "merge_gauge_rich_selector_reentry_summary.json", summary)

    report = [
        "# Merge/Gauge Rich Selector Re-entry Screen",
        "",
        "This is an existing-row retrospective screen over v94 measured merge/gauge action outcomes.",
        "It does not authorize M4, runtime pilots, or full validation.",
        "",
        "## Summary",
        "",
        f"- labelled rows: `{summary['labelled_row_count']}`",
        f"- bad / good rows: `{summary['bad_label_count']}` / `{summary['good_label_count']}`",
        f"- feature count: `{summary['feature_count']}`",
        f"- threshold atoms: `{summary['threshold_atom_count']}`",
        f"- candidate policies: `{summary['candidate_policy_count']}`",
        f"- retrospective passing candidates: `{summary['retrospective_passing_candidate_count']}`",
        f"- promotion-floor passing candidates: `{summary['promotion_floor_passing_candidate_count']}`",
        f"- action-authorized candidates: `{summary['action_authorized_candidate_count']}`",
        f"- passing selected signatures: `{summary['passing_selected_signature_count']}`",
        f"- best policy: `{summary['best_policy_id']}`",
        f"- best family: `{summary['best_family']}`",
        f"- best BA: `{summary['best_balanced_accuracy']}`",
        f"- best bad/good rows: `{summary['best_bad_rows']}` / `{summary['best_good_rows']}`",
        f"- best bad recall / good FPR: `{summary['best_bad_recall']}` / `{summary['best_good_FPR']}`",
        f"- best bad sequence coverage: `{summary['best_bad_sequence_coverage']}`",
        f"- best bad median I/J: `{summary['best_bad_median_I_J_runtime_proxy']}`",
        f"- best random p95 bad median I/J: `{summary['best_random_p95_bad_median_I_J']}`",
        f"- best margin vs random p95: `{summary['best_bad_median_margin_vs_random_p95']}`",
        f"- best selected sequences: `{summary.get('best_selected_sequence_ids', '')}`",
        f"- missing selected sequences: `{summary.get('best_missing_sequence_ids', '')}`",
        f"- leave-one-selected-sequence-out pass count: `{summary.get('best_leave_one_selected_sequence_out_pass_count', '')}` / `{summary.get('best_leave_one_selected_sequence_out_trial_count', '')}`",
        f"- sequence stability status: `{summary.get('best_signal_stability_status', '')}`",
        f"- LOSO holdout status: `{summary.get('loso_holdout_status', '')}`",
        f"- LOSO train splits with retrospective pass: `{summary.get('loso_train_split_with_retrospective_pass_count', '')}` / `{summary.get('loso_holdout_sequence_count', '')}`",
        f"- LOSO train splits with promotion-floor pass: `{summary.get('loso_train_split_with_promotion_floor_pass_count', '')}` / `{summary.get('loso_holdout_sequence_count', '')}`",
        f"- best action authorized: `{summary['best_action_authorized']}`",
        f"- best promotion-floor pass: `{summary['best_promotion_floor_pass']}`",
        f"- action block reason: `{summary['best_action_block_reason']}`",
        "",
        "## Interpretation",
        "",
    ]
    if passing:
        report.extend(
            [
                "The richer whitelist family found a weak retrospective selector signal, but it is not promotion-ready.",
                "The best pass is low-recall / low-BA and only slightly beats the same-count random p95 margin.",
                "It also fails the leave-one-selected-sequence-out stability check.",
                "Because this was selected after existing outcomes were available, the only valid next step is a fresh predeclared holdout or measured-control rerun.",
            ]
        )
    else:
        report.append("No richer whitelist selector passed the retrospective diagnostic gate.")
    report.extend(["", "Runtime action remains disallowed."])
    (OUT / "merge_gauge_rich_selector_reentry_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    spec = [
        "# Merge/Gauge Rich Selector Next Rerun Spec",
        "",
        "This is not a runtime authorization. It is a preregistration boundary for a future rerun.",
        "",
        "## Fixed Family",
        "",
        "- Use only the whitelist in `merge_gauge_rich_selector_reentry_summary.json`.",
        "- Evaluate 1D quantile thresholds and 2D AND combinations at q25/q40/q50/q60/q75/q90.",
        "- Do not add semantic mass, future-error, label, or measured outcome fields to selector inputs.",
        "- Keep same-count random controls with the fixed seed/repeat policy or a stricter declared alternative.",
        "",
        "## Promotion Guard",
        "",
        "- Passing the retrospective screen is insufficient.",
        "- A future rerun should require fresh predeclared evaluation, no bad negative-improvement rows, good harm protection, and replication beyond this tiny-margin signal.",
        f"- Suggested stricter promotion floor for the future rerun: BA >= `{PROMOTION_MIN_BA}` and bad-median margin vs random p95 >= `{PROMOTION_MIN_MARGIN}`.",
        "",
        "## Current Retrospective Signal",
        "",
        f"- retrospective passing candidates: `{summary['retrospective_passing_candidate_count']}`",
        f"- promotion-floor passing candidates: `{summary['promotion_floor_passing_candidate_count']}`",
        f"- best policy: `{summary['best_policy_id']}`",
        f"- best BA: `{summary['best_balanced_accuracy']}`",
        f"- best margin vs random p95: `{summary['best_bad_median_margin_vs_random_p95']}`",
        f"- sequence stability status: `{summary.get('best_signal_stability_status', '')}`",
        f"- LOSO holdout status: `{summary.get('loso_holdout_status', '')}`",
        f"- action authorized now: `{summary['best_action_authorized']}`",
    ]
    (OUT / "merge_gauge_rich_selector_next_rerun_spec.md").write_text("\n".join(spec) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
