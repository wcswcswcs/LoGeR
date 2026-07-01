#!/usr/bin/env python3
"""Audit non-semantic merge/gauge selector re-entry candidates.

This is a read-only Outcome-D follow-up for v101. It reuses v94 measured
merge-alpha action-surface rows and searches simple pre-action numeric carrier
selectors. It does not authorize runtime action.
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
ACTION_SUMMARY = V94_ROOT / "phase6_object_source_action_surface" / "phase6_object_source_action_surface_summary.json"
PHASE3S_SUMMARY = V94_ROOT / "phase3s_merge_gauge_actuator_sweep_max16_confirm" / "runtime_probe_sensitivity_summary.json"

FEATURES = [
    "native_curr_postmerge_sim3_rmse",
    "native_curr_handoff_transfer_rmse",
    "native_abs_log_scale_jump_runtime",
    "native_boundary_update_norm",
    "native_merge_residual_after_abs",
    "carrier_error_boundary_update_norm",
    "carrier_error_merge_residual_after_abs",
    "carrier_error_abs_log_scale_jump_runtime",
    "carrier_error_composite_z",
]

QUANTILES = [0.25, 0.40, 0.50, 0.60, 0.75, 0.90]
RANDOM_REPEATS = 512
RANDOM_SEED = 10194


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def fnum(value: Any) -> float | None:
    try:
        out = float(str(value))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    vals = sorted(values)
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
    action_rows = read_csv(ACTION_ROWS)
    carrier_by_pair = {row["pair_id"]: row for row in read_csv(CARRIER_ROWS)}
    rows: list[dict[str, Any]] = []
    for row in action_rows:
        out: dict[str, Any] = dict(row)
        carrier = carrier_by_pair.get(row["pair_id"], {})
        for key in FEATURES:
            if key not in out and key in carrier:
                out[key] = carrier[key]
        rows.append(out)
    return rows


def selected_stats(all_rows: list[dict[str, Any]], selected: list[dict[str, Any]]) -> dict[str, Any]:
    bad_total = sum(1 for row in all_rows if row["case_label_offline_only"] == "bad")
    good_total = sum(1 for row in all_rows if row["case_label_offline_only"] == "good")
    bad_rows = [row for row in selected if row["case_label_offline_only"] == "bad"]
    good_rows = [row for row in selected if row["case_label_offline_only"] == "good"]
    bad_ij = [fnum(row.get("I_J_runtime_proxy")) for row in bad_rows]
    bad_ij = [v for v in bad_ij if v is not None]
    good_w = [fnum(row.get("W_good_runtime_proxy")) for row in good_rows]
    good_w = [v for v in good_w if v is not None]
    bad_hits = [row["pair_id"] for row in bad_rows]
    good_hits = [row["pair_id"] for row in good_rows]
    bad_median = median(bad_ij) if bad_ij else None
    bad_min = min(bad_ij) if bad_ij else None
    good_max = max(good_w) if good_w else None
    good_median = median(good_w) if good_w else None
    return {
        "selected_row_count": len(selected),
        "bad_rows": len(bad_rows),
        "good_rows": len(good_rows),
        "bad_recall": len(bad_rows) / bad_total if bad_total else 0.0,
        "good_FPR": len(good_rows) / good_total if good_total else 0.0,
        "balanced_accuracy": ((len(bad_rows) / bad_total if bad_total else 0.0) + (1.0 - (len(good_rows) / good_total if good_total else 0.0))) / 2.0,
        "bad_sequence_coverage": len({row["seq"] for row in bad_rows}),
        "selected_sequence_coverage": len({row["seq"] for row in selected}),
        "bad_median_I_J_runtime_proxy": bad_median,
        "bad_min_I_J_runtime_proxy": bad_min,
        "bad_negative_improvement_rows": sum(1 for v in bad_ij if v < 0.0),
        "good_median_worsen_runtime_proxy": good_median,
        "good_max_worsen_runtime_proxy": good_max,
        "good_worsen_gt_0p02_rows": sum(1 for v in good_w if v > 0.02),
        "bad_hits": ",".join(bad_hits),
        "good_hits": ",".join(good_hits),
    }


def random_control(all_rows: list[dict[str, Any]], selected_count: int) -> dict[str, Any]:
    rng = random.Random(RANDOM_SEED + selected_count)
    if selected_count <= 0:
        return {
            "same_count_random_repeats": RANDOM_REPEATS,
            "same_count_random_bad_median_I_J_p50": "",
            "same_count_random_bad_median_I_J_p95": "",
            "same_count_random_BA_p95": "",
        }
    bad_medians: list[float] = []
    bas: list[float] = []
    for _ in range(RANDOM_REPEATS):
        sample = rng.sample(all_rows, min(selected_count, len(all_rows)))
        stats = selected_stats(all_rows, sample)
        med = stats["bad_median_I_J_runtime_proxy"]
        if med is not None and math.isfinite(float(med)):
            bad_medians.append(float(med))
        bas.append(float(stats["balanced_accuracy"]))
    return {
        "same_count_random_repeats": RANDOM_REPEATS,
        "same_count_random_bad_median_I_J_p50": quantile(bad_medians, 0.50) if bad_medians else "",
        "same_count_random_bad_median_I_J_p95": quantile(bad_medians, 0.95) if bad_medians else "",
        "same_count_random_BA_p95": quantile(bas, 0.95) if bas else "",
    }


def evaluate_policy(all_rows: list[dict[str, Any]], policy_id: str, feature: str, direction: str, threshold: float) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for row in all_rows:
        val = fnum(row.get(feature))
        if val is None:
            continue
        if direction == "ge" and val >= threshold:
            selected.append(row)
        elif direction == "le" and val <= threshold:
            selected.append(row)
    stats = selected_stats(all_rows, selected)
    controls = random_control(all_rows, stats["selected_row_count"])
    bad_median = stats["bad_median_I_J_runtime_proxy"]
    rand_p95 = controls["same_count_random_bad_median_I_J_p95"]
    bad_margin = ""
    if bad_median is not None and rand_p95 != "":
        bad_margin = float(bad_median) - float(rand_p95)
    good_max = stats["good_max_worsen_runtime_proxy"]
    good_gate = good_max is None or float(good_max) <= 0.02
    pass_gate = (
        stats["bad_rows"] >= 3
        and stats["bad_sequence_coverage"] >= 3
        and stats["good_FPR"] <= 0.25
        and bad_median is not None
        and float(bad_median) >= 0.05
        and stats["bad_negative_improvement_rows"] == 0
        and good_gate
        and bad_margin != ""
        and float(bad_margin) > 0.0
    )
    row: dict[str, Any] = {
        "policy_id": policy_id,
        "feature": feature,
        "direction": direction,
        "threshold": threshold,
        **stats,
        **controls,
        "bad_median_margin_vs_same_count_random_p95": bad_margin,
        "selector_gate_pass": pass_gate,
        "claim_level": "diagnostic_selector_reentry_no_runtime",
    }
    return row


def build_candidates(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in FEATURES:
        vals = [fnum(row.get(feature)) for row in all_rows]
        vals = [v for v in vals if v is not None]
        if len(vals) < 4:
            continue
        for q in QUANTILES:
            threshold = quantile(vals, q)
            for direction in ("ge", "le"):
                policy_id = f"{feature}_{direction}_q{int(q * 100):02d}"
                rows.append(evaluate_policy(all_rows, policy_id, feature, direction, threshold))
    rows.sort(
        key=lambda row: (
            not bool(row["selector_gate_pass"]),
            -float(row["balanced_accuracy"]),
            -float(row["bad_rows"]),
            float(row["good_FPR"]),
            -float(row["bad_median_I_J_runtime_proxy"] or -1e99),
        )
    )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = join_rows()
    candidates = build_candidates(rows)
    passing = [row for row in candidates if row["selector_gate_pass"]]
    best = candidates[0] if candidates else {}
    action_summary = read_json(ACTION_SUMMARY)
    phase3s_summary = read_json(PHASE3S_SUMMARY)

    fieldnames = [
        "policy_id",
        "feature",
        "direction",
        "threshold",
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
        "good_median_worsen_runtime_proxy",
        "good_max_worsen_runtime_proxy",
        "good_worsen_gt_0p02_rows",
        "same_count_random_repeats",
        "same_count_random_bad_median_I_J_p50",
        "same_count_random_bad_median_I_J_p95",
        "same_count_random_BA_p95",
        "bad_median_margin_vs_same_count_random_p95",
        "selector_gate_pass",
        "bad_hits",
        "good_hits",
        "claim_level",
    ]
    write_csv(OUT / "merge_gauge_selector_reentry_candidate_metrics.csv", candidates, fieldnames)
    write_csv(OUT / "merge_gauge_selector_reentry_passing_candidates.csv", passing, fieldnames)
    selected_rows: list[dict[str, Any]] = []
    if best:
        for row in rows:
            val = fnum(row.get(best["feature"]))
            if val is None:
                continue
            keep = val >= float(best["threshold"]) if best["direction"] == "ge" else val <= float(best["threshold"])
            if keep:
                selected_rows.append(
                    {
                        "policy_id": best["policy_id"],
                        "pair_id": row["pair_id"],
                        "seq": row["seq"],
                        "label": row["case_label_offline_only"],
                        "failure_type_primary": row.get("failure_type_primary", ""),
                        "feature_value": val,
                        "I_J_runtime_proxy": row.get("I_J_runtime_proxy", ""),
                        "W_good_runtime_proxy": row.get("W_good_runtime_proxy", ""),
                    }
                )
    write_csv(
        OUT / "merge_gauge_selector_reentry_best_selected_rows.csv",
        selected_rows,
        [
            "policy_id",
            "pair_id",
            "seq",
            "label",
            "failure_type_primary",
            "feature_value",
            "I_J_runtime_proxy",
            "W_good_runtime_proxy",
        ],
    )

    summary = {
        "schema": "acl2_v101_merge_gauge_selector_reentry_audit_v1",
        "source_action_rows": str(ACTION_ROWS),
        "source_carrier_rows": str(CARRIER_ROWS),
        "labelled_row_count": len(rows),
        "candidate_policy_count": len(candidates),
        "passing_candidate_count": len(passing),
        "best_policy_id": best.get("policy_id", ""),
        "best_selector_gate_pass": bool(best.get("selector_gate_pass", False)),
        "best_balanced_accuracy": best.get("balanced_accuracy", ""),
        "best_bad_rows": best.get("bad_rows", ""),
        "best_good_rows": best.get("good_rows", ""),
        "best_bad_sequence_coverage": best.get("bad_sequence_coverage", ""),
        "best_bad_median_I_J_runtime_proxy": best.get("bad_median_I_J_runtime_proxy", ""),
        "best_random_p95_bad_median_I_J": best.get("same_count_random_bad_median_I_J_p95", ""),
        "best_bad_median_margin_vs_random_p95": best.get("bad_median_margin_vs_same_count_random_p95", ""),
        "object_source_action_surface_gate_pass": action_summary.get("phase6_object_source_action_surface_gate_pass"),
        "object_source_actual_minus_best_control": action_summary.get("actual_minus_best_control"),
        "phase3s_merge_alpha_actuator_gate_pass": phase3s_summary.get("phase3r_runtime_probe_gate_pass"),
        "runtime_action_allowed": False,
        "full_validation_allowed": False,
        "claim": (
            "diagnostic selector re-entry only; candidates reuse measured v94 action-surface rows and require "
            "a fresh predeclared measured-control rerun before any runtime pilot"
        ),
    }
    with (OUT / "merge_gauge_selector_reentry_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    report = [
        "# Merge/Gauge Selector Re-entry Audit",
        "",
        "This audit searches simple non-semantic numeric selectors over v94 measured merge-alpha action-surface rows.",
        "It uses native/carrier-error pre-action fields only; probe outcome fields are used only for evaluation.",
        "",
        "## Summary",
        "",
        f"- labelled measured rows: `{summary['labelled_row_count']}`",
        f"- candidate policies: `{summary['candidate_policy_count']}`",
        f"- passing candidates: `{summary['passing_candidate_count']}`",
        f"- best policy: `{summary['best_policy_id']}`",
        f"- best selector gate pass: `{summary['best_selector_gate_pass']}`",
        f"- best balanced accuracy: `{summary['best_balanced_accuracy']}`",
        f"- best bad rows: `{summary['best_bad_rows']}`",
        f"- best good rows: `{summary['best_good_rows']}`",
        f"- best bad sequence coverage: `{summary['best_bad_sequence_coverage']}`",
        f"- best bad median I/J runtime proxy: `{summary['best_bad_median_I_J_runtime_proxy']}`",
        f"- best random p95 bad median I/J: `{summary['best_random_p95_bad_median_I_J']}`",
        f"- best bad-median margin vs random p95: `{summary['best_bad_median_margin_vs_random_p95']}`",
        "",
        "## Interpretation",
        "",
    ]
    if passing:
        report.append(
            "At least one non-semantic numeric selector passes this offline diagnostic gate, but it is not action-authorized because the rows reuse v94 measured outcomes. The next step would be a predeclared rerun with measured selection controls."
        )
    else:
        report.append(
            "No non-semantic numeric selector passed the strict offline diagnostic gate. The merge/gauge actuator remains a measured signal, but selector redesign is still unresolved."
        )
    report.append("")
    report.append("Runtime action remains disallowed.")
    (OUT / "merge_gauge_selector_reentry_report.md").write_text("\n".join(report) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
