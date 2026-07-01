#!/usr/bin/env python3
"""Build v94 Phase4 semantic evidence taxonomy from offline audit fields."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v94tf_semantic_gauge_failure_localization_causal_memory_control")
DEFAULT_PHASE1 = ROOT / "phase1_boundary_failure_atlas/boundary_failure_rows.csv"
DEFAULT_PHASE3 = ROOT / "phase3_formal_merge_alpha_sensitivity/phase3_formal_gate_summary.json"
DEFAULT_OUT = ROOT / "phase4_semantic_evidence_taxonomy"

RISK_CATEGORIES = {
    "SEM_WEAK_CONTEXT",
    "SEM_INVALID_BOUNDARY",
    "SEM_DYNAMIC_TRANSIENT",
    "SEM_MULTIMODE_UNSAFE",
    "SEM_LOWOBS_ABSTAIN",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def has_number(row: dict[str, str], key: str) -> bool:
    value = row.get(key, "")
    if value == "":
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def source_level(row: dict[str, str]) -> tuple[str, str]:
    component_fields = ["component_boundary_ratio", "cross_component_ratio", "same_component_ratio"]
    semantic_fields = ["semantic_valid_mass", "semantic_invalid_mass", "semantic_context_mass"]
    if all(has_number(row, key) for key in component_fields):
        return "C", "component_topology_feature_match_fallback"
    if any(has_number(row, key) for key in semantic_fields):
        return "D", "dense_semantic_mass_only"
    return "UNKNOWN", "no_semantic_or_component_source"


def score_row(row: dict[str, str]) -> dict[str, float]:
    semantic_invalid = f(row.get("semantic_invalid_mass"))
    semantic_context = f(row.get("semantic_context_mass"))
    cross_component = f(row.get("cross_component_ratio"))
    same_component = f(row.get("same_component_ratio"))
    dynamic = f(row.get("dynamic_or_transient_ratio"))
    observability = f(row.get("observability_score"), default=0.0)
    semantic_entropy = f(row.get("semantic_mode_entropy"))
    local_entropy = f(row.get("local_shape_mode_entropy"))
    invalid_score = max(semantic_invalid, cross_component, dynamic)
    context_score = max(semantic_context, max(0.0, 1.0 - observability))
    stable_score = same_component * observability * max(0.0, 1.0 - invalid_score) * max(0.0, 1.0 - context_score)
    multimode_score = min(1.0, max(semantic_entropy, local_entropy) / 4.0)
    lowobs_score = max(0.0, 1.0 - observability)
    return {
        "S_stable": stable_score,
        "S_invalid": invalid_score,
        "S_context": context_score,
        "S_multi": multimode_score,
        "S_lowobs": lowobs_score,
    }


def assign_taxonomy(scores: dict[str, float]) -> tuple[str, str]:
    # Fixed, no-label thresholds derived from the plan score definitions. The
    # order keeps hard invalid/low-observability cases from being hidden by the
    # very common multimode entropy signal.
    if scores["S_invalid"] >= 0.75:
        return "SEM_INVALID_BOUNDARY", "S_invalid>=0.75"
    if scores["S_context"] >= 0.75:
        return "SEM_WEAK_CONTEXT", "S_context>=0.75"
    if scores["S_lowobs"] >= 0.65:
        return "SEM_LOWOBS_ABSTAIN", "S_lowobs>=0.65"
    if scores["S_stable"] >= 0.15 and scores["S_invalid"] < 0.50 and scores["S_context"] < 0.65:
        return "SEM_STABLE_REFERENCE", "S_stable>=0.15 and invalid/context guarded"
    if scores["S_multi"] >= 0.80:
        return "SEM_MULTIMODE_UNSAFE", "S_multi>=0.80"
    if scores["S_context"] >= 0.55:
        return "SEM_WEAK_CONTEXT", "S_context>=0.55 fallback"
    return "SEM_UNKNOWN", "no_fixed_rule_matched"


def is_risk(category: str) -> bool:
    return category in RISK_CATEGORIES


def metric(rows: list[dict[str, Any]], pred_key: str, name: str) -> dict[str, Any]:
    labelled = [row for row in rows if row.get("case_label_offline_only") in {"bad", "good"}]
    bad = [row for row in labelled if row.get("case_label_offline_only") == "bad"]
    good = [row for row in labelled if row.get("case_label_offline_only") == "good"]
    bad_recall = sum(bool(row[pred_key]) for row in bad) / len(bad) if bad else 0.0
    good_fpr = sum(bool(row[pred_key]) for row in good) / len(good) if good else 0.0
    return {
        "policy": name,
        "labelled_rows": len(labelled),
        "bad_rows": len(bad),
        "good_rows": len(good),
        "positive_rows": sum(bool(row[pred_key]) for row in labelled),
        "bad_recall": bad_recall,
        "good_FPR": good_fpr,
        "balanced_accuracy": 0.5 * (bad_recall + 1.0 - good_fpr),
        "sequence_coverage": len({row.get("seq") for row in labelled}),
    }


def metric_from_categories(rows: list[dict[str, Any]], category_key: str, role: str, name: str) -> dict[str, Any]:
    pred_key = f"__pred_{category_key}_{role}"
    for row in rows:
        row[pred_key] = str(row.get(category_key) or "") == role
    out = metric(rows, pred_key, name)
    for row in rows:
        row.pop(pred_key, None)
    out["semantic_role"] = role
    out["category_key"] = category_key
    return out


def rotated(values: list[str], amount: int) -> list[str]:
    if not values:
        return []
    amount %= len(values)
    return values[amount:] + values[:amount]


def add_control_predictions(rows: list[dict[str, Any]]) -> None:
    categories = [str(row["semantic_evidence_type"]) for row in rows]
    semantic_rot = rotated(categories, 7)
    component_order = sorted(range(len(rows)), key=lambda idx: (f(rows[idx].get("component_boundary_ratio")), idx))
    component_rot_values = rotated([categories[idx] for idx in component_order], 11)
    component_rot = ["" for _ in rows]
    for idx, cat in zip(component_order, component_rot_values):
        component_rot[idx] = cat
    by_seq: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_seq[str(row.get("seq"))].append(idx)
    regime_rot = ["" for _ in rows]
    for indices in by_seq.values():
        seq_cats = [categories[idx] for idx in indices]
        for idx, cat in zip(indices, rotated(seq_cats, 1)):
            regime_rot[idx] = cat
    for row, sem, comp, reg in zip(rows, semantic_rot, component_rot, regime_rot):
        row["semantic_shuffle_category"] = sem
        row["component_shuffle_category"] = comp
        row["regime_shuffle_category"] = reg
        row["semantic_shuffle_risk"] = is_risk(sem)
        row["component_shuffle_risk"] = is_risk(comp)
        row["regime_shuffle_risk"] = is_risk(reg)
        primary = str(row.get("failure_type_primary") or "")
        secondary = str(row.get("failure_type_secondary") or "")
        row["geometry_conflict_risk"] = primary in {
            "HANDOFF_SCALE",
            "HANDOFF_GAUGE",
            "LOW_OBSERVABILITY",
            "MULTIMODE_CONFLICT",
        } or any(token in secondary for token in ["SEMANTIC_INVALID", "LOW_OBSERVABILITY", "MULTIMODE_CONFLICT"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-rows", type=Path, default=DEFAULT_PHASE1)
    parser.add_argument("--phase3-summary", type=Path, default=DEFAULT_PHASE3)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    phase3 = read_json(args.phase3_summary)
    if not phase3.get("phase3_repaired_gate_pass") and not args.force:
        raise SystemExit("Phase3 repaired gate did not pass; use --force only for diagnostics.")

    input_rows = read_csv_rows(args.phase1_rows)
    rows: list[dict[str, Any]] = []
    for row in input_rows:
        level, reason = source_level(row)
        scores = score_row(row)
        category, assignment_reason = assign_taxonomy(scores)
        out = dict(row)
        out.update(scores)
        out["semantic_source_level"] = level
        out["semantic_source_level_reason"] = reason
        out["semantic_evidence_type"] = category
        out["semantic_taxonomy_assignment_reason"] = assignment_reason
        out["semantic_taxonomy_risk"] = is_risk(category)
        rows.append(out)
    add_control_predictions(rows)

    actual = metric(rows, "semantic_taxonomy_risk", "semantic_taxonomy_actual")
    geometry = metric(rows, "geometry_conflict_risk", "geometry_conflict_control")
    semantic_shuffle = metric(rows, "semantic_shuffle_risk", "semantic_shuffle_control")
    component_shuffle = metric(rows, "component_shuffle_risk", "component_shuffle_control")
    regime_shuffle = metric(rows, "regime_shuffle_risk", "regime_shuffle_control")

    for control in [semantic_shuffle, component_shuffle, regime_shuffle, geometry]:
        control["actual_balanced_accuracy_margin"] = actual["balanced_accuracy"] - control["balanced_accuracy"]
        control["actual_good_FPR_margin"] = control["good_FPR"] - actual["good_FPR"]

    semantic_roles = [
        "SEM_STABLE_REFERENCE",
        "SEM_WEAK_CONTEXT",
        "SEM_INVALID_BOUNDARY",
        "SEM_DYNAMIC_TRANSIENT",
        "SEM_MULTIMODE_UNSAFE",
        "SEM_LOWOBS_ABSTAIN",
    ]
    role_rows: list[dict[str, Any]] = []
    for role in semantic_roles:
        actual_role = metric_from_categories(rows, "semantic_evidence_type", role, f"{role}_actual")
        semantic_role = metric_from_categories(rows, "semantic_shuffle_category", role, f"{role}_semantic_shuffle")
        component_role = metric_from_categories(rows, "component_shuffle_category", role, f"{role}_component_shuffle")
        regime_role = metric_from_categories(rows, "regime_shuffle_category", role, f"{role}_regime_shuffle")
        shuffle_margins_for_role = [
            actual_role["balanced_accuracy"] - semantic_role["balanced_accuracy"],
            actual_role["balanced_accuracy"] - component_role["balanced_accuracy"],
            actual_role["balanced_accuracy"] - regime_role["balanced_accuracy"],
        ]
        role_rows.append(
            {
                "semantic_role": role,
                "bad_recall": actual_role["bad_recall"],
                "good_FPR": actual_role["good_FPR"],
                "balanced_accuracy": actual_role["balanced_accuracy"],
                "positive_rows": actual_role["positive_rows"],
                "semantic_shuffle_margin": shuffle_margins_for_role[0],
                "component_shuffle_margin": shuffle_margins_for_role[1],
                "regime_shuffle_margin": shuffle_margins_for_role[2],
                "max_shuffle_margin": max(shuffle_margins_for_role),
                "good_FPR_lower_than_geometry": actual_role["good_FPR"] < geometry["good_FPR"],
                "role_gate_pass": actual_role["good_FPR"] < geometry["good_FPR"]
                and max(shuffle_margins_for_role) >= 0.05,
            }
        )
    best_role = max(role_rows, key=lambda row: (bool(row["role_gate_pass"]), row["balanced_accuracy"]))

    category_counts = Counter(str(row["semantic_evidence_type"]) for row in rows)
    source_counts = Counter(str(row["semantic_source_level"]) for row in rows)
    assigned_ratio = sum(str(row["semantic_evidence_type"]) != "SEM_UNKNOWN" for row in rows) / len(rows)
    source_recorded_ratio = sum(str(row["semantic_source_level"]) != "UNKNOWN" for row in rows) / len(rows)
    labelled_good = [row for row in rows if row.get("case_label_offline_only") == "good"]
    false_positives = [row for row in labelled_good if row["semantic_taxonomy_risk"]]
    good_fpr_lower_than_geometry = bool(best_role["good_FPR_lower_than_geometry"])
    shuffle_margins = [
        semantic_shuffle["actual_balanced_accuracy_margin"],
        component_shuffle["actual_balanced_accuracy_margin"],
        regime_shuffle["actual_balanced_accuracy_margin"],
    ]
    categories_non_empty = all(category_counts.get(cat, 0) > 0 for cat in [
        "SEM_STABLE_REFERENCE",
        "SEM_INVALID_BOUNDARY",
        "SEM_WEAK_CONTEXT",
    ])

    checks = {
        "semantic_evidence_assigned_ge_90pct": assigned_ratio >= 0.90,
        "source_level_recorded_for_all_rows": source_recorded_ratio >= 1.0,
        "level_A_B_coverage_reported": True,
        "stable_invalid_context_non_empty": categories_non_empty,
        "semantic_role_good_FPR_lower_than_geometry_conflict": good_fpr_lower_than_geometry,
        "one_semantic_role_shuffle_margin_ge_0p05": bool(best_role["max_shuffle_margin"] >= 0.05),
    }
    gate_pass = all(checks.values())
    blockers = [name for name, passed in checks.items() if not passed]
    summary = {
        "phase": "Phase4_semantic_evidence_taxonomy",
        "entered": True,
        "phase4_semantic_taxonomy_gate_pass": gate_pass,
        "blocker": "" if gate_pass else ";".join(blockers),
        "row_count": len(rows),
        "assigned_ratio": assigned_ratio,
        "source_level_recorded_ratio": source_recorded_ratio,
        "source_level_counts": dict(source_counts),
        "level_A_coverage": float(source_counts.get("A", 0) / len(rows)),
        "level_B_coverage": float(source_counts.get("B", 0) / len(rows)),
        "level_C_coverage": float(source_counts.get("C", 0) / len(rows)),
        "level_D_coverage": float(source_counts.get("D", 0) / len(rows)),
        "semantic_evidence_type_counts": dict(category_counts),
        "actual_policy_metrics": actual,
        "geometry_conflict_control_metrics": geometry,
        "semantic_shuffle_control_metrics": semantic_shuffle,
        "component_shuffle_control_metrics": component_shuffle,
        "regime_shuffle_control_metrics": regime_shuffle,
        "semantic_role_metrics": role_rows,
        "best_semantic_role": best_role,
        "checks": checks,
        "false_positive_good_rows": len(false_positives),
        "false_positive_good_fraction": len(false_positives) / len(labelled_good) if labelled_good else 0.0,
        "taxonomy_rule_note": "Fixed no-label rules over component topology, semantic masses, observability, and entropy; no bad/good labels used for assignment.",
        "phase5_semantic_carrier_alignment_allowed": gate_pass,
        "counterfactual_allowed": False,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    coverage_rows = [
        {"semantic_source_level": level, "row_count": int(count), "fraction": float(count / len(rows))}
        for level, count in sorted(source_counts.items())
    ]
    policy_rows = [actual, geometry, semantic_shuffle, component_shuffle, regime_shuffle]
    fp_rows = [
        {
            "pair_id": row.get("pair_id"),
            "seq": row.get("seq"),
            "prev_chunk": row.get("prev_chunk"),
            "curr_chunk": row.get("curr_chunk"),
            "semantic_evidence_type": row.get("semantic_evidence_type"),
            "semantic_taxonomy_assignment_reason": row.get("semantic_taxonomy_assignment_reason"),
            "S_invalid": row.get("S_invalid"),
            "S_context": row.get("S_context"),
            "S_stable": row.get("S_stable"),
            "S_multi": row.get("S_multi"),
            "failure_type_primary": row.get("failure_type_primary"),
            "failure_type_secondary": row.get("failure_type_secondary"),
        }
        for row in false_positives
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "semantic_evidence_rows.csv", rows)
    write_csv(args.out_dir / "semantic_evidence_by_boundary.csv", rows)
    write_csv(args.out_dir / "semantic_source_coverage.csv", coverage_rows)
    write_csv(args.out_dir / "semantic_taxonomy_policy_metrics.csv", policy_rows)
    write_csv(args.out_dir / "semantic_taxonomy_role_metrics.csv", role_rows)
    write_csv(args.out_dir / "semantic_false_positive_analysis.csv", fp_rows)
    write_json(args.out_dir / "semantic_taxonomy_summary.json", summary)

    print(f"phase4_semantic_taxonomy_gate_pass={gate_pass}")
    print(f"blocker={summary['blocker']}")
    print(f"assigned_ratio={assigned_ratio}")
    print(f"source_level_counts={dict(source_counts)}")
    print(f"semantic_evidence_type_counts={dict(category_counts)}")
    print(f"actual_bad_recall={actual['bad_recall']}")
    print(f"actual_good_FPR={actual['good_FPR']}")
    print(f"geometry_good_FPR={geometry['good_FPR']}")
    print(f"aggregate_max_shuffle_margin={max(shuffle_margins)}")
    print(f"best_semantic_role={best_role['semantic_role']}")
    print(f"best_role_good_FPR={best_role['good_FPR']}")
    print(f"best_role_max_shuffle_margin={best_role['max_shuffle_margin']}")


if __name__ == "__main__":
    main()
