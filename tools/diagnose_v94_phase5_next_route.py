#!/usr/bin/env python3
"""Diagnose v94 Phase5 next-route repair space without promoting a method."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v94tf_semantic_gauge_failure_localization_causal_memory_control")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def f(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def quantile(values: list[float], q: float) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return float("nan")
    pos = (len(finite) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return finite[int(pos)]
    frac = pos - lo
    return finite[lo] * (1.0 - frac) + finite[hi] * frac


def metric(
    rows: list[dict[str, Any]], pred: list[bool], policy: str, kind: str, policy_note: str = ""
) -> dict[str, Any]:
    labelled = [idx for idx, row in enumerate(rows) if row.get("case_label_offline_only") in {"bad", "good"}]
    bad = [idx for idx in labelled if rows[idx].get("case_label_offline_only") == "bad"]
    good = [idx for idx in labelled if rows[idx].get("case_label_offline_only") == "good"]
    bad_hits = [idx for idx in bad if pred[idx]]
    good_hits = [idx for idx in good if pred[idx]]
    bad_recall = len(bad_hits) / len(bad) if bad else 0.0
    good_fpr = len(good_hits) / len(good) if good else 0.0
    pos_indices = [idx for idx in labelled if pred[idx]]
    out = {
        "policy": policy,
        "kind": kind,
        "policy_note": policy_note,
        "labelled_rows": len(labelled),
        "bad_rows": len(bad),
        "good_rows": len(good),
        "positive_rows": len(pos_indices),
        "bad_recall": bad_recall,
        "good_FPR": good_fpr,
        "balanced_accuracy": 0.5 * (bad_recall + 1.0 - good_fpr),
        "loso_positive_folds": len({rows[idx].get("seq") for idx in pos_indices}),
        "positive_sequences": ",".join(sorted({str(rows[idx].get("seq")) for idx in pos_indices})),
        "bad_hits": ",".join(str(rows[idx].get("pair_id")) for idx in bad_hits),
        "good_hits": ",".join(str(rows[idx].get("pair_id")) for idx in good_hits),
    }
    out["gate_like_pass"] = bool(
        out["bad_recall"] >= 0.60 and out["good_FPR"] <= 0.25 and out["loso_positive_folds"] >= 3
    )
    return out


def bools(rows: list[dict[str, Any]], func) -> list[bool]:
    return [bool(func(row)) for row in rows]


def or_pred(*preds: list[bool]) -> list[bool]:
    return [any(values) for values in zip(*preds)]


def and_pred(*preds: list[bool]) -> list[bool]:
    return [all(values) for values in zip(*preds)]


def not_pred(pred: list[bool]) -> list[bool]:
    return [not value for value in pred]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "phase5_next_route_diagnostic")
    args = parser.parse_args()

    phase1_summary = read_json(args.root / "phase1_boundary_failure_atlas/phase1_gate_summary.json")
    phase5_summary = read_json(args.root / "phase5_semantic_carrier_alignment/semantic_carrier_alignment_summary.json")
    phase1_rows = read_csv_rows(args.root / "phase1_boundary_failure_atlas/boundary_failure_rows.csv")
    rows = read_csv_rows(args.root / "phase5_semantic_carrier_alignment/semantic_carrier_alignment_rows.csv")
    labelled_rows = [row for row in rows if row.get("case_label_offline_only") in {"bad", "good"}]

    quantiles = phase1_summary.get("quantiles", {})
    if not isinstance(quantiles, dict):
        quantiles = {}
    thresholds = {
        "merge_residual_after_q75": f(quantiles.get("merge_residual_after_q75")),
        "raw_overlap_residual_q75": f(quantiles.get("raw_overlap_residual_q75")),
        "raw_overlap_residual_q95": f(quantiles.get("raw_overlap_residual_q95")),
        "boundary_update_norm_q75": f(quantiles.get("boundary_update_norm_q75")),
        "boundary_update_norm_q95": f(quantiles.get("boundary_update_norm_q95")),
        "abs_log_scale_jump_q75": f(quantiles.get("abs_log_scale_jump_q75")),
        "abs_log_scale_jump_q95": f(quantiles.get("abs_log_scale_jump_q95")),
    }
    if not math.isfinite(thresholds["merge_residual_after_q75"]):
        thresholds["merge_residual_after_q75"] = thresholds["raw_overlap_residual_q75"]
    if not math.isfinite(thresholds["merge_residual_after_q75"]):
        thresholds["merge_residual_after_q75"] = quantile([f(row.get("raw_overlap_residual")) for row in phase1_rows], 0.75)
    if not math.isfinite(thresholds["raw_overlap_residual_q75"]):
        thresholds["raw_overlap_residual_q75"] = quantile([f(row.get("raw_overlap_residual")) for row in phase1_rows], 0.75)
    if not math.isfinite(thresholds["raw_overlap_residual_q95"]):
        thresholds["raw_overlap_residual_q95"] = quantile([f(row.get("raw_overlap_residual")) for row in phase1_rows], 0.95)

    atoms: dict[str, list[bool]] = {
        "SEM_INVALID": bools(labelled_rows, lambda row: row.get("semantic_evidence_type") == "SEM_INVALID_BOUNDARY"),
        "SEM_WEAK_CONTEXT": bools(labelled_rows, lambda row: row.get("semantic_evidence_type") == "SEM_WEAK_CONTEXT"),
        "SEM_LOWOBS": bools(labelled_rows, lambda row: row.get("semantic_evidence_type") == "SEM_LOWOBS_ABSTAIN"),
        "SEM_MULTIMODE": bools(labelled_rows, lambda row: row.get("semantic_evidence_type") == "SEM_MULTIMODE_UNSAFE"),
        "S_INVALID_GE_075": bools(labelled_rows, lambda row: f(row.get("S_invalid")) >= 0.75),
        "S_CONTEXT_GE_075": bools(labelled_rows, lambda row: f(row.get("S_context")) >= 0.75),
        "S_CONTEXT_GE_055": bools(labelled_rows, lambda row: f(row.get("S_context")) >= 0.55),
        "S_LOWOBS_GE_065": bools(labelled_rows, lambda row: f(row.get("S_lowobs")) >= 0.65),
        "S_MULTI_GE_080": bools(labelled_rows, lambda row: f(row.get("S_multi")) >= 0.80),
        "RESIDUAL_GE_PHASE1_Q75": bools(
            labelled_rows,
            lambda row: f(row.get("carrier_error_merge_residual_after_abs"))
            >= thresholds["merge_residual_after_q75"],
        ),
        "RESIDUAL_GE_PHASE1_Q95": bools(
            labelled_rows,
            lambda row: f(row.get("carrier_error_merge_residual_after_abs"))
            >= thresholds["raw_overlap_residual_q95"],
        ),
        "BOUNDARY_GE_PHASE1_Q75": bools(
            labelled_rows,
            lambda row: f(row.get("carrier_error_boundary_update_norm"))
            >= thresholds["boundary_update_norm_q75"],
        ),
        "BOUNDARY_GE_PHASE1_Q95": bools(
            labelled_rows,
            lambda row: f(row.get("carrier_error_boundary_update_norm"))
            >= thresholds["boundary_update_norm_q95"],
        ),
        "SCALE_GE_PHASE1_Q75": bools(
            labelled_rows,
            lambda row: f(row.get("carrier_error_abs_log_scale_jump_runtime"))
            >= thresholds["abs_log_scale_jump_q75"],
        ),
        "SCALE_GE_PHASE1_Q95": bools(
            labelled_rows,
            lambda row: f(row.get("carrier_error_abs_log_scale_jump_runtime"))
            >= thresholds["abs_log_scale_jump_q95"],
        ),
    }

    policy_rows: list[dict[str, Any]] = []
    for name, pred in atoms.items():
        policy_rows.append(metric(labelled_rows, pred, name, "atom", "single fixed semantic/carrier atom"))

    for (name_a, pred_a), (name_b, pred_b) in itertools.combinations(atoms.items(), 2):
        policy_rows.append(metric(labelled_rows, or_pred(pred_a, pred_b), f"{name_a}_OR_{name_b}", "or2"))
        policy_rows.append(metric(labelled_rows, and_pred(pred_a, pred_b), f"{name_a}_AND_{name_b}", "and2"))

    targeted_specs = [
        (
            "SEM_INVALID_OR_SEM_WEAK_CONTEXT",
            or_pred(atoms["SEM_INVALID"], atoms["SEM_WEAK_CONTEXT"]),
            "best low-FPR semantic expansion seen in prior final-report next-route",
        ),
        (
            "SEM_INVALID_OR_SEM_LOWOBS",
            or_pred(atoms["SEM_INVALID"], atoms["SEM_LOWOBS"]),
            "object/low-observability expansion candidate",
        ),
        (
            "SEM_INVALID_OR_SCALE_GE_PHASE1_Q75",
            or_pred(atoms["SEM_INVALID"], atoms["SCALE_GE_PHASE1_Q75"]),
            "plan carrier subfield repair: scale proxy q75, no label tuning",
        ),
        (
            "SCALE_GE_PHASE1_Q75_AND_NOT_SEM_MULTIMODE",
            and_pred(atoms["SCALE_GE_PHASE1_Q75"], not_pred(atoms["SEM_MULTIMODE"])),
            "good-FPR guard candidate using semantic role only",
        ),
        (
            "SCALE_GE_PHASE1_Q75_AND_WEAK_OR_LOWOBS_OR_INVALID",
            and_pred(
                atoms["SCALE_GE_PHASE1_Q75"],
                or_pred(atoms["SEM_INVALID"], atoms["SEM_WEAK_CONTEXT"], atoms["SEM_LOWOBS"]),
            ),
            "good-FPR guard candidate using fixed non-multimode roles",
        ),
        (
            "SEM_INVALID_OR_ANY_RESIDUAL_GE_PHASE1_Q75",
            or_pred(atoms["SEM_INVALID"], atoms["RESIDUAL_GE_PHASE1_Q75"]),
            "current canonical q75 carrier-event policy",
        ),
    ]
    for name, pred, note in targeted_specs:
        policy_rows.append(metric(labelled_rows, pred, name, "targeted", note))

    dedup: dict[str, dict[str, Any]] = {}
    for row in policy_rows:
        dedup[row["policy"]] = row
    policy_rows = sorted(
        dedup.values(),
        key=lambda row: (
            bool(row["gate_like_pass"]),
            f(row["bad_recall"]),
            -f(row["good_FPR"]),
            f(row["loso_positive_folds"]),
            -f(row["positive_rows"]),
        ),
        reverse=True,
    )

    current_policy = next(
        row for row in policy_rows if row["policy"] == "SEM_INVALID_OR_ANY_RESIDUAL_GE_PHASE1_Q75"
    )
    current_hits = set(filter(None, str(current_policy["bad_hits"]).split(",")))
    miss_rows: list[dict[str, Any]] = []
    for row in labelled_rows:
        if row.get("case_label_offline_only") != "bad" or str(row.get("pair_id")) in current_hits:
            continue
        miss_rows.append(
            {
                "pair_id": row.get("pair_id"),
                "seq": row.get("seq"),
                "semantic_evidence_type": row.get("semantic_evidence_type"),
                "failure_type_primary": row.get("failure_type_primary"),
                "S_invalid": row.get("S_invalid"),
                "S_context": row.get("S_context"),
                "S_lowobs": row.get("S_lowobs"),
                "S_multi": row.get("S_multi"),
                "carrier_error_merge_residual_after_abs": row.get("carrier_error_merge_residual_after_abs"),
                "carrier_error_boundary_update_norm": row.get("carrier_error_boundary_update_norm"),
                "carrier_error_abs_log_scale_jump_runtime": row.get("carrier_error_abs_log_scale_jump_runtime"),
                "miss_reason": "not_sem_invalid_and_merge_residual_below_phase1_q75",
            }
        )

    seq00_seq05_bad = [
        {
            "pair_id": row.get("pair_id"),
            "seq": row.get("seq"),
            "semantic_evidence_type": row.get("semantic_evidence_type"),
            "failure_type_primary": row.get("failure_type_primary"),
            "S_invalid": row.get("S_invalid"),
            "S_context": row.get("S_context"),
            "S_lowobs": row.get("S_lowobs"),
            "S_multi": row.get("S_multi"),
            "carrier_error_merge_residual_after_abs": row.get("carrier_error_merge_residual_after_abs"),
            "carrier_error_boundary_update_norm": row.get("carrier_error_boundary_update_norm"),
            "carrier_error_abs_log_scale_jump_runtime": row.get("carrier_error_abs_log_scale_jump_runtime"),
        }
        for row in labelled_rows
        if row.get("case_label_offline_only") == "bad" and str(row.get("seq")).zfill(2) in {"00", "05"}
    ]

    gate_like = [row for row in policy_rows if row["gate_like_pass"]]
    low_fpr = [row for row in policy_rows if f(row["good_FPR"]) <= 0.25]
    best_low_fpr = max(
        low_fpr,
        key=lambda row: (f(row["bad_recall"]), f(row["loso_positive_folds"]), -f(row["good_FPR"])),
        default={},
    )
    high_recall = [row for row in policy_rows if f(row["bad_recall"]) >= 0.60]
    best_high_recall = min(
        high_recall,
        key=lambda row: (f(row["good_FPR"]), -f(row["bad_recall"]), -f(row["loso_positive_folds"])),
        default={},
    )
    source_counts: dict[str, int] = {}
    for row in labelled_rows:
        key = str(row.get("semantic_source_level") or "")
        source_counts[key] = source_counts.get(key, 0) + 1

    summary = {
        "phase": "Phase5_next_route_diagnostic",
        "diagnostic_only": True,
        "method_promoted": False,
        "phase5_gate_pass_after_diagnostic": False,
        "input_phase5_gate_pass": phase5_summary.get("phase5_semantic_carrier_alignment_gate_pass"),
        "input_phase5_blocker": phase5_summary.get("blocker"),
        "labelled_rows": len(labelled_rows),
        "bad_rows": sum(row.get("case_label_offline_only") == "bad" for row in labelled_rows),
        "good_rows": sum(row.get("case_label_offline_only") == "good" for row in labelled_rows),
        "thresholds": thresholds,
        "semantic_source_level_counts_on_labelled_rows": source_counts,
        "policies_evaluated": len(policy_rows),
        "gate_like_policy_count": len(gate_like),
        "best_low_fpr_policy": best_low_fpr,
        "best_high_recall_policy": best_high_recall,
        "current_policy": current_policy,
        "current_policy_missed_bad_rows": len(miss_rows),
        "seq00_seq05_bad_rows": len(seq00_seq05_bad),
        "seq00_seq05_bad_rows_missed_by_current_policy": sum(
            row.get("pair_id") not in current_hits for row in seq00_seq05_bad
        ),
        "conclusion": (
            "No fixed semantic/carrier q75/q95 candidate in this diagnostic satisfies "
            "bad_recall>=0.60, good_FPR<=0.25, and LOSO>=3. "
            "Scale q75 recovers recall but has high good_FPR; semantic expansions keep FPR controlled but remain sparse."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "policy_metrics.csv", policy_rows)
    write_csv(args.out_dir / "current_policy_missed_bad_rows.csv", miss_rows)
    write_csv(args.out_dir / "seq00_seq05_bad_detail.csv", seq00_seq05_bad)
    write_json(args.out_dir / "phase5_next_route_diagnostic_summary.json", summary)

    analysis_lines = [
        "# ACL2 v94 Phase5 Next-Route Diagnostic",
        "",
        "This artifact is diagnostic-only. It does not promote counterfactual, runtime action, or TTT.",
        "",
        "## Summary",
        "",
        f"- policies_evaluated: `{summary['policies_evaluated']}`",
        f"- gate_like_policy_count: `{summary['gate_like_policy_count']}`",
        f"- current_policy_missed_bad_rows: `{summary['current_policy_missed_bad_rows']}`",
        f"- seq00_seq05_bad_rows_missed_by_current_policy: `{summary['seq00_seq05_bad_rows_missed_by_current_policy']}`",
        f"- best_low_fpr_policy: `{best_low_fpr.get('policy')}`",
        f"- best_low_fpr_bad_recall: `{best_low_fpr.get('bad_recall')}`",
        f"- best_low_fpr_good_FPR: `{best_low_fpr.get('good_FPR')}`",
        f"- best_low_fpr_loso: `{best_low_fpr.get('loso_positive_folds')}`",
        f"- best_high_recall_policy: `{best_high_recall.get('policy')}`",
        f"- best_high_recall_bad_recall: `{best_high_recall.get('bad_recall')}`",
        f"- best_high_recall_good_FPR: `{best_high_recall.get('good_FPR')}`",
        "",
        "## Conclusion",
        "",
        summary["conclusion"],
    ]
    (args.out_dir / "analysis.md").write_text("\n".join(analysis_lines) + "\n", encoding="utf-8")

    print(f"diagnostic_only={summary['diagnostic_only']}")
    print(f"policies_evaluated={summary['policies_evaluated']}")
    print(f"gate_like_policy_count={summary['gate_like_policy_count']}")
    print(f"best_low_fpr_policy={best_low_fpr.get('policy')}")
    print(f"best_low_fpr_bad_recall={best_low_fpr.get('bad_recall')}")
    print(f"best_low_fpr_good_FPR={best_low_fpr.get('good_FPR')}")
    print(f"best_low_fpr_loso={best_low_fpr.get('loso_positive_folds')}")
    print(f"best_high_recall_policy={best_high_recall.get('policy')}")
    print(f"best_high_recall_bad_recall={best_high_recall.get('bad_recall')}")
    print(f"best_high_recall_good_FPR={best_high_recall.get('good_FPR')}")
    print(f"current_policy_missed_bad_rows={summary['current_policy_missed_bad_rows']}")
    print(f"seq00_seq05_bad_rows_missed_by_current_policy={summary['seq00_seq05_bad_rows_missed_by_current_policy']}")
    print("phase5_gate_pass_after_diagnostic=False")


if __name__ == "__main__":
    main()
