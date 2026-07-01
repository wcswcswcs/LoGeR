#!/usr/bin/env python3
"""Fixed Phase6 repair search for the v94 object-source counterfactual branch."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnose_v94_phase6_object_source_counterfactual import (  # noqa: E402
    ROOT,
    V93_CF_EFFECT_ROWS,
    V93_OBJECT_ROWS,
    apply_policy,
    build_atoms,
    counterfactual_metrics,
    effect_rows,
    f,
    object_order_rotate,
    quantile,
    read_csv_rows,
    read_json,
    regime_rotate,
    rotate,
    same_count_random,
    write_csv,
    write_json,
)


def phase5_metric(rows: list[dict[str, Any]], mask: list[bool], policy: str) -> dict[str, Any]:
    labelled = [idx for idx, row in enumerate(rows) if row.get("base_case_type") in {"bad", "good"}]
    bad = [idx for idx in labelled if rows[idx].get("base_case_type") == "bad"]
    good = [idx for idx in labelled if rows[idx].get("base_case_type") == "good"]
    bad_hits = [idx for idx in bad if mask[idx]]
    good_hits = [idx for idx in good if mask[idx]]
    positives = [idx for idx in labelled if mask[idx]]
    bad_recall = len(bad_hits) / len(bad) if bad else 0.0
    good_fpr = len(good_hits) / len(good) if good else 0.0
    balanced_accuracy = 0.5 * (bad_recall + 1.0 - good_fpr)
    controls = {
        "same_count_rot7": rotate(mask, 7),
        "object_order_rot5": object_order_rotate(rows, mask, 5),
        "regime_rot1": regime_rotate(rows, mask, 1),
        "regime_rot2": regime_rotate(rows, mask, 2),
    }
    control_bas = {}
    for name, control_mask in controls.items():
        control_bad_hits = [idx for idx in bad if control_mask[idx]]
        control_good_hits = [idx for idx in good if control_mask[idx]]
        control_bad_recall = len(control_bad_hits) / len(bad) if bad else 0.0
        control_good_fpr = len(control_good_hits) / len(good) if good else 0.0
        control_bas[name] = 0.5 * (control_bad_recall + 1.0 - control_good_fpr)
    margins = [balanced_accuracy - value for value in control_bas.values()]
    min_margin = min(margins) if margins else 0.0
    return {
        "policy": policy,
        "phase5_bad_recall": bad_recall,
        "phase5_good_FPR": good_fpr,
        "phase5_loso_positive_folds": len({rows[idx].get("seq") for idx in positives}),
        "phase5_balanced_accuracy": balanced_accuracy,
        "phase5_min_control_margin": min_margin,
        "phase5_bad_hits": ",".join(str(rows[idx].get("pair_id")) for idx in bad_hits),
        "phase5_good_hits": ",".join(str(rows[idx].get("pair_id")) for idx in good_hits),
        "phase5_gate_pass": bool(
            bad_recall >= 0.60
            and good_fpr <= 0.25
            and len({rows[idx].get("seq") for idx in positives}) >= 3
            and min_margin >= 0.05
        ),
    }


def enrich_counterfactual_gates(rows: list[dict[str, Any]], mask: list[bool], policy: str) -> dict[str, Any]:
    actual = counterfactual_metrics(rows, mask, policy)
    control_masks = {
        "CONTROL_same_count_random": same_count_random(rows, mask),
        "CONTROL_object_order_rot5": object_order_rotate(rows, mask, 5),
        "CONTROL_regime_rot1": regime_rotate(rows, mask, 1),
        "CONTROL_regime_rot2": regime_rotate(rows, mask, 2),
        "CONTROL_same_count_rot7": rotate(mask, 7),
    }
    controls = [counterfactual_metrics(rows, control_mask, name) for name, control_mask in control_masks.items()]
    control_best_bad_ratio = max(
        [
            control.get("bad_median_residual_improvement_ratio")
            for control in controls
            if control.get("bad_median_residual_improvement_ratio") is not None
        ],
        default=None,
    )
    actual_bad_ratio = actual.get("bad_median_residual_improvement_ratio")
    actual_minus_best_control = (
        float(actual_bad_ratio - control_best_bad_ratio)
        if actual_bad_ratio is not None and control_best_bad_ratio is not None
        else None
    )
    good_ratio = actual.get("good_median_residual_worsen_ratio")
    good_max_ratio = actual.get("good_max_residual_worsen_ratio")
    actual["phase6_bad_residual_improvement_gate"] = actual_bad_ratio is not None and actual_bad_ratio >= 0.10
    actual["phase6_good_median_protection_gate"] = good_ratio is not None and good_ratio <= 0.02
    actual["phase6_good_catastrophic_worsen_absent"] = good_max_ratio is not None and good_max_ratio <= 0.20
    actual["phase6_sequence_coverage_gate"] = int(actual.get("sequence_coverage") or 0) >= 3
    actual["phase6_actual_minus_best_control"] = actual_minus_best_control
    actual["phase6_beats_best_control_gate"] = (
        actual_minus_best_control is not None and actual_minus_best_control >= 0.05
    )
    actual["phase6_counterfactual_gate_pass"] = bool(
        actual["phase6_bad_residual_improvement_gate"]
        and actual["phase6_good_median_protection_gate"]
        and actual["phase6_good_catastrophic_worsen_absent"]
        and actual["phase6_beats_best_control_gate"]
        and actual["phase6_sequence_coverage_gate"]
    )
    return actual


def and_mask(*masks: list[bool]) -> list[bool]:
    return [all(values) for values in zip(*masks)]


def not_mask(mask: list[bool]) -> list[bool]:
    return [not value for value in mask]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--object-identity-rows", type=Path, default=V93_OBJECT_ROWS)
    parser.add_argument("--v93-counterfactual-effect-rows", type=Path, default=V93_CF_EFFECT_ROWS)
    parser.add_argument("--phase5-object-summary", type=Path, default=ROOT / "phase5_object_source_extension/phase5_object_source_extension_summary.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "phase6_object_source_repair_search")
    args = parser.parse_args()

    phase5_rows = read_csv_rows(args.root / "phase5_semantic_carrier_alignment/semantic_carrier_alignment_rows.csv")
    phase5_by_pair = {row["pair_id"]: row for row in phase5_rows}
    object_rows = read_csv_rows(args.object_identity_rows)
    object_by_pair = {row["pair_id"]: row for row in object_rows}
    cf_rows = read_csv_rows(args.v93_counterfactual_effect_rows)
    selected_policy = (read_json(args.phase5_object_summary).get("selected_policy") or {}).get("policy")

    joined: list[dict[str, Any]] = []
    for cf_row in cf_rows:
        pair_id = str(cf_row.get("pair_id"))
        row: dict[str, Any] = dict(cf_row)
        for source in (phase5_by_pair.get(pair_id) or {}, object_by_pair.get(pair_id) or {}):
            for key, value in source.items():
                row.setdefault(key, value)
        joined.append(row)

    thresholds = {
        "boundary_global_cross_ratio_q75": quantile([f(row.get("boundary_global_cross_ratio")) for row in object_rows], 0.75),
        "boundary_new_id_ratio_q75": quantile([f(row.get("boundary_new_id_ratio")) for row in object_rows], 0.75),
        "object_boundary_ratio_q75": quantile([f(row.get("object_boundary_ratio")) for row in object_rows], 0.75),
        "radio_boundary_mean_q75": quantile([f(row.get("radio_boundary_mean")) for row in object_rows], 0.75),
    }
    atoms = build_atoms(joined, thresholds)
    sem_lowobs = atoms["SEM_LOWOBS"]
    sem_multimode = [row.get("semantic_evidence_type") == "SEM_MULTIMODE_UNSAFE" for row in joined]
    trace_delta_positive = [f(row.get("native_merge_residual_delta")) > 0.0 for row in joined]
    trace_delta_nonnegative = [f(row.get("native_merge_residual_delta")) >= 0.0 for row in joined]
    boundary_update_q75 = quantile([f(row.get("native_boundary_update_norm")) for row in joined], 0.75)
    boundary_update_high = [f(row.get("native_boundary_update_norm")) >= boundary_update_q75 for row in joined]

    base_policies = [
        selected_policy,
        "GLOBAL_CROSS_GE_Q75_OR_SEM_INVALID",
        "GLOBAL_NEW_GE_Q75_OR_SEM_INVALID",
        "GLOBAL_CROSS_GE_Q75_OR_SEM_INVALID_OR_SEM_LOWOBS",
        "GLOBAL_NEW_GE_Q75_OR_SEM_INVALID_OR_SEM_LOWOBS",
        "GLOBAL_CROSS_GE_Q75_OR_SEM_INVALID_OR_SEM_WEAK_CONTEXT",
        "GLOBAL_NEW_GE_Q75_OR_SEM_INVALID_OR_SEM_WEAK_CONTEXT",
    ]
    base_policies = [policy for idx, policy in enumerate(base_policies) if policy and policy not in base_policies[:idx]]

    candidate_masks: dict[str, list[bool]] = {}
    for policy in base_policies:
        try:
            base_mask = apply_policy(policy, atoms)
        except ValueError:
            continue
        candidate_masks[policy] = base_mask
        guards = {
            "__AND_TRACE_DELTA_POSITIVE": trace_delta_positive,
            "__AND_TRACE_DELTA_NONNEGATIVE": trace_delta_nonnegative,
            "__AND_NOT_SEM_LOWOBS": not_mask(sem_lowobs),
            "__AND_NOT_SEM_MULTIMODE": not_mask(sem_multimode),
            "__AND_NOT_LOWOBS_NOT_MULTIMODE": and_mask(not_mask(sem_lowobs), not_mask(sem_multimode)),
            "__AND_BOUNDARY_UPDATE_GE_Q75": boundary_update_high,
            "__AND_TRACE_POSITIVE_BOUNDARY_UPDATE_GE_Q75": and_mask(trace_delta_positive, boundary_update_high),
        }
        for suffix, guard in guards.items():
            candidate_masks[f"{policy}{suffix}"] = and_mask(base_mask, guard)

    metric_rows: list[dict[str, Any]] = []
    for policy, mask in candidate_masks.items():
        p5 = phase5_metric(joined, mask, policy)
        p6 = enrich_counterfactual_gates(joined, mask, policy)
        row = {**p5, **p6}
        row["repair_candidate_gate_pass"] = bool(row["phase5_gate_pass"] and row["phase6_counterfactual_gate_pass"])
        metric_rows.append(row)
    metric_rows = sorted(
        metric_rows,
        key=lambda row: (
            bool(row["repair_candidate_gate_pass"]),
            bool(row["phase6_counterfactual_gate_pass"]),
            f(row.get("bad_median_residual_improvement_ratio")),
            -f(row.get("good_max_residual_worsen_ratio")),
            bool(row["phase5_gate_pass"]),
            f(row["phase5_bad_recall"]),
            -f(row["phase5_good_FPR"]),
            f(row["phase5_min_control_margin"]),
        ),
        reverse=True,
    )
    best = metric_rows[0] if metric_rows else {}
    best_mask = candidate_masks.get(str(best.get("policy")), []) if best else []
    passing = [row for row in metric_rows if row["repair_candidate_gate_pass"]]
    phase5_passing = [row for row in metric_rows if row["phase5_gate_pass"]]
    phase6_passing = [row for row in metric_rows if row["phase6_counterfactual_gate_pass"]]
    summary = {
        "phase": "Phase6_object_source_repair_search",
        "diagnostic_only": True,
        "candidate_count": len(metric_rows),
        "repair_candidate_gate_pass": bool(passing),
        "passing_candidate_count": len(passing),
        "phase5_passing_candidate_count": len(phase5_passing),
        "phase6_passing_candidate_count": len(phase6_passing),
        "best_candidate": best,
        "boundary_update_norm_q75": boundary_update_q75,
        "thresholds": thresholds,
        "runtime_action_allowed": bool(passing),
        "ttt_allowed": False,
        "note": "Fixed no-training repair guards were tested once; trace-delta guards are diagnostic and still require replay before runtime.",
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "repair_policy_metrics.csv", metric_rows)
    write_csv(args.out_dir / "best_candidate_effect_rows.csv", effect_rows(joined, best_mask, str(best.get("policy"))))
    write_json(args.out_dir / "phase6_object_source_repair_search_summary.json", summary)
    (args.out_dir / "analysis.md").write_text(
        "\n".join(
            [
                "# ACL2 v94 Phase6 Object-Source Repair Search",
                "",
                f"- candidate_count: `{len(metric_rows)}`",
                f"- repair_candidate_gate_pass: `{summary['repair_candidate_gate_pass']}`",
                f"- passing_candidate_count: `{len(passing)}`",
                f"- phase5_passing_candidate_count: `{len(phase5_passing)}`",
                f"- phase6_passing_candidate_count: `{len(phase6_passing)}`",
                f"- best_candidate: `{best.get('policy')}`",
                f"- best_bad_median_residual_improvement_ratio: `{best.get('bad_median_residual_improvement_ratio')}`",
                f"- best_good_max_residual_worsen_ratio: `{best.get('good_max_residual_worsen_ratio')}`",
                "",
                "No runtime action is allowed unless a candidate passes both localization and counterfactual gates.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"repair_candidate_gate_pass={summary['repair_candidate_gate_pass']}")
    print(f"candidate_count={summary['candidate_count']}")
    print(f"passing_candidate_count={summary['passing_candidate_count']}")
    print(f"phase5_passing_candidate_count={summary['phase5_passing_candidate_count']}")
    print(f"phase6_passing_candidate_count={summary['phase6_passing_candidate_count']}")
    print(f"best_candidate={best.get('policy')}")
    print(f"best_bad_median_residual_improvement_ratio={best.get('bad_median_residual_improvement_ratio')}")
    print(f"best_good_max_residual_worsen_ratio={best.get('good_max_residual_worsen_ratio')}")
    print(f"runtime_action_allowed={summary['runtime_action_allowed']}")


if __name__ == "__main__":
    main()
