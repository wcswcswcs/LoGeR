#!/usr/bin/env python3
"""Audit cache/top-k identity and query-head re-entry for ACL2 v101.

This read-only Outcome-D follow-up summarizes v97/v98 diagnostic carrier
signals and all already measured v98 Stage7f/Stage7h action-pilot variants.
It does not authorize runtime action.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any


V97_ROOT = Path("results/acl2_v97tf_semantic_scale_evidence_gauge_safe_memory_control")
V98_ROOT = Path("results/acl2_v98tf_semantic_guided_swa_cache_topk_gauge_safe_memory_control")
V101_ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
OUT = V101_ROOT / "final_decision"

ACTION_GATE_RULE = (
    "case_count>=6 and failed_job_count==0 and improved_cases>=4 and worse_cases<=2 "
    "and median_improvement_ratio>=0.05"
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def fnum(value: Any) -> float | None:
    try:
        out = float(str(value))
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def best_gate_row(rows: list[dict[str, str]]) -> dict[str, str]:
    if not rows:
        return {}

    def key(row: dict[str, str]) -> tuple[int, float, float, float]:
        return (
            1 if truthy(row.get("gate_pass")) else 0,
            fnum(row.get("balanced_accuracy")) or -1e99,
            fnum(row.get("abs_corr_L3_handoff_transfer_penalty")) or -1e99,
            -(fnum(row.get("good_FPR")) or 1e99),
        )

    return max(rows, key=key)


def signal_rows() -> list[dict[str, Any]]:
    v97_e2 = read_json(V97_ROOT / "trackE2_swa_carrier_search_beyond_route_mass" / "summary.json")
    v97_k = read_json(V97_ROOT / "trackK_semantic_scale_evidence_eligibility" / "summary.json")
    v98_e = read_json(V98_ROOT / "stage7e_ttt_stable_anchor_id_hook" / "summary.json")
    v98_g = read_json(V98_ROOT / "stage7g_anchor_id_query_head_risk_attribution" / "summary.json")
    stage7e_best = best_gate_row(read_csv(V98_ROOT / "stage7e_ttt_stable_anchor_id_hook" / "cue_control_metrics.csv"))
    stage7g_best = best_gate_row(
        read_csv(V98_ROOT / "stage7g_anchor_id_query_head_risk_attribution" / "cue_control_metrics.csv")
    )
    v97_e2_best = (v97_e2.get("per_head_top_passed_carriers") or [{}])[0]

    return [
        {
            "source_id": "v97_trackE2_cache_kv_stability",
            "gate_pass": v97_e2.get("gate_pass", ""),
            "runtime_action_allowed": v97_e2.get("runtime_action_allowed", ""),
            "best_cue": ",".join(v97_e2.get("passed_carriers", [])),
            "balanced_accuracy": "",
            "bad_recall": "",
            "good_FPR": "",
            "abs_corr_L3": "",
            "selected_or_passed_count": len(v97_e2.get("passed_carriers", [])),
            "key_evidence": json.dumps(
                {
                    "carrier_row_count": v97_e2.get("carrier_row_count"),
                    "per_head_carrier_gate_pass_count": v97_e2.get("per_head_carrier_gate_pass_count"),
                    "top_per_head_carrier": v97_e2_best.get("carrier_metric"),
                    "top_per_head_BA": v97_e2_best.get("best_balanced_accuracy"),
                },
                sort_keys=True,
            ),
        },
        {
            "source_id": "v97_trackK_swa_cache_eligibility",
            "gate_pass": v97_k.get("any_eligibility_cue_gate_pass", ""),
            "runtime_action_allowed": v97_k.get("runtime_action_allowed", ""),
            "best_cue": v97_k.get("best_cache_carrier_score_name", ""),
            "balanced_accuracy": "",
            "bad_recall": "",
            "good_FPR": "",
            "abs_corr_L3": "",
            "selected_or_passed_count": v97_k.get("cache_carrier_eligibility_gate_pass_count", ""),
            "key_evidence": json.dumps(
                {
                    "swa_cache_eligibility_gate_pass": v97_k.get("swa_cache_eligibility_gate_pass"),
                    "swa_cache_internal_diagnostic_gate_pass": v97_k.get(
                        "swa_cache_internal_diagnostic_gate_pass"
                    ),
                    "read_gate_pass": v97_k.get("read_gate_pass"),
                    "ttt_gate_pass": v97_k.get("ttt_gate_pass"),
                },
                sort_keys=True,
            ),
        },
        {
            "source_id": "v98_stage7e_anchor_id_hook",
            "gate_pass": v98_e.get("gate_pass", ""),
            "runtime_action_allowed": v98_e.get("runtime_action_allowed", ""),
            "best_cue": v98_e.get("best_cue", stage7e_best.get("cue_name", "")),
            "balanced_accuracy": stage7e_best.get("balanced_accuracy", ""),
            "bad_recall": v98_e.get("best_cue_bad_recall", stage7e_best.get("bad_recall", "")),
            "good_FPR": v98_e.get("best_cue_good_FPR", stage7e_best.get("good_FPR", "")),
            "abs_corr_L3": v98_e.get("best_cue_abs_corr_L3", stage7e_best.get("abs_corr_L3_handoff_transfer_penalty", "")),
            "selected_or_passed_count": stage7e_best.get("selected_case_count", ""),
            "key_evidence": json.dumps(
                {
                    "persistent_anchor_id_available": v98_e.get("persistent_anchor_id_available"),
                    "write_to_swa_topk_chain_case_count": v98_e.get("write_to_swa_topk_chain_case_count"),
                    "primary_blocker": v98_e.get("primary_blocker"),
                },
                sort_keys=True,
            ),
        },
        {
            "source_id": "v98_stage7g_query_head_risk",
            "gate_pass": v98_g.get("gate_pass", ""),
            "runtime_action_allowed": v98_g.get("runtime_action_allowed", ""),
            "best_cue": v98_g.get("best_cue", stage7g_best.get("cue_name", "")),
            "balanced_accuracy": stage7g_best.get("balanced_accuracy", ""),
            "bad_recall": v98_g.get("best_cue_bad_recall", stage7g_best.get("bad_recall", "")),
            "good_FPR": v98_g.get("best_cue_good_FPR", stage7g_best.get("good_FPR", "")),
            "abs_corr_L3": v98_g.get("best_cue_abs_corr_L3", stage7g_best.get("abs_corr_L3_handoff_transfer_penalty", "")),
            "selected_or_passed_count": v98_g.get("selected_case_count", stage7g_best.get("selected_case_count", "")),
            "key_evidence": json.dumps(
                {
                    "query_head_gate_pass": v98_g.get("query_head_gate_pass"),
                    "selective_action_mass_gate_pass": v98_g.get("selective_action_mass_gate_pass"),
                    "id_specific_risk_cue_gate_pass": v98_g.get("id_specific_risk_cue_gate_pass"),
                    "true_positive_cases": stage7g_best.get("true_positive_cases", ""),
                    "primary_blocker": v98_g.get("primary_blocker"),
                },
                sort_keys=True,
            ),
        },
    ]


def comparison_paths() -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    stage7f = V98_ROOT / "stage7f_action_pilot_variant_comparison.csv"
    if stage7f.exists():
        paths.append(("stage7f_anchor_gate", stage7f))
    for path in sorted(V98_ROOT.glob("stage7h_query_soft_*_comparison.csv")):
        paths.append(("stage7h_query_soft", path))
    return paths


def action_variant_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case_rows: list[dict[str, Any]] = []
    by_variant: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for family, path in comparison_paths():
        for row in read_csv(path):
            variant = row.get("variant", path.stem.replace("_comparison", ""))
            out = {
                "action_family": family,
                "source_csv": str(path),
                "variant": variant,
                "case_id": row.get("case_id", ""),
                "seq": row.get("seq", ""),
                "returncode": row.get("returncode", "0"),
                "baseline_aligned_ate_rmse_m": row.get("baseline_aligned_ate_rmse_m", ""),
                "action_aligned_ate_rmse_m": row.get("action_aligned_ate_rmse_m", ""),
                "action_minus_baseline_ate_rmse_m": row.get("action_minus_baseline_ate_rmse_m", ""),
                "improvement_ratio_vs_baseline": row.get("improvement_ratio_vs_baseline", ""),
                "baseline_final_error_m": row.get("baseline_final_error_m", ""),
                "action_final_error_m": row.get("action_final_error_m", ""),
                "action_minus_baseline_final_error_m": row.get("action_minus_baseline_final_error_m", ""),
                "selected_frac": row.get("gate_selected_frac_mean", row.get("mean_query_soft_selected_frac", "")),
                "stable_anchor_preservation_ratio": row.get("mean_attention_mass_stable_anchor_preservation_ratio", ""),
            }
            case_rows.append(out)
            by_variant.setdefault((family, variant), []).append(out)

    variant_rows: list[dict[str, Any]] = []
    for (family, variant), rows in by_variant.items():
        deltas = [v for row in rows if (v := fnum(row.get("action_minus_baseline_ate_rmse_m"))) is not None]
        ratios = [v for row in rows if (v := fnum(row.get("improvement_ratio_vs_baseline"))) is not None]
        final_deltas = [
            v for row in rows if (v := fnum(row.get("action_minus_baseline_final_error_m"))) is not None
        ]
        returncodes = [int(fnum(row.get("returncode")) or 0) for row in rows]
        improved = sum(1 for value in deltas if value < 0.0)
        worse = sum(1 for value in deltas if value > 0.0)
        same = len(deltas) - improved - worse
        failed = sum(1 for value in returncodes if value != 0)
        median_ratio = median(ratios) if ratios else ""
        mean_ratio = mean(ratios) if ratios else ""
        median_delta = median(deltas) if deltas else ""
        mean_delta = mean(deltas) if deltas else ""
        median_final_delta = median(final_deltas) if final_deltas else ""
        gate_pass = (
            len(rows) >= 6
            and failed == 0
            and improved >= 4
            and worse <= 2
            and median_ratio != ""
            and float(median_ratio) >= 0.05
        )
        variant_rows.append(
            {
                "action_family": family,
                "variant": variant,
                "case_count": len(rows),
                "failed_job_count": failed,
                "improved_ate_case_count": improved,
                "worse_ate_case_count": worse,
                "same_ate_case_count": same,
                "median_improvement_ratio_vs_baseline": median_ratio,
                "mean_improvement_ratio_vs_baseline": mean_ratio,
                "median_action_minus_baseline_ate_rmse_m": median_delta,
                "mean_action_minus_baseline_ate_rmse_m": mean_delta,
                "median_action_minus_baseline_final_error_m": median_final_delta,
                "max_improvement_ratio_vs_baseline": max(ratios) if ratios else "",
                "min_improvement_ratio_vs_baseline": min(ratios) if ratios else "",
                "gate_pass": gate_pass,
                "gate_rule": ACTION_GATE_RULE,
                "source_csvs": ";".join(sorted({row["source_csv"] for row in rows})),
            }
        )

    variant_rows.sort(
        key=lambda row: (
            not bool(row["gate_pass"]),
            -(fnum(row.get("median_improvement_ratio_vs_baseline")) or -1e99),
            -int(row["improved_ate_case_count"]),
            int(row["worse_ate_case_count"]),
        )
    )
    return variant_rows, case_rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    signals = signal_rows()
    variants, cases = action_variant_rows()
    best_variant = variants[0] if variants else {}

    write_csv(
        OUT / "cache_topk_identity_query_reentry_signal_summary.csv",
        signals,
        [
            "source_id",
            "gate_pass",
            "runtime_action_allowed",
            "best_cue",
            "balanced_accuracy",
            "bad_recall",
            "good_FPR",
            "abs_corr_L3",
            "selected_or_passed_count",
            "key_evidence",
        ],
    )
    write_csv(
        OUT / "cache_topk_identity_query_reentry_action_variant_metrics.csv",
        variants,
        [
            "action_family",
            "variant",
            "case_count",
            "failed_job_count",
            "improved_ate_case_count",
            "worse_ate_case_count",
            "same_ate_case_count",
            "median_improvement_ratio_vs_baseline",
            "mean_improvement_ratio_vs_baseline",
            "median_action_minus_baseline_ate_rmse_m",
            "mean_action_minus_baseline_ate_rmse_m",
            "median_action_minus_baseline_final_error_m",
            "max_improvement_ratio_vs_baseline",
            "min_improvement_ratio_vs_baseline",
            "gate_pass",
            "gate_rule",
            "source_csvs",
        ],
    )
    write_csv(
        OUT / "cache_topk_identity_query_reentry_case_outcomes.csv",
        cases,
        [
            "action_family",
            "source_csv",
            "variant",
            "case_id",
            "seq",
            "returncode",
            "baseline_aligned_ate_rmse_m",
            "action_aligned_ate_rmse_m",
            "action_minus_baseline_ate_rmse_m",
            "improvement_ratio_vs_baseline",
            "baseline_final_error_m",
            "action_final_error_m",
            "action_minus_baseline_final_error_m",
            "selected_frac",
            "stable_anchor_preservation_ratio",
        ],
    )

    v98_final = read_json(V98_ROOT / "final_decision" / "summary.json")
    passing = [row for row in variants if row["gate_pass"]]
    summary = {
        "schema": "acl2_v101_cache_topk_identity_query_reentry_audit_v1",
        "v97_final_taxonomy": read_json(V97_ROOT / "final_decision" / "summary.json").get("final_taxonomy"),
        "v98_final_taxonomy": v98_final.get("final_taxonomy"),
        "diagnostic_signal_count": len(signals),
        "diagnostic_gate_pass_count": sum(1 for row in signals if truthy(row.get("gate_pass"))),
        "action_variant_count": len(variants),
        "action_case_outcome_row_count": len(cases),
        "action_variant_gate_pass_count": len(passing),
        "best_action_family": best_variant.get("action_family", ""),
        "best_action_variant": best_variant.get("variant", ""),
        "best_action_gate_pass": bool(best_variant.get("gate_pass", False)),
        "best_action_improved_ate_case_count": best_variant.get("improved_ate_case_count", ""),
        "best_action_worse_ate_case_count": best_variant.get("worse_ate_case_count", ""),
        "best_action_median_improvement_ratio_vs_baseline": best_variant.get(
            "median_improvement_ratio_vs_baseline", ""
        ),
        "stage7f_best_variant": v98_final.get("stage7f_best_variant"),
        "stage7f_best_variant_median_improvement_ratio_vs_baseline": v98_final.get(
            "stage7f_best_variant_median_improvement_ratio_vs_baseline"
        ),
        "stage7h_variant": v98_final.get("stage7h_variant"),
        "stage7h_median_improvement_ratio_vs_baseline": v98_final.get(
            "stage7h_median_improvement_ratio_vs_baseline"
        ),
        "runtime_action_allowed": False,
        "full_validation_allowed": False,
        "v101_goal_achieved": False,
        "claim": (
            "cache/top-k and query-head identity cues remain diagnostic, but all audited v98 measured "
            "Stage7f/Stage7h action-pilot variants fail the 6-case action gate."
        ),
    }
    with (OUT / "cache_topk_identity_query_reentry_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    report = [
        "# Cache/Top-K Identity Query Re-entry Audit",
        "",
        "This audit summarizes v97/v98 diagnostic cache/top-k and query-head signals plus already measured v98 action pilots.",
        "It is diagnostic only and does not authorize runtime action.",
        "",
        "## Summary",
        "",
        f"- diagnostic signals: `{summary['diagnostic_signal_count']}`",
        f"- diagnostic gate-pass signals: `{summary['diagnostic_gate_pass_count']}`",
        f"- action variants audited: `{summary['action_variant_count']}`",
        f"- action case rows: `{summary['action_case_outcome_row_count']}`",
        f"- action variant gate-pass count: `{summary['action_variant_gate_pass_count']}`",
        f"- best action family: `{summary['best_action_family']}`",
        f"- best action variant: `{summary['best_action_variant']}`",
        f"- best improved/worse cases: `{summary['best_action_improved_ate_case_count']}` / `{summary['best_action_worse_ate_case_count']}`",
        f"- best median improvement ratio: `{summary['best_action_median_improvement_ratio_vs_baseline']}`",
        "",
        "## Interpretation",
        "",
    ]
    if passing:
        report.append(
            "At least one existing action-pilot variant passes the offline action gate, but runtime remains disallowed until a fresh predeclared measured-control rerun."
        )
    else:
        report.append(
            "The carrier/query-head diagnostic signals are real, but the existing Stage7f/Stage7h action forms do not translate them into robust ATE improvement. Re-entering this route requires a new predeclared action design and controls, not another replay of the same anchor-gate/query-soft variants."
        )
    report.append("")
    report.append("Runtime action remains disallowed.")
    (OUT / "cache_topk_identity_query_reentry_report.md").write_text("\n".join(report) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
