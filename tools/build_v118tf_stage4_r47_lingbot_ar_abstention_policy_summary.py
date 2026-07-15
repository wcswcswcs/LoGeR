#!/usr/bin/env python3
"""Assemble ACL2 v118 R47 token-polarity abstention policy from existing runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
STAGE = RESULT_ROOT / "stage4_r47_lingbot_ar_abstention_policy_reused_stress"
OUT = STAGE / "summary"
DEV_ROOT = ROOT / "results/acl2_v116tf_fast_semantic_causal_memory_influence/task2_l2t/token_semantics"
HOLDOUT_TOKEN_ROOT = RESULT_ROOT / "stage4_r42_lingbot_ar_token_gated_oriented_source_value_holdout/token_semantics"
R45_ROWS = RESULT_ROOT / "stage4_r45_lingbot_ar_uniform_token_polarity_source_value/summary/stage4_r45_lingbot_ar_anchor_read_rows.csv"
R45_POLARITY = RESULT_ROOT / "stage4_r45_lingbot_ar_uniform_token_polarity_source_value/summary/stage4_r45_lingbot_ar_uniform_token_polarity_rows.csv"
R46_ROWS = RESULT_ROOT / "stage4_r46_lingbot_ar_uniform_token_polarity_reused_holdout/summary/stage4_r46_lingbot_ar_anchor_read_rows.csv"
R46_POLARITY = RESULT_ROOT / "stage4_r46_lingbot_ar_uniform_token_polarity_reused_holdout/summary/stage4_r46_lingbot_ar_uniform_token_polarity_reused_holdout_rows.csv"
SEQS = ("00", "02", "01", "05")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def token_root_for_seq(seq: str) -> Path:
    return DEV_ROOT if seq in {"00", "02"} else HOLDOUT_TOKEN_ROOT


def token_stats(seq: str) -> dict[str, Any]:
    root = token_root_for_seq(seq)
    filled = np.load(root / f"seq{seq}_filled.npy").astype(bool)
    out: dict[str, Any] = {
        "seq": seq,
        "token_weight_root": rel(root),
        "filled_mean": float(filled.mean()),
    }
    for channel in ("dynamic", "boundary", "lowtrust", "stable", "weak", "confidence"):
        arr = np.load(root / f"seq{seq}_{channel}.npy")
        vals = arr[filled] if filled.any() else arr.reshape(-1)
        vals = vals.astype("float64", copy=False)
        out[f"{channel}_mean"] = float(vals.mean())
        out[f"{channel}_p90"] = float(np.quantile(vals, 0.90))
        out[f"{channel}_p99"] = float(np.quantile(vals, 0.99))
    out["dynamic_plus_lowtrust_mean"] = out["dynamic_mean"] + out["lowtrust_mean"]
    out["weak_plus_lowtrust_mean"] = out["weak_mean"] + out["lowtrust_mean"]
    out["stable_to_weak_lowtrust"] = out["stable_mean"] / max(1e-9, out["weak_plus_lowtrust_mean"])
    return out


def source_rows() -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    rows = read_csv(R45_ROWS) + read_csv(R46_ROWS)
    polarity_rows = read_csv(R45_POLARITY) + read_csv(R46_POLARITY)
    polarity = {row["seq"]: row for row in polarity_rows}
    return rows, polarity


def metrics_by_seq(rows: list[dict[str, str]], polarity: dict[str, dict[str, str]]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for seq in SEQS:
        seq_rows = [row for row in rows if row["seq"] == seq]
        selected_mode = polarity[seq]["selected_token_weight_mode"]
        seq_out: dict[str, dict[str, Any]] = {}
        for row in seq_rows:
            role = row["role"]
            if role == "candidate":
                key = "risk" if selected_mode == "risk_suppress_plus_stable_x_frame" else "reverse"
            elif role == "token_opposite_polarity_control":
                key = "reverse" if selected_mode == "risk_suppress_plus_stable_x_frame" else "risk"
            elif role == "token_random_control":
                key = "random"
            else:
                continue
            seq_out[key] = {
                "source_role": role,
                "method": row["method"],
                "ate": fnum(row["ate"]),
                "rpe_rot": fnum(row["rpe_rot"]),
                "rpe_trans": fnum(row["rpe_trans"]),
                "rel_vs_default": fnum(row["ate_rel_improvement_vs_default"]),
            }
            seq_out["baseline"] = {
                "source_role": "baseline",
                "method": row["baseline_method"],
                "ate": fnum(row["baseline_ate"]),
                "rpe_rot": fnum(row["baseline_rpe_rot"]),
                "rpe_trans": fnum(row["baseline_rpe_trans"]),
                "rel_vs_default": 0.0,
            }
        out[seq] = seq_out
    return out


def choose_action(seq: str, corr: float, stats: dict[str, Any]) -> tuple[str, str]:
    ratio = fnum(stats["stable_to_weak_lowtrust"])
    if corr <= 0.0:
        return "reverse", "nonpositive_internal_semantic_corr"
    if corr >= 0.50:
        return "risk", "strong_positive_internal_semantic_corr"
    if ratio >= 0.20:
        return "reverse", "moderate_positive_corr_high_stable_to_weak_lowtrust"
    return "abstain", "moderate_positive_corr_low_stable_to_weak_lowtrust"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, polarity = source_rows()
    by_seq = metrics_by_seq(rows, polarity)
    stats_by_seq = {seq: token_stats(seq) for seq in SEQS}

    policy_rows: list[dict[str, Any]] = []
    selected_rels: list[float] = []
    active_rows: list[dict[str, Any]] = []
    for seq in SEQS:
        corr = fnum(polarity[seq]["internal_semantic_corr"])
        action, reason = choose_action(seq, corr, stats_by_seq[seq])
        selected_key = "baseline" if action == "abstain" else action
        opposite_key = (
            "baseline"
            if action == "abstain"
            else ("reverse" if action == "risk" else "risk")
        )
        random_key = "baseline" if action == "abstain" else "random"
        selected = by_seq[seq][selected_key]
        opposite = by_seq[seq][opposite_key]
        random = by_seq[seq][random_key]
        rel_value = fnum(selected["rel_vs_default"])
        selected_rels.append(rel_value)
        row = {
            "schema": "acl2_v118tf_stage4_r47_abstention_policy_row_v1",
            "seq": seq,
            "action": action,
            "selected_metric_key": selected_key,
            "selection_reason": reason,
            "internal_semantic_corr": corr,
            "stable_to_weak_lowtrust": stats_by_seq[seq]["stable_to_weak_lowtrust"],
            "dynamic_plus_lowtrust_mean": stats_by_seq[seq]["dynamic_plus_lowtrust_mean"],
            "selected_ate": selected["ate"],
            "baseline_ate": by_seq[seq]["baseline"]["ate"],
            "selected_rel_vs_default": rel_value,
            "opposite_control_ate": opposite["ate"],
            "opposite_candidate_minus_control": selected["ate"] - opposite["ate"],
            "random_control_ate": random["ate"],
            "random_candidate_minus_control": selected["ate"] - random["ate"],
            "selected_source_method": selected["method"],
            "opposite_source_method": opposite["method"],
            "random_source_method": random["method"],
        }
        policy_rows.append(row)
        if action != "abstain":
            active_rows.append(row)

    active_better_opposite = bool(active_rows) and all(
        fnum(row["opposite_candidate_minus_control"]) < 0.0 for row in active_rows
    )
    active_better_random = bool(active_rows) and all(
        fnum(row["random_candidate_minus_control"]) < 0.0 for row in active_rows
    )
    all_nonharm = all(value >= -1e-12 for value in selected_rels)
    median_rel = median(selected_rels)
    active_median_rel = median([fnum(row["selected_rel_vs_default"]) for row in active_rows]) if active_rows else 0.0
    selected_improvement_count = sum(1 for value in selected_rels if value > 0.0)
    abstain_count = sum(1 for row in policy_rows if row["action"] == "abstain")
    policy_gate = bool(active_rows) and active_better_opposite and active_better_random and all_nonharm and median_rel >= 0.03
    decision = (
        "AR_SOURCE_VALUE_SCALING_ABSTENTION_POLICY_REUSED_STRESS_SUPPORTED_REQUIRES_FRESH_VALIDATION"
        if policy_gate
        else "AR_SOURCE_VALUE_SCALING_ABSTENTION_POLICY_REUSED_STRESS_NO_GO"
    )

    policy_rows_path = OUT / "stage4_r47_lingbot_ar_abstention_policy_rows.csv"
    token_stats_path = OUT / "stage4_r47_lingbot_ar_token_stats_rows.csv"
    summary_path = OUT / "stage4_r47_lingbot_ar_abstention_policy_summary.json"
    report_path = OUT / "STAGE4_R47_LINGBOT_AR_ABSTENTION_POLICY_REPORT.md"
    write_csv(policy_rows_path, policy_rows)
    write_csv(token_stats_path, list(stats_by_seq.values()))
    summary = {
        "schema": "acl2_v118tf_stage4_r47_lingbot_ar_abstention_policy_summary_v1",
        "stage4_r47_decision": decision,
        "global_goal_achieved": False,
        "claim_level": "reused_holdout_policy_assembly_not_blind_success",
        "complete": True,
        "policy_gate": policy_gate,
        "all_sequences_nonharm": all_nonharm,
        "candidate_better_active_controls": active_better_opposite and active_better_random,
        "active_better_opposite": active_better_opposite,
        "active_better_random": active_better_random,
        "selected_improvement_count": selected_improvement_count,
        "abstain_count": abstain_count,
        "sequence_count": len(SEQS),
        "median_rel_vs_default": median_rel,
        "active_median_rel_vs_default": active_median_rel,
        "max_harm": abs(min(selected_rels)) if min(selected_rels) < 0 else 0.0,
        "rule": {
            "threshold_provenance": "post_r46_repair_hypothesis_from_reused_holdout_stress",
            "inputs": ["internal_semantic_corr", "stable_to_weak_lowtrust"],
            "logic": [
                "if corr <= 0: reverse",
                "elif corr >= 0.50: risk",
                "elif stable_to_weak_lowtrust >= 0.20: reverse",
                "else: abstain",
            ],
        },
        "boundary": (
            "R47 assembles an abstention policy from existing R45/R46 outputs and runtime-available token/internal features. "
            "Because the rule was proposed after R46 and includes reused 01/05 evidence, it is only a repair hypothesis "
            "and requires a new untouched validation set before any global success claim."
        ),
        "references": {
            "r45_rows": rel(R45_ROWS),
            "r46_rows": rel(R46_ROWS),
            "r45_polarity": rel(R45_POLARITY),
            "r46_polarity": rel(R46_POLARITY),
        },
        "outputs": {
            "policy_rows": rel(policy_rows_path),
            "token_stats": rel(token_stats_path),
            "summary": rel(summary_path),
            "report": rel(report_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# Stage4-R47 LingBot AR Abstention Policy",
        "",
        f"decision: `{decision}`",
        "global_goal_achieved: `False`",
        "claim_level: `reused_holdout_policy_assembly_not_blind_success`",
        "",
        "```json",
        json.dumps(
            {
                "policy_gate": policy_gate,
                "median_rel_vs_default": median_rel,
                "max_harm": summary["max_harm"],
                "selected_improvement_count": selected_improvement_count,
                "abstain_count": abstain_count,
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
