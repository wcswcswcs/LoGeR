#!/usr/bin/env python3
"""Summarize ACL2 v110R B1 full-control continuation metrics."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_v109tf_stage2_f_core_ablation_metrics as stage2m  # noqa: E402
import build_v110r_stage3_pilot_metrics as stage3m  # noqa: E402
import build_v110r_stage4_full_validation_metrics as stage4m  # noqa: E402


base = stage3m.base

RESULT_ROOT = ROOT / "results/acl2_v110r_lingbot_multisurface_full_ate_boost_semantic_causality"
STAGE4 = RESULT_ROOT / "stage4_full_00_01_02_05_validation"
STAGE8 = RESULT_ROOT / "stage8_b1_full_controls"
CONFIG_ROWS = STAGE8 / "action_config_rows.csv"
RUN_RESULTS = STAGE8 / "run_results.csv"
WORKSPACE = STAGE8 / "workspace"
SEQUENCES = ("00", "01", "02", "05")

SEMANTIC_PLUS = "B1_semantic_plus_internal"
SEMANTIC_ONLY = "B1_semantic_only"
CONTROL_POLICIES = (
    "B1_internal_only",
    "B1_semantic_shuffle",
    "B1_same_count_random",
    "B1_low_risk_reverse",
)
CAUSAL_MARGIN = 0.02


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base.clean_json(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def safe_float(value: Any) -> float:
    return stage2m.safe_float(value)


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def median(values: list[float]) -> float:
    return base.median([value for value in values if math.isfinite(value)])


def mean(values: list[float]) -> float:
    return base.mean([value for value in values if math.isfinite(value)])


def rel_by_seq(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        str(row.get("seq", "")): safe_float(row.get("full_ATE_sim3_relative_improvement_vs_baseline", "nan"))
        for row in rows
    }


def stage4_b1_semantic_rows() -> list[dict[str, Any]]:
    rows = []
    for row in read_csv(STAGE4 / "policy_summary_rows.csv"):
        if row.get("candidate_id") == "B1" and row.get("policy_id") in {SEMANTIC_PLUS, SEMANTIC_ONLY}:
            rows.append(
                {
                    "schema": "acl2_v110r_stage8_combined_policy_summary_row_v1",
                    "source_stage": "stage4_full_validation",
                    **row,
                }
            )
    return rows


def median_for(policy_rows: list[dict[str, Any]], policy_id: str) -> float:
    for row in policy_rows:
        if row.get("policy_id") == policy_id:
            return safe_float(row.get("median_full_rel", "nan"))
    return float("nan")


def mean_for(policy_rows: list[dict[str, Any]], policy_id: str) -> float:
    for row in policy_rows:
        if row.get("policy_id") == policy_id:
            return safe_float(row.get("mean_full_rel", "nan"))
    return float("nan")


def per_sequence_summary(full_rows: list[dict[str, Any]], stage4_full_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_policy: dict[str, list[dict[str, Any]]] = {}
    for row in full_rows:
        by_policy.setdefault(str(row.get("policy_id", "")), []).append(row)
    for row in stage4_full_rows:
        if row.get("policy_id") in {SEMANTIC_PLUS, SEMANTIC_ONLY}:
            by_policy.setdefault(str(row.get("policy_id", "")), []).append(row)

    out: list[dict[str, Any]] = []
    for seq in SEQUENCES:
        plus_rel = rel_by_seq(by_policy.get(SEMANTIC_PLUS, [])).get(seq, float("nan"))
        only_rel = rel_by_seq(by_policy.get(SEMANTIC_ONLY, [])).get(seq, float("nan"))
        control_values = {
            policy_id: rel_by_seq(by_policy.get(policy_id, [])).get(seq, float("nan"))
            for policy_id in CONTROL_POLICIES
        }
        finite_controls = {key: value for key, value in control_values.items() if math.isfinite(value)}
        best_control_policy = max(finite_controls, key=finite_controls.get) if finite_controls else ""
        best_control_rel = finite_controls.get(best_control_policy, float("nan"))
        out.append(
            {
                "schema": "acl2_v110r_stage8_b1_per_sequence_control_row_v1",
                "seq": seq,
                "semantic_plus_full_rel": plus_rel,
                "semantic_only_full_rel": only_rel,
                "internal_only_full_rel": control_values["B1_internal_only"],
                "semantic_shuffle_full_rel": control_values["B1_semantic_shuffle"],
                "same_count_random_full_rel": control_values["B1_same_count_random"],
                "low_risk_reverse_full_rel": control_values["B1_low_risk_reverse"],
                "best_control_policy_id": best_control_policy,
                "best_control_full_rel": best_control_rel,
                "semantic_plus_minus_best_control": (
                    plus_rel - best_control_rel
                    if math.isfinite(plus_rel) and math.isfinite(best_control_rel)
                    else float("nan")
                ),
                "semantic_plus_minus_semantic_only": (
                    plus_rel - only_rel
                    if math.isfinite(plus_rel) and math.isfinite(only_rel)
                    else float("nan")
                ),
            }
        )
    return out


def semantic_decision_row(policy_rows: list[dict[str, Any]], metric_complete: bool, all_action: bool) -> dict[str, Any]:
    plus_med = median_for(policy_rows, SEMANTIC_PLUS)
    only_med = median_for(policy_rows, SEMANTIC_ONLY)
    internal_med = median_for(policy_rows, "B1_internal_only")
    shuffle_med = median_for(policy_rows, "B1_semantic_shuffle")
    random_med = median_for(policy_rows, "B1_same_count_random")
    reverse_med = median_for(policy_rows, "B1_low_risk_reverse")
    control_medians = {
        "B1_semantic_only": only_med,
        "B1_internal_only": internal_med,
        "B1_semantic_shuffle": shuffle_med,
        "B1_same_count_random": random_med,
        "B1_low_risk_reverse": reverse_med,
    }
    finite_controls = {key: value for key, value in control_medians.items() if math.isfinite(value)}
    best_control_policy = max(finite_controls, key=finite_controls.get) if finite_controls else ""
    best_control_med = finite_controls.get(best_control_policy, float("nan"))

    deltas = {
        "semantic_plus_minus_semantic_only_median": plus_med - only_med
        if math.isfinite(plus_med) and math.isfinite(only_med)
        else float("nan"),
        "semantic_plus_minus_internal_only_median": plus_med - internal_med
        if math.isfinite(plus_med) and math.isfinite(internal_med)
        else float("nan"),
        "semantic_plus_minus_semantic_shuffle_median": plus_med - shuffle_med
        if math.isfinite(plus_med) and math.isfinite(shuffle_med)
        else float("nan"),
        "semantic_plus_minus_same_count_random_median": plus_med - random_med
        if math.isfinite(plus_med) and math.isfinite(random_med)
        else float("nan"),
        "semantic_plus_minus_low_risk_reverse_median": plus_med - reverse_med
        if math.isfinite(plus_med) and math.isfinite(reverse_med)
        else float("nan"),
        "semantic_plus_minus_best_control_median": plus_med - best_control_med
        if math.isfinite(plus_med) and math.isfinite(best_control_med)
        else float("nan"),
    }
    required_deltas = [
        deltas["semantic_plus_minus_semantic_only_median"],
        deltas["semantic_plus_minus_internal_only_median"],
        deltas["semantic_plus_minus_semantic_shuffle_median"],
        deltas["semantic_plus_minus_same_count_random_median"],
    ]
    semantic_causality_pass = bool(
        metric_complete
        and all_action
        and math.isfinite(plus_med)
        and all(math.isfinite(delta) and delta >= CAUSAL_MARGIN for delta in required_deltas)
    )
    if not metric_complete or not all_action:
        taxonomy = "STAGE8_B1_CONTROLS_INCOMPLETE"
        blocker = "missing_metric_or_action_fidelity"
    elif semantic_causality_pass:
        taxonomy = "STAGE8_B1_SEMANTIC_CAUSALITY_REPAIRED"
        blocker = ""
    elif math.isfinite(deltas["semantic_plus_minus_semantic_only_median"]) and deltas["semantic_plus_minus_semantic_only_median"] < CAUSAL_MARGIN:
        taxonomy = "STAGE8_B1_SEMANTIC_CONTENT_NOT_CAUSAL"
        blocker = "semantic_plus_internal_does_not_beat_semantic_only_by_margin"
    elif math.isfinite(deltas["semantic_plus_minus_semantic_shuffle_median"]) and deltas["semantic_plus_minus_semantic_shuffle_median"] < CAUSAL_MARGIN:
        taxonomy = "STAGE8_B1_SEMANTIC_SHUFFLE_MATCHES"
        blocker = "semantic_shuffle_matches_or_exceeds_semantic_plus_internal"
    elif math.isfinite(deltas["semantic_plus_minus_same_count_random_median"]) and deltas["semantic_plus_minus_same_count_random_median"] < CAUSAL_MARGIN:
        taxonomy = "STAGE8_B1_SCHEDULE_COUNT_CONTROL_MATCHES"
        blocker = "same_count_random_matches_or_exceeds_semantic_plus_internal"
    else:
        taxonomy = "STAGE8_B1_SEMANTIC_CAUSALITY_FAIL_OTHER_CONTROL"
        blocker = "one_or_more_required_control_margins_failed"

    return {
        "schema": "acl2_v110r_stage8_b1_semantic_decision_row_v1",
        "candidate_id": "B1",
        "semantic_plus_policy_id": SEMANTIC_PLUS,
        "semantic_plus_median_full_rel": plus_med,
        "semantic_plus_mean_full_rel": mean_for(policy_rows, SEMANTIC_PLUS),
        "semantic_only_median_full_rel": only_med,
        "internal_only_median_full_rel": internal_med,
        "semantic_shuffle_median_full_rel": shuffle_med,
        "same_count_random_median_full_rel": random_med,
        "low_risk_reverse_median_full_rel": reverse_med,
        "best_control_policy_id": best_control_policy,
        "best_control_median_full_rel": best_control_med,
        **deltas,
        "causal_margin_required": CAUSAL_MARGIN,
        "metric_complete": metric_complete,
        "all_action_fidelity": all_action,
        "semantic_causality_pass": semantic_causality_pass,
        "taxonomy": taxonomy,
        "blocker": blocker,
    }


def build_report(summary: dict[str, Any], decision: dict[str, Any], combined_rows: list[dict[str, Any]]) -> str:
    ranked = sorted(combined_rows, key=lambda row: safe_float(row.get("median_full_rel", "nan")), reverse=True)
    lines = [
        "# ACL2 v110R Stage8 B1 Full Control Report",
        "",
        f"taxonomy: {summary['taxonomy']}",
        f"semantic_causality_pass: {summary['semantic_causality_pass']}",
        f"blocker: {summary['blocker']}",
        "",
        "## Policy Ranking",
        "",
    ]
    for row in ranked:
        lines.append(
            "- {policy_id}: median_full_rel={median} mean_full_rel={mean} improved={improved}/4 max_harm={harm} source={source}".format(
                policy_id=row.get("policy_id", ""),
                median=row.get("median_full_rel", ""),
                mean=row.get("mean_full_rel", ""),
                improved=row.get("improved_seq_count", ""),
                harm=row.get("max_harm", ""),
                source=row.get("source_stage", "stage8_b1_full_controls"),
            )
        )
    lines.extend(
        [
            "",
            "## Causality Deltas",
            "",
            f"semantic_plus_minus_semantic_only_median: {decision['semantic_plus_minus_semantic_only_median']}",
            f"semantic_plus_minus_internal_only_median: {decision['semantic_plus_minus_internal_only_median']}",
            f"semantic_plus_minus_semantic_shuffle_median: {decision['semantic_plus_minus_semantic_shuffle_median']}",
            f"semantic_plus_minus_same_count_random_median: {decision['semantic_plus_minus_same_count_random_median']}",
            f"semantic_plus_minus_best_control_median: {decision['semantic_plus_minus_best_control_median']}",
            "",
            "## Interpretation",
            "",
            "This continuation uses the original Stage4 B1 semantic rows and newly run four-sequence B1 control rows. "
            "It only changes the semantic-causality evidence boundary; it does not retune B1 or use selected-window substitutes.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    stage3m.OUT = STAGE8
    stage3m.CONFIG_ROWS = CONFIG_ROWS
    stage3m.RUN_RESULTS = RUN_RESULTS
    stage3m.WORKSPACE = WORKSPACE
    stage3m.SEQUENCES = SEQUENCES
    stage3m.install_stage3_overrides()

    config_rows = read_csv(CONFIG_ROWS)
    latest = stage2m.latest_run_results(read_csv(RUN_RESULTS))
    full_rows, rolling_rows, local_rows, fidelity_rows = stage2m.metric_rows(config_rows, latest)
    for rows in (full_rows, rolling_rows, local_rows, fidelity_rows):
        stage3m.add_candidate_metadata(rows)
        for row in rows:
            row["schema"] = str(row.get("schema", "")).replace("acl2_v110r_stage3", "acl2_v110r_stage8")

    f19_control = stage4m.f19_rows()
    control_policy_rows = stage4m.policy_summary_rows(full_rows, rolling_rows, fidelity_rows, f19_control)
    for row in control_policy_rows:
        row["schema"] = "acl2_v110r_stage8_control_policy_summary_row_v1"
        row["source_stage"] = "stage8_b1_full_controls"
    semantic_rows = stage4_b1_semantic_rows()
    combined_rows = semantic_rows + control_policy_rows
    stage4_full_rows = read_csv(STAGE4 / "full_metric_rows.csv")
    per_seq_rows = per_sequence_summary(full_rows, stage4_full_rows)

    metric_complete = len(full_rows) == len(config_rows) and all(bool_value(row.get("metric_available")) for row in full_rows)
    all_action = len(fidelity_rows) == len(config_rows) and all(bool_value(row.get("action_fidelity_pass")) for row in fidelity_rows)
    decision = semantic_decision_row(combined_rows, metric_complete, all_action)
    observed_counts: dict[str, int] = {}
    for row in latest.values():
        if stage2m.safe_rc(row) == 0:
            phase = str(row.get("phase", ""))
            observed_counts[phase] = observed_counts.get(phase, 0) + 1

    summary = {
        "schema": "acl2_v110r_stage8_b1_full_control_summary_v1",
        "metric_complete": metric_complete,
        "all_action_fidelity": all_action,
        "observed_prepare_count": observed_counts.get("prepare", 0),
        "observed_run_worker_count": observed_counts.get("run_worker", 0),
        "observed_evaluate_count": observed_counts.get("evaluate", 0),
        "observed_report_count": observed_counts.get("report", 0),
        "expected_run_worker_count": len(config_rows),
        "full_metric_row_count": len(full_rows),
        "rolling_metric_row_count": len(rolling_rows),
        "local_handoff_metric_row_count": len(local_rows),
        "action_fidelity_row_count": len(fidelity_rows),
        "control_policy_summary_row_count": len(control_policy_rows),
        "combined_policy_summary_row_count": len(combined_rows),
        "semantic_causality_pass": decision["semantic_causality_pass"],
        "taxonomy": decision["taxonomy"],
        "blocker": decision["blocker"],
        "outputs": {
            "full_metric_rows": rel(STAGE8 / "full_metric_rows.csv"),
            "rolling_metric_rows": rel(STAGE8 / "rolling_metric_rows.csv"),
            "local_handoff_metric_rows": rel(STAGE8 / "local_handoff_metric_rows.csv"),
            "action_fidelity_rows": rel(STAGE8 / "action_fidelity_rows.csv"),
            "control_policy_summary_rows": rel(STAGE8 / "control_policy_summary_rows.csv"),
            "combined_policy_summary_rows": rel(STAGE8 / "combined_policy_summary_rows.csv"),
            "per_sequence_control_rows": rel(STAGE8 / "per_sequence_control_rows.csv"),
            "semantic_decision_rows": rel(STAGE8 / "semantic_decision_rows.csv"),
            "report": rel(STAGE8 / "B1_FULL_CONTROL_SEMANTIC_CAUSALITY_REPORT.md"),
            "summary": rel(STAGE8 / "stage8_summary.json"),
        },
    }

    write_csv(STAGE8 / "full_metric_rows.csv", full_rows)
    write_csv(STAGE8 / "rolling_metric_rows.csv", rolling_rows)
    write_csv(STAGE8 / "local_handoff_metric_rows.csv", local_rows)
    write_csv(STAGE8 / "action_fidelity_rows.csv", fidelity_rows)
    write_csv(STAGE8 / "control_policy_summary_rows.csv", control_policy_rows)
    write_csv(STAGE8 / "combined_policy_summary_rows.csv", combined_rows)
    write_csv(STAGE8 / "per_sequence_control_rows.csv", per_seq_rows)
    write_csv(STAGE8 / "semantic_decision_rows.csv", [decision])
    write_json(STAGE8 / "stage8_summary.json", summary)
    write_text(STAGE8 / "B1_FULL_CONTROL_SEMANTIC_CAUSALITY_REPORT.md", build_report(summary, decision, combined_rows))
    print(json.dumps(base.clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
