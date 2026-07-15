#!/usr/bin/env python3
"""Summarize ACL2 v111TF T2 trajectory context-token ablation metrics."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_v109tf_stage2_f_core_ablation_metrics as stage2m  # noqa: E402
import build_v110r_stage3_pilot_metrics as stage3m  # noqa: E402


RESULT_ROOT = ROOT / "results/acl2_v111tf_lingbot_semantic_aware_memory_management_anchor_local_trajectory"
T2 = RESULT_ROOT / "batch_t_t2_context_token_ablation"
CONFIG_ROWS = T2 / "action_config_rows.csv"
RUN_RESULTS = T2 / "run_results.csv"
WORKSPACE = T2 / "workspace"
SEQUENCES = ("00", "01", "02", "05")

B1_MEDIAN_FULL_REL = 0.17413068803456322
B1_MEAN_FULL_REL = 0.18754824888948118
B1_90PCT_MEDIAN_GATE = B1_MEDIAN_FULL_REL * 0.9
MAX_HARM_GATE = 0.01
STRONG_MARGIN = 0.03


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stage3m.base.clean_json(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def t2_phase_status_for(cfg: dict[str, str], latest: dict[tuple[str, str], dict[str, str]]) -> tuple[dict[str, Any], bool, bool]:
    status: dict[str, Any] = {}
    all_metric_phase_success = True
    all_phase_success = True
    seq = cfg["seq"]
    for phase in ("prepare", "run_worker", "evaluate", "report"):
        if phase == "prepare":
            run_name = f"kitti_lingbot_v111tf_t2_prepare_{seq}"
        else:
            run_name = f"kitti_lingbot_v111tf_t2_{cfg['policy_id']}_{seq}_{phase}"
        row = latest.get((run_name, phase))
        rc = stage2m.safe_rc(row)
        status[f"{phase}_returncode"] = rc
        status[f"{phase}_duration_sec"] = row.get("duration_sec", "") if row else ""
        if phase in {"prepare", "run_worker", "evaluate"}:
            all_metric_phase_success = all_metric_phase_success and rc == 0
        all_phase_success = all_phase_success and rc == 0
    return status, all_metric_phase_success, all_phase_success


def install_t2_metric_overrides() -> None:
    stage2m.OUT = T2
    stage2m.CONFIG_ROWS = CONFIG_ROWS
    stage2m.RUN_RESULTS = RUN_RESULTS
    stage2m.WORKSPACE = WORKSPACE
    stage2m.SEQUENCES = SEQUENCES
    stage2m.phase_status_for = t2_phase_status_for

    original_action_fidelity = stage2m.action_fidelity_row

    def t2_action_fidelity_row(cfg: dict[str, str], latest: dict[tuple[str, str], dict[str, str]]) -> dict[str, Any]:
        row = original_action_fidelity(cfg, latest)
        run_name = f"kitti_lingbot_v111tf_t2_{cfg['policy_id']}_{cfg['seq']}_run_worker"
        run_row = latest.get((run_name, "run_worker"), {})
        row["schema"] = "acl2_v111tf_t2_action_fidelity_row_v1"
        row["run_worker_returncode"] = run_row.get("returncode", "")
        row["run_worker_duration_sec"] = run_row.get("duration_sec", "")
        row["context_token_type_mask"] = cfg.get("context_token_type_mask", "")
        row["claim_role"] = cfg.get("claim_role", "")
        return row

    stage2m.action_fidelity_row = t2_action_fidelity_row


def rel_by_seq(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        str(row.get("seq", "")): safe_float(row.get("full_ATE_sim3_relative_improvement_vs_baseline", "nan"))
        for row in rows
    }


def policy_summary_rows(
    full_rows: list[dict[str, Any]],
    rolling_rows: list[dict[str, Any]],
    fidelity_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rolling_by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fidelity_by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in full_rows:
        by_policy[str(row["policy_id"])].append(row)
    for row in rolling_rows:
        rolling_by_policy[str(row["policy_id"])].append(row)
    for row in fidelity_rows:
        fidelity_by_policy[str(row["policy_id"])].append(row)

    out: list[dict[str, Any]] = []
    for policy_id in sorted(by_policy):
        rows = by_policy[policy_id]
        rels_by_seq = rel_by_seq(rows)
        rels = [rels_by_seq.get(seq, float("nan")) for seq in SEQUENCES]
        roll_rels = [
            safe_float(row.get("rolling_ATE_p90_relative_improvement_vs_baseline", "nan"))
            for row in rolling_by_policy.get(policy_id, [])
        ]
        finals = [
            safe_float(row.get("final_error_relative_improvement_vs_baseline", "nan"))
            for row in rows
        ]
        locals_ = [
            safe_float(row.get("local_window_ATE_rel_improvement_vs_baseline_median", "nan"))
            for row in rows
        ]
        median_full = stage3m.base.median(rels)
        mean_full = stage3m.base.mean(rels)
        improved_count = sum(1 for value in rels if math.isfinite(value) and value > 0.0)
        max_harm = stage3m.base.max_rel_harm(rels)
        rolling_p90_median = stage3m.base.median(roll_rels)
        final_median = stage3m.base.median(finals)
        local_harm = stage3m.base.max_rel_harm(locals_)
        action_pass_count = sum(1 for row in fidelity_by_policy.get(policy_id, []) if bool_value(row.get("action_fidelity_pass")))
        metric_complete = len(rows) == len(SEQUENCES) and all(bool_value(row.get("metric_available")) for row in rows)
        all_action = action_pass_count == len(SEQUENCES)
        b1_90_geometry_gate = bool(
            metric_complete
            and all_action
            and math.isfinite(median_full)
            and median_full >= B1_90PCT_MEDIAN_GATE
            and max_harm <= MAX_HARM_GATE
            and math.isfinite(rolling_p90_median)
            and rolling_p90_median >= 0.0
        )
        strong_success_gate = bool(
            metric_complete
            and all_action
            and max_harm <= MAX_HARM_GATE
            and (
                (math.isfinite(median_full) and median_full >= B1_MEDIAN_FULL_REL + STRONG_MARGIN)
                or (math.isfinite(mean_full) and mean_full >= B1_MEAN_FULL_REL + STRONG_MARGIN)
            )
        )
        sample = rows[0]
        out.append(
            {
                "schema": "acl2_v111tf_t2_policy_summary_row_v1",
                "candidate_id": "T2",
                "surface_id": sample.get("surface_id", ""),
                "policy_id": policy_id,
                "policy_family": sample.get("policy_family", ""),
                "sequence_count": len(rows),
                "metric_complete": metric_complete,
                "action_fidelity_pass_count": action_pass_count,
                "all_action_fidelity": all_action,
                "median_full_rel": median_full,
                "mean_full_rel": mean_full,
                "improved_seq_count": improved_count,
                "max_harm": max_harm,
                "rolling_p90_median_rel": rolling_p90_median,
                "final_error_median_rel": final_median,
                "local_window_median_harm": local_harm,
                "seq00_full_rel": rels_by_seq.get("00", ""),
                "seq01_full_rel": rels_by_seq.get("01", ""),
                "seq02_full_rel": rels_by_seq.get("02", ""),
                "seq05_full_rel": rels_by_seq.get("05", ""),
                "b1_90pct_median_gate": B1_90PCT_MEDIAN_GATE,
                "t2_b1_90_geometry_gate_pass": b1_90_geometry_gate,
                "t2_strong_success_gate_pass": strong_success_gate,
                "median_full_rel_minus_b1_median": (
                    median_full - B1_MEDIAN_FULL_REL if math.isfinite(median_full) else float("nan")
                ),
                "mean_full_rel_minus_b1_mean": (
                    mean_full - B1_MEAN_FULL_REL if math.isfinite(mean_full) else float("nan")
                ),
                "claim_boundary": (
                    "T2 token-type geometry/mechanism summary only; semantic causality controls are not satisfied here."
                ),
            }
        )
    return out


def parity_crosscheck_rows(policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_policy = {row["policy_id"]: row for row in policy_rows}
    pairs = [
        ("default_off_vs_no_action", "T2_no_action_mask_all1_default_off", None, 0.0),
        ("new_all_context_vs_legacy_all_special", "T2_default_context_tokens", "T2_default_context_tokens_legacy_context_only", 0.0),
        ("new_anchor_only_vs_legacy_anchor_special", "T2_anchor_only", "T2_anchor_only_for_high_risk_else_default", 0.0),
    ]
    rows: list[dict[str, Any]] = []
    for pair_id, left, right, expected_delta in pairs:
        left_row = by_policy.get(left, {})
        right_row = by_policy.get(right, {})
        left_med = safe_float(left_row.get("median_full_rel", "nan"))
        if right is None:
            right_med = expected_delta
        else:
            right_med = safe_float(right_row.get("median_full_rel", "nan"))
        rows.append(
            {
                "schema": "acl2_v111tf_t2_parity_crosscheck_row_v1",
                "pair_id": pair_id,
                "left_policy_id": left,
                "right_policy_id": right or "v105_no_action_zero_improvement",
                "left_median_full_rel": left_med,
                "right_median_full_rel": right_med,
                "median_delta": left_med - right_med if math.isfinite(left_med) and math.isfinite(right_med) else float("nan"),
                "median_delta_expected": expected_delta,
            }
        )
    return rows


def build_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    ranked = sorted(rows, key=lambda row: safe_float(row.get("median_full_rel", "nan")), reverse=True)
    lines = [
        "# ACL2 v111TF T2 Context-Token Ablation Report",
        "",
        f"metric_complete: {summary['metric_complete']}",
        f"all_action_fidelity: {summary['all_action_fidelity']}",
        f"parity_complete_all_sequences: {summary['parity_complete_all_sequences']}",
        f"taxonomy: {summary['taxonomy']}",
        f"blocker: {summary['blocker']}",
        "",
        "## Policy Ranking",
        "",
    ]
    for row in ranked:
        lines.append(
            "- {policy_id}: median={median} mean={mean} improved={improved}/4 max_harm={harm} "
            "rolling={rolling} final={final} local_harm={local_harm} b1_90_gate={gate} strong={strong}".format(
                policy_id=row.get("policy_id", ""),
                median=row.get("median_full_rel", ""),
                mean=row.get("mean_full_rel", ""),
                improved=row.get("improved_seq_count", ""),
                harm=row.get("max_harm", ""),
                rolling=row.get("rolling_p90_median_rel", ""),
                final=row.get("final_error_median_rel", ""),
                local_harm=row.get("local_window_median_harm", ""),
                gate=row.get("t2_b1_90_geometry_gate_pass", ""),
                strong=row.get("t2_strong_success_gate_pass", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "T2 evaluates compact context-token carriers under B1-selected high-risk keyframes. It does not by itself satisfy semantic-causality controls.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    install_t2_metric_overrides()
    config_rows = read_csv(CONFIG_ROWS)
    latest = stage2m.latest_run_results(read_csv(RUN_RESULTS))
    full_rows, rolling_rows, local_rows, fidelity_rows = stage2m.metric_rows(config_rows, latest)
    for rows in (full_rows, rolling_rows, local_rows, fidelity_rows):
        for row in rows:
            row["schema"] = str(row.get("schema", "")).replace("acl2_v109tf_stage2", "acl2_v111tf_t2")
            cfg = next((cfg for cfg in config_rows if cfg["policy_id"] == row.get("policy_id") and cfg["seq"] == row.get("seq")), {})
            row["context_token_type_mask"] = cfg.get("context_token_type_mask", "")
            row["claim_role"] = cfg.get("claim_role", "")

    policy_rows = policy_summary_rows(full_rows, rolling_rows, fidelity_rows)
    crosscheck_rows = parity_crosscheck_rows(policy_rows)
    parity_summary = stage3m.base.load_json(T2 / "t2_parity_summary.json")
    observed_counts: dict[str, int] = {}
    for row in latest.values():
        if stage2m.safe_rc(row) == 0:
            phase = str(row.get("phase", ""))
            observed_counts[phase] = observed_counts.get(phase, 0) + 1

    metric_complete = len(full_rows) == len(config_rows) and all(bool_value(row.get("metric_available")) for row in full_rows)
    all_action = len(fidelity_rows) == len(config_rows) and all(bool_value(row.get("action_fidelity_pass")) for row in fidelity_rows)
    geometry_pass = [row for row in policy_rows if bool_value(row.get("t2_b1_90_geometry_gate_pass"))]
    strong_pass = [row for row in policy_rows if bool_value(row.get("t2_strong_success_gate_pass"))]
    best = max(policy_rows, key=lambda row: safe_float(row.get("median_full_rel", "nan"))) if policy_rows else {}
    if strong_pass:
        taxonomy = "T2_STRONG_GEOMETRY_PASS_SEMANTIC_CAUSALITY_PENDING"
        blocker = "semantic_causality_controls_not_run_for_token_type_policy"
    elif geometry_pass:
        taxonomy = "T2_B1_90_GEOMETRY_PASS_SEMANTIC_CAUSALITY_PENDING"
        blocker = "semantic_causality_controls_not_run_for_token_type_policy"
    else:
        taxonomy = "T2_GEOMETRY_FAIL_OR_BELOW_B1_90"
        blocker = "no_token_type_policy_satisfied_b1_90_geometry_gate"

    summary = {
        "schema": "acl2_v111tf_t2_context_token_metric_summary_v1",
        "metric_complete": metric_complete,
        "all_action_fidelity": all_action,
        "parity_complete_all_sequences": bool(parity_summary.get("parity_complete_all_sequences")),
        "completed_parity_pass": bool(parity_summary.get("completed_parity_pass")),
        "taxonomy": taxonomy,
        "blocker": blocker,
        "observed_prepare_count": observed_counts.get("prepare", 0),
        "observed_run_worker_count": observed_counts.get("run_worker", 0),
        "observed_evaluate_count": observed_counts.get("evaluate", 0),
        "observed_report_count": observed_counts.get("report", 0),
        "expected_run_worker_count": len(config_rows),
        "full_metric_row_count": len(full_rows),
        "rolling_metric_row_count": len(rolling_rows),
        "local_handoff_metric_row_count": len(local_rows),
        "action_fidelity_row_count": len(fidelity_rows),
        "policy_summary_row_count": len(policy_rows),
        "geometry_pass_policy_ids": [row["policy_id"] for row in geometry_pass],
        "strong_success_policy_ids": [row["policy_id"] for row in strong_pass],
        "best_policy_by_median_full_rel": best.get("policy_id", ""),
        "best_policy_median_full_rel": best.get("median_full_rel", ""),
        "best_policy_mean_full_rel": best.get("mean_full_rel", ""),
        "b1_reference": {
            "median_full_rel": B1_MEDIAN_FULL_REL,
            "mean_full_rel": B1_MEAN_FULL_REL,
            "b1_90pct_median_gate": B1_90PCT_MEDIAN_GATE,
            "strong_margin": STRONG_MARGIN,
        },
        "semantic_causality_claim_allowed": False,
        "semantic_causality_claim_blocker": "T2 token-type ablation has no semantic shuffle/random/same-bucket/schedule controls.",
        "outputs": {
            "full_metric_rows": rel(T2 / "full_metric_rows.csv"),
            "rolling_metric_rows": rel(T2 / "rolling_metric_rows.csv"),
            "local_handoff_metric_rows": rel(T2 / "local_handoff_metric_rows.csv"),
            "action_fidelity_rows": rel(T2 / "action_fidelity_rows.csv"),
            "policy_summary_rows": rel(T2 / "policy_summary_rows.csv"),
            "parity_crosscheck_rows": rel(T2 / "t2_parity_crosscheck_rows.csv"),
            "report": rel(T2 / "T2_CONTEXT_TOKEN_ABLATION_REPORT.md"),
            "summary": rel(T2 / "t2_metric_summary.json"),
        },
    }

    write_csv(T2 / "full_metric_rows.csv", full_rows)
    write_csv(T2 / "rolling_metric_rows.csv", rolling_rows)
    write_csv(T2 / "local_handoff_metric_rows.csv", local_rows)
    write_csv(T2 / "action_fidelity_rows.csv", fidelity_rows)
    write_csv(T2 / "policy_summary_rows.csv", policy_rows)
    write_csv(T2 / "t2_parity_crosscheck_rows.csv", crosscheck_rows)
    write_json(T2 / "t2_metric_summary.json", summary)
    write_text(T2 / "T2_CONTEXT_TOKEN_ABLATION_REPORT.md", build_report(summary, policy_rows))
    print(json.dumps(stage3m.base.clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
