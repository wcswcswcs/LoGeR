#!/usr/bin/env python3
"""Build ACL2 v110R Stage4 full KITTI validation metrics and decision summary."""

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

import build_v110r_stage3_pilot_metrics as stage3m  # noqa: E402
import build_v109tf_stage2_f_core_ablation_metrics as stage2m  # noqa: E402


base = stage3m.base

RESULT_ROOT = ROOT / "results/acl2_v110r_lingbot_multisurface_full_ate_boost_semantic_causality"
STAGE0 = RESULT_ROOT / "stage0_evidence_freeze"
OUT = RESULT_ROOT / "stage4_full_00_01_02_05_validation"
CONFIG_ROWS = OUT / "action_config_rows.csv"
RUN_RESULTS = OUT / "run_results.csv"
WORKSPACE = OUT / "workspace"
SEQUENCES = ("00", "01", "02", "05")

GEOMETRY_MEDIAN_FULL_REL_MIN = 0.10
GEOMETRY_MEAN_FULL_REL_MIN = 0.12
GEOMETRY_IMPROVED_SEQ_MIN = 3
GEOMETRY_MAX_HARM_MAX = 0.01
GEOMETRY_ROLLING_P90_MEDIAN_MIN = 0.0
GEOMETRY_FINAL_ERROR_MEDIAN_MIN = 0.0
GEOMETRY_LOCAL_HARM_MAX = 0.02
F19_MEDIAN_FULL_REL = 0.07601700114005772
F19_MEAN_FULL_REL = 0.10856739499696623
F19_STRONG_MARGIN = 0.02
HARD_NEGATIVE_REL_MIN = 0.03
F19_00_02_DROP_TOL = 0.03


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


def finite(values: list[float]) -> list[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def candidate_id(policy_id: str) -> str:
    return policy_id.split("_", 1)[0] if "_" in policy_id else policy_id


def f19_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(STAGE0 / "f19_champion_metrics.csv"):
        seq = row["seq"]
        rel_value = safe_float(row.get("full_ATE_sim3_relative_improvement_vs_baseline", "nan"))
        rows.append(
            {
                "schema": "acl2_v110r_stage4_fixed_control_row_v1",
                "control_id": "F19",
                "policy_id": row.get("policy_id", "F19_dynamic_or_special_admitted_high_risk_else_weak_context"),
                "seq": seq,
                "baseline_full_ATE_sim3": row.get("baseline_full_ATE_sim3", ""),
                "full_ATE_sim3": row.get("full_ATE_sim3", ""),
                "full_ATE_sim3_relative_improvement_vs_baseline": rel_value,
                "full_RPE_translation": row.get("full_RPE_translation", ""),
                "full_RPE_rotation": row.get("full_RPE_rotation", ""),
                "final_error_m": row.get("final_error_m", ""),
                "final_error_relative_improvement_vs_baseline": row.get("final_error_relative_improvement_vs_baseline", ""),
                "rolling_metric_note": "frozen_stage0_f19_champion_row",
                "source": row.get("source", ""),
            }
        )
    return rows


def no_action_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(STAGE0 / "frozen_baseline_table.csv"):
        rows.append(
            {
                "schema": "acl2_v110r_stage4_fixed_control_row_v1",
                "control_id": "NO_ACTION",
                "policy_id": "LBM_STREAM_DEFAULT",
                "seq": row.get("seq", ""),
                "baseline_full_ATE_sim3": row.get("ATE_full_sim3_m", ""),
                "full_ATE_sim3": row.get("ATE_full_sim3_m", ""),
                "full_ATE_sim3_relative_improvement_vs_baseline": 0.0,
                "full_RPE_translation": row.get("benchmark_rpe_trans", ""),
                "full_RPE_rotation": row.get("benchmark_rpe_rot", ""),
                "final_error_m": row.get("final_error_m", ""),
                "final_error_relative_improvement_vs_baseline": 0.0,
                "rolling_metric_note": "frozen_stage0_no_action_baseline",
                "source": row.get("source", ""),
            }
        )
    return rows


def rel_by_seq(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        str(row.get("seq", "")): safe_float(row.get("full_ATE_sim3_relative_improvement_vs_baseline", "nan"))
        for row in rows
    }


def policy_summary_rows(
    full_rows: list[dict[str, Any]],
    rolling_rows: list[dict[str, Any]],
    fidelity_rows: list[dict[str, Any]],
    f19_control_rows: list[dict[str, Any]],
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

    f19_rel = rel_by_seq(f19_control_rows)
    rows_out: list[dict[str, Any]] = []
    for policy_id in sorted(by_policy):
        rows = by_policy[policy_id]
        rels_by_seq = rel_by_seq(rows)
        rels = [rels_by_seq.get(seq, float("nan")) for seq in SEQUENCES]
        finals = [
            safe_float(row.get("final_error_relative_improvement_vs_baseline", "nan"))
            for row in rows
        ]
        locals_ = [
            safe_float(row.get("local_window_ATE_rel_improvement_vs_baseline_median", "nan"))
            for row in rows
        ]
        roll_rels = [
            safe_float(row.get("rolling_ATE_p90_relative_improvement_vs_baseline", "nan"))
            for row in rolling_by_policy.get(policy_id, [])
        ]
        action_pass_count = sum(1 for row in fidelity_by_policy.get(policy_id, []) if bool_value(row.get("action_fidelity_pass")))
        median_full = base.median(rels)
        mean_full = base.mean(rels)
        improved_count = sum(1 for value in rels if math.isfinite(value) and value > 0.0)
        max_harm = base.max_rel_harm(rels)
        median_rolling = base.median(roll_rels)
        median_final = base.median(finals)
        local_harm = base.max_rel_harm(locals_)
        metric_complete = len(rows) == len(SEQUENCES) and all(bool_value(row.get("metric_available")) for row in rows)
        all_action = action_pass_count == len(SEQUENCES)
        median_or_mean_gate = (
            (math.isfinite(median_full) and median_full >= GEOMETRY_MEDIAN_FULL_REL_MIN)
            or (math.isfinite(mean_full) and mean_full >= GEOMETRY_MEAN_FULL_REL_MIN)
        )
        seq_gate = improved_count >= GEOMETRY_IMPROVED_SEQ_MIN and max_harm <= GEOMETRY_MAX_HARM_MAX
        rolling_gate = math.isfinite(median_rolling) and median_rolling > GEOMETRY_ROLLING_P90_MEDIAN_MIN
        final_gate = math.isfinite(median_final) and median_final >= GEOMETRY_FINAL_ERROR_MEDIAN_MIN
        local_gate = local_harm <= GEOMETRY_LOCAL_HARM_MAX
        geometry_pass = bool(metric_complete and all_action and median_or_mean_gate and seq_gate and rolling_gate and final_gate and local_gate)
        strong_median_gate = math.isfinite(median_full) and median_full >= F19_MEDIAN_FULL_REL + F19_STRONG_MARGIN
        strong_mean_gate = math.isfinite(mean_full) and mean_full >= F19_MEAN_FULL_REL + F19_STRONG_MARGIN
        hard_negative_gate = (
            max(
                rels_by_seq.get("01", float("nan")),
                rels_by_seq.get("05", float("nan")),
            )
            >= HARD_NEGATIVE_REL_MIN
            and rels_by_seq.get("00", float("-inf")) >= f19_rel.get("00", float("nan")) - F19_00_02_DROP_TOL
            and rels_by_seq.get("02", float("-inf")) >= f19_rel.get("02", float("nan")) - F19_00_02_DROP_TOL
        )
        strong_improvement_pass = bool(geometry_pass and (strong_median_gate or strong_mean_gate or hard_negative_gate))
        rows_out.append(
            {
                "schema": "acl2_v110r_stage4_policy_summary_row_v1",
                "candidate_id": candidate_id(policy_id),
                "surface_id": rows[0].get("surface_id", ""),
                "policy_id": policy_id,
                "policy_family": rows[0].get("policy_family", ""),
                "sequence_count": len(rows),
                "metric_complete": metric_complete,
                "action_fidelity_pass_count": action_pass_count,
                "all_action_fidelity": all_action,
                "median_full_rel": median_full,
                "mean_full_rel": mean_full,
                "improved_seq_count": improved_count,
                "max_harm": max_harm,
                "rolling_p90_median_rel": median_rolling,
                "final_error_median_rel": median_final,
                "local_window_median_harm": local_harm,
                "seq00_full_rel": rels_by_seq.get("00", ""),
                "seq01_full_rel": rels_by_seq.get("01", ""),
                "seq02_full_rel": rels_by_seq.get("02", ""),
                "seq05_full_rel": rels_by_seq.get("05", ""),
                "median_or_mean_gate_pass": median_or_mean_gate,
                "sequence_gate_pass": seq_gate,
                "rolling_gate_pass": rolling_gate,
                "final_error_gate_pass": final_gate,
                "local_gate_pass": local_gate,
                "stage4_geometry_pass": geometry_pass,
                "strong_median_gate_pass": strong_median_gate,
                "strong_mean_gate_pass": strong_mean_gate,
                "hard_negative_gate_pass": hard_negative_gate,
                "strong_improvement_pass": strong_improvement_pass,
                "median_full_rel_minus_f19_median": (
                    median_full - F19_MEDIAN_FULL_REL if math.isfinite(median_full) else float("nan")
                ),
                "mean_full_rel_minus_f19_mean": (
                    mean_full - F19_MEAN_FULL_REL if math.isfinite(mean_full) else float("nan")
                ),
                "stage4_interpretation": (
                    "geometry_and_strong_improvement_pass_needs_stage6_causality"
                    if strong_improvement_pass
                    else "geometry_pass_only_needs_stage6_or_no_promotion"
                    if geometry_pass
                    else "stage4_geometry_fail"
                ),
            }
        )
    return rows_out


def semantic_stage4_rows(policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_candidate_family = {
        (str(row["candidate_id"]), str(row["policy_family"])): row
        for row in policy_rows
    }
    candidates = sorted({str(row["candidate_id"]) for row in policy_rows})
    out: list[dict[str, Any]] = []
    for cand in candidates:
        sem_plus = by_candidate_family.get((cand, "semantic_plus_internal"), {})
        sem_only = by_candidate_family.get((cand, "semantic_only"), {})
        plus_med = safe_float(sem_plus.get("median_full_rel", "nan"))
        only_med = safe_float(sem_only.get("median_full_rel", "nan"))
        out.append(
            {
                "schema": "acl2_v110r_stage4_semantic_control_row_v1",
                "candidate_id": cand,
                "surface_id": sem_plus.get("surface_id") or sem_only.get("surface_id", ""),
                "semantic_plus_policy_id": sem_plus.get("policy_id", ""),
                "semantic_only_policy_id": sem_only.get("policy_id", ""),
                "semantic_plus_median_full_rel": plus_med,
                "semantic_only_median_full_rel": only_med,
                "semantic_plus_minus_semantic_only_median": (
                    plus_med - only_med if math.isfinite(plus_med) and math.isfinite(only_med) else float("nan")
                ),
                "semantic_plus_geometry_pass": bool_value(sem_plus.get("stage4_geometry_pass", False)),
                "semantic_only_geometry_pass": bool_value(sem_only.get("stage4_geometry_pass", False)),
                "semantic_causality_claim_allowed": False,
                "blocker": "semantic_plus_internal_and_semantic_only_are_not_separated_by_stage4; stronger controls deferred_to_stage6",
            }
        )
    return out


def build_report(summary: dict[str, Any], policy_rows: list[dict[str, Any]], semantic_rows: list[dict[str, Any]]) -> str:
    ranked = sorted(policy_rows, key=lambda row: safe_float(row.get("median_full_rel", "nan")), reverse=True)
    lines = [
        "# ACL2 v110R Stage4 Full KITTI Validation Report",
        "",
        f"stage4_geometry_pass_any: {summary['stage4_geometry_pass_any']}",
        f"stage4_strong_improvement_pass_any: {summary['stage4_strong_improvement_pass_any']}",
        f"taxonomy: {summary['taxonomy']}",
        f"blocker: {summary['blocker']}",
        "",
        "## Policy Summary",
        "",
    ]
    for row in ranked:
        lines.append(
            "- {policy_id}: median={median} mean={mean} improved={improved}/4 max_harm={harm} "
            "rolling={rolling} final={final} local_harm={local_harm} geometry_pass={geo} strong_pass={strong}".format(
                policy_id=row.get("policy_id", ""),
                median=row.get("median_full_rel", ""),
                mean=row.get("mean_full_rel", ""),
                improved=row.get("improved_seq_count", ""),
                harm=row.get("max_harm", ""),
                rolling=row.get("rolling_p90_median_rel", ""),
                final=row.get("final_error_median_rel", ""),
                local_harm=row.get("local_window_median_harm", ""),
                geo=row.get("stage4_geometry_pass", ""),
                strong=row.get("strong_improvement_pass", ""),
            )
        )
    lines.extend(["", "## Semantic Boundary", ""])
    for row in semantic_rows:
        lines.append(
            "- {candidate_id}: semantic_plus_median={plus} semantic_only_median={only} delta={delta} claim_allowed={claim}".format(
                candidate_id=row.get("candidate_id", ""),
                plus=row.get("semantic_plus_median_full_rel", ""),
                only=row.get("semantic_only_median_full_rel", ""),
                delta=row.get("semantic_plus_minus_semantic_only_median", ""),
                claim=row.get("semantic_causality_claim_allowed", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Stage4 validates four-sequence geometry only. Because semantic_plus_internal and semantic_only are still not separated, Stage4 alone does not authorize a semantic-causality method claim.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    stage3m.OUT = OUT
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
            row["schema"] = str(row.get("schema", "")).replace("acl2_v110r_stage3", "acl2_v110r_stage4")

    f19_control = f19_rows()
    fixed_controls = no_action_rows() + f19_control
    policy_rows = policy_summary_rows(full_rows, rolling_rows, fidelity_rows, f19_control)
    semantic_rows = semantic_stage4_rows(policy_rows)
    geometry_pass = [row for row in policy_rows if bool_value(row.get("stage4_geometry_pass"))]
    strong_pass = [row for row in policy_rows if bool_value(row.get("strong_improvement_pass"))]

    observed_counts = defaultdict(int)
    for row in latest.values():
        if stage2m.safe_rc(row) == 0:
            observed_counts[str(row.get("phase", ""))] += 1

    if strong_pass:
        taxonomy = "STAGE4_GEOMETRY_STRONG_PASS_SEMANTIC_CAUSALITY_PENDING"
        blocker = "stage6_semantic_causality_controls_pending"
    elif geometry_pass:
        taxonomy = "STAGE4_GEOMETRY_PASS_STRONG_IMPROVEMENT_FAIL"
        blocker = "strong_improvement_gate_not_met"
    else:
        taxonomy = "STAGE4_GEOMETRY_FAIL"
        blocker = "no_candidate_satisfied_full_four_sequence_geometry_gate"

    summary = {
        "schema": "acl2_v110r_stage4_summary_v1",
        "stage4_geometry_pass_any": bool(geometry_pass),
        "stage4_strong_improvement_pass_any": bool(strong_pass),
        "taxonomy": taxonomy,
        "blocker": blocker,
        "metric_complete": len(full_rows) == len(config_rows) and all(bool_value(row.get("metric_available")) for row in full_rows),
        "all_action_fidelity": len(fidelity_rows) == len(config_rows) and all(bool_value(row.get("action_fidelity_pass")) for row in fidelity_rows),
        "observed_prepare_count": observed_counts["prepare"],
        "observed_run_worker_count": observed_counts["run_worker"],
        "observed_evaluate_count": observed_counts["evaluate"],
        "observed_report_count": observed_counts["report"],
        "expected_run_worker_count": len(config_rows),
        "full_metric_row_count": len(full_rows),
        "rolling_metric_row_count": len(rolling_rows),
        "local_handoff_metric_row_count": len(local_rows),
        "action_fidelity_row_count": len(fidelity_rows),
        "fixed_control_row_count": len(fixed_controls),
        "policy_summary_row_count": len(policy_rows),
        "semantic_control_row_count": len(semantic_rows),
        "geometry_pass_policy_ids": [row["policy_id"] for row in geometry_pass],
        "strong_improvement_policy_ids": [row["policy_id"] for row in strong_pass],
        "best_policy_by_median_full_rel": (
            max(policy_rows, key=lambda row: safe_float(row.get("median_full_rel", "nan")))["policy_id"]
            if policy_rows else ""
        ),
        "f19_reference": {
            "median_full_rel": F19_MEDIAN_FULL_REL,
            "mean_full_rel": F19_MEAN_FULL_REL,
        },
        "outputs": {
            "full_metric_rows": rel(OUT / "full_metric_rows.csv"),
            "rolling_metric_rows": rel(OUT / "rolling_metric_rows.csv"),
            "local_handoff_metric_rows": rel(OUT / "local_handoff_metric_rows.csv"),
            "action_fidelity_rows": rel(OUT / "action_fidelity_rows.csv"),
            "fixed_control_rows": rel(OUT / "fixed_control_rows.csv"),
            "policy_summary_rows": rel(OUT / "policy_summary_rows.csv"),
            "semantic_control_rows": rel(OUT / "semantic_control_rows.csv"),
            "stage4_report": rel(OUT / "stage4_validation_report.md"),
        },
    }

    write_csv(OUT / "full_metric_rows.csv", full_rows)
    write_csv(OUT / "rolling_metric_rows.csv", rolling_rows)
    write_csv(OUT / "local_handoff_metric_rows.csv", local_rows)
    write_csv(OUT / "action_fidelity_rows.csv", fidelity_rows)
    write_csv(OUT / "fixed_control_rows.csv", fixed_controls)
    write_csv(OUT / "policy_summary_rows.csv", policy_rows)
    write_csv(OUT / "semantic_control_rows.csv", semantic_rows)
    write_json(OUT / "stage4_summary.json", summary)
    write_text(OUT / "stage4_validation_report.md", build_report(summary, policy_rows, semantic_rows))
    print(json.dumps(base.clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
