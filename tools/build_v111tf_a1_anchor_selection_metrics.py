#!/usr/bin/env python3
"""Summarize ACL2 v111TF A1 delayed anchor-frame selection metrics."""

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
A1 = RESULT_ROOT / "batch_a_a1_anchor_selection"
CONFIG_ROWS = A1 / "action_config_rows.csv"
RUN_RESULTS = A1 / "run_results.csv"
WORKSPACE = A1 / "workspace"
SEQUENCES = ("00", "01", "02", "05")
DEFAULT_POLICY = "A1_default_first_n"

V105 = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
V105_WORKSPACE = V105 / "stage1_lingbot_baseline/workspace/kitti_v105_00_01_02_05"
V105_METHOD = "lingbot_map_stream_default"

A1_MEDIAN_GATE_VS_DEFAULT = 0.05
MAX_HARM_GATE = 0.01
LOCAL_HARM_GATE = 0.02
RANDOM_POLICY_FAMILY = "random_same_first32"
SEMANTIC_POLICY_FAMILIES = {"topQ", "low_dynamic", "high_stable"}
EXPECTED_RANDOM_POLICY_COUNT = 21


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


def percentile_nearest_rank(values: list[float], quantile: float) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return float("nan")
    rank = max(1, math.ceil(quantile * len(finite)))
    return finite[min(rank, len(finite)) - 1]


def rel_improvement(base: float, action: float) -> float:
    if not math.isfinite(base) or not math.isfinite(action) or base == 0:
        return float("nan")
    return (base - action) / abs(base)


def parse_matrix(path: Path) -> list[float]:
    if not path.exists():
        return []
    values: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for part in line.split():
            try:
                value = float(part)
            except ValueError:
                return []
            if not math.isfinite(value):
                return []
            values.append(value)
    return values


def max_abs_diff(left: Path, right: Path) -> float:
    lv = parse_matrix(left)
    rv = parse_matrix(right)
    if not lv or not rv or len(lv) != len(rv):
        return float("nan")
    return max(abs(a - b) for a, b in zip(lv, rv)) if lv else 0.0


def a1_phase_status_for(cfg: dict[str, str], latest: dict[tuple[str, str], dict[str, str]]) -> tuple[dict[str, Any], bool, bool]:
    status: dict[str, Any] = {}
    all_metric_phase_success = True
    all_phase_success = True
    seq = cfg["seq"]
    for phase in ("prepare", "run_worker", "evaluate", "report"):
        if phase == "prepare":
            run_name = f"kitti_lingbot_v111tf_a1_prepare_{seq}"
        else:
            run_name = f"kitti_lingbot_v111tf_a1_{cfg['policy_id']}_{seq}_{phase}"
        row = latest.get((run_name, phase))
        rc = stage2m.safe_rc(row)
        status[f"{phase}_returncode"] = rc
        status[f"{phase}_duration_sec"] = row.get("duration_sec", "") if row else ""
        if phase in {"prepare", "run_worker", "evaluate"}:
            all_metric_phase_success = all_metric_phase_success and rc == 0
        all_phase_success = all_phase_success and rc == 0
    return status, all_metric_phase_success, all_phase_success


def install_a1_metric_overrides() -> None:
    stage2m.OUT = A1
    stage2m.CONFIG_ROWS = CONFIG_ROWS
    stage2m.RUN_RESULTS = RUN_RESULTS
    stage2m.WORKSPACE = WORKSPACE
    stage2m.SEQUENCES = SEQUENCES
    stage2m.phase_status_for = a1_phase_status_for

    original_action_fidelity = stage2m.action_fidelity_row

    def a1_action_fidelity_row(cfg: dict[str, str], latest: dict[tuple[str, str], dict[str, str]]) -> dict[str, Any]:
        row = original_action_fidelity(cfg, latest)
        run_name = f"kitti_lingbot_v111tf_a1_{cfg['policy_id']}_{cfg['seq']}_run_worker"
        run_row = latest.get((run_name, "run_worker"), {})
        row["schema"] = "acl2_v111tf_a1_action_fidelity_row_v1"
        row["candidate_id"] = "A1"
        row["run_worker_returncode"] = run_row.get("returncode", "")
        row["run_worker_duration_sec"] = run_row.get("duration_sec", "")
        row["M"] = cfg.get("M", "")
        row["num_anchor"] = cfg.get("num_anchor", "")
        row["scale_frame_indices"] = cfg.get("scale_frame_indices", "")
        row["latency_frames"] = cfg.get("latency_frames", "")
        row["latency_policy"] = cfg.get("latency_policy", "")
        return row

    stage2m.action_fidelity_row = a1_action_fidelity_row


def augment_rows(rows: list[dict[str, Any]], config_rows: list[dict[str, str]]) -> None:
    cfg_by_key = {(row["policy_id"], row["seq"]): row for row in config_rows}
    for row in rows:
        row["schema"] = str(row.get("schema", "")).replace("acl2_v109tf_stage2", "acl2_v111tf_a1")
        cfg = cfg_by_key.get((str(row.get("policy_id", "")), str(row.get("seq", ""))), {})
        for key in (
            "M",
            "num_anchor",
            "scale_frame_indices",
            "latency_frames",
            "latency_policy",
            "anchor_quality_mean",
            "dynamic_mass_mean",
            "stable_structure_mass_mean",
            "boundary_mass_mean",
            "weak_context_mass_mean",
        ):
            row[key] = cfg.get(key, "")


def default_v105_parity_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seq in SEQUENCES:
        a1_root = WORKSPACE / f"kitti_v111tf_a1_fullseq_{seq}" / seq / f"lingbot_map_v111tf_a1_{DEFAULT_POLICY}_{seq}"
        v105_root = V105_WORKSPACE / seq / V105_METHOD
        traj_diff = max_abs_diff(a1_root / "traj.txt", v105_root / "traj.txt")
        intr_diff = max_abs_diff(a1_root / "intrinsics.txt", v105_root / "intrinsics.txt")
        rows.append(
            {
                "schema": "acl2_v111tf_a1_default_v105_parity_row_v1",
                "seq": seq,
                "left_policy_id": DEFAULT_POLICY,
                "right_policy_id": "v105_lingbot_stream_default",
                "traj_max_abs_diff": traj_diff,
                "intrinsics_max_abs_diff": intr_diff,
                "parity_pass": math.isfinite(traj_diff)
                and math.isfinite(intr_diff)
                and traj_diff == 0.0
                and intr_diff == 0.0,
                "left_traj": rel(a1_root / "traj.txt"),
                "right_traj": rel(v105_root / "traj.txt"),
            }
        )
    return rows


def policy_summary_rows(
    full_rows: list[dict[str, Any]],
    rolling_rows: list[dict[str, Any]],
    fidelity_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rolling_by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fidelity_by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    full_by_key = {(row["policy_id"], row["seq"]): row for row in full_rows}
    rolling_by_key = {(row["policy_id"], row["seq"]): row for row in rolling_rows}
    for row in full_rows:
        by_policy[str(row["policy_id"])].append(row)
    for row in rolling_rows:
        rolling_by_policy[str(row["policy_id"])].append(row)
    for row in fidelity_rows:
        fidelity_by_policy[str(row["policy_id"])].append(row)

    out: list[dict[str, Any]] = []
    for policy_id in sorted(by_policy):
        rows = by_policy[policy_id]
        rels_v105_by_seq = {
            str(row["seq"]): safe_float(row.get("full_ATE_sim3_relative_improvement_vs_baseline", "nan"))
            for row in rows
        }
        rels_default_by_seq: dict[str, float] = {}
        rolling_default_by_seq: dict[str, float] = {}
        for seq in SEQUENCES:
            row = full_by_key.get((policy_id, seq), {})
            default = full_by_key.get((DEFAULT_POLICY, seq), {})
            rels_default_by_seq[seq] = rel_improvement(
                safe_float(default.get("full_ATE_sim3", "nan")),
                safe_float(row.get("full_ATE_sim3", "nan")),
            )
            rrow = rolling_by_key.get((policy_id, seq), {})
            rdefault = rolling_by_key.get((DEFAULT_POLICY, seq), {})
            rolling_default_by_seq[seq] = rel_improvement(
                safe_float(rdefault.get("action_rolling_ATE_p90", "nan")),
                safe_float(rrow.get("action_rolling_ATE_p90", "nan")),
            )
        rels_v105 = [rels_v105_by_seq.get(seq, float("nan")) for seq in SEQUENCES]
        rels_default = [rels_default_by_seq.get(seq, float("nan")) for seq in SEQUENCES]
        rolling_default = [rolling_default_by_seq.get(seq, float("nan")) for seq in SEQUENCES]
        finals = [safe_float(row.get("final_error_relative_improvement_vs_baseline", "nan")) for row in rows]
        locals_ = [safe_float(row.get("local_window_ATE_rel_improvement_vs_baseline_median", "nan")) for row in rows]
        median_v105 = stage3m.base.median(rels_v105)
        mean_v105 = stage3m.base.mean(rels_v105)
        median_default = stage3m.base.median(rels_default)
        mean_default = stage3m.base.mean(rels_default)
        improved_default = sum(1 for value in rels_default if math.isfinite(value) and value > 0.0)
        max_harm_default = stage3m.base.max_rel_harm(rels_default)
        rolling_default_median = stage3m.base.median(rolling_default)
        final_median = stage3m.base.median(finals)
        local_harm_v105 = stage3m.base.max_rel_harm(locals_)
        action_pass_count = sum(1 for row in fidelity_by_policy.get(policy_id, []) if bool_value(row.get("action_fidelity_pass")))
        metric_complete = len(rows) == len(SEQUENCES) and all(bool_value(row.get("metric_available")) for row in rows)
        all_action = action_pass_count == len(SEQUENCES)
        a1_geometry_gate = bool(
            policy_id != DEFAULT_POLICY
            and metric_complete
            and all_action
            and improved_default >= 3
            and math.isfinite(median_default)
            and median_default >= A1_MEDIAN_GATE_VS_DEFAULT
            and max_harm_default <= MAX_HARM_GATE
            and math.isfinite(rolling_default_median)
            and rolling_default_median > 0.0
            and local_harm_v105 <= LOCAL_HARM_GATE
        )
        sample = rows[0]
        out.append(
            {
                "schema": "acl2_v111tf_a1_policy_summary_row_v1",
                "candidate_id": "A1",
                "surface_id": sample.get("surface_id", ""),
                "policy_id": policy_id,
                "policy_family": sample.get("policy_family", ""),
                "sequence_count": len(rows),
                "metric_complete": metric_complete,
                "action_fidelity_pass_count": action_pass_count,
                "all_action_fidelity": all_action,
                "median_full_rel_vs_v105": median_v105,
                "mean_full_rel_vs_v105": mean_v105,
                "median_full_rel_vs_a1_default": median_default,
                "mean_full_rel_vs_a1_default": mean_default,
                "improved_seq_count_vs_a1_default": improved_default,
                "max_harm_vs_a1_default": max_harm_default,
                "rolling_p90_median_rel_vs_a1_default": rolling_default_median,
                "final_error_median_rel_vs_v105": final_median,
                "local_window_median_harm_vs_v105": local_harm_v105,
                "seq00_full_rel_vs_a1_default": rels_default_by_seq.get("00", ""),
                "seq01_full_rel_vs_a1_default": rels_default_by_seq.get("01", ""),
                "seq02_full_rel_vs_a1_default": rels_default_by_seq.get("02", ""),
                "seq05_full_rel_vs_a1_default": rels_default_by_seq.get("05", ""),
                "seq00_full_rel_vs_v105": rels_v105_by_seq.get("00", ""),
                "seq01_full_rel_vs_v105": rels_v105_by_seq.get("01", ""),
                "seq02_full_rel_vs_v105": rels_v105_by_seq.get("02", ""),
                "seq05_full_rel_vs_v105": rels_v105_by_seq.get("05", ""),
                "M": sample.get("M", ""),
                "num_anchor": sample.get("num_anchor", ""),
                "latency_frames_max": max(safe_float(row.get("latency_frames", "nan")) for row in rows),
                "anchor_quality_mean_avg": stage3m.base.mean([safe_float(row.get("anchor_quality_mean", "nan")) for row in rows]),
                "dynamic_mass_mean_avg": stage3m.base.mean([safe_float(row.get("dynamic_mass_mean", "nan")) for row in rows]),
                "stable_structure_mass_mean_avg": stage3m.base.mean([safe_float(row.get("stable_structure_mass_mean", "nan")) for row in rows]),
                "a1_geometry_gate_pass": a1_geometry_gate,
                "a1_median_gate_vs_default": A1_MEDIAN_GATE_VS_DEFAULT,
                "claim_boundary": "A1 semantic anchor selection requires random P95 controls after geometry gate before semantic causality claim.",
            }
        )
    return out


def build_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    ranked = sorted(rows, key=lambda row: safe_float(row.get("median_full_rel_vs_a1_default", "nan")), reverse=True)
    lines = [
        "# ACL2 v111TF A1 Anchor Selection Report",
        "",
        f"metric_complete: {summary['metric_complete']}",
        f"all_action_fidelity: {summary['all_action_fidelity']}",
        f"default_v105_parity_pass: {summary['a1_default_v105_parity_pass']}",
        f"taxonomy: {summary['taxonomy']}",
        f"blocker: {summary['blocker']}",
        f"random_p95_median_vs_default: {summary.get('random_same_first32_p95_median_full_rel_vs_a1_default', '')}",
        f"semantic_random_p95_pass_policy_ids: {summary.get('a1_semantic_random_p95_pass_policy_ids', [])}",
        "",
        "## Policy Ranking Vs A1 Default",
        "",
    ]
    for row in ranked:
        lines.append(
            "- {policy_id}: median_vs_default={median_d} mean_vs_default={mean_d} improved={improved}/4 "
            "max_harm={harm} rolling_vs_default={rolling} median_vs_v105={median_v105} gate={gate}".format(
                policy_id=row.get("policy_id", ""),
                median_d=row.get("median_full_rel_vs_a1_default", ""),
                mean_d=row.get("mean_full_rel_vs_a1_default", ""),
                improved=row.get("improved_seq_count_vs_a1_default", ""),
                harm=row.get("max_harm_vs_a1_default", ""),
                rolling=row.get("rolling_p90_median_rel_vs_a1_default", ""),
                median_v105=row.get("median_full_rel_vs_v105", ""),
                gate=row.get("a1_geometry_gate_pass", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "A1 changes delayed anchor/scale initialization frame selection. When random same-first32 controls are present, the semantic gate compares geometry-pass semantic policies against their nearest-rank P95.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    install_a1_metric_overrides()
    config_rows = read_csv(CONFIG_ROWS)
    latest = stage2m.latest_run_results(read_csv(RUN_RESULTS))
    full_rows, rolling_rows, local_rows, fidelity_rows = stage2m.metric_rows(config_rows, latest)
    for rows in (full_rows, rolling_rows, local_rows, fidelity_rows):
        augment_rows(rows, config_rows)
    parity_rows = default_v105_parity_rows()
    policy_rows = policy_summary_rows(full_rows, rolling_rows, fidelity_rows)

    observed_counts: dict[str, int] = {}
    for row in latest.values():
        if stage2m.safe_rc(row) == 0:
            phase = str(row.get("phase", ""))
            observed_counts[phase] = observed_counts.get(phase, 0) + 1

    metric_complete = len(full_rows) == len(config_rows) and all(bool_value(row.get("metric_available")) for row in full_rows)
    all_action = len(fidelity_rows) == len(config_rows) and all(bool_value(row.get("action_fidelity_pass")) for row in fidelity_rows)
    parity_pass = len(parity_rows) == len(SEQUENCES) and all(bool_value(row.get("parity_pass")) for row in parity_rows)
    geometry_pass = [row for row in policy_rows if bool_value(row.get("a1_geometry_gate_pass"))]
    semantic_rows = [row for row in policy_rows if row.get("policy_family") in SEMANTIC_POLICY_FAMILIES]
    semantic_geometry_pass = [row for row in geometry_pass if row.get("policy_family") in SEMANTIC_POLICY_FAMILIES]
    random_rows = [row for row in policy_rows if row.get("policy_family") == RANDOM_POLICY_FAMILY]
    random_values = [safe_float(row.get("median_full_rel_vs_a1_default", "nan")) for row in random_rows]
    random_p95 = percentile_nearest_rank(random_values, 0.95)
    random_max = max((value for value in random_values if math.isfinite(value)), default=float("nan"))
    random_controls_complete = (
        len(random_rows) == EXPECTED_RANDOM_POLICY_COUNT
        and all(bool_value(row.get("metric_complete")) for row in random_rows)
        and all(bool_value(row.get("all_action_fidelity")) for row in random_rows)
    )
    semantic_random_p95_pass = [
        row
        for row in semantic_geometry_pass
        if math.isfinite(safe_float(row.get("median_full_rel_vs_a1_default", "nan")))
        and math.isfinite(random_p95)
        and safe_float(row.get("median_full_rel_vs_a1_default", "nan")) > random_p95
    ]
    best = max(policy_rows, key=lambda row: safe_float(row.get("median_full_rel_vs_a1_default", "nan"))) if policy_rows else {}
    best_semantic = (
        max(semantic_rows, key=lambda row: safe_float(row.get("median_full_rel_vs_a1_default", "nan")))
        if semantic_rows
        else {}
    )
    if not metric_complete:
        taxonomy = "A1_METRICS_INCOMPLETE"
        blocker = "not_all_manifest_rows_have_metric_outputs"
    elif not parity_pass:
        taxonomy = "A1_HOOK_PARITY_FAIL"
        blocker = "default_first_n_scale_frame_indices_does_not_match_v105_default"
    elif semantic_geometry_pass and not random_controls_complete:
        taxonomy = "A1_GEOMETRY_PASS_RANDOM_P95_PENDING"
        blocker = "random_same_firstM_seed0_to_20_controls_not_run"
    elif semantic_random_p95_pass:
        taxonomy = "A1_GEOMETRY_PASS_RANDOM_P95_PASS"
        blocker = ""
    elif semantic_geometry_pass:
        taxonomy = "A1_GEOMETRY_PASS_RANDOM_P95_FAIL"
        blocker = "random_same_first32_p95_matches_or_exceeds_all_semantic_geometry_pass_candidates"
    elif geometry_pass:
        taxonomy = "A1_RANDOM_CONTROL_GEOMETRY_PASS_NO_SEMANTIC_POLICY_PASS"
        blocker = "only_random_or_control_policy_satisfied_geometry_gate"
    else:
        taxonomy = "A1_GEOMETRY_FAIL_OR_BELOW_DEFAULT_GATE"
        blocker = "no_anchor_selection_policy_satisfied_default_relative_geometry_gate"

    summary = {
        "schema": "acl2_v111tf_a1_anchor_selection_metric_summary_v1",
        "metric_complete": metric_complete,
        "all_action_fidelity": all_action,
        "a1_default_v105_parity_pass": parity_pass,
        "taxonomy": taxonomy,
        "blocker": blocker,
        "observed_prepare_count": observed_counts.get("prepare", 0),
        "observed_run_worker_count": observed_counts.get("run_worker", 0),
        "observed_evaluate_count": observed_counts.get("evaluate", 0),
        "observed_report_count": observed_counts.get("report", 0),
        "expected_prepare_count": len(SEQUENCES),
        "expected_run_worker_count": len(config_rows),
        "full_metric_row_count": len(full_rows),
        "rolling_metric_row_count": len(rolling_rows),
        "local_handoff_metric_row_count": len(local_rows),
        "action_fidelity_row_count": len(fidelity_rows),
        "policy_summary_row_count": len(policy_rows),
        "a1_geometry_pass_policy_ids": [row["policy_id"] for row in geometry_pass],
        "a1_semantic_geometry_pass_policy_ids": [row["policy_id"] for row in semantic_geometry_pass],
        "a1_random_geometry_pass_policy_ids": [
            row["policy_id"] for row in geometry_pass if row.get("policy_family") == RANDOM_POLICY_FAMILY
        ],
        "a1_semantic_random_p95_pass_policy_ids": [row["policy_id"] for row in semantic_random_p95_pass],
        "best_policy_by_median_vs_default": best.get("policy_id", ""),
        "best_policy_median_full_rel_vs_a1_default": best.get("median_full_rel_vs_a1_default", ""),
        "best_policy_mean_full_rel_vs_a1_default": best.get("mean_full_rel_vs_a1_default", ""),
        "best_semantic_policy_by_median_vs_default": best_semantic.get("policy_id", ""),
        "best_semantic_policy_median_full_rel_vs_a1_default": best_semantic.get(
            "median_full_rel_vs_a1_default", ""
        ),
        "best_semantic_policy_mean_full_rel_vs_a1_default": best_semantic.get("mean_full_rel_vs_a1_default", ""),
        "random_same_first32_policy_count": len(random_rows),
        "random_same_first32_controls_complete": random_controls_complete,
        "random_same_first32_p95_median_full_rel_vs_a1_default": random_p95,
        "random_same_first32_max_median_full_rel_vs_a1_default": random_max,
        "random_same_first32_median_values_sorted": sorted(
            value for value in random_values if math.isfinite(value)
        ),
        "a1_plan_reference": {
            "median_full_rel_vs_default_gate": A1_MEDIAN_GATE_VS_DEFAULT,
            "max_harm_gate": MAX_HARM_GATE,
            "local_harm_gate_vs_v105": LOCAL_HARM_GATE,
            "random_p95_rule": "semantic geometry-pass policy must exceed nearest-rank P95 of A1_random_same_first32_seed0..20 median_full_rel_vs_a1_default",
        },
        "semantic_causality_claim_allowed": bool(semantic_random_p95_pass),
        "semantic_causality_claim_blocker": ""
        if semantic_random_p95_pass
        else (
            "A1 random same-first32 controls incomplete."
            if semantic_geometry_pass and not random_controls_complete
            else "A1 semantic geometry-pass policy does not exceed random same-first32 P95."
            if semantic_geometry_pass
            else "No A1 semantic policy passed the geometry gate."
        ),
        "outputs": {
            "full_metric_rows": rel(A1 / "full_metric_rows.csv"),
            "rolling_metric_rows": rel(A1 / "rolling_metric_rows.csv"),
            "local_handoff_metric_rows": rel(A1 / "local_handoff_metric_rows.csv"),
            "action_fidelity_rows": rel(A1 / "action_fidelity_rows.csv"),
            "default_v105_parity_rows": rel(A1 / "a1_default_v105_parity_rows.csv"),
            "policy_summary_rows": rel(A1 / "policy_summary_rows.csv"),
            "report": rel(A1 / "A1_ANCHOR_SELECTION_REPORT.md"),
            "summary": rel(A1 / "a1_metric_summary.json"),
        },
    }

    write_csv(A1 / "full_metric_rows.csv", full_rows)
    write_csv(A1 / "rolling_metric_rows.csv", rolling_rows)
    write_csv(A1 / "local_handoff_metric_rows.csv", local_rows)
    write_csv(A1 / "action_fidelity_rows.csv", fidelity_rows)
    write_csv(A1 / "a1_default_v105_parity_rows.csv", parity_rows)
    write_csv(A1 / "policy_summary_rows.csv", policy_rows)
    write_json(A1 / "a1_metric_summary.json", summary)
    write_text(A1 / "A1_ANCHOR_SELECTION_REPORT.md", build_report(summary, policy_rows))
    print(json.dumps(stage3m.base.clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
