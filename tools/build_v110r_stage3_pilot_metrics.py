#!/usr/bin/env python3
"""Build ACL2 v110R Stage3 KITTI 00/02 pilot metrics and gate summary."""

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


base = stage2m.base

RESULT_ROOT = ROOT / "results/acl2_v110r_lingbot_multisurface_full_ate_boost_semantic_causality"
OUT = RESULT_ROOT / "stage3_pilot_00_02"
CONFIG_ROWS = OUT / "action_config_rows.csv"
RUN_RESULTS = OUT / "run_results.csv"
WORKSPACE = OUT / "workspace"
SEQUENCES = ("00", "02")

MEDIAN_FULL_REL_MIN = 0.10
MIN_SEQ_FULL_REL_MIN = 0.0
ROLLING_P90_MEDIAN_REL_MIN = 0.05
LOCAL_WINDOW_MAX_HARM = 0.02
F19_SEQ_REL = {
    "00": 0.28131771113956844,
    "02": 0.09675451430025214,
}
F19_CLOSE_REL_TOL = 0.02
PROMOTION_FAMILIES = {"semantic_plus_internal", "semantic_only", "internal_only", "same_count_random"}
SEMANTIC_FAMILIES = {"semantic_plus_internal", "semantic_only"}
INTERNAL_BASELINE_FAMILIES = {"internal_only", "same_count_random"}


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


def safe_rc(row: dict[str, str] | None) -> int:
    return stage2m.safe_rc(row)


def finite(values: list[float]) -> list[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def candidate_id_from_policy(policy_id: str) -> str:
    return policy_id.split("_", 1)[0] if "_" in policy_id else policy_id


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def phase_status_for(
    cfg: dict[str, str],
    latest: dict[tuple[str, str], dict[str, str]],
) -> tuple[dict[str, Any], bool, bool]:
    status: dict[str, Any] = {}
    all_metric_phase_success = True
    all_phase_success = True
    seq = cfg["seq"]
    for phase in ("prepare", "run_worker", "evaluate", "report"):
        if phase == "prepare":
            run_name = f"kitti_lingbot_v110r_stage3_prepare_{seq}"
        else:
            run_name = f"kitti_lingbot_v110r_stage3_{cfg['policy_id']}_{seq}_{phase}"
        row = latest.get((run_name, phase))
        rc = safe_rc(row)
        status[f"{phase}_returncode"] = rc
        status[f"{phase}_duration_sec"] = row.get("duration_sec", "") if row else ""
        if phase in {"prepare", "run_worker", "evaluate"}:
            all_metric_phase_success = all_metric_phase_success and rc == 0
        all_phase_success = all_phase_success and rc == 0
    return status, all_metric_phase_success, all_phase_success


def action_fidelity_row(
    cfg: dict[str, str],
    latest: dict[tuple[str, str], dict[str, str]],
) -> dict[str, Any]:
    action_file = Path(cfg["action_file"])
    action_rows = base.load_jsonl(action_file)
    expected = base.parse_indices(cfg.get("selected_global_frame_indices", ""))
    expected_field = cfg.get("expected_action_field", "")
    mode = cfg.get("stage3_action_mode") or cfg.get("stage4_action_mode") or cfg.get("stage2_action_mode", "")
    observed = {
        int(float(row.get("sample_position", -1)))
        for row in action_rows
        if base.boolish(row.get(expected_field, False))
    }
    base_keyframes = {
        int(float(row.get("sample_position", -1)))
        for row in action_rows
        if base.boolish(row.get("base_is_keyframe", False))
    }
    final_keyframes = {
        int(float(row.get("sample_position", -1)))
        for row in action_rows
        if base.boolish(row.get("final_is_keyframe", False))
    }
    effective: set[int] = set()
    trace_error_rows = 0
    for row in action_rows:
        try:
            sample = int(float(row.get("sample_position", -1)))
        except ValueError:
            trace_error_rows += 1
            continue
        if mode == "anchor_special_only":
            if (
                base.boolish(row.get("forced_anchor_only", False))
                and base.boolish(row.get("forced_context_only", False))
                and base.boolish(row.get("context_only_append", False))
                and str(row.get("context_only_special_mode", "")) == "scale_only"
            ):
                effective.add(sample)
        elif base.boolish(row.get(expected_field, False)):
            effective.add(sample)

    missing = expected - observed
    unexpected = observed - expected
    ineffective = expected - effective
    action_fidelity_pass = (
        action_file.exists()
        and observed == expected
        and effective == expected
        and trace_error_rows == 0
    )
    run_name = f"kitti_lingbot_v110r_stage3_{cfg['policy_id']}_{cfg['seq']}_run_worker"
    run_row = latest.get((run_name, "run_worker"), {})
    return {
        "schema": "acl2_v110r_stage3_action_fidelity_row_v1",
        "candidate_id": cfg.get("candidate_id", candidate_id_from_policy(cfg["policy_id"])),
        "surface_id": cfg["surface_id"],
        "policy_id": cfg["policy_id"],
        "policy_family": cfg["policy_family"],
        "seq": cfg["seq"],
        "dataset": cfg["dataset"],
        "method": cfg["method"],
        "action_name": cfg["action_name"],
        "stage3_action_mode": mode,
        "stage4_action_mode": cfg.get("stage4_action_mode", mode),
        "expected_action_field": expected_field,
        "expected_action_frame_count": len(expected),
        "observed_action_frame_count": len(observed),
        "action_effective_frame_count": len(effective),
        "action_noop_frame_count": len(ineffective),
        "expected_keyframe_count": len(expected),
        "observed_keyframe_count": len(observed & base_keyframes),
        "special_token_operation_count": len(effective) if mode == "anchor_special_only" else "",
        "trace_error_rows": trace_error_rows,
        "action_file_exists": action_file.exists(),
        "action_fidelity_pass": action_fidelity_pass,
        "observed_action_indices": ";".join(str(x) for x in sorted(observed)),
        "effective_action_indices": ";".join(str(x) for x in sorted(effective)),
        "missing_expected_indices": ";".join(str(x) for x in sorted(missing)),
        "unexpected_observed_indices": ";".join(str(x) for x in sorted(unexpected)),
        "ineffective_expected_indices": ";".join(str(x) for x in sorted(ineffective)),
        "base_keyframe_count_observed_log": len(base_keyframes),
        "final_keyframe_count_observed_log": len(final_keyframes),
        "action_log_rows": len(action_rows),
        "action_file": rel(action_file),
        "run_worker_returncode": run_row.get("returncode", ""),
        "run_worker_duration_sec": run_row.get("duration_sec", ""),
    }


def install_stage3_overrides() -> None:
    stage2m.OUT = OUT
    stage2m.CONFIG_ROWS = CONFIG_ROWS
    stage2m.RUN_RESULTS = RUN_RESULTS
    stage2m.WORKSPACE = WORKSPACE
    stage2m.SEQUENCES = SEQUENCES
    stage2m.phase_status_for = phase_status_for
    stage2m.action_fidelity_row = action_fidelity_row


def add_candidate_metadata(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        policy_id = str(row.get("policy_id", ""))
        row.setdefault("candidate_id", candidate_id_from_policy(policy_id))
        if row.get("schema"):
            row["schema"] = str(row["schema"]).replace("acl2_v109tf_stage2", "acl2_v110r_stage3")


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

    median_by_policy = {
        policy_id: base.median([
            safe_float(row.get("full_ATE_sim3_relative_improvement_vs_baseline", "nan"))
            for row in rows
        ])
        for policy_id, rows in by_policy.items()
    }
    out: list[dict[str, Any]] = []
    for policy_id in sorted(by_policy):
        rows = by_policy[policy_id]
        candidate_id = candidate_id_from_policy(policy_id)
        family = str(rows[0].get("policy_family", ""))
        rels = [
            safe_float(row.get("full_ATE_sim3_relative_improvement_vs_baseline", "nan"))
            for row in rows
        ]
        final_rels = [
            safe_float(row.get("final_error_relative_improvement_vs_baseline", "nan"))
            for row in rows
        ]
        local_rels = [
            safe_float(row.get("local_window_ATE_rel_improvement_vs_baseline_median", "nan"))
            for row in rows
        ]
        rolling_rels = [
            safe_float(row.get("rolling_ATE_p90_relative_improvement_vs_baseline", "nan"))
            for row in rolling_by_policy.get(policy_id, [])
        ]
        fids = fidelity_by_policy.get(policy_id, [])
        action_fidelity_pass_count = sum(1 for row in fids if bool_value(row.get("action_fidelity_pass")))
        metric_seq_count = len(rows)
        all_metrics_available = metric_seq_count == len(SEQUENCES) and all(bool_value(row.get("metric_available")) for row in rows)
        action_fidelity_pass_all = action_fidelity_pass_count == len(SEQUENCES)
        median_full = base.median(rels)
        min_full = min(finite(rels), default=float("nan"))
        median_rolling = base.median(rolling_rels)
        local_harm = base.max_rel_harm(local_rels)
        internal_median = median_by_policy.get(f"{candidate_id}_internal_only", float("nan"))
        if family in INTERNAL_BASELINE_FAMILIES:
            semantic_or_internal_condition = True
            stage3_role = "internal_or_schedule_baseline"
        elif family in SEMANTIC_FAMILIES:
            semantic_or_internal_condition = math.isfinite(internal_median) and median_full > internal_median
            stage3_role = "semantic_candidate"
        else:
            semantic_or_internal_condition = False
            stage3_role = "control_only"
        full_gate = (
            math.isfinite(median_full)
            and median_full >= MEDIAN_FULL_REL_MIN
            and math.isfinite(min_full)
            and min_full >= MIN_SEQ_FULL_REL_MIN
        )
        rolling_gate = math.isfinite(median_rolling) and median_rolling >= ROLLING_P90_MEDIAN_REL_MIN
        local_gate = local_harm <= LOCAL_WINDOW_MAX_HARM
        stage3_candidate_pass = bool(
            all_metrics_available
            and full_gate
            and rolling_gate
            and local_gate
            and action_fidelity_pass_all
            and semantic_or_internal_condition
            and family in PROMOTION_FAMILIES
        )
        f19_close_count = 0
        for row in rows:
            seq = str(row.get("seq", ""))
            rel_value = safe_float(row.get("full_ATE_sim3_relative_improvement_vs_baseline", "nan"))
            if math.isfinite(rel_value) and seq in F19_SEQ_REL and rel_value >= F19_SEQ_REL[seq] - F19_CLOSE_REL_TOL:
                f19_close_count += 1
        out.append(
            {
                "schema": "acl2_v110r_stage3_policy_summary_row_v1",
                "candidate_id": candidate_id,
                "surface_id": rows[0].get("surface_id", ""),
                "policy_id": policy_id,
                "policy_family": family,
                "stage3_role": stage3_role,
                "sequence_count": metric_seq_count,
                "all_metrics_available": all_metrics_available,
                "action_fidelity_pass_count": action_fidelity_pass_count,
                "action_fidelity_pass_all": action_fidelity_pass_all,
                "median_full_rel_00_02": median_full,
                "mean_full_rel_00_02": base.mean(rels),
                "min_seq_full_rel_00_02": min_full,
                "improved_seq_count_00_02": sum(1 for value in rels if math.isfinite(value) and value > 0.0),
                "max_full_harm_00_02": base.max_rel_harm(rels),
                "median_final_error_rel_00_02": base.median(final_rels),
                "rolling_p90_median_rel_00_02": median_rolling,
                "local_window_median_harm": local_harm,
                "same_candidate_internal_only_median_full_rel_00_02": internal_median,
                "median_full_rel_minus_internal_only": (
                    median_full - internal_median
                    if math.isfinite(median_full) and math.isfinite(internal_median)
                    else float("nan")
                ),
                "full_gate_pass": full_gate,
                "rolling_gate_pass": rolling_gate,
                "local_gate_pass": local_gate,
                "semantic_plus_internal_or_internal_baseline_condition_pass": semantic_or_internal_condition,
                "stage3_candidate_pass": stage3_candidate_pass,
                "f19_close_sequence_count": f19_close_count,
                "promotion_note": "eligible_for_stage4_selection" if stage3_candidate_pass else "not_promoted_by_stage3_gate",
            }
        )
    return out


def semantic_control_rows(policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_candidate_family: dict[tuple[str, str], dict[str, Any]] = {}
    for row in policy_rows:
        by_candidate_family[(str(row["candidate_id"]), str(row["policy_family"]))] = row

    candidates = sorted({str(row["candidate_id"]) for row in policy_rows})
    out: list[dict[str, Any]] = []
    for candidate_id in candidates:
        semantic_plus = by_candidate_family.get((candidate_id, "semantic_plus_internal"), {})
        internal = by_candidate_family.get((candidate_id, "internal_only"), {})
        semantic_only = by_candidate_family.get((candidate_id, "semantic_only"), {})
        shuffle = by_candidate_family.get((candidate_id, "semantic_shuffle"), {})
        random = by_candidate_family.get((candidate_id, "same_count_random"), {})
        low_reverse = by_candidate_family.get((candidate_id, "low_risk_reverse"), {})
        sem_med = safe_float(semantic_plus.get("median_full_rel_00_02", "nan"))
        internal_med = safe_float(internal.get("median_full_rel_00_02", "nan"))
        shuffle_med = safe_float(shuffle.get("median_full_rel_00_02", "nan"))
        random_med = safe_float(random.get("median_full_rel_00_02", "nan"))
        out.append(
            {
                "schema": "acl2_v110r_stage3_semantic_control_row_v1",
                "candidate_id": candidate_id,
                "surface_id": semantic_plus.get("surface_id") or internal.get("surface_id") or "",
                "semantic_plus_policy_id": semantic_plus.get("policy_id", ""),
                "semantic_plus_median_full_rel_00_02": sem_med,
                "internal_only_median_full_rel_00_02": internal_med,
                "semantic_only_median_full_rel_00_02": semantic_only.get("median_full_rel_00_02", ""),
                "semantic_shuffle_median_full_rel_00_02": shuffle_med,
                "same_count_random_median_full_rel_00_02": random_med,
                "low_risk_reverse_median_full_rel_00_02": low_reverse.get("median_full_rel_00_02", ""),
                "semantic_plus_minus_internal": (
                    sem_med - internal_med if math.isfinite(sem_med) and math.isfinite(internal_med) else float("nan")
                ),
                "semantic_plus_minus_semantic_shuffle": (
                    sem_med - shuffle_med if math.isfinite(sem_med) and math.isfinite(shuffle_med) else float("nan")
                ),
                "semantic_plus_minus_same_count_random": (
                    sem_med - random_med if math.isfinite(sem_med) and math.isfinite(random_med) else float("nan")
                ),
                "semantic_plus_beats_internal_only": math.isfinite(sem_med) and math.isfinite(internal_med) and sem_med > internal_med,
                "semantic_plus_stage3_candidate_pass": bool_value(semantic_plus.get("stage3_candidate_pass", False)),
                "control_boundary": "Stage3 alone cannot claim semantic causality; role_rotation/same_bucket_random/schedule_only were not generated in this pass.",
            }
        )
    return out


def build_report(summary: dict[str, Any], policy_rows: list[dict[str, Any]], semantic_rows: list[dict[str, Any]]) -> str:
    top = sorted(
        policy_rows,
        key=lambda row: safe_float(row.get("median_full_rel_00_02", "nan")),
        reverse=True,
    )
    lines = [
        "# ACL2 v110R Stage3 KITTI 00/02 Pilot Report",
        "",
        f"stage3_pass: {summary['stage3_pass']}",
        f"taxonomy: {summary['taxonomy']}",
        f"blocker: {summary['blocker']}",
        f"observed_run_worker: {summary['observed_run_worker_count']}/{summary['expected_run_worker_count']}",
        f"observed_evaluate: {summary['observed_evaluate_count']}/{summary['expected_evaluate_count']}",
        f"observed_report: {summary['observed_report_count']}/{summary['expected_report_count']}",
        f"all_action_fidelity: {summary['all_action_fidelity']}",
        "",
        "## Stage3 Gate",
        "",
        "- median_full_rel_00_02 >= 0.10",
        "- min_seq_full_rel_00_02 >= 0",
        "- rolling_p90_median_rel_00_02 >= 0.05",
        "- local_window_median_harm <= 0.02",
        "- action_fidelity_pass_all = true",
        "- semantic_plus_internal > internal_only, unless policy is explicit internal/schedule baseline",
        "",
        "## Top Policies By Median Full Rel",
        "",
    ]
    for row in top[:12]:
        lines.append(
            "- {policy_id}: family={family} candidate={candidate} median={median} min={min_rel} "
            "rolling={rolling} local_harm={local_harm} action={action} pass={passed}".format(
                policy_id=row.get("policy_id", ""),
                family=row.get("policy_family", ""),
                candidate=row.get("candidate_id", ""),
                median=row.get("median_full_rel_00_02", ""),
                min_rel=row.get("min_seq_full_rel_00_02", ""),
                rolling=row.get("rolling_p90_median_rel_00_02", ""),
                local_harm=row.get("local_window_median_harm", ""),
                action=row.get("action_fidelity_pass_all", ""),
                passed=row.get("stage3_candidate_pass", ""),
            )
        )
    lines.extend(["", "## Stage4 Selected Policies", ""])
    if summary["stage4_selected_policy_ids"]:
        for policy_id in summary["stage4_selected_policy_ids"]:
            lines.append(f"- {policy_id}")
    else:
        lines.append("- none")
    lines.extend(["", "## Semantic Control Boundary", ""])
    for row in semantic_rows:
        lines.append(
            "- {candidate_id}: sem_plus={sem} internal={internal} shuffle={shuffle} same_count_random={random} "
            "sem_minus_internal={delta} sem_pass={sem_pass}".format(
                candidate_id=row.get("candidate_id", ""),
                sem=row.get("semantic_plus_median_full_rel_00_02", ""),
                internal=row.get("internal_only_median_full_rel_00_02", ""),
                shuffle=row.get("semantic_shuffle_median_full_rel_00_02", ""),
                random=row.get("same_count_random_median_full_rel_00_02", ""),
                delta=row.get("semantic_plus_minus_internal", ""),
                sem_pass=row.get("semantic_plus_stage3_candidate_pass", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Stage3 is a 00/02 full-sequence pilot only. A pass nominates Stage4 validation; it is not a four-sequence method success and not a semantic-causality success.",
        ]
    )
    return "\n".join(lines)


def build_partial_signal(policy_rows: list[dict[str, Any]]) -> str:
    near = [row for row in policy_rows if int(row.get("f19_close_sequence_count", 0)) > 0]
    lines = [
        "# ACL2 v110R Stage3 Partial Signal Report",
        "",
        "Rows here did not necessarily pass Stage3. They are listed because at least one of KITTI 00/02 is within the registered F19-close tolerance.",
        "",
    ]
    if not near:
        lines.append("No F19-close partial signals under the current tolerance.")
    for row in sorted(near, key=lambda item: safe_float(item.get("median_full_rel_00_02", "nan")), reverse=True):
        lines.append(
            "- {policy_id}: family={family} median_full_rel_00_02={median} f19_close_sequence_count={count} "
            "stage3_candidate_pass={passed}".format(
                policy_id=row.get("policy_id", ""),
                family=row.get("policy_family", ""),
                median=row.get("median_full_rel_00_02", ""),
                count=row.get("f19_close_sequence_count", ""),
                passed=row.get("stage3_candidate_pass", ""),
            )
        )
    return "\n".join(lines)


def main() -> None:
    install_stage3_overrides()
    config_rows = read_csv(CONFIG_ROWS)
    run_rows = read_csv(RUN_RESULTS)
    latest = stage2m.latest_run_results(run_rows)
    full_rows, rolling_rows, local_rows, fidelity_rows = stage2m.metric_rows(config_rows, latest)
    for rows in (full_rows, rolling_rows, local_rows, fidelity_rows):
        add_candidate_metadata(rows)
    policy_rows = policy_summary_rows(full_rows, rolling_rows, fidelity_rows)
    semantic_rows = semantic_control_rows(policy_rows)

    expected_run_worker_count = len(config_rows)
    expected_prepare_count = len(SEQUENCES)
    observed_run_worker_count = sum(1 for row in latest.values() if row.get("phase") == "run_worker" and safe_rc(row) == 0)
    observed_evaluate_count = sum(1 for row in latest.values() if row.get("phase") == "evaluate" and safe_rc(row) == 0)
    observed_report_count = sum(1 for row in latest.values() if row.get("phase") == "report" and safe_rc(row) == 0)
    observed_prepare_count = sum(1 for row in latest.values() if row.get("phase") == "prepare" and safe_rc(row) == 0)
    metric_complete = (
        len(full_rows) == expected_run_worker_count
        and observed_prepare_count >= expected_prepare_count
        and observed_run_worker_count >= expected_run_worker_count
        and observed_evaluate_count >= expected_run_worker_count
        and all(bool_value(row.get("metric_available")) for row in full_rows)
    )
    all_action_fidelity = len(fidelity_rows) == expected_run_worker_count and all(
        bool_value(row.get("action_fidelity_pass")) for row in fidelity_rows
    )
    selected = [
        row for row in policy_rows
        if bool_value(row.get("stage3_candidate_pass")) and str(row.get("policy_family")) in PROMOTION_FAMILIES
    ]
    selected = sorted(selected, key=lambda row: safe_float(row.get("median_full_rel_00_02", "nan")), reverse=True)
    if not metric_complete:
        taxonomy = "STAGE3_METRICS_NOT_COMPLETE"
        blocker = "stage3_metrics_not_complete"
    elif not all_action_fidelity:
        taxonomy = "STAGE3_ACTION_FIDELITY_FAIL"
        blocker = "stage3_action_fidelity_fail"
    elif selected:
        taxonomy = "STAGE3_CANDIDATES_PASS_FOR_STAGE4_VALIDATION"
        blocker = ""
    else:
        taxonomy = "STAGE3_NO_CANDIDATE_PASS"
        blocker = "no_policy_satisfied_all_stage3_gate_conditions"

    summary = {
        "schema": "acl2_v110r_stage3_summary_v1",
        "stage3_pass": bool(selected),
        "taxonomy": taxonomy,
        "blocker": blocker,
        "metric_complete": metric_complete,
        "all_action_fidelity": all_action_fidelity,
        "expected_prepare_count": expected_prepare_count,
        "observed_prepare_count": observed_prepare_count,
        "expected_run_worker_count": expected_run_worker_count,
        "observed_run_worker_count": observed_run_worker_count,
        "expected_evaluate_count": expected_run_worker_count,
        "observed_evaluate_count": observed_evaluate_count,
        "expected_report_count": expected_run_worker_count,
        "observed_report_count": observed_report_count,
        "full_metric_row_count": len(full_rows),
        "rolling_metric_row_count": len(rolling_rows),
        "local_handoff_metric_row_count": len(local_rows),
        "action_fidelity_row_count": len(fidelity_rows),
        "policy_summary_row_count": len(policy_rows),
        "semantic_control_row_count": len(semantic_rows),
        "thresholds": {
            "median_full_rel_00_02": MEDIAN_FULL_REL_MIN,
            "min_seq_full_rel_00_02": MIN_SEQ_FULL_REL_MIN,
            "rolling_p90_median_rel_00_02": ROLLING_P90_MEDIAN_REL_MIN,
            "local_window_median_harm": LOCAL_WINDOW_MAX_HARM,
            "f19_close_rel_tolerance": F19_CLOSE_REL_TOL,
        },
        "stage4_selected_policy_ids": [row["policy_id"] for row in selected],
        "stage4_selected_action_names": [row["policy_id"] for row in selected],
        "best_policy_by_median_full_rel": selected[0]["policy_id"] if selected else (
            max(policy_rows, key=lambda row: safe_float(row.get("median_full_rel_00_02", "nan")))["policy_id"]
            if policy_rows else ""
        ),
        "outputs": {
            "full_metric_rows": rel(OUT / "full_metric_rows.csv"),
            "rolling_metric_rows": rel(OUT / "rolling_metric_rows.csv"),
            "local_handoff_metric_rows": rel(OUT / "local_handoff_metric_rows.csv"),
            "action_fidelity_rows": rel(OUT / "action_fidelity_rows.csv"),
            "semantic_control_rows": rel(OUT / "semantic_control_rows.csv"),
            "policy_summary_rows": rel(OUT / "policy_summary_rows.csv"),
            "stage4_candidate_selection": rel(OUT / "stage4_candidate_selection.csv"),
            "stage3_report": rel(OUT / "stage3_pilot_report.md"),
            "partial_signal_report": rel(OUT / "partial_signal_report.md"),
        },
    }

    write_csv(OUT / "full_metric_rows.csv", full_rows)
    write_csv(OUT / "rolling_metric_rows.csv", rolling_rows)
    write_csv(OUT / "local_handoff_metric_rows.csv", local_rows)
    write_csv(OUT / "action_fidelity_rows.csv", fidelity_rows)
    write_csv(OUT / "semantic_control_rows.csv", semantic_rows)
    write_csv(OUT / "policy_summary_rows.csv", policy_rows)
    write_csv(OUT / "stage4_candidate_selection.csv", selected)
    write_json(OUT / "stage3_summary.json", summary)
    write_text(OUT / "stage3_pilot_report.md", build_report(summary, policy_rows, semantic_rows))
    write_text(OUT / "partial_signal_report.md", build_partial_signal(policy_rows))
    print(json.dumps(base.clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
