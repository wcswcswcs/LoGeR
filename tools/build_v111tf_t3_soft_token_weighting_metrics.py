#!/usr/bin/env python3
"""Summarize ACL2 v111TF T3 semantic-aware soft token-weighting metrics."""

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
T3 = RESULT_ROOT / "batch_t_t3_soft_token_weighting"
CONFIG_ROWS = T3 / "action_config_rows.csv"
RUN_RESULTS = T3 / "run_results.csv"
WORKSPACE = T3 / "workspace"
SOFT_WEIGHT_FRAME_ROWS = T3 / "soft_weight_frame_rows.csv"
SEQUENCES = ("00", "01", "02", "05")

B1_MEDIAN_FULL_REL = 0.17413068803456322
B1_MEAN_FULL_REL = 0.18754824888948118
B1_90PCT_MEDIAN_GATE = B1_MEDIAN_FULL_REL * 0.9
MAX_HARM_GATE = 0.01
T3_MEDIAN_GATE = 0.10
STRONG_MARGIN = 0.03
MASK_TOL = 1e-5


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


def parse_mask(raw: Any) -> list[float]:
    values: list[float] = []
    for part in str(raw).split(","):
        text = part.strip()
        if not text:
            continue
        try:
            value = float(text)
        except ValueError:
            return []
        if not math.isfinite(value):
            return []
        values.append(value)
    return values


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


def t3_phase_status_for(cfg: dict[str, str], latest: dict[tuple[str, str], dict[str, str]]) -> tuple[dict[str, Any], bool, bool]:
    status: dict[str, Any] = {}
    all_metric_phase_success = True
    all_phase_success = True
    seq = cfg["seq"]
    for phase in ("prepare", "run_worker", "evaluate", "report"):
        if phase == "prepare":
            run_name = f"kitti_lingbot_v111tf_t3_prepare_{seq}"
        else:
            run_name = f"kitti_lingbot_v111tf_t3_{cfg['policy_id']}_{seq}_{phase}"
        row = latest.get((run_name, phase))
        rc = stage2m.safe_rc(row)
        status[f"{phase}_returncode"] = rc
        status[f"{phase}_duration_sec"] = row.get("duration_sec", "") if row else ""
        if phase in {"prepare", "run_worker", "evaluate"}:
            all_metric_phase_success = all_metric_phase_success and rc == 0
        all_phase_success = all_phase_success and rc == 0
    return status, all_metric_phase_success, all_phase_success


def install_t3_metric_overrides() -> None:
    stage2m.OUT = T3
    stage2m.CONFIG_ROWS = CONFIG_ROWS
    stage2m.RUN_RESULTS = RUN_RESULTS
    stage2m.WORKSPACE = WORKSPACE
    stage2m.SEQUENCES = SEQUENCES
    stage2m.phase_status_for = t3_phase_status_for

    original_action_fidelity = stage2m.action_fidelity_row

    def t3_action_fidelity_row(cfg: dict[str, str], latest: dict[tuple[str, str], dict[str, str]]) -> dict[str, Any]:
        row = original_action_fidelity(cfg, latest)
        run_name = f"kitti_lingbot_v111tf_t3_{cfg['policy_id']}_{cfg['seq']}_run_worker"
        run_row = latest.get((run_name, "run_worker"), {})
        row["schema"] = "acl2_v111tf_t3_action_fidelity_row_v1"
        row["candidate_id"] = "T3"
        row["run_worker_returncode"] = run_row.get("returncode", "")
        row["run_worker_duration_sec"] = run_row.get("duration_sec", "")
        row["risk_mode"] = cfg.get("risk_mode", "")
        row["gamma_cam"] = cfg.get("gamma_cam", "")
        row["gamma_reg"] = cfg.get("gamma_reg", "")
        row["nu"] = cfg.get("nu", "")
        row["camera_token_weight_mean"] = cfg.get("camera_token_weight_mean", "")
        row["register_token_weight_mean"] = cfg.get("register_token_weight_mean", "")
        row["anchor_token_weight_mean"] = cfg.get("anchor_token_weight_mean", "")
        return row

    stage2m.action_fidelity_row = t3_action_fidelity_row


def rel_by_seq(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        str(row.get("seq", "")): safe_float(row.get("full_ATE_sim3_relative_improvement_vs_baseline", "nan"))
        for row in rows
    }


def mask_audit_rows(config_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    expected_by_key = {
        (row["policy_id"], row["seq"], str(int(float(row["frame"])))): row
        for row in read_csv(SOFT_WEIGHT_FRAME_ROWS)
    }
    out: list[dict[str, Any]] = []
    for cfg in config_rows:
        expected_frames = stage3m.base.parse_indices(cfg.get("selected_global_frame_indices", ""))
        action_rows = stage3m.base.load_jsonl(Path(cfg["action_file"]))
        action_by_frame: dict[int, dict[str, Any]] = {}
        for action in action_rows:
            try:
                frame = int(float(action.get("sample_position", -1)))
            except (TypeError, ValueError):
                continue
            action_by_frame[frame] = action
        mismatch_frames: list[str] = []
        missing_frames: list[str] = []
        mode_mismatch: list[str] = []
        max_mask_abs_diff = 0.0
        observed_mask_count = 0
        for frame in expected_frames:
            action = action_by_frame.get(frame)
            if not action:
                missing_frames.append(str(frame))
                continue
            expected = expected_by_key.get((cfg["policy_id"], cfg["seq"], str(frame)), {})
            expected_mask = parse_mask(expected.get("token_type_mask", ""))
            observed_mask = parse_mask(action.get("token_type_mask", ""))
            if observed_mask:
                observed_mask_count += 1
            if (
                not bool_value(action.get("forced_context_only"))
                or not bool_value(action.get("context_only_append"))
                or str(action.get("context_only_special_mode", "")) != "token_mask"
            ):
                mode_mismatch.append(str(frame))
            if len(expected_mask) != 6 or len(observed_mask) != 6:
                mismatch_frames.append(str(frame))
                continue
            diffs = [abs(a - b) for a, b in zip(expected_mask, observed_mask)]
            local_max = max(diffs)
            max_mask_abs_diff = max(max_mask_abs_diff, local_max)
            if local_max > MASK_TOL:
                mismatch_frames.append(str(frame))
        pass_flag = not missing_frames and not mismatch_frames and not mode_mismatch
        out.append(
            {
                "schema": "acl2_v111tf_t3_mask_audit_row_v1",
                "policy_id": cfg["policy_id"],
                "policy_family": cfg["policy_family"],
                "seq": cfg["seq"],
                "risk_mode": cfg.get("risk_mode", ""),
                "expected_frame_count": len(expected_frames),
                "observed_mask_count": observed_mask_count,
                "max_mask_abs_diff": max_mask_abs_diff,
                "mask_tolerance": MASK_TOL,
                "missing_action_frames": ";".join(missing_frames),
                "mask_mismatch_frames": ";".join(mismatch_frames),
                "mode_mismatch_frames": ";".join(mode_mismatch),
                "mask_audit_pass": pass_flag,
            }
        )
    return out


def policy_summary_rows(
    full_rows: list[dict[str, Any]],
    rolling_rows: list[dict[str, Any]],
    fidelity_rows: list[dict[str, Any]],
    mask_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rolling_by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fidelity_by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mask_by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in full_rows:
        by_policy[str(row["policy_id"])].append(row)
    for row in rolling_rows:
        rolling_by_policy[str(row["policy_id"])].append(row)
    for row in fidelity_rows:
        fidelity_by_policy[str(row["policy_id"])].append(row)
    for row in mask_rows:
        mask_by_policy[str(row["policy_id"])].append(row)

    out: list[dict[str, Any]] = []
    for policy_id in sorted(by_policy):
        rows = by_policy[policy_id]
        rels_by_seq = rel_by_seq(rows)
        rels = [rels_by_seq.get(seq, float("nan")) for seq in SEQUENCES]
        roll_rels = [
            safe_float(row.get("rolling_ATE_p90_relative_improvement_vs_baseline", "nan"))
            for row in rolling_by_policy.get(policy_id, [])
        ]
        finals = [safe_float(row.get("final_error_relative_improvement_vs_baseline", "nan")) for row in rows]
        locals_ = [safe_float(row.get("local_window_ATE_rel_improvement_vs_baseline_median", "nan")) for row in rows]
        median_full = stage3m.base.median(rels)
        mean_full = stage3m.base.mean(rels)
        improved_count = sum(1 for value in rels if math.isfinite(value) and value > 0.0)
        max_harm = stage3m.base.max_rel_harm(rels)
        rolling_p90_median = stage3m.base.median(roll_rels)
        final_median = stage3m.base.median(finals)
        local_harm = stage3m.base.max_rel_harm(locals_)
        action_pass_count = sum(1 for row in fidelity_by_policy.get(policy_id, []) if bool_value(row.get("action_fidelity_pass")))
        mask_pass_count = sum(1 for row in mask_by_policy.get(policy_id, []) if bool_value(row.get("mask_audit_pass")))
        metric_complete = len(rows) == len(SEQUENCES) and all(bool_value(row.get("metric_available")) for row in rows)
        all_action = action_pass_count == len(SEQUENCES)
        all_mask = mask_pass_count == len(SEQUENCES)
        t3_plan_geometry_gate = bool(
            metric_complete
            and all_action
            and all_mask
            and improved_count >= 3
            and math.isfinite(median_full)
            and median_full >= T3_MEDIAN_GATE
            and max_harm <= MAX_HARM_GATE
        )
        b1_90_geometry_gate = bool(
            metric_complete
            and all_action
            and all_mask
            and math.isfinite(median_full)
            and median_full >= B1_90PCT_MEDIAN_GATE
            and max_harm <= MAX_HARM_GATE
            and math.isfinite(rolling_p90_median)
            and rolling_p90_median >= 0.0
        )
        b1_exceed_gate = bool(
            metric_complete
            and all_action
            and all_mask
            and max_harm <= MAX_HARM_GATE
            and math.isfinite(median_full)
            and median_full >= B1_MEDIAN_FULL_REL
        )
        strong_success_gate = bool(
            metric_complete
            and all_action
            and all_mask
            and max_harm <= MAX_HARM_GATE
            and (
                (math.isfinite(median_full) and median_full >= B1_MEDIAN_FULL_REL + STRONG_MARGIN)
                or (math.isfinite(mean_full) and mean_full >= B1_MEAN_FULL_REL + STRONG_MARGIN)
            )
        )
        sample = rows[0]
        out.append(
            {
                "schema": "acl2_v111tf_t3_policy_summary_row_v1",
                "candidate_id": "T3",
                "surface_id": sample.get("surface_id", ""),
                "policy_id": policy_id,
                "policy_family": sample.get("policy_family", ""),
                "risk_mode": sample.get("risk_mode", ""),
                "sequence_count": len(rows),
                "metric_complete": metric_complete,
                "action_fidelity_pass_count": action_pass_count,
                "mask_audit_pass_count": mask_pass_count,
                "all_action_fidelity": all_action,
                "all_mask_audit": all_mask,
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
                "camera_token_weight_mean": sample.get("camera_token_weight_mean", ""),
                "register_token_weight_mean": sample.get("register_token_weight_mean", ""),
                "anchor_token_weight_mean": sample.get("anchor_token_weight_mean", ""),
                "t3_median_gate": T3_MEDIAN_GATE,
                "b1_90pct_median_gate": B1_90PCT_MEDIAN_GATE,
                "t3_plan_geometry_gate_pass": t3_plan_geometry_gate,
                "t3_b1_90_geometry_gate_pass": b1_90_geometry_gate,
                "t3_b1_exceed_gate_pass": b1_exceed_gate,
                "t3_strong_success_gate_pass": strong_success_gate,
                "median_full_rel_minus_b1_median": (
                    median_full - B1_MEDIAN_FULL_REL if math.isfinite(median_full) else float("nan")
                ),
                "mean_full_rel_minus_b1_mean": (
                    mean_full - B1_MEAN_FULL_REL if math.isfinite(mean_full) else float("nan")
                ),
                "claim_boundary": "T3 soft weighting geometry/mechanism summary only; semantic causality controls are not satisfied here.",
            }
        )
    return out


def stable_geometry_ladders(policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["policy_id"]: row for row in policy_rows}
    pairs = [
        ("raw_mild_medium", "T3_soft_mild_raw", "T3_soft_medium_raw"),
        ("raw_medium_strong", "T3_soft_medium_raw", "T3_soft_strong_raw"),
        ("znorm_mild_medium", "T3_soft_mild_znorm", "T3_soft_medium_znorm"),
        ("znorm_medium_strong", "T3_soft_medium_znorm", "T3_soft_strong_znorm"),
    ]
    rows: list[dict[str, Any]] = []
    for pair_id, left, right in pairs:
        lrow = by_id.get(left, {})
        rrow = by_id.get(right, {})
        rows.append(
            {
                "schema": "acl2_v111tf_t3_stability_pair_row_v1",
                "pair_id": pair_id,
                "left_policy_id": left,
                "right_policy_id": right,
                "left_t3_plan_geometry_gate_pass": bool_value(lrow.get("t3_plan_geometry_gate_pass")),
                "right_t3_plan_geometry_gate_pass": bool_value(rrow.get("t3_plan_geometry_gate_pass")),
                "adjacent_pair_pass": bool_value(lrow.get("t3_plan_geometry_gate_pass"))
                and bool_value(rrow.get("t3_plan_geometry_gate_pass")),
                "left_median_full_rel": lrow.get("median_full_rel", ""),
                "right_median_full_rel": rrow.get("median_full_rel", ""),
            }
        )
    return rows


def parity_trajectory_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seq in SEQUENCES:
        t3_root = WORKSPACE / f"kitti_v111tf_t3_fullseq_{seq}" / seq / f"lingbot_map_v111tf_t3_T3_map_all1_context_parity_{seq}"
        t2_root = T2 / "workspace" / f"kitti_v111tf_t2_fullseq_{seq}" / seq / f"lingbot_map_v111tf_t2_T2_default_context_tokens_{seq}"
        traj_diff = max_abs_diff(t3_root / "traj.txt", t2_root / "traj.txt")
        intr_diff = max_abs_diff(t3_root / "intrinsics.txt", t2_root / "intrinsics.txt")
        rows.append(
            {
                "schema": "acl2_v111tf_t3_t2_parity_traj_row_v1",
                "seq": seq,
                "left_policy_id": "T3_map_all1_context_parity",
                "right_policy_id": "T2_default_context_tokens",
                "traj_max_abs_diff": traj_diff,
                "intrinsics_max_abs_diff": intr_diff,
                "parity_pass": math.isfinite(traj_diff)
                and math.isfinite(intr_diff)
                and traj_diff == 0.0
                and intr_diff == 0.0,
                "left_traj": rel(t3_root / "traj.txt"),
                "right_traj": rel(t2_root / "traj.txt"),
            }
        )
    return rows


def build_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    ranked = sorted(rows, key=lambda row: safe_float(row.get("median_full_rel", "nan")), reverse=True)
    lines = [
        "# ACL2 v111TF T3 Soft Token-Weighting Report",
        "",
        f"metric_complete: {summary['metric_complete']}",
        f"all_action_fidelity: {summary['all_action_fidelity']}",
        f"all_mask_audit: {summary['all_mask_audit']}",
        f"t3_t2_all1_parity_pass: {summary['t3_t2_all1_parity_pass']}",
        f"taxonomy: {summary['taxonomy']}",
        f"blocker: {summary['blocker']}",
        "",
        "## Policy Ranking",
        "",
    ]
    for row in ranked:
        lines.append(
            "- {policy_id}: median={median} mean={mean} improved={improved}/4 max_harm={harm} "
            "rolling={rolling} final={final} local_harm={local_harm} plan_gate={plan} b1_exceed={b1}".format(
                policy_id=row.get("policy_id", ""),
                median=row.get("median_full_rel", ""),
                mean=row.get("mean_full_rel", ""),
                improved=row.get("improved_seq_count", ""),
                harm=row.get("max_harm", ""),
                rolling=row.get("rolling_p90_median_rel", ""),
                final=row.get("final_error_median_rel", ""),
                local_harm=row.get("local_window_median_harm", ""),
                plan=row.get("t3_plan_geometry_gate_pass", ""),
                b1=row.get("t3_b1_exceed_gate_pass", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "T3 evaluates semantic-risk/support-derived soft weights on compact context tokens. It does not by itself satisfy semantic-causality controls.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    install_t3_metric_overrides()
    config_rows = read_csv(CONFIG_ROWS)
    latest = stage2m.latest_run_results(read_csv(RUN_RESULTS))
    full_rows, rolling_rows, local_rows, fidelity_rows = stage2m.metric_rows(config_rows, latest)
    mask_rows = mask_audit_rows(config_rows)
    for rows in (full_rows, rolling_rows, local_rows, fidelity_rows):
        for row in rows:
            row["schema"] = str(row.get("schema", "")).replace("acl2_v109tf_stage2", "acl2_v111tf_t3")
            cfg = next((cfg for cfg in config_rows if cfg["policy_id"] == row.get("policy_id") and cfg["seq"] == row.get("seq")), {})
            for key in (
                "risk_mode",
                "gamma_cam",
                "gamma_reg",
                "nu",
                "camera_token_weight_mean",
                "register_token_weight_mean",
                "anchor_token_weight_mean",
            ):
                row[key] = cfg.get(key, "")

    policy_rows = policy_summary_rows(full_rows, rolling_rows, fidelity_rows, mask_rows)
    stability_rows = stable_geometry_ladders(policy_rows)
    parity_rows = parity_trajectory_rows()
    observed_counts: dict[str, int] = {}
    for row in latest.values():
        if stage2m.safe_rc(row) == 0:
            phase = str(row.get("phase", ""))
            observed_counts[phase] = observed_counts.get(phase, 0) + 1

    metric_complete = len(full_rows) == len(config_rows) and all(bool_value(row.get("metric_available")) for row in full_rows)
    all_action = len(fidelity_rows) == len(config_rows) and all(bool_value(row.get("action_fidelity_pass")) for row in fidelity_rows)
    all_mask = len(mask_rows) == len(config_rows) and all(bool_value(row.get("mask_audit_pass")) for row in mask_rows)
    plan_pass = [row for row in policy_rows if bool_value(row.get("t3_plan_geometry_gate_pass"))]
    b1_90_pass = [row for row in policy_rows if bool_value(row.get("t3_b1_90_geometry_gate_pass"))]
    b1_exceed = [row for row in policy_rows if bool_value(row.get("t3_b1_exceed_gate_pass"))]
    strong_pass = [row for row in policy_rows if bool_value(row.get("t3_strong_success_gate_pass"))]
    stable_pass = [row for row in stability_rows if bool_value(row.get("adjacent_pair_pass"))]
    parity_pass = len(parity_rows) == len(SEQUENCES) and all(bool_value(row.get("parity_pass")) for row in parity_rows)
    best = max(policy_rows, key=lambda row: safe_float(row.get("median_full_rel", "nan"))) if policy_rows else {}
    if strong_pass:
        taxonomy = "T3_STRONG_GEOMETRY_PASS_SEMANTIC_CAUSALITY_PENDING"
        blocker = "semantic_causality_controls_not_run_for_soft_token_policy"
    elif b1_exceed:
        taxonomy = "T3_B1_EXCEED_GEOMETRY_PASS_SEMANTIC_CAUSALITY_PENDING"
        blocker = "semantic_causality_controls_not_run_for_soft_token_policy"
    elif plan_pass and stable_pass:
        taxonomy = "T3_PLAN_GEOMETRY_PASS_STABLE_LADDER_SEMANTIC_CAUSALITY_PENDING"
        blocker = "semantic_causality_controls_not_run_for_soft_token_policy"
    elif plan_pass:
        taxonomy = "T3_SINGLE_POLICY_GEOMETRY_PASS_UNSTABLE_LADDER"
        blocker = "only_single_or_non_adjacent_soft_policy_passed_t3_geometry_gate"
    else:
        taxonomy = "T3_GEOMETRY_FAIL_OR_BELOW_PLAN_GATE"
        blocker = "no_soft_token_policy_satisfied_t3_geometry_gate"

    summary = {
        "schema": "acl2_v111tf_t3_soft_token_metric_summary_v1",
        "metric_complete": metric_complete,
        "all_action_fidelity": all_action,
        "all_mask_audit": all_mask,
        "t3_t2_all1_parity_pass": parity_pass,
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
        "mask_audit_row_count": len(mask_rows),
        "policy_summary_row_count": len(policy_rows),
        "t3_plan_geometry_pass_policy_ids": [row["policy_id"] for row in plan_pass],
        "t3_b1_90_geometry_pass_policy_ids": [row["policy_id"] for row in b1_90_pass],
        "t3_b1_exceed_policy_ids": [row["policy_id"] for row in b1_exceed],
        "t3_strong_success_policy_ids": [row["policy_id"] for row in strong_pass],
        "stable_adjacent_pass_pair_ids": [row["pair_id"] for row in stable_pass],
        "best_policy_by_median_full_rel": best.get("policy_id", ""),
        "best_policy_median_full_rel": best.get("median_full_rel", ""),
        "best_policy_mean_full_rel": best.get("mean_full_rel", ""),
        "b1_reference": {
            "median_full_rel": B1_MEDIAN_FULL_REL,
            "mean_full_rel": B1_MEAN_FULL_REL,
            "b1_90pct_median_gate": B1_90PCT_MEDIAN_GATE,
            "strong_margin": STRONG_MARGIN,
        },
        "t3_plan_reference": {
            "median_full_rel_gate": T3_MEDIAN_GATE,
            "improved_seq_count_gate": 3,
            "max_harm_gate": MAX_HARM_GATE,
            "stable_ladder_rule": "adjacent mild/medium/strong pair must pass for stable-method claim",
        },
        "semantic_causality_claim_allowed": False,
        "semantic_causality_claim_blocker": "T3 soft weighting grid has no semantic shuffle/random/same-bucket/schedule controls.",
        "outputs": {
            "full_metric_rows": rel(T3 / "full_metric_rows.csv"),
            "rolling_metric_rows": rel(T3 / "rolling_metric_rows.csv"),
            "local_handoff_metric_rows": rel(T3 / "local_handoff_metric_rows.csv"),
            "action_fidelity_rows": rel(T3 / "action_fidelity_rows.csv"),
            "mask_audit_rows": rel(T3 / "mask_audit_rows.csv"),
            "policy_summary_rows": rel(T3 / "policy_summary_rows.csv"),
            "stability_pair_rows": rel(T3 / "t3_stability_pair_rows.csv"),
            "parity_trajectory_rows": rel(T3 / "t3_t2_all1_parity_trajectory_rows.csv"),
            "report": rel(T3 / "T3_SOFT_TOKEN_WEIGHTING_REPORT.md"),
            "summary": rel(T3 / "t3_metric_summary.json"),
        },
    }

    write_csv(T3 / "full_metric_rows.csv", full_rows)
    write_csv(T3 / "rolling_metric_rows.csv", rolling_rows)
    write_csv(T3 / "local_handoff_metric_rows.csv", local_rows)
    write_csv(T3 / "action_fidelity_rows.csv", fidelity_rows)
    write_csv(T3 / "mask_audit_rows.csv", mask_rows)
    write_csv(T3 / "policy_summary_rows.csv", policy_rows)
    write_csv(T3 / "t3_stability_pair_rows.csv", stability_rows)
    write_csv(T3 / "t3_t2_all1_parity_trajectory_rows.csv", parity_rows)
    write_json(T3 / "t3_metric_summary.json", summary)
    write_text(T3 / "T3_SOFT_TOKEN_WEIGHTING_REPORT.md", build_report(summary, policy_rows))
    print(json.dumps(stage3m.base.clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
