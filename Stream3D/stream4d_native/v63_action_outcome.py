from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .v47_common import ROOT, parse_bool, parse_float, parse_int, read_csv, utc_now, write_csv, write_json


DEFAULT_D4RT_ROWS = "outputs/audit/v63_d4rt_query/query_result_rows.csv"


@dataclass(frozen=True)
class V63ActionOutcomeConfig:
    d4rt_query_result_rows: str | Path = DEFAULT_D4RT_ROWS
    output_root: str | Path = "outputs/audit/v63_action_outcome"
    visualization_root: str | Path = "outputs/audit/v63_visualizations/action_outcome"
    min_accepted_frames: int = 2
    min_in_bounds_ratio: float = 0.80
    min_source_score: float = 0.25


def build_v63_action_outcome(config: V63ActionOutcomeConfig | None = None) -> dict[str, Any]:
    cfg = config or V63ActionOutcomeConfig()
    real_rows = read_csv(_project(cfg.d4rt_query_result_rows))
    base_rows = [_score_row(row, cfg, control_variant="real_d4rt") for row in real_rows]
    r0_rows = [row for row in base_rows if row["policy_id"] == "R0_real_policy"]
    shuffled_rows = _shuffled_association_rows(r0_rows, cfg)
    no_temporal_rows = []
    for row in real_rows:
        if row.get("policy_id") != "R0_real_policy":
            continue
        scored = _score_row(row, cfg, control_variant="no_temporal_source_frame_only")
        scored["policy_id"] = "R2_no_temporal_source_frame_only"
        scored["control_id"] = "R2_no_temporal_source_frame_only"
        no_temporal_rows.append(scored)
    all_rows = [*base_rows, *shuffled_rows, *no_temporal_rows]
    utility_rows = _utility_rows(all_rows)
    summary = _summary(all_rows, utility_rows, cfg)
    return {
        "summary": summary,
        "action_outcome_rows": all_rows,
        "action_utility_rows": utility_rows,
    }


def write_v63_action_outcome(result: dict[str, Any], config: V63ActionOutcomeConfig | None = None) -> dict[str, str]:
    cfg = config or V63ActionOutcomeConfig()
    output_root = _project(cfg.output_root)
    visual_root = _project(cfg.visualization_root)
    output_root.mkdir(parents=True, exist_ok=True)
    visual_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "action_outcome_summary.json"
    rows_path = output_root / "action_outcome_rows.csv"
    utility_path = output_root / "action_utility_rows.csv"
    write_json(summary_path, result["summary"])
    write_csv(rows_path, result["action_outcome_rows"])
    write_csv(utility_path, result["action_utility_rows"])
    visuals = _write_visualizations(result, visual_root)
    return {
        "action_outcome_summary": _rel(summary_path),
        "action_outcome_rows": _rel(rows_path),
        "action_utility_rows": _rel(utility_path),
        **visuals,
    }


def _score_row(row: dict[str, str], cfg: V63ActionOutcomeConfig, *, control_variant: str) -> dict[str, Any]:
    policy_id = row.get("policy_id", "")
    planned_action = _normalized_planned_action(row)
    candidate_type = row.get("candidate_type", "")
    accepted_count = parse_int(row.get("accepted_track_count"))
    valid_count = parse_int(row.get("valid_track_count"))
    accepted_ratio = parse_float(row.get("accepted_track_ratio"))
    in_bounds_count = parse_int(row.get("in_bounds_track_count"))
    in_bounds_ratio = in_bounds_count / float(max(valid_count, 1))
    outside_residual_rate = 1.0 - in_bounds_ratio
    source = _source_frame_evidence(row)
    if control_variant == "no_temporal_source_frame_only":
        evidence_score = source["source_frame_score"]
        source_frame_has_evidence = bool(source["source_frame_in_bounds"] and source["source_frame_score"] >= float(cfg.min_source_score))
        # Source-frame visibility proves the clicked source point exists, but it is not a temporal material query.
        # It therefore cannot directly confirm/quarantine ownership without target-frame evidence.
        valid_material_evidence = False
        stable_track = False
        accepted_count_eval = int(source_frame_has_evidence)
        accepted_ratio_eval = 0.0
        in_bounds_ratio_eval = 1.0 if source["source_frame_in_bounds"] else 0.0
        outside_residual_eval = 1.0 - in_bounds_ratio_eval
    else:
        evidence_score = 0.55 * accepted_ratio + 0.35 * in_bounds_ratio + 0.10 * source["source_frame_score"]
        stable_track = accepted_count >= int(cfg.min_accepted_frames)
        valid_material_evidence = bool(stable_track and in_bounds_ratio >= float(cfg.min_in_bounds_ratio))
        accepted_count_eval = accepted_count
        accepted_ratio_eval = accepted_ratio
        in_bounds_ratio_eval = in_bounds_ratio
        outside_residual_eval = outside_residual_rate
    outcome = _action_outcome(
        planned_action=planned_action,
        candidate_type=candidate_type,
        valid_material_evidence=valid_material_evidence,
        outside_residual_rate=outside_residual_eval,
        has_k_mat=parse_bool(row.get("has_K_mat")),
    )
    flags = _action_flags(planned_action, candidate_type, outcome, valid_material_evidence)
    utility = _utility_score(planned_action, outcome, flags, valid_material_evidence)
    return {
        **row,
        "planned_action": planned_action,
        "control_variant": control_variant,
        "action_outcome": outcome,
        "valid_material_evidence": valid_material_evidence,
        "stable_track": stable_track,
        "evidence_score": evidence_score,
        "accepted_track_count_eval": accepted_count_eval,
        "accepted_track_ratio_eval": accepted_ratio_eval,
        "in_bounds_valid_ratio": in_bounds_ratio_eval,
        "outside_residual_rate": outside_residual_eval,
        "source_frame_score": source["source_frame_score"],
        "source_frame_visibility": source["source_frame_visibility"],
        "source_frame_confidence": source["source_frame_confidence"],
        "source_frame_in_bounds": source["source_frame_in_bounds"],
        "action_success": flags["action_success"],
        "safe_defer": flags["safe_defer"],
        "false_confirm": flags["false_confirm"],
        "over_quarantine": flags["over_quarantine"],
        "noise_failure": flags["noise_failure"],
        "action_utility": utility,
        "utility_rule": "success_reward_minus_false_confirm_over_quarantine_noise_and_cost",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": False,
    }


def _normalized_planned_action(row: dict[str, str]) -> str:
    policy_id = row.get("policy_id", "")
    if policy_id in {"C0_v62_original", "C1_random_matched"}:
        return "control_noop"
    if policy_id == "C3_semantic_only":
        return "control_semantic_only"
    return row.get("planned_action", "")


def _action_outcome(
    *,
    planned_action: str,
    candidate_type: str,
    valid_material_evidence: bool,
    outside_residual_rate: float,
    has_k_mat: bool,
) -> str:
    if not valid_material_evidence:
        return "defer"
    if planned_action == "confirm" and candidate_type == "heldout_recovery" and has_k_mat and outside_residual_rate < 0.20:
        return "confirm"
    if planned_action in {"quarantine", "control_mask_boundary", "control_mask_only"}:
        return "quarantine"
    if planned_action == "reject_decoy":
        return "quarantine"
    if outside_residual_rate >= 0.20:
        return "quarantine"
    if planned_action == "defer" or candidate_type in {"unknown_defer", "control"}:
        return "defer"
    return "defer"


def _action_flags(planned_action: str, candidate_type: str, outcome: str, valid_material_evidence: bool) -> dict[str, bool]:
    success = False
    if planned_action == "confirm":
        success = outcome == "confirm"
    elif planned_action in {"quarantine", "control_mask_boundary", "control_mask_only"}:
        success = outcome == "quarantine"
    elif planned_action == "reject_decoy":
        success = outcome != "confirm"
    elif planned_action == "defer":
        success = outcome == "defer"
    false_confirm = bool(outcome == "confirm" and (planned_action != "confirm" or candidate_type in {"decoy_rejection", "unknown_defer", "shortcut_quarantine", "control"}))
    over_quarantine = bool(outcome == "quarantine" and planned_action == "confirm")
    noise_failure = bool(not valid_material_evidence and planned_action in {"confirm", "quarantine", "reject_decoy", "control_mask_boundary", "control_mask_only"})
    safe_defer = bool(planned_action == "defer" and outcome == "defer")
    return {
        "action_success": success,
        "false_confirm": false_confirm,
        "over_quarantine": over_quarantine,
        "noise_failure": noise_failure,
        "safe_defer": safe_defer,
    }


def _utility_score(planned_action: str, outcome: str, flags: dict[str, bool], valid_material_evidence: bool) -> float:
    score = 0.0
    if flags["action_success"]:
        score += 1.0
    if outcome in {"confirm", "quarantine"} and valid_material_evidence:
        score += 0.25
    if flags["safe_defer"]:
        score += 0.35
    if flags["false_confirm"]:
        score -= 2.0
    if flags["over_quarantine"]:
        score -= 0.8
    if flags["noise_failure"]:
        score -= 0.45
    if planned_action == "reject_decoy" and outcome != "confirm":
        score += 0.25
    score -= 0.05
    return float(score)


def _shuffled_association_rows(r0_rows: list[dict[str, Any]], cfg: V63ActionOutcomeConfig) -> list[dict[str, Any]]:
    if not r0_rows:
        return []
    shift = max(1, len(r0_rows) // 3)
    shuffled: list[dict[str, Any]] = []
    for idx, row in enumerate(r0_rows):
        donor = r0_rows[(idx + shift) % len(r0_rows)]
        new_row = dict(row)
        for key in [
            "planned_action",
            "candidate_type",
            "query_history_id",
            "has_K_mat",
            "has_K_mask",
            "has_K_sem",
            "decoy_source_history_id",
            "decoy_history_id",
        ]:
            new_row[key] = donor.get(key, "")
        new_row["policy_id"] = "R1_shuffled_history_association"
        new_row["control_id"] = "R1_shuffled_history_association"
        new_row["shuffle_donor_v63_candidate_id"] = donor.get("v63_candidate_id", "")
        shuffled.append(_score_row(new_row, cfg, control_variant="shuffled_history_association"))
    return shuffled


def _source_frame_evidence(row: dict[str, str]) -> dict[str, Any]:
    carrier_path = _project(row.get("carrier_batch_npz", ""))
    query_index = parse_int(row.get("d4rt_query_index"))
    support_frame = parse_int(row.get("support_frame_id"))
    if not carrier_path.exists():
        return {
            "source_frame_score": 0.0,
            "source_frame_visibility": 0.0,
            "source_frame_confidence": 0.0,
            "source_frame_in_bounds": False,
        }
    carrier = np.load(carrier_path)
    frame_ids = [int(value) for value in np.asarray(carrier["frame_ids"]).tolist()]
    try:
        local_idx = frame_ids.index(int(support_frame))
    except ValueError:
        local_idx = 0
    visibility = float(np.asarray(carrier["visibility_prob"])[local_idx, query_index])
    confidence = float(np.asarray(carrier["confidence_prob"])[local_idx, query_index])
    uv = np.asarray(carrier["uv_pred"])[local_idx, query_index]
    valid = bool(np.asarray(carrier["valid"])[local_idx, query_index])
    in_bounds = bool(valid and 0.0 <= float(uv[0]) <= 1.0 and 0.0 <= float(uv[1]) <= 1.0)
    return {
        "source_frame_score": visibility * confidence if in_bounds else 0.0,
        "source_frame_visibility": visibility,
        "source_frame_confidence": confidence,
        "source_frame_in_bounds": in_bounds,
    }


def _utility_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row.get("policy_id", ""), row.get("control_variant", "")), []).append(row)
    out: list[dict[str, Any]] = []
    for (policy_id, variant), group_rows in sorted(groups.items()):
        utilities = [parse_float(row.get("action_utility")) for row in group_rows]
        valid = [parse_bool(row.get("valid_material_evidence")) for row in group_rows]
        confirm_or_quarantine = [row.get("action_outcome") in {"confirm", "quarantine"} for row in group_rows]
        out.append(
            {
                "policy_id": policy_id,
                "control_variant": variant,
                "query_count": len(group_rows),
                "valid_material_evidence_rate": _mean_bool(valid),
                "confirm_or_quarantine_rate": _mean_bool(confirm_or_quarantine),
                "action_success_rate": _mean_bool(parse_bool(row.get("action_success")) for row in group_rows),
                "safe_defer_rate": _mean_bool(parse_bool(row.get("safe_defer")) for row in group_rows),
                "false_confirm_rate": _mean_bool(parse_bool(row.get("false_confirm")) for row in group_rows),
                "over_quarantine_rate": _mean_bool(parse_bool(row.get("over_quarantine")) for row in group_rows),
                "noise_failure_rate": _mean_bool(parse_bool(row.get("noise_failure")) for row in group_rows),
                "mean_action_utility": float(np.mean(utilities)) if utilities else None,
                "min_action_utility": float(np.min(utilities)) if utilities else None,
                "max_action_utility": float(np.max(utilities)) if utilities else None,
                "action_outcome_counts": dict(Counter(row.get("action_outcome", "") for row in group_rows)),
                "planned_action_counts": dict(Counter(row.get("planned_action", "") for row in group_rows)),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": False,
            }
        )
    return out


def _summary(rows: list[dict[str, Any]], utility_rows: list[dict[str, Any]], cfg: V63ActionOutcomeConfig) -> dict[str, Any]:
    utility_by_policy = {row["policy_id"]: row for row in utility_rows}
    r0 = utility_by_policy.get("R0_real_policy", {})
    shuffled = utility_by_policy.get("R1_shuffled_history_association", {})
    no_temporal = utility_by_policy.get("R2_no_temporal_source_frame_only", {})
    fixed_controls = [row for row in utility_rows if row["policy_id"].startswith("C") and row["control_variant"] == "real_d4rt"]
    best_fixed_utility = max((parse_float(row.get("mean_action_utility")) for row in fixed_controls), default=0.0)
    best_fixed_cq = max((parse_float(row.get("confirm_or_quarantine_rate")) for row in fixed_controls), default=0.0)
    real_utility = parse_float(r0.get("mean_action_utility"))
    shuffled_utility = parse_float(shuffled.get("mean_action_utility"))
    no_temporal_utility = parse_float(no_temporal.get("mean_action_utility"))
    real_minus_shuffled = real_utility - shuffled_utility
    real_minus_no_temporal = real_utility - no_temporal_utility
    real_minus_best_fixed = real_utility - best_fixed_utility
    real_cq = parse_float(r0.get("confirm_or_quarantine_rate"))
    gate = {
        "query_valid_material_evidence_rate_ge_0_50": parse_float(r0.get("valid_material_evidence_rate")) >= 0.50,
        "query_to_confirm_or_quarantine_rate_ge_best_fixed_plus_0_15": real_cq >= best_fixed_cq + 0.15,
        "false_confirm_rate_le_0_05": parse_float(r0.get("false_confirm_rate")) <= 0.05,
        "real_minus_shuffled_query_utility_ge_0_10": real_minus_shuffled >= 0.10,
        "real_minus_no_temporal_query_utility_ge_0_05": real_minus_no_temporal >= 0.05,
        "real_minus_best_fixed_utility_positive": real_minus_best_fixed > 0.0,
    }
    gate["pass"] = bool(all(gate.values()))
    return {
        "phase": "v63_action_outcome",
        "created_at": utc_now(),
        "method_status": "action_utility_evaluated_from_real_D4RT_and_association_controls",
        "input_d4rt_query_result_rows": _rel(cfg.d4rt_query_result_rows),
        "row_count": len(rows),
        "utility_row_count": len(utility_rows),
        "best_fixed_control_mean_action_utility": best_fixed_utility,
        "best_fixed_control_confirm_or_quarantine_rate": best_fixed_cq,
        "real_minus_best_fixed_utility": real_minus_best_fixed,
        "real_minus_shuffled_query_utility": real_minus_shuffled,
        "real_minus_no_temporal_query_utility": real_minus_no_temporal,
        "r0_metrics": r0,
        "shuffled_metrics": shuffled,
        "no_temporal_metrics": no_temporal,
        "utility_note": (
            "Phase 4 uses a transparent action-utility proxy from D4RT track stability, in-bounds support, planned action, "
            "and decoy/defer safety flags. It is not a GT AP metric and does not use GT for prediction."
        ),
        "gate": gate,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": False,
    }


def _mean_bool(values: Any) -> float | None:
    vals = [bool(value) for value in values]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _write_visualizations(result: dict[str, Any], root: Path) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    utility_rows = result["action_utility_rows"]
    utility_path = root / "mean_action_utility_by_policy.png"
    _bar_png(
        utility_path,
        "v63 Phase 4 mean action utility",
        {row["policy_id"] + ":" + row["control_variant"].replace("_", "-"): parse_float(row.get("mean_action_utility")) for row in utility_rows},
    )
    cq_path = root / "confirm_or_quarantine_rate_by_policy.png"
    _bar_png(
        cq_path,
        "v63 Phase 4 confirm/quarantine rate",
        {row["policy_id"] + ":" + row["control_variant"].replace("_", "-"): parse_float(row.get("confirm_or_quarantine_rate")) for row in utility_rows},
    )
    return {
        "mean_action_utility_by_policy": _rel(utility_path),
        "confirm_or_quarantine_rate_by_policy": _rel(cq_path),
    }


def _bar_png(path: Path, title: str, values: dict[str, float]) -> None:
    labels = list(values)
    nums = [float(values[label]) for label in labels]
    width = max(1200, 140 * max(1, len(labels)))
    height = 620
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(image, title, (36, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (32, 36, 44), 2, cv2.LINE_AA)
    min_v = min(0.0, min(nums) if nums else 0.0)
    max_v = max(1.0, max(nums) if nums else 1.0)
    chart_x, chart_y, chart_h = 80, 130, 340
    step = int((width - 180) / max(1, len(labels)))
    bar_w = min(90, max(36, int(step * 0.52)))
    zero_y = int(chart_y + chart_h - (0.0 - min_v) / max(max_v - min_v, 1e-9) * chart_h)
    cv2.line(image, (chart_x, zero_y), (width - 80, zero_y), (90, 96, 104), 1)
    for idx, (label, value) in enumerate(zip(labels, nums)):
        x0 = chart_x + idx * step + max(0, (step - bar_w) // 2)
        y = int(chart_y + chart_h - (value - min_v) / max(max_v - min_v, 1e-9) * chart_h)
        y0, y1 = sorted([zero_y, y])
        color = (72 + 31 * idx % 150, 132 + 19 * idx % 95, 210 - 17 * idx % 120)
        cv2.rectangle(image, (x0, y0), (x0 + bar_w, y1), color, -1)
        cv2.rectangle(image, (x0, y0), (x0 + bar_w, y1), (45, 49, 55), 1)
        cv2.putText(image, f"{value:.3f}", (x0 - 4, max(80, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (32, 36, 44), 1, cv2.LINE_AA)
        cv2.putText(image, label[:24], (x0 - 28, chart_y + chart_h + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 66, 74), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)
