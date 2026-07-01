from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .v47_common import ROOT, parse_float, read_csv, utc_now, write_csv, write_json


DEFAULT_V63_CANDIDATES = "outputs/audit/v63_query_candidates/query_candidate_rows.csv"


@dataclass(frozen=True)
class V63QueryPolicyConfig:
    query_candidate_rows: str | Path = DEFAULT_V63_CANDIDATES
    output_root: str | Path = "outputs/audit/v63_query_policy"
    visualization_root: str | Path = "outputs/audit/v63_visualizations/query_policy"
    query_budget: int = 64


def build_v63_query_policy(config: V63QueryPolicyConfig | None = None) -> dict[str, Any]:
    cfg = config or V63QueryPolicyConfig()
    rows = read_csv(_project(cfg.query_candidate_rows))
    method_rows = [row for row in rows if row.get("row_role") == "method_candidate"]
    control_rows = [row for row in rows if row.get("row_role") == "baseline_control"]
    selected_method = _select_method(method_rows, cfg.query_budget)
    selected_controls = _select_controls(control_rows, cfg.query_budget)
    selected_rows = [*_policy_rows(selected_method, "R0_real_policy"), *_control_policy_rows(selected_controls)]
    metric_rows = _metric_rows(selected_rows, cfg.query_budget)
    method_counts = Counter(row["candidate_type"] for row in selected_method)
    control_counts = Counter(row["control_id"] for rows_for_control in selected_controls.values() for row in rows_for_control)
    action_counts = Counter(row["planned_action"] for row in selected_rows if row["policy_id"] == "R0_real_policy")
    gate = {
        "real_policy_query_count_eq_budget": len(selected_method) == int(cfg.query_budget),
        "real_policy_type_counts_balanced": len(set(method_counts.values())) == 1 and len(method_counts) == 4,
        "all_controls_query_count_eq_budget": all(len(rows_for_control) == int(cfg.query_budget) for rows_for_control in selected_controls.values()),
        "all_required_controls_present": all(name in selected_controls for name in ["C0_v62_original", "C1_random_matched", "C2_mask_boundary", "C3_semantic_only", "C4_K_mask_only_ablation"]),
        "selection_uses_gt_for_prediction": False,
    }
    gate["pass"] = bool(
        gate["real_policy_query_count_eq_budget"]
        and gate["real_policy_type_counts_balanced"]
        and gate["all_controls_query_count_eq_budget"]
        and gate["all_required_controls_present"]
        and gate["selection_uses_gt_for_prediction"] is False
    )
    summary = {
        "phase": "v63_query_policy",
        "created_at": utc_now(),
        "method_status": "pre_D4RT_action_utility_policy_only",
        "query_budget": int(cfg.query_budget),
        "real_policy_query_count": len(selected_method),
        "real_policy_type_counts": dict(method_counts),
        "real_policy_action_counts": dict(action_counts),
        "control_query_counts": {key: len(value) for key, value in selected_controls.items()},
        "policy_note": (
            "Phase 2 selects equal-budget queries using candidate_type, K flags, support count, and Phase 1 selection_score. "
            "The utility is a pre-D4RT planning heuristic, not measured query outcome."
        ),
        "planned_actions": {
            "heldout_recovery": "confirm",
            "shortcut_quarantine": "quarantine",
            "decoy_rejection": "reject_decoy",
            "unknown_defer": "defer",
        },
        "input_paths": {
            "query_candidate_rows": _rel(cfg.query_candidate_rows),
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": False,
        "gate": gate,
    }
    return {
        "summary": summary,
        "selected_query_rows": selected_rows,
        "query_policy_metric_rows": metric_rows,
    }


def write_v63_query_policy(result: dict[str, Any], config: V63QueryPolicyConfig | None = None) -> dict[str, str]:
    cfg = config or V63QueryPolicyConfig()
    output_root = _project(cfg.output_root)
    visual_root = _project(cfg.visualization_root)
    output_root.mkdir(parents=True, exist_ok=True)
    visual_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "query_policy_summary.json"
    rows_path = output_root / "selected_query_rows.csv"
    metric_path = output_root / "query_policy_metric_rows.csv"
    write_json(summary_path, result["summary"])
    write_csv(rows_path, result["selected_query_rows"])
    write_csv(metric_path, result["query_policy_metric_rows"])
    visuals = _write_visualizations(result, visual_root)
    return {
        "query_policy_summary": _rel(summary_path),
        "selected_query_rows": _rel(rows_path),
        "query_policy_metric_rows": _rel(metric_path),
        **visuals,
    }


def _select_method(rows: list[dict[str, str]], budget: int) -> list[dict[str, str]]:
    by_type: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_type.setdefault(row.get("candidate_type", ""), []).append(row)
    required = ["heldout_recovery", "shortcut_quarantine", "decoy_rejection", "unknown_defer"]
    per_type = int(budget) // len(required)
    selected: list[dict[str, str]] = []
    for candidate_type in required:
        selected.extend(_sort_for_policy(by_type.get(candidate_type, []), candidate_type)[:per_type])
    remainder = int(budget) - len(selected)
    if remainder > 0:
        selected_ids = {row.get("v63_candidate_id", "") for row in selected}
        leftovers = [row for row in _sort_for_policy(rows, "remainder") if row.get("v63_candidate_id", "") not in selected_ids]
        selected.extend(leftovers[:remainder])
    return selected


def _select_controls(rows: list[dict[str, str]], budget: int) -> dict[str, list[dict[str, str]]]:
    by_control: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_control.setdefault(row.get("control_id", ""), []).append(row)
    out: dict[str, list[dict[str, str]]] = {}
    for control_id in ["C0_v62_original", "C1_random_matched", "C2_mask_boundary", "C3_semantic_only", "C4_K_mask_only_ablation"]:
        out[control_id] = _sort_for_policy(by_control.get(control_id, []), control_id)[: int(budget)]
    return out


def _policy_rows(rows: list[dict[str, str]], policy_id: str) -> list[dict[str, Any]]:
    return [_policy_row(row, policy_id, _planned_action(row), row.get("candidate_type", "")) for row in rows]


def _control_policy_rows(selected_controls: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for control_id, rows in selected_controls.items():
        for row in rows:
            out.append(_policy_row(row, control_id, _control_action(control_id, row), row.get("candidate_type", "")))
    return out


def _policy_row(row: dict[str, str], policy_id: str, planned_action: str, policy_stratum: str) -> dict[str, Any]:
    utility = _policy_utility(row, planned_action)
    return {
        **row,
        "policy_id": policy_id,
        "policy_stratum": policy_stratum,
        "planned_action": planned_action,
        "pre_d4rt_expected_utility": utility,
        "utility_source": "phase2_pre_D4RT_heuristic_not_measured_outcome",
        "policy_selection_uses_gt_for_prediction": False,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": False,
    }


def _planned_action(row: dict[str, str]) -> str:
    return {
        "heldout_recovery": "confirm",
        "shortcut_quarantine": "quarantine",
        "decoy_rejection": "reject_decoy",
        "unknown_defer": "defer",
    }.get(row.get("candidate_type", ""), "defer")


def _control_action(control_id: str, row: dict[str, str]) -> str:
    if control_id == "C4_K_mask_only_ablation":
        return "control_mask_only"
    if control_id == "C3_semantic_only":
        return "control_semantic_only"
    if control_id == "C2_mask_boundary":
        return "control_mask_boundary"
    if control_id in {"C0_v62_original", "C1_random_matched"}:
        return "control_noop"
    return _planned_action(row)


def _policy_utility(row: dict[str, str], planned_action: str) -> float:
    base = parse_float(row.get("selection_score"))
    action_bonus = {
        "confirm": 0.18,
        "quarantine": 0.16,
        "reject_decoy": 0.14,
        "defer": 0.12,
        "control_mask_boundary": 0.05,
        "control_semantic_only": 0.05,
        "control_mask_only": 0.05,
    }.get(planned_action, 0.05)
    support_term = min(parse_float(row.get("support_observation_count")) / 8.0, 1.0) * 0.05
    return float(base + action_bonus + support_term)


def _sort_for_policy(rows: list[dict[str, str]], salt: str) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: (_policy_utility(row, _planned_action(row)), _stable_unit(_stable_key(row, salt))), reverse=True)


def _metric_rows(rows: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    by_policy: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_policy.setdefault(row["policy_id"], []).append(row)
    metric_rows: list[dict[str, Any]] = []
    for policy_id, policy_rows in sorted(by_policy.items()):
        utilities = [parse_float(row.get("pre_d4rt_expected_utility")) for row in policy_rows]
        action_counts = Counter(row.get("planned_action", "") for row in policy_rows)
        metric_rows.append(
            {
                "policy_id": policy_id,
                "selected_query_count": len(policy_rows),
                "query_budget": int(budget),
                "budget_filled": len(policy_rows) == int(budget),
                "mean_pre_d4rt_expected_utility": float(np.mean(utilities)) if utilities else None,
                "min_pre_d4rt_expected_utility": float(np.min(utilities)) if utilities else None,
                "max_pre_d4rt_expected_utility": float(np.max(utilities)) if utilities else None,
                "planned_action_counts": dict(action_counts),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": False,
            }
        )
    return metric_rows


def _stable_key(row: dict[str, str], salt: str) -> str:
    return "|".join([salt, row.get("v63_candidate_id", ""), row.get("material_node_id", ""), row.get("query_history_id", "")])


def _stable_unit(value: str) -> float:
    digest = __import__("hashlib").sha256(value.encode("utf-8")).digest()
    raw = int.from_bytes(digest[:8], "big")
    return raw / float((1 << 64) - 1)


def _write_visualizations(result: dict[str, Any], root: Path) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    action_path = root / "policy_action_counts.png"
    real_rows = [row for row in result["selected_query_rows"] if row["policy_id"] == "R0_real_policy"]
    _write_bar_png(action_path, "v63 Phase 2 real policy actions", Counter(row["planned_action"] for row in real_rows))
    control_path = root / "policy_control_counts.png"
    _write_bar_png(control_path, "v63 Phase 2 policy/control budgets", Counter(row["policy_id"] for row in result["selected_query_rows"]))
    return {"policy_action_counts": _rel(action_path), "policy_control_counts": _rel(control_path)}


def _write_bar_png(path: Path, title: str, counts: Counter[str]) -> None:
    labels = list(counts)
    values = [int(counts[label]) for label in labels]
    width = max(900, 150 * max(1, len(labels)))
    height = 520
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(image, title, (36, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (32, 36, 44), 2, cv2.LINE_AA)
    max_value = max(values) if values else 1
    chart_x, chart_y, chart_h = 80, 120, 280
    step = int((width - 180) / max(1, len(labels)))
    bar_w = min(100, max(48, int(step * 0.52)))
    for idx, (label, value) in enumerate(zip(labels, values)):
        x0 = chart_x + idx * step + max(0, (step - bar_w) // 2)
        y1 = chart_y + chart_h
        y0 = int(y1 - value / max_value * chart_h)
        color = (80 + 29 * idx % 140, 140 + 17 * idx % 95, 210 - 23 * idx % 120)
        cv2.rectangle(image, (x0, y0), (x0 + bar_w, y1), color, -1)
        cv2.rectangle(image, (x0, y0), (x0 + bar_w, y1), (45, 49, 55), 1)
        cv2.putText(image, str(value), (x0 + 10, max(chart_y + 24, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (32, 36, 44), 1, cv2.LINE_AA)
        cv2.putText(image, label.replace("_", " ")[:22], (x0 - 18, y1 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (60, 66, 74), 1, cv2.LINE_AA)
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
