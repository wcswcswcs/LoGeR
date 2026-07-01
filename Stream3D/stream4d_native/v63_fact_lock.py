from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .v47_common import ROOT, read_json, utc_now, write_csv, write_json


PRIMARY_OUTCOME = "outputs/audit/v62_active_query_phase5c_material_outcome_primary/material_outcome_summary.json"
WITH_NEGATIVE_OUTCOME = "outputs/audit/v62_active_query_phase5e_material_outcome_with_negatives/material_outcome_summary.json"
WITH_NEGATIVE_AUC = "outputs/audit/v62_active_query_phase5e_control_auc_with_negatives/query_control_auc_summary.json"
V62_FINAL = "outputs/audit/v62_final/final_decision.json"


PLAN_EXPECTED: dict[str, Any] = {
    "primary_query_count": 80,
    "primary_valid_material_evidence_rate": 0.5125,
    "primary_query_to_confirm_rate": 0.2,
    "primary_query_to_quarantine_rate": 0.3,
    "primary_query_to_confirm_or_quarantine_rate": 0.5,
    "primary_confirm_count": 16,
    "primary_quarantine_count": 24,
    "primary_unresolved_count": 40,
    "with_negative_query_count": 164,
    "with_negative_valid_material_evidence_rate": 0.5914634146341463,
    "with_negative_query_to_confirm_or_quarantine_rate": 0.5365853658536586,
    "with_negative_confirm_count": 37,
    "with_negative_quarantine_count": 51,
    "with_negative_unresolved_count": 76,
    "with_negative_real_query_AUC": 0.9102870813397129,
    "with_negative_real_minus_shuffled_query_AUC": 0.26315789473684215,
    "with_negative_real_minus_no_temporal_query_AUC": 0.22562799043062198,
    "with_negative_diagnostic_real_query_AUC": 0.45226860254083484,
    "with_negative_diagnostic_real_minus_shuffled_query_AUC": 0.007622504537205088,
    "with_negative_diagnostic_real_minus_no_temporal_query_AUC": -0.03484573502722321,
}


@dataclass(frozen=True)
class V63FactLockConfig:
    output_root: str | Path = "outputs/audit/v63_phase0_fact_lock"
    visualization_root: str | Path = "outputs/audit/v63_visualizations/phase0"
    tolerance: float = 1.0e-12


def build_v63_fact_lock(config: V63FactLockConfig | None = None) -> dict[str, Any]:
    cfg = config or V63FactLockConfig()
    required_paths = {
        "v62_final_decision": V62_FINAL,
        "v62_primary_material_outcome": PRIMARY_OUTCOME,
        "v62_with_negative_material_outcome": WITH_NEGATIVE_OUTCOME,
        "v62_with_negative_control_auc": WITH_NEGATIVE_AUC,
    }
    path_status = [
        {
            "artifact": key,
            "path": path,
            "exists": _project(path).exists(),
            "repair_if_missing": _repair_hint(key),
        }
        for key, path in required_paths.items()
    ]
    missing = [row["path"] for row in path_status if not row["exists"]]
    payload: dict[str, Any] = {
        "v62_final": _read_if_exists(V62_FINAL),
        "primary": _read_if_exists(PRIMARY_OUTCOME),
        "with_negative": _read_if_exists(WITH_NEGATIVE_OUTCOME),
        "with_negative_auc": _read_if_exists(WITH_NEGATIVE_AUC),
    }
    observed = _observed_facts(payload)
    comparison_rows = _comparison_rows(observed, cfg.tolerance)
    all_expected_match = bool(comparison_rows) and all(bool(row["matches_plan_expected"]) for row in comparison_rows)
    all_inputs_exist = not missing
    final = payload["v62_final"]
    claim_table = final.get("claim_table", {}) if isinstance(final, dict) else {}
    claim_facts = {
        claim: {
            "label": row.get("label"),
            "pass": row.get("pass"),
            "evidence": row.get("evidence"),
        }
        for claim, row in claim_table.items()
        if isinstance(row, dict)
    }
    claim_c = claim_facts.get("Claim C", {})
    fact_lock_pass = bool(all_inputs_exist and all_expected_match and claim_c.get("pass") is False)
    metric_rows = _metric_rows(payload, observed, comparison_rows)
    summary = {
        "phase": "v63_phase0_fact_lock",
        "created_at": utc_now(),
        "plan_path": "docs/stream4d_v63_soma_query_interventional_material_evidence_plan.md",
        "input_artifacts": path_status,
        "missing_artifacts": missing,
        "v62_decision_label": final.get("decision_label") if isinstance(final, dict) else None,
        "v62_claim_table": claim_facts,
        "observed_v62_followup_facts": observed,
        "plan_expected_values": PLAN_EXPECTED,
        "plan_expected_comparison": comparison_rows,
        "claim_c_reframe": {
            "previous_v62_final_label": claim_c.get("label"),
            "previous_v62_final_pass": claim_c.get("pass"),
            "v62_followup_signal": (
                "Real D4RT outcome labels have strong temporal evidence separation, but this is not an independent "
                "objective-aligned active-query method claim."
            ),
            "blocker": "diagnostic_label_AUC_mismatch_and_self_referential_outcome_protocol",
            "v63_required_repair": (
                "Build independent heldout/future/decoy query candidates, action-utility controls, and ownership-state "
                "interventions before promoting Claim C."
            ),
        },
        "gate": {
            "required_v62_artifacts_exist": all_inputs_exist,
            "plan_expected_values_match_current_artifacts": all_expected_match,
            "claim_c_still_blocked_in_v62_final": claim_c.get("pass") is False,
            "pass": fact_lock_pass,
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "method_status": "fact_lock_only_no_v63_method_claim",
    }
    return {
        "summary": summary,
        "metric_rows": metric_rows,
    }


def write_v63_fact_lock(result: dict[str, Any], config: V63FactLockConfig | None = None) -> dict[str, str]:
    cfg = config or V63FactLockConfig()
    output_root = _project(cfg.output_root)
    visual_root = _project(cfg.visualization_root)
    output_root.mkdir(parents=True, exist_ok=True)
    visual_root.mkdir(parents=True, exist_ok=True)
    fact_lock_summary = output_root / "fact_lock_summary.json"
    metric_rows = output_root / "v62_followup_metric_rows.csv"
    write_json(fact_lock_summary, result["summary"])
    write_csv(metric_rows, result["metric_rows"])
    figure = visual_root / "v62_claimC_failure_reframe.png"
    _write_reframe_png(figure, result["summary"])
    return {
        "fact_lock_summary": _rel(fact_lock_summary),
        "v62_followup_metric_rows": _rel(metric_rows),
        "v62_claimC_failure_reframe": _rel(figure),
    }


def _observed_facts(payload: dict[str, Any]) -> dict[str, Any]:
    primary = payload.get("primary") or {}
    negative = payload.get("with_negative") or {}
    auc = payload.get("with_negative_auc") or {}
    primary_counts = primary.get("material_outcome_counts") or {}
    negative_counts = negative.get("material_outcome_counts") or {}
    return {
        "primary_query_count": primary.get("query_count"),
        "primary_valid_material_evidence_rate": primary.get("valid_material_evidence_rate"),
        "primary_query_to_confirm_rate": primary.get("query_to_confirm_rate"),
        "primary_query_to_quarantine_rate": primary.get("query_to_quarantine_rate"),
        "primary_query_to_confirm_or_quarantine_rate": primary.get("query_to_confirm_or_quarantine_rate"),
        "primary_confirm_count": primary_counts.get("confirm"),
        "primary_quarantine_count": primary_counts.get("quarantine"),
        "primary_unresolved_count": primary_counts.get("unresolved"),
        "with_negative_query_count": negative.get("query_count"),
        "with_negative_valid_material_evidence_rate": negative.get("valid_material_evidence_rate"),
        "with_negative_query_to_confirm_or_quarantine_rate": negative.get("query_to_confirm_or_quarantine_rate"),
        "with_negative_confirm_count": negative_counts.get("confirm"),
        "with_negative_quarantine_count": negative_counts.get("quarantine"),
        "with_negative_unresolved_count": negative_counts.get("unresolved"),
        "with_negative_real_query_AUC": auc.get("real_query_AUC"),
        "with_negative_real_minus_shuffled_query_AUC": auc.get("real_minus_shuffled_query_AUC"),
        "with_negative_real_minus_no_temporal_query_AUC": auc.get("real_minus_no_temporal_query_AUC"),
        "with_negative_diagnostic_real_query_AUC": auc.get("diagnostic_real_query_AUC"),
        "with_negative_diagnostic_real_minus_shuffled_query_AUC": auc.get("diagnostic_real_minus_shuffled_query_AUC"),
        "with_negative_diagnostic_real_minus_no_temporal_query_AUC": auc.get("diagnostic_real_minus_no_temporal_query_AUC"),
    }


def _comparison_rows(observed: dict[str, Any], tolerance: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, expected in PLAN_EXPECTED.items():
        value = observed.get(key)
        rows.append(
            {
                "metric_name": key,
                "observed_value": value,
                "plan_expected_value": expected,
                "absolute_error": _abs_error(value, expected),
                "matches_plan_expected": _matches(value, expected, tolerance),
                "tolerance": tolerance,
            }
        )
    return rows


def _metric_rows(payload: dict[str, Any], observed: dict[str, Any], comparison_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sources = {
        "primary": PRIMARY_OUTCOME,
        "with_negative": WITH_NEGATIVE_OUTCOME,
        "with_negative_auc": WITH_NEGATIVE_AUC,
        "v62_final": V62_FINAL,
    }
    groups = {
        "primary": [key for key in observed if key.startswith("primary_")],
        "with_negative": [
            key
            for key in observed
            if key.startswith("with_negative_") and "_AUC" not in key and "real_minus" not in key and "diagnostic" not in key
        ],
        "with_negative_auc": [key for key in observed if key.startswith("with_negative_") and key not in {"with_negative_query_count"} and ("AUC" in key or "real_minus" in key or "diagnostic" in key)],
    }
    comparison_by_key = {row["metric_name"]: row for row in comparison_rows}
    for group, keys in groups.items():
        for key in keys:
            comp = comparison_by_key.get(key, {})
            rows.append(
                {
                    "metric_group": group,
                    "metric_name": key,
                    "value": observed.get(key),
                    "plan_expected_value": comp.get("plan_expected_value"),
                    "matches_plan_expected": comp.get("matches_plan_expected"),
                    "evidence_path": sources[group],
                    "interpretation": _interpret_metric(key, observed.get(key)),
                    "uses_gt_for_prediction": False,
                }
            )
    final = payload.get("v62_final") or {}
    for claim, claim_payload in (final.get("claim_table") or {}).items():
        rows.append(
            {
                "metric_group": "v62_final_claim",
                "metric_name": claim,
                "value": claim_payload.get("label"),
                "pass": claim_payload.get("pass"),
                "evidence_path": V62_FINAL,
                "interpretation": claim_payload.get("evidence"),
                "uses_gt_for_prediction": False,
            }
        )
    rows.append(
        {
            "metric_group": "v63_phase0_decision",
            "metric_name": "blocker_reframe",
            "value": "diagnostic_label_AUC_mismatch_and_self_referential_outcome_protocol",
            "evidence_path": WITH_NEGATIVE_AUC,
            "interpretation": "Outcome-label AUC is strong; diagnostic-label AUC does not support independent active-query success.",
            "uses_gt_for_prediction": False,
        }
    )
    return rows


def _interpret_metric(key: str, value: Any) -> str:
    if key == "with_negative_real_query_AUC":
        return "Strong separation against v62 material outcome labels; not independent method success."
    if key == "with_negative_diagnostic_real_query_AUC":
        return "Diagnostic-label AUC remains below random-like/useful active-query threshold."
    if "real_minus_no_temporal" in key or "real_minus_shuffled" in key:
        return "Control delta used to locate the v62/v63 target mismatch."
    if "confirm" in key or "quarantine" in key:
        return "Action outcome count/rate for confirm/quarantine/defer framing."
    return f"Observed v62 follow-up value: {value}"


def _write_reframe_png(path: Path, summary: dict[str, Any]) -> None:
    observed = summary.get("observed_v62_followup_facts") or {}
    bars = [
        ("outcome AUC", observed.get("with_negative_real_query_AUC"), (58, 134, 255)),
        ("diag AUC", observed.get("with_negative_diagnostic_real_query_AUC"), (245, 126, 67)),
        ("real-shuf", observed.get("with_negative_real_minus_shuffled_query_AUC"), (72, 181, 118)),
        ("diag-shuf", observed.get("with_negative_diagnostic_real_minus_shuffled_query_AUC"), (196, 92, 204)),
    ]
    width, height = 1000, 560
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(image, "v63 Phase 0: v62 Claim C Failure Reframe", (42, 54), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 34, 42), 2, cv2.LINE_AA)
    cv2.putText(image, "Strong outcome signal does not equal independent active-query success", (42, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (82, 88, 96), 1, cv2.LINE_AA)
    chart_x, chart_y, chart_w, chart_h = 90, 150, 840, 270
    cv2.rectangle(image, (chart_x, chart_y), (chart_x + chart_w, chart_y + chart_h), (225, 230, 236), 1)
    for tick in range(0, 6):
        y = int(chart_y + chart_h - tick * chart_h / 5)
        cv2.line(image, (chart_x, y), (chart_x + chart_w, y), (238, 241, 245), 1)
        cv2.putText(image, f"{tick / 5:.1f}", (36, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (92, 98, 106), 1, cv2.LINE_AA)
    bar_w = 120
    gap = 70
    for idx, (label, raw_value, color) in enumerate(bars):
        value = float(raw_value) if raw_value is not None and math.isfinite(float(raw_value)) else 0.0
        display = max(0.0, min(1.0, value))
        x0 = chart_x + 80 + idx * (bar_w + gap)
        x1 = x0 + bar_w
        y1 = chart_y + chart_h
        y0 = int(y1 - display * chart_h)
        cv2.rectangle(image, (x0, y0), (x1, y1), color, -1)
        cv2.rectangle(image, (x0, y0), (x1, y1), (45, 49, 55), 1)
        cv2.putText(image, f"{value:.3f}", (x0 + 8, max(chart_y + 22, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 34, 42), 1, cv2.LINE_AA)
        cv2.putText(image, label, (x0 - 8, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (45, 49, 55), 1, cv2.LINE_AA)
    cv2.putText(image, "Phase 0 decision: lock v62 facts, keep Claim C blocked, require v63 heldout/future/decoy intervention evidence.", (42, 486), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (30, 34, 42), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def _read_if_exists(path: str | Path) -> dict[str, Any]:
    full = _project(path)
    return read_json(full) if full.exists() else {}


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def _matches(value: Any, expected: Any, tolerance: float) -> bool:
    if isinstance(expected, int) and not isinstance(expected, bool):
        return value is not None and int(value) == int(expected)
    try:
        return value is not None and abs(float(value) - float(expected)) <= tolerance
    except (TypeError, ValueError):
        return value == expected


def _abs_error(value: Any, expected: Any) -> float | None:
    try:
        return abs(float(value) - float(expected))
    except (TypeError, ValueError):
        return None


def _repair_hint(artifact: str) -> str:
    return {
        "v62_final_decision": "run Stream3D/tools/build_v62_final_decision.py after verifying v62 phase summaries",
        "v62_primary_material_outcome": "rerun Stream3D/tools/run_v62_active_query_material_outcome.py for primary smoke roots",
        "v62_with_negative_material_outcome": "rerun negative candidate smoke and Stream3D/tools/run_v62_active_query_material_outcome.py",
        "v62_with_negative_control_auc": "rerun Stream3D/tools/run_v62_active_query_control_auc.py for with-negative material outcome rows",
    }.get(artifact, "locate or regenerate the missing v62 artifact before proceeding")
