#!/usr/bin/env python3
"""Audit v101 same-space state and Q2 true-stage readiness.

v101 intentionally separates same-space instrumentation from action readiness.
This audit summarizes whether the existing Track S2/U/V/W/Q2 artifacts can
promote from proxy diagnostics into a true-stage admission/action surface.
It does not modify any gate outcome or run runtime pilots.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
FINAL = ROOT / "final_decision"


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"read_error": f"{type(exc).__name__}:{exc}"}
    return payload if isinstance(payload, dict) else {"value": payload}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_clean(v) for v in value]
    if isinstance(value, tuple):
        return [json_clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_clean(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    if not keys:
        keys = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: stringify(row.get(key, "")) for key in keys})


def stringify(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(json_clean(value), ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return str(value)


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def nonempty(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def summarize_state_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    identity_counts: Counter[str] = Counter()
    support_flag_counts: Counter[str] = Counter()
    state_status_counts: Counter[str] = Counter()
    taxonomy_counts: Counter[str] = Counter()
    proxy_only_count = 0
    geometry_terms_available_count = 0
    scale_proxy_count = 0
    temporal_proxy_count = 0
    r_write_nonempty = 0
    r_cache_nonempty = 0
    r_ref_nonempty = 0
    forbidden_runtime_count = 0
    for row in rows:
        identity_counts[row.get("identity_resolution_level", "")] += 1
        state_status_counts[row.get("state_status", "")] += 1
        taxonomy_counts[row.get("target_taxonomy", "")] += 1
        proxy_only_count += int(as_bool(row.get("proxy_only", "")))
        geometry_terms_available_count += int(as_bool(row.get("geometry_sidecar_terms_available", "")))
        scale_proxy_count += int(as_bool(row.get("scale_observability_score_is_proxy", "")))
        temporal_proxy_count += int(as_bool(row.get("temporal_proxy_only", "")))
        r_write_nonempty += int(nonempty(row.get("r_write_cache")))
        r_cache_nonempty += int(nonempty(row.get("r_cache_current")))
        r_ref_nonempty += int(nonempty(row.get("r_ref_current")))
        forbidden_runtime_count += int(row.get("forbidden_behavior", "") == "runtime_scale_update_without_Q2_M4")
        for part in row.get("support_source_flags", "").split(";"):
            if part:
                support_flag_counts[part] += 1
    return {
        "state_row_count": len(rows),
        "identity_resolution_level_counts": dict(identity_counts),
        "support_source_flag_counts": dict(support_flag_counts),
        "state_status_counts_from_rows": dict(state_status_counts),
        "target_taxonomy_counts_from_rows": dict(taxonomy_counts),
        "proxy_only_row_count": proxy_only_count,
        "geometry_sidecar_terms_available_row_count": geometry_terms_available_count,
        "scale_observability_score_is_proxy_row_count": scale_proxy_count,
        "temporal_proxy_only_row_count": temporal_proxy_count,
        "r_write_cache_nonempty_row_count": r_write_nonempty,
        "r_cache_current_nonempty_row_count": r_cache_nonempty,
        "r_ref_current_nonempty_row_count": r_ref_nonempty,
        "forbidden_runtime_without_q2_m4_row_count": forbidden_runtime_count,
    }


def make_readiness_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "readiness_item": "stage0_same_space_instrumentation",
            "status": "pass" if summary["v100_trackS_gate_pass"] is True else "fail",
            "evidence": f"best_canonical_space={summary['v100_best_canonical_space']}; v100_trackS_gate_pass={summary['v100_trackS_gate_pass']}",
            "action_authorizing": False,
        },
        {
            "readiness_item": "explicit_no_action_parity_current_artifact",
            "status": "weak_or_missing",
            "evidence": "Plan text records no-action parity, but current v101 stage0 summary does not carry a dedicated no_action_parity field.",
            "action_authorizing": False,
        },
        {
            "readiness_item": "trackU_true_current_support_strict",
            "status": "pass" if summary["trackU_true_current_support_strict_pass"] is True else "fail",
            "evidence": f"gate_pass={summary['trackU_gate_pass']}; proxy_only={summary['trackU_proxy_only']}; true_current_support_strict_pass={summary['trackU_true_current_support_strict_pass']}",
            "action_authorizing": False,
        },
        {
            "readiness_item": "trackV_true_scale_observability",
            "status": "pass" if summary["trackV_gate_pass"] is True else "fail",
            "evidence": f"gate_pass={summary['trackV_gate_pass']}; geometry_materialization_pass={summary['trackV_geometry_materialization_pass']}; proxy_only={summary['trackV_proxy_only']}",
            "action_authorizing": False,
        },
        {
            "readiness_item": "trackS2_state_estimator_gate",
            "status": "pass" if summary["trackS2_gate_pass"] is True else "fail",
            "evidence": f"supported_consistent_mean_L3={summary['trackS2_supported_consistent_mean_L3']}; unsupported_inconsistent_mean_L3={summary['trackS2_unsupported_inconsistent_mean_L3']}; proxy_only={summary['trackS2_proxy_only']}",
            "action_authorizing": False,
        },
        {
            "readiness_item": "trackQ2_true_stage",
            "status": "pass" if summary["trackQ2_true_stage_pass"] is True else "fail",
            "evidence": f"proxy_only={summary['trackQ2_proxy_only']}; proxy_stage_pass={summary['trackQ2_proxy_stage_pass']}; true_stage_pass={summary['trackQ2_true_stage_pass']}; good_FPR={summary['trackQ2_good_FPR']}",
            "action_authorizing": False,
        },
        {
            "readiness_item": "target_universe_sequence_coverage",
            "status": "pass" if summary["trackQ2_selected_positive_sequence_coverage"] >= 2 else "fail",
            "evidence": f"target_positive_count={summary['trackQ2_target_positive_count']}; safe_good_count={summary['trackQ2_safe_good_count']}; selected_positive_sequence_coverage={summary['trackQ2_selected_positive_sequence_coverage']}",
            "action_authorizing": False,
        },
        {
            "readiness_item": "write_cache_current_chain",
            "status": "pass"
            if summary["r_write_cache_nonempty_row_count"] > 0
            and summary["r_cache_current_nonempty_row_count"] > 0
            and summary["r_ref_current_nonempty_row_count"] > 0
            else "fail",
            "evidence": f"r_write_cache_nonempty={summary['r_write_cache_nonempty_row_count']}; r_cache_current_nonempty={summary['r_cache_current_nonempty_row_count']}; r_ref_current_nonempty={summary['r_ref_current_nonempty_row_count']}",
            "action_authorizing": False,
        },
    ]
    return rows


def write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# ACL2 v101 State/Q2 Readiness Audit",
        "",
        "This audit separates reusable same-space instrumentation from action readiness.",
        "",
        "## Summary",
        "",
        f"- native_same_space_instrumentation_reusable: {summary['native_same_space_instrumentation_reusable']}",
        f"- state_q2_readiness_pass: {summary['state_q2_readiness_pass']}",
        f"- action_ready: {summary['action_ready']}",
        f"- trackS2_gate_pass: {summary['trackS2_gate_pass']}",
        f"- trackQ2_true_stage_pass: {summary['trackQ2_true_stage_pass']}",
        f"- trackU_true_current_support_strict_pass: {summary['trackU_true_current_support_strict_pass']}",
        f"- trackV_gate_pass: {summary['trackV_gate_pass']}",
        f"- proxy_only_row_count: {summary['proxy_only_row_count']}",
        f"- r_write_cache_nonempty_row_count: {summary['r_write_cache_nonempty_row_count']}",
        f"- r_cache_current_nonempty_row_count: {summary['r_cache_current_nonempty_row_count']}",
        f"- r_ref_current_nonempty_row_count: {summary['r_ref_current_nonempty_row_count']}",
        "",
        "## Readiness Rows",
        "",
    ]
    for row in rows:
        lines.append(f"- {row['readiness_item']}: status={row['status']}; {row['evidence']}")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Same-space instrumentation is reusable, but action readiness remains blocked by proxy-only current support/observability, failed S2/Q2 gates, insufficient target universe coverage, and missing write/cache/current chain materialization.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    stage0 = read_json(ROOT / "stage0_v101_evidence_ledger/summary.json")
    track_u = read_json(ROOT / "trackU_true_current_support/current_support_summary.json")
    track_v = read_json(ROOT / "trackV_anchor_scale_observability/observability_summary.json")
    track_v_geom = read_json(ROOT / "trackV_anchor_scale_observability/per_anchor_geometry_observability_summary.json")
    track_w = read_json(ROOT / "trackW_anchor_memory_role/role_summary.json")
    track_s2 = read_json(ROOT / "trackS2_anchor_state_estimator/state_estimator_summary.json")
    track_q2 = read_json(ROOT / "trackQ2_scale_update_admission/Q2_summary.json")
    state_rows = read_rows(ROOT / "trackS2_anchor_state_estimator/anchor_state_rows.csv")
    row_summary = summarize_state_rows(state_rows)

    q2_metrics = track_q2.get("metrics", {})
    summary: dict[str, Any] = {
        "schema": "acl2_v101_state_q2_readiness_v1",
        "v100_trackS_gate_pass": stage0.get("v100_trackS_gate_pass", False),
        "v100_best_canonical_space": stage0.get("v100_best_canonical_space", ""),
        "v100_trackQ_proxy_only": stage0.get("v100_trackQ_proxy_only", ""),
        "trackU_gate_pass": track_u.get("gate_pass", False),
        "trackU_proxy_only": track_u.get("proxy_only", True),
        "trackU_true_current_support_strict_pass": track_u.get("true_current_support_strict_pass", False),
        "trackV_gate_pass": track_v.get("gate_pass", False),
        "trackV_proxy_only": track_v.get("proxy_only", True),
        "trackV_geometry_materialization_pass": track_v_geom.get("geometry_materialization_pass", False),
        "trackW_gate_pass": track_w.get("gate_pass", False),
        "trackW_proxy_only": track_w.get("proxy_only", True),
        "trackS2_gate_pass": track_s2.get("gate_pass", False),
        "trackS2_proxy_only": track_s2.get("proxy_only", True),
        "trackS2_supported_consistent_mean_L3": track_s2.get("supported_consistent_mean_L3", ""),
        "trackS2_unsupported_inconsistent_mean_L3": track_s2.get("unsupported_inconsistent_mean_L3", ""),
        "trackQ2_gate_pass": track_q2.get("gate_pass", False),
        "trackQ2_proxy_stage_pass": track_q2.get("proxy_stage_pass", False),
        "trackQ2_true_stage_pass": track_q2.get("true_stage_pass", False),
        "trackQ2_proxy_only": track_q2.get("proxy_only", True),
        "trackQ2_good_FPR": q2_metrics.get("good_FPR", ""),
        "trackQ2_bad_recall": q2_metrics.get("bad_recall", ""),
        "trackQ2_balanced_accuracy": q2_metrics.get("balanced_accuracy", ""),
        "trackQ2_target_positive_count": track_q2.get("target_positive_count", ""),
        "trackQ2_safe_good_count": track_q2.get("safe_good_count", ""),
        "trackQ2_selected_positive_sequence_coverage": q2_metrics.get("selected_positive_sequence_coverage", 0),
        **row_summary,
        "runtime_action_allowed": False,
        "full_validation_run": False,
        "full_method_success": False,
        "claim": "State/Q2 readiness audit only; no action is authorized.",
    }
    summary["native_same_space_instrumentation_reusable"] = (
        summary["v100_trackS_gate_pass"] is True
        and summary["v100_best_canonical_space"] == "S-B_preprojection_hidden"
    )
    summary["state_q2_readiness_pass"] = (
        summary["native_same_space_instrumentation_reusable"] is True
        and summary["trackU_true_current_support_strict_pass"] is True
        and summary["trackV_gate_pass"] is True
        and summary["trackS2_gate_pass"] is True
        and summary["trackQ2_true_stage_pass"] is True
        and summary["r_write_cache_nonempty_row_count"] > 0
        and summary["r_cache_current_nonempty_row_count"] > 0
        and summary["r_ref_current_nonempty_row_count"] > 0
    )
    summary["action_ready"] = False
    summary["blocked_reason"] = (
        "Same-space instrumentation is reusable, but Track U/V/S2/Q2 remain proxy/failed for action: "
        "true current support strict pass is false, Track V gate is false, S2 gate is false, Q2 true-stage is false, "
        "target sequence coverage is insufficient, and write/cache/current residual chain fields are empty."
    )
    readiness_rows = make_readiness_rows(summary)
    write_rows(FINAL / "state_q2_readiness_rows.csv", readiness_rows)
    write_json(FINAL / "state_q2_readiness_summary.json", summary)
    write_report(FINAL / "state_q2_readiness_report.md", summary, readiness_rows)
    print(json.dumps(json_clean(summary), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
