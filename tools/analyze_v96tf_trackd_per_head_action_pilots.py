#!/usr/bin/env python3
"""Summarize v96 Track D READ action pilots."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v96tf_vggt4d_informed_semantic_gauge_preserving_memory_control")
OUT_DIR = ROOT / "trackD_read_gauge_preserving_action"
PILOTS = [
    ("layer8_head4", 8, 4, ROOT / "trackD_read_per_head_dg_q90_l8h4_action_pilot"),
    ("layer10_head12", 10, 12, ROOT / "trackD_read_per_head_dg_q90_l10h12_action_pilot"),
    ("layer6_head4", 6, 4, ROOT / "trackD_read_per_head_dg_q90_l6h4_action_pilot"),
    (
        "layer8_head4_anchor_rescue",
        8,
        4,
        ROOT / "trackD_read_per_head_dg_q90_anchor_rescue_l8h4_action_pilot",
    ),
    (
        "layer10_head12_anchor_rescue",
        10,
        12,
        ROOT / "trackD_read_per_head_dg_q90_anchor_rescue_l10h12_action_pilot",
    ),
    (
        "layer6_head4_anchor_rescue",
        6,
        4,
        ROOT / "trackD_read_per_head_dg_q90_anchor_rescue_l6h4_action_pilot",
    ),
]
L07_PILOTS = [
    (
        "l07_gauge_norm_t030",
        ROOT / "trackD_read_gauge_normalized_l07_action_pilot_v2",
        "READ21_GATED_L07_GAUGE_NORM_T030",
    ),
    (
        "l07_gauge_norm_t045",
        ROOT / "trackD_read_gauge_normalized_l07_action_pilot_t045",
        "READ21_GATED_L07_GAUGE_NORM_T045",
    ),
    (
        "l07_gauge_norm_t050",
        ROOT / "trackD_read_gauge_normalized_l07_action_pilot_t050",
        "READ21_GATED_L07_GAUGE_NORM_T050",
    ),
    (
        "l07_confneutral_t030",
        ROOT / "trackD_read_gauge_normalized_l07_confneutral_t030_pilot",
        "READ21_GATED_L07_CONFNEUTRAL_T030",
    ),
    (
        "l07_confneutral_t035",
        ROOT / "trackD_read_gauge_normalized_l07_confneutral_t035_pilot_v2",
        "READ21_GATED_L07_CONFNEUTRAL_T035",
    ),
    (
        "l07_confneutral_t045",
        ROOT / "trackD_read_gauge_normalized_l07_confneutral_t045_pilot",
        "READ21_GATED_L07_CONFNEUTRAL_T045",
    ),
    (
        "l07_confneutral_t050",
        ROOT / "trackD_read_gauge_normalized_l07_confneutral_t050_pilot",
        "READ21_GATED_L07_CONFNEUTRAL_T050",
    ),
    (
        "l07_confneutral_anchorcomp_t050_gatebound",
        ROOT / "trackD_read_gauge_normalized_l07_confneutral_anchorcomp_t050_pilot_v3_gatebound",
        "READ21_GATED_L07_CONFNEUTRAL_ANCHORCOMP_T050",
    ),
    (
        "l07_confneutral_anchorcomp_t050_gatebound_minscore0005",
        ROOT / "trackD_read_gauge_normalized_l07_confneutral_anchorcomp_t050_pilot_v4_gatebound_minscore0005",
        "READ21_GATED_L07_CONFNEUTRAL_ANCHORCOMP_T050",
    ),
]
CARRIER_L07_PILOTS = [
    (
        "carrier_gated_l07_l8h4_qtail_t050",
        8,
        4,
        "tail",
        ROOT / "trackD_carrier_gated_l07_l8h4_qtail_t050_pilot_v1",
        "READ21_GATED_L07_CONFNEUTRAL_T050",
        "gated chunk-level L07 body scoped by raw-QK carrier head/query",
    ),
    (
        "carrier_ungated_l07_l8h4_qtail_t050",
        8,
        4,
        "tail",
        ROOT / "trackD_carrier_ungated_l07_l8h4_qtail_t050_pilot_v1",
        "READ_L07_CONFNEUTRAL_CARRIER_T050",
        "ungated L07 body scoped by raw-QK carrier head/query",
    ),
    (
        "carrier_ungated_l07_l10h12_qtail_t050",
        10,
        12,
        "tail",
        ROOT / "trackD_carrier_ungated_l07_l10h12_qtail_t050_pilot_v1",
        "READ_L07_CONFNEUTRAL_CARRIER_T050",
        "ungated L07 body scoped by raw-QK carrier head/query",
    ),
    (
        "carrier_ungated_l07_l6h4_qtail_t050",
        6,
        4,
        "tail",
        ROOT / "trackD_carrier_ungated_l07_l6h4_qtail_t050_pilot_v1",
        "READ_L07_CONFNEUTRAL_CARRIER_T050",
        "ungated L07 body scoped by raw-QK carrier head/query",
    ),
]
QKPAIR_KEYSTAB_PILOTS = [
    (
        "qkpair_keystab_l8h4_qtail_t050",
        8,
        4,
        "tail",
        ROOT / "trackD_qkpair_keystab_l8h4_qtail_t050_pilot_v1",
        "READ_QKPAIR_KEYSTAB_CARRIER_T050",
        "source-target QK pair key-stability body scoped by sampled carrier head/query",
    ),
    (
        "qkpair_keystab_l10h12_qtail_t050",
        10,
        12,
        "tail",
        ROOT / "trackD_qkpair_keystab_l10h12_qtail_t050_pilot_v1",
        "READ_QKPAIR_KEYSTAB_CARRIER_T050",
        "source-target QK pair key-stability body scoped by sampled carrier head/query",
    ),
    (
        "qkpair_keystab_l6h4_qtail_t050",
        6,
        4,
        "tail",
        ROOT / "trackD_qkpair_keystab_l6h4_qtail_t050_pilot_v1",
        "READ_QKPAIR_KEYSTAB_CARRIER_T050",
        "source-target QK pair key-stability body scoped by sampled carrier head/query",
    ),
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out and abs(out) != float("inf") else default


def _xml_escape(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_gate_margin_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    threshold = 0.05
    bar_x = 250
    bar_w = 420
    row_h = 42
    height = 96 + row_h * max(len(rows), 1) * 2
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="920" height="%d" viewBox="0 0 920 %d">' % (height, height),
        '<rect width="920" height="%d" fill="#ffffff"/>' % height,
        '<text x="24" y="34" font-family="monospace" font-size="18" fill="#111111">TrackD per-head action gate margins</text>',
        '<text x="24" y="58" font-family="monospace" font-size="13" fill="#444444">Bars are scaled to the 0.05 mechanism threshold; rows above the marker passed the short-window margin gate.</text>',
        '<line x1="%d" y1="78" x2="%d" y2="%d" stroke="#666666" stroke-dasharray="4 4"/>' % (
            bar_x + bar_w,
            bar_x + bar_w,
            height - 16,
        ),
        '<text x="%d" y="74" font-family="monospace" font-size="12" fill="#444444">0.05 gate</text>' % (bar_x + bar_w - 34),
    ]
    y = 100
    for row in rows:
        pilot = _xml_escape(row.get("pilot", ""))
        metric = _xml_escape(row.get("best_bad_improvement_metric", ""))
        bad = max(0.0, finite(row.get("best_bad_improvement_vs_baseline"), 0.0))
        margin = max(0.0, finite(row.get("best_candidate_margin_vs_random_same_mass"), 0.0))
        bad_w = min(bad / threshold, 1.0) * bar_w
        margin_w = min(margin / threshold, 1.0) * bar_w
        parts.extend([
            '<text x="24" y="%d" font-family="monospace" font-size="12" fill="#111111">%s</text>' % (y, pilot),
            '<text x="24" y="%d" font-family="monospace" font-size="11" fill="#555555">%s</text>' % (y + 15, metric),
            '<rect x="%d" y="%d" width="%.2f" height="12" fill="#b23b3b"/>' % (bar_x, y - 10, bad_w),
            '<text x="%d" y="%d" font-family="monospace" font-size="11" fill="#111111">bad %.6f</text>' % (bar_x + bar_w + 14, y, bad),
            '<rect x="%d" y="%d" width="%.2f" height="12" fill="#3366aa"/>' % (bar_x, y + 9, margin_w),
            '<text x="%d" y="%d" font-family="monospace" font-size="11" fill="#111111">margin %.6f</text>' % (bar_x + bar_w + 14, y + 19, margin),
        ])
        y += row_h * 2
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def analyze() -> dict[str, Any]:
    carrier = read_json(ROOT / "trackG_read_qk_carrier_localization/summary.json")
    stage7 = read_json(ROOT / "stage7_full_validation" / "summary.json")
    rows: list[dict[str, Any]] = []
    pilot_summaries: dict[str, Any] = {}
    for pilot_name, layer, head, root in PILOTS:
        summary = read_json(root / "summary.json")
        pilot_summaries[pilot_name] = summary
        metric_decisions = summary.get("metric_decisions", {}) if isinstance(summary.get("metric_decisions"), dict) else {}
        for metric, decision in metric_decisions.items():
            row = {
                "pilot": pilot_name,
                "layer": layer,
                "head": head,
                "pilot_root": str(root),
                "status": summary.get("status", ""),
                "gate_pass": bool(summary.get("gate_pass", False)),
                "trace_fidelity_pass": bool(summary.get("trace_fidelity_pass", False)),
                "stable_anchor_proxy_pass": bool(summary.get("stable_anchor_proxy_pass", False)),
                "good_safety_pass": bool(summary.get("good_safety_pass", False)),
                "trace_median_attention_mass_delta": summary.get("trace_median_attention_mass_delta"),
                "stable_anchor_preservation_proxy": summary.get("stable_anchor_preservation_proxy"),
                "good_j_short_proxy_worsen_ratio": summary.get("good_j_short_proxy_worsen_ratio"),
                "metric": metric,
            }
            if isinstance(decision, dict):
                row.update(decision)
            rows.append(row)
    for pilot_name, root, candidate in L07_PILOTS:
        summary = read_json(root / "summary.json")
        pilot_summaries[pilot_name] = summary
        metric_decisions = summary.get("metric_decisions", {}) if isinstance(summary.get("metric_decisions"), dict) else {}
        for metric, decision in metric_decisions.items():
            row = {
                "pilot": pilot_name,
                "pilot_family": "gauge_normalized_l07",
                "layer": "",
                "head": "",
                "pilot_root": str(root),
                "candidate": candidate,
                "status": summary.get("status", ""),
                "gate_pass": bool(summary.get("gate_pass", False)),
                "trace_fidelity_pass": bool(summary.get("trace_fidelity_pass", False)),
                "stable_anchor_proxy_pass": bool(summary.get("stable_anchor_proxy_pass", False)),
                "global_safety_proxy_pass": bool(summary.get("global_safety_proxy_pass", False)),
                "good_safety_pass": bool(summary.get("good_safety_pass", False)),
                "trace_median_attention_mass_delta": summary.get("trace_median_attention_mass_delta"),
                "trace_frame_bias_negative_pair_mass_lift_median": summary.get("trace_frame_bias_negative_pair_mass_lift_median"),
                "trace_beta_frame_effective_active_median": summary.get("trace_beta_frame_effective_active_median"),
                "trace_beta_raw_frame_bias_energy_active_median": summary.get("trace_beta_raw_frame_bias_energy_active_median"),
                "stable_anchor_preservation_proxy": summary.get("stable_anchor_preservation_proxy"),
                "good_j_short_proxy_worsen_ratio": summary.get("good_j_short_proxy_worsen_ratio"),
                "metric": metric,
            }
            if isinstance(decision, dict):
                row.update(decision)
            rows.append(row)
    for pilot_name, layer, head, query_region, root, candidate, note in CARRIER_L07_PILOTS:
        summary = read_json(root / "summary.json")
        pilot_summaries[pilot_name] = summary
        metric_decisions = summary.get("metric_decisions", {}) if isinstance(summary.get("metric_decisions"), dict) else {}
        for metric, decision in metric_decisions.items():
            row = {
                "pilot": pilot_name,
                "pilot_family": "carrier_scoped_l07",
                "layer": layer,
                "head": head,
                "frame_bias_query_region": query_region,
                "pilot_root": str(root),
                "candidate": candidate,
                "status": summary.get("status", ""),
                "gate_pass": bool(summary.get("gate_pass", False)),
                "trace_fidelity_pass": bool(summary.get("trace_fidelity_pass", False)),
                "stable_anchor_proxy_pass": bool(summary.get("stable_anchor_proxy_pass", False)),
                "global_safety_proxy_pass": bool(summary.get("global_safety_proxy_pass", False)),
                "good_safety_pass": bool(summary.get("good_safety_pass", False)),
                "trace_median_attention_mass_delta": summary.get("trace_median_attention_mass_delta"),
                "trace_frame_bias_negative_pair_mass_lift_median": summary.get("trace_frame_bias_negative_pair_mass_lift_median"),
                "trace_beta_frame_effective_active_median": summary.get("trace_beta_frame_effective_active_median"),
                "trace_beta_raw_frame_bias_energy_active_median": summary.get("trace_beta_raw_frame_bias_energy_active_median"),
                "trace_beta_frame_effective_median": summary.get("trace_beta_frame_effective_median"),
                "trace_beta_raw_frame_bias_energy_median": summary.get("trace_beta_raw_frame_bias_energy_median"),
                "trace_v95_gate_active_frac_median": summary.get("trace_v95_gate_active_frac_median"),
                "stable_anchor_preservation_proxy": summary.get("stable_anchor_preservation_proxy"),
                "good_j_short_proxy_worsen_ratio": summary.get("good_j_short_proxy_worsen_ratio"),
                "carrier_scope_note": note,
                "metric": metric,
            }
            if isinstance(decision, dict):
                row.update(decision)
            rows.append(row)
    for pilot_name, layer, head, query_region, root, candidate, note in QKPAIR_KEYSTAB_PILOTS:
        summary = read_json(root / "summary.json")
        pilot_summaries[pilot_name] = summary
        metric_decisions = summary.get("metric_decisions", {}) if isinstance(summary.get("metric_decisions"), dict) else {}
        for metric, decision in metric_decisions.items():
            row = {
                "pilot": pilot_name,
                "pilot_family": "qkpair_keystab",
                "layer": layer,
                "head": head,
                "frame_bias_query_region": query_region,
                "pilot_root": str(root),
                "candidate": candidate,
                "status": summary.get("status", ""),
                "gate_pass": bool(summary.get("gate_pass", False)),
                "trace_fidelity_pass": bool(summary.get("trace_fidelity_pass", False)),
                "stable_anchor_proxy_pass": bool(summary.get("stable_anchor_proxy_pass", False)),
                "global_safety_proxy_pass": bool(summary.get("global_safety_proxy_pass", False)),
                "good_safety_pass": bool(summary.get("good_safety_pass", False)),
                "trace_median_attention_mass_delta": summary.get("trace_median_attention_mass_delta"),
                "trace_frame_bias_negative_pair_mass_lift_median": summary.get("trace_frame_bias_negative_pair_mass_lift_median"),
                "trace_beta_frame_effective_median": summary.get("trace_beta_frame_effective_median"),
                "trace_beta_raw_frame_bias_energy_median": summary.get("trace_beta_raw_frame_bias_energy_median"),
                "trace_v95_gate_active_frac_median": summary.get("trace_v95_gate_active_frac_median"),
                "stable_anchor_preservation_proxy": summary.get("stable_anchor_preservation_proxy"),
                "good_j_short_proxy_worsen_ratio": summary.get("good_j_short_proxy_worsen_ratio"),
                "carrier_scope_note": note,
                "metric": metric,
            }
            if isinstance(decision, dict):
                row.update(decision)
            rows.append(row)
    completed = [item for item in pilot_summaries.values() if item.get("status") == "complete"]
    expected_pilot_count = len(PILOTS) + len(L07_PILOTS) + len(CARRIER_L07_PILOTS) + len(QKPAIR_KEYSTAB_PILOTS)
    any_gate_pass = any(bool(item.get("gate_pass", False)) for item in pilot_summaries.values())
    any_bad_metric_pass = any(
        bool(row.get("bad_metric_gate_pass", False))
        for row in rows
    )
    best_bad_improvement_row = max(
        rows,
        key=lambda row: finite(row.get("bad_improvement_vs_baseline"), -999.0),
        default={},
    )
    best_control_margin_row = max(
        rows,
        key=lambda row: finite(row.get("candidate_margin_vs_random_same_mass"), -999.0),
        default={},
    )
    best_required_control_margin_row = max(
        rows,
        key=lambda row: finite(
            row.get("candidate_min_margin_vs_required_controls", row.get("candidate_margin_vs_random_same_mass")),
            -999.0,
        ),
        default={},
    )
    trace_all = bool(completed) and all(bool(item.get("trace_fidelity_pass", False)) for item in completed)
    stable_all = bool(completed) and all(bool(item.get("stable_anchor_proxy_pass", False)) for item in completed)
    good_all = bool(completed) and all(bool(item.get("good_safety_pass", False)) for item in completed)
    stage7_classification = str(stage7.get("classification", ""))
    stage7_full_no_go = stage7_classification == "MECHANISM_PASS_FULL_NO_GO"
    summary = {
        "stage": "TrackD_read_gauge_preserving_action_pilots",
        "status": "complete" if len(completed) == expected_pilot_count else "partial",
        "classification": (
            "READ_ACTION_MECHANISM_PASS_STAGE7_FULL_NO_GO"
            if any_gate_pass and stage7_full_no_go
            else
            "READ_ACTION_MECHANISM_PASS_PENDING_STAGE7"
            if any_gate_pass
            else "READ_ACTION_TRACE_PASS_CONTROL_SPECIFICITY_NO_GO"
        ),
        "carrier_prereq_gate_pass": bool(carrier.get("carrier_localization_gate_pass", False)),
        "gate_pass": any_gate_pass,
        "method_success": False,
        "mechanism_success": bool(any_gate_pass),
        "full_method_success": False,
        "runtime_action_allowed": False,
        "pilot_count": expected_pilot_count,
        "completed_pilot_count": len(completed),
        "pilot_roots": {
            **{name: str(path) for name, _, _, path in PILOTS},
            **{name: str(path) for name, path, _candidate in L07_PILOTS},
            **{name: str(path) for name, _layer, _head, _query_region, path, _candidate, _note in CARRIER_L07_PILOTS},
            **{name: str(path) for name, _layer, _head, _query_region, path, _candidate, _note in QKPAIR_KEYSTAB_PILOTS},
        },
        "pilots": pilot_summaries,
        "trace_fidelity_pass_all_completed": trace_all,
        "stable_anchor_proxy_pass_all_completed": stable_all,
        "good_safety_pass_all_completed": good_all,
        "bad_metric_pass_any": any_bad_metric_pass,
        "best_bad_improvement_vs_baseline": finite(best_bad_improvement_row.get("bad_improvement_vs_baseline"), 0.0),
        "best_bad_improvement_pilot": best_bad_improvement_row.get("pilot", ""),
        "best_bad_improvement_metric": best_bad_improvement_row.get("metric", ""),
        "best_candidate_margin_vs_random_same_mass": finite(best_control_margin_row.get("candidate_margin_vs_random_same_mass"), 0.0),
        "best_control_margin_pilot": best_control_margin_row.get("pilot", ""),
        "best_control_margin_metric": best_control_margin_row.get("metric", ""),
        "best_candidate_min_margin_vs_required_controls": finite(
            best_required_control_margin_row.get(
                "candidate_min_margin_vs_required_controls",
                best_required_control_margin_row.get("candidate_margin_vs_random_same_mass"),
            ),
            0.0,
        ),
        "best_required_control_margin_pilot": best_required_control_margin_row.get("pilot", ""),
        "best_required_control_margin_metric": best_required_control_margin_row.get("metric", ""),
        "gate_rule": (
            "A per-head action pilot must pass bad READ_LOCAL >=5% improvement, beat controls >=5%, "
            "good-control safety, trace-fidelity, and stable-anchor gates before runtime or Stage7."
        ),
        "failure_reason": (
            "" if any_gate_pass else
            "Per-head DG-Q90 carrier diagnostic passed and gauge-normalized L07 actions produced valid frame-bias "
            "traces with good-control safety, but none of the tested action bodies passed the full short-window "
            "mechanism gate. DG-Q90 variants did not move READ_LOCAL geometry enough; confidence-coupled L07 "
            "T030/T045/T050 improved scale_cv but failed required-control specificity, and carrier-scoped L07 "
            "variants over the sampled raw-QK layer/head/query rows did not reach a >=5% bad-metric or required-control "
            "margin. The source-target QK-pair key-stability variants also failed to move attention mass and bad metrics "
            "enough to pass."
        ),
        "stage7_context": {
            "status": stage7.get("status", ""),
            "classification": stage7_classification,
            "gate_pass": bool(stage7.get("gate_pass", False)),
            "candidate_count": stage7.get("candidate_count"),
            "best_delta_aligned_ate_rmse_m": stage7.get("best_delta_aligned_ate_rmse_m"),
            "best_delta_final_error_m": stage7.get("best_delta_final_error_m"),
            "note": (
                "An earlier confidence-neutral L07 pilot passed the short-window mechanism gate but failed Stage7 "
                "full validation; newly added carrier-scoped pilots did not add a new mechanism-pass candidate."
            ),
        },
        "mechanism_pass_interpretation": (
            "At least one confidence-neutral L07 pilot passed the short-window mechanism gate; this is not a full method "
            "success and only permits Stage7 full validation."
            if any_gate_pass
            else ""
        ),
    }
    failure_rows: list[dict[str, Any]] = []
    visual_rows: list[dict[str, Any]] = []
    for pilot_name, layer, head, root in PILOTS:
        summary_i = pilot_summaries.get(pilot_name, {})
        metric_rows = [row for row in rows if row.get("pilot") == pilot_name]
        best_bad_row = max(
            metric_rows,
            key=lambda row: finite(row.get("bad_improvement_vs_baseline"), -999.0),
            default={},
        )
        best_margin_row = max(
            metric_rows,
            key=lambda row: finite(row.get("candidate_margin_vs_random_same_mass"), -999.0),
            default={},
        )
        failure_rows.append({
            "pilot": pilot_name,
            "layer": layer,
            "head": head,
            "pilot_root": str(root),
            "status": summary_i.get("status", "missing"),
            "gate_pass": bool(summary_i.get("gate_pass", False)),
            "carrier_prereq_gate_pass": bool(carrier.get("carrier_localization_gate_pass", False)),
            "trace_fidelity_pass": bool(summary_i.get("trace_fidelity_pass", False)),
            "stable_anchor_proxy_pass": bool(summary_i.get("stable_anchor_proxy_pass", False)),
            "good_safety_pass": bool(summary_i.get("good_safety_pass", False)),
            "bad_metric_pass": any(bool(row.get("bad_metric_gate_pass", False)) for row in metric_rows),
            "best_bad_improvement_vs_baseline": finite(best_bad_row.get("bad_improvement_vs_baseline"), 0.0),
            "best_bad_improvement_metric": best_bad_row.get("metric", ""),
            "best_candidate_margin_vs_random_same_mass": finite(
                best_margin_row.get("candidate_margin_vs_random_same_mass"),
                0.0,
            ),
            "best_margin_metric": best_margin_row.get("metric", ""),
            "failure_attribution": (
                "not_run_or_missing"
                if summary_i.get("status") != "complete"
                else "bad_metric_and_control_margin_below_gate"
                if not any(bool(row.get("bad_metric_gate_pass", False)) for row in metric_rows)
                else ""
            ),
        })
    for pilot_name, root, candidate in L07_PILOTS:
        summary_i = pilot_summaries.get(pilot_name, {})
        metric_rows = [row for row in rows if row.get("pilot") == pilot_name]
        best_bad_row = max(
            metric_rows,
            key=lambda row: finite(row.get("bad_improvement_vs_baseline"), -999.0),
            default={},
        )
        best_margin_row = max(
            metric_rows,
            key=lambda row: finite(
                row.get("candidate_min_margin_vs_required_controls", row.get("candidate_margin_vs_random_same_mass")),
                -999.0,
            ),
            default={},
        )
        failure_rows.append({
            "pilot": pilot_name,
            "layer": "",
            "head": "",
            "pilot_root": str(root),
            "candidate": candidate,
            "status": summary_i.get("status", "missing"),
            "gate_pass": bool(summary_i.get("gate_pass", False)),
            "carrier_prereq_gate_pass": bool(carrier.get("carrier_localization_gate_pass", False)),
            "trace_fidelity_pass": bool(summary_i.get("trace_fidelity_pass", False)),
            "stable_anchor_proxy_pass": bool(summary_i.get("stable_anchor_proxy_pass", False)),
            "global_safety_proxy_pass": bool(summary_i.get("global_safety_proxy_pass", False)),
            "good_safety_pass": bool(summary_i.get("good_safety_pass", False)),
            "bad_metric_pass": any(bool(row.get("bad_metric_gate_pass", False)) for row in metric_rows),
            "best_bad_improvement_vs_baseline": finite(best_bad_row.get("bad_improvement_vs_baseline"), 0.0),
            "best_bad_improvement_metric": best_bad_row.get("metric", ""),
            "best_candidate_margin_vs_random_same_mass": finite(
                best_bad_row.get("candidate_margin_vs_random_same_mass"),
                0.0,
            ),
            "best_candidate_min_margin_vs_required_controls": finite(
                best_margin_row.get("candidate_min_margin_vs_required_controls"),
                0.0,
            ),
            "best_margin_metric": best_margin_row.get("metric", ""),
            "failure_attribution": (
                "not_run_or_missing"
                if summary_i.get("status") != "complete"
                else "required_control_specificity_below_gate"
                if not any(bool(row.get("bad_metric_gate_pass", False)) for row in metric_rows)
                else ""
            ),
        })
    for pilot_name, layer, head, query_region, root, candidate, note in CARRIER_L07_PILOTS:
        summary_i = pilot_summaries.get(pilot_name, {})
        metric_rows = [row for row in rows if row.get("pilot") == pilot_name]
        best_bad_row = max(
            metric_rows,
            key=lambda row: finite(row.get("bad_improvement_vs_baseline"), -999.0),
            default={},
        )
        best_margin_row = max(
            metric_rows,
            key=lambda row: finite(
                row.get("candidate_min_margin_vs_required_controls", row.get("candidate_margin_vs_random_same_mass")),
                -999.0,
            ),
            default={},
        )
        failure_rows.append({
            "pilot": pilot_name,
            "pilot_family": "carrier_scoped_l07",
            "layer": layer,
            "head": head,
            "frame_bias_query_region": query_region,
            "pilot_root": str(root),
            "candidate": candidate,
            "status": summary_i.get("status", "missing"),
            "gate_pass": bool(summary_i.get("gate_pass", False)),
            "carrier_prereq_gate_pass": bool(carrier.get("carrier_localization_gate_pass", False)),
            "trace_fidelity_pass": bool(summary_i.get("trace_fidelity_pass", False)),
            "stable_anchor_proxy_pass": bool(summary_i.get("stable_anchor_proxy_pass", False)),
            "global_safety_proxy_pass": bool(summary_i.get("global_safety_proxy_pass", False)),
            "good_safety_pass": bool(summary_i.get("good_safety_pass", False)),
            "bad_metric_pass": any(bool(row.get("bad_metric_gate_pass", False)) for row in metric_rows),
            "best_bad_improvement_vs_baseline": finite(best_bad_row.get("bad_improvement_vs_baseline"), 0.0),
            "best_bad_improvement_metric": best_bad_row.get("metric", ""),
            "best_candidate_margin_vs_random_same_mass": finite(
                best_bad_row.get("candidate_margin_vs_random_same_mass"),
                0.0,
            ),
            "best_candidate_min_margin_vs_required_controls": finite(
                best_margin_row.get("candidate_min_margin_vs_required_controls"),
                0.0,
            ),
            "best_margin_metric": best_margin_row.get("metric", ""),
            "failure_attribution": (
                "not_run_or_missing"
                if summary_i.get("status") != "complete"
                else "raw_qk_carrier_scoped_l07_below_mechanism_gate"
                if not any(bool(row.get("bad_metric_gate_pass", False)) for row in metric_rows)
                else ""
            ),
            "carrier_scope_note": note,
        })
    for pilot_name, layer, head, query_region, root, candidate, note in QKPAIR_KEYSTAB_PILOTS:
        summary_i = pilot_summaries.get(pilot_name, {})
        metric_rows = [row for row in rows if row.get("pilot") == pilot_name]
        best_bad_row = max(
            metric_rows,
            key=lambda row: finite(row.get("bad_improvement_vs_baseline"), -999.0),
            default={},
        )
        best_margin_row = max(
            metric_rows,
            key=lambda row: finite(
                row.get("candidate_min_margin_vs_required_controls", row.get("candidate_margin_vs_random_same_mass")),
                -999.0,
            ),
            default={},
        )
        failure_rows.append({
            "pilot": pilot_name,
            "pilot_family": "qkpair_keystab",
            "layer": layer,
            "head": head,
            "frame_bias_query_region": query_region,
            "pilot_root": str(root),
            "candidate": candidate,
            "status": summary_i.get("status", "missing"),
            "gate_pass": bool(summary_i.get("gate_pass", False)),
            "carrier_prereq_gate_pass": bool(carrier.get("carrier_localization_gate_pass", False)),
            "trace_fidelity_pass": bool(summary_i.get("trace_fidelity_pass", False)),
            "stable_anchor_proxy_pass": bool(summary_i.get("stable_anchor_proxy_pass", False)),
            "global_safety_proxy_pass": bool(summary_i.get("global_safety_proxy_pass", False)),
            "good_safety_pass": bool(summary_i.get("good_safety_pass", False)),
            "bad_metric_pass": any(bool(row.get("bad_metric_gate_pass", False)) for row in metric_rows),
            "best_bad_improvement_vs_baseline": finite(best_bad_row.get("bad_improvement_vs_baseline"), 0.0),
            "best_bad_improvement_metric": best_bad_row.get("metric", ""),
            "best_candidate_margin_vs_random_same_mass": finite(
                best_bad_row.get("candidate_margin_vs_random_same_mass"),
                0.0,
            ),
            "best_candidate_min_margin_vs_required_controls": finite(
                best_margin_row.get("candidate_min_margin_vs_required_controls"),
                0.0,
            ),
            "best_margin_metric": best_margin_row.get("metric", ""),
            "failure_attribution": (
                "not_run_or_missing"
                if summary_i.get("status") != "complete"
                else "qkpair_keystab_below_mechanism_gate"
                if not any(bool(row.get("bad_metric_gate_pass", False)) for row in metric_rows)
                else ""
            ),
            "carrier_scope_note": note,
        })
    write_csv(OUT_DIR / "rows.csv", rows)
    write_csv(
        OUT_DIR / "gate_checks.csv",
        [
            {"gate": "carrier_prereq_gate_pass", "pass": summary["carrier_prereq_gate_pass"], "value": summary["carrier_prereq_gate_pass"]},
            {"gate": "trace_fidelity_pass_all_completed", "pass": trace_all, "value": trace_all},
            {"gate": "stable_anchor_proxy_pass_all_completed", "pass": stable_all, "value": stable_all},
            {"gate": "good_safety_pass_all_completed", "pass": good_all, "value": good_all},
            {"gate": "bad_metric_pass_any", "pass": any_bad_metric_pass, "value": summary["best_bad_improvement_vs_baseline"]},
            {"gate": "best_required_control_margin", "pass": summary["best_candidate_min_margin_vs_required_controls"] >= 0.05, "value": summary["best_candidate_min_margin_vs_required_controls"]},
            {"gate": "per_head_action_mechanism_gate_pass", "pass": any_gate_pass, "value": any_gate_pass},
            {"gate": "runtime_action_allowed", "pass": False, "value": False},
        ],
    )
    write_csv(OUT_DIR / "failure_attribution.csv", failure_rows)
    panel_path = OUT_DIR / "visual_panels" / "per_head_action_gate_margins.svg"
    write_gate_margin_svg(panel_path, failure_rows)
    visual_rows.append({
        "visual_id": "per_head_action_gate_margins",
        "case_id": "TrackD_summary",
        "source_path": str(panel_path),
        "exists": panel_path.is_file(),
        "source_version": "v96tf",
        "note": "Bad-case improvement and same-mass margin compared with the 0.05 mechanism threshold.",
    })
    write_json(OUT_DIR / "summary.json", summary)
    write_text(
        OUT_DIR / "failure_report.md",
        f"""# Track D READ Action Pilot Report

Gate pass: `{any_gate_pass}`.

Carrier prereq passed: `{summary['carrier_prereq_gate_pass']}`.

Best bad improvement vs baseline: `{summary['best_bad_improvement_vs_baseline']}` from `{summary['best_bad_improvement_pilot']}` / `{summary['best_bad_improvement_metric']}`.

Best margin vs same-mass random: `{summary['best_candidate_margin_vs_random_same_mass']}` from `{summary['best_control_margin_pilot']}` / `{summary['best_control_margin_metric']}`.

Mechanism-pass interpretation: `{summary['mechanism_pass_interpretation'] or 'No short-window mechanism-pass action exists in the tested set.'}`.

Runtime action allowed: `{summary['runtime_action_allowed']}`.

Trace fidelity, stable-anchor proxy, and good-control safety are recorded separately from full-method success. The earlier mechanism-pass confidence-neutral L07 pilot failed Stage7 full validation; carrier-scoped raw-QK L07 and QK-pair key-stability follow-ups are recorded here as additional No-Go evidence, not runtime candidates.
""",
    )
    write_text(
        OUT_DIR / "what_would_have_to_be_true_to_pass.md",
        "# What Would Have To Be True To Pass\n\nA READ action must improve bad READ_LOCAL L1/L2/scale metrics by >=5%, beat all required controls by >=5%, keep good-control worsen <=2%, preserve stable/gauge safety evidence, and then pass Stage7 full validation. Confidence-neutral L07 pilots can satisfy the mechanism gate, but they still need full-sequence no-worse or improved ATE, no final-error regression, and no rolling-window regression before runtime promotion.",
    )
    write_text(
        OUT_DIR / "next_route_recommendation.md",
        "# Next Route Recommendation\n\n"
        "Do not continue small sweeps of the same DG-Q90 per-head source-bias family, the same confidence-coupled L07 energy-cap family, the sampled carrier-scoped L07 dense frame-bias body, or the sampled QK-pair key-stability body. The carrier diagnostic passes and trace/good/gauge gates are partially useful, but the newly tested raw-QK layer/head/query-scoped L07 and QK-pair variants still do not reach mechanism gate and the earlier mechanism-pass L07 branch failed Stage7 full validation.\n\n"
        "Recommended next route is either a different source-target actuator with verified nonzero trace mass and stronger pair support, or the plan's diagnostic branches: raw SWA transport trace before any SWA action, and TTT write eligibility diagnosis before any TTT runtime action. No runtime READ action is currently allowed.",
    )
    write_csv(OUT_DIR / "visual_manifest.csv", visual_rows)
    return summary


def main() -> None:
    summary = analyze()
    print(json.dumps({k: summary[k] for k in ("status", "classification", "gate_pass", "runtime_action_allowed")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
