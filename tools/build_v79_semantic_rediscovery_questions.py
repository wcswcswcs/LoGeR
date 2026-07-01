#!/usr/bin/env python3
"""Build ACL2 v79 Phase7 semantic rediscovery questions.

This is intentionally report-only: it reads real v79 Phase2/3/4/5 decisions and
creates the mandatory rediscovery handoff files without generating fake visual
panels or fabricated alignment scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_REPORT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v79tf_revised_semantic_three_memory_control/report_final"
)
DEFAULT_OUT_DIR = DEFAULT_REPORT_ROOT / "phase7_semantic_pca_qkv_ttt_rediscovery"

QUESTION_FIELDS = [
    "failed_phase",
    "failed_candidate",
    "failure_reason",
    "memory_body",
    "old_tap_or_action",
    "what_visual_evidence_was_missing",
    "new_visual_question",
    "new_tap_or_hook_to_dump",
    "new_overlay_required",
    "new_candidate_hypothesis",
    "priority",
]

REVIEW_FIELDS = [
    "artifact_group",
    "path",
    "status",
    "review_note",
]


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {}
            for field in fields:
                value = row.get(field, "")
                clean[field] = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value
            writer.writerow(clean)


def _phase2_summary(root: Path) -> Dict[str, Any]:
    path = root / "phase2_semantic_read_global_control/phase2_semantic_read_gate_summary.json"
    payload = _load_json(path) or {}
    decisions = payload.get("decisions") if isinstance(payload, dict) else {}
    read1 = decisions.get("READ1_L07_SEMANTIC_LAYOUT_SELECT", {}) if isinstance(decisions, dict) else {}
    return {
        "path": str(path),
        "available": path.exists(),
        "phase2_gate_pass": bool(payload.get("phase2_gate_pass")),
        "read1_phase2_gate_pass": bool(read1.get("phase2_gate_pass")),
        "read1_metric_passes": read1.get("mechanism_metric_passes") or read1.get("mechanism_passes") or [],
        "read1_comparisons": read1.get("comparisons", {}),
    }


def _phase3_summary(root: Path) -> Dict[str, Any]:
    base = root / "phase3_semantic_swa_handoff"
    paths = [
        base / "source_side_phase9_subset_chunk09/phase9_swa_cache_value_decision.json",
        base / "source_side_phase9_subset_chunk09_alllayers_strong035/phase9_swa_cache_value_decision.json",
    ]
    payloads = [_load_json(path) or {} for path in paths]
    return {
        "paths": [str(path) for path in paths],
        "available": [path.exists() for path in paths],
        "any_gate_pass": any(bool(p.get("phase9_any_gate_pass")) for p in payloads if isinstance(p, dict)),
        "payloads": payloads,
    }


def _phase4_summary(root: Path) -> Dict[str, Any]:
    path = root / "phase4_semantic_ttt_write_update/fivechunk_7_11_role_control/phase4_ttt_write_role_decision.json"
    payload = _load_json(path) or {}
    return {
        "path": str(path),
        "available": path.exists(),
        "phase4_any_gate_pass": bool(payload.get("phase4_any_gate_pass")),
        "decisions": payload.get("decisions", {}),
    }


def _phase5_summary(root: Path) -> Dict[str, Any]:
    path = root / "phase5_cross_memory_semantic_handshake/read_to_ttt_fivechunk_7_11/phase5_cross_memory_handshake_decision.json"
    payload = _load_json(path) or {}
    return {
        "path": str(path),
        "available": path.exists(),
        "phase5_any_gate_pass": bool(payload.get("phase5_any_gate_pass")),
        "decisions": payload.get("decisions", {}),
    }


def _latest_readcuefix_hs23_summary(root: Path) -> Dict[str, Any]:
    base = root / "phase7_semantic_pca_qkv_ttt_rediscovery/sourceboost_qkattn_midtailq_hs23_hs24_readcuepriorfix_smoke_2045"
    csv_path = base / "phase5_cross_memory_handshake_metrics.csv"
    decision_path = base / "phase5_cross_memory_handshake_decision.json"
    rows = {row.get("run", ""): row for row in _load_csv_rows(csv_path)}
    metric_keys = [
        "local_sim3_ate_rmse_m",
        "head_tail_pose_sim3_rmse_m",
        "overlap3_to_future_pose_sim3_rmse_m",
        "scale_cv_head_mid_tail_pose_sim3",
        "segment_ate_rmse_m",
    ]
    case_names = [
        "HS1_READ_ONLY_BEST_SEM",
        "HS8_GEOMETRY_ONLY_HANDSHAKE",
        "HS9_RANDOM_ROLE_HANDSHAKE",
        "HS23_SWA_STABLE_TOP25_QKATTN_SOURCEBOOST_MIDTAILQ_TTT_POS_SEM",
        "HS24_RANDOM_SWA_STABLE_TOP25_QKATTN_SOURCEBOOST_MIDTAILQ_TTT_POS_SEM",
    ]
    cases: Dict[str, Dict[str, float | None]] = {}
    for case in case_names:
        row = rows.get(case, {})
        cases[case] = {key: _as_float(row.get(key)) for key in metric_keys}
    hs23 = cases.get("HS23_SWA_STABLE_TOP25_QKATTN_SOURCEBOOST_MIDTAILQ_TTT_POS_SEM", {})
    control_names = [name for name in case_names if name.startswith(("HS1_", "HS8_", "HS9_", "HS24_"))]
    beats_all_controls = False
    if hs23 and control_names:
        beats_all_controls = all(
            all(
                hs23.get(key) is not None
                and cases[control].get(key) is not None
                and hs23[key] < cases[control][key]  # type: ignore[index]
                for key in metric_keys
            )
            for control in control_names
        )
    decision = _load_json(decision_path) or {}
    return {
        "base": str(base),
        "metrics_csv": str(csv_path),
        "decision_json": str(decision_path),
        "available": csv_path.exists(),
        "phase5_any_gate_pass": bool(decision.get("phase5_any_gate_pass")),
        "hs23_beats_all_listed_controls": beats_all_controls,
        "cases": cases,
    }


def _latest_pair910_carrier_summary(root: Path) -> Dict[str, Any]:
    overlap_path = root / (
        "phase7_semantic_pca_qkv_ttt_rediscovery/"
        "pair9_10_read_swa_carrier_audit_2104/read_swa_overlap_alignment_summary.json"
    )
    corridor_path = root / (
        "phase7_semantic_pca_qkv_ttt_rediscovery/"
        "pair9_10_read_redirection_corridor_2104/read_redirection_corridor_summary.json"
    )
    overlap = _load_json(overlap_path) or {}
    corridor = _load_json(corridor_path) or {}
    return {
        "overlap_summary": str(overlap_path),
        "corridor_summary": str(corridor_path),
        "available": overlap_path.exists() and corridor_path.exists(),
        "carrier_viable_for_smoke": bool(overlap.get("carrier_viable_for_smoke")),
        "redirection_viable_for_smoke": bool(corridor.get("redirection_viable_for_smoke")),
        "candidate_count": overlap.get("candidate_count"),
        "random_control_count": overlap.get("random_control_count"),
        "corridor_candidate_count": corridor.get("candidate_count"),
        "corridor_random_candidate_count": corridor.get("random_candidate_count"),
        "corridor_candidate_better_than_random": bool(corridor.get("candidate_better_than_random")),
        "corridor_repeat_ok": bool(corridor.get("repeat_ok")),
    }


def _latest_ttt_regime_summary(root: Path) -> Dict[str, Any]:
    base = root / "phase4_ttt_long_window_regime_action_smoke_readcuefix_2108"
    csv_path = base / "long_window_ttt_regime_action_metrics.csv"
    decision_path = base / "long_window_ttt_regime_action_decision.json"
    rows = {row.get("case", ""): row for row in _load_csv_rows(csv_path)}
    case_names = [
        "LW0_READPATH_NATIVE",
        "LW1_TTT_SEMANTIC_BASE",
        "LW11_TTT_STATE_ENERGY_DIR_B0_MIN075",
        "LW13_TTT_TAIL_STATE_SELECTIVE_B0_MIN075",
        "LW14_TTT_TAIL_SOFT_DIR_B0_MIN075",
        "LW12_TTT_OVERLAP_DYNAMIC_VETO_B0_BLEND050",
    ]
    cases = {
        case: {
            "window5_joint_sim3_rmse_m": _as_float(rows.get(case, {}).get("window5_joint_sim3_rmse_m")),
            "window5_subchunk_scale_cv": _as_float(rows.get(case, {}).get("window5_subchunk_scale_cv")),
            "ttt_write_commit_filter_applied_true_count": _as_float(
                rows.get(case, {}).get("ttt_write_commit_filter_applied_true_count")
            ),
            "ttt_write_commit_filter_activation_rate_mean": _as_float(
                rows.get(case, {}).get("ttt_write_commit_filter_activation_rate_mean")
            ),
        }
        for case in case_names
    }
    decision = _load_json(decision_path) or {}
    return {
        "base": str(base),
        "metrics_csv": str(csv_path),
        "decision_json": str(decision_path),
        "available": csv_path.exists(),
        "any_single_window_improves_ge_min_ratio": bool(decision.get("any_single_window_improves_ge_min_ratio")),
        "any_single_window_improves_vs_ttt_baseline": bool(decision.get("any_single_window_improves_vs_ttt_baseline")),
        "runtime_promotion_allowed": bool(decision.get("runtime_promotion_allowed")),
        "cases": cases,
    }


def _question_rows(summary: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    rows = [
        {
            "failed_phase": "phase2_to_long_horizon",
            "failed_candidate": "READ1_L07_SEMANTIC_LAYOUT_SELECT",
            "failure_reason": "READ1 passed the short-memory scale_cv/local mechanism gate on chunk10, but did not transfer to five-window READ->TTT handshake.",
            "memory_body": "short_term_read/global_attention",
            "old_tap_or_action": "Global-K L07 read cue v78.l07_l13.l07_action_only",
            "what_visual_evidence_was_missing": "direct READ stable/harm/context role mask intersected with head-tail/scale failure regions and later TTT positive/negative roles",
            "new_visual_question": "Which L07 READ-selected tokens are actually stable geometry support versus harmful/context tokens on the chunk7-11 long window?",
            "new_tap_or_hook_to_dump": "READ_TTT_ROLE_ALIGNMENT_LOG plus READ active mask panels for L07 Global-K",
            "new_overlay_required": "RGB;semantic;confidence;D_geo;head_tail;scale_cv;READ_active;TTT_positive;TTT_negative;same_mass_random",
            "new_candidate_hypothesis": "HYP-V79-REDISC-001_READ_ROLE_INTERSECTION",
            "priority": "P0",
        },
        {
            "failed_phase": "phase3_swa",
            "failed_candidate": "P9_10_SOURCE_GATE_DISAGREEMENT_K_LAST and all-layer strong035",
            "failure_reason": "SWA K/V action fidelity produced tiny or control-matched gains; route-random/same-mass controls often matched or beat semantic action.",
            "memory_body": "mid_term_swa/overlap_handoff",
            "old_tap_or_action": "SWA source gate/replace on K/V, last/all layers",
            "what_visual_evidence_was_missing": "direct current-head/cache-tail K/V alignment map with semantic role composition and overlap residual/future failure overlay",
            "new_visual_question": "Is the useful SWA carrier a K/V alignment boundary, an overlap merge residual, or neither?",
            "new_tap_or_hook_to_dump": "READ_SWA_ROLE_ALIGNMENT_LOG; compact current/cache K/V cosine maps; overlap residual boundary panels",
            "new_overlay_required": "RGB;semantic;confidence;overlap_residual;future_after_overlap;current_K;cache_K;current_V;cache_V;semantic_route;random_route",
            "new_candidate_hypothesis": "HYP-V79-REDISC-002_DIRECT_KV_ALIGNMENT",
            "priority": "P0",
        },
        {
            "failed_phase": "phase4_ttt",
            "failed_candidate": "T1/T2/T3 semantic TTT write role",
            "failure_reason": "TTT write mass, post-zp delta, and next state hash changed, but five-window head-tail/future/scale metrics did not improve and often worsened.",
            "memory_body": "long_term_ttt/write_update",
            "old_tap_or_action": "fine_ttt_lowstuff_highd_short write role with positive=1.25 negative=0.70",
            "what_visual_evidence_was_missing": "branch/layer update carrier showing whether post-zp delta touches the geometry carrier or only broad low-observability/context mass",
            "new_visual_question": "Where does TTT post-zp/action delta land relative to long-window corridor/exposure/road-edge regime shifts?",
            "new_tap_or_hook_to_dump": "TTT_BRANCH_LAYER_UPDATE_GATE with per-branch role mass and post_zp_delta overlays",
            "new_overlay_required": "RGB;semantic;confidence;regime_shift;road_edge;shadow_exposure;post_zp_delta;branch_layer_update;future_error",
            "new_candidate_hypothesis": "HYP-V79-REDISC-003_TTT_GEOMETRY_CARRIER",
            "priority": "P0",
        },
        {
            "failed_phase": "phase5_read_to_ttt",
            "failed_candidate": "HS5_READ_TO_TTT_SEM",
            "failure_reason": "READ->TTT combo worsened all mechanism metrics versus baseline and failed to beat best single path, geometry-only, and random-role controls.",
            "memory_body": "cross_memory_read_to_ttt",
            "old_tap_or_action": "READ1 L07 semantic read plus TTT fine role write",
            "what_visual_evidence_was_missing": "direct READ active role intersect TTT positive/negative role; current artifacts only expose marginal read output and marginal TTT role counts",
            "new_visual_question": "Does READ1 select evidence that TTT classifies oppositely, causing destructive write persistence?",
            "new_tap_or_hook_to_dump": "READ_TTT_ROLE_ALIGNMENT_LOG with per-token intersection counts and conflict map",
            "new_overlay_required": "READ_stable;READ_harm;TTT_positive;TTT_negative;conflict;trajectory_error;same_mass_random",
            "new_candidate_hypothesis": "HYP-V79-REDISC-004_READ_TTT_CONFLICT",
            "priority": "P0",
        },
        {
            "failed_phase": "phase1_long_target_visual_pattern",
            "failed_candidate": "five-window chunks7-11",
            "failure_reason": "Long-window bad case pattern is dominated by regime changes rather than one semantic category: shadow/exposure shifts, narrow corridor curvature, road cracks, and wall/road boundary drift.",
            "memory_body": "long_term_regime_shift",
            "old_tap_or_action": "five-window semantic write target mining",
            "what_visual_evidence_was_missing": "bad-vs-success visual contrast tied to regime shift and geometry carrier, not just label class counts",
            "new_visual_question": "Which stable-looking urban/corridor tokens remain geometrically stable across the five-window regime change?",
            "new_tap_or_hook_to_dump": "long_window_regime_shift_panel plus stable corridor continuity and downstream future overlays",
            "new_overlay_required": "bad_case;successful_reference;shadow_exposure;road_edge_continuity;wall_boundary;window5_error;TTT_delta",
            "new_candidate_hypothesis": "HYP-V79-REDISC-005_REGIME_SHIFT_OBSERVABILITY",
            "priority": "P1",
        },
        {
            "failed_phase": "phase3_merge_boundary",
            "failed_candidate": "SWA overlap handoff controls",
            "failure_reason": "SWA overlap/future metrics may be merge-boundary/gauge dominated; semantic route edits alone did not move the boundary enough.",
            "memory_body": "mid_term_merge_boundary",
            "old_tap_or_action": "SWA source route gate/replace",
            "what_visual_evidence_was_missing": "merge-boundary residual and adjacent-chunk raw overlap residual before/after semantic route action",
            "new_visual_question": "Is the adjacent chunks8-9 failure really a semantic handoff problem or a merge/gauge boundary problem?",
            "new_tap_or_hook_to_dump": "new_merge_boundary_visual_panels with raw overlap residual and Sim3 boundary jump",
            "new_overlay_required": "prev_tail;curr_head;raw_overlap_residual;boundary_jump;semantic_route;same_mass_random;future_after_overlap",
            "new_candidate_hypothesis": "HYP-V79-REDISC-006_MERGE_BOUNDARY_FIRST",
            "priority": "P1",
        },
    ]
    latest = (summary or {}).get("latest", {})
    if latest.get("readcuefix_hs23_hs24", {}).get("available"):
        rows.append(
            {
                "failed_phase": "phase5_hs23_hs24_readcuefix",
                "failed_candidate": "HS23_SOURCEBOOST_QKATTN_MIDTAILQ_TTT_POS_SEM",
                "failure_reason": "After fixing inactive-chunk READ cue prior consumption, HS23 no longer suffered read-cue degradation, but still failed to beat READ-only, geometry-only, random-role, or same-mass random controls.",
                "memory_body": "cross_memory_swa_to_ttt",
                "old_tap_or_action": "SWA stable top25 QK-attention sourceboost plus mid-tail TTT positive semantic role",
                "what_visual_evidence_was_missing": "direct evidence that sourceboost-selected tokens are semantic stable carriers rather than same-mass/random-selected route mass",
                "new_visual_question": "Which HS23 selected source tokens are uniquely semantic and repeatedly aligned with lower overlap/future error after the read-cue prior fix?",
                "new_tap_or_hook_to_dump": "READ_SWA_ROLE_ALIGNMENT_LOG plus selected-source-vs-random token composition and QK attention panels",
                "new_overlay_required": "READ_active;SWA_sourceboost;QK_attention;semantic_role;random_same_mass;future_error;scale_cv",
                "new_candidate_hypothesis": "HYP-V79-REDISC-007_SOURCEBOOST_AFTER_READCUEFIX",
                "priority": "P0",
            }
        )
    if latest.get("pair9_10_carrier", {}).get("available"):
        rows.append(
            {
                "failed_phase": "phase7_pair9_10_carrier_audit",
                "failed_candidate": "pair9_10_READ_SWA_carrier_and_redirection_corridor",
                "failure_reason": "READ/SWA carrier overlap had only 3 candidate tokens versus 43 random-control tokens, and the redirection corridor did not beat random or satisfy repeat_ok.",
                "memory_body": "mid_term_swa/overlap_handoff",
                "old_tap_or_action": "source-gate disagreement K-last and stable source-replace V-last around chunk9-10",
                "what_visual_evidence_was_missing": "group-stratified source composition and boundary residual evidence showing why random same-mass owns more viable carrier mass",
                "new_visual_question": "Is pair9-10 dominated by semantic false positives, boundary/gauge residual, or read cue selection that random controls cover better?",
                "new_tap_or_hook_to_dump": "group-stratified READ/SWA carrier table; merge-boundary residual panels; random-vs-semantic carrier composition",
                "new_overlay_required": "READ_q90;SWA_stable;SWA_disagreement;random_carrier;semantic_label;overlap_residual;boundary_jump",
                "new_candidate_hypothesis": "HYP-V79-REDISC-008_PAIR9_10_RANDOM_CARRIER_ADVANTAGE",
                "priority": "P0",
            }
        )
    if latest.get("ttt_regime_readcuefix", {}).get("available"):
        rows.append(
            {
                "failed_phase": "phase4_ttt_long_window_regime_readcuefix",
                "failed_candidate": "LW11/LW12/LW13/LW14 guarded TTT regime actions",
                "failure_reason": "State-energy and tail-risk guards changed TTT state or matched baseline, but no candidate improved over LW1 TTT semantic baseline; runtime promotion was false.",
                "memory_body": "long_term_ttt/write_update",
                "old_tap_or_action": "state-energy directional, tail-state selective, tail soft directional, and overlap dynamic-veto TTT controls",
                "what_visual_evidence_was_missing": "evidence that branch0 guarded write deltas land on a geometry carrier instead of broad low-observability tail/context mass",
                "new_visual_question": "Which branch/layer TTT updates are both non-random and geometrically causal across the chunk7-11 regime shift?",
                "new_tap_or_hook_to_dump": "TTT_BRANCH_LAYER_UPDATE_GATE with per-branch causal ablation and next-probe state hash delta",
                "new_overlay_required": "branch0_delta;post_zp_delta;tail_risk;semantic_role;regime_shift;window5_error;random_write_mass",
                "new_candidate_hypothesis": "HYP-V79-REDISC-009_TTT_GUARD_NO_GEOMETRY_CARRIER",
                "priority": "P0",
            }
        )
    return rows


def _hypothesis_bank(rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> str:
    latest = summary.get("latest", {})
    lines = [
        "# v79 Phase7 Semantic Rediscovery Hypothesis Bank",
        "",
        "Status: rediscovery_required_for_cross_memory_and_long_horizon.",
        "",
        "Evidence summary:",
        f"- Phase2 READ gate pass: `{summary['phase2'].get('phase2_gate_pass')}`.",
        f"- Phase3 SWA any gate pass: `{summary['phase3'].get('any_gate_pass')}`.",
        f"- Phase4 TTT any gate pass: `{summary['phase4'].get('phase4_any_gate_pass')}`.",
        f"- Phase5 READ->TTT any gate pass: `{summary['phase5'].get('phase5_any_gate_pass')}`.",
        f"- Latest HS23/HS24 read-cue-fix gate pass: `{latest.get('readcuefix_hs23_hs24', {}).get('phase5_any_gate_pass')}`.",
        f"- Latest pair9-10 carrier viable: `{latest.get('pair9_10_carrier', {}).get('carrier_viable_for_smoke')}`; redirection viable: `{latest.get('pair9_10_carrier', {}).get('redirection_viable_for_smoke')}`.",
        f"- Latest TTT regime runtime promotion allowed: `{latest.get('ttt_regime_readcuefix', {}).get('runtime_promotion_allowed')}`.",
        "",
        "Important boundary: no direct READ_TTT/READ_SWA role-intersection log exists in the current artifacts, so alignment scores are not claimed.",
        "",
        "Hypotheses:",
    ]
    for row in rows:
        lines.extend(
            [
                "",
                f"## {row['new_candidate_hypothesis']}",
                f"- Priority: `{row['priority']}`",
                f"- Memory body: `{row['memory_body']}`",
                f"- Triggered by: `{row['failed_phase']} / {row['failed_candidate']}`",
                f"- Failure reason: {row['failure_reason']}",
                f"- New visual question: {row['new_visual_question']}",
                f"- Required hook/tap: `{row['new_tap_or_hook_to_dump']}`",
                f"- Required overlays: `{row['new_overlay_required']}`",
                "- Stop rule: do not promote unless the new hook shows direct role/action alignment and the candidate beats same-mass/random/geometry controls on the relevant memory metric.",
            ]
        )
    return "\n".join(lines) + "\n"


def _visual_insight(summary: Dict[str, Any]) -> str:
    latest = summary.get("latest", {})
    pair = latest.get("pair9_10_carrier", {})
    ttt = latest.get("ttt_regime_readcuefix", {})
    return "\n".join(
        [
            "# v79 Phase7 Visual Insight",
            "",
            "No new visual panels were generated in this builder pass. This file records the rediscovery target, not a confirmed visual clue.",
            "",
            "Current evidence says:",
            "- Short-term READ has a real weak positive mechanism signal on chunk10, especially scale_cv/local support.",
            "- SWA source K/V handoff and stronger all-layer amplification did not pass; controls match or beat the semantic route.",
            "- TTT semantic write changes state/post-zp delta but does not reach the long-window geometry carrier.",
            "- READ->TTT handshake worsens the five-window metrics and lacks a direct role-intersection hook.",
            f"- After inactive READ-cue-prior repair, HS23/HS24 sourceboost still did not produce a semantic win; phase5 gate pass is `{latest.get('readcuefix_hs23_hs24', {}).get('phase5_any_gate_pass')}`.",
            f"- Pair9-10 READ/SWA carrier audit is not viable: carrier count `{pair.get('candidate_count')}` vs random `{pair.get('random_control_count')}`, corridor repeat_ok `{pair.get('corridor_repeat_ok')}`.",
            f"- Latest TTT guarded-regime smoke has runtime promotion `{ttt.get('runtime_promotion_allowed')}` and no candidate improves over the TTT baseline.",
            "- The five-chunk bad case pattern remains important: continuous shadow/exposure shifts, narrow curved corridor, road cracks, and wall/road boundary changes; the next visual check should focus on long-window appearance/geometry regime shift rather than one object class.",
            "",
            "Therefore the next useful work is direct alignment instrumentation plus targeted visual panels, not scalar sweeps.",
            "",
            f"Phase2 summary source: `{summary['phase2'].get('path')}`",
            f"Phase5 summary source: `{summary['phase5'].get('path')}`",
            f"Latest HS23/HS24 read-cue-fix source: `{latest.get('readcuefix_hs23_hs24', {}).get('metrics_csv')}`",
            f"Latest pair9-10 carrier source: `{pair.get('overlap_summary')}`",
            f"Latest TTT regime source: `{ttt.get('metrics_csv')}`",
        ]
    ) + "\n"


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("new_qkv_visual_panels", "new_ttt_branch_visual_panels", "new_merge_boundary_visual_panels"):
        (out_dir / name).mkdir(parents=True, exist_ok=True)

    summary = {
        "schema": "acl2_v79_phase7_semantic_rediscovery_v1",
        "report_root": str(args.report_root),
        "out_dir": str(out_dir),
        "phase2": _phase2_summary(args.report_root),
        "phase3": _phase3_summary(args.report_root),
        "phase4": _phase4_summary(args.report_root),
        "phase5": _phase5_summary(args.report_root),
        "latest": {
            "readcuefix_hs23_hs24": _latest_readcuefix_hs23_summary(args.report_root),
            "pair9_10_carrier": _latest_pair910_carrier_summary(args.report_root),
            "ttt_regime_readcuefix": _latest_ttt_regime_summary(args.report_root),
        },
        "status": "rediscovery_required_for_cross_memory_and_long_horizon",
        "no_fake_visual_panels": True,
    }
    rows = _question_rows(summary)
    review_rows = [
        {
            "artifact_group": "new_qkv_visual_panels",
            "path": str(out_dir / "new_qkv_visual_panels"),
            "status": "not_generated",
            "review_note": "Questions created; visual panel generation still required.",
        },
        {
            "artifact_group": "new_ttt_branch_visual_panels",
            "path": str(out_dir / "new_ttt_branch_visual_panels"),
            "status": "not_generated",
            "review_note": "TTT branch/layer carrier panels still required.",
        },
        {
            "artifact_group": "new_merge_boundary_visual_panels",
            "path": str(out_dir / "new_merge_boundary_visual_panels"),
            "status": "not_generated",
            "review_note": "Merge-boundary panels still required.",
        },
    ]
    visual_audit = {
        "schema": "acl2_v79_phase7_visual_integrity_audit_v1",
        "gate_pass": False,
        "reason": "rediscovery_questions_only_no_new_visual_panels_generated",
        "no_fake_visuals": True,
        "required_panel_dirs_created": True,
        "visual_review_rows": len(review_rows),
    }

    _write_csv(out_dir / "failed_semantic_action_to_visual_question.csv", rows, QUESTION_FIELDS)
    _write_csv(out_dir / "visual_review.csv", review_rows, REVIEW_FIELDS)
    (out_dir / "new_semantic_memory_hypothesis_bank.md").write_text(_hypothesis_bank(rows, summary), encoding="utf-8")
    (out_dir / "visual_insight.md").write_text(_visual_insight(summary), encoding="utf-8")
    (out_dir / "visual_integrity_audit.json").write_text(json.dumps(visual_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "rediscovery_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"out_dir": str(out_dir), "questions": len(rows), "status": summary["status"], "visual_gate_pass": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
