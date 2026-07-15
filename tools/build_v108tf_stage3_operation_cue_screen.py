#!/usr/bin/env python3
"""Build ACL2 v108TF Stage3 operation x semantic cue pre-screen.

Stage3 ranks action-surface candidates for full KITTI pilots.  It is not a
geometry-success gate; every candidate that changes behavior still needs Stage4
full sequence metrics and action fidelity.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v108tf_lingbot_memory_operation_action_surface_search"
STAGE1 = RESULT_ROOT / "stage1_action_surface_contract"
STAGE2 = RESULT_ROOT / "stage2_semantic_cue_bank"
OUT = RESULT_ROOT / "stage3_operation_cue_screen"


SURFACE_DEFS = {
    "A": {
        "name": "anchor_scale_frame_initialization",
        "score": "Q_ref_sem_risk_strict",
        "direction": "low",
        "internal": "operation_row_count",
        "expected_action_scope": "force_non_keyframe_skip_or_snap_low_reference_quality_keyframes",
        "primary_operation": "initialization",
    },
    "B": {
        "name": "cache_append_write_control",
        "score": "Q_ref_sem_balanced",
        "direction": "low",
        "internal": "cache_append_count",
        "expected_action_scope": "cache_append_write_suppress_or_context_only_for_low_reference_quality_frames",
        "primary_operation": "cache_append",
    },
    "C": {
        "name": "retention_eviction_budget_keep_drop",
        "score": "Q_ref_sem_risk_strict",
        "direction": "low",
        "internal": "retention_eviction_count",
        "expected_action_scope": "semantic_eviction_or_budget_drop_candidate_requires_new_hook",
        "primary_operation": "retention_eviction_budget",
    },
    "D": {
        "name": "trajectory_memory_write_retention",
        "score": "semantic_continuity_score",
        "direction": "low",
        "internal": "trajectory_write_count",
        "expected_action_scope": "trajectory_write_gate_candidate_requires_new_hook",
        "primary_operation": "trajectory_write",
    },
    "E": {
        "name": "local_preserve_reference_trajectory_block",
        "score": "Q_ref_sem_risk_strict",
        "direction": "low",
        "internal": "local_reference_count",
        "expected_action_scope": "local_preserve_reference_block_guarded_by_semantic_risk",
        "primary_operation": "local_reference",
    },
    "F": {
        "name": "special_token_camera_register_trajectory_routing",
        "score": "Q_ref_sem_balanced",
        "direction": "low",
        "internal": "special_token_count",
        "expected_action_scope": "special_token_routing_context_only_or_anchor_only_for_low_reference_quality_frames",
        "primary_operation": "special_token",
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        if value == "":
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def quantile(values: list[float], q: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def fmt(value: Any) -> str:
    try:
        val = float(value)
    except Exception:
        return "" if value is None else str(value)
    if not math.isfinite(val):
        return ""
    return f"{val:.8g}"


def by_seq_counts(cases: list[dict[str, Any]], field: str | None = None) -> str:
    counts: Counter[str] = Counter()
    for case in cases:
        seq = str(case["seq_id"])
        counts[seq] += int(case.get(field, 1)) if field else 1
    return ";".join(f"{seq}:{counts[seq]}" for seq in sorted(counts))


def load_feasibility() -> dict[str, dict[str, str]]:
    return {row["surface_id"]: row for row in read_csv(STAGE1 / "action_surface_implementation_feasibility.csv")}


def build_surface_cases() -> dict[str, list[dict[str, Any]]]:
    frame_rows = read_csv(STAGE2 / "frame_semantic_summary.csv")
    frame_map = {(row["seq_id"], int(row["frame_id"])): row for row in frame_rows}
    op_rows = read_csv(STAGE2 / "operation_semantic_summary.csv")
    accum: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in op_rows:
        seq = row.get("seq_id", "")
        frame_id = int(float(row.get("current_frame") or row.get("semantic_join_frame_start") or -1))
        if frame_id < 0:
            continue
        frame = frame_map.get((seq, frame_id), {})
        for surface_id in [sid for sid in row.get("candidate_surface_ids", "").split(";") if sid]:
            key = (surface_id, seq, frame_id)
            case = accum.setdefault(
                key,
                {
                    "surface_id": surface_id,
                    "seq_id": seq,
                    "frame_id": frame_id,
                    "target_ids": set(),
                    "target_kinds": set(),
                    "operation_types": Counter(),
                    "token_types": Counter(),
                    "operation_row_count": 0,
                    "keyframe_count": 0,
                    "scale_frame_count": 0,
                    "cache_append_count": 0,
                    "retention_eviction_count": 0,
                    "trajectory_write_count": 0,
                    "local_reference_count": 0,
                    "special_token_count": 0,
                    "source_age_values": [],
                    "context_paths": Counter(),
                },
            )
            case["operation_row_count"] += 1
            case["target_ids"].add(row.get("target_id", ""))
            case["target_kinds"].add(row.get("target_kind", ""))
            op = row.get("operation_type", "")
            case["operation_types"][op] += 1
            token_type = row.get("token_type", "")
            case["token_types"][token_type] += 1
            case["keyframe_count"] += int(boolish(row.get("keyframe_flag", "")))
            case["scale_frame_count"] += int(boolish(row.get("scale_frame_flag", "")))
            case["cache_append_count"] += int(op == "cache_append")
            case["retention_eviction_count"] += int(op in {"retention", "eviction", "budget_keep", "budget_drop"})
            case["trajectory_write_count"] += int(op == "trajectory_write" or boolish(row.get("trajectory_memory_flag", "")))
            case["local_reference_count"] += int(row.get("context_path", "") == "local_pose_reference_window")
            case["special_token_count"] += int(op == "special_token_update" or token_type != "image_patch")
            case["source_age_values"].append(fnum(row.get("source_frame_age", "")))
            case["context_paths"][row.get("context_path", "")] += 1
            for col in [
                "stable_structure_mass",
                "dynamic_mass",
                "boundary_mass",
                "weak_context_mass",
                "road_ground_mass",
                "sky_lowobs_mass",
                "semantic_trust_mean",
                "semantic_purity_mean",
                "semantic_boundary_risk",
                "semantic_continuity_score",
                "Q_ref_sem_balanced",
                "Q_ref_sem_risk_strict",
                "Q_ref_sem_stable_strict",
            ]:
                case[col] = fnum(frame.get(col, ""))
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in accum.values():
        kinds = {k for k in case["target_kinds"] if k}
        case["diagnostic_label"] = "bad" if any(k.startswith("high_l3") for k in kinds) else "safe_good"
        case["target_ids"] = sorted(t for t in case["target_ids"] if t)
        case["target_kinds"] = sorted(k for k in kinds if k)
        case["operation_types"] = ";".join(f"{k}:{v}" for k, v in case["operation_types"].most_common())
        case["token_types"] = ";".join(f"{k}:{v}" for k, v in case["token_types"].most_common())
        case["context_paths"] = ";".join(f"{k}:{v}" for k, v in case["context_paths"].most_common())
        case["source_age_mean"] = mean(case.pop("source_age_values"))
        out[case["surface_id"]].append(case)
    for surface_id in out:
        out[surface_id].sort(key=lambda c: (c["seq_id"], c["frame_id"]))
    return out


def threshold_select(cases: list[dict[str, Any]], score_col: str, direction: str, q: float) -> tuple[list[dict[str, Any]], float]:
    values = [fnum(c.get(score_col, "")) for c in cases]
    threshold = quantile(values, q)
    if not math.isfinite(threshold):
        return [], threshold
    if direction == "low":
        selected = [c for c in cases if fnum(c.get(score_col, "")) <= threshold]
    else:
        selected = [c for c in cases if fnum(c.get(score_col, "")) >= threshold]
    return selected, threshold


def threshold_select_per_seq(cases: list[dict[str, Any]], score_col: str, direction: str, q: float) -> tuple[list[dict[str, Any]], dict[str, float]]:
    cases_by_seq: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        cases_by_seq[str(case["seq_id"])].append(case)
    selected: list[dict[str, Any]] = []
    thresholds: dict[str, float] = {}
    for seq, group in sorted(cases_by_seq.items()):
        group_selected, threshold = threshold_select(group, score_col, direction, q)
        thresholds[seq] = threshold
        selected.extend(group_selected)
    return sorted(selected, key=lambda c: (c["seq_id"], c["frame_id"])), thresholds


def threshold_for_case(threshold: float | dict[str, float], case: dict[str, Any]) -> float:
    if isinstance(threshold, dict):
        return threshold.get(str(case["seq_id"]), float("nan"))
    return threshold


def threshold_repr(threshold: Any) -> str:
    if isinstance(threshold, dict):
        return ";".join(f"{seq}:{fmt(value)}" for seq, value in sorted(threshold.items()))
    return fmt(threshold)


def deterministic_same_count(cases: list[dict[str, Any]], counts_by_seq: Counter[str], salt: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    cases_by_seq: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        cases_by_seq[case["seq_id"]].append(case)
    for seq, group in cases_by_seq.items():
        n = min(counts_by_seq.get(seq, 0), len(group))
        ranked = sorted(
            group,
            key=lambda c: hashlib.sha256(f"{salt}|{seq}|{c['frame_id']}".encode("utf-8")).hexdigest(),
        )
        selected.extend(ranked[:n])
    return sorted(selected, key=lambda c: (c["seq_id"], c["frame_id"]))


def semantic_shuffle_select(cases: list[dict[str, Any]], score_col: str, direction: str, threshold: float | dict[str, float], internal_col: str | None, internal_threshold: float | None) -> list[dict[str, Any]]:
    if not cases:
        return []
    ordered = sorted(cases, key=lambda c: (c["seq_id"], c["frame_id"]))
    scores = [fnum(c.get(score_col, "")) for c in ordered]
    shifted = scores[1:] + scores[:1]
    selected: list[dict[str, Any]] = []
    for case, score in zip(ordered, shifted):
        case_threshold = threshold_for_case(threshold, case)
        sem_ok = score <= case_threshold if direction == "low" else score >= case_threshold
        int_ok = True
        if internal_col and internal_threshold is not None and math.isfinite(internal_threshold):
            int_ok = fnum(case.get(internal_col, "")) >= internal_threshold
        if sem_ok and int_ok:
            selected.append(case)
    return selected


def policy_metrics(surface_id: str, policy_id: str, policy_family: str, selected: list[dict[str, Any]], all_cases: list[dict[str, Any]], feasibility: dict[str, str], forbidden_repeat_match: bool) -> dict[str, Any]:
    selected_keys = {(c["seq_id"], c["frame_id"]) for c in selected}
    bad_total = sum(1 for c in all_cases if c["diagnostic_label"] == "bad")
    good_total = sum(1 for c in all_cases if c["diagnostic_label"] != "bad")
    bad_selected = sum(1 for c in selected if c["diagnostic_label"] == "bad")
    good_selected = sum(1 for c in selected if c["diagnostic_label"] != "bad")
    bad_recall = bad_selected / bad_total if bad_total else 0.0
    good_fpr = good_selected / good_total if good_total else 0.0
    balanced = 0.5 * (bad_recall + (1.0 - good_fpr)) if (bad_total or good_total) else 0.0
    seq_cov = len({c["seq_id"] for c in selected})
    keyframe_overlap = sum(1 for c in selected if int(c.get("keyframe_count", 0)) > 0)
    op_overlap = sum(int(c.get("operation_row_count", 0)) for c in selected)
    surface = SURFACE_DEFS[surface_id]
    primary_count_field = {
        "A": "keyframe_count",
        "B": "cache_append_count",
        "C": "retention_eviction_count",
        "D": "trajectory_write_count",
        "E": "local_reference_count",
        "F": "special_token_count",
    }[surface_id]
    action_effective_overlap = sum(int(c.get(primary_count_field, 0)) for c in selected)
    pilot_allowed = boolish(feasibility.get("full_sequence_pilot_allowed", "False"))
    top_gate = bool(seq_cov >= 2 and action_effective_overlap > 0 and not forbidden_repeat_match)
    full_candidate = bool(pilot_allowed and top_gate and policy_family in {"semantic_plus_internal", "internal_only", "semantic_shuffle", "same_count_random", "low_risk_reverse"})
    return {
        "schema": "acl2_v108tf_stage3_surface_policy_metric_row_v1",
        "surface_id": surface_id,
        "policy_id": policy_id,
        "policy_family": policy_family,
        "selection_count": len(selected),
        "selected_sequence_coverage": seq_cov,
        "selected_frame_count_by_seq": by_seq_counts(selected),
        "selected_keyframe_count_by_seq": by_seq_counts([c for c in selected if int(c.get("keyframe_count", 0)) > 0]),
        "selected_operation_count_by_seq": by_seq_counts(selected, "operation_row_count"),
        "action_effective_overlap": action_effective_overlap,
        "base_keyframe_overlap": keyframe_overlap,
        "operation_row_overlap": op_overlap,
        "diagnostic_bad_total": bad_total,
        "diagnostic_good_total": good_total,
        "diagnostic_bad_selected": bad_selected,
        "diagnostic_good_selected": good_selected,
        "diagnostic_bad_recall": bad_recall,
        "diagnostic_good_fpr": good_fpr,
        "diagnostic_balanced_accuracy": balanced,
        "forbidden_repeat_match": forbidden_repeat_match,
        "surface_stage1_pilot_allowed": pilot_allowed,
        "top_candidate_min_gate_pass": top_gate,
        "full_sequence_candidate": full_candidate,
        "blocked_reason": "" if full_candidate else ("stage1_new_hook_needed_or_surface_not_allowed" if not pilot_allowed else ("no_action_effective_overlap_or_seq_coverage" if not top_gate else "")),
        "expected_action_scope": surface["expected_action_scope"],
    }


def summarize_selected(selected: list[dict[str, Any]]) -> tuple[str, str]:
    sem_cols = [
        "stable_structure_mass",
        "dynamic_mass",
        "boundary_mass",
        "weak_context_mass",
        "sky_lowobs_mass",
        "semantic_continuity_score",
        "Q_ref_sem_balanced",
        "Q_ref_sem_risk_strict",
        "Q_ref_sem_stable_strict",
    ]
    int_cols = [
        "operation_row_count",
        "keyframe_count",
        "scale_frame_count",
        "cache_append_count",
        "retention_eviction_count",
        "trajectory_write_count",
        "local_reference_count",
        "special_token_count",
        "source_age_mean",
    ]
    sem = ";".join(f"{col}:{fmt(mean([fnum(c.get(col, '')) for c in selected]))}" for col in sem_cols)
    internal = ";".join(f"{col}:{fmt(mean([fnum(c.get(col, '')) for c in selected]))}" for col in int_cols)
    return sem, internal


def make_policy_row(surface_id: str, policy_id: str, policy_family: str, cue_family: str, selected: list[dict[str, Any]], metric: dict[str, Any], score_col: str, threshold: float) -> dict[str, Any]:
    sem_dist, internal_dist = summarize_selected(selected)
    return {
        "schema": "acl2_v108tf_stage3_surface_policy_row_v1",
        "surface_id": surface_id,
        "surface_name": SURFACE_DEFS[surface_id]["name"],
        "policy_id": policy_id,
        "policy_family": policy_family,
        "cue_family": cue_family,
        "score_column": score_col,
        "score_threshold": threshold_repr(threshold),
        "selection_count": len(selected),
        "selected_frame_count_by_seq": metric["selected_frame_count_by_seq"],
        "selected_keyframe_count_by_seq": metric["selected_keyframe_count_by_seq"],
        "selected_operation_count_by_seq": metric["selected_operation_count_by_seq"],
        "semantic_role_distribution": sem_dist,
        "internal_feature_distribution": internal_dist,
        "expected_action_scope": metric["expected_action_scope"],
        "full_sequence_candidate": metric["full_sequence_candidate"],
        "blocked_reason": metric["blocked_reason"],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    feasibility = load_feasibility()
    cases_by_surface = build_surface_cases()
    policy_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    top_by_surface: dict[str, dict[str, Any]] = {}

    for surface_id in sorted(SURFACE_DEFS):
        surface = SURFACE_DEFS[surface_id]
        cases = cases_by_surface.get(surface_id, [])
        if not cases:
            continue
        score_col = surface["score"]
        direction = surface["direction"]
        semantic_q = 0.25 if direction == "low" else 0.75
        semantic_selected, semantic_threshold = threshold_select_per_seq(cases, score_col, direction, semantic_q)
        internal_col = surface["internal"]
        internal_threshold = quantile([fnum(c.get(internal_col, "")) for c in cases], 0.50)
        semantic_plus = [
            c for c in semantic_selected
            if not math.isfinite(internal_threshold) or fnum(c.get(internal_col, "")) >= internal_threshold
        ]
        if not semantic_plus:
            semantic_plus = semantic_selected
        counts_by_seq = Counter(c["seq_id"] for c in semantic_plus)
        internal_selected = [
            c for c in cases
            if math.isfinite(internal_threshold) and fnum(c.get(internal_col, "")) >= internal_threshold
        ]
        if len(internal_selected) > len(semantic_plus) and semantic_plus:
            internal_selected = deterministic_same_count(internal_selected, counts_by_seq, f"{surface_id}|internal_same_count")
        shuffle_selected = semantic_shuffle_select(cases, score_col, direction, semantic_threshold, internal_col, internal_threshold)
        if len(shuffle_selected) > len(semantic_plus) and semantic_plus:
            shuffle_selected = deterministic_same_count(shuffle_selected, counts_by_seq, f"{surface_id}|shuffle_same_count")
        random_selected = deterministic_same_count(cases, counts_by_seq, f"{surface_id}|same_count_random")
        reverse_q = 0.75 if direction == "low" else 0.25
        reverse_direction = "high" if direction == "low" else "low"
        reverse_selected, reverse_threshold = threshold_select_per_seq(cases, score_col, reverse_direction, reverse_q)
        if len(reverse_selected) > len(semantic_plus) and semantic_plus:
            reverse_selected = deterministic_same_count(reverse_selected, counts_by_seq, f"{surface_id}|low_risk_reverse")

        candidates = [
            ("semantic_plus_internal", f"{surface_id}1_semantic_plus_internal", "semantic_plus_internal", semantic_plus, semantic_threshold, False),
            ("internal_only", f"{surface_id}1_internal_only", "internal_only", internal_selected, internal_threshold, False),
            ("semantic_only", f"{surface_id}1_semantic_only", "semantic_only", semantic_selected, semantic_threshold, False),
            ("semantic_shuffle", f"{surface_id}1_semantic_shuffle", "control_semantic_shuffle", shuffle_selected, semantic_threshold, False),
            ("same_count_random", f"{surface_id}1_same_count_random", "control_same_count_random", random_selected, float("nan"), False),
            ("low_risk_reverse", f"{surface_id}1_low_risk_reverse", "control_low_risk_reverse", reverse_selected, reverse_threshold, False),
        ]
        for policy_family, policy_id, cue_family, selected, threshold, forbidden in candidates:
            metric = policy_metrics(surface_id, policy_id, policy_family, selected, cases, feasibility.get(surface_id, {}), forbidden)
            metric_rows.append(metric)
            policy_rows.append(make_policy_row(surface_id, policy_id, policy_family, cue_family, selected, metric, score_col if policy_family != "same_count_random" else "deterministic_hash", threshold))
            for rank, case in enumerate(selected, start=1):
                frame_rows.append(
                    {
                        "schema": "acl2_v108tf_stage3_surface_policy_frame_row_v1",
                        "surface_id": surface_id,
                        "policy_id": policy_id,
                        "policy_family": policy_family,
                        "seq_id": case["seq_id"],
                        "frame_id": case["frame_id"],
                        "selected_rank": rank,
                        "diagnostic_label": case["diagnostic_label"],
                        "target_ids": ";".join(case["target_ids"]),
                        "target_kinds": ";".join(case["target_kinds"]),
                        "operation_row_count": case["operation_row_count"],
                        "keyframe_count": case["keyframe_count"],
                        "score_value": fmt(case.get(score_col, "")),
                        "Q_ref_sem_balanced": fmt(case.get("Q_ref_sem_balanced", "")),
                        "Q_ref_sem_risk_strict": fmt(case.get("Q_ref_sem_risk_strict", "")),
                        "semantic_continuity_score": fmt(case.get("semantic_continuity_score", "")),
                        "operation_types": case["operation_types"],
                    }
                )
        primary_metric = next(row for row in metric_rows if row["policy_id"] == f"{surface_id}1_semantic_plus_internal")
        top_by_surface[surface_id] = primary_metric

    metric_by_policy = {row["policy_id"]: row for row in metric_rows}
    for row in metric_rows:
        sid = row["surface_id"]
        primary = metric_by_policy.get(f"{sid}1_semantic_plus_internal", {})
        internal = metric_by_policy.get(f"{sid}1_internal_only", {})
        random = metric_by_policy.get(f"{sid}1_same_count_random", {})
        shuffle = metric_by_policy.get(f"{sid}1_semantic_shuffle", {})
        row["semantic_plus_minus_internal_ba"] = (
            fnum(primary.get("diagnostic_balanced_accuracy", "")) - fnum(internal.get("diagnostic_balanced_accuracy", ""))
            if primary and internal else float("nan")
        )
        row["semantic_plus_minus_random_ba"] = (
            fnum(primary.get("diagnostic_balanced_accuracy", "")) - fnum(random.get("diagnostic_balanced_accuracy", ""))
            if primary and random else float("nan")
        )
        row["semantic_plus_minus_shuffle_ba"] = (
            fnum(primary.get("diagnostic_balanced_accuracy", "")) - fnum(shuffle.get("diagnostic_balanced_accuracy", ""))
            if primary and shuffle else float("nan")
        )

    write_csv(OUT / "surface_policy_rows.csv", policy_rows)
    write_csv(OUT / "surface_policy_metrics.csv", metric_rows)
    write_csv(OUT / "surface_policy_frame_rows.csv", frame_rows)

    full_candidates = [row for row in metric_rows if bool(row.get("full_sequence_candidate")) and row["policy_family"] in {"semantic_plus_internal", "internal_only", "same_count_random", "semantic_shuffle"}]
    primary_ready = [row for row in metric_rows if bool(row.get("full_sequence_candidate")) and row["policy_family"] == "semantic_plus_internal"]
    blocked_surfaces = [
        sid for sid, row in feasibility.items()
        if not boolish(row.get("full_sequence_pilot_allowed", "False"))
    ]
    summary = {
        "schema": "acl2_v108tf_stage3_summary_v1",
        "stage3_pass": bool(primary_ready),
        "stage3_action_surface_screen_pass": bool(primary_ready),
        "surface_count": len(SURFACE_DEFS),
        "surface_case_counts": {sid: len(cases_by_surface.get(sid, [])) for sid in sorted(SURFACE_DEFS)},
        "full_sequence_candidate_count": len(full_candidates),
        "semantic_plus_internal_full_sequence_candidate_count": len(primary_ready),
        "full_sequence_candidate_policy_ids": [row["policy_id"] for row in full_candidates],
        "semantic_plus_internal_candidate_policy_ids": [row["policy_id"] for row in primary_ready],
        "surfaces_with_semantic_plus_internal_candidate": sorted({row["surface_id"] for row in primary_ready}),
        "surfaces_stage1_new_hook_needed_or_not_allowed": blocked_surfaces,
        "top_candidate_by_surface": {
            sid: {
                "policy_id": row["policy_id"],
                "full_sequence_candidate": row["full_sequence_candidate"],
                "selection_count": row["selection_count"],
                "selected_sequence_coverage": row["selected_sequence_coverage"],
                "action_effective_overlap": row["action_effective_overlap"],
                "diagnostic_balanced_accuracy": row["diagnostic_balanced_accuracy"],
                "blocked_reason": row["blocked_reason"],
            }
            for sid, row in sorted(top_by_surface.items())
        },
        "outputs": {
            "surface_policy_rows": rel(OUT / "surface_policy_rows.csv"),
            "surface_policy_metrics": rel(OUT / "surface_policy_metrics.csv"),
            "surface_policy_frame_rows": rel(OUT / "surface_policy_frame_rows.csv"),
            "surface_rank_report": rel(OUT / "surface_rank_report.md"),
        },
        "caution": "Stage3 is a pre-screen only; no full KITTI geometry success is claimed here.",
    }
    write_json(OUT / "stage3_summary.json", summary)

    report_lines = [
        "# v108TF Stage3 Surface Rank Report",
        "",
        "Stage3 ranks action-surface candidates for Stage4.  It does not claim geometry improvement.",
        "",
        "| surface | top policy | full candidate | selection | seq coverage | action overlap | diagnostic BA | note |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for sid in sorted(SURFACE_DEFS):
        row = top_by_surface.get(sid, {})
        note = row.get("blocked_reason", "") or SURFACE_DEFS[sid]["expected_action_scope"]
        report_lines.append(
            f"| {sid} `{SURFACE_DEFS[sid]['name']}` | `{row.get('policy_id', '')}` | {row.get('full_sequence_candidate', False)} | {row.get('selection_count', 0)} | {row.get('selected_sequence_coverage', 0)} | {row.get('action_effective_overlap', 0)} | {fmt(row.get('diagnostic_balanced_accuracy', ''))} | {note} |"
        )
    report_lines.extend(
        [
            "",
            "Stage4-ready primary semantic+internal candidates:",
            "",
            *[f"- `{row['policy_id']}` surface `{row['surface_id']}`" for row in primary_ready],
            "",
            "Controls with `full_sequence_candidate=true` are available in `surface_policy_metrics.csv`; they are not method claims.",
        ]
    )
    (OUT / "surface_rank_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
