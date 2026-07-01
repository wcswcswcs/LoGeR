#!/usr/bin/env python3
"""Run v101 Stage7 support-conditioned diagnostics without authorizing action.

This script joins v101 target/support/observability/role/state rows with v100
F4/R/R2 diagnostic artifacts.  It materializes N3/C5/F5/R3 diagnostic evidence
where possible, but every output remains action-blocked because the clean target
universe, true-stage Q2, strict Track V controls, and M4 simulator gates fail.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
TRACK_T = ROOT / "trackT_drift_target_relabel"
TRACK_S2 = ROOT / "trackS2_anchor_state_estimator"

V100 = Path("results/acl2_v100tf_same_space_semantic_anchor_latent_state_memory_control")
V100_F4 = V100 / "trackF4_ttt_write_to_use_same_space"
V100_R = V100 / "trackR_edge_head_control_audit"
V100_R2 = V100 / "trackR2_anchor_edge_identity_control_audit"

OUT_N3 = ROOT / "trackN3_anchor_identity_graph_cleaned_targets"
OUT_C5 = ROOT / "trackC5_identity_latent_gauge_with_support"
OUT_F5 = ROOT / "trackF5_ttt_write_to_use_state_chain"
OUT_R3 = ROOT / "trackR3_query_head_anchor_edge_audit_true_support"

POS_TAX = "HANDOFF_SCALE_GAUGE_TARGET"
SAFE_TAX = "SAFE_GOOD"
EPS = 1.0e-12


def f(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
            clean: dict[str, Any] = {}
            for key in keys:
                value = row.get(key, "")
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, ensure_ascii=False, sort_keys=True)
                clean[key] = value
            writer.writerow(clean)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def mean(values: list[Any]) -> float:
    vals = [f(v) for v in values if math.isfinite(f(v))]
    return sum(vals) / len(vals) if vals else math.nan


def quantile(values: list[Any], q: float) -> float:
    vals = sorted(f(v) for v in values if math.isfinite(f(v)))
    if not vals:
        return math.nan
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def pearson(xs: list[Any], ys: list[Any]) -> float:
    pairs: list[tuple[float, float]] = []
    for x, y in zip(xs, ys):
        fx = f(x)
        fy = f(y)
        if math.isfinite(fx) and math.isfinite(fy):
            pairs.append((fx, fy))
    if len(pairs) < 2:
        return math.nan
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= EPS or vy <= EPS:
        return math.nan
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def safe_div(num: float, den: float) -> float:
    return num / den if den else math.nan


def seq_from_case(case_id: str) -> str:
    return case_id.split("_", 1)[0] if "_" in case_id else ""


def state_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("case_id", "")), str(row.get("anchor_id", ""))


def is_nonempty_number(value: Any) -> bool:
    return math.isfinite(f(value))


def target_maps() -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    target_rows = read_rows(TRACK_T / "target_universe_v101.csv")
    return {row.get("case_id", ""): row for row in target_rows}, target_rows


def state_rows() -> tuple[list[dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    rows = read_rows(TRACK_S2 / "anchor_state_rows.csv")
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = state_key(row)
        if key[0] and key[1] and key not in by_key:
            by_key[key] = row
    return rows, by_key


def metric_eval(
    case_rows: list[dict[str, Any]],
    score_field: str,
    *,
    direction: str = "higher_bad",
    cue_name: str,
    random_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eval_rows = [
        row
        for row in case_rows
        if row.get("target_taxonomy") in {POS_TAX, SAFE_TAX} and math.isfinite(f(row.get(score_field)))
    ]
    positives = [row for row in eval_rows if row.get("target_taxonomy") == POS_TAX]
    safe = [row for row in eval_rows if row.get("target_taxonomy") == SAFE_TAX]
    k = len(positives)
    selected: set[str] = set()
    if k > 0:
        reverse = direction == "higher_bad"
        ranked = sorted(eval_rows, key=lambda row: f(row.get(score_field)), reverse=reverse)
        selected = {row["case_id"] for row in ranked[:k]}
    tp = sum(1 for row in positives if row["case_id"] in selected)
    fp = sum(1 for row in safe if row["case_id"] in selected)
    recall = safe_div(tp, len(positives))
    fpr = safe_div(fp, len(safe))
    tnr = math.nan if not math.isfinite(fpr) else 1.0 - fpr
    ba = (recall + tnr) / 2.0 if math.isfinite(recall) and math.isfinite(tnr) else math.nan
    selected_positive_seqs = {seq_from_case(row["case_id"]) for row in positives if row["case_id"] in selected}
    corr_all = pearson([row.get(score_field) for row in case_rows], [row.get("L3_handoff_transfer_penalty_proxy") for row in case_rows])

    rng = random.Random(random_seed)
    random_bas: list[float] = []
    all_ids = [row["case_id"] for row in eval_rows]
    positive_ids = {row["case_id"] for row in positives}
    safe_ids = {row["case_id"] for row in safe}
    if k > 0 and len(eval_rows) >= k:
        for _ in range(512):
            sample = set(rng.sample(all_ids, k))
            r_tp = len(sample & positive_ids)
            r_fp = len(sample & safe_ids)
            r_recall = safe_div(r_tp, len(positive_ids))
            r_fpr = safe_div(r_fp, len(safe_ids))
            if math.isfinite(r_recall) and math.isfinite(r_fpr):
                random_bas.append((r_recall + (1.0 - r_fpr)) / 2.0)
    random_mean = mean(random_bas)
    random_p95 = quantile(random_bas, 0.95)
    margin = ba - random_p95 if math.isfinite(ba) and math.isfinite(random_p95) else math.nan
    gate_without_controls = (
        math.isfinite(recall)
        and math.isfinite(fpr)
        and recall >= 0.65
        and fpr <= 0.25
        and len(selected_positive_seqs) >= 3
        and math.isfinite(corr_all)
        and abs(corr_all) >= 0.50
    )
    details = []
    for row in eval_rows:
        kind = "selected_positive" if row["case_id"] in selected and row.get("target_taxonomy") == POS_TAX else ""
        if row["case_id"] in selected and row.get("target_taxonomy") == SAFE_TAX:
            kind = "false_positive_safe_good"
        if row["case_id"] not in selected and row.get("target_taxonomy") == POS_TAX:
            kind = "missed_positive"
        if kind:
            details.append(
                {
                    "cue_name": cue_name,
                    "row_kind": kind,
                    "case_id": row["case_id"],
                    "seq": seq_from_case(row["case_id"]),
                    "target_taxonomy": row.get("target_taxonomy", ""),
                    "score_field": score_field,
                    "score_value": row.get(score_field, ""),
                    "L3_handoff_transfer_penalty_proxy": row.get("L3_handoff_transfer_penalty_proxy", ""),
                }
            )
    metrics = {
        "cue_name": cue_name,
        "score_field": score_field,
        "direction": direction,
        "available_case_count": len(case_rows),
        "eval_case_count": len(eval_rows),
        "positive_case_count": len(positives),
        "safe_good_case_count": len(safe),
        "selected_case_count": len(selected),
        "bad_recall": recall,
        "good_FPR": fpr,
        "balanced_accuracy": ba,
        "selected_positive_sequence_coverage": len(selected_positive_seqs),
        "abs_corr_L3_all_cases": abs(corr_all) if math.isfinite(corr_all) else math.nan,
        "corr_L3_all_cases": corr_all,
        "same_count_random_repeats": len(random_bas),
        "same_count_random_BA_mean": random_mean,
        "same_count_random_BA_p95": random_p95,
        "same_count_random_margin_vs_p95": margin,
        "anchor_id_rotation_margin": math.nan,
        "semantic_label_rotation_margin": math.nan,
        "query_head_random_margin": math.nan,
        "control_margins_available": False,
        "gate_without_controls_pass": gate_without_controls,
        "gate_pass": False,
        "gate_blocker": "Track T target counts/coverage and required identity/semantic/query-head controls are insufficient.",
    }
    return metrics, details


def summarize_track(
    out_dir: Path,
    summary_name: str,
    summary: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    failure_lines: list[str],
    pass_requirements: str,
    next_attempt: str,
) -> None:
    write_json(out_dir / summary_name, summary)
    write_json(out_dir / "blocked_summary.json", {**summary, "run_allowed": False, "runtime_action_allowed": False})
    write_rows(out_dir / "gate_checks.csv", summary.get("gate_checks", []))
    write_rows(
        out_dir / "not_run_manifest.csv",
        [
            {
                "track": summary.get("track", ""),
                "not_run": False,
                "diagnostic_run": True,
                "runtime_action_run": False,
                "status": summary.get("status", ""),
                "reason": "; ".join(failure_lines),
            }
        ],
    )
    write_text(out_dir / "failure_report.md", "\n".join(f"- {line}" for line in failure_lines))
    write_text(out_dir / "control_gap_report.md", "Required control margins are not complete:\n" + "\n".join(f"- {line}" for line in summary.get("control_gaps", [])))
    write_text(out_dir / "what_would_have_to_be_true_to_pass.md", pass_requirements)
    write_text(out_dir / "next_attempt_recommendation.md", next_attempt)
    if metric_rows:
        write_rows(out_dir / "metric_summary.csv", metric_rows)


def build_case_aggregates(rows: list[dict[str, str]], target_by_case: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("case_id", "")].append(row)
    out: list[dict[str, Any]] = []
    for case_id, parts in sorted(grouped.items()):
        target = target_by_case.get(case_id, {})
        stale_terms = []
        fresh_terms = []
        for row in parts:
            query = f(row.get("query_hit_max"), 0.0)
            rsame = f(row.get("R_same"), 0.0)
            support = f(row.get("S_cur_combined"), 0.0)
            oscale = f(row.get("O_scale"), 0.0)
            role = row.get("role", "")
            state = row.get("state_status", "")
            if role == "stale_candidate" or state == "unsupported_inconsistent":
                stale_terms.append(query * (1.0 - support) * rsame)
            if role in {"local_recent", "landmark"} or state == "supported_consistent":
                fresh_terms.append(query * support * oscale * max(0.0, 1.0 - rsame))
        out.append(
            {
                "case_id": case_id,
                "seq": target.get("seq", seq_from_case(case_id)),
                "target_taxonomy": target.get("target_taxonomy", ""),
                "L3_handoff_transfer_penalty_proxy": target.get("L3_handoff_transfer_penalty_proxy", ""),
                "anchor_count": len(parts),
                "stale_unsupported_overread_score": mean(stale_terms),
                "fresh_supported_landmark_score": mean(fresh_terms),
                "stale_minus_fresh_score": (mean(stale_terms) if math.isfinite(mean(stale_terms)) else 0.0)
                - (mean(fresh_terms) if math.isfinite(mean(fresh_terms)) else 0.0),
                "mean_S_cur": mean([row.get("S_cur_combined") for row in parts]),
                "mean_O_scale": mean([row.get("O_scale") for row in parts]),
                "mean_R_same": mean([row.get("R_same") for row in parts]),
                "mean_K_anchor": mean([row.get("K_anchor") for row in parts]),
                "query_hit_max": max([f(row.get("query_hit_max"), 0.0) for row in parts] or [0.0]),
                "stale_candidate_frac": mean([1.0 if row.get("role") == "stale_candidate" else 0.0 for row in parts]),
                "unsupported_inconsistent_frac": mean([1.0 if row.get("state_status") == "unsupported_inconsistent" else 0.0 for row in parts]),
                "supported_consistent_frac": mean([1.0 if row.get("state_status") == "supported_consistent" else 0.0 for row in parts]),
            }
        )
    return out


def run_n3(rows: list[dict[str, str]], target_by_case: dict[str, dict[str, str]]) -> dict[str, Any]:
    case_rows = build_case_aggregates(rows, target_by_case)
    metrics: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for cue, field in [
        ("N3_stale_minus_fresh", "stale_minus_fresh_score"),
        ("N3_stale_unsupported_overread", "stale_unsupported_overread_score"),
        ("N3_unsupported_inconsistent_frac", "unsupported_inconsistent_frac"),
    ]:
        metric, detail = metric_eval(case_rows, field, cue_name=cue, random_seed=3103 + len(metrics))
        metrics.append(metric)
        details.extend(detail)
    best = max(metrics, key=lambda row: f(row.get("balanced_accuracy"), -1.0)) if metrics else {}
    failure_lines = [
        "Track T has only one clean HANDOFF target and five SAFE_GOOD controls, so sequence coverage cannot reach the N3 gate.",
        "Anchor-id, semantic-label, and query-head rotation controls are not materialized for this v101 support-conditioned N3 join.",
        "This diagnostic uses same-space anchor/state rows only; it does not authorize any identity-specific action.",
    ]
    summary = {
        "schema": "acl2_v101_trackN3_support_conditioned_identity_graph_diagnostic_v1",
        "track": "N3",
        "status": "complete_diagnostic_blocked",
        "gate_pass": False,
        "case_count": len(case_rows),
        "anchor_row_count": len(rows),
        "positive_case_count": sum(1 for row in case_rows if row.get("target_taxonomy") == POS_TAX),
        "safe_good_case_count": sum(1 for row in case_rows if row.get("target_taxonomy") == SAFE_TAX),
        "best_cue": best.get("cue_name", ""),
        "best_balanced_accuracy": best.get("balanced_accuracy", math.nan),
        "best_good_FPR": best.get("good_FPR", math.nan),
        "best_bad_recall": best.get("bad_recall", math.nan),
        "best_selected_positive_sequence_coverage": best.get("selected_positive_sequence_coverage", math.nan),
        "runtime_action_allowed": False,
        "control_gaps": [
            "anchor-id rotation margin missing",
            "semantic-label rotation margin missing",
            "query-head random margin missing",
        ],
        "gate_checks": [
            {"gate": "handoff_target_count_ge_8", "pass": False, "observed": 1},
            {"gate": "safe_good_count_ge_6", "pass": False, "observed": 5},
            {"gate": "selected_positive_sequence_coverage_ge_3", "pass": False, "observed": best.get("selected_positive_sequence_coverage", "")},
            {"gate": "control_margins_available", "pass": False, "observed": False},
            {"gate": "runtime_action_allowed", "pass": False, "observed": False},
        ],
        "failure_lines": failure_lines,
    }
    write_rows(OUT_N3 / "anchor_graph_pattern_rows.csv", case_rows)
    write_rows(OUT_N3 / "false_positive_false_negative_rows.csv", details)
    summarize_track(
        OUT_N3,
        "N3_summary.json",
        summary,
        metrics,
        failure_lines,
        "N3 needs sequence-covered clean HANDOFF targets, SAFE_GOOD controls, and anchor-id/semantic/query-head controls before action.",
        "Materialize stable instance/component identity and query-head controls for an expanded v100-schema target universe.",
    )
    return summary


def run_c5(rows: list[dict[str, str]], target_by_case: dict[str, dict[str, str]]) -> dict[str, Any]:
    rsame_hi = quantile([row.get("R_same") for row in rows], 0.75)
    rsame_lo = quantile([row.get("R_same") for row in rows], 0.25)
    support_hi = quantile([row.get("S_cur_combined") for row in rows], 0.75)
    support_lo = quantile([row.get("S_cur_combined") for row in rows], 0.25)
    oscale_hi = quantile([row.get("O_scale") for row in rows], 0.75)
    oscale_lo = quantile([row.get("O_scale") for row in rows], 0.25)
    per_case_q90 = {
        case_id: quantile([row.get("query_hit_max") for row in parts], 0.90)
        for case_id, parts in defaultdict(list, {}).items()
    }
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("case_id", "")].append(row)
    per_case_q90 = {case_id: quantile([row.get("query_hit_max") for row in parts], 0.90) for case_id, parts in grouped.items()}

    group_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for case_id, parts in sorted(grouped.items()):
        target = target_by_case.get(case_id, {})
        q90 = per_case_q90.get(case_id, math.nan)
        a_mass = 0.0
        b_mass = 0.0
        high_count = 0
        for row in parts:
            query = f(row.get("query_hit_max"), 0.0)
            rsame = f(row.get("R_same"), 0.0)
            support = f(row.get("S_cur_combined"), 0.0)
            oscale = f(row.get("O_scale"), 0.0)
            if not (math.isfinite(q90) and query >= q90):
                continue
            high_count += 1
            group = "high_hit_other"
            if rsame >= rsame_hi and support <= support_lo and oscale <= oscale_lo:
                group = "A_high_hit_high_R_low_support_low_O"
                a_mass += query
            elif rsame <= rsame_lo and support >= support_hi and oscale >= oscale_hi:
                group = "B_high_hit_low_R_high_support_high_O"
                b_mass += query
            group_rows.append(
                {
                    "case_id": case_id,
                    "seq": target.get("seq", seq_from_case(case_id)),
                    "target_taxonomy": target.get("target_taxonomy", ""),
                    "anchor_id": row.get("anchor_id", ""),
                    "semantic_label": row.get("semantic_label", ""),
                    "query_hit_max": query,
                    "R_same": row.get("R_same", ""),
                    "S_cur_combined": row.get("S_cur_combined", ""),
                    "O_scale": row.get("O_scale", ""),
                    "role": row.get("role", ""),
                    "state_status": row.get("state_status", ""),
                    "group": group,
                    "proxy_only": row.get("proxy_only", ""),
                }
            )
        case_rows.append(
            {
                "case_id": case_id,
                "seq": target.get("seq", seq_from_case(case_id)),
                "target_taxonomy": target.get("target_taxonomy", ""),
                "L3_handoff_transfer_penalty_proxy": target.get("L3_handoff_transfer_penalty_proxy", ""),
                "high_hit_anchor_count": high_count,
                "groupA_query_mass": a_mass,
                "groupB_query_mass": b_mass,
                "groupA_minus_groupB_query_mass": a_mass - b_mass,
                "groupA_present": a_mass > 0.0,
                "groupB_present": b_mass > 0.0,
            }
        )
    metrics: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    metric, detail = metric_eval(case_rows, "groupA_minus_groupB_query_mass", cue_name="C5_groupA_minus_groupB", random_seed=5105)
    metrics.append(metric)
    details.extend(detail)
    best = metrics[0] if metrics else {}
    failure_lines = [
        "C5 high-hit grouping uses v101 support/observability proxies and cannot pass without true-stage Q2/V controls.",
        "Only one clean HANDOFF target is available, so Group A/B separation cannot meet sequence coverage >=3.",
        "Identity and query-head controls are missing for the support-conditioned high-hit groups.",
    ]
    summary = {
        "schema": "acl2_v101_trackC5_support_conditioned_latent_gauge_diagnostic_v1",
        "track": "C5",
        "status": "complete_diagnostic_blocked",
        "gate_pass": False,
        "anchor_group_row_count": len(group_rows),
        "case_count": len(case_rows),
        "groupA_anchor_count": sum(1 for row in group_rows if row.get("group") == "A_high_hit_high_R_low_support_low_O"),
        "groupB_anchor_count": sum(1 for row in group_rows if row.get("group") == "B_high_hit_low_R_high_support_high_O"),
        "R_same_q75": rsame_hi,
        "R_same_q25": rsame_lo,
        "S_cur_q75": support_hi,
        "S_cur_q25": support_lo,
        "O_scale_q75": oscale_hi,
        "O_scale_q25": oscale_lo,
        "balanced_accuracy": best.get("balanced_accuracy", math.nan),
        "good_FPR": best.get("good_FPR", math.nan),
        "bad_recall": best.get("bad_recall", math.nan),
        "runtime_action_allowed": False,
        "control_gaps": [
            "semantic/identity rotation controls missing",
            "query-head random controls missing",
            "true current support and strict O_scale gate missing",
        ],
        "gate_checks": [
            {"gate": "sequence_coverage_ge_3", "pass": False, "observed": best.get("selected_positive_sequence_coverage", "")},
            {"gate": "safe_good_FPR_le_0p25", "pass": math.isfinite(f(best.get("good_FPR"))) and f(best.get("good_FPR")) <= 0.25, "observed": best.get("good_FPR", "")},
            {"gate": "control_margins_available", "pass": False, "observed": False},
            {"gate": "runtime_action_allowed", "pass": False, "observed": False},
        ],
        "failure_lines": failure_lines,
    }
    write_rows(OUT_C5 / "high_hit_anchor_groups.csv", group_rows)
    write_rows(OUT_C5 / "harmful_safe_use_rows.csv", case_rows)
    write_rows(OUT_C5 / "false_positive_false_negative_rows.csv", details)
    write_rows(OUT_C5 / "latent_support_interaction_metrics.csv", metrics)
    summarize_track(
        OUT_C5,
        "C5_summary.json",
        summary,
        metrics,
        failure_lines,
        "C5 needs Group A/B separation on sequence-covered clean targets plus semantic/identity/query-head controls.",
        "Expand clean handoff target traces and materialize instance/query-head controls before considering latent gauge action.",
    )
    return summary


def run_f5(rows: list[dict[str, str]], target_by_case: dict[str, dict[str, str]]) -> dict[str, Any]:
    f4_rows = read_rows(V100_F4 / "rows.csv")
    case_agg = {row["case_id"]: row for row in build_case_aggregates(rows, target_by_case)}
    proxy_rows: list[dict[str, Any]] = []
    for row in f4_rows:
        case_id = row.get("case_id", "")
        target = target_by_case.get(case_id, {})
        agg = case_agg.get(case_id, {})
        write_risk = f(row.get("write_cache_current_risk"), 0.0)
        support = f(agg.get("mean_S_cur"), 0.0)
        oscale = f(agg.get("mean_O_scale"), 0.0)
        proxy_rows.append(
            {
                "case_id": case_id,
                "seq": target.get("seq", row.get("seq", "")),
                "target_taxonomy": target.get("target_taxonomy", ""),
                "L3_handoff_transfer_penalty_proxy": target.get("L3_handoff_transfer_penalty_proxy", row.get("L3_handoff_transfer_penalty_proxy", "")),
                "v100_write_to_use_chain_coverage": row.get("write_to_use_chain_coverage", ""),
                "v100_write_cache_current_risk": row.get("write_cache_current_risk", ""),
                "v100_R_write_cache_mean": row.get("R_write_cache_mean", ""),
                "v100_R_cache_current_mean": row.get("R_cache_current_mean", ""),
                "mean_S_cur_v101": agg.get("mean_S_cur", ""),
                "mean_O_scale_v101": agg.get("mean_O_scale", ""),
                "mean_K_anchor_v101": agg.get("mean_K_anchor", ""),
                "support_conditioned_write_use_risk_proxy": write_risk * max(0.0, 1.0 - support) * max(0.0, 1.0 - oscale),
                "claim_level": "case_level_proxy_only_not_per_anchor_write_to_use",
            }
        )
    metrics: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for cue, field in [
        ("F5_support_conditioned_write_use_risk_proxy", "support_conditioned_write_use_risk_proxy"),
        ("F5_v100_write_cache_current_risk", "v100_write_cache_current_risk"),
    ]:
        metric, detail = metric_eval(proxy_rows, field, cue_name=cue, random_seed=5505 + len(metrics))
        metrics.append(metric)
        details.extend(detail)
    nonempty_counts = {
        "r_write_cache_nonempty": sum(1 for row in rows if is_nonempty_number(row.get("r_write_cache"))),
        "r_cache_current_nonempty": sum(1 for row in rows if is_nonempty_number(row.get("r_cache_current"))),
        "r_ref_current_nonempty": sum(1 for row in rows if is_nonempty_number(row.get("r_ref_current"))),
        "anchor_state_row_count": len(rows),
    }
    best = max(metrics, key=lambda row: f(row.get("balanced_accuracy"), -1.0)) if metrics else {}
    failure_lines = [
        "v101 per-anchor write/cache/current residual fields are not materialized in anchor_state_rows, so F5 remains case-level proxy only.",
        "Track T target/support universe is too small for sequence-covered support-conditioned write-to-use risk.",
        "Identity/semantic controls are not available for the F5 support-conditioned proxy.",
    ]
    summary = {
        "schema": "acl2_v101_trackF5_support_conditioned_write_to_use_diagnostic_v1",
        "track": "F5",
        "status": "complete_diagnostic_blocked",
        "gate_pass": False,
        "case_count": len(proxy_rows),
        "materialization_audit": nonempty_counts,
        "best_cue": best.get("cue_name", ""),
        "best_balanced_accuracy": best.get("balanced_accuracy", math.nan),
        "best_good_FPR": best.get("good_FPR", math.nan),
        "best_bad_recall": best.get("bad_recall", math.nan),
        "runtime_action_allowed": False,
        "control_gaps": [
            "per-anchor TTT write-to-cache/use chain missing in v101 rows",
            "identity/semantic rotation controls missing",
            "query-head random controls missing",
        ],
        "gate_checks": [
            {"gate": "per_anchor_write_chain_materialized", "pass": nonempty_counts["r_write_cache_nonempty"] > 0, "observed": nonempty_counts["r_write_cache_nonempty"]},
            {"gate": "safe_good_FPR_le_0p25", "pass": math.isfinite(f(best.get("good_FPR"))) and f(best.get("good_FPR")) <= 0.25, "observed": best.get("good_FPR", "")},
            {"gate": "control_margins_available", "pass": False, "observed": False},
            {"gate": "runtime_action_allowed", "pass": False, "observed": False},
        ],
        "failure_lines": failure_lines,
    }
    write_rows(OUT_F5 / "write_to_use_proxy_rows.csv", proxy_rows)
    write_rows(OUT_F5 / "write_to_use_materialization_audit.csv", [nonempty_counts])
    write_rows(OUT_F5 / "false_positive_false_negative_rows.csv", details)
    write_rows(OUT_F5 / "latent_support_interaction_metrics.csv", metrics)
    summarize_track(
        OUT_F5,
        "F5_summary.json",
        summary,
        metrics,
        failure_lines,
        "F5 needs per-anchor write->cache->query use materialization joined to S_cur/O_scale/K_anchor and controls.",
        "Add diagnostic-only per-anchor write/use dumps before any TTT metadata/action proposal.",
    )
    return summary


def run_r3(rows_by_key: dict[tuple[str, str], dict[str, str]], target_by_case: dict[str, dict[str, str]]) -> dict[str, Any]:
    per_case: dict[str, dict[str, Any]] = {}
    top_rows: list[dict[str, Any]] = []
    joined_count = 0
    total_count = 0
    for edge in read_rows(V100_R2 / "anchor_edge_rows.csv"):
        total_count += 1
        case_id = edge.get("case_id", "")
        anchor_id = edge.get("anchor_id", "")
        state = rows_by_key.get((case_id, anchor_id))
        if not state:
            continue
        joined_count += 1
        query = f(edge.get("anchor_query_hit_frac"), 0.0)
        top1 = f(edge.get("anchor_top1_hit_frac"), 0.0)
        rsame = f(state.get("R_same"), f(edge.get("anchor_hidden_R_same"), 0.0))
        support = f(state.get("S_cur_combined"), 0.0)
        oscale = f(state.get("O_scale"), 0.0)
        edge_risk = query * max(0.0, rsame) * max(0.0, 1.0 - support) * max(0.0, 1.0 - oscale)
        safe_score = query * support * oscale * max(0.0, 1.0 - rsame)
        target = target_by_case.get(case_id, {})
        agg = per_case.setdefault(
            case_id,
            {
                "case_id": case_id,
                "seq": target.get("seq", seq_from_case(case_id)),
                "target_taxonomy": target.get("target_taxonomy", ""),
                "L3_handoff_transfer_penalty_proxy": target.get("L3_handoff_transfer_penalty_proxy", edge.get("L3_handoff_transfer_penalty_proxy", "")),
                "joined_anchor_edge_row_count": 0,
                "edge_risk_sum": 0.0,
                "edge_safe_score_sum": 0.0,
                "edge_risk_max": 0.0,
                "edge_top1_hit_max": 0.0,
                "stale_edge_risk_sum": 0.0,
            },
        )
        agg["joined_anchor_edge_row_count"] += 1
        agg["edge_risk_sum"] += edge_risk
        agg["edge_safe_score_sum"] += safe_score
        agg["edge_risk_max"] = max(f(agg["edge_risk_max"], 0.0), edge_risk)
        agg["edge_top1_hit_max"] = max(f(agg["edge_top1_hit_max"], 0.0), top1)
        if state.get("role") == "stale_candidate" or state.get("state_status") == "unsupported_inconsistent":
            agg["stale_edge_risk_sum"] += edge_risk
        if edge_risk > 0.0:
            top_rows.append(
                {
                    "case_id": case_id,
                    "seq": target.get("seq", seq_from_case(case_id)),
                    "target_taxonomy": target.get("target_taxonomy", ""),
                    "head_idx": edge.get("head_idx", ""),
                    "anchor_id": anchor_id,
                    "semantic_label": state.get("semantic_label", edge.get("semantic_class", "")),
                    "anchor_query_hit_frac": edge.get("anchor_query_hit_frac", ""),
                    "anchor_top1_hit_frac": edge.get("anchor_top1_hit_frac", ""),
                    "R_same": rsame,
                    "S_cur_combined": support,
                    "O_scale": oscale,
                    "role": state.get("role", ""),
                    "state_status": state.get("state_status", ""),
                    "support_conditioned_edge_risk": edge_risk,
                    "support_conditioned_safe_edge_score": safe_score,
                    "claim_level": "joined_v100_R2_edge_with_v101_proxy_support_observability",
                }
            )
    case_rows = []
    for row in per_case.values():
        count = f(row.get("joined_anchor_edge_row_count"), 0.0)
        row["edge_risk_mean"] = safe_div(f(row.get("edge_risk_sum"), 0.0), count)
        row["edge_safe_score_mean"] = safe_div(f(row.get("edge_safe_score_sum"), 0.0), count)
        row["stale_edge_risk_mean"] = safe_div(f(row.get("stale_edge_risk_sum"), 0.0), count)
        row["edge_risk_minus_safe_mean"] = (row["edge_risk_mean"] if math.isfinite(row["edge_risk_mean"]) else 0.0) - (
            row["edge_safe_score_mean"] if math.isfinite(row["edge_safe_score_mean"]) else 0.0
        )
        case_rows.append(row)
    top_rows = sorted(top_rows, key=lambda row: f(row.get("support_conditioned_edge_risk"), 0.0), reverse=True)[:512]
    metrics: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for cue, field in [
        ("R3_edge_risk_max", "edge_risk_max"),
        ("R3_edge_risk_mean", "edge_risk_mean"),
        ("R3_stale_edge_risk_mean", "stale_edge_risk_mean"),
    ]:
        metric, detail = metric_eval(case_rows, field, cue_name=cue, random_seed=7303 + len(metrics))
        metrics.append(metric)
        details.extend(detail)
    best = max(metrics, key=lambda row: f(row.get("balanced_accuracy"), -1.0)) if metrics else {}
    failure_lines = [
        "R3 joined v100 R2 anchor-edge rows with v101 support/observability, but query-head random and identity/semantic controls remain incomplete.",
        "Clean HANDOFF target sequence coverage remains one, so an edge action cannot pass the R3 gate.",
        "The joined O_scale/current-support terms are still diagnostic/proxy-level for action purposes.",
    ]
    summary = {
        "schema": "acl2_v101_trackR3_support_conditioned_edge_audit_diagnostic_v1",
        "track": "R3",
        "status": "complete_diagnostic_blocked",
        "gate_pass": False,
        "v100_anchor_edge_row_count": total_count,
        "joined_anchor_edge_row_count": joined_count,
        "joined_case_count": len(case_rows),
        "top_edge_rows_written": len(top_rows),
        "best_cue": best.get("cue_name", ""),
        "best_balanced_accuracy": best.get("balanced_accuracy", math.nan),
        "best_good_FPR": best.get("good_FPR", math.nan),
        "best_bad_recall": best.get("bad_recall", math.nan),
        "runtime_action_allowed": False,
        "control_gaps": [
            "query-head random margin unavailable for support-conditioned joined rows",
            "anchor-id rotation margin unavailable for support-conditioned joined rows",
            "semantic-label rotation margin unavailable for support-conditioned joined rows",
        ],
        "gate_checks": [
            {"gate": "joined_anchor_edge_rows_gt_0", "pass": joined_count > 0, "observed": joined_count},
            {"gate": "selected_positive_sequence_coverage_ge_3", "pass": False, "observed": best.get("selected_positive_sequence_coverage", "")},
            {"gate": "control_margins_available", "pass": False, "observed": False},
            {"gate": "runtime_action_allowed", "pass": False, "observed": False},
        ],
        "failure_lines": failure_lines,
    }
    write_rows(OUT_R3 / "support_conditioned_anchor_edge_top_rows.csv", top_rows)
    write_rows(OUT_R3 / "support_conditioned_anchor_edge_case_rows.csv", case_rows)
    write_rows(OUT_R3 / "false_positive_false_negative_rows.csv", details)
    write_rows(OUT_R3 / "edge_metric_summary.csv", metrics)
    summarize_track(
        OUT_R3,
        "R3_summary.json",
        summary,
        metrics,
        failure_lines,
        "R3 needs support-conditioned edge metrics with query-head/anchor-id/semantic controls and sequence-covered target/safe-good gates.",
        "Materialize query-head controls on the joined support-conditioned edge rows after expanding clean targets.",
    )
    return summary


def main() -> None:
    target_by_case, _ = target_maps()
    rows, rows_by_key = state_rows()
    summaries = {
        "N3": run_n3(rows, target_by_case),
        "C5": run_c5(rows, target_by_case),
        "F5": run_f5(rows, target_by_case),
        "R3": run_r3(rows_by_key, target_by_case),
    }
    print(json.dumps(summaries, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
