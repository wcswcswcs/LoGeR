#!/usr/bin/env python3
"""Build ACL2 v97-TF semantic scale evidence / gauge-safe memory artifacts.

This builder is conservative: it only consumes landed v96 artifacts and never
promotes a runtime action. Missing v97-only evidence is written as missing
reason rows instead of being filled with default values.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v97tf_semantic_scale_evidence_gauge_safe_memory_control")
V96_ROOT = Path("results/acl2_v96tf_vggt4d_informed_semantic_gauge_preserving_memory_control")
EPS = 1.0e-9


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"read_error": f"{type(exc).__name__}:{exc}", "path": str(path)}
    return payload if isinstance(payload, dict) else {"value": payload}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(fieldnames or [])
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def f(value: Any, default: float = math.nan) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() in {"", "nan", "None", "null"}:
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def b(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return {}
    return parsed if isinstance(parsed, dict) else {}


def margin_by_key(mapping: dict[str, Any], needle: str) -> float:
    values = [f(value) for key, value in mapping.items() if needle in key]
    values = finite_values(values)
    return min(values) if values else math.nan


def finite_values(values: list[float]) -> list[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def mean(values: list[float]) -> float:
    vals = finite_values(values)
    return float(sum(vals) / len(vals)) if vals else math.nan


def median(values: list[float]) -> float:
    vals = finite_values(values)
    return float(statistics.median(vals)) if vals else math.nan


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return math.nan
    xvals, yvals = zip(*pairs)
    mx = sum(xvals) / len(xvals)
    my = sum(yvals) / len(yvals)
    vx = sum((x - mx) ** 2 for x in xvals)
    vy = sum((y - my) ** 2 for y in yvals)
    if vx <= 0 or vy <= 0:
        return math.nan
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def rmse(values: list[float]) -> float:
    vals = finite_values(values)
    return math.sqrt(sum(value * value for value in vals) / len(vals)) if vals else math.nan


def stable_rank(seed: str, items: list[str]) -> list[str]:
    return sorted(items, key=lambda item: hashlib.sha256(f"{seed}:{item}".encode("utf-8")).hexdigest())


def best_threshold(values_by_case: dict[str, float], positive_cases: set[str], negative_cases: set[str], *, higher_bad: bool) -> dict[str, Any]:
    cases = sorted(positive_cases | negative_cases)
    values = [values_by_case.get(case, math.nan) for case in cases]
    labels = [1 if case in positive_cases else 0 for case in cases]
    thresholds = sorted({value for value in values if math.isfinite(value)})
    pos = sum(labels)
    neg = len(labels) - pos
    best: dict[str, Any] = {
        "balanced_accuracy": 0.0,
        "threshold": math.nan,
        "direction": "higher_bad" if higher_bad else "lower_bad",
        "tp": 0,
        "tn": 0,
        "pos": pos,
        "neg": neg,
    }
    for threshold in thresholds:
        preds = [1 if (value >= threshold if higher_bad else value <= threshold) else 0 for value in values]
        tp = sum(1 for pred, label in zip(preds, labels) if pred == 1 and label == 1)
        tn = sum(1 for pred, label in zip(preds, labels) if pred == 0 and label == 0)
        tpr = tp / pos if pos else 0.0
        tnr = tn / neg if neg else 0.0
        score = 0.5 * (tpr + tnr)
        if (score, tp + tn) > (best["balanced_accuracy"], best["tp"] + best["tn"]):
            best.update({"balanced_accuracy": score, "threshold": threshold, "tp": tp, "tn": tn})
    return best


def selected_from_threshold(values_by_case: dict[str, float], threshold: float, *, higher_bad: bool) -> set[str]:
    if not math.isfinite(threshold):
        return set()
    if higher_bad:
        return {case for case, value in values_by_case.items() if math.isfinite(value) and value >= threshold}
    return {case for case, value in values_by_case.items() if math.isfinite(value) and value <= threshold}


def signal(selected: set[str], positives: set[str], negatives: set[str]) -> float:
    recall = len(selected & positives) / len(positives) if positives else 0.0
    fpr = len(selected & negatives) / len(negatives) if negatives else 0.0
    return recall - fpr


def same_count_margin(selected: set[str], all_cases: list[str], positives: set[str], negatives: set[str], *, seeds: int = 64) -> float:
    actual = signal(selected, positives, negatives)
    controls = []
    for idx in range(seeds):
        control = set(stable_rank(f"same_count_{idx}", all_cases)[: len(selected)])
        controls.append(signal(control, positives, negatives))
    return actual - median(controls)


def sequence_margin(selected: set[str], all_cases: list[str], positives: set[str], negatives: set[str], seq_by_case: dict[str, str], *, seeds: int = 64) -> float:
    actual = signal(selected, positives, negatives)
    selected_counts: dict[str, int] = defaultdict(int)
    seq_cases: dict[str, list[str]] = defaultdict(list)
    for case in all_cases:
        seq_cases[seq_by_case.get(case, "")].append(case)
    for case in selected:
        selected_counts[seq_by_case.get(case, "")] += 1
    controls = []
    for idx in range(seeds):
        control: set[str] = set()
        for seq, cases in sorted(seq_cases.items()):
            control.update(stable_rank(f"seq_count_{idx}_{seq}", cases)[: selected_counts.get(seq, 0)])
        controls.append(signal(control, positives, negatives))
    return actual - median(controls)


def rotated_margin(values_by_case: dict[str, float], threshold: float, positives: set[str], negatives: set[str], *, higher_bad: bool) -> float:
    cases = sorted(positives | negatives)
    if len(cases) < 2 or not math.isfinite(threshold):
        return math.nan
    actual = signal(selected_from_threshold(values_by_case, threshold, higher_bad=higher_bad), positives, negatives)
    values = [values_by_case.get(case, math.nan) for case in cases]
    controls = []
    for shift in range(1, len(cases)):
        rotated = {case: values[(idx - shift) % len(values)] for idx, case in enumerate(cases)}
        controls.append(signal(selected_from_threshold(rotated, threshold, higher_bad=higher_bad), positives, negatives))
    return actual - median(controls)


def load_sources(v96_root: Path) -> dict[str, Any]:
    return {
        "v96_root": str(v96_root),
        "final": read_json(v96_root / "final_decision/final_decision.json"),
        "build": read_json(v96_root / "build_summary.json"),
        "stage7": read_json(v96_root / "stage7_full_validation/summary.json"),
        "stage7_rows": read_rows(v96_root / "stage7_full_validation/rows.csv"),
        "track_a": read_rows(v96_root / "trackA_case_response_atlas/rows.csv"),
        "semantic_mass": read_rows(v96_root / "trackJ_semantic_region_bank/per_case_region_mass.csv"),
        "track_d": read_json(v96_root / "trackD_read_gauge_preserving_action/summary.json"),
        "track_d_rows": read_rows(v96_root / "trackD_read_gauge_preserving_action/rows.csv"),
        "track_g_cue": read_json(v96_root / "trackG_read_cue_refinement/summary.json"),
        "track_g_feature": read_rows(v96_root / "trackG_read_cue_refinement/feature_metric_summary.csv"),
        "raw_qk": read_json(v96_root / "trackG_read_qk_carrier_localization/summary.json"),
        "raw_qk_pairwise": read_json(v96_root / "trackG_read_qk_carrier_localization_sampled_pairwise_followup/summary.json"),
        "raw_qk_rows": read_rows(v96_root / "trackG_read_qk_carrier_localization/per_case_layer_region_rows.csv"),
        "swa_decision": read_json(v96_root / "route_decisions/trackE_trackC_swa_route_decision.json"),
        "swa_rows": read_rows(v96_root / "trackE_swa_raw_transport_trace_swa_atlas_v1/trackE_swa_raw_transport_trace_case_rows.csv"),
        "ttt_summary": read_json(v96_root / "trackF_ttt_write_trace_replay_contribution_branch_scale_state_atlas_v1/analysis_fixed_pair/summary.json"),
        "ttt_rows": read_rows(v96_root / "trackF_ttt_write_trace_replay_contribution_branch_scale_state_atlas_v1/analysis_fixed_pair/case_rows.csv"),
    }


def _has_swa_raw_trace_payloads(trace_root: Path) -> bool:
    return trace_root.exists() and any(trace_root.glob("*/READ_NO_ACTION/swa_raw_transport_trace/*.pt"))


def swa_raw_trace_roots(root: Path, sources: dict[str, Any], *, include_v96: bool, include_v97_topk: bool) -> list[Path]:
    roots: list[Path] = []
    if include_v96:
        roots.append(Path(str(sources["v96_root"])) / "trackE_swa_raw_transport_trace_swa_atlas_v1")
    if include_v97_topk:
        key_stability_root = root / "trackE2_swa_key_stability_fallback_probe"
        topk_identity_root = root / "trackE2_swa_topk_identity_trace_probe"
        roots.append(key_stability_root if _has_swa_raw_trace_payloads(key_stability_root) else topk_identity_root)
    return roots


def swa_raw_trace_payload_paths(root: Path, sources: dict[str, Any], *, include_v96: bool, include_v97_topk: bool) -> list[Path]:
    roots = swa_raw_trace_roots(root, sources, include_v96=include_v96, include_v97_topk=include_v97_topk)
    paths: list[Path] = []
    for trace_root in roots:
        if trace_root.exists():
            paths.extend(sorted(trace_root.glob("*/READ_NO_ACTION/swa_raw_transport_trace/*.pt")))
    return paths


def action_labels(row: dict[str, str]) -> set[str]:
    return {label for label in row.get("action_response_labels", "").split(";") if label}


def case_maps(sources: dict[str, Any]) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], dict[str, str]]:
    atlas = {row["case_id"]: row for row in sources["track_a"] if row.get("case_id")}
    sem = {row["case_id"]: row for row in sources["semantic_mass"] if row.get("case_id")}
    seq = {case: row.get("seq", "") for case, row in atlas.items()}
    return atlas, sem, seq


def semantic_scores(case_id: str, atlas: dict[str, dict[str, str]], sem: dict[str, dict[str, str]]) -> dict[str, float]:
    row = sem.get(case_id, {})
    stable = f(row.get("STABLE_ANCHOR_token_mass"), 0.0)
    weak = f(row.get("WEAK_SCALE_CONTEXT_token_mass"), 0.0)
    veg = f(row.get("VEGETATION_REPETITIVE_token_mass"), 0.0)
    low = f(row.get("LOW_OBSERVABILITY_token_mass"), 0.0)
    dynamic = f(row.get("DYNAMIC_OBJECT_token_mass"), 0.0)
    boundary = f(row.get("OBJECT_BOUNDARY_BAND_token_mass"), 0.0)
    multimode = f(row.get("MULTIMODE_CONFLICT_token_mass"), 0.0)
    unknown = f(row.get("UNKNOWN_CONTEXT_token_mass"), 0.0)
    invalid = min(1.0, dynamic + boundary + low + multimode)
    lowobs = min(1.0, low + unknown)
    scale_observability = stable / (stable + weak + veg + low + unknown + EPS)
    temporal = 1.0 - min(1.0, dynamic + multimode)
    geometry = 1.0 - min(1.0, boundary + low)
    eligibility = stable * scale_observability * temporal * geometry * (1.0 - invalid) * (1.0 - lowobs)
    return {
        "stable_anchor_score": stable,
        "weak_context_ratio": weak + veg + low,
        "dynamic_boundary_ratio": dynamic + boundary,
        "semantic_risk_score": min(1.0, invalid + weak + veg),
        "scale_observability_score": scale_observability,
        "semantic_scale_eligibility_score": eligibility,
        "semantic_scale_risk_score": 1.0 - eligibility,
        "weak_over_stable": weak / (stable + EPS),
        "lowstuff_over_stable": (weak + veg + low) / (stable + EPS),
        "L2_error": f(atlas.get(case_id, {}).get("L2_intra_scale_cv")),
        "head_tail_error": f(atlas.get(case_id, {}).get("L2_head_tail_proxy_error")),
        "L3_handoff_transfer_penalty_proxy": f(atlas.get(case_id, {}).get("L3_handoff_transfer_penalty_proxy")),
        "L3_adjacent_log_scale_jump": f(atlas.get(case_id, {}).get("L3_adjacent_log_scale_jump")),
    }


def build_observatory(root: Path, sources: dict[str, Any]) -> dict[str, Any]:
    out = root / "trackI_scale_gauge_evidence_observatory_v2"
    atlas, sem, _seq = case_maps(sources)
    per_case_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for case_id, row in sorted(atlas.items()):
        labels = ";".join(sorted(action_labels(row)))
        scores = semantic_scores(case_id, atlas, sem)
        out_row = {
            "case_id": case_id,
            "seq": row.get("seq", ""),
            "prev_chunk": row.get("prev_chunk", ""),
            "curr_chunk": row.get("curr_chunk", ""),
            "case_label": row.get("case_label_offline_only", ""),
            "action_response_label": labels,
            "L1_local_sim3_ate": row.get("L1_local_sim3_ate", ""),
            "L1_local_sim3_scale": row.get("L1_local_sim3_scale", ""),
            "L2_intra_scale_cv": row.get("L2_intra_scale_cv", ""),
            "L2_head_tail_proxy_error": row.get("L2_head_tail_proxy_error", ""),
            "L3_handoff_transfer_penalty_proxy": row.get("L3_handoff_transfer_penalty_proxy", ""),
            "L3_adjacent_log_scale_jump": row.get("L3_adjacent_log_scale_jump", ""),
            "L4_future_error_1chunk": row.get("L4_future_error_1chunk", ""),
            "L4_future_error_3chunk": row.get("L4_future_error_3chunk", ""),
            "L4_future_error_5chunk": row.get("L4_future_error_5chunk", ""),
            "L5_scale_evidence_eligibility": scores["semantic_scale_eligibility_score"],
            "L5_scale_evidence_risk": scores["semantic_scale_risk_score"],
            "read_allowed": False,
            "transmit_allowed": False,
            "write_allowed": False,
            "source_artifact": "v96_trackA_and_semantic_region_bank",
        }
        per_case_rows.append(out_row)
        if not row.get("L0_ATE_full"):
            missing_rows.append({
                "case_id": case_id,
                "field": "L0_ATE_full",
                "missing_reason": "v96/v95 case atlas contains no promoted full-sequence L0 for this offline diagnostic row",
                "repair_attempt": "kept missing; Stage7 candidate rows parsed separately",
            })
    stage7_rows = []
    for row in sources["stage7_rows"]:
        stage7_rows.append({
            **row,
            "gauge_harm_score_proxy": max(0.0, f(row.get("delta_final_error_m"), 0.0))
            + max(0.0, f(row.get("rolling_worse_fraction_max"), 0.0) - 0.05)
            + max(0.0, abs(f(row.get("delta_yaw_rmse_deg"), 0.0)) - 2.0) / 100.0,
            "classification": "local_or_full_candidate_full_gate_failed"
            if not b(row.get("strict_full_gate_pass")) else "strict_full_gate_pass",
        })
    write_rows(out / "per_case_metrics.csv", per_case_rows)
    write_rows(out / "rows.csv", per_case_rows + stage7_rows)
    write_rows(out / "per_action_response.csv", [
        {"action_response_label": label, "count": count}
        for label, count in sorted(Counter(label for row in per_case_rows for label in row["action_response_label"].split(";") if label).items())
    ])
    write_rows(out / "scale_evidence_eligibility_rows.csv", [
        {**{"case_id": case_id}, **semantic_scores(case_id, atlas, sem)}
        for case_id in sorted(atlas)
    ])
    write_rows(out / "active_inactive_tradeoff_rows.csv", stage7_rows)
    write_rows(out / "full_sequence_gate_rows.csv", stage7_rows)
    write_rows(out / "latent_gauge_shift_rows.csv", [
        {
            "source": "v96_stage7_proxy",
            "available": False,
            "missing_reason": "READ before/after stable-anchor latent embeddings were not saved in v96; Stage7 final-error/rolling/yaw rows are proxy only",
        }
    ])
    write_rows(out / "metric_missing_reason.csv", missing_rows)
    write_rows(out / "visual_manifest.csv", [])
    summary = {
        "schema": "acl2_v97_scale_gauge_evidence_observatory_v2",
        "status": "complete",
        "source_root": str(V96_ROOT),
        "case_rows": len(per_case_rows),
        "stage7_candidate_rows": len(stage7_rows),
        "action_response_label_coverage": sum(1 for row in per_case_rows if row["action_response_label"]) / max(1, len(per_case_rows)),
        "stage7_candidates_have_L0_L4_tradeoff": bool(stage7_rows),
        "trackF_diagnostic_available": bool(sources["ttt_rows"]),
        "trackE_trackC_diagnostic_available": bool(sources["swa_rows"]),
        "metric_missing_rows": len(missing_rows),
        "gate_pass": bool(per_case_rows and stage7_rows and sources["ttt_rows"] and sources["swa_rows"]),
    }
    write_json(out / "summary.json", summary)
    write_text(out / "failure_report.md", "# Track I Observatory v2\n\nNo parser blocker found. Missing L0 rows are preserved in metric_missing_reason.csv; Stage7 candidates are parsed separately.")
    write_text(out / "what_would_have_to_be_true_to_pass.md", "# What Would Have To Be True\n\nAll action-touched rows need explicit labels, Stage7 candidates need full gate rows, and missing L0 values must remain marked missing unless a real full-sequence artifact exists.")
    return summary


def build_track_k(root: Path, sources: dict[str, Any]) -> dict[str, Any]:
    out = root / "trackK_semantic_scale_evidence_eligibility"
    atlas, sem, seq = case_maps(sources)
    read_rows: list[dict[str, Any]] = []
    read_values: dict[str, float] = {}
    read_weak: dict[str, float] = {}
    read_score_candidates: dict[str, dict[str, float]] = defaultdict(dict)
    positives: set[str] = set()
    negatives: set[str] = set()
    qk_rows_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for qk_row in sources.get("raw_qk_rows", []):
        if qk_row.get("case_id"):
            qk_rows_by_case[qk_row["case_id"]].append(qk_row)
    for case_id, row in sorted(atlas.items()):
        labels = action_labels(row)
        if "READ_LOCAL_BAD" in labels:
            positives.add(case_id)
        if "GOOD_PROTECTION" in labels:
            negatives.add(case_id)
        scores = semantic_scores(case_id, atlas, sem)
        qk_case_rows = qk_rows_by_case.get(case_id, [])
        internal_instability = mean([abs(f(qk_row.get("affected_mass_delta"))) for qk_row in qk_case_rows])
        raw_qk_weak_over_stable = mean([f(qk_row.get("weak_over_stable_attention_mass")) for qk_row in qk_case_rows])
        read_values[case_id] = scores["semantic_scale_risk_score"]
        read_weak[case_id] = scores["weak_over_stable"]
        read_score_candidates["semantic_scale_risk_score_with_scale_observability_repair"][case_id] = scores["semantic_scale_risk_score"]
        read_score_candidates["weak_over_stable"][case_id] = scores["weak_over_stable"]
        read_score_candidates["weak_context_ratio"][case_id] = scores["weak_context_ratio"]
        read_score_candidates["lowstuff_over_stable"][case_id] = scores["lowstuff_over_stable"]
        read_score_candidates["dynamic_boundary_ratio"][case_id] = scores["dynamic_boundary_ratio"]
        read_score_candidates["scale_observability_score"][case_id] = scores["scale_observability_score"]
        read_score_candidates["stable_anchor_deficit"][case_id] = 1.0 - scores["stable_anchor_score"]
        read_score_candidates["semantic_risk_score"][case_id] = scores["semantic_risk_score"]
        read_score_candidates["observability_context_risk_composite"][case_id] = (
            (1.0 - scores["scale_observability_score"])
            + scores["weak_context_ratio"]
            + scores["dynamic_boundary_ratio"]
        )
        read_score_candidates["stable_deficit_lowstuff_risk_composite"][case_id] = (
            (1.0 - scores["stable_anchor_score"])
            + scores["lowstuff_over_stable"]
            + scores["dynamic_boundary_ratio"]
        )
        if math.isfinite(internal_instability):
            read_score_candidates["raw_qk_attention_delta_abs_internal_instability"][case_id] = internal_instability
        if math.isfinite(raw_qk_weak_over_stable):
            read_score_candidates["raw_qk_weak_over_stable_attention_mass"][case_id] = raw_qk_weak_over_stable
        read_rows.append({
            "case_id": case_id,
            "seq": row.get("seq", ""),
            "is_read_bad": "READ_LOCAL_BAD" in labels,
            "is_good_control": "GOOD_PROTECTION" in labels,
            "raw_qk_attention_delta_abs_internal_instability": internal_instability if math.isfinite(internal_instability) else "",
            "raw_qk_weak_over_stable_attention_mass": raw_qk_weak_over_stable if math.isfinite(raw_qk_weak_over_stable) else "",
            "raw_qk_internal_instability_missing_reason": ""
            if qk_case_rows
            else "no per-case raw-QK rows in v96 raw_qk carrier localization artifact",
            **scores,
        })
    all_read = sorted(positives | negatives)
    read_score_directions = {
        "semantic_scale_risk_score_with_scale_observability_repair": True,
        "weak_over_stable": True,
        "weak_context_ratio": True,
        "lowstuff_over_stable": True,
        "dynamic_boundary_ratio": True,
        "scale_observability_score": False,
        "stable_anchor_deficit": True,
        "semantic_risk_score": True,
        "observability_context_risk_composite": True,
        "stable_deficit_lowstuff_risk_composite": True,
        "raw_qk_attention_delta_abs_internal_instability": True,
        "raw_qk_weak_over_stable_attention_mass": True,
    }
    stable_anchor_values = [row["stable_anchor_score"] for row in read_rows if math.isfinite(f(row.get("stable_anchor_score")))]
    weak_context_values = [row["weak_context_ratio"] for row in read_rows if math.isfinite(f(row.get("weak_context_ratio")))]
    instability_values = [f(row.get("raw_qk_attention_delta_abs_internal_instability")) for row in read_rows]
    stable_anchor_median = median([f(value) for value in stable_anchor_values])
    weak_context_median = median([f(value) for value in weak_context_values])
    instability_median = median(instability_values)

    def audit_read_score(name: str, values: dict[str, float], *, higher_bad: bool) -> tuple[dict[str, Any], set[str]]:
        finite_values_by_case = {
            case: value
            for case, value in values.items()
            if case in all_read and math.isfinite(value)
        }
        available_pos = {case for case in positives if case in finite_values_by_case}
        available_neg = {case for case in negatives if case in finite_values_by_case}
        best = best_threshold(finite_values_by_case, available_pos, available_neg, higher_bad=higher_bad) if available_pos and available_neg else {
            "balanced_accuracy": 0.0,
            "threshold": math.nan,
            "direction": "higher_bad" if higher_bad else "lower_bad",
            "tp": 0,
            "tn": 0,
            "pos": len(available_pos),
            "neg": len(available_neg),
        }
        selected = selected_from_threshold(finite_values_by_case, f(best.get("threshold")), higher_bad=higher_bad)
        l2_values = [semantic_scores(case, atlas, sem)["L2_error"] for case in all_read]
        head_tail_values = [semantic_scores(case, atlas, sem)["head_tail_error"] for case in all_read]
        score_values = [finite_values_by_case.get(case, math.nan) for case in all_read]
        corr_l2 = pearson(score_values, l2_values)
        corr_head_tail = pearson(score_values, head_tail_values)
        direction_correct = (corr_l2 > 0) if higher_bad else (corr_l2 < 0)
        missed_positive = positives - selected
        false_positive = negatives & selected
        selected_positive_seqs = sorted({seq.get(case, "") for case in selected & positives})
        missed_seq00_seq05 = sorted(case for case in missed_positive if seq.get(case, "") in {"00", "05"})
        coverage = len(selected_positive_seqs)
        good_fpr = len(false_positive) / max(1, len(available_neg))
        bad_recall = len(selected & available_pos) / max(1, len(available_pos))
        same_margin = same_count_margin(selected, sorted(available_pos | available_neg), available_pos, available_neg)
        rotation_margin = rotated_margin(finite_values_by_case, f(best.get("threshold")), available_pos, available_neg, higher_bad=higher_bad)
        gate = (
            bad_recall >= 0.70
            and good_fpr <= 0.25
            and coverage >= 3
            and same_margin >= 0.05
            and rotation_margin >= 0.05
            and direction_correct
        )
        return {
            "score_name": name,
            "direction": "higher_bad" if higher_bad else "lower_bad",
            "available_case_count": len(finite_values_by_case),
            "bad_recall": bad_recall,
            "good_FPR": good_fpr,
            "positive_sequence_coverage": coverage,
            "selected_positive_sequences": ";".join(selected_positive_seqs),
            "seq00_seq05_missed_positive_count": len(missed_seq00_seq05),
            "seq00_seq05_missed_positives": ";".join(missed_seq00_seq05),
            "missed_positive_count": len(missed_positive),
            "false_positive_count": len(false_positive),
            "global_same_count_margin": same_margin,
            "semantic_rotation_margin": rotation_margin,
            "corr_score_L2_error": corr_l2 if math.isfinite(corr_l2) else "",
            "corr_score_head_tail_error": corr_head_tail if math.isfinite(corr_head_tail) else "",
            "corr_direction_correct": direction_correct if math.isfinite(corr_l2) else False,
            "best_threshold": best.get("threshold", ""),
            "best_balanced_accuracy": best.get("balanced_accuracy", 0.0),
            "sequence_fragile": coverage < 3,
            "gate_pass": gate,
            "repair_attempt": "plan-directed proxy audit: split stable-anchor, weak-context, internal-instability, and scale-observability candidates without threshold-only sweep",
        }, selected

    read_proxy_audit_rows: list[dict[str, Any]] = []
    read_selected_by_score: dict[str, set[str]] = {}
    for score_name, values in read_score_candidates.items():
        row, selected = audit_read_score(score_name, values, higher_bad=read_score_directions[score_name])
        read_proxy_audit_rows.append(row)
        read_selected_by_score[score_name] = selected

    read_loso_rows: list[dict[str, Any]] = []
    read_sequences = sorted({seq.get(case, "") for case in all_read if seq.get(case, "")})
    for score_name, values in read_score_candidates.items():
        higher_bad = read_score_directions[score_name]
        finite_values_by_case = {
            case: value
            for case, value in values.items()
            if case in all_read and math.isfinite(value)
        }
        for heldout_seq in read_sequences:
            heldout_cases = {case for case in all_read if seq.get(case, "") == heldout_seq and case in finite_values_by_case}
            train_cases = {case for case in all_read if seq.get(case, "") != heldout_seq and case in finite_values_by_case}
            train_pos = positives & train_cases
            train_neg = negatives & train_cases
            heldout_pos = positives & heldout_cases
            heldout_neg = negatives & heldout_cases
            if not train_pos or not train_neg:
                read_loso_rows.append({
                    "score_name": score_name,
                    "heldout_seq": heldout_seq,
                    "available": False,
                    "missing_reason": "train split lacks positive or negative cases",
                    "train_positive_count": len(train_pos),
                    "train_negative_count": len(train_neg),
                    "heldout_positive_count": len(heldout_pos),
                    "heldout_negative_count": len(heldout_neg),
                })
                continue
            train_best = best_threshold(finite_values_by_case, train_pos, train_neg, higher_bad=higher_bad)
            selected = selected_from_threshold(finite_values_by_case, f(train_best.get("threshold")), higher_bad=higher_bad)
            heldout_recall = len(selected & heldout_pos) / len(heldout_pos) if heldout_pos else ""
            heldout_fpr = len(selected & heldout_neg) / len(heldout_neg) if heldout_neg else ""
            heldout_signal = (
                f(heldout_recall, 0.0) - f(heldout_fpr, 0.0)
                if heldout_pos and heldout_neg
                else ""
            )
            heldout_pass = (
                bool(heldout_pos)
                and bool(heldout_neg)
                and f(heldout_recall, 0.0) >= 0.70
                and f(heldout_fpr, 1.0) <= 0.25
            )
            read_loso_rows.append({
                "score_name": score_name,
                "heldout_seq": heldout_seq,
                "available": True,
                "direction": "higher_bad" if higher_bad else "lower_bad",
                "train_positive_count": len(train_pos),
                "train_negative_count": len(train_neg),
                "heldout_positive_count": len(heldout_pos),
                "heldout_negative_count": len(heldout_neg),
                "threshold_from_train": train_best.get("threshold", ""),
                "train_balanced_accuracy": train_best.get("balanced_accuracy", 0.0),
                "heldout_recall": heldout_recall,
                "heldout_good_FPR": heldout_fpr,
                "heldout_signal": heldout_signal,
                "heldout_pass": heldout_pass,
            })

    loso_available_rows = [row for row in read_loso_rows if b(row.get("available")) and row.get("heldout_positive_count") and row.get("heldout_negative_count")]
    loso_score_pass_counts = Counter(
        row.get("score_name", "")
        for row in loso_available_rows
        if b(row.get("heldout_pass"))
    )
    loso_scores_all_available = {
        score_name: [
            row for row in loso_available_rows
            if row.get("score_name") == score_name
        ]
        for score_name in read_score_candidates
    }
    loso_sequence_robust_scores = [
        score_name for score_name, rows_for_score in loso_scores_all_available.items()
        if rows_for_score and all(b(row.get("heldout_pass")) for row in rows_for_score)
    ]
    loso_best_score = max(
        read_score_candidates.keys(),
        key=lambda name: (
            loso_score_pass_counts.get(name, 0),
            mean([f(row.get("heldout_signal")) for row in loso_scores_all_available.get(name, [])]),
        ),
    ) if read_score_candidates else ""
    current_read_metrics = next(
        row for row in read_proxy_audit_rows
        if row["score_name"] == "semantic_scale_risk_score_with_scale_observability_repair"
    )
    best_read_proxy = max(
        read_proxy_audit_rows,
        key=lambda row: (
            b(row.get("gate_pass")),
            f(row.get("best_balanced_accuracy"), 0.0),
            f(row.get("bad_recall"), 0.0) - f(row.get("good_FPR"), 1.0),
            f(row.get("global_same_count_margin"), 0.0),
        ),
    )
    passing_read_proxy_rows = [row for row in read_proxy_audit_rows if b(row.get("gate_pass"))]
    read_metrics = {
        **current_read_metrics,
        "best_proxy_score_name": best_read_proxy.get("score_name", ""),
        "best_proxy_balanced_accuracy": best_read_proxy.get("best_balanced_accuracy", 0.0),
        "best_proxy_gate_pass": best_read_proxy.get("gate_pass", False),
        "passing_proxy_score_names": [row["score_name"] for row in passing_read_proxy_rows],
        "weak_over_stable_balanced_accuracy": next(
            (row["best_balanced_accuracy"] for row in read_proxy_audit_rows if row["score_name"] == "weak_over_stable"),
            0.0,
        ),
        "weak_over_stable_signal": signal(
            read_selected_by_score.get("weak_over_stable", set()),
            positives,
            negatives,
        ),
    }
    read_gate = bool(passing_read_proxy_rows)
    read_selected = read_selected_by_score["semantic_scale_risk_score_with_scale_observability_repair"]

    case_lookup = {row["case_id"]: row for row in read_rows}
    decomposition_rows = []
    for score_name, selected in read_selected_by_score.items():
        missed_positive = positives - selected
        false_positive = negatives & selected
        for bucket_name, cases in [("missed_positive", missed_positive), ("false_positive", false_positive)]:
            for case in sorted(cases):
                row = case_lookup.get(case, {})
                stable_anchor_too_low = math.isfinite(stable_anchor_median) and f(row.get("stable_anchor_score")) < stable_anchor_median
                weak_context_too_high = math.isfinite(weak_context_median) and f(row.get("weak_context_ratio")) > weak_context_median
                internal_instability_high = (
                    math.isfinite(instability_median)
                    and math.isfinite(f(row.get("raw_qk_attention_delta_abs_internal_instability")))
                    and f(row.get("raw_qk_attention_delta_abs_internal_instability")) > instability_median
                )
                decomposition_rows.append({
                    "score_name": score_name,
                    "case_id": case,
                    "seq": seq.get(case, ""),
                    "bucket": bucket_name,
                    "stable_anchor_too_low_vs_all_read_median": stable_anchor_too_low,
                    "weak_context_too_high_vs_all_read_median": weak_context_too_high,
                    "internal_instability_high_vs_available_raw_qk_median": internal_instability_high
                    if math.isfinite(f(row.get("raw_qk_attention_delta_abs_internal_instability")))
                    else "",
                    "internal_instability_missing_reason": row.get("raw_qk_internal_instability_missing_reason", ""),
                    "stable_anchor_score": row.get("stable_anchor_score", ""),
                    "weak_context_ratio": row.get("weak_context_ratio", ""),
                    "scale_observability_score": row.get("scale_observability_score", ""),
                    "raw_qk_attention_delta_abs_internal_instability": row.get("raw_qk_attention_delta_abs_internal_instability", ""),
                    "L2_error": row.get("L2_error", ""),
                    "head_tail_error": row.get("head_tail_error", ""),
                })

    missed = [{"case_id": case, "seq": seq.get(case, ""), "reason": "below current semantic-scale-risk selected threshold"} for case in sorted(positives - read_selected)]
    false_pos = [{"case_id": case, "seq": seq.get(case, ""), "reason": "good control selected by current semantic-scale-risk threshold"} for case in sorted(negatives & read_selected)]

    swa_rows: list[dict[str, Any]] = []
    swa_values: dict[str, float] = {}
    swa_pos: set[str] = set()
    swa_neg: set[str] = set()
    for row in sources["swa_rows"]:
        case_id = row.get("case_id", "")
        bucket = row.get("bucket", "")
        if bucket == "SWA_HANDOFF_GOOD_CONTROL":
            swa_neg.add(case_id)
        else:
            swa_pos.add(case_id)
        stable = f(row.get("trace_swa_raw_transport_stable_pair_mass_mean"), 0.0)
        unreliable = f(row.get("trace_swa_raw_transport_unreliable_pair_mass_mean"), 0.0)
        residual = f(row.get("trace_swa_raw_transport_feature_residual_mean"), 0.0)
        score = residual + unreliable - stable
        swa_values[case_id] = score
        swa_rows.append({
            **row,
            "handoff_scale_evidence_risk_score": score,
            "L3_handoff_transfer_penalty_proxy": atlas.get(case_id, {}).get("L3_handoff_transfer_penalty_proxy", ""),
        })
    all_swa = sorted(swa_pos | swa_neg)
    swa_best = best_threshold(swa_values, swa_pos, swa_neg, higher_bad=True)
    swa_selected = selected_from_threshold(swa_values, swa_best["threshold"], higher_bad=True)
    swa_l3 = [f(atlas.get(case, {}).get("L3_handoff_transfer_penalty_proxy")) for case in all_swa]
    swa_scores = [swa_values.get(case, math.nan) for case in all_swa]
    swa_metrics = {
        "score_name": "feature_residual_plus_unreliable_minus_stable",
        "best_balanced_accuracy": swa_best["balanced_accuracy"],
        "L3_handoff_transfer_penalty_abs_corr": abs(pearson(swa_scores, swa_l3)),
        "same_count_margin": same_count_margin(swa_selected, all_swa, swa_pos, swa_neg),
        "sequence_margin": sequence_margin(swa_selected, all_swa, swa_pos, swa_neg, seq),
        "stable_group_nonempty_frac": sources["swa_decision"].get("stable_group_nonempty_frac", 0.0),
        "fallback_diagnostic_only": False,
    }
    swa_gate = (
        swa_metrics["best_balanced_accuracy"] >= 0.70
        and swa_metrics["L3_handoff_transfer_penalty_abs_corr"] >= 0.30
        and swa_metrics["same_count_margin"] >= 0.05
        and swa_metrics["sequence_margin"] >= 0.05
        and f(swa_metrics["stable_group_nonempty_frac"], 0.0) >= 0.50
    )
    swa_trace_case_acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    swa_trace_read_errors: list[dict[str, Any]] = []
    swa_trace_payload_count = 0
    raw_swa_roots = swa_raw_trace_roots(root, sources, include_v96=False, include_v97_topk=True)
    if not any(_has_swa_raw_trace_payloads(trace_root) for trace_root in raw_swa_roots):
        raw_swa_roots = swa_raw_trace_roots(root, sources, include_v96=True, include_v97_topk=False)
    raw_swa_source_roots = [str(path) for path in raw_swa_roots]
    try:
        import torch  # noqa: PLC0415

        for trace_root in raw_swa_roots:
            if not trace_root.exists():
                continue
            for path in sorted(trace_root.glob("*/READ_NO_ACTION/swa_raw_transport_trace/*.pt")):
                case_id = path.parents[2].name
                try:
                    payload = torch.load(path, map_location="cpu")
                except Exception as exc:  # noqa: BLE001
                    swa_trace_read_errors.append({"path": str(path), "error": f"{type(exc).__name__}:{exc}"})
                    continue
                if not isinstance(payload, dict):
                    swa_trace_read_errors.append({"path": str(path), "error": f"unexpected_payload_type:{type(payload).__name__}"})
                    continue
                swa_trace_payload_count += 1
                stable_strict = f(payload.get("stable_pair_strict_tokens"), 0.0)
                fallback_used = 1.0 if bool(payload.get("stable_pair_fallback_used")) else 0.0
                stable_tokens = f(payload.get("stable_pair_tokens"), 0.0)
                semantic_lowd_tokens = f(payload.get("stable_pair_semantic_lowd_tokens"), 0.0)
                lowd_nonunreliable_tokens = f(payload.get("stable_pair_lowd_nonunreliable_tokens"), 0.0)
                for key, value in [
                    ("strict_stable_nonempty", 1.0 if stable_strict > 0 else 0.0),
                    ("fallback_used", fallback_used),
                    ("stable_pair_strict_tokens", stable_strict),
                    ("stable_pair_tokens", stable_tokens),
                    ("stable_pair_semantic_lowd_tokens", semantic_lowd_tokens),
                    ("stable_pair_lowd_nonunreliable_tokens", lowd_nonunreliable_tokens),
                    ("cache_k_stability_mean", f(payload.get("cache_k_stability_mean"))),
                    ("cache_v_stability_mean", f(payload.get("cache_v_stability_mean"))),
                    ("qk_similarity_mean", f(payload.get("qk_similarity_mean"))),
                    ("qk_similarity_max_mean", f(payload.get("qk_similarity_max_mean"))),
                    ("feature_transport_residual_mean", f(payload.get("feature_transport_residual_mean"))),
                    ("route_entropy_mean", f(payload.get("route_entropy_mean"))),
                ]:
                    if math.isfinite(value):
                        swa_trace_case_acc[case_id][key].append(value)
    except Exception as exc:  # noqa: BLE001
        swa_trace_read_errors.append({"path": ";".join(raw_swa_source_roots), "error": f"{type(exc).__name__}:{exc}"})

    swa_topk_case_acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    swa_topk_payload_count = 0
    try:
        import torch  # noqa: PLC0415

        for path in swa_raw_trace_payload_paths(root, sources, include_v96=False, include_v97_topk=True):
            case_id = path.parents[2].name
            try:
                payload = torch.load(path, map_location="cpu")
            except Exception as exc:  # noqa: BLE001
                swa_trace_read_errors.append({
                    "path": str(path),
                    "source": "v97_topk_identity_probe",
                    "error": f"{type(exc).__name__}:{exc}",
                })
                continue
            if not isinstance(payload, dict):
                swa_trace_read_errors.append({
                    "path": str(path),
                    "source": "v97_topk_identity_probe",
                    "error": f"unexpected_payload_type:{type(payload).__name__}",
                })
                continue
            if not bool(payload.get("topk_identity_available")):
                continue
            swa_topk_payload_count += 1
            for key, value in [
                ("top1_cache_index_unique_frac_mean", f(payload.get("top1_cache_index_unique_frac_mean"))),
                ("top1_cache_frame_unique_frac_mean", f(payload.get("top1_cache_frame_unique_frac_mean"))),
                ("top1_cache_index_switch_rate_mean", f(payload.get("top1_cache_index_switch_rate_mean"))),
                ("top1_cache_frame_switch_rate_mean", f(payload.get("top1_cache_frame_switch_rate_mean"))),
                ("top1_same_frame_frac_mean", f(payload.get("top1_same_frame_frac_mean"))),
                ("topk_query_frame_hit_frac_mean", f(payload.get("topk_query_frame_hit_frac_mean"))),
                ("topk_same_frame_frac_mean", f(payload.get("topk_same_frame_frac_mean"))),
                ("top1_abs_frame_delta_mean", f(payload.get("top1_abs_frame_delta_mean"))),
            ]:
                if math.isfinite(value):
                    swa_topk_case_acc[case_id][key].append(value)
    except Exception as exc:  # noqa: BLE001
        swa_trace_read_errors.append({
            "path": str(root / "trackE2_swa_topk_identity_trace_probe"),
            "source": "v97_topk_identity_probe",
            "error": f"{type(exc).__name__}:{exc}",
        })

    swa_strict_rows: list[dict[str, Any]] = []
    swa_cache_values_by_metric: dict[str, dict[str, float]] = defaultdict(dict)
    swa_cache_specs = {
        "cache_k_stability_lower_bad": ("cache_k_stability_mean", False),
        "cache_v_stability_lower_bad": ("cache_v_stability_mean", False),
        "qk_similarity_mean_lower_bad": ("qk_similarity_mean", False),
        "qk_similarity_max_lower_bad": ("qk_similarity_max_mean", False),
        "feature_residual_higher_bad": ("feature_transport_residual_mean", True),
        "route_entropy_higher_bad": ("route_entropy_mean", True),
        "topk_top1_index_unique_frac_higher_bad": ("top1_cache_index_unique_frac_mean", True),
        "topk_top1_frame_unique_frac_higher_bad": ("top1_cache_frame_unique_frac_mean", True),
        "topk_top1_index_switch_rate_higher_bad": ("top1_cache_index_switch_rate_mean", True),
        "topk_top1_frame_switch_rate_higher_bad": ("top1_cache_frame_switch_rate_mean", True),
        "topk_top1_same_frame_frac_lower_bad": ("top1_same_frame_frac_mean", False),
        "topk_query_frame_hit_frac_lower_bad": ("topk_query_frame_hit_frac_mean", False),
        "topk_same_frame_frac_lower_bad": ("topk_same_frame_frac_mean", False),
        "topk_top1_abs_frame_delta_higher_bad": ("top1_abs_frame_delta_mean", True),
    }
    for case_id in all_swa:
        acc = swa_trace_case_acc.get(case_id, {})
        topk_acc = swa_topk_case_acc.get(case_id, {})
        strict_frac = mean(acc.get("strict_stable_nonempty", []))
        fallback_frac = mean(acc.get("fallback_used", []))
        row = {
            "case_id": case_id,
            "seq": seq.get(case_id, ""),
            "bucket": "SWA_HANDOFF_GOOD_CONTROL" if case_id in swa_neg else "SWA_HANDOFF_NON_GOOD",
            "payload_count": len(acc.get("fallback_used", [])),
            "strict_stable_nonempty_frac": strict_frac if math.isfinite(strict_frac) else "",
            "fallback_used_frac": fallback_frac if math.isfinite(fallback_frac) else "",
            "stable_pair_strict_tokens_mean": mean(acc.get("stable_pair_strict_tokens", [])),
            "stable_pair_tokens_mean": mean(acc.get("stable_pair_tokens", [])),
            "stable_pair_semantic_lowd_tokens_mean": mean(acc.get("stable_pair_semantic_lowd_tokens", [])),
            "stable_pair_lowd_nonunreliable_tokens_mean": mean(acc.get("stable_pair_lowd_nonunreliable_tokens", [])),
            "cache_k_stability_mean": mean(acc.get("cache_k_stability_mean", [])),
            "cache_v_stability_mean": mean(acc.get("cache_v_stability_mean", [])),
            "qk_similarity_mean": mean(acc.get("qk_similarity_mean", [])),
            "qk_similarity_max_mean": mean(acc.get("qk_similarity_max_mean", [])),
            "feature_transport_residual_mean": mean(acc.get("feature_transport_residual_mean", [])),
            "route_entropy_mean": mean(acc.get("route_entropy_mean", [])),
            "topk_identity_payload_count": len(topk_acc.get("top1_cache_index_unique_frac_mean", [])),
            "top1_cache_index_unique_frac_mean": mean(topk_acc.get("top1_cache_index_unique_frac_mean", [])),
            "top1_cache_frame_unique_frac_mean": mean(topk_acc.get("top1_cache_frame_unique_frac_mean", [])),
            "top1_cache_index_switch_rate_mean": mean(topk_acc.get("top1_cache_index_switch_rate_mean", [])),
            "top1_cache_frame_switch_rate_mean": mean(topk_acc.get("top1_cache_frame_switch_rate_mean", [])),
            "top1_same_frame_frac_mean": mean(topk_acc.get("top1_same_frame_frac_mean", [])),
            "topk_query_frame_hit_frac_mean": mean(topk_acc.get("topk_query_frame_hit_frac_mean", [])),
            "topk_same_frame_frac_mean": mean(topk_acc.get("topk_same_frame_frac_mean", [])),
            "top1_abs_frame_delta_mean": mean(topk_acc.get("top1_abs_frame_delta_mean", [])),
            "L3_handoff_transfer_penalty_proxy": atlas.get(case_id, {}).get("L3_handoff_transfer_penalty_proxy", ""),
        }
        swa_strict_rows.append(row)
        for metric, (field, _higher_bad) in swa_cache_specs.items():
            value = f(row.get(field))
            if math.isfinite(value):
                swa_cache_values_by_metric[metric][case_id] = value
    strict_case_frac = mean([
        1.0 if f(row.get("strict_stable_nonempty_frac"), 0.0) > 0.0 else 0.0
        for row in swa_strict_rows
    ])
    fallback_case_frac = mean([
        1.0 if f(row.get("fallback_used_frac"), 0.0) > 0.0 else 0.0
        for row in swa_strict_rows
    ])
    swa_cache_rows: list[dict[str, Any]] = []
    for metric, (_field, higher_bad) in swa_cache_specs.items():
        values = swa_cache_values_by_metric.get(metric, {})
        best = best_threshold(values, swa_pos, swa_neg, higher_bad=higher_bad)
        selected = selected_from_threshold(values, f(best.get("threshold")), higher_bad=higher_bad)
        l3 = [f(atlas.get(case, {}).get("L3_handoff_transfer_penalty_proxy")) for case in all_swa]
        xs = [values.get(case, math.nan) for case in all_swa]
        same_margin = same_count_margin(selected, all_swa, swa_pos, swa_neg)
        seq_margin = sequence_margin(selected, all_swa, swa_pos, swa_neg, seq)
        abs_corr = abs(pearson(xs, l3))
        internal_gate = (
            len(values) == len(all_swa)
            and best["balanced_accuracy"] >= 0.70
            and abs_corr >= 0.30
            and same_margin >= 0.05
            and seq_margin >= 0.05
        )
        strict_gate = f(strict_case_frac, 0.0) >= 0.50
        eligibility_gate = internal_gate and strict_gate
        swa_cache_rows.append({
            "score_name": metric,
            "direction": "higher_bad" if higher_bad else "lower_bad",
            "available_case_count": len(values),
            "best_balanced_accuracy": best["balanced_accuracy"],
            "threshold": best["threshold"],
            "L3_handoff_transfer_penalty_abs_corr": abs_corr,
            "same_count_margin": same_margin,
            "sequence_margin": seq_margin,
            "strict_stable_nonempty_case_frac": strict_case_frac,
            "fallback_used_case_frac": fallback_case_frac,
            "internal_cache_carrier_gate_pass": internal_gate,
            "strict_stable_gate_pass": strict_gate,
            "eligibility_gate_pass": eligibility_gate,
            "diagnostic_only_reason": ""
            if eligibility_gate
            else (
                "strict stable SWA group absent/fallback-dominated; cache carrier is diagnostic-only"
                if internal_gate and not strict_gate
                else "cache carrier did not pass internal BA/correlation/margin gate"
            ),
            "selected_cases": ";".join(sorted(selected)),
            "false_positive_cases": ";".join(sorted(selected & swa_neg)),
            "missed_positive_cases": ";".join(sorted(swa_pos - selected)),
        })
    passed_swa_cache_internal = [row for row in swa_cache_rows if b(row.get("internal_cache_carrier_gate_pass"))]
    passed_swa_cache_eligible = [row for row in swa_cache_rows if b(row.get("eligibility_gate_pass"))]
    best_swa_cache = max(
        swa_cache_rows,
        key=lambda row: (
            b(row.get("eligibility_gate_pass")),
            b(row.get("internal_cache_carrier_gate_pass")),
            f(row.get("best_balanced_accuracy"), 0.0),
            f(row.get("L3_handoff_transfer_penalty_abs_corr"), 0.0),
        ),
    ) if swa_cache_rows else {}
    swa_metrics.update({
        "strict_stable_nonempty_case_frac_from_raw_trace": strict_case_frac,
        "fallback_used_case_frac_from_raw_trace": fallback_case_frac,
        "raw_trace_source_roots": raw_swa_source_roots,
        "raw_trace_payload_count": swa_trace_payload_count,
        "raw_trace_read_error_count": len(swa_trace_read_errors),
        "topk_identity_payload_count": swa_topk_payload_count,
        "topk_identity_case_count": len(swa_topk_case_acc),
        "cache_carrier_candidate_count": len(swa_cache_rows),
        "cache_carrier_internal_gate_pass_count": len(passed_swa_cache_internal),
        "cache_carrier_eligibility_gate_pass_count": len(passed_swa_cache_eligible),
        "best_cache_carrier_score_name": best_swa_cache.get("score_name", ""),
        "best_cache_carrier_internal_gate_pass": best_swa_cache.get("internal_cache_carrier_gate_pass", False),
        "best_cache_carrier_eligibility_gate_pass": best_swa_cache.get("eligibility_gate_pass", False),
        "fallback_diagnostic_only": f(strict_case_frac, 0.0) < 0.50 and bool(passed_swa_cache_internal),
    })

    ttt_rows: list[dict[str, Any]] = []
    risk_cases: set[str] = set()
    good_cases: set[str] = set()
    read_csv_rows = globals()["read_rows"]
    f2_root = root / "trackF2_ttt_stable_anchor_retention_missing_good_write"
    f2_retention_rows = read_csv_rows(f2_root / "ttt_retention_rows.csv")
    f2_audit_rows = read_csv_rows(f2_root / "retention_definition_audit_rows.csv")
    f2_by_case = {row.get("case_id", ""): row for row in f2_retention_rows if row.get("case_id")}
    for row in sources["ttt_rows"]:
        case_id = row.get("case_id", "")
        if row.get("bucket") == "ttt_write_risk":
            risk_cases.add(case_id)
        if row.get("bucket") == "good_control":
            good_cases.add(case_id)
        stable_write = f(row.get("persistent_write_mass"))
        transient = f(row.get("transient_write_mass"))
        no_write = f(row.get("no_write_mass"))
        uniformity_proxy = f(row.get("write_mass_mean"))
        missing_good = stable_write - max(transient, no_write)
        exact_row = f2_by_case.get(case_id, {})
        retention_value = f(exact_row.get("stable_anchor_retention"))
        ttt_row = {
            **row,
            "stable_anchor_write_eligibility_proxy": stable_write,
            "missing_good_write_score_proxy": missing_good,
            "write_uniformity_score_proxy": uniformity_proxy,
            "stable_anchor_retention_score": retention_value if math.isfinite(retention_value) else "",
            "stable_anchor_residual_score": exact_row.get("stable_anchor_residual", ""),
            "stable_anchor_write_prior_exact": exact_row.get("stable_anchor_write_prior_exact", ""),
            "exact_prior_missing_good_write_score": exact_row.get("exact_prior_missing_good_write_score", ""),
            "exact_prior_write_uniformity_score": exact_row.get("exact_prior_write_uniformity_score", ""),
            "exact_prior_risk_write_mass": exact_row.get("exact_prior_risk_write_mass", ""),
            "exact_retention_chunk_count": exact_row.get("exact_retention_chunk_count", ""),
            "exact_write_map_chunk_count": exact_row.get("exact_write_map_chunk_count", ""),
            "retention_missing_reason": ""
            if math.isfinite(retention_value)
            else "exact F2 stable-anchor retention not available; v96 proxy rows are diagnostic-only",
        }
        ttt_rows.append(ttt_row)
    risk_stable = [f(row["stable_anchor_write_eligibility_proxy"]) for row in ttt_rows if row.get("bucket") == "ttt_write_risk"]
    good_stable = [f(row["stable_anchor_write_eligibility_proxy"]) for row in ttt_rows if row.get("bucket") == "good_control"]
    risk_uniform = [f(row["write_uniformity_score_proxy"]) for row in ttt_rows if row.get("bucket") == "ttt_write_risk"]
    good_uniform = [f(row["write_uniformity_score_proxy"]) for row in ttt_rows if row.get("bucket") == "good_control"]
    f2_audit_by_metric = {row.get("metric", ""): row for row in f2_audit_rows if row.get("metric")}

    def audit_bool(row: dict[str, Any], key: str) -> bool:
        return b(row.get(key, False))

    retention_audit = f2_audit_by_metric.get("stable_anchor_retention_exact", {})
    missing_good_audit = f2_audit_by_metric.get("exact_prior_missing_good_write_score", {})
    uniformity_audit = f2_audit_by_metric.get("exact_prior_write_uniformity_score", {})
    stable_mass_audit = f2_audit_by_metric.get("exact_prior_stable_anchor_write_mass", {})
    ttt_exact_retention_gate = audit_bool(retention_audit, "separation_gate_pass")
    ttt_exact_missing_good_margin = f(missing_good_audit.get("risk_minus_good"))
    ttt_exact_missing_good_fpr = f(missing_good_audit.get("good_fpr"), 1.0)
    ttt_exact_missing_good_gate = (
        math.isfinite(ttt_exact_missing_good_margin)
        and ttt_exact_missing_good_margin >= 0.05
        and ttt_exact_missing_good_fpr <= 0.25
    )
    ttt_exact_uniformity_margin = f(uniformity_audit.get("risk_minus_good"))
    ttt_exact_uniformity_fpr = f(uniformity_audit.get("good_fpr"), 1.0)
    ttt_exact_uniformity_gate = (
        math.isfinite(ttt_exact_uniformity_margin)
        and ttt_exact_uniformity_margin >= 0.05
        and ttt_exact_uniformity_fpr <= 0.25
    )
    ttt_exact_stable_mass_margin = f(stable_mass_audit.get("risk_minus_good"))
    ttt_exact_stable_mass_fpr = f(stable_mass_audit.get("good_fpr"), 1.0)
    ttt_exact_stable_mass_gate = (
        math.isfinite(ttt_exact_stable_mass_margin)
        and ttt_exact_stable_mass_margin <= -0.05
        and ttt_exact_stable_mass_fpr <= 0.25
    )
    ttt_exact_available = bool(f2_retention_rows) and bool(f2_audit_rows)
    ttt_exact_gate = (
        ttt_exact_retention_gate
        or ttt_exact_missing_good_gate
        or ttt_exact_uniformity_gate
        or ttt_exact_stable_mass_gate
    )
    ttt_exact_audit_rows = [
        {
            "metric": "stable_anchor_retention_exact",
            "source_metric": retention_audit.get("metric", ""),
            "risk_minus_good": retention_audit.get("risk_minus_good", ""),
            "balanced_accuracy": retention_audit.get("balanced_accuracy", ""),
            "good_fpr": retention_audit.get("good_fpr", ""),
            "trackk_gate_component_pass": ttt_exact_retention_gate,
            "gate_reason": "retention must separate risk/good with BA>=0.70, good FPR<=0.25, and risk lower by plan/F2 margin",
        },
        {
            "metric": "exact_prior_missing_good_write_score",
            "source_metric": missing_good_audit.get("metric", ""),
            "risk_minus_good": missing_good_audit.get("risk_minus_good", ""),
            "balanced_accuracy": missing_good_audit.get("balanced_accuracy", ""),
            "good_fpr": missing_good_audit.get("good_fpr", ""),
            "trackk_gate_component_pass": ttt_exact_missing_good_gate,
            "gate_reason": "TrackK requires missing-good-write score risk-good margin >=0.05 and good FPR<=0.25",
        },
        {
            "metric": "exact_prior_write_uniformity_score",
            "source_metric": uniformity_audit.get("metric", ""),
            "risk_minus_good": uniformity_audit.get("risk_minus_good", ""),
            "balanced_accuracy": uniformity_audit.get("balanced_accuracy", ""),
            "good_fpr": uniformity_audit.get("good_fpr", ""),
            "trackk_gate_component_pass": ttt_exact_uniformity_gate,
            "gate_reason": "TrackK requires write uniformity higher in risk by >=0.05 and good FPR<=0.25",
        },
        {
            "metric": "exact_prior_stable_anchor_write_mass",
            "source_metric": stable_mass_audit.get("metric", ""),
            "risk_minus_good": stable_mass_audit.get("risk_minus_good", ""),
            "balanced_accuracy": stable_mass_audit.get("balanced_accuracy", ""),
            "good_fpr": stable_mass_audit.get("good_fpr", ""),
            "trackk_gate_component_pass": ttt_exact_stable_mass_gate,
            "gate_reason": "F2-compatible stable write mass requires risk-good margin <=-0.05 and good FPR<=0.25",
        },
    ]
    ttt_metrics = {
        "risk_case_count": len(risk_cases),
        "good_control_count": len(good_cases),
        "stable_anchor_write_mass_risk_minus_good": median(risk_stable) - median(good_stable),
        "write_uniformity_risk_minus_good": median(risk_uniform) - median(good_uniform),
        "condition_source": sources["ttt_summary"].get("condition_map_source", "replay_contribution"),
        "replay_runtime_eligible": sources["ttt_summary"].get("replay_runtime_eligible", False),
        "proxy_runtime_eligible": sources["ttt_summary"].get("proxy_runtime_eligible", False),
        "exact_f2_rows_available": ttt_exact_available,
        "exact_retention_case_count": len(f2_retention_rows),
        "exact_retention_gate_pass": ttt_exact_retention_gate,
        "exact_missing_good_write_gate_pass": ttt_exact_missing_good_gate,
        "exact_write_uniformity_gate_pass": ttt_exact_uniformity_gate,
        "exact_stable_anchor_write_mass_gate_pass": ttt_exact_stable_mass_gate,
        "exact_prior_missing_good_write_score_risk_minus_good": ttt_exact_missing_good_margin
        if math.isfinite(ttt_exact_missing_good_margin)
        else "",
        "exact_prior_write_uniformity_score_risk_minus_good": ttt_exact_uniformity_margin
        if math.isfinite(ttt_exact_uniformity_margin)
        else "",
        "exact_prior_stable_anchor_write_mass_risk_minus_good": ttt_exact_stable_mass_margin
        if math.isfinite(ttt_exact_stable_mass_margin)
        else "",
    }
    ttt_gate = bool(
        ttt_exact_gate
        if ttt_exact_available
        else (
            (
                ttt_metrics["stable_anchor_write_mass_risk_minus_good"] <= -0.05
                or ttt_metrics["write_uniformity_risk_minus_good"] >= 0.05
            )
            and not ttt_metrics["replay_runtime_eligible"]
            and ttt_metrics["condition_source"] == "exact_token"
        )
    )
    swa_cache_gate = bool(passed_swa_cache_eligible)
    write_rows(out / "read_eligibility_rows.csv", read_rows)
    write_rows(out / "read_proxy_audit_rows.csv", read_proxy_audit_rows)
    write_rows(out / "read_loso_sequence_audit_rows.csv", read_loso_rows)
    write_rows(out / "read_failure_decomposition_rows.csv", decomposition_rows)
    write_rows(out / "swa_eligibility_rows.csv", swa_rows)
    write_rows(out / "swa_strict_stable_fallback_audit_rows.csv", swa_strict_rows)
    write_rows(out / "swa_cache_carrier_eligibility_rows.csv", swa_cache_rows)
    write_rows(out / "swa_raw_trace_read_errors.csv", swa_trace_read_errors)
    write_rows(out / "ttt_eligibility_rows.csv", ttt_rows)
    write_rows(out / "ttt_exact_eligibility_audit_rows.csv", ttt_exact_audit_rows)
    write_rows(out / "cue_control_metrics.csv", [
        {"track": "READ", **read_metrics, "gate_pass": read_gate},
        {"track": "SWA", **swa_metrics, "gate_pass": swa_gate},
        {
            "track": "SWA_CACHE_CARRIER",
            "score_name": best_swa_cache.get("score_name", ""),
            "best_balanced_accuracy": best_swa_cache.get("best_balanced_accuracy", ""),
            "L3_handoff_transfer_penalty_abs_corr": best_swa_cache.get("L3_handoff_transfer_penalty_abs_corr", ""),
            "same_count_margin": best_swa_cache.get("same_count_margin", ""),
            "sequence_margin": best_swa_cache.get("sequence_margin", ""),
            "strict_stable_nonempty_case_frac_from_raw_trace": strict_case_frac,
            "fallback_used_case_frac_from_raw_trace": fallback_case_frac,
            "raw_trace_source_roots": ";".join(raw_swa_source_roots),
            "raw_trace_payload_count": swa_trace_payload_count,
            "raw_trace_read_error_count": len(swa_trace_read_errors),
            "topk_identity_payload_count": swa_topk_payload_count,
            "topk_identity_case_count": len(swa_topk_case_acc),
            "cache_carrier_candidate_count": len(swa_cache_rows),
            "cache_carrier_internal_gate_pass_count": len(passed_swa_cache_internal),
            "cache_carrier_eligibility_gate_pass_count": len(passed_swa_cache_eligible),
            "best_cache_carrier_score_name": best_swa_cache.get("score_name", ""),
            "best_cache_carrier_internal_gate_pass": best_swa_cache.get("internal_cache_carrier_gate_pass", False),
            "best_cache_carrier_eligibility_gate_pass": best_swa_cache.get("eligibility_gate_pass", False),
            "fallback_diagnostic_only": f(strict_case_frac, 0.0) < 0.50 and bool(passed_swa_cache_internal),
            "gate_pass": swa_cache_gate,
        },
        {"track": "TTT", **ttt_metrics, "gate_pass": ttt_gate},
    ])
    write_rows(out / "missed_cases.csv", missed)
    write_rows(out / "false_positive_cases.csv", false_pos)
    any_gate = read_gate or swa_gate or swa_cache_gate or ttt_gate
    summary = {
        "schema": "acl2_v97_trackK_semantic_scale_evidence_eligibility_v1",
        "status": "complete",
        "read_gate_pass": read_gate,
        "swa_gate_pass": swa_gate,
        "swa_cache_eligibility_gate_pass": swa_cache_gate,
        "swa_cache_internal_diagnostic_gate_pass": bool(passed_swa_cache_internal),
        "ttt_gate_pass": ttt_gate,
        "any_eligibility_cue_gate_pass": any_gate,
        "read_metrics": read_metrics,
        "swa_metrics": swa_metrics,
        "swa_cache_metrics": {
            "strict_stable_nonempty_case_frac": strict_case_frac,
            "fallback_used_case_frac": fallback_case_frac,
            "raw_trace_source_roots": raw_swa_source_roots,
            "raw_trace_payload_count": swa_trace_payload_count,
            "raw_trace_read_error_count": len(swa_trace_read_errors),
            "topk_identity_payload_count": swa_topk_payload_count,
            "topk_identity_case_count": len(swa_topk_case_acc),
            "cache_carrier_candidate_count": len(swa_cache_rows),
            "cache_carrier_internal_gate_pass_count": len(passed_swa_cache_internal),
            "cache_carrier_eligibility_gate_pass_count": len(passed_swa_cache_eligible),
            "best_cache_carrier_score_name": best_swa_cache.get("score_name", ""),
            "best_cache_carrier_internal_gate_pass": best_swa_cache.get("internal_cache_carrier_gate_pass", False),
            "best_cache_carrier_eligibility_gate_pass": best_swa_cache.get("eligibility_gate_pass", False),
            "diagnostic_only_reason": best_swa_cache.get("diagnostic_only_reason", ""),
        },
        "ttt_metrics": ttt_metrics,
        "read_proxy_candidate_count": len(read_proxy_audit_rows),
        "read_passing_proxy_score_names": [row["score_name"] for row in passing_read_proxy_rows],
        "read_loso_sequence_audit_rows": len(read_loso_rows),
        "read_loso_sequence_robust_score_names": loso_sequence_robust_scores,
        "read_loso_best_score_name": loso_best_score,
        "read_loso_best_score_pass_count": loso_score_pass_counts.get(loso_best_score, 0) if loso_best_score else 0,
        "read_failure_decomposition_rows": len(decomposition_rows),
        "runtime_action_allowed": False,
    }
    write_json(out / "summary.json", summary)
    if passed_swa_cache_eligible:
        swa_cache_report = (
            f"SWA cache/top-k eligibility has {len(passed_swa_cache_eligible)} pass rows; "
            f"strict stable SWA coverage is {strict_case_frac} and fallback usage is {fallback_case_frac}. "
            "This is only a TrackK cue eligibility result and still requires the downstream H2/C2/F2/action gates."
        )
    elif passed_swa_cache_internal and f(strict_case_frac, 0.0) < 0.50:
        swa_cache_report = (
            f"SWA cache K/V stability has internal diagnostic carriers ({len(passed_swa_cache_internal)} pass rows), "
            f"but strict stable SWA coverage is {strict_case_frac} and fallback usage is {fallback_case_frac}, "
            "so it remains diagnostic-only and cannot become a semantic eligibility cue."
        )
    else:
        swa_cache_report = (
            f"SWA cache/top-k strict stable coverage is {strict_case_frac} and fallback usage is {fallback_case_frac}, "
            f"but no cache carrier passed the full internal BA/correlation/margin gate "
            f"(internal pass rows={len(passed_swa_cache_internal)})."
        )
    write_text(out / "failure_report.md", (
        "# Track K Failure / Gate Report\n\n"
        "READ now audits multiple plan-directed cue families: stable-anchor deficit, weak context, scale observability, semantic risk, and raw-QK attention-delta internal-instability proxies where available. "
        f"Current semantic-scale-risk cue gate={read_metrics['gate_pass']}; best proxy={read_metrics['best_proxy_score_name']} with BA={read_metrics['best_proxy_balanced_accuracy']}; passing proxies={read_metrics['passing_proxy_score_names']}. "
        f"Current cue sequence_fragile={read_metrics['sequence_fragile']}, positive_sequence_coverage={read_metrics['positive_sequence_coverage']}, good_FPR={read_metrics['good_FPR']}, corr_score_L2_error={read_metrics['corr_score_L2_error']}. "
        f"LOSO robust proxies={loso_sequence_robust_scores}; best LOSO pass-count proxy={loso_best_score} with pass_count={loso_score_pass_counts.get(loso_best_score, 0) if loso_best_score else 0}. "
        f"{swa_cache_report} "
        f"TTT now reads F2 exact retention/write-prior evidence when available; exact_ttt_gate={ttt_gate}, exact_retention_gate={ttt_exact_retention_gate}, exact_missing_good_gate={ttt_exact_missing_good_gate}, exact_uniformity_gate={ttt_exact_uniformity_gate}. "
        "No runtime action is allowed from proxy/replay maps."
    ))
    write_text(out / "what_would_have_to_be_true_to_pass.md", "# What Would Have To Be True\n\nREAD needs recall/FPR/sequence and control margins at gate. SWA needs BA, L3 correlation, same-count and sequence margins plus non-fallback strict stable anchors. TTT needs exact stable-anchor retention separation, exact missing-good-write margin, or exact write-uniformity margin with good FPR <= 0.25; replay/proxy-only maps are insufficient.")
    return summary


def build_h2(root: Path, sources: dict[str, Any]) -> dict[str, Any]:
    out = root / "trackH2_l07_component_decomposition"
    component_rows: list[dict[str, Any]] = []
    by_pilot: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sources["track_d_rows"]:
        by_pilot[row.get("pilot", "")].append(row)
    for pilot, rows in sorted(by_pilot.items()):
        l2_rows = [row for row in rows if row.get("metric") in {"scale_cv_head_mid_tail_pose_sim3", "head10_to_tail10_pose_sim3_rmse_m"}]
        best = max(l2_rows or rows, key=lambda row: f(row.get("bad_improvement_vs_baseline"), -999.0))
        component = "unknown"
        if "anchorcomp" in pilot:
            component = "stable_anchor_compensation_body"
        elif "confneutral" in pilot:
            component = "confidence_neutral_l07_body"
        elif "gauge_norm" in pilot:
            component = "geometry_or_gauge_normalized_l07_body"
        elif "carrier" in pilot:
            component = "raw_qk_carrier_scoped_l07_body"
        elif "qkpair" in pilot:
            component = "source_target_qkpair_keystability_body"
        elif "layer" in pilot:
            component = "dg_q90_per_head_source_bias_body"
        bad_improve = f(best.get("bad_improvement_vs_baseline"), 0.0)
        required_margins = parse_mapping(best.get("candidate_margins_vs_required_controls", ""))
        required_medians = parse_mapping(best.get("bad_required_control_medians", ""))
        conf_margin = margin_by_key(required_margins, "CONFIDENCE_SHUFFLE")
        label_margin = margin_by_key(required_margins, "LABEL_SHUFFLE")
        same_mass_margin = margin_by_key(required_margins, "SAME_MASS_RANDOM")
        group_margin = margin_by_key(required_margins, "GROUP_RANDOM")
        geometry_margin = margin_by_key(required_margins, "GEOMETRY_CONTROL")
        available_control_margins = finite_values([
            label_margin,
            same_mass_margin,
            group_margin,
            f(best.get("candidate_margin_vs_semantic_rotation"), math.nan),
            f(best.get("candidate_margin_vs_random_same_mass"), math.nan),
        ])
        sem_margin = min(
            available_control_margins,
            default=math.nan,
        )
        required_min_margin = f(best.get("candidate_min_margin_vs_required_controls"), math.nan)
        stable = f(best.get("stable_anchor_preservation_proxy"), math.nan)
        good_worsen = f(best.get("good_worsen_ratio"), 0.0)
        controls_available = b(best.get("required_controls_all_available")) or bool(required_margins)
        gate = (
            bad_improve >= 0.05
            and good_worsen <= 0.02
            and sem_margin >= 0.05
            and controls_available
            and math.isfinite(conf_margin)
            and conf_margin >= 0.05
            and stable >= 0.98
            and b(best.get("global_safety_proxy_pass"))
        )
        component_rows.append({
            "pilot": pilot,
            "component_family": component,
            "candidate": best.get("candidate", ""),
            "metric": best.get("metric", ""),
            "bad_L2_improvement": bad_improve,
            "good_worsen": good_worsen,
            "semantic_specificity_margin_proxy": sem_margin,
            "candidate_minus_confidence_shuffle": conf_margin if math.isfinite(conf_margin) else "",
            "confidence_shuffle_available": math.isfinite(conf_margin),
            "candidate_minus_label_shuffle": label_margin if math.isfinite(label_margin) else "",
            "candidate_minus_same_mass_random": same_mass_margin if math.isfinite(same_mass_margin) else "",
            "candidate_minus_group_random": group_margin if math.isfinite(group_margin) else "",
            "candidate_minus_geometry_control": geometry_margin if math.isfinite(geometry_margin) else "",
            "candidate_min_margin_vs_required_controls": required_min_margin if math.isfinite(required_min_margin) else "",
            "required_controls_all_available": controls_available,
            "required_control_medians": json.dumps(required_medians, sort_keys=True) if required_medians else "",
            "required_control_margins": json.dumps(required_margins, sort_keys=True) if required_margins else "",
            "confidence_shuffle_missing_reason": ""
            if math.isfinite(conf_margin)
            else "No confidence-shuffle required-control margin was available for this TrackD row",
            "stable_anchor_preservation": stable,
            "gauge_harm_proxy_available": best.get("global_safety_proxy_pass", ""),
            "h2_component_gate_pass": gate,
            "source_pilot_root": best.get("pilot_root", ""),
        })
    any_local_mechanism = any(f(row.get("bad_L2_improvement"), 0.0) >= 0.05 for row in component_rows)
    semantic_specific_gate = any(b(row.get("h2_component_gate_pass")) for row in component_rows)
    confidence_shuffle_available = any(b(row.get("confidence_shuffle_available")) for row in component_rows)
    best_component = max(component_rows, key=lambda row: f(row.get("bad_L2_improvement"), -999.0), default={})
    best_passing_component = max(
        [row for row in component_rows if b(row.get("h2_component_gate_pass"))],
        key=lambda row: f(row.get("bad_L2_improvement"), -999.0),
        default={},
    )
    write_rows(out / "component_rows.csv", component_rows)
    write_rows(out / "gate_checks.csv", [
        {"gate": "local_L2_mechanism_exists", "pass": any_local_mechanism},
        {"gate": "semantic_specific_component_with_confidence_control", "pass": semantic_specific_gate},
        {"gate": "confidence_shuffle_available", "pass": confidence_shuffle_available},
    ])
    summary = {
        "schema": "acl2_v97_trackH2_l07_component_decomposition_v1",
        "status": "complete",
        "component_row_count": len(component_rows),
        "local_L2_mechanism_exists": any_local_mechanism,
        "semantic_specific_component_gate_pass": semantic_specific_gate,
        "gate_pass": semantic_specific_gate,
        "confidence_shuffle_available": confidence_shuffle_available,
        "best_component_by_bad_L2_improvement": best_component,
        "best_passing_component": best_passing_component,
        "classification": "LOCAL_MECHANISM_EXISTS_SEMANTIC_SPECIFICITY_CONTROL_GATE_FAIL"
        if any_local_mechanism and confidence_shuffle_available and not semantic_specific_gate
        else "LOCAL_MECHANISM_EXISTS_SEMANTIC_SPECIFICITY_BLOCKED_BY_CONFIDENCE_CONTROL_MISSING"
        if any_local_mechanism and not confidence_shuffle_available
        else "SEMANTIC_SPECIFIC_COMPONENT_PASS"
        if semantic_specific_gate
        else "NO_LOCAL_COMPONENT_SIGNAL",
        "runtime_action_allowed": False,
    }
    write_json(out / "summary.json", summary)
    write_text(out / "confidence_leakage_report.md", (
        "# Confidence Leakage Report\n\n"
        "Explicit confidence-shuffle required-control margins are now parsed from v96 TrackD rows when present. "
        "Rows with candidate_minus_confidence_shuffle below 0.05 remain confidence-leakage risks; rows passing H2 still do not authorize runtime action without Track K and Track C2 gates."
    ))
    write_text(out / "failure_report.md", (
        "# Track H2 Report\n\n"
        f"confidence_shuffle_available={confidence_shuffle_available}. "
        f"semantic_specific_component_gate_pass={semantic_specific_gate}. "
        "The report is based on landed v96 TrackD required-control margins, including confidence, label, same-mass, group-random, and geometry controls."
    ))
    write_text(out / "what_would_have_to_be_true_to_pass.md", "# What Would Have To Be True\n\nA v97 H2 pass needs semantic component rows that beat label/same-mass/group-random and confidence-shuffle controls by at least 0.05 while preserving stable anchors and not worsening gauge-harm proxy. Even if H2 passes, Stage7 remains blocked until Track K and Track C2 also pass.")
    return summary


def build_c2_read_latent_smoke_rows(out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    mask_path = V96_ROOT / "trackJ_semantic_region_bank/semantic_region_masks.pt"
    smoke_roots = sorted(path for path in out.glob("read_latent_dump_smoke*") if path.is_dir())
    if not smoke_roots or not mask_path.exists():
        return [], [], []
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return ([{
            "alignment_type": "smoke_load",
            "available": False,
            "missing_reason": f"torch_import_failed:{type(exc).__name__}:{exc}",
        }], [], [])

    try:
        masks_by_case = torch.load(mask_path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        return ([{
            "alignment_type": "smoke_load",
            "available": False,
            "missing_reason": f"semantic_region_mask_load_failed:{type(exc).__name__}:{exc}",
        }], [], [])

    alignment_rows: list[dict[str, Any]] = []
    read_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    for smoke_root in smoke_roots:
        for case_dir in sorted(path for path in smoke_root.iterdir() if path.is_dir()):
            case_id = case_dir.name
            case_masks = masks_by_case.get(case_id, {}) if isinstance(masks_by_case, dict) else {}
            region_masks = case_masks.get("region_token_masks", {}) if isinstance(case_masks, dict) else {}
            baseline_files = sorted((case_dir / "READ_NO_ACTION/pca_features").glob("chunk_*.pt"))
            if not baseline_files:
                alignment_rows.append({
                    "smoke_root": smoke_root.name,
                    "case_id": case_id,
                    "alignment_type": "smoke_load",
                    "available": False,
                    "missing_reason": "baseline READ_NO_ACTION pca feature dump missing",
                })
                continue
            try:
                baseline = torch.load(baseline_files[0], map_location="cpu")
            except Exception as exc:  # noqa: BLE001
                alignment_rows.append({
                    "smoke_root": smoke_root.name,
                    "case_id": case_id,
                    "alignment_type": "smoke_load",
                    "available": False,
                    "missing_reason": f"baseline_load_failed:{type(exc).__name__}:{exc}",
                })
                continue
            for variant_dir in sorted(path for path in case_dir.iterdir() if path.is_dir() and path.name != "READ_NO_ACTION"):
                feature_files = sorted((variant_dir / "pca_features").glob("chunk_*.pt"))
                if not feature_files:
                    alignment_rows.append({
                        "smoke_root": smoke_root.name,
                        "case_id": case_id,
                        "variant": variant_dir.name,
                        "alignment_type": "smoke_load",
                        "available": False,
                        "missing_reason": "variant pca feature dump missing",
                    })
                    continue
                try:
                    variant_payload = torch.load(feature_files[0], map_location="cpu")
                except Exception as exc:  # noqa: BLE001
                    alignment_rows.append({
                        "smoke_root": smoke_root.name,
                        "case_id": case_id,
                        "variant": variant_dir.name,
                        "alignment_type": "smoke_load",
                        "available": False,
                        "missing_reason": f"variant_load_failed:{type(exc).__name__}:{exc}",
                    })
                    continue
                for tap in sorted(key for key in baseline if str(key).startswith("tap::")):
                    before = baseline.get(tap)
                    after = variant_payload.get(tap)
                    if not (torch.is_tensor(before) and torch.is_tensor(after) and tuple(before.shape) == tuple(after.shape)):
                        continue
                    for region in ("STABLE_ANCHOR", "WEAK_SCALE_CONTEXT", "DYNAMIC_OBJECT"):
                        mask = region_masks.get(region) if isinstance(region_masks, dict) else None
                        if not torch.is_tensor(mask):
                            continue
                        mask_bool = mask.detach().cpu().bool()
                        if before.ndim != 5 or tuple(mask_bool.shape) != tuple(before.shape[:1] + before.shape[2:4]):
                            continue
                        before_region = before.detach().cpu().float().permute(0, 2, 3, 1, 4)[mask_bool].reshape(-1, before.shape[-1])
                        after_region = after.detach().cpu().float().permute(0, 2, 3, 1, 4)[mask_bool].reshape(-1, after.shape[-1])
                        if before_region.numel() == 0 or after_region.numel() == 0:
                            continue
                        delta = after_region - before_region
                        identity_residual = float(torch.linalg.vector_norm(delta, dim=-1).mean().item())
                        denom = torch.clamp((before_region * before_region).sum(), min=EPS)
                        scalar = float(((before_region * after_region).sum() / denom).item())
                        scalar_residual = float(torch.linalg.vector_norm(after_region - scalar * before_region, dim=-1).mean().item())
                        try:
                            u, _s, vh = torch.linalg.svd(before_region.T @ after_region, full_matrices=False)
                            rotation = u @ vh
                            orthogonal_residual = float(torch.linalg.vector_norm(after_region - before_region @ rotation, dim=-1).mean().item())
                        except Exception:  # noqa: BLE001
                            orthogonal_residual = math.nan
                        baseline_norm = float(torch.linalg.vector_norm(before_region, dim=-1).mean().item())
                        row = {
                            "smoke_root": smoke_root.name,
                            "case_id": case_id,
                            "variant": variant_dir.name,
                            "tap": tap.removeprefix("tap::"),
                            "region_type": region,
                            "token_count": int(mask_bool.sum().item()),
                            "feature_vector_count": int(before_region.shape[0]),
                            "identity_residual": identity_residual,
                            "scalar_residual": scalar_residual,
                            "scalar_coeff": scalar,
                            "orthogonal_residual": orthogonal_residual if math.isfinite(orthogonal_residual) else "",
                            "baseline_feature_norm": baseline_norm,
                            "identity_residual_over_baseline_norm": identity_residual / (baseline_norm + EPS),
                            "scalar_residual_over_baseline_norm": scalar_residual / (baseline_norm + EPS),
                            "orthogonal_residual_over_baseline_norm": orthogonal_residual / (baseline_norm + EPS)
                            if math.isfinite(orthogonal_residual)
                            else "",
                            "baseline_feature_path": str(baseline_files[0]),
                            "variant_feature_path": str(feature_files[0]),
                            "available": True,
                        }
                        alignment_rows.append(row)
                        if variant_dir.name == "READ21_GATED_L07_CONFNEUTRAL_ANCHORCOMP_T050":
                            read_rows.append(row)
                        control_name = ""
                        if "CONFIDENCE_SHUFFLE" in variant_dir.name:
                            control_name = "confidence_shuffle"
                        elif "LABEL_SHUFFLE" in variant_dir.name:
                            control_name = "semantic_label_shuffle"
                        elif "SAME_MASS_RANDOM" in variant_dir.name:
                            control_name = "same_mass_random"
                        elif "GROUP_RANDOM" in variant_dir.name:
                            control_name = "group_random"
                        elif "GEOMETRY_CONTROL" in variant_dir.name:
                            control_name = "geometry_control"
                        if control_name:
                            control_rows.append({**row, "control": control_name})
    return alignment_rows, read_rows, control_rows


def build_c2_downstream_correlation_rows(
    out: Path,
    smoke_alignment_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join the broad C2 latent dump with local-window metrics for correlation audit."""
    available_rows = [row for row in smoke_alignment_rows if b(row.get("available"))]
    smoke_roots = sorted({str(row.get("smoke_root", "")) for row in available_rows if row.get("smoke_root")})
    if not smoke_roots:
        return [], {
            "primary_cohort": "",
            "primary_cohort_case_count": 0,
            "primary_cohort_alignment_rows": 0,
            "downstream_correlation_available": False,
            "downstream_correlation_missing_reason": "no available C2 latent alignment rows",
            "stage7_full_latent_correlation_available": False,
            "stage7_full_latent_correlation_missing_reason": "Stage7 full candidates do not have READ before/after stable-anchor latent dumps",
        }

    def root_score(root_name: str) -> tuple[int, int, int]:
        rows = [row for row in available_rows if row.get("smoke_root") == root_name]
        case_count = len({row.get("case_id") for row in rows})
        all8_bonus = 1 if "all8" in root_name else 0
        return case_count, len(rows), all8_bonus

    primary_root = max(smoke_roots, key=root_score)
    primary_rows = [row for row in available_rows if row.get("smoke_root") == primary_root]
    primary_case_count = len({row.get("case_id") for row in primary_rows})

    metrics_path = out / primary_root / "rows.csv"
    if not metrics_path.exists():
        return [], {
            "primary_cohort": primary_root,
            "primary_cohort_case_count": primary_case_count,
            "primary_cohort_alignment_rows": len(primary_rows),
            "downstream_correlation_available": False,
            "downstream_correlation_missing_reason": f"metrics rows missing:{metrics_path}",
            "stage7_full_latent_correlation_available": False,
            "stage7_full_latent_correlation_missing_reason": "Stage7 full candidates do not have READ before/after stable-anchor latent dumps",
        }

    metric_rows = read_rows(metrics_path)
    metric_by_case_variant = {
        (row.get("case_id", ""), row.get("variant", "")): row
        for row in metric_rows
        if row.get("case_id") and row.get("variant")
    }
    baseline_by_case = {
        row.get("case_id", ""): row
        for row in metric_rows
        if row.get("case_id") and row.get("variant") == "READ_NO_ACTION"
    }

    residual_by_case_variant_region: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    residual_over_norm_by_case_variant_region: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in primary_rows:
        region = str(row.get("region_type", ""))
        if region not in {"STABLE_ANCHOR", "WEAK_SCALE_CONTEXT", "DYNAMIC_OBJECT"}:
            continue
        key = (str(row.get("case_id", "")), str(row.get("variant", "")), region)
        residual = f(row.get("identity_residual"))
        residual_over_norm = f(row.get("identity_residual_over_baseline_norm"))
        if math.isfinite(residual):
            residual_by_case_variant_region[key].append(residual)
        if math.isfinite(residual_over_norm):
            residual_over_norm_by_case_variant_region[key].append(residual_over_norm)

    case_rows: list[dict[str, Any]] = []
    for (case_id, variant, region), values in sorted(residual_by_case_variant_region.items()):
        metrics = metric_by_case_variant.get((case_id, variant), {})
        baseline = baseline_by_case.get(case_id, {})
        if not metrics or not baseline:
            continue
        baseline_scale_cv = f(baseline.get("scale_cv_head_mid_tail_pose_sim3"))
        variant_scale_cv = f(metrics.get("scale_cv_head_mid_tail_pose_sim3"))
        baseline_head_tail = f(baseline.get("head10_to_tail10_pose_sim3_rmse_m"))
        variant_head_tail = f(metrics.get("head10_to_tail10_pose_sim3_rmse_m"))
        baseline_local_ate = f(baseline.get("local_sim3_ate_rmse_m"))
        variant_local_ate = f(metrics.get("local_sim3_ate_rmse_m"))
        baseline_final = f(baseline.get("local_sim3_finalerr_m"))
        variant_final = f(metrics.get("local_sim3_finalerr_m"))
        baseline_yaw = f(baseline.get("local_sim3_yaw_rmse_deg"))
        variant_yaw = f(metrics.get("local_sim3_yaw_rmse_deg"))
        baseline_scale = f(baseline.get("local_sim3_scale"))
        variant_scale = f(metrics.get("local_sim3_scale"))
        residual_median = median(values)
        residual_over_norm = median(residual_over_norm_by_case_variant_region.get((case_id, variant, region), []))
        case_rows.append({
            "row_type": "case_metric_join",
            "smoke_root": primary_root,
            "case_id": case_id,
            "seq": metrics.get("seq", ""),
            "chunk": metrics.get("chunk", ""),
            "bucket": metrics.get("bucket", ""),
            "variant": variant,
            "region_type": region,
            "latent_identity_residual_median": residual_median,
            "latent_identity_residual_over_norm_median": residual_over_norm
            if math.isfinite(residual_over_norm)
            else "",
            "latent_active": abs(residual_median) > 1.0e-9 if math.isfinite(residual_median) else False,
            "scale_cv_improvement_vs_baseline": (baseline_scale_cv - variant_scale_cv) / (abs(baseline_scale_cv) + EPS)
            if math.isfinite(baseline_scale_cv) and math.isfinite(variant_scale_cv)
            else "",
            "head_tail_improvement_vs_baseline": (baseline_head_tail - variant_head_tail) / (abs(baseline_head_tail) + EPS)
            if math.isfinite(baseline_head_tail) and math.isfinite(variant_head_tail)
            else "",
            "local_ate_improvement_vs_baseline": (baseline_local_ate - variant_local_ate) / (abs(baseline_local_ate) + EPS)
            if math.isfinite(baseline_local_ate) and math.isfinite(variant_local_ate)
            else "",
            "local_final_error_delta": variant_final - baseline_final
            if math.isfinite(variant_final) and math.isfinite(baseline_final)
            else "",
            "local_yaw_rmse_delta_deg": variant_yaw - baseline_yaw
            if math.isfinite(variant_yaw) and math.isfinite(baseline_yaw)
            else "",
            "local_scale_delta": variant_scale - baseline_scale
            if math.isfinite(variant_scale) and math.isfinite(baseline_scale)
            else "",
            "baseline_scale_cv": baseline_scale_cv if math.isfinite(baseline_scale_cv) else "",
            "variant_scale_cv": variant_scale_cv if math.isfinite(variant_scale_cv) else "",
            "baseline_local_final_error": baseline_final if math.isfinite(baseline_final) else "",
            "variant_local_final_error": variant_final if math.isfinite(variant_final) else "",
        })

    correlation_targets = [
        ("scale_cv_improvement_vs_baseline", "higher_better"),
        ("head_tail_improvement_vs_baseline", "higher_better"),
        ("local_ate_improvement_vs_baseline", "higher_better"),
        ("local_final_error_delta", "lower_better"),
        ("local_yaw_rmse_delta_deg", "lower_abs_better"),
        ("local_scale_delta", "lower_abs_better"),
    ]
    correlation_rows: list[dict[str, Any]] = []
    variants = sorted({row.get("variant", "") for row in case_rows if row.get("region_type") == "STABLE_ANCHOR"})
    for variant in variants:
        variant_rows = [
            row for row in case_rows
            if row.get("variant") == variant and row.get("region_type") == "STABLE_ANCHOR"
        ]
        xs = [f(row.get("latent_identity_residual_median")) for row in variant_rows]
        for target, direction in correlation_targets:
            ys = [f(row.get(target)) for row in variant_rows]
            corr = pearson(xs, ys)
            finite_pairs = [
                (x, y) for x, y in zip(xs, ys)
                if math.isfinite(x) and math.isfinite(y)
            ]
            active_pairs = [
                (x, y) for x, y in finite_pairs
                if abs(x) > 1.0e-9
            ]
            bucket_counts = Counter(str(row.get("bucket", "")) for row in variant_rows)
            correlation_rows.append({
                "row_type": "correlation_summary",
                "smoke_root": primary_root,
                "variant": variant,
                "region_type": "STABLE_ANCHOR",
                "target_metric": target,
                "target_direction": direction,
                "pearson_r": corr if math.isfinite(corr) else "",
                "abs_pearson_r": abs(corr) if math.isfinite(corr) else "",
                "pair_count": len(finite_pairs),
                "active_pair_count": len(active_pairs),
                "case_count": len({row.get("case_id") for row in variant_rows}),
                "read_local_bad_count": bucket_counts.get("READ_LOCAL_BAD", 0),
                "good_protection_count": bucket_counts.get("GOOD_PROTECTION", 0),
            })

    candidate_variant = "READ21_GATED_L07_CONFNEUTRAL_ANCHORCOMP_T050"
    control_variants = [variant for variant in variants if variant != candidate_variant]

    def corr_value(variant: str, target: str) -> float:
        for row in correlation_rows:
            if row.get("variant") == variant and row.get("target_metric") == target:
                return f(row.get("pearson_r"))
        return math.nan

    def abs_corr_value(variant: str, target: str) -> float:
        value = corr_value(variant, target)
        return abs(value) if math.isfinite(value) else math.nan

    candidate_scale_corr = corr_value(candidate_variant, "scale_cv_improvement_vs_baseline")
    candidate_final_abs_corr = abs_corr_value(candidate_variant, "local_final_error_delta")
    candidate_yaw_abs_corr = abs_corr_value(candidate_variant, "local_yaw_rmse_delta_deg")
    candidate_scale_delta_abs_corr = abs_corr_value(candidate_variant, "local_scale_delta")
    max_control_scale_corr = max(
        finite_values([corr_value(variant, "scale_cv_improvement_vs_baseline") for variant in control_variants]),
        default=math.nan,
    )
    max_control_final_abs_corr = max(
        finite_values([abs_corr_value(variant, "local_final_error_delta") for variant in control_variants]),
        default=math.nan,
    )
    max_control_yaw_abs_corr = max(
        finite_values([abs_corr_value(variant, "local_yaw_rmse_delta_deg") for variant in control_variants]),
        default=math.nan,
    )
    max_control_scale_delta_abs_corr = max(
        finite_values([abs_corr_value(variant, "local_scale_delta") for variant in control_variants]),
        default=math.nan,
    )
    candidate_case_rows = [
        row for row in case_rows
        if row.get("variant") == candidate_variant and row.get("region_type") == "STABLE_ANCHOR"
    ]
    candidate_active_count = sum(1 for row in candidate_case_rows if b(row.get("latent_active")))
    effective_case_count = len(candidate_case_rows)
    control_comparable_or_stronger = bool(
        (math.isfinite(max_control_scale_corr) and math.isfinite(candidate_scale_corr) and max_control_scale_corr >= candidate_scale_corr - 0.05)
        or (math.isfinite(max_control_final_abs_corr) and math.isfinite(candidate_final_abs_corr) and max_control_final_abs_corr >= candidate_final_abs_corr - 0.05)
        or (math.isfinite(max_control_yaw_abs_corr) and math.isfinite(candidate_yaw_abs_corr) and max_control_yaw_abs_corr >= candidate_yaw_abs_corr - 0.05)
        or (
            math.isfinite(max_control_scale_delta_abs_corr)
            and math.isfinite(candidate_scale_delta_abs_corr)
            and max_control_scale_delta_abs_corr >= candidate_scale_delta_abs_corr - 0.05
        )
    )
    local_window_correlation_gate_pass = bool(
        effective_case_count >= 8
        and candidate_active_count >= 5
        and math.isfinite(candidate_scale_corr)
        and candidate_scale_corr >= 0.30
        and math.isfinite(candidate_final_abs_corr)
        and candidate_final_abs_corr >= 0.30
        and not control_comparable_or_stronger
    )
    summary = {
        "primary_cohort": primary_root,
        "primary_cohort_case_count": primary_case_count,
        "primary_cohort_alignment_rows": len(primary_rows),
        "downstream_correlation_available": bool(correlation_rows),
        "downstream_correlation_row_count": len(correlation_rows),
        "downstream_case_metric_join_rows": len(case_rows),
        "candidate_downstream_effective_case_count": effective_case_count,
        "candidate_downstream_active_residual_case_count": candidate_active_count,
        "candidate_corr_scale_cv_improvement": candidate_scale_corr if math.isfinite(candidate_scale_corr) else "",
        "candidate_abs_corr_local_final_error_delta": candidate_final_abs_corr
        if math.isfinite(candidate_final_abs_corr)
        else "",
        "candidate_abs_corr_local_yaw_delta": candidate_yaw_abs_corr
        if math.isfinite(candidate_yaw_abs_corr)
        else "",
        "candidate_abs_corr_local_scale_delta": candidate_scale_delta_abs_corr
        if math.isfinite(candidate_scale_delta_abs_corr)
        else "",
        "max_control_corr_scale_cv_improvement": max_control_scale_corr
        if math.isfinite(max_control_scale_corr)
        else "",
        "max_control_abs_corr_local_final_error_delta": max_control_final_abs_corr
        if math.isfinite(max_control_final_abs_corr)
        else "",
        "max_control_abs_corr_local_yaw_delta": max_control_yaw_abs_corr
        if math.isfinite(max_control_yaw_abs_corr)
        else "",
        "max_control_abs_corr_local_scale_delta": max_control_scale_delta_abs_corr
        if math.isfinite(max_control_scale_delta_abs_corr)
        else "",
        "downstream_control_comparable_or_stronger": control_comparable_or_stronger,
        "local_window_downstream_correlation_gate_pass": local_window_correlation_gate_pass,
        "stage7_full_latent_correlation_available": False,
        "stage7_full_latent_correlation_missing_reason": "Stage7 full candidates have final-error/global-yaw rows but no before/after stable-anchor latent dumps; local-window C2 smoke cannot be promoted as full-sequence latent ruler evidence.",
    }
    return case_rows + correlation_rows, summary


def build_c2_stage7_latent_dump_feasibility_rows(sources: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    v96_root = Path(str(sources.get("v96_root") or V96_ROOT))
    replay_manifest_root = ROOT / "trackC_semantic_latent_gauge_ruler" / "stage7_full_latent_replay_dryrun"
    replay_manifests: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted(replay_manifest_root.glob("**/replay_manifest.json")):
        manifest = read_json(manifest_path)
        source_config = manifest.get("source_hmc_config")
        if not source_config:
            continue
        replay_manifests[str(Path(str(source_config)).resolve())] = {
            "manifest_path": str(manifest_path),
            "manifest": manifest,
        }
    rows: list[dict[str, Any]] = []
    for rollout in sorted(v96_root.glob("stage7_seq*_full*/rollouts/*FULL")):
        config_path = rollout / "hmc_config.yaml"
        replay_entry = replay_manifests.get(str(config_path.resolve()), {})
        replay_manifest = replay_entry.get("manifest", {})
        replay_run = replay_manifest.get("run", {}) if isinstance(replay_manifest.get("run", {}), dict) else {}
        replay_unsupported = replay_manifest.get("unsupported_config_keys", [])
        replay_unsupported_count = len(replay_unsupported) if isinstance(replay_unsupported, list) else ""
        replay_manifest_exists = bool(replay_manifest)
        replay_launched = bool(replay_run)
        config_text = config_path.read_text(encoding="utf-8", errors="replace") if config_path.exists() else ""
        pca_flag_enabled = any(
            line.strip().lower() in {
                "v68_export_full_pca_debug: 1",
                "v68_export_full_pca_debug: true",
                "v68_export_full_pca_debug: yes",
            }
            for line in config_text.splitlines()
        )
        pca_feature_files = sorted((rollout / "pca_features").glob("*.pt")) if (rollout / "pca_features").exists() else []
        rows.append({
            "stage7_root": str(rollout.parents[1]),
            "rollout_root": str(rollout),
            "candidate": rollout.name,
            "hmc_config_exists": config_path.exists(),
            "v68_export_full_pca_debug_enabled": pca_flag_enabled,
            "pca_feature_file_count": len(pca_feature_files),
            "full_sequence_latent_dump_available": pca_flag_enabled and bool(pca_feature_files),
            "diagnostic_replay_manifest_exists": replay_manifest_exists,
            "diagnostic_replay_manifest": replay_entry.get("manifest_path", ""),
            "diagnostic_replay_dry_run": replay_manifest.get("dry_run", ""),
            "diagnostic_replay_command_arg_count": replay_manifest.get("command_arg_count", ""),
            "diagnostic_replay_command_sha256": replay_manifest.get("command_sha256", ""),
            "diagnostic_replay_unsupported_config_key_count": replay_unsupported_count,
            "diagnostic_replay_run_returncode": replay_run.get("returncode", ""),
            "diagnostic_replay_pca_feature_file_count": replay_run.get("pca_feature_file_count", ""),
            "diagnostic_full_dump_launched_in_v97": replay_launched,
            "not_launched_reason": (
                "audited v97 diagnostic replay dry-run manifest exists, but full replay was not executed because v97 prerequisite gates still fail and the plan only permits Stage7 full sequence after mechanism+predictor pass"
                if replay_manifest_exists and not replay_launched
                else "v97 prerequisite gates still fail, so Stage7/full-sequence action validation is not allowed; existing full rollouts also lack v68 PCA latent dumps, and no audited full-dump replay command was found in the Stage7 audit script"
            ),
        })
    available_count = sum(1 for row in rows if b(row.get("full_sequence_latent_dump_available")))
    replay_manifest_total_count = len(replay_manifests)
    replay_manifest_matching_count = sum(1 for row in rows if b(row.get("diagnostic_replay_manifest_exists")))
    replay_manifest_clean_matching_count = sum(
        1
        for row in rows
        if b(row.get("diagnostic_replay_manifest_exists")) and f(row.get("diagnostic_replay_unsupported_config_key_count"), 1.0) == 0.0
    )
    replay_dry_run_only_count = sum(
        1
        for row in rows
        if b(row.get("diagnostic_replay_manifest_exists"))
        and b(row.get("diagnostic_replay_dry_run"))
        and not b(row.get("diagnostic_full_dump_launched_in_v97"))
    )
    replay_launched_count = sum(1 for row in rows if b(row.get("diagnostic_full_dump_launched_in_v97")))
    summary = {
        "stage7_full_latent_dump_feasibility_audited": True,
        "stage7_full_rollout_count": len(rows),
        "stage7_full_latent_dump_available_count": available_count,
        "stage7_full_latent_dump_available_any": available_count > 0,
        "stage7_full_latent_replay_manifest_total_count": replay_manifest_total_count,
        "stage7_full_latent_replay_manifest_matching_rollout_count": replay_manifest_matching_count,
        "stage7_full_latent_replay_manifest_clean_matching_rollout_count": replay_manifest_clean_matching_count,
        "stage7_full_latent_replay_dry_run_only_matching_rollout_count": replay_dry_run_only_count,
        "stage7_full_latent_replay_launched_count": replay_launched_count,
        "stage7_full_latent_replay_command_available_any": replay_manifest_matching_count > 0,
        "stage7_full_latent_dump_not_run_reason": (
            "v97 prerequisite gates still fail; v97 cannot launch Stage7/full-sequence action validation. "
            "Audited dry-run replay manifests now exist for saved full rollouts, but no full replay was executed and no full-rollout v68 PCA latent dump is available."
            if replay_manifest_matching_count
            else "v97 prerequisite gates still fail; v97 cannot launch Stage7/full-sequence action validation. "
            "A feasibility scan found no existing full-rollout v68 PCA latent dump to correlate with final-error/global-yaw."
        ),
    }
    return rows, summary


def build_c2(root: Path, sources: dict[str, Any]) -> dict[str, Any]:
    out = root / "trackC_semantic_latent_gauge_ruler"
    atlas, sem, _seq = case_maps(sources)
    anchor_rows: list[dict[str, Any]] = []
    for case_id, row in sorted(sem.items()):
        stable = f(row.get("STABLE_ANCHOR_token_mass"), 0.0)
        low = f(row.get("LOW_OBSERVABILITY_token_mass"), 0.0)
        dynamic = f(row.get("DYNAMIC_OBJECT_token_mass"), 0.0)
        score = stable * (1.0 - min(1.0, low + dynamic))
        anchor_rows.append({
            "anchor_id": f"{case_id}_stable_anchor_proxy",
            "case_id": case_id,
            "seq": row.get("seq", ""),
            "memory_body": "semantic_region_bank_proxy",
            "semantic_type": "STABLE_ANCHOR",
            "internal_stability": "",
            "scale_observability": semantic_scores(case_id, atlas, sem)["scale_observability_score"],
            "track_length": "",
            "anchor_score": score,
            "weak_anchor": score < 0.05,
        })
    smoke_alignment_rows, smoke_read_rows, smoke_control_rows = build_c2_read_latent_smoke_rows(out)
    read_rows = []
    for row in smoke_read_rows:
        read_rows.append({
            **row,
            "source": "v97_read_latent_dump_smoke_h2_anchorcomp_t050",
            "proxy_interpretation": "single-case READ before/after latent proxy residual; gate cannot pass without multi-case correlation/heldout control",
        })
    for row in sources["stage7_rows"]:
        read_rows.append({
            "candidate": row.get("candidate", ""),
            "stable_anchor_latent_residual": "",
            "latent_residual_missing_reason": "v96 Stage7 did not save before/after stable-anchor latent features for low-DOF alignment",
            "delta_final_error_m": row.get("delta_final_error_m", ""),
            "delta_aligned_ate_rmse_m": row.get("delta_aligned_ate_rmse_m", ""),
            "rolling_worse_fraction_max": row.get("rolling_worse_fraction_max", ""),
            "proxy_interpretation": "local/full gauge harm observable, latent ruler unavailable",
        })
    swa_decision = sources["swa_decision"]
    swa_rows = [{
        "source": swa_decision.get("source_artifacts", {}).get("trackC_latent_gauge_summary", ""),
        "stable_anchor_residual": "",
        "proxy_max_balanced_accuracy": swa_decision.get("trackC_max_balanced_accuracy", ""),
        "proxy_max_abs_pearson_with_L3": swa_decision.get("trackC_max_abs_pearson_with_L3", ""),
        "trackC_gate_pass": swa_decision.get("trackC_gate_pass", ""),
        "classification": swa_decision.get("classification", ""),
        "missing_reason": "SWA latent-gauge decision is route/feature proxy, not READ before/after low-DOF stable-anchor latent alignment",
    }]
    ttt_rows = [{
        "stable_anchor_retention": "",
        "missing_reason": "TTT fixed-pair analysis lacks stable-anchor before/after latent retention tensors",
        "source": "v96_trackF_analysis_fixed_pair",
    }]
    downstream_rows, downstream_summary = build_c2_downstream_correlation_rows(out, smoke_alignment_rows)
    stage7_latent_rows, stage7_latent_summary = build_c2_stage7_latent_dump_feasibility_rows(sources)
    write_rows(out / "anchor_rows.csv", anchor_rows)
    write_rows(out / "read_latent_ruler_rows.csv", read_rows)
    write_rows(out / "swa_latent_ruler_rows.csv", swa_rows)
    write_rows(out / "ttt_latent_ruler_rows.csv", ttt_rows)
    write_rows(out / "low_dof_alignment_rows.csv", smoke_alignment_rows or [{
        "alignment_type": "identity/scalar/orthogonal/near_identity_ridge",
        "available": False,
        "missing_reason": "no saved stable-anchor latent vectors before/after action",
    }])
    write_rows(out / "random_control_rows.csv", smoke_control_rows or [{
        "control": "semantic_shuffle",
        "available": False,
        "missing_reason": "no latent alignment matrix to shuffle",
    }])
    write_rows(out / "downstream_correlation_rows.csv", downstream_rows or [{
        "row_type": "missing",
        "available": False,
        "missing_reason": downstream_summary.get("downstream_correlation_missing_reason", "no downstream correlation rows"),
    }])
    write_rows(out / "stage7_full_latent_dump_feasibility_rows.csv", stage7_latent_rows or [{
        "available": False,
        "missing_reason": "no v96 stage7 full rollout directories found",
    }])
    read_latent_available = any(b(row.get("available")) for row in smoke_alignment_rows)
    smoke_case_count = len({row.get("case_id") for row in smoke_alignment_rows if b(row.get("available"))})
    smoke_variant_count = len({row.get("variant") for row in smoke_alignment_rows if b(row.get("available"))})
    semantic_shuffle_available = any(row.get("control") == "semantic_label_shuffle" for row in smoke_control_rows)
    identity_residuals = [
        f(row.get("identity_residual"))
        for row in smoke_alignment_rows
        if b(row.get("available")) and math.isfinite(f(row.get("identity_residual")))
    ]
    max_identity_residual = max(identity_residuals) if identity_residuals else math.nan
    nonzero_identity_residual_rows = sum(1 for value in identity_residuals if abs(value) > 1.0e-9)
    stable_rows = [
        row for row in smoke_alignment_rows
        if b(row.get("available")) and row.get("region_type") == "STABLE_ANCHOR"
    ]
    candidate_variant = "READ21_GATED_L07_CONFNEUTRAL_ANCHORCOMP_T050"
    active_residual_cases = {
        row.get("case_id")
        for row in stable_rows
        if row.get("variant") == candidate_variant and abs(f(row.get("identity_residual"))) > 1.0e-9
    }
    stable_active_rows = [row for row in stable_rows if row.get("case_id") in active_residual_cases]

    def stable_active_median(variant: str) -> float:
        values = [
            f(row.get("identity_residual"))
            for row in stable_active_rows
            if row.get("variant") == variant and math.isfinite(f(row.get("identity_residual")))
        ]
        return float(median(values)) if values else math.nan

    candidate_stable_active_median = stable_active_median(candidate_variant)
    confidence_stable_active_median = stable_active_median("READ21_GATED_L07_CONFIDENCE_SHUFFLE_T050")
    label_stable_active_median = stable_active_median("READ21_GATED_L07_CONFNEUTRAL_ANCHORCOMP_LABEL_SHUFFLE_T050")
    same_mass_stable_active_median = stable_active_median("READ21_GATED_L07_CONFNEUTRAL_ANCHORCOMP_SAME_MASS_RANDOM_T050")
    group_stable_active_median = stable_active_median("READ21_GATED_L07_CONFNEUTRAL_ANCHORCOMP_GROUP_RANDOM_T050")
    geometry_stable_active_median = stable_active_median("READ21_GATED_L07_GEOMETRY_CONTROL_T050")
    candidate_minus_label = (
        candidate_stable_active_median - label_stable_active_median
        if math.isfinite(candidate_stable_active_median) and math.isfinite(label_stable_active_median)
        else math.nan
    )
    candidate_minus_confidence = (
        candidate_stable_active_median - confidence_stable_active_median
        if math.isfinite(candidate_stable_active_median) and math.isfinite(confidence_stable_active_median)
        else math.nan
    )
    control_comparable_threshold = 0.01
    semantic_shuffle_comparable = (
        math.isfinite(candidate_minus_label)
        and candidate_minus_label <= control_comparable_threshold
    )
    confidence_shuffle_comparable = (
        math.isfinite(candidate_minus_confidence)
        and candidate_minus_confidence <= control_comparable_threshold
    )
    summary = {
        "schema": "acl2_v97_trackC2_semantic_latent_gauge_ruler_v1",
        "status": "complete_diagnostic_blocked",
        "anchor_rows": len(anchor_rows),
        "weak_anchor_rows": sum(1 for row in anchor_rows if row["weak_anchor"]),
        "read_latent_residual_available": read_latent_available,
        "read_latent_smoke_case_count": smoke_case_count,
        "read_latent_smoke_variant_count": smoke_variant_count,
        "read_latent_smoke_alignment_rows": len([row for row in smoke_alignment_rows if b(row.get("available"))]),
        "semantic_shuffle_control_available": semantic_shuffle_available,
        "read_latent_smoke_max_identity_residual": max_identity_residual if math.isfinite(max_identity_residual) else "",
        "read_latent_smoke_nonzero_identity_residual_rows": nonzero_identity_residual_rows,
        "read_latent_smoke_active_residual_case_count": len(active_residual_cases),
        "read_latent_smoke_stable_anchor_candidate_median_identity_residual_active_cases": candidate_stable_active_median
        if math.isfinite(candidate_stable_active_median)
        else "",
        "read_latent_smoke_stable_anchor_confidence_shuffle_median_identity_residual_active_cases": confidence_stable_active_median
        if math.isfinite(confidence_stable_active_median)
        else "",
        "read_latent_smoke_stable_anchor_label_shuffle_median_identity_residual_active_cases": label_stable_active_median
        if math.isfinite(label_stable_active_median)
        else "",
        "read_latent_smoke_stable_anchor_same_mass_median_identity_residual_active_cases": same_mass_stable_active_median
        if math.isfinite(same_mass_stable_active_median)
        else "",
        "read_latent_smoke_stable_anchor_group_random_median_identity_residual_active_cases": group_stable_active_median
        if math.isfinite(group_stable_active_median)
        else "",
        "read_latent_smoke_stable_anchor_geometry_control_median_identity_residual_active_cases": geometry_stable_active_median
        if math.isfinite(geometry_stable_active_median)
        else "",
        "read_latent_smoke_candidate_minus_label_shuffle_stable_anchor_median_identity_residual_active_cases": candidate_minus_label
        if math.isfinite(candidate_minus_label)
        else "",
        "read_latent_smoke_candidate_minus_confidence_shuffle_stable_anchor_median_identity_residual_active_cases": candidate_minus_confidence
        if math.isfinite(candidate_minus_confidence)
        else "",
        "read_latent_smoke_control_comparable_threshold": control_comparable_threshold,
        "read_latent_smoke_semantic_shuffle_comparable_or_stronger": semantic_shuffle_comparable,
        "read_latent_smoke_confidence_shuffle_comparable_or_stronger": confidence_shuffle_comparable,
        **downstream_summary,
        **stage7_latent_summary,
        "swa_proxy_available": bool(swa_decision),
        "ttt_retention_available": False,
        "gate_pass": False,
        "classification": "READ_LATENT_SMOKE_AVAILABLE_DOWNSTREAM_CORRELATION_CONTROL_COMPARABLE_GATE_BLOCKED"
        if read_latent_available
        and downstream_summary.get("downstream_correlation_available")
        and (
            downstream_summary.get("downstream_control_comparable_or_stronger")
            or f(downstream_summary.get("candidate_downstream_active_residual_case_count"), 0.0) < 5
        )
        else "READ_LATENT_SMOKE_AVAILABLE_ZERO_RESIDUAL_NO_ACTION_RESPONSE_GATE_BLOCKED"
        if read_latent_available and nonzero_identity_residual_rows == 0
        else "READ_LATENT_SMOKE_AVAILABLE_SEMANTIC_CONTROL_COMPARABLE_GATE_BLOCKED"
        if read_latent_available and (semantic_shuffle_comparable or confidence_shuffle_comparable)
        else "READ_LATENT_SMOKE_AVAILABLE_GATE_BLOCKED_SAMPLE_AND_CORRELATION_MISSING"
        if read_latent_available
        else "ANCHOR_COVERAGE_AVAILABLE_LATENT_ALIGNMENT_MISSING",
        "runtime_action_allowed": False,
    }
    write_json(out / "summary.json", summary)
    if read_latent_available:
        write_text(out / "latent_ruler_report.md", (
            "# Track C2 Latent Ruler Report\n\n"
            "A v97 single-case READ latent dump smoke is available and was aligned with v96 semantic token masks. "
            f"Smoke cases={smoke_case_count}, variants={smoke_variant_count}, alignment_rows={summary['read_latent_smoke_alignment_rows']}. "
            f"Max identity residual={summary['read_latent_smoke_max_identity_residual']}, nonzero identity residual rows={nonzero_identity_residual_rows}. "
            f"Active residual cases={len(active_residual_cases)}; stable-anchor candidate median identity residual on active cases={summary['read_latent_smoke_stable_anchor_candidate_median_identity_residual_active_cases']}; "
            f"label-shuffle median={summary['read_latent_smoke_stable_anchor_label_shuffle_median_identity_residual_active_cases']}; "
            f"confidence-shuffle median={summary['read_latent_smoke_stable_anchor_confidence_shuffle_median_identity_residual_active_cases']}. "
            f"Primary downstream cohort={summary.get('primary_cohort')}, candidate active residual cases={summary.get('candidate_downstream_active_residual_case_count')}, "
            f"candidate corr(scale_cv_improvement)={summary.get('candidate_corr_scale_cv_improvement')}, "
            f"candidate abs corr(local_final_error_delta)={summary.get('candidate_abs_corr_local_final_error_delta')}. "
            f"Stage7 full rollout latent-dump scan: rollout_count={summary.get('stage7_full_rollout_count')}, "
            f"latent_dump_available_count={summary.get('stage7_full_latent_dump_available_count')}, "
            f"replay_manifest_matching_count={summary.get('stage7_full_latent_replay_manifest_matching_rollout_count')}, "
            f"replay_launched_count={summary.get('stage7_full_latent_replay_launched_count')}. "
            "This repairs the raw missing-dump blocker and adds local-window downstream correlation audit, but the Track C2 gate still fails because semantic/confidence controls are comparable or stronger, active residual support remains thin, and Stage7 full final-error/global-yaw latent correlation is unavailable without full-sequence before/after stable-anchor latent dumps."
        ))
        write_text(out / "failure_report.md", (
            "# Track C2 Failure Report\n\n"
            "READ latent residual smoke is available and includes an active-response case, so the original missing-dump blocker is repaired. "
            "C2 remains blocked for promotion because the active stable-anchor residual is not semantic-specific: label/confidence controls are comparable to candidate, local-window downstream correlations are also matched or exceeded by controls, active residual case count is still too small for a robust ruler, TTT retention tensors are not C2 READ latent evidence, and Stage7 full final-error/global-yaw latent correlation cannot be measured from current full-sequence artifacts. "
            "A full-rollout feasibility scan found no existing Stage7 v68 PCA latent dumps. Audited dry-run replay manifests now exist, but no new full-sequence dump was launched because v97 TrackK/C2/F2 prerequisite gates still fail and the plan only permits Stage7 full sequence after mechanism+predictor pass."
        ))
    else:
        write_text(out / "latent_ruler_report.md", "# Track C2 Latent Ruler Report\n\nSemantic stable-anchor coverage exists through v96 semantic region masks, but v96 did not save before/after latent vectors needed for low-DOF latent ruler alignment. Stage7 full failure can be described by final-error/rolling proxies only.")
        write_text(out / "failure_report.md", "# Track C2 Failure Report\n\nBlocked for READ/TTT latent ruler promotion because stable-anchor latent residuals and semantic-shuffle controls are missing.")
    write_text(out / "what_would_have_to_be_true_to_pass.md", "# What Would Have To Be True\n\nSave stable-anchor latent features before/after READ/SWA/TTT actions, run identity/scalar/orthogonal/near-identity alignment, and show residual correlation with final error/global yaw/scale shift while semantic shuffle fails. The current local-window all8 C2 audit is not enough: candidate correlation must exceed label/confidence/same-mass/group/geometry controls and active residual support must be broad enough for a real ruler claim.")
    return summary


def build_f2(root: Path, sources: dict[str, Any]) -> dict[str, Any]:
    out = root / "trackF2_ttt_stable_anchor_retention_missing_good_write"
    exact_rows: list[dict[str, Any]] = []
    exact_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exact_write_rows: list[dict[str, Any]] = []
    exact_write_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exact_root = out / "exact_retention_probe_v1"
    if exact_root.exists():
        try:
            import torch  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            exact_rows.append({
                "case_id": "",
                "chunk_idx": "",
                "available": False,
                "missing_reason": f"torch_import_failed:{type(exc).__name__}:{exc}",
            })
        else:
            for path in sorted(exact_root.glob("*/TTT_PROBE/ttt_spatial_post_delta_maps/*.pt")):
                case_id = path.parents[2].name
                try:
                    payload = torch.load(path, map_location="cpu")
                except Exception as exc:  # noqa: BLE001
                    exact_rows.append({
                        "case_id": case_id,
                        "chunk_idx": "",
                        "path": str(path),
                        "available": False,
                        "missing_reason": f"load_failed:{type(exc).__name__}:{exc}",
                    })
                    continue
                if not isinstance(payload, dict):
                    exact_rows.append({
                        "case_id": case_id,
                        "chunk_idx": "",
                        "path": str(path),
                        "available": False,
                        "missing_reason": "payload_not_dict",
                    })
                    continue
                retention = payload.get("stable_anchor_retention_patch")
                residual = payload.get("stable_anchor_residual_patch")
                mask = payload.get("stable_anchor_mask_patch")
                energy = payload.get("U_ttt_write_replay_contribution_patch")
                prior = payload.get("ttt_write_prior_patch")
                if not (torch.is_tensor(retention) and torch.is_tensor(residual) and torch.is_tensor(mask)):
                    row = {
                        "case_id": case_id,
                        "chunk_idx": payload.get("chunk_idx", ""),
                        "path": str(path),
                        "available": False,
                        "missing_reason": "stable_anchor_retention/residual/mask patch missing",
                    }
                    exact_rows.append(row)
                    continue
                mask_bool = mask.detach().cpu().float() > 0.5
                ret_vals = retention.detach().cpu().float()[mask_bool]
                res_vals = residual.detach().cpu().float()[mask_bool]
                finite_ret = ret_vals[torch.isfinite(ret_vals)]
                finite_res = res_vals[torch.isfinite(res_vals)]
                energy_mean = ""
                if torch.is_tensor(energy):
                    e_vals = energy.detach().cpu().float()[mask_bool]
                    e_vals = e_vals[torch.isfinite(e_vals)]
                    if int(e_vals.numel()) > 0:
                        energy_mean = float(e_vals.mean().item())
                prior_mean = ""
                if torch.is_tensor(prior):
                    p_vals = prior.detach().cpu().float()[mask_bool]
                    p_vals = p_vals[torch.isfinite(p_vals)]
                    if int(p_vals.numel()) > 0:
                        prior_mean = float(p_vals.mean().item())
                provenance = payload.get("condition_map_provenance") or {}
                replay_provenance = provenance.get("replay_contribution") or {}
                row = {
                    "case_id": case_id,
                    "chunk_idx": payload.get("chunk_idx", ""),
                    "path": str(path),
                    "available": int(finite_ret.numel()) > 0 and int(finite_res.numel()) > 0,
                    "stable_anchor_token_count": int(mask_bool.sum().item()),
                    "stable_anchor_retention_exact": float(finite_ret.mean().item()) if int(finite_ret.numel()) > 0 else "",
                    "stable_anchor_residual_exact": float(finite_res.mean().item()) if int(finite_res.numel()) > 0 else "",
                    "stable_anchor_write_energy_exact": energy_mean,
                    "stable_anchor_write_prior_exact": prior_mean,
                    "retention_source": replay_provenance.get("stable_anchor_retention_source", ""),
                    "runtime_action_allowed_from_retention": False,
                    "missing_reason": "" if int(finite_ret.numel()) > 0 and int(finite_res.numel()) > 0 else "no finite stable-anchor retention values",
                }
                exact_rows.append(row)
                if b(row["available"]):
                    exact_by_case[case_id].append(row)

                def tensor_mean_on_mask(tensor: Any, bool_mask: Any) -> float:
                    if not torch.is_tensor(tensor):
                        return math.nan
                    values = tensor.detach().cpu().float()[bool_mask]
                    values = values[torch.isfinite(values)]
                    return float(values.mean().item()) if int(values.numel()) > 0 else math.nan

                def entropy_uniformity(tensor: Any) -> tuple[float, float, int]:
                    if not torch.is_tensor(tensor):
                        return math.nan, math.nan, 0
                    values = tensor.detach().cpu().float().reshape(-1)
                    values = values[torch.isfinite(values)]
                    values = torch.clamp(values, min=0.0)
                    if int(values.numel()) <= 1:
                        return math.nan, math.nan, int(values.numel())
                    total = values.sum()
                    if not torch.isfinite(total) or float(total.item()) <= 0.0:
                        return math.nan, math.nan, int(values.numel())
                    probs = values / total
                    entropy = float((-(probs * torch.log(probs + 1.0e-12))).sum().item())
                    uniformity = entropy / math.log(float(values.numel()))
                    return entropy, uniformity, int(values.numel())

                def top_quantile_mean(write_tensor: Any, score_tensor: Any, quantile: float = 0.90) -> tuple[float, int]:
                    if not (torch.is_tensor(write_tensor) and torch.is_tensor(score_tensor)):
                        return math.nan, 0
                    write_values = write_tensor.detach().cpu().float().reshape(-1)
                    score_values = score_tensor.detach().cpu().float().reshape(-1)
                    finite = torch.isfinite(write_values) & torch.isfinite(score_values)
                    if int(finite.sum().item()) == 0:
                        return math.nan, 0
                    write_values = write_values[finite]
                    score_values = score_values[finite]
                    threshold = torch.quantile(score_values, float(quantile))
                    selected = score_values >= threshold
                    if int(selected.sum().item()) == 0:
                        return math.nan, 0
                    return float(write_values[selected].mean().item()), int(selected.sum().item())

                risk_sources = {
                    "D_tok_top10": payload.get("D_tok_patch"),
                    "scale_risk_replay_top10": payload.get("S_scale_risk_replay_contribution_patch"),
                    "conflict_replay_top10": payload.get("C_ttt_conflict_replay_contribution_patch"),
                }
                for write_map_name, write_map, gate_family, note in [
                    (
                        "ttt_write_prior_patch",
                        prior,
                        "write_map",
                        "diagnostic TTT write prior map; can test missing-good-write distribution but cannot by itself authorize runtime action",
                    ),
                    (
                        "U_ttt_write_replay_contribution_patch",
                        energy,
                        "diagnostic_energy",
                        "projected write-energy/replay contribution map; diagnostic-only because it is not the configured write prior",
                    ),
                ]:
                    if not torch.is_tensor(write_map):
                        continue
                    write_entropy, write_uniformity, finite_count = entropy_uniformity(write_map)
                    stable_mass = tensor_mean_on_mask(write_map, mask_bool)
                    risk_masses: dict[str, float] = {}
                    risk_counts: dict[str, int] = {}
                    for risk_name, risk_tensor in risk_sources.items():
                        value, count = top_quantile_mean(write_map, risk_tensor)
                        risk_masses[risk_name] = value
                        risk_counts[risk_name] = count
                    finite_risk_items = [
                        (risk_name, value)
                        for risk_name, value in risk_masses.items()
                        if math.isfinite(value)
                    ]
                    best_risk_source = max(finite_risk_items, key=lambda item: item[1])[0] if finite_risk_items else ""
                    best_risk_mass = dict(finite_risk_items).get(best_risk_source, math.nan) if finite_risk_items else math.nan
                    exact_write_row = {
                        "case_id": case_id,
                        "chunk_idx": payload.get("chunk_idx", ""),
                        "path": str(path),
                        "write_map": write_map_name,
                        "gate_family": gate_family,
                        "available": torch.is_tensor(write_map) and math.isfinite(write_uniformity),
                        "finite_write_token_count": finite_count,
                        "stable_anchor_token_count": int(mask_bool.sum().item()),
                        "write_entropy_exact": write_entropy if math.isfinite(write_entropy) else "",
                        "write_uniformity_score_exact": write_uniformity if math.isfinite(write_uniformity) else "",
                        "stable_anchor_write_mass_exact": stable_mass if math.isfinite(stable_mass) else "",
                        "risk_write_mass_exact": best_risk_mass if math.isfinite(best_risk_mass) else "",
                        "risk_write_mass_source": best_risk_source,
                        "missing_good_write_score_exact": stable_mass - best_risk_mass
                        if math.isfinite(stable_mass) and math.isfinite(best_risk_mass)
                        else "",
                        "D_tok_top10_write_mass": risk_masses.get("D_tok_top10", ""),
                        "D_tok_top10_token_count": risk_counts.get("D_tok_top10", 0),
                        "scale_risk_replay_top10_write_mass": risk_masses.get("scale_risk_replay_top10", ""),
                        "scale_risk_replay_top10_token_count": risk_counts.get("scale_risk_replay_top10", 0),
                        "conflict_replay_top10_write_mass": risk_masses.get("conflict_replay_top10", ""),
                        "conflict_replay_top10_token_count": risk_counts.get("conflict_replay_top10", 0),
                        "runtime_action_allowed_from_write_map": False,
                        "diagnostic_note": note,
                    }
                    exact_write_rows.append(exact_write_row)
                    if b(exact_write_row.get("available")):
                        exact_write_by_case[case_id].append(exact_write_row)
    write_rows(out / "exact_write_map_audit_rows.csv", exact_write_rows)
    write_rows(out / "exact_retention_rows.csv", exact_rows)

    rows = []
    for row in sources["ttt_rows"]:
        case_id = row.get("case_id", "")
        exact_case_rows = exact_by_case.get(case_id, [])
        exact_write_case_rows = exact_write_by_case.get(case_id, [])

        def median_write_metric(write_map_name: str, field: str) -> float:
            return median([
                f(item.get(field))
                for item in exact_write_case_rows
                if item.get("write_map") == write_map_name
            ])

        exact_retention = median([f(item.get("stable_anchor_retention_exact")) for item in exact_case_rows])
        exact_residual = median([f(item.get("stable_anchor_residual_exact")) for item in exact_case_rows])
        exact_energy = median([f(item.get("stable_anchor_write_energy_exact")) for item in exact_case_rows])
        exact_prior = median([f(item.get("stable_anchor_write_prior_exact")) for item in exact_case_rows])
        exact_prior_uniformity = median_write_metric("ttt_write_prior_patch", "write_uniformity_score_exact")
        exact_prior_risk_write = median_write_metric("ttt_write_prior_patch", "risk_write_mass_exact")
        exact_prior_missing_good = median_write_metric("ttt_write_prior_patch", "missing_good_write_score_exact")
        exact_energy_uniformity = median_write_metric("U_ttt_write_replay_contribution_patch", "write_uniformity_score_exact")
        exact_energy_risk_write = median_write_metric("U_ttt_write_replay_contribution_patch", "risk_write_mass_exact")
        exact_energy_missing_good = median_write_metric("U_ttt_write_replay_contribution_patch", "missing_good_write_score_exact")
        stable_write = f(row.get("persistent_write_mass"))
        transient = f(row.get("transient_write_mass"))
        no_write = f(row.get("no_write_mass"))
        rows.append({
            "case_id": case_id,
            "bucket": row.get("bucket", ""),
            "stable_anchor_write_mass_proxy": stable_write,
            "missing_good_write_score_proxy": stable_write - max(transient, no_write),
            "write_uniformity_score_proxy": row.get("write_mass_mean", ""),
            "stable_anchor_retention": exact_retention if math.isfinite(exact_retention) else "",
            "stable_anchor_residual": exact_residual if math.isfinite(exact_residual) else "",
            "stable_anchor_write_energy_exact": exact_energy if math.isfinite(exact_energy) else "",
            "stable_anchor_write_prior_exact": exact_prior if math.isfinite(exact_prior) else "",
            "exact_prior_write_uniformity_score": exact_prior_uniformity if math.isfinite(exact_prior_uniformity) else "",
            "exact_prior_risk_write_mass": exact_prior_risk_write if math.isfinite(exact_prior_risk_write) else "",
            "exact_prior_missing_good_write_score": exact_prior_missing_good if math.isfinite(exact_prior_missing_good) else "",
            "exact_energy_write_uniformity_score": exact_energy_uniformity if math.isfinite(exact_energy_uniformity) else "",
            "exact_energy_risk_write_mass": exact_energy_risk_write if math.isfinite(exact_energy_risk_write) else "",
            "exact_energy_missing_good_write_score": exact_energy_missing_good if math.isfinite(exact_energy_missing_good) else "",
            "exact_retention_chunk_count": len(exact_case_rows),
            "exact_write_map_chunk_count": len(exact_write_case_rows),
            "retention_missing_reason": ""
            if math.isfinite(exact_retention)
            else "exact stable-anchor retention map absent for this case",
            "condition_map_source": row.get("condition_map_source", ""),
            "condition_replay_contribution_not_runtime_eligible": row.get("condition_replay_contribution_not_runtime_eligible", ""),
            "best_component_risk_enrichment": row.get("best_component_risk_enrichment", ""),
            "gate_risk_enrichment": row.get("gate_risk_enrichment", ""),
        })
    risk = [row for row in rows if row["bucket"] == "ttt_write_risk"]
    good = [row for row in rows if row["bucket"] == "good_control"]
    stable_margin = median([f(row["stable_anchor_write_mass_proxy"]) for row in risk]) - median([f(row["stable_anchor_write_mass_proxy"]) for row in good])
    uniform_margin = median([f(row["write_uniformity_score_proxy"]) for row in risk]) - median([f(row["write_uniformity_score_proxy"]) for row in good])
    exact_values_by_case = {
        row["case_id"]: f(row.get("stable_anchor_retention"))
        for row in rows
        if math.isfinite(f(row.get("stable_anchor_retention")))
    }
    risk_cases = {row["case_id"] for row in risk if row["case_id"] in exact_values_by_case}
    good_cases = {row["case_id"] for row in good if row["case_id"] in exact_values_by_case}
    exact_retention_margin = median([exact_values_by_case[case] for case in risk_cases]) - median([
        exact_values_by_case[case] for case in good_cases
    ])
    threshold_audit = best_threshold(
        exact_values_by_case,
        risk_cases,
        good_cases,
        higher_bad=False,
    ) if risk_cases and good_cases else {
        "balanced_accuracy": 0.0,
        "threshold": math.nan,
        "tp": 0,
        "tn": 0,
        "pos": len(risk_cases),
        "neg": len(good_cases),
    }
    selected = selected_from_threshold(
        exact_values_by_case,
        f(threshold_audit.get("threshold")),
        higher_bad=False,
    )
    good_fpr = len(selected & good_cases) / len(good_cases) if good_cases else 1.0
    retention_gate = (
        len(risk_cases) >= 3
        and len(good_cases) >= 3
        and f(threshold_audit.get("balanced_accuracy"), 0.0) >= 0.70
        and math.isfinite(exact_retention_margin)
        and exact_retention_margin <= -0.05
        and good_fpr <= 0.25
    )

    atlas, _sem, _seq = case_maps(sources)
    all_cases = sorted({row["case_id"] for row in rows if row.get("case_id")})
    l3_by_case = {
        case: f(atlas.get(case, {}).get("L3_handoff_transfer_penalty_proxy"))
        for case in all_cases
    }

    def audit_retention_metric(
        metric_name: str,
        field: str,
        *,
        higher_bad: bool,
        retention_family: bool,
        gate_family: str | None = None,
        diagnostic_note: str,
    ) -> dict[str, Any]:
        family = gate_family or ("retention" if retention_family else "diagnostic")
        values = {
            row["case_id"]: f(row.get(field))
            for row in rows
            if math.isfinite(f(row.get(field)))
        }
        metric_risk_cases = {case for case in risk_cases | {row["case_id"] for row in risk} if case in values}
        metric_good_cases = {case for case in good_cases | {row["case_id"] for row in good} if case in values}
        risk_median = median([values[case] for case in metric_risk_cases])
        good_median = median([values[case] for case in metric_good_cases])
        risk_minus_good = risk_median - good_median if math.isfinite(risk_median) and math.isfinite(good_median) else math.nan
        audit = best_threshold(values, metric_risk_cases, metric_good_cases, higher_bad=higher_bad) if metric_risk_cases and metric_good_cases else {
            "balanced_accuracy": 0.0,
            "threshold": math.nan,
            "direction": "higher_bad" if higher_bad else "lower_bad",
            "tp": 0,
            "tn": 0,
            "pos": len(metric_risk_cases),
            "neg": len(metric_good_cases),
        }
        selected_cases = selected_from_threshold(values, f(audit.get("threshold")), higher_bad=higher_bad)
        metric_good_fpr = len(selected_cases & metric_good_cases) / len(metric_good_cases) if metric_good_cases else 1.0
        xs = [values.get(case, math.nan) for case in all_cases]
        ys = [l3_by_case.get(case, math.nan) for case in all_cases]
        corr = pearson(xs, ys)
        expected_margin_ok = (
            math.isfinite(risk_minus_good)
            and (risk_minus_good >= 0.05 if higher_bad else risk_minus_good <= -0.05)
        )
        gate_eligible = family in {"retention", "write_mass", "write_uniformity"}
        separation_gate = (
            gate_eligible
            and len(metric_risk_cases) >= 3
            and len(metric_good_cases) >= 3
            and f(audit.get("balanced_accuracy"), 0.0) >= 0.70
            and metric_good_fpr <= 0.25
            and expected_margin_ok
        )
        long_drift_gate = separation_gate and math.isfinite(corr) and abs(corr) >= 0.30
        return {
            "metric": metric_name,
            "field": field,
            "retention_family": retention_family,
            "gate_family": family,
            "direction": audit.get("direction", "higher_bad" if higher_bad else "lower_bad"),
            "risk_case_count": len(metric_risk_cases),
            "good_case_count": len(metric_good_cases),
            "risk_median": risk_median if math.isfinite(risk_median) else "",
            "good_median": good_median if math.isfinite(good_median) else "",
            "risk_minus_good": risk_minus_good if math.isfinite(risk_minus_good) else "",
            "balanced_accuracy": audit.get("balanced_accuracy", 0.0),
            "threshold": audit.get("threshold", ""),
            "good_fpr": metric_good_fpr,
            "pearson_with_L3_handoff_transfer_penalty": corr if math.isfinite(corr) else "",
            "abs_corr_L3_handoff_transfer_penalty": abs(corr) if math.isfinite(corr) else "",
            "separation_gate_pass": separation_gate,
            "long_drift_gate_pass": long_drift_gate,
            "diagnostic_note": diagnostic_note,
        }

    retention_definition_rows = [
        audit_retention_metric(
            "stable_anchor_retention_exact",
            "stable_anchor_retention",
            higher_bad=False,
            retention_family=True,
            diagnostic_note="original exact stable-anchor retention; risk should be lower than good to support missing-good-retention",
        ),
        audit_retention_metric(
            "stable_anchor_residual_exact",
            "stable_anchor_residual",
            higher_bad=True,
            retention_family=True,
            diagnostic_note="alternate exact residual definition; risk should be higher than good and correlate with long-drift proxy",
        ),
        audit_retention_metric(
            "stable_anchor_write_energy_exact",
            "stable_anchor_write_energy_exact",
            higher_bad=True,
            retention_family=False,
            diagnostic_note="diagnostic write-energy/control-map quantity only; high correlation cannot be promoted as retention evidence",
        ),
        audit_retention_metric(
            "stable_anchor_write_prior_exact",
            "stable_anchor_write_prior_exact",
            higher_bad=True,
            retention_family=False,
            diagnostic_note="diagnostic write-prior quantity only; not before/after stable-anchor retention",
        ),
        audit_retention_metric(
            "exact_prior_stable_anchor_write_mass",
            "stable_anchor_write_prior_exact",
            higher_bad=False,
            retention_family=False,
            gate_family="write_mass",
            diagnostic_note="exact TTT write-prior mass over stable-anchor tokens; risk should be lower than good to support missing-good-write",
        ),
        audit_retention_metric(
            "exact_prior_missing_good_write_score",
            "exact_prior_missing_good_write_score",
            higher_bad=False,
            retention_family=False,
            gate_family="write_mass",
            diagnostic_note="stable-anchor prior mass minus strongest risk-top10 prior mass; more negative in risk cases would support missing-good-write",
        ),
        audit_retention_metric(
            "exact_prior_write_uniformity_score",
            "exact_prior_write_uniformity_score",
            higher_bad=True,
            retention_family=False,
            gate_family="write_uniformity",
            diagnostic_note="entropy/logN of the exact TTT write-prior map; risk should be more uniform by >=0.05 for the uniform-write hypothesis",
        ),
        audit_retention_metric(
            "exact_energy_missing_good_write_score",
            "exact_energy_missing_good_write_score",
            higher_bad=False,
            retention_family=False,
            diagnostic_note="projected write-energy stable-minus-risk score; diagnostic-only because it is not the configured write-prior map",
        ),
        audit_retention_metric(
            "exact_energy_write_uniformity_score",
            "exact_energy_write_uniformity_score",
            higher_bad=True,
            retention_family=False,
            diagnostic_note="entropy/logN of projected write-energy/replay contribution; diagnostic-only non-prior quantity",
        ),
    ]
    write_rows(out / "retention_definition_audit_rows.csv", retention_definition_rows)
    retention_family_rows = [row for row in retention_definition_rows if b(row.get("retention_family"))]
    write_family_rows = [
        row for row in retention_definition_rows
        if row.get("gate_family") in {"write_mass", "write_uniformity"}
    ]
    new_definition_gate = any(b(row.get("separation_gate_pass")) for row in retention_family_rows)
    long_drift_gate = any(b(row.get("long_drift_gate_pass")) for row in retention_family_rows)
    write_map_gate = any(b(row.get("long_drift_gate_pass")) for row in write_family_rows)
    best_retention_family_corr = max(
        [f(row.get("abs_corr_L3_handoff_transfer_penalty")) for row in retention_family_rows],
        default=math.nan,
    )
    best_write_family_corr = max(
        [f(row.get("abs_corr_L3_handoff_transfer_penalty")) for row in write_family_rows],
        default=math.nan,
    )
    high_corr_non_retention_rows = [
        row for row in retention_definition_rows
        if row.get("gate_family") == "diagnostic" and f(row.get("abs_corr_L3_handoff_transfer_penalty")) >= 0.30
    ]
    gate = (new_definition_gate and long_drift_gate) or write_map_gate
    write_rows(out / "ttt_retention_rows.csv", rows)
    exact_retention_available = bool(exact_values_by_case)
    exact_retention_full_case_coverage = len(exact_values_by_case) == len(rows) and len(rows) > 0
    retention_metric = next((row for row in retention_definition_rows if row["metric"] == "stable_anchor_retention_exact"), {})
    residual_metric = next((row for row in retention_definition_rows if row["metric"] == "stable_anchor_residual_exact"), {})
    energy_metric = next((row for row in retention_definition_rows if row["metric"] == "stable_anchor_write_energy_exact"), {})
    prior_metric = next((row for row in retention_definition_rows if row["metric"] == "stable_anchor_write_prior_exact"), {})
    summary = {
        "schema": "acl2_v97_trackF2_ttt_missing_good_write_v1",
        "status": "complete_diagnostic_no_action",
        "case_rows": len(rows),
        "risk_case_count": len(risk),
        "good_control_count": len(good),
        "stable_anchor_write_mass_risk_minus_good": stable_margin,
        "write_uniformity_risk_minus_good": uniform_margin,
        "stable_anchor_retention_available": exact_retention_available,
        "exact_retention_chunk_count": len([row for row in exact_rows if b(row.get("available"))]),
        "exact_retention_case_count": len(exact_values_by_case),
        "exact_retention_full_case_coverage": exact_retention_full_case_coverage,
        "exact_write_map_audit_row_count": len(exact_write_rows),
        "exact_write_map_case_count": len(exact_write_by_case),
        "stable_anchor_retention_risk_minus_good": exact_retention_margin
        if math.isfinite(exact_retention_margin)
        else "",
        "stable_anchor_retention_balanced_accuracy": threshold_audit.get("balanced_accuracy", 0.0),
        "stable_anchor_retention_good_fpr": good_fpr,
        "stable_anchor_retention_threshold": threshold_audit.get("threshold", ""),
        "stable_anchor_retention_gate_pass": retention_gate,
        "retention_definition_audit_available": bool(retention_definition_rows),
        "retention_definition_audit_row_count": len(retention_definition_rows),
        "stable_anchor_retention_abs_corr_L3_handoff": retention_metric.get("abs_corr_L3_handoff_transfer_penalty", ""),
        "stable_anchor_residual_risk_minus_good": residual_metric.get("risk_minus_good", ""),
        "stable_anchor_residual_balanced_accuracy": residual_metric.get("balanced_accuracy", 0.0),
        "stable_anchor_residual_good_fpr": residual_metric.get("good_fpr", ""),
        "stable_anchor_residual_abs_corr_L3_handoff": residual_metric.get("abs_corr_L3_handoff_transfer_penalty", ""),
        "stable_anchor_residual_gate_pass": residual_metric.get("long_drift_gate_pass", False),
        "stable_anchor_write_energy_exact_abs_corr_L3_handoff": energy_metric.get("abs_corr_L3_handoff_transfer_penalty", ""),
        "stable_anchor_write_prior_exact_abs_corr_L3_handoff": prior_metric.get("abs_corr_L3_handoff_transfer_penalty", ""),
        "f2_new_retention_definition_gate_pass": new_definition_gate,
        "f2_long_drift_correlation_gate_pass": long_drift_gate,
        "f2_exact_write_map_gate_pass": write_map_gate,
        "best_retention_family_abs_corr_L3_handoff": best_retention_family_corr if math.isfinite(best_retention_family_corr) else "",
        "best_write_family_abs_corr_L3_handoff": best_write_family_corr if math.isfinite(best_write_family_corr) else "",
        "high_corr_non_retention_metrics": [row.get("metric") for row in high_corr_non_retention_rows],
        "high_corr_non_retention_metric_note": (
            "Some projected write-energy/replay-contribution diagnostics may correlate with L3, but they are not configured write-prior mass or before/after stable-anchor retention and remain action-ineligible."
            if high_corr_non_retention_rows
            else ""
        ),
        "exact_retention_hook_feasibility_checked": True,
        "exact_retention_hook_candidate_files": [
            "run_pipeline_abc_v2.py:_dump_ttt_spatial_post_delta_map",
            "loger/pipeline/hybrid_memory_controller.py:HybridMemoryController.run_probe/run_controlled",
            "loger/pipeline/ttt_write_controller.py:TTT write cache update path",
        ],
        "existing_ttt_dump_proxy_fields": [
            "ttt_write_prior_patch",
            "D_tok_patch",
            "R_ttt_tok_patch",
            "C_ttt_conflict_replay_contribution_patch",
            "S_scale_risk_replay_contribution_patch",
            "C_ttt_conflict_proxy_patch",
            "S_scale_risk_proxy_patch",
        ],
        "exact_retention_hook_blocker": (
            ""
            if exact_retention_available
            else (
                "existing v68/v96 TTT dumps expose write priors, role maps, replay/proxy condition maps, "
                "and projected post-delta norms, but not stable-anchor latent features before and after the "
                "TTT fast-weight/cache write; adding F2 exact retention requires a diagnostic-only hook around "
                "the TTT write cache update, not a parser-only repair"
            )
        ),
        "gate_pass": gate,
        "classification": "TTT_STABLE_RETENTION_SIGNAL_DIAGNOSTIC_PASS_NO_ACTION"
        if gate
        else "TTT_EXACT_WRITE_MAP_AVAILABLE_UNIFORMITY_MISSING_GOOD_GATE_FAIL"
        if exact_write_rows
        else "TTT_STABLE_RETENTION_EXACT_AVAILABLE_NEW_DEFINITION_LONG_DRIFT_GATE_FAIL"
        if exact_retention_full_case_coverage
        else "TTT_STABLE_RETENTION_PARTIAL_EXACT_AVAILABLE_GATE_FAIL"
        if exact_retention_available
        else "TTT_STABLE_RETENTION_BLOCKED_EXACT_MAPS_MISSING",
        "runtime_action_allowed": False,
    }
    write_json(out / "summary.json", summary)
    if exact_retention_available:
        write_text(out / "missing_good_write_report.md", (
            "# Missing-Good-Write Report\n\n"
            "v97 added a diagnostic-only stable-anchor retention dump around TTT replay contribution. "
            f"Exact retention case coverage: {len(exact_values_by_case)}/{len(rows)}; "
            f"exact write-map audit rows: {len(exact_write_rows)}; "
            f"risk-minus-good retention margin: {exact_retention_margin if math.isfinite(exact_retention_margin) else 'missing'}; "
            f"BA: {threshold_audit.get('balanced_accuracy', 0.0)}; good FPR: {good_fpr}; "
            f"retention-family best abs L3 correlation: {summary['best_retention_family_abs_corr_L3_handoff']}; "
            f"write-map-family best abs L3 correlation: {summary['best_write_family_abs_corr_L3_handoff']}. "
            "The maps remain diagnostic-only and do not authorize a TTT runtime action. "
            "Exact write-prior mass/uniformity rows are evaluated against the missing-good-write gate, while projected write-energy rows are reported separately as diagnostic-only non-prior quantities."
        ))
        write_text(out / "failure_report.md", (
            "# Track F2 Failure Report\n\n"
            "Exact diagnostic retention maps are now available, and v97 also audited an alternate residual definition, exact write-prior mass/uniformity, and long-drift L3 correlation. "
            "The F2 gate still fails because retention/residual separation is too weak and exact write-prior missing-good/uniformity margins do not satisfy the plan thresholds; high projected write-energy correlations, where present, are diagnostic-only non-prior quantities. "
            f"Current classification: `{summary['classification']}`."
        ))
    else:
        write_text(out / "missing_good_write_report.md", (
            "# Missing-Good-Write Report\n\n"
            "v96 TrackF replay-contribution diagnostics are available, but exact stable-anchor retention maps are not. "
            "Proxy rows are kept diagnostic-only; no TTT runtime action is allowed.\n\n"
            "Feasibility check: `run_pipeline_abc_v2.py` already dumps TTT write priors, D/R role maps, replay/proxy condition maps, and projected post-delta norms. "
            "The exact F2 retention quantity would require stable-anchor latent features immediately before and after the TTT fast-weight/cache write. "
            "Those tensors are not present in the current dump payload, so the safe repair direction is a new diagnostic-only hook around the TTT write cache update path in `loger/pipeline/ttt_write_controller.py` / `loger/pipeline/hybrid_memory_controller.py`."
        ))
        write_text(out / "failure_report.md", "# Track F2 Failure Report\n\nThe bad-write/fixed-pair path already failed in v96. v97 F2 cannot promote missing-good-write without exact stable-anchor write/retention evidence.")
    write_text(out / "what_would_have_to_be_true_to_pass.md", "# What Would Have To Be True\n\nExact C/S maps and stable-anchor before/after retention or residual must show a risk/control margin with good FPR <= 0.25 and long-drift correlation >= 0.30. Write-energy/prior correlations alone are insufficient because they are not before/after stable-anchor retention.")
    return summary


def build_e2(root: Path, sources: dict[str, Any]) -> dict[str, Any]:
    out = root / "trackE2_swa_carrier_search_beyond_route_mass"
    atlas, _sem, seq = case_maps(sources)
    cases = []
    pos: set[str] = set()
    neg: set[str] = set()
    values_by_metric: dict[str, dict[str, float]] = defaultdict(dict)
    swa_by_case = {row.get("case_id", ""): row for row in sources["swa_rows"] if row.get("case_id")}
    specs = {
        "stable_pair_mass_lower_bad": ("trace_swa_raw_transport_stable_pair_mass_mean", False),
        "unreliable_pair_mass_higher_bad": ("trace_swa_raw_transport_unreliable_pair_mass_mean", True),
        "feature_residual_higher_bad": ("trace_swa_raw_transport_feature_residual_mean", True),
        "route_entropy_higher_bad": ("trace_swa_raw_transport_route_entropy_mean", True),
        "cache_k_stability_lower_bad": ("trace_swa_raw_transport_cache_k_stability_mean", False),
        "cache_v_stability_lower_bad": ("trace_swa_raw_transport_cache_v_stability_mean", False),
        "qk_similarity_mean_lower_bad": ("trace_swa_raw_transport_qk_similarity_mean", False),
    }
    for row in sources["swa_rows"]:
        case_id = row.get("case_id", "")
        cases.append(case_id)
        if row.get("bucket") == "SWA_HANDOFF_GOOD_CONTROL":
            neg.add(case_id)
        else:
            pos.add(case_id)
        for metric, (field, _higher_bad) in specs.items():
            values_by_metric[metric][case_id] = f(row.get(field))
    carrier_rows = []
    carrier_case_rows = []
    all_cases = sorted(set(cases))
    for metric, (_field, higher_bad) in specs.items():
        values = values_by_metric[metric]
        best = best_threshold(values, pos, neg, higher_bad=higher_bad)
        selected = selected_from_threshold(values, best["threshold"], higher_bad=higher_bad)
        l3 = [f(atlas.get(case, {}).get("L3_handoff_transfer_penalty_proxy")) for case in all_cases]
        xs = [values.get(case, math.nan) for case in all_cases]
        same_margin = same_count_margin(selected, all_cases, pos, neg)
        seq_margin = sequence_margin(selected, all_cases, pos, neg, seq)
        abs_corr = abs(pearson(xs, l3))
        tp_cases = sorted(selected & pos)
        fp_cases = sorted(selected & neg)
        fn_cases = sorted(pos - selected)
        tn_cases = sorted(neg - selected)
        gate = (
            best["balanced_accuracy"] >= 0.70
            and abs_corr >= 0.30
            and same_margin >= 0.05
            and seq_margin >= 0.05
            and f(sources["swa_decision"].get("stable_group_nonempty_frac"), 0.0) >= 0.50
        )
        carrier_rows.append({
            "carrier_metric": metric,
            "best_balanced_accuracy": best["balanced_accuracy"],
            "threshold": best["threshold"],
            "direction": best["direction"],
            "abs_correlation_with_L3_handoff_transfer_penalty": abs_corr,
            "same_count_margin": same_margin,
            "sequence_margin": seq_margin,
            "strict_stable_or_anchor_group_coverage": sources["swa_decision"].get("stable_group_nonempty_frac", ""),
            "positive_case_count": len(pos),
            "good_control_case_count": len(neg),
            "selected_case_count": len(selected),
            "tp": len(tp_cases),
            "fp": len(fp_cases),
            "fn": len(fn_cases),
            "tn": len(tn_cases),
            "true_positive_cases": ";".join(tp_cases),
            "false_positive_cases": ";".join(fp_cases),
            "missed_positive_cases": ";".join(fn_cases),
            "gate_pass": gate,
        })
        for case_id in all_cases:
            selected_case = case_id in selected
            label_bad = case_id in pos
            swa_row = swa_by_case.get(case_id, {})
            carrier_case_rows.append({
                "carrier_metric": metric,
                "case_id": case_id,
                "seq": seq.get(case_id, ""),
                "bucket": swa_row.get("bucket", ""),
                "label_bad": label_bad,
                "value": values.get(case_id, math.nan),
                "threshold": best["threshold"],
                "direction": best["direction"],
                "predicted_bad": selected_case,
                "outcome": (
                    "tp" if selected_case and label_bad
                    else "fp" if selected_case and not label_bad
                    else "fn" if (not selected_case) and label_bad
                    else "tn"
                ),
                "L3_handoff_transfer_penalty_proxy": atlas.get(case_id, {}).get("L3_handoff_transfer_penalty_proxy", ""),
                "trace_available_frac": swa_row.get("trace_swa_raw_transport_available_frac", ""),
                "stable_group_nonempty_frac": swa_row.get("trace_swa_raw_transport_stable_nonempty_frac", ""),
                "unreliable_group_nonempty_frac": swa_row.get("trace_swa_raw_transport_unreliable_nonempty_frac", ""),
                "metric_gate_pass": gate,
            })
    passed = [row for row in carrier_rows if row["gate_pass"]]
    write_rows(out / "carrier_rows.csv", carrier_rows)
    write_rows(out / "carrier_case_audit_rows.csv", carrier_case_rows)
    passed_details = [
        {
            "carrier_metric": row["carrier_metric"],
            "best_balanced_accuracy": row["best_balanced_accuracy"],
            "threshold": row["threshold"],
            "direction": row["direction"],
            "abs_correlation_with_L3_handoff_transfer_penalty": row["abs_correlation_with_L3_handoff_transfer_penalty"],
            "same_count_margin": row["same_count_margin"],
            "sequence_margin": row["sequence_margin"],
            "tp": row["tp"],
            "fp": row["fp"],
            "fn": row["fn"],
            "tn": row["tn"],
            "false_positive_cases": row["false_positive_cases"],
            "missed_positive_cases": row["missed_positive_cases"],
        }
        for row in passed
    ]
    per_head_carrier_rows: list[dict[str, Any]] = []
    per_head_case_rows: list[dict[str, Any]] = []
    per_head_read_errors: list[dict[str, Any]] = []
    per_head_payload_file_count = 0
    per_head_specs = {
        "per_head_cache_k_stability_lower_bad": ("cache_K_stability_by_head", False),
        "per_head_cache_v_stability_lower_bad": ("cache_V_stability_by_head", False),
        "per_head_q_to_cache_k_similarity_mean_lower_bad": ("current_Q_to_cache_K_similarity_mean_by_head", False),
        "per_head_q_to_cache_k_similarity_max_lower_bad": ("current_Q_to_cache_K_similarity_max_by_head", False),
        "per_head_route_entropy_higher_bad": ("route_entropy_mean_by_head", True),
        "per_head_feature_residual_higher_bad": ("feature_transport_residual_by_head", True),
        "per_head_stable_pair_mass_lower_bad": ("stable_structure_pair_mass_by_head", False),
        "per_head_unreliable_pair_mass_higher_bad": ("unreliable_dynamic_boundary_pair_mass_by_head", True),
        "per_head_stable_actual_minus_random_lower_bad": ("stable_route_actual_minus_random_by_head", False),
        "per_head_unreliable_actual_minus_random_higher_bad": ("unreliable_route_actual_minus_random_by_head", True),
        "per_head_top1_cache_index_unique_frac_higher_bad": (
            "current_Q_to_cache_K_top1_cache_index_unique_frac_by_head", True
        ),
        "per_head_top1_cache_frame_unique_frac_higher_bad": (
            "current_Q_to_cache_K_top1_cache_frame_unique_frac_by_head", True
        ),
        "per_head_top1_cache_index_switch_rate_higher_bad": (
            "current_Q_to_cache_K_top1_cache_index_switch_rate_by_head", True
        ),
        "per_head_top1_cache_frame_switch_rate_higher_bad": (
            "current_Q_to_cache_K_top1_cache_frame_switch_rate_by_head", True
        ),
        "per_head_top1_same_frame_frac_lower_bad": (
            "current_Q_to_cache_K_top1_same_frame_frac_by_head", False
        ),
        "per_head_topk_query_frame_hit_frac_lower_bad": (
            "current_Q_to_cache_K_topk_query_frame_hit_frac_by_head", False
        ),
        "per_head_topk_same_frame_frac_lower_bad": (
            "current_Q_to_cache_K_topk_same_frame_frac_by_head", False
        ),
        "per_head_top1_abs_frame_delta_higher_bad": (
            "current_Q_to_cache_K_top1_abs_frame_delta_mean_by_head", True
        ),
    }
    per_head_values: dict[tuple[str, int, int, int, bool], dict[str, float]] = defaultdict(dict)
    raw_trace_root = Path(str(sources["v96_root"])) / "trackE_swa_raw_transport_trace_swa_atlas_v1"
    topk_identity_payload_file_count = 0
    try:
        import torch  # noqa: PLC0415

        for path in swa_raw_trace_payload_paths(root, sources, include_v96=True, include_v97_topk=True):
            case_id = path.parents[2].name
            try:
                payload = torch.load(path, map_location="cpu")
            except Exception as exc:  # noqa: BLE001
                per_head_read_errors.append({"path": str(path), "error": f"{type(exc).__name__}:{exc}"})
                continue
            if not isinstance(payload, dict):
                per_head_read_errors.append({"path": str(path), "error": f"unexpected_payload_type:{type(payload).__name__}"})
                continue
            per_head_payload_file_count += 1
            if bool(payload.get("topk_identity_available")):
                topk_identity_payload_file_count += 1
            swa_layer_idx = int(f(payload.get("swa_layer_idx"), -1))
            actual_layer = int(f(payload.get("layer"), swa_layer_idx))
            for metric, (field, higher_bad) in per_head_specs.items():
                tensor = payload.get(field)
                if not hasattr(tensor, "detach"):
                    continue
                vals = tensor.detach().cpu().float().reshape(-1).tolist()
                for head_idx, value in enumerate(vals):
                    if math.isfinite(float(value)):
                        per_head_values[(metric, swa_layer_idx, actual_layer, int(head_idx), higher_bad)][case_id] = float(value)
    except Exception as exc:  # noqa: BLE001
        per_head_read_errors.append({"path": str(raw_trace_root), "error": f"{type(exc).__name__}:{exc}"})

    for (metric, swa_layer_idx, actual_layer, head_idx, higher_bad), values in sorted(per_head_values.items()):
        best = best_threshold(values, pos, neg, higher_bad=higher_bad)
        selected = selected_from_threshold(values, best["threshold"], higher_bad=higher_bad)
        l3 = [f(atlas.get(case, {}).get("L3_handoff_transfer_penalty_proxy")) for case in all_cases]
        xs = [values.get(case, math.nan) for case in all_cases]
        same_margin = same_count_margin(selected, all_cases, pos, neg)
        seq_margin = sequence_margin(selected, all_cases, pos, neg, seq)
        abs_corr = abs(pearson(xs, l3))
        finite_case_count = len([case for case in all_cases if math.isfinite(values.get(case, math.nan))])
        tp_cases = sorted(selected & pos)
        fp_cases = sorted(selected & neg)
        fn_cases = sorted(pos - selected)
        tn_cases = sorted(neg - selected)
        gate = (
            finite_case_count == len(all_cases)
            and best["balanced_accuracy"] >= 0.70
            and abs_corr >= 0.30
            and same_margin >= 0.05
            and seq_margin >= 0.05
            and f(sources["swa_decision"].get("stable_group_nonempty_frac"), 0.0) >= 0.50
        )
        per_head_carrier_rows.append({
            "carrier_metric": metric,
            "swa_layer_idx": swa_layer_idx,
            "actual_layer": actual_layer,
            "head_idx": head_idx,
            "best_balanced_accuracy": best["balanced_accuracy"],
            "threshold": best["threshold"],
            "direction": best["direction"],
            "abs_correlation_with_L3_handoff_transfer_penalty": abs_corr,
            "same_count_margin": same_margin,
            "sequence_margin": seq_margin,
            "finite_case_count": finite_case_count,
            "positive_case_count": len(pos),
            "good_control_case_count": len(neg),
            "selected_case_count": len(selected),
            "tp": len(tp_cases),
            "fp": len(fp_cases),
            "fn": len(fn_cases),
            "tn": len(tn_cases),
            "true_positive_cases": ";".join(tp_cases),
            "false_positive_cases": ";".join(fp_cases),
            "missed_positive_cases": ";".join(fn_cases),
            "gate_pass": gate,
        })
        if gate:
            for case_id in all_cases:
                selected_case = case_id in selected
                label_bad = case_id in pos
                per_head_case_rows.append({
                    "carrier_metric": metric,
                    "swa_layer_idx": swa_layer_idx,
                    "actual_layer": actual_layer,
                    "head_idx": head_idx,
                    "case_id": case_id,
                    "seq": seq.get(case_id, ""),
                    "bucket": swa_by_case.get(case_id, {}).get("bucket", ""),
                    "label_bad": label_bad,
                    "value": values.get(case_id, math.nan),
                    "threshold": best["threshold"],
                    "direction": best["direction"],
                    "predicted_bad": selected_case,
                    "outcome": (
                        "tp" if selected_case and label_bad
                        else "fp" if selected_case and not label_bad
                        else "fn" if (not selected_case) and label_bad
                        else "tn"
                    ),
                    "L3_handoff_transfer_penalty_proxy": atlas.get(case_id, {}).get("L3_handoff_transfer_penalty_proxy", ""),
                })
    write_rows(out / "per_head_carrier_rows.csv", per_head_carrier_rows)
    write_rows(out / "per_head_carrier_case_audit_rows.csv", per_head_case_rows or [{
        "available": False,
        "missing_reason": "no per-head carrier passed the E2 gate",
    }])
    write_rows(out / "per_head_read_errors.csv", per_head_read_errors)
    per_head_passed = [row for row in per_head_carrier_rows if row["gate_pass"]]
    per_head_passed_sorted = sorted(
        per_head_passed,
        key=lambda row: (
            f(row.get("best_balanced_accuracy"), 0.0),
            f(row.get("abs_correlation_with_L3_handoff_transfer_penalty"), 0.0),
            -f(row.get("fp"), 0.0),
        ),
        reverse=True,
    )
    per_head_top_details = [
        {
            "carrier_metric": row["carrier_metric"],
            "swa_layer_idx": row["swa_layer_idx"],
            "actual_layer": row["actual_layer"],
            "head_idx": row["head_idx"],
            "best_balanced_accuracy": row["best_balanced_accuracy"],
            "abs_correlation_with_L3_handoff_transfer_penalty": row["abs_correlation_with_L3_handoff_transfer_penalty"],
            "threshold": row["threshold"],
            "direction": row["direction"],
            "tp": row["tp"],
            "fp": row["fp"],
            "fn": row["fn"],
            "tn": row["tn"],
        }
        for row in per_head_passed_sorted[:10]
    ]
    summary = {
        "schema": "acl2_v97_trackE2_swa_carrier_search_v1",
        "status": "complete",
        "carrier_row_count": len(carrier_rows),
        "carrier_case_audit_row_count": len(carrier_case_rows),
        "per_head_payload_file_count": per_head_payload_file_count,
        "topk_identity_payload_file_count": topk_identity_payload_file_count,
        "per_head_carrier_row_count": len(per_head_carrier_rows),
        "per_head_carrier_gate_pass_count": len(per_head_passed),
        "per_head_carrier_case_audit_row_count": len(per_head_case_rows),
        "per_head_read_error_count": len(per_head_read_errors),
        "per_head_top_passed_carriers": per_head_top_details,
        "topk_identity_available": topk_identity_payload_file_count > 0,
        "topk_identity_missing_reason": (
            ""
            if topk_identity_payload_file_count > 0
            else (
                "v96 raw SWA transport trace stores sampled_query_indices and aggregate/per-head similarity/stability stats, "
                "but not cache top-k identity indices over time; top-k identity instability needs a new diagnostic hook."
            )
        ),
        "positive_case_count": len(pos),
        "good_control_case_count": len(neg),
        "gate_pass": bool(passed),
        "passed_carriers": [row["carrier_metric"] for row in passed],
        "passed_carrier_details": passed_details,
        "classification": "SWA_NEW_CARRIER_DIAGNOSTIC_PASS_ACTION_NOT_RUN" if passed else "SWA_TRACE_PASS_NOT_CARRIER",
        "runtime_action_allowed": False,
    }
    write_json(out / "summary.json", summary)
    if passed_details:
        passed_lines = "\n".join(
            "- {carrier_metric}: BA={best_balanced_accuracy}, abs_L3_corr={abs_correlation_with_L3_handoff_transfer_penalty}, "
            "threshold={threshold}, direction={direction}, margins=({same_count_margin},{sequence_margin}), "
            "TP/FP/FN/TN={tp}/{fp}/{fn}/{tn}, false_positive_cases={false_positive_cases}, missed_positive_cases={missed_positive_cases}".format(**row)
            for row in passed_details
        )
    else:
        passed_lines = "No carrier passed the E2 diagnostic gate."
    write_text(out / "failure_report.md", (
        "# Track E2 Report\n\n"
        f"SWA carrier candidates were evaluated from {len(all_cases)} v96 raw transport trace cases "
        f"({len(pos)} non-good handoff cases, {len(neg)} good controls). "
        "Carrier rows and case-level threshold outcomes are written to `carrier_rows.csv` and `carrier_case_audit_rows.csv`.\n\n"
        f"{passed_lines}\n\n"
        "Any pass here is diagnostic-only: v97 still does not run old source-gate/source-replace/alpha actions, "
        "and cache-stability carriers do not by themselves solve Track K/C2/F2 or authorize runtime promotion.\n\n"
        f"Per-head carrier search parsed {per_head_payload_file_count} raw trace payloads "
        f"({topk_identity_payload_file_count} with top-k identity) and produced {len(per_head_carrier_rows)} rows; "
        f"{len(per_head_passed)} per-head rows passed the same diagnostic gate. "
        + (
            "Top-k identity instability is now parsed from the v97 diagnostic probe when present."
            if topk_identity_payload_file_count > 0
            else "The existing trace does not store cache top-k identity indices over time, so top-k identity instability remains a missing-artifact follow-up rather than a parsed result."
        )
    ))
    write_text(out / "what_would_have_to_be_true_to_pass.md", (
        "# What Would Have To Be True\n\n"
        "A SWA carrier must reach BA >= 0.70, abs L3 correlation >= 0.30, same-count and sequence margins >= 0.05, "
        "and valid stable/anchor group coverage in at least half of cases before any new action design. "
        "A future action would additionally need a causal, non-GT cache-stability or top-k-identity mechanism, controls against old route-mass leakage, "
        "and Track K/C2 compatibility before Stage7."
    ))
    return summary


def build_d3_stage7_end_region(root: Path, sources: dict[str, Any]) -> dict[str, Any]:
    out = root / "trackD3_stage7_end_region_compensator_diagnostic"
    end_rows: list[dict[str, Any]] = []
    compensator_rows: list[dict[str, Any]] = []
    selector_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []

    for stage7_row in sources["stage7_rows"]:
        source_json = Path(str(stage7_row.get("source_json", "")))
        diagnostics_dir = source_json.parent / "diagnostics"
        per_frame_path = diagnostics_dir / "per_frame_errors.csv"
        baseline = str(stage7_row.get("baseline", "READ0_NATIVE"))
        candidate = str(stage7_row.get("candidate", ""))
        if not per_frame_path.exists():
            missing_rows.append({
                "candidate": candidate,
                "missing_reason": f"per_frame_errors.csv missing:{per_frame_path}",
            })
            continue
        gate_chunk_paths = sorted(source_json.parent.glob("gate_chunk_deltas*.csv"))
        if gate_chunk_paths:
            gate_rows = read_rows(gate_chunk_paths[0])
            active_gate_rows = [row for row in gate_rows if b(row.get("gate_active"))]
            inactive_gate_rows = [row for row in gate_rows if not b(row.get("gate_active"))]
            active_chunks = [int(f(row.get("chunk_idx"), -1)) for row in active_gate_rows if math.isfinite(f(row.get("chunk_idx")))]
            all_chunks = [int(f(row.get("chunk_idx"), -1)) for row in gate_rows if math.isfinite(f(row.get("chunk_idx")))]
            gate_effective = ";".join(sorted({row.get("gate_effective", "") for row in gate_rows if row.get("gate_effective")}))
            max_chunk = max(all_chunks) if all_chunks else -1
            selector_rows.append({
                "candidate": candidate,
                "source_json": str(source_json),
                "gate_chunk_deltas_csv": str(gate_chunk_paths[0]),
                "active_chunk_count": len(active_chunks),
                "active_chunks": ";".join(str(chunk) for chunk in active_chunks),
                "active_min_chunk": min(active_chunks) if active_chunks else "",
                "active_max_chunk": max(active_chunks) if active_chunks else "",
                "max_chunk": max_chunk if max_chunk >= 0 else "",
                "active_tail_only": bool(active_chunks) and min(active_chunks) >= int(math.floor(0.75 * max_chunk)) if max_chunk >= 0 else "",
                "chunk_id_rule_detected": ("CHUNK" in candidate and "_ONLY" in candidate) or ("chunk_eq_" in gate_effective),
                "chunk_ge_rule_detected": "chunk_ge_" in gate_effective,
                "gate_effective_values": gate_effective,
                "active_delta_candidate_minus_baseline_mean": mean([
                    f(row.get("delta_candidate_minus_baseline_m")) for row in active_gate_rows
                ]),
                "inactive_delta_candidate_minus_baseline_mean": mean([
                    f(row.get("delta_candidate_minus_baseline_m")) for row in inactive_gate_rows
                ]),
                "active_delta_end_error_mean": mean([
                    f(row.get("delta_end_error_m")) for row in active_gate_rows
                ]),
                "inactive_delta_end_error_mean": mean([
                    f(row.get("delta_end_error_m")) for row in inactive_gate_rows
                ]),
                "selector_dependency_flag": len(active_chunks) <= 2
                or ("CHUNK" in candidate and "_ONLY" in candidate)
                or ("chunk_eq_" in gate_effective),
                "diagnostic_only": True,
            })
        else:
            selector_rows.append({
                "candidate": candidate,
                "source_json": str(source_json),
                "available": False,
                "missing_reason": f"gate_chunk_deltas csv missing under {source_json.parent}",
            })
        per_rows = read_rows(per_frame_path)
        by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in per_rows:
            if row.get("run"):
                by_run[row["run"]].append(row)
        base_rows = sorted(by_run.get(baseline, []), key=lambda row: int(f(row.get("frame"), -1)))
        cand_rows = sorted(by_run.get(candidate, []), key=lambda row: int(f(row.get("frame"), -1)))
        if not base_rows or not cand_rows:
            missing_rows.append({
                "candidate": candidate,
                "missing_reason": f"baseline_or_candidate_rows_missing:{baseline}:{candidate}:{per_frame_path}",
            })
            continue
        base_by_frame = {int(f(row.get("frame"), -1)): row for row in base_rows}
        cand_by_frame = {int(f(row.get("frame"), -1)): row for row in cand_rows}
        frames = sorted(set(base_by_frame) & set(cand_by_frame))
        if not frames:
            missing_rows.append({
                "candidate": candidate,
                "missing_reason": f"no overlapping frames:{baseline}:{candidate}:{per_frame_path}",
            })
            continue

        def row_error(row: dict[str, str]) -> float:
            return f(row.get("aligned_error_m"))

        def row_vec(row: dict[str, str]) -> tuple[float, float, float]:
            return (
                f(row.get("aligned_error_x_m")),
                f(row.get("aligned_error_y_m")),
                f(row.get("aligned_error_z_m")),
            )

        delta_by_frame = {
            frame: row_error(cand_by_frame[frame]) - row_error(base_by_frame[frame])
            for frame in frames
            if math.isfinite(row_error(cand_by_frame[frame])) and math.isfinite(row_error(base_by_frame[frame]))
        }
        final_frame = max(frames)
        tail50 = frames[-min(50, len(frames)):]
        tail100 = frames[-min(100, len(frames)):]
        tail10pct = frames[-max(1, int(math.ceil(0.10 * len(frames)))):]
        first_half = frames[: max(1, len(frames) // 2)]
        second_half = frames[max(1, len(frames) // 2):]

        def mean_delta(selected_frames: list[int]) -> float:
            return mean([delta_by_frame.get(frame, math.nan) for frame in selected_frames])

        end_row = {
            "candidate": candidate,
            "source_json": str(source_json),
            "baseline": baseline,
            "strict_full_gate_pass": stage7_row.get("strict_full_gate_pass", ""),
            "strict_full_gate_reason": stage7_row.get("strict_full_gate_reason", ""),
            "delta_aligned_ate_rmse_m": stage7_row.get("delta_aligned_ate_rmse_m", ""),
            "delta_final_error_m": stage7_row.get("delta_final_error_m", ""),
            "delta_yaw_rmse_deg": stage7_row.get("delta_yaw_rmse_deg", ""),
            "rolling_worse_fraction_max": stage7_row.get("rolling_worse_fraction_max", ""),
            "active_count": stage7_row.get("active_count", ""),
            "active_mean_delta_m": stage7_row.get("active_mean_delta_m", ""),
            "inactive_mean_delta_m": stage7_row.get("inactive_mean_delta_m", ""),
            "overlap_frame_count": len(frames),
            "final_frame": final_frame,
            "final_frame_delta_aligned_error_m": delta_by_frame.get(final_frame, ""),
            "tail50_mean_delta_aligned_error_m": mean_delta(tail50),
            "tail100_mean_delta_aligned_error_m": mean_delta(tail100),
            "tail10pct_mean_delta_aligned_error_m": mean_delta(tail10pct),
            "first_half_mean_delta_aligned_error_m": mean_delta(first_half),
            "second_half_mean_delta_aligned_error_m": mean_delta(second_half),
            "end_region_worse_than_first_half": mean_delta(tail100) > mean_delta(first_half)
            if math.isfinite(mean_delta(tail100)) and math.isfinite(mean_delta(first_half))
            else "",
            "chunk_selector_like": int(f(stage7_row.get("active_count"), 0.0)) <= 2,
            "diagnostic_only": True,
        }
        end_rows.append(end_row)

        for window_name, selected_frames in [
            ("tail50", tail50),
            ("tail100", tail100),
            ("tail10pct", tail10pct),
        ]:
            delta_vecs = []
            for frame in selected_frames:
                bvec = row_vec(base_by_frame[frame])
                cvec = row_vec(cand_by_frame[frame])
                if all(math.isfinite(value) for value in bvec + cvec):
                    delta_vecs.append(tuple(c - b for c, b in zip(cvec, bvec)))
            if not delta_vecs:
                compensator_rows.append({
                    "candidate": candidate,
                    "oracle_window": window_name,
                    "available": False,
                    "missing_reason": "no finite aligned error vectors in selected tail window",
                })
                continue
            comp = tuple(mean([vec[idx] for vec in delta_vecs]) for idx in range(3))
            base_norms = []
            candidate_norms = []
            compensated_norms = []
            tail_compensated_norms = []
            tail_base_norms = []
            for frame in frames:
                bvec = row_vec(base_by_frame[frame])
                cvec = row_vec(cand_by_frame[frame])
                if not all(math.isfinite(value) for value in bvec + cvec):
                    continue
                base_norm = math.sqrt(sum(value * value for value in bvec))
                cand_norm = math.sqrt(sum(value * value for value in cvec))
                corrected = tuple(c - shift for c, shift in zip(cvec, comp))
                corrected_norm = math.sqrt(sum(value * value for value in corrected))
                base_norms.append(base_norm)
                candidate_norms.append(cand_norm)
                compensated_norms.append(corrected_norm)
                if frame in selected_frames:
                    tail_base_norms.append(base_norm)
                    tail_compensated_norms.append(corrected_norm)
            final_base_vec = row_vec(base_by_frame[final_frame])
            final_cand_vec = row_vec(cand_by_frame[final_frame])
            final_corrected_vec = tuple(c - shift for c, shift in zip(final_cand_vec, comp))
            final_base_error = math.sqrt(sum(value * value for value in final_base_vec))
            final_candidate_error = math.sqrt(sum(value * value for value in final_cand_vec))
            final_corrected_error = math.sqrt(sum(value * value for value in final_corrected_vec))
            compensated_ate = rmse(compensated_norms)
            base_ate = rmse(base_norms)
            candidate_ate = rmse(candidate_norms)
            tail_compensated = mean(tail_compensated_norms)
            tail_base = mean(tail_base_norms)
            compensator_rows.append({
                "candidate": candidate,
                "oracle_window": window_name,
                "available": True,
                "compensator_dx_m": comp[0],
                "compensator_dy_m": comp[1],
                "compensator_dz_m": comp[2],
                "baseline_ate_rmse_m": base_ate,
                "candidate_ate_rmse_m": candidate_ate,
                "oracle_compensated_ate_rmse_m": compensated_ate,
                "oracle_delta_ate_vs_baseline_m": compensated_ate - base_ate
                if math.isfinite(compensated_ate) and math.isfinite(base_ate)
                else "",
                "baseline_final_error_m": final_base_error,
                "candidate_final_error_m": final_candidate_error,
                "oracle_compensated_final_error_m": final_corrected_error,
                "oracle_delta_final_error_vs_baseline_m": final_corrected_error - final_base_error,
                "tail_window_baseline_mean_error_m": tail_base if math.isfinite(tail_base) else "",
                "tail_window_oracle_compensated_mean_error_m": tail_compensated
                if math.isfinite(tail_compensated)
                else "",
                "oracle_uses_gt_aligned_error_vectors": True,
                "runtime_action_allowed": False,
                "interpretation": "posthoc GT-vector offset diagnostic only; cannot be promoted as runtime compensator",
            })

    write_rows(out / "end_region_contribution_rows.csv", end_rows or [{
        "available": False,
        "missing_reason": "no Stage7 end-region rows could be built",
    }])
    write_rows(out / "oracle_compensator_rows.csv", compensator_rows or [{
        "available": False,
        "missing_reason": "no oracle compensator rows could be built",
    }])
    write_rows(out / "chunk_selector_dependence_rows.csv", selector_rows or [{
        "available": False,
        "missing_reason": "no chunk selector rows could be built",
    }])
    write_rows(out / "missing_rows.csv", missing_rows)
    final_worse_rows = [row for row in end_rows if f(row.get("delta_final_error_m")) > 0]
    chunk_selector_rows = [row for row in end_rows if b(row.get("chunk_selector_like"))]
    tail100_worse_rows = [
        row for row in end_rows
        if math.isfinite(f(row.get("tail100_mean_delta_aligned_error_m"))) and f(row.get("tail100_mean_delta_aligned_error_m")) > 0
    ]
    oracle_final_help_rows = [
        row for row in compensator_rows
        if b(row.get("available")) and math.isfinite(f(row.get("oracle_delta_final_error_vs_baseline_m"))) and f(row.get("oracle_delta_final_error_vs_baseline_m")) <= 0
    ]
    oracle_full_pass_rows = [
        row for row in oracle_final_help_rows
        if math.isfinite(f(row.get("oracle_delta_ate_vs_baseline_m"))) and f(row.get("oracle_delta_ate_vs_baseline_m")) <= -0.3
    ]
    selector_dependency_rows = [row for row in selector_rows if b(row.get("selector_dependency_flag"))]
    chunk_id_rule_rows = [row for row in selector_rows if b(row.get("chunk_id_rule_detected"))]
    active_tail_only_rows = [row for row in selector_rows if b(row.get("active_tail_only"))]
    summary = {
        "schema": "acl2_v97_trackD3_stage7_end_region_compensator_diagnostic_v1",
        "status": "complete_diagnostic_no_action" if end_rows else "missing",
        "stage7_candidate_rows": len(sources["stage7_rows"]),
        "end_region_rows": len(end_rows),
        "missing_rows": len(missing_rows),
        "final_error_worse_count": len(final_worse_rows),
        "chunk_selector_like_count": len(chunk_selector_rows),
        "tail100_mean_worse_count": len(tail100_worse_rows),
        "oracle_compensator_rows": len(compensator_rows),
        "oracle_compensator_final_no_worse_count": len(oracle_final_help_rows),
        "oracle_compensator_strict_ate_pass_count": len(oracle_full_pass_rows),
        "chunk_selector_dependence_rows": len(selector_rows),
        "selector_dependency_flag_count": len(selector_dependency_rows),
        "chunk_id_rule_detected_count": len(chunk_id_rule_rows),
        "active_tail_only_count": len(active_tail_only_rows),
        "gate_pass": False,
        "runtime_action_allowed": False,
        "classification": "STAGE7_END_REGION_GAUGE_HARM_DIAGNOSTIC_ONLY_NO_ACTION"
        if end_rows
        else "STAGE7_END_REGION_DIAGNOSTIC_MISSING",
        "interpretation": (
            "End-region and oracle offset diagnostics are evidence only. Oracle rows use GT aligned-error vectors, "
            "so they can identify offset-like failure modes but cannot authorize a runtime compensator or Stage7 promotion."
        ),
    }
    write_json(out / "summary.json", summary)
    write_text(out / "failure_report.md", (
        "# Track D3 Stage7 End-Region Diagnostic\n\n"
        f"Built end-region rows for {len(end_rows)} Stage7 candidates and oracle compensator rows for {len(compensator_rows)} candidate/window pairs. "
        f"Final-error worse candidates: {len(final_worse_rows)}; chunk-selector-like candidates: {len(chunk_selector_rows)}; "
        f"tail100 mean worse rows: {len(tail100_worse_rows)}; selector dependency flags: {len(selector_dependency_rows)}; "
        f"chunk-id rules detected: {len(chunk_id_rule_rows)}. "
        "The oracle compensator uses GT aligned-error vectors and is diagnostic-only, so it does not permit runtime action."
    ))
    write_text(out / "what_would_have_to_be_true_to_pass.md", (
        "# What Would Have To Be True\n\n"
        "A real compensator would need a non-GT predictor available before runtime action, sequence-balanced evidence, no chunk-id selector dependence, "
        "and then Track K/C2 mechanism gates plus Stage7 full validation. The current oracle offset rows cannot satisfy those requirements."
    ))
    return summary


def build_final(root: Path, summaries: dict[str, dict[str, Any]], sources: dict[str, Any]) -> dict[str, Any]:
    out = root / "final_decision"
    stage0 = summaries["observatory"].get("gate_pass", False)
    trackk = summaries["trackk"]
    h2 = summaries["h2"]
    c2 = summaries["c2"]
    f2 = summaries["f2"]
    e2 = summaries["e2"]
    d3 = summaries.get("d3_stage7_end_region", {})
    action_allowed = bool(trackk.get("any_eligibility_cue_gate_pass") and h2.get("gate_pass") and c2.get("gate_pass"))
    full_validation_allowed = False
    if action_allowed:
        taxonomy = "SEMANTIC_SCALE_CUE_PASS_ACTION_BLOCKED"
    elif h2.get("local_L2_mechanism_exists"):
        taxonomy = "MECHANISM_PASS_FULL_NO_GO"
    elif trackk.get("any_eligibility_cue_gate_pass"):
        taxonomy = "SEMANTIC_SCALE_CUE_PASS_ACTION_BLOCKED"
    else:
        taxonomy = "DIAGNOSTIC_ONLY"
    answers = {
        "semantic_guided_scale_evidence_cue_found": bool(trackk.get("any_eligibility_cue_gate_pass")),
        "semantic_specific_scale_control_action_found": bool(h2.get("semantic_specific_component_gate_pass")),
        "read_local_pass_full_fail_gauge_safety_solved": False,
        "stable_semantic_anchors_latent_ruler_proven": bool(c2.get("gate_pass")),
        "ttt_missing_good_write_or_retention_found": bool(f2.get("gate_pass")),
        "new_swa_carrier_found": bool(e2.get("gate_pass")),
        "runtime_action_allowed": action_allowed,
        "stage7_full_validation_run_in_v97": False,
    }
    blocker_gates = []
    if not trackk.get("any_eligibility_cue_gate_pass"):
        blocker_gates.append("TrackK semantic scale evidence eligibility")
    if not h2.get("gate_pass"):
        blocker_gates.append("TrackH2 semantic-specific L07 component")
    if not c2.get("gate_pass"):
        blocker_gates.append("TrackC2 stable-anchor latent ruler")
    if not f2.get("gate_pass"):
        blocker_gates.append("TrackF2 exact stable-anchor retention")
    primary_blocker = (
        "No v97 runtime action is allowed because prerequisite gates did not all pass: "
        + "; ".join(blocker_gates)
        + ". "
        "v96 confidence-neutral L07 remains a local mechanism with Stage7 full No-Go: "
        f"best candidate {sources['stage7'].get('best_candidate_by_delta_ate')} had delta ATE "
        f"{sources['stage7'].get('best_delta_aligned_ate_rmse_m')} and delta final error "
        f"{sources['stage7'].get('best_delta_final_error_m')}."
    )
    c2_replay_note = (
        "Audited Stage7 replay dry-run manifests now exist for saved full rollouts, but no full replay was executed and no full latent dump is available because the Stage7 full-sequence gate remains blocked."
        if c2.get("stage7_full_latent_replay_command_available_any")
        else "No audited Stage7 full latent replay command is available yet."
    )
    trackk_swa_cache = trackk.get("swa_cache_metrics") or {}
    if trackk_swa_cache and trackk_swa_cache.get("best_cache_carrier_eligibility_gate_pass"):
        trackk_swa_note = (
            "Track K SWA cache/top-k eligibility now has an eligible cue row: "
            f"strict stable coverage is {trackk_swa_cache.get('strict_stable_nonempty_case_frac')} and fallback usage is {trackk_swa_cache.get('fallback_used_case_frac')}. "
            "This does not authorize runtime action without the H2/C2/F2 gates."
        )
    elif trackk_swa_cache:
        trackk_swa_note = (
            "Track K SWA cache/top-k eligibility was audited separately: "
            f"strict stable coverage is {trackk_swa_cache.get('strict_stable_nonempty_case_frac')} and fallback usage is {trackk_swa_cache.get('fallback_used_case_frac')}, "
            f"but the best carrier eligibility gate is {trackk_swa_cache.get('best_cache_carrier_eligibility_gate_pass')}; "
            "it remains diagnostic-only rather than an action cue."
        )
    else:
        trackk_swa_note = ""
    e2_passed = e2.get("passed_carrier_details") or []
    if e2_passed:
        e2_brief = "; ".join(
            f"{row.get('carrier_metric')} BA={row.get('best_balanced_accuracy')} abs_L3_corr={row.get('abs_correlation_with_L3_handoff_transfer_penalty')}"
            for row in e2_passed
        )
        if e2.get("topk_identity_available"):
            topk_note = (
                f"top-k identity parsed from {e2.get('topk_identity_payload_file_count', 0)} v97 diagnostic payloads."
            )
        else:
            topk_note = (
                "top-k identity is not available in existing v96 trace "
                f"({e2.get('topk_identity_missing_reason', '')})."
            )
        per_head_note = (
            f" Per-head raw-trace search found {e2.get('per_head_carrier_gate_pass_count', 0)} diagnostic gate-pass rows; "
            f"{topk_note}"
            if e2.get("per_head_carrier_row_count")
            else ""
        )
        e2_next = (
            f"Track E2 did find diagnostic SWA cache-stability carriers ({e2_brief}); this is a new carrier signal, "
            "but it is not the old route-mass action and remains action-ineligible until a causal non-GT cache-stability/top-k-identity mechanism and controls are designed."
            f"{per_head_note}"
        )
    else:
        e2_next = "Track E2 did not find a new SWA carrier that can drive action design."
    if h2.get("gate_pass"):
        f2_next = (
            "F2 exact retention maps plus exact write-prior mass/uniformity and residual/long-drift audits are now available but did not pass; projected write-energy correlations remain diagnostic-only, so next useful TTT work needs a genuinely new downstream-tied retention actuator or broader instrumentation, not proxy promotion."
            if f2.get("stable_anchor_retention_available")
            else "next useful work includes exact TTT retention maps."
        )
        next_route = (
            "Do not run old READ beta/cap, weak-context skip, Track E source gate/source replace/alpha sweeps, or TTT no-write action. "
            "H2 confidence-control parsing now supports a semantic-specific local component claim, but action remains blocked: Track K proxy and LOSO audits remain sequence-fragile/no-correct-L2-direction, and Track C2 latent ruler evidence still fails; "
            "C2 local downstream correlation has been audited and is still control-comparable, so next useful READ work is broader/full-sequence before/after stable-anchor latent dumps tied to final-error/global-yaw rather than another local smoke reuse. "
            f"{c2_replay_note} {f2_next} {trackk_swa_note} {e2_next} Stage7 end-region/oracle-offset diagnostics have been computed but are diagnostic-only because they use GT aligned-error vectors and do not provide a runtime predictor."
        )
    else:
        f2_next = (
            "F2 exact retention maps are available but gate-failed."
            if f2.get("stable_anchor_retention_available")
            else "TTT requires exact stable-anchor retention maps before action simulation."
        )
        next_route = (
            "Do not run old READ beta/cap, weak-context skip, Track E source gate/source replace/alpha sweeps, or TTT no-write action. "
            "Next useful work is exact H2 confidence-shuffle/component pilots plus READ before/after stable-anchor latent dumps; "
            f"{c2_replay_note} {f2_next}"
        )
    decision = {
        "schema": "acl2_v97_final_decision_v1",
        "status": "complete",
        "final_taxonomy": taxonomy,
        "method_success": False,
        "mechanism_success": bool(h2.get("local_L2_mechanism_exists")),
        "full_method_success": False,
        "runtime_action_allowed": action_allowed,
        "stage7_full_validation_allowed": full_validation_allowed,
        "stage7_full_latent_replay_manifest_total_count": c2.get("stage7_full_latent_replay_manifest_total_count", 0),
        "stage7_full_latent_replay_matching_rollout_count": c2.get("stage7_full_latent_replay_manifest_matching_rollout_count", 0),
        "stage7_full_latent_replay_launched_count": c2.get("stage7_full_latent_replay_launched_count", 0),
        "stage7_full_latent_dump_available_count": c2.get("stage7_full_latent_dump_available_count", 0),
        "stage0_observatory_gate_pass": stage0,
        "trackK_any_eligibility_gate_pass": trackk.get("any_eligibility_cue_gate_pass", False),
        "trackH2_gate_pass": h2.get("gate_pass", False),
        "trackC2_gate_pass": c2.get("gate_pass", False),
        "trackF2_gate_pass": f2.get("gate_pass", False),
        "trackE2_gate_pass": e2.get("gate_pass", False),
        "trackD3_stage7_end_region_gate_pass": d3.get("gate_pass", False),
        "trackD3_stage7_end_region_classification": d3.get("classification", ""),
        "answers": answers,
        "primary_blocker": primary_blocker,
        "next_route_recommendation": next_route,
    }
    write_json(out / "final_decision.json", decision)
    write_json(out / "summary.json", decision)
    write_rows(out / "gate_checks.csv", [
        {"gate": "Stage0_observatory", "pass": stage0},
        {"gate": "TrackK_any_eligibility", "pass": trackk.get("any_eligibility_cue_gate_pass", False)},
        {"gate": "TrackH2_semantic_specific_component", "pass": h2.get("gate_pass", False)},
        {"gate": "TrackC2_latent_ruler", "pass": c2.get("gate_pass", False)},
        {"gate": "TrackF2_missing_good_write", "pass": f2.get("gate_pass", False)},
        {"gate": "TrackE2_new_swa_carrier", "pass": e2.get("gate_pass", False)},
        {"gate": "TrackD3_stage7_end_region_diagnostic_only", "pass": d3.get("gate_pass", False)},
        {"gate": "Runtime_action_allowed", "pass": action_allowed},
    ])
    write_text(out / "final_report.md", f"""# ACL2 v97-TF Final Report

Final taxonomy: `{taxonomy}`.

This is not a full method success. v97 did not run or promote a runtime action. Stage7 full validation was not rerun because the v97 prerequisite gates did not all pass.

Answers:

1. semantic-guided scale evidence cue found: `{answers['semantic_guided_scale_evidence_cue_found']}`
2. semantic-specific scale-control action found: `{answers['semantic_specific_scale_control_action_found']}`
3. READ local-pass/full-fail gauge safety solved: `False`
4. stable semantic anchors latent ruler proven: `{answers['stable_semantic_anchors_latent_ruler_proven']}`
5. TTT missing-good-write / retention found: `{answers['ttt_missing_good_write_or_retention_found']}`
6. new SWA carrier found: `{answers['new_swa_carrier_found']}`
7. runtime action allowed: `{action_allowed}`

Primary blocker:

{decision['primary_blocker']}
""")
    write_text(out / "failure_report.md", decision["primary_blocker"])
    write_text(out / "next_route_recommendation.md", decision["next_route_recommendation"])
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--v96-root", type=Path, default=V96_ROOT)
    args = parser.parse_args()
    sources = load_sources(args.v96_root)
    args.root.mkdir(parents=True, exist_ok=True)
    summaries = {
        "observatory": build_observatory(args.root, sources),
        "f2": build_f2(args.root, sources),
        "trackk": build_track_k(args.root, sources),
        "h2": build_h2(args.root, sources),
        "c2": build_c2(args.root, sources),
        "e2": build_e2(args.root, sources),
        "d3_stage7_end_region": build_d3_stage7_end_region(args.root, sources),
    }
    decision = build_final(args.root, summaries, sources)
    write_json(args.root / "build_summary.json", {
        "schema": "acl2_v97_build_summary_v1",
        "root": str(args.root),
        "v96_root": str(args.v96_root),
        "final_taxonomy": decision["final_taxonomy"],
        "runtime_action_allowed": decision["runtime_action_allowed"],
        "stage7_full_validation_allowed": decision["stage7_full_validation_allowed"],
    })
    print(json.dumps({
        "root": str(args.root),
        "final_taxonomy": decision["final_taxonomy"],
        "runtime_action_allowed": decision["runtime_action_allowed"],
        "stage7_full_validation_allowed": decision["stage7_full_validation_allowed"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
