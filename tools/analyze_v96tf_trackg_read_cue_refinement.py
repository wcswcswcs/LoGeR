#!/usr/bin/env python3
"""Analyze v96 Track G/J READ cue refinement from existing artifacts.

This is a diagnostic-only script.  It does not run a LoGeR action and does not
promote a runtime method.  The goal is to test whether the READ weak-context
semantic signal can satisfy the plan's cue-v2 classifier requirements, then
cross-check that against the already-run J4 action outcomes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v96tf_vggt4d_informed_semantic_gauge_preserving_memory_control")
OUT_DIR = ROOT / "trackG_read_cue_refinement"
J4_DIRS = [
    "trackJ_read_skip_pilot",
    "trackJ_read_skip_pilot_repair_early_quarter",
    "trackJ_read_skip_pilot_repair_anchor_compensation",
    "trackJ_read_skip_pilot_repair_anchor_weak_compensation",
    "trackJ_read_skip_pilot_repair_anchor_weak_rho020",
]
EPS = 1.0e-6


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"read_error": f"{type(exc).__name__}:{exc}", "path": str(path)}
    return payload if isinstance(payload, dict) else {"value": payload}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
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
            writer.writerow(row)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() in {"", "nan", "None", "null"}:
            return default
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x <= 0 or den_y <= 0:
        return 0.0
    return num / (den_x * den_y)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            out[order[k]] = rank
        i = j
    return out


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    return pearson(ranks(xs), ranks(ys))


def stable_hash_rank(seed: str, items: list[str]) -> list[str]:
    return sorted(items, key=lambda item: hashlib.sha256(f"{seed}:{item}".encode("utf-8")).hexdigest())


def format_cases(cases: set[str] | list[str]) -> str:
    return ";".join(sorted(cases))


def make_features(per_case_row: dict[str, str]) -> dict[str, float]:
    weak = safe_float(per_case_row.get("WEAK_SCALE_CONTEXT_token_mass"))
    veg = safe_float(per_case_row.get("VEGETATION_REPETITIVE_token_mass"))
    low = safe_float(per_case_row.get("LOW_OBSERVABILITY_token_mass"))
    dynamic = safe_float(per_case_row.get("DYNAMIC_OBJECT_token_mass"))
    boundary = safe_float(per_case_row.get("OBJECT_BOUNDARY_BAND_token_mass"))
    multimode = safe_float(per_case_row.get("MULTIMODE_CONFLICT_token_mass"))
    stable = safe_float(per_case_row.get("STABLE_ANCHOR_token_mass"))
    lowstuff = weak + veg + low
    return {
        "weak_scale_context_mass": weak,
        "vegetation_repetitive_mass": veg,
        "low_observability_mass": low,
        "dynamic_object_mass": dynamic,
        "object_boundary_mass": boundary,
        "multimode_conflict_mass": multimode,
        "stable_anchor_mass": stable,
        "weak_plus_low_mass": weak + low,
        "weak_plus_vegetation_mass": weak + veg,
        "lowstuff_sum_mass": lowstuff,
        "weak_over_stable": weak / (stable + EPS),
        "lowstuff_over_stable": lowstuff / (stable + EPS),
        "weak_minus_stable": weak - stable,
        "lowstuff_minus_stable": lowstuff - stable,
        "weak_times_inverse_stable": weak * (1.0 - stable),
        "lowstuff_times_inverse_stable": lowstuff * (1.0 - stable),
    }


def score_selection(selected: set[str], positives: set[str], bad_only: set[str], goods: set[str], atlas: dict[str, dict[str, str]]) -> dict[str, float]:
    positive_recall = len(selected & positives) / len(positives) if positives else 0.0
    bad_only_recall = len(selected & bad_only) / len(bad_only) if bad_only else 0.0
    good_fpr = len(selected & goods) / len(goods) if goods else 0.0
    positive_seq_coverage = len({atlas[case_id].get("seq", "") for case_id in selected & positives})
    bad_only_seq_coverage = len({atlas[case_id].get("seq", "") for case_id in selected & bad_only})
    return {
        "bad_recall": positive_recall,
        "bad_only_recall": bad_only_recall,
        "good_FPR": good_fpr,
        "positive_sequence_coverage": float(positive_seq_coverage),
        "bad_only_sequence_coverage": float(bad_only_seq_coverage),
        "cue_signal": positive_recall - good_fpr,
    }


def same_count_controls(
    selected: set[str],
    all_cases: list[str],
    positives: set[str],
    bad_only: set[str],
    goods: set[str],
    atlas: dict[str, dict[str, str]],
    per_seq: bool,
    seeds: int = 64,
) -> list[float]:
    scores: list[float] = []
    seq_to_cases: dict[str, list[str]] = defaultdict(list)
    for case_id in all_cases:
        seq_to_cases[atlas[case_id].get("seq", "")].append(case_id)
    selected_seq_counts: dict[str, int] = defaultdict(int)
    for case_id in selected:
        selected_seq_counts[atlas[case_id].get("seq", "")] += 1

    for seed_idx in range(seeds):
        if not per_seq:
            control = set(stable_hash_rank(f"global_same_count_{seed_idx}", all_cases)[: len(selected)])
        else:
            control: set[str] = set()
            for seq, seq_cases in sorted(seq_to_cases.items()):
                count = selected_seq_counts.get(seq, 0)
                control.update(stable_hash_rank(f"seq_same_count_{seed_idx}_{seq}", seq_cases)[:count])
        scores.append(score_selection(control, positives, bad_only, goods, atlas)["cue_signal"])
    return scores


def semantic_rotation_scores(
    feature_values: dict[str, float],
    threshold: float,
    all_cases: list[str],
    positives: set[str],
    bad_only: set[str],
    goods: set[str],
    atlas: dict[str, dict[str, str]],
) -> list[float]:
    values = [feature_values[case_id] for case_id in all_cases]
    scores: list[float] = []
    for shift in range(1, len(all_cases)):
        rotated = {case_id: values[(idx - shift) % len(values)] for idx, case_id in enumerate(all_cases)}
        selected = {case_id for case_id in all_cases if rotated[case_id] >= threshold}
        scores.append(score_selection(selected, positives, bad_only, goods, atlas)["cue_signal"])
    return scores


def feature_metric_summary(
    features_by_case: dict[str, dict[str, float]],
    atlas: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    metrics = {
        "L1_local_sim3_ate": "L1_local_sim3_ate",
        "L2_head_tail_proxy_error": "L2_head_tail_proxy_error",
        "L1_local_sim3_scale": "L1_local_sim3_scale",
        "L3_gauge_jump_proxy": "L3_gauge_jump_proxy",
    }
    rows: list[dict[str, Any]] = []
    feature_names = sorted(next(iter(features_by_case.values())).keys()) if features_by_case else []
    for feature_name in feature_names:
        for metric_name, metric_field in metrics.items():
            xs: list[float] = []
            ys: list[float] = []
            for case_id, feature_map in features_by_case.items():
                metric = safe_float(atlas[case_id].get(metric_field), default=float("nan"))
                if math.isfinite(metric):
                    xs.append(feature_map[feature_name])
                    ys.append(metric)
            rows.append(
                {
                    "feature_name": feature_name,
                    "metric": metric_name,
                    "case_count": len(xs),
                    "pearson": pearson(xs, ys),
                    "spearman": spearman(xs, ys),
                }
            )
    return rows


def summarize_j4_action_gap(root: Path) -> dict[str, Any]:
    summaries: dict[str, dict[str, Any]] = {}
    best_improvement = -1.0
    best_improvement_track = ""
    best_improvement_metric = ""
    best_semantic_margin = -1.0
    best_semantic_margin_track = ""
    completed_count = 0
    gate_pass_tracks: list[str] = []
    for track in J4_DIRS:
        summary = read_json(root / track / "summary.json")
        summaries[track] = summary
        if summary.get("status") == "complete":
            completed_count += 1
        if boolish(summary.get("gate_pass")):
            gate_pass_tracks.append(track)
        metric_decisions = summary.get("metric_decisions", {})
        if isinstance(metric_decisions, dict):
            for metric_name, decision in metric_decisions.items():
                if not isinstance(decision, dict):
                    continue
                improvement = safe_float(decision.get("bad_improvement_vs_baseline"), default=float("nan"))
                semantic_margin = safe_float(decision.get("candidate_margin_vs_semantic_rotation"), default=float("nan"))
                if math.isfinite(improvement) and improvement > best_improvement:
                    best_improvement = improvement
                    best_improvement_track = track
                    best_improvement_metric = metric_name
                if math.isfinite(semantic_margin) and semantic_margin > best_semantic_margin:
                    best_semantic_margin = semantic_margin
                    best_semantic_margin_track = track
    return {
        "completed_track_count": completed_count,
        "gate_pass_tracks": gate_pass_tracks,
        "any_gate_pass": bool(gate_pass_tracks),
        "best_bad_metric_improvement": best_improvement if best_improvement >= 0 else None,
        "best_bad_metric_improvement_track": best_improvement_track,
        "best_bad_metric_improvement_metric": best_improvement_metric,
        "best_candidate_margin_vs_semantic_rotation": best_semantic_margin if best_semantic_margin >= -0.5 else None,
        "best_candidate_margin_vs_semantic_rotation_track": best_semantic_margin_track,
        "summaries": summaries,
    }


def build_visual_manifest(
    best_row: dict[str, Any] | None,
    semantic_rows: list[dict[str, str]],
    positives: set[str],
    goods: set[str],
) -> list[dict[str, Any]]:
    if not best_row:
        return []
    selected = set(str(best_row.get("selected_cases", "")).split(";")) - {""}
    missed = set(str(best_row.get("missed_positive_cases", "")).split(";")) - {""}
    false_positives = selected & goods
    false_negatives = missed & positives
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row in semantic_rows:
        key = (row.get("case_id", ""), row.get("region_type", ""))
        indexed.setdefault(key, row)
    out: list[dict[str, Any]] = []
    for group_name, case_ids in (("false_positive_good_control", false_positives), ("false_negative_read_positive", false_negatives)):
        for case_id in sorted(case_ids):
            row = indexed.get((case_id, "WEAK_SCALE_CONTEXT")) or next(
                (item for item in semantic_rows if item.get("case_id") == case_id),
                {},
            )
            out.append(
                {
                    "visual_id": f"{group_name}_{case_id}",
                    "case_id": case_id,
                    "error_group": group_name,
                    "region_type": row.get("region_type", ""),
                    "source_path": row.get("visual_panel_path", ""),
                    "exists": str(Path(row.get("visual_panel_path", "")).exists()) if row.get("visual_panel_path") else "",
                    "note": "Track G READ cue refinement threshold audit",
                }
            )
    return out


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)

    input_paths = {
        "trackA_rows": root / "trackA_case_response_atlas/rows.csv",
        "per_case_region_mass": root / "trackJ_semantic_region_bank/per_case_region_mass.csv",
        "semantic_region_rows": root / "trackJ_semantic_region_bank/semantic_region_rows.csv",
        "j3_region_candidate_summary": root / "trackJ_skip_impact_diagnostic/region_candidate_summary.csv",
    }
    atlas_rows = read_csv(input_paths["trackA_rows"])
    per_case_rows = read_csv(input_paths["per_case_region_mass"])
    semantic_rows = read_csv(input_paths["semantic_region_rows"])
    j3_rows = read_csv(input_paths["j3_region_candidate_summary"])
    atlas = {row["case_id"]: row for row in atlas_rows}
    per_case = {row["case_id"]: row for row in per_case_rows if row.get("case_id") in atlas}
    all_cases = sorted(set(atlas) & set(per_case))
    features_by_case = {case_id: make_features(per_case[case_id]) for case_id in all_cases}

    positives = {
        case_id
        for case_id, row in atlas.items()
        if "READ_LOCAL_BAD" in row.get("action_response_labels", "") and "GOOD_PROTECTION" not in row.get("action_response_labels", "")
    }
    bad_only = {
        case_id
        for case_id, row in atlas.items()
        if "READ_LOCAL_BAD" in row.get("action_response_labels", "") and row.get("case_label_offline_only") == "bad"
    }
    goods = {case_id for case_id, row in atlas.items() if "GOOD_PROTECTION" in row.get("action_response_labels", "")}
    positives &= set(all_cases)
    bad_only &= set(all_cases)
    goods &= set(all_cases)

    threshold_rows: list[dict[str, Any]] = []
    feature_names = sorted(next(iter(features_by_case.values())).keys()) if features_by_case else []
    for feature_name in feature_names:
        feature_values = {case_id: features_by_case[case_id][feature_name] for case_id in all_cases}
        thresholds = sorted(set(feature_values.values()))
        for threshold in thresholds:
            selected = {case_id for case_id in all_cases if feature_values[case_id] >= threshold}
            base = score_selection(selected, positives, bad_only, goods, atlas)
            global_controls = same_count_controls(selected, all_cases, positives, bad_only, goods, atlas, per_seq=False)
            seq_controls = same_count_controls(selected, all_cases, positives, bad_only, goods, atlas, per_seq=True)
            rotation_controls = semantic_rotation_scores(feature_values, threshold, all_cases, positives, bad_only, goods, atlas)
            global_margin = base["cue_signal"] - median(global_controls)
            seq_margin = base["cue_signal"] - median(seq_controls)
            rotation_margin = base["cue_signal"] - median(rotation_controls)
            failure_reasons: list[str] = []
            if base["bad_recall"] < 0.70:
                failure_reasons.append("bad_recall_lt_0.70")
            if base["good_FPR"] > 0.25:
                failure_reasons.append("good_FPR_gt_0.25")
            if global_margin < 0.05:
                failure_reasons.append("global_same_count_margin_lt_0.05_proxy")
            if seq_margin < 0.05:
                failure_reasons.append("seq_count_margin_lt_0.05_proxy")
            if rotation_margin < 0.05:
                failure_reasons.append("semantic_rotation_margin_lt_0.05_proxy")
            if base["positive_sequence_coverage"] < 3:
                failure_reasons.append("positive_sequence_coverage_lt_3")
            gate_pass = not failure_reasons
            margins = [global_margin, seq_margin, rotation_margin]
            ranking_score = base["bad_recall"] - base["good_FPR"] + min(margins) + 0.05 * base["positive_sequence_coverage"]
            threshold_rows.append(
                {
                    "feature_name": feature_name,
                    "direction": "high_is_risk",
                    "threshold": threshold,
                    "selected_count": len(selected),
                    "bad_recall": base["bad_recall"],
                    "bad_only_recall": base["bad_only_recall"],
                    "good_FPR": base["good_FPR"],
                    "positive_sequence_coverage": int(base["positive_sequence_coverage"]),
                    "bad_only_sequence_coverage": int(base["bad_only_sequence_coverage"]),
                    "cue_signal": base["cue_signal"],
                    "global_same_count_margin_proxy": global_margin,
                    "seq_count_margin_proxy": seq_margin,
                    "semantic_rotation_margin_proxy": rotation_margin,
                    "read_cue_v2_proxy_gate_pass": gate_pass,
                    "failed_gate_count": len(failure_reasons),
                    "failure_reasons": ";".join(failure_reasons),
                    "ranking_score": ranking_score,
                    "selected_positive_cases": format_cases(selected & positives),
                    "selected_bad_only_cases": format_cases(selected & bad_only),
                    "selected_good_controls": format_cases(selected & goods),
                    "missed_positive_cases": format_cases(positives - selected),
                    "selected_cases": format_cases(selected),
                }
            )

    best_rows = sorted(
        threshold_rows,
        key=lambda row: (
            boolish(row["read_cue_v2_proxy_gate_pass"]),
            -int(row["failed_gate_count"]),
            safe_float(row["ranking_score"], default=-999.0),
            safe_float(row["bad_recall"]),
            -safe_float(row["good_FPR"]),
        ),
        reverse=True,
    )
    best_row = best_rows[0] if best_rows else None
    passing_rows = [row for row in threshold_rows if boolish(row.get("read_cue_v2_proxy_gate_pass"))]
    close_rows = [
        row
        for row in best_rows
        if safe_float(row.get("bad_recall")) >= 0.70
        and safe_float(row.get("good_FPR")) <= 0.25
        and int(row.get("failed_gate_count", 99)) <= 2
    ]

    metric_rows = feature_metric_summary(features_by_case, atlas)
    j4_gap = summarize_j4_action_gap(root)
    j3_read_weak_rows = [
        row
        for row in j3_rows
        if row.get("target", "").startswith("READ") and row.get("region_type") == "WEAK_SCALE_CONTEXT"
    ]
    read_weak_j3_diagnostic_useful = any(boolish(row.get("diagnostic_useful")) for row in j3_read_weak_rows)

    exact_gate_available = False
    read_cue_v2_proxy_gate_pass = bool(passing_rows)
    cue_action_gap_present = bool(j4_gap.get("completed_track_count")) and not bool(j4_gap.get("any_gate_pass"))
    saliency_not_geometry = read_weak_j3_diagnostic_useful and (not read_cue_v2_proxy_gate_pass or cue_action_gap_present)
    classification = (
        "READ_CUE_V2_PROXY_PASS_ACTION_GAP"
        if read_cue_v2_proxy_gate_pass and cue_action_gap_present
        else "READ_CUE_V2_NO_GO_SALIENCY_NOT_GEOMETRY_LIKELY"
        if saliency_not_geometry
        else "READ_CUE_V2_NO_GO"
    )

    input_hashes = {
        name: {
            "path": str(path),
            "exists": path.exists(),
            "sha256": sha256(path),
        }
        for name, path in input_paths.items()
    }
    summary = {
        "stage": "TrackG_READ_cue_refinement",
        "status": "complete",
        "method_success": False,
        "mechanism_success": False,
        "runtime_action_allowed": False,
        "read_cue_v2_exact_gate_available": exact_gate_available,
        "read_cue_v2_proxy_gate_pass": read_cue_v2_proxy_gate_pass,
        "classification": classification,
        "saliency_not_geometry_likely_under_current_evidence": saliency_not_geometry,
        "cue_action_gap_present": cue_action_gap_present,
        "threshold_rows": len(threshold_rows),
        "passing_threshold_rows": len(passing_rows),
        "close_threshold_rows": len(close_rows),
        "case_counts": {
            "all_cases": len(all_cases),
            "read_positive_not_good": len(positives),
            "read_bad_only": len(bad_only),
            "good_controls": len(goods),
        },
        "best_proxy_candidate": best_row or {},
        "j3_read_weak_context_rows": j3_read_weak_rows,
        "j4_action_gap": {
            key: value for key, value in j4_gap.items() if key != "summaries"
        },
        "input_hashes": input_hashes,
        "gate_rule": (
            "Plan READ cue v2 proxy gate: bad_recall>=0.70, good_FPR<=0.25, "
            "global_same_count_margin_proxy>=0.05, seq_count_margin_proxy>=0.05, "
            "semantic_rotation_margin_proxy>=0.05, positive_sequence_coverage>=3. "
            "Exact READ same-count/rotation tensors were not rerun here; margins are deterministic artifact-level proxies."
        ),
    }

    write_csv(out / "rows.csv", threshold_rows)
    write_csv(out / "best_read_cue_rows.csv", best_rows[:25])
    write_csv(out / "close_read_cue_rows.csv", close_rows[:25])
    write_csv(out / "feature_metric_summary.csv", metric_rows)
    write_csv(
        out / "gate_checks.csv",
        [
            {"gate": "read_cue_v2_exact_gate_available", "pass": exact_gate_available, "value": exact_gate_available},
            {"gate": "read_cue_v2_proxy_gate_pass", "pass": read_cue_v2_proxy_gate_pass, "value": read_cue_v2_proxy_gate_pass},
            {"gate": "J3_READ_WEAK_SCALE_CONTEXT_diagnostic_useful", "pass": read_weak_j3_diagnostic_useful, "value": read_weak_j3_diagnostic_useful},
            {"gate": "J4_READ_action_gap_present", "pass": cue_action_gap_present, "value": cue_action_gap_present},
            {"gate": "runtime_action_allowed", "pass": False, "value": False},
            {"gate": "mechanism_success", "pass": False, "value": classification},
        ],
    )
    write_json(out / "summary.json", summary)
    write_json(out / "j4_action_gap_summary.json", j4_gap)
    write_csv(out / "visual_manifest.csv", build_visual_manifest(best_row, semantic_rows, positives, goods))
    write_text(
        out / "cue_action_gap_report.md",
        f"""# Track G READ Cue-Action Gap Report

Status: diagnostic-only, no runtime action allowed.

J3 found READ WEAK_SCALE_CONTEXT diagnostic utility: `{read_weak_j3_diagnostic_useful}`.
J4 action tracks completed: `{j4_gap.get('completed_track_count')}`.
J4 gate-pass tracks: `{','.join(j4_gap.get('gate_pass_tracks', [])) or 'none'}`.

Best J4 bad-metric improvement:

- track: `{j4_gap.get('best_bad_metric_improvement_track')}`
- metric: `{j4_gap.get('best_bad_metric_improvement_metric')}`
- value: `{j4_gap.get('best_bad_metric_improvement')}`

Best J4 margin vs semantic rotation:

- track: `{j4_gap.get('best_candidate_margin_vs_semantic_rotation_track')}`
- value: `{j4_gap.get('best_candidate_margin_vs_semantic_rotation')}`

Interpretation: the existing weak-context semantic signal is diagnostic, but the
tested early K/logit-side action ladder did not produce a geometry mechanism.
This report does not distinguish final causality between cue insufficiency and
actuator mismatch; the paired cue-v2 threshold scan is recorded in `rows.csv`.
""",
    )
    if best_row:
        best_text = (
            f"Best proxy candidate: `{best_row['feature_name']}` >= `{best_row['threshold']}`; "
            f"bad_recall `{best_row['bad_recall']}`, good_FPR `{best_row['good_FPR']}`, "
            f"coverage `{best_row['positive_sequence_coverage']}`, failures `{best_row['failure_reasons'] or 'none'}`."
        )
    else:
        best_text = "No candidate rows were produced."
    write_text(
        out / "saliency_not_geometry_report.md",
        f"""# Track G READ Saliency-vs-Geometry Report

Classification: `{classification}`.

{best_text}

Evidence chain:

- J3 READ WEAK_SCALE_CONTEXT was useful as a metric-overlap diagnostic.
- The thresholded READ cue-v2 proxy did not produce a service-ready gate pass.
- The closest candidate still failed at least one required gate.
- J4 weak-context, early-quarter, stable-anchor, anchor_weak, and rho=0.2
  action variants all remained mechanism No-Go.

Conclusion: under current artifacts, WEAK_SCALE_CONTEXT behaves like a semantic
saliency / difficulty correlate rather than a promoted geometry-causal READ
control cue. A future repair should localize per-layer/per-head Q/K geometry
carriers or test a different actuator before Stage7.
""",
    )
    write_text(
        out / "failure_report.md",
        f"# Track G READ Cue Refinement Failure Report\n\n{best_text}\n\nNo runtime action is promoted. Classification: `{classification}`.",
    )
    write_text(
        out / "what_would_have_to_be_true_to_pass.md",
        "# What Would Have To Be True To Pass\n\nA READ cue must pass the v2 thresholds on bad recall, good-control FPR, global same-count margin, sequence-preserving count margin, semantic-rotation margin, and positive sequence coverage; then a matched action body must improve bad READ_LOCAL metrics by >=5%, beat controls by >=5%, keep good controls safe, and preserve stable-anchor evidence before Stage7.",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = analyze(parse_args())
    print(json.dumps({k: summary[k] for k in ("status", "classification", "read_cue_v2_proxy_gate_pass", "runtime_action_allowed")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
