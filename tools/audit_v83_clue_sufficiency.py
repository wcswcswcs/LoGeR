#!/usr/bin/env python3
"""Audit ACL2 v83 Phase2 clue sufficiency with fixed, training-free rankings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_MATRIX = Path(
    "results/acl2_v83tf_clue_sufficiency_vs_action_misuse/"
    "phase1_unified_clue_matrix/unified_clue_matrix.csv"
)
DEFAULT_OUT_DIR = Path(
    "results/acl2_v83tf_clue_sufficiency_vs_action_misuse/phase2_clue_sufficiency"
)

GROUP_FIELDS = {
    "G0": [
        "raw_overlap_residual",
        "overlap_scale_residual",
        "boundary_jump",
        "future_after_overlap",
        "confidence_weighted_residual",
        "prev_to_curr_scale_jump",
        "window5_joint_sim3_rmse",
        "window5_subchunk_scale_cv",
        "downstream_future_consistency",
        "low_observability_score",
        "regime_shift_score",
        "local_sim3_ate",
        "head_to_tail",
        "scale_cv",
        "intra_scale_variance",
        "J_mid",
        "J_long",
        "J_short",
    ],
    "G1": [
        "stable_mass",
        "harm_mass",
        "context_mass",
        "semantic_confidence_mean",
        "patch_purity_mean",
        "lowtrust_stuff_ratio",
        "sky_context_ratio",
        "dynamic_thing_ratio",
        "road_edge_continuity",
        "corridor_stability",
        "overlap_semantic_agreement",
    ],
    "G2": [
        "thing_moving_ratio",
        "thing_static_ratio",
        "stuff_static_ratio",
        "structure_ratio",
        "track_lifespan_proxy",
        "source_type_consistency",
    ],
    "G3": [
        "same_object_ratio",
        "cross_object_boundary_ratio",
        "object_boundary_score",
        "object_interior_score",
        "temporal_stability",
        "radio_lowtrust_ratio",
    ],
    "G4": [
        "READ_used_stable_mass",
        "READ_used_harm_mass",
        "QK_pair_compatibility",
        "query_risk_mass",
        "read_entropy",
    ],
    "G5": [
        "current_Q_alignment",
        "cache_K_alignment",
        "cache_V_alignment",
        "K_risk_delta",
        "V_protect_delta",
        "route_mass",
        "head_layer_sensitivity",
        "actual_vs_random_route_delta",
    ],
    "G6": [
        "boundary_transform_residual",
        "merge_raw_overlap_residual",
        "postmerge_pose_sensitivity",
        "reset_relative_position",
        "gauge_hold_signal",
    ],
    "G7": [
        "selected_low_support_ratio",
        "continuous_low_support_cluster_len",
        "update_conflict",
        "post_zp_delta",
        "write_mass_stable",
        "write_mass_harm",
        "write_mass_context",
    ],
}

COMBINATIONS = {
    "C0_geometry_only": ["G0"],
    "C1_semantic_only": ["G1", "G2"],
    "C2_radio_only": ["G3"],
    "C3_internal_only": ["G4", "G5"],
    "C4_geometry_semantic": ["G0", "G1", "G2"],
    "C5_geometry_radio": ["G0", "G3"],
    "C6_geometry_internal": ["G0", "G4", "G5"],
    "C7_full_combined": ["G0", "G1", "G2", "G3", "G4", "G5"],
    "C8_merge_augmented": ["G0", "G1", "G2", "G3", "G4", "G5", "G6"],
    "C9_ttt_augmented": ["G0", "G1", "G2", "G3", "G4", "G5", "G7"],
}

LOW_IS_RISK = {
    "stable_mass",
    "semantic_confidence_mean",
    "patch_purity_mean",
    "road_edge_continuity",
    "corridor_stability",
    "same_object_ratio",
    "object_interior_score",
    "READ_used_stable_mass",
    "QK_pair_compatibility",
    "current_Q_alignment",
    "cache_K_alignment",
    "cache_V_alignment",
    "V_protect_delta",
    "route_mass",
    "actual_vs_random_route_delta",
    "gauge_hold_signal",
    "write_mass_stable",
}

SEMANTIC_FIELDS = set(GROUP_FIELDS["G1"] + GROUP_FIELDS["G2"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--selection-quantile", type=float, default=0.75)
    parser.add_argument("--bootstrap-iters", type=int, default=500)
    parser.add_argument("--control-iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=8302)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize(row.get(key, "")) for key in keys})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def serialize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def safe_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = max(0.0, min(1.0, q)) * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def auc_score(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    pairs = [(int(y), float(s)) for y, s in zip(labels, scores) if math.isfinite(float(s))]
    pos = [s for y, s in pairs if y == 1]
    neg = [s for y, s in pairs if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    total = float(len(pos) * len(neg))
    for ps in pos:
        for ns in neg:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / total


def normalize_feature(rows: list[dict[str, str]], field: str) -> dict[str, float]:
    values: list[tuple[str, float]] = []
    for row in rows:
        value = safe_float(row.get(field))
        if value is not None:
            values.append((row["row_id"], value))
    if not values:
        return {}
    vals = [v for _, v in values]
    lo = min(vals)
    hi = max(vals)
    if math.isclose(lo, hi):
        return {}
    out: dict[str, float] = {}
    for row_id, value in values:
        norm = (value - lo) / (hi - lo)
        out[row_id] = 1.0 - norm if field in LOW_IS_RISK else norm
    return out


def feature_list(groups: Sequence[str]) -> list[str]:
    fields: list[str] = []
    for group in groups:
        for field in GROUP_FIELDS[group]:
            if field not in fields:
                fields.append(field)
    return fields


def score_rows(rows: list[dict[str, str]], fields: Sequence[str]) -> tuple[dict[str, float], dict[str, Any]]:
    normalized = {field: normalize_feature(rows, field) for field in fields}
    used_fields = [field for field, values in normalized.items() if values]
    scores: dict[str, float] = {}
    feature_counts: dict[str, int] = {}
    for row in rows:
        vals = [normalized[field][row["row_id"]] for field in used_fields if row["row_id"] in normalized[field]]
        if vals:
            scores[row["row_id"]] = mean(vals)
            feature_counts[row["row_id"]] = len(vals)
    meta = {
        "requested_fields": list(fields),
        "used_fields": used_fields,
        "dropped_constant_or_empty_fields": [field for field in fields if field not in used_fields],
        "scored_rows": len(scores),
        "feature_count_min": min(feature_counts.values()) if feature_counts else 0,
        "feature_count_max": max(feature_counts.values()) if feature_counts else 0,
        "feature_count_mean": mean(feature_counts.values()) if feature_counts else 0.0,
    }
    return scores, meta


def metrics_for_scores(
    rows: list[dict[str, str]],
    scores: Mapping[str, float],
    selection_quantile: float,
) -> dict[str, Any]:
    scored = [row for row in rows if row["row_id"] in scores]
    score_vals = [scores[row["row_id"]] for row in scored]
    threshold = percentile(score_vals, selection_quantile)
    if threshold is None:
        return {
            "scored_rows": 0,
            "threshold": None,
            "bad_recall": None,
            "good_false_positive_rate": None,
            "balanced_accuracy": None,
            "auc": None,
            "positive_count": 0,
            "top5_hit_rate": None,
            "top10_hit_rate": None,
            "sequence_coverage": [],
        }
    selected = [row for row in scored if scores[row["row_id"]] >= threshold]
    bad = [row for row in scored if int(row.get("target_label") or 0) == 1]
    good = [row for row in scored if int(row.get("target_label") or 0) == 0]
    selected_bad = [row for row in selected if int(row.get("target_label") or 0) == 1]
    selected_good = [row for row in selected if int(row.get("target_label") or 0) == 0]
    labels = [int(row.get("target_label") or 0) for row in scored]
    ordered = sorted(scored, key=lambda row: scores[row["row_id"]], reverse=True)
    top5 = ordered[:5]
    top10 = ordered[:10]
    bad_recall = len(selected_bad) / len(bad) if bad else None
    good_fpr = len(selected_good) / len(good) if good else None
    tnr = 1.0 - good_fpr if good_fpr is not None else None
    return {
        "scored_rows": len(scored),
        "threshold": threshold,
        "bad_rows": len(bad),
        "good_rows": len(good),
        "selected_bad": len(selected_bad),
        "selected_good": len(selected_good),
        "max_possible_bad_recall_at_positive_count": (min(len(selected), len(bad)) / len(bad)) if bad else None,
        "bad_recall": bad_recall,
        "good_false_positive_rate": good_fpr,
        "balanced_accuracy": ((bad_recall + tnr) / 2.0) if bad_recall is not None and tnr is not None else None,
        "auc": auc_score(labels, score_vals),
        "positive_count": len(selected),
        "top5_hit_rate": (sum(int(row.get("target_label") or 0) for row in top5) / len(top5)) if top5 else None,
        "top10_hit_rate": (sum(int(row.get("target_label") or 0) for row in top10) / len(top10)) if top10 else None,
        "sequence_coverage": sorted({row.get("seq", "") for row in scored if row.get("seq")}),
        "selected_sequence_coverage": sorted({row.get("seq", "") for row in selected if row.get("seq")}),
    }


def bootstrap_auc_ci(rows: list[dict[str, str]], scores: Mapping[str, float], iters: int, seed: int) -> list[float | None]:
    scored = [row for row in rows if row["row_id"] in scores]
    if len(scored) < 4:
        return [None, None]
    rng = random.Random(seed)
    aucs: list[float] = []
    for _ in range(iters):
        sample = [scored[rng.randrange(len(scored))] for _ in range(len(scored))]
        labels = [int(row.get("target_label") or 0) for row in sample]
        vals = [scores[row["row_id"]] for row in sample]
        auc = auc_score(labels, vals)
        if auc is not None:
            aucs.append(auc)
    if not aucs:
        return [None, None]
    return [percentile(aucs, 0.025), percentile(aucs, 0.975)]


def leave_one_seq_auc(rows: list[dict[str, str]], scores: Mapping[str, float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for seq in sorted({row.get("seq", "") for row in rows if row.get("seq")}):
        held = [row for row in rows if row.get("seq") == seq and row["row_id"] in scores]
        auc = auc_score([int(row.get("target_label") or 0) for row in held], [scores[row["row_id"]] for row in held])
        out[seq] = auc
    positive = sum(1 for value in out.values() if value is not None and value > 0.5)
    return {"auc_by_seq": out, "positive_folds": positive}


def control_auc_p95(rows: list[dict[str, str]], scores: Mapping[str, float], iters: int, seed: int) -> float | None:
    scored = [row for row in rows if row["row_id"] in scores]
    if len(scored) < 4:
        return None
    labels = [int(row.get("target_label") or 0) for row in scored]
    vals = [scores[row["row_id"]] for row in scored]
    rng = random.Random(seed)
    aucs: list[float] = []
    for _ in range(iters):
        shuffled = list(labels)
        rng.shuffle(shuffled)
        auc = auc_score(shuffled, vals)
        if auc is not None:
            aucs.append(auc)
    return percentile(aucs, 0.95) if aucs else None


def semantic_shuffle_auc_p95(
    rows: list[dict[str, str]],
    combo_fields: Sequence[str],
    iters: int,
    seed: int,
) -> float | None:
    semantic = [field for field in combo_fields if field in SEMANTIC_FIELDS]
    if not semantic:
        return None
    base_fields = [field for field in combo_fields if field not in SEMANTIC_FIELDS]
    rng = random.Random(seed)
    aucs: list[float] = []
    for _ in range(iters):
        shuffled_rows = [dict(row) for row in rows]
        for field in semantic:
            values = [row.get(field, "") for row in shuffled_rows]
            rng.shuffle(values)
            for row, value in zip(shuffled_rows, values):
                row[field] = value
        scores, _ = score_rows(shuffled_rows, list(base_fields) + semantic)
        scored = [row for row in shuffled_rows if row["row_id"] in scores]
        auc = auc_score(
            [int(row.get("target_label") or 0) for row in scored],
            [scores[row["row_id"]] for row in scored],
        )
        if auc is not None:
            aucs.append(auc)
    return percentile(aucs, 0.95) if aucs else None


def audit(rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    combo_scores: dict[str, dict[str, float]] = {}
    results: list[dict[str, Any]] = []
    for idx, (combo, groups) in enumerate(COMBINATIONS.items()):
        fields = feature_list(groups)
        scores, score_meta = score_rows(rows, fields)
        combo_scores[combo] = scores
        metrics = metrics_for_scores(rows, scores, args.selection_quantile)
        ci = bootstrap_auc_ci(rows, scores, args.bootstrap_iters, args.seed + idx)
        loso = leave_one_seq_auc(rows, scores)
        permutation_p95 = control_auc_p95(rows, scores, args.control_iters, args.seed + 1000 + idx)
        semantic_p95 = semantic_shuffle_auc_p95(rows, fields, args.control_iters, args.seed + 2000 + idx)
        geometry_auc = None
        if combo != "C0_geometry_only" and combo_scores.get("C0_geometry_only"):
            c0 = metrics_for_scores(rows, combo_scores["C0_geometry_only"], args.selection_quantile)
            geometry_auc = c0.get("auc")
        auc = metrics.get("auc")
        delta_sem = (auc - geometry_auc) if auc is not None and geometry_auc is not None and any(g in groups for g in ("G1", "G2")) else None
        delta_shuffle = (auc - semantic_p95) if auc is not None and semantic_p95 is not None else None
        pass_gate = (
            (metrics.get("bad_recall") is not None and metrics["bad_recall"] >= 0.60)
            and (metrics.get("good_false_positive_rate") is not None and metrics["good_false_positive_rate"] <= 0.25)
            and len(metrics.get("sequence_coverage") or []) >= 3
            and (delta_sem is None or delta_sem >= 0.05)
            and (delta_shuffle is None or delta_shuffle >= 0.05)
            and loso["positive_folds"] >= 3
        )
        row = {
            "combo": combo,
            "groups": groups,
            "fields": fields,
            **score_meta,
            **metrics,
            "auc_ci95_low": ci[0],
            "auc_ci95_high": ci[1],
            "leave_one_sequence_out_auc_json": loso["auc_by_seq"],
            "leave_one_sequence_out_positive_folds": loso["positive_folds"],
            "semantic_shuffle_p95": semantic_p95,
            "same_mass_random_p95": permutation_p95,
            "delta_sem_vs_geometry_auc": delta_sem,
            "delta_shuffle_auc": delta_shuffle,
            "gate_pass": pass_gate,
        }
        results.append(row)
    passing = [row for row in results if row["gate_pass"]]
    summary = {
        "schema": "acl2_v83_phase2_clue_sufficiency_summary_v1",
        "matrix": str(args.matrix),
        "rows": len(rows),
        "row_counts_by_scope": dict(Counter(row.get("row_scope", "") for row in rows)),
        "case_counts": dict(Counter(row.get("case_type", "") for row in rows)),
        "selection_quantile": args.selection_quantile,
        "bootstrap_iters": args.bootstrap_iters,
        "control_iters": args.control_iters,
        "seed": args.seed,
        "combo_count": len(results),
        "passing_combos": [row["combo"] for row in passing],
        "phase2_gate_pass": bool(passing),
        "best_by_auc": max(results, key=lambda row: row.get("auc") or -1)["combo"] if results else "",
        "best_by_balanced_accuracy": max(results, key=lambda row: row.get("balanced_accuracy") or -1)["combo"] if results else "",
        "interpretation": interpret(results),
    }
    return results, summary


def interpret(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_combo = {row["combo"]: row for row in results}
    c0_pass = bool(by_combo.get("C0_geometry_only", {}).get("gate_pass"))
    c4_pass = bool(by_combo.get("C4_geometry_semantic", {}).get("gate_pass"))
    c7_pass = bool(by_combo.get("C7_full_combined", {}).get("gate_pass"))
    if c0_pass and not (c4_pass or c7_pass):
        decision = "geometry_clues_sufficient_semantic_current_roles_not_helpful"
    elif not c0_pass and (c4_pass or c7_pass):
        decision = "semantic_or_combined_clues_useful"
    elif not any(bool(row.get("gate_pass")) for row in results):
        decision = "clues_insufficient_or_current_fixed_rules_too_weak"
    elif c7_pass:
        decision = "full_combined_clues_sufficient"
    else:
        decision = "some_clue_group_sufficient"
    return {
        "decision": decision,
        "c0_geometry_only_pass": c0_pass,
        "c4_geometry_semantic_pass": c4_pass,
        "c7_full_combined_pass": c7_pass,
    }


def write_report(path: Path, rows: list[dict[str, Any]], summary: Mapping[str, Any]) -> None:
    lines = [
        "# ACL2 v83 Phase2 Clue Sufficiency Report",
        "",
        f"phase2_gate_pass: `{summary['phase2_gate_pass']}`",
        f"passing_combos: `{summary['passing_combos']}`",
        f"interpretation: `{summary['interpretation']['decision']}`",
        "",
        "| Combo | Gate | Bad Recall | Good FPR | Balanced Acc | AUC | LOSO+ | DeltaSem | DeltaShuffle |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {combo} | {gate} | {bad} | {fpr} | {bal} | {auc} | {loso} | {dsem} | {dshuf} |".format(
                combo=row["combo"],
                gate=row["gate_pass"],
                bad=format_float(row.get("bad_recall")),
                fpr=format_float(row.get("good_false_positive_rate")),
                bal=format_float(row.get("balanced_accuracy")),
                auc=format_float(row.get("auc")),
                loso=row.get("leave_one_sequence_out_positive_folds"),
                dsem=format_float(row.get("delta_sem_vs_geometry_auc")),
                dshuf=format_float(row.get("delta_shuffle_auc")),
            )
        )
    lines.extend(
        [
            "",
            "No classifier, logistic regression, decision tree, or learned weights were used. Scores are fixed-direction min-max risk rankings with equal averaging over present features.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def format_float(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.6f}"


def main() -> None:
    args = parse_args()
    rows = read_csv(args.matrix)
    results, summary = audit(rows, args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "clue_sufficiency_rows.csv", results)
    write_json(args.out_dir / "clue_sufficiency_summary.json", summary)
    write_report(args.out_dir / "clue_sufficiency_report.md", results, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
