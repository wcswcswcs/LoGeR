from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from stream4d_native.v49_mosaic_stage1 import (
    _component_gt_counts,
    _greedy_select,
    _gt_mask_totals,
    _hypothesis_generation_support_score,
    _score_hypothesis,
    _unpack_components,
    evaluate_component_assignment,
    load_optional_csv,
    load_optional_json,
    parse_float,
    parse_int,
    utc_now,
    write_bundle,
)


def _bool(value: Any) -> bool:
    return str(value).lower() == "true"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _source(row: dict[str, Any]) -> str:
    return str(row.get("component_set_candidate_source") or row.get("candidate_generation_source") or "")


def _score_semantic_low_context(row: dict[str, Any]) -> float:
    source = _source(row)
    size = parse_float(row.get("hypothesis_size"), 1.0)
    semantic = parse_float(row.get("semantic_set_score"), 1.0)
    context = parse_float(row.get("context_overlap_proxy"))
    conflict = parse_float(row.get("hypothesis_conflict_rate"))
    support_over = max(0.0, parse_float(row.get("mask_support_score")) - 25.0)
    reliability = parse_float(row.get("mask_reliability_min"), parse_float(row.get("mask_reliability_mean")))
    source_bonus = {
        "singleton": 0.06,
        "pair_edge": 0.08,
        "multi_scale_parent_containment": -0.05,
        "pair_neighborhood": -0.35,
    }.get(source, 0.0)
    return (
        0.72 * semantic
        + 0.25 * reliability
        - 0.52 * context
        - 0.35 * conflict
        - 0.018 * support_over
        - 0.08 * max(0.0, size - 2.0)
        + source_bonus
    )


def _coverage_for_row(row: dict[str, Any], component_column: str, comp_gt_counts: dict[str, Counter[str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for comp in _unpack_components(row.get(component_column)):
        counts.update(comp_gt_counts.get(comp, Counter()))
    return counts


def _best_coverage_map(
    rows: list[dict[str, Any]],
    component_column: str,
    row_kind: str,
    gt_totals: Counter[str],
    comp_gt_counts: dict[str, Counter[str]],
) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        counts = _coverage_for_row(row, component_column, comp_gt_counts)
        total = sum(counts.values())
        if total <= 0:
            continue
        dominant_gt, dominant_count = counts.most_common(1)[0]
        purity = dominant_count / total
        for gt, count in counts.items():
            coverage = count / gt_totals[gt]
            previous = best.get(gt)
            if previous and (coverage < previous["coverage"] or (coverage == previous["coverage"] and purity <= previous["purity"])):
                continue
            if row_kind == "hypothesis":
                row_id = row.get("hypothesis_id")
                score = parse_float(row.get("hypothesis_support_score"))
                size = parse_int(row.get("hypothesis_size"))
                same = _bool(row.get("same_GT_set"))
            else:
                row_id = str(idx)
                score = parse_float(row.get("nonGT_set_score"))
                size = parse_int(row.get("set_size"))
                same = _bool(row.get("same_GT_set"))
            best[gt] = {
                "coverage": coverage,
                "purity": purity,
                "dominant_gt": dominant_gt,
                "row_id": row_id,
                "source": _source(row) if row_kind == "hypothesis" else str(row.get("candidate_source") or ""),
                "score": score,
                "same_gt_set": same,
                "size": size,
            }
    return best


def _recall_rate(gt_totals: Counter[str], best: dict[str, dict[str, Any]], threshold: float) -> float:
    return sum(1 for gt in gt_totals if best.get(gt, {}).get("coverage", 0.0) >= threshold) / max(len(gt_totals), 1)


def _source_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[_source(row)].append(row)
    out: dict[str, dict[str, Any]] = {}
    for source, source_rows in sorted(by_source.items()):
        out[source] = {
            "count": len(source_rows),
            "mean_purity": _mean([parse_float(row.get("hypothesis_purity")) for row in source_rows]),
            "same_gt_rate": _mean([1.0 if _bool(row.get("same_GT_set")) else 0.0 for row in source_rows]),
            "mean_completeness": _mean([parse_float(row.get("hypothesis_completeness")) for row in source_rows]),
            "mean_temporal": _mean([parse_float(row.get("temporal_support_score")) for row in source_rows]),
            "mean_conflict": _mean([parse_float(row.get("hypothesis_conflict_rate")) for row in source_rows]),
        }
    return out


def _ranking_row(name: str, rows: list[dict[str, Any]], score_fn: Callable[[dict[str, Any]], float]) -> dict[str, Any]:
    ordered = sorted(rows, key=score_fn, reverse=True)
    top = ordered[: min(1000, len(ordered))]
    return {
        "variant": name,
        "top1000_mean_purity": _mean([parse_float(row.get("hypothesis_purity")) for row in top]),
        "top1000_same_gt_rate": _mean([1.0 if _bool(row.get("same_GT_set")) else 0.0 for row in top]),
        "top1000_mean_completeness": _mean([parse_float(row.get("hypothesis_completeness")) for row in top]),
        "top1000_conflict_rate": _mean([parse_float(row.get("hypothesis_conflict_rate")) for row in top]),
        "top1000_source_counts": dict(Counter(_source(row) for row in top)),
    }


def _selection_row(
    name: str,
    rows: list[dict[str, Any]],
    score_fn: Callable[[dict[str, Any]], float],
    vote_rows: list[dict[str, str]],
    max_per_scene: int,
) -> dict[str, Any]:
    local_rows = []
    for row in rows:
        local = dict(row)
        local["dry_generation_score"] = score_fn(local)
        local_rows.append(local)
    selected = _greedy_select(local_rows, score_key="dry_generation_score", max_per_scene=max_per_scene)
    comp_to_obj: dict[str, str] = {}
    for idx, hypothesis in enumerate(selected):
        obj = f"{hypothesis.get('scene')}|{name}|h{idx:04d}"
        for comp in _unpack_components(hypothesis.get("components")):
            comp_to_obj[comp] = obj
    metrics = evaluate_component_assignment(vote_rows, comp_to_obj)
    return {
        "variant": name,
        "selected_hypothesis_count": len(selected),
        "selected_source_counts": dict(Counter(_source(row) for row in selected)),
        "selected_mean_hypothesis_purity": _mean([parse_float(row.get("hypothesis_purity")) for row in selected]),
        "selected_mean_hypothesis_completeness": _mean([parse_float(row.get("hypothesis_completeness")) for row in selected]),
        "selected_object_count": metrics.get("selected_object_count"),
        "mean_predictions_per_scene": metrics.get("mean_predictions_per_scene"),
        "object_size_mean": _mean([parse_float(row.get("hypothesis_size"), 1.0) for row in selected]) or 1.0,
        "4D_ARI": metrics.get("4D_ARI"),
        "4D_purity": metrics.get("4D_purity"),
        "4D_completeness": metrics.get("4D_completeness"),
        "conflict_rate": metrics.get("conflict_rate"),
        "scene0081_ARI": metrics.get("scene0081_ARI"),
        "scene0591_completeness": metrics.get("scene0591_completeness"),
    }


def build_gt_fragmentation_hypothesis_recall(max_per_scene: int = 150) -> dict[str, Any]:
    vote_rows = load_optional_csv("outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_mask_vote_rows.csv")
    hypothesis_payload = load_optional_json("outputs/audit/v49_hypothesis_generation/hypothesis_generation_summary.json")
    atlas_payload = load_optional_json("outputs/audit/v49_component_completion_atlas/component_completion_atlas_summary.json")
    hypothesis_rows = [dict(row) for row in hypothesis_payload.get("hypothesis_rows", [])]
    atlas_rows = [dict(row) for row in atlas_payload.get("component_set_rows", [])]
    scored_rows = [_score_hypothesis(dict(row)) for row in hypothesis_rows]
    gt_totals = _gt_mask_totals(vote_rows)
    comp_gt_counts = _component_gt_counts(vote_rows)
    gt_component_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for comp, counts in comp_gt_counts.items():
        for gt, count in counts.items():
            gt_component_counts[gt][comp] = count

    best_hypothesis = _best_coverage_map(scored_rows, "components", "hypothesis", gt_totals, comp_gt_counts)
    best_atlas = _best_coverage_map(atlas_rows, "component_set", "atlas", gt_totals, comp_gt_counts)
    thresholds = [0.25, 0.50, 0.75, 0.90]
    gt_rows: list[dict[str, Any]] = []
    for gt, total in gt_totals.items():
        components = gt_component_counts[gt]
        hyp = best_hypothesis.get(gt, {})
        atlas = best_atlas.get(gt, {})
        gt_rows.append(
            {
                "gt_object": gt,
                "mask_count": int(total),
                "component_count": len(components),
                "dominant_component_coverage": max(components.values()) / total if components else 0.0,
                "best_hypothesis_coverage": hyp.get("coverage", 0.0),
                "best_hypothesis_purity": hyp.get("purity"),
                "best_hypothesis_source": hyp.get("source"),
                "best_hypothesis_id": hyp.get("row_id"),
                "best_hypothesis_size": hyp.get("size"),
                "best_hypothesis_same_gt_set": hyp.get("same_gt_set"),
                "best_atlas_coverage": atlas.get("coverage", 0.0),
                "best_atlas_purity": atlas.get("purity"),
                "best_atlas_source": atlas.get("source"),
                "best_atlas_size": atlas.get("size"),
                "best_atlas_same_gt_set": atlas.get("same_gt_set"),
            }
        )

    ranking_variants: list[tuple[str, Callable[[dict[str, Any]], float]]] = [
        ("current_hypothesis_support_score", lambda row: parse_float(row.get("hypothesis_support_score"))),
        ("score_boundary_prototype_context_hard", lambda row: parse_float(row.get("score_boundary_prototype_context_hard"))),
        ("score_persistent_contradiction_prefilter", lambda row: parse_float(row.get("score_persistent_contradiction_prefilter"))),
        ("generation_semantic_low_context_source_balanced", _score_semantic_low_context),
        ("generation_support_repair", _hypothesis_generation_support_score),
    ]
    ranking_rows = [_ranking_row(name, scored_rows, score_fn) for name, score_fn in ranking_variants]
    selection_rows = [_selection_row(name, scored_rows, score_fn, vote_rows, max_per_scene) for name, score_fn in ranking_variants]

    recall_metrics = {f"hypothesis_recall@{threshold}": _recall_rate(gt_totals, best_hypothesis, threshold) for threshold in thresholds}
    recall_metrics.update({f"atlas_recall@{threshold}": _recall_rate(gt_totals, best_atlas, threshold) for threshold in thresholds})
    for threshold in thresholds:
        missing = [row for row in gt_rows if parse_float(row.get("best_hypothesis_coverage")) < threshold]
        recall_metrics[f"missing_hypothesis_at_{threshold}"] = len(missing)
        recall_metrics[f"missing_fragmented_ge5_at_{threshold}"] = sum(1 for row in missing if parse_int(row.get("component_count")) >= 5)
        recall_metrics[f"missing_large_ge20masks_at_{threshold}"] = sum(1 for row in missing if parse_int(row.get("mask_count")) >= 20)

    repair_candidates = [
        row
        for row in ranking_rows
        if parse_float(row.get("top1000_mean_purity")) >= 0.75 and parse_float(row.get("top1000_conflict_rate")) <= 0.20
    ]
    selection_passlike = [
        row
        for row in selection_rows
        if parse_float(row.get("4D_ARI")) >= 0.5195982004318763 + 0.035
        and parse_float(row.get("4D_completeness")) >= 0.5245503159941662 + 0.07
        and parse_float(row.get("4D_purity")) >= 0.875
    ]
    gate = {
        "diagnostic_inputs_available": bool(vote_rows and hypothesis_rows and atlas_rows),
        "phase3_repair_candidate_found": bool(repair_candidates),
        "selection_stage1_repair_found": bool(selection_passlike),
        "pass": bool(repair_candidates) and bool(selection_passlike),
    }
    return {
        "phase": "v49_gt_fragmentation_hypothesis_recall",
        "created_at": utc_now(),
        "inputs": {
            "vote_rows": "outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_mask_vote_rows.csv",
            "hypothesis_generation": "outputs/audit/v49_hypothesis_generation/hypothesis_generation_summary.json",
            "component_completion_atlas": "outputs/audit/v49_component_completion_atlas/component_completion_atlas_summary.json",
        },
        "baseline_hypothesis_generation_gate": hypothesis_payload.get("gate"),
        "baseline_hypothesis_generation_metrics": hypothesis_payload.get("metrics"),
        "gt_object_count": len(gt_totals),
        "component_count": len(comp_gt_counts),
        "hypothesis_count": len(hypothesis_rows),
        "atlas_set_count": len(atlas_rows),
        "recall_metrics": recall_metrics,
        "top1000_source_summary": _source_summary(scored_rows[:1000]),
        "all_hypothesis_source_summary": _source_summary(scored_rows),
        "gt_object_recall_rows": sorted(gt_rows, key=lambda row: str(row.get("gt_object"))),
        "top_fragmented_gt_objects": sorted(gt_rows, key=lambda row: (parse_int(row.get("component_count")), parse_int(row.get("mask_count"))), reverse=True)[:20],
        "low_coverage_large_gt_objects": sorted(
            [row for row in gt_rows if parse_int(row.get("mask_count")) >= 20 and parse_float(row.get("best_hypothesis_coverage")) < 0.50],
            key=lambda row: (parse_int(row.get("component_count")), parse_int(row.get("mask_count"))),
            reverse=True,
        )[:20],
        "ranking_repair_dry_run_rows": ranking_rows,
        "selection_dry_run_rows": selection_rows,
        "gate": gate,
        "failure_label": None if gate["pass"] else "NO_GO_SELECTION_STAGE1_AFTER_HYPOTHESIS_RECALL_REPAIR",
        "recommendation": (
            "Use the generation_support_repair ranking to prevent pair_neighborhood/high-overlap false hypotheses "
            "from dominating Phase 3 top-k, but do not claim Stage-1 success unless downstream selection/control gates pass."
        ),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Stream4D v49 GT fragmentation and hypothesis-recall repair candidates.")
    parser.add_argument("--output-root", default="outputs/audit/v49_gt_fragmentation_hypothesis_recall")
    parser.add_argument("--max-per-scene", type=int, default=150)
    args = parser.parse_args()
    payload = build_gt_fragmentation_hypothesis_recall(max_per_scene=args.max_per_scene)
    write_bundle(
        Path(args.output_root),
        "gt_fragmentation_hypothesis_recall_summary",
        payload,
        {
            "gt_object_recall_rows": payload["gt_object_recall_rows"],
            "ranking_repair_dry_run_rows": payload["ranking_repair_dry_run_rows"],
            "selection_dry_run_rows": payload["selection_dry_run_rows"],
        },
    )
    print(
        {
            "summary": f"{args.output_root}/gt_fragmentation_hypothesis_recall_summary.json",
            "gate": payload["gate"],
            "failure_label": payload["failure_label"],
        }
    )


if __name__ == "__main__":
    main()
