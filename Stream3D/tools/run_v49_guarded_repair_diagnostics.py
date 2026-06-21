from __future__ import annotations

import argparse
from itertools import product
from typing import Any

from stream4d_native.v49_mosaic_stage1 import (
    _unpack_components,
    evaluate_component_assignment,
    load_optional_csv,
    load_optional_json,
    parse_float,
    parse_int,
    utc_now,
    write_bundle,
)


def _score_compact(row: dict[str, Any], support_soft: float, support_hard: float) -> float:
    semantic = parse_float(row.get("semantic_set_score"), 1.0)
    temporal = parse_float(row.get("temporal_support_score"))
    mask = parse_float(row.get("mask_support_score"))
    conflict = parse_float(row.get("hypothesis_conflict_rate"))
    size = parse_float(row.get("hypothesis_size"), 1.0)
    over_support = max(0.0, mask - support_soft) / max(1.0, support_hard - support_soft)
    return (
        0.62 * semantic
        + 0.14 * min(temporal, 0.70)
        + 0.10 * min(mask, support_soft) / support_soft
        - 1.20 * conflict
        - 0.90 * over_support
        - 0.05 * max(0.0, size - 2.0)
    )


def _select_and_eval(
    hypotheses: list[dict[str, Any]],
    vote_rows: list[dict[str, str]],
    candidates: list[dict[str, Any]],
    *,
    run_id: str,
    fill_score_key: str = "score_semantic_guard",
) -> tuple[int, dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_components: set[str] = set()
    scene_counts: dict[str, int] = {}

    def add(row: dict[str, Any]) -> None:
        comps = _unpack_components(row.get("components"))
        scene = str(row.get("scene") or "")
        if not comps or any(comp in used_components for comp in comps):
            return
        if scene_counts.get(scene, 0) >= 150:
            return
        selected.append(row)
        used_components.update(comps)
        scene_counts[scene] = scene_counts.get(scene, 0) + 1

    for row in candidates:
        add(row)
    singletons = [row for row in hypotheses if parse_int(row.get("hypothesis_size")) == 1]
    for row in sorted(singletons, key=lambda item: parse_float(item.get(fill_score_key)), reverse=True):
        add(row)

    comp_to_obj: dict[str, str] = {}
    for idx, row in enumerate(selected):
        obj = f"{row.get('scene')}|{run_id}|h{idx:04d}"
        for comp in _unpack_components(row.get("components")):
            comp_to_obj[comp] = obj
    metrics = evaluate_component_assignment(vote_rows, comp_to_obj)
    multi_count = sum(1 for row in selected if parse_int(row.get("hypothesis_size")) > 1)
    return multi_count, metrics


def _row_from_metrics(kind: str, params: dict[str, Any], multi_count: int, metrics: dict[str, Any]) -> dict[str, Any]:
    row = {
        "diagnostic_kind": kind,
        **params,
        "multi_hypothesis_count": multi_count,
        "4D_ARI": metrics.get("4D_ARI"),
        "4D_purity": metrics.get("4D_purity"),
        "4D_completeness": metrics.get("4D_completeness"),
        "conflict_rate": metrics.get("conflict_rate"),
        "scene0011_purity": metrics.get("scene0011_purity"),
        "scene0050_purity": metrics.get("scene0050_purity"),
        "scene0081_ARI": metrics.get("scene0081_ARI"),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    row["stage_core_pass"] = bool(
        parse_float(row.get("4D_ARI")) >= 0.485
        and parse_float(row.get("4D_purity")) >= 0.875
        and parse_float(row.get("4D_completeness")) >= 0.555
        and parse_float(row.get("conflict_rate")) <= 0.10
    )
    row["stage_core_pass_count"] = sum(
        [
            parse_float(row.get("4D_ARI")) >= 0.485,
            parse_float(row.get("4D_purity")) >= 0.875,
            parse_float(row.get("4D_completeness")) >= 0.555,
            parse_float(row.get("conflict_rate")) <= 0.10,
        ]
    )
    return row


def build_guarded_repair_diagnostics() -> dict[str, Any]:
    scoring = load_optional_json("outputs/audit/v49_hypothesis_scoring/hypothesis_scoring_summary.json")
    selection = load_optional_json("outputs/audit/v49_hypothesis_selection/hypothesis_selection_summary.json")
    hypotheses = [dict(row) for row in scoring.get("hypothesis_rows", [])]
    vote_rows = load_optional_csv("outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_mask_vote_rows.csv")

    raw_metrics = evaluate_component_assignment(vote_rows)
    raw_row = _row_from_metrics("raw_u32_fill", {}, 0, raw_metrics)

    oracle_rows: list[dict[str, Any]] = []
    for min_purity in [1.0, 0.95, 0.90, 0.875, 0.85, 0.80, 0.75]:
        for min_completeness in [0.01, 0.05, 0.10, 0.20, 0.30]:
            candidates = []
            for row in hypotheses:
                if parse_int(row.get("hypothesis_size")) <= 1 or parse_int(row.get("hypothesis_size")) > 4:
                    continue
                if parse_float(row.get("hypothesis_purity")) < min_purity:
                    continue
                if parse_float(row.get("hypothesis_completeness")) < min_completeness:
                    continue
                if parse_float(row.get("hypothesis_conflict_rate")) > 0.20:
                    continue
                candidates.append(row)
            candidates.sort(
                key=lambda row: (
                    parse_float(row.get("hypothesis_purity")),
                    parse_float(row.get("hypothesis_completeness")),
                    -parse_float(row.get("hypothesis_conflict_rate")),
                ),
                reverse=True,
            )
            multi_count, metrics = _select_and_eval(hypotheses, vote_rows, candidates, run_id="oracle")
            if multi_count:
                oracle_rows.append(
                    _row_from_metrics(
                        "gt_oracle_not_method",
                        {"min_hypothesis_purity": min_purity, "min_hypothesis_completeness": min_completeness},
                        multi_count,
                        metrics,
                    )
                )

    progressive_rows: list[dict[str, Any]] = []
    quality_cases = {
        "strict_semantic": {"max_conflict": 0.0, "min_semantic": 0.95, "max_risk": 0.0, "max_size": 3},
        "guarded": {"max_conflict": 0.05, "min_semantic": 0.85, "max_risk": 0.05, "max_size": 3},
    }
    for score_key in ["score_semantic_guard", "score_guarded_full"]:
        for quality, config in quality_cases.items():
            filtered = []
            for row in hypotheses:
                if parse_int(row.get("hypothesis_size")) <= 1:
                    continue
                if parse_int(row.get("hypothesis_size")) > config["max_size"]:
                    continue
                if parse_float(row.get("hypothesis_conflict_rate")) > config["max_conflict"]:
                    continue
                if parse_float(row.get("semantic_set_score"), 1.0) < config["min_semantic"]:
                    continue
                if parse_float(row.get("large_support_risk")) > config["max_risk"]:
                    continue
                filtered.append(row)
            ranked = sorted(filtered, key=lambda row: parse_float(row.get(score_key)), reverse=True)
            for topn in [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]:
                multi_count, metrics = _select_and_eval(hypotheses, vote_rows, ranked[:topn], run_id="progressive")
                progressive_rows.append(
                    _row_from_metrics(
                        "progressive_non_gt",
                        {"score_key": score_key, "quality": quality, "topn": topn},
                        multi_count,
                        metrics,
                    )
                )

    compact_rows: list[dict[str, Any]] = []
    for min_semantic, max_size, support_soft, support_hard, topn in product(
        [0.55, 0.65, 0.75, 0.85],
        [2, 3],
        [12, 20, 30],
        [25, 35, 50],
        [20, 40, 70, 110],
    ):
        candidates = []
        for row in hypotheses:
            size = parse_int(row.get("hypothesis_size"))
            if size <= 1 or size > max_size:
                continue
            if parse_float(row.get("hypothesis_conflict_rate")) > 0.0:
                continue
            if parse_float(row.get("semantic_set_score"), 1.0) < min_semantic:
                continue
            if parse_float(row.get("mask_support_score")) > support_hard:
                continue
            candidates.append((_score_compact(row, support_soft, support_hard), row))
        ranked = [row for _score, row in sorted(candidates, key=lambda item: item[0], reverse=True)[:topn]]
        multi_count, metrics = _select_and_eval(hypotheses, vote_rows, ranked, run_id="compact")
        compact_rows.append(
            _row_from_metrics(
                "compact_support_non_gt",
                {
                    "min_semantic": min_semantic,
                    "max_size": max_size,
                    "support_soft": support_soft,
                    "support_hard": support_hard,
                    "topn": topn,
                },
                multi_count,
                metrics,
            )
        )

    def best(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {}
        return max(
            rows,
            key=lambda row: (
                parse_int(row.get("stage_core_pass_count")),
                parse_float(row.get("4D_ARI")),
                parse_float(row.get("4D_purity")),
                parse_float(row.get("4D_completeness")),
            ),
        )

    payload = {
        "phase": "v49_guarded_repair_diagnostics",
        "created_at": utc_now(),
        "raw_row": raw_row,
        "current_best_real_row": selection.get("best_real_row"),
        "best_oracle_row": best(oracle_rows),
        "best_progressive_non_gt_row": best(progressive_rows),
        "best_compact_support_non_gt_row": best(compact_rows),
        "oracle_any_stage_core_pass": any(row.get("stage_core_pass") for row in oracle_rows),
        "progressive_any_stage_core_pass": any(row.get("stage_core_pass") for row in progressive_rows),
        "compact_support_any_stage_core_pass": any(row.get("stage_core_pass") for row in compact_rows),
        "interpretation": (
            "GT-only oracle shows whether candidates can pass in principle; non-GT progressive/compact rows test the plan's "
            "conflict/semantic/large-support repair direction without using GT for prediction."
        ),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "oracle_rows": oracle_rows,
        "progressive_rows": progressive_rows,
        "compact_support_rows": compact_rows,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v49 guarded repair diagnostics.")
    parser.add_argument("--output-root", default="outputs/audit/v49_guarded_repair_diagnostics")
    args = parser.parse_args()
    payload = build_guarded_repair_diagnostics()
    write_bundle(
        args.output_root,
        "guarded_repair_diagnostics_summary",
        payload,
        {
            "guarded_oracle_rows": payload["oracle_rows"],
            "guarded_progressive_rows": payload["progressive_rows"],
            "guarded_compact_support_rows": payload["compact_support_rows"],
        },
    )
    print(
        {
            "summary": f"{args.output_root}/guarded_repair_diagnostics_summary.json",
            "oracle_any_stage_core_pass": payload["oracle_any_stage_core_pass"],
            "progressive_any_stage_core_pass": payload["progressive_any_stage_core_pass"],
            "compact_support_any_stage_core_pass": payload["compact_support_any_stage_core_pass"],
        }
    )


if __name__ == "__main__":
    main()
