from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from tools.run_v46_raw_carrier_incidence_repair import ROOT, _json_safe, _rank_auc, _safe_mean, _safe_quantile


POSITIVE_VARIANTS = {
    "P4_vc_q_temporal": ("P4_shuffled_vc_q_temporal", "P3_view_consensus_q"),
    "P5_p4_semantic_boost_capped": ("P5_shuffled_semantic_boost_capped", "P5_no_temporal_semantic_boost_capped"),
    "P5_p4_semantic_linear_capped": ("P5_shuffled_semantic_boost_capped", "P5_no_temporal_semantic_boost_capped"),
    "P5_p4_semantic_product_rescore_capped": (
        "P5_shuffled_semantic_product_rescore_capped",
        "P5_no_temporal_semantic_product_rescore_capped",
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
            writer.writerow({key: _csv_cell(row.get(key)) for key in keys})


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True)
    return value


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def _precision_at_k(rows: list[dict[str, Any]], score_key: str, k: int) -> float | None:
    ranked = sorted(rows, key=lambda row: _as_float(row.get(score_key)), reverse=True)[: min(int(k), len(rows))]
    if not ranked:
        return None
    return float(sum(1 for row in ranked if _as_bool(row.get("diagnostic_same_gt"))) / len(ranked))


def _negative_precision(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return float(sum(1 for row in rows if not _as_bool(row.get("diagnostic_same_gt"))) / len(rows))


def _local_candidate_precision(rows: list[dict[str, Any]], score_key: str, topk: int) -> tuple[float | None, int]:
    by_node: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for side in ["left", "right"]:
            node_key = f"{row.get('scene')}:{row.get(f'{side}_node_id')}"
            by_node.setdefault(node_key, []).append(row)
    hits = 0
    total = 0
    for candidates in by_node.values():
        ranked = sorted(candidates, key=lambda row: _as_float(row.get(score_key)), reverse=True)[: min(int(topk), len(candidates))]
        for row in ranked:
            total += 1
            hits += int(_as_bool(row.get("diagnostic_same_gt")))
    if total == 0:
        return None, 0
    return float(hits / total), int(len(by_node))


def _local_summary_for_variant(
    *,
    scene: str,
    filter_key: str,
    rows: list[dict[str, Any]],
    removed_rows: list[dict[str, Any]],
    variant: str,
    topk_values: list[int],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    shuffled_key, no_temporal_key = POSITIVE_VARIANTS[variant]
    for topk in topk_values:
        precision, node_count = _local_candidate_precision(rows, variant, topk)
        shared_precision, _shared_node_count = _local_candidate_precision(rows, "shared_carrier_jaccard", topk)
        shuffled_precision, _shuffled_node_count = _local_candidate_precision(rows, shuffled_key, topk)
        no_temporal_precision, _no_temporal_node_count = _local_candidate_precision(rows, no_temporal_key, topk)
        feature_precision, _feature_node_count = _local_candidate_precision(rows, "P6_feature_only", topk)
        out.append(
            {
                "scene": scene,
                "filter_key": filter_key,
                "variant": variant,
                "local_topk": int(topk),
                "edge_count_after_filter": len(rows),
                "edge_count_removed": len(removed_rows),
                "removed_negative_precision": _negative_precision(removed_rows),
                "node_count_with_candidates": node_count,
                "local_precision": precision,
                "shared_local_precision": shared_precision,
                "shuffled_local_precision": shuffled_precision,
                "no_temporal_local_precision": no_temporal_precision,
                "P6_feature_only_local_precision": feature_precision,
                "local_precision_minus_shared": None
                if precision is None or shared_precision is None
                else float(precision - shared_precision),
                "local_precision_minus_shuffled": None
                if precision is None or shuffled_precision is None
                else float(precision - shuffled_precision),
                "local_precision_minus_no_temporal": None
                if precision is None or no_temporal_precision is None
                else float(precision - no_temporal_precision),
                "P6_feature_only_minus_P5_local_precision": None
                if precision is None or feature_precision is None
                else float(feature_precision - precision),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
                "diagnostic_only": True,
            }
        )
    return out


def _summary_for_variant(
    *,
    scene: str,
    filter_key: str,
    rows: list[dict[str, Any]],
    removed_rows: list[dict[str, Any]],
    variant: str,
    positive_threshold: float,
) -> dict[str, Any]:
    labels = [_as_bool(row.get("diagnostic_same_gt")) for row in rows]
    scores = [_as_float(row.get(variant)) for row in rows]
    shared_scores = [_as_float(row.get("shared_carrier_jaccard")) for row in rows]
    shuffled_key, no_temporal_key = POSITIVE_VARIANTS[variant]
    shuffled_scores = [_as_float(row.get(shuffled_key)) for row in rows]
    no_temporal_scores = [_as_float(row.get(no_temporal_key)) for row in rows]
    auc = _rank_auc(labels, scores)
    shared_auc = _rank_auc(labels, shared_scores)
    shuffled_auc = _rank_auc(labels, shuffled_scores)
    no_temporal_auc = _rank_auc(labels, no_temporal_scores)
    p5 = _precision_at_k(rows, variant, 5000)
    shared_p5 = _precision_at_k(rows, "shared_carrier_jaccard", 5000)
    feature_only_auc = _rank_auc(labels, [_as_float(row.get("P6_feature_only")) for row in rows])
    out = {
        "scene": scene,
        "filter_key": filter_key,
        "variant": variant,
        "edge_count_after_filter": len(rows),
        "edge_count_removed": len(removed_rows),
        "removed_negative_precision": _negative_precision(removed_rows),
        "positive_edge_density@threshold": float(sum(1 for score in scores if score >= float(positive_threshold)) / max(len(scores), 1)),
        "edge_same_gt_AUC": auc,
        "shared_edge_AUC_after_filter": shared_auc,
        "shuffled_edge_AUC_after_filter": shuffled_auc,
        "no_temporal_edge_AUC_after_filter": no_temporal_auc,
        "edge_precision@top5k": p5,
        "shared_precision@top5k_after_filter": shared_p5,
        "score_mean": _safe_mean(scores),
        "score_p90": _safe_quantile(scores, 0.90),
        "real_minus_shared_edge_AUC": None if auc is None or shared_auc is None else float(auc - shared_auc),
        "precision_top5k_minus_shared": None if p5 is None or shared_p5 is None else float(p5 - shared_p5),
        "real_minus_shuffled_edge_AUC": None if auc is None or shuffled_auc is None else float(auc - shuffled_auc),
        "real_minus_no_temporal_edge_AUC": None if auc is None or no_temporal_auc is None else float(auc - no_temporal_auc),
        "P6_feature_only_edge_AUC_after_filter": feature_only_auc,
        "P6_feature_only_minus_P5_edge_AUC": None if auc is None or feature_only_auc is None else float(feature_only_auc - auc),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "diagnostic_only": True,
    }
    out["P6_feature_only_beats_full_P5"] = bool(
        out["P6_feature_only_minus_P5_edge_AUC"] is not None and out["P6_feature_only_minus_P5_edge_AUC"] > 0.0
    )
    is_p5 = variant.startswith("P5_")
    out["gate_pass"] = bool(
        is_p5
        and out["real_minus_shared_edge_AUC"] is not None
        and out["real_minus_shared_edge_AUC"] >= 0.08
        and out["precision_top5k_minus_shared"] is not None
        and out["precision_top5k_minus_shared"] >= 0.10
        and out["real_minus_shuffled_edge_AUC"] is not None
        and out["real_minus_shuffled_edge_AUC"] >= 0.10
        and out["real_minus_no_temporal_edge_AUC"] is not None
        and out["real_minus_no_temporal_edge_AUC"] >= 0.08
        and not out["P6_feature_only_beats_full_P5"]
    )
    return out


def _filter_rows(rows: list[dict[str, Any]], filter_key: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    removed = [row for row in rows if _as_bool(row.get(filter_key))]
    kept = [row for row in rows if not _as_bool(row.get(filter_key))]
    return kept, removed


def _scene_rows(
    input_root: Path,
    filter_key: str,
    positive_threshold: float,
    topk_values: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _read_csv(input_root / "raw_visual_semantic_edge_rows.csv")
    scenes = sorted({str(row.get("scene")) for row in rows})
    summary_rows: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []
    for scene in scenes:
        scene_rows = [row for row in rows if str(row.get("scene")) == scene]
        kept, removed = _filter_rows(scene_rows, filter_key)
        for variant in POSITIVE_VARIANTS:
            summary_rows.append(
                _summary_for_variant(
                    scene=scene,
                    filter_key=filter_key,
                    rows=kept,
                    removed_rows=removed,
                    variant=variant,
                    positive_threshold=float(positive_threshold),
                )
            )
            local_rows.extend(
                _local_summary_for_variant(
                    scene=scene,
                    filter_key=filter_key,
                    rows=kept,
                    removed_rows=removed,
                    variant=variant,
                    topk_values=topk_values,
                )
            )
    return summary_rows, local_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="N4-filtered positive-edge diagnostic for v46 raw visual semantic repair outputs.")
    parser.add_argument("--input-roots", required=True, help="Comma-separated roots containing raw_visual_semantic_edge_rows.csv.")
    parser.add_argument("--filter-keys", required=True, help="Comma-separated N4 flag columns to use as hard-veto filters.")
    parser.add_argument("--positive-threshold", type=float, default=0.50)
    parser.add_argument("--local-topk", default="1,3")
    parser.add_argument("--output-root", default="outputs/audit/v46_n4_filtered_positive_repair")
    args = parser.parse_args()

    input_roots = [ROOT / item.strip() for item in str(args.input_roots).split(",") if item.strip()]
    filter_keys = [item.strip() for item in str(args.filter_keys).split(",") if item.strip()]
    topk_values = [int(item.strip()) for item in str(args.local_topk).split(",") if item.strip()]
    all_summary_rows: list[dict[str, Any]] = []
    all_local_rows: list[dict[str, Any]] = []
    for input_root in input_roots:
        for filter_key in filter_keys:
            rows, local_rows = _scene_rows(input_root, filter_key, float(args.positive_threshold), topk_values)
            for row in rows:
                row["input_root"] = str(input_root.relative_to(ROOT) if input_root.is_relative_to(ROOT) else input_root)
            for row in local_rows:
                row["input_root"] = str(input_root.relative_to(ROOT) if input_root.is_relative_to(ROOT) else input_root)
            all_summary_rows.extend(rows)
            all_local_rows.extend(local_rows)

    gate_rows = [row for row in all_summary_rows if str(row.get("variant")).startswith("P5_")]
    gate = {
        "any_scene_variant_gate_pass": any(bool(row.get("gate_pass")) for row in gate_rows),
        "all_scene_variant_gate_pass": False,
        "pass": False,
        "diagnostic_only": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    for input_root in sorted({str(row.get("input_root")) for row in gate_rows}):
        for filter_key in sorted({str(row.get("filter_key")) for row in gate_rows}):
            for variant in sorted({str(row.get("variant")) for row in gate_rows}):
                selected = [
                    row
                    for row in gate_rows
                    if str(row.get("input_root")) == input_root and str(row.get("filter_key")) == filter_key and str(row.get("variant")) == variant
                ]
                if selected and all(bool(row.get("gate_pass")) for row in selected):
                    gate["all_scene_variant_gate_pass"] = True
    gate["pass"] = bool(gate["all_scene_variant_gate_pass"])

    payload = {
        "phase": "v46_n4_filtered_positive_repair",
        "created_at": _utc_now(),
        "input_roots": [str(root.relative_to(ROOT) if root.is_relative_to(ROOT) else root) for root in input_roots],
        "filter_keys": filter_keys,
        "positive_threshold": float(args.positive_threshold),
        "local_topk": topk_values,
        "summary_rows": all_summary_rows,
        "local_candidate_rows": all_local_rows,
        "gate": gate,
        "note": "Diagnostic only: N4 hard-veto filtered positive-edge metrics test whether semantic contradiction filtering fixes P5 ranking.",
    }
    out = ROOT / str(args.output_root)
    _write_json(out / "n4_filtered_positive_repair.json", payload)
    _write_csv(out / "n4_filtered_positive_summary_rows.csv", all_summary_rows)
    _write_csv(out / "n4_filtered_local_candidate_rows.csv", all_local_rows)
    print(json.dumps({"summary": str(out / "n4_filtered_positive_repair.json"), "gate": gate}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
