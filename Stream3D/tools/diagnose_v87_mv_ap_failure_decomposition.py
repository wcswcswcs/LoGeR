#!/usr/bin/env python3
"""Diagnose Stream4D v87 MV AP failure after Phase4 No-Go.

This script is diagnostic-only: it reads fixed prediction/evaluation artifacts
and writes a failure casebook. GT-derived oracle scores are reported only as
upper-bound diagnostics and are never written back into method predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
AP_THRESHOLDS = [round(0.50 + 0.05 * idx, 2) for idx in range(10)]


def _repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "Stream3D":
        return REPO / path
    return ROOT / path


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    return int(round(_num(value, float(default))))


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _variant_from_object_id(mv_object_id: str) -> str:
    return str(mv_object_id).split(":", 1)[0]


def _is_real_variant(variant: str) -> bool:
    return variant.startswith(("B0_", "B1_", "B2_", "B3_", "B4_", "B5_"))


def _variant_role(variant: str) -> str:
    if variant == "B0_local_only":
        return "local_baseline"
    if _is_real_variant(variant):
        return "history_readout"
    if "single_largest" in variant:
        return "single_largest_area_control"
    if "area" in variant:
        return "area_risk_control"
    if "semantic" in variant:
        return "semantic_control"
    if "shuffled" in variant:
        return "shuffled_control"
    if "stale" in variant:
        return "stale_control"
    if "hash" in variant:
        return "hash_control"
    return "control_or_other"


def _ap_from_iou_rows(iou_rows: list[dict[str, Any]], scores: dict[str, float], threshold: float) -> float:
    gt_ids = sorted({str(row.get("gt_object_id", "")) for row in iou_rows})
    pred_ids = sorted({str(row.get("mv_object_id", "")) for row in iou_rows})
    if not gt_ids:
        return 0.0
    if not pred_ids:
        return 0.0
    best_iou = {
        (str(row.get("mv_object_id", "")), str(row.get("gt_object_id", ""))): _num(row.get("mv_iou"), 0.0)
        for row in iou_rows
    }
    ordered = sorted(pred_ids, key=lambda pred_id: (-scores.get(pred_id, 0.0), pred_id))
    matched_gt: set[str] = set()
    curve: list[tuple[float, float]] = []
    tp = 0
    fp = 0
    for pred_id in ordered:
        best_gt = ""
        best = 0.0
        for gt_id in gt_ids:
            if gt_id in matched_gt:
                continue
            value = best_iou.get((pred_id, gt_id), 0.0)
            if value > best:
                best = value
                best_gt = gt_id
        if best_gt and best >= threshold:
            matched_gt.add(best_gt)
            tp += 1
        else:
            fp += 1
        curve.append((_safe_ratio(tp, len(gt_ids)), _safe_ratio(tp, tp + fp)))
    recalls = [0.0] + [row[0] for row in curve] + [1.0]
    precisions = [1.0] + [row[1] for row in curve] + [0.0]
    for idx in range(len(precisions) - 2, -1, -1):
        precisions[idx] = max(precisions[idx], precisions[idx + 1])
    ap = 0.0
    for idx in range(len(recalls) - 1):
        if recalls[idx + 1] != recalls[idx]:
            ap += (recalls[idx + 1] - recalls[idx]) * precisions[idx + 1]
    return float(ap)


def _ap_mean(iou_rows: list[dict[str, Any]], scores: dict[str, float]) -> float:
    return _mean([_ap_from_iou_rows(iou_rows, scores, threshold) for threshold in AP_THRESHOLDS])


def _frame_stats(frame_rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in frame_rows:
        grouped[str(row.get("mv_object_id", ""))].append(row)
    stats: dict[str, dict[str, float]] = {}
    for obj, rows in grouped.items():
        stats[obj] = {
            "frame_mask_count": float(len(rows)),
            "mask_area_sum": sum(_num(row.get("mask_area"), 0.0) for row in rows),
            "mean_mask_area": _mean([_num(row.get("mask_area"), 0.0) for row in rows]),
            "broad_mask_rate": _safe_ratio(sum(str(row.get("broad_mask_flag", "")).lower() == "true" for row in rows), len(rows)),
            "mean_adapter_score": _mean([_num(row.get("adapter_score"), 0.0) for row in rows]),
            "mean_native_support": _mean([_num(row.get("native_carrier_support_count"), 0.0) for row in rows]),
        }
    return stats


def _object_stats(
    iou_rows: list[dict[str, str]],
    object_meta: dict[str, dict[str, str]],
    frame_stats: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in iou_rows:
        scene = str(row.get("scene_id", ""))
        obj = str(row.get("mv_object_id", ""))
        grouped[(scene, _variant_from_object_id(obj), obj)].append(row)
    out: list[dict[str, Any]] = []
    for (scene, variant, obj), rows in sorted(grouped.items()):
        ious = sorted([_num(row.get("mv_iou"), 0.0) for row in rows], reverse=True)
        top = max(rows, key=lambda row: _num(row.get("mv_iou"), 0.0)) if rows else {}
        stats = frame_stats.get(obj, {})
        meta = object_meta.get(obj, {})
        out.append(
            {
                "scene_id": scene,
                "variant": variant,
                "variant_role": _variant_role(variant),
                "mv_object_id": obj,
                "history_id": meta.get("history_id", ""),
                "history_state": meta.get("history_state", ""),
                "object_score": meta.get("object_score", ""),
                "top_gt_object_id": top.get("gt_object_id", ""),
                "max_iou": ious[0] if ious else 0.0,
                "second_iou": ious[1] if len(ious) > 1 else 0.0,
                "gt_count_iou_ge_0p10": sum(value >= 0.10 for value in ious),
                "gt_count_iou_ge_0p25": sum(value >= 0.25 for value in ious),
                "gt_count_iou_ge_0p50": sum(value >= 0.50 for value in ious),
                "overmerge_proxy_multi_gt_ge_0p10": sum(value >= 0.10 for value in ious) >= 2,
                "extent_miss_proxy_max_iou_lt_0p25": (ious[0] if ious else 0.0) < 0.25,
                **stats,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-root", default="outputs/audit/v87_phase2_mv_tube_materializer")
    parser.add_argument("--phase3-root", default="outputs/audit/v87_phase3_mv_ap_evaluator")
    parser.add_argument("--phase4-root", default="outputs/audit/v87_phase4_dev_mv_ap")
    parser.add_argument("--phase8-root", default="outputs/audit/v87_phase8_casebook")
    parser.add_argument("--output-root", default="outputs/audit/v87_phase9_failure_decomposition")
    parser.add_argument("--top-k", type=int, default=25)
    args = parser.parse_args()

    phase2 = _repo_path(args.phase2_root)
    phase3 = _repo_path(args.phase3_root)
    phase4 = _repo_path(args.phase4_root)
    phase8 = _repo_path(args.phase8_root)
    out = _repo_path(args.output_root)

    metrics = _read_csv_rows(phase3 / "mv_metric_rows.csv")
    iou_rows = _read_csv_rows(phase3 / "mv_iou_matrix_rows.csv")
    frame_rows = _read_csv_rows(phase2 / "mv_object_frame_mask_rows.csv")
    object_rows = _read_csv_rows(phase2 / "mv_object_rows.csv")
    phase4_summary = json.loads((phase4 / "dev_mv_summary.json").read_text(encoding="utf-8"))
    final_decision = json.loads((phase8 / "final_decision.json").read_text(encoding="utf-8"))

    metric_by_scene_variant = {
        (str(row.get("scene_id", "")), str(row.get("variant", ""))): row
        for row in metrics
        if row.get("split") == "dev"
    }
    local_by_scene = {
        scene: row
        for (scene, variant), row in metric_by_scene_variant.items()
        if variant == "B0_local_only"
    }
    controls_by_scene: dict[str, list[dict[str, str]]] = defaultdict(list)
    for (scene, variant), row in metric_by_scene_variant.items():
        if not _is_real_variant(variant):
            controls_by_scene[scene].append(row)
    best_control_by_scene = {
        scene: max(rows, key=lambda row: _num(row.get("MV_AP50"), 0.0))
        for scene, rows in controls_by_scene.items()
    }

    variant_summary_rows: list[dict[str, Any]] = []
    for (scene, variant), row in sorted(metric_by_scene_variant.items()):
        local = local_by_scene.get(scene, {})
        best_control = best_control_by_scene.get(scene, {})
        diagnosis: list[str] = []
        if variant == "B0_local_only":
            diagnosis.append("local_baseline")
        elif _is_real_variant(variant):
            if _num(row.get("MV_AP50"), 0.0) < _num(local.get("MV_AP50"), 0.0):
                diagnosis.append("history_materialization_harmful_vs_local")
            if _num(row.get("MV_AP50"), 0.0) <= _num(best_control.get("MV_AP50"), 0.0):
                diagnosis.append(f"control_explains_or_beats_real:{best_control.get('variant','')}")
            if _num(row.get("MV_SF50"), 0.0) < _num(local.get("MV_SF50"), 0.0):
                diagnosis.append("score_free_candidate_universe_deficit_vs_local")
            elif _num(row.get("MV_AP50"), 0.0) < _num(local.get("MV_AP50"), 0.0):
                diagnosis.append("ranking_or_score_blocker_possible")
            if _num(row.get("MV_AP25"), 0.0) > _num(local.get("MV_AP25"), 0.0) and _num(row.get("MV_AP50"), 0.0) < _num(local.get("MV_AP50"), 0.0):
                diagnosis.append("ap25_only_gain_objects_too_coarse_or_scores_poor")
        else:
            diagnosis.append(_variant_role(variant))
        variant_summary_rows.append(
            {
                "scene_id": scene,
                "variant": variant,
                "variant_role": _variant_role(variant),
                "MV_AP": row.get("MV_AP", ""),
                "MV_AP50": row.get("MV_AP50", ""),
                "MV_AP25": row.get("MV_AP25", ""),
                "MV_SF50": row.get("MV_SF50", ""),
                "MV_SF25": row.get("MV_SF25", ""),
                "pred_object_count": row.get("pred_object_count", ""),
                "gt_object_count": row.get("gt_object_count", ""),
                "B0_MV_AP50": local.get("MV_AP50", ""),
                "B0_MV_SF50": local.get("MV_SF50", ""),
                "best_control_variant": best_control.get("variant", ""),
                "best_control_MV_AP50": best_control.get("MV_AP50", ""),
                "real_minus_local_AP50": _num(row.get("MV_AP50"), 0.0) - _num(local.get("MV_AP50"), 0.0),
                "real_minus_best_control_AP50": _num(row.get("MV_AP50"), 0.0) - _num(best_control.get("MV_AP50"), 0.0),
                "sf50_minus_local_sf50": _num(row.get("MV_SF50"), 0.0) - _num(local.get("MV_SF50"), 0.0),
                "diagnosis": ";".join(diagnosis),
            }
        )

    object_meta = {str(row.get("mv_object_id", "")): row for row in object_rows}
    frame_stat = _frame_stats(frame_rows)
    object_decomp_rows = _object_stats(iou_rows, object_meta, frame_stat)

    top_iou_rows = sorted(
        [
            {
                **row,
                "variant": _variant_from_object_id(str(row.get("mv_object_id", ""))),
                "variant_role": _variant_role(_variant_from_object_id(str(row.get("mv_object_id", "")))),
            }
            for row in iou_rows
            if str(row.get("scene_id", "")) in local_by_scene
        ],
        key=lambda row: _num(row.get("mv_iou"), 0.0),
        reverse=True,
    )[: args.top_k]

    iou_by_group: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in iou_rows:
        scene = str(row.get("scene_id", ""))
        variant = _variant_from_object_id(str(row.get("mv_object_id", "")))
        if scene in local_by_scene and (variant == "B0_local_only" or _is_real_variant(variant)):
            iou_by_group[(scene, variant)].append(row)

    score_rows: list[dict[str, Any]] = []
    for (scene, variant), rows in sorted(iou_by_group.items()):
        pred_ids = sorted({str(row.get("mv_object_id", "")) for row in rows})
        max_iou_by_pred = {
            pred_id: max([_num(row.get("mv_iou"), 0.0) for row in rows if str(row.get("mv_object_id", "")) == pred_id] or [0.0])
            for pred_id in pred_ids
        }
        current_scores = {pred_id: _num(object_meta.get(pred_id, {}).get("object_score"), 0.0) for pred_id in pred_ids}
        support_scores = {pred_id: frame_stat.get(pred_id, {}).get("frame_mask_count", 0.0) for pred_id in pred_ids}
        adapter_scores = {pred_id: frame_stat.get(pred_id, {}).get("mean_adapter_score", 0.0) for pred_id in pred_ids}
        area_penalty_scores = {pred_id: -math.log1p(frame_stat.get(pred_id, {}).get("mask_area_sum", 0.0)) for pred_id in pred_ids}
        native_support_scores = {pred_id: frame_stat.get(pred_id, {}).get("mean_native_support", 0.0) for pred_id in pred_ids}
        variants = [
            ("S0_current_object_score", current_scores, False),
            ("S1_support_frame_count_gt_free", support_scores, False),
            ("S2_mean_adapter_score_gt_free", adapter_scores, False),
            ("S3_area_penalty_gt_free", area_penalty_scores, False),
            ("S4_mean_native_support_gt_free", native_support_scores, False),
            ("ORACLE_max_iou_gt_diagnostic_only", max_iou_by_pred, True),
        ]
        for score_name, scores, uses_gt in variants:
            score_rows.append(
                {
                    "scene_id": scene,
                    "variant": variant,
                    "score_variant": score_name,
                    "diagnostic_uses_gt_for_score": uses_gt,
                    "MV_AP": _ap_mean(rows, scores),
                    "MV_AP50": _ap_from_iou_rows(rows, scores, 0.50),
                    "MV_AP25": _ap_from_iou_rows(rows, scores, 0.25),
                    "B0_MV_AP50": local_by_scene.get(scene, {}).get("MV_AP50", ""),
                    "B0_MV_SF50": local_by_scene.get(scene, {}).get("MV_SF50", ""),
                    "pred_object_count": len(pred_ids),
                    "gt_object_count": len({str(row.get("gt_object_id", "")) for row in rows}),
                }
            )

    real_rows = [
        row
        for row in variant_summary_rows
        if row["variant_role"] == "history_readout"
    ]
    local_beats_all_history = all(_num(row.get("real_minus_local_AP50"), 0.0) < 0.0 for row in real_rows)
    controls_explain_all_history = all(_num(row.get("real_minus_best_control_AP50"), 0.0) <= 0.0 for row in real_rows)
    score_free_deficit_all_history = all(_num(row.get("sf50_minus_local_sf50"), 0.0) < 0.0 for row in real_rows)
    oracle_rows = [row for row in score_rows if row["score_variant"] == "ORACLE_max_iou_gt_diagnostic_only" and row["variant_role" if "variant_role" in row else "variant"]]
    best_history_oracle_ap50 = max(
        [
            _num(row.get("MV_AP50"), 0.0)
            for row in score_rows
            if _is_real_variant(str(row.get("variant", ""))) and row.get("variant") != "B0_local_only" and row.get("score_variant") == "ORACLE_max_iou_gt_diagnostic_only"
        ]
        or [0.0]
    )
    best_local_ap50 = max([_num(row.get("MV_AP50"), 0.0) for row in local_by_scene.values()] or [0.0])

    summary = {
        "schema": "stream4d_v87_phase9_failure_decomposition_v1",
        "phase": "v87_phase9_failure_decomposition",
        "source_final_decision": final_decision.get("final_decision", ""),
        "source_phase4_decision": phase4_summary.get("decision", ""),
        "local_beats_all_history_variants_on_dev_AP50": local_beats_all_history,
        "controls_explain_or_beat_all_history_variants_on_dev_AP50": controls_explain_all_history,
        "score_free_sf50_deficit_all_history_vs_local": score_free_deficit_all_history,
        "best_history_oracle_AP50_diagnostic_only": best_history_oracle_ap50,
        "best_local_current_AP50": best_local_ap50,
        "score_calibration_likely_sufficient": best_history_oracle_ap50 >= best_local_ap50 + 0.03,
        "primary_failure_attribution": (
            "history_materialization_candidate_universe_deficit_and_area_control_explanation"
            if local_beats_all_history and controls_explain_all_history and score_free_deficit_all_history
            else "mixed_failure_needs_case_review"
        ),
        "recommended_next_repair": (
            "try_gt_free_id_only_local_fallback_materializer_before_score_tuning"
            if local_beats_all_history and score_free_deficit_all_history
            else "inspect_score_calibration_or_casebook"
        ),
    }

    _write_csv(out / "variant_failure_summary_rows.csv", variant_summary_rows)
    _write_csv(out / "object_absorption_rows.csv", object_decomp_rows)
    _write_csv(out / "top_iou_case_rows.csv", top_iou_rows)
    _write_csv(out / "score_oracle_and_gtfree_ranking_rows.csv", score_rows)
    _write_json(out / "failure_decomposition_summary.json", summary)
    print(json.dumps({"decision": summary["primary_failure_attribution"], "phase": "phase9"}, sort_keys=True))


if __name__ == "__main__":
    main()
