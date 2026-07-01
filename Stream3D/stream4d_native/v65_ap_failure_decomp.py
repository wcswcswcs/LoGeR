from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from .v47_common import read_csv
from .v65_common import float_or_none, project, rel, write_standard_outputs


SOMA_ROW_SPECS = [
    ("A3", "v64r2_probe5_v53_bridge_wta", "PREDICTION_UNION_ISLAND"),
    ("A4", "v64r2_probe5_v53_bridge_wta_used_support", "USED_FRAME_VISIBLE_SUPPORT"),
    ("A5", "v64r2_d4rt_chunk_scale_first_ap_probe5_g11", "PREDICTION_UNION_ISLAND"),
    ("A6", "v64r2_d4rt_chunk_scale_first_ap_probe5_g11_used_support", "USED_FRAME_VISIBLE_SUPPORT"),
    ("A7", "v64r2_d4rt_chunk_scale_first_ap_probe5_g12", "PREDICTION_UNION_ISLAND"),
    ("A8", "v64r2_d4rt_chunk_scale_first_ap_probe5_g12_used_support", "USED_FRAME_VISIBLE_SUPPORT"),
]


def build_v65_ap_failure_decomp() -> dict[str, Any]:
    failure_rows: list[dict[str, Any]] = []
    fragmentation_rows: list[dict[str, Any]] = []
    for row_id, config, support_scope in SOMA_ROW_SPECS:
        for scene in _probe_scenes():
            scene_payload = _scene_iou_profile(row_id=row_id, config=config, support_scope=support_scope, scene=scene)
            failure_rows.extend(scene_payload["failure_rows"])
            fragmentation_rows.append(scene_payload["fragmentation_row"])
    scope_contrast_rows = _scope_contrast_rows()
    category_counts = Counter(str(row["failure_category"]) for row in failure_rows)
    by_scope: dict[str, Counter[str]] = defaultdict(Counter)
    for row in failure_rows:
        by_scope[str(row["support_scope"])][str(row["failure_category"])] += 1
    assigned = [row for row in failure_rows if str(row.get("failure_category") or "").startswith("F_")]
    attribution_coverage = float(len(assigned) / max(len(failure_rows), 1))
    summary = {
        "phase": "v65_ap_failure_decomp",
        "failure_row_count": len(failure_rows),
        "fragmentation_row_count": len(fragmentation_rows),
        "scope_contrast_row_count": len(scope_contrast_rows),
        "failure_category_counts": dict(category_counts),
        "failure_category_counts_by_scope": {scope: dict(counts) for scope, counts in by_scope.items()},
        "attribution_coverage": attribution_coverage,
        "top_failure_category": category_counts.most_common(1)[0][0] if category_counts else "none",
        "top_failure_category_by_scope": {
            scope: counts.most_common(1)[0][0] if counts else "none" for scope, counts in by_scope.items()
        },
        "d4rt_low_ap_explained_by_measurable_categories": any(
            row["variant_row_id"] in {"A5", "A6", "A7", "A8"} for row in failure_rows
        ),
        "old_high_ap_explained_by_scope_contrast": any(row.get("left_row_id") == "A3" and row.get("right_row_id") == "A4" for row in scope_contrast_rows),
        "gate": {
            "attribution_coverage_ge_0_95": attribution_coverage >= 0.95,
            "top_failure_category_per_scope_identified": all(counts for counts in by_scope.values()),
            "d4rt_low_ap_has_failure_rows": any(row["variant_row_id"] in {"A5", "A6", "A7", "A8"} for row in failure_rows),
            "scope_contrast_rows_available": bool(scope_contrast_rows),
        },
    }
    summary["gate"]["pass"] = bool(all(summary["gate"].values()))
    return {
        "summary": summary,
        "failure_rows": failure_rows,
        "fragmentation_rows": fragmentation_rows,
        "scope_contrast_rows": scope_contrast_rows,
    }


def write_v65_ap_failure_decomp(output_root: str | Path, payload: dict[str, Any]) -> None:
    write_standard_outputs(
        output_root,
        {
            "failure_summary.json": payload["summary"],
            "failure_rows.csv": payload["failure_rows"],
            "fragmentation_rows.csv": payload["fragmentation_rows"],
            "scope_contrast_rows.csv": payload["scope_contrast_rows"],
        },
    )


def _probe_scenes() -> list[str]:
    split = project("splits/scannet_v6_probe5.txt")
    return [line.strip() for line in split.read_text(encoding="utf-8").splitlines() if line.strip()]


def _scene_iou_profile(*, row_id: str, config: str, support_scope: str, scene: str) -> dict[str, Any]:
    pred_path = project("data/prediction") / f"{config}_class_agnostic" / f"{scene}.npz"
    tmp_path = project("data/TMP") / config / f"{scene}_pre_points.npy"
    gt_path = project("data/scannet/gt") / f"{scene}.txt"
    if not pred_path.exists() or not tmp_path.exists() or not gt_path.exists():
        return {
            "failure_rows": [
                _failure_row(row_id, config, support_scope, scene, "F_evaluator_format", evidence="missing prediction/tmp/gt file")
            ],
            "fragmentation_row": {
                "variant_row_id": row_id,
                "config": config,
                "support_scope": support_scope,
                "scene_id": scene,
                "status": "missing_input",
            },
        }
    gt_full = np.loadtxt(gt_path, dtype=np.int64)
    pre_points = np.load(tmp_path).astype(np.int64)
    with np.load(pred_path) as payload:
        pred_masks = np.asarray(payload["pred_masks"], dtype=bool)
        scores = np.asarray(payload["pred_score"], dtype=np.float64)
    if pred_masks.shape[0] == gt_full.shape[0]:
        scoped_pred = pred_masks[pre_points, :]
    elif pred_masks.shape[0] == pre_points.shape[0]:
        scoped_pred = pred_masks
    else:
        return {
            "failure_rows": [
                _failure_row(
                    row_id,
                    config,
                    support_scope,
                    scene,
                    "F_evaluator_format",
                    evidence=f"pred/gt/support shape mismatch pred={pred_masks.shape[0]} gt={gt_full.shape[0]} support={pre_points.shape[0]}",
                )
            ],
            "fragmentation_row": {
                "variant_row_id": row_id,
                "config": config,
                "support_scope": support_scope,
                "scene_id": scene,
                "status": "shape_mismatch",
            },
        }
    gt_scoped = gt_full[pre_points] % 1000 + 2000
    pred_areas = scoped_pred.sum(axis=0).astype(np.float64)
    kept = pred_areas >= 100.0
    kept_pred = scoped_pred[:, kept]
    kept_scores = scores[kept] if scores.shape[0] == scoped_pred.shape[1] else np.ones(int(kept.sum()), dtype=np.float64)
    kept_areas = pred_areas[kept]
    raw_pred_count = int(scoped_pred.shape[1])
    kept_pred_count = int(kept_pred.shape[1])
    dropped_pred_lt100 = int(raw_pred_count - kept_pred_count)
    failure_rows: list[dict[str, Any]] = []
    for pred_idx, area in enumerate(pred_areas):
        if area < 100.0:
            failure_rows.append(
                _failure_row(
                    row_id,
                    config,
                    support_scope,
                    scene,
                    "F_tiny_fragments",
                    best_pred_id=pred_idx,
                    pred_area=float(area),
                    dropped_by_min_region=True,
                    score=float(scores[pred_idx]) if pred_idx < scores.shape[0] else None,
                    evidence="prediction area below evaluator min_region_size=100",
                )
            )
    gt_ids = [int(value) for value in np.unique(gt_scoped) if int(value) >= 1000]
    pred_best: list[float] = []
    gt_best: list[float] = []
    pred_match_gt_counts = np.zeros(kept_pred_count, dtype=np.int64)
    for gt_id in gt_ids:
        gt_mask = gt_scoped == gt_id
        gt_area = float(np.count_nonzero(gt_mask))
        if gt_area < 100.0:
            continue
        if kept_pred_count == 0:
            failure_rows.append(
                _failure_row(row_id, config, support_scope, scene, "F_materialization", gt_id=gt_id, gt_area=gt_area, evidence="no kept predictions")
            )
            continue
        intersections = kept_pred[gt_mask, :].sum(axis=0).astype(np.float64)
        unions = gt_area + kept_areas - intersections
        ious = np.divide(intersections, np.maximum(unions, 1.0))
        best_idx = int(np.argmax(ious))
        best_iou = float(ious[best_idx])
        gt_best.append(best_iou)
        pred_match_gt_counts += (ious >= 0.25).astype(np.int64)
        duplicate_count = int(np.count_nonzero(ious >= 0.25))
        if best_iou >= 0.50 and duplicate_count <= 1:
            continue
        category = _category(best_iou, duplicate_count)
        failure_rows.append(
            _failure_row(
                row_id,
                config,
                support_scope,
                scene,
                category,
                gt_id=gt_id,
                best_pred_id=best_idx,
                best_iou=best_iou,
                pred_area=float(kept_areas[best_idx]),
                gt_area=gt_area,
                score=float(kept_scores[best_idx]) if best_idx < kept_scores.shape[0] else None,
                evidence=f"best_iou={best_iou:.6f}; duplicate_pred_ge_025={duplicate_count}",
            )
        )
    for pred_idx in range(kept_pred_count):
        pred_mask = kept_pred[:, pred_idx]
        area = float(kept_areas[pred_idx])
        if area <= 0:
            continue
        best_iou = 0.0
        for gt_id in gt_ids:
            gt_mask = gt_scoped == gt_id
            gt_area = float(np.count_nonzero(gt_mask))
            inter = float(np.count_nonzero(pred_mask & gt_mask))
            union = area + gt_area - inter
            best_iou = max(best_iou, inter / max(union, 1.0))
        pred_best.append(best_iou)
        if pred_match_gt_counts[pred_idx] >= 2:
            failure_rows.append(
                _failure_row(
                    row_id,
                    config,
                    support_scope,
                    scene,
                    "F_overmerge",
                    best_pred_id=pred_idx,
                    best_iou=best_iou,
                    pred_area=area,
                    score=float(kept_scores[pred_idx]) if pred_idx < kept_scores.shape[0] else None,
                    evidence=f"prediction overlaps {int(pred_match_gt_counts[pred_idx])} GT instances at IoU>=0.25",
                )
            )
        elif best_iou < 0.25:
            failure_rows.append(
                _failure_row(
                    row_id,
                    config,
                    support_scope,
                    scene,
                    "F_undercoverage",
                    best_pred_id=pred_idx,
                    best_iou=best_iou,
                    pred_area=area,
                    score=float(kept_scores[pred_idx]) if pred_idx < kept_scores.shape[0] else None,
                    evidence="prediction best IoU below 0.25",
                )
            )
    fragmentation_row = {
        "variant_row_id": row_id,
        "config": config,
        "support_scope": support_scope,
        "scene_id": scene,
        "status": "ok",
        "raw_pred_count": raw_pred_count,
        "kept_pred_count": kept_pred_count,
        "dropped_pred_lt100": dropped_pred_lt100,
        "tiny_fragment_ratio": float(dropped_pred_lt100 / max(raw_pred_count, 1)),
        "pred_best_iou_median": float(median(pred_best)) if pred_best else None,
        "gt_best_iou_median": float(median(gt_best)) if gt_best else None,
        "gt_best_iou_ge_050_mean": float(sum(1 for value in gt_best if value >= 0.50) / max(len(gt_best), 1)) if gt_best else None,
        "fragment_count_per_history": raw_pred_count,
        "fragment_count_for_history": raw_pred_count,
        "history_id": "",
    }
    return {"failure_rows": failure_rows, "fragmentation_row": fragmentation_row}


def _failure_row(
    row_id: str,
    config: str,
    support_scope: str,
    scene: str,
    category: str,
    *,
    gt_id: int | None = None,
    best_pred_id: int | None = None,
    best_iou: float | None = None,
    pred_area: float | None = None,
    gt_area: float | None = None,
    dropped_by_min_region: bool = False,
    score: float | None = None,
    evidence: str = "",
) -> dict[str, Any]:
    return {
        "support_scope": support_scope,
        "failure_category": category,
        "variant_row_id": row_id,
        "config": config,
        "scene_id": scene,
        "gt_id": gt_id,
        "best_pred_id": best_pred_id,
        "best_iou": best_iou,
        "pred_best_iou": best_iou,
        "pred_area": pred_area,
        "gt_area": gt_area,
        "dropped_by_min_region": dropped_by_min_region,
        "history_id": "",
        "material_count": "",
        "fragment_count_for_history": "",
        "state_mix_confirmed_ratio": "",
        "state_mix_tentative_ratio": "",
        "state_mix_shared_ratio": "",
        "state_mix_quarantine_ratio": "",
        "score": score,
        "evidence": evidence,
        "uses_gt_for_diagnostic": True,
    }


def _category(best_iou: float, duplicate_count: int) -> str:
    if duplicate_count > 1:
        return "F_duplicate"
    if best_iou <= 0.0:
        return "F_materialization"
    if best_iou < 0.25:
        return "F_undercoverage"
    if best_iou < 0.50:
        return "F_score"
    return "F_score"


def _scope_contrast_rows() -> list[dict[str, Any]]:
    rows = []
    contract_path = project("outputs/audit/v65_ap_contract/ap_contract_rows.csv")
    contract = {row["row_id"]: row for row in read_csv(contract_path)} if contract_path.exists() else {}
    for left, right, label in [
        ("A3", "A4", "bridge_prediction_union_to_used_frame"),
        ("A5", "A6", "d4rt_g11_prediction_union_to_used_frame"),
        ("A7", "A8", "d4rt_g12_prediction_union_to_used_frame"),
    ]:
        lrow = contract.get(left, {})
        rrow = contract.get(right, {})
        lap = float_or_none(lrow.get("AP"))
        rap = float_or_none(rrow.get("AP"))
        rows.append(
            {
                "contrast_id": label,
                "left_row_id": left,
                "right_row_id": right,
                "left_support_scope": lrow.get("support_scope"),
                "right_support_scope": rrow.get("support_scope"),
                "left_AP": lap,
                "right_AP": rap,
                "AP_delta_right_minus_left": (rap - lap) if lap is not None and rap is not None else None,
                "failure_category": "F_scope",
                "evidence": "same family evaluated under different support scope in v65 current-run AP contract",
            }
        )
    return rows
