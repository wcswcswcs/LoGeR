from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/audit/v91_scene0011_support_extent_failure"
M1_ROOT = ROOT / "outputs/audit/v91_phase4_witness_cover_multimask_materialization"
RADIUS_ROOT = ROOT / "outputs/audit/v91_phase4_witness_cover_radius_sweep"
PHASE8_ROOT = ROOT / "outputs/audit/v91_phase8_dev_selection"
M1_VARIANT = "V91_M1_W8a_top2_r16_drop5_sceneorig"
R16_VARIANT = "V91_P4R3_W9a_carve_r16_drop5_sceneorig"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(_jsonable(row))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def _median(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.median(vals)) if vals else 0.0


def _max(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.max(vals)) if vals else 0.0


def _metric_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, family in [
        (RADIUS_ROOT / "mv_metric_rows.csv", "radius_r16_reference"),
        (M1_ROOT / "mv_metric_rows.csv", "multimask_materialization"),
    ]:
        for row in _read_csv(path):
            variant = row.get("variant_id", row.get("variant", ""))
            if variant not in {R16_VARIANT, M1_VARIANT}:
                continue
            rows.append(
                {
                    "variant_id": variant,
                    "family": family,
                    "scene_id": row.get("scene_id", ""),
                    "MV_AP_window": _num(row.get("MV_AP_window", row.get("MV_AP"))),
                    "MV_AP50_window": _num(row.get("MV_AP50_window", row.get("MV_AP50"))),
                    "MV_AP25_window": _num(row.get("MV_AP25_window", row.get("MV_AP25"))),
                    "score_free_Match50_window": _num(row.get("score_free_Match50_window")),
                    "SF50_precision": _num(row.get("score_free_Match50_precision_window", row.get("SF50_precision"))),
                    "SF50_recall": _num(row.get("score_free_Match50_recall_window", row.get("SF50_recall"))),
                    "gt_best_iou_mean": _num(row.get("gt_best_iou_mean")),
                    "gt_recall_best_iou_ge_050": _num(row.get("gt_recall_best_iou_ge_050")),
                    "pred_object_count": _num(row.get("pred_object_count")),
                    "gt_object_count": _num(row.get("gt_object_count")),
                    "frame_mask_row_count": _num(row.get("frame_mask_row_count")),
                    "unique_frame_mask_count": _num(row.get("unique_frame_mask_count")),
                    "missing_mask_raster_count": _int(row.get("missing_mask_raster_count")),
                    "same_frame_collision_count": _int(row.get("same_frame_collision_count")),
                }
            )
    return rows


def _scene_delta_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["variant_id"], row["scene_id"]): row for row in metric_rows}
    rows: list[dict[str, Any]] = []
    for scene in sorted({row["scene_id"] for row in metric_rows}):
        r16 = by_key.get((R16_VARIANT, scene), {})
        m1 = by_key.get((M1_VARIANT, scene), {})
        if not r16 or not m1:
            continue
        rows.append(
            {
                "scene_id": scene,
                "M1_minus_R16_MV_AP_window": _num(m1.get("MV_AP_window")) - _num(r16.get("MV_AP_window")),
                "M1_minus_R16_MV_AP50_window": _num(m1.get("MV_AP50_window")) - _num(r16.get("MV_AP50_window")),
                "M1_minus_R16_MV_AP25_window": _num(m1.get("MV_AP25_window")) - _num(r16.get("MV_AP25_window")),
                "M1_minus_R16_score_free_Match50": _num(m1.get("score_free_Match50_window")) - _num(r16.get("score_free_Match50_window")),
                "M1_MV_AP50_window": _num(m1.get("MV_AP50_window")),
                "R16_MV_AP50_window": _num(r16.get("MV_AP50_window")),
                "M1_score_free_Match50_window": _num(m1.get("score_free_Match50_window")),
                "R16_score_free_Match50_window": _num(r16.get("score_free_Match50_window")),
            }
        )
    return rows


def _support_summary(path: Path, variant_id: str, family: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(path):
        if row.get("variant_id") != variant_id:
            continue
        groups[(row.get("scene_id", ""), row.get("window_id", ""))].append(row)
    rows: list[dict[str, Any]] = []
    for (scene, window), group in sorted(groups.items()):
        rows.append(
            {
                "family": family,
                "variant_id": variant_id,
                "scene_id": scene,
                "window_id": window,
                "row_count": len(group),
                "unique_objects": len({row.get("mv_object_id", "") for row in group}),
                "unique_frames": len({_int(row.get("frame_id"), -1) for row in group}),
                "support_carrier_count_mean": _mean([_num(row.get("support_carrier_count")) for row in group]),
                "support_carrier_count_median": _median([_num(row.get("support_carrier_count")) for row in group]),
                "support_heatmap_area_mean": _mean([_num(row.get("support_heatmap_area")) for row in group]),
                "selected_mask_area_mean": _mean([_num(row.get("selected_mask_area")) for row in group]),
                "generated_mask_area_mean": _mean([_num(row.get("generated_mask_area")) for row in group]),
                "support_to_mask_ratio_mean": _mean([_num(row.get("support_to_mask_ratio")) for row in group]),
                "mask_to_support_ratio_mean": _mean([_num(row.get("mask_to_support_ratio")) for row in group]),
                "support_coverage_mean": _mean([_num(row.get("support_coverage")) for row in group]),
                "support_density_mean": _mean([_num(row.get("support_density")) for row in group]),
                "hard_negative_density_mean": _mean([_num(row.get("hard_negative_density")) for row in group]),
                "broad_risk_rate": _mean([1.0 if _bool(row.get("broad_risk")) else 0.0 for row in group]),
                "area_ratio_mean": _mean([_num(row.get("area_ratio")) for row in group]),
                "area_ratio_max": _max([_num(row.get("area_ratio")) for row in group]),
            }
        )
    scene_rows: list[dict[str, Any]] = []
    scene_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scene_groups[row["scene_id"]].append(row)
    for scene, group in sorted(scene_groups.items()):
        total = sum(_num(row.get("row_count")) for row in group)
        scene_rows.append(
            {
                "family": family,
                "variant_id": variant_id,
                "scene_id": scene,
                "window_id": "__SCENE__",
                "row_count": int(total),
                "window_count": len(group),
                "support_carrier_count_mean": _mean([_num(row.get("support_carrier_count_mean")) for row in group]),
                "support_heatmap_area_mean": _mean([_num(row.get("support_heatmap_area_mean")) for row in group]),
                "generated_mask_area_mean": _mean([_num(row.get("generated_mask_area_mean")) for row in group]),
                "support_to_mask_ratio_mean": _mean([_num(row.get("support_to_mask_ratio_mean")) for row in group]),
                "mask_to_support_ratio_mean": _mean([_num(row.get("mask_to_support_ratio_mean")) for row in group]),
                "support_coverage_mean": _mean([_num(row.get("support_coverage_mean")) for row in group]),
                "support_density_mean": _mean([_num(row.get("support_density_mean")) for row in group]),
                "hard_negative_density_mean": _mean([_num(row.get("hard_negative_density_mean")) for row in group]),
                "broad_risk_rate": _mean([_num(row.get("broad_risk_rate")) for row in group]),
                "area_ratio_mean": _mean([_num(row.get("area_ratio_mean")) for row in group]),
                "area_ratio_max": _max([_num(row.get("area_ratio_max")) for row in group]),
            }
        )
    return scene_rows + rows


def _source_conflict_rows() -> list[dict[str, Any]]:
    selected = [row for row in _read_csv(M1_ROOT / "source_selection_rows.csv") if row.get("variant_id") == f"{M1_VARIANT}_source"]
    dropped = [row for row in _read_csv(M1_ROOT / "dropped_source_rows.csv") if row.get("variant_id") == f"{M1_VARIANT}_source"]
    out: list[dict[str, Any]] = []
    for scene in sorted({row.get("scene_id", "") for row in selected + dropped}):
        sel = [row for row in selected if row.get("scene_id") == scene]
        drp = [row for row in dropped if row.get("scene_id") == scene]
        frames = {_int(row.get("frame_id"), -1) for row in sel}
        out.append(
            {
                "scene_id": scene,
                "source_selected_rows": len(sel),
                "source_dropped_rows": len(drp),
                "source_drop_per_selected": len(drp) / max(1, len(sel)),
                "unique_selected_frames": len(frames),
                "selected_rows_per_frame": len(sel) / max(1, len(frames)),
                "selected_broad_risk_rate": _mean([1.0 if _bool(row.get("broad_background_risk")) else 0.0 for row in sel]),
                "selected_area_ratio_mean": _mean([_num(row.get("area_ratio")) for row in sel]),
                "selected_area_ratio_max": _max([_num(row.get("area_ratio")) for row in sel]),
                "selected_support_count_mean": _mean([_num(row.get("support_count")) for row in sel]),
                "selected_support_count_median": _median([_num(row.get("support_count")) for row in sel]),
                "dropped_broad_risk_rate": _mean([1.0 if _bool(row.get("broad_background_risk")) else 0.0 for row in drp]),
                "dropped_area_ratio_mean": _mean([_num(row.get("area_ratio")) for row in drp]),
                "dropped_support_count_mean": _mean([_num(row.get("support_count")) for row in drp]),
                "uses_gt_for_prediction": any(_bool(row.get("uses_gt_for_prediction")) for row in sel + drp),
                "uses_future": any(_bool(row.get("uses_future")) for row in sel + drp),
            }
        )
    return out


def run() -> dict[str, Any]:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    phase8 = json.loads((PHASE8_ROOT / "summary.json").read_text(encoding="utf-8"))
    metric_rows = _metric_rows()
    scene_delta_rows = _scene_delta_rows(metric_rows)
    support_rows = []
    support_rows.extend(_support_summary(RADIUS_ROOT / "support_quality_rows.csv", R16_VARIANT, "radius_r16_reference"))
    support_rows.extend(_support_summary(M1_ROOT / "support_quality_rows.csv", M1_VARIANT, "multimask_materialization"))
    source_conflict_rows = _source_conflict_rows()
    by_delta_scene = {row["scene_id"]: row for row in scene_delta_rows}
    scene0011 = by_delta_scene.get("scene0011_00", {})
    scene0050 = by_delta_scene.get("scene0050_00", {})
    scene_support = {(row["variant_id"], row["scene_id"], row["window_id"]): row for row in support_rows}
    m1_s11_support = scene_support.get((M1_VARIANT, "scene0011_00", "__SCENE__"), {})
    m1_s50_support = scene_support.get((M1_VARIANT, "scene0050_00", "__SCENE__"), {})
    source_by_scene = {row["scene_id"]: row for row in source_conflict_rows}
    s11_source = source_by_scene.get("scene0011_00", {})
    s50_source = source_by_scene.get("scene0050_00", {})
    diagnosis_rows = [
        {
            "diagnosis": "scene0011_dominates_failure",
            "evidence": "M1 scene0011 AP50 and score-free Match50 are far below scene0050",
            "scene0011_MV_AP50": scene0011.get("M1_MV_AP50_window", ""),
            "scene0050_MV_AP50": scene0050.get("M1_MV_AP50_window", ""),
            "scene0011_score_free_Match50": scene0011.get("M1_score_free_Match50_window", ""),
            "scene0050_score_free_Match50": scene0050.get("M1_score_free_Match50_window", ""),
        },
        {
            "diagnosis": "multimask_does_not_fix_scene0011",
            "evidence": "M1 AP50 delta vs r16 is negative on scene0011 while aggregate best improves through scene0050",
            "scene0011_M1_minus_R16_AP50": scene0011.get("M1_minus_R16_MV_AP50_window", ""),
            "scene0050_M1_minus_R16_AP50": scene0050.get("M1_minus_R16_MV_AP50_window", ""),
        },
        {
            "diagnosis": "gt_free_support_profile",
            "evidence": "support/source statistics are GT-free and identify whether scene0011 has broader or less reliable selected masks",
            "scene0011_broad_risk_rate": m1_s11_support.get("broad_risk_rate", ""),
            "scene0050_broad_risk_rate": m1_s50_support.get("broad_risk_rate", ""),
            "scene0011_support_coverage": m1_s11_support.get("support_coverage_mean", ""),
            "scene0050_support_coverage": m1_s50_support.get("support_coverage_mean", ""),
            "scene0011_source_drop_per_selected": s11_source.get("source_drop_per_selected", ""),
            "scene0050_source_drop_per_selected": s50_source.get("source_drop_per_selected", ""),
        },
    ]
    summary = {
        "phase": "v91_scene0011_support_extent_failure",
        "schema": "stream4d_v91_scene0011_support_extent_failure_v1",
        "phase8_best_variant": phase8.get("best_real_variant", ""),
        "phase8_dev_progress_gate_pass": phase8.get("dev_progress_gate_pass"),
        "phase8_failed_gate": "best_real_MV_AP50_window_ge_control_plus_0p010",
        "M1_scene0011_MV_AP50_window": scene0011.get("M1_MV_AP50_window", 0.0),
        "M1_scene0050_MV_AP50_window": scene0050.get("M1_MV_AP50_window", 0.0),
        "M1_scene0011_score_free_Match50_window": scene0011.get("M1_score_free_Match50_window", 0.0),
        "M1_scene0050_score_free_Match50_window": scene0050.get("M1_score_free_Match50_window", 0.0),
        "M1_scene0011_minus_R16_AP50": scene0011.get("M1_minus_R16_MV_AP50_window", 0.0),
        "M1_scene0050_minus_R16_AP50": scene0050.get("M1_minus_R16_MV_AP50_window", 0.0),
        "M1_scene0011_broad_risk_rate": m1_s11_support.get("broad_risk_rate", 0.0),
        "M1_scene0050_broad_risk_rate": m1_s50_support.get("broad_risk_rate", 0.0),
        "M1_scene0011_support_coverage_mean": m1_s11_support.get("support_coverage_mean", 0.0),
        "M1_scene0050_support_coverage_mean": m1_s50_support.get("support_coverage_mean", 0.0),
        "M1_scene0011_source_drop_per_selected": s11_source.get("source_drop_per_selected", 0.0),
        "M1_scene0050_source_drop_per_selected": s50_source.get("source_drop_per_selected", 0.0),
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "decision": "DIAGNOSED_SCENE0011_EXTENT_SUPPORT_FAILURE_CONTINUE_DEV_REPAIR",
        "next_action": "try GT-free scene-risk-conditioned materialization only if condition derives from source/support statistics; do not run holdout",
        "runtime_sec": time.time() - started,
    }
    _write_csv(OUT / "scene_metric_rows.csv", metric_rows)
    _write_csv(OUT / "scene_delta_rows.csv", scene_delta_rows)
    _write_csv(OUT / "support_scene_window_summary_rows.csv", support_rows)
    _write_csv(OUT / "source_conflict_scene_rows.csv", source_conflict_rows)
    _write_csv(OUT / "diagnosis_rows.csv", diagnosis_rows)
    _write_json(OUT / "summary.json", summary)
    outputs = [
        OUT / "scene_metric_rows.csv",
        OUT / "scene_delta_rows.csv",
        OUT / "support_scene_window_summary_rows.csv",
        OUT / "source_conflict_scene_rows.csv",
        OUT / "diagnosis_rows.csv",
        OUT / "summary.json",
    ]
    _write_json(OUT / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run()
