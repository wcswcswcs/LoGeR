#!/usr/bin/env python3
"""Build the v99 Phase8 F2-anchored fusion matrix.

The script only reports AP for variants that are backed by existing evaluated
rows or by score-level combinations that can be re-evaluated from Phase2 rows.
Unavailable controls are recorded as unavailable rather than substituted with a
nearby method.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools import build_v99_phase1_f2_base_reproduction as p1  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase8_fusion_matrix"
PHASE0_DIR = AUDIT_ROOT / "v99_phase0_fact_lock"
PHASE2_DIR = AUDIT_ROOT / "v99_phase2_f2_strengthening"
PHASE4_DIR = AUDIT_ROOT / "v99_phase4_f2_da3_link_verifier"
PHASE5_DIR = AUDIT_ROOT / "v99_phase5_d4rt_anchor_verifier"
PHASE6_DIR = AUDIT_ROOT / "v99_phase6_dense_semantic_residual"
PHASE7_DIR = AUDIT_ROOT / "v99_phase7_broad_mask_split"


def _rel(path: Path | str) -> str:
    q = Path(path)
    try:
        return q.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return q.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in keys})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _bool_false(value: Any) -> bool:
    return str(value).strip().lower() in {"false", "0", "", "none"}


def _row_key(row: dict[str, Any]) -> tuple[str, str, int, int, str]:
    return (
        str(row.get("scene_id", "")),
        str(row.get("chunk_id", "")),
        int(float(row.get("frame_id", 0))),
        int(float(row.get("selected_mask_id", 0))),
        str(row.get("mv_object_id", "")),
    )


def _variant_rows(path: Path, variant_id: str) -> list[dict[str, Any]]:
    rows = [dict(row) for row in _read_csv(path) if row.get("variant_id") == variant_id]
    if not rows:
        raise RuntimeError(f"no rows for {variant_id} in {path}")
    return rows


def _metric_row(path: Path, variant_id: str) -> dict[str, Any]:
    rows = [dict(row) for row in _read_csv(path) if row.get("variant_id") == variant_id]
    if not rows:
        raise RuntimeError(f"no metric row for {variant_id} in {path}")
    if len(rows) != 1:
        raise RuntimeError(f"expected one metric row for {variant_id} in {path}, got {len(rows)}")
    return rows[0]


def _copy_metric_row(
    row: dict[str, Any],
    *,
    variant_id: str,
    role: str,
    method_family: str,
    source_variant_id: str,
    source_file: Path | str,
    evaluation_status: str = "evaluated",
    notes: str = "",
) -> dict[str, Any]:
    out = dict(row)
    out["schema_version"] = "stream4d_v99_phase8_metric_aggregate_v1"
    out["phase_id"] = "v99_phase8_fusion_matrix"
    out["variant_id"] = variant_id
    out["phase8_role"] = role
    out["method_family"] = method_family
    out["phase8_source_variant_id"] = source_variant_id
    out["phase8_source_file"] = _rel(source_file)
    out["evaluation_status"] = evaluation_status
    out["notes"] = notes
    return out


def _unavailable_metric_row(
    *,
    variant_id: str,
    role: str,
    method_family: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v99_phase8_metric_aggregate_v1",
        "phase_id": "v99_phase8_fusion_matrix",
        "variant_id": variant_id,
        "phase8_role": role,
        "method_family": method_family,
        "evaluation_status": "not_available_in_current_artifacts",
        "notes": notes,
    }


def _phase0_f2_base_metric_row() -> dict[str, Any]:
    rows = _read_csv(PHASE0_DIR / "baseline_metric_rows.csv")
    local = next(row for row in rows if row.get("row_id") == "F2_base_full_dev_MV_AP_window")
    scene = next(row for row in rows if row.get("row_id") == "F2_base_full_dev_MV_AP_scene")
    return {
        "schema_version": "stream4d_v99_phase8_metric_aggregate_v1",
        "phase_id": "v99_phase8_fusion_matrix",
        "variant_id": "B0_F2_base",
        "scene_count": local.get("scene_count", scene.get("scene_count", "")),
        "MV_AP_window": local.get("MV_AP_window", ""),
        "MV_AP50_window": local.get("MV_AP50_window", ""),
        "MV_AP25_window": local.get("MV_AP25_window", ""),
        "ScoreFreeMatch50_window": local.get("ScoreFreeMatch50_window", ""),
        "MV_AP_scene": scene.get("MV_AP_scene", ""),
        "MV_AP50_scene": scene.get("MV_AP50_scene", ""),
        "MV_AP25_scene": scene.get("MV_AP25_scene", ""),
        "ScoreFreeMatch50_scene": scene.get("ScoreFreeMatch50_scene", ""),
        "same_frame_collision_count": local.get("same_frame_collision_count", ""),
        "pixel_collision_rate": local.get("pixel_collision_rate", ""),
        "missing_mask_raster_count": local.get("missing_mask_raster_count", ""),
        "score_scope": "phase0_locked_artifact",
        "future_chunk_access_count": "",
        "causality_scope": "phase0_fact_lock",
        "method_chunk_size": 32,
        "frame_stride": 5,
        "uses_gt_for_prediction": local.get("uses_gt_for_prediction", "False"),
        "uses_future": local.get("uses_future", "False"),
        "phase8_role": "real",
        "method_family": "F2_base_locked",
        "phase8_source_variant_id": local.get("variant_id", ""),
        "phase8_source_file": local.get("source_file", ""),
        "evaluation_status": "metric_source_phase0_fact_lock",
        "notes": "B0 is the locked v98.1/Phase0 F2_base metric; Phase8 does not regenerate rows for it.",
    }


def _score_delta_by_key(parent_rows: list[dict[str, Any]], component_rows: list[dict[str, Any]]) -> dict[tuple[str, str, int, int, str], float]:
    parent_score = {_row_key(row): _num(row.get("score")) for row in parent_rows}
    out: dict[tuple[str, str, int, int, str], float] = {}
    for row in component_rows:
        key = _row_key(row)
        if key not in parent_score:
            raise RuntimeError(f"component row key missing from parent rows: {key}")
        out[key] = _num(row.get("score")) - parent_score[key]
    missing = set(parent_score) - set(out)
    if missing:
        sample = next(iter(missing))
        raise RuntimeError(f"component rows missing {len(missing)} parent keys; sample={sample}")
    return out


def _combined_rows(
    parent_rows: list[dict[str, Any]],
    *,
    variant_id: str,
    score_policy: str,
    deltas: list[dict[tuple[str, str, int, int, str], float]],
    component_names: list[str],
    role: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in parent_rows:
        key = _row_key(row)
        score = _num(row.get("score"))
        for delta in deltas:
            score += delta.get(key, 0.0)
        new = dict(row)
        new["variant_id"] = variant_id
        new["score"] = float(score)
        new["score_policy"] = score_policy
        new["phase8_parent_variant_id"] = row.get("variant_id")
        new["phase8_component_names"] = "+".join(component_names)
        new["phase8_role"] = role
        new["uses_gt_for_prediction"] = False
        new["uses_future"] = False
        rows.append(new)
    return rows


def _evaluate_rows(variant_id: str, rows: list[dict[str, Any]], scope: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_scene_rows, frame_rows = p1._evaluate_variant(variant_id, rows, scope)
    aggregate_rows = p1._aggregate_metrics(metric_scene_rows)
    if len(aggregate_rows) != 1:
        raise RuntimeError(f"expected one aggregate row for {variant_id}, got {len(aggregate_rows)}")
    return aggregate_rows[0], metric_scene_rows, frame_rows


def _control_best(rows: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    available = [row for row in rows if row.get("evaluation_status") != "not_available_in_current_artifacts" and row.get(metric) not in ("", None)]
    if not available:
        return None
    return max(available, key=lambda row: _num(row.get(metric)))


def _safety_pass(row: dict[str, Any]) -> bool:
    pixel = row.get("pixel_collision_rate")
    pixel_ok = True if pixel in ("", None) else _num(pixel, 1.0) <= 0.02
    return (
        int(_num(row.get("same_frame_collision_count"), 0.0)) == 0
        and int(_num(row.get("missing_mask_raster_count"), 0.0)) == 0
        and pixel_ok
        and _bool_false(row.get("uses_gt_for_prediction"))
        and _bool_false(row.get("uses_future"))
    )


def main() -> int:
    started = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase0 = json.loads((PHASE0_DIR / "summary.json").read_text(encoding="utf-8"))
    phase2_summary = json.loads((PHASE2_DIR / "best_variant_summary.json").read_text(encoding="utf-8"))
    phase4_summary = json.loads((PHASE4_DIR / "summary.json").read_text(encoding="utf-8"))
    phase5_summary = json.loads((PHASE5_DIR / "summary.json").read_text(encoding="utf-8"))
    phase6_summary = json.loads((PHASE6_DIR / "summary.json").read_text(encoding="utf-8"))
    phase7_summary = json.loads((PHASE7_DIR / "summary.json").read_text(encoding="utf-8"))

    parent_variant = str(phase2_summary["best_variant_id"])
    parent_rows = _variant_rows(PHASE2_DIR / "mv_object_frame_mask_rows.csv", parent_variant)
    scope = p1._load_source_scope()

    da3_variant = str(phase4_summary["best_real_variant"])
    d4rt_variant = str(phase5_summary["best_real_variant"])
    dense_variant = str(phase6_summary["best_real_variant"])
    broad_variant = str(phase7_summary["best_real_variant"])

    da3_rows = _variant_rows(PHASE4_DIR / "mv_object_frame_mask_rows.csv", da3_variant)
    d4rt_rows = _variant_rows(PHASE5_DIR / "mv_object_frame_mask_rows.csv", d4rt_variant)
    d4rt_shuffled_rows = _variant_rows(PHASE5_DIR / "mv_object_frame_mask_rows.csv", "P5_C1_shuffled_D4RT_anchor")
    d4rt_stale_rows = _variant_rows(PHASE5_DIR / "mv_object_frame_mask_rows.csv", "P5_C4_stale_D4RT_anchor")
    dense_rows = _variant_rows(PHASE6_DIR / "mv_object_frame_mask_rows.csv", dense_variant)
    dense_shuffled_rows = _variant_rows(PHASE6_DIR / "mv_object_frame_mask_rows.csv", "P6_C1_shuffled_dense_dino_boost_plus_veto")

    da3_delta = _score_delta_by_key(parent_rows, da3_rows)
    d4rt_delta = _score_delta_by_key(parent_rows, d4rt_rows)
    d4rt_shuffled_delta = _score_delta_by_key(parent_rows, d4rt_shuffled_rows)
    d4rt_stale_delta = _score_delta_by_key(parent_rows, d4rt_stale_rows)
    dense_delta = _score_delta_by_key(parent_rows, dense_rows)
    dense_shuffled_delta = _score_delta_by_key(parent_rows, dense_shuffled_rows)

    variant_config_rows = [
        {
            "schema_version": "stream4d_v99_phase8_variant_config_v1",
            "phase_id": "v99_phase8_fusion_matrix",
            "variant_id": "B0_F2_base",
            "role": "real",
            "method_family": "F2_base_locked",
            "source": _rel(PHASE0_DIR / "baseline_metric_rows.csv"),
            "evaluation_status": "metric_source_phase0_fact_lock",
        },
        {
            "schema_version": "stream4d_v99_phase8_variant_config_v1",
            "phase_id": "v99_phase8_fusion_matrix",
            "variant_id": "B1_F2_enhanced_best",
            "role": "real",
            "method_family": "F2_enhanced",
            "source_variant_id": parent_variant,
            "source": _rel(PHASE2_DIR / "mv_object_frame_mask_rows.csv"),
            "evaluation_status": "evaluated_phase2_artifact",
        },
        {
            "schema_version": "stream4d_v99_phase8_variant_config_v1",
            "phase_id": "v99_phase8_fusion_matrix",
            "variant_id": "B2_F2_plus_DA3_link_verifier",
            "role": "real",
            "method_family": "F2_plus_DA3_link",
            "source_variant_id": da3_variant,
            "source": _rel(PHASE4_DIR / "mv_object_frame_mask_rows.csv"),
            "evaluation_status": "evaluated_phase4_artifact",
        },
        {
            "schema_version": "stream4d_v99_phase8_variant_config_v1",
            "phase_id": "v99_phase8_fusion_matrix",
            "variant_id": "B3_F2_plus_D4RT_anchor_verifier",
            "role": "real",
            "method_family": "F2_plus_D4RT_anchor",
            "source_variant_id": d4rt_variant,
            "source": _rel(PHASE5_DIR / "mv_object_frame_mask_rows.csv"),
            "evaluation_status": "evaluated_phase5_artifact_but_control_failed",
        },
        {
            "schema_version": "stream4d_v99_phase8_variant_config_v1",
            "phase_id": "v99_phase8_fusion_matrix",
            "variant_id": "B4_F2_plus_dense_semantic_residual",
            "role": "real",
            "method_family": "F2_plus_dense_semantic",
            "source_variant_id": dense_variant,
            "source": _rel(PHASE6_DIR / "mv_object_frame_mask_rows.csv"),
            "evaluation_status": "evaluated_phase6_artifact_but_control_failed",
        },
        {
            "schema_version": "stream4d_v99_phase8_variant_config_v1",
            "phase_id": "v99_phase8_fusion_matrix",
            "variant_id": "B5_F2_plus_DA3_broad_split",
            "role": "real",
            "method_family": "F2_plus_DA3_broad_split",
            "source_variant_id": broad_variant,
            "source": _rel(PHASE7_DIR / "mv_object_frame_mask_rows.csv"),
            "evaluation_status": "evaluated_phase7_artifact_but_split_failed",
        },
        {
            "schema_version": "stream4d_v99_phase8_variant_config_v1",
            "phase_id": "v99_phase8_fusion_matrix",
            "variant_id": "B6_F2_plus_DA3_link_plus_D4RT_anchor",
            "role": "real",
            "method_family": "score_delta_combo",
            "source_variant_id": f"{da3_variant}+{d4rt_variant}",
            "source": "phase2_rows_plus_phase4_phase5_score_deltas",
            "evaluation_status": "evaluated_phase8_score_delta_combo",
        },
        {
            "schema_version": "stream4d_v99_phase8_variant_config_v1",
            "phase_id": "v99_phase8_fusion_matrix",
            "variant_id": "B7_F2_plus_DA3_link_plus_dense_semantic",
            "role": "real",
            "method_family": "score_delta_combo",
            "source_variant_id": f"{da3_variant}+{dense_variant}",
            "source": "phase2_rows_plus_phase4_phase6_score_deltas",
            "evaluation_status": "evaluated_phase8_score_delta_combo",
        },
        {
            "schema_version": "stream4d_v99_phase8_variant_config_v1",
            "phase_id": "v99_phase8_fusion_matrix",
            "variant_id": "B8_F2_plus_D4RT_anchor_plus_dense_semantic",
            "role": "real",
            "method_family": "score_delta_combo",
            "source_variant_id": f"{d4rt_variant}+{dense_variant}",
            "source": "phase2_rows_plus_phase5_phase6_score_deltas",
            "evaluation_status": "evaluated_phase8_score_delta_combo",
        },
        {
            "schema_version": "stream4d_v99_phase8_variant_config_v1",
            "phase_id": "v99_phase8_fusion_matrix",
            "variant_id": "B9_F2_plus_DA3_D4RT_dense_semantic_all",
            "role": "real",
            "method_family": "score_delta_combo",
            "source_variant_id": f"{da3_variant}+{d4rt_variant}+{dense_variant}",
            "source": "phase2_rows_plus_phase4_phase5_phase6_score_deltas",
            "evaluation_status": "evaluated_phase8_score_delta_combo",
        },
        {
            "schema_version": "stream4d_v99_phase8_variant_config_v1",
            "phase_id": "v99_phase8_fusion_matrix",
            "variant_id": "B10_F2_plus_DA3_D4RT_scene_stitch_only",
            "role": "real",
            "method_family": "scene_stitch_only",
            "source": "deferred_to_phase9_scene_local2history",
            "evaluation_status": "deferred_to_phase9",
        },
    ]

    combo_specs = [
        (
            "B6_F2_plus_DA3_link_plus_D4RT_anchor",
            "phase8_parent_plus_DA3_real_delta_plus_D4RT_real_delta",
            [da3_delta, d4rt_delta],
            [da3_variant, d4rt_variant],
            "real",
        ),
        (
            "B7_F2_plus_DA3_link_plus_dense_semantic",
            "phase8_parent_plus_DA3_real_delta_plus_dense_real_delta",
            [da3_delta, dense_delta],
            [da3_variant, dense_variant],
            "real",
        ),
        (
            "B8_F2_plus_D4RT_anchor_plus_dense_semantic",
            "phase8_parent_plus_D4RT_real_delta_plus_dense_real_delta",
            [d4rt_delta, dense_delta],
            [d4rt_variant, dense_variant],
            "real",
        ),
        (
            "B9_F2_plus_DA3_D4RT_dense_semantic_all",
            "phase8_parent_plus_DA3_real_delta_plus_D4RT_real_delta_plus_dense_real_delta",
            [da3_delta, d4rt_delta, dense_delta],
            [da3_variant, d4rt_variant, dense_variant],
            "real",
        ),
        (
            "C8_DA3_link_plus_shuffled_D4RT",
            "phase8_control_DA3_real_delta_plus_D4RT_shuffled_delta",
            [da3_delta, d4rt_shuffled_delta],
            [da3_variant, "P5_C1_shuffled_D4RT_anchor"],
            "control",
        ),
        (
            "C9_DA3_link_plus_shuffled_dense",
            "phase8_control_DA3_real_delta_plus_dense_shuffled_delta",
            [da3_delta, dense_shuffled_delta],
            [da3_variant, "P6_C1_shuffled_dense_dino_boost_plus_veto"],
            "control",
        ),
        (
            "C10_shuffled_D4RT_plus_shuffled_dense",
            "phase8_control_D4RT_shuffled_delta_plus_dense_shuffled_delta",
            [d4rt_shuffled_delta, dense_shuffled_delta],
            ["P5_C1_shuffled_D4RT_anchor", "P6_C1_shuffled_dense_dino_boost_plus_veto"],
            "control",
        ),
        (
            "C11_DA3_link_plus_stale_D4RT",
            "phase8_control_DA3_real_delta_plus_D4RT_stale_delta",
            [da3_delta, d4rt_stale_delta],
            [da3_variant, "P5_C4_stale_D4RT_anchor"],
            "control",
        ),
    ]

    combo_rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    metric_scene_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    combo_metric_rows: list[dict[str, Any]] = []
    combo_control_rows: list[dict[str, Any]] = []
    for variant_id, score_policy, deltas, component_names, role in combo_specs:
        rows = _combined_rows(
            parent_rows,
            variant_id=variant_id,
            score_policy=score_policy,
            deltas=deltas,
            component_names=component_names,
            role=role,
        )
        combo_rows_by_variant[variant_id] = rows
        agg, scene_rows, frames = _evaluate_rows(variant_id, rows, scope)
        copied = _copy_metric_row(
            agg,
            variant_id=variant_id,
            role=role,
            method_family="score_delta_combo",
            source_variant_id="+".join(component_names),
            source_file="phase8_recomputed_from_phase2_parent_rows",
            evaluation_status="evaluated_phase8_score_delta_combo",
        )
        if role == "control":
            combo_control_rows.append(copied)
        else:
            combo_metric_rows.append(copied)
        metric_scene_rows.extend(scene_rows)
        frame_rows.extend(frames)

    variant_metric_rows: list[dict[str, Any]] = [
        _phase0_f2_base_metric_row(),
        _copy_metric_row(
            _metric_row(PHASE2_DIR / "variant_metric_rows.csv", parent_variant),
            variant_id="B1_F2_enhanced_best",
            role="real",
            method_family="F2_enhanced",
            source_variant_id=parent_variant,
            source_file=PHASE2_DIR / "variant_metric_rows.csv",
            evaluation_status="evaluated_phase2_artifact",
        ),
        _copy_metric_row(
            _metric_row(PHASE4_DIR / "variant_metric_rows.csv", da3_variant),
            variant_id="B2_F2_plus_DA3_link_verifier",
            role="real",
            method_family="F2_plus_DA3_link",
            source_variant_id=da3_variant,
            source_file=PHASE4_DIR / "variant_metric_rows.csv",
            evaluation_status="evaluated_phase4_artifact",
        ),
        _copy_metric_row(
            _metric_row(PHASE5_DIR / "variant_metric_rows.csv", d4rt_variant),
            variant_id="B3_F2_plus_D4RT_anchor_verifier",
            role="real",
            method_family="F2_plus_D4RT_anchor",
            source_variant_id=d4rt_variant,
            source_file=PHASE5_DIR / "variant_metric_rows.csv",
            evaluation_status="evaluated_phase5_artifact_but_control_failed",
        ),
        _copy_metric_row(
            _metric_row(PHASE6_DIR / "variant_metric_rows.csv", dense_variant),
            variant_id="B4_F2_plus_dense_semantic_residual",
            role="real",
            method_family="F2_plus_dense_semantic",
            source_variant_id=dense_variant,
            source_file=PHASE6_DIR / "variant_metric_rows.csv",
            evaluation_status="evaluated_phase6_artifact_but_control_failed",
        ),
        _copy_metric_row(
            _metric_row(PHASE7_DIR / "variant_metric_rows.csv", broad_variant),
            variant_id="B5_F2_plus_DA3_broad_split",
            role="real",
            method_family="F2_plus_DA3_broad_split",
            source_variant_id=broad_variant,
            source_file=PHASE7_DIR / "variant_metric_rows.csv",
            evaluation_status="evaluated_phase7_artifact_but_split_failed",
        ),
    ]
    variant_metric_rows.extend(combo_metric_rows)
    variant_metric_rows.append(
        _unavailable_metric_row(
            variant_id="B10_F2_plus_DA3_D4RT_scene_stitch_only",
            role="real",
            method_family="scene_stitch_only",
            notes="Deferred to Phase9; no AP is reported in Phase8.",
        )
    )

    phase0_baselines = _read_csv(PHASE0_DIR / "baseline_metric_rows.csv")
    best_locked_control = next(row for row in phase0_baselines if row.get("row_id") == "best_locked_control_MV_AP_window")
    control_metric_rows: list[dict[str, Any]] = [
        _unavailable_metric_row(
            variant_id="C0_mask_only_frame_count",
            role="control",
            method_family="mask_only_frame_count",
            notes="No current v99 row artifact exists for pure mask-only frame-count control; not substituted with F2 semantic rows.",
        ),
        _copy_metric_row(
            {
                "MV_AP_window": best_locked_control.get("MV_AP_window", ""),
                "MV_AP50_window": best_locked_control.get("MV_AP50_window", ""),
                "MV_AP25_window": best_locked_control.get("MV_AP25_window", ""),
                "scene_count": best_locked_control.get("scene_count", ""),
                "same_frame_collision_count": best_locked_control.get("same_frame_collision_count", ""),
                "pixel_collision_rate": best_locked_control.get("pixel_collision_rate", ""),
                "missing_mask_raster_count": best_locked_control.get("missing_mask_raster_count", ""),
                "uses_gt_for_prediction": best_locked_control.get("uses_gt_for_prediction", "False"),
                "uses_future": best_locked_control.get("uses_future", "False"),
            },
            variant_id="C1_area_semantic_control",
            role="control",
            method_family="v98_locked_control",
            source_variant_id=best_locked_control.get("variant_id", ""),
            source_file=best_locked_control.get("source_file", ""),
            evaluation_status="metric_source_phase0_fact_lock",
        ),
        _unavailable_metric_row(
            variant_id="C2_DA3_only_geometry",
            role="control",
            method_family="DA3_only_geometry",
            notes="No DA3-only object-birth row artifact exists in v99; DA3 was tested as verifier only.",
        ),
        _copy_metric_row(
            _metric_row(PHASE5_DIR / "variant_metric_rows.csv", "P5_C1_shuffled_D4RT_anchor"),
            variant_id="C3_D4RT_shuffled_anchor",
            role="control",
            method_family="D4RT_shuffled_anchor",
            source_variant_id="P5_C1_shuffled_D4RT_anchor",
            source_file=PHASE5_DIR / "variant_metric_rows.csv",
            evaluation_status="evaluated_phase5_control",
        ),
        _copy_metric_row(
            _metric_row(PHASE6_DIR / "variant_metric_rows.csv", "P6_C1_shuffled_dense_dino_boost_plus_veto"),
            variant_id="C4_semantic_shuffled_residual",
            role="control",
            method_family="dense_semantic_shuffled",
            source_variant_id="P6_C1_shuffled_dense_dino_boost_plus_veto",
            source_file=PHASE6_DIR / "variant_metric_rows.csv",
            evaluation_status="evaluated_phase6_control",
        ),
        _unavailable_metric_row(
            variant_id="C5_dense_semantic_only",
            role="control",
            method_family="dense_semantic_only",
            notes="No dense-semantic-only object-birth row artifact exists; dense was tested as a Phase2 score verifier.",
        ),
        _copy_metric_row(
            _metric_row(PHASE5_DIR / "variant_metric_rows.csv", "P5_C4_stale_D4RT_anchor"),
            variant_id="C6_stale_D4RT_anchor",
            role="control",
            method_family="stale_D4RT_anchor",
            source_variant_id="P5_C4_stale_D4RT_anchor",
            source_file=PHASE5_DIR / "variant_metric_rows.csv",
            evaluation_status="evaluated_phase5_control",
        ),
        _unavailable_metric_row(
            variant_id="C7_random_DA3_link_with_same_degree",
            role="control",
            method_family="random_DA3_link",
            notes="Phase4 produced DA3 verifier/no-op/veto controls but not a same-degree random DA3 link control.",
        ),
        _copy_metric_row(
            _metric_row(PHASE2_DIR / "variant_metric_rows.csv", parent_variant),
            variant_id="C8_parent_F2_enhanced_no_aux",
            role="control",
            method_family="synchronized_no_aux_parent",
            source_variant_id=parent_variant,
            source_file=PHASE2_DIR / "variant_metric_rows.csv",
            evaluation_status="evaluated_phase2_artifact",
            notes="Extra synchronized parent control used for auxiliary module comparison.",
        ),
    ]
    control_metric_rows.extend(combo_control_rows)

    all_rows: list[dict[str, Any]] = []
    for rows in [
        _variant_rows(PHASE2_DIR / "mv_object_frame_mask_rows.csv", parent_variant),
        _variant_rows(PHASE4_DIR / "mv_object_frame_mask_rows.csv", da3_variant),
        _variant_rows(PHASE5_DIR / "mv_object_frame_mask_rows.csv", d4rt_variant),
        _variant_rows(PHASE6_DIR / "mv_object_frame_mask_rows.csv", dense_variant),
        _variant_rows(PHASE7_DIR / "mv_object_frame_mask_rows.csv", broad_variant),
    ]:
        all_rows.extend(rows)
    for rows in combo_rows_by_variant.values():
        all_rows.extend(rows)

    real_candidates = [
        row
        for row in variant_metric_rows
        if row.get("phase8_role") == "real"
        and row.get("evaluation_status") != "not_available_in_current_artifacts"
        and row.get("variant_id") != "B0_F2_base"
        and row.get("MV_AP_window") not in ("", None)
    ]
    best_real = max(real_candidates, key=lambda row: (_num(row.get("MV_AP_window")), _num(row.get("MV_AP50_window")), _num(row.get("MV_AP_scene"))))
    best_control_window = _control_best(control_metric_rows, "MV_AP_window")
    best_control_scene = _control_best(control_metric_rows, "MV_AP_scene")

    f2_base_window = float(phase0["F2_base_full_dev_MV_AP_window"])
    f2_base_ap50_window = float(phase0["F2_base_full_dev_MV_AP50_window"])
    f2_base_scene = float(phase0["F2_base_full_dev_MV_AP_scene"])
    f2_base_ap50_scene = float(phase0["F2_base_full_dev_MV_AP50_scene"])

    local_gate = bool(
        _num(best_real.get("MV_AP_window")) >= f2_base_window + 0.005
        and _num(best_real.get("MV_AP50_window")) >= f2_base_ap50_window + 0.010
    )
    scene_gate = bool(
        _num(best_real.get("MV_AP_scene")) >= f2_base_scene + 0.010
        and _num(best_real.get("MV_AP50_scene")) >= f2_base_ap50_scene + 0.015
        and _num(best_real.get("MV_AP_window")) >= f2_base_window - 0.003
    )
    control_window_pass = (
        best_control_window is not None
        and _num(best_real.get("MV_AP_window")) > _num(best_control_window.get("MV_AP_window"))
    )
    control_scene_pass = (
        best_control_scene is not None
        and _num(best_real.get("MV_AP_scene")) > _num(best_control_scene.get("MV_AP_scene"))
    )
    control_pass = bool((local_gate and control_window_pass) or (scene_gate and control_scene_pass))
    safety_pass = _safety_pass(best_real)

    gate_rows = [
        {
            "gate_id": "best_real_local_gate_vs_F2_base",
            "pass": local_gate,
            "expected": f"MV_AP_window>={f2_base_window + 0.005}; MV_AP50_window>={f2_base_ap50_window + 0.010}",
            "observed": f"{best_real['variant_id']} MV_AP_window={best_real.get('MV_AP_window')}; MV_AP50_window={best_real.get('MV_AP50_window')}",
            "severity": "candidate",
        },
        {
            "gate_id": "best_real_scene_gate_vs_F2_base",
            "pass": scene_gate,
            "expected": f"MV_AP_scene>={f2_base_scene + 0.010}; MV_AP50_scene>={f2_base_ap50_scene + 0.015}; MV_AP_window>={f2_base_window - 0.003}",
            "observed": f"{best_real['variant_id']} MV_AP_scene={best_real.get('MV_AP_scene')}; MV_AP50_scene={best_real.get('MV_AP50_scene')}; MV_AP_window={best_real.get('MV_AP_window')}",
            "severity": "candidate",
        },
        {
            "gate_id": "best_real_exceeds_best_synchronized_control",
            "pass": control_pass,
            "expected": "local candidate exceeds best control on MV_AP_window or scene candidate exceeds best control on MV_AP_scene",
            "observed": (
                f"best_real={best_real['variant_id']} window={best_real.get('MV_AP_window')} scene={best_real.get('MV_AP_scene')}; "
                f"best_control_window={best_control_window.get('variant_id') if best_control_window else ''}:{best_control_window.get('MV_AP_window') if best_control_window else ''}; "
                f"best_control_scene={best_control_scene.get('variant_id') if best_control_scene else ''}:{best_control_scene.get('MV_AP_scene') if best_control_scene else ''}"
            ),
            "severity": "required",
        },
        {
            "gate_id": "safety_gates_for_best_real",
            "pass": safety_pass,
            "expected": "same_frame_collision_count=0; pixel_collision_rate<=0.02 if present; missing_mask_raster_count=0; no GT/future",
            "observed": (
                f"same_frame_collision_count={best_real.get('same_frame_collision_count')}; "
                f"pixel_collision_rate={best_real.get('pixel_collision_rate')}; "
                f"missing_mask_raster_count={best_real.get('missing_mask_raster_count')}; "
                f"uses_gt_for_prediction={best_real.get('uses_gt_for_prediction')}; uses_future={best_real.get('uses_future')}"
            ),
            "severity": "required",
        },
    ]
    failure_rows = [
        {
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": "If Phase8 fails candidate/control gates, keep unproven auxiliary modules diagnostic and advance to Phase9 scene-local2history only with proven branches.",
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]

    if (local_gate or scene_gate) and control_pass and safety_pass:
        decision = "PASS_FUSION_MATRIX_CANDIDATE_FOR_PHASE9_AND_HOLDOUT"
        phase8_pass = True
    elif not local_gate and not scene_gate:
        decision = "KEEP_F2_AS_MAIN_METHOD"
        phase8_pass = False
    elif not control_pass:
        decision = "NO_GO_CONTROL_BIAS"
        phase8_pass = False
    else:
        decision = "NO_GO_FUSION_SAFETY"
        phase8_pass = False

    summary = {
        "schema_version": "stream4d_v99_phase8_fusion_matrix_summary_v1",
        "phase_id": "v99_phase8_fusion_matrix",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "decision": decision,
        "phase8_pass": phase8_pass,
        "best_real_variant": best_real["variant_id"],
        "best_real_MV_AP_window": float(_num(best_real.get("MV_AP_window"))),
        "best_real_MV_AP50_window": float(_num(best_real.get("MV_AP50_window"))),
        "best_real_MV_AP_scene": float(_num(best_real.get("MV_AP_scene"))),
        "best_real_MV_AP50_scene": float(_num(best_real.get("MV_AP50_scene"))),
        "best_real_source_variant_id": best_real.get("phase8_source_variant_id", ""),
        "best_control_window_variant": best_control_window.get("variant_id") if best_control_window else "",
        "best_control_window_MV_AP_window": float(_num(best_control_window.get("MV_AP_window"))) if best_control_window else "",
        "best_control_scene_variant": best_control_scene.get("variant_id") if best_control_scene else "",
        "best_control_scene_MV_AP_scene": float(_num(best_control_scene.get("MV_AP_scene"))) if best_control_scene else "",
        "F2_base_MV_AP_window": f2_base_window,
        "F2_base_MV_AP50_window": f2_base_ap50_window,
        "F2_base_MV_AP_scene": f2_base_scene,
        "F2_base_MV_AP50_scene": f2_base_ap50_scene,
        "local_gate_pass": local_gate,
        "scene_gate_pass": scene_gate,
        "control_gate_pass": control_pass,
        "safety_gate_pass": safety_pass,
        "DA3_component_source_variant": da3_variant,
        "D4RT_component_source_variant": d4rt_variant,
        "dense_component_source_variant": dense_variant,
        "broad_split_source_variant": broad_variant,
        "B10_scene_stitch_deferred_to_phase9": True,
        "outputs": {
            "variant_config_rows": _rel(OUT_DIR / "variant_config_rows.csv"),
            "variant_metric_rows": _rel(OUT_DIR / "variant_metric_rows.csv"),
            "control_metric_rows": _rel(OUT_DIR / "control_metric_rows.csv"),
            "variant_gate_rows": _rel(OUT_DIR / "variant_gate_rows.csv"),
            "variant_failure_rows": _rel(OUT_DIR / "variant_failure_rows.csv"),
            "best_variant_summary": _rel(OUT_DIR / "best_variant_summary.json"),
            "mv_object_frame_mask_rows": _rel(OUT_DIR / "mv_object_frame_mask_rows.csv"),
            "variant_metric_scene_rows": _rel(OUT_DIR / "variant_metric_scene_rows.csv"),
            "frame_eval_rows": _rel(OUT_DIR / "frame_eval_rows.csv"),
        },
    }

    _write_csv(OUT_DIR / "variant_config_rows.csv", variant_config_rows)
    _write_csv(OUT_DIR / "variant_metric_rows.csv", variant_metric_rows)
    _write_csv(OUT_DIR / "control_metric_rows.csv", control_metric_rows)
    _write_csv(OUT_DIR / "variant_gate_rows.csv", gate_rows)
    _write_csv(OUT_DIR / "variant_failure_rows.csv", failure_rows)
    _write_csv(OUT_DIR / "variant_metric_scene_rows.csv", metric_scene_rows)
    _write_csv(OUT_DIR / "frame_eval_rows.csv", frame_rows)
    _write_csv(OUT_DIR / "mv_object_frame_mask_rows.csv", all_rows)
    _write_json(OUT_DIR / "best_variant_summary.json", summary)
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if phase8_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
