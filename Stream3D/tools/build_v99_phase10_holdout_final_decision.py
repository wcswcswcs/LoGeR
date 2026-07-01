#!/usr/bin/env python3
"""Run v99 Phase10 holdout for the frozen Phase8 local candidate."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools import build_v98_1_canonical_holdout_metrics as holdout  # noqa: E402
from tools import build_v99_phase1_f2_base_reproduction as p1  # noqa: E402
from tools import build_v99_phase4_f2_da3_link_verifier as p4  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase10_holdout_final_decision"
PHASE0_DIR = AUDIT_ROOT / "v99_phase0_fact_lock"
PHASE2_DIR = AUDIT_ROOT / "v99_phase2_f2_strengthening"
PHASE8_DIR = AUDIT_ROOT / "v99_phase8_fusion_matrix"
PHASE9_DIR = AUDIT_ROOT / "v99_phase9_scene_local2history"
HOLDOUT_SOURCE_ROWS = AUDIT_ROOT / "v98_phase13_holdout/source_container_rows.csv"
HOLDOUT_FIXED_ROWS = PHASE2_DIR / "holdout_mv_object_frame_mask_rows.csv"
BASE_PHASE2_HOLDOUT_SUMMARY = PHASE2_DIR / "holdout_summary.json"
HOLDOUT_VARIANT = "B2_F2_plus_DA3_link_verifier__holdout_fixed"
EPS = 1e-4

HOLDOUT_DA3_SCENES = {
    "scene0011_00": {
        "input": AUDIT_ROOT / "v98_phase1_provider_contract/da3_streaming_holdout_scene0011_input",
        "output": AUDIT_ROOT / "v98_phase1_provider_contract/da3_streaming_holdout_scene0011",
    },
    "scene0050_00": {
        "input": AUDIT_ROOT / "v98_phase1_provider_contract/da3_streaming_holdout_scene0050_input",
        "output": AUDIT_ROOT / "v98_phase1_provider_contract/da3_streaming_holdout_scene0050",
    },
}


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
    if isinstance(value, np.ndarray):
        return value.tolist()
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


def _load_holdout_da3_maps() -> dict[str, dict[str, Any]]:
    maps: dict[str, dict[str, Any]] = {}
    for scene, spec in HOLDOUT_DA3_SCENES.items():
        manifest = _read_csv(Path(spec["input"]) / "frame_manifest_rows.csv")
        maps[scene] = {
            "frame_to_idx": {int(row["frame_id"]): int(row["da3_frame_index"]) for row in manifest},
            "idx_to_frame": {int(row["da3_frame_index"]): int(row["frame_id"]) for row in manifest},
            "poses": p4._read_poses(Path(spec["output"]) / "camera_poses.txt"),
            "output": Path(spec["output"]),
            "input": Path(spec["input"]),
        }
    return maps


def _build_holdout_link_metrics(rows: list[dict[str, Any]], source_scope: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    da3_maps = _load_holdout_da3_maps()
    geom_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_object[str(row["mv_object_id"])].append(row)
    link_rows: list[dict[str, Any]] = []
    scores: dict[str, list[float]] = defaultdict(list)
    conflicts: dict[str, int] = defaultdict(int)
    for oid, vals in by_object.items():
        vals_sorted = sorted(vals, key=lambda row: (str(row["scene_id"]), int(row["frame_id"]), int(row["mask_id"])))
        for a, b in zip(vals_sorted[:-1], vals_sorted[1:]):
            if a["scene_id"] != b["scene_id"] or int(a["frame_id"]) == int(b["frame_id"]):
                continue
            ga = p4._mask_geometry(
                str(a["scene_id"]),
                int(a["frame_id"]),
                int(a["mask_id"]),
                da3_maps=da3_maps,
                mask_path_by_frame=source_scope["mask_path_by_frame"],
                cache=geom_cache,
            )
            gb = p4._mask_geometry(
                str(b["scene_id"]),
                int(b["frame_id"]),
                int(b["mask_id"]),
                da3_maps=da3_maps,
                mask_path_by_frame=source_scope["mask_path_by_frame"],
                cache=geom_cache,
            )
            confident = bool(ga["valid"] and gb["valid"] and ga["point_count"] >= 32 and gb["point_count"] >= 32)
            if confident:
                dist = float(np.linalg.norm(ga["centroid"] - gb["centroid"]))
                denom = max(0.10, float(ga["radius"]) + float(gb["radius"]) + 0.25)
                consistency = float(math.exp(-dist / denom))
                conf_mean = 0.5 * (float(ga["mean_conf"]) + float(gb["mean_conf"]))
            else:
                dist = ""
                consistency = 0.0
                conf_mean = 0.0
            conflict = bool(confident and consistency < 0.10)
            if confident:
                scores[oid].append(consistency)
            if conflict:
                conflicts[oid] += 1
            link_rows.append(
                {
                    "schema_version": "stream4d_v99_phase10_holdout_da3_link_v1",
                    "phase_id": "v99_phase10_holdout_final_decision",
                    "variant_id": HOLDOUT_VARIANT,
                    "mv_object_id": oid,
                    "scene_id": a["scene_id"],
                    "frame_a": int(a["frame_id"]),
                    "mask_a": int(a["mask_id"]),
                    "frame_b": int(b["frame_id"]),
                    "mask_b": int(b["mask_id"]),
                    "da3_confident": confident,
                    "geometry_distance": dist,
                    "geometry_consistency": consistency,
                    "mean_da3_confidence": conf_mean,
                    "geo_conflict": conflict,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    obj_metric: dict[str, dict[str, float]] = {}
    for oid in by_object:
        vals = scores.get(oid, [])
        obj_metric[oid] = {
            "mean_geometry_consistency": float(np.mean(vals)) if vals else 0.0,
            "confident_link_count": float(len(vals)),
            "conflict_count": float(conflicts.get(oid, 0)),
        }
    return link_rows, obj_metric


def _norm_metric(values: dict[str, float]) -> dict[str, float]:
    vals = list(values.values())
    if not vals:
        return {}
    lo = min(vals)
    hi = max(vals)
    if hi - lo <= 1e-12:
        return {key: 0.0 for key in values}
    return {key: (val - lo) / (hi - lo) for key, val in values.items()}


def _apply_da3_boost(rows: list[dict[str, Any]], obj_metric: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    geom_norm = _norm_metric({oid: vals["mean_geometry_consistency"] for oid, vals in obj_metric.items()})
    out: list[dict[str, Any]] = []
    for row in rows:
        oid = str(row["mv_object_id"])
        new = dict(row)
        new["variant_id"] = HOLDOUT_VARIANT
        new["variant"] = HOLDOUT_VARIANT
        new["score"] = _num(row.get("score")) + EPS * geom_norm.get(oid, 0.0)
        new["score_policy"] = str(row.get("score_policy", "")) + "__phase10_da3_boost_only"
        new["fixed_dev_variant_id"] = "B2_F2_plus_DA3_link_verifier"
        new["da3_geometry_consistency"] = obj_metric.get(oid, {}).get("mean_geometry_consistency", 0.0)
        new["da3_confident_link_count"] = obj_metric.get(oid, {}).get("confident_link_count", 0.0)
        new["da3_conflict_count"] = obj_metric.get(oid, {}).get("conflict_count", 0.0)
        new["uses_gt_for_prediction"] = False
        new["uses_future"] = False
        out.append(new)
    return out


def main() -> int:
    started = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase0 = json.loads((PHASE0_DIR / "summary.json").read_text(encoding="utf-8"))
    phase8 = json.loads((PHASE8_DIR / "summary.json").read_text(encoding="utf-8"))
    phase9 = json.loads((PHASE9_DIR / "summary.json").read_text(encoding="utf-8"))
    phase2_holdout = json.loads(BASE_PHASE2_HOLDOUT_SUMMARY.read_text(encoding="utf-8"))
    source_scope = holdout._load_source_scope(HOLDOUT_SOURCE_ROWS)
    parent_rows = [dict(row) for row in _read_csv(HOLDOUT_FIXED_ROWS)]
    if not parent_rows:
        raise RuntimeError(f"no holdout parent rows in {HOLDOUT_FIXED_ROWS}")

    link_rows, obj_metric = _build_holdout_link_metrics(parent_rows, source_scope)
    boosted_rows = _apply_da3_boost(parent_rows, obj_metric)
    metric_rows, case_rows, top_rows = holdout._evaluate_variant(HOLDOUT_VARIANT, boosted_rows, source_scope)
    aggregate_rows = holdout._aggregate(metric_rows, family="v99_phase10_B2_DA3_link_holdout")
    if len(aggregate_rows) != 1:
        raise RuntimeError(f"expected one holdout aggregate row, got {len(aggregate_rows)}")
    aggregate = aggregate_rows[0]
    holdout_window = _num(aggregate.get("mean_MV_AP_window"))
    holdout_ap50 = _num(aggregate.get("mean_MV_AP50_window"))
    f2_holdout_window = float(phase0["F2_base_holdout_MV_AP_window"])
    f2_holdout_ap50 = float(phase0["F2_base_holdout_MV_AP50_window"])
    local_holdout_gate = bool(
        holdout_window >= f2_holdout_window + 0.005
        and holdout_ap50 >= f2_holdout_ap50 + 0.010
        and int(_num(aggregate.get("same_frame_collision_count"), 1)) == 0
        and int(_num(aggregate.get("missing_mask_raster_count"), 1)) == 0
        and not source_scope["uses_future"]
        and not source_scope["uses_gt_for_prediction"]
    )
    no_drop_gate = bool(holdout_window >= f2_holdout_window - 0.005)
    delta_vs_f2_holdout = holdout_window - f2_holdout_window
    delta_vs_phase2_holdout = holdout_window - float(phase2_holdout["holdout_MV_AP_window"])
    phase8_dev_gate = bool(phase8.get("phase8_pass") and phase8.get("local_gate_pass"))
    phase9_scene_gate = bool(phase9.get("phase9_pass"))
    final_label = (
        "GO_F2_ANCHORED_GEOMETRY_AUX"
        if phase8_dev_gate and local_holdout_gate
        else "NO_GO_KEEP_F2_BASE"
    )
    gate_rows = [
        {
            "gate_id": "phase8_dev_local_candidate_passed",
            "pass": phase8_dev_gate,
            "expected": "Phase8 local candidate gate pass",
            "observed": f"phase8_pass={phase8.get('phase8_pass')}; local_gate_pass={phase8.get('local_gate_pass')}; best={phase8.get('best_real_variant')}",
            "severity": "required",
        },
        {
            "gate_id": "phase10_local_holdout_MV_AP_window_ge_F2_holdout_plus_0p005",
            "pass": holdout_window >= f2_holdout_window + 0.005,
            "expected": f">={f2_holdout_window + 0.005}",
            "observed": holdout_window,
            "severity": "required_holdout",
        },
        {
            "gate_id": "phase10_local_holdout_MV_AP50_window_ge_F2_holdout_plus_0p010",
            "pass": holdout_ap50 >= f2_holdout_ap50 + 0.010,
            "expected": f">={f2_holdout_ap50 + 0.010}",
            "observed": holdout_ap50,
            "severity": "required_holdout",
        },
        {
            "gate_id": "phase10_holdout_no_collision_missing_future_gt",
            "pass": int(_num(aggregate.get("same_frame_collision_count"), 1)) == 0
            and int(_num(aggregate.get("missing_mask_raster_count"), 1)) == 0
            and not source_scope["uses_future"]
            and not source_scope["uses_gt_for_prediction"],
            "expected": "same_frame_collision_count=0; missing_mask_raster_count=0; no future/GT",
            "observed": (
                f"same_frame_collision_count={aggregate.get('same_frame_collision_count')}; "
                f"missing_mask_raster_count={aggregate.get('missing_mask_raster_count')}; "
                f"uses_future={source_scope['uses_future']}; uses_gt_for_prediction={source_scope['uses_gt_for_prediction']}"
            ),
            "severity": "required",
        },
        {
            "gate_id": "phase9_scene_candidate_passed",
            "pass": phase9_scene_gate,
            "expected": "Phase9 scene/local2history gate pass if claiming scene method",
            "observed": f"phase9_pass={phase9.get('phase9_pass')}; best_scene={phase9.get('best_scene_variant')}",
            "severity": "scene_optional",
        },
    ]
    failure_rows = [
        {
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": (
                "Do not claim holdout-qualified v99 method; keep F2_base as stable baseline and DA3/D4RT/dense branches diagnostic unless a future frozen config passes holdout."
                if row["severity"] != "scene_optional"
                else "Scene branch remains No-Go; report only local dev evidence, not scene stitching success."
            ),
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    freeze_config = {
        "schema_version": "stream4d_v99_phase10_freeze_config_v1",
        "method_family": "F2_plus_DA3_link_verifier",
        "frozen_dev_candidate": phase8.get("best_real_variant"),
        "phase8_source_variant": phase8.get("best_real_source_variant_id"),
        "parent_holdout_rows": _rel(HOLDOUT_FIXED_ROWS),
        "score_policy": "holdout Phase2 fixed score + 1e-4 normalized DA3 geometry consistency",
        "DA3_provider": "official DA3-Streaming holdout artifacts from v98_phase1_provider_contract",
        "DA3_chunk_protocol": "holdout provider artifact; no future/GT according to frame manifest",
        "D4RT_role": "diagnostic_only",
        "dense_semantic_role": "diagnostic_only",
        "scene_stitching_policy": "not_promoted_phase9_no_go",
        "holdout_tuning": "none; config frozen from Phase8 before this run",
    }
    summary = {
        "schema_version": "stream4d_v99_phase10_holdout_final_decision_summary_v1",
        "phase_id": "v99_phase10_holdout_final_decision",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "decision": final_label,
        "phase10_pass": local_holdout_gate,
        "frozen_candidate": phase8.get("best_real_variant"),
        "holdout_variant_id": HOLDOUT_VARIANT,
        "holdout_MV_AP_window": holdout_window,
        "holdout_MV_AP50_window": holdout_ap50,
        "holdout_MV_AP25_window": _num(aggregate.get("mean_MV_AP25_window")),
        "holdout_score_free_Match50_window": _num(aggregate.get("mean_score_free_Match50_window")),
        "F2_base_holdout_MV_AP_window": f2_holdout_window,
        "F2_base_holdout_MV_AP50_window": f2_holdout_ap50,
        "delta_vs_F2_base_holdout_MV_AP_window": delta_vs_f2_holdout,
        "delta_vs_phase2_holdout_no_aux_MV_AP_window": delta_vs_phase2_holdout,
        "strict_local_holdout_gate_pass": local_holdout_gate,
        "diagnostic_no_drop_gate_pass": no_drop_gate,
        "same_frame_collision_count": int(_num(aggregate.get("same_frame_collision_count"), 1)),
        "missing_mask_raster_count": int(_num(aggregate.get("missing_mask_raster_count"), 1)),
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "phase8_dev_gate_pass": phase8_dev_gate,
        "phase9_scene_gate_pass": phase9_scene_gate,
        "holdout_da3_link_count": len(link_rows),
        "holdout_da3_confident_link_count": sum(1 for row in link_rows if str(row.get("da3_confident")).lower() == "true" or row.get("da3_confident") is True),
        "holdout_da3_link_geometry_consistency_mean": float(np.mean([_num(row.get("geometry_consistency")) for row in link_rows if row.get("da3_confident")])) if link_rows else 0.0,
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "final_decision_summary": _rel(OUT_DIR / "final_decision_summary.json"),
            "freeze_config": _rel(OUT_DIR / "freeze_config.json"),
            "holdout_da3_link_rows": _rel(OUT_DIR / "holdout_da3_link_rows.csv"),
            "holdout_mv_object_frame_mask_rows": _rel(OUT_DIR / "holdout_mv_object_frame_mask_rows.csv"),
            "holdout_metric_rows": _rel(OUT_DIR / "holdout_metric_rows.csv"),
            "holdout_metric_aggregate_rows": _rel(OUT_DIR / "holdout_metric_aggregate_rows.csv"),
            "holdout_case_rows": _rel(OUT_DIR / "holdout_case_rows.csv"),
            "holdout_top_iou_rows": _rel(OUT_DIR / "holdout_top_iou_rows.csv"),
            "variant_gate_rows": _rel(OUT_DIR / "variant_gate_rows.csv"),
            "variant_failure_rows": _rel(OUT_DIR / "variant_failure_rows.csv"),
        },
    }
    _write_json(OUT_DIR / "freeze_config.json", freeze_config)
    _write_csv(OUT_DIR / "holdout_da3_link_rows.csv", link_rows)
    _write_csv(OUT_DIR / "holdout_mv_object_frame_mask_rows.csv", boosted_rows)
    _write_csv(OUT_DIR / "holdout_metric_rows.csv", metric_rows)
    _write_csv(OUT_DIR / "holdout_metric_aggregate_rows.csv", aggregate_rows)
    _write_csv(OUT_DIR / "holdout_case_rows.csv", case_rows)
    _write_csv(OUT_DIR / "holdout_top_iou_rows.csv", top_rows)
    _write_csv(OUT_DIR / "variant_gate_rows.csv", gate_rows)
    _write_csv(OUT_DIR / "variant_failure_rows.csv", failure_rows)
    _write_json(OUT_DIR / "summary.json", summary)
    _write_json(OUT_DIR / "final_decision_summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if local_holdout_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
