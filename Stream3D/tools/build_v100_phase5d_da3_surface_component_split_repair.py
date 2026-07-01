#!/usr/bin/env python3
"""Run v100 DA3 surface-component split repair on Phase2c rows."""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools import build_v99_phase1_f2_base_reproduction as p1  # noqa: E402
from tools import build_v100_phase4h_overlap3_exact_history_memory as p4h  # noqa: E402
from tools import build_v100_phase5c_da3_broad_split_repair as p5c  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v100_phase5d_da3_surface_component_split_repair"
PHASE2C_DIR = AUDIT_ROOT / "v100_phase2c_overlap3_local_repair"
PHASE2C_SUMMARY = PHASE2C_DIR / "summary.json"

MIN_COMPONENT_AREA_PX = 4096


def _rel(path: Path | str) -> str:
    return p4h._rel(path)


def _num(value: Any, default: float = 0.0) -> float:
    return p4h._num(value, default)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    p4h._write_csv(path, rows)


def _write_json(path: Path, payload: Any) -> None:
    p4h._write_json(path, payload)


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    p4h._write_parquet(path, rows)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _should_split(row: dict[str, Any], area_ratio: float, variant: str) -> bool:
    score = _num(row.get("score"), 1.0)
    if variant == "G4S1_q3_surface_components_broad":
        return area_ratio >= 0.30
    if variant == "G4S2_q3_surface_components_score_lt_0p85":
        return area_ratio >= 0.30 and score < 0.85
    if variant == "G4S3_q3_surface_components_score_lt_0p75":
        return area_ratio >= 0.25 and score < 0.75
    return False


def _surface_components(
    label: np.ndarray,
    mask_id: int,
    depth: np.ndarray,
    *,
    bins: int = 3,
    max_components: int = 4,
) -> tuple[list[np.ndarray], dict[str, Any]] | None:
    mask = label == int(mask_id)
    parent_area = int(np.count_nonzero(mask))
    if parent_area < 2 * MIN_COMPONENT_AREA_PX:
        return None
    depth_resized = cv2.resize(depth, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_LINEAR)
    valid = mask & np.isfinite(depth_resized) & (depth_resized > 0)
    vals = depth_resized[valid]
    if vals.size < 2 * MIN_COMPONENT_AREA_PX:
        return None
    edges = np.quantile(vals, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    if len(edges) == 0 or float(np.max(edges) - np.min(edges)) <= 1e-6:
        return None
    bin_ids = np.digitize(depth_resized, edges, right=False)
    candidates: list[tuple[int, np.ndarray, int]] = []
    for bin_id in range(bins):
        binary = (valid & (bin_ids == bin_id)).astype(np.uint8)
        if int(binary.sum()) < MIN_COMPONENT_AREA_PX:
            continue
        comp_count, comp_label, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for comp_idx in range(1, comp_count):
            area = int(stats[comp_idx, cv2.CC_STAT_AREA])
            if area >= MIN_COMPONENT_AREA_PX:
                candidates.append((area, comp_label == comp_idx, bin_id))
    if len(candidates) < 2:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[:max_components]
    covered = int(sum(item[0] for item in selected))
    if covered / max(1, parent_area) < 0.45:
        return None
    comps = [item[1] for item in selected]
    meta = {
        "component_count": len(comps),
        "component_area_px": [int(item[0]) for item in selected],
        "depth_bin_ids": [int(item[2]) for item in selected],
        "covered_parent_area_ratio": float(covered / max(1, parent_area)),
        "quantile_edges": [float(x) for x in edges],
    }
    return comps, meta


def _build_variant(
    *,
    split: str,
    variant: str,
    parent_rows: list[dict[str, Any]],
    scope: dict[str, Any],
    da3_maps: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Path], list[dict[str, Any]], dict[str, Any]]:
    if variant == "G4S0_phase2c_no_split":
        rows = []
        for row in parent_rows:
            new = dict(row)
            new["schema_version"] = "stream4d_v100_phase5d_mv_object_frame_mask_row_v1"
            new["phase_id"] = "v100_phase5d_da3_surface_component_split_repair"
            new["variant_id"] = variant
            new["variant"] = variant
            new["phase5d_split_action"] = "keep_original_baseline"
            new["uses_gt_for_prediction"] = False
            new["uses_future"] = False
            rows.append(new)
        return rows, {}, [], {"generated_mask_frame_count": 0, "missing_depth_frame_count": 0}

    rows_by_frame: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in parent_rows:
        rows_by_frame[(str(row["scene_id"]), int(row["frame_id"]))].append(row)

    out_rows: list[dict[str, Any]] = []
    mask_paths: dict[str, Path] = {}
    split_rows: list[dict[str, Any]] = []
    missing_depth_frames = 0
    mask_out_dir = OUT_DIR / "generated_masks" / split / variant

    for (scene, frame), rows in sorted(rows_by_frame.items()):
        src_path = scope["mask_path_by_frame"].get((scene, frame))
        if src_path is None or not src_path.exists():
            out_rows.extend(dict(row, variant_id=variant, variant=variant, phase5d_split_action="keep_original_missing_mask_scope") for row in rows)
            continue
        label = p1._read_label(src_path).copy()
        original = label.copy()
        depth = p5c._load_depth(scene, frame, da3_maps)
        if depth is None:
            missing_depth_frames += 1
        max_label = int(label.max()) + 1
        frame_changed = False
        for row in rows:
            mask_id = int(row["selected_mask_id"])
            area_px = int(np.count_nonzero(original == mask_id))
            area_ratio = float(area_px / max(1, original.size))
            split_result = None
            if depth is not None and _should_split(row, area_ratio, variant):
                split_result = _surface_components(original, mask_id, depth)
            if split_result is None:
                new = dict(row)
                new["schema_version"] = "stream4d_v100_phase5d_mv_object_frame_mask_row_v1"
                new["phase_id"] = "v100_phase5d_da3_surface_component_split_repair"
                new["variant_id"] = variant
                new["variant"] = variant
                new["phase5d_parent_mv_object_id"] = row.get("mv_object_id", "")
                new["phase5d_split_action"] = "keep_original"
                new["uses_gt_for_prediction"] = False
                new["uses_future"] = False
                out_rows.append(new)
                continue
            components, meta = split_result
            label[original == mask_id] = 0
            frame_changed = True
            for comp_idx, comp_mask in enumerate(components):
                new_id = max_label
                max_label += 1
                label[comp_mask] = new_id
                parent_oid = str(row["mv_object_id"])
                new_oid = f"{parent_oid}|phase5d_surface{comp_idx}"
                comp_area = int(np.count_nonzero(comp_mask))
                new = dict(row)
                new["schema_version"] = "stream4d_v100_phase5d_mv_object_frame_mask_row_v1"
                new["phase_id"] = "v100_phase5d_da3_surface_component_split_repair"
                new["variant_id"] = variant
                new["variant"] = variant
                new["mv_object_id"] = new_oid
                new["object_id"] = new_oid
                new["selected_mask_id"] = int(new_id)
                new["mask_id_or_generated_id"] = int(new_id)
                new["score"] = float(_num(row.get("score"), 1.0) * 0.72)
                new["score_policy"] = f"{row.get('score_policy', '')}__phase5d_surface_component_parent_times_0p72"
                new["object_id_policy"] = "phase5d_da3_surface_component_identity"
                new["phase5d_parent_mv_object_id"] = parent_oid
                new["phase5d_parent_mask_id"] = mask_id
                new["phase5d_split_action"] = "da3_depth_quantile_connected_component"
                new["uses_gt_for_prediction"] = False
                new["uses_future"] = False
                out_rows.append(new)
                split_rows.append(
                    {
                        "schema_version": "stream4d_v100_phase5d_split_row_v1",
                        "phase_id": "v100_phase5d_da3_surface_component_split_repair",
                        "variant_id": variant,
                        "dataset_split": split,
                        "scene_id": scene,
                        "frame_id": frame,
                        "parent_mv_object_id": parent_oid,
                        "parent_mask_id": mask_id,
                        "generated_mask_id": int(new_id),
                        "component_index": comp_idx,
                        "parent_area_ratio": area_ratio,
                        "component_area_px": comp_area,
                        "component_count_for_parent": meta["component_count"],
                        "covered_parent_area_ratio": meta["covered_parent_area_ratio"],
                        "depth_bin_id": meta["depth_bin_ids"][comp_idx],
                        "quantile_edges": json.dumps(meta["quantile_edges"]),
                        "parent_score": _num(row.get("score"), 1.0),
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )
        if frame_changed:
            out_path = mask_out_dir / scene / f"{frame}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), label.astype(np.uint16))
            mask_paths[f"{scene}|{frame}"] = out_path
    return out_rows, mask_paths, split_rows, {
        "generated_mask_frame_count": len(mask_paths),
        "missing_depth_frame_count": missing_depth_frames,
    }


def main() -> int:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase2c = _read_json(PHASE2C_SUMMARY)
    if not bool(phase2c.get("phase2c_pass")):
        raise RuntimeError("Phase5d requires v100 Phase2c overlap3 local pass")
    baselines = p4h._phase0_baselines()
    f2_dev = baselines["F2_base_full_dev"]
    f2_holdout = baselines["F2_base_holdout"]
    phase2c_df = pd.read_parquet(PHASE2C_DIR / "mv_object_frame_mask_rows.parquet")
    parent_by_split = {
        split: [dict(row) for row in sub.to_dict(orient="records")]
        for split, sub in phase2c_df.groupby("dataset_split")
    }
    scopes = {split: p5c._eval_scope_for_split(split) for split in ["dev", "holdout"]}
    da3_maps_by_split = {
        "dev": p5c._load_da3_maps(p5c.DEV_DA3),
        "holdout": p5c._load_da3_maps(p5c.HOLDOUT_DA3),
    }
    variants = [
        "G4S0_phase2c_no_split",
        "G4S1_q3_surface_components_broad",
        "G4S2_q3_surface_components_score_lt_0p85",
        "G4S3_q3_surface_components_score_lt_0p75",
    ]

    all_rows: list[dict[str, Any]] = []
    rows_by_variant_split: dict[tuple[str, str], list[dict[str, Any]]] = {}
    split_rows: list[dict[str, Any]] = []
    metric_scene_rows: list[dict[str, Any]] = []
    frame_eval_rows: list[dict[str, Any]] = []
    variant_metric_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    broad_counts = {split: p5c._broad_mask_count(parent_by_split[split], scopes[split]) for split in ["dev", "holdout"]}

    for variant in variants:
        for split in ["dev", "holdout"]:
            rows, generated_masks, splits, stats = _build_variant(
                split=split,
                variant=variant,
                parent_rows=parent_by_split[split],
                scope=scopes[split],
                da3_maps=da3_maps_by_split[split],
            )
            variant_scope = dict(scopes[split])
            mask_map = dict(scopes[split]["mask_path_by_frame"])
            for key, path in generated_masks.items():
                scene, frame = key.split("|")
                mask_map[(scene, int(frame))] = path
            variant_scope["mask_path_by_frame"] = mask_map
            p4h._set_inputs(split)
            per_scene, frames = p1._evaluate_variant(variant, rows, variant_scope)
            aggregate = p1._aggregate_metrics(per_scene)[0]
            aggregate["schema_version"] = "stream4d_v100_phase5d_metric_aggregate_row_v1"
            aggregate["phase_id"] = "v100_phase5d_da3_surface_component_split_repair"
            aggregate["variant_id"] = variant
            aggregate["dataset_split"] = split
            aggregate["metric_source"] = "fresh_v65_eval_on_phase2c_rows_after_da3_surface_component_split"
            aggregate["split_component_rows"] = len(splits)
            aggregate["split_success_count"] = len({(r["scene_id"], r["frame_id"], r["parent_mask_id"]) for r in splits})
            aggregate["generated_mask_frame_count"] = stats["generated_mask_frame_count"]
            aggregate["missing_depth_frame_count"] = stats["missing_depth_frame_count"]
            aggregate["broad_mask_count"] = broad_counts[split]
            aggregate["uses_gt_for_prediction"] = False
            aggregate["uses_future"] = False
            variant_metric_rows.append(aggregate)
            for item in per_scene:
                item["phase_id"] = "v100_phase5d_da3_surface_component_split_repair"
                item["dataset_split"] = split
                item["variant_id"] = variant
            for item in frames:
                item["phase_id"] = "v100_phase5d_da3_surface_component_split_repair"
                item["dataset_split"] = split
                item["variant_id"] = variant
            metric_scene_rows.extend(per_scene)
            frame_eval_rows.extend(frames)
            split_rows.extend(splits)
            all_rows.extend(rows)
            rows_by_variant_split[(variant, split)] = rows
            config_rows.append(
                {
                    "schema_version": "stream4d_v100_phase5d_variant_config_row_v1",
                    "phase_id": "v100_phase5d_da3_surface_component_split_repair",
                    "variant_id": variant,
                    "dataset_split": split,
                    "emitted_rows": len(rows),
                    "split_component_rows": len(splits),
                    "split_success_count": aggregate["split_success_count"],
                    "generated_mask_frame_count": stats["generated_mask_frame_count"],
                    "missing_depth_frame_count": stats["missing_depth_frame_count"],
                    "broad_mask_count": broad_counts[split],
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )

    by_variant: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in variant_metric_rows:
        by_variant[str(row["variant_id"])][str(row["dataset_split"])] = row

    def _rank(variant: str) -> tuple[float, float, float, float, float, float]:
        hold = by_variant[variant]["holdout"]
        dev = by_variant[variant]["dev"]
        return (
            _num(hold.get("MV_AP_scene")),
            _num(hold.get("MV_AP50_scene")),
            _num(dev.get("MV_AP_scene")),
            _num(dev.get("MV_AP50_scene")),
            _num(hold.get("MV_AP_window")),
            _num(dev.get("MV_AP_window")),
        )

    best_variant_id = max(variants, key=_rank)
    split_variants = [variant for variant in variants if variant != "G4S0_phase2c_no_split"]
    best_split_variant_id = max(split_variants, key=_rank)
    best_dev = by_variant[best_variant_id]["dev"]
    best_hold = by_variant[best_variant_id]["holdout"]
    best_split_hold = by_variant[best_split_variant_id]["holdout"]
    baseline_hold = by_variant["G4S0_phase2c_no_split"]["holdout"]

    dev_scene_gate = _num(best_dev.get("MV_AP_scene")) >= _num(f2_dev["MV_AP_scene"]) + 0.010
    dev_ap50_gate = _num(best_dev.get("MV_AP50_scene")) >= _num(f2_dev["MV_AP50_scene"]) + 0.015
    hold_scene_gate = _num(best_hold.get("MV_AP_scene")) >= _num(f2_holdout["MV_AP_scene"]) + 0.006
    hold_ap50_gate = _num(best_hold.get("MV_AP50_scene")) >= _num(f2_holdout["MV_AP50_scene"]) + 0.010
    local_drop_dev = float(phase2c["dev_MV_AP_window"]) - _num(best_dev.get("MV_AP_window"))
    local_drop_hold = float(phase2c["holdout_MV_AP_window"]) - _num(best_hold.get("MV_AP_window"))
    local_gate = local_drop_dev <= 0.003 and local_drop_hold <= 0.003
    split_improves_scene = _num(best_split_hold.get("MV_AP_scene")) > _num(baseline_hold.get("MV_AP_scene"))
    safety_gate = (
        int(_num(best_dev.get("same_frame_collision_count"))) == 0
        and int(_num(best_hold.get("same_frame_collision_count"))) == 0
        and _num(best_dev.get("pixel_collision_rate")) <= 0.02
        and _num(best_hold.get("pixel_collision_rate")) <= 0.02
        and int(_num(best_dev.get("missing_mask_raster_count"))) == 0
        and int(_num(best_hold.get("missing_mask_raster_count"))) == 0
    )
    phase5d_pass = bool(
        best_variant_id != "G4S0_phase2c_no_split"
        and dev_scene_gate
        and dev_ap50_gate
        and hold_scene_gate
        and hold_ap50_gate
        and local_gate
        and safety_gate
    )

    gate_rows = [
        {
            "gate_id": "best_variant_is_non_baseline_surface_split",
            "pass": best_variant_id != "G4S0_phase2c_no_split",
            "expected": "best ranked variant uses DA3 surface component split",
            "observed": best_variant_id,
            "severity": "repair_required",
        },
        {
            "gate_id": "best_surface_split_holdout_scene_improves_baseline",
            "pass": split_improves_scene,
            "expected": f">{baseline_hold.get('MV_AP_scene')}",
            "observed": best_split_hold.get("MV_AP_scene"),
            "severity": "repair_required",
        },
        {
            "gate_id": "mv_ap_scene_dev_ge_f2_base_plus_0p010",
            "pass": dev_scene_gate,
            "expected": _num(f2_dev["MV_AP_scene"]) + 0.010,
            "observed": best_dev.get("MV_AP_scene"),
            "severity": "required_scene",
        },
        {
            "gate_id": "mv_ap50_scene_dev_ge_f2_base_plus_0p015",
            "pass": dev_ap50_gate,
            "expected": _num(f2_dev["MV_AP50_scene"]) + 0.015,
            "observed": best_dev.get("MV_AP50_scene"),
            "severity": "required_scene",
        },
        {
            "gate_id": "mv_ap_scene_holdout_ge_f2_base_plus_0p006",
            "pass": hold_scene_gate,
            "expected": _num(f2_holdout["MV_AP_scene"]) + 0.006,
            "observed": best_hold.get("MV_AP_scene"),
            "severity": "required_scene",
        },
        {
            "gate_id": "mv_ap50_scene_holdout_ge_f2_base_plus_0p010",
            "pass": hold_ap50_gate,
            "expected": _num(f2_holdout["MV_AP50_scene"]) + 0.010,
            "observed": best_hold.get("MV_AP50_scene"),
            "severity": "required_scene",
        },
        {
            "gate_id": "local_window_ap_drop_le_0p003",
            "pass": local_gate,
            "expected": "<=0.003 dev and holdout",
            "observed": f"dev_drop={local_drop_dev}; holdout_drop={local_drop_hold}",
            "severity": "protect_local",
        },
        {
            "gate_id": "collision_missing_mask_safety",
            "pass": safety_gate,
            "expected": "collision=0 pixel<=0.02 missing_mask=0",
            "observed": f"dev_collision={best_dev.get('same_frame_collision_count')} hold_collision={best_hold.get('same_frame_collision_count')} dev_missing={best_dev.get('missing_mask_raster_count')} hold_missing={best_hold.get('missing_mask_raster_count')}",
            "severity": "required_safety",
        },
    ]
    failure_rows = [
        {
            "schema_version": "stream4d_v100_phase5d_failure_row_v1",
            "phase_id": "v100_phase5d_da3_surface_component_split_repair",
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": "If connected DA3 surface components still fail, do not keep sweeping split heuristics; the missing signal is cross-chunk identity, not per-frame broad-mask splitting.",
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]

    best_rows = rows_by_variant_split[(best_variant_id, "dev")] + rows_by_variant_split[(best_variant_id, "holdout")]
    variant_metric_csv = OUT_DIR / "variant_metric_rows.csv"
    scene_metric_csv = OUT_DIR / "mv_metric_scene_rows.csv"
    frame_csv = OUT_DIR / "frame_eval_rows.csv"
    split_csv = OUT_DIR / "split_rows.csv"
    config_csv = OUT_DIR / "variant_config_rows.csv"
    gate_csv = OUT_DIR / "variant_gate_rows.csv"
    failure_csv = OUT_DIR / "variant_failure_rows.csv"
    best_parquet = OUT_DIR / "best_mv_object_frame_mask_rows.parquet"
    all_parquet = OUT_DIR / "all_variant_mv_object_frame_mask_rows.parquet"
    performance_csv = OUT_DIR / "performance_rows.csv"
    artifact_csv = OUT_DIR / "artifact_manifest_rows.csv"
    summary_json = OUT_DIR / "summary.json"

    performance_rows = [
        {
            "schema_version": "stream4d_v100_phase5d_performance_row_v1",
            "phase_id": "v100_phase5d_da3_surface_component_split_repair",
            "case_id": "phase2c_da3_surface_component_split_and_v65_eval",
            "runtime_sec": time.time() - started,
            "variant_count": len(variants),
            "split_count": 2,
            "v65_evaluator_runs": len(variants) * 2,
            "emitted_row_count": len(all_rows),
            "split_component_row_count": len(split_rows),
        }
    ]

    _write_csv(variant_metric_csv, variant_metric_rows)
    _write_csv(scene_metric_csv, metric_scene_rows)
    _write_csv(frame_csv, frame_eval_rows)
    _write_csv(split_csv, split_rows)
    _write_csv(config_csv, config_rows)
    _write_csv(gate_csv, gate_rows)
    _write_csv(failure_csv, failure_rows)
    _write_parquet(best_parquet, best_rows)
    _write_parquet(all_parquet, all_rows)
    _write_csv(performance_csv, performance_rows)
    _write_csv(
        artifact_csv,
        p5c._artifact_rows(
            [
                (summary_json, "summary", "Phase5d DA3 surface component split summary"),
                (variant_metric_csv, "metrics", "aggregate metrics per variant/split"),
                (scene_metric_csv, "metrics", "per-scene evaluator rows"),
                (frame_csv, "diagnostic", "frame evaluator rows"),
                (split_csv, "diagnostic", "materialized DA3 surface components"),
                (config_csv, "config", "variant config rows"),
                (gate_csv, "gates", "gate rows"),
                (failure_csv, "failures", "failure rows"),
                (best_parquet, "prediction_rows", "best variant rows"),
                (all_parquet, "prediction_rows", "all variant rows"),
                (performance_csv, "performance", "runtime row"),
            ]
        ),
    )
    summary = {
        "schema_version": "stream4d_v100_phase5d_da3_surface_component_split_repair_summary_v1",
        "phase_id": "v100_phase5d_da3_surface_component_split_repair",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": time.time() - started,
        "decision": "PASS_ENTER_PHASE6_DA3_SURFACE_COMPONENT_SPLIT" if phase5d_pass else "BLOCK_PHASE5D_DA3_SURFACE_COMPONENT_SPLIT_REPAIR",
        "phase5d_pass": phase5d_pass,
        "best_variant_id": best_variant_id,
        "best_split_variant_id": best_split_variant_id,
        "best_dev_MV_AP_window": _num(best_dev.get("MV_AP_window")),
        "best_dev_MV_AP50_window": _num(best_dev.get("MV_AP50_window")),
        "best_dev_MV_AP_scene": _num(best_dev.get("MV_AP_scene")),
        "best_dev_MV_AP50_scene": _num(best_dev.get("MV_AP50_scene")),
        "best_holdout_MV_AP_window": _num(best_hold.get("MV_AP_window")),
        "best_holdout_MV_AP50_window": _num(best_hold.get("MV_AP50_window")),
        "best_holdout_MV_AP_scene": _num(best_hold.get("MV_AP_scene")),
        "best_holdout_MV_AP50_scene": _num(best_hold.get("MV_AP50_scene")),
        "best_split_holdout_MV_AP_scene": _num(best_split_hold.get("MV_AP_scene")),
        "best_split_holdout_MV_AP50_scene": _num(best_split_hold.get("MV_AP50_scene")),
        "baseline_holdout_MV_AP_scene": _num(baseline_hold.get("MV_AP_scene")),
        "phase2c_dev_MV_AP_window": float(phase2c["dev_MV_AP_window"]),
        "phase2c_holdout_MV_AP_window": float(phase2c["holdout_MV_AP_window"]),
        "local_window_AP_drop": {"dev": local_drop_dev, "holdout": local_drop_hold},
        "broad_mask_count": broad_counts,
        "split_component_row_count": len(split_rows),
        "failure_count": len(failure_rows),
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "outputs": {
            "summary": _rel(summary_json),
            "variant_metric_rows": _rel(variant_metric_csv),
            "mv_metric_scene_rows": _rel(scene_metric_csv),
            "frame_eval_rows": _rel(frame_csv),
            "split_rows": _rel(split_csv),
            "variant_config_rows": _rel(config_csv),
            "variant_gate_rows": _rel(gate_csv),
            "variant_failure_rows": _rel(failure_csv),
            "best_mv_object_frame_mask_rows": _rel(best_parquet),
            "all_variant_mv_object_frame_mask_rows": _rel(all_parquet),
            "performance_rows": _rel(performance_csv),
            "artifact_manifest_rows": _rel(artifact_csv),
        },
    }
    _write_json(summary_json, summary)
    print(json.dumps(p4h._jsonable(summary), indent=2, sort_keys=True))
    return 0 if phase5d_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
