#!/usr/bin/env python3
"""Run v95 Phase4 family D component-first object assignment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_v95_phase4_core_conditioned_expansion as base  # noqa: E402
from tools import build_v95_phase4C_object_competition as comp  # noqa: E402
from tools import run_v90_dev_extent_score_cross_audit as phase7d  # noqa: E402
from tools import run_v91_phase4_radius_sweep as radius_sweep  # noqa: E402


PHASE_ID = "v95_phase4D_component_assignment"
RUN_ID = "v95_phase4D_component_assignment"
OUT = ROOT / "outputs/audit/v95_phase4D_component_assignment"
PHASE0 = ROOT / "outputs/audit/v95_phase0_fact_lock/summary.json"
PHASE1 = ROOT / "outputs/audit/v95_phase1_physical_source_registry"
PHASE3 = ROOT / "outputs/audit/v95_phase3_object_query"


def _variant_specs() -> list[dict[str, Any]]:
    return [
        {
            "variant_id": "D0_conservative_components",
            "family": "real",
            "min_edge_radio": 0.88,
            "min_edge_weight": 0.82,
            "assignment_mode": "query_similarity",
            "margin_floor": 0.06,
            "risk_cap": 0.42,
            "area_cap": 0.52,
            "description": "Conservative RADIO/edge components assigned by query similarity.",
        },
        {
            "variant_id": "D1_medium_components",
            "family": "real",
            "min_edge_radio": 0.84,
            "min_edge_weight": 0.74,
            "assignment_mode": "query_similarity",
            "margin_floor": 0.04,
            "risk_cap": 0.50,
            "area_cap": 0.62,
            "description": "Medium components assigned by query similarity.",
        },
        {
            "variant_id": "D2_large_components",
            "family": "real",
            "min_edge_radio": 0.78,
            "min_edge_weight": 0.62,
            "assignment_mode": "query_similarity",
            "margin_floor": 0.02,
            "risk_cap": 0.62,
            "area_cap": 0.76,
            "description": "Larger components for high-recall component assignment.",
        },
        {
            "variant_id": "D3_component_by_core_support",
            "family": "real",
            "min_edge_radio": 0.82,
            "min_edge_weight": 0.70,
            "assignment_mode": "core_support",
            "margin_floor": 0.00,
            "risk_cap": 0.58,
            "area_cap": 0.66,
            "description": "Components assigned mainly by embedded core support.",
        },
        {
            "variant_id": "D4_component_by_query_similarity",
            "family": "real",
            "min_edge_radio": 0.82,
            "min_edge_weight": 0.70,
            "assignment_mode": "query_similarity",
            "margin_floor": 0.02,
            "risk_cap": 0.58,
            "area_cap": 0.66,
            "description": "Components assigned by query similarity with core tie-break.",
        },
        {
            "variant_id": "D5_unknown_retention",
            "family": "real",
            "min_edge_radio": 0.80,
            "min_edge_weight": 0.68,
            "assignment_mode": "unknown_retention",
            "margin_floor": 0.08,
            "risk_cap": 0.48,
            "area_cap": 0.58,
            "description": "Component assignment with low-margin/high-risk unknown retention.",
        },
    ]


def _components(region_indices: list[int], adjacency: dict[int, list[tuple[int, float, float]]], spec: dict[str, Any]) -> list[set[int]]:
    region_set = set(int(i) for i in region_indices)
    seen: set[int] = set()
    comps: list[set[int]] = []
    min_radio = float(spec.get("min_edge_radio", 0.82))
    min_weight = float(spec.get("min_edge_weight", 0.0))
    for start in sorted(region_set):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp_set = {start}
        while stack:
            src = stack.pop()
            for dst, edge_radio, edge_weight in adjacency.get(src, []):
                if dst not in region_set or dst in seen:
                    continue
                if edge_radio < min_radio or edge_weight < min_weight:
                    continue
                seen.add(dst)
                stack.append(dst)
                comp_set.add(dst)
        comps.append(comp_set)
    return comps


def _assign_components(
    object_rows: dict[str, list[dict[str, Any]]],
    comps: list[set[int]],
    spec: dict[str, Any],
) -> tuple[dict[str, set[int]], dict[str, Any]]:
    rows_by_obj_region = {
        object_id: {int(row["region_index"]): row for row in rows}
        for object_id, rows in object_rows.items()
    }
    selected: dict[str, set[int]] = {object_id: set() for object_id in object_rows}
    comp_assignments = 0
    unknown_components = 0
    shared_core_components = 0
    for comp_set in comps:
        candidate_scores: list[tuple[str, float, float, int]] = []
        for object_id, row_map in rows_by_obj_region.items():
            rows = [row_map[idx] for idx in comp_set if idx in row_map]
            if not rows:
                continue
            unary = base._mean([base._num(row.get("object_unary_score")) for row in rows])
            proto = base._mean([base._num(row.get("proto_similarity")) for row in rows])
            risk = base._mean([base._num(row.get("risk_score")) for row in rows])
            d4rt = base._mean([base._num(row.get("D4RT_witness_score")) for row in rows])
            core_count = sum(1 for row in rows if base._bool(row.get("is_core_region")))
            if str(spec.get("assignment_mode")) == "core_support":
                score = 0.62 * min(1.0, core_count) + 0.24 * unary + 0.08 * d4rt - 0.18 * risk
            else:
                score = 0.54 * unary + 0.18 * proto + 0.16 * min(1.0, core_count) + 0.08 * d4rt - 0.20 * risk
            candidate_scores.append((object_id, float(score), float(risk), int(core_count)))
        if not candidate_scores:
            continue
        candidate_scores.sort(key=lambda item: item[1], reverse=True)
        best_object, best_score, best_risk, best_core = candidate_scores[0]
        second = candidate_scores[1][1] if len(candidate_scores) > 1 else -999.0
        margin = best_score - second
        if sum(1 for _oid, _score, _risk, core_count in candidate_scores if core_count > 0) > 1:
            shared_core_components += 1
        if best_core <= 0 and (margin < float(spec.get("margin_floor", 0.0)) or best_risk > float(spec.get("risk_cap", 1.0))):
            unknown_components += 1
            continue
        if best_risk > float(spec.get("risk_cap", 1.0)) and str(spec.get("assignment_mode")) == "unknown_retention":
            unknown_components += 1
            continue
        selected[best_object].update(comp_set)
        comp_assignments += 1
    capped: dict[str, set[int]] = {}
    for object_id, indices in selected.items():
        capped[object_id] = base._cap_by_area(object_rows[object_id], indices, float(spec.get("area_cap", 1.0)))
    return capped, {
        "component_count": len(comps),
        "assigned_component_count": comp_assignments,
        "unknown_component_count": unknown_components,
        "shared_core_component_count": shared_core_components,
    }


def _object_score(rows: list[dict[str, Any]], selected: set[int]) -> float:
    selected_rows = [row for row in rows if int(row["region_index"]) in selected]
    return float(
        np.clip(
            0.50
            + 0.36 * base._mean([base._num(row.get("object_unary_score")) for row in selected_rows])
            + 0.18 * base._mean([base._num(row.get("proto_similarity")) for row in selected_rows])
            - 0.18 * base._mean([base._num(row.get("risk_score")) for row in selected_rows]),
            0.0,
            1.0,
        )
    )


def _gate_rows(
    variant_metric_rows: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    phase0: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool, dict[str, Any]]:
    spec_by_id = {spec["variant_id"]: spec for spec in specs}
    required_ap = base._num(phase0.get("required_MV_AP_window"))
    required_ap50 = base._num(phase0.get("required_MV_AP50_window"))
    v91_ap = base._num(phase0.get("v91_best_MV_AP_window"))
    v91_ap50 = base._num(phase0.get("v91_best_MV_AP50_window"))
    control_ap = base._num(phase0.get("best_control_MV_AP_window"))
    control_ap50 = base._num(phase0.get("best_control_MV_AP50_window"))
    candidate_required_ap = max(v91_ap + 0.002, control_ap + 0.005)
    candidate_required_ap50 = max(v91_ap50 + 0.004, control_ap50 + 0.010)
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    any_pass = False
    best_real: dict[str, Any] = {}
    for row in variant_metric_rows:
        variant_id = str(row.get("variant_id", ""))
        spec = spec_by_id.get(variant_id, {})
        mv_ap = base._num(row.get("mean_MV_AP_window"))
        mv_ap50 = base._num(row.get("mean_MV_AP50_window"))
        collision = base._int(row.get("same_frame_collision_count"))
        missing = base._int(row.get("missing_mask_raster_count"))
        is_real = spec.get("family") == "real"
        progress_gate = bool(mv_ap >= candidate_required_ap and mv_ap50 >= candidate_required_ap50)
        final_threshold_gate = bool(mv_ap >= required_ap and mv_ap50 >= required_ap50)
        provenance_gate = bool(
            collision == 0
            and missing == 0
            and not base._bool(row.get("uses_gt_for_prediction"))
            and not base._bool(row.get("uses_future"))
        )
        pass_gate = bool(is_real and progress_gate and provenance_gate)
        any_pass = any_pass or pass_gate
        if is_real and (
            not best_real
            or (mv_ap, mv_ap50)
            > (
                base._num(best_real.get("mean_MV_AP_window"), -999.0),
                base._num(best_real.get("mean_MV_AP50_window"), -999.0),
            )
        ):
            best_real = dict(row)
        gate_rows.append(
            {
                "schema_version": "stream4d_v95_phase4D_variant_gate_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": variant_id,
                "family": spec.get("family", ""),
                "MV_AP_window": mv_ap,
                "MV_AP50_window": mv_ap50,
                "MV_AP25_window": row.get("mean_MV_AP25_window", ""),
                "ScoreFreeMatch50_window": row.get("mean_score_free_Match50_window", ""),
                "candidate_required_MV_AP_window": candidate_required_ap,
                "candidate_required_MV_AP50_window": candidate_required_ap50,
                "final_required_MV_AP_window": required_ap,
                "final_required_MV_AP50_window": required_ap50,
                "phase4_candidate_gate_pass": pass_gate,
                "phase4_final_threshold_gate_pass": final_threshold_gate,
                "progress_gate_pass": progress_gate,
                "provenance_gate_pass": provenance_gate,
                "same_frame_collision_count": collision,
                "missing_mask_raster_count": missing,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        if is_real and not pass_gate:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v95_phase4D_failure_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "variant_id": variant_id,
                    "failure_type": "PHASE4_FAMILY_D_NO_CANDIDATE_GATE",
                    "MV_AP_window": mv_ap,
                    "MV_AP50_window": mv_ap50,
                    "repair_direction": (
                        "Family D component granularity and object ownership remain mismatched; "
                        "return to Phase3 object-query specificity per v95 plan."
                    ),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    return gate_rows, failure_rows, any_pass, best_real


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = base._resolve(args.output_root)
    if out.exists() and args.clean:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    query_root = base._resolve(args.query_root)
    phase0 = json.loads(PHASE0.read_text(encoding="utf-8"))
    phase3 = json.loads((query_root / "summary.json").read_text(encoding="utf-8"))
    if phase3.get("decision") != "PASS_V95_PHASE3_OBJECT_QUERY_READY":
        raise RuntimeError("v95 Phase3 must pass before Phase4D")
    specs = _variant_specs()
    variant_ids = [spec["variant_id"] for spec in specs]
    allowed_objects = comp._allowed_objects_by_query_support(query_root, int(args.min_query_support))
    allowed_sources = None
    if int(args.max_sources) > 0:
        allowed_sources = set()
        for source, _objects in comp._iter_source_groups(query_root / "region_object_unary_rows.csv", int(args.max_sources)):
            allowed_sources.add(source)
    source_meta = base._load_source_meta(base._resolve(args.source_container_rows))
    region_nodes = base._load_region_nodes(base._resolve(args.region_node_rows), allowed_sources)
    edge_adjacency = base._load_edge_adjacency(base._resolve(args.region_edge_rows), allowed_sources, region_nodes)
    frame_writer = base.FrameWriter(out, variant_ids)
    label_cache: dict[tuple[str, int], np.ndarray] = {}
    generated_rows: list[dict[str, Any]] = []
    mv_rows: list[dict[str, Any]] = []
    ownership_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    area_ratios_by_variant: dict[str, list[float]] = defaultdict(list)
    core_retention_by_variant: dict[str, list[float]] = defaultdict(list)
    processed_sources = 0
    config_rows = [
        {
            "schema_version": "stream4d_v95_phase4D_variant_config_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "created_at": base._created_at(),
            **spec,
            "query_root": base._rel(query_root),
            "selected_query_family": phase3.get("selected_query_family", ""),
            "min_query_support": int(args.min_query_support),
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for spec in specs
    ]
    for source_key, object_raw_rows in comp._iter_source_groups(query_root / "region_object_unary_rows.csv", int(args.max_sources)):
        scene, window, frame_id, mask_id = source_key
        meta = source_meta.get(source_key)
        nodes = region_nodes.get(source_key, {})
        if not meta or not nodes:
            continue
        mask_path = base._resolve(meta.get("mask_path", ""))
        frame_key = (scene, int(frame_id))
        if frame_key not in label_cache:
            label_cache[frame_key] = base._read_label(mask_path)
        source_mask = label_cache[frame_key] == int(mask_id)
        if not np.any(source_mask):
            continue
        frame_writer.ensure_frame(scene, int(frame_id), source_mask.shape)
        source_area = float(max(1, int(np.count_nonzero(source_mask))))
        object_rows = comp._prepare_object_rows(object_raw_rows, nodes, source_area, allowed_objects)
        if not object_rows:
            continue
        processed_sources += 1
        region_indices = sorted({int(row["region_index"]) for rows in object_rows.values() for row in rows})
        for spec in specs:
            variant_id = spec["variant_id"]
            comps = _components(region_indices, edge_adjacency.get(source_key, {}), spec)
            selected_by_object, comp_stats = _assign_components(object_rows, comps, spec)
            component_rows.append(
                {
                    "schema_version": "stream4d_v95_phase4D_component_summary_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "variant_id": variant_id,
                    "scene_id": scene,
                    "window_id": window,
                    "frame_id": int(frame_id),
                    "source_mask_id": int(mask_id),
                    **comp_stats,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
            for object_id, selected in selected_by_object.items():
                if not selected:
                    continue
                rows = object_rows[object_id]
                mask = base._node_mask(nodes, selected, source_mask)
                new_id = frame_writer.add_mask(variant_id, mask)
                if new_id <= 0:
                    continue
                selected_area = int(np.count_nonzero(mask))
                if selected_area <= 0:
                    continue
                core = {int(row["region_index"]) for row in rows if base._bool(row.get("is_core_region"))}
                core_retention = len(core & selected) / max(1, len(core))
                area_ratio = selected_area / source_area
                area_ratios_by_variant[variant_id].append(float(area_ratio))
                core_retention_by_variant[variant_id].append(float(core_retention))
                object_score = _object_score(rows, selected)
                gen_path = out / "generated_masks" / variant_id / scene / "mask" / f"{int(frame_id)}.png"
                generated_rows.append(
                    {
                        "schema_version": "stream4d_v95_phase4D_generated_mask_v1",
                        "phase_id": PHASE_ID,
                        "run_id": RUN_ID,
                        "variant_id": variant_id,
                        "scene_id": scene,
                        "window_id": window,
                        "frame_id": int(frame_id),
                        "source_mask_id": int(mask_id),
                        "object_id": object_id,
                        "new_mask_id": int(new_id),
                        "generated_mask_path": base._rel(gen_path),
                        "source_mask_area": int(source_area),
                        "generated_mask_area": int(selected_area),
                        "generated_area_ratio": float(area_ratio),
                        "selected_region_count": int(len(selected)),
                        "core_region_count": int(len(core)),
                        "core_retention_rate": float(core_retention),
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )
                mv_object_id = f"{variant_id}:{object_id}"
                mv_rows.append(
                    {
                        "split": "dev",
                        "scene_id": scene,
                        "source_variant": variant_id,
                        "variant": variant_id,
                        "mv_object_id": mv_object_id,
                        "frame_id": int(frame_id),
                        "mask_id": int(new_id),
                        "frame_mask_score": object_score,
                        "object_score": object_score,
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                        "uses_rgbd_pose_mesh": False,
                        "materializable": True,
                        "selection_reason": f"v95_phase4D_component_{spec['assignment_mode']}_from_{phase3.get('selected_query_family', '')}",
                    }
                )
                ownership_rows.append(
                    {
                        "schema_version": "stream4d_v95_phase4D_ownership_audit_v1",
                        "phase_id": PHASE_ID,
                        "run_id": RUN_ID,
                        "variant_id": variant_id,
                        "scene_id": scene,
                        "window_id": window,
                        "frame_id": int(frame_id),
                        "source_mask_id": int(mask_id),
                        "object_id": object_id,
                        "selected_region_count": int(len(selected)),
                        "total_region_count": int(len(rows)),
                        "generated_area_ratio": float(area_ratio),
                        "core_retention_rate": float(core_retention),
                        "object_score": object_score,
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )
        if int(args.progress_every_sources) > 0 and processed_sources % int(args.progress_every_sources) == 0:
            print(
                json.dumps(
                    {
                        "phase": PHASE_ID,
                        "processed_sources": processed_sources,
                        "generated_mask_rows": len(generated_rows),
                        "elapsed_sec": time.time() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    frame_writer.flush()
    base._write_csv(out / "variant_config_rows.csv", config_rows)
    base._write_csv(out / "generated_mask_rows.csv", generated_rows)
    base._write_csv(out / "mv_object_rows.csv", [{"variant_id": row["variant"], "mv_object_id": row["mv_object_id"], "uses_gt_for_prediction": False, "uses_future": False} for row in mv_rows])
    base._write_csv(out / "mv_object_frame_mask_rows.csv", mv_rows)
    base._write_csv(out / "ownership_audit_rows.csv", ownership_rows)
    base._write_csv(out / "component_rows.csv", component_rows)
    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    if not bool(args.skip_eval):
        radius_sweep.OUT = out
        for spec in specs:
            variant_id = spec["variant_id"]
            rows = [row for row in mv_rows if row.get("variant") == variant_id]
            metrics, cases = radius_sweep._evaluate_variant(variant_id, rows)
            metric_rows.extend(metrics)
            case_rows.extend({**case, "variant_id": variant_id} for case in cases)
    aggregate_rows = phase7d._aggregate(metric_rows) if metric_rows else []
    aggregate_by_variant = {row.get("variant_id", ""): row for row in aggregate_rows}
    variant_metric_rows: list[dict[str, Any]] = []
    for spec in specs:
        variant_id = spec["variant_id"]
        agg = aggregate_by_variant.get(variant_id, {})
        variant_metric_rows.append(
            {
                "schema_version": "stream4d_v95_phase4D_variant_metric_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": variant_id,
                "family": spec["family"],
                "mean_MV_AP_window": agg.get("mean_MV_AP_window", ""),
                "mean_MV_AP50_window": agg.get("mean_MV_AP50_window", ""),
                "mean_MV_AP25_window": agg.get("mean_MV_AP25_window", ""),
                "mean_score_free_Match50_window": agg.get("mean_score_free_Match50_window", ""),
                "mean_generated_area_ratio": base._mean(area_ratios_by_variant.get(variant_id, [])),
                "generated_area_ratio_p10": base._percentile(area_ratios_by_variant.get(variant_id, []), 10),
                "generated_area_ratio_p90": base._percentile(area_ratios_by_variant.get(variant_id, []), 90),
                "core_retention_rate": base._mean(core_retention_by_variant.get(variant_id, [])),
                "same_frame_collision_count": agg.get("same_frame_collision_count", 0),
                "missing_mask_raster_count": agg.get("missing_mask_raster_count", 0),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    gate_rows, gate_failure_rows, any_pass, best_real = _gate_rows(variant_metric_rows, specs, phase0) if metric_rows else ([], [], False, {})
    failure_rows.extend(gate_failure_rows)
    base._write_csv(out / "mv_metric_rows.csv", metric_rows)
    base._write_csv(out / "mv_metric_aggregate_rows.csv", aggregate_rows)
    base._write_csv(out / "variant_metric_rows.csv", variant_metric_rows)
    base._write_csv(out / "variant_gate_rows.csv", gate_rows)
    base._write_csv(out / "variant_failure_rows.csv", failure_rows)
    base._write_csv(out / "casebook_rows.csv", case_rows)
    base._write_csv(out / "mv_iou_matrix_rows.csv", [])
    base._write_csv(out / "scorefree_match_rows.csv", [])
    summary = {
        "schema": "stream4d_v95_phase4D_component_assignment_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": base._created_at(),
        "decision": "PASS_V95_PHASE4_FAMILY_D_CANDIDATE_GATE" if any_pass else ("SMOKE_V95_PHASE4D_MATERIALIZATION_ONLY" if bool(args.skip_eval) else "NO_GO_V95_PHASE4_FAMILY_D_NO_CANDIDATE_GATE"),
        "family": "D_component_first_pooling",
        "query_root": base._rel(query_root),
        "selected_query_family": phase3.get("selected_query_family", ""),
        "min_query_support": int(args.min_query_support),
        "allowed_object_count": "" if allowed_objects is None else len(allowed_objects),
        "processed_sources": processed_sources,
        "best_real_variant_id": best_real.get("variant_id", ""),
        "best_real_MV_AP_window": best_real.get("mean_MV_AP_window", ""),
        "best_real_MV_AP50_window": best_real.get("mean_MV_AP50_window", ""),
        "best_real_MV_AP25_window": best_real.get("mean_MV_AP25_window", ""),
        "best_real_ScoreFreeMatch50_window": best_real.get("mean_score_free_Match50_window", ""),
        "phase4_candidate_gate_pass": any_pass,
        "uses_gt_for_prediction_count": 0,
        "uses_future_count": 0,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "duration_sec": time.time() - started,
        "row_counts": {
            "variant_config_rows": len(config_rows),
            "generated_mask_rows": len(generated_rows),
            "mv_object_rows": len(mv_rows),
            "mv_object_frame_mask_rows": len(mv_rows),
            "ownership_audit_rows": len(ownership_rows),
            "component_rows": len(component_rows),
            "mv_metric_rows": len(metric_rows),
            "mv_metric_aggregate_rows": len(aggregate_rows),
            "variant_metric_rows": len(variant_metric_rows),
            "variant_gate_rows": len(gate_rows),
            "variant_failure_rows": len(failure_rows),
            "casebook_rows": len(case_rows),
        },
    }
    base._write_json(out / "summary.json", summary)
    outputs = [
        out / "variant_config_rows.csv",
        out / "generated_mask_rows.csv",
        out / "mv_object_rows.csv",
        out / "mv_object_frame_mask_rows.csv",
        out / "ownership_audit_rows.csv",
        out / "component_rows.csv",
        out / "mv_metric_rows.csv",
        out / "mv_metric_aggregate_rows.csv",
        out / "variant_metric_rows.csv",
        out / "variant_gate_rows.csv",
        out / "variant_failure_rows.csv",
        out / "casebook_rows.csv",
        out / "summary.json",
    ]
    base._write_json(out / "SHA256SUMS.json", {base._rel(path): base._sha256(path) for path in outputs if path.exists()})
    print(json.dumps(base._jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-root", default=str(PHASE3))
    parser.add_argument("--output-root", default=str(OUT))
    parser.add_argument("--source-container-rows", default=str(PHASE1 / "source_container_rows.csv"))
    parser.add_argument("--region-node-rows", default=str(PHASE1 / "region_node_rows.csv"))
    parser.add_argument("--region-edge-rows", default=str(PHASE1 / "region_edge_rows.csv"))
    parser.add_argument("--min-query-support", type=int, default=30)
    parser.add_argument("--max-sources", type=int, default=0)
    parser.add_argument("--progress-every-sources", type=int, default=512)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
