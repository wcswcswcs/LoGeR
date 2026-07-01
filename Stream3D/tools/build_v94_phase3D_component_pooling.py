#!/usr/bin/env python3
"""Run v94 Phase3D component pooling variants."""

from __future__ import annotations

import argparse
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

from tools import build_v93_phase5b_unknown_background_field as base  # noqa: E402
from tools import build_v94_phase3A_greedy_assignment as phase3a  # noqa: E402
from tools import build_v94_phase3B_random_walker as phase3b  # noqa: E402


PHASE_ID = "v94_phase3D_component_pooling"
RUN_ID = "v94_phase3D_component_pooling_gpu"
OUT = ROOT / "outputs/audit/v94_phase3D_component_pooling"
V94_PHASE0 = ROOT / "outputs/audit/v94_phase0_fact_lock"
V94_PHASE3A_REPAIR = ROOT / "outputs/audit/v94_phase3A_greedy_assignment_edge_repair"
V94_PHASE3B = ROOT / "outputs/audit/v94_phase3B_random_walker"
V94_PHASE3C = ROOT / "outputs/audit/v94_phase3C_constrained_cut"
V93_FIELD_ROOT = ROOT / "outputs/audit/v93_phase5_boundary_affinity_field"


def _jsonable(value: Any) -> Any:
    return phase3a._jsonable(value)


def _write_json(path: Path, payload: Any) -> None:
    phase3a._write_json(path, payload)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    phase3a._write_csv(path, rows)


def _read_json(path: Path) -> dict[str, Any]:
    return phase3a._read_json(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    return phase3a._read_csv(path)


def _num(value: Any, default: float = 0.0) -> float:
    return phase3a._num(value, default)


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _variant_specs() -> list[dict[str, Any]]:
    return [
        {
            "variant_id": "D0_current_whole_source_replay",
            "family": "baseline",
            "mode": "whole",
            "base_variant": "F0_whole_source_baseline",
            "description": "Whole-source replay baseline in the Phase3D evaluator path.",
        },
        {
            "variant_id": "D1_component_pool_radio_affinity",
            "family": "real",
            "mode": "v94_component_pool",
            "base_variant": "F2_D4RT_RADIO_pairwise",
            "edge_threshold": 0.64,
            "score_threshold": 0.54,
            "component_barrier": 0.02,
            "area_cap": 0.90,
            "min_area_fraction": 0.52,
            "description": "High-affinity RADIO/D4RT components, moderate object score threshold.",
        },
        {
            "variant_id": "D2_component_pool_soft_affinity",
            "family": "real",
            "mode": "v94_component_pool",
            "base_variant": "F2_D4RT_RADIO_pairwise",
            "edge_threshold": 0.56,
            "score_threshold": 0.48,
            "component_barrier": 0.01,
            "area_cap": 0.94,
            "min_area_fraction": 0.66,
            "description": "Softer component pooling with more permissive component edges.",
        },
        {
            "variant_id": "D3_component_pool_source_preserve",
            "family": "real",
            "mode": "v94_component_pool",
            "base_variant": "F2_D4RT_RADIO_pairwise",
            "edge_threshold": 0.48,
            "score_threshold": 0.40,
            "component_barrier": 0.00,
            "area_cap": 0.985,
            "min_area_fraction": 0.84,
            "description": "High-recall source-preserving component pooling.",
        },
        {
            "variant_id": "D4_component_pool_strong_boundary",
            "family": "real",
            "mode": "v94_component_pool",
            "base_variant": "F4_RADIO_edge_barrier",
            "edge_threshold": 0.70,
            "score_threshold": 0.58,
            "component_barrier": 0.22,
            "area_cap": 0.84,
            "min_area_fraction": 0.42,
            "description": "Strong boundary component split to test whether component cuts improve AP50.",
        },
        {
            "variant_id": "D5_component_pool_large_merge",
            "family": "real",
            "mode": "v94_component_pool",
            "base_variant": "F2_D4RT_RADIO_pairwise",
            "edge_threshold": 0.42,
            "score_threshold": 0.34,
            "component_barrier": 0.00,
            "area_cap": 0.995,
            "min_area_fraction": 0.90,
            "description": "Large merged components and very high recall to test over-fragmentation as the blocker.",
        },
        {
            "variant_id": "D6_component_pool_negative_veto",
            "family": "real",
            "mode": "v94_component_pool",
            "base_variant": "F2_D4RT_RADIO_pairwise",
            "edge_threshold": 0.54,
            "score_threshold": 0.50,
            "component_barrier": 0.04,
            "negative_veto": 0.22,
            "area_cap": 0.90,
            "min_area_fraction": 0.56,
            "description": "Component pooling with D4RT/negative veto.",
        },
    ]


def _component_labels(n: int, edge_u: np.ndarray, edge_v: np.ndarray, weights: np.ndarray, threshold: float) -> tuple[np.ndarray, int]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for u_raw, v_raw, w_raw in zip(edge_u.tolist(), edge_v.tolist(), weights.tolist()):
        if float(w_raw) >= threshold:
            u = int(u_raw)
            v = int(v_raw)
            if 0 <= u < n and 0 <= v < n:
                union(u, v)
    roots: dict[int, int] = {}
    labels = np.zeros(n, dtype=np.int32)
    for idx in range(n):
        root = find(idx)
        label = roots.setdefault(root, len(roots))
        labels[idx] = label
    return labels, len(roots)


def _select_regions(
    *,
    spec: dict[str, Any],
    key: tuple[str, int, int],
    nodes: list[dict[str, Any]],
    features: np.ndarray,
    prob: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_weight: np.ndarray,
    base_variant_idx: int,
    reference_selected: set[int],
    device: Any,
) -> tuple[set[int], np.ndarray, dict[str, float]]:
    del key, reference_selected
    torch = base.torch
    n = len(nodes)
    if n == 0:
        return set(), np.zeros(0, dtype=np.float32), {}
    if str(spec.get("mode")) == "whole":
        return set(range(n)), np.ones(n, dtype=np.float32), {"component_pool_count": 1.0, "component_selected_count": 1.0}

    started = time.perf_counter()
    sem = torch.as_tensor(features[:, 0], dtype=torch.float32, device=device)
    d4rt = torch.as_tensor(features[:, 1], dtype=torch.float32, device=device)
    inside = torch.as_tensor(features[:, 2], dtype=torch.float32, device=device)
    source_bar = torch.as_tensor(features[:, 3], dtype=torch.float32, device=device)
    nested_bar = torch.as_tensor(features[:, 4], dtype=torch.float32, device=device)
    competing_bar = torch.as_tensor(features[:, 5], dtype=torch.float32, device=device)
    negative = torch.as_tensor(features[:, 6], dtype=torch.float32, device=device)
    p = torch.as_tensor(prob, dtype=torch.float32, device=device)
    boundary = torch.maximum(source_bar, torch.maximum(nested_bar, competing_bar))

    node_score = (
        0.50 * p
        + 0.20 * d4rt
        + 0.18 * sem
        + 0.10 * inside
        - float(spec.get("component_barrier", 0.0)) * boundary
        - float(spec.get("negative_veto", 0.08)) * negative
        - 0.04 * competing_bar
    )
    score_np = torch.nan_to_num(node_score, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0).detach().cpu().numpy()
    if edge_weight.ndim == 2 and edge_weight.shape[1] > int(base_variant_idx):
        weights = edge_weight[:, int(base_variant_idx)].astype(np.float32)
    elif edge_weight.ndim == 1:
        weights = edge_weight.astype(np.float32)
    else:
        weights = np.ones(edge_u.shape[0], dtype=np.float32)
    labels, comp_count = _component_labels(n, edge_u, edge_v, weights, float(spec.get("edge_threshold", 0.5)))
    pixel_area = np.array([max(1, int(float(row.get("pixel_count", row.get("area_px", 1)) or 1))) for row in nodes], dtype=np.float64)
    seed_score = features[:, 1] + 0.45 * features[:, 0] + 0.20 * features[:, 2]
    seed_mask = seed_score >= float(np.quantile(seed_score, 0.82)) if n else np.zeros(0, dtype=bool)
    selected: set[int] = set()
    comp_scores: list[float] = []
    comp_areas: list[float] = []
    selected_comps = 0
    for comp_id in range(comp_count):
        comp_idx = np.nonzero(labels == comp_id)[0]
        if comp_idx.size == 0:
            continue
        comp_area = float(np.sum(pixel_area[comp_idx]))
        comp_score = float(np.average(score_np[comp_idx], weights=pixel_area[comp_idx]))
        comp_seed = bool(np.any(seed_mask[comp_idx]))
        comp_scores.append(comp_score)
        comp_areas.append(comp_area)
        if comp_score >= float(spec.get("score_threshold", 0.5)) or comp_seed:
            selected.update(int(i) for i in comp_idx.tolist())
            selected_comps += 1
    must_keep = {int(i) for i, keep in enumerate(seed_mask.tolist()) if keep}
    selected |= must_keep
    selected = base._enforce_min_area(selected, nodes, float(spec.get("min_area_fraction", 0.0)), score_np)
    selected = base._cap_by_area(selected, nodes, float(spec.get("area_cap", 1.0)), score_np, must_keep)
    selected_comp_count = len({int(labels[i]) for i in selected}) if selected else 0
    diagnostics = {
        "component_pool_count": float(comp_count),
        "component_selected_count": float(selected_comp_count),
        "component_selected_initial_count": float(selected_comps),
        "component_score_mean": float(np.mean(comp_scores)) if comp_scores else 0.0,
        "component_score_max": float(np.max(comp_scores)) if comp_scores else 0.0,
        "component_area_mean": float(np.mean(comp_areas)) if comp_areas else 0.0,
        "component_max_area_fraction": float(np.max(comp_areas) / max(1.0, np.sum(pixel_area))) if comp_areas else 0.0,
        "component_pool_runtime_ms": 1000.0 * (time.perf_counter() - started),
        "component_count_proxy": float(selected_comp_count),
        "positive_seed_fraction": float(np.mean(seed_mask)) if n else 0.0,
        "score_mean": float(np.mean(score_np)) if score_np.size else 0.0,
    }
    return selected, score_np, diagnostics


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def _phase3d_gate_rows(metric_rows: list[dict[str, str]], specs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool, dict[str, Any]]:
    phase0 = _read_json(V94_PHASE0 / "summary.json")
    phase3a = _read_json(V94_PHASE3A_REPAIR / "summary.json")
    phase3b_summary = _read_json(V94_PHASE3B / "summary.json")
    phase3c_summary = _read_json(V94_PHASE3C / "summary.json")
    best_prev_ap = max(_num(phase3a.get("best_A_MV_AP_window")), _num(phase3b_summary.get("best_B_MV_AP_window")), _num(phase3c_summary.get("best_C_MV_AP_window")))
    best_prev_ap50 = max(
        _num(phase3a.get("best_A_MV_AP50_window")),
        _num(phase3b_summary.get("best_B_MV_AP50_window")),
        _num(phase3c_summary.get("best_C_MV_AP50_window")),
    )
    control_ap = _num(phase0.get("best_control_MV_AP_window"))
    control_ap50 = _num(phase0.get("best_control_MV_AP50_window"))
    spec_by_id = {spec["variant_id"]: spec for spec in specs}
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    any_pass = False
    best_real: dict[str, Any] = {}
    for row in metric_rows:
        variant_id = row.get("variant_id", "")
        spec = spec_by_id.get(variant_id, {})
        is_real = spec.get("family") == "real"
        mv_ap = _num(row.get("mean_MV_AP_window"))
        mv_ap50 = _num(row.get("mean_MV_AP50_window"))
        missing = _num(row.get("missing_mask_raster_count"))
        collision = _num(row.get("same_frame_collision_count"))
        improvement_gate = (mv_ap >= best_prev_ap + 0.003) or (mv_ap50 >= best_prev_ap50 + 0.006)
        locked_control_gate = (mv_ap > control_ap) and (mv_ap50 > control_ap50)
        provenance_gate = (
            str(row.get("uses_gt_for_prediction", "False")).lower() == "false"
            and str(row.get("uses_future", "False")).lower() == "false"
            and missing == 0.0
            and collision == 0.0
        )
        pass_gate = bool(is_real and improvement_gate and locked_control_gate and provenance_gate)
        any_pass = any_pass or pass_gate
        if is_real and (not best_real or (mv_ap, mv_ap50) > (_num(best_real.get("mean_MV_AP_window")), _num(best_real.get("mean_MV_AP50_window")))):
            best_real = dict(row)
        gate_rows.append(
            {
                "schema_version": "stream4d_v94_phase3D_variant_gate_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": variant_id,
                "family": spec.get("family", ""),
                "MV_AP_window": mv_ap,
                "MV_AP50_window": mv_ap50,
                "best_previous_MV_AP_window": best_prev_ap,
                "best_previous_MV_AP50_window": best_prev_ap50,
                "best_control_MV_AP_window": control_ap,
                "best_control_MV_AP50_window": control_ap50,
                "phase3D_improvement_gate_pass": improvement_gate,
                "locked_control_gate_pass": locked_control_gate,
                "provenance_gate_pass": provenance_gate,
                "phase3D_candidate_gate_pass": pass_gate,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        if is_real and not pass_gate:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v94_phase3D_failure_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "variant_id": variant_id,
                    "failure_type": "PHASE3D_COMPONENT_POOLING_NO_CANDIDATE_GATE",
                    "MV_AP_window": mv_ap,
                    "MV_AP50_window": mv_ap50,
                    "best_previous_MV_AP_window": best_prev_ap,
                    "best_previous_MV_AP50_window": best_prev_ap50,
                    "repair_direction": "If component pooling cannot beat A/B/C, enter failure decomposition; only then consider adaptive D4RT sampling.",
                    "created_at": _created_at(),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    return gate_rows, failure_rows, any_pass, best_real


def _postprocess(out: Path, started: float) -> dict[str, Any]:
    specs = _variant_specs()
    phase3a._rewrite_csv_schema(out / "variant_config_rows.csv", "stream4d_v94_phase3D_variant_config_v1")
    phase3a._rewrite_csv_schema(out / "generated_mask_rows.csv", "stream4d_v94_phase3D_generated_mask_v1")
    phase3a._rewrite_csv_schema(out / "assignment_summary_rows.csv", "stream4d_v94_phase3D_assignment_summary_v1")
    phase3a._rewrite_csv_schema(out / "source_failure_rows.csv", "stream4d_v94_phase3D_source_failure_v1")
    phase3a._rewrite_csv_schema(out / "variant_metric_rows.csv", "stream4d_v94_phase3D_variant_metric_v1")
    metric_rows = _read_csv(out / "variant_metric_rows.csv")
    gate_rows, failure_rows, any_pass, best_real = _phase3d_gate_rows(metric_rows, specs)
    _write_csv(out / "variant_gate_rows.csv", gate_rows)
    _write_csv(out / "variant_failure_rows.csv", failure_rows)
    assignment_rows = _read_csv(out / "assignment_summary_rows.csv")
    by_variant: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in assignment_rows:
        by_variant[str(row.get("variant_id", ""))].append(row)
    component_pool_rows: list[dict[str, Any]] = []
    for spec in specs:
        rows = by_variant.get(spec["variant_id"], [])
        component_pool_rows.append(
            {
                "schema_version": "stream4d_v94_phase3D_component_pool_summary_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": spec["variant_id"],
                "family": spec["family"],
                "component_pool_count_mean": _mean([_num(row.get("component_pool_count")) for row in rows]),
                "component_selected_count_mean": _mean([_num(row.get("component_selected_count")) for row in rows]),
                "component_score_mean": _mean([_num(row.get("component_score_mean")) for row in rows]),
                "component_score_max_mean": _mean([_num(row.get("component_score_max")) for row in rows]),
                "component_area_mean": _mean([_num(row.get("component_area_mean")) for row in rows]),
                "component_max_area_fraction_mean": _mean([_num(row.get("component_max_area_fraction")) for row in rows]),
                "component_pool_runtime_ms_mean": _mean([_num(row.get("component_pool_runtime_ms")) for row in rows]),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    _write_csv(out / "component_pool_rows.csv", component_pool_rows)
    _write_csv(
        out / "component_pool_config_rows.csv",
        [
            {
                "schema_version": "stream4d_v94_phase3D_component_pool_config_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                **spec,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
            for spec in specs
        ],
    )
    component_rows = [
        {
            "schema_version": "stream4d_v94_phase3D_component_summary_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "variant_id": row.get("variant_id", ""),
            "scene_id": row.get("scene_id", ""),
            "frame_id": row.get("frame_id", ""),
            "source_mask_id": row.get("source_mask_id", ""),
            "component_extraction_mode": "edge_affinity_connected_components_cpu_dsu_score_gpu_tensor",
            "component_count_proxy": row.get("component_count_proxy", ""),
            "component_pool_count": row.get("component_pool_count", ""),
            "component_selected_count": row.get("component_selected_count", ""),
            "selected_region_count": row.get("selected_region_count", ""),
            "total_region_count": row.get("total_region_count", ""),
            "generated_mask_area_ratio": row.get("generated_mask_area_ratio", ""),
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for row in assignment_rows
    ]
    _write_csv(out / "component_rows.csv", component_rows)
    base_summary = _read_json(out / "summary.json")
    summary = dict(base_summary)
    summary.update(
        {
            "schema": "stream4d_v94_phase3D_component_pooling_summary_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "decision": "PASS_V94_PHASE3D_CANDIDATE_GATE" if any_pass else "NO_GO_V94_PHASE3D_COMPONENT_POOLING_NO_CANDIDATE_GATE",
            "any_phase3D_candidate_gate_pass": any_pass,
            "best_D_variant_id": best_real.get("variant_id", ""),
            "best_D_MV_AP_window": best_real.get("mean_MV_AP_window", ""),
            "best_D_MV_AP50_window": best_real.get("mean_MV_AP50_window", ""),
            "field_artifact_mode": "v93_npz_field_shards_reused; node scoring uses GPU tensors, component connected components use CPU DSU over edge lists, evaluator remains CSV/CPU",
            "duration_sec": time.time() - started,
            "row_counts": {
                **base_summary.get("row_counts", {}),
                "component_pool_rows": len(component_pool_rows),
                "component_pool_config_rows": len(specs),
                "component_rows": len(component_rows),
                "variant_gate_rows": len(gate_rows),
                "variant_failure_rows": len(failure_rows),
            },
        }
    )
    _write_json(out / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUT))
    parser.add_argument("--field-root", default=str(V93_FIELD_ROOT))
    parser.add_argument("--source-container-rows", default=str(ROOT / "outputs/audit/v94_phase1_canonical_graph/container_rows.csv"))
    parser.add_argument("--region-node-rows", default=str(ROOT / "outputs/audit/v94_phase1_canonical_graph/region_node_rows.csv"))
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--progress-every-shards", type=int, default=1)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = Path(args.output_root)
    if out.exists() and args.clean:
        shutil.rmtree(out)
    base.PHASE_ID = PHASE_ID
    base.RUN_ID = RUN_ID
    base.OUT = out
    base._variant_specs = _variant_specs
    base._select_regions = _select_regions
    base_args = argparse.Namespace(
        output_root=args.output_root,
        field_root=args.field_root,
        source_container_rows=args.source_container_rows,
        region_node_rows=args.region_node_rows,
        max_shards=args.max_shards,
        progress_every_shards=args.progress_every_shards,
        clean=args.clean,
    )
    base.run(base_args)
    summary = _postprocess(out, started)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run(parse_args())
