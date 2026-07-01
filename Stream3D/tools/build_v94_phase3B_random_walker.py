#!/usr/bin/env python3
"""Run v94 Phase3B seeded random-walker / harmonic propagation variants."""

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

from tools import build_v93_phase5b_unknown_background_field as base  # noqa: E402
from tools import build_v94_phase3A_greedy_assignment as phase3a  # noqa: E402


PHASE_ID = "v94_phase3B_random_walker"
RUN_ID = "v94_phase3B_random_walker_gpu"
OUT = ROOT / "outputs/audit/v94_phase3B_random_walker"
V94_PHASE0 = ROOT / "outputs/audit/v94_phase0_fact_lock"
V94_PHASE3A_REPAIR = ROOT / "outputs/audit/v94_phase3A_greedy_assignment_edge_repair"
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
            "variant_id": "B0_current_whole_source_replay",
            "family": "baseline",
            "mode": "whole",
            "base_variant": "F0_whole_source_baseline",
            "description": "Whole-source replay baseline in the Phase3B evaluator path.",
        },
        {
            "variant_id": "B1_random_walker_d4rt_seeds_radio_edges",
            "family": "real",
            "mode": "v94_random_walker",
            "base_variant": "F2_D4RT_RADIO_pairwise",
            "threshold": 0.52,
            "pos_quantile": 0.82,
            "neg_quantile": 0.82,
            "alpha": 0.72,
            "iters": 28,
            "edge_power": 1.0,
            "edge_barrier": 0.04,
            "area_cap": 0.92,
            "min_area_fraction": 0.55,
            "description": "D4RT/prototype positive seeds propagated on RADIO+D4RT pairwise edges.",
        },
        {
            "variant_id": "B2_random_walker_soft_d4rt_seeds",
            "family": "real",
            "mode": "v94_random_walker",
            "base_variant": "F2_D4RT_RADIO_pairwise",
            "threshold": 0.48,
            "pos_quantile": 0.76,
            "neg_quantile": 0.86,
            "alpha": 0.68,
            "iters": 32,
            "edge_power": 1.0,
            "edge_barrier": 0.02,
            "area_cap": 0.95,
            "min_area_fraction": 0.66,
            "description": "Softer D4RT seeds with weaker clamping and higher recall.",
        },
        {
            "variant_id": "B3_random_walker_with_mask_edge_barrier",
            "family": "real",
            "mode": "v94_random_walker",
            "base_variant": "F4_RADIO_edge_barrier",
            "threshold": 0.50,
            "pos_quantile": 0.80,
            "neg_quantile": 0.78,
            "alpha": 0.70,
            "iters": 30,
            "edge_power": 1.6,
            "edge_barrier": 0.16,
            "area_cap": 0.88,
            "min_area_fraction": 0.50,
            "description": "Random walker with stronger mask-edge barrier in the graph affinity.",
        },
        {
            "variant_id": "B4_random_walker_with_competing_edges",
            "family": "real",
            "mode": "v94_random_walker",
            "base_variant": "F2_D4RT_RADIO_pairwise",
            "threshold": 0.50,
            "pos_quantile": 0.80,
            "neg_quantile": 0.76,
            "alpha": 0.70,
            "iters": 30,
            "edge_power": 1.2,
            "edge_barrier": 0.08,
            "competing_negative": 0.22,
            "area_cap": 0.88,
            "min_area_fraction": 0.52,
            "description": "Competing/nested edge risk becomes stronger negative seed evidence.",
        },
        {
            "variant_id": "B5_random_walker_bg_unknown_labels",
            "family": "real",
            "mode": "v94_random_walker",
            "base_variant": "F2_D4RT_RADIO_pairwise",
            "threshold": 0.56,
            "pos_quantile": 0.84,
            "neg_quantile": 0.70,
            "alpha": 0.64,
            "iters": 28,
            "edge_power": 1.0,
            "edge_barrier": 0.12,
            "background_negative": 0.25,
            "area_cap": 0.82,
            "min_area_fraction": 0.40,
            "description": "Conservative random walker with background/unknown-like negative seeds.",
        },
        {
            "variant_id": "B6_random_walker_container_component_limited",
            "family": "real",
            "mode": "v94_random_walker",
            "base_variant": "F2_D4RT_RADIO_pairwise",
            "threshold": 0.46,
            "pos_quantile": 0.74,
            "neg_quantile": 0.88,
            "alpha": 0.78,
            "iters": 36,
            "edge_power": 1.35,
            "edge_barrier": 0.03,
            "area_cap": 0.96,
            "min_area_fraction": 0.72,
            "description": "High-recall propagation with component-aware diagnostics and mild area limiting.",
        },
        {
            "variant_id": "B7_random_walker_low_clamp_high_recall",
            "family": "real",
            "mode": "v94_random_walker",
            "base_variant": "F2_D4RT_RADIO_pairwise",
            "threshold": 0.38,
            "pos_quantile": 0.70,
            "neg_quantile": 0.93,
            "alpha": 0.84,
            "iters": 42,
            "edge_power": 0.85,
            "edge_barrier": 0.01,
            "background_negative": 0.04,
            "competing_negative": 0.03,
            "area_cap": 0.985,
            "min_area_fraction": 0.84,
            "description": "Repair for seed-stuck propagation: weak negative clamp and high-recall source-preserving diffusion.",
        },
        {
            "variant_id": "B8_random_walker_unary_residual_source_preserve",
            "family": "real",
            "mode": "v94_random_walker",
            "base_variant": "F2_D4RT_RADIO_pairwise",
            "threshold": 0.34,
            "pos_quantile": 0.68,
            "neg_quantile": 0.95,
            "alpha": 0.58,
            "iters": 36,
            "edge_power": 0.75,
            "edge_barrier": 0.00,
            "background_negative": 0.03,
            "competing_negative": 0.02,
            "area_cap": 0.995,
            "min_area_fraction": 0.88,
            "description": "Repair for undercoverage: preserve unary residual and fill high-recall source support unless evidence is strongly negative.",
        },
    ]


def _edge_affinity(
    *,
    edge_weight: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    boundary: Any,
    base_variant_idx: int,
    spec: dict[str, Any],
    device: Any,
) -> Any:
    torch = base.torch
    if edge_u.size == 0:
        return torch.zeros(0, dtype=torch.float32, device=device)
    if edge_weight.ndim == 2 and edge_weight.shape[1] > int(base_variant_idx):
        raw = edge_weight[:, int(base_variant_idx)]
    elif edge_weight.ndim == 1:
        raw = edge_weight
    else:
        raw = np.ones(edge_u.shape[0], dtype=np.float32)
    e_w = torch.as_tensor(raw, dtype=torch.float32, device=device)
    e_w = torch.nan_to_num(e_w, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    power = float(spec.get("edge_power", 1.0))
    if power != 1.0:
        e_w = torch.pow(e_w, power)
    e_u = torch.as_tensor(edge_u, dtype=torch.long, device=device)
    e_v = torch.as_tensor(edge_v, dtype=torch.long, device=device)
    barrier = float(spec.get("edge_barrier", 0.0))
    if barrier > 0.0:
        local_boundary = torch.maximum(boundary[e_u], boundary[e_v])
        e_w = e_w * torch.clamp(1.0 - barrier * local_boundary, min=0.0, max=1.0)
    return e_w


def _weighted_neighbor_average(p: Any, edge_u_t: Any, edge_v_t: Any, edge_w_t: Any) -> Any:
    torch = base.torch
    n = int(p.numel())
    if n == 0 or edge_u_t.numel() == 0:
        return p
    accum = torch.zeros(n, dtype=torch.float32, device=p.device)
    denom = torch.zeros(n, dtype=torch.float32, device=p.device)
    accum.index_add_(0, edge_u_t, edge_w_t * p[edge_v_t])
    accum.index_add_(0, edge_v_t, edge_w_t * p[edge_u_t])
    denom.index_add_(0, edge_u_t, edge_w_t)
    denom.index_add_(0, edge_v_t, edge_w_t)
    return torch.where(denom > 1.0e-6, accum / torch.clamp(denom, min=1.0e-6), p)


def _selected_component_count(selected: set[int], edge_u: np.ndarray, edge_v: np.ndarray, edge_weight: np.ndarray, base_variant_idx: int) -> int:
    if not selected:
        return 0
    parent = {idx: idx for idx in selected}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if edge_weight.ndim == 2 and edge_weight.shape[1] > int(base_variant_idx):
        weights = edge_weight[:, int(base_variant_idx)]
    elif edge_weight.ndim == 1:
        weights = edge_weight
    else:
        weights = np.ones(edge_u.shape[0], dtype=np.float32)
    for u_raw, v_raw, w_raw in zip(edge_u.tolist(), edge_v.tolist(), weights.tolist()):
        u = int(u_raw)
        v = int(v_raw)
        if float(w_raw) >= 0.50 and u in parent and v in parent:
            union(u, v)
    return len({find(idx) for idx in selected})


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
        return set(range(n)), np.ones(n, dtype=np.float32), {"component_count_proxy": 1.0, "solver_success": 1.0}

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

    base_unary = (
        0.48 * p
        + 0.20 * d4rt
        + 0.18 * sem
        + 0.10 * inside
        - float(spec.get("background_negative", 0.10)) * negative
        - float(spec.get("competing_negative", 0.08)) * competing_bar
        - 0.04 * nested_bar
    )
    base_unary = torch.nan_to_num(base_unary, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    pos_score = d4rt + 0.45 * sem + 0.20 * inside
    neg_score = negative + 0.45 * competing_bar + 0.25 * source_bar + 0.18 * nested_bar
    pos_thr = torch.quantile(pos_score, float(spec.get("pos_quantile", 0.80)))
    neg_thr = torch.quantile(neg_score, float(spec.get("neg_quantile", 0.80)))
    pos_seed = pos_score >= pos_thr
    neg_seed = (neg_score >= neg_thr) & (~pos_seed)
    if not bool(pos_seed.any()):
        pos_seed[int(torch.argmax(pos_score).detach().cpu())] = True
    if not bool(neg_seed.any()) and n > 1:
        neg_seed[int(torch.argmax(neg_score).detach().cpu())] = True
    conflict_seed = pos_seed & neg_seed
    neg_seed = neg_seed & (~pos_seed)

    e_u_t = torch.as_tensor(edge_u, dtype=torch.long, device=device)
    e_v_t = torch.as_tensor(edge_v, dtype=torch.long, device=device)
    e_w_t = _edge_affinity(
        edge_weight=edge_weight,
        edge_u=edge_u,
        edge_v=edge_v,
        boundary=boundary,
        base_variant_idx=base_variant_idx,
        spec=spec,
        device=device,
    )

    alpha = float(spec.get("alpha", 0.70))
    value = base_unary.clone()
    value[pos_seed] = 1.0
    value[neg_seed] = 0.0
    residual = torch.tensor(0.0, dtype=torch.float32, device=device)
    iters = int(spec.get("iters", 24))
    for _ in range(max(1, iters)):
        neighbor = _weighted_neighbor_average(value, e_u_t, e_v_t, e_w_t)
        new_value = alpha * neighbor + (1.0 - alpha) * base_unary
        new_value[pos_seed] = 1.0
        new_value[neg_seed] = 0.0
        residual = torch.mean(torch.abs(new_value - value))
        value = new_value
        if float(residual.detach().cpu()) < 1.0e-4:
            break

    score_np = value.detach().cpu().numpy()
    selected = {int(i) for i in torch.nonzero(value >= float(spec.get("threshold", 0.5)), as_tuple=False).flatten().detach().cpu().tolist()}
    must_keep = {int(i) for i in torch.nonzero(pos_seed, as_tuple=False).flatten().detach().cpu().tolist()}
    selected |= must_keep
    selected = base._enforce_min_area(selected, nodes, float(spec.get("min_area_fraction", 0.0)), score_np)
    selected = base._cap_by_area(selected, nodes, float(spec.get("area_cap", 1.0)), score_np, must_keep)
    component_count = _selected_component_count(selected, edge_u, edge_v, edge_weight, base_variant_idx)
    diagnostics = {
        "solver_success": 1.0 if np.all(np.isfinite(score_np)) else 0.0,
        "solver_runtime_ms": 1000.0 * (time.perf_counter() - started),
        "solver_iteration_count": float(iters),
        "solver_residual": float(residual.detach().cpu()),
        "seed_label_coverage": float((torch.count_nonzero(pos_seed) + torch.count_nonzero(neg_seed)).detach().cpu()) / max(1, n),
        "positive_seed_fraction": float(torch.count_nonzero(pos_seed).detach().cpu()) / max(1, n),
        "negative_seed_fraction": float(torch.count_nonzero(neg_seed).detach().cpu()) / max(1, n),
        "seed_conflict_rate": float(torch.count_nonzero(conflict_seed).detach().cpu()) / max(1, n),
        "connected_component_split_count": float(component_count),
        "component_count_proxy": float(component_count),
        "random_walker_score_mean": float(torch.mean(value).detach().cpu()),
        "edge_affinity_mean": float(torch.mean(e_w_t).detach().cpu()) if e_w_t.numel() else 0.0,
    }
    return selected, score_np, diagnostics


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def _phase3b_gate_rows(metric_rows: list[dict[str, str]], specs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool, dict[str, Any]]:
    phase0 = _read_json(V94_PHASE0 / "summary.json")
    phase3a = _read_json(V94_PHASE3A_REPAIR / "summary.json")
    best_a_ap = _num(phase3a.get("best_A_MV_AP_window"))
    best_a_ap50 = _num(phase3a.get("best_A_MV_AP50_window"))
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
        improvement_gate = (mv_ap >= best_a_ap + 0.003) or (mv_ap50 >= best_a_ap50 + 0.006)
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
                "schema_version": "stream4d_v94_phase3B_variant_gate_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": variant_id,
                "family": spec.get("family", ""),
                "MV_AP_window": mv_ap,
                "MV_AP50_window": mv_ap50,
                "best_A_MV_AP_window": best_a_ap,
                "best_A_MV_AP50_window": best_a_ap50,
                "best_control_MV_AP_window": control_ap,
                "best_control_MV_AP50_window": control_ap50,
                "phase3B_improvement_gate_pass": improvement_gate,
                "locked_control_gate_pass": locked_control_gate,
                "provenance_gate_pass": provenance_gate,
                "phase3B_candidate_gate_pass": pass_gate,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        if is_real and not pass_gate:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v94_phase3B_failure_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "variant_id": variant_id,
                    "failure_type": "PHASE3B_RANDOM_WALKER_NO_CANDIDATE_GATE",
                    "MV_AP_window": mv_ap,
                    "MV_AP50_window": mv_ap50,
                    "best_A_MV_AP_window": best_a_ap,
                    "best_A_MV_AP50_window": best_a_ap50,
                    "best_control_MV_AP_window": control_ap,
                    "best_control_MV_AP50_window": control_ap50,
                    "repair_direction": "If propagation becomes whole-source, strengthen edge/RADIO contrast; if stuck at seeds, reduce seed clamp or raise pairwise alpha; if B family stays flat, enter C/D.",
                    "created_at": _created_at(),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    return gate_rows, failure_rows, any_pass, best_real


def _postprocess(out: Path, started: float) -> dict[str, Any]:
    specs = _variant_specs()
    phase3a._rewrite_csv_schema(out / "variant_config_rows.csv", "stream4d_v94_phase3B_variant_config_v1")
    phase3a._rewrite_csv_schema(out / "generated_mask_rows.csv", "stream4d_v94_phase3B_generated_mask_v1")
    phase3a._rewrite_csv_schema(out / "assignment_summary_rows.csv", "stream4d_v94_phase3B_assignment_summary_v1")
    phase3a._rewrite_csv_schema(out / "source_failure_rows.csv", "stream4d_v94_phase3B_source_failure_v1")
    phase3a._rewrite_csv_schema(out / "variant_metric_rows.csv", "stream4d_v94_phase3B_variant_metric_v1")
    metric_rows = _read_csv(out / "variant_metric_rows.csv")
    gate_rows, failure_rows, any_pass, best_real = _phase3b_gate_rows(metric_rows, specs)
    _write_csv(out / "variant_gate_rows.csv", gate_rows)
    _write_csv(out / "variant_failure_rows.csv", failure_rows)

    assignment_rows = _read_csv(out / "assignment_summary_rows.csv")
    by_variant: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in assignment_rows:
        by_variant[str(row.get("variant_id", ""))].append(row)
    solver_rows: list[dict[str, Any]] = []
    for spec in specs:
        rows = by_variant.get(spec["variant_id"], [])
        solver_rows.append(
            {
                "schema_version": "stream4d_v94_phase3B_solver_summary_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": spec["variant_id"],
                "family": spec["family"],
                "solver_success_rate": _mean([_num(row.get("solver_success")) for row in rows]),
                "solver_runtime_ms_mean": _mean([_num(row.get("solver_runtime_ms")) for row in rows]),
                "solver_iteration_count_mean": _mean([_num(row.get("solver_iteration_count")) for row in rows]),
                "solver_residual_mean": _mean([_num(row.get("solver_residual")) for row in rows]),
                "seed_label_coverage_mean": _mean([_num(row.get("seed_label_coverage")) for row in rows]),
                "seed_conflict_rate_mean": _mean([_num(row.get("seed_conflict_rate")) for row in rows]),
                "connected_component_split_count_mean": _mean([_num(row.get("connected_component_split_count")) for row in rows]),
                "region_count_skipped_due_large_source": 0,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    _write_csv(out / "solver_rows.csv", solver_rows)
    _write_csv(
        out / "random_walker_config_rows.csv",
        [
            {
                "schema_version": "stream4d_v94_phase3B_random_walker_config_v1",
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
            "schema_version": "stream4d_v94_phase3B_component_summary_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "variant_id": row.get("variant_id", ""),
            "scene_id": row.get("scene_id", ""),
            "frame_id": row.get("frame_id", ""),
            "source_mask_id": row.get("source_mask_id", ""),
            "component_extraction_mode": "selected_region_graph_components_weight_ge_0.50_proxy",
            "component_count_proxy": row.get("component_count_proxy", ""),
            "connected_component_split_count": row.get("connected_component_split_count", ""),
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
            "schema": "stream4d_v94_phase3B_random_walker_summary_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "decision": "PASS_V94_PHASE3B_CANDIDATE_GATE" if any_pass else "NO_GO_V94_PHASE3B_RANDOM_WALKER_NO_CANDIDATE_GATE",
            "any_phase3B_candidate_gate_pass": any_pass,
            "best_B_variant_id": best_real.get("variant_id", ""),
            "best_B_MV_AP_window": best_real.get("mean_MV_AP_window", ""),
            "best_B_MV_AP50_window": best_real.get("mean_MV_AP50_window", ""),
            "best_A_reference_artifact": str(V94_PHASE3A_REPAIR.relative_to(ROOT) / "summary.json"),
            "field_artifact_mode": "v93_npz_field_shards_reused; harmonic propagation runs on GPU tensors, evaluator remains CSV/CPU",
            "duration_sec": time.time() - started,
            "row_counts": {
                **base_summary.get("row_counts", {}),
                "solver_rows": len(solver_rows),
                "random_walker_config_rows": len(specs),
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
