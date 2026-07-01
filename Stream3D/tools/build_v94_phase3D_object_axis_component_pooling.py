#!/usr/bin/env python3
"""Run v94 Phase3D object-axis boundary-aware component pooling."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_v92_phase5_source_container_field as v92field  # noqa: E402
from tools import build_v93_phase5_boundary_affinity_field as phase5  # noqa: E402
from tools import build_v94_phase3A_object_axis_smoke as phase3a_obj  # noqa: E402
from tools import run_v90_dev_extent_score_cross_audit as phase7d  # noqa: E402
from tools import run_v91_phase4_radius_sweep as radius_sweep  # noqa: E402


PHASE_ID = "v94_phase3D_object_axis_component_pooling"
RUN_ID = "v94_phase3D_object_axis_component_pooling"
OUT = ROOT / "outputs/audit/v94_phase3D_object_axis_component_pooling"
DEFAULT_FIELD_ROOT = ROOT / "outputs/audit/v94_phase7c_object_axis_field_full_dev_combined"
V94_PHASE0 = ROOT / "outputs/audit/v94_phase0_fact_lock"
REGION_EDGE_ROWS = ROOT / "outputs/audit/v94_phase1_canonical_graph/region_edge_rows.csv"


def _jsonable(value: Any) -> Any:
    return phase3a_obj._jsonable(value)


def _resolve(raw: str | Path) -> Path:
    return phase3a_obj._resolve(raw)


def _rel(path: Path | str) -> str:
    return phase3a_obj._rel(path)


def _write_json(path: Path, payload: Any) -> None:
    phase3a_obj._write_json(path, payload)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    phase3a_obj._write_csv(path, rows)


def _read_json(path: Path) -> dict[str, Any]:
    return phase3a_obj._read_json(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _num(value: Any, default: float = 0.0) -> float:
    return phase3a_obj._num(value, default)


def _int(value: Any, default: int = -1) -> int:
    return phase3a_obj._int(value, default)


def _mean(values: list[float]) -> float:
    return phase3a_obj._mean(values)


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _variant_specs() -> list[dict[str, Any]]:
    return [
        {
            "variant_id": "OD0_whole_source_per_object_control",
            "family": "baseline",
            "mode": "whole_source",
            "description": "Whole-source per-object control in the object-axis component-pooling evaluator.",
        },
        {
            "variant_id": "OD1_component_pool_radio_affinity",
            "family": "real",
            "mode": "component_pool",
            "score_scale": 12.0,
            "alpha": 0.70,
            "propagation_iters": 20,
            "edge_threshold": 0.62,
            "component_score_threshold": 0.58,
            "d4rt_weight": 0.08,
            "negative_weight": 0.18,
            "boundary_weight": 0.04,
            "max_region_fraction": 0.70,
            "description": "Component pooling on high-affinity boundary-aware components, moderate object threshold.",
        },
        {
            "variant_id": "OD2_component_pool_soft_affinity",
            "family": "real",
            "mode": "component_pool",
            "score_scale": 9.0,
            "alpha": 0.78,
            "propagation_iters": 28,
            "edge_threshold": 0.54,
            "component_score_threshold": 0.50,
            "d4rt_weight": 0.10,
            "negative_weight": 0.14,
            "boundary_weight": 0.02,
            "max_region_fraction": 0.82,
            "description": "Softer component pooling with permissive component edges.",
        },
        {
            "variant_id": "OD3_component_pool_source_preserve",
            "family": "real",
            "mode": "component_pool",
            "score_scale": 8.0,
            "alpha": 0.82,
            "propagation_iters": 32,
            "edge_threshold": 0.44,
            "component_score_threshold": 0.42,
            "d4rt_weight": 0.12,
            "negative_weight": 0.10,
            "boundary_weight": 0.00,
            "max_region_fraction": 0.94,
            "description": "High-recall source-preserving component pooling.",
        },
        {
            "variant_id": "OD4_component_pool_strong_boundary",
            "family": "real",
            "mode": "component_pool",
            "score_scale": 12.0,
            "alpha": 0.70,
            "propagation_iters": 20,
            "edge_threshold": 0.72,
            "component_score_threshold": 0.62,
            "d4rt_weight": 0.08,
            "negative_weight": 0.20,
            "boundary_weight": 0.26,
            "max_region_fraction": 0.70,
            "description": "Strong boundary split to test AP50 sensitivity.",
        },
        {
            "variant_id": "OD5_component_pool_large_merge",
            "family": "real",
            "mode": "component_pool",
            "score_scale": 10.0,
            "alpha": 0.76,
            "propagation_iters": 28,
            "edge_threshold": 0.36,
            "component_score_threshold": 0.36,
            "d4rt_weight": 0.10,
            "negative_weight": 0.12,
            "boundary_weight": 0.00,
            "max_region_fraction": 0.98,
            "description": "Large merged component pooling to test region fragmentation as the blocker.",
        },
        {
            "variant_id": "OD6_component_pool_negative_veto",
            "family": "real",
            "mode": "component_pool",
            "score_scale": 11.0,
            "alpha": 0.72,
            "propagation_iters": 24,
            "edge_threshold": 0.56,
            "component_score_threshold": 0.54,
            "d4rt_weight": 0.16,
            "negative_weight": 0.28,
            "boundary_weight": 0.04,
            "max_region_fraction": 0.78,
            "description": "Component pooling with D4RT-positive and hard-negative veto.",
        },
        {
            "variant_id": "OD_CTRL_shuffled_component_pool",
            "family": "control",
            "mode": "shuffled_component_pool",
            "score_scale": 10.0,
            "alpha": 0.76,
            "propagation_iters": 24,
            "edge_threshold": 0.54,
            "component_score_threshold": 0.50,
            "d4rt_weight": 0.10,
            "negative_weight": 0.14,
            "boundary_weight": 0.02,
            "max_region_fraction": 0.88,
            "description": "Deterministically shuffle object rows before component pooling; must not be counted as real.",
        },
    ]


def _source_key_parts(raw: str) -> tuple[str, str, int, int]:
    return phase3a_obj._source_key_parts(raw)


def _load_region_edges(path: Path, selected_keys: set[tuple[str, str, int, int]]) -> tuple[dict[tuple[str, str, int, int], list[tuple[str, str, float, float]]], dict[str, Any]]:
    edges: dict[tuple[str, str, int, int], list[tuple[str, str, float, float]]] = defaultdict(list)
    scanned = 0
    matched = 0
    if not path.exists():
        return edges, {"edge_rows_scanned": 0, "edge_rows_matched": 0, "edge_rows_missing": True}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scanned += 1
            key = (
                str(row.get("scene_id", "")),
                str(row.get("window_id", "")),
                _int(row.get("frame_id"), -1),
                _int(row.get("source_mask_id"), -1),
            )
            if key not in selected_keys:
                continue
            a = str(row.get("region_id_a") or row.get("region_u") or "")
            b = str(row.get("region_id_b") or row.get("region_v") or "")
            if not a or not b:
                continue
            edge_weight = _num(row.get("edge_weight"), 1.0)
            barrier = max(
                _num(row.get("mask_edge_barrier")),
                _num(row.get("nested_edge_barrier")),
                _num(row.get("competing_edge_barrier")),
                _num(row.get("semantic_gradient_barrier")),
                _num(row.get("d4rt_conflict_barrier")),
            )
            edges[key].append((a, b, float(max(0.0, min(1.0, edge_weight))), float(max(0.0, barrier))))
            matched += 1
    return edges, {"edge_rows_scanned": scanned, "edge_rows_matched": matched, "edge_rows_missing": False}


def _fallback_grid_edges(feature_yx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if feature_yx.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32), 0
    coords = np.asarray(feature_yx, dtype=np.int32)
    buckets: dict[tuple[int, int], int] = {}
    for idx, (fy, fx) in enumerate(coords.tolist()):
        buckets[(int(fy), int(fx))] = int(idx)
    u: list[int] = []
    v: list[int] = []
    for idx, (fy, fx) in enumerate(coords.tolist()):
        for dy, dx in ((1, 0), (0, 1)):
            other = buckets.get((int(fy) + dy, int(fx) + dx))
            if other is not None:
                u.append(int(idx))
                v.append(int(other))
    return np.asarray(u, dtype=np.int64), np.asarray(v, dtype=np.int64), np.ones(len(u), dtype=np.float32), len(u)


def _edge_arrays_for_source(
    edge_rows: list[tuple[str, str, float, float]],
    region_ids: list[str],
    feature_yx: np.ndarray,
    barrier_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, int]:
    id_to_idx = {rid: idx for idx, rid in enumerate(region_ids)}
    u: list[int] = []
    v: list[int] = []
    w: list[float] = []
    for a, b, edge_weight, barrier in edge_rows:
        ia = id_to_idx.get(a)
        ib = id_to_idx.get(b)
        if ia is None or ib is None or ia == ib:
            continue
        keep = float(edge_weight) * max(0.0, 1.0 - barrier_scale * float(barrier))
        if keep <= 0.0:
            continue
        u.append(int(ia))
        v.append(int(ib))
        w.append(float(min(1.0, keep)))
    if u:
        return np.asarray(u, dtype=np.int64), np.asarray(v, dtype=np.int64), np.asarray(w, dtype=np.float32), "region_edge_rows", len(u)
    fu, fv, fw, count = _fallback_grid_edges(feature_yx)
    return fu, fv, fw, "feature_grid_fallback", count


def _propagate_probabilities(
    scores: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_w: np.ndarray,
    spec: dict[str, Any],
    device: torch.device,
    seed_text: str,
) -> tuple[np.ndarray, dict[str, float]]:
    started = time.perf_counter()
    score_arr = np.asarray(scores, dtype=np.float32)
    if str(spec.get("mode")).startswith("shuffled") and score_arr.shape[0] > 1:
        rng = np.random.default_rng(phase3a_obj._stable_seed(seed_text))
        score_arr = score_arr[rng.permutation(score_arr.shape[0]), :]
    score_t = torch.as_tensor(score_arr, dtype=torch.float32, device=device)
    logits = score_t * float(spec.get("score_scale", 10.0))
    prob0 = torch.softmax(logits, dim=0)
    prob = prob0.clone()
    residual = torch.tensor(0.0, dtype=torch.float32, device=device)
    if edge_u.size:
        eu = torch.as_tensor(edge_u, dtype=torch.long, device=device)
        ev = torch.as_tensor(edge_v, dtype=torch.long, device=device)
        ew = torch.as_tensor(edge_w, dtype=torch.float32, device=device).clamp(0.0, 1.0)
        alpha = float(spec.get("alpha", 0.72))
        for _ in range(max(1, int(spec.get("propagation_iters", spec.get("iters", 24))))):
            accum = torch.zeros_like(prob)
            denom = torch.zeros(prob.shape[1], dtype=torch.float32, device=device)
            accum.index_add_(1, eu, ew.unsqueeze(0) * prob[:, ev])
            accum.index_add_(1, ev, ew.unsqueeze(0) * prob[:, eu])
            denom.index_add_(0, eu, ew)
            denom.index_add_(0, ev, ew)
            neighbor = torch.where(denom.unsqueeze(0) > 1.0e-6, accum / denom.clamp_min(1.0e-6).unsqueeze(0), prob)
            new_prob = (1.0 - alpha) * prob0 + alpha * neighbor
            new_prob = new_prob / new_prob.sum(dim=0, keepdim=True).clamp_min(1.0e-6)
            residual = torch.mean(torch.abs(new_prob - prob))
            prob = new_prob
            if float(residual.detach().cpu()) < 1.0e-5:
                break
    prob_np = prob.detach().cpu().numpy().astype(np.float32)
    diagnostics = {
        "gpu_propagation_runtime_ms": 1000.0 * (time.perf_counter() - started),
        "gpu_solver_residual": float(residual.detach().cpu()),
        "edge_count_used": float(edge_u.size),
        "propagated_prob_mean": float(np.mean(prob_np)) if prob_np.size else 0.0,
        "propagated_prob_max": float(np.max(prob_np)) if prob_np.size else 0.0,
    }
    return prob_np, diagnostics


def _node_feature_arrays(nodes: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    d4rt_raw = np.asarray([_num(row.get("d4rt_witness_mass")) for row in nodes], dtype=np.float32)
    negative_raw = np.asarray([_num(row.get("hard_negative_witness_mass")) for row in nodes], dtype=np.float32)
    d4rt = np.log1p(np.maximum(d4rt_raw, 0.0)).astype(np.float32)
    negative = np.log1p(np.maximum(negative_raw, 0.0)).astype(np.float32)
    if d4rt.size and float(np.max(d4rt)) > 0.0:
        d4rt = d4rt / float(np.max(d4rt))
    if negative.size and float(np.max(negative)) > 0.0:
        negative = negative / float(np.max(negative))
    source_cos = np.asarray([_num(row.get("source_mean_cosine"), 0.0) for row in nodes], dtype=np.float32)
    background = np.asarray([_num(row.get("background_risk"), 0.0) for row in nodes], dtype=np.float32)
    broad = np.asarray([1.0 if str(row.get("broad_risk", "")).lower() == "true" else 0.0 for row in nodes], dtype=np.float32)
    boundary_token = np.asarray([1.0 if str(row.get("boundary_token", "")).lower() == "true" else 0.0 for row in nodes], dtype=np.float32)
    edge_distance = np.asarray([_num(row.get("source_edge_distance"), 0.0) for row in nodes], dtype=np.float32)
    edge_distance = np.nan_to_num(edge_distance, nan=0.0, posinf=0.0, neginf=0.0)
    near_edge = 1.0 / (1.0 + np.maximum(edge_distance, 0.0))
    boundary = np.maximum(0.35 * boundary_token, near_edge).astype(np.float32)
    area = np.asarray([max(1.0, _num(row.get("pixel_count"), _num(row.get("area_px"), 1.0))) for row in nodes], dtype=np.float32)
    return {
        "d4rt": d4rt,
        "negative": negative,
        "source_cos": np.clip(source_cos, 0.0, 1.0),
        "background": np.clip(background, 0.0, 1.0),
        "broad": broad,
        "boundary": np.clip(boundary, 0.0, 1.0),
        "area": area,
    }


def _weighted_neighbor_average(values: torch.Tensor, edge_u: torch.Tensor, edge_v: torch.Tensor, edge_w: torch.Tensor) -> torch.Tensor:
    if edge_u.numel() == 0:
        return values
    accum = torch.zeros_like(values)
    denom = torch.zeros_like(values)
    accum.index_add_(0, edge_u, edge_w * values[edge_v])
    accum.index_add_(0, edge_v, edge_w * values[edge_u])
    denom.index_add_(0, edge_u, edge_w)
    denom.index_add_(0, edge_v, edge_w)
    return torch.where(denom > 1.0e-6, accum / denom.clamp_min(1.0e-6), values)


def _cut_energy(margin: torch.Tensor, selected: torch.Tensor, edge_u: torch.Tensor, edge_v: torch.Tensor, edge_w: torch.Tensor, pairwise_weight: float) -> torch.Tensor:
    data = torch.sum(torch.where(selected, torch.clamp(0.5 - margin, min=0.0), torch.clamp(0.5 + margin, min=0.0)))
    if edge_u.numel() == 0:
        return data
    disagree = selected[edge_u] != selected[edge_v]
    return data + float(pairwise_weight) * torch.sum(edge_w[disagree])


def _component_count(selected: set[int], edge_u: np.ndarray, edge_v: np.ndarray) -> int:
    if not selected:
        return 0
    parent = {idx: idx for idx in selected}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for u_raw, v_raw in zip(edge_u.tolist(), edge_v.tolist(), strict=False):
        u = int(u_raw)
        v = int(v_raw)
        if u in parent and v in parent:
            union(u, v)
    return len({find(idx) for idx in selected})


def _component_labels(n: int, edge_u: np.ndarray, edge_v: np.ndarray, edge_w: np.ndarray, threshold: float) -> tuple[np.ndarray, int]:
    if n <= 0:
        return np.zeros(0, dtype=np.int32), 0
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for u_raw, v_raw, w_raw in zip(edge_u.tolist(), edge_v.tolist(), edge_w.tolist(), strict=False):
        if float(w_raw) < threshold:
            continue
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


def _select_component_regions_for_variant(
    spec: dict[str, Any],
    object_pairs: list[tuple[int, str]],
    scores: np.ndarray,
    prob: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_w: np.ndarray,
    nodes: list[dict[str, Any]],
    device: torch.device,
) -> dict[int, tuple[set[int], np.ndarray, dict[str, float]]]:
    r_regions = int(prob.shape[1]) if prob.ndim == 2 else 0
    if not object_pairs or r_regions <= 0:
        return {}
    if str(spec.get("mode")) == "whole_source":
        conflict_count = float(max(0, len(object_pairs) - 1) * r_regions)
        return {
            int(local_idx): (
                set(range(r_regions)),
                np.ones(r_regions, dtype=np.float32),
                {
                    "component_pool_count": 1.0,
                    "component_selected_count": 1.0,
                    "component_score_mean": 1.0,
                    "component_score_max": 1.0,
                    "region_ownership_conflict_count": conflict_count,
                    "pre_wta_region_ownership_conflict_count": conflict_count,
                    "cannot_link_violation_count": 0.0,
                    "component_count_proxy": 1.0,
                },
            )
            for local_idx, _object_key in object_pairs
        }

    feats = _node_feature_arrays(nodes)
    labels_np, comp_count = _component_labels(r_regions, edge_u, edge_v, edge_w, float(spec.get("edge_threshold", 0.5)))
    if comp_count <= 0:
        return {}
    labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)
    area = torch.as_tensor(feats["area"], dtype=torch.float32, device=device).clamp_min(1.0)
    comp_area = torch.zeros(comp_count, dtype=torch.float32, device=device)
    comp_area.index_add_(0, labels, area)
    d4rt = torch.as_tensor(feats["d4rt"], dtype=torch.float32, device=device)
    negative = torch.as_tensor(feats["negative"], dtype=torch.float32, device=device)
    source_cos = torch.as_tensor(feats["source_cos"], dtype=torch.float32, device=device)
    background = torch.as_tensor(feats["background"], dtype=torch.float32, device=device)
    broad = torch.as_tensor(feats["broad"], dtype=torch.float32, device=device)
    boundary = torch.as_tensor(feats["boundary"], dtype=torch.float32, device=device)
    prob_t = torch.as_tensor(prob, dtype=torch.float32, device=device)
    score_t = torch.as_tensor(np.nan_to_num(scores, nan=-1.0e9, neginf=-1.0e9, posinf=1.0), dtype=torch.float32, device=device)

    local_ids = [int(local_idx) for local_idx, _object_key in object_pairs]
    node_score_rows: list[torch.Tensor] = []
    comp_score_rows: list[torch.Tensor] = []
    selected_comp_rows: list[torch.Tensor] = []
    diagnostics_by_local: dict[int, dict[str, float]] = {}
    for local_idx in local_ids:
        obj_prob = prob_t[int(local_idx)]
        other_prob = torch.amax(torch.cat([prob_t[: int(local_idx)], prob_t[int(local_idx) + 1 :]], dim=0), dim=0) if prob_t.shape[0] > 1 else torch.zeros_like(obj_prob)
        margin_prob = obj_prob - other_prob
        obj_unary = torch.sigmoid(6.0 * (score_t[int(local_idx)] - 0.50))
        node_score = (
            0.50 * obj_prob
            + 0.25 * obj_unary
            + 0.16 * margin_prob
            + float(spec.get("d4rt_weight", 0.0)) * d4rt
            + 0.05 * source_cos
            - float(spec.get("negative_weight", 0.0)) * negative
            - float(spec.get("boundary_weight", 0.0)) * boundary
            - 0.08 * background
            - 0.06 * broad
        )
        node_score = torch.nan_to_num(node_score, nan=-1.0, posinf=1.0, neginf=-1.0)
        comp_sum = torch.zeros(comp_count, dtype=torch.float32, device=device)
        comp_sum.index_add_(0, labels, area * node_score)
        comp_score = comp_sum / comp_area.clamp_min(1.0)
        selected_comp = comp_score >= float(spec.get("component_score_threshold", 0.5))
        node_score_rows.append(node_score)
        comp_score_rows.append(comp_score)
        selected_comp_rows.append(selected_comp)
        diagnostics_by_local[int(local_idx)] = {
            "component_pool_count": float(comp_count),
            "component_selected_count": float(torch.count_nonzero(selected_comp).detach().cpu()),
            "component_score_mean": float(torch.mean(comp_score).detach().cpu()),
            "component_score_max": float(torch.max(comp_score).detach().cpu()) if comp_score.numel() else 0.0,
        }

    comp_scores_t = torch.stack(comp_score_rows, dim=0)
    selected_comp_t = torch.stack(selected_comp_rows, dim=0)
    comp_candidate_counts_t = torch.count_nonzero(selected_comp_t, dim=0)
    pre_wta_component_conflict = float(torch.sum(torch.clamp(comp_candidate_counts_t - 1, min=0)).detach().cpu())
    selected_score_t = torch.where(selected_comp_t, comp_scores_t, torch.full_like(comp_scores_t, -1.0e9))
    winner_score_t, winner_row_t = torch.max(selected_score_t, dim=0)
    comp_owner_t = torch.where(winner_score_t > -1.0e8, winner_row_t, torch.full_like(winner_row_t, -1))

    max_fraction = float(spec.get("max_region_fraction", 1.0))
    max_area = max_fraction * float(torch.sum(area).detach().cpu())
    for row_idx in range(len(local_ids)):
        owned_comp = torch.nonzero(comp_owner_t == row_idx, as_tuple=False).flatten()
        if owned_comp.numel() == 0:
            continue
        owned_area = comp_area[owned_comp]
        if float(torch.sum(owned_area).detach().cpu()) <= max_area:
            continue
        owned_scores = comp_scores_t[row_idx, owned_comp]
        order = torch.argsort(owned_scores, descending=True)
        keep: list[int] = []
        running = 0.0
        for idx in order.detach().cpu().tolist():
            comp_id = int(owned_comp[int(idx)].detach().cpu())
            area_val = float(comp_area[comp_id].detach().cpu())
            if keep and running + area_val > max_area:
                continue
            keep.append(comp_id)
            running += area_val
        drop = comp_owner_t == row_idx
        comp_owner_t[drop] = -1
        if keep:
            comp_owner_t[torch.as_tensor(keep, dtype=torch.long, device=device)] = row_idx

    comp_owner_np = comp_owner_t.detach().cpu().numpy().astype(np.int32)
    region_owner_np = comp_owner_np[labels_np]
    node_score_np = torch.stack(node_score_rows, dim=0).detach().cpu().numpy().astype(np.float32)
    out: dict[int, tuple[set[int], np.ndarray, dict[str, float]]] = {}
    for row_idx, local_idx in enumerate(local_ids):
        selected_set = {int(i) for i in np.nonzero(region_owner_np == row_idx)[0].tolist()}
        diag = dict(diagnostics_by_local[int(local_idx)])
        selected_comp_count = len({int(labels_np[idx]) for idx in selected_set})
        diag.update(
            {
                "component_selected_count_after_wta": float(selected_comp_count),
                "region_ownership_conflict_count": 0.0,
                "pre_wta_region_ownership_conflict_count": float(pre_wta_component_conflict),
                "cannot_link_violation_count": 0.0,
                "component_count_proxy": float(selected_comp_count),
            }
        )
        out[int(local_idx)] = (selected_set, node_score_np[row_idx], diag)
    return out


def _select_cut_regions_for_variant(
    spec: dict[str, Any],
    object_pairs: list[tuple[int, str]],
    scores: np.ndarray,
    prob: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_w: np.ndarray,
    nodes: list[dict[str, Any]],
    device: torch.device,
) -> dict[int, tuple[set[int], np.ndarray, dict[str, float]]]:
    r_regions = int(prob.shape[1]) if prob.ndim == 2 else 0
    if not object_pairs or r_regions <= 0:
        return {}
    if str(spec.get("mode")) == "whole_source":
        out: dict[int, tuple[set[int], np.ndarray, dict[str, float]]] = {}
        conflict_count = float(max(0, len(object_pairs) - 1) * r_regions)
        for local_idx, _object_key in object_pairs:
            score = np.ones(r_regions, dtype=np.float32)
            out[int(local_idx)] = (
                set(range(r_regions)),
                score,
                {
                    "cut_iteration_count": 0.0,
                    "energy_before": 0.0,
                    "energy_after": 0.0,
                    "energy_delta": 0.0,
                    "label_changed_region_count": 0.0,
                    "object_area_change_ratio": 1.0,
                    "region_ownership_conflict_count": conflict_count,
                    "pre_wta_region_ownership_conflict_count": conflict_count,
                    "cannot_link_violation_count": 0.0,
                    "component_count_proxy": 1.0,
                    "positive_seed_fraction": 1.0,
                },
            )
        return out

    feats = _node_feature_arrays(nodes)
    d4rt = torch.as_tensor(feats["d4rt"], dtype=torch.float32, device=device)
    negative = torch.as_tensor(feats["negative"], dtype=torch.float32, device=device)
    source_cos = torch.as_tensor(feats["source_cos"], dtype=torch.float32, device=device)
    background = torch.as_tensor(feats["background"], dtype=torch.float32, device=device)
    broad = torch.as_tensor(feats["broad"], dtype=torch.float32, device=device)
    boundary = torch.as_tensor(feats["boundary"], dtype=torch.float32, device=device)
    prob_t = torch.as_tensor(prob, dtype=torch.float32, device=device)
    score_t = torch.as_tensor(np.nan_to_num(scores, nan=-1.0e9, neginf=-1.0e9, posinf=1.0), dtype=torch.float32, device=device)
    eu = torch.as_tensor(edge_u, dtype=torch.long, device=device)
    ev = torch.as_tensor(edge_v, dtype=torch.long, device=device)
    ew = torch.as_tensor(edge_w, dtype=torch.float32, device=device).clamp(0.0, 1.0)

    score_rows: list[torch.Tensor] = []
    selected_rows: list[torch.Tensor] = []
    diagnostics_by_local: dict[int, dict[str, float]] = {}
    local_ids = [int(local_idx) for local_idx, _object_key in object_pairs]
    for local_idx in local_ids:
        obj_prob = prob_t[int(local_idx)]
        other_prob = torch.amax(torch.cat([prob_t[: int(local_idx)], prob_t[int(local_idx) + 1 :]], dim=0), dim=0) if prob_t.shape[0] > 1 else torch.zeros_like(obj_prob)
        margin_prob = obj_prob - other_prob
        obj_unary = torch.sigmoid(6.0 * (score_t[int(local_idx)] - 0.50))
        margin = (
            0.54 * obj_prob
            + 0.24 * obj_unary
            + 0.18 * margin_prob
            + float(spec.get("d4rt_weight", 0.0)) * d4rt
            + 0.05 * source_cos
            - float(spec.get("negative_weight", 0.0)) * negative
            - float(spec.get("boundary_weight", 0.0)) * boundary
            - 0.08 * background
            - 0.06 * broad
        )
        threshold = float(spec.get("threshold", 0.0))
        seed_score = obj_prob + 0.35 * obj_unary + float(spec.get("d4rt_weight", 0.0)) * d4rt - 0.25 * negative
        seed_threshold = torch.quantile(seed_score, 0.88) if seed_score.numel() else torch.tensor(1.0, device=device)
        pos_seed = seed_score >= seed_threshold
        selected = (margin >= threshold) | pos_seed
        energy_before = _cut_energy(margin, selected, eu, ev, ew, float(spec.get("pairwise_weight", 0.0)))
        changed_total = torch.tensor(0, dtype=torch.long, device=device)
        for _ in range(max(1, int(spec.get("cut_iters", 4)))):
            neighbor_keep = _weighted_neighbor_average(selected.float(), eu, ev, ew)
            proposal = margin + float(spec.get("pairwise_weight", 0.0)) * (neighbor_keep - 0.5)
            new_selected = (proposal >= threshold) | pos_seed
            changed = torch.count_nonzero(new_selected != selected)
            changed_total = changed_total + changed
            selected = new_selected
            if int(changed.detach().cpu()) == 0:
                break
        neighbor_keep = _weighted_neighbor_average(selected.float(), eu, ev, ew)
        cut_score = torch.sigmoid(4.0 * (margin + float(spec.get("pairwise_weight", 0.0)) * (neighbor_keep - 0.5)))
        energy_after = _cut_energy(margin, selected, eu, ev, ew, float(spec.get("pairwise_weight", 0.0)))
        score_rows.append(cut_score)
        selected_rows.append(selected)
        diagnostics_by_local[int(local_idx)] = {
            "cut_iteration_count": float(spec.get("cut_iters", 0)),
            "energy_before": float(energy_before.detach().cpu()),
            "energy_after": float(energy_after.detach().cpu()),
            "energy_delta": float((energy_after - energy_before).detach().cpu()),
            "label_changed_region_count": float(changed_total.detach().cpu()),
            "positive_seed_fraction": float(torch.count_nonzero(pos_seed).detach().cpu()) / max(1, r_regions),
        }

    cut_scores_t = torch.stack(score_rows, dim=0)
    selected_t = torch.stack(selected_rows, dim=0)
    selected_score_t = torch.where(selected_t, cut_scores_t, torch.full_like(cut_scores_t, -1.0e9))
    candidate_counts_t = torch.count_nonzero(selected_t, dim=0)
    initial_selected_counts = torch.count_nonzero(selected_t, dim=1).detach().cpu().numpy().astype(np.float64)
    conflict_count = float(torch.sum(torch.clamp(candidate_counts_t - 1, min=0)).detach().cpu())
    winner_score_t, winner_row_t = torch.max(selected_score_t, dim=0)
    owned_t = winner_score_t > -1.0e8
    ownership = torch.where(owned_t, winner_row_t, torch.full_like(winner_row_t, -1))

    min_count = max(1, int(math.ceil(float(spec.get("min_region_fraction", 0.0)) * r_regions)))
    max_count = max(1, int(math.floor(float(spec.get("max_region_fraction", 1.0)) * r_regions)))
    max_count = max(min_count, min(max_count, r_regions))
    for row_idx in range(len(local_ids)):
        row_score = cut_scores_t[row_idx]
        owned_count = int(torch.count_nonzero(ownership == row_idx).detach().cpu())
        if owned_count > max_count:
            owned_indices = torch.nonzero(ownership == row_idx, as_tuple=False).flatten()
            keep_local = torch.topk(row_score[owned_indices], k=max_count).indices
            keep_indices = owned_indices[keep_local]
            drop_mask = ownership == row_idx
            ownership[drop_mask] = -1
            ownership[keep_indices] = row_idx
        owned_count = int(torch.count_nonzero(ownership == row_idx).detach().cpu())
        if owned_count < min_count:
            need = min_count - owned_count
            available = ownership < 0
            if torch.any(available):
                available_indices = torch.nonzero(available, as_tuple=False).flatten()
                take = min(int(need), int(available_indices.numel()))
                if take > 0:
                    take_local = torch.topk(row_score[available_indices], k=take).indices
                    ownership[available_indices[take_local]] = row_idx

    ownership_np = ownership.detach().cpu().numpy().astype(np.int32)
    score_np = cut_scores_t.detach().cpu().numpy().astype(np.float32)
    out: dict[int, tuple[set[int], np.ndarray, dict[str, float]]] = {}
    for row_idx, local_idx in enumerate(local_ids):
        selected_set = {int(i) for i in np.nonzero(ownership_np == row_idx)[0].tolist()}
        diag = dict(diagnostics_by_local[int(local_idx)])
        diag.update(
            {
                "object_area_change_ratio": float(len(selected_set) / max(1.0, initial_selected_counts[row_idx])),
                "region_ownership_conflict_count": 0.0,
                "pre_wta_region_ownership_conflict_count": conflict_count,
                "cannot_link_violation_count": 0.0,
                "component_count_proxy": float(_component_count(selected_set, edge_u, edge_v)),
            }
        )
        out[int(local_idx)] = (selected_set, score_np[row_idx], diag)
    return out


def _materialize_variant_rows(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = _resolve(args.output_root)
    if out.exists() and args.clean:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    field_root = _resolve(args.field_root)
    shard_paths = sorted((field_root / "field_shards").glob("object_axis_unary_shard_*.npz"))
    if int(args.max_shards) > 0:
        shard_paths = shard_paths[: int(args.max_shards)]
    if not shard_paths:
        raise RuntimeError(f"No object-axis unary shards found under {field_root / 'field_shards'}")

    specs = _variant_specs()
    variant_ids = [spec["variant_id"] for spec in specs]
    created_at = _created_at()
    source_key_texts: list[str] = []
    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as data:
            source_key_texts.extend(str(value) for value in data["source_keys"].tolist())
    if int(args.max_sources) > 0:
        source_key_texts = source_key_texts[: int(args.max_sources)]
    selected_source_keys = {_source_key_parts(raw) for raw in source_key_texts}
    physical_source_keys = {(scene, frame_id, mask_id) for scene, _window, frame_id, mask_id in selected_source_keys}
    source_meta = v92field._load_source_meta(_resolve(args.source_container_rows))
    node_maps = phase3a_obj._load_region_nodes(_resolve(args.region_node_rows), physical_source_keys)
    edge_maps, edge_stats = _load_region_edges(_resolve(args.region_edge_rows), selected_source_keys)
    device = torch.device(str(args.device) if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")

    writer = phase5.ScoreWTAFrameWriter(out, variant_ids)
    label_cache: dict[tuple[str, int], np.ndarray] = {}
    generated_rows: list[dict[str, Any]] = []
    mv_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    source_summary_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    config_rows = [
        {
            "schema_version": "stream4d_v94_phase3D_object_axis_config_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "variant_id": spec["variant_id"],
            "family": spec["family"],
            "mode": spec["mode"],
            "description": spec["description"],
            "created_at": created_at,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for spec in specs
    ]

    source_limit_set = set(source_key_texts)
    processed_source_count = 0
    materialized_object_count = 0
    score_protocol_counts: Counter[str] = Counter()
    edge_source_counts: Counter[str] = Counter()
    gpu_runtime_ms: list[float] = []
    solver_residuals: list[float] = []
    source_object_counts: list[float] = []
    source_region_counts: list[float] = []

    for shard_i, shard_path in enumerate(shard_paths):
        with np.load(shard_path, allow_pickle=False) as data:
            source_keys = [str(value) for value in data["source_keys"].tolist()]
            object_keys = [str(value) for value in data["object_keys"].tolist()]
            object_source_index = data["object_source_index"].astype(np.int32)
            object_local_index = data["object_local_index"].astype(np.int32)
            region_source_index = data["region_source_index"].astype(np.int32)
            region_ids_all = [str(value) for value in data["region_ids"].tolist()]
            region_indices = data["region_indices"].astype(np.int32)
            region_feature_yx = data["region_feature_yx"].astype(np.int32)
            unary_source_index = data["unary_source_index"].astype(np.int32)
            unary_object_local_index = data["unary_object_local_index"].astype(np.int32)
            unary_region_local_index = data["unary_region_local_index"].astype(np.int32)
            unary_cosine = data["unary_cosine"].astype(np.float32)

            for source_idx, raw_source_key in enumerate(source_keys):
                if source_limit_set and raw_source_key not in source_limit_set:
                    continue
                scene, window_id, frame_id, mask_id = _source_key_parts(raw_source_key)
                key4 = (scene, window_id, frame_id, mask_id)
                physical_key = (scene, frame_id, mask_id)
                meta = source_meta.get(physical_key)
                node_map = node_maps.get(physical_key, {})
                if not meta or not node_map:
                    failure_rows.append(
                        {
                            "schema_version": "stream4d_v94_phase3D_object_axis_failure_v1",
                            "phase_id": PHASE_ID,
                            "run_id": RUN_ID,
                            "source_key": raw_source_key,
                            "failure_type": "missing_source_meta_or_region_nodes",
                            "uses_gt_for_prediction": False,
                            "uses_future": False,
                        }
                    )
                    continue
                region_mask = region_source_index == source_idx
                source_region_indices = region_indices[region_mask]
                source_region_ids = [region_ids_all[pos] for pos in np.nonzero(region_mask)[0].tolist()]
                source_feature_yx = region_feature_yx[region_mask]
                nodes = [node_map.get(int(region_index)) for region_index in source_region_indices.tolist()]
                if any(node is None for node in nodes):
                    failure_rows.append(
                        {
                            "schema_version": "stream4d_v94_phase3D_object_axis_failure_v1",
                            "phase_id": PHASE_ID,
                            "run_id": RUN_ID,
                            "source_key": raw_source_key,
                            "failure_type": "missing_region_node_for_shard_index",
                            "uses_gt_for_prediction": False,
                            "uses_future": False,
                        }
                    )
                    continue
                nodes_typed: list[dict[str, Any]] = [node for node in nodes if node is not None]
                object_mask = object_source_index == source_idx
                object_pairs = sorted(
                    [(int(local_idx), object_keys[obj_pos]) for obj_pos, local_idx in enumerate(object_local_index) if bool(object_mask[obj_pos])],
                    key=lambda item: item[0],
                )
                if not object_pairs or not nodes_typed:
                    continue
                object_by_local = {local_idx: object_key for local_idx, object_key in object_pairs}
                k_objects = max(object_by_local) + 1
                r_regions = len(nodes_typed)
                scores = np.full((k_objects, r_regions), -1.0e9, dtype=np.float32)
                source_unary_mask = unary_source_index == source_idx
                for obj_idx, region_idx, value in zip(
                    unary_object_local_index[source_unary_mask],
                    unary_region_local_index[source_unary_mask],
                    unary_cosine[source_unary_mask],
                    strict=False,
                ):
                    if 0 <= int(obj_idx) < k_objects and 0 <= int(region_idx) < r_regions:
                        scores[int(obj_idx), int(region_idx)] = float(value)
                if not np.any(np.isfinite(scores) & (scores > -1.0e8)):
                    continue
                edge_u, edge_v, edge_w, edge_source, edge_count = _edge_arrays_for_source(
                    edge_maps.get(key4, []),
                    source_region_ids,
                    source_feature_yx,
                    float(args.barrier_scale),
                )
                edge_source_counts[edge_source] += 1

                mask_path = _resolve(str(meta.get("mask_path", "")))
                frame_key = (scene, frame_id)
                if frame_key not in label_cache:
                    label_cache[frame_key] = v92field._read_label(mask_path)
                label = label_cache[frame_key]
                source_mask = label == int(mask_id)
                if not np.any(source_mask):
                    continue
                writer.ensure_frame(scene, frame_id, label.shape)
                source_area = int(np.count_nonzero(source_mask))
                processed_source_count += 1
                source_object_counts.append(float(len(object_pairs)))
                source_region_counts.append(float(r_regions))

                prob_by_variant: dict[str, tuple[np.ndarray, dict[str, float]]] = {}
                for spec in specs:
                    if spec["mode"] == "whole_source":
                        continue
                    prob, diag = _propagate_probabilities(scores, edge_u, edge_v, edge_w, spec, device, f"{raw_source_key}|{spec['variant_id']}")
                    prob_by_variant[str(spec["variant_id"])] = (prob, diag)
                    gpu_runtime_ms.append(float(diag.get("gpu_propagation_runtime_ms", 0.0)))
                    solver_residuals.append(float(diag.get("gpu_solver_residual", 0.0)))

                for spec in specs:
                    variant_id = str(spec["variant_id"])
                    if spec["mode"] == "whole_source":
                        prob = np.ones((k_objects, r_regions), dtype=np.float32) / max(1, k_objects)
                        diag = {"gpu_propagation_runtime_ms": 0.0, "gpu_solver_residual": 0.0}
                    else:
                        prob, diag = prob_by_variant[variant_id]
                    component_selections = _select_component_regions_for_variant(spec, object_pairs, scores, prob, edge_u, edge_v, edge_w, nodes_typed, device)
                    for object_local_idx, object_key in object_pairs:
                        selected, component_scores, component_diag = component_selections.get(int(object_local_idx), (set(), np.zeros(r_regions, dtype=np.float32), {}))
                        if not selected:
                            continue
                        mask = v92field._node_mask(nodes_typed, selected, source_mask)
                        if not np.any(mask):
                            continue
                        selected_prob = [float(prob[int(object_local_idx), idx]) for idx in selected]
                        if prob.shape[0] >= 2:
                            other_prob = np.max(np.delete(prob, int(object_local_idx), axis=0), axis=0)
                            prob_margin = prob[int(object_local_idx), :] - other_prob
                        else:
                            prob_margin = np.zeros(r_regions, dtype=np.float32)
                        selected_margin = [float(prob_margin[idx]) for idx in selected]
                        selected_component_score = [float(component_scores[idx]) for idx in selected]
                        selected_unary = [float(scores[int(object_local_idx), idx]) for idx in selected]
                        area = int(np.count_nonzero(mask))
                        area_ratio = float(area / max(1, source_area))
                        object_score = float(0.50 * _mean(selected_component_score) + 0.25 * _mean(selected_prob) + 0.15 * _mean(selected_unary) + 0.10 * min(1.0, area_ratio))
                        score_protocol = "object_axis_gpu_component_pool_score_prob_unary_area"
                        score_protocol_counts[score_protocol] += 1
                        generated_row = {
                            "schema_version": "stream4d_v94_phase3D_object_axis_generated_mask_v1",
                            "phase_id": PHASE_ID,
                            "run_id": RUN_ID,
                            "variant_id": variant_id,
                            "scene_id": scene,
                            "split": "dev",
                            "window_id": window_id,
                            "frame_id": frame_id,
                            "source_mask_id": mask_id,
                            "new_mask_id": "",
                            "object_hypothesis_id": object_key,
                            "generated_mask_path": _rel(out / "generated_masks" / variant_id / scene / "mask" / f"{int(frame_id)}.png"),
                            "source_mask_area": source_area,
                            "generated_mask_area_before_frame_wta": area,
                            "generated_mask_area": area,
                            "generated_mask_area_ratio": area_ratio,
                            "selected_region_count": len(selected),
                            "total_region_count": r_regions,
                            "mean_selected_unary_cosine": _mean(selected_unary),
                            "mean_selected_propagated_prob": _mean(selected_prob),
                            "mean_selected_margin": _mean(selected_margin),
                            "mean_selected_component_score": _mean(selected_component_score),
                            "component_pool_count": component_diag.get("component_pool_count", ""),
                            "component_selected_count": component_diag.get("component_selected_count", ""),
                            "component_selected_count_after_wta": component_diag.get("component_selected_count_after_wta", ""),
                            "component_score_mean": component_diag.get("component_score_mean", ""),
                            "component_score_max": component_diag.get("component_score_max", ""),
                            "region_ownership_conflict_count": component_diag.get("region_ownership_conflict_count", ""),
                            "pre_wta_region_ownership_conflict_count": component_diag.get("pre_wta_region_ownership_conflict_count", ""),
                            "cannot_link_violation_count": component_diag.get("cannot_link_violation_count", ""),
                            "component_count_proxy": component_diag.get("component_count_proxy", ""),
                            "edge_source": edge_source,
                            "edge_count_used": edge_count,
                            "score_protocol": score_protocol,
                            "solver_backend": "torch_cuda_edge_propagation_plus_component_pooling" if str(device).startswith("cuda") else "torch_cpu_edge_propagation_plus_component_pooling",
                            "gpu_device": str(device),
                            "uses_gt_for_prediction": False,
                            "uses_future": False,
                        }
                        mv_row = {
                            "split": "dev",
                            "scene_id": scene,
                            "source_variant": variant_id,
                            "variant": variant_id,
                            "mv_object_id": f"{variant_id}:{object_key}",
                            "frame_id": frame_id,
                            "mask_id": "",
                            "frame_mask_score": object_score,
                            "object_score": object_score,
                            "uses_gt_for_prediction": False,
                            "uses_future": False,
                            "uses_rgbd_pose_mesh": False,
                            "materializable": True,
                            "selection_reason": f"v94_phase3D_object_axis_{spec['mode']}_{score_protocol}",
                        }
                        writer.add(variant_id, mask, object_score, generated_row, mv_row)
                        materialized_object_count += 1
                        assignment_rows.append(
                            {
                                "schema_version": "stream4d_v94_phase3D_object_axis_assignment_summary_v1",
                                "phase_id": PHASE_ID,
                                "run_id": RUN_ID,
                                "variant_id": variant_id,
                                "scene_id": scene,
                                "window_id": window_id,
                                "frame_id": frame_id,
                                "source_mask_id": mask_id,
                                "canonical_object_key": object_key,
                                "object_local_index": object_local_idx,
                                "selected_region_count": len(selected),
                                "total_region_count": r_regions,
                                "selected_region_fraction": float(len(selected) / max(1, r_regions)),
                                "generated_mask_area_ratio": area_ratio,
                                "mean_selected_unary_cosine": _mean(selected_unary),
                                "mean_selected_propagated_prob": _mean(selected_prob),
                                "mean_selected_margin": _mean(selected_margin),
                                "mean_selected_component_score": _mean(selected_component_score),
                                "component_pool_count": component_diag.get("component_pool_count", ""),
                                "component_selected_count": component_diag.get("component_selected_count", ""),
                                "component_selected_count_after_wta": component_diag.get("component_selected_count_after_wta", ""),
                                "component_score_mean": component_diag.get("component_score_mean", ""),
                                "component_score_max": component_diag.get("component_score_max", ""),
                                "region_ownership_conflict_count": component_diag.get("region_ownership_conflict_count", ""),
                                "pre_wta_region_ownership_conflict_count": component_diag.get("pre_wta_region_ownership_conflict_count", ""),
                                "cannot_link_violation_count": component_diag.get("cannot_link_violation_count", ""),
                                "component_count_proxy": component_diag.get("component_count_proxy", ""),
                                "edge_source": edge_source,
                                "edge_count_used": edge_count,
                                "gpu_propagation_runtime_ms": diag.get("gpu_propagation_runtime_ms", ""),
                                "gpu_solver_residual": diag.get("gpu_solver_residual", ""),
                                "uses_gt_for_prediction": False,
                                "uses_future": False,
                            }
                        )
                source_summary_rows.append(
                    {
                        "schema_version": "stream4d_v94_phase3D_object_axis_source_summary_v1",
                        "phase_id": PHASE_ID,
                        "run_id": RUN_ID,
                        "source_key": raw_source_key,
                        "canonical_object_count": len(object_pairs),
                        "region_count": r_regions,
                        "edge_source": edge_source,
                        "edge_count_used": edge_count,
                        "top_unary_mean": float(np.max(scores, axis=0).mean()),
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )
                if int(args.progress_every_sources) > 0 and processed_source_count % int(args.progress_every_sources) == 0:
                    print(json.dumps({"phase": PHASE_ID, "processed_source_count": processed_source_count, "materialized_object_count_before_frame_wta": materialized_object_count, "edge_source_counts": dict(edge_source_counts), "elapsed_sec": time.time() - started}, sort_keys=True), flush=True)

    writer.flush()
    generated_rows = writer.generated_rows
    mv_rows = writer.mv_rows
    object_rows = phase5._object_rows_from_mv(mv_rows)

    radius_sweep.OUT = out
    metric_rows: list[dict[str, Any]] = []
    casebook_rows: list[dict[str, Any]] = []
    for variant_id in variant_ids:
        rows = [row for row in mv_rows if row.get("variant") == variant_id]
        metrics, cases = radius_sweep._evaluate_variant(variant_id, rows)
        metric_rows.extend(metrics)
        casebook_rows.extend({**row, "variant_id": variant_id} for row in cases)
    aggregate_rows = phase7d._aggregate(metric_rows)

    assignment_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in assignment_rows:
        assignment_by_variant[str(row.get("variant_id", ""))].append(row)
    variant_metric_rows: list[dict[str, Any]] = []
    for row in aggregate_rows:
        variant_id = str(row.get("variant_id", ""))
        group = assignment_by_variant.get(variant_id, [])
        variant_metric_rows.append(
            {
                "schema_version": "stream4d_v94_phase3D_object_axis_variant_metric_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": variant_id,
                "scene_id": "ALL_DEV",
                "split": "dev",
                "source_artifact": _rel(out / "mv_metric_aggregate_rows.csv"),
                "created_at": created_at,
                "scene_count": row.get("scene_count", ""),
                "mean_MV_AP_window": row.get("mean_MV_AP_window", ""),
                "mean_MV_AP50_window": row.get("mean_MV_AP50_window", ""),
                "mean_MV_AP25_window": row.get("mean_MV_AP25_window", ""),
                "mean_score_free_Match50_window": row.get("mean_score_free_Match50_window", ""),
                "mean_gt_object_count": row.get("mean_gt_object_count", ""),
                "mean_pred_object_count": row.get("mean_pred_object_count", ""),
                "same_frame_collision_count": row.get("same_frame_collision_count", ""),
                "missing_mask_raster_count": row.get("missing_mask_raster_count", ""),
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "mean_generated_area_ratio": _mean([_num(item.get("generated_mask_area_ratio")) for item in group]),
                "object_region_count_mean": _mean([_num(item.get("selected_region_count")) for item in group]),
                "selected_region_fraction_mean": _mean([_num(item.get("selected_region_fraction")) for item in group]),
                "mean_selected_unary_cosine": _mean([_num(item.get("mean_selected_unary_cosine")) for item in group]),
                "mean_selected_propagated_prob": _mean([_num(item.get("mean_selected_propagated_prob")) for item in group]),
                "mean_selected_margin": _mean([_num(item.get("mean_selected_margin")) for item in group]),
                "mean_selected_component_score": _mean([_num(item.get("mean_selected_component_score")) for item in group]),
                "component_pool_count_mean": _mean([_num(item.get("component_pool_count")) for item in group]),
                "component_selected_count_mean": _mean([_num(item.get("component_selected_count")) for item in group]),
                "component_selected_count_after_wta_mean": _mean([_num(item.get("component_selected_count_after_wta")) for item in group]),
                "component_score_mean": _mean([_num(item.get("component_score_mean")) for item in group]),
                "component_score_max_mean": _mean([_num(item.get("component_score_max")) for item in group]),
                "region_ownership_conflict_count_sum": sum(_num(item.get("region_ownership_conflict_count")) for item in group),
                "pre_wta_region_ownership_conflict_count_sum": sum(_num(item.get("pre_wta_region_ownership_conflict_count")) for item in group),
                "cannot_link_violation_count_sum": sum(_num(item.get("cannot_link_violation_count")) for item in group),
                "component_count_proxy_mean": _mean([_num(item.get("component_count_proxy")) for item in group]),
            }
        )

    phase0 = _read_json(V94_PHASE0 / "summary.json")
    phase3a_object_axis = _read_json(ROOT / "outputs/audit/v94_phase3A_object_axis_full_dev_combined/summary.json")
    phase3b_object_axis = _read_json(ROOT / "outputs/audit/v94_phase3B_object_axis_propagation/summary.json")
    phase3c_object_axis = _read_json(ROOT / "outputs/audit/v94_phase3C_object_axis_constrained_cut/summary.json")
    required_ap = _num(phase0.get("required_MV_AP_window"))
    required_ap50 = _num(phase0.get("required_MV_AP50_window"))
    previous_object_axis_ap = max(
        _num(phase3a_object_axis.get("best_real_MV_AP_window")),
        _num(phase3b_object_axis.get("best_real_MV_AP_window")),
        _num(phase3c_object_axis.get("best_real_MV_AP_window")),
    )
    previous_object_axis_ap50 = max(
        _num(phase3a_object_axis.get("best_real_MV_AP50_window")),
        _num(phase3b_object_axis.get("best_real_MV_AP50_window")),
        _num(phase3c_object_axis.get("best_real_MV_AP50_window")),
    )
    specs_by_id = {spec["variant_id"]: spec for spec in specs}
    gate_rows: list[dict[str, Any]] = []
    failure_rows_out: list[dict[str, Any]] = []
    for row in variant_metric_rows:
        variant_id = str(row.get("variant_id", ""))
        spec = specs_by_id.get(variant_id, {})
        mv_ap = _num(row.get("mean_MV_AP_window"))
        mv_ap50 = _num(row.get("mean_MV_AP50_window"))
        missing = int(_num(row.get("missing_mask_raster_count")))
        collision = int(_num(row.get("same_frame_collision_count")))
        provenance_gate = bool(missing == 0 and collision == 0)
        candidate_gate = bool(
            spec.get("family") == "real"
            and mv_ap >= previous_object_axis_ap + 0.003
            and mv_ap50 >= previous_object_axis_ap50 + 0.006
            and provenance_gate
        )
        gate_pass = bool(spec.get("family") == "real" and mv_ap >= required_ap and mv_ap50 >= required_ap50 and provenance_gate)
        gate_rows.append(
            {
                "schema_version": "stream4d_v94_phase3D_object_axis_gate_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": variant_id,
                "family": spec.get("family", ""),
                "phase3D_candidate_gate_pass": candidate_gate,
                "dev_progress_gate_pass": gate_pass,
                "MV_AP_window": mv_ap,
                "MV_AP50_window": mv_ap50,
                "previous_object_axis_best_MV_AP_window": previous_object_axis_ap,
                "previous_object_axis_best_MV_AP50_window": previous_object_axis_ap50,
                "required_MV_AP_window": required_ap,
                "required_MV_AP50_window": required_ap50,
                "same_frame_collision_count": collision,
                "missing_mask_raster_count": missing,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        if spec.get("family") == "real" and not gate_pass:
            failure_rows_out.append(
                {
                    "schema_version": "stream4d_v94_phase3D_object_axis_failure_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "variant_id": variant_id,
                    "failure_type": "OBJECT_AXIS_COMPONENT_POOLING_DEV_GATE_FAILED",
                    "MV_AP_window": mv_ap,
                    "MV_AP50_window": mv_ap50,
                    "phase3D_candidate_gate_pass": candidate_gate,
                    "repair_direction": "If component pooling over-prunes, lower component threshold; if it still stays below A/B/C or controls, mark object-axis field signal insufficient for local gate.",
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    best_real = max(
        [row for row in variant_metric_rows if specs_by_id.get(str(row.get("variant_id", "")), {}).get("family") == "real"],
        key=lambda row: (_num(row.get("mean_MV_AP_window"), -999.0), _num(row.get("mean_MV_AP50_window"), -999.0)),
        default={},
    )
    best_real_gate = bool(
        best_real
        and _num(best_real.get("mean_MV_AP_window")) >= required_ap
        and _num(best_real.get("mean_MV_AP50_window")) >= required_ap50
        and int(_num(best_real.get("missing_mask_raster_count"))) == 0
        and int(_num(best_real.get("same_frame_collision_count"))) == 0
    )
    best_real_candidate_gate = bool(
        best_real
        and _num(best_real.get("mean_MV_AP_window")) >= previous_object_axis_ap + 0.003
        and _num(best_real.get("mean_MV_AP50_window")) >= previous_object_axis_ap50 + 0.006
        and int(_num(best_real.get("missing_mask_raster_count"))) == 0
        and int(_num(best_real.get("same_frame_collision_count"))) == 0
    )

    _write_csv(out / "variant_config_rows.csv", config_rows)
    _write_csv(out / "generated_mask_rows.csv", generated_rows)
    _write_csv(out / "mv_object_rows.csv", object_rows)
    _write_csv(out / "mv_object_frame_mask_rows.csv", mv_rows)
    _write_csv(out / "assignment_summary_rows.csv", assignment_rows)
    _write_csv(out / "source_summary_rows.csv", source_summary_rows)
    _write_csv(out / "source_failure_rows.csv", failure_rows)
    _write_csv(out / "mv_metric_rows.csv", metric_rows)
    _write_csv(out / "mv_metric_aggregate_rows.csv", aggregate_rows)
    _write_csv(out / "variant_metric_rows.csv", variant_metric_rows)
    _write_csv(out / "variant_gate_rows.csv", gate_rows)
    _write_csv(out / "variant_failure_rows.csv", failure_rows_out)
    _write_csv(out / "casebook_rows.csv", casebook_rows)

    summary = {
        "schema": "stream4d_v94_phase3D_object_axis_component_pooling_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "decision": "PASS_V94_PHASE3D_OBJECT_AXIS_COMPONENT_POOLING_GATE" if best_real_gate else "NO_GO_V94_PHASE3D_OBJECT_AXIS_COMPONENT_POOLING_GATE",
        "created_at": created_at,
        "duration_sec": float(time.time() - started),
        "field_root": _rel(field_root),
        "field_shard_count": len(shard_paths),
        "processed_source_count": processed_source_count,
        "materialized_object_count_before_frame_wta": materialized_object_count,
        "generated_mask_rows_after_frame_wta": len(generated_rows),
        "mv_object_frame_mask_rows": len(mv_rows),
        "variant_count": len(specs),
        "best_real_variant_id": best_real.get("variant_id", ""),
        "best_real_MV_AP_window": best_real.get("mean_MV_AP_window", ""),
        "best_real_MV_AP50_window": best_real.get("mean_MV_AP50_window", ""),
        "phase3D_candidate_gate_pass": best_real_candidate_gate,
        "previous_object_axis_best_MV_AP_window": previous_object_axis_ap,
        "previous_object_axis_best_MV_AP50_window": previous_object_axis_ap50,
        "dev_progress_gate_pass": best_real_gate,
        "required_MV_AP_window": required_ap,
        "required_MV_AP50_window": required_ap50,
        "edge_source_counts": dict(edge_source_counts),
        "edge_rows_scanned": edge_stats.get("edge_rows_scanned", ""),
        "edge_rows_matched": edge_stats.get("edge_rows_matched", ""),
        "gpu_device": str(device),
        "gpu_propagation_runtime_ms_mean": _mean(gpu_runtime_ms),
        "gpu_solver_residual_mean": _mean(solver_residuals),
        "source_object_count_mean": _mean(source_object_counts),
        "source_region_count_mean": _mean(source_region_counts),
        "failure_count": len(failure_rows),
        "row_counts": {
            "variant_config_rows": len(config_rows),
            "generated_mask_rows": len(generated_rows),
            "mv_object_rows": len(object_rows),
            "mv_object_frame_mask_rows": len(mv_rows),
            "assignment_summary_rows": len(assignment_rows),
            "source_summary_rows": len(source_summary_rows),
            "source_failure_rows": len(failure_rows),
            "mv_metric_rows": len(metric_rows),
            "variant_metric_rows": len(variant_metric_rows),
            "variant_gate_rows": len(gate_rows),
            "variant_failure_rows": len(failure_rows_out),
            "casebook_rows": len(casebook_rows),
        },
        "score_protocol_counts": dict(score_protocol_counts),
        "input_artifacts": {
            _rel(path): _sha256(path)
            for path in [
                *shard_paths,
                _resolve(args.source_container_rows),
                _resolve(args.region_node_rows),
                _resolve(args.region_edge_rows),
            ]
            if path.exists()
        },
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_json(out / "summary.json", summary)
    outputs = [
        out / "summary.json",
        out / "variant_config_rows.csv",
        out / "generated_mask_rows.csv",
        out / "mv_object_rows.csv",
        out / "mv_object_frame_mask_rows.csv",
        out / "assignment_summary_rows.csv",
        out / "source_summary_rows.csv",
        out / "source_failure_rows.csv",
        out / "mv_metric_rows.csv",
        out / "mv_metric_aggregate_rows.csv",
        out / "variant_metric_rows.csv",
        out / "variant_gate_rows.csv",
        out / "variant_failure_rows.csv",
        out / "casebook_rows.csv",
    ]
    _write_json(out / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs if path.exists()})
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUT))
    parser.add_argument("--field-root", default=str(DEFAULT_FIELD_ROOT))
    parser.add_argument("--source-container-rows", default=str(ROOT / "outputs/audit/v94_phase1_canonical_graph/container_rows.csv"))
    parser.add_argument("--region-node-rows", default=str(ROOT / "outputs/audit/v94_phase1_canonical_graph/region_node_rows.csv"))
    parser.add_argument("--region-edge-rows", default=str(REGION_EDGE_ROWS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--barrier-scale", type=float, default=0.35)
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--max-sources", type=int, default=0)
    parser.add_argument("--progress-every-sources", type=int, default=256)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    summary = _materialize_variant_rows(args)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run(parse_args())
