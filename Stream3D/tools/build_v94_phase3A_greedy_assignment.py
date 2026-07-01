#!/usr/bin/env python3
"""Run v94 Phase3A greedy assignment variants from GPU field shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_v93_phase5b_unknown_background_field as base  # noqa: E402

PHASE_ID = "v94_phase3A_greedy_assignment"
DEFAULT_RUN_ID = "v94_phase3A_greedy_assignment_gpu"
RUN_ID = DEFAULT_RUN_ID
OUT = ROOT / "outputs/audit/v94_phase3A_greedy_assignment"
V94_PHASE0 = ROOT / "outputs/audit/v94_phase0_fact_lock"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fieldnames})


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rewrite_csv_schema(path: Path, schema_version: str) -> None:
    rows = _read_csv(path)
    if not rows:
        return
    for row in rows:
        if "schema_version" in row:
            row["schema_version"] = schema_version
    _write_csv(path, rows)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def _v94_variant_specs(variant_set: str = "main") -> list[dict[str, Any]]:
    if variant_set == "d4rt_control":
        return [
            {
                "variant_id": "A0_current_whole_source_replay",
                "family": "baseline",
                "mode": "whole",
                "base_variant": "F0_whole_source_baseline",
                "description": "Replay whole-source region assignment to verify materialization/evaluator wiring.",
            },
            {
                "variant_id": "CTRL8_shuffled_D4RT_witness",
                "family": "control",
                "mode": "v94_shuffled_d4rt_witness",
                "base_variant": "SHUFFLED_D4RT_NO_ORIGINAL_UNARY_SCORE",
                "threshold": 0.20,
                "w_prob": 0.0,
                "w_d4rt": 0.18,
                "w_sem": 0.18,
                "w_inside": 0.08,
                "w_source": 0.0,
                "w_nested": 0.0,
                "w_competing": 0.0,
                "w_negative": 0.08,
                "area_cap": 0.98,
                "min_area_fraction": 0.82,
                "description": "Same-support control that deterministically shuffles the explicit D4RT region-witness channel inside each source before region scoring; original F2 unary probability and object score are not used.",
            },
        ]
    if variant_set == "edge_repair":
        return [
            {
                "variant_id": "A0_current_whole_source_replay",
                "family": "baseline",
                "mode": "whole",
                "base_variant": "F0_whole_source_baseline",
                "description": "Replay whole-source region assignment to verify materialization/evaluator wiring.",
            },
            {
                "variant_id": "A1r_greedy_d4rt_radio_no_edge_replay",
                "family": "real",
                "mode": "v94_greedy",
                "base_variant": "F2_D4RT_RADIO_pairwise",
                "threshold": 0.34,
                "w_prob": 0.50,
                "w_d4rt": 0.22,
                "w_sem": 0.20,
                "w_inside": 0.08,
                "w_source": 0.0,
                "w_nested": 0.0,
                "w_competing": 0.0,
                "w_negative": 0.16,
                "area_cap": 0.92,
                "min_area_fraction": 0.55,
                "description": "A1 no-edge replay inside the repair run for direct same-run comparison.",
            },
            {
                "variant_id": "A8_edge_confidence_temperature",
                "family": "real",
                "mode": "v94_soft_edge_confidence",
                "base_variant": "F2_D4RT_RADIO_pairwise",
                "threshold": 0.28,
                "w_prob": 0.52,
                "w_d4rt": 0.20,
                "w_sem": 0.18,
                "w_inside": 0.08,
                "w_source": 0.01,
                "w_nested": 0.01,
                "w_competing": 0.02,
                "w_negative": 0.12,
                "graph_mix": 0.22,
                "w_soft_edge_cut": 0.05,
                "area_cap": 0.95,
                "min_area_fraction": 0.66,
                "description": "Use graph edge confidence as soft score propagation, with only a mild low-confidence cut penalty.",
            },
            {
                "variant_id": "A9_source_preserve_soft_edge",
                "family": "real",
                "mode": "v94_soft_edge_confidence",
                "base_variant": "F2_D4RT_RADIO_pairwise",
                "threshold": 0.22,
                "w_prob": 0.50,
                "w_d4rt": 0.18,
                "w_sem": 0.18,
                "w_inside": 0.08,
                "w_source": 0.00,
                "w_nested": 0.00,
                "w_competing": 0.01,
                "w_negative": 0.08,
                "graph_mix": 0.14,
                "w_soft_edge_cut": 0.02,
                "area_cap": 0.98,
                "min_area_fraction": 0.78,
                "description": "Source-preserving relaxed readout: high recall, graph-smoothed unary, no hard edge barrier.",
            },
            {
                "variant_id": "A10_relaxed_no_edge_source_preserve",
                "family": "real",
                "mode": "v94_greedy",
                "base_variant": "F2_D4RT_RADIO_pairwise",
                "threshold": 0.20,
                "w_prob": 0.50,
                "w_d4rt": 0.18,
                "w_sem": 0.18,
                "w_inside": 0.08,
                "w_source": 0.0,
                "w_nested": 0.0,
                "w_competing": 0.0,
                "w_negative": 0.08,
                "area_cap": 0.98,
                "min_area_fraction": 0.82,
                "description": "No-edge relaxed source-preserving baseline to separate edge repair from area shrinkage.",
            },
            {
                "variant_id": "A11_edge_confidence_high_recall",
                "family": "real",
                "mode": "v94_soft_edge_confidence",
                "base_variant": "F2_D4RT_RADIO_pairwise",
                "threshold": 0.18,
                "w_prob": 0.48,
                "w_d4rt": 0.18,
                "w_sem": 0.18,
                "w_inside": 0.08,
                "w_source": 0.0,
                "w_nested": 0.0,
                "w_competing": 0.0,
                "w_negative": 0.06,
                "graph_mix": 0.32,
                "w_soft_edge_cut": 0.01,
                "area_cap": 0.99,
                "min_area_fraction": 0.86,
                "description": "High-recall graph propagation variant to test whether Phase3A mainly lost AP through under-coverage.",
            },
            {
                "variant_id": "A12_soft_unknown_bg_relaxed",
                "family": "real",
                "mode": "v94_unknown_bg",
                "base_variant": "F2_D4RT_RADIO_pairwise",
                "threshold": 0.24,
                "unknown_margin": 0.24,
                "w_prob": 0.48,
                "w_d4rt": 0.18,
                "w_sem": 0.18,
                "w_inside": 0.08,
                "w_source": 0.01,
                "w_nested": 0.01,
                "w_competing": 0.02,
                "w_negative": 0.08,
                "area_cap": 0.96,
                "min_area_fraction": 0.74,
                "description": "Relaxed unknown/background branch after A7 over-rejected support.",
            },
        ]
    return [
        {
            "variant_id": "A0_current_whole_source_replay",
            "family": "baseline",
            "mode": "whole",
            "base_variant": "F0_whole_source_baseline",
            "description": "Replay whole-source region assignment to verify v94 materialization/evaluator wiring.",
        },
        {
            "variant_id": "A1_greedy_d4rt_radio_no_edge",
            "family": "real",
            "mode": "v94_greedy",
            "base_variant": "F2_D4RT_RADIO_pairwise",
            "threshold": 0.34,
            "w_prob": 0.50,
            "w_d4rt": 0.22,
            "w_sem": 0.20,
            "w_inside": 0.08,
            "w_source": 0.0,
            "w_nested": 0.0,
            "w_competing": 0.0,
            "w_negative": 0.16,
            "area_cap": 0.92,
            "min_area_fraction": 0.55,
            "description": "Greedy D4RT+RADIO unary without mask-edge barrier.",
        },
        {
            "variant_id": "A2_greedy_d4rt_radio_outer_edge",
            "family": "real",
            "mode": "v94_greedy",
            "base_variant": "F2_D4RT_RADIO_pairwise",
            "threshold": 0.32,
            "w_prob": 0.50,
            "w_d4rt": 0.20,
            "w_sem": 0.18,
            "w_inside": 0.08,
            "w_source": 0.10,
            "w_nested": 0.0,
            "w_competing": 0.0,
            "w_negative": 0.16,
            "area_cap": 0.90,
            "min_area_fraction": 0.52,
            "description": "Greedy unary with source outer-edge barrier.",
        },
        {
            "variant_id": "A3_greedy_d4rt_radio_nested_edge",
            "family": "real",
            "mode": "v94_greedy",
            "base_variant": "F2_D4RT_RADIO_pairwise",
            "threshold": 0.33,
            "w_prob": 0.50,
            "w_d4rt": 0.20,
            "w_sem": 0.18,
            "w_inside": 0.08,
            "w_source": 0.04,
            "w_nested": 0.14,
            "w_competing": 0.0,
            "w_negative": 0.16,
            "area_cap": 0.86,
            "min_area_fraction": 0.48,
            "description": "Greedy unary with nested-edge barrier.",
        },
        {
            "variant_id": "A4_greedy_d4rt_radio_competing_edge",
            "family": "real",
            "mode": "v94_greedy",
            "base_variant": "F2_D4RT_RADIO_pairwise",
            "threshold": 0.33,
            "w_prob": 0.50,
            "w_d4rt": 0.20,
            "w_sem": 0.18,
            "w_inside": 0.08,
            "w_source": 0.02,
            "w_nested": 0.0,
            "w_competing": 0.18,
            "w_negative": 0.18,
            "area_cap": 0.84,
            "min_area_fraction": 0.45,
            "description": "Greedy unary with competing-edge barrier.",
        },
        {
            "variant_id": "A5_greedy_d4rt_radio_all_edges",
            "family": "real",
            "mode": "v94_greedy",
            "base_variant": "F2_D4RT_RADIO_pairwise",
            "threshold": 0.33,
            "w_prob": 0.52,
            "w_d4rt": 0.18,
            "w_sem": 0.18,
            "w_inside": 0.08,
            "w_source": 0.06,
            "w_nested": 0.12,
            "w_competing": 0.16,
            "w_negative": 0.18,
            "area_cap": 0.82,
            "min_area_fraction": 0.42,
            "description": "Greedy unary with source/nested/competing edge barriers.",
        },
        {
            "variant_id": "A6_greedy_soft_d4rt_uncertainty",
            "family": "real",
            "mode": "v94_greedy",
            "base_variant": "F2_D4RT_RADIO_pairwise",
            "threshold": 0.28,
            "w_prob": 0.42,
            "w_d4rt": 0.34,
            "w_sem": 0.16,
            "w_inside": 0.08,
            "w_source": 0.04,
            "w_nested": 0.08,
            "w_competing": 0.10,
            "w_negative": 0.14,
            "area_cap": 0.88,
            "min_area_fraction": 0.55,
            "description": "Softer probabilistic D4RT witness field with uncertainty-tolerant expansion.",
        },
        {
            "variant_id": "A7_greedy_multiobject_with_unknown_bg",
            "family": "real",
            "mode": "v94_unknown_bg",
            "base_variant": "F2_D4RT_RADIO_pairwise",
            "threshold": 0.36,
            "unknown_margin": 0.10,
            "w_prob": 0.50,
            "w_d4rt": 0.20,
            "w_sem": 0.16,
            "w_inside": 0.08,
            "w_source": 0.07,
            "w_nested": 0.12,
            "w_competing": 0.16,
            "w_negative": 0.20,
            "area_cap": 0.78,
            "min_area_fraction": 0.36,
            "description": "Greedy assignment with explicit unknown/background outside option.",
        },
    ]


def _edge_confidence_signal(
    *,
    score: Any,
    boundary: Any,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_weight: np.ndarray,
    base_variant_idx: int,
    device: Any,
) -> tuple[Any, Any]:
    torch = base.torch
    n = int(score.numel())
    if n == 0 or edge_u.size == 0 or edge_v.size == 0:
        zeros = torch.zeros(n, dtype=torch.float32, device=device)
        return score, zeros
    e_u = torch.as_tensor(edge_u, dtype=torch.long, device=device)
    e_v = torch.as_tensor(edge_v, dtype=torch.long, device=device)
    if edge_weight.ndim == 2 and edge_weight.shape[1] > int(base_variant_idx):
        e_w_np = edge_weight[:, int(base_variant_idx)]
    elif edge_weight.ndim == 1:
        e_w_np = edge_weight
    else:
        e_w_np = np.ones(edge_u.shape[0], dtype=np.float32)
    e_w = torch.as_tensor(e_w_np, dtype=torch.float32, device=device)
    e_w = torch.nan_to_num(e_w, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    score_sum = torch.zeros(n, dtype=torch.float32, device=device)
    weight_sum = torch.zeros(n, dtype=torch.float32, device=device)
    score_sum.index_add_(0, e_u, e_w * score[e_v])
    score_sum.index_add_(0, e_v, e_w * score[e_u])
    weight_sum.index_add_(0, e_u, e_w)
    weight_sum.index_add_(0, e_v, e_w)
    neighbor_score = torch.where(weight_sum > 1.0e-6, score_sum / torch.clamp(weight_sum, min=1.0e-6), score)

    cut_sum = torch.zeros(n, dtype=torch.float32, device=device)
    cut_count = torch.zeros(n, dtype=torch.float32, device=device)
    low_conf = 1.0 - e_w
    cut_sum.index_add_(0, e_u, low_conf * boundary[e_v])
    cut_sum.index_add_(0, e_v, low_conf * boundary[e_u])
    one = torch.ones_like(low_conf)
    cut_count.index_add_(0, e_u, one)
    cut_count.index_add_(0, e_v, one)
    edge_cut_penalty = torch.where(cut_count > 0.0, cut_sum / torch.clamp(cut_count, min=1.0), torch.zeros_like(cut_count))
    return neighbor_score, edge_cut_penalty


def _v94_select_regions(
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
    del reference_selected
    torch = base.torch
    n = len(nodes)
    if n == 0:
        return set(), np.zeros(0, dtype=np.float32), {}
    if str(spec.get("mode")) == "whole":
        return set(range(n)), np.ones(n, dtype=np.float32), {"component_count_proxy": 1.0}

    sem = torch.as_tensor(features[:, 0], dtype=torch.float32, device=device)
    d4rt = torch.as_tensor(features[:, 1], dtype=torch.float32, device=device)
    inside = torch.as_tensor(features[:, 2], dtype=torch.float32, device=device)
    source_bar = torch.as_tensor(features[:, 3], dtype=torch.float32, device=device)
    nested_bar = torch.as_tensor(features[:, 4], dtype=torch.float32, device=device)
    competing_bar = torch.as_tensor(features[:, 5], dtype=torch.float32, device=device)
    negative = torch.as_tensor(features[:, 6], dtype=torch.float32, device=device)
    p = torch.as_tensor(prob, dtype=torch.float32, device=device)
    mode = str(spec.get("mode", ""))
    d4rt_original = d4rt
    d4rt_shuffle_l1 = torch.zeros((), dtype=torch.float32, device=device)
    d4rt_shuffle_corr = torch.ones((), dtype=torch.float32, device=device)
    if mode == "v94_shuffled_d4rt_witness" and n > 1:
        rng = np.random.default_rng(_stable_seed(f"{key[0]}|{key[1]}|{key[2]}|{spec.get('variant_id', '')}|d4rt"))
        perm_np = rng.permutation(n).astype(np.int64)
        perm = torch.as_tensor(perm_np, dtype=torch.long, device=device)
        d4rt = d4rt[perm]
        d4rt_shuffle_l1 = torch.mean(torch.abs(d4rt - d4rt_original))
        centered_a = d4rt_original - torch.mean(d4rt_original)
        centered_b = d4rt - torch.mean(d4rt)
        denom = torch.sqrt(torch.sum(centered_a * centered_a) * torch.sum(centered_b * centered_b))
        d4rt_shuffle_corr = torch.where(denom > 1.0e-8, torch.sum(centered_a * centered_b) / denom, torch.zeros_like(denom))

    object_score = (
        float(spec.get("w_prob", 0.5)) * p
        + float(spec.get("w_d4rt", 0.2)) * d4rt
        + float(spec.get("w_sem", 0.2)) * sem
        + float(spec.get("w_inside", 0.1)) * inside
        - float(spec.get("w_source", 0.0)) * source_bar
        - float(spec.get("w_nested", 0.0)) * nested_bar
        - float(spec.get("w_competing", 0.0)) * competing_bar
        - float(spec.get("w_negative", 0.0)) * negative
    )
    if mode == "v94_shuffled_d4rt_witness":
        object_score = (
            float(spec.get("w_d4rt", 0.18)) * d4rt
            + float(spec.get("w_sem", 0.18)) * sem
            + float(spec.get("w_inside", 0.08)) * inside
            - float(spec.get("w_source", 0.0)) * source_bar
            - float(spec.get("w_nested", 0.0)) * nested_bar
            - float(spec.get("w_competing", 0.0)) * competing_bar
            - float(spec.get("w_negative", 0.08)) * negative
        )
    boundary = torch.maximum(source_bar, torch.maximum(nested_bar, competing_bar))
    unknown = 0.40 * boundary + 0.25 * (1.0 - torch.clamp(d4rt, 0.0, 1.0)) + 0.20 * torch.abs(p - 0.5) + 0.15 * negative
    background = 0.45 * negative + 0.24 * competing_bar + 0.18 * source_bar + 0.13 * (1.0 - torch.clamp(sem, 0.0, 1.0))
    edge_cut_penalty = torch.zeros(n, dtype=torch.float32, device=device)
    if mode == "v94_soft_edge_confidence":
        neighbor_score, edge_cut_penalty = _edge_confidence_signal(
            score=object_score,
            boundary=boundary,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_weight=edge_weight,
            base_variant_idx=base_variant_idx,
            device=device,
        )
        graph_mix = float(spec.get("graph_mix", 0.0))
        object_score = (1.0 - graph_mix) * object_score + graph_mix * neighbor_score
        object_score = object_score - float(spec.get("w_soft_edge_cut", 0.0)) * edge_cut_penalty
    selected_t = object_score >= float(spec.get("threshold", 0.0))
    if mode == "v94_unknown_bg":
        margin = float(spec.get("unknown_margin", 0.0))
        selected_t = selected_t & ((object_score - unknown) >= -margin) & ((object_score - background) >= -margin)

    score_np = object_score.detach().cpu().numpy()
    selected = {int(i) for i in torch.nonzero(selected_t, as_tuple=False).flatten().detach().cpu().tolist()}
    seed_feature = d4rt.detach().cpu().numpy() if mode == "v94_shuffled_d4rt_witness" else features[:, 1]
    seed_like = {int(i) for i, value in enumerate((seed_feature + 0.35 * features[:, 0]).tolist()) if value >= 0.82}
    selected |= seed_like
    selected = base._enforce_min_area(selected, nodes, float(spec.get("min_area_fraction", 0.0)), score_np)
    selected = base._cap_by_area(selected, nodes, float(spec.get("area_cap", 1.0)), score_np, seed_like)
    diagnostics = {
        "object_score_mean": float(torch.mean(object_score).detach().cpu()),
        "unknown_score_mean": float(torch.mean(unknown).detach().cpu()),
        "background_score_mean": float(torch.mean(background).detach().cpu()),
        "edge_cut_penalty_mean": float(torch.mean(edge_cut_penalty).detach().cpu()),
        "graph_mix": float(spec.get("graph_mix", 0.0)),
        "seed_like_count": float(len(seed_like)),
        "component_count_proxy": 1.0,
        "d4rt_shuffle_l1_mean": float(d4rt_shuffle_l1.detach().cpu()),
        "d4rt_shuffle_corr": float(d4rt_shuffle_corr.detach().cpu()),
        "uses_original_unary_prob_in_score": float(0.0 if mode == "v94_shuffled_d4rt_witness" else 1.0),
    }
    return selected, score_np, diagnostics


def _v94_gate_rows(metric_rows: list[dict[str, str]], specs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool, dict[str, Any]]:
    phase0 = _read_json(V94_PHASE0 / "summary.json")
    v91_ap = _num(phase0.get("v91_best_MV_AP_window"))
    v91_ap50 = _num(phase0.get("v91_best_MV_AP50_window"))
    control_ap = _num(phase0.get("best_control_MV_AP_window"))
    control_ap50 = _num(phase0.get("best_control_MV_AP50_window"))
    required_ap = max(v91_ap + 0.004, control_ap + 0.006)
    required_ap50 = max(v91_ap50 + 0.008, control_ap50 + 0.0)
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
        progress_gate = mv_ap >= required_ap and mv_ap50 >= required_ap50
        provenance_gate = str(row.get("uses_gt_for_prediction", "False")).lower() == "false" and str(row.get("uses_future", "False")).lower() == "false"
        pass_gate = bool(is_real and progress_gate and provenance_gate)
        any_pass = any_pass or pass_gate
        if is_real and (not best_real or (mv_ap, mv_ap50) > (_num(best_real.get("mean_MV_AP_window")), _num(best_real.get("mean_MV_AP50_window")))):
            best_real = dict(row)
        gate_rows.append(
            {
                "schema_version": "stream4d_v94_phase3A_variant_gate_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": variant_id,
                "family": spec.get("family", ""),
                "MV_AP_window": mv_ap,
                "MV_AP50_window": mv_ap50,
                "required_MV_AP_window": required_ap,
                "required_MV_AP50_window": required_ap50,
                "phase3A_candidate_gate_pass": pass_gate,
                "progress_gate_pass": progress_gate,
                "provenance_gate_pass": provenance_gate,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        if is_real and not pass_gate:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v94_phase3A_failure_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "variant_id": variant_id,
                    "failure_type": "PHASE3A_GREEDY_NO_CANDIDATE_GATE",
                    "MV_AP_window": mv_ap,
                    "MV_AP50_window": mv_ap50,
                    "required_MV_AP_window": required_ap,
                    "required_MV_AP50_window": required_ap50,
                    "repair_direction": "If A variants undercut area, relax edge/background; if over-broad, strengthen competing/nested barrier; if all A variants flat, move to Phase3B.",
                    "created_at": _created_at(),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    return gate_rows, failure_rows, any_pass, best_real


def _postprocess(out: Path, started: float, variant_set: str) -> dict[str, Any]:
    specs = _v94_variant_specs(variant_set)
    _rewrite_csv_schema(out / "variant_config_rows.csv", "stream4d_v94_phase3A_variant_config_v1")
    _rewrite_csv_schema(out / "generated_mask_rows.csv", "stream4d_v94_phase3A_generated_mask_v1")
    _rewrite_csv_schema(out / "assignment_summary_rows.csv", "stream4d_v94_phase3A_assignment_summary_v1")
    _rewrite_csv_schema(out / "source_failure_rows.csv", "stream4d_v94_phase3A_source_failure_v1")
    _rewrite_csv_schema(out / "variant_metric_rows.csv", "stream4d_v94_phase3A_variant_metric_v1")
    metric_rows = _read_csv(out / "variant_metric_rows.csv")
    gate_rows, failure_rows, any_pass, best_real = _v94_gate_rows(metric_rows, specs)
    _write_csv(out / "variant_gate_rows.csv", gate_rows)
    _write_csv(out / "variant_failure_rows.csv", failure_rows)
    assignment_summary = _read_csv(out / "assignment_summary_rows.csv")
    component_rows = [
        {
            "schema_version": "stream4d_v94_phase3A_component_summary_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "variant_id": row.get("variant_id", ""),
            "scene_id": row.get("scene_id", ""),
            "frame_id": row.get("frame_id", ""),
            "source_mask_id": row.get("source_mask_id", ""),
            "component_extraction_mode": "greedy_selected_region_set_no_explicit_component_pooling",
            "component_count_proxy": row.get("component_count_proxy", 1),
            "selected_region_count": row.get("selected_region_count", ""),
            "total_region_count": row.get("total_region_count", ""),
            "generated_mask_area_ratio": row.get("generated_mask_area_ratio", ""),
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for row in assignment_summary
    ]
    _write_csv(out / "component_rows.csv", component_rows)
    field_manifest = {
        "schema": "stream4d_v94_phase3A_field_artifact_manifest_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "field_unary_source": "Reuses v93 Phase5 GPU/Triton field_shards/*.npz as unary feature input; no full CSV unary table is emitted.",
        "field_assignment_source": "assignment_summary_rows.csv plus generated masks; full region assignment CSV is intentionally not expanded.",
        "method_field_mode": "npz_shards_input_compact_csv_outputs",
        "source_field_root": "outputs/audit/v93_phase5_boundary_affinity_field/field_shards",
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_json(out / "field_artifact_manifest.json", field_manifest)
    base_summary = _read_json(out / "summary.json")
    summary = dict(base_summary)
    summary.update(
        {
            "schema": "stream4d_v94_phase3A_greedy_assignment_summary_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "variant_set": variant_set,
            "decision": "PASS_V94_PHASE3A_CANDIDATE_GATE" if any_pass else "NO_GO_V94_PHASE3A_GREEDY_NO_CANDIDATE_GATE",
            "any_phase3A_candidate_gate_pass": any_pass,
            "best_A_variant_id": best_real.get("variant_id", ""),
            "best_A_MV_AP_window": best_real.get("mean_MV_AP_window", ""),
            "best_A_MV_AP50_window": best_real.get("mean_MV_AP50_window", ""),
            "field_artifact_mode": "v93_npz_field_shards_reused; no full field_unary_rows.csv expansion",
            "duration_sec": time.time() - started,
            "row_counts": {
                **base_summary.get("row_counts", {}),
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
    parser.add_argument("--variant-set", choices=["main", "edge_repair", "d4rt_control"], default="main")
    parser.add_argument("--field-root", default=str(ROOT / "outputs/audit/v93_phase5_boundary_affinity_field"))
    parser.add_argument("--source-container-rows", default=str(ROOT / "outputs/audit/v94_phase1_canonical_graph/container_rows.csv"))
    parser.add_argument("--region-node-rows", default=str(ROOT / "outputs/audit/v94_phase1_canonical_graph/region_node_rows.csv"))
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--progress-every-shards", type=int, default=1)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    global RUN_ID
    started = time.time()
    out = Path(args.output_root)
    if out.exists() and args.clean:
        shutil.rmtree(out)
    RUN_ID = DEFAULT_RUN_ID if args.variant_set == "main" else f"v94_phase3A_{args.variant_set}_gpu"
    base.PHASE_ID = PHASE_ID
    base.RUN_ID = RUN_ID
    base.OUT = OUT
    base._variant_specs = lambda: _v94_variant_specs(args.variant_set)
    base._select_regions = _v94_select_regions
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
    summary = _postprocess(out, started, args.variant_set)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run(parse_args())
