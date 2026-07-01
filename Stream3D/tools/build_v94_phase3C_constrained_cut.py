#!/usr/bin/env python3
"""Run v94 Phase3C constrained graph-cut approximation variants."""

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
from tools import build_v94_phase3B_random_walker as phase3b  # noqa: E402


PHASE_ID = "v94_phase3C_constrained_cut"
RUN_ID = "v94_phase3C_constrained_cut_gpu"
OUT = ROOT / "outputs/audit/v94_phase3C_constrained_cut"
V94_PHASE0 = ROOT / "outputs/audit/v94_phase0_fact_lock"
V94_PHASE3A_REPAIR = ROOT / "outputs/audit/v94_phase3A_greedy_assignment_edge_repair"
V94_PHASE3B = ROOT / "outputs/audit/v94_phase3B_random_walker"
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
            "variant_id": "C0_current_whole_source_replay",
            "family": "baseline",
            "mode": "whole",
            "base_variant": "F0_whole_source_baseline",
            "description": "Whole-source replay baseline in the Phase3C evaluator path.",
        },
        {
            "variant_id": "C1_binary_cut_each_object",
            "family": "real",
            "mode": "v94_constrained_cut",
            "base_variant": "F2_D4RT_RADIO_pairwise",
            "threshold": 0.02,
            "pairwise_weight": 0.32,
            "edge_power": 1.0,
            "barrier_weight": 0.04,
            "iters": 4,
            "area_cap": 0.94,
            "min_area_fraction": 0.58,
            "description": "Binary constrained cut with moderate pairwise smoothing.",
        },
        {
            "variant_id": "C2_binary_cut_with_multiobject_WTA",
            "family": "real",
            "mode": "v94_constrained_cut",
            "base_variant": "F2_D4RT_RADIO_pairwise",
            "threshold": -0.01,
            "pairwise_weight": 0.26,
            "edge_power": 0.9,
            "barrier_weight": 0.02,
            "iters": 5,
            "area_cap": 0.96,
            "min_area_fraction": 0.68,
            "description": "Source-preserving cut approximation; WTA conflict count is tracked as zero for current one-object-per-source materialization.",
        },
        {
            "variant_id": "C3_alpha_expansion_approx_3round",
            "family": "real",
            "mode": "v94_constrained_cut",
            "base_variant": "F2_D4RT_RADIO_pairwise",
            "threshold": -0.04,
            "pairwise_weight": 0.36,
            "edge_power": 0.85,
            "barrier_weight": 0.02,
            "iters": 9,
            "area_cap": 0.985,
            "min_area_fraction": 0.80,
            "description": "Three-round alpha-expansion-like approximation with high-recall fill.",
        },
        {
            "variant_id": "C4_cut_with_strong_edge_barrier",
            "family": "real",
            "mode": "v94_constrained_cut",
            "base_variant": "F4_RADIO_edge_barrier",
            "threshold": 0.08,
            "pairwise_weight": 0.30,
            "edge_power": 1.8,
            "barrier_weight": 0.22,
            "iters": 5,
            "area_cap": 0.86,
            "min_area_fraction": 0.42,
            "description": "Strong edge-barrier cut to test whether explicit boundary stopping helps AP50.",
        },
        {
            "variant_id": "C5_cut_with_soft_edge_barrier",
            "family": "real",
            "mode": "v94_constrained_cut",
            "base_variant": "F2_D4RT_RADIO_pairwise",
            "threshold": -0.02,
            "pairwise_weight": 0.34,
            "edge_power": 0.95,
            "barrier_weight": 0.06,
            "iters": 6,
            "area_cap": 0.97,
            "min_area_fraction": 0.74,
            "description": "Soft edge-barrier cut after A/B showed hard barriers are destructive.",
        },
        {
            "variant_id": "C6_cut_with_D4RT_negative_veto",
            "family": "real",
            "mode": "v94_constrained_cut",
            "base_variant": "F2_D4RT_RADIO_pairwise",
            "threshold": 0.04,
            "pairwise_weight": 0.28,
            "edge_power": 1.0,
            "barrier_weight": 0.03,
            "negative_veto": 0.24,
            "iters": 5,
            "area_cap": 0.90,
            "min_area_fraction": 0.55,
            "description": "D4RT/negative evidence veto variant to test whether hard negatives reduce over-broad masks.",
        },
    ]


def _cut_energy(margin: Any, selected: Any, edge_u_t: Any, edge_v_t: Any, edge_w_t: Any, pairwise_weight: float) -> Any:
    torch = base.torch
    data_energy = torch.sum(torch.where(selected, torch.clamp(0.5 - margin, min=0.0), torch.clamp(0.5 + margin, min=0.0)))
    if edge_u_t.numel() == 0:
        return data_energy
    disagree = selected[edge_u_t] != selected[edge_v_t]
    return data_energy + float(pairwise_weight) * torch.sum(edge_w_t[disagree])


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
        return set(range(n)), np.ones(n, dtype=np.float32), {"cut_iteration_count": 0.0, "label_changed_region_count": 0.0}

    sem = torch.as_tensor(features[:, 0], dtype=torch.float32, device=device)
    d4rt = torch.as_tensor(features[:, 1], dtype=torch.float32, device=device)
    inside = torch.as_tensor(features[:, 2], dtype=torch.float32, device=device)
    source_bar = torch.as_tensor(features[:, 3], dtype=torch.float32, device=device)
    nested_bar = torch.as_tensor(features[:, 4], dtype=torch.float32, device=device)
    competing_bar = torch.as_tensor(features[:, 5], dtype=torch.float32, device=device)
    negative = torch.as_tensor(features[:, 6], dtype=torch.float32, device=device)
    p = torch.as_tensor(prob, dtype=torch.float32, device=device)
    boundary = torch.maximum(source_bar, torch.maximum(nested_bar, competing_bar))

    object_unary = 0.50 * p + 0.20 * d4rt + 0.18 * sem + 0.10 * inside
    bg_risk = 0.42 * negative + 0.22 * competing_bar + 0.16 * source_bar + 0.10 * nested_bar + 0.10 * (1.0 - sem)
    margin = object_unary - bg_risk - float(spec.get("negative_veto", 0.0)) * negative
    margin = margin - float(spec.get("barrier_weight", 0.0)) * boundary

    e_u_t = torch.as_tensor(edge_u, dtype=torch.long, device=device)
    e_v_t = torch.as_tensor(edge_v, dtype=torch.long, device=device)
    e_w_t = phase3b._edge_affinity(
        edge_weight=edge_weight,
        edge_u=edge_u,
        edge_v=edge_v,
        boundary=boundary,
        base_variant_idx=base_variant_idx,
        spec=spec,
        device=device,
    )
    threshold = float(spec.get("threshold", 0.0))
    selected_t = margin >= threshold
    seed_score = d4rt + 0.45 * sem + 0.20 * inside
    pos_seed = seed_score >= torch.quantile(seed_score, 0.82)
    selected_t = selected_t | pos_seed
    energy_before = _cut_energy(margin, selected_t, e_u_t, e_v_t, e_w_t, float(spec.get("pairwise_weight", 0.0)))
    prev = selected_t.clone()
    changed_total = torch.tensor(0, dtype=torch.long, device=device)
    for _ in range(max(1, int(spec.get("iters", 1)))):
        neighbor_keep = phase3b._weighted_neighbor_average(selected_t.float(), e_u_t, e_v_t, e_w_t)
        proposal_score = margin + float(spec.get("pairwise_weight", 0.0)) * (neighbor_keep - 0.5)
        new_selected = proposal_score >= threshold
        new_selected = new_selected | pos_seed
        changed = torch.count_nonzero(new_selected != selected_t)
        changed_total = changed_total + changed
        selected_t = new_selected
        if int(changed.detach().cpu()) == 0:
            break
    energy_after = _cut_energy(margin, selected_t, e_u_t, e_v_t, e_w_t, float(spec.get("pairwise_weight", 0.0)))
    score = torch.sigmoid(4.0 * (margin + float(spec.get("pairwise_weight", 0.0)) * (phase3b._weighted_neighbor_average(selected_t.float(), e_u_t, e_v_t, e_w_t) - 0.5)))
    score_np = score.detach().cpu().numpy()
    selected = {int(i) for i in torch.nonzero(selected_t, as_tuple=False).flatten().detach().cpu().tolist()}
    must_keep = {int(i) for i in torch.nonzero(pos_seed, as_tuple=False).flatten().detach().cpu().tolist()}
    selected |= must_keep
    selected_before_area = len(selected)
    selected = base._enforce_min_area(selected, nodes, float(spec.get("min_area_fraction", 0.0)), score_np)
    selected = base._cap_by_area(selected, nodes, float(spec.get("area_cap", 1.0)), score_np, must_keep)
    component_count = phase3b._selected_component_count(selected, edge_u, edge_v, edge_weight, base_variant_idx)
    diagnostics = {
        "cut_iteration_count": float(spec.get("iters", 0)),
        "energy_before": float(energy_before.detach().cpu()),
        "energy_after": float(energy_after.detach().cpu()),
        "energy_delta": float((energy_after - energy_before).detach().cpu()),
        "label_changed_region_count": float(changed_total.detach().cpu()),
        "object_area_change_ratio": float(len(selected) / max(1, selected_before_area)),
        "region_ownership_conflict_count": 0.0,
        "cannot_link_violation_count": 0.0,
        "component_count_proxy": float(component_count),
        "connected_component_split_count": float(component_count),
        "cut_score_mean": float(torch.mean(score).detach().cpu()),
        "positive_seed_fraction": float(torch.count_nonzero(pos_seed).detach().cpu()) / max(1, n),
    }
    return selected, score_np, diagnostics


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def _phase3c_gate_rows(metric_rows: list[dict[str, str]], specs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool, dict[str, Any]]:
    phase0 = _read_json(V94_PHASE0 / "summary.json")
    phase3a = _read_json(V94_PHASE3A_REPAIR / "summary.json")
    phase3b_summary = _read_json(V94_PHASE3B / "summary.json")
    best_prev_ap = max(_num(phase3a.get("best_A_MV_AP_window")), _num(phase3b_summary.get("best_B_MV_AP_window")))
    best_prev_ap50 = max(_num(phase3a.get("best_A_MV_AP50_window")), _num(phase3b_summary.get("best_B_MV_AP50_window")))
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
        improvement_gate = (mv_ap >= best_prev_ap + 0.003) and (mv_ap50 >= best_prev_ap50 + 0.006)
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
                "schema_version": "stream4d_v94_phase3C_variant_gate_v1",
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
                "phase3C_improvement_gate_pass": improvement_gate,
                "locked_control_gate_pass": locked_control_gate,
                "provenance_gate_pass": provenance_gate,
                "phase3C_candidate_gate_pass": pass_gate,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        if is_real and not pass_gate:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v94_phase3C_failure_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "variant_id": variant_id,
                    "failure_type": "PHASE3C_CONSTRAINED_CUT_NO_CANDIDATE_GATE",
                    "MV_AP_window": mv_ap,
                    "MV_AP50_window": mv_ap50,
                    "best_previous_MV_AP_window": best_prev_ap,
                    "best_previous_MV_AP50_window": best_prev_ap50,
                    "repair_direction": "If strong cuts under-cover, use soft/high-recall cut; if soft cuts remain below A/B, enter Phase3D component pooling.",
                    "created_at": _created_at(),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    return gate_rows, failure_rows, any_pass, best_real


def _postprocess(out: Path, started: float) -> dict[str, Any]:
    specs = _variant_specs()
    phase3a._rewrite_csv_schema(out / "variant_config_rows.csv", "stream4d_v94_phase3C_variant_config_v1")
    phase3a._rewrite_csv_schema(out / "generated_mask_rows.csv", "stream4d_v94_phase3C_generated_mask_v1")
    phase3a._rewrite_csv_schema(out / "assignment_summary_rows.csv", "stream4d_v94_phase3C_assignment_summary_v1")
    phase3a._rewrite_csv_schema(out / "source_failure_rows.csv", "stream4d_v94_phase3C_source_failure_v1")
    phase3a._rewrite_csv_schema(out / "variant_metric_rows.csv", "stream4d_v94_phase3C_variant_metric_v1")
    metric_rows = _read_csv(out / "variant_metric_rows.csv")
    gate_rows, failure_rows, any_pass, best_real = _phase3c_gate_rows(metric_rows, specs)
    _write_csv(out / "variant_gate_rows.csv", gate_rows)
    _write_csv(out / "variant_failure_rows.csv", failure_rows)

    assignment_rows = _read_csv(out / "assignment_summary_rows.csv")
    by_variant: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in assignment_rows:
        by_variant[str(row.get("variant_id", ""))].append(row)
    cut_rows: list[dict[str, Any]] = []
    for spec in specs:
        rows = by_variant.get(spec["variant_id"], [])
        cut_rows.append(
            {
                "schema_version": "stream4d_v94_phase3C_cut_summary_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": spec["variant_id"],
                "family": spec["family"],
                "cut_iteration_count_mean": _mean([_num(row.get("cut_iteration_count")) for row in rows]),
                "energy_before_mean": _mean([_num(row.get("energy_before")) for row in rows]),
                "energy_after_mean": _mean([_num(row.get("energy_after")) for row in rows]),
                "energy_delta_mean": _mean([_num(row.get("energy_delta")) for row in rows]),
                "label_changed_region_count_mean": _mean([_num(row.get("label_changed_region_count")) for row in rows]),
                "object_area_change_ratio_mean": _mean([_num(row.get("object_area_change_ratio")) for row in rows]),
                "region_ownership_conflict_count_sum": sum(_num(row.get("region_ownership_conflict_count")) for row in rows),
                "cannot_link_violation_count_sum": sum(_num(row.get("cannot_link_violation_count")) for row in rows),
                "component_count_proxy_mean": _mean([_num(row.get("component_count_proxy")) for row in rows]),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    _write_csv(out / "cut_diagnostic_rows.csv", cut_rows)
    _write_csv(
        out / "constrained_cut_config_rows.csv",
        [
            {
                "schema_version": "stream4d_v94_phase3C_constrained_cut_config_v1",
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
            "schema_version": "stream4d_v94_phase3C_component_summary_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "variant_id": row.get("variant_id", ""),
            "scene_id": row.get("scene_id", ""),
            "frame_id": row.get("frame_id", ""),
            "source_mask_id": row.get("source_mask_id", ""),
            "component_extraction_mode": "constrained_cut_selected_region_graph_components_weight_ge_0.50_proxy",
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
            "schema": "stream4d_v94_phase3C_constrained_cut_summary_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "decision": "PASS_V94_PHASE3C_CANDIDATE_GATE" if any_pass else "NO_GO_V94_PHASE3C_CONSTRAINED_CUT_NO_CANDIDATE_GATE",
            "any_phase3C_candidate_gate_pass": any_pass,
            "best_C_variant_id": best_real.get("variant_id", ""),
            "best_C_MV_AP_window": best_real.get("mean_MV_AP_window", ""),
            "best_C_MV_AP50_window": best_real.get("mean_MV_AP50_window", ""),
            "field_artifact_mode": "v93_npz_field_shards_reused; constrained cut runs on GPU tensors, evaluator remains CSV/CPU",
            "duration_sec": time.time() - started,
            "row_counts": {
                **base_summary.get("row_counts", {}),
                "cut_diagnostic_rows": len(cut_rows),
                "constrained_cut_config_rows": len(specs),
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
