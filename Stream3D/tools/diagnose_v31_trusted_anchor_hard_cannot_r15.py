from __future__ import annotations

import argparse
import gc
import itertools
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.diagnose_v30_cannot_link_clique import _negative_pair_counts, _scene_records_and_measurements
from tools.diagnose_v31_tube_assignment_r8 import _negative_adjacency
from tools.diagnose_v31_trusted_anchor_assignment_r14 import (
    _build_positive_support,
    _control_factor_rows,
    _control_seed_rows,
    _load_seed_role_rows,
    _mean,
    _select_anchor_seeds,
    _selected_from_assignments,
)
from tools.run_v28_proposal_selection import _load_gt_labels
from tools.run_v30_object_slot_ownership import _read_split
from tools.run_v31_seed_anchor_lowtail import (
    _aggregate_solver_rows,
    _eval_v29_selected,
    _f,
    _load_split_repair_factor_rows,
    _row_core_ids,
    _write_csv,
)


def _negative_to_slot(
    tid: int,
    slot_ids: set[int],
    negative_adjacency: dict[int, list[tuple[int, int]]],
    *,
    min_count: int,
) -> int:
    total = 0
    for other, count in negative_adjacency.get(int(tid), []):
        if int(count) >= int(min_count) and int(other) in slot_ids:
            total += int(count)
    return int(total)


def _hard_assign(
    support: dict[int, dict[int, float]],
    *,
    negative_adjacency: dict[int, list[tuple[int, int]]],
    negative_min_count: int,
    unknown_threshold: float,
    margin_ratio: float,
    margin_add: float,
    max_slot_tubes: int,
) -> tuple[dict[int, int], dict[str, Any]]:
    assignments: dict[int, int] = {}
    slot_ids: dict[int, set[int]] = defaultdict(set)
    ordered = sorted(
        support,
        key=lambda tid: (max(support[int(tid)].values()) if support[int(tid)] else 0.0, len(support[int(tid)])),
        reverse=True,
    )
    rejected_unknown = 0
    rejected_margin = 0
    rejected_capacity = 0
    rejected_hard_cannot = 0

    for tid in ordered:
        feasible: list[tuple[float, int]] = []
        blocked = False
        for slot_idx, positive in support[int(tid)].items():
            ids = slot_ids[int(slot_idx)]
            if int(max_slot_tubes) > 0 and len(ids) >= int(max_slot_tubes):
                continue
            neg = _negative_to_slot(int(tid), ids, negative_adjacency, min_count=int(negative_min_count))
            if neg > 0:
                blocked = True
                continue
            feasible.append((float(positive), int(slot_idx)))
        if not feasible:
            if blocked:
                rejected_hard_cannot += 1
            else:
                rejected_capacity += 1
            continue
        feasible.sort(reverse=True)
        best_score, best_slot = feasible[0]
        second_score = feasible[1][0] if len(feasible) > 1 else 0.0
        if best_score < float(unknown_threshold):
            rejected_unknown += 1
            continue
        if best_score < float(margin_ratio) * float(second_score) + float(margin_add):
            rejected_margin += 1
            continue
        assignments[int(tid)] = int(best_slot)
        slot_ids[int(best_slot)].add(int(tid))

    remaining_neg = 0
    for _slot_idx, ids in slot_ids.items():
        for tid in ids:
            for other, count in negative_adjacency.get(int(tid), []):
                if int(tid) < int(other) and int(other) in ids and int(count) >= int(negative_min_count):
                    remaining_neg += int(count)
    return assignments, {
        "assigned_tube_count": int(len(assignments)),
        "active_slot_count": int(sum(1 for ids in slot_ids.values() if ids)),
        "rejected_unknown_tube_count": int(rejected_unknown),
        "rejected_margin_tube_count": int(rejected_margin),
        "rejected_capacity_tube_count": int(rejected_capacity),
        "rejected_hard_cannot_tube_count": int(rejected_hard_cannot),
        "remaining_negative_weight": int(remaining_neg),
    }


def _scene_result(
    *,
    scene: str,
    control_kind: str,
    role_rows: list[dict[str, Any]],
    factor_rows: list[dict[str, Any]],
    gt_labels: dict[int, int],
    config: dict[str, Any],
    negative_adjacency: dict[int, list[tuple[int, int]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    solver = str(config["solver"]) if control_kind == "real" else f"{config['solver']}_{control_kind}"
    t0 = time.time()
    seeds, seed_stats = _select_anchor_seeds(
        role_rows,
        seed_roles=set(str(config["seed_roles"]).split("+")),
        max_seed_slots=int(config["max_seed_slots"]),
        seed_min_new=int(config["seed_min_new"]),
        seed_max_overlap=float(config["seed_max_overlap"]),
    )
    support, support_stats = _build_positive_support(
        role_rows=role_rows,
        factor_rows=factor_rows,
        seeds=seeds,
        evidence_roles=set(str(config["evidence_roles"]).split("+")) if config["evidence_roles"] else set(),
        include_split_factors=bool(config["include_split_factors"]),
        evidence_topn=int(config["evidence_topn"]),
        evidence_min_overlap=float(config["evidence_min_overlap"]),
        evidence_top_slots=int(config["evidence_top_slots"]),
        support_power=float(config["support_power"]),
        base_prior=float(config["base_prior"]),
        temporal_gain=float(config["temporal_gain"]),
        risk_gain=float(config["risk_gain"]),
    )
    assignments, assign_stats = _hard_assign(
        support,
        negative_adjacency=negative_adjacency,
        negative_min_count=int(config["negative_min_count"]),
        unknown_threshold=float(config["unknown_threshold"]),
        margin_ratio=float(config["margin_ratio"]),
        margin_add=float(config["margin_add"]),
        max_slot_tubes=int(config["max_slot_tubes"]),
    )
    selected = _selected_from_assignments(
        scene,
        seeds,
        assignments,
        solver,
        min_slot_tubes=int(config["min_slot_tubes"]),
    )
    runtime = float(time.time() - t0)
    metrics = _eval_v29_selected(selected, gt_labels)
    selected_ids = set().union(*(set(_row_core_ids(row)) for row in selected)) if selected else set()
    split_ids = set().union(*(set(_row_core_ids(row)) for row in factor_rows)) if factor_rows else set()
    broad_rows = [row for row in role_rows if str(row.get("seed_role")) == "broad_observation"]
    broad_explained = 0
    for row in broad_rows:
        core = set(_row_core_ids(row))
        if core and selected_ids & core:
            broad_explained += 1
    row = {
        "scene": scene,
        "solver": solver,
        "control_kind": control_kind,
        "slot_count": int(len(role_rows) + len(factor_rows)),
        "active_slot_count": int(len(selected)),
        "owned_tube_ratio": metrics["owned_tube_ratio"],
        "unknown_tube_ratio": metrics["unknown_tube_ratio"],
        "coverage_factor_explained_ratio": float(len(selected_ids & split_ids) / max(len(split_ids), 1)),
        "broad_observation_explained_ratio": float(broad_explained / max(len(broad_rows), 1)),
        "cannot_link_violation_count": assign_stats["remaining_negative_weight"],
        "boundary_violation_rate": _mean([_f(item, "boundary_score") for item in seeds]) or 0.0,
        "appearance_consistency": 1.0,
        "motion_consistency": _mean([_f(item, "multi_frame_support_score") for item in seeds]) or 0.0,
        "solver_runtime_sec": runtime,
        "solver_iterations": 1,
        "ARI": metrics["ARI"],
        "purity": metrics["purity"],
        "completeness": metrics["completeness"],
        "overmerge": metrics["overmerge"],
        "oversplit": metrics["oversplit"],
        "num_moves_attempted": int(support_stats["support_tube_count"]),
        "num_moves_accepted": int(assign_stats["assigned_tube_count"]),
        **{f"r15_{key}": value for key, value in {**seed_stats, **support_stats, **assign_stats}.items()},
        **{f"cfg_{key}": value for key, value in config.items() if key != "solver"},
    }
    selected_rows = []
    for rank, item in enumerate(selected):
        selected_rows.append(
            {
                "scene": scene,
                "solver": solver,
                "control_kind": control_kind,
                "rank": int(rank),
                "proposal_id": item.get("proposal_id"),
                "proposal_type": item.get("proposal_type"),
                "seed_id": item.get("seed_id"),
                "seed_role": item.get("seed_role"),
                "seed_proposal_id": item.get("seed_proposal_id"),
                "core_tube_count": len(_row_core_ids(item)),
                "core_tube_ids": ";".join(str(v) for v in _row_core_ids(item)),
                "uses_gt_for_prediction": False,
            }
        )
    return row, selected_rows


def _search_configs() -> list[dict[str, Any]]:
    configs = []
    grid = itertools.product(
        [
            ("trusted_anchor", "trusted_anchor+candidate_anchor+coverage_candidate", True),
            ("trusted_anchor+candidate_anchor", "trusted_anchor+candidate_anchor+coverage_candidate+broad_observation", True),
        ],
        [56, 80],
        [0.72, 0.86],
        [0.03, 0.08, 0.15],
        [0.2, 0.6, 1.0],
        [1, 3],
        [2, 3],
    )
    for idx, (
        role_bundle,
        max_seed_slots,
        seed_max_overlap,
        evidence_min_overlap,
        unknown_threshold,
        negative_min_count,
        min_slot_tubes,
    ) in enumerate(grid):
        seed_roles, evidence_roles, include_split = role_bundle
        configs.append(
            {
                "solver": f"R15_hard_cannot_search_{idx:04d}",
                "seed_roles": seed_roles,
                "evidence_roles": evidence_roles,
                "include_split_factors": bool(include_split),
                "max_seed_slots": int(max_seed_slots),
                "seed_min_new": 3,
                "seed_max_overlap": float(seed_max_overlap),
                "evidence_topn": 1500,
                "evidence_min_overlap": float(evidence_min_overlap),
                "evidence_top_slots": 1,
                "support_power": 0.5,
                "base_prior": 2.0,
                "temporal_gain": 0.30,
                "risk_gain": 0.35,
                "negative_min_count": int(negative_min_count),
                "unknown_threshold": float(unknown_threshold),
                "margin_ratio": 1.0,
                "margin_add": 0.0,
                "max_slot_tubes": 0,
                "min_slot_tubes": int(min_slot_tubes),
            }
        )
    return configs


def run(args: argparse.Namespace) -> dict[str, Any]:
    scenes = _read_split(Path(args.split))
    gt_by_scene = {
        scene: _load_gt_labels(
            Path(args.cache_root),
            scene,
            int(args.max_tubes_per_window),
            int(args.image_width),
            int(args.image_height),
        )
        for scene in scenes
    }
    role_rows = _load_seed_role_rows(Path(args.seed_role_csv))
    factor_rows, _stats = _load_split_repair_factor_rows(Path(args.proposal_row_csv))
    role_controls = _control_seed_rows(role_rows)
    factor_controls = _control_factor_rows(factor_rows)
    requested_controls = [part.strip() for part in str(args.controls).split(",") if part.strip()]
    configs = _search_configs()

    measurement_args = argparse.Namespace(
        cache_root=args.cache_root,
        max_tubes_per_window=args.max_tubes_per_window,
        image_width=args.image_width,
        image_height=args.image_height,
    )
    negative_adjacency_by_scene = {}
    for scene in scenes:
        print(f"[r15] building negative adjacency {scene}", flush=True)
        stream, records, masks_by_frame, measurements, diag = _scene_records_and_measurements(measurement_args, scene)
        counts = _negative_pair_counts(measurements)
        negative_adjacency_by_scene[scene] = _negative_adjacency(counts)
        print(f"[r15] {scene} measurements={diag.get('measurement_count')} negative_pairs={len(counts)}", flush=True)
        del stream, records, masks_by_frame, measurements, diag, counts
        gc.collect()

    scene_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for cfg_idx, config in enumerate(configs):
        print(f"[r15] config {cfg_idx + 1}/{len(configs)} {config['solver']}", flush=True)
        for control_kind in requested_controls:
            role_items = role_controls[control_kind]
            factor_items = factor_controls[control_kind]
            for scene in scenes:
                scene_role_rows = [row for row in role_items if str(row.get("scene")) == scene]
                scene_factor_rows = [row for row in factor_items if str(row.get("scene")) == scene]
                row, selected = _scene_result(
                    scene=scene,
                    control_kind=control_kind,
                    role_rows=scene_role_rows,
                    factor_rows=scene_factor_rows,
                    gt_labels=gt_by_scene[scene],
                    config=config,
                    negative_adjacency=negative_adjacency_by_scene[scene],
                )
                scene_rows.append(row)
                selected_rows.extend(selected)

    aggregate = _aggregate_solver_rows(scene_rows)
    out_dir = Path(args.out_dir)
    _write_csv(out_dir / args.summary_name, scene_rows + aggregate)
    if args.selected_name:
        _write_csv(out_dir / args.selected_name, selected_rows)
    best = max(
        (row for row in aggregate if row.get("control_kind") == "real"),
        key=lambda row: (float(row.get("ARI") or 0.0), float(row.get("purity") or 0.0)),
        default={},
    )
    return {"config_count": len(configs), "best_real": best}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose v31 r15 trusted-anchor hard cannot-link assignment.")
    parser.add_argument("--controls", default="real")
    parser.add_argument("--out-dir", default="outputs/audit/v31_slot_ownership")
    parser.add_argument("--summary-name", default="r15_hard_cannot_assignment_real_summary.csv")
    parser.add_argument("--selected-name", default="")
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v26_occupancy_d5_warmstart64_probe5_topup20_patch2")
    parser.add_argument("--max-tubes-per-window", type=int, default=640)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--seed-role-csv", default="outputs/audit/v31_seed_roles/seed_role_rows.csv")
    parser.add_argument(
        "--proposal-row-csv",
        default="outputs/audit/v28_proposal_oracle_repair_temporal_track_640_pruned_shared2_t20_probe5/v28_proposal_oracle_repair_temporal_track_640_pruned_shared2_t20_probe5_proposal_rows.csv",
    )
    return parser


def main() -> None:
    print(run(build_parser().parse_args()), flush=True)


if __name__ == "__main__":
    main()
