from __future__ import annotations

import argparse
import gc
import itertools
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tools.diagnose_v30_cannot_link_clique import _negative_pair_counts, _scene_records_and_measurements
from tools.diagnose_v31_e8_graph_split_r13 import _positive_pair_counts
from tools.diagnose_v31_trusted_anchor_assignment_r14 import (
    _control_factor_rows,
    _control_seed_rows,
    _load_seed_role_rows,
    _mean,
    _overlap_min_norm,
    _row_weight,
    _select_anchor_seeds,
    _selected_from_assignments,
)
from tools.diagnose_v31_trusted_anchor_hard_cannot_r15 import _hard_assign
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


def _adjacency(counts: Counter[tuple[int, int]]) -> dict[int, list[tuple[int, int]]]:
    out: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (left, right), count in counts.items():
        out[int(left)].append((int(right), int(count)))
        out[int(right)].append((int(left), int(count)))
    return out


def _positive_to_seed(
    tid: int,
    seed_ids: set[int],
    positive_adjacency: dict[int, list[tuple[int, int]]],
    *,
    min_count: int,
) -> int:
    total = 0
    for other, count in positive_adjacency.get(int(tid), []):
        if int(count) >= int(min_count) and int(other) in seed_ids:
            total += int(count)
    return int(total)


def _build_positive_gated_support(
    *,
    role_rows: list[dict[str, Any]],
    factor_rows: list[dict[str, Any]],
    seeds: list[dict[str, Any]],
    evidence_roles: set[str],
    include_split_factors: bool,
    positive_adjacency: dict[int, list[tuple[int, int]]],
    positive_min_count: int,
    evidence_topn: int,
    evidence_min_overlap: float,
    support_power: float,
    base_prior: float,
    temporal_gain: float,
    risk_gain: float,
) -> tuple[dict[int, dict[int, float]], dict[str, Any]]:
    seed_sets = [set(_row_core_ids(row)) for row in seeds]
    support: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for slot_idx, seed in enumerate(seeds):
        seed_weight = _row_weight(seed, temporal_gain=temporal_gain, risk_gain=risk_gain)
        for tid in seed_sets[slot_idx]:
            support[int(tid)][slot_idx] += float(base_prior) * float(seed_weight)

    evidence: list[dict[str, Any]] = [row for row in role_rows if str(row.get("seed_role")) in evidence_roles]
    if include_split_factors:
        evidence.extend(factor_rows)
    evidence = sorted(
        evidence,
        key=lambda row: (_row_weight(row, temporal_gain=temporal_gain, risk_gain=risk_gain), len(_row_core_ids(row))),
        reverse=True,
    )[: int(evidence_topn)]

    matched = 0
    added_tubes = 0
    blocked_tubes = 0
    role_counts: Counter[str] = Counter()
    matched_role_counts: Counter[str] = Counter()
    for row in evidence:
        core = set(_row_core_ids(row))
        if not core:
            continue
        role_key = str(row.get("seed_role") or row.get("proposal_type") or "split_factor")
        role_counts[role_key] += 1
        overlaps: list[tuple[float, int]] = []
        for slot_idx, seed in enumerate(seed_sets):
            ov = _overlap_min_norm(core, seed)
            if ov >= float(evidence_min_overlap):
                overlaps.append((ov, slot_idx))
        if not overlaps:
            continue
        ov, slot_idx = max(overlaps)
        seed_ids = seed_sets[int(slot_idx)]
        add_ids = {
            int(tid)
            for tid in core
            if int(tid) in seed_ids
            or _positive_to_seed(int(tid), seed_ids, positive_adjacency, min_count=int(positive_min_count)) > 0
        }
        if not add_ids:
            blocked_tubes += len(core)
            continue
        matched += 1
        matched_role_counts[role_key] += 1
        blocked_tubes += max(len(core) - len(add_ids), 0)
        added_tubes += len(add_ids)
        weight = _row_weight(row, temporal_gain=temporal_gain, risk_gain=risk_gain)
        add = float(weight * math.pow(max(ov, 1e-6), float(support_power)))
        for tid in add_ids:
            support[int(tid)][int(slot_idx)] += add

    stats: dict[str, Any] = {
        "evidence_row_count": int(len(evidence)),
        "evidence_row_matched_count": int(matched),
        "positive_gated_added_tube_count": int(added_tubes),
        "positive_gated_blocked_tube_count": int(blocked_tubes),
        "support_tube_count": int(len(support)),
    }
    for role, count in sorted(role_counts.items()):
        stats[f"evidence_role_{role}_count"] = int(count)
    for role, count in sorted(matched_role_counts.items()):
        stats[f"evidence_role_{role}_matched_count"] = int(count)
    return support, stats


def _scene_result(
    *,
    scene: str,
    control_kind: str,
    role_rows: list[dict[str, Any]],
    factor_rows: list[dict[str, Any]],
    gt_labels: dict[int, int],
    config: dict[str, Any],
    positive_adjacency: dict[int, list[tuple[int, int]]],
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
    support, support_stats = _build_positive_gated_support(
        role_rows=role_rows,
        factor_rows=factor_rows,
        seeds=seeds,
        evidence_roles=set(str(config["evidence_roles"]).split("+")) if config["evidence_roles"] else set(),
        include_split_factors=bool(config["include_split_factors"]),
        positive_adjacency=positive_adjacency,
        positive_min_count=int(config["positive_min_count"]),
        evidence_topn=int(config["evidence_topn"]),
        evidence_min_overlap=float(config["evidence_min_overlap"]),
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
        **{f"r16_{key}": value for key, value in {**seed_stats, **support_stats, **assign_stats}.items()},
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
        [80],
        [0.72, 0.86],
        [0.03, 0.08],
        [1, 2, 3],
        [0.2, 0.6],
        [1, 3],
    )
    for idx, (
        role_bundle,
        max_seed_slots,
        seed_max_overlap,
        evidence_min_overlap,
        positive_min_count,
        unknown_threshold,
        negative_min_count,
    ) in enumerate(grid):
        seed_roles, evidence_roles, include_split = role_bundle
        configs.append(
            {
                "solver": f"R16_positive_gate_search_{idx:04d}",
                "seed_roles": seed_roles,
                "evidence_roles": evidence_roles,
                "include_split_factors": bool(include_split),
                "max_seed_slots": int(max_seed_slots),
                "seed_min_new": 3,
                "seed_max_overlap": float(seed_max_overlap),
                "evidence_topn": 1500,
                "evidence_min_overlap": float(evidence_min_overlap),
                "positive_min_count": int(positive_min_count),
                "support_power": 0.5,
                "base_prior": 2.0,
                "temporal_gain": 0.30,
                "risk_gain": 0.35,
                "negative_min_count": int(negative_min_count),
                "unknown_threshold": float(unknown_threshold),
                "margin_ratio": 1.0,
                "margin_add": 0.0,
                "max_slot_tubes": 0,
                "min_slot_tubes": 2,
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
    positive_adjacency_by_scene = {}
    negative_adjacency_by_scene = {}
    for scene in scenes:
        print(f"[r16] building pair adjacency {scene}", flush=True)
        stream, records, masks_by_frame, measurements, diag = _scene_records_and_measurements(measurement_args, scene)
        pos_counts = _positive_pair_counts(measurements)
        neg_counts = _negative_pair_counts(measurements)
        positive_adjacency_by_scene[scene] = _adjacency(pos_counts)
        negative_adjacency_by_scene[scene] = _adjacency(neg_counts)
        print(
            f"[r16] {scene} measurements={diag.get('measurement_count')} positive_pairs={len(pos_counts)} negative_pairs={len(neg_counts)}",
            flush=True,
        )
        del stream, records, masks_by_frame, measurements, diag, pos_counts, neg_counts
        gc.collect()

    scene_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for cfg_idx, config in enumerate(configs):
        print(f"[r16] config {cfg_idx + 1}/{len(configs)} {config['solver']}", flush=True)
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
                    positive_adjacency=positive_adjacency_by_scene[scene],
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
    parser = argparse.ArgumentParser(description="Diagnose v31 r16 positive-gated safe coverage assignment.")
    parser.add_argument("--controls", default="real")
    parser.add_argument("--out-dir", default="outputs/audit/v31_slot_ownership")
    parser.add_argument("--summary-name", default="r16_positive_gate_assignment_real_summary.csv")
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
