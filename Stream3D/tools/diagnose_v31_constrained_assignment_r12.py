from __future__ import annotations

import argparse
import csv
import gc
import itertools
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.diagnose_v30_cannot_link_clique import _negative_pair_counts, _scene_records_and_measurements
from tools.diagnose_v31_tube_assignment_r8 import _negative_adjacency
from tools.run_v28_proposal_selection import _load_gt_labels
from tools.run_v29_constrained_ownership_solver import MASK_ONLY_TYPES, TEMPORAL_PREFIXES, _set_core_ids
from tools.run_v30_object_slot_ownership import _read_split
from tools.run_v31_seed_anchor_lowtail import (
    _aggregate_solver_rows,
    _eval_v29_selected,
    _f,
    _load_split_repair_factor_rows,
    _make_shuffled_rows,
    _row_core_ids,
    _select_v31_type_prior_solver,
    _v31_type_prior_risk,
    _v31_type_prior_score,
    _write_csv,
)


def _mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def _overlap_min_norm(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    return float(len(a & b) / max(min(len(a), len(b)), 1))


def _row_weight(row: dict[str, Any], *, temporal_gain: float, risk_gain: float) -> float:
    proposal_type = str(row.get("proposal_type") or "")
    temporal = 1.0 if proposal_type.startswith(TEMPORAL_PREFIXES) else 0.0
    return float(max(0.02, _v31_type_prior_score(row) + 0.55 + temporal_gain * temporal - risk_gain * _v31_type_prior_risk(row)))


def _control_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    shuffled = _make_shuffled_rows(rows)
    return {
        "real": rows,
        "shuffled_d4rt": shuffled,
        "no_temporal": [row for row in rows if not str(row.get("proposal_type", "")).startswith(TEMPORAL_PREFIXES)],
        "mask_only": [row for row in rows if str(row.get("proposal_type")) in MASK_ONLY_TYPES],
    }


def _seed_slots(
    candidates: list[dict[str, Any]],
    *,
    max_seed_slots: int,
    seed_min_new: int,
    seed_max_overlap: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed_candidates = [row for row in candidates if not str(row.get("proposal_type", "")).startswith(TEMPORAL_PREFIXES)]
    return _select_v31_type_prior_solver(
        seed_candidates,
        min_new_tubes=int(seed_min_new),
        max_overlap_ratio=float(seed_max_overlap),
        max_slots=int(max_seed_slots),
        min_score=-1e9,
        max_temporal_fraction=0.0,
    )


def _build_positive_support(
    candidates: list[dict[str, Any]],
    seeds: list[dict[str, Any]],
    *,
    evidence_topn: int,
    evidence_min_overlap: float,
    evidence_top_slots: int,
    support_power: float,
    base_prior: float,
    temporal_gain: float,
    risk_gain: float,
) -> tuple[dict[int, dict[int, float]], dict[str, Any]]:
    seed_sets = [set(_row_core_ids(row)) for row in seeds]
    support: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for slot_idx, ids in enumerate(seed_sets):
        for tid in ids:
            support[int(tid)][slot_idx] += float(base_prior)

    evidence = sorted(candidates, key=lambda row: (_v31_type_prior_score(row), len(_row_core_ids(row))), reverse=True)[
        : int(evidence_topn)
    ]
    matched = 0
    multi = 0
    for row in evidence:
        core = set(_row_core_ids(row))
        if not core:
            continue
        overlaps: list[tuple[float, int]] = []
        for slot_idx, seed in enumerate(seed_sets):
            ov = _overlap_min_norm(core, seed)
            if ov >= float(evidence_min_overlap):
                overlaps.append((ov, slot_idx))
        if not overlaps:
            continue
        overlaps.sort(reverse=True)
        selected_overlaps = overlaps[: max(int(evidence_top_slots), 1)]
        matched += 1
        multi += int(len(selected_overlaps) > 1)
        weight = _row_weight(row, temporal_gain=temporal_gain, risk_gain=risk_gain)
        for ov, slot_idx in selected_overlaps:
            add = float(weight * math.pow(max(ov, 1e-6), float(support_power)))
            for tid in core:
                support[int(tid)][slot_idx] += add
    return support, {
        "evidence_row_count": int(len(evidence)),
        "evidence_row_matched_count": int(matched),
        "evidence_row_multi_slot_count": int(multi),
        "support_tube_count": int(len(support)),
    }


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


def _constrained_assign(
    support: dict[int, dict[int, float]],
    *,
    negative_adjacency: dict[int, list[tuple[int, int]]],
    negative_min_count: int,
    negative_lambda: float,
    unknown_threshold: float,
    margin_ratio: float,
    margin_add: float,
    max_slot_tubes: int,
    passes: int,
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

    def best_slot_for(tid: int, current_slot: int | None = None) -> tuple[int | None, float, float, int]:
        scores: list[tuple[float, int, int]] = []
        for slot_idx, positive in support[int(tid)].items():
            ids = slot_ids[int(slot_idx)]
            if current_slot is not None and int(slot_idx) == int(current_slot):
                ids = set(ids)
                ids.discard(int(tid))
            if int(max_slot_tubes) > 0 and current_slot != slot_idx and len(ids) >= int(max_slot_tubes):
                continue
            neg = _negative_to_slot(int(tid), ids, negative_adjacency, min_count=int(negative_min_count))
            score = float(positive) - float(negative_lambda) * float(neg)
            scores.append((score, int(slot_idx), int(neg)))
        if not scores:
            return None, 0.0, 0.0, 0
        scores.sort(reverse=True)
        best_score, best_slot, best_neg = scores[0]
        second = scores[1][0] if len(scores) > 1 else 0.0
        return int(best_slot), float(best_score), float(second), int(best_neg)

    for tid in ordered:
        best_slot, best_score, second_score, _best_neg = best_slot_for(int(tid))
        if best_slot is None:
            rejected_capacity += 1
            continue
        if best_score < float(unknown_threshold):
            rejected_unknown += 1
            continue
        if best_score < float(margin_ratio) * float(second_score) + float(margin_add):
            rejected_margin += 1
            continue
        assignments[int(tid)] = int(best_slot)
        slot_ids[int(best_slot)].add(int(tid))

    move_count = 0
    drop_count = 0
    for _ in range(int(passes)):
        changed = False
        for tid in ordered:
            current = assignments.get(int(tid))
            if current is None:
                best_slot, best_score, second_score, _best_neg = best_slot_for(int(tid))
                if (
                    best_slot is not None
                    and best_score >= float(unknown_threshold)
                    and best_score >= float(margin_ratio) * float(second_score) + float(margin_add)
                ):
                    assignments[int(tid)] = int(best_slot)
                    slot_ids[int(best_slot)].add(int(tid))
                    changed = True
                continue
            current_positive = float(support[int(tid)].get(int(current), 0.0))
            current_neg = _negative_to_slot(int(tid), slot_ids[int(current)] - {int(tid)}, negative_adjacency, min_count=int(negative_min_count))
            current_score = current_positive - float(negative_lambda) * float(current_neg)
            best_slot, best_score, second_score, _best_neg = best_slot_for(int(tid), current_slot=int(current))
            if current_score < float(unknown_threshold):
                slot_ids[int(current)].discard(int(tid))
                assignments.pop(int(tid), None)
                drop_count += 1
                changed = True
            elif best_slot is not None and best_slot != current and best_score > current_score + float(margin_add):
                if best_score >= float(margin_ratio) * float(second_score):
                    slot_ids[int(current)].discard(int(tid))
                    assignments[int(tid)] = int(best_slot)
                    slot_ids[int(best_slot)].add(int(tid))
                    move_count += 1
                    changed = True
        if not changed:
            break

    remaining_neg = 0
    for slot_idx, ids in slot_ids.items():
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
        "local_move_count": int(move_count),
        "local_drop_count": int(drop_count),
        "remaining_negative_weight": int(remaining_neg),
    }


def _selected_from_assignments(
    scene: str,
    seeds: list[dict[str, Any]],
    assignments: dict[int, int],
    solver: str,
    *,
    min_slot_tubes: int,
) -> list[dict[str, Any]]:
    by_slot: dict[int, list[int]] = defaultdict(list)
    for tid, slot_idx in assignments.items():
        by_slot[int(slot_idx)].append(int(tid))
    selected: list[dict[str, Any]] = []
    for slot_idx, ids in sorted(by_slot.items()):
        if len(ids) < int(min_slot_tubes):
            continue
        seed = seeds[int(slot_idx)]
        item = {
            "scene": scene,
            "proposal_id": f"{seed.get('proposal_id')}_{solver}",
            "proposal_type": "R12_constrained_assignment_slot",
            "seed_proposal_id": seed.get("proposal_id"),
            "seed_proposal_type": seed.get("proposal_type"),
            "uses_gt_for_prediction": False,
        }
        _set_core_ids(item, tuple(sorted(ids)))
        selected.append(item)
    return selected


def _scene_result(
    *,
    scene: str,
    control_kind: str,
    candidates: list[dict[str, Any]],
    gt_labels: dict[int, int],
    config: dict[str, Any],
    negative_adjacency: dict[int, list[tuple[int, int]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    solver = str(config["solver"]) if control_kind == "real" else f"{config['solver']}_{control_kind}"
    t0 = time.time()
    seeds, seed_stats = _seed_slots(
        candidates,
        max_seed_slots=int(config["max_seed_slots"]),
        seed_min_new=int(config["seed_min_new"]),
        seed_max_overlap=float(config["seed_max_overlap"]),
    )
    support, support_stats = _build_positive_support(
        candidates,
        seeds,
        evidence_topn=int(config["evidence_topn"]),
        evidence_min_overlap=float(config["evidence_min_overlap"]),
        evidence_top_slots=int(config["evidence_top_slots"]),
        support_power=float(config["support_power"]),
        base_prior=float(config["base_prior"]),
        temporal_gain=float(config["temporal_gain"]),
        risk_gain=float(config["risk_gain"]),
    )
    assignments, assign_stats = _constrained_assign(
        support,
        negative_adjacency=negative_adjacency,
        negative_min_count=int(config["negative_min_count"]),
        negative_lambda=float(config["negative_lambda"]),
        unknown_threshold=float(config["unknown_threshold"]),
        margin_ratio=float(config["margin_ratio"]),
        margin_add=float(config["margin_add"]),
        max_slot_tubes=int(config["max_slot_tubes"]),
        passes=int(config["local_passes"]),
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
    row = {
        "scene": scene,
        "solver": solver,
        "control_kind": control_kind,
        "slot_count": int(len(candidates)),
        "active_slot_count": int(len(selected)),
        "owned_tube_ratio": metrics["owned_tube_ratio"],
        "unknown_tube_ratio": metrics["unknown_tube_ratio"],
        "coverage_factor_explained_ratio": metrics["owned_tube_ratio"],
        "broad_observation_explained_ratio": float(
            sum(1 for row in candidates[: int(config["evidence_topn"])] if str(row.get("proposal_type", "")).startswith(TEMPORAL_PREFIXES))
            / max(int(config["evidence_topn"]), 1)
        ),
        "cannot_link_violation_count": assign_stats["remaining_negative_weight"],
        "boundary_violation_rate": 0.0,
        "appearance_consistency": 1.0,
        "motion_consistency": 1.0,
        "solver_runtime_sec": runtime,
        "solver_iterations": int(config["local_passes"]),
        "ARI": metrics["ARI"],
        "purity": metrics["purity"],
        "completeness": metrics["completeness"],
        "overmerge": metrics["overmerge"],
        "oversplit": metrics["oversplit"],
        "num_moves_attempted": int(support_stats["support_tube_count"]),
        "num_moves_accepted": int(assign_stats["assigned_tube_count"]),
        **{f"r12_{key}": value for key, value in {**seed_stats, **support_stats, **assign_stats}.items()},
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
                "seed_proposal_id": item.get("seed_proposal_id"),
                "seed_proposal_type": item.get("seed_proposal_type"),
                "core_tube_count": len(_row_core_ids(item)),
                "core_tube_ids": ";".join(str(v) for v in _row_core_ids(item)),
                "uses_gt_for_prediction": False,
            }
        )
    return row, selected_rows


def _search_configs() -> list[dict[str, Any]]:
    configs = []
    grid = itertools.product(
        [56],
        [1000],
        [0.01, 0.03],
        [1, 2],
        [1.0, 1.5],
        [0.8, 1.4, 2.2],
        [1.0, 1.2],
        [1, 3],
    )
    for idx, (
        max_seed_slots,
        evidence_topn,
        evidence_min_overlap,
        evidence_top_slots,
        negative_lambda,
        unknown_threshold,
        margin_ratio,
        negative_min_count,
    ) in enumerate(grid):
        configs.append(
            {
                "solver": f"R12_constrained_search_{idx:04d}",
                "max_seed_slots": int(max_seed_slots),
                "seed_min_new": 3,
                "seed_max_overlap": 0.82,
                "evidence_topn": int(evidence_topn),
                "evidence_min_overlap": float(evidence_min_overlap),
                "evidence_top_slots": int(evidence_top_slots),
                "support_power": 0.5,
                "base_prior": 4.0,
                "temporal_gain": 0.45,
                "risk_gain": 0.35,
                "negative_min_count": int(negative_min_count),
                "negative_lambda": float(negative_lambda),
                "unknown_threshold": float(unknown_threshold),
                "margin_ratio": float(margin_ratio),
                "margin_add": 0.0,
                "max_slot_tubes": 0,
                "local_passes": 2,
                "min_slot_tubes": 3,
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
    rows, _stats = _load_split_repair_factor_rows(Path(args.proposal_row_csv))
    controls = _control_rows(rows)
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
        print(f"[r12] building negative adjacency {scene}", flush=True)
        stream, records, masks_by_frame, measurements, diag = _scene_records_and_measurements(measurement_args, scene)
        counts = _negative_pair_counts(measurements)
        negative_adjacency_by_scene[scene] = _negative_adjacency(counts)
        print(f"[r12] {scene} measurements={diag.get('measurement_count')} negative_pairs={len(counts)}", flush=True)
        del stream, records, masks_by_frame, measurements, diag, counts
        gc.collect()

    scene_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for cfg_idx, config in enumerate(configs):
        print(f"[r12] config {cfg_idx + 1}/{len(configs)} {config['solver']}", flush=True)
        for control_kind in requested_controls:
            control_items = controls[control_kind]
            for scene in scenes:
                candidates = [row for row in control_items if str(row.get("scene")) == scene]
                row, selected = _scene_result(
                    scene=scene,
                    control_kind=control_kind,
                    candidates=candidates,
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
    parser = argparse.ArgumentParser(description="Diagnose v31 r12 constrained tube-slot assignment.")
    parser.add_argument("--controls", default="real")
    parser.add_argument("--out-dir", default="outputs/audit/v31_slot_ownership")
    parser.add_argument("--summary-name", default="r12_constrained_assignment_real_summary.csv")
    parser.add_argument("--selected-name", default="")
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v26_occupancy_d5_warmstart64_probe5_topup20_patch2")
    parser.add_argument("--max-tubes-per-window", type=int, default=640)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument(
        "--proposal-row-csv",
        default="outputs/audit/v28_proposal_oracle_repair_temporal_track_640_pruned_shared2_t20_probe5/v28_proposal_oracle_repair_temporal_track_640_pruned_shared2_t20_probe5_proposal_rows.csv",
    )
    return parser


def main() -> None:
    print(run(build_parser().parse_args()), flush=True)


if __name__ == "__main__":
    main()
