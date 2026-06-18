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

from tools.run_v28_proposal_selection import _load_gt_labels
from tools.run_v29_constrained_ownership_solver import MASK_ONLY_TYPES, TEMPORAL_PREFIXES, _set_core_ids
from tools.run_v30_object_slot_ownership import _read_split
from tools.diagnose_v30_cannot_link_clique import _negative_pair_counts, _scene_records_and_measurements
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
    max_slots: int,
    min_new: int,
    max_overlap: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed_candidates = [row for row in candidates if not str(row.get("proposal_type", "")).startswith(TEMPORAL_PREFIXES)]
    return _select_v31_type_prior_solver(
        seed_candidates,
        min_new_tubes=int(min_new),
        max_overlap_ratio=float(max_overlap),
        max_slots=int(max_slots),
        min_score=-1e9,
        max_temporal_fraction=0.0,
    )


def _row_weight(row: dict[str, Any], *, temporal_gain: float, risk_gain: float) -> float:
    score = _v31_type_prior_score(row)
    risk = _v31_type_prior_risk(row)
    proposal_type = str(row.get("proposal_type") or "")
    temporal = 1.0 if proposal_type.startswith(TEMPORAL_PREFIXES) else 0.0
    return float(max(0.02, score + 0.55 + temporal_gain * temporal - risk_gain * risk))


def _negative_adjacency(
    negative_pair_counts: Counter[tuple[int, int]] | None,
) -> dict[int, list[tuple[int, int]]]:
    adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
    if negative_pair_counts is None:
        return adjacency
    for (left, right), count in negative_pair_counts.items():
        left_i = int(left)
        right_i = int(right)
        count_i = int(count)
        adjacency[left_i].append((right_i, count_i))
        adjacency[right_i].append((left_i, count_i))
    return adjacency


def _prune_negative_conflicts(
    values: list[tuple[float, int]],
    *,
    negative_adjacency: dict[int, list[tuple[int, int]]],
    min_count: int,
    rounds: int,
    min_slot_tubes: int,
) -> tuple[list[tuple[float, int]], int, int]:
    kept = {int(tid): float(confidence) for confidence, tid in values}
    if len(kept) < int(min_slot_tubes):
        return values, 0, 0
    bad_score: Counter[int] = Counter()
    bad_pairs = 0
    for tid in list(kept):
        for other, count in negative_adjacency.get(int(tid), []):
            if int(other) not in kept or int(tid) >= int(other) or int(count) < int(min_count):
                continue
            bad_pairs += 1
            bad_score[int(tid)] += int(count)
            bad_score[int(other)] += int(count)
    removed = 0
    while bad_pairs > 0 and removed < int(rounds) and len(kept) > int(min_slot_tubes):
        victim = max(kept, key=lambda tid: (int(bad_score[int(tid)]), -float(kept[int(tid)])))
        for other, count in negative_adjacency.get(int(victim), []):
            if int(other) not in kept or int(count) < int(min_count):
                continue
            bad_pairs -= 1
            bad_score[int(other)] -= int(count)
        kept.pop(int(victim), None)
        bad_score.pop(int(victim), None)
        removed += 1
    return sorted((confidence, tid) for tid, confidence in kept.items()), int(removed), int(max(bad_pairs, 0))


def _assign_tubes_to_seed_slots(
    scene: str,
    candidates: list[dict[str, Any]],
    *,
    max_seed_slots: int,
    seed_min_new: int,
    seed_max_overlap: float,
    evidence_topn: int,
    evidence_min_overlap: float,
    evidence_top_slots: int,
    support_power: float,
    support_threshold: float,
    margin_ratio: float,
    margin_add: float,
    base_prior: float,
    temporal_gain: float,
    risk_gain: float,
    max_slot_tubes: int,
    min_slot_tubes: int,
    negative_prune_min_count: int = 0,
    negative_prune_rounds: int = 0,
    negative_pair_counts: Counter[tuple[int, int]] | None = None,
    negative_adjacency: dict[int, list[tuple[int, int]]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seeds, seed_stats = _seed_slots(
        candidates,
        max_slots=max_seed_slots,
        min_new=seed_min_new,
        max_overlap=seed_max_overlap,
    )
    seed_sets = [set(_row_core_ids(row)) for row in seeds]
    support: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    best_source_count = 0
    ambiguous_count = 0
    for slot_idx, ids in enumerate(seed_sets):
        for tid in ids:
            support[int(tid)][slot_idx] += float(base_prior)

    evidence = sorted(candidates, key=lambda row: (_v31_type_prior_score(row), len(_row_core_ids(row))), reverse=True)[
        : int(evidence_topn)
    ]
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
        if len(selected_overlaps) > 1:
            ambiguous_count += 1
        best_source_count += 1
        weight = _row_weight(row, temporal_gain=temporal_gain, risk_gain=risk_gain)
        for ov, slot_idx in selected_overlaps:
            add = float(weight * math.pow(max(ov, 1e-6), float(support_power)))
            for tid in core:
                support[int(tid)][slot_idx] += add

    assigned_by_slot: dict[int, list[tuple[float, int]]] = defaultdict(list)
    rejected_low_support = 0
    rejected_ambiguous = 0
    for tid, values in support.items():
        ranked = sorted(values.items(), key=lambda pair: pair[1], reverse=True)
        if not ranked:
            continue
        best_slot, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        if best_score < float(support_threshold):
            rejected_low_support += 1
            continue
        if best_score < float(margin_ratio) * second_score + float(margin_add):
            rejected_ambiguous += 1
            continue
        confidence = float(best_score - second_score)
        assigned_by_slot[int(best_slot)].append((confidence, int(tid)))

    selected: list[dict[str, Any]] = []
    capped_tube_count = 0
    negative_pruned_tube_count = 0
    remaining_negative_pair_count = 0
    if negative_adjacency is None:
        negative_adjacency = _negative_adjacency(negative_pair_counts)
    for slot_idx, values in sorted(assigned_by_slot.items()):
        values = sorted(values, reverse=True)
        if negative_adjacency and int(negative_prune_min_count) > 0 and int(negative_prune_rounds) > 0:
            values, removed, remaining = _prune_negative_conflicts(
                values,
                negative_adjacency=negative_adjacency,
                min_count=int(negative_prune_min_count),
                rounds=int(negative_prune_rounds),
                min_slot_tubes=int(min_slot_tubes),
            )
            negative_pruned_tube_count += int(removed)
            remaining_negative_pair_count += int(remaining)
        if int(max_slot_tubes) > 0 and len(values) > int(max_slot_tubes):
            capped_tube_count += len(values) - int(max_slot_tubes)
            values = values[: int(max_slot_tubes)]
        ids = tuple(sorted(tid for _, tid in values))
        if len(ids) < int(min_slot_tubes):
            continue
        seed = seeds[slot_idx]
        item = {
            "scene": scene,
            "proposal_id": f"{seed.get('proposal_id')}_r8tube",
            "proposal_type": "R8_v31_tube_assignment_slot",
            "seed_proposal_id": seed.get("proposal_id"),
            "seed_proposal_type": seed.get("proposal_type"),
            "core_tube_count_before_assignment": len(_row_core_ids(seed)),
            "uses_gt_for_prediction": False,
        }
        _set_core_ids(item, ids)
        selected.append(item)

    stats = {
        **seed_stats,
        "seed_slot_count": int(len(seeds)),
        "evidence_row_count": int(len(evidence)),
        "evidence_row_matched_count": int(best_source_count),
        "evidence_row_multi_slot_count": int(ambiguous_count),
        "support_tube_count": int(len(support)),
        "rejected_low_support_tube_count": int(rejected_low_support),
        "rejected_ambiguous_tube_count": int(rejected_ambiguous),
        "assigned_tube_count": int(sum(len(values) for values in assigned_by_slot.values())),
        "capped_tube_count": int(capped_tube_count),
        "negative_pruned_tube_count": int(negative_pruned_tube_count),
        "remaining_negative_pair_count": int(remaining_negative_pair_count),
        "active_slot_count": int(len(selected)),
    }
    return selected, stats


def _scene_result(
    *,
    scene: str,
    control_kind: str,
    candidates: list[dict[str, Any]],
    gt_labels: dict[int, int],
    config: dict[str, Any],
    negative_pair_counts: Counter[tuple[int, int]] | None = None,
    negative_adjacency: dict[int, list[tuple[int, int]]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    t0 = time.time()
    assignment_config = {key: value for key, value in config.items() if key != "solver"}
    assignment_config["negative_pair_counts"] = negative_pair_counts
    assignment_config["negative_adjacency"] = negative_adjacency
    selected, stats = _assign_tubes_to_seed_slots(scene, candidates, **assignment_config)
    runtime = float(time.time() - t0)
    metrics = _eval_v29_selected(selected, gt_labels)
    solver_name = str(config["solver"]) if control_kind == "real" else f"{config['solver']}_{control_kind}"
    selected_ids = set().union(*(set(_row_core_ids(row)) for row in selected)) if selected else set()
    candidate_ids = set().union(*(set(_row_core_ids(row)) for row in candidates)) if candidates else set()
    temporal_evidence = [
        row
        for row in sorted(candidates, key=lambda row: (_v31_type_prior_score(row), len(_row_core_ids(row))), reverse=True)[
            : int(config["evidence_topn"])
        ]
        if str(row.get("proposal_type", "")).startswith(TEMPORAL_PREFIXES)
    ]
    row = {
        "scene": scene,
        "solver": solver_name,
        "control_kind": control_kind,
        "slot_count": int(len(candidates)),
        "active_slot_count": int(len(selected)),
        "owned_tube_ratio": metrics["owned_tube_ratio"],
        "unknown_tube_ratio": metrics["unknown_tube_ratio"],
        "coverage_factor_explained_ratio": float(len(selected_ids & candidate_ids) / max(len(candidate_ids), 1)),
        "broad_observation_explained_ratio": float(len(temporal_evidence) / max(int(config["evidence_topn"]), 1)),
        "cannot_link_violation_count": 0,
        "boundary_violation_rate": _mean([_f(item, "boundary_risk") for item in selected]) or 0.0,
        "appearance_consistency": 1.0,
        "motion_consistency": 1.0,
        "energy_anchor": float(config["base_prior"]),
        "energy_coverage": 1.0 - metrics["owned_tube_ratio"],
        "energy_cannot": 0.0,
        "energy_boundary": 0.0,
        "energy_unknown": metrics["unknown_tube_ratio"],
        "energy_size": _mean([len(_row_core_ids(item)) for item in selected]) or 0.0,
        "solver_iterations": int(stats["evidence_row_count"]),
        "solver_runtime_sec": runtime,
        "ARI": metrics["ARI"],
        "purity": metrics["purity"],
        "completeness": metrics["completeness"],
        "overmerge": metrics["overmerge"],
        "oversplit": metrics["oversplit"],
        "num_moves_attempted": int(stats["evidence_row_count"]),
        "num_moves_accepted": int(stats["evidence_row_matched_count"]),
        **{f"r8_{key}": value for key, value in stats.items()},
        **{f"cfg_{key}": value for key, value in config.items() if key != "solver"},
    }
    selected_rows: list[dict[str, Any]] = []
    for rank, item in enumerate(selected):
        selected_rows.append(
            {
                "scene": scene,
                "solver": solver_name,
                "control_kind": control_kind,
                "rank": int(rank),
                "proposal_id": item.get("proposal_id"),
                "proposal_type": item.get("proposal_type"),
                "seed_proposal_id": item.get("seed_proposal_id"),
                "seed_proposal_type": item.get("seed_proposal_type"),
                "core_tube_count": len(_row_core_ids(item)),
                "core_tube_ids": ";".join(str(tid) for tid in _row_core_ids(item)),
                "uses_gt_for_prediction": False,
            }
        )
    return row, selected_rows


def _make_search_configs(profile: str) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    if profile == "r9":
        grid = itertools.product(
            [56],
            [1000, 1600],
            [0.01, 0.03, 0.06],
            [1.5, 3.0],
            [1.00, 1.15],
            [1, 2],
            [0.5, 1.0],
            [0],
        )
        for idx, (
            max_seed_slots,
            evidence_topn,
            evidence_min_overlap,
            support_threshold,
            margin_ratio,
            evidence_top_slots,
            support_power,
            max_slot_tubes,
        ) in enumerate(grid):
            configs.append(
                {
                    "solver": f"R9_tube_wtm_search_{idx:04d}",
                    "max_seed_slots": int(max_seed_slots),
                    "seed_min_new": 3,
                    "seed_max_overlap": 0.82,
                    "evidence_topn": int(evidence_topn),
                    "evidence_min_overlap": float(evidence_min_overlap),
                    "evidence_top_slots": int(evidence_top_slots),
                    "support_power": float(support_power),
                    "support_threshold": float(support_threshold),
                    "margin_ratio": float(margin_ratio),
                    "margin_add": 0.0,
                    "base_prior": 4.0,
                    "temporal_gain": 0.45,
                    "risk_gain": 0.35,
                    "max_slot_tubes": int(max_slot_tubes),
                    "min_slot_tubes": 3,
                }
            )
        return configs

    if profile == "r10":
        grid = itertools.product(
            [56],
            [1000],
            [0.01, 0.03],
            [1.5, 3.0],
            [1.00, 1.15],
            [1, 2],
            [0.5],
            [0],
            [1, 2],
        )
        for idx, (
            max_seed_slots,
            evidence_topn,
            evidence_min_overlap,
            support_threshold,
            margin_ratio,
            evidence_top_slots,
            support_power,
            max_slot_tubes,
            negative_prune_min_count,
        ) in enumerate(grid):
            configs.append(
                {
                    "solver": f"R10_neg_pruned_tube_search_{idx:04d}",
                    "max_seed_slots": int(max_seed_slots),
                    "seed_min_new": 3,
                    "seed_max_overlap": 0.82,
                    "evidence_topn": int(evidence_topn),
                    "evidence_min_overlap": float(evidence_min_overlap),
                    "evidence_top_slots": int(evidence_top_slots),
                    "support_power": float(support_power),
                    "support_threshold": float(support_threshold),
                    "margin_ratio": float(margin_ratio),
                    "margin_add": 0.0,
                    "base_prior": 4.0,
                    "temporal_gain": 0.45,
                    "risk_gain": 0.35,
                    "max_slot_tubes": int(max_slot_tubes),
                    "min_slot_tubes": 3,
                    "negative_prune_min_count": int(negative_prune_min_count),
                    "negative_prune_rounds": 96,
                }
            )
        return configs

    grid = itertools.product(
        [36, 56],
        [600, 1000],
        [0.03, 0.08, 0.15],
        [3.0, 6.0],
        [1.10, 1.35],
        [2],
        [1.0],
        [0, 160],
    )
    for idx, (
        max_seed_slots,
        evidence_topn,
        evidence_min_overlap,
        support_threshold,
        margin_ratio,
        evidence_top_slots,
        support_power,
        max_slot_tubes,
    ) in enumerate(grid):
        configs.append(
            {
                "solver": f"R8_tube_vote_search_{idx:04d}",
                "max_seed_slots": int(max_seed_slots),
                "seed_min_new": 3,
                "seed_max_overlap": 0.82,
                "evidence_topn": int(evidence_topn),
                "evidence_min_overlap": float(evidence_min_overlap),
                "evidence_top_slots": int(evidence_top_slots),
                "support_power": float(support_power),
                "support_threshold": float(support_threshold),
                "margin_ratio": float(margin_ratio),
                "margin_add": 0.05,
                "base_prior": 6.0,
                "temporal_gain": 0.30,
                "risk_gain": 0.50,
                "max_slot_tubes": int(max_slot_tubes),
                "min_slot_tubes": 3,
            }
        )
    return configs


def _score_finalist(row: dict[str, Any]) -> float:
    ari = float(row.get("ARI") or 0.0)
    purity = float(row.get("purity") or 0.0)
    completeness = float(row.get("completeness") or 0.0)
    scene0081 = float(row.get("scene0081_ARI") or 0.0)
    return float(
        1.20 * min(ari / 0.35, 1.2)
        + min(purity / 0.85, 1.1)
        + min(completeness / 0.50, 1.1)
        + min(scene0081 / 0.20, 1.1)
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _configs_from_search(path: Path, topk: int) -> list[dict[str, Any]]:
    all_rows = _read_csv(path)
    rows = [row for row in all_rows if row.get("scene") == "ALL" and row.get("control_kind") == "real"]
    cfg_by_solver = {
        str(row.get("solver")): row
        for row in all_rows
        if row.get("scene") != "ALL" and row.get("control_kind") == "real" and row.get("cfg_max_seed_slots") not in (None, "")
    }
    ranked = sorted(rows, key=_score_finalist, reverse=True)[: int(topk)]
    configs: list[dict[str, Any]] = []
    for idx, row in enumerate(ranked):
        cfg_row = cfg_by_solver[str(row.get("solver"))]
        configs.append(
            {
                "solver": f"R8_tube_vote_finalist_{idx:02d}_from_{row.get('solver')}",
                "max_seed_slots": int(float(cfg_row["cfg_max_seed_slots"])),
                "seed_min_new": int(float(cfg_row["cfg_seed_min_new"])),
                "seed_max_overlap": float(cfg_row["cfg_seed_max_overlap"]),
                "evidence_topn": int(float(cfg_row["cfg_evidence_topn"])),
                "evidence_min_overlap": float(cfg_row["cfg_evidence_min_overlap"]),
                "evidence_top_slots": int(float(cfg_row["cfg_evidence_top_slots"])),
                "support_power": float(cfg_row["cfg_support_power"]),
                "support_threshold": float(cfg_row["cfg_support_threshold"]),
                "margin_ratio": float(cfg_row["cfg_margin_ratio"]),
                "margin_add": float(cfg_row["cfg_margin_add"]),
                "base_prior": float(cfg_row["cfg_base_prior"]),
                "temporal_gain": float(cfg_row["cfg_temporal_gain"]),
                "risk_gain": float(cfg_row["cfg_risk_gain"]),
                "max_slot_tubes": int(float(cfg_row["cfg_max_slot_tubes"])),
                "min_slot_tubes": int(float(cfg_row["cfg_min_slot_tubes"])),
                "negative_prune_min_count": int(float(cfg_row.get("cfg_negative_prune_min_count") or 0)),
                "negative_prune_rounds": int(float(cfg_row.get("cfg_negative_prune_rounds") or 0)),
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
    configs = (
        _make_search_configs(str(args.search_profile))
        if args.preset == "search"
        else _configs_from_search(Path(args.finalist_source_csv), int(args.finalist_topk))
    )
    negative_counts_by_scene: dict[str, Counter[tuple[int, int]]] = {}
    negative_adjacency_by_scene: dict[str, dict[int, list[tuple[int, int]]]] = {}
    if any(int(config.get("negative_prune_min_count") or 0) > 0 for config in configs):
        for scene in scenes:
            print(f"[r10] building negative pair counts {scene}", flush=True)
            _stream, _records, _masks_by_frame, measurements, meas_diag = _scene_records_and_measurements(args, scene)
            counts = _negative_pair_counts(measurements)
            negative_counts_by_scene[scene] = counts
            negative_adjacency_by_scene[scene] = _negative_adjacency(counts)
            print(
                f"[r10] {scene} measurements={meas_diag.get('measurement_count')} negative_pairs={len(counts)}",
                flush=True,
            )
            del _stream, _records, _masks_by_frame, measurements, meas_diag
            gc.collect()
    scene_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for cfg_idx, config in enumerate(configs):
        print(f"[r8] config {cfg_idx + 1}/{len(configs)} {config['solver']}", flush=True)
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
                    negative_pair_counts=negative_counts_by_scene.get(scene),
                    negative_adjacency=negative_adjacency_by_scene.get(scene),
                )
                scene_rows.append(row)
                selected_rows.extend(selected)
    aggregate = _aggregate_solver_rows(scene_rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / args.summary_name, scene_rows + aggregate)
    if args.selected_name:
        _write_csv(out_dir / args.selected_name, selected_rows)
    best_real = max(
        (row for row in aggregate if row.get("control_kind") == "real"),
        key=lambda row: _score_finalist(row),
        default={},
    )
    return {"config_count": len(configs), "best_real": best_real}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose v31 r8 tube-level assignment from proposal evidence.")
    parser.add_argument("--preset", choices=["search", "finalist"], default="search")
    parser.add_argument("--search-profile", choices=["r8", "r9", "r10"], default="r8")
    parser.add_argument("--controls", default="real")
    parser.add_argument("--out-dir", default="outputs/audit/v31_slot_ownership")
    parser.add_argument("--summary-name", default="r8_tube_assignment_search_summary.csv")
    parser.add_argument("--selected-name", default="")
    parser.add_argument("--finalist-source-csv", default="outputs/audit/v31_slot_ownership/r8_tube_assignment_search_summary.csv")
    parser.add_argument("--finalist-topk", type=int, default=24)
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
    payload = run(build_parser().parse_args())
    print(payload)


if __name__ == "__main__":
    main()
