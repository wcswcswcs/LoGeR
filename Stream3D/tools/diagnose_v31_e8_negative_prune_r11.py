from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path
from typing import Any

from tools.diagnose_v30_cannot_link_clique import _negative_pair_counts, _scene_records_and_measurements
from tools.diagnose_v31_tube_assignment_r8 import _negative_adjacency, _prune_negative_conflicts
from tools.run_v28_proposal_selection import _load_gt_labels
from tools.run_v29_constrained_ownership_solver import _set_core_ids
from tools.run_v30_object_slot_ownership import _read_split
from tools.run_v31_seed_anchor_lowtail import _aggregate_solver_rows, _eval_v29_selected, _row_core_ids, _write_csv


def _clone_slot(row: dict[str, Any], ids: list[int], solver: str) -> dict[str, Any]:
    item = {
        "scene": row["scene"],
        "proposal_id": f"{row.get('proposal_id')}_{solver}",
        "proposal_type": row.get("proposal_type"),
        "source_solver": row.get("solver"),
        "source_proposal_id": row.get("proposal_id"),
        "uses_gt_for_prediction": False,
    }
    _set_core_ids(item, tuple(sorted(int(v) for v in ids)))
    return item


def run(args: argparse.Namespace) -> None:
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

    with Path(args.selected_slot_rows).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    base = [
        row
        for row in rows
        if row.get("solver") == args.source_solver and row.get("control_kind") == args.control_kind
    ]
    print("base_selected_rows", len(base), flush=True)

    negative_adjacency_by_scene = {}
    measurement_args = argparse.Namespace(
        cache_root=args.cache_root,
        max_tubes_per_window=args.max_tubes_per_window,
        image_width=args.image_width,
        image_height=args.image_height,
    )
    for scene in scenes:
        print("[r11] building negative adjacency", scene, flush=True)
        stream, records, masks_by_frame, measurements, diag = _scene_records_and_measurements(measurement_args, scene)
        counts = _negative_pair_counts(measurements)
        negative_adjacency_by_scene[scene] = _negative_adjacency(counts)
        print("[r11]", scene, "measurements", diag.get("measurement_count"), "negative_pairs", len(counts), flush=True)
        del stream, records, masks_by_frame, measurements, diag, counts
        gc.collect()

    scene_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    configs = [
        (min_count, rounds)
        for min_count in args.negative_prune_min_counts
        for rounds in args.negative_prune_rounds
    ]
    for idx, (min_count, rounds) in enumerate(configs):
        solver = f"R11_E8_neg_prune_mc{min_count}_r{rounds}"
        print("[r11] config", idx + 1, "/", len(configs), solver, flush=True)
        for scene in scenes:
            selected: list[dict[str, Any]] = []
            pruned_total = 0
            remaining_total = 0
            source_rows = [row for row in base if row.get("scene") == scene]
            for rank, row in enumerate(source_rows):
                values = [(1.0, int(tid)) for tid in _row_core_ids(row)]
                pruned_values, removed, remaining = _prune_negative_conflicts(
                    values,
                    negative_adjacency=negative_adjacency_by_scene[scene],
                    min_count=int(min_count),
                    rounds=int(rounds),
                    min_slot_tubes=int(args.min_slot_tubes),
                )
                ids = [int(tid) for _confidence, tid in pruned_values]
                pruned_total += int(removed)
                remaining_total += int(remaining)
                if len(ids) < int(args.min_slot_tubes):
                    continue
                item = _clone_slot(row, ids, solver)
                selected.append(item)
                selected_rows.append(
                    {
                        "scene": scene,
                        "solver": solver,
                        "control_kind": args.control_kind,
                        "rank": rank,
                        "proposal_id": item["proposal_id"],
                        "proposal_type": item.get("proposal_type"),
                        "source_proposal_id": row.get("proposal_id"),
                        "core_tube_count": len(ids),
                        "core_tube_ids": ";".join(str(v) for v in sorted(ids)),
                        "uses_gt_for_prediction": False,
                    }
                )
            metrics = _eval_v29_selected(selected, gt_by_scene[scene])
            scene_rows.append(
                {
                    "scene": scene,
                    "solver": solver,
                    "control_kind": args.control_kind,
                    "slot_count": len(source_rows),
                    "active_slot_count": len(selected),
                    "owned_tube_ratio": metrics["owned_tube_ratio"],
                    "unknown_tube_ratio": metrics["unknown_tube_ratio"],
                    "coverage_factor_explained_ratio": metrics["owned_tube_ratio"],
                    "broad_observation_explained_ratio": 0.0,
                    "cannot_link_violation_count": remaining_total,
                    "boundary_violation_rate": 0.0,
                    "appearance_consistency": 1.0,
                    "motion_consistency": 1.0,
                    "solver_runtime_sec": 0.0,
                    "solver_iterations": int(rounds),
                    "ARI": metrics["ARI"],
                    "purity": metrics["purity"],
                    "completeness": metrics["completeness"],
                    "overmerge": metrics["overmerge"],
                    "oversplit": metrics["oversplit"],
                    "num_moves_attempted": int(rounds),
                    "num_moves_accepted": pruned_total,
                    "r11_negative_pruned_tube_count": pruned_total,
                    "r11_remaining_negative_pair_count": remaining_total,
                    "cfg_negative_prune_min_count": int(min_count),
                    "cfg_negative_prune_rounds": int(rounds),
                }
            )

    out_dir = Path(args.out_dir)
    _write_csv(out_dir / args.summary_name, scene_rows + _aggregate_solver_rows(scene_rows))
    _write_csv(out_dir / args.selected_name, selected_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose v31 r11 negative pruning on E8 selected slots.")
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v26_occupancy_d5_warmstart64_probe5_topup20_patch2")
    parser.add_argument("--max-tubes-per-window", type=int, default=640)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--selected-slot-rows", default="outputs/audit/v31_slot_ownership/selected_slot_rows.csv")
    parser.add_argument("--source-solver", default="E8_temporal_type_prior_high_coverage")
    parser.add_argument("--control-kind", default="real")
    parser.add_argument("--negative-prune-min-counts", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    parser.add_argument("--negative-prune-rounds", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128])
    parser.add_argument("--min-slot-tubes", type=int, default=3)
    parser.add_argument("--out-dir", default="outputs/audit/v31_slot_ownership")
    parser.add_argument("--summary-name", default="r11_e8_negative_prune_real_summary.csv")
    parser.add_argument("--selected-name", default="r11_e8_negative_prune_real_selected_rows.csv")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
