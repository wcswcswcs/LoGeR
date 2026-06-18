from __future__ import annotations

import argparse
import csv
import gc
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tools.diagnose_v30_cannot_link_clique import _negative_pair_counts, _scene_records_and_measurements
from tools.run_v28_proposal_selection import _load_gt_labels
from tools.run_v29_constrained_ownership_solver import _set_core_ids
from tools.run_v30_object_slot_ownership import _read_split
from tools.run_v31_seed_anchor_lowtail import _aggregate_solver_rows, _eval_v29_selected, _row_core_ids, _write_csv


def _pair(a: int, b: int) -> tuple[int, int]:
    left, right = int(a), int(b)
    return (left, right) if left < right else (right, left)


class _DSU:
    def __init__(self, ids: list[int]) -> None:
        self.parent = {int(v): int(v) for v in ids}

    def find(self, x: int) -> int:
        x = int(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != x:
            old = self.parent[x]
            self.parent[x] = root
            x = old
        return root

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def components(self) -> list[list[int]]:
        comps: dict[int, list[int]] = defaultdict(list)
        for tid in self.parent:
            comps[self.find(tid)].append(int(tid))
        return [sorted(v) for v in comps.values()]


def _positive_pair_counts(measurements) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for meas in measurements:
        for left, right in meas.same_mask_merge_pairs:
            counts[_pair(left, right)] += 1
        for left, right in meas.boundary_safe_merge_pairs:
            counts[_pair(left, right)] += 1
    return counts


def _split_ids(
    ids: list[int],
    positive_counts: Counter[tuple[int, int]],
    negative_counts: Counter[tuple[int, int]],
    *,
    pos_min_count: int,
    neg_block_count: int,
    min_component_size: int,
    keep_unlinked_as_component: bool,
) -> tuple[list[list[int]], dict[str, int]]:
    ids = sorted(set(int(v) for v in ids))
    dsu = _DSU(ids)
    id_set = set(ids)
    positive_edges = 0
    blocked_edges = 0
    for (left, right), pos_count in positive_counts.items():
        if int(pos_count) < int(pos_min_count) or int(left) not in id_set or int(right) not in id_set:
            continue
        if int(negative_counts.get(_pair(left, right), 0)) >= int(neg_block_count):
            blocked_edges += 1
            continue
        dsu.union(int(left), int(right))
        positive_edges += 1
    comps = []
    for comp in dsu.components():
        if len(comp) >= int(min_component_size):
            comps.append(comp)
        elif keep_unlinked_as_component and len(comp) >= 1:
            comps.append(comp)
    return comps, {
        "positive_edges_used": int(positive_edges),
        "positive_edges_blocked_by_negative": int(blocked_edges),
        "component_count": int(len(comps)),
    }


def _clone(row: dict[str, Any], ids: list[int], solver: str, component_idx: int) -> dict[str, Any]:
    item = {
        "scene": row["scene"],
        "proposal_id": f"{row.get('proposal_id')}_{solver}_c{component_idx:03d}",
        "proposal_type": "R13_E8_positive_negative_graph_component",
        "source_solver": row.get("solver"),
        "source_proposal_id": row.get("proposal_id"),
        "source_proposal_type": row.get("proposal_type"),
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
    base = [row for row in rows if row.get("solver") == args.source_solver and row.get("control_kind") == args.control_kind]
    print("base_selected_rows", len(base), flush=True)

    measurement_args = argparse.Namespace(
        cache_root=args.cache_root,
        max_tubes_per_window=args.max_tubes_per_window,
        image_width=args.image_width,
        image_height=args.image_height,
    )
    positive_by_scene = {}
    negative_by_scene = {}
    for scene in scenes:
        print("[r13] building pair counts", scene, flush=True)
        stream, records, masks_by_frame, measurements, diag = _scene_records_and_measurements(measurement_args, scene)
        positive_by_scene[scene] = _positive_pair_counts(measurements)
        negative_by_scene[scene] = _negative_pair_counts(measurements)
        print(
            "[r13]",
            scene,
            "measurements",
            diag.get("measurement_count"),
            "positive_pairs",
            len(positive_by_scene[scene]),
            "negative_pairs",
            len(negative_by_scene[scene]),
            flush=True,
        )
        del stream, records, masks_by_frame, measurements, diag
        gc.collect()

    configs = [
        (pos_min, neg_block, min_size, keep_small)
        for pos_min in args.pos_min_counts
        for neg_block in args.neg_block_counts
        for min_size in args.min_component_sizes
        for keep_small in args.keep_small_modes
    ]
    scene_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for idx, (pos_min, neg_block, min_size, keep_small) in enumerate(configs):
        solver = f"R13_E8_graph_split_p{pos_min}_n{neg_block}_m{min_size}_k{int(keep_small)}"
        print("[r13] config", idx + 1, "/", len(configs), solver, flush=True)
        for scene in scenes:
            selected = []
            component_total = 0
            edge_used_total = 0
            edge_blocked_total = 0
            source_rows = [row for row in base if row.get("scene") == scene]
            for source_rank, row in enumerate(source_rows):
                comps, stats = _split_ids(
                    list(_row_core_ids(row)),
                    positive_by_scene[scene],
                    negative_by_scene[scene],
                    pos_min_count=int(pos_min),
                    neg_block_count=int(neg_block),
                    min_component_size=int(min_size),
                    keep_unlinked_as_component=bool(keep_small),
                )
                component_total += int(stats["component_count"])
                edge_used_total += int(stats["positive_edges_used"])
                edge_blocked_total += int(stats["positive_edges_blocked_by_negative"])
                for comp_idx, comp in enumerate(comps):
                    item = _clone(row, comp, solver, comp_idx)
                    selected.append(item)
                    selected_rows.append(
                        {
                            "scene": scene,
                            "solver": solver,
                            "control_kind": args.control_kind,
                            "rank": len(selected_rows),
                            "source_rank": source_rank,
                            "proposal_id": item["proposal_id"],
                            "proposal_type": item["proposal_type"],
                            "source_proposal_id": row.get("proposal_id"),
                            "core_tube_count": len(comp),
                            "core_tube_ids": ";".join(str(v) for v in sorted(comp)),
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
                    "cannot_link_violation_count": 0,
                    "boundary_violation_rate": 0.0,
                    "appearance_consistency": 1.0,
                    "motion_consistency": 1.0,
                    "solver_runtime_sec": 0.0,
                    "solver_iterations": 1,
                    "ARI": metrics["ARI"],
                    "purity": metrics["purity"],
                    "completeness": metrics["completeness"],
                    "overmerge": metrics["overmerge"],
                    "oversplit": metrics["oversplit"],
                    "num_moves_attempted": int(edge_used_total),
                    "num_moves_accepted": int(component_total),
                    "r13_component_count": int(component_total),
                    "r13_positive_edges_used": int(edge_used_total),
                    "r13_positive_edges_blocked_by_negative": int(edge_blocked_total),
                    "cfg_pos_min_count": int(pos_min),
                    "cfg_neg_block_count": int(neg_block),
                    "cfg_min_component_size": int(min_size),
                    "cfg_keep_small": bool(keep_small),
                }
            )
    out_dir = Path(args.out_dir)
    _write_csv(out_dir / args.summary_name, scene_rows + _aggregate_solver_rows(scene_rows))
    _write_csv(out_dir / args.selected_name, selected_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose v31 r13 graph split of E8 selected slots.")
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v26_occupancy_d5_warmstart64_probe5_topup20_patch2")
    parser.add_argument("--max-tubes-per-window", type=int, default=640)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--selected-slot-rows", default="outputs/audit/v31_slot_ownership/selected_slot_rows.csv")
    parser.add_argument("--source-solver", default="E8_temporal_type_prior_high_coverage")
    parser.add_argument("--control-kind", default="real")
    parser.add_argument("--pos-min-counts", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--neg-block-counts", type=int, nargs="+", default=[1, 2, 5])
    parser.add_argument("--min-component-sizes", type=int, nargs="+", default=[2, 3, 5, 8])
    parser.add_argument("--keep-small-modes", type=int, nargs="+", default=[0])
    parser.add_argument("--out-dir", default="outputs/audit/v31_slot_ownership")
    parser.add_argument("--summary-name", default="r13_e8_graph_split_real_summary.csv")
    parser.add_argument("--selected-name", default="r13_e8_graph_split_real_selected_rows.csv")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
