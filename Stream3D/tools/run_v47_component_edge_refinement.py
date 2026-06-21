from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from stream4d_native.v47_common import (
    ROOT,
    adjusted_rand_score,
    cluster_completeness,
    cluster_purity,
    parse_bool,
    parse_float,
    parse_int,
    read_csv,
    write_csv,
    write_json,
)


class _StringUnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def _node_component_rows(mask_vote_rows: list[dict[str, Any]]) -> tuple[dict[int, str], dict[str, dict[str, Any]]]:
    node_to_component: dict[int, str] = {}
    component_meta: dict[str, dict[str, Any]] = {}
    for row in mask_vote_rows:
        node_id = parse_int(row.get("node_id"))
        component_id = str(row.get("predicted_component_object_id") or row.get("predicted_supertrack_object_id"))
        node_to_component[node_id] = component_id
        meta = component_meta.setdefault(
            component_id,
            {"component_id": component_id, "scene": str(row.get("scene")), "frames": set(), "mask_count": 0},
        )
        meta["frames"].add(parse_int(row.get("frame_id")))
        meta["mask_count"] += 1
    return node_to_component, component_meta


def _candidate_pair_stats(
    edge_rows: list[dict[str, Any]],
    node_to_component: dict[int, str],
    component_meta: dict[str, dict[str, Any]],
    edge_types: set[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    stats: dict[tuple[str, str], dict[str, Any]] = {}
    for row in edge_rows:
        if str(row.get("edge_type")) not in edge_types:
            continue
        src_component = node_to_component.get(parse_int(row.get("src_node_id")))
        dst_component = node_to_component.get(parse_int(row.get("dst_node_id")))
        if not src_component or not dst_component or src_component == dst_component:
            continue
        left = component_meta.get(src_component)
        right = component_meta.get(dst_component)
        if not left or not right or left.get("scene") != right.get("scene"):
            continue
        key = tuple(sorted([src_component, dst_component]))
        item = stats.setdefault(
            key,
            {
                "component_left": key[0],
                "component_right": key[1],
                "scene": left.get("scene"),
                "edge_count": 0,
                "same_frame_conflict": bool(set(left["frames"]) & set(right["frames"])),
                "max_A5_d4rt_semantic_confirmation": 0.0,
                "max_A4_d4rt_visible_veto": 0.0,
                "max_A8_no_temporal_control": 0.0,
                "max_A7_shuffled_D4RT": 0.0,
                "min_visible_outside": 1.0,
                "max_forward_visible_carrier_count": 0,
                "max_backward_visible_carrier_count": 0,
                "diagnostic_same_gt_edge_count": 0,
            },
        )
        item["edge_count"] += 1
        for score_key in [
            "A5_d4rt_semantic_confirmation",
            "A4_d4rt_visible_veto",
            "A8_no_temporal_control",
            "A7_shuffled_D4RT",
        ]:
            stat_key = f"max_{score_key}"
            item[stat_key] = max(parse_float(item.get(stat_key)), parse_float(row.get(score_key)))
        item["min_visible_outside"] = min(parse_float(item.get("min_visible_outside"), 1.0), parse_float(row.get("visible_outside"), 1.0))
        item["max_forward_visible_carrier_count"] = max(
            parse_int(item.get("max_forward_visible_carrier_count")), parse_int(row.get("forward_visible_carrier_count"))
        )
        item["max_backward_visible_carrier_count"] = max(
            parse_int(item.get("max_backward_visible_carrier_count")), parse_int(row.get("backward_visible_carrier_count"))
        )
        if parse_bool(row.get("diagnostic_same_gt")):
            item["diagnostic_same_gt_edge_count"] += 1
    return stats


def _evaluate(
    *,
    component_ids: list[str],
    mask_vote_rows: list[dict[str, Any]],
    candidate_stats: dict[tuple[str, str], dict[str, Any]],
    score_key: str,
    score_threshold: float,
    min_pair_edge_count: int,
    max_visible_outside: float,
    min_forward_visible_carriers: int,
    min_backward_visible_carriers: int,
    forbid_same_frame_conflict: bool,
) -> dict[str, Any]:
    uf = _StringUnionFind(component_ids)
    selected_pairs: list[dict[str, Any]] = []
    score_col = f"max_{score_key}"
    for pair, row in candidate_stats.items():
        if parse_int(row.get("edge_count")) < int(min_pair_edge_count):
            continue
        if parse_float(row.get(score_col)) < float(score_threshold):
            continue
        if parse_float(row.get("min_visible_outside"), 1.0) > float(max_visible_outside):
            continue
        if parse_int(row.get("max_forward_visible_carrier_count")) < int(min_forward_visible_carriers):
            continue
        if parse_int(row.get("max_backward_visible_carrier_count")) < int(min_backward_visible_carriers):
            continue
        if forbid_same_frame_conflict and bool(row.get("same_frame_conflict")):
            continue
        uf.union(pair[0], pair[1])
        selected_pairs.append(row)

    true_labels: list[str] = []
    pred_labels: list[str] = []
    for row in mask_vote_rows:
        gt = str(row.get("diagnostic_gt_instance", ""))
        if not gt:
            continue
        component_id = str(row.get("predicted_component_object_id") or row.get("predicted_supertrack_object_id"))
        if component_id not in uf.parent:
            continue
        true_labels.append(gt)
        pred_labels.append(uf.find(component_id))
    return {
        "score_key": score_key,
        "score_threshold": float(score_threshold),
        "min_pair_edge_count": int(min_pair_edge_count),
        "max_visible_outside": float(max_visible_outside),
        "min_forward_visible_carriers": int(min_forward_visible_carriers),
        "min_backward_visible_carriers": int(min_backward_visible_carriers),
        "forbid_same_frame_conflict": bool(forbid_same_frame_conflict),
        "selected_pair_count": len(selected_pairs),
        "cluster_count": len(set(pred_labels)),
        "ARI": adjusted_rand_score(true_labels, pred_labels),
        "purity": cluster_purity(true_labels, pred_labels),
        "completeness": cluster_completeness(true_labels, pred_labels),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _parse_float_list(spec: str) -> list[float]:
    return [float(item) for item in str(spec).split(",") if item.strip()]


def _parse_int_list(spec: str) -> list[int]:
    return [int(float(item)) for item in str(spec).split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v47 component-level temporal edge refinement scan.")
    parser.add_argument("--mask-vote-rows", default="outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_mask_vote_rows.csv")
    parser.add_argument("--edge-table", default="outputs/audit/v47_adjacent_edges_gap2_only/temporal_candidate_edge_table.csv")
    parser.add_argument("--edge-types", default="adjacent,skip")
    parser.add_argument("--score-thresholds", default="0.99,0.97,0.95,0.93,0.90,0.87,0.85,0.80,0.75")
    parser.add_argument("--min-pair-edge-counts", default="1,2,3")
    parser.add_argument("--max-visible-outside-values", default="0.45,0.60,0.75,0.90,1.0")
    parser.add_argument("--min-visible-carrier-values", default="0,4,8,16,30")
    parser.add_argument("--output-root", default="outputs/audit/v47_component_edge_refinement_union32_gap2")
    args = parser.parse_args()

    mask_vote_rows = read_csv(ROOT / str(args.mask_vote_rows))
    edge_rows = read_csv(ROOT / str(args.edge_table))
    edge_types = {item.strip() for item in str(args.edge_types).split(",") if item.strip()}
    node_to_component, component_meta = _node_component_rows(mask_vote_rows)
    candidate_stats = _candidate_pair_stats(edge_rows, node_to_component, component_meta, edge_types)
    component_ids = sorted(component_meta)

    rows: list[dict[str, Any]] = []
    score_thresholds = _parse_float_list(args.score_thresholds)
    min_pair_edge_counts = _parse_int_list(args.min_pair_edge_counts)
    max_visible_outside_values = _parse_float_list(args.max_visible_outside_values)
    min_visible_carrier_values = _parse_int_list(args.min_visible_carrier_values)
    for score_key in ["A5_d4rt_semantic_confirmation", "A4_d4rt_visible_veto", "A8_no_temporal_control", "A7_shuffled_D4RT"]:
        for score_threshold in score_thresholds:
            for min_pair_edge_count in min_pair_edge_counts:
                for max_visible_outside in max_visible_outside_values:
                    for min_visible_carriers in min_visible_carrier_values:
                        for forbid_conflict in [True, False]:
                            rows.append(
                                _evaluate(
                                    component_ids=component_ids,
                                    mask_vote_rows=mask_vote_rows,
                                    candidate_stats=candidate_stats,
                                    score_key=score_key,
                                    score_threshold=score_threshold,
                                    min_pair_edge_count=min_pair_edge_count,
                                    max_visible_outside=max_visible_outside,
                                    min_forward_visible_carriers=min_visible_carriers,
                                    min_backward_visible_carriers=min_visible_carriers,
                                    forbid_same_frame_conflict=forbid_conflict,
                                )
                            )

    rows.sort(key=lambda row: (parse_float(row.get("ARI")), parse_float(row.get("purity"))), reverse=True)
    real_rows = [row for row in rows if row["score_key"] in {"A5_d4rt_semantic_confirmation", "A4_d4rt_visible_veto"}]
    no_temporal_rows = [row for row in rows if row["score_key"] == "A8_no_temporal_control"]
    shuffled_rows = [row for row in rows if row["score_key"] == "A7_shuffled_D4RT"]
    best_real = real_rows[0] if real_rows else {}
    best_no_temporal = no_temporal_rows[0] if no_temporal_rows else {}
    best_shuffled = shuffled_rows[0] if shuffled_rows else {}
    summary = {
        "phase": "v47_component_edge_refinement",
        "mask_vote_rows": str(ROOT / str(args.mask_vote_rows)),
        "edge_table": str(ROOT / str(args.edge_table)),
        "edge_types": sorted(edge_types),
        "component_count": len(component_ids),
        "candidate_pair_count": len(candidate_stats),
        "rows": len(rows),
        "best_row": rows[0] if rows else None,
        "best_real_row": best_real,
        "best_no_temporal_row": best_no_temporal,
        "best_shuffled_row": best_shuffled,
        "best_real_minus_best_no_temporal_ARI": parse_float(best_real.get("ARI")) - parse_float(best_no_temporal.get("ARI")),
        "best_real_minus_best_shuffled_ARI": parse_float(best_real.get("ARI")) - parse_float(best_shuffled.get("ARI")),
        "gate": {
            "ARI_pass": bool(parse_float(best_real.get("ARI")) >= 0.465),
            "purity_pass": bool(parse_float(best_real.get("purity")) >= 0.865),
            "completeness_pass": bool(parse_float(best_real.get("completeness")) >= 0.535),
            "real_minus_no_temporal_pass": bool(
                parse_float(best_real.get("ARI")) - parse_float(best_no_temporal.get("ARI")) >= 0.15
            ),
            "real_minus_shuffled_pass": bool(parse_float(best_real.get("ARI")) - parse_float(best_shuffled.get("ARI")) >= 0.25),
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    summary["gate"]["pass"] = bool(all(summary["gate"].values()))

    out_root = ROOT / str(args.output_root)
    write_csv(out_root / "component_edge_refinement_scan_rows.csv", rows)
    write_csv(out_root / "component_edge_refinement_pair_stats.csv", list(candidate_stats.values()))
    write_json(out_root / "component_edge_refinement_summary.json", summary)
    print({"summary": str(out_root / "component_edge_refinement_summary.json"), "gate": summary["gate"]})


if __name__ == "__main__":
    main()
