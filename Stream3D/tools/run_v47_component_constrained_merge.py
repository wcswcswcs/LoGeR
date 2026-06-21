from __future__ import annotations

import argparse
from collections import defaultdict
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

    def union(self, left: str, right: str) -> str:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left
        return root_left


def _node_component_rows(mask_vote_rows: list[dict[str, Any]]) -> tuple[dict[int, str], dict[str, dict[str, Any]]]:
    node_to_component: dict[int, str] = {}
    component_meta: dict[str, dict[str, Any]] = {}
    for row in mask_vote_rows:
        component_id = str(row.get("predicted_component_object_id") or row.get("predicted_supertrack_object_id"))
        node_to_component[parse_int(row.get("node_id"))] = component_id
        meta = component_meta.setdefault(
            component_id,
            {
                "component_id": component_id,
                "scene": str(row.get("scene")),
                "frames": set(),
                "mask_count": 0,
            },
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
        src = node_to_component.get(parse_int(row.get("src_node_id")))
        dst = node_to_component.get(parse_int(row.get("dst_node_id")))
        if not src or not dst or src == dst:
            continue
        left = component_meta.get(src)
        right = component_meta.get(dst)
        if not left or not right or left["scene"] != right["scene"]:
            continue
        key = tuple(sorted([src, dst]))
        item = stats.setdefault(
            key,
            {
                "component_left": key[0],
                "component_right": key[1],
                "scene": left["scene"],
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
            key_name = f"max_{score_key}"
            item[key_name] = max(parse_float(item.get(key_name)), parse_float(row.get(score_key)))
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


def _labels(mask_vote_rows: list[dict[str, Any]], uf: _StringUnionFind) -> tuple[list[str], list[str]]:
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
    return true_labels, pred_labels


def _evaluate(
    *,
    component_ids: list[str],
    component_meta: dict[str, dict[str, Any]],
    mask_vote_rows: list[dict[str, Any]],
    candidate_stats: dict[tuple[str, str], dict[str, Any]],
    score_key: str,
    score_threshold: float,
    min_pair_edge_count: int,
    max_visible_outside: float,
    min_visible_carriers: int,
    max_selected_pairs: int,
    enforce_cluster_frame_exclusion: bool,
    forbid_pair_same_frame_conflict: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    score_col = f"max_{score_key}"
    candidates = []
    for pair, row in candidate_stats.items():
        if parse_int(row.get("edge_count")) < min_pair_edge_count:
            continue
        if parse_float(row.get(score_col)) < score_threshold:
            continue
        if parse_float(row.get("min_visible_outside"), 1.0) > max_visible_outside:
            continue
        if parse_int(row.get("max_forward_visible_carrier_count")) < min_visible_carriers:
            continue
        if parse_int(row.get("max_backward_visible_carrier_count")) < min_visible_carriers:
            continue
        if forbid_pair_same_frame_conflict and bool(row.get("same_frame_conflict")):
            continue
        candidates.append(row)
    candidates.sort(
        key=lambda row: (
            parse_float(row.get(score_col)),
            parse_int(row.get("edge_count")),
            -parse_float(row.get("min_visible_outside"), 1.0),
            parse_int(row.get("max_forward_visible_carrier_count")) + parse_int(row.get("max_backward_visible_carrier_count")),
        ),
        reverse=True,
    )

    uf = _StringUnionFind(component_ids)
    cluster_frames = {component_id: set(component_meta[component_id]["frames"]) for component_id in component_ids}
    selected_rows: list[dict[str, Any]] = []
    skipped_cluster_conflict = 0
    for row in candidates:
        left = str(row["component_left"])
        right = str(row["component_right"])
        root_left = uf.find(left)
        root_right = uf.find(right)
        if root_left == root_right:
            continue
        if enforce_cluster_frame_exclusion and cluster_frames[root_left] & cluster_frames[root_right]:
            skipped_cluster_conflict += 1
            continue
        new_root = uf.union(root_left, root_right)
        old_root = root_right if new_root == root_left else root_left
        cluster_frames[new_root] = cluster_frames[root_left] | cluster_frames[root_right]
        cluster_frames.pop(old_root, None)
        selected_rows.append(
            {
                **row,
                "score_key": score_key,
                "score_threshold": score_threshold,
                "selected_score": parse_float(row.get(score_col)),
                "selected_rank": len(selected_rows),
            }
        )
        if max_selected_pairs >= 0 and len(selected_rows) >= max_selected_pairs:
            break

    true_labels, pred_labels = _labels(mask_vote_rows, uf)
    result = {
        "score_key": score_key,
        "score_threshold": float(score_threshold),
        "min_pair_edge_count": int(min_pair_edge_count),
        "max_visible_outside": float(max_visible_outside),
        "min_visible_carriers": int(min_visible_carriers),
        "max_selected_pairs": int(max_selected_pairs),
        "enforce_cluster_frame_exclusion": bool(enforce_cluster_frame_exclusion),
        "forbid_pair_same_frame_conflict": bool(forbid_pair_same_frame_conflict),
        "candidate_pair_count_after_filter": len(candidates),
        "selected_pair_count": len(selected_rows),
        "skipped_cluster_conflict_count": skipped_cluster_conflict,
        "cluster_count": len(set(pred_labels)),
        "ARI": adjusted_rand_score(true_labels, pred_labels),
        "purity": cluster_purity(true_labels, pred_labels),
        "completeness": cluster_completeness(true_labels, pred_labels),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    return result, selected_rows


def _parse_float_list(spec: str) -> list[float]:
    return [float(item) for item in str(spec).split(",") if item.strip()]


def _parse_int_list(spec: str) -> list[int]:
    return [int(float(item)) for item in str(spec).split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v47 cluster-constrained component merge scan.")
    parser.add_argument("--mask-vote-rows", default="outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_mask_vote_rows.csv")
    parser.add_argument("--edge-table", default="outputs/audit/v47_adjacent_edges_gap2_only/temporal_candidate_edge_table.csv")
    parser.add_argument("--edge-types", default="adjacent,skip")
    parser.add_argument("--score-thresholds", default="0.99,0.97,0.95,0.93,0.90,0.87,0.85,0.80,0.75,0.70,0.65")
    parser.add_argument("--min-pair-edge-counts", default="1,2,3")
    parser.add_argument("--max-visible-outside-values", default="0.45,0.60,0.75,0.90,1.0")
    parser.add_argument("--min-visible-carrier-values", default="0,4,8,16,30")
    parser.add_argument("--max-selected-pair-values", default="25,50,75,100,150,200,300,500,-1")
    parser.add_argument("--output-root", default="outputs/audit/v47_component_constrained_merge_union32_gap2")
    args = parser.parse_args()

    mask_vote_rows = read_csv(ROOT / str(args.mask_vote_rows))
    edge_rows = read_csv(ROOT / str(args.edge_table))
    edge_types = {item.strip() for item in str(args.edge_types).split(",") if item.strip()}
    node_to_component, component_meta = _node_component_rows(mask_vote_rows)
    candidate_stats = _candidate_pair_stats(edge_rows, node_to_component, component_meta, edge_types)
    component_ids = sorted(component_meta)

    rows: list[dict[str, Any]] = []
    selected_by_signature: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for score_key in ["A5_d4rt_semantic_confirmation", "A4_d4rt_visible_veto", "A8_no_temporal_control", "A7_shuffled_D4RT"]:
        for score_threshold in _parse_float_list(args.score_thresholds):
            for min_pair_edge_count in _parse_int_list(args.min_pair_edge_counts):
                for max_visible_outside in _parse_float_list(args.max_visible_outside_values):
                    for min_visible_carriers in _parse_int_list(args.min_visible_carrier_values):
                        for max_selected_pairs in _parse_int_list(args.max_selected_pair_values):
                            for enforce_cluster_frame_exclusion in [True, False]:
                                for forbid_pair_same_frame_conflict in [True, False]:
                                    result, selected = _evaluate(
                                        component_ids=component_ids,
                                        component_meta=component_meta,
                                        mask_vote_rows=mask_vote_rows,
                                        candidate_stats=candidate_stats,
                                        score_key=score_key,
                                        score_threshold=score_threshold,
                                        min_pair_edge_count=min_pair_edge_count,
                                        max_visible_outside=max_visible_outside,
                                        min_visible_carriers=min_visible_carriers,
                                        max_selected_pairs=max_selected_pairs,
                                        enforce_cluster_frame_exclusion=enforce_cluster_frame_exclusion,
                                        forbid_pair_same_frame_conflict=forbid_pair_same_frame_conflict,
                                    )
                                    signature = (
                                        score_key,
                                        score_threshold,
                                        min_pair_edge_count,
                                        max_visible_outside,
                                        min_visible_carriers,
                                        max_selected_pairs,
                                        enforce_cluster_frame_exclusion,
                                        forbid_pair_same_frame_conflict,
                                    )
                                    selected_by_signature[signature] = selected
                                    rows.append(result)

    rows.sort(key=lambda row: (parse_float(row.get("ARI")), parse_float(row.get("purity"))), reverse=True)
    real_rows = [row for row in rows if row["score_key"] in {"A5_d4rt_semantic_confirmation", "A4_d4rt_visible_veto"}]
    no_temporal_rows = [row for row in rows if row["score_key"] == "A8_no_temporal_control"]
    shuffled_rows = [row for row in rows if row["score_key"] == "A7_shuffled_D4RT"]
    best_real = real_rows[0] if real_rows else {}
    best_no_temporal = no_temporal_rows[0] if no_temporal_rows else {}
    best_shuffled = shuffled_rows[0] if shuffled_rows else {}

    def signature(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            row.get("score_key"),
            row.get("score_threshold"),
            row.get("min_pair_edge_count"),
            row.get("max_visible_outside"),
            row.get("min_visible_carriers"),
            row.get("max_selected_pairs"),
            row.get("enforce_cluster_frame_exclusion"),
            row.get("forbid_pair_same_frame_conflict"),
        )

    summary = {
        "phase": "v47_component_constrained_merge",
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
    write_csv(out_root / "component_constrained_merge_scan_rows.csv", rows)
    write_csv(out_root / "component_constrained_merge_pair_stats.csv", list(candidate_stats.values()))
    write_csv(out_root / "component_constrained_merge_best_real_selected_pairs.csv", selected_by_signature.get(signature(best_real), []))
    write_csv(out_root / "component_constrained_merge_best_no_temporal_selected_pairs.csv", selected_by_signature.get(signature(best_no_temporal), []))
    write_csv(out_root / "component_constrained_merge_best_shuffled_selected_pairs.csv", selected_by_signature.get(signature(best_shuffled), []))
    write_json(out_root / "component_constrained_merge_summary.json", summary)
    print({"summary": str(out_root / "component_constrained_merge_summary.json"), "gate": summary["gate"]})


if __name__ == "__main__":
    main()
