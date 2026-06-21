from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import networkx as nx

from stream4d_native.v47_common import (
    UnionFind,
    adjusted_rand_score,
    cluster_completeness,
    cluster_purity,
    parse_bool,
    parse_float,
    parse_int,
    safe_mean,
    safe_quantile,
)


def select_min_cost_circulation_edges(
    *,
    edge_rows: list[dict[str, Any]],
    score_key: str,
    min_score: float,
    edge_types: set[str],
    max_visible_outside: float,
    min_visible_carriers: int,
    respect_edge_accept_candidate: bool,
    cost_scale: int = 1_000_000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    graph = nx.DiGraph()
    src = "__source__"
    sink = "__sink__"
    graph.add_node(src, demand=0)
    graph.add_node(sink, demand=0)

    best: dict[tuple[int, int], tuple[float, int, dict[str, Any]]] = {}
    node_ids: set[int] = set()
    rejected = Counter()
    for row in edge_rows:
        if str(row.get("edge_type")) not in edge_types:
            rejected["edge_type"] += 1
            continue
        if respect_edge_accept_candidate and not parse_bool(row.get("edge_accept_candidate", True)):
            rejected["edge_accept_candidate"] += 1
            continue
        if parse_float(row.get("visible_outside"), 1.0) > float(max_visible_outside):
            rejected["visible_outside"] += 1
            continue
        if parse_int(row.get("forward_visible_carrier_count")) < int(min_visible_carriers):
            rejected["visible_carrier_count"] += 1
            continue
        if parse_int(row.get("backward_visible_carrier_count")) < int(min_visible_carriers):
            rejected["visible_carrier_count"] += 1
            continue
        score = parse_float(row.get(score_key))
        if score < float(min_score):
            rejected["min_score"] += 1
            continue
        left = parse_int(row.get("src_node_id"))
        right = parse_int(row.get("dst_node_id"))
        if left == right:
            rejected["self_edge"] += 1
            continue
        cost = int(round((float(min_score) - score) * int(cost_scale)))
        if cost >= 0:
            rejected["non_negative_cost"] += 1
            continue
        key = (left, right)
        if key not in best or cost < best[key][1]:
            best[key] = (score, cost, row)
            node_ids.add(left)
            node_ids.add(right)

    graph.add_edge(sink, src, capacity=max(len(node_ids), 1), weight=0)
    for node_id in sorted(node_ids):
        graph.add_edge(src, f"L:{node_id}", capacity=1, weight=0)
        graph.add_edge(f"R:{node_id}", sink, capacity=1, weight=0)
    for (left, right), (score, cost, row) in best.items():
        graph.add_edge(f"L:{left}", f"R:{right}", capacity=1, weight=cost, score=score, edge_row=row)

    if not best:
        return [], {
            "solver_type": "networkx_min_cost_circulation",
            "candidate_edge_count": 0,
            "selected_edge_count": 0,
            "flow_cost": 0,
            "rejected": dict(rejected),
        }

    flow_cost, flow = nx.network_simplex(graph)
    selected: list[dict[str, Any]] = []
    for (left, right), (score, cost, row) in best.items():
        if flow.get(f"L:{left}", {}).get(f"R:{right}", 0) > 0:
            selected.append(
                dict(
                    row,
                    true_flow_score=float(score),
                    true_flow_cost=int(cost),
                    selected_for_true_flow=True,
                    solver_type="networkx_min_cost_circulation",
                )
            )
    selected.sort(key=lambda row: (parse_int(row.get("src_node_id")), parse_int(row.get("dst_node_id"))))
    return selected, {
        "solver_type": "networkx_min_cost_circulation",
        "candidate_edge_count": len(best),
        "selected_edge_count": len(selected),
        "flow_cost": int(flow_cost),
        "rejected": dict(rejected),
    }


def evaluate_tracks(mask_rows: list[dict[str, Any]], selected_edges: list[dict[str, Any]]) -> dict[str, Any]:
    node_ids = [parse_int(row.get("node_id")) for row in mask_rows]
    uf = UnionFind(node_ids)
    for row in selected_edges:
        uf.union(parse_int(row.get("src_node_id")), parse_int(row.get("dst_node_id")))

    labels_by_root: dict[int, str] = {}
    pred_labels: list[str] = []
    true_labels: list[str] = []
    node_rows: list[dict[str, Any]] = []
    for row in mask_rows:
        node_id = parse_int(row.get("node_id"))
        root = uf.find(node_id)
        if root not in labels_by_root:
            labels_by_root[root] = f"tf{len(labels_by_root):05d}"
        pred = labels_by_root[root]
        gt = str(row.get("diagnostic_gt_instance", ""))
        node_rows.append(
            {
                "node_id": node_id,
                "true_flow_track_id": pred,
                "scene": row.get("scene"),
                "frame_id": row.get("frame_id"),
                "mask_id": row.get("mask_id"),
                "diagnostic_gt_instance": gt,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
        if gt:
            true_labels.append(gt)
            pred_labels.append(pred)

    track_sizes = Counter(row["true_flow_track_id"] for row in node_rows)
    track_frames: dict[str, set[int]] = defaultdict(set)
    scene_true: dict[str, list[str]] = defaultdict(list)
    scene_pred: dict[str, list[str]] = defaultdict(list)
    for row in node_rows:
        track_frames[str(row["true_flow_track_id"])].add(parse_int(row.get("frame_id")))
        gt = str(row.get("diagnostic_gt_instance", ""))
        if gt:
            scene_true[str(row.get("scene"))].append(gt)
            scene_pred[str(row.get("scene"))].append(str(row["true_flow_track_id"]))

    metrics = {
        "selected_edge_count": len(selected_edges),
        "track_count": len(track_sizes),
        "birth_count": len(track_sizes),
        "death_count": len(track_sizes),
        "temporal_span_mean": safe_mean(len(frames) for frames in track_frames.values()),
        "track_length_mean": safe_mean(track_sizes.values()),
        "track_length_p50": safe_quantile(track_sizes.values(), 0.50),
        "track_length_p90": safe_quantile(track_sizes.values(), 0.90),
        "ARI": adjusted_rand_score(true_labels, pred_labels),
        "purity": cluster_purity(true_labels, pred_labels),
        "completeness": cluster_completeness(true_labels, pred_labels),
        "scene0081_ARI": adjusted_rand_score(scene_true["scene0081_01"], scene_pred["scene0081_01"])
        if scene_true.get("scene0081_01")
        else None,
        "scene0011_purity": cluster_purity(scene_true["scene0011_00"], scene_pred["scene0011_00"])
        if scene_true.get("scene0011_00")
        else None,
        "scene0050_purity": cluster_purity(scene_true["scene0050_00"], scene_pred["scene0050_00"])
        if scene_true.get("scene0050_00")
        else None,
        "scene0591_completeness": cluster_completeness(scene_true["scene0591_00"], scene_pred["scene0591_00"])
        if scene_true.get("scene0591_00")
        else None,
        "birth_from_d4rt_tube_count": 0,
        "maskless_object_count": 0,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    return {"track_rows": node_rows, "metrics": metrics}

