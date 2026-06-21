from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .v47_common import (
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


def _select_edges(
    edge_rows: list[dict[str, Any]],
    *,
    score_key: str,
    min_score: float,
    edge_types: set[str],
    respect_edge_accept_candidate: bool,
) -> list[dict[str, Any]]:
    outgoing: set[int] = set()
    incoming: set[int] = set()
    selected: list[dict[str, Any]] = []
    candidates = [
        row
        for row in edge_rows
        if str(row.get("edge_type")) in edge_types
        and parse_float(row.get(score_key)) >= float(min_score)
        and (not bool(respect_edge_accept_candidate) or parse_bool(row.get("edge_accept_candidate", True)))
    ]
    candidates.sort(key=lambda row: (parse_float(row.get(score_key)), -parse_float(row.get("edge_cost"))), reverse=True)
    for row in candidates:
        src = parse_int(row["src_node_id"])
        dst = parse_int(row["dst_node_id"])
        if src in outgoing or dst in incoming:
            continue
        outgoing.add(src)
        incoming.add(dst)
        selected.append(row)
    return selected


def build_tracklets(
    *,
    mask_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    score_key: str = "A5_d4rt_semantic_confirmation",
    min_score: float = 0.30,
    edge_types: set[str] | None = None,
    respect_edge_accept_candidate: bool = True,
) -> dict[str, Any]:
    edge_types = edge_types or {"adjacent"}
    node_ids = [parse_int(row["node_id"]) for row in mask_rows]
    uf = UnionFind(node_ids)
    selected = _select_edges(
        edge_rows,
        score_key=score_key,
        min_score=float(min_score),
        edge_types=edge_types,
        respect_edge_accept_candidate=bool(respect_edge_accept_candidate),
    )
    for row in selected:
        uf.union(parse_int(row["src_node_id"]), parse_int(row["dst_node_id"]))
    labels_by_root: dict[int, str] = {}
    pred_labels: list[str] = []
    true_labels: list[str] = []
    labeled_node_count = 0
    node_rows: list[dict[str, Any]] = []
    for row in mask_rows:
        node_id = parse_int(row["node_id"])
        root = uf.find(node_id)
        if root not in labels_by_root:
            labels_by_root[root] = f"t{len(labels_by_root):05d}"
        pred = labels_by_root[root]
        gt = str(row.get("diagnostic_gt_instance", ""))
        node_rows.append(
            {
                "node_id": node_id,
                "tracklet_id": pred,
                "scene": row.get("scene"),
                "frame_id": row.get("frame_id"),
                "mask_id": row.get("mask_id"),
                "diagnostic_gt_instance": gt,
                "uses_gt_for_prediction": False,
            }
        )
        if gt:
            labeled_node_count += 1
            pred_labels.append(pred)
            true_labels.append(gt)
    tracklet_sizes = Counter(row["tracklet_id"] for row in node_rows)
    tracklet_gt_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in node_rows:
        gt = str(row.get("diagnostic_gt_instance", ""))
        if gt:
            tracklet_gt_counts[str(row["tracklet_id"])][gt] += 1
    impure_tracklets = sum(1 for counts in tracklet_gt_counts.values() if len(counts) > 1)
    summary = {
        "phase": "v47_tracklet_construction",
        "score_key": score_key,
        "min_score": float(min_score),
        "edge_types": sorted(edge_types),
        "respect_edge_accept_candidate": bool(respect_edge_accept_candidate),
        "tracklet_count": len(tracklet_sizes),
        "selected_edge_count": len(selected),
        "tracklet_length_mean": safe_mean(tracklet_sizes.values()),
        "tracklet_length_p50": safe_quantile(tracklet_sizes.values(), 0.50),
        "tracklet_length_p90": safe_quantile(tracklet_sizes.values(), 0.90),
        "tracklet_purity": cluster_purity(true_labels, pred_labels),
        "tracklet_completeness": cluster_completeness(true_labels, pred_labels),
        "tracklet_ARI": adjusted_rand_score(true_labels, pred_labels),
        "fragmentation_rate": None,
        "ID_switch_rate": float(impure_tracklets / max(len(tracklet_gt_counts), 1)),
        "labeled_node_count": labeled_node_count,
        "scene0081_tracklet_purity": _scene_metric(node_rows, "scene0081_01"),
        "scene0591_tracklet_purity": _scene_metric(node_rows, "scene0591_00"),
        "birth_from_d4rt_tube_count": 0,
        "maskless_object_count": 0,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    summary["gate"] = {
        "tracklet_purity_pass": bool(summary["tracklet_purity"] >= 0.90),
        "tracklet_length_mean_pass": bool((summary["tracklet_length_mean"] or 0.0) >= 1.30),
        "scene0081_tracklet_purity_pass": bool((summary["scene0081_tracklet_purity"] or 0.0) >= 0.88),
        "scene0591_tracklet_purity_pass": bool((summary["scene0591_tracklet_purity"] or 0.0) >= 0.88),
    }
    summary["gate"]["pass"] = bool(all(summary["gate"].values()))
    selected_rows = [dict(row, selected_for_tracklet=True) for row in selected]
    return {"tracklet_rows": node_rows, "selected_edge_rows": selected_rows, "summary": summary}


def _scene_metric(node_rows: list[dict[str, Any]], scene: str) -> float | None:
    pred: list[str] = []
    true: list[str] = []
    for row in node_rows:
        if str(row.get("scene")) != scene:
            continue
        gt = str(row.get("diagnostic_gt_instance", ""))
        if not gt:
            continue
        pred.append(str(row["tracklet_id"]))
        true.append(gt)
    if not true:
        return None
    return cluster_purity(true, pred)
