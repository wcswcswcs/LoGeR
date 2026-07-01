from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from stream4d_native.v67_local_baselines import _oracle_majority_mapping_bundle  # noqa: E402
from stream4d_native.v68_local_graph_solver import (  # noqa: E402
    DSU,
    _can_merge,
    _node_from_token,
    _row_from_mapping,
    _same_frame_violation_count,
    _summarize_variant_all,
)
from tools.run_v65_scene_multiview_ap import _sha256, _write_csv, _write_json  # noqa: E402
from tools.run_v66_local_chunk_eval import _chunk_rows, _float_or_none, _frame_data, _load_csv_rows, _mean, _rel  # noqa: E402
from tools.run_v66_scene_mv_ap_probe5 import DEFAULT_SCENES, _discover_pipeline_root, _mask_dir_from_pipeline, _parse_csv_list  # noqa: E402


CAPSULE_VARIANTS = [
    "OC1_candidate_identity",
    "OC2_appearance_top1_tracklet_min025",
    "OC3_appearance_cc_t060",
    "OC4_shared_ledger_carrier_veto_t055",
    "OC5_temporal_coverage_reward_t045",
    "OC6_single_frame_penalty_seed055_absorb035",
    "OC11_shared_anchor_coref_t055_b050",
    "OC12_shared_anchor_coref_drop_singletons_t055_b050",
]
ORACLE_VARIANTS = [
    "OC7_oracle_all_candidates_diagnostic",
    "OC8_oracle_nonshared_candidates_diagnostic",
    "OC9_oracle_all_candidates_cannotlink_diagnostic",
    "OC10_oracle_nonshared_cannotlink_diagnostic",
]


def _rooted(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _load_candidates(path: Path, scenes: set[str]) -> dict[str, dict[tuple[int, int], dict[str, Any]]]:
    out: dict[str, dict[tuple[int, int], dict[str, Any]]] = defaultdict(dict)
    for row in _load_csv_rows(path):
        scene = str(row.get("scene_id") or "")
        if scene not in scenes or not _parse_bool(row.get("representative_available")):
            continue
        node = (_safe_int(row.get("frame_id")), _safe_int(row.get("mask_id")))
        out[str(row.get("chunk_id"))][node] = {
            "shared": _parse_bool(row.get("shared_support_only")),
            "underseg": _parse_bool(row.get("underseg_risk")),
            "large": _parse_bool(row.get("large_mask_risk")),
            "small": _parse_bool(row.get("small_mask_risk")),
            "area_ratio": _safe_float(row.get("area_ratio")),
            "signature": str(row.get("repeated_signature_id") or ""),
            "semantic": str(row.get("semantic_mode_id") or ""),
        }
    return out


def _load_edge_rows(path: Path, scenes: set[str], score_key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _load_csv_rows(path):
        scene = str(row.get("scene_id") or "")
        if scene not in scenes:
            continue
        left = _node_from_token(str(row.get("node_i") or ""))
        right = _node_from_token(str(row.get("node_j") or ""))
        if left is None or right is None:
            continue
        score = _safe_float(row.get(score_key), 0.0)
        out[str(row.get("chunk_id"))].append(
            {
                "left": left,
                "right": right,
                "score": score,
                "base_score": score,
                "same_frame": _parse_bool(row.get("same_frame")),
                "hard_negative": _parse_bool(row.get("hard_negative_candidate")),
                "frame_delta": abs(int(left[0]) - int(right[0])),
                "appearance_mode_match": _parse_bool(row.get("appearance_mode_match")),
                "signature_match": _parse_bool(row.get("signature_match")),
                "semantic_match": _parse_bool(row.get("semantic_match")),
            }
        )
    return out


def _load_carrier_stats(path: Path, scenes: set[str]) -> dict[tuple[str, tuple[int, int], tuple[int, int]], dict[str, float]]:
    stats: dict[tuple[str, tuple[int, int], tuple[int, int]], dict[str, float]] = {}
    for row in _load_csv_rows(path):
        scene = str(row.get("scene_id") or "")
        if scene not in scenes:
            continue
        chunk_id = str(row.get("chunk_id") or "")
        left = (_safe_int(row.get("anchor_frame")), _safe_int(row.get("anchor_mask")))
        right = (_safe_int(row.get("candidate_frame")), _safe_int(row.get("candidate_mask")))
        inside = _safe_float(row.get("inside_ratio"), 0.0)
        outside = _safe_float(row.get("outside_ratio"), 0.0)
        visible = _safe_float(row.get("visible_carrier_count"), 0.0)
        value = {"inside": inside, "outside": outside, "visible": visible, "residual": inside - outside}
        for key in [(chunk_id, left, right), (chunk_id, right, left)]:
            old = stats.get(key)
            if old is None or (value["residual"], value["visible"]) > (old["residual"], old["visible"]):
                stats[key] = value
    return stats


def _temporal_score(frame_delta: int) -> float:
    steps = max(1.0, abs(float(frame_delta)) / 5.0)
    return float(1.0 / (1.0 + 0.18 * max(0.0, steps - 1.0)))


def _prepare_edges(
    *,
    variant: str,
    chunk_id: str,
    raw_edges: list[dict[str, Any]],
    nodes: set[tuple[int, int]],
    shared_nodes: set[tuple[int, int]],
    carrier_stats: dict[tuple[str, tuple[int, int], tuple[int, int]], dict[str, float]],
    capsule_rows: list[dict[str, Any]],
    scene: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    carrier_seen = 0
    carrier_vetoed = 0
    shared_edge_count = 0
    for edge in raw_edges:
        left = edge["left"]
        right = edge["right"]
        if left not in nodes or right not in nodes:
            continue
        if bool(edge.get("same_frame")) or int(left[0]) == int(right[0]):
            continue
        if variant in {"OC4_shared_ledger_carrier_veto_t055", "OC5_temporal_coverage_reward_t045", "OC6_single_frame_penalty_seed055_absorb035"}:
            if left in shared_nodes or right in shared_nodes:
                shared_edge_count += 1
                continue
        score = float(edge.get("base_score") or 0.0)
        stats = carrier_stats.get((chunk_id, left, right))
        if stats is not None:
            carrier_seen += 1
            if variant in {
                "OC4_shared_ledger_carrier_veto_t055",
                "OC5_temporal_coverage_reward_t045",
                "OC6_single_frame_penalty_seed055_absorb035",
                "OC11_shared_anchor_coref_t055_b050",
                "OC12_shared_anchor_coref_drop_singletons_t055_b050",
            }:
                if stats["visible"] >= 8 and stats["outside"] >= 0.62 and stats["inside"] <= 0.42:
                    carrier_vetoed += 1
                    continue
            if variant in {
                "OC5_temporal_coverage_reward_t045",
                "OC6_single_frame_penalty_seed055_absorb035",
                "OC11_shared_anchor_coref_t055_b050",
                "OC12_shared_anchor_coref_drop_singletons_t055_b050",
            }:
                score += 0.08 * max(0.0, stats["residual"])
        if variant in {
            "OC5_temporal_coverage_reward_t045",
            "OC6_single_frame_penalty_seed055_absorb035",
            "OC11_shared_anchor_coref_t055_b050",
            "OC12_shared_anchor_coref_drop_singletons_t055_b050",
        }:
            score += 0.08 * _temporal_score(int(edge.get("frame_delta") or 0))
        prepared_edge = dict(edge)
        prepared_edge["score"] = float(min(1.0, max(0.0, score)))
        prepared_edge["carrier_inside"] = None if stats is None else stats["inside"]
        prepared_edge["carrier_outside"] = None if stats is None else stats["outside"]
        prepared_edge["carrier_visible"] = None if stats is None else stats["visible"]
        prepared.append(prepared_edge)
    for edge in sorted(prepared, key=lambda item: float(item.get("score") or 0.0), reverse=True)[:2000]:
        capsule_rows.append(
            {
                "scene_id": scene,
                "chunk_id": chunk_id,
                "capsule_variant": variant,
                "left_frame": edge["left"][0],
                "left_mask": edge["left"][1],
                "right_frame": edge["right"][0],
                "right_mask": edge["right"][1],
                "capsule_edge_score": edge["score"],
                "base_edge_score": edge["base_score"],
                "frame_delta": edge["frame_delta"],
                "carrier_inside": edge.get("carrier_inside"),
                "carrier_outside": edge.get("carrier_outside"),
                "carrier_visible": edge.get("carrier_visible"),
                "uses_gt_for_prediction": False,
                "diagnostic_only": False,
                "forbidden_for_method_table": False,
            }
        )
    return prepared, {
        "candidate_edge_count": int(len(raw_edges)),
        "usable_edge_count": int(len(prepared)),
        "carrier_matched_edge_count": int(carrier_seen),
        "carrier_vetoed_edge_count": int(carrier_vetoed),
        "shared_edge_reject_count": int(shared_edge_count),
    }


def _identity_mapping(nodes: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    return {node: idx + 1 for idx, node in enumerate(sorted(nodes))}


def _edge_cc_mapping(
    nodes: set[tuple[int, int]],
    edges: list[dict[str, Any]],
    *,
    threshold: float,
    reject: set[tuple[int, int]] | None = None,
) -> dict[tuple[int, int], int]:
    reject = reject or set()
    active = sorted(nodes - reject)
    dsu = DSU(active)
    members: dict[tuple[int, int], set[tuple[int, int]]] = {node: {node} for node in active}
    for edge in sorted(edges, key=lambda row: float(row.get("score") or 0.0), reverse=True):
        if float(edge.get("score") or 0.0) < float(threshold):
            continue
        left, right = edge["left"], edge["right"]
        if left not in members or right not in members:
            continue
        root_left = dsu.find(left)
        root_right = dsu.find(right)
        if root_left == root_right or not _can_merge(members[root_left], members[root_right]):
            continue
        dsu.union(root_left, root_right)
        new_root = dsu.find(root_left)
        merged = members.pop(root_left, {root_left}) | members.pop(root_right, {root_right})
        members.pop(root_right, None)
        members[new_root] = merged
    root_to_id: dict[tuple[int, int], int] = {}
    mapping: dict[tuple[int, int], int] = {}
    for node in active:
        root = dsu.find(node)
        if root not in root_to_id:
            root_to_id[root] = len(root_to_id) + 1
        mapping[node] = root_to_id[root]
    return mapping


def _topk_mapping(nodes: set[tuple[int, int]], edges: list[dict[str, Any]], *, topk: int, min_score: float) -> dict[tuple[int, int], int]:
    by_node: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        by_node[edge["left"]].append(edge)
        by_node[edge["right"]].append(edge)
    selected: dict[tuple[tuple[int, int], tuple[int, int]], dict[str, Any]] = {}
    for node, items in by_node.items():
        for edge in sorted(items, key=lambda item: float(item.get("score") or 0.0), reverse=True)[: int(topk)]:
            if float(edge.get("score") or 0.0) >= float(min_score):
                selected[tuple(sorted([edge["left"], edge["right"]]))] = edge
    return _edge_cc_mapping(nodes, list(selected.values()), threshold=min_score)


def _seed_absorb_selected_mapping(
    nodes: set[tuple[int, int]],
    edges: list[dict[str, Any]],
    *,
    seed_threshold: float,
    absorb_threshold: float,
    reject: set[tuple[int, int]],
    keep_singletons: bool,
) -> dict[tuple[int, int], int]:
    active = sorted(nodes - reject)
    dsu = DSU(active)
    members: dict[tuple[int, int], set[tuple[int, int]]] = {node: {node} for node in active}
    seeded_nodes: set[tuple[int, int]] = set()
    for edge in sorted(edges, key=lambda row: float(row.get("score") or 0.0), reverse=True):
        if float(edge.get("score") or 0.0) < float(seed_threshold):
            continue
        left, right = edge["left"], edge["right"]
        if left not in members or right not in members:
            continue
        root_left = dsu.find(left)
        root_right = dsu.find(right)
        if root_left == root_right or not _can_merge(members[root_left], members[root_right]):
            continue
        dsu.union(root_left, root_right)
        new_root = dsu.find(root_left)
        merged = members.pop(root_left, {root_left}) | members.pop(root_right, {root_right})
        members.pop(root_right, None)
        members[new_root] = merged
        seeded_nodes.update({left, right})
    root_members: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    for node in active:
        root_members[dsu.find(node)].add(node)
    selected_roots = {root for root, group in root_members.items() if len(group & seeded_nodes) >= 2 and len({frame for frame, _mask in group}) >= 2}
    object_frames: dict[tuple[int, int], set[int]] = {root: {frame for frame, _mask in root_members[root]} for root in selected_roots}
    node_to_root: dict[tuple[int, int], tuple[int, int]] = {}
    for root in selected_roots:
        for node in root_members[root]:
            node_to_root[node] = root
    by_node: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        by_node[edge["left"]].append(edge)
        by_node[edge["right"]].append(edge)
    for node in active:
        if node in node_to_root:
            continue
        best: tuple[float, tuple[int, int]] | None = None
        for edge in by_node.get(node, []):
            score = float(edge.get("score") or 0.0)
            if score < float(absorb_threshold):
                continue
            other = edge["right"] if edge["left"] == node else edge["left"]
            root = node_to_root.get(other)
            if root is None or int(node[0]) in object_frames[root]:
                continue
            if best is None or score > best[0]:
                best = (score, root)
        if best is not None:
            root = best[1]
            node_to_root[node] = root
            object_frames[root].add(int(node[0]))
    if keep_singletons:
        for node in active:
            if node not in node_to_root:
                node_to_root[node] = node
    root_to_id: dict[tuple[int, int], int] = {}
    mapping: dict[tuple[int, int], int] = {}
    for node, root in sorted(node_to_root.items()):
        if root not in root_to_id:
            root_to_id[root] = len(root_to_id) + 1
        mapping[node] = root_to_id[root]
    return mapping


def _shared_anchor_coref_mapping(
    nodes: set[tuple[int, int]],
    shared_nodes: set[tuple[int, int]],
    edges: list[dict[str, Any]],
    *,
    base_threshold: float,
    bridge_threshold: float,
    drop_singletons: bool,
) -> dict[tuple[int, int], int]:
    base_mapping = _edge_cc_mapping(nodes, edges, threshold=base_threshold, reject=shared_nodes)
    members: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for node, object_id in base_mapping.items():
        members[int(object_id)].add(node)
    parent = {object_id: object_id for object_id in members}

    def find(object_id: int) -> int:
        parent.setdefault(object_id, object_id)
        if parent[object_id] != object_id:
            parent[object_id] = find(parent[object_id])
        return parent[object_id]

    def union(left_id: int, right_id: int) -> bool:
        left_root = find(left_id)
        right_root = find(right_id)
        if left_root == right_root:
            return False
        if not _can_merge(members[left_root], members[right_root]):
            return False
        parent[right_root] = left_root
        members[left_root] = members[left_root] | members.pop(right_root)
        return True

    evidence_by_shared: dict[tuple[int, int], dict[int, float]] = defaultdict(dict)
    for edge in edges:
        left = edge["left"]
        right = edge["right"]
        score = float(edge.get("score") or 0.0)
        if score < float(bridge_threshold):
            continue
        shared: tuple[int, int] | None = None
        other: tuple[int, int] | None = None
        if left in shared_nodes and right in base_mapping:
            shared, other = left, right
        elif right in shared_nodes and left in base_mapping:
            shared, other = right, left
        if shared is None or other is None:
            continue
        object_id = int(base_mapping[other])
        old = evidence_by_shared[shared].get(object_id)
        if old is None or score > old:
            evidence_by_shared[shared][object_id] = score

    for object_scores in evidence_by_shared.values():
        ranked = sorted(object_scores.items(), key=lambda item: item[1], reverse=True)[:6]
        for left_idx, (left_id, _left_score) in enumerate(ranked):
            for right_id, _right_score in ranked[left_idx + 1 :]:
                union(left_id, right_id)

    merged_members: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for object_id, group in list(members.items()):
        merged_members[find(object_id)].update(group)

    root_to_id: dict[int, int] = {}
    mapping: dict[tuple[int, int], int] = {}
    for root, group in sorted(merged_members.items(), key=lambda item: sorted(item[1])):
        if drop_singletons and len({frame for frame, _mask in group}) < 2:
            continue
        root_to_id[root] = len(root_to_id) + 1
        for node in group:
            mapping[node] = root_to_id[root]
    return mapping


def _capsule_temporal_span(mapping: dict[tuple[int, int], int]) -> float | None:
    frames_by_object: dict[int, set[int]] = defaultdict(set)
    for (frame_id, _mask_id), object_id in mapping.items():
        frames_by_object[int(object_id)].add(int(frame_id))
    spans = [len(frames) for frames in frames_by_object.values()]
    return float(np.mean(spans)) if spans else None


def _oracle_cannotlink_mapping(
    *,
    frame_data: list[dict[str, Any]],
    nodes: set[tuple[int, int]],
) -> tuple[dict[tuple[int, int], int], dict[str, Any]]:
    node_set = set(nodes)
    best_by_gt_frame: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    candidate_count = 0
    label_by_node: dict[tuple[int, int], int] = {}
    for item in frame_data:
        frame_id = int(item["frame_id"])
        mask = item["mask"]
        gt = item["gt"]
        if mask is None or gt is None:
            continue
        for mask_id in np.unique(mask):
            mask_id_i = int(mask_id)
            node = (frame_id, mask_id_i)
            if mask_id_i <= 0 or node not in node_set:
                continue
            pixels = gt[mask == mask_id_i]
            pixels = pixels[pixels > 0]
            if pixels.size == 0:
                continue
            labels, counts = np.unique(pixels, return_counts=True)
            idx = int(np.argmax(counts))
            label = int(labels[idx])
            count = int(counts[idx])
            candidate_count += 1
            label_by_node[node] = label
            key = (label, frame_id)
            old = best_by_gt_frame.get(key)
            if old is None or count > old[0]:
                best_by_gt_frame[key] = (count, node)
    gt_to_object: dict[int, int] = {}
    mapping: dict[tuple[int, int], int] = {}
    for (label, _frame_id), (_count, node) in sorted(best_by_gt_frame.items()):
        if label not in gt_to_object:
            gt_to_object[label] = len(gt_to_object) + 1
        mapping[node] = gt_to_object[label]
    return mapping, {
        "oracle_candidate_with_gt_count": int(candidate_count),
        "oracle_selected_count": int(len(mapping)),
        "oracle_gt_object_count": int(len(gt_to_object)),
        "same_frame_cannot_link_violation_count": _same_frame_violation_count(mapping),
    }


def _build_capsule_mapping(
    *,
    variant: str,
    nodes: set[tuple[int, int]],
    shared_nodes: set[tuple[int, int]],
    edges: list[dict[str, Any]],
) -> dict[tuple[int, int], int]:
    if variant == "OC1_candidate_identity":
        return _identity_mapping(nodes)
    if variant == "OC2_appearance_top1_tracklet_min025":
        return _topk_mapping(nodes, edges, topk=1, min_score=0.25)
    if variant == "OC3_appearance_cc_t060":
        return _edge_cc_mapping(nodes, edges, threshold=0.60)
    if variant == "OC4_shared_ledger_carrier_veto_t055":
        return _edge_cc_mapping(nodes, edges, threshold=0.55, reject=shared_nodes)
    if variant == "OC5_temporal_coverage_reward_t045":
        return _edge_cc_mapping(nodes, edges, threshold=0.45, reject=shared_nodes)
    if variant == "OC6_single_frame_penalty_seed055_absorb035":
        return _seed_absorb_selected_mapping(
            nodes,
            edges,
            seed_threshold=0.55,
            absorb_threshold=0.35,
            reject=shared_nodes,
            keep_singletons=False,
        )
    if variant == "OC11_shared_anchor_coref_t055_b050":
        return _shared_anchor_coref_mapping(
            nodes,
            shared_nodes,
            edges,
            base_threshold=0.55,
            bridge_threshold=0.50,
            drop_singletons=False,
        )
    if variant == "OC12_shared_anchor_coref_drop_singletons_t055_b050":
        return _shared_anchor_coref_mapping(
            nodes,
            shared_nodes,
            edges,
            base_threshold=0.55,
            bridge_threshold=0.50,
            drop_singletons=True,
        )
    raise ValueError(f"unknown v70 object capsule variant: {variant}")


def _summarize_capsule_variant(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    base = _summarize_variant_all(rows, variant)
    subset = [row for row in rows if row["variant"] == variant]
    base.update(
        {
            "capsule_variant": variant,
            "capsule_count": int(sum(int(float(row.get("local_object_count") or 0)) for row in subset)),
            "capsule_count_per_chunk": _mean([_float_or_none(row.get("local_object_count")) for row in subset]),
            "selected_capsule_count": int(sum(int(float(row.get("selected_capsule_count") or 0)) for row in subset)),
            "support_masklet_count_mean": _mean([_float_or_none(row.get("support_pair_count")) for row in subset]),
            "shared_masklet_count_mean": _mean([_float_or_none(row.get("shared_mask_count")) for row in subset]),
            "local_SF25": base.get("local_AP25_mean"),
            "local_SF50": base.get("local_score_free_match50_recall_mean"),
            "local_AP50": base.get("local_AP50_mean"),
            "GT_best_IoU_mean": base.get("local_GT_best_IoU_mean_mean"),
            "pred_best_IoU_median": base.get("local_pred_best_IoU_median_mean"),
            "single_frame_object_rate": base.get("single_frame_object_rate_mean"),
            "fragments_per_GT@0.10": base.get("mean_fragments_per_GT@0.10_mean"),
            "GT_per_pred@0.10": base.get("mean_GT_per_pred@0.10_mean"),
            "duplicate_frame_mask_conflict_rate": base.get("local_duplicate_frame_mask_conflict_rate_mean"),
            "same_frame_violation_count": base.get("same_frame_cannot_link_violation_count_sum"),
            "underseg_false_bridge_rate": _mean([_float_or_none(row.get("underseg_false_bridge_rate")) for row in subset]),
            "capsule_temporal_span_mean": _mean([_float_or_none(row.get("capsule_temporal_span_mean")) for row in subset]),
            "carrier_vetoed_edge_count_mean": _mean([_float_or_none(row.get("carrier_vetoed_edge_count")) for row in subset]),
            "unknown_mask_count_mean": base.get("unknown_mask_count_mean"),
        }
    )
    return base


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _rooted(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    scenes = set(_parse_csv_list(args.scenes))
    variants = _parse_csv_list(args.variants) or CAPSULE_VARIANTS
    candidate_path = _rooted(args.candidate_rows)
    edge_path = _rooted(args.edge_rows)
    witness_path = _rooted(args.witness_rows)
    candidate_by_chunk = _load_candidates(candidate_path, scenes)
    edge_by_chunk = _load_edge_rows(edge_path, scenes, str(args.edge_score_key))
    carrier_stats = _load_carrier_stats(witness_path, scenes)
    capsule_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    pipeline_roots: dict[str, str] = {}
    for scene in sorted(scenes):
        print(f"[v70-object-capsule] scene={scene}", file=sys.stderr, flush=True)
        pipeline_root = _discover_pipeline_root(scene)
        if pipeline_root is None:
            missing_rows.append({"scene_id": scene, "missing": "soma_fullscene_pipeline_root"})
            continue
        pipeline_roots[scene] = _rel(pipeline_root)
        mask_dir = _mask_dir_from_pipeline(pipeline_root)
        stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
        all_stride_frames = stream.frame_ids(stride=int(args.stride), max_frames=None)
        for chunk in _chunk_rows(pipeline_root, scene):
            t0 = time.time()
            chunk_id = str(chunk.get("chunk_id"))
            raw_start = int(float(chunk.get("raw_frame_start") or 0))
            raw_end = int(float(chunk.get("raw_frame_end") or raw_start))
            frame_ids = [int(frame) for frame in all_stride_frames if raw_start <= int(frame) <= raw_end]
            candidate_meta = candidate_by_chunk.get(chunk_id, {})
            nodes = set(candidate_meta)
            if not frame_ids or not nodes:
                continue
            shared_nodes = {node for node, meta in candidate_meta.items() if bool(meta.get("shared")) or bool(meta.get("underseg"))}
            frame_data = _frame_data(scene, frame_ids, mask_dir)
            raw_edges = edge_by_chunk.get(chunk_id, [])
            for variant in variants:
                edges, edge_diag = _prepare_edges(
                    variant=variant,
                    chunk_id=chunk_id,
                    raw_edges=raw_edges,
                    nodes=nodes,
                    shared_nodes=shared_nodes,
                    carrier_stats=carrier_stats,
                    capsule_rows=capsule_rows,
                    scene=scene,
                )
                mapping = _build_capsule_mapping(variant=variant, nodes=nodes, shared_nodes=shared_nodes, edges=edges)
                selected = set(mapping)
                underseg_selected = len(selected & shared_nodes)
                diag = {
                    "duplicate_frame_mask_conflict_pairs": 0,
                    "duplicate_frame_mask_conflict_rate": 0.0,
                    "same_frame_cannot_link_violation_count": _same_frame_violation_count(mapping),
                    "shared_mask_count": int(len(shared_nodes)),
                    "reject_mask_count": int(len(nodes - selected)),
                    "unknown_mask_count": int(len(nodes - selected)),
                    "underseg_false_bridge_rate": float(underseg_selected / max(1, len(selected))),
                }
                diag.update(edge_diag)
                row = _row_from_mapping(
                    scene=scene,
                    chunk_id=chunk_id,
                    variant=variant,
                    frame_ids=frame_ids,
                    chunk=chunk,
                    frame_data=frame_data,
                    mapping=mapping,
                    diag=diag,
                    pipeline_root=pipeline_root,
                    uses_gt_for_prediction=False,
                    forbidden_for_method_table=False,
                )
                row.update(diag)
                row["selected_capsule_count"] = int(len(set(mapping.values())))
                row["capsule_temporal_span_mean"] = _capsule_temporal_span(mapping)
                row["runtime_sec"] = float(time.time() - t0)
                row["edge_score_key"] = str(args.edge_score_key)
                row["candidate_rows"] = _rel(candidate_path)
                row["edge_rows"] = _rel(edge_path)
                row["witness_rows"] = _rel(witness_path)
                chunk_rows.append(row)
            if bool(args.include_oracle):
                oracle_specs = [
                    ("OC7_oracle_all_candidates_diagnostic", nodes),
                    ("OC8_oracle_nonshared_candidates_diagnostic", nodes - shared_nodes),
                ]
                for variant, oracle_nodes in oracle_specs:
                    if not oracle_nodes:
                        continue
                    oracle_mapping, oracle_diag = _oracle_majority_mapping_bundle(
                        frame_data=frame_data,
                        selected_pairs=set(),
                        representative_pairs=oracle_nodes,
                    )["representative"]
                    diag = {
                        "duplicate_frame_mask_conflict_pairs": 0,
                        "duplicate_frame_mask_conflict_rate": 0.0,
                        "same_frame_cannot_link_violation_count": _same_frame_violation_count(oracle_mapping),
                        "shared_mask_count": int(len(shared_nodes)),
                        "reject_mask_count": int(len(nodes - oracle_nodes)),
                        "unknown_mask_count": int(len(nodes - set(oracle_mapping))),
                        "underseg_false_bridge_rate": float(len(set(oracle_mapping) & shared_nodes) / max(1, len(oracle_mapping))),
                        "selected_capsule_count": int(len(set(oracle_mapping.values()))),
                    }
                    diag.update(oracle_diag)
                    row = _row_from_mapping(
                        scene=scene,
                        chunk_id=chunk_id,
                        variant=variant,
                        frame_ids=frame_ids,
                        chunk=chunk,
                        frame_data=frame_data,
                        mapping=oracle_mapping,
                        diag=diag,
                        pipeline_root=pipeline_root,
                        uses_gt_for_prediction=True,
                        forbidden_for_method_table=True,
                    )
                    row.update(diag)
                    row["selected_capsule_count"] = int(len(set(oracle_mapping.values())))
                    row["capsule_temporal_span_mean"] = _capsule_temporal_span(oracle_mapping)
                    row["runtime_sec"] = float(time.time() - t0)
                    row["edge_score_key"] = str(args.edge_score_key)
                    row["candidate_rows"] = _rel(candidate_path)
                    row["edge_rows"] = _rel(edge_path)
                    row["witness_rows"] = _rel(witness_path)
                    chunk_rows.append(row)
                cannotlink_specs = [
                    ("OC9_oracle_all_candidates_cannotlink_diagnostic", nodes),
                    ("OC10_oracle_nonshared_cannotlink_diagnostic", nodes - shared_nodes),
                ]
                for variant, oracle_nodes in cannotlink_specs:
                    if not oracle_nodes:
                        continue
                    oracle_mapping, oracle_diag = _oracle_cannotlink_mapping(frame_data=frame_data, nodes=oracle_nodes)
                    diag = {
                        "duplicate_frame_mask_conflict_pairs": 0,
                        "duplicate_frame_mask_conflict_rate": 0.0,
                        "same_frame_cannot_link_violation_count": _same_frame_violation_count(oracle_mapping),
                        "shared_mask_count": int(len(shared_nodes)),
                        "reject_mask_count": int(len(nodes - oracle_nodes)),
                        "unknown_mask_count": int(len(nodes - set(oracle_mapping))),
                        "underseg_false_bridge_rate": float(len(set(oracle_mapping) & shared_nodes) / max(1, len(oracle_mapping))),
                        "selected_capsule_count": int(len(set(oracle_mapping.values()))),
                    }
                    diag.update(oracle_diag)
                    row = _row_from_mapping(
                        scene=scene,
                        chunk_id=chunk_id,
                        variant=variant,
                        frame_ids=frame_ids,
                        chunk=chunk,
                        frame_data=frame_data,
                        mapping=oracle_mapping,
                        diag=diag,
                        pipeline_root=pipeline_root,
                        uses_gt_for_prediction=True,
                        forbidden_for_method_table=True,
                    )
                    row.update(diag)
                    row["selected_capsule_count"] = int(len(set(oracle_mapping.values())))
                    row["capsule_temporal_span_mean"] = _capsule_temporal_span(oracle_mapping)
                    row["runtime_sec"] = float(time.time() - t0)
                    row["edge_score_key"] = str(args.edge_score_key)
                    row["candidate_rows"] = _rel(candidate_path)
                    row["edge_rows"] = _rel(edge_path)
                    row["witness_rows"] = _rel(witness_path)
                    chunk_rows.append(row)
    metric_variants = list(variants) + (ORACLE_VARIANTS if bool(args.include_oracle) else [])
    metric_rows = [_summarize_capsule_variant(chunk_rows, variant) for variant in metric_variants]
    method_rows = [row for row in metric_rows if not bool(row.get("uses_gt_for_prediction"))]
    best = max(method_rows, key=lambda row: float(row.get("local_SF50") or 0.0), default={})
    baseline_closure = json.loads(_rooted(args.true_closure_summary).read_text(encoding="utf-8"))
    v69_c10 = baseline_closure.get("v69r2_C10_reproduction") or {}
    c10_sf50 = _float_or_none(v69_c10.get("single_anchor_SF50"))
    c10_gt = _float_or_none(v69_c10.get("single_anchor_GT_best_IoU_mean"))
    c10_single = _float_or_none(v69_c10.get("single_anchor_single_frame_rate"))
    sf50 = _float_or_none(best.get("local_SF50"))
    ap50 = _float_or_none(best.get("local_AP50"))
    gt_best = _float_or_none(best.get("GT_best_IoU_mean"))
    single = _float_or_none(best.get("single_frame_object_rate"))
    frag10 = _float_or_none(best.get("fragments_per_GT@0.10"))
    over10 = _float_or_none(best.get("GT_per_pred@0.10"))
    dup = _float_or_none(best.get("duplicate_frame_mask_conflict_rate"))
    violations = int(float(best.get("same_frame_violation_count") or 0)) if best else 0
    partial_gate = {
        "local_SF50_ge_v69r2_C10_plus_0p15": sf50 is not None and c10_sf50 is not None and sf50 >= c10_sf50 + 0.15,
        "GT_best_IoU_ge_v69r2_C10_plus_0p08": gt_best is not None and c10_gt is not None and gt_best >= c10_gt + 0.08,
        "single_frame_object_rate_le_v69r2_C10_minus_0p20": single is not None and c10_single is not None and single <= c10_single - 0.20,
        "same_frame_violation_count_eq_0": violations == 0,
    }
    partial_gate["pass"] = all(bool(value) for value in partial_gate.values())
    full_gate = {
        "local_SF50_ge_0p30": sf50 is not None and sf50 >= 0.30,
        "local_AP50_ge_0p05": ap50 is not None and ap50 >= 0.05,
        "GT_best_IoU_ge_0p25": gt_best is not None and gt_best >= 0.25,
        "single_frame_object_rate_le_0p50": single is not None and single <= 0.50,
        "fragments_per_GT_0p10_le_2p0": frag10 is not None and frag10 <= 2.0,
        "GT_per_pred_0p10_le_1p5": over10 is not None and over10 <= 1.5,
        "duplicate_frame_mask_conflict_rate_le_0p02": dup is not None and dup <= 0.02,
        "same_frame_violation_count_eq_0": violations == 0,
    }
    full_gate["pass"] = all(bool(value) for value in full_gate.values())
    if full_gate["pass"]:
        decision = "GO_OBJECT_CAPSULE_LOCAL_REPAIR"
    elif partial_gate["pass"]:
        decision = "PARTIAL_OBJECT_CAPSULE_SIGNAL"
    elif single is not None and single > 0.50:
        decision = "NO_GO_OBJECT_CAPSULE_OVERFRAGMENT"
    elif over10 is not None and over10 > 1.5:
        decision = "NO_GO_OBJECT_CAPSULE_OVERMERGE"
    else:
        decision = "NO_GO_OBJECT_CAPSULE_LOCAL_EVIDENCE"
    summary = {
        "phase": "v70_object_capsule",
        "decision": decision,
        "partial_gate": partial_gate,
        "full_gate": full_gate,
        "best_capsule_variant": best,
        "baseline_v69r2_C10": v69_c10,
        "candidate_rows": _rel(candidate_path),
        "edge_rows": _rel(edge_path),
        "edge_score_key": str(args.edge_score_key),
        "witness_rows": _rel(witness_path),
        "true_closure_summary": _rel(_rooted(args.true_closure_summary)),
        "scenes": sorted(scenes),
        "pipeline_roots": pipeline_roots,
        "variants": variants,
        "include_oracle": bool(args.include_oracle),
        "rows": {
            "capsule_summary_json": _rel(output_root / "capsule_summary.json"),
            "capsule_rows_csv": _rel(output_root / "capsule_rows.csv"),
            "capsule_metric_rows_csv": _rel(output_root / "capsule_metric_rows.csv"),
            "capsule_chunk_rows_csv": _rel(output_root / "capsule_chunk_rows.csv"),
            "missing_input_rows_csv": _rel(output_root / "missing_input_rows.csv"),
        },
        "notes": [
            "OC variants use v68 frozen-appearance edge evidence as mask evidence and v70 carrier witness only as reward/veto.",
            "GT labels are used only by the evaluator and optional oracle diagnostics, never by non-oracle method variants.",
            "When enabled, OC7/OC8 oracle rows are diagnostic-only, GT-derived, and forbidden for method tables.",
            "Underseg/shared masks are excluded from core-bridge variants OC4-OC6 and used only as non-core merge evidence in OC11/OC12.",
        ],
    }
    _write_csv(output_root / "capsule_rows.csv", capsule_rows)
    _write_csv(output_root / "capsule_metric_rows.csv", metric_rows)
    _write_csv(output_root / "capsule_chunk_rows.csv", chunk_rows)
    _write_csv(output_root / "missing_input_rows.csv", missing_rows)
    _write_json(output_root / "capsule_summary.json", summary)
    sha_rows = []
    for path in [
        output_root / "capsule_summary.json",
        output_root / "capsule_rows.csv",
        output_root / "capsule_metric_rows.csv",
        output_root / "capsule_chunk_rows.csv",
        output_root / "missing_input_rows.csv",
    ]:
        sha_rows.append({"path": _rel(path), "sha256": _sha256(path), "bytes": int(path.stat().st_size)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream4D v70 object capsule assembly repair.")
    parser.add_argument("--output-root", default="outputs/audit/v70_object_capsules")
    parser.add_argument("--candidate-rows", default="outputs/audit/v68_candidate_bank/candidate_mask_rows.csv")
    parser.add_argument("--edge-rows", default="outputs/audit/v68_edge_audit_dinov2/edge_rows.csv")
    parser.add_argument("--edge-score-key", default="score_combined_frozen_appearance")
    parser.add_argument("--witness-rows", default="outputs/audit/v70_carrier_witness/witness_rows.csv")
    parser.add_argument("--true-closure-summary", default="outputs/audit/v70_true_material_closure/closure_summary.json")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--variants", default=",".join(CAPSULE_VARIANTS))
    parser.add_argument("--include-oracle", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
