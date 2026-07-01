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

from stream4d_native.v67_local_baselines import _oracle_majority_mapping_bundle  # noqa: E402
from tools.run_v65_scene_multiview_ap import _sha256, _write_csv, _write_json  # noqa: E402
from tools.run_v66_local_chunk_eval import (  # noqa: E402
    _chunk_rows,
    _evaluate_frame_data,
    _float_or_none,
    _frame_data,
    _load_csv_rows,
    _mean,
    _rel,
    _score_free,
)
from tools.run_v66_scene_mv_ap_probe5 import (  # noqa: E402
    DEFAULT_SCENES,
    _discover_pipeline_root,
    _mask_dir_from_pipeline,
    _parse_csv_list,
)
from stream4d.scannet_stream import ScanNetStream  # noqa: E402


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _node(scene: str, frame_id: Any, mask_id: Any) -> tuple[int, int]:
    return (int(float(frame_id)), int(float(mask_id)))


def _node_from_token(token: str) -> tuple[int, int] | None:
    parts = str(token).split(":")
    if len(parts) != 3:
        return None
    try:
        return int(parts[1]), int(parts[2])
    except ValueError:
        return None


class DSU:
    def __init__(self, nodes: list[tuple[int, int]]) -> None:
        self.parent = {node: node for node in nodes}

    def find(self, node: tuple[int, int]) -> tuple[int, int]:
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, left: tuple[int, int], right: tuple[int, int]) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def _object_frame_stats(
    *,
    frame_data: list[dict[str, Any]],
    mapping: dict[tuple[int, int], int],
) -> dict[str, Any]:
    object_frames: dict[int, set[int]] = defaultdict(set)
    support_pair_count = 0
    for item in frame_data:
        frame_id = int(item["frame_id"])
        mask = item["mask"]
        if mask is None:
            continue
        for mask_id in np.unique(mask):
            mask_id_i = int(mask_id)
            if mask_id_i <= 0:
                continue
            object_id = int(mapping.get((frame_id, mask_id_i), 0))
            if object_id <= 0:
                continue
            object_frames[object_id].add(frame_id)
            support_pair_count += 1
    object_count = len(object_frames)
    single_frame_count = sum(1 for frames in object_frames.values() if len(frames) == 1)
    return {
        "support_pair_count": int(support_pair_count),
        "mean_masks_per_object": float(support_pair_count / max(1, object_count)),
        "single_frame_object_rate": float(single_frame_count / max(1, object_count)),
    }


def _frag_overmerge_at(iou: np.ndarray, threshold: float) -> tuple[float | None, float | None]:
    if iou.size == 0:
        return None, None
    frag = [int(np.count_nonzero(iou[:, col] >= threshold)) for col in range(iou.shape[1])] if iou.shape[1] else []
    over = [int(np.count_nonzero(iou[row, :] >= threshold)) for row in range(iou.shape[0])] if iou.shape[0] else []
    return (
        float(np.mean(frag)) if frag else None,
        float(np.mean(over)) if over else None,
    )


def _same_frame_violation_count(mapping: dict[tuple[int, int], int]) -> int:
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for (frame_id, _mask_id), object_id in mapping.items():
        counts[(int(object_id), int(frame_id))] += 1
    return int(sum(max(0, count - 1) for count in counts.values()))


def _mapping_from_groups(groups: dict[tuple[int, int], tuple[int, int]]) -> dict[tuple[int, int], int]:
    root_to_id: dict[tuple[int, int], int] = {}
    mapping: dict[tuple[int, int], int] = {}
    for node in sorted(groups):
        root = groups[node]
        if root not in root_to_id:
            root_to_id[root] = len(root_to_id) + 1
        mapping[node] = root_to_id[root]
    return mapping


def _identity_mapping(nodes: set[tuple[int, int]], *, reject: set[tuple[int, int]] | None = None) -> dict[tuple[int, int], int]:
    reject = reject or set()
    return {node: idx + 1 for idx, node in enumerate(sorted(nodes - reject))}


def _can_merge(
    left_members: set[tuple[int, int]],
    right_members: set[tuple[int, int]],
) -> bool:
    frames = {frame_id for frame_id, _mask_id in left_members}
    return not any(frame_id in frames for frame_id, _mask_id in right_members)


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
        score = float(edge.get("score") or 0.0)
        if score < threshold:
            continue
        left = edge["left"]
        right = edge["right"]
        if left not in members or right not in members:
            continue
        if int(left[0]) == int(right[0]):
            continue
        root_left = dsu.find(left)
        root_right = dsu.find(right)
        if root_left == root_right:
            continue
        if not _can_merge(members[root_left], members[root_right]):
            continue
        dsu.union(root_left, root_right)
        new_root = dsu.find(root_left)
        old_root = root_right if new_root == root_left else root_left
        members[new_root] = members.pop(root_left, {root_left}) | members.pop(root_right, {root_right})
        members.pop(old_root, None)
    groups = {node: dsu.find(node) for node in active}
    return _mapping_from_groups(groups)


def _seed_absorb_mapping(
    nodes: set[tuple[int, int]],
    edges: list[dict[str, Any]],
    *,
    seed_threshold: float,
    absorb_threshold: float,
    reject: set[tuple[int, int]] | None = None,
) -> dict[tuple[int, int], int]:
    reject = reject or set()
    mapping = _edge_cc_mapping(nodes, edges, threshold=seed_threshold, reject=reject)
    object_frames: dict[int, set[int]] = defaultdict(set)
    object_members: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for node, object_id in mapping.items():
        object_frames[int(object_id)].add(int(node[0]))
        object_members[int(object_id)].add(node)
    unmapped = sorted((nodes - reject) - set(mapping))
    by_node: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        by_node[edge["left"]].append(edge)
        by_node[edge["right"]].append(edge)
    for node in unmapped:
        best: tuple[float, int] | None = None
        for edge in by_node.get(node, []):
            score = float(edge.get("score") or 0.0)
            if score < absorb_threshold:
                continue
            other = edge["right"] if edge["left"] == node else edge["left"]
            object_id = int(mapping.get(other, 0))
            if object_id <= 0 or int(node[0]) in object_frames.get(object_id, set()):
                continue
            if best is None or score > best[0]:
                best = (score, object_id)
        if best is not None:
            object_id = best[1]
            mapping[node] = object_id
            object_frames[object_id].add(int(node[0]))
            object_members[object_id].add(node)
    return mapping


def _topk_edge_mapping(
    nodes: set[tuple[int, int]],
    edges: list[dict[str, Any]],
    *,
    topk: int,
    min_score: float,
) -> dict[tuple[int, int], int]:
    by_node: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        if edge["left"] in nodes and edge["right"] in nodes and int(edge["left"][0]) != int(edge["right"][0]):
            by_node[edge["left"]].append(edge)
            by_node[edge["right"]].append(edge)
    selected: dict[tuple[int, int], dict[str, Any]] = {}
    for node, node_edges in by_node.items():
        for edge in sorted(node_edges, key=lambda item: float(item.get("score") or 0.0), reverse=True)[: int(topk)]:
            if float(edge.get("score") or 0.0) >= float(min_score):
                key = tuple(sorted([edge["left"], edge["right"]]))
                selected[key] = edge
    return _edge_cc_mapping(nodes, list(selected.values()), threshold=min_score)


def _row_from_mapping(
    *,
    scene: str,
    chunk_id: str,
    variant: str,
    frame_ids: list[int],
    chunk: dict[str, Any],
    frame_data: list[dict[str, Any]],
    mapping: dict[tuple[int, int], int],
    diag: dict[str, Any],
    pipeline_root: Path,
    uses_gt_for_prediction: bool,
    forbidden_for_method_table: bool,
) -> dict[str, Any]:
    summary, iou, _pred_ids, _gt_ids = _evaluate_frame_data(
        frame_data=frame_data,
        variant=variant,
        mapping=mapping,
        raw_per_frame_masks=False,
    )
    frag10, over10 = _frag_overmerge_at(iou, 0.10)
    frag25, over25 = _frag_overmerge_at(iou, 0.25)
    object_stats = _object_frame_stats(frame_data=frame_data, mapping=mapping)
    return {
        "scene_id": scene,
        "chunk_id": chunk_id,
        "variant": variant,
        "chunk_frame_count": int(len(frame_ids)),
        "frame_min": int(frame_ids[0]),
        "frame_max": int(frame_ids[-1]),
        "mask_count": int(float(chunk.get("mask_count") or 0)),
        "pred_object_count": summary.get("evaluated_pred_count"),
        "gt_object_count": summary.get("evaluated_gt_count"),
        "local_object_count": summary.get("evaluated_pred_count"),
        "local_AP": summary.get("ap"),
        "local_AP50": summary.get("ap50"),
        "local_AP25": summary.get("ap25"),
        "local_SF25": (summary.get("score_free_match_at_025") or {}).get("recall"),
        "local_SF50": (summary.get("score_free_match_at_050") or {}).get("recall"),
        "local_score_free_match50_recall": _score_free(summary),
        "GT_best_IoU_mean": summary.get("gt_best_iou_mean"),
        "local_GT_best_IoU_mean": summary.get("gt_best_iou_mean"),
        "pred_best_IoU_median": summary.get("pred_best_iou_median"),
        "local_pred_best_IoU_median": summary.get("pred_best_iou_median"),
        "mean_fragments_per_GT@0.10": frag10,
        "mean_GT_per_pred@0.10": over10,
        "fragmentation_mean": frag25,
        "local_fragmentation_mean": frag25,
        "overmerge_mean": over25,
        "local_overmerge_mean": over25,
        "duplicate_frame_mask_conflict_rate": diag.get("duplicate_frame_mask_conflict_rate", 0.0),
        "local_duplicate_frame_mask_conflict_rate": diag.get("duplicate_frame_mask_conflict_rate", 0.0),
        "duplicate_frame_mask_conflict_pairs": diag.get("duplicate_frame_mask_conflict_pairs", 0),
        "same_frame_cannot_link_violation_count": diag.get("same_frame_cannot_link_violation_count", 0),
        "support_pair_count": object_stats.get("support_pair_count", ""),
        "selected_mask_count": int(len(mapping)),
        "mean_masks_per_object": object_stats["mean_masks_per_object"],
        "single_frame_object_rate": object_stats["single_frame_object_rate"],
        "shared_mask_count": diag.get("shared_mask_count", 0),
        "reject_mask_count": diag.get("reject_mask_count", 0),
        "unknown_mask_count": diag.get("unknown_mask_count", 0),
        "uses_gt_for_prediction": bool(uses_gt_for_prediction),
        "forbidden_for_method_table": bool(forbidden_for_method_table),
        "diagnostic_only": True,
        "pipeline_root": _rel(pipeline_root),
    }


def _summarize_variant_all(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    subset = [row for row in rows if row["variant"] == variant]
    same_frame_sum = int(sum(int(float(row.get("same_frame_cannot_link_violation_count") or 0)) for row in subset))
    return {
        "variant": variant,
        "chunk_count": len(subset),
        "scene_count": len({row["scene_id"] for row in subset}),
        "local_AP50_mean": _mean([_float_or_none(row.get("local_AP50")) for row in subset]),
        "local_AP25_mean": _mean([_float_or_none(row.get("local_AP25")) for row in subset]),
        "local_SF50_mean": _mean([_float_or_none(row.get("local_SF50")) for row in subset]),
        "local_score_free_match50_recall_mean": _mean([_float_or_none(row.get("local_score_free_match50_recall")) for row in subset]),
        "local_GT_best_IoU_mean_mean": _mean([_float_or_none(row.get("local_GT_best_IoU_mean")) for row in subset]),
        "local_pred_best_IoU_median_mean": _mean([_float_or_none(row.get("local_pred_best_IoU_median")) for row in subset]),
        "local_duplicate_frame_mask_conflict_rate_mean": _mean([_float_or_none(row.get("local_duplicate_frame_mask_conflict_rate")) for row in subset]),
        "local_object_count_mean": _mean([_float_or_none(row.get("local_object_count")) for row in subset]),
        "local_gt_count_mean": _mean([_float_or_none(row.get("local_gt_count")) for row in subset]),
        "mean_masks_per_object_mean": _mean([_float_or_none(row.get("mean_masks_per_object")) for row in subset]),
        "single_frame_object_rate_mean": _mean([_float_or_none(row.get("single_frame_object_rate")) for row in subset]),
        "mean_fragments_per_GT@0.10_mean": _mean([_float_or_none(row.get("mean_fragments_per_GT@0.10")) for row in subset]),
        "mean_GT_per_pred@0.10_mean": _mean([_float_or_none(row.get("mean_GT_per_pred@0.10")) for row in subset]),
        "same_frame_cannot_link_violation_count_sum": same_frame_sum,
        "shared_mask_count_mean": _mean([_float_or_none(row.get("shared_mask_count")) for row in subset]),
        "reject_mask_count_mean": _mean([_float_or_none(row.get("reject_mask_count")) for row in subset]),
        "unknown_mask_count_mean": _mean([_float_or_none(row.get("unknown_mask_count")) for row in subset]),
        "uses_gt_for_prediction": any(bool(row.get("uses_gt_for_prediction")) for row in subset),
        "forbidden_for_method_table": any(bool(row.get("forbidden_for_method_table")) for row in subset),
        "diagnostic_only": True,
    }


def _load_candidates(path: Path, scenes: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    scene_set = set(scenes)
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in _load_csv_rows(path):
        scene = str(row.get("scene_id"))
        if scene not in scene_set or not _parse_bool(row.get("representative_available")):
            continue
        chunk_id = str(row.get("chunk_id"))
        node = _node(scene, row.get("frame_id"), row.get("mask_id"))
        out[chunk_id][f"{node[0]}:{node[1]}"] = {
            "node": node,
            "shared": _parse_bool(row.get("shared_support_only")),
            "underseg": _parse_bool(row.get("underseg_risk")),
            "small": _parse_bool(row.get("small_mask_risk")),
            "large": _parse_bool(row.get("large_mask_risk")),
        }
    return out


def _load_edges(path: Path, scenes: list[str], score_key: str) -> dict[str, list[dict[str, Any]]]:
    scene_set = set(scenes)
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _load_csv_rows(path):
        scene = str(row.get("scene_id"))
        if scene not in scene_set:
            continue
        left = _node_from_token(str(row.get("node_i")))
        right = _node_from_token(str(row.get("node_j")))
        if left is None or right is None:
            continue
        out[str(row.get("chunk_id"))].append(
            {
                "left": left,
                "right": right,
                "score": float(row.get(score_key) or 0.0),
                "same_frame": _parse_bool(row.get("same_frame")),
                "hard_negative": _parse_bool(row.get("hard_negative_candidate")),
            }
        )
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    scenes = _parse_csv_list(args.scenes)
    candidate_path = Path(args.candidate_rows)
    edge_path = Path(args.edge_rows)
    if not candidate_path.is_absolute():
        candidate_path = ROOT / candidate_path
    if not edge_path.is_absolute():
        edge_path = ROOT / edge_path
    candidate_by_chunk = _load_candidates(candidate_path, scenes)
    edges_by_chunk = _load_edges(edge_path, scenes, str(args.edge_score_key))

    rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    pipeline_roots: dict[str, str] = {}
    for scene in scenes:
        print(f"[v68-local-solver] scene={scene}", file=sys.stderr, flush=True)
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
            if not frame_ids:
                continue
            frame_data = _frame_data(scene, frame_ids, mask_dir)
            chunk_candidates = candidate_by_chunk.get(chunk_id, {})
            nodes = {meta["node"] for meta in chunk_candidates.values()}
            shared = {meta["node"] for meta in chunk_candidates.values() if meta.get("shared") or meta.get("underseg")}
            edges = edges_by_chunk.get(chunk_id, [])
            if not nodes:
                missing_rows.append({"scene_id": scene, "chunk_id": chunk_id, "missing": "v68_candidate_nodes"})
                continue
            mappings: list[tuple[str, dict[tuple[int, int], int], bool, bool, dict[str, Any]]] = []
            mappings.append(("S1_seed_only_representative_identity", _identity_mapping(nodes), False, False, {}))
            mappings.append(("S2_dino_edge_cc_t090", _edge_cc_mapping(nodes, edges, threshold=0.90), False, False, {}))
            mappings.append(("S3_dino_edge_cc_t080", _edge_cc_mapping(nodes, edges, threshold=0.80), False, False, {}))
            mappings.append(("S4_seed085_absorb070", _seed_absorb_mapping(nodes, edges, seed_threshold=0.85, absorb_threshold=0.70), False, False, {}))
            mappings.append(("S5_signed_conflict_cc_t075", _edge_cc_mapping(nodes, edges, threshold=0.75), False, False, {}))
            mappings.append(("S6_signed_conflict_cc_t065", _edge_cc_mapping(nodes, edges, threshold=0.65), False, False, {}))
            mappings.append(("S8_signed_conflict_cc_t055", _edge_cc_mapping(nodes, edges, threshold=0.55), False, False, {}))
            mappings.append(("S12_signed_conflict_cc_t045", _edge_cc_mapping(nodes, edges, threshold=0.45), False, False, {}))
            mappings.append(("S13_signed_conflict_cc_t035", _edge_cc_mapping(nodes, edges, threshold=0.35), False, False, {}))
            mappings.append(("S14_top1_edge_track_min025", _topk_edge_mapping(nodes, edges, topk=1, min_score=0.25), False, False, {}))
            mappings.append(("S15_top2_edge_track_min025", _topk_edge_mapping(nodes, edges, topk=2, min_score=0.25), False, False, {}))
            mappings.append(("S10_seed070_absorb050", _seed_absorb_mapping(nodes, edges, seed_threshold=0.70, absorb_threshold=0.50), False, False, {}))
            mappings.append(("S11_seed060_absorb045", _seed_absorb_mapping(nodes, edges, seed_threshold=0.60, absorb_threshold=0.45), False, False, {}))
            s7_mapping = _seed_absorb_mapping(nodes, edges, seed_threshold=0.80, absorb_threshold=0.88, reject=shared)
            mappings.append(("S7_underseg_shared_seed080_absorb088", s7_mapping, False, False, {"shared_mask_count": len(shared), "reject_mask_count": len(shared)}))
            oracle_mapping, oracle_diag = _oracle_majority_mapping_bundle(
                frame_data=frame_data,
                selected_pairs=set(),
                representative_pairs=nodes,
            )["representative"]
            mappings.append(("S9_oracle_representative_majority_GT", oracle_mapping, True, True, oracle_diag))
            for variant, mapping, uses_gt, forbidden, extra_diag in mappings:
                violations = _same_frame_violation_count(mapping)
                diag = {
                    "duplicate_frame_mask_conflict_pairs": 0,
                    "duplicate_frame_mask_conflict_rate": 0.0,
                    "same_frame_cannot_link_violation_count": violations,
                    "shared_mask_count": extra_diag.get("shared_mask_count", 0),
                    "reject_mask_count": extra_diag.get("reject_mask_count", 0),
                    "unknown_mask_count": max(0, len(nodes) - len(mapping)),
                }
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
                    uses_gt_for_prediction=uses_gt,
                    forbidden_for_method_table=forbidden,
                )
                row["runtime_sec"] = float(time.time() - t0)
                row["edge_score_key"] = str(args.edge_score_key)
                rows.append(row)
    variant_summary_rows = [_summarize_variant_all(rows, variant) for variant in sorted({row["variant"] for row in rows})]
    non_oracle = [row for row in variant_summary_rows if not bool(row.get("uses_gt_for_prediction"))]
    best = max(non_oracle, key=lambda row: float(row.get("local_score_free_match50_recall_mean") or 0.0), default={})
    sf50 = _float_or_none(best.get("local_score_free_match50_recall_mean"))
    ap50 = _float_or_none(best.get("local_AP50_mean"))
    gt_best = _float_or_none(best.get("local_GT_best_IoU_mean_mean"))
    dup = _float_or_none(best.get("local_duplicate_frame_mask_conflict_rate_mean"))
    single = _float_or_none(best.get("single_frame_object_rate_mean"))
    frag10 = _float_or_none(best.get("mean_fragments_per_GT@0.10_mean"))
    over10 = _float_or_none(best.get("mean_GT_per_pred@0.10_mean"))
    violations = int(float(best.get("same_frame_cannot_link_violation_count_sum") or 0)) if best else 0
    gate = {
        "all_pipeline_roots_available": len(pipeline_roots) == len(scenes),
        "best_S_local_SF50_ge_0p30": sf50 is not None and sf50 >= 0.30,
        "best_S_local_AP50_ge_0p05": ap50 is not None and ap50 >= 0.05,
        "best_S_GT_best_IoU_ge_0p25": gt_best is not None and gt_best >= 0.25,
        "best_S_duplicate_rate_le_0p02": dup is not None and dup <= 0.02,
        "best_S_same_frame_cannot_link_violation_count_eq_0": violations == 0,
        "best_S_single_frame_object_rate_le_0p50": single is not None and single <= 0.50,
        "best_S_fragments_per_GT_0p10_le_2p0": frag10 is not None and frag10 <= 2.0,
        "best_S_GT_per_pred_0p10_le_1p5": over10 is not None and over10 <= 1.5,
    }
    gate["pass"] = bool(all(gate.values()))
    if gate["pass"]:
        decision = "PASS_LOCAL_GRAPH_SOLVER"
    elif single is not None and single > 0.50:
        decision = "NO_GO_OVERFRAGMENT"
    elif over10 is not None and over10 > 1.5:
        decision = "NO_GO_OVERMERGE"
    else:
        decision = "NO_GO_LOCAL_SOLVER"
    _write_csv(output_root / "local_solver_rows.csv", rows)
    _write_csv(output_root / "local_variant_summary_rows.csv", variant_summary_rows)
    _write_csv(output_root / "missing_input_rows.csv", missing_rows)
    summary = {
        "phase": "v68_local_graph_solver",
        "decision": decision,
        "diagnostic_only": True,
        "uses_gt_for_prediction": False,
        "edge_rows": _rel(edge_path),
        "candidate_rows": _rel(candidate_path),
        "edge_score_key": str(args.edge_score_key),
        "scenes": scenes,
        "pipeline_roots": pipeline_roots,
        "gate": gate,
        "best_S": best,
        "rows": {
            "local_solver_rows_csv": _rel(output_root / "local_solver_rows.csv"),
            "local_variant_summary_rows_csv": _rel(output_root / "local_variant_summary_rows.csv"),
            "missing_input_rows_csv": _rel(output_root / "missing_input_rows.csv"),
        },
        "notes": [
            "S1-S7 use v68 candidate rows and calibrated DINO edge scores only; GT labels are used only by the evaluator.",
            "All non-oracle variants enforce same-frame cannot-link by refusing merges that would place two masks from one frame in one object.",
            "S9 oracle representative majority GT is diagnostic-only and forbidden for method tables.",
        ],
    }
    _write_json(output_root / "local_solver_summary.json", summary)
    sha_rows = []
    for path in [
        output_root / "local_solver_summary.json",
        output_root / "local_solver_rows.csv",
        output_root / "local_variant_summary_rows.csv",
        output_root / "missing_input_rows.csv",
    ]:
        if path.exists():
            sha_rows.append({"path": _rel(path), "sha256": _sha256(path), "bytes": int(path.stat().st_size)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stream4D v68 seeded local graph solver.")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--candidate-rows", default="outputs/audit/v68_candidate_bank/candidate_mask_rows.csv")
    parser.add_argument("--edge-rows", default="outputs/audit/v68_edge_audit_dinov2/edge_rows.csv")
    parser.add_argument("--edge-score-key", default="score_combined_frozen_appearance")
    parser.add_argument("--output-root", default="outputs/audit/v68_local_graph_solver")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
