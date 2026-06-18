from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tools.run_v26_object_quality_diagnostics import _json_safe, _read_split
from tools.run_v28_proposal_oracle import _cluster_metrics
from tools.run_v37_temporal_curriculum import (
    UnionFind,
    _collect_observations,
    _component_stats,
    _components_chain_then_closure,
    _filter_edges_by_rgb,
    _filtered_edges,
    _frame_rank_map,
    _gate,
    _labels_for_components,
    _load_gt,
    _load_masks,
    _load_tubes,
    _region_diagnostics,
    _rgb_similarity,
    _safe_div,
    _sample_all_pairs,
    _sample_same_frame_pairs,
    _select_best_stage,
    _shuffle_supports,
    _split_components_by_rgb,
    _support_pair_counts,
)


@dataclass
class SceneState:
    scene: str
    nodes: list[Any]
    frame_rank: dict[int, int]
    gt_labels: dict[int, int]
    support_by_region: dict[int, Counter[int]]
    support_by_tube: dict[int, Counter[int]]
    observation_count_by_tube: dict[int, int]
    diagnostics: dict[int, dict[str, Any]]
    components: list[list[int]]
    adaptive_fraction: float
    support_density: float
    base_edge_info: dict[str, Any]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _component_descriptors(
    state: SceneState,
    components: list[list[int]],
) -> tuple[list[Counter[int]], list[set[int]], list[list[float] | None]]:
    supports: list[Counter[int]] = []
    frames: list[set[int]] = []
    rgbs: list[list[float] | None] = []
    for component in components:
        support_counter: Counter[int] = Counter()
        frame_set: set[int] = set()
        rgb_values = []
        for node_id in component:
            support_counter.update(state.support_by_region.get(int(node_id), Counter()))
            frame_id = int(state.nodes[int(node_id)].frame_id)
            frame_set.add(int(state.frame_rank.get(frame_id, frame_id)))
            rgb = state.diagnostics.get(int(node_id), {}).get("rgb_mean")
            if rgb is not None:
                rgb_values.append(rgb)
        supports.append(support_counter)
        frames.append(frame_set)
        rgbs.append(np.mean(np.asarray(rgb_values, dtype=np.float32), axis=0).tolist() if rgb_values else None)
    return supports, frames, rgbs


def _merge_components(
    state: SceneState,
    components: list[list[int]],
    *,
    min_shared_tubes: int,
    min_jaccard: float,
    min_rgb_similarity: float,
    max_frame_gap: int | None,
    allow_rgb_fallback: bool = False,
    max_rgb_fallback_per_component: int = 0,
) -> tuple[list[list[int]], dict[str, Any]]:
    supports, frames, rgbs = _component_descriptors(state, components)
    tube_to_components: dict[int, list[int]] = defaultdict(list)
    for comp_id, counter in enumerate(supports):
        for tube_id in counter:
            tube_to_components[int(tube_id)].append(int(comp_id))

    pair_shared: Counter[tuple[int, int]] = Counter()
    for comp_ids in tube_to_components.values():
        unique = sorted(set(comp_ids))
        for pos, left in enumerate(unique):
            for right in unique[pos + 1 :]:
                pair_shared[(int(left), int(right))] += 1

    candidate_pairs = 0
    rejected_same_frame = 0
    rejected_gap = 0
    rejected_jaccard = 0
    rejected_rgb = 0
    rgb_fallback_candidate_pairs = 0
    rgb_fallback_edges = 0
    edges = []
    for (left, right), shared in pair_shared.items():
        if frames[left] & frames[right]:
            rejected_same_frame += 1
            continue
        if max_frame_gap is not None:
            gap = min(abs(int(a) - int(b)) for a in frames[left] for b in frames[right]) if frames[left] and frames[right] else 10**9
            if gap > int(max_frame_gap):
                rejected_gap += 1
                continue
        if int(shared) < int(min_shared_tubes):
            continue
        union = len(set(supports[left]) | set(supports[right]))
        jaccard = float(shared / max(union, 1))
        if jaccard < float(min_jaccard):
            rejected_jaccard += 1
            continue
        candidate_pairs += 1
        if min_rgb_similarity > 0.0:
            sim = _rgb_similarity(rgbs[left], rgbs[right])
            if sim is None or float(sim) < float(min_rgb_similarity):
                rejected_rgb += 1
                continue
        edges.append((int(shared), float(jaccard), int(left), int(right)))

    if allow_rgb_fallback and min_rgb_similarity > 0.0 and max_frame_gap is not None:
        fallback_by_component: dict[int, list[tuple[float, int, int]]] = defaultdict(list)
        rank_to_components: dict[int, set[int]] = defaultdict(set)
        for comp_id, frame_set in enumerate(frames):
            for rank in frame_set:
                rank_to_components[int(rank)].add(int(comp_id))
        rgb_array = np.zeros((len(components), 3), dtype=np.float32)
        rgb_valid = np.zeros((len(components),), dtype=bool)
        for comp_id, rgb in enumerate(rgbs):
            if rgb is not None:
                rgb_array[int(comp_id)] = np.asarray(rgb, dtype=np.float32)
                rgb_valid[int(comp_id)] = True
        for left in range(len(components)):
            if not bool(rgb_valid[left]):
                continue
            candidate_rights: set[int] = set()
            for rank in frames[left]:
                for target_rank in range(int(rank) - int(max_frame_gap), int(rank) + int(max_frame_gap) + 1):
                    candidate_rights.update(rank_to_components.get(int(target_rank), set()))
            filtered = []
            for right in candidate_rights:
                if right <= left or not bool(rgb_valid[right]):
                    continue
                if pair_shared.get((left, right), 0) > 0:
                    continue
                if frames[left] & frames[right]:
                    continue
                if not frames[left] or not frames[right]:
                    continue
                gap = min(abs(int(a) - int(b)) for a in frames[left] for b in frames[right])
                if gap <= int(max_frame_gap):
                    filtered.append(int(right))
            if not filtered:
                continue
            cand = np.asarray(filtered, dtype=np.int64)
            dist = np.linalg.norm((rgb_array[cand] - rgb_array[left][None, :]) / 255.0, axis=1)
            sim_values = np.maximum(0.0, 1.0 - dist / np.sqrt(3.0))
            keep_idx = np.flatnonzero(sim_values >= float(min_rgb_similarity))
            if keep_idx.size == 0:
                continue
            order = keep_idx[np.argsort(sim_values[keep_idx])[::-1]]
            cap = int(max_rgb_fallback_per_component)
            if cap > 0:
                order = order[:cap]
            for idx in order.tolist():
                right = int(cand[int(idx)])
                sim = float(sim_values[int(idx)])
                rgb_fallback_candidate_pairs += 1
                fallback_by_component[left].append((sim, left, right))
                fallback_by_component[right].append((sim, left, right))
        selected: set[tuple[int, int]] = set()
        for comp_id, items in fallback_by_component.items():
            cap = int(max_rgb_fallback_per_component)
            kept = sorted(items, reverse=True)[:cap] if cap > 0 else sorted(items, reverse=True)
            for _sim, left, right in kept:
                selected.add((min(left, right), max(left, right)))
        for left, right in sorted(selected):
            sim = _rgb_similarity(rgbs[left], rgbs[right])
            if sim is None:
                continue
            edges.append((0, float(sim), int(left), int(right)))
            rgb_fallback_edges += 1

    uf = UnionFind(len(components))
    root_frames: dict[int, set[int]] = {idx: set(frame_set) for idx, frame_set in enumerate(frames)}
    accepted = 0
    rejected_dynamic_same_frame = 0
    for _shared, _jaccard, left, right in sorted(edges, reverse=True):
        root_left = uf.find(left)
        root_right = uf.find(right)
        if root_left == root_right:
            continue
        if root_frames.get(root_left, set()) & root_frames.get(root_right, set()):
            rejected_dynamic_same_frame += 1
            continue
        if uf.union(root_left, root_right):
            new_root = uf.find(root_left)
            old_root = root_right if new_root == root_left else root_left
            root_frames[new_root] = root_frames.get(root_left, set()) | root_frames.get(root_right, set())
            root_frames.pop(old_root, None)
            accepted += 1

    merged: dict[int, list[int]] = defaultdict(list)
    for comp_id, component in enumerate(components):
        merged[uf.find(comp_id)].extend(component)
    info = {
        "memory_candidate_pairs": int(candidate_pairs),
        "memory_accepted_merges": int(accepted),
        "memory_rejected_same_frame": int(rejected_same_frame),
        "memory_rejected_dynamic_same_frame": int(rejected_dynamic_same_frame),
        "memory_rejected_gap": int(rejected_gap),
        "memory_rejected_jaccard": int(rejected_jaccard),
        "memory_rejected_rgb": int(rejected_rgb),
        "memory_min_shared_tubes": int(min_shared_tubes),
        "memory_min_jaccard": float(min_jaccard),
        "memory_min_rgb_similarity": float(min_rgb_similarity),
        "memory_max_frame_gap": None if max_frame_gap is None else int(max_frame_gap),
        "memory_allow_rgb_fallback": bool(allow_rgb_fallback),
        "memory_max_rgb_fallback_per_component": int(max_rgb_fallback_per_component),
        "memory_rgb_fallback_candidate_pairs": int(rgb_fallback_candidate_pairs),
        "memory_rgb_fallback_edges": int(rgb_fallback_edges),
    }
    return list(merged.values()), info


def _merge_components_rgb_temporal_topk(
    state: SceneState,
    components: list[list[int]],
    *,
    min_rgb_similarity: float,
    max_frame_gap: int,
    max_rgb_fallback_per_component: int,
) -> tuple[list[list[int]], dict[str, Any]]:
    _supports, frames, rgbs = _component_descriptors(state, components)
    rank_to_components: dict[int, list[int]] = defaultdict(list)
    for comp_id, frame_set in enumerate(frames):
        for rank in frame_set:
            rank_to_components[int(rank)].append(int(comp_id))

    rgb_array = np.zeros((len(components), 3), dtype=np.float32)
    rgb_valid = np.zeros((len(components),), dtype=bool)
    for comp_id, rgb in enumerate(rgbs):
        if rgb is not None:
            rgb_array[int(comp_id)] = np.asarray(rgb, dtype=np.float32)
            rgb_valid[int(comp_id)] = True

    candidates_by_component: dict[int, list[tuple[float, int, int]]] = defaultdict(list)
    candidate_pairs = 0
    for rank, left_ids_raw in sorted(rank_to_components.items()):
        left_ids = np.asarray([idx for idx in sorted(set(left_ids_raw)) if bool(rgb_valid[idx])], dtype=np.int64)
        if left_ids.size == 0:
            continue
        for gap in range(1, int(max_frame_gap) + 1):
            right_ids = np.asarray(
                [idx for idx in sorted(set(rank_to_components.get(int(rank) + gap, []))) if bool(rgb_valid[idx])],
                dtype=np.int64,
            )
            if right_ids.size == 0:
                continue
            diff = (rgb_array[left_ids][:, None, :] - rgb_array[right_ids][None, :, :]) / 255.0
            sim = np.maximum(0.0, 1.0 - np.linalg.norm(diff, axis=2) / np.sqrt(3.0))
            left_best = np.argmax(sim, axis=1)
            for left_pos, right_pos in enumerate(left_best.tolist()):
                score = float(sim[int(left_pos), int(right_pos)])
                if score < float(min_rgb_similarity):
                    continue
                left = int(left_ids[int(left_pos)])
                right = int(right_ids[int(right_pos)])
                if frames[left] & frames[right]:
                    continue
                pair = (min(left, right), max(left, right))
                candidates_by_component[left].append((score, pair[0], pair[1]))
                candidates_by_component[right].append((score, pair[0], pair[1]))
                candidate_pairs += 1
            right_best = np.argmax(sim, axis=0)
            for right_pos, left_pos in enumerate(right_best.tolist()):
                score = float(sim[int(left_pos), int(right_pos)])
                if score < float(min_rgb_similarity):
                    continue
                left = int(left_ids[int(left_pos)])
                right = int(right_ids[int(right_pos)])
                if frames[left] & frames[right]:
                    continue
                pair = (min(left, right), max(left, right))
                candidates_by_component[left].append((score, pair[0], pair[1]))
                candidates_by_component[right].append((score, pair[0], pair[1]))
                candidate_pairs += 1

    selected: set[tuple[int, int]] = set()
    cap = int(max_rgb_fallback_per_component)
    for items in candidates_by_component.values():
        kept = sorted(items, reverse=True)[:cap] if cap > 0 else sorted(items, reverse=True)
        for _score, left, right in kept:
            selected.add((int(left), int(right)))

    edges = []
    for left, right in sorted(selected):
        sim = _rgb_similarity(rgbs[left], rgbs[right])
        if sim is None:
            continue
        edges.append((float(sim), int(left), int(right)))

    uf = UnionFind(len(components))
    root_frames: dict[int, set[int]] = {idx: set(frame_set) for idx, frame_set in enumerate(frames)}
    accepted = 0
    rejected_dynamic_same_frame = 0
    for _sim, left, right in sorted(edges, reverse=True):
        root_left = uf.find(left)
        root_right = uf.find(right)
        if root_left == root_right:
            continue
        if root_frames.get(root_left, set()) & root_frames.get(root_right, set()):
            rejected_dynamic_same_frame += 1
            continue
        if uf.union(root_left, root_right):
            new_root = uf.find(root_left)
            old_root = root_right if new_root == root_left else root_left
            root_frames[new_root] = root_frames.get(root_left, set()) | root_frames.get(root_right, set())
            root_frames.pop(old_root, None)
            accepted += 1

    merged: dict[int, list[int]] = defaultdict(list)
    for comp_id, component in enumerate(components):
        merged[uf.find(comp_id)].extend(component)
    info = {
        "memory_candidate_pairs": int(candidate_pairs),
        "memory_accepted_merges": int(accepted),
        "memory_rejected_same_frame": 0,
        "memory_rejected_dynamic_same_frame": int(rejected_dynamic_same_frame),
        "memory_rejected_gap": 0,
        "memory_rejected_jaccard": 0,
        "memory_rejected_rgb": 0,
        "memory_min_shared_tubes": 0,
        "memory_min_jaccard": 0.0,
        "memory_min_rgb_similarity": float(min_rgb_similarity),
        "memory_max_frame_gap": int(max_frame_gap),
        "memory_allow_rgb_fallback": True,
        "memory_max_rgb_fallback_per_component": int(max_rgb_fallback_per_component),
        "memory_rgb_fallback_candidate_pairs": int(candidate_pairs),
        "memory_rgb_fallback_edges": int(len(edges)),
        "memory_strategy": "rgb_temporal_topk",
    }
    return list(merged.values()), info


def _tube_error_proxy(labels_pred: dict[int, int], gt_labels: dict[int, int]) -> dict[str, Any]:
    pred_by_gt: dict[int, set[int]] = defaultdict(set)
    gt_by_pred: dict[int, set[int]] = defaultdict(set)
    for tube_id, pred in labels_pred.items():
        gt = int(gt_labels.get(int(tube_id), 0))
        if gt <= 0:
            continue
        pred_by_gt[gt].add(int(pred))
        gt_by_pred[int(pred)].add(gt)
    id_switches = sum(max(len(preds) - 1, 0) for preds in pred_by_gt.values())
    merge_errors = sum(1 for gts in gt_by_pred.values() if len(gts) > 1)
    fragmentation = float(np.mean([len(preds) for preds in pred_by_gt.values()])) if pred_by_gt else None
    return {
        "ID_switches": int(id_switches),
        "fragmentation": fragmentation,
        "merge_errors": int(merge_errors),
        "metric_scope_note": "tube-level proxy from GT-to-pred and pred-to-GT label sets; not framewise tracker IDSW",
    }


def _evaluate_components(state: SceneState, variant: str, components: list[list[int]], memory_info: dict[str, Any]) -> dict[str, Any]:
    labels_pred, unknown_ratio = _labels_for_components(
        components,
        state.support_by_tube,
        state.observation_count_by_tube,
        state.gt_labels,
        min_support=1,
        min_fraction=float(state.adaptive_fraction),
    )
    metrics = _cluster_metrics(labels_pred, state.gt_labels)
    component_count = int(len(components))
    labeled_tube_ids = [int(tid) for tid in sorted(labels_pred) if int(state.gt_labels.get(int(tid), 0)) > 0]
    predicted_object_labels = {int(labels_pred[tid]) for tid in labeled_tube_ids if int(labels_pred[tid]) <= component_count}
    predicted_unknown_labels = {int(labels_pred[tid]) for tid in labeled_tube_ids if int(labels_pred[tid]) > component_count}
    row = {
        "scene": state.scene,
        "variant": variant,
        "4D_ARI": metrics.get("ari"),
        "4D_purity": metrics.get("purity"),
        "4D_completeness": metrics.get("completeness"),
        "unknown_tube_ratio": float(unknown_ratio),
        "labeled_tube_count": metrics.get("labeled_tube_count"),
        "predicted_object_count_labeled": int(len(predicted_object_labels)),
        "predicted_unknown_count_labeled": int(len(predicted_unknown_labels)),
        "component_count": component_count,
        "temporal_span_mean": _component_stats(state.nodes, components, state.frame_rank).get("masklet_temporal_span_mean"),
        "support_density": float(state.support_density),
        "adaptive_fraction": float(state.adaptive_fraction),
        **memory_info,
        **_tube_error_proxy(labels_pred, state.gt_labels),
        "_labels_true": [int(state.gt_labels[int(tid)]) for tid in labeled_tube_ids],
        "_labels_pred": [int(labels_pred[int(tid)]) for tid in labeled_tube_ids],
    }
    return row


def _aggregate_rows(scene_rows: list[dict[str, Any]], local_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scene_rows:
        by_variant[str(row["variant"])].append(row)
    out = []
    local_ari = float(local_metrics.get("ARI"))
    local_purity = float(local_metrics.get("purity"))
    for variant, items in sorted(by_variant.items()):
        all_true: list[int] = []
        all_pred: list[int] = []
        true_offset = 0
        pred_offset = 0
        for item in items:
            true_vals = [int(v) for v in item.get("_labels_true", [])]
            pred_vals = [int(v) for v in item.get("_labels_pred", [])]
            all_true.extend([value + true_offset for value in true_vals])
            all_pred.extend([value + pred_offset for value in pred_vals])
            true_offset += (max(true_vals) + 11) if true_vals else 11
            pred_offset += (max(pred_vals) + 11) if pred_vals else 11
        metrics = _cluster_metrics(
            {idx: pred for idx, pred in enumerate(all_pred)},
            {idx: true for idx, true in enumerate(all_true)},
        )
        total_labeled = sum(int(row.get("labeled_tube_count") or 0) for row in items)
        weighted_unknown = sum(float(row.get("unknown_tube_ratio") or 0.0) * int(row.get("labeled_tube_count") or 0) for row in items)
        row = {
            "variant": variant,
            "4D_ARI": metrics.get("ari"),
            "4D_purity": metrics.get("purity"),
            "4D_completeness": metrics.get("completeness"),
            "unknown_tube_ratio": _safe_div(weighted_unknown, total_labeled),
            "scene0081_ARI": next((item.get("4D_ARI") for item in items if item.get("scene") == "scene0081_01"), None),
            "temporal_span_mean": float(np.mean([float(item["temporal_span_mean"]) for item in items if item.get("temporal_span_mean") is not None])),
            "mean_predictions_per_scene": float(
                np.mean(
                    [
                        float(item["predicted_object_count_labeled"])
                        for item in items
                        if item.get("predicted_object_count_labeled") is not None
                    ]
                )
            ),
            "mean_unknown_labels_per_scene": float(
                np.mean(
                    [
                        float(item["predicted_unknown_count_labeled"])
                        for item in items
                        if item.get("predicted_unknown_count_labeled") is not None
                    ]
                )
            ),
            "ID_switches": sum(int(item.get("ID_switches") or 0) for item in items),
            "fragmentation": float(np.mean([float(item["fragmentation"]) for item in items if item.get("fragmentation") is not None])),
            "merge_errors": sum(int(item.get("merge_errors") or 0) for item in items),
            "memory_candidate_pairs": sum(int(item.get("memory_candidate_pairs") or 0) for item in items),
            "memory_accepted_merges": sum(int(item.get("memory_accepted_merges") or 0) for item in items),
            "metric_scope_note": "tube-level proxy metrics; no AP/export metric is included here",
        }
        row["pass_4D_local_ARI_tolerance"] = bool(row["4D_ARI"] is not None and float(row["4D_ARI"]) >= local_ari - 0.03)
        row["pass_4D_purity_tolerance"] = bool(row["4D_purity"] is not None and float(row["4D_purity"]) >= local_purity - 0.03)
        row["pass_4D_temporal_span"] = bool(row["temporal_span_mean"] is not None and float(row["temporal_span_mean"]) >= 1.20)
        out.append(row)
    return out


def _public_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]


def _finalize_4d_summary_rows(summary_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    no_temporal = next((row for row in summary_rows if row["variant"] == "I5_no_temporal_control"), {})
    real_candidates = [row for row in summary_rows if not str(row["variant"]).startswith("I5_")]
    for row in summary_rows:
        if str(row["variant"]).startswith("I5_"):
            row["real_wins_controls"] = False
        else:
            row["real_minus_no_temporal"] = (
                None
                if row.get("4D_ARI") is None or no_temporal.get("4D_ARI") is None
                else float(row["4D_ARI"]) - float(no_temporal["4D_ARI"])
            )
            row["real_wins_controls"] = bool(row["real_minus_no_temporal"] is not None and row["real_minus_no_temporal"] >= 0.05)
        row["pass_4D_gate"] = bool(
            row.get("pass_4D_local_ARI_tolerance")
            and row.get("pass_4D_purity_tolerance")
            and row.get("pass_4D_temporal_span")
            and row.get("real_wins_controls")
        )

    passing = [row for row in real_candidates if row.get("pass_4D_gate")]
    best_pool = passing or real_candidates
    best = max(best_pool, key=lambda row: float(row.get("4D_ARI") or -999.0), default={})
    selection_policy = "max_4D_ARI_among_passing_variants" if passing else "max_4D_ARI_among_real_variants"
    return passing, best, selection_policy


def _update_pair_row_count(
    row_count: int,
    nodes: list[Any],
    support_by_region: dict[int, Counter[int]],
    pair_counts: Counter[tuple[int, int]],
    frame_rank: dict[int, int],
    args: argparse.Namespace,
) -> int:
    support_pairs = len(pair_counts)
    same_frame_pairs = len(_sample_same_frame_pairs(nodes, max_pairs_per_frame=int(args.max_same_frame_pairs_per_frame), seed=int(args.seed) + 11))
    all_sample_pairs = len(_sample_all_pairs(nodes, max_pairs=int(args.max_allpair_samples_per_scene), seed=int(args.seed) + 23))
    shuffled_region, _, _ = _shuffle_supports(support_by_region, seed=int(args.seed) + 37)
    shuffled_pair_counts = _support_pair_counts(
        nodes,
        shuffled_region,
        max_pairs_per_tube=int(args.max_support_pairs_per_tube),
        seed=int(args.seed) + 41,
        frame_rank=frame_rank,
    )
    return row_count + support_pairs + same_frame_pairs + all_sample_pairs + min(len(shuffled_pair_counts), int(args.max_shuffled_pair_rows_per_scene)) + all_sample_pairs // 2


def _build_scene_state(scene: str, args: argparse.Namespace, pair_row_count: int) -> tuple[SceneState, int]:
    nodes, labels_by_frame, _manifest = _load_masks(Path(args.mask_root), scene, args.source, args.mode, int(args.min_region_area))
    frame_rank = _frame_rank_map(labels_by_frame)
    tubes = _load_tubes(scene, args)
    gt_labels = _load_gt(scene, tubes, args)
    support_by_region, support_by_tube, observation_count_by_tube = _collect_observations(nodes, labels_by_frame, tubes, args)
    support_sets = {idx: set(counter.keys()) for idx, counter in support_by_region.items()}
    diagnostics, _gt_area = _region_diagnostics(scene, nodes, labels_by_frame, compute_rgb=True)
    pair_counts = _support_pair_counts(
        nodes,
        support_by_region,
        max_pairs_per_tube=int(args.max_support_pairs_per_tube),
        seed=int(args.seed) + pair_row_count,
        frame_rank=frame_rank,
    )
    short_edges = _filtered_edges(nodes, support_sets, pair_counts, frame_rank, max_delta=1, min_shared=2, min_jaccard=0.0)
    closure_edges = _filtered_edges(nodes, support_sets, pair_counts, frame_rank, min_delta=9, max_delta=None, min_shared=1, min_jaccard=0.0)
    closure_edges, rejected_rgb = _filter_edges_by_rgb(closure_edges, diagnostics, min_rgb_similarity=0.90)
    components, edge_info = _components_chain_then_closure(nodes, short_edges, closure_edges, frame_rank=frame_rank)
    components, split_info = _split_components_by_rgb(nodes, components, diagnostics, min_rgb_similarity=0.90)
    edge_info.update(split_info)
    edge_info["rejected_visual"] = int(rejected_rgb)
    support_density = _safe_div(edge_info.get("candidate_edges"), len(nodes))
    adaptive_fraction = 0.60 if float(support_density) >= 0.10 else 0.05
    state = SceneState(
        scene=scene,
        nodes=nodes,
        frame_rank=frame_rank,
        gt_labels=gt_labels,
        support_by_region=support_by_region,
        support_by_tube=support_by_tube,
        observation_count_by_tube=observation_count_by_tube,
        diagnostics=diagnostics,
        components=components,
        adaptive_fraction=adaptive_fraction,
        support_density=float(support_density),
        base_edge_info=edge_info,
    )
    return state, _update_pair_row_count(pair_row_count, nodes, support_by_region, pair_counts, frame_rank, args)


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    decision_path = Path(args.local_decision)
    local_decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if not bool(local_decision.get("pass_3D_gate")) or not bool(local_decision.get("pass_controls")):
        raise RuntimeError(f"Local v37 decision is not eligible for Phase I: {decision_path}")
    local_metrics = local_decision.get("best_metrics", {})
    scenes = _read_split(Path(args.split))
    setattr(_load_masks, "max_regions_per_scene", int(args.max_regions_per_scene))

    states = []
    pair_row_count = 0
    for scene in scenes:
        state, pair_row_count = _build_scene_state(scene, args, pair_row_count)
        states.append(state)

    variants: list[tuple[str, dict[str, Any] | str | None]] = [
        ("I0_no_memory_local_F31", None),
        ("I1_masklet_id_memory_noop", None),
        ("I2_shared_tube_memory_s1", {"min_shared_tubes": 1, "min_jaccard": 0.0, "min_rgb_similarity": 0.0, "max_frame_gap": None}),
        ("I2_shared_tube_memory_s2", {"min_shared_tubes": 2, "min_jaccard": 0.0, "min_rgb_similarity": 0.0, "max_frame_gap": None}),
        ("I2_shared_tube_memory_s1_j001", {"min_shared_tubes": 1, "min_jaccard": 0.01, "min_rgb_similarity": 0.0, "max_frame_gap": None}),
        ("I3_appearance_d4rt_memory_rgb090", {"min_shared_tubes": 1, "min_jaccard": 0.0, "min_rgb_similarity": 0.90, "max_frame_gap": None}),
        ("I3_appearance_d4rt_memory_rgb095", {"min_shared_tubes": 1, "min_jaccard": 0.0, "min_rgb_similarity": 0.95, "max_frame_gap": None}),
        ("I3_appearance_d4rt_memory_s2_rgb090", {"min_shared_tubes": 2, "min_jaccard": 0.0, "min_rgb_similarity": 0.90, "max_frame_gap": None}),
        ("I5_no_temporal_control", "no_temporal_control"),
    ]
    if bool(args.include_i4):
        variants.extend([
        (
            "I4_sparse_rgb_temporal_gap1_rgb099_top1",
            {
                "strategy": "rgb_temporal_topk",
                "min_rgb_similarity": 0.99,
                "max_frame_gap": 1,
                "max_rgb_fallback_per_component": 1,
            },
        ),
        (
            "I4_sparse_rgb_temporal_gap2_rgb099_top1",
            {
                "strategy": "rgb_temporal_topk",
                "min_rgb_similarity": 0.99,
                "max_frame_gap": 2,
                "max_rgb_fallback_per_component": 1,
            },
        ),
        (
            "I4_sparse_rgb_temporal_gap2_rgb0995_top1",
            {
                "strategy": "rgb_temporal_topk",
                "min_rgb_similarity": 0.995,
                "max_frame_gap": 2,
                "max_rgb_fallback_per_component": 1,
            },
        ),
        ])

    scene_rows = []
    for state in states:
        for variant, params in variants:
            if params == "no_temporal_control":
                components = [[idx] for idx in range(len(state.nodes))]
                memory_info = {
                    "memory_candidate_pairs": 0,
                    "memory_accepted_merges": 0,
                    "memory_rejected_same_frame": 0,
                    "memory_rejected_dynamic_same_frame": 0,
                    "memory_rejected_gap": 0,
                    "memory_rejected_jaccard": 0,
                    "memory_rejected_rgb": 0,
                    "memory_min_shared_tubes": 0,
                    "memory_min_jaccard": 0.0,
                    "memory_min_rgb_similarity": 0.0,
                    "memory_max_frame_gap": None,
                    "memory_allow_rgb_fallback": False,
                    "memory_max_rgb_fallback_per_component": 0,
                    "memory_rgb_fallback_candidate_pairs": 0,
                    "memory_rgb_fallback_edges": 0,
                }
            elif params is None:
                components = state.components
                memory_info = {
                    "memory_candidate_pairs": 0,
                    "memory_accepted_merges": 0,
                    "memory_rejected_same_frame": 0,
                    "memory_rejected_dynamic_same_frame": 0,
                    "memory_rejected_gap": 0,
                    "memory_rejected_jaccard": 0,
                    "memory_rejected_rgb": 0,
                    "memory_min_shared_tubes": 0,
                    "memory_min_jaccard": 0.0,
                    "memory_min_rgb_similarity": 0.0,
                    "memory_max_frame_gap": None,
                    "memory_allow_rgb_fallback": False,
                    "memory_max_rgb_fallback_per_component": 0,
                    "memory_rgb_fallback_candidate_pairs": 0,
                    "memory_rgb_fallback_edges": 0,
                }
            elif params.get("strategy") == "rgb_temporal_topk":
                components, memory_info = _merge_components_rgb_temporal_topk(
                    state,
                    state.components,
                    min_rgb_similarity=float(params["min_rgb_similarity"]),
                    max_frame_gap=int(params["max_frame_gap"]),
                    max_rgb_fallback_per_component=int(params["max_rgb_fallback_per_component"]),
                )
            else:
                components, memory_info = _merge_components(state, state.components, **params)
            scene_rows.append(_evaluate_components(state, str(variant), components, memory_info))

    summary_rows = _aggregate_rows(scene_rows, local_metrics)
    passing, best, selection_policy = _finalize_4d_summary_rows(summary_rows)
    final = {
        "plan": "docs/stream4d_v37_temporal_curriculum_masklet_plan.md",
        "phase": "I_4d_if_allowed",
        "local_decision": str(decision_path),
        "local_best_stage": local_decision.get("best_stage"),
        "local_ARI": local_metrics.get("ARI"),
        "local_purity": local_metrics.get("purity"),
        "final_status": "GO_4D_MEMORY" if passing else "NO_GO_4D_MEMORY_GATE_FAILED",
        "best_variant": best.get("variant"),
        "best_metrics": best,
        "passing_variant_count": int(len(passing)),
        "selection_policy": selection_policy,
        "notes": [
            "This is a tube-level 4D memory diagnostic after the v37 3D gate.",
            "ID_switches/fragmentation/merge_errors are tube-level label-set proxies, not framewise tracker metrics.",
            "No AP/AP50/AP25 export is performed by this tool.",
        ],
    }
    _write_csv(root / "4d_memory_scene_rows.csv", _public_rows(scene_rows))
    _write_csv(root / "4d_memory_summary.csv", summary_rows)
    _write_json(root / "4d_memory_summary.json", summary_rows)
    _write_json(root / "4d_memory_decision.json", final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v37 Phase I 4D memory diagnostics after the 3D gate is allowed.")
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--mask-root", default="outputs/audit/v37_dino_compact_filter_sources")
    parser.add_argument("--source", default="v37_dino_compact_filter")
    parser.add_argument("--mode", default="dino_k2_compact060_filter")
    parser.add_argument("--local-decision", default="outputs/audit/v37_adaptive_density_final_probe5/v37_final_decision/decision_summary.json")
    parser.add_argument("--output-root", default="outputs/audit/v37_4d_if_allowed")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v21_3_occupancy_d5_warmstart64_probe5_r1")
    parser.add_argument("--max-tubes-per-window", type=int, default=640)
    parser.add_argument("--image-width", type=int, default=1296)
    parser.add_argument("--image-height", type=int, default=968)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--min-region-area", type=int, default=64)
    parser.add_argument("--max-regions-per-scene", type=int, default=0)
    parser.add_argument("--max-support-pairs-per-tube", type=int, default=20000)
    parser.add_argument("--max-same-frame-pairs-per-frame", type=int, default=4000)
    parser.add_argument("--max-allpair-samples-per-scene", type=int, default=30000)
    parser.add_argument("--max-shuffled-pair-rows-per-scene", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=3701)
    parser.add_argument("--include-i4", action="store_true", help="Include experimental RGB fallback variants; may be slow.")
    final = run(parser.parse_args())
    print(json.dumps(_json_safe(final), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
