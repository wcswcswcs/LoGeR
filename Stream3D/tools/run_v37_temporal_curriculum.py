from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stream4d.scannet_stream import ScanNetStream
from tools.run_v26_object_quality_diagnostics import _auc, _json_safe, _read_split
from tools.run_v28_proposal_oracle import _cluster_metrics
from tools.run_v36_external_downstream_assignment import (
    RegionNode,
    UnionFind,
    _assign_tubes,
    _collect_observations,
    _component_frames,
    _load_gt,
    _load_masks,
    _load_tubes,
    _shuffle_supports,
)


LOCAL_GATE = {
    "ARI": 0.40,
    "purity": 0.85,
    "completeness": 0.50,
    "unknown_tube_ratio_max": 0.40,
    "scene0081_ARI": 0.20,
}
V36_CHAIN2_ARI = 0.29829760873172384
BEST_STAGE_EXCLUDED_PREFIXES = ("D4_", "D5_", "E5_", "E6_", "F5_", "F6_", "F7_", "F8_")


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


def _mean(values: list[Any]) -> float | None:
    vals = []
    for value in values:
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else None


def _quantile(values: list[Any], q: float) -> float | None:
    vals = []
    for value in values:
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            vals.append(v)
    return float(np.quantile(vals, q)) if vals else None


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _gate(metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "ari_pass": metrics.get("ARI") is not None and float(metrics["ARI"]) >= LOCAL_GATE["ARI"],
        "purity_pass": metrics.get("purity") is not None and float(metrics["purity"]) >= LOCAL_GATE["purity"],
        "completeness_pass": metrics.get("completeness") is not None and float(metrics["completeness"]) >= LOCAL_GATE["completeness"],
        "unknown_pass": metrics.get("unknown_tube_ratio") is not None
        and float(metrics["unknown_tube_ratio"]) <= LOCAL_GATE["unknown_tube_ratio_max"],
        "scene0081_pass": metrics.get("scene0081_ARI") is not None
        and float(metrics["scene0081_ARI"]) >= LOCAL_GATE["scene0081_ARI"],
    }
    checks["pass_3D_gate"] = bool(all(checks.values()))
    return checks


def _select_best_stage(stage_summary: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    best_real_candidates = [
        row
        for row in stage_summary
        if not str(row["stage"]).startswith(BEST_STAGE_EXCLUDED_PREFIXES) and row.get("ARI") is not None
    ]
    passing_candidates = [row for row in best_real_candidates if bool(row.get("pass_3D_gate"))]
    candidate_pool = passing_candidates or best_real_candidates
    best = max(candidate_pool, key=lambda row: float(row.get("ARI") or -999.0), default={})
    selection = {
        "candidate_stage_count": int(len(best_real_candidates)),
        "passing_stage_count": int(len(passing_candidates)),
        "selection_policy": (
            "max_ARI_among_pass_3D_gate_candidates"
            if passing_candidates
            else "max_ARI_among_real_method_candidates"
        ),
    }
    return best, selection


def _load_instance_map(stream: ScanNetStream, frame_id: int) -> np.ndarray | None:
    candidates = [
        stream.root / "instance" / "instance" / f"{int(frame_id)}.png",
        stream.root / "instance" / f"{int(frame_id)}.png",
    ]
    for path in candidates:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is not None:
            if image.ndim == 3:
                image = image[..., 0]
            return image.astype(np.int64)
    return None


def _node_mask(labels_by_frame: dict[int, np.ndarray], node: RegionNode) -> np.ndarray:
    label = labels_by_frame[int(node.frame_id)]
    return label == int(node.node_id) + 1


def _region_diagnostics(
    scene: str,
    nodes: list[RegionNode],
    labels_by_frame: dict[int, np.ndarray],
    *,
    compute_rgb: bool,
) -> tuple[dict[int, dict[str, Any]], dict[tuple[int, int], int]]:
    stream = ScanNetStream(seq_name=scene)
    gt_cache: dict[int, np.ndarray | None] = {}
    rgb_cache: dict[int, np.ndarray] = {}
    frame_gt_area: Counter[tuple[int, int]] = Counter()
    out: dict[int, dict[str, Any]] = {}
    for node in nodes:
        frame = int(node.frame_id)
        if frame not in gt_cache:
            gt_cache[frame] = _load_instance_map(stream, frame)
            if gt_cache[frame] is not None:
                vals, counts = np.unique(gt_cache[frame], return_counts=True)
                for value, count in zip(vals.tolist(), counts.tolist()):
                    if int(value) > 0:
                        frame_gt_area[(frame, int(value))] = int(count)
        mask = _node_mask(labels_by_frame, node)
        gt = gt_cache.get(frame)
        counts: Counter[int] = Counter()
        if gt is not None:
            if gt.shape != mask.shape:
                gt = cv2.resize(gt.astype(np.int32), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            vals, overlaps = np.unique(gt[mask], return_counts=True)
            for value, count in zip(vals.tolist(), overlaps.tolist()):
                if int(value) > 0:
                    counts[int(value)] += int(count)
        labeled = int(sum(counts.values()))
        dominant_gt = int(counts.most_common(1)[0][0]) if counts else 0
        dominant_count = int(counts.most_common(1)[0][1]) if counts else 0
        rgb_mean = None
        if compute_rgb:
            if frame not in rgb_cache:
                rgb_cache[frame] = stream.load_rgb(frame)
            rgb = rgb_cache[frame]
            if rgb.shape[:2] != mask.shape:
                rgb = cv2.resize(rgb, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
            pix = rgb[mask]
            if pix.size:
                rgb_mean = [float(v) for v in np.mean(pix.astype(np.float32), axis=0).tolist()]
        out[int(node.node_id)] = {
            "gt_counts": dict(counts),
            "dominant_gt": dominant_gt,
            "dominant_gt_ratio": _safe_div(dominant_count, labeled),
            "positive_gt_count": int(len(counts)),
            "labeled_pixel_count": labeled,
            "mixed_seed_flag": bool(len(counts) > 1 and _safe_div(dominant_count, labeled) < 0.95),
            "rgb_mean": rgb_mean,
        }
    return out, {tuple(k): int(v) for k, v in frame_gt_area.items()}


def _rgb_similarity(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    dist = float(np.linalg.norm((va - vb) / 255.0))
    return float(max(0.0, 1.0 - dist / math.sqrt(3.0)))


def _bucket(delta: int) -> str:
    if delta == 0:
        return "B0_dt0_same_frame"
    if delta == 1:
        return "B1_dt1_adjacent"
    if delta == 2:
        return "B2_dt2_near"
    if 3 <= delta <= 4:
        return "B3_dt3_4_short"
    if 5 <= delta <= 8:
        return "B4_dt5_8_mid"
    return "B5_dt_gt8_far"


def _frame_rank_map(labels_by_frame: dict[int, np.ndarray]) -> dict[int, int]:
    return {int(frame): idx for idx, frame in enumerate(sorted(int(frame) for frame in labels_by_frame))}


def _temporal_delta(nodes: list[RegionNode], left: int, right: int, frame_rank: dict[int, int]) -> int:
    frame_left = int(nodes[int(left)].frame_id)
    frame_right = int(nodes[int(right)].frame_id)
    if frame_left in frame_rank and frame_right in frame_rank:
        return abs(int(frame_rank[frame_left]) - int(frame_rank[frame_right]))
    return abs(frame_left - frame_right)


def _support_pair_counts(
    nodes: list[RegionNode],
    support_by_region: dict[int, Counter[int]],
    *,
    max_pairs_per_tube: int,
    seed: int,
    frame_rank: dict[int, int] | None = None,
) -> Counter[tuple[int, int]]:
    tube_to_regions: dict[int, list[int]] = defaultdict(list)
    for region, counter in support_by_region.items():
        for tube in counter:
            tube_to_regions[int(tube)].append(int(region))
    rng = np.random.default_rng(int(seed))
    pair_counts: Counter[tuple[int, int]] = Counter()
    for tube, raw_regions in tube_to_regions.items():
        regions = sorted(
            set(raw_regions),
            key=lambda idx: (
                int(frame_rank.get(int(nodes[idx].frame_id), int(nodes[idx].frame_id))) if frame_rank else int(nodes[idx].frame_id),
                int(idx),
            ),
        )
        pairs: list[tuple[int, int]] = []
        for pos, left in enumerate(regions):
            for right in regions[pos + 1 :]:
                if int(nodes[left].frame_id) == int(nodes[right].frame_id):
                    continue
                pairs.append((min(left, right), max(left, right)))
        if max_pairs_per_tube > 0 and len(pairs) > max_pairs_per_tube:
            keep = rng.choice(len(pairs), size=max_pairs_per_tube, replace=False)
            pairs = [pairs[int(idx)] for idx in keep.tolist()]
        for pair in pairs:
            pair_counts[pair] += 1
    return pair_counts


def _sample_same_frame_pairs(
    nodes: list[RegionNode],
    *,
    max_pairs_per_frame: int,
    seed: int,
) -> list[tuple[int, int]]:
    by_frame: dict[int, list[int]] = defaultdict(list)
    for node in nodes:
        by_frame[int(node.frame_id)].append(int(node.node_id))
    rng = np.random.default_rng(int(seed))
    out = []
    for frame, ids in by_frame.items():
        if len(ids) < 2:
            continue
        pairs = [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))]
        if len(pairs) > max_pairs_per_frame:
            keep = rng.choice(len(pairs), size=max_pairs_per_frame, replace=False)
            pairs = [pairs[int(idx)] for idx in keep.tolist()]
        out.extend(pairs)
    return out


def _sample_all_pairs(
    nodes: list[RegionNode],
    *,
    max_pairs: int,
    seed: int,
) -> list[tuple[int, int]]:
    rng = np.random.default_rng(int(seed))
    n = len(nodes)
    if n < 2 or max_pairs <= 0:
        return []
    pairs: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = max_pairs * 20
    while len(pairs) < max_pairs and attempts < max_attempts:
        left = int(rng.integers(0, n))
        right = int(rng.integers(0, n - 1))
        if right >= left:
            right += 1
        a, b = (left, right) if left < right else (right, left)
        pairs.add((a, b))
        attempts += 1
    return sorted(pairs)


def _pair_row(
    *,
    scene: str,
    nodes: list[RegionNode],
    support_sets: dict[int, set[int]],
    pair_counts: Counter[tuple[int, int]],
    diagnostics: dict[int, dict[str, Any]],
    pair: tuple[int, int],
    source: str,
    frame_rank: dict[int, int],
) -> dict[str, Any]:
    left, right = int(pair[0]), int(pair[1])
    nl, nr = nodes[left], nodes[right]
    set_l = support_sets.get(left, set())
    set_r = support_sets.get(right, set())
    shared = int(pair_counts.get((min(left, right), max(left, right)), len(set_l & set_r)))
    union = len(set_l | set_r)
    dleft = diagnostics.get(left, {})
    dright = diagnostics.get(right, {})
    gt_l = int(dleft.get("dominant_gt") or 0)
    gt_r = int(dright.get("dominant_gt") or 0)
    labeled_pair = bool(gt_l > 0 and gt_r > 0)
    global_delta = abs(int(nl.frame_id) - int(nr.frame_id))
    delta = _temporal_delta(nodes, left, right, frame_rank)
    return {
        "scene": scene,
        "source": source,
        "region_i": left,
        "region_j": right,
        "frame_i": int(nl.frame_id),
        "frame_j": int(nr.frame_id),
        "delta_t": int(delta),
        "global_frame_delta": int(global_delta),
        "delta_bucket": _bucket(delta),
        "shared_d4rt_tube_count": shared,
        "shared_d4rt_jaccard": _safe_div(shared, union),
        "dino_similarity": None,
        "rgb_similarity": _rgb_similarity(dleft.get("rgb_mean"), dright.get("rgb_mean")),
        "mask_overlap_score": None if int(nl.frame_id) != int(nr.frame_id) else 0.0,
        "canonical_distance_normalized": None,
        "same_frame_cannot_link": bool(int(nl.frame_id) == int(nr.frame_id)),
        "visible_outside_conflict": None,
        "boundary_cross_score": None,
        "diagnostic_same_GT": bool(labeled_pair and gt_l == gt_r),
        "diagnostic_diff_GT": bool(labeled_pair and gt_l != gt_r),
        "diagnostic_labeled_pair": labeled_pair,
        "dominant_GT_i": gt_l,
        "dominant_GT_j": gt_r,
    }


def _summarize_pair_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["delta_bucket"])].append(row)
        if row.get("source") == "all_pair_sample":
            groups["B6_all_pair_sample"].append(row)
        if row.get("source") == "shuffled_d4rt_support":
            groups["B7_shuffled_d4rt"].append(row)
        if row.get("source") == "no_temporal_sample":
            groups["B8_no_temporal"].append(row)
    out = []
    for bucket, bucket_rows in sorted(groups.items()):
        labeled = [row for row in bucket_rows if row.get("diagnostic_labeled_pair")]
        same = [row for row in labeled if row.get("diagnostic_same_GT")]
        diff = [row for row in labeled if row.get("diagnostic_diff_GT")]
        labels = np.asarray([1 if row.get("diagnostic_same_GT") else 0 for row in labeled], dtype=np.int64)
        scores = np.asarray([float(row.get("shared_d4rt_jaccard") or 0.0) for row in labeled], dtype=np.float64)
        scene0081 = [row for row in bucket_rows if row.get("scene") == "scene0081_01" and row.get("diagnostic_labeled_pair")]
        scene0081_same = [row for row in scene0081 if row.get("diagnostic_same_GT")]
        scene0081_diff = [row for row in scene0081 if row.get("diagnostic_diff_GT")]
        out.append(
            {
                "delta_bin": bucket,
                "pair_count": int(len(bucket_rows)),
                "labeled_pair_count": int(len(labeled)),
                "same_GT_ratio": _safe_div(len(same), len(labeled)),
                "diff_GT_ratio": _safe_div(len(diff), len(labeled)),
                "false_merge_rate": _safe_div(len(diff), len(labeled)),
                "merge_AUC": _auc(labels, scores) if len(labeled) else None,
                "cut_AUC": _auc(1 - labels, 1.0 - scores) if len(labeled) else None,
                "D4RT_shared_count_p50": _quantile([row.get("shared_d4rt_tube_count") for row in bucket_rows], 0.5),
                "D4RT_jaccard_p50": _quantile([row.get("shared_d4rt_jaccard") for row in bucket_rows], 0.5),
                "DINO_similarity_sameGT_mean": None,
                "DINO_similarity_diffGT_mean": None,
                "RGB_similarity_sameGT_mean": _mean([row.get("rgb_similarity") for row in same]),
                "RGB_similarity_diffGT_mean": _mean([row.get("rgb_similarity") for row in diff]),
                "scene0081_same_GT_ratio": _safe_div(len(scene0081_same), len(scene0081)),
                "scene0081_false_merge_rate": _safe_div(len(scene0081_diff), len(scene0081)),
            }
        )
    return out


def _seed_metrics(
    scene: str,
    nodes: list[RegionNode],
    labels_by_frame: dict[int, np.ndarray],
    diagnostics: dict[int, dict[str, Any]],
    gt_area: dict[tuple[int, int], int],
    active_node_ids: set[int],
    *,
    variant: str,
    status: str,
) -> dict[str, Any]:
    active = [node for node in nodes if int(node.node_id) in active_node_ids]
    ratios = [float(diagnostics[int(node.node_id)].get("dominant_gt_ratio") or 0.0) for node in active]
    mixed = [bool(diagnostics[int(node.node_id)].get("mixed_seed_flag")) for node in active]
    best_iou: dict[tuple[int, int], float] = defaultdict(float)
    for node in active:
        diag = diagnostics[int(node.node_id)]
        counts = {int(k): int(v) for k, v in dict(diag.get("gt_counts") or {}).items()}
        if not counts:
            continue
        area = int(node.area)
        for gt, overlap in counts.items():
            denom = area + int(gt_area.get((int(node.frame_id), int(gt)), 0)) - int(overlap)
            if denom > 0:
                best_iou[(int(node.frame_id), int(gt))] = max(best_iou[(int(node.frame_id), int(gt))], float(overlap / denom))
    return {
        "scene": scene,
        "variant": variant,
        "status": status,
        "same_frame_seed_count": int(len(active)),
        "same_frame_mixed_seed_rate": _safe_div(sum(1 for value in mixed if value), len(mixed)),
        "seed_purity_mean": _mean(ratios),
        "seed_purity_p10": _quantile(ratios, 0.10),
        "seed_GT_coverage@0.05": _safe_div(sum(1 for key in gt_area if best_iou.get(key, 0.0) >= 0.05), len(gt_area)),
        "seed_GT_coverage@0.10": _safe_div(sum(1 for key in gt_area if best_iou.get(key, 0.0) >= 0.10), len(gt_area)),
    }


def _aggregate_seed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row["variant"])].append(row)
    for variant, items in sorted(by_variant.items()):
        out.append(
            {
                "variant": variant,
                "status": ";".join(sorted(set(str(row.get("status")) for row in items))),
                "purity": _mean([row.get("seed_purity_mean") for row in items]),
                "p10": _mean([row.get("seed_purity_p10") for row in items]),
                "GT_cov@0.05": _mean([row.get("seed_GT_coverage@0.05") for row in items]),
                "GT_cov@0.10": _mean([row.get("seed_GT_coverage@0.10") for row in items]),
                "mixed_rate": _mean([row.get("same_frame_mixed_seed_rate") for row in items]),
                "scene0081": next((row.get("seed_GT_coverage@0.05") for row in items if row.get("scene") == "scene0081_01"), None),
                "scene0081_seed_purity": next((row.get("seed_purity_mean") for row in items if row.get("scene") == "scene0081_01"), None),
                "scene_count": int(len(items)),
            }
        )
    return out


def _edge_score(
    pair: tuple[int, int],
    nodes: list[RegionNode],
    support_sets: dict[int, set[int]],
    pair_counts: Counter[tuple[int, int]],
    frame_rank: dict[int, int],
) -> tuple[float, int, float, int]:
    left, right = pair
    shared = int(pair_counts[pair])
    union = len(support_sets.get(left, set()) | support_sets.get(right, set()))
    jaccard = _safe_div(shared, union)
    delta = _temporal_delta(nodes, left, right, frame_rank)
    score = float(shared + 4.0 * jaccard - 0.03 * delta)
    return score, shared, jaccard, delta


def _filtered_edges(
    nodes: list[RegionNode],
    support_sets: dict[int, set[int]],
    pair_counts: Counter[tuple[int, int]],
    frame_rank: dict[int, int],
    *,
    min_delta: int = 1,
    max_delta: int | None = None,
    min_shared: int = 1,
    min_jaccard: float = 0.0,
) -> list[tuple[float, int, float, int, int, int]]:
    out = []
    for pair in pair_counts:
        score, shared, jaccard, delta = _edge_score(pair, nodes, support_sets, pair_counts, frame_rank)
        if delta < min_delta:
            continue
        if max_delta is not None and delta > max_delta:
            continue
        if shared < min_shared or jaccard < min_jaccard:
            continue
        out.append((score, shared, jaccard, delta, int(pair[0]), int(pair[1])))
    return sorted(out, reverse=True)


def _filter_edges_by_rgb(
    edges: list[tuple[float, int, float, int, int, int]],
    diagnostics: dict[int, dict[str, Any]] | None,
    *,
    min_rgb_similarity: float,
) -> tuple[list[tuple[float, int, float, int, int, int]], int]:
    if min_rgb_similarity <= 0.0:
        return edges, 0
    if diagnostics is None:
        return [], len(edges)
    kept = []
    rejected = 0
    for edge in edges:
        _score, _shared, _jaccard, _delta, left, right = edge
        sim = _rgb_similarity(
            diagnostics.get(int(left), {}).get("rgb_mean"),
            diagnostics.get(int(right), {}).get("rgb_mean"),
        )
        if sim is not None and float(sim) >= float(min_rgb_similarity):
            kept.append(edge)
        else:
            rejected += 1
    return kept, int(rejected)


def _components_from_edges(
    nodes: list[RegionNode],
    edges: list[tuple[float, int, float, int, int, int]],
    *,
    mode: str,
    frame_rank: dict[int, int],
    stable_only_long_range: bool = False,
) -> tuple[list[list[int]], dict[str, Any]]:
    uf = UnionFind(len(nodes))
    frame_sets: dict[int, set[int]] = {idx: {int(node.frame_id)} for idx, node in enumerate(nodes)}
    rank_sets: dict[int, set[int]] = {
        idx: {int(frame_rank.get(int(node.frame_id), int(node.frame_id)))} for idx, node in enumerate(nodes)
    }
    members: dict[int, list[int]] = {idx: [idx] for idx in range(len(nodes))}
    used_pair_regions: set[tuple[int, int, int]] = set()
    accepted = 0
    rejected_same_frame = 0
    rejected_matching = 0
    rejected_unstable = 0
    for score, shared, jaccard, delta, left, right in edges:
        root_left = uf.find(left)
        root_right = uf.find(right)
        if root_left == root_right:
            continue
        if frame_sets[root_left] & frame_sets[root_right]:
            rejected_same_frame += 1
            continue
        if stable_only_long_range and delta > 8:
            span_left = max(rank_sets[root_left]) - min(rank_sets[root_left]) + 1
            span_right = max(rank_sets[root_right]) - min(rank_sets[root_right]) + 1
            if max(span_left, span_right) < 2:
                rejected_unstable += 1
                continue
        if mode in {"bipartite", "flow_top1"}:
            fl, fr = int(nodes[left].frame_id), int(nodes[right].frame_id)
            key_l = (min(fl, fr), max(fl, fr), left)
            key_r = (min(fl, fr), max(fl, fr), right)
            if key_l in used_pair_regions or key_r in used_pair_regions:
                rejected_matching += 1
                continue
            used_pair_regions.add(key_l)
            used_pair_regions.add(key_r)
        if uf.union(root_left, root_right):
            new_root = uf.find(root_left)
            old_root = root_right if new_root == root_left else root_left
            frame_sets[new_root] = frame_sets.get(root_left, set()) | frame_sets.get(root_right, set())
            rank_sets[new_root] = rank_sets.get(root_left, set()) | rank_sets.get(root_right, set())
            members[new_root] = members.get(root_left, []) + members.get(root_right, [])
            frame_sets.pop(old_root, None)
            rank_sets.pop(old_root, None)
            members.pop(old_root, None)
            accepted += 1
    groups: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(nodes)):
        groups[uf.find(idx)].append(idx)
    return list(groups.values()), {
        "accepted_edges": int(accepted),
        "candidate_edges": int(len(edges)),
        "rejected_same_frame_conflict": int(rejected_same_frame),
        "rejected_matching": int(rejected_matching),
        "rejected_unstable_long_range": int(rejected_unstable),
    }


def _components_chain_then_closure(
    nodes: list[RegionNode],
    short_edges: list[tuple[float, int, float, int, int, int]],
    closure_edges: list[tuple[float, int, float, int, int, int]],
    *,
    frame_rank: dict[int, int],
) -> tuple[list[list[int]], dict[str, Any]]:
    uf = UnionFind(len(nodes))
    frame_sets: dict[int, set[int]] = {idx: {int(node.frame_id)} for idx, node in enumerate(nodes)}
    rank_sets: dict[int, set[int]] = {
        idx: {int(frame_rank.get(int(node.frame_id), int(node.frame_id)))} for idx, node in enumerate(nodes)
    }
    used_pair_regions: set[tuple[int, int, int]] = set()
    stats = Counter()

    def merge_edges(edges: list[tuple[float, int, float, int, int, int]], *, phase: str, bipartite: bool) -> None:
        for _score, _shared, _jaccard, delta, left, right in edges:
            root_left = uf.find(left)
            root_right = uf.find(right)
            if root_left == root_right:
                continue
            if frame_sets[root_left] & frame_sets[root_right]:
                stats[f"{phase}_rejected_same_frame_conflict"] += 1
                continue
            if phase == "closure" and delta > 8:
                span_left = max(rank_sets[root_left]) - min(rank_sets[root_left]) + 1
                span_right = max(rank_sets[root_right]) - min(rank_sets[root_right]) + 1
                if max(span_left, span_right) < 2:
                    stats[f"{phase}_rejected_unstable_long_range"] += 1
                    continue
            if bipartite:
                fl, fr = int(nodes[left].frame_id), int(nodes[right].frame_id)
                key_l = (min(fl, fr), max(fl, fr), left)
                key_r = (min(fl, fr), max(fl, fr), right)
                if key_l in used_pair_regions or key_r in used_pair_regions:
                    stats[f"{phase}_rejected_matching"] += 1
                    continue
                used_pair_regions.add(key_l)
                used_pair_regions.add(key_r)
            if uf.union(root_left, root_right):
                new_root = uf.find(root_left)
                old_root = root_right if new_root == root_left else root_left
                frame_sets[new_root] = frame_sets.get(root_left, set()) | frame_sets.get(root_right, set())
                rank_sets[new_root] = rank_sets.get(root_left, set()) | rank_sets.get(root_right, set())
                frame_sets.pop(old_root, None)
                rank_sets.pop(old_root, None)
                stats[f"{phase}_accepted_edges"] += 1

    merge_edges(short_edges, phase="short", bipartite=True)
    merge_edges(closure_edges, phase="closure", bipartite=False)
    groups: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(nodes)):
        groups[uf.find(idx)].append(idx)
    return list(groups.values()), {
        "accepted_edges": int(stats["short_accepted_edges"] + stats["closure_accepted_edges"]),
        "candidate_edges": int(len(short_edges) + len(closure_edges)),
        "rejected_same_frame_conflict": int(
            stats["short_rejected_same_frame_conflict"] + stats["closure_rejected_same_frame_conflict"]
        ),
        "rejected_matching": int(stats["short_rejected_matching"] + stats["closure_rejected_matching"]),
        "rejected_unstable_long_range": int(stats["closure_rejected_unstable_long_range"]),
        "short_accepted_edges": int(stats["short_accepted_edges"]),
        "closure_accepted_edges": int(stats["closure_accepted_edges"]),
        "short_candidate_edges": int(len(short_edges)),
        "closure_candidate_edges": int(len(closure_edges)),
    }


def _split_components_by_rgb(
    nodes: list[RegionNode],
    components: list[list[int]],
    diagnostics: dict[int, dict[str, Any]] | None,
    *,
    min_rgb_similarity: float,
) -> tuple[list[list[int]], dict[str, int]]:
    if diagnostics is None or min_rgb_similarity <= 0.0:
        return components, {"rgb_split_components": 0, "rgb_split_new_components": 0}
    refined: list[list[int]] = []
    split_components = 0
    added_components = 0
    for component in components:
        if len(component) <= 1:
            refined.append(list(component))
            continue
        local = UnionFind(len(component))
        for i, left in enumerate(component):
            left_rgb = diagnostics.get(int(nodes[int(left)].node_id), {}).get("rgb_mean")
            if left_rgb is None:
                continue
            for j in range(i + 1, len(component)):
                right = int(component[j])
                if int(nodes[int(left)].frame_id) == int(nodes[right].frame_id):
                    continue
                right_rgb = diagnostics.get(int(nodes[right].node_id), {}).get("rgb_mean")
                sim = _rgb_similarity(left_rgb, right_rgb)
                if sim is not None and float(sim) >= float(min_rgb_similarity):
                    local.union(i, j)
        groups: dict[int, list[int]] = defaultdict(list)
        for i, node_idx in enumerate(component):
            groups[local.find(i)].append(int(node_idx))
        parts = list(groups.values())
        if len(parts) > 1:
            split_components += 1
            added_components += len(parts) - 1
        refined.extend(parts)
    return refined, {"rgb_split_components": int(split_components), "rgb_split_new_components": int(added_components)}


def _drop_rgb_incoherent_components(
    nodes: list[RegionNode],
    components: list[list[int]],
    diagnostics: dict[int, dict[str, Any]] | None,
    *,
    min_pairwise_similarity: float,
    max_component_nodes: int = 0,
) -> tuple[list[list[int]], dict[str, int]]:
    if diagnostics is None or min_pairwise_similarity <= 0.0:
        return components, {"rgb_unknown_components": 0, "rgb_unknown_nodes": 0}
    kept: list[list[int]] = []
    unknown_components = 0
    unknown_nodes = 0
    for component in components:
        if len(component) <= 1:
            kept.append(list(component))
            continue
        if int(max_component_nodes) > 0 and len(component) > int(max_component_nodes):
            kept.append(list(component))
            continue
        rgb_values = []
        for node_idx in component:
            rgb = diagnostics.get(int(nodes[int(node_idx)].node_id), {}).get("rgb_mean")
            if rgb is not None:
                rgb_values.append((int(node_idx), rgb))
        if len(rgb_values) < 2:
            kept.append(list(component))
            continue
        rgb_arr = np.asarray([rgb for _node_idx, rgb in rgb_values], dtype=np.float32)
        center = np.mean(rgb_arr, axis=0).tolist()
        min_similarity = min(
            float(sim)
            for sim in (_rgb_similarity(rgb, center) for _node_idx, rgb in rgb_values)
            if sim is not None
        )
        if min_similarity < float(min_pairwise_similarity):
            unknown_components += 1
            unknown_nodes += len(component)
            continue
        kept.append(list(component))
    return kept, {"rgb_unknown_components": int(unknown_components), "rgb_unknown_nodes": int(unknown_nodes)}


def _isolate_rgb_outlier_nodes(
    nodes: list[RegionNode],
    components: list[list[int]],
    diagnostics: dict[int, dict[str, Any]] | None,
    *,
    min_center_similarity: float,
    max_component_nodes: int = 0,
) -> tuple[list[list[int]], dict[str, int]]:
    if diagnostics is None or min_center_similarity <= 0.0:
        return components, {"rgb_outlier_components": 0, "rgb_outlier_nodes": 0}
    refined: list[list[int]] = []
    outlier_components = 0
    outlier_nodes = 0
    for component in components:
        component = [int(idx) for idx in component]
        if len(component) <= 2:
            refined.append(component)
            continue
        if int(max_component_nodes) > 0 and len(component) > int(max_component_nodes):
            refined.append(component)
            continue
        rgb_values = []
        for node_idx in component:
            rgb = diagnostics.get(int(nodes[int(node_idx)].node_id), {}).get("rgb_mean")
            if rgb is not None:
                rgb_values.append((int(node_idx), rgb))
        if len(rgb_values) < 3:
            refined.append(component)
            continue
        rgb_arr = np.asarray([rgb for _node_idx, rgb in rgb_values], dtype=np.float32)
        center = np.median(rgb_arr, axis=0).tolist()
        low_nodes: set[int] = set()
        for node_idx, rgb in rgb_values:
            sim = _rgb_similarity(rgb, center)
            if sim is not None and float(sim) < float(min_center_similarity):
                low_nodes.add(int(node_idx))
        if not low_nodes or len(low_nodes) >= len(component):
            refined.append(component)
            continue
        core = [idx for idx in component if idx not in low_nodes]
        if len(core) >= 2:
            refined.append(core)
            refined.extend([[idx] for idx in component if idx in low_nodes])
            outlier_components += 1
            outlier_nodes += len(low_nodes)
        else:
            refined.append(component)
    return refined, {"rgb_outlier_components": int(outlier_components), "rgb_outlier_nodes": int(outlier_nodes)}


def _singleton_rgb_incoherent_components(
    nodes: list[RegionNode],
    components: list[list[int]],
    diagnostics: dict[int, dict[str, Any]] | None,
    *,
    min_center_similarity: float,
    max_component_nodes: int = 0,
) -> tuple[list[list[int]], dict[str, int]]:
    if diagnostics is None or min_center_similarity <= 0.0:
        return components, {"rgb_singleton_components": 0, "rgb_singleton_nodes": 0}
    refined: list[list[int]] = []
    singleton_components = 0
    singleton_nodes = 0
    for component in components:
        component = [int(idx) for idx in component]
        if len(component) <= 1:
            refined.append(component)
            continue
        if int(max_component_nodes) > 0 and len(component) > int(max_component_nodes):
            refined.append(component)
            continue
        rgb_values = []
        for node_idx in component:
            rgb = diagnostics.get(int(nodes[int(node_idx)].node_id), {}).get("rgb_mean")
            if rgb is not None:
                rgb_values.append((int(node_idx), rgb))
        if len(rgb_values) < 2:
            refined.append(component)
            continue
        rgb_arr = np.asarray([rgb for _node_idx, rgb in rgb_values], dtype=np.float32)
        center = np.mean(rgb_arr, axis=0).tolist()
        min_similarity = min(
            float(sim)
            for sim in (_rgb_similarity(rgb, center) for _node_idx, rgb in rgb_values)
            if sim is not None
        )
        if min_similarity < float(min_center_similarity):
            refined.extend([[idx] for idx in component])
            singleton_components += 1
            singleton_nodes += len(component)
            continue
        refined.append(component)
    return refined, {"rgb_singleton_components": int(singleton_components), "rgb_singleton_nodes": int(singleton_nodes)}


def _component_stats(nodes: list[RegionNode], components: list[list[int]], frame_rank: dict[int, int]) -> dict[str, Any]:
    spans = []
    violations = 0
    for comp in components:
        frames = [int(nodes[idx].frame_id) for idx in comp]
        ranks = [int(frame_rank.get(frame, frame)) for frame in frames]
        if frames:
            spans.append(max(ranks) - min(ranks) + 1)
            violations += len(frames) - len(set(frames))
    return {
        "masklet_count": int(len(components)),
        "masklet_temporal_span_mean": _mean(spans),
        "same_frame_cannot_link_violations": int(violations),
    }


def _labels_for_components(
    components: list[list[int]],
    support_by_tube: dict[int, Counter[int]],
    observation_count_by_tube: dict[int, int],
    gt_labels: dict[int, int],
    *,
    min_support: int,
    min_fraction: float,
) -> tuple[dict[int, int], float]:
    return _assign_tubes(
        components,
        support_by_tube,
        observation_count_by_tube,
        gt_labels,
        min_support=min_support,
        min_fraction=min_fraction,
    )


def _labels_for_components_margin_unknown(
    components: list[list[int]],
    support_by_tube: dict[int, Counter[int]],
    observation_count_by_tube: dict[int, int],
    gt_labels: dict[int, int],
    *,
    min_support: int,
    min_fraction: float,
    min_margin_fraction: float,
) -> tuple[dict[int, int], float, dict[str, int]]:
    node_to_component = {}
    for comp_idx, component in enumerate(components):
        for node_id in component:
            node_to_component[int(node_id)] = int(comp_idx)
    labels_pred: dict[int, int] = {}
    unknown_count = 0
    margin_unknown = 0
    next_unknown = len(components) + 1
    for tube_id, gt in sorted(gt_labels.items()):
        if int(gt) <= 0:
            continue
        comp_counts: Counter[int] = Counter()
        for node_id, count in support_by_tube.get(int(tube_id), Counter()).items():
            comp = node_to_component.get(int(node_id))
            if comp is not None:
                comp_counts[int(comp)] += int(count)
        if not comp_counts:
            labels_pred[int(tube_id)] = next_unknown
            next_unknown += 1
            unknown_count += 1
            continue
        top = comp_counts.most_common(2)
        comp, count = top[0]
        second = int(top[1][1]) if len(top) > 1 else 0
        obs = max(int(observation_count_by_tube.get(int(tube_id), 0)), 1)
        frac = float(count / obs)
        margin = float((int(count) - int(second)) / obs)
        if int(count) < int(min_support) or frac < float(min_fraction) or margin < float(min_margin_fraction):
            labels_pred[int(tube_id)] = next_unknown
            next_unknown += 1
            unknown_count += 1
            if margin < float(min_margin_fraction):
                margin_unknown += 1
        else:
            labels_pred[int(tube_id)] = int(comp)
    labeled = sum(1 for value in gt_labels.values() if int(value) > 0)
    return labels_pred, float(unknown_count / max(labeled, 1)), {"tube_margin_unknown": int(margin_unknown)}


def _evaluate_stage_scene(
    scene: str,
    nodes: list[RegionNode],
    support_by_region: dict[int, Counter[int]],
    support_by_tube: dict[int, Counter[int]],
    observation_count_by_tube: dict[int, int],
    gt_labels: dict[int, int],
    pair_counts: Counter[tuple[int, int]],
    support_sets: dict[int, set[int]],
    frame_rank: dict[int, int],
    diagnostics: dict[int, dict[str, Any]] | None = None,
    *,
    stage: str,
    min_support: int,
    min_fraction: float,
    shuffle_control: bool = False,
    seed: int = 0,
    public_stage: str | None = None,
) -> dict[str, Any]:
    assignment_stage = str(public_stage or stage)
    local_support_by_tube = support_by_tube
    local_obs = observation_count_by_tube
    local_pair_counts = pair_counts
    local_support_sets = support_sets
    if shuffle_control:
        shuffled_region, shuffled_tube, shuffled_obs = _shuffle_supports(support_by_region, seed=seed)
        local_support_by_tube = shuffled_tube
        local_obs = shuffled_obs
        local_support_sets = {idx: set(counter.keys()) for idx, counter in shuffled_region.items()}
        local_pair_counts = _support_pair_counts(nodes, shuffled_region, max_pairs_per_tube=20000, seed=seed, frame_rank=frame_rank)
    if stage in {"D5_mask_only", "F7_mask_only", "F6_no_temporal"}:
        components = [[idx] for idx in range(len(nodes))]
        edge_info = {"accepted_edges": 0, "candidate_edges": 0}
    elif stage == "D0_greedy_adjacent":
        edges = _filtered_edges(nodes, local_support_sets, local_pair_counts, frame_rank, max_delta=1, min_shared=1, min_jaccard=0.0)
        components, edge_info = _components_from_edges(nodes, edges, mode="greedy", frame_rank=frame_rank)
    elif stage in {"D1_bipartite_adjacent", "D2_flow_adjacent", "E0_adjacent_masklets"}:
        edges = _filtered_edges(nodes, local_support_sets, local_pair_counts, frame_rank, max_delta=1, min_shared=2, min_jaccard=0.0)
        components, edge_info = _components_from_edges(nodes, edges, mode="bipartite", frame_rank=frame_rank)
    elif stage == "D3_reactivation_skip1":
        edges = _filtered_edges(nodes, local_support_sets, local_pair_counts, frame_rank, max_delta=2, min_shared=2, min_jaccard=0.0)
        components, edge_info = _components_from_edges(nodes, edges, mode="flow_top1", frame_rank=frame_rank)
    elif stage == "D4_shuffled_d4rt":
        edges = _filtered_edges(nodes, local_support_sets, local_pair_counts, frame_rank, max_delta=1, min_shared=2, min_jaccard=0.0)
        components, edge_info = _components_from_edges(nodes, edges, mode="bipartite", frame_rank=frame_rank)
    elif stage == "E1_dt2_strict":
        edges = _filtered_edges(nodes, local_support_sets, local_pair_counts, frame_rank, max_delta=2, min_shared=2, min_jaccard=0.0)
        components, edge_info = _components_from_edges(nodes, edges, mode="bipartite", frame_rank=frame_rank)
    elif stage == "E2_dt4_medium":
        edges = _filtered_edges(nodes, local_support_sets, local_pair_counts, frame_rank, max_delta=4, min_shared=2, min_jaccard=0.0)
        components, edge_info = _components_from_edges(nodes, edges, mode="bipartite", frame_rank=frame_rank)
    elif stage == "E3_dt8_medium":
        edges = _filtered_edges(nodes, local_support_sets, local_pair_counts, frame_rank, max_delta=8, min_shared=1, min_jaccard=0.0)
        components, edge_info = _components_from_edges(nodes, edges, mode="bipartite", frame_rank=frame_rank)
    elif stage == "E4_long_range_track_closure":
        edges = _filtered_edges(nodes, local_support_sets, local_pair_counts, frame_rank, min_delta=1, max_delta=None, min_shared=2, min_jaccard=0.0)
        components, edge_info = _components_from_edges(nodes, edges, mode="bipartite", frame_rank=frame_rank, stable_only_long_range=True)
    elif stage == "E4a_chain_dt1_then_far_loose1":
        short_edges = _filtered_edges(nodes, local_support_sets, local_pair_counts, frame_rank, max_delta=1, min_shared=2, min_jaccard=0.0)
        closure_edges = _filtered_edges(nodes, local_support_sets, local_pair_counts, frame_rank, min_delta=9, max_delta=None, min_shared=1, min_jaccard=0.0)
        components, edge_info = _components_chain_then_closure(nodes, short_edges, closure_edges, frame_rank=frame_rank)
    elif stage == "E4a_chain_dt2_then_far_loose1":
        short_edges = _filtered_edges(nodes, local_support_sets, local_pair_counts, frame_rank, max_delta=2, min_shared=2, min_jaccard=0.0)
        closure_edges = _filtered_edges(nodes, local_support_sets, local_pair_counts, frame_rank, min_delta=9, max_delta=None, min_shared=1, min_jaccard=0.0)
        components, edge_info = _components_chain_then_closure(nodes, short_edges, closure_edges, frame_rank=frame_rank)
    elif stage in {
        "E4a_rgb085_chain_dt1_then_far_loose1",
        "E4a_rgb090_chain_dt1_then_far_loose1",
        "E4a_rgb095_chain_dt1_then_far_loose1",
        "E4c_rgb085_chain_dt1_all_edges",
        "E4c_rgb090_chain_dt1_all_edges",
        "E4d_rgb090_chain_dt1_component_split",
        "E4d_rgb095_chain_dt1_component_split",
        "E4e_rgb090_component_unknown087",
        "E4e_rgb090_component_unknown088",
        "E4e_rgb090_component_unknown089",
        "E4e_rgb090_component_unknown080",
        "E4e_rgb090_component_unknown085",
        "E4e_rgb090_component_unknown090",
        "E4f_rgb090_unknown080_small40",
        "E4f_rgb090_unknown080_small48",
        "E4f_rgb090_unknown080_small56",
        "E4f_rgb090_unknown080_small32",
        "E4f_rgb090_unknown080_small64",
        "E4g_rgb090_isolate080",
        "E4g_rgb090_isolate085",
        "E4g_rgb090_isolate080_small128",
        "E4h_rgb090_singleton080",
        "E4h_rgb090_singleton085",
        "E4h_rgb090_singleton080_small128",
    }:
        threshold = {
            "E4a_rgb085_chain_dt1_then_far_loose1": 0.85,
            "E4a_rgb090_chain_dt1_then_far_loose1": 0.90,
            "E4a_rgb095_chain_dt1_then_far_loose1": 0.95,
            "E4c_rgb085_chain_dt1_all_edges": 0.85,
            "E4c_rgb090_chain_dt1_all_edges": 0.90,
            "E4d_rgb090_chain_dt1_component_split": 0.90,
            "E4d_rgb095_chain_dt1_component_split": 0.95,
            "E4e_rgb090_component_unknown087": 0.90,
            "E4e_rgb090_component_unknown088": 0.90,
            "E4e_rgb090_component_unknown089": 0.90,
            "E4e_rgb090_component_unknown080": 0.90,
            "E4e_rgb090_component_unknown085": 0.90,
            "E4e_rgb090_component_unknown090": 0.90,
            "E4f_rgb090_unknown080_small40": 0.90,
            "E4f_rgb090_unknown080_small48": 0.90,
            "E4f_rgb090_unknown080_small56": 0.90,
            "E4f_rgb090_unknown080_small32": 0.90,
            "E4f_rgb090_unknown080_small64": 0.90,
            "E4g_rgb090_isolate080": 0.90,
            "E4g_rgb090_isolate085": 0.90,
            "E4g_rgb090_isolate080_small128": 0.90,
            "E4h_rgb090_singleton080": 0.90,
            "E4h_rgb090_singleton085": 0.90,
            "E4h_rgb090_singleton080_small128": 0.90,
        }[stage]
        short_edges = _filtered_edges(nodes, local_support_sets, local_pair_counts, frame_rank, max_delta=1, min_shared=2, min_jaccard=0.0)
        closure_edges = _filtered_edges(nodes, local_support_sets, local_pair_counts, frame_rank, min_delta=9, max_delta=None, min_shared=1, min_jaccard=0.0)
        rejected_rgb = 0
        if stage.startswith("E4c_"):
            short_edges, rejected_short_rgb = _filter_edges_by_rgb(short_edges, diagnostics, min_rgb_similarity=threshold)
            rejected_rgb += int(rejected_short_rgb)
        closure_edges, rejected_closure_rgb = _filter_edges_by_rgb(closure_edges, diagnostics, min_rgb_similarity=threshold)
        rejected_rgb += int(rejected_closure_rgb)
        components, edge_info = _components_chain_then_closure(nodes, short_edges, closure_edges, frame_rank=frame_rank)
        if stage.startswith(("E4d_", "E4e_", "E4f_", "E4g_", "E4h_")):
            components, split_info = _split_components_by_rgb(
                nodes,
                components,
                diagnostics,
                min_rgb_similarity=threshold,
            )
            edge_info.update(split_info)
        if stage.startswith(("E4e_", "E4f_")):
            unknown_threshold = {
                "E4e_rgb090_component_unknown087": 0.87,
                "E4e_rgb090_component_unknown088": 0.88,
                "E4e_rgb090_component_unknown089": 0.89,
                "E4e_rgb090_component_unknown080": 0.80,
                "E4e_rgb090_component_unknown085": 0.85,
                "E4e_rgb090_component_unknown090": 0.90,
                "E4f_rgb090_unknown080_small40": 0.80,
                "E4f_rgb090_unknown080_small48": 0.80,
                "E4f_rgb090_unknown080_small56": 0.80,
                "E4f_rgb090_unknown080_small32": 0.80,
                "E4f_rgb090_unknown080_small64": 0.80,
            }[stage]
            max_unknown_nodes = {
                "E4f_rgb090_unknown080_small40": 40,
                "E4f_rgb090_unknown080_small48": 48,
                "E4f_rgb090_unknown080_small56": 56,
                "E4f_rgb090_unknown080_small32": 32,
                "E4f_rgb090_unknown080_small64": 64,
            }.get(stage, 0)
            components, unknown_info = _drop_rgb_incoherent_components(
                nodes,
                components,
                diagnostics,
                min_pairwise_similarity=unknown_threshold,
                max_component_nodes=max_unknown_nodes,
            )
            edge_info.update(unknown_info)
            edge_info["rgb_unknown_similarity_threshold"] = float(unknown_threshold)
            edge_info["rgb_unknown_max_component_nodes"] = int(max_unknown_nodes)
        if stage.startswith("E4g_"):
            outlier_threshold = {
                "E4g_rgb090_isolate080": 0.80,
                "E4g_rgb090_isolate085": 0.85,
                "E4g_rgb090_isolate080_small128": 0.80,
            }[stage]
            max_outlier_nodes = {
                "E4g_rgb090_isolate080_small128": 128,
            }.get(stage, 0)
            components, outlier_info = _isolate_rgb_outlier_nodes(
                nodes,
                components,
                diagnostics,
                min_center_similarity=outlier_threshold,
                max_component_nodes=max_outlier_nodes,
            )
            edge_info.update(outlier_info)
            edge_info["rgb_outlier_similarity_threshold"] = float(outlier_threshold)
            edge_info["rgb_outlier_max_component_nodes"] = int(max_outlier_nodes)
        if stage.startswith("E4h_"):
            singleton_threshold = {
                "E4h_rgb090_singleton080": 0.80,
                "E4h_rgb090_singleton085": 0.85,
                "E4h_rgb090_singleton080_small128": 0.80,
            }[stage]
            max_singleton_nodes = {
                "E4h_rgb090_singleton080_small128": 128,
            }.get(stage, 0)
            components, singleton_info = _singleton_rgb_incoherent_components(
                nodes,
                components,
                diagnostics,
                min_center_similarity=singleton_threshold,
                max_component_nodes=max_singleton_nodes,
            )
            edge_info.update(singleton_info)
            edge_info["rgb_singleton_similarity_threshold"] = float(singleton_threshold)
            edge_info["rgb_singleton_max_component_nodes"] = int(max_singleton_nodes)
        edge_info["rejected_visual"] = int(rejected_rgb)
        edge_info["rgb_similarity_threshold"] = float(threshold)
    elif stage == "E4b_chain_dt4_then_far_loose1":
        short_edges = _filtered_edges(nodes, local_support_sets, local_pair_counts, frame_rank, max_delta=4, min_shared=2, min_jaccard=0.0)
        closure_edges = _filtered_edges(nodes, local_support_sets, local_pair_counts, frame_rank, min_delta=9, max_delta=None, min_shared=1, min_jaccard=0.0)
        components, edge_info = _components_chain_then_closure(nodes, short_edges, closure_edges, frame_rank=frame_rank)
    elif stage == "E5_direct_far_pair_negative":
        edges = _filtered_edges(nodes, local_support_sets, local_pair_counts, frame_rank, min_delta=9, max_delta=None, min_shared=1, min_jaccard=0.0)
        components, edge_info = _components_from_edges(nodes, edges, mode="greedy", frame_rank=frame_rank)
    elif stage == "E6_sparse_allpair_negative":
        edges = _filtered_edges(nodes, local_support_sets, local_pair_counts, frame_rank, min_delta=1, max_delta=None, min_shared=1, min_jaccard=0.0)
        components, edge_info = _components_from_edges(nodes, edges, mode="greedy", frame_rank=frame_rank)
    else:
        raise ValueError(stage)
    margin_unknown_threshold = {
        "F19_rgb090_component_split_margin005": 0.05,
        "F20_rgb090_component_split_margin010": 0.10,
        "F21_rgb090_component_split_margin015": 0.15,
        "F22_rgb090_component_split_margin020": 0.20,
    }.get(assignment_stage, 0.0)
    adaptive_fraction_threshold = 0.0
    adaptive_support_density = None
    adaptive_guarded_low_fraction = 0
    if assignment_stage == "F31_rgb090_component_split_adaptive_density010_frac060":
        adaptive_support_density = _safe_div(edge_info.get("candidate_edges"), len(nodes))
        if float(adaptive_support_density) >= 0.10:
            adaptive_fraction_threshold = 0.60
        else:
            adaptive_fraction_threshold = float(min_fraction)
            adaptive_guarded_low_fraction = 1
    if adaptive_fraction_threshold > 0.0:
        labels_pred, unknown_ratio = _labels_for_components(
            components,
            local_support_by_tube,
            local_obs,
            gt_labels,
            min_support=min_support,
            min_fraction=adaptive_fraction_threshold,
        )
        assign_info = {
            "tube_adaptive_min_fraction": float(adaptive_fraction_threshold),
            "tube_adaptive_support_density": float(adaptive_support_density or 0.0),
            "tube_adaptive_guarded_low_fraction": int(adaptive_guarded_low_fraction),
        }
    elif margin_unknown_threshold > 0.0:
        labels_pred, unknown_ratio, assign_info = _labels_for_components_margin_unknown(
            components,
            local_support_by_tube,
            local_obs,
            gt_labels,
            min_support=min_support,
            min_fraction=min_fraction,
            min_margin_fraction=margin_unknown_threshold,
        )
    else:
        labels_pred, unknown_ratio = _labels_for_components(
            components,
            local_support_by_tube,
            local_obs,
            gt_labels,
            min_support=min_support,
            min_fraction=min_fraction,
        )
        assign_info = {}
    metrics = _cluster_metrics(labels_pred, gt_labels)
    labeled_ids = [tid for tid in sorted(labels_pred) if int(gt_labels.get(int(tid), 0)) > 0]
    row = {
        "scene": scene,
        "stage": stage,
        "ARI": metrics.get("ari"),
        "purity": metrics.get("purity"),
        "completeness": metrics.get("completeness"),
        "unknown_tube_ratio": float(unknown_ratio),
        "labeled_tube_count": metrics.get("labeled_tube_count"),
        **_component_stats(nodes, components, frame_rank),
        **edge_info,
        **assign_info,
        "min_support": int(min_support),
        "min_fraction": float(min_fraction),
        "tube_margin_unknown_threshold": float(margin_unknown_threshold),
        "tube_adaptive_fraction_threshold": float(adaptive_fraction_threshold),
        "_labels_true": [int(gt_labels[int(tid)]) for tid in labeled_ids],
        "_labels_pred": [int(labels_pred[int(tid)]) for tid in labeled_ids],
    }
    return row


def _aggregate_stage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stage[str(row["stage"])].append(row)
    out = []
    for stage, items in sorted(by_stage.items()):
        all_true: list[int] = []
        all_pred: list[int] = []
        true_offset = 0
        pred_offset = 0
        for item in items:
            true_vals = [int(v) for v in item.get("_labels_true", [])]
            pred_vals = [int(v) for v in item.get("_labels_pred", [])]
            all_true.extend([int(v) + true_offset for v in true_vals])
            all_pred.extend([int(v) + pred_offset for v in pred_vals])
            true_offset += (max(true_vals) + 11) if true_vals else 11
            pred_offset += (max(pred_vals) + 11) if pred_vals else 11
        aggregate_metrics = _cluster_metrics(
            {idx: pred for idx, pred in enumerate(all_pred)},
            {idx: true for idx, true in enumerate(all_true)},
        )
        total_labeled = sum(int(row.get("labeled_tube_count") or 0) for row in items)
        weighted_unknown = sum(float(row.get("unknown_tube_ratio") or 0.0) * int(row.get("labeled_tube_count") or 0) for row in items)
        row = {
            "stage": stage,
            "ARI": aggregate_metrics.get("ari"),
            "purity": aggregate_metrics.get("purity"),
            "completeness": aggregate_metrics.get("completeness"),
            "unknown_tube_ratio": _safe_div(weighted_unknown, total_labeled),
            "scene0081_ARI": next((row.get("ARI") for row in items if row.get("scene") == "scene0081_01"), None),
            "masklet_span_mean": _mean([row.get("masklet_temporal_span_mean") for row in items]),
            "same_frame_cannot_link_violations": sum(int(row.get("same_frame_cannot_link_violations") or 0) for row in items),
            "accepted_expansions": sum(int(row.get("accepted_edges") or 0) for row in items),
            "new_edges": sum(int(row.get("candidate_edges") or 0) for row in items),
            "rejected_conflict": sum(int(row.get("rejected_same_frame_conflict") or 0) for row in items),
            "rejected_visual": sum(int(row.get("rejected_visual") or 0) for row in items),
            "rgb_split_components": sum(int(row.get("rgb_split_components") or 0) for row in items),
            "rgb_split_new_components": sum(int(row.get("rgb_split_new_components") or 0) for row in items),
            "rgb_unknown_components": sum(int(row.get("rgb_unknown_components") or 0) for row in items),
            "rgb_unknown_nodes": sum(int(row.get("rgb_unknown_nodes") or 0) for row in items),
            "rgb_outlier_components": sum(int(row.get("rgb_outlier_components") or 0) for row in items),
            "rgb_outlier_nodes": sum(int(row.get("rgb_outlier_nodes") or 0) for row in items),
            "rgb_singleton_components": sum(int(row.get("rgb_singleton_components") or 0) for row in items),
            "rgb_singleton_nodes": sum(int(row.get("rgb_singleton_nodes") or 0) for row in items),
            "tube_margin_unknown": sum(int(row.get("tube_margin_unknown") or 0) for row in items),
            "tube_adaptive_guarded_low_fraction": sum(
                int(row.get("tube_adaptive_guarded_low_fraction") or 0) for row in items
            ),
            "rejected_no_d4rt_support": None,
        }
        row.update(_gate(row))
        out.append(row)
    return out


def _public_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in row.items() if not str(key).startswith("_")} for row in rows]


def _control_summary(stage_rows: list[dict[str, Any]], best_stage: str) -> dict[str, Any]:
    lookup = {str(row["stage"]): row for row in stage_rows}
    best = lookup.get(best_stage, {})
    shuffled = lookup.get("D4_shuffled_d4rt", {})
    no_temporal = lookup.get("F6_no_temporal") or lookup.get("D5_mask_only", {})
    mask_only = lookup.get("F7_mask_only") or lookup.get("D5_mask_only", {})
    ari = best.get("ARI")
    controls = {
        "best_stage": best_stage,
        "best_ARI": ari,
        "shuffled_ARI": shuffled.get("ARI"),
        "no_temporal_ARI": no_temporal.get("ARI"),
        "mask_only_ARI": mask_only.get("ARI"),
        "real_minus_shuffled": None,
        "real_minus_no_temporal": None,
        "real_minus_mask_only": None,
        "real_minus_v36_chain2": None,
        "control_gate_pass": False,
    }
    if ari is not None:
        if shuffled.get("ARI") is not None:
            controls["real_minus_shuffled"] = float(ari - shuffled["ARI"])
        if no_temporal.get("ARI") is not None:
            controls["real_minus_no_temporal"] = float(ari - no_temporal["ARI"])
        if mask_only.get("ARI") is not None:
            controls["real_minus_mask_only"] = float(ari - mask_only["ARI"])
        controls["real_minus_v36_chain2"] = float(ari - V36_CHAIN2_ARI)
    controls["control_gate_pass"] = bool(
        controls["real_minus_shuffled"] is not None
        and controls["real_minus_shuffled"] >= 0.20
        and controls["real_minus_no_temporal"] is not None
        and controls["real_minus_no_temporal"] >= 0.05
        and controls["real_minus_mask_only"] is not None
        and controls["real_minus_mask_only"] >= 0.05
        and controls["real_minus_v36_chain2"] is not None
        and controls["real_minus_v36_chain2"] >= 0.05
    )
    return controls


def _mask_source_uses_gt_for_prediction(source: str, mode: str) -> bool:
    text = f"{source}:{mode}".lower()
    return "oracle" in text or "gt_split" in text


def _learned_diagnostic(pair_rows: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import f1_score, roc_auc_score
    except Exception as exc:  # pragma: no cover - optional dependency path
        rows = [{"fold": "ALL", "target": "adjacent_or_short_pair_same_object", "status": f"not_run:{type(exc).__name__}"}]
        _write_csv(out_dir / "learned_diagnostic_summary.csv", rows)
        return rows
    labeled = [
        row
        for row in pair_rows
        if row.get("diagnostic_labeled_pair") and int(row.get("delta_t") or 0) <= 4 and row.get("source") in {"d4rt_support", "all_pair_sample"}
    ]
    rows = []
    scenes = sorted({str(row["scene"]) for row in labeled})
    for scene in scenes:
        train = [row for row in labeled if row["scene"] != scene]
        test = [row for row in labeled if row["scene"] == scene]
        if len(train) < 20 or len(test) < 10:
            rows.append({"fold": scene, "target": "adjacent_or_short_pair_same_object", "status": "not_enough_pairs"})
            continue
        def feats(items: list[dict[str, Any]]) -> np.ndarray:
            return np.asarray(
                [
                    [
                        float(row.get("shared_d4rt_tube_count") or 0.0),
                        float(row.get("shared_d4rt_jaccard") or 0.0),
                        float(row.get("delta_t") or 0.0),
                        float(row.get("rgb_similarity") if row.get("rgb_similarity") is not None else 0.0),
                    ]
                    for row in items
                ],
                dtype=np.float32,
            )
        y_train = np.asarray([1 if row.get("diagnostic_same_GT") else 0 for row in train], dtype=np.int64)
        y_test = np.asarray([1 if row.get("diagnostic_same_GT") else 0 for row in test], dtype=np.int64)
        if len(set(y_train.tolist())) < 2 or len(set(y_test.tolist())) < 2:
            rows.append({"fold": scene, "target": "adjacent_or_short_pair_same_object", "status": "single_class"})
            continue
        clf = RandomForestClassifier(n_estimators=120, max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=3701)
        clf.fit(feats(train), y_train)
        prob = clf.predict_proba(feats(test))[:, 1]
        pred = (prob >= 0.5).astype(np.int64)
        bins = np.linspace(0.0, 1.0, 11)
        ece = 0.0
        for left, right in zip(bins[:-1], bins[1:]):
            mask = (prob >= left) & (prob < right if right < 1.0 else prob <= right)
            if np.any(mask):
                ece += float(np.mean(mask)) * abs(float(np.mean(prob[mask])) - float(np.mean(y_test[mask])))
        rows.append(
            {
                "fold": scene,
                "target": "adjacent_or_short_pair_same_object",
                "status": "ok",
                "AUC": float(roc_auc_score(y_test, prob)),
                "F1": float(f1_score(y_test, pred)),
                "calibration_error": float(ece),
                "ARI": None,
                "purity": None,
                "completeness": None,
            }
        )
    ok = [row for row in rows if row.get("status") == "ok"]
    if ok:
        rows.append(
            {
                "fold": "MEAN",
                "target": "adjacent_or_short_pair_same_object",
                "status": "ok",
                "AUC": _mean([row.get("AUC") for row in ok]),
                "F1": _mean([row.get("F1") for row in ok]),
                "calibration_error": _mean([row.get("calibration_error") for row in ok]),
                "ARI": None,
                "purity": None,
                "completeness": None,
            }
        )
    _write_csv(out_dir / "learned_diagnostic_summary.csv", rows)
    _write_json(out_dir / "learned_diagnostic_summary.json", rows)
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    setattr(_load_masks, "max_regions_per_scene", int(args.max_regions_per_scene))
    scenes = _read_split(Path(args.split))
    root = Path(args.output_root)
    phase_b = root / "v37_delta_t_pair_attribution"
    phase_c = root / "v37_same_frame_objectlets"
    phase_d = root / "v37_adjacent_merge"
    phase_e = root / "v37_curriculum_diffusion"
    phase_f = root / "v37_tube_assignment"
    phase_g = root / "v37_scene0081_hardcase"
    phase_h = root / "v37_learned_diagnostic"
    final_dir = root / "v37_final_decision"
    for path in [phase_b, phase_c, phase_d, phase_e, phase_f, phase_g, phase_h, final_dir]:
        path.mkdir(parents=True, exist_ok=True)

    all_pair_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    stage_scene_rows: list[dict[str, Any]] = []
    scene0081_false_merge = []
    scene0081_false_cut = []
    manifests = {}

    for scene in scenes:
        nodes, labels_by_frame, mask_manifest = _load_masks(Path(args.mask_root), scene, args.source, args.mode, int(args.min_region_area))
        frame_rank = _frame_rank_map(labels_by_frame)
        tubes = _load_tubes(scene, args)
        gt_labels = _load_gt(scene, tubes, args)
        support_by_region, support_by_tube, obs_by_tube = _collect_observations(nodes, labels_by_frame, tubes, args)
        support_sets = {idx: set(counter.keys()) for idx, counter in support_by_region.items()}
        diagnostics, gt_area = _region_diagnostics(scene, nodes, labels_by_frame, compute_rgb=bool(args.compute_rgb))
        pair_counts = _support_pair_counts(
            nodes,
            support_by_region,
            max_pairs_per_tube=int(args.max_support_pairs_per_tube),
            seed=int(args.seed) + len(all_pair_rows),
            frame_rank=frame_rank,
        )
        support_pairs = list(pair_counts.keys())
        same_frame_pairs = _sample_same_frame_pairs(
            nodes,
            max_pairs_per_frame=int(args.max_same_frame_pairs_per_frame),
            seed=int(args.seed) + 11,
        )
        all_sample_pairs = _sample_all_pairs(nodes, max_pairs=int(args.max_allpair_samples_per_scene), seed=int(args.seed) + 23)
        shuffled_region, _, _ = _shuffle_supports(support_by_region, seed=int(args.seed) + 37)
        shuffled_pair_counts = _support_pair_counts(
            nodes,
            shuffled_region,
            max_pairs_per_tube=int(args.max_support_pairs_per_tube),
            seed=int(args.seed) + 41,
            frame_rank=frame_rank,
        )
        for pair in support_pairs:
            all_pair_rows.append(
                _pair_row(
                    scene=scene,
                    nodes=nodes,
                    support_sets=support_sets,
                    pair_counts=pair_counts,
                    diagnostics=diagnostics,
                    pair=pair,
                    source="d4rt_support",
                    frame_rank=frame_rank,
                )
            )
        for pair in same_frame_pairs:
            all_pair_rows.append(
                _pair_row(
                    scene=scene,
                    nodes=nodes,
                    support_sets=support_sets,
                    pair_counts=pair_counts,
                    diagnostics=diagnostics,
                    pair=pair,
                    source="same_frame_sample",
                    frame_rank=frame_rank,
                )
            )
        for pair in all_sample_pairs:
            all_pair_rows.append(
                _pair_row(
                    scene=scene,
                    nodes=nodes,
                    support_sets=support_sets,
                    pair_counts=pair_counts,
                    diagnostics=diagnostics,
                    pair=pair,
                    source="all_pair_sample",
                    frame_rank=frame_rank,
                )
            )
        shuffled_sets = {idx: set(counter.keys()) for idx, counter in shuffled_region.items()}
        for pair in list(shuffled_pair_counts.keys())[: int(args.max_shuffled_pair_rows_per_scene)]:
            all_pair_rows.append(
                _pair_row(
                    scene=scene,
                    nodes=nodes,
                    support_sets=shuffled_sets,
                    pair_counts=shuffled_pair_counts,
                    diagnostics=diagnostics,
                    pair=pair,
                    source="shuffled_d4rt_support",
                    frame_rank=frame_rank,
                )
            )
        for pair in all_sample_pairs[: int(args.max_allpair_samples_per_scene) // 2]:
            row = _pair_row(
                scene=scene,
                nodes=nodes,
                support_sets={},
                pair_counts=Counter(),
                diagnostics=diagnostics,
                pair=pair,
                source="no_temporal_sample",
                frame_rank=frame_rank,
            )
            all_pair_rows.append(row)

        all_nodes = {int(node.node_id) for node in nodes}
        supported = {idx for idx, tubeset in support_sets.items() if len(tubeset) >= 1}
        supported2 = {idx for idx, tubeset in support_sets.items() if len(tubeset) >= 2}
        compact_supported = {idx for idx in supported if int(nodes[idx].area) >= int(args.min_region_area) and int(nodes[idx].area) <= int(args.max_seed_area)}
        seed_rows.extend(
            [
                _seed_metrics(scene, nodes, labels_by_frame, diagnostics, gt_area, all_nodes, variant="C0_raw_watershed_all_masks", status="ok"),
                _seed_metrics(
                    scene,
                    nodes,
                    labels_by_frame,
                    diagnostics,
                    gt_area,
                    set(),
                    variant="C1_watershed_plus_DINO_intra_mask_split",
                    status="not_run_no_full_frame_dino_split_artifact",
                ),
                _seed_metrics(
                    scene,
                    nodes,
                    labels_by_frame,
                    diagnostics,
                    gt_area,
                    all_nodes,
                    variant="C2_watershed_boundary_split",
                    status="proxy_same_as_existing_watershed_boundaries",
                ),
                _seed_metrics(
                    scene,
                    nodes,
                    labels_by_frame,
                    diagnostics,
                    gt_area,
                    supported2,
                    variant="C3_D4RT_tube_seeded_supported2",
                    status="ok_method_filter_by_d4rt_support_count",
                ),
                _seed_metrics(
                    scene,
                    nodes,
                    labels_by_frame,
                    diagnostics,
                    gt_area,
                    compact_supported,
                    variant="C4_hybrid_boundary_D4RT_compact",
                    status="ok_method_filter_by_d4rt_support_and_area",
                ),
            ]
        )
        stage_specs = [
            ("D0_greedy_adjacent", 1, 0.0, False),
            ("D1_bipartite_adjacent", 1, 0.0, False),
            ("D2_flow_adjacent", 1, 0.0, False),
            ("D3_reactivation_skip1", 1, 0.0, False),
            ("D4_shuffled_d4rt", 1, 0.0, True),
            ("D5_mask_only", 1, 0.0, False),
            ("E0_adjacent_masklets", 1, 0.0, False),
            ("E1_dt2_strict", 1, 0.0, False),
            ("E2_dt4_medium", 1, 0.0, False),
            ("E3_dt8_medium", 1, 0.0, False),
            ("E4_long_range_track_closure", 1, 0.0, False),
            ("E4a_chain_dt1_then_far_loose1", 1, 0.0, False),
            ("E4a_chain_dt2_then_far_loose1", 1, 0.0, False),
            ("E4a_rgb085_chain_dt1_then_far_loose1", 1, 0.0, False),
            ("E4a_rgb090_chain_dt1_then_far_loose1", 1, 0.0, False),
            ("E4a_rgb095_chain_dt1_then_far_loose1", 1, 0.0, False),
            ("E4c_rgb085_chain_dt1_all_edges", 1, 0.0, False),
            ("E4c_rgb090_chain_dt1_all_edges", 1, 0.0, False),
            ("E4d_rgb090_chain_dt1_component_split", 1, 0.0, False),
            ("E4d_rgb095_chain_dt1_component_split", 1, 0.0, False),
            ("E4e_rgb090_component_unknown087", 1, 0.05, False),
            ("E4e_rgb090_component_unknown088", 1, 0.05, False),
            ("E4e_rgb090_component_unknown089", 1, 0.05, False),
            ("E4e_rgb090_component_unknown080", 1, 0.05, False),
            ("E4e_rgb090_component_unknown085", 1, 0.05, False),
            ("E4e_rgb090_component_unknown090", 1, 0.05, False),
            ("E4f_rgb090_unknown080_small40", 1, 0.05, False),
            ("E4f_rgb090_unknown080_small48", 1, 0.05, False),
            ("E4f_rgb090_unknown080_small56", 1, 0.05, False),
            ("E4f_rgb090_unknown080_small32", 1, 0.05, False),
            ("E4f_rgb090_unknown080_small64", 1, 0.05, False),
            ("E4g_rgb090_isolate080", 1, 0.05, False),
            ("E4g_rgb090_isolate085", 1, 0.05, False),
            ("E4g_rgb090_isolate080_small128", 1, 0.05, False),
            ("E4h_rgb090_singleton080", 1, 0.05, False),
            ("E4h_rgb090_singleton085", 1, 0.05, False),
            ("E4h_rgb090_singleton080_small128", 1, 0.05, False),
            ("E4b_chain_dt4_then_far_loose1", 1, 0.0, False),
            ("E5_direct_far_pair_negative", 1, 0.0, False),
            ("E6_sparse_allpair_negative", 1, 0.0, False),
            ("F0_containment_only", 1, 0.0, False),
            ("F1_containment_visibility", 1, 0.05, False),
            ("F2_F1_plus_DINO_similarity_unavailable", 1, 0.05, False),
            ("F3_temporal_chain_consistency", 2, 0.10, False),
            ("F4_F3_unknown_state", 2, 0.25, False),
            ("F5_shuffled_D4RT", 1, 0.05, True),
            ("F6_no_temporal", 1, 0.05, False),
            ("F7_mask_only", 1, 0.05, False),
            ("F8_direct_allpair_baseline", 1, 0.0, False),
            ("F9_rgb090_chain_dt1_support1_frac005", 1, 0.05, False),
            ("F10_rgb090_chain_dt1_support2_frac005", 2, 0.05, False),
            ("F11_rgb090_chain_dt1_support2_frac010", 2, 0.10, False),
            ("F12_rgb090_chain_dt1_support1_frac010", 1, 0.10, False),
            ("F13_rgb090_chain_dt1_support1_frac015", 1, 0.15, False),
            ("F14_rgb090_chain_dt1_support1_frac020", 1, 0.20, False),
            ("F15_rgb085_all_edges_support1_frac005", 1, 0.05, False),
            ("F16_rgb090_all_edges_support1_frac005", 1, 0.05, False),
            ("F17_rgb090_component_split_support1_frac005", 1, 0.05, False),
            ("F18_rgb095_component_split_support1_frac005", 1, 0.05, False),
            ("F19_rgb090_component_split_margin005", 1, 0.05, False),
            ("F20_rgb090_component_split_margin010", 1, 0.05, False),
            ("F21_rgb090_component_split_margin015", 1, 0.05, False),
            ("F22_rgb090_component_split_margin020", 1, 0.05, False),
            ("F23_rgb090_component_split_support1_frac030", 1, 0.30, False),
            ("F24_rgb090_component_split_support1_frac040", 1, 0.40, False),
            ("F25_rgb090_component_split_support1_frac050", 1, 0.50, False),
            ("F26_rgb090_component_split_support1_frac060", 1, 0.60, False),
            ("F27_rgb090_component_split_support1_frac070", 1, 0.70, False),
            ("F28_rgb090_component_split_support1_frac080", 1, 0.80, False),
            ("F29_rgb090_component_split_support1_frac085", 1, 0.85, False),
            ("F30_rgb090_component_split_support1_frac090", 1, 0.90, False),
            ("F31_rgb090_component_split_adaptive_density010_frac060", 1, 0.05, False),
        ]
        stage_alias = {
            "F0_containment_only": "E0_adjacent_masklets",
            "F1_containment_visibility": "E1_dt2_strict",
            "F2_F1_plus_DINO_similarity_unavailable": "E1_dt2_strict",
            "F3_temporal_chain_consistency": "E2_dt4_medium",
            "F4_F3_unknown_state": "E2_dt4_medium",
            "F5_shuffled_D4RT": "D4_shuffled_d4rt",
            "F8_direct_allpair_baseline": "E6_sparse_allpair_negative",
            "F9_rgb090_chain_dt1_support1_frac005": "E4a_rgb090_chain_dt1_then_far_loose1",
            "F10_rgb090_chain_dt1_support2_frac005": "E4a_rgb090_chain_dt1_then_far_loose1",
            "F11_rgb090_chain_dt1_support2_frac010": "E4a_rgb090_chain_dt1_then_far_loose1",
            "F12_rgb090_chain_dt1_support1_frac010": "E4a_rgb090_chain_dt1_then_far_loose1",
            "F13_rgb090_chain_dt1_support1_frac015": "E4a_rgb090_chain_dt1_then_far_loose1",
            "F14_rgb090_chain_dt1_support1_frac020": "E4a_rgb090_chain_dt1_then_far_loose1",
            "F15_rgb085_all_edges_support1_frac005": "E4c_rgb085_chain_dt1_all_edges",
            "F16_rgb090_all_edges_support1_frac005": "E4c_rgb090_chain_dt1_all_edges",
            "F17_rgb090_component_split_support1_frac005": "E4d_rgb090_chain_dt1_component_split",
            "F18_rgb095_component_split_support1_frac005": "E4d_rgb095_chain_dt1_component_split",
            "F19_rgb090_component_split_margin005": "E4d_rgb090_chain_dt1_component_split",
            "F20_rgb090_component_split_margin010": "E4d_rgb090_chain_dt1_component_split",
            "F21_rgb090_component_split_margin015": "E4d_rgb090_chain_dt1_component_split",
            "F22_rgb090_component_split_margin020": "E4d_rgb090_chain_dt1_component_split",
            "F23_rgb090_component_split_support1_frac030": "E4d_rgb090_chain_dt1_component_split",
            "F24_rgb090_component_split_support1_frac040": "E4d_rgb090_chain_dt1_component_split",
            "F25_rgb090_component_split_support1_frac050": "E4d_rgb090_chain_dt1_component_split",
            "F26_rgb090_component_split_support1_frac060": "E4d_rgb090_chain_dt1_component_split",
            "F27_rgb090_component_split_support1_frac070": "E4d_rgb090_chain_dt1_component_split",
            "F28_rgb090_component_split_support1_frac080": "E4d_rgb090_chain_dt1_component_split",
            "F29_rgb090_component_split_support1_frac085": "E4d_rgb090_chain_dt1_component_split",
            "F30_rgb090_component_split_support1_frac090": "E4d_rgb090_chain_dt1_component_split",
            "F31_rgb090_component_split_adaptive_density010_frac060": "E4d_rgb090_chain_dt1_component_split",
        }
        for stage, min_support, min_fraction, shuffled in stage_specs:
            eval_stage = stage_alias.get(stage, stage)
            row = _evaluate_stage_scene(
                scene,
                nodes,
                support_by_region,
                support_by_tube,
                obs_by_tube,
                gt_labels,
                pair_counts,
                support_sets,
                frame_rank,
                diagnostics,
                stage=eval_stage,
                min_support=min_support,
                min_fraction=min_fraction,
                shuffle_control=shuffled,
                seed=int(args.seed) + 71,
                public_stage=stage,
            )
            row["stage"] = stage
            row["effective_stage"] = eval_stage
            stage_scene_rows.append(row)
        if scene == "scene0081_01":
            scene_rows = [row for row in all_pair_rows if row.get("scene") == scene and row.get("diagnostic_labeled_pair")]
            scene0081_false_merge = sorted(
                [row for row in scene_rows if row.get("diagnostic_diff_GT")],
                key=lambda row: (float(row.get("shared_d4rt_jaccard") or 0.0), int(row.get("shared_d4rt_tube_count") or 0)),
                reverse=True,
            )[:50]
            scene0081_false_cut = sorted(
                [row for row in scene_rows if row.get("diagnostic_same_GT")],
                key=lambda row: (float(row.get("shared_d4rt_jaccard") or 0.0), int(row.get("shared_d4rt_tube_count") or 0)),
            )[:50]
        manifests[scene] = {
            **mask_manifest,
            "tube_count": int(len(tubes)),
            "gt_labeled_tube_count": int(sum(1 for value in gt_labels.values() if int(value) > 0)),
            "support_pair_count": int(len(support_pairs)),
            "same_frame_pair_sample_count": int(len(same_frame_pairs)),
            "all_pair_sample_count": int(len(all_sample_pairs)),
            "frame_delta_definition": "rank distance in sorted available mask frames, not raw global frame-id difference",
            "first_mask_frames": sorted(labels_by_frame)[:8],
        }

    pair_summary = _summarize_pair_rows(all_pair_rows)
    seed_summary = _aggregate_seed_rows(seed_rows)
    stage_summary = _aggregate_stage_rows(stage_scene_rows)
    best, selection = _select_best_stage(stage_summary)
    best_stage = str(best.get("stage") or "")
    controls = _control_summary(stage_summary, best_stage) if best_stage else {}
    source_uses_gt_for_prediction = _mask_source_uses_gt_for_prediction(args.source, args.mode)
    best_uses_rgb_for_prediction = "rgb" in best_stage.lower()
    final = {
        "plan": "docs/stream4d_v37_temporal_curriculum_masklet_plan.md",
        "final_status": "UNKNOWN",
        "best_stage": best_stage,
        "best_metrics": best,
        "selection": selection,
        "controls": controls,
        "pass_3D_gate": bool(best.get("pass_3D_gate")),
        "pass_controls": bool(controls.get("control_gate_pass")),
        "allowed_4d": bool((not source_uses_gt_for_prediction) and best.get("pass_3D_gate") and controls.get("control_gate_pass")),
        "allowed_ap": bool((not source_uses_gt_for_prediction) and best.get("pass_3D_gate") and controls.get("control_gate_pass")),
        "phaseB_hypothesis": {},
        "manifests": manifests,
        "manifest": {
            "is_method_result": bool(best) and not source_uses_gt_for_prediction,
            "is_diagnostic_only": bool(source_uses_gt_for_prediction),
            "forbidden_for_method_table": ["GT instance labels in mask source"] if source_uses_gt_for_prediction else [],
            "uses_gt_for_prediction": bool(source_uses_gt_for_prediction),
            "uses_gt_for_diagnostic_labels": True,
            "uses_rgb_for_prediction": bool(best_uses_rgb_for_prediction),
            "uses_rgbd_for_prediction": False,
            "uses_pose_for_prediction": False,
            "uses_scannet_mesh_for_prediction": False,
            "uses_eval_sim3_for_prediction": False,
            "uses_d4rt_self_sim3": True,
            "uses_frozen_visual_backbone": False,
            "visual_backbone_name": "none; DINO split unavailable in full-frame all_masks artifacts",
            "mask_source": f"{args.source}:{args.mode}",
            "temporal_curriculum_enabled": True,
            "temporal_stage": best_stage,
            "geometry_field": (
                "D4RT uv/visibility/confidence support plus RGB mean closure gate"
                if best_uses_rgb_for_prediction
                else "D4RT uv/visibility/confidence support; canonical geometry not used for prediction score"
            ),
            "coordinate_frame": "image uv for tube-to-mask support",
            "alignment_source": "d4rt_self_sim3 tube cache",
        },
    }
    lookup = {row["delta_bin"]: row for row in pair_summary}
    b1, b5, b6 = lookup.get("B1_dt1_adjacent", {}), lookup.get("B5_dt_gt8_far", {}), lookup.get("B6_all_pair_sample", {})
    final["phaseB_hypothesis"] = {
        "dt1_same_GT_minus_far": None
        if b1.get("same_GT_ratio") is None or b5.get("same_GT_ratio") is None
        else float(b1.get("same_GT_ratio") - b5.get("same_GT_ratio")),
        "dt1_false_merge_minus_far": None
        if b1.get("false_merge_rate") is None or b5.get("false_merge_rate") is None
        else float(b1.get("false_merge_rate") - b5.get("false_merge_rate")),
        "dt1_merge_AUC_minus_allpair": None
        if b1.get("merge_AUC") is None or b6.get("merge_AUC") is None
        else float(b1.get("merge_AUC") - b6.get("merge_AUC")),
        "pass_min_evidence": False,
    }
    phaseb = final["phaseB_hypothesis"]
    phaseb["pass_min_evidence"] = bool(
        phaseb["dt1_same_GT_minus_far"] is not None
        and phaseb["dt1_same_GT_minus_far"] >= 0.10
        and phaseb["dt1_false_merge_minus_far"] is not None
        and phaseb["dt1_false_merge_minus_far"] <= -0.10
        and phaseb["dt1_merge_AUC_minus_allpair"] is not None
        and phaseb["dt1_merge_AUC_minus_allpair"] >= 0.05
    )
    if final["pass_3D_gate"] and final["pass_controls"]:
        final["final_status"] = "GO_3D_TEMPORAL_CURRICULUM"
    elif not phaseb["pass_min_evidence"]:
        final["final_status"] = "NO_GO_PHASEB_TEMPORAL_RELATION_NOT_PROVEN"
    elif not final["pass_3D_gate"]:
        final["final_status"] = "NO_GO_3D_GATE_FAILED"
    else:
        final["final_status"] = "NO_GO_CONTROL_GATE_FAILED"

    _write_csv(phase_b / "delta_t_pair_rows.csv", all_pair_rows)
    _write_json(phase_b / "delta_t_pair_rows.json", all_pair_rows[: int(args.max_pair_rows_json)])
    _write_csv(phase_b / "delta_t_pair_summary.csv", pair_summary)
    _write_json(phase_b / "delta_t_pair_summary.json", pair_summary)
    _write_csv(phase_c / "same_frame_seed_scene_rows.csv", seed_rows)
    _write_csv(phase_c / "same_frame_seed_summary.csv", seed_summary)
    _write_json(phase_c / "same_frame_seed_summary.json", seed_summary)
    _write_csv(phase_d / "adjacent_merge_scene_rows.csv", _public_rows([row for row in stage_scene_rows if str(row["stage"]).startswith("D")]))
    _write_csv(phase_e / "curriculum_stage_scene_rows.csv", _public_rows([row for row in stage_scene_rows if str(row["stage"]).startswith("E")]))
    _write_csv(phase_f / "tube_assignment_scene_rows.csv", _public_rows([row for row in stage_scene_rows if str(row["stage"]).startswith("F")]))
    _write_csv(phase_e / "curriculum_stage_summary.csv", [row for row in stage_summary if str(row["stage"]).startswith("E")])
    _write_csv(phase_f / "tube_assignment_summary.csv", [row for row in stage_summary if str(row["stage"]).startswith("F")])
    _write_csv(phase_d / "adjacent_merge_summary.csv", [row for row in stage_summary if str(row["stage"]).startswith("D")])
    _write_json(phase_d / "adjacent_merge_summary.json", [row for row in stage_summary if str(row["stage"]).startswith("D")])
    _write_json(phase_e / "curriculum_stage_summary.json", [row for row in stage_summary if str(row["stage"]).startswith("E")])
    _write_json(phase_f / "tube_assignment_summary.json", [row for row in stage_summary if str(row["stage"]).startswith("F")])
    _write_csv(phase_g / "scene0081_false_merge_examples.csv", scene0081_false_merge)
    _write_csv(phase_g / "scene0081_false_cut_examples.csv", scene0081_false_cut)
    _write_json(phase_g / "scene0081_summary.json", {
        "scene0081_false_merge_example_count": len(scene0081_false_merge),
        "scene0081_false_cut_example_count": len(scene0081_false_cut),
        "scene0081_stage_rows": _public_rows([row for row in stage_scene_rows if row.get("scene") == "scene0081_01"]),
    })
    learned_rows = _learned_diagnostic(all_pair_rows, phase_h)
    final["learned_diagnostic"] = {
        "mean_AUC": next((row.get("AUC") for row in learned_rows if row.get("fold") == "MEAN"), None),
        "mean_F1": next((row.get("F1") for row in learned_rows if row.get("fold") == "MEAN"), None),
        "note": "LOSO pair-classifier diagnostic only; no learned solver ARI was promoted as method success.",
    }
    _write_json(final_dir / "decision_summary.json", final)
    _write_json(final_dir / "all_stage_summary.json", stage_summary)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v37 temporal-curriculum masklet diagnostics.")
    parser.add_argument("--mask-root", default="outputs/audit/v36_external_mask_source_all")
    parser.add_argument("--source", default="watershed")
    parser.add_argument("--mode", default="all_masks")
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v21_3_occupancy_d5_warmstart64_probe5_r1")
    parser.add_argument("--output-root", default="outputs/audit")
    parser.add_argument("--max-tubes-per-window", type=int, default=640)
    parser.add_argument("--image-width", type=int, default=1296)
    parser.add_argument("--image-height", type=int, default=968)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--min-region-area", type=int, default=64)
    parser.add_argument("--max-regions-per-scene", type=int, default=0)
    parser.add_argument("--max-support-pairs-per-tube", type=int, default=20000)
    parser.add_argument("--max-same-frame-pairs-per-frame", type=int, default=200)
    parser.add_argument("--max-allpair-samples-per-scene", type=int, default=30000)
    parser.add_argument("--max-shuffled-pair-rows-per-scene", type=int, default=30000)
    parser.add_argument("--max-pair-rows-json", type=int, default=20000)
    parser.add_argument("--max-seed-area", type=int, default=200000)
    parser.add_argument("--compute-rgb", action="store_true")
    parser.add_argument("--seed", type=int, default=3701)
    args = parser.parse_args()
    print(json.dumps(_json_safe(run(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
