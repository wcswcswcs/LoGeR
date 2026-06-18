from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from typing import Any

import numpy as np


class UnionFind:
    def __init__(self, items: list[int]) -> None:
        self.parent = {int(item): int(item) for item in items}

    def find(self, x: int) -> int:
        x = int(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(int(a)), self.find(int(b))
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True

    def labels(self) -> dict[int, int]:
        root_to_idx: dict[int, int] = {}
        out: dict[int, int] = {}
        for item in sorted(self.parent):
            root = self.find(item)
            if root not in root_to_idx:
                root_to_idx[root] = len(root_to_idx)
            out[item] = root_to_idx[root]
        return out


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _bool(value: Any) -> bool:
    return str(value).lower() == "true"


def _int(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key)
    if value in (None, ""):
        return int(default)
    return int(float(value))


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, ""):
        return float(default)
    return float(value)


def _ari(labels_true: list[int], labels_pred: list[int]) -> float | None:
    if len(labels_true) < 2:
        return None
    n = len(labels_true)
    contingency: dict[tuple[int, int], int] = defaultdict(int)
    true_counts: Counter[int] = Counter(labels_true)
    pred_counts: Counter[int] = Counter(labels_pred)
    for true, pred in zip(labels_true, labels_pred):
        contingency[(int(true), int(pred))] += 1
    sum_comb = sum(comb(v, 2) for v in contingency.values() if v >= 2)
    sum_true = sum(comb(v, 2) for v in true_counts.values() if v >= 2)
    sum_pred = sum(comb(v, 2) for v in pred_counts.values() if v >= 2)
    total = comb(n, 2)
    expected = sum_true * sum_pred / total if total else 0.0
    max_index = 0.5 * (sum_true + sum_pred)
    denom = max_index - expected
    if denom == 0.0:
        return 0.0
    return float((sum_comb - expected) / denom)


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]
    return float(np.mean(vals)) if vals else None


def _signed_score(row: dict[str, Any]) -> float:
    merge = _float(row, "merge_score")
    boundary_safe = _int(row, "boundary_safe_count")
    same_mask = _int(row, "same_mask_count")
    neg = _negative_count(row)
    return float(merge + 0.04 * np.log1p(boundary_safe + same_mask) - 0.05 * np.log1p(neg))


def _appearance_motion_bonus(row: dict[str, Any]) -> float:
    bonus = 0.0
    appearance = row.get("appearance_similarity")
    if appearance not in (None, ""):
        app = _float(row, "appearance_similarity")
        bonus += 0.60 * (app - 0.96)
    motion = row.get("motion_consistency")
    if motion not in (None, ""):
        mot = float(np.clip(_float(row, "motion_consistency"), -1.0, 1.0))
        bonus += 0.05 * mot
    return float(bonus)


def _negative_count(row: dict[str, Any]) -> int:
    return (
        _int(row, "boundary_cross_count")
        + _int(row, "same_frame_cannot_link_count")
        + _int(row, "visible_outside_count")
    )


def _strong_local_core_edge(row: dict[str, Any]) -> bool:
    if not _bool(row.get("predicted_merge", False)) or not _bool(row.get("same_chunk", False)):
        return False
    if _int(row, "boundary_cross_count") > 0 or _int(row, "same_frame_cannot_link_count") > 0:
        return False
    if _int(row, "visible_outside_count") > _int(row, "same_mask_count"):
        return False
    return _int(row, "boundary_safe_count") > 0 and _signed_score(row) > 0.05


def _local_grow_edge(row: dict[str, Any]) -> bool:
    if not _bool(row.get("predicted_merge", False)) or not _bool(row.get("same_chunk", False)):
        return False
    if _int(row, "boundary_cross_count") > 0 or _int(row, "same_frame_cannot_link_count") > 0:
        return False
    if _int(row, "visible_outside_count") > max(_int(row, "same_mask_count"), 1):
        return False
    return _signed_score(row) > 0.12


def _local_appearance_motion_edge(row: dict[str, Any]) -> bool:
    if not _bool(row.get("predicted_merge", False)) or not _bool(row.get("same_chunk", False)):
        return False
    if _int(row, "boundary_cross_count") > 0 or _int(row, "same_frame_cannot_link_count") > 0:
        return False
    if _int(row, "visible_outside_count") > max(_int(row, "same_mask_count"), 1):
        return False
    appearance = row.get("appearance_similarity")
    if appearance not in (None, "") and _float(row, "appearance_similarity") < 0.94:
        return False
    return _signed_score(row) + _appearance_motion_bonus(row) > 0.12


def _fringe_assignment_edges(uf: UnionFind, rows: list[dict[str, Any]]) -> list[tuple[float, int, int]]:
    labels = uf.labels()
    sizes: Counter[int] = Counter(labels.values())
    best: dict[int, tuple[float, int, int]] = {}
    for row in rows:
        if not _bool(row.get("predicted_merge", False)) or not _bool(row.get("same_chunk", False)):
            continue
        if _int(row, "boundary_cross_count") > 0 or _int(row, "same_frame_cannot_link_count") > 0:
            continue
        if _int(row, "visible_outside_count") > max(_int(row, "same_mask_count"), 1):
            continue
        ti, tj = _int(row, "tube_i"), _int(row, "tube_j")
        ci, cj = labels.get(ti), labels.get(tj)
        if ci is None or cj is None or ci == cj:
            continue
        score = _signed_score(row)
        if score <= 0.04:
            continue
        candidates: list[tuple[int, int, int]] = []
        if sizes[ci] <= 2 and sizes[cj] >= 2:
            candidates.append((ci, ti, tj))
        if sizes[cj] <= 2 and sizes[ci] >= 2:
            candidates.append((cj, tj, ti))
        for small_comp, small_tube, large_tube in candidates:
            prev = best.get(small_comp)
            if prev is None or score > prev[0]:
                best[small_comp] = (score, small_tube, large_tube)
    return sorted(best.values(), reverse=True)


def _compact_labels(labels: dict[int, int]) -> dict[int, int]:
    root_to_idx: dict[int, int] = {}
    out: dict[int, int] = {}
    for tube_id, root in sorted(labels.items()):
        root = int(root)
        if root not in root_to_idx:
            root_to_idx[root] = len(root_to_idx)
        out[int(tube_id)] = root_to_idx[root]
    return out


def _ownership_assignment_labels(
    tube_ids: list[int],
    rows: list[dict[str, Any]],
    *,
    use_appearance_motion: bool = False,
) -> tuple[dict[int, int], int]:
    """Attach fringe components to one strong local core without core-core union."""
    uf = UnionFind(tube_ids)
    kept = 0
    for row in rows:
        if _strong_local_core_edge(row):
            kept += int(uf.union(_int(row, "tube_i"), _int(row, "tube_j")))

    base_labels = uf.labels()
    sizes: Counter[int] = Counter(base_labels.values())
    core_components = {comp for comp, size in sizes.items() if size >= 2}
    if not core_components:
        return base_labels, kept

    scores: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if not _bool(row.get("predicted_merge", False)) or not _bool(row.get("same_chunk", False)):
            continue
        if _int(row, "boundary_cross_count") > 0 or _int(row, "same_frame_cannot_link_count") > 0:
            continue
        if _int(row, "visible_outside_count") > max(_int(row, "same_mask_count"), 1):
            continue
        ti, tj = _int(row, "tube_i"), _int(row, "tube_j")
        ci, cj = base_labels.get(ti), base_labels.get(tj)
        if ci is None or cj is None or ci == cj:
            continue
        row_score = _signed_score(row)
        if use_appearance_motion:
            row_score += _appearance_motion_bonus(row)
        if row_score <= 0.05:
            continue
        if ci not in core_components and cj in core_components:
            scores[ci][cj].append(row_score)
        if cj not in core_components and ci in core_components:
            scores[cj][ci].append(row_score)

    assigned: dict[int, int] = {}
    for small_comp, core_scores in scores.items():
        ranked: list[tuple[float, float, int, int]] = []
        for core_comp, vals in core_scores.items():
            support = len(vals)
            total = float(np.sum(vals))
            mean = float(np.mean(vals))
            ranked.append((total, mean, support, core_comp))
        ranked.sort(reverse=True)
        best_total, best_mean, best_support, best_core = ranked[0]
        second_total = ranked[1][0] if len(ranked) > 1 else 0.0
        has_margin = best_total >= second_total + 0.05
        # One strong edge can attach an isolated tube; otherwise require aggregate support.
        if (best_support >= 2 and best_mean >= 0.08 and has_margin) or (
            best_support == 1 and best_total >= 0.35 and has_margin
        ):
            assigned[small_comp] = best_core

    raw_labels: dict[int, int] = {}
    for tube_id, comp in base_labels.items():
        raw_labels[tube_id] = assigned.get(comp, comp)
    return _compact_labels(raw_labels), kept + len(assigned)


def _memory_labels_from_core_labels(
    core_labels: dict[int, int],
    rows: list[dict[str, Any]],
    variant: str,
) -> tuple[dict[int, int], int, int]:
    core_pair_rows: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _bool(row.get("same_chunk", False)):
            continue
        if not _bool(row.get("same_submap", False)):
            continue
        ci = core_labels.get(_int(row, "tube_i"))
        cj = core_labels.get(_int(row, "tube_j"))
        if ci is None or cj is None or ci == cj:
            continue
        core_pair_rows[tuple(sorted((ci, cj)))].append(row)
    candidates = []
    for pair, pair_rows in core_pair_rows.items():
        pos_count = sum(1 for row in pair_rows if _bool(row.get("predicted_merge", False)))
        if pos_count <= 0:
            continue
        neg_count = sum(_negative_count(row) for row in pair_rows)
        same_mask = sum(_int(row, "same_mask_count") for row in pair_rows)
        signed_mean = float(np.mean([_signed_score(row) for row in pair_rows]))
        merge_mean = float(np.mean([_float(row, "merge_score") for row in pair_rows]))
        candidates.append((signed_mean, merge_mean, pos_count, neg_count, same_mask, pair, pair_rows))
    candidates.sort(reverse=True)
    core_uf = UnionFind(sorted(set(core_labels.values())))
    memory_merge_count = 0
    for signed_mean, merge_mean, pos_count, neg_count, same_mask, pair, pair_rows in candidates:
        keep = False
        if variant == "H1_object_memory_shared":
            keep = neg_count == 0 and pos_count >= 1
        elif variant == "H2_object_memory_signed":
            keep = signed_mean > 0.15 and neg_count <= max(pos_count, same_mask, 1)
        elif variant == "H3_object_memory_conservative":
            keep = pos_count >= 2 and signed_mean > 0.20 and neg_count == 0
        elif variant == "H4_memory_after_fringe":
            keep = signed_mean > 0.15 and neg_count <= max(pos_count, same_mask, 1)
        elif variant == "H5_memory_appearance_motion":
            app_motion_mean = float(np.mean([_appearance_motion_bonus(row) for row in pair_rows]))
            keep = signed_mean + app_motion_mean > 0.16 and neg_count <= max(pos_count, same_mask, 1)
        elif variant == "H6_memory_after_ownership":
            keep = signed_mean > 0.18 and pos_count >= 2 and neg_count <= max(same_mask, 1)
        elif variant == "H7_memory_after_ownership_appmotion":
            app_motion_mean = float(np.mean([_appearance_motion_bonus(row) for row in pair_rows]))
            keep = signed_mean + app_motion_mean > 0.18 and pos_count >= 2 and neg_count <= max(same_mask, 1)
        if keep:
            memory_merge_count += int(core_uf.union(pair[0], pair[1]))
    remap = {tube_id: core_uf.find(comp) for tube_id, comp in core_labels.items()}
    return _compact_labels(remap), len(candidates), memory_merge_count


def _window0_seed_assignment_labels(
    full_tube_ids: list[int],
    full_rows: list[dict[str, Any]],
    window_rows: list[dict[str, Any]],
    *,
    mode: str,
) -> tuple[dict[int, int], int]:
    """Use window0 objects as fixed seeds and assign later tubes without seed-seed union."""
    full_tube_set = {int(tube_id) for tube_id in full_tube_ids}
    window_tubes = sorted(
        {
            _int(row, f"tube_{side}")
            for row in window_rows
            for side in ("i", "j")
            if _int(row, f"tube_{side}") in full_tube_set
        }
    )
    if not window_tubes:
        return {int(tube_id): idx for idx, tube_id in enumerate(sorted(full_tube_set))}, 0

    seed_uf = UnionFind(window_tubes)
    kept = 0
    for row in window_rows:
        ti, tj = _int(row, "tube_i"), _int(row, "tube_j")
        if ti not in full_tube_set or tj not in full_tube_set:
            continue
        if _bool(row.get("predicted_merge", False)):
            kept += int(seed_uf.union(ti, tj))

    seed_labels = seed_uf.labels()
    seed_components = set(seed_labels.values())
    seed_sizes: Counter[int] = Counter(seed_labels.values())
    next_label = max(seed_components, default=-1) + 1
    labels: dict[int, int] = {}
    for tube_id in sorted(full_tube_set):
        if tube_id in seed_labels:
            labels[tube_id] = int(seed_labels[tube_id])
        else:
            labels[tube_id] = next_label
            next_label += 1

    candidate_scores: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in full_rows:
        ti, tj = _int(row, "tube_i"), _int(row, "tube_j")
        if ti not in labels or tj not in labels:
            continue
        si = seed_labels.get(ti)
        sj = seed_labels.get(tj)
        if si is not None and sj is not None:
            continue
        if si is None and sj is None:
            continue
        tube_id = tj if si is not None else ti
        seed = si if si is not None else sj
        if seed is None:
            continue
        neg = _negative_count(row)
        # Cannot-link/boundary-cross rows are hard evidence against assigning to this seed.
        if _int(row, "boundary_cross_count") > 0 or _int(row, "same_frame_cannot_link_count") > 0:
            candidate_scores[tube_id][seed].append(-0.25 * (1.0 + np.log1p(neg)))
            continue
        if _int(row, "visible_outside_count") > 2 * max(_int(row, "same_mask_count"), 1):
            candidate_scores[tube_id][seed].append(-0.10 * np.log1p(neg))
            continue
        if not _bool(row.get("predicted_merge", False)):
            continue
        score = _signed_score(row)
        if mode == "appearance":
            score += _appearance_motion_bonus(row)
        if mode == "broad":
            score += 0.03 * np.log1p(seed_sizes.get(seed, 1))
        candidate_scores[tube_id][seed].append(float(score))

    assigned_count = 0
    for tube_id, per_seed in candidate_scores.items():
        if tube_id in seed_labels:
            continue
        ranked: list[tuple[float, float, int, int]] = []
        for seed, vals in per_seed.items():
            positive_vals = [float(v) for v in vals if float(v) > 0.0]
            if not positive_vals:
                continue
            total = float(np.sum(vals))
            mean_positive = float(np.mean(positive_vals))
            support = len(positive_vals)
            ranked.append((total, mean_positive, support, int(seed)))
        if not ranked:
            continue
        ranked.sort(reverse=True)
        best_total, best_mean, best_support, best_seed = ranked[0]
        second_total = ranked[1][0] if len(ranked) > 1 else 0.0
        if mode == "broad":
            keep = best_total >= 0.12 and best_mean >= 0.06 and best_total >= second_total + 0.01
        else:
            keep = (
                best_support >= 2
                and best_total >= 0.20
                and best_mean >= 0.08
                and best_total >= second_total + 0.03
            ) or (best_support == 1 and best_total >= 0.45 and best_total >= second_total + 0.05)
        if keep:
            labels[tube_id] = best_seed
            assigned_count += 1
    return _compact_labels(labels), kept + assigned_count


def _tube_metadata(rows: list[dict[str, Any]]) -> tuple[list[int], dict[int, int], dict[int, set[int]]]:
    tube_ids: set[int] = set()
    label_votes: dict[int, Counter[int]] = defaultdict(Counter)
    tube_chunks: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        for side in ("i", "j"):
            tube_id = _int(row, f"tube_{side}")
            gt = _int(row, f"gt_{side}")
            chunk = _int(row, f"chunk_{side}")
            tube_ids.add(tube_id)
            tube_chunks[tube_id].add(chunk)
            if gt > 0:
                label_votes[tube_id][gt] += 1
    labels = {
        tube_id: int(votes.most_common(1)[0][0])
        for tube_id, votes in label_votes.items()
        if votes
    }
    return sorted(tube_ids), labels, tube_chunks


def _metrics(
    *,
    scene: str,
    variant: str,
    labels_pred: dict[int, int],
    labels_gt: dict[int, int],
    tube_chunks: dict[int, set[int]],
    rows: list[dict[str, Any]],
    kept_edge_count: int,
    memory_merge_count: int = 0,
    memory_candidate_count: int = 0,
) -> dict[str, Any]:
    labeled = [tube_id for tube_id in sorted(labels_pred) if labels_gt.get(tube_id, 0) > 0]
    true = [int(labels_gt[tube_id]) for tube_id in labeled]
    pred = [int(labels_pred[tube_id]) for tube_id in labeled]
    comp_to_labels: dict[int, Counter[int]] = defaultdict(Counter)
    gt_to_comps: dict[int, set[int]] = defaultdict(set)
    comp_to_tubes: dict[int, list[int]] = defaultdict(list)
    for tube_id, comp in labels_pred.items():
        comp_to_tubes[int(comp)].append(int(tube_id))
        gt = int(labels_gt.get(tube_id, 0))
        if gt > 0:
            comp_to_labels[int(comp)][gt] += 1
            gt_to_comps[gt].add(int(comp))
    purity_num = sum(max(counts.values()) for counts in comp_to_labels.values() if counts)
    completeness_num = 0
    for gt, comps in gt_to_comps.items():
        completeness_num += max(comp_to_labels[comp].get(gt, 0) for comp in comps)
    overmerge_count = sum(1 for counts in comp_to_labels.values() if len(counts) > 1)
    oversplit_count = sum(1 for comps in gt_to_comps.values() if len(comps) > 1)
    same_comp_same = 0
    same_comp_diff = 0
    cut_violations = 0
    cannot_link_violations = 0
    for row in rows:
        ti, tj = _int(row, "tube_i"), _int(row, "tube_j")
        if labels_pred.get(ti) != labels_pred.get(tj):
            continue
        if _bool(row.get("same_gt", False)):
            same_comp_same += 1
        elif _bool(row.get("different_gt", False)):
            same_comp_diff += 1
        if _negative_count(row) > 0:
            cut_violations += 1
        if _int(row, "same_frame_cannot_link_count") > 0:
            cannot_link_violations += 1
    chunk_spans = []
    multi_chunk = 0
    for tubes in comp_to_tubes.values():
        chunks = set()
        for tube_id in tubes:
            chunks.update(tube_chunks.get(tube_id, set()))
        if chunks:
            span = max(chunks) - min(chunks) + 1
            chunk_spans.append(span)
            if len(chunks) > 1:
                multi_chunk += 1
    largest = max((len(tubes) for tubes in comp_to_tubes.values()), default=0)
    return {
        "scene": scene,
        "variant": variant,
        "tube_count": int(len(labels_pred)),
        "labeled_tube_count": int(len(labeled)),
        "object_count": int(len(comp_to_tubes)),
        "largest_component_ratio": float(largest / max(len(labels_pred), 1)),
        "kept_edge_count": int(kept_edge_count),
        "memory_candidate_count": int(memory_candidate_count),
        "memory_merge_count": int(memory_merge_count),
        "ari": _ari(true, pred),
        "purity": float(purity_num / max(len(labeled), 1)),
        "completeness": float(completeness_num / max(len(labeled), 1)),
        "overmerge_count": int(overmerge_count),
        "oversplit_count": int(oversplit_count),
        "same_component_same_gt_pair_count": int(same_comp_same),
        "same_component_diff_gt_pair_count": int(same_comp_diff),
        "false_merge_rate": float(same_comp_diff / max(same_comp_same + same_comp_diff, 1)),
        "internal_cut_violation_count": int(cut_violations),
        "same_frame_cannot_link_violation_count": int(cannot_link_violations),
        "object_chunk_span_mean": float(np.mean(chunk_spans)) if chunk_spans else 0.0,
        "multi_chunk_object_ratio": float(multi_chunk / max(len(comp_to_tubes), 1)),
        "is_diagnostic_only": True,
    }


def _partition_rows(scene: str, rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    tube_ids, gt_labels, tube_chunks = _tube_metadata(rows)
    if variant in {
        "L6_local_core_ownership_assign",
        "H6_memory_after_ownership",
        "L7_local_core_ownership_appmotion",
        "H7_memory_after_ownership_appmotion",
    }:
        core_labels, kept = _ownership_assignment_labels(
            tube_ids,
            rows,
            use_appearance_motion=variant in {
                "L7_local_core_ownership_appmotion",
                "H7_memory_after_ownership_appmotion",
            },
        )
        if variant.startswith("H"):
            final_labels, memory_candidate_count, memory_merge_count = _memory_labels_from_core_labels(
                core_labels,
                rows,
                variant,
            )
            return _metrics(
                scene=scene,
                variant=variant,
                labels_pred=final_labels,
                labels_gt=gt_labels,
                tube_chunks=tube_chunks,
                rows=rows,
                kept_edge_count=kept,
                memory_merge_count=memory_merge_count,
                memory_candidate_count=memory_candidate_count,
            )
        return _metrics(
            scene=scene,
            variant=variant,
            labels_pred=core_labels,
            labels_gt=gt_labels,
            tube_chunks=tube_chunks,
            rows=rows,
            kept_edge_count=kept,
        )

    uf = UnionFind(tube_ids)
    kept = 0
    if variant in {"L0_full_positive_cc", "L0_window0_positive_cc"}:
        for row in rows:
            if _bool(row.get("predicted_merge", False)):
                kept += int(uf.union(_int(row, "tube_i"), _int(row, "tube_j")))
    elif variant == "L1_same_chunk_positive_cc":
        for row in rows:
            if _bool(row.get("predicted_merge", False)) and _bool(row.get("same_chunk", False)):
                kept += int(uf.union(_int(row, "tube_i"), _int(row, "tube_j")))
    elif variant == "L2_local_strict_cores":
        for row in rows:
            if _strong_local_core_edge(row):
                kept += int(uf.union(_int(row, "tube_i"), _int(row, "tube_j")))
    elif variant in {
        "L3_local_core_grow",
        "H1_object_memory_shared",
        "H2_object_memory_signed",
        "H3_object_memory_conservative",
    }:
        for row in rows:
            if _strong_local_core_edge(row):
                kept += int(uf.union(_int(row, "tube_i"), _int(row, "tube_j")))
        for row in rows:
            if _local_grow_edge(row):
                kept += int(uf.union(_int(row, "tube_i"), _int(row, "tube_j")))
    elif variant in {"L4_local_core_fringe_assign", "H4_memory_after_fringe"}:
        for row in rows:
            if _strong_local_core_edge(row):
                kept += int(uf.union(_int(row, "tube_i"), _int(row, "tube_j")))
        for _, small_tube, large_tube in _fringe_assignment_edges(uf, rows):
            kept += int(uf.union(small_tube, large_tube))
    elif variant in {"L5_local_appearance_motion_grow", "H5_memory_appearance_motion"}:
        for row in rows:
            if _strong_local_core_edge(row):
                kept += int(uf.union(_int(row, "tube_i"), _int(row, "tube_j")))
        for row in rows:
            if _local_appearance_motion_edge(row):
                kept += int(uf.union(_int(row, "tube_i"), _int(row, "tube_j")))
    else:
        raise ValueError(f"unknown variant: {variant}")
    memory_candidate_count = 0
    memory_merge_count = 0
    if variant.startswith("H"):
        core_labels = uf.labels()
        core_pair_rows: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if _bool(row.get("same_chunk", False)):
                continue
            if not _bool(row.get("same_submap", False)):
                continue
            ci = core_labels.get(_int(row, "tube_i"))
            cj = core_labels.get(_int(row, "tube_j"))
            if ci is None or cj is None or ci == cj:
                continue
            core_pair_rows[tuple(sorted((ci, cj)))].append(row)
        candidates = []
        for pair, pair_rows in core_pair_rows.items():
            pos_count = sum(1 for row in pair_rows if _bool(row.get("predicted_merge", False)))
            if pos_count <= 0:
                continue
            neg_count = sum(_negative_count(row) for row in pair_rows)
            same_mask = sum(_int(row, "same_mask_count") for row in pair_rows)
            signed_mean = float(np.mean([_signed_score(row) for row in pair_rows]))
            merge_mean = float(np.mean([_float(row, "merge_score") for row in pair_rows]))
            candidates.append((signed_mean, merge_mean, pos_count, neg_count, same_mask, pair))
        candidates.sort(reverse=True)
        memory_candidate_count = len(candidates)
        core_uf = UnionFind(sorted(set(core_labels.values())))
        for signed_mean, merge_mean, pos_count, neg_count, same_mask, pair in candidates:
            keep = False
            if variant == "H1_object_memory_shared":
                keep = neg_count == 0 and pos_count >= 1
            elif variant == "H2_object_memory_signed":
                keep = signed_mean > 0.15 and neg_count <= max(pos_count, same_mask, 1)
            elif variant == "H3_object_memory_conservative":
                keep = pos_count >= 2 and signed_mean > 0.20 and neg_count == 0
            elif variant == "H4_memory_after_fringe":
                keep = signed_mean > 0.15 and neg_count <= max(pos_count, same_mask, 1)
            elif variant == "H5_memory_appearance_motion":
                app_motion_mean = float(np.mean([_appearance_motion_bonus(row) for row in pair_rows]))
                keep = signed_mean + app_motion_mean > 0.16 and neg_count <= max(pos_count, same_mask, 1)
            if keep:
                memory_merge_count += int(core_uf.union(pair[0], pair[1]))
        remap = {tube_id: core_uf.find(comp) for tube_id, comp in core_labels.items()}
        root_to_idx: dict[int, int] = {}
        final_labels: dict[int, int] = {}
        for tube_id, root in sorted(remap.items()):
            if root not in root_to_idx:
                root_to_idx[root] = len(root_to_idx)
            final_labels[tube_id] = root_to_idx[root]
        return _metrics(
            scene=scene,
            variant=variant,
            labels_pred=final_labels,
            labels_gt=gt_labels,
            tube_chunks=tube_chunks,
            rows=rows,
            kept_edge_count=kept,
            memory_merge_count=memory_merge_count,
            memory_candidate_count=memory_candidate_count,
        )
    return _metrics(
        scene=scene,
        variant=variant,
        labels_pred=uf.labels(),
        labels_gt=gt_labels,
        tube_chunks=tube_chunks,
        rows=rows,
        kept_edge_count=kept,
    )


def _partition_window_seed_rows(
    scene: str,
    full_rows: list[dict[str, Any]],
    window_rows: list[dict[str, Any]],
    variant: str,
) -> dict[str, Any]:
    tube_ids, gt_labels, tube_chunks = _tube_metadata(full_rows)
    if variant == "W1_window0_seed_assign":
        mode = "strict"
    elif variant == "W2_window0_seed_assign_broad":
        mode = "broad"
    elif variant == "W3_window0_seed_assign_appmotion":
        mode = "appearance"
    else:
        raise ValueError(f"unknown window-seed variant: {variant}")
    labels_pred, kept = _window0_seed_assignment_labels(tube_ids, full_rows, window_rows, mode=mode)
    return _metrics(
        scene=scene,
        variant=variant,
        labels_pred=labels_pred,
        labels_gt=gt_labels,
        tube_chunks=tube_chunks,
        rows=full_rows,
        kept_edge_count=kept,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run v27 local-core and object-memory diagnostics.")
    parser.add_argument("--pair-rows", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--label", default="v27_local_core_memory")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = _read_csv(Path(args.pair_rows))
    full_rows = [
        row
        for row in rows
        if str(row.get("category", "")).startswith(("B0_", "B1_", "B2_", "B3_", "B4_", "B5_"))
        and not str(row.get("category", "")).startswith("B6_")
    ]
    window_rows = [row for row in rows if str(row.get("category", "")).startswith("B7_window0_")]
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in full_rows:
        by_scene[str(row["scene"])].append(row)
    window_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in window_rows:
        window_by_scene[str(row["scene"])].append(row)

    variants = [
        "L0_full_positive_cc",
        "L1_same_chunk_positive_cc",
        "L2_local_strict_cores",
        "L3_local_core_grow",
        "L4_local_core_fringe_assign",
        "L5_local_appearance_motion_grow",
        "L6_local_core_ownership_assign",
        "L7_local_core_ownership_appmotion",
        "H1_object_memory_shared",
        "H2_object_memory_signed",
        "H3_object_memory_conservative",
        "H4_memory_after_fringe",
        "H5_memory_appearance_motion",
        "H6_memory_after_ownership",
        "H7_memory_after_ownership_appmotion",
    ]
    scene_rows: list[dict[str, Any]] = []
    for scene, scene_pair_rows in sorted(by_scene.items()):
        for variant in variants:
            scene_rows.append(_partition_rows(scene, scene_pair_rows, variant))
        if scene in window_by_scene:
            for variant in [
                "W1_window0_seed_assign",
                "W2_window0_seed_assign_broad",
                "W3_window0_seed_assign_appmotion",
            ]:
                scene_rows.append(
                    _partition_window_seed_rows(scene, scene_pair_rows, window_by_scene[scene], variant)
                )
    for scene, scene_pair_rows in sorted(window_by_scene.items()):
        scene_rows.append(_partition_rows(scene, scene_pair_rows, "L0_window0_positive_cc"))

    summary_rows: list[dict[str, Any]] = []
    for variant in sorted({row["variant"] for row in scene_rows}):
        items = [row for row in scene_rows if row["variant"] == variant]
        summary_rows.append(
            {
                "variant": variant,
                "scene_count": int(len(items)),
                "ari_mean": _mean(items, "ari"),
                "purity_mean": _mean(items, "purity"),
                "completeness_mean": _mean(items, "completeness"),
                "false_merge_rate_mean": _mean(items, "false_merge_rate"),
                "largest_component_ratio_mean": _mean(items, "largest_component_ratio"),
                "object_count_mean": _mean(items, "object_count"),
                "kept_edge_count_mean": _mean(items, "kept_edge_count"),
                "memory_candidate_count_mean": _mean(items, "memory_candidate_count"),
                "memory_merge_count_mean": _mean(items, "memory_merge_count"),
                "overmerge_count_mean": _mean(items, "overmerge_count"),
                "oversplit_count_mean": _mean(items, "oversplit_count"),
                "internal_cut_violation_count_mean": _mean(items, "internal_cut_violation_count"),
                "same_frame_cannot_link_violation_count_mean": _mean(
                    items, "same_frame_cannot_link_violation_count"
                ),
                "object_chunk_span_mean": _mean(items, "object_chunk_span_mean"),
                "multi_chunk_object_ratio_mean": _mean(items, "multi_chunk_object_ratio"),
            }
        )

    manifest = {
        "label": str(args.label),
        "pair_rows": str(args.pair_rows),
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_metrics": True,
        "scope": "sampled_pair_rows_local_core_memory_diagnostic",
        "phase_g_local_cores_attempted": True,
        "phase_h_object_memory_attempted": True,
        "note": "Uses sampled pair rows from v27 attribution. It is not a full tube graph or AP export.",
    }
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / f"{args.label}_scene_rows.csv", scene_rows)
    _write_csv(output_root / f"{args.label}_summary.csv", summary_rows)
    (output_root / f"{args.label}_scene_rows.json").write_text(
        json.dumps(_json_safe(scene_rows), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / f"{args.label}_summary.json").write_text(
        json.dumps(_json_safe(summary_rows), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / f"{args.label}_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe({"manifest": manifest, "summary": summary_rows}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
