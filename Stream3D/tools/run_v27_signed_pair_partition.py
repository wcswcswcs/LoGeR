from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from typing import Any

import numpy as np


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


def _is_merge_scope_category(row: dict[str, Any]) -> bool:
    category = str(row.get("category", ""))
    return category.startswith(("B0_", "B2_", "B3_", "B4_"))


def _signed_score(row: dict[str, Any]) -> float:
    merge_score = _float(row, "merge_score")
    same_mask = _int(row, "same_mask_count")
    boundary_safe = _int(row, "boundary_safe_count")
    neg = (
        _int(row, "boundary_cross_count")
        + _int(row, "same_frame_cannot_link_count")
        + _int(row, "visible_outside_count")
    )
    return float(merge_score + 0.04 * np.log1p(max(same_mask + boundary_safe, 0)) - 0.05 * np.log1p(max(neg, 0)))


def _cut_penalty(row: dict[str, Any]) -> float:
    boundary_cross = _int(row, "boundary_cross_count")
    cannot_link = _int(row, "same_frame_cannot_link_count")
    visible_outside = _int(row, "visible_outside_count")
    same_mask = max(_int(row, "same_mask_count"), 1)
    penalty = 0.35 * (boundary_cross + cannot_link)
    if visible_outside > same_mask:
        penalty += 0.06 * np.log1p(visible_outside - same_mask)
    return float(penalty)


def _row_cut_score(row: dict[str, Any]) -> float:
    return max(_float(row, "cut_score"), 0.0)


def _ari(labels_true: list[int], labels_pred: list[int]) -> float | None:
    if len(labels_true) < 2:
        return None
    n = len(labels_true)
    contingency: dict[tuple[int, int], int] = defaultdict(int)
    true_counts: Counter[int] = Counter(labels_true)
    pred_counts: Counter[int] = Counter(labels_pred)
    for t, p in zip(labels_true, labels_pred):
        contingency[(int(t), int(p))] += 1
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


def _variant_keep(row: dict[str, Any], variant: str) -> bool:
    if not _bool(row.get("predicted_merge", False)):
        return False
    boundary_cross = _int(row, "boundary_cross_count")
    cannot_link = _int(row, "same_frame_cannot_link_count")
    visible_outside = _int(row, "visible_outside_count")
    same_mask = _int(row, "same_mask_count")
    if variant == "S0_positive_only":
        return True
    if variant == "S1_boundary_veto":
        return boundary_cross == 0
    if variant == "S2_boundary_cannot_veto":
        return boundary_cross == 0 and cannot_link == 0
    if variant == "S3_negative_majority_veto":
        return boundary_cross == 0 and cannot_link == 0 and visible_outside <= same_mask
    if variant == "S4_soft_signed_score":
        merge_score = _float(row, "merge_score")
        neg = boundary_cross + cannot_link + visible_outside
        signed_score = merge_score + 0.05 * np.log1p(max(same_mask, 0)) - 0.05 * np.log1p(max(neg, 0))
        return bool(signed_score > 0.25)
    if variant == "S5_window0_positive_only":
        return True
    if variant == "S6_merge_scope_positive_only":
        return _is_merge_scope_category(row)
    if variant == "S7_merge_scope_negative_veto":
        return _is_merge_scope_category(row) and boundary_cross == 0 and cannot_link == 0 and visible_outside <= same_mask
    if variant == "S8_merge_scope_soft_signed":
        return _is_merge_scope_category(row) and _signed_score(row) > 0.25
    raise ValueError(f"unknown variant: {variant}")


def _agglomerative_partition(parent: dict[int, int], rows: list[dict[str, Any]], *, variant: str) -> set[tuple[int, int]]:
    rows_by_pair: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pair = tuple(sorted((_int(row, "tube_i"), _int(row, "tube_j"))))
        rows_by_pair[pair].append(row)

    members: dict[int, set[int]] = {int(tube_id): {int(tube_id)} for tube_id in parent}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def cluster_evidence(ra: int, rb: int) -> tuple[float, float, float, int]:
        left = members[ra]
        right = members[rb]
        if len(left) > len(right):
            left, right = right, left
        merge_gain = 0.0
        cut = 0.0
        normalized_cut = 0.0
        support = 0
        for a in left:
            for b in right:
                pair = tuple(sorted((int(a), int(b))))
                for row in rows_by_pair.get(pair, []):
                    if _bool(row.get("predicted_merge", False)) and _is_merge_scope_category(row):
                        score = _signed_score(row)
                        if score > 0.0:
                            merge_gain += score
                            support += 1
                    cut += _cut_penalty(row)
                    normalized_cut += _row_cut_score(row)
        return float(merge_gain), float(cut), float(normalized_cut), int(support)

    def union(ra: int, rb: int) -> None:
        if len(members[ra]) < len(members[rb]):
            ra, rb = rb, ra
        parent[rb] = ra
        members[ra].update(members.pop(rb))

    candidate_rows = [
        row
        for row in rows
        if _bool(row.get("predicted_merge", False)) and _is_merge_scope_category(row)
    ]
    candidate_rows.sort(key=lambda row: (-_signed_score(row), _float(row, "distance_normalized", 999.0)))
    kept: set[tuple[int, int]] = set()
    for row in candidate_rows:
        a, b = _int(row, "tube_i"), _int(row, "tube_j")
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        merge_gain, cut, normalized_cut, support = cluster_evidence(ra, rb)
        smaller = min(len(members[ra]), len(members[rb]))
        larger = max(len(members[ra]), len(members[rb]))
        support_density = float(support / max(smaller, 1))
        row_score = _signed_score(row)
        if variant == "S9_signed_agglomerative_cluster_veto":
            keep = support >= 1 and merge_gain >= 0.20 and merge_gain - cut > 0.05 and cut <= max(0.75 * merge_gain, 0.35)
        elif variant == "S10_signed_agglomerative_conservative":
            keep = support >= 2 and merge_gain >= 0.35 and merge_gain - cut > 0.15 and cut <= 0.50 * merge_gain
        elif variant == "S11_signed_agglomerative_balanced":
            small_fringe_attach = smaller <= 2 and row_score >= 0.25 and normalized_cut <= max(4.0 * merge_gain, 12.0)
            core_merge = (
                smaller > 2
                and support_density >= 0.20
                and merge_gain >= 0.55
                and normalized_cut <= max(2.0 * merge_gain, 8.0)
                and cut <= max(1.50 * merge_gain, 1.0)
            )
            large_guard = larger < 48 or normalized_cut <= max(1.25 * merge_gain, 6.0)
            keep = bool((small_fringe_attach or core_merge) and large_guard)
        elif variant == "S12_signed_agglomerative_fringe_first":
            small_fringe_attach = smaller <= 3 and row_score >= 0.18 and normalized_cut <= max(5.0 * merge_gain, 18.0)
            medium_attach = (
                smaller <= 8
                and support_density >= 0.15
                and merge_gain >= 0.40
                and normalized_cut <= max(3.0 * merge_gain, 12.0)
            )
            core_merge = (
                smaller > 8
                and support_density >= 0.25
                and merge_gain >= 0.80
                and normalized_cut <= max(1.50 * merge_gain, 8.0)
            )
            keep = bool(small_fringe_attach or medium_attach or core_merge)
        else:
            raise ValueError(f"unknown agglomerative variant: {variant}")
        if keep:
            kept.add(tuple(sorted((a, b))))
            union(ra, rb)
    return kept


def _partition(scene: str, rows: list[dict[str, Any]], *, variant: str) -> dict[str, Any]:
    parent: dict[int, int] = {}
    label_votes: dict[int, Counter[int]] = defaultdict(Counter)
    edge_rows = []
    negative_rows = []
    for row in rows:
        for side in ("i", "j"):
            tube_id = _int(row, f"tube_{side}")
            gt = _int(row, f"gt_{side}")
            parent.setdefault(tube_id, tube_id)
            if gt > 0:
                label_votes[tube_id][gt] += 1
        if _bool(row.get("predicted_merge", False)):
            edge_rows.append(row)
        if _int(row, "boundary_cross_count") > 0 or _int(row, "same_frame_cannot_link_count") > 0 or _int(row, "visible_outside_count") > 0:
            negative_rows.append(row)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if variant in {
        "S9_signed_agglomerative_cluster_veto",
        "S10_signed_agglomerative_conservative",
        "S11_signed_agglomerative_balanced",
        "S12_signed_agglomerative_fringe_first",
    }:
        kept_edges = _agglomerative_partition(parent, rows, variant=variant)
    else:
        kept_edges: set[tuple[int, int]] = set()
        for row in rows:
            if _variant_keep(row, variant):
                pair = tuple(sorted((_int(row, "tube_i"), _int(row, "tube_j"))))
                kept_edges.add(pair)
                union(pair[0], pair[1])

    comp_id: dict[int, int] = {}
    root_to_idx: dict[int, int] = {}
    for tube_id in sorted(parent):
        root = find(tube_id)
        if root not in root_to_idx:
            root_to_idx[root] = len(root_to_idx)
        comp_id[tube_id] = root_to_idx[root]

    labels: dict[int, int] = {}
    for tube_id, votes in label_votes.items():
        labels[tube_id] = int(votes.most_common(1)[0][0]) if votes else 0
    labeled = [tube_id for tube_id, gt in labels.items() if int(gt) > 0]
    labels_true = [int(labels[tube_id]) for tube_id in labeled]
    labels_pred = [int(comp_id.get(tube_id, -1)) for tube_id in labeled]
    comp_to_labels: dict[int, Counter[int]] = defaultdict(Counter)
    for tube_id in labeled:
        comp_to_labels[int(comp_id.get(tube_id, -1))][int(labels[tube_id])] += 1
    purity_num = sum(max(counts.values()) for counts in comp_to_labels.values() if counts)
    overmerge_count = sum(1 for counts in comp_to_labels.values() if len(counts) > 1)
    gt_to_comps: dict[int, set[int]] = defaultdict(set)
    for tube_id in labeled:
        gt_to_comps[int(labels[tube_id])].add(int(comp_id.get(tube_id, -1)))
    oversplit_count = sum(1 for comps in gt_to_comps.values() if len(comps) > 1)

    pred_same = 0
    pred_diff = 0
    for row in rows:
        pair = tuple(sorted((_int(row, "tube_i"), _int(row, "tube_j"))))
        if pair not in kept_edges:
            continue
        if _bool(row.get("same_gt", False)):
            pred_same += 1
        elif _bool(row.get("different_gt", False)):
            pred_diff += 1

    neg_diff = sum(1 for row in negative_rows if _bool(row.get("different_gt", False)))
    neg_labeled = sum(1 for row in negative_rows if _bool(row.get("gt_labeled_pair", False)))
    component_sizes = Counter(comp_id.values())
    largest = max(component_sizes.values(), default=0)
    return {
        "scene": scene,
        "variant": variant,
        "tube_count": int(len(parent)),
        "labeled_tube_count": int(len(labeled)),
        "edge_candidate_count": int(len(edge_rows)),
        "kept_edge_count": int(len(kept_edges)),
        "component_count": int(len(component_sizes)),
        "largest_component_size": int(largest),
        "largest_component_ratio": float(largest / max(len(parent), 1)),
        "ari": _ari(labels_true, labels_pred),
        "purity": float(purity_num / max(len(labeled), 1)),
        "overmerge_count": int(overmerge_count),
        "oversplit_count": int(oversplit_count),
        "pred_same_gt_edge_count": int(pred_same),
        "pred_diff_gt_edge_count": int(pred_diff),
        "false_merge_rate": float(pred_diff / max(pred_same + pred_diff, 1)),
        "negative_edge_different_gt_ratio": float(neg_diff / max(neg_labeled, 1)),
        "is_diagnostic_only": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run v27 sampled signed-pair partition diagnostics.")
    parser.add_argument("--pair-rows", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--label", default="v27_signed_pair_partition")
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
        "S0_positive_only",
        "S1_boundary_veto",
        "S2_boundary_cannot_veto",
        "S3_negative_majority_veto",
        "S4_soft_signed_score",
        "S6_merge_scope_positive_only",
        "S7_merge_scope_negative_veto",
        "S8_merge_scope_soft_signed",
        "S9_signed_agglomerative_cluster_veto",
        "S10_signed_agglomerative_conservative",
        "S11_signed_agglomerative_balanced",
        "S12_signed_agglomerative_fringe_first",
    ]
    partition_rows: list[dict[str, Any]] = []
    for scene, scene_rows in sorted(by_scene.items()):
        for variant in variants:
            partition_rows.append(_partition(scene, scene_rows, variant=variant))
    for scene, scene_rows in sorted(window_by_scene.items()):
        partition_rows.append(_partition(scene, scene_rows, variant="S5_window0_positive_only"))

    summary_rows = []
    for variant in sorted({row["variant"] for row in partition_rows}):
        items = [row for row in partition_rows if row["variant"] == variant]
        summary_rows.append(
            {
                "variant": variant,
                "scene_count": int(len(items)),
                "ari_mean": _mean(items, "ari"),
                "purity_mean": _mean(items, "purity"),
                "false_merge_rate_mean": _mean(items, "false_merge_rate"),
                "largest_component_ratio_mean": _mean(items, "largest_component_ratio"),
                "kept_edge_count_mean": _mean(items, "kept_edge_count"),
                "overmerge_count_mean": _mean(items, "overmerge_count"),
                "oversplit_count_mean": _mean(items, "oversplit_count"),
                "negative_edge_different_gt_ratio_mean": _mean(items, "negative_edge_different_gt_ratio"),
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
        "scope": "sampled_pair_rows_partition_diagnostic",
        "note": "Partition uses sampled pair rows from v27 attribution, not a complete method graph.",
    }
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / f"{args.label}_partition_rows.csv", partition_rows)
    _write_csv(output_root / f"{args.label}_summary.csv", summary_rows)
    (output_root / f"{args.label}_partition_rows.json").write_text(
        json.dumps(_json_safe(partition_rows), indent=2, sort_keys=True),
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
    print(json.dumps(_json_safe({"manifest": manifest, "summary": summary_rows}), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
