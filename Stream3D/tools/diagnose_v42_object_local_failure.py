from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _auc_fast(labels: list[bool], scores: list[float]) -> float | None:
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    n_pos = sum(1 for _score, label in pairs if label)
    n_neg = len(pairs) - n_pos
    if not n_pos or not n_neg:
        return None
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        next_index = index + 1
        while next_index < len(pairs) and pairs[next_index][0] == pairs[index][0]:
            next_index += 1
        avg_rank = (index + 1 + next_index) / 2.0
        rank_sum += avg_rank * sum(1 for _score, label in pairs[index:next_index] if label)
        index = next_index
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _mean(values: list[float]) -> float | None:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else None


def _quantile(values: list[float], q: float) -> float | None:
    return float(np.quantile(np.asarray(values, dtype=np.float64), float(q))) if values else None


def _qpack(prefix: str, values: list[float]) -> dict[str, Any]:
    return {
        f"{prefix}_count": int(len(values)),
        f"{prefix}_mean": _mean(values),
        f"{prefix}_p10": _quantile(values, 0.10),
        f"{prefix}_p25": _quantile(values, 0.25),
        f"{prefix}_p50": _quantile(values, 0.50),
        f"{prefix}_p75": _quantile(values, 0.75),
        f"{prefix}_p90": _quantile(values, 0.90),
    }


def _to_int(value: str, default: int = 0) -> int:
    if value in {"", "None", "nan"}:
        return int(default)
    return int(float(value))


def _to_float(value: str, default: float = 0.0) -> float:
    if value in {"", "None", "nan"}:
        return float(default)
    return float(value)


class _UnionFind:
    def __init__(self, values: list[int]) -> None:
        self.parent = {int(value): int(value) for value in values}

    def find(self, value: int) -> int:
        value = int(value)
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        root_left = self.find(int(left))
        root_right = self.find(int(right))
        if root_left != root_right:
            self.parent[root_right] = root_left


def _prediction_scene_tokens(
    token_rows: list[dict[str, str]],
    edge_rows: list[dict[str, str]],
    *,
    min_affinity: float,
    topk: int,
    min_token_fraction: float,
    min_tokens: int,
    min_frames: int,
) -> set[int]:
    token_ids = [_to_int(row["token_id"]) for row in token_rows]
    token_frame = {_to_int(row["token_id"]): _to_int(row["frame_id"]) for row in token_rows}
    if not token_ids:
        return set()
    uf = _UnionFind(token_ids)
    candidates = [
        row
        for row in edge_rows
        if row.get("same_frame_cannot_link") != "True"
        and _to_float(row.get("semantic_affinity", "0")) >= float(min_affinity)
        and _to_int(row["token_i"]) in token_frame
        and _to_int(row["token_j"]) in token_frame
    ]
    neighbors: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for row in candidates:
        left = _to_int(row["token_i"])
        right = _to_int(row["token_j"])
        score = _to_float(row["semantic_affinity"])
        neighbors[left].append((score, right))
        neighbors[right].append((score, left))
    top_neighbors: dict[int, set[int]] = {}
    for token_id, rows in neighbors.items():
        rows.sort(key=lambda item: item[0], reverse=True)
        top_neighbors[int(token_id)] = {int(other) for _score, other in rows[: max(int(topk), 1)]}
    for row in candidates:
        left = _to_int(row["token_i"])
        right = _to_int(row["token_j"])
        if right in top_neighbors.get(left, set()) and left in top_neighbors.get(right, set()):
            uf.union(left, right)

    members_by_root: dict[int, list[int]] = defaultdict(list)
    for token_id in token_ids:
        members_by_root[uf.find(token_id)].append(token_id)
    min_component_tokens = max(int(min_tokens), int(np.ceil(float(min_token_fraction) * max(len(token_ids), 1))))
    scene_tokens: set[int] = set()
    for members in members_by_root.values():
        frames = {token_frame[token_id] for token_id in members}
        if len(members) >= min_component_tokens and len(frames) >= int(min_frames):
            scene_tokens.update(int(token_id) for token_id in members)
    return scene_tokens


def _edge_area_ratio(row: dict[str, str]) -> float:
    left = max(_to_float(row.get("area_i", "0")), 1.0)
    right = max(_to_float(row.get("area_j", "0")), 1.0)
    return float(max(left, right) / max(min(left, right), 1.0))


def _edge_min_purity(row: dict[str, str]) -> float:
    return min(_to_float(row.get("purity_i", "0")), _to_float(row.get("purity_j", "0")))


def _edge_min_iou(row: dict[str, str]) -> float:
    return min(_to_float(row.get("iou_i", "0")), _to_float(row.get("iou_j", "0")))


def _edge_delta_frame(row: dict[str, str]) -> int:
    return abs(_to_int(row.get("frame_i", "0")) - _to_int(row.get("frame_j", "0")))


def _source_rows(
    *,
    scene: str,
    source: str,
    token_rows: list[dict[str, str]],
    edge_rows: list[dict[str, str]],
    role_min_affinity: float,
    role_topk: int,
    role_min_token_fraction: float,
    role_min_tokens: int,
    role_min_frames: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source_tokens = [row for row in token_rows if row.get("source") == source]
    source_edges = [row for row in edge_rows if row.get("source") == source]
    scene_tokens = _prediction_scene_tokens(
        source_tokens,
        source_edges,
        min_affinity=role_min_affinity,
        topk=role_topk,
        min_token_fraction=role_min_token_fraction,
        min_tokens=role_min_tokens,
        min_frames=role_min_frames,
    )
    labeled = [
        row
        for row in source_edges
        if row.get("diagnostic_same_gt") in {"True", "False"}
        and _to_int(row["token_i"]) not in scene_tokens
        and _to_int(row["token_j"]) not in scene_tokens
    ]
    positives = [row for row in labeled if row["diagnostic_same_gt"] == "True"]
    negatives = [row for row in labeled if row["diagnostic_same_gt"] == "False"]
    pos_scores = [_to_float(row["semantic_affinity"]) for row in positives]
    neg_scores = [_to_float(row["semantic_affinity"]) for row in negatives]
    labels = [row["diagnostic_same_gt"] == "True" for row in labeled]
    scores = [_to_float(row["semantic_affinity"]) for row in labeled]
    pos_p50 = _quantile(pos_scores, 0.50)
    neg_p90 = _quantile(neg_scores, 0.90)
    neg_p95 = _quantile(neg_scores, 0.95)

    same_frame_neg = [row for row in negatives if row.get("same_frame_cannot_link") == "True"]
    cross_frame_neg = [row for row in negatives if row.get("same_frame_cannot_link") != "True"]
    cross_frame_pos = [row for row in positives if _edge_delta_frame(row) > 0]
    same_frame_pos = [row for row in positives if _edge_delta_frame(row) == 0]
    summary: dict[str, Any] = {
        "scene": scene,
        "source": source,
        "role_assignment_uses_gt": False,
        "uses_gt_for_diagnostic_labels": True,
        "scene_role_token_count": int(len(scene_tokens)),
        "object_local_edge_count": int(len(labeled)),
        "object_local_positive_edge_count": int(len(positives)),
        "object_local_negative_edge_count": int(len(negatives)),
        "object_local_semantic_affinity_AUC": _auc_fast(labels, scores),
        "positive_below_negative_p90_rate": float(sum(score <= neg_p90 for score in pos_scores) / max(len(pos_scores), 1))
        if neg_p90 is not None
        else None,
        "negative_above_positive_median_rate": float(sum(score >= pos_p50 for score in neg_scores) / max(len(neg_scores), 1))
        if pos_p50 is not None
        else None,
        "negative_above_0_70_rate": float(sum(score >= 0.70 for score in neg_scores) / max(len(neg_scores), 1)),
        "negative_above_0_80_rate": float(sum(score >= 0.80 for score in neg_scores) / max(len(neg_scores), 1)),
        "same_frame_negative_count": int(len(same_frame_neg)),
        "same_frame_negative_mean": _mean([_to_float(row["semantic_affinity"]) for row in same_frame_neg]),
        "same_frame_negative_ge_negative_p90_rate": float(
            sum(_to_float(row["semantic_affinity"]) >= neg_p90 for row in same_frame_neg) / max(len(same_frame_neg), 1)
        )
        if neg_p90 is not None
        else None,
        "cross_frame_negative_count": int(len(cross_frame_neg)),
        "cross_frame_negative_mean": _mean([_to_float(row["semantic_affinity"]) for row in cross_frame_neg]),
        "cross_frame_positive_count": int(len(cross_frame_pos)),
        "cross_frame_positive_mean": _mean([_to_float(row["semantic_affinity"]) for row in cross_frame_pos]),
        "same_frame_positive_count": int(len(same_frame_pos)),
        "same_frame_positive_mean": _mean([_to_float(row["semantic_affinity"]) for row in same_frame_pos]),
        "positive_low_purity_edge_rate": float(
            sum(_edge_min_purity(row) < 0.80 for row in positives) / max(len(positives), 1)
        ),
        "negative_low_purity_edge_rate": float(
            sum(_edge_min_purity(row) < 0.80 for row in negatives) / max(len(negatives), 1)
        ),
        "positive_low_iou_edge_rate": float(sum(_edge_min_iou(row) < 0.10 for row in positives) / max(len(positives), 1)),
        "negative_low_iou_edge_rate": float(sum(_edge_min_iou(row) < 0.10 for row in negatives) / max(len(negatives), 1)),
        "positive_area_ratio_mean": _mean([_edge_area_ratio(row) for row in positives]),
        "negative_area_ratio_mean": _mean([_edge_area_ratio(row) for row in negatives]),
        "negative_p90_threshold": neg_p90,
        "negative_p95_threshold": neg_p95,
        "positive_median_threshold": pos_p50,
    }
    summary.update(_qpack("positive_semantic", pos_scores))
    summary.update(_qpack("negative_semantic", neg_scores))

    negatives_for_auc = neg_scores
    instance_rows: list[dict[str, Any]] = []
    positives_by_gt: dict[str, list[dict[str, str]]] = defaultdict(list)
    token_counter = Counter(
        row.get("diagnostic_gt_instance", "")
        for row in source_tokens
        if _to_int(row["token_id"]) not in scene_tokens and row.get("diagnostic_gt_instance") not in {"", "None", "0"}
    )
    for row in positives:
        gt = row.get("gt_i", "")
        if gt and gt == row.get("gt_j"):
            positives_by_gt[str(gt)].append(row)
    for gt, rows in positives_by_gt.items():
        gt_scores = [_to_float(row["semantic_affinity"]) for row in rows]
        instance_rows.append(
            {
                "scene": scene,
                "source": source,
                "gt_instance": gt,
                "object_local_token_count": int(token_counter.get(gt, 0)),
                "positive_edge_count": int(len(rows)),
                "positive_score_mean": _mean(gt_scores),
                "positive_score_p10": _quantile(gt_scores, 0.10),
                "positive_score_p50": _quantile(gt_scores, 0.50),
                "positive_below_negative_p90_rate": float(
                    sum(score <= neg_p90 for score in gt_scores) / max(len(gt_scores), 1)
                )
                if neg_p90 is not None
                else None,
                "instance_vs_all_negative_auc": _auc_fast(
                    [True] * len(gt_scores) + [False] * len(negatives_for_auc),
                    gt_scores + negatives_for_auc,
                ),
                "low_purity_edge_rate": float(sum(_edge_min_purity(row) < 0.80 for row in rows) / max(len(rows), 1)),
                "low_iou_edge_rate": float(sum(_edge_min_iou(row) < 0.10 for row in rows) / max(len(rows), 1)),
                "mean_delta_frame": _mean([float(_edge_delta_frame(row)) for row in rows]),
            }
        )
    instance_rows.sort(key=lambda row: (float(row["instance_vs_all_negative_auc"] or 1.0), -int(row["positive_edge_count"])))

    pair_rows: list[dict[str, Any]] = []
    high_threshold = pos_p50 if pos_p50 is not None else 0.70
    negatives_by_pair: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in negatives:
        left = str(row.get("gt_i", ""))
        right = str(row.get("gt_j", ""))
        if not left or not right:
            continue
        key = tuple(sorted((left, right)))
        negatives_by_pair[key].append(row)
    for (left, right), rows in negatives_by_pair.items():
        pair_scores = [_to_float(row["semantic_affinity"]) for row in rows]
        high_rows = [row for row in rows if _to_float(row["semantic_affinity"]) >= float(high_threshold)]
        pair_rows.append(
            {
                "scene": scene,
                "source": source,
                "gt_pair": f"{left}:{right}",
                "negative_edge_count": int(len(rows)),
                "negative_score_mean": _mean(pair_scores),
                "negative_score_p90": _quantile(pair_scores, 0.90),
                "negative_score_p95": _quantile(pair_scores, 0.95),
                "negative_above_positive_median_count": int(len(high_rows)),
                "negative_above_positive_median_rate": float(len(high_rows) / max(len(rows), 1)),
                "same_frame_negative_rate": float(
                    sum(row.get("same_frame_cannot_link") == "True" for row in rows) / max(len(rows), 1)
                ),
                "low_purity_edge_rate": float(sum(_edge_min_purity(row) < 0.80 for row in rows) / max(len(rows), 1)),
                "low_iou_edge_rate": float(sum(_edge_min_iou(row) < 0.10 for row in rows) / max(len(rows), 1)),
                "mean_delta_frame": _mean([float(_edge_delta_frame(row)) for row in rows]),
            }
        )
    pair_rows.sort(
        key=lambda row: (
            -int(row["negative_above_positive_median_count"]),
            -float(row["negative_score_p95"] or 0.0),
        )
    )
    temporal_rows: list[dict[str, Any]] = []
    bins: list[tuple[str, int, int | None]] = [
        ("delta_0", 0, 0),
        ("delta_1_300", 1, 300),
        ("delta_301_700", 301, 700),
        ("delta_gt700", 701, None),
    ]
    for name, low, high in bins:
        subset = [
            row
            for row in labeled
            if _edge_delta_frame(row) >= int(low) and (high is None or _edge_delta_frame(row) <= int(high))
        ]
        subset_labels = [row["diagnostic_same_gt"] == "True" for row in subset]
        subset_scores = [_to_float(row["semantic_affinity"]) for row in subset]
        subset_pos = [_to_float(row["semantic_affinity"]) for row in subset if row["diagnostic_same_gt"] == "True"]
        subset_neg = [_to_float(row["semantic_affinity"]) for row in subset if row["diagnostic_same_gt"] == "False"]
        subset_neg_p90 = _quantile(subset_neg, 0.90)
        temporal_rows.append(
            {
                "scene": scene,
                "source": source,
                "temporal_bin": name,
                "delta_min": int(low),
                "delta_max": "" if high is None else int(high),
                "object_local_edge_count": int(len(subset)),
                "positive_edge_count": int(len(subset_pos)),
                "negative_edge_count": int(len(subset_neg)),
                "semantic_affinity_AUC": _auc_fast(subset_labels, subset_scores),
                "positive_score_mean": _mean(subset_pos),
                "positive_score_p10": _quantile(subset_pos, 0.10),
                "positive_score_p50": _quantile(subset_pos, 0.50),
                "negative_score_mean": _mean(subset_neg),
                "negative_score_p90": subset_neg_p90,
                "positive_below_bin_negative_p90_rate": float(
                    sum(score <= subset_neg_p90 for score in subset_pos) / max(len(subset_pos), 1)
                )
                if subset_neg_p90 is not None
                else None,
                "positive_low_iou_edge_rate": float(
                    sum(_edge_min_iou(row) < 0.10 for row in subset if row["diagnostic_same_gt"] == "True")
                    / max(len(subset_pos), 1)
                ),
            }
        )
    return summary, instance_rows[:10], pair_rows[:10], temporal_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument("--sources", required=True)
    parser.add_argument("--role-min-affinity", type=float, default=0.60)
    parser.add_argument("--role-topk", type=int, default=4)
    parser.add_argument("--role-min-token-fraction", type=float, default=0.08)
    parser.add_argument("--role-min-tokens", type=int, default=24)
    parser.add_argument("--role-min-frames", type=int, default=4)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    audit_root = Path(args.audit_root)
    output_root = Path(args.output_root)
    scenes = [item.strip() for item in str(args.scenes).split(",") if item.strip()]
    sources = [item.strip() for item in str(args.sources).split(",") if item.strip()]
    summary_rows: list[dict[str, Any]] = []
    instance_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    for scene in scenes:
        token_rows = _read_csv(audit_root / scene / "part_token_rows.csv")
        edge_rows = _read_csv(audit_root / scene / "part_edge_rows.csv")
        for source in sources:
            summary, instances, pairs, temporal = _source_rows(
                scene=scene,
                source=source,
                token_rows=token_rows,
                edge_rows=edge_rows,
                role_min_affinity=float(args.role_min_affinity),
                role_topk=int(args.role_topk),
                role_min_token_fraction=float(args.role_min_token_fraction),
                role_min_tokens=int(args.role_min_tokens),
                role_min_frames=int(args.role_min_frames),
            )
            summary_rows.append(summary)
            instance_rows.extend(instances)
            pair_rows.extend(pairs)
            temporal_rows.extend(temporal)
    _write_csv(output_root / "object_local_failure_summary_rows.csv", summary_rows)
    _write_csv(output_root / "object_local_low_positive_instance_rows.csv", instance_rows)
    _write_csv(output_root / "object_local_high_negative_pair_rows.csv", pair_rows)
    _write_csv(output_root / "object_local_temporal_bin_rows.csv", temporal_rows)
    print(
        json.dumps(
            {
                "summary_csv": str(output_root / "object_local_failure_summary_rows.csv"),
                "instance_csv": str(output_root / "object_local_low_positive_instance_rows.csv"),
                "pair_csv": str(output_root / "object_local_high_negative_pair_rows.csv"),
                "temporal_csv": str(output_root / "object_local_temporal_bin_rows.csv"),
                "summary_rows": len(summary_rows),
                "instance_rows": len(instance_rows),
                "pair_rows": len(pair_rows),
                "temporal_rows": len(temporal_rows),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
