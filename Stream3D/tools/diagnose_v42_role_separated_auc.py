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


def _component_roles(
    token_rows: list[dict[str, str]],
    edge_rows: list[dict[str, str]],
    *,
    mode: str,
    min_affinity: float,
    topk: int,
    min_token_fraction: float,
    min_tokens: int,
    min_frames: int,
) -> tuple[set[int], dict[int, dict[str, Any]]]:
    token_ids = [int(row["token_id"]) for row in token_rows]
    token_frame = {int(row["token_id"]): int(float(row["frame_id"])) for row in token_rows}
    uf = _UnionFind(token_ids)
    candidate_edges = [
        edge
        for edge in edge_rows
        if edge.get("same_frame_cannot_link") != "True"
        and float(edge.get("semantic_affinity", "0")) >= float(min_affinity)
        and int(edge["token_i"]) in token_frame
        and int(edge["token_j"]) in token_frame
    ]
    if mode == "threshold":
        for edge in candidate_edges:
            uf.union(int(edge["token_i"]), int(edge["token_j"]))
    elif mode == "mutual_topk":
        neighbors: dict[int, list[tuple[float, int]]] = defaultdict(list)
        for edge in candidate_edges:
            left = int(edge["token_i"])
            right = int(edge["token_j"])
            score = float(edge["semantic_affinity"])
            neighbors[left].append((score, right))
            neighbors[right].append((score, left))
        top_neighbors: dict[int, set[int]] = {}
        for token_id, rows in neighbors.items():
            rows.sort(key=lambda item: item[0], reverse=True)
            top_neighbors[int(token_id)] = {int(other) for _score, other in rows[: max(int(topk), 1)]}
        for edge in candidate_edges:
            left = int(edge["token_i"])
            right = int(edge["token_j"])
            if right in top_neighbors.get(left, set()) and left in top_neighbors.get(right, set()):
                uf.union(left, right)
    else:
        raise ValueError(f"unsupported role graph mode: {mode}")
    members_by_root: dict[int, list[int]] = defaultdict(list)
    for token_id in token_ids:
        members_by_root[uf.find(token_id)].append(token_id)
    component_rows: dict[int, dict[str, Any]] = {}
    scene_tokens: set[int] = set()
    n_tokens = max(len(token_ids), 1)
    min_component_tokens = max(int(min_tokens), int(np.ceil(float(min_token_fraction) * n_tokens)))
    for root, members in members_by_root.items():
        frames = {token_frame[token_id] for token_id in members}
        is_scene_role = len(members) >= min_component_tokens and len(frames) >= int(min_frames)
        if is_scene_role:
            scene_tokens.update(int(token_id) for token_id in members)
        component_rows[int(root)] = {
            "component_root": int(root),
            "component_token_count": int(len(members)),
            "component_frame_count": int(len(frames)),
            "component_is_scene_role": bool(is_scene_role),
        }
    return scene_tokens, component_rows


def _source_row(
    *,
    scene: str,
    source: str,
    token_rows: list[dict[str, str]],
    edge_rows: list[dict[str, str]],
    min_affinity: float,
    mode: str,
    topk: int,
    min_token_fraction: float,
    min_tokens: int,
    min_frames: int,
) -> dict[str, Any]:
    source_tokens = [row for row in token_rows if row.get("source") == source]
    source_edges = [row for row in edge_rows if row.get("source") == source]
    scene_tokens, components = _component_roles(
        source_tokens,
        source_edges,
        mode=str(mode),
        min_affinity=float(min_affinity),
        topk=int(topk),
        min_token_fraction=float(min_token_fraction),
        min_tokens=int(min_tokens),
        min_frames=int(min_frames),
    )
    labeled = [row for row in source_edges if row.get("diagnostic_same_gt") in {"True", "False"}]
    all_labels = [row["diagnostic_same_gt"] == "True" for row in labeled]
    all_scores = [float(row["semantic_affinity"]) for row in labeled]
    object_edges = [
        row
        for row in labeled
        if int(row["token_i"]) not in scene_tokens and int(row["token_j"]) not in scene_tokens
    ]
    scene_touch_edges = [
        row
        for row in labeled
        if int(row["token_i"]) in scene_tokens or int(row["token_j"]) in scene_tokens
    ]
    scene_scene_edges = [
        row
        for row in labeled
        if int(row["token_i"]) in scene_tokens and int(row["token_j"]) in scene_tokens
    ]
    positive_edges = [row for row in labeled if row.get("diagnostic_same_gt") == "True"]
    positive_scene_touch = [
        row
        for row in positive_edges
        if int(row["token_i"]) in scene_tokens or int(row["token_j"]) in scene_tokens
    ]
    gt_counter = Counter(
        row.get("diagnostic_gt_instance", "")
        for row in source_tokens
        if int(row["token_id"]) in scene_tokens and row.get("diagnostic_gt_instance") not in {"", "None", "0"}
    )
    largest_components = sorted(components.values(), key=lambda row: row["component_token_count"], reverse=True)[:3]
    return {
        "scene": scene,
        "source": source,
        "role_assignment_uses_gt": False,
        "uses_gt_for_diagnostic_labels": True,
        "role_graph_mode": str(mode),
        "role_topk": int(topk) if str(mode) == "mutual_topk" else "",
        "role_min_affinity": float(min_affinity),
        "role_min_token_fraction": float(min_token_fraction),
        "role_min_tokens": int(min_tokens),
        "role_min_frames": int(min_frames),
        "part_token_count": int(len(source_tokens)),
        "scene_role_token_count": int(len(scene_tokens)),
        "scene_role_token_fraction": float(len(scene_tokens) / max(len(source_tokens), 1)),
        "scene_role_component_count": int(sum(1 for row in components.values() if row["component_is_scene_role"])),
        "largest_component_token_counts": ",".join(str(row["component_token_count"]) for row in largest_components),
        "largest_component_frame_counts": ",".join(str(row["component_frame_count"]) for row in largest_components),
        "all_pair_semantic_affinity_AUC": _auc_fast(all_labels, all_scores),
        "object_local_semantic_affinity_AUC": _auc_fast(
            [row["diagnostic_same_gt"] == "True" for row in object_edges],
            [float(row["semantic_affinity"]) for row in object_edges],
        ),
        "scene_touch_semantic_affinity_AUC": _auc_fast(
            [row["diagnostic_same_gt"] == "True" for row in scene_touch_edges],
            [float(row["semantic_affinity"]) for row in scene_touch_edges],
        ),
        "scene_scene_semantic_affinity_AUC": _auc_fast(
            [row["diagnostic_same_gt"] == "True" for row in scene_scene_edges],
            [float(row["semantic_affinity"]) for row in scene_scene_edges],
        ),
        "gt_labeled_edge_count": int(len(labeled)),
        "object_local_edge_count": int(len(object_edges)),
        "scene_touch_edge_count": int(len(scene_touch_edges)),
        "scene_scene_edge_count": int(len(scene_scene_edges)),
        "positive_edge_count": int(len(positive_edges)),
        "positive_scene_touch_edge_count": int(len(positive_scene_touch)),
        "positive_scene_touch_edge_share": float(len(positive_scene_touch) / max(len(positive_edges), 1)),
        "scene_role_top_gt_ids": ",".join(gt for gt, _count in gt_counter.most_common(3)),
        "scene_role_top_gt_token_counts": ",".join(str(count) for _gt, count in gt_counter.most_common(3)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument("--sources", required=True)
    parser.add_argument("--role-graph-mode", choices=["threshold", "mutual_topk"], default="threshold")
    parser.add_argument("--role-min-affinity", type=float, default=0.60)
    parser.add_argument("--role-topk", type=int, default=4)
    parser.add_argument("--role-min-token-fraction", type=float, default=0.12)
    parser.add_argument("--role-min-tokens", type=int, default=24)
    parser.add_argument("--role-min-frames", type=int, default=4)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    audit_root = Path(args.audit_root)
    scenes = [item.strip() for item in str(args.scenes).split(",") if item.strip()]
    sources = [item.strip() for item in str(args.sources).split(",") if item.strip()]
    out: list[dict[str, Any]] = []
    for scene in scenes:
        token_rows = _read_csv(audit_root / scene / "part_token_rows.csv")
        edge_rows = _read_csv(audit_root / scene / "part_edge_rows.csv")
        for source in sources:
            out.append(
                _source_row(
                    scene=scene,
                    source=source,
                    token_rows=token_rows,
                    edge_rows=edge_rows,
                    mode=str(args.role_graph_mode),
                    min_affinity=float(args.role_min_affinity),
                    topk=int(args.role_topk),
                    min_token_fraction=float(args.role_min_token_fraction),
                    min_tokens=int(args.role_min_tokens),
                    min_frames=int(args.role_min_frames),
                )
            )
    _write_csv(Path(args.output_csv), out)
    print(json.dumps({"output_csv": str(args.output_csv), "row_count": len(out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
