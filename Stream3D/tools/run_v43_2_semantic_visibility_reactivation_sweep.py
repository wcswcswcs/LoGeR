from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tools.run_v26_object_quality_diagnostics import _json_safe, _read_split
from tools.run_v37_4d_if_allowed import (
    SceneState,
    _build_scene_state,
    _labels_for_components,
    _merge_components_rgb_temporal_topk,
)
from tools.run_v37_temporal_curriculum import _load_masks, _load_tubes
from stream4d_native.d4rt_scene_builder import source_xy_from_uv
from tools.run_v43_2_oversplit_merge_residual_sweep import (
    _aggregate_rows,
    _changed_from_base,
    _evaluate_labels,
    _public_rows,
)


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
            writer.writerow({key: row.get(key, "") for key in keys})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _csv_bool(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"true", "1", "yes"}


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, item: int) -> int:
        item = int(item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> bool:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return False
        self.parent[root_right] = root_left
        return True


def _semantic_scene_dir(root: Path, scene: str, source: str, mode: str) -> Path:
    scene_first = root / scene / source / mode
    if scene_first.exists():
        return scene_first
    return root / source / mode


def _load_semantic_mask(mask_dir: Path, source: str, frame_id: int, mask_id: int) -> np.ndarray | None:
    path = mask_dir / f"{source}_frame{int(frame_id):06d}_masks.npz"
    if not path.exists():
        return None
    data = np.load(path)
    masks = np.asarray(data["masks"], dtype=bool)
    index = int(mask_id) - 1
    if index < 0 or index >= int(masks.shape[0]):
        return None
    return np.asarray(masks[index], dtype=bool)


def _map_tokens_to_v37_nodes(
    *,
    scene: str,
    token_rows: list[dict[str, str]],
    semantic_mask_dir: Path,
    semantic_mask_source: str,
    labels_by_frame: dict[int, np.ndarray],
    node_area: dict[int, int],
    min_iou: float,
) -> tuple[dict[int, int], dict[str, Any]]:
    token_to_node: dict[int, int] = {}
    frame_cache: dict[tuple[int, int], np.ndarray | None] = {}
    missing_mask = 0
    missing_frame = 0
    low_iou = 0
    ious: list[float] = []
    for row in token_rows:
        token_id = int(row["token_id"])
        frame_id = int(row["frame_id"])
        mask_id = int(row["mask_id"])
        label = labels_by_frame.get(frame_id)
        if label is None:
            missing_frame += 1
            continue
        cache_key = (frame_id, mask_id)
        if cache_key not in frame_cache:
            frame_cache[cache_key] = _load_semantic_mask(semantic_mask_dir, semantic_mask_source, frame_id, mask_id)
        mask = frame_cache[cache_key]
        if mask is None:
            missing_mask += 1
            continue
        if mask.shape != label.shape:
            mask = cv2.resize(mask.astype(np.uint8), (label.shape[1], label.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        if not np.any(mask):
            low_iou += 1
            continue
        values, counts = np.unique(label[mask], return_counts=True)
        best_node = None
        best_iou = 0.0
        mask_area = int(mask.sum())
        for value, count in zip(values.tolist(), counts.tolist()):
            node_value = int(value) - 1
            if node_value < 0:
                continue
            inter = int(count)
            union = int(mask_area + int(node_area.get(node_value, 0)) - inter)
            if union <= 0:
                continue
            iou = float(inter / union)
            if iou > best_iou:
                best_iou = iou
                best_node = node_value
        if best_node is None or best_iou < float(min_iou):
            low_iou += 1
            continue
        token_to_node[token_id] = int(best_node)
        ious.append(float(best_iou))
    diagnostics = {
        "scene": scene,
        "semantic_mask_dir": str(semantic_mask_dir),
        "token_count": int(len(token_rows)),
        "mapped_token_count": int(len(token_to_node)),
        "missing_semantic_mask_count": int(missing_mask),
        "missing_v37_frame_count": int(missing_frame),
        "low_iou_token_count": int(low_iou),
        "map_iou_min": float(min_iou),
        "map_iou_mean": float(np.mean(np.asarray(ious, dtype=np.float64))) if ious else 0.0,
        "map_iou_p10": float(np.quantile(np.asarray(ious, dtype=np.float64), 0.10)) if ious else 0.0,
    }
    return token_to_node, diagnostics


def _visible_tube(tube: Any, local_idx: int, *, min_visibility: float, min_confidence: float) -> bool:
    uv = np.asarray(tube.uv[local_idx], dtype=np.float32)
    return bool(
        np.isfinite(uv).all()
        and 0.0 <= float(uv[0]) <= 1.0
        and 0.0 <= float(uv[1]) <= 1.0
        and float(tube.visibility[local_idx]) >= float(min_visibility)
        and float(tube.confidence[local_idx]) >= float(min_confidence)
    )


def _map_tokens_to_components_by_tube(
    *,
    scene: str,
    token_rows: list[dict[str, str]],
    semantic_mask_dir: Path,
    semantic_mask_source: str,
    tubes: list[Any],
    base_labels: dict[int, int],
    component_count: int,
    min_token_tubes: int,
    min_component_fraction: float,
    min_visibility: float,
    min_confidence: float,
) -> tuple[dict[int, int], dict[str, Any]]:
    rows_by_frame: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in token_rows:
        rows_by_frame[int(row["frame_id"])].append(row)
    token_masks_by_frame: dict[int, list[tuple[int, np.ndarray]]] = {}
    missing_mask = 0
    for frame_id, rows in rows_by_frame.items():
        current = []
        for row in rows:
            mask = _load_semantic_mask(
                semantic_mask_dir,
                semantic_mask_source,
                int(frame_id),
                int(row["mask_id"]),
            )
            if mask is None:
                missing_mask += 1
                continue
            current.append((int(row["token_id"]), np.asarray(mask, dtype=bool)))
        token_masks_by_frame[int(frame_id)] = current

    support_by_token: dict[int, Counter[int]] = defaultdict(Counter)
    visible_measurements = 0
    component_measurements = 0
    for tube in tubes:
        tube_label = int(base_labels.get(int(tube.tube_id), 0))
        if tube_label <= 0 or tube_label > int(component_count):
            continue
        frames = np.asarray(tube.target_frames_global, dtype=np.int64)
        for local_idx, frame_id in enumerate(frames.tolist()):
            token_masks = token_masks_by_frame.get(int(frame_id))
            if not token_masks:
                continue
            if not _visible_tube(
                tube,
                local_idx,
                min_visibility=float(min_visibility),
                min_confidence=float(min_confidence),
            ):
                continue
            visible_measurements += 1
            shape = token_masks[0][1].shape
            x, y = source_xy_from_uv(tube.uv[local_idx], image_width=shape[1], image_height=shape[0])
            for token_id, mask in token_masks:
                if bool(mask[int(y), int(x)]):
                    support_by_token[int(token_id)][tube_label] += 1
                    component_measurements += 1

    token_to_component: dict[int, int] = {}
    low_support = 0
    low_fraction = 0
    fractions: list[float] = []
    for row in token_rows:
        token_id = int(row["token_id"])
        counter = support_by_token.get(token_id, Counter())
        total = int(sum(counter.values()))
        if total < int(min_token_tubes):
            low_support += 1
            continue
        comp, count = counter.most_common(1)[0]
        fraction = float(count / max(total, 1))
        if fraction < float(min_component_fraction):
            low_fraction += 1
            continue
        token_to_component[token_id] = int(comp) - 1
        fractions.append(float(fraction))
    diagnostics = {
        "scene": scene,
        "semantic_mask_dir": str(semantic_mask_dir),
        "mapping_mode": "tube",
        "token_count": int(len(token_rows)),
        "mapped_token_count": int(len(token_to_component)),
        "missing_semantic_mask_count": int(missing_mask),
        "visible_tube_measurement_count": int(visible_measurements),
        "component_token_measurement_count": int(component_measurements),
        "low_tube_support_token_count": int(low_support),
        "low_component_fraction_token_count": int(low_fraction),
        "min_token_tubes": int(min_token_tubes),
        "min_component_fraction": float(min_component_fraction),
        "component_fraction_mean": float(np.mean(np.asarray(fractions, dtype=np.float64))) if fractions else 0.0,
        "component_fraction_p10": float(np.quantile(np.asarray(fractions, dtype=np.float64), 0.10)) if fractions else 0.0,
    }
    return token_to_component, diagnostics


def _component_frames(state: SceneState, components: list[list[int]]) -> list[set[int]]:
    return [
        {int(state.frame_rank.get(int(state.nodes[int(node_id)].frame_id), int(state.nodes[int(node_id)].frame_id))) for node_id in comp}
        for comp in components
    ]


def _component_support_counts(state: SceneState, components: list[list[int]]) -> list[int]:
    counts = []
    for comp in components:
        tubes: set[int] = set()
        for node_id in comp:
            tubes.update(int(tube_id) for tube_id in state.support_by_region.get(int(node_id), Counter()).keys())
        counts.append(int(len(tubes)))
    return counts


def _semantic_merge_components(
    *,
    state: SceneState,
    base_components: list[list[int]],
    edge_rows: list[dict[str, str]],
    token_to_node: dict[int, int],
    token_to_component: dict[int, int] | None = None,
    semantic_threshold: float,
    object_threshold: float,
    min_edge_support: int,
    max_token_rank_gap: int,
    max_edges_per_component: int,
    max_component_support: int,
) -> tuple[list[list[int]], dict[str, Any], list[dict[str, Any]]]:
    node_to_component = {
        int(node_id): int(comp_id)
        for comp_id, component in enumerate(base_components)
        for node_id in component
    }
    frames = _component_frames(state, base_components)
    support_counts = _component_support_counts(state, base_components)
    semantic_frames = sorted(
        {
            int(row["frame_i"])
            for row in edge_rows
            if str(row.get("frame_i", "")).strip()
        }
        | {
            int(row["frame_j"])
            for row in edge_rows
            if str(row.get("frame_j", "")).strip()
        }
    )
    semantic_rank = {frame_id: rank for rank, frame_id in enumerate(semantic_frames)}
    pair_best: dict[tuple[int, int], dict[str, Any]] = {}
    considered_edges = 0
    mapped_edges = 0
    rejected_same_frame = 0
    rejected_rank_gap = 0
    rejected_threshold = 0
    rejected_nonlocal = 0
    for row in edge_rows:
        if _csv_bool(row, "same_frame_cannot_link"):
            rejected_same_frame += 1
            continue
        semantic = _csv_float(row, "semantic_affinity")
        obj = _csv_float(row, "object_affinity")
        if semantic < float(semantic_threshold) or obj < float(object_threshold):
            rejected_threshold += 1
            continue
        considered_edges += 1
        token_i = int(row["token_i"])
        token_j = int(row["token_j"])
        if token_to_component is None:
            node_i = token_to_node.get(token_i)
            node_j = token_to_node.get(token_j)
            if node_i is None or node_j is None:
                continue
            comp_i = node_to_component.get(node_i)
            comp_j = node_to_component.get(node_j)
        else:
            comp_i = token_to_component.get(token_i)
            comp_j = token_to_component.get(token_j)
        if comp_i is None or comp_j is None or comp_i == comp_j:
            continue
        mapped_edges += 1
        frame_i = int(row["frame_i"])
        frame_j = int(row["frame_j"])
        rank_gap = abs(int(semantic_rank.get(frame_i, frame_i)) - int(semantic_rank.get(frame_j, frame_j)))
        if rank_gap > int(max_token_rank_gap):
            rejected_rank_gap += 1
            continue
        if min(int(support_counts[comp_i]), int(support_counts[comp_j])) > int(max_component_support):
            rejected_nonlocal += 1
            continue
        left, right = sorted((int(comp_i), int(comp_j)))
        score = float(semantic + obj)
        key = (left, right)
        current = pair_best.get(key)
        if current is None:
            pair_best[key] = {
                "left": left,
                "right": right,
                "score": score,
                "semantic_affinity_max": float(semantic),
                "object_affinity_max": float(obj),
                "edge_support": 1,
                "rank_gap_min": int(rank_gap),
            }
        else:
            current["edge_support"] = int(current["edge_support"]) + 1
            current["score"] = max(float(current["score"]), score)
            current["semantic_affinity_max"] = max(float(current["semantic_affinity_max"]), float(semantic))
            current["object_affinity_max"] = max(float(current["object_affinity_max"]), float(obj))
            current["rank_gap_min"] = min(int(current["rank_gap_min"]), int(rank_gap))

    pair_rows = [
        row for row in pair_best.values() if int(row["edge_support"]) >= int(min_edge_support)
    ]
    by_component: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        by_component[int(row["left"])].append(row)
        by_component[int(row["right"])].append(row)
    selected_keys: set[tuple[int, int]] = set()
    for rows in by_component.values():
        for row in sorted(rows, key=lambda item: (float(item["score"]), int(item["edge_support"])), reverse=True)[
            : int(max_edges_per_component)
        ]:
            selected_keys.add((int(row["left"]), int(row["right"])))
    selected_rows = [row for row in pair_rows if (int(row["left"]), int(row["right"])) in selected_keys]

    uf = _UnionFind(len(base_components))
    root_frames: dict[int, set[int]] = {idx: set(frame_set) for idx, frame_set in enumerate(frames)}
    accepted = 0
    rejected_dynamic_same_frame = 0
    for row in sorted(selected_rows, key=lambda item: (float(item["score"]), int(item["edge_support"])), reverse=True):
        left = int(row["left"])
        right = int(row["right"])
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
    for comp_id, component in enumerate(base_components):
        merged[uf.find(comp_id)].extend(component)

    info = {
        "semantic_edge_count": int(len(edge_rows)),
        "semantic_considered_edges": int(considered_edges),
        "semantic_mapped_edges": int(mapped_edges),
        "semantic_candidate_pairs": int(len(pair_best)),
        "semantic_supported_candidate_pairs": int(len(pair_rows)),
        "semantic_selected_pairs": int(len(selected_rows)),
        "semantic_accepted_merges": int(accepted),
        "semantic_rejected_same_frame": int(rejected_same_frame),
        "semantic_rejected_rank_gap": int(rejected_rank_gap),
        "semantic_rejected_threshold": int(rejected_threshold),
        "semantic_rejected_nonlocal": int(rejected_nonlocal),
        "semantic_rejected_dynamic_same_frame": int(rejected_dynamic_same_frame),
        "semantic_threshold": float(semantic_threshold),
        "semantic_object_threshold": float(object_threshold),
        "semantic_min_edge_support": int(min_edge_support),
        "semantic_max_token_rank_gap": int(max_token_rank_gap),
        "semantic_max_edges_per_component": int(max_edges_per_component),
        "semantic_max_component_support": int(max_component_support),
        "memory_strategy": "v42_semantic_visibility_reactivation",
    }
    return list(merged.values()), info, selected_rows


def _variant_name(
    *,
    map_iou: float,
    semantic: float,
    obj: float,
    min_edges: int,
    rank_gap: int,
    max_support: int,
) -> str:
    return (
        f"SVR_iou{map_iou:.2f}_sem{semantic:.2f}_obj{obj:.2f}_e{min_edges}_rg{rank_gap}_sup{max_support}"
        .replace(".", "p")
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    setattr(_load_masks, "max_regions_per_scene", int(args.max_regions_per_scene))
    scenes = _read_split(Path(args.split))
    states: list[SceneState] = []
    pair_row_count = 0
    for scene in scenes:
        state, pair_row_count = _build_scene_state(scene, args, pair_row_count)
        states.append(state)

    map_ious = [float(v) for v in str(args.map_ious).split(",") if v]
    semantics = [float(v) for v in str(args.semantic_thresholds).split(",") if v]
    objects = [float(v) for v in str(args.object_thresholds).split(",") if v]
    min_edges_values = [int(v) for v in str(args.min_edge_supports).split(",") if v]
    rank_gaps = [int(v) for v in str(args.max_token_rank_gaps).split(",") if v]
    max_supports = [int(v) for v in str(args.max_component_supports).split(",") if v]

    base_by_scene: dict[str, tuple[list[list[int]], dict[int, int]]] = {}
    scene_rows: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []
    selected_pair_rows: list[dict[str, Any]] = []
    token_rows_by_scene_source: dict[tuple[str, str], list[dict[str, str]]] = {}
    edge_rows_by_scene_source: dict[tuple[str, str], list[dict[str, str]]] = {}
    for state in states:
        base_components, _base_memory_info = _merge_components_rgb_temporal_topk(
            state,
            state.components,
            min_rgb_similarity=float(args.i4_rgb),
            max_frame_gap=int(args.i4_gap),
            max_rgb_fallback_per_component=int(args.i4_topk),
        )
        base_labels, _ = _labels_for_components(
            base_components,
            state.support_by_tube,
            state.observation_count_by_tube,
            state.gt_labels,
            min_support=1,
            min_fraction=float(state.adaptive_fraction),
        )
        base_by_scene[state.scene] = (base_components, base_labels)
        scene_dir = Path(args.part_graph_root) / state.scene
        token_path = scene_dir / "part_token_rows.csv"
        edge_path = scene_dir / "part_edge_rows.csv"
        if token_path.exists() and edge_path.exists():
            token_rows = [row for row in _read_csv(token_path) if str(row.get("source", "")) == str(args.part_source)]
            edge_rows = [row for row in _read_csv(edge_path) if str(row.get("source", "")) == str(args.part_source)]
        else:
            token_rows = []
            edge_rows = []
        token_rows_by_scene_source[(state.scene, str(args.part_source))] = token_rows
        edge_rows_by_scene_source[(state.scene, str(args.part_source))] = edge_rows

    token_node_cache: dict[tuple[str, float], tuple[dict[int, int], dict[int, int] | None, dict[str, Any]]] = {}
    for map_iou in map_ious:
        for state in states:
            token_rows = token_rows_by_scene_source.get((state.scene, str(args.part_source)), [])
            semantic_mask_dir = _semantic_scene_dir(
                Path(args.semantic_mask_root),
                state.scene,
                str(args.semantic_mask_source),
                str(args.semantic_mask_mode),
            )
            if str(args.mapping_mode) == "tube":
                base_components, base_labels = base_by_scene[state.scene]
                tubes = _load_tubes(state.scene, args)
                token_to_component, diagnostics = _map_tokens_to_components_by_tube(
                    scene=state.scene,
                    token_rows=token_rows,
                    semantic_mask_dir=semantic_mask_dir,
                    semantic_mask_source=str(args.semantic_mask_source),
                    tubes=tubes,
                    base_labels=base_labels,
                    component_count=len(base_components),
                    min_token_tubes=int(args.min_token_tubes),
                    min_component_fraction=float(args.min_token_component_fraction),
                    min_visibility=float(args.min_visibility),
                    min_confidence=float(args.min_confidence),
                )
                token_to_node: dict[int, int] = {}
            else:
                nodes, labels_by_frame, _manifest = _load_masks(
                    Path(args.mask_root),
                    state.scene,
                    str(args.source),
                    str(args.mode),
                    int(args.min_region_area),
                )
                node_area = {int(node.node_id): int(node.area) for node in nodes}
                token_to_node, diagnostics = _map_tokens_to_v37_nodes(
                    scene=state.scene,
                    token_rows=token_rows,
                    semantic_mask_dir=semantic_mask_dir,
                    semantic_mask_source=str(args.semantic_mask_source),
                    labels_by_frame=labels_by_frame,
                    node_area=node_area,
                    min_iou=float(map_iou),
                )
                diagnostics["mapping_mode"] = "overlap"
                token_to_component = None
            token_node_cache[(state.scene, float(map_iou))] = (token_to_node, token_to_component, diagnostics)
            map_rows.append(diagnostics)

    for map_iou in map_ious:
        for semantic in semantics:
            for obj in objects:
                for min_edges in min_edges_values:
                    for rank_gap in rank_gaps:
                        for max_support in max_supports:
                            variant = _variant_name(
                                map_iou=map_iou,
                                semantic=semantic,
                                obj=obj,
                                min_edges=min_edges,
                                rank_gap=rank_gap,
                                max_support=max_support,
                            )
                            for state in states:
                                base_components, base_labels = base_by_scene[state.scene]
                                token_to_node, token_to_component, mapping = token_node_cache[(state.scene, float(map_iou))]
                                edge_rows = edge_rows_by_scene_source.get((state.scene, str(args.part_source)), [])
                                components, semantic_info, selected_rows = _semantic_merge_components(
                                    state=state,
                                    base_components=base_components,
                                    edge_rows=edge_rows,
                                    token_to_node=token_to_node,
                                    token_to_component=token_to_component,
                                    semantic_threshold=float(semantic),
                                    object_threshold=float(obj),
                                    min_edge_support=int(min_edges),
                                    max_token_rank_gap=int(rank_gap),
                                    max_edges_per_component=int(args.max_edges_per_component),
                                    max_component_support=int(max_support),
                                )
                                labels, unknown_ratio = _labels_for_components(
                                    components,
                                    state.support_by_tube,
                                    state.observation_count_by_tube,
                                    state.gt_labels,
                                    min_support=1,
                                    min_fraction=float(state.adaptive_fraction),
                                )
                                labeled = len([tid for tid, gt in state.gt_labels.items() if int(gt) > 0])
                                info = {
                                    **semantic_info,
                                    **_changed_from_base(state, base_labels, labels),
                                    "unknown_count": int(round(float(unknown_ratio) * labeled)),
                                    "mapped_token_count": int(mapping["mapped_token_count"]),
                                    "token_count": int(mapping["token_count"]),
                                    "map_iou_min": float(map_iou),
                                    "mapping_mode": str(args.mapping_mode),
                                }
                                scene_rows.append(_evaluate_labels(state, variant, components, labels, info))
                                for row in selected_rows[: int(args.max_selected_pair_rows_per_scene)]:
                                    selected_pair_rows.append({"scene": state.scene, "variant": variant, **row})

    summary_rows = _aggregate_rows(scene_rows)
    for row in summary_rows:
        row["semantic_phase_gate_proxy_pass"] = bool(
            float(row.get("4D_ARI") or -999.0) >= 0.42599481039581194 + 0.035
            and float(row.get("4D_completeness") or -999.0) >= 0.5056972999752292 + 0.015
            and float(row.get("4D_purity") or -999.0) >= 0.8673519940549913 - 0.003
            and float(row.get("changed_object_ratio") or 999.0) <= 0.20
        )
        row["minimum_significant_gate_pass"] = bool(
            float(row.get("4D_ARI") or -999.0) >= 0.485
            and float(row.get("4D_purity") or -999.0) >= 0.875
            and float(row.get("4D_completeness") or -999.0) >= 0.555
            and float(row.get("temporal_span_mean") or -999.0) >= 1.70
            and float(row.get("scene0081_ARI") or -999.0) >= 0.270
            and float(row.get("mean_predictions_per_scene") or 999.0) <= 150.0
        )
    best_by_ari = max(summary_rows, key=lambda row: float(row.get("4D_ARI") or -999.0), default={})
    best_by_scene0081 = max(summary_rows, key=lambda row: float(row.get("scene0081_ARI") or -999.0), default={})
    passing = [row for row in summary_rows if row.get("semantic_phase_gate_proxy_pass")]
    significant = [row for row in summary_rows if row.get("minimum_significant_gate_pass")]
    payload = {
        "phase": "v43_2_semantic_visibility_reactivation_sweep",
        "status": "PASS_SEMANTIC_VISIBILITY_REACTIVATION_SWEEP" if significant else "NO_GO_SEMANTIC_VISIBILITY_REACTIVATION_SWEEP",
        "variant_count": int(len(summary_rows)),
        "scene_count": int(len(states)),
        "best_by_ari": best_by_ari,
        "best_by_scene0081": best_by_scene0081,
        "passing_semantic_phase_proxy_count": int(len(passing)),
        "passing_minimum_significant_count": int(len(significant)),
        "policy": {
            "prediction_uses_gt": False,
            "gt_used_only_for_diagnostic_precision_and_scoring": True,
            "semantic_source": str(args.part_source),
            "semantic_mask_root": str(args.semantic_mask_root),
            "mapping_mode": str(args.mapping_mode),
            "residual_scope": "mapped frozen v42 semantic token edges used as local visibility reactivation merges over v37 I4 components",
            "global_all_pair_object_graph": False,
        },
    }
    _write_json(output_root / "semantic_visibility_reactivation_summary.json", payload)
    _write_csv(output_root / "semantic_visibility_reactivation_summary_rows.csv", summary_rows)
    _write_csv(output_root / "semantic_visibility_reactivation_scene_rows.csv", _public_rows(scene_rows))
    _write_csv(output_root / "semantic_visibility_reactivation_mapping_rows.csv", map_rows)
    _write_csv(output_root / "semantic_visibility_reactivation_selected_pair_rows.csv", selected_pair_rows)
    print(json.dumps(_json_safe(payload), indent=2, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--mask-root", default="outputs/audit/v37_dino_compact_filter_sources")
    parser.add_argument("--source", default="v37_dino_compact_filter")
    parser.add_argument("--mode", default="dino_k2_compact060_filter")
    parser.add_argument("--part-graph-root", default="outputs/audit/v42_structure_affinity_twohop_backfill8_max480_r1")
    parser.add_argument("--part-source", default="dinov2_maskcut")
    parser.add_argument("--semantic-mask-root", default="outputs/audit/v42_source_audit_external")
    parser.add_argument("--semantic-mask-source", default="dinov2_maskcut")
    parser.add_argument("--semantic-mask-mode", default="sample8")
    parser.add_argument("--mapping-mode", choices=["overlap", "tube"], default="overlap")
    parser.add_argument("--local-decision", default="outputs/audit/v37_adaptive_density_final_probe5/v37_final_decision/decision_summary.json")
    parser.add_argument("--output-root", default="outputs/audit/v43_2_semantic_visibility_reactivation_sweep")
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
    parser.add_argument("--seed", type=int, default=4326)
    parser.add_argument("--i4-rgb", type=float, default=0.99)
    parser.add_argument("--i4-gap", type=int, default=2)
    parser.add_argument("--i4-topk", type=int, default=1)
    parser.add_argument("--map-ious", default="0.05,0.10,0.20")
    parser.add_argument("--semantic-thresholds", default="0.70,0.80,0.90")
    parser.add_argument("--object-thresholds", default="0.20,0.35,0.50")
    parser.add_argument("--min-edge-supports", default="1,2")
    parser.add_argument("--max-token-rank-gaps", default="1,2")
    parser.add_argument("--max-component-supports", default="12,24,48")
    parser.add_argument("--max-edges-per-component", type=int, default=1)
    parser.add_argument("--max-selected-pair-rows-per-scene", type=int, default=50)
    parser.add_argument("--min-token-tubes", type=int, default=1)
    parser.add_argument("--min-token-component-fraction", type=float, default=0.50)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
