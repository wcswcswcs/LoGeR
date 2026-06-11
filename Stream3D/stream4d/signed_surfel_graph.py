from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .measurement_bank import MeasurementBank, json_safe


EDGE_TYPE_IDS = {
    "E_2d_grid": 0,
    "E_2d_knn": 1,
    "E_cross_frame_consistency": 2,
}
EDGE_TYPE_NAMES = {value: key for key, value in EDGE_TYPE_IDS.items()}


@dataclass
class SignedSurfelGraph:
    scene: str
    num_nodes: int
    src: np.ndarray
    dst: np.ndarray
    edge_type: np.ndarray
    num_visible_together: np.ndarray
    mean_uv_distance: np.ndarray
    median_uv_distance: np.ndarray
    mean_rgb_distance: np.ndarray
    trajectory_relative_motion_variance: np.ndarray
    precut_keep: np.ndarray
    meta: dict[str, Any]

    @property
    def num_edges(self) -> int:
        return int(self.src.shape[0])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            scene=np.asarray(self.scene),
            num_nodes=np.asarray(self.num_nodes, dtype=np.int64),
            src=self.src.astype(np.int64, copy=False),
            dst=self.dst.astype(np.int64, copy=False),
            edge_type=self.edge_type.astype(np.int16, copy=False),
            num_visible_together=self.num_visible_together.astype(np.int16, copy=False),
            mean_uv_distance=self.mean_uv_distance.astype(np.float32, copy=False),
            median_uv_distance=self.median_uv_distance.astype(np.float32, copy=False),
            mean_rgb_distance=self.mean_rgb_distance.astype(np.float32, copy=False),
            trajectory_relative_motion_variance=self.trajectory_relative_motion_variance.astype(np.float32, copy=False),
            precut_keep=self.precut_keep.astype(bool, copy=False),
            meta_json=np.asarray(json.dumps(json_safe(self.meta), sort_keys=True)),
        )

    @classmethod
    def load(cls, path: Path) -> "SignedSurfelGraph":
        with np.load(path, allow_pickle=False) as data:
            precut_keep = (
                np.asarray(data["precut_keep"], dtype=bool)
                if "precut_keep" in data.files
                else np.ones(np.asarray(data["src"]).shape, dtype=bool)
            )
            return cls(
                scene=str(data["scene"].item()),
                num_nodes=int(data["num_nodes"].item()),
                src=np.asarray(data["src"], dtype=np.int64),
                dst=np.asarray(data["dst"], dtype=np.int64),
                edge_type=np.asarray(data["edge_type"], dtype=np.int16),
                num_visible_together=np.asarray(data["num_visible_together"], dtype=np.int16),
                mean_uv_distance=np.asarray(data["mean_uv_distance"], dtype=np.float32),
                median_uv_distance=np.asarray(data["median_uv_distance"], dtype=np.float32),
                mean_rgb_distance=np.asarray(data["mean_rgb_distance"], dtype=np.float32),
                trajectory_relative_motion_variance=np.asarray(
                    data["trajectory_relative_motion_variance"], dtype=np.float32
                ),
                precut_keep=precut_keep,
                meta=json.loads(str(data["meta_json"].item())),
            )


def _add_edge(edges: dict[tuple[int, int, int], None], a: int, b: int, edge_type: str) -> None:
    if int(a) == int(b):
        return
    lo, hi = sorted((int(a), int(b)))
    edges[(lo, hi, EDGE_TYPE_IDS[edge_type])] = None


def _grid_edges_from_source_layout(bank: MeasurementBank, *, max_diagonal: bool = True) -> dict[tuple[int, int, int], None]:
    edges: dict[tuple[int, int, int], None] = {}
    by_frame: dict[int, list[int]] = defaultdict(list)
    for idx, frame_id in enumerate(np.asarray(bank.src_frame_global, dtype=np.int64).tolist()):
        by_frame[int(frame_id)].append(int(idx))

    neighbor_offsets = [(0, 1), (1, 0)]
    if max_diagonal:
        neighbor_offsets.extend([(1, 1), (1, -1)])

    src_xy = np.asarray(bank.src_xy, dtype=np.int64)
    for indices in by_frame.values():
        if len(indices) < 4:
            continue
        arr = np.asarray(indices, dtype=np.int64)
        xy = src_xy[arr]
        order = np.lexsort((xy[:, 0], xy[:, 1]))
        arr = arr[order]
        side = int(round(np.sqrt(arr.shape[0])))
        if side * side != arr.shape[0]:
            side = int(np.floor(np.sqrt(arr.shape[0])))
            usable = side * side
            if usable <= 0:
                continue
            arr = arr[:usable]
        grid = arr.reshape(side, side)
        for y in range(side):
            for x in range(side):
                a = int(grid[y, x])
                for dy, dx in neighbor_offsets:
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < side and 0 <= xx < side:
                        _add_edge(edges, a, int(grid[yy, xx]), "E_2d_grid")
    return edges


def _knn_edges_for_frames(
    bank: MeasurementBank,
    *,
    k: int,
    max_frames: int,
    max_nodes_per_frame: int,
) -> dict[tuple[int, int, int], None]:
    edges: dict[tuple[int, int, int], None] = {}
    uv = np.asarray(bank.uv_pred, dtype=np.float32)
    visible = np.asarray(bank.visible_ok, dtype=bool)
    frames = range(min(int(max_frames), uv.shape[0])) if max_frames > 0 else range(uv.shape[0])
    try:
        from scipy.spatial import cKDTree
    except Exception:
        cKDTree = None

    for frame_idx in frames:
        indices = np.flatnonzero(visible[frame_idx])
        if indices.size <= 1:
            continue
        if max_nodes_per_frame > 0 and indices.size > int(max_nodes_per_frame):
            take = np.linspace(0, indices.size - 1, int(max_nodes_per_frame), dtype=np.int64)
            indices = indices[take]
        pts = uv[frame_idx, indices]
        k_eff = min(int(k) + 1, pts.shape[0])
        if cKDTree is not None:
            _, neigh = cKDTree(pts).query(pts, k=k_eff)
        else:
            # Tiny-test fallback.
            dist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
            neigh = np.argsort(dist, axis=1)[:, :k_eff]
        if neigh.ndim == 1:
            neigh = neigh[:, None]
        for local, row in enumerate(neigh):
            a = int(indices[local])
            for n in row.tolist():
                b = int(indices[int(n)])
                if b != a:
                    _add_edge(edges, a, b, "E_2d_knn")
    return edges


def _cross_frame_edges(bank: MeasurementBank, *, max_neighbors: int, max_edges: int) -> dict[tuple[int, int, int], None]:
    edges: dict[tuple[int, int, int], None] = {}
    mean_uv = _mean_visible_uv(bank)
    visible_count = np.asarray(bank.visible_ok, dtype=bool).sum(axis=0)
    candidates = np.flatnonzero(visible_count > 0)
    if candidates.size <= 1:
        return edges
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return edges
    k_eff = min(int(max_neighbors) + 1, candidates.shape[0])
    _, neigh = cKDTree(mean_uv[candidates]).query(mean_uv[candidates], k=k_eff)
    if neigh.ndim == 1:
        neigh = neigh[:, None]
    added = 0
    for local, row in enumerate(neigh):
        a = int(candidates[local])
        for n in row.tolist():
            b = int(candidates[int(n)])
            if a == b:
                continue
            together = np.count_nonzero(np.asarray(bank.visible_ok[:, a], dtype=bool) & np.asarray(bank.visible_ok[:, b], dtype=bool))
            if together < 3:
                continue
            _add_edge(edges, a, b, "E_cross_frame_consistency")
            added += 1
            if max_edges > 0 and added >= int(max_edges):
                return edges
    return edges


def _mean_visible_uv(bank: MeasurementBank) -> np.ndarray:
    uv = np.asarray(bank.uv_pred, dtype=np.float32)
    visible = np.asarray(bank.visible_ok, dtype=bool)
    out = np.zeros((bank.num_surfels, 2), dtype=np.float32)
    for idx in range(bank.num_surfels):
        ok = visible[:, idx]
        if np.any(ok):
            out[idx] = np.mean(uv[ok, idx], axis=0)
    return out


def _edge_metrics(bank: MeasurementBank, src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    uv = np.asarray(bank.uv_pred, dtype=np.float32)
    visible = np.asarray(bank.visible_ok, dtype=bool)
    rgb = np.asarray(bank.src_rgb, dtype=np.float32)
    num_visible = np.zeros((src.shape[0],), dtype=np.int16)
    mean_dist = np.zeros((src.shape[0],), dtype=np.float32)
    median_dist = np.zeros((src.shape[0],), dtype=np.float32)
    rgb_dist = np.linalg.norm(rgb[src] - rgb[dst], axis=1).astype(np.float32)
    motion_var = np.zeros((src.shape[0],), dtype=np.float32)
    for idx, (a, b) in enumerate(zip(src.tolist(), dst.tolist())):
        ok = visible[:, a] & visible[:, b]
        num_visible[idx] = int(np.count_nonzero(ok))
        if not np.any(ok):
            mean_dist[idx] = 0.0
            median_dist[idx] = 0.0
            motion_var[idx] = 0.0
            continue
        delta = uv[ok, a] - uv[ok, b]
        dist = np.linalg.norm(delta, axis=1)
        mean_dist[idx] = float(np.mean(dist))
        median_dist[idx] = float(np.median(dist))
        motion_var[idx] = float(np.mean(np.var(delta, axis=0))) if delta.shape[0] > 1 else 0.0
    return num_visible, mean_dist, median_dist, rgb_dist, motion_var


def _precut_keep_edges(
    bank: MeasurementBank,
    src: np.ndarray,
    dst: np.ndarray,
    mean_uv_distance: np.ndarray,
    mean_rgb_distance: np.ndarray,
    *,
    mask_disagreement_ratio: float,
    source_rgb_discontinuity: float,
    uv_discontinuity: float,
) -> np.ndarray:
    visible = np.asarray(bank.visible_ok, dtype=bool)
    target_mask = np.asarray(bank.target_mask_id, dtype=np.int64)
    src_mask = np.asarray(bank.src_mask_id, dtype=np.int64)
    keep = np.ones((src.shape[0],), dtype=bool)
    if src.shape[0] == 0:
        return keep
    for idx, (a, b) in enumerate(zip(src.tolist(), dst.tolist())):
        ok = visible[:, a] & visible[:, b]
        if np.any(ok):
            ids_a = target_mask[ok, a]
            ids_b = target_mask[ok, b]
            pos_a = ids_a > 0
            pos_b = ids_b > 0
            disagreement = (pos_a & pos_b & (ids_a != ids_b)) | (pos_a ^ pos_b)
            if float(np.mean(disagreement.astype(np.float32))) >= float(mask_disagreement_ratio):
                keep[idx] = False
                continue
        both_source_positive = int(src_mask[a]) > 0 and int(src_mask[b]) > 0
        source_disagrees = both_source_positive and int(src_mask[a]) != int(src_mask[b])
        strong_appearance_jump = float(mean_rgb_distance[idx]) >= float(source_rgb_discontinuity)
        strong_uv_jump = float(mean_uv_distance[idx]) >= float(uv_discontinuity)
        if source_disagrees and (strong_appearance_jump or strong_uv_jump):
            keep[idx] = False
        elif strong_appearance_jump and strong_uv_jump:
            keep[idx] = False
    return keep


def connected_component_sizes(
    num_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
    active: np.ndarray | None = None,
) -> list[int]:
    adj: list[list[int]] = [[] for _ in range(int(num_nodes))]
    if active is None:
        active = np.ones((src.shape[0],), dtype=bool)
    for a, b, use_edge in zip(src.tolist(), dst.tolist(), np.asarray(active, dtype=bool).tolist()):
        if not use_edge:
            continue
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    seen = np.zeros((int(num_nodes),), dtype=bool)
    sizes: list[int] = []
    for start in range(int(num_nodes)):
        if seen[start]:
            continue
        seen[start] = True
        q: deque[int] = deque([start])
        size = 0
        while q:
            node = q.popleft()
            size += 1
            for nxt in adj[node]:
                if not seen[nxt]:
                    seen[nxt] = True
                    q.append(nxt)
        sizes.append(size)
    return sizes


def summarize_signed_surfel_graph(graph: SignedSurfelGraph, bank: MeasurementBank | None = None) -> dict[str, Any]:
    edge_counts = {
        EDGE_TYPE_NAMES[int(edge_type)]: int(np.count_nonzero(graph.edge_type == int(edge_type)))
        for edge_type in sorted(set(graph.edge_type.tolist()))
    }
    raw_component_sizes = connected_component_sizes(graph.num_nodes, graph.src, graph.dst)
    raw_largest = max(raw_component_sizes) if raw_component_sizes else 0
    component_sizes = connected_component_sizes(graph.num_nodes, graph.src, graph.dst, graph.precut_keep)
    largest = max(component_sizes) if component_sizes else 0
    row: dict[str, Any] = {
        "scene": graph.scene,
        "status": "ok",
        "num_nodes": int(graph.num_nodes),
        "num_edges": int(graph.num_edges),
        "edge_counts": edge_counts,
        "raw_largest_graph_component_ratio": float(raw_largest / max(graph.num_nodes, 1)),
        "largest_graph_component_ratio": float(largest / max(graph.num_nodes, 1)),
        "num_components": int(len(component_sizes)),
        "raw_num_components": int(len(raw_component_sizes)),
        "precut_removed_edge_ratio": float(1.0 - np.count_nonzero(graph.precut_keep) / max(graph.num_edges, 1)),
        "mean_visible_together_per_edge": float(np.mean(graph.num_visible_together)) if graph.num_edges else 0.0,
        "mean_uv_distance": float(np.mean(graph.mean_uv_distance)) if graph.num_edges else 0.0,
        "median_uv_distance": float(np.median(graph.mean_uv_distance)) if graph.num_edges else 0.0,
        "mean_rgb_distance": float(np.mean(graph.mean_rgb_distance)) if graph.num_edges else 0.0,
        "trajectory_relative_motion_variance_mean": float(np.mean(graph.trajectory_relative_motion_variance))
        if graph.num_edges
        else 0.0,
    }
    if bank is not None:
        visible_counts = np.asarray(bank.visible_ok, dtype=bool).sum(axis=0).astype(np.float64)
        positive_counts = np.asarray(bank.positive_observation, dtype=bool).sum(axis=0).astype(np.float64)
        uv_in01 = (
            np.asarray(bank.valid, dtype=bool)
            & np.isfinite(bank.uv_pred).all(axis=2)
            & (bank.uv_pred[:, :, 0] >= 0.0)
            & (bank.uv_pred[:, :, 0] <= 1.0)
            & (bank.uv_pred[:, :, 1] >= 0.0)
            & (bank.uv_pred[:, :, 1] <= 1.0)
        )
        row.update(
            {
                "num_visible_surfels_per_frame_mean": float(np.mean(np.asarray(bank.visible_ok, dtype=bool).sum(axis=1))),
                "track_length_visible_mean": float(np.mean(visible_counts)) if visible_counts.size else 0.0,
                "track_length_visible_p10": float(np.percentile(visible_counts, 10)) if visible_counts.size else 0.0,
                "track_length_visible_p90": float(np.percentile(visible_counts, 90)) if visible_counts.size else 0.0,
                "uv_in01_rate": float(np.count_nonzero(uv_in01) / max(int(uv_in01.size), 1)),
                "self_uv_error_p90": bank.meta.get("self_uv_error_p90_mean"),
                "cycle_uv_error_p90": bank.meta.get("cycle_uv_error_p90_mean"),
                "surfel_coverage_2d_per_frame": float(np.mean(np.asarray(bank.visible_ok, dtype=bool).sum(axis=1) / max(bank.num_surfels, 1))),
                "unobserved_surfel_ratio": float(np.count_nonzero(positive_counts == 0) / max(bank.num_surfels, 1)),
                "ambiguous_surfel_ratio": float(
                    np.count_nonzero((np.asarray(bank.src_mask_id) > 0) & (positive_counts > 0) & (np.asarray(bank.negative_observation, dtype=bool).sum(axis=0) > 0))
                    / max(np.count_nonzero(np.asarray(bank.src_mask_id) > 0), 1)
                ),
            }
        )
    row["phase1_gate"] = {
        "num_nodes_mean_ge_10k": bool(row["num_nodes"] >= 10000),
        "visible_track_length_mean_ge_10": bool(row.get("track_length_visible_mean", 0.0) >= 10.0),
        "uv_in01_rate_ge_0p95": bool(row.get("uv_in01_rate", 0.0) >= 0.95),
        "cycle_uv_error_p90_le_5px": bool((row.get("cycle_uv_error_p90") or 999.0) <= 5.0),
        "e_2d_knn_edges_ge_8x_visible_nodes": bool(
            edge_counts.get("E_2d_knn", 0) >= 8.0 * row.get("num_visible_surfels_per_frame_mean", row["num_nodes"])
        ),
        "largest_component_ratio_between_0p3_0p95": bool(0.3 <= row["largest_graph_component_ratio"] <= 0.95),
        "unobserved_surfel_ratio_le_0p05": bool(row.get("unobserved_surfel_ratio", 1.0) <= 0.05),
    }
    row["phase1_pass"] = bool(all(row["phase1_gate"].values()))
    return row


def build_signed_surfel_graph(
    bank: MeasurementBank,
    *,
    knn_k: int = 8,
    knn_max_frames: int = 16,
    knn_max_nodes_per_frame: int = 0,
    include_diagonal_grid: bool = True,
    cross_frame_neighbors: int = 4,
    cross_frame_max_edges: int = 0,
    precut_mask_disagreement_ratio: float = 0.25,
    precut_source_rgb_discontinuity: float = 0.45,
    precut_uv_discontinuity: float = 0.06,
) -> SignedSurfelGraph:
    edges: dict[tuple[int, int, int], None] = {}
    edges.update(_grid_edges_from_source_layout(bank, max_diagonal=include_diagonal_grid))
    edges.update(
        _knn_edges_for_frames(
            bank,
            k=int(knn_k),
            max_frames=int(knn_max_frames),
            max_nodes_per_frame=int(knn_max_nodes_per_frame),
        )
    )
    if int(cross_frame_neighbors) > 0:
        edges.update(
            _cross_frame_edges(
                bank,
                max_neighbors=int(cross_frame_neighbors),
                max_edges=int(cross_frame_max_edges),
            )
        )
    triples = np.asarray(sorted(edges.keys()), dtype=np.int64) if edges else np.zeros((0, 3), dtype=np.int64)
    src = triples[:, 0].astype(np.int64, copy=False)
    dst = triples[:, 1].astype(np.int64, copy=False)
    edge_type = triples[:, 2].astype(np.int16, copy=False)
    visible, mean_dist, median_dist, rgb_dist, motion_var = _edge_metrics(bank, src, dst)
    precut_keep = _precut_keep_edges(
        bank,
        src,
        dst,
        mean_dist,
        rgb_dist,
        mask_disagreement_ratio=float(precut_mask_disagreement_ratio),
        source_rgb_discontinuity=float(precut_source_rgb_discontinuity),
        uv_discontinuity=float(precut_uv_discontinuity),
    )
    return SignedSurfelGraph(
        scene=bank.scene,
        num_nodes=bank.num_surfels,
        src=src,
        dst=dst,
        edge_type=edge_type,
        num_visible_together=visible,
        mean_uv_distance=mean_dist,
        median_uv_distance=median_dist,
        mean_rgb_distance=rgb_dist,
        trajectory_relative_motion_variance=motion_var,
        precut_keep=precut_keep,
        meta={
            "algorithm": "v18_signed_surfel_graph",
            "knn_k": int(knn_k),
            "knn_max_frames": int(knn_max_frames),
            "knn_max_nodes_per_frame": int(knn_max_nodes_per_frame),
            "include_diagonal_grid": bool(include_diagonal_grid),
            "cross_frame_neighbors": int(cross_frame_neighbors),
            "cross_frame_max_edges": int(cross_frame_max_edges),
            "precut_mask_disagreement_ratio": float(precut_mask_disagreement_ratio),
            "precut_source_rgb_discontinuity": float(precut_source_rgb_discontinuity),
            "precut_uv_discontinuity": float(precut_uv_discontinuity),
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic": False,
        },
    )
