from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from stream4d.export_scannet import ScanNetExporter
from stream4d.measurement_bank import MeasurementBank, json_safe, read_seq_list
from stream4d.scannet_stream import ScanNetStream
from stream4d.signed_graph_partition import _mask_votes
from stream4d.signed_surfel_graph import SignedSurfelGraph
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


def _load_scene_points(stream: ScanNetStream) -> np.ndarray:
    import open3d as o3d

    return np.asarray(o3d.io.read_point_cloud(str(stream.mesh_path)).points, dtype=np.float32)


def _backproject_uv_to_mesh_with_dist(
    stream: ScanNetStream,
    scene_points: np.ndarray,
    scene_tree: Any,
    frame_id: int,
    uv: np.ndarray,
    *,
    nn_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    out = np.full((uv.shape[0],), -1, dtype=np.int64)
    dist_out = np.full((uv.shape[0],), np.inf, dtype=np.float32)
    if uv.size == 0:
        return out, dist_out
    depth = stream.load_depth(int(frame_id))
    pose = stream.load_pose(int(frame_id))
    if not np.isfinite(pose).all():
        return out, dist_out
    intr = stream.load_intrinsics()
    h, w = depth.shape
    x = np.rint(uv[:, 0] * float(max(w - 1, 1))).astype(np.int64)
    y = np.rint(uv[:, 1] * float(max(h - 1, 1))).astype(np.int64)
    in_bounds = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if not np.any(in_bounds):
        return out, dist_out
    z = depth[y[in_bounds], x[in_bounds]]
    valid = np.isfinite(z) & (z > 0.0)
    if not np.any(valid):
        return out, dist_out
    x_valid = x[in_bounds][valid].astype(np.float32)
    y_valid = y[in_bounds][valid].astype(np.float32)
    z_valid = z[valid].astype(np.float32)
    fx = float(intr[0, 0])
    fy = float(intr[1, 1])
    cx = float(intr[0, 2])
    cy = float(intr[1, 2])
    cam = np.stack(
        [(x_valid - cx) * z_valid / fx, (y_valid - cy) * z_valid / fy, z_valid, np.ones_like(z_valid)],
        axis=1,
    )
    world = (pose @ cam.T).T[:, :3].astype(np.float32)
    finite = np.isfinite(world).all(axis=1)
    if not np.any(finite):
        return out, dist_out
    dist, idx = scene_tree.query(world[finite], k=1, distance_upper_bound=float(nn_radius))
    hit = np.isfinite(dist) & (idx < scene_points.shape[0])
    original_indices = np.flatnonzero(in_bounds)[valid][finite]
    out[original_indices[hit]] = idx[hit].astype(np.int64)
    dist_out[original_indices[hit]] = dist[hit].astype(np.float32)
    return out, dist_out


def _vote_entropy(counts: Counter[int]) -> float:
    total = float(sum(counts.values()))
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = float(count) / total
        if p > 0.0:
            entropy -= p * math.log(p)
    return float(entropy)


def _surfel_mesh_hits(
    bank: MeasurementBank,
    stream: ScanNetStream,
    scene_points: np.ndarray,
    scene_tree: Any,
    gt_ids: np.ndarray,
    *,
    nn_radius: float,
    max_frames: int,
) -> dict[str, Any]:
    visible = np.asarray(bank.visible_ok, dtype=bool)
    frame_ids = np.asarray(bank.frame_ids, dtype=np.int64)
    frames = range(min(int(max_frames), frame_ids.shape[0])) if max_frames > 0 else range(frame_ids.shape[0])
    hits_by_surfel: list[list[int]] = [[] for _ in range(bank.num_surfels)]
    dist_by_surfel: list[list[float]] = [[] for _ in range(bank.num_surfels)]
    frame_hits_by_surfel: list[set[int]] = [set() for _ in range(bank.num_surfels)]
    hit_surfel_index: list[int] = []
    hit_vertex_index: list[int] = []
    hit_frame_index: list[int] = []
    hit_distance: list[float] = []
    total_queries = 0
    total_hits = 0

    for frame_idx in frames:
        surfels = np.flatnonzero(visible[frame_idx])
        if surfels.size == 0:
            continue
        mesh_ids, distances = _backproject_uv_to_mesh_with_dist(
            stream,
            scene_points,
            scene_tree,
            int(frame_ids[frame_idx]),
            np.asarray(bank.uv_pred[frame_idx, surfels], dtype=np.float32),
            nn_radius=float(nn_radius),
        )
        total_queries += int(mesh_ids.shape[0])
        hit = mesh_ids >= 0
        total_hits += int(np.count_nonzero(hit))
        for surfel_idx, mesh_idx, dist in zip(surfels[hit].tolist(), mesh_ids[hit].tolist(), distances[hit].tolist()):
            hits_by_surfel[int(surfel_idx)].append(int(mesh_idx))
            dist_by_surfel[int(surfel_idx)].append(float(dist))
            frame_hits_by_surfel[int(surfel_idx)].add(int(frame_idx))
            hit_surfel_index.append(int(surfel_idx))
            hit_vertex_index.append(int(mesh_idx))
            hit_frame_index.append(int(frame_idx))
            hit_distance.append(float(dist))

    labels = np.full((bank.num_surfels,), -1, dtype=np.int64)
    best_vertex = np.full((bank.num_surfels,), -1, dtype=np.int64)
    hit_count = np.zeros((bank.num_surfels,), dtype=np.int32)
    hit_frame_count = np.zeros((bank.num_surfels,), dtype=np.int16)
    median_nn_distance = np.full((bank.num_surfels,), np.nan, dtype=np.float32)
    vote_entropy = np.zeros((bank.num_surfels,), dtype=np.float32)
    gt_vote_count = np.zeros((bank.num_surfels,), dtype=np.int16)
    for idx, vertices in enumerate(hits_by_surfel):
        if not vertices:
            continue
        hit_count[idx] = int(len(vertices))
        hit_frame_count[idx] = int(len(frame_hits_by_surfel[idx]))
        median_nn_distance[idx] = float(np.median(dist_by_surfel[idx]))
        vertex_counts = Counter(int(v) for v in vertices)
        best_vertex[idx] = int(vertex_counts.most_common(1)[0][0])
        label_counts = Counter(int(gt_ids[v]) for v in vertices if int(gt_ids[v]) >= 1000)
        vote_entropy[idx] = _vote_entropy(label_counts)
        if label_counts:
            label, count = label_counts.most_common(1)[0]
            labels[idx] = int(label)
            gt_vote_count[idx] = int(count)

    unique_hit_vertices = np.unique(np.asarray(hit_vertex_index, dtype=np.int64)) if hit_vertex_index else np.empty((0,), dtype=np.int64)
    gt_instances = np.unique(gt_ids[gt_ids >= 1000])
    covered_gt_instances = np.unique(gt_ids[unique_hit_vertices][gt_ids[unique_hit_vertices] >= 1000]) if unique_hit_vertices.size else np.empty((0,), dtype=np.int64)
    return {
        "labels": labels,
        "best_vertex": best_vertex,
        "hit_count": hit_count,
        "hit_frame_count": hit_frame_count,
        "median_nn_distance": median_nn_distance,
        "vote_entropy": vote_entropy,
        "gt_vote_count": gt_vote_count,
        "hits_by_surfel": hits_by_surfel,
        "hit_surfel_index": np.asarray(hit_surfel_index, dtype=np.int64),
        "hit_vertex_index": np.asarray(hit_vertex_index, dtype=np.int64),
        "hit_frame_index": np.asarray(hit_frame_index, dtype=np.int16),
        "hit_distance": np.asarray(hit_distance, dtype=np.float32),
        "surfel_hit_rate": float(np.count_nonzero(hit_count > 0) / max(bank.num_surfels, 1)),
        "mean_hit_frames_per_surfel": float(np.mean(hit_frame_count[hit_frame_count > 0])) if np.any(hit_frame_count > 0) else 0.0,
        "node_gt_label_coverage": float(np.count_nonzero(labels >= 1000) / max(bank.num_surfels, 1)),
        "mesh_vertex_coverage_ratio": float(unique_hit_vertices.shape[0] / max(scene_points.shape[0], 1)),
        "covered_gt_instance_count": int(covered_gt_instances.shape[0]),
        "covered_gt_instance_ratio": float(covered_gt_instances.shape[0] / max(gt_instances.shape[0], 1)),
        "backproject_queries": int(total_queries),
        "backproject_hits": int(total_hits),
        "backproject_hit_rate": float(total_hits / max(total_queries, 1)),
        "nn_radius": float(nn_radius),
        "max_frames": int(max_frames),
    }


def _edge_labels(graph: SignedSurfelGraph, surfel_gt: np.ndarray) -> dict[str, Any]:
    a = surfel_gt[graph.src]
    b = surfel_gt[graph.dst]
    known = (a >= 1000) & (b >= 1000)
    same = known & (a == b)
    cut = known & (a != b)
    return {
        "known": known,
        "same": same,
        "cut": cut,
        "edge_gt_label_coverage": float(np.count_nonzero(known) / max(graph.num_edges, 1)),
        "same_gt_edges": int(np.count_nonzero(same)),
        "cut_gt_edges": int(np.count_nonzero(cut)),
        "unknown_gt_edges": int(graph.num_edges - np.count_nonzero(known)),
    }


def _label_groups(labels: np.ndarray, *, min_surfels: int) -> list[tuple[int, np.ndarray]]:
    groups: list[tuple[int, np.ndarray]] = []
    for label in sorted(int(v) for v in np.unique(labels[labels >= 1000]).tolist()):
        surfels = np.flatnonzero(labels == int(label)).astype(np.int64)
        if surfels.shape[0] >= int(min_surfels):
            groups.append((int(label), surfels))
    return groups


def _edge_oracle_components(
    graph: SignedSurfelGraph,
    labels: np.ndarray,
    *,
    min_surfels: int,
    use_precut: bool,
) -> list[tuple[int, np.ndarray]]:
    adj: list[list[int]] = [[] for _ in range(graph.num_nodes)]
    active = np.asarray(graph.precut_keep, dtype=bool) if use_precut else np.ones((graph.num_edges,), dtype=bool)
    for a, b, keep in zip(graph.src.tolist(), graph.dst.tolist(), active.tolist()):
        if not keep:
            continue
        if labels[int(a)] >= 1000 and labels[int(a)] == labels[int(b)]:
            adj[int(a)].append(int(b))
            adj[int(b)].append(int(a))
    seen = np.zeros((graph.num_nodes,), dtype=bool)
    comps: list[tuple[int, np.ndarray]] = []
    for start in range(graph.num_nodes):
        if seen[start] or labels[start] < 1000:
            continue
        seen[start] = True
        q: deque[int] = deque([start])
        nodes: list[int] = []
        while q:
            node = q.popleft()
            nodes.append(int(node))
            for nxt in adj[node]:
                if not seen[nxt]:
                    seen[nxt] = True
                    q.append(nxt)
        if len(nodes) >= int(min_surfels):
            comps.append((int(labels[start]), np.asarray(nodes, dtype=np.int64)))
    comps.sort(key=lambda item: (-item[1].shape[0], int(item[1][0]) if item[1].size else -1))
    return comps


def _points_for_surfels(hits: dict[str, Any], surfels: np.ndarray, *, mode: str) -> set[int]:
    surfels = np.asarray(surfels, dtype=np.int64)
    point_ids: set[int] = set()
    if surfels.size == 0:
        return point_ids
    if mode == "best":
        best = np.asarray(hits["best_vertex"], dtype=np.int64)[surfels]
        point_ids.update(int(v) for v in best[best >= 0].tolist())
        return point_ids
    if mode == "union":
        hit_lists = hits["hits_by_surfel"]
        for surfel in surfels.tolist():
            point_ids.update(int(v) for v in hit_lists[int(surfel)])
        return point_ids
    raise ValueError(f"Unsupported point mode: {mode}")


def _dilate_points(scene_points: np.ndarray, tree: Any, point_ids: set[int], radius: float) -> set[int]:
    if radius <= 0.0 or not point_ids:
        return set(point_ids)
    seed = np.asarray(sorted(point_ids), dtype=np.int64)
    seed = seed[(seed >= 0) & (seed < scene_points.shape[0])]
    if seed.size == 0:
        return set()
    neighbors = tree.query_ball_point(scene_points[seed], r=float(radius))
    out = set(int(v) for v in seed.tolist())
    for item in neighbors:
        out.update(int(v) for v in item)
    return out


def _object_dict_from_point_groups(groups: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for idx, group in enumerate(groups):
        point_ids = np.asarray(sorted(int(v) for v in group.get("point_ids", set())), dtype=np.int64)
        out[idx] = {
            "point_ids": point_ids,
            "mask_list": list(group.get("mask_list", [])),
            "carrier_ids": np.asarray(group.get("carrier_ids", []), dtype=np.int64),
            "expected_gt_label": int(group.get("expected_gt_label", -1)),
        }
    return out


def _export_point_groups(
    *,
    stream: ScanNetStream,
    groups: list[dict[str, Any]],
    output_config: str,
    min_points_per_object: int,
    export_point_dilate_radius: float = 0.0,
) -> dict[str, Any]:
    exporter = ScanNetExporter(
        stream,
        output_config=output_config,
        export_support_mode="reuse_point_ids",
        export_point_dilate_radius=float(export_point_dilate_radius),
        export_min_points_per_object=int(min_points_per_object),
        export_score_mode="area",
    )
    return exporter.export_object_dict_points(_object_dict_from_point_groups(groups))


def _export_posterior_groups(
    *,
    stream: ScanNetStream,
    bank: MeasurementBank,
    groups: list[tuple[int, np.ndarray]],
    output_config: str,
    min_points_per_object: int,
    core_nn_radius: float,
    fringe_nn_radius: float,
    fringe_radius: float,
    fringe_max_ratio: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    object_dict: dict[int, dict[str, Any]] = {}
    for idx, (label, surfels) in enumerate(groups):
        object_dict[idx] = {
            "mask_list": _mask_votes(bank, surfels, max_votes=8),
            "carrier_ids": surfels,
            "core_surfels": surfels,
            "fringe_surfels": np.empty((0,), dtype=np.int64),
            "unknown_surfels": np.empty((0,), dtype=np.int64),
            "reject_surfels": np.empty((0,), dtype=np.int64),
            "expected_gt_label": int(label),
        }
    exporter = ScanNetExporter(
        stream,
        output_config=output_config,
        export_support_mode="posterior_support",
        export_core_nn_radius=float(core_nn_radius),
        export_fringe_nn_radius=float(fringe_nn_radius),
        export_fringe_radius=float(fringe_radius),
        export_fringe_max_ratio=float(fringe_max_ratio),
        export_min_points_per_object=int(min_points_per_object),
        export_score_mode="observations",
    )
    diag = exporter.export_object_slot_posterior_support(object_dict, bank)
    expected_by_object_id = {idx: int(label) for idx, (label, _) in enumerate(groups)}
    point_groups = _load_exported_point_groups(stream, output_config, expected_by_object_id)
    return point_groups, diag


def _export_densify_groups(
    *,
    stream: ScanNetStream,
    groups: list[dict[str, Any]],
    output_config: str,
    min_points_per_object: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    exporter = ScanNetExporter(
        stream,
        output_config=output_config,
        export_support_mode="reliable_densify",
        export_min_points_per_object=int(min_points_per_object),
        export_score_mode="selection_quality",
        export_max_masks_per_object=int(args.m3_max_masks_per_object),
        export_mask_min_relative_coverage=float(args.m3_mask_min_relative_coverage),
        export_mask_sample_stride=int(args.m3_mask_sample_stride),
        export_mask_max_pixels=int(args.m3_mask_max_pixels),
        export_nn_radius=float(args.m3_nn_radius),
        densify_boundary_erosion=int(args.m3_boundary_erosion),
        densify_small_mask_area=int(args.m3_small_mask_area),
        densify_seed_distance_px=float(args.m3_seed_distance_px),
        densify_min_seed_pixels=int(args.m3_min_seed_pixels),
        densify_seed_keep_mode=str(args.m3_seed_keep_mode),
        densify_seed_min_support_views=int(args.m3_seed_min_support_views),
        densify_mask_selection_mode=str(args.m3_mask_selection_mode),
        densify_enable_wta=bool(args.m3_enable_wta),
    )
    return exporter.export_object_dict_reliable_densify(_object_dict_from_point_groups(groups))


def _load_exported_point_groups(
    stream: ScanNetStream,
    output_config: str,
    expected_by_object_id: dict[int, int] | None = None,
) -> list[dict[str, Any]]:
    path = stream.object_dir / output_config / "object_dict.npy"
    if not path.exists():
        return []
    loaded = np.load(path, allow_pickle=True).item()
    groups: list[dict[str, Any]] = []
    for object_id, value in sorted(loaded.items(), key=lambda item: int(item[0])):
        expected = int(value.get("expected_gt_label", -1))
        if expected_by_object_id is not None:
            expected = int(expected_by_object_id.get(int(object_id), expected))
        groups.append(
            {
                "object_id": int(object_id),
                "point_ids": set(int(v) for v in np.asarray(value.get("point_ids", []), dtype=np.int64).tolist()),
                "carrier_ids": np.asarray(value.get("carrier_ids", []), dtype=np.int64),
                "mask_list": list(value.get("mask_list", [])),
                "expected_gt_label": expected,
            }
        )
    return groups


def _groups_for_variant(
    *,
    variant: str,
    bank: MeasurementBank,
    stream: ScanNetStream,
    scene_points: np.ndarray,
    scene_tree: Any,
    hits: dict[str, Any],
    surfel_groups: list[tuple[int, np.ndarray]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    point_mode = "best" if variant == "M0" else "union"
    groups: list[dict[str, Any]] = []
    seed_points = 0
    after_points = 0
    for label, surfels in surfel_groups:
        points = _points_for_surfels(hits, surfels, mode=point_mode)
        seed_points += len(points)
        if variant == "M2":
            points = _dilate_points(scene_points, scene_tree, points, float(args.m2_dilation_radius))
        after_points += len(points)
        groups.append(
            {
                "expected_gt_label": int(label),
                "carrier_ids": surfels,
                "point_ids": points,
                "mask_list": _mask_votes(bank, surfels, max_votes=8),
            }
        )
    diag: dict[str, Any] = {
        "materialization_seed_points": float(seed_points),
        "materialization_after_points": float(after_points),
        "materialization_expansion_ratio": float(after_points / max(seed_points, 1)),
    }
    if variant in {"M0", "M1", "M2"}:
        return groups, diag
    if variant == "M3":
        tmp_config = f"{args.output_config_prefix}_{variant.lower()}_tmp_densify_{bank.scene}"
        export_diag = _export_densify_groups(
            stream=stream,
            groups=groups,
            output_config=tmp_config,
            min_points_per_object=int(args.min_points_per_object),
            args=args,
        )
        expected_by_object_id = {idx: int(group.get("expected_gt_label", -1)) for idx, group in enumerate(groups)}
        loaded = _load_exported_point_groups(stream, tmp_config, expected_by_object_id)
        diag.update(export_diag)
        return loaded, diag
    raise ValueError(f"Unsupported materialization variant: {variant}")


def _regroup_points_by_gt(point_groups: list[dict[str, Any]], gt_ids: np.ndarray) -> list[dict[str, Any]]:
    by_label: dict[int, set[int]] = defaultdict(set)
    for group in point_groups:
        for point_id in group.get("point_ids", set()):
            pid = int(point_id)
            if pid < 0 or pid >= gt_ids.shape[0]:
                continue
            label = int(gt_ids[pid])
            if label >= 1000:
                by_label[label].add(pid)
    out: list[dict[str, Any]] = []
    for label in sorted(by_label):
        out.append({"expected_gt_label": int(label), "point_ids": by_label[label], "mask_list": [], "carrier_ids": []})
    return out


def _point_group_purity(groups: list[dict[str, Any]], gt_ids: np.ndarray) -> dict[str, float]:
    total = 0
    correct = 0
    gt_labeled = 0
    contaminating = 0
    for group in groups:
        expected = int(group.get("expected_gt_label", -1))
        points = [int(v) for v in group.get("point_ids", set()) if 0 <= int(v) < gt_ids.shape[0]]
        total += len(points)
        if expected < 1000:
            continue
        labels = gt_ids[np.asarray(points, dtype=np.int64)] if points else np.empty((0,), dtype=np.int64)
        gt_labeled += int(np.count_nonzero(labels >= 1000))
        correct += int(np.count_nonzero(labels == expected))
        contaminating += int(np.count_nonzero((labels >= 1000) & (labels != expected)))
    return {
        "purity": float(correct / max(gt_labeled, 1)),
        "contamination_ratio": float(contaminating / max(gt_labeled, 1)),
        "non_gt_point_ratio": float((total - gt_labeled) / max(total, 1)),
    }


def _parse_metric(path: Path) -> dict[str, float | None]:
    if not path.exists():
        return {"ap": None, "ap50": None, "ap25": None}
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {"ap": None, "ap50": None, "ap25": None}
    parts = lines[-1].replace("\t", ",").split(",")
    if len(parts) < 3:
        return {"ap": None, "ap50": None, "ap25": None}
    return {"ap": float(parts[0]), "ap50": float(parts[1]), "ap25": float(parts[2])}


def _write_manifest(output_config: str, args: argparse.Namespace, *, oracle: str, variant: str) -> None:
    manifest = build_prediction_manifest(
        root=".",
        output_config=output_config,
        is_method_result=False,
        is_diagnostic_only=True,
        uses_gt=True,
        gt_usage=f"v19 {variant} {oracle} GT-only diagnostic oracle",
        source_configs=[args.bank_root, args.graph_root],
        pre_points_policy="recompute",
        support_policy=f"v19_{variant}_{oracle}_materialization_diagnostic",
        notes="Diagnostic-only GT oracle. Forbidden for method tables.",
        extra={
            "algorithm": "v19_4d_tubecover_materialization_diagnostic",
            "materialization_variant": variant,
            "oracle": oracle,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic": True,
            "alignment_source": "gt_or_rgbd_eval_only",
            "is_method_result": False,
            "is_diagnostic_only": True,
            "forbidden_for_method_table": True,
            "gt_selected_output": True,
            "eval_policy": "oracle_diagnostic_own_recompute",
        },
    )
    write_prediction_manifest(output_config, manifest, root=".", pred_suffix="class_agnostic")


def _run_eval(output_config: str, args: argparse.Namespace) -> dict[str, float | None]:
    metric_path = Path("data/evaluation/scannet") / f"{output_config}_class_agnostic.txt"
    if not args.skip_eval:
        cmd = [
            sys.executable,
            "-m",
            "evaluation.evaluate",
            "--pred_path",
            f"data/prediction/{output_config}_class_agnostic",
            "--gt_path",
            args.gt_root,
            "--dataset",
            "scannet",
            "--output_file",
            str(metric_path),
            "--tmp_root",
            "data/TMP",
            "--tmp_config",
            output_config,
            "--no_class",
            "--require-manifest",
            "--allow-oracle-eval",
        ]
        subprocess.run(cmd, check=True)
    return _parse_metric(metric_path)


def _save_hits(scene_dir: Path, hits: dict[str, Any]) -> None:
    scene_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        scene_dir / "surfel_mesh_hits.npz",
        labels=np.asarray(hits["labels"], dtype=np.int64),
        best_vertex=np.asarray(hits["best_vertex"], dtype=np.int64),
        hit_count=np.asarray(hits["hit_count"], dtype=np.int32),
        hit_frame_count=np.asarray(hits["hit_frame_count"], dtype=np.int16),
        median_nn_distance=np.asarray(hits["median_nn_distance"], dtype=np.float32),
        vote_entropy=np.asarray(hits["vote_entropy"], dtype=np.float32),
        gt_vote_count=np.asarray(hits["gt_vote_count"], dtype=np.int16),
        hit_surfel_index=np.asarray(hits["hit_surfel_index"], dtype=np.int64),
        hit_vertex_index=np.asarray(hits["hit_vertex_index"], dtype=np.int64),
        hit_frame_index=np.asarray(hits["hit_frame_index"], dtype=np.int16),
        hit_distance=np.asarray(hits["hit_distance"], dtype=np.float32),
    )


def _component_stats_for_gt(graph: SignedSurfelGraph, labels: np.ndarray, gt_label: int) -> tuple[int, float]:
    nodes = np.flatnonzero(labels == int(gt_label))
    if nodes.size == 0:
        return 0, 0.0
    node_set = set(int(v) for v in nodes.tolist())
    adj: dict[int, list[int]] = {int(v): [] for v in nodes.tolist()}
    for a, b, keep in zip(graph.src.tolist(), graph.dst.tolist(), np.asarray(graph.precut_keep, dtype=bool).tolist()):
        if not keep:
            continue
        if int(a) in node_set and int(b) in node_set:
            adj[int(a)].append(int(b))
            adj[int(b)].append(int(a))
    seen: set[int] = set()
    sizes: list[int] = []
    for start in nodes.tolist():
        start = int(start)
        if start in seen:
            continue
        seen.add(start)
        q: deque[int] = deque([start])
        size = 0
        while q:
            cur = q.popleft()
            size += 1
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        sizes.append(size)
    return len(sizes), float(max(sizes) / max(nodes.size, 1)) if sizes else 0.0


def classify_failure_type(
    *,
    gt_point_count: int,
    num_surfels: int,
    mesh_recall: float,
    num_components: int,
    largest_component_ratio: float,
    best_component_recall: float,
    best_component_precision: float,
    exported_point_count: int,
) -> str:
    if num_surfels <= 0 and mesh_recall <= 0.005:
        return "no_surfel_coverage"
    if num_surfels <= 0:
        return "label_missing"
    if num_components >= 4 and largest_component_ratio < 0.50:
        return "fragmented"
    if best_component_precision < 0.60 and best_component_recall >= 0.20:
        return "overmerged"
    if exported_point_count < max(20, int(0.02 * gt_point_count)) and num_surfels >= 5:
        return "export_lost"
    if mesh_recall < 0.25 or best_component_recall < 0.25:
        return "underfilled"
    return "underfilled"


def _failure_decomposition_rows(
    *,
    scene: str,
    graph: SignedSurfelGraph,
    labels: np.ndarray,
    hits: dict[str, Any],
    gt_ids: np.ndarray,
    oracle_b_groups: list[tuple[int, np.ndarray]],
    oracle_c_groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    hit_vertices = np.asarray(hits["hit_vertex_index"], dtype=np.int64)
    mesh_hits_by_gt: Counter[int] = Counter(int(gt_ids[v]) for v in hit_vertices.tolist() if int(gt_ids[v]) >= 1000)
    surfel_counts: Counter[int] = Counter(int(v) for v in labels[labels >= 1000].tolist())
    exported_by_gt: Counter[int] = Counter()
    for group in oracle_c_groups:
        expected = int(group.get("expected_gt_label", -1))
        if expected >= 1000:
            exported_by_gt[expected] += len(group.get("point_ids", set()))

    component_best: dict[int, tuple[float, float, float]] = defaultdict(lambda: (0.0, 0.0, 0.0))
    for expected, surfels in oracle_b_groups:
        if expected < 1000:
            continue
        comp_labels = labels[surfels]
        comp_size = int(surfels.shape[0])
        label_count = int(np.count_nonzero(comp_labels == int(expected)))
        gt_total = int(surfel_counts.get(int(expected), 0))
        precision = float(label_count / max(comp_size, 1))
        recall = float(label_count / max(gt_total, 1))
        union = gt_total + comp_size - label_count
        iou = float(label_count / max(union, 1))
        if iou > component_best[int(expected)][0]:
            component_best[int(expected)] = (iou, precision, recall)

    rows: list[dict[str, Any]] = []
    gt_instances = [int(v) for v in np.unique(gt_ids[gt_ids >= 1000]).tolist()]
    edge_a = labels[graph.src]
    edge_b = labels[graph.dst]
    for gt_label in gt_instances:
        gt_point_count = int(np.count_nonzero(gt_ids == int(gt_label)))
        num_surfels = int(surfel_counts.get(gt_label, 0))
        incident = (edge_a == int(gt_label)) | (edge_b == int(gt_label))
        known_incident = incident & (edge_a >= 1000) & (edge_b >= 1000)
        inside = known_incident & (edge_a == int(gt_label)) & (edge_b == int(gt_label))
        cross = known_incident & (edge_a != edge_b)
        num_components, largest_ratio = _component_stats_for_gt(graph, labels, gt_label)
        best_iou, best_precision, best_recall = component_best[gt_label]
        mesh_recall = float(mesh_hits_by_gt.get(gt_label, 0) / max(gt_point_count, 1))
        exported_count = int(exported_by_gt.get(gt_label, 0))
        failure_type = classify_failure_type(
            gt_point_count=gt_point_count,
            num_surfels=num_surfels,
            mesh_recall=mesh_recall,
            num_components=num_components,
            largest_component_ratio=largest_ratio,
            best_component_recall=best_recall,
            best_component_precision=best_precision,
            exported_point_count=exported_count,
        )
        rows.append(
            {
                "scene": scene,
                "gt_instance_id": int(gt_label),
                "gt_point_count": gt_point_count,
                "num_surfels_projected_to_gt": num_surfels,
                "surfel_node_coverage_ratio": float(num_surfels / max(labels.shape[0], 1)),
                "num_edges_inside_gt": int(np.count_nonzero(inside)),
                "num_edges_cross_gt_boundary": int(np.count_nonzero(cross)),
                "num_labeled_edges": int(np.count_nonzero(known_incident)),
                "edge_coverage_ratio": float(np.count_nonzero(known_incident) / max(np.count_nonzero(incident), 1)),
                "num_connected_components_inside_gt": int(num_components),
                "largest_component_ratio_inside_gt": float(largest_ratio),
                "best_oracle_component_iou": float(best_iou),
                "best_oracle_component_precision": float(best_precision),
                "best_oracle_component_recall": float(best_recall),
                "mesh_hit_recall": float(mesh_recall),
                "oracle_c_exported_points_in_gt": int(exported_count),
                "failure_type": failure_type,
            }
        )
    return rows


def _scene_oracle_exports(
    *,
    args: argparse.Namespace,
    variant: str,
    scene: str,
    bank: MeasurementBank,
    graph: SignedSurfelGraph,
    stream: ScanNetStream,
    scene_points: np.ndarray,
    scene_tree: Any,
    gt_ids: np.ndarray,
    hits: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = np.asarray(hits["labels"], dtype=np.int64)
    edge_labels = _edge_labels(graph, labels)
    label_groups = _label_groups(labels, min_surfels=int(args.oracle_a_min_surfels))
    edge_groups = _edge_oracle_components(
        graph,
        labels,
        min_surfels=int(args.oracle_b_min_surfels),
        use_precut=bool(args.oracle_b_use_precut),
    )
    oracle_configs = {
        "oracle_a": f"{args.output_config_prefix}_{variant.lower()}_oracle_a_probe5",
        "oracle_b": f"{args.output_config_prefix}_{variant.lower()}_oracle_b_probe5",
        "oracle_c": f"{args.output_config_prefix}_{variant.lower()}_oracle_c_probe5",
    }

    export_diags: dict[str, Any] = {}
    if variant == "M0":
        a_point_groups, export_diags["oracle_a"] = _export_posterior_groups(
            stream=stream,
            bank=bank,
            groups=label_groups,
            output_config=oracle_configs["oracle_a"],
            min_points_per_object=int(args.min_points_per_object),
            core_nn_radius=float(args.m0_core_nn_radius),
            fringe_nn_radius=float(args.m0_fringe_nn_radius),
            fringe_radius=float(args.m0_fringe_radius),
            fringe_max_ratio=float(args.m0_fringe_max_ratio),
        )
        b_point_groups, export_diags["oracle_b"] = _export_posterior_groups(
            stream=stream,
            bank=bank,
            groups=edge_groups,
            output_config=oracle_configs["oracle_b"],
            min_points_per_object=int(args.min_points_per_object),
            core_nn_radius=float(args.m0_core_nn_radius),
            fringe_nn_radius=float(args.m0_fringe_nn_radius),
            fringe_radius=float(args.m0_fringe_radius),
            fringe_max_ratio=float(args.m0_fringe_max_ratio),
        )
    else:
        a_point_groups, a_diag = _groups_for_variant(
            variant=variant,
            bank=bank,
            stream=stream,
            scene_points=scene_points,
            scene_tree=scene_tree,
            hits=hits,
            surfel_groups=label_groups,
            args=args,
        )
        b_point_groups, b_diag = _groups_for_variant(
            variant=variant,
            bank=bank,
            stream=stream,
            scene_points=scene_points,
            scene_tree=scene_tree,
            hits=hits,
            surfel_groups=edge_groups,
            args=args,
        )
        export_diags["oracle_a"] = {
            **a_diag,
            **_export_point_groups(
                stream=stream,
                groups=a_point_groups,
                output_config=oracle_configs["oracle_a"],
                min_points_per_object=int(args.min_points_per_object),
            ),
        }
        export_diags["oracle_b"] = {
            **b_diag,
            **_export_point_groups(
                stream=stream,
                groups=b_point_groups,
                output_config=oracle_configs["oracle_b"],
                min_points_per_object=int(args.min_points_per_object),
            ),
        }

    c_seed_groups = _regroup_points_by_gt(a_point_groups, gt_ids)
    export_diags["oracle_c"] = _export_point_groups(
        stream=stream,
        groups=c_seed_groups,
        output_config=oracle_configs["oracle_c"],
        min_points_per_object=int(args.min_points_per_object),
    )
    materialized_points: set[int] = set()
    for group in c_seed_groups:
        materialized_points.update(int(v) for v in group.get("point_ids", set()))
    materialized_point_ids = np.asarray(sorted(materialized_points), dtype=np.int64)
    gt_instances = np.unique(gt_ids[gt_ids >= 1000])
    if materialized_point_ids.size:
        materialized_gt_instances = np.unique(gt_ids[materialized_point_ids][gt_ids[materialized_point_ids] >= 1000])
    else:
        materialized_gt_instances = np.empty((0,), dtype=np.int64)

    rows = []
    for oracle_name, groups in (
        ("oracle_a", a_point_groups),
        ("oracle_b", b_point_groups),
        ("oracle_c", c_seed_groups),
    ):
        purity = _point_group_purity(groups, gt_ids)
        rows.append(
            {
                "scene": scene,
                "variant": variant,
                "oracle": oracle_name,
                "output_config": oracle_configs[oracle_name],
                "num_groups_before_export": int(len(groups)),
                **purity,
                **{f"{oracle_name}_{k}": v for k, v in export_diags.get(oracle_name, {}).items()},
            }
        )

    scene_row = {
        "scene": scene,
        "variant": variant,
        "num_surfels": int(bank.num_surfels),
        "num_edges": int(graph.num_edges),
        "node_gt_label_coverage": float(hits["node_gt_label_coverage"]),
        "edge_gt_label_coverage": float(edge_labels["edge_gt_label_coverage"]),
        "surfel_hit_rate": float(hits["surfel_hit_rate"]),
        "mean_hit_frames_per_surfel": float(hits["mean_hit_frames_per_surfel"]),
        "raw_mesh_hit_vertex_coverage_ratio": float(hits["mesh_vertex_coverage_ratio"]),
        "raw_covered_gt_instance_count": int(hits["covered_gt_instance_count"]),
        "raw_covered_gt_instance_ratio": float(hits["covered_gt_instance_ratio"]),
        "mesh_vertex_coverage_ratio": float(materialized_point_ids.shape[0] / max(gt_ids.shape[0], 1)),
        "covered_gt_instance_count": int(materialized_gt_instances.shape[0]),
        "covered_gt_instance_ratio": float(materialized_gt_instances.shape[0] / max(gt_instances.shape[0], 1)),
        "backproject_hit_rate": float(hits["backproject_hit_rate"]),
        "same_gt_edges": int(edge_labels["same_gt_edges"]),
        "cut_gt_edges": int(edge_labels["cut_gt_edges"]),
        "unknown_gt_edges": int(edge_labels["unknown_gt_edges"]),
        "num_oracle_a_groups": int(len(label_groups)),
        "num_oracle_b_components": int(len(edge_groups)),
        "oracle_b_component_size_mean": float(np.mean([arr.shape[0] for _, arr in edge_groups])) if edge_groups else 0.0,
        "oracle_b_component_size_p90": float(np.percentile([arr.shape[0] for _, arr in edge_groups], 90)) if edge_groups else 0.0,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic": True,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
    }
    failure_rows = _failure_decomposition_rows(
        scene=scene,
        graph=graph,
        labels=labels,
        hits=hits,
        gt_ids=gt_ids,
        oracle_b_groups=edge_groups,
        oracle_c_groups=c_seed_groups,
    )
    return {"scene_row": scene_row, "oracle_rows": rows, "failure_rows": failure_rows}, rows


def _numeric_mean(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float, np.generic)) and value is not None and not isinstance(value, bool)
        }
    )
    return {
        key: float(np.mean([float(row[key]) for row in rows if row.get(key) is not None]))
        for key in keys
        if any(row.get(key) is not None for row in rows)
    }


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys() if not isinstance(row.get(key), (dict, list, set))})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_markdown(prefix: Path, payload: dict[str, Any]) -> None:
    aggregate = payload["aggregate"]
    lines = [
        f"# Stream4D v19 Materialization Diagnostic: {aggregate['variant']}",
        "",
        "## Gates",
        "",
    ]
    for key, value in aggregate.get("gates", {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Aggregate Coverage", ""])
    for key, value in aggregate.get("numeric_mean", {}).items():
        if key in {
            "node_gt_label_coverage",
            "edge_gt_label_coverage",
            "surfel_hit_rate",
            "mean_hit_frames_per_surfel",
            "mesh_vertex_coverage_ratio",
            "covered_gt_instance_ratio",
            "backproject_hit_rate",
        }:
            lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Oracle Metrics", ""])
    lines.append("| oracle | AP | AP50 | AP25 | gate |")
    lines.append("|---|---:|---:|---:|---|")
    for oracle, metrics in aggregate.get("oracle_metrics", {}).items():
        lines.append(
            "| "
            + " | ".join(
                [
                    oracle,
                    str(metrics.get("ap")),
                    str(metrics.get("ap50")),
                    str(metrics.get("ap25")),
                    str(metrics.get("gt_oracle_gate")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Failure Types", ""])
    failure_counts = aggregate.get("failure_type_counts", {})
    for key, value in sorted(failure_counts.items()):
        lines.append(f"- {key}: `{value}`")
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["M0", "M1", "M2", "M3"], required=True)
    parser.add_argument("--bank-root", default="outputs/v14_measurement_bank_bank16_cropformer")
    parser.add_argument("--graph-root", default="outputs/audit/v18_phase1_repair_precut_k16_d015")
    parser.add_argument("--seq-list", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--output-config-prefix", default="stream4d_v19")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--scannet-root", default="data/scannet/processed")
    parser.add_argument("--gt-root", default="data/scannet/gt")
    parser.add_argument("--nn-radius", type=float, default=0.05)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--oracle-a-min-surfels", type=int, default=1)
    parser.add_argument("--oracle-b-min-surfels", type=int, default=20)
    parser.add_argument("--oracle-b-use-precut", action="store_true")
    parser.add_argument("--min-points-per-object", type=int, default=20)
    parser.add_argument("--m0-core-nn-radius", type=float, default=0.05)
    parser.add_argument("--m0-fringe-nn-radius", type=float, default=0.05)
    parser.add_argument("--m0-fringe-radius", type=float, default=0.0)
    parser.add_argument("--m0-fringe-max-ratio", type=float, default=0.35)
    parser.add_argument("--m2-dilation-radius", type=float, default=0.03)
    parser.add_argument("--m3-nn-radius", type=float, default=0.08)
    parser.add_argument("--m3-max-masks-per-object", type=int, default=8)
    parser.add_argument("--m3-mask-min-relative-coverage", type=float, default=0.0)
    parser.add_argument("--m3-mask-sample-stride", type=int, default=1)
    parser.add_argument("--m3-mask-max-pixels", type=int, default=50000)
    parser.add_argument("--m3-boundary-erosion", type=int, default=1)
    parser.add_argument("--m3-small-mask-area", type=int, default=400)
    parser.add_argument("--m3-seed-distance-px", type=float, default=32.0)
    parser.add_argument("--m3-min-seed-pixels", type=int, default=1)
    parser.add_argument("--m3-seed-keep-mode", default="all")
    parser.add_argument("--m3-seed-min-support-views", type=int, default=1)
    parser.add_argument("--m3-mask-selection-mode", default="coverage_component_density")
    parser.add_argument("--m3-enable-wta", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    prefix = Path(args.output_prefix)
    scene_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    scenes = read_seq_list(Path(args.seq_list))
    for scene in scenes:
        bank = MeasurementBank.load(Path(args.bank_root) / scene / "measurement_bank.npz")
        graph = SignedSurfelGraph.load(Path(args.graph_root) / scene / "signed_surfel_graph.npz")
        stream = ScanNetStream(seq_name=scene, backbone=args.backbone, root=args.scannet_root)
        scene_points = _load_scene_points(stream)
        scene_tree = cKDTree(scene_points)
        gt_ids = np.loadtxt(Path(args.gt_root) / f"{scene}.txt", dtype=np.int64)
        if gt_ids.shape[0] != scene_points.shape[0]:
            raise RuntimeError(f"{scene}: GT/mesh vertex count mismatch: {gt_ids.shape[0]} vs {scene_points.shape[0]}")
        hits = _surfel_mesh_hits(
            bank,
            stream,
            scene_points,
            scene_tree,
            gt_ids,
            nn_radius=float(args.nn_radius),
            max_frames=int(args.max_frames),
        )
        scene_dir = prefix.parent / scene
        _save_hits(scene_dir, hits)
        scene_payload, _ = _scene_oracle_exports(
            args=args,
            variant=args.variant,
            scene=scene,
            bank=bank,
            graph=graph,
            stream=stream,
            scene_points=scene_points,
            scene_tree=scene_tree,
            gt_ids=gt_ids,
            hits=hits,
        )
        scene_rows.append(scene_payload["scene_row"])
        oracle_rows.extend(scene_payload["oracle_rows"])
        failure_rows.extend(scene_payload["failure_rows"])
        (scene_dir / "failure_decomposition.json").write_text(
            json.dumps(json_safe(scene_payload["failure_rows"]), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    oracle_metrics: dict[str, dict[str, Any]] = {}
    for oracle in ("oracle_a", "oracle_b", "oracle_c"):
        output_config = f"{args.output_config_prefix}_{args.variant.lower()}_{oracle}_probe5"
        _write_manifest(output_config, args, oracle=oracle, variant=args.variant)
        metrics = _run_eval(output_config, args)
        metrics["gt_oracle_gate"] = bool(
            (metrics.get("ap") or 0.0) >= 0.45
            and (metrics.get("ap50") or 0.0) >= 0.60
            and (metrics.get("ap25") or 0.0) >= 0.72
        )
        oracle_metrics[oracle] = metrics

    numeric = _numeric_mean(scene_rows)
    failure_type_counts = Counter(str(row["failure_type"]) for row in failure_rows)
    gates = {
        "coverage_materialization_gate": bool(
            numeric.get("node_gt_label_coverage", 0.0) >= 0.70
            and numeric.get("edge_gt_label_coverage", 0.0) >= 0.60
            and numeric.get("covered_gt_instance_ratio", 0.0) >= 0.65
            and numeric.get("mesh_vertex_coverage_ratio", 0.0) >= 0.25
        ),
        "oracle_c_min_gate": bool(
            (oracle_metrics.get("oracle_c", {}).get("ap50") or 0.0) >= 0.60
            and (oracle_metrics.get("oracle_c", {}).get("ap25") or 0.0) >= 0.72
            and numeric.get("mesh_vertex_coverage_ratio", 0.0) >= 0.25
            and numeric.get("covered_gt_instance_ratio", 0.0) >= 0.65
        ),
        "phase2a_success_gate": False,
    }
    gates["phase2a_success_gate"] = bool(gates["coverage_materialization_gate"] and gates["oracle_c_min_gate"])
    aggregate = {
        "phase": "v19_phase1_phase2a_materialization",
        "variant": args.variant,
        "num_scenes": int(len(scene_rows)),
        "numeric_mean": numeric,
        "oracle_metrics": oracle_metrics,
        "failure_type_counts": dict(failure_type_counts),
        "gates": gates,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic": True,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
    }
    payload = {
        "args": vars(args),
        "aggregate": aggregate,
        "scene_rows": scene_rows,
        "oracle_rows": oracle_rows,
        "failure_rows": failure_rows,
    }
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    _write_table(prefix.with_name(prefix.name + "_scene_rows.csv"), scene_rows)
    _write_table(prefix.with_name(prefix.name + "_oracle_rows.csv"), oracle_rows)
    _write_table(prefix.with_name(prefix.name + "_failure_rows.csv"), failure_rows)
    _write_markdown(prefix, payload)
    print(json.dumps(json_safe(aggregate), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
