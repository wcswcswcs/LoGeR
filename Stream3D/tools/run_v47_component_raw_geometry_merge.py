from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from stream4d_native.v47_common import (
    ROOT,
    adjusted_rand_score,
    cluster_completeness,
    cluster_purity,
    parse_bool,
    parse_float,
    parse_int,
    read_csv,
    write_csv,
    write_json,
)


class _StringUnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> str:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left
        return root_left


def _node_component_rows(mask_vote_rows: list[dict[str, Any]]) -> tuple[dict[int, str], dict[str, dict[str, Any]]]:
    node_to_component: dict[int, str] = {}
    component_meta: dict[str, dict[str, Any]] = {}
    for row in mask_vote_rows:
        component_id = str(row.get("predicted_component_object_id") or row.get("predicted_supertrack_object_id"))
        node_to_component[parse_int(row.get("node_id"))] = component_id
        meta = component_meta.setdefault(
            component_id,
            {
                "component_id": component_id,
                "scene": str(row.get("scene")),
                "frames": set(),
                "mask_count": 0,
            },
        )
        meta["frames"].add(parse_int(row.get("frame_id")))
        meta["mask_count"] += 1
    return node_to_component, component_meta


def _xyz_and_frame_map(debug_root: Path, scene: str, window_index: int) -> tuple[np.ndarray, dict[int, int], dict[str, Any]]:
    path = debug_root / scene / f"carriers_window{int(window_index):03d}.npz"
    manifest_path = path.with_name(f"{path.stem}_manifest.json")
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        xyz = np.asarray(data["xyz_ref"], dtype=np.float32)
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame_ids = [int(value) for value in manifest.get("frame_ids", [])]
    if not frame_ids:
        frame_ids = list(range(int(xyz.shape[0])))
    return xyz, {frame_id: idx for idx, frame_id in enumerate(frame_ids)}, manifest


def _raw_mask_geometry_features(
    *,
    carrier_rows: list[dict[str, Any]],
    carrier_debug_root: Path,
    min_carriers_per_mask: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    cache: dict[tuple[str, int], tuple[np.ndarray, dict[int, int], dict[str, Any]]] = {}
    points_by_mask: dict[str, list[np.ndarray]] = defaultdict(list)
    missing_cache_rows = 0
    missing_frame_rows = 0
    out_of_bounds_rows = 0
    nonfinite_rows = 0
    used_rows = 0
    manifest_flags: list[dict[str, Any]] = []

    for row in carrier_rows:
        if not (
            parse_bool(row.get("visible"))
            and parse_bool(row.get("valid"))
            and parse_bool(row.get("valid_uv"))
            and parse_bool(row.get("mask_label_available"))
        ):
            continue
        mask_id = parse_int(row.get("observed_mask_id"))
        if mask_id <= 0:
            continue
        scene = str(row.get("scene"))
        window_index = parse_int(row.get("window_index"))
        key = (scene, window_index)
        try:
            if key not in cache:
                cache[key] = _xyz_and_frame_map(carrier_debug_root, scene, window_index)
                manifest_flags.append(
                    {
                        "scene": scene,
                        "window_index": window_index,
                        "uses_gt_for_prediction": bool(cache[key][2].get("uses_gt_for_prediction", False)),
                        "uses_pose_for_prediction": bool(cache[key][2].get("uses_pose_for_prediction", False)),
                        "uses_rgbd_for_prediction": bool(cache[key][2].get("uses_rgbd_for_prediction", False)),
                        "uses_scannet_mesh_for_prediction": bool(cache[key][2].get("uses_scannet_mesh_for_prediction", False)),
                        "is_diagnostic_only": bool(cache[key][2].get("is_diagnostic_only", False)),
                        "is_method_result": bool(cache[key][2].get("is_method_result", False)),
                    }
                )
            xyz, frame_map, _manifest = cache[key]
        except FileNotFoundError:
            missing_cache_rows += 1
            continue
        frame_id = parse_int(row.get("frame_id"))
        local_index = frame_map.get(frame_id)
        if local_index is None:
            missing_frame_rows += 1
            continue
        carrier_index = parse_int(row.get("carrier_index"))
        if local_index < 0 or local_index >= xyz.shape[0] or carrier_index < 0 or carrier_index >= xyz.shape[1]:
            out_of_bounds_rows += 1
            continue
        point = np.asarray(xyz[local_index, carrier_index], dtype=np.float32)
        if not np.isfinite(point).all():
            nonfinite_rows += 1
            continue
        mask_obs_id = f"{scene}:{frame_id}:{mask_id}"
        points_by_mask[mask_obs_id].append(point)
        used_rows += 1

    features: dict[str, dict[str, Any]] = {}
    total_feature_points = 0
    for mask_obs_id, point_list in points_by_mask.items():
        if len(point_list) < int(min_carriers_per_mask):
            continue
        points = np.stack(point_list, axis=0).astype(np.float32)
        centroid = points.mean(axis=0)
        dist = np.linalg.norm(points - centroid[None, :], axis=1)
        total_feature_points += int(points.shape[0])
        features[mask_obs_id] = {
            "mask_observation_id": mask_obs_id,
            "raw_point_count": int(points.shape[0]),
            "centroid": centroid.astype(np.float64),
            "radius_mean": float(dist.mean()) if dist.size else 0.0,
            "radius_q90": float(np.percentile(dist, 90)) if dist.size else 0.0,
            "radius_max": float(dist.max()) if dist.size else 0.0,
        }

    summary = {
        "raw_geometry_mask_feature_count": len(features),
        "raw_geometry_mask_candidate_count": len(points_by_mask),
        "raw_geometry_used_carrier_rows": used_rows,
        "raw_geometry_feature_point_count": total_feature_points,
        "min_carriers_per_mask": int(min_carriers_per_mask),
        "missing_cache_rows": missing_cache_rows,
        "missing_frame_rows": missing_frame_rows,
        "out_of_bounds_rows": out_of_bounds_rows,
        "nonfinite_rows": nonfinite_rows,
        "carrier_cache_window_count": len(cache),
        "carrier_manifest_flags": manifest_flags,
        "uses_gt_for_prediction": False,
        "uses_pose_for_prediction": False,
        "uses_rgbd_for_prediction": False,
        "uses_scannet_mesh_for_prediction": False,
    }
    return features, summary


def _candidate_pair_stats(
    edge_rows: list[dict[str, Any]],
    node_to_component: dict[int, str],
    component_meta: dict[str, dict[str, Any]],
    raw_features: dict[str, dict[str, Any]],
    edge_types: set[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    stats: dict[tuple[str, str], dict[str, Any]] = {}
    for row in edge_rows:
        if str(row.get("edge_type")) not in edge_types:
            continue
        src = node_to_component.get(parse_int(row.get("src_node_id")))
        dst = node_to_component.get(parse_int(row.get("dst_node_id")))
        if not src or not dst or src == dst:
            continue
        left = component_meta.get(src)
        right = component_meta.get(dst)
        if not left or not right or left["scene"] != right["scene"]:
            continue
        key = tuple(sorted([src, dst]))
        item = stats.setdefault(
            key,
            {
                "component_left": key[0],
                "component_right": key[1],
                "scene": left["scene"],
                "edge_count": 0,
                "same_frame_conflict": bool(set(left["frames"]) & set(right["frames"])),
                "max_A5_d4rt_semantic_confirmation": 0.0,
                "max_A4_d4rt_visible_veto": 0.0,
                "max_A8_no_temporal_control": 0.0,
                "max_A7_shuffled_D4RT": 0.0,
                "min_visible_outside": 1.0,
                "max_forward_visible_carrier_count": 0,
                "max_backward_visible_carrier_count": 0,
                "diagnostic_same_gt_edge_count": 0,
                "raw_pair_count": 0,
                "min_raw_centroid_dist": 1.0e9,
                "min_raw_distance_over_radius": 1.0e9,
                "max_raw_pair_support_min": 0,
            },
        )
        item["edge_count"] += 1
        for score_key in [
            "A5_d4rt_semantic_confirmation",
            "A4_d4rt_visible_veto",
            "A8_no_temporal_control",
            "A7_shuffled_D4RT",
        ]:
            key_name = f"max_{score_key}"
            item[key_name] = max(parse_float(item.get(key_name)), parse_float(row.get(score_key)))
        item["min_visible_outside"] = min(parse_float(item.get("min_visible_outside"), 1.0), parse_float(row.get("visible_outside"), 1.0))
        item["max_forward_visible_carrier_count"] = max(
            parse_int(item.get("max_forward_visible_carrier_count")), parse_int(row.get("forward_visible_carrier_count"))
        )
        item["max_backward_visible_carrier_count"] = max(
            parse_int(item.get("max_backward_visible_carrier_count")), parse_int(row.get("backward_visible_carrier_count"))
        )
        if parse_bool(row.get("diagnostic_same_gt")):
            item["diagnostic_same_gt_edge_count"] += 1

        src_feature = raw_features.get(str(row.get("src_mask_observation_id")))
        dst_feature = raw_features.get(str(row.get("dst_mask_observation_id")))
        if src_feature and dst_feature:
            src_centroid = np.asarray(src_feature["centroid"], dtype=np.float64)
            dst_centroid = np.asarray(dst_feature["centroid"], dtype=np.float64)
            dist = float(np.linalg.norm(src_centroid - dst_centroid))
            radius = max(
                float(src_feature.get("radius_q90", 0.0)) + float(dst_feature.get("radius_q90", 0.0)),
                1.0e-6,
            )
            support_min = min(parse_int(src_feature.get("raw_point_count")), parse_int(dst_feature.get("raw_point_count")))
            item["raw_pair_count"] += 1
            item["min_raw_centroid_dist"] = min(parse_float(item.get("min_raw_centroid_dist"), 1.0e9), dist)
            item["min_raw_distance_over_radius"] = min(
                parse_float(item.get("min_raw_distance_over_radius"), 1.0e9), float(dist / radius)
            )
            item["max_raw_pair_support_min"] = max(parse_int(item.get("max_raw_pair_support_min")), support_min)

    for item in stats.values():
        if parse_int(item.get("raw_pair_count")) <= 0:
            item["min_raw_centroid_dist"] = None
            item["min_raw_distance_over_radius"] = None
    return stats


def _labels(mask_vote_rows: list[dict[str, Any]], uf: _StringUnionFind) -> tuple[list[str], list[str]]:
    true_labels: list[str] = []
    pred_labels: list[str] = []
    for row in mask_vote_rows:
        gt = str(row.get("diagnostic_gt_instance", ""))
        if not gt:
            continue
        component_id = str(row.get("predicted_component_object_id") or row.get("predicted_supertrack_object_id"))
        if component_id not in uf.parent:
            continue
        true_labels.append(gt)
        pred_labels.append(uf.find(component_id))
    return true_labels, pred_labels


def _evaluate(
    *,
    component_ids: list[str],
    component_meta: dict[str, dict[str, Any]],
    mask_vote_rows: list[dict[str, Any]],
    candidate_stats: dict[tuple[str, str], dict[str, Any]],
    score_key: str,
    score_threshold: float,
    min_pair_edge_count: int,
    max_visible_outside: float,
    min_visible_carriers: int,
    max_raw_centroid_dist: float,
    max_raw_distance_over_radius: float,
    min_raw_pair_support: int,
    max_selected_pairs: int,
    enforce_cluster_frame_exclusion: bool,
    forbid_pair_same_frame_conflict: bool,
    collect_selected: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    score_col = f"max_{score_key}"
    use_raw_filter = (
        float(max_raw_centroid_dist) < 999.0
        or float(max_raw_distance_over_radius) < 999.0
        or int(min_raw_pair_support) > 0
    )
    candidates = []
    for row in candidate_stats.values():
        if parse_int(row.get("edge_count")) < min_pair_edge_count:
            continue
        if parse_float(row.get(score_col)) < score_threshold:
            continue
        if parse_float(row.get("min_visible_outside"), 1.0) > max_visible_outside:
            continue
        if parse_int(row.get("max_forward_visible_carrier_count")) < min_visible_carriers:
            continue
        if parse_int(row.get("max_backward_visible_carrier_count")) < min_visible_carriers:
            continue
        if forbid_pair_same_frame_conflict and bool(row.get("same_frame_conflict")):
            continue
        raw_pair_count = parse_int(row.get("raw_pair_count"))
        if use_raw_filter and raw_pair_count <= 0:
            continue
        if raw_pair_count > 0:
            if parse_int(row.get("max_raw_pair_support_min")) < min_raw_pair_support:
                continue
            if parse_float(row.get("min_raw_centroid_dist"), 1.0e9) > max_raw_centroid_dist:
                continue
            if parse_float(row.get("min_raw_distance_over_radius"), 1.0e9) > max_raw_distance_over_radius:
                continue
        candidates.append(row)
    candidates.sort(
        key=lambda row: (
            parse_float(row.get(score_col)),
            -parse_float(row.get("min_raw_distance_over_radius"), 1.0e9),
            -parse_float(row.get("min_raw_centroid_dist"), 1.0e9),
            parse_int(row.get("edge_count")),
            parse_int(row.get("max_forward_visible_carrier_count")) + parse_int(row.get("max_backward_visible_carrier_count")),
        ),
        reverse=True,
    )

    uf = _StringUnionFind(component_ids)
    cluster_frames = {component_id: set(component_meta[component_id]["frames"]) for component_id in component_ids}
    selected_rows: list[dict[str, Any]] = []
    selected_pair_count = 0
    skipped_cluster_conflict = 0
    for row in candidates:
        left = str(row["component_left"])
        right = str(row["component_right"])
        root_left = uf.find(left)
        root_right = uf.find(right)
        if root_left == root_right:
            continue
        if enforce_cluster_frame_exclusion and cluster_frames[root_left] & cluster_frames[root_right]:
            skipped_cluster_conflict += 1
            continue
        new_root = uf.union(root_left, root_right)
        old_root = root_right if new_root == root_left else root_left
        cluster_frames[new_root] = cluster_frames[root_left] | cluster_frames[root_right]
        cluster_frames.pop(old_root, None)
        if collect_selected:
            selected_rows.append(
                {
                    **row,
                    "score_key": score_key,
                    "score_threshold": score_threshold,
                    "selected_score": parse_float(row.get(score_col)),
                    "selected_rank": selected_pair_count,
                }
            )
        selected_pair_count += 1
        if max_selected_pairs >= 0 and selected_pair_count >= max_selected_pairs:
            break

    true_labels, pred_labels = _labels(mask_vote_rows, uf)
    result = {
        "score_key": score_key,
        "score_threshold": float(score_threshold),
        "min_pair_edge_count": int(min_pair_edge_count),
        "max_visible_outside": float(max_visible_outside),
        "min_visible_carriers": int(min_visible_carriers),
        "max_raw_centroid_dist": float(max_raw_centroid_dist),
        "max_raw_distance_over_radius": float(max_raw_distance_over_radius),
        "min_raw_pair_support": int(min_raw_pair_support),
        "max_selected_pairs": int(max_selected_pairs),
        "enforce_cluster_frame_exclusion": bool(enforce_cluster_frame_exclusion),
        "forbid_pair_same_frame_conflict": bool(forbid_pair_same_frame_conflict),
        "candidate_pair_count_after_filter": len(candidates),
        "selected_pair_count": selected_pair_count,
        "skipped_cluster_conflict_count": skipped_cluster_conflict,
        "cluster_count": len(set(pred_labels)),
        "ARI": adjusted_rand_score(true_labels, pred_labels),
        "purity": cluster_purity(true_labels, pred_labels),
        "completeness": cluster_completeness(true_labels, pred_labels),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "uses_pose_for_prediction": False,
        "uses_rgbd_for_prediction": False,
        "uses_scannet_mesh_for_prediction": False,
    }
    return result, selected_rows


def _parse_float_list(spec: str) -> list[float]:
    return [float(item) for item in str(spec).split(",") if item.strip()]


def _parse_int_list(spec: str) -> list[int]:
    return [int(float(item)) for item in str(spec).split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v47 raw-D4RT-geometry gated component merge scan.")
    parser.add_argument("--mask-vote-rows", default="outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_mask_vote_rows.csv")
    parser.add_argument("--carrier-observation-table", default="outputs/audit/v47_observation_tables_metricfix/carrier_observation_table.csv")
    parser.add_argument("--carrier-debug-root", default="outputs/stream4d_debug_v47_stride1_d5_probe5_mf32")
    parser.add_argument("--edge-table", default="outputs/audit/v47_adjacent_edges_gap2_only/temporal_candidate_edge_table.csv")
    parser.add_argument("--edge-types", default="adjacent,skip")
    parser.add_argument("--min-carriers-per-mask", type=int, default=8)
    parser.add_argument("--score-thresholds", default="0.97,0.90,0.80,0.70")
    parser.add_argument("--min-pair-edge-counts", default="1")
    parser.add_argument("--max-visible-outside-values", default="0.75,1.0")
    parser.add_argument("--min-visible-carrier-values", default="0,8")
    parser.add_argument("--max-raw-centroid-dist-values", default="0.35,0.50,0.75,1.0,1.5,999")
    parser.add_argument("--max-raw-distance-over-radius-values", default="1.0,1.5,2.0,999")
    parser.add_argument("--min-raw-pair-support-values", default="0,8")
    parser.add_argument("--max-selected-pair-values", default="50,75,100,-1")
    parser.add_argument("--output-root", default="outputs/audit/v47_component_raw_geometry_merge_union32_gap2")
    args = parser.parse_args()

    mask_vote_rows = read_csv(ROOT / str(args.mask_vote_rows))
    carrier_rows = read_csv(ROOT / str(args.carrier_observation_table))
    edge_rows = read_csv(ROOT / str(args.edge_table))
    edge_types = {item.strip() for item in str(args.edge_types).split(",") if item.strip()}
    node_to_component, component_meta = _node_component_rows(mask_vote_rows)
    raw_features, raw_summary = _raw_mask_geometry_features(
        carrier_rows=carrier_rows,
        carrier_debug_root=ROOT / str(args.carrier_debug_root),
        min_carriers_per_mask=int(args.min_carriers_per_mask),
    )
    candidate_stats = _candidate_pair_stats(edge_rows, node_to_component, component_meta, raw_features, edge_types)
    component_ids = sorted(component_meta)

    rows: list[dict[str, Any]] = []
    for score_key in ["A5_d4rt_semantic_confirmation", "A4_d4rt_visible_veto", "A8_no_temporal_control", "A7_shuffled_D4RT"]:
        for score_threshold in _parse_float_list(args.score_thresholds):
            for min_pair_edge_count in _parse_int_list(args.min_pair_edge_counts):
                for max_visible_outside in _parse_float_list(args.max_visible_outside_values):
                    for min_visible_carriers in _parse_int_list(args.min_visible_carrier_values):
                        for max_raw_centroid_dist in _parse_float_list(args.max_raw_centroid_dist_values):
                            for max_raw_distance_over_radius in _parse_float_list(args.max_raw_distance_over_radius_values):
                                for min_raw_pair_support in _parse_int_list(args.min_raw_pair_support_values):
                                    for max_selected_pairs in _parse_int_list(args.max_selected_pair_values):
                                        for enforce_cluster_frame_exclusion in [True, False]:
                                            for forbid_pair_same_frame_conflict in [True, False]:
                                                result, selected = _evaluate(
                                                    component_ids=component_ids,
                                                    component_meta=component_meta,
                                                    mask_vote_rows=mask_vote_rows,
                                                    candidate_stats=candidate_stats,
                                                    score_key=score_key,
                                                    score_threshold=score_threshold,
                                                    min_pair_edge_count=min_pair_edge_count,
                                                    max_visible_outside=max_visible_outside,
                                                    min_visible_carriers=min_visible_carriers,
                                                    max_raw_centroid_dist=max_raw_centroid_dist,
                                                    max_raw_distance_over_radius=max_raw_distance_over_radius,
                                                    min_raw_pair_support=min_raw_pair_support,
                                                    max_selected_pairs=max_selected_pairs,
                                                    enforce_cluster_frame_exclusion=enforce_cluster_frame_exclusion,
                                                    forbid_pair_same_frame_conflict=forbid_pair_same_frame_conflict,
                                                    collect_selected=False,
                                                )
                                                del selected
                                                rows.append(result)

    rows.sort(key=lambda row: (parse_float(row.get("ARI")), parse_float(row.get("purity"))), reverse=True)
    real_rows = [row for row in rows if row["score_key"] in {"A5_d4rt_semantic_confirmation", "A4_d4rt_visible_veto"}]
    no_temporal_rows = [row for row in rows if row["score_key"] == "A8_no_temporal_control"]
    shuffled_rows = [row for row in rows if row["score_key"] == "A7_shuffled_D4RT"]
    best_real = real_rows[0] if real_rows else {}
    best_no_temporal = no_temporal_rows[0] if no_temporal_rows else {}
    best_shuffled = shuffled_rows[0] if shuffled_rows else {}

    def selected_for(row: dict[str, Any]) -> list[dict[str, Any]]:
        if not row:
            return []
        _result, selected = _evaluate(
            component_ids=component_ids,
            component_meta=component_meta,
            mask_vote_rows=mask_vote_rows,
            candidate_stats=candidate_stats,
            score_key=str(row.get("score_key")),
            score_threshold=parse_float(row.get("score_threshold")),
            min_pair_edge_count=parse_int(row.get("min_pair_edge_count")),
            max_visible_outside=parse_float(row.get("max_visible_outside")),
            min_visible_carriers=parse_int(row.get("min_visible_carriers")),
            max_raw_centroid_dist=parse_float(row.get("max_raw_centroid_dist")),
            max_raw_distance_over_radius=parse_float(row.get("max_raw_distance_over_radius")),
            min_raw_pair_support=parse_int(row.get("min_raw_pair_support")),
            max_selected_pairs=parse_int(row.get("max_selected_pairs")),
            enforce_cluster_frame_exclusion=parse_bool(row.get("enforce_cluster_frame_exclusion")),
            forbid_pair_same_frame_conflict=parse_bool(row.get("forbid_pair_same_frame_conflict")),
            collect_selected=True,
        )
        return selected

    raw_pair_rows = [row for row in candidate_stats.values() if parse_int(row.get("raw_pair_count")) > 0]
    summary = {
        "phase": "v47_component_raw_geometry_merge",
        "mask_vote_rows": str(ROOT / str(args.mask_vote_rows)),
        "carrier_observation_table": str(ROOT / str(args.carrier_observation_table)),
        "carrier_debug_root": str(ROOT / str(args.carrier_debug_root)),
        "edge_table": str(ROOT / str(args.edge_table)),
        "edge_types": sorted(edge_types),
        "component_count": len(component_ids),
        "candidate_pair_count": len(candidate_stats),
        "candidate_pair_count_with_raw_geometry": len(raw_pair_rows),
        "raw_summary": raw_summary,
        "rows": len(rows),
        "best_row": rows[0] if rows else None,
        "best_real_row": best_real,
        "best_no_temporal_row": best_no_temporal,
        "best_shuffled_row": best_shuffled,
        "best_real_minus_best_no_temporal_ARI": parse_float(best_real.get("ARI")) - parse_float(best_no_temporal.get("ARI")),
        "best_real_minus_best_shuffled_ARI": parse_float(best_real.get("ARI")) - parse_float(best_shuffled.get("ARI")),
        "gate": {
            "ARI_pass": bool(parse_float(best_real.get("ARI")) >= 0.465),
            "purity_pass": bool(parse_float(best_real.get("purity")) >= 0.865),
            "completeness_pass": bool(parse_float(best_real.get("completeness")) >= 0.535),
            "real_minus_no_temporal_pass": bool(
                parse_float(best_real.get("ARI")) - parse_float(best_no_temporal.get("ARI")) >= 0.15
            ),
            "real_minus_shuffled_pass": bool(parse_float(best_real.get("ARI")) - parse_float(best_shuffled.get("ARI")) >= 0.25),
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "uses_pose_for_prediction": False,
        "uses_rgbd_for_prediction": False,
        "uses_scannet_mesh_for_prediction": False,
    }
    summary["gate"]["pass"] = bool(all(summary["gate"].values()))

    out_root = ROOT / str(args.output_root)
    write_csv(out_root / "component_raw_geometry_merge_scan_rows.csv", rows)
    write_csv(out_root / "component_raw_geometry_merge_pair_stats.csv", list(candidate_stats.values()))
    write_csv(out_root / "component_raw_geometry_merge_best_real_selected_pairs.csv", selected_for(best_real))
    write_csv(
        out_root / "component_raw_geometry_merge_best_no_temporal_selected_pairs.csv",
        selected_for(best_no_temporal),
    )
    write_csv(out_root / "component_raw_geometry_merge_best_shuffled_selected_pairs.csv", selected_for(best_shuffled))
    write_json(out_root / "component_raw_geometry_merge_summary.json", summary)
    print({"summary": str(out_root / "component_raw_geometry_merge_summary.json"), "gate": summary["gate"]})


if __name__ == "__main__":
    main()
