from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from typing import Any

import numpy as np

from stream4d_native.v47_common import (
    ROOT,
    adjusted_rand_score,
    cluster_completeness,
    cluster_purity,
    cosine,
    parse_bool,
    parse_float,
    parse_int,
    read_csv,
    safe_mean,
    write_csv,
    write_json,
)


class _StringUnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}
        self.members = {item: {item} for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> str:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return root_left
        if len(self.members[root_left]) < len(self.members[root_right]):
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        self.members[root_left].update(self.members[root_right])
        self.members.pop(root_right, None)
        return root_left


def _parse_float_list(spec: str) -> list[float]:
    return [float(item) for item in str(spec).split(",") if item.strip()]


def _parse_bool_list(spec: str) -> list[bool]:
    return [str(item).strip().lower() in {"1", "true", "yes", "y"} for item in str(spec).split(",") if item.strip()]


def _loads_feature(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(v) for v in value]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return []
    return [float(v) for v in loaded] if isinstance(loaded, list) else []


def _mean_feature(features: list[list[float]]) -> list[float]:
    if not features:
        return []
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return []
    mean = arr.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm > 0.0:
        mean = mean / norm
    return [float(value) for value in mean.tolist()]


def _stable_shuffle(values: list[str], seed: str) -> dict[str, str]:
    keyed = [(hashlib.sha1(f"{seed}:{idx}:{value}".encode("utf-8")).hexdigest(), value) for idx, value in enumerate(values)]
    keyed.sort()
    ordered = [value for _key, value in keyed]
    if len(ordered) <= 1:
        return {value: value for value in ordered}
    shifted = ordered[1:] + ordered[:1]
    return dict(zip(ordered, shifted))


def _component_meta(mask_vote_rows: list[dict[str, Any]], mask_rows: list[dict[str, Any]]) -> tuple[dict[int, str], dict[str, dict[str, Any]]]:
    mask_by_obs = {str(row.get("mask_observation_id")): row for row in mask_rows}
    node_to_component: dict[int, str] = {}
    meta: dict[str, dict[str, Any]] = {}
    for row in mask_vote_rows:
        component = str(row.get("predicted_component_object_id") or row.get("predicted_supertrack_object_id"))
        node_id = parse_int(row.get("node_id"))
        node_to_component[node_id] = component
        item = meta.setdefault(
            component,
            {
                "component_id": component,
                "scene": str(row.get("scene")),
                "frames": set(),
                "nodes": [],
                "features": [],
                "mask_count": 0,
            },
        )
        item["frames"].add(parse_int(row.get("frame_id")))
        item["nodes"].append(node_id)
        item["mask_count"] += 1
        feature = _loads_feature(mask_by_obs.get(str(row.get("mask_observation_id")), {}).get("core_feature"))
        if feature:
            item["features"].append(feature)
    for item in meta.values():
        item["feature"] = _mean_feature(item["features"])
    return node_to_component, meta


def _pair_stats(
    edge_rows: list[dict[str, Any]],
    node_to_component: dict[int, str],
    component_meta: dict[str, dict[str, Any]],
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
                "max_A5_d4rt_semantic_confirmation": 0.0,
                "max_A8_no_temporal_control": 0.0,
                "max_A7_shuffled_D4RT": 0.0,
                "min_visible_outside": 1.0,
                "same_frame_conflict": bool(set(left["frames"]) & set(right["frames"])),
            },
        )
        item["edge_count"] += 1
        for key_name in ["A5_d4rt_semantic_confirmation", "A8_no_temporal_control", "A7_shuffled_D4RT"]:
            item[f"max_{key_name}"] = max(parse_float(item.get(f"max_{key_name}")), parse_float(row.get(key_name)))
        item["min_visible_outside"] = min(parse_float(item.get("min_visible_outside"), 1.0), parse_float(row.get("visible_outside"), 1.0))
    return stats


def _semantic_pairs(component_meta: dict[str, dict[str, Any]], *, shuffled: bool) -> dict[tuple[str, str], float]:
    by_scene: dict[str, list[str]] = defaultdict(list)
    for component, item in component_meta.items():
        by_scene[str(item["scene"])].append(component)
    pair_scores: dict[tuple[str, str], float] = {}
    for scene, components in by_scene.items():
        components = sorted(components)
        shuffle_map = _stable_shuffle(components, f"v47_mdl_semantic_shuffle:{scene}") if shuffled else {c: c for c in components}
        for idx, left in enumerate(components):
            left_feature = component_meta[left].get("feature", [])
            for right in components[idx + 1 :]:
                right_feature = component_meta[shuffle_map[right]].get("feature", [])
                pair_scores[(left, right)] = cosine(left_feature, right_feature)
    return pair_scores


def _complete_link_ok(
    left_members: set[str],
    right_members: set[str],
    pair_scores: dict[tuple[str, str], float],
    threshold: float,
) -> bool:
    for left in left_members:
        for right in right_members:
            if left == right:
                continue
            key = tuple(sorted([left, right]))
            if parse_float(pair_scores.get(key), -1.0) < float(threshold):
                return False
    return True


def _metrics(mask_vote_rows: list[dict[str, Any]], component_meta: dict[str, dict[str, Any]], uf: _StringUnionFind) -> dict[str, Any]:
    true_labels: list[str] = []
    pred_labels: list[str] = []
    scene_true: dict[str, list[str]] = defaultdict(list)
    scene_pred: dict[str, list[str]] = defaultdict(list)
    scene_clusters: dict[str, set[str]] = defaultdict(set)
    frames_by_root: dict[str, set[int]] = defaultdict(set)
    for component, item in component_meta.items():
        root = uf.find(component)
        scene_clusters[str(item["scene"])].add(root)
        frames_by_root[root].update(item["frames"])
    for row in mask_vote_rows:
        gt = str(row.get("diagnostic_gt_instance", ""))
        if not gt:
            continue
        component = str(row.get("predicted_component_object_id") or row.get("predicted_supertrack_object_id"))
        if component not in uf.parent:
            continue
        pred = uf.find(component)
        scene = str(row.get("scene"))
        true_labels.append(gt)
        pred_labels.append(pred)
        scene_true[scene].append(gt)
        scene_pred[scene].append(pred)

    def first_scene(prefix: str) -> str:
        return next((scene for scene in sorted(scene_true) if scene.startswith(prefix)), "")

    def scene_metric(prefix: str, metric: str) -> float | None:
        scene = first_scene(prefix)
        if not scene:
            return None
        if metric == "ARI":
            return adjusted_rand_score(scene_true[scene], scene_pred[scene])
        if metric == "purity":
            return cluster_purity(scene_true[scene], scene_pred[scene])
        if metric == "completeness":
            return cluster_completeness(scene_true[scene], scene_pred[scene])
        raise ValueError(metric)

    return {
        "ARI": adjusted_rand_score(true_labels, pred_labels),
        "purity": cluster_purity(true_labels, pred_labels),
        "completeness": cluster_completeness(true_labels, pred_labels),
        "temporal_span_mean": safe_mean(len(frames) for frames in frames_by_root.values()),
        "mean_predictions_per_scene": safe_mean(len(clusters) for clusters in scene_clusters.values()),
        "scene0081_ARI": scene_metric("scene0081", "ARI"),
        "scene0011_purity": scene_metric("scene0011", "purity"),
        "scene0050_purity": scene_metric("scene0050", "purity"),
        "scene0591_completeness": scene_metric("scene0591", "completeness"),
        "cluster_count": len(set(pred_labels)),
        "birth_from_d4rt_tube_count": 0,
        "maskless_object_count": 0,
    }


def _evaluate(
    *,
    variant: str,
    component_ids: list[str],
    component_meta: dict[str, dict[str, Any]],
    mask_vote_rows: list[dict[str, Any]],
    pair_scores: dict[tuple[str, str], float],
    pair_stats: dict[tuple[str, str], dict[str, Any]],
    threshold: float,
    min_d4rt: float,
    max_visible_outside: float,
    allow_frame_overlap: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    uf = _StringUnionFind(component_ids)
    candidates: list[tuple[float, str, str, dict[str, Any]]] = []
    for pair, score in pair_scores.items():
        left, right = pair
        if component_meta[left]["scene"] != component_meta[right]["scene"]:
            continue
        stats = pair_stats.get(pair, {})
        if variant == "M2_d4rt_confirmed_complete_link":
            if parse_float(stats.get("max_A5_d4rt_semantic_confirmation")) < float(min_d4rt):
                continue
            if parse_float(stats.get("min_visible_outside"), 1.0) > float(max_visible_outside):
                continue
        elif variant == "M3_no_temporal_confirmed_control":
            if parse_float(stats.get("max_A8_no_temporal_control")) < float(min_d4rt):
                continue
            if parse_float(stats.get("min_visible_outside"), 1.0) > float(max_visible_outside):
                continue
        elif variant == "M4_shuffled_d4rt_confirmed_control":
            if parse_float(stats.get("max_A7_shuffled_D4RT")) < float(min_d4rt):
                continue
            if parse_float(stats.get("min_visible_outside"), 1.0) > float(max_visible_outside):
                continue
        if parse_float(score) < float(threshold):
            continue
        candidates.append((parse_float(score), left, right, stats))
    candidates.sort(reverse=True)
    selected: list[dict[str, Any]] = []
    skipped_frame_overlap = 0
    skipped_complete_link = 0
    for score, left, right, stats in candidates:
        root_left = uf.find(left)
        root_right = uf.find(right)
        if root_left == root_right:
            continue
        left_members = uf.members[root_left]
        right_members = uf.members[root_right]
        if not allow_frame_overlap:
            left_frames = set().union(*(component_meta[item]["frames"] for item in left_members))
            right_frames = set().union(*(component_meta[item]["frames"] for item in right_members))
            if left_frames & right_frames:
                skipped_frame_overlap += 1
                continue
        if not _complete_link_ok(left_members, right_members, pair_scores, threshold):
            skipped_complete_link += 1
            continue
        uf.union(root_left, root_right)
        selected.append(
            {
                "component_left": left,
                "component_right": right,
                "variant": variant,
                "semantic_score": score,
                "threshold": float(threshold),
                "min_d4rt": float(min_d4rt),
                "max_visible_outside": float(max_visible_outside),
                "selected_rank": len(selected),
                **stats,
            }
        )
    result = {
        "variant": variant,
        "threshold": float(threshold),
        "min_d4rt": float(min_d4rt),
        "max_visible_outside": float(max_visible_outside),
        "allow_frame_overlap": bool(allow_frame_overlap),
        "candidate_pair_count_after_filter": len(candidates),
        "merge_count": len(selected),
        "skipped_frame_overlap_count": skipped_frame_overlap,
        "skipped_complete_link_count": skipped_complete_link,
        **_metrics(mask_vote_rows, component_meta, uf),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    return result, selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v47 carrier-component MDL-style semantic complete-link scan with controls.")
    parser.add_argument("--observation-root", default="outputs/audit/v47_observation_tables_metricfix")
    parser.add_argument("--carrier-root", default="outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix")
    parser.add_argument("--edge-table", default="outputs/audit/v47_adjacent_edges_gap2_only/temporal_candidate_edge_table.csv")
    parser.add_argument("--edge-types", default="adjacent,skip")
    parser.add_argument("--thresholds", default="0.9997,0.9995,0.999,0.9985,0.998,0.997,0.996,0.995,0.992,0.990")
    parser.add_argument("--min-d4rt-values", default="0.97,0.90,0.75")
    parser.add_argument("--max-visible-outside-values", default="0.45,0.75,1.0")
    parser.add_argument("--allow-frame-overlap-values", default="false,true")
    parser.add_argument("--output-root", default="outputs/audit/v47_carrier_component_mdl_semantic")
    args = parser.parse_args()

    mask_rows = read_csv(ROOT / str(args.observation_root) / "mask_observation_table.csv")
    mask_vote_rows = read_csv(ROOT / str(args.carrier_root) / "carrier_supertrack_mask_vote_rows.csv")
    edge_rows = read_csv(ROOT / str(args.edge_table))
    edge_types = {item.strip() for item in str(args.edge_types).split(",") if item.strip()}
    node_to_component, component_meta = _component_meta(mask_vote_rows, mask_rows)
    component_ids = sorted(component_meta)
    pair_stats = _pair_stats(edge_rows, node_to_component, component_meta, edge_types)
    semantic_pairs = _semantic_pairs(component_meta, shuffled=False)
    shuffled_pairs = _semantic_pairs(component_meta, shuffled=True)

    rows: list[dict[str, Any]] = []
    selected_by_signature: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for threshold in _parse_float_list(args.thresholds):
        for allow_frame_overlap in _parse_bool_list(args.allow_frame_overlap_values):
            for variant, pairs in [
                ("M1_semantic_complete_link", semantic_pairs),
                ("M1_shuffled_semantic_control", shuffled_pairs),
            ]:
                result, selected = _evaluate(
                    variant=variant,
                    component_ids=component_ids,
                    component_meta=component_meta,
                    mask_vote_rows=mask_vote_rows,
                    pair_scores=pairs,
                    pair_stats=pair_stats,
                    threshold=threshold,
                    min_d4rt=0.0,
                    max_visible_outside=1.0,
                    allow_frame_overlap=allow_frame_overlap,
                )
                signature = (variant, result["threshold"], result["min_d4rt"], result["max_visible_outside"], result["allow_frame_overlap"])
                selected_by_signature[signature] = selected
                rows.append(result)
            for min_d4rt in _parse_float_list(args.min_d4rt_values):
                for max_visible_outside in _parse_float_list(args.max_visible_outside_values):
                    for variant, pairs in [
                        ("M2_d4rt_confirmed_complete_link", semantic_pairs),
                        ("M3_no_temporal_confirmed_control", semantic_pairs),
                        ("M4_shuffled_d4rt_confirmed_control", semantic_pairs),
                    ]:
                        result, selected = _evaluate(
                            variant=variant,
                            component_ids=component_ids,
                            component_meta=component_meta,
                            mask_vote_rows=mask_vote_rows,
                            pair_scores=pairs,
                            pair_stats=pair_stats,
                            threshold=threshold,
                            min_d4rt=min_d4rt,
                            max_visible_outside=max_visible_outside,
                            allow_frame_overlap=allow_frame_overlap,
                        )
                        signature = (
                            variant,
                            result["threshold"],
                            result["min_d4rt"],
                            result["max_visible_outside"],
                            result["allow_frame_overlap"],
                        )
                        selected_by_signature[signature] = selected
                        rows.append(result)

    rows.sort(key=lambda row: (parse_float(row.get("ARI")), parse_float(row.get("purity"))), reverse=True)
    real_rows = [row for row in rows if row["variant"] in {"M1_semantic_complete_link", "M2_d4rt_confirmed_complete_link"}]
    shuffled_semantic_rows = [row for row in rows if row["variant"] == "M1_shuffled_semantic_control"]
    no_temporal_rows = [row for row in rows if row["variant"] == "M3_no_temporal_confirmed_control"]
    shuffled_d4rt_rows = [row for row in rows if row["variant"] == "M4_shuffled_d4rt_confirmed_control"]
    safe_real_rows = [row for row in real_rows if parse_float(row.get("purity")) >= 0.875]
    best_real = max(safe_real_rows or real_rows, key=lambda row: (parse_float(row.get("ARI")), parse_float(row.get("purity"))), default={})
    best_shuffled_semantic = max(shuffled_semantic_rows, key=lambda row: parse_float(row.get("ARI")), default={})
    best_no_temporal = max(no_temporal_rows, key=lambda row: parse_float(row.get("ARI")), default={})
    best_shuffled_d4rt = max(shuffled_d4rt_rows, key=lambda row: parse_float(row.get("ARI")), default={})

    def signature(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            row.get("variant"),
            row.get("threshold"),
            row.get("min_d4rt"),
            row.get("max_visible_outside"),
            row.get("allow_frame_overlap"),
        )

    gate = {
        "ARI_pass": bool(parse_float(best_real.get("ARI")) >= 0.465),
        "purity_pass": bool(parse_float(best_real.get("purity")) >= 0.875),
        "completeness_pass": bool(parse_float(best_real.get("completeness")) >= 0.535),
        "real_minus_shuffled_semantic_pass": bool(parse_float(best_real.get("ARI")) - parse_float(best_shuffled_semantic.get("ARI")) >= 0.02),
        "real_minus_no_temporal_pass": bool(parse_float(best_real.get("ARI")) - parse_float(best_no_temporal.get("ARI")) >= 0.02),
        "real_minus_shuffled_d4rt_pass": bool(parse_float(best_real.get("ARI")) - parse_float(best_shuffled_d4rt.get("ARI")) >= 0.02),
    }
    gate["pass"] = bool(all(gate.values()))
    summary = {
        "phase": "v47_carrier_component_mdl_semantic",
        "carrier_root": str(ROOT / str(args.carrier_root)),
        "edge_table": str(ROOT / str(args.edge_table)),
        "component_count": len(component_ids),
        "pair_stat_count": len(pair_stats),
        "rows": len(rows),
        "best_real_row": best_real,
        "best_shuffled_semantic_row": best_shuffled_semantic,
        "best_no_temporal_row": best_no_temporal,
        "best_shuffled_d4rt_row": best_shuffled_d4rt,
        "best_real_minus_best_shuffled_semantic_ARI": parse_float(best_real.get("ARI")) - parse_float(best_shuffled_semantic.get("ARI")),
        "best_real_minus_best_no_temporal_ARI": parse_float(best_real.get("ARI")) - parse_float(best_no_temporal.get("ARI")),
        "best_real_minus_best_shuffled_d4rt_ARI": parse_float(best_real.get("ARI")) - parse_float(best_shuffled_d4rt.get("ARI")),
        "gate": gate,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }

    out = ROOT / str(args.output_root)
    write_csv(out / "carrier_component_mdl_semantic_scan_rows.csv", rows)
    write_csv(out / "carrier_component_mdl_semantic_best_real_selected_pairs.csv", selected_by_signature.get(signature(best_real), []))
    write_csv(
        out / "carrier_component_mdl_semantic_best_shuffled_semantic_selected_pairs.csv",
        selected_by_signature.get(signature(best_shuffled_semantic), []),
    )
    write_csv(out / "carrier_component_mdl_semantic_best_no_temporal_selected_pairs.csv", selected_by_signature.get(signature(best_no_temporal), []))
    write_csv(out / "carrier_component_mdl_semantic_best_shuffled_d4rt_selected_pairs.csv", selected_by_signature.get(signature(best_shuffled_d4rt), []))
    write_json(out / "carrier_component_mdl_semantic_summary.json", summary)
    print({"summary": str(out / "carrier_component_mdl_semantic_summary.json"), "gate": gate})


if __name__ == "__main__":
    main()
