from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
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


def _parse_root_list(spec: str) -> list[str]:
    return [item.strip() for item in str(spec).split(",") if item.strip()]


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


def _stable_order(values: list[str], seed: str) -> list[str]:
    keyed = [(hashlib.sha1(f"{seed}:{idx}:{value}".encode("utf-8")).hexdigest(), value) for idx, value in enumerate(values)]
    keyed.sort()
    return [value for _key, value in keyed]


def _stable_shuffle_map(values: list[str], seed: str) -> dict[str, str]:
    ordered = _stable_order(values, seed)
    if len(ordered) <= 1:
        return {value: value for value in ordered}
    shifted = ordered[1:] + ordered[:1]
    return dict(zip(ordered, shifted))


def _component_meta(fine_vote_rows: list[dict[str, Any]], mask_rows: list[dict[str, Any]]) -> tuple[dict[int, str], dict[str, dict[str, Any]]]:
    mask_by_obs = {str(row.get("mask_observation_id")): row for row in mask_rows}
    node_to_fine: dict[int, str] = {}
    meta: dict[str, dict[str, Any]] = {}
    for row in fine_vote_rows:
        component = str(row.get("predicted_component_object_id") or row.get("predicted_supertrack_object_id"))
        node_id = parse_int(row.get("node_id"))
        node_to_fine[node_id] = component
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
    return node_to_fine, meta


def _coarse_parent_map(
    *,
    fine_vote_rows: list[dict[str, Any]],
    coarse_vote_rows: list[dict[str, Any]],
    node_to_fine: dict[int, str],
    component_meta: dict[str, dict[str, Any]],
    coarse_name: str,
) -> dict[str, str]:
    coarse_by_node = {
        parse_int(row.get("node_id")): str(row.get("predicted_component_object_id") or row.get("predicted_supertrack_object_id"))
        for row in coarse_vote_rows
    }
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in fine_vote_rows:
        node_id = parse_int(row.get("node_id"))
        fine = node_to_fine.get(node_id)
        coarse = coarse_by_node.get(node_id)
        if not fine or coarse is None:
            continue
        scene = str(component_meta.get(fine, {}).get("scene", row.get("scene")))
        counts[fine][f"{coarse_name}:{scene}:{coarse}"] += 1
    parent: dict[str, str] = {}
    for fine in component_meta:
        if counts.get(fine):
            parent[fine] = counts[fine].most_common(1)[0][0]
        else:
            scene = str(component_meta[fine]["scene"])
            parent[fine] = f"{coarse_name}:{scene}:missing:{fine}"
    return parent


def _shuffle_parent_map(parent: dict[str, str], component_meta: dict[str, dict[str, Any]], seed: str) -> dict[str, str]:
    by_scene: dict[str, list[str]] = defaultdict(list)
    for component in parent:
        by_scene[str(component_meta[component]["scene"])].append(component)
    shuffled: dict[str, str] = {}
    for scene, components in by_scene.items():
        components = sorted(components)
        parents = [parent[component] for component in components]
        shuffled_parents = _stable_order(parents, f"{seed}:{scene}:parents")
        for component, new_parent in zip(components, shuffled_parents):
            shuffled[component] = f"shuffled:{new_parent}"
    return shuffled


def _semantic_pairs(component_meta: dict[str, dict[str, Any]], *, shuffled: bool) -> dict[tuple[str, str], float]:
    by_scene: dict[str, list[str]] = defaultdict(list)
    for component, item in component_meta.items():
        by_scene[str(item["scene"])].append(component)
    scores: dict[tuple[str, str], float] = {}
    for scene, components in by_scene.items():
        components = sorted(components)
        shuffle_map = _stable_shuffle_map(components, f"v47_coarse_to_fine_semantic_shuffle:{scene}") if shuffled else {c: c for c in components}
        for idx, left in enumerate(components):
            left_feature = component_meta[left].get("feature", [])
            for right in components[idx + 1 :]:
                right_feature = component_meta[shuffle_map[right]].get("feature", [])
                scores[(left, right)] = cosine(left_feature, right_feature)
    return scores


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


def _metrics(
    fine_vote_rows: list[dict[str, Any]],
    component_meta: dict[str, dict[str, Any]],
    root_for_component: dict[str, str],
) -> dict[str, Any]:
    true_labels: list[str] = []
    pred_labels: list[str] = []
    scene_true: dict[str, list[str]] = defaultdict(list)
    scene_pred: dict[str, list[str]] = defaultdict(list)
    scene_clusters: dict[str, set[str]] = defaultdict(set)
    frames_by_root: dict[str, set[int]] = defaultdict(set)
    for component, item in component_meta.items():
        root = root_for_component.get(component, component)
        scene_clusters[str(item["scene"])].add(root)
        frames_by_root[root].update(item["frames"])
    for row in fine_vote_rows:
        gt = str(row.get("diagnostic_gt_instance", ""))
        if not gt:
            continue
        component = str(row.get("predicted_component_object_id") or row.get("predicted_supertrack_object_id"))
        pred = root_for_component.get(component, component)
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


def _evaluate_parent_baseline(
    *,
    variant: str,
    coarse_root: str,
    parent: dict[str, str],
    fine_vote_rows: list[dict[str, Any]],
    component_meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "variant": variant,
        "coarse_root": coarse_root,
        "threshold": None,
        "allow_frame_overlap": None,
        "candidate_pair_count_after_filter": 0,
        "merge_count": 0,
        "skipped_frame_overlap_count": 0,
        "skipped_complete_link_count": 0,
        **_metrics(fine_vote_rows, component_meta, parent),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _evaluate_semantic_split(
    *,
    variant: str,
    coarse_root: str,
    component_ids: list[str],
    component_meta: dict[str, dict[str, Any]],
    fine_vote_rows: list[dict[str, Any]],
    parent: dict[str, str],
    pair_scores: dict[tuple[str, str], float],
    threshold: float,
    allow_frame_overlap: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    uf = _StringUnionFind(component_ids)
    by_parent: dict[str, list[str]] = defaultdict(list)
    for component in component_ids:
        by_parent[parent.get(component, component)].append(component)

    candidates: list[tuple[float, str, str, str]] = []
    for parent_id, members in by_parent.items():
        members = sorted(members)
        for idx, left in enumerate(members):
            for right in members[idx + 1 :]:
                score = parse_float(pair_scores.get((left, right), pair_scores.get((right, left), -1.0)))
                if score >= float(threshold):
                    candidates.append((score, left, right, parent_id))
    candidates.sort(reverse=True)

    selected: list[dict[str, Any]] = []
    skipped_frame_overlap = 0
    skipped_complete_link = 0
    for score, left, right, parent_id in candidates:
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
                "coarse_parent": parent_id,
                "variant": variant,
                "coarse_root": coarse_root,
                "semantic_score": score,
                "threshold": float(threshold),
                "allow_frame_overlap": bool(allow_frame_overlap),
                "selected_rank": len(selected),
            }
        )
    root_for_component = {component: uf.find(component) for component in component_ids}
    result = {
        "variant": variant,
        "coarse_root": coarse_root,
        "threshold": float(threshold),
        "allow_frame_overlap": bool(allow_frame_overlap),
        "candidate_pair_count_after_filter": len(candidates),
        "merge_count": len(selected),
        "skipped_frame_overlap_count": skipped_frame_overlap,
        "skipped_complete_link_count": skipped_complete_link,
        **_metrics(fine_vote_rows, component_meta, root_for_component),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    return result, selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v47 coarse carrier evidence constrained fine-component MDL split/merge scan.")
    parser.add_argument("--observation-root", default="outputs/audit/v47_observation_tables_metricfix")
    parser.add_argument("--fine-carrier-root", default="outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix")
    parser.add_argument(
        "--coarse-carrier-roots",
        default=(
            "outputs/audit/v47_carrier_supertrack_union_36_fine_metricfix,"
            "outputs/audit/v47_carrier_supertrack_union_40_fine_metricfix,"
            "outputs/audit/v47_carrier_supertrack_union_44_fine_metricfix,"
            "outputs/audit/v47_carrier_supertrack_union_48_fine_metricfix,"
            "outputs/audit/v47_carrier_supertrack_union_52_fine_metricfix,"
            "outputs/audit/v47_carrier_supertrack_union_56_fine_metricfix,"
            "outputs/audit/v47_carrier_supertrack_union_60_fine_metricfix,"
            "outputs/audit/v47_carrier_supertrack_union_64_metricfix,"
            "outputs/audit/v47_carrier_supertrack_union_128_metricfix,"
            "outputs/audit/v47_carrier_supertrack_union_256_metricfix"
        ),
    )
    parser.add_argument("--thresholds", default="0.9997,0.9995,0.999,0.998,0.996,0.995,0.992,0.990,0.985,0.980,0.970,0.950,0.930,0.900,0.850")
    parser.add_argument("--allow-frame-overlap-values", default="false,true")
    parser.add_argument("--output-root", default="outputs/audit/v47_carrier_coarse_to_fine_mdl")
    args = parser.parse_args()

    mask_rows = read_csv(ROOT / str(args.observation_root) / "mask_observation_table.csv")
    fine_vote_rows = read_csv(ROOT / str(args.fine_carrier_root) / "carrier_supertrack_mask_vote_rows.csv")
    node_to_fine, component_meta = _component_meta(fine_vote_rows, mask_rows)
    component_ids = sorted(component_meta)
    semantic_pairs = _semantic_pairs(component_meta, shuffled=False)
    shuffled_semantic_pairs = _semantic_pairs(component_meta, shuffled=True)

    rows: list[dict[str, Any]] = []
    selected_by_signature: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for coarse_root in _parse_root_list(args.coarse_carrier_roots):
        coarse_path = ROOT / coarse_root
        coarse_vote_path = coarse_path / "carrier_supertrack_mask_vote_rows.csv"
        if not coarse_vote_path.exists():
            continue
        coarse_name = Path(coarse_root).name
        coarse_vote_rows = read_csv(coarse_vote_path)
        parent = _coarse_parent_map(
            fine_vote_rows=fine_vote_rows,
            coarse_vote_rows=coarse_vote_rows,
            node_to_fine=node_to_fine,
            component_meta=component_meta,
            coarse_name=coarse_name,
        )
        shuffled_parent = _shuffle_parent_map(parent, component_meta, f"v47_coarse_parent_shuffle:{coarse_name}")
        rows.append(
            _evaluate_parent_baseline(
                variant="D0_coarse_parent_baseline",
                coarse_root=coarse_root,
                parent=parent,
                fine_vote_rows=fine_vote_rows,
                component_meta=component_meta,
            )
        )
        rows.append(
            _evaluate_parent_baseline(
                variant="D0_shuffled_coarse_parent_control",
                coarse_root=coarse_root,
                parent=shuffled_parent,
                fine_vote_rows=fine_vote_rows,
                component_meta=component_meta,
            )
        )
        for threshold in _parse_float_list(args.thresholds):
            for allow_frame_overlap in _parse_bool_list(args.allow_frame_overlap_values):
                specs = [
                    ("D1_coarse_constrained_semantic_split", parent, semantic_pairs),
                    ("D2_shuffled_coarse_control", shuffled_parent, semantic_pairs),
                    ("D3_shuffled_semantic_control", parent, shuffled_semantic_pairs),
                ]
                for variant, parent_map, pairs in specs:
                    result, selected = _evaluate_semantic_split(
                        variant=variant,
                        coarse_root=coarse_root,
                        component_ids=component_ids,
                        component_meta=component_meta,
                        fine_vote_rows=fine_vote_rows,
                        parent=parent_map,
                        pair_scores=pairs,
                        threshold=threshold,
                        allow_frame_overlap=allow_frame_overlap,
                    )
                    signature = (result["variant"], result["coarse_root"], result["threshold"], result["allow_frame_overlap"])
                    selected_by_signature[signature] = selected
                    rows.append(result)

    rows.sort(key=lambda row: (parse_float(row.get("ARI")), parse_float(row.get("purity"))), reverse=True)
    real_rows = [row for row in rows if row["variant"] in {"D0_coarse_parent_baseline", "D1_coarse_constrained_semantic_split"}]
    shuffled_coarse_rows = [row for row in rows if row["variant"] in {"D0_shuffled_coarse_parent_control", "D2_shuffled_coarse_control"}]
    shuffled_semantic_rows = [row for row in rows if row["variant"] == "D3_shuffled_semantic_control"]
    safe_real_rows = [row for row in real_rows if parse_float(row.get("purity")) >= 0.875]
    best_real = max(safe_real_rows or real_rows, key=lambda row: (parse_float(row.get("ARI")), parse_float(row.get("purity"))), default={})
    best_shuffled_coarse = max(shuffled_coarse_rows, key=lambda row: parse_float(row.get("ARI")), default={})
    best_shuffled_semantic = max(shuffled_semantic_rows, key=lambda row: parse_float(row.get("ARI")), default={})

    def signature(row: dict[str, Any]) -> tuple[Any, ...]:
        return (row.get("variant"), row.get("coarse_root"), row.get("threshold"), row.get("allow_frame_overlap"))

    gate = {
        "ARI_pass": bool(parse_float(best_real.get("ARI")) >= 0.465),
        "purity_pass": bool(parse_float(best_real.get("purity")) >= 0.875),
        "completeness_pass": bool(parse_float(best_real.get("completeness")) >= 0.535),
        "real_minus_shuffled_coarse_pass": bool(parse_float(best_real.get("ARI")) - parse_float(best_shuffled_coarse.get("ARI")) >= 0.02),
        "real_minus_shuffled_semantic_pass": bool(parse_float(best_real.get("ARI")) - parse_float(best_shuffled_semantic.get("ARI")) >= 0.02),
    }
    gate["pass"] = bool(all(gate.values()))
    summary = {
        "phase": "v47_carrier_coarse_to_fine_mdl",
        "fine_carrier_root": str(ROOT / str(args.fine_carrier_root)),
        "component_count": len(component_ids),
        "rows": len(rows),
        "best_real_row": best_real,
        "best_shuffled_coarse_row": best_shuffled_coarse,
        "best_shuffled_semantic_row": best_shuffled_semantic,
        "best_real_minus_best_shuffled_coarse_ARI": parse_float(best_real.get("ARI")) - parse_float(best_shuffled_coarse.get("ARI")),
        "best_real_minus_best_shuffled_semantic_ARI": parse_float(best_real.get("ARI")) - parse_float(best_shuffled_semantic.get("ARI")),
        "gate": gate,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }

    out = ROOT / str(args.output_root)
    write_csv(out / "carrier_coarse_to_fine_mdl_scan_rows.csv", rows)
    write_csv(out / "carrier_coarse_to_fine_mdl_best_real_selected_pairs.csv", selected_by_signature.get(signature(best_real), []))
    write_csv(
        out / "carrier_coarse_to_fine_mdl_best_shuffled_coarse_selected_pairs.csv",
        selected_by_signature.get(signature(best_shuffled_coarse), []),
    )
    write_csv(
        out / "carrier_coarse_to_fine_mdl_best_shuffled_semantic_selected_pairs.csv",
        selected_by_signature.get(signature(best_shuffled_semantic), []),
    )
    write_json(out / "carrier_coarse_to_fine_mdl_summary.json", summary)
    print({"summary": str(out / "carrier_coarse_to_fine_mdl_summary.json"), "gate": gate})


if __name__ == "__main__":
    main()
