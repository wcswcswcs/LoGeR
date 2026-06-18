from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.scannet_stream import ScanNetStream
from stream4d_native.d4rt_scene_builder import D4RTNativeSceneBuilder, source_xy_from_uv
from stream4d_native.object_tube_io import TubeRecord
from tools.run_v26_object_quality_diagnostics import _json_safe, _read_split, assign_gt_labels
from tools.run_v28_proposal_oracle import _cluster_metrics
from tools.run_v36_masklet_first_identity import LOCAL_GATE
from tools.trace_v25_real_geometry_flow import chunks_to_records, load_scene_chunks_from_cache


DEFAULT_SPECS = [
    "watershed:probe5_full32",
    "dinov2_maskcut:sample8",
    "efficientsam3:sample8",
]


@dataclass
class RegionNode:
    node_id: int
    scene: str
    source: str
    mode: str
    frame_id: int
    mask_index: int
    area: int


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> bool:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return False
        if self.size[root_left] < self.size[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        self.size[root_left] += self.size[root_right]
        return True


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
            writer.writerow(row)


def _parse_specs(text: str) -> list[tuple[str, str]]:
    specs = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"external spec must be source:mode, got {item!r}")
        source, mode = item.split(":", 1)
        specs.append((source.strip(), mode.strip()))
    return specs


def _scene_external_dir(mask_root: Path, scene: str, source: str, mode: str) -> Path:
    scene_first = mask_root / scene / source / mode
    if scene_first.exists():
        return scene_first
    return mask_root / source / mode


def _load_masks(mask_root: Path, scene: str, source: str, mode: str, min_area: int) -> tuple[list[RegionNode], dict[int, np.ndarray], dict[str, Any]]:
    root = _scene_external_dir(mask_root, scene, source, mode)
    nodes: list[RegionNode] = []
    masks_by_frame: dict[int, list[np.ndarray]] = {}
    mask_by_node_id: dict[int, np.ndarray] = {}
    missing = not root.exists()
    if not missing:
        for path in sorted(root.glob(f"{source}_frame*_masks.npz")):
            stem = path.stem
            frame_text = stem.split("_frame", 1)[1].split("_", 1)[0]
            frame_id = int(frame_text)
            data = np.load(path)
            masks = np.asarray(data["masks"], dtype=bool)
            kept = []
            for idx, mask in enumerate(masks):
                area = int(mask.sum())
                if area < int(min_area):
                    continue
                node = RegionNode(
                    node_id=len(nodes),
                    scene=scene,
                    source=source,
                    mode=mode,
                    frame_id=frame_id,
                    mask_index=int(idx),
                    area=area,
                )
                nodes.append(node)
                mask_by_node_id[int(node.node_id)] = mask
                kept.append(mask)
            masks_by_frame[frame_id] = kept
    max_regions = int(getattr(_load_masks, "max_regions_per_scene", 0))
    capped = False
    uncapped_region_count = len(nodes)
    if max_regions > 0 and len(nodes) > max_regions:
        capped = True
        keep_old_ids = {
            int(node.node_id)
            for node in sorted(nodes, key=lambda item: (int(item.area), -int(item.frame_id)), reverse=True)[:max_regions]
        }
        rebuilt_nodes: list[RegionNode] = []
        rebuilt_masks: dict[int, list[np.ndarray]] = defaultdict(list)
        for node in sorted(nodes, key=lambda item: (int(item.frame_id), int(item.mask_index), int(item.node_id))):
            if int(node.node_id) not in keep_old_ids:
                continue
            frame_masks = rebuilt_masks[int(node.frame_id)]
            new_node = RegionNode(
                node_id=len(rebuilt_nodes),
                scene=node.scene,
                source=node.source,
                mode=node.mode,
                frame_id=node.frame_id,
                mask_index=len(frame_masks),
                area=node.area,
            )
            rebuilt_nodes.append(new_node)
            frame_masks.append(mask_by_node_id[int(node.node_id)])
        nodes = rebuilt_nodes
        masks_by_frame = dict(rebuilt_masks)
    labels_by_frame: dict[int, np.ndarray] = {}
    for frame_id, masks in masks_by_frame.items():
        if not masks:
            continue
        shape = masks[0].shape
        label = np.zeros(shape, dtype=np.int32)
        frame_nodes = [node for node in nodes if int(node.frame_id) == int(frame_id)]
        for node, mask in zip(frame_nodes, masks):
            label[np.asarray(mask, dtype=bool)] = int(node.node_id) + 1
        labels_by_frame[int(frame_id)] = label
    manifest = {
        "scene": scene,
        "source": source,
        "mode": mode,
        "root": str(root),
        "missing": bool(missing),
        "frame_count": int(len(labels_by_frame)),
        "region_count": int(len(nodes)),
        "uncapped_region_count": int(uncapped_region_count),
        "max_regions_per_scene": int(max_regions),
        "region_cap_applied": bool(capped),
        "min_area": int(min_area),
    }
    return nodes, labels_by_frame, manifest


def _load_tubes(scene: str, args: argparse.Namespace) -> list[TubeRecord]:
    chunks, _ = load_scene_chunks_from_cache(
        Path(args.cache_root) / scene,
        max_tubes_per_window=int(args.max_tubes_per_window),
        image_width=int(args.image_width),
        image_height=int(args.image_height),
    )
    builder = D4RTNativeSceneBuilder(object(), {"model": {"input": {"clip_frames": 32}}}, temporal_chunk_size=32, temporal_chunk_stride=16)
    return chunks_to_records(builder.stitch_to_canonical(chunks))


def _load_gt(scene: str, tubes: list[TubeRecord], args: argparse.Namespace) -> dict[int, int]:
    return assign_gt_labels(
        tubes,
        stream=ScanNetStream(seq_name=scene),
        min_visibility=float(args.min_visibility),
        min_confidence=float(args.min_confidence),
    )


def _visible(tube: TubeRecord, local_idx: int, args: argparse.Namespace) -> bool:
    uv = np.asarray(tube.uv[local_idx], dtype=np.float32)
    return bool(
        np.isfinite(uv).all()
        and 0.0 <= float(uv[0]) <= 1.0
        and 0.0 <= float(uv[1]) <= 1.0
        and float(tube.visibility[local_idx]) >= float(args.min_visibility)
        and float(tube.confidence[local_idx]) >= float(args.min_confidence)
    )


def _collect_observations(
    nodes: list[RegionNode],
    labels_by_frame: dict[int, np.ndarray],
    tubes: list[TubeRecord],
    args: argparse.Namespace,
) -> tuple[dict[int, Counter[int]], dict[int, Counter[int]], dict[int, int]]:
    support_by_region: dict[int, Counter[int]] = defaultdict(Counter)
    support_by_tube: dict[int, Counter[int]] = defaultdict(Counter)
    observation_count_by_tube: Counter[int] = Counter()
    node_lookup = {int(node.node_id): node for node in nodes}
    for tube in tubes:
        frames = np.asarray(tube.target_frames_global, dtype=np.int64)
        for local_idx, frame_id in enumerate(frames.tolist()):
            label = labels_by_frame.get(int(frame_id))
            if label is None or not _visible(tube, local_idx, args):
                continue
            x, y = source_xy_from_uv(tube.uv[local_idx], image_width=label.shape[1], image_height=label.shape[0])
            node_value = int(label[int(y), int(x)]) - 1
            if node_value < 0 or node_value not in node_lookup:
                continue
            support_by_region[node_value][int(tube.tube_id)] += 1
            support_by_tube[int(tube.tube_id)][node_value] += 1
            observation_count_by_tube[int(tube.tube_id)] += 1
    return support_by_region, support_by_tube, {int(k): int(v) for k, v in observation_count_by_tube.items()}


def _component_frames(component: list[int], nodes: list[RegionNode]) -> set[int]:
    return {int(nodes[idx].frame_id) for idx in component}


def _make_components(
    nodes: list[RegionNode],
    support_by_region: dict[int, Counter[int]],
    *,
    variant: str,
    min_shared_tubes: int,
    min_shared_jaccard: float,
    seed: int,
) -> list[list[int]]:
    if variant == "no_temporal":
        return [[idx] for idx in range(len(nodes))]

    region_support = {idx: set(counter.keys()) for idx, counter in support_by_region.items()}
    tube_to_regions: dict[int, list[int]] = defaultdict(list)
    for region, tubes in region_support.items():
        for tube in tubes:
            tube_to_regions[int(tube)].append(int(region))

    pair_counts: Counter[tuple[int, int]] = Counter()
    for regions in tube_to_regions.values():
        regions = sorted(set(regions), key=lambda idx: (int(nodes[idx].frame_id), int(idx)))
        if variant.endswith("_chain"):
            prev = None
            for region in regions:
                if prev is not None and int(nodes[prev].frame_id) != int(nodes[region].frame_id):
                    pair_counts[(min(prev, region), max(prev, region))] += 1
                prev = region
        else:
            if len(regions) > 24:
                regions = regions[:24]
            for pos, left in enumerate(regions):
                for right in regions[pos + 1 :]:
                    if int(nodes[left].frame_id) == int(nodes[right].frame_id):
                        continue
                    pair_counts[(left, right)] += 1

    uf = UnionFind(len(nodes))
    members: dict[int, list[int]] = {idx: [idx] for idx in range(len(nodes))}
    frame_sets: dict[int, set[int]] = {idx: {int(node.frame_id)} for idx, node in enumerate(nodes)}
    edges = []
    for (left, right), shared in pair_counts.items():
        union = len(region_support.get(left, set()) | region_support.get(right, set()))
        jaccard = float(shared / max(union, 1))
        if int(shared) >= int(min_shared_tubes) and jaccard >= float(min_shared_jaccard):
            edges.append((int(shared), float(jaccard), left, right))
    for _, _, left, right in sorted(edges, reverse=True):
        root_left = uf.find(left)
        root_right = uf.find(right)
        if root_left == root_right:
            continue
        if frame_sets[root_left] & frame_sets[root_right]:
            continue
        if uf.union(root_left, root_right):
            new_root = uf.find(root_left)
            old_root = root_right if new_root == root_left else root_left
            members[new_root] = members.get(root_left, []) + members.get(root_right, [])
            frame_sets[new_root] = frame_sets.get(root_left, set()) | frame_sets.get(root_right, set())
            members.pop(old_root, None)
            frame_sets.pop(old_root, None)

    groups: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(nodes)):
        groups[uf.find(idx)].append(idx)
    return list(groups.values())


def _shuffle_supports(
    support_by_region: dict[int, Counter[int]],
    *,
    seed: int,
) -> tuple[dict[int, Counter[int]], dict[int, Counter[int]], dict[int, int]]:
    rng = np.random.default_rng(int(seed))
    all_tubes = sorted({int(tube) for counter in support_by_region.values() for tube in counter})
    if not all_tubes:
        return defaultdict(Counter), defaultdict(Counter), {}
    shuffled_region: dict[int, Counter[int]] = defaultdict(Counter)
    shuffled_tube: dict[int, Counter[int]] = defaultdict(Counter)
    observation_count: Counter[int] = Counter()
    for region, counter in support_by_region.items():
        tubes = list(counter.keys())
        sampled = rng.choice(all_tubes, size=len(tubes), replace=len(tubes) > len(all_tubes))
        for old_tube, new_tube in zip(tubes, sampled.tolist()):
            count = int(counter[int(old_tube)])
            shuffled_region[int(region)][int(new_tube)] += count
            shuffled_tube[int(new_tube)][int(region)] += count
            observation_count[int(new_tube)] += count
    return shuffled_region, shuffled_tube, {int(k): int(v) for k, v in observation_count.items()}


def _stable_seed_offset(*parts: str) -> int:
    text = "::".join(str(part) for part in parts)
    return int(sum((idx + 1) * ord(ch) for idx, ch in enumerate(text)) % 100000)


def _assign_tubes(
    components: list[list[int]],
    support_by_tube: dict[int, Counter[int]],
    observation_count_by_tube: dict[int, int],
    gt_labels: dict[int, int],
    *,
    min_support: int,
    min_fraction: float,
) -> tuple[dict[int, int], float]:
    node_to_component = {}
    for comp_idx, component in enumerate(components):
        for node_id in component:
            node_to_component[int(node_id)] = int(comp_idx)
    labels_pred: dict[int, int] = {}
    unknown_count = 0
    next_unknown = len(components) + 1
    for tube_id, gt in sorted(gt_labels.items()):
        if int(gt) <= 0:
            continue
        comp_counts: Counter[int] = Counter()
        for node_id, count in support_by_tube.get(int(tube_id), Counter()).items():
            comp = node_to_component.get(int(node_id))
            if comp is not None:
                comp_counts[int(comp)] += int(count)
        if not comp_counts:
            labels_pred[int(tube_id)] = next_unknown
            next_unknown += 1
            unknown_count += 1
            continue
        comp, count = comp_counts.most_common(1)[0]
        frac = float(count / max(int(observation_count_by_tube.get(int(tube_id), 0)), 1))
        if int(count) < int(min_support) or frac < float(min_fraction):
            labels_pred[int(tube_id)] = next_unknown
            next_unknown += 1
            unknown_count += 1
        else:
            labels_pred[int(tube_id)] = int(comp)
    labeled = sum(1 for value in gt_labels.values() if int(value) > 0)
    return labels_pred, float(unknown_count / max(labeled, 1))


def _gate(metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "ari_pass": metrics.get("ARI") is not None and float(metrics["ARI"]) >= LOCAL_GATE["ARI"],
        "purity_pass": metrics.get("purity") is not None and float(metrics["purity"]) >= LOCAL_GATE["purity"],
        "completeness_pass": metrics.get("completeness") is not None and float(metrics["completeness"]) >= LOCAL_GATE["completeness"],
        "unknown_pass": metrics.get("unknown_tube_ratio") is not None
        and float(metrics["unknown_tube_ratio"]) <= LOCAL_GATE["unknown_tube_ratio_max"],
        "scene0081_pass": metrics.get("scene0081_ARI") is not None
        and float(metrics["scene0081_ARI"]) >= LOCAL_GATE["scene0081_ARI"],
    }
    checks["pass_3D_gate"] = bool(all(checks.values()))
    return checks


def _evaluate_spec(
    *,
    source: str,
    mode: str,
    scenes: list[str],
    args: argparse.Namespace,
    variant: str,
    min_shared_tubes: int,
    min_shared_jaccard: float,
    min_support: int,
    min_fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scene_rows = []
    all_true: list[int] = []
    all_pred: list[int] = []
    pred_offset = 0
    total_unknown = 0
    total_labeled = 0
    for scene in scenes:
        nodes, labels_by_frame, manifest = _load_masks(Path(args.mask_root), scene, source, mode, int(args.min_region_area))
        tubes = _load_tubes(scene, args)
        gt_labels = _load_gt(scene, tubes, args)
        if not nodes:
            row = {
                **manifest,
                "variant": variant,
                "status": "missing_or_empty_masks",
                "ARI": None,
                "purity": None,
                "completeness": None,
                "unknown_tube_ratio": None,
            }
            scene_rows.append(row)
            continue
        support_by_region, support_by_tube, observation_count_by_tube = _collect_observations(nodes, labels_by_frame, tubes, args)
        if variant.startswith("shuffled_d4rt"):
            support_by_region, support_by_tube, observation_count_by_tube = _shuffle_supports(
                support_by_region,
                seed=int(args.shuffle_seed) + _stable_seed_offset(scene, source, mode),
            )
        components = _make_components(
            nodes,
            support_by_region,
            variant=variant,
            min_shared_tubes=min_shared_tubes,
            min_shared_jaccard=min_shared_jaccard,
            seed=int(args.shuffle_seed),
        )
        labels_pred, unknown_ratio = _assign_tubes(
            components,
            support_by_tube,
            observation_count_by_tube,
            gt_labels,
            min_support=min_support,
            min_fraction=min_fraction,
        )
        metrics = _cluster_metrics(labels_pred, gt_labels)
        labeled_ids = [tid for tid in sorted(labels_pred) if int(gt_labels.get(int(tid), 0)) > 0]
        for tid in labeled_ids:
            all_true.append(int(gt_labels[int(tid)]))
            all_pred.append(int(labels_pred[int(tid)]) + pred_offset)
        pred_offset += len(components) + len(labeled_ids) + 10
        total_labeled += len(labeled_ids)
        total_unknown += int(round(float(unknown_ratio) * max(len(labeled_ids), 1)))
        frame_counts = Counter(int(node.frame_id) for node in nodes)
        same_frame_violations = sum(max(count - 1, 0) for count in frame_counts.values())
        row = {
            **manifest,
            "variant": variant,
            "status": "ok",
            "tube_observed_count": int(len(observation_count_by_tube)),
            "component_count": int(len(components)),
            "regions_per_component_mean": float(np.mean([len(comp) for comp in components])) if components else None,
            "temporal_span_mean": float(
                np.mean([
                    max(_component_frames(comp, nodes)) - min(_component_frames(comp, nodes)) + 1
                    for comp in components
                    if comp
                ])
            )
            if components
            else None,
            "same_frame_cannot_link_violations_region_level": int(same_frame_violations),
            "ARI": metrics.get("ari"),
            "purity": metrics.get("purity"),
            "completeness": metrics.get("completeness"),
            "unknown_tube_ratio": float(unknown_ratio),
            "labeled_tube_count": metrics.get("labeled_tube_count"),
        }
        scene_rows.append(row)
    aggregate_metrics = _cluster_metrics(
        {idx: pred for idx, pred in enumerate(all_pred)},
        {idx: true for idx, true in enumerate(all_true)},
    )
    all_row = {
        "source": source,
        "mode": mode,
        "variant": variant,
        "scene": "ALL",
        "scene_count": int(len(scenes)),
        "ARI": aggregate_metrics.get("ari"),
        "purity": aggregate_metrics.get("purity"),
        "completeness": aggregate_metrics.get("completeness"),
        "unknown_tube_ratio": float(total_unknown / max(total_labeled, 1)),
        "labeled_tube_count": int(total_labeled),
        "scene0081_ARI": next((row.get("ARI") for row in scene_rows if row.get("scene") == "scene0081_01"), None),
    }
    all_row.update(_gate(all_row))
    return scene_rows, all_row


def run(args: argparse.Namespace) -> dict[str, Any]:
    setattr(_load_masks, "max_regions_per_scene", int(args.max_regions_per_scene))
    scenes = _read_split(Path(args.split))
    specs = _parse_specs(args.specs)
    variants_all = [
        ("real_support_chain", 1, 0.0, 1, 0.0),
        ("real_support_chain2", 2, 0.0, 1, 0.0),
        ("real_support_chain_unknown", 2, 0.0, 2, 0.10),
        ("real_support_ultra_loose", 1, 0.0, 1, 0.0),
        ("real_support_broad", 1, 0.01, 1, 0.02),
        ("real_support_loose", 2, 0.02, 1, 0.05),
        ("real_support_strict", 4, 0.05, 2, 0.15),
        ("real_support_unknown", 3, 0.04, 2, 0.25),
        ("no_temporal", 999999, 1.0, 1, 0.05),
        ("shuffled_d4rt_chain", 1, 0.0, 1, 0.0),
        ("shuffled_d4rt", 2, 0.02, 1, 0.05),
    ]
    requested = {item.strip() for item in str(args.variants).split(",") if item.strip()}
    variants = [item for item in variants_all if not requested or item[0] in requested]
    scene_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for source, mode in specs:
        for variant, min_shared, min_jaccard, min_support, min_fraction in variants:
            rows, summary = _evaluate_spec(
                source=source,
                mode=mode,
                scenes=scenes,
                args=args,
                variant=variant,
                min_shared_tubes=min_shared,
                min_shared_jaccard=min_jaccard,
                min_support=min_support,
                min_fraction=min_fraction,
            )
            for row in rows:
                row.update(
                    {
                        "min_shared_tubes": int(min_shared),
                        "min_shared_jaccard": float(min_jaccard),
                        "min_support": int(min_support),
                        "min_fraction": float(min_fraction),
                    }
                )
            summary.update(
                {
                    "min_shared_tubes": int(min_shared),
                    "min_shared_jaccard": float(min_jaccard),
                    "min_support": int(min_support),
                    "min_fraction": float(min_fraction),
                }
            )
            scene_rows.extend(rows)
            summary_rows.append(summary)
    real_rows = [row for row in summary_rows if str(row["variant"]).startswith("real")]
    best_real = max(real_rows, key=lambda row: float(row.get("ARI") if row.get("ARI") is not None else -999.0), default=None)
    controls = {
        "no_temporal": max(
            [row for row in summary_rows if row["variant"] == "no_temporal"],
            key=lambda row: float(row.get("ARI") if row.get("ARI") is not None else -999.0),
            default=None,
        ),
        "shuffled_d4rt": max(
            [row for row in summary_rows if str(row["variant"]).startswith("shuffled_d4rt")],
            key=lambda row: float(row.get("ARI") if row.get("ARI") is not None else -999.0),
            default=None,
        ),
    }
    control_gate = {
        "best_real_key": None if best_real is None else f"{best_real['source']}:{best_real['mode']}:{best_real['variant']}",
        "best_real_ARI": None if best_real is None else best_real.get("ARI"),
        "no_temporal_best_ARI": None if controls["no_temporal"] is None else controls["no_temporal"].get("ARI"),
        "shuffled_best_ARI": None if controls["shuffled_d4rt"] is None else controls["shuffled_d4rt"].get("ARI"),
        "real_minus_no_temporal": None,
        "real_minus_shuffled": None,
        "control_gate_pass": False,
    }
    if best_real is not None and controls["no_temporal"] is not None and best_real.get("ARI") is not None and controls["no_temporal"].get("ARI") is not None:
        control_gate["real_minus_no_temporal"] = float(best_real["ARI"] - controls["no_temporal"]["ARI"])
    if best_real is not None and controls["shuffled_d4rt"] is not None and best_real.get("ARI") is not None and controls["shuffled_d4rt"].get("ARI") is not None:
        control_gate["real_minus_shuffled"] = float(best_real["ARI"] - controls["shuffled_d4rt"]["ARI"])
    control_gate["control_gate_pass"] = bool(
        control_gate["real_minus_no_temporal"] is not None
        and control_gate["real_minus_no_temporal"] >= 0.05
        and control_gate["real_minus_shuffled"] is not None
        and control_gate["real_minus_shuffled"] >= 0.20
    )
    manifest = {
        "plan": "docs/stream4d_v36_masklet_first_object_identity_plan.md",
        "phase": "v36_phaseF_external_downstream_assignment",
        "is_method_result": True,
        "is_diagnostic_only": False,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "uses_rgbd_for_prediction": False,
        "uses_pose_for_prediction": False,
        "uses_scannet_mesh_for_prediction": False,
        "uses_eval_sim3_for_prediction": False,
        "uses_d4rt_self_sim3": True,
        "uses_frozen_visual_backbone": any(source == "dinov2_maskcut" for source, _ in specs),
        "mask_source": ",".join(f"{source}:{mode}" for source, mode in specs),
        "object_identity_source": "external_2d_masks_plus_d4rt_uv_containment",
        "geometry_field": "D4RT uv/visibility/confidence only",
        "coordinate_frame": "image_space_uv_for_assignment",
        "alignment_source": "d4rt_self_sim3_cache_for_tube_records",
        "scenes": scenes,
        "control_gate": control_gate,
        "best_real": best_real,
        "pass_3D_gate": bool(best_real.get("pass_3D_gate")) if best_real else False,
        "allowed_4d": bool(best_real.get("pass_3D_gate") and control_gate["control_gate_pass"]) if best_real else False,
        "allowed_ap": bool(best_real.get("pass_3D_gate") and control_gate["control_gate_pass"]) if best_real else False,
    }
    out_root = Path(args.output_root)
    _write_csv(out_root / "external_downstream_scene_rows.csv", scene_rows)
    _write_json(out_root / "external_downstream_scene_rows.json", scene_rows)
    _write_csv(out_root / "external_downstream_summary.csv", summary_rows)
    _write_json(out_root / "external_downstream_summary.json", summary_rows)
    _write_json(out_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-root", default="outputs/audit/v36_external_mask_source_all")
    parser.add_argument("--specs", default=",".join(DEFAULT_SPECS))
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v21_3_occupancy_d5_warmstart64_probe5_r1")
    parser.add_argument("--output-root", default="outputs/audit/v36_external_downstream_assignment")
    parser.add_argument("--max-tubes-per-window", type=int, default=640)
    parser.add_argument("--image-width", type=int, default=1296)
    parser.add_argument("--image-height", type=int, default=968)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--min-region-area", type=int, default=64)
    parser.add_argument("--max-regions-per-scene", type=int, default=0)
    parser.add_argument("--variants", default="")
    parser.add_argument("--shuffle-seed", type=int, default=3606)
    args = parser.parse_args()
    manifest = run(args)
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
