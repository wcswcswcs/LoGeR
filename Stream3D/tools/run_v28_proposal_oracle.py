from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from math import comb
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stream4d.scannet_stream import ScanNetStream
from stream4d_native.d4rt_scene_builder import D4RTNativeSceneBuilder
from stream4d_native.measurement_bank import MaskMeasurement, build_measurement_bank
from stream4d_native.object_tube_io import TubeRecord
from tools.run_v26_object_quality_diagnostics import _auc, _json_safe, _read_split, _write_csv, assign_gt_labels
from tools.trace_v25_real_geometry_flow import chunks_to_records, load_scene_chunks_from_cache


def _stable_scene_seed(seed: int, scene: str) -> int:
    import hashlib

    digest = hashlib.sha256(f"{int(seed)}:{scene}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _copy_array_or_none(value: np.ndarray | None) -> np.ndarray | None:
    return None if value is None else np.asarray(value).copy()


def _shuffle_d4rt_records_for_control(records: list[TubeRecord], *, seed: int, scene: str) -> list[TubeRecord]:
    """Return a shuffled-D4RT control while keeping tube IDs for GT diagnostics.

    Tube ids stay fixed, but each id receives another tube's method-visible D4RT
    trajectory/geometry. GT labels are computed from the original records.
    """

    if len(records) < 2:
        return list(records)
    order = np.arange(len(records), dtype=np.int64)
    rng = np.random.default_rng(_stable_scene_seed(seed, scene))
    rng.shuffle(order)
    if np.all(order == np.arange(len(records), dtype=np.int64)):
        order = np.roll(order, 1)
    out: list[TubeRecord] = []
    for target, donor_idx in zip(records, order.tolist()):
        donor = records[int(donor_idx)]
        out.append(
            replace(
                target,
                source_frame_global=int(donor.source_frame_global),
                source_xy=tuple(int(v) for v in donor.source_xy),
                source_uv=tuple(float(v) for v in donor.source_uv),
                target_frames_global=_copy_array_or_none(donor.target_frames_global),
                uv=_copy_array_or_none(donor.uv),
                visibility=_copy_array_or_none(donor.visibility),
                confidence=_copy_array_or_none(donor.confidence),
                xyz_local=_copy_array_or_none(donor.xyz_local),
                xyz_ref0=_copy_array_or_none(donor.xyz_ref0),
                xyz_canonical=_copy_array_or_none(donor.xyz_canonical),
                T_chunk_to_canonical=donor.T_chunk_to_canonical,
                alignment_quality=dict(donor.alignment_quality),
                coordinate_frame=str(donor.coordinate_frame),
                scale_status=str(donor.scale_status),
                allow_metric_merge=bool(donor.allow_metric_merge),
                alignment_source=str(donor.alignment_source),
                transform_id=donor.transform_id,
            )
        )
    return out


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    scene: str
    frame_id: int
    mask_id: int
    proposal_type: str
    core_tube_ids: tuple[int, ...]
    fringe_tube_ids: tuple[int, ...]
    boundary_tube_ids: tuple[int, ...]
    region_area: float
    features: dict[str, float | int | None]


def _mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def _quantile(values: list[float], q: float) -> float | None:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.quantile(vals, q)) if vals else None


def _ari(labels_true: list[int], labels_pred: list[int]) -> float | None:
    if len(labels_true) < 2:
        return None
    n = len(labels_true)
    contingency: dict[tuple[int, int], int] = defaultdict(int)
    true_counts: Counter[int] = Counter(labels_true)
    pred_counts: Counter[int] = Counter(labels_pred)
    for true, pred in zip(labels_true, labels_pred):
        contingency[(int(true), int(pred))] += 1
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


def _tube_frame_index(tube: TubeRecord, frame_id: int) -> int | None:
    frames = np.asarray(tube.target_frames_global, dtype=np.int64).tolist()
    try:
        return int(frames.index(int(frame_id)))
    except ValueError:
        return None


def _tube_xy(tube: TubeRecord, frame_id: int, shape: tuple[int, int]) -> tuple[int, int] | None:
    idx = _tube_frame_index(tube, int(frame_id))
    if idx is None:
        return None
    uv = np.asarray(tube.uv, dtype=np.float32)[idx]
    if not (np.isfinite(uv).all() and 0.0 <= float(uv[0]) <= 1.0 and 0.0 <= float(uv[1]) <= 1.0):
        return None
    height, width = int(shape[0]), int(shape[1])
    x = int(np.clip(np.rint(float(uv[0]) * (width - 1)), 0, width - 1))
    y = int(np.clip(np.rint(float(uv[1]) * (height - 1)), 0, height - 1))
    return x, y


def _tube_visible_conf(tube: TubeRecord, frame_id: int) -> tuple[float, float] | None:
    idx = _tube_frame_index(tube, int(frame_id))
    if idx is None:
        return None
    return float(np.asarray(tube.visibility)[idx]), float(np.asarray(tube.confidence)[idx])


def _tube_temporal_length(tube: TubeRecord) -> int:
    visibility = np.asarray(tube.visibility, dtype=np.float32)
    confidence = np.asarray(tube.confidence, dtype=np.float32)
    return int(np.count_nonzero((visibility >= 0.5) & (confidence >= 0.5)))


def _tube_xyz_rep(tube: TubeRecord) -> np.ndarray | None:
    xyz = tube.xyz_canonical if tube.xyz_canonical is not None else tube.xyz_local
    arr = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    if not np.isfinite(arr).any():
        return None
    rep = np.nanmedian(arr, axis=0)
    if not np.isfinite(rep).all():
        return None
    return rep.astype(np.float32)


def _farthest_seed_indices(points: np.ndarray, k: int) -> list[int]:
    if points.shape[0] == 0:
        return []
    k = int(max(1, min(k, points.shape[0])))
    center = np.mean(points, axis=0)
    first = int(np.argmax(np.linalg.norm(points - center[None, :], axis=1)))
    seeds = [first]
    while len(seeds) < k:
        dists = np.min(
            np.stack([np.linalg.norm(points - points[idx][None, :], axis=1) for idx in seeds], axis=0),
            axis=0,
        )
        next_idx = int(np.argmax(dists))
        if next_idx in seeds:
            break
        seeds.append(next_idx)
    return seeds


def _assign_to_seeds(points: np.ndarray, seed_points: np.ndarray) -> np.ndarray:
    dists = np.linalg.norm(points[:, None, :] - seed_points[None, :, :], axis=2)
    return np.argmin(dists, axis=1)


def _clusters_from_features(
    ids: list[int],
    features: np.ndarray,
    *,
    max_clusters: int,
    min_tubes: int,
) -> list[list[int]]:
    if len(ids) < max(int(min_tubes) * 2, 2):
        return []
    features = np.asarray(features, dtype=np.float32)
    valid = np.isfinite(features).all(axis=1)
    if int(np.count_nonzero(valid)) < max(int(min_tubes) * 2, 2):
        return []
    ids_valid = [int(ids[i]) for i in np.where(valid)[0].tolist()]
    feats = features[valid]
    scale = np.nanstd(feats, axis=0)
    scale[scale < 1e-6] = 1.0
    feats = (feats - np.nanmean(feats, axis=0)) / scale
    k = min(int(max_clusters), max(2, int(np.ceil(np.sqrt(len(ids_valid) / max(int(min_tubes), 1))))))
    seeds = _farthest_seed_indices(feats, k)
    if len(seeds) < 2:
        return []
    labels = _assign_to_seeds(feats, feats[seeds])
    clusters: list[list[int]] = []
    for label in sorted(set(labels.tolist())):
        cluster = [ids_valid[idx] for idx, item in enumerate(labels.tolist()) if int(item) == int(label)]
        if len(cluster) >= int(min_tubes):
            clusters.append(sorted(cluster))
    return clusters


def _gradient_map(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb.astype(np.uint8)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _proposal_features(
    *,
    meas: MaskMeasurement,
    core_ids: list[int],
    all_inside: set[int],
    by_id: dict[int, TubeRecord],
    mask: np.ndarray,
    rgb: np.ndarray,
    gradient: np.ndarray,
    repeat_counts: dict[tuple[int, int, int], int],
) -> dict[str, float | int | None]:
    frame_id = int(meas.frame_global)
    distances = [float(meas.mask_distance_to_boundary_per_tube.get(int(tid), 0.0)) for tid in core_ids]
    boundary = set(int(v) for v in meas.boundary_tube_ids)
    boundary_count = sum(1 for tid in core_ids if int(tid) in boundary)
    vis: list[float] = []
    conf: list[float] = []
    xy: list[tuple[int, int]] = []
    rgb_vals: list[np.ndarray] = []
    grad_vals: list[float] = []
    temporal_lengths: list[float] = []
    xyz_reps: list[np.ndarray] = []
    repeat_vals: list[float] = []
    for tid in core_ids:
        tube = by_id.get(int(tid))
        if tube is None:
            continue
        vc = _tube_visible_conf(tube, frame_id)
        if vc is not None:
            vis.append(float(vc[0]))
            conf.append(float(vc[1]))
        xy_item = _tube_xy(tube, frame_id, mask.shape)
        if xy_item is not None:
            x, y = xy_item
            xy.append((x, y))
            rgb_vals.append(rgb[y, x].astype(np.float32) / 255.0)
            grad_vals.append(float(gradient[y, x]))
        temporal_lengths.append(float(_tube_temporal_length(tube)))
        rep = _tube_xyz_rep(tube)
        if rep is not None:
            xyz_reps.append(rep)
        repeat_vals.append(float(repeat_counts.get((frame_id, int(meas.mask_id), int(tid)), 1)))
    area_ratio = float(len(core_ids) / max(len(all_inside), 1))
    rgb_arr = np.stack(rgb_vals, axis=0) if rgb_vals else None
    xyz_arr = np.stack(xyz_reps, axis=0) if xyz_reps else None
    xy_arr = np.asarray(xy, dtype=np.float32) if xy else None
    compactness = None
    if xyz_arr is not None and xyz_arr.shape[0] > 1:
        center = np.mean(xyz_arr, axis=0)
        compactness = float(np.mean(np.linalg.norm(xyz_arr - center[None, :], axis=1)))
    xy_compactness = None
    if xy_arr is not None and xy_arr.shape[0] > 1:
        center2 = np.mean(xy_arr, axis=0)
        xy_compactness = float(np.mean(np.linalg.norm(xy_arr - center2[None, :], axis=1)))
    overlap_ratio = float(len(set(core_ids) & all_inside) / max(len(core_ids), 1))
    return {
        "mask_area": int(meas.mask_area),
        "proposal_area": float(max(float(meas.mask_area) * area_ratio, float(len(core_ids)))),
        "proposal_area_ratio": area_ratio,
        "eroded_interior_ratio": float(
            sum(1 for tid in core_ids if bool(meas.mask_eroded_interior_flag_per_tube.get(int(tid), False)))
            / max(len(core_ids), 1)
        ),
        "boundary_contact_ratio": float(boundary_count / max(len(core_ids), 1)),
        "boundary_risk": float(boundary_count / max(len(core_ids), 1)),
        "core_tube_count": int(len(core_ids)),
        "fringe_tube_count": int(len(all_inside - set(core_ids))),
        "tube_density": float(len(core_ids) / max(float(meas.mask_area), 1.0)),
        "visibility_mean": _mean(vis),
        "confidence_mean": _mean(conf),
        "tube_temporal_length_mean": _mean(temporal_lengths),
        "tube_canonical_compactness": compactness,
        "tube_xy_compactness": xy_compactness,
        "appearance_variance": float(np.mean(np.var(rgb_arr, axis=0))) if rgb_arr is not None and rgb_arr.shape[0] > 1 else 0.0,
        "image_gradient_boundary_score": _mean(grad_vals),
        "mask_distance_mean": _mean(distances),
        "mask_distance_p10": _quantile(distances, 0.10),
        "mask_distance_p50": _quantile(distances, 0.50),
        "mask_distance_p90": _quantile(distances, 0.90),
        "visible_outside_negative_rate": float(len(meas.outside_visible_tube_ids) / max(len(core_ids), 1)),
        "same_frame_cannot_link_rate": float(
            len(meas.same_frame_different_mask_cannot_link_pairs) / max(len(core_ids), 1)
        ),
        "mask_temporal_repeat_score": _mean(repeat_vals),
        "overlap_with_other_proposals": overlap_ratio,
    }


def _make_proposal(
    *,
    scene: str,
    meas: MaskMeasurement,
    proposal_type: str,
    index: int,
    core_ids: list[int],
    all_inside: set[int],
    by_id: dict[int, TubeRecord],
    mask: np.ndarray,
    rgb: np.ndarray,
    gradient: np.ndarray,
    repeat_counts: dict[tuple[int, int, int], int],
) -> Proposal | None:
    core = tuple(sorted(set(int(v) for v in core_ids)))
    if not core:
        return None
    boundary = tuple(sorted(set(int(v) for v in meas.boundary_tube_ids) & set(core)))
    fringe = tuple(sorted(set(all_inside) - set(core)))
    features = _proposal_features(
        meas=meas,
        core_ids=list(core),
        all_inside=all_inside,
        by_id=by_id,
        mask=mask,
        rgb=rgb,
        gradient=gradient,
        repeat_counts=repeat_counts,
    )
    return Proposal(
        proposal_id=f"{scene}_f{int(meas.frame_global):06d}_m{int(meas.mask_id):04d}_{proposal_type}_{index:03d}",
        scene=scene,
        frame_id=int(meas.frame_global),
        mask_id=int(meas.mask_id),
        proposal_type=proposal_type,
        core_tube_ids=core,
        fringe_tube_ids=fringe,
        boundary_tube_ids=boundary,
        region_area=float(features.get("proposal_area") or 0.0),
        features=features,
    )


def _repeat_counts(measurements: list[MaskMeasurement], window: int) -> dict[tuple[int, int, int], int]:
    by_mask_tube: dict[tuple[int, int], list[int]] = defaultdict(list)
    for meas in measurements:
        for tid in meas.inside_tube_ids:
            by_mask_tube[(int(meas.mask_id), int(tid))].append(int(meas.frame_global))
    out: dict[tuple[int, int, int], int] = {}
    for meas in measurements:
        for tid in meas.inside_tube_ids:
            frames = by_mask_tube.get((int(meas.mask_id), int(tid)), [])
            count = sum(1 for frame in frames if abs(int(frame) - int(meas.frame_global)) <= int(window))
            out[(int(meas.frame_global), int(meas.mask_id), int(tid))] = int(count)
    return out


def _temporal_track_components(
    measurements: list[MaskMeasurement],
    *,
    min_tubes: int,
    min_shared_tubes: int,
    max_frame_gap: int,
    min_overlap_ratio: float,
) -> list[list[MaskMeasurement]]:
    """Link mask measurements into GT-free temporal tube-overlap tracks."""

    nodes = [item for item in measurements if len(set(int(v) for v in item.inside_tube_ids)) >= int(min_tubes)]
    if len(nodes) < 2:
        return []
    tube_sets = [set(int(v) for v in item.inside_tube_ids) for item in nodes]
    parent = list(range(len(nodes)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for i in range(len(nodes)):
        frame_i = int(nodes[i].frame_global)
        set_i = tube_sets[i]
        for j in range(i + 1, len(nodes)):
            if int(max_frame_gap) >= 0 and abs(frame_i - int(nodes[j].frame_global)) > int(max_frame_gap):
                continue
            set_j = tube_sets[j]
            overlap = len(set_i & set_j)
            if overlap < int(min_shared_tubes):
                continue
            overlap_ratio = float(overlap / max(min(len(set_i), len(set_j)), 1))
            if overlap_ratio >= float(min_overlap_ratio):
                union(i, j)

    grouped: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(nodes)):
        grouped[find(idx)].append(idx)
    components: list[list[MaskMeasurement]] = []
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        union_ids: set[int] = set()
        for idx in indices:
            union_ids.update(tube_sets[idx])
        if len(union_ids) >= int(min_tubes):
            components.append([nodes[idx] for idx in indices])
    return components


def _add_temporal_track_union_proposals(
    *,
    scene: str,
    proposals: list[Proposal],
    measurements: list[MaskMeasurement],
    by_id: dict[int, TubeRecord],
    masks: dict[int, np.ndarray],
    rgbs: dict[int, np.ndarray],
    gradients: dict[int, np.ndarray],
    repeat: dict[tuple[int, int, int], int],
    min_tubes: int,
    min_shared_tubes: int,
    max_frame_gap: int,
    overlap_thresholds: tuple[float, ...],
) -> None:
    seen_unions: set[tuple[int, ...]] = set()
    for threshold in overlap_thresholds:
        components = _temporal_track_components(
            measurements,
            min_tubes=int(min_tubes),
            min_shared_tubes=int(min_shared_tubes),
            max_frame_gap=int(max_frame_gap),
            min_overlap_ratio=float(threshold),
        )
        for comp_idx, component in enumerate(components):
            union_ids = tuple(sorted({int(tid) for meas in component for tid in meas.inside_tube_ids if int(tid) in by_id}))
            if len(union_ids) < int(min_tubes) or union_ids in seen_unions:
                continue
            seen_unions.add(union_ids)
            representative = max(component, key=lambda item: len(set(int(v) for v in item.inside_tube_ids) & set(union_ids)))
            frame_id = int(representative.frame_global)
            proposal = _make_proposal(
                scene=scene,
                meas=representative,
                proposal_type=f"R8_temporal_tube_overlap_track_union_t{int(round(threshold * 100)):02d}",
                index=comp_idx,
                core_ids=list(union_ids),
                all_inside=set(int(v) for v in representative.inside_tube_ids),
                by_id=by_id,
                mask=masks[frame_id],
                rgb=rgbs[frame_id],
                gradient=gradients[frame_id],
                repeat_counts=repeat,
            )
            if proposal is not None:
                proposals.append(proposal)


def _temporal_track_consensus_core_ids(
    component: list[MaskMeasurement],
    *,
    min_vote_ratio: float,
    min_vote_count: int,
) -> tuple[int, ...]:
    votes: Counter[int] = Counter()
    for meas in component:
        votes.update(set(int(tid) for tid in meas.inside_tube_ids))
    required = max(int(min_vote_count), int(np.ceil(float(min_vote_ratio) * max(len(component), 1))))
    return tuple(sorted(tid for tid, count in votes.items() if int(count) >= required))


def _add_temporal_track_consensus_proposals(
    *,
    scene: str,
    proposals: list[Proposal],
    measurements: list[MaskMeasurement],
    by_id: dict[int, TubeRecord],
    masks: dict[int, np.ndarray],
    rgbs: dict[int, np.ndarray],
    gradients: dict[int, np.ndarray],
    repeat: dict[tuple[int, int, int], int],
    min_tubes: int,
    min_shared_tubes: int,
    max_frame_gap: int,
    overlap_thresholds: tuple[float, ...],
    consensus_ratios: tuple[float, ...],
) -> None:
    seen_cores: set[tuple[int, ...]] = set()
    for threshold in overlap_thresholds:
        components = _temporal_track_components(
            measurements,
            min_tubes=int(min_tubes),
            min_shared_tubes=int(min_shared_tubes),
            max_frame_gap=int(max_frame_gap),
            min_overlap_ratio=float(threshold),
        )
        for ratio in consensus_ratios:
            for comp_idx, component in enumerate(components):
                core_ids = tuple(
                    tid
                    for tid in _temporal_track_consensus_core_ids(
                        component,
                        min_vote_ratio=float(ratio),
                        min_vote_count=2,
                    )
                    if int(tid) in by_id
                )
                if len(core_ids) < int(min_tubes) or core_ids in seen_cores:
                    continue
                seen_cores.add(core_ids)
                core_set = set(core_ids)
                representative = max(
                    component,
                    key=lambda item: len(set(int(v) for v in item.inside_tube_ids) & core_set),
                )
                frame_id = int(representative.frame_global)
                proposal = _make_proposal(
                    scene=scene,
                    meas=representative,
                    proposal_type=(
                        f"R9_temporal_tube_overlap_track_consensus_"
                        f"t{int(round(threshold * 100)):02d}_v{int(round(ratio * 100)):02d}"
                    ),
                    index=comp_idx,
                    core_ids=list(core_ids),
                    all_inside=set(int(v) for v in representative.inside_tube_ids),
                    by_id=by_id,
                    mask=masks[frame_id],
                    rgb=rgbs[frame_id],
                    gradient=gradients[frame_id],
                    repeat_counts=repeat,
                )
                if proposal is not None:
                    proposals.append(proposal)


def _visible_negative_pruned_core_ids(union_ids: tuple[int, ...], meas: MaskMeasurement) -> tuple[int, ...]:
    outside_visible = set(int(tid) for tid in meas.outside_visible_tube_ids)
    return tuple(sorted(int(tid) for tid in union_ids if int(tid) not in outside_visible))


def _is_temporal_proposal_type(proposal_type: str) -> bool:
    return str(proposal_type).startswith(
        (
            "R8_temporal_tube_overlap_track_union",
            "R9_temporal_tube_overlap_track_consensus",
            "R10_temporal_tube_overlap_visible_negative_pruned",
            "R11_temporal_visible_negative_pruned_canonical_split",
            "R12_temporal_visible_negative_eroded_pruned",
        )
    )


def _filter_temporal_proposals(
    proposals: list[Proposal],
    *,
    max_cannot_link_rate: float | None,
) -> list[Proposal]:
    if max_cannot_link_rate is None or not np.isfinite(float(max_cannot_link_rate)):
        return proposals
    out: list[Proposal] = []
    for proposal in proposals:
        if not _is_temporal_proposal_type(proposal.proposal_type):
            out.append(proposal)
            continue
        value = proposal.features.get("same_frame_cannot_link_rate")
        try:
            rate = float(value if value is not None else 0.0)
        except (TypeError, ValueError):
            rate = 0.0
        if rate <= float(max_cannot_link_rate):
            out.append(proposal)
    return out


def _visible_negative_eroded_core_ids(union_ids: tuple[int, ...], meas: MaskMeasurement) -> tuple[int, ...]:
    outside_visible = set(int(tid) for tid in meas.outside_visible_tube_ids)
    inside = set(int(tid) for tid in meas.inside_tube_ids)
    eroded = {
        int(tid)
        for tid, flag in meas.mask_eroded_interior_flag_per_tube.items()
        if bool(flag)
    }
    return tuple(
        sorted(
            int(tid)
            for tid in union_ids
            if int(tid) not in outside_visible and (int(tid) not in inside or int(tid) in eroded)
        )
    )


def _canonical_clusters_for_core(
    core_ids: tuple[int, ...],
    by_id: dict[int, TubeRecord],
    *,
    max_clusters: int,
    min_tubes: int,
) -> list[list[int]]:
    ids: list[int] = []
    reps: list[np.ndarray] = []
    for tid in core_ids:
        tube = by_id.get(int(tid))
        rep = _tube_xyz_rep(tube) if tube is not None else None
        if rep is None:
            continue
        ids.append(int(tid))
        reps.append(rep)
    if len(ids) < max(int(min_tubes) * 2, 2):
        return []
    return _clusters_from_features(
        ids,
        np.stack(reps, axis=0),
        max_clusters=int(max_clusters),
        min_tubes=int(min_tubes),
    )


def _add_temporal_track_negative_pruned_proposals(
    *,
    scene: str,
    proposals: list[Proposal],
    measurements: list[MaskMeasurement],
    by_id: dict[int, TubeRecord],
    masks: dict[int, np.ndarray],
    rgbs: dict[int, np.ndarray],
    gradients: dict[int, np.ndarray],
    repeat: dict[tuple[int, int, int], int],
    min_tubes: int,
    min_shared_tubes: int,
    max_frame_gap: int,
    overlap_thresholds: tuple[float, ...],
    max_split_clusters: int,
    enable_eroded_prune: bool,
) -> None:
    seen_cores: set[tuple[int, ...]] = set()
    for threshold in overlap_thresholds:
        components = _temporal_track_components(
            measurements,
            min_tubes=int(min_tubes),
            min_shared_tubes=int(min_shared_tubes),
            max_frame_gap=int(max_frame_gap),
            min_overlap_ratio=float(threshold),
        )
        for comp_idx, component in enumerate(components):
            union_ids = tuple(sorted({int(tid) for meas in component for tid in meas.inside_tube_ids if int(tid) in by_id}))
            if len(union_ids) < int(min_tubes):
                continue
            for meas_idx, meas in enumerate(component):
                core_ids = _visible_negative_pruned_core_ids(union_ids, meas)
                if len(core_ids) < int(min_tubes) or core_ids == union_ids or core_ids in seen_cores:
                    continue
                seen_cores.add(core_ids)
                frame_id = int(meas.frame_global)
                proposal = _make_proposal(
                    scene=scene,
                    meas=meas,
                    proposal_type=f"R10_temporal_tube_overlap_visible_negative_pruned_t{int(round(threshold * 100)):02d}",
                    index=comp_idx * 1000 + meas_idx,
                    core_ids=list(core_ids),
                    all_inside=set(int(v) for v in meas.inside_tube_ids),
                    by_id=by_id,
                    mask=masks[frame_id],
                    rgb=rgbs[frame_id],
                    gradient=gradients[frame_id],
                    repeat_counts=repeat,
                )
                if proposal is not None:
                    proposals.append(proposal)
                if enable_eroded_prune:
                    eroded_core_ids = _visible_negative_eroded_core_ids(union_ids, meas)
                    if (
                        len(eroded_core_ids) >= int(min_tubes)
                        and eroded_core_ids != core_ids
                        and eroded_core_ids != union_ids
                        and eroded_core_ids not in seen_cores
                    ):
                        seen_cores.add(eroded_core_ids)
                        eroded_proposal = _make_proposal(
                            scene=scene,
                            meas=meas,
                            proposal_type=f"R12_temporal_visible_negative_eroded_pruned_t{int(round(threshold * 100)):02d}",
                            index=comp_idx * 1000 + meas_idx,
                            core_ids=list(eroded_core_ids),
                            all_inside=set(int(v) for v in meas.inside_tube_ids),
                            by_id=by_id,
                            mask=masks[frame_id],
                            rgb=rgbs[frame_id],
                            gradient=gradients[frame_id],
                            repeat_counts=repeat,
                        )
                        if eroded_proposal is not None:
                            proposals.append(eroded_proposal)
                if int(max_split_clusters) <= 1:
                    continue
                for cluster_idx, cluster in enumerate(
                    _canonical_clusters_for_core(
                        core_ids,
                        by_id,
                        max_clusters=int(max_split_clusters),
                        min_tubes=int(min_tubes),
                    )
                ):
                    cluster_core = tuple(sorted(int(v) for v in cluster))
                    if len(cluster_core) < int(min_tubes) or cluster_core == core_ids or cluster_core in seen_cores:
                        continue
                    seen_cores.add(cluster_core)
                    split_proposal = _make_proposal(
                        scene=scene,
                        meas=meas,
                        proposal_type=(
                            f"R11_temporal_visible_negative_pruned_canonical_split_"
                            f"t{int(round(threshold * 100)):02d}"
                        ),
                        index=comp_idx * 100000 + meas_idx * 100 + cluster_idx,
                        core_ids=list(cluster_core),
                        all_inside=set(int(v) for v in meas.inside_tube_ids),
                        by_id=by_id,
                        mask=masks[frame_id],
                        rgb=rgbs[frame_id],
                        gradient=gradients[frame_id],
                        repeat_counts=repeat,
                    )
                    if split_proposal is not None:
                        proposals.append(split_proposal)


def generate_proposals_for_scene(
    *,
    scene: str,
    records: list[TubeRecord],
    measurements: list[MaskMeasurement],
    stream: ScanNetStream,
    min_tubes: int,
    max_clusters: int,
    temporal_window: int,
    temporal_track_window: int,
    temporal_track_min_shared_tubes: int,
    temporal_track_overlap_thresholds: tuple[float, ...],
    temporal_track_consensus_ratios: tuple[float, ...],
    max_temporal_split_clusters: int,
    max_temporal_cannot_link_rate: float | None,
    enable_temporal_eroded_prune: bool,
) -> list[Proposal]:
    by_id = {int(tube.tube_id): tube for tube in records}
    frame_ids = sorted({int(meas.frame_global) for meas in measurements})
    masks = {frame_id: stream.load_mask(frame_id) for frame_id in frame_ids}
    rgbs = {frame_id: stream.load_rgb(frame_id) for frame_id in frame_ids}
    gradients = {frame_id: _gradient_map(rgbs[frame_id]) for frame_id in frame_ids}
    repeat = _repeat_counts(measurements, int(temporal_window))
    by_mask_id: dict[int, list[MaskMeasurement]] = defaultdict(list)
    for item in measurements:
        by_mask_id[int(item.mask_id)].append(item)
    proposals: list[Proposal] = []
    for meas in measurements:
        frame_id = int(meas.frame_global)
        mask = masks[frame_id]
        rgb = rgbs[frame_id]
        gradient = gradients[frame_id]
        all_inside = set(int(v) for v in meas.inside_tube_ids)
        if len(all_inside) < int(min_tubes):
            continue
        local_index = 0

        def add(proposal_type: str, ids: list[int], *, allow_temporal_union: bool = False) -> None:
            nonlocal local_index
            if allow_temporal_union:
                valid_tube_ids = set(by_id)
                ids = sorted(set(int(v) for v in ids if int(v) in valid_tube_ids))
            else:
                ids = sorted(set(int(v) for v in ids if int(v) in all_inside))
            if len(ids) < int(min_tubes):
                return
            proposal = _make_proposal(
                scene=scene,
                meas=meas,
                proposal_type=proposal_type,
                index=local_index,
                core_ids=ids,
                all_inside=all_inside,
                by_id=by_id,
                mask=mask,
                rgb=rgb,
                gradient=gradient,
                repeat_counts=repeat,
            )
            local_index += 1
            if proposal is not None:
                proposals.append(proposal)

        full_ids = sorted(all_inside)
        add("R0_full_mask_region", full_ids)
        eroded_ids = [
            int(tid)
            for tid in full_ids
            if bool(meas.mask_eroded_interior_flag_per_tube.get(int(tid), False))
        ]
        add("R1_boundary_eroded_interior", eroded_ids)

        xy_rows: list[list[float]] = []
        valid_ids: list[int] = []
        for tid in full_ids:
            tube = by_id.get(int(tid))
            xy = _tube_xy(tube, frame_id, mask.shape) if tube is not None else None
            if xy is None:
                continue
            dist = float(meas.mask_distance_to_boundary_per_tube.get(int(tid), 0.0))
            valid_ids.append(int(tid))
            xy_rows.append([float(xy[0]), float(xy[1]), dist])
        if valid_ids:
            xy_features = np.asarray(xy_rows, dtype=np.float32)
            for cluster in _clusters_from_features(
                valid_ids,
                xy_features,
                max_clusters=int(max_clusters),
                min_tubes=int(min_tubes),
            ):
                add("R2_distance_watershed_region", cluster)
            for cluster in _clusters_from_features(
                valid_ids,
                xy_features[:, :2],
                max_clusters=int(max_clusters),
                min_tubes=int(min_tubes),
            ):
                add("R3_d4rt_tube_seeded_voronoi", cluster)

            color_features: list[list[float]] = []
            color_ids: list[int] = []
            for tid in valid_ids:
                tube = by_id.get(int(tid))
                xy = _tube_xy(tube, frame_id, mask.shape) if tube is not None else None
                if xy is None:
                    continue
                x, y = xy
                color = rgb[y, x].astype(np.float32) / 255.0
                color_ids.append(int(tid))
                color_features.append(
                    [
                        float(x) / max(mask.shape[1], 1),
                        float(y) / max(mask.shape[0], 1),
                        float(color[0]),
                        float(color[1]),
                        float(color[2]),
                        float(gradient[y, x]) / 255.0,
                    ]
                )
            if color_ids:
                for cluster in _clusters_from_features(
                    color_ids,
                    np.asarray(color_features, dtype=np.float32),
                    max_clusters=min(int(max_clusters), 3),
                    min_tubes=int(min_tubes),
                ):
                    add("R4_image_gradient_split", cluster)

        xyz_ids: list[int] = []
        xyz_features: list[np.ndarray] = []
        for tid in full_ids:
            tube = by_id.get(int(tid))
            rep = _tube_xyz_rep(tube) if tube is not None else None
            if rep is not None:
                xyz_ids.append(int(tid))
                xyz_features.append(rep)
        if xyz_ids:
            for cluster in _clusters_from_features(
                xyz_ids,
                np.stack(xyz_features, axis=0),
                max_clusters=int(max_clusters),
                min_tubes=int(min_tubes),
            ):
                add("R5_d4rt_canonical_adjacency_split", cluster)

        consensus_ids = [tid for tid in full_ids if repeat.get((frame_id, int(meas.mask_id), int(tid)), 1) >= 2]
        add("R6_mask_overlap_consensus_region", consensus_ids)
        union_ids: set[int] = set()
        for neighbor in by_mask_id.get(int(meas.mask_id), []):
            if abs(int(neighbor.frame_global) - int(frame_id)) <= int(temporal_window):
                union_ids.update(int(v) for v in neighbor.inside_tube_ids)
        add("R6_mask_overlap_consensus_union", sorted(union_ids), allow_temporal_union=True)

        core_ids: list[int] = []
        for tid in eroded_ids:
            tube = by_id.get(int(tid))
            vc = _tube_visible_conf(tube, frame_id) if tube is not None else None
            if vc is not None and float(vc[0]) >= 0.7 and float(vc[1]) >= 0.7:
                core_ids.append(int(tid))
        add("R7_high_purity_core_region", core_ids)
    _add_temporal_track_union_proposals(
        scene=scene,
        proposals=proposals,
        measurements=measurements,
        by_id=by_id,
        masks=masks,
        rgbs=rgbs,
        gradients=gradients,
        repeat=repeat,
        min_tubes=int(min_tubes),
        min_shared_tubes=int(temporal_track_min_shared_tubes),
        max_frame_gap=int(temporal_track_window),
        overlap_thresholds=tuple(float(v) for v in temporal_track_overlap_thresholds),
    )
    _add_temporal_track_consensus_proposals(
        scene=scene,
        proposals=proposals,
        measurements=measurements,
        by_id=by_id,
        masks=masks,
        rgbs=rgbs,
        gradients=gradients,
        repeat=repeat,
        min_tubes=int(min_tubes),
        min_shared_tubes=int(temporal_track_min_shared_tubes),
        max_frame_gap=int(temporal_track_window),
        overlap_thresholds=tuple(float(v) for v in temporal_track_overlap_thresholds),
        consensus_ratios=tuple(float(v) for v in temporal_track_consensus_ratios),
    )
    _add_temporal_track_negative_pruned_proposals(
        scene=scene,
        proposals=proposals,
        measurements=measurements,
        by_id=by_id,
        masks=masks,
        rgbs=rgbs,
        gradients=gradients,
        repeat=repeat,
        min_tubes=int(min_tubes),
        min_shared_tubes=int(temporal_track_min_shared_tubes),
        max_frame_gap=int(temporal_track_window),
        overlap_thresholds=tuple(float(v) for v in temporal_track_overlap_thresholds),
        max_split_clusters=int(max_temporal_split_clusters),
        enable_eroded_prune=bool(enable_temporal_eroded_prune),
    )
    proposals = _filter_temporal_proposals(
        proposals,
        max_cannot_link_rate=max_temporal_cannot_link_rate,
    )
    return _dedupe_proposals(proposals)


def _dedupe_proposals(proposals: list[Proposal]) -> list[Proposal]:
    seen: set[tuple[str, int, int, str, tuple[int, ...]]] = set()
    out: list[Proposal] = []
    for proposal in proposals:
        key = (
            proposal.scene,
            int(proposal.frame_id),
            int(proposal.mask_id),
            proposal.proposal_type,
            proposal.core_tube_ids,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(proposal)
    return out


def _proposal_diag(
    proposal: Proposal,
    gt_labels: dict[int, int],
    gt_counts: Counter[int],
) -> dict[str, Any]:
    counts: Counter[int] = Counter()
    labeled = 0
    for tid in proposal.core_tube_ids:
        gt = int(gt_labels.get(int(tid), 0))
        if gt <= 0:
            continue
        counts[gt] += 1
        labeled += 1
    if counts:
        best_gt, overlap = counts.most_common(1)[0]
        best_total = int(gt_counts.get(int(best_gt), 0))
        purity = float(overlap / max(labeled, 1))
        completeness = float(overlap / max(best_total, 1))
        iou = float(overlap / max(labeled + best_total - overlap, 1))
    else:
        best_gt, purity, completeness, iou = 0, None, None, None
    return {
        "proposal_purity": purity,
        "proposal_completeness": completeness,
        "proposal_best_GT": int(best_gt),
        "proposal_best_IoU": iou,
        "proposal_labeled_tube_count": int(labeled),
        "_gt_overlap_counts": {int(gt): int(count) for gt, count in counts.items()},
        "_proposal_labeled_tube_count": int(labeled),
    }


def _cluster_metrics(labels_pred: dict[int, int], gt_labels: dict[int, int]) -> dict[str, Any]:
    labeled = [tube_id for tube_id in sorted(labels_pred) if int(gt_labels.get(int(tube_id), 0)) > 0]
    true = [int(gt_labels[tube_id]) for tube_id in labeled]
    pred = [int(labels_pred[tube_id]) for tube_id in labeled]
    comp_to_labels: dict[int, Counter[int]] = defaultdict(Counter)
    gt_to_comps: dict[int, set[int]] = defaultdict(set)
    for tube_id in labeled:
        comp = int(labels_pred[tube_id])
        gt = int(gt_labels[tube_id])
        comp_to_labels[comp][gt] += 1
        gt_to_comps[gt].add(comp)
    purity_num = sum(max(counts.values()) for counts in comp_to_labels.values() if counts)
    completeness_num = 0
    for gt, comps in gt_to_comps.items():
        completeness_num += max(comp_to_labels[comp].get(gt, 0) for comp in comps)
    return {
        "ari": _ari(true, pred),
        "purity": float(purity_num / max(len(labeled), 1)),
        "completeness": float(completeness_num / max(len(labeled), 1)),
        "overmerge": int(sum(1 for counts in comp_to_labels.values() if len(counts) > 1)),
        "oversplit": int(sum(1 for comps in gt_to_comps.values() if len(comps) > 1)),
        "labeled_tube_count": int(len(labeled)),
    }


def _pool_members(proposals: list[dict[str, Any]], pool: str) -> list[dict[str, Any]]:
    mapping = {
        "O0_full_mask": {"R0_full_mask_region"},
        "O1_eroded": {"R1_boundary_eroded_interior"},
        "O2_watershed": {"R2_distance_watershed_region"},
        "O3_d4rt_tube_seeded": {"R3_d4rt_tube_seeded_voronoi"},
        "O4_image_gradient": {"R4_image_gradient_split"},
        "O5_hybrid": {
            "R0_full_mask_region",
            "R1_boundary_eroded_interior",
            "R2_distance_watershed_region",
            "R3_d4rt_tube_seeded_voronoi",
            "R4_image_gradient_split",
            "R5_d4rt_canonical_adjacency_split",
            "R6_mask_overlap_consensus_region",
            "R6_mask_overlap_consensus_union",
            "R7_high_purity_core_region",
            "R8_temporal_tube_overlap_track_union_t20",
            "R8_temporal_tube_overlap_track_union_t35",
            "R8_temporal_tube_overlap_track_union_t50",
            "R8_temporal_tube_overlap_track_union_t70",
        },
    }
    if pool == "O5_hybrid":
        return [
            row
            for row in proposals
            if row["proposal_type"] in mapping[pool]
            or str(row["proposal_type"]).startswith("R8_temporal_tube_overlap_track_union")
            or str(row["proposal_type"]).startswith("R9_temporal_tube_overlap_track_consensus")
            or str(row["proposal_type"]).startswith("R10_temporal_tube_overlap_visible_negative_pruned")
            or str(row["proposal_type"]).startswith("R11_temporal_visible_negative_pruned_canonical_split")
            or str(row["proposal_type"]).startswith("R12_temporal_visible_negative_eroded_pruned")
        ]
    return [row for row in proposals if row["proposal_type"] in mapping[pool]]


def _oracle_summary_for_pool(
    *,
    scene: str,
    pool: str,
    proposal_rows: list[dict[str, Any]],
    gt_labels: dict[int, int],
    gt_counts: Counter[int],
) -> dict[str, Any]:
    rows = _pool_members(proposal_rows, pool) if not pool.startswith("O6_") else proposal_rows
    labeled_tubes = sorted(tid for tid, gt in gt_labels.items() if int(gt) > 0)
    per_gt_best: dict[int, tuple[float, dict[str, Any] | None]] = {}
    for gt in sorted(gt_counts):
        best_iou = -1.0
        best_row: dict[str, Any] | None = None
        for row in rows:
            overlap_counts = row.get("_gt_overlap_counts")
            if isinstance(overlap_counts, dict):
                overlap = int(overlap_counts.get(int(gt), overlap_counts.get(str(int(gt)), 0)))
                labeled = int(row.get("_proposal_labeled_tube_count", row.get("proposal_labeled_tube_count", 0)) or 0)
            else:
                tube_ids = set(int(v) for v in row["_core_tube_ids"])
                overlap = sum(1 for tid in tube_ids if int(gt_labels.get(int(tid), 0)) == int(gt))
                labeled = sum(1 for tid in tube_ids if int(gt_labels.get(int(tid), 0)) > 0)
            iou = float(overlap / max(labeled + int(gt_counts[gt]) - overlap, 1))
            if iou > best_iou:
                best_iou = iou
                best_row = row
        per_gt_best[int(gt)] = (float(max(best_iou, 0.0)), best_row)
    selected: dict[str, dict[str, Any]] = {}
    for _, row in per_gt_best.values():
        if row is not None:
            selected[str(row["proposal_id"])] = row
    labels_pred: dict[int, int] = {}
    selected_items = list(selected.values())
    for idx, row in enumerate(selected_items):
        for tid in row["_core_tube_ids"]:
            tid = int(tid)
            if tid not in labels_pred:
                labels_pred[tid] = idx
    next_label = len(selected_items)
    for tid in labeled_tubes:
        if tid not in labels_pred:
            labels_pred[int(tid)] = next_label
            next_label += 1
    metrics = _cluster_metrics(labels_pred, gt_labels)
    best_ious = [score for score, _ in per_gt_best.values()]
    return {
        "scene": scene,
        "pool": pool,
        "proposal_count": int(len(rows)),
        "oracle_ARI": metrics["ari"],
        "oracle_purity": metrics["purity"],
        "oracle_completeness": metrics["completeness"],
        "oracle_overmerge": metrics["overmerge"],
        "oracle_oversplit": metrics["oversplit"],
        "oracle_per_GT_best_IoU_mean": _mean(best_ious),
        "GT_count": int(len(gt_counts)),
        "covered_GT_count": int(sum(1 for score in best_ious if score > 0.0)),
        "GT_with_best_IoU_ge_025": float(sum(1 for score in best_ious if score >= 0.25) / max(len(best_ious), 1)),
        "GT_with_best_IoU_ge_050": float(sum(1 for score in best_ious if score >= 0.50) / max(len(best_ious), 1)),
        "is_diagnostic_only": True,
    }


def _aggregate_oracle(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pool[str(row["pool"])].append(row)
    for pool, items in sorted(by_pool.items()):
        all_row = {
            "scene": "ALL",
            "pool": pool,
            "proposal_count": int(sum(int(r["proposal_count"]) for r in items)),
            "oracle_ARI": _mean([float(r["oracle_ARI"]) for r in items if r.get("oracle_ARI") is not None]),
            "oracle_purity": _mean([float(r["oracle_purity"]) for r in items if r.get("oracle_purity") is not None]),
            "oracle_completeness": _mean(
                [float(r["oracle_completeness"]) for r in items if r.get("oracle_completeness") is not None]
            ),
            "oracle_overmerge": _mean([float(r["oracle_overmerge"]) for r in items]),
            "oracle_oversplit": _mean([float(r["oracle_oversplit"]) for r in items]),
            "oracle_per_GT_best_IoU_mean": _mean(
                [float(r["oracle_per_GT_best_IoU_mean"]) for r in items if r.get("oracle_per_GT_best_IoU_mean") is not None]
            ),
            "GT_count": int(sum(int(r["GT_count"]) for r in items)),
            "covered_GT_count": int(sum(int(r["covered_GT_count"]) for r in items)),
            "GT_with_best_IoU_ge_025": _mean([float(r["GT_with_best_IoU_ge_025"]) for r in items]),
            "GT_with_best_IoU_ge_050": _mean([float(r["GT_with_best_IoU_ge_050"]) for r in items]),
            "scene_count": int(len(items)),
            "scene0081_oracle_ARI": next(
                (r.get("oracle_ARI") for r in items if str(r.get("scene")) == "scene0081_01"),
                None,
            ),
            "is_diagnostic_only": True,
        }
        out.append(all_row)
    return out


def _pool_summary(proposal_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pool in ["O0_full_mask", "O1_eroded", "O2_watershed", "O3_d4rt_tube_seeded", "O4_image_gradient", "O5_hybrid"]:
        rows = _pool_members(proposal_rows, pool)
        tube_sets = [tuple(sorted(int(v) for v in row["_core_tube_ids"])) for row in rows]
        duplicate_count = len(tube_sets) - len(set(tube_sets))
        out.append(
            {
                "pool": pool,
                "proposal_count": int(len(rows)),
                "proposal_count_per_mask": None,
                "mean_proposal_area": _mean([float(r["region_area"]) for r in rows]),
                "mean_core_tubes": _mean([float(r["num_core_tubes"]) for r in rows]),
                "proposal_purity_mean": _mean(
                    [float(r["proposal_purity"]) for r in rows if r.get("proposal_purity") is not None]
                ),
                "proposal_purity_p10": _quantile(
                    [float(r["proposal_purity"]) for r in rows if r.get("proposal_purity") is not None],
                    0.10,
                ),
                "proposal_completeness_mean": _mean(
                    [float(r["proposal_completeness"]) for r in rows if r.get("proposal_completeness") is not None]
                ),
                "proposal_best_IoU_mean": _mean(
                    [float(r["proposal_best_IoU"]) for r in rows if r.get("proposal_best_IoU") is not None]
                ),
                "proposal_best_IoU_p50": _quantile(
                    [float(r["proposal_best_IoU"]) for r in rows if r.get("proposal_best_IoU") is not None],
                    0.50,
                ),
                "proposal_best_IoU_p90": _quantile(
                    [float(r["proposal_best_IoU"]) for r in rows if r.get("proposal_best_IoU") is not None],
                    0.90,
                ),
                "duplicate_proposal_rate": float(duplicate_count / max(len(rows), 1)),
                "overlap_between_proposals": None,
                "is_diagnostic_only": True,
            }
        )
    return out


def _feature_auc_rows(proposal_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    feature_names = [
        "mask_area",
        "proposal_area",
        "proposal_area_ratio",
        "eroded_interior_ratio",
        "boundary_contact_ratio",
        "boundary_risk",
        "core_tube_count",
        "fringe_tube_count",
        "tube_density",
        "visibility_mean",
        "confidence_mean",
        "tube_temporal_length_mean",
        "tube_canonical_compactness",
        "tube_xy_compactness",
        "appearance_variance",
        "image_gradient_boundary_score",
        "visible_outside_negative_rate",
        "same_frame_cannot_link_rate",
        "mask_temporal_repeat_score",
    ]
    out: list[dict[str, Any]] = []
    scenes = sorted(set(str(row["scene"]) for row in proposal_rows))
    for feature in feature_names:
        values: list[float] = []
        labels_purity: list[int] = []
        labels_iou: list[int] = []
        labels_false: list[int] = []
        labels_complete: list[int] = []
        for row in proposal_rows:
            value = row.get(feature)
            if value is None:
                continue
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(score) or row.get("proposal_purity") is None or row.get("proposal_best_IoU") is None:
                continue
            values.append(score)
            purity = float(row.get("proposal_purity") or 0.0)
            iou = float(row.get("proposal_best_IoU") or 0.0)
            completeness = float(row.get("proposal_completeness") or 0.0)
            labels_purity.append(int(purity >= 0.85 and iou >= 0.25))
            labels_iou.append(int(iou >= 0.25))
            labels_false.append(int(purity <= 0.60))
            labels_complete.append(int(completeness >= 0.50))
        def auc_oriented(labels: list[int]) -> float | None:
            if not values:
                return None
            raw = _auc(np.asarray(labels, dtype=np.int64), np.asarray(values, dtype=np.float64))
            if raw is None:
                return None
            return float(max(raw, 1.0 - raw))

        scene_good_scores: dict[str, float] = {}
        scene_false_scores: dict[str, float] = {}
        for scene in scenes:
            sub = [row for row in proposal_rows if str(row["scene"]) == scene]
            sub_values: list[float] = []
            sub_good_labels: list[int] = []
            sub_false_labels: list[int] = []
            for row in sub:
                value = row.get(feature)
                if value is None or row.get("proposal_purity") is None or row.get("proposal_best_IoU") is None:
                    continue
                try:
                    score = float(value)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(score):
                    continue
                sub_values.append(score)
                purity = float(row.get("proposal_purity") or 0.0)
                iou = float(row.get("proposal_best_IoU") or 0.0)
                sub_good_labels.append(int(purity >= 0.85 and iou >= 0.25))
                sub_false_labels.append(int(purity <= 0.60))
            if sub_values:
                values_arr = np.asarray(sub_values, dtype=np.float64)
                raw_good = _auc(np.asarray(sub_good_labels, dtype=np.int64), values_arr)
                raw_false = _auc(np.asarray(sub_false_labels, dtype=np.int64), values_arr)
                if raw_good is not None:
                    scene_good_scores[scene] = float(max(raw_good, 1.0 - raw_good))
                if raw_false is not None:
                    scene_false_scores[scene] = float(max(raw_false, 1.0 - raw_false))
        good_stability = int(sum(1 for v in scene_good_scores.values() if v >= 0.60))
        false_stability = int(sum(1 for v in scene_false_scores.values() if v >= 0.60))
        out.append(
            {
                "feature": feature,
                "purity_AUC": auc_oriented(labels_purity),
                "IoU_AUC": auc_oriented(labels_iou),
                "false_merge_AUC": auc_oriented(labels_false),
                "completeness_AUC": auc_oriented(labels_complete),
                "scene0081_AUC": scene_good_scores.get("scene0081_01"),
                "scene0081_false_merge_AUC": scene_false_scores.get("scene0081_01"),
                "good_feature_stability_across_scenes": good_stability,
                "false_merge_feature_stability_across_scenes": false_stability,
                "feature_stability_across_scenes": int(max(good_stability, false_stability)),
                "scene_count_with_auc": int(max(len(scene_good_scores), len(scene_false_scores))),
                "is_diagnostic_only": True,
            }
        )
    return out


def _write_figures(output_root: Path, proposal_rows: list[dict[str, Any]], oracle_rows: list[dict[str, Any]], feature_rows: list[dict[str, Any]]) -> None:
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    all_oracle = [row for row in oracle_rows if row["scene"] == "ALL"]
    if all_oracle:
        pools = [str(row["pool"]) for row in all_oracle]
        ari = [float(row["oracle_ARI"] or 0.0) for row in all_oracle]
        purity = [float(row["oracle_purity"] or 0.0) for row in all_oracle]
        completeness = [float(row["oracle_completeness"] or 0.0) for row in all_oracle]
        x = np.arange(len(pools))
        plt.figure(figsize=(9, 4))
        plt.plot(x, ari, marker="o", label="ARI")
        plt.plot(x, purity, marker="o", label="purity")
        plt.plot(x, completeness, marker="o", label="completeness")
        plt.xticks(x, pools, rotation=35, ha="right", fontsize=7)
        plt.ylim(0.0, 1.02)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "proposal_oracle_table.png", dpi=150)
        plt.close()

    top = sorted(
        [row for row in feature_rows if row.get("purity_AUC") is not None],
        key=lambda row: float(row.get("purity_AUC") or 0.0),
        reverse=True,
    )[:12]
    if top:
        plt.figure(figsize=(9, 4))
        plt.bar(range(len(top)), [float(row["purity_AUC"]) for row in top])
        plt.xticks(range(len(top)), [str(row["feature"]) for row in top], rotation=35, ha="right", fontsize=7)
        plt.ylabel("oriented AUC")
        plt.tight_layout()
        plt.savefig(fig_dir / "proposal_feature_auc.png", dpi=150)
        plt.close()


def _write_report(output_root: Path, label: str, oracle_rows: list[dict[str, Any]], feature_rows: list[dict[str, Any]]) -> None:
    all_rows = {str(row["pool"]): row for row in oracle_rows if row["scene"] == "ALL"}
    o5 = all_rows.get("O5_hybrid", {})
    top_features = sorted(
        [row for row in feature_rows if row.get("purity_AUC") is not None],
        key=lambda row: float(row.get("purity_AUC") or 0.0),
        reverse=True,
    )[:5]
    lines = [
        f"# {label} proposal oracle report",
        "",
        "This report is diagnostic-only. GT labels are used only for oracle metrics and feature labels.",
        "",
        "## O5 hybrid oracle",
        "",
        f"- oracle_ARI: {o5.get('oracle_ARI')}",
        f"- oracle_purity: {o5.get('oracle_purity')}",
        f"- oracle_completeness: {o5.get('oracle_completeness')}",
        f"- GT_with_best_IoU_ge_025: {o5.get('GT_with_best_IoU_ge_025')}",
        f"- scene0081_oracle_ARI: {o5.get('scene0081_oracle_ARI')}",
        "",
        "## Top feature AUC",
        "",
    ]
    for row in top_features:
        lines.append(
            f"- {row['feature']}: purity_AUC={row.get('purity_AUC')}, false_merge_AUC={row.get('false_merge_AUC')}, scene0081_AUC={row.get('scene0081_AUC')}"
        )
    lines.append("")
    (output_root / f"{label}_report.md").write_text("\n".join(lines), encoding="utf-8")


def _proposal_row(proposal: Proposal, diag: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "proposal_id": proposal.proposal_id,
        "scene": proposal.scene,
        "frame_id": int(proposal.frame_id),
        "mask_id": int(proposal.mask_id),
        "proposal_type": proposal.proposal_type,
        "region_area": float(proposal.region_area),
        "num_core_tubes": int(len(proposal.core_tube_ids)),
        "num_fringe_tubes": int(len(proposal.fringe_tube_ids)),
        "num_boundary_tubes": int(len(proposal.boundary_tube_ids)),
        "core_tube_ids": ";".join(str(v) for v in proposal.core_tube_ids),
        "fringe_tube_ids": ";".join(str(v) for v in proposal.fringe_tube_ids[:64]),
        "boundary_tube_ids": ";".join(str(v) for v in proposal.boundary_tube_ids[:64]),
        "_core_tube_ids": list(proposal.core_tube_ids),
        "is_diagnostic_only": True,
    }
    row.update(proposal.features)
    row.update(diag)
    return row


def _gt_oracle_rows(scene: str, gt_labels: dict[int, int], gt_counts: Counter[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gt in sorted(gt_counts):
        tube_ids = sorted(tid for tid, label in gt_labels.items() if int(label) == int(gt))
        if not tube_ids:
            continue
        rows.append(
            {
                "proposal_id": f"{scene}_GT_{int(gt):04d}",
                "scene": scene,
                "frame_id": -1,
                "mask_id": -1,
                "proposal_type": "O6_gt_oracle_upper_bound_forbidden",
                "region_area": float(len(tube_ids)),
                "num_core_tubes": int(len(tube_ids)),
                "num_fringe_tubes": 0,
                "num_boundary_tubes": 0,
                "core_tube_ids": ";".join(str(v) for v in tube_ids),
                "fringe_tube_ids": "",
                "boundary_tube_ids": "",
                "_core_tube_ids": tube_ids,
                "proposal_purity": 1.0,
                "proposal_completeness": 1.0,
                "proposal_best_GT": int(gt),
                "proposal_best_IoU": 1.0,
                "proposal_labeled_tube_count": int(len(tube_ids)),
                "is_diagnostic_only": True,
                "forbidden_for_method_table": True,
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    scenes = _read_split(Path(args.split))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    all_proposal_rows: list[dict[str, Any]] = []
    all_oracle_scene_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    for scene in scenes:
        if not (Path(args.cache_root) / scene).exists():
            continue
        if args.debug_progress:
            print(f"[v28_proposal_oracle] loading {scene}", flush=True)
        chunks, _ = load_scene_chunks_from_cache(
            Path(args.cache_root) / scene,
            max_tubes_per_window=int(args.max_tubes_per_window),
            image_width=int(args.image_width),
            image_height=int(args.image_height),
        )
        builder = D4RTNativeSceneBuilder(object(), {"model": {"input": {"clip_frames": 32}}}, temporal_chunk_size=32, temporal_chunk_stride=16)
        records_real = chunks_to_records(builder.stitch_to_canonical(chunks))
        records = (
            _shuffle_d4rt_records_for_control(records_real, seed=int(args.shuffle_d4rt_seed), scene=scene)
            if bool(args.shuffle_d4rt_control)
            else records_real
        )
        stream = ScanNetStream(seq_name=scene)
        frame_ids = sorted({int(v) for tube in records for v in np.asarray(tube.target_frames_global, dtype=np.int64).tolist()})
        masks_by_frame = {frame_id: stream.load_mask(frame_id) for frame_id in frame_ids}
        measurements, meas_diag = build_measurement_bank(
            records,
            masks_by_frame=masks_by_frame,
            min_visibility=float(args.min_visibility),
            min_confidence=float(args.min_confidence),
        )
        gt_labels = assign_gt_labels(
            records_real,
            stream=stream,
            min_visibility=float(args.min_visibility),
            min_confidence=float(args.min_confidence),
        )
        gt_counts = Counter(int(v) for v in gt_labels.values() if int(v) > 0)
        proposals = generate_proposals_for_scene(
            scene=scene,
            records=records,
            measurements=measurements,
            stream=stream,
            min_tubes=int(args.min_tubes_per_proposal),
            max_clusters=int(args.max_clusters_per_mask),
            temporal_window=int(args.temporal_consensus_window),
            temporal_track_window=int(args.temporal_track_window),
            temporal_track_min_shared_tubes=int(args.temporal_track_min_shared_tubes),
            temporal_track_overlap_thresholds=_parse_thresholds(str(args.temporal_track_overlap_thresholds)),
            temporal_track_consensus_ratios=_parse_thresholds(str(args.temporal_track_consensus_ratios)),
            max_temporal_split_clusters=int(args.max_temporal_split_clusters),
            max_temporal_cannot_link_rate=float(args.max_temporal_cannot_link_rate),
            enable_temporal_eroded_prune=bool(args.enable_temporal_eroded_prune),
        )
        proposal_rows = [_proposal_row(p, _proposal_diag(p, gt_labels, gt_counts)) for p in proposals]
        gt_rows = _gt_oracle_rows(scene, gt_labels, gt_counts)
        all_proposal_rows.extend(proposal_rows)
        for pool in ["O0_full_mask", "O1_eroded", "O2_watershed", "O3_d4rt_tube_seeded", "O4_image_gradient", "O5_hybrid"]:
            all_oracle_scene_rows.append(
                _oracle_summary_for_pool(
                    scene=scene,
                    pool=pool,
                    proposal_rows=proposal_rows,
                    gt_labels=gt_labels,
                    gt_counts=gt_counts,
                )
            )
        all_oracle_scene_rows.append(
            _oracle_summary_for_pool(
                scene=scene,
                pool="O6_gt_oracle_upper_bound_forbidden",
                proposal_rows=gt_rows,
                gt_labels=gt_labels,
                gt_counts=gt_counts,
            )
        )
        scene_rows.append(
            {
                "scene": scene,
                "tube_count": int(len(records_real)),
                "method_tube_count": int(len(records)),
                "labeled_tube_count": int(sum(1 for v in gt_labels.values() if int(v) > 0)),
                "gt_count": int(len(gt_counts)),
                "measurement_count": int(len(measurements)),
                "proposal_count": int(len(proposal_rows)),
                "same_mask_pair_count": int(meas_diag.get("same_mask_pair_count", 0)),
                "is_diagnostic_only": True,
            }
        )

    oracle_rows = all_oracle_scene_rows + _aggregate_oracle(all_oracle_scene_rows)
    pool_rows = _pool_summary(all_proposal_rows)
    feature_rows = _feature_auc_rows(all_proposal_rows)

    label = str(args.label)
    _write_csv(output_root / f"{label}_proposal_rows.csv", [{k: v for k, v in row.items() if not k.startswith("_")} for row in all_proposal_rows])
    (output_root / f"{label}_proposal_rows.json").write_text(
        json.dumps(_json_safe(all_proposal_rows), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for suffix, rows in [
        ("pool_summary", pool_rows),
        ("oracle_summary", oracle_rows),
        ("feature_summary", feature_rows),
        ("scene_rows", scene_rows),
    ]:
        _write_csv(output_root / f"{label}_{suffix}.csv", rows)
        (output_root / f"{label}_{suffix}.json").write_text(
            json.dumps(_json_safe(rows), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    manifest = {
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "uses_rgbd_for_prediction": False,
        "uses_pose_for_prediction": False,
        "uses_scannet_mesh_for_prediction": False,
        "uses_eval_sim3_for_prediction": False,
        "uses_d4rt_self_sim3": True,
        "geometry_field": "D4RT canonical tubes plus image-space uv/visibility/confidence",
        "coordinate_frame": "d4rt_canonical for tube features; image space for mask region features",
        "alignment_source": "D4RT self-Sim3 from cached tube records",
        "cache_root": str(args.cache_root),
        "split": str(args.split),
        "scene_count": int(len(scene_rows)),
        "proposal_count": int(len(all_proposal_rows)),
        "phase_b_complete": True,
        "phase_c_feature_summary_written": True,
        "gt_oracle_upper_bound_forbidden": True,
        "shuffle_d4rt_control": bool(args.shuffle_d4rt_control),
        "shuffle_d4rt_seed": int(args.shuffle_d4rt_seed),
        "shuffle_d4rt_control_semantics": (
            "tube ids kept fixed for GT diagnostics; method-visible D4RT uv/visibility/confidence/xyz trajectories "
            "are deterministically permuted across tubes within each scene"
            if bool(args.shuffle_d4rt_control)
            else ""
        ),
    }
    (output_root / f"{label}_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_figures(output_root, all_proposal_rows, oracle_rows, feature_rows)
    _write_report(output_root, label, oracle_rows, feature_rows)
    return {
        "manifest": manifest,
        "oracle_all": [row for row in oracle_rows if row["scene"] == "ALL"],
        "top_features": sorted(
            feature_rows,
            key=lambda row: float(row.get("purity_AUC") or 0.0),
            reverse=True,
        )[:5],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run v28 mask-region proposal oracle and feature diagnostics.")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--label", default="v28_proposal_oracle")
    parser.add_argument("--max-tubes-per-window", type=int, default=160)
    parser.add_argument("--image-width", type=int, default=1296)
    parser.add_argument("--image-height", type=int, default=968)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--min-tubes-per-proposal", type=int, default=3)
    parser.add_argument("--max-clusters-per-mask", type=int, default=4)
    parser.add_argument("--temporal-consensus-window", type=int, default=2)
    parser.add_argument("--temporal-track-window", type=int, default=16)
    parser.add_argument("--temporal-track-min-shared-tubes", type=int, default=3)
    parser.add_argument("--temporal-track-overlap-thresholds", default="0.35,0.50,0.70")
    parser.add_argument("--temporal-track-consensus-ratios", default="0.50,0.67,1.00")
    parser.add_argument("--max-temporal-split-clusters", type=int, default=0)
    parser.add_argument("--max-temporal-cannot-link-rate", type=float, default=float("inf"))
    parser.add_argument("--enable-temporal-eroded-prune", action="store_true")
    parser.add_argument("--shuffle-d4rt-control", action="store_true")
    parser.add_argument("--shuffle-d4rt-seed", type=int, default=2808)
    parser.add_argument("--debug-progress", action="store_true")
    return parser


def _parse_thresholds(text: str) -> tuple[float, ...]:
    values: list[float] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if not (0.0 < value <= 1.0):
            raise ValueError(f"temporal track overlap threshold out of range: {item}")
        values.append(value)
    if not values:
        raise ValueError("at least one temporal track overlap threshold is required")
    return tuple(values)


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
