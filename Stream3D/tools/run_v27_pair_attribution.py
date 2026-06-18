from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stream4d.scannet_stream import ScanNetStream
from stream4d_native.d4rt_scene_builder import D4RTNativeSceneBuilder
from stream4d_native.measurement_bank import MaskMeasurement, build_measurement_bank, count_pair_measurement_evidence
from stream4d_native.object_tube_io import MergeGeometryError, TubeRecord
from stream4d_native.signed_tube_graph import build_signed_tube_graph
from tools.run_v26_object_quality_diagnostics import (
    _auc,
    _json_safe,
    _read_split,
    _write_csv,
    assign_gt_labels,
)
from tools.trace_v25_real_geometry_flow import chunks_to_records, load_scene_chunks_from_cache


def _mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def _quantile(values: list[float], q: float) -> float | None:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.quantile(vals, q)) if vals else None


def _top_precision(labels: list[int], scores: list[float], positive_value: int) -> float | None:
    pairs = [(int(label), float(score)) for label, score in zip(labels, scores) if np.isfinite(float(score))]
    if not pairs:
        return None
    pairs.sort(key=lambda item: item[1], reverse=True)
    k = max(1, int(np.ceil(0.10 * len(pairs))))
    top = pairs[:k]
    return float(sum(1 for label, _ in top if label == int(positive_value)) / max(len(top), 1))


def _frame_index_map(tube: TubeRecord) -> dict[int, int]:
    return {int(frame): idx for idx, frame in enumerate(np.asarray(tube.target_frames_global, dtype=np.int64).tolist())}


def _common_frame_indices(left: TubeRecord, right: TubeRecord) -> list[tuple[int, int, int]]:
    left_map = _frame_index_map(left)
    right_map = _frame_index_map(right)
    frames = sorted(set(left_map) & set(right_map))
    return [(int(frame), int(left_map[frame]), int(right_map[frame])) for frame in frames]


def _motion_consistency(left: TubeRecord, right: TubeRecord) -> float | None:
    common = _common_frame_indices(left, right)
    if len(common) < 2:
        return None
    xyz_left = np.asarray(left.xyz_canonical if left.xyz_canonical is not None else left.xyz_local, dtype=np.float32)
    xyz_right = np.asarray(right.xyz_canonical if right.xyz_canonical is not None else right.xyz_local, dtype=np.float32)
    sims: list[float] = []
    for (_, li0, ri0), (_, li1, ri1) in zip(common[:-1], common[1:]):
        vl = xyz_left[li1] - xyz_left[li0]
        vr = xyz_right[ri1] - xyz_right[ri0]
        nl = float(np.linalg.norm(vl))
        nr = float(np.linalg.norm(vr))
        if nl <= 1e-6 or nr <= 1e-6:
            continue
        sims.append(float(np.dot(vl, vr) / (nl * nr)))
    return _mean(sims)


def _appearance_feature(stream: ScanNetStream, tube: TubeRecord, rgb_cache: dict[int, np.ndarray]) -> np.ndarray | None:
    frames = np.asarray(tube.target_frames_global, dtype=np.int64)
    uv = np.asarray(tube.uv, dtype=np.float32)
    visibility = np.asarray(tube.visibility, dtype=np.float32)
    confidence = np.asarray(tube.confidence, dtype=np.float32)
    candidates = np.where((visibility >= 0.5) & (confidence >= 0.5) & np.isfinite(uv).all(axis=1))[0]
    if candidates.size == 0:
        return None
    idx = int(candidates[0])
    frame_id = int(frames[idx])
    if frame_id not in rgb_cache:
        try:
            rgb_cache[frame_id] = stream.load_rgb(frame_id)
        except FileNotFoundError:
            return None
    image = rgb_cache[frame_id]
    height, width = image.shape[:2]
    x = int(np.clip(np.rint(float(uv[idx, 0]) * (width - 1)), 0, width - 1))
    y = int(np.clip(np.rint(float(uv[idx, 1]) * (height - 1)), 0, height - 1))
    r = 3
    patch = image[max(0, y - r) : min(height, y + r + 1), max(0, x - r) : min(width, x + r + 1)]
    if patch.size == 0:
        return None
    patch_f = patch.astype(np.float32) / 255.0
    return np.concatenate([patch_f.reshape(-1, 3).mean(axis=0), patch_f.reshape(-1, 3).std(axis=0)]).astype(np.float32)


def _cosine(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None:
        return None
    nl = float(np.linalg.norm(left))
    nr = float(np.linalg.norm(right))
    if nl <= 1e-8 or nr <= 1e-8:
        return None
    return float(np.dot(left, right) / (nl * nr))


def _pair_row(
    *,
    scene: str,
    category: str,
    left: TubeRecord,
    right: TubeRecord,
    gt_labels: dict[int, int],
    threshold: float,
    pair_evidence: dict[tuple[int, int], dict[str, int]],
    stream: ScanNetStream,
    rgb_cache: dict[int, np.ndarray],
    appearance_cache: dict[int, np.ndarray | None],
    same_frame_cannot_link_count: int = 0,
    compute_appearance: bool = False,
) -> dict[str, Any]:
    pair = tuple(sorted((int(left.tube_id), int(right.tube_id))))
    evidence = pair_evidence.get(pair, {})
    distance: float | None = None
    score: float | None = None
    distance_normalized: float | None = None
    predicted_merge = False
    guard_pass = False
    guard_reason = "not_attempted"
    geometry_field = None
    try:
        geom_l, geom_r, guard = left.get_geometry_for_merge(right, "v27_pair_attribution", merge_type="diagnostic_pair")
        rep_l = np.nanmedian(np.asarray(geom_l, dtype=np.float32).reshape(-1, 3), axis=0)
        rep_r = np.nanmedian(np.asarray(geom_r, dtype=np.float32).reshape(-1, 3), axis=0)
        if np.isfinite(rep_l).all() and np.isfinite(rep_r).all():
            distance = float(np.linalg.norm(rep_l - rep_r))
            distance_normalized = float(distance / max(float(threshold), 1e-6))
            score = float(max(0.0, 1.0 - distance_normalized))
            predicted_merge = bool(distance <= float(threshold))
        guard_pass = bool(guard.get("guard_pass", False))
        guard_reason = str(guard.get("guard_reason", "unknown"))
        geometry_field = guard.get("geometry_field_used")
    except MergeGeometryError as exc:
        guard_reason = str(exc)

    label_i = int(gt_labels.get(int(left.tube_id), 0))
    label_j = int(gt_labels.get(int(right.tube_id), 0))
    gt_labeled = bool(label_i > 0 and label_j > 0)
    same_gt = bool(gt_labeled and label_i == label_j)
    different_gt = bool(gt_labeled and label_i != label_j)
    appearance_similarity = None
    if compute_appearance:
        if int(left.tube_id) not in appearance_cache:
            appearance_cache[int(left.tube_id)] = _appearance_feature(stream, left, rgb_cache)
        if int(right.tube_id) not in appearance_cache:
            appearance_cache[int(right.tube_id)] = _appearance_feature(stream, right, rgb_cache)
        appearance_similarity = _cosine(appearance_cache.get(int(left.tube_id)), appearance_cache.get(int(right.tube_id)))
    motion_consistency = _motion_consistency(left, right)
    same_mask_count = int(evidence.get("same_mask_count", 0))
    boundary_safe_count = int(evidence.get("boundary_safe_count", 0))
    boundary_cross_count = int(evidence.get("boundary_cross_count", 0))
    visible_outside_count = int(evidence.get("visible_outside_negative_count", 0))
    cannot_link_count = max(int(same_frame_cannot_link_count), int(evidence.get("same_frame_cannot_link_count", 0)))
    cut_score = float(
        boundary_cross_count + visible_outside_count + cannot_link_count - (score if score is not None else 0.0)
    )
    return {
        "scene": scene,
        "category": category,
        "tube_i": int(left.tube_id),
        "tube_j": int(right.tube_id),
        "chunk_i": int(left.chunk_id),
        "chunk_j": int(right.chunk_id),
        "submap_i": int(left.submap_id),
        "submap_j": int(right.submap_id),
        "same_chunk": bool(int(left.chunk_id) == int(right.chunk_id)),
        "same_submap": bool(int(left.submap_id) == int(right.submap_id)),
        "common_frame_count": int(len(_common_frame_indices(left, right))),
        "gt_i": label_i,
        "gt_j": label_j,
        "gt_labeled_pair": gt_labeled,
        "same_gt": same_gt,
        "different_gt": different_gt,
        "guard_pass": guard_pass,
        "guard_reason": guard_reason,
        "geometry_field": geometry_field,
        "distance": distance,
        "distance_normalized": distance_normalized,
        "merge_score": score,
        "cut_score": cut_score,
        "predicted_merge": predicted_merge,
        "same_mask_count": same_mask_count,
        "mask_cooccurrence_count": same_mask_count,
        "boundary_safe_count": boundary_safe_count,
        "boundary_cross_count": boundary_cross_count,
        "visible_outside_count": visible_outside_count,
        "same_frame_cannot_link_count": cannot_link_count,
        "appearance_similarity": appearance_similarity,
        "motion_consistency": motion_consistency,
        "alignment_uncertainty": None,
        "is_diagnostic_only": True,
    }


def _limited_append_pair(
    pairs: list[tuple[int, int]],
    seen: set[tuple[int, int]],
    left: int,
    right: int,
    max_pairs: int,
) -> None:
    if len(pairs) >= int(max_pairs):
        return
    pair = tuple(sorted((int(left), int(right))))
    if pair[0] == pair[1] or pair in seen:
        return
    seen.add(pair)
    pairs.append(pair)


def _different_mask_pairs(
    measurements: list[MaskMeasurement],
    *,
    max_pairs: int,
) -> list[tuple[int, int]]:
    by_frame: dict[int, list[MaskMeasurement]] = defaultdict(list)
    for meas in measurements:
        by_frame[int(meas.frame_global)].append(meas)
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for frame_id in sorted(by_frame):
        frame_measurements = sorted(by_frame[frame_id], key=lambda m: int(m.mask_id))
        for idx, left in enumerate(frame_measurements):
            left_ids = sorted(set(int(v) for v in left.inside_tube_ids))
            if not left_ids:
                continue
            for right in frame_measurements[idx + 1 :]:
                right_ids = sorted(set(int(v) for v in right.inside_tube_ids))
                if not right_ids:
                    continue
                for li in left_ids[:16]:
                    for rj in right_ids[:16]:
                        _limited_append_pair(pairs, seen, li, rj, max_pairs)
                        if len(pairs) >= int(max_pairs):
                            return pairs
    return pairs


def _same_chunk_sample_pairs(tubes: list[TubeRecord], *, max_pairs: int) -> list[tuple[int, int]]:
    by_chunk: dict[int, list[int]] = defaultdict(list)
    frame_sets = {int(t.tube_id): set(np.asarray(t.target_frames_global, dtype=np.int64).tolist()) for t in tubes}
    for tube in tubes:
        by_chunk[int(tube.chunk_id)].append(int(tube.tube_id))
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for chunk_id in sorted(by_chunk):
        ids = sorted(by_chunk[chunk_id])
        for idx, left in enumerate(ids):
            for right in ids[idx + 1 :]:
                if frame_sets[left] & frame_sets[right]:
                    continue
                _limited_append_pair(pairs, seen, left, right, max_pairs)
                if len(pairs) >= int(max_pairs):
                    return pairs
    return pairs


def _rows_from_pairs(
    *,
    scene: str,
    category: str,
    pairs: list[tuple[int, int]],
    by_id: dict[int, TubeRecord],
    gt_labels: dict[int, int],
    threshold: float,
    pair_evidence: dict[tuple[int, int], dict[str, int]],
    stream: ScanNetStream,
    rgb_cache: dict[int, np.ndarray],
    appearance_cache: dict[int, np.ndarray | None],
    cannot_link: bool = False,
    compute_appearance: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left_id, right_id in pairs:
        left = by_id.get(int(left_id))
        right = by_id.get(int(right_id))
        if left is None or right is None:
            continue
        rows.append(
            _pair_row(
                scene=scene,
                category=category,
                left=left,
                right=right,
                gt_labels=gt_labels,
                threshold=threshold,
                pair_evidence=pair_evidence,
                stream=stream,
                rgb_cache=rgb_cache,
                appearance_cache=appearance_cache,
                same_frame_cannot_link_count=1 if cannot_link else 0,
                compute_appearance=bool(compute_appearance),
            )
        )
    return rows


def _summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(str(row["scene"]), str(row["category"]))].append(row)
        by_key[("ALL", str(row["category"]))].append(row)
    for (scene, category), items in sorted(by_key.items()):
        labeled = [r for r in items if bool(r.get("gt_labeled_pair", False))]
        same = [r for r in labeled if bool(r.get("same_gt", False))]
        diff = [r for r in labeled if bool(r.get("different_gt", False))]
        pred = [r for r in labeled if bool(r.get("predicted_merge", False))]
        false_merge = [r for r in pred if bool(r.get("different_gt", False))]
        false_cut = [r for r in same if not bool(r.get("predicted_merge", False))]
        merge_labels = [1 if bool(r.get("same_gt", False)) else 0 for r in labeled]
        cut_labels = [1 if bool(r.get("different_gt", False)) else 0 for r in labeled]
        merge_scores = [float(r["merge_score"]) if r.get("merge_score") is not None else float("nan") for r in labeled]
        cut_scores = [float(r["cut_score"]) if r.get("cut_score") is not None else float("nan") for r in labeled]
        same_pred = [r for r in same if bool(r.get("predicted_merge", False))]
        diff_pred = [r for r in diff if bool(r.get("predicted_merge", False))]
        distances = [float(r["distance_normalized"]) for r in items if r.get("distance_normalized") is not None]
        out.append(
            {
                "scene": scene,
                "category": category,
                "pair_count": int(len(items)),
                "gt_labeled_pair_count": int(len(labeled)),
                "same_GT_ratio": float(len(same) / max(len(labeled), 1)),
                "different_GT_ratio": float(len(diff) / max(len(labeled), 1)),
                "predicted_merge_count": int(len(pred)),
                "false_merge_count": int(len(false_merge)),
                "false_merge_rate": float(len(false_merge) / max(len(pred), 1)),
                "false_cut_count": int(len(false_cut)),
                "false_cut_rate": float(len(false_cut) / max(len(same), 1)),
                "mean_distance_normalized": _mean(distances),
                "distance_p10": _quantile(distances, 0.10),
                "distance_p50": _quantile(distances, 0.50),
                "distance_p90": _quantile(distances, 0.90),
                "mask_cooccurrence_count": int(sum(int(r.get("mask_cooccurrence_count", 0)) for r in items)),
                "boundary_safe_count": int(sum(int(r.get("boundary_safe_count", 0)) for r in items)),
                "boundary_cross_count": int(sum(int(r.get("boundary_cross_count", 0)) for r in items)),
                "visible_outside_count": int(sum(int(r.get("visible_outside_count", 0)) for r in items)),
                "same_frame_cannot_link_count": int(sum(int(r.get("same_frame_cannot_link_count", 0)) for r in items)),
                "appearance_similarity_mean": _mean(
                    [float(r["appearance_similarity"]) for r in items if r.get("appearance_similarity") is not None]
                ),
                "motion_consistency_mean": _mean(
                    [float(r["motion_consistency"]) for r in items if r.get("motion_consistency") is not None]
                ),
                "alignment_uncertainty_mean": _mean(
                    [float(r["alignment_uncertainty"]) for r in items if r.get("alignment_uncertainty") is not None]
                ),
                "merge_score_AUC": _auc(np.asarray(merge_labels, dtype=np.int64), np.asarray(merge_scores, dtype=np.float64))
                if labeled
                else None,
                "cut_score_AUC": _auc(np.asarray(cut_labels, dtype=np.int64), np.asarray(cut_scores, dtype=np.float64))
                if labeled
                else None,
                "precision_top10pct_merge": _top_precision(merge_labels, merge_scores, 1),
                "precision_top10pct_cut": _top_precision(cut_labels, cut_scores, 1),
                "same_GT_positive_edge_ratio": float(len(same_pred) / max(len(same), 1)),
                "different_GT_positive_edge_ratio": float(len(diff_pred) / max(len(diff), 1)),
                "guard_pass_ratio": float(sum(1 for r in items if bool(r.get("guard_pass", False))) / max(len(items), 1)),
                "is_diagnostic_only": True,
            }
        )
    return out


def _dominant_false_merge_source(summary_rows: list[dict[str, Any]]) -> str:
    all_rows = [
        row
        for row in summary_rows
        if row["scene"] == "ALL"
        and str(row["category"]).startswith("B")
        and not str(row["category"]).startswith("B6_")
        and not str(row["category"]).startswith("B7_")
    ]
    if not all_rows:
        return "unknown"
    ranked = sorted(all_rows, key=lambda row: int(row.get("false_merge_count", 0)), reverse=True)
    category = str(ranked[0]["category"])
    mapping = {
        "B0_same_frame_same_mask": "local_same_mask",
        "B1_same_frame_different_mask": "boundary_region",
        "B2_same_chunk_different_frame": "mask_noise",
        "B3_cross_chunk_same_submap": "cross_chunk_edges",
        "B4_cross_chunk_near_overlap": "cross_chunk_edges",
        "B5_weak_or_blocked": "weak_alignment_leak",
        "B6_shuffled_label_control": "appearance_ambiguous",
        "B7_window0_same_mask": "local_same_mask",
    }
    return mapping.get(category, "unknown")


def _write_figures(output_root: Path, pair_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    labeled = [r for r in pair_rows if bool(r.get("gt_labeled_pair", False)) and r.get("distance_normalized") is not None]
    same_dist = [float(r["distance_normalized"]) for r in labeled if bool(r.get("same_gt", False))]
    diff_dist = [float(r["distance_normalized"]) for r in labeled if bool(r.get("different_gt", False))]
    plt.figure(figsize=(7, 4))
    if same_dist:
        plt.hist(same_dist, bins=40, alpha=0.6, label="same GT")
    if diff_dist:
        plt.hist(diff_dist, bins=40, alpha=0.6, label="different GT")
    plt.xlabel("distance / threshold")
    plt.ylabel("pair count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "distance_distribution_same_vs_different_gt.png", dpi=150)
    plt.close()

    all_summary = [r for r in summary_rows if r["scene"] == "ALL"]
    cats = [str(r["category"]) for r in all_summary]
    values = [float(r.get("false_merge_rate", 0.0)) for r in all_summary]
    plt.figure(figsize=(9, 4))
    plt.bar(range(len(cats)), values)
    plt.xticks(range(len(cats)), cats, rotation=45, ha="right", fontsize=7)
    plt.ylabel("false merge rate")
    plt.tight_layout()
    plt.savefig(fig_dir / "per_category_false_merge_rate.png", dpi=150)
    plt.close()

    neg_same = [int(r.get("visible_outside_count", 0)) for r in labeled if bool(r.get("same_gt", False))]
    neg_diff = [int(r.get("visible_outside_count", 0)) for r in labeled if bool(r.get("different_gt", False))]
    plt.figure(figsize=(7, 4))
    if neg_same:
        plt.hist(neg_same, bins=20, alpha=0.6, label="same GT")
    if neg_diff:
        plt.hist(neg_diff, bins=20, alpha=0.6, label="different GT")
    plt.xlabel("visible-outside negative count")
    plt.ylabel("pair count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "negative_evidence_distribution.png", dpi=150)
    plt.close()


def run_cache(
    *,
    cache_root: Path,
    split: Path,
    max_tubes_per_window: int,
    image_width: int,
    image_height: int,
    threshold_alpha: float,
    min_visibility: float,
    min_confidence: float,
    max_pairs_per_category: int,
    compute_appearance: bool,
    debug_progress: bool = False,
    category_prefix: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scenes = _read_split(split)
    pair_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    for scene in scenes:
        if not (cache_root / scene).exists():
            continue
        if debug_progress:
            print(f"[v27_pair_attribution] loading {scene} from {cache_root}", flush=True)
        chunks, load_diag = load_scene_chunks_from_cache(
            cache_root / scene,
            max_tubes_per_window=int(max_tubes_per_window),
            image_width=int(image_width),
            image_height=int(image_height),
        )
        builder = D4RTNativeSceneBuilder(object(), {"model": {"input": {"clip_frames": 32}}}, temporal_chunk_size=32, temporal_chunk_stride=16)
        records = chunks_to_records(builder.stitch_to_canonical(chunks))
        stream = ScanNetStream(seq_name=scene)
        frame_ids = sorted({int(v) for tube in records for v in np.asarray(tube.target_frames_global, dtype=np.int64).tolist()})
        masks_by_frame = {frame_id: stream.load_mask(frame_id) for frame_id in frame_ids}
        measurements, meas_diag = build_measurement_bank(
            records,
            masks_by_frame=masks_by_frame,
            min_visibility=float(min_visibility),
            min_confidence=float(min_confidence),
        )
        graph = build_signed_tube_graph(records, measurements, threshold_alpha=float(threshold_alpha))
        threshold = float(graph.diagnostics.get("distance_threshold", float(threshold_alpha)))
        gt_labels = assign_gt_labels(records, stream=stream, min_visibility=float(min_visibility), min_confidence=float(min_confidence))
        by_id = {int(t.tube_id): t for t in records}
        rgb_cache: dict[int, np.ndarray] = {}
        appearance_cache: dict[int, np.ndarray | None] = {}
        same_mask_pairs = sorted(
            {
                tuple(sorted((int(left), int(right))))
                for meas in measurements
                for left, right in meas.same_mask_merge_pairs
            }
        )[: int(max_pairs_per_category)]
        diff_mask_pairs = _different_mask_pairs(measurements, max_pairs=int(max_pairs_per_category))
        same_chunk_diff_frame_pairs = _same_chunk_sample_pairs(records, max_pairs=int(max_pairs_per_category))
        all_pairs_for_evidence = set(same_mask_pairs) | set(diff_mask_pairs) | set(same_chunk_diff_frame_pairs)
        pair_evidence = count_pair_measurement_evidence(measurements, all_pairs_for_evidence)
        same_mask_rows = _rows_from_pairs(
            scene=scene,
            category=f"{category_prefix}B0_same_frame_same_mask",
            pairs=same_mask_pairs,
            by_id=by_id,
            gt_labels=gt_labels,
            threshold=threshold,
            pair_evidence=pair_evidence,
            stream=stream,
            rgb_cache=rgb_cache,
            appearance_cache=appearance_cache,
            compute_appearance=bool(compute_appearance),
        )
        pair_rows.extend(same_mask_rows)
        pair_rows.extend(
            _rows_from_pairs(
                scene=scene,
                category=f"{category_prefix}B1_same_frame_different_mask",
                pairs=diff_mask_pairs,
                by_id=by_id,
                gt_labels=gt_labels,
                threshold=threshold,
                pair_evidence=pair_evidence,
                stream=stream,
                rgb_cache=rgb_cache,
                appearance_cache=appearance_cache,
                cannot_link=True,
                compute_appearance=bool(compute_appearance),
            )
        )
        pair_rows.extend(
            _rows_from_pairs(
                scene=scene,
                category=f"{category_prefix}B2_same_chunk_different_frame",
                pairs=same_chunk_diff_frame_pairs,
                by_id=by_id,
                gt_labels=gt_labels,
                threshold=threshold,
                pair_evidence=pair_evidence,
                stream=stream,
                rgb_cache=rgb_cache,
                appearance_cache=appearance_cache,
                compute_appearance=bool(compute_appearance),
            )
        )
        cross_same_submap = [
            (int(r["tube_i"]), int(r["tube_j"]))
            for r in same_mask_rows
            if not bool(r.get("same_chunk", False)) and bool(r.get("same_submap", False))
        ][: int(max_pairs_per_category)]
        near_overlap = [
            (int(r["tube_i"]), int(r["tube_j"]))
            for r in same_mask_rows
            if not bool(r.get("same_chunk", False)) and int(r.get("common_frame_count", 0)) > 0
        ][: int(max_pairs_per_category)]
        pair_rows.extend(
            _rows_from_pairs(
                scene=scene,
                category=f"{category_prefix}B3_cross_chunk_same_submap",
                pairs=cross_same_submap,
                by_id=by_id,
                gt_labels=gt_labels,
                threshold=threshold,
                pair_evidence=pair_evidence,
                stream=stream,
                rgb_cache=rgb_cache,
                appearance_cache=appearance_cache,
                compute_appearance=bool(compute_appearance),
            )
        )
        pair_rows.extend(
            _rows_from_pairs(
                scene=scene,
                category=f"{category_prefix}B4_cross_chunk_near_overlap",
                pairs=near_overlap,
                by_id=by_id,
                gt_labels=gt_labels,
                threshold=threshold,
                pair_evidence=pair_evidence,
                stream=stream,
                rgb_cache=rgb_cache,
                appearance_cache=appearance_cache,
                compute_appearance=bool(compute_appearance),
            )
        )
        blocked_pairs = [
            (int(event.get("tube_i", -1)), int(event.get("tube_j", -1)))
            for event in graph.blocked_events
            if int(event.get("tube_i", -1)) >= 0 and int(event.get("tube_j", -1)) >= 0
        ][: int(max_pairs_per_category)]
        pair_rows.extend(
            _rows_from_pairs(
                scene=scene,
                category=f"{category_prefix}B5_weak_or_blocked",
                pairs=blocked_pairs,
                by_id=by_id,
                gt_labels=gt_labels,
                threshold=threshold,
                pair_evidence=pair_evidence,
                stream=stream,
                rgb_cache=rgb_cache,
                appearance_cache=appearance_cache,
                compute_appearance=bool(compute_appearance),
            )
        )
        scene_rows.append(
            {
                "scene": scene,
                "cache_root": str(cache_root),
                "record_count": int(len(records)),
                "measurement_count": int(len(measurements)),
                "same_mask_pair_count": int(len(same_mask_pairs)),
                "different_mask_pair_count": int(len(diff_mask_pairs)),
                "same_chunk_different_frame_pair_count": int(len(same_chunk_diff_frame_pairs)),
                "blocked_event_count": int(len(graph.blocked_events)),
                "weak_alignment_chunk_count": int(load_diag.get("weak_alignment_chunk_count", 0)),
                "distance_threshold": threshold,
                "spacing_median": float(graph.diagnostics.get("spacing_median", 0.0)),
                **{f"measurement_{k}": v for k, v in meas_diag.items()},
            }
        )
        if debug_progress:
            print(f"[v27_pair_attribution] finished {scene}: records={len(records)} pairs_so_far={len(pair_rows)}", flush=True)
    return pair_rows, scene_rows


def _append_shuffled_control(pair_rows: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    out: list[dict[str, Any]] = []
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        if str(row["category"]).startswith("B"):
            by_scene[str(row["scene"])].append(row)
    for scene, rows in by_scene.items():
        tube_labels: dict[int, int] = {}
        for row in rows:
            tube_labels[int(row["tube_i"])] = int(row.get("gt_i", 0))
            tube_labels[int(row["tube_j"])] = int(row.get("gt_j", 0))
        ids = sorted(tube_labels)
        labels = np.asarray([tube_labels[tube_id] for tube_id in ids], dtype=np.int64)
        rng.shuffle(labels)
        shuffled = {tube_id: int(label) for tube_id, label in zip(ids, labels.tolist())}
        for row in rows[: max(1, min(len(rows), 50000))]:
            label_i = int(shuffled.get(int(row["tube_i"]), 0))
            label_j = int(shuffled.get(int(row["tube_j"]), 0))
            gt_labeled = bool(label_i > 0 and label_j > 0)
            new = dict(row)
            new.update(
                {
                    "category": "B6_shuffled_label_control",
                    "gt_i": label_i,
                    "gt_j": label_j,
                    "gt_labeled_pair": gt_labeled,
                    "same_gt": bool(gt_labeled and label_i == label_j),
                    "different_gt": bool(gt_labeled and label_i != label_j),
                }
            )
            out.append(new)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run v27 pair-level false-merge attribution diagnostics.")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--window0-cache-root", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--label", default="v27_pair_attribution")
    parser.add_argument("--max-tubes-per-window", type=int, default=160)
    parser.add_argument("--image-width", type=int, default=1296)
    parser.add_argument("--image-height", type=int, default=968)
    parser.add_argument("--threshold-alpha", type=float, default=3.0)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--max-pairs-per-category", type=int, default=20000)
    parser.add_argument("--appearance-mode", choices=["off", "rgb_patch"], default="off")
    parser.add_argument("--debug-progress", action="store_true")
    parser.add_argument("--seed", type=int, default=2700)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pair_rows, scene_rows = run_cache(
        cache_root=Path(args.cache_root),
        split=Path(args.split),
        max_tubes_per_window=int(args.max_tubes_per_window),
        image_width=int(args.image_width),
        image_height=int(args.image_height),
        threshold_alpha=float(args.threshold_alpha),
        min_visibility=float(args.min_visibility),
        min_confidence=float(args.min_confidence),
        max_pairs_per_category=int(args.max_pairs_per_category),
        compute_appearance=bool(args.appearance_mode == "rgb_patch"),
        debug_progress=bool(args.debug_progress),
    )
    pair_rows.extend(_append_shuffled_control(pair_rows, seed=int(args.seed)))
    if args.window0_cache_root:
        window_rows, window_scene_rows = run_cache(
            cache_root=Path(args.window0_cache_root),
            split=Path(args.split),
            max_tubes_per_window=int(args.max_tubes_per_window),
            image_width=int(args.image_width),
            image_height=int(args.image_height),
            threshold_alpha=float(args.threshold_alpha),
            min_visibility=float(args.min_visibility),
            min_confidence=float(args.min_confidence),
            max_pairs_per_category=int(args.max_pairs_per_category),
            compute_appearance=bool(args.appearance_mode == "rgb_patch"),
            debug_progress=bool(args.debug_progress),
            category_prefix="B7_window0_",
        )
        pair_rows.extend([row for row in window_rows if "same_frame_same_mask" in str(row["category"])])
        scene_rows.extend([{**row, "cache_role": "window0"} for row in window_scene_rows])
    summary_rows = _summarize_rows(pair_rows)
    dominant = _dominant_false_merge_source(summary_rows)
    phase_c_boundary_negative_complete = bool(
        sum(int(row.get("measurement_num_boundary_safe_merge_pairs", 0) or 0) for row in scene_rows) > 0
        and sum(int(row.get("measurement_num_same_frame_cannot_link_pairs", 0) or 0) for row in scene_rows) > 0
        and sum(int(row.get("measurement_num_visible_outside_negative_pairs", 0) or 0) for row in scene_rows) > 0
    )
    phase_c_appearance_motion_complete = bool(
        args.appearance_mode == "rgb_patch"
        and sum(int(row.get("measurement_num_appearance_pairs", 0) or 0) for row in scene_rows) > 0
        and sum(int(row.get("measurement_num_motion_pairs", 0) or 0) for row in scene_rows) > 0
    )
    missing_phase_c_fields: list[str] = []
    if not phase_c_boundary_negative_complete:
        missing_phase_c_fields.extend(
            [
                "boundary_safe_merge_pairs",
                "boundary_crossing_cut_pairs",
                "mask_distance_to_boundary_per_tube",
                "mask_eroded_interior_flag_per_tube",
                "mask_boundary_band_flag_per_tube",
            ]
        )
    if not phase_c_appearance_motion_complete:
        missing_phase_c_fields.extend(["appearance_feature_per_tube", "appearance_similarity_pairs", "motion_consistency_pairs"])
    manifest = {
        "label": str(args.label),
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
        "uses_gt_for_prediction": False,
        "uses_gt_sim3_for_prediction": False,
        "uses_rgbd_for_prediction": False,
        "uses_pose_for_prediction": False,
        "uses_scannet_mesh_for_prediction": False,
        "uses_d4rt_self_sim3": True,
        "alignment_source": "d4rt_self_sim3_or_same_chunk_identity",
        "coordinate_frame": "d4rt_canonical_for_cross_chunk_guarded_metric_pairs",
        "uses_gt_for_diagnostic_labels": True,
        "uses_rgb_for_diagnostic_appearance": bool(args.appearance_mode == "rgb_patch"),
        "threshold_alpha": float(args.threshold_alpha),
        "appearance_mode": str(args.appearance_mode),
        "max_pairs_per_category": int(args.max_pairs_per_category),
        "dominant_false_merge_source": dominant,
        "phase_b_complete": True,
        "phase_c_boundary_negative_fields_complete": phase_c_boundary_negative_complete,
        "phase_c_appearance_motion_fields_complete": phase_c_appearance_motion_complete,
        "phase_c_measurement_fields_complete": bool(
            phase_c_boundary_negative_complete and phase_c_appearance_motion_complete
        ),
        "missing_phase_c_fields": missing_phase_c_fields,
    }
    _write_csv(output_root / f"{args.label}_pair_rows.csv", pair_rows)
    (output_root / f"{args.label}_pair_rows.json").write_text(json.dumps(_json_safe(pair_rows), indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(output_root / f"{args.label}_category_summary.csv", summary_rows)
    (output_root / f"{args.label}_category_summary.json").write_text(
        json.dumps(_json_safe(summary_rows), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(output_root / f"{args.label}_scene_rows.csv", scene_rows)
    (output_root / f"{args.label}_scene_rows.json").write_text(json.dumps(_json_safe(scene_rows), indent=2, sort_keys=True), encoding="utf-8")
    (output_root / f"{args.label}_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")
    _write_figures(output_root, pair_rows, summary_rows)
    report_lines = [
        "# v27 Pair Attribution Diagnostic",
        "",
        "Diagnostic-only. GT labels are used only for attribution metrics.",
        "",
        f"- pair rows: {len(pair_rows)}",
        f"- categories: {len({row['category'] for row in pair_rows})}",
        f"- dominant_false_merge_source: `{dominant}`",
        "",
        "## ALL Category Summary",
        "",
        "| category | pairs | same-GT | diff-GT | false merge | false cut | merge AUC | cut AUC | distance p50 | boundary safe | boundary cross | negative | cannot-link |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        if row["scene"] != "ALL":
            continue
        report_lines.append(
            "| {category} | {pair_count} | {same_GT_ratio:.6f} | {different_GT_ratio:.6f} | {false_merge_rate:.6f} | {false_cut_rate:.6f} | {merge_auc} | {cut_auc} | {dist_p50} | {bs} | {bc} | {neg} | {cl} |".format(
                category=row["category"],
                pair_count=int(row["pair_count"]),
                same_GT_ratio=float(row["same_GT_ratio"]),
                different_GT_ratio=float(row["different_GT_ratio"]),
                false_merge_rate=float(row["false_merge_rate"]),
                false_cut_rate=float(row["false_cut_rate"]),
                merge_auc="NA" if row["merge_score_AUC"] is None else f"{float(row['merge_score_AUC']):.6f}",
                cut_auc="NA" if row["cut_score_AUC"] is None else f"{float(row['cut_score_AUC']):.6f}",
                dist_p50="NA" if row["distance_p50"] is None else f"{float(row['distance_p50']):.6f}",
                bs=int(row["boundary_safe_count"]),
                bc=int(row["boundary_cross_count"]),
                neg=int(row["visible_outside_count"]),
                cl=int(row["same_frame_cannot_link_count"]),
            )
        )
    report_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Boundary distance-transform and same-frame cannot-link fields are recorded when `phase_c_boundary_negative_fields_complete=true` in the manifest.",
            "- Appearance is optional RGB patch mean/std diagnostic, not a learned feature; `appearance_mode=off` leaves it uncomputed.",
            "- Motion consistency is pair-level only where overlapping frame geometry is available.",
        ]
    )
    (output_root / f"{args.label}_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe({"manifest": manifest, "summary_rows": [r for r in summary_rows if r["scene"] == "ALL"]}), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
