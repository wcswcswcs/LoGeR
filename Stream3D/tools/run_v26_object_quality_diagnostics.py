from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stream4d.scannet_stream import ScanNetStream
from stream4d_native.d4rt_scene_builder import D4RTNativeSceneBuilder
from stream4d_native.measurement_bank import MaskMeasurement, build_measurement_bank, count_pair_measurement_evidence
from stream4d_native.object_tube_io import MergeGeometryError, TubeRecord
from stream4d_native.signed_tube_graph import TubeGraphEdge, build_signed_tube_graph
from stream4d_native.tube_cover import select_tube_cover
from stream4d_native.tube_partition import (
    filter_edges_by_mutual_topk,
    filter_edges_by_min_score,
    filter_edges_by_pair_evidence,
    partition_tube_graph,
)
from tools.trace_v25_real_geometry_flow import chunks_to_records, load_scene_chunks_from_cache


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _read_split(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float)) and row.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def _load_gt_instance(scene_root: Path, frame_id: int) -> np.ndarray | None:
    zip_path = scene_root / f"{scene_root.name}_2d-instance.zip"
    if not zip_path.exists():
        return None
    member = f"instance/{int(frame_id)}.png"
    try:
        with zipfile.ZipFile(zip_path) as zf:
            if member not in set(zf.namelist()):
                return None
            data = np.frombuffer(zf.read(member), dtype=np.uint8)
    except (KeyError, zipfile.BadZipFile):
        return None
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3:
        image = image[..., 0]
    return image.astype(np.int64)


def _tube_visible(tube: TubeRecord, local_idx: int, *, min_visibility: float, min_confidence: float) -> bool:
    return bool(
        float(tube.visibility[local_idx]) >= float(min_visibility)
        and float(tube.confidence[local_idx]) >= float(min_confidence)
        and np.isfinite(tube.uv[local_idx]).all()
        and 0.0 <= float(tube.uv[local_idx, 0]) <= 1.0
        and 0.0 <= float(tube.uv[local_idx, 1]) <= 1.0
    )


def _xy_from_uv(uv: np.ndarray, shape: tuple[int, int]) -> tuple[int, int]:
    height, width = int(shape[0]), int(shape[1])
    x = int(np.clip(np.rint(float(uv[0]) * (width - 1)), 0, width - 1))
    y = int(np.clip(np.rint(float(uv[1]) * (height - 1)), 0, height - 1))
    return x, y


def assign_gt_labels(
    tubes: list[TubeRecord],
    *,
    stream: ScanNetStream,
    min_visibility: float,
    min_confidence: float,
) -> dict[int, int]:
    cache: dict[int, np.ndarray | None] = {}
    labels: dict[int, int] = {}
    for tube in tubes:
        counts: Counter[int] = Counter()
        for local_idx, frame_id in enumerate(np.asarray(tube.target_frames_global, dtype=np.int64).tolist()):
            if not _tube_visible(tube, local_idx, min_visibility=min_visibility, min_confidence=min_confidence):
                continue
            if int(frame_id) not in cache:
                cache[int(frame_id)] = _load_gt_instance(stream.root, int(frame_id))
            gt = cache[int(frame_id)]
            if gt is None:
                continue
            x, y = _xy_from_uv(tube.uv[local_idx], gt.shape)
            gt_id = int(gt[y, x])
            if gt_id > 0:
                counts[gt_id] += 1
        labels[int(tube.tube_id)] = int(counts.most_common(1)[0][0]) if counts else 0
    return labels


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(np.count_nonzero(pos))
    n_neg = int(np.count_nonzero(neg))
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    return float((np.sum(ranks[pos]) - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg))


def _candidate_scores(
    tubes: list[TubeRecord],
    measurements: list[MaskMeasurement],
    *,
    threshold_alpha: float,
) -> tuple[list[dict[str, Any]], list[TubeGraphEdge], list[dict[str, Any]], dict[tuple[int, int], dict[str, int]]]:
    graph = build_signed_tube_graph(tubes, measurements, threshold_alpha=float(threshold_alpha))
    by_edge = {tuple(sorted((edge.tube_i, edge.tube_j))): edge for edge in graph.edges}
    blocked = list(graph.blocked_events)
    by_id = {int(t.tube_id): t for t in tubes}
    pairs: set[tuple[int, int]] = set()
    for meas in measurements:
        for left, right in meas.same_mask_merge_pairs:
            pairs.add(tuple(sorted((int(left), int(right)))))
    pair_evidence = count_pair_measurement_evidence(measurements, pairs)
    spacing = float(graph.diagnostics.get("spacing_median", 1.0))
    threshold = float(graph.diagnostics.get("distance_threshold", float(threshold_alpha) * spacing))
    rows: list[dict[str, Any]] = []
    for left_id, right_id in sorted(pairs):
        left = by_id.get(left_id)
        right = by_id.get(right_id)
        if left is None or right is None:
            continue
        try:
            geom_l, geom_r, guard = left.get_geometry_for_merge(right, "v26_object_quality", merge_type="metric_edge")
        except MergeGeometryError:
            continue
        rep_l = np.nanmedian(np.asarray(geom_l, dtype=np.float32).reshape(-1, 3), axis=0)
        rep_r = np.nanmedian(np.asarray(geom_r, dtype=np.float32).reshape(-1, 3), axis=0)
        if not (np.isfinite(rep_l).all() and np.isfinite(rep_r).all()):
            continue
        dist = float(np.linalg.norm(rep_l - rep_r))
        evidence = pair_evidence.get((int(left_id), int(right_id)), {})
        same_mask_count = int(evidence.get("same_mask_count", 0))
        negative_count = int(evidence.get("visible_outside_negative_count", 0))
        evidence_total = same_mask_count + negative_count
        score = float(max(0.0, 1.0 - dist / max(threshold, 1e-6)))
        negative_adjusted_score = (
            float(score * same_mask_count / max(evidence_total, 1))
            if evidence_total > 0
            else float(score)
        )
        predicted_edge = bool((left_id, right_id) in by_edge)
        rows.append(
            {
                "tube_i": int(left_id),
                "tube_j": int(right_id),
                "distance": dist,
                "score": score,
                "negative_adjusted_score": negative_adjusted_score,
                "same_mask_count": same_mask_count,
                "visible_outside_negative_count": negative_count,
                "negative_evidence_total": int(evidence_total),
                "negative_majority_edge": bool(predicted_edge and same_mask_count > negative_count),
                "negative_strict_edge": bool(predicted_edge and negative_count == 0),
                "predicted_edge": predicted_edge,
                "same_chunk": bool(int(left.chunk_id) == int(right.chunk_id)),
                "guard_reason": guard.get("guard_reason"),
            }
        )
    return rows, graph.edges, blocked, pair_evidence


def edge_quality_rows(
    *,
    scene: str,
    candidate_rows: list[dict[str, Any]],
    gt_labels: dict[int, int],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    tube_ids = sorted(gt_labels)
    shuffled_values = np.asarray([gt_labels[t] for t in tube_ids], dtype=np.int64)
    rng.shuffle(shuffled_values)
    shuffled = {tube_id: int(label) for tube_id, label in zip(tube_ids, shuffled_values.tolist())}

    def summarize(
        name: str,
        labels_by_tube: dict[int, int],
        *,
        mask_only: bool = False,
        score_key: str = "score",
        predicted_key: str = "predicted_edge",
    ) -> dict[str, Any]:
        y_true: list[int] = []
        scores: list[float] = []
        pred_same = 0
        pred_diff = 0
        negative_total = 0
        for row in candidate_rows:
            li = int(labels_by_tube.get(int(row["tube_i"]), 0))
            lj = int(labels_by_tube.get(int(row["tube_j"]), 0))
            if li <= 0 or lj <= 0:
                continue
            same = int(li == lj)
            y_true.append(same)
            scores.append(float(row.get(score_key, row["score"])))
            predicted = True if mask_only else bool(row.get(predicted_key, False))
            negative_total += int(row.get("visible_outside_negative_count", 0))
            if predicted and same:
                pred_same += 1
            elif predicted and not same:
                pred_diff += 1
        return {
            "scene": scene,
            "variant": name,
            "candidate_pair_count": int(len(y_true)),
            "edge_auc": _auc(np.asarray(y_true, dtype=np.int64), np.asarray(scores, dtype=np.float64)) if y_true else None,
            "pred_same_gt_edge_count": int(pred_same),
            "pred_cross_gt_edge_count": int(pred_diff),
            "false_merge_rate": float(pred_diff / max(pred_same + pred_diff, 1)),
            "edge_precision_same_gt": float(pred_same / max(pred_same + pred_diff, 1)),
            "visible_outside_negative_count": int(negative_total),
            "score_key": score_key,
            "predicted_key": "mask_only" if mask_only else predicted_key,
            "is_diagnostic_only": True,
        }

    return [
        summarize("G0_mask_only_same_mask_pairs", gt_labels, mask_only=True),
        summarize("G1_real_d4rt_metric_edges", gt_labels),
        summarize(
            "G2_real_d4rt_negative_majority_edges",
            gt_labels,
            score_key="negative_adjusted_score",
            predicted_key="negative_majority_edge",
        ),
        summarize(
            "G3_real_d4rt_negative_strict_edges",
            gt_labels,
            score_key="negative_adjusted_score",
            predicted_key="negative_strict_edge",
        ),
        summarize("G5_shuffled_gt_labels_control", shuffled),
        summarize(
            "G6_shuffled_negative_majority_control",
            shuffled,
            score_key="negative_adjusted_score",
            predicted_key="negative_majority_edge",
        ),
        summarize(
            "G7_shuffled_negative_strict_control",
            shuffled,
            score_key="negative_adjusted_score",
            predicted_key="negative_strict_edge",
        ),
    ]


def _ari(labels_true: list[int], labels_pred: list[int]) -> float | None:
    if len(labels_true) < 2:
        return None
    n = len(labels_true)
    contingency: dict[tuple[int, int], int] = defaultdict(int)
    true_counts: Counter[int] = Counter(labels_true)
    pred_counts: Counter[int] = Counter(labels_pred)
    for t, p in zip(labels_true, labels_pred):
        contingency[(int(t), int(p))] += 1
    sum_comb = sum(comb(v, 2) for v in contingency.values() if v >= 2)
    sum_true = sum(comb(v, 2) for v in true_counts.values() if v >= 2)
    sum_pred = sum(comb(v, 2) for v in pred_counts.values() if v >= 2)
    total = comb(n, 2)
    expected = sum_true * sum_pred / total if total else 0.0
    max_index = 0.5 * (sum_true + sum_pred)
    denom = max_index - expected
    if denom == 0:
        return 0.0
    return float((sum_comb - expected) / denom)


def partition_quality(
    *,
    scene: str,
    tubes: list[TubeRecord],
    edges: list[TubeGraphEdge],
    gt_labels: dict[int, int],
    variant: str,
) -> dict[str, Any]:
    part = partition_tube_graph([int(t.tube_id) for t in tubes], edges)
    comp_id: dict[int, int] = {}
    for idx, comp in enumerate(part.components):
        for tube_id in comp:
            comp_id[int(tube_id)] = int(idx)
    labeled = [tube_id for tube_id, label in gt_labels.items() if int(label) > 0]
    labels_true = [int(gt_labels[tube_id]) for tube_id in labeled]
    labels_pred = [int(comp_id.get(tube_id, -1)) for tube_id in labeled]
    purity_num = 0
    overmerge = 0
    comp_to_labels: dict[int, Counter[int]] = defaultdict(Counter)
    for tube_id in labeled:
        comp_to_labels[int(comp_id.get(tube_id, -1))][int(gt_labels[tube_id])] += 1
    for counts in comp_to_labels.values():
        if len(counts) > 1:
            overmerge += 1
        purity_num += max(counts.values()) if counts else 0
    gt_to_comps: dict[int, set[int]] = defaultdict(set)
    for tube_id in labeled:
        gt_to_comps[int(gt_labels[tube_id])].add(int(comp_id.get(tube_id, -1)))
    oversplit = sum(1 for comps in gt_to_comps.values() if len(comps) > 1)
    return {
        "scene": scene,
        "variant": variant,
        "component_count": int(part.diagnostics["component_count"]),
        "largest_component_size": int(part.diagnostics["largest_component_size"]),
        "largest_component_ratio": float(part.diagnostics["largest_component_size"] / max(len(tubes), 1)),
        "positive_edge_count": int(part.diagnostics["positive_edge_count"]),
        "labeled_tube_count": int(len(labeled)),
        "ari": _ari(labels_true, labels_pred),
        "purity": float(purity_num / max(len(labeled), 1)),
        "overmerge_count": int(overmerge),
        "oversplit_count": int(oversplit),
        "is_diagnostic_only": True,
    }


def run_scene(
    *,
    scene: str,
    cache_root: Path,
    max_tubes_per_window: int,
    image_width: int,
    image_height: int,
    threshold_alpha: float,
    min_visibility: float,
    min_confidence: float,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
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
    cover_all = select_tube_cover(measurements)
    cover_greedy = select_tube_cover(measurements, strategy="greedy_tube_cover")
    candidate_rows, edges, blocked, pair_evidence = _candidate_scores(
        records,
        cover_all.selected_measurements,
        threshold_alpha=float(threshold_alpha),
    )
    gt_labels = assign_gt_labels(records, stream=stream, min_visibility=float(min_visibility), min_confidence=float(min_confidence))
    measurement_rows = [
        {
            "scene": scene,
            "variant": "T0_all_measurements",
            **meas_diag,
            "selected_measurement_count": int(len(cover_all.selected_measurements)),
            "covered_tube_count": int(len(cover_all.covered_tube_ids)),
            "tube_node_coverage": float(len(cover_all.covered_tube_ids) / max(len(records), 1)),
            "is_diagnostic_only": True,
        },
        {
            "scene": scene,
            "variant": "T2_greedy_node_cover",
            **meas_diag,
            "selected_measurement_count": int(len(cover_greedy.selected_measurements)),
            "covered_tube_count": int(len(cover_greedy.covered_tube_ids)),
            "tube_node_coverage": float(len(cover_greedy.covered_tube_ids) / max(len(records), 1)),
            "is_diagnostic_only": True,
        },
    ]
    edge_rows = edge_quality_rows(scene=scene, candidate_rows=candidate_rows, gt_labels=gt_labels, rng=rng)
    negative_majority_edges = filter_edges_by_pair_evidence(edges, pair_evidence, mode="negative_majority")
    negative_strict_edges = filter_edges_by_pair_evidence(edges, pair_evidence, mode="negative_strict")
    partition_rows = [
        partition_quality(scene=scene, tubes=records, edges=edges, gt_labels=gt_labels, variant="P0_real_d4rt_partition"),
        partition_quality(
            scene=scene,
            tubes=records,
            edges=negative_majority_edges,
            gt_labels=gt_labels,
            variant="P1_negative_majority_partition",
        ),
        partition_quality(
            scene=scene,
            tubes=records,
            edges=negative_strict_edges,
            gt_labels=gt_labels,
            variant="P2_negative_strict_partition",
        ),
        partition_quality(
            scene=scene,
            tubes=records,
            edges=filter_edges_by_mutual_topk(negative_majority_edges, top_k=1),
            gt_labels=gt_labels,
            variant="P3_negative_majority_mutual_top1_partition",
        ),
        partition_quality(
            scene=scene,
            tubes=records,
            edges=filter_edges_by_mutual_topk(negative_majority_edges, top_k=2),
            gt_labels=gt_labels,
            variant="P4_negative_majority_mutual_top2_partition",
        ),
        partition_quality(
            scene=scene,
            tubes=records,
            edges=filter_edges_by_min_score(negative_majority_edges, min_score=0.25),
            gt_labels=gt_labels,
            variant="P5_negative_majority_score025_partition",
        ),
        partition_quality(
            scene=scene,
            tubes=records,
            edges=filter_edges_by_min_score(negative_majority_edges, min_score=0.50),
            gt_labels=gt_labels,
            variant="P6_negative_majority_score050_partition",
        ),
        partition_quality(
            scene=scene,
            tubes=records,
            edges=filter_edges_by_min_score(negative_majority_edges, min_score=0.75),
            gt_labels=gt_labels,
            variant="P7_negative_majority_score075_partition",
        ),
        partition_quality(scene=scene, tubes=records, edges=[], gt_labels=gt_labels, variant="P8_no_edges_control"),
    ]
    scene_summary = {
        "scene": scene,
        "record_count": int(len(records)),
        "measurement_count": int(len(measurements)),
        "candidate_pair_count": int(len(candidate_rows)),
        "positive_edge_count": int(len(edges)),
        "candidate_pair_visible_outside_negative_count": int(
            sum(int(v.get("visible_outside_negative_count", 0)) for v in pair_evidence.values())
        ),
        "blocked_event_count": int(len(blocked)),
        "weak_alignment_chunk_count": int(load_diag.get("weak_alignment_chunk_count", 0)),
    }
    return measurement_rows, edge_rows, partition_rows, scene_summary


def aggregate(edge_rows: list[dict[str, Any]], partition_rows: list[dict[str, Any]]) -> dict[str, Any]:
    real = [row for row in edge_rows if row["variant"] == "G1_real_d4rt_metric_edges"]
    real_neg_major = [row for row in edge_rows if row["variant"] == "G2_real_d4rt_negative_majority_edges"]
    real_neg_strict = [row for row in edge_rows if row["variant"] == "G3_real_d4rt_negative_strict_edges"]
    shuffle = [row for row in edge_rows if row["variant"] == "G5_shuffled_gt_labels_control"]
    shuffle_neg_major = [row for row in edge_rows if row["variant"] == "G6_shuffled_negative_majority_control"]
    shuffle_neg_strict = [row for row in edge_rows if row["variant"] == "G7_shuffled_negative_strict_control"]
    mask = [row for row in edge_rows if row["variant"] == "G0_mask_only_same_mask_pairs"]
    part = [row for row in partition_rows if row["variant"] == "P0_real_d4rt_partition"]
    part_neg_major = [row for row in partition_rows if row["variant"] == "P1_negative_majority_partition"]
    part_neg_strict = [row for row in partition_rows if row["variant"] == "P2_negative_strict_partition"]
    part_neg_top1 = [row for row in partition_rows if row["variant"] == "P3_negative_majority_mutual_top1_partition"]
    part_neg_top2 = [row for row in partition_rows if row["variant"] == "P4_negative_majority_mutual_top2_partition"]
    part_neg_score025 = [row for row in partition_rows if row["variant"] == "P5_negative_majority_score025_partition"]
    part_neg_score050 = [row for row in partition_rows if row["variant"] == "P6_negative_majority_score050_partition"]
    part_neg_score075 = [row for row in partition_rows if row["variant"] == "P7_negative_majority_score075_partition"]
    no_edges = [row for row in partition_rows if row["variant"] == "P8_no_edges_control"]
    real_auc = _mean(real, "edge_auc")
    shuffle_auc = _mean(shuffle, "edge_auc")
    real_neg_major_auc = _mean(real_neg_major, "edge_auc")
    shuffle_neg_major_auc = _mean(shuffle_neg_major, "edge_auc")
    real_neg_strict_auc = _mean(real_neg_strict, "edge_auc")
    shuffle_neg_strict_auc = _mean(shuffle_neg_strict, "edge_auc")
    edge_margin = (
        float(real_auc) - float(shuffle_auc)
        if real_auc is not None and shuffle_auc is not None
        else None
    )
    neg_major_edge_margin = (
        float(real_neg_major_auc) - float(shuffle_neg_major_auc)
        if real_neg_major_auc is not None and shuffle_neg_major_auc is not None
        else None
    )
    neg_strict_edge_margin = (
        float(real_neg_strict_auc) - float(shuffle_neg_strict_auc)
        if real_neg_strict_auc is not None and shuffle_neg_strict_auc is not None
        else None
    )
    part_ari = _mean(part, "ari")
    part_neg_major_ari = _mean(part_neg_major, "ari")
    part_neg_strict_ari = _mean(part_neg_strict, "ari")
    part_neg_top1_ari = _mean(part_neg_top1, "ari")
    part_neg_top2_ari = _mean(part_neg_top2, "ari")
    part_neg_score025_ari = _mean(part_neg_score025, "ari")
    part_neg_score050_ari = _mean(part_neg_score050, "ari")
    part_neg_score075_ari = _mean(part_neg_score075, "ari")
    no_edges_ari = _mean(no_edges, "ari")
    ari_margin = (
        float(part_ari) - float(no_edges_ari)
        if part_ari is not None and no_edges_ari is not None
        else None
    )
    return {
        "scene_count": int(len({row["scene"] for row in edge_rows})),
        "real_edge_auc_mean": real_auc,
        "shuffle_edge_auc_mean": shuffle_auc,
        "edge_auc_margin_vs_shuffle": edge_margin,
        "negative_majority_edge_auc_mean": real_neg_major_auc,
        "negative_majority_shuffle_edge_auc_mean": shuffle_neg_major_auc,
        "negative_majority_edge_auc_margin_vs_shuffle": neg_major_edge_margin,
        "negative_strict_edge_auc_mean": real_neg_strict_auc,
        "negative_strict_shuffle_edge_auc_mean": shuffle_neg_strict_auc,
        "negative_strict_edge_auc_margin_vs_shuffle": neg_strict_edge_margin,
        "mask_only_false_merge_rate_mean": _mean(mask, "false_merge_rate"),
        "real_false_merge_rate_mean": _mean(real, "false_merge_rate"),
        "negative_majority_false_merge_rate_mean": _mean(real_neg_major, "false_merge_rate"),
        "negative_strict_false_merge_rate_mean": _mean(real_neg_strict, "false_merge_rate"),
        "real_partition_ari_mean": part_ari,
        "negative_majority_partition_ari_mean": part_neg_major_ari,
        "negative_strict_partition_ari_mean": part_neg_strict_ari,
        "negative_majority_mutual_top1_partition_ari_mean": part_neg_top1_ari,
        "negative_majority_mutual_top2_partition_ari_mean": part_neg_top2_ari,
        "negative_majority_score025_partition_ari_mean": part_neg_score025_ari,
        "negative_majority_score050_partition_ari_mean": part_neg_score050_ari,
        "negative_majority_score075_partition_ari_mean": part_neg_score075_ari,
        "no_edges_partition_ari_mean": no_edges_ari,
        "partition_ari_margin_vs_no_edges": ari_margin,
        "real_partition_purity_mean": _mean(part, "purity"),
        "negative_majority_partition_purity_mean": _mean(part_neg_major, "purity"),
        "negative_strict_partition_purity_mean": _mean(part_neg_strict, "purity"),
        "negative_majority_mutual_top1_partition_purity_mean": _mean(part_neg_top1, "purity"),
        "negative_majority_mutual_top2_partition_purity_mean": _mean(part_neg_top2, "purity"),
        "negative_majority_score025_partition_purity_mean": _mean(part_neg_score025, "purity"),
        "negative_majority_score050_partition_purity_mean": _mean(part_neg_score050, "purity"),
        "negative_majority_score075_partition_purity_mean": _mean(part_neg_score075, "purity"),
        "no_edges_partition_purity_mean": _mean(no_edges, "purity"),
        "stop4_real_beats_shuffle": bool(
            edge_margin is not None and float(edge_margin) > 0.0
        ),
        "stop4_negative_majority_real_beats_shuffle": bool(
            neg_major_edge_margin is not None and float(neg_major_edge_margin) > 0.0
        ),
        "stop4_negative_strict_real_beats_shuffle": bool(
            neg_strict_edge_margin is not None and float(neg_strict_edge_margin) > 0.0
        ),
        "phase_de_strong_edge_gate_margin005": bool(edge_margin is not None and float(edge_margin) >= 0.05),
        "phase_de_negative_majority_edge_gate_margin005": bool(
            neg_major_edge_margin is not None and float(neg_major_edge_margin) >= 0.05
        ),
        "phase_de_negative_strict_edge_gate_margin005": bool(
            neg_strict_edge_margin is not None and float(neg_strict_edge_margin) >= 0.05
        ),
        "stop5_partition_improves": bool(
            ari_margin is not None and float(ari_margin) > 0.05
        ),
        "stop5_negative_majority_partition_improves_over_no_edges": bool(
            part_neg_major_ari is not None and no_edges_ari is not None and float(part_neg_major_ari) - float(no_edges_ari) > 0.05
        ),
        "stop5_negative_strict_partition_improves_over_no_edges": bool(
            part_neg_strict_ari is not None and no_edges_ari is not None and float(part_neg_strict_ari) - float(no_edges_ari) > 0.05
        ),
        "stop5_negative_top1_partition_improves_over_no_edges": bool(
            part_neg_top1_ari is not None and no_edges_ari is not None and float(part_neg_top1_ari) - float(no_edges_ari) > 0.05
        ),
        "stop5_negative_top2_partition_improves_over_no_edges": bool(
            part_neg_top2_ari is not None and no_edges_ari is not None and float(part_neg_top2_ari) - float(no_edges_ari) > 0.05
        ),
        "stop5_negative_score025_partition_improves_over_no_edges": bool(
            part_neg_score025_ari is not None and no_edges_ari is not None and float(part_neg_score025_ari) - float(no_edges_ari) > 0.05
        ),
        "stop5_negative_score050_partition_improves_over_no_edges": bool(
            part_neg_score050_ari is not None and no_edges_ari is not None and float(part_neg_score050_ari) - float(no_edges_ari) > 0.05
        ),
        "stop5_negative_score075_partition_improves_over_no_edges": bool(
            part_neg_score075_ari is not None and no_edges_ari is not None and float(part_neg_score075_ari) - float(no_edges_ari) > 0.05
        ),
        "method_result": False,
        "is_diagnostic_only": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run v26 object-quality measurement/edge/partition diagnostics.")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--max-tubes-per-window", type=int, default=160)
    parser.add_argument("--image-width", type=int, default=1296)
    parser.add_argument("--image-height", type=int, default=968)
    parser.add_argument("--threshold-alpha", type=float, default=2.0)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--label", default="v26_object_quality")
    parser.add_argument("--seed", type=int, default=2600)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scenes = _read_split(Path(args.split))
    rng = np.random.default_rng(int(args.seed))
    measurement_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    partition_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    for scene in scenes:
        if not (Path(args.cache_root) / scene).exists():
            continue
        m_rows, e_rows, p_rows, s_row = run_scene(
            scene=scene,
            cache_root=Path(args.cache_root),
            max_tubes_per_window=int(args.max_tubes_per_window),
            image_width=int(args.image_width),
            image_height=int(args.image_height),
            threshold_alpha=float(args.threshold_alpha),
            min_visibility=float(args.min_visibility),
            min_confidence=float(args.min_confidence),
            rng=rng,
        )
        measurement_rows.extend(m_rows)
        edge_rows.extend(e_rows)
        partition_rows.extend(p_rows)
        scene_rows.append(s_row)
    summary = aggregate(edge_rows, partition_rows)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for suffix, rows in [
        ("measurement_rows", measurement_rows),
        ("edge_rows", edge_rows),
        ("partition_rows", partition_rows),
        ("scene_rows", scene_rows),
    ]:
        _write_csv(output_root / f"{args.label}_{suffix}.csv", rows)
        (output_root / f"{args.label}_{suffix}.json").write_text(
            json.dumps(_json_safe(rows), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    (output_root / f"{args.label}_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
