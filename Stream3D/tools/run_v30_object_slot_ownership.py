from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stream4d.scannet_stream import ScanNetStream
from stream4d_native.d4rt_scene_builder import D4RTNativeSceneBuilder
from stream4d_native.measurement_bank import build_measurement_bank
from tools.run_v27_pair_attribution import _appearance_feature, _cosine
from tools.run_v26_object_quality_diagnostics import _json_safe, _read_split, _write_csv
from tools.run_v28_proposal_oracle import _cluster_metrics
from tools.run_v28_proposal_selection import _load_gt_labels
from tools.run_v29_constrained_ownership_solver import (
    SMALL_CHILD_PREFIXES,
    TEMPORAL_PREFIXES,
    _add_diagnostic_gt_fields,
    _core_ids,
    _eval_selected,
    _float,
    _int,
    _is_o5,
    _mean,
    _proposal_gt_counts,
    _quantile,
    _selected_ids_from_csv,
    _set_core_ids,
    _type_bucket,
)
from tools.trace_v25_real_geometry_flow import chunks_to_records, load_scene_chunks_from_cache


REAL_ROOT = "v28_proposal_oracle_repair_temporal_track_640_pruned_shared2_t20_guard5_probe5_r2"
P11_ROOT = "v28_proposal_selection_guard5_p11_ownership_expansion_r1"
STRICT_ROOT = "v28_proposal_selection_guard5_strict_score02_r3_with_p8_proxy"
MEDIUM_ROOT = "v29_medium_proposals"
NATIVE_RGB_BOUNDARY_PROFILE = "native_rgb_boundary_repair"

METHOD_MANIFEST_BASE: dict[str, Any] = {
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
    "geometry_field": "D4RT canonical proposal tube memberships plus mask/proposal features",
    "coordinate_frame": "d4rt_canonical tubes; image-space mask/proposal metadata",
    "alignment_source": "D4RT self-Sim3 inherited from v28 proposal artifacts",
}


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _is_broad_observation(row: dict[str, Any], scene_p80: dict[str, float]) -> bool:
    proposal_type = str(row.get("proposal_type") or "")
    core_count = len(_core_ids(row))
    return proposal_type.startswith(TEMPORAL_PREFIXES) and core_count >= max(20.0, 0.65 * scene_p80.get(str(row.get("scene")), 40.0))


def _labels_from_slot_sets(slot_sets: list[set[int]], gt_labels: dict[int, int]) -> tuple[dict[int, int], int]:
    labels_pred: dict[int, int] = {}
    for slot_idx, ids in enumerate(slot_sets):
        for tid in sorted(ids):
            tid = int(tid)
            if int(gt_labels.get(tid, 0)) > 0 and tid not in labels_pred:
                labels_pred[tid] = slot_idx
    next_label = len(slot_sets)
    unknown_count = 0
    for tid, gt in sorted(gt_labels.items()):
        if int(gt) <= 0:
            continue
        tid = int(tid)
        if tid not in labels_pred:
            labels_pred[tid] = next_label
            next_label += 1
            unknown_count += 1
    return labels_pred, unknown_count


def _eval_slot_sets(slot_sets: list[set[int]], gt_labels: dict[int, int]) -> dict[str, Any]:
    labels_pred, unknown_count = _labels_from_slot_sets(slot_sets, gt_labels)
    metrics = _cluster_metrics(labels_pred, gt_labels)
    labeled = [int(tid) for tid, gt in gt_labels.items() if int(gt) > 0]
    owned = set().union(*slot_sets) if slot_sets else set()
    metrics.update(
        {
            "ARI": metrics.pop("ari"),
            "purity": metrics["purity"],
            "completeness": metrics["completeness"],
            "overmerge": metrics["overmerge"],
            "oversplit": metrics["oversplit"],
            "unknown_tube_count": int(unknown_count),
            "unknown_tube_ratio": float(unknown_count / max(len(labeled), 1)),
            "owned_tube_ratio": float(len(owned & set(labeled)) / max(len(labeled), 1)),
            "labeled_tube_count": int(len(labeled)),
        }
    )
    return metrics


def _aggregate_variant(scene_rows: list[dict[str, Any]], variant_key: str = "variant") -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scene_rows:
        grouped[str(row[variant_key])].append(row)
    out: list[dict[str, Any]] = []
    for variant, rows in sorted(grouped.items()):
        all_row: dict[str, Any] = {"scene": "ALL", variant_key: variant, "scene_count": int(len(rows))}
        for key in (
            "ARI",
            "purity",
            "completeness",
            "unknown_tube_ratio",
            "owned_tube_ratio",
            "overmerge",
            "oversplit",
            "decomposition_oracle_ARI",
            "decomposition_oracle_purity",
            "decomposition_oracle_completeness",
            "GT_coverage_after_decomposition",
        ):
            vals = [float(row[key]) for row in rows if row.get(key) is not None and str(row.get(key)) != ""]
            if vals:
                all_row[key] = _mean(vals)
        all_row["scene0081_ARI"] = next((row.get("ARI") for row in rows if row.get("scene") == "scene0081_01"), None)
        all_row["scene0081_decomposition_oracle_ARI"] = next(
            (row.get("decomposition_oracle_ARI") for row in rows if row.get("scene") == "scene0081_01"),
            None,
        )
        out.append(all_row)
    return out


def _proposal_union_quality(ids: set[int], gt_labels: dict[int, int], gt_counts: Counter[int]) -> dict[str, Any]:
    counts: Counter[int] = Counter()
    labeled = 0
    for tid in ids:
        gt = int(gt_labels.get(int(tid), 0))
        if gt > 0:
            counts[gt] += 1
            labeled += 1
    if not counts:
        return {
            "purity": None,
            "completeness": None,
            "best_iou": None,
            "covered_GT_count": 0,
            "dominant_GT_ratio": None,
            "secondary_GT_ratio": None,
        }
    best_gt, best_overlap = counts.most_common(1)[0]
    second_overlap = counts.most_common(2)[1][1] if len(counts) > 1 else 0
    return {
        "purity": float(best_overlap / max(labeled, 1)),
        "completeness": float(best_overlap / max(gt_counts.get(int(best_gt), 0), 1)),
        "best_iou": float(best_overlap / max(labeled + gt_counts.get(int(best_gt), 0) - best_overlap, 1)),
        "covered_GT_count": int(len(counts)),
        "dominant_GT_ratio": float(best_overlap / max(labeled, 1)),
        "secondary_GT_ratio": float(second_overlap / max(labeled, 1)),
    }


def _fresh_proposal_gt_counts(row: dict[str, Any], gt_labels: dict[int, int]) -> tuple[Counter[int], int]:
    counts: Counter[int] = Counter()
    labeled = 0
    for tid in _core_ids(row):
        gt = int(gt_labels.get(int(tid), 0))
        if gt > 0:
            counts[gt] += 1
            labeled += 1
    return counts, int(labeled)


def _refresh_diagnostic_gt_fields(row: dict[str, Any], gt_labels: dict[int, int], gt_counts: Counter[int]) -> None:
    counts, labeled = _fresh_proposal_gt_counts(row, gt_labels)
    best_gt = 0
    best_overlap = 0
    if counts:
        best_gt, best_overlap = counts.most_common(1)[0]
    purity = float(best_overlap / max(labeled, 1)) if labeled else None
    completeness = float(best_overlap / max(int(gt_counts.get(int(best_gt), 0)), 1)) if best_gt else None
    iou = None
    if best_gt:
        iou = float(best_overlap / max(labeled + int(gt_counts.get(int(best_gt), 0)) - best_overlap, 1))
    row["_gt_overlap_counts"] = {int(k): int(v) for k, v in counts.items()}
    row["_proposal_labeled_tube_count"] = int(labeled)
    row["proposal_labeled_tube_count"] = int(labeled)
    row["proposal_best_GT"] = int(best_gt)
    row["proposal_purity"] = purity
    row["proposal_completeness"] = completeness
    row["proposal_best_IoU"] = iou


def _quality_proxy(row: dict[str, Any], scene_stats: dict[str, dict[str, float]]) -> float:
    scene = str(row.get("scene"))
    stats = scene_stats.get(scene, {})
    core = len(_core_ids(row))
    core_target = stats.get("core_p45", 20.0)
    core_scale = max(stats.get("core_p80", 60.0) - stats.get("core_p20", 5.0), 1.0)
    core_balance = max(0.0, 1.0 - abs(core - core_target) / core_scale)
    area_norm_cannot = _float(row, "same_frame_cannot_link_rate") / max(_float(row, "proposal_area_ratio", 1e-6), 1e-6)
    score = (
        0.32 * _float(row, "eroded_interior_ratio")
        + 0.20 * _float(row, "visibility_mean", 0.5)
        + 0.16 * _float(row, "confidence_mean", 0.5)
        + 0.14 * _float(row, "mask_temporal_repeat_score")
        + 0.18 * core_balance
        - 0.07 * min(area_norm_cannot, 8.0)
        - 0.08 * min(_float(row, "visible_outside_negative_rate"), 8.0)
        - 0.05 * min(_float(row, "boundary_risk"), 8.0)
        - 0.04 * min(_float(row, "tube_canonical_compactness"), 8.0)
    )
    return float(score)


def _scene_stats(rows_by_scene: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    fields = [
        "same_frame_cannot_link_rate",
        "visible_outside_negative_rate",
        "boundary_risk",
        "tube_canonical_compactness",
        "appearance_variance",
        "eroded_interior_ratio",
        "image_gradient_boundary_score",
        "mask_distance_mean",
        "visibility_mean",
        "confidence_mean",
        "mask_temporal_repeat_score",
        "proposal_area_ratio",
        "overlap_with_other_proposals",
        "tube_density",
    ]
    for scene, rows in rows_by_scene.items():
        stats: dict[str, float] = {}
        core_vals = [len(_core_ids(row)) for row in rows if len(_core_ids(row)) > 0]
        for qname, q in (("p20", 0.20), ("p30", 0.30), ("p45", 0.45), ("p60", 0.60), ("p70", 0.70), ("p80", 0.80), ("p90", 0.90)):
            stats[f"core_{qname}"] = float(np.quantile(core_vals, q)) if core_vals else 0.0
        for field in fields:
            vals = [_float(row, field) for row in rows if row.get(field) not in {None, ""}]
            for qname, q in (("p25", 0.25), ("p35", 0.35), ("p40", 0.40), ("p50", 0.50), ("p60", 0.60), ("p65", 0.65), ("p70", 0.70), ("p75", 0.75), ("p80", 0.80)):
                stats[f"{field}_{qname}"] = float(np.quantile(vals, q)) if vals else 0.0
        out[scene] = stats
    return out


def _normalized_seed_quality(row: dict[str, Any], scene_stats: dict[str, dict[str, float]]) -> float:
    scene = str(row.get("scene"))
    stats = scene_stats.get(scene, {})
    core = len(_core_ids(row))
    core_mid = stats.get("core_p60", 24.0)
    core_scale = max(stats.get("core_p90", 80.0) - stats.get("core_p20", 4.0), 1.0)
    size_balance = max(0.0, 1.0 - abs(core - core_mid) / core_scale)
    compact_norm = min(_float(row, "tube_canonical_compactness") / max(stats.get("tube_canonical_compactness_p50", 1e-6), 1e-6), 3.0)
    appearance_norm = min(_float(row, "appearance_variance") / max(stats.get("appearance_variance_p50", 1e-6), 1e-6), 3.0)
    boundary_norm = min(_float(row, "boundary_risk") / max(stats.get("boundary_risk_p75", 1e-6), 1e-6), 3.0)
    distance_norm = min(_float(row, "mask_distance_mean") / max(stats.get("mask_distance_mean_p60", 1e-6), 1e-6), 1.5)
    score = (
        0.30 * _float(row, "eroded_interior_ratio")
        + 0.22 * distance_norm
        + 0.16 * _float(row, "visibility_mean", 0.5)
        + 0.12 * size_balance
        - 0.18 * compact_norm
        - 0.12 * appearance_norm
        - 0.08 * boundary_norm
    )
    return float(score)


def _overlap_supported_seed_quality(row: dict[str, Any], scene_stats: dict[str, dict[str, float]]) -> float:
    scene = str(row.get("scene"))
    stats = scene_stats.get(scene, {})
    overlap_norm = min(
        _float(row, "overlap_with_other_proposals")
        / max(stats.get("overlap_with_other_proposals_p60", 1e-6), 1e-6),
        3.0,
    )
    distance_norm = min(
        _float(row, "mask_distance_mean")
        / max(stats.get("mask_distance_mean_p60", 1e-6), 1e-6),
        2.0,
    )
    density_norm = min(_float(row, "tube_density") / max(stats.get("tube_density_p60", 1e-6), 1e-6), 2.0)
    return float(
        _normalized_seed_quality(row, scene_stats)
        + 0.10 * overlap_norm
        + 0.08 * distance_norm
        + 0.05 * density_norm
    )


def _native_rgb_boundary_seed_quality(row: dict[str, Any], scene_stats: dict[str, dict[str, float]]) -> float:
    rgb_p10 = max(_float(row, "native_rgb_pair_cos_p10", 0.0), 0.0)
    rgb_mean = max(_float(row, "native_rgb_pair_cos_mean", 0.0), 0.0)
    safe_ratio = max(_float(row, "native_boundary_safe_ratio", 0.0), 0.0)
    boundary_norm = min(max(_float(row, "native_boundary_distance_p10", 0.0), 0.0) / 16.0, 2.0)
    support = max(_float(row, "native_frame_mask_support_ratio", 0.0), 0.0)
    return float(
        _normalized_seed_quality(row, scene_stats)
        + 0.24 * rgb_p10
        + 0.08 * rgb_mean
        + 0.16 * safe_ratio
        + 0.08 * boundary_norm
        + 0.04 * support
    )


def _proposal_overlap(a: set[int], b: set[int]) -> float:
    return float(len(a & b) / max(min(len(a), len(b)), 1))


def _feature_pair_stats(features: list[np.ndarray]) -> tuple[float | None, float | None]:
    sims: list[float] = []
    for idx, left in enumerate(features):
        for right in features[idx + 1 :]:
            sim = _cosine(left, right)
            if sim is not None and np.isfinite(float(sim)):
                sims.append(float(sim))
    if not sims:
        return None, None
    arr = np.asarray(sims, dtype=np.float64)
    return float(np.mean(arr)), float(np.quantile(arr, 0.10))


def _annotate_native_rgb_boundary_features(
    *,
    rows_by_scene: dict[str, list[dict[str, Any]]],
    scene_stats: dict[str, dict[str, float]],
    scenes: list[str],
    cache_root: Path,
    max_tubes_per_window: int,
    image_width: int,
    image_height: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    feature_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    for scene in scenes:
        rows = rows_by_scene.get(scene, [])
        scene_dir = cache_root / scene
        if not scene_dir.exists():
            scene_rows.append({"scene": scene, "status": "missing_cache", "candidate_count": int(len(rows))})
            continue
        chunks, _ = load_scene_chunks_from_cache(
            scene_dir,
            max_tubes_per_window=int(max_tubes_per_window),
            image_width=int(image_width),
            image_height=int(image_height),
        )
        builder = D4RTNativeSceneBuilder(
            object(),
            {"model": {"input": {"clip_frames": 32}}},
            temporal_chunk_size=32,
            temporal_chunk_stride=16,
        )
        records = chunks_to_records(builder.stitch_to_canonical(chunks))
        records_by_id = {int(tube.tube_id): tube for tube in records}
        stream = ScanNetStream(seq_name=scene)
        frame_ids = sorted(
            {int(v) for tube in records for v in np.asarray(tube.target_frames_global, dtype=np.int64).tolist()}
        )
        masks_by_frame = {frame_id: stream.load_mask(frame_id) for frame_id in frame_ids}
        measurements, meas_diag = build_measurement_bank(
            records,
            masks_by_frame=masks_by_frame,
            min_visibility=0.5,
            min_confidence=0.5,
        )
        measurements_by_frame_mask = {(int(meas.frame_global), int(meas.mask_id)): meas for meas in measurements}
        rgb_cache: dict[int, np.ndarray] = {}
        appearance_cache: dict[int, np.ndarray | None] = {}
        annotated = 0
        eligible = 0
        stats = scene_stats.get(scene, {})
        for row in rows:
            ids = _core_ids(row)
            core = len(ids)
            ptype = str(row.get("proposal_type") or "")
            if not (
                ids
                and not ptype.startswith(TEMPORAL_PREFIXES)
                and 2 <= core <= 20
                and _float(row, "eroded_interior_ratio") >= 0.60
                and _float(row, "visible_outside_negative_rate")
                <= stats.get("visible_outside_negative_rate_p80", float("inf"))
                and _float(row, "boundary_risk") <= stats.get("boundary_risk_p80", float("inf"))
                and _float(row, "tube_canonical_compactness") <= stats.get("tube_canonical_compactness_p80", float("inf"))
            ):
                continue
            eligible += 1
            features: list[np.ndarray] = []
            for tid in ids:
                tube = records_by_id.get(int(tid))
                if tube is None:
                    continue
                if int(tid) not in appearance_cache:
                    appearance_cache[int(tid)] = _appearance_feature(stream, tube, rgb_cache)
                feature = appearance_cache[int(tid)]
                if feature is not None:
                    features.append(feature)
            rgb_mean, rgb_p10 = _feature_pair_stats(features)
            meas = measurements_by_frame_mask.get((_int(row, "frame_id"), _int(row, "mask_id")))
            support = 0
            boundary_safe: list[float] = []
            boundary_distances: list[float] = []
            if meas is not None:
                inside = {int(v) for v in meas.inside_tube_ids}
                support = int(sum(1 for tid in ids if int(tid) in inside))
                for tid in ids:
                    tid = int(tid)
                    if tid not in meas.mask_distance_to_boundary_per_tube:
                        continue
                    dist = float(meas.mask_distance_to_boundary_per_tube[tid])
                    boundary_distances.append(dist)
                    boundary_safe.append(1.0 if bool(meas.mask_eroded_interior_flag_per_tube.get(tid, False)) else 0.0)
            native_fields = {
                "native_record_present_ratio": float(sum(1 for tid in ids if int(tid) in records_by_id) / max(len(ids), 1)),
                "native_rgb_valid_ratio": float(len(features) / max(len(ids), 1)),
                "native_rgb_pair_cos_mean": rgb_mean,
                "native_rgb_pair_cos_p10": rgb_p10,
                "native_frame_mask_measurement_exists": bool(meas is not None),
                "native_frame_mask_support_ratio": float(support / max(len(ids), 1)),
                "native_boundary_safe_ratio": float(np.mean(boundary_safe)) if boundary_safe else None,
                "native_boundary_distance_mean": float(np.mean(boundary_distances)) if boundary_distances else None,
                "native_boundary_distance_p10": float(np.quantile(np.asarray(boundary_distances), 0.10))
                if boundary_distances
                else None,
            }
            row.update(native_fields)
            feature_rows.append(
                {
                    "proposal_id": row.get("proposal_id"),
                    "scene": scene,
                    "proposal_type": row.get("proposal_type"),
                    "frame_id": row.get("frame_id"),
                    "mask_id": row.get("mask_id"),
                    "core_tube_count": int(core),
                    **native_fields,
                }
            )
            annotated += 1
        scene_rows.append(
            {
                "scene": scene,
                "status": "ok",
                "candidate_count": int(len(rows)),
                "eligible_prefilter_candidate_count": int(eligible),
                "annotated_candidate_count": int(annotated),
                "record_count": int(len(records)),
                "measurement_count": int(len(measurements)),
                "same_mask_pair_count": int(meas_diag.get("same_mask_pair_count", 0)),
                "visible_outside_negative_pair_count": int(meas_diag.get("num_visible_outside_negative_pairs", 0)),
                "uses_gt_for_feature_generation": False,
                "uses_rgb_for_feature_generation": True,
                "uses_image_masks_for_feature_generation": True,
            }
        )
    return feature_rows, scene_rows


def _aggregate_candidate_row(
    *,
    scene: str,
    proposal_id: str,
    proposal_type: str,
    core_ids: set[int],
    contributors: list[dict[str, Any]],
    seed_source_hint: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "proposal_id": proposal_id,
        "scene": scene,
        "proposal_type": proposal_type,
        "seed_source_hint": seed_source_hint,
        "source_proposal_ids": ";".join(str(item.get("proposal_id")) for item in contributors),
        "source_proposal_count": int(len(contributors)),
        "frame_id": _int(contributors[0], "frame_id") if contributors else 0,
        "mask_id": _int(contributors[0], "mask_id") if contributors else 0,
    }
    _set_core_ids(row, tuple(sorted(core_ids)))
    for key in (
        "appearance_variance",
        "boundary_contact_ratio",
        "boundary_risk",
        "confidence_mean",
        "eroded_interior_ratio",
        "image_gradient_boundary_score",
        "mask_area",
        "mask_distance_mean",
        "mask_temporal_repeat_score",
        "overlap_with_other_proposals",
        "proposal_area",
        "proposal_area_ratio",
        "region_area",
        "same_frame_cannot_link_rate",
        "tube_canonical_compactness",
        "tube_density",
        "tube_temporal_length_mean",
        "tube_xy_compactness",
        "visibility_mean",
        "visible_outside_negative_rate",
    ):
        vals = [_float(source, key) for source in contributors if source.get(key) not in {None, ""}]
        row[key] = _mean(vals) if vals else 0.0
    return row


def _build_broad_agreement_seed_candidates(
    scene: str,
    rows: list[dict[str, Any]],
    broad_rows: list[dict[str, Any]],
    scene_stats: dict[str, dict[str, float]],
    *,
    expand_union: bool = False,
) -> list[dict[str, Any]]:
    stats = scene_stats[scene]
    max_child_core = max(8.0, min(stats.get("core_p80", 80.0), 96.0))
    child_rows: list[dict[str, Any]] = []
    for row in rows:
        ptype = str(row.get("proposal_type") or "")
        core = len(_core_ids(row))
        if core < 2 or core > max_child_core or ptype.startswith(TEMPORAL_PREFIXES):
            continue
        if _float(row, "eroded_interior_ratio") < stats.get("eroded_interior_ratio_p40", 0.0):
            continue
        if _float(row, "tube_canonical_compactness") > stats.get("tube_canonical_compactness_p70", float("inf")):
            continue
        if _float(row, "appearance_variance") > stats.get("appearance_variance_p70", float("inf")):
            continue
        if _float(row, "boundary_risk") > stats.get("boundary_risk_p80", float("inf")):
            continue
        child_rows.append(row)

    child_core_sets = {str(row.get("proposal_id")): set(_core_ids(row)) for row in child_rows}
    synthetic: list[dict[str, Any]] = []
    seen_cores: list[set[int]] = []
    for broad_idx, broad in enumerate(broad_rows):
        broad_ids = set(_core_ids(broad))
        if len(broad_ids) < 12:
            continue
        inner = [
            row
            for row in child_rows
            if len(child_core_sets[str(row.get("proposal_id"))] & broad_ids) / max(len(child_core_sets[str(row.get("proposal_id"))]), 1)
            >= 0.60
        ]
        if len(inner) < 3:
            continue
        ranked_inner = sorted(inner, key=lambda item: _normalized_seed_quality(item, scene_stats), reverse=True)[:60]
        for anchor_idx, anchor in enumerate(ranked_inner[:36]):
            anchor_ids = child_core_sets[str(anchor.get("proposal_id"))] & broad_ids
            if len(anchor_ids) < 3:
                continue
            neighbors_with_ids: list[tuple[dict[str, Any], set[int], float]] = []
            for item in ranked_inner:
                item_ids = child_core_sets[str(item.get("proposal_id"))] & broad_ids
                if _proposal_overlap(anchor_ids, item_ids) < 0.35:
                    continue
                weight = max(_normalized_seed_quality(item, scene_stats) + 1.0, 0.05)
                neighbors_with_ids.append((item, item_ids, weight))
            if len(neighbors_with_ids) < 2:
                continue
            if expand_union:
                support_ids = set(anchor_ids)
                for _, item_ids, _ in sorted(neighbors_with_ids, key=lambda item: item[2], reverse=True)[:24]:
                    support_ids.update(item_ids)
            else:
                support_ids = set(anchor_ids)
            vote_count: Counter[int] = Counter()
            vote_weight: Counter[int] = Counter()
            for _, item_ids, weight in neighbors_with_ids:
                for tid in support_ids & item_ids:
                    vote_count[int(tid)] += 1
                    vote_weight[int(tid)] += weight
            min_votes = max(2, min(5 if expand_union else 4, int(math.ceil((0.18 if expand_union else 0.22) * len(neighbors_with_ids)))))
            core_ids = {tid for tid in support_ids if vote_count[int(tid)] >= min_votes}
            if len(core_ids) < 2:
                ranked_tubes = sorted(support_ids, key=lambda tid: (vote_count[int(tid)], vote_weight[int(tid)]), reverse=True)
                keep_fraction = 0.70 if expand_union else 0.55
                core_ids = set(ranked_tubes[: max(2, min(len(ranked_tubes), int(math.ceil(keep_fraction * len(anchor_ids)))))])
            max_core = max(8, int(min(stats.get("core_p80" if expand_union else "core_p70", 60.0), 120.0 if expand_union else 80.0)))
            if len(core_ids) > max_core:
                core_ids = set(
                    sorted(core_ids, key=lambda tid: (vote_count[int(tid)], vote_weight[int(tid)]), reverse=True)[:max_core]
                )
            if len(core_ids) < (4 if expand_union else 2):
                continue
            if any(_proposal_overlap(core_ids, existing) >= 0.88 for existing in seen_cores):
                continue
            contributors = [
                item
                for item, _, _ in sorted(neighbors_with_ids, key=lambda entry: _normalized_seed_quality(entry[0], scene_stats), reverse=True)[:8]
            ]
            source_hint = "S9_broad_supported_union_core" if expand_union else "S8_broad_agreement_core"
            proposal_type = "R31_broad_supported_union_core" if expand_union else "R30_broad_agreement_core"
            out = _aggregate_candidate_row(
                scene=scene,
                proposal_id=f"{scene}_v30_{'broad_supported_union' if expand_union else 'broad_agreement'}_{broad_idx:04d}_{anchor_idx:04d}",
                proposal_type=proposal_type,
                core_ids=core_ids,
                contributors=contributors,
                seed_source_hint=source_hint,
            )
            out["agreement_neighbor_count"] = int(len(neighbors_with_ids))
            out["agreement_vote_mean"] = _mean([float(vote_count[int(tid)]) for tid in core_ids])
            out["agreement_core_fraction"] = float(len(core_ids) / max(len(anchor_ids), 1))
            synthetic.append(out)
            seen_cores.append(core_ids)
    return synthetic


def _build_broad_cooccurrence_seed_candidates(
    scene: str,
    rows: list[dict[str, Any]],
    broad_rows: list[dict[str, Any]],
    scene_stats: dict[str, dict[str, float]],
    *,
    edge_vote_ratio: float = 0.10,
    max_edge_votes: int = 7,
    min_component_size: int = 5,
) -> list[dict[str, Any]]:
    stats = scene_stats[scene]
    max_child_core = max(8.0, min(stats.get("core_p80", 80.0), 96.0))
    child_rows: list[dict[str, Any]] = []
    for row in rows:
        ptype = str(row.get("proposal_type") or "")
        core = len(_core_ids(row))
        if core < 3 or core > max_child_core or ptype.startswith(TEMPORAL_PREFIXES):
            continue
        if _float(row, "tube_canonical_compactness") > stats.get("tube_canonical_compactness_p75", float("inf")):
            continue
        if _float(row, "appearance_variance") > stats.get("appearance_variance_p75", float("inf")):
            continue
        if _float(row, "boundary_risk") > stats.get("boundary_risk_p80", float("inf")):
            continue
        child_rows.append(row)

    child_core_sets = {str(row.get("proposal_id")): set(_core_ids(row)) for row in child_rows}
    synthetic: list[dict[str, Any]] = []
    seen_cores: list[set[int]] = []
    for broad_idx, broad in enumerate(broad_rows):
        broad_ids = set(_core_ids(broad))
        if len(broad_ids) < 12:
            continue
        inner = [
            row
            for row in child_rows
            if len(child_core_sets[str(row.get("proposal_id"))] & broad_ids) / max(len(child_core_sets[str(row.get("proposal_id"))]), 1)
            >= 0.55
        ]
        if len(inner) < 4:
            continue
        ranked_inner = sorted(inner, key=lambda item: _normalized_seed_quality(item, scene_stats), reverse=True)[:80]
        node_votes: Counter[int] = Counter()
        edge_votes: Counter[tuple[int, int]] = Counter()
        for item in ranked_inner:
            ids = sorted(child_core_sets[str(item.get("proposal_id"))] & broad_ids)
            if len(ids) < 2:
                continue
            for tid in ids:
                node_votes[int(tid)] += 1
            for idx, a in enumerate(ids):
                for b in ids[idx + 1 :]:
                    edge_votes[(int(a), int(b))] += 1
        if not edge_votes:
            continue
        min_edge_votes = max(2, min(int(max_edge_votes), int(math.ceil(float(edge_vote_ratio) * len(ranked_inner)))))
        adjacency: dict[int, set[int]] = defaultdict(set)
        for (a, b), votes in edge_votes.items():
            if votes < min_edge_votes:
                continue
            adjacency[a].add(b)
            adjacency[b].add(a)
        visited: set[int] = set()
        components: list[set[int]] = []
        for tid in sorted(adjacency):
            if tid in visited:
                continue
            stack = [tid]
            comp: set[int] = set()
            visited.add(tid)
            while stack:
                cur = stack.pop()
                comp.add(cur)
                for nxt in adjacency.get(cur, set()):
                    if nxt in visited:
                        continue
                    visited.add(nxt)
                    stack.append(nxt)
            components.append(comp)
        max_core = max(8, int(min(stats.get("core_p80", 60.0), 120.0)))
        for comp_idx, comp in enumerate(sorted(components, key=lambda item: (-len(item), sorted(item)[:1]))[:24]):
            if len(comp) < int(min_component_size):
                continue
            core_ids = set(comp)
            if len(core_ids) > max_core:
                core_ids = set(sorted(core_ids, key=lambda tid: node_votes[int(tid)], reverse=True)[:max_core])
            if any(_proposal_overlap(core_ids, existing) >= 0.88 for existing in seen_cores):
                continue
            contributors = [
                item
                for item in sorted(
                    ranked_inner,
                    key=lambda source: len(core_ids & child_core_sets[str(source.get("proposal_id"))])
                    / max(len(core_ids), 1),
                    reverse=True,
                )
                if len(core_ids & child_core_sets[str(item.get("proposal_id"))]) > 0
            ][:8]
            if len(contributors) < 2:
                continue
            out = _aggregate_candidate_row(
                scene=scene,
                proposal_id=f"{scene}_v30_broad_cooccurrence_{broad_idx:04d}_{comp_idx:04d}",
                proposal_type="R32_broad_cooccurrence_component",
                core_ids=core_ids,
                contributors=contributors,
                seed_source_hint="S10_broad_cooccurrence_component",
            )
            out["cooccurrence_component_size"] = int(len(core_ids))
            out["cooccurrence_min_edge_votes"] = int(min_edge_votes)
            out["cooccurrence_node_vote_mean"] = _mean([float(node_votes[int(tid)]) for tid in core_ids])
            synthetic.append(out)
            seen_cores.append(core_ids)
    return synthetic


def _load_medium_rows(path: Path, source_by_id: dict[str, dict[str, Any]], gt_by_scene: dict[str, dict[int, int]]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    per_scene_index: Counter[str] = Counter()
    for item in _read_csv_dicts(path):
        scene = str(item.get("scene"))
        source_ids = [sid for sid in str(item.get("source_proposal_ids") or "").split(";") if sid]
        sources = [source_by_id[sid] for sid in source_ids if sid in source_by_id]
        if not sources or scene not in gt_by_scene:
            continue
        core_ids = sorted({tid for source in sources for tid in _core_ids(source)})
        if not core_ids:
            continue
        row: dict[str, Any] = {
            "proposal_id": str(item.get("proposal_id") or f"{scene}_v30_medium_{per_scene_index[scene]:06d}"),
            "scene": scene,
            "proposal_type": str(item.get("proposal_type") or "R13_v29_medium"),
            "seed_source_hint": "S6_v29_medium",
            "frame_id": _int(sources[0], "frame_id"),
            "mask_id": _int(sources[0], "mask_id"),
            "source_proposal_ids": ";".join(source_ids),
            "source_proposal_count": int(len(sources)),
        }
        _set_core_ids(row, tuple(core_ids))
        for key in (
            "appearance_variance",
            "boundary_contact_ratio",
            "boundary_risk",
            "confidence_mean",
            "eroded_interior_ratio",
            "image_gradient_boundary_score",
            "mask_area",
            "mask_distance_mean",
            "mask_temporal_repeat_score",
            "overlap_with_other_proposals",
            "proposal_area",
            "proposal_area_ratio",
            "region_area",
            "same_frame_cannot_link_rate",
            "tube_canonical_compactness",
            "tube_density",
            "tube_temporal_length_mean",
            "tube_xy_compactness",
            "visibility_mean",
            "visible_outside_negative_rate",
        ):
            vals = [_float(source, key) for source in sources]
            row[key] = _mean(vals) if vals else 0.0
        gt_counts = Counter(int(gt) for gt in gt_by_scene[scene].values() if int(gt) > 0)
        _refresh_diagnostic_gt_fields(row, gt_by_scene[scene], gt_counts)
        per_scene_index[scene] += 1
        rows.append(row)
    return rows


def _select_seed_candidates(
    rows_by_scene: dict[str, list[dict[str, Any]]],
    broad_rows_by_scene: dict[str, list[dict[str, Any]]],
    scene_stats: dict[str, dict[str, float]],
    *,
    profile: str,
) -> list[dict[str, Any]]:
    seed_rows: list[dict[str, Any]] = []
    profile_relax = {
        "strict": 0.0,
        "coverage_repair": 0.20,
        "purity_repair": -0.15,
        "consensus_repair": -0.05,
        "support_expand_repair": -0.02,
        "cooccurrence_repair": -0.05,
        "cooccurrence_loose_repair": -0.02,
        "overlap_support_repair": -0.02,
        "dense_eroded_repair": 0.20,
        NATIVE_RGB_BOUNDARY_PROFILE: 0.20,
    }[profile]
    caps = {
        "strict": {"per_source": 80, "total": 220},
        "coverage_repair": {"per_source": 150, "total": 420},
        "purity_repair": {"per_source": 70, "total": 180},
        "consensus_repair": {"per_source": 220, "total": 620},
        "support_expand_repair": {"per_source": 260, "total": 760},
        "cooccurrence_repair": {"per_source": 260, "total": 760},
        "cooccurrence_loose_repair": {"per_source": 320, "total": 860},
        "overlap_support_repair": {"per_source": 360, "total": 860},
        "dense_eroded_repair": {"per_source": 6000, "total": 6000},
        NATIVE_RGB_BOUNDARY_PROFILE: {"per_source": 6000, "total": 6000},
    }[profile]
    broad_repair_profiles = {
        "consensus_repair",
        "support_expand_repair",
        "cooccurrence_repair",
        "cooccurrence_loose_repair",
        "overlap_support_repair",
        "dense_eroded_repair",
        NATIVE_RGB_BOUNDARY_PROFILE,
    }
    for scene, rows in rows_by_scene.items():
        stats = scene_stats[scene]
        max_seed_core = max(
            10.0,
            min(
                stats["core_p80"] * (1.0 + profile_relax),
                120.0
                if profile not in {"coverage_repair", *broad_repair_profiles}
                else 180.0,
            ),
        )
        source_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            ptype = str(row.get("proposal_type") or "")
            core = len(_core_ids(row))
            if core < 2 or core > max_seed_core:
                continue
            area_norm_cannot = _float(row, "same_frame_cannot_link_rate") / max(_float(row, "proposal_area_ratio", 1e-6), 1e-6)
            risk_ok = (
                _float(row, "visible_outside_negative_rate") <= stats["visible_outside_negative_rate_p75"] * (1.15 + profile_relax)
                and _float(row, "boundary_risk") <= stats["boundary_risk_p75"] * (1.15 + profile_relax)
                and area_norm_cannot <= 10.0 * (1.25 + profile_relax)
            )
            if profile in broad_repair_profiles:
                risk_ok = (
                    _float(row, "visible_outside_negative_rate") <= stats["visible_outside_negative_rate_p70"] * 1.08
                    and _float(row, "boundary_risk") <= stats["boundary_risk_p70"] * 1.05
                    and _float(row, "tube_canonical_compactness") <= stats["tube_canonical_compactness_p70"]
                    and _float(row, "appearance_variance") <= stats["appearance_variance_p70"]
                )
            if profile in {"support_expand_repair", "cooccurrence_repair", "cooccurrence_loose_repair", "overlap_support_repair"} and core < 5:
                continue
            if (
                profile not in {"dense_eroded_repair", NATIVE_RGB_BOUNDARY_PROFILE}
                and ptype.startswith(("R1_", "R7_"))
                and risk_ok
                and _float(row, "eroded_interior_ratio") >= stats["eroded_interior_ratio_p40"]
            ):
                source_candidates["S0_high_purity_eroded_core"].append(row)
            if (
                profile not in broad_repair_profiles
                and ptype.startswith("R2_")
                and risk_ok
                and _float(row, "confidence_mean", 0.5) >= stats["confidence_mean_p40"]
            ):
                source_candidates["S1_high_confidence_watershed"].append(row)
            if (
                profile not in {"dense_eroded_repair", NATIVE_RGB_BOUNDARY_PROFILE}
                and ptype.startswith(("R3_", "R5_"))
                and risk_ok
                and _float(row, "tube_canonical_compactness") <= stats["tube_canonical_compactness_p70"] * (1.1 + profile_relax)
            ):
                source_candidates["S2_d4rt_compact_region"].append(row)
            if (
                profile not in {"dense_eroded_repair", NATIVE_RGB_BOUNDARY_PROFILE}
                and ptype.startswith("R6_")
                and risk_ok
                and _float(row, "mask_temporal_repeat_score") >= stats["mask_temporal_repeat_score_p40"]
            ):
                source_candidates["S3_mask_overlap_consensus_core"].append(row)
            if (
                profile not in broad_repair_profiles
                and ptype.startswith(("R9_", "R10_"))
                and risk_ok
                and core <= stats["core_p70"] * (1.2 + profile_relax)
            ):
                source_candidates["S4_temporal_stable_small_core"].append(row)
            if (
                profile not in {"dense_eroded_repair", NATIVE_RGB_BOUNDARY_PROFILE}
                and
                risk_ok
                and _float(row, "appearance_variance") <= stats["appearance_variance_p50"] * (1.1 + profile_relax)
                and _float(row, "tube_canonical_compactness") <= stats["tube_canonical_compactness_p50"] * (1.1 + profile_relax)
            ):
                source_candidates["S5_appearance_compact_core"].append(row)
            if profile not in broad_repair_profiles and ptype.startswith("R13_") and risk_ok:
                source_candidates["S6_v29_medium_high_prior"].append(row)
            if (
                profile == "overlap_support_repair"
                and not ptype.startswith(TEMPORAL_PREFIXES)
                and risk_ok
                and _float(row, "overlap_with_other_proposals") >= stats["overlap_with_other_proposals_p60"]
                and _float(row, "mask_distance_mean") >= stats["mask_distance_mean_p40"]
                and _float(row, "visibility_mean", 0.5) >= stats["visibility_mean_p40"]
            ):
                source_candidates["S12_overlap_supported_seed"].append(row)
            if (
                profile == "dense_eroded_repair"
                and not ptype.startswith(TEMPORAL_PREFIXES)
                and 2 <= core <= 120
                and _float(row, "eroded_interior_ratio") >= 0.60
                and _float(row, "visible_outside_negative_rate") <= stats["visible_outside_negative_rate_p80"]
                and _float(row, "boundary_risk") <= stats["boundary_risk_p80"]
                and _float(row, "tube_canonical_compactness") <= stats["tube_canonical_compactness_p80"]
            ):
                source_candidates["S13_dense_eroded_non_temporal_seed"].append(row)
            if (
                profile == NATIVE_RGB_BOUNDARY_PROFILE
                and not ptype.startswith(TEMPORAL_PREFIXES)
                and 2 <= core <= 20
                and _float(row, "eroded_interior_ratio") >= 0.60
                and _float(row, "visible_outside_negative_rate") <= stats["visible_outside_negative_rate_p80"]
                and _float(row, "boundary_risk") <= stats["boundary_risk_p80"]
                and _float(row, "tube_canonical_compactness") <= stats["tube_canonical_compactness_p80"]
                and _float(row, "native_rgb_pair_cos_p10", -1.0) >= 0.90
                and _float(row, "native_boundary_safe_ratio", -1.0) >= 1.0
                and _float(row, "native_boundary_distance_p10", -1.0) >= 4.0
            ):
                source_candidates["S14_native_rgb_boundary_seed"].append(row)

        if profile == "coverage_repair":
            for broad in broad_rows_by_scene.get(scene, []):
                broad_ids = set(_core_ids(broad))
                inner = [
                    row
                    for row in rows
                    if row is not broad
                    and len(_core_ids(row)) >= 2
                    and len(_core_ids(row)) <= max_seed_core
                    and len(set(_core_ids(row)) & broad_ids) / max(len(_core_ids(row)), 1) >= 0.65
                    and not str(row.get("proposal_type", "")).startswith(TEMPORAL_PREFIXES)
                ]
                source_candidates["S7_broad_internal_reseed"].extend(sorted(inner, key=lambda item: _quality_proxy(item, scene_stats), reverse=True)[:8])
        if profile == "consensus_repair":
            source_candidates["S8_broad_agreement_core"].extend(
                _build_broad_agreement_seed_candidates(scene, rows, broad_rows_by_scene.get(scene, []), scene_stats)
            )
        if profile == "support_expand_repair":
            source_candidates["S9_broad_supported_union_core"].extend(
                _build_broad_agreement_seed_candidates(
                    scene,
                    rows,
                    broad_rows_by_scene.get(scene, []),
                    scene_stats,
                    expand_union=True,
                )
            )
        if profile == "cooccurrence_repair":
            source_candidates["S10_broad_cooccurrence_component"].extend(
                _build_broad_cooccurrence_seed_candidates(scene, rows, broad_rows_by_scene.get(scene, []), scene_stats)
            )
        if profile == "cooccurrence_loose_repair":
            source_candidates["S11_broad_cooccurrence_loose_component"].extend(
                _build_broad_cooccurrence_seed_candidates(
                    scene,
                    rows,
                    broad_rows_by_scene.get(scene, []),
                    scene_stats,
                    edge_vote_ratio=0.04,
                    max_edge_votes=4,
                    min_component_size=4,
                )
            )

        ranked_scene: list[tuple[float, str, dict[str, Any]]] = []
        for source, items in source_candidates.items():
            seen_ids: set[str] = set()
            rank_fn = (
                _overlap_supported_seed_quality
                if profile == "overlap_support_repair"
                else _native_rgb_boundary_seed_quality
                if profile == NATIVE_RGB_BOUNDARY_PROFILE
                else _normalized_seed_quality
                if profile in broad_repair_profiles
                else _quality_proxy
            )
            ranked = sorted(items, key=lambda item: rank_fn(item, scene_stats), reverse=True)
            for row in ranked[: caps["per_source"]]:
                pid = str(row.get("proposal_id"))
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)
                ranked_scene.append((rank_fn(row, scene_stats), source, row))
        ranked_scene.sort(key=lambda item: item[0], reverse=True)
        accepted: list[set[int]] = []
        for score, source, row in ranked_scene:
            ids = set(_core_ids(row))
            if not ids:
                continue
            duplicate = False
            for existing in accepted:
                if len(ids & existing) / max(min(len(ids), len(existing)), 1) >= 0.88:
                    duplicate = True
                    break
            if duplicate:
                continue
            out = dict(row)
            out["seed_id"] = f"{scene}_{source}_{len(accepted):05d}"
            out["seed_source"] = source
            out["seed_score"] = float(score)
            accepted.append(ids)
            seed_rows.append(out)
            if len(accepted) >= caps["total"]:
                break
    return seed_rows


def _seed_summary(seed_rows: list[dict[str, Any]], scenes: list[str], gt_by_scene: dict[str, dict[int, int]], broad_rows_by_scene: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_source: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for seed in seed_rows:
        gt_labels = gt_by_scene[str(seed["scene"])]
        gt_counts = Counter(int(gt) for gt in gt_labels.values() if int(gt) > 0)
        _refresh_diagnostic_gt_fields(seed, gt_labels, gt_counts)
        counts, labeled = _fresh_proposal_gt_counts(seed, gt_labels)
        seed["seed_dominant_GT"] = int(counts.most_common(1)[0][0]) if counts else 0
        seed["seed_dominant_GT_ratio"] = float(counts.most_common(1)[0][1] / max(labeled, 1)) if counts else None
        seed_ids = set(_core_ids(seed))
        seed["overlaps_broad_observation"] = bool(
            any(len(seed_ids & set(_core_ids(broad))) > 0 for broad in broad_rows_by_scene.get(str(seed["scene"]), []))
        )
    for source in sorted({str(row["seed_source"]) for row in seed_rows}):
        items = [row for row in seed_rows if row["seed_source"] == source]
        per_source.append(
            {
                "seed_source": source,
                "count": int(len(items)),
                "purity_mean": _mean([_float(row, "proposal_purity") for row in items]),
                "purity_p10": _quantile([_float(row, "proposal_purity") for row in items], 0.10),
                "best_IoU_mean": _mean([_float(row, "proposal_best_IoU") for row in items]),
                "core_tube_count_p50": _quantile([len(_core_ids(row)) for row in items], 0.50),
                "core_tube_count_p90": _quantile([len(_core_ids(row)) for row in items], 0.90),
                "broad_overlap_count": int(sum(1 for row in items if row.get("overlaps_broad_observation"))),
            }
        )
    for scene in scenes:
        items = [row for row in seed_rows if row["scene"] == scene]
        gt_counts = Counter(int(gt) for gt in gt_by_scene[scene].values() if int(gt) > 0)
        gt_best_iou: dict[int, float] = {int(gt): 0.0 for gt in gt_counts}
        for seed in items:
            counts, labeled = _fresh_proposal_gt_counts(seed, gt_by_scene[scene])
            for gt, overlap in counts.items():
                if int(gt) not in gt_counts:
                    continue
                iou = float(overlap / max(labeled + int(gt_counts[int(gt)]) - overlap, 1))
                gt_best_iou[int(gt)] = max(gt_best_iou[int(gt)], iou)
        duplicate_pairs = 0
        checked_pairs = 0
        for idx, a in enumerate(items):
            a_ids = set(_core_ids(a))
            for b in items[idx + 1 :]:
                checked_pairs += 1
                b_ids = set(_core_ids(b))
                if len(a_ids & b_ids) / max(min(len(a_ids), len(b_ids)), 1) >= 0.88:
                    duplicate_pairs += 1
        owned = set().union(*(set(_core_ids(row)) for row in items)) if items else set()
        labeled = {int(tid) for tid, gt in gt_by_scene[scene].items() if int(gt) > 0}
        summary_rows.append(
            {
                "scene": scene,
                "seed_count": int(len(items)),
                "seed_core_tube_count_p50": _quantile([len(_core_ids(row)) for row in items], 0.50),
                "seed_core_tube_count_p90": _quantile([len(_core_ids(row)) for row in items], 0.90),
                "seed_overlap_with_broad_count": int(sum(1 for row in items if row.get("overlaps_broad_observation"))),
                "seed_duplicate_rate": float(duplicate_pairs / max(checked_pairs, 1)),
                "seed_unknown_coverage": float(1.0 - len(owned & labeled) / max(len(labeled), 1)),
                "seed_purity_mean": _mean([_float(row, "proposal_purity") for row in items]),
                "seed_purity_p10": _quantile([_float(row, "proposal_purity") for row in items], 0.10),
                "seed_best_IoU_mean": _mean([_float(row, "proposal_best_IoU") for row in items]),
                "GT_with_seed_IoU_ge_0.10": float(sum(1 for val in gt_best_iou.values() if val >= 0.10) / max(len(gt_best_iou), 1)),
                "GT_with_seed_IoU_ge_0.25": float(sum(1 for val in gt_best_iou.values() if val >= 0.25) / max(len(gt_best_iou), 1)),
                "seed_overmerge_rate": float(sum(1 for row in items if _float(row, "proposal_purity", 0.0) < 0.75) / max(len(items), 1)),
            }
        )
    all_row = {
        "scene": "ALL",
        "seed_count": int(len(seed_rows)),
        "seed_count_per_scene": ";".join(f"{row['scene']}={row['seed_count']}" for row in summary_rows),
        "seed_core_tube_count_p50": _quantile([len(_core_ids(row)) for row in seed_rows], 0.50),
        "seed_core_tube_count_p90": _quantile([len(_core_ids(row)) for row in seed_rows], 0.90),
        "seed_overlap_with_broad_count": int(sum(int(row["seed_overlap_with_broad_count"]) for row in summary_rows)),
        "seed_duplicate_rate": _mean([float(row["seed_duplicate_rate"]) for row in summary_rows]),
        "seed_unknown_coverage": _mean([float(row["seed_unknown_coverage"]) for row in summary_rows]),
        "seed_purity_mean": _mean([_float(row, "proposal_purity") for row in seed_rows]),
        "seed_purity_p10": _quantile([_float(row, "proposal_purity") for row in seed_rows], 0.10),
        "seed_best_IoU_mean": _mean([_float(row, "proposal_best_IoU") for row in seed_rows]),
        "GT_with_seed_IoU_ge_0.10": _mean([float(row["GT_with_seed_IoU_ge_0.10"]) for row in summary_rows]),
        "GT_with_seed_IoU_ge_0.25": _mean([float(row["GT_with_seed_IoU_ge_0.25"]) for row in summary_rows]),
        "scene0081_GT_with_seed_IoU_ge_0.10": next(
            (row["GT_with_seed_IoU_ge_0.10"] for row in summary_rows if row["scene"] == "scene0081_01"),
            None,
        ),
        "seed_overmerge_rate": _mean([float(row["seed_overmerge_rate"]) for row in summary_rows]),
    }
    summary_rows.append(all_row)
    return summary_rows, per_source


def _build_broad_diagnostic(
    rows_by_scene: dict[str, list[dict[str, Any]]],
    gt_by_scene: dict[str, dict[int, int]],
    p11_ids: set[str],
    oracle_ids: set[str],
    scene_p80: dict[str, float],
    *,
    out_dir: Path,
    max_rejected_per_scene: int = 80,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    broad_rows: list[dict[str, Any]] = []
    broad_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scene, rows in rows_by_scene.items():
        gt_labels = gt_by_scene[scene]
        gt_counts = Counter(int(gt) for gt in gt_labels.values() if int(gt) > 0)
        selected = [row for row in rows if str(row.get("proposal_id")) in p11_ids]
        selected_by_id = {str(row.get("proposal_id")): row for row in selected}
        scene_baseline = _eval_selected(selected, gt_labels)
        broad_candidates = [row for row in rows if _is_broad_observation(row, scene_p80)]
        must_keep = [
            row
            for row in broad_candidates
            if str(row.get("proposal_id")) in p11_ids or str(row.get("proposal_id")) in oracle_ids
        ]
        rejected = [
            row
            for row in broad_candidates
            if str(row.get("proposal_id")) not in p11_ids and str(row.get("proposal_id")) not in oracle_ids
        ]
        rejected_sample = sorted(rejected, key=lambda item: len(_core_ids(item)), reverse=True)[: int(max_rejected_per_scene)]
        diagnostic_candidates = {str(row.get("proposal_id")): row for row in must_keep + rejected_sample}
        core_sets = {str(row.get("proposal_id")): set(_core_ids(row)) for row in rows}
        for row in diagnostic_candidates.values():
            pid = str(row.get("proposal_id"))
            broad_ids = core_sets[pid]
            child_rows = [
                child
                for child in rows
                if child is not row
                and not str(child.get("proposal_type", "")).startswith(TEMPORAL_PREFIXES)
                and len(core_sets[str(child.get("proposal_id"))]) < len(broad_ids)
                and len(core_sets[str(child.get("proposal_id"))] & broad_ids) / max(len(core_sets[str(child.get("proposal_id"))]), 1) >= 0.50
            ]
            seed_like_rows = [
                child
                for child in child_rows
                if str(child.get("proposal_type", "")).startswith(("R1_", "R2_", "R3_", "R5_", "R6_", "R7_"))
            ]
            ranked_children = sorted(child_rows, key=lambda item: (_float(item, "eroded_interior_ratio"), _float(item, "visibility_mean")), reverse=True)[:8]
            child_union = set().union(*(set(_core_ids(child)) for child in ranked_children)) if ranked_children else set()
            child_q = _proposal_union_quality(child_union, gt_labels, gt_counts) if child_union else {}
            drop_delta_ari = None
            drop_delta_comp = None
            if pid in selected_by_id:
                dropped = [item for item in selected if str(item.get("proposal_id")) != pid]
                dropped_metrics = _eval_selected(dropped, gt_labels)
                if dropped_metrics.get("ARI") is not None and scene_baseline.get("ARI") is not None:
                    drop_delta_ari = float(dropped_metrics["ARI"] - scene_baseline["ARI"])
                if dropped_metrics.get("completeness") is not None and scene_baseline.get("completeness") is not None:
                    drop_delta_comp = float(dropped_metrics["completeness"] - scene_baseline["completeness"])
            whole_q = _proposal_union_quality(broad_ids, gt_labels, gt_counts)
            mode = "child_selection_failure"
            if len(seed_like_rows) == 0 and len(child_rows) == 0:
                mode = "child_pool_missing"
            elif (drop_delta_ari is not None and drop_delta_ari > -0.03 and drop_delta_comp is not None and drop_delta_comp > -0.05 and _float(row, "proposal_purity", 1.0) < 0.75):
                mode = "removable_false_merge"
            elif child_q and child_q.get("purity") is not None and child_q["purity"] >= 0.85 and child_q.get("completeness", 0.0) >= 0.45:
                mode = "replaceable_by_children"
            elif (
                _float(row, "proposal_purity", 1.0) < 0.80
                and drop_delta_ari is not None
                and (drop_delta_ari <= -0.05 or (drop_delta_comp is not None and drop_delta_comp <= -0.10))
            ):
                mode = "essential_mixed_observation"
            broad_out = {
                "proposal_id": pid,
                "scene": scene,
                "proposal_type": row.get("proposal_type"),
                "core_tube_count": int(len(broad_ids)),
                "proposal_purity": row.get("proposal_purity"),
                "proposal_completeness": row.get("proposal_completeness"),
                "proposal_best_IoU": row.get("proposal_best_IoU"),
                "selected_by_P11": bool(pid in p11_ids),
                "selected_by_oracle": bool(pid in oracle_ids),
                "overlap_seed_count": int(len(seed_like_rows)),
                "overlap_child_count": int(len(child_rows)),
                "covered_GT_count": whole_q.get("covered_GT_count"),
                "dominant_GT_ratio": whole_q.get("dominant_GT_ratio"),
                "secondary_GT_ratio": whole_q.get("secondary_GT_ratio"),
                "purity_if_selected_whole": whole_q.get("purity"),
                "best_child_union_purity_diagnostic": child_q.get("purity"),
                "best_child_union_completeness_diagnostic": child_q.get("completeness"),
                "drop_effect_on_ARI_diagnostic": drop_delta_ari,
                "drop_effect_on_completeness_diagnostic": drop_delta_comp,
                "mode": mode,
            }
            broad_rows.append(broad_out)
            broad_by_scene[scene].append(row)
    _write_csv(out_dir / "broad_proposal_rows.csv", broad_rows)
    mode_rows = []
    for mode in sorted({str(row["mode"]) for row in broad_rows}):
        items = [row for row in broad_rows if row["mode"] == mode]
        mode_rows.append(
            {
                "mode": mode,
                "count": int(len(items)),
                "selected_by_P11_count": int(sum(1 for row in items if row["selected_by_P11"])),
                "purity_mean": _mean([_float(row, "proposal_purity") for row in items]),
                "drop_ARI_mean": _mean([_float(row, "drop_effect_on_ARI_diagnostic") for row in items if row.get("drop_effect_on_ARI_diagnostic") is not None]),
                "drop_completeness_mean": _mean([_float(row, "drop_effect_on_completeness_diagnostic") for row in items if row.get("drop_effect_on_completeness_diagnostic") is not None]),
            }
        )
    _write_csv(out_dir / "broad_mode_summary.csv", mode_rows)
    if broad_rows:
        figures = out_dir / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        modes = [row["mode"] for row in mode_rows]
        counts = [row["count"] for row in mode_rows]
        plt.figure(figsize=(8, 3.2))
        plt.bar(modes, counts, color="#4C78A8")
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(figures / "broad_mode_counts.png", dpi=160)
        plt.close()
        _write_json(
            figures / "figure_manifest.json",
            {
                "figures": ["broad_mode_counts.png"],
                "uses_gt_for_visualization": True,
                "uses_gt_for_prediction": False,
            },
        )
    manifest = {
        **METHOD_MANIFEST_BASE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "v30_phaseB_broad_observation_diagnostic",
        "artifact_files": ["broad_proposal_rows.csv", "broad_mode_summary.csv"],
        "broad_count": int(len(broad_rows)),
        "rejected_broad_sampling": {
            "max_rejected_per_scene": int(max_rejected_per_scene),
            "selection_rule": "all P11/oracle broad rows plus largest rejected broad rows by core_tube_count per scene",
        },
        "mode_summary": mode_rows,
    }
    _write_json(out_dir / "manifest.json", manifest)
    return broad_rows, broad_by_scene


def _decomposition_oracle(
    scenes: list[str],
    rows_by_scene: dict[str, list[dict[str, Any]]],
    gt_by_scene: dict[str, dict[int, int]],
    seed_rows: list[dict[str, Any]],
    broad_by_scene: dict[str, list[dict[str, Any]]],
    p11_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for seed in seed_rows:
        seed_by_scene[str(seed["scene"])].append(seed)
    scene_rows: list[dict[str, Any]] = []
    for scene in scenes:
        gt_labels = gt_by_scene[scene]
        p11_selected = [row for row in rows_by_scene[scene] if str(row.get("proposal_id")) in p11_ids]
        p11_broad = [row for row in p11_selected if str(row.get("proposal_type", "")).startswith(TEMPORAL_PREFIXES)]
        non_broad = [row for row in p11_selected if row not in p11_broad]
        d0 = _eval_selected(p11_selected, gt_labels)
        d1 = _eval_selected(non_broad, gt_labels)

        replacement_rows: list[dict[str, Any]] = list(non_broad)
        replacement_ids = {str(row.get("proposal_id")) for row in replacement_rows}
        scene_stat_lookup = _scene_stats({scene: rows_by_scene[scene]})
        for broad in p11_broad:
            broad_ids = set(_core_ids(broad))
            candidates = [
                row
                for row in rows_by_scene[scene]
                if str(row.get("proposal_id")) not in replacement_ids
                and not str(row.get("proposal_type", "")).startswith(TEMPORAL_PREFIXES)
                and len(_core_ids(row)) < len(broad_ids)
                and len(set(_core_ids(row)) & broad_ids) / max(len(_core_ids(row)), 1) >= 0.55
            ]
            for child in sorted(candidates, key=lambda item: _quality_proxy(item, scene_stat_lookup), reverse=True)[:8]:
                replacement_rows.append(child)
                replacement_ids.add(str(child.get("proposal_id")))
        d2 = _eval_selected(replacement_rows, gt_labels)

        gt_counts = Counter(int(gt) for gt in gt_labels.values() if int(gt) > 0)
        gt_to_slot: dict[int, set[int]] = {}
        for seed in seed_by_scene[scene]:
            dominant = int(seed.get("seed_dominant_GT") or 0)
            if dominant <= 0:
                continue
            gt_to_slot.setdefault(dominant, set()).update(_core_ids(seed))
        for obs in broad_by_scene.get(scene, []):
            compatible_seed_gts: set[int] = set()
            obs_ids = set(_core_ids(obs))
            for seed in seed_by_scene[scene]:
                if len(obs_ids & set(_core_ids(seed))) > 0 and int(seed.get("seed_dominant_GT") or 0) > 0:
                    compatible_seed_gts.add(int(seed["seed_dominant_GT"]))
            for tid in obs_ids:
                gt = int(gt_labels.get(int(tid), 0))
                if gt > 0 and gt in compatible_seed_gts:
                    gt_to_slot.setdefault(gt, set()).add(int(tid))
        d3_slots = [ids for _, ids in sorted(gt_to_slot.items()) if ids]
        d3 = _eval_slot_sets(d3_slots, gt_labels)
        d4_slots = [{int(tid) for tid, label in gt_labels.items() if int(label) == gt} for gt in sorted(gt_counts)]
        d4 = _eval_slot_sets(d4_slots, gt_labels)

        assigned_gt = set()
        for slot in d3_slots:
            for tid in slot:
                gt = int(gt_labels.get(int(tid), 0))
                if gt > 0:
                    assigned_gt.add(gt)
        rows = [
            ("D0_select_whole_broad_P11", d0),
            ("D1_drop_broad_P11", d1),
            ("D2_replace_by_existing_children", d2),
            ("D3_seed_slot_decomposition_oracle", d3),
            ("D4_GT_full_oracle_forbidden", d4),
        ]
        for variant, metrics in rows:
            scene_rows.append(
                {
                    "scene": scene,
                    "variant": variant,
                    "ARI": metrics["ARI"],
                    "purity": metrics["purity"],
                    "completeness": metrics["completeness"],
                    "unknown_tube_ratio": metrics.get("unknown_tube_ratio"),
                    "owned_tube_ratio": metrics.get("owned_tube_ratio"),
                    "overmerge": metrics.get("overmerge"),
                    "oversplit": metrics.get("oversplit"),
                    "decomposition_oracle_ARI": metrics["ARI"] if variant == "D3_seed_slot_decomposition_oracle" else None,
                    "decomposition_oracle_purity": metrics["purity"] if variant == "D3_seed_slot_decomposition_oracle" else None,
                    "decomposition_oracle_completeness": metrics["completeness"] if variant == "D3_seed_slot_decomposition_oracle" else None,
                    "GT_coverage_after_decomposition": float(len(assigned_gt) / max(len(gt_counts), 1))
                    if variant == "D3_seed_slot_decomposition_oracle"
                    else None,
                    "slot_count": len(d3_slots) if variant == "D3_seed_slot_decomposition_oracle" else None,
                }
            )
    all_rows = _aggregate_variant(scene_rows)
    summary_rows = scene_rows + all_rows
    all_d3 = next(row for row in all_rows if row["variant"] == "D3_seed_slot_decomposition_oracle")
    gate = {
        "D3_ARI_ge_0.40": bool(float(all_d3.get("ARI") or 0.0) >= 0.40),
        "D3_purity_ge_0.85": bool(float(all_d3.get("purity") or 0.0) >= 0.85),
        "D3_completeness_ge_0.50": bool(float(all_d3.get("completeness") or 0.0) >= 0.50),
        "D3_scene0081_ARI_ge_0.20": bool(float(all_d3.get("scene0081_ARI") or 0.0) >= 0.20),
    }
    d0 = next(row for row in all_rows if row["variant"] == "D0_select_whole_broad_P11")
    d1 = next(row for row in all_rows if row["variant"] == "D1_drop_broad_P11")
    gate["D3_improves_over_D0_ARI"] = bool(float(all_d3["ARI"]) > float(d0["ARI"]))
    gate["D3_improves_over_D1_ARI"] = bool(float(all_d3["ARI"]) > float(d1["ARI"]))
    gate["phaseD_gate_pass"] = all(gate.values())
    return summary_rows, gate


def run(args: argparse.Namespace) -> dict[str, Any]:
    audit_root = Path(args.audit_root)
    out_base = Path(args.out_base)
    out_base.mkdir(parents=True, exist_ok=True)
    proposal_root = audit_root / REAL_ROOT
    proposal_rows = json.loads((proposal_root / f"{REAL_ROOT}_proposal_rows.json").read_text(encoding="utf-8"))
    proposal_rows = [row for row in proposal_rows if _is_o5(row)]
    for row in proposal_rows:
        _set_core_ids(row, _core_ids(row))
    scenes = _read_split(Path(args.split))
    rows_by_scene: dict[str, list[dict[str, Any]]] = {scene: [] for scene in scenes}
    for row in proposal_rows:
        scene = str(row.get("scene"))
        if scene in rows_by_scene:
            rows_by_scene[scene].append(row)

    gt_by_scene: dict[str, dict[int, int]] = {}
    for scene in scenes:
        gt_by_scene[scene] = _load_gt_labels(
            Path(args.cache_root),
            scene,
            int(args.max_tubes_per_window),
            int(args.image_width),
            int(args.image_height),
        )
    for scene, rows in rows_by_scene.items():
        gt_counts = Counter(int(gt) for gt in gt_by_scene[scene].values() if int(gt) > 0)
        for row in rows:
            _refresh_diagnostic_gt_fields(row, gt_by_scene[scene], gt_counts)

    source_by_id = {str(row["proposal_id"]): row for row in proposal_rows}
    medium_rows = _load_medium_rows(audit_root / MEDIUM_ROOT / "medium_proposal_rows.csv", source_by_id, gt_by_scene)
    if medium_rows:
        for row in medium_rows:
            scene = str(row.get("scene"))
            if scene in rows_by_scene:
                rows_by_scene[scene].append(row)

    scene_stats = _scene_stats(rows_by_scene)
    native_feature_rows: list[dict[str, Any]] = []
    native_scene_rows: list[dict[str, Any]] = []
    if str(args.seed_profile) == NATIVE_RGB_BOUNDARY_PROFILE:
        native_feature_rows, native_scene_rows = _annotate_native_rgb_boundary_features(
            rows_by_scene=rows_by_scene,
            scene_stats=scene_stats,
            scenes=scenes,
            cache_root=Path(args.cache_root),
            max_tubes_per_window=int(args.max_tubes_per_window),
            image_width=int(args.image_width),
            image_height=int(args.image_height),
        )
        native_dir = out_base / "v30_native_rgb_boundary_features"
        _write_csv(native_dir / "candidate_feature_rows.csv", native_feature_rows)
        _write_csv(native_dir / "scene_summary.csv", native_scene_rows)
        _write_json(
            native_dir / "manifest.json",
            {
                **METHOD_MANIFEST_BASE,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "phase": "v30_native_rgb_boundary_feature_annotation",
                "seed_profile": str(args.seed_profile),
                "uses_rgb_for_prediction": True,
                "uses_image_masks_for_prediction": True,
                "uses_rgbd_for_prediction": True,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": False,
                "feature_thresholds": {
                    "native_rgb_pair_cos_p10_min": 0.90,
                    "native_boundary_safe_ratio_min": 1.0,
                    "native_boundary_distance_p10_min": 4.0,
                    "core_tube_count_max": 20,
                },
                "artifact_files": ["candidate_feature_rows.csv", "scene_summary.csv"],
            },
        )

    p11_ids = _selected_ids_from_csv(audit_root / P11_ROOT / f"{P11_ROOT}_selected_proposals.csv", "P11_calibrated_ownership_expansion")
    strict_ids = _selected_ids_from_csv(audit_root / STRICT_ROOT / f"{STRICT_ROOT}_selected_proposals.csv", "P4_greedy_set_packing")
    oracle_ids: set[str] = set()
    for scene in scenes:
        gt_counts = Counter(int(gt) for gt in gt_by_scene[scene].values() if int(gt) > 0)
        for gt in sorted(gt_counts):
            best_iou = -1.0
            best_row: dict[str, Any] | None = None
            for row in rows_by_scene[scene]:
                if not _is_o5(row):
                    continue
                counts, labeled = _fresh_proposal_gt_counts(row, gt_by_scene[scene])
                overlap = int(counts.get(int(gt), 0))
                iou = float(overlap / max(labeled + gt_counts[int(gt)] - overlap, 1))
                if iou > best_iou:
                    best_iou = iou
                    best_row = row
            if best_row is not None:
                oracle_ids.add(str(best_row["proposal_id"]))

    scene_p80 = {scene: stats["core_p80"] for scene, stats in scene_stats.items()}
    broad_dir = out_base / "v30_broad_observation_diagnostic"
    broad_rows, broad_by_scene = _build_broad_diagnostic(
        rows_by_scene,
        gt_by_scene,
        p11_ids,
        oracle_ids,
        scene_p80,
        out_dir=broad_dir,
    )
    seed_rows = _select_seed_candidates(rows_by_scene, broad_by_scene, scene_stats, profile=str(args.seed_profile))
    seed_summary_rows, seed_source_rows = _seed_summary(seed_rows, scenes, gt_by_scene, broad_by_scene)

    seed_dir = out_base / "v30_seed_slots"
    seed_export_rows = []
    for row in seed_rows:
        seed_export_rows.append(
            {
                "seed_id": row.get("seed_id"),
                "proposal_id": row.get("proposal_id"),
                "scene": row.get("scene"),
                "seed_source": row.get("seed_source"),
                "proposal_type": row.get("proposal_type"),
                "core_tube_count": len(_core_ids(row)),
                "seed_score": row.get("seed_score"),
                "seed_dominant_GT": row.get("seed_dominant_GT"),
                "seed_dominant_GT_ratio": row.get("seed_dominant_GT_ratio"),
                "proposal_purity_diagnostic": row.get("proposal_purity"),
                "proposal_best_IoU_diagnostic": row.get("proposal_best_IoU"),
                "proposal_completeness_diagnostic": row.get("proposal_completeness"),
                "overlaps_broad_observation": row.get("overlaps_broad_observation"),
                "native_rgb_valid_ratio": row.get("native_rgb_valid_ratio"),
                "native_rgb_pair_cos_mean": row.get("native_rgb_pair_cos_mean"),
                "native_rgb_pair_cos_p10": row.get("native_rgb_pair_cos_p10"),
                "native_frame_mask_support_ratio": row.get("native_frame_mask_support_ratio"),
                "native_boundary_safe_ratio": row.get("native_boundary_safe_ratio"),
                "native_boundary_distance_p10": row.get("native_boundary_distance_p10"),
                "core_tube_ids": ";".join(str(tid) for tid in _core_ids(row)),
            }
        )
    _write_csv(seed_dir / "seed_slot_rows.csv", seed_export_rows)
    _write_csv(seed_dir / "seed_summary.csv", seed_summary_rows)
    _write_csv(seed_dir / "seed_source_summary.csv", seed_source_rows)
    seed_all = next(row for row in seed_summary_rows if row["scene"] == "ALL")
    seed_gate = {
        "seed_purity_mean_ge_0.90": bool(float(seed_all.get("seed_purity_mean") or 0.0) >= 0.90),
        "seed_purity_p10_ge_0.75": bool(float(seed_all.get("seed_purity_p10") or 0.0) >= 0.75),
        "GT_seed_IoU_010_ge_0.70": bool(float(seed_all.get("GT_with_seed_IoU_ge_0.10") or 0.0) >= 0.70),
        "GT_seed_IoU_025_ge_0.45": bool(float(seed_all.get("GT_with_seed_IoU_ge_0.25") or 0.0) >= 0.45),
        "scene0081_seed_IoU_010_ge_0.50": bool(float(seed_all.get("scene0081_GT_with_seed_IoU_ge_0.10") or 0.0) >= 0.50),
    }
    seed_gate["phaseC_seed_gate_pass"] = all(seed_gate.values())
    _write_json(
        seed_dir / "manifest.json",
        {
            **METHOD_MANIFEST_BASE,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "phase": "v30_phaseC_seed_slots",
            "seed_profile": str(args.seed_profile),
            "seed_selection_uses_gt": False,
            "uses_rgb_for_prediction": bool(str(args.seed_profile) == NATIVE_RGB_BOUNDARY_PROFILE),
            "uses_image_masks_for_prediction": bool(str(args.seed_profile) == NATIVE_RGB_BOUNDARY_PROFILE),
            "uses_rgbd_for_prediction": bool(str(args.seed_profile) == NATIVE_RGB_BOUNDARY_PROFILE),
            "diagnostic_gt_fields": [
                "proposal_purity_diagnostic",
                "proposal_best_IoU_diagnostic",
                "GT_with_seed_IoU_ge_0.10",
                "GT_with_seed_IoU_ge_0.25",
            ],
            "native_rgb_boundary_feature_rows": int(len(native_feature_rows)),
            "native_rgb_boundary_scene_rows": int(len(native_scene_rows)),
            "seed_gate": seed_gate,
            "artifact_files": ["seed_slot_rows.csv", "seed_summary.csv", "seed_source_summary.csv"],
        },
    )
    figures = seed_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if seed_source_rows:
        labels = [row["seed_source"].replace("_", "\n") for row in seed_source_rows]
        counts = [row["count"] for row in seed_source_rows]
        plt.figure(figsize=(10, 3.5))
        plt.bar(labels, counts, color="#59A14F")
        plt.ylabel("seed count")
        plt.tight_layout()
        plt.savefig(figures / "seed_source_counts.png", dpi=160)
        plt.close()
    _write_json(
        figures / "figure_manifest.json",
        {
            "figures": ["seed_source_counts.png"] if seed_source_rows else [],
            "uses_gt_for_visualization": True,
            "uses_gt_for_prediction": False,
        },
    )

    decomp_rows: list[dict[str, Any]] = []
    decomp_gate: dict[str, Any] = {"phaseD_gate_pass": False, "not_run_reason": None}
    if seed_gate["phaseC_seed_gate_pass"]:
        decomp_rows, decomp_gate = _decomposition_oracle(scenes, rows_by_scene, gt_by_scene, seed_rows, broad_by_scene, p11_ids)
    else:
        decomp_gate["not_run_reason"] = "Phase C seed gate failed; v30 plan forbids Phase D/E promotion before seed repair."
    decomp_dir = out_base / "v30_decomposition_oracle"
    if decomp_rows:
        _write_csv(decomp_dir / "decomposition_summary.csv", decomp_rows)
    else:
        _write_csv(decomp_dir / "decomposition_summary.csv", [])
    _write_json(
        decomp_dir / "manifest.json",
        {
            **METHOD_MANIFEST_BASE,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "phase": "v30_phaseD_decomposition_oracle",
            "seed_profile": str(args.seed_profile),
            "decomposition_uses_gt_for_assignment": True,
            "decomposition_oracle_is_method_result": False,
            "decomposition_gate": decomp_gate,
            "artifact_files": ["decomposition_summary.csv"],
        },
    )

    gate_summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed_profile": str(args.seed_profile),
        "phaseB_broad_rows": int(len(broad_rows)),
        "phaseC_seed_gate": seed_gate,
        "phaseD_decomposition_gate": decomp_gate,
        "native_rgb_boundary_feature_rows": int(len(native_feature_rows)),
        "native_rgb_boundary_scene_rows": int(len(native_scene_rows)),
        "can_run_phaseE_non_gt_solver": bool(seed_gate["phaseC_seed_gate_pass"] and decomp_gate.get("phaseD_gate_pass")),
        "stop_rule": None,
    }
    if not seed_gate["phaseC_seed_gate_pass"]:
        gate_summary["stop_rule"] = "Stop 1 seed slots insufficient"
    elif not decomp_gate.get("phaseD_gate_pass"):
        gate_summary["stop_rule"] = "Stop 2 decomposition oracle failed"
    _write_json(out_base / "v30_phaseBCD_gate_summary.json", gate_summary)
    return gate_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stream4D v30 object-slot ownership Phase B/C/D diagnostics.")
    parser.add_argument("--audit-root", default="outputs/audit")
    parser.add_argument("--out-base", default="outputs/audit")
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v26_occupancy_d5_warmstart64_probe5_topup20_patch2")
    parser.add_argument("--max-tubes-per-window", type=int, default=640)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument(
        "--seed-profile",
        choices=[
            "strict",
            "coverage_repair",
            "purity_repair",
            "consensus_repair",
            "support_expand_repair",
            "cooccurrence_repair",
            "cooccurrence_loose_repair",
            "overlap_support_repair",
            "dense_eroded_repair",
            NATIVE_RGB_BOUNDARY_PROFILE,
        ],
        default="strict",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(_json_safe(run(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
