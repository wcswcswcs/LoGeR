#!/usr/bin/env python3
"""Build the v67-S scale observability ledger for KITTI01.

This diagnostic intentionally separates image-level semantic observability from
point-level evidence. If per-chunk geometry is unavailable it still computes a
semantic/confidence ledger, but marks point-level fields as unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


GROUP_LABELS: Mapping[str, Sequence[str]] = {
    "dynamic": (
        "person",
        "car",
        "truck",
        "bus",
        "van",
        "rider",
        "cyclist",
        "bicycle",
        "motorcycle",
        "animal",
    ),
    "sky_context": ("sky", "cloud", "horizon"),
    "vegetation_farstuff": ("grass", "tree", "vegetation", "plant", "terrain", "mountain"),
    "vertical_static": (
        "building",
        "house",
        "wall",
        "handrail_or_fence",
        "fence",
        "pole",
        "traffic sign",
        "traffic light",
        "billboard_or_bulletin_board",
        "bridge",
    ),
    "vertical_static_weak": ("other_construction",),
    "ground_static": ("road", "ground", "sidewalk", "crosswalk", "floor"),
    "void_lowtrust": ("void", "unknown", "unlabeled"),
}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _normalise_label_names(label_names: Any) -> Dict[int, str]:
    if isinstance(label_names, Mapping):
        return {int(k): str(v) for k, v in label_names.items()}
    return {int(i): str(v) for i, v in enumerate(label_names)}


def _invert_label_names(label_names: Mapping[int, str]) -> Dict[str, int]:
    return {v: k for k, v in label_names.items()}


def _ids_for(names: Iterable[str], label_to_id: Mapping[str, int]) -> List[int]:
    return [int(label_to_id[n]) for n in names if n in label_to_id]


def _mask_for(labels: torch.Tensor, ids: Sequence[int]) -> torch.Tensor:
    out = torch.zeros_like(labels, dtype=torch.bool)
    for label_id in ids:
        out |= labels == int(label_id)
    return out


def _weighted_ratio(mask: torch.Tensor, weight: torch.Tensor, denom: float) -> float:
    if denom <= 0:
        return float("nan")
    return float(weight[mask].sum().item() / denom)


def _road_boundary(ground: torch.Tensor) -> torch.Tensor:
    edge = torch.zeros_like(ground, dtype=torch.bool)
    edge[:, 1:, :] |= ground[:, 1:, :] != ground[:, :-1, :]
    edge[:, :-1, :] |= ground[:, 1:, :] != ground[:, :-1, :]
    edge[:, :, 1:] |= ground[:, :, 1:] != ground[:, :, :-1]
    edge[:, :, :-1] |= ground[:, :, 1:] != ground[:, :, :-1]
    pooled = F.max_pool2d(edge[:, None].float(), kernel_size=5, stride=1, padding=2)
    return pooled[:, 0] > 0


def _spatial_score(anchor_weight: torch.Tensor, grid_h: int = 4, grid_w: int = 6) -> Tuple[float, float, float]:
    _, height, width = anchor_weight.shape
    masses: List[float] = []
    for gy in range(grid_h):
        y0 = int(round(gy * height / grid_h))
        y1 = int(round((gy + 1) * height / grid_h))
        for gx in range(grid_w):
            x0 = int(round(gx * width / grid_w))
            x1 = int(round((gx + 1) * width / grid_w))
            masses.append(float(anchor_weight[:, y0:y1, x0:x1].sum().item()))
    total = sum(masses)
    if total <= 0:
        return 0.0, 0.0, 0.0
    probs = [m / total for m in masses if m > 0]
    entropy = -sum(p * math.log(p) for p in probs) / math.log(len(masses))
    coverage = sum(1 for m in masses if m > 0) / len(masses)
    return float(entropy * coverage), float(entropy), float(coverage)


def _condition_score(points: torch.Tensor, weight: torch.Tensor, max_points: int = 50000) -> float:
    mask = torch.isfinite(points).all(dim=-1) & torch.isfinite(weight) & (weight > 0)
    count = int(mask.sum().item())
    if count < 16:
        return float("nan")
    pts = points[mask].float().reshape(-1, 3)
    w = weight[mask].float().reshape(-1)
    if pts.shape[0] > max_points:
        idx = torch.linspace(0, pts.shape[0] - 1, steps=max_points, device=pts.device).long()
        pts = pts[idx]
        w = w[idx]
    wsum = w.sum()
    if not torch.isfinite(wsum) or float(wsum.item()) <= 0.0:
        return float("nan")
    mean = (pts * w[:, None]).sum(dim=0) / wsum
    centered = pts - mean[None, :]
    cov = (centered * w[:, None]).T @ centered / wsum
    try:
        s = torch.linalg.svdvals(cov)
    except RuntimeError:
        return float("nan")
    if s.numel() < 3 or float(s[0].item()) <= 0.0:
        return float("nan")
    cond = torch.sqrt((s[1] / (s[0] + 1e-9)).clamp_min(0.0) * (s[2] / (s[0] + 1e-9)).clamp_min(0.0))
    return float((5.0 * cond).clamp(0.0, 1.0).item())


def _resize_labels(labels: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(labels[:, None].float(), size=size, mode="nearest").squeeze(1).long()


def _resize_conf(conf: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(conf[:, None].float(), size=size, mode="nearest").squeeze(1).float().clamp(0.0, 1.0)


def _entropy_from_probs(values: Sequence[float]) -> float:
    vals = [max(0.0, float(x)) for x in values]
    total = sum(vals)
    if total <= 0:
        return 0.0
    probs = [v / total for v in vals if v > 0]
    if len(probs) <= 1:
        return 0.0
    return float(-sum(p * math.log(p) for p in probs) / math.log(len(vals)))


def _csv_by_chunk(path: Path, method: Optional[str] = None) -> Dict[int, Dict[str, str]]:
    if not path.exists():
        return {}
    out: Dict[int, Dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if method is not None and row.get("method") != method:
                continue
            if row.get("chunk_id") in (None, ""):
                continue
            out[int(row["chunk_id"])] = row
    return out


def _dynamic_rows_by_chunk(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        return {}
    key_map = {
        "prior_dynamic_mass_D_gt_001": "explicit_dynamic_mass_D_gt_001",
        "prior_dynamic_mass_D_gt_050": "explicit_dynamic_mass_D_gt_050",
        "prior_dynamic_mass_D_gt_075": "explicit_dynamic_mass_D_gt_075",
        "prior_old_dyn_coverage": "explicit_dynamic_old_dyn_coverage",
        "prior_old_dyn_iou": "explicit_dynamic_old_dyn_iou",
        "prior_old_dyn_recall": "explicit_dynamic_old_dyn_recall",
        "prior_corr_D_old_dyn": "explicit_dynamic_corr_D_old_dyn",
        "prior_hmc_write_corr_score_dyn": "explicit_dynamic_write_corr_score_dyn",
        "prior_hmc_write_corr_score_exp_dyn": "explicit_dynamic_write_corr_score_exp_dyn",
    }
    out: Dict[int, Dict[str, Any]] = {}
    for obj in _read_jsonl(path):
        if "chunk_idx" not in obj:
            continue
        row = {}
        for source, dest in key_map.items():
            row[dest] = obj.get(source)
        out[int(obj["chunk_idx"])] = row
    return out


def _float(row: Mapping[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in ("", "nan", "None", None):
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(values, dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def _spearman(rows: Sequence[Mapping[str, Any]], x_key: str, y_key: str) -> Optional[float]:
    pairs = []
    for row in rows:
        x = row.get(x_key)
        y = row.get(y_key)
        if x is None or y is None:
            continue
        try:
            xf = float(x)
            yf = float(y)
        except (TypeError, ValueError):
            continue
        if math.isfinite(xf) and math.isfinite(yf):
            pairs.append((xf, yf))
    if len(pairs) < 3:
        return None
    x = _rankdata(np.array([p[0] for p in pairs], dtype=float))
    y = _rankdata(np.array([p[1] for p in pairs], dtype=float))
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _auc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    score_arr = np.array(scores, dtype=float)
    label_arr = np.array(labels, dtype=int)
    keep = np.isfinite(score_arr)
    score_arr = score_arr[keep]
    label_arr = label_arr[keep]
    pos = label_arr == 1
    neg = label_arr == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _rankdata(score_arr)
    rank_sum_pos = float(ranks[pos].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _top_mask(values: Sequence[float], top_fraction: float, high_is_bad: bool = True) -> List[bool]:
    arr = np.array(values, dtype=float)
    finite = np.isfinite(arr)
    out = [False] * len(arr)
    if finite.sum() == 0:
        return out
    vals = arr[finite]
    q = 1.0 - top_fraction if high_is_bad else top_fraction
    thresh = float(np.quantile(vals, q))
    for idx, value in enumerate(arr):
        if not math.isfinite(float(value)):
            continue
        out[idx] = value >= thresh if high_is_bad else value <= thresh
    return out


def _precision_recall(pred: Sequence[bool], target: Sequence[bool]) -> Dict[str, float]:
    tp = sum(1 for p, t in zip(pred, target) if p and t)
    fp = sum(1 for p, t in zip(pred, target) if p and not t)
    fn = sum(1 for p, t in zip(pred, target) if not p and t)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return {"precision": float(precision), "recall": float(recall), "tp": tp, "fp": fp, "fn": fn}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_ledger(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.full_semantic_pt, map_location="cpu", weights_only=False)
    sem = payload["semantic_segmentation"]
    label_names = _normalise_label_names(sem["label_names"])
    label_to_id = _invert_label_names(label_names)
    group_ids = {group: _ids_for(names, label_to_id) for group, names in GROUP_LABELS.items()}
    mapping = {
        "groups": {group: {"labels": list(GROUP_LABELS[group]), "ids_present": ids} for group, ids in group_ids.items()},
        "label_names": label_names,
        "source": str(args.full_semantic_pt),
    }
    (out_dir / "semantic_group_mapping_v67s.json").write_text(
        json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rows_index = _read_jsonl(args.stage_c_cache / "cache_index.jsonl")
    taxonomy = _csv_by_chunk(args.v62_report / "phase7_taxonomy" / "chunk_error_taxonomy.csv", method="h35")
    scale = _csv_by_chunk(args.v62_report / "phase5_intrachunk_scale" / "h35_intrachunk_scale_metrics.csv", method="h35")
    inter = _csv_by_chunk(args.v62_report / "phase4_interchunk" / "h35_interchunk_metrics.csv", method="h35")
    gap = _csv_by_chunk(args.v62_report / "phase7_taxonomy" / "h35_c9_gap_taxonomy.csv")
    dynamic_rows = _dynamic_rows_by_chunk(args.h35_run / "hmc_state_hash.jsonl")

    rows: List[Dict[str, Any]] = []
    prev_q: Optional[float] = None
    for index_row in rows_index:
        chunk_id = int(index_row["chunk_idx"])
        chunk = torch.load(args.stage_c_cache / index_row["chunk"] / "masklet.pt", map_location="cpu", weights_only=False)
        csem = chunk["semantic_segmentation"]
        labels = csem["label_maps"].long()
        conf = csem.get("confidence_maps")
        if conf is None:
            conf = torch.ones_like(labels, dtype=torch.float32)
            confidence_source = "missing_fallback_ones"
        else:
            conf = conf.float()
            confidence_source = "semantic_segmentation.confidence_maps"
            if conf.dtype == torch.uint8:
                conf = conf / 255.0
            conf = conf.clamp(0.0, 1.0)

        point_level_available = False
        local_points = None
        geom_conf = None
        labels_eval = labels
        sem_conf_eval = conf
        geo_path = args.per_chunk_geometry_dir / f"chunk_{chunk_id:03d}.pt" if args.per_chunk_geometry_dir else None
        if geo_path is not None and geo_path.exists():
            geo = torch.load(geo_path, map_location="cpu", weights_only=False)
            local_points = geo.get("local_points")
            geom_conf = geo.get("conf")
            if local_points is not None and geom_conf is not None:
                point_level_available = True
                geom_conf = geom_conf.float().clamp(0.0, 1.0)
                labels_eval = _resize_labels(labels, tuple(geom_conf.shape[-2:]))
                sem_conf_eval = _resize_conf(conf, tuple(geom_conf.shape[-2:]))
        if point_level_available and local_points is not None and geom_conf is not None:
            valid = torch.isfinite(local_points).all(dim=-1) & torch.isfinite(geom_conf) & (geom_conf > 0.05)
            weight = (geom_conf * sem_conf_eval * valid.float()).float()
            d_geo_tensor = (1.0 - geom_conf).clamp(0.0, 1.0)
            d_geo_source = "1-geom_conf_from_per_chunk_geometry"
        else:
            valid = torch.ones_like(labels_eval, dtype=torch.bool)
            weight = sem_conf_eval
            d_geo_tensor = (1.0 - sem_conf_eval).clamp(0.0, 1.0)
            d_geo_source = "1-confidence_maps_fallback_no_Dg"
        denom = float(weight.sum().item())

        masks = {group: _mask_for(labels_eval, ids) for group, ids in group_ids.items()}
        vertical_mask = masks["vertical_static"] | masks["vertical_static_weak"]
        ground_mask = masks["ground_static"]
        road_boundary = _road_boundary(ground_mask)
        height = labels_eval.shape[1]
        lower_half = torch.zeros_like(labels_eval, dtype=torch.bool)
        lower_half[:, height // 2 :, :] = True
        near_vertical_mask = vertical_mask & lower_half
        anchor_mask = (vertical_mask | road_boundary) & valid
        anchor_weight = weight * anchor_mask.float()

        vertical_ratio = _weighted_ratio(masks["vertical_static"], weight, denom)
        vertical_weak_ratio = _weighted_ratio(masks["vertical_static_weak"], weight, denom)
        vertical_total_ratio = _weighted_ratio(vertical_mask, weight, denom)
        vertical_lower_half_ratio = _weighted_ratio(near_vertical_mask, weight, denom)
        road_boundary_ratio = _weighted_ratio(road_boundary, weight, denom)
        ground_ratio = _weighted_ratio(ground_mask, weight, denom)
        sky_ratio = _weighted_ratio(masks["sky_context"], weight, denom)
        dynamic_ratio = _weighted_ratio(masks["dynamic"], weight, denom)
        vegetation_ratio = _weighted_ratio(masks["vegetation_farstuff"], weight, denom)
        void_ratio = _weighted_ratio(masks["void_lowtrust"], weight, denom)
        if point_level_available and local_points is not None:
            dist = torch.linalg.norm(local_points.float(), dim=-1)
            dist_valid = dist[valid & torch.isfinite(dist)]
            if dist_valid.numel() > 0:
                near_thresh = torch.quantile(dist_valid, 0.50)
                near_mask = dist <= near_thresh
                near_static_proxy = _weighted_ratio(near_mask & anchor_mask, weight, denom)
                near_static_source = "point_level_distance_median_static_anchor"
            else:
                near_static_proxy = float("nan")
                near_static_source = "point_level_distance_unavailable"
        elif args.near_static_mode == "lower_half_proxy":
            near_static_proxy = min(1.0, vertical_lower_half_ratio + road_boundary_ratio)
            near_static_source = "image_level_lower_half_vertical_or_road_boundary_proxy_no_local_points"
        else:
            near_static_proxy = min(1.0, vertical_total_ratio + road_boundary_ratio)
            near_static_source = "image_level_vertical_or_road_boundary_proxy_no_local_points"
        spatial_score, spatial_entropy, grid_coverage = _spatial_score(anchor_weight)
        condition_score = (
            _condition_score(local_points.float(), anchor_weight)
            if point_level_available and local_points is not None
            else float("nan")
        )
        fallback_condition_score = spatial_score * vertical_total_ratio
        road_plane_dominance = ground_ratio * (1.0 - vertical_total_ratio) * (1.0 - road_boundary_ratio)
        semantic_entropy = _entropy_from_probs(
            [dynamic_ratio, sky_ratio, vegetation_ratio, vertical_total_ratio, ground_ratio, void_ratio]
        )
        geometry_conf = float(conf.mean().item())
        if point_level_available and geom_conf is not None:
            geometry_conf = float(geom_conf[valid].mean().item()) if int(valid.sum().item()) else float("nan")
            d_geo = float(d_geo_tensor[valid].mean().item()) if int(valid.sum().item()) else float("nan")
        else:
            geometry_conf = float(conf.mean().item())
            d_geo = float(d_geo_tensor.mean().item())

        if math.isfinite(condition_score):
            anchor_quality = (
                0.30 * vertical_total_ratio
                + 0.20 * road_boundary_ratio
                + 0.15 * near_static_proxy
                + 0.20 * spatial_score
                + 0.15 * condition_score
            )
            condition_source = "point_level_weighted_covariance"
        else:
            anchor_quality = (
                0.35 * vertical_total_ratio
                + 0.25 * road_boundary_ratio
                + 0.15 * near_static_proxy
                + 0.25 * spatial_score
            )
            condition_source = "unavailable_no_per_chunk_geometry" if not point_level_available else "unavailable_low_anchor_support"
        weak_evidence = (
            0.20 * sky_ratio
            + 0.15 * dynamic_ratio
            + 0.20 * vegetation_ratio
            + 0.25 * road_plane_dominance
            + 0.20 * void_ratio
        )
        observability_preset = str(getattr(args, "observability_preset", "v67s_default"))
        if observability_preset == "risk_calibrated_pointgeom":
            # O1 repair diagnostic after the default Q failed: vertical/static mass
            # alone correlated with worse future error, so this preset emphasizes
            # point-level confidence and scene-diversity cues instead of treating
            # every vertical/static pixel as an anchor.
            condition_term = condition_score if math.isfinite(condition_score) else fallback_condition_score
            anchor_quality = (
                0.35 * geometry_conf
                + 0.25 * dynamic_ratio
                + 0.20 * vegetation_ratio
                + 0.20 * (1.0 - road_plane_dominance)
                + 0.10 * condition_term
            )
            weak_evidence = (
                0.35 * vertical_total_ratio
                + 0.30 * sky_ratio
                + 0.20 * semantic_entropy
                + 0.15 * d_geo
            )
            condition_source = f"{condition_source}+risk_calibrated_pointgeom"
        q_scale = max(0.0, min(1.0, anchor_quality / (anchor_quality + weak_evidence + 1e-9)))
        q_smooth = q_scale if prev_q is None else 0.5 * prev_q + 0.5 * q_scale
        prev_q = q_smooth
        anchor_type = "anchor_rich" if q_scale >= 0.60 else "anchor_sparse" if q_scale <= 0.35 else "ambiguous"
        straight_road = (
            ground_ratio >= 0.45
            and sky_ratio >= 0.20
            and vertical_total_ratio <= 0.08
            and road_boundary_ratio <= 0.05
            and fallback_condition_score <= 0.20
        )

        tax_row = taxonomy.get(chunk_id, {})
        scale_row = scale.get(chunk_id, {})
        inter_row = inter.get(chunk_id, {})
        gap_row = gap.get(chunk_id, {})
        flags = tax_row.get("error_type_flags", "")
        primary = tax_row.get("primary_error_type", "")
        type_ed = ("TYPE_E" in primary) or ("TYPE_D" in primary) or ("TYPE_E" in flags) or ("TYPE_D" in flags)
        row = {
            "chunk_id": chunk_id,
            "frame_start": int(index_row["start_frame"]),
            "frame_end": int(index_row["end_frame"]),
            "Q_scale": q_scale,
            "Q_scale_smoothed": q_smooth,
            "observability_preset": observability_preset,
            "anchor_type": anchor_type,
            "straight_road_anchor_sparse": bool(straight_road),
            "vertical_static_ratio": vertical_ratio,
            "vertical_static_weak_ratio": vertical_weak_ratio,
            "vertical_static_total_ratio": vertical_total_ratio,
            "vertical_static_lower_half_ratio": vertical_lower_half_ratio,
            "road_boundary_ratio": road_boundary_ratio,
            "ground_static_ratio": ground_ratio,
            "road_plane_dominance": road_plane_dominance,
            "near_static_ratio": near_static_proxy,
            "near_static_ratio_source": near_static_source,
            "anchor_spatial_score": spatial_score,
            "anchor_spatial_entropy": spatial_entropy,
            "anchor_grid_coverage": grid_coverage,
            "condition_score": condition_score,
            "condition_score_source": condition_source,
            "fallback_condition_score": fallback_condition_score,
            "sky_ratio": sky_ratio,
            "dynamic_ratio": dynamic_ratio,
            "vegetation_ratio": vegetation_ratio,
            "void_lowtrust_ratio": void_ratio,
            "semantic_entropy": semantic_entropy,
            "geometry_confidence_mean": geometry_conf,
            "D_geo_mean": d_geo,
            "D_geo_source": d_geo_source,
            "semantic_confidence_source": confidence_source,
            "point_level_observability_available": point_level_available,
            "per_chunk_geometry_path": str(geo_path) if geo_path is not None and geo_path.exists() else "",
            "v62_taxonomy_type": primary,
            "v62_type_e_or_d_high_risk": bool(type_ed),
            "global_chunk_ate": _float(tax_row, "global_chunk_ate"),
            "local_sim3_chunk_ate": _float(tax_row, "local_sim3_chunk_ate"),
            "head_to_tail_transfer_ratio": _float(scale_row, "head_to_tail_transfer_ratio"),
            "intra_scale_variance": _float(scale_row, "intra_scale_variance"),
            "future_after_overlap_error": _float(inter_row, "nonoverlap_future_error_after_overlap_sim3"),
            "scale_jump_vs_prev": _float(inter_row, "abs_scale_jump_gtlocal"),
            "rolling100_error": _float(tax_row, "rolling100_error"),
            "H35_minus_C9_gap": _float(gap_row, "h35_minus_c9_global_chunk_ate"),
        }
        row.update(dynamic_rows.get(chunk_id, {}))
        rows.append(row)

    fields = [
        "chunk_id",
        "frame_start",
        "frame_end",
        "Q_scale",
        "Q_scale_smoothed",
        "observability_preset",
        "anchor_type",
        "straight_road_anchor_sparse",
        "vertical_static_ratio",
        "vertical_static_weak_ratio",
        "vertical_static_total_ratio",
        "vertical_static_lower_half_ratio",
        "road_boundary_ratio",
        "ground_static_ratio",
        "road_plane_dominance",
        "near_static_ratio",
        "near_static_ratio_source",
        "anchor_spatial_score",
        "anchor_spatial_entropy",
        "anchor_grid_coverage",
        "condition_score",
        "condition_score_source",
        "fallback_condition_score",
        "sky_ratio",
        "dynamic_ratio",
        "vegetation_ratio",
        "void_lowtrust_ratio",
        "semantic_entropy",
        "geometry_confidence_mean",
        "D_geo_mean",
        "D_geo_source",
        "semantic_confidence_source",
        "point_level_observability_available",
        "per_chunk_geometry_path",
        "explicit_dynamic_mass_D_gt_001",
        "explicit_dynamic_mass_D_gt_050",
        "explicit_dynamic_mass_D_gt_075",
        "explicit_dynamic_old_dyn_coverage",
        "explicit_dynamic_old_dyn_iou",
        "explicit_dynamic_old_dyn_recall",
        "explicit_dynamic_corr_D_old_dyn",
        "explicit_dynamic_write_corr_score_dyn",
        "explicit_dynamic_write_corr_score_exp_dyn",
        "v62_taxonomy_type",
        "v62_type_e_or_d_high_risk",
        "global_chunk_ate",
        "local_sim3_chunk_ate",
        "head_to_tail_transfer_ratio",
        "intra_scale_variance",
        "future_after_overlap_error",
        "scale_jump_vs_prev",
        "rolling100_error",
        "H35_minus_C9_gap",
    ]
    _write_csv(out_dir / "chunk_scale_observability.csv", rows, fields)

    analysis = _analyse(rows)
    (out_dir / "observability_analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(out_dir / "feature_importance.csv", analysis["feature_importance"], analysis["feature_importance_fields"])
    (out_dir / "observability_report.md").write_text(_report_text(args, analysis), encoding="utf-8")
    print(json.dumps(analysis["gate"], indent=2, sort_keys=True))


def _analyse(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    corr_head = _spearman(rows, "Q_scale", "head_to_tail_transfer_ratio")
    corr_future = _spearman(rows, "Q_scale", "future_after_overlap_error")
    corr_intra = _spearman(rows, "Q_scale", "intra_scale_variance")
    corr_gap = _spearman(rows, "Q_scale", "H35_minus_C9_gap")
    risk_scores = [1.0 - float(row["Q_scale"]) for row in rows]
    risk_labels = [1 if row["v62_type_e_or_d_high_risk"] else 0 for row in rows]
    auc_type_ed = _auc(risk_scores, risk_labels)
    head_values = [float(row["head_to_tail_transfer_ratio"]) for row in rows]
    top30_head = _top_mask(head_values, 0.30, high_is_bad=True)
    anchor_sparse = [row["anchor_type"] == "anchor_sparse" for row in rows]
    anchor_rich = [row["anchor_type"] == "anchor_rich" for row in rows]
    top30_anchor_sparse_frac = (
        sum(1 for a, t in zip(anchor_sparse, top30_head) if a and t) / max(1, sum(1 for t in top30_head if t))
    )
    bottom50_error = _top_mask([float(row["global_chunk_ate"]) for row in rows], 0.50, high_is_bad=False)
    pr_sparse_head = _precision_recall(anchor_sparse, top30_head)
    pr_rich_lowerr = _precision_recall(anchor_rich, bottom50_error)
    predictive_pass = (
        (corr_head is not None and corr_head <= -0.30)
        or (corr_future is not None and corr_future <= -0.30)
        or (auc_type_ed is not None and auc_type_ed >= 0.65)
        or top30_anchor_sparse_frac >= 0.60
    )
    features = [
        "vertical_static_total_ratio",
        "vertical_static_lower_half_ratio",
        "road_boundary_ratio",
        "ground_static_ratio",
        "road_plane_dominance",
        "near_static_ratio",
        "anchor_spatial_score",
        "fallback_condition_score",
        "sky_ratio",
        "dynamic_ratio",
        "vegetation_ratio",
        "void_lowtrust_ratio",
        "semantic_entropy",
        "geometry_confidence_mean",
        "D_geo_mean",
        "explicit_dynamic_mass_D_gt_050",
        "explicit_dynamic_mass_D_gt_075",
        "explicit_dynamic_old_dyn_coverage",
        "explicit_dynamic_old_dyn_iou",
        "explicit_dynamic_corr_D_old_dyn",
        "explicit_dynamic_write_corr_score_dyn",
        "explicit_dynamic_write_corr_score_exp_dyn",
    ]
    targets = ["head_to_tail_transfer_ratio", "future_after_overlap_error", "intra_scale_variance", "H35_minus_C9_gap"]
    importance = []
    for feature in features:
        item: Dict[str, Any] = {"feature": feature}
        abs_vals = []
        for target in targets:
            corr = _spearman(rows, feature, target)
            item[f"spearman_{target}"] = corr
            if corr is not None:
                abs_vals.append(abs(corr))
        item["max_abs_spearman"] = max(abs_vals) if abs_vals else None
        importance.append(item)
    importance.sort(key=lambda x: -1.0 if x["max_abs_spearman"] is None else -float(x["max_abs_spearman"]))
    return {
        "num_chunks": len(rows),
        "counts": {
            "anchor_sparse": sum(anchor_sparse),
            "ambiguous": sum(1 for row in rows if row["anchor_type"] == "ambiguous"),
            "anchor_rich": sum(anchor_rich),
            "straight_road_anchor_sparse": sum(1 for row in rows if row["straight_road_anchor_sparse"]),
            "type_e_or_d_high_risk": sum(risk_labels),
        },
        "correlations": {
            "spearman_Q_scale_head_to_tail_transfer_ratio": corr_head,
            "spearman_Q_scale_future_after_overlap_error": corr_future,
            "spearman_Q_scale_intra_scale_variance": corr_intra,
            "spearman_Q_scale_H35_minus_C9_gap": corr_gap,
        },
        "auc": {
            "anchor_sparse_score_1_minus_Q_for_TYPE_E_or_D": auc_type_ed,
        },
        "precision_recall": {
            "anchor_sparse_for_top30_head_to_tail": pr_sparse_head,
            "anchor_rich_for_bottom50_global_chunk_ate": pr_rich_lowerr,
        },
        "top30_scale_risk_chunks_anchor_sparse_fraction": float(top30_anchor_sparse_frac),
        "gate": {
            "scale_observability_predictive_pass": bool(predictive_pass),
            "criteria": {
                "corr_Q_head_to_tail_le_neg0p30": corr_head is not None and corr_head <= -0.30,
                "corr_Q_future_after_overlap_le_neg0p30": corr_future is not None and corr_future <= -0.30,
                "auc_anchor_sparse_TYPE_E_D_ge_0p65": auc_type_ed is not None and auc_type_ed >= 0.65,
                "top30_scale_risk_anchor_sparse_frac_ge_0p60": top30_anchor_sparse_frac >= 0.60,
            },
        },
        "feature_importance": importance,
        "feature_importance_fields": [
            "feature",
            "spearman_head_to_tail_transfer_ratio",
            "spearman_future_after_overlap_error",
            "spearman_intra_scale_variance",
            "spearman_H35_minus_C9_gap",
            "max_abs_spearman",
        ],
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        return f"{value:.6f}"
    return str(value)


def _report_text(args: argparse.Namespace, analysis: Mapping[str, Any]) -> str:
    corr = analysis["correlations"]
    gate = analysis["gate"]
    counts = analysis["counts"]
    if args.per_chunk_geometry_dir is None:
        scope_caveat = [
            "- Existing H35 artifacts did not expose per-chunk point geometry for this run, so this ledger uses image-level dense semantic labels and confidence maps plus v62 pose/scale diagnostics.",
            "- `condition_score` is unavailable and `near_static_ratio` is an image-level vertical/road-boundary proxy.",
            "- This report cannot claim point-level overlap semantic success.",
        ]
    else:
        scope_caveat = [
            "- Per-chunk geometry was provided; this ledger uses resized dense semantic labels/confidence maps plus point-level geometry confidence and weighted covariance condition score.",
            "- This report is still an observability diagnostic, not an online controller result.",
        ]
    if getattr(args, "observability_preset", "v67s_default") != "v67s_default":
        scope_caveat.append(
            f"- observability_preset={args.observability_preset}; this is a post-default repair diagnostic and must not be treated as a pre-registered method success by itself."
        )
    return "\n".join(
        [
            "# v67-S Phase O1 Scale Observability Report",
            "",
            "Inputs:",
            "",
            f"- full_semantic_pt: `{args.full_semantic_pt}`",
            f"- stage_c_cache: `{args.stage_c_cache}`",
            f"- h35_run: `{args.h35_run}`",
            f"- v62_report: `{args.v62_report}`",
            f"- per_chunk_geometry_dir: `{args.per_chunk_geometry_dir}`",
            f"- observability_preset: `{args.observability_preset}`",
            "",
            "Scope caveat:",
            "",
            *scope_caveat,
            "",
            "Counts:",
            "",
            f"- chunks: {analysis['num_chunks']}",
            f"- anchor_sparse: {counts['anchor_sparse']}",
            f"- ambiguous: {counts['ambiguous']}",
            f"- anchor_rich: {counts['anchor_rich']}",
            f"- straight_road_anchor_sparse: {counts['straight_road_anchor_sparse']}",
            f"- TYPE_E_or_D high-risk chunks: {counts['type_e_or_d_high_risk']}",
            "",
            "Gate metrics:",
            "",
            f"- Spearman(Q_scale, head_to_tail_transfer_ratio): {_fmt(corr['spearman_Q_scale_head_to_tail_transfer_ratio'])}",
            f"- Spearman(Q_scale, future_after_overlap_error): {_fmt(corr['spearman_Q_scale_future_after_overlap_error'])}",
            f"- Spearman(Q_scale, intra_scale_variance): {_fmt(corr['spearman_Q_scale_intra_scale_variance'])}",
            f"- Spearman(Q_scale, H35_minus_C9_gap): {_fmt(corr['spearman_Q_scale_H35_minus_C9_gap'])}",
            f"- AUC(1-Q_scale -> TYPE_E/TYPE_D): {_fmt(analysis['auc']['anchor_sparse_score_1_minus_Q_for_TYPE_E_or_D'])}",
            f"- top30 head-to-tail chunks anchor_sparse fraction: {_fmt(analysis['top30_scale_risk_chunks_anchor_sparse_fraction'])}",
            "",
            f"Predictive pass: `{gate['scale_observability_predictive_pass']}`",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", default="01")
    parser.add_argument("--full_semantic_pt", required=True, type=Path)
    parser.add_argument("--stage_c_cache", required=True, type=Path)
    parser.add_argument("--h35_run", required=True, type=Path)
    parser.add_argument("--v62_report", required=True, type=Path)
    parser.add_argument("--per_chunk_geometry_dir", type=Path)
    parser.add_argument("--merge_trace_dir", type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument(
        "--near_static_mode",
        choices=("all_vertical_proxy", "lower_half_proxy"),
        default="all_vertical_proxy",
        help="Image-level fallback used when point-level near/static geometry is unavailable.",
    )
    parser.add_argument(
        "--observability_preset",
        choices=("v67s_default", "risk_calibrated_pointgeom"),
        default="v67s_default",
        help="Q_scale formula preset. Non-default presets are repair diagnostics after the default O1 gate fails.",
    )
    args = parser.parse_args()
    build_ledger(args)


if __name__ == "__main__":
    main()
