#!/usr/bin/env python3
"""ACL2 v70 RADIO sidecar MERGE oracle over materialized raw overlap pairs.

This is an offline R5 mechanism diagnostic, not an online HMC result. It uses
the same H35 parity raw overlap pairs and centered pose-metric gate as the v69
overlap-pair oracle, but replaces the fit-point selector with RADIO sidecar
masks plus the required label/shuffle/random controls.
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

try:
    from diagnose_v67_boundary_sim3_action_oracle import (
        _best_mechanism_improvement,
        _rmse_dist,
        _rotation_delta_deg,
        _rotation_power,
        _safe_improvement_ratio,
        _target_chunks,
    )
    from diagnose_v67_offline_scale_controller import (
        DEFAULT_GT,
        _load_kitti_gt,
        _load_postmerge_trajectory,
        _load_trace,
        _metric_result_row,
    )
    from diagnose_v67_overlap_pair_action_oracle import _fit_pair_correction, _load_pair, _parse_source
    from diagnose_v69_centered_overlap_pair_action_oracle import (
        CENTER_MODES,
        _apply_centered_correction,
        _boundary_jump,
        _center_for_mode,
        _make_baseline_row,
        _tail100_delta,
        _transform_points,
    )
    from v70_radio_sidecar_common import parse_chunks, utc_now
except ImportError:  # pragma: no cover
    from tools.diagnose_v67_boundary_sim3_action_oracle import (
        _best_mechanism_improvement,
        _rmse_dist,
        _rotation_delta_deg,
        _rotation_power,
        _safe_improvement_ratio,
        _target_chunks,
    )
    from tools.diagnose_v67_offline_scale_controller import (
        DEFAULT_GT,
        _load_kitti_gt,
        _load_postmerge_trajectory,
        _load_trace,
        _metric_result_row,
    )
    from tools.diagnose_v67_overlap_pair_action_oracle import _fit_pair_correction, _load_pair, _parse_source
    from tools.diagnose_v69_centered_overlap_pair_action_oracle import (
        CENTER_MODES,
        _apply_centered_correction,
        _boundary_jump,
        _center_for_mode,
        _make_baseline_row,
        _tail100_delta,
        _transform_points,
    )
    from tools.v70_radio_sidecar_common import parse_chunks, utc_now


RADIO_CANDIDATES = {
    "radio_only",
    "radio_semantic_correspondence",
    "radio_robust_kernel",
    "radio_object_internal_validation",
    "radio_weighted_geometry",
    "radio_internal_ranked_geometry",
    "radio_plan_weighted_kernel_ranked",
    "radio_plan_activity_kernel_ranked",
    "radio_plan_weighted_kernel_weighted_fit",
    "radio_plan_activity_kernel_weighted_fit",
    "radio_semantic_residual_kernel_ranked",
    "radio_semantic_residual_kernel_weighted_fit",
}
CONTROL_CANDIDATES = {
    "current_label_confidence_only",
    "current_label_shuffle",
    "current_confidence_shuffle",
    "radio_feature_spatial_shuffle",
    "radio_component_id_shuffle",
    "radio_confidence_temporal_shuffle",
    "radio_weighted_geometry_component_shuffle",
    "radio_weighted_geometry_feature_shuffle",
    "radio_weighted_geometry_confidence_temporal_shuffle",
    "radio_plan_weighted_kernel_component_shuffle",
    "radio_plan_weighted_kernel_feature_shuffle",
    "radio_plan_weighted_kernel_robust_shuffle",
    "radio_plan_activity_kernel_component_shuffle",
    "radio_plan_activity_kernel_feature_shuffle",
    "radio_plan_activity_kernel_activity_shuffle",
    "radio_plan_weighted_kernel_feature_shuffle_weighted_fit",
    "radio_plan_weighted_kernel_robust_shuffle_weighted_fit",
    "radio_plan_activity_kernel_component_shuffle_weighted_fit",
    "radio_plan_activity_kernel_feature_shuffle_weighted_fit",
    "radio_plan_activity_kernel_activity_shuffle_weighted_fit",
    "radio_semantic_residual_kernel_residual_shuffle",
    "radio_semantic_residual_kernel_semantic_shuffle",
    "radio_semantic_residual_kernel_residual_shuffle_weighted_fit",
    "radio_semantic_residual_kernel_semantic_shuffle_weighted_fit",
    "same_entropy_random_proxy_attention",
    "same_degree_random_affinity_graph",
}
BASELINE_CANDIDATES = {"geometry_only"}


def _float(value: Any, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def _finite_mean(values: Iterable[Any]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(xs)) if xs else None


def _finite_median(values: Iterable[Any]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.median(xs)) if xs else None


def _safe_tag(value: str) -> str:
    out = []
    for ch in str(value):
        if ch.isalnum():
            out.append(ch)
        elif ch in {"_", "-"}:
            out.append(ch)
        elif ch == ".":
            out.append("p")
        else:
            out.append("_")
    return "".join(out).strip("_") or "x"


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _np(value: Any, *, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _index_sidecars(sidecar_dirs: Sequence[Path]) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for root in sidecar_dirs:
        for path in sorted(root.glob("chunk_*/radio_sidecar.pt")):
            name = path.parent.name
            try:
                chunk_id = int(name.split("_")[1])
            except (IndexError, ValueError):
                payload = torch.load(path, map_location="cpu", weights_only=False)
                chunk_id = int(payload["chunk_id"])
            out[chunk_id] = path
    return out


def _load_sidecar(sidecar_index: Mapping[int, Path], chunk_id: int, cache: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    if chunk_id not in cache:
        path = sidecar_index.get(int(chunk_id))
        if path is None:
            raise FileNotFoundError(f"missing sidecar for chunk {chunk_id}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: expected dict payload")
        cache[int(chunk_id)] = payload
    return cache[int(chunk_id)]


def _detect_coord_order(coords: np.ndarray, height: int, width: int) -> str:
    if coords.size == 0 or coords.ndim != 2 or coords.shape[1] < 2:
        return "yx"
    max0 = float(np.nanmax(coords[:, 0]))
    max1 = float(np.nanmax(coords[:, 1]))
    yx_ok = max0 <= height + 2 and max1 <= width + 2
    xy_ok = max1 <= height + 2 and max0 <= width + 2
    if yx_ok and not xy_ok:
        return "yx"
    if xy_ok and not yx_ok:
        return "xy"
    return "yx"


def _field(payload: Mapping[str, Any], name: str) -> np.ndarray:
    if name not in payload:
        raise KeyError(f"sidecar missing field {name}")
    return _np(payload[name])


def _sample_sidecar(pair: Mapping[str, Any], payload: Mapping[str, Any], prefix: str) -> Tuple[Dict[str, np.ndarray], str]:
    points = _np(pair[f"{prefix}_overlap_points"], dtype=np.float64)
    n = int(points.shape[0])
    frame_ids = _np(pair.get(f"{prefix}_frame_ids", np.full(n, -1)), dtype=np.int64).reshape(-1)
    coords = _np(pair.get(f"{prefix}_pixel_coords", np.zeros((n, 2))), dtype=np.float64)
    if frame_ids.shape[0] != n or coords.shape[0] != n:
        raise ValueError(f"{prefix}: frame_ids/pixel_coords length mismatch with overlap points")

    start = int(payload["global_start_frame"])
    height = int(payload["frame_height"])
    width = int(payload["frame_width"])
    grid_h, grid_w = [int(x) for x in payload["patch_grid"]]
    order = _detect_coord_order(coords, height, width)
    if order == "yx":
        yy, xx = coords[:, 0], coords[:, 1]
    else:
        xx, yy = coords[:, 0], coords[:, 1]
    local = frame_ids - start
    rr = np.floor(np.clip(yy, 0, max(height - 1, 0)) / max(float(height), 1.0) * grid_h).astype(np.int64)
    cc = np.floor(np.clip(xx, 0, max(width - 1, 0)) / max(float(width), 1.0) * grid_w).astype(np.int64)
    rr = np.clip(rr, 0, grid_h - 1)
    cc = np.clip(cc, 0, grid_w - 1)
    valid = (
        np.isfinite(points).all(axis=1)
        & (local >= 0)
        & (local < int(_field(payload, "object_component_id").shape[0]))
        & np.isfinite(xx)
        & np.isfinite(yy)
    )

    def sample_scalar(name: str, fill: float = np.nan) -> np.ndarray:
        arr = _field(payload, name)
        out = np.full((n,), fill, dtype=np.float64)
        idx = np.where(valid)[0]
        if idx.size:
            out[idx] = arr[local[idx], rr[idx], cc[idx]].astype(np.float64)
        return out

    def sample_feat(name: str) -> np.ndarray:
        arr = _field(payload, name).astype(np.float64, copy=False)
        dim = int(arr.shape[-1])
        out = np.full((n, dim), np.nan, dtype=np.float64)
        idx = np.where(valid)[0]
        if idx.size:
            out[idx] = arr[local[idx], rr[idx], cc[idx], :]
        return out

    component = sample_scalar("object_component_id", fill=-1).astype(np.int64)
    boundary = sample_scalar("object_boundary_score", fill=np.nan)
    dynamic = sample_scalar("radio_dynamic_score", fill=np.nan)
    sky = sample_scalar("radio_sky_context_score", fill=np.nan)
    lowtrust = sample_scalar("radio_lowtrust_score", fill=np.nan)
    risk = np.nanmax(np.stack([boundary, dynamic, sky, lowtrust], axis=0), axis=0)
    return {
        "valid": valid,
        "component": component,
        "confidence": sample_scalar("radio_confidence", fill=np.nan),
        "stability": sample_scalar("temporal_stability", fill=np.nan),
        "boundary": boundary,
        "interior": sample_scalar("object_interior_score", fill=np.nan),
        "dynamic": dynamic,
        "sky": sky,
        "lowtrust": lowtrust,
        "risk": risk,
        "feat": sample_feat("radio_feat_pca"),
        "patch_row": rr,
        "patch_col": cc,
        "local_frame": local,
    }, order


def _feature_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    out = np.full((a.shape[0],), np.nan, dtype=np.float64)
    if not np.any(valid):
        return out
    aa = a[valid]
    bb = b[valid]
    denom = np.linalg.norm(aa, axis=1) * np.linalg.norm(bb, axis=1)
    keep = denom > 1e-12
    vals = np.full((aa.shape[0],), np.nan, dtype=np.float64)
    vals[keep] = np.sum(aa[keep] * bb[keep], axis=1) / denom[keep]
    out[np.where(valid)[0]] = vals
    return out


def _pair_array(pair: Mapping[str, Any], key: str, n: int, default: float = np.nan) -> np.ndarray:
    value = pair.get(key)
    if value is None:
        return np.full((n,), default, dtype=np.float64)
    arr = _np(value)
    if arr.shape[0] != n:
        return np.full((n,), default, dtype=np.float64)
    return arr.reshape(-1).astype(np.float64, copy=False)


def _random_mask(base: np.ndarray, keep_count: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros_like(base, dtype=bool)
    idx = np.where(base)[0]
    keep = min(int(keep_count), int(idx.size))
    if keep > 0:
        chosen = rng.choice(idx, size=keep, replace=False)
        out[chosen] = True
    return out


def _component_match_mask(
    pair: Mapping[str, Any],
    prev_s: Mapping[str, np.ndarray],
    curr_s: Mapping[str, np.ndarray],
    *,
    semantic_min_conf: float,
    min_feature_cos: float,
    component_match_mode: str,
) -> np.ndarray:
    n = int(_np(pair["prev_overlap_points"]).shape[0])
    base = prev_s["valid"] & curr_s["valid"]
    same_id = (prev_s["component"] == curr_s["component"]) & (prev_s["component"] >= 0)
    feature_cos = np.nan_to_num(_feature_cosine(prev_s["feat"], curr_s["feat"]), nan=-1.0)
    if component_match_mode == "id":
        return base & same_id
    if component_match_mode == "feature":
        return base & (feature_cos >= float(min_feature_cos))
    if component_match_mode == "feature_label":
        prev_label = _pair_array(pair, "prev_semantic_labels", n, -1).astype(np.int64)
        curr_label = _pair_array(pair, "curr_semantic_labels", n, -2).astype(np.int64)
        prev_sem_conf = _pair_array(pair, "prev_semantic_conf", n, 0.0)
        curr_sem_conf = _pair_array(pair, "curr_semantic_conf", n, 0.0)
        same_label = (prev_label == curr_label) & (prev_label >= 0)
        sem_conf = np.nan_to_num(np.minimum(prev_sem_conf, curr_sem_conf), nan=0.0)
        return base & same_label & (sem_conf >= float(semantic_min_conf)) & (feature_cos >= float(min_feature_cos))
    raise ValueError(f"unsupported component_match_mode={component_match_mode!r}")


def _candidate_masks(
    pair: Mapping[str, Any],
    prev_s: Mapping[str, np.ndarray],
    curr_s: Mapping[str, np.ndarray],
    *,
    semantic_min_conf: float,
    radio_min_confidence: float,
    radio_min_stability: float,
    radio_max_risk: float,
    radio_min_interior: float,
    min_feature_cos: float,
    component_match_mode: str,
    semantic_robust_sigma_m: float,
    semantic_harmful_threshold: float,
    seed: int,
    curr_chunk: int,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    n = int(_np(pair["prev_overlap_points"]).shape[0])
    prev_all = _np(pair["prev_overlap_points"], dtype=np.float64)
    curr_all = _np(pair["curr_overlap_points"], dtype=np.float64)
    pair_residual = np.linalg.norm(prev_all - curr_all, axis=1)
    pair_residual = np.nan_to_num(pair_residual, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
    robust_sigma = max(float(semantic_robust_sigma_m), 1e-6)
    residual_kernel = 1.0 / (1.0 + np.square(pair_residual / robust_sigma))
    prev_conf = _pair_array(pair, "prev_conf", n, 1.0)
    curr_conf = _pair_array(pair, "curr_conf", n, 1.0)
    geom_score = np.nan_to_num(np.minimum(prev_conf, curr_conf), nan=0.0, posinf=0.0, neginf=0.0)
    prev_label = _pair_array(pair, "prev_semantic_labels", n, -1).astype(np.int64)
    curr_label = _pair_array(pair, "curr_semantic_labels", n, -2).astype(np.int64)
    prev_sem_conf = _pair_array(pair, "prev_semantic_conf", n, 0.0)
    curr_sem_conf = _pair_array(pair, "curr_semantic_conf", n, 0.0)

    base = prev_s["valid"] & curr_s["valid"] & np.isfinite(geom_score)
    same_label = (prev_label == curr_label) & (prev_label >= 0)
    sem_conf = np.nan_to_num(np.minimum(prev_sem_conf, curr_sem_conf), nan=0.0)

    same_comp_id = (prev_s["component"] == curr_s["component"]) & (prev_s["component"] >= 0)
    radio_conf = np.nan_to_num(np.minimum(prev_s["confidence"], curr_s["confidence"]), nan=0.0)
    radio_stability = np.nan_to_num(np.minimum(prev_s["stability"], curr_s["stability"]), nan=0.0)
    radio_risk = np.nan_to_num(np.maximum(prev_s["risk"], curr_s["risk"]), nan=1.0)
    radio_activity_risk = np.nan_to_num(
        np.maximum.reduce([
            prev_s["dynamic"],
            curr_s["dynamic"],
            prev_s["sky"],
            curr_s["sky"],
            prev_s["lowtrust"],
            curr_s["lowtrust"],
        ]),
        nan=1.0,
    )
    radio_dynamic_sky = np.nan_to_num(
        np.maximum.reduce([
            prev_s["dynamic"],
            curr_s["dynamic"],
            prev_s["sky"],
            curr_s["sky"],
        ]),
        nan=1.0,
    )
    radio_lowtrust = np.nan_to_num(np.maximum(prev_s["lowtrust"], curr_s["lowtrust"]), nan=1.0)
    radio_interior = np.nan_to_num(np.minimum(prev_s["interior"], curr_s["interior"]), nan=0.0)
    feature_cos = np.nan_to_num(_feature_cosine(prev_s["feat"], curr_s["feat"]), nan=-1.0)
    same_feature = feature_cos >= float(min_feature_cos)
    same_feature_label = same_label & (sem_conf >= float(semantic_min_conf)) & same_feature
    if component_match_mode == "id":
        same_comp = same_comp_id
    elif component_match_mode == "feature":
        same_comp = same_feature
    elif component_match_mode == "feature_label":
        same_comp = same_feature_label
    else:
        raise ValueError(f"unsupported component_match_mode={component_match_mode!r}")
    radio_score = np.clip(
        0.35 * radio_conf + 0.25 * radio_stability + 0.20 * radio_interior + 0.20 * np.clip((feature_cos + 1.0) / 2.0, 0.0, 1.0),
        0.0,
        1.0,
    )
    risk_keep = np.clip(1.0 - radio_risk, 0.0, 1.0)
    weighted_score = geom_score * (
        0.30 * radio_conf
        + 0.20 * radio_stability
        + 0.15 * radio_interior
        + 0.15 * risk_keep
        + 0.10 * np.clip((feature_cos + 1.0) / 2.0, 0.0, 1.0)
        + 0.10 * same_comp.astype(np.float64)
    )
    radio_core = (
        base
        & same_comp
        & (radio_conf >= float(radio_min_confidence))
        & (radio_stability >= float(radio_min_stability))
        & (radio_risk <= float(radio_max_risk))
    )
    radio_semantic = radio_core & (feature_cos >= float(min_feature_cos))
    radio_robust = radio_core & (radio_risk <= float(radio_max_risk))
    radio_internal = radio_core & (radio_interior >= float(radio_min_interior))

    rng = np.random.default_rng(int(seed) + int(curr_chunk) * 1009)
    perm_comp = rng.permutation(n)
    perm_conf = rng.permutation(n)
    perm_feat = rng.permutation(n)
    label_perm = rng.permutation(n)
    conf_perm = rng.permutation(n)
    same_comp_id_shuf = (prev_s["component"] == curr_s["component"][perm_comp]) & (prev_s["component"] >= 0)
    feature_cos_shuf = np.nan_to_num(_feature_cosine(prev_s["feat"], curr_s["feat"][perm_feat]), nan=-1.0)
    label_shuffle = (prev_label == curr_label[label_perm]) & (prev_label >= 0)
    conf_shuffle = np.nan_to_num(np.minimum(prev_sem_conf, curr_sem_conf[conf_perm]), nan=0.0)
    same_feature_shuf = feature_cos_shuf >= float(min_feature_cos)
    same_feature_label_shuf = label_shuffle & (conf_shuffle >= float(semantic_min_conf)) & same_feature_shuf
    if component_match_mode == "id":
        same_comp_shuf = same_comp_id_shuf
    elif component_match_mode == "feature":
        same_comp_shuf = same_feature_shuf
    else:
        same_comp_shuf = same_feature_label_shuf
    conf_shuf = np.nan_to_num(np.minimum(prev_s["confidence"], curr_s["confidence"][perm_conf]), nan=0.0)
    stability_shuf = np.nan_to_num(np.minimum(prev_s["stability"], curr_s["stability"][perm_conf]), nan=0.0)
    risk_shuf = np.nan_to_num(np.maximum(prev_s["risk"], curr_s["risk"][perm_conf]), nan=1.0)
    activity_risk_shuf = np.nan_to_num(
        np.maximum.reduce([
            prev_s["dynamic"],
            curr_s["dynamic"][perm_conf],
            prev_s["sky"],
            curr_s["sky"][perm_conf],
            prev_s["lowtrust"],
            curr_s["lowtrust"][perm_conf],
        ]),
        nan=1.0,
    )
    dynamic_sky_shuf = np.nan_to_num(
        np.maximum.reduce([
            prev_s["dynamic"],
            curr_s["dynamic"][perm_conf],
            prev_s["sky"],
            curr_s["sky"][perm_conf],
        ]),
        nan=1.0,
    )
    lowtrust_shuf = np.nan_to_num(np.maximum(prev_s["lowtrust"], curr_s["lowtrust"][perm_conf]), nan=1.0)
    residual_kernel_shuf = residual_kernel[rng.permutation(n)]
    weighted_component_shuffle_score = geom_score * (
        0.30 * radio_conf
        + 0.20 * radio_stability
        + 0.15 * radio_interior
        + 0.15 * risk_keep
        + 0.10 * np.clip((feature_cos + 1.0) / 2.0, 0.0, 1.0)
        + 0.10 * same_comp_shuf.astype(np.float64)
    )
    weighted_feature_shuffle_score = geom_score * (
        0.30 * radio_conf
        + 0.20 * radio_stability
        + 0.15 * radio_interior
        + 0.15 * risk_keep
        + 0.10 * np.clip((feature_cos_shuf + 1.0) / 2.0, 0.0, 1.0)
        + 0.10 * same_comp.astype(np.float64)
    )
    weighted_confidence_shuffle_score = geom_score * (
        0.30 * conf_shuf
        + 0.20 * stability_shuf
        + 0.15 * radio_interior
        + 0.15 * np.clip(1.0 - risk_shuf, 0.0, 1.0)
        + 0.10 * np.clip((feature_cos + 1.0) / 2.0, 0.0, 1.0)
        + 0.10 * same_comp.astype(np.float64)
    )
    feature_sim = np.clip((feature_cos + 1.0) / 2.0, 0.0, 1.0)
    feature_sim_shuf = np.clip((feature_cos_shuf + 1.0) / 2.0, 0.0, 1.0)

    def plan_robust_kernel(
        conf: np.ndarray,
        stability: np.ndarray,
        risk: np.ndarray,
        same_component: np.ndarray,
    ) -> np.ndarray:
        robust = np.full((n,), 0.1, dtype=np.float64)
        uncertain = (
            (conf >= 0.75 * float(radio_min_confidence))
            & (stability >= 0.50 * float(radio_min_stability))
            & (risk <= float(radio_max_risk))
        )
        static = (
            same_component
            & (conf >= float(radio_min_confidence))
            & (stability >= float(radio_min_stability))
            & (risk <= float(radio_max_risk))
        )
        robust[uncertain] = 0.5
        robust[static] = 1.0
        return robust

    plan_robust = plan_robust_kernel(radio_conf, radio_stability, radio_risk, same_comp)
    plan_robust_component_shuffle = plan_robust_kernel(radio_conf, radio_stability, radio_risk, same_comp_shuf)
    plan_robust_temporal_shuffle = plan_robust_kernel(conf_shuf, stability_shuf, risk_shuf, same_comp)
    plan_weighted_kernel_score = geom_score * feature_sim * plan_robust
    plan_weighted_kernel_component_shuffle_score = geom_score * feature_sim * plan_robust_component_shuffle
    plan_weighted_kernel_feature_shuffle_score = geom_score * feature_sim_shuf * plan_robust
    plan_weighted_kernel_robust_shuffle_score = geom_score * feature_sim * plan_robust_temporal_shuffle
    plan_activity_robust = plan_robust_kernel(radio_conf, radio_stability, radio_activity_risk, same_comp)
    plan_activity_robust_component_shuffle = plan_robust_kernel(
        radio_conf,
        radio_stability,
        radio_activity_risk,
        same_comp_shuf,
    )
    plan_activity_robust_temporal_shuffle = plan_robust_kernel(
        conf_shuf,
        stability_shuf,
        activity_risk_shuf,
        same_comp,
    )
    plan_activity_kernel_score = geom_score * feature_sim * plan_activity_robust
    plan_activity_kernel_component_shuffle_score = geom_score * feature_sim * plan_activity_robust_component_shuffle
    plan_activity_kernel_feature_shuffle_score = geom_score * feature_sim_shuf * plan_activity_robust
    plan_activity_kernel_activity_shuffle_score = geom_score * feature_sim * plan_activity_robust_temporal_shuffle

    def semantic_role_weight(
        conf: np.ndarray,
        stability: np.ndarray,
        dynamic_sky: np.ndarray,
        lowtrust: np.ndarray,
        same_component: np.ndarray,
    ) -> np.ndarray:
        role = np.full((n,), 0.3, dtype=np.float64)
        harmful_threshold = float(semantic_harmful_threshold)
        stable = (
            same_component
            & (conf >= float(radio_min_confidence))
            & (stability >= float(radio_min_stability))
            & (dynamic_sky <= harmful_threshold)
            & (lowtrust <= harmful_threshold)
        )
        harmful = dynamic_sky > harmful_threshold
        role[stable] = 1.0
        role[harmful] = 0.1
        return role

    semantic_role = semantic_role_weight(
        radio_conf,
        radio_stability,
        radio_dynamic_sky,
        radio_lowtrust,
        same_comp,
    )
    semantic_role_shuf = semantic_role_weight(
        conf_shuf,
        stability_shuf,
        dynamic_sky_shuf,
        lowtrust_shuf,
        same_comp_shuf,
    )
    semantic_residual_kernel_score = geom_score * residual_kernel * semantic_role
    semantic_residual_kernel_residual_shuffle_score = geom_score * residual_kernel_shuf * semantic_role
    semantic_residual_kernel_semantic_shuffle_score = geom_score * residual_kernel * semantic_role_shuf
    same_entropy = _random_mask(base, int(radio_semantic.sum()), rng)
    same_degree = _random_mask(base, int(radio_internal.sum()), rng)

    out: Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str, Any]]] = {
        "geometry_only": (base, geom_score, {"control_kind": "geometry"}),
        "current_label_confidence_only": (
            base & same_label & (sem_conf >= float(semantic_min_conf)),
            sem_conf,
            {"control_kind": "current_semantic"},
        ),
        "current_label_shuffle": (
            base & label_shuffle & (sem_conf >= float(semantic_min_conf)),
            sem_conf,
            {"control_kind": "current_label_shuffle"},
        ),
        "current_confidence_shuffle": (
            base & same_label & (conf_shuffle >= float(semantic_min_conf)),
            conf_shuffle,
            {"control_kind": "current_confidence_shuffle"},
        ),
        "radio_only": (base & same_comp, radio_score, {"control_kind": "radio_component"}),
        "radio_semantic_correspondence": (
            radio_semantic,
            radio_score,
            {"control_kind": "radio_component_feature_conf_stability"},
        ),
        "radio_robust_kernel": (
            radio_robust,
            radio_score,
            {"control_kind": "radio_conf_stability_lowrisk"},
        ),
        "radio_object_internal_validation": (
            radio_internal,
            radio_score,
            {"control_kind": "radio_internal_lowrisk"},
        ),
        "radio_weighted_geometry": (
            base,
            weighted_score,
            {"control_kind": "radio_weighted_geometry_ranked_all_pairs"},
        ),
        "radio_internal_ranked_geometry": (
            base,
            geom_score * (0.55 * same_comp.astype(np.float64) + 0.30 * radio_interior + 0.15 * risk_keep),
            {"control_kind": "radio_internal_ranked_all_pairs"},
        ),
        "radio_plan_weighted_kernel_ranked": (
            base,
            plan_weighted_kernel_score,
            {
                "control_kind": "radio_plan_weighted_kernel_ranked_all_pairs",
                "plan_weighted_kernel_is_ranked_selector": True,
            },
        ),
        "radio_plan_weighted_kernel_weighted_fit": (
            base,
            plan_weighted_kernel_score,
            {
                "control_kind": "radio_plan_weighted_kernel_weighted_fit",
                "plan_weighted_kernel_is_ranked_selector": True,
                "use_weighted_umeyama_fit": True,
                "fit_weight_kind": "plan_weighted_kernel_score",
            },
        ),
        "radio_plan_activity_kernel_ranked": (
            base,
            plan_activity_kernel_score,
            {
                "control_kind": "radio_plan_activity_kernel_ranked_all_pairs",
                "plan_activity_kernel_is_ranked_selector": True,
                "plan_activity_kernel_excludes_boundary_from_active_risk": True,
            },
        ),
        "radio_plan_activity_kernel_weighted_fit": (
            base,
            plan_activity_kernel_score,
            {
                "control_kind": "radio_plan_activity_kernel_weighted_fit",
                "plan_activity_kernel_is_ranked_selector": True,
                "plan_activity_kernel_excludes_boundary_from_active_risk": True,
                "use_weighted_umeyama_fit": True,
                "fit_weight_kind": "plan_activity_kernel_score",
            },
        ),
        "radio_semantic_residual_kernel_ranked": (
            base,
            semantic_residual_kernel_score,
            {
                "control_kind": "radio_semantic_residual_kernel_ranked_all_pairs",
                "semantic_residual_kernel_is_ranked_selector": True,
            },
        ),
        "radio_semantic_residual_kernel_weighted_fit": (
            base,
            semantic_residual_kernel_score,
            {
                "control_kind": "radio_semantic_residual_kernel_weighted_fit",
                "semantic_residual_kernel_is_ranked_selector": True,
                "use_weighted_umeyama_fit": True,
                "fit_weight_kind": "semantic_residual_kernel_score",
            },
        ),
        "radio_feature_spatial_shuffle": (
            base
            & same_comp_shuf
            & (feature_cos_shuf >= float(min_feature_cos))
            & (radio_conf >= float(radio_min_confidence))
            & (radio_stability >= float(radio_min_stability)),
            radio_score,
            {"control_kind": "radio_feature_spatial_shuffle"},
        ),
        "radio_component_id_shuffle": (
            base
            & same_comp_shuf
            & (radio_conf >= float(radio_min_confidence))
            & (radio_stability >= float(radio_min_stability)),
            radio_score,
            {"control_kind": "radio_component_shuffle"},
        ),
        "radio_confidence_temporal_shuffle": (
            base
            & same_comp
            & (conf_shuf >= float(radio_min_confidence))
            & (stability_shuf >= float(radio_min_stability))
            & (risk_shuf <= float(radio_max_risk)),
            radio_score,
            {"control_kind": "radio_confidence_temporal_shuffle"},
        ),
        "radio_weighted_geometry_component_shuffle": (
            base,
            weighted_component_shuffle_score,
            {"control_kind": "radio_weighted_geometry_component_shuffle"},
        ),
        "radio_weighted_geometry_feature_shuffle": (
            base,
            weighted_feature_shuffle_score,
            {"control_kind": "radio_weighted_geometry_feature_shuffle"},
        ),
        "radio_weighted_geometry_confidence_temporal_shuffle": (
            base,
            weighted_confidence_shuffle_score,
            {"control_kind": "radio_weighted_geometry_confidence_temporal_shuffle"},
        ),
        "radio_plan_weighted_kernel_component_shuffle": (
            base,
            plan_weighted_kernel_component_shuffle_score,
            {"control_kind": "radio_plan_weighted_kernel_component_shuffle"},
        ),
        "radio_plan_weighted_kernel_feature_shuffle": (
            base,
            plan_weighted_kernel_feature_shuffle_score,
            {"control_kind": "radio_plan_weighted_kernel_feature_shuffle"},
        ),
        "radio_plan_weighted_kernel_robust_shuffle": (
            base,
            plan_weighted_kernel_robust_shuffle_score,
            {"control_kind": "radio_plan_weighted_kernel_robust_shuffle"},
        ),
        "radio_plan_weighted_kernel_feature_shuffle_weighted_fit": (
            base,
            plan_weighted_kernel_feature_shuffle_score,
            {
                "control_kind": "radio_plan_weighted_kernel_feature_shuffle_weighted_fit",
                "use_weighted_umeyama_fit": True,
                "fit_weight_kind": "plan_weighted_kernel_feature_shuffle_score",
            },
        ),
        "radio_plan_weighted_kernel_robust_shuffle_weighted_fit": (
            base,
            plan_weighted_kernel_robust_shuffle_score,
            {
                "control_kind": "radio_plan_weighted_kernel_robust_shuffle_weighted_fit",
                "use_weighted_umeyama_fit": True,
                "fit_weight_kind": "plan_weighted_kernel_robust_shuffle_score",
            },
        ),
        "radio_plan_activity_kernel_component_shuffle": (
            base,
            plan_activity_kernel_component_shuffle_score,
            {"control_kind": "radio_plan_activity_kernel_component_shuffle"},
        ),
        "radio_plan_activity_kernel_feature_shuffle": (
            base,
            plan_activity_kernel_feature_shuffle_score,
            {"control_kind": "radio_plan_activity_kernel_feature_shuffle"},
        ),
        "radio_plan_activity_kernel_activity_shuffle": (
            base,
            plan_activity_kernel_activity_shuffle_score,
            {"control_kind": "radio_plan_activity_kernel_activity_shuffle"},
        ),
        "radio_plan_activity_kernel_component_shuffle_weighted_fit": (
            base,
            plan_activity_kernel_component_shuffle_score,
            {
                "control_kind": "radio_plan_activity_kernel_component_shuffle_weighted_fit",
                "use_weighted_umeyama_fit": True,
                "fit_weight_kind": "plan_activity_kernel_component_shuffle_score",
            },
        ),
        "radio_plan_activity_kernel_feature_shuffle_weighted_fit": (
            base,
            plan_activity_kernel_feature_shuffle_score,
            {
                "control_kind": "radio_plan_activity_kernel_feature_shuffle_weighted_fit",
                "use_weighted_umeyama_fit": True,
                "fit_weight_kind": "plan_activity_kernel_feature_shuffle_score",
            },
        ),
        "radio_plan_activity_kernel_activity_shuffle_weighted_fit": (
            base,
            plan_activity_kernel_activity_shuffle_score,
            {
                "control_kind": "radio_plan_activity_kernel_activity_shuffle_weighted_fit",
                "use_weighted_umeyama_fit": True,
                "fit_weight_kind": "plan_activity_kernel_activity_shuffle_score",
            },
        ),
        "radio_semantic_residual_kernel_residual_shuffle": (
            base,
            semantic_residual_kernel_residual_shuffle_score,
            {"control_kind": "radio_semantic_residual_kernel_residual_shuffle"},
        ),
        "radio_semantic_residual_kernel_semantic_shuffle": (
            base,
            semantic_residual_kernel_semantic_shuffle_score,
            {"control_kind": "radio_semantic_residual_kernel_semantic_shuffle"},
        ),
        "radio_semantic_residual_kernel_residual_shuffle_weighted_fit": (
            base,
            semantic_residual_kernel_residual_shuffle_score,
            {
                "control_kind": "radio_semantic_residual_kernel_residual_shuffle_weighted_fit",
                "use_weighted_umeyama_fit": True,
                "fit_weight_kind": "semantic_residual_kernel_residual_shuffle_score",
            },
        ),
        "radio_semantic_residual_kernel_semantic_shuffle_weighted_fit": (
            base,
            semantic_residual_kernel_semantic_shuffle_score,
            {
                "control_kind": "radio_semantic_residual_kernel_semantic_shuffle_weighted_fit",
                "use_weighted_umeyama_fit": True,
                "fit_weight_kind": "semantic_residual_kernel_semantic_shuffle_score",
            },
        ),
        "same_entropy_random_proxy_attention": (
            same_entropy,
            geom_score,
            {"control_kind": "same_entropy_random_proxy_attention", "random_keep_count": int(radio_semantic.sum())},
        ),
        "same_degree_random_affinity_graph": (
            same_degree,
            geom_score,
            {"control_kind": "same_degree_random_affinity_graph", "random_keep_count": int(radio_internal.sum())},
        ),
    }
    for name, (_, _, meta) in out.items():
        meta.update({
            "base_valid_count": int(base.sum()),
            "same_component_count": int((base & same_comp).sum()),
            "same_component_id_count": int((base & same_comp_id).sum()),
            "same_feature_match_count": int((base & same_feature).sum()),
            "same_feature_label_match_count": int((base & same_feature_label).sum()),
            "component_match_mode": str(component_match_mode),
            "radio_core_count": int(radio_core.sum()),
            "median_radio_confidence": _finite_median(radio_conf[base]),
            "median_radio_stability": _finite_median(radio_stability[base]),
            "median_radio_risk": _finite_median(radio_risk[base]),
            "median_radio_activity_risk": _finite_median(radio_activity_risk[base]),
            "median_feature_cos": _finite_median(feature_cos[base]),
            "plan_weighted_kernel_static_count": int((base & (plan_robust >= 1.0)).sum()),
            "plan_weighted_kernel_uncertain_count": int((base & (plan_robust >= 0.5) & (plan_robust < 1.0)).sum()),
            "plan_weighted_kernel_active_lowtrust_count": int((base & (plan_robust <= 0.1)).sum()),
            "median_plan_weighted_kernel_score": _finite_median(plan_weighted_kernel_score[base]),
            "plan_activity_kernel_static_count": int((base & (plan_activity_robust >= 1.0)).sum()),
            "plan_activity_kernel_uncertain_count": int((base & (plan_activity_robust >= 0.5) & (plan_activity_robust < 1.0)).sum()),
            "plan_activity_kernel_active_lowtrust_count": int((base & (plan_activity_robust <= 0.1)).sum()),
            "median_plan_activity_kernel_score": _finite_median(plan_activity_kernel_score[base]),
            "semantic_robust_sigma_m": float(robust_sigma),
            "semantic_harmful_threshold": float(semantic_harmful_threshold),
            "semantic_residual_kernel_stable_count": int((base & (semantic_role >= 1.0)).sum()),
            "semantic_residual_kernel_context_count": int((base & (semantic_role > 0.1) & (semantic_role < 1.0)).sum()),
            "semantic_residual_kernel_harmful_count": int((base & (semantic_role <= 0.1)).sum()),
            "median_semantic_residual_kernel": _finite_median(residual_kernel[base]),
            "median_semantic_residual_role_weight": _finite_median(semantic_role[base]),
            "median_semantic_residual_kernel_score": _finite_median(semantic_residual_kernel_score[base]),
        })
    return out


def _estimate_sequence_median_overlap_residual(pair_files: Sequence[Path]) -> float:
    chunks: List[np.ndarray] = []
    for pair_file in pair_files:
        pair = _load_pair(pair_file)
        prev = _np(pair["prev_overlap_points"], dtype=np.float64)
        curr = _np(pair["curr_overlap_points"], dtype=np.float64)
        if prev.shape != curr.shape or prev.ndim != 2 or prev.shape[1] != 3:
            continue
        valid = np.isfinite(prev).all(axis=1) & np.isfinite(curr).all(axis=1)
        residual = np.linalg.norm(prev[valid] - curr[valid], axis=1)
        residual = residual[np.isfinite(residual)]
        if residual.size:
            chunks.append(residual)
    if not chunks:
        return 1.0
    merged = np.concatenate(chunks)
    sigma = float(np.median(merged))
    return sigma if math.isfinite(sigma) and sigma > 1e-6 else 1.0


def _select_indices(mask: np.ndarray, score: np.ndarray, max_points: int) -> np.ndarray:
    idx = np.where(mask & np.isfinite(score))[0]
    if idx.size == 0:
        return idx.astype(np.int64)
    values = score[idx]
    order = np.lexsort((idx, -values))
    idx = idx[order]
    if int(max_points) > 0:
        idx = idx[: int(max_points)]
    return idx.astype(np.int64)


def _weighted_fit_pair_correction(
    prev_points: np.ndarray,
    curr_points: np.ndarray,
    weights: np.ndarray,
    *,
    with_scale: bool,
) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    if prev_points.shape != curr_points.shape or prev_points.shape[0] < 3:
        return None, None, None, "not_enough_points"
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weights.shape[0] != prev_points.shape[0]:
        return None, None, None, "weight_length_mismatch"
    valid = (
        np.isfinite(prev_points).all(axis=1)
        & np.isfinite(curr_points).all(axis=1)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    if int(valid.sum()) < 3:
        return None, None, None, "not_enough_positive_weight_points"
    src = np.asarray(curr_points[valid], dtype=np.float64)
    dst = np.asarray(prev_points[valid], dtype=np.float64)
    w = np.asarray(weights[valid], dtype=np.float64)
    w_sum = float(w.sum())
    if not math.isfinite(w_sum) or w_sum <= 1e-12:
        return None, None, None, "nonpositive_weight_sum"
    w = w / w_sum
    try:
        mx = np.sum(src * w[:, None], axis=0)
        my = np.sum(dst * w[:, None], axis=0)
        x = src - mx
        y = dst - my
        cov = (y * w[:, None]).T @ x
        u, s, vt = np.linalg.svd(cov)
        d = np.eye(3)
        if np.linalg.det(u @ vt) < 0.0:
            d[-1, -1] = -1.0
        rot = u @ d @ vt
        if with_scale:
            var_x = float(np.sum(w * np.sum(x * x, axis=1)))
            scale = float(np.trace(np.diag(s) @ d) / max(var_x, 1e-12))
        else:
            scale = 1.0
        trans = my - scale * (rot @ mx)
    except Exception as exc:  # noqa: BLE001 - diagnostic records fit failures.
        return None, None, None, f"weighted_fit_error:{type(exc).__name__}:{exc}"
    if not np.isfinite(scale) or not np.all(np.isfinite(rot)) or not np.all(np.isfinite(trans)):
        return None, None, None, "weighted_fit_nonfinite"
    return float(scale), rot, trans, None


def _weight_summary(weights: Optional[np.ndarray]) -> Dict[str, Any]:
    if weights is None:
        return {
            "weighted_fit_used": False,
            "fit_weight_min": "",
            "fit_weight_median": "",
            "fit_weight_max": "",
            "fit_weight_effective_point_count": "",
        }
    w = np.asarray(weights, dtype=np.float64)
    w = w[np.isfinite(w) & (w > 0.0)]
    if w.size == 0:
        return {
            "weighted_fit_used": True,
            "fit_weight_min": "",
            "fit_weight_median": "",
            "fit_weight_max": "",
            "fit_weight_effective_point_count": 0.0,
        }
    ess = float((w.sum() * w.sum()) / max(float(np.sum(w * w)), 1e-12))
    return {
        "weighted_fit_used": True,
        "fit_weight_min": float(np.min(w)),
        "fit_weight_median": float(np.median(w)),
        "fit_weight_max": float(np.max(w)),
        "fit_weight_effective_point_count": ess,
    }


def _split_fit_validation(indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if indices.size < 6:
        return indices, np.empty((0,), dtype=np.int64)
    split = max(3, int(math.ceil(indices.size * 0.8)))
    split = min(split, indices.size - 3)
    return indices[:split], indices[split:]


def _rmse_for_indices(prev_all: np.ndarray, curr_all: np.ndarray, idx: np.ndarray) -> float:
    if idx.size < 3:
        return float("nan")
    return _rmse_dist(prev_all[idx], curr_all[idx])


def _candidate_family(name: str) -> str:
    if name in RADIO_CANDIDATES:
        return "radio"
    if name in CONTROL_CANDIDATES:
        return "control"
    if name in BASELINE_CANDIDATES:
        return "baseline"
    return "unknown"


def _best_by_chunk(rows: Sequence[Dict[str, Any]], family: Optional[str] = None) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        if family is not None and row.get("candidate_family") != family:
            continue
        chunk = int(row.get("curr_chunk"))
        current = out.get(chunk)
        if current is None or _float(row.get("best_mechanism_improvement")) > _float(current.get("best_mechanism_improvement")):
            out[chunk] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=_parse_source, required=True)
    parser.add_argument("--radio-sidecar-dir", type=Path, action="append", required=True)
    parser.add_argument("--overlap-pairs-dir", type=Path, default=None)
    parser.add_argument("--target-chunks", default="6,7,8,10,12,19,20,29,30,31,32")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--head-len", type=int, default=10)
    parser.add_argument("--max-fit-points", type=int, default=20000)
    parser.add_argument("--min-filter-fit-points", type=int, default=512)
    parser.add_argument("--semantic-min-conf", type=float, default=0.5)
    parser.add_argument("--radio-min-confidence", type=float, default=0.45)
    parser.add_argument("--radio-min-stability", type=float, default=0.35)
    parser.add_argument("--radio-max-risk", type=float, default=0.85)
    parser.add_argument("--radio-min-interior", type=float, default=0.25)
    parser.add_argument("--min-feature-cos", type=float, default=-0.10)
    parser.add_argument(
        "--semantic-robust-sigma-m",
        type=float,
        default=None,
        help=(
            "Fixed sigma for the O-A3 semantic residual robust kernel. "
            "When omitted, it is estimated as the median raw overlap residual "
            "over the selected target pair files."
        ),
    )
    parser.add_argument(
        "--semantic-harmful-threshold",
        type=float,
        default=0.5,
        help=(
            "Semantic role threshold for O-A3: dynamic/sky scores above this "
            "threshold get r_sem=0.1; stable evidence must stay below it."
        ),
    )
    parser.add_argument(
        "--component-match-mode",
        choices=["id", "feature", "feature_label"],
        default="id",
        help=(
            "Proxy used for cross-side RADIO object correspondence. id preserves "
            "the original local component-id behavior; feature_label is intended "
            "for RADSeg sidecars whose component ids are local to a cache."
        ),
    )
    parser.add_argument("--max-safe-rotation-deg", type=float, default=2.0)
    parser.add_argument("--max-safe-overlap-displacement-m", type=float, default=0.5)
    parser.add_argument("--max-safe-log-scale-delta", type=float, default=0.03)
    parser.add_argument("--min-raw-overlap-improvement-ratio", type=float, default=0.20)
    parser.add_argument("--min-validation-overlap-improvement-ratio", type=float, default=0.00)
    parser.add_argument("--min-mechanism-improvement-ratio", type=float, default=0.05)
    parser.add_argument("--max-ate-regression-m", type=float, default=0.30)
    parser.add_argument(
        "--require-object-internal-validation",
        action="store_true",
        help=(
            "Require held-out same-RADIO-component validation residual to improve. "
            "This is the MERGE_R4 guard against all-overlap overfit."
        ),
    )
    parser.add_argument("--damping-alpha", type=float, action="append", default=None)
    parser.add_argument("--center-mode", action="append", choices=CENTER_MODES, default=None)
    parser.add_argument("--scope", action="append", choices=["current", "boundary_span", "future"], default=None)
    parser.add_argument("--action-family", action="append", choices=["SE3_PAIR", "SIM3_PAIR"], default=None)
    parser.add_argument("--seed", type=int, default=7070)
    args = parser.parse_args()

    source_label, run_dir = args.source
    pairs_dir = args.overlap_pairs_dir or (run_dir / "overlap_pairs")
    target_chunks = set(parse_chunks(args.target_chunks))
    pair_files = [
        path for path in sorted(pairs_dir.glob("chunk_*_*.pt"))
        if int(path.stem.split("_")[-1]) in target_chunks
    ]
    if not pair_files:
        raise FileNotFoundError(f"No target overlap pair files in {pairs_dir}")

    sidecar_index = _index_sidecars(args.radio_sidecar_dir)
    sidecar_cache: Dict[int, Dict[str, Any]] = {}
    trace = _load_trace(run_dir / "merge_state_trace.jsonl")
    frames, poses, frame_to_chunk = _load_postmerge_trajectory(run_dir / "postmerge_global_pose.jsonl")
    _, _, gt_pos = _load_kitti_gt(args.gt)
    baseline = _make_baseline_row(frames, poses, gt_pos, trace, args.chunk_size, args.chunk_overlap, args.head_len)
    semantic_robust_sigma_m = (
        float(args.semantic_robust_sigma_m)
        if args.semantic_robust_sigma_m is not None
        else _estimate_sequence_median_overlap_residual(pair_files)
    )
    damping_alphas = args.damping_alpha if args.damping_alpha is not None else [0.5, 0.25]
    center_modes = args.center_mode if args.center_mode is not None else ["fit_centroid_interp", "current_chunk_mean_pose"]
    scopes = args.scope if args.scope is not None else ["current", "boundary_span", "future"]
    action_families = args.action_family if args.action_family is not None else ["SE3_PAIR", "SIM3_PAIR"]
    action_defs = [(name, name == "SIM3_PAIR") for name in action_families]

    rows: List[Dict[str, Any]] = []
    fit_failures: List[Dict[str, Any]] = []
    missing_sidecar_chunks: List[int] = []

    for pair_file in pair_files:
        pair = _load_pair(pair_file)
        curr_chunk = int(pair.get("curr_chunk"))
        prev_chunk = int(pair.get("prev_chunk", curr_chunk - 1))
        try:
            prev_sidecar = _load_sidecar(sidecar_index, prev_chunk, sidecar_cache)
            curr_sidecar = _load_sidecar(sidecar_index, curr_chunk, sidecar_cache)
        except FileNotFoundError:
            missing_sidecar_chunks.extend([prev_chunk, curr_chunk])
            fit_failures.append({
                "pair_file": str(pair_file),
                "prev_chunk": prev_chunk,
                "curr_chunk": curr_chunk,
                "fit_failure": "missing_radio_sidecar",
            })
            continue

        try:
            prev_s, prev_order = _sample_sidecar(pair, prev_sidecar, "prev")
            curr_s, curr_order = _sample_sidecar(pair, curr_sidecar, "curr")
        except Exception as exc:  # noqa: BLE001 - diagnostic records bad inputs.
            fit_failures.append({
                "pair_file": str(pair_file),
                "prev_chunk": prev_chunk,
                "curr_chunk": curr_chunk,
                "fit_failure": f"sidecar_sampling_error:{type(exc).__name__}:{exc}",
            })
            continue

        prev_all = _np(pair["prev_overlap_points"], dtype=np.float64)
        curr_all = _np(pair["curr_overlap_points"], dtype=np.float64)
        masks = _candidate_masks(
            pair,
            prev_s,
            curr_s,
            semantic_min_conf=float(args.semantic_min_conf),
            radio_min_confidence=float(args.radio_min_confidence),
            radio_min_stability=float(args.radio_min_stability),
            radio_max_risk=float(args.radio_max_risk),
            radio_min_interior=float(args.radio_min_interior),
            min_feature_cos=float(args.min_feature_cos),
            component_match_mode=str(args.component_match_mode),
            semantic_robust_sigma_m=float(semantic_robust_sigma_m),
            semantic_harmful_threshold=float(args.semantic_harmful_threshold),
            seed=int(args.seed),
            curr_chunk=curr_chunk,
        )

        for candidate_type, (mask, score, mask_meta) in masks.items():
            indices = _select_indices(mask, score, int(args.max_fit_points))
            fit_count = int(indices.size)
            if fit_count < int(args.min_filter_fit_points):
                fit_failures.append({
                    "pair_file": str(pair_file),
                    "prev_chunk": prev_chunk,
                    "curr_chunk": curr_chunk,
                    "candidate_type": candidate_type,
                    "candidate_family": _candidate_family(candidate_type),
                    "filter_fit_point_count": fit_count,
                    "valid_pair_count": int(mask_meta.get("base_valid_count", 0)),
                    "fit_failure": f"filter_support_too_small:{fit_count}<{int(args.min_filter_fit_points)}",
                    **mask_meta,
                })
                continue
            fit_idx, val_idx = _split_fit_validation(indices)
            prev_fit = prev_all[fit_idx]
            curr_fit = curr_all[fit_idx]
            fit_weights = (
                np.nan_to_num(score[fit_idx], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
                if bool(mask_meta.get("use_weighted_umeyama_fit"))
                else None
            )
            fit_weight_meta = _weight_summary(fit_weights)
            prev_val = prev_all[val_idx] if val_idx.size else np.empty((0, 3), dtype=np.float64)
            curr_val = curr_all[val_idx] if val_idx.size else np.empty((0, 3), dtype=np.float64)
            raw_fit_before = _rmse_dist(prev_fit, curr_fit)
            raw_val_before = _rmse_dist(prev_val, curr_val) if val_idx.size >= 3 else float("nan")
            for action_name, with_scale in action_defs:
                if fit_weights is not None:
                    full_scale, full_rot, full_trans, fail_reason = _weighted_fit_pair_correction(
                        prev_fit,
                        curr_fit,
                        fit_weights,
                        with_scale=with_scale,
                    )
                else:
                    full_scale, full_rot, full_trans, fail_reason = _fit_pair_correction(
                        prev_fit,
                        curr_fit,
                        with_scale=with_scale,
                    )
                if fail_reason is not None or full_scale is None or full_rot is None or full_trans is None:
                    fit_failures.append({
                        "pair_file": str(pair_file),
                        "prev_chunk": prev_chunk,
                        "curr_chunk": curr_chunk,
                        "candidate_type": candidate_type,
                        "candidate_family": _candidate_family(candidate_type),
                        "action_family": action_name,
                        "filter_fit_point_count": fit_count,
                        "fit_point_count": int(fit_idx.size),
                        "validation_point_count": int(val_idx.size),
                        "fit_failure": fail_reason,
                        "fit_weight_kind": str(mask_meta.get("fit_weight_kind", "")),
                        **fit_weight_meta,
                        **mask_meta,
                    })
                    continue
                for damping_alpha in damping_alphas:
                    scale = float(full_scale) ** float(damping_alpha) if with_scale else 1.0
                    rot = _rotation_power(full_rot, float(damping_alpha))
                    trans = float(damping_alpha) * full_trans
                    rot_deg = _rotation_delta_deg(rot)
                    abs_log_scale = abs(math.log(max(float(scale), 1e-12)))
                    safe_rotation = rot_deg <= float(args.max_safe_rotation_deg)
                    safe_scale = abs_log_scale <= float(args.max_safe_log_scale_delta)
                    alpha_tag = _safe_tag(str(damping_alpha))
                    damped_action_name = f"{action_name}_A{alpha_tag}"
                    for center_mode in center_modes:
                        center = _center_for_mode(
                            center_mode,
                            curr_fit,
                            frames,
                            poses,
                            frame_to_chunk,
                            curr_chunk,
                        )
                        if center is None:
                            fit_failures.append({
                                "pair_file": str(pair_file),
                                "prev_chunk": prev_chunk,
                                "curr_chunk": curr_chunk,
                                "candidate_type": candidate_type,
                                "candidate_family": _candidate_family(candidate_type),
                                "action_family": action_name,
                                "center_mode": center_mode,
                                "fit_failure": "missing_center",
                            })
                            continue
                        fit_corrected = _transform_points(
                            curr_fit,
                            center_mode=center_mode,
                            alpha=float(damping_alpha),
                            scale=scale,
                            rot=rot,
                            trans=trans,
                            full_scale=float(full_scale),
                            full_rot=full_rot,
                            full_trans=full_trans,
                            center=center,
                        )
                        val_corrected = (
                            _transform_points(
                                curr_val,
                                center_mode=center_mode,
                                alpha=float(damping_alpha),
                                scale=scale,
                                rot=rot,
                                trans=trans,
                                full_scale=float(full_scale),
                                full_rot=full_rot,
                                full_trans=full_trans,
                                center=center,
                            )
                            if val_idx.size >= 3
                            else np.empty((0, 3), dtype=np.float64)
                        )
                        raw_fit_after = _rmse_dist(prev_fit, fit_corrected)
                        raw_val_after = _rmse_dist(prev_val, val_corrected) if val_idx.size >= 3 else float("nan")
                        raw_fit_improvement = _safe_improvement_ratio(raw_fit_before, raw_fit_after)
                        raw_val_improvement = _safe_improvement_ratio(raw_val_before, raw_val_after)
                        overlap_displacement = _rmse_dist(curr_fit, fit_corrected)
                        safe_displacement = overlap_displacement <= float(args.max_safe_overlap_displacement_m)
                        safe_correction = bool(safe_rotation and safe_displacement and (safe_scale or not with_scale))

                        all_corrected = _transform_points(
                            curr_all,
                            center_mode=center_mode,
                            alpha=float(damping_alpha),
                            scale=scale,
                            rot=rot,
                            trans=trans,
                            full_scale=float(full_scale),
                            full_rot=full_rot,
                            full_trans=full_trans,
                            center=center,
                        )
                        same_comp = _component_match_mask(
                            pair,
                            prev_s,
                            curr_s,
                            semantic_min_conf=float(args.semantic_min_conf),
                            min_feature_cos=float(args.min_feature_cos),
                            component_match_mode=str(args.component_match_mode),
                        )
                        valid_radio = prev_s["valid"] & curr_s["valid"]
                        internal_idx = np.where(valid_radio & same_comp)[0]
                        cross_idx = np.where(valid_radio & ~same_comp)[0]
                        if val_idx.size:
                            val_internal_idx = val_idx[valid_radio[val_idx] & same_comp[val_idx]]
                        else:
                            val_internal_idx = np.empty((0,), dtype=np.int64)
                        internal_residual_before = _rmse_for_indices(prev_all, curr_all, internal_idx)
                        internal_residual = _rmse_for_indices(prev_all, all_corrected, internal_idx)
                        cross_residual_before = _rmse_for_indices(prev_all, curr_all, cross_idx)
                        cross_residual = _rmse_for_indices(prev_all, all_corrected, cross_idx)
                        internal_validation_before = _rmse_for_indices(prev_all, curr_all, val_internal_idx)
                        internal_validation_after = _rmse_for_indices(prev_all, all_corrected, val_internal_idx)
                        internal_validation_improvement = _safe_improvement_ratio(
                            internal_validation_before,
                            internal_validation_after,
                        )

                        for scope in scopes:
                            targets = _target_chunks(trace, curr_chunk, scope)
                            candidate_poses = _apply_centered_correction(
                                frames,
                                poses,
                                frame_to_chunk,
                                targets,
                                center_mode=center_mode,
                                alpha=float(damping_alpha),
                                scale=scale,
                                rot=rot,
                                trans=trans,
                                full_scale=float(full_scale),
                                full_rot=full_rot,
                                full_trans=full_trans,
                                center=center,
                            )
                            controller_rows = [
                                {
                                    "chunk_idx": int(c),
                                    "action": "radio_merge_oracle" if c in targets else "baseline",
                                    "source_scale": float(trace[c]["scale"]),
                                    "ctrl_scale": float(trace[c]["scale"]),
                                    "is_estimated_boundary": str(trace[c].get("row", {}).get("transform_reason", "")) == "estimated_overlap_transform",
                                }
                                for c in sorted(trace)
                            ]
                            candidate_name = (
                                f"{damped_action_name}_{scope}_b{curr_chunk}_"
                                f"center_{_safe_tag(center_mode)}_{_safe_tag(candidate_type)}"
                            )
                            metric = _metric_result_row(
                                candidate_name,
                                frames,
                                candidate_poses,
                                gt_pos,
                                baseline,
                                controller_rows,
                                int(baseline.get("source_nonunit_scale_count", 0)),
                                bool(baseline.get("source_has_scale_state", False)),
                                args.chunk_size,
                                args.chunk_overlap,
                                args.head_len,
                            )
                            best_mech = _best_mechanism_improvement(metric)
                            ate_delta = _float(metric.get("delta_vs_baseline_global_ate"))
                            raw_support_pass = bool(
                                math.isfinite(raw_fit_improvement)
                                and raw_fit_improvement >= float(args.min_raw_overlap_improvement_ratio)
                                and (
                                    not math.isfinite(raw_val_improvement)
                                    or raw_val_improvement >= float(args.min_validation_overlap_improvement_ratio)
                                )
                            )
                            mechanism_pass = bool(
                                math.isfinite(best_mech)
                                and best_mech >= float(args.min_mechanism_improvement_ratio)
                            )
                            ate_guard_pass = bool(
                                math.isfinite(ate_delta)
                                and ate_delta <= float(args.max_ate_regression_m)
                            )
                            object_internal_validation_pass = bool(
                                math.isfinite(internal_validation_improvement)
                                and internal_validation_improvement >= float(args.min_validation_overlap_improvement_ratio)
                            )
                            object_internal_gate_pass = bool(
                                object_internal_validation_pass
                                or not bool(args.require_object_internal_validation)
                            )
                            gate_pass = bool(
                                raw_support_pass
                                and mechanism_pass
                                and ate_guard_pass
                                and safe_correction
                                and object_internal_gate_pass
                            )
                            trace_row = trace.get(curr_chunk, {}).get("row", {})
                            emitted = [int(x) for x in trace_row.get("emitted_frame_ids", [])]
                            entry_frame = emitted[0] if emitted else int(trace_row.get("start_frame", -1))
                            boundary_jump_before = _boundary_jump(frames, poses, entry_frame - 1, entry_frame)
                            boundary_jump_after = _boundary_jump(frames, candidate_poses, entry_frame - 1, entry_frame)
                            j_v70_proxy = (
                                float(best_mech)
                                + 0.25 * float(raw_fit_improvement if math.isfinite(raw_fit_improvement) else 0.0)
                                + 0.10 * float(raw_val_improvement if math.isfinite(raw_val_improvement) else 0.0)
                                - max(0.0, ate_delta) / max(float(args.max_ate_regression_m), 1e-12)
                            )
                            metric.update({
                                "source_label": source_label,
                                "source_run": str(run_dir),
                                "overlap_pair_file": str(pair_file),
                                "prev_chunk": int(prev_chunk),
                                "curr_chunk": int(curr_chunk),
                                "candidate_type": candidate_type,
                                "candidate_family": _candidate_family(candidate_type),
                                "prev_pixel_coord_order": prev_order,
                                "curr_pixel_coord_order": curr_order,
                                "action_family": action_name,
                                "damped_action_family": damped_action_name,
                                "damping_alpha": float(damping_alpha),
                                "center_mode": center_mode,
                                "center_x": float(center[0]),
                                "center_y": float(center[1]),
                                "center_z": float(center[2]),
                                "scope": scope,
                                "target_chunk_count": int(len(targets)),
                                "target_chunks_first": int(targets[0]) if targets else None,
                                "target_chunks_last": int(targets[-1]) if targets else None,
                                "valid_pair_count": int(mask_meta.get("base_valid_count", 0)),
                                "filter_fit_point_count": int(fit_count),
                                "fit_point_count": int(fit_idx.size),
                                "validation_point_count": int(val_idx.size),
                                "fit_weight_kind": str(mask_meta.get("fit_weight_kind", "")),
                                **fit_weight_meta,
                                "raw_overlap_fit_before_m": raw_fit_before,
                                "raw_overlap_fit_after_m": raw_fit_after,
                                "raw_overlap_fit_improvement_ratio": raw_fit_improvement,
                                "raw_overlap_validation_before_m": raw_val_before,
                                "raw_overlap_validation_after_m": raw_val_after,
                                "raw_overlap_validation_improvement_ratio": raw_val_improvement,
                                "object_internal_validation_before_m": internal_validation_before,
                                "object_internal_validation_after_m": internal_validation_after,
                                "object_internal_validation_improvement_ratio": internal_validation_improvement,
                                "object_internal_validation_point_count": int(val_internal_idx.size),
                                "object_internal_validation_residual_m": internal_residual,
                                "object_internal_validation_residual_before_m": internal_residual_before,
                                "object_cross_boundary_residual_m": cross_residual,
                                "object_cross_boundary_residual_before_m": cross_residual_before,
                                "raw_overlap_before_m": raw_fit_before,
                                "raw_overlap_after_m": raw_fit_after,
                                "raw_overlap_improvement_ratio": raw_fit_improvement,
                                "correction_overlap_displacement_m": overlap_displacement,
                                "correction_scale": float(scale),
                                "correction_abs_log_scale_delta": abs_log_scale,
                                "correction_rotation_deg": rot_deg,
                                "correction_translation_norm_m": float(np.linalg.norm(trans)),
                                "safe_rotation_pass": safe_rotation,
                                "safe_overlap_displacement_pass": safe_displacement,
                                "safe_scale_pass": safe_scale if with_scale else True,
                                "safe_correction_pass": safe_correction,
                                "best_mechanism_improvement": best_mech,
                                "J_v70_offline_merge_proxy": j_v70_proxy,
                                "J_v70_official_online": "",
                                "J_v70_is_offline_proxy": True,
                                "raw_support_pass": raw_support_pass,
                                "mechanism_pass": mechanism_pass,
                                "ate_guard_pass": ate_guard_pass,
                                "object_internal_validation_pass": object_internal_validation_pass,
                                "object_internal_validation_required": bool(args.require_object_internal_validation),
                                "object_internal_validation_match_mode": str(args.component_match_mode),
                                "object_internal_gate_pass": object_internal_gate_pass,
                                "oracle_action_gate_pass": gate_pass,
                                "boundary_jump_before_m": boundary_jump_before,
                                "boundary_jump_after_m": boundary_jump_after,
                                "boundary_jump_delta_m": (
                                    boundary_jump_after - boundary_jump_before
                                    if math.isfinite(boundary_jump_before) and math.isfinite(boundary_jump_after)
                                    else float("nan")
                                ),
                                "tail100_rmse_delta_raw_pose_m": _tail100_delta(frames, candidate_poses, poses, gt_pos),
                                **mask_meta,
                            })
                            rows.append(metric)

    rows.sort(key=lambda row: (
        not bool(row.get("oracle_action_gate_pass")),
        -_float(row.get("best_mechanism_improvement")),
        -_float(row.get("raw_overlap_validation_improvement_ratio")),
        _float(row.get("delta_vs_baseline_global_ate")),
        str(row.get("candidate")),
    ))
    gate_rows = [row for row in rows if bool(row.get("oracle_action_gate_pass"))]
    radio_gate_rows = [row for row in gate_rows if row.get("candidate_family") == "radio"]
    control_gate_rows = [row for row in gate_rows if row.get("candidate_family") == "control"]
    baseline_gate_rows = [row for row in gate_rows if row.get("candidate_family") == "baseline"]
    best_radio = _best_by_chunk(rows, family="radio")
    best_control = _best_by_chunk(rows, family="control")
    best_baseline = _best_by_chunk(rows, family="baseline")
    radio_beats_controls_chunks: List[int] = []
    for chunk, row in best_radio.items():
        radio_mech = _float(row.get("best_mechanism_improvement"))
        control_mech = _float(best_control.get(chunk, {}).get("best_mechanism_improvement"), -1e9)
        baseline_mech = _float(best_baseline.get(chunk, {}).get("best_mechanism_improvement"), -1e9)
        if bool(row.get("oracle_action_gate_pass")) and radio_mech > max(control_mech, baseline_mech):
            radio_beats_controls_chunks.append(int(chunk))
    best_radio_gate_by_chunk = _best_by_chunk(radio_gate_rows, family="radio")
    best_radio_mech_values = [row.get("best_mechanism_improvement") for row in best_radio_gate_by_chunk.values()]
    r5_gate_pass = bool(
        len(radio_beats_controls_chunks) >= 4
        and (_finite_median(best_radio_mech_values) or float("-inf")) >= float(args.min_mechanism_improvement_ratio)
    )

    candidate_counts: Dict[str, Dict[str, int]] = {}
    for row in rows:
        key = str(row.get("candidate_type"))
        candidate_counts.setdefault(key, {"rows": 0, "gate_pass": 0})
        candidate_counts[key]["rows"] += 1
        candidate_counts[key]["gate_pass"] += int(bool(row.get("oracle_action_gate_pass")))

    summary = {
        "schema": "acl2_v70_radio_merge_oracle_summary_v1",
        "created_at": utc_now(),
        "source_label": source_label,
        "source_run": str(run_dir),
        "overlap_pairs_dir": str(pairs_dir),
        "radio_sidecar_dirs": [str(x) for x in args.radio_sidecar_dir],
        "target_chunks": sorted(target_chunks),
        "pair_files": len(pair_files),
        "rows": len(rows),
        "fit_failures": fit_failures,
        "missing_sidecar_chunks": sorted(set(missing_sidecar_chunks)),
        "counts": {
            "safe_correction_pass": sum(bool(row.get("safe_correction_pass")) for row in rows),
            "raw_support_pass": sum(bool(row.get("raw_support_pass")) for row in rows),
            "mechanism_pass": sum(bool(row.get("mechanism_pass")) for row in rows),
            "ate_guard_pass": sum(bool(row.get("ate_guard_pass")) for row in rows),
            "oracle_action_gate_pass": len(gate_rows),
            "radio_gate_rows": len(radio_gate_rows),
            "control_gate_rows": len(control_gate_rows),
            "baseline_gate_rows": len(baseline_gate_rows),
        },
        "candidate_counts": candidate_counts,
        "radio_gate_chunks": sorted({int(row.get("curr_chunk")) for row in radio_gate_rows}),
        "control_gate_chunks": sorted({int(row.get("curr_chunk")) for row in control_gate_rows}),
        "baseline_gate_chunks": sorted({int(row.get("curr_chunk")) for row in baseline_gate_rows}),
        "radio_beats_controls_chunks": sorted(radio_beats_controls_chunks),
        "median_best_radio_gate_mechanism_improvement": _finite_median(best_radio_mech_values),
        "mean_raw_overlap_fit_improvement_ratio": _finite_mean(row.get("raw_overlap_fit_improvement_ratio") for row in rows),
        "mean_raw_overlap_validation_improvement_ratio": _finite_mean(row.get("raw_overlap_validation_improvement_ratio") for row in rows),
        "mean_best_mechanism_improvement": _finite_mean(row.get("best_mechanism_improvement") for row in rows),
        "gate_rule": {
            "min_filter_fit_points": int(args.min_filter_fit_points),
            "semantic_min_conf": float(args.semantic_min_conf),
            "radio_min_confidence": float(args.radio_min_confidence),
            "radio_min_stability": float(args.radio_min_stability),
            "radio_max_risk": float(args.radio_max_risk),
            "radio_min_interior": float(args.radio_min_interior),
            "min_feature_cos": float(args.min_feature_cos),
            "semantic_robust_sigma_m": float(semantic_robust_sigma_m),
            "semantic_robust_sigma_source": (
                "cli" if args.semantic_robust_sigma_m is not None else "target_pair_median_raw_overlap_residual"
            ),
            "semantic_harmful_threshold": float(args.semantic_harmful_threshold),
            "component_match_mode": str(args.component_match_mode),
            "min_raw_overlap_improvement_ratio": float(args.min_raw_overlap_improvement_ratio),
            "min_validation_overlap_improvement_ratio": float(args.min_validation_overlap_improvement_ratio),
            "min_mechanism_improvement_ratio": float(args.min_mechanism_improvement_ratio),
            "max_ate_regression_m": float(args.max_ate_regression_m),
            "require_object_internal_validation": bool(args.require_object_internal_validation),
            "max_safe_rotation_deg": float(args.max_safe_rotation_deg),
            "max_safe_overlap_displacement_m": float(args.max_safe_overlap_displacement_m),
            "max_safe_log_scale_delta": float(args.max_safe_log_scale_delta),
            "damping_alphas": [float(x) for x in damping_alphas],
            "center_modes": list(center_modes),
            "scopes": list(scopes),
            "action_families": list(action_families),
            "r5_pass_rule": ">=4 chunks with RADIO gate row beating best control/baseline, plus median best RADIO gate mechanism improvement >= threshold",
        },
        "best_row": rows[0] if rows else {},
        "best_radio_row": next((row for row in rows if row.get("candidate_family") == "radio"), {}),
        "best_control_row": next((row for row in rows if row.get("candidate_family") == "control"), {}),
        "best_baseline_row": next((row for row in rows if row.get("candidate_family") == "baseline"), {}),
        "r5_merge_oracle_gate_pass": r5_gate_pass,
        "r6_online_allowed_by_this_oracle": r5_gate_pass,
        "decision": "allow_r6_smoke_if_other_r5_gates_pass" if r5_gate_pass else "no_go_r6_continue_r5_repair",
        "note": (
            "Offline pose-level MERGE oracle. J_v70_offline_merge_proxy is a documented proxy, "
            "not an official online HMC J_v70 or 704F ATE result."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "radio_merge_oracle_results.csv", rows)
    (args.out_dir / "radio_merge_oracle_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True))


if __name__ == "__main__":
    main()
