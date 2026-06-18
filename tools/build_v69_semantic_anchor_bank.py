#!/usr/bin/env python3
"""Build ACL2 v69 per-chunk Semantic Anchor Bank artifacts.

This is an audit/diagnostic builder. It consumes the dense Stage-C semantic
label/confidence maps, H35 parity per-chunk geometry, and optional v68 selected
feature dumps. Chunks without v68 features are still materialized, but are
marked ``gram_motion_unavailable=True`` in both the .pt payload and summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loger.pipeline.semantic_prior_generator import (  # noqa: E402
    _mode_pool_dense_semantic_patches,
    _normalize_dense_semantic_confidence,
)

try:
    from scipy import ndimage
except Exception:  # pragma: no cover - handled at runtime.
    ndimage = None


VERTICAL_KEYS = (
    "building",
    "house",
    "wall",
    "handrail_or_fence",
    "fence",
    "pole",
    "traffic sign",
    "traffic light",
    "bridge",
    "billboard",
    "construction",
)
ROAD_KEYS = ("road", "ground", "crosswalk", "sidewalk", "floor")
DYNAMIC_KEYS = ("person", "rider", "bicycle", "motorcycle", "bus", "truck", "car")
SKY_KEYS = ("sky", "cloud")
VEGETATION_KEYS = ("tree", "grass", "mountain", "vegetation", "plant", "terrain")
LOWTRUST_KEYS = ("void", "unknown", "unlabeled")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", default="01")
    parser.add_argument("--stage-c-cache", required=True, type=Path)
    parser.add_argument("--full-semantic-pt", required=True, type=Path)
    parser.add_argument("--per-chunk-geometry-dir", required=True, type=Path)
    parser.add_argument("--v68-feature-dir", type=Path)
    parser.add_argument("--v68-selection-json", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--patch-grid", nargs=2, type=int, default=(19, 66))
    parser.add_argument("--anchor-threshold", type=float, default=0.25)
    parser.add_argument("--profile-name", default="default")
    parser.add_argument("--component-mode", choices=("connected", "label_group"), default="connected")
    parser.add_argument("--split-road-boundary", action="store_true")
    parser.add_argument("--allow-vertical-static-weak", action="store_true")
    parser.add_argument("--vertical-static-weak-threshold", type=float, default=0.12)
    parser.add_argument("--max-instances-per-chunk", type=int, default=160)
    parser.add_argument("--target-chunks", default="6,7,8,10,12,19,20,29,30,31,32")
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def _torch_load(path: Path) -> Dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(obj)}")
    return obj


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _ids_containing(label_names: Sequence[str], words: Iterable[str]) -> List[int]:
    lowered = [str(x).lower() for x in label_names]
    keys = [str(w).lower() for w in words]
    return [idx for idx, name in enumerate(lowered) if any(k in name for k in keys)]


def _mask_from_ids(labels: torch.Tensor, ids: Sequence[int]) -> torch.Tensor:
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for idx in ids:
        mask |= labels == int(idx)
    return mask


def _robust01(x: torch.Tensor) -> torch.Tensor:
    vals = x.detach().cpu().float()
    finite = torch.isfinite(vals)
    if not bool(finite.any().item()):
        return torch.zeros_like(vals)
    good = vals[finite]
    lo = torch.quantile(good, 0.05)
    hi = torch.quantile(good, 0.95)
    if float((hi - lo).abs().item()) < 1e-8:
        lo = good.min()
        hi = good.max()
    return ((vals - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)


def _road_boundary(road_or_ground: torch.Tensor) -> torch.Tensor:
    edge = torch.zeros_like(road_or_ground, dtype=torch.bool)
    edge[:, 1:, :] |= road_or_ground[:, 1:, :] != road_or_ground[:, :-1, :]
    edge[:, :-1, :] |= road_or_ground[:, 1:, :] != road_or_ground[:, :-1, :]
    edge[:, :, 1:] |= road_or_ground[:, :, 1:] != road_or_ground[:, :, :-1]
    edge[:, :, :-1] |= road_or_ground[:, :, 1:] != road_or_ground[:, :, :-1]
    pooled = F.max_pool2d(edge[:, None].float(), kernel_size=3, stride=1, padding=1)
    return pooled[:, 0] > 0


def _patch_semantic(stage_payload: Mapping[str, Any], patch_grid: Tuple[int, int]) -> Dict[str, Any]:
    sem = stage_payload.get("semantic_segmentation")
    if not isinstance(sem, Mapping):
        raise KeyError("Stage-C payload lacks semantic_segmentation dict")
    labels = sem.get("label_maps")
    if not torch.is_tensor(labels):
        raise KeyError("Stage-C semantic_segmentation lacks label_maps tensor")
    conf_raw = sem.get("confidence_maps")
    conf_norm, _ = _normalize_dense_semantic_confidence(conf_raw, target_shape=tuple(labels.shape))
    if conf_norm is None:
        conf_norm = torch.ones_like(labels, dtype=torch.float32)
    patch_label, purity, patch_conf = _mode_pool_dense_semantic_patches(
        labels.detach().cpu().long(),
        conf_norm.detach().cpu().float(),
        patch_grid=patch_grid,
    )
    label_names = [str(x) for x in (sem.get("label_names") or [])]
    trust = (patch_conf.float().clamp(0.0, 1.0) * purity.float().clamp(0.0, 1.0).square()).clamp(0.0, 1.0)

    vertical = _mask_from_ids(patch_label, _ids_containing(label_names, VERTICAL_KEYS))
    ground = _mask_from_ids(patch_label, _ids_containing(label_names, ROAD_KEYS))
    dynamic = _mask_from_ids(patch_label, _ids_containing(label_names, DYNAMIC_KEYS))
    sky = _mask_from_ids(patch_label, _ids_containing(label_names, SKY_KEYS))
    vegetation = _mask_from_ids(patch_label, _ids_containing(label_names, VEGETATION_KEYS))
    lowtrust_label = _mask_from_ids(patch_label, _ids_containing(label_names, LOWTRUST_KEYS))
    boundary = _road_boundary(ground)
    lowtrust = lowtrust_label | (patch_conf < 0.50) | (purity < 0.50)

    static_support = (
        1.00 * vertical.float()
        + 0.70 * (boundary & ground).float()
        + 0.30 * (ground & ~boundary).float()
    ).clamp(0.0, 1.0)
    risk = (
        1.00 * dynamic.float()
        + 0.90 * sky.float()
        + 0.70 * vegetation.float()
        + 1.00 * lowtrust.float()
        + 0.80 * boundary.float() * (1.0 - vertical.float())
    ).clamp(0.0, 1.0)

    return {
        "label_patch": patch_label.long(),
        "semantic_confidence_patch": patch_conf.float().clamp(0.0, 1.0),
        "patch_purity": purity.float().clamp(0.0, 1.0),
        "semantic_trust": trust.float(),
        "label_names": label_names,
        "vertical_mask": vertical,
        "ground_mask": ground,
        "road_boundary_mask": boundary & ground,
        "dynamic_mask": dynamic,
        "sky_mask": sky,
        "vegetation_mask": vegetation,
        "lowtrust_mask": lowtrust,
        "semantic_boundary_mask": boundary,
        "static_support": static_support.float(),
        "semantic_risk": risk.float(),
    }


def _area_resize_3d(x: torch.Tensor, patch_grid: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(x[:, None].float(), size=patch_grid, mode="area")[:, 0].float()


def _resize_points_to_patch(points: torch.Tensor, patch_grid: Tuple[int, int]) -> torch.Tensor:
    comps = []
    for dim in range(3):
        comps.append(_area_resize_3d(points[..., dim].float(), patch_grid))
    return torch.stack(comps, dim=-1)


def _geometry_patch(geom_path: Path, patch_grid: Tuple[int, int]) -> Dict[str, Any]:
    payload = _torch_load(geom_path)
    conf = payload.get("conf")
    if not torch.is_tensor(conf):
        raise KeyError(f"No conf tensor in {geom_path}")
    geom_conf = _area_resize_3d(conf.detach().cpu().float().clamp(0.0, 1.0), patch_grid).clamp(0.0, 1.0)
    D_geo = (1.0 - geom_conf).clamp(0.0, 1.0)
    points = payload.get("points")
    patch_points = None
    if torch.is_tensor(points):
        patch_points = _resize_points_to_patch(points.detach().cpu().float(), patch_grid)
    return {
        "geom_conf": geom_conf,
        "D_geo": D_geo,
        "patch_points": patch_points,
        "geometry_payload": payload,
    }


def _parse_selected_layers(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return [{"tap": "global_k_raw_patchvec_layers", "layer": 5}, {"tap": "global_k_raw_patchvec_layers", "layer": 7}]
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("selected_layers") or []
    out: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        tap = str(row.get("tap") or "")
        if not tap:
            continue
        layer = int(row.get("layer"))
        key = (tap, layer)
        if key in seen:
            continue
        seen.add(key)
        out.append({"tap": tap, "layer": layer})
    return out


def _feature_for_layer(payload: Mapping[str, Any], tap: str, layer: int) -> Optional[torch.Tensor]:
    tensor = payload.get(f"tap::{tap}")
    if not torch.is_tensor(tensor):
        return None
    meta = dict(dict(payload.get("taps") or {}).get(tap) or {})
    selected_layers = [int(x) for x in (meta.get("selected_layers") or [])]
    if selected_layers:
        if int(layer) not in selected_layers:
            return None
        pos = selected_layers.index(int(layer))
    else:
        pos = int(layer)
    if tensor.ndim != 5 or pos < 0 or pos >= int(tensor.shape[1]):
        return None
    return tensor[:, pos].detach().cpu().float()


def _gram_row_temporal_instability(features: torch.Tensor) -> torch.Tensor:
    T, H, W, D = [int(x) for x in features.shape]
    if T < 2:
        return torch.zeros((T, H, W), dtype=torch.float32)
    flat = torch.nn.functional.normalize(features.reshape(T, H * W, D).float(), dim=-1, eps=1e-6)
    mean: Optional[torch.Tensor] = None
    m2: Optional[torch.Tensor] = None
    count = 0
    for t in range(T):
        gram = flat[t] @ flat[t].T
        count += 1
        if mean is None:
            mean = gram.clone()
            m2 = torch.zeros_like(gram)
            continue
        delta = gram - mean
        mean = mean + delta / float(count)
        assert m2 is not None
        m2 = m2 + delta * (gram - mean)
    assert m2 is not None
    var_rows = (m2 / float(max(count - 1, 1))).mean(dim=1)
    score = _robust01(var_rows.reshape(H, W))
    return score[None].repeat(T, 1, 1)


def _gram_motion_for_chunk(
    feature_path: Path,
    selected_layers: Sequence[Mapping[str, Any]],
    patch_grid: Tuple[int, int],
) -> Tuple[torch.Tensor, bool, List[Dict[str, Any]]]:
    if not feature_path.exists():
        return torch.zeros((0, int(patch_grid[0]), int(patch_grid[1])), dtype=torch.float32), True, []
    payload = _torch_load(feature_path)
    motion_parts: List[torch.Tensor] = []
    debug: List[Dict[str, Any]] = []
    for item in selected_layers:
        tap = str(item["tap"])
        layer = int(item["layer"])
        feat = _feature_for_layer(payload, tap, layer)
        row: Dict[str, Any] = {"tap": tap, "layer": layer, "available": feat is not None}
        if feat is not None:
            score = _gram_row_temporal_instability(feat)
            if tuple(score.shape[-2:]) != tuple(patch_grid):
                score = F.interpolate(score[:, None].float(), size=patch_grid, mode="bilinear", align_corners=False)[:, 0]
            motion_parts.append(score.float())
            row.update({"motion_mean": float(score.mean().item()), "motion_q90": float(torch.quantile(score.reshape(-1), 0.90).item())})
        debug.append(row)
    if not motion_parts:
        T = int(payload.get("end_frame", 0)) - int(payload.get("start_frame", 0))
        return torch.zeros((max(T, 1), int(patch_grid[0]), int(patch_grid[1])), dtype=torch.float32), True, debug
    return torch.stack(motion_parts, dim=0).mean(dim=0).clamp(0.0, 1.0), False, debug


def _spatial_entropy(mask_or_weight: torch.Tensor, grid_h: int = 4, grid_w: int = 6) -> Tuple[float, float]:
    weight = mask_or_weight.detach().cpu().float()
    if weight.ndim != 3:
        weight = weight.reshape(-1, int(weight.shape[-2]), int(weight.shape[-1]))
    _, H, W = [int(x) for x in weight.shape]
    masses: List[float] = []
    for gy in range(grid_h):
        y0 = int(round(gy * H / grid_h))
        y1 = int(round((gy + 1) * H / grid_h))
        for gx in range(grid_w):
            x0 = int(round(gx * W / grid_w))
            x1 = int(round((gx + 1) * W / grid_w))
            masses.append(float(weight[:, y0:y1, x0:x1].sum().item()))
    total = sum(masses)
    if total <= 0.0:
        return 0.0, 0.0
    probs = [m / total for m in masses if m > 0.0]
    entropy = -sum(p * math.log(p) for p in probs) / math.log(len(masses))
    coverage = sum(1 for m in masses if m > 0.0) / len(masses)
    return float(entropy), float(coverage)


def _condition_score(points_patch: Optional[torch.Tensor], weight: torch.Tensor) -> Optional[float]:
    if points_patch is None:
        return None
    mask = torch.isfinite(points_patch).all(dim=-1) & torch.isfinite(weight) & (weight > 0)
    if int(mask.sum().item()) < 16:
        return None
    pts = points_patch[mask].float().reshape(-1, 3)
    w = weight[mask].float().reshape(-1)
    wsum = w.sum()
    if not torch.isfinite(wsum) or float(wsum.item()) <= 0.0:
        return None
    mean = (pts * w[:, None]).sum(dim=0) / wsum
    centered = pts - mean[None, :]
    cov = (centered * w[:, None]).T @ centered / wsum
    try:
        s = torch.linalg.svdvals(cov)
    except RuntimeError:
        return None
    if s.numel() < 3 or float(s[0].item()) <= 0.0:
        return None
    cond = torch.sqrt((s[1] / (s[0] + 1e-9)).clamp_min(0.0) * (s[2] / (s[0] + 1e-9)).clamp_min(0.0))
    return float((5.0 * cond).clamp(0.0, 1.0).item())


def _label_name(label_names: Sequence[str], label_id: int) -> str:
    if 0 <= int(label_id) < len(label_names):
        return str(label_names[int(label_id)])
    return f"id_{int(label_id)}"


def _semantic_group(label: str) -> str:
    name = str(label).lower()
    if any(k in name for k in DYNAMIC_KEYS):
        return "dynamic"
    if any(k in name for k in SKY_KEYS):
        return "sky"
    if any(k in name for k in VEGETATION_KEYS):
        return "vegetation"
    if any(k in name for k in VERTICAL_KEYS):
        return "vertical_static"
    if any(k in name for k in ROAD_KEYS):
        return "ground_or_road"
    if any(k in name for k in LOWTRUST_KEYS):
        return "lowtrust"
    return "other"


def _frame_frac(mask: torch.Tensor, frame_slice: slice) -> float:
    sub = mask[frame_slice]
    if sub.numel() == 0:
        return 0.0
    return float((sub.reshape(sub.shape[0], -1).sum(dim=1) > 0).float().mean().item())


def _instance_rows(
    *,
    chunk_id: int,
    frame_start: int,
    frame_end: int,
    candidate_mask: torch.Tensor,
    anchor_score: torch.Tensor,
    risk_score: torch.Tensor,
    sem: Mapping[str, Any],
    geom: Mapping[str, Any],
    gram_motion: torch.Tensor,
    gram_unavailable: bool,
    anchor_threshold: float,
    allow_vertical_static_weak: bool,
    max_instances: int,
    component_mode: str,
    split_road_boundary: bool,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    T, H, W = [int(x) for x in candidate_mask.shape]
    anchor_id_map = torch.full((T, H, W), -1, dtype=torch.long)
    instances: List[Dict[str, Any]] = []
    current_id = 0
    label_names = list(sem["label_names"])
    labels = sem["label_patch"].long()
    structure = np.ones((3, 3, 3), dtype=np.int8)

    for label_id in sorted(int(x) for x in torch.unique(labels[candidate_mask]).tolist()):
        label_mask = (labels == label_id) & candidate_mask
        if not bool(label_mask.any().item()):
            continue
        label = _label_name(label_names, label_id)
        group = _semantic_group(label)
        if component_mode == "label_group":
            if split_road_boundary and group == "ground_or_road":
                boundary = label_mask & sem["road_boundary_mask"]
                nonboundary = label_mask & ~sem["road_boundary_mask"]
                comp_ids = []
                if bool(boundary.any().item()):
                    comp_ids.append((1, boundary.numpy()))
                if bool(nonboundary.any().item()):
                    comp_ids.append((2, nonboundary.numpy()))
            else:
                comp_ids = [(1, label_mask.numpy())]
        elif ndimage is None:
            comp_arr = label_mask.numpy().astype(np.int32)
            comp_ids = [(1, comp_arr)]
        else:
            comp_arr, num = ndimage.label(label_mask.numpy().astype(np.uint8), structure=structure)
            comp_ids = [(idx, comp_arr == idx) for idx in range(1, int(num) + 1)]
        for _, comp_np in comp_ids:
            comp = torch.from_numpy(comp_np).bool()
            num_tokens = int(comp.sum().item())
            if num_tokens <= 0:
                continue
            score_vals = anchor_score[comp]
            risk_vals = risk_score[comp]
            trust_vals = sem["semantic_trust"][comp]
            geom_vals = geom["geom_conf"][comp]
            D_vals = geom["D_geo"][comp]
            gram_vals = gram_motion[comp] if gram_motion.shape == comp.shape else torch.zeros_like(score_vals)
            entropy, coverage = _spatial_entropy(comp.float())
            cond = _condition_score(geom.get("patch_points"), comp.float() * anchor_score)
            vertical_ratio = float((sem["vertical_mask"][comp]).float().mean().item())
            road_boundary_ratio = float((sem["road_boundary_mask"][comp]).float().mean().item())
            head_frac = _frame_frac(comp, slice(0, max(1, T // 4)))
            tail_frac = _frame_frac(comp, slice(max(0, T - max(1, T // 4)), T))
            overlap_frac = _frame_frac(comp, slice(0, min(3, T)))
            mean_score = float(score_vals.mean().item())
            mean_trust = float(trust_vals.mean().item())
            mean_D = float(D_vals.mean().item())
            mean_gram = float(gram_vals.mean().item()) if gram_vals.numel() else 0.0
            reject_reasons: List[str] = []
            if mean_score < anchor_threshold:
                reject_reasons.append("anchor_score_below_threshold")
            if num_tokens < 32:
                reject_reasons.append("num_tokens_lt_32")
            if coverage < 0.15:
                reject_reasons.append("grid_coverage_lt_0p15")
            if entropy < 0.30:
                reject_reasons.append("spatial_entropy_lt_0p30")
            if mean_trust < 0.50:
                reject_reasons.append("semantic_trust_lt_0p50")
            if (not gram_unavailable) and mean_gram > 0.60:
                reject_reasons.append("gram_motion_gt_0p60")
            if mean_D > 0.70:
                reject_reasons.append("D_geo_gt_0p70")
            if group in {"sky", "dynamic"}:
                reject_reasons.append(f"{group}_majority")
            weak_reasons = set(reject_reasons)
            if group == "vertical_static" and allow_vertical_static_weak:
                weak_reasons -= {
                    "anchor_score_below_threshold",
                    "num_tokens_lt_32",
                    "grid_coverage_lt_0p15",
                    "spatial_entropy_lt_0p30",
                }
                if mean_score < max(0.10, 0.60 * float(anchor_threshold)):
                    weak_reasons.add("vertical_static_weak_score_lt_floor")
                if num_tokens < 8:
                    weak_reasons.add("vertical_static_weak_num_tokens_lt_8")
                if coverage < 0.05:
                    weak_reasons.add("vertical_static_weak_grid_coverage_lt_0p05")
                if entropy < 0.10:
                    weak_reasons.add("vertical_static_weak_spatial_entropy_lt_0p10")
                if mean_trust < 0.45:
                    weak_reasons.add("vertical_static_weak_trust_lt_0p45")
                if (not gram_unavailable) and mean_gram > 0.75:
                    weak_reasons.add("vertical_static_weak_gram_gt_0p75")
                if mean_D > 0.75:
                    weak_reasons.add("vertical_static_weak_D_geo_gt_0p75")
            valid_base = not reject_reasons
            weak_base = bool(group == "vertical_static" and allow_vertical_static_weak and not weak_reasons)
            scale_support = (cond is not None and cond >= 0.10) or (road_boundary_ratio + vertical_ratio >= 0.10)
            weak_scale_support = scale_support or (cond is not None and cond >= 0.05) or vertical_ratio >= 0.25
            valid_scale = bool((valid_base and scale_support) or (weak_base and weak_scale_support))
            valid_read = bool(valid_base or weak_base)
            valid_ttt = bool(valid_base and mean_D <= 0.65 and mean_trust >= 0.60)
            if (valid_base or weak_base) and not (scale_support or (weak_base and weak_scale_support)):
                scale_reject = "scale_condition_or_boundary_support_low"
            else:
                scale_reject = ""
            if current_id < max_instances:
                anchor_id_map[comp] = int(current_id)
            row = {
                "chunk_id": int(chunk_id),
                "anchor_id": int(current_id),
                "label_majority": label,
                "semantic_group": group,
                "frame_span": [int(frame_start), int(frame_end)],
                "num_tokens": int(num_tokens),
                "mean_semantic_trust": mean_trust,
                "mean_geom_conf": float(geom_vals.mean().item()),
                "mean_D_geo": mean_D,
                "mean_gram_motion": mean_gram,
                "mean_overlap_residual": None,
                "near_ratio": float((geom_vals > torch.quantile(geom["geom_conf"].reshape(-1), 0.50)).float().mean().item()),
                "grid_coverage": coverage,
                "spatial_entropy": entropy,
                "condition_score_3d": cond,
                "head_visible": bool(head_frac > 0.0),
                "tail_visible": bool(tail_frac > 0.0),
                "overlap_visible": bool(overlap_frac > 0.0),
                "anchor_score": mean_score,
                "risk_score": float(risk_vals.mean().item()),
                "anchor_valid_for_scale": valid_scale,
                "anchor_valid_for_read": valid_read,
                "anchor_valid_for_ttt": valid_ttt,
                "vertical_static_weak": bool(weak_base and not valid_base),
                "reject_reason": "pass" if valid_base else ("vertical_static_weak_pass" if weak_base else ";".join(reject_reasons)),
                "scale_reject_reason": scale_reject,
                "gram_motion_unavailable": bool(gram_unavailable),
            }
            instances.append(row)
            current_id += 1

    instances.sort(key=lambda r: float(r["anchor_score"]) * int(r["num_tokens"]), reverse=True)
    kept_ids = {int(row["anchor_id"]): idx for idx, row in enumerate(instances[:max_instances])}
    remap = torch.full_like(anchor_id_map, -1)
    for old_id, new_id in kept_ids.items():
        remap[anchor_id_map == old_id] = int(new_id)
    for idx, row in enumerate(instances[:max_instances]):
        row["anchor_id"] = int(idx)
    return remap, instances[:max_instances]


def _mass(mask: torch.Tensor, weight: torch.Tensor) -> float:
    denom = float(weight.sum().item())
    if denom <= 0.0:
        return 0.0
    return float(weight[mask].sum().item() / denom)


def _summary_row(
    *,
    chunk_id: int,
    frame_start: int,
    frame_end: int,
    sem: Mapping[str, Any],
    geom: Mapping[str, Any],
    gram_motion: torch.Tensor,
    gram_unavailable: bool,
    anchor_score: torch.Tensor,
    risk_score: torch.Tensor,
    instances: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    valid_read = [r for r in instances if bool(r.get("anchor_valid_for_read"))]
    valid_scale = [r for r in instances if bool(r.get("anchor_valid_for_scale"))]
    valid_ttt = [r for r in instances if bool(r.get("anchor_valid_for_ttt"))]
    entropy, coverage = _spatial_entropy(anchor_score.clamp_min(0.0))
    cond = _condition_score(geom.get("patch_points"), anchor_score.clamp_min(0.0))
    score_flat = anchor_score.reshape(-1).float()
    topk = min(max(1, int(round(0.10 * score_flat.numel()))), int(score_flat.numel()))
    top10_mean = float(torch.topk(score_flat, topk).values.mean().item()) if score_flat.numel() else 0.0
    anchor_mass = anchor_score.clamp_min(0.0)
    summary = {
        "chunk_id": int(chunk_id),
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
        "valid_anchor_count": int(len(valid_read)),
        "valid_scale_anchor_count": int(len(valid_scale)),
        "valid_read_anchor_count": int(len(valid_read)),
        "valid_ttt_anchor_count": int(len(valid_ttt)),
        "vertical_anchor_mass": _mass(sem["vertical_mask"], anchor_mass),
        "road_boundary_anchor_mass": _mass(sem["road_boundary_mask"], anchor_mass),
        "ground_anchor_mass": _mass(sem["ground_mask"], anchor_mass),
        "dynamic_risk_mass": _mass(sem["dynamic_mask"], risk_score.clamp_min(0.0)),
        "sky_risk_mass": _mass(sem["sky_mask"], risk_score.clamp_min(0.0)),
        "vegetation_risk_mass": _mass(sem["vegetation_mask"], risk_score.clamp_min(0.0)),
        "semantic_trust_mean": float(sem["semantic_trust"].mean().item()),
        "gram_motion_mean": None if gram_unavailable else float(gram_motion.mean().item()),
        "gram_motion_unavailable": bool(gram_unavailable),
        "anchor_score_mean": float(anchor_score.mean().item()),
        "anchor_score_top10_mean": top10_mean,
        "anchor_grid_coverage": coverage,
        "anchor_spatial_entropy": entropy,
        "condition_score_median": cond,
        "anchor_visible_head_frac": float(np.mean([1.0 if r.get("head_visible") else 0.0 for r in valid_read])) if valid_read else 0.0,
        "anchor_visible_tail_frac": float(np.mean([1.0 if r.get("tail_visible") else 0.0 for r in valid_read])) if valid_read else 0.0,
        "anchor_visible_overlap_frac": float(np.mean([1.0 if r.get("overlap_visible") else 0.0 for r in valid_read])) if valid_read else 0.0,
        "straight_road_anchor_sparse": bool(len(valid_scale) == 0 and _mass(sem["ground_mask"], anchor_mass) > 0.50),
    }
    pass_reasons: List[str] = []
    if summary["valid_read_anchor_count"] < 1:
        pass_reasons.append("no_valid_read_anchor")
    if summary["valid_scale_anchor_count"] < 1:
        pass_reasons.append("no_valid_scale_anchor")
    if summary["anchor_grid_coverage"] < 0.15:
        pass_reasons.append("anchor_grid_coverage_lt_0p15")
    if summary["anchor_spatial_entropy"] < 0.30:
        pass_reasons.append("anchor_spatial_entropy_lt_0p30")
    summary["anchor_bank_quality_pass"] = len(pass_reasons) == 0
    summary["reject_reason"] = "pass" if not pass_reasons else ";".join(pass_reasons)
    return summary


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    if not rows and fieldnames is None:
        return
    fields = list(fieldnames) if fieldnames is not None else sorted({k for row in rows for k in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _jsonable(row.get(k)) for k in fields})


def _stage_chunks(stage_c_cache: Path) -> List[Tuple[int, int, int, Path]]:
    rows: List[Tuple[int, int, int, Path]] = []
    for path in sorted(stage_c_cache.glob("chunk_*/masklet.pt")):
        m = re.search(r"chunk_(\d+)_(\d+)_(\d+)", str(path.parent.name))
        if not m:
            continue
        rows.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), path))
    return sorted(rows)


def _group_distribution(instances: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in instances:
        buckets[(str(row.get("semantic_group")), str(row.get("label_majority")))].append(row)
    out = []
    for (group, label), rows in sorted(buckets.items()):
        out.append(
            {
                "semantic_group": group,
                "label_majority": label,
                "anchor_instances": int(len(rows)),
                "valid_read": int(sum(1 for r in rows if bool(r.get("anchor_valid_for_read")))),
                "valid_scale": int(sum(1 for r in rows if bool(r.get("anchor_valid_for_scale")))),
                "token_count": int(sum(int(r.get("num_tokens") or 0) for r in rows)),
                "mean_anchor_score": float(np.mean([_safe_float(r.get("anchor_score"), 0.0) for r in rows])),
            }
        )
    return out


def _save_figures(
    out_dir: Path,
    chunk_id: int,
    sem: Mapping[str, Any],
    anchor_score: torch.Tensor,
    risk_score: torch.Tensor,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    label_img = sem["label_patch"].float().mean(dim=0).numpy()
    score_img = anchor_score.float().mean(dim=0).numpy()
    risk_img = risk_score.float().mean(dim=0).numpy()
    trust_img = sem["semantic_trust"].float().mean(dim=0).numpy()
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), constrained_layout=True)
    for ax, img, title in zip(axes, (label_img, trust_img, score_img, risk_img), ("label_mean", "semantic_trust", "anchor_score", "risk_score")):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"v69 anchor audit chunk {chunk_id:03d}")
    fig.savefig(out_dir / f"chunk{chunk_id:03d}_anchor_overlay.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    patch_grid = (int(args.patch_grid[0]), int(args.patch_grid[1]))
    selected_layers = _parse_selected_layers(args.v68_selection_json)
    target_chunks = {int(x) for x in str(args.target_chunks).split(",") if str(x).strip()}
    out_dir = args.out_dir
    bank_dir = out_dir / "anchor_bank"
    figure_dir = out_dir / "figures"
    bank_dir.mkdir(parents=True, exist_ok=True)

    stage_rows = _stage_chunks(args.stage_c_cache)
    if not stage_rows:
        raise FileNotFoundError(f"No chunk_*/masklet.pt files under {args.stage_c_cache}")

    summary_rows: List[Dict[str, Any]] = []
    all_instances: List[Dict[str, Any]] = []
    layer_debug_rows: List[Dict[str, Any]] = []
    missing_geometry: List[int] = []
    missing_gram: List[int] = []

    for chunk_id, frame_start, frame_end, stage_path in stage_rows:
        stage_payload = _torch_load(stage_path)
        sem = _patch_semantic(stage_payload, patch_grid)
        geom_path = args.per_chunk_geometry_dir / f"chunk_{chunk_id:03d}.pt"
        if not geom_path.exists():
            missing_geometry.append(int(chunk_id))
            T = int(sem["label_patch"].shape[0])
            geom = {
                "geom_conf": torch.ones((T, patch_grid[0], patch_grid[1]), dtype=torch.float32),
                "D_geo": torch.zeros((T, patch_grid[0], patch_grid[1]), dtype=torch.float32),
                "patch_points": None,
                "geometry_payload": {},
            }
        else:
            geom = _geometry_patch(geom_path, patch_grid)

        feature_path = (args.v68_feature_dir / f"chunk_{chunk_id:03d}.pt") if args.v68_feature_dir else Path("__missing__")
        gram_motion, gram_unavailable, gram_debug = _gram_motion_for_chunk(feature_path, selected_layers, patch_grid)
        sem_T = int(sem["label_patch"].shape[0])
        if int(gram_motion.shape[0]) == 0:
            gram_motion = torch.zeros((sem_T, patch_grid[0], patch_grid[1]), dtype=torch.float32)
        elif int(gram_motion.shape[0]) != sem_T:
            gram_motion = F.interpolate(
                gram_motion[None, None].float(),
                size=(sem_T, patch_grid[0], patch_grid[1]),
                mode="trilinear",
                align_corners=False,
            )[0, 0]
        if gram_unavailable:
            missing_gram.append(int(chunk_id))
        for row in gram_debug:
            layer_debug_rows.append({"chunk_id": int(chunk_id), **row})

        geom_support = (geom["geom_conf"] * (1.0 - geom["D_geo"])).clamp(0.0, 1.0)
        gram_stable = (1.0 - gram_motion).clamp(0.0, 1.0)
        anchor_score = (
            sem["semantic_trust"]
            * sem["static_support"]
            * geom_support
            * gram_stable
            * (1.0 - sem["semantic_risk"])
        ).clamp(0.0, 1.0)
        risk_score = (
            sem["semantic_trust"]
            * sem["semantic_risk"]
            * torch.maximum(geom["D_geo"], gram_motion)
        ).clamp(0.0, 1.0)
        candidate_mask = anchor_score >= float(args.anchor_threshold)
        if bool(args.allow_vertical_static_weak):
            weak_mask = (
                sem["vertical_mask"]
                & (anchor_score >= float(args.vertical_static_weak_threshold))
                & (sem["semantic_trust"] >= 0.45)
                & (geom["D_geo"] <= 0.75)
                & ((gram_motion <= 0.75) | bool(gram_unavailable))
            )
            candidate_mask = candidate_mask | weak_mask
        anchor_id_map, instances = _instance_rows(
            chunk_id=chunk_id,
            frame_start=frame_start,
            frame_end=frame_end,
            candidate_mask=candidate_mask,
            anchor_score=anchor_score,
            risk_score=risk_score,
            sem=sem,
            geom=geom,
            gram_motion=gram_motion,
            gram_unavailable=gram_unavailable,
            anchor_threshold=float(args.anchor_threshold),
            allow_vertical_static_weak=bool(args.allow_vertical_static_weak),
            max_instances=int(args.max_instances_per_chunk),
            component_mode=str(args.component_mode),
            split_road_boundary=bool(args.split_road_boundary),
        )
        summary = _summary_row(
            chunk_id=chunk_id,
            frame_start=frame_start,
            frame_end=frame_end,
            sem=sem,
            geom=geom,
            gram_motion=gram_motion,
            gram_unavailable=gram_unavailable,
            anchor_score=anchor_score,
            risk_score=risk_score,
            instances=instances,
        )
        payload = {
            "schema": "acl2_v69_semantic_anchor_bank_v1",
            "seq": str(args.seq),
            "profile_name": str(args.profile_name),
            "component_mode": str(args.component_mode),
            "split_road_boundary": bool(args.split_road_boundary),
            "allow_vertical_static_weak": bool(args.allow_vertical_static_weak),
            "vertical_static_weak_threshold": float(args.vertical_static_weak_threshold),
            "chunk_id": int(chunk_id),
            "frame_start": int(frame_start),
            "frame_end": int(frame_end),
            "patch_grid": [int(patch_grid[0]), int(patch_grid[1])],
            "label_patch": sem["label_patch"].short(),
            "semantic_confidence_patch": sem["semantic_confidence_patch"].half(),
            "patch_purity": sem["patch_purity"].half(),
            "semantic_trust": sem["semantic_trust"].half(),
            "geom_conf": geom["geom_conf"].half(),
            "D_geo": geom["D_geo"].half(),
            "gram_motion": gram_motion.half(),
            "gram_motion_unavailable": bool(gram_unavailable),
            "overlap_support": None,
            "anchor_score": anchor_score.half(),
            "risk_score": risk_score.half(),
            "anchor_id_map": anchor_id_map.short(),
            "anchor_instances": [dict(r) for r in instances],
            "chunk_anchor_summary": dict(summary),
            "source_paths": {
                "stage_c_masklet": str(stage_path),
                "per_chunk_geometry": str(geom_path) if geom_path.exists() else None,
                "v68_feature": str(feature_path) if feature_path.exists() else None,
                "v68_selection_json": str(args.v68_selection_json) if args.v68_selection_json else None,
                "full_semantic_pt": str(args.full_semantic_pt),
            },
        }
        torch.save(payload, bank_dir / f"chunk_{chunk_id:03d}.pt")
        _write_json(bank_dir / f"chunk_{chunk_id:03d}_summary.json", summary)
        summary_rows.append(summary)
        all_instances.extend(dict(r) for r in instances)
        if not args.no_figures and int(chunk_id) in target_chunks:
            _save_figures(figure_dir, int(chunk_id), sem, anchor_score, risk_score)

    summary_fields = [
        "chunk_id",
        "frame_start",
        "frame_end",
        "valid_anchor_count",
        "valid_scale_anchor_count",
        "valid_read_anchor_count",
        "valid_ttt_anchor_count",
        "vertical_anchor_mass",
        "road_boundary_anchor_mass",
        "ground_anchor_mass",
        "dynamic_risk_mass",
        "sky_risk_mass",
        "vegetation_risk_mass",
        "semantic_trust_mean",
        "gram_motion_mean",
        "gram_motion_unavailable",
        "anchor_score_mean",
        "anchor_score_top10_mean",
        "anchor_grid_coverage",
        "anchor_spatial_entropy",
        "condition_score_median",
        "anchor_visible_head_frac",
        "anchor_visible_tail_frac",
        "anchor_visible_overlap_frac",
        "straight_road_anchor_sparse",
        "anchor_bank_quality_pass",
        "reject_reason",
    ]
    _write_csv(out_dir / "anchor_bank_summary_by_chunk.csv", summary_rows, summary_fields)
    _write_csv(out_dir / "anchor_instances.csv", all_instances)
    _write_csv(out_dir / "anchor_group_distribution.csv", _group_distribution(all_instances))
    _write_csv(out_dir / "selected_layer_motion_debug.csv", layer_debug_rows)

    target_required = {8, 10, 19, 20, 29, 30, 31, 32}
    chunks_with_bank = {int(r["chunk_id"]) for r in summary_rows}
    quality_pass = sum(1 for r in summary_rows if bool(r.get("anchor_bank_quality_pass")))
    valid_read_chunks = sum(1 for r in summary_rows if int(r.get("valid_read_anchor_count") or 0) >= 1)
    valid_scale_chunks = sum(1 for r in summary_rows if int(r.get("valid_scale_anchor_count") or 0) >= 1)
    gate = {
        "bank_files_38_of_38": len(chunks_with_bank) == 38,
        "valid_read_anchor_ge_30_chunks": valid_read_chunks >= 30,
        "valid_scale_anchor_ge_20_chunks": valid_scale_chunks >= 20,
        "target_chunks_have_rows": target_required.issubset(chunks_with_bank),
        "quality_pass_ge_70_percent": quality_pass >= math.ceil(0.70 * max(1, len(summary_rows))),
    }
    audit = {
        "schema": "acl2_v69_anchor_bank_audit_v1",
        "seq": str(args.seq),
        "profile_name": str(args.profile_name),
        "anchor_threshold": float(args.anchor_threshold),
        "component_mode": str(args.component_mode),
        "split_road_boundary": bool(args.split_road_boundary),
        "allow_vertical_static_weak": bool(args.allow_vertical_static_weak),
        "vertical_static_weak_threshold": float(args.vertical_static_weak_threshold),
        "stage_c_cache": str(args.stage_c_cache),
        "full_semantic_pt": str(args.full_semantic_pt),
        "per_chunk_geometry_dir": str(args.per_chunk_geometry_dir),
        "v68_feature_dir": str(args.v68_feature_dir) if args.v68_feature_dir else None,
        "v68_selection_json": str(args.v68_selection_json) if args.v68_selection_json else None,
        "selected_layers": selected_layers,
        "num_chunks": int(len(summary_rows)),
        "bank_files": int(len(list(bank_dir.glob("chunk_*.pt")))),
        "valid_read_anchor_chunks": int(valid_read_chunks),
        "valid_scale_anchor_chunks": int(valid_scale_chunks),
        "anchor_bank_quality_pass_chunks": int(quality_pass),
        "missing_geometry_chunks": missing_geometry,
        "gram_motion_unavailable_chunks": missing_gram,
        "phaseA_gate": gate,
        "phaseA_gate_pass": bool(all(gate.values())),
        "outputs": {
            "anchor_bank_dir": str(bank_dir),
            "anchor_bank_summary_by_chunk": str(out_dir / "anchor_bank_summary_by_chunk.csv"),
            "anchor_instances": str(out_dir / "anchor_instances.csv"),
            "anchor_group_distribution": str(out_dir / "anchor_group_distribution.csv"),
            "selected_layer_motion_debug": str(out_dir / "selected_layer_motion_debug.csv"),
            "figures": str(figure_dir),
        },
    }
    _write_json(out_dir / "anchor_bank_audit.json", audit)
    print(json.dumps(_jsonable(audit), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
