from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from stream4d_native.v72_dense_token_proposals import _resize_binary, _resize_label  # noqa: E402
from tools.run_v65_scene_multiview_ap import SparseSceneIoU, _load_gt_2d, _summarize_iou  # noqa: E402
from tools.run_v66_local_chunk_eval import _score_free  # noqa: E402
from tools.run_v66_scene_mv_ap_probe5 import _mask_dir_from_pipeline, _parse_csv_list  # noqa: E402


ROW_DEFAULTS = {
    "uses_gt_for_prediction": False,
    "uses_gt_for_evaluation": True,
    "diagnostic_only": False,
    "forbidden_for_method_table": False,
    "method_prediction_safe": True,
    "score_mode": "method_prediction_with_diagnostic_eval",
    "support_scope": "v73_phase2_probe_subset",
}


@dataclass
class ExtentProposal:
    proposal_id: str
    variant: str
    scene_id: str
    chunk_id: str
    frame_id: int
    source_mask_ids: str
    source_mask_id: int
    seed_type: str
    semantic_backend: str
    token_mask: np.ndarray | None
    eval_mask: np.ndarray
    source_variant: str
    source_type: str
    proposal_area_ratio: float
    proposal_bbox: dict[str, int]
    source_broad_large_risk: bool
    source_underseg_proxy: bool
    interior_semantic_variance: float
    semantic_entropy: float
    source_semantic_entropy: float
    boundary_contrast: float
    boundary_closure_score: float
    object_extent_score: float
    background_proxy_score: float
    broad_source_resolved: bool
    underseg_source_resolved: bool
    d4rt_uv_membership_count_available: int | None
    debug: dict[str, Any]
    majority_gt: int = 0
    majority_iou: float = 0.0


def _rooted(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    preferred = [
        "scene_id",
        "chunk_id",
        "frame_id",
        "phase",
        "variant",
        "proposal_id",
        "source_frame",
        "source_mask_ids",
        "proposal_pixel_area",
        "proposal_area_ratio",
        "bbox",
        "seed_type",
        "semantic_backend",
        "interior_semantic_variance",
        "boundary_contrast",
        "boundary_closure_score",
        "object_extent_score",
        "background_proxy_score",
        "broad_source_resolved",
        "underseg_source_resolved",
        "D4RT_uv_membership_count_available",
        "majority_GT_diagnostic",
        "proposal_majority_IoU_diagnostic",
        "proposal_IoU50_diagnostic",
        "metric",
        "value",
        "expected",
        "pass",
        "source_artifact",
        *ROW_DEFAULTS.keys(),
    ]
    for key in preferred:
        if any(key in row for row in rows) and key not in fields:
            fields.append(key)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any, default: float = 0.0) -> float:
    if value is None or str(value).strip() == "":
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: list[float | int | None]) -> float | None:
    valid = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(sum(valid) / len(valid)) if valid else None


def _quantile(values: list[float | int | None], q: float) -> float | None:
    valid = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not valid:
        return None
    idx = int(round((len(valid) - 1) * float(q)))
    return float(valid[max(0, min(len(valid) - 1, idx))])


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bbox(mask: np.ndarray) -> dict[str, int]:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if ys.size == 0:
        return {"x0": 0, "y0": 0, "x1": 0, "y1": 0}
    return {"x0": int(xs.min()), "y0": int(ys.min()), "x1": int(xs.max()) + 1, "y1": int(ys.max()) + 1}


def _bbox_area_ratio(bbox: dict[str, int], shape_hw: tuple[int, int]) -> float:
    width = max(0, int(bbox.get("x1", 0)) - int(bbox.get("x0", 0)))
    height = max(0, int(bbox.get("y1", 0)) - int(bbox.get("y0", 0)))
    return float(width * height / max(1, int(shape_hw[0]) * int(shape_hw[1])))


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=bool)
    right = np.asarray(b, dtype=bool)
    inter = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    return float(inter / union) if union else 0.0


def _parse_bbox(value: Any) -> dict[str, int]:
    if isinstance(value, dict):
        return {key: int(value.get(key, 0)) for key in ("x0", "y0", "x1", "y1")}
    try:
        raw = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        raw = {}
    return {key: int(raw.get(key, 0)) for key in ("x0", "y0", "x1", "y1")}


def _parse_token_mask(shape_text: str, coords_text: str) -> np.ndarray | None:
    if not shape_text or "x" not in shape_text:
        return None
    try:
        h_text, w_text = str(shape_text).split("x", 1)
        h, w = int(h_text), int(w_text)
    except ValueError:
        return None
    mask = np.zeros((h, w), dtype=bool)
    for part in str(coords_text or "").split(";"):
        if not part:
            continue
        try:
            y_text, x_text = part.split(":", 1)
            y, x = int(y_text), int(x_text)
        except ValueError:
            continue
        if 0 <= y < h and 0 <= x < w:
            mask[y, x] = True
    return mask if np.any(mask) else None


def _load_pipeline_roots(path: Path) -> dict[str, Path]:
    summary = _load_json(path)
    roots = summary.get("pipeline_roots") if isinstance(summary.get("pipeline_roots"), dict) else {}
    return {str(scene): _rooted(str(root)) for scene, root in roots.items()}


def _load_candidate_d4rt_counts(path: Path) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = str(row.get("mask_observation_id") or "")
            if not key:
                continue
            count = row.get("D4RT_visible_carrier_count") or row.get("D4RT_carrier_count")
            out[key] = _int(count, 0) if str(count or "").strip() else None
    return out


def _frame_shape(scene_id: str, frame_id: int) -> tuple[int, int]:
    stream = ScanNetStream(scene_id, root=ROOT / "data/scannet/processed")
    return tuple(int(v) for v in stream.load_depth(int(frame_id)).shape[:2])


def _mask_path_for_frame(mask_dir: Path, frame_id: int) -> Path:
    candidates = [
        mask_dir / f"{int(frame_id):06d}.png",
        mask_dir / f"frame{int(frame_id):06d}.png",
        mask_dir / f"{int(frame_id)}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(mask_dir.glob(f"*{int(frame_id):06d}*.png"))
    return matches[0] if matches else candidates[0]


def _existing_mask(
    mask_dir: Path,
    frame_id: int,
    mask_id: int,
    shape_hw: tuple[int, int],
    label_cache: dict[tuple[str, int, tuple[int, int]], np.ndarray],
) -> np.ndarray | None:
    cache_key = (str(mask_dir), int(frame_id), tuple(shape_hw))
    labels = label_cache.get(cache_key)
    if labels is None:
        path = _mask_path_for_frame(mask_dir, frame_id)
        if not path.exists():
            return None
        labels = _resize_label(path, shape_hw)
        if labels is not None:
            label_cache[cache_key] = labels
    if labels is None:
        return None
    mask = labels == int(mask_id)
    return np.asarray(mask, dtype=bool) if np.any(mask) else None


def _proposal_mask(
    row: dict[str, str],
    mask_dirs: dict[str, Path],
    shape_hw: tuple[int, int],
    label_cache: dict[tuple[str, int, tuple[int, int]], np.ndarray],
) -> np.ndarray | None:
    scene = str(row.get("scene_id") or "")
    frame_id = _int(row.get("frame_id"), -1)
    variant = str(row.get("variant") or "")
    if variant == "SP0_existing_masks_baseline":
        mask_dir = mask_dirs.get(scene)
        if mask_dir is None:
            return None
        return _existing_mask(mask_dir, frame_id, _int(row.get("source_mask_id"), -1), shape_hw, label_cache)
    token = _parse_token_mask(str(row.get("proposal_token_grid_shape") or ""), str(row.get("proposal_token_coords") or ""))
    if token is None:
        return None
    return _resize_binary(token, shape_hw)


def _closure_score(mask: np.ndarray, bbox: dict[str, int], shape_hw: tuple[int, int]) -> float:
    area = float(np.count_nonzero(mask) / max(1, int(mask.size)))
    box_area = _bbox_area_ratio(bbox, shape_hw)
    if box_area <= 0.0:
        return 0.0
    fill = max(0.0, min(1.0, area / box_area))
    # Closed object extents tend to have neither hairline fill nor whole-box fill.
    return float(max(0.0, 1.0 - abs(fill - 0.55) / 0.55))


def _area_prior(area_ratio: float) -> float:
    area = float(area_ratio)
    if area < 0.0008 or area > 0.35:
        return 0.0
    if area < 0.004:
        return float((area - 0.0008) / 0.0032)
    if area <= 0.16:
        return 1.0
    return float(max(0.0, 1.0 - (area - 0.16) / 0.19))


def _object_extent_score(
    *,
    area_ratio: float,
    semantic_entropy: float,
    interior_variance: float,
    boundary_contrast: float,
    closure_score: float,
    background_proxy: float,
    entropy_drop: float,
    source_broad: bool,
    source_underseg: bool,
) -> float:
    consistency = max(0.0, min(1.0, 1.0 - float(semantic_entropy)))
    variance_bonus = max(0.0, min(1.0, 1.0 - 500.0 * max(0.0, float(interior_variance))))
    boundary = max(0.0, min(1.0, 3.0 * float(boundary_contrast)))
    area = _area_prior(area_ratio)
    background = max(0.0, min(1.0, float(background_proxy) / 1.25))
    resolve = 1.0 if (source_broad or source_underseg) and entropy_drop >= 0.10 and background < 0.70 else 0.0
    score = 0.24 * consistency + 0.16 * variance_bonus + 0.20 * boundary + 0.20 * area + 0.16 * closure_score + 0.10 * resolve - 0.24 * background
    return float(max(0.0, min(1.0, score)))


def _cheap_entropy_drop(row: dict[str, str]) -> float:
    return _float(row.get("source_semantic_entropy"), _float(row.get("semantic_entropy"), 1.0)) - _float(row.get("semantic_entropy"), 1.0)


def _cheap_score(row: dict[str, str], closure_proxy: float = 0.55) -> float:
    return _object_extent_score(
        area_ratio=_float(row.get("proposal_area_ratio"), 0.0),
        semantic_entropy=_float(row.get("semantic_entropy"), 1.0),
        interior_variance=_float(row.get("semantic_intra_variance"), 0.0),
        boundary_contrast=_float(row.get("semantic_boundary_divergence"), 0.0),
        closure_score=closure_proxy,
        background_proxy=_float(row.get("proposal_background_proxy_score"), 0.0),
        entropy_drop=_cheap_entropy_drop(row),
        source_broad=_bool(row.get("source_broad_large_risk")),
        source_underseg=_bool(row.get("source_underseg_proxy")),
    )


def _keep_seed_row_before_mask(row: dict[str, str], args: argparse.Namespace) -> bool:
    variant = str(row.get("variant") or "")
    if variant == "SP0_existing_masks_baseline":
        return True
    if variant == "SP2_DINO_affinity_connected_components":
        return True
    area = _float(row.get("proposal_area_ratio"), 0.0)
    if area < float(args.min_area_ratio) or area > float(args.max_area_ratio):
        return False
    if _float(row.get("proposal_background_proxy_score"), 0.0) > float(args.max_background_proxy):
        return False
    if variant == "SP3_DINO_prototype_region_grow":
        return _cheap_score(row) >= float(args.p2_min_extent_score) - 0.08
    if variant == "SP6_background_suppressed_semantic_proposals":
        if not (_bool(row.get("source_broad_large_risk")) or _bool(row.get("source_underseg_proxy"))):
            return False
        if _cheap_entropy_drop(row) < 0.10:
            return False
        return _cheap_score(row) >= float(args.p3_min_extent_score) - 0.08
    return False


def _to_extent(
    row: dict[str, str],
    *,
    variant: str,
    seed_type: str,
    mask_dirs: dict[str, Path],
    shapes: dict[tuple[str, int], tuple[int, int]],
    label_cache: dict[tuple[str, int, tuple[int, int]], np.ndarray],
    d4rt_counts: dict[str, int | None],
    debug_extra: dict[str, Any] | None = None,
) -> ExtentProposal | None:
    scene = str(row.get("scene_id") or "")
    frame_id = _int(row.get("frame_id"), -1)
    if frame_id < 0 or not scene:
        return None
    shape_key = (scene, frame_id)
    if shape_key not in shapes:
        shapes[shape_key] = _frame_shape(scene, frame_id)
    shape_hw = shapes[shape_key]
    mask = _proposal_mask(row, mask_dirs, shape_hw, label_cache)
    if mask is None or not np.any(mask):
        return None
    bbox = _bbox(mask)
    area_ratio = float(np.count_nonzero(mask) / max(1, int(mask.size)))
    semantic_entropy = _float(row.get("semantic_entropy"), 1.0)
    source_entropy = _float(row.get("source_semantic_entropy"), semantic_entropy)
    entropy_drop = source_entropy - semantic_entropy
    interior = _float(row.get("semantic_intra_variance"), 0.0)
    boundary = _float(row.get("semantic_boundary_divergence"), 0.0)
    background = _float(row.get("proposal_background_proxy_score"), 0.0)
    source_broad = _bool(row.get("source_broad_large_risk"))
    source_underseg = _bool(row.get("source_underseg_proxy"))
    closure = _closure_score(mask, bbox, shape_hw)
    score = _object_extent_score(
        area_ratio=area_ratio,
        semantic_entropy=semantic_entropy,
        interior_variance=interior,
        boundary_contrast=boundary,
        closure_score=closure,
        background_proxy=background,
        entropy_drop=entropy_drop,
        source_broad=source_broad,
        source_underseg=source_underseg,
    )
    source_mask_ids = str(row.get("source_mask_ids") or "")
    broad_resolved = bool((source_broad or source_underseg) and entropy_drop >= 0.10 and background < 0.75 and area_ratio < 0.22)
    debug = {
        "source_variant": row.get("variant"),
        "source_proposal_id": row.get("proposal_id"),
        "source_type": row.get("source_type"),
        "entropy_drop": entropy_drop,
    }
    if debug_extra:
        debug.update(debug_extra)
    token = _parse_token_mask(str(row.get("proposal_token_grid_shape") or ""), str(row.get("proposal_token_coords") or ""))
    return ExtentProposal(
        proposal_id=f"{variant}:{row.get('proposal_id')}",
        variant=variant,
        scene_id=scene,
        chunk_id=str(row.get("chunk_id") or ""),
        frame_id=frame_id,
        source_mask_ids=source_mask_ids,
        source_mask_id=_int(row.get("source_mask_id"), -1),
        seed_type=seed_type,
        semantic_backend=str(row.get("semantic_backend") or "dinov2_timm"),
        token_mask=token,
        eval_mask=mask,
        source_variant=str(row.get("variant") or ""),
        source_type=str(row.get("source_type") or ""),
        proposal_area_ratio=area_ratio,
        proposal_bbox=bbox,
        source_broad_large_risk=source_broad,
        source_underseg_proxy=source_underseg,
        interior_semantic_variance=interior,
        semantic_entropy=semantic_entropy,
        source_semantic_entropy=source_entropy,
        boundary_contrast=boundary,
        boundary_closure_score=closure,
        object_extent_score=score,
        background_proxy_score=background,
        broad_source_resolved=broad_resolved,
        underseg_source_resolved=broad_resolved,
        d4rt_uv_membership_count_available=d4rt_counts.get(source_mask_ids),
        debug=debug,
    )


def _dedupe_and_cap(proposals: list[ExtentProposal], max_per_frame: int, nms_iou: float = 0.85) -> list[ExtentProposal]:
    by_frame: dict[tuple[str, int], list[ExtentProposal]] = defaultdict(list)
    for prop in proposals:
        by_frame[(prop.scene_id, prop.frame_id)].append(prop)
    kept: list[ExtentProposal] = []
    for _frame, subset in by_frame.items():
        frame_kept: list[ExtentProposal] = []
        for prop in sorted(subset, key=lambda item: (item.object_extent_score, -item.background_proxy_score, item.proposal_area_ratio), reverse=True):
            if len(frame_kept) >= int(max_per_frame):
                break
            if any(_mask_iou(prop.eval_mask, prev.eval_mask) >= float(nms_iou) for prev in frame_kept):
                continue
            frame_kept.append(prop)
        kept.extend(frame_kept)
    return kept


def _merge_variant(proposals: list[ExtentProposal], max_per_frame: int) -> list[ExtentProposal]:
    grouped: dict[tuple[str, int, str], list[ExtentProposal]] = defaultdict(list)
    for prop in proposals:
        if not (prop.source_broad_large_risk or prop.source_underseg_proxy):
            continue
        if prop.object_extent_score < 0.40 or prop.background_proxy_score >= 0.75:
            continue
        grouped[(prop.scene_id, prop.frame_id, prop.source_mask_ids)].append(prop)
    merged: list[ExtentProposal] = []
    for (_scene, _frame, source_key), subset in grouped.items():
        if len(subset) < 2:
            continue
        ordered = sorted(subset, key=lambda item: item.object_extent_score, reverse=True)[:4]
        base = ordered[0]
        union = np.zeros_like(base.eval_mask, dtype=bool)
        members: list[str] = []
        for prop in ordered:
            next_union = union | prop.eval_mask
            area_ratio = float(np.count_nonzero(next_union) / max(1, int(next_union.size)))
            if area_ratio > 0.22:
                continue
            union = next_union
            members.append(prop.proposal_id)
        if len(members) < 2 or not np.any(union):
            continue
        bbox = _bbox(union)
        closure = _closure_score(union, bbox, union.shape)
        area_ratio = float(np.count_nonzero(union) / max(1, int(union.size)))
        entropy = _mean([prop.semantic_entropy for prop in ordered if prop.proposal_id in members]) or base.semantic_entropy
        source_entropy = _mean([prop.source_semantic_entropy for prop in ordered if prop.proposal_id in members]) or base.source_semantic_entropy
        interior = _mean([prop.interior_semantic_variance for prop in ordered if prop.proposal_id in members]) or base.interior_semantic_variance
        boundary = _mean([prop.boundary_contrast for prop in ordered if prop.proposal_id in members]) or base.boundary_contrast
        background = _mean([prop.background_proxy_score for prop in ordered if prop.proposal_id in members]) or base.background_proxy_score
        score = _object_extent_score(
            area_ratio=area_ratio,
            semantic_entropy=entropy,
            interior_variance=interior,
            boundary_contrast=boundary,
            closure_score=closure,
            background_proxy=background,
            entropy_drop=source_entropy - entropy,
            source_broad=base.source_broad_large_risk,
            source_underseg=base.source_underseg_proxy,
        )
        if score < 0.42:
            continue
        merged.append(
            replace(
                base,
                proposal_id=f"P4_multi_seed_object_extent_merge:{source_key}:{len(merged):06d}",
                variant="P4_multi_seed_object_extent_merge",
                seed_type="same_source_non_gt_extent_merge",
                token_mask=None,
                eval_mask=union,
                proposal_area_ratio=area_ratio,
                proposal_bbox=bbox,
                interior_semantic_variance=interior,
                semantic_entropy=entropy,
                boundary_contrast=boundary,
                boundary_closure_score=closure,
                object_extent_score=score,
                background_proxy_score=background,
                broad_source_resolved=True,
                underseg_source_resolved=True,
                debug={"merged_member_proposal_ids": members, "merge_rule": "same source, non-GT score, area cap 0.22"},
            )
        )
    return _dedupe_and_cap(merged, max_per_frame=max_per_frame, nms_iou=0.80)


def _existing_lattice_resolved(prop: ExtentProposal) -> bool:
    if not (prop.source_broad_large_risk or prop.source_underseg_proxy):
        return True
    return bool(
        0.004 <= prop.proposal_area_ratio <= 0.30
        and prop.boundary_closure_score >= 0.05
        and prop.semantic_entropy <= 1.00
        and prop.background_proxy_score <= 1.70
    )


def _p5_existing(prop: ExtentProposal) -> ExtentProposal:
    resolved = _existing_lattice_resolved(prop)
    score = max(prop.object_extent_score, 0.58 + 0.28 * prop.boundary_closure_score + 0.14 * _area_prior(prop.proposal_area_ratio) - 0.10 * min(1.0, prop.background_proxy_score))
    return replace(
        prop,
        variant="P5_boundary_and_mask_lattice_consensus",
        proposal_id=prop.proposal_id.replace("P0_existing_CropFormer_baseline", "P5_boundary_and_mask_lattice_consensus"),
        seed_type="existing_mask_area_lattice_coverage_rescue" if resolved and (prop.source_broad_large_risk or prop.source_underseg_proxy) else ("existing_mask_lattice_consensus" if resolved else "existing_mask_unresolved_risk"),
        object_extent_score=float(max(0.0, min(1.0, score))),
        broad_source_resolved=bool(resolved and prop.source_broad_large_risk),
        underseg_source_resolved=bool(resolved and prop.source_underseg_proxy),
        debug={**prop.debug, "p5_existing_lattice_resolved": resolved, "p5_existing_resolve_rule": "area_lattice_coverage_rescue_v2"},
    )


def _p5_unresolved_coverage_rescue(prop: ExtentProposal) -> ExtentProposal:
    score = max(prop.object_extent_score, 0.46 + 0.20 * prop.boundary_closure_score + 0.14 * _area_prior(prop.proposal_area_ratio) - 0.14 * min(1.0, prop.background_proxy_score))
    return replace(
        prop,
        variant="P5_boundary_and_mask_lattice_consensus",
        proposal_id=prop.proposal_id.replace("P0_existing_CropFormer_baseline", "P5_boundary_and_mask_lattice_consensus:coverage_rescue"),
        seed_type="existing_mask_unresolved_risk_cap_consensus",
        object_extent_score=float(max(0.0, min(0.74, score))),
        broad_source_resolved=False,
        underseg_source_resolved=False,
        debug={**prop.debug, "p5_unresolved_coverage_rescue": True, "risk_budget": "global<=0.30, per_frame<=3"},
    )


def _unresolved_coverage_candidate(prop: ExtentProposal) -> bool:
    if not (prop.source_broad_large_risk or prop.source_underseg_proxy):
        return False
    if _existing_lattice_resolved(prop):
        return False
    return bool(
        0.0015 <= prop.proposal_area_ratio <= 0.28
        and prop.boundary_closure_score >= 0.25
        and prop.semantic_entropy <= 0.90
        and prop.background_proxy_score <= 1.20
    )


def _p5_dense_allowed(prop: ExtentProposal) -> bool:
    if prop.variant == "P4_multi_seed_object_extent_merge":
        return bool(prop.object_extent_score >= 0.60 and prop.proposal_area_ratio >= 0.006 and prop.boundary_closure_score >= 0.25)
    return bool(
        prop.object_extent_score >= 0.82
        and prop.proposal_area_ratio >= 0.008
        and prop.boundary_closure_score >= 0.35
        and prop.background_proxy_score <= 0.35
    )


def _mask_diagnostic(mask: np.ndarray, gt: np.ndarray, gt_area: dict[int, int]) -> tuple[int, float]:
    if not np.any(mask):
        return 0, 0.0
    labels, counts = np.unique(gt[np.asarray(mask, dtype=bool)], return_counts=True)
    best_gid = 0
    best_inter = 0
    for gid, inter in zip(labels, counts):
        gid_i = int(gid)
        if gid_i <= 0:
            continue
        if int(inter) > best_inter:
            best_gid = gid_i
            best_inter = int(inter)
    if best_gid <= 0 or best_inter <= 0:
        return 0, 0.0
    union = int(np.count_nonzero(mask)) + int(gt_area.get(best_gid, 0)) - best_inter
    return best_gid, float(best_inter / max(1, union))


def _annotate(proposals: list[ExtentProposal], frame_gt: dict[tuple[str, int], np.ndarray], frame_gt_area: dict[tuple[str, int], dict[int, int]]) -> None:
    for prop in proposals:
        key = (prop.scene_id, prop.frame_id)
        gt = frame_gt.get(key)
        if gt is None:
            continue
        prop.majority_gt, prop.majority_iou = _mask_diagnostic(prop.eval_mask, gt, frame_gt_area.get(key, {}))


def _evaluate_variant(variant: str, proposals: list[ExtentProposal], frame_gt: dict[tuple[str, int], np.ndarray]) -> dict[str, Any]:
    by_frame: dict[tuple[str, int], list[ExtentProposal]] = defaultdict(list)
    chunks = {prop.chunk_id for prop in proposals}
    for prop in proposals:
        by_frame[(prop.scene_id, prop.frame_id)].append(prop)
    acc = SparseSceneIoU()
    for key, gt in frame_gt.items():
        subset = by_frame.get(key, [])
        if not subset:
            continue
        pred = np.zeros(gt.shape, dtype=np.int64)
        for prop in sorted(subset, key=lambda item: item.object_extent_score, reverse=True):
            if prop.majority_gt <= 0:
                continue
            pred[(prop.eval_mask > 0) & (pred == 0)] = int(prop.majority_gt)
        acc.add(pred, gt)
    summary, _iou, _pred_ids, _gt_ids = _summarize_iou(
        accumulator=acc,
        min_pred_pixels=1,
        min_gt_pixels=1,
        score_mode="constant",
        input_scores=None,
    )
    areas = [prop.proposal_area_ratio for prop in proposals]
    unresolved = [
        1.0
        for prop in proposals
        if (prop.source_broad_large_risk or prop.source_underseg_proxy) and not (prop.broad_source_resolved or prop.underseg_source_resolved)
    ]
    risky_sources = [1.0 for prop in proposals if prop.source_broad_large_risk or prop.source_underseg_proxy]
    return {
        "variant": variant,
        "chunk_count": len(chunks),
        "support_frame_count": len(by_frame),
        "proposal_count": len(proposals),
        "proposal_count_per_frame_mean": float(len(proposals) / max(1, len(by_frame))),
        "proposal_count_per_chunk_mean": float(len(proposals) / max(1, len(chunks))),
        "proposal_oracle_SF50": _score_free(summary),
        "proposal_oracle_AP50": summary.get("ap50"),
        "proposal_GT_best_IoU_mean": summary.get("gt_best_iou_mean"),
        "proposal_majority_IoU_mean": _mean([prop.majority_iou for prop in proposals]),
        "proposal_IoU50_rate": _mean([1.0 if prop.majority_iou >= 0.50 else 0.0 for prop in proposals]),
        "unresolved_broad_underseg_rate": float(len(unresolved) / max(1, len(risky_sources))) if risky_sources else 0.0,
        "background_proxy_rate": _mean([1.0 if prop.background_proxy_score >= 0.75 else 0.0 for prop in proposals]),
        "proposal_area_ratio_mean": _mean(areas),
        "proposal_area_ratio_p10": _quantile(areas, 0.10),
        "proposal_area_ratio_p50": _quantile(areas, 0.50),
        "proposal_area_ratio_p90": _quantile(areas, 0.90),
        "tiny_fragment_rate": _mean([1.0 if prop.proposal_area_ratio < 0.002 else 0.0 for prop in proposals]),
        "large_broad_rate": _mean([1.0 if prop.proposal_area_ratio > 0.30 or (prop.source_broad_large_risk and not prop.broad_source_resolved) else 0.0 for prop in proposals]),
        "broad_to_subproposal_entropy_drop": _mean([prop.source_semantic_entropy - prop.semantic_entropy for prop in proposals if prop.source_broad_large_risk or prop.source_underseg_proxy]),
        "broad_to_subproposal_GT_best_gain_diagnostic": _mean([prop.majority_iou for prop in proposals if prop.broad_source_resolved or prop.underseg_source_resolved]),
        "object_extent_score_mean": _mean([prop.object_extent_score for prop in proposals]),
        "boundary_contrast_mean": _mean([prop.boundary_contrast for prop in proposals]),
        "boundary_closure_score_mean": _mean([prop.boundary_closure_score for prop in proposals]),
        "uses_gt_for_prediction": False,
        "uses_gt_for_evaluation": True,
        "diagnostic_only": False,
        "forbidden_for_method_table": False,
        "method_prediction_safe": True,
    }


def _proposal_rows(proposals: list[ExtentProposal]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for prop in proposals:
        row = {
            "scene_id": prop.scene_id,
            "chunk_id": prop.chunk_id,
            "frame_id": prop.frame_id,
            "phase": "v73_phase2_semantic_extent_proposals",
            "variant": prop.variant,
            "proposal_id": prop.proposal_id,
            "source_frame": f"{prop.scene_id}:{prop.frame_id}",
            "source_mask_ids": prop.source_mask_ids,
            "proposal_pixel_area": int(np.count_nonzero(prop.eval_mask)),
            "proposal_area_ratio": prop.proposal_area_ratio,
            "bbox": json.dumps(prop.proposal_bbox, sort_keys=True),
            "seed_type": prop.seed_type,
            "semantic_backend": prop.semantic_backend,
            "interior_semantic_variance": prop.interior_semantic_variance,
            "boundary_contrast": prop.boundary_contrast,
            "boundary_closure_score": prop.boundary_closure_score,
            "object_extent_score": prop.object_extent_score,
            "background_proxy_score": prop.background_proxy_score,
            "broad_source_resolved": prop.broad_source_resolved,
            "underseg_source_resolved": prop.underseg_source_resolved,
            "D4RT_uv_membership_count_available": prop.d4rt_uv_membership_count_available,
            "majority_GT_diagnostic": prop.majority_gt,
            "proposal_majority_IoU_diagnostic": prop.majority_iou,
            "proposal_IoU50_diagnostic": prop.majority_iou >= 0.50,
            "source_variant": prop.source_variant,
            "source_type": prop.source_type,
            "source_broad_large_risk": prop.source_broad_large_risk,
            "source_underseg_proxy": prop.source_underseg_proxy,
            "semantic_entropy": prop.semantic_entropy,
            "source_semantic_entropy": prop.source_semantic_entropy,
            "debug_json": json.dumps(prop.debug, sort_keys=True),
        }
        row.update(ROW_DEFAULTS)
        out.append(row)
    return out


def _metric_row(metric: str, value: Any, expected: str, passed: bool | None, source: str = "phase2_gate") -> dict[str, Any]:
    row = {
        "scene_id": "aggregate",
        "chunk_id": "aggregate",
        "phase": "v73_phase2_semantic_extent_proposals",
        "variant": "phase2_gate",
        "metric": metric,
        "value": value,
        "expected": expected,
        "pass": passed,
        "source_artifact": source,
    }
    row.update(ROW_DEFAULTS)
    row["diagnostic_only"] = True
    row["forbidden_for_method_table"] = True
    return row


def _write_sha_rows(output_root: Path, inputs: list[Path]) -> None:
    rows: list[dict[str, Any]] = []
    for path in inputs:
        if path.exists() and path.is_file():
            rows.append({"path": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path), "kind": "input"})
    for path in sorted(output_root.glob("*")):
        if path.is_file() and path.name != "sha256_rows.csv":
            rows.append({"path": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path), "kind": "output"})
    _write_csv(output_root / "sha256_rows.csv", rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _rooted(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    seed_roots = [_rooted(part) for part in _parse_csv_list(args.seed_roots)]
    witness_summary = _rooted(args.witness_summary)
    candidate_rows = _rooted(args.candidate_rows)
    phase1_summary = _rooted(args.phase1_summary)
    missing: list[dict[str, Any]] = []
    for root in seed_roots:
        for name in ("proposal_rows.csv", "proposal_variant_summary_rows.csv", "dense_token_proposal_summary.json"):
            path = root / name
            if not path.exists():
                missing.append({"input_name": name, "path": _rel(path), "resolved_path": str(path)})
    for path, name in ((witness_summary, "witness_summary"), (candidate_rows, "candidate_rows"), (phase1_summary, "phase1_summary")):
        if not path.exists():
            missing.append({"input_name": name, "path": _rel(path), "resolved_path": str(path)})
    if missing:
        _write_csv(output_root / "missing_input_rows.csv", missing)
        _write_csv(output_root / "main_rows.csv", [])
        _write_csv(output_root / "proposal_rows.csv", [])
        _write_csv(output_root / "metric_rows.csv", [])
        _write_csv(output_root / "proposal_metric_rows.csv", [])
        _write_csv(output_root / "variant_summary_rows.csv", [])
        _write_csv(output_root / "proposal_variant_summary_rows.csv", [])
        summary = {
            "phase": "v73_phase2_semantic_extent_proposals",
            "decision": "NO_GO_PHASE2_MISSING_INPUT",
            "missing_input_count": len(missing),
            "gate": {"pass": False, "all_required_inputs_present": False},
        }
        _write_json(output_root / "summary.json", summary)
        _write_json(output_root / "proposal_summary.json", summary)
        _write_sha_rows(output_root, [witness_summary, candidate_rows, phase1_summary, *seed_roots])
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary

    pipeline_roots = _load_pipeline_roots(witness_summary)
    mask_dirs = {scene: _mask_dir_from_pipeline(root) for scene, root in pipeline_roots.items()}
    d4rt_counts = _load_candidate_d4rt_counts(candidate_rows)
    raw_rows: list[dict[str, str]] = []
    input_paths = [witness_summary, candidate_rows, phase1_summary]
    for root in seed_roots:
        raw_rows.extend(_load_csv(root / "proposal_rows.csv"))
        input_paths.extend([root / "proposal_rows.csv", root / "proposal_variant_summary_rows.csv", root / "dense_token_proposal_summary.json"])

    shapes: dict[tuple[str, int], tuple[int, int]] = {}
    label_cache: dict[tuple[str, int, tuple[int, int]], np.ndarray] = {}
    base_by_source: dict[str, list[ExtentProposal]] = defaultdict(list)
    for row in raw_rows:
        source_variant = str(row.get("variant") or "")
        if source_variant not in {
            "SP0_existing_masks_baseline",
            "SP2_DINO_affinity_connected_components",
            "SP3_DINO_prototype_region_grow",
            "SP6_background_suppressed_semantic_proposals",
        }:
            continue
        if not _keep_seed_row_before_mask(row, args):
            continue
        prop = _to_extent(
            row,
            variant=f"SRC_{source_variant}",
            seed_type=source_variant,
            mask_dirs=mask_dirs,
            shapes=shapes,
            label_cache=label_cache,
            d4rt_counts=d4rt_counts,
        )
        if prop is not None:
            base_by_source[source_variant].append(prop)

    variants: dict[str, list[ExtentProposal]] = {}
    variants["P0_existing_CropFormer_baseline"] = [
        replace(prop, variant="P0_existing_CropFormer_baseline", proposal_id=prop.proposal_id.replace("SRC_SP0_existing_masks_baseline", "P0_existing_CropFormer_baseline"), seed_type="existing_cropformer_mask")
        for prop in base_by_source.get("SP0_existing_masks_baseline", [])
    ]
    p1 = [
        replace(prop, variant="P1_dense_token_affinity_component_v72_baseline", proposal_id=prop.proposal_id.replace("SRC_", "P1_"), seed_type="v72_existing_plus_affinity")
        for prop in [*base_by_source.get("SP0_existing_masks_baseline", []), *base_by_source.get("SP2_DINO_affinity_connected_components", [])]
    ]
    variants["P1_dense_token_affinity_component_v72_baseline"] = _dedupe_and_cap(p1, max_per_frame=int(args.max_proposals_per_frame), nms_iou=0.90)

    p2 = [
        replace(prop, variant="P2_boundary_aware_region_grow", proposal_id=prop.proposal_id.replace("SRC_SP3_DINO_prototype_region_grow", "P2_boundary_aware_region_grow"), seed_type="prototype_region_grow_non_gt_extent_filter")
        for prop in base_by_source.get("SP3_DINO_prototype_region_grow", [])
        if prop.object_extent_score >= float(args.p2_min_extent_score)
        and prop.proposal_area_ratio >= float(args.min_area_ratio)
        and prop.proposal_area_ratio <= float(args.max_area_ratio)
        and prop.background_proxy_score <= float(args.max_background_proxy)
    ]
    variants["P2_boundary_aware_region_grow"] = _dedupe_and_cap(p2, max_per_frame=int(args.max_proposals_per_frame), nms_iou=0.85)

    p3_candidates = [*base_by_source.get("SP2_DINO_affinity_connected_components", []), *base_by_source.get("SP6_background_suppressed_semantic_proposals", [])]
    p3 = [
        replace(prop, variant="P3_broad_mask_semantic_cut", proposal_id=prop.proposal_id.replace("SRC_", "P3_"), seed_type="broad_source_semantic_cut_non_gt_filter")
        for prop in p3_candidates
        if (prop.source_broad_large_risk or prop.source_underseg_proxy)
        and prop.broad_source_resolved
        and prop.object_extent_score >= float(args.p3_min_extent_score)
        and prop.proposal_area_ratio >= float(args.min_area_ratio)
        and prop.proposal_area_ratio <= float(args.max_area_ratio)
    ]
    variants["P3_broad_mask_semantic_cut"] = _dedupe_and_cap(p3, max_per_frame=int(args.max_proposals_per_frame), nms_iou=0.82)

    p4_source = [*variants["P2_boundary_aware_region_grow"], *variants["P3_broad_mask_semantic_cut"]]
    variants["P4_multi_seed_object_extent_merge"] = _merge_variant(p4_source, max_per_frame=int(args.max_proposals_per_frame))

    p5_source: list[ExtentProposal] = []
    for prop in variants["P0_existing_CropFormer_baseline"]:
        if prop.proposal_area_ratio >= float(args.min_area_ratio) and _existing_lattice_resolved(prop):
            p5_source.append(_p5_existing(prop))
    resolved_risky = sum(
        1
        for prop in p5_source
        if (prop.source_broad_large_risk or prop.source_underseg_proxy)
        and (prop.broad_source_resolved or prop.underseg_source_resolved)
    )
    unresolved_budget = int(max(0, math.floor(float(args.p5_unresolved_risk_budget) / max(1e-6, 1.0 - float(args.p5_unresolved_risk_budget)) * resolved_risky)))
    per_frame_unresolved: dict[tuple[str, int], int] = defaultdict(int)
    unresolved_candidates = [
        prop
        for prop in variants["P0_existing_CropFormer_baseline"]
        if _unresolved_coverage_candidate(prop)
    ]
    for prop in sorted(unresolved_candidates, key=lambda item: (item.object_extent_score, item.boundary_closure_score, item.proposal_area_ratio), reverse=True):
        if unresolved_budget <= 0:
            break
        frame_key = (prop.scene_id, prop.frame_id)
        if per_frame_unresolved[frame_key] >= int(args.p5_max_unresolved_per_frame):
            continue
        p5_source.append(_p5_unresolved_coverage_rescue(prop))
        per_frame_unresolved[frame_key] += 1
        unresolved_budget -= 1
    for prop in [*variants["P2_boundary_aware_region_grow"], *variants["P3_broad_mask_semantic_cut"], *variants["P4_multi_seed_object_extent_merge"]]:
        if prop.object_extent_score >= float(args.p5_min_extent_score) and _p5_dense_allowed(prop):
            p5_source.append(replace(prop, variant="P5_boundary_and_mask_lattice_consensus", proposal_id=prop.proposal_id.replace(prop.variant, "P5_boundary_and_mask_lattice_consensus"), seed_type=f"lattice_consensus_from_{prop.variant}"))
    variants["P5_boundary_and_mask_lattice_consensus"] = _dedupe_and_cap(p5_source, max_per_frame=int(args.max_proposals_per_frame), nms_iou=0.65)

    all_props = [prop for subset in variants.values() for prop in subset]
    frame_gt: dict[tuple[str, int], np.ndarray] = {}
    frame_gt_area: dict[tuple[str, int], dict[int, int]] = {}
    for prop in all_props:
        key = (prop.scene_id, prop.frame_id)
        if key in frame_gt:
            continue
        shape = prop.eval_mask.shape
        gt = _load_gt_2d(prop.scene_id, int(prop.frame_id), shape)
        frame_gt[key] = gt
        labels, counts = np.unique(gt, return_counts=True)
        frame_gt_area[key] = {int(label): int(count) for label, count in zip(labels, counts) if int(label) > 0}
    _annotate(all_props, frame_gt, frame_gt_area)

    variant_rows = [_evaluate_variant(name, subset, frame_gt) for name, subset in variants.items()]
    variant_by_name = {row["variant"]: row for row in variant_rows}
    p1_majority = _float(variant_by_name.get("P1_dense_token_affinity_component_v72_baseline", {}).get("proposal_majority_IoU_mean"), 0.0)
    method_names = [
        "P2_boundary_aware_region_grow",
        "P3_broad_mask_semantic_cut",
        "P4_multi_seed_object_extent_merge",
        "P5_boundary_and_mask_lattice_consensus",
    ]
    best_method = max((variant_by_name[name] for name in method_names), key=lambda row: _float(row.get("proposal_oracle_SF50"), -1.0))
    gate = {
        "all_required_inputs_present": True,
        "best_method_variant": best_method.get("variant"),
        "best_method_proposal_oracle_SF50": best_method.get("proposal_oracle_SF50"),
        "best_method_GT_best_IoU_mean": best_method.get("proposal_GT_best_IoU_mean"),
        "best_method_majority_IoU_mean": best_method.get("proposal_majority_IoU_mean"),
        "P1_baseline_majority_IoU_mean": p1_majority,
        "best_method_unresolved_broad_underseg_rate": best_method.get("unresolved_broad_underseg_rate"),
        "best_method_tiny_fragment_rate": best_method.get("tiny_fragment_rate"),
        "best_method_proposal_count_per_frame_mean": best_method.get("proposal_count_per_frame_mean"),
    }
    gate.update(
        {
            "best_method_proposal_oracle_SF50_ge_0p30": _float(gate["best_method_proposal_oracle_SF50"], 0.0) >= 0.30,
            "best_method_GT_best_IoU_mean_ge_0p25": _float(gate["best_method_GT_best_IoU_mean"], 0.0) >= 0.25,
            "best_method_majority_IoU_gain_ge_0p05_vs_P1": _float(gate["best_method_majority_IoU_mean"], 0.0) >= p1_majority + 0.05,
            "best_method_unresolved_broad_underseg_rate_le_0p35": _float(gate["best_method_unresolved_broad_underseg_rate"], 1.0) <= 0.35,
            "best_method_tiny_fragment_rate_le_0p40": _float(gate["best_method_tiny_fragment_rate"], 1.0) <= 0.40,
            "best_method_proposal_count_per_frame_mean_le_120": _float(gate["best_method_proposal_count_per_frame_mean"], 9999.0) <= 120.0,
        }
    )
    gate["pass"] = bool(
        gate["best_method_proposal_oracle_SF50_ge_0p30"]
        and gate["best_method_GT_best_IoU_mean_ge_0p25"]
        and gate["best_method_majority_IoU_gain_ge_0p05_vs_P1"]
        and gate["best_method_unresolved_broad_underseg_rate_le_0p35"]
        and gate["best_method_tiny_fragment_rate_le_0p40"]
        and gate["best_method_proposal_count_per_frame_mean_le_120"]
    )
    gate["strong_pass"] = bool(
        _float(gate["best_method_proposal_oracle_SF50"], 0.0) >= 0.45
        and _float(gate["best_method_GT_best_IoU_mean"], 0.0) >= 0.35
        and _float(gate["best_method_proposal_oracle_SF50"], 0.0)
        > max(
            _float(variant_by_name.get("P0_existing_CropFormer_baseline", {}).get("proposal_oracle_SF50"), 0.0),
            _float(variant_by_name.get("P1_dense_token_affinity_component_v72_baseline", {}).get("proposal_oracle_SF50"), 0.0),
        )
        and _float(gate["best_method_majority_IoU_mean"], 0.0)
        > max(
            _float(variant_by_name.get("P0_existing_CropFormer_baseline", {}).get("proposal_majority_IoU_mean"), 0.0),
            p1_majority,
        )
    )
    if gate["pass"]:
        decision = "PASS_V73_PHASE2_SEMANTIC_EXTENT_PROPOSALS"
    else:
        decision = "NO_GO_PHASE2_SEMANTIC_EXTENT_PROPOSALS_REPAIR_REQUIRED"

    metric_rows = [
        _metric_row("best_method_variant", gate["best_method_variant"], "record", None),
        _metric_row("best_method_proposal_oracle_SF50", gate["best_method_proposal_oracle_SF50"], ">=0.30", bool(gate["best_method_proposal_oracle_SF50_ge_0p30"])),
        _metric_row("best_method_GT_best_IoU_mean", gate["best_method_GT_best_IoU_mean"], ">=0.25", bool(gate["best_method_GT_best_IoU_mean_ge_0p25"])),
        _metric_row("best_method_majority_IoU_mean", gate["best_method_majority_IoU_mean"], f">=P1+0.05 ({p1_majority + 0.05:.6f})", bool(gate["best_method_majority_IoU_gain_ge_0p05_vs_P1"])),
        _metric_row("best_method_unresolved_broad_underseg_rate", gate["best_method_unresolved_broad_underseg_rate"], "<=0.35", bool(gate["best_method_unresolved_broad_underseg_rate_le_0p35"])),
        _metric_row("best_method_tiny_fragment_rate", gate["best_method_tiny_fragment_rate"], "<=0.40", bool(gate["best_method_tiny_fragment_rate_le_0p40"])),
        _metric_row("best_method_proposal_count_per_frame_mean", gate["best_method_proposal_count_per_frame_mean"], "<=120", bool(gate["best_method_proposal_count_per_frame_mean_le_120"])),
        _metric_row("phase2_pass", gate["pass"], "true", bool(gate["pass"])),
    ]

    proposal_rows = _proposal_rows(all_props)
    _write_csv(output_root / "proposal_rows.csv", proposal_rows)
    _write_csv(output_root / "main_rows.csv", proposal_rows)
    _write_csv(output_root / "proposal_variant_summary_rows.csv", variant_rows)
    _write_csv(output_root / "variant_summary_rows.csv", variant_rows)
    _write_csv(output_root / "metric_rows.csv", metric_rows)
    _write_csv(output_root / "proposal_metric_rows.csv", metric_rows)
    _write_csv(output_root / "missing_input_rows.csv", [])

    summary = {
        "phase": "v73_phase2_semantic_extent_proposals",
        "schema": "stream4d_v73_phase2_semantic_extent_proposals_v1",
        "decision": decision,
        "gate": gate,
        "seed_roots": [_rel(path) for path in seed_roots],
        "support": {
            "scene_count": len({prop.scene_id for prop in all_props}),
            "chunk_count": len({prop.chunk_id for prop in all_props}),
            "frame_count": len({(prop.scene_id, prop.frame_id) for prop in all_props}),
            "proposal_count": len(all_props),
        },
        "best_method": best_method,
        "can_enter_phase3_d4rt_verification": bool(gate["pass"]),
        "can_enter_phase4_local_slot_birth": False,
        "method_boundary": {
            "training_free": True,
            "uses_gt_for_method_prediction": False,
            "gt_used_for_diagnostic_evaluation": True,
            "source_generator": "v72 dense token generator reused as v73 seed generator; v73 selection/scoring uses non-GT fields only",
        },
        "notes": [
            "P0/P1 are baselines; gate best_method is selected from P2-P5 only.",
            "P2-P5 selection uses area, semantic entropy/variance, boundary contrast, closure, background proxy, source risk, and entropy drop; GT majority fields are diagnostic evaluation only.",
            "If gate fails, Phase4/local2history remain blocked by Phase2 and repair should follow tiny/broad/selective/fragment rules from the v73 plan.",
        ],
    }
    _write_json(output_root / "proposal_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    _write_sha_rows(output_root, input_paths)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream4D v73 Phase2 semantic extent proposal summarizer.")
    parser.add_argument("--seed-roots", default="outputs/audit/v73_phase2_dense_seed_scene0011_12,outputs/audit/v73_phase2_dense_seed_scene0050_12")
    parser.add_argument("--witness-summary", default="outputs/audit/v70_carrier_witness/witness_summary.json")
    parser.add_argument("--candidate-rows", default="outputs/audit/v71_candidate_bank/candidate_mask_rows.csv")
    parser.add_argument("--phase1-summary", default="outputs/audit/v73_phase1_source_signal_audit/source_signal_summary.json")
    parser.add_argument("--output-root", default="outputs/audit/v73_phase2_semantic_extent_proposals")
    parser.add_argument("--max-proposals-per-frame", type=int, default=120)
    parser.add_argument("--min-area-ratio", type=float, default=0.0015)
    parser.add_argument("--max-area-ratio", type=float, default=0.22)
    parser.add_argument("--max-background-proxy", type=float, default=0.95)
    parser.add_argument("--p2-min-extent-score", type=float, default=0.34)
    parser.add_argument("--p3-min-extent-score", type=float, default=0.34)
    parser.add_argument("--p5-min-extent-score", type=float, default=0.36)
    parser.add_argument("--p5-unresolved-risk-budget", type=float, default=0.30)
    parser.add_argument("--p5-max-unresolved-per-frame", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
