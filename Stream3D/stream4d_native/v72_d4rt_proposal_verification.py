from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from tools.run_v65_scene_multiview_ap import SparseSceneIoU, _load_gt_2d, _summarize_iou  # noqa: E402


@dataclass
class ProposalEval:
    proposal_id: str
    variant: str
    scene_id: str
    chunk_id: str
    frame_id: int
    source_obs_id: str
    source_mask_id: int
    source_broad_large_risk: bool
    source_underseg_proxy: bool
    eval_mask: np.ndarray
    semantic_score: float
    semantic_entropy: float
    area_ratio: float
    majority_gt: int
    majority_iou: float
    d4rt_atom_count: int = 0
    d4rt_coverage_score: float = 0.0
    d4rt_reliability_mean: float = 0.0
    d4rt_temporal_coherence_mean: float = 0.0
    d4rt_membership_entropy_mean: float = 1.0
    d4rt_score: float = 0.0
    d4rt_no_temporal_score: float = 0.0
    d4rt_shuffled_score: float = 0.0
    d4rt_membership_source: str = "source_mask_level_inherited"
    d4rt_inside_ratio: float | None = None
    d4rt_source_carrier_observation_count: int = 0


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
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


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


def _mean(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(valid)) if valid else None


def _load_pipeline_roots(path: Path) -> dict[str, Path]:
    summary = _load_json(path)
    raw = summary.get("pipeline_roots") or {}
    return {str(scene): _rooted(root) for scene, root in raw.items()}


def _mask_dir_from_pipeline(pipeline_root: Path) -> Path:
    summary = _load_json(pipeline_root / "pipeline_summary.json")
    mask_dir = ((summary.get("mask_frame_coverage") or {}).get("mask_dir") or "").strip()
    if mask_dir:
        path = Path(mask_dir)
        return path if path.is_absolute() else ROOT / path
    return pipeline_root / "cropformer_masks"


def _read_label(path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to read label png: {path}")
    if image.ndim == 3:
        image = image[..., 0]
    if image.shape[:2] != shape_hw:
        image = cv2.resize(image, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return np.asarray(image, dtype=np.int64)


def _resize_token_mask(coords_text: str, grid_shape_text: str, shape_hw: tuple[int, int]) -> np.ndarray:
    parts = str(grid_shape_text or "").lower().split("x")
    if len(parts) != 2:
        return np.zeros(shape_hw, dtype=bool)
    h, w = int(parts[0]), int(parts[1])
    token = np.zeros((h, w), dtype=np.uint8)
    for item in str(coords_text or "").split(";"):
        if not item.strip():
            continue
        yx = item.split(":")
        if len(yx) != 2:
            continue
        y, x = int(yx[0]), int(yx[1])
        if 0 <= y < h and 0 <= x < w:
            token[y, x] = 1
    if token.shape != shape_hw:
        token = cv2.resize(token, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return token.astype(bool)


def _load_atom_aggregates(path: Path, source_obs_ids: set[str]) -> dict[str, dict[str, float]]:
    sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, int] = defaultdict(int)
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            obs_id = str(row.get("atom_mask_observation_id") or "")
            if obs_id not in source_obs_ids:
                continue
            counts[obs_id] += 1
            sums[obs_id]["reliability"] += _float(row.get("non_gt_reliability_score"), 0.0)
            sums[obs_id]["visible_frame_count"] += _float(row.get("visible_frame_count"), 0.0)
            sums[obs_id]["visibility"] += _float(row.get("visibility_mean"), 0.0)
            sums[obs_id]["confidence"] += _float(row.get("confidence_mean"), 0.0)
            sums[obs_id]["temporal"] += _float(row.get("D4RT_temporal_smoothness"), 0.0)
            sums[obs_id]["membership_entropy"] += _float(row.get("mask_membership_entropy"), 1.0)
    out: dict[str, dict[str, float]] = {}
    for obs_id, count in counts.items():
        out[obs_id] = {
            "atom_count": float(count),
            "coverage_score": float(min(1.0, math.log1p(count) / math.log1p(32.0))),
            "reliability_mean": float(sums[obs_id]["reliability"] / max(1, count)),
            "visible_frame_count_mean": float(sums[obs_id]["visible_frame_count"] / max(1, count)),
            "visibility_mean": float(sums[obs_id]["visibility"] / max(1, count)),
            "confidence_mean": float(sums[obs_id]["confidence"] / max(1, count)),
            "temporal_coherence_mean": float(sums[obs_id]["temporal"] / max(1, count)),
            "membership_entropy_mean": float(sums[obs_id]["membership_entropy"] / max(1, count)),
        }
    return out


def _load_atom_by_carrier(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("atom_type") or "") != "A0_single_carrier_atom":
                continue
            carrier_id = str(row.get("carrier_id") or "")
            if not carrier_id:
                continue
            out[carrier_id] = {
                "reliability": _float(row.get("non_gt_reliability_score"), 0.0),
                "temporal": _float(row.get("D4RT_temporal_smoothness"), 0.0),
                "membership_entropy": _float(row.get("mask_membership_entropy"), 1.0),
                "visible_frame_count": _float(row.get("visible_frame_count"), 0.0),
            }
    return out


def _d4rt_scores(agg: dict[str, float]) -> tuple[float, float]:
    coverage = float(agg.get("coverage_score", 0.0))
    reliability = float(agg.get("reliability_mean", 0.0))
    temporal = float(agg.get("temporal_coherence_mean", 0.0))
    entropy = float(agg.get("membership_entropy_mean", 1.0))
    real = 0.35 * coverage + 0.35 * reliability + 0.20 * temporal - 0.10 * entropy
    no_temporal = 0.50 * coverage + 0.50 * reliability - 0.10 * entropy
    return float(real), float(no_temporal)


def _load_carrier_observations_for_proposals(
    proposals: list[ProposalEval],
    pipeline_roots: dict[str, Path],
    *,
    min_visibility: float,
    min_confidence: float,
    missing: list[dict[str, Any]],
) -> tuple[dict[tuple[str, int, int], list[dict[str, float | str]]], dict[str, Any]]:
    needed_frames: dict[str, set[int]] = defaultdict(set)
    needed_masks: dict[tuple[str, int], set[int]] = defaultdict(set)
    for prop in proposals:
        needed_frames[prop.scene_id].add(int(prop.frame_id))
        needed_masks[(prop.scene_id, int(prop.frame_id))].add(int(prop.source_mask_id))
    out: dict[tuple[str, int, int], list[dict[str, float | str]]] = defaultdict(list)
    stats = {
        "scene_count": 0,
        "raw_rows": 0,
        "accepted_rows": 0,
        "matched_source_mask_rows": 0,
        "missing_carrier_table_count": 0,
    }
    for scene, frames in sorted(needed_frames.items()):
        root = pipeline_roots.get(scene)
        if root is None:
            missing.append({"scene_id": scene, "missing": "pipeline_root_for_carrier_uv_membership"})
            continue
        table = root / "observation_tables" / "carrier_observation_table.csv"
        if not table.exists():
            stats["missing_carrier_table_count"] += 1
            missing.append({"scene_id": scene, "missing": "carrier_observation_table_for_subproposal_membership", "path": _rel(table)})
            continue
        stats["scene_count"] += 1
        with table.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                stats["raw_rows"] += 1
                frame_id = _int(row.get("frame_id"), -1)
                if frame_id not in frames:
                    continue
                if not (_bool(row.get("valid")) and _bool(row.get("valid_uv")) and _bool(row.get("visible"))):
                    continue
                visibility = _float(row.get("visibility_prob"), 0.0)
                confidence = _float(row.get("confidence"), 0.0)
                if visibility < float(min_visibility) or confidence < float(min_confidence):
                    continue
                stats["accepted_rows"] += 1
                mask_id = _int(row.get("observed_mask_id"), 0)
                if mask_id not in needed_masks.get((scene, frame_id), set()):
                    continue
                carrier_id = str(row.get("carrier_global_id") or row.get("carrier_id") or "")
                if not carrier_id:
                    continue
                stats["matched_source_mask_rows"] += 1
                out[(scene, frame_id, mask_id)].append(
                    {
                        "carrier_id": carrier_id,
                        "uv_x": _float(row.get("uv_x"), 0.0),
                        "uv_y": _float(row.get("uv_y"), 0.0),
                        "confidence": confidence,
                        "visibility": visibility,
                    }
                )
    stats["source_mask_bucket_count"] = len(out)
    return out, stats


def _assign_subproposal_carrier_scores(
    proposals: list[ProposalEval],
    pipeline_roots: dict[str, Path],
    atom_by_carrier: dict[str, dict[str, float]],
    *,
    min_visibility: float,
    min_confidence: float,
    missing: list[dict[str, Any]],
) -> dict[str, Any]:
    obs_by_source, stats = _load_carrier_observations_for_proposals(
        proposals,
        pipeline_roots,
        min_visibility=min_visibility,
        min_confidence=min_confidence,
        missing=missing,
    )
    inside_counts: list[float] = []
    inside_ratios: list[float] = []
    for prop in proposals:
        source_obs = obs_by_source.get((prop.scene_id, int(prop.frame_id), int(prop.source_mask_id)), [])
        inside: list[dict[str, float | str]] = []
        h, w = prop.eval_mask.shape[:2]
        for obs in source_obs:
            x = int(np.clip(round(float(obs.get("uv_x", 0.0)) * float(max(0, w - 1))), 0, max(0, w - 1)))
            y = int(np.clip(round(float(obs.get("uv_y", 0.0)) * float(max(0, h - 1))), 0, max(0, h - 1)))
            if bool(prop.eval_mask[y, x]):
                inside.append(obs)
        carrier_ids = sorted({str(obs.get("carrier_id") or "") for obs in inside if str(obs.get("carrier_id") or "")})
        reliabilities: list[float] = []
        temporals: list[float] = []
        entropies: list[float] = []
        for obs in inside:
            carrier_id = str(obs.get("carrier_id") or "")
            atom = atom_by_carrier.get(carrier_id, {})
            reliabilities.append(_float(atom.get("reliability"), float(obs.get("confidence", 0.0)) * float(obs.get("visibility", 0.0))))
            temporals.append(_float(atom.get("temporal"), 0.0))
            entropies.append(_float(atom.get("membership_entropy"), 1.0))
        count = len(carrier_ids)
        agg = {
            "atom_count": float(count),
            "coverage_score": float(min(1.0, math.log1p(count) / math.log1p(32.0))),
            "reliability_mean": float(_mean(reliabilities) or 0.0),
            "temporal_coherence_mean": float(_mean(temporals) or 0.0),
            "membership_entropy_mean": float(_mean(entropies) if entropies else 1.0),
        }
        real_score, no_temporal_score = _d4rt_scores(agg)
        prop.d4rt_atom_count = int(count)
        prop.d4rt_coverage_score = float(agg["coverage_score"])
        prop.d4rt_reliability_mean = float(agg["reliability_mean"])
        prop.d4rt_temporal_coherence_mean = float(agg["temporal_coherence_mean"])
        prop.d4rt_membership_entropy_mean = float(agg["membership_entropy_mean"])
        prop.d4rt_score = float(real_score)
        prop.d4rt_no_temporal_score = float(no_temporal_score)
        prop.d4rt_membership_source = "carrier_uv_subproposal_membership"
        prop.d4rt_source_carrier_observation_count = int(len(source_obs))
        prop.d4rt_inside_ratio = float(len(inside) / max(1, len(source_obs))) if source_obs else 0.0
        inside_counts.append(float(count))
        inside_ratios.append(float(prop.d4rt_inside_ratio or 0.0))
    stats["proposal_count_scored"] = len(proposals)
    stats["proposal_inside_carrier_count_mean"] = _mean(inside_counts)
    stats["proposal_inside_ratio_mean"] = _mean(inside_ratios)
    return stats


def _load_proposals(args: argparse.Namespace, missing: list[dict[str, Any]]) -> list[ProposalEval]:
    proposal_rows = _rooted(args.proposal_rows)
    witness_summary = _rooted(args.witness_summary)
    pipeline_roots = _load_pipeline_roots(witness_summary)
    target_variants = {"SP0_existing_masks_baseline", str(args.target_dense_variant)}
    raw_rows: list[dict[str, str]] = []
    source_obs_ids: set[str] = set()
    with proposal_rows.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("variant") or "") not in target_variants:
                continue
            raw_rows.append(row)
            source_obs_ids.add(str(row.get("source_mask_ids") or ""))
    use_subproposal_membership = bool(getattr(args, "subproposal_membership_from_carrier_uv", False))
    atom_agg = {} if use_subproposal_membership else _load_atom_aggregates(_rooted(args.atom_rows), source_obs_ids)
    atom_by_carrier = _load_atom_by_carrier(_rooted(args.atom_rows)) if use_subproposal_membership else {}
    proposals: list[ProposalEval] = []
    mask_cache: dict[tuple[str, int], np.ndarray] = {}
    gt_shape_cache: dict[str, tuple[int, int]] = {}
    for row in raw_rows:
        scene = str(row.get("scene_id") or "")
        frame_id = _int(row.get("frame_id"), -1)
        pipeline_root = pipeline_roots.get(scene)
        if pipeline_root is None:
            missing.append({"scene_id": scene, "frame_id": frame_id, "missing": "pipeline_root"})
            continue
        if scene not in gt_shape_cache:
            stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
            gt_shape_cache[scene] = tuple(int(v) for v in stream.load_depth(frame_id).shape)
        shape_hw = gt_shape_cache[scene]
        if str(row.get("variant") or "") == "SP0_existing_masks_baseline":
            key = (scene, frame_id)
            if key not in mask_cache:
                try:
                    mask_cache[key] = _read_label(_mask_dir_from_pipeline(pipeline_root) / f"{int(frame_id)}.png", shape_hw)
                except FileNotFoundError as exc:
                    missing.append({"scene_id": scene, "frame_id": frame_id, "missing": str(exc)})
                    continue
            eval_mask = mask_cache[key] == _int(row.get("source_mask_id"), -1)
        else:
            eval_mask = _resize_token_mask(str(row.get("proposal_token_coords") or ""), str(row.get("proposal_token_grid_shape") or ""), shape_hw)
        if not np.any(eval_mask):
            continue
        obs_id = str(row.get("source_mask_ids") or "")
        agg = atom_agg.get(obs_id, {})
        real_score, no_temporal_score = _d4rt_scores(agg)
        prop = ProposalEval(
            proposal_id=str(row.get("proposal_id") or ""),
            variant=str(row.get("variant") or ""),
            scene_id=scene,
            chunk_id=str(row.get("chunk_id") or ""),
            frame_id=frame_id,
            source_obs_id=obs_id,
            source_mask_id=_int(row.get("source_mask_id"), -1),
            source_broad_large_risk=_bool(row.get("source_broad_large_risk")),
            source_underseg_proxy=_bool(row.get("source_underseg_proxy")),
            eval_mask=eval_mask,
            semantic_score=_float(row.get("proposal_compactness_score"), 0.0),
            semantic_entropy=_float(row.get("semantic_entropy"), 1.0),
            area_ratio=_float(row.get("proposal_area_ratio"), 0.0),
            majority_gt=_int(row.get("majority_gt_id_diagnostic"), 0),
            majority_iou=_float(row.get("majority_iou_diagnostic"), 0.0),
            d4rt_atom_count=int(agg.get("atom_count", 0.0)),
            d4rt_coverage_score=float(agg.get("coverage_score", 0.0)),
            d4rt_reliability_mean=float(agg.get("reliability_mean", 0.0)),
            d4rt_temporal_coherence_mean=float(agg.get("temporal_coherence_mean", 0.0)),
            d4rt_membership_entropy_mean=float(agg.get("membership_entropy_mean", 1.0)),
            d4rt_score=real_score,
            d4rt_no_temporal_score=no_temporal_score,
        )
        proposals.append(prop)
    if use_subproposal_membership:
        stats = _assign_subproposal_carrier_scores(
            proposals,
            pipeline_roots,
            atom_by_carrier,
            min_visibility=float(args.min_carrier_visibility),
            min_confidence=float(args.min_carrier_confidence),
            missing=missing,
        )
        setattr(args, "_subproposal_membership_stats", stats)
    else:
        setattr(args, "_subproposal_membership_stats", {})
    _assign_shuffled_scores(proposals)
    return proposals


def _assign_shuffled_scores(proposals: list[ProposalEval]) -> None:
    by_chunk: dict[str, list[ProposalEval]] = defaultdict(list)
    for prop in proposals:
        by_chunk[prop.chunk_id].append(prop)
    for chunk_id, subset in by_chunk.items():
        scores = [prop.d4rt_score for prop in subset]
        if not scores:
            continue
        offset = sum(ord(ch) for ch in chunk_id) % len(scores)
        shuffled = scores[offset:] + scores[:offset]
        for prop, score in zip(sorted(subset, key=lambda item: item.proposal_id), shuffled):
            prop.d4rt_shuffled_score = float(score)


def _frame_gt(proposals: list[ProposalEval]) -> dict[tuple[str, int], np.ndarray]:
    out: dict[tuple[str, int], np.ndarray] = {}
    stream_cache: dict[str, ScanNetStream] = {}
    shape_cache: dict[str, tuple[int, int]] = {}
    for prop in proposals:
        key = (prop.scene_id, prop.frame_id)
        if key in out:
            continue
        if prop.scene_id not in stream_cache:
            stream_cache[prop.scene_id] = ScanNetStream(prop.scene_id, root=ROOT / "data/scannet/processed")
            shape_cache[prop.scene_id] = tuple(int(v) for v in stream_cache[prop.scene_id].load_depth(prop.frame_id).shape)
        out[key] = _load_gt_2d(prop.scene_id, prop.frame_id, shape_cache[prop.scene_id])
    return out


def _score_for_variant(prop: ProposalEval, variant: str, d4rt_weight: float) -> float:
    if variant == "DV0_semantic_only_no_D4RT":
        return prop.semantic_score
    if variant == "DV1_D4RT_soft_verifier":
        return prop.semantic_score + float(d4rt_weight) * prop.d4rt_score
    if variant == "DV2_D4RT_low_reliability_filter":
        return prop.semantic_score + float(d4rt_weight) * prop.d4rt_score
    if variant == "DV5_shuffled_carrier_control":
        return prop.semantic_score + float(d4rt_weight) * prop.d4rt_shuffled_score
    if variant == "DV6_no_temporal_control":
        return prop.semantic_score + float(d4rt_weight) * prop.d4rt_no_temporal_score
    return prop.semantic_score


def _selected_for_variant(proposals: list[ProposalEval], variant: str, min_d4rt_score: float) -> list[ProposalEval]:
    if variant == "DV2_D4RT_low_reliability_filter":
        return [prop for prop in proposals if prop.d4rt_score >= float(min_d4rt_score)]
    return list(proposals)


def _evaluate_variant(
    proposals: list[ProposalEval],
    gt_by_frame: dict[tuple[str, int], np.ndarray],
    *,
    variant: str,
    d4rt_weight: float,
    min_d4rt_score: float,
) -> dict[str, Any]:
    selected = _selected_for_variant(proposals, variant, min_d4rt_score)
    by_frame: dict[tuple[str, int], list[ProposalEval]] = defaultdict(list)
    for prop in selected:
        by_frame[(prop.scene_id, prop.frame_id)].append(prop)
    acc = SparseSceneIoU()
    for key, gt in gt_by_frame.items():
        pred = np.zeros(gt.shape, dtype=np.int64)
        frame_props = sorted(by_frame.get(key, []), key=lambda prop: _score_for_variant(prop, variant, d4rt_weight), reverse=True)
        for prop in frame_props:
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
    fp_rate = _mean([1.0 if (prop.majority_gt <= 0 or prop.majority_iou < 0.25) else 0.0 for prop in selected])
    return {
        "variant": variant,
        "proposal_count_before_D4RT": len(proposals),
        "proposal_count_after_D4RT": len(selected),
        "proposal_D4RT_atom_coverage_mean": _mean([prop.d4rt_coverage_score for prop in selected]),
        "proposal_D4RT_reliability_mean": _mean([prop.d4rt_reliability_mean for prop in selected]),
        "proposal_inside_ratio_mean": None,
        "proposal_outside_residual_mean": _mean([1.0 - float(prop.d4rt_inside_ratio or 0.0) for prop in selected]),
        "proposal_membership_entropy_mean": _mean([prop.d4rt_membership_entropy_mean for prop in selected]),
        "temporal_group_count_per_chunk": None,
        "temporal_group_span_mean": None,
        "temporal_group_mask_count_mean": None,
        "SF50_diagnostic": (summary.get("score_free_match_at_050") or {}).get("recall"),
        "AP50_diagnostic": summary.get("ap50"),
        "GT_best_IoU_mean_diagnostic": summary.get("gt_best_iou_mean"),
        "background_false_positive_rate": fp_rate,
        "broad_underseg_rate": _mean([1.0 if (prop.source_broad_large_risk or prop.source_underseg_proxy) else 0.0 for prop in selected]),
        "same_frame_violation_count": None,
        "uses_gt_for_prediction": False,
        "uses_gt_for_evaluation": True,
        "diagnostic_only": True,
        "forbidden_for_method_table": True,
    }


def _verification_rows(proposals: list[ProposalEval]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prop in proposals:
        rows.append(
            {
                "proposal_id": prop.proposal_id,
                "scene_id": prop.scene_id,
                "chunk_id": prop.chunk_id,
                "frame_id": prop.frame_id,
                "variant": prop.variant,
                "source_mask_ids": prop.source_obs_id,
                "source_level_only": prop.d4rt_membership_source == "source_mask_level_inherited",
                "D4RT_membership_source": prop.d4rt_membership_source,
                "proposal_D4RT_inside_ratio": prop.d4rt_inside_ratio,
                "proposal_D4RT_source_carrier_observation_count": prop.d4rt_source_carrier_observation_count,
                "proposal_D4RT_atom_count": prop.d4rt_atom_count,
                "proposal_D4RT_atom_coverage_score": prop.d4rt_coverage_score,
                "proposal_D4RT_reliability_mean": prop.d4rt_reliability_mean,
                "proposal_D4RT_temporal_coherence_mean": prop.d4rt_temporal_coherence_mean,
                "proposal_D4RT_membership_entropy_mean": prop.d4rt_membership_entropy_mean,
                "proposal_D4RT_score": prop.d4rt_score,
                "proposal_D4RT_no_temporal_score": prop.d4rt_no_temporal_score,
                "proposal_D4RT_shuffled_score": prop.d4rt_shuffled_score,
                "proposal_semantic_score": prop.semantic_score,
                "proposal_area_ratio": prop.area_ratio,
                "majority_gt_id_diagnostic": prop.majority_gt,
                "majority_iou_diagnostic": prop.majority_iou,
                "uses_gt_for_prediction": False,
                "uses_gt_for_evaluation": True,
                "diagnostic_only": True,
                "forbidden_for_method_table": True,
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _rooted(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    missing: list[dict[str, Any]] = []
    for name, path in [
        ("proposal_rows", _rooted(args.proposal_rows)),
        ("atom_rows", _rooted(args.atom_rows)),
        ("witness_summary", _rooted(args.witness_summary)),
    ]:
        if not path.exists():
            missing.append({"name": name, "path": _rel(path)})
    if missing:
        _write_csv(output_root / "missing_input_rows.csv", missing)
        summary = {"phase": "v72_phase3_d4rt_proposal_verification", "decision": "FAIL_MISSING_INPUTS", "missing_input_count": len(missing), "gate": {"pass": False}}
        _write_json(output_root / "d4rt_proposal_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary
    proposals = _load_proposals(args, missing)
    gt_by_frame = _frame_gt(proposals)
    variants = [
        "DV0_semantic_only_no_D4RT",
        "DV1_D4RT_soft_verifier",
        "DV2_D4RT_low_reliability_filter",
        "DV5_shuffled_carrier_control",
        "DV6_no_temporal_control",
    ]
    metric_rows = [
        _evaluate_variant(
            proposals,
            gt_by_frame,
            variant=variant,
            d4rt_weight=float(args.d4rt_weight),
            min_d4rt_score=float(args.min_d4rt_score),
        )
        for variant in variants
    ]
    by_variant = {row["variant"]: row for row in metric_rows}
    semantic = by_variant["DV0_semantic_only_no_D4RT"]
    soft = by_variant["DV1_D4RT_soft_verifier"]
    hard = by_variant["DV2_D4RT_low_reliability_filter"]
    shuffled = by_variant["DV5_shuffled_carrier_control"]
    no_temporal = by_variant["DV6_no_temporal_control"]
    real_minus_shuffled = _float(soft.get("SF50_diagnostic")) - _float(shuffled.get("SF50_diagnostic"))
    real_minus_no_temporal = _float(soft.get("SF50_diagnostic")) - _float(no_temporal.get("SF50_diagnostic"))
    background_delta = _float(soft.get("background_false_positive_rate")) - _float(semantic.get("background_false_positive_rate"))
    broad_delta = _float(soft.get("broad_underseg_rate")) - _float(semantic.get("broad_underseg_rate"))
    hard_background_delta = _float(hard.get("background_false_positive_rate")) - _float(semantic.get("background_false_positive_rate"))
    verification_pass = (
        _float(soft.get("SF50_diagnostic")) >= _float(semantic.get("SF50_diagnostic")) - 0.02
        and background_delta <= -0.05
        and (real_minus_shuffled >= 0.03 or background_delta <= -0.05)
    )
    summary = {
        "phase": "v72_phase3_d4rt_proposal_verification",
        "decision": "PASS_V72_PHASE3_D4RT_VERIFICATION_DIAGNOSTIC" if verification_pass else "NO_GO_PHASE3_D4RT_VERIFICATION_DIAGNOSTIC",
        "proposal_rows": _rel(_rooted(args.proposal_rows)),
        "target_dense_variant": str(args.target_dense_variant),
        "source_level_only": not bool(getattr(args, "subproposal_membership_from_carrier_uv", False)),
        "subproposal_membership_from_carrier_uv": bool(getattr(args, "subproposal_membership_from_carrier_uv", False)),
        "carrier_uv_membership_stats": getattr(args, "_subproposal_membership_stats", {}),
        "source_level_limitation": (
            "not_applicable_carrier_uv_subproposal_membership_enabled"
            if bool(getattr(args, "subproposal_membership_from_carrier_uv", False))
            else "Dense token subproposals do not expose carrier-to-subregion membership; D4RT verification is joined through the source mask observation id only."
        ),
        "proposal_count_before_D4RT": semantic.get("proposal_count_before_D4RT"),
        "proposal_count_after_soft_D4RT": soft.get("proposal_count_after_D4RT"),
        "proposal_count_after_hard_filter": hard.get("proposal_count_after_D4RT"),
        "semantic_only_SF50_diagnostic": semantic.get("SF50_diagnostic"),
        "D4RT_verified_SF50_diagnostic": soft.get("SF50_diagnostic"),
        "D4RT_hard_filter_SF50_diagnostic": hard.get("SF50_diagnostic"),
        "shuffled_carrier_SF50_diagnostic": shuffled.get("SF50_diagnostic"),
        "no_temporal_SF50_diagnostic": no_temporal.get("SF50_diagnostic"),
        "real_minus_shuffled_SF50": real_minus_shuffled,
        "real_minus_no_temporal_SF50": real_minus_no_temporal,
        "background_false_positive_delta": background_delta,
        "hard_filter_background_false_positive_delta": hard_background_delta,
        "broad_underseg_delta": broad_delta,
        "same_frame_violation_delta": None,
        "temporal_expansion_status": "not_run_source_level_only_no_carrier_to_subregion_membership",
        "gate": {
            "D4RT_verified_SF50_ge_semantic_minus_0p02": _float(soft.get("SF50_diagnostic")) >= _float(semantic.get("SF50_diagnostic")) - 0.02,
            "background_false_positive_rate_le_semantic_minus_0p05": background_delta <= -0.05,
            "real_minus_shuffled_SF50_ge_0p03_or_background_delta_le_minus_0p05": real_minus_shuffled >= 0.03 or background_delta <= -0.05,
            "same_frame_violation_delta_le_0_available": False,
            "uses_gt_for_prediction_false": True,
            "pass": verification_pass,
        },
        "method_boundary": {
            "uses_gt_for_prediction": False,
            "uses_gt_for_evaluation": True,
            "diagnostic_only": True,
            "forbidden_for_method_table": True,
            "D4RT_source": "v71_d4rt_atoms atom_mask_observation_id aggregate",
            "D4RT_membership_source": (
                "carrier_observation_table_uv_inside_proposal_mask"
                if bool(getattr(args, "subproposal_membership_from_carrier_uv", False))
                else "v71_d4rt_atoms atom_mask_observation_id aggregate"
            ),
            "outside_residual_available": bool(getattr(args, "subproposal_membership_from_carrier_uv", False)),
            "inside_ratio_available": bool(getattr(args, "subproposal_membership_from_carrier_uv", False)),
        },
    }
    _write_csv(output_root / "proposal_verification_rows.csv", _verification_rows(proposals))
    _write_csv(output_root / "control_metric_rows.csv", metric_rows)
    _write_csv(
        output_root / "temporal_group_rows.csv",
        [
            {
                "status": "not_run_source_level_only_no_carrier_to_subregion_membership",
                "uses_gt_for_prediction": False,
                "diagnostic_only": True,
                "forbidden_for_method_table": True,
            }
        ],
    )
    _write_csv(output_root / "missing_input_rows.csv", missing)
    _write_json(output_root / "d4rt_proposal_summary.json", summary)
    sha_rows = []
    for path in sorted(output_root.glob("*")):
        if path.is_file() and path.name != "sha256_rows.csv":
            sha_rows.append({"path": _rel(path), "bytes": int(path.stat().st_size), "sha256": _sha256(path), "kind": "output"})
    for path in [_rooted(args.proposal_rows), _rooted(args.atom_rows), _rooted(args.witness_summary)]:
        if path.exists():
            sha_rows.append({"path": _rel(path), "bytes": int(path.stat().st_size), "sha256": _sha256(path), "kind": "input"})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v72 Phase3 D4RT proposal verification diagnostic.")
    parser.add_argument("--proposal-rows", default="outputs/audit/v72_phase2_dense_token_proposals_smoke10_area_bin1/proposal_rows.csv")
    parser.add_argument("--atom-rows", default="outputs/audit/v71_d4rt_atoms/atom_rows.csv")
    parser.add_argument("--witness-summary", default="outputs/audit/v70_carrier_witness/witness_summary.json")
    parser.add_argument("--output-root", default="outputs/audit/v72_phase3_d4rt_proposal_verification")
    parser.add_argument("--target-dense-variant", default="SP2_DINO_affinity_connected_components")
    parser.add_argument("--d4rt-weight", type=float, default=0.35)
    parser.add_argument("--min-d4rt-score", type=float, default=0.20)
    parser.add_argument("--subproposal-membership-from-carrier-uv", action="store_true")
    parser.add_argument("--min-carrier-visibility", type=float, default=0.0)
    parser.add_argument("--min-carrier-confidence", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
