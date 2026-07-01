from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from stream4d_native.v67_local_baselines import _row_from_eval, _summarize_variant_all  # noqa: E402
from stream4d_native.v67_mask_universe import _frame_mask_stats  # noqa: E402
from tools.run_v65_scene_multiview_ap import _sha256, _write_csv, _write_json  # noqa: E402
from tools.run_v66_local_chunk_eval import _chunk_rows, _evaluate_frame_data, _float_or_none, _frame_data, _mean, _rel, _score_free  # noqa: E402
from tools.run_v66_scene_mv_ap_probe5 import DEFAULT_SCENES, _mask_dir_from_pipeline, _parse_csv_list  # noqa: E402


@dataclass(frozen=True)
class CandidateMask:
    scene: str
    chunk_id: str
    frame_id: int
    mask_id: int
    obs_id: str
    area_ratio: float
    semantic_entropy: float
    semantic_prototype_margin: float
    trajectory_entropy: float
    d4rt_reliability: float | None
    semantic_prototype_id: str
    source_flags: str
    representative_available: bool
    high_quality_raw_available: bool
    small_mask_risk: bool
    broad_large_risk: bool
    underseg_proxy: bool
    same_frame_overlap_count: float
    same_frame_competing_mask_count: float


@dataclass(frozen=True)
class KeyAtom:
    key_id: str
    scene: str
    chunk_id: str
    frame_id: int
    mask_id: int
    obs_id: str
    atom_weight: float
    d4rt_weight: float
    semantic_weight: float
    d4rt_reliability: float | None
    semantic_only: bool
    semantic_prototype_id: str


@dataclass(frozen=True)
class SetCoverConfig:
    variant: str
    d4rt_gain_weight: float
    semantic_gain_weight: float
    area_penalty: float
    semantic_entropy_penalty: float
    trajectory_entropy_penalty: float
    redundancy_penalty: float
    same_frame_penalty: float
    broad_large_penalty: float
    underseg_penalty: float
    frame_bonus: float
    prototype_bonus: float
    reliability_bonus: float
    use_d4rt_atoms: bool
    use_semantic_atoms: bool
    balanced: bool
    oracle: bool = False


CONFIGS = [
    SetCoverConfig("SC0_area_only_baseline", 0.0, 0.0, -0.50, 0.0, 0.0, 0.05, 0.0, -0.20, 0.0, 0.0, 0.0, 0.0, True, True, False),
    SetCoverConfig("SC1_D4RT_atom_cover", 1.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, True, False, False),
    SetCoverConfig("SC2_D4RT_atom_cover_area_penalty", 1.0, 0.0, 1.00, 0.0, 0.15, 0.10, 0.05, 0.25, 0.25, 0.0, 0.0, 0.15, True, False, False),
    SetCoverConfig("SC3_semantic_atom_cover", 0.0, 1.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.05, 0.05, 0.0, False, True, False),
    SetCoverConfig("SC4_semantic_compact_cover", 0.0, 1.0, 0.50, 1.00, 0.0, 0.10, 0.05, 0.35, 0.30, 0.10, 0.20, 0.0, False, True, True),
    SetCoverConfig("SC5_geo_semantic_cover", 0.65, 1.0, 0.35, 0.35, 0.10, 0.12, 0.05, 0.30, 0.25, 0.10, 0.15, 0.10, True, True, False),
    SetCoverConfig("SC6_geo_semantic_specificity", 0.55, 1.0, 1.00, 1.40, 0.45, 0.35, 0.20, 0.95, 0.85, 0.15, 0.35, 0.20, True, True, True),
    SetCoverConfig("SC7_geo_semantic_balanced", 0.45, 1.0, 0.90, 1.25, 0.40, 0.30, 0.18, 0.85, 0.75, 0.35, 0.60, 0.20, True, True, True),
    SetCoverConfig("SC8_geo_semantic_reliability_weighted", 0.45, 1.0, 0.85, 1.20, 0.35, 0.25, 0.15, 0.80, 0.70, 0.35, 0.55, 0.45, True, True, True),
    SetCoverConfig("SC10_clean_mid_area_proto_margin_repair", 0.35, 0.95, 0.25, 0.35, 0.10, 0.25, 0.08, 1.35, 1.20, 0.20, 0.35, 0.15, True, True, True),
    SetCoverConfig("SC11_clean_mid_area_objectness_rank_repair", 0.05, 0.15, 0.05, -0.20, 0.00, 0.45, 0.08, 2.50, 2.30, 0.35, 0.45, 0.05, True, True, True),
    SetCoverConfig("SC9_oracle_set_cover_diagnostic", 0.5, 0.5, 0.25, 0.25, 0.0, 0.05, 0.0, 0.20, 0.20, 0.0, 0.0, 0.0, True, True, False, True),
]


def _rooted(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float(value: Any, default: float = 0.0) -> float:
    if value is None or str(value).strip() == "":
        return float(default)
    try:
        out = float(value)
    except Exception:
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(float(v) for v in values)
    idx = int(round((len(vals) - 1) * q))
    return float(vals[max(0, min(len(vals) - 1, idx))])


def _load_pipeline_roots(path: Path, scenes: list[str]) -> dict[str, Path]:
    summary = _load_json(path)
    raw = summary.get("pipeline_roots") or {}
    return {scene: _rooted(raw[scene]) for scene in scenes if raw.get(scene)}


def _load_candidates(path: Path, scenes: set[str]) -> dict[str, list[CandidateMask]]:
    out: dict[str, list[CandidateMask]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scene = str(row.get("scene_id") or "")
            if scenes and scene not in scenes:
                continue
            chunk_id = str(row.get("chunk_id") or "")
            broad_large = _bool(row.get("broad_background_risk")) or _bool(row.get("large_mask_risk")) or (_float(row.get("area_ratio"), 0.0) >= 0.30)
            underseg = _float(row.get("underseg_proxy_score"), 0.0) >= 0.75
            rel = row.get("D4RT_carrier_reliability_mean")
            out[chunk_id].append(
                CandidateMask(
                    scene=scene,
                    chunk_id=chunk_id,
                    frame_id=_int(row.get("frame_id"), -1),
                    mask_id=_int(row.get("mask_id"), -1),
                    obs_id=str(row.get("mask_observation_id") or ""),
                    area_ratio=_float(row.get("area_ratio"), 0.0),
                    semantic_entropy=_float(row.get("semantic_entropy"), 1.0),
                    semantic_prototype_margin=_float(row.get("semantic_prototype_margin"), 0.0),
                    trajectory_entropy=_float(row.get("D4RT_carrier_trajectory_entropy"), 0.0),
                    d4rt_reliability=None if rel in (None, "") else _float(rel, 0.0),
                    semantic_prototype_id=str(row.get("semantic_prototype_id") or ""),
                    source_flags=str(row.get("candidate_source_flags") or ""),
                    representative_available=_bool(row.get("representative_available")),
                    high_quality_raw_available=_bool(row.get("raw_cropformer_available")),
                    small_mask_risk=_bool(row.get("small_mask_risk")),
                    broad_large_risk=bool(broad_large),
                    underseg_proxy=bool(underseg),
                    same_frame_overlap_count=_float(row.get("same_frame_overlap_count"), 0.0),
                    same_frame_competing_mask_count=_float(row.get("same_frame_competing_mask_count"), 0.0),
                )
            )
    return out


def _load_key_atoms(path: Path, scenes: set[str], variant: str) -> dict[str, list[KeyAtom]]:
    out: dict[str, list[KeyAtom]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("key_atom_variant") != variant:
                continue
            scene = str(row.get("scene_id") or "")
            if scenes and scene not in scenes:
                continue
            rel_available = _bool(row.get("D4RT_reliability_available"))
            rel = _float(row.get("D4RT_reliability"), 0.0) if rel_available else None
            weight = max(1e-6, _float(row.get("selection_weight"), 1.0))
            semantic_only = _bool(row.get("semantic_only_repair_atom"))
            d4rt_weight = weight if rel_available else 0.0
            sem_weight = weight if semantic_only or str(row.get("semantic_prototype_id") or "") else 0.0
            atom = KeyAtom(
                key_id=str(row.get("key_atom_id") or ""),
                scene=scene,
                chunk_id=str(row.get("chunk_id") or ""),
                frame_id=_int(row.get("frame_id"), -1),
                mask_id=_int(row.get("mask_id"), -1),
                obs_id=str(row.get("mask_observation_id") or ""),
                atom_weight=weight,
                d4rt_weight=d4rt_weight,
                semantic_weight=sem_weight,
                d4rt_reliability=rel,
                semantic_only=semantic_only,
                semantic_prototype_id=str(row.get("semantic_prototype_id") or ""),
            )
            out[atom.chunk_id].append(atom)
    return out


def _candidate_by_pair(candidates: list[CandidateMask]) -> dict[tuple[int, int], CandidateMask]:
    return {(c.frame_id, c.mask_id): c for c in candidates}


def _mask_iou_values(mask: np.ndarray, candidate_ids: list[int], key_ids: list[int]) -> dict[tuple[int, int], float]:
    if not candidate_ids or not key_ids:
        return {}
    binary: dict[int, np.ndarray] = {}
    area: dict[int, int] = {}
    for mid in sorted(set(candidate_ids) | set(key_ids)):
        arr = mask == int(mid)
        binary[mid] = arr
        area[mid] = int(arr.sum())
    out: dict[tuple[int, int], float] = {}
    for cid in candidate_ids:
        if area.get(cid, 0) <= 0:
            continue
        cand = binary[cid]
        for kid in key_ids:
            if area.get(kid, 0) <= 0:
                continue
            if cid == kid:
                out[(cid, kid)] = 1.0
                continue
            inter = int(np.logical_and(cand, binary[kid]).sum())
            if inter <= 0:
                continue
            union = area[cid] + area[kid] - inter
            if union > 0:
                out[(cid, kid)] = float(inter / union)
    return out


def _build_coverage(
    frame_data: list[dict[str, Any]],
    candidates: list[CandidateMask],
    key_atoms: list[KeyAtom],
    *,
    min_iou: float,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    candidates_by_frame: dict[int, list[CandidateMask]] = defaultdict(list)
    keys_by_frame: dict[int, list[KeyAtom]] = defaultdict(list)
    for cand in candidates:
        candidates_by_frame[cand.frame_id].append(cand)
    for atom in key_atoms:
        keys_by_frame[atom.frame_id].append(atom)
    coverage: dict[str, dict[str, float]] = defaultdict(dict)
    coverage_kind: dict[str, dict[str, float]] = defaultdict(dict)
    frame_mask = {int(item["frame_id"]): item["mask"] for item in frame_data}
    for frame_id, frame_candidates in candidates_by_frame.items():
        frame_keys = keys_by_frame.get(frame_id, [])
        mask = frame_mask.get(frame_id)
        if mask is None or not frame_keys:
            continue
        ious = _mask_iou_values(
            np.asarray(mask, dtype=np.int64),
            [cand.mask_id for cand in frame_candidates],
            [atom.mask_id for atom in frame_keys],
        )
        atoms_by_mask = defaultdict(list)
        for atom in frame_keys:
            atoms_by_mask[atom.mask_id].append(atom)
        for cand in frame_candidates:
            for atom in frame_keys:
                cov = 1.0 if cand.mask_id == atom.mask_id else ious.get((cand.mask_id, atom.mask_id), 0.0)
                if cov < min_iou:
                    continue
                coverage[cand.obs_id][atom.key_id] = max(coverage[cand.obs_id].get(atom.key_id, 0.0), float(cov))
                coverage_kind[cand.obs_id][atom.key_id] = 1.0 if atom.d4rt_weight > 0 else 0.0
    return coverage, coverage_kind


def _prefilter_candidates(candidates: list[CandidateMask], key_atoms: list[KeyAtom], limit: int) -> list[CandidateMask]:
    if limit <= 0 or len(candidates) <= limit:
        return list(candidates)
    key_obs = {atom.obs_id for atom in key_atoms}
    selected: list[CandidateMask] = []
    used: set[str] = set()

    def add(cand: CandidateMask) -> None:
        if cand.obs_id in used or len(selected) >= limit:
            return
        selected.append(cand)
        used.add(cand.obs_id)

    def score(cand: CandidateMask) -> tuple[float, float, float, str]:
        rel = cand.d4rt_reliability if cand.d4rt_reliability is not None else 0.0
        risk = (1.0 if cand.broad_large_risk else 0.0) + (1.0 if cand.underseg_proxy else 0.0) + (0.5 if cand.small_mask_risk else 0.0)
        source_bonus = (1.25 if cand.representative_available else 0.0) + (0.45 if cand.high_quality_raw_available else 0.0)
        value = source_bonus + rel + (1.0 - cand.semantic_entropy) - 0.75 * risk - 0.25 * cand.area_ratio
        return (value, -cand.semantic_entropy, -cand.area_ratio, cand.obs_id)

    for cand in sorted(candidates, key=lambda c: (c.obs_id not in key_obs, -score(c)[0], c.obs_id)):
        if cand.obs_id in key_obs:
            add(cand)
    by_frame: dict[int, list[CandidateMask]] = defaultdict(list)
    by_proto: dict[str, list[CandidateMask]] = defaultdict(list)
    for cand in candidates:
        if cand.obs_id in used:
            continue
        by_frame[cand.frame_id].append(cand)
        by_proto[cand.semantic_prototype_id].append(cand)
    for frame_id in sorted(by_frame):
        add(max(by_frame[frame_id], key=score))
        if len(selected) >= limit:
            return selected
    for proto in sorted(by_proto, key=lambda p: (len(by_proto[p]), p)):
        add(max(by_proto[proto], key=score))
        if len(selected) >= limit:
            return selected
    for cand in sorted(candidates, key=score, reverse=True):
        add(cand)
        if len(selected) >= limit:
            break
    return selected


def _diagnostic_mask_stats(frame_data: list[dict[str, Any]], pairs: set[tuple[int, int]]) -> dict[tuple[int, int], dict[str, Any]]:
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for item in frame_data:
        frame_id = int(item["frame_id"])
        mask = item["mask"]
        gt = item["gt"]
        if mask is None or gt is None:
            continue
        stats = _frame_mask_stats(np.asarray(mask, dtype=np.int64), np.asarray(gt, dtype=np.int64))
        for mask_id, value in stats.items():
            key = (frame_id, int(mask_id))
            if key not in pairs:
                continue
            out[key] = value
    return out


def _oracle_gain(cand: CandidateMask, diag_stats: dict[tuple[int, int], dict[str, Any]]) -> float:
    stats = diag_stats.get((cand.frame_id, cand.mask_id), {})
    return float(stats.get("majority_iou") or stats.get("majority_purity") or 0.0)


def _select_set_cover(
    *,
    config: SetCoverConfig,
    candidates: list[CandidateMask],
    key_atoms: list[KeyAtom],
    coverage: dict[str, dict[str, float]],
    diagnostic_stats: dict[tuple[int, int], dict[str, Any]],
    target_coverage: float,
    min_masks: int,
    max_masks: int,
    min_gain: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not candidates or not key_atoms:
        return [], {"stop_reason": "empty_candidates_or_atoms"}
    atom_by_id = {atom.key_id: atom for atom in key_atoms}
    total_weight = sum(atom.atom_weight for atom in key_atoms)
    total_d4rt = sum(atom.d4rt_weight for atom in key_atoms)
    total_sem = sum(atom.semantic_weight for atom in key_atoms)
    covered: dict[str, float] = {atom.key_id: 0.0 for atom in key_atoms}
    selected: list[dict[str, Any]] = []
    selected_obs: set[str] = set()
    selected_frames: Counter[int] = Counter()
    selected_protos: Counter[str] = Counter()
    stop_reason = "max_masks"

    def ratios() -> tuple[float, float, float]:
        cov_total = sum(atom_by_id[k].atom_weight * min(1.0, v) for k, v in covered.items())
        cov_d4rt = sum(atom_by_id[k].d4rt_weight * min(1.0, v) for k, v in covered.items())
        cov_sem = sum(atom_by_id[k].semantic_weight * min(1.0, v) for k, v in covered.items())
        return (
            float(cov_total / max(1e-9, total_weight)),
            float(cov_d4rt / max(1e-9, total_d4rt)) if total_d4rt > 0 else 0.0,
            float(cov_sem / max(1e-9, total_sem)) if total_sem > 0 else 0.0,
        )

    if config.oracle:
        oracle_candidates: list[tuple[float, float, CandidateMask, int]] = []
        best_by_gt: dict[int, tuple[float, float, CandidateMask, int]] = {}
        for cand in candidates:
            stats = diagnostic_stats.get((cand.frame_id, cand.mask_id), {})
            gid = int(stats.get("majority_gt") or 0)
            iou = float(stats.get("majority_iou") or 0.0)
            purity = float(stats.get("majority_purity") or 0.0)
            item = (iou, purity, cand, gid)
            oracle_candidates.append(item)
            if gid > 0 and (gid not in best_by_gt or item[:2] > best_by_gt[gid][:2]):
                best_by_gt[gid] = item
        selected_oracle_items = sorted(best_by_gt.values(), key=lambda x: (x[0], x[1]), reverse=True)
        used_oracle = {item[2].obs_id for item in selected_oracle_items}
        for item in sorted(oracle_candidates, key=lambda x: (x[0], x[1]), reverse=True):
            if len(selected_oracle_items) >= max_masks:
                break
            if item[2].obs_id in used_oracle:
                continue
            selected_oracle_items.append(item)
            used_oracle.add(item[2].obs_id)
        selected_rows: list[dict[str, Any]] = []
        for rank, (_iou, _purity, cand, _gid) in enumerate(selected_oracle_items[:max_masks]):
            cov = coverage.get(cand.obs_id, {})
            new_total = 0.0
            new_d4rt = 0.0
            new_sem = 0.0
            new_count = 0
            for key_id, cov_value in cov.items():
                atom = atom_by_id.get(key_id)
                if atom is None:
                    continue
                old = covered.get(key_id, 0.0)
                inc = max(0.0, min(1.0, cov_value) - old)
                if inc <= 0.0:
                    continue
                new_total += atom.atom_weight * inc
                new_d4rt += atom.d4rt_weight * inc
                new_sem += atom.semantic_weight * inc
                new_count += 1
            for key_id, cov_value in cov.items():
                covered[key_id] = max(covered.get(key_id, 0.0), min(1.0, float(cov_value)))
            cov_ratio, d4rt_ratio, sem_ratio = ratios()
            selected_rows.append(
                {
                    "candidate": cand,
                    "rank": rank,
                    "score_total": float(_iou + _purity),
                    "score_new_atom_coverage": new_total,
                    "score_d4rt_coverage": new_d4rt,
                    "score_semantic_coverage": new_sem,
                    "new_key_atom_count": new_count,
                    "new_key_atom_weight": new_total,
                    "covered_key_atom_count_after_selection": sum(1 for v in covered.values() if v >= 0.25),
                    "covered_key_atom_weight_ratio_after_selection": cov_ratio,
                    "covered_D4RT_atom_weight_ratio_after_selection": d4rt_ratio,
                    "covered_semantic_atom_weight_ratio_after_selection": sem_ratio,
                }
            )
        cov_ratio, d4rt_ratio, sem_ratio = ratios()
        return selected_rows, {
            "stop_reason": "oracle_gt_best_iou_budget",
            "covered_key_atom_weight_ratio": cov_ratio,
            "covered_D4RT_atom_weight_ratio": d4rt_ratio,
            "covered_semantic_atom_weight_ratio": sem_ratio,
            "covered_key_atom_count": sum(1 for v in covered.values() if v >= 0.25),
            "total_key_atom_count": len(key_atoms),
            "total_key_atom_weight": total_weight,
            "selected_mask_count": len(selected_rows),
        }

    by_obs = {cand.obs_id: cand for cand in candidates}
    for rank in range(max_masks):
        best: tuple[float, CandidateMask, dict[str, Any]] | None = None
        for cand in candidates:
            if cand.obs_id in selected_obs:
                continue
            cov = coverage.get(cand.obs_id, {})
            new_total = 0.0
            new_d4rt = 0.0
            new_sem = 0.0
            new_count = 0
            for key_id, cov_value in cov.items():
                atom = atom_by_id.get(key_id)
                if atom is None:
                    continue
                old = covered.get(key_id, 0.0)
                inc = max(0.0, min(1.0, cov_value) - old)
                if inc <= 0.0:
                    continue
                if (atom.d4rt_weight > 0 and config.use_d4rt_atoms) or (atom.semantic_weight > 0 and config.use_semantic_atoms):
                    new_total += atom.atom_weight * inc
                    new_d4rt += atom.d4rt_weight * inc
                    new_sem += atom.semantic_weight * inc
                    new_count += 1
            if config.oracle:
                new_total += 2.0 * _oracle_gain(cand, diagnostic_stats)
            score = 0.0
            if config.variant == "SC0_area_only_baseline":
                score = math.sqrt(max(0.0, cand.area_ratio))
            else:
                score += config.d4rt_gain_weight * new_d4rt
                score += config.semantic_gain_weight * new_sem
            if config.balanced:
                if selected_frames[cand.frame_id] == 0:
                    score += config.frame_bonus
                if selected_protos[cand.semantic_prototype_id] == 0:
                    score += config.prototype_bonus
            if cand.d4rt_reliability is not None:
                score += config.reliability_bonus * cand.d4rt_reliability
            if not config.oracle:
                if config.variant == "SC10_clean_mid_area_proto_margin_repair":
                    clean = not cand.broad_large_risk and not cand.underseg_proxy
                    if clean and 0.015 <= cand.area_ratio <= 0.22:
                        score += 2.25
                    elif clean and 0.008 <= cand.area_ratio < 0.015:
                        score += 0.75
                    elif cand.area_ratio < 0.006:
                        score -= 1.00
                    score += 3.00 * cand.semantic_prototype_margin
                if config.variant == "SC11_clean_mid_area_objectness_rank_repair":
                    clean = not cand.broad_large_risk and not cand.underseg_proxy
                    if clean and 0.015 <= cand.area_ratio <= 0.22:
                        score += 5.00
                    elif clean and 0.008 <= cand.area_ratio < 0.015:
                        score += 1.25
                    elif cand.area_ratio < 0.006:
                        score -= 1.75
                    elif cand.area_ratio > 0.22:
                        score -= 1.00
                    score += 3.50 * cand.semantic_prototype_margin
                    score += 0.35 * cand.semantic_entropy
                if cand.representative_available:
                    if config.variant in {"SC6_geo_semantic_specificity", "SC7_geo_semantic_balanced", "SC8_geo_semantic_reliability_weighted"}:
                        score += 1.75
                    elif config.variant in {"SC10_clean_mid_area_proto_margin_repair", "SC11_clean_mid_area_objectness_rank_repair"}:
                        score += 1.10
                    elif config.variant in {"SC5_geo_semantic_cover", "SC4_semantic_compact_cover"}:
                        score += 0.85
                    else:
                        score += 0.25
                if cand.high_quality_raw_available:
                    score += 0.35
            score -= config.area_penalty * cand.area_ratio
            score -= config.semantic_entropy_penalty * cand.semantic_entropy
            score -= config.trajectory_entropy_penalty * cand.trajectory_entropy
            score -= config.redundancy_penalty * (selected_frames[cand.frame_id] + selected_protos[cand.semantic_prototype_id]) / max(1, rank + 1)
            score -= config.same_frame_penalty * (cand.same_frame_overlap_count + cand.same_frame_competing_mask_count) / 10.0
            if cand.broad_large_risk:
                score -= config.broad_large_penalty
            if cand.underseg_proxy:
                score -= config.underseg_penalty
            if cand.small_mask_risk and config.variant in {"SC4_semantic_compact_cover", "SC6_geo_semantic_specificity", "SC7_geo_semantic_balanced", "SC8_geo_semantic_reliability_weighted"}:
                score -= 0.75
            detail = {
                "score_total": score,
                "score_new_atom_coverage": new_total,
                "score_d4rt_coverage": new_d4rt,
                "score_semantic_coverage": new_sem,
                "new_key_atom_count": new_count,
                "new_key_atom_weight": new_total,
            }
            if best is None or score > best[0]:
                best = (float(score), cand, detail)
        if best is None:
            stop_reason = "no_candidate"
            break
        best_score, cand, detail = best
        if rank >= min_masks and best_score < min_gain:
            stop_reason = "marginal_gain_below_min"
            break
        selected_obs.add(cand.obs_id)
        selected_frames[cand.frame_id] += 1
        selected_protos[cand.semantic_prototype_id] += 1
        cov = coverage.get(cand.obs_id, {})
        for key_id, cov_value in cov.items():
            covered[key_id] = max(covered.get(key_id, 0.0), min(1.0, float(cov_value)))
        cov_ratio, d4rt_ratio, sem_ratio = ratios()
        selected.append(
            {
                "candidate": cand,
                "rank": rank,
                **detail,
                "covered_key_atom_count_after_selection": sum(1 for v in covered.values() if v >= 0.25),
                "covered_key_atom_weight_ratio_after_selection": cov_ratio,
                "covered_D4RT_atom_weight_ratio_after_selection": d4rt_ratio,
                "covered_semantic_atom_weight_ratio_after_selection": sem_ratio,
            }
        )
        if rank + 1 >= min_masks and cov_ratio >= target_coverage:
            stop_reason = "target_coverage"
            break
    cov_ratio, d4rt_ratio, sem_ratio = ratios()
    return selected, {
        "stop_reason": stop_reason,
        "covered_key_atom_weight_ratio": cov_ratio,
        "covered_D4RT_atom_weight_ratio": d4rt_ratio,
        "covered_semantic_atom_weight_ratio": sem_ratio,
        "covered_key_atom_count": sum(1 for v in covered.values() if v >= 0.25),
        "total_key_atom_count": len(key_atoms),
        "total_key_atom_weight": total_weight,
        "selected_mask_count": len(selected),
    }


def _selected_mapping(selected: list[dict[str, Any]]) -> tuple[dict[tuple[int, int], int], dict[str, Any]]:
    mapping: dict[tuple[int, int], int] = {}
    for idx, row in enumerate(selected):
        cand: CandidateMask = row["candidate"]
        mapping[(cand.frame_id, cand.mask_id)] = idx + 1
    return mapping, {
        "selected_mask_count": len(selected),
        "support_pair_count": len(mapping),
        "duplicate_frame_mask_conflict_pairs": 0,
        "duplicate_frame_mask_conflict_rate": 0.0,
        "max_objects_per_frame_mask": 1 if mapping else 0,
    }


def _oracle_eval_for_selected(
    *,
    frame_data: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    diagnostic_stats: dict[tuple[int, int], dict[str, Any]],
    variant: str,
) -> dict[str, Any]:
    mapping: dict[tuple[int, int], int] = {}
    skipped = 0
    for row in selected:
        cand: CandidateMask = row["candidate"]
        stats = diagnostic_stats.get((cand.frame_id, cand.mask_id), {})
        gid = int(stats.get("majority_gt") or 0)
        if gid <= 0:
            skipped += 1
            continue
        mapping[(cand.frame_id, cand.mask_id)] = gid
    summary, _iou, _pred_ids, _gt_ids = _evaluate_frame_data(
        frame_data=frame_data,
        variant=f"{variant}_oracle_representative_diagnostic",
        mapping=mapping,
        raw_per_frame_masks=False,
    )
    return {
        "representative_oracle_SF50": _score_free(summary),
        "representative_oracle_AP50": summary.get("ap50"),
        "representative_oracle_AP25": summary.get("ap25"),
        "representative_oracle_AP": summary.get("ap"),
        "representative_GT_best_IoU_mean": summary.get("gt_best_iou_mean"),
        "representative_pred_best_IoU_median": summary.get("pred_best_iou_median"),
        "representative_oracle_mapping_count": len(mapping),
        "representative_oracle_skipped_no_gt": skipped,
        "representative_oracle_uses_gt_for_selection": False,
        "representative_oracle_diagnostic_only": True,
    }


def _selected_mask_rows(
    *,
    config: SetCoverConfig,
    selected: list[dict[str, Any]],
    diagnostic_stats: dict[tuple[int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in selected:
        cand: CandidateMask = row["candidate"]
        stats = diagnostic_stats.get((cand.frame_id, cand.mask_id), {})
        rows.append(
            {
                "scene_id": cand.scene,
                "chunk_id": cand.chunk_id,
                "variant": config.variant,
                "selection_rank": row["rank"],
                "frame_id": cand.frame_id,
                "mask_id": cand.mask_id,
                "mask_observation_id": cand.obs_id,
                "score_total": row["score_total"],
                "score_new_atom_coverage": row["score_new_atom_coverage"],
                "score_d4rt_coverage": row["score_d4rt_coverage"],
                "score_semantic_coverage": row["score_semantic_coverage"],
                "penalty_area": config.area_penalty * cand.area_ratio,
                "penalty_semantic_entropy": config.semantic_entropy_penalty * cand.semantic_entropy,
                "penalty_trajectory_entropy": config.trajectory_entropy_penalty * cand.trajectory_entropy,
                "penalty_redundancy": config.redundancy_penalty,
                "penalty_same_frame_conflict": config.same_frame_penalty * (cand.same_frame_overlap_count + cand.same_frame_competing_mask_count) / 10.0,
                "new_key_atom_count": row["new_key_atom_count"],
                "new_key_atom_weight": row["new_key_atom_weight"],
                "covered_key_atom_count_after_selection": row["covered_key_atom_count_after_selection"],
                "covered_key_atom_weight_ratio_after_selection": row["covered_key_atom_weight_ratio_after_selection"],
                "selected_mask_area_ratio": cand.area_ratio,
                "selected_mask_semantic_entropy": cand.semantic_entropy,
                "selected_mask_semantic_prototype_margin": cand.semantic_prototype_margin,
                "selected_mask_trajectory_entropy": cand.trajectory_entropy,
                "selected_mask_D4RT_reliability_mean": cand.d4rt_reliability,
                "selected_mask_semantic_prototype_id": cand.semantic_prototype_id,
                "selected_mask_source_flags": cand.source_flags,
                "uses_gt_for_prediction": bool(config.oracle),
                "diagnostic_best_GT_id": stats.get("majority_gt"),
                "diagnostic_best_GT_iou": stats.get("majority_iou"),
                "diagnostic_majority_GT_id": stats.get("majority_gt"),
                "diagnostic_majority_GT_purity": stats.get("majority_purity"),
                "diagnostic_underseg_GT_count": stats.get("positive_gt_count"),
                "forbidden_for_method_table": bool(config.oracle),
                "diagnostic_only": bool(config.oracle),
            }
        )
    return rows


def _chunk_metric_row(
    *,
    config: SetCoverConfig,
    scene: str,
    chunk_id: str,
    frame_ids: list[int],
    chunk: dict[str, Any],
    frame_data: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    cover_diag: dict[str, Any],
    diagnostic_stats: dict[tuple[int, int], dict[str, Any]],
    pipeline_root: Path,
) -> dict[str, Any]:
    mapping, diag = _selected_mapping(selected)
    row = _row_from_eval(
        scene=scene,
        chunk_id=chunk_id,
        variant=config.variant,
        frame_ids=frame_ids,
        chunk=chunk,
        frame_data=frame_data,
        mapping=mapping,
        raw_per_frame_masks=False,
        diag={**diag, **cover_diag},
        uses_gt_for_prediction=bool(config.oracle),
        forbidden_for_method_table=bool(config.oracle),
        pipeline_root=pipeline_root,
    )
    row.update(cover_diag)
    row.update(
        _oracle_eval_for_selected(
            frame_data=frame_data,
            selected=selected,
            diagnostic_stats=diagnostic_stats,
            variant=config.variant,
        )
    )
    if selected:
        cands = [entry["candidate"] for entry in selected]
        row.update(
            {
                "selected_mask_area_ratio_mean": _mean([cand.area_ratio for cand in cands]),
                "selected_mask_area_ratio_p90": _percentile([cand.area_ratio for cand in cands], 0.90),
                "selected_mask_semantic_entropy_mean": _mean([cand.semantic_entropy for cand in cands]),
                "selected_mask_semantic_prototype_margin_mean": _mean([cand.semantic_prototype_margin for cand in cands]),
                "selected_mask_trajectory_entropy_mean": _mean([cand.trajectory_entropy for cand in cands]),
                "selected_mask_D4RT_reliability_mean": _mean([cand.d4rt_reliability for cand in cands if cand.d4rt_reliability is not None]),
                "selected_semantic_prototype_count": len({cand.semantic_prototype_id for cand in cands if cand.semantic_prototype_id}),
                "selected_frame_count": len({cand.frame_id for cand in cands}),
                "selected_frame_coverage_rate": len({cand.frame_id for cand in cands}) / max(1, len(frame_ids)),
                "broad_large_selected_rate": sum(1 for cand in cands if cand.broad_large_risk) / max(1, len(cands)),
                "underseg_proxy_selected_rate": sum(1 for cand in cands if cand.underseg_proxy) / max(1, len(cands)),
                "same_frame_overlap_selected_rate": _mean([1.0 if cand.same_frame_overlap_count > 0 else 0.0 for cand in cands]),
            }
        )
    return row


def _summarize_setcover(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    base = _summarize_variant_all(rows, variant)
    subset = [row for row in rows if row.get("variant") == variant]
    selected_counts = [_float_or_none(row.get("selected_mask_count")) for row in subset]
    selected_counts_f = [float(v) for v in selected_counts if v is not None]
    out = {
        "variant": variant,
        "selected_mask_count_total": int(sum(selected_counts_f)),
        "selected_mask_count_per_chunk_mean": _mean(selected_counts),
        "selected_mask_count_per_frame_mean": _mean([
            (float(row.get("selected_mask_count") or 0.0) / max(1.0, float(row.get("chunk_frame_count") or 1.0))) for row in subset
        ]),
        "selected_mask_count_per_chunk_p90": _percentile(selected_counts_f, 0.90),
        "covered_key_atom_weight_ratio": _mean([_float_or_none(row.get("covered_key_atom_weight_ratio")) for row in subset]),
        "covered_D4RT_atom_weight_ratio": _mean([_float_or_none(row.get("covered_D4RT_atom_weight_ratio")) for row in subset]),
        "covered_semantic_atom_weight_ratio": _mean([_float_or_none(row.get("covered_semantic_atom_weight_ratio")) for row in subset]),
        "selected_mask_area_ratio_mean": _mean([_float_or_none(row.get("selected_mask_area_ratio_mean")) for row in subset]),
        "selected_mask_area_ratio_p90": _mean([_float_or_none(row.get("selected_mask_area_ratio_p90")) for row in subset]),
        "selected_mask_semantic_entropy_mean": _mean([_float_or_none(row.get("selected_mask_semantic_entropy_mean")) for row in subset]),
        "selected_mask_semantic_prototype_margin_mean": _mean([_float_or_none(row.get("selected_mask_semantic_prototype_margin_mean")) for row in subset]),
        "selected_mask_trajectory_entropy_mean": _mean([_float_or_none(row.get("selected_mask_trajectory_entropy_mean")) for row in subset]),
        "selected_mask_D4RT_reliability_mean": _mean([_float_or_none(row.get("selected_mask_D4RT_reliability_mean")) for row in subset]),
        "selected_semantic_prototype_count_mean": _mean([_float_or_none(row.get("selected_semantic_prototype_count")) for row in subset]),
        "selected_frame_coverage_rate": _mean([_float_or_none(row.get("selected_frame_coverage_rate")) for row in subset]),
        "broad_large_selected_rate": _mean([_float_or_none(row.get("broad_large_selected_rate")) for row in subset]),
        "underseg_proxy_selected_rate": _mean([_float_or_none(row.get("underseg_proxy_selected_rate")) for row in subset]),
        "same_frame_overlap_selected_rate": _mean([_float_or_none(row.get("same_frame_overlap_selected_rate")) for row in subset]),
        "representative_oracle_SF50": _mean([_float_or_none(row.get("representative_oracle_SF50")) for row in subset]),
        "representative_oracle_AP50": _mean([_float_or_none(row.get("representative_oracle_AP50")) for row in subset]),
        "representative_GT_best_IoU_mean": _mean([_float_or_none(row.get("representative_GT_best_IoU_mean")) for row in subset]),
        "representative_pred_best_IoU_median": _mean([_float_or_none(row.get("representative_pred_best_IoU_median")) for row in subset]),
        "uses_gt_for_prediction": base.get("uses_gt_for_prediction"),
        "forbidden_for_method_table": base.get("forbidden_for_method_table"),
        "diagnostic_only": base.get("diagnostic_only"),
    }
    out.update({f"local_eval_{k}": v for k, v in base.items() if k not in out})
    return out


def _variant_by_name(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("variant")): row for row in rows}


def _write_visuals(visual_root: Path, summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    visual_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    specs = [
        ("setcover_coverage_ratio.png", "covered_key_atom_weight_ratio"),
        ("setcover_selected_count.png", "selected_mask_count_per_chunk_mean"),
        ("setcover_broad_rate.png", "broad_large_selected_rate"),
        ("setcover_oracle_sf50.png", "representative_oracle_SF50"),
    ]
    for filename, field in specs:
        labels = [str(row["variant"]).split("_")[0] for row in summaries]
        values = [row.get(field) for row in summaries]
        img = np.full((420, max(900, 110 * len(labels)), 3), 255, dtype=np.uint8)
        cv2.putText(img, field, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
        valid = [float(v) for v in values if v is not None]
        ymax = max(valid) * 1.15 if valid else 1.0
        ymax = max(1e-6, ymax)
        x0, y0, h = 50, 70, 270
        step = (img.shape[1] - 90) / max(1, len(labels))
        bw = max(12, int(step * 0.65))
        for idx, (label, value) in enumerate(zip(labels, values)):
            val = float(value) if value is not None else 0.0
            bh = int(h * max(0.0, min(1.0, val / ymax)))
            x = int(x0 + idx * step + (step - bw) * 0.5)
            cv2.rectangle(img, (x, y0 + h - bh), (x + bw, y0 + h), (50, 120, 210), -1)
            cv2.putText(img, f"{val:.3g}", (x, y0 + h - bh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, label, (x, y0 + h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        path = visual_root / filename
        cv2.imwrite(str(path), img)
        rows.append({"path": _rel(path), "kind": "bar_plot", "field": field, "sha256": _sha256(path), "bytes": path.stat().st_size})
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    scenes = _parse_csv_list(args.scenes)
    scene_set = set(scenes)
    output_root = _rooted(args.output_root)
    visual_root = _rooted(args.visual_root)
    output_root.mkdir(parents=True, exist_ok=True)
    visual_root.mkdir(parents=True, exist_ok=True)
    candidate_root = _rooted(args.candidate_root)
    key_atom_root = _rooted(args.key_atom_root)
    atom_summary = _load_json(_rooted(args.atom_root) / "atom_summary.json")
    atom_metrics = atom_summary.get("key_metrics") if isinstance(atom_summary.get("key_metrics"), dict) else atom_summary
    diagnostic_gt_mean = _float(atom_metrics.get("diagnostic_GT_count_per_chunk_mean"), 21.515923566878982)
    max_masks = int(args.max_masks_per_chunk or max(1, math.floor(3.0 * diagnostic_gt_mean)))
    min_masks = int(args.min_masks_per_chunk or math.ceil(0.5 * diagnostic_gt_mean))
    candidates_by_chunk = _load_candidates(candidate_root / "candidate_mask_rows.csv", scene_set)
    key_atoms_by_chunk = _load_key_atoms(key_atom_root / "key_atom_rows.csv", scene_set, args.key_atom_variant)
    pipeline_roots = _load_pipeline_roots(_rooted(args.witness_summary), scenes)
    missing_rows: list[dict[str, Any]] = []
    for path in [candidate_root / "candidate_mask_rows.csv", key_atom_root / "key_atom_rows.csv"]:
        if not path.exists():
            missing_rows.append({"path": _rel(path), "missing": True})
    if set(pipeline_roots) != scene_set:
        missing_rows.append({"missing": "pipeline_roots", "available_scenes": sorted(pipeline_roots), "requested_scenes": scenes})

    selected_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    processed_chunks = 0
    for scene in scenes:
        pipeline_root = pipeline_roots.get(scene)
        if pipeline_root is None:
            continue
        mask_dir = _mask_dir_from_pipeline(pipeline_root)
        stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
        all_stride_frames = stream.frame_ids(stride=int(args.stride), max_frames=None)
        for chunk in _chunk_rows(pipeline_root, scene):
            chunk_id = str(chunk.get("chunk_id"))
            if chunk_id not in candidates_by_chunk and chunk_id not in key_atoms_by_chunk:
                continue
            if int(args.max_chunks) > 0 and processed_chunks >= int(args.max_chunks):
                break
            processed_chunks += 1
            print(f"[v71-setcover] chunk {processed_chunks}: {chunk_id}", file=sys.stderr, flush=True)
            raw_start = int(float(chunk.get("raw_frame_start") or 0))
            raw_end = int(float(chunk.get("raw_frame_end") or raw_start))
            frame_ids = [int(frame) for frame in all_stride_frames if raw_start <= int(frame) <= raw_end]
            if not frame_ids:
                continue
            data = _frame_data(scene, frame_ids, mask_dir)
            key_atoms = key_atoms_by_chunk.get(chunk_id, [])
            candidates = _prefilter_candidates(candidates_by_chunk.get(chunk_id, []), key_atoms, int(args.max_candidates_per_chunk))
            pairs = {(cand.frame_id, cand.mask_id) for cand in candidates}
            diagnostic_stats = _diagnostic_mask_stats(data, pairs)
            coverage, _coverage_kind = _build_coverage(data, candidates, key_atoms, min_iou=float(args.min_cover_iou))
            for config in CONFIGS:
                selected, cover_diag = _select_set_cover(
                    config=config,
                    candidates=candidates,
                    key_atoms=key_atoms,
                    coverage=coverage,
                    diagnostic_stats=diagnostic_stats,
                    target_coverage=float(args.target_coverage),
                    min_masks=min_masks,
                    max_masks=max_masks,
                    min_gain=float(args.min_gain),
                )
                selected_rows.extend(_selected_mask_rows(config=config, selected=selected, diagnostic_stats=diagnostic_stats))
                metric_rows.append(
                    _chunk_metric_row(
                        config=config,
                        scene=scene,
                        chunk_id=chunk_id,
                        frame_ids=frame_ids,
                        chunk=chunk,
                        frame_data=data,
                        selected=selected,
                        cover_diag=cover_diag,
                        diagnostic_stats=diagnostic_stats,
                        pipeline_root=pipeline_root,
                    )
                )

    variant_summary_rows = [_summarize_setcover(metric_rows, variant) for variant in sorted({row["variant"] for row in metric_rows})]
    lookup = _variant_by_name(variant_summary_rows)
    non_oracle_rows = [row for row in variant_summary_rows if not bool(row.get("uses_gt_for_prediction"))]
    method_rows = [
        row
        for row in non_oracle_rows
        if str(row.get("variant"))
        in {
            "SC5_geo_semantic_cover",
            "SC6_geo_semantic_specificity",
            "SC7_geo_semantic_balanced",
            "SC8_geo_semantic_reliability_weighted",
            "SC10_clean_mid_area_proto_margin_repair",
            "SC11_clean_mid_area_objectness_rank_repair",
        }
    ]
    best_method = max(method_rows, key=lambda row: float(row.get("representative_oracle_SF50") or 0.0), default={})
    d4rt_only = max([lookup.get("SC1_D4RT_atom_cover", {}), lookup.get("SC2_D4RT_atom_cover_area_penalty", {})], key=lambda row: float(row.get("representative_oracle_SF50") or 0.0), default={})
    sem_only = max([lookup.get("SC3_semantic_atom_cover", {}), lookup.get("SC4_semantic_compact_cover", {})], key=lambda row: float(row.get("representative_oracle_SF50") or 0.0), default={})
    fusion_rows = [
        lookup.get(name, {})
        for name in [
            "SC5_geo_semantic_cover",
            "SC6_geo_semantic_specificity",
            "SC7_geo_semantic_balanced",
            "SC8_geo_semantic_reliability_weighted",
            "SC10_clean_mid_area_proto_margin_repair",
            "SC11_clean_mid_area_objectness_rank_repair",
        ]
    ]
    best_fusion = max(fusion_rows, key=lambda row: float(row.get("representative_oracle_SF50") or 0.0), default={})

    def val(row: dict[str, Any], key: str) -> float | None:
        value = row.get(key)
        return None if value in (None, "") else float(value)

    best_sf50 = val(best_method, "representative_oracle_SF50")
    best_gt = val(best_method, "representative_GT_best_IoU_mean")
    best_cov = val(best_method, "covered_key_atom_weight_ratio")
    best_d4rt_cov = val(best_method, "covered_D4RT_atom_weight_ratio")
    best_sem_cov = val(best_method, "covered_semantic_atom_weight_ratio")
    best_count = val(best_method, "selected_mask_count_per_chunk_mean")
    best_broad = val(best_method, "broad_large_selected_rate")
    best_under = val(best_method, "underseg_proxy_selected_rate")
    gate = {
        "all_inputs_present": not any(row.get("missing") for row in missing_rows),
        "best_method_variant": best_method.get("variant"),
        "best_method_variant_representative_oracle_SF50_ge_0p30": best_sf50 is not None and best_sf50 >= 0.30,
        "best_method_variant_representative_GT_best_IoU_mean_ge_0p25": best_gt is not None and best_gt >= 0.25,
        "covered_key_atom_weight_ratio_ge_0p75": best_cov is not None and best_cov >= 0.75,
        "covered_D4RT_atom_weight_ratio_ge_0p65": best_d4rt_cov is not None and best_d4rt_cov >= 0.65,
        "covered_semantic_atom_weight_ratio_ge_0p65": best_sem_cov is not None and best_sem_cov >= 0.65,
        "selected_mask_count_per_chunk_mean_ge_0p5_gt": best_count is not None and best_count >= 0.5 * diagnostic_gt_mean,
        "selected_mask_count_per_chunk_mean_le_3p0_gt": best_count is not None and best_count <= 3.0 * diagnostic_gt_mean,
        "broad_large_selected_rate_le_0p30": best_broad is not None and best_broad <= 0.30,
        "underseg_proxy_selected_rate_le_0p35": best_under is not None and best_under <= 0.35,
        "non_oracle_uses_gt_for_prediction_false": not any(bool(row.get("uses_gt_for_prediction")) for row in non_oracle_rows),
    }
    first_stage_pass = all(
        gate[key]
        for key in [
            "best_method_variant_representative_oracle_SF50_ge_0p30",
            "best_method_variant_representative_GT_best_IoU_mean_ge_0p25",
            "covered_key_atom_weight_ratio_ge_0p75",
            "covered_D4RT_atom_weight_ratio_ge_0p65",
            "covered_semantic_atom_weight_ratio_ge_0p65",
            "selected_mask_count_per_chunk_mean_ge_0p5_gt",
            "selected_mask_count_per_chunk_mean_le_3p0_gt",
            "broad_large_selected_rate_le_0p30",
            "underseg_proxy_selected_rate_le_0p35",
            "non_oracle_uses_gt_for_prediction_false",
        ]
    )
    fusion_sf = val(best_fusion, "representative_oracle_SF50")
    d4rt_sf = val(d4rt_only, "representative_oracle_SF50")
    sem_sf = val(sem_only, "representative_oracle_SF50")
    fusion_gt = val(best_fusion, "representative_GT_best_IoU_mean")
    d4rt_gt = val(d4rt_only, "representative_GT_best_IoU_mean")
    sem_gt = val(sem_only, "representative_GT_best_IoU_mean")
    fusion_broad = val(best_fusion, "broad_large_selected_rate")
    d4rt_broad = val(d4rt_only, "broad_large_selected_rate")
    sem_broad = val(sem_only, "broad_large_selected_rate")
    gate.update(
        {
            "best_fusion_variant": best_fusion.get("variant"),
            "fusion_SF50_ge_best_single_plus_0p05": fusion_sf is not None and d4rt_sf is not None and sem_sf is not None and fusion_sf >= max(d4rt_sf, sem_sf) + 0.05,
            "fusion_GT_best_IoU_ge_best_single_plus_0p05": fusion_gt is not None and d4rt_gt is not None and sem_gt is not None and fusion_gt >= max(d4rt_gt, sem_gt) + 0.05,
            "fusion_broad_large_le_best_single_plus_0p05": fusion_broad is not None and d4rt_broad is not None and sem_broad is not None and fusion_broad <= min(d4rt_broad, sem_broad) + 0.05,
            "first_stage_representative_selection_pass": bool(first_stage_pass),
        }
    )
    decision = "PASS_V71_REPRESENTATIVE_SETCOVER" if first_stage_pass else "NO_GO_PHASE5_REPRESENTATIVE_SETCOVER"
    summary = {
        "decision": decision,
        "gate": gate,
        "plan_phase": "Phase5_Geo_Semantic_Representative_Mask_Set_Cover",
        "config": {
            "key_atom_variant": args.key_atom_variant,
            "target_coverage": float(args.target_coverage),
            "min_cover_iou": float(args.min_cover_iou),
            "min_masks_per_chunk": min_masks,
            "max_masks_per_chunk": max_masks,
            "max_candidates_per_chunk_after_prefilter": int(args.max_candidates_per_chunk),
            "candidate_prefilter_policy": "keep exact key-atom mask observations first, then frame/prototype low-risk semantic candidates ranked by non-GT metadata",
            "diagnostic_GT_count_per_chunk_mean": diagnostic_gt_mean,
        },
        "best_method": best_method,
        "best_fusion": best_fusion,
        "best_d4rt_only": d4rt_only,
        "best_semantic_only": sem_only,
        "variant_count": len(variant_summary_rows),
        "processed_chunk_count": processed_chunks,
        "selected_mask_row_count": len(selected_rows),
        "metric_row_count": len(metric_rows),
        "notes": [
            "Non-oracle set-cover variants use only key atom/candidate metadata and mask overlap; GT diagnostics are computed after selection.",
            "SC9 is oracle diagnostic and forbidden for method tables.",
            "Coverage Cov(m,a) uses same-frame mask IoU with exact same mask id equal to 1.0.",
        ],
    }
    summary_path = output_root / "setcover_summary.json"
    selected_path = output_root / "selected_mask_rows.csv"
    variant_path = output_root / "setcover_variant_summary_rows.csv"
    metric_path = output_root / "setcover_metric_rows.csv"
    missing_path = output_root / "missing_input_rows.csv"
    vis_path = output_root / "visualization_rows.csv"
    _write_json(summary_path, summary)
    _write_csv(selected_path, selected_rows)
    _write_csv(variant_path, variant_summary_rows)
    _write_csv(metric_path, metric_rows)
    _write_csv(missing_path, missing_rows)
    vis_rows = _write_visuals(visual_root, variant_summary_rows)
    _write_csv(vis_path, vis_rows)
    sha_rows = []
    for path in [summary_path, selected_path, variant_path, metric_path, missing_path, vis_path] + [visual_root / Path(row["path"]).name for row in vis_rows]:
        if path.exists():
            sha_rows.append({"path": _rel(path), "sha256": _sha256(path), "bytes": path.stat().st_size})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps({"decision": decision, "output_root": _rel(output_root), "visual_root": _rel(visual_root), "gate": gate}, indent=2, sort_keys=True))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stream4D v71 representative mask weighted set cover.")
    parser.add_argument("--candidate-root", default="outputs/audit/v71_candidate_bank")
    parser.add_argument("--key-atom-root", default="outputs/audit/v71_key_atoms")
    parser.add_argument("--atom-root", default="outputs/audit/v71_d4rt_atoms")
    parser.add_argument("--witness-summary", default="outputs/audit/v70_carrier_witness/witness_summary.json")
    parser.add_argument("--output-root", default="outputs/audit/v71_representative_setcover")
    parser.add_argument("--visual-root", default="outputs/audit/v71_visualizations/representative_setcover")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--key-atom-variant", default="K8_reliability_weighted_geo_semantic_FPS")
    parser.add_argument("--target-coverage", type=float, default=0.75)
    parser.add_argument("--min-cover-iou", type=float, default=0.05)
    parser.add_argument("--min-gain", type=float, default=0.0)
    parser.add_argument("--min-masks-per-chunk", type=int, default=0)
    parser.add_argument("--max-masks-per-chunk", type=int, default=0)
    parser.add_argument("--max-candidates-per-chunk", type=int, default=240)
    parser.add_argument("--max-chunks", type=int, default=0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
