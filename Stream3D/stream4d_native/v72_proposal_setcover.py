from __future__ import annotations

import argparse
import csv
import hashlib
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
from stream4d_native.v72_dense_token_proposals import _load_gt_2d, _load_pipeline_roots, _resize_binary, _resize_label  # noqa: E402
from tools.run_v65_scene_multiview_ap import SparseSceneIoU, _summarize_iou  # noqa: E402
from tools.run_v66_local_chunk_eval import _chunk_rows, _score_free  # noqa: E402
from tools.run_v66_scene_mv_ap_probe5 import DEFAULT_SCENES, _mask_dir_from_pipeline, _parse_csv_list  # noqa: E402


@dataclass(frozen=True)
class ProposalUnit:
    unit_key: str
    proposal_id: str
    scene_id: str
    chunk_id: str
    frame_id: int
    source_mask_id: int
    source_obs_id: str
    source_variant: str
    token_grid_shape: str
    token_coords: str
    area_ratio: float
    semantic_entropy: float
    source_semantic_entropy: float
    semantic_score: float
    semantic_margin: float
    background_proxy_score: float
    broad_risk: bool
    underseg_risk: bool
    majority_gt: int
    majority_iou: float
    objectness_score: float
    objectness_rank: int
    d4rt_score: float
    d4rt_atom_count: int
    d4rt_reliability: float
    semantic_prototype_id: str


@dataclass(frozen=True)
class KeyAtom:
    key_id: str
    chunk_id: str
    frame_id: int
    mask_id: int
    obs_id: str
    weight: float
    semantic_weight: float
    d4rt_weight: float
    semantic_prototype_id: str
    d4rt_uv_x: float | None = None
    d4rt_uv_y: float | None = None


@dataclass(frozen=True)
class CoverConfig:
    variant: str
    semantic_weight: float
    d4rt_weight: float
    objectness_weight: float
    d4rt_score_weight: float
    area_penalty: float
    entropy_penalty: float
    risk_penalty: float
    frame_bonus: float
    source_conflict_penalty: float
    risk_cap: float | None = None
    oracle: bool = False


CONFIGS = [
    CoverConfig("PSC0_v71_single_mask_baseline", 0.35, 0.35, 0.20, 0.00, 0.05, 0.00, 0.25, 0.02, 1.00),
    CoverConfig("PSC1_semantic_proposal_cover", 1.00, 0.00, 0.30, 0.00, 0.03, 0.02, 0.35, 0.03, 1.00),
    CoverConfig("PSC2_D4RT_verified_proposal_cover", 0.15, 1.00, 0.20, 0.45, 0.03, 0.00, 0.30, 0.02, 1.00),
    CoverConfig("PSC3_temporal_expanded_proposal_cover", 0.65, 0.45, 0.30, 0.20, 0.02, 0.00, 0.30, 0.10, 1.00),
    CoverConfig("PSC4_fusion_MDL_cover", 0.85, 0.65, 0.50, 0.25, 0.12, 0.02, 0.70, 0.04, 1.25),
    CoverConfig("PSC5_background_suppressed_fusion", 0.85, 0.65, 0.55, 0.25, 0.10, 0.03, 1.25, 0.04, 1.50, risk_cap=0.35),
    CoverConfig("PSC7_semantic_prototype_temporal_group_repair", 0.95, 0.35, 0.60, 0.15, 0.08, 0.02, 0.55, 0.25, 1.00),
    CoverConfig("PSC8_D4RT_supported_temporal_group_repair", 0.70, 0.85, 0.55, 0.35, 0.08, 0.02, 0.75, 0.25, 1.00, risk_cap=0.35),
    CoverConfig("PSC9_decomposed_preferred_temporal_group_repair", 0.95, 0.45, 0.75, 0.20, 0.06, 0.03, 0.65, 0.25, 1.00, risk_cap=0.35),
    CoverConfig("PSC10_clean_decomposed_D4RT_group_repair", 0.75, 0.95, 0.70, 0.45, 0.06, 0.03, 0.85, 0.25, 1.00, risk_cap=0.35),
    CoverConfig("PSC6_oracle_proposal_selection_diagnostic", 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, oracle=True),
]


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


def _float(value: Any, default: float = 0.0) -> float:
    if value is None or str(value).strip() == "":
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return float(out) if math.isfinite(out) else None


def _int(value: Any, default: int = 0) -> int:
    if value is None or str(value).strip() == "":
        return int(default)
    try:
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


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(float(value) for value in values)
    idx = int(round((len(vals) - 1) * float(q)))
    return float(vals[max(0, min(len(vals) - 1, idx))])


def _unit_key(chunk_id: str, proposal_id: str) -> str:
    return f"{chunk_id}|{proposal_id}"


def _load_rank_scores(path: Path, objectness_variant: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("objectness_variant") or "") != objectness_variant:
                continue
            key = _unit_key(str(row.get("chunk_id") or ""), str(row.get("proposal_id") or ""))
            out[key] = {"score": _float(row.get("objectness_score"), 0.0), "rank": _float(row.get("rank"), 0.0)}
    return out


def _load_d4rt_scores(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = _unit_key(str(row.get("chunk_id") or ""), str(row.get("proposal_id") or ""))
            out[key] = {
                "d4rt_score": _float(row.get("proposal_D4RT_score"), 0.0),
                "d4rt_atom_count": _float(row.get("proposal_D4RT_atom_count"), 0.0),
                "d4rt_reliability": _float(row.get("proposal_D4RT_reliability_mean"), 0.0),
            }
    return out


def _load_source_prototypes(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            obs_id = str(row.get("mask_observation_id") or "")
            if obs_id:
                out[obs_id] = str(row.get("semantic_prototype_id") or "")
    return out


def _load_proposals(
    path: Path,
    rank_rows: Path,
    verification_rows: Path,
    candidate_rows: Path,
    target_dense_variant: str,
    objectness_variant: str,
) -> list[ProposalUnit]:
    rank = _load_rank_scores(rank_rows, objectness_variant)
    d4rt = _load_d4rt_scores(verification_rows)
    source_prototypes = _load_source_prototypes(candidate_rows)
    allowed = {"SP0_existing_masks_baseline", str(target_dense_variant)}
    proposals: list[ProposalUnit] = []
    max_score_by_chunk: dict[str, float] = defaultdict(float)
    raw_rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("variant") or "") not in allowed:
                continue
            chunk_id = str(row.get("chunk_id") or "")
            proposal_id = str(row.get("proposal_id") or "")
            key = _unit_key(chunk_id, proposal_id)
            row_rank = rank.get(key, {})
            score = _float(row_rank.get("score"), _float(row.get("proposal_compactness_score"), 0.0))
            row["_unit_key"] = key
            row["_objectness_score_raw"] = score
            row["_objectness_rank"] = _float(row_rank.get("rank"), 999999.0)
            raw_rows.append(row)
            max_score_by_chunk[chunk_id] = max(max_score_by_chunk[chunk_id], score)
    for row in raw_rows:
        chunk_id = str(row.get("chunk_id") or "")
        proposal_id = str(row.get("proposal_id") or "")
        key = str(row["_unit_key"])
        raw_score = float(row["_objectness_score_raw"])
        norm_score = raw_score / max(1e-9, max_score_by_chunk.get(chunk_id, raw_score, ))
        aux = d4rt.get(key, {})
        source_mask_id = _int(row.get("source_mask_id"), -1)
        proposals.append(
            ProposalUnit(
                unit_key=key,
                proposal_id=proposal_id,
                scene_id=str(row.get("scene_id") or ""),
                chunk_id=chunk_id,
                frame_id=_int(row.get("frame_id"), -1),
                source_mask_id=source_mask_id,
                source_obs_id=str(row.get("source_mask_ids") or ""),
                source_variant=str(row.get("variant") or ""),
                token_grid_shape=str(row.get("proposal_token_grid_shape") or ""),
                token_coords=str(row.get("proposal_token_coords") or ""),
                area_ratio=_float(row.get("proposal_area_ratio"), 0.0),
                semantic_entropy=_float(row.get("semantic_entropy"), 1.0),
                source_semantic_entropy=_float(row.get("source_semantic_entropy"), _float(row.get("semantic_entropy"), 1.0)),
                semantic_score=_float(row.get("proposal_compactness_score"), 0.0),
                semantic_margin=_float(row.get("semantic_prototype_margin"), 0.0),
                background_proxy_score=_float(row.get("proposal_background_proxy_score"), 0.0),
                broad_risk=_bool(row.get("source_broad_large_risk")),
                underseg_risk=_bool(row.get("source_underseg_proxy")),
                majority_gt=_int(row.get("majority_gt_id_diagnostic"), 0),
                majority_iou=_float(row.get("majority_iou_diagnostic"), 0.0),
                objectness_score=float(norm_score),
                objectness_rank=int(float(row["_objectness_rank"])),
                d4rt_score=_float(aux.get("d4rt_score"), 0.0),
                d4rt_atom_count=int(_float(aux.get("d4rt_atom_count"), 0.0)),
                d4rt_reliability=_float(aux.get("d4rt_reliability"), 0.0),
                semantic_prototype_id=source_prototypes.get(str(row.get("source_mask_ids") or ""), ""),
            )
        )
    return proposals


def _load_key_atoms(path: Path, variant: str, chunks: set[str]) -> list[KeyAtom]:
    atoms: list[KeyAtom] = []
    if not path.exists():
        return atoms
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("key_atom_variant") or "") != variant:
                continue
            chunk_id = str(row.get("chunk_id") or "")
            if chunks and chunk_id not in chunks:
                continue
            weight = max(1e-9, _float(row.get("selection_weight"), 1.0))
            d4rt_available = _bool(row.get("D4RT_reliability_available"))
            has_semantic = bool(str(row.get("semantic_prototype_id") or "")) or _bool(row.get("semantic_only_repair_atom"))
            atoms.append(
                KeyAtom(
                    key_id=str(row.get("key_atom_id") or ""),
                    chunk_id=chunk_id,
                    frame_id=_int(row.get("frame_id"), -1),
                    mask_id=_int(row.get("mask_id"), -1),
                    obs_id=str(row.get("mask_observation_id") or ""),
                    weight=weight,
                    semantic_weight=weight if has_semantic else 0.0,
                    d4rt_weight=weight if d4rt_available else 0.0,
                    semantic_prototype_id=str(row.get("semantic_prototype_id") or ""),
                    d4rt_uv_x=_optional_float(row.get("D4RT_uv_x")),
                    d4rt_uv_y=_optional_float(row.get("D4RT_uv_y")),
                )
            )
    return atoms


def _atom_inside_proposal_uv(prop: ProposalUnit, atom: KeyAtom) -> bool:
    if atom.d4rt_uv_x is None or atom.d4rt_uv_y is None:
        return True
    if prop.source_variant == "SP0_existing_masks_baseline":
        return True
    token_mask = _parse_token_mask(prop.token_grid_shape, prop.token_coords)
    if token_mask.size <= 1 or not np.any(token_mask):
        return False
    h, w = token_mask.shape
    x = int(np.clip(round(float(atom.d4rt_uv_x) * float(max(0, w - 1))), 0, max(0, w - 1)))
    y = int(np.clip(round(float(atom.d4rt_uv_y) * float(max(0, h - 1))), 0, max(0, h - 1)))
    return bool(token_mask[y, x])


def _build_coverage(proposals: list[ProposalUnit], atoms: list[KeyAtom], *, key_atom_uv_coverage: bool) -> dict[str, dict[str, float]]:
    atoms_by_source: dict[tuple[str, int, str], list[KeyAtom]] = defaultdict(list)
    for atom in atoms:
        atoms_by_source[(atom.chunk_id, atom.frame_id, atom.obs_id)].append(atom)
    coverage: dict[str, dict[str, float]] = defaultdict(dict)
    for prop in proposals:
        matched = atoms_by_source.get((prop.chunk_id, prop.frame_id, prop.source_obs_id), [])
        for atom in matched:
            if key_atom_uv_coverage and not _atom_inside_proposal_uv(prop, atom):
                continue
            coverage[prop.unit_key][atom.key_id] = 1.0
    return coverage


def _is_decomposed(prop: ProposalUnit) -> bool:
    return prop.source_variant != "SP0_existing_masks_baseline"


def _proposal_unresolved_risk(prop: ProposalUnit) -> bool:
    if not _is_decomposed(prop):
        return bool(prop.broad_risk or prop.underseg_risk or prop.background_proxy_score >= 0.75 or prop.area_ratio >= 0.30)
    entropy_drop = float(prop.source_semantic_entropy) - float(prop.semantic_entropy)
    if prop.area_ratio >= 0.30 or prop.background_proxy_score >= 0.75:
        return True
    if (prop.broad_risk or prop.underseg_risk) and entropy_drop < 0.10:
        return True
    if prop.underseg_risk and prop.area_ratio >= 0.18:
        return True
    return False


def _selection_conflict_key(prop: ProposalUnit) -> tuple[int, str]:
    if _is_decomposed(prop):
        return (prop.frame_id, prop.proposal_id)
    return (prop.frame_id, prop.source_obs_id)


def _select_oracle(subset: list[ProposalUnit], max_count: int) -> list[tuple[ProposalUnit, dict[str, Any]]]:
    by_gt: dict[int, ProposalUnit] = {}
    for prop in subset:
        if prop.majority_gt <= 0:
            continue
        best = by_gt.get(prop.majority_gt)
        if best is None or (prop.majority_iou, prop.objectness_score) > (best.majority_iou, best.objectness_score):
            by_gt[prop.majority_gt] = prop
    selected = sorted(by_gt.values(), key=lambda item: (item.majority_iou, item.objectness_score), reverse=True)
    for prop in sorted(subset, key=lambda item: (item.majority_iou, item.objectness_score), reverse=True):
        if len(selected) >= max_count:
            break
        if prop in selected:
            continue
        selected.append(prop)
    return [(prop, {"score_total": prop.majority_iou, "new_total_weight": 0.0, "new_semantic_weight": 0.0, "new_d4rt_weight": 0.0}) for prop in selected[:max_count]]


def _select_greedy(
    *,
    config: CoverConfig,
    subset: list[ProposalUnit],
    atoms: list[KeyAtom],
    coverage: dict[str, dict[str, float]],
    max_count: int,
    min_gain: float,
) -> tuple[list[tuple[ProposalUnit, dict[str, Any]]], dict[str, Any]]:
    if config.variant == "PSC0_v71_single_mask_baseline":
        subset = [prop for prop in subset if prop.source_variant == "SP0_existing_masks_baseline"]
    if config.oracle:
        selected = _select_oracle(subset, max_count)
        return selected, {"stop_reason": "oracle_diagnostic_gt_iou", "risk_cap_applied": False}
    atom_by_id = {atom.key_id: atom for atom in atoms}
    covered: set[str] = set()
    selected: list[tuple[ProposalUnit, dict[str, Any]]] = []
    selected_sources: set[tuple[int, str]] = set()
    selected_frames: Counter[int] = Counter()
    selected_risky = 0
    stop_reason = "max_count"
    for rank in range(int(max_count)):
        best: tuple[float, ProposalUnit, dict[str, Any]] | None = None
        for prop in subset:
            source_key = _selection_conflict_key(prop)
            if source_key in selected_sources:
                continue
            is_risky = _proposal_unresolved_risk(prop)
            if config.risk_cap is not None:
                next_count = len(selected) + 1
                next_risky = selected_risky + (1 if is_risky else 0)
                if next_risky / max(1, next_count) > float(config.risk_cap):
                    continue
            cov = coverage.get(prop.unit_key, {})
            new_total = 0.0
            new_sem = 0.0
            new_d4rt = 0.0
            new_count = 0
            for key_id, value in cov.items():
                if key_id in covered or value <= 0.0:
                    continue
                atom = atom_by_id.get(key_id)
                if atom is None:
                    continue
                new_total += atom.weight
                new_sem += atom.semantic_weight
                new_d4rt += atom.d4rt_weight
                new_count += 1
            score = 0.0
            score += config.semantic_weight * new_sem
            score += config.d4rt_weight * new_d4rt
            score += config.objectness_weight * prop.objectness_score
            score += config.d4rt_score_weight * prop.d4rt_score
            if selected_frames[prop.frame_id] == 0:
                score += config.frame_bonus
            score -= config.area_penalty * prop.area_ratio
            score -= config.entropy_penalty * prop.semantic_entropy
            if is_risky:
                score -= config.risk_penalty
            if source_key in selected_sources:
                score -= config.source_conflict_penalty
            detail = {
                "score_total": float(score),
                "new_total_weight": float(new_total),
                "new_semantic_weight": float(new_sem),
                "new_d4rt_weight": float(new_d4rt),
                "new_atom_count": int(new_count),
            }
            if best is None or score > best[0]:
                best = (float(score), prop, detail)
        if best is None:
            stop_reason = "no_candidate_after_constraints"
            break
        best_score, prop, detail = best
        if rank > 0 and best_score < float(min_gain):
            stop_reason = "marginal_gain_below_min"
            break
        selected.append((prop, detail))
        selected_sources.add(_selection_conflict_key(prop))
        selected_frames[prop.frame_id] += 1
        if _proposal_unresolved_risk(prop):
            selected_risky += 1
        for key_id in coverage.get(prop.unit_key, {}):
            covered.add(key_id)
    return selected, {"stop_reason": stop_reason, "risk_cap_applied": config.risk_cap is not None}


def _proposal_group_key(prop: ProposalUnit) -> str:
    proto = prop.semantic_prototype_id.strip()
    if proto:
        return f"proto:{proto}"
    return f"source:{prop.source_obs_id}"


def _member_score(prop: ProposalUnit, variant: str) -> tuple[float, float, float, float]:
    if variant == "PSC7_semantic_prototype_temporal_group_repair":
        score = float(prop.objectness_score) + 0.35 * float(prop.d4rt_score) - 0.05 * float(prop.area_ratio)
        return (float(score), float(prop.objectness_score), float(prop.d4rt_score), -float(prop.area_ratio))
    if variant == "PSC8_D4RT_supported_temporal_group_repair":
        unresolved_penalty = 0.45 if _proposal_unresolved_risk(prop) else 0.0
        score = (
            float(prop.objectness_score)
            + 0.55 * float(prop.d4rt_score)
            - unresolved_penalty
            - 0.08 * float(prop.background_proxy_score)
            - 0.05 * float(prop.area_ratio)
        )
        return (float(score), float(prop.objectness_score), float(prop.d4rt_score), -float(prop.area_ratio))
    dense_bonus = 0.0
    if "decomposed" in variant:
        dense_bonus = 0.50 if _is_decomposed(prop) else -0.25
    unresolved_penalty = 0.75 if _proposal_unresolved_risk(prop) else 0.0
    source_risk_penalty = 0.20 if (prop.broad_risk or prop.underseg_risk) else 0.0
    score = (
        float(prop.objectness_score)
        + 0.35 * float(prop.d4rt_score)
        + dense_bonus
        - unresolved_penalty
        - source_risk_penalty
        - 0.10 * float(prop.background_proxy_score)
        - 0.05 * float(prop.area_ratio)
        - 0.03 * float(prop.semantic_entropy)
    )
    return (float(score), float(prop.objectness_score), float(prop.d4rt_score), -float(prop.area_ratio))


def _build_temporal_groups(subset: list[ProposalUnit], *, max_members: int, variant: str) -> list[list[ProposalUnit]]:
    by_group_frame: dict[str, dict[int, list[ProposalUnit]]] = defaultdict(lambda: defaultdict(list))
    for prop in subset:
        if "decomposed" in variant and not _is_decomposed(prop) and _proposal_unresolved_risk(prop):
            continue
        by_group_frame[_proposal_group_key(prop)][prop.frame_id].append(prop)
    groups: list[list[ProposalUnit]] = []
    for _group_id, by_frame in by_group_frame.items():
        members: list[ProposalUnit] = []
        for frame_id in sorted(by_frame):
            best = max(
                by_frame[frame_id],
                key=lambda item: _member_score(item, variant),
            )
            members.append(best)
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda item: (item.objectness_score, item.d4rt_score), reverse=True)[: int(max_members)]
        groups.append(sorted(members, key=lambda item: item.frame_id))
    return groups


def _select_group_greedy(
    *,
    config: CoverConfig,
    subset: list[ProposalUnit],
    atoms: list[KeyAtom],
    coverage: dict[str, dict[str, float]],
    max_count: int,
    min_gain: float,
    max_members_per_group: int,
) -> tuple[list[tuple[list[ProposalUnit], dict[str, Any]]], dict[str, Any]]:
    atom_by_id = {atom.key_id: atom for atom in atoms}
    groups = _build_temporal_groups(subset, max_members=max_members_per_group, variant=config.variant)
    covered: set[str] = set()
    selected: list[tuple[list[ProposalUnit], dict[str, Any]]] = []
    used_group_keys: set[tuple[str, ...]] = set()
    selected_risky = 0
    stop_reason = "max_count"
    for rank in range(int(max_count)):
        best: tuple[float, list[ProposalUnit], dict[str, Any]] | None = None
        for group in groups:
            group_keys = tuple(sorted({_proposal_group_key(prop) for prop in group}))
            if group_keys in used_group_keys:
                continue
            group_risky = sum(1 for prop in group if _proposal_unresolved_risk(prop)) / max(1, len(group))
            if config.risk_cap is not None and group_risky > float(config.risk_cap):
                continue
            new_total = 0.0
            new_sem = 0.0
            new_d4rt = 0.0
            new_count = 0
            for prop in group:
                for key_id, value in coverage.get(prop.unit_key, {}).items():
                    if key_id in covered or value <= 0.0:
                        continue
                    atom = atom_by_id.get(key_id)
                    if atom is None:
                        continue
                    new_total += atom.weight
                    new_sem += atom.semantic_weight
                    new_d4rt += atom.d4rt_weight
                    new_count += 1
            objectness = _mean([prop.objectness_score for prop in group]) or 0.0
            d4rt_score = _mean([prop.d4rt_score for prop in group]) or 0.0
            area = _mean([prop.area_ratio for prop in group]) or 0.0
            entropy = _mean([prop.semantic_entropy for prop in group]) or 0.0
            frame_span = len({prop.frame_id for prop in group})
            score = 0.0
            score += config.semantic_weight * new_sem
            score += config.d4rt_weight * new_d4rt
            score += config.objectness_weight * objectness
            score += config.d4rt_score_weight * d4rt_score
            score += config.frame_bonus * math.log1p(frame_span)
            score -= config.area_penalty * area
            score -= config.entropy_penalty * entropy
            score -= config.risk_penalty * group_risky
            detail = {
                "score_total": float(score),
                "new_total_weight": float(new_total),
                "new_semantic_weight": float(new_sem),
                "new_d4rt_weight": float(new_d4rt),
                "new_atom_count": int(new_count),
                "group_frame_count": int(frame_span),
                "group_member_count": int(len(group)),
                "group_risk_rate": float(group_risky),
                "group_objectness_mean": float(objectness),
            }
            if best is None or score > best[0]:
                best = (float(score), group, detail)
        if best is None:
            stop_reason = "no_group_after_constraints"
            break
        best_score, group, detail = best
        if rank > 0 and best_score < float(min_gain):
            stop_reason = "marginal_group_gain_below_min"
            break
        selected.append((group, detail))
        used_group_keys.add(tuple(sorted({_proposal_group_key(prop) for prop in group})))
        if float(detail["group_risk_rate"]) > float(config.risk_cap or 1.0):
            selected_risky += 1
        for prop in group:
            covered.update(coverage.get(prop.unit_key, {}).keys())
    return selected, {
        "stop_reason": stop_reason,
        "risk_cap_applied": config.risk_cap is not None,
        "candidate_group_count": len(groups),
        "max_members_per_group": int(max_members_per_group),
    }


def _parse_token_mask(shape: str, coords: str) -> np.ndarray:
    if "x" not in shape:
        return np.zeros((1, 1), dtype=bool)
    h_text, w_text = shape.split("x", 1)
    h, w = int(float(h_text)), int(float(w_text))
    mask = np.zeros((h, w), dtype=bool)
    for item in str(coords or "").split(";"):
        if not item or ":" not in item:
            continue
        y_text, x_text = item.split(":", 1)
        y, x = int(float(y_text)), int(float(x_text))
        if 0 <= y < h and 0 <= x < w:
            mask[y, x] = True
    return mask


def _materialize_mask(
    prop: ProposalUnit,
    *,
    mask_dir: Path,
    eval_shape_hw: tuple[int, int],
    mask_cache: dict[int, np.ndarray | None],
) -> np.ndarray:
    if prop.source_variant == "SP0_existing_masks_baseline":
        if prop.frame_id not in mask_cache:
            mask_cache[prop.frame_id] = _resize_label(mask_dir / f"{prop.frame_id}.png", eval_shape_hw)
        label = mask_cache.get(prop.frame_id)
        if label is None:
            return np.zeros(eval_shape_hw, dtype=bool)
        return np.asarray(label == int(prop.source_mask_id), dtype=bool)
    token_mask = _parse_token_mask(prop.token_grid_shape, prop.token_coords)
    return _resize_binary(token_mask, eval_shape_hw)


def _evaluate_selected_oracle(
    *,
    scene: str,
    frame_ids: list[int],
    selected: list[ProposalUnit],
    mask_dir: Path,
) -> dict[str, Any]:
    if not frame_ids:
        return {
            "representative_proposal_oracle_SF50_diagnostic": 0.0,
            "representative_proposal_AP50_diagnostic": 0.0,
            "representative_proposal_AP25_diagnostic": 0.0,
            "representative_proposal_GT_best_IoU_mean_diagnostic": 0.0,
            "proposal_pred_best_IoU_median_diagnostic": 0.0,
            "diagnostic_GT_count": 0,
        }
    stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
    eval_shape_cache: dict[int, tuple[int, int]] = {}
    gt_cache: dict[int, np.ndarray] = {}
    mask_cache: dict[int, np.ndarray | None] = {}
    selected_by_frame: dict[int, list[ProposalUnit]] = defaultdict(list)
    for prop in selected:
        selected_by_frame[prop.frame_id].append(prop)
    acc = SparseSceneIoU()
    gt_ids: set[int] = set()
    for frame_id in frame_ids:
        depth_shape = tuple(int(v) for v in stream.load_depth(int(frame_id)).shape)
        eval_shape_cache[int(frame_id)] = depth_shape
        gt = _load_gt_2d(scene, int(frame_id), depth_shape)
        gt_cache[int(frame_id)] = gt
        gt_ids.update(int(label) for label in np.unique(gt) if int(label) > 0)
        pred = np.zeros(depth_shape, dtype=np.int64)
        for prop in sorted(selected_by_frame.get(int(frame_id), []), key=lambda item: item.objectness_score, reverse=True):
            if prop.majority_gt <= 0:
                continue
            binary = _materialize_mask(prop, mask_dir=mask_dir, eval_shape_hw=depth_shape, mask_cache=mask_cache)
            pred[(binary > 0) & (pred == 0)] = int(prop.majority_gt)
        acc.add(pred, gt)
    summary, _iou, _pred_ids, _gt_ids = _summarize_iou(
        accumulator=acc,
        min_pred_pixels=1,
        min_gt_pixels=1,
        score_mode="constant",
        input_scores=None,
    )
    return {
        "representative_proposal_oracle_SF50_diagnostic": _score_free(summary),
        "representative_proposal_AP50_diagnostic": summary.get("ap50"),
        "representative_proposal_AP25_diagnostic": summary.get("ap25"),
        "representative_proposal_AP_diagnostic": summary.get("ap"),
        "representative_proposal_GT_best_IoU_mean_diagnostic": summary.get("gt_best_iou_mean"),
        "proposal_pred_best_IoU_median_diagnostic": summary.get("pred_best_iou_median"),
        "diagnostic_GT_count": int(len(gt_ids)),
    }


def _coverage_summary(selected: list[ProposalUnit], atoms: list[KeyAtom], coverage: dict[str, dict[str, float]]) -> dict[str, Any]:
    atom_by_id = {atom.key_id: atom for atom in atoms}
    total_weight = sum(atom.weight for atom in atoms)
    total_sem = sum(atom.semantic_weight for atom in atoms)
    total_d4rt = sum(atom.d4rt_weight for atom in atoms)
    covered: set[str] = set()
    for prop in selected:
        covered.update(coverage.get(prop.unit_key, {}).keys())
    cov_total = sum(atom_by_id[key].weight for key in covered if key in atom_by_id)
    cov_sem = sum(atom_by_id[key].semantic_weight for key in covered if key in atom_by_id)
    cov_d4rt = sum(atom_by_id[key].d4rt_weight for key in covered if key in atom_by_id)
    return {
        "covered_total_atom_weight_ratio": cov_total / max(1e-9, total_weight),
        "covered_semantic_atom_weight_ratio": cov_sem / max(1e-9, total_sem) if total_sem > 0 else 0.0,
        "covered_D4RT_atom_weight_ratio": cov_d4rt / max(1e-9, total_d4rt) if total_d4rt > 0 else 0.0,
        "covered_atom_count": len(covered),
        "total_atom_count": len(atoms),
        "total_atom_weight": total_weight,
        "total_semantic_atom_weight": total_sem,
        "total_D4RT_atom_weight": total_d4rt,
    }


def _same_frame_conflicts(selected: list[ProposalUnit]) -> tuple[int, float]:
    counts = Counter(_selection_conflict_key(prop) for prop in selected)
    conflict = sum(max(0, count - 1) for count in counts.values())
    return int(conflict), float(conflict / max(1, len(selected)))


def _summarize_variant(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    subset = [row for row in rows if row.get("variant") == variant]
    selected_counts = [_float(row.get("selected_proposal_count"), 0.0) for row in subset]
    gt_counts = [_float(row.get("diagnostic_GT_count"), 0.0) for row in subset]
    incidence_levels = sorted({str(row.get("atom_incidence_level") or "") for row in subset if str(row.get("atom_incidence_level") or "")})
    return {
        "variant": variant,
        "processed_chunk_count": len(subset),
        "proposal_count_per_chunk_mean": _mean([_float(row.get("proposal_count"), 0.0) for row in subset]),
        "selected_proposal_count_per_chunk_mean": _mean(selected_counts),
        "selected_region_count_per_chunk_mean": _mean(selected_counts),
        "covered_semantic_atom_weight_ratio": _mean([_float(row.get("covered_semantic_atom_weight_ratio"), 0.0) for row in subset]),
        "covered_D4RT_atom_weight_ratio": _mean([_float(row.get("covered_D4RT_atom_weight_ratio"), 0.0) for row in subset]),
        "covered_total_atom_weight_ratio": _mean([_float(row.get("covered_total_atom_weight_ratio"), 0.0) for row in subset]),
        "selected_proposal_area_ratio_mean": _mean([_float(row.get("selected_proposal_area_ratio_mean"), 0.0) for row in subset]),
        "selected_proposal_area_ratio_p90": _mean([_float(row.get("selected_proposal_area_ratio_p90"), 0.0) for row in subset]),
        "selected_proposal_temporal_span_mean": _mean([_float(row.get("selected_proposal_temporal_span_mean"), 1.0) for row in subset]),
        "selected_proposal_objectness_mean": _mean([_float(row.get("selected_proposal_objectness_mean"), 0.0) for row in subset]),
        "selected_broad_source_rate": _mean([_float(row.get("selected_broad_source_rate"), 0.0) for row in subset]),
        "unresolved_broad_underseg_rate": _mean([_float(row.get("unresolved_broad_underseg_rate"), 0.0) for row in subset]),
        "same_frame_violation_count": int(sum(_float(row.get("same_frame_violation_count"), 0.0) for row in subset)),
        "duplicate_frame_mask_conflict_rate": _mean([_float(row.get("duplicate_frame_mask_conflict_rate"), 0.0) for row in subset]),
        "representative_proposal_oracle_SF50_diagnostic": _mean([_float(row.get("representative_proposal_oracle_SF50_diagnostic"), 0.0) for row in subset]),
        "representative_proposal_AP50_diagnostic": _mean([_float(row.get("representative_proposal_AP50_diagnostic"), 0.0) for row in subset]),
        "representative_proposal_GT_best_IoU_mean_diagnostic": _mean([_float(row.get("representative_proposal_GT_best_IoU_mean_diagnostic"), 0.0) for row in subset]),
        "proposal_pred_best_IoU_median_diagnostic": _mean([_float(row.get("proposal_pred_best_IoU_median_diagnostic"), 0.0) for row in subset]),
        "diagnostic_GT_count_per_chunk_mean": _mean(gt_counts),
        "selected_count_over_GT_count_mean": _mean([
            _float(row.get("selected_proposal_count"), 0.0) / max(1.0, _float(row.get("diagnostic_GT_count"), 0.0)) for row in subset
        ]),
        "uses_gt_for_prediction": variant == "PSC6_oracle_proposal_selection_diagnostic",
        "uses_gt_for_evaluation": True,
        "diagnostic_only": True,
        "forbidden_for_method_table": variant == "PSC6_oracle_proposal_selection_diagnostic",
        "atom_incidence_level": ";".join(incidence_levels) if incidence_levels else "",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _rooted(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    proposal_rows = _rooted(args.proposal_rows)
    verification_rows = _rooted(args.verification_rows)
    objectness_rank_rows = _rooted(args.objectness_rank_rows)
    key_atom_rows = _rooted(args.key_atom_rows)
    candidate_rows = _rooted(args.candidate_rows)
    witness_summary = _rooted(args.witness_summary)
    missing_rows = []
    for name, path in [
        ("proposal_rows", proposal_rows),
        ("verification_rows", verification_rows),
        ("objectness_rank_rows", objectness_rank_rows),
        ("key_atom_rows", key_atom_rows),
        ("candidate_rows", candidate_rows),
        ("witness_summary", witness_summary),
    ]:
        if not path.exists():
            missing_rows.append({"name": name, "path": _rel(path)})
    if missing_rows:
        _write_csv(output_root / "missing_input_rows.csv", missing_rows)
        summary = {"phase": "v72_phase5_proposal_setcover", "decision": "FAIL_MISSING_INPUTS", "gate": {"pass": False}, "missing_input_count": len(missing_rows)}
        _write_json(output_root / "proposal_setcover_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary

    proposals = _load_proposals(
        proposal_rows,
        objectness_rank_rows,
        verification_rows,
        candidate_rows,
        str(args.target_dense_variant),
        str(args.objectness_variant),
    )
    chunks = {prop.chunk_id for prop in proposals}
    atoms = _load_key_atoms(key_atom_rows, str(args.key_atom_variant), chunks)
    key_atom_uv_coverage = bool(getattr(args, "key_atom_uv_coverage", False))
    coverage = _build_coverage(proposals, atoms, key_atom_uv_coverage=key_atom_uv_coverage)
    atom_incidence_level = "key_atom_uv_inside_proposal_with_no_uv_source_fallback" if key_atom_uv_coverage else "source_mask_level_inherited"
    scenes = sorted({prop.scene_id for prop in proposals}) or _parse_csv_list(args.scenes)
    pipeline_roots = _load_pipeline_roots(witness_summary, scenes)
    proposals_by_chunk: dict[str, list[ProposalUnit]] = defaultdict(list)
    atoms_by_chunk: dict[str, list[KeyAtom]] = defaultdict(list)
    for prop in proposals:
        proposals_by_chunk[prop.chunk_id].append(prop)
    for atom in atoms:
        atoms_by_chunk[atom.chunk_id].append(atom)

    chunk_frame_ids: dict[str, list[int]] = {}
    chunk_scene: dict[str, str] = {}
    chunk_mask_dir: dict[str, Path] = {}
    for scene in scenes:
        pipeline_root = pipeline_roots.get(scene)
        if pipeline_root is None:
            continue
        stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
        all_stride_frames = stream.frame_ids(stride=int(args.stride), max_frames=None)
        mask_dir = _mask_dir_from_pipeline(pipeline_root)
        for chunk in _chunk_rows(pipeline_root, scene):
            chunk_id = str(chunk.get("chunk_id") or "")
            if chunk_id not in chunks:
                continue
            raw_start = int(float(chunk.get("raw_frame_start") or 0))
            raw_end = int(float(chunk.get("raw_frame_end") or raw_start))
            frame_ids = [int(frame) for frame in all_stride_frames if raw_start <= int(frame) <= raw_end]
            chunk_frame_ids[chunk_id] = frame_ids
            chunk_scene[chunk_id] = scene
            chunk_mask_dir[chunk_id] = mask_dir

    metric_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    max_count = int(args.max_proposals_per_chunk)
    for chunk_id in sorted(proposals_by_chunk):
        subset = proposals_by_chunk[chunk_id]
        chunk_atoms = atoms_by_chunk.get(chunk_id, [])
        scene = chunk_scene.get(chunk_id) or (subset[0].scene_id if subset else "")
        frame_ids = chunk_frame_ids.get(chunk_id, sorted({prop.frame_id for prop in subset}))
        mask_dir = chunk_mask_dir.get(chunk_id)
        if mask_dir is None and pipeline_roots.get(scene) is not None:
            mask_dir = _mask_dir_from_pipeline(pipeline_roots[scene])
        if mask_dir is None:
            missing_rows.append({"name": "mask_dir", "chunk_id": chunk_id, "scene_id": scene})
            continue
        for config in CONFIGS:
            group_mode = config.variant in {
                "PSC7_semantic_prototype_temporal_group_repair",
                "PSC8_D4RT_supported_temporal_group_repair",
                "PSC9_decomposed_preferred_temporal_group_repair",
                "PSC10_clean_decomposed_D4RT_group_repair",
            }
            if group_mode:
                selected_groups_with_detail, select_diag = _select_group_greedy(
                    config=config,
                    subset=subset,
                    atoms=chunk_atoms,
                    coverage=coverage,
                    max_count=max_count,
                    min_gain=float(args.min_gain),
                    max_members_per_group=int(args.max_members_per_group),
                )
                selected = [prop for group, _detail in selected_groups_with_detail for prop in group]
                selected_unit_count = len(selected_groups_with_detail)
                selected_region_count = len(selected)
                temporal_span_mean = _mean([float(detail.get("group_frame_count") or 0.0) for _group, detail in selected_groups_with_detail])
            else:
                selected_with_detail, select_diag = _select_greedy(
                    config=config,
                    subset=subset,
                    atoms=chunk_atoms,
                    coverage=coverage,
                    max_count=max_count,
                    min_gain=float(args.min_gain),
                )
                selected = [prop for prop, _detail in selected_with_detail]
                selected_unit_count = len(selected_with_detail)
                selected_region_count = len(selected)
                temporal_span_mean = 1.0 if selected else None
            cov = _coverage_summary(selected, chunk_atoms, coverage)
            eval_diag = _evaluate_selected_oracle(scene=scene, frame_ids=frame_ids, selected=selected, mask_dir=mask_dir)
            conflict_count, conflict_rate = _same_frame_conflicts(selected)
            areas = [prop.area_ratio for prop in selected]
            row = {
                "variant": config.variant,
                "scene_id": scene,
                "chunk_id": chunk_id,
                "proposal_count": len(subset),
                "atom_count": len(chunk_atoms),
                "selected_proposal_count": selected_unit_count,
                "selected_region_count": selected_region_count,
                "selected_proposal_area_ratio_mean": _mean(areas),
                "selected_proposal_area_ratio_p90": _percentile(areas, 0.90),
                "selected_proposal_temporal_span_mean": temporal_span_mean,
                "selected_proposal_objectness_mean": _mean([prop.objectness_score for prop in selected]),
                "selected_broad_source_rate": _mean([1.0 if prop.broad_risk else 0.0 for prop in selected]) or 0.0,
                "unresolved_broad_underseg_rate": _mean([1.0 if _proposal_unresolved_risk(prop) else 0.0 for prop in selected]) or 0.0,
                "same_frame_violation_count": conflict_count,
                "duplicate_frame_mask_conflict_rate": conflict_rate,
                "stop_reason": select_diag.get("stop_reason"),
                "risk_cap_applied": select_diag.get("risk_cap_applied"),
                "candidate_group_count": select_diag.get("candidate_group_count"),
                "max_members_per_group": select_diag.get("max_members_per_group"),
                "uses_gt_for_prediction": bool(config.oracle),
                "uses_gt_for_evaluation": True,
                "diagnostic_only": True,
                "forbidden_for_method_table": bool(config.oracle),
                "atom_incidence_level": atom_incidence_level,
            }
            row.update(cov)
            row.update(eval_diag)
            metric_rows.append(row)
            covered_seen: set[str] = set()
            if group_mode:
                for rank, (group, detail) in enumerate(selected_groups_with_detail):
                    group_id = f"{config.variant}:{chunk_id}:{rank:04d}"
                    for member_rank, prop in enumerate(group):
                        new_atoms = [key for key in coverage.get(prop.unit_key, {}) if key not in covered_seen]
                        covered_seen.update(coverage.get(prop.unit_key, {}).keys())
                        selected_rows.append(
                            {
                                "variant": config.variant,
                                "scene_id": scene,
                                "chunk_id": chunk_id,
                                "selection_rank": rank,
                                "selection_unit_id": group_id,
                                "selection_unit_member_rank": member_rank,
                                "proposal_id": prop.proposal_id,
                                "source_variant": prop.source_variant,
                                "frame_id": prop.frame_id,
                                "source_mask_id": prop.source_mask_id,
                                "source_obs_id": prop.source_obs_id,
                                "semantic_prototype_id": prop.semantic_prototype_id,
                                "proposal_area_ratio": prop.area_ratio,
                                "proposal_semantic_entropy": prop.semantic_entropy,
                                "source_semantic_entropy": prop.source_semantic_entropy,
                                "objectness_score": prop.objectness_score,
                                "d4rt_score": prop.d4rt_score,
                                "majority_gt_id_diagnostic": prop.majority_gt,
                                "majority_iou_diagnostic": prop.majority_iou,
                                "source_broad_large_risk": prop.broad_risk,
                                "source_underseg_proxy": prop.underseg_risk,
                                "proposal_unresolved_broad_underseg_risk": _proposal_unresolved_risk(prop),
                                "new_atom_count": len(new_atoms),
                                "new_total_weight": detail.get("new_total_weight"),
                                "new_semantic_weight": detail.get("new_semantic_weight"),
                                "new_d4rt_weight": detail.get("new_d4rt_weight"),
                                "score_total": detail.get("score_total"),
                                "group_frame_count": detail.get("group_frame_count"),
                                "group_member_count": detail.get("group_member_count"),
                                "group_risk_rate": detail.get("group_risk_rate"),
                                "uses_gt_for_prediction": bool(config.oracle),
                                "uses_gt_for_evaluation": True,
                                "diagnostic_only": True,
                                "forbidden_for_method_table": bool(config.oracle),
                            }
                        )
            else:
                for rank, (prop, detail) in enumerate(selected_with_detail):
                    new_atoms = [key for key in coverage.get(prop.unit_key, {}) if key not in covered_seen]
                    covered_seen.update(coverage.get(prop.unit_key, {}).keys())
                    selected_rows.append(
                        {
                            "variant": config.variant,
                            "scene_id": scene,
                            "chunk_id": chunk_id,
                            "selection_rank": rank,
                            "selection_unit_id": f"{config.variant}:{chunk_id}:{rank:04d}",
                            "selection_unit_member_rank": 0,
                            "proposal_id": prop.proposal_id,
                            "source_variant": prop.source_variant,
                            "frame_id": prop.frame_id,
                            "source_mask_id": prop.source_mask_id,
                            "source_obs_id": prop.source_obs_id,
                            "semantic_prototype_id": prop.semantic_prototype_id,
                            "proposal_area_ratio": prop.area_ratio,
                            "proposal_semantic_entropy": prop.semantic_entropy,
                            "source_semantic_entropy": prop.source_semantic_entropy,
                            "objectness_score": prop.objectness_score,
                            "d4rt_score": prop.d4rt_score,
                            "majority_gt_id_diagnostic": prop.majority_gt,
                            "majority_iou_diagnostic": prop.majority_iou,
                            "source_broad_large_risk": prop.broad_risk,
                            "source_underseg_proxy": prop.underseg_risk,
                            "proposal_unresolved_broad_underseg_risk": _proposal_unresolved_risk(prop),
                            "new_atom_count": len(new_atoms),
                            "new_total_weight": detail.get("new_total_weight"),
                            "new_semantic_weight": detail.get("new_semantic_weight"),
                            "new_d4rt_weight": detail.get("new_d4rt_weight"),
                            "score_total": detail.get("score_total"),
                            "uses_gt_for_prediction": bool(config.oracle),
                            "uses_gt_for_evaluation": True,
                            "diagnostic_only": True,
                            "forbidden_for_method_table": bool(config.oracle),
                        }
                    )

    variant_summary_rows = [_summarize_variant(metric_rows, variant) for variant in sorted({row["variant"] for row in metric_rows})]
    non_oracle = [row for row in variant_summary_rows if not _bool(row.get("uses_gt_for_prediction")) and row.get("variant") != "PSC0_v71_single_mask_baseline"]
    best_method = max(non_oracle, key=lambda row: _float(row.get("representative_proposal_oracle_SF50_diagnostic"), -1.0), default={})
    diagnostic_gt_mean = _float(best_method.get("diagnostic_GT_count_per_chunk_mean"), 0.0)
    selected_count = _float(best_method.get("selected_proposal_count_per_chunk_mean"), 0.0)
    unresolved = _float(best_method.get("unresolved_broad_underseg_rate"), 1.0)
    gate = {
        "all_inputs_present": not missing_rows,
        "best_method_variant": best_method.get("variant"),
        "representative_proposal_oracle_SF50_ge_0p30": _float(best_method.get("representative_proposal_oracle_SF50_diagnostic"), 0.0) >= 0.30,
        "representative_proposal_GT_best_IoU_mean_ge_0p25": _float(best_method.get("representative_proposal_GT_best_IoU_mean_diagnostic"), 0.0) >= 0.25,
        "covered_total_atom_weight_ratio_ge_0p70": _float(best_method.get("covered_total_atom_weight_ratio"), 0.0) >= 0.70,
        "covered_semantic_atom_weight_ratio_ge_0p65": _float(best_method.get("covered_semantic_atom_weight_ratio"), 0.0) >= 0.65,
        "covered_D4RT_atom_weight_ratio_ge_0p20_initial": _float(best_method.get("covered_D4RT_atom_weight_ratio"), 0.0) >= 0.20,
        "selected_proposal_count_per_chunk_mean_le_3p5_gt": selected_count <= 3.5 * max(1.0, diagnostic_gt_mean),
        "unresolved_broad_underseg_rate_le_0p35": unresolved <= 0.35,
        "same_frame_violation_count_eq_0": _float(best_method.get("same_frame_violation_count"), 1.0) == 0.0,
        "uses_gt_for_prediction_false": not _bool(best_method.get("uses_gt_for_prediction")),
    }
    pass_keys = [
        "representative_proposal_oracle_SF50_ge_0p30",
        "representative_proposal_GT_best_IoU_mean_ge_0p25",
        "covered_total_atom_weight_ratio_ge_0p70",
        "covered_semantic_atom_weight_ratio_ge_0p65",
        "covered_D4RT_atom_weight_ratio_ge_0p20_initial",
        "selected_proposal_count_per_chunk_mean_le_3p5_gt",
        "unresolved_broad_underseg_rate_le_0p35",
        "same_frame_violation_count_eq_0",
        "uses_gt_for_prediction_false",
    ]
    phase5_pass = all(bool(gate[key]) for key in pass_keys)
    summary = {
        "phase": "v72_phase5_proposal_level_representative_setcover",
        "decision": "PASS_V72_PHASE5_PROPOSAL_SETCOVER" if phase5_pass else "NO_GO_PHASE5_PROPOSAL_SETCOVER",
        "proposal_rows": _rel(proposal_rows),
        "verification_rows": _rel(verification_rows),
        "objectness_rank_rows": _rel(objectness_rank_rows),
        "candidate_rows": _rel(candidate_rows),
        "key_atom_rows": _rel(key_atom_rows),
        "key_atom_variant": str(args.key_atom_variant),
        "target_dense_variant": str(args.target_dense_variant),
        "objectness_variant": str(args.objectness_variant),
        "processed_chunk_count": len({row["chunk_id"] for row in metric_rows}),
        "proposal_count": len(proposals),
        "key_atom_count": len(atoms),
        "selected_proposal_row_count": len(selected_rows),
        "variant_count": len(variant_summary_rows),
        "best_method": best_method,
        "gate": {**gate, "pass": phase5_pass},
        "method_boundary": {
            "uses_gt_for_prediction": False,
            "uses_gt_for_evaluation": True,
            "diagnostic_only": True,
            "forbidden_for_method_table": True,
            "atom_incidence_level": atom_incidence_level,
            "subproposal_D4RT_membership_available": bool(key_atom_uv_coverage),
        },
        "notes": [
            (
                "PSC0-PSC5 use non-GT proposal metadata, v72 Phase4 objectness, v71 K8 key atoms, and key-atom UV incidence for dense subproposals."
                if key_atom_uv_coverage
                else "PSC0-PSC5 use non-GT proposal metadata, v72 Phase4 objectness, v71 K8 key atoms, and source-mask-level inherited atom incidence."
            ),
            "PSC7/PSC8 merge adjacent proposals by non-GT source semantic prototype from the v71 candidate bank.",
            (
                "Dense-token proposal D4RT/key-atom coverage uses D4RT_uv_x/D4RT_uv_y when available; atoms without UV coordinates retain source-level fallback and are marked in the incidence label."
                if key_atom_uv_coverage
                else "Dense-token proposal rows do not contain carrier-to-subregion membership, so D4RT coverage is diagnostic source-level inherited evidence rather than true subproposal D4RT verification."
            ),
            "GT is used only after selection to assign majority-GT oracle labels and compute diagnostic IoU/AP/SF50.",
        ],
    }
    _write_csv(output_root / "proposal_setcover_metric_rows.csv", metric_rows)
    _write_csv(output_root / "proposal_setcover_variant_summary_rows.csv", variant_summary_rows)
    _write_csv(output_root / "selected_proposal_rows.csv", selected_rows)
    _write_csv(output_root / "missing_input_rows.csv", missing_rows)
    _write_json(output_root / "proposal_setcover_summary.json", summary)
    sha_rows = []
    for path in sorted(output_root.glob("*")):
        if path.is_file() and path.name != "sha256_rows.csv":
            sha_rows.append({"path": _rel(path), "bytes": int(path.stat().st_size), "sha256": _sha256(path), "kind": "output"})
    for path in [proposal_rows, verification_rows, objectness_rank_rows, candidate_rows, key_atom_rows, witness_summary]:
        if path.exists():
            sha_rows.append({"path": _rel(path), "bytes": int(path.stat().st_size), "sha256": _sha256(path), "kind": "input"})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v72 Phase5 proposal-level representative set-cover diagnostic.")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--proposal-rows", default="outputs/audit/v72_phase2_dense_token_proposals_smoke10_area_bin1/proposal_rows.csv")
    parser.add_argument("--verification-rows", default="outputs/audit/v72_phase3_d4rt_proposal_verification_area_bin1/proposal_verification_rows.csv")
    parser.add_argument("--objectness-rank-rows", default="outputs/audit/v72_phase4_objectness_ranking_area_bin1_riskcap_bgproxyfix/objectness_rank_rows.csv")
    parser.add_argument("--candidate-rows", default="outputs/audit/v71_candidate_bank/candidate_mask_rows.csv")
    parser.add_argument("--key-atom-rows", default="outputs/audit/v71_key_atoms/key_atom_rows.csv")
    parser.add_argument("--witness-summary", default="outputs/audit/v70_carrier_witness/witness_summary.json")
    parser.add_argument("--output-root", default="outputs/audit/v72_phase5_proposal_setcover")
    parser.add_argument("--key-atom-variant", default="K8_reliability_weighted_geo_semantic_FPS")
    parser.add_argument("--target-dense-variant", default="SP2_DINO_affinity_connected_components")
    parser.add_argument("--objectness-variant", default="OR9_risk_capped_area_repair")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-proposals-per-chunk", type=int, default=64)
    parser.add_argument("--max-members-per-group", type=int, default=24)
    parser.add_argument("--min-gain", type=float, default=-10.0)
    parser.add_argument("--key-atom-uv-coverage", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
