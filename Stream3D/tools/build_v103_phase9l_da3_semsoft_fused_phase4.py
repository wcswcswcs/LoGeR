#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_v103_phase3_fast_carrier_reliability_filter import (  # noqa: E402
    OBJECT_LIKE_AREA_MIN,
    _load_scene_summary_and_masks,
    _load_semantic,
)
from diagnose_v103_phase9g_da3_seed_gaussian_neighborhood import (  # noqa: E402
    AUDIT_ROOT,
    DEFAULT_PHASE9E_ROOT,
    PLAN_DOC,
    SCENE_SPECS,
    _frame_label_lookup,
    _hit_counts,
    _load_or_project_da3,
    _load_phase9e_supports,
    _project_phase,
    _rel,
    _write_csv,
    _write_json,
)
from diagnose_v103_phase9h_da3_object_local_components import (  # noqa: E402
    _component_labels,
    _ratio,
)
from diagnose_v103_phase9i_da3_seed_growth_objectlike_graph import _rank_and_cap  # noqa: E402
from diagnose_v103_phase9j_broad_mask_threshold_relaxation import _mask_rows  # noqa: E402


PHASE_ID = "v103_phase9l_da3_semsoft_fused_phase4"
DEFAULT_PHASE4_ROOT = AUDIT_ROOT / "v103_phase4_positive_core_affinity_q5c_repair5_r12_dual_role"
DEFAULT_PHASE3_ROOT = AUDIT_ROOT / "v103_phase3_carrier_reliability_filter_q5c_objlike16384_competing_repair5"
DEFAULT_PROJECTION_CACHE_ROOT = AUDIT_ROOT / "v103_phase9g_da3_seed_gaussian_neighborhood_r1"
DEFAULT_OUT = AUDIT_ROOT / "v103_phase9l_da3_semsoft_fused_phase4_positive_core_r1"
SKETCH_SEED = 10317

K5_VARIANT = {
    "variant_id": "k5_semsoft_allposseed_area020_veto050_knn8_q90_obs005_count4_min8_seed4",
    "seed_source": "all_positive",
    "object_hit_min": 1,
    "object_ratio_min": 0.03,
    "seed_hit_min": 1,
    "seed_ratio_min": 0.03,
    "veto_ratio_max": 0.50,
    "broad_ratio_max": 1.01,
    "max_universe_gaussians": 250000,
    "component_knn_k": 8,
    "edge_radius_quantile": 0.90,
    "edge_radius_factor": 2.0,
    "min_component_gaussians": 8,
    "max_component_gaussians": 80000,
    "min_seed_gaussians_per_component": 4,
    "obs_selected_ratio_min": 0.05,
    "obs_selected_count_min": 4,
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _semantic_soft_candidates(mask_rows: list[dict[str, Any]], area_max: float = 0.20) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for row in mask_rows:
        area = float(row["area_ratio"])
        if float(OBJECT_LIKE_AREA_MIN) <= area <= float(area_max):
            out.add((int(row["frame_id"]), int(row["mask_id"])))
    return out


def _area_broad_masks(mask_rows: list[dict[str, Any]], area_min: float = 0.20) -> set[tuple[int, int]]:
    return {
        (int(row["frame_id"]), int(row["mask_id"]))
        for row in mask_rows
        if float(row["area_ratio"]) >= float(area_min)
    }


def _obs_meta_soft(scene_id: str, mask_rows: list[dict[str, Any]], candidate_masks: set[tuple[int, int]]) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for row in mask_rows:
        key = (int(row["frame_id"]), int(row["mask_id"]))
        obs = f"{scene_id}:{key[0]}:{key[1]}"
        meta[obs] = {
            "frame_id": key[0],
            "mask_id": key[1],
            "is_object_like": bool(key in candidate_masks),
            "is_broad": bool(float(row["area_ratio"]) >= 0.20),
            "semantic_broad_risk": bool(row["semantic_broad_risk"]),
            "area_ratio": float(row["area_ratio"]),
        }
    return meta


def _load_phase3_reliability(phase3_root: Path, scene: str, carrier_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    path = phase3_root / scene / "carrier_reliability_rows.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path, columns=["carrier_id", "reliability_s2", "broad_mask_participation_rate"])
    ids = df["carrier_id"].to_numpy(dtype=np.int64, copy=False)
    order = np.argsort(ids, kind="mergesort")
    ids_sorted = ids[order]
    rel_sorted = df["reliability_s2"].to_numpy(dtype=np.float32, copy=False)[order]
    broad_sorted = df["broad_mask_participation_rate"].to_numpy(dtype=np.float32, copy=False)[order]
    rel = np.zeros((carrier_id.shape[0],), dtype=np.float32)
    broad = np.ones((carrier_id.shape[0],), dtype=np.float32)
    if ids_sorted.size:
        pos = np.searchsorted(ids_sorted, np.asarray(carrier_id, dtype=np.int64))
        found = (pos < ids_sorted.shape[0]) & (ids_sorted[np.minimum(pos, ids_sorted.shape[0] - 1)] == carrier_id)
        if np.any(found):
            rel[found] = rel_sorted[pos[found]]
            broad[found] = broad_sorted[pos[found]]
    return rel, broad


def _obs_count_bits(prefix: str, obs_set: set[str], obs_meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    known = [obs for obs in obs_set if obs in obs_meta]
    obj = [obs for obs in known if bool(obs_meta[obs]["is_object_like"])]
    broad = [obs for obs in known if bool(obs_meta[obs]["is_broad"])]
    sem_broad = [obs for obs in known if bool(obs_meta[obs].get("semantic_broad_risk", False))]
    return {
        f"{prefix}_obs_count": int(len(obs_set)),
        f"{prefix}_known_obs_count": int(len(known)),
        f"{prefix}_object_like_obs_count": int(len(obj)),
        f"{prefix}_object_like_obs_rate": float(len(obj) / max(len(known), 1)),
        f"{prefix}_broad_obs_count": int(len(broad)),
        f"{prefix}_broad_obs_rate": float(len(broad) / max(len(known), 1)),
        f"{prefix}_semantic_broad_risk_obs_count": int(len(sem_broad)),
        f"{prefix}_semantic_broad_risk_obs_rate": float(len(sem_broad) / max(len(known), 1)),
    }


def _component_observation_rows(
    *,
    scene_id: str,
    mask_by_frame: np.ndarray,
    frame_df: pd.DataFrame,
    candidate_idx: np.ndarray,
    labels: np.ndarray,
    keep_component: np.ndarray,
    component_sizes: np.ndarray,
    component_seed_counts: np.ndarray,
    positive_obs: set[str],
    obs_meta: dict[str, dict[str, Any]],
    veto_ratio: np.ndarray,
    obs_selected_ratio_min: float,
    obs_selected_count_min: int,
    component_frame_policy: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    component_rows: list[dict[str, Any]] = []
    obs_rows: list[dict[str, Any]] = []
    if candidate_idx.size == 0 or labels.size == 0 or not np.any(keep_component):
        return component_rows, obs_rows

    frame_ids = [int(row.frame_id) for row in frame_df.itertuples(index=False)]
    kept_component_ids = np.flatnonzero(keep_component).astype(int).tolist()
    idx_by_component: dict[int, np.ndarray] = {
        comp_id: candidate_idx[labels == int(comp_id)].astype(np.int64, copy=False)
        for comp_id in kept_component_ids
    }

    selected_by_component: dict[int, set[str]] = {int(c): set() for c in kept_component_ids}
    obs_bits_by_component: dict[int, list[dict[str, Any]]] = {int(c): [] for c in kept_component_ids}

    for comp_id, comp_idx in idx_by_component.items():
        if comp_idx.size == 0:
            continue
        for fi, frame_id in enumerate(frame_ids):
            frame_labels_all = np.asarray(mask_by_frame[fi], dtype=np.int32)
            labels_frame = frame_labels_all[comp_idx]
            valid = labels_frame > 0
            component_visible = int(np.count_nonzero(valid))
            if component_visible == 0:
                continue
            labels_valid = labels_frame[valid].astype(np.int64, copy=False)
            label_ids, counts = np.unique(labels_valid, return_counts=True)
            total_labels = frame_labels_all[frame_labels_all > 0]
            total_counts = np.bincount(total_labels)
            for label, count in zip(label_ids.tolist(), counts.tolist()):
                if int(count) < int(obs_selected_count_min):
                    continue
                mask_total = int(total_counts[int(label)]) if int(label) < int(total_counts.shape[0]) else int(count)
                mask_cover_ratio = float(count / max(mask_total, 1))
                if mask_cover_ratio < float(obs_selected_ratio_min):
                    continue
                component_membership_ratio = float(count / max(component_visible, 1))
                component_frame_visible_rate = float(component_visible / max(int(component_sizes[int(comp_id)]), 1))
                obs = f"{scene_id}:{int(frame_id)}:{int(label)}"
                meta = obs_meta.get(obs, {})
                selected_by_component[int(comp_id)].add(obs)
                obs_bits_by_component[int(comp_id)].append(
                    {
                        "schema_version": "stream4d_v103_phase9l_component_observation_row_v1",
                        "phase_id": PHASE_ID,
                        "scene_id": scene_id,
                        "variant_id": K5_VARIANT["variant_id"],
                        "component_id": int(comp_id),
                        "frame_id": int(frame_id),
                        "frame_local_index": int(fi),
                        "mask_id": int(label),
                        "mask_observation_id": obs,
                        "component_mask_gaussian_count": int(count),
                        "component_frame_visible_gaussian_count": int(component_visible),
                        "mask_total_gaussian_count": int(mask_total),
                        "component_mask_cover_ratio": mask_cover_ratio,
                        "component_mask_membership_ratio": component_membership_ratio,
                        "component_frame_visible_rate": component_frame_visible_rate,
                        "is_positive_anchor_obs": bool(obs in positive_obs),
                        "is_induced_obs": bool(obs not in positive_obs),
                        "is_object_like": bool(meta.get("is_object_like", False)),
                        "is_broad": bool(meta.get("is_broad", True)),
                        "semantic_broad_risk": bool(meta.get("semantic_broad_risk", False)),
                        "area_ratio": float(meta.get("area_ratio", 0.0)),
                        "uses_gt_for_prediction": False,
                    }
                )

    if component_frame_policy == "wta_object_like":
        for comp_id, rows in list(obs_bits_by_component.items()):
            by_frame: dict[int, list[dict[str, Any]]] = {}
            for row in rows:
                by_frame.setdefault(int(row["frame_local_index"]), []).append(row)
            kept: list[dict[str, Any]] = []
            for frame_rows in by_frame.values():
                best = max(
                    frame_rows,
                    key=lambda row: (
                        int(bool(row["is_object_like"])),
                        int(not bool(row["is_broad"])),
                        int(bool(row["is_positive_anchor_obs"])),
                        float(row["component_mask_membership_ratio"]),
                        float(row["component_mask_cover_ratio"]),
                        int(row["component_mask_gaussian_count"]),
                    ),
                )
                best["component_frame_policy"] = component_frame_policy
                kept.append(best)
            obs_bits_by_component[int(comp_id)] = kept
            selected_by_component[int(comp_id)] = {str(row["mask_observation_id"]) for row in kept}
    elif component_frame_policy == "all_masks":
        for rows in obs_bits_by_component.values():
            for row in rows:
                row["component_frame_policy"] = component_frame_policy
    else:
        raise ValueError(f"unsupported component_frame_policy={component_frame_policy}")

    for comp_id in kept_component_ids:
        selected_obs = selected_by_component[int(comp_id)]
        if not selected_obs:
            continue
        induced_obs = selected_obs - positive_obs
        bits = _obs_count_bits("selected", selected_obs, obs_meta)
        bits.update(_obs_count_bits("induced", induced_obs, obs_meta))
        selected_known = max(int(bits["selected_known_obs_count"]), 1)
        object_like_rate = float(bits["selected_object_like_obs_count"]) / float(selected_known)
        broad_rate = float(bits["selected_broad_obs_count"]) / float(selected_known)
        semantic_broad_rate = float(bits["selected_semantic_broad_risk_obs_count"]) / float(selected_known)
        comp_idx = idx_by_component[int(comp_id)]
        component_veto_ratio_mean = float(np.mean(veto_ratio[comp_idx])) if comp_idx.size else 1.0
        seed_density = float(component_seed_counts[int(comp_id)] / max(int(component_sizes[int(comp_id)]), 1))
        reliability = (
            max(0.0, min(1.0, object_like_rate))
            * max(0.0, 1.0 - broad_rate)
            * max(0.0, 1.0 - component_veto_ratio_mean)
            * math.sqrt(max(0.0, min(1.0, seed_density)))
        )
        reliability = float(np.clip(reliability, 0.0, 1.0))
        is_clean = object_like_rate >= 0.60 and broad_rate <= 0.30
        row = {
            "schema_version": "stream4d_v103_phase9l_component_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene_id,
            "variant_id": K5_VARIANT["variant_id"],
            "component_id": int(comp_id),
            "component_gaussian_count": int(component_sizes[int(comp_id)]),
            "component_seed_gaussian_count": int(component_seed_counts[int(comp_id)]),
            "component_seed_density": seed_density,
            "component_veto_ratio_mean": component_veto_ratio_mean,
            "component_reliability": reliability,
            "component_broad_risk": broad_rate,
            "component_semantic_broad_risk": semantic_broad_rate,
            "is_clean_component": bool(is_clean),
            "included_as_da3_primitive": bool(is_clean and reliability > 0.0 and bool(induced_obs)),
            "uses_gt_for_prediction": False,
        }
        row.update(bits)
        component_rows.append(row)
        for obs_row in obs_bits_by_component[int(comp_id)]:
            obs_row["component_reliability"] = reliability
            obs_row["component_broad_risk"] = broad_rate
            obs_row["is_clean_component"] = bool(is_clean)
            obs_row["included_as_da3_primitive"] = bool(is_clean and reliability > 0.0 and bool(induced_obs))
            obs_rows.append(obs_row)

    return component_rows, obs_rows


def _countsketch(
    carrier_idx: np.ndarray,
    mask_idx: np.ndarray,
    b_ia: np.ndarray,
    mask_weight: np.ndarray,
    carrier_count: int,
    sketch_dim: int,
    device: torch.device,
) -> np.ndarray:
    if carrier_count == 0 or carrier_idx.size == 0:
        return np.zeros((carrier_count, sketch_dim), dtype=np.float32)
    with torch.no_grad():
        c_t = torch.as_tensor(carrier_idx, dtype=torch.long, device=device)
        m_t = torch.as_tensor(mask_idx, dtype=torch.long, device=device)
        b_t = torch.as_tensor(b_ia, dtype=torch.float32, device=device)
        w_t = torch.as_tensor(mask_weight, dtype=torch.float32, device=device)
        bucket = ((m_t * 2654435761 + SKETCH_SEED) % int(sketch_dim)).to(torch.long)
        sign = torch.where(((m_t * 1103515245 + SKETCH_SEED) % 2) == 0, 1.0, -1.0).to(torch.float32)
        values = torch.sqrt(w_t[m_t]) * b_t * sign
        out = torch.zeros((int(carrier_count), int(sketch_dim)), dtype=torch.float32, device=device)
        out.index_put_((c_t, bucket), values, accumulate=True)
        out = torch.nn.functional.normalize(out, p=2, dim=1, eps=1e-12)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return out.detach().cpu().numpy().astype(np.float32, copy=False)


def _run_scene(scene_id: str, args: argparse.Namespace, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    phase4_root = _project(args.phase4_root)
    phase3_root = _project(args.phase3_root)
    phase9e_root = _project_phase(args.phase9e_root)
    projection_cache_root = _project_phase(args.projection_cache_root) if str(args.projection_cache_root).strip() else None
    out = _project_phase(args.output_root)
    scene_out = out / scene_id
    scene_out.mkdir(parents=True, exist_ok=True)

    base = torch.load(phase4_root / scene_id / "primitive_incidence_sparse.pt", map_location="cpu")
    base_carrier_id = base["carrier_id"].cpu().numpy().astype(np.int64)
    base_local = base["carrier_local_index"].cpu().numpy().astype(np.int64)
    base_mask_idx = base["mask_observation_index"].cpu().numpy().astype(np.int64)
    base_frame = base["frame_local_index"].cpu().numpy().astype(np.int64)
    base_mask_id = base["mask_id"].cpu().numpy().astype(np.int64)
    base_b = base["B_ia"].cpu().numpy().astype(np.float32)
    mask_frame = base["mask_frame"].cpu().numpy().astype(np.int64)
    mask_label = base["mask_label"].cpu().numpy().astype(np.int64)
    mask_is_object = base["mask_is_object_like"].cpu().numpy().astype(bool)
    mask_is_broad = base["mask_is_broad"].cpu().numpy().astype(bool)
    mask_weight = base["mask_weight"].cpu().numpy().astype(np.float32)
    obs_lookup = {(int(frame), int(label)): int(idx) for idx, (frame, label) in enumerate(zip(mask_frame.tolist(), mask_label.tolist()))}

    d4rt_rel, d4rt_broad = _load_phase3_reliability(phase3_root, scene_id, base_carrier_id)
    spec = dict(SCENE_SPECS[scene_id])
    spec["phase2_root"] = _project(spec["phase2_root"])
    _summary, frame_ids, masks = _load_scene_summary_and_masks(scene_id, spec["phase2_root"])
    _feature_index, _features, semantic_meta, semantic_constants = _load_semantic(scene_id, spec)
    mask_rows = _mask_rows(scene_id, frame_ids, masks, semantic_meta)
    candidate_masks = _semantic_soft_candidates(mask_rows, area_max=float(args.semantic_soft_area_max))
    broad_masks = _area_broad_masks(mask_rows, area_min=float(args.area_broad_min))
    positive_masks, veto_masks = _load_phase9e_supports(phase9e_root, scene_id)
    positive_obs = {f"{scene_id}:{int(frame_id)}:{int(mask_id)}" for frame_id, mask_id in positive_masks}

    mask_by_frame, xyz, frame_df, _meta, manifest = _load_or_project_da3(scene_id, out, projection_cache_root)
    obs_meta = _obs_meta_soft(scene_id, mask_rows, candidate_masks)
    mask_is_object_fused = mask_is_object.copy()
    mask_is_broad_fused = mask_is_broad.copy()
    mask_weight_fused = mask_weight.copy()
    mask_semantic_broad_risk = np.zeros_like(mask_is_broad_fused, dtype=bool)
    mask_area_ratio = np.zeros_like(mask_weight_fused, dtype=np.float32)
    semsoft_promoted_set: set[int] = set()
    semsoft_promoted_semantic_risk_set: set[int] = set()
    if str(args.semantic_broad_mode) == "soft_candidate_risk":
        object_non_broad_weight = mask_weight[mask_is_object & ~mask_is_broad]
        weight_floor = float(np.median(object_non_broad_weight)) if object_non_broad_weight.size else 1.0
        weight_floor *= float(args.semantic_soft_weight_scale)
        for obs_idx, (frame_local, label) in enumerate(zip(mask_frame.tolist(), mask_label.tolist())):
            if int(frame_local) < 0 or int(frame_local) >= len(frame_ids):
                continue
            obs = f"{scene_id}:{int(frame_ids[int(frame_local)])}:{int(label)}"
            meta_bits = obs_meta.get(obs)
            if not meta_bits:
                continue
            mask_semantic_broad_risk[int(obs_idx)] = bool(meta_bits.get("semantic_broad_risk", False))
            mask_area_ratio[int(obs_idx)] = float(meta_bits.get("area_ratio", 0.0))
            if (
                str(args.semantic_soft_promotion_scope) == "all_candidates"
                and bool(meta_bits.get("is_object_like", False))
                and not bool(meta_bits.get("is_broad", True))
            ):
                was_broad = bool(mask_is_broad_fused[int(obs_idx)])
                mask_is_object_fused[int(obs_idx)] = True
                mask_is_broad_fused[int(obs_idx)] = False
                mask_weight_fused[int(obs_idx)] = max(float(mask_weight_fused[int(obs_idx)]), weight_floor)
                if was_broad:
                    semsoft_promoted_set.add(int(obs_idx))
                    if bool(meta_bits.get("semantic_broad_risk", False)):
                        semsoft_promoted_semantic_risk_set.add(int(obs_idx))
    elif str(args.semantic_broad_mode) != "hard_base":
        raise ValueError(f"unsupported semantic_broad_mode={args.semantic_broad_mode}")
    max_label = int(np.max(mask_by_frame)) if mask_by_frame.size else 0
    object_lookup = _frame_label_lookup(frame_df, candidate_masks, max_label)
    seed_lookup = _frame_label_lookup(frame_df, positive_masks, max_label)
    veto_lookup = _frame_label_lookup(frame_df, veto_masks, max_label)
    broad_lookup = _frame_label_lookup(frame_df, broad_masks, max_label)
    object_hit = _hit_counts(mask_by_frame, object_lookup)
    seed_hit = _hit_counts(mask_by_frame, seed_lookup)
    veto_hit = _hit_counts(mask_by_frame, veto_lookup)
    broad_hit = _hit_counts(mask_by_frame, broad_lookup)
    visible_count = np.sum(mask_by_frame > 0, axis=0).astype(np.int16)
    object_ratio = _ratio(object_hit, visible_count)
    seed_ratio = _ratio(seed_hit, visible_count)
    veto_ratio = _ratio(veto_hit, visible_count)
    broad_ratio = _ratio(broad_hit, visible_count)

    variant = dict(K5_VARIANT)
    universe = (
        (visible_count > 0)
        & (object_hit >= int(variant["object_hit_min"]))
        & (object_ratio >= float(variant["object_ratio_min"]))
        & (veto_ratio <= float(variant["veto_ratio_max"]))
        & (broad_ratio <= float(variant["broad_ratio_max"]))
    )
    seed = universe & (seed_hit >= int(variant["seed_hit_min"])) & (seed_ratio >= float(variant["seed_ratio_min"]))
    universe_idx_raw = np.flatnonzero(universe).astype(np.int64, copy=False)
    seed_idx = np.flatnonzero(seed).astype(np.int64, copy=False)
    universe_idx = _rank_and_cap(
        universe_idx=universe_idx_raw,
        seed_idx=seed_idx,
        max_count=int(variant["max_universe_gaussians"]),
        object_hit=object_hit,
        object_ratio=object_ratio,
        seed_hit=seed_hit,
        broad_ratio=broad_ratio,
    )
    labels, component_info = _component_labels(
        xyz=xyz,
        candidate_idx=universe_idx,
        k=int(variant["component_knn_k"]),
        edge_radius_quantile=float(variant["edge_radius_quantile"]),
        edge_radius_factor=float(variant["edge_radius_factor"]),
    )
    component_sizes = np.bincount(labels, minlength=int(component_info["component_count_total"])) if labels.size else np.asarray([], dtype=np.int64)
    seed_for_candidate = seed[universe_idx] if universe_idx.size else np.asarray([], dtype=bool)
    component_seed_counts = (
        np.bincount(labels, weights=seed_for_candidate.astype(np.int32), minlength=int(component_info["component_count_total"])).astype(np.int64)
        if labels.size
        else np.asarray([], dtype=np.int64)
    )
    keep_component = (
        (component_sizes >= int(variant["min_component_gaussians"]))
        & (component_sizes <= int(variant["max_component_gaussians"]))
        & (component_seed_counts >= int(variant["min_seed_gaussians_per_component"]))
    ) if component_sizes.size else np.asarray([], dtype=bool)

    component_rows, component_obs_rows = _component_observation_rows(
        scene_id=scene_id,
        mask_by_frame=mask_by_frame,
        frame_df=frame_df,
        candidate_idx=universe_idx,
        labels=labels,
        keep_component=keep_component,
        component_sizes=component_sizes,
        component_seed_counts=component_seed_counts,
        positive_obs=positive_obs,
        obs_meta=obs_meta,
        veto_ratio=veto_ratio,
        obs_selected_ratio_min=float(variant["obs_selected_ratio_min"]),
        obs_selected_count_min=int(variant["obs_selected_count_min"]),
        component_frame_policy=str(args.da3_component_frame_policy),
    )
    included_components = [row for row in component_rows if bool(row["included_as_da3_primitive"])]
    component_local_by_id = {
        int(row["component_id"]): int(base_carrier_id.shape[0] + idx)
        for idx, row in enumerate(included_components)
    }
    component_rel_by_id = {int(row["component_id"]): float(row["component_reliability"]) for row in included_components}
    component_broad_by_id = {int(row["component_id"]): float(row["component_broad_risk"]) for row in included_components}

    da3_incidence_rows: list[list[float]] = []
    da3_case_rows: list[dict[str, Any]] = []
    induced_obs: set[str] = set()
    for row in component_obs_rows:
        comp_id = int(row["component_id"])
        if comp_id not in component_local_by_id:
            continue
        lookup_key = (int(row["frame_local_index"]), int(row["mask_id"]))
        obs_idx = obs_lookup.get(lookup_key)
        if obs_idx is None:
            continue
        rel = float(component_rel_by_id[comp_id])
        b_val = rel * float(row["component_frame_visible_rate"]) * float(row["component_mask_membership_ratio"])
        if not np.isfinite(b_val) or b_val <= 0.0:
            continue
        da3_incidence_rows.append(
            [
                float(component_local_by_id[comp_id]),
                float(obs_idx),
                float(row["frame_local_index"]),
                float(row["mask_id"]),
                float(b_val),
            ]
        )
        case = dict(row)
        case["mask_observation_index"] = int(obs_idx)
        case["synthetic_carrier_local_index"] = int(component_local_by_id[comp_id])
        case["B_ia"] = float(b_val)
        da3_case_rows.append(case)
        if bool(row["is_induced_obs"]):
            induced_obs.add(str(row["mask_observation_id"]))

    if str(args.semantic_broad_mode) == "soft_candidate_risk" and str(args.semantic_soft_promotion_scope) == "da3_supported":
        object_non_broad_weight = mask_weight[mask_is_object & ~mask_is_broad]
        weight_floor = float(np.median(object_non_broad_weight)) if object_non_broad_weight.size else 1.0
        weight_floor *= float(args.semantic_soft_weight_scale)
        for row in da3_case_rows:
            obs_idx = int(row["mask_observation_index"])
            obs = str(row["mask_observation_id"])
            meta_bits = obs_meta.get(obs, {})
            if bool(meta_bits.get("is_object_like", False)) and not bool(meta_bits.get("is_broad", True)):
                was_broad = bool(mask_is_broad_fused[obs_idx])
                mask_is_object_fused[obs_idx] = True
                mask_is_broad_fused[obs_idx] = False
                mask_weight_fused[obs_idx] = max(float(mask_weight_fused[obs_idx]), weight_floor)
                if was_broad:
                    semsoft_promoted_set.add(obs_idx)
                    if bool(meta_bits.get("semantic_broad_risk", False)):
                        semsoft_promoted_semantic_risk_set.add(obs_idx)

    da3_incidence = np.asarray(da3_incidence_rows, dtype=np.float64).reshape(-1, 5)
    fused_carrier_id = np.concatenate(
        [
            base_carrier_id,
            -900_000_000_000 - np.asarray([int(row["component_id"]) for row in included_components], dtype=np.int64),
        ]
    ).astype(np.int64, copy=False)
    fused_rel = np.concatenate(
        [
            d4rt_rel,
            np.asarray([float(row["component_reliability"]) for row in included_components], dtype=np.float32),
        ]
    ).astype(np.float32, copy=False)
    fused_broad = np.concatenate(
        [
            d4rt_broad,
            np.asarray([float(row["component_broad_risk"]) for row in included_components], dtype=np.float32),
        ]
    ).astype(np.float32, copy=False)
    if da3_incidence.size:
        fused_incidence = np.concatenate(
            [
                np.stack([base_local, base_mask_idx, base_frame, base_mask_id, base_b], axis=1).astype(np.float64),
                da3_incidence,
            ],
            axis=0,
        )
    else:
        fused_incidence = np.stack([base_local, base_mask_idx, base_frame, base_mask_id, base_b], axis=1).astype(np.float64)

    feature = _countsketch(
        carrier_idx=fused_incidence[:, 0].astype(np.int64),
        mask_idx=fused_incidence[:, 1].astype(np.int64),
        b_ia=fused_incidence[:, 4].astype(np.float32),
        mask_weight=mask_weight_fused,
        carrier_count=int(fused_carrier_id.shape[0]),
        sketch_dim=int(args.sketch_dim),
        device=device,
    )
    norm = np.linalg.norm(feature, axis=1)

    incidence_path = scene_out / "primitive_incidence_sparse.pt"
    feature_path = scene_out / "primitive_affinity_feature.pt"
    provider = ["d4rt_positive_core" for _ in range(base_carrier_id.shape[0])] + ["da3_semantic_soft_component" for _ in included_components]
    torch.save(
        {
            "schema_version": "stream4d_v103_phase9l_fused_primitive_incidence_sparse_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene_id,
            "carrier_id": torch.as_tensor(fused_carrier_id, dtype=torch.int64),
            "carrier_local_index": torch.as_tensor(fused_incidence[:, 0].astype(np.int64), dtype=torch.int64),
            "mask_observation_index": torch.as_tensor(fused_incidence[:, 1].astype(np.int64), dtype=torch.int64),
            "frame_local_index": torch.as_tensor(fused_incidence[:, 2].astype(np.int64), dtype=torch.int64),
            "mask_id": torch.as_tensor(fused_incidence[:, 3].astype(np.int64), dtype=torch.int64),
            "B_ia": torch.as_tensor(fused_incidence[:, 4].astype(np.float32), dtype=torch.float32),
            "mask_frame": torch.as_tensor(mask_frame, dtype=torch.int64),
            "mask_label": torch.as_tensor(mask_label, dtype=torch.int64),
            "mask_is_object_like": torch.as_tensor(mask_is_object_fused, dtype=torch.bool),
            "mask_is_broad": torch.as_tensor(mask_is_broad_fused, dtype=torch.bool),
            "mask_weight": torch.as_tensor(mask_weight_fused, dtype=torch.float32),
            "mask_semantic_broad_risk": torch.as_tensor(mask_semantic_broad_risk, dtype=torch.bool),
            "mask_area_ratio": torch.as_tensor(mask_area_ratio, dtype=torch.float32),
            "carrier_reliability": torch.as_tensor(fused_rel, dtype=torch.float32),
            "carrier_broad_risk": torch.as_tensor(fused_broad, dtype=torch.float32),
            "primitive_provider": provider,
            "base_phase4_root": _rel(phase4_root),
            "phase9e_root": _rel(phase9e_root),
            "projection_cache_root": "" if projection_cache_root is None else _rel(projection_cache_root),
            "da3_variant_id": str(variant["variant_id"]),
            "B_ia_formula": "D4RT rows preserve base Phase4 B_ia; DA3 component rows use component_reliability * component_frame_visible_rate * component_mask_membership_ratio.",
            "semantic_broad_mode": str(args.semantic_broad_mode),
            "semantic_soft_promotion_scope": str(args.semantic_soft_promotion_scope),
            "da3_component_frame_policy": str(args.da3_component_frame_policy),
            "mask_weight_policy": (
                "reuse_base_phase4_mask_weight_for_same_mask_observation_vocabulary"
                if str(args.semantic_broad_mode) == "hard_base"
                else "promote_semantic_soft_area_non_broad_candidate_masks_with_weight_floor"
            ),
            "uses_gt": False,
            "uses_future": False,
        },
        incidence_path,
    )
    torch.save(
        {
            "schema_version": "stream4d_v103_phase9l_fused_primitive_affinity_feature_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene_id,
            "carrier_id": torch.as_tensor(fused_carrier_id, dtype=torch.int64),
            "feature": torch.as_tensor(feature, dtype=torch.float16),
            "feature_norm_source_dtype": "float32",
            "sketch_dim": int(args.sketch_dim),
            "sketch_seed": SKETCH_SEED,
            "base_phase4_root": _rel(phase4_root),
            "da3_variant_id": str(variant["variant_id"]),
            "uses_gt": False,
            "uses_future": False,
        },
        feature_path,
    )

    _write_csv(scene_out / "da3_semsoft_component_rows.csv", component_rows)
    _write_csv(scene_out / "da3_semsoft_component_observation_rows.csv", component_obs_rows)
    _write_csv(scene_out / "da3_semsoft_incidence_case_rows.csv", da3_case_rows)

    base_support = np.bincount(base_mask_idx, minlength=int(mask_frame.shape[0])).astype(np.int64)
    fused_support = np.bincount(fused_incidence[:, 1].astype(np.int64), minlength=int(mask_frame.shape[0])).astype(np.int64)
    da3_support = fused_support - base_support
    newly_supported = (base_support == 0) & (da3_support > 0)
    da3_supported = da3_support > 0
    clean_component_count = int(sum(1 for row in component_rows if bool(row["is_clean_component"])))
    included_component_count = int(len(included_components))
    induced_object_like = [obs for obs in induced_obs if bool(obs_meta.get(obs, {}).get("is_object_like", False))]
    induced_broad = [obs for obs in induced_obs if bool(obs_meta.get(obs, {}).get("is_broad", True))]
    semantic_risk_induced = [obs for obs in induced_obs if bool(obs_meta.get(obs, {}).get("semantic_broad_risk", False))]
    metric = {
        "schema_version": "stream4d_v103_phase9l_metric_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene_id,
        "base_phase4_root": _rel(phase4_root),
        "phase9e_root": _rel(phase9e_root),
        "projection_cache_mode": manifest.get("cache_mode", ""),
        "projection_cache_root": manifest.get("cache_root", ""),
        "semantic_constants": semantic_constants,
        "base_d4rt_primitive_count": int(base_carrier_id.shape[0]),
        "fused_primitive_count": int(fused_carrier_id.shape[0]),
        "semantic_broad_mode": str(args.semantic_broad_mode),
        "semantic_soft_promotion_scope": str(args.semantic_soft_promotion_scope),
        "da3_component_frame_policy": str(args.da3_component_frame_policy),
        "semantic_soft_promoted_mask_observation_count": int(len(semsoft_promoted_set)),
        "semantic_soft_promoted_semantic_broad_risk_count": int(len(semsoft_promoted_semantic_risk_set)),
        "da3_clean_component_count": clean_component_count,
        "da3_included_component_count": included_component_count,
        "base_incidence_row_count": int(base_b.shape[0]),
        "da3_incidence_row_count": int(da3_incidence.shape[0]),
        "fused_incidence_row_count": int(fused_incidence.shape[0]),
        "base_mask_observation_support_nonzero_rate": float(np.mean(base_support > 0)) if base_support.size else 0.0,
        "fused_mask_observation_support_nonzero_rate": float(np.mean(fused_support > 0)) if fused_support.size else 0.0,
        "newly_supported_mask_observation_count": int(np.count_nonzero(newly_supported)),
        "newly_supported_base_object_like_count": int(np.count_nonzero(newly_supported & mask_is_object)),
        "newly_supported_base_broad_count": int(np.count_nonzero(newly_supported & mask_is_broad)),
        "newly_supported_object_like_count": int(np.count_nonzero(newly_supported & mask_is_object_fused)),
        "newly_supported_broad_count": int(np.count_nonzero(newly_supported & mask_is_broad_fused)),
        "da3_supported_mask_observation_count": int(np.count_nonzero(da3_supported)),
        "da3_supported_base_object_like_count": int(np.count_nonzero(da3_supported & mask_is_object)),
        "da3_supported_base_broad_count": int(np.count_nonzero(da3_supported & mask_is_broad)),
        "da3_supported_object_like_count": int(np.count_nonzero(da3_supported & mask_is_object_fused)),
        "da3_supported_broad_count": int(np.count_nonzero(da3_supported & mask_is_broad_fused)),
        "da3_induced_obs_count": int(len(induced_obs)),
        "da3_induced_object_like_obs_count": int(len(induced_object_like)),
        "da3_induced_broad_obs_count": int(len(induced_broad)),
        "da3_induced_broad_obs_rate": float(len(induced_broad) / max(len(induced_obs), 1)),
        "da3_induced_semantic_broad_risk_obs_count": int(len(semantic_risk_induced)),
        "da3_induced_semantic_broad_risk_obs_rate": float(len(semantic_risk_induced) / max(len(induced_obs), 1)),
        "feature_valid_rate": float(np.mean(norm > 0.0)) if norm.size else 0.0,
        "da3_reliability_mean": float(np.mean([float(row["component_reliability"]) for row in included_components])) if included_components else 0.0,
        "uses_gt_for_feature": False,
        "uses_future": False,
    }
    gates = [
        ("has_included_da3_component", included_component_count > 0, included_component_count, ">0"),
        ("has_da3_induced_observations", int(len(induced_obs)) > 0, int(len(induced_obs)), ">0"),
        ("da3_induced_broad_obs_rate_le_0p30", metric["da3_induced_broad_obs_rate"] <= 0.30, metric["da3_induced_broad_obs_rate"], 0.30),
        ("feature_valid_rate_ge_0p95", metric["feature_valid_rate"] >= 0.95, metric["feature_valid_rate"], 0.95),
    ]
    gate_rows = [
        {
            "schema_version": "stream4d_v103_phase9l_gate_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene_id,
            "gate_name": name,
            "pass": bool(ok),
            "observed": observed,
            "required": required,
        }
        for name, ok, observed, required in gates
    ]
    failure_rows = [
        {
            "schema_version": "stream4d_v103_phase9l_failure_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene_id,
            "failure_id": row["gate_name"],
            "severity": "blocking",
            "evidence": f"observed={row['observed']} required={row['required']}",
            "uses_gt_for_prediction": False,
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    artifact_rows = [
        {
            "schema_version": "stream4d_v103_phase9l_artifact_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene_id,
            "role": "primitive_incidence_sparse",
            "path": _rel(incidence_path),
            "exists": incidence_path.exists(),
            "size_bytes": incidence_path.stat().st_size if incidence_path.exists() else 0,
        },
        {
            "schema_version": "stream4d_v103_phase9l_artifact_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene_id,
            "role": "primitive_affinity_feature",
            "path": _rel(feature_path),
            "exists": feature_path.exists(),
            "size_bytes": feature_path.stat().st_size if feature_path.exists() else 0,
        },
        {
            "schema_version": "stream4d_v103_phase9l_artifact_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene_id,
            "role": "da3_semsoft_incidence_case_rows",
            "path": _rel(scene_out / "da3_semsoft_incidence_case_rows.csv"),
            "exists": (scene_out / "da3_semsoft_incidence_case_rows.csv").exists(),
            "size_bytes": (scene_out / "da3_semsoft_incidence_case_rows.csv").stat().st_size if (scene_out / "da3_semsoft_incidence_case_rows.csv").exists() else 0,
        },
    ]
    _write_json(scene_out / "scene_summary.json", {"metric": metric, "failure_count": len(failure_rows), "component_info": component_info})
    return metric, gate_rows, failure_rows, artifact_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v103 Phase9l: fuse semantic-soft DA3 component primitives into the positive-core Phase4 affinity interface.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--phase4-root", default=str(DEFAULT_PHASE4_ROOT))
    parser.add_argument("--phase3-root", default=str(DEFAULT_PHASE3_ROOT))
    parser.add_argument("--phase9e-root", default=str(DEFAULT_PHASE9E_ROOT))
    parser.add_argument("--projection-cache-root", default=str(DEFAULT_PROJECTION_CACHE_ROOT))
    parser.add_argument("--scene", choices=["all", "scene0011_00", "scene0050_00"], default="all")
    parser.add_argument("--sketch-dim", type=int, default=2048)
    parser.add_argument("--semantic-soft-area-max", type=float, default=0.20)
    parser.add_argument("--area-broad-min", type=float, default=0.20)
    parser.add_argument("--semantic-broad-mode", choices=["hard_base", "soft_candidate_risk"], default="hard_base")
    parser.add_argument("--semantic-soft-promotion-scope", choices=["all_candidates", "da3_supported"], default="all_candidates")
    parser.add_argument("--semantic-soft-weight-scale", type=float, default=0.50)
    parser.add_argument("--da3-component-frame-policy", choices=["all_masks", "wta_object_like"], default="all_masks")
    return parser


def main() -> int:
    t0 = time.time()
    args = build_parser().parse_args()
    out = _project_phase(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")
    phase4_summary = _read_json(_project(args.phase4_root) / "summary.json")
    if not bool(phase4_summary.get("phase4_pass", False)):
        raise RuntimeError(f"base Phase4 root did not pass: {args.phase4_root}")
    scenes = list(SCENE_SPECS.keys()) if args.scene == "all" else [str(args.scene)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    for scene in scenes:
        try:
            metrics, gates, failures, artifacts = _run_scene(scene, args, device)
            metric_rows.append(metrics)
            gate_rows.extend(gates)
            failure_rows.extend(failures)
            artifact_rows.extend(artifacts)
        except Exception as exc:
            failure = {
                "schema_version": "stream4d_v103_phase9l_failure_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "failure_id": "exception",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "uses_gt_for_prediction": False,
            }
            failure_rows.append(failure)
    _write_csv(out / "phase9l_metric_rows.csv", metric_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    _write_csv(out / "artifact_rows.csv", artifact_rows)
    phase4_ready = len(failure_rows) == 0 and len(metric_rows) == len(scenes)
    summary = {
        "schema_version": "stream4d_v103_phase9l_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": "PASS_PHASE9L_DA3_SEMSOFT_FUSED_PHASE4_ARTIFACT_READY" if phase4_ready else "NO_GO_PHASE9L_DA3_SEMSOFT_FUSED_PHASE4_ARTIFACT",
        "phase4_pass": bool(phase4_ready),
        "phase9l_artifact_ready": bool(phase4_ready),
        "failure_count": len(failure_rows),
        "scene_ids": scenes,
        "base_phase4_root": _rel(_project(args.phase4_root)),
        "phase3_root": _rel(_project(args.phase3_root)),
        "phase9e_root": _rel(_project_phase(args.phase9e_root)),
        "projection_cache_root": "" if not str(args.projection_cache_root).strip() else _rel(_project_phase(args.projection_cache_root)),
        "sketch_dim": int(args.sketch_dim),
        "sketch_seed": SKETCH_SEED,
        "semantic_broad_mode": str(args.semantic_broad_mode),
        "semantic_soft_promotion_scope": str(args.semantic_soft_promotion_scope),
        "semantic_soft_weight_scale": float(args.semantic_soft_weight_scale),
        "da3_component_frame_policy": str(args.da3_component_frame_policy),
        "plan_doc": _rel(PLAN_DOC),
        "uses_gt_for_feature": False,
        "uses_future": False,
        "truthfulness_note": (
            "Phase9l keeps D4RT positive-core primitives as the base Phase4 field and adds only clean semantic-soft "
            "DA3 component primitives as supplemental provider rows. E3/veto support is used only through the veto-ratio "
            "risk gate inherited from Phase9k. This artifact is Phase4-compatible but does not emit AP predictions."
        ),
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "phase9l_metric_rows": _rel(out / "phase9l_metric_rows.csv"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "artifact_rows": _rel(out / "artifact_rows.csv"),
            "last_command": _rel(out / "last_command.txt"),
        },
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if phase4_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
