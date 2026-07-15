#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
TOOLS_DIR = Path(__file__).resolve().parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from build_v103_phase6_mask_clustering_local_object_birth import (  # noqa: E402
    _evaluate_variant,
    _jsonable,
    _read_json,
    _write_csv,
    _write_json,
)
from build_v103_phase6d_f2_skeleton_affinity_merge import (  # noqa: E402
    DEFAULT_F2_ROOT,
    DEFAULT_SUBSET_BASELINE,
    UnionFind,
    _adapt_f2_rows,
    _broad_support_risk,
    _load_phase5_scene,
    _load_subset_baseline,
    _object_tables,
    _project,
    _rel,
    _specific_conflict,
)


PHASE_ID = "v103_r2_phase4_da3_semsoft_edge_intervention"
DEFAULT_OUT = AUDIT_ROOT / PHASE_ID
DEFAULT_PHASE5_ROOT = AUDIT_ROOT / "v103_phase5_da3_bridge_pair_phase9n_suppS1_d4rt48mix_s5repair_r3_allclean"
DEFAULT_PHASE1_ROOT = AUDIT_ROOT / "v103_r2_phase1_semantic_soft_candidate_universe_r2_riskcap055"
DEFAULT_PHASE2_ROOT = AUDIT_ROOT / "v103_r2_phase2_da3_semsoft_support_alpha_density_topk_reliable_veto_r7_from_phase1riskcap"
DEFAULT_PHASE3_ROOT = AUDIT_ROOT / "v103_r2_phase3_da3_semsoft_feature_gate_r6_riskcap_loo_fix"
DEFAULT_PHASE2_SCENE0011 = AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0011_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8"
DEFAULT_PHASE2_SCENE0050 = AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0050_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8"


OBJECT_SPECIFIC_VARIANTS = [
    {
        "variant_id": "R2_4_V0_f2_skeleton_replay",
        "kind": "replay",
        "is_control": False,
        "anchor_threshold": 1.01,
        "da3_threshold": 1.01,
        "support_weak_threshold": 1.01,
        "topk_per_object": 0,
        "use_veto": True,
    },
    {
        "variant_id": "R2_4_V1_anchor065_da3proposal035_top1_veto",
        "kind": "anchor_da3_proposal",
        "is_control": False,
        "anchor_threshold": 0.65,
        "da3_threshold": 0.35,
        "min_da3_bridge_components": 1,
        "min_da3_feature_similarity": 0.05,
        "support_weak_threshold": 0.70,
        "topk_per_object": 1,
        "use_veto": True,
        "max_pair_risk": 0.55,
        "max_component_object_degree": 4,
        "broad_support_veto": True,
        "broad_support_min_broad_rate": 0.70,
        "broad_support_max_object_like_rate": 0.50,
        "broad_support_min_support_mean": 1000.0,
        "control_family": "r2_4_v1",
    },
    {
        "variant_id": "R2_4_V2_anchor065_da3proposal020_top1_vetopriority",
        "kind": "anchor_da3_proposal",
        "is_control": False,
        "anchor_threshold": 0.65,
        "da3_threshold": 0.20,
        "min_da3_bridge_components": 1,
        "min_da3_feature_similarity": -0.05,
        "support_weak_threshold": 0.65,
        "topk_per_object": 1,
        "use_veto": True,
        "max_pair_risk": 0.55,
        "max_component_object_degree": 4,
        "broad_support_veto": True,
        "broad_support_min_broad_rate": 0.70,
        "broad_support_max_object_like_rate": 0.50,
        "broad_support_min_support_mean": 1000.0,
        "control_family": "r2_4_v2",
    },
    {
        "variant_id": "R2_4_R1_shuffled_da3_for_V1",
        "kind": "anchor_da3_proposal",
        "is_control": True,
        "control_for": "R2_4_V1_anchor065_da3proposal035_top1_veto",
        "anchor_threshold": 0.65,
        "da3_threshold": 0.35,
        "min_da3_bridge_components": 1,
        "min_da3_feature_similarity": 0.05,
        "support_weak_threshold": 0.70,
        "topk_per_object": 1,
        "use_veto": True,
        "shuffle_da3": True,
        "max_pair_risk": 0.55,
        "max_component_object_degree": 4,
        "broad_support_veto": True,
        "broad_support_min_broad_rate": 0.70,
        "broad_support_max_object_like_rate": 0.50,
        "broad_support_min_support_mean": 1000.0,
    },
    {
        "variant_id": "R2_4_R2_shuffled_da3_for_V2",
        "kind": "anchor_da3_proposal",
        "is_control": True,
        "control_for": "R2_4_V2_anchor065_da3proposal020_top1_vetopriority",
        "anchor_threshold": 0.65,
        "da3_threshold": 0.20,
        "min_da3_bridge_components": 1,
        "min_da3_feature_similarity": -0.05,
        "support_weak_threshold": 0.65,
        "topk_per_object": 1,
        "use_veto": True,
        "shuffle_da3": True,
        "max_pair_risk": 0.55,
        "max_component_object_degree": 4,
        "broad_support_veto": True,
        "broad_support_min_broad_rate": 0.70,
        "broad_support_max_object_like_rate": 0.50,
        "broad_support_min_support_mean": 1000.0,
    },
]

VETO_PRIORITY_VARIANTS = [
    {
        "variant_id": "R2_4VETO_V0_f2_skeleton_replay",
        "kind": "replay",
        "is_control": False,
        "anchor_threshold": 1.01,
        "da3_threshold": 1.01,
        "support_weak_threshold": 1.01,
        "topk_per_object": 0,
        "use_veto": True,
    },
    {
        "variant_id": "R2_4VETO_V1_lowrisk030_da3proposal035_top1",
        "kind": "anchor_da3_proposal",
        "is_control": False,
        "anchor_threshold": 0.65,
        "da3_threshold": 0.35,
        "min_da3_bridge_components": 1,
        "min_da3_feature_similarity": 0.05,
        "min_anchor_for_da3": -1.0,
        "support_weak_threshold": 0.80,
        "topk_per_object": 1,
        "use_veto": True,
        "max_pair_risk": 0.30,
        "max_component_object_degree": 4,
        "broad_support_veto": True,
        "broad_support_min_broad_rate": 0.70,
        "broad_support_max_object_like_rate": 0.50,
        "broad_support_min_support_mean": 1000.0,
    },
    {
        "variant_id": "R2_4VETO_V2_lowrisk030_anchorpath065_da3proposal020",
        "kind": "anchor_da3_proposal",
        "is_control": False,
        "anchor_threshold": 0.65,
        "da3_threshold": 0.20,
        "min_da3_bridge_components": 1,
        "min_da3_feature_similarity": 0.05,
        "min_anchor_for_da3": 0.65,
        "support_weak_threshold": 0.80,
        "topk_per_object": 1,
        "use_veto": True,
        "max_pair_risk": 0.30,
        "max_component_object_degree": 4,
        "broad_support_veto": True,
        "broad_support_min_broad_rate": 0.70,
        "broad_support_max_object_like_rate": 0.50,
        "broad_support_min_support_mean": 1000.0,
    },
    {
        "variant_id": "R2_4VETO_R1_shuffled_da3_for_V1",
        "kind": "anchor_da3_proposal",
        "is_control": True,
        "control_for": "R2_4VETO_V1_lowrisk030_da3proposal035_top1",
        "anchor_threshold": 0.65,
        "da3_threshold": 0.35,
        "min_da3_bridge_components": 1,
        "min_da3_feature_similarity": 0.05,
        "min_anchor_for_da3": -1.0,
        "support_weak_threshold": 0.80,
        "topk_per_object": 1,
        "use_veto": True,
        "shuffle_da3": True,
        "max_pair_risk": 0.30,
        "max_component_object_degree": 4,
        "broad_support_veto": True,
        "broad_support_min_broad_rate": 0.70,
        "broad_support_max_object_like_rate": 0.50,
        "broad_support_min_support_mean": 1000.0,
    },
    {
        "variant_id": "R2_4VETO_R2_shuffled_da3_for_V2",
        "kind": "anchor_da3_proposal",
        "is_control": True,
        "control_for": "R2_4VETO_V2_lowrisk030_anchorpath065_da3proposal020",
        "anchor_threshold": 0.65,
        "da3_threshold": 0.20,
        "min_da3_bridge_components": 1,
        "min_da3_feature_similarity": 0.05,
        "min_anchor_for_da3": 0.65,
        "support_weak_threshold": 0.80,
        "topk_per_object": 1,
        "use_veto": True,
        "shuffle_da3": True,
        "max_pair_risk": 0.30,
        "max_component_object_degree": 4,
        "broad_support_veto": True,
        "broad_support_min_broad_rate": 0.70,
        "broad_support_max_object_like_rate": 0.50,
        "broad_support_min_support_mean": 1000.0,
    },
]

VARIANT_FAMILIES = {
    "object_specific": OBJECT_SPECIFIC_VARIANTS,
    "veto_priority": VETO_PRIORITY_VARIANTS,
}


def _write_csv_local(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12 or not np.isfinite(norm):
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32, copy=False)


def _obs_id(scene: str, frame_id: int, mask_id: int) -> str:
    return f"{scene}:{int(frame_id)}:{int(mask_id)}"


def _load_phase3_features(phase3_root: Path) -> dict[str, dict[str, Any]]:
    payload = torch.load(phase3_root / "role_extended_mask_feature.pt", map_location="cpu", weights_only=False)
    out: dict[str, dict[str, Any]] = {}
    for scene, scene_payload in payload["scenes"].items():
        out[str(scene)] = {
            "mask_observation_id": [str(v) for v in scene_payload["mask_observation_id"]],
            "mask_feature": np.asarray(scene_payload["mask_feature"], dtype=np.float32),
            "baseline_feature": np.asarray(scene_payload["baseline_feature"], dtype=np.float32),
        }
    return out


def _object_r2_features(
    base: pd.DataFrame,
    phase3_features: dict[str, dict[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for (scene, oid), group in base.groupby(["scene_id", "mv_object_id"], sort=False):
        scene = str(scene)
        oid = str(oid)
        payload = phase3_features.get(scene, {})
        obs_to_idx = {obs: idx for idx, obs in enumerate(payload.get("mask_observation_id", []))}
        idxs: list[int] = []
        for row in group.to_dict("records"):
            obs = _obs_id(scene, int(row["frame_id"]), int(row["selected_mask_id"]))
            if obs in obs_to_idx:
                idxs.append(int(obs_to_idx[obs]))
        if idxs:
            mask_feat = np.asarray(payload["mask_feature"][np.asarray(sorted(set(idxs)), dtype=np.int64)], dtype=np.float32).mean(axis=0)
            base_feat = np.asarray(payload["baseline_feature"][np.asarray(sorted(set(idxs)), dtype=np.int64)], dtype=np.float32).mean(axis=0)
        else:
            mask_dim = int(np.asarray(payload.get("mask_feature", np.zeros((1, 1), dtype=np.float32))).shape[1])
            base_dim = int(np.asarray(payload.get("baseline_feature", np.zeros((1, 1), dtype=np.float32))).shape[1])
            mask_feat = np.zeros((mask_dim,), dtype=np.float32)
            base_feat = np.zeros((base_dim,), dtype=np.float32)
        out.setdefault(scene, {})[oid] = {
            "da3_feature": _normalize(mask_feat),
            "support_feature": _normalize(base_feat),
            "matched_r2_mask_count": np.asarray([len(set(idxs))], dtype=np.int64),
        }
    return out


def _best_phase2_variant_by_scene(phase2_root: Path) -> dict[str, str]:
    summary = _read_json(phase2_root / "summary.json")
    return {
        str(scene): str(row["variant_id"])
        for scene, row in summary.get("best_by_scene", {}).items()
    }


def _load_candidate_risk(phase1_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    df = pd.read_csv(phase1_root / "candidate_universe_rows.csv")
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in df.to_dict("records"):
        out[str(row["scene_id"])][str(row["mask_observation_id"])] = {
            "risk_score": float(row.get("risk_score", 0.0) or 0.0),
            "semantic_broad_flag": _as_bool(row.get("semantic_broad_flag", False)),
            "candidate_delta_type": str(row.get("candidate_delta_type", "")),
            "A_anchor_support_count": int(row.get("A_anchor_support_count", 0) or 0),
            "S_support_count": int(row.get("S_support_count", 0) or 0),
            "V_veto_support_count": int(row.get("V_veto_support_count", 0) or 0),
        }
    return out


def _base_obs_to_objects(base: pd.DataFrame) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for row in base.to_dict("records"):
        scene = str(row["scene_id"])
        obs = _obs_id(scene, int(row["frame_id"]), int(row["selected_mask_id"]))
        out[obs].add(str(row["mv_object_id"]))
    return out


def _load_component_meta(phase2_root: Path, best_variant: dict[str, str]) -> dict[tuple[str, int], dict[str, Any]]:
    usecols = [
        "scene_id",
        "variant_id",
        "component_id",
        "component_seed_gaussian_count",
        "component_alpha_mean",
        "component_density_log_mean",
        "component_quality_score_mean",
        "component_semantic_risk_mean",
        "component_semantic_risk_p90",
        "is_clean_component",
    ]
    df = pd.read_csv(phase2_root / "da3_semsoft_component_rows.csv", usecols=usecols)
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for row in df.to_dict("records"):
        scene = str(row["scene_id"])
        if str(row["variant_id"]) != best_variant.get(scene, ""):
            continue
        if not _as_bool(row.get("is_clean_component", False)):
            continue
        rows[(scene, int(row["component_id"]))] = row
    return rows


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((str(a), str(b))))  # type: ignore[return-value]


def _load_da3_pair_stats(
    *,
    phase2_root: Path,
    best_variant: dict[str, str],
    base: pd.DataFrame,
    candidate_risk: dict[str, dict[str, dict[str, Any]]],
) -> tuple[dict[str, dict[tuple[str, str], dict[str, Any]]], dict[str, float]]:
    incidence = pd.read_parquet(phase2_root / "da3_semsoft_primitive_incidence_rows.parquet")
    obs_to_objects = _base_obs_to_objects(base)
    component_meta = _load_component_meta(phase2_root, best_variant)
    pair_stats: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    norm_values: dict[str, list[float]] = defaultdict(list)
    for scene, scene_rows in incidence.groupby("scene_id", sort=True):
        scene = str(scene)
        best_id = best_variant.get(scene, "")
        scene_rows = scene_rows[
            (scene_rows["variant_id"].astype(str) == best_id)
            & scene_rows["emitted_to_support"].map(_as_bool)
        ].copy()
        for component_id, comp_rows in scene_rows.groupby("component_id", sort=False):
            cid = int(component_id)
            meta = component_meta.get((scene, cid))
            if meta is None:
                continue
            object_touch: dict[str, dict[str, Any]] = {}
            for row in comp_rows.to_dict("records"):
                obs = str(row["mask_observation_id"])
                objects = obs_to_objects.get(obs, set())
                if not objects:
                    continue
                risk_info = candidate_risk.get(scene, {}).get(obs, {})
                risk = float(row.get("risk_score", risk_info.get("risk_score", 0.0)) or 0.0)
                for oid in objects:
                    item = object_touch.setdefault(
                        oid,
                        {
                            "gaussian_count": 0.0,
                            "risk_values": [],
                            "raw_veto_hits": 0,
                            "reliable_veto_conflicts": 0,
                            "example_obs": [],
                        },
                    )
                    item["gaussian_count"] += float(row.get("component_mask_gaussian_count", 0.0) or 0.0)
                    item["risk_values"].append(risk)
                    item["raw_veto_hits"] += int(_as_bool(row.get("V_veto_hit", False)))
                    item["reliable_veto_conflicts"] += int(_as_bool(row.get("V_veto_reliable_conflict", False)))
                    if len(item["example_obs"]) < 4:
                        item["example_obs"].append(obs)
            if len(object_touch) < 2:
                continue
            degree = len(object_touch)
            quality = float(meta.get("component_quality_score_mean", 1.0) or 1.0)
            for oid_a, oid_b in combinations(sorted(object_touch), 2):
                a = object_touch[oid_a]
                b = object_touch[oid_b]
                key = _pair_key(oid_a, oid_b)
                item = pair_stats[scene].setdefault(
                    key,
                    {
                        "component_bridge_count": 0,
                        "da3_bridge_score_raw": 0.0,
                        "da3_bridge_score_object_specific_raw": 0.0,
                        "component_seed_path_count": 0,
                        "object_specific_seed_path_count": 0,
                        "reliable_veto_conflict_count": 0,
                        "raw_veto_hit_count": 0,
                        "component_object_degree_max": 0,
                        "component_object_degree_min": 999999,
                        "object_specific_component_count": 0,
                        "component_pair_concentration_max": 0.0,
                        "component_pair_concentration_values": [],
                        "object_specific_pair_concentration_max": 0.0,
                        "risk_values": [],
                        "object_specific_risk_values": [],
                        "alpha_values": [],
                        "density_log_values": [],
                        "quality_values": [],
                        "example_component_ids": [],
                        "example_mask_observation_ids": [],
                    },
                )
                component_total_mass = float(sum(float(v["gaussian_count"]) for v in object_touch.values()))
                pair_mass_raw = float(a["gaussian_count"]) + float(b["gaussian_count"])
                pair_concentration = pair_mass_raw / max(component_total_mass, 1e-12)
                mass = min(float(a["gaussian_count"]), float(b["gaussian_count"]))
                bridge = float(np.sqrt(max(mass, 0.0))) * max(quality, 0.01)
                risks = [float(v) for v in a["risk_values"]] + [float(v) for v in b["risk_values"]]
                risk_max = max(risks) if risks else 0.0
                bridge *= max(1.0 - risk_max, 0.02)
                pair_reliable_veto_count = int(a["reliable_veto_conflicts"] + b["reliable_veto_conflicts"])
                object_specific = int(degree) <= 4 and pair_reliable_veto_count == 0
                item["component_bridge_count"] += 1
                item["da3_bridge_score_raw"] += bridge
                seed_path = int(float(meta.get("component_seed_gaussian_count", 0.0) or 0.0) > 0.0)
                item["component_seed_path_count"] += seed_path
                item["reliable_veto_conflict_count"] += pair_reliable_veto_count
                item["raw_veto_hit_count"] += int(a["raw_veto_hits"] + b["raw_veto_hits"])
                item["component_object_degree_max"] = max(int(item["component_object_degree_max"]), int(degree))
                item["component_object_degree_min"] = min(int(item["component_object_degree_min"]), int(degree))
                item["component_pair_concentration_max"] = max(float(item["component_pair_concentration_max"]), float(pair_concentration))
                item["component_pair_concentration_values"].append(float(pair_concentration))
                item["risk_values"].extend(risks)
                if object_specific:
                    item["object_specific_component_count"] += 1
                    item["da3_bridge_score_object_specific_raw"] += bridge
                    item["object_specific_seed_path_count"] += seed_path
                    item["object_specific_pair_concentration_max"] = max(float(item["object_specific_pair_concentration_max"]), float(pair_concentration))
                    item["object_specific_risk_values"].extend(risks)
                item["alpha_values"].append(float(meta.get("component_alpha_mean", 0.0) or 0.0))
                item["density_log_values"].append(float(meta.get("component_density_log_mean", 0.0) or 0.0))
                item["quality_values"].append(quality)
                if len(item["example_component_ids"]) < 5:
                    item["example_component_ids"].append(str(cid))
                if len(item["example_mask_observation_ids"]) < 5:
                    item["example_mask_observation_ids"].extend((a["example_obs"] + b["example_obs"])[: max(0, 5 - len(item["example_mask_observation_ids"]))])
    norm_by_scene: dict[str, float] = {}
    for scene, stats in pair_stats.items():
        for item in stats.values():
            values = [float(v) for v in item["risk_values"]]
            item["risk_mean"] = float(np.mean(values)) if values else 0.0
            item["risk_max"] = float(np.max(values)) if values else 0.0
            specific_values = [float(v) for v in item["object_specific_risk_values"]]
            item["object_specific_risk_mean"] = float(np.mean(specific_values)) if specific_values else item["risk_mean"]
            item["object_specific_risk_max"] = float(np.max(specific_values)) if specific_values else item["risk_max"]
            item["component_alpha_mean"] = float(np.mean(item["alpha_values"])) if item["alpha_values"] else 0.0
            item["component_density_log_mean"] = float(np.mean(item["density_log_values"])) if item["density_log_values"] else 0.0
            item["component_quality_score_mean"] = float(np.mean(item["quality_values"])) if item["quality_values"] else 0.0
            item["component_pair_concentration_mean"] = float(np.mean(item["component_pair_concentration_values"])) if item["component_pair_concentration_values"] else 0.0
            norm_values[scene].append(float(item["da3_bridge_score_object_specific_raw"]))
        positive = np.asarray([v for v in norm_values[scene] if v > 0.0], dtype=np.float32)
        norm_by_scene[scene] = float(np.percentile(positive, 95)) if positive.size else 1.0
        if norm_by_scene[scene] <= 1e-12:
            norm_by_scene[scene] = 1.0
        for item in stats.values():
            item["da3_bridge_score_all_components"] = float(min(1.0, float(item["da3_bridge_score_raw"]) / norm_by_scene[scene]))
            item["da3_bridge_score"] = float(min(1.0, float(item["da3_bridge_score_object_specific_raw"]) / norm_by_scene[scene]))
            item["example_component_ids"] = ";".join(item["example_component_ids"])
            item["example_mask_observation_ids"] = ";".join(item["example_mask_observation_ids"][:5])
    return pair_stats, norm_by_scene


def _component_conflict(
    *,
    uf: UnionFind,
    object_ids: list[str],
    object_a: str,
    object_b: str,
    by_object: dict[str, pd.DataFrame],
) -> tuple[bool, str]:
    root_a = uf.find(object_a)
    root_b = uf.find(object_b)
    if root_a == root_b:
        return False, ""
    members_a = [oid for oid in object_ids if uf.find(oid) == root_a]
    members_b = [oid for oid in object_ids if uf.find(oid) == root_b]
    for a in members_a:
        for b in members_b:
            if _specific_conflict(by_object[a], by_object[b]):
                return True, "component_specific_same_frame_conflict"
    return False, ""


def _candidate_edges(
    *,
    scene: str,
    scene_base: pd.DataFrame,
    phase5_object_features: dict[str, np.ndarray],
    r2_object_features: dict[str, dict[str, np.ndarray]],
    da3_pair_stats: dict[tuple[str, str], dict[str, Any]],
    norm_basis: float,
    variant: dict[str, Any],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    object_ids = sorted(scene_base["mv_object_id"].astype(str).unique().tolist())
    if str(variant["kind"]) == "replay":
        return []
    anchor = np.stack([phase5_object_features[oid] for oid in object_ids], axis=0)
    da3 = np.stack([r2_object_features.get(oid, {}).get("da3_feature", np.zeros((1,), dtype=np.float32)) for oid in object_ids], axis=0)
    support = np.stack([r2_object_features.get(oid, {}).get("support_feature", np.zeros((1,), dtype=np.float32)) for oid in object_ids], axis=0)
    da3_lookup_ids = list(object_ids)
    if bool(variant.get("shuffle_da3", False)) and len(object_ids) > 1:
        da3 = da3[rng.permutation(len(object_ids))]
        da3_lookup_ids = rng.permutation(object_ids).tolist()
    sim_anchor = anchor @ anchor.T
    sim_da3_feature = da3 @ da3.T
    sim_support = support @ support.T
    by_object = {oid: scene_base[scene_base["mv_object_id"].astype(str) == oid] for oid in object_ids}
    rows: list[dict[str, Any]] = []
    for i, oid_a in enumerate(object_ids[:-1]):
        for j in range(i + 1, len(object_ids)):
            oid_b = object_ids[j]
            pair = _pair_key(oid_a, oid_b)
            da3_pair = _pair_key(str(da3_lookup_ids[i]), str(da3_lookup_ids[j]))
            stats = da3_pair_stats.get(da3_pair, {})
            k_anchor = float(sim_anchor[i, j])
            k_da3_feature = float(sim_da3_feature[i, j])
            k_support = float(sim_support[i, j])
            bridge_score = float(stats.get("da3_bridge_score", 0.0))
            k_da3 = float(0.65 * bridge_score + 0.35 * max(k_da3_feature, 0.0))
            specific = _specific_conflict(by_object[oid_a], by_object[oid_b])
            broad_veto = _broad_support_risk(by_object[oid_a], variant) or _broad_support_risk(by_object[oid_b], variant)
            reliable_veto = int(stats.get("reliable_veto_conflict_count", 0)) > 0
            pair_risk = float(stats.get("object_specific_risk_max", stats.get("risk_max", 0.0)))
            high_risk = bool(pair_risk > float(variant.get("max_pair_risk", 1.0)))
            object_specific_count = int(stats.get("object_specific_component_count", 0))
            high_degree = object_specific_count <= 0 and int(stats.get("component_object_degree_max", 0)) > int(variant.get("max_component_object_degree", 999999))
            hard_block = bool(specific or (bool(variant.get("use_veto", False)) and (broad_veto or reliable_veto or high_risk or high_degree)))
            anchor_positive = k_anchor >= float(variant.get("anchor_threshold", 1.01))
            da3_positive = (
                object_specific_count >= int(variant.get("min_da3_bridge_components", 1))
                and int(stats.get("object_specific_seed_path_count", 0)) > 0
                and k_da3 >= float(variant.get("da3_threshold", 1.01))
                and k_da3_feature >= float(variant.get("min_da3_feature_similarity", -1.0))
                and k_anchor >= float(variant.get("min_anchor_for_da3", -1.0))
            )
            weak_support = (not anchor_positive and not da3_positive) and (k_support >= float(variant.get("support_weak_threshold", 1.01)))
            if not (anchor_positive or da3_positive or weak_support or hard_block):
                continue
            source = "none"
            if anchor_positive and da3_positive:
                source = "anchor_plus_DA3soft"
            elif anchor_positive:
                source = "anchor"
            elif da3_positive:
                source = "DA3soft"
            elif weak_support:
                source = "weak_support"
            rows.append(
                {
                    "object_a": oid_a,
                    "object_b": oid_b,
                    "object_a_da3_lookup": str(da3_pair[0]),
                    "object_b_da3_lookup": str(da3_pair[1]),
                    "K_anchor": k_anchor,
                    "K_DA3_soft": k_da3,
                    "K_DA3_bridge": bridge_score,
                    "K_DA3_feature": k_da3_feature,
                    "K_support": k_support,
                    "K_total": float(k_anchor + k_da3 + 0.10 * max(k_support, 0.0) - pair_risk),
                    "edge_source": source,
                    "anchor_positive": bool(anchor_positive),
                    "DA3soft_positive": bool(da3_positive),
                    "weak_support_only": bool(weak_support),
                    "hard_block": hard_block,
                    "specific_conflict": bool(specific),
                    "broad_support_veto": bool(broad_veto),
                    "V_veto_reliable_conflict": bool(reliable_veto),
                    "risk_block": bool(high_risk),
                    "component_degree_block": bool(high_degree),
                    "Risk_ab": pair_risk,
                    "DA3_bridge_component_count": int(stats.get("component_bridge_count", 0)),
                    "DA3_object_specific_component_count": object_specific_count,
                    "DA3_bridge_score_raw": float(stats.get("da3_bridge_score_raw", 0.0)),
                    "DA3_bridge_score_object_specific_raw": float(stats.get("da3_bridge_score_object_specific_raw", 0.0)),
                    "DA3_bridge_score_all_components": float(stats.get("da3_bridge_score_all_components", 0.0)),
                    "DA3_bridge_score_norm_basis_p95": norm_basis,
                    "DA3_anchor_seed_path_count": int(stats.get("component_seed_path_count", 0)),
                    "DA3_object_specific_seed_path_count": int(stats.get("object_specific_seed_path_count", 0)),
                    "DA3_raw_veto_hit_count": int(stats.get("raw_veto_hit_count", 0)),
                    "DA3_reliable_veto_conflict_count": int(stats.get("reliable_veto_conflict_count", 0)),
                    "DA3_component_object_degree_max": int(stats.get("component_object_degree_max", 0)),
                    "DA3_component_object_degree_min": int(stats.get("component_object_degree_min", 0)),
                    "DA3_component_pair_concentration_max": float(stats.get("component_pair_concentration_max", 0.0)),
                    "DA3_component_pair_concentration_mean": float(stats.get("component_pair_concentration_mean", 0.0)),
                    "DA3_object_specific_pair_concentration_max": float(stats.get("object_specific_pair_concentration_max", 0.0)),
                    "DA3_all_component_risk_max": float(stats.get("risk_max", 0.0)),
                    "DA3_object_specific_risk_max": float(stats.get("object_specific_risk_max", 0.0)),
                    "DA3_component_alpha_mean": float(stats.get("component_alpha_mean", 0.0)),
                    "DA3_component_density_log_mean": float(stats.get("component_density_log_mean", 0.0)),
                    "DA3_component_quality_score_mean": float(stats.get("component_quality_score_mean", 0.0)),
                    "DA3_example_component_ids": str(stats.get("example_component_ids", "")),
                    "DA3_example_mask_observation_ids": str(stats.get("example_mask_observation_ids", "")),
                    "shuffle_da3": bool(variant.get("shuffle_da3", False)),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    rows.sort(
        key=lambda row: (
            int(bool(row["anchor_positive"])),
            float(row["K_anchor"]),
            int(bool(row["DA3soft_positive"])),
            float(row["K_DA3_soft"]),
            float(row["K_support"]),
        ),
        reverse=True,
    )
    return rows


def _mark_topk(edges: list[dict[str, Any]], topk: int) -> None:
    if topk <= 0:
        for edge in edges:
            edge["selected_by_topk"] = bool(edge.get("anchor_positive", False) or edge.get("DA3soft_positive", False))
        return
    by_object_anchor: dict[str, list[int]] = defaultdict(list)
    by_object_da3: dict[str, list[int]] = defaultdict(list)
    for idx, edge in enumerate(edges):
        edge["selected_by_topk"] = False
        if bool(edge.get("anchor_positive", False)):
            by_object_anchor[str(edge["object_a"])].append(idx)
            by_object_anchor[str(edge["object_b"])].append(idx)
        if bool(edge.get("DA3soft_positive", False)):
            by_object_da3[str(edge["object_a"])].append(idx)
            by_object_da3[str(edge["object_b"])].append(idx)
        edge["selected_by_topk"] = False
    selected: set[int] = set()
    for indices in by_object_anchor.values():
        ordered = sorted(indices, key=lambda idx: (float(edges[idx]["K_anchor"]), float(edges[idx]["K_support"])), reverse=True)
        selected.update(ordered[:topk])
    for indices in by_object_da3.values():
        ordered = sorted(indices, key=lambda idx: (float(edges[idx]["K_DA3_soft"]), float(edges[idx]["K_DA3_bridge"])), reverse=True)
        selected.update(ordered[:topk])
    for idx in selected:
        edges[idx]["selected_by_topk"] = True


def _materialize_variant(
    *,
    base: pd.DataFrame,
    phase5_object_features: dict[str, np.ndarray],
    r2_object_features: dict[str, dict[str, np.ndarray]],
    da3_pair_stats_by_scene: dict[str, dict[tuple[str, str], dict[str, Any]]],
    da3_norm_by_scene: dict[str, float],
    variant: dict[str, Any],
    chunk_id: str,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    scene_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    edge_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(10324)
    structure = {
        "added_anchor_edge_count": 0,
        "added_DA3soft_edge_count": 0,
        "blocked_veto_edge_count": 0,
        "weak_support_edge_count": 0,
        "da3soft_candidate_edge_count": 0,
        "anchor_candidate_edge_count": 0,
        "accepted_specific_conflict_count": 0,
        "cluster_count": 0,
        "object_count_delta_vs_replay": 0,
        "frame_mask_count_delta_vs_replay": 0,
    }
    for scene, scene_base in base.groupby("scene_id", sort=True):
        scene = str(scene)
        object_ids = sorted(scene_base["mv_object_id"].astype(str).unique().tolist())
        by_object = {oid: scene_base[scene_base["mv_object_id"].astype(str) == oid] for oid in object_ids}
        uf = UnionFind(object_ids)
        edges = _candidate_edges(
            scene=scene,
            scene_base=scene_base,
            phase5_object_features=phase5_object_features,
            r2_object_features=r2_object_features.get(scene, {}),
            da3_pair_stats=da3_pair_stats_by_scene.get(scene, {}),
            norm_basis=float(da3_norm_by_scene.get(scene, 1.0)),
            variant=variant,
            rng=rng,
        )
        _mark_topk(edges, int(variant.get("topk_per_object", 0)))
        for rank, edge in enumerate(edges):
            accepted = False
            reason = ""
            if bool(edge["hard_block"]):
                if bool(edge.get("specific_conflict", False)):
                    reason = "specific_same_frame_conflict"
                elif bool(edge.get("broad_support_veto", False)):
                    reason = "broad_support_veto"
                elif bool(edge.get("V_veto_reliable_conflict", False)):
                    reason = "reliable_veto_conflict"
                elif bool(edge.get("risk_block", False)):
                    reason = "risk_ab_over_limit"
                elif bool(edge.get("component_degree_block", False)):
                    reason = "da3_component_object_degree_over_limit"
                else:
                    reason = "hard_veto"
                structure["blocked_veto_edge_count"] += 1
            elif bool(edge["weak_support_only"]):
                reason = "weak_support_only_no_union"
                structure["weak_support_edge_count"] += 1
            elif not bool(edge.get("selected_by_topk", False)):
                reason = "not_selected_by_topk"
            elif uf.find(str(edge["object_a"])) != uf.find(str(edge["object_b"])):
                conflict, conflict_reason = _component_conflict(
                    uf=uf,
                    object_ids=object_ids,
                    object_a=str(edge["object_a"]),
                    object_b=str(edge["object_b"]),
                    by_object=by_object,
                )
                if conflict:
                    reason = conflict_reason
                    structure["blocked_veto_edge_count"] += 1
                else:
                    uf.union(str(edge["object_a"]), str(edge["object_b"]))
                    accepted = True
                    structure["added_anchor_edge_count"] += int(bool(edge.get("anchor_positive", False)))
                    structure["added_DA3soft_edge_count"] += int(bool(edge.get("DA3soft_positive", False)))
                    structure["accepted_specific_conflict_count"] += int(bool(edge.get("specific_conflict", False)))
            structure["da3soft_candidate_edge_count"] += int(bool(edge.get("DA3soft_positive", False)))
            structure["anchor_candidate_edge_count"] += int(bool(edge.get("anchor_positive", False)))
            edge_rows.append(
                {
                    "schema_version": "stream4d_v103_r2_phase4_scaffold_edge_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": str(variant["variant_id"]),
                    "scene_id": scene,
                    "edge_rank": int(rank),
                    "accepted_union": bool(accepted),
                    "reject_reason": reason,
                    "selected_by_topk": bool(edge.get("selected_by_topk", False)),
                    **edge,
                }
            )
        components: dict[str, list[str]] = defaultdict(list)
        for oid in object_ids:
            components[uf.find(oid)].append(oid)
        structure["cluster_count"] += len(components)
        structure["object_count_delta_vs_replay"] += int(len(components) - len(object_ids))
        for comp_idx, members in enumerate(sorted(components.values(), key=lambda v: (-len(v), v[0]))):
            comp_rows = scene_base[scene_base["mv_object_id"].astype(str).isin(members)].copy()
            object_id = f"{variant['variant_id']}:{scene}:{chunk_id}:r2p4_{comp_idx:05d}"
            frames = sorted(comp_rows["frame_id"].astype(int).unique().tolist())
            base_score = float(comp_rows.get("score", pd.Series([0.0])).max())
            object_rows.append(
                {
                    "schema_version": "stream4d_v103_r2_phase4_object_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": str(variant["variant_id"]),
                    "scene_id": scene,
                    "mv_object_id": object_id,
                    "source_object_ids": json.dumps(members, sort_keys=True),
                    "source_object_count": int(len(members)),
                    "frame_count": int(len(frames)),
                    "object_score": base_score,
                    "score_policy": "max_f2_score_no_da3_replacement",
                    "selected_broad_rate": float(comp_rows["selected_mask_is_broad"].astype(bool).mean()),
                    "selected_object_like_rate": float(comp_rows["selected_mask_is_object_like"].astype(bool).mean()),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
            for _frame, group in comp_rows.groupby("frame_id", sort=True):
                best = sorted(
                    group.to_dict("records"),
                    key=lambda row: (
                        -float(row.get("score", 0.0)),
                        -int(row.get("phase5_support_count", 0)),
                        -int(bool(row.get("selected_mask_is_object_like", False))),
                        int(bool(row.get("selected_mask_is_broad", True))),
                        int(row.get("selected_mask_id", 0)),
                    ),
                )[0]
                scene_rows[scene].append(
                    {
                        "schema_version": "stream4d_v103_r2_phase4_object_frame_mask_row_v1",
                        "phase_id": PHASE_ID,
                        "variant_id": str(variant["variant_id"]),
                        "mv_object_id": object_id,
                        "object_id": object_id,
                        "scene_id": scene,
                        "chunk_id": str(best.get("chunk_id", chunk_id)),
                        "window_id": str(best.get("window_id", chunk_id)),
                        "frame_local_index": int(best["frame_local_index"]),
                        "frame_id": int(best["frame_id"]),
                        "selected_mask_id": int(best["selected_mask_id"]),
                        "mask_id_or_generated_id": int(best["mask_id_or_generated_id"]),
                        "object_score": base_score,
                        "score": base_score,
                        "support_count": int(best.get("phase5_support_count", 0) or 0),
                        "selected_mask_is_broad": bool(best.get("selected_mask_is_broad", False)),
                        "selected_mask_is_object_like": bool(best.get("selected_mask_is_object_like", False)),
                        "node_policy": "f2_overlap3_object_skeleton",
                        "emit_policy": "component_wta_by_f2_score_support",
                        "readout_mode": "r2_phase4_da3_semsoft_edge_intervention",
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )
    structure["frame_mask_count_delta_vs_replay"] = int(sum(len(v) for v in scene_rows.values()) - len(base))
    return scene_rows, edge_rows, object_rows, structure


def _control_metric_rows(metric_rows: list[dict[str, Any]], variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(row["variant_id"]): row for row in metric_rows}
    rows: list[dict[str, Any]] = []
    for variant in variants:
        if bool(variant.get("is_control", False)):
            continue
        real = by_id.get(str(variant["variant_id"]))
        if real is None:
            continue
        for control in variants:
            if str(control.get("control_for", "")) != str(variant["variant_id"]):
                continue
            ctrl = by_id.get(str(control["variant_id"]))
            if ctrl is None:
                continue
            rows.append(
                {
                    "schema_version": "stream4d_v103_r2_phase4_control_metric_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": str(variant["variant_id"]),
                    "control_variant_id": str(control["variant_id"]),
                    "control_type": "shuffled_DA3soft",
                    "real_minus_control_MV_AP_window": float(real["MV_AP_window"]) - float(ctrl["MV_AP_window"]),
                    "real_minus_control_MV_AP50_window": float(real["MV_AP50_window"]) - float(ctrl["MV_AP50_window"]),
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_eval": True,
                }
            )
    return rows


def _fragmentation_rows(window_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "stream4d_v103_r2_phase4_fragmentation_diagnostic_row_v1",
            "phase_id": PHASE_ID,
            "variant_id": row["variant_id"],
            "scene_id": row["scene_id"],
            "window_id": row["window_id"],
            "GT_fragment_count_mean_diagnostic": row.get("gt_fragment_count_mean", 0.0),
            "GT_fragment_count_ge2_rate_diagnostic": row.get("gt_fragment_count_ge2_rate", 0.0),
            "uses_gt_for_prediction": False,
            "uses_gt_for_eval": True,
        }
        for row in window_rows
    ]


def _gate_rows(best_real: dict[str, Any], replay: dict[str, Any], best_shuffled: dict[str, Any], accepted_specific_conflict_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    specs = [
        (
            "best_real_MV_AP_window_ge_replay_plus_0p005",
            float(best_real.get("MV_AP_window", 0.0)) >= float(replay.get("MV_AP_window", 0.0)) + 0.005,
            best_real.get("MV_AP_window", 0.0),
            float(replay.get("MV_AP_window", 0.0)) + 0.005,
            "If AP is unchanged and DA3 edge count is small, return to R2-2 clean component/support coverage.",
        ),
        (
            "best_real_MV_AP50_window_ge_replay_plus_0p010",
            float(best_real.get("MV_AP50_window", 0.0)) >= float(replay.get("MV_AP50_window", 0.0)) + 0.010,
            best_real.get("MV_AP50_window", 0.0),
            float(replay.get("MV_AP50_window", 0.0)) + 0.010,
            "If AP50 does not move, inspect whether DA3 edges connect fragmented same-object masks.",
        ),
        (
            "best_real_minus_best_shuffled_MV_AP_window_ge_0p003",
            float(best_real.get("MV_AP_window", 0.0)) - float(best_shuffled.get("MV_AP_window", 0.0)) >= 0.003,
            float(best_real.get("MV_AP_window", 0.0)) - float(best_shuffled.get("MV_AP_window", 0.0)),
            0.003,
            "If real-shuffled margin is small, return to R2-3 and check DA3 support object-specificity.",
        ),
        (
            "same_frame_collision_count_eq_0",
            int(best_real.get("same_frame_collision_count", 0)) == 0,
            int(best_real.get("same_frame_collision_count", 0)),
            0,
            "Stop this family and generate a collision casebook if same-frame collisions appear.",
        ),
        (
            "pixel_collision_rate_eq_0",
            float(best_real.get("pixel_collision_rate", 0.0)) == 0.0,
            best_real.get("pixel_collision_rate", 0.0),
            0.0,
            "Stop this family and generate a collision casebook if pixel collision appears.",
        ),
        (
            "missing_mask_raster_count_eq_0",
            int(best_real.get("missing_mask_raster_count", 0)) == 0,
            int(best_real.get("missing_mask_raster_count", 0)),
            0,
            "Fix mask-root/provenance alignment before interpreting AP.",
        ),
        (
            "accepted_specific_conflict_count_eq_0",
            int(accepted_specific_conflict_count) == 0,
            int(accepted_specific_conflict_count),
            0,
            "Raise veto priority if specific same-frame conflict is accepted.",
        ),
    ]
    gate_rows = [
        {
            "schema_version": "stream4d_v103_r2_phase4_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_name": name,
            "pass": bool(ok),
            "observed": observed,
            "required": required,
            "best_real_variant_id": best_real.get("variant_id", ""),
            "replay_variant_id": replay.get("variant_id", ""),
            "best_shuffled_variant_id": best_shuffled.get("variant_id", ""),
            "repair_direction": repair,
        }
        for name, ok, observed, required, repair in specs
    ]
    failure_rows = [
        {
            "schema_version": "stream4d_v103_r2_phase4_failure_row_v1",
            "phase_id": PHASE_ID,
            "failure_id": row["gate_name"],
            "severity": "blocking",
            "evidence": f"observed={row['observed']} required={row['required']}",
            "repair_direction": row["repair_direction"],
            "uses_gt_for_prediction": False,
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    return gate_rows, failure_rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")
    variants = VARIANT_FAMILIES[str(args.variant_family)]

    phase1_root = _project(args.phase1_root)
    phase2_root = _project(args.phase2_root)
    phase3_root = _project(args.phase3_root)
    phase2_summaries = {
        "scene0011_00": _read_json(_project(args.scene0011_phase2_root) / "summary.json"),
        "scene0050_00": _read_json(_project(args.scene0050_phase2_root) / "summary.json"),
    }
    phase5_payloads = {scene: _load_phase5_scene(_project(args.phase5_root), scene) for scene in phase2_summaries}
    base = _adapt_f2_rows(
        f2_root=_project(args.f2_root),
        phase2_summaries=phase2_summaries,
        phase5_payloads=phase5_payloads,
        dataset_split=str(args.dataset_split),
        chunk_id=str(args.chunk_id),
    )
    object_rows_df, phase5_object_features = _object_tables(base, phase5_payloads)
    phase3_features = _load_phase3_features(phase3_root)
    r2_object_features = _object_r2_features(base, phase3_features)
    best_variant = _best_phase2_variant_by_scene(phase2_root)
    candidate_risk = _load_candidate_risk(phase1_root)
    da3_pair_stats, da3_norm_by_scene = _load_da3_pair_stats(
        phase2_root=phase2_root,
        best_variant=best_variant,
        base=base,
        candidate_risk=candidate_risk,
    )

    metric_rows: list[dict[str, Any]] = []
    window_rows_all: list[dict[str, Any]] = []
    selected_rows_all: list[dict[str, Any]] = []
    scaffold_edge_rows: list[dict[str, Any]] = []
    object_rows_all: list[dict[str, Any]] = []
    for variant in variants:
        scene_rows, edge_rows, object_rows, structure = _materialize_variant(
            base=base,
            phase5_object_features=phase5_object_features,
            r2_object_features=r2_object_features,
            da3_pair_stats_by_scene=da3_pair_stats,
            da3_norm_by_scene=da3_norm_by_scene,
            variant=variant,
            chunk_id=str(args.chunk_id),
        )
        window_rows, aggregate, selected_rows, pixel_collision_count, missing_count, _frame_count = _evaluate_variant(
            variant_id=str(variant["variant_id"]),
            scene_rows=scene_rows,
            phase2_summaries=phase2_summaries,
            min_pred_pixels=int(args.min_pred_pixels),
            min_gt_pixels=int(args.min_gt_pixels),
            use_cupy_iou=not bool(args.disable_cupy_iou),
            cupy_device_id=int(args.cupy_device_id),
        )
        fragment_mean = float(np.mean([float(row.get("gt_fragment_count_mean", 0.0)) for row in window_rows])) if window_rows else 0.0
        aggregate.update(
            {
                "schema_version": "stream4d_v103_r2_phase4_mv_metric_row_v1",
                "phase_id": PHASE_ID,
                "variant_kind": str(variant["kind"]),
                "is_control": bool(variant.get("is_control", False)),
                "control_for": str(variant.get("control_for", "")),
                "anchor_threshold": float(variant.get("anchor_threshold", 0.0)),
                "da3_threshold": float(variant.get("da3_threshold", 0.0)),
                "support_weak_threshold": float(variant.get("support_weak_threshold", 0.0)),
                "topk_per_object": int(variant.get("topk_per_object", 0)),
                "use_veto": bool(variant.get("use_veto", False)),
                "shuffle_da3": bool(variant.get("shuffle_da3", False)),
                "score_policy": "max_f2_score_no_da3_replacement",
                "pixel_collision_count": int(pixel_collision_count),
                "missing_mask_raster_count": int(missing_count),
                "dataset_split": str(args.dataset_split),
                "chunk_id": str(args.chunk_id),
                "metric_scope": f"same_subset_as_v103_phase6_c0001_{args.chunk_id}",
                "fragment_count_mean": fragment_mean,
                "uses_gt_for_prediction": False,
                "uses_gt_for_eval": True,
                "uses_future": False,
                **structure,
            }
        )
        metric_rows.append(aggregate)
        window_rows_all.extend(window_rows)
        selected_rows_all.extend(selected_rows)
        scaffold_edge_rows.extend(edge_rows)
        object_rows_all.extend(object_rows)

    replay_variant_id = str(variants[0]["variant_id"])
    replay = next((row for row in metric_rows if str(row.get("variant_id")) == replay_variant_id), {})
    real_rows = [row for row in metric_rows if not bool(row.get("is_control", False)) and str(row.get("variant_id")) != replay_variant_id]
    shuffled_rows = [row for row in metric_rows if bool(row.get("is_control", False))]
    best_real = max(real_rows, key=lambda row: (float(row.get("MV_AP_window", 0.0)), float(row.get("MV_AP50_window", 0.0))), default={})
    best_shuffled = max(shuffled_rows, key=lambda row: (float(row.get("MV_AP_window", 0.0)), float(row.get("MV_AP50_window", 0.0))), default={})
    best_real_edges = [row for row in scaffold_edge_rows if str(row.get("variant_id")) == str(best_real.get("variant_id", ""))]
    accepted_specific_conflict_count = int(sum(1 for row in best_real_edges if bool(row.get("accepted_union", False)) and bool(row.get("specific_conflict", False))))
    gate_rows, failure_rows = _gate_rows(best_real, replay, best_shuffled, accepted_specific_conflict_count)
    control_rows = _control_metric_rows(metric_rows, variants)
    replay_fragment = float(replay.get("fragment_count_mean", 0.0) or 0.0)
    fragment_count_delta = float(best_real.get("fragment_count_mean", 0.0) or 0.0) - replay_fragment
    subset_baseline = _load_subset_baseline(_project(args.subset_baseline_rows))
    phase2_summary = _read_json(phase2_root / "summary.json")

    _write_csv_local(out / "scaffold_edge_rows.csv", scaffold_edge_rows)
    _write_csv_local(out / "merge_selected_rows.csv", selected_rows_all)
    _write_csv_local(out / "object_frame_mask_rows.csv", selected_rows_all)
    _write_csv_local(out / "object_rows.csv", object_rows_all)
    _write_csv_local(out / "control_metric_rows.csv", control_rows)
    _write_csv_local(out / "mv_metric_rows.csv", metric_rows)
    _write_csv_local(out / "mv_window_metric_rows.csv", window_rows_all)
    _write_csv_local(out / "fragmentation_diagnostic_rows.csv", _fragmentation_rows(window_rows_all))
    _write_csv_local(out / "gate_rows.csv", gate_rows)
    _write_csv_local(out / "failure_rows.csv", failure_rows)

    decision = "PASS_R2_4_LOCAL_EDGE_INTERVENTION" if not failure_rows else "NO_GO_R2_4_LOCAL_EDGE_INTERVENTION_INSUFFICIENT"
    summary = {
        "schema_version": "stream4d_v103_r2_phase4_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "failure_count": len(failure_rows),
        "variant_family": str(args.variant_family),
        "variant_count": len(variants),
        "variant_limit_observed": len(variants) <= 5,
        "best_real_variant_id": best_real.get("variant_id", ""),
        "best_shuffled_variant_id": best_shuffled.get("variant_id", ""),
        "replay_variant_id": replay.get("variant_id", ""),
        "replay_MV_AP_window": replay.get("MV_AP_window", ""),
        "replay_MV_AP50_window": replay.get("MV_AP50_window", ""),
        "best_real_MV_AP_window": best_real.get("MV_AP_window", ""),
        "best_real_MV_AP50_window": best_real.get("MV_AP50_window", ""),
        "best_real_minus_replay_MV_AP_window": float(best_real.get("MV_AP_window", 0.0)) - float(replay.get("MV_AP_window", 0.0)) if replay else "",
        "best_real_minus_replay_MV_AP50_window": float(best_real.get("MV_AP50_window", 0.0)) - float(replay.get("MV_AP50_window", 0.0)) if replay else "",
        "best_real_minus_best_shuffled_MV_AP_window": float(best_real.get("MV_AP_window", 0.0)) - float(best_shuffled.get("MV_AP_window", 0.0)) if best_shuffled else "",
        "same_frame_collision_count": best_real.get("same_frame_collision_count", ""),
        "pixel_collision_rate": best_real.get("pixel_collision_rate", ""),
        "missing_mask_raster_count": best_real.get("missing_mask_raster_count", ""),
        "accepted_specific_conflict_count": accepted_specific_conflict_count,
        "added_anchor_edge_count": int(best_real.get("added_anchor_edge_count", 0) or 0),
        "added_DA3soft_edge_count": int(best_real.get("added_DA3soft_edge_count", 0) or 0),
        "blocked_veto_edge_count": int(best_real.get("blocked_veto_edge_count", 0) or 0),
        "weak_support_edge_count": int(best_real.get("weak_support_edge_count", 0) or 0),
        "fragment_count_delta": fragment_count_delta,
        "da3soft_candidate_edge_count": int(best_real.get("da3soft_candidate_edge_count", 0) or 0),
        "anchor_candidate_edge_count": int(best_real.get("anchor_candidate_edge_count", 0) or 0),
        "da3_pair_object_pair_count": int(sum(len(v) for v in da3_pair_stats.values())),
        "da3_pair_norm_basis_by_scene": da3_norm_by_scene,
        "f2_subset_baseline": subset_baseline,
        "phase1_root": _rel(phase1_root),
        "phase2_root": _rel(phase2_root),
        "phase3_root": _rel(phase3_root),
        "phase2_best_variant_by_scene": best_variant,
        "gaussian_quality_gate": "inherited_from_R2_2_hit_gaussian_alpha_and_density_proxy_quantile_filter",
        "gaussian_quality_alpha_source_by_scene": {
            scene: row.get("gaussian_quality_alpha_source", "")
            for scene, row in phase2_summary.get("best_by_scene", {}).items()
        },
        "gaussian_quality_density_source_by_scene": {
            scene: row.get("gaussian_quality_density_source", "")
            for scene, row in phase2_summary.get("best_by_scene", {}).items()
        },
        "per_mask_topk_high_quality_by_scene": {
            scene: row.get("per_mask_topk_high_quality", "")
            for scene, row in phase2_summary.get("best_by_scene", {}).items()
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
        "truthfulness_note": "R2-4 starts from the c0001 F2 mask-view skeleton, keeps DA3 as high-alpha/high-density semantic-soft primitive support, uses DA3 only as an edge proposal with veto priority, and evaluates AP only after GT-free materialization.",
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "scaffold_edge_rows": _rel(out / "scaffold_edge_rows.csv"),
            "merge_selected_rows": _rel(out / "merge_selected_rows.csv"),
            "object_frame_mask_rows": _rel(out / "object_frame_mask_rows.csv"),
            "object_rows": _rel(out / "object_rows.csv"),
            "control_metric_rows": _rel(out / "control_metric_rows.csv"),
            "mv_metric_rows": _rel(out / "mv_metric_rows.csv"),
            "fragmentation_diagnostic_rows": _rel(out / "fragmentation_diagnostic_rows.csv"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "last_command": _rel(out / "last_command.txt"),
        },
    }
    _write_json(out / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v103 R2-4 scaffolded mask graph DA3-semsoft edge intervention.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--f2-root", default=str(DEFAULT_F2_ROOT))
    parser.add_argument("--phase5-root", default=str(DEFAULT_PHASE5_ROOT))
    parser.add_argument("--phase1-root", default=str(DEFAULT_PHASE1_ROOT))
    parser.add_argument("--phase2-root", default=str(DEFAULT_PHASE2_ROOT))
    parser.add_argument("--phase3-root", default=str(DEFAULT_PHASE3_ROOT))
    parser.add_argument("--variant-family", choices=sorted(VARIANT_FAMILIES), default="object_specific")
    parser.add_argument("--scene0011-phase2-root", default=str(DEFAULT_PHASE2_SCENE0011))
    parser.add_argument("--scene0050-phase2-root", default=str(DEFAULT_PHASE2_SCENE0050))
    parser.add_argument("--subset-baseline-rows", default=str(DEFAULT_SUBSET_BASELINE))
    parser.add_argument("--dataset-split", default="dev")
    parser.add_argument("--chunk-id", default="c0001")
    parser.add_argument("--min-pred-pixels", type=int, default=20)
    parser.add_argument("--min-gt-pixels", type=int, default=20)
    parser.add_argument("--cupy-device-id", type=int, default=0)
    parser.add_argument("--disable-cupy-iou", action="store_true")
    return parser.parse_args()


def main() -> int:
    summary = build(parse_args())
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if str(summary["decision"]).startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
