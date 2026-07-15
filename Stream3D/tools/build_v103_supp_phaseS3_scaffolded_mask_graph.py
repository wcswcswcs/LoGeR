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
    DEFAULT_PHASE2_SCENE0011,
    DEFAULT_PHASE2_SCENE0050,
    _evaluate_variant,
    _jsonable,
    _project,
    _read_json,
    _rel,
    _write_csv,
    _write_json,
)
from build_v103_phase6d_f2_skeleton_affinity_merge import (  # noqa: E402
    DEFAULT_F2_ROOT,
    DEFAULT_SUBSET_BASELINE,
    UnionFind,
    _adapt_f2_rows,
    _load_subset_baseline,
    _specific_conflict,
)


PHASE_ID = "v103_supp_phaseS3_scaffolded_mask_graph"
DEFAULT_OUT = AUDIT_ROOT / "v103_supp_phaseS3_scaffolded_mask_graph"
DEFAULT_PHASES2_ROOT = AUDIT_ROOT / "v103_supp_phaseS2_role_aware_affinity"


VARIANTS = [
    {
        "variant_id": "S3_V0_baseline_skeleton_replay",
        "kind": "baseline",
        "anchor_threshold": 1.01,
        "support_threshold": 1.01,
        "topk_per_object": 0,
        "use_veto": False,
        "is_control": False,
    },
    {
        "variant_id": "S3_V1_anchor_positive_only",
        "kind": "anchor",
        "anchor_threshold": 0.80,
        "support_threshold": 0.0,
        "topk_per_object": 1,
        "use_veto": False,
        "is_control": False,
    },
    {
        "variant_id": "S3_V2_anchor_positive_plus_support_compatibility",
        "kind": "anchor_support",
        "anchor_threshold": 0.75,
        "support_threshold": 0.45,
        "topk_per_object": 1,
        "use_veto": False,
        "is_control": False,
    },
    {
        "variant_id": "S3_V3_anchor_positive_plus_veto_block",
        "kind": "anchor_veto",
        "anchor_threshold": 0.75,
        "support_threshold": 0.0,
        "topk_per_object": 1,
        "use_veto": True,
        "is_control": False,
    },
    {
        "variant_id": "S3_V4_anchor_support_veto_all",
        "kind": "anchor_support_veto",
        "anchor_threshold": 0.70,
        "support_threshold": 0.40,
        "weak_support_threshold": 0.70,
        "topk_per_object": 1,
        "use_veto": True,
        "is_control": False,
    },
    {
        "variant_id": "S3_V5_score_repair_anchor_stability_only",
        "kind": "score_repair",
        "anchor_threshold": 1.01,
        "support_threshold": 1.01,
        "topk_per_object": 0,
        "use_veto": True,
        "score_policy": "anchor_stability_boost",
        "is_control": False,
    },
    {
        "variant_id": "S3_V6_veto_only_score_suppress",
        "kind": "score_repair",
        "anchor_threshold": 1.01,
        "support_threshold": 1.01,
        "topk_per_object": 0,
        "use_veto": True,
        "score_policy": "veto_risk_suppress",
        "is_control": False,
    },
    {
        "variant_id": "S3_V7_direct_pair_rel070_support_veto",
        "kind": "direct_pair",
        "anchor_threshold": 1.01,
        "support_threshold": 0.40,
        "topk_per_object": 1,
        "use_veto": True,
        "direct_pair_support_min_count": 1,
        "direct_pair_reliability_mean_min": 0.70,
        "is_control": False,
    },
    {
        "variant_id": "S3_V8_anchor_or_direct_pair_rel070_veto",
        "kind": "anchor_or_direct_pair",
        "anchor_threshold": 0.75,
        "support_threshold": 0.45,
        "topk_per_object": 1,
        "use_veto": True,
        "direct_pair_support_min_count": 1,
        "direct_pair_reliability_mean_min": 0.70,
        "is_control": False,
    },
    {
        "variant_id": "S3_R0_shuffled_anchor_for_V1",
        "kind": "anchor",
        "anchor_threshold": 0.80,
        "support_threshold": 0.0,
        "topk_per_object": 1,
        "use_veto": False,
        "shuffle_anchor": True,
        "control_for": "S3_V1_anchor_positive_only",
        "is_control": True,
    },
    {
        "variant_id": "S3_R1_shuffled_anchor_for_V2",
        "kind": "anchor_support",
        "anchor_threshold": 0.75,
        "support_threshold": 0.45,
        "topk_per_object": 1,
        "use_veto": False,
        "shuffle_anchor": True,
        "control_for": "S3_V2_anchor_positive_plus_support_compatibility",
        "is_control": True,
    },
    {
        "variant_id": "S3_R2_shuffled_anchor_for_V4",
        "kind": "anchor_support_veto",
        "anchor_threshold": 0.70,
        "support_threshold": 0.40,
        "weak_support_threshold": 0.70,
        "topk_per_object": 1,
        "use_veto": True,
        "shuffle_anchor": True,
        "control_for": "S3_V4_anchor_support_veto_all",
        "is_control": True,
    },
    {
        "variant_id": "S3_R3_role_permutation_support_as_anchor_for_V4",
        "kind": "anchor_support_veto",
        "anchor_threshold": 0.70,
        "support_threshold": 0.40,
        "weak_support_threshold": 0.70,
        "topk_per_object": 1,
        "use_veto": True,
        "role_permutation_support_as_anchor": True,
        "control_for": "S3_V4_anchor_support_veto_all",
        "is_control": True,
    },
    {
        "variant_id": "S3_R4_shuffled_direct_pair_for_V7",
        "kind": "direct_pair",
        "anchor_threshold": 1.01,
        "support_threshold": 0.40,
        "topk_per_object": 1,
        "use_veto": True,
        "direct_pair_support_min_count": 1,
        "direct_pair_reliability_mean_min": 0.70,
        "shuffle_direct_pair": True,
        "control_for": "S3_V7_direct_pair_rel070_support_veto",
        "is_control": True,
    },
    {
        "variant_id": "S3_R5_shuffled_direct_pair_for_V8",
        "kind": "anchor_or_direct_pair",
        "anchor_threshold": 0.75,
        "support_threshold": 0.45,
        "topk_per_object": 1,
        "use_veto": True,
        "direct_pair_support_min_count": 1,
        "direct_pair_reliability_mean_min": 0.70,
        "shuffle_direct_pair": True,
        "control_for": "S3_V8_anchor_or_direct_pair_rel070_veto",
        "is_control": True,
    },
]


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _load_s2_feature(root: Path, filename: str) -> dict[str, dict[str, Any]]:
    payload = torch.load(root / filename, map_location="cpu")
    out: dict[str, dict[str, Any]] = {}
    for scene, scene_payload in payload["scenes"].items():
        feature = scene_payload["feature"].to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
        norm = np.linalg.norm(feature, axis=1, keepdims=True)
        feature = feature / np.maximum(norm, 1e-12)
        feature[~np.isfinite(feature)] = 0.0
        out[str(scene)] = {
            "feature": feature,
            "mask_frame": scene_payload["mask_frame"].cpu().numpy().astype(np.int64),
            "mask_label": scene_payload["mask_label"].cpu().numpy().astype(np.int64),
            "mask_is_broad": scene_payload["mask_is_broad"].cpu().numpy().astype(bool),
            "mask_is_object_like": scene_payload["mask_is_object_like"].cpu().numpy().astype(bool),
            "support_count": scene_payload["support_count"].cpu().numpy().astype(np.int64),
        }
    return out


def _load_s2_payloads(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    anchor = _load_s2_feature(root, "mask_feature_anchor.pt")
    support = _load_s2_feature(root, "mask_feature_support.pt")
    anchor_support = _load_s2_feature(root, "mask_feature_anchor_support.pt")
    scenes = sorted(set(anchor).intersection(support).intersection(anchor_support))
    return {
        scene: {
            "A": anchor[scene],
            "S": support[scene],
            "AS": anchor_support[scene],
        }
        for scene in scenes
    }


def _phase5_index(payload: dict[str, Any]) -> dict[tuple[int, int], int]:
    return {
        (int(frame), int(label)): int(idx)
        for idx, (frame, label) in enumerate(zip(payload["mask_frame"].tolist(), payload["mask_label"].tolist()))
    }


def _obs_id(scene: str, frame_id: int, mask_id: int) -> str:
    return f"{scene}:{int(frame_id)}:{int(mask_id)}"


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _direct_pair_support_by_scene(
    base: pd.DataFrame,
    phase9n_root: Path | None,
) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    if phase9n_root is None:
        return {}
    out: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for scene, scene_base in base.groupby("scene_id", sort=True):
        scene = str(scene)
        pair_path = phase9n_root / scene / "da3_bridge_pair_primitive_rows.csv"
        if not pair_path.exists():
            out[scene] = {}
            continue
        obs_to_objects: dict[str, set[str]] = defaultdict(set)
        for row in scene_base.to_dict("records"):
            obs_to_objects[_obs_id(scene, int(row["frame_id"]), int(row["selected_mask_id"]))].add(str(row["mv_object_id"]))
        rows = pd.read_csv(pair_path)
        stats: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows.to_dict("records"):
            objects_a = obs_to_objects.get(str(row.get("mask_a_observation_id", "")), set())
            objects_b = obs_to_objects.get(str(row.get("mask_b_observation_id", "")), set())
            if not objects_a or not objects_b:
                continue
            for oid_a in objects_a:
                for oid_b in objects_b:
                    if oid_a == oid_b:
                        continue
                    key = tuple(sorted((oid_a, oid_b)))
                    item = stats.setdefault(
                        key,
                        {
                            "count": 0,
                            "reliability_sum": 0.0,
                            "reliability_max": 0.0,
                            "broad_risk_max": 0.0,
                            "diagnostic_same_gt_count": 0,
                            "diagnostic_different_gt_count": 0,
                            "example_pair_ids": [],
                        },
                    )
                    rel = float(row.get("carrier_reliability", row.get("B_ia", 0.0)) or 0.0)
                    broad = float(row.get("carrier_broad_risk", row.get("broad_contamination_score", 0.0)) or 0.0)
                    item["count"] += 1
                    item["reliability_sum"] += rel
                    item["reliability_max"] = max(float(item["reliability_max"]), rel)
                    item["broad_risk_max"] = max(float(item["broad_risk_max"]), broad)
                    item["diagnostic_same_gt_count"] += int(_as_bool(row.get("diagnostic_same_gt", False)))
                    item["diagnostic_different_gt_count"] += int(_as_bool(row.get("diagnostic_different_gt", False)))
                    if len(item["example_pair_ids"]) < 5:
                        item["example_pair_ids"].append(
                            f"{row.get('mask_a_observation_id', '')}|{row.get('mask_b_observation_id', '')}"
                        )
        for item in stats.values():
            count = int(item["count"])
            item["reliability_mean"] = float(item["reliability_sum"]) / max(count, 1)
            item["diagnostic_same_gt_rate"] = float(item["diagnostic_same_gt_count"]) / max(count, 1)
            item["diagnostic_different_gt_rate"] = float(item["diagnostic_different_gt_count"]) / max(count, 1)
        out[scene] = stats
    return out


def _object_feature_tables(
    base: pd.DataFrame,
    s2_payloads: dict[str, dict[str, dict[str, Any]]],
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    object_rows: list[dict[str, Any]] = []
    object_features: dict[str, dict[str, np.ndarray]] = {}
    for (scene, oid), group in base.groupby(["scene_id", "mv_object_id"], sort=False):
        scene = str(scene)
        oid = str(oid)
        idxs = np.asarray(sorted({int(v) for v in group["phase5_mask_index"].tolist() if int(v) >= 0}), dtype=np.int64)
        object_features[oid] = {}
        for role in ["A", "S", "AS"]:
            feature = s2_payloads[scene][role]["feature"]
            if idxs.size:
                centroid = feature[idxs].mean(axis=0)
                centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
            else:
                centroid = np.zeros((feature.shape[1],), dtype=np.float32)
            object_features[oid][role] = centroid.astype(np.float32, copy=False)
        anchor_valid = []
        support_valid = []
        for idx in idxs.tolist():
            anchor_valid.append(float(np.linalg.norm(s2_payloads[scene]["A"]["feature"][idx]) > 0.0))
            support_valid.append(float(np.linalg.norm(s2_payloads[scene]["S"]["feature"][idx]) > 0.0))
        object_rows.append(
            {
                "schema_version": "stream4d_v103_supp_phaseS3_object_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "mv_object_id": oid,
                "frame_count": int(group["frame_id"].nunique()),
                "row_count": int(len(group)),
                "original_score": float(group.get("score", pd.Series([0.0])).max()),
                "selected_broad_rate": float(group["selected_mask_is_broad"].astype(bool).mean()),
                "selected_object_like_rate": float(group["selected_mask_is_object_like"].astype(bool).mean()),
                "s2_mask_match_rate": float(np.mean(group["phase5_mask_index"].astype(int).to_numpy() >= 0)),
                "support_count_mean": float(group["phase5_support_count"].astype(float).mean()),
                "anchor_valid_mask_rate": float(np.mean(anchor_valid)) if anchor_valid else 0.0,
                "support_valid_mask_rate": float(np.mean(support_valid)) if support_valid else 0.0,
                "uses_gt_for_prediction": False,
            }
        )
    return pd.DataFrame(object_rows), object_features


def _load_veto_object_pairs(root: Path, base: pd.DataFrame) -> tuple[set[tuple[str, str]], dict[tuple[str, str], float]]:
    path = root / "veto_pair_rows.parquet"
    if not path.exists():
        return set(), {}
    mask_to_objects: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in base.to_dict("records"):
        idx = int(row.get("phase5_mask_index", -1))
        if idx >= 0:
            mask_to_objects[(str(row["scene_id"]), idx)].add(str(row["mv_object_id"]))
    veto_df = pd.read_parquet(path)
    veto_pairs: set[tuple[str, str]] = set()
    veto_score: dict[tuple[str, str], float] = {}
    for row in veto_df.to_dict("records"):
        scene = str(row["scene_id"])
        objs_a = mask_to_objects.get((scene, int(row["mask_a"])), set())
        objs_b = mask_to_objects.get((scene, int(row["mask_b"])), set())
        for a in objs_a:
            for b in objs_b:
                if a == b:
                    continue
                key = tuple(sorted((a, b)))
                veto_pairs.add(key)
                veto_score[key] = max(float(veto_score.get(key, 0.0)), float(row.get("veto_score", 0.0)))
    return veto_pairs, veto_score


def _candidate_edges(
    *,
    scene_base: pd.DataFrame,
    object_features: dict[str, dict[str, np.ndarray]],
    veto_pairs: set[tuple[str, str]],
    veto_score: dict[tuple[str, str], float],
    direct_pair_support: dict[tuple[str, str], dict[str, Any]],
    variant: dict[str, Any],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if str(variant["kind"]) == "baseline":
        return []
    object_ids = sorted(scene_base["mv_object_id"].astype(str).unique().tolist())
    anchor = np.stack([object_features[oid]["A"] for oid in object_ids], axis=0)
    support = np.stack([object_features[oid]["S"] for oid in object_ids], axis=0)
    if bool(variant.get("role_permutation_support_as_anchor", False)):
        anchor = support.copy()
    if bool(variant.get("shuffle_anchor", False)) and len(object_ids) > 1:
        anchor = anchor[rng.permutation(len(object_ids))]
    direct_lookup_ids = list(object_ids)
    if bool(variant.get("shuffle_direct_pair", False)) and len(object_ids) > 1:
        permuted = rng.permutation(object_ids).tolist()
        direct_lookup_ids = [str(v) for v in permuted]
    sim_a = anchor @ anchor.T
    sim_s = support @ support.T
    threshold_a = float(variant.get("anchor_threshold", 1.01))
    threshold_s = float(variant.get("support_threshold", 1.01))
    weak_s = float(variant.get("weak_support_threshold", 1.01))
    by_object = {oid: scene_base[scene_base["mv_object_id"].astype(str) == oid] for oid in object_ids}
    local_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    weak_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    for i, oid_a in enumerate(object_ids[:-1]):
        for j in range(i + 1, len(object_ids)):
            oid_b = object_ids[j]
            pair = tuple(sorted((oid_a, oid_b)))
            anchor_score = float(sim_a[i, j])
            support_score = float(sim_s[i, j])
            specific = _specific_conflict(by_object[oid_a], by_object[oid_b])
            veto = pair in veto_pairs
            direct_pair = tuple(sorted((direct_lookup_ids[i], direct_lookup_ids[j])))
            direct = direct_pair_support.get(direct_pair, {})
            direct_count = int(direct.get("count", 0))
            direct_reliability_mean = float(direct.get("reliability_mean", 0.0))
            direct_reliability_max = float(direct.get("reliability_max", 0.0))
            direct_required = (
                int(variant.get("direct_pair_support_min_count", 0)) > 0
                or float(variant.get("direct_pair_reliability_mean_min", 0.0)) > 0.0
                or float(variant.get("direct_pair_reliability_max_min", 0.0)) > 0.0
            )
            direct_positive = (
                direct_required
                and direct_count >= int(variant.get("direct_pair_support_min_count", 0))
                and direct_reliability_mean >= float(variant.get("direct_pair_reliability_mean_min", 0.0))
                and direct_reliability_max >= float(variant.get("direct_pair_reliability_max_min", 0.0))
                and support_score >= threshold_s
            )
            hard = bool(specific or (bool(variant.get("use_veto", False)) and veto))
            positive = False
            weak = False
            if str(variant["kind"]) == "score_repair":
                positive = False
            elif str(variant["kind"]) in {"anchor", "anchor_veto"}:
                positive = anchor_score >= threshold_a
            elif str(variant["kind"]) == "direct_pair":
                positive = bool(direct_positive)
            elif str(variant["kind"]) == "anchor_or_direct_pair":
                positive = ((anchor_score >= threshold_a) and (support_score >= threshold_s)) or bool(direct_positive)
            else:
                positive = (anchor_score >= threshold_a) and (support_score >= threshold_s)
                weak = (not positive) and (support_score >= weak_s)
            row = {
                "object_a": oid_a,
                "object_b": oid_b,
                "K_anchor": anchor_score,
                "K_support": support_score,
                "K_anchor_support": 0.5 * (anchor_score + support_score),
                "specific_conflict": bool(specific),
                "veto_hard_cannot_link": bool(veto),
                "veto_score": float(veto_score.get(pair, 0.0)),
                "direct_pair_lookup_object_a": str(direct_pair[0]),
                "direct_pair_lookup_object_b": str(direct_pair[1]),
                "direct_pair_positive": bool(direct_positive),
                "direct_pair_support_count": direct_count,
                "direct_pair_reliability_mean": direct_reliability_mean,
                "direct_pair_reliability_max": direct_reliability_max,
                "direct_pair_broad_risk_max": float(direct.get("broad_risk_max", 0.0)),
                "direct_pair_diagnostic_same_gt_count": int(direct.get("diagnostic_same_gt_count", 0)),
                "direct_pair_diagnostic_different_gt_count": int(direct.get("diagnostic_different_gt_count", 0)),
                "direct_pair_diagnostic_same_gt_rate": float(direct.get("diagnostic_same_gt_rate", 0.0)),
                "direct_pair_diagnostic_different_gt_rate": float(direct.get("diagnostic_different_gt_rate", 0.0)),
                "direct_pair_example_pair_ids": ";".join(direct.get("example_pair_ids", [])),
                "shuffle_direct_pair": bool(variant.get("shuffle_direct_pair", False)),
                "positive_candidate": bool(positive),
                "weak_support_only": bool(weak),
                "hard_block": bool(hard),
            }
            if hard and (positive or weak):
                blocked_rows.append(row)
            elif positive:
                local_by_source[oid_a].append(row)
            elif weak:
                weak_rows.append(row)
    selected: list[dict[str, Any]] = []
    topk = int(variant.get("topk_per_object", 0))
    for rows in local_by_source.values():
        rows.sort(key=lambda row: (float(row["K_anchor_support"]), float(row["K_anchor"])), reverse=True)
        selected.extend(rows[:topk] if topk > 0 else rows)
    selected.extend(weak_rows)
    selected.extend(blocked_rows)
    selected.sort(key=lambda row: (float(row["K_anchor_support"]), float(row["K_anchor"])), reverse=True)
    return selected


def _count_duplicate_claims(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    if df.empty:
        return 0
    grouped = df.groupby(["scene_id", "frame_id", "selected_mask_id"]).size()
    return int(np.sum(np.maximum(grouped.to_numpy(dtype=np.int64) - 1, 0)))


def _cannot_link_violations(
    components: dict[str, list[str]],
    by_object: dict[str, pd.DataFrame],
    veto_pairs: set[tuple[str, str]],
) -> int:
    count = 0
    for members in components.values():
        if len(members) < 2:
            continue
        for a, b in combinations(sorted(members), 2):
            pair = tuple(sorted((a, b)))
            if pair in veto_pairs or _specific_conflict(by_object[a], by_object[b]):
                count += 1
    return int(count)


def _component_conflict(
    *,
    uf: UnionFind,
    object_ids: list[str],
    object_a: str,
    object_b: str,
    by_object: dict[str, pd.DataFrame],
    veto_pairs: set[tuple[str, str]],
) -> tuple[bool, str]:
    root_a = uf.find(object_a)
    root_b = uf.find(object_b)
    if root_a == root_b:
        return False, ""
    members_a = [oid for oid in object_ids if uf.find(oid) == root_a]
    members_b = [oid for oid in object_ids if uf.find(oid) == root_b]
    for a in members_a:
        for b in members_b:
            pair = tuple(sorted((a, b)))
            if pair in veto_pairs:
                return True, "component_veto_cannot_link"
            if _specific_conflict(by_object[a], by_object[b]):
                return True, "component_specific_same_frame_conflict"
    return False, ""


def _materialize_variant(
    *,
    base: pd.DataFrame,
    object_features: dict[str, dict[str, np.ndarray]],
    object_rows: pd.DataFrame,
    veto_pairs: set[tuple[str, str]],
    veto_score: dict[tuple[str, str], float],
    direct_pair_support_by_scene: dict[str, dict[tuple[str, str], dict[str, Any]]],
    variant: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    scene_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    edge_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(10317)
    structure = {
        "baseline_edge_count": 0,
        "added_anchor_edge_count": 0,
        "blocked_veto_edge_count": 0,
        "weak_support_edge_count": 0,
        "cannot_link_violation_count_after_clustering": 0,
        "post_split_count": 0,
        "cluster_count": 0,
        "largest_cluster_ratio": 0.0,
        "object_count_delta_vs_baseline": 0,
        "frame_mask_count_delta_vs_baseline": 0,
    }
    object_info = object_rows.set_index("mv_object_id").to_dict("index") if not object_rows.empty else {}
    for scene, scene_base in base.groupby("scene_id", sort=True):
        object_ids = sorted(scene_base["mv_object_id"].astype(str).unique().tolist())
        by_object = {oid: scene_base[scene_base["mv_object_id"].astype(str) == oid] for oid in object_ids}
        uf = UnionFind(object_ids)
        edges = _candidate_edges(
            scene_base=scene_base,
            object_features=object_features,
            veto_pairs=veto_pairs,
            veto_score=veto_score,
            direct_pair_support=direct_pair_support_by_scene.get(str(scene), {}),
            variant=variant,
            rng=rng,
        )
        for rank, edge in enumerate(edges):
            accepted = False
            reason = ""
            if bool(edge["weak_support_only"]):
                reason = "weak_support_only_no_union"
                structure["weak_support_edge_count"] += 1
            elif bool(edge["hard_block"]):
                reason = "specific_or_veto_cannot_link"
                structure["blocked_veto_edge_count"] += int(bool(edge["veto_hard_cannot_link"]))
            elif bool(edge["positive_candidate"]) and uf.find(str(edge["object_a"])) != uf.find(str(edge["object_b"])):
                conflict, conflict_reason = _component_conflict(
                    uf=uf,
                    object_ids=object_ids,
                    object_a=str(edge["object_a"]),
                    object_b=str(edge["object_b"]),
                    by_object=by_object,
                    veto_pairs=veto_pairs,
                )
                if conflict:
                    reason = conflict_reason
                    structure["blocked_veto_edge_count"] += int(conflict_reason == "component_veto_cannot_link")
                else:
                    uf.union(str(edge["object_a"]), str(edge["object_b"]))
                    accepted = True
                    structure["added_anchor_edge_count"] += 1
            edge_rows.append(
                {
                    "schema_version": "stream4d_v103_supp_phaseS3_edge_intervention_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": str(variant["variant_id"]),
                    "scene_id": str(scene),
                    "edge_rank": int(rank),
                    "object_a": str(edge["object_a"]),
                    "object_b": str(edge["object_b"]),
                    "K_anchor": float(edge["K_anchor"]),
                    "K_support": float(edge["K_support"]),
                    "K_anchor_support": float(edge["K_anchor_support"]),
                    "positive_candidate": bool(edge["positive_candidate"]),
                    "weak_support_only": bool(edge["weak_support_only"]),
                    "accepted_union": bool(accepted),
                    "reject_reason": reason,
                    "specific_conflict": bool(edge["specific_conflict"]),
                    "veto_hard_cannot_link": bool(edge["veto_hard_cannot_link"]),
                    "veto_score": float(edge["veto_score"]),
                    "direct_pair_lookup_object_a": str(edge.get("direct_pair_lookup_object_a", "")),
                    "direct_pair_lookup_object_b": str(edge.get("direct_pair_lookup_object_b", "")),
                    "direct_pair_positive": bool(edge.get("direct_pair_positive", False)),
                    "direct_pair_support_count": int(edge.get("direct_pair_support_count", 0)),
                    "direct_pair_reliability_mean": float(edge.get("direct_pair_reliability_mean", 0.0)),
                    "direct_pair_reliability_max": float(edge.get("direct_pair_reliability_max", 0.0)),
                    "direct_pair_broad_risk_max": float(edge.get("direct_pair_broad_risk_max", 0.0)),
                    "direct_pair_diagnostic_same_gt_count": int(edge.get("direct_pair_diagnostic_same_gt_count", 0)),
                    "direct_pair_diagnostic_different_gt_count": int(edge.get("direct_pair_diagnostic_different_gt_count", 0)),
                    "direct_pair_diagnostic_same_gt_rate": float(edge.get("direct_pair_diagnostic_same_gt_rate", 0.0)),
                    "direct_pair_diagnostic_different_gt_rate": float(edge.get("direct_pair_diagnostic_different_gt_rate", 0.0)),
                    "direct_pair_example_pair_ids": str(edge.get("direct_pair_example_pair_ids", "")),
                    "shuffle_anchor": bool(variant.get("shuffle_anchor", False)),
                    "shuffle_direct_pair": bool(variant.get("shuffle_direct_pair", False)),
                    "role_permutation_support_as_anchor": bool(variant.get("role_permutation_support_as_anchor", False)),
                    "uses_gt_for_prediction": False,
                }
            )
        components: dict[str, list[str]] = defaultdict(list)
        for oid in object_ids:
            components[uf.find(oid)].append(oid)
        structure["cannot_link_violation_count_after_clustering"] += _cannot_link_violations(components, by_object, veto_pairs)
        structure["cluster_count"] += len(components)
        largest = max((len(v) for v in components.values()), default=0)
        structure["largest_cluster_ratio"] = max(structure["largest_cluster_ratio"], float(largest / max(len(object_ids), 1)))
        structure["object_count_delta_vs_baseline"] += int(len(components) - len(object_ids))
        for comp_idx, members in enumerate(sorted(components.values(), key=lambda v: (-len(v), v[0]))):
            comp_rows = scene_base[scene_base["mv_object_id"].astype(str).isin(members)].copy()
            object_id = f"{variant['variant_id']}:{scene}:c0000:s3_{comp_idx:05d}"
            frames = sorted(comp_rows["frame_id"].astype(int).unique().tolist())
            base_score = float(comp_rows.get("score", pd.Series([0.0])).max())
            if str(variant.get("score_policy", "")) == "anchor_stability_boost":
                anchor_rates = [float(object_info.get(oid, {}).get("anchor_valid_mask_rate", 0.0)) for oid in members]
                score = base_score + 0.03 * float(np.mean(anchor_rates) if anchor_rates else 0.0)
            elif str(variant.get("score_policy", "")) == "veto_risk_suppress":
                veto_touch = [sum(1 for pair in veto_pairs if oid in pair) for oid in members]
                risk = min(1.0, float(np.mean(veto_touch) if veto_touch else 0.0) / 5.0)
                score = max(0.0, base_score - 0.05 * risk)
            else:
                score = base_score
            cluster_rows.append(
                {
                    "schema_version": "stream4d_v103_supp_phaseS3_cluster_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": str(variant["variant_id"]),
                    "scene_id": str(scene),
                    "mv_object_id": object_id,
                    "source_object_ids": json.dumps(members, sort_keys=True),
                    "source_object_count": int(len(members)),
                    "frame_count": int(len(frames)),
                    "object_score": float(score),
                    "base_object_score": float(base_score),
                    "uses_gt_for_prediction": False,
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
                scene_rows[str(scene)].append(
                    {
                        "schema_version": "stream4d_v103_supp_phaseS3_object_frame_mask_row_v1",
                        "phase_id": PHASE_ID,
                        "variant_id": str(variant["variant_id"]),
                        "mv_object_id": object_id,
                        "object_id": object_id,
                        "scene_id": str(scene),
                        "chunk_id": str(best.get("chunk_id", "c0000")),
                        "window_id": str(best.get("window_id", "c0000")),
                        "frame_local_index": int(best["frame_local_index"]),
                        "frame_id": int(best["frame_id"]),
                        "selected_mask_id": int(best["selected_mask_id"]),
                        "mask_id_or_generated_id": int(best["mask_id_or_generated_id"]),
                        "object_score": float(score),
                        "score": float(score),
                        "support_count": int(best.get("phase5_support_count", 0) or 0),
                        "node_policy": "f2_overlap3_object_skeleton",
                        "emit_policy": "component_wta_by_f2_score_support",
                        "readout_mode": "supp_s3_scaffolded_mask_graph",
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )
    structure["frame_mask_count_delta_vs_baseline"] = int(sum(len(v) for v in scene_rows.values()) - len(base))
    return scene_rows, edge_rows, cluster_rows, structure


def _control_deltas(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(row["variant_id"]): row for row in metric_rows}
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        if bool(variant.get("is_control", False)):
            continue
        vid = str(variant["variant_id"])
        real = by_id.get(vid)
        if real is None:
            continue
        controls = [v for v in VARIANTS if str(v.get("control_for", "")) == vid]
        if not controls:
            continue
        for control in controls:
            cid = str(control["variant_id"])
            ctrl = by_id.get(cid)
            if ctrl is None:
                continue
            if bool(control.get("role_permutation_support_as_anchor", False)):
                control_type = "role_permutation"
            elif bool(control.get("shuffle_direct_pair", False)):
                control_type = "shuffled_direct_pair"
            else:
                control_type = "shuffled_anchor"
            rows.append(
                {
                    "schema_version": "stream4d_v103_supp_phaseS3_control_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": vid,
                    "control_variant_id": cid,
                    "control_type": control_type,
                    "real_minus_control_MV_AP_window": float(real["MV_AP_window"]) - float(ctrl["MV_AP_window"]),
                    "real_minus_control_MV_AP50_window": float(real["MV_AP50_window"]) - float(ctrl["MV_AP50_window"]),
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_eval": True,
                }
            )
    return rows


def _fragmentation_rows(window_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in window_rows:
        rows.append(
            {
                "schema_version": "stream4d_v103_supp_phaseS3_fragmentation_diagnostic_row_v1",
                "phase_id": PHASE_ID,
                "variant_id": row["variant_id"],
                "scene_id": row["scene_id"],
                "window_id": row["window_id"],
                "GT_fragment_count_mean_diagnostic": row.get("gt_fragment_count_mean", 0.0),
                "GT_fragment_count_ge2_rate_diagnostic": row.get("gt_fragment_count_ge2_rate", 0.0),
                "GT_fragment_count_p90_diagnostic": "",
                "union_minus_best_IoU_mean_diagnostic": "",
                "same_GT_split_rate_diagnostic": "",
                "pred_multi_GT_touch_rate_diagnostic": "",
                "uses_gt_for_prediction": False,
                "uses_gt_for_eval": True,
            }
        )
    return rows


def _gate_rows(best: dict[str, Any], baseline: dict[str, float], baseline_row: dict[str, Any], control_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    control_delta = 0.0
    for row in control_rows:
        if str(row["variant_id"]) == str(best.get("variant_id", "")) and str(row["control_type"]) in {
            "shuffled_anchor",
            "shuffled_direct_pair",
        }:
            control_delta = max(control_delta, float(row["real_minus_control_MV_AP_window"]))
    baseline_pixel = float(baseline_row.get("pixel_collision_rate", 0.0)) if baseline_row else 0.0
    specs = [
        ("MV_AP_window_ge_baseline_plus_0p005", float(best.get("MV_AP_window", 0.0)) >= float(baseline.get("MV_AP_window", 0.0)) + 0.005, best.get("MV_AP_window", 0.0), float(baseline.get("MV_AP_window", 0.0)) + 0.005),
        ("MV_AP50_window_ge_baseline_plus_0p010", float(best.get("MV_AP50_window", 0.0)) >= float(baseline.get("MV_AP50_window", 0.0)) + 0.010, best.get("MV_AP50_window", 0.0), float(baseline.get("MV_AP50_window", 0.0)) + 0.010),
        ("real_minus_shuffled_MV_AP_window_ge_0p003", control_delta >= 0.003, control_delta, 0.003),
        ("same_frame_collision_count_eq_0", int(best.get("same_frame_collision_count", 0)) == 0, best.get("same_frame_collision_count", 0), 0),
        ("pixel_collision_rate_le_baseline_plus_0p005", float(best.get("pixel_collision_rate", 0.0)) <= baseline_pixel + 0.005, best.get("pixel_collision_rate", 0.0), baseline_pixel + 0.005),
        ("cannot_link_violation_count_after_clustering_eq_0", int(best.get("cannot_link_violation_count_after_clustering", 0)) == 0, best.get("cannot_link_violation_count_after_clustering", 0), 0),
    ]
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for name, ok, observed, required in specs:
        gate_rows.append(
            {
                "schema_version": "stream4d_v103_supp_phaseS3_gate_row_v1",
                "phase_id": PHASE_ID,
                "variant_id": best.get("variant_id", ""),
                "gate_name": name,
                "pass": bool(ok),
                "observed": observed,
                "required": required,
                "uses_gt_for_prediction": False,
                "uses_gt_for_eval": True,
            }
        )
        if not ok:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v103_supp_phaseS3_failure_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": best.get("variant_id", ""),
                    "failure_id": name,
                    "severity": "blocking",
                    "evidence": f"observed={observed} required={required}",
                    "repair_direction": "Follow Phase S3 ladder: if real-control is weak, stop positive anchor boost and inspect veto-only/history; if AP drops with lower fragmentation, inspect overmerge/collision; if edge counts are near zero, return to S1 anchor coverage.",
                }
            )
    return gate_rows, failure_rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")

    phaseS2_root = _project(args.phaseS2_root)
    phaseS2_summary = _read_json(phaseS2_root / "summary.json")
    if phaseS2_summary.get("decision") != "PASS_ENTER_PHASES3_SCAFFOLDED_MASK_GRAPH":
        raise RuntimeError(f"Phase S2 has not passed: {phaseS2_root / 'summary.json'}")
    phase2_summaries = {
        "scene0011_00": _read_json(_project(args.scene0011_phase2_root) / "summary.json"),
        "scene0050_00": _read_json(_project(args.scene0050_phase2_root) / "summary.json"),
    }
    s2_payloads = _load_s2_payloads(phaseS2_root)
    as_payloads = {scene: payloads["AS"] for scene, payloads in s2_payloads.items()}
    base = _adapt_f2_rows(
        f2_root=_project(args.f2_root),
        phase2_summaries=phase2_summaries,
        phase5_payloads=as_payloads,
        dataset_split=str(args.dataset_split),
        chunk_id=str(args.chunk_id),
    )
    object_rows_df, object_features = _object_feature_tables(base, s2_payloads)
    veto_pairs, veto_score = _load_veto_object_pairs(phaseS2_root, base)
    phase9n_root = _project(args.phase9n_root) if str(args.phase9n_root).strip() else None
    direct_pair_support = _direct_pair_support_by_scene(base, phase9n_root)

    metric_rows: list[dict[str, Any]] = []
    window_rows_all: list[dict[str, Any]] = []
    selected_rows_all: list[dict[str, Any]] = []
    edge_rows_all: list[dict[str, Any]] = []
    cluster_rows_all: list[dict[str, Any]] = []
    for variant in VARIANTS:
        scene_rows, edge_rows, cluster_rows, structure = _materialize_variant(
            base=base,
            object_features=object_features,
            object_rows=object_rows_df,
            veto_pairs=veto_pairs,
            veto_score=veto_score,
            direct_pair_support_by_scene=direct_pair_support,
            variant=variant,
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
        duplicate_claims = _count_duplicate_claims(selected_rows)
        aggregate.update(
            {
                "schema_version": "stream4d_v103_supp_phaseS3_variant_metric_row_v1",
                "phase_id": PHASE_ID,
                "variant_kind": str(variant["kind"]),
                "is_control": bool(variant.get("is_control", False)),
                "control_for": str(variant.get("control_for", "")),
                "anchor_threshold": float(variant.get("anchor_threshold", 0.0)),
                "support_threshold": float(variant.get("support_threshold", 0.0)),
                "direct_pair_support_min_count": int(variant.get("direct_pair_support_min_count", 0)),
                "direct_pair_reliability_mean_min": float(variant.get("direct_pair_reliability_mean_min", 0.0)),
                "topk_per_object": int(variant.get("topk_per_object", 0)),
                "use_veto": bool(variant.get("use_veto", False)),
                "shuffle_anchor": bool(variant.get("shuffle_anchor", False)),
                "shuffle_direct_pair": bool(variant.get("shuffle_direct_pair", False)),
                "role_permutation_support_as_anchor": bool(variant.get("role_permutation_support_as_anchor", False)),
                "pixel_collision_count": int(pixel_collision_count),
                "missing_mask_raster_count": int(missing_count),
                "duplicate_mask_claim_count": int(duplicate_claims),
                "object_count": int(structure["cluster_count"]),
                "frame_mask_count": int(len(selected_rows)),
                "dataset_split": str(args.dataset_split),
                "chunk_id": str(args.chunk_id),
                "metric_scope": f"same_subset_as_v103_phase6_first32_{args.chunk_id}",
                "uses_gt_for_prediction": False,
                "uses_gt_for_eval": True,
                **structure,
            }
        )
        metric_rows.append(aggregate)
        window_rows_all.extend(window_rows)
        selected_rows_all.extend(selected_rows)
        edge_rows_all.extend(edge_rows)
        cluster_rows_all.extend(cluster_rows)

    real_rows = [row for row in metric_rows if not bool(row.get("is_control", False))]
    best = max(real_rows, key=lambda r: (float(r.get("MV_AP_window", 0.0)), float(r.get("MV_AP50_window", 0.0))), default={})
    baseline_row = next((row for row in metric_rows if str(row["variant_id"]) == "S3_V0_baseline_skeleton_replay"), {})
    file_baseline = _load_subset_baseline(_project(args.subset_baseline_rows))
    if str(args.gate_baseline_source) == "replay_variant" and baseline_row:
        baseline = {
            "MV_AP_window": float(baseline_row.get("MV_AP_window", 0.0)),
            "MV_AP50_window": float(baseline_row.get("MV_AP50_window", 0.0)),
            "MV_AP25_window": float(baseline_row.get("MV_AP25_window", 0.0)),
            "ScoreFreeMatch50_window": float(baseline_row.get("ScoreFreeMatch50_window", 0.0)),
        }
    else:
        baseline = file_baseline
    control_rows = _control_deltas(metric_rows)
    gate_rows, failure_rows = _gate_rows(best, baseline, baseline_row, control_rows)

    _write_csv(out / "f2_object_rows.csv", object_rows_df.to_dict("records"))
    _write_csv(out / "variant_metric_rows.csv", metric_rows)
    _write_csv(out / "variant_scene_metric_rows_diagnostic.csv", window_rows_all)
    _write_csv(out / "object_frame_mask_rows.csv", selected_rows_all)
    _write_csv(out / "control_rows.csv", control_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    _write_csv(out / "fragmentation_diagnostic_rows.csv", _fragmentation_rows(window_rows_all))
    _write_parquet(out / "edge_intervention_rows.parquet", edge_rows_all)
    _write_parquet(out / "cluster_rows.parquet", cluster_rows_all)

    pass_s3 = not failure_rows
    decision = "PASS_ENTER_PHASES4_POST_BIRTH_HISTORY_INHERITANCE" if pass_s3 else "NO_GO_REPAIR_PHASES3_SCAFFOLDED_MASK_GRAPH"
    summary = {
        "schema_version": "stream4d_v103_supp_phaseS3_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "phaseS3_pass": bool(pass_s3),
        "failure_count": len(failure_rows),
        "variant_count": len(VARIANTS),
        "best_variant_id": best.get("variant_id", ""),
        "best_MV_AP_window": best.get("MV_AP_window", ""),
        "best_MV_AP50_window": best.get("MV_AP50_window", ""),
        "baseline_subset": baseline,
        "file_baseline_subset": file_baseline,
        "gate_baseline_source": str(args.gate_baseline_source),
        "best_minus_baseline_MV_AP_window": float(best.get("MV_AP_window", 0.0)) - float(baseline.get("MV_AP_window", 0.0)) if baseline else "",
        "phaseS2_root": _rel(phaseS2_root),
        "phase9n_root": _rel(phase9n_root) if phase9n_root is not None else "",
        "scaffold_feature_row_match_rate": float(np.mean(base["phase5_mask_index"].astype(int).to_numpy() >= 0)) if len(base) else 0.0,
        "scaffold_object_match_rate_mean": float(object_rows_df["s2_mask_match_rate"].astype(float).mean()) if not object_rows_df.empty else 0.0,
        "veto_object_pair_count": len(veto_pairs),
        "direct_pair_object_pair_count": int(sum(len(v) for v in direct_pair_support.values())),
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
        "truthfulness_note": "S3 starts from the existing F2 mask-view skeleton and only adds anchor-supported unions, support-only diagnostics, score repair, or veto/cannot-link blocks. AP uses GT only for evaluation.",
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "variant_metric_rows": _rel(out / "variant_metric_rows.csv"),
            "variant_scene_metric_rows_diagnostic": _rel(out / "variant_scene_metric_rows_diagnostic.csv"),
            "edge_intervention_rows": _rel(out / "edge_intervention_rows.parquet"),
            "cluster_rows": _rel(out / "cluster_rows.parquet"),
            "object_frame_mask_rows": _rel(out / "object_frame_mask_rows.csv"),
            "control_rows": _rel(out / "control_rows.csv"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "fragmentation_diagnostic_rows": _rel(out / "fragmentation_diagnostic_rows.csv"),
            "last_command": _rel(out / "last_command.txt"),
        },
    }
    _write_json(out / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v103 supplement Phase S3 scaffolded mask graph intervention.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--phaseS2-root", default=str(DEFAULT_PHASES2_ROOT))
    parser.add_argument("--phase9n-root", default="")
    parser.add_argument("--f2-root", default=str(DEFAULT_F2_ROOT))
    parser.add_argument("--scene0011-phase2-root", default=str(DEFAULT_PHASE2_SCENE0011))
    parser.add_argument("--scene0050-phase2-root", default=str(DEFAULT_PHASE2_SCENE0050))
    parser.add_argument("--subset-baseline-rows", default=str(DEFAULT_SUBSET_BASELINE))
    parser.add_argument("--gate-baseline-source", choices=["subset_file", "replay_variant"], default="subset_file")
    parser.add_argument("--dataset-split", default="dev")
    parser.add_argument("--chunk-id", default="c0000")
    parser.add_argument("--min-pred-pixels", type=int, default=20)
    parser.add_argument("--min-gt-pixels", type=int, default=20)
    parser.add_argument("--cupy-device-id", type=int, default=0)
    parser.add_argument("--disable-cupy-iou", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = build(args)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if summary["phaseS3_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
