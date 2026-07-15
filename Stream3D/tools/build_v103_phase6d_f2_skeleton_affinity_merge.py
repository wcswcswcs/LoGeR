#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools.build_v103_phase6_mask_clustering_local_object_birth import (  # noqa: E402
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


PHASE_ID = "v103_phase6d_f2_skeleton_affinity_merge"
DEFAULT_F2_ROOT = STREAM3D_ROOT / "outputs/audit/v100_phase2c_overlap3_local_repair"
DEFAULT_PHASE5_ROOT = STREAM3D_ROOT / "outputs/audit/v103_phase5_positive_core_pooling_q5c_repair5_r1"
DEFAULT_OUT = STREAM3D_ROOT / "outputs/audit/v103_phase6d_f2_skeleton_affinity_merge_r1"
DEFAULT_SUBSET_BASELINE = STREAM3D_ROOT / "outputs/audit/v103_phase6_baseline_subset_contract_r1/baseline_subset_metric_rows.csv"


VARIANTS = [
    {
        "variant_id": "D0_f2_original_replay",
        "merge_threshold": 1.01,
        "topk_per_object": 0,
        "shuffle_affinity": False,
    },
    {
        "variant_id": "D1_affinity_merge_tau090_top1_specific_veto",
        "merge_threshold": 0.90,
        "topk_per_object": 1,
        "shuffle_affinity": False,
    },
    {
        "variant_id": "D2_affinity_merge_tau085_top1_specific_veto",
        "merge_threshold": 0.85,
        "topk_per_object": 1,
        "shuffle_affinity": False,
    },
    {
        "variant_id": "D3_affinity_merge_tau090_top2_specific_veto",
        "merge_threshold": 0.90,
        "topk_per_object": 2,
        "shuffle_affinity": False,
    },
    {
        "variant_id": "D4_affinity_merge_tau080_top1_specific_veto",
        "merge_threshold": 0.80,
        "topk_per_object": 1,
        "shuffle_affinity": False,
    },
    {
        "variant_id": "D5_affinity_merge_tau075_top1_specific_veto",
        "merge_threshold": 0.75,
        "topk_per_object": 1,
        "shuffle_affinity": False,
    },
    {
        "variant_id": "D6_affinity_merge_tau080_top2_specific_veto",
        "merge_threshold": 0.80,
        "topk_per_object": 2,
        "shuffle_affinity": False,
    },
    {
        "variant_id": "D7_affinity_merge_tau075_top1_broad_support_veto",
        "merge_threshold": 0.75,
        "topk_per_object": 1,
        "shuffle_affinity": False,
        "broad_support_veto": True,
        "broad_support_min_broad_rate": 0.70,
        "broad_support_max_object_like_rate": 0.50,
        "broad_support_min_support_mean": 1000.0,
    },
    {
        "variant_id": "D8_affinity_merge_tau070_top1_broad_support_veto",
        "merge_threshold": 0.70,
        "topk_per_object": 1,
        "shuffle_affinity": False,
        "broad_support_veto": True,
        "broad_support_min_broad_rate": 0.70,
        "broad_support_max_object_like_rate": 0.50,
        "broad_support_min_support_mean": 1000.0,
    },
    {
        "variant_id": "D9_affinity_merge_tau065_top1_broad_support_veto",
        "merge_threshold": 0.65,
        "topk_per_object": 1,
        "shuffle_affinity": False,
        "broad_support_veto": True,
        "broad_support_min_broad_rate": 0.70,
        "broad_support_max_object_like_rate": 0.50,
        "broad_support_min_support_mean": 1000.0,
    },
    {
        "variant_id": "D10_affinity_merge_tau070_top2_broad_support_veto",
        "merge_threshold": 0.70,
        "topk_per_object": 2,
        "shuffle_affinity": False,
        "broad_support_veto": True,
        "broad_support_min_broad_rate": 0.70,
        "broad_support_max_object_like_rate": 0.50,
        "broad_support_min_support_mean": 1000.0,
    },
    {
        "variant_id": "D11_tau090_top1_merge_count_score_boost",
        "merge_threshold": 0.90,
        "topk_per_object": 1,
        "shuffle_affinity": False,
        "score_policy": "merge_count_boost",
        "merge_count_score_boost": 0.05,
    },
    {
        "variant_id": "D12_tau090_top1_frame_coverage_score",
        "merge_threshold": 0.90,
        "topk_per_object": 1,
        "shuffle_affinity": False,
        "score_policy": "frame_coverage",
    },
    {
        "variant_id": "D13_tau065_top1_direct_pair_min1_broad_support_veto",
        "merge_threshold": 0.65,
        "topk_per_object": 1,
        "shuffle_affinity": False,
        "broad_support_veto": True,
        "broad_support_min_broad_rate": 0.70,
        "broad_support_max_object_like_rate": 0.50,
        "broad_support_min_support_mean": 1000.0,
        "direct_pair_support_min_count": 1,
    },
    {
        "variant_id": "D14_tau065_top1_direct_pair_min2_broad_support_veto",
        "merge_threshold": 0.65,
        "topk_per_object": 1,
        "shuffle_affinity": False,
        "broad_support_veto": True,
        "broad_support_min_broad_rate": 0.70,
        "broad_support_max_object_like_rate": 0.50,
        "broad_support_min_support_mean": 1000.0,
        "direct_pair_support_min_count": 2,
    },
    {
        "variant_id": "D15_tau075_top1_direct_pair_min1_specific_veto",
        "merge_threshold": 0.75,
        "topk_per_object": 1,
        "shuffle_affinity": False,
        "direct_pair_support_min_count": 1,
    },
    {
        "variant_id": "R1_shuffled_affinity_merge_tau090_top1_control",
        "merge_threshold": 0.90,
        "topk_per_object": 1,
        "shuffle_affinity": True,
    },
    {
        "variant_id": "R2_shuffled_affinity_merge_tau080_top1_control",
        "merge_threshold": 0.80,
        "topk_per_object": 1,
        "shuffle_affinity": True,
    },
    {
        "variant_id": "R3_shuffled_affinity_merge_tau075_top1_broad_support_veto_control",
        "merge_threshold": 0.75,
        "topk_per_object": 1,
        "shuffle_affinity": True,
        "broad_support_veto": True,
        "broad_support_min_broad_rate": 0.70,
        "broad_support_max_object_like_rate": 0.50,
        "broad_support_min_support_mean": 1000.0,
    },
    {
        "variant_id": "R4_shuffled_affinity_merge_tau070_top1_broad_support_veto_control",
        "merge_threshold": 0.70,
        "topk_per_object": 1,
        "shuffle_affinity": True,
        "broad_support_veto": True,
        "broad_support_min_broad_rate": 0.70,
        "broad_support_max_object_like_rate": 0.50,
        "broad_support_min_support_mean": 1000.0,
    },
    {
        "variant_id": "R5_shuffled_affinity_merge_tau065_top1_broad_support_veto_control",
        "merge_threshold": 0.65,
        "topk_per_object": 1,
        "shuffle_affinity": True,
        "broad_support_veto": True,
        "broad_support_min_broad_rate": 0.70,
        "broad_support_max_object_like_rate": 0.50,
        "broad_support_min_support_mean": 1000.0,
    },
    {
        "variant_id": "R6_shuffled_tau090_top1_merge_count_score_boost_control",
        "merge_threshold": 0.90,
        "topk_per_object": 1,
        "shuffle_affinity": True,
        "score_policy": "merge_count_boost",
        "merge_count_score_boost": 0.05,
    },
    {
        "variant_id": "R7_shuffled_tau090_top1_frame_coverage_score_control",
        "merge_threshold": 0.90,
        "topk_per_object": 1,
        "shuffle_affinity": True,
        "score_policy": "frame_coverage",
    },
    {
        "variant_id": "R8_shuffled_tau065_top1_direct_pair_min1_broad_support_veto_control",
        "merge_threshold": 0.65,
        "topk_per_object": 1,
        "shuffle_affinity": True,
        "broad_support_veto": True,
        "broad_support_min_broad_rate": 0.70,
        "broad_support_max_object_like_rate": 0.50,
        "broad_support_min_support_mean": 1000.0,
        "direct_pair_support_min_count": 1,
    },
    {
        "variant_id": "R9_shuffled_tau065_top1_direct_pair_min2_broad_support_veto_control",
        "merge_threshold": 0.65,
        "topk_per_object": 1,
        "shuffle_affinity": True,
        "broad_support_veto": True,
        "broad_support_min_broad_rate": 0.70,
        "broad_support_max_object_like_rate": 0.50,
        "broad_support_min_support_mean": 1000.0,
        "direct_pair_support_min_count": 2,
    },
]


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}
        self.size = {item: 1 for item in items}

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            nxt = self.parent[item]
            self.parent[item] = root
            item = nxt
        return root

    def union(self, a: str, b: str) -> str:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return ra
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return ra


def _load_subset_baseline(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    row = df.iloc[0]
    return {
        "MV_AP_window": float(row["MV_AP_window"]),
        "MV_AP50_window": float(row["MV_AP50_window"]),
        "MV_AP25_window": float(row["MV_AP25_window"]),
        "ScoreFreeMatch50_window": float(row["ScoreFreeMatch50_window"]),
    }


def _load_phase5_scene(phase5_root: Path, scene: str) -> dict[str, Any]:
    payload = torch.load(phase5_root / scene / "mask_level_feature.pt", map_location="cpu")
    feature = payload["feature"].to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
    norm = np.linalg.norm(feature, axis=1, keepdims=True)
    feature = feature / np.maximum(norm, 1e-12)
    feature[~np.isfinite(feature)] = 0.0
    return {
        "feature": feature,
        "mask_frame": payload["mask_frame"].cpu().numpy().astype(np.int64),
        "mask_label": payload["mask_label"].cpu().numpy().astype(np.int64),
        "mask_is_broad": payload["mask_is_broad"].cpu().numpy().astype(bool),
        "mask_is_object_like": payload["mask_is_object_like"].cpu().numpy().astype(bool),
        "support_count": payload["support_count"].cpu().numpy().astype(np.int64),
    }


def _phase5_index(payload: dict[str, Any]) -> dict[tuple[int, int], int]:
    return {
        (int(frame), int(label)): int(idx)
        for idx, (frame, label) in enumerate(zip(payload["mask_frame"].tolist(), payload["mask_label"].tolist()))
    }


def _adapt_f2_rows(
    *,
    f2_root: Path,
    phase2_summaries: dict[str, dict[str, Any]],
    phase5_payloads: dict[str, dict[str, Any]],
    dataset_split: str,
    chunk_id: str,
) -> pd.DataFrame:
    rows = pd.read_parquet(f2_root / "mv_object_frame_mask_rows.parquet")
    rows = rows[(rows["dataset_split"].astype(str) == dataset_split) & (rows["chunk_id"].astype(str) == chunk_id)].copy()
    frame_to_local = {
        scene: {int(frame_id): idx for idx, frame_id in enumerate(summary["frame_ids"])}
        for scene, summary in phase2_summaries.items()
    }
    indexes = {scene: _phase5_index(payload) for scene, payload in phase5_payloads.items()}
    out: list[dict[str, Any]] = []
    for row in rows.to_dict("records"):
        scene = str(row["scene_id"])
        frame_id = int(row["frame_id"])
        if scene not in frame_to_local or frame_id not in frame_to_local[scene]:
            continue
        local = int(frame_to_local[scene][frame_id])
        mask_id = int(row["selected_mask_id"])
        phase5_idx = indexes[scene].get((local, mask_id), -1)
        payload = phase5_payloads[scene]
        if phase5_idx >= 0:
            is_broad = bool(payload["mask_is_broad"][phase5_idx])
            is_object = bool(payload["mask_is_object_like"][phase5_idx])
            support = int(payload["support_count"][phase5_idx])
        else:
            is_broad = True
            is_object = False
            support = 0
        new = dict(row)
        new.update(
            {
                "frame_local_index": local,
                "phase5_mask_index": int(phase5_idx),
                "selected_mask_is_broad": is_broad,
                "selected_mask_is_object_like": is_object,
                "phase5_support_count": support,
            }
        )
        out.append(new)
    if not out:
        raise RuntimeError(f"no F2 rows match v103 first32 subset for dataset_split={dataset_split} chunk_id={chunk_id}")
    return pd.DataFrame(out)


def _object_tables(base: pd.DataFrame, phase5_payloads: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    object_rows: list[dict[str, Any]] = []
    object_features: dict[str, np.ndarray] = {}
    for (scene, oid), group in base.groupby(["scene_id", "mv_object_id"], sort=False):
        scene = str(scene)
        oid = str(oid)
        idxs = [int(v) for v in group["phase5_mask_index"].tolist() if int(v) >= 0]
        if idxs:
            feat = phase5_payloads[scene]["feature"][np.asarray(sorted(set(idxs)), dtype=np.int64)]
            centroid = feat.mean(axis=0)
            centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
            object_features[oid] = centroid.astype(np.float32, copy=False)
        else:
            object_features[oid] = np.zeros((phase5_payloads[scene]["feature"].shape[1],), dtype=np.float32)
        object_rows.append(
            {
                "schema_version": "stream4d_v103_phase6d_f2_object_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "mv_object_id": oid,
                "frame_count": int(group["frame_id"].nunique()),
                "row_count": int(len(group)),
                "original_score": float(group.get("score", pd.Series([0.0])).max()),
                "selected_broad_rate": float(group["selected_mask_is_broad"].astype(bool).mean()),
                "selected_object_like_rate": float(group["selected_mask_is_object_like"].astype(bool).mean()),
                "phase5_mask_match_rate": float(np.mean(group["phase5_mask_index"].astype(int).to_numpy() >= 0)),
                "support_mean": float(group["phase5_support_count"].astype(float).mean()),
                "uses_gt_for_prediction": False,
            }
        )
    return pd.DataFrame(object_rows), object_features


def _obs_id(scene: str, frame_id: int, mask_id: int) -> str:
    return f"{scene}:{int(frame_id)}:{int(mask_id)}"


def _direct_pair_support_by_scene(base: pd.DataFrame, phase9n_root: Path | None) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
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
            objects_a = obs_to_objects.get(str(row["mask_a_observation_id"]), set())
            objects_b = obs_to_objects.get(str(row["mask_b_observation_id"]), set())
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
                    item["diagnostic_same_gt_count"] += int(bool(row.get("diagnostic_same_gt", False)))
                    item["diagnostic_different_gt_count"] += int(bool(row.get("diagnostic_different_gt", False)))
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


def _specific_conflict(group_a: pd.DataFrame, group_b: pd.DataFrame) -> bool:
    by_frame_a = {int(row["frame_id"]): row for row in group_a.to_dict("records")}
    for row_b in group_b.to_dict("records"):
        frame = int(row_b["frame_id"])
        row_a = by_frame_a.get(frame)
        if row_a is None:
            continue
        if int(row_a["selected_mask_id"]) == int(row_b["selected_mask_id"]):
            continue
        a_specific = bool(row_a["selected_mask_is_object_like"]) and not bool(row_a["selected_mask_is_broad"])
        b_specific = bool(row_b["selected_mask_is_object_like"]) and not bool(row_b["selected_mask_is_broad"])
        if a_specific and b_specific:
            return True
    return False


def _broad_support_risk(group: pd.DataFrame, variant: dict[str, Any]) -> bool:
    if not bool(variant.get("broad_support_veto", False)):
        return False
    broad_rate = float(group["selected_mask_is_broad"].astype(bool).mean())
    object_like_rate = float(group["selected_mask_is_object_like"].astype(bool).mean())
    support_mean = float(group["phase5_support_count"].astype(float).mean())
    return (
        broad_rate >= float(variant.get("broad_support_min_broad_rate", 0.70))
        and object_like_rate <= float(variant.get("broad_support_max_object_like_rate", 0.50))
        and support_mean >= float(variant.get("broad_support_min_support_mean", 1000.0))
    )


def _candidate_edges(
    *,
    scene_base: pd.DataFrame,
    object_features: dict[str, np.ndarray],
    variant: dict[str, Any],
    rng: np.random.Generator,
    direct_pair_support: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    object_ids = sorted(scene_base["mv_object_id"].astype(str).unique().tolist())
    features = np.stack([object_features[oid] for oid in object_ids], axis=0)
    if bool(variant.get("shuffle_affinity", False)) and len(object_ids) > 1:
        features = features[rng.permutation(len(object_ids))]
    sim = features @ features.T
    threshold = float(variant["merge_threshold"])
    edges: list[dict[str, Any]] = []
    by_object = {oid: scene_base[scene_base["mv_object_id"].astype(str) == oid] for oid in object_ids}
    for i, oid_a in enumerate(object_ids[:-1]):
        local: list[dict[str, Any]] = []
        for j in range(i + 1, len(object_ids)):
            oid_b = object_ids[j]
            score = float(sim[i, j])
            if score < threshold:
                continue
            conflict = _specific_conflict(by_object[oid_a], by_object[oid_b])
            broad_support_veto = _broad_support_risk(by_object[oid_a], variant) or _broad_support_risk(by_object[oid_b], variant)
            pair_key = tuple(sorted((oid_a, oid_b)))
            direct = direct_pair_support.get(pair_key, {})
            direct_count = int(direct.get("count", 0))
            direct_reliability_mean = float(direct.get("reliability_mean", 0.0))
            direct_reliability_max = float(direct.get("reliability_max", 0.0))
            if direct_count < int(variant.get("direct_pair_support_min_count", 0)):
                continue
            if direct_reliability_mean < float(variant.get("direct_pair_reliability_mean_min", 0.0)):
                continue
            if direct_reliability_max < float(variant.get("direct_pair_reliability_max_min", 0.0)):
                continue
            local.append(
                {
                    "object_a": oid_a,
                    "object_b": oid_b,
                    "affinity": score,
                    "specific_conflict": conflict,
                    "broad_support_veto": broad_support_veto,
                    "direct_pair_support_count": direct_count,
                    "direct_pair_reliability_mean": direct_reliability_mean,
                    "direct_pair_reliability_max": direct_reliability_max,
                    "direct_pair_broad_risk_max": float(direct.get("broad_risk_max", 0.0)),
                    "direct_pair_diagnostic_same_gt_count": int(direct.get("diagnostic_same_gt_count", 0)),
                    "direct_pair_diagnostic_different_gt_count": int(direct.get("diagnostic_different_gt_count", 0)),
                    "direct_pair_diagnostic_same_gt_rate": float(direct.get("diagnostic_same_gt_rate", 0.0)),
                    "direct_pair_diagnostic_different_gt_rate": float(direct.get("diagnostic_different_gt_rate", 0.0)),
                    "direct_pair_example_pair_ids": ";".join(direct.get("example_pair_ids", [])),
                }
            )
        local.sort(key=lambda row: row["affinity"], reverse=True)
        topk = int(variant.get("topk_per_object", 0))
        edges.extend(local[:topk] if topk > 0 else local)
    edges.sort(key=lambda row: row["affinity"], reverse=True)
    return edges


def _materialize_variant(
    *,
    base: pd.DataFrame,
    object_features: dict[str, np.ndarray],
    variant: dict[str, Any],
    chunk_id: str,
    direct_pair_support_by_scene: dict[str, dict[tuple[str, str], dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    scene_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    edge_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(10317)
    for scene, scene_base in base.groupby("scene_id", sort=True):
        object_ids = sorted(scene_base["mv_object_id"].astype(str).unique().tolist())
        uf = UnionFind(object_ids)
        edges = _candidate_edges(
            scene_base=scene_base,
            object_features=object_features,
            variant=variant,
            rng=rng,
            direct_pair_support=direct_pair_support_by_scene.get(str(scene), {}),
        )
        for rank, edge in enumerate(edges):
            accepted = False
            reason = ""
            if bool(edge["specific_conflict"]):
                reason = "specific_same_frame_conflict"
            elif bool(edge.get("broad_support_veto", False)):
                reason = "broad_support_risk_veto"
            elif uf.find(str(edge["object_a"])) != uf.find(str(edge["object_b"])):
                uf.union(str(edge["object_a"]), str(edge["object_b"]))
                accepted = True
            edge_rows.append(
                {
                    "schema_version": "stream4d_v103_phase6d_merge_edge_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": str(variant["variant_id"]),
                    "scene_id": str(scene),
                    "edge_rank": int(rank),
                    "object_a": str(edge["object_a"]),
                    "object_b": str(edge["object_b"]),
                    "affinity": float(edge["affinity"]),
                    "accepted_union": bool(accepted),
                    "reject_reason": reason,
                    "specific_conflict": bool(edge.get("specific_conflict", False)),
                    "broad_support_veto": bool(edge.get("broad_support_veto", False)),
                    "direct_pair_support_min_count": int(variant.get("direct_pair_support_min_count", 0)),
                    "direct_pair_support_count": int(edge.get("direct_pair_support_count", 0)),
                    "direct_pair_reliability_mean": float(edge.get("direct_pair_reliability_mean", 0.0)),
                    "direct_pair_reliability_max": float(edge.get("direct_pair_reliability_max", 0.0)),
                    "direct_pair_broad_risk_max": float(edge.get("direct_pair_broad_risk_max", 0.0)),
                    "direct_pair_diagnostic_same_gt_count": int(edge.get("direct_pair_diagnostic_same_gt_count", 0)),
                    "direct_pair_diagnostic_different_gt_count": int(edge.get("direct_pair_diagnostic_different_gt_count", 0)),
                    "direct_pair_diagnostic_same_gt_rate": float(edge.get("direct_pair_diagnostic_same_gt_rate", 0.0)),
                    "direct_pair_diagnostic_different_gt_rate": float(edge.get("direct_pair_diagnostic_different_gt_rate", 0.0)),
                    "direct_pair_example_pair_ids": str(edge.get("direct_pair_example_pair_ids", "")),
                    "shuffle_affinity": bool(variant.get("shuffle_affinity", False)),
                    "uses_gt_for_prediction": False,
                }
            )
        components: dict[str, list[str]] = defaultdict(list)
        for oid in object_ids:
            components[uf.find(oid)].append(oid)
        for comp_idx, members in enumerate(sorted(components.values(), key=lambda v: (-len(v), v[0]))):
            comp_rows = scene_base[scene_base["mv_object_id"].astype(str).isin(members)].copy()
            object_id = f"{variant['variant_id']}:{scene}:{chunk_id}:merged_{comp_idx:05d}"
            frames = sorted(comp_rows["frame_id"].astype(int).unique().tolist())
            base_score = float(comp_rows.get("score", pd.Series([0.0])).max())
            score_policy = str(variant.get("score_policy", "max_f2_score"))
            if score_policy == "merge_count_boost":
                score = base_score + float(variant.get("merge_count_score_boost", 0.05)) * float(max(0, len(members) - 1))
            elif score_policy == "frame_coverage":
                score = float(len(frames) / 32.0)
            else:
                score = base_score
            cluster_rows.append(
                {
                    "schema_version": "stream4d_v103_phase6d_merge_cluster_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": str(variant["variant_id"]),
                    "scene_id": str(scene),
                    "mv_object_id": object_id,
                    "source_object_count": int(len(members)),
                    "frame_count": int(len(frames)),
                    "object_score": score,
                    "base_object_score": base_score,
                    "score_policy": score_policy,
                    "selected_broad_rate": float(comp_rows["selected_mask_is_broad"].astype(bool).mean()),
                    "selected_object_like_rate": float(comp_rows["selected_mask_is_object_like"].astype(bool).mean()),
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
                        "schema_version": "stream4d_v103_phase6d_f2_skeleton_frame_mask_row_v1",
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
                        "object_score": score,
                        "score": score,
                        "support_count": int(best.get("phase5_support_count", 0) or 0),
                        "node_policy": "f2_overlap3_object_skeleton",
                        "emit_policy": "component_wta_by_f2_score_support",
                        "readout_mode": "f2_skeleton_primitive_affinity_merge",
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )
    return scene_rows, edge_rows, cluster_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge F2 skeleton objects with v103 primitive affinity features and shuffled control.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--f2-root", default=str(DEFAULT_F2_ROOT))
    parser.add_argument("--phase5-root", default=str(DEFAULT_PHASE5_ROOT))
    parser.add_argument("--phase9n-root", default="")
    parser.add_argument("--scene0011-phase2-root", default=str(DEFAULT_PHASE2_SCENE0011))
    parser.add_argument("--scene0050-phase2-root", default=str(DEFAULT_PHASE2_SCENE0050))
    parser.add_argument("--subset-baseline-rows", default=str(DEFAULT_SUBSET_BASELINE))
    parser.add_argument("--dataset-split", default="dev")
    parser.add_argument("--chunk-id", default="c0000")
    parser.add_argument("--min-pred-pixels", type=int, default=20)
    parser.add_argument("--min-gt-pixels", type=int, default=20)
    parser.add_argument("--cupy-device-id", type=int, default=0)
    parser.add_argument("--disable-cupy-iou", action="store_true")
    return parser


def main() -> int:
    t0 = time.time()
    args = build_parser().parse_args()
    out = _project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")
    phase2_summaries = {
        "scene0011_00": _read_json(_project(args.scene0011_phase2_root) / "summary.json"),
        "scene0050_00": _read_json(_project(args.scene0050_phase2_root) / "summary.json"),
    }
    phase5_root = _project(args.phase5_root)
    phase5_payloads = {scene: _load_phase5_scene(phase5_root, scene) for scene in phase2_summaries}
    base = _adapt_f2_rows(
        f2_root=_project(args.f2_root),
        phase2_summaries=phase2_summaries,
        phase5_payloads=phase5_payloads,
        dataset_split=str(args.dataset_split),
        chunk_id=str(args.chunk_id),
    )
    object_rows_df, object_features = _object_tables(base, phase5_payloads)
    phase9n_root = _project(args.phase9n_root) if str(args.phase9n_root).strip() else None
    direct_pair_support = _direct_pair_support_by_scene(base, phase9n_root)

    all_metric_rows: list[dict[str, Any]] = []
    all_window_rows: list[dict[str, Any]] = []
    all_selected_rows: list[dict[str, Any]] = []
    all_edge_rows: list[dict[str, Any]] = []
    all_cluster_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        scene_rows, edge_rows, cluster_rows = _materialize_variant(
            base=base,
            object_features=object_features,
            variant=variant,
            chunk_id=str(args.chunk_id),
            direct_pair_support_by_scene=direct_pair_support,
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
        aggregate.update(
            {
                "phase_id": PHASE_ID,
                "merge_threshold": float(variant["merge_threshold"]),
                "topk_per_object": int(variant["topk_per_object"]),
                "shuffle_affinity": bool(variant.get("shuffle_affinity", False)),
                "broad_support_veto": bool(variant.get("broad_support_veto", False)),
                "broad_support_min_broad_rate": float(variant.get("broad_support_min_broad_rate", 0.0)),
                "broad_support_max_object_like_rate": float(variant.get("broad_support_max_object_like_rate", 1.0)),
                "broad_support_min_support_mean": float(variant.get("broad_support_min_support_mean", 0.0)),
                "direct_pair_support_min_count": int(variant.get("direct_pair_support_min_count", 0)),
                "score_policy": str(variant.get("score_policy", "max_f2_score")),
                "merge_count_score_boost": float(variant.get("merge_count_score_boost", 0.0)),
                "accepted_merge_count": int(sum(1 for row in edge_rows if row["accepted_union"])),
                "candidate_edge_count": int(len(edge_rows)),
                "pixel_collision_count": int(pixel_collision_count),
                "missing_mask_raster_count": int(missing_count),
                "dataset_split": str(args.dataset_split),
                "chunk_id": str(args.chunk_id),
                "metric_scope": f"same_subset_as_v103_phase6_first32_{args.chunk_id}",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        all_metric_rows.append(aggregate)
        all_window_rows.extend(window_rows)
        all_selected_rows.extend(selected_rows)
        all_edge_rows.extend(edge_rows)
        all_cluster_rows.extend(cluster_rows)

    subset_baseline = _load_subset_baseline(_project(args.subset_baseline_rows))
    best_overall = max(all_metric_rows, key=lambda r: (float(r.get("MV_AP_window", 0.0)), float(r.get("MV_AP50_window", 0.0))), default={})
    real_rows = [row for row in all_metric_rows if not bool(row.get("shuffle_affinity", False))]
    shuffled_rows = [row for row in all_metric_rows if bool(row.get("shuffle_affinity", False))]
    replay = next((row for row in all_metric_rows if str(row.get("variant_id")) == "D0_f2_original_replay"), {})
    best_real = max(real_rows, key=lambda r: (float(r.get("MV_AP_window", 0.0)), float(r.get("MV_AP50_window", 0.0))), default={})
    best_shuffled = max(shuffled_rows, key=lambda r: (float(r.get("MV_AP_window", 0.0)), float(r.get("MV_AP50_window", 0.0))), default={})
    best_real_variant_id = str(best_real.get("variant_id", ""))
    accepted_specific_conflict_count = int(
        sum(
            1
            for row in all_edge_rows
            if str(row.get("variant_id", "")) == best_real_variant_id
            and bool(row.get("accepted_union", False))
            and bool(row.get("specific_conflict", False))
        )
    )
    replay_pixel = float(replay.get("pixel_collision_rate", 0.0)) if replay else 0.0
    gate_specs = [
        (
            "best_real_MV_AP_window_ge_replay_plus_0p005",
            float(best_real.get("MV_AP_window", 0.0)) >= float(replay.get("MV_AP_window", 0.0)) + 0.005,
            best_real.get("MV_AP_window", 0.0),
            float(replay.get("MV_AP_window", 0.0)) + 0.005,
        ),
        (
            "best_real_MV_AP50_window_ge_replay_plus_0p010",
            float(best_real.get("MV_AP50_window", 0.0)) >= float(replay.get("MV_AP50_window", 0.0)) + 0.010,
            best_real.get("MV_AP50_window", 0.0),
            float(replay.get("MV_AP50_window", 0.0)) + 0.010,
        ),
        (
            "best_real_minus_best_shuffled_MV_AP_window_ge_0p003",
            float(best_real.get("MV_AP_window", 0.0)) - float(best_shuffled.get("MV_AP_window", 0.0)) >= 0.003,
            float(best_real.get("MV_AP_window", 0.0)) - float(best_shuffled.get("MV_AP_window", 0.0)),
            0.003,
        ),
        (
            "same_frame_collision_count_eq_0",
            int(best_real.get("same_frame_collision_count", 0)) == 0,
            int(best_real.get("same_frame_collision_count", 0)),
            0,
        ),
        (
            "pixel_collision_rate_le_replay_plus_0p005",
            float(best_real.get("pixel_collision_rate", 0.0)) <= replay_pixel + 0.005,
            best_real.get("pixel_collision_rate", 0.0),
            replay_pixel + 0.005,
        ),
        (
            "missing_mask_raster_count_eq_0",
            int(best_real.get("missing_mask_raster_count", 0)) == 0,
            int(best_real.get("missing_mask_raster_count", 0)),
            0,
        ),
        (
            "accepted_specific_conflict_count_eq_0",
            accepted_specific_conflict_count == 0,
            accepted_specific_conflict_count,
            0,
        ),
    ]
    gate_rows = [
        {
            "schema_version": "stream4d_v103_phase6d_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_name": name,
            "pass": bool(ok),
            "observed": observed,
            "required": required,
            "best_real_variant_id": best_real_variant_id,
            "replay_variant_id": replay.get("variant_id", ""),
            "best_shuffled_variant_id": best_shuffled.get("variant_id", ""),
        }
        for name, ok, observed, required in gate_specs
    ]
    failure_rows = [
        {
            "schema_version": "stream4d_v103_phase6d_failure_row_v1",
            "phase_id": PHASE_ID,
            "failure_id": row["gate_name"],
            "severity": "blocking",
            "evidence": f"observed={row['observed']} required={row['required']}",
            "uses_gt_for_prediction": False,
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    decision = "PASS_PHASE6D_S3_STYLE_LOCAL_GATE" if not failure_rows else "NO_GO_PHASE6D_S3_STYLE_LOCAL_GATE"
    _write_csv(out / "f2_object_rows.csv", object_rows_df.to_dict("records"))
    _write_csv(out / "merge_edge_rows.csv", all_edge_rows)
    _write_csv(out / "merge_cluster_rows.csv", all_cluster_rows)
    _write_csv(out / "merge_metric_rows.csv", all_metric_rows)
    _write_csv(out / "merge_window_rows.csv", all_window_rows)
    _write_csv(out / "merge_selected_rows.csv", all_selected_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    summary = {
        "schema_version": "stream4d_v103_phase6d_f2_skeleton_affinity_merge_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "failure_count": len(failure_rows),
        "variant_count": len(VARIANTS),
        "best_variant_id": best_overall.get("variant_id", ""),
        "best_MV_AP_window": best_overall.get("MV_AP_window", ""),
        "best_MV_AP50_window": best_overall.get("MV_AP50_window", ""),
        "best_real_variant_id": best_real.get("variant_id", ""),
        "best_real_MV_AP_window": best_real.get("MV_AP_window", ""),
        "best_real_MV_AP50_window": best_real.get("MV_AP50_window", ""),
        "best_shuffled_variant_id": best_shuffled.get("variant_id", ""),
        "best_shuffled_MV_AP_window": best_shuffled.get("MV_AP_window", ""),
        "best_shuffled_MV_AP50_window": best_shuffled.get("MV_AP50_window", ""),
        "replay_variant_id": replay.get("variant_id", ""),
        "replay_MV_AP_window": replay.get("MV_AP_window", ""),
        "replay_MV_AP50_window": replay.get("MV_AP50_window", ""),
        "best_real_minus_replay_MV_AP_window": float(best_real.get("MV_AP_window", 0.0)) - float(replay.get("MV_AP_window", 0.0)) if replay else "",
        "best_real_minus_replay_MV_AP50_window": float(best_real.get("MV_AP50_window", 0.0)) - float(replay.get("MV_AP50_window", 0.0)) if replay else "",
        "best_real_minus_best_shuffled_MV_AP_window": float(best_real.get("MV_AP_window", 0.0)) - float(best_shuffled.get("MV_AP_window", 0.0)) if best_shuffled else "",
        "best_real_minus_best_shuffled_MV_AP50_window": float(best_real.get("MV_AP50_window", 0.0)) - float(best_shuffled.get("MV_AP50_window", 0.0)) if best_shuffled else "",
        "accepted_specific_conflict_count_best_real": accepted_specific_conflict_count,
        "phase9n_root": _rel(phase9n_root) if phase9n_root is not None else "",
        "f2_subset_baseline": subset_baseline,
        "best_minus_f2_subset_MV_AP_window": float(best_overall.get("MV_AP_window", 0.0)) - float(subset_baseline.get("MV_AP_window", 0.0))
        if subset_baseline
        else "",
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "truthfulness_note": "F2 skeleton object-frame masks are merged only by GT-free v103 primitive affinity object centroids; R1 shuffles affinities as a control.",
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "f2_object_rows": _rel(out / "f2_object_rows.csv"),
            "merge_edge_rows": _rel(out / "merge_edge_rows.csv"),
            "merge_cluster_rows": _rel(out / "merge_cluster_rows.csv"),
            "merge_metric_rows": _rel(out / "merge_metric_rows.csv"),
            "merge_window_rows": _rel(out / "merge_window_rows.csv"),
            "merge_selected_rows": _rel(out / "merge_selected_rows.csv"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "last_command": _rel(out / "last_command.txt"),
        },
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
