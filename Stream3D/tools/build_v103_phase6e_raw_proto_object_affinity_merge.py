#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from build_v103_phase6_mask_clustering_local_object_birth import (  # noqa: E402
    DEFAULT_PHASE2_SCENE0011,
    DEFAULT_PHASE2_SCENE0050,
    _evaluate_variant,
    _load_baseline,
    _project,
    _read_json,
    _rel,
    _write_csv,
    _write_json,
)


PHASE_ID = "v103_phase6e_raw_proto_object_affinity_merge"
DEFAULT_RAW_PHASE6_ROOT = STREAM3D_ROOT / "outputs/audit/v103_phase6_positive_core_da3_bridge_pair_phase9n_r4_no_broad_or_rel070_raw_broad_support_veto"
DEFAULT_PHASE5_ROOT = STREAM3D_ROOT / "outputs/audit/v103_phase5_positive_core_da3_bridge_pair_phase9n_r3_no_broad_or_rel070"
DEFAULT_BASELINE_ROWS = STREAM3D_ROOT / "outputs/audit/v103_phase0_contract/baseline_metric_rows.csv"
DEFAULT_OUT = STREAM3D_ROOT / "outputs/audit/v103_phase6e_raw_proto_object_affinity_merge_r1"
DEFAULT_SEED_VARIANTS = [
    "M2_constrained_all_supported_strict_l2o_tau070_top8_min2",
    "M2_constrained_all_supported_strict_l2o_tau080_top4_min2",
    "M7_repair_all_supported_broad_support_veto_tau070_top8_min2",
]


MERGE_VARIANTS: list[dict[str, Any]] = [
    {
        "merge_variant_id": "E0_seed_replay",
        "merge_threshold": 1.01,
        "topk_per_object": 0,
        "shuffle_affinity": False,
        "broad_support_veto": False,
    },
    {
        "merge_variant_id": "E1_proto_merge_tau075_top1",
        "merge_threshold": 0.75,
        "topk_per_object": 1,
        "shuffle_affinity": False,
        "broad_support_veto": False,
    },
    {
        "merge_variant_id": "E2_proto_merge_tau075_top1_broad_support_veto",
        "merge_threshold": 0.75,
        "topk_per_object": 1,
        "shuffle_affinity": False,
        "broad_support_veto": True,
        "broad_support_min_broad_rate": 0.70,
        "broad_support_max_object_like_rate": 0.50,
        "broad_support_min_support_mean": 1000.0,
    },
    {
        "merge_variant_id": "E3_proto_merge_tau085_top1_broad_support_veto",
        "merge_threshold": 0.85,
        "topk_per_object": 1,
        "shuffle_affinity": False,
        "broad_support_veto": True,
        "broad_support_min_broad_rate": 0.70,
        "broad_support_max_object_like_rate": 0.50,
        "broad_support_min_support_mean": 1000.0,
    },
    {
        "merge_variant_id": "R1_shuffled_proto_merge_tau075_top1_broad_support_veto",
        "merge_threshold": 0.75,
        "topk_per_object": 1,
        "shuffle_affinity": True,
        "broad_support_veto": True,
        "broad_support_min_broad_rate": 0.70,
        "broad_support_max_object_like_rate": 0.50,
        "broad_support_min_support_mean": 1000.0,
    },
]


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


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


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.map(_parse_bool)


def _load_phase5_scene(phase5_root: Path, scene: str) -> dict[str, np.ndarray]:
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


def _prepare_seed_rows(raw_phase6_root: Path, phase2_summaries: dict[str, dict[str, Any]], seed_variants: list[str]) -> pd.DataFrame:
    path = raw_phase6_root / "local_object_frame_mask_rows.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = df[df["variant_id"].astype(str).isin(seed_variants)].copy()
    if df.empty:
        raise RuntimeError(f"no selected rows for seed variants={seed_variants}")
    df["source_seed_variant_id"] = df["variant_id"].astype(str)
    df["source_proto_object_id"] = df["mv_object_id"].astype(str)
    df["proto_key"] = df["source_seed_variant_id"].astype(str) + "::" + df["source_proto_object_id"].astype(str)
    df["phase5_mask_index"] = df["selected_mask_observation_index"].astype(int)
    df["phase5_support_count"] = df["support_count"].astype(int)
    df["selected_mask_is_broad"] = _bool_series(df["selected_mask_is_broad"])
    df["selected_mask_is_object_like"] = _bool_series(df["selected_mask_is_object_like"])
    if "frame_id" not in df.columns:
        frame_ids = {
            scene: [int(v) for v in summary["frame_ids"]]
            for scene, summary in phase2_summaries.items()
        }
        df["frame_id"] = [
            frame_ids[str(scene)][int(local)]
            for scene, local in zip(df["scene_id"].astype(str).tolist(), df["frame_local_index"].astype(int).tolist())
        ]
    return df


def _object_tables(base: pd.DataFrame, phase5_payloads: dict[str, dict[str, np.ndarray]]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    features: dict[str, np.ndarray] = {}
    for (seed, scene, oid), group in base.groupby(["source_seed_variant_id", "scene_id", "source_proto_object_id"], sort=False):
        scene = str(scene)
        oid = str(oid)
        key = str(group["proto_key"].iloc[0])
        idxs = sorted({int(v) for v in group["phase5_mask_index"].tolist() if int(v) >= 0})
        if idxs:
            feat = phase5_payloads[scene]["feature"][np.asarray(idxs, dtype=np.int64)]
            centroid = feat.mean(axis=0)
            centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
            features[key] = centroid.astype(np.float32, copy=False)
        else:
            features[key] = np.zeros((phase5_payloads[scene]["feature"].shape[1],), dtype=np.float32)
        broad = _bool_series(group["selected_mask_is_broad"])
        obj = _bool_series(group["selected_mask_is_object_like"])
        rows.append(
            {
                "schema_version": "stream4d_v103_phase6e_proto_object_row_v1",
                "phase_id": PHASE_ID,
                "source_seed_variant_id": str(seed),
                "scene_id": scene,
                "source_proto_object_id": oid,
                "proto_key": key,
                "frame_count": int(group["frame_id"].astype(int).nunique()),
                "row_count": int(len(group)),
                "original_score": float(group.get("score", pd.Series([0.0])).astype(float).max()),
                "selected_broad_rate": float(broad.mean()),
                "selected_object_like_rate": float(obj.mean()),
                "support_mean": float(group["phase5_support_count"].astype(float).mean()),
                "uses_gt_for_prediction": False,
            }
        )
    return pd.DataFrame(rows), features


def _specific_conflict(group_a: pd.DataFrame, group_b: pd.DataFrame) -> bool:
    by_frame_a = {int(row["frame_id"]): row for row in group_a.to_dict("records")}
    for row_b in group_b.to_dict("records"):
        frame = int(row_b["frame_id"])
        row_a = by_frame_a.get(frame)
        if row_a is None:
            continue
        if int(row_a["selected_mask_id"]) == int(row_b["selected_mask_id"]):
            continue
        a_specific = _parse_bool(row_a["selected_mask_is_object_like"]) and not _parse_bool(row_a["selected_mask_is_broad"])
        b_specific = _parse_bool(row_b["selected_mask_is_object_like"]) and not _parse_bool(row_b["selected_mask_is_broad"])
        if a_specific and b_specific:
            return True
    return False


def _broad_support_risk(group: pd.DataFrame, variant: dict[str, Any]) -> bool:
    if not bool(variant.get("broad_support_veto", False)):
        return False
    broad_rate = float(_bool_series(group["selected_mask_is_broad"]).mean())
    object_like_rate = float(_bool_series(group["selected_mask_is_object_like"]).mean())
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
) -> list[dict[str, Any]]:
    object_ids = sorted(scene_base["proto_key"].astype(str).unique().tolist())
    if len(object_ids) < 2:
        return []
    features = np.stack([object_features[oid] for oid in object_ids], axis=0)
    if bool(variant.get("shuffle_affinity", False)):
        features = features[rng.permutation(len(object_ids))]
    sim = features @ features.T
    threshold = float(variant["merge_threshold"])
    by_object = {oid: scene_base[scene_base["proto_key"].astype(str) == oid] for oid in object_ids}
    edges: list[dict[str, Any]] = []
    for i, oid_a in enumerate(object_ids[:-1]):
        local: list[dict[str, Any]] = []
        for j in range(i + 1, len(object_ids)):
            oid_b = object_ids[j]
            score = float(sim[i, j])
            if score < threshold:
                continue
            local.append(
                {
                    "object_a": oid_a,
                    "object_b": oid_b,
                    "affinity": score,
                    "specific_conflict": _specific_conflict(by_object[oid_a], by_object[oid_b]),
                    "broad_support_veto": _broad_support_risk(by_object[oid_a], variant)
                    or _broad_support_risk(by_object[oid_b], variant),
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
    seed_variant_id: str,
    merge_variant: dict[str, Any],
) -> tuple[str, dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    variant_id = f"{seed_variant_id}__{merge_variant['merge_variant_id']}"
    subset = base[base["source_seed_variant_id"].astype(str) == seed_variant_id].copy()
    scene_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    edge_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(10317)
    for scene, scene_base in subset.groupby("scene_id", sort=True):
        object_ids = sorted(scene_base["proto_key"].astype(str).unique().tolist())
        uf = UnionFind(object_ids)
        edges = _candidate_edges(scene_base=scene_base, object_features=object_features, variant=merge_variant, rng=rng)
        for rank, edge in enumerate(edges):
            accepted = False
            reason = ""
            if bool(edge["specific_conflict"]):
                reason = "specific_same_frame_conflict"
            elif bool(edge["broad_support_veto"]):
                reason = "broad_support_risk_veto"
            elif uf.find(str(edge["object_a"])) != uf.find(str(edge["object_b"])):
                uf.union(str(edge["object_a"]), str(edge["object_b"]))
                accepted = True
            edge_rows.append(
                {
                    "schema_version": "stream4d_v103_phase6e_proto_merge_edge_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": variant_id,
                    "source_seed_variant_id": seed_variant_id,
                    "merge_variant_id": str(merge_variant["merge_variant_id"]),
                    "scene_id": str(scene),
                    "edge_rank": int(rank),
                    "object_a": str(edge["object_a"]),
                    "object_b": str(edge["object_b"]),
                    "affinity": float(edge["affinity"]),
                    "accepted_union": bool(accepted),
                    "reject_reason": reason,
                    "specific_conflict": bool(edge["specific_conflict"]),
                    "broad_support_veto": bool(edge["broad_support_veto"]),
                    "shuffle_affinity": bool(merge_variant.get("shuffle_affinity", False)),
                    "uses_gt_for_prediction": False,
                }
            )
        components: dict[str, list[str]] = defaultdict(list)
        for oid in object_ids:
            components[uf.find(oid)].append(oid)
        for comp_idx, members in enumerate(sorted(components.values(), key=lambda v: (-len(v), v[0]))):
            comp_rows = scene_base[scene_base["proto_key"].astype(str).isin(members)].copy()
            object_id = f"{variant_id}:{scene}:c0000:proto_merged_{comp_idx:05d}"
            score = float(comp_rows.get("score", pd.Series([0.0])).astype(float).max())
            cluster_rows.append(
                {
                    "schema_version": "stream4d_v103_phase6e_proto_merge_cluster_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": variant_id,
                    "source_seed_variant_id": seed_variant_id,
                    "merge_variant_id": str(merge_variant["merge_variant_id"]),
                    "scene_id": str(scene),
                    "mv_object_id": object_id,
                    "source_object_count": int(len(members)),
                    "frame_count": int(comp_rows["frame_id"].astype(int).nunique()),
                    "object_score": score,
                    "selected_broad_rate": float(_bool_series(comp_rows["selected_mask_is_broad"]).mean()),
                    "selected_object_like_rate": float(_bool_series(comp_rows["selected_mask_is_object_like"]).mean()),
                    "support_mean": float(comp_rows["phase5_support_count"].astype(float).mean()),
                    "uses_gt_for_prediction": False,
                }
            )
            for _frame, group in comp_rows.groupby("frame_id", sort=True):
                best = sorted(
                    group.to_dict("records"),
                    key=lambda row: (
                        -float(row.get("score", 0.0)),
                        -int(row.get("phase5_support_count", 0)),
                        -int(_parse_bool(row.get("selected_mask_is_object_like", False))),
                        int(_parse_bool(row.get("selected_mask_is_broad", True))),
                        int(row.get("selected_mask_id", 0)),
                    ),
                )[0]
                scene_rows[str(scene)].append(
                    {
                        "schema_version": "stream4d_v103_phase6e_proto_object_frame_mask_row_v1",
                        "phase_id": PHASE_ID,
                        "variant_id": variant_id,
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
                        "node_policy": str(best.get("node_policy", "raw_phase6_seed")),
                        "emit_policy": "proto_component_wta_by_score_support",
                        "readout_mode": "phase6e_raw_proto_object_affinity_merge",
                        "selected_mask_is_broad": _parse_bool(best.get("selected_mask_is_broad", False)),
                        "selected_mask_is_object_like": _parse_bool(best.get("selected_mask_is_object_like", False)),
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )
    return variant_id, scene_rows, edge_rows, cluster_rows


def _variant_seed_delta(metric_rows: list[dict[str, Any]]) -> dict[str, float]:
    by_variant = {str(row["variant_id"]): row for row in metric_rows}
    seed_replay: dict[str, float] = {}
    out: dict[str, float] = {}
    for vid, row in by_variant.items():
        seed = str(row.get("source_seed_variant_id", ""))
        if vid.endswith("__E0_seed_replay"):
            seed_replay[seed] = float(row.get("MV_AP_window", 0.0))
    for vid, row in by_variant.items():
        seed = str(row.get("source_seed_variant_id", ""))
        out[vid] = float(row.get("MV_AP_window", 0.0)) - float(seed_replay.get(seed, 0.0))
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v103 Phase6e: merge raw Phase6 proto-objects with Phase9n primitive affinity features.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--raw-phase6-root", default=str(DEFAULT_RAW_PHASE6_ROOT))
    parser.add_argument("--phase5-root", default=str(DEFAULT_PHASE5_ROOT))
    parser.add_argument("--scene0011-phase2-root", default=str(DEFAULT_PHASE2_SCENE0011))
    parser.add_argument("--scene0050-phase2-root", default=str(DEFAULT_PHASE2_SCENE0050))
    parser.add_argument("--baseline-rows", default=str(DEFAULT_BASELINE_ROWS))
    parser.add_argument("--seed-variants", default=",".join(DEFAULT_SEED_VARIANTS))
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
    phase2_roots = {
        "scene0011_00": _project(args.scene0011_phase2_root),
        "scene0050_00": _project(args.scene0050_phase2_root),
    }
    phase2_summaries = {scene: _read_json(root / "summary.json") for scene, root in phase2_roots.items()}
    seed_variants = [item.strip() for item in str(args.seed_variants).split(",") if item.strip()]
    raw_phase6_root = _project(args.raw_phase6_root)
    phase5_root = _project(args.phase5_root)
    baseline = _load_baseline(_project(args.baseline_rows))
    base = _prepare_seed_rows(raw_phase6_root, phase2_summaries, seed_variants)
    phase5_payloads = {scene: _load_phase5_scene(phase5_root, scene) for scene in phase2_roots}
    proto_rows_df, object_features = _object_tables(base, phase5_payloads)

    all_metric_rows: list[dict[str, Any]] = []
    all_window_rows: list[dict[str, Any]] = []
    all_selected_rows: list[dict[str, Any]] = []
    all_edge_rows: list[dict[str, Any]] = []
    all_cluster_rows: list[dict[str, Any]] = []
    for seed in seed_variants:
        for merge_variant in MERGE_VARIANTS:
            variant_id, scene_rows, edge_rows, cluster_rows = _materialize_variant(
                base=base,
                object_features=object_features,
                seed_variant_id=seed,
                merge_variant=merge_variant,
            )
            window_rows, aggregate, selected_rows, pixel_collision_count, missing_count, _frame_count = _evaluate_variant(
                variant_id=variant_id,
                scene_rows=scene_rows,
                phase2_summaries=phase2_summaries,
                min_pred_pixels=int(args.min_pred_pixels),
                min_gt_pixels=int(args.min_gt_pixels),
                use_cupy_iou=not bool(args.disable_cupy_iou),
                cupy_device_id=int(args.cupy_device_id),
            )
            for row in window_rows:
                row["phase_id"] = PHASE_ID
                row["source_seed_variant_id"] = seed
                row["merge_variant_id"] = str(merge_variant["merge_variant_id"])
            for row in selected_rows:
                row["phase_id"] = PHASE_ID
            aggregate.update(
                {
                    "phase_id": PHASE_ID,
                    "source_seed_variant_id": seed,
                    "merge_variant_id": str(merge_variant["merge_variant_id"]),
                    "merge_threshold": float(merge_variant["merge_threshold"]),
                    "topk_per_object": int(merge_variant["topk_per_object"]),
                    "shuffle_affinity": bool(merge_variant.get("shuffle_affinity", False)),
                    "broad_support_veto": bool(merge_variant.get("broad_support_veto", False)),
                    "broad_support_min_broad_rate": float(merge_variant.get("broad_support_min_broad_rate", 0.0)),
                    "broad_support_max_object_like_rate": float(merge_variant.get("broad_support_max_object_like_rate", 1.0)),
                    "broad_support_min_support_mean": float(merge_variant.get("broad_support_min_support_mean", 0.0)),
                    "accepted_merge_count": int(sum(1 for row in edge_rows if row["accepted_union"])),
                    "candidate_edge_count": int(len(edge_rows)),
                    "pixel_collision_count": int(pixel_collision_count),
                    "missing_mask_raster_count": int(missing_count),
                    "metric_scope": "first32_dev_subset_window_mean; raw Phase6 seed proto-object merge",
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
            all_metric_rows.append(aggregate)
            all_window_rows.extend(window_rows)
            all_selected_rows.extend(selected_rows)
            all_edge_rows.extend(edge_rows[:50000])
            all_cluster_rows.extend(cluster_rows)

    seed_delta = _variant_seed_delta(all_metric_rows)
    for row in all_metric_rows:
        row["minus_seed_replay_MV_AP_window"] = float(seed_delta.get(str(row["variant_id"]), 0.0))

    baseline_ap = float(baseline.get("MV_AP_window", 0.0))
    baseline_ap50 = float(baseline.get("MV_AP50_window", 0.0))
    best = max(all_metric_rows, key=lambda row: (float(row.get("MV_AP_window", 0.0)), float(row.get("MV_AP50_window", 0.0))), default={})
    failure_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for row in all_metric_rows:
        checks = [
            ("uses_gt_for_prediction_false", not bool(row["uses_gt_for_prediction"]), row["uses_gt_for_prediction"], False),
            ("uses_future_false", not bool(row["uses_future"]), row["uses_future"], False),
            ("pixel_collision_rate_le_0p02", float(row["pixel_collision_rate"]) <= 0.02, row["pixel_collision_rate"], 0.02),
            ("MV_AP_window_ge_baseline_minus_0p003", float(row["MV_AP_window"]) >= baseline_ap - 0.003, row["MV_AP_window"], baseline_ap - 0.003),
            ("MV_AP50_window_ge_baseline_minus_0p006", float(row["MV_AP50_window"]) >= baseline_ap50 - 0.006, row["MV_AP50_window"], baseline_ap50 - 0.006),
        ]
        for name, ok, observed, required in checks:
            gate_rows.append(
                {
                    "schema_version": "stream4d_v103_phase6e_gate_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": row["variant_id"],
                    "gate_name": name,
                    "pass": bool(ok),
                    "observed": observed,
                    "required": required,
                }
            )
            if row.get("variant_id") == best.get("variant_id") and not bool(ok):
                failure_rows.append(
                    {
                        "schema_version": "stream4d_v103_phase6e_failure_row_v1",
                        "phase_id": PHASE_ID,
                        "variant_id": row["variant_id"],
                        "failure_id": name,
                        "severity": "blocking",
                        "evidence": f"observed={observed} required={required}",
                        "repair_direction": "If this fails, raw Phase6 proto seeds are too weak; build stronger GT-free proto-object seeds before history or DA3-as-object branches.",
                    }
                )

    _write_csv(out / "proto_object_rows.csv", proto_rows_df.to_dict("records"))
    _write_csv(out / "proto_merge_edge_rows.csv", all_edge_rows)
    _write_csv(out / "proto_merge_cluster_rows.csv", all_cluster_rows)
    _write_csv(out / "proto_merge_selected_rows.csv", all_selected_rows)
    _write_csv(out / "proto_merge_metric_rows.csv", all_metric_rows)
    _write_csv(out / "proto_merge_window_rows.csv", all_window_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    summary = {
        "schema_version": "stream4d_v103_phase6e_raw_proto_object_affinity_merge_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": "PASS_PHASE6E_RAW_PROTO_OBJECT_AFFINITY_MERGE" if not failure_rows else "NO_GO_PHASE6E_RAW_PROTO_OBJECT_AFFINITY_MERGE",
        "phase6e_pass": not failure_rows,
        "failure_count": len(failure_rows),
        "best_variant_id": best.get("variant_id", ""),
        "best_MV_AP_window": best.get("MV_AP_window", ""),
        "best_MV_AP50_window": best.get("MV_AP50_window", ""),
        "best_minus_seed_replay_MV_AP_window": best.get("minus_seed_replay_MV_AP_window", ""),
        "baseline_contract": baseline,
        "raw_phase6_root": _rel(raw_phase6_root),
        "phase5_root": _rel(phase5_root),
        "seed_variants": seed_variants,
        "variant_count": len(all_metric_rows),
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
        "truthfulness_note": "Phase6e uses raw Phase6 mask clusters as proto-object seeds, then merges proto-objects with GT-free v103 mask-level primitive affinity centroids and object-level broad-support veto. GT is used only by the canonical evaluator.",
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "proto_object_rows": _rel(out / "proto_object_rows.csv"),
            "proto_merge_edge_rows": _rel(out / "proto_merge_edge_rows.csv"),
            "proto_merge_cluster_rows": _rel(out / "proto_merge_cluster_rows.csv"),
            "proto_merge_selected_rows": _rel(out / "proto_merge_selected_rows.csv"),
            "proto_merge_metric_rows": _rel(out / "proto_merge_metric_rows.csv"),
            "proto_merge_window_rows": _rel(out / "proto_merge_window_rows.csv"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "last_command": _rel(out / "last_command.txt"),
        },
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if not failure_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
