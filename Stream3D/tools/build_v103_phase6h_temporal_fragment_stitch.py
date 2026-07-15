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


PHASE_ID = "v103_phase6h_temporal_fragment_stitch"
DEFAULT_PHASE6G_ROOT = STREAM3D_ROOT / "outputs/audit/v103_phase6g_temporal_continuity_tracklet_birth_r1_i13"
DEFAULT_PHASE9N_ROOT = STREAM3D_ROOT / "outputs/audit/v103_phase9n_da3_bridge_pair_fused_phase4_r6_i13_e3_veto_ratio_all_pairs"
DEFAULT_PHASE5_ROOT = STREAM3D_ROOT / "outputs/audit/v103_phase5_da3_bridge_pair_phase9n_r6_i13_e3_veto_ratio_all_pairs"
DEFAULT_BASELINE_ROWS = STREAM3D_ROOT / "outputs/audit/v103_phase0_contract/baseline_metric_rows.csv"
DEFAULT_OUT = STREAM3D_ROOT / "outputs/audit/v103_phase6h_temporal_fragment_stitch_r1_i13"


VARIANTS: list[dict[str, Any]] = [
    {
        "variant_id": "H0_G0_proto080_endpoint060_gap8",
        "base_variant_id": "G0_obj_nonbroad_gap1_tau055_top2_min2",
        "proto_threshold": 0.80,
        "endpoint_threshold": 0.60,
        "combined_threshold": 0.80,
        "max_fragment_gap": 8,
        "da3_mode": "boost",
        "da3_bonus": 0.10,
        "min_da3_score": 0.40,
        "max_pair_broad_risk": 0.50,
    },
    {
        "variant_id": "H1_G0_proto070_endpoint050_gap16",
        "base_variant_id": "G0_obj_nonbroad_gap1_tau055_top2_min2",
        "proto_threshold": 0.70,
        "endpoint_threshold": 0.50,
        "combined_threshold": 0.72,
        "max_fragment_gap": 16,
        "da3_mode": "boost",
        "da3_bonus": 0.12,
        "min_da3_score": 0.40,
        "max_pair_broad_risk": 0.50,
    },
    {
        "variant_id": "H2_G0_proto060_endpoint045_gap31",
        "base_variant_id": "G0_obj_nonbroad_gap1_tau055_top2_min2",
        "proto_threshold": 0.60,
        "endpoint_threshold": 0.45,
        "combined_threshold": 0.65,
        "max_fragment_gap": 31,
        "da3_mode": "boost",
        "da3_bonus": 0.15,
        "min_da3_score": 0.40,
        "max_pair_broad_risk": 0.50,
    },
    {
        "variant_id": "H3_G3_proto080_endpoint060_gap8",
        "base_variant_id": "G3_supported_nonbroad_gap1_tau070_top1_min2",
        "proto_threshold": 0.80,
        "endpoint_threshold": 0.60,
        "combined_threshold": 0.80,
        "max_fragment_gap": 8,
        "da3_mode": "boost",
        "da3_bonus": 0.10,
        "min_da3_score": 0.40,
        "max_pair_broad_risk": 0.50,
    },
    {
        "variant_id": "H4_G4_proto070_endpoint050_gap16",
        "base_variant_id": "G4_supported_nonbroad_gap2_tau075_top1_min3",
        "proto_threshold": 0.70,
        "endpoint_threshold": 0.50,
        "combined_threshold": 0.72,
        "max_fragment_gap": 16,
        "da3_mode": "boost",
        "da3_bonus": 0.12,
        "min_da3_score": 0.40,
        "max_pair_broad_risk": 0.50,
    },
    {
        "variant_id": "H5_G8_proto060_endpoint045_gap31",
        "base_variant_id": "G8_obj_nonbroad_da3boost_gap4_tau055_bonus015_top1_min3",
        "proto_threshold": 0.60,
        "endpoint_threshold": 0.45,
        "combined_threshold": 0.65,
        "max_fragment_gap": 31,
        "da3_mode": "boost",
        "da3_bonus": 0.15,
        "min_da3_score": 0.40,
        "max_pair_broad_risk": 0.50,
    },
    {
        "variant_id": "H6_G0_proto055_endpoint040_da3bonus020_gap31",
        "base_variant_id": "G0_obj_nonbroad_gap1_tau055_top2_min2",
        "proto_threshold": 0.55,
        "endpoint_threshold": 0.40,
        "combined_threshold": 0.62,
        "max_fragment_gap": 31,
        "da3_mode": "boost",
        "da3_bonus": 0.20,
        "min_da3_score": 0.40,
        "max_pair_broad_risk": 0.50,
    },
    {
        "variant_id": "H7_G6_proto055_endpoint040_da3required_gap31",
        "base_variant_id": "G6_obj_nonbroad_da3required_gap2_tau045_rel040_top2_min2",
        "proto_threshold": 0.55,
        "endpoint_threshold": 0.40,
        "combined_threshold": 0.55,
        "max_fragment_gap": 31,
        "da3_mode": "required",
        "min_da3_score": 0.40,
        "max_pair_broad_risk": 0.50,
    },
]


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {str(item): str(item) for item in items}
        self.size = {str(item): 1 for item in items}

    def find(self, item: str) -> str:
        item = str(item)
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

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


def _pair_key(a: int, b: int) -> tuple[int, int]:
    return (min(int(a), int(b)), max(int(a), int(b)))


def _load_da3_pair_map(phase9n_root: Path, scene: str) -> dict[tuple[int, int], dict[str, float]]:
    path = phase9n_root / scene / "da3_bridge_pair_primitive_rows.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    out: dict[tuple[int, int], dict[str, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            a = int(row["mask_a_observation_index"])
            b = int(row["mask_b_observation_index"])
            key = _pair_key(a, b)
            score = float(row.get("B_ia", 0.0) or 0.0)
            prev = out.get(key)
            if prev is None or score > float(prev["da3_score"]):
                out[key] = {
                    "da3_score": score,
                    "carrier_broad_risk": float(row.get("carrier_broad_risk", 1.0) or 1.0),
                    "final_bridge_score": float(row.get("final_bridge_score", score) or score),
                }
    return out


def _load_phase5_payload(phase5_root: Path, scene: str) -> dict[str, np.ndarray]:
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


def _obs_lookup(payload: dict[str, np.ndarray]) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for idx, (frame, label) in enumerate(zip(payload["mask_frame"].tolist(), payload["mask_label"].tolist())):
        out[(int(frame), int(label))] = int(idx)
    return out


def _load_base_rows(phase6g_root: Path, base_variant_id: str, scene: str) -> list[dict[str, Any]]:
    df = pd.read_csv(phase6g_root / "temporal_frame_rows.csv")
    rows = df[(df["variant_id"] == base_variant_id) & (df["scene_id"] == scene)].copy()
    return rows.to_dict("records")


def _component_tracks(
    *,
    rows: list[dict[str, Any]],
    payload: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    lookup = _obs_lookup(payload)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["mv_object_id"])].append(row)
    out: dict[str, dict[str, Any]] = {}
    for oid, obj_rows in grouped.items():
        frames: dict[int, dict[str, Any]] = {}
        obs_indices: list[int] = []
        for row in obj_rows:
            local = int(row["frame_local_index"])
            label = int(row["selected_mask_id"])
            obs = lookup.get((local, label))
            if obs is None:
                continue
            new = dict(row)
            new["selected_mask_observation_index"] = int(obs)
            frames[local] = new
            obs_indices.append(int(obs))
        if not obs_indices:
            continue
        proto = np.mean(payload["feature"][np.asarray(obs_indices, dtype=np.int64)], axis=0)
        proto = proto / max(float(np.linalg.norm(proto)), 1e-12)
        out[oid] = {
            "mv_object_id": oid,
            "frames": frames,
            "obs_indices": obs_indices,
            "prototype": proto.astype(np.float32, copy=False),
            "frame_set": set(frames),
            "frame_count": len(frames),
            "mean_object_score": float(np.mean([float(row.get("object_score", 0.0)) for row in obj_rows])),
            "object_like_ratio": float(np.mean(payload["mask_is_object_like"][obs_indices])),
            "broad_ratio": float(np.mean(payload["mask_is_broad"][obs_indices])),
            "support_mean": float(np.mean(payload["support_count"][obs_indices])),
        }
    return out


def _closest_endpoint_score(a: dict[str, Any], b: dict[str, Any], feature: np.ndarray) -> tuple[float, int, int, int]:
    best_score = -1.0
    best_a = -1
    best_b = -1
    best_gap = 10**9
    for frame_a, row_a in a["frames"].items():
        obs_a = int(row_a["selected_mask_observation_index"])
        for frame_b, row_b in b["frames"].items():
            obs_b = int(row_b["selected_mask_observation_index"])
            gap = abs(int(frame_a) - int(frame_b))
            score = float(np.dot(feature[obs_a], feature[obs_b]))
            if gap < best_gap or (gap == best_gap and score > best_score):
                best_gap = gap
                best_score = score
                best_a = obs_a
                best_b = obs_b
    return best_score, best_a, best_b, best_gap


def _max_da3_between(
    a: dict[str, Any],
    b: dict[str, Any],
    da3_pair_map: dict[tuple[int, int], dict[str, float]],
    *,
    min_da3_score: float,
    max_pair_broad_risk: float,
) -> tuple[float, float]:
    best_score = 0.0
    best_risk = 0.0
    for obs_a in a["obs_indices"]:
        for obs_b in b["obs_indices"]:
            row = da3_pair_map.get(_pair_key(int(obs_a), int(obs_b)))
            if row is None:
                continue
            score = float(row["da3_score"])
            risk = float(row["carrier_broad_risk"])
            if score >= min_da3_score and risk <= max_pair_broad_risk and score > best_score:
                best_score = score
                best_risk = risk
    return best_score, best_risk


def _merge_scene(
    *,
    scene: str,
    base_rows: list[dict[str, Any]],
    payload: dict[str, np.ndarray],
    da3_pair_map: dict[tuple[int, int], dict[str, float]],
    phase2_summary: dict[str, Any],
    variant: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    tracks = _component_tracks(rows=base_rows, payload=payload)
    object_ids = sorted(tracks)
    uf = UnionFind(object_ids)
    comp_frames: dict[str, set[int]] = {oid: set(tracks[oid]["frame_set"]) for oid in object_ids}
    edge_candidates: list[dict[str, Any]] = []
    for i, oid_a in enumerate(object_ids[:-1]):
        track_a = tracks[oid_a]
        for oid_b in object_ids[i + 1 :]:
            track_b = tracks[oid_b]
            if track_a["frame_set"] & track_b["frame_set"]:
                continue
            endpoint_score, endpoint_a, endpoint_b, nearest_gap = _closest_endpoint_score(track_a, track_b, payload["feature"])
            if nearest_gap > int(variant["max_fragment_gap"]):
                continue
            proto_score = float(np.dot(track_a["prototype"], track_b["prototype"]))
            da3_score, da3_risk = _max_da3_between(
                track_a,
                track_b,
                da3_pair_map,
                min_da3_score=float(variant.get("min_da3_score", 0.0)),
                max_pair_broad_risk=float(variant.get("max_pair_broad_risk", 1.01)),
            )
            da3_mode = str(variant.get("da3_mode", "none"))
            has_da3 = da3_score >= float(variant.get("min_da3_score", 0.0)) and da3_score > 0.0
            combined = proto_score + 0.25 * endpoint_score
            if da3_mode == "boost" and has_da3:
                combined += float(variant.get("da3_bonus", 0.0)) * da3_score
            keep = (
                proto_score >= float(variant["proto_threshold"])
                and endpoint_score >= float(variant["endpoint_threshold"])
                and combined >= float(variant["combined_threshold"])
            )
            reject_reason = ""
            if da3_mode == "required" and not has_da3:
                keep = False
                reject_reason = "missing_required_da3_validation"
            elif not keep:
                reject_reason = "below_proto_endpoint_or_combined_threshold"
            edge_candidates.append(
                {
                    "object_a": oid_a,
                    "object_b": oid_b,
                    "proto_affinity": proto_score,
                    "endpoint_affinity": endpoint_score,
                    "nearest_frame_gap": int(nearest_gap),
                    "endpoint_obs_a": int(endpoint_a),
                    "endpoint_obs_b": int(endpoint_b),
                    "da3_score": da3_score,
                    "carrier_broad_risk": da3_risk,
                    "combined_affinity": combined,
                    "has_da3_validation": bool(has_da3),
                    "candidate_keep": bool(keep),
                    "reject_reason": reject_reason,
                }
            )
    edge_candidates.sort(key=lambda row: float(row["combined_affinity"]), reverse=True)
    edge_rows: list[dict[str, Any]] = []
    accepted_count = 0
    for rank, edge in enumerate(edge_candidates):
        oid_a = str(edge["object_a"])
        oid_b = str(edge["object_b"])
        ra = uf.find(oid_a)
        rb = uf.find(oid_b)
        accepted = False
        reject_reason = str(edge["reject_reason"])
        if bool(edge["candidate_keep"]) and ra != rb:
            if comp_frames[ra] & comp_frames[rb]:
                reject_reason = "same_frame_fragment_conflict"
            else:
                new_root = uf.union(ra, rb)
                old_a = comp_frames.pop(ra, set())
                old_b = comp_frames.pop(rb, set())
                comp_frames[new_root] = set(old_a) | set(old_b)
                accepted = True
                accepted_count += 1
        edge_rows.append(
            {
                "schema_version": "stream4d_v103_phase6h_fragment_stitch_edge_row_v1",
                "phase_id": PHASE_ID,
                "variant_id": str(variant["variant_id"]),
                "base_variant_id": str(variant["base_variant_id"]),
                "scene_id": scene,
                "edge_rank": int(rank),
                **edge,
                "accepted_union": bool(accepted),
                "reject_reason": reject_reason,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )

    groups: dict[str, list[str]] = defaultdict(list)
    for oid in object_ids:
        groups[uf.find(oid)].append(oid)
    frame_ids = [int(v) for v in phase2_summary["frame_ids"]]
    cluster_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for idx, (_root, members) in enumerate(sorted(groups.items(), key=lambda item: (-len(set().union(*(tracks[m]["frame_set"] for m in item[1]))), item[0]))):
        rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        obs_indices: list[int] = []
        for member in members:
            for frame, row in tracks[member]["frames"].items():
                rows_by_frame[int(frame)].append(row)
                obs_indices.append(int(row["selected_mask_observation_index"]))
        selected: dict[int, dict[str, Any]] = {}
        for frame, candidates in rows_by_frame.items():
            selected[int(frame)] = max(
                candidates,
                key=lambda row: (
                    float(row.get("object_score", 0.0)),
                    int(row.get("support_count", 0)),
                    -int(row.get("selected_mask_id", 0)),
                ),
            )
        selected_obs = [int(row["selected_mask_observation_index"]) for row in selected.values()]
        selected_broad = float(np.mean(payload["mask_is_broad"][selected_obs])) if selected_obs else 0.0
        selected_object = float(np.mean(payload["mask_is_object_like"][selected_obs])) if selected_obs else 0.0
        score = float(len(selected) / 32.0) * max(0.05, 1.0 - 0.50 * selected_broad) * max(
            0.05, min(1.0, float(np.mean([tracks[m]["mean_object_score"] for m in members])) * 4.0)
        )
        object_id = f"{variant['variant_id']}:{scene}:c0000:stitched_{idx:05d}"
        cluster_rows.append(
            {
                "schema_version": "stream4d_v103_phase6h_fragment_stitch_cluster_row_v1",
                "phase_id": PHASE_ID,
                "variant_id": str(variant["variant_id"]),
                "base_variant_id": str(variant["base_variant_id"]),
                "scene_id": scene,
                "window_id": "c0000",
                "mv_object_id": object_id,
                "fragment_count": int(len(members)),
                "frame_count": int(len(selected)),
                "object_score": score,
                "selected_broad_mask_ratio": selected_broad,
                "selected_object_like_mask_ratio": selected_object,
                "mean_support_count": float(np.mean(payload["support_count"][selected_obs])) if selected_obs else 0.0,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        for frame, row in selected.items():
            frame_rows.append(
                {
                    "schema_version": "stream4d_v103_phase6h_fragment_stitch_frame_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": str(variant["variant_id"]),
                    "base_variant_id": str(variant["base_variant_id"]),
                    "mv_object_id": object_id,
                    "object_id": object_id,
                    "scene_id": scene,
                    "chunk_id": "c0000",
                    "window_id": "c0000",
                    "frame_local_index": int(frame),
                    "frame_id": int(frame_ids[int(frame)]),
                    "selected_mask_id": int(row["selected_mask_id"]),
                    "mask_id_or_generated_id": int(row["selected_mask_id"]),
                    "selected_mask_observation_index": int(row["selected_mask_observation_index"]),
                    "object_score": score,
                    "score": score,
                    "support_count": int(row.get("support_count", 0)),
                    "emit_policy": "fragment_stitch_keep_best_base_mask_per_frame",
                    "readout_mode": "phase6h_temporal_fragment_stitch",
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    diag = {
        "base_fragment_count": int(len(object_ids)),
        "merge_candidate_count": int(len(edge_candidates)),
        "accepted_merge_count": int(accepted_count),
        "stitched_cluster_count": int(len(cluster_rows)),
        "emitted_frame_row_count": int(len(frame_rows)),
    }
    return cluster_rows, frame_rows, edge_rows, diag


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v103 Phase6h: stitch high-quality temporal fragments into longer GT-free mask tubes.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--phase6g-root", default=str(DEFAULT_PHASE6G_ROOT))
    parser.add_argument("--phase9n-root", default=str(DEFAULT_PHASE9N_ROOT))
    parser.add_argument("--phase5-root", default=str(DEFAULT_PHASE5_ROOT))
    parser.add_argument("--scene0011-phase2-root", default=str(DEFAULT_PHASE2_SCENE0011))
    parser.add_argument("--scene0050-phase2-root", default=str(DEFAULT_PHASE2_SCENE0050))
    parser.add_argument("--baseline-rows", default=str(DEFAULT_BASELINE_ROWS))
    parser.add_argument("--scene", default="all", choices=["all", "scene0011_00", "scene0050_00"])
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
    phase6g_root = _project(args.phase6g_root)
    phase9n_root = _project(args.phase9n_root)
    phase5_root = _project(args.phase5_root)
    phase2_roots = {
        "scene0011_00": _project(args.scene0011_phase2_root),
        "scene0050_00": _project(args.scene0050_phase2_root),
    }
    if args.scene != "all":
        phase2_roots = {args.scene: phase2_roots[args.scene]}
    phase2_summaries = {scene: _read_json(root / "summary.json") for scene, root in phase2_roots.items()}
    phase5_payloads = {scene: _load_phase5_payload(phase5_root, scene) for scene in phase2_roots}
    da3_pair_maps = {scene: _load_da3_pair_map(phase9n_root, scene) for scene in phase2_roots}
    baseline = _load_baseline(_project(args.baseline_rows))

    all_cluster_rows: list[dict[str, Any]] = []
    all_frame_rows: list[dict[str, Any]] = []
    all_selected_rows: list[dict[str, Any]] = []
    all_edge_rows: list[dict[str, Any]] = []
    all_diag_rows: list[dict[str, Any]] = []
    all_metric_rows: list[dict[str, Any]] = []
    all_window_rows: list[dict[str, Any]] = []

    for variant in VARIANTS:
        scene_rows: dict[str, list[dict[str, Any]]] = {}
        for scene, payload in phase5_payloads.items():
            base_rows = _load_base_rows(phase6g_root, str(variant["base_variant_id"]), scene)
            clusters, frames, edges, diag = _merge_scene(
                scene=scene,
                base_rows=base_rows,
                payload=payload,
                da3_pair_map=da3_pair_maps[scene],
                phase2_summary=phase2_summaries[scene],
                variant=variant,
            )
            scene_rows[scene] = frames
            all_cluster_rows.extend(clusters)
            all_frame_rows.extend(frames)
            all_edge_rows.extend(edges[:40000])
            all_diag_rows.append(
                {
                    "schema_version": "stream4d_v103_phase6h_scene_diag_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": str(variant["variant_id"]),
                    "base_variant_id": str(variant["base_variant_id"]),
                    "scene_id": scene,
                    **diag,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
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
        for row in window_rows:
            row["phase_id"] = PHASE_ID
        aggregate.update(
            {
                "phase_id": PHASE_ID,
                "base_variant_id": str(variant["base_variant_id"]),
                "proto_threshold": float(variant["proto_threshold"]),
                "endpoint_threshold": float(variant["endpoint_threshold"]),
                "combined_threshold": float(variant["combined_threshold"]),
                "max_fragment_gap": int(variant["max_fragment_gap"]),
                "da3_mode": str(variant.get("da3_mode", "none")),
                "pixel_collision_count": int(pixel_collision_count),
                "missing_mask_raster_count": int(missing_count),
                "metric_scope": "first32_dev_subset_window_mean; temporal fragment stitch",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        all_metric_rows.append(aggregate)
        all_window_rows.extend(window_rows)
        all_selected_rows.extend(selected_rows)

    baseline_ap = float(baseline.get("MV_AP_window", 0.0))
    baseline_ap50 = float(baseline.get("MV_AP50_window", 0.0))
    best = max(all_metric_rows, key=lambda row: (float(row.get("MV_AP_window", 0.0)), float(row.get("MV_AP50_window", 0.0))), default={})
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for row in all_metric_rows:
        checks = [
            ("uses_gt_for_prediction_false", not bool(row["uses_gt_for_prediction"]), row["uses_gt_for_prediction"], False),
            ("uses_future_false", not bool(row["uses_future"]), row["uses_future"], False),
            ("MV_AP_window_ge_baseline_minus_0p003", float(row["MV_AP_window"]) >= baseline_ap - 0.003, row["MV_AP_window"], baseline_ap - 0.003),
            ("MV_AP50_window_ge_baseline_minus_0p006", float(row["MV_AP50_window"]) >= baseline_ap50 - 0.006, row["MV_AP50_window"], baseline_ap50 - 0.006),
        ]
        for name, ok, observed, required in checks:
            gate_rows.append(
                {
                    "schema_version": "stream4d_v103_phase6h_gate_row_v1",
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
                        "schema_version": "stream4d_v103_phase6h_failure_row_v1",
                        "phase_id": PHASE_ID,
                        "variant_id": row["variant_id"],
                        "failure_id": name,
                        "severity": "blocking",
                        "evidence": f"observed={observed} required={required}",
                        "repair_direction": "Fragment stitching still fails the raw local-object gate; inspect whether merges over-fragment scene0011 or over-merge scene0050 before history phases.",
                    }
                )

    _write_csv(out / "fragment_stitch_cluster_rows.csv", all_cluster_rows)
    _write_csv(out / "fragment_stitch_frame_rows.csv", all_frame_rows)
    _write_csv(out / "fragment_stitch_eval_selected_rows.csv", all_selected_rows)
    _write_csv(out / "fragment_stitch_edge_rows.csv", all_edge_rows)
    _write_csv(out / "fragment_stitch_scene_diag_rows.csv", all_diag_rows)
    _write_csv(out / "fragment_stitch_metric_rows.csv", all_metric_rows)
    _write_csv(out / "fragment_stitch_window_rows.csv", all_window_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    summary = {
        "schema_version": "stream4d_v103_phase6h_temporal_fragment_stitch_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": "PASS_PHASE6H_TEMPORAL_FRAGMENT_STITCH" if not failure_rows else "NO_GO_PHASE6H_TEMPORAL_FRAGMENT_STITCH",
        "phase6h_pass": not failure_rows,
        "failure_count": len(failure_rows),
        "best_variant_id": best.get("variant_id", ""),
        "best_MV_AP_window": best.get("MV_AP_window", ""),
        "best_MV_AP50_window": best.get("MV_AP50_window", ""),
        "baseline_contract": baseline,
        "phase6g_root": _rel(phase6g_root),
        "phase9n_root": _rel(phase9n_root),
        "phase5_root": _rel(phase5_root),
        "variant_count": len(VARIANTS),
        "scene_ids": sorted(phase2_roots),
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
        "truthfulness_note": "Phase6h stitches Phase6g short mask tubes using only GT-free fragment prototypes, endpoint feature similarity, and optional DA3 pair validation/boost. GT is used only by the canonical evaluator.",
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "fragment_stitch_cluster_rows": _rel(out / "fragment_stitch_cluster_rows.csv"),
            "fragment_stitch_frame_rows": _rel(out / "fragment_stitch_frame_rows.csv"),
            "fragment_stitch_eval_selected_rows": _rel(out / "fragment_stitch_eval_selected_rows.csv"),
            "fragment_stitch_edge_rows": _rel(out / "fragment_stitch_edge_rows.csv"),
            "fragment_stitch_scene_diag_rows": _rel(out / "fragment_stitch_scene_diag_rows.csv"),
            "fragment_stitch_metric_rows": _rel(out / "fragment_stitch_metric_rows.csv"),
            "fragment_stitch_window_rows": _rel(out / "fragment_stitch_window_rows.csv"),
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
