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
    _pair_values,
    _project,
    _read_json,
    _rel,
    _write_csv,
    _write_json,
)


PHASE_ID = "v103_phase6g_temporal_continuity_tracklet_birth"
DEFAULT_PHASE9N_ROOT = STREAM3D_ROOT / "outputs/audit/v103_phase9n_da3_bridge_pair_fused_phase4_r6_i13_e3_veto_ratio_all_pairs"
DEFAULT_PHASE5_ROOT = STREAM3D_ROOT / "outputs/audit/v103_phase5_da3_bridge_pair_phase9n_r6_i13_e3_veto_ratio_all_pairs"
DEFAULT_BASELINE_ROWS = STREAM3D_ROOT / "outputs/audit/v103_phase0_contract/baseline_metric_rows.csv"
DEFAULT_OUT = STREAM3D_ROOT / "outputs/audit/v103_phase6g_temporal_continuity_tracklet_birth_r1_i13"


VARIANTS: list[dict[str, Any]] = [
    {
        "variant_id": "G0_obj_nonbroad_gap1_tau055_top2_min2",
        "node_policy": "object_like_non_broad",
        "max_frame_gap": 1,
        "temporal_threshold": 0.55,
        "topk_per_mask": 2,
        "min_object_frames": 2,
        "da3_mode": "none",
    },
    {
        "variant_id": "G1_obj_nonbroad_gap1_tau060_top2_min2",
        "node_policy": "object_like_non_broad",
        "max_frame_gap": 1,
        "temporal_threshold": 0.60,
        "topk_per_mask": 2,
        "min_object_frames": 2,
        "da3_mode": "none",
    },
    {
        "variant_id": "G2_obj_nonbroad_gap2_tau065_top2_min2",
        "node_policy": "object_like_non_broad",
        "max_frame_gap": 2,
        "temporal_threshold": 0.65,
        "topk_per_mask": 2,
        "min_object_frames": 2,
        "da3_mode": "none",
    },
    {
        "variant_id": "G3_supported_nonbroad_gap1_tau070_top1_min2",
        "node_policy": "supported_non_broad",
        "max_frame_gap": 1,
        "temporal_threshold": 0.70,
        "topk_per_mask": 1,
        "min_object_frames": 2,
        "da3_mode": "none",
    },
    {
        "variant_id": "G4_supported_nonbroad_gap2_tau075_top1_min3",
        "node_policy": "supported_non_broad",
        "max_frame_gap": 2,
        "temporal_threshold": 0.75,
        "topk_per_mask": 1,
        "min_object_frames": 3,
        "da3_mode": "none",
    },
    {
        "variant_id": "G5_obj_nonbroad_da3boost_gap1_tau050_bonus025_top2_min2",
        "node_policy": "object_like_non_broad",
        "max_frame_gap": 1,
        "temporal_threshold": 0.50,
        "combined_threshold": 0.58,
        "topk_per_mask": 2,
        "min_object_frames": 2,
        "da3_mode": "boost",
        "da3_bonus": 0.25,
        "min_da3_score": 0.40,
        "max_pair_broad_risk": 0.50,
    },
    {
        "variant_id": "G6_obj_nonbroad_da3required_gap2_tau045_rel040_top2_min2",
        "node_policy": "object_like_non_broad",
        "max_frame_gap": 2,
        "temporal_threshold": 0.45,
        "combined_threshold": 0.45,
        "topk_per_mask": 2,
        "min_object_frames": 2,
        "da3_mode": "required",
        "min_da3_score": 0.40,
        "max_pair_broad_risk": 0.50,
    },
    {
        "variant_id": "G7_supported_nonbroad_da3boost_gap1_tau060_bonus020_top2_min2",
        "node_policy": "supported_non_broad",
        "max_frame_gap": 1,
        "temporal_threshold": 0.60,
        "combined_threshold": 0.66,
        "topk_per_mask": 2,
        "min_object_frames": 2,
        "da3_mode": "boost",
        "da3_bonus": 0.20,
        "min_da3_score": 0.40,
        "max_pair_broad_risk": 0.50,
        "require_object_like_component": True,
    },
    {
        "variant_id": "G8_obj_nonbroad_da3boost_gap4_tau055_bonus015_top1_min3",
        "node_policy": "object_like_non_broad",
        "max_frame_gap": 4,
        "temporal_threshold": 0.55,
        "combined_threshold": 0.60,
        "topk_per_mask": 1,
        "min_object_frames": 3,
        "da3_mode": "boost",
        "da3_bonus": 0.15,
        "min_da3_score": 0.40,
        "max_pair_broad_risk": 0.50,
    },
]


class UnionFind:
    def __init__(self, items: list[int]) -> None:
        self.parent = {int(item): int(item) for item in items}
        self.size = {int(item): 1 for item in items}

    def find(self, item: int) -> int:
        item = int(item)
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: int, b: int) -> int:
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
                    "semantic_residual_cosine": float(row.get("semantic_residual_cosine", 0.0) or 0.0),
                    "broad_contamination_score": float(row.get("broad_contamination_score", 1.0) or 1.0),
                }
    return out


def _node_mask(payload: dict[str, np.ndarray], variant: dict[str, Any]) -> np.ndarray:
    support = payload["support_count"] >= int(variant.get("min_support_count", 1))
    non_broad = ~payload["mask_is_broad"].astype(bool)
    object_like = payload["mask_is_object_like"].astype(bool)
    policy = str(variant["node_policy"])
    if policy == "object_like_non_broad":
        return object_like & non_broad
    if policy == "supported_non_broad":
        return support & non_broad
    if policy == "object_like_or_supported_non_broad":
        return non_broad & (object_like | support)
    raise ValueError(f"unsupported node_policy={policy}")


def _temporal_candidate_pairs(
    *,
    payload: dict[str, np.ndarray],
    node_mask: np.ndarray,
    max_frame_gap: int,
) -> np.ndarray:
    by_frame: dict[int, list[int]] = defaultdict(list)
    for idx in np.flatnonzero(node_mask).tolist():
        by_frame[int(payload["mask_frame"][int(idx)])].append(int(idx))
    pairs: list[tuple[int, int]] = []
    frames = sorted(by_frame)
    frame_set = set(frames)
    for frame in frames:
        for gap in range(1, int(max_frame_gap) + 1):
            other = frame + gap
            if other not in frame_set:
                continue
            for a in by_frame[frame]:
                for b in by_frame[other]:
                    pairs.append((int(a), int(b)))
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64)


def _filter_rank_edges(
    *,
    payload: dict[str, np.ndarray],
    candidate_pairs: np.ndarray,
    temporal_scores: np.ndarray,
    da3_pair_map: dict[tuple[int, int], dict[str, float]],
    variant: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    da3_mode = str(variant.get("da3_mode", "none"))
    min_da3 = float(variant.get("min_da3_score", 0.0))
    max_pair_broad_risk = float(variant.get("max_pair_broad_risk", 1.01))
    da3_bonus = float(variant.get("da3_bonus", 0.0))
    temporal_threshold = float(variant["temporal_threshold"])
    combined_threshold = float(variant.get("combined_threshold", temporal_threshold))
    for idx, ((a, b), temporal_score) in enumerate(zip(candidate_pairs.tolist(), temporal_scores.tolist())):
        key = _pair_key(int(a), int(b))
        da3 = da3_pair_map.get(key)
        da3_score = float(da3["da3_score"]) if da3 is not None else 0.0
        broad_risk = float(da3["carrier_broad_risk"]) if da3 is not None else 0.0
        has_da3 = da3 is not None and da3_score >= min_da3 and broad_risk <= max_pair_broad_risk
        combined = float(temporal_score) + (da3_bonus * da3_score if has_da3 else 0.0)
        keep = float(temporal_score) >= temporal_threshold and combined >= combined_threshold
        reject_reason = ""
        if da3_mode == "required" and not has_da3:
            keep = False
            reject_reason = "missing_required_da3_validation"
        elif da3 is not None and broad_risk > max_pair_broad_risk:
            keep = False
            reject_reason = "da3_pair_broad_risk_veto"
        elif not keep:
            reject_reason = "below_temporal_or_combined_threshold"
        if keep:
            rows.append(
                {
                    "candidate_rank": int(idx),
                    "mask_a": int(a),
                    "mask_b": int(b),
                    "frame_a": int(payload["mask_frame"][int(a)]),
                    "frame_b": int(payload["mask_frame"][int(b)]),
                    "temporal_affinity": float(temporal_score),
                    "da3_score": da3_score,
                    "carrier_broad_risk": broad_risk,
                    "combined_affinity": combined,
                    "has_da3_validation": bool(has_da3),
                    "reject_reason": reject_reason,
                }
            )
    if not rows:
        return rows
    topk = int(variant.get("topk_per_mask", 0))
    if topk > 0:
        selected = np.zeros((len(rows),), dtype=bool)
        by_mask: dict[int, list[int]] = defaultdict(list)
        for idx, row in enumerate(rows):
            by_mask[int(row["mask_a"])].append(idx)
            by_mask[int(row["mask_b"])].append(idx)
        for indices in by_mask.values():
            order = sorted(indices, key=lambda i: float(rows[i]["combined_affinity"]), reverse=True)[:topk]
            selected[np.asarray(order, dtype=np.int64)] = True
        rows = [row for idx, row in enumerate(rows) if bool(selected[idx])]
    return sorted(rows, key=lambda row: float(row["combined_affinity"]), reverse=True)


def _cannot_links(nodes: list[int], payload: dict[str, np.ndarray]) -> set[tuple[int, int]]:
    by_frame: dict[int, list[int]] = defaultdict(list)
    for node in nodes:
        by_frame[int(payload["mask_frame"][int(node)])].append(int(node))
    out: set[tuple[int, int]] = set()
    for masks in by_frame.values():
        for i, a in enumerate(masks[:-1]):
            for b in masks[i + 1 :]:
                a_specific = bool(payload["mask_is_object_like"][a]) and not bool(payload["mask_is_broad"][a])
                b_specific = bool(payload["mask_is_object_like"][b]) and not bool(payload["mask_is_broad"][b])
                if a_specific and b_specific:
                    out.add(_pair_key(a, b))
    return out


def _would_violate(comp_members: dict[int, set[int]], cannot: set[tuple[int, int]], ra: int, rb: int) -> bool:
    a_members = comp_members.get(int(ra), {int(ra)})
    b_members = comp_members.get(int(rb), {int(rb)})
    if len(a_members) > len(b_members):
        a_members, b_members = b_members, a_members
    for a in a_members:
        for b in b_members:
            if _pair_key(a, b) in cannot:
                return True
    return False


def _cluster_scene(
    *,
    scene: str,
    payload: dict[str, np.ndarray],
    da3_pair_map: dict[tuple[int, int], dict[str, float]],
    phase2_summary: dict[str, Any],
    variant: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    node_mask = _node_mask(payload, variant)
    candidate_pairs = _temporal_candidate_pairs(
        payload=payload,
        node_mask=node_mask,
        max_frame_gap=int(variant["max_frame_gap"]),
    )
    temporal_scores = _pair_values(
        payload["feature"],
        candidate_pairs,
        pair_affinity_mode="strict_leave_two_out_bucket_zeroed",
        device=device,
        batch_size=batch_size,
    )
    ranked = _filter_rank_edges(
        payload=payload,
        candidate_pairs=candidate_pairs,
        temporal_scores=temporal_scores,
        da3_pair_map=da3_pair_map,
        variant=variant,
    )
    nodes = sorted({int(row["mask_a"]) for row in ranked} | {int(row["mask_b"]) for row in ranked})
    if not nodes:
        return [], [], [], {
            "candidate_pair_count": int(candidate_pairs.shape[0]),
            "kept_edge_count": 0,
            "accepted_edge_count": 0,
            "node_count": 0,
            "cluster_count": 0,
        }

    uf = UnionFind(nodes)
    comp_members: dict[int, set[int]] = {node: {node} for node in nodes}
    cannot = _cannot_links(nodes, payload)
    node_degree: dict[int, int] = defaultdict(int)
    node_score: dict[int, float] = defaultdict(float)
    edge_rows: list[dict[str, Any]] = []
    accepted_count = 0
    for rank, row in enumerate(ranked):
        a = int(row["mask_a"])
        b = int(row["mask_b"])
        ra = uf.find(a)
        rb = uf.find(b)
        accepted = False
        reject_reason = str(row.get("reject_reason", ""))
        if ra != rb:
            if _would_violate(comp_members, cannot, ra, rb):
                reject_reason = "specific_same_frame_cannot_link"
            else:
                new_root = uf.union(ra, rb)
                old_a = comp_members.pop(ra, {ra})
                old_b = comp_members.pop(rb, {rb})
                comp_members[new_root] = set(old_a) | set(old_b)
                node_degree[a] += 1
                node_degree[b] += 1
                node_score[a] += float(row["combined_affinity"])
                node_score[b] += float(row["combined_affinity"])
                accepted = True
                accepted_count += 1
        edge_rows.append(
            {
                "schema_version": "stream4d_v103_phase6g_temporal_edge_row_v1",
                "phase_id": PHASE_ID,
                "variant_id": str(variant["variant_id"]),
                "scene_id": scene,
                "edge_rank": int(rank),
                "mask_a": a,
                "mask_b": b,
                "frame_a": int(row["frame_a"]),
                "frame_b": int(row["frame_b"]),
                "temporal_affinity": float(row["temporal_affinity"]),
                "da3_score": float(row["da3_score"]),
                "carrier_broad_risk": float(row["carrier_broad_risk"]),
                "combined_affinity": float(row["combined_affinity"]),
                "has_da3_validation": bool(row["has_da3_validation"]),
                "accepted_union": bool(accepted),
                "reject_reason": reject_reason,
                "edge_policy": f"temporal_strict_l2o_da3_{variant.get('da3_mode', 'none')}",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )

    groups: dict[int, list[int]] = defaultdict(list)
    for node in nodes:
        groups[uf.find(node)].append(node)
    frame_ids = [int(v) for v in phase2_summary["frame_ids"]]
    cluster_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    object_idx = 0
    for _root, masks in sorted(groups.items(), key=lambda item: (-len({int(payload["mask_frame"][m]) for m in item[1]}), item[0])):
        by_frame: dict[int, list[int]] = defaultdict(list)
        for mask_idx in masks:
            by_frame[int(payload["mask_frame"][mask_idx])].append(int(mask_idx))
        if len(by_frame) < int(variant["min_object_frames"]):
            continue
        if bool(variant.get("require_object_like_component", False)) and not any(payload["mask_is_object_like"][m] for m in masks):
            continue
        selected: dict[int, int] = {}
        for frame, candidates in by_frame.items():
            best = max(
                candidates,
                key=lambda m: (
                    int(payload["mask_is_object_like"][m]),
                    int(node_degree.get(m, 0)),
                    float(node_score.get(m, 0.0)),
                    int(payload["support_count"][m]),
                    -int(payload["mask_label"][m]),
                ),
            )
            selected[int(frame)] = int(best)
        selected_masks = list(selected.values())
        selected_broad = float(np.mean(payload["mask_is_broad"][selected_masks])) if selected_masks else 0.0
        selected_object = float(np.mean(payload["mask_is_object_like"][selected_masks])) if selected_masks else 0.0
        mean_edge = float(np.mean([node_score.get(m, 0.0) / max(1, node_degree.get(m, 0)) for m in selected_masks])) if selected_masks else 0.0
        score = float(len(selected) / 32.0) * max(0.05, 1.0 - 0.50 * selected_broad) * max(0.10, min(1.0, mean_edge))
        object_id = f"{variant['variant_id']}:{scene}:c0000:temporal_{object_idx:05d}"
        object_idx += 1
        cluster_rows.append(
            {
                "schema_version": "stream4d_v103_phase6g_temporal_cluster_row_v1",
                "phase_id": PHASE_ID,
                "variant_id": str(variant["variant_id"]),
                "scene_id": scene,
                "window_id": "c0000",
                "mv_object_id": object_id,
                "mask_count": int(len(masks)),
                "frame_count": int(len(selected)),
                "object_score": score,
                "selected_broad_mask_ratio": selected_broad,
                "selected_object_like_mask_ratio": selected_object,
                "mean_support_count": float(np.mean(payload["support_count"][masks])) if masks else 0.0,
                "mean_selected_edge_score": mean_edge,
                "node_policy": str(variant["node_policy"]),
                "da3_mode": str(variant.get("da3_mode", "none")),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        for frame, mask_idx in selected.items():
            frame_rows.append(
                {
                    "schema_version": "stream4d_v103_phase6g_temporal_frame_mask_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": str(variant["variant_id"]),
                    "mv_object_id": object_id,
                    "object_id": object_id,
                    "scene_id": scene,
                    "chunk_id": "c0000",
                    "window_id": "c0000",
                    "frame_local_index": int(frame),
                    "frame_id": int(frame_ids[int(frame)]),
                    "selected_mask_id": int(payload["mask_label"][mask_idx]),
                    "mask_id_or_generated_id": int(payload["mask_label"][mask_idx]),
                    "object_score": score,
                    "score": score,
                    "support_count": int(payload["support_count"][mask_idx]),
                    "node_policy": str(variant["node_policy"]),
                    "emit_policy": "temporal_wta_by_objectlike_degree_support",
                    "readout_mode": "phase6g_temporal_continuity_tracklet_birth",
                    "selected_mask_is_broad": bool(payload["mask_is_broad"][mask_idx]),
                    "selected_mask_is_object_like": bool(payload["mask_is_object_like"][mask_idx]),
                    "node_degree": int(node_degree.get(mask_idx, 0)),
                    "node_score_sum": float(node_score.get(mask_idx, 0.0)),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    scene_diag = {
        "candidate_pair_count": int(candidate_pairs.shape[0]),
        "kept_edge_count": int(len(ranked)),
        "accepted_edge_count": int(accepted_count),
        "node_count": int(len(nodes)),
        "cluster_count": int(len(cluster_rows)),
        "emitted_frame_row_count": int(len(frame_rows)),
        "da3_validated_kept_edge_count": int(sum(1 for row in edge_rows if row["has_da3_validation"])),
    }
    return cluster_rows, frame_rows, edge_rows, scene_diag


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v103 Phase6g: GT-free temporal continuity seed proposal with DA3 boost/veto.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--phase9n-root", default=str(DEFAULT_PHASE9N_ROOT))
    parser.add_argument("--phase5-root", default=str(DEFAULT_PHASE5_ROOT))
    parser.add_argument("--scene0011-phase2-root", default=str(DEFAULT_PHASE2_SCENE0011))
    parser.add_argument("--scene0050-phase2-root", default=str(DEFAULT_PHASE2_SCENE0050))
    parser.add_argument("--baseline-rows", default=str(DEFAULT_BASELINE_ROWS))
    parser.add_argument("--scene", default="all", choices=["all", "scene0011_00", "scene0050_00"])
    parser.add_argument("--min-pred-pixels", type=int, default=20)
    parser.add_argument("--min-gt-pixels", type=int, default=20)
    parser.add_argument("--pair-batch-size", type=int, default=8192)
    parser.add_argument("--cupy-device-id", type=int, default=0)
    parser.add_argument("--disable-cupy-iou", action="store_true")
    return parser


def main() -> int:
    t0 = time.time()
    args = build_parser().parse_args()
    out = _project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")

    phase9n_root = _project(args.phase9n_root)
    phase5_root = _project(args.phase5_root)
    phase5_summary = _read_json(phase5_root / "summary.json")
    if not bool(phase5_summary.get("phase5_pass")):
        raise RuntimeError(f"Phase6g requires Phase5 pass: {phase5_root}")

    phase2_roots = {
        "scene0011_00": _project(args.scene0011_phase2_root),
        "scene0050_00": _project(args.scene0050_phase2_root),
    }
    if args.scene != "all":
        phase2_roots = {args.scene: phase2_roots[args.scene]}
    phase2_summaries = {scene: _read_json(root / "summary.json") for scene, root in phase2_roots.items()}
    phase5_payloads = {scene: _load_phase5_scene(phase5_root, scene) for scene in phase2_roots}
    da3_pair_maps = {scene: _load_da3_pair_map(phase9n_root, scene) for scene in phase2_roots}
    baseline = _load_baseline(_project(args.baseline_rows))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    all_metric_rows: list[dict[str, Any]] = []
    all_window_rows: list[dict[str, Any]] = []
    all_cluster_rows: list[dict[str, Any]] = []
    all_frame_rows: list[dict[str, Any]] = []
    all_selected_rows: list[dict[str, Any]] = []
    all_edge_rows: list[dict[str, Any]] = []
    all_scene_diag_rows: list[dict[str, Any]] = []

    for variant in VARIANTS:
        scene_rows: dict[str, list[dict[str, Any]]] = {}
        for scene, payload in phase5_payloads.items():
            clusters, frames, edges, diag = _cluster_scene(
                scene=scene,
                payload=payload,
                da3_pair_map=da3_pair_maps[scene],
                phase2_summary=phase2_summaries[scene],
                variant=variant,
                device=device,
                batch_size=int(args.pair_batch_size),
            )
            scene_rows[scene] = frames
            all_cluster_rows.extend(clusters)
            all_frame_rows.extend(frames)
            all_edge_rows.extend(edges[:30000])
            all_scene_diag_rows.append(
                {
                    "schema_version": "stream4d_v103_phase6g_scene_diag_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": str(variant["variant_id"]),
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
                "node_policy": str(variant["node_policy"]),
                "da3_mode": str(variant.get("da3_mode", "none")),
                "max_frame_gap": int(variant["max_frame_gap"]),
                "temporal_threshold": float(variant["temporal_threshold"]),
                "combined_threshold": float(variant.get("combined_threshold", variant["temporal_threshold"])),
                "topk_per_mask": int(variant["topk_per_mask"]),
                "min_object_frames": int(variant["min_object_frames"]),
                "pixel_collision_count": int(pixel_collision_count),
                "missing_mask_raster_count": int(missing_count),
                "metric_scope": "first32_dev_subset_window_mean; temporal continuity seed proposal",
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
                    "schema_version": "stream4d_v103_phase6g_gate_row_v1",
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
                        "schema_version": "stream4d_v103_phase6g_failure_row_v1",
                        "phase_id": PHASE_ID,
                        "variant_id": row["variant_id"],
                        "failure_id": name,
                        "severity": "blocking",
                        "evidence": f"observed={observed} required={required}",
                        "repair_direction": "Temporal continuity seeds did not recover the raw local-object gate; inspect whether scene0011 lacks object-like temporal edges or whether scoring suppresses the recovered masks.",
                    }
                )

    _write_csv(out / "temporal_cluster_rows.csv", all_cluster_rows)
    _write_csv(out / "temporal_frame_rows.csv", all_frame_rows)
    _write_csv(out / "temporal_eval_selected_rows.csv", all_selected_rows)
    _write_csv(out / "temporal_edge_rows.csv", all_edge_rows)
    _write_csv(out / "temporal_scene_diag_rows.csv", all_scene_diag_rows)
    _write_csv(out / "temporal_metric_rows.csv", all_metric_rows)
    _write_csv(out / "temporal_window_rows.csv", all_window_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    peak_mb = None
    if device.type == "cuda":
        peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    summary = {
        "schema_version": "stream4d_v103_phase6g_temporal_continuity_tracklet_birth_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": "PASS_PHASE6G_TEMPORAL_CONTINUITY_TRACKLET_BIRTH" if not failure_rows else "NO_GO_PHASE6G_TEMPORAL_CONTINUITY_TRACKLET_BIRTH",
        "phase6g_pass": not failure_rows,
        "failure_count": len(failure_rows),
        "best_variant_id": best.get("variant_id", ""),
        "best_MV_AP_window": best.get("MV_AP_window", ""),
        "best_MV_AP50_window": best.get("MV_AP50_window", ""),
        "baseline_contract": baseline,
        "phase9n_root": _rel(phase9n_root),
        "phase5_root": _rel(phase5_root),
        "variant_count": len(VARIANTS),
        "scene_ids": sorted(phase2_roots),
        "gpu_device": str(device),
        "gpu_memory_peak_MB": peak_mb,
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
        "truthfulness_note": "Phase6g uses GT-free mask features, adjacent-frame temporal continuity, and DA3 pair validation/boost only. GT is used only by the canonical evaluator and diagnostic window metrics.",
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "temporal_cluster_rows": _rel(out / "temporal_cluster_rows.csv"),
            "temporal_frame_rows": _rel(out / "temporal_frame_rows.csv"),
            "temporal_eval_selected_rows": _rel(out / "temporal_eval_selected_rows.csv"),
            "temporal_edge_rows": _rel(out / "temporal_edge_rows.csv"),
            "temporal_scene_diag_rows": _rel(out / "temporal_scene_diag_rows.csv"),
            "temporal_metric_rows": _rel(out / "temporal_metric_rows.csv"),
            "temporal_window_rows": _rel(out / "temporal_window_rows.csv"),
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
