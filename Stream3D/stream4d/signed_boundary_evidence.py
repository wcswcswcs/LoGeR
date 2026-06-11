from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .measurement_bank import MeasurementBank, json_safe
from .signed_surfel_graph import SignedSurfelGraph


@dataclass
class SignedBoundaryEvidence:
    scene: str
    variant: str
    merge_weight: np.ndarray
    cut_weight: np.ndarray
    cut_score: np.ndarray
    num_frames_used: np.ndarray
    meta: dict[str, Any]

    @property
    def num_edges(self) -> int:
        return int(self.cut_score.shape[0])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            scene=np.asarray(self.scene),
            variant=np.asarray(self.variant),
            merge_weight=self.merge_weight.astype(np.float32, copy=False),
            cut_weight=self.cut_weight.astype(np.float32, copy=False),
            cut_score=self.cut_score.astype(np.float32, copy=False),
            num_frames_used=self.num_frames_used.astype(np.int16, copy=False),
            meta_json=np.asarray(json.dumps(json_safe(self.meta), sort_keys=True)),
        )

    @classmethod
    def load(cls, path: Path) -> "SignedBoundaryEvidence":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                scene=str(data["scene"].item()),
                variant=str(data["variant"].item()),
                merge_weight=np.asarray(data["merge_weight"], dtype=np.float32),
                cut_weight=np.asarray(data["cut_weight"], dtype=np.float32),
                cut_score=np.asarray(data["cut_score"], dtype=np.float32),
                num_frames_used=np.asarray(data["num_frames_used"], dtype=np.int16),
                meta=json.loads(str(data["meta_json"].item())),
            )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def _variant_flags(variant: str) -> dict[str, bool]:
    if variant == "E0_mask_co_membership_baseline":
        return {"signed": False, "boundary": False, "temporal": False, "appearance": False, "single_frame": False, "shuffle": False}
    if variant == "E1_mask_signed":
        return {"signed": True, "boundary": False, "temporal": False, "appearance": False, "single_frame": False, "shuffle": False}
    if variant == "E2_mask_signed_boundary_safe":
        return {"signed": True, "boundary": True, "temporal": False, "appearance": False, "single_frame": False, "shuffle": False}
    if variant == "E3_mask_signed_depth_normal":
        return {"signed": True, "boundary": True, "temporal": False, "appearance": True, "single_frame": False, "shuffle": False}
    if variant == "E4_mask_signed_d4rt_temporal":
        return {"signed": True, "boundary": True, "temporal": True, "appearance": False, "single_frame": False, "shuffle": False}
    if variant == "E5_full_signed":
        return {"signed": True, "boundary": True, "temporal": True, "appearance": True, "single_frame": False, "shuffle": False}
    if variant == "E6_shuffle_d4rt":
        return {"signed": True, "boundary": True, "temporal": True, "appearance": True, "single_frame": False, "shuffle": True}
    if variant == "E7_no_temporal":
        return {"signed": True, "boundary": True, "temporal": False, "appearance": True, "single_frame": True, "shuffle": False}
    raise ValueError(f"Unsupported v18 evidence variant: {variant}")


def build_signed_boundary_evidence(
    bank: MeasurementBank,
    graph: SignedSurfelGraph,
    *,
    variant: str,
    boundary_safe_px: float = 3.0,
    cut_lambda: float = 1.0,
    merge_lambda: float = 0.65,
    bias: float = 0.0,
    seed: int = 18,
) -> SignedBoundaryEvidence:
    flags = _variant_flags(variant)
    visible = np.asarray(bank.visible_ok, dtype=bool)
    mask_id = np.asarray(bank.target_mask_id, dtype=np.int64)
    boundary = np.asarray(bank.boundary_distance, dtype=np.float32)
    confidence = np.asarray(bank.confidence, dtype=np.float32)
    rgb = np.asarray(bank.src_rgb, dtype=np.float32)
    src = np.asarray(graph.src, dtype=np.int64)
    dst = np.asarray(graph.dst, dtype=np.int64)

    if flags["shuffle"]:
        rng = np.random.default_rng(int(seed))
        perm = rng.permutation(bank.num_surfels)
        # Keep graph topology fixed, but break D4RT material identity for evidence reads.
        read_src = perm[src]
        read_dst = perm[dst]
    else:
        read_src = src
        read_dst = dst

    merge = np.zeros((graph.num_edges,), dtype=np.float32)
    cut = np.zeros((graph.num_edges,), dtype=np.float32)
    used = np.zeros((graph.num_edges,), dtype=np.int16)

    for edge_idx, (a, b) in enumerate(zip(read_src.tolist(), read_dst.tolist())):
        both_visible = visible[:, a] & visible[:, b]
        frames = np.flatnonzero(both_visible)
        if frames.size == 0:
            continue
        if flags["single_frame"]:
            frames = frames[:1]
        ids_a = mask_id[frames, a]
        ids_b = mask_id[frames, b]
        pos_a = ids_a > 0
        pos_b = ids_b > 0
        same_inside = pos_a & pos_b & (ids_a == ids_b)
        if flags["signed"]:
            diff_positive = pos_a & pos_b & (ids_a != ids_b)
            inside_outside = pos_a ^ pos_b
            cut_vote = diff_positive | inside_outside
        else:
            cut_vote = np.zeros_like(same_inside, dtype=bool)

        frame_weight = np.ones((frames.shape[0],), dtype=np.float32)
        if flags["temporal"]:
            frame_weight *= np.sqrt(np.clip(confidence[frames, a] * confidence[frames, b], 0.0, 1.0))
        if flags["boundary"]:
            safe_a = boundary[frames, a] >= float(boundary_safe_px)
            safe_b = boundary[frames, b] >= float(boundary_safe_px)
            merge_frame_weight = frame_weight * (safe_a & safe_b).astype(np.float32)
            cut_frame_weight = frame_weight * np.maximum(
                np.clip(boundary[frames, a] / max(float(boundary_safe_px), 1e-6), 0.0, 1.0),
                np.clip(boundary[frames, b] / max(float(boundary_safe_px), 1e-6), 0.0, 1.0),
            )
        else:
            merge_frame_weight = frame_weight
            cut_frame_weight = frame_weight

        merge[edge_idx] = float(np.sum(merge_frame_weight[same_inside]))
        cut[edge_idx] = float(np.sum(cut_frame_weight[cut_vote]))
        used[edge_idx] = int(frames.shape[0])

    if flags["appearance"] and graph.num_edges:
        rgb_dist = np.linalg.norm(rgb[src] - rgb[dst], axis=1).astype(np.float32)
        # Source RGB is a weak discontinuity proxy. It is intentionally bounded
        # so appearance cannot dominate signed mask evidence.
        cut += np.clip(rgb_dist / 0.35, 0.0, 1.0).astype(np.float32) * 0.25

    raw = float(cut_lambda) * cut - float(merge_lambda) * merge + float(bias)
    cut_score = _sigmoid(raw).astype(np.float32)
    return SignedBoundaryEvidence(
        scene=bank.scene,
        variant=variant,
        merge_weight=merge,
        cut_weight=cut,
        cut_score=cut_score,
        num_frames_used=used,
        meta={
            "algorithm": "v18_signed_boundary_evidence",
            "variant": variant,
            "boundary_safe_px": float(boundary_safe_px),
            "cut_lambda": float(cut_lambda),
            "merge_lambda": float(merge_lambda),
            "bias": float(bias),
            "seed": int(seed),
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic": False,
            "notes": "Non-GT edge evidence from predicted 2D mask membership, boundary distance, D4RT temporal confidence, and source RGB proxy.",
        },
    )


def summarize_signed_boundary_evidence(evidence: SignedBoundaryEvidence) -> dict[str, Any]:
    if evidence.num_edges == 0:
        q = {key: 0.0 for key in ("cut_score_p50", "cut_score_p90", "cut_score_p99")}
    else:
        q = {
            "cut_score_p50": float(np.percentile(evidence.cut_score, 50)),
            "cut_score_p90": float(np.percentile(evidence.cut_score, 90)),
            "cut_score_p99": float(np.percentile(evidence.cut_score, 99)),
        }
    return {
        "scene": evidence.scene,
        "variant": evidence.variant,
        "num_edges": int(evidence.num_edges),
        "merge_weight_mean": float(np.mean(evidence.merge_weight)) if evidence.num_edges else 0.0,
        "cut_weight_mean": float(np.mean(evidence.cut_weight)) if evidence.num_edges else 0.0,
        "cut_score_mean": float(np.mean(evidence.cut_score)) if evidence.num_edges else 0.0,
        "frames_used_mean": float(np.mean(evidence.num_frames_used)) if evidence.num_edges else 0.0,
        **q,
    }
