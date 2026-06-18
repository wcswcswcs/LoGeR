from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .semantic_part_tokens import SemanticPartToken


@dataclass(frozen=True)
class PartGraphEdge:
    token_i: int
    token_j: int
    semantic_affinity: float
    same_frame_cannot_link: bool
    boundary_penalty: float
    spatial_distance: float
    object_affinity: float
    diagnostic_same_gt: bool | None


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = np.asarray(vec_a, dtype=np.float32).reshape(-1)
    b = np.asarray(vec_b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def build_part_graph_edges(
    tokens: list[SemanticPartToken],
    *,
    semantic_affinity_mode: str = "cosine",
    structure_topk: int = 8,
    structure_min_affinity: float = 0.25,
    structure_decay: float = 0.95,
    structure_temporal_window: int | None = None,
    structure_temporal_rank_window: int | None = None,
) -> list[PartGraphEdge]:
    edges: list[PartGraphEdge] = []
    if not tokens:
        return edges
    semantic_matrix = _semantic_affinity_matrix(
        tokens,
        mode=str(semantic_affinity_mode),
        structure_topk=int(structure_topk),
        structure_min_affinity=float(structure_min_affinity),
        structure_decay=float(structure_decay),
        structure_temporal_window=structure_temporal_window,
        structure_temporal_rank_window=structure_temporal_rank_window,
    )
    max_h = max(t.centroid_y for t in tokens) + 1.0
    max_w = max(t.centroid_x for t in tokens) + 1.0
    scale = max(float(np.hypot(max_h, max_w)), 1.0)
    for idx, a in enumerate(tokens):
        for right_idx in range(idx + 1, len(tokens)):
            b = tokens[right_idx]
            semantic = float(semantic_matrix[idx, right_idx])
            same_frame = int(a.frame_id) == int(b.frame_id)
            boundary_penalty = max(0.0, float(a.boundary_contrast + b.boundary_contrast) * 0.5)
            spatial_distance = float(np.hypot(a.centroid_y - b.centroid_y, a.centroid_x - b.centroid_x) / scale)
            cannot = same_frame and a.mask_id != b.mask_id
            affinity = semantic - 0.15 * boundary_penalty - 0.10 * spatial_distance - (0.30 if cannot else 0.0)
            same_gt = None
            if a.diagnostic_gt_instance is not None and b.diagnostic_gt_instance is not None:
                same_gt = int(a.diagnostic_gt_instance) == int(b.diagnostic_gt_instance)
            edges.append(
                PartGraphEdge(
                    token_i=int(a.token_id),
                    token_j=int(b.token_id),
                    semantic_affinity=float(semantic),
                    same_frame_cannot_link=bool(cannot),
                    boundary_penalty=float(boundary_penalty),
                    spatial_distance=float(spatial_distance),
                    object_affinity=float(affinity),
                    diagnostic_same_gt=same_gt,
                )
            )
    return edges


def _semantic_affinity_matrix(
    tokens: list[SemanticPartToken],
    *,
    mode: str,
    structure_topk: int,
    structure_min_affinity: float,
    structure_decay: float,
    structure_temporal_window: int | None,
    structure_temporal_rank_window: int | None,
) -> np.ndarray:
    raw = _raw_cosine_matrix(tokens)
    if mode == "cosine":
        return raw
    if mode == "temporal_chain_structure":
        rank_window = 1 if structure_temporal_rank_window is None else max(1, int(structure_temporal_rank_window))
        support = _temporal_chain_support_matrix(
            raw,
            tokens,
            topk=int(structure_topk),
            min_affinity=float(structure_min_affinity),
            rank_window=rank_window,
        )
        structure = _widest_path_matrix(support)
        out = np.maximum(raw, float(structure_decay) * structure)
        np.fill_diagonal(out, 1.0)
        return out.astype(np.float32)
    temporal_window = None
    if mode == "temporal_widest_structure":
        temporal_window = None if structure_temporal_window is None else max(0, int(structure_temporal_window))
    support = _structure_support_matrix(
        raw,
        tokens,
        topk=int(structure_topk),
        min_affinity=float(structure_min_affinity),
        temporal_window=temporal_window,
    )
    if mode == "twohop_structure":
        structure = np.zeros_like(support, dtype=np.float32)
        for idx in range(support.shape[0]):
            structure[idx] = np.minimum(support[idx][None, :], support).max(axis=1)
        structure = np.maximum(structure, structure.T)
    elif mode in {"widest_structure", "temporal_widest_structure"}:
        structure = _widest_path_matrix(support)
    else:
        raise ValueError(f"unsupported semantic affinity mode: {mode}")
    out = np.maximum(raw, float(structure_decay) * structure)
    np.fill_diagonal(out, 1.0)
    return out.astype(np.float32)


def _raw_cosine_matrix(tokens: list[SemanticPartToken]) -> np.ndarray:
    features = np.asarray([np.asarray(token.feature, dtype=np.float32).reshape(-1) for token in tokens], dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(norms, 1e-8)
    return np.clip(features @ features.T, -1.0, 1.0).astype(np.float32)


def _structure_support_matrix(
    raw: np.ndarray,
    tokens: list[SemanticPartToken],
    *,
    topk: int,
    min_affinity: float,
    temporal_window: int | None,
) -> np.ndarray:
    n = int(raw.shape[0])
    support = np.zeros((n, n), dtype=np.float32)
    if n <= 1:
        return support
    k = max(1, min(int(topk), n - 1))
    for idx, token in enumerate(tokens):
        scores = np.asarray(raw[idx], dtype=np.float32).copy()
        scores[idx] = -np.inf
        for other_idx, other in enumerate(tokens):
            if int(token.frame_id) == int(other.frame_id) and int(token.mask_id) != int(other.mask_id):
                scores[other_idx] = -np.inf
            if temporal_window is not None and abs(int(token.frame_id) - int(other.frame_id)) > int(temporal_window):
                scores[other_idx] = -np.inf
        candidates = np.argpartition(scores, -k)[-k:]
        for candidate in candidates:
            score = float(scores[int(candidate)])
            if np.isfinite(score) and score >= float(min_affinity):
                support[idx, int(candidate)] = score
    support = np.maximum(support, support.T)
    np.fill_diagonal(support, 0.0)
    return support


def _temporal_chain_support_matrix(
    raw: np.ndarray,
    tokens: list[SemanticPartToken],
    *,
    topk: int,
    min_affinity: float,
    rank_window: int,
) -> np.ndarray:
    n = int(raw.shape[0])
    directed = np.zeros((n, n), dtype=np.float32)
    if n <= 1:
        return directed
    frame_ids = sorted({int(token.frame_id) for token in tokens})
    frame_rank = {frame_id: rank for rank, frame_id in enumerate(frame_ids)}
    k = max(1, min(int(topk), n - 1))
    for idx, token in enumerate(tokens):
        scores = np.asarray(raw[idx], dtype=np.float32).copy()
        scores[idx] = -np.inf
        token_rank = int(frame_rank[int(token.frame_id)])
        for other_idx, other in enumerate(tokens):
            other_rank = int(frame_rank[int(other.frame_id)])
            rank_delta = abs(token_rank - other_rank)
            if rank_delta == 0 or rank_delta > int(rank_window):
                scores[other_idx] = -np.inf
        candidates = np.argpartition(scores, -k)[-k:]
        for candidate in candidates:
            score = float(scores[int(candidate)])
            if np.isfinite(score) and score >= float(min_affinity):
                directed[idx, int(candidate)] = score
    mutual = np.minimum(directed, directed.T)
    np.fill_diagonal(mutual, 0.0)
    return mutual.astype(np.float32)


def _widest_path_matrix(support: np.ndarray) -> np.ndarray:
    structure = np.asarray(support, dtype=np.float32).copy()
    for pivot in range(structure.shape[0]):
        structure = np.maximum(structure, np.minimum(structure[:, pivot : pivot + 1], structure[pivot : pivot + 1, :]))
    np.fill_diagonal(structure, 0.0)
    return structure.astype(np.float32)


def auc_from_scores(labels: list[bool], scores: list[float]) -> float | None:
    pairs = [(bool(label), float(score)) for label, score in zip(labels, scores)]
    positives = [score for label, score in pairs if label]
    negatives = [score for label, score in pairs if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0.0
    for pos in positives:
        for neg in negatives:
            total += 1.0
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return float(wins / total) if total else None


def summarize_part_graph(tokens: list[SemanticPartToken], edges: list[PartGraphEdge]) -> dict[str, Any]:
    gt_labeled = [t for t in tokens if t.diagnostic_gt_instance is not None]
    mixed = [
        t
        for t in gt_labeled
        if t.diagnostic_gt_purity is not None and float(t.diagnostic_gt_purity) < 0.80
    ]
    labels = [bool(e.diagnostic_same_gt) for e in edges if e.diagnostic_same_gt is not None]
    semantic_scores = [float(e.semantic_affinity) for e in edges if e.diagnostic_same_gt is not None]
    object_scores = [float(e.object_affinity) for e in edges if e.diagnostic_same_gt is not None]
    same_frame_diff = [
        e
        for e in edges
        if e.same_frame_cannot_link and e.diagnostic_same_gt is not None and not bool(e.diagnostic_same_gt)
    ]
    false_merge_like = [e for e in same_frame_diff if float(e.object_affinity) >= 0.50]
    purity_vals = [float(t.diagnostic_gt_purity) for t in gt_labeled if t.diagnostic_gt_purity is not None]
    return {
        "part_token_count": int(len(tokens)),
        "edge_count": int(len(edges)),
        "gt_labeled_token_count": int(len(gt_labeled)),
        "mixed_part_rate": float(len(mixed) / max(len(gt_labeled), 1)),
        "part_purity_diagnostic_mean": float(np.mean(purity_vals)) if purity_vals else None,
        "semantic_affinity_AUC": auc_from_scores(labels, semantic_scores),
        "object_part_compatibility_AUC": auc_from_scores(labels, object_scores),
        "same_frame_same_class_false_merge_rate": float(len(false_merge_like) / max(len(same_frame_diff), 1)),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": bool(gt_labeled),
    }
