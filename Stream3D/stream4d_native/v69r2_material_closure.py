from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from stream4d_native.v68_local_graph_solver import _row_from_mapping, _same_frame_violation_count, _summarize_variant_all  # noqa: E402
from stream4d_native.v69r2_anchor_bank import _pair, _prep_candidate_rows  # noqa: E402
from tools.run_v65_scene_multiview_ap import _sha256, _write_csv, _write_json  # noqa: E402
from tools.run_v66_local_chunk_eval import _chunk_rows, _float_or_none, _frame_data, _load_csv_rows, _mean, _rel  # noqa: E402
from tools.run_v66_scene_mv_ap_probe5 import DEFAULT_SCENES, _discover_pipeline_root, _mask_dir_from_pipeline, _parse_csv_list  # noqa: E402


def _rooted(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _node_from_token(token: str) -> tuple[int, int] | None:
    parts = str(token or "").split(":")
    if len(parts) != 3:
        return None
    try:
        return (int(parts[1]), int(parts[2]))
    except ValueError:
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    parsed = _float_or_none(value)
    return float(default if parsed is None else parsed)


def _load_anchor_pairs(path: Path, variant: str, scenes: set[str]) -> dict[str, set[tuple[int, int]]]:
    out: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for row in _load_csv_rows(path):
        if str(row.get("anchor_variant")) != variant:
            continue
        scene = str(row.get("scene_id"))
        if scene not in scenes:
            continue
        out[str(row.get("chunk_id"))].add((int(float(row.get("frame_id") or 0)), int(float(row.get("mask_id") or 0))))
    return out


def _load_edges(path: Path, scenes: set[str]) -> dict[str, dict[tuple[int, int], list[dict[str, Any]]]]:
    out: dict[str, dict[tuple[int, int], list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in _load_csv_rows(path):
        scene = str(row.get("scene_id"))
        if scene not in scenes:
            continue
        left = _node_from_token(str(row.get("node_i") or ""))
        right = _node_from_token(str(row.get("node_j") or ""))
        if left is None or right is None:
            continue
        edge = {
            "left": left,
            "right": right,
            "same_frame": _parse_bool(row.get("same_frame")),
            "frame_delta": int(float(row.get("frame_delta") or 0)),
            "component_intersection_count": int(float(row.get("component_intersection_count") or 0)),
            "component_cosine": _safe_float(row.get("component_cosine")),
            "component_jaccard": _safe_float(row.get("component_jaccard")),
            "score_material_overlap": _safe_float(row.get("score_material_overlap")),
            "score_material_residual": _safe_float(row.get("score_material_residual")),
            "score_signature": _safe_float(row.get("score_signature")),
            "score_temporal_adjacent": _safe_float(row.get("score_temporal_adjacent")),
            "score_appearance_relative": _safe_float(row.get("score_appearance_relative")),
            "score_combined_frozen_appearance": _safe_float(row.get("score_combined_frozen_appearance")),
            "score_combined_shuffled_control": _safe_float(row.get("score_combined_shuffled_control")),
            "score_combined_no_temporal": _safe_float(row.get("score_combined_no_temporal")),
        }
        chunk_id = str(row.get("chunk_id"))
        out[chunk_id][left].append(edge)
        out[chunk_id][right].append(edge)
    return out


def _stable_unit_interval(*parts: Any) -> float:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def _component_temporal_score(frame_delta: int) -> float:
    delta = abs(int(frame_delta))
    if delta <= 0:
        return 0.0
    return float(1.0 / (1.0 + 0.18 * max(0, delta - 1)))


def _component_edges_for_chunk(
    *,
    chunk_id: str,
    anchors: set[tuple[int, int]],
    candidate_meta: dict[tuple[int, int], dict[str, Any]],
    max_component_fanout: int,
    max_candidates_per_anchor: int,
    min_shared_components: int,
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    inverted: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for node, meta in candidate_meta.items():
        for component in meta.get("_components", set()):
            inverted[str(component)].append(node)
    component_weight = {
        component: float(1.0 / np.log2(2.0 + len(nodes)))
        for component, nodes in inverted.items()
        if len(nodes) <= max_component_fanout
    }
    edges_by_anchor: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for anchor in sorted(anchors):
        anchor_meta = candidate_meta.get(anchor)
        if anchor_meta is None:
            continue
        anchor_components = set(anchor_meta.get("_components", set()))
        if not anchor_components:
            continue
        anchor_weight_total = sum(component_weight.get(str(component), 0.0) for component in anchor_components)
        if anchor_weight_total <= 0.0:
            continue
        candidate_shared: dict[tuple[int, int], set[str]] = defaultdict(set)
        for component in anchor_components:
            if str(component) not in component_weight:
                continue
            for other in inverted.get(str(component), []):
                if other != anchor:
                    candidate_shared[other].add(str(component))
        scored: list[dict[str, Any]] = []
        for other, shared in candidate_shared.items():
            if len(shared) < int(min_shared_components):
                continue
            other_meta = candidate_meta.get(other)
            if other_meta is None:
                continue
            frame_delta = abs(int(other[0]) - int(anchor[0]))
            if frame_delta == 0:
                continue
            other_components = set(other_meta.get("_components", set()))
            if not other_components:
                continue
            shared_weight = sum(component_weight.get(str(component), 0.0) for component in shared)
            other_weight_total = sum(component_weight.get(str(component), 0.0) for component in other_components)
            if shared_weight <= 0.0 or other_weight_total <= 0.0:
                continue
            union_count = len(anchor_components | other_components)
            component_anchor_recall = float(len(shared) / max(1, len(anchor_components)))
            component_candidate_precision = float(len(shared) / max(1, len(other_components)))
            component_jaccard = float(len(shared) / max(1, union_count))
            rarity_recall = float(shared_weight / max(1e-9, anchor_weight_total))
            rarity_precision = float(shared_weight / max(1e-9, other_weight_total))
            material_direct = float(0.42 * rarity_recall + 0.38 * rarity_precision + 0.20 * component_jaccard)
            signature = 1.0 if anchor_meta.get("_signature") and anchor_meta.get("_signature") == other_meta.get("_signature") else 0.0
            dino_mode_match = bool(
                anchor_meta.get("_DINO_feature_valid")
                and other_meta.get("_DINO_feature_valid")
                and anchor_meta.get("_DINO_mode_id")
                and anchor_meta.get("_DINO_mode_id") == other_meta.get("_DINO_mode_id")
            )
            dino_score = 0.86 if dino_mode_match else 0.0
            temporal_score = _component_temporal_score(frame_delta)
            shuffled = _stable_unit_interval("v69r2_component_shuffle", chunk_id, anchor, other)
            scored.append(
                {
                    "left": anchor,
                    "right": other,
                    "same_frame": False,
                    "frame_delta": frame_delta,
                    "component_intersection_count": int(len(shared)),
                    "component_cosine": material_direct,
                    "component_jaccard": component_jaccard,
                    "component_anchor_recall": component_anchor_recall,
                    "component_candidate_precision": component_candidate_precision,
                    "component_rarity_recall": rarity_recall,
                    "component_rarity_precision": rarity_precision,
                    "score_material_overlap": component_anchor_recall,
                    "score_material_residual": material_direct,
                    "score_signature": signature,
                    "score_temporal_adjacent": temporal_score,
                    "score_appearance_relative": dino_score,
                    "score_combined_frozen_appearance": dino_score,
                    "score_combined_shuffled_control": shuffled,
                    "score_combined_no_temporal": material_direct,
                    "candidate_source": "component_index",
                }
            )
        scored = sorted(
            scored,
            key=lambda edge: (
                float(edge["score_material_residual"]),
                int(edge["component_intersection_count"]),
                float(edge["component_candidate_precision"]),
                float(edge["score_temporal_adjacent"]),
            ),
            reverse=True,
        )[: int(max_candidates_per_anchor)]
        if scored:
            edges_by_anchor[anchor] = scored
    return edges_by_anchor


def _direct_component_scores(anchor_meta: dict[str, Any], other_meta: dict[str, Any]) -> dict[str, float]:
    anchor_components = set(anchor_meta.get("_components", set()))
    other_components = set(other_meta.get("_components", set()))
    shared = anchor_components & other_components
    if not anchor_components or not other_components or not shared:
        return {
            "shared_count": 0.0,
            "anchor_recall": 0.0,
            "candidate_precision": 0.0,
            "jaccard": 0.0,
            "material_direct": 0.0,
        }
    anchor_recall = float(len(shared) / max(1, len(anchor_components)))
    candidate_precision = float(len(shared) / max(1, len(other_components)))
    jaccard = float(len(shared) / max(1, len(anchor_components | other_components)))
    material_direct = float(0.42 * anchor_recall + 0.38 * candidate_precision + 0.20 * jaccard)
    return {
        "shared_count": float(len(shared)),
        "anchor_recall": anchor_recall,
        "candidate_precision": candidate_precision,
        "jaccard": jaccard,
        "material_direct": material_direct,
    }


def _tracklet_edges_for_chunk(
    *,
    chunk_id: str,
    anchors: set[tuple[int, int]],
    candidate_meta: dict[tuple[int, int], dict[str, Any]],
    max_candidates_per_anchor: int,
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    by_dino: dict[str, list[tuple[int, int]]] = defaultdict(list)
    by_signature: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for node, meta in candidate_meta.items():
        if meta.get("_DINO_mode_id"):
            by_dino[str(meta["_DINO_mode_id"])].append(node)
        if meta.get("_signature"):
            by_signature[str(meta["_signature"])].append(node)
    edges_by_anchor: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for anchor in sorted(anchors):
        anchor_meta = candidate_meta.get(anchor)
        if anchor_meta is None:
            continue
        candidates: set[tuple[int, int]] = set()
        if anchor_meta.get("_DINO_mode_id"):
            candidates.update(by_dino.get(str(anchor_meta["_DINO_mode_id"]), []))
        if anchor_meta.get("_signature"):
            candidates.update(by_signature.get(str(anchor_meta["_signature"]), []))
        scored: list[dict[str, Any]] = []
        for other in candidates:
            if other == anchor or other not in candidate_meta:
                continue
            frame_delta = abs(int(other[0]) - int(anchor[0]))
            if frame_delta == 0:
                continue
            other_meta = candidate_meta[other]
            dino_match = bool(anchor_meta.get("_DINO_mode_id") and anchor_meta.get("_DINO_mode_id") == other_meta.get("_DINO_mode_id"))
            signature_match = bool(anchor_meta.get("_signature") and anchor_meta.get("_signature") == other_meta.get("_signature"))
            scores = _direct_component_scores(anchor_meta, other_meta)
            dino_score = 0.86 if dino_match else 0.0
            signature = 1.0 if signature_match else 0.0
            temporal_score = _component_temporal_score(frame_delta)
            material_direct = float(scores["material_direct"])
            shuffled = _stable_unit_interval("v69r2_tracklet_shuffle", chunk_id, anchor, other)
            scored.append(
                {
                    "left": anchor,
                    "right": other,
                    "same_frame": False,
                    "frame_delta": frame_delta,
                    "component_intersection_count": int(scores["shared_count"]),
                    "component_cosine": material_direct,
                    "component_jaccard": float(scores["jaccard"]),
                    "component_anchor_recall": float(scores["anchor_recall"]),
                    "component_candidate_precision": float(scores["candidate_precision"]),
                    "component_rarity_recall": float(scores["anchor_recall"]),
                    "component_rarity_precision": float(scores["candidate_precision"]),
                    "score_material_overlap": float(scores["anchor_recall"]),
                    "score_material_residual": material_direct,
                    "score_signature": signature,
                    "score_temporal_adjacent": temporal_score,
                    "score_appearance_relative": dino_score,
                    "score_combined_frozen_appearance": dino_score,
                    "score_combined_shuffled_control": shuffled,
                    "score_combined_no_temporal": 0.74 * dino_score + 0.18 * signature + 0.08 * material_direct,
                    "candidate_source": "tracklet_index",
                }
            )
        scored = sorted(
            scored,
            key=lambda edge: (
                float(edge["score_appearance_relative"]),
                float(edge["score_signature"]),
                float(edge["score_material_residual"]),
                float(edge["score_temporal_adjacent"]),
            ),
            reverse=True,
        )[: int(max_candidates_per_anchor)]
        if scored:
            edges_by_anchor[anchor] = scored
    return edges_by_anchor


def _variant_score(variant: str, edge: dict[str, Any], candidate_underseg: bool) -> tuple[float, str]:
    material = float(edge["score_material_residual"])
    inside = float(edge["score_material_overlap"])
    dino = float(edge["score_appearance_relative"])
    signature = float(edge["score_signature"])
    temporal = float(edge["score_temporal_adjacent"])
    shuffled = float(edge["score_combined_shuffled_control"])
    no_temporal = float(edge["score_combined_no_temporal"])
    if variant == "C0_DINO_only":
        return dino, "core" if dino >= 0.72 else "reject"
    if variant == "C1_material_inside":
        return inside, "core" if inside >= 0.18 else "reject"
    if variant == "C2_material_residual":
        return material, "core" if material >= 0.12 else "reject"
    if variant == "C3_material_DINO":
        score = 0.55 * material + 0.45 * dino
        return score, "core" if score >= 0.36 and (material >= 0.05 or dino >= 0.78) else "reject"
    if variant == "C4_material_DINO_signature":
        score = 0.48 * material + 0.38 * dino + 0.14 * signature
        return score, "core" if score >= 0.35 and (material >= 0.05 or dino >= 0.76) else "reject"
    if variant == "C5_C4_underseg_shared":
        score = 0.48 * material + 0.38 * dino + 0.14 * signature
        if candidate_underseg:
            return score, "shared" if score >= 0.48 and material >= 0.10 else "reject"
        return score, "core" if score >= 0.35 and (material >= 0.05 or dino >= 0.76) else "reject"
    if variant == "C6_C5_temporal":
        score = 0.42 * material + 0.34 * dino + 0.12 * signature + 0.12 * temporal
        if candidate_underseg:
            return score, "shared" if score >= 0.47 and material >= 0.10 else "reject"
        return score, "core" if score >= 0.34 and (material >= 0.05 or dino >= 0.74) else "reject"
    if variant == "C7_shuffled_material_control":
        return shuffled, "core" if shuffled >= 0.72 else "reject"
    if variant == "C8_no_temporal_control":
        return no_temporal, "core" if no_temporal >= 0.12 else "reject"
    if variant == "C10_component_index_overlap":
        score = material
        if candidate_underseg:
            return score, "shared" if score >= 0.32 else "reject"
        return score, "core" if score >= 0.20 and int(edge.get("component_intersection_count") or 0) >= 1 else "reject"
    if variant == "C11_component_index_guarded":
        score = 0.74 * material + 0.18 * dino + 0.08 * signature
        if candidate_underseg:
            return score, "shared" if score >= 0.46 and material >= 0.24 else "reject"
        return score, "core" if score >= 0.24 and material >= 0.16 else "reject"
    if variant == "C12_component_index_temporal_guarded":
        score = 0.62 * material + 0.16 * dino + 0.08 * signature + 0.14 * temporal
        if candidate_underseg:
            return score, "shared" if score >= 0.44 and material >= 0.24 else "reject"
        return score, "core" if score >= 0.24 and material >= 0.15 else "reject"
    if variant == "C13_component_index_no_temporal_control":
        score = 0.72 * material + 0.20 * dino + 0.08 * signature
        if candidate_underseg:
            return score, "shared" if score >= 0.44 and material >= 0.24 else "reject"
        return score, "core" if score >= 0.24 and material >= 0.15 else "reject"
    if variant == "C14_component_index_shuffled_control":
        return shuffled, "core" if shuffled >= 0.985 else "reject"
    if variant == "C15_component_index_recall_relaxed":
        anchor_recall = float(edge.get("component_anchor_recall", inside))
        candidate_precision = float(edge.get("component_candidate_precision", material))
        score = 0.58 * anchor_recall + 0.30 * material + 0.12 * candidate_precision
        if candidate_underseg:
            return score, "shared" if score >= 0.42 and material >= 0.16 else "reject"
        return score, "core" if score >= 0.18 and material >= 0.08 else "reject"
    if variant == "C16_component_index_temporal_recall_relaxed":
        anchor_recall = float(edge.get("component_anchor_recall", inside))
        candidate_precision = float(edge.get("component_candidate_precision", material))
        score = 0.50 * anchor_recall + 0.24 * material + 0.10 * candidate_precision + 0.16 * temporal
        if candidate_underseg:
            return score, "shared" if score >= 0.42 and material >= 0.16 else "reject"
        return score, "core" if score >= 0.18 and material >= 0.08 else "reject"
    if variant == "C17_component_index_no_temporal_relaxed_control":
        anchor_recall = float(edge.get("component_anchor_recall", inside))
        candidate_precision = float(edge.get("component_candidate_precision", material))
        score = 0.56 * anchor_recall + 0.32 * material + 0.12 * candidate_precision
        if candidate_underseg:
            return score, "shared" if score >= 0.42 and material >= 0.16 else "reject"
        return score, "core" if score >= 0.18 and material >= 0.08 else "reject"
    if variant == "C18_tracklet_dino_signature":
        score = 0.62 * dino + 0.25 * signature + 0.13 * material
        if candidate_underseg:
            return score, "shared" if score >= 0.72 and material >= 0.08 else "reject"
        return score, "core" if score >= 0.62 and (signature > 0.0 or material >= 0.04) else "reject"
    if variant == "C19_tracklet_dino_d4rt_guard":
        score = 0.68 * dino + 0.14 * signature + 0.18 * material
        if candidate_underseg:
            return score, "shared" if score >= 0.72 and material >= 0.10 else "reject"
        return score, "core" if score >= 0.60 and (material >= 0.02 or signature > 0.0) else "reject"
    if variant == "C20_tracklet_temporal_guard":
        score = 0.58 * dino + 0.12 * signature + 0.16 * material + 0.14 * temporal
        if candidate_underseg:
            return score, "shared" if score >= 0.72 and material >= 0.10 else "reject"
        return score, "core" if score >= 0.60 and (material >= 0.02 or signature > 0.0) else "reject"
    if variant == "C21_tracklet_no_material_control":
        score = 0.76 * dino + 0.24 * signature
        if candidate_underseg:
            return score, "shared" if score >= 0.82 else "reject"
        return score, "core" if score >= 0.65 else "reject"
    if variant == "C22_tracklet_no_temporal_control":
        score = 0.70 * dino + 0.18 * signature + 0.12 * material
        if candidate_underseg:
            return score, "shared" if score >= 0.72 and material >= 0.10 else "reject"
        return score, "core" if score >= 0.60 and (material >= 0.02 or signature > 0.0) else "reject"
    raise ValueError(f"unknown closure variant: {variant}")


def _object_temporal_span(mapping: dict[tuple[int, int], int]) -> float | None:
    frames_by_object: dict[int, set[int]] = defaultdict(set)
    for (frame_id, _mask_id), object_id in mapping.items():
        frames_by_object[int(object_id)].add(int(frame_id))
    spans = [len(frames) for frames in frames_by_object.values()]
    return float(np.mean(spans)) if spans else None


def _build_mapping_for_variant(
    *,
    variant: str,
    anchors: set[tuple[int, int]],
    candidate_meta: dict[tuple[int, int], dict[str, Any]],
    edges_by_anchor: dict[tuple[int, int], list[dict[str, Any]]],
    max_supports_per_anchor: int,
    closure_rows: list[dict[str, Any]],
    scene: str,
    chunk_id: str,
) -> tuple[dict[tuple[int, int], int], dict[str, Any]]:
    mapping: dict[tuple[int, int], int] = {}
    used_masks: set[tuple[int, int]] = set()
    object_id = 0
    core_counts: list[int] = []
    support_counts: list[int] = []
    shared_counts: list[int] = []
    reject_counts: list[int] = []
    candidate_counts: list[int] = []
    underseg_bridge = 0
    visible_anchor_count = 0
    visible_material_counts: list[int] = []
    for anchor in sorted(anchors):
        meta = candidate_meta.get(anchor)
        if meta is None:
            continue
        object_id += 1
        object_members: set[tuple[int, int]] = {anchor}
        used_frames: set[int] = {int(anchor[0])}
        component_count = int(meta["_component_count"])
        visible_material_counts.append(component_count)
        if component_count > 0:
            visible_anchor_count += 1
        candidates: list[tuple[float, str, tuple[int, int], dict[str, Any]]] = []
        for edge in edges_by_anchor.get(anchor, []):
            other = edge["right"] if edge["left"] == anchor else edge["left"]
            if other == anchor or other not in candidate_meta or int(other[0]) in used_frames:
                continue
            other_meta = candidate_meta[other]
            score, role = _variant_score(variant, edge, bool(other_meta["_underseg"]))
            candidates.append((score, role, other, edge))
        candidate_counts.append(len(candidates))
        selected = 0
        reject_count = 0
        shared_count = 0
        support_count = 0
        core_count = 1
        for score, role, other, edge in sorted(candidates, reverse=True, key=lambda item: item[0]):
            if selected >= max_supports_per_anchor:
                break
            if role == "reject":
                reject_count += 1
                continue
            if other in used_masks or int(other[0]) in used_frames:
                reject_count += 1
                continue
            if role == "shared":
                shared_count += 1
                underseg_bridge += 1
            else:
                object_members.add(other)
                used_frames.add(int(other[0]))
                core_count += 1
                support_count += 1
            selected += 1
            if len(closure_rows) < 1_000_000:
                closure_rows.append(
                    {
                        "scene_id": scene,
                        "chunk_id": chunk_id,
                        "closure_variant": variant,
                        "anchor_frame": anchor[0],
                        "anchor_mask": anchor[1],
                        "candidate_frame": other[0],
                        "candidate_mask": other[1],
                        "role": role,
                        "closure_score": score,
                        "material_inside_proxy": edge["score_material_overlap"],
                        "material_residual_proxy": edge["score_material_residual"],
                        "dino_score": edge["score_appearance_relative"],
                        "signature_score": edge["score_signature"],
                        "temporal_score": edge["score_temporal_adjacent"],
                        "candidate_method_underseg": bool(candidate_meta[other]["_underseg"]),
                        "uses_gt_for_prediction": False,
                        "diagnostic_only": True,
                    }
                )
        for member in object_members:
            if member not in used_masks:
                mapping[member] = object_id
                used_masks.add(member)
        core_counts.append(core_count)
        support_counts.append(support_count)
        shared_counts.append(shared_count)
        reject_counts.append(reject_count)
    diag = {
        "core_mask_count_mean": _mean([float(v) for v in core_counts]),
        "support_mask_count_mean": _mean([float(v) for v in support_counts]),
        "shared_mask_count_mean": _mean([float(v) for v in shared_counts]),
        "reject_mask_count_mean": _mean([float(v) for v in reject_counts]),
        "candidate_masks_per_anchor": _mean([float(v) for v in candidate_counts]),
        "anchor_with_visible_material_rate": float(visible_anchor_count / max(1, len(visible_material_counts))),
        "mean_visible_material_per_anchor": _mean([float(v) for v in visible_material_counts]),
        "underseg_bridge_rate": float(underseg_bridge / max(1, sum(len(v) for v in edges_by_anchor.values()))),
        "same_frame_cannot_link_violation_count": _same_frame_violation_count(mapping),
        "shared_mask_count": int(sum(shared_counts)),
        "reject_mask_count": int(sum(reject_counts)),
        "unknown_mask_count": 0,
    }
    return mapping, diag


def _summarize_closure_variant(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    base = _summarize_variant_all(rows, variant)
    subset = [row for row in rows if row["variant"] == variant]
    base.update(
        {
            "closure_variant": variant,
            "anchor_count": int(sum(int(float(row.get("anchor_count") or 0)) for row in subset)),
            "anchor_with_visible_material_rate": _mean([_float_or_none(row.get("anchor_with_visible_material_rate")) for row in subset]),
            "mean_visible_material_per_anchor": _mean([_float_or_none(row.get("mean_visible_material_per_anchor")) for row in subset]),
            "candidate_masks_per_anchor": _mean([_float_or_none(row.get("candidate_masks_per_anchor")) for row in subset]),
            "core_mask_count_mean": _mean([_float_or_none(row.get("core_mask_count_mean")) for row in subset]),
            "support_mask_count_mean": _mean([_float_or_none(row.get("support_mask_count_mean")) for row in subset]),
            "shared_mask_count_mean": _mean([_float_or_none(row.get("shared_mask_count_mean")) for row in subset]),
            "reject_mask_count_mean": _mean([_float_or_none(row.get("reject_mask_count_mean")) for row in subset]),
            "single_anchor_SF50": base.get("local_score_free_match50_recall_mean"),
            "single_anchor_AP50": base.get("local_AP50_mean"),
            "single_anchor_GT_best_IoU_mean": base.get("local_GT_best_IoU_mean_mean"),
            "single_anchor_temporal_span_mean": _mean([_float_or_none(row.get("single_anchor_temporal_span_mean")) for row in subset]),
            "single_anchor_single_frame_rate": base.get("single_frame_object_rate_mean"),
            "underseg_bridge_rate": _mean([_float_or_none(row.get("underseg_bridge_rate")) for row in subset]),
        }
    )
    return base


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _rooted(args.output_root)
    visual_root = _rooted(args.visual_root)
    output_root.mkdir(parents=True, exist_ok=True)
    visual_root.mkdir(parents=True, exist_ok=True)
    scenes = _parse_csv_list(args.scenes)
    scene_set = set(scenes)
    candidate_path = _rooted(args.candidate_rows)
    edge_path = _rooted(args.edge_rows)
    anchor_path = _rooted(args.anchor_rows)
    candidates = _prep_candidate_rows(candidate_path, scenes, edge_path)
    candidate_by_chunk: dict[str, dict[tuple[int, int], dict[str, Any]]] = defaultdict(dict)
    for row in candidates:
        candidate_by_chunk[str(row["_chunk_id"])][_pair(row)] = row
    anchors_by_chunk = _load_anchor_pairs(anchor_path, str(args.anchor_variant), scene_set)
    edges_by_chunk = _load_edges(edge_path, scene_set)
    variants = [item.strip() for item in str(args.variants).split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    closure_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    pipeline_roots: dict[str, str] = {}
    for scene in scenes:
        print(f"[v69r2-material-closure] scene={scene}", file=sys.stderr, flush=True)
        pipeline_root = _discover_pipeline_root(scene)
        if pipeline_root is None:
            missing_rows.append({"scene_id": scene, "missing": "soma_fullscene_pipeline_root"})
            continue
        pipeline_roots[scene] = _rel(pipeline_root)
        mask_dir = _mask_dir_from_pipeline(pipeline_root)
        stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
        all_stride_frames = stream.frame_ids(stride=int(args.stride), max_frames=None)
        for chunk in _chunk_rows(pipeline_root, scene):
            t0 = time.time()
            chunk_id = str(chunk.get("chunk_id"))
            anchors = anchors_by_chunk.get(chunk_id, set())
            candidate_meta = candidate_by_chunk.get(chunk_id, {})
            edge_proxy_edges = edges_by_chunk.get(chunk_id, {})
            component_edges = {}
            if str(args.candidate_source) in {"component_index", "hybrid"}:
                component_edges = _component_edges_for_chunk(
                    chunk_id=chunk_id,
                    anchors=anchors,
                    candidate_meta=candidate_meta,
                    max_component_fanout=int(args.component_max_fanout),
                    max_candidates_per_anchor=int(args.component_max_candidates_per_anchor),
                    min_shared_components=int(args.component_min_shared_components),
                )
            if str(args.candidate_source) == "tracklet_index":
                edges = _tracklet_edges_for_chunk(
                    chunk_id=chunk_id,
                    anchors=anchors,
                    candidate_meta=candidate_meta,
                    max_candidates_per_anchor=int(args.component_max_candidates_per_anchor),
                )
            elif str(args.candidate_source) == "component_index":
                edges = component_edges
            elif str(args.candidate_source) == "hybrid":
                edges = defaultdict(list)
                for anchor, items in edge_proxy_edges.items():
                    edges[anchor].extend(items)
                for anchor, items in component_edges.items():
                    edges[anchor].extend(items)
            else:
                edges = edge_proxy_edges
            raw_start = int(float(chunk.get("raw_frame_start") or 0))
            raw_end = int(float(chunk.get("raw_frame_end") or raw_start))
            frame_ids = [int(frame) for frame in all_stride_frames if raw_start <= int(frame) <= raw_end]
            if not frame_ids or not anchors:
                continue
            frame_data = _frame_data(scene, frame_ids, mask_dir)
            for variant in variants:
                mapping, diag = _build_mapping_for_variant(
                    variant=variant,
                    anchors=anchors,
                    candidate_meta=candidate_meta,
                    edges_by_anchor=edges,
                    max_supports_per_anchor=int(args.max_supports_per_anchor),
                    closure_rows=closure_rows,
                    scene=scene,
                    chunk_id=chunk_id,
                )
                diag["anchor_count"] = int(len(anchors))
                row = _row_from_mapping(
                    scene=scene,
                    chunk_id=chunk_id,
                    variant=variant,
                    frame_ids=frame_ids,
                    chunk=chunk,
                    frame_data=frame_data,
                    mapping=mapping,
                    diag=diag,
                    pipeline_root=pipeline_root,
                    uses_gt_for_prediction=False,
                    forbidden_for_method_table=False,
                )
                row.update(diag)
                row["single_anchor_temporal_span_mean"] = _object_temporal_span(mapping)
                row["runtime_sec"] = float(time.time() - t0)
                row["anchor_variant"] = str(args.anchor_variant)
                row["candidate_source"] = str(args.candidate_source)
                row["component_max_fanout"] = int(args.component_max_fanout)
                row["component_max_candidates_per_anchor"] = int(args.component_max_candidates_per_anchor)
                row["component_min_shared_components"] = int(args.component_min_shared_components)
                rows.append(row)
    metric_rows = [_summarize_closure_variant(rows, variant) for variant in variants]
    control_variants = {
        "C7_shuffled_material_control",
        "C8_no_temporal_control",
        "C13_component_index_no_temporal_control",
        "C14_component_index_shuffled_control",
        "C17_component_index_no_temporal_relaxed_control",
        "C21_tracklet_no_material_control",
        "C22_tracklet_no_temporal_control",
    }
    non_control = [row for row in metric_rows if row.get("closure_variant") not in control_variants]
    best = max(non_control, key=lambda row: float(row.get("single_anchor_SF50") or 0.0), default={})
    by_variant = {str(row.get("closure_variant")): row for row in metric_rows}
    shuffled = by_variant.get("C14_component_index_shuffled_control") or by_variant.get("C7_shuffled_material_control", {})
    no_temporal = (
        by_variant.get("C17_component_index_no_temporal_relaxed_control")
        or by_variant.get("C22_tracklet_no_temporal_control")
        or by_variant.get("C13_component_index_no_temporal_control")
        or by_variant.get("C8_no_temporal_control", {})
    )
    best_sf50 = _float_or_none(best.get("single_anchor_SF50"))
    best_gt = _float_or_none(best.get("single_anchor_GT_best_IoU_mean"))
    best_span = _float_or_none(best.get("single_anchor_temporal_span_mean"))
    best_single = _float_or_none(best.get("single_anchor_single_frame_rate"))
    best_violation = int(float(best.get("same_frame_cannot_link_violation_count_sum") or 0)) if best else 0
    shuffled_sf50 = _float_or_none(shuffled.get("single_anchor_SF50"))
    no_temporal_sf50 = _float_or_none(no_temporal.get("single_anchor_SF50"))
    v68_sf50 = float(args.v68_best_solver_sf50)
    v68_gt = float(args.v68_best_solver_gt_best_iou)
    v68_single = float(args.v68_best_solver_single_frame_rate)
    real_minus_shuffled = None if best_sf50 is None or shuffled_sf50 is None else float(best_sf50 - shuffled_sf50)
    real_minus_no_temporal = None if best_sf50 is None or no_temporal_sf50 is None else float(best_sf50 - no_temporal_sf50)
    gate = {
        "all_pipeline_roots_available": len(pipeline_roots) == len(scenes),
        "anchor_with_visible_material_rate_ge_0p50": _float_or_none(best.get("anchor_with_visible_material_rate")) is not None and float(best.get("anchor_with_visible_material_rate")) >= 0.50,
        "single_anchor_SF50_ge_v68_plus_0p10": best_sf50 is not None and best_sf50 >= v68_sf50 + 0.10,
        "single_anchor_GT_best_IoU_ge_v68_plus_0p05": best_gt is not None and best_gt >= v68_gt + 0.05,
        "single_anchor_temporal_span_mean_ge_2p0": best_span is not None and best_span >= 2.0,
        "single_anchor_single_frame_rate_le_v68_minus_0p15": best_single is not None and best_single <= v68_single - 0.15,
        "same_frame_violation_count_eq_0": best_violation == 0,
        "real_minus_shuffled_SF50_ge_0p05": real_minus_shuffled is not None and real_minus_shuffled >= 0.05,
        "real_minus_no_temporal_SF50_ge_0p03": real_minus_no_temporal is not None and real_minus_no_temporal >= 0.03,
    }
    gate["pass"] = all(bool(v) for v in gate.values())
    decision = "PASS_MATERIAL_CLOSURE" if gate["pass"] else "NO_GO_MATERIAL_CLOSURE"
    _write_csv(output_root / "closure_rows.csv", closure_rows)
    _write_csv(output_root / "closure_metric_rows.csv", metric_rows)
    _write_csv(output_root / "closure_chunk_rows.csv", rows)
    _write_csv(output_root / "missing_input_rows.csv", missing_rows)
    if str(args.candidate_source) == "component_index":
        candidate_source_note = "candidate_source=component_index builds anchor-centric candidates from candidate D4RT component IDs instead of reusing sparse v68 edge rows."
    elif str(args.candidate_source) == "tracklet_index":
        candidate_source_note = "candidate_source=tracklet_index builds anchor-centric candidates from DINO-mode and repeated-signature tracklets, with D4RT component overlap used as a guard/score."
    elif str(args.candidate_source) == "hybrid":
        candidate_source_note = "candidate_source=hybrid combines sparse v68 edge rows with component-index D4RT candidates."
    else:
        candidate_source_note = "candidate_source=edge_proxy reuses sparse v68 edge rows as material-closure candidates."
    summary = {
        "phase": "v69r2_material_closure",
        "decision": decision,
        "gate": gate,
        "best_closure_variant": best,
        "real_minus_shuffled_SF50": real_minus_shuffled,
        "real_minus_no_temporal_SF50": real_minus_no_temporal,
        "anchor_variant": str(args.anchor_variant),
        "anchor_rows": _rel(anchor_path),
        "candidate_rows": _rel(candidate_path),
        "edge_rows": _rel(edge_path),
        "candidate_source": str(args.candidate_source),
        "component_max_fanout": int(args.component_max_fanout),
        "component_max_candidates_per_anchor": int(args.component_max_candidates_per_anchor),
        "component_min_shared_components": int(args.component_min_shared_components),
        "scenes": scenes,
        "pipeline_roots": pipeline_roots,
        "rows": {
            "closure_rows_csv": _rel(output_root / "closure_rows.csv"),
            "closure_metric_rows_csv": _rel(output_root / "closure_metric_rows.csv"),
            "closure_chunk_rows_csv": _rel(output_root / "closure_chunk_rows.csv"),
            "missing_input_rows_csv": _rel(output_root / "missing_input_rows.csv"),
        },
        "notes": [
            "Material closure uses v68 edge-row D4RT component overlap/residual as a material proxy; no GT labels are used for prediction.",
            "Same-frame conflicts are hard rejected by construction.",
            "C7 and C8 are controls and are not selected as best method variants.",
            candidate_source_note,
        ],
    }
    _write_json(output_root / "closure_summary.json", summary)
    sha_rows = []
    for path in [
        output_root / "closure_summary.json",
        output_root / "closure_rows.csv",
        output_root / "closure_metric_rows.csv",
        output_root / "closure_chunk_rows.csv",
        output_root / "missing_input_rows.csv",
    ]:
        if path.exists():
            sha_rows.append({"path": _rel(path), "sha256": _sha256(path), "bytes": int(path.stat().st_size)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v69-r2 Phase 2: anchor-centric material closure.")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--candidate-rows", default="outputs/audit/v68_candidate_bank/candidate_mask_rows.csv")
    parser.add_argument("--edge-rows", default="outputs/audit/v68_edge_audit_dinov2/edge_rows.csv")
    parser.add_argument("--anchor-rows", default="outputs/audit/v69r2_anchor_bank_repair5_nogt_underseg/anchor_rows.csv")
    parser.add_argument("--anchor-variant", default="A9_clean_recall_support_floor_u15")
    parser.add_argument("--output-root", default="outputs/audit/v69r2_material_closure")
    parser.add_argument("--visual-root", default="outputs/audit/v69r2_visualizations/material_closure")
    parser.add_argument("--variants", default="C0_DINO_only,C1_material_inside,C2_material_residual,C3_material_DINO,C4_material_DINO_signature,C5_C4_underseg_shared,C6_C5_temporal,C7_shuffled_material_control,C8_no_temporal_control")
    parser.add_argument("--max-supports-per-anchor", type=int, default=8)
    parser.add_argument("--candidate-source", choices=["edge_proxy", "component_index", "tracklet_index", "hybrid"], default="edge_proxy")
    parser.add_argument("--component-max-fanout", type=int, default=128)
    parser.add_argument("--component-max-candidates-per-anchor", type=int, default=64)
    parser.add_argument("--component-min-shared-components", type=int, default=1)
    parser.add_argument("--v68-best-solver-sf50", type=float, default=0.030280195610762346)
    parser.add_argument("--v68-best-solver-gt-best-iou", type=float, default=0.18437335018519382)
    parser.add_argument("--v68-best-solver-single-frame-rate", type=float, default=0.5815968043093369)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
