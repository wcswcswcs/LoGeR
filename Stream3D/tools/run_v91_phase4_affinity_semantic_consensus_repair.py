from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d_native.v91_mask_feature_store import load_mask_feature_store  # noqa: E402
from tools import run_v90_dev_extent_score_cross_audit as phase7d  # noqa: E402
from tools import run_v90_geo_semantic_witness_cover as phase4  # noqa: E402
from tools import run_v91_phase4_adaptive_uncertainty_materialization as adaptive  # noqa: E402
from tools import run_v91_phase4_ap50_control_repair as v91repair  # noqa: E402
from tools import run_v91_phase4_radius_sweep as radius_sweep  # noqa: E402
from tools import run_v91_phase4_scene_risk_materialization as scene_risk  # noqa: E402


OUT = ROOT / "outputs/audit/v91_phase4_affinity_semantic_consensus_repair"
SUPPORT_ROWS = ROOT / "outputs/audit/v90_phase3_v82_full_support/native_carrier_support_rows.csv"
V80_PHASE1 = ROOT / "outputs/audit/v80_phase1_streaming_affinity_features_dev_r79_semguard013125_parentmin30_incl090_signed030_ownerhard"
V80_PHASE2 = ROOT / "outputs/audit/v80_phase2_signed_affinity_dev_r79_semguard013125_parentmin30_incl090_signed030_ownerhard"
V80_PHASE4 = ROOT / "outputs/audit/v80_phase4_scale_clustering_dev_r79_semguard013125_parentmin30_incl090_signed030_ownerhard"
V80_CLUSTER_ROWS = V80_PHASE4 / "carrier_cluster_rows.csv"
V80_OWNERSHIP_ROWS = V80_PHASE4 / "object_mask_ownership_rows.csv"
SEMANTIC_VECTOR_ROWS = ROOT / "outputs/audit/v81_dino_feature_json_scene0011_scene0050/mask_feature_rows.csv"
RADIO_FEATURE_STORE = ROOT / "outputs/audit/v91_radio_mask_features_npz"
REFERENCE_PHASE8 = ROOT / "outputs/audit/v91_phase8_dev_selection/summary.json"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(adaptive._jsonable(row))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(adaptive._jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _variant_specs(variant_prefix: str = "V91_AF", semantic_tag: str = "semvec", include_hard_gates: bool = False) -> list[dict[str, Any]]:
    base = {
        "high_risk_max_masks": 4,
        "high_risk_extra_score_delta": 0.55,
        "high_risk_allow_broad_extra": True,
        "low_risk_max_masks": 2,
        "low_risk_extra_score_delta": 0.30,
        "low_risk_allow_broad_extra": False,
        "broad_rate_threshold": 0.65,
        "drop_per_selected_threshold": 1.0,
        "radius": 16,
        "support_point_radius": 3,
        "semantic_split": False,
        "strict_allowed": False,
    }
    specs = [
        {**base, "variant_id": f"{variant_prefix}1_v80_owner_{semantic_tag}_top1_r16", "score_mode": "v80_owner_semvec", "high_risk_max_masks": 1, "low_risk_max_masks": 1},
        {**base, "variant_id": f"{variant_prefix}2_v80_owner_{semantic_tag}_top2_r16", "score_mode": "v80_owner_semvec", "high_risk_max_masks": 2, "low_risk_max_masks": 2},
        {**base, "variant_id": f"{variant_prefix}3_v80_owner_{semantic_tag}_broadsafe_r24", "score_mode": "v80_owner_semvec_broadsafe", "high_risk_max_masks": 3, "low_risk_max_masks": 2, "radius": 24},
        {**base, "variant_id": f"{variant_prefix}4_v80_owner_{semantic_tag}_split_r16", "score_mode": "v80_owner_semantic_split", "high_risk_max_masks": 4, "low_risk_max_masks": 2, "semantic_split": True},
        {**base, "variant_id": f"{variant_prefix}5_v80_owner_{semantic_tag}_highprecision_r12_spr5", "score_mode": "v80_owner_highprecision", "high_risk_max_masks": 2, "low_risk_max_masks": 1, "radius": 12, "support_point_radius": 5, "strict_allowed": True},
    ]
    if include_hard_gates:
        specs.extend(
            [
                {**base, "variant_id": f"{variant_prefix}6_{semantic_tag}_cos90_top1_r16", "score_mode": "radio_cosine_gate", "high_risk_max_masks": 1, "low_risk_max_masks": 1, "min_semantic_cosine": 0.90, "low_risk_allow_broad_extra": False, "high_risk_allow_broad_extra": False},
                {**base, "variant_id": f"{variant_prefix}7_{semantic_tag}_cos85_top2_r24", "score_mode": "radio_cosine_gate", "high_risk_max_masks": 2, "low_risk_max_masks": 1, "radius": 24, "min_semantic_cosine": 0.85, "low_risk_allow_broad_extra": False, "high_risk_allow_broad_extra": False},
                {**base, "variant_id": f"{variant_prefix}8_{semantic_tag}_cos80_owner_r16_spr5", "score_mode": "radio_cosine_owner_gate", "high_risk_max_masks": 2, "low_risk_max_masks": 1, "support_point_radius": 5, "min_semantic_cosine": 0.80, "strict_allowed": True},
            ]
        )
    return specs


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _resolve_path(text: str) -> Path:
    path = Path(str(text))
    return path if path.is_absolute() else ROOT / path


def _resolve_csv_paths(text: str) -> list[Path]:
    return [_resolve_path(part.strip()) for part in str(text).split(",") if part.strip()]


def _slot_chunk_cluster(slot: str) -> tuple[int, int]:
    chunk_id = -1
    cluster_id = -1
    for part in str(slot).split(":"):
        if part.startswith("c") and part[1:].isdigit():
            chunk_id = int(part[1:])
        if part.startswith("cluster") and part[len("cluster") :].isdigit():
            cluster_id = int(part[len("cluster") :])
    return chunk_id, cluster_id


def _safe_token(value: str) -> str:
    out = []
    for ch in str(value):
        out.append(ch if ch.isalnum() else "_")
    return "".join(out).strip("_") or "missing"


def _load_v80_context() -> tuple[dict[tuple[str, int, int], dict[str, Any]], dict[tuple[str, int, int, int, int], dict[str, Any]]]:
    cluster: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in _read_csv(V80_CLUSTER_ROWS):
        if str(row.get("scale", "")) != "object":
            continue
        scene = str(row.get("scene_id", ""))
        chunk_id = adaptive._int(row.get("chunk_id"), -1)
        cluster_id = adaptive._int(row.get("cluster_id"), -1)
        if not scene or chunk_id < 0 or cluster_id < 0:
            continue
        cluster[(scene, chunk_id, cluster_id)] = dict(row)

    ownership: dict[tuple[str, int, int, int, int], dict[str, Any]] = {}
    for row in _read_csv(V80_OWNERSHIP_ROWS):
        if adaptive._bool(row.get("uses_gt_for_prediction")):
            continue
        scene = str(row.get("scene_id", ""))
        chunk_id = adaptive._int(row.get("chunk_id"), -1)
        cluster_id = adaptive._int(row.get("cluster_id"), -1)
        frame_id = adaptive._int(row.get("frame_id"), -1)
        mask_id = adaptive._int(row.get("mask_id"), -1)
        if not scene or chunk_id < 0 or cluster_id < 0 or frame_id < 0 or mask_id <= 0:
            continue
        ownership[(scene, chunk_id, cluster_id, frame_id, mask_id)] = dict(row)
    return cluster, ownership


def _load_semantic_vectors(csv_path: Path, feature_store_path: Path | None = None) -> dict[tuple[str, int, int], np.ndarray]:
    if feature_store_path is not None:
        store = load_mask_feature_store(feature_store_path)
        return store.as_keyed_dict()
    vectors: dict[tuple[str, int, int], np.ndarray] = {}
    if not csv_path.exists():
        return vectors
    for row in _read_csv(csv_path):
        if adaptive._bool(row.get("uses_gt_for_prediction")) or not adaptive._bool(row.get("feature_available", "True")):
            continue
        scene = str(row.get("scene_id", ""))
        frame_id = adaptive._int(row.get("frame_id"), -1)
        mask_id = adaptive._int(row.get("mask_id"), -1)
        if not scene or frame_id < 0 or mask_id <= 0:
            continue
        try:
            vec = np.asarray(json.loads(str(row.get("feature_json", "[]"))), dtype=np.float32)
        except json.JSONDecodeError:
            continue
        if vec.size == 0 or not np.all(np.isfinite(vec)):
            continue
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-8:
            continue
        vectors[(scene, frame_id, mask_id)] = vec / norm
    return vectors


def _slot_semantic_vectors(candidates: list[dict[str, Any]], semantic_vectors: dict[tuple[str, int, int], np.ndarray]) -> dict[tuple[str, str], np.ndarray]:
    accum: dict[tuple[str, str], np.ndarray] = {}
    weights: dict[tuple[str, str], float] = defaultdict(float)
    fallback: dict[tuple[str, str], np.ndarray] = {}
    fallback_weights: dict[tuple[str, str], float] = defaultdict(float)
    for row in candidates:
        scene = str(row.get("scene_id", ""))
        slot = str(row.get("local_slot_id", ""))
        key = (scene, slot)
        vec = semantic_vectors.get((scene, adaptive._int(row.get("frame_id"), -1), adaptive._int(row.get("mask_id"), -1)))
        if vec is None:
            continue
        margin = max(0.0, adaptive._num(row.get("semantic_prototype_margin"), 0.0))
        weight = math.log1p(max(1.0, adaptive._num(row.get("support_count"), 1.0))) * (0.25 + margin)
        if adaptive._bool(row.get("broad_background_risk")):
            weight *= 0.25
        fallback[key] = fallback.get(key, np.zeros_like(vec)) + vec * max(1e-4, weight)
        fallback_weights[key] += max(1e-4, weight)
        if adaptive._bool(row.get("broad_background_risk")):
            continue
        accum[key] = accum.get(key, np.zeros_like(vec)) + vec * max(1e-4, weight)
        weights[key] += max(1e-4, weight)
    out: dict[tuple[str, str], np.ndarray] = {}
    for key, vec in fallback.items():
        chosen = accum.get(key, vec)
        denom = weights.get(key, fallback_weights.get(key, 1.0))
        proto = chosen / max(1e-8, denom)
        norm = float(np.linalg.norm(proto))
        if norm > 1e-8:
            out[key] = proto / norm
    return out


def _candidate_affinity_features(
    row: dict[str, Any],
    points: list[dict[str, Any]],
    cluster_context: dict[tuple[str, int, int], dict[str, Any]],
    ownership_context: dict[tuple[str, int, int, int, int], dict[str, Any]],
    slot_proto: str,
    slot_vec: np.ndarray | None,
    semantic_vectors: dict[tuple[str, int, int], np.ndarray],
) -> dict[str, Any]:
    scene = str(row.get("scene_id", ""))
    slot = str(row.get("local_slot_id", ""))
    chunk_id, cluster_id = _slot_chunk_cluster(slot)
    frame_id = adaptive._int(row.get("frame_id"), -1)
    mask_id = adaptive._int(row.get("mask_id"), -1)
    support_count = adaptive._num(row.get("support_count"), len(points))
    carrier_count = max(1.0, adaptive._num(row.get("carrier_count_unique"), support_count))
    owner = ownership_context.get((scene, chunk_id, cluster_id, frame_id, mask_id), {})
    cluster = cluster_context.get((scene, chunk_id, cluster_id), {})
    ownership_precision = adaptive._num(owner.get("object_mask_ownership_precision"), 0.0)
    ownership_recall = adaptive._num(owner.get("object_mask_ownership_recall"), 0.0)
    ownership_f1 = adaptive._num(owner.get("object_mask_ownership_F1"), 0.0)
    ownership_allowed = adaptive._bool(owner.get("object_mask_ownership_allowed"))
    ownership_margin = adaptive._num(owner.get("object_mask_ownership_score_margin"), 0.0)
    ownership_rank = adaptive._num(owner.get("object_mask_ownership_rank"), 999.0)
    mean_internal = adaptive._num(cluster.get("mean_internal_affinity"), 0.0)
    mean_signed = adaptive._num(cluster.get("mean_signed_affinity"), 0.0)
    cluster_carriers = adaptive._num(cluster.get("carrier_count"), 0.0)
    cluster_span = adaptive._num(cluster.get("visible_frame_span"), 0.0)
    conf = adaptive._num(row.get("confidence_mean"), 1.0)
    vis = adaptive._num(row.get("visibility_mean"), 1.0)
    density = adaptive._num(row.get("observed_density_mean"), 0.0)
    internal_affinity = 0.55 * math.log1p(carrier_count) * conf * vis + 0.45 * math.log1p(max(0.0, cluster_carriers)) + 40.0 * density
    proto = str(row.get("semantic_prototype_id", ""))
    proto_match = 1.0 if slot_proto and proto == slot_proto else 0.0
    semantic_margin = adaptive._num(row.get("semantic_prototype_margin"), 0.0)
    semantic_entropy = adaptive._num(row.get("semantic_entropy"), 1.0)
    vec = semantic_vectors.get((scene, frame_id, mask_id))
    semantic_cosine = float(np.dot(vec, slot_vec)) if vec is not None and slot_vec is not None else 0.0
    semantic_score = 0.65 * proto_match + 1.40 * semantic_margin + 0.90 * semantic_cosine - 0.25 * semantic_entropy
    area = max(1.0, adaptive._num(row.get("mask_area"), 1.0))
    area_ratio = adaptive._num(row.get("area_ratio"), 0.0)
    broad = adaptive._bool(row.get("broad_background_risk"))
    support_density = support_count / math.sqrt(area)
    broad_leak_risk = (1.0 if broad else 0.0) + max(0.0, area_ratio - 0.25)
    return {
        "v80_chunk_id": int(chunk_id),
        "v80_cluster_id": int(cluster_id),
        "v80_ownership_available": bool(owner),
        "v80_ownership_allowed": bool(ownership_allowed),
        "v80_ownership_precision": float(ownership_precision),
        "v80_ownership_recall": float(ownership_recall),
        "v80_ownership_f1": float(ownership_f1),
        "v80_ownership_score_margin": float(ownership_margin),
        "v80_ownership_rank": float(ownership_rank),
        "v80_cluster_carrier_count": float(cluster_carriers),
        "v80_cluster_visible_frame_span": float(cluster_span),
        "v80_cluster_mean_internal_affinity": float(mean_internal),
        "v80_cluster_mean_signed_affinity": float(mean_signed),
        "internal_affinity": float(internal_affinity),
        "semantic_proto_match": float(proto_match),
        "semantic_vector_cosine_to_slot": float(semantic_cosine),
        "semantic_score": float(semantic_score),
        "support_density": float(support_density),
        "area_ratio": float(area_ratio),
        "broad_leak_risk": float(broad_leak_risk),
        "broad_background_risk": bool(broad),
    }


def _score_from_features(row: dict[str, Any], features: dict[str, Any], score_mode: str) -> float:
    owner_f1 = float(features["v80_ownership_f1"])
    owner_precision = float(features["v80_ownership_precision"])
    owner_recall = float(features["v80_ownership_recall"])
    owner_allowed = 1.0 if bool(features["v80_ownership_allowed"]) else 0.0
    owner_margin = float(features["v80_ownership_score_margin"])
    mean_signed = float(features["v80_cluster_mean_signed_affinity"])
    internal = float(features["internal_affinity"])
    semantic = float(features["semantic_score"])
    sem_cos = float(features["semantic_vector_cosine_to_slot"])
    support_density = float(features["support_density"])
    original = adaptive._num(row.get("selection_score"), 0.0)
    risk = float(features["broad_leak_risk"]) + 0.35 * max(0.0, 0.45 - sem_cos)
    if score_mode == "v80_owner_semvec":
        return 2.20 * owner_f1 + 0.80 * owner_precision + 0.45 * owner_recall + 0.70 * owner_allowed + 1.10 * semantic + 0.35 * support_density + 0.25 * math.log1p(internal) + 0.25 * mean_signed - 0.65 * risk
    if score_mode == "v80_owner_semvec_broadsafe":
        return 2.35 * owner_f1 + 1.05 * owner_precision + 0.35 * owner_recall + 0.85 * owner_allowed + 1.20 * semantic + 0.30 * support_density + 0.30 * math.log1p(internal) + 0.25 * mean_signed - 1.15 * risk
    if score_mode == "v80_owner_semantic_split":
        return 1.95 * owner_f1 + 0.85 * owner_precision + 0.55 * owner_recall + 0.65 * owner_allowed + 1.35 * semantic + 0.45 * support_density + 0.20 * math.log1p(internal) + 0.35 * mean_signed - 0.70 * risk
    if score_mode == "v80_owner_highprecision":
        return 1.55 * owner_f1 + 1.85 * owner_precision + 0.25 * owner_recall + 0.90 * owner_allowed + 0.80 * semantic + 0.40 * support_density + 0.25 * math.log1p(internal) + 0.40 * owner_margin - 1.25 * risk
    if score_mode == "radio_cosine_gate":
        return 4.50 * sem_cos + 0.65 * owner_f1 + 0.35 * owner_precision + 0.35 * support_density + 0.20 * math.log1p(internal) - 0.95 * risk
    if score_mode == "radio_cosine_owner_gate":
        return 3.50 * sem_cos + 1.20 * owner_f1 + 0.90 * owner_precision + 0.65 * owner_allowed + 0.30 * support_density + 0.20 * math.log1p(internal) - 0.85 * risk
    return original


def _is_high_risk_scene(scene_profile: dict[str, dict[str, float]], scene: str, spec: dict[str, Any]) -> bool:
    return scene_risk._is_high_risk_scene(scene_profile.get(scene, {}), spec)


def _select_affinity_semantic_rows(
    candidates: list[dict[str, Any]],
    support_points: dict[tuple[str, str, int, int], list[dict[str, Any]]],
    slot_to_obj: dict[tuple[str, str], str],
    slot_to_proto: dict[tuple[str, str], str],
    frame_to_window_index: dict[tuple[str, int], int],
    frame_to_window_id: dict[tuple[str, int], str],
    scene_profile: dict[str, dict[str, float]],
    spec: dict[str, Any],
    cluster_context: dict[tuple[str, int, int], dict[str, Any]],
    ownership_context: dict[tuple[str, int, int, int, int], dict[str, Any]],
    semantic_vectors: dict[tuple[str, int, int], np.ndarray],
    slot_vectors: dict[tuple[str, str], np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source_variant = f"{spec['variant_id']}_source"
    by_slot_frame: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_slot_frame[(str(row["scene_id"]), str(row["local_slot_id"]), adaptive._int(row.get("frame_id"), -1))].append(row)

    policy_rows_by_scene: dict[str, dict[str, Any]] = {}
    pre_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for (scene, slot, frame_id), items in sorted(by_slot_frame.items()):
        slot_key = (scene, slot)
        if slot_key not in slot_to_obj:
            continue
        high_risk = _is_high_risk_scene(scene_profile, scene, spec)
        max_masks = int(spec["high_risk_max_masks"] if high_risk else spec["low_risk_max_masks"])
        extra_delta = float(spec["high_risk_extra_score_delta"] if high_risk else spec["low_risk_extra_score_delta"])
        allow_broad = bool(spec["high_risk_allow_broad_extra"] if high_risk else spec["low_risk_allow_broad_extra"])
        policy_rows_by_scene[scene] = {
            "variant_id": spec["variant_id"],
            "scene_id": scene,
            "scene_policy_is_high_risk": high_risk,
            "scene_profile_selected_broad_risk_rate": scene_profile.get(scene, {}).get("selected_broad_risk_rate", 0.0),
            "scene_profile_source_drop_per_selected": scene_profile.get(scene, {}).get("source_drop_per_selected", 0.0),
            "score_mode": spec["score_mode"],
            "max_masks": max_masks,
            "extra_score_delta": extra_delta,
            "allow_broad_extra": allow_broad,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        scored: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for item in items:
            mask_id = adaptive._int(item.get("mask_id"), -1)
            points = support_points.get((scene, slot, frame_id, mask_id), [])
            features = _candidate_affinity_features(
                item,
                points,
                cluster_context,
                ownership_context,
                slot_to_proto.get(slot_key, ""),
                slot_vectors.get(slot_key),
                semantic_vectors,
            )
            min_semantic_cosine = spec.get("min_semantic_cosine")
            if min_semantic_cosine is not None and float(features.get("semantic_vector_cosine_to_slot", 0.0)) < float(min_semantic_cosine):
                continue
            if bool(spec.get("strict_allowed")) and not bool(features.get("v80_ownership_allowed")):
                continue
            score = _score_from_features(item, features, str(spec["score_mode"]))
            scored.append((score, item, features))
            feature_rows.append(
                {
                    "variant_id": spec["variant_id"],
                    "scene_id": scene,
                    "local_slot_id": slot,
                    "frame_id": int(frame_id),
                    "mask_id": int(mask_id),
                    **features,
                    "score_mode": spec["score_mode"],
                    "affinity_semantic_score": float(score),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
        scored.sort(key=lambda triple: triple[0], reverse=True)
        if not scored:
            continue
        selected = [scored[0]]
        top_score = float(scored[0][0])
        for score, item, features in scored[1:]:
            if len(selected) >= max_masks:
                break
            if not allow_broad and adaptive._bool(item.get("broad_background_risk")):
                continue
            if bool(spec.get("strict_allowed")) and float(features.get("v80_ownership_precision", 0.0)) < 0.45:
                continue
            if float(score) >= top_score - extra_delta:
                selected.append((score, item, features))
        for rank, (score, item, features) in enumerate(selected, start=1):
            object_suffix = ""
            if bool(spec.get("semantic_split")):
                proto = str(item.get("semantic_prototype_id", ""))
                if proto and proto != slot_to_proto.get(slot_key, "") and adaptive._num(item.get("semantic_prototype_margin"), 0.0) >= 0.04:
                    object_suffix = f":sem_{_safe_token(proto)}"
            row = {
                **item,
                **features,
                "variant_id": source_variant,
                "mv_object_id": f"{source_variant}:{slot_to_obj[slot_key]}{object_suffix}",
                "window_index": int(frame_to_window_index.get((scene, frame_id), -1)),
                "window_id": frame_to_window_id.get((scene, frame_id), ""),
                "selection_score": float(score),
                "selection_rank": int(rank),
                "selection_stage": "pre_conflict_affinity_semantic_consensus",
                "selection_reason": f"v91_affinity_semantic_consensus_{spec['score_mode']}",
                "scene_policy_is_high_risk": high_risk,
                "scene_profile_selected_broad_risk_rate": scene_profile.get(scene, {}).get("selected_broad_risk_rate", 0.0),
                "scene_profile_source_drop_per_selected": scene_profile.get(scene, {}).get("source_drop_per_selected", 0.0),
                "risk_penalty": 1.0 if adaptive._bool(item.get("broad_background_risk")) else 0.0,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
            pre_rows.append(row)

    kept: dict[tuple[str, int, int], dict[str, Any]] = {}
    dropped: list[dict[str, Any]] = []
    for row in sorted(
        pre_rows,
        key=lambda r: (
            adaptive._num(r.get("selection_score")),
            adaptive._num(r.get("v80_ownership_f1")),
            adaptive._num(r.get("semantic_score")),
        ),
        reverse=True,
    ):
        key = (str(row["scene_id"]), adaptive._int(row.get("frame_id"), -1), adaptive._int(row.get("mask_id"), -1))
        old = kept.get(key)
        if old is None:
            kept[key] = {
                **row,
                "selection_stage": "post_conflict_affinity_semantic_wta",
                "conflict_dropped": False,
                "ownership_wta_winner": True,
            }
        else:
            dropped.append(
                {
                    **row,
                    "selection_stage": "dropped_by_same_frame_mask_affinity_semantic_wta",
                    "conflict_dropped": True,
                    "kept_mv_object_id": old.get("mv_object_id", ""),
                    "kept_selection_score": old.get("selection_score", ""),
                    "kept_v80_ownership_f1": old.get("v80_ownership_f1", ""),
                    "kept_semantic_score": old.get("semantic_score", ""),
                }
            )
    final_rows = sorted(
        kept.values(),
        key=lambda r: (r["variant_id"], r["scene_id"], r["local_slot_id"], adaptive._int(r.get("frame_id")), -adaptive._num(r.get("selection_score"))),
    )
    return final_rows, dropped, list(policy_rows_by_scene.values()), feature_rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    global OUT
    started = time.time()
    OUT = _resolve_path(args.output_root)
    OUT.mkdir(parents=True, exist_ok=True)
    phase4.OUT = OUT
    radius_sweep.OUT = OUT
    mask_dirs = phase4._mask_dir_by_scene()
    frame_to_window_index, frame_to_window_id = phase4._window_maps()
    _source_rows, slot_to_obj, slot_to_proto, slot_to_area = phase4._load_source_rows()
    semantic_feature_rows = _resolve_csv_paths(args.semantic_feature_rows)
    if semantic_feature_rows:
        phase4.SEMANTIC_FEATURE_ROWS = semantic_feature_rows
    semantic_features = phase4._load_semantic_features()
    candidates, support_points = phase4._load_support_candidates(SUPPORT_ROWS, set(slot_to_obj), semantic_features, mask_dirs)
    slot_to_proto, slot_to_area = phase4._fill_slot_priors_from_candidates(candidates, slot_to_proto, slot_to_area)
    cluster_context, ownership_context = _load_v80_context()
    feature_store_path = _resolve_path(args.feature_store) if str(args.feature_store).strip() else None
    semantic_vector_csv = _resolve_path(args.semantic_vector_rows)
    semantic_vectors = _load_semantic_vectors(semantic_vector_csv, feature_store_path)
    slot_vectors = _slot_semantic_vectors(candidates, semantic_vectors)
    baselines = v91repair._phase8_baselines()
    phase8 = json.loads(REFERENCE_PHASE8.read_text(encoding="utf-8")) if REFERENCE_PHASE8.exists() else {}
    profile_rows, scene_profile = scene_risk._scene_profile_rows(
        candidates,
        slot_to_obj,
        slot_to_proto,
        slot_to_area,
        frame_to_window_index,
        frame_to_window_id,
    )

    config_rows: list[dict[str, Any]] = []
    scene_policy_rows: list[dict[str, Any]] = []
    source_selection_rows_all: list[dict[str, Any]] = []
    dropped_source_rows_all: list[dict[str, Any]] = []
    affinity_feature_rows_all: list[dict[str, Any]] = []
    generated_rows_all: list[dict[str, Any]] = []
    selected_rows_all: list[dict[str, Any]] = []
    eval_rows_all: list[dict[str, Any]] = []
    scored_rows_all: list[dict[str, Any]] = []
    support_quality_all: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []

    semantic_source_label = str(args.semantic_source_label)
    for spec in _variant_specs(str(args.variant_prefix), str(args.semantic_tag), bool(args.include_hard_gates)):
        variant_id = str(spec["variant_id"])
        source_variant = f"{variant_id}_source"
        source_rows, dropped_source_rows, policy_rows, affinity_feature_rows = _select_affinity_semantic_rows(
            candidates,
            support_points,
            slot_to_obj,
            slot_to_proto,
            frame_to_window_index,
            frame_to_window_id,
            scene_profile,
            spec,
            cluster_context,
            ownership_context,
            semantic_vectors,
            slot_vectors,
        )
        generated_rows, selected_rows, eval_rows = phase4._generate_carved_masks(
            source_rows,
            support_points,
            mask_dirs,
            radius=int(spec["radius"]),
            support_point_radius=int(spec["support_point_radius"]),
            variant=variant_id,
            source_variant=source_variant,
        )
        feature_map = radius_sweep._feature_map(selected_rows, generated_rows)
        scored_rows, keep_flags = v91repair._variant_rows(
            eval_rows,
            feature_map,
            variant_id=variant_id,
            risk_mode="drop_broad_low_h9_5",
            score_mode="broad_scene_orig_ge065",
            group_name="affinity_semantic_consensus_repair",
        )
        support_quality = v91repair._support_quality_rows(scored_rows, feature_map, variant_id)
        metrics, cases = radius_sweep._evaluate_variant(variant_id, scored_rows)

        scene_policy_rows.extend(policy_rows)
        source_selection_rows_all.extend(source_rows)
        dropped_source_rows_all.extend(dropped_source_rows)
        affinity_feature_rows_all.extend(affinity_feature_rows)
        generated_rows_all.extend(generated_rows)
        selected_rows_all.extend(selected_rows)
        eval_rows_all.extend(eval_rows)
        scored_rows_all.extend(scored_rows)
        support_quality_all.extend(support_quality)
        metric_rows.extend(metrics)
        case_rows.extend({**row, "variant_id": variant_id} for row in cases)

        broad_values = [1.0 if adaptive._bool(row.get("broad_risk")) else 0.0 for row in support_quality]
        risk_rows.append(
            {
                "variant_id": variant_id,
                "parent_variant_id": "V91_AD4_sr2_adapt_sig8_b05_j075_r12",
                "score_mode": spec["score_mode"],
                "radius": int(spec["radius"]),
                "support_point_radius": int(spec["support_point_radius"]),
                "source_selected_rows": len(source_rows),
                "source_conflict_dropped_rows": len(dropped_source_rows),
                "selected_rows": len(scored_rows),
                "pre_filter_eval_rows": len(eval_rows),
                "dropped_rows": len(eval_rows) - int(sum(1 for flag in keep_flags if flag)),
                "affinity_feature_rows": len(affinity_feature_rows),
                "v80_ownership_available_rate": adaptive._mean([1.0 if adaptive._bool(row.get("v80_ownership_available")) else 0.0 for row in affinity_feature_rows]),
                "v80_ownership_allowed_rate": adaptive._mean([1.0 if adaptive._bool(row.get("v80_ownership_allowed")) else 0.0 for row in affinity_feature_rows]),
                "v80_ownership_f1_mean": adaptive._mean([adaptive._num(row.get("v80_ownership_f1")) for row in affinity_feature_rows]),
                "semantic_vector_cosine_to_slot_mean": adaptive._mean([adaptive._num(row.get("semantic_vector_cosine_to_slot")) for row in affinity_feature_rows]),
                "semantic_proto_match_mean": adaptive._mean([adaptive._num(row.get("semantic_proto_match")) for row in affinity_feature_rows]),
                "risk_penalty_mean": adaptive._mean(broad_values),
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "uses_affinity_feature": True,
                "uses_semantic_feature": True,
            }
        )
        config_rows.append(
            {
                "variant_id": variant_id,
                "parent_variant_id": "V91_AD4_sr2_adapt_sig8_b05_j075_r12",
                "changed_module": "phase4_affinity_semantic_consensus_readout",
                "changed_parameters": (
                    f"score_mode={spec['score_mode']}; radius={int(spec['radius'])}; "
                    f"support_point_radius={int(spec['support_point_radius'])}; "
                    f"high_risk_max_masks={int(spec['high_risk_max_masks'])}; low_risk_max_masks={int(spec['low_risk_max_masks'])}; "
                    f"min_semantic_cosine={spec.get('min_semantic_cosine', '')}; "
                    "affinity_source=v80_r79_signed_scale_cluster_and_object_mask_ownership; "
                    f"semantic_source={semantic_source_label}; feature_store={adaptive._rel(feature_store_path) if feature_store_path else adaptive._rel(semantic_vector_csv)}"
                ),
                "reason_for_change": (
                    "User-corrected v91 direction: use the gate-passing v80 signed-scale affinity field, object mask ownership, "
                    f"and {semantic_source_label} semantic feature-vector consistency for source masklet readout instead of RGB/appearance boundary heuristics or the failed v79 top-mask feature."
                ),
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "uses_affinity_feature": True,
                "uses_semantic_feature": True,
                "dev_only_or_holdout": "dev_only",
                "expected_blocker": "SOURCE_BOUNDARY_READOUT_BLOCKER",
            }
        )

    aggregate_rows = phase7d._aggregate(metric_rows)
    control_rows = v91repair._add_gate_rows(aggregate_rows, baselines)
    best = adaptive._best_row(control_rows)
    passing = [row for row in control_rows if adaptive._bool(row.get("v91_phase8_progress_gate_pass"))]
    reference_mv_ap = adaptive._num(phase8.get("best_real_MV_AP_window"))
    reference_mv_ap50 = adaptive._num(phase8.get("best_real_MV_AP50_window"))
    best_delta_mv_ap = adaptive._num(best.get("mean_MV_AP_window")) - reference_mv_ap
    best_delta_mv_ap50 = adaptive._num(best.get("mean_MV_AP50_window")) - reference_mv_ap50

    variant_gate_rows: list[dict[str, Any]] = []
    variant_failure_rows: list[dict[str, Any]] = []
    for row in control_rows:
        cfg = next((item for item in config_rows if item.get("variant_id") == row.get("variant_id")), {})
        gate_row = {
            "variant_id": row.get("variant_id", ""),
            "parent_variant_id": "V91_AD4_sr2_adapt_sig8_b05_j075_r12",
            "changed_terms": f"v80 signed-scale carrier affinity ownership + {semantic_source_label} semantic vector readout",
            "changed_parameters": cfg.get("changed_parameters", ""),
            "reason_for_change": cfg.get("reason_for_change", ""),
            "uses_gt_for_prediction": row.get("uses_gt_for_prediction", "False"),
            "uses_future": row.get("uses_future", "False"),
            "metric_source": "v65 local-window evaluator via radius_sweep._evaluate_variant",
            "MV_AP_window": row.get("mean_MV_AP_window", ""),
            "MV_AP50_window": row.get("mean_MV_AP50_window", ""),
            "MV_AP25_window": row.get("mean_MV_AP25_window", ""),
            "score_free_Match50_window": row.get("mean_score_free_Match50_window", ""),
            "best_control_MV_AP_window": row.get("best_control_MV_AP_window", ""),
            "best_control_MV_AP50_window": row.get("best_control_MV_AP50_window", ""),
            "real_minus_best_control_MV_AP_window": row.get("real_minus_best_control_MV_AP_window", ""),
            "real_minus_best_control_MV_AP50_window": row.get("real_minus_best_control_MV_AP50_window", ""),
            "same_frame_collision_count": row.get("same_frame_collision_count", ""),
            "missing_mask_raster_count": row.get("missing_mask_raster_count", ""),
            "gate_pass": row.get("v91_phase8_progress_gate_pass", ""),
            "failure_type": adaptive._control_failure(row),
        }
        variant_gate_rows.append(gate_row)
        if not adaptive._bool(row.get("v91_phase8_progress_gate_pass")):
            variant_failure_rows.append(gate_row)

    next_action = {
        "decision": "DEV_GATE_PASS_READY_FOR_PHASE8_FREEZE" if passing else "NO_PROGRESS_IN_AFFINITY_SEMANTIC_CONSENSUS_FAMILY",
        "reason": (
            "At least one affinity+semantic consensus variant passed v91 Phase8 progress gate."
            if passing
            else "Five affinity+semantic consensus variants did not pass the v91 Phase8 progress gate; stop this family unless a new affinity readout mechanism is introduced."
        ),
        "best_variant_id": best.get("variant_id", ""),
        "best_delta_vs_phase8_best_MV_AP_window": best_delta_mv_ap,
        "best_delta_vs_phase8_best_MV_AP50_window": best_delta_mv_ap50,
        "do_not_run_holdout": not bool(passing),
        "recommended_next": "refresh Phase8 and freeze before holdout" if passing else "do not run holdout; continue affinity-feature readout only if new mechanism is introduced",
    }
    summary = {
        "phase": "v91_phase4_affinity_semantic_consensus_repair",
        "schema": "stream4d_v91_phase4_affinity_semantic_consensus_repair_v1",
        "variant_count": len(config_rows),
        "best_variant_id": best.get("variant_id", ""),
        "best_variant_gate": best,
        "reference_phase8_best_variant": phase8.get("best_real_variant", ""),
        "reference_phase8_best_MV_AP_window": reference_mv_ap,
        "reference_phase8_best_MV_AP50_window": reference_mv_ap50,
        "best_delta_vs_phase8_best_MV_AP_window": best_delta_mv_ap,
        "best_delta_vs_phase8_best_MV_AP50_window": best_delta_mv_ap50,
        "family_stop_rule_applies": (not passing) and best_delta_mv_ap < 0.002,
        "any_v91_phase8_progress_gate_pass": bool(passing),
        "decision": next_action["decision"],
        "next_action": next_action["recommended_next"],
        "row_counts": {
            "variant_config_rows": len(config_rows),
            "scene_profile_rows": len(profile_rows),
            "scene_policy_rows": len(scene_policy_rows),
            "source_selection_rows": len(source_selection_rows_all),
            "dropped_source_rows": len(dropped_source_rows_all),
            "affinity_feature_rows": len(affinity_feature_rows_all),
            "generated_rows": len(generated_rows_all),
            "selected_rows": len(selected_rows_all),
            "eval_rows": len(eval_rows_all),
            "scored_rows": len(scored_rows_all),
            "support_quality_rows": len(support_quality_all),
            "metric_rows": len(metric_rows),
            "casebook_rows": len(case_rows),
            "control_metric_rows": len(control_rows),
            "variant_gate_rows": len(variant_gate_rows),
            "variant_failure_rows": len(variant_failure_rows),
        },
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "uses_affinity_feature": True,
        "uses_semantic_feature": True,
        "affinity_source": {
            "v80_phase1": adaptive._rel(V80_PHASE1),
            "v80_phase2": adaptive._rel(V80_PHASE2),
            "v80_phase4": adaptive._rel(V80_PHASE4),
        },
        "semantic_source": {
            "label": semantic_source_label,
            "feature_store": adaptive._rel(feature_store_path) if feature_store_path else "",
            "semantic_vector_rows": adaptive._rel(semantic_vector_csv),
            "semantic_feature_rows": [adaptive._rel(path) for path in semantic_feature_rows],
            "semantic_vector_count": len(semantic_vectors),
            "slot_vector_count": len(slot_vectors),
        },
        "metric_source": "v65 local-window evaluator via radius_sweep._evaluate_variant",
        "runtime_sec": time.time() - started,
    }

    _write_csv(OUT / "variant_config_rows.csv", config_rows)
    _write_csv(OUT / "scene_profile_rows.csv", profile_rows)
    _write_csv(OUT / "scene_policy_rows.csv", scene_policy_rows)
    _write_csv(OUT / "source_selection_rows.csv", source_selection_rows_all)
    _write_csv(OUT / "dropped_source_rows.csv", dropped_source_rows_all)
    _write_csv(OUT / "affinity_semantic_feature_rows.csv", affinity_feature_rows_all)
    _write_csv(OUT / "generated_rows.csv", generated_rows_all)
    _write_csv(OUT / "selected_rows.csv", selected_rows_all)
    _write_csv(OUT / "mv_object_frame_mask_rows.csv", eval_rows_all)
    _write_csv(OUT / "scored_frame_mask_rows.csv", scored_rows_all)
    _write_csv(OUT / "support_quality_rows.csv", support_quality_all)
    _write_csv(OUT / "mv_metric_rows.csv", metric_rows)
    _write_csv(OUT / "mv_metric_aggregate_rows.csv", aggregate_rows)
    _write_csv(OUT / "control_metric_rows.csv", control_rows)
    _write_csv(OUT / "variant_gate_rows.csv", variant_gate_rows)
    _write_csv(OUT / "variant_failure_rows.csv", variant_failure_rows)
    _write_csv(OUT / "risk_rows.csv", risk_rows)
    _write_csv(OUT / "casebook_rows.csv", case_rows)
    _write_json(OUT / "best_variant_summary.json", best)
    _write_json(OUT / "next_action_recommendation.json", next_action)
    _write_json(OUT / "summary.json", summary)
    outputs = [
        OUT / "variant_config_rows.csv",
        OUT / "scene_profile_rows.csv",
        OUT / "scene_policy_rows.csv",
        OUT / "source_selection_rows.csv",
        OUT / "dropped_source_rows.csv",
        OUT / "affinity_semantic_feature_rows.csv",
        OUT / "generated_rows.csv",
        OUT / "selected_rows.csv",
        OUT / "mv_object_frame_mask_rows.csv",
        OUT / "scored_frame_mask_rows.csv",
        OUT / "support_quality_rows.csv",
        OUT / "mv_metric_rows.csv",
        OUT / "mv_metric_aggregate_rows.csv",
        OUT / "control_metric_rows.csv",
        OUT / "variant_gate_rows.csv",
        OUT / "variant_failure_rows.csv",
        OUT / "risk_rows.csv",
        OUT / "casebook_rows.csv",
        OUT / "best_variant_summary.json",
        OUT / "next_action_recommendation.json",
        OUT / "summary.json",
    ]
    _write_json(OUT / "SHA256SUMS.json", {adaptive._rel(path): adaptive._sha256(path) for path in outputs if path.exists()})
    print(json.dumps(adaptive._jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v91 Phase4 affinity+semantic consensus readout repair.")
    parser.add_argument("--output-root", default="outputs/audit/v91_phase4_affinity_semantic_consensus_repair")
    parser.add_argument("--semantic-feature-rows", default="outputs/audit/v71_semantic_features/mask_feature_rows.csv,outputs/audit/v81_dino_feature_json_scene0011_scene0050/mask_feature_rows.csv")
    parser.add_argument("--semantic-vector-rows", default=str(SEMANTIC_VECTOR_ROWS.relative_to(ROOT)))
    parser.add_argument("--feature-store", default="")
    parser.add_argument("--semantic-source-label", default="DINO_feature_json")
    parser.add_argument("--variant-prefix", default="V91_AF")
    parser.add_argument("--semantic-tag", default="semvec")
    parser.add_argument("--include-hard-gates", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
