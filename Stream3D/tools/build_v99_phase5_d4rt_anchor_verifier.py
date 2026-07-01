#!/usr/bin/env python3
"""Evaluate D4RT reliable anchors as an F2 verifier/cannot-link layer."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools import build_v99_phase1_f2_base_reproduction as p1  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase5_d4rt_anchor_verifier"
PHASE0_SUMMARY = AUDIT_ROOT / "v99_phase0_fact_lock/summary.json"
PHASE2_DIR = AUDIT_ROOT / "v99_phase2_f2_strengthening"
PHASE2_SUMMARY = PHASE2_DIR / "best_variant_summary.json"
D4RT_ROOT = AUDIT_ROOT / "v97_phase2_d4rt_micro_tracks_full_D3_gpu7"
D4RT_ROWS = D4RT_ROOT / "micro_track_rows.csv"
D4RT_QUALITY_ROWS = D4RT_ROOT / "micro_track_quality_rows.csv"
D4RT_SUMMARY = D4RT_ROOT / "summary.json"

RELIABILITY_SIGMA_J = 0.08
BOOST_EPS = 1e-4
CONFLICT_EPS = 1e-4
TAU_OVERLAP = 0.02
TAU_CONFLICT = 0.01
MIN_ANCHOR_MASS = 1e-3
LOCAL2HISTORY_MERGE_TAU = 0.12
RANDOM_SEED = 9905


def _rel(path: Path | str) -> str:
    q = Path(path)
    try:
        return q.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return q.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in keys})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _hash_float(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "little")
    return float(value / float(2**64 - 1))


class QueryStat:
    def __init__(self, scene: str, window: str) -> None:
        self.scene = scene
        self.window = window
        self.row_count = 0
        self.in_image_count = 0
        self.conf_sum = 0.0
        self.visibility_sum = 0.0
        self.last_label: int | None = None
        self.transition_count = 0
        self.flip_count = 0
        self.frames: set[int] = set()

    def add(self, *, confidence: float, visibility: float, in_image: bool, label_id: int | None, frame_id: int) -> None:
        self.row_count += 1
        self.conf_sum += confidence
        self.visibility_sum += visibility
        if in_image:
            self.in_image_count += 1
            self.frames.add(frame_id)
            label = int(label_id or 0)
            if self.last_label is not None:
                self.transition_count += 1
                if label != self.last_label:
                    self.flip_count += 1
            self.last_label = label


def _load_quality() -> dict[tuple[str, str], dict[str, float]]:
    quality: dict[tuple[str, str], dict[str, float]] = {}
    with D4RT_QUALITY_ROWS.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            quality[(row.get("scene_id", ""), row.get("window_id", ""))] = {
                "jitter": _num(row.get("projection_jitter_p90"), _num(row.get("projection_jitter_mean"), 0.0)),
                "window_flip": _num(row.get("mask_membership_flip_rate"), 0.0),
                "target_frame_count": _num(row.get("target_frame_count"), 0.0),
            }
    return quality


def _phase2_best_rows() -> tuple[str, list[dict[str, Any]]]:
    summary = json.loads(PHASE2_SUMMARY.read_text(encoding="utf-8"))
    variant = str(summary["best_variant_id"])
    rows = [dict(row) for row in _read_csv(PHASE2_DIR / "mv_object_frame_mask_rows.csv") if row.get("variant_id") == variant]
    if not rows:
        raise RuntimeError(f"no rows for Phase2 best variant {variant}")
    return variant, rows


def _selected_mask_index(rows: list[dict[str, Any]]) -> set[tuple[str, int, int]]:
    return {
        (str(row["scene_id"]), int(row["frame_id"]), int(row["selected_mask_id"]))
        for row in rows
    }


def _build_anchor_sets(
    parent_rows: list[dict[str, Any]],
    scope: dict[str, Any],
) -> tuple[
    dict[str, dict[tuple[str, int, int], dict[str, float]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    selected_masks = _selected_mask_index(parent_rows)
    quality = _load_quality()
    label_cache: dict[tuple[str, int], np.ndarray | None] = {}
    query_stats: dict[str, QueryStat] = {}
    query_hits: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    source_uses_gt = False
    source_uses_future = False
    raw_row_count = 0
    uv_in01_count = 0
    selected_mask_hit_count = 0

    def get_label(scene: str, frame: int) -> np.ndarray | None:
        key = (scene, frame)
        if key in label_cache:
            return label_cache[key]
        path = scope["mask_path_by_frame"].get(key)
        if path is None or not path.exists():
            label_cache[key] = None
        else:
            label_cache[key] = p1._read_label(path)
        return label_cache[key]

    with D4RT_ROWS.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw_row_count += 1
            scene = row.get("scene_id", "")
            window = row.get("window_id", "")
            query_id = row.get("query_id", "")
            if not scene or not query_id:
                continue
            frame = p1._int(row.get("target_frame_id"), -1)
            stat = query_stats.get(query_id)
            if stat is None:
                stat = QueryStat(scene, window)
                query_stats[query_id] = stat
            source_uses_gt = source_uses_gt or _bool(row.get("uses_gt_for_prediction"))
            source_uses_future = source_uses_future or _bool(row.get("uses_future"))
            in_image = _bool(row.get("uv_in01"))
            label_id: int | None = None
            if in_image:
                uv_in01_count += 1
                label = get_label(scene, frame)
                if label is not None:
                    u = int(round(_num(row.get("u_tgt"))))
                    v = int(round(_num(row.get("v_tgt"))))
                    if 0 <= v < label.shape[0] and 0 <= u < label.shape[1]:
                        label_id = int(label[v, u])
                        if (scene, frame, label_id) in selected_masks:
                            query_hits[query_id].append((scene, frame, label_id))
                            selected_mask_hit_count += 1
            stat.add(
                confidence=_num(row.get("confidence")),
                visibility=_num(row.get("visibility")),
                in_image=in_image,
                label_id=label_id,
                frame_id=frame,
            )

    query_metric_rows: list[dict[str, Any]] = []
    reliabilities_by_window: dict[tuple[str, str], list[float]] = defaultdict(list)
    query_reliability: dict[str, float] = {}
    query_bucket: dict[str, dict[str, bool]] = {}
    for query_id, stat in query_stats.items():
        q = quality.get((stat.scene, stat.window), {})
        confidence = stat.conf_sum / max(1, stat.row_count)
        visibility = stat.visibility_sum / max(1, stat.row_count)
        support_rate = stat.in_image_count / max(1, stat.row_count)
        flip_rate = stat.flip_count / max(1, stat.transition_count)
        jitter = max(0.0, q.get("jitter", 0.0))
        reliability = float(confidence * visibility * support_rate * math.exp(-jitter / RELIABILITY_SIGMA_J) * (1.0 - flip_rate))
        query_reliability[query_id] = reliability
        reliabilities_by_window[(stat.scene, stat.window)].append(reliability)
        query_metric_rows.append(
            {
                "schema_version": "stream4d_v99_phase5_d4rt_query_reliability_v1",
                "phase_id": "v99_phase5_d4rt_anchor_verifier",
                "query_id": query_id,
                "scene_id": stat.scene,
                "window_id": stat.window,
                "target_row_count": stat.row_count,
                "visible_in_image_count": stat.in_image_count,
                "support_rate": support_rate,
                "confidence_mean": confidence,
                "visibility_mean": visibility,
                "jitter_proxy_window_p90": jitter,
                "flip_rate_query": flip_rate,
                "reliability_score": reliability,
                "selected_mask_hit_count": len(query_hits.get(query_id, [])),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )

    thresholds: dict[tuple[str, str], dict[str, float]] = {}
    for key, values in reliabilities_by_window.items():
        arr = np.asarray(values, dtype=np.float32)
        thresholds[key] = {
            "R20": float(np.quantile(arr, 0.80)),
            "R40": float(np.quantile(arr, 0.60)),
            "LOW20": float(np.quantile(arr, 0.20)),
        }
    for query_id, stat in query_stats.items():
        t = thresholds.get((stat.scene, stat.window), {"R20": float("inf"), "R40": float("inf"), "LOW20": -float("inf")})
        reliability = query_reliability[query_id]
        query_bucket[query_id] = {
            "R20": reliability >= t["R20"],
            "R40": reliability >= t["R40"],
            "LOW20": reliability <= t["LOW20"],
        }

    real_r20: dict[tuple[str, int, int], dict[str, float]] = defaultdict(dict)
    real_r40: dict[tuple[str, int, int], dict[str, float]] = defaultdict(dict)
    low20: dict[tuple[str, int, int], dict[str, float]] = defaultdict(dict)
    for query_id, hits in query_hits.items():
        reliability = query_reliability[query_id]
        buckets = query_bucket[query_id]
        for key in hits:
            if buckets["R20"]:
                real_r20[key][query_id] = max(real_r20[key].get(query_id, 0.0), reliability)
            if buckets["R40"]:
                real_r40[key][query_id] = max(real_r40[key].get(query_id, 0.0), reliability)
            if buckets["LOW20"]:
                low20[key][query_id] = max(low20[key].get(query_id, 0.0), reliability)

    all_carriers = sorted(query_reliability)
    rnd = random.Random(RANDOM_SEED)
    shuffled_r20: dict[tuple[str, int, int], dict[str, float]] = defaultdict(dict)
    for key, weights in real_r20.items():
        key_text = f"{key[0]}:{key[1]}:{key[2]}"
        for cid, weight in weights.items():
            if not all_carriers:
                continue
            digest = hashlib.sha1(f"{key_text}|{cid}|mask_local_shuffle|{RANDOM_SEED}".encode("utf-8")).digest()
            mapped = all_carriers[int.from_bytes(digest[:8], "little") % len(all_carriers)]
            shuffled_r20[key][mapped] = max(shuffled_r20[key].get(mapped, 0.0), weight)

    random_same_count: dict[tuple[str, int, int], dict[str, float]] = defaultdict(dict)
    for key, weights in real_r20.items():
        count = len(weights)
        if count <= 0 or not all_carriers:
            continue
        sample = rnd.sample(all_carriers, min(count, len(all_carriers)))
        mean_weight = float(np.mean(list(weights.values()))) if weights else 0.0
        random_same_count[key] = {cid: mean_weight for cid in sample}

    frames_by_scene = {
        scene: sorted({frame for (s, frame, _mask) in real_r20 if s == scene})
        for scene in {s for (s, _frame, _mask) in real_r20}
    }
    prev_frame_by_scene: dict[tuple[str, int], int] = {}
    for scene, frames in frames_by_scene.items():
        for idx, frame in enumerate(frames):
            if idx > 0:
                prev_frame_by_scene[(scene, frame)] = frames[idx - 1]
    stale_r20: dict[tuple[str, int, int], dict[str, float]] = defaultdict(dict)
    for scene, frame, mask_id in real_r20:
        prev = prev_frame_by_scene.get((scene, frame))
        if prev is not None:
            stale_r20[(scene, frame, mask_id)] = dict(real_r20.get((scene, prev, mask_id), {}))

    carrier_rows: list[dict[str, Any]] = []
    for query_id, row in zip([r["query_id"] for r in query_metric_rows], query_metric_rows):
        buckets = query_bucket.get(query_id, {})
        if not (buckets.get("R20") or buckets.get("R40") or buckets.get("LOW20")):
            continue
        carrier_rows.append(
            {
                "schema_version": "stream4d_v99_phase5_d4rt_reliable_anchor_v1",
                "phase_id": "v99_phase5_d4rt_anchor_verifier",
                "carrier_id": query_id,
                "scene_id": row["scene_id"],
                "window_id": row["window_id"],
                "reliability_score": row["reliability_score"],
                "confidence_mean": row["confidence_mean"],
                "visibility_mean": row["visibility_mean"],
                "support_rate": row["support_rate"],
                "jitter_proxy_window_p90": row["jitter_proxy_window_p90"],
                "flip_rate_query": row["flip_rate_query"],
                "selected_R20": buckets.get("R20", False),
                "selected_R40": buckets.get("R40", False),
                "selected_LOW20": buckets.get("LOW20", False),
                "selected_mask_hit_count": row["selected_mask_hit_count"],
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )

    set_family = {
        "real_R20": real_r20,
        "real_R40": real_r40,
        "shuffled_R20": shuffled_r20,
        "lowconf_R20": low20,
        "random_same_count_R20": random_same_count,
        "stale_R20": stale_r20,
    }
    stats = {
        "d4rt_raw_row_count": raw_row_count,
        "d4rt_uv_in01_count": uv_in01_count,
        "d4rt_query_count": len(query_stats),
        "selected_mask_hit_count": selected_mask_hit_count,
        "mask_observation_with_real_R20_anchor_count": len(real_r20),
        "mask_observation_with_real_R40_anchor_count": len(real_r40),
        "source_uses_gt_for_prediction": source_uses_gt,
        "source_uses_future": source_uses_future,
        "reliability_sigma_j": RELIABILITY_SIGMA_J,
        "top_quantile_scope": "per_scene_window_query_reliability",
        "shuffled_control_policy": "mask_observation_local_carrier_hash_shuffle_preserve_anchor_count_and_weight",
    }
    return set_family, carrier_rows, query_metric_rows, stats


def _weighted_jaccard(a: dict[str, float], b: dict[str, float]) -> tuple[float, float, float, int]:
    if not a and not b:
        return 0.0, 0.0, 0.0, 0
    keys = set(a) | set(b)
    inter = 0.0
    union = 0.0
    shared = 0
    for key in keys:
        av = float(a.get(key, 0.0))
        bv = float(b.get(key, 0.0))
        if av > 0.0 and bv > 0.0:
            shared += 1
        inter += min(av, bv)
        union += max(av, bv)
    return float(inter / max(1e-12, union)), float(sum(a.values())), float(sum(b.values())), shared


def _link_metrics(
    parent_rows: list[dict[str, Any]],
    anchor_sets: dict[tuple[str, int, int], dict[str, float]],
    *,
    family_id: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in parent_rows:
        by_object[str(row["mv_object_id"])].append(row)
    link_rows: list[dict[str, Any]] = []
    obj_overlap: dict[str, list[float]] = defaultdict(list)
    obj_conflict: dict[str, int] = defaultdict(int)
    for oid, vals in sorted(by_object.items()):
        vals_sorted = sorted(vals, key=lambda r: (str(r["scene_id"]), str(r["chunk_id"]), int(r["frame_id"]), int(r["selected_mask_id"])))
        for a, b in zip(vals_sorted[:-1], vals_sorted[1:]):
            if a["scene_id"] != b["scene_id"] or a["chunk_id"] != b["chunk_id"] or int(a["frame_id"]) == int(b["frame_id"]):
                continue
            key_a = (str(a["scene_id"]), int(a["frame_id"]), int(a["selected_mask_id"]))
            key_b = (str(b["scene_id"]), int(b["frame_id"]), int(b["selected_mask_id"]))
            wa = anchor_sets.get(key_a, {})
            wb = anchor_sets.get(key_b, {})
            overlap, mass_a, mass_b, shared = _weighted_jaccard(wa, wb)
            confident = mass_a >= MIN_ANCHOR_MASS and mass_b >= MIN_ANCHOR_MASS
            conflict = bool(confident and overlap < TAU_CONFLICT)
            if confident:
                obj_overlap[oid].append(overlap)
            if conflict:
                obj_conflict[oid] += 1
            link_rows.append(
                {
                    "schema_version": "stream4d_v99_phase5_d4rt_link_metric_v1",
                    "phase_id": "v99_phase5_d4rt_anchor_verifier",
                    "anchor_family": family_id,
                    "mv_object_id": oid,
                    "scene_id": a["scene_id"],
                    "chunk_id": a["chunk_id"],
                    "frame_a": int(a["frame_id"]),
                    "mask_a": int(a["selected_mask_id"]),
                    "frame_b": int(b["frame_id"]),
                    "mask_b": int(b["selected_mask_id"]),
                    "anchor_mass_a": mass_a,
                    "anchor_mass_b": mass_b,
                    "shared_anchor_count": shared,
                    "anchor_overlap": overlap,
                    "anchor_confident": confident,
                    "anchor_conflict": conflict,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    obj_metric: dict[str, dict[str, float]] = {}
    for oid in by_object:
        vals = obj_overlap.get(oid, [])
        obj_metric[oid] = {
            "mean_anchor_overlap": float(np.mean(vals)) if vals else 0.0,
            "confident_link_count": float(len(vals)),
            "conflict_count": float(obj_conflict.get(oid, 0)),
        }
    return link_rows, obj_metric


def _norm(values: dict[str, float]) -> dict[str, float]:
    vals = list(values.values())
    if not vals:
        return {}
    lo = min(vals)
    hi = max(vals)
    if hi - lo <= 1e-12:
        return {key: 0.0 for key in values}
    return {key: (val - lo) / (hi - lo) for key, val in values.items()}


class DSU:
    def __init__(self, ids: list[str]) -> None:
        self.parent = {item: item for item in ids}
        self.size = {item: 1 for item in ids}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def _local2history_merge_map(parent_rows: list[dict[str, Any]], anchor_sets: dict[tuple[str, int, int], dict[str, float]]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in parent_rows:
        by_object[str(row["mv_object_id"])].append(row)
    ids = sorted(by_object)
    dsu = DSU(ids)
    object_sets: dict[str, dict[str, float]] = {}
    object_frames: dict[str, set[tuple[str, int]]] = {}
    object_scene: dict[str, str] = {}
    object_chunk: dict[str, str] = {}
    for oid, vals in by_object.items():
        merged: dict[str, float] = {}
        frames: set[tuple[str, int]] = set()
        for row in vals:
            key = (str(row["scene_id"]), int(row["frame_id"]), int(row["selected_mask_id"]))
            frames.add((str(row["scene_id"]), int(row["frame_id"])))
            for cid, weight in anchor_sets.get(key, {}).items():
                merged[cid] = max(merged.get(cid, 0.0), weight)
        object_sets[oid] = merged
        object_frames[oid] = frames
        object_scene[oid] = str(vals[0]["scene_id"])
        object_chunk[oid] = str(vals[0]["chunk_id"])

    merge_rows: list[dict[str, Any]] = []
    pairs_considered = 0
    chunks_by_scene: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for oid in ids:
        chunks_by_scene[object_scene[oid]][object_chunk[oid]].append(oid)
    for scene, chunk_map in sorted(chunks_by_scene.items()):
        chunk_ids = sorted(chunk_map)
        for left, right in zip(chunk_ids[:-1], chunk_ids[1:]):
            for oid_a in chunk_map[left]:
                if not object_sets[oid_a]:
                    continue
                for oid_b in chunk_map[right]:
                    if not object_sets[oid_b] or object_frames[oid_a] & object_frames[oid_b]:
                        continue
                    pairs_considered += 1
                    overlap, mass_a, mass_b, shared = _weighted_jaccard(object_sets[oid_a], object_sets[oid_b])
                    if overlap >= LOCAL2HISTORY_MERGE_TAU:
                        dsu.union(oid_a, oid_b)
                        merge_rows.append(
                            {
                                "schema_version": "stream4d_v99_phase5_d4rt_local2history_merge_v1",
                                "phase_id": "v99_phase5_d4rt_anchor_verifier",
                                "scene_id": scene,
                                "left_chunk_id": left,
                                "right_chunk_id": right,
                                "mv_object_id_a": oid_a,
                                "mv_object_id_b": oid_b,
                                "object_anchor_overlap": overlap,
                                "anchor_mass_a": mass_a,
                                "anchor_mass_b": mass_b,
                                "shared_anchor_count": shared,
                                "merge_tau": LOCAL2HISTORY_MERGE_TAU,
                                "uses_gt_for_prediction": False,
                                "uses_future": False,
                            }
                        )
    mapping = {oid: f"D5_d4rt_l2h:{dsu.find(oid)}" for oid in ids}
    if not merge_rows:
        merge_rows.append(
            {
                "schema_version": "stream4d_v99_phase5_d4rt_local2history_merge_v1",
                "phase_id": "v99_phase5_d4rt_anchor_verifier",
                "scene_id": "",
                "left_chunk_id": "",
                "right_chunk_id": "",
                "mv_object_id_a": "",
                "mv_object_id_b": "",
                "object_anchor_overlap": "",
                "anchor_mass_a": "",
                "anchor_mass_b": "",
                "shared_anchor_count": "",
                "merge_tau": LOCAL2HISTORY_MERGE_TAU,
                "pairs_considered": pairs_considered,
                "note": "no local2history merges met the D4RT anchor overlap threshold",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return mapping, merge_rows


def _variant_rows(
    parent_rows: list[dict[str, Any]],
    metrics_by_family: dict[str, dict[str, dict[str, float]]],
    *,
    l2h_map: dict[str, str],
) -> list[dict[str, Any]]:
    variants = {
        "P5_B0_phase2_best_no_d4rt": ("parent_phase2_score_replay", "real_R20", "none"),
        "P5_D1_anchor_boost_top20": ("phase2_score_plus_1e-4_d4rt_anchor_overlap_top20", "real_R20", "boost"),
        "P5_D2_anchor_veto_top20": ("phase2_score_minus_1e-4_d4rt_conflict_top20", "real_R20", "veto"),
        "P5_D3_anchor_boost_plus_veto_top20": ("phase2_score_plus_overlap_minus_conflict_top20", "real_R20", "boost_veto"),
        "P5_D4_anchor_boost_plus_veto_top40": ("phase2_score_plus_overlap_minus_conflict_top40", "real_R40", "boost_veto"),
        "P5_D5_anchor_local2history_only": ("phase2_score_replay_object_id_merge_by_d4rt_anchor", "real_R20", "l2h"),
        "P5_C1_shuffled_D4RT_anchor": ("control_shuffled_anchor_boost_plus_veto", "shuffled_R20", "boost_veto"),
        "P5_C2_low_conf_D4RT_anchor": ("control_low_conf_anchor_boost_plus_veto", "lowconf_R20", "boost_veto"),
        "P5_C3_random_anchor_same_count": ("control_random_anchor_same_count_boost_plus_veto", "random_same_count_R20", "boost_veto"),
        "P5_C4_stale_D4RT_anchor": ("control_stale_previous_frame_anchor_boost_plus_veto", "stale_R20", "boost_veto"),
    }
    norm_cache: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
    for family, metrics in metrics_by_family.items():
        overlap_norm = _norm({oid: vals["mean_anchor_overlap"] for oid, vals in metrics.items()})
        conflict_norm = _norm({oid: vals["conflict_count"] for oid, vals in metrics.items()})
        norm_cache[family] = (overlap_norm, conflict_norm)

    out: list[dict[str, Any]] = []
    for variant_id, (policy, family, mode) in variants.items():
        overlap_norm, conflict_norm = norm_cache.get(family, ({}, {}))
        family_metrics = metrics_by_family.get(family, {})
        for row in parent_rows:
            oid = str(row["mv_object_id"])
            score = _num(row.get("score"), 1.0)
            if mode in {"boost", "boost_veto"}:
                score += BOOST_EPS * overlap_norm.get(oid, 0.0)
            if mode in {"veto", "boost_veto"}:
                score -= CONFLICT_EPS * conflict_norm.get(oid, 0.0)
            new = dict(row)
            if mode == "l2h":
                new["mv_object_id"] = l2h_map.get(oid, oid)
                new["object_id"] = new["mv_object_id"]
                new["object_id_policy"] = "d4rt_anchor_local2history_merge"
            new["variant_id"] = variant_id
            new["score"] = float(score)
            new["score_policy"] = policy
            new["phase5_parent_variant_id"] = row["variant_id"]
            new["d4rt_anchor_family"] = family
            new["d4rt_mean_anchor_overlap"] = family_metrics.get(oid, {}).get("mean_anchor_overlap", 0.0)
            new["d4rt_confident_link_count"] = family_metrics.get(oid, {}).get("confident_link_count", 0.0)
            new["d4rt_conflict_count"] = family_metrics.get(oid, {}).get("conflict_count", 0.0)
            new["uses_gt_for_prediction"] = False
            new["uses_future"] = False
            out.append(new)
    return out


def main() -> int:
    started = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase0 = json.loads(PHASE0_SUMMARY.read_text(encoding="utf-8"))
    phase2_summary = json.loads(PHASE2_SUMMARY.read_text(encoding="utf-8"))
    if not bool(phase2_summary.get("phase2_full_pass")):
        raise RuntimeError("Phase5 requires Phase2 full pass")
    d4rt_summary = json.loads(D4RT_SUMMARY.read_text(encoding="utf-8"))
    parent_variant, parent_rows = _phase2_best_rows()
    scope = p1._load_source_scope()
    set_family, carrier_rows, query_metric_rows, anchor_stats = _build_anchor_sets(parent_rows, scope)

    link_rows: list[dict[str, Any]] = []
    metrics_by_family: dict[str, dict[str, dict[str, float]]] = {}
    link_summary_by_family: dict[str, dict[str, float]] = {}
    for family, anchor_sets in set_family.items():
        family_links, obj_metric = _link_metrics(parent_rows, anchor_sets, family_id=family)
        link_rows.extend(family_links)
        metrics_by_family[family] = obj_metric
        overlaps = [_num(row["anchor_overlap"]) for row in family_links if _bool(row["anchor_confident"])]
        link_summary_by_family[family] = {
            "link_count": float(len(family_links)),
            "anchor_confident_link_count": float(sum(1 for row in family_links if _bool(row["anchor_confident"]))),
            "anchor_overlap_link_count": float(sum(1 for row in family_links if _num(row["anchor_overlap"]) >= TAU_OVERLAP)),
            "anchor_conflict_link_count": float(sum(1 for row in family_links if _bool(row["anchor_conflict"]))),
            "anchor_overlap_mean": float(np.mean(overlaps)) if overlaps else 0.0,
        }

    l2h_map, l2h_merge_rows = _local2history_merge_map(parent_rows, set_family["real_R20"])
    all_rows = _variant_rows(parent_rows, metrics_by_family, l2h_map=l2h_map)

    metric_scene_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for variant in sorted({row["variant_id"] for row in all_rows}):
        rows = [row for row in all_rows if row["variant_id"] == variant]
        metrics, frames = p1._evaluate_variant(variant, rows, scope)
        metric_scene_rows.extend(metrics)
        frame_rows.extend(frames)
    aggregate_rows = p1._aggregate_metrics(metric_scene_rows)

    family_by_variant = {
        "P5_B0_phase2_best_no_d4rt": "real_R20",
        "P5_D1_anchor_boost_top20": "real_R20",
        "P5_D2_anchor_veto_top20": "real_R20",
        "P5_D3_anchor_boost_plus_veto_top20": "real_R20",
        "P5_D4_anchor_boost_plus_veto_top40": "real_R40",
        "P5_D5_anchor_local2history_only": "real_R20",
        "P5_C1_shuffled_D4RT_anchor": "shuffled_R20",
        "P5_C2_low_conf_D4RT_anchor": "lowconf_R20",
        "P5_C3_random_anchor_same_count": "random_same_count_R20",
        "P5_C4_stale_D4RT_anchor": "stale_R20",
    }
    real_variants = {
        "P5_D1_anchor_boost_top20",
        "P5_D2_anchor_veto_top20",
        "P5_D3_anchor_boost_plus_veto_top20",
        "P5_D4_anchor_boost_plus_veto_top40",
        "P5_D5_anchor_local2history_only",
    }
    control_variants = {
        "P5_C1_shuffled_D4RT_anchor",
        "P5_C2_low_conf_D4RT_anchor",
        "P5_C3_random_anchor_same_count",
        "P5_C4_stale_D4RT_anchor",
    }
    for row in aggregate_rows:
        variant = str(row["variant_id"])
        family = family_by_variant.get(variant, "real_R20")
        summary = link_summary_by_family.get(family, {})
        selected_carriers = [
            r
            for r in carrier_rows
            if (_bool(r.get("selected_R20")) if family.endswith("R20") else _bool(r.get("selected_R40")))
        ]
        row["d4rt_anchor_family"] = family
        row["reliable_anchor_count"] = len(selected_carriers)
        row["reliable_anchor_rate"] = float(row["reliable_anchor_count"] / max(1, anchor_stats["d4rt_query_count"]))
        row["anchor_visibility_mean"] = p1._mean([r.get("visibility_mean") for r in selected_carriers])
        row["anchor_jitter_p90"] = p1._mean([r.get("jitter_proxy_window_p90") for r in selected_carriers])
        row["anchor_flip_median"] = (
            float(np.median([_num(r.get("flip_rate_query")) for r in selected_carriers]))
            if selected_carriers
            else 0.0
        )
        row["anchor_overlap_link_count"] = int(summary.get("anchor_overlap_link_count", 0.0))
        row["anchor_conflict_link_count"] = int(summary.get("anchor_conflict_link_count", 0.0))
        row["shuffled_anchor_overlap_count"] = int(link_summary_by_family.get("shuffled_R20", {}).get("anchor_overlap_link_count", 0.0))
        row["anchor_confident_link_count"] = int(summary.get("anchor_confident_link_count", 0.0))
        row["anchor_overlap_mean"] = summary.get("anchor_overlap_mean", 0.0)
        row["local2history_merge_count"] = max(0, len([r for r in l2h_merge_rows if r.get("mv_object_id_a")]))
        row["uses_gt_for_prediction"] = False
        row["uses_future"] = False

    base_window = float(phase0["F2_base_full_dev_MV_AP_window"])
    base_ap50 = float(phase0["F2_base_full_dev_MV_AP50_window"])
    base_scene = float(phase0["F2_base_full_dev_MV_AP_scene"])
    phase2_window = float(phase2_summary["best_MV_AP_window"])
    phase2_scene = float(phase2_summary["best_MV_AP_scene"])
    best_real = max([row for row in aggregate_rows if row["variant_id"] in real_variants], key=lambda row: (float(row["MV_AP_window"]), float(row["MV_AP_scene"]), float(row["MV_AP50_window"])))
    best_shuffled = max([row for row in aggregate_rows if row["variant_id"] == "P5_C1_shuffled_D4RT_anchor"], key=lambda row: (float(row["MV_AP_window"]), float(row["MV_AP_scene"])))
    best_control = max([row for row in aggregate_rows if row["variant_id"] in control_variants], key=lambda row: (float(row["MV_AP_window"]), float(row["MV_AP_scene"])))
    real_vs_shuffled_window = float(best_real["MV_AP_window"]) - float(best_shuffled["MV_AP_window"])
    real_vs_shuffled_scene = float(best_real["MV_AP_scene"]) - float(best_shuffled["MV_AP_scene"])
    plan_local_success = bool(float(best_real["MV_AP_window"]) >= base_window + 0.005 and float(best_real["MV_AP50_window"]) >= base_ap50 + 0.010)
    plan_scene_success = bool(float(best_real["MV_AP_scene"]) >= base_scene + 0.010 and float(best_real["MV_AP_window"]) >= base_window - 0.003)
    control_margin_success = bool(real_vs_shuffled_window >= 0.005 or real_vs_shuffled_scene >= 0.010)
    contribution_vs_phase2 = bool(float(best_real["MV_AP_window"]) > phase2_window + 1e-12 or float(best_real["MV_AP_scene"]) > phase2_scene + 1e-12)
    safety_pass = bool(
        int(best_real["same_frame_collision_count"]) == 0
        and float(best_real["pixel_collision_rate"]) <= 0.02
        and int(best_real["missing_mask_raster_count"]) == 0
    )
    phase5_pass = bool((plan_local_success or plan_scene_success) and control_margin_success and safety_pass)
    gate_rows = [
        {
            "gate_id": "D4RT_plan_local_success_vs_F2_base",
            "pass": plan_local_success,
            "expected": f"MV_AP_window>={base_window + 0.005} and MV_AP50_window>={base_ap50 + 0.010}",
            "observed": f"MV_AP_window={best_real['MV_AP_window']}; MV_AP50_window={best_real['MV_AP50_window']}",
            "severity": "plan_success",
        },
        {
            "gate_id": "D4RT_plan_scene_success_vs_F2_base",
            "pass": plan_scene_success,
            "expected": f"MV_AP_scene>={base_scene + 0.010} and MV_AP_window>={base_window - 0.003}",
            "observed": f"MV_AP_scene={best_real['MV_AP_scene']}; MV_AP_window={best_real['MV_AP_window']}",
            "severity": "plan_success_alternative",
        },
        {
            "gate_id": "real_minus_shuffled_D4RT_margin",
            "pass": control_margin_success,
            "expected": "real-shuffled >=0.005 MV_AP_window or >=0.010 MV_AP_scene",
            "observed": f"window_margin={real_vs_shuffled_window}; scene_margin={real_vs_shuffled_scene}",
            "severity": "required_control",
        },
        {
            "gate_id": "D4RT_increment_over_phase2_window_or_scene",
            "pass": contribution_vs_phase2,
            "expected": f"window>{phase2_window} or scene>{phase2_scene}",
            "observed": f"window={best_real['MV_AP_window']}; scene={best_real['MV_AP_scene']}",
            "severity": "diagnostic_contribution",
        },
        {
            "gate_id": "same_frame_collision_count_eq_0",
            "pass": int(best_real["same_frame_collision_count"]) == 0,
            "expected": "0",
            "observed": best_real["same_frame_collision_count"],
            "severity": "required",
        },
        {
            "gate_id": "pixel_collision_rate_le_0p02",
            "pass": float(best_real["pixel_collision_rate"]) <= 0.02,
            "expected": "<=0.02",
            "observed": best_real["pixel_collision_rate"],
            "severity": "required",
        },
        {
            "gate_id": "missing_mask_raster_count_eq_0",
            "pass": int(best_real["missing_mask_raster_count"]) == 0,
            "expected": "0",
            "observed": best_real["missing_mask_raster_count"],
            "severity": "required",
        },
    ]
    blocking_failures = []
    for row in gate_rows:
        if bool(row["pass"]):
            continue
        if row["gate_id"] == "D4RT_plan_local_success_vs_F2_base" and plan_scene_success:
            continue
        if row["gate_id"] == "D4RT_plan_scene_success_vs_F2_base" and plan_local_success:
            continue
        if row["severity"] == "diagnostic_contribution":
            continue
        blocking_failures.append(row)
    failure_rows = [
        {
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": "follow Phase5 repair ladder: if real≈shuffled keep D4RT diagnostic; if veto helps keep cannot-link only; if top20 sparse/top40 noisy keep local2history diagnostic only",
        }
        for row in blocking_failures
    ]
    decision = "PASS_D4RT_ANCHOR_VERIFIER_CONTRIBUTION" if phase5_pass else "NO_GO_D4RT_ANCHOR_DIAGNOSTIC_ONLY"

    casebook_rows = [
        {
            "schema_version": "stream4d_v99_phase5_casebook_v1",
            "phase_id": "v99_phase5_d4rt_anchor_verifier",
            "rank": idx,
            "variant_id": row["variant_id"],
            "d4rt_anchor_family": row["d4rt_anchor_family"],
            "MV_AP_window": row["MV_AP_window"],
            "MV_AP50_window": row["MV_AP50_window"],
            "MV_AP_scene": row["MV_AP_scene"],
            "MV_AP50_scene": row["MV_AP50_scene"],
            "delta_vs_phase2_window": float(row["MV_AP_window"]) - phase2_window,
            "delta_vs_phase2_scene": float(row["MV_AP_scene"]) - phase2_scene,
            "anchor_confident_link_count": row["anchor_confident_link_count"],
            "anchor_overlap_link_count": row["anchor_overlap_link_count"],
            "anchor_conflict_link_count": row["anchor_conflict_link_count"],
        }
        for idx, row in enumerate(sorted(aggregate_rows, key=lambda r: float(r["MV_AP_window"]), reverse=True), start=1)
    ]

    summary = {
        "schema_version": "stream4d_v99_phase5_d4rt_anchor_verifier_summary_v1",
        "phase_id": "v99_phase5_d4rt_anchor_verifier",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "decision": decision,
        "phase5_pass": phase5_pass,
        "parent_phase2_variant": parent_variant,
        "d4rt_source_root": _rel(D4RT_ROOT),
        "d4rt_source_decision": d4rt_summary.get("decision"),
        "d4rt_source_decode_scope": d4rt_summary.get("decode_scope"),
        "d4rt_source_uses_gt_for_prediction": d4rt_summary.get("uses_gt_for_prediction"),
        "d4rt_source_uses_future": d4rt_summary.get("uses_future"),
        "best_real_variant": best_real["variant_id"],
        "best_real_MV_AP_window": float(best_real["MV_AP_window"]),
        "best_real_MV_AP50_window": float(best_real["MV_AP50_window"]),
        "best_real_MV_AP_scene": float(best_real["MV_AP_scene"]),
        "best_real_MV_AP50_scene": float(best_real["MV_AP50_scene"]),
        "best_shuffled_variant": best_shuffled["variant_id"],
        "best_shuffled_MV_AP_window": float(best_shuffled["MV_AP_window"]),
        "best_shuffled_MV_AP_scene": float(best_shuffled["MV_AP_scene"]),
        "best_control_variant": best_control["variant_id"],
        "best_control_MV_AP_window": float(best_control["MV_AP_window"]),
        "best_control_MV_AP_scene": float(best_control["MV_AP_scene"]),
        "real_minus_shuffled_MV_AP_window": real_vs_shuffled_window,
        "real_minus_shuffled_MV_AP_scene": real_vs_shuffled_scene,
        "phase2_best_MV_AP_window": phase2_window,
        "phase2_best_MV_AP_scene": phase2_scene,
        "delta_best_real_vs_phase2_window": float(best_real["MV_AP_window"]) - phase2_window,
        "delta_best_real_vs_phase2_scene": float(best_real["MV_AP_scene"]) - phase2_scene,
        "blocking_failure_count": len(failure_rows),
        "anchor_stats": anchor_stats,
        "link_summary_by_family": link_summary_by_family,
        "local2history_merge_count": max(0, len([r for r in l2h_merge_rows if r.get("mv_object_id_a")])),
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "outputs": {
            "d4rt_query_reliability_rows": _rel(OUT_DIR / "d4rt_query_reliability_rows.csv"),
            "d4rt_reliable_anchor_rows": _rel(OUT_DIR / "d4rt_reliable_anchor_rows.csv"),
            "link_metric_rows": _rel(OUT_DIR / "link_metric_rows.csv"),
            "local2history_merge_rows": _rel(OUT_DIR / "local2history_merge_rows.csv"),
            "variant_metric_rows": _rel(OUT_DIR / "variant_metric_rows.csv"),
            "variant_metric_scene_rows": _rel(OUT_DIR / "variant_metric_scene_rows.csv"),
            "variant_gate_rows": _rel(OUT_DIR / "variant_gate_rows.csv"),
            "variant_failure_rows": _rel(OUT_DIR / "variant_failure_rows.csv"),
            "casebook_rows": _rel(OUT_DIR / "casebook_rows.csv"),
            "summary": _rel(OUT_DIR / "summary.json"),
        },
    }
    _write_csv(OUT_DIR / "d4rt_query_reliability_rows.csv", query_metric_rows)
    _write_csv(OUT_DIR / "d4rt_reliable_anchor_rows.csv", carrier_rows)
    _write_csv(OUT_DIR / "link_metric_rows.csv", link_rows)
    _write_csv(OUT_DIR / "local2history_merge_rows.csv", l2h_merge_rows)
    _write_csv(OUT_DIR / "variant_metric_rows.csv", aggregate_rows)
    _write_csv(OUT_DIR / "variant_metric_scene_rows.csv", metric_scene_rows)
    _write_csv(OUT_DIR / "variant_gate_rows.csv", gate_rows)
    _write_csv(OUT_DIR / "variant_failure_rows.csv", failure_rows)
    _write_csv(OUT_DIR / "casebook_rows.csv", casebook_rows)
    _write_csv(OUT_DIR / "mv_object_frame_mask_rows.csv", all_rows)
    _write_csv(OUT_DIR / "frame_eval_rows.csv", frame_rows)
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if phase5_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
