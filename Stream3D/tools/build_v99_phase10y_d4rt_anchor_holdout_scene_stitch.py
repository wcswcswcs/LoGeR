#!/usr/bin/env python3
"""Holdout D4RT reliable-anchor scene stitching with chunk32/overlap3.

This is a repair follow-up after Phase10R/X:

* DA3 and D4RT provider artifacts must be chunk_size=32 with 3 overlap.
* D4RT must use its own self-overlap stitched coordinates before anchor use.
* This phase does not mix DA3 and D4RT geometry, because the explicit
  DA3<->D4RT Sim3/scale alignment audit is not implemented here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools import build_v99_phase1_f2_base_reproduction as p1  # noqa: E402
from tools import build_v99_phase10k_holdout_chunk_object_birth_sweep as p10k  # noqa: E402
from tools import build_v99_phase10o_overlap3_scene_stitch_repair as p10o  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase10y_d4rt_anchor_holdout_scene_stitch"
PHASE0_DIR = AUDIT_ROOT / "v99_phase0_fact_lock"
PHASE10O_DIR = AUDIT_ROOT / "v99_phase10o_overlap3_scene_stitch_repair"
D4RT_ROOTS = [
    AUDIT_ROOT / "v99_phase10w_d4rt_holdout_chunk32o3_stitched_scene0011",
    AUDIT_ROOT / "v99_phase10w_d4rt_holdout_chunk32o3_stitched_scene0050",
]
WINDOW_ALIGNMENT_ROWS: Path | None = None
REQUIRE_WINDOW_ALIGNMENT = False

BASE_VARIANT = "O0_overlap3_chunk_birth_primary_emit"
CHUNK_SIZE = 32
OVERLAP = 3
RELIABILITY_SIGMA_J = 0.08
MIN_ANCHOR_MASS = 1e-3
TAU_CONFLICT = 0.01
RANDOM_SEED = 9910
MAX_CANDIDATES_PER_ADJACENT_PAIR = 50000


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
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _jsonable(row.get(field, "")) for field in fields})


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


def _int(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _load_window_alignment_scores() -> tuple[dict[tuple[str, str], dict[str, float]], dict[str, Any]]:
    if WINDOW_ALIGNMENT_ROWS is None:
        return {}, {"window_alignment_used": False}
    path = Path(WINDOW_ALIGNMENT_ROWS)
    rows = _read_csv(path)
    scores: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        scores[(row.get("scene_id", ""), row.get("window_id", ""))] = {
            "alignment_score": _num(row.get("alignment_score"), 0.0),
            "aligned_residual_p50_m": _num(row.get("aligned_residual_p50_m"), float("nan")),
            "aligned_residual_p90_m": _num(row.get("aligned_residual_p90_m"), float("nan")),
            "prefix_anchor_count": _num(row.get("prefix_anchor_count"), 0.0),
            "uses_future": 1.0 if _bool(row.get("uses_future")) else 0.0,
        }
    return scores, {
        "window_alignment_used": True,
        "window_alignment_rows": _rel(path),
        "window_alignment_row_count": len(rows),
        "require_window_alignment": REQUIRE_WINDOW_ALIGNMENT,
        "window_alignment_uses_future_count": sum(1 for value in scores.values() if value.get("uses_future", 0.0) > 0.0),
    }


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


class DSU:
    def __init__(self, ids: list[str]) -> None:
        self.parent = {item: item for item in ids}
        self.size = {item: 1 for item in ids}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: str, b: str) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


def _base_rows() -> list[dict[str, Any]]:
    rows = [
        dict(row)
        for row in _read_csv(PHASE10O_DIR / "mv_object_frame_mask_rows.csv")
        if row.get("variant_id") == BASE_VARIANT
    ]
    if not rows:
        raise RuntimeError(f"missing Phase10O base rows for {BASE_VARIANT}")
    return rows


def _selected_mask_index(rows: list[dict[str, Any]]) -> set[tuple[str, int, int]]:
    return {
        (str(row["scene_id"]), _int(row["frame_id"], -1), _int(row["selected_mask_id"], -1))
        for row in rows
    }


def _d4rt_row_to_label_xy(row: dict[str, Any], label: np.ndarray) -> tuple[int, int]:
    label_h, label_w = int(label.shape[0]), int(label.shape[1])
    grid_w = _int(row.get("d4rt_output_width"), label_w)
    grid_h = _int(row.get("d4rt_output_height"), label_h)
    u = _num(row.get("u_tgt"))
    v = _num(row.get("v_tgt"))
    if grid_w > 1 and grid_h > 1 and (grid_w != label_w or grid_h != label_h):
        u = (u / float(grid_w - 1)) * float(max(1, label_w - 1))
        v = (v / float(grid_h - 1)) * float(max(1, label_h - 1))
    return int(round(u)), int(round(v))


def _parse_frame_ids(text: str) -> list[int]:
    out: list[int] = []
    for tok in str(text).replace(" ", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(_int(tok, -1))
    return [v for v in out if v >= 0]


def _load_quality_and_contract() -> tuple[dict[tuple[str, str], dict[str, float]], dict[str, Any]]:
    quality: dict[tuple[str, str], dict[str, float]] = {}
    windows_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_uses_gt = False
    source_uses_future = False
    root_rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for root in D4RT_ROOTS:
        summary_path = root / "summary.json"
        rows_path = root / "micro_track_rows.csv"
        quality_path = root / "micro_track_quality_rows.csv"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        root_rows.append(
            {
                "root": _rel(root),
                "summary_exists": summary_path.exists(),
                "rows_exists": rows_path.exists(),
                "quality_exists": quality_path.exists(),
                "decision": summary.get("decision", ""),
                "d4rt_applies_overlap_stitch": summary.get("d4rt_applies_overlap_stitch", ""),
                "stitched_track_row_count": summary.get("stitched_track_row_count", ""),
                "uses_gt_for_prediction": summary.get("uses_gt_for_prediction", ""),
                "uses_future": summary.get("uses_future", ""),
            }
        )
        if not rows_path.exists() or not quality_path.exists():
            blockers.append(f"missing D4RT rows or quality file under {_rel(root)}")
            continue
        if not bool(summary.get("d4rt_applies_overlap_stitch", False)):
            blockers.append(f"D4RT self-overlap stitch not proved for {_rel(root)}")
        source_uses_gt = source_uses_gt or bool(summary.get("uses_gt_for_prediction", False))
        source_uses_future = source_uses_future or bool(summary.get("uses_future", False))
        for row in _read_csv(quality_path):
            scene = row.get("scene_id", "")
            window = row.get("window_id", "")
            frames = _parse_frame_ids(row.get("frame_ids", ""))
            target_count = _int(row.get("target_frame_count"), len(frames))
            quality[(scene, window)] = {
                "jitter": _num(row.get("projection_jitter_p90"), _num(row.get("projection_jitter_mean"), 0.0)),
                "window_flip": _num(row.get("mask_membership_flip_rate"), 0.0),
                "target_frame_count": float(target_count),
            }
            windows_by_scene[scene].append(
                {
                    "window_id": window,
                    "target_frame_count": target_count,
                    "frame_ids": frames,
                    "uses_gt_for_prediction": _bool(row.get("uses_gt_for_prediction")),
                    "uses_future": _bool(row.get("uses_future")),
                }
            )
            source_uses_gt = source_uses_gt or _bool(row.get("uses_gt_for_prediction"))
            source_uses_future = source_uses_future or _bool(row.get("uses_future"))

    overlap_rows: list[dict[str, Any]] = []
    observed_overlaps: list[int] = []
    observed_chunk_counts: list[int] = []
    for scene, windows in sorted(windows_by_scene.items()):
        ordered = sorted(windows, key=lambda row: str(row["window_id"]))
        for row in ordered:
            observed_chunk_counts.append(int(row["target_frame_count"]))
            if int(row["target_frame_count"]) > CHUNK_SIZE:
                blockers.append(f"{scene} {row['window_id']} has target_frame_count>{CHUNK_SIZE}")
            if int(row["target_frame_count"]) == CHUNK_SIZE:
                pass
        for left, right in zip(ordered[:-1], ordered[1:]):
            shared = sorted(set(left["frame_ids"]) & set(right["frame_ids"]))
            observed_overlaps.append(len(shared))
            overlap_rows.append(
                {
                    "schema_version": "stream4d_v99_phase10y_d4rt_overlap_contract_v1",
                    "phase_id": "v99_phase10y_d4rt_anchor_holdout_scene_stitch",
                    "scene_id": scene,
                    "left_window_id": left["window_id"],
                    "right_window_id": right["window_id"],
                    "shared_frame_count": len(shared),
                    "shared_frames": " ".join(str(v) for v in shared),
                    "required_overlap": OVERLAP,
                    "contract_pass": len(shared) == OVERLAP,
                }
            )
            if len(shared) != OVERLAP:
                blockers.append(f"{scene} {left['window_id']}->{right['window_id']} overlap is {len(shared)}, expected {OVERLAP}")

    contract = {
        "required_chunk_size": CHUNK_SIZE,
        "required_overlap": OVERLAP,
        "observed_chunk_count_min": min(observed_chunk_counts) if observed_chunk_counts else 0,
        "observed_chunk_count_max": max(observed_chunk_counts) if observed_chunk_counts else 0,
        "observed_overlap_min": min(observed_overlaps) if observed_overlaps else 0,
        "observed_overlap_max": max(observed_overlaps) if observed_overlaps else 0,
        "final_partial_chunk_allowed": True,
        "d4rt_self_overlap_stitch_required": True,
        "d4rt_self_overlap_stitch_contract_pass": not blockers and not source_uses_gt and not source_uses_future,
        "uses_gt_for_prediction": source_uses_gt,
        "uses_future": source_uses_future,
        "root_rows": root_rows,
        "overlap_rows": overlap_rows,
        "blockers": blockers,
    }
    return quality, contract


def _iter_d4rt_rows() -> Any:
    for root in D4RT_ROOTS:
        rows_path = root / "micro_track_rows.csv"
        with rows_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                yield root, row


def _build_anchor_sets(
    parent_rows: list[dict[str, Any]],
    scope: dict[str, Any],
    quality: dict[tuple[str, str], dict[str, float]],
) -> tuple[
    dict[str, dict[tuple[str, int, int], dict[str, float]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    selected_masks = _selected_mask_index(parent_rows)
    window_alignment, window_alignment_stats = _load_window_alignment_scores()
    label_cache: dict[tuple[str, int], np.ndarray | None] = {}
    query_stats: dict[str, QueryStat] = {}
    query_hits: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    raw_row_count = 0
    uv_in01_count = 0
    selected_mask_hit_count = 0
    overlap_stitch_false_count = 0
    source_uses_gt = False
    source_uses_future = False
    coordinate_modes: Counter[str] = Counter()

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

    for _root, row in _iter_d4rt_rows():
        raw_row_count += 1
        scene = row.get("scene_id", "")
        window = row.get("window_id", "")
        query_id = row.get("query_id", "")
        if not scene or not query_id:
            continue
        frame = _int(row.get("target_frame_id"), -1)
        stat = query_stats.get(query_id)
        if stat is None:
            stat = QueryStat(scene, window)
            query_stats[query_id] = stat
        source_uses_gt = source_uses_gt or _bool(row.get("uses_gt_for_prediction"))
        source_uses_future = source_uses_future or _bool(row.get("uses_future"))
        if not _bool(row.get("overlap_stitch_applied")):
            overlap_stitch_false_count += 1
        coordinate_modes[str(row.get("geometry_coordinate_mode", ""))] += 1
        in_image = _bool(row.get("uv_in01"))
        label_id: int | None = None
        if in_image:
            uv_in01_count += 1
            label = get_label(scene, frame)
            if label is not None:
                u, v = _d4rt_row_to_label_xy(row, label)
                if 0 <= v < label.shape[0] and 0 <= u < label.shape[1]:
                    label_id = int(label[v, u])
                    key = (scene, frame, label_id)
                    if key in selected_masks:
                        query_hits[query_id].append(key)
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
        alignment = window_alignment.get((stat.scene, stat.window))
        if alignment is None:
            alignment_score = 0.0 if REQUIRE_WINDOW_ALIGNMENT else 1.0
            alignment_p50 = float("nan")
            alignment_p90 = float("nan")
            alignment_prefix_count = 0.0
        else:
            alignment_score = float(alignment.get("alignment_score", 0.0))
            alignment_p50 = float(alignment.get("aligned_residual_p50_m", float("nan")))
            alignment_p90 = float(alignment.get("aligned_residual_p90_m", float("nan")))
            alignment_prefix_count = float(alignment.get("prefix_anchor_count", 0.0))
        reliability = float(
            confidence
            * visibility
            * support_rate
            * math.exp(-jitter / RELIABILITY_SIGMA_J)
            * (1.0 - flip_rate)
            * alignment_score
        )
        query_reliability[query_id] = reliability
        reliabilities_by_window[(stat.scene, stat.window)].append(reliability)
        query_metric_rows.append(
            {
                "schema_version": "stream4d_v99_phase10y_d4rt_query_reliability_v1",
                "phase_id": "v99_phase10y_d4rt_anchor_holdout_scene_stitch",
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
                "alignment_score": alignment_score,
                "alignment_residual_p50_m": alignment_p50,
                "alignment_residual_p90_m": alignment_p90,
                "alignment_prefix_anchor_count": alignment_prefix_count,
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
            digest = hashlib.sha1(f"{key_text}|{cid}|holdout_shuffle|{RANDOM_SEED}".encode("utf-8")).digest()
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
    for row in query_metric_rows:
        query_id = str(row["query_id"])
        buckets = query_bucket.get(query_id, {})
        if not (buckets.get("R20") or buckets.get("R40") or buckets.get("LOW20")):
            continue
        carrier_rows.append(
            {
                "schema_version": "stream4d_v99_phase10y_d4rt_reliable_anchor_v1",
                "phase_id": "v99_phase10y_d4rt_anchor_holdout_scene_stitch",
                "carrier_id": query_id,
                "scene_id": row["scene_id"],
                "window_id": row["window_id"],
                "reliability_score": row["reliability_score"],
                "confidence_mean": row["confidence_mean"],
                "visibility_mean": row["visibility_mean"],
                "support_rate": row["support_rate"],
                "jitter_proxy_window_p90": row["jitter_proxy_window_p90"],
                "flip_rate_query": row["flip_rate_query"],
                "alignment_score": row["alignment_score"],
                "alignment_residual_p50_m": row["alignment_residual_p50_m"],
                "alignment_residual_p90_m": row["alignment_residual_p90_m"],
                "alignment_prefix_anchor_count": row["alignment_prefix_anchor_count"],
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
        "overlap_stitch_false_row_count": overlap_stitch_false_count,
        "geometry_coordinate_modes": dict(coordinate_modes),
        "reliability_sigma_j": RELIABILITY_SIGMA_J,
        "top_quantile_scope": "per_scene_window_query_reliability",
        "cross_model_geometry_edge_used": False,
        "da3_d4rt_sim3_alignment_used": bool(window_alignment_stats.get("window_alignment_used", False)),
        **window_alignment_stats,
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


def _object_infos(rows: list[dict[str, Any]], anchor_sets: dict[tuple[str, int, int], dict[str, float]]) -> dict[str, dict[str, Any]]:
    infos: dict[str, dict[str, Any]] = defaultdict(lambda: {"rows": [], "frames": set(), "chunks": set(), "anchors": {}})
    for row in rows:
        oid = str(row["mv_object_id"])
        scene = str(row["scene_id"])
        chunk = str(row["chunk_id"])
        frame = _int(row["frame_id"], -1)
        mask = _int(row["selected_mask_id"], -1)
        infos[oid]["rows"].append(row)
        infos[oid]["scene_id"] = scene
        infos[oid]["chunk_id"] = chunk
        infos[oid]["frames"].add((scene, frame))
        infos[oid]["chunks"].add(chunk)
        for cid, weight in anchor_sets.get((scene, frame, mask), {}).items():
            anchors = infos[oid]["anchors"]
            anchors[cid] = max(float(anchors.get(cid, 0.0)), float(weight))
    return dict(infos)


def _chunk_index(chunk_id: str) -> int:
    if chunk_id.startswith("c"):
        return int(chunk_id[1:])
    return int(chunk_id)


def _candidate_pairs(
    rows: list[dict[str, Any]],
    anchor_sets: dict[tuple[str, int, int], dict[str, float]],
    *,
    family_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    infos = _object_infos(rows, anchor_sets)
    ids = sorted(infos)
    by_scene_chunk: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    anchor_mass: dict[str, float] = {}
    for oid in ids:
        by_scene_chunk[str(infos[oid]["scene_id"])][str(infos[oid]["chunk_id"])].append(oid)
        anchor_mass[oid] = float(sum(float(v) for v in infos[oid]["anchors"].values()))

    candidate_rows: list[dict[str, Any]] = []
    link_rows: list[dict[str, Any]] = []
    for oid, info in sorted(infos.items()):
        vals = sorted(info["rows"], key=lambda r: (str(r["scene_id"]), str(r["chunk_id"]), _int(r["frame_id"]), _int(r["selected_mask_id"])))
        for a, b in zip(vals[:-1], vals[1:]):
            if a["scene_id"] != b["scene_id"] or a["chunk_id"] != b["chunk_id"] or _int(a["frame_id"]) == _int(b["frame_id"]):
                continue
            key_a = (str(a["scene_id"]), _int(a["frame_id"]), _int(a["selected_mask_id"]))
            key_b = (str(b["scene_id"]), _int(b["frame_id"]), _int(b["selected_mask_id"]))
            overlap, mass_a, mass_b, shared = _weighted_jaccard(anchor_sets.get(key_a, {}), anchor_sets.get(key_b, {}))
            confident = mass_a >= MIN_ANCHOR_MASS and mass_b >= MIN_ANCHOR_MASS
            link_rows.append(
                {
                    "schema_version": "stream4d_v99_phase10y_d4rt_link_metric_v1",
                    "phase_id": "v99_phase10y_d4rt_anchor_holdout_scene_stitch",
                    "anchor_family": family_id,
                    "mv_object_id": oid,
                    "scene_id": a["scene_id"],
                    "chunk_id": a["chunk_id"],
                    "frame_a": _int(a["frame_id"]),
                    "mask_a": _int(a["selected_mask_id"]),
                    "frame_b": _int(b["frame_id"]),
                    "mask_b": _int(b["selected_mask_id"]),
                    "anchor_mass_a": mass_a,
                    "anchor_mass_b": mass_b,
                    "shared_anchor_count": shared,
                    "anchor_overlap": overlap,
                    "anchor_confident": confident,
                    "anchor_conflict": bool(confident and overlap < TAU_CONFLICT),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )

    for scene, chunk_map in sorted(by_scene_chunk.items()):
        chunk_ids = sorted(chunk_map, key=_chunk_index)
        for left, right in zip(chunk_ids[:-1], chunk_ids[1:]):
            right_by_carrier: dict[str, list[tuple[str, float]]] = defaultdict(list)
            for oid_b in chunk_map[right]:
                anchors_b = infos[oid_b]["anchors"]
                if not anchors_b:
                    continue
                for carrier, weight in anchors_b.items():
                    right_by_carrier[carrier].append((oid_b, float(weight)))
            pair_inter: dict[tuple[str, str], float] = defaultdict(float)
            pair_shared: Counter[tuple[str, str]] = Counter()
            for oid_a in chunk_map[left]:
                anchors_a = infos[oid_a]["anchors"]
                if not anchors_a:
                    continue
                for carrier, weight_a in anchors_a.items():
                    for oid_b, weight_b in right_by_carrier.get(carrier, []):
                        if infos[oid_a]["frames"] & infos[oid_b]["frames"]:
                            continue
                        key = (oid_a, oid_b)
                        pair_inter[key] += min(float(weight_a), float(weight_b))
                        pair_shared[key] += 1
            pair_rows: list[dict[str, Any]] = []
            for (oid_a, oid_b), inter in pair_inter.items():
                mass_a = anchor_mass.get(oid_a, 0.0)
                mass_b = anchor_mass.get(oid_b, 0.0)
                union = max(1e-12, mass_a + mass_b - inter)
                overlap = float(inter / union)
                pair_rows.append(
                    {
                        "schema_version": "stream4d_v99_phase10y_d4rt_local2history_candidate_v1",
                        "phase_id": "v99_phase10y_d4rt_anchor_holdout_scene_stitch",
                        "scene_id": scene,
                        "left_chunk_id": left,
                        "right_chunk_id": right,
                        "mv_object_id_a": oid_a,
                        "mv_object_id_b": oid_b,
                        "anchor_family": family_id,
                        "object_anchor_overlap": overlap,
                        "anchor_mass_a": mass_a,
                        "anchor_mass_b": mass_b,
                        "shared_anchor_count": int(pair_shared[(oid_a, oid_b)]),
                        "candidate_generator": "shared_carrier_inverted_index",
                        "candidate_truncated": False,
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )
            pair_rows = sorted(
                pair_rows,
                key=lambda row: (_num(row.get("object_anchor_overlap")), _num(row.get("shared_anchor_count"))),
                reverse=True,
            )
            if len(pair_rows) > MAX_CANDIDATES_PER_ADJACENT_PAIR:
                for row in pair_rows[MAX_CANDIDATES_PER_ADJACENT_PAIR:]:
                    row["candidate_truncated"] = True
                pair_rows = pair_rows[:MAX_CANDIDATES_PER_ADJACENT_PAIR]
            candidate_rows.extend(pair_rows)
    stats = {
        "object_count": len(ids),
        "object_with_anchor_count": sum(1 for oid in ids if infos[oid]["anchors"]),
        "candidate_pair_count": len(candidate_rows),
        "candidate_generator": "shared_carrier_inverted_index",
        "max_candidates_per_adjacent_pair": MAX_CANDIDATES_PER_ADJACENT_PAIR,
        "candidate_truncated_count": sum(1 for row in candidate_rows if _bool(row.get("candidate_truncated"))),
        "within_object_link_count": len(link_rows),
        "within_object_confident_link_count": sum(1 for row in link_rows if _bool(row.get("anchor_confident"))),
        "within_object_conflict_link_count": sum(1 for row in link_rows if _bool(row.get("anchor_conflict"))),
    }
    return candidate_rows, link_rows, stats


def _merge_mapping(
    ids: list[str],
    candidates: list[dict[str, Any]],
    *,
    tau: float,
    variant_id: str,
    policy: str,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    dsu = DSU(ids)
    accepted: list[dict[str, Any]] = []
    used_left: set[tuple[str, str, str]] = set()
    used_right: set[tuple[str, str, str]] = set()
    for row in sorted(candidates, key=lambda r: (_num(r.get("object_anchor_overlap")), _num(r.get("shared_anchor_count"))), reverse=True):
        overlap = _num(row.get("object_anchor_overlap"))
        if overlap < tau:
            continue
        a = str(row["mv_object_id_a"])
        b = str(row["mv_object_id_b"])
        if a not in dsu.parent or b not in dsu.parent:
            continue
        if policy == "one_to_one_greedy":
            left_key = (str(row["scene_id"]), str(row["left_chunk_id"]), a)
            right_key = (str(row["scene_id"]), str(row["right_chunk_id"]), b)
            if left_key in used_left or right_key in used_right:
                continue
            if dsu.union(a, b):
                used_left.add(left_key)
                used_right.add(right_key)
            else:
                continue
        else:
            if not dsu.union(a, b):
                continue
        new = dict(row)
        new["schema_version"] = "stream4d_v99_phase10y_d4rt_local2history_merge_v1"
        new["variant_id"] = variant_id
        new["merge_tau"] = tau
        new["merge_policy"] = policy
        accepted.append(new)
    return {oid: f"{variant_id}:{dsu.find(oid)}" for oid in ids}, accepted


def _apply_mapping(rows: list[dict[str, Any]], *, variant_id: str, mapping: dict[str, str], policy: str, anchor_family: str, tau: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        oid = str(row["mv_object_id"])
        new = dict(row)
        new["variant_id"] = variant_id
        new["variant"] = variant_id
        new["phase10y_parent_mv_object_id"] = oid
        new["mv_object_id"] = mapping.get(oid, f"{variant_id}:{oid}")
        new["object_id"] = new["mv_object_id"]
        new["object_id_policy"] = f"d4rt_anchor_local2history_{policy}"
        new["score_scope"] = "current_chunk_score_scene_stitched_identity"
        new["score_policy"] = str(row.get("score_policy", "")) + "__phase10y_d4rt_anchor_scene_stitch"
        new["d4rt_anchor_family"] = anchor_family
        new["d4rt_merge_tau"] = tau
        new["uses_gt_for_prediction"] = False
        new["uses_future"] = False
        out.append(new)
    return out


def _component_stats(mapping: dict[str, str]) -> dict[str, Any]:
    comps: dict[str, int] = defaultdict(int)
    for root in mapping.values():
        comps[root] += 1
    original = len(mapping)
    scene_objects = len(comps)
    merged = original - scene_objects
    large = sum(1 for size in comps.values() if size > 3)
    return {
        "scene_object_count": scene_objects,
        "history_merge_count": merged,
        "history_split_count": 0,
        "duplicate_scene_object_count": merged,
        "fragmentation_rate_proxy": float(scene_objects / original) if original else 0.0,
        "overmerge_rate_proxy_large_component_gt3": float(large / scene_objects) if scene_objects else 0.0,
        "max_component_size": max(comps.values()) if comps else 0,
    }


def _aggregate_decoupled(variant_id: str, local_rows: list[dict[str, Any]], scene_rows: list[dict[str, Any]], stats: dict[str, Any]) -> dict[str, Any]:
    local_agg = p1._aggregate_metrics(local_rows)[0]
    scene_agg = p1._aggregate_metrics(scene_rows)[0]
    row = dict(local_agg)
    for key, value in scene_agg.items():
        if key.endswith("_scene"):
            row[key] = value
    row["variant_id"] = variant_id
    row["metric_composition"] = "local_from_phase10o_primary_chunk_ids_scene_from_d4rt_anchor_stitched_ids"
    row.update(stats)
    return row


def _eval(variant_id: str, rows: list[dict[str, Any]], eval_scope: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return p1._evaluate_variant(variant_id, rows, eval_scope)


def main() -> int:
    started = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p10k._patch_phase1_inputs()
    phase0 = json.loads((PHASE0_DIR / "summary.json").read_text(encoding="utf-8"))
    phase10o = json.loads((PHASE10O_DIR / "summary.json").read_text(encoding="utf-8"))

    scope = p10o._build_overlap3_scope()
    eval_scope = p10o._eval_scope_from_overlap(scope)
    base_rows = _base_rows()
    quality, contract = _load_quality_and_contract()
    print(
        json.dumps(
            {
                "event": "loaded_contract",
                "d4rt_self_overlap_stitch_contract_pass": contract.get("d4rt_self_overlap_stitch_contract_pass"),
                "observed_overlap_min": contract.get("observed_overlap_min"),
                "observed_overlap_max": contract.get("observed_overlap_max"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    set_family, carrier_rows, query_metric_rows, anchor_stats = _build_anchor_sets(base_rows, scope, quality)
    print(
        json.dumps(
            {
                "event": "built_anchor_sets",
                "d4rt_raw_row_count": anchor_stats.get("d4rt_raw_row_count"),
                "d4rt_query_count": anchor_stats.get("d4rt_query_count"),
                "selected_mask_hit_count": anchor_stats.get("selected_mask_hit_count"),
                "mask_observation_with_real_R20_anchor_count": anchor_stats.get("mask_observation_with_real_R20_anchor_count"),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    ids = sorted({str(row["mv_object_id"]) for row in base_rows})
    base_metric_rows, base_frame_rows = _eval(BASE_VARIANT, base_rows, eval_scope)
    base_agg = p1._aggregate_metrics(base_metric_rows)[0]
    base_agg["metric_composition"] = "phase10o_primary_chunk_scoped_ids"
    base_agg.update(_component_stats({oid: f"{BASE_VARIANT}:{oid}" for oid in ids}))
    base_agg["history_candidate_count"] = 0
    base_agg["accepted_history_merge_edge_count"] = 0
    base_agg["d4rt_anchor_family"] = "none"
    base_agg["d4rt_merge_tau"] = ""
    base_agg["d4rt_merge_policy"] = "none"
    base_agg["formal_ap_claim_allowed"] = False

    family_candidates: dict[str, list[dict[str, Any]]] = {}
    family_stats: dict[str, dict[str, Any]] = {}
    link_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for family_id, anchors in set_family.items():
        candidates, links, stats = _candidate_pairs(base_rows, anchors, family_id=family_id)
        family_candidates[family_id] = candidates
        family_stats[family_id] = stats
        candidate_rows.extend(candidates)
        link_rows.extend(links)
        print(
            json.dumps(
                {
                    "event": "built_family_candidates",
                    "family_id": family_id,
                    "candidate_pair_count": stats.get("candidate_pair_count"),
                    "object_with_anchor_count": stats.get("object_with_anchor_count"),
                    "within_object_confident_link_count": stats.get("within_object_confident_link_count"),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    variant_specs = [
        {"variant_id": "Y1_d4rt_R20_tau0p12_dsu", "family": "real_R20", "tau": 0.12, "policy": "threshold_dsu", "is_control": False},
        {"variant_id": "Y2_d4rt_R20_tau0p08_greedy", "family": "real_R20", "tau": 0.08, "policy": "one_to_one_greedy", "is_control": False},
        {"variant_id": "Y3_d4rt_R20_tau0p04_greedy", "family": "real_R20", "tau": 0.04, "policy": "one_to_one_greedy", "is_control": False},
        {"variant_id": "Y4_d4rt_R40_tau0p08_greedy", "family": "real_R40", "tau": 0.08, "policy": "one_to_one_greedy", "is_control": False},
        {"variant_id": "Y5_d4rt_R40_tau0p04_greedy", "family": "real_R40", "tau": 0.04, "policy": "one_to_one_greedy", "is_control": False},
        {"variant_id": "Y6_d4rt_R40_tau0p02_greedy", "family": "real_R40", "tau": 0.02, "policy": "one_to_one_greedy", "is_control": False},
        {"variant_id": "YC1_shuffled_R20_tau0p04_greedy", "family": "shuffled_R20", "tau": 0.04, "policy": "one_to_one_greedy", "is_control": True},
        {"variant_id": "YC2_random_R20_tau0p04_greedy", "family": "random_same_count_R20", "tau": 0.04, "policy": "one_to_one_greedy", "is_control": True},
        {"variant_id": "YC3_stale_R20_tau0p04_greedy", "family": "stale_R20", "tau": 0.04, "policy": "one_to_one_greedy", "is_control": True},
    ]

    metric_rows: list[dict[str, Any]] = [base_agg]
    same_identity_metric_rows: list[dict[str, Any]] = [base_agg]
    scene_metric_rows: list[dict[str, Any]] = list(base_metric_rows)
    frame_rows: list[dict[str, Any]] = list(base_frame_rows)
    merge_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = [
        {
            "schema_version": "stream4d_v99_phase10y_variant_config_v1",
            "phase_id": "v99_phase10y_d4rt_anchor_holdout_scene_stitch",
            "variant_id": BASE_VARIANT,
            "anchor_family": "none",
            "merge_tau": "",
            "merge_policy": "none",
            "candidate_count": 0,
            "accepted_history_merge_edge_count": 0,
            "chunk_size": CHUNK_SIZE,
            "overlap": OVERLAP,
            "cross_model_geometry_edge_used": False,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
    ]

    for spec in variant_specs:
        variant_id = str(spec["variant_id"])
        family = str(spec["family"])
        tau = float(spec["tau"])
        policy = str(spec["policy"])
        candidates = family_candidates.get(family, [])
        mapping, accepted = _merge_mapping(ids, candidates, tau=tau, variant_id=variant_id, policy=policy)
        rows = _apply_mapping(base_rows, variant_id=variant_id, mapping=mapping, policy=policy, anchor_family=family, tau=tau)
        per_metric, frames = _eval(variant_id, rows, eval_scope)
        same_agg = p1._aggregate_metrics(per_metric)[0]
        stats = _component_stats(mapping)
        same_agg["metric_composition"] = "local_and_scene_from_same_d4rt_anchor_stitched_ids"
        same_agg["history_candidate_count"] = len(candidates)
        same_agg["accepted_history_merge_edge_count"] = len(accepted)
        same_agg["d4rt_anchor_family"] = family
        same_agg["d4rt_merge_tau"] = tau
        same_agg["d4rt_merge_policy"] = policy
        same_agg["is_control"] = bool(spec["is_control"])
        same_agg["formal_ap_claim_allowed"] = False
        same_agg.update(stats)

        decoupled = _aggregate_decoupled(variant_id, base_metric_rows, per_metric, stats)
        decoupled["history_candidate_count"] = len(candidates)
        decoupled["accepted_history_merge_edge_count"] = len(accepted)
        decoupled["d4rt_anchor_family"] = family
        decoupled["d4rt_merge_tau"] = tau
        decoupled["d4rt_merge_policy"] = policy
        decoupled["is_control"] = bool(spec["is_control"])
        decoupled["formal_ap_claim_allowed"] = False

        metric_rows.append(decoupled)
        same_identity_metric_rows.append(same_agg)
        scene_metric_rows.extend(per_metric)
        frame_rows.extend(frames)
        merge_rows.extend(accepted)
        print(
            json.dumps(
                {
                    "event": "evaluated_variant",
                    "variant_id": variant_id,
                    "accepted_history_merge_edge_count": len(accepted),
                    "MV_AP_window": decoupled.get("MV_AP_window"),
                    "MV_AP_scene": decoupled.get("MV_AP_scene"),
                    "same_identity_MV_AP_window": same_agg.get("MV_AP_window"),
                    "same_identity_MV_AP_scene": same_agg.get("MV_AP_scene"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        config_rows.append(
            {
                "schema_version": "stream4d_v99_phase10y_variant_config_v1",
                "phase_id": "v99_phase10y_d4rt_anchor_holdout_scene_stitch",
                "variant_id": variant_id,
                "anchor_family": family,
                "merge_tau": tau,
                "merge_policy": policy,
                "candidate_count": len(candidates),
                "accepted_history_merge_edge_count": len(accepted),
                "is_control": bool(spec["is_control"]),
                "chunk_size": CHUNK_SIZE,
                "overlap": OVERLAP,
                "cross_model_geometry_edge_used": False,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )

    real_variants = {str(spec["variant_id"]) for spec in variant_specs if not bool(spec["is_control"])}
    control_variants = {str(spec["variant_id"]) for spec in variant_specs if bool(spec["is_control"])}
    best_real = max([row for row in metric_rows if row["variant_id"] in real_variants], key=lambda row: (_num(row.get("MV_AP_scene")), _num(row.get("MV_AP50_scene")), _num(row.get("MV_AP_window"))))
    best_control = max([row for row in metric_rows if row["variant_id"] in control_variants], key=lambda row: (_num(row.get("MV_AP_scene")), _num(row.get("MV_AP50_scene")), _num(row.get("MV_AP_window"))))
    best_same_identity_real = max([row for row in same_identity_metric_rows if row["variant_id"] in real_variants], key=lambda row: (_num(row.get("MV_AP_scene")), _num(row.get("MV_AP50_scene")), _num(row.get("MV_AP_window"))))

    holdout_scene_gate = float(phase0["F2_base_holdout_MV_AP_scene"]) + 0.010
    holdout_scene_ap50_gate = float(phase0["F2_base_holdout_MV_AP50_scene"]) + 0.015
    holdout_window_floor = float(phase0["F2_base_holdout_MV_AP_window"]) - 0.003
    strict_local_gate_window = float(phase0["F2_base_holdout_MV_AP_window"]) + 0.005
    strict_local_gate_ap50 = float(phase0["F2_base_holdout_MV_AP50_window"]) + 0.010
    local_gate = _num(best_real.get("MV_AP_window")) >= strict_local_gate_window and _num(best_real.get("MV_AP50_window")) >= strict_local_gate_ap50
    scene_gate = (
        _num(best_real.get("MV_AP_scene")) >= holdout_scene_gate
        and _num(best_real.get("MV_AP50_scene")) >= holdout_scene_ap50_gate
        and _num(best_real.get("MV_AP_window")) >= holdout_window_floor
    )
    same_identity_window_floor_gate = _num(best_same_identity_real.get("MV_AP_window")) >= holdout_window_floor
    control_margin = _num(best_real.get("MV_AP_scene")) - _num(best_control.get("MV_AP_scene"))
    control_margin_gate = control_margin >= 0.010
    safety_gate = (
        int(_num(best_real.get("same_frame_collision_count"), 1)) == 0
        and int(_num(best_real.get("missing_mask_raster_count"), 1)) == 0
        and not bool(scope.get("source_uses_future", False))
        and not bool(scope.get("source_uses_gt_for_prediction", False))
        and not bool(anchor_stats.get("source_uses_future", False))
        and not bool(anchor_stats.get("source_uses_gt_for_prediction", False))
        and int(anchor_stats.get("overlap_stitch_false_row_count", 1)) == 0
        and bool(contract.get("d4rt_self_overlap_stitch_contract_pass", False))
    )
    metric_gate_pass = bool(local_gate and scene_gate and same_identity_window_floor_gate and control_margin_gate and safety_gate)

    gate_rows = [
        {
            "gate_id": "d4rt_provider_chunk32_overlap3_self_stitch",
            "pass": bool(contract.get("d4rt_self_overlap_stitch_contract_pass", False)),
            "expected": "chunk_size=32, overlap=3, D4RT self-overlap stitch true, no GT/future",
            "observed": f"chunk_count_minmax={contract.get('observed_chunk_count_min')}/{contract.get('observed_chunk_count_max')} overlap_minmax={contract.get('observed_overlap_min')}/{contract.get('observed_overlap_max')} blockers={len(contract.get('blockers', []))}",
            "severity": "provider_contract",
        },
        {
            "gate_id": "strict_local_holdout_gate_decoupled",
            "pass": local_gate,
            "expected": f"MV_AP_window>={strict_local_gate_window} and MV_AP50_window>={strict_local_gate_ap50}",
            "observed": f"{best_real['variant_id']} MV_AP_window={best_real.get('MV_AP_window')} MV_AP50_window={best_real.get('MV_AP50_window')}",
            "severity": "method_gate",
        },
        {
            "gate_id": "scene_holdout_gate_vs_F2_holdout",
            "pass": scene_gate,
            "expected": f"MV_AP_scene>={holdout_scene_gate} and MV_AP50_scene>={holdout_scene_ap50_gate} and MV_AP_window>={holdout_window_floor}",
            "observed": f"{best_real['variant_id']} MV_AP_scene={best_real.get('MV_AP_scene')} MV_AP50_scene={best_real.get('MV_AP50_scene')} MV_AP_window={best_real.get('MV_AP_window')}",
            "severity": "scene_method_gate",
        },
        {
            "gate_id": "same_identity_window_floor",
            "pass": same_identity_window_floor_gate,
            "expected": f"same-identity MV_AP_window>={holdout_window_floor}",
            "observed": f"{best_same_identity_real['variant_id']} MV_AP_window={best_same_identity_real.get('MV_AP_window')} MV_AP_scene={best_same_identity_real.get('MV_AP_scene')}",
            "severity": "diagnostic_consistency",
        },
        {
            "gate_id": "real_minus_control_d4rt_margin",
            "pass": control_margin_gate,
            "expected": "best real D4RT scene AP exceeds best shuffled/random/stale control by >=0.010",
            "observed": f"best_real={best_real['variant_id']} scene={best_real.get('MV_AP_scene')} best_control={best_control['variant_id']} scene={best_control.get('MV_AP_scene')} margin={control_margin}",
            "severity": "required_control",
        },
        {
            "gate_id": "safety_no_gt_no_future_no_missing_no_collision",
            "pass": safety_gate,
            "expected": "uses_gt_for_prediction=false; uses_future=false; same_frame_collision_count=0; missing_mask=0; overlap_stitch_false_row_count=0",
            "observed": f"scope_uses_gt={scope.get('source_uses_gt_for_prediction')} scope_uses_future={scope.get('source_uses_future')} anchor_uses_gt={anchor_stats.get('source_uses_gt_for_prediction')} anchor_uses_future={anchor_stats.get('source_uses_future')} overlap_stitch_false={anchor_stats.get('overlap_stitch_false_row_count')} same_frame_collision={best_real.get('same_frame_collision_count')} missing_mask={best_real.get('missing_mask_raster_count')}",
            "severity": "safety",
        },
        {
            "gate_id": "formal_claim_allowed_after_cross_model_alignment",
            "pass": False,
            "expected": "DA3<->D4RT Sim3/scale alignment implemented and audited before any mixed geometry edge",
            "observed": "this phase uses D4RT-only anchors; cross_model_geometry_edge_used=false",
            "severity": "formal_claim_blocker",
        },
    ]
    failure_rows = [
        {
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": (
                "If D4RT real anchor does not beat controls, audit uv_tgt label lookup, reliability bucket quality, "
                "and D4RT stitch scales; do not use DA3-D4RT raw geometry until explicit Sim3/scale alignment passes."
            ),
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]

    casebook_rows = [
        {
            "schema_version": "stream4d_v99_phase10y_casebook_v1",
            "phase_id": "v99_phase10y_d4rt_anchor_holdout_scene_stitch",
            "rank": idx,
            "variant_id": row["variant_id"],
            "metric_composition": row.get("metric_composition"),
            "d4rt_anchor_family": row.get("d4rt_anchor_family"),
            "d4rt_merge_tau": row.get("d4rt_merge_tau"),
            "d4rt_merge_policy": row.get("d4rt_merge_policy"),
            "is_control": row.get("is_control", False),
            "MV_AP_window": row.get("MV_AP_window"),
            "MV_AP50_window": row.get("MV_AP50_window"),
            "MV_AP_scene": row.get("MV_AP_scene"),
            "MV_AP50_scene": row.get("MV_AP50_scene"),
            "accepted_history_merge_edge_count": row.get("accepted_history_merge_edge_count"),
            "history_merge_count": row.get("history_merge_count"),
            "scene_object_count": row.get("scene_object_count"),
            "max_component_size": row.get("max_component_size"),
        }
        for idx, row in enumerate(sorted(metric_rows, key=lambda r: (_num(r.get("MV_AP_scene")), _num(r.get("MV_AP50_scene"))), reverse=True), start=1)
    ]

    summary = {
        "schema_version": "stream4d_v99_phase10y_d4rt_anchor_holdout_scene_stitch_summary_v1",
        "phase_id": "v99_phase10y_d4rt_anchor_holdout_scene_stitch",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "decision": "GO_D4RT_ANCHOR_HOLDOUT_SCENE_STITCH_METRIC_REPAIRED_FORMAL_ALIGNMENT_REQUIRED" if metric_gate_pass else "NO_GO_D4RT_ANCHOR_HOLDOUT_SCENE_STITCH",
        "metric_gate_pass": metric_gate_pass,
        "local_gate_pass": bool(local_gate),
        "scene_gate_pass": bool(scene_gate),
        "same_identity_window_floor_gate_pass": bool(same_identity_window_floor_gate),
        "control_margin_gate_pass": bool(control_margin_gate),
        "safety_gate_pass": bool(safety_gate),
        "formal_claim_allowed": False,
        "cross_model_geometry_edge_used": False,
        "da3_d4rt_sim3_alignment_used": bool(anchor_stats.get("da3_d4rt_sim3_alignment_used", False)),
        "method_chunk_size": CHUNK_SIZE,
        "method_chunk_overlap": OVERLAP,
        "frame_stride": 5,
        "source_phase10o_summary": _rel(PHASE10O_DIR / "summary.json"),
        "source_phase10o_best_MV_AP_scene": phase10o.get("best_MV_AP_scene"),
        "best_real_variant_id": best_real["variant_id"],
        "best_real_MV_AP_window": float(_num(best_real.get("MV_AP_window"))),
        "best_real_MV_AP50_window": float(_num(best_real.get("MV_AP50_window"))),
        "best_real_MV_AP_scene": float(_num(best_real.get("MV_AP_scene"))),
        "best_real_MV_AP50_scene": float(_num(best_real.get("MV_AP50_scene"))),
        "best_same_identity_real_variant_id": best_same_identity_real["variant_id"],
        "best_same_identity_real_MV_AP_window": float(_num(best_same_identity_real.get("MV_AP_window"))),
        "best_same_identity_real_MV_AP_scene": float(_num(best_same_identity_real.get("MV_AP_scene"))),
        "best_control_variant_id": best_control["variant_id"],
        "best_control_MV_AP_window": float(_num(best_control.get("MV_AP_window"))),
        "best_control_MV_AP_scene": float(_num(best_control.get("MV_AP_scene"))),
        "real_minus_control_MV_AP_scene": float(control_margin),
        "base_MV_AP_window": float(_num(base_agg.get("MV_AP_window"))),
        "base_MV_AP_scene": float(_num(base_agg.get("MV_AP_scene"))),
        "phase10p_best_MV_AP_scene": json.loads((AUDIT_ROOT / "v99_phase10p_overlap3_scene_stitch_semantic_sweep/summary.json").read_text(encoding="utf-8")).get("best_MV_AP_scene"),
        "F2_base_holdout_MV_AP_window": float(phase0["F2_base_holdout_MV_AP_window"]),
        "F2_base_holdout_MV_AP50_window": float(phase0["F2_base_holdout_MV_AP50_window"]),
        "F2_base_holdout_MV_AP_scene": float(phase0["F2_base_holdout_MV_AP_scene"]),
        "F2_base_holdout_MV_AP50_scene": float(phase0["F2_base_holdout_MV_AP50_scene"]),
        "anchor_stats": anchor_stats,
        "family_stats": family_stats,
        "d4rt_contract": contract,
        "blocking_failure_count": len(failure_rows),
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "d4rt_query_reliability_rows": _rel(OUT_DIR / "d4rt_query_reliability_rows.csv"),
            "d4rt_reliable_anchor_rows": _rel(OUT_DIR / "d4rt_reliable_anchor_rows.csv"),
            "link_metric_rows": _rel(OUT_DIR / "link_metric_rows.csv"),
            "local2history_candidate_rows": _rel(OUT_DIR / "local2history_candidate_rows.csv"),
            "local2history_merge_rows": _rel(OUT_DIR / "local2history_merge_rows.csv"),
            "variant_metric_rows": _rel(OUT_DIR / "variant_metric_rows.csv"),
            "same_identity_metric_rows": _rel(OUT_DIR / "same_identity_metric_rows.csv"),
            "variant_metric_scene_rows": _rel(OUT_DIR / "variant_metric_scene_rows.csv"),
            "variant_gate_rows": _rel(OUT_DIR / "variant_gate_rows.csv"),
            "variant_failure_rows": _rel(OUT_DIR / "variant_failure_rows.csv"),
            "variant_config_rows": _rel(OUT_DIR / "variant_config_rows.csv"),
            "casebook_rows": _rel(OUT_DIR / "casebook_rows.csv"),
            "frame_eval_rows": _rel(OUT_DIR / "frame_eval_rows.csv"),
            "d4rt_overlap_contract_rows": _rel(OUT_DIR / "d4rt_overlap_contract_rows.csv"),
            "d4rt_provider_root_rows": _rel(OUT_DIR / "d4rt_provider_root_rows.csv"),
        },
    }

    _write_csv(OUT_DIR / "d4rt_query_reliability_rows.csv", query_metric_rows)
    _write_csv(OUT_DIR / "d4rt_reliable_anchor_rows.csv", carrier_rows)
    _write_csv(OUT_DIR / "link_metric_rows.csv", link_rows)
    _write_csv(OUT_DIR / "local2history_candidate_rows.csv", candidate_rows)
    _write_csv(OUT_DIR / "local2history_merge_rows.csv", merge_rows)
    _write_csv(OUT_DIR / "variant_metric_rows.csv", metric_rows)
    _write_csv(OUT_DIR / "same_identity_metric_rows.csv", same_identity_metric_rows)
    _write_csv(OUT_DIR / "variant_metric_scene_rows.csv", scene_metric_rows)
    _write_csv(OUT_DIR / "variant_gate_rows.csv", gate_rows)
    _write_csv(OUT_DIR / "variant_failure_rows.csv", failure_rows)
    _write_csv(OUT_DIR / "variant_config_rows.csv", config_rows)
    _write_csv(OUT_DIR / "casebook_rows.csv", casebook_rows)
    _write_csv(OUT_DIR / "frame_eval_rows.csv", frame_rows)
    _write_csv(OUT_DIR / "d4rt_overlap_contract_rows.csv", contract.get("overlap_rows", []))
    _write_csv(OUT_DIR / "d4rt_provider_root_rows.csv", contract.get("root_rows", []))
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if metric_gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
