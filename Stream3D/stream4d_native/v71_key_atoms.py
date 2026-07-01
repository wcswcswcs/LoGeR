from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_v65_scene_multiview_ap import _load_gt_2d, _read_label_png, _sha256, _write_csv, _write_json  # noqa: E402
from tools.run_v66_local_chunk_eval import _rel  # noqa: E402
from tools.run_v66_scene_mv_ap_probe5 import DEFAULT_SCENES, _mask_dir_from_pipeline, _parse_csv_list  # noqa: E402


KEY_ATOM_FIELDS = [
    "key_atom_variant",
    "selection_rank",
    "scene_id",
    "chunk_id",
    "key_atom_id",
    "source_atom_id",
    "source_atom_type",
    "source_kind",
    "frame_id",
    "mask_id",
    "mask_observation_id",
    "semantic_prototype_id",
    "semantic_backend",
    "semantic_entropy",
    "D4RT_reliability_available",
    "D4RT_reliability",
    "D4RT_visible_frame_count",
    "D4RT_frame_span",
    "D4RT_motion_magnitude",
    "D4RT_uv_x",
    "D4RT_uv_y",
    "mask_membership_entropy",
    "background_or_plane_proxy",
    "selection_weight",
    "selection_distance_mode",
    "semantic_only_repair_atom",
    "uses_gt_for_prediction",
    "diagnostic_only",
    "forbidden_for_method_table",
]


VARIANT_FIELDS = [
    "key_atom_variant",
    "available",
    "availability_reason",
    "selection_distance_mode",
    "key_atom_count",
    "key_atom_count_per_chunk_mean",
    "key_atom_count_per_frame_mean",
    "key_atom_D4RT_reliability_mean",
    "key_atom_D4RT_reliability_coverage_rate",
    "key_atom_visible_frame_count_mean",
    "key_atom_semantic_prototype_count",
    "key_atom_semantic_entropy",
    "key_atom_frame_coverage_rate",
    "key_atom_frame_coverage_denominator",
    "key_atom_D4RT_spatial_coverage_radius_mean",
    "key_atom_semantic_coverage_radius_mean",
    "key_atom_trajectory_coverage_radius_mean",
    "key_atom_background_or_plane_proxy_rate",
    "key_atom_mask_membership_entropy_mean",
    "key_atom_diagnostic_GT_instance_coverage_count",
    "key_atom_diagnostic_GT_instance_coverage_ratio",
    "diagnostic_GT_instance_total_count",
    "uses_gt_for_prediction",
    "semantic_only_repair_atom_count",
    "D4RT_atom_count",
    "candidate_semantic_prototype_count_raw",
    "atom_universe_semantic_prototype_count",
    "gate_count_per_chunk_ge_2x_diagnostic_gt_mean",
    "gate_frame_coverage_ge_0p70",
    "gate_prototype_ge_0p5_candidate_raw",
    "gate_prototype_ge_0p5_atom_universe",
    "gate_D4RT_reliability_mean_ge_atom_mean",
    "gate_background_proxy_improved_ge_0p10_if_geo_semantic",
    "gate_diagnostic_GT_coverage_not_regress_baseline_minus_0p05",
    "gate_uses_gt_for_prediction_false",
    "gate_pass_raw_candidate_prototype_denominator",
    "gate_pass_atom_universe_denominator",
]


@dataclass(slots=True)
class AtomItem:
    scene: str
    chunk_id: str
    source_id: str
    source_type: str
    source_kind: str
    frame_id: int
    mask_id: int
    mask_observation_id: str
    semantic_prototype_id: str
    semantic_backend: str
    semantic_entropy: float
    d4rt_reliability: float | None
    visible_frame_count: float
    visible_frame_ratio: float
    frame_span: float
    motion_magnitude: float
    uv_x: float
    uv_y: float
    mask_membership_entropy: float
    background_proxy: bool
    semantic_only_repair_atom: bool
    stable_hash: int
    weight: float = 1.0
    feature_cache: dict[str, np.ndarray] = field(default_factory=dict)


def _rooted(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float(value: Any, default: float | None = None) -> float | None:
    if value is None or str(value).strip() == "":
        return default
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _mean(values: list[float | None]) -> float | None:
    valid = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(sum(valid) / len(valid)) if valid else None


def _percentile(values: list[float | None], q: float) -> float | None:
    valid = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not valid:
        return None
    idx = int(round((len(valid) - 1) * q))
    idx = max(0, min(len(valid) - 1, idx))
    return float(valid[idx])


def _entropy(counter: Counter[Any]) -> float:
    total = float(sum(counter.values()))
    if total <= 0.0 or len(counter) <= 1:
        return 0.0
    value = 0.0
    for count in counter.values():
        p = float(count) / total
        value -= p * math.log(max(p, 1e-12))
    return float(value / max(math.log(len(counter)), 1e-12))


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _parse_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return {}


def _parse_mask_observation_id(text: Any) -> tuple[str, int, int] | None:
    parts = str(text or "").split(":")
    if len(parts) != 3:
        return None
    try:
        return parts[0], int(parts[1]), int(parts[2])
    except Exception:
        return None


def _chunk_num(chunk_id: str) -> int:
    text = str(chunk_id)
    if ":chunk" in text:
        text = text.rsplit(":chunk", 1)[-1]
    return _int(text, 0)


def _load_pipeline_roots(path: Path, scenes: list[str]) -> dict[str, Path]:
    summary = _load_json(path)
    raw = summary.get("pipeline_roots") or {}
    out: dict[str, Path] = {}
    for scene in scenes:
        value = raw.get(scene)
        if value:
            out[scene] = _rooted(value)
    return out


def _load_atom_items(path: Path, scenes: set[str] | None, max_rows: int | None) -> tuple[list[AtomItem], Counter[str]]:
    items: list[AtomItem] = []
    source_type_counts: Counter[str] = Counter()
    if not path.exists():
        return items, source_type_counts
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if scenes and row.get("scene_id") not in scenes:
                continue
            if row.get("atom_type") != "A0_single_carrier_atom":
                source_type_counts[row.get("atom_type") or ""] += 1
                continue
            obs = _parse_mask_observation_id(row.get("atom_mask_observation_id"))
            if obs is None:
                continue
            pos = _parse_json_dict(row.get("D4RT_position_mean"))
            uv_x = _float(pos.get("uv_x"), 0.5) or 0.5
            uv_y = _float(pos.get("uv_y"), 0.5) or 0.5
            sem_entropy = _float(row.get("semantic_entropy_mean"), 1.0) or 1.0
            mask_entropy = _float(row.get("mask_membership_entropy"), 1.0) or 1.0
            reliability = _float(row.get("non_gt_reliability_score"), None)
            atom = AtomItem(
                scene=str(row.get("scene_id") or obs[0]),
                chunk_id=str(row.get("chunk_id") or ""),
                source_id=str(row.get("atom_id") or ""),
                source_type=str(row.get("atom_type") or "A0_single_carrier_atom"),
                source_kind="D4RT_atom",
                frame_id=obs[1],
                mask_id=obs[2],
                mask_observation_id=f"{obs[0]}:{obs[1]}:{obs[2]}",
                semantic_prototype_id=str(row.get("semantic_prototype_id") or ""),
                semantic_backend=str(row.get("semantic_backend") or ""),
                semantic_entropy=sem_entropy,
                d4rt_reliability=reliability,
                visible_frame_count=_float(row.get("visible_frame_count"), 0.0) or 0.0,
                visible_frame_ratio=_float(row.get("visible_frame_ratio"), 0.0) or 0.0,
                frame_span=_float(row.get("frame_span"), 0.0) or 0.0,
                motion_magnitude=_float(row.get("D4RT_motion_magnitude"), 0.0) or 0.0,
                uv_x=float(uv_x),
                uv_y=float(uv_y),
                mask_membership_entropy=mask_entropy,
                background_proxy=bool(mask_entropy >= 0.98 or sem_entropy >= 0.72),
                semantic_only_repair_atom=False,
                stable_hash=_stable_int(str(row.get("atom_id") or "")),
            )
            items.append(atom)
            source_type_counts[atom.source_type] += 1
            if max_rows and len(items) >= max_rows:
                break
    return items, source_type_counts


def _load_candidate_items(path: Path, scenes: set[str] | None, max_rows: int | None) -> tuple[list[AtomItem], Counter[str], set[tuple[str, int]]]:
    items: list[AtomItem] = []
    prototype_counter: Counter[str] = Counter()
    frame_universe: set[tuple[str, int]] = set()
    if not path.exists():
        return items, prototype_counter, frame_universe
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scene = str(row.get("scene_id") or "")
            if scenes and scene not in scenes:
                continue
            if not _bool(row.get("semantic_feature_available")):
                continue
            frame_id = _int(row.get("frame_id"), -1)
            mask_id = _int(row.get("mask_id"), -1)
            if frame_id < 0 or mask_id < 0:
                continue
            frame_universe.add((scene, frame_id))
            proto = str(row.get("semantic_prototype_id") or "")
            prototype_counter[proto] += 1
            x0 = _float(row.get("bbox_x0"), 0.0) or 0.0
            y0 = _float(row.get("bbox_y0"), 0.0) or 0.0
            x1 = _float(row.get("bbox_x1"), x0) or x0
            y1 = _float(row.get("bbox_y1"), y0) or y0
            # ScanNet mask export is 640x480 in this pipeline; bbox ratios are
            # used only as chunk-local descriptors, not physical geometry.
            uv_x = max(0.0, min(1.0, (x0 + x1) * 0.5 / 640.0))
            uv_y = max(0.0, min(1.0, (y0 + y1) * 0.5 / 480.0))
            sem_entropy = _float(row.get("semantic_entropy"), 1.0) or 1.0
            broad = _bool(row.get("broad_background_risk"))
            large = _bool(row.get("large_mask_risk"))
            underseg = (_float(row.get("underseg_proxy_score"), 0.0) or 0.0) >= 0.75
            atom = AtomItem(
                scene=scene,
                chunk_id=str(row.get("chunk_id") or ""),
                source_id=str(row.get("mask_observation_id") or f"{scene}:{frame_id}:{mask_id}"),
                source_type="A2_semantic_candidate_mask_atom",
                source_kind="semantic_candidate_mask",
                frame_id=frame_id,
                mask_id=mask_id,
                mask_observation_id=str(row.get("mask_observation_id") or f"{scene}:{frame_id}:{mask_id}"),
                semantic_prototype_id=proto,
                semantic_backend=str(row.get("semantic_backend") or ""),
                semantic_entropy=sem_entropy,
                d4rt_reliability=None,
                visible_frame_count=1.0,
                visible_frame_ratio=1.0,
                frame_span=0.0,
                motion_magnitude=0.0,
                uv_x=uv_x,
                uv_y=uv_y,
                mask_membership_entropy=0.0,
                background_proxy=bool(broad or large or underseg or sem_entropy >= 0.72),
                semantic_only_repair_atom=True,
                stable_hash=_stable_int(str(row.get("mask_observation_id") or "")),
            )
            items.append(atom)
            if max_rows and len(items) >= max_rows:
                break
    return items, prototype_counter, frame_universe


def _proto_hash_vector(proto: str, dims: int = 64) -> np.ndarray:
    vec = np.zeros((dims,), dtype=np.float32)
    for token in str(proto or "").split("|"):
        token = token.strip()
        if not token:
            continue
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dims
        sign = 1.0 if token[:1] == "p" else -1.0 if token[:1] == "n" else 0.5
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 1e-9:
        vec /= norm
    return vec


def _feature_matrix(items: list[AtomItem], mode: str) -> np.ndarray:
    rows: list[np.ndarray] = []
    frames = [item.frame_id for item in items]
    frame_min = float(min(frames)) if frames else 0.0
    frame_span = float(max(frames) - min(frames)) if frames and max(frames) > min(frames) else 1.0
    for item in items:
        parts: list[np.ndarray] = []
        if "geo" in mode:
            parts.append(np.asarray([item.uv_x, item.uv_y], dtype=np.float32))
        if "traj" in mode:
            parts.append(
                np.asarray(
                    [
                        min(1.0, item.motion_magnitude * 40.0),
                        min(1.0, item.visible_frame_count / 32.0),
                        min(1.0, item.frame_span / 160.0),
                    ],
                    dtype=np.float32,
                )
            )
        if "sem" in mode:
            sem = _proto_hash_vector(item.semantic_prototype_id)
            parts.append(sem)
            parts.append(np.asarray([min(1.0, item.semantic_entropy)], dtype=np.float32))
        if "vis" in mode:
            parts.append(np.asarray([min(1.0, item.visible_frame_ratio), min(1.0, item.visible_frame_count / 32.0)], dtype=np.float32))
        if "frame" in mode:
            parts.append(np.asarray([(float(item.frame_id) - frame_min) / frame_span], dtype=np.float32))
        rows.append(np.concatenate(parts) if parts else np.zeros((1,), dtype=np.float32))
    if not rows:
        return np.zeros((0, 1), dtype=np.float32)
    matrix = np.vstack(rows).astype(np.float32, copy=False)
    if matrix.shape[0] > 1:
        mins = np.nanmin(matrix, axis=0)
        maxs = np.nanmax(matrix, axis=0)
        scale = np.where((maxs - mins) > 1e-9, maxs - mins, 1.0)
        matrix = (matrix - mins) / scale
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=0.0)
    return matrix


def _sample_items(items: list[AtomItem], limit: int, salt: str) -> list[AtomItem]:
    if len(items) <= limit:
        return list(items)
    return sorted(items, key=lambda item: _stable_int(f"{salt}:{item.source_id}:{item.mask_observation_id}"))[:limit]


def _coverage_radius(items: list[AtomItem], selected: list[AtomItem], mode: str) -> float | None:
    if not items or not selected:
        return None
    # Full all-universe/all-selected distance is prohibitively large for v71
    # (hundreds of thousands of atom-like rows times tens of thousands of keys).
    # This deterministic sample preserves comparability across variants without
    # affecting the key atom selection itself.
    items = _sample_items(items, 4096, f"coverage-universe-{mode}")
    selected = _sample_items(selected, 1024, f"coverage-selected-{mode}")
    index = {id(item): idx for idx, item in enumerate(items)}
    selected_indices = [index[id(item)] for item in selected if id(item) in index]
    if not selected_indices:
        # Selected rows may be absent from the universe sample; include a
        # deterministic selected sample for the diagnostic distance anchors.
        items = _sample_items(items + selected, 4096, f"coverage-union-{mode}")
        index = {id(item): idx for idx, item in enumerate(items)}
        selected_indices = [index[id(item)] for item in selected if id(item) in index]
    if not selected_indices:
        return None
    matrix = _feature_matrix(items, mode)
    selected_matrix = matrix[selected_indices]
    best = np.full((matrix.shape[0],), np.inf, dtype=np.float32)
    batch = 64
    for start in range(0, selected_matrix.shape[0], batch):
        chunk = selected_matrix[start : start + batch]
        diff = matrix[:, None, :] - chunk[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        best = np.minimum(best, dist.min(axis=1))
    return float(np.mean(best))


def _base_weight(item: AtomItem, proto_counts: Counter[str], density_counts: Counter[tuple[str, int, int]]) -> float:
    if item.d4rt_reliability is not None:
        base = max(0.0, min(1.0, float(item.d4rt_reliability)))
    else:
        base = 0.35 + 0.45 * (1.0 - max(0.0, min(1.0, item.semantic_entropy)))
    proto_count = max(1, proto_counts.get(item.semantic_prototype_id, 1))
    rarity = max(0.75, min(3.0, math.sqrt(max(1.0, len(proto_counts)) / proto_count)))
    visibility = 1.0 + 0.30 * min(1.0, item.visible_frame_count / 16.0)
    bin_key = (item.chunk_id, int(item.uv_x * 6.0), int(item.uv_y * 6.0))
    density = 1.0 / math.sqrt(max(1, density_counts.get(bin_key, 1)))
    background = 0.35 if item.background_proxy else 1.0
    return float(base * rarity * visibility * density * background)


def _assign_weights(items: list[AtomItem]) -> None:
    proto_counts = Counter(item.semantic_prototype_id for item in items)
    density_counts = Counter((item.chunk_id, int(item.uv_x * 6.0), int(item.uv_y * 6.0)) for item in items)
    for item in items:
        item.weight = _base_weight(item, proto_counts, density_counts)


def _fps_select(
    items: list[AtomItem],
    k: int,
    mode: str,
    weighted: bool,
    preselected: list[AtomItem] | None = None,
) -> list[AtomItem]:
    if k <= 0 or not items:
        return []
    if len(items) <= k:
        return list(items)
    matrix = _feature_matrix(items, mode)
    weights = np.asarray([max(1e-6, item.weight) for item in items], dtype=np.float32)
    index_by_id = {id(item): idx for idx, item in enumerate(items)}
    selected: list[int] = []
    selected_set: set[int] = set()
    min_dist = np.full((len(items),), np.inf, dtype=np.float32)

    def add_index(idx: int) -> None:
        if idx in selected_set:
            return
        selected.append(idx)
        selected_set.add(idx)
        diff = matrix - matrix[idx]
        dist = np.sum(diff * diff, axis=1)
        np.minimum(min_dist, dist, out=min_dist)
        min_dist[list(selected_set)] = -np.inf

    for item in preselected or []:
        idx = index_by_id.get(id(item))
        if idx is not None:
            add_index(idx)
            if len(selected) >= k:
                break
    if not selected:
        first = int(np.argmax(weights if weighted else np.asarray([-(item.stable_hash % 10**12) for item in items], dtype=np.float64)))
        add_index(first)
    while len(selected) < k:
        if weighted:
            scores = min_dist * (0.25 + weights) + 0.01 * weights
        else:
            scores = min_dist
        idx = int(np.argmax(scores))
        if not math.isfinite(float(scores[idx])) or idx in selected_set:
            remaining = [i for i in range(len(items)) if i not in selected_set]
            if not remaining:
                break
            idx = max(remaining, key=lambda i: items[i].weight if weighted else -items[i].stable_hash)
        add_index(idx)
    return [items[idx] for idx in selected[:k]]


def _balanced_preselect(items: list[AtomItem], k: int, *, frame_quota: bool, prototype_quota: bool, reliability_first: bool) -> list[AtomItem]:
    if not items or k <= 0:
        return []
    selected: list[AtomItem] = []
    used: set[int] = set()

    def score(item: AtomItem) -> tuple[float, float, int]:
        rel = item.d4rt_reliability if item.d4rt_reliability is not None else 0.0
        primary = item.weight + (0.25 * rel if reliability_first else 0.0) - (0.35 if item.background_proxy else 0.0)
        return (primary, -item.semantic_entropy, -item.stable_hash)

    def add(item: AtomItem) -> None:
        if id(item) in used or len(selected) >= k:
            return
        selected.append(item)
        used.add(id(item))

    if frame_quota:
        by_frame: dict[int, list[AtomItem]] = defaultdict(list)
        for item in items:
            by_frame[item.frame_id].append(item)
        frame_cap = min(k, max(1, int(round(k * 0.45))))
        for frame_id in sorted(by_frame, key=lambda f: (-len(by_frame[f]), f)):
            add(max(by_frame[frame_id], key=score))
            if len(selected) >= frame_cap:
                break
    if prototype_quota and len(selected) < k:
        by_proto: dict[str, list[AtomItem]] = defaultdict(list)
        for item in items:
            if id(item) not in used:
                by_proto[item.semantic_prototype_id].append(item)
        for proto in sorted(by_proto, key=lambda p: (len(by_proto[p]), p)):
            add(max(by_proto[proto], key=score))
            if len(selected) >= k:
                break
    return selected


def _select_by_chunk(items_by_chunk: dict[str, list[AtomItem]], k: int, mode: str, variant: str) -> list[AtomItem]:
    selected: list[AtomItem] = []
    for chunk_id in sorted(items_by_chunk, key=lambda x: (x.split(":")[0], _chunk_num(x))):
        items = items_by_chunk[chunk_id]
        if variant == "K0_random_atoms":
            chosen = sorted(items, key=lambda item: item.stable_hash)[:k]
        elif variant == "K7_geo_semantic_balanced_kcenter":
            pre = _balanced_preselect(items, k, frame_quota=True, prototype_quota=True, reliability_first=True)
            chosen = _fps_select(items, k, mode, weighted=True, preselected=pre)
        elif variant == "K8_reliability_weighted_geo_semantic_FPS":
            pre = _balanced_preselect(items, k, frame_quota=True, prototype_quota=True, reliability_first=True)
            chosen = _fps_select(items, k, mode, weighted=True, preselected=pre)
        elif variant in {"K3_semantic_FPS_DINO", "K5_geo_semantic_FPS_D4RT_DINO"}:
            pre = _balanced_preselect(items, k, frame_quota=True, prototype_quota=True, reliability_first=(variant.startswith("K5")))
            chosen = _fps_select(items, k, mode, weighted=variant.startswith("K5"), preselected=pre)
        else:
            chosen = _fps_select(items, k, mode, weighted=False)
        selected.extend(chosen)
    return selected


def _dedupe_selected(items: list[AtomItem]) -> list[AtomItem]:
    best: dict[str, AtomItem] = {}
    order: list[str] = []
    for item in items:
        key = item.mask_observation_id or item.source_id
        if key not in best:
            best[key] = item
            order.append(key)
            continue
        prev = best[key]
        prev_rel = prev.d4rt_reliability if prev.d4rt_reliability is not None else -1.0
        rel = item.d4rt_reliability if item.d4rt_reliability is not None else -1.0
        if rel > prev_rel or (rel == prev_rel and item.weight > prev.weight):
            best[key] = item
    return [best[key] for key in order]


def _make_variant_inputs(
    variant: str,
    d4rt_by_chunk: dict[str, list[AtomItem]],
    semantic_by_chunk: dict[str, list[AtomItem]],
    args: argparse.Namespace,
) -> tuple[dict[str, list[AtomItem]], int, str, str, bool, str]:
    if variant == "K0_random_atoms":
        return d4rt_by_chunk, args.d4rt_key_atoms_per_chunk, "stable_random", "available", True, "random"
    if variant == "K1_D4RT_spatial_FPS":
        return d4rt_by_chunk, args.d4rt_key_atoms_per_chunk, "geo", "available", True, "geo"
    if variant == "K2_D4RT_trajectory_FPS":
        return d4rt_by_chunk, args.d4rt_key_atoms_per_chunk, "traj_vis_frame", "available", True, "traj+vis+frame"
    if variant == "K3_semantic_FPS_DINO":
        return semantic_by_chunk, args.key_atoms_per_chunk, "sem_frame", "available", True, "sem+frame"
    if variant == "K4_semantic_FPS_RADIO":
        return {}, 0, "sem_radio", "RADIO features unavailable in Phase3", False, "sem_radio"
    if variant == "K5_geo_semantic_FPS_D4RT_DINO":
        merged = _merge_variant_chunks(d4rt_by_chunk, semantic_by_chunk, args)
        return merged, args.key_atoms_per_chunk, "geo_traj_sem_vis_frame", "available", True, "geo+traj+sem+vis+frame"
    if variant == "K6_geo_semantic_FPS_D4RT_RADIO":
        return {}, 0, "geo_sem_radio", "RADIO features unavailable in Phase3", False, "geo+sem_radio"
    if variant == "K7_geo_semantic_balanced_kcenter":
        merged = _merge_variant_chunks(d4rt_by_chunk, semantic_by_chunk, args)
        return merged, args.key_atoms_per_chunk, "geo_traj_sem_vis_frame", "available", True, "balanced_geo+sem"
    if variant == "K8_reliability_weighted_geo_semantic_FPS":
        merged = _merge_variant_chunks(d4rt_by_chunk, semantic_by_chunk, args)
        return merged, args.key_atoms_per_chunk, "geo_traj_sem_vis_frame", "available", True, "weighted_geo+sem"
    return {}, 0, "", "unknown variant", False, ""


def _merge_variant_chunks(
    d4rt_by_chunk: dict[str, list[AtomItem]],
    semantic_by_chunk: dict[str, list[AtomItem]],
    args: argparse.Namespace,
) -> dict[str, list[AtomItem]]:
    out: dict[str, list[AtomItem]] = {}
    all_chunks = sorted(set(d4rt_by_chunk) | (set(semantic_by_chunk) if args.semantic_only_repair else set()))
    for chunk_id in all_chunks:
        items: list[AtomItem] = []
        d4rt = sorted(d4rt_by_chunk.get(chunk_id, []), key=lambda item: (-item.weight, item.stable_hash))
        sem = semantic_by_chunk.get(chunk_id, []) if args.semantic_only_repair else []
        items.extend(d4rt[: args.max_d4rt_candidates_per_chunk])
        items.extend(sem)
        out[chunk_id] = items
    return out


def _build_key_rows(variant: str, selected: list[AtomItem], mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, item in enumerate(selected):
        rows.append(
            {
                "key_atom_variant": variant,
                "selection_rank": rank,
                "scene_id": item.scene,
                "chunk_id": item.chunk_id,
                "key_atom_id": f"{variant}:{rank:08d}",
                "source_atom_id": item.source_id,
                "source_atom_type": item.source_type,
                "source_kind": item.source_kind,
                "frame_id": int(item.frame_id),
                "mask_id": int(item.mask_id),
                "mask_observation_id": item.mask_observation_id,
                "semantic_prototype_id": item.semantic_prototype_id,
                "semantic_backend": item.semantic_backend,
                "semantic_entropy": item.semantic_entropy,
                "D4RT_reliability_available": item.d4rt_reliability is not None,
                "D4RT_reliability": item.d4rt_reliability,
                "D4RT_visible_frame_count": item.visible_frame_count,
                "D4RT_frame_span": item.frame_span,
                "D4RT_motion_magnitude": item.motion_magnitude,
                "D4RT_uv_x": item.uv_x,
                "D4RT_uv_y": item.uv_y,
                "mask_membership_entropy": item.mask_membership_entropy,
                "background_or_plane_proxy": item.background_proxy,
                "selection_weight": item.weight,
                "selection_distance_mode": mode,
                "semantic_only_repair_atom": item.semantic_only_repair_atom,
                "uses_gt_for_prediction": False,
                "diagnostic_only": False,
                "forbidden_for_method_table": False,
            }
        )
    return rows


def _diagnostic_gt_coverage(
    *,
    selected_by_variant: dict[str, list[AtomItem]],
    pipeline_roots: dict[str, Path],
    frame_universe: set[tuple[str, int]],
    max_frames: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if not pipeline_roots or not frame_universe:
        return {}, [{"diagnostic_status": "missing_pipeline_roots_or_frame_universe"}]
    missing_rows: list[dict[str, Any]] = []
    mask_dirs: dict[str, Path] = {}
    for scene, root in pipeline_roots.items():
        try:
            mask_dirs[scene] = _mask_dir_from_pipeline(root)
        except Exception as exc:
            missing_rows.append({"scene_id": scene, "diagnostic_status": "missing_mask_dir", "detail": str(exc)})
    if not mask_dirs:
        return {}, missing_rows

    frames = sorted(frame_universe)
    if max_frames > 0:
        frames = frames[:max_frames]
    gt_cache: dict[tuple[str, int], np.ndarray] = {}
    pred_cache: dict[tuple[str, int], np.ndarray] = {}
    all_gt: set[tuple[str, int]] = set()
    for scene, frame_id in frames:
        mask_dir = mask_dirs.get(scene)
        if mask_dir is None:
            continue
        pred_path = mask_dir / f"{int(frame_id)}.png"
        if not pred_path.exists():
            missing_rows.append({"scene_id": scene, "frame_id": frame_id, "diagnostic_status": "missing_pred_mask", "path": _rel(pred_path)})
            continue
        try:
            pred = _read_label_png(pred_path, (480, 640))
            gt = _load_gt_2d(scene, frame_id, pred.shape[:2])
        except Exception as exc:
            missing_rows.append({"scene_id": scene, "frame_id": frame_id, "diagnostic_status": "load_failed", "detail": str(exc)})
            continue
        pred_cache[(scene, frame_id)] = pred
        gt_cache[(scene, frame_id)] = gt
        for gid in np.unique(gt):
            gid_int = int(gid)
            if gid_int > 0:
                all_gt.add((scene, gid_int))

    result: dict[str, dict[str, Any]] = {}
    total = len(all_gt)
    for variant, selected in selected_by_variant.items():
        covered: set[tuple[str, int]] = set()
        evaluated_masks = 0
        for item in selected:
            key = (item.scene, item.frame_id)
            pred = pred_cache.get(key)
            gt = gt_cache.get(key)
            if pred is None or gt is None:
                continue
            pixels = pred == int(item.mask_id)
            if not np.any(pixels):
                continue
            positive = gt[pixels]
            positive = positive[positive > 0]
            if positive.size == 0:
                evaluated_masks += 1
                continue
            values, counts = np.unique(positive, return_counts=True)
            gid = int(values[int(np.argmax(counts))])
            covered.add((item.scene, gid))
            evaluated_masks += 1
        ratio = float(len(covered) / total) if total > 0 else None
        result[variant] = {
            "key_atom_diagnostic_GT_instance_coverage_count": int(len(covered)),
            "key_atom_diagnostic_GT_instance_coverage_ratio": ratio,
            "diagnostic_GT_instance_total_count": int(total),
            "diagnostic_evaluated_mask_count": int(evaluated_masks),
            "diagnostic_frame_count": int(len(gt_cache)),
            "diagnostic_only": True,
            "diagnostic_gt_used_for_selection": False,
        }
    return result, missing_rows


def _summarize_variant(
    *,
    variant: str,
    selected: list[AtomItem],
    all_items_for_radius: list[AtomItem],
    available: bool,
    availability_reason: str,
    mode: str,
    candidate_frame_count: int,
    candidate_proto_count: int,
    atom_proto_count: int,
    atom_reliability_mean: float | None,
    diagnostic_gt_mean: float | None,
    k1_background_rate: float | None,
    diagnostic_gt: dict[str, Any] | None,
    baseline_diag_ratio: float | None,
) -> dict[str, Any]:
    chunks = {item.chunk_id for item in selected}
    frames = {(item.scene, item.frame_id) for item in selected}
    rel_values = [item.d4rt_reliability for item in selected if item.d4rt_reliability is not None]
    visible_values = [item.visible_frame_count for item in selected if item.d4rt_reliability is not None]
    protos = [item.semantic_prototype_id for item in selected if item.semantic_prototype_id]
    background_rate = _mean([1.0 if item.background_proxy else 0.0 for item in selected])
    d4rt_count = sum(1 for item in selected if item.d4rt_reliability is not None)
    semantic_only_count = sum(1 for item in selected if item.semantic_only_repair_atom)
    count_per_chunk = float(len(selected) / len(chunks)) if chunks else 0.0
    count_per_frame = float(len(selected) / len(frames)) if frames else 0.0
    frame_den = max(1, candidate_frame_count)
    frame_cov = float(len(frames) / frame_den)
    rel_mean = _mean(rel_values)
    gt = diagnostic_gt or {}
    diag_ratio = gt.get("key_atom_diagnostic_GT_instance_coverage_ratio")
    is_geo_sem = variant in {
        "K5_geo_semantic_FPS_D4RT_DINO",
        "K6_geo_semantic_FPS_D4RT_RADIO",
        "K7_geo_semantic_balanced_kcenter",
        "K8_reliability_weighted_geo_semantic_FPS",
    }
    gt_mean_threshold = 2.0 * float(diagnostic_gt_mean or 0.0)
    raw_proto_threshold = 0.5 * float(candidate_proto_count)
    atom_proto_threshold = 0.5 * float(atom_proto_count)
    gate_count = count_per_chunk >= gt_mean_threshold if available else False
    gate_frame = frame_cov >= 0.70 if available else False
    gate_proto_raw = len(set(protos)) >= raw_proto_threshold if available else False
    gate_proto_atom = len(set(protos)) >= atom_proto_threshold if available else False
    gate_rel = rel_mean is not None and atom_reliability_mean is not None and rel_mean >= atom_reliability_mean
    if not is_geo_sem or k1_background_rate is None or background_rate is None:
        gate_background = True if available else False
    else:
        gate_background = background_rate <= (float(k1_background_rate) - 0.10)
    if not is_geo_sem or diag_ratio is None or baseline_diag_ratio is None:
        gate_diag = True if available else False
    else:
        gate_diag = float(diag_ratio) >= (float(baseline_diag_ratio) - 0.05)
    gate_uses_gt = True
    gate_pass_raw = bool(available and gate_count and gate_frame and gate_proto_raw and gate_rel and gate_background and gate_diag and gate_uses_gt)
    gate_pass_atom = bool(available and gate_count and gate_frame and gate_proto_atom and gate_rel and gate_background and gate_diag and gate_uses_gt)
    return {
        "key_atom_variant": variant,
        "available": bool(available),
        "availability_reason": availability_reason,
        "selection_distance_mode": mode,
        "key_atom_count": int(len(selected)),
        "key_atom_count_per_chunk_mean": count_per_chunk,
        "key_atom_count_per_frame_mean": count_per_frame,
        "key_atom_D4RT_reliability_mean": rel_mean,
        "key_atom_D4RT_reliability_coverage_rate": float(d4rt_count / len(selected)) if selected else 0.0,
        "key_atom_visible_frame_count_mean": _mean(visible_values),
        "key_atom_semantic_prototype_count": int(len(set(protos))),
        "key_atom_semantic_entropy": _entropy(Counter(protos)),
        "key_atom_frame_coverage_rate": frame_cov,
        "key_atom_frame_coverage_denominator": int(frame_den),
        "key_atom_D4RT_spatial_coverage_radius_mean": _coverage_radius(all_items_for_radius, selected, "geo"),
        "key_atom_semantic_coverage_radius_mean": _coverage_radius(all_items_for_radius, selected, "sem"),
        "key_atom_trajectory_coverage_radius_mean": _coverage_radius(all_items_for_radius, selected, "traj_vis_frame"),
        "key_atom_background_or_plane_proxy_rate": background_rate,
        "key_atom_mask_membership_entropy_mean": _mean([item.mask_membership_entropy for item in selected]),
        "key_atom_diagnostic_GT_instance_coverage_count": gt.get("key_atom_diagnostic_GT_instance_coverage_count"),
        "key_atom_diagnostic_GT_instance_coverage_ratio": diag_ratio,
        "diagnostic_GT_instance_total_count": gt.get("diagnostic_GT_instance_total_count"),
        "uses_gt_for_prediction": False,
        "semantic_only_repair_atom_count": int(semantic_only_count),
        "D4RT_atom_count": int(d4rt_count),
        "candidate_semantic_prototype_count_raw": int(candidate_proto_count),
        "atom_universe_semantic_prototype_count": int(atom_proto_count),
        "gate_count_per_chunk_ge_2x_diagnostic_gt_mean": bool(gate_count),
        "gate_frame_coverage_ge_0p70": bool(gate_frame),
        "gate_prototype_ge_0p5_candidate_raw": bool(gate_proto_raw),
        "gate_prototype_ge_0p5_atom_universe": bool(gate_proto_atom),
        "gate_D4RT_reliability_mean_ge_atom_mean": bool(gate_rel),
        "gate_background_proxy_improved_ge_0p10_if_geo_semantic": bool(gate_background),
        "gate_diagnostic_GT_coverage_not_regress_baseline_minus_0p05": bool(gate_diag),
        "gate_uses_gt_for_prediction_false": bool(gate_uses_gt),
        "gate_pass_raw_candidate_prototype_denominator": gate_pass_raw,
        "gate_pass_atom_universe_denominator": gate_pass_atom,
    }


def _bar_plot(path: Path, title: str, labels: list[str], values: list[float | None], ymax: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = max(900, 110 * max(1, len(labels)))
    height = 420
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(image, title, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    valid = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    top = ymax if ymax is not None else (max(valid) * 1.15 if valid else 1.0)
    top = max(float(top), 1e-6)
    plot_x0, plot_y0 = 50, 70
    plot_w, plot_h = width - 90, height - 140
    cv2.rectangle(image, (plot_x0, plot_y0), (plot_x0 + plot_w, plot_y0 + plot_h), (220, 220, 220), 1)
    bar_w = max(10, int(plot_w / max(1, len(labels)) * 0.65))
    step = plot_w / max(1, len(labels))
    for idx, (label, value) in enumerate(zip(labels, values)):
        x = int(plot_x0 + idx * step + (step - bar_w) * 0.5)
        val = float(value) if value is not None and math.isfinite(float(value)) else 0.0
        bar_h = int(plot_h * max(0.0, min(1.0, val / top)))
        color = (60, 110, 210) if value is not None else (180, 180, 180)
        cv2.rectangle(image, (x, plot_y0 + plot_h - bar_h), (x + bar_w, plot_y0 + plot_h), color, -1)
        cv2.putText(image, f"{val:.3g}", (x, plot_y0 + plot_h - bar_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        short = label.split("_")[0]
        cv2.putText(image, short, (x, plot_y0 + plot_h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), image)


def _write_visualizations(visual_root: Path, summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = [str(row["key_atom_variant"]) for row in summaries]
    rows: list[dict[str, Any]] = []
    plots = [
        ("key_atom_reliability_mean.png", "D4RT reliability mean by key atom variant", "key_atom_D4RT_reliability_mean", 1.0),
        ("key_atom_background_proxy_rate.png", "Background/plane proxy rate by variant", "key_atom_background_or_plane_proxy_rate", 1.0),
        ("key_atom_frame_coverage_rate.png", "Frame coverage rate by variant", "key_atom_frame_coverage_rate", 1.0),
        ("key_atom_semantic_prototype_count.png", "Semantic prototype count by variant", "key_atom_semantic_prototype_count", None),
    ]
    for filename, title, field, ymax in plots:
        path = visual_root / filename
        _bar_plot(path, title, labels, [row.get(field) for row in summaries], ymax)
        rows.append({"path": _rel(path), "kind": "bar_plot", "field": field, "sha256": _sha256(path), "bytes": path.stat().st_size})
    return rows


def _write_sha_rows(output_root: Path, files: list[Path]) -> None:
    rows = []
    for path in files:
        if path.exists():
            rows.append({"path": _rel(path), "sha256": _sha256(path), "bytes": path.stat().st_size})
    _write_csv(output_root / "sha256_rows.csv", rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _rooted(args.output_root)
    visual_root = _rooted(args.visual_root)
    output_root.mkdir(parents=True, exist_ok=True)
    visual_root.mkdir(parents=True, exist_ok=True)

    scenes = _parse_csv_list(args.scenes) if args.scenes else list(DEFAULT_SCENES)
    scene_set = set(scenes)
    atom_root = _rooted(args.atom_root)
    candidate_root = _rooted(args.candidate_root)
    semantic_root = _rooted(args.semantic_root)

    atom_summary = _load_json(atom_root / "atom_summary.json")
    semantic_summary = _load_json(semantic_root / "semantic_summary.json")
    candidate_summary = _load_json(candidate_root / "candidate_bank_summary.json")
    missing_rows: list[dict[str, Any]] = []
    for required in [
        atom_root / "atom_rows.csv",
        atom_root / "atom_summary.json",
        candidate_root / "candidate_mask_rows.csv",
        candidate_root / "candidate_bank_summary.json",
        semantic_root / "semantic_summary.json",
    ]:
        if not required.exists():
            missing_rows.append({"path": _rel(required), "missing": True})

    d4rt_items, atom_source_counts = _load_atom_items(atom_root / "atom_rows.csv", scene_set, args.max_atom_rows)
    semantic_items, candidate_proto_counter, candidate_frame_universe = _load_candidate_items(
        candidate_root / "candidate_mask_rows.csv", scene_set, args.max_candidate_rows
    )
    if args.max_chunks > 0:
        keep_chunks = set(sorted({item.chunk_id for item in d4rt_items} | {item.chunk_id for item in semantic_items})[: args.max_chunks])
        d4rt_items = [item for item in d4rt_items if item.chunk_id in keep_chunks]
        semantic_items = [item for item in semantic_items if item.chunk_id in keep_chunks]
        candidate_frame_universe = {(item.scene, item.frame_id) for item in semantic_items}

    all_for_weights = list(d4rt_items) + (list(semantic_items) if args.semantic_only_repair else [])
    _assign_weights(all_for_weights)
    d4rt_by_chunk: dict[str, list[AtomItem]] = defaultdict(list)
    semantic_by_chunk: dict[str, list[AtomItem]] = defaultdict(list)
    for item in d4rt_items:
        d4rt_by_chunk[item.chunk_id].append(item)
    for item in semantic_items:
        semantic_by_chunk[item.chunk_id].append(item)

    variants = [
        "K0_random_atoms",
        "K1_D4RT_spatial_FPS",
        "K2_D4RT_trajectory_FPS",
        "K3_semantic_FPS_DINO",
        "K4_semantic_FPS_RADIO",
        "K5_geo_semantic_FPS_D4RT_DINO",
        "K6_geo_semantic_FPS_D4RT_RADIO",
        "K7_geo_semantic_balanced_kcenter",
        "K8_reliability_weighted_geo_semantic_FPS",
    ]
    selected_by_variant: dict[str, list[AtomItem]] = {}
    key_rows: list[dict[str, Any]] = []
    variant_modes: dict[str, str] = {}
    variant_available: dict[str, bool] = {}
    variant_reasons: dict[str, str] = {}

    for variant in variants:
        by_chunk, per_chunk, mode, reason, available, mode_label = _make_variant_inputs(variant, d4rt_by_chunk, semantic_by_chunk, args)
        variant_modes[variant] = mode_label
        variant_available[variant] = available
        variant_reasons[variant] = reason
        if not available:
            selected_by_variant[variant] = []
            continue
        selected = _select_by_chunk(by_chunk, per_chunk, mode, variant)
        selected = _dedupe_selected(selected)
        selected_by_variant[variant] = selected
        key_rows.extend(_build_key_rows(variant, selected, mode_label))

    pipeline_roots = _load_pipeline_roots(_rooted(args.witness_summary), scenes)
    if args.skip_diagnostic_gt:
        diagnostic_gt: dict[str, dict[str, Any]] = {}
        missing_rows.append({"diagnostic_status": "diagnostic_gt_skipped_by_cli"})
    else:
        diagnostic_gt, diagnostic_missing = _diagnostic_gt_coverage(
            selected_by_variant=selected_by_variant,
            pipeline_roots=pipeline_roots,
            frame_universe=candidate_frame_universe,
            max_frames=args.max_diagnostic_gt_frames,
        )
        missing_rows.extend(diagnostic_missing)

    k1_background_rate = None
    if selected_by_variant.get("K1_D4RT_spatial_FPS"):
        k1_background_rate = _mean([1.0 if item.background_proxy else 0.0 for item in selected_by_variant["K1_D4RT_spatial_FPS"]])
    baseline_ratios = [
        diagnostic_gt.get(name, {}).get("key_atom_diagnostic_GT_instance_coverage_ratio")
        for name in ["K1_D4RT_spatial_FPS", "K3_semantic_FPS_DINO", "K4_semantic_FPS_RADIO"]
    ]
    baseline_ratios_valid = [float(v) for v in baseline_ratios if v is not None]
    baseline_diag_ratio = max(baseline_ratios_valid) if baseline_ratios_valid else None

    candidate_proto_count = int(semantic_summary.get("semantic_prototype_count") or len(candidate_proto_counter))
    atom_proto_count = len({item.semantic_prototype_id for item in d4rt_items if item.semantic_prototype_id} | {item.semantic_prototype_id for item in semantic_items if item.source_kind == "D4RT_atom"})
    if atom_proto_count == 0:
        atom_proto_count = len({item.semantic_prototype_id for item in d4rt_items if item.semantic_prototype_id})
    atom_metrics = atom_summary.get("key_metrics") if isinstance(atom_summary.get("key_metrics"), dict) else atom_summary
    atom_reliability_mean = _float(atom_metrics.get("atom_reliability_mean"), None)
    diagnostic_gt_mean = _float(atom_metrics.get("diagnostic_GT_count_per_chunk_mean"), None)
    candidate_frame_count = len(candidate_frame_universe)

    summaries: list[dict[str, Any]] = []
    all_radius_items = list(d4rt_items) + (list(semantic_items) if args.semantic_only_repair else [])
    for variant in variants:
        summaries.append(
            _summarize_variant(
                variant=variant,
                selected=selected_by_variant.get(variant, []),
                all_items_for_radius=all_radius_items,
                available=variant_available.get(variant, False),
                availability_reason=variant_reasons.get(variant, ""),
                mode=variant_modes.get(variant, ""),
                candidate_frame_count=candidate_frame_count,
                candidate_proto_count=candidate_proto_count,
                atom_proto_count=atom_proto_count,
                atom_reliability_mean=atom_reliability_mean,
                diagnostic_gt_mean=diagnostic_gt_mean,
                k1_background_rate=k1_background_rate,
                diagnostic_gt=diagnostic_gt.get(variant),
                baseline_diag_ratio=baseline_diag_ratio,
            )
        )

    best_raw = [row for row in summaries if row.get("gate_pass_raw_candidate_prototype_denominator")]
    best_atom = [row for row in summaries if row.get("gate_pass_atom_universe_denominator")]
    selected_method = "K8_reliability_weighted_geo_semantic_FPS"
    selected_row = next((row for row in summaries if row["key_atom_variant"] == selected_method), {})
    gate = {
        "all_inputs_present": not any(row.get("missing") for row in missing_rows),
        "selected_method": selected_method,
        "selected_method_gate_pass_raw_candidate_prototype_denominator": bool(selected_row.get("gate_pass_raw_candidate_prototype_denominator")),
        "selected_method_gate_pass_atom_universe_denominator": bool(selected_row.get("gate_pass_atom_universe_denominator")),
        "any_variant_gate_pass_raw_candidate_prototype_denominator": bool(best_raw),
        "any_variant_gate_pass_atom_universe_denominator": bool(best_atom),
        "semantic_only_repair_enabled": bool(args.semantic_only_repair),
        "RADIO_available": False,
        "diagnostic_gt_skipped": bool(args.skip_diagnostic_gt),
    }
    decision = "PASS_V71_KEY_ATOMS" if gate["selected_method_gate_pass_raw_candidate_prototype_denominator"] else "NO_GO_PHASE4_KEY_ATOMS"
    if gate["selected_method_gate_pass_atom_universe_denominator"] and not gate["selected_method_gate_pass_raw_candidate_prototype_denominator"]:
        decision = "PARTIAL_PHASE4_KEY_ATOMS_ATOM_UNIVERSE_PASS_RAW_CANDIDATE_PROTO_FAIL"

    summary = {
        "decision": decision,
        "gate": gate,
        "plan_phase": "Phase4_Geo_Semantic_FPS_Key_Atom_Selection",
        "selection_scope": {
            "scenes": scenes,
            "D4RT_A0_atom_count_loaded": len(d4rt_items),
            "semantic_candidate_atom_count_loaded": len(semantic_items),
            "D4RT_chunk_count": len(d4rt_by_chunk),
            "semantic_candidate_chunk_count": len(semantic_by_chunk),
            "candidate_frame_count": candidate_frame_count,
            "candidate_semantic_prototype_count_raw": candidate_proto_count,
            "D4RT_A0_semantic_prototype_count": len({item.semantic_prototype_id for item in d4rt_items if item.semantic_prototype_id}),
            "atom_universe_semantic_prototype_count": atom_proto_count,
            "atom_source_type_counts": dict(atom_source_counts),
        },
        "upstream_metrics": {
            "atom_reliability_mean": atom_reliability_mean,
            "diagnostic_GT_count_per_chunk_mean": diagnostic_gt_mean,
            "candidate_bank_decision": candidate_summary.get("decision"),
            "semantic_feature_decision": semantic_summary.get("decision"),
            "atom_decision": atom_summary.get("decision"),
        },
        "variants": summaries,
        "diagnostic_gt_coverage": diagnostic_gt,
        "method_notes": {
            "semantic_only_repair_atoms": (
                "Candidate-mask-derived semantic atoms are enabled as a Phase4 repair path. "
                "They carry mask_observation_id and DINO semantic prototype, do not claim D4RT 3D position, "
                "and have D4RT_reliability_available=false."
            ),
            "GT_policy": "GT is used only after selection for diagnostic coverage ratios; all method rows set uses_gt_for_prediction=false.",
            "RADIO_policy": "K4/K6 are recorded unavailable because Phase3 RADIO_feature_success_rate was 0.",
            "raw_candidate_prototype_gate": "The plan's raw candidate prototype denominator is kept as a strict gate and is not silently replaced.",
            "coverage_radius_policy": "Coverage radii are deterministic-sample diagnostics (max 4096 universe rows, max 1024 selected rows per variant/mode) to avoid O(N*K) full distance blow-up; selection itself is not sampled by this policy.",
        },
    }

    summary_path = output_root / "key_atom_summary.json"
    rows_path = output_root / "key_atom_rows.csv"
    variants_path = output_root / "key_atom_variant_summary_rows.csv"
    missing_path = output_root / "missing_input_rows.csv"
    diag_path = output_root / "diagnostic_gt_coverage_rows.csv"
    _write_json(summary_path, summary)
    _write_csv(rows_path, key_rows)
    _write_csv(variants_path, summaries)
    _write_csv(missing_path, missing_rows)
    diag_rows = []
    for variant, data in diagnostic_gt.items():
        row = {"key_atom_variant": variant}
        row.update(data)
        diag_rows.append(row)
    _write_csv(diag_path, diag_rows)
    vis_rows = _write_visualizations(visual_root, summaries)
    vis_path = output_root / "visualization_rows.csv"
    _write_csv(vis_path, vis_rows)
    _write_sha_rows(output_root, [summary_path, rows_path, variants_path, missing_path, diag_path, vis_path] + [visual_root / row["path"].split("key_atoms/")[-1] for row in vis_rows])
    print(json.dumps({"decision": decision, "output_root": _rel(output_root), "visual_root": _rel(visual_root), "gate": gate}, indent=2, sort_keys=True))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atom-root", default="outputs/audit/v71_d4rt_atoms")
    parser.add_argument("--candidate-root", default="outputs/audit/v71_candidate_bank")
    parser.add_argument("--semantic-root", default="outputs/audit/v71_semantic_features")
    parser.add_argument("--witness-summary", default="outputs/audit/v70_carrier_witness/witness_summary.json")
    parser.add_argument("--output-root", default="outputs/audit/v71_key_atoms")
    parser.add_argument("--visual-root", default="outputs/audit/v71_visualizations/key_atoms")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--key-atoms-per-chunk", type=int, default=160)
    parser.add_argument("--d4rt-key-atoms-per-chunk", type=int, default=160)
    parser.add_argument("--max-d4rt-candidates-per-chunk", type=int, default=420)
    parser.add_argument("--semantic-only-repair", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-diagnostic-gt", action="store_true")
    parser.add_argument("--max-diagnostic-gt-frames", type=int, default=0)
    parser.add_argument("--max-atom-rows", type=int, default=0)
    parser.add_argument("--max-candidate-rows", type=int, default=0)
    parser.add_argument("--max-chunks", type=int, default=0)
    args = parser.parse_args(argv)
    args.max_atom_rows = int(args.max_atom_rows) or None
    args.max_candidate_rows = int(args.max_candidate_rows) or None
    return args


if __name__ == "__main__":
    run(parse_args())
