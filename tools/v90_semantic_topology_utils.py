#!/usr/bin/env python3
"""Shared helpers for ACL2 v90 semantic topology audits."""

from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from v86_soft_latent_utils import safe_float, safe_int, seq_norm, stable_hash_float


ROOT = Path("results/acl2_v90tf_semantic_object_topology_scale_mode_memory_control")
V89_ROOT = Path("results/acl2_v89tf_semantic_scale_mode_observability_memory_control")
V89_LEDGER = V89_ROOT / "phase1_semantic_scale_mode_ledger"
V89_FEATURE = V89_ROOT / "phase3_feature_match_semantic_ruler"
IMAGE_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences")
GRID_H = 19
GRID_W = 66
KITTI_H = 376
KITTI_W = 1241


def num(value: Any, default: float = 0.0) -> float:
    out = safe_float(value)
    return default if out is None else float(out)


def int0(value: Any) -> int:
    return int(safe_int(value) or 0)


def pair_id(seq: Any, prev: Any, curr: Any) -> str:
    return f"{seq_norm(seq)}_{int0(prev):03d}_{int0(curr):03d}"


def pair_key(row: Any) -> tuple[str, int, int]:
    return seq_norm(row["seq"]), int0(row["prev_chunk"]), int0(row["curr_chunk"])


def load_raw(path: str | Path) -> dict[str, np.ndarray] | None:
    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:
        return None
    keys = [
        "prev_pixel_coords",
        "curr_pixel_coords",
        "prev_frame_ids",
        "curr_frame_ids",
        "prev_semantic_labels",
        "curr_semantic_labels",
        "prev_semantic_conf",
        "curr_semantic_conf",
    ]
    out: dict[str, np.ndarray] = {}
    for key in keys:
        value = obj.get(key) if isinstance(obj, dict) else None
        if value is None:
            return None
        out[key] = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    for key in ["prev_overlap_local_points", "curr_overlap_local_points"]:
        value = obj.get(key) if isinstance(obj, dict) else None
        if value is not None:
            out[key] = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    return out


def median_frame(raw: dict[str, np.ndarray], side: str) -> int:
    key = "prev_frame_ids" if side == "prev" else "curr_frame_ids"
    vals = pd.to_numeric(pd.Series(raw.get(key, [])), errors="coerce").dropna()
    return int(vals.median()) if len(vals) else 0


def image_path(seq: str, frame: int, image_root: Path = IMAGE_ROOT) -> Path | None:
    for cam in ("image_2", "image_3"):
        path = image_root / seq_norm(seq) / cam / f"{int(frame):06d}.png"
        if path.exists():
            return path
    return None


def patch_coords(pixels_yx: np.ndarray, radius: int = 0) -> list[list[tuple[int, int]]]:
    arr = np.asarray(pixels_yx, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return []
    y = np.clip(arr[:, 0], 0, KITTI_H - 1)
    x = np.clip(arr[:, 1], 0, KITTI_W - 1)
    py = np.floor(y / (KITTI_H / GRID_H)).astype(int).clip(0, GRID_H - 1)
    px = np.floor(x / (KITTI_W / GRID_W)).astype(int).clip(0, GRID_W - 1)
    out: list[list[tuple[int, int]]] = []
    for yy, xx in zip(py, px):
        cells: list[tuple[int, int]] = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = int(yy + dy), int(xx + dx)
                if 0 <= ny < GRID_H and 0 <= nx < GRID_W:
                    cells.append((ny, nx))
        out.append(cells)
    return out


def _majority_label(items: list[tuple[int, float]]) -> tuple[int, float, int]:
    if not items:
        return -1, 0.0, 0
    counts: Counter[int] = Counter()
    conf_sum: defaultdict[int, float] = defaultdict(float)
    for label, conf in items:
        counts[int(label)] += 1
        conf_sum[int(label)] += float(conf)
    label, count = counts.most_common(1)[0]
    return int(label), float(conf_sum[label] / max(count, 1)), int(count)


def build_components_for_side(raw: dict[str, np.ndarray], side: str, radius: int = 0) -> tuple[list[dict[str, Any]], dict[tuple[int, int], dict[str, Any]], dict[int, int]]:
    labels_key = "prev_semantic_labels" if side == "prev" else "curr_semantic_labels"
    conf_key = "prev_semantic_conf" if side == "prev" else "curr_semantic_conf"
    pix_key = "prev_pixel_coords" if side == "prev" else "curr_pixel_coords"
    labels = np.asarray(raw[labels_key]).astype(int)
    confs = np.asarray(raw[conf_key]).astype(float)
    cells = patch_coords(raw[pix_key], radius=radius)
    cell_items: defaultdict[tuple[int, int], list[tuple[int, float]]] = defaultdict(list)
    raw_cell_count: defaultdict[tuple[int, int], int] = defaultdict(int)
    for label, conf, expanded in zip(labels, confs, cells):
        for cell in expanded:
            cell_items[cell].append((int(label), float(conf)))
            raw_cell_count[cell] += 1
    grid_label: dict[tuple[int, int], int] = {}
    grid_conf: dict[tuple[int, int], float] = {}
    grid_raw: dict[tuple[int, int], int] = {}
    for cell, items in cell_items.items():
        label, mean_conf, count = _majority_label(items)
        grid_label[cell] = label
        grid_conf[cell] = mean_conf
        grid_raw[cell] = count
    visited: set[tuple[int, int]] = set()
    rows: list[dict[str, Any]] = []
    cell_to_component: dict[tuple[int, int], dict[str, Any]] = {}
    label_component_counter: defaultdict[int, int] = defaultdict(int)
    for cell, label in sorted(grid_label.items(), key=lambda kv: (kv[1], kv[0][0], kv[0][1])):
        if cell in visited:
            continue
        label_component_counter[label] += 1
        comp_id = label_component_counter[label]
        queue = deque([cell])
        visited.add(cell)
        cells_in_comp: list[tuple[int, int]] = []
        while queue:
            cur = queue.popleft()
            cells_in_comp.append(cur)
            y, x = cur
            for nb in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if nb in visited:
                    continue
                if grid_label.get(nb) == label:
                    visited.add(nb)
                    queue.append(nb)
        boundary = 0
        for y, x in cells_in_comp:
            for nb in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if nb not in grid_label or grid_label.get(nb) != label:
                    boundary += 1
                    break
        patch_count = len(cells_in_comp)
        raw_count = sum(grid_raw.get(c, 0) for c in cells_in_comp)
        mean_conf = float(np.mean([grid_conf.get(c, 0.0) for c in cells_in_comp])) if cells_in_comp else 0.0
        row = {
            "side": side,
            "label_compact_id": int(label),
            "component_id": int(comp_id),
            "component_key": f"{side}:L{int(label)}:C{int(comp_id)}",
            "patch_count": patch_count,
            "raw_point_count": raw_count,
            "boundary_patch_count": boundary,
            "interior_patch_count": max(0, patch_count - boundary),
            "boundary_ratio": float(boundary / max(patch_count, 1)),
            "interior_ratio": float(max(0, patch_count - boundary) / max(patch_count, 1)),
            "mean_conf": mean_conf,
            "confidence_available": bool(mean_conf > 0.0),
            "dynamic_label_flag": bool(int(label) >= 250),
            "class_name_available": False,
            "label_mapping_status": "compact_project_local_id_no_class_names",
            "has_radio": False,
            "has_track": False,
        }
        rows.append(row)
        for c in cells_in_comp:
            cell_to_component[c] = row
    label_component_counts = {int(label): int(count) for label, count in label_component_counter.items()}
    return rows, cell_to_component, label_component_counts


def nearest_component_for_point(cell_map: dict[tuple[int, int], dict[str, Any]], cell: tuple[int, int]) -> dict[str, Any] | None:
    if cell in cell_map:
        return cell_map[cell]
    y, x = cell
    best: tuple[int, dict[str, Any]] | None = None
    for nb, comp in cell_map.items():
        dist = abs(nb[0] - y) + abs(nb[1] - x)
        if best is None or dist < best[0]:
            best = (dist, comp)
    return best[1] if best is not None and best[0] <= 2 else None


def stable_shuffle(values: pd.Series, salt: str) -> pd.Series:
    arr = values.to_numpy(copy=True)
    if len(arr) <= 1:
        return pd.Series(arr, index=values.index)
    order = sorted(range(len(arr)), key=lambda i: stable_hash_float(salt, i))
    shuffled = arr[order].copy()
    shuffled = np.roll(shuffled, 1)
    out = arr.copy()
    for dst, value in zip(order, shuffled):
        out[dst] = value
    return pd.Series(out, index=values.index)


def spearman(values: pd.Series, target: pd.Series) -> float | None:
    from v86_soft_latent_utils import spearman_rho

    return spearman_rho(values.tolist(), target.tolist())


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def metric_for_signal(df: pd.DataFrame, signal: str, values: pd.Series, geometry_ref: dict[str, Any] | None = None) -> dict[str, Any]:
    labelled = df[pd.to_numeric(df["abs_log_scale_jump_gt"], errors="coerce").notna()].copy()
    if len(labelled) == 0:
        return {"signal": signal, "available_rows": 0, "sequence_coverage": 0, "signal_pass": False}
    v = pd.to_numeric(values.loc[labelled.index], errors="coerce").fillna(0.0)
    y = pd.to_numeric(labelled["abs_log_scale_jump_gt"], errors="coerce")
    scale_threshold = float(y.quantile(0.75))
    threshold = float(v.quantile(0.75))
    flags = v >= threshold
    high = y >= scale_threshold
    bad = labelled["base_case_type"].astype(str).eq("bad")
    good_low = labelled["base_case_type"].astype(str).eq("good") & (~high)
    good = labelled["base_case_type"].astype(str).eq("good")
    rho = spearman(v, y)
    controls = {
        "semantic_label_shuffle": stable_shuffle(v, f"{signal}:semantic_label_shuffle"),
        "component_id_shuffle": stable_shuffle(v, f"{signal}:component_id_shuffle"),
        "boundary_interior_shuffle": stable_shuffle(v, f"{signal}:boundary_interior_shuffle"),
        "feature_match_correspondence_shuffle": stable_shuffle(v, f"{signal}:feature_match_correspondence_shuffle"),
        "scale_mode_sign_shuffle": -v,
        "same_geometry_topology_shuffle": stable_shuffle(v, f"{signal}:same_geometry_topology_shuffle"),
    }
    control_rhos = {name: spearman(ctrl, y) for name, ctrl in controls.items()}
    finite_controls = [float(x) for x in control_rhos.values() if x is not None and math.isfinite(float(x))]
    max_control = max(finite_controls) if finite_controls else None
    semantic_shuffle_rho = control_rhos.get("semantic_label_shuffle")
    component_shuffle_rho = control_rhos.get("component_id_shuffle")
    semantic_margin = None if rho is None or semantic_shuffle_rho is None else float(rho - semantic_shuffle_rho)
    component_margin = None if rho is None or component_shuffle_rho is None else float(rho - component_shuffle_rho)
    max_margin = None if rho is None or max_control is None else float(rho - max_control)
    bad_or_high = bad | high
    bad_recall = float((flags & bad_or_high).sum() / max(int(bad_or_high.sum()), 1))
    good_fpr = float((flags & good_low).sum() / max(int(good_low.sum()), 1))
    good_any_fpr = float((flags & good).sum() / max(int(good.sum()), 1))
    geom_rho = None if geometry_ref is None else geometry_ref.get("spearman_rho_abs_log_scale_jump")
    geom_bad = 0.0 if geometry_ref is None else float(geometry_ref.get("bad_recall") or 0.0)
    geom_fpr = 1.0 if geometry_ref is None else float(geometry_ref.get("good_false_positive_rate") or 1.0)
    is_topology = signal.startswith("T") or signal.startswith("P")
    lift = bool(is_topology and rho is not None and geom_rho is not None and rho >= float(geom_rho) + 0.05 and rho >= 0.30)
    specificity = bool(is_topology and semantic_margin is not None and component_margin is not None and semantic_margin >= 0.05 and component_margin >= 0.05)
    good_protection = bool(is_topology and good_fpr <= max(0.25, geom_fpr - 0.15) and bad_recall >= max(0.45, geom_bad - 0.10))
    signal_pass = bool(is_topology and specificity and (lift or (bad_recall >= 0.55 and good_fpr <= 0.25) or good_protection))
    return {
        "signal": signal,
        "is_topology_conditioned": is_topology,
        "available_rows": int(len(labelled)),
        "sequence_coverage": int(labelled["seq"].astype(str).str.zfill(2).nunique()),
        "spearman_rho_abs_log_scale_jump": rho,
        "max_control_rho": max_control,
        "semantic_shuffle_rho": semantic_shuffle_rho,
        "component_shuffle_rho": component_shuffle_rho,
        "semantic_shuffle_margin": semantic_margin,
        "component_shuffle_margin": component_margin,
        "max_control_margin": max_margin,
        "bad_recall": bad_recall,
        "good_false_positive_rate": good_fpr,
        "good_any_fpr": good_any_fpr,
        "balanced_accuracy": float(0.5 * (bad_recall + (1.0 - good_fpr))),
        "signal_threshold_q75": threshold,
        "scale_high_threshold_q75": scale_threshold,
        "criterion_topology_lift": lift,
        "criterion_topology_specificity": specificity,
        "criterion_good_protection": good_protection,
        "signal_pass": signal_pass,
    }
