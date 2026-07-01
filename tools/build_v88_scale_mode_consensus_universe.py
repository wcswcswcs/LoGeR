#!/usr/bin/env python3
"""Build v88 Phase1 signed scale-mode consensus universe."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from v86_soft_latent_utils import effective_sample_size, finite_median, safe_float, stable_hash_float, write_csv, write_json


DEFAULT_PHASE1 = Path(
    "results/acl2_v87tf_scale_conditioned_latent_gauge_carrier/phase1_scale_conditioned_pair_universe_k16_r1_median_abs"
)
DEFAULT_ANCHOR_ROWS = Path(
    "results/acl2_v85tf_latent_anchor_alignment_pairwise_memory_ruler/phase1_anchor_pair_universe/anchor_pair_rows.csv"
)
DEFAULT_V87_CF_ROWS = Path(
    "results/acl2_v87tf_scale_conditioned_latent_gauge_carrier/phase8_merge_gauge_direct_pair_weighting/raw_overlap_geometry_counterfactual_rows.csv"
)
DEFAULT_OUT = Path("results/acl2_v88tf_scale_mode_consensus_gauge_update_attribution/phase1_scale_mode_consensus_universe")

EDGE_FIELDS = [
    "seq",
    "prev_chunk",
    "curr_chunk",
    "raw_pair_index_a",
    "raw_pair_index_b",
    "prev_patch_y",
    "prev_patch_x",
    "curr_patch_y",
    "curr_patch_x",
    "prev_frame_id",
    "curr_frame_id",
    "prev_edge_length",
    "curr_edge_length",
    "signed_log_shape_ratio",
    "abs_log_shape_ratio",
    "edge_weight",
    "conf_weight",
    "semantic_compatibility",
    "same_label",
    "same_role",
    "dynamic_or_boundary_flag",
    "zero_conf_flag",
    "high_residual_flag",
    "parallax_or_depth_spread_proxy",
    "overlap_residual",
    "source_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", type=Path, default=DEFAULT_PHASE1)
    parser.add_argument("--anchor-rows", type=Path, default=DEFAULT_ANCHOR_ROWS)
    parser.add_argument("--v87-counterfactual-rows", type=Path, default=DEFAULT_V87_CF_ROWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument("--patch-radius", type=int, default=0)
    parser.add_argument("--edge-sampling-cap", type=int, default=50_000)
    parser.add_argument("--min-raw-points", type=int, default=500)
    parser.add_argument("--hist-bin-min", type=float, default=-0.5)
    parser.add_argument("--hist-bin-max", type=float, default=0.5)
    parser.add_argument("--hist-bin-count", type=int, default=41)
    return parser.parse_args()


def _norm_seq(value: Any) -> str:
    return str(value).zfill(2)


def _load_raw(path: str) -> dict[str, np.ndarray] | None:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001
        return None
    keys = [
        "prev_overlap_local_points",
        "curr_overlap_local_points",
        "prev_conf",
        "curr_conf",
        "prev_frame_ids",
        "curr_frame_ids",
        "prev_pixel_coords",
        "curr_pixel_coords",
        "prev_semantic_labels",
        "curr_semantic_labels",
        "prev_semantic_conf",
        "curr_semantic_conf",
    ]
    out: dict[str, np.ndarray] = {}
    for key in keys:
        value = obj.get(key)
        if value is None:
            return None
        out[key] = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    return out


def _patch_coords_from_pixels(pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # raw overlap pixel coords are x,y; patch id follows y * 66 + x.
    patch_x = np.floor(pixels[:, 0].astype(np.float64) / 14.0).astype(np.int64)
    patch_y = np.floor(pixels[:, 1].astype(np.float64) / 14.0).astype(np.int64)
    patch_x = np.clip(patch_x, 0, 65)
    patch_y = np.clip(patch_y, 0, 18)
    return patch_y, patch_x, patch_y * 66 + patch_x


def _source_by_pair(anchor_rows: pd.DataFrame) -> dict[tuple[str, int, int], str]:
    out: dict[tuple[str, int, int], str] = {}
    for _, row in anchor_rows.drop_duplicates(["seq", "prev_chunk", "curr_chunk", "source_path"]).iterrows():
        key = (_norm_seq(row["seq"]), int(row["prev_chunk"]), int(row["curr_chunk"]))
        source = str(row.get("source_path") or "")
        if source:
            out.setdefault(key, source)
    return out


def _native_scale_by_pair(path: Path) -> dict[tuple[str, int, int], tuple[float, str]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: dict[tuple[str, int, int], tuple[float, str]] = {}
    for _, row in df.iterrows():
        scale = safe_float(row.get("native_scale"))
        if scale is None or scale <= 0:
            continue
        key = (_norm_seq(row.get("seq")), int(row["prev_chunk"]), int(row["curr_chunk"]))
        out[key] = (math.log(scale), "v87_phase8_native_scale")
    return out


def _weighted_sim3_scale(src: np.ndarray, dst: np.ndarray, weights: np.ndarray) -> float | None:
    mask = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1) & np.isfinite(weights) & (weights > 0)
    if int(mask.sum()) < 8:
        return None
    x = src[mask].astype(np.float64)
    y = dst[mask].astype(np.float64)
    w = weights[mask].astype(np.float64)
    w = w / max(float(w.sum()), 1e-12)
    mux = np.sum(x * w[:, None], axis=0)
    muy = np.sum(y * w[:, None], axis=0)
    xc = x - mux
    yc = y - muy
    cov = (yc * w[:, None]).T @ xc
    try:
        u, s, vt = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        return None
    d = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1] = -1.0
    var_x = float(np.sum(w * np.sum(xc * xc, axis=1)))
    if var_x <= 1e-12:
        return None
    scale = float(np.sum(s * d) / var_x)
    if not math.isfinite(scale) or scale <= 0:
        return None
    return math.log(scale)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float | None:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return None
    v = values[mask]
    w = weights[mask]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cdf = np.cumsum(w) / max(float(w.sum()), 1e-12)
    return float(v[np.searchsorted(cdf, q, side="left").clip(0, len(v) - 1)])


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return None
    return float(np.sum(values[mask] * weights[mask]) / max(float(weights[mask].sum()), 1e-12))


def _patch_group_edges(
    *,
    patch_ids: np.ndarray,
    patch_y: np.ndarray,
    patch_x: np.ndarray,
    pixels: np.ndarray,
    valid_idx: np.ndarray,
    knn_k: int,
    cap: int,
    salt: str,
) -> tuple[np.ndarray, np.ndarray]:
    edge_a: list[np.ndarray] = []
    edge_b: list[np.ndarray] = []
    for patch_id in np.unique(patch_ids[valid_idx]):
        center_y = int(patch_id // 66)
        center_x = int(patch_id % 66)
        if knn_k <= 0:
            continue
        neighborhood = (
            (np.abs(patch_y[valid_idx] - center_y) <= 0)
            & (np.abs(patch_x[valid_idx] - center_x) <= 0)
            & (patch_ids[valid_idx] == patch_id)
        )
        idx = valid_idx[neighborhood]
        if idx.size < 2:
            continue
        order = np.lexsort((pixels[idx, 0], pixels[idx, 1]))
        idx = idx[order]
        max_offset = min(knn_k, idx.size - 1)
        for offset in range(1, max_offset + 1):
            edge_a.append(idx[:-offset])
            edge_b.append(idx[offset:])
    if not edge_a:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    a = np.concatenate(edge_a).astype(np.int64)
    b = np.concatenate(edge_b).astype(np.int64)
    if a.size > cap:
        hashes = np.array([stable_hash_float(salt, int(x), int(y)) for x, y in zip(a, b)], dtype=np.float64)
        keep = np.argsort(hashes, kind="mergesort")[:cap]
        a = a[keep]
        b = b[keep]
    return a, b


def _edge_arrays(raw: dict[str, np.ndarray], source_path: str, seq: str, prev_chunk: int, curr_chunk: int, args: argparse.Namespace) -> dict[str, Any]:
    prev = raw["prev_overlap_local_points"].astype(np.float64)
    curr = raw["curr_overlap_local_points"].astype(np.float64)
    prev_pixels = raw["prev_pixel_coords"].astype(np.float64)
    curr_pixels = raw["curr_pixel_coords"].astype(np.float64)
    prev_y, prev_x, prev_patch_id = _patch_coords_from_pixels(prev_pixels)
    curr_y, curr_x, _ = _patch_coords_from_pixels(curr_pixels)
    conf = np.minimum(np.clip(raw["prev_conf"].astype(np.float64), 0.0, 1.0), np.clip(raw["curr_conf"].astype(np.float64), 0.0, 1.0))
    point_residual = np.linalg.norm(curr - prev, axis=1)
    finite = (
        np.isfinite(prev).all(axis=1)
        & np.isfinite(curr).all(axis=1)
        & np.isfinite(prev_pixels).all(axis=1)
        & np.isfinite(curr_pixels).all(axis=1)
        & np.isfinite(conf)
    )
    valid_idx = np.flatnonzero(finite)
    if valid_idx.size < args.min_raw_points:
        return {"status": f"too_few_raw_points:{valid_idx.size}", "edge_count": 0}
    a, b = _patch_group_edges(
        patch_ids=prev_patch_id,
        patch_y=prev_y,
        patch_x=prev_x,
        pixels=prev_pixels,
        valid_idx=valid_idx,
        knn_k=args.knn_k,
        cap=args.edge_sampling_cap,
        salt=f"{seq}:{prev_chunk}:{curr_chunk}",
    )
    if a.size == 0:
        return {"status": "no_edges", "edge_count": 0}
    prev_len = np.linalg.norm(prev[a] - prev[b], axis=1)
    curr_len = np.linalg.norm(curr[a] - curr[b], axis=1)
    length_mask = np.isfinite(prev_len) & np.isfinite(curr_len) & (prev_len > 1e-6) & (curr_len > 1e-6)
    a = a[length_mask]
    b = b[length_mask]
    prev_len = prev_len[length_mask]
    curr_len = curr_len[length_mask]
    if a.size == 0:
        return {"status": "no_valid_length_edges", "edge_count": 0}
    signed = np.log((curr_len + 1e-6) / (prev_len + 1e-6))
    abs_signed = np.abs(signed)
    conf_weight = np.minimum.reduce([conf[a], conf[b]])
    prev_sem = raw["prev_semantic_labels"].astype(np.int64)
    curr_sem = raw["curr_semantic_labels"].astype(np.int64)
    prev_sem_conf = np.clip(raw["prev_semantic_conf"].astype(np.float64), 0.0, 1.0)
    curr_sem_conf = np.clip(raw["curr_semantic_conf"].astype(np.float64), 0.0, 1.0)
    same_label_point = (prev_sem == curr_sem) & (prev_sem >= 0)
    same_label = same_label_point[a] & same_label_point[b] & (prev_sem[a] == prev_sem[b])
    same_role = same_label.copy()
    semantic_conf = np.minimum.reduce([prev_sem_conf[a], curr_sem_conf[a], prev_sem_conf[b], curr_sem_conf[b]])
    semantic_compat = np.where(same_label, semantic_conf, 0.5 * semantic_conf)
    dynamic_or_boundary = (~same_label) | (semantic_conf < 0.25)
    zero_conf = (conf[a] <= 1e-6) | (conf[b] <= 1e-6)
    edge_residual = 0.5 * (point_residual[a] + point_residual[b])
    residual_q75 = float(np.quantile(edge_residual[np.isfinite(edge_residual)], 0.75)) if np.isfinite(edge_residual).any() else 1.0
    high_residual = edge_residual > residual_q75
    spread = 0.5 * (np.abs(prev[a, 2] - prev[b, 2]) + np.abs(curr[a, 2] - curr[b, 2]))
    spread_q75 = float(np.quantile(spread[np.isfinite(spread)], 0.75)) if np.isfinite(spread).any() else 1.0
    parallax_proxy = np.clip(spread / max(spread_q75, 1e-12), 0.0, 1.0)
    overlap_weight = np.exp(-edge_residual / max(residual_q75, 1e-12))
    edge_weight = conf_weight * semantic_compat * overlap_weight * (0.25 + 0.75 * parallax_proxy)
    edge_weight = np.where(zero_conf, 0.0, edge_weight)
    return {
        "status": "ok",
        "edge_count": int(a.size),
        "a": a,
        "b": b,
        "prev_patch_y": prev_y[a],
        "prev_patch_x": prev_x[a],
        "curr_patch_y": curr_y[a],
        "curr_patch_x": curr_x[a],
        "prev_frame_id": raw["prev_frame_ids"][a],
        "curr_frame_id": raw["curr_frame_ids"][a],
        "prev_edge_length": prev_len,
        "curr_edge_length": curr_len,
        "signed_log_shape_ratio": signed,
        "abs_log_shape_ratio": abs_signed,
        "edge_weight": edge_weight,
        "conf_weight": conf_weight,
        "semantic_compatibility": semantic_compat,
        "same_label": same_label,
        "same_role": same_role,
        "dynamic_or_boundary_flag": dynamic_or_boundary,
        "zero_conf_flag": zero_conf,
        "high_residual_flag": high_residual,
        "parallax_or_depth_spread_proxy": parallax_proxy,
        "overlap_residual": edge_residual,
        "native_sim3_delta": _weighted_sim3_scale(prev[valid_idx], curr[valid_idx], conf[valid_idx]),
    }


def _histogram_stats(values: np.ndarray, weights: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    bins = np.linspace(args.hist_bin_min, args.hist_bin_max, args.hist_bin_count + 1)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if int(mask.sum()) == 0:
        return {"available": False, "hist_bins": bins, "hist_mass": np.zeros(args.hist_bin_count)}
    v = values[mask]
    w = weights[mask]
    hist, edges = np.histogram(v, bins=bins, weights=w)
    total = float(hist.sum())
    if total <= 1e-12:
        return {"available": False, "hist_bins": bins, "hist_mass": hist}
    probs = hist / total
    top_order = np.argsort(hist)[::-1]
    top1 = int(top_order[0])
    top2 = int(top_order[1]) if len(top_order) > 1 else top1
    mu = _weighted_quantile(v, w, 0.5)
    mad = None if mu is None else _weighted_quantile(np.abs(v - mu), w, 0.5)
    trimmed_mask = (v >= np.quantile(v, 0.10)) & (v <= np.quantile(v, 0.90))
    trimmed_mean = _weighted_mean(v[trimmed_mask], w[trimmed_mask]) if trimmed_mask.any() else _weighted_mean(v, w)
    return {
        "available": True,
        "hist_bins": edges,
        "hist_mass": hist,
        "weighted_mode_mu": mu,
        "weighted_trimmed_mean": trimmed_mean,
        "weighted_mode_abs_mu": None if mu is None else abs(mu),
        "weighted_mode_mad": mad,
        "mode_mass_top1": float(probs[top1]),
        "mode_mass_top2": float(probs[top2]),
        "mode_gap_top1_top2": float(probs[top1] - probs[top2]),
        "mode_entropy": float(-np.sum(probs[probs > 0] * np.log(probs[probs > 0]))),
        "mode_bimodality": float(probs[top1] - probs[top2]),
        "dominant_bin_index": top1,
        "dominant_bin_left": float(edges[top1]),
        "dominant_bin_right": float(edges[top1 + 1]),
        "dominant_bin_center": float(0.5 * (edges[top1] + edges[top1 + 1])),
    }


def _mode_mass(flags: np.ndarray, weights: np.ndarray, mode_mask: np.ndarray) -> float:
    denom = float(weights[mode_mask].sum())
    if denom <= 1e-12:
        return 0.0
    return float(weights[mode_mask & flags].sum() / denom)


def _write_edge_rows(writer: csv.DictWriter, pair_meta: dict[str, Any], edge_data: dict[str, Any]) -> None:
    n = int(edge_data["edge_count"])
    for i in range(n):
        writer.writerow(
            {
                "seq": pair_meta["seq"],
                "prev_chunk": pair_meta["prev_chunk"],
                "curr_chunk": pair_meta["curr_chunk"],
                "raw_pair_index_a": int(edge_data["a"][i]),
                "raw_pair_index_b": int(edge_data["b"][i]),
                "prev_patch_y": int(edge_data["prev_patch_y"][i]),
                "prev_patch_x": int(edge_data["prev_patch_x"][i]),
                "curr_patch_y": int(edge_data["curr_patch_y"][i]),
                "curr_patch_x": int(edge_data["curr_patch_x"][i]),
                "prev_frame_id": int(edge_data["prev_frame_id"][i]),
                "curr_frame_id": int(edge_data["curr_frame_id"][i]),
                "prev_edge_length": f"{float(edge_data['prev_edge_length'][i]):.12g}",
                "curr_edge_length": f"{float(edge_data['curr_edge_length'][i]):.12g}",
                "signed_log_shape_ratio": f"{float(edge_data['signed_log_shape_ratio'][i]):.12g}",
                "abs_log_shape_ratio": f"{float(edge_data['abs_log_shape_ratio'][i]):.12g}",
                "edge_weight": f"{float(edge_data['edge_weight'][i]):.12g}",
                "conf_weight": f"{float(edge_data['conf_weight'][i]):.12g}",
                "semantic_compatibility": f"{float(edge_data['semantic_compatibility'][i]):.12g}",
                "same_label": bool(edge_data["same_label"][i]),
                "same_role": bool(edge_data["same_role"][i]),
                "dynamic_or_boundary_flag": bool(edge_data["dynamic_or_boundary_flag"][i]),
                "zero_conf_flag": bool(edge_data["zero_conf_flag"][i]),
                "high_residual_flag": bool(edge_data["high_residual_flag"][i]),
                "parallax_or_depth_spread_proxy": f"{float(edge_data['parallax_or_depth_spread_proxy'][i]):.12g}",
                "overlap_residual": f"{float(edge_data['overlap_residual'][i]):.12g}",
                "source_path": pair_meta["source_path"],
            }
        )


def _plot_histogram(out_path: Path, hist: dict[str, Any], title: str, native_delta: float | None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    edges = hist["hist_bins"]
    mass = hist["hist_mass"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(centers, mass, width=float(edges[1] - edges[0]), color="#3366AA", alpha=0.75)
    if hist.get("weighted_mode_mu") is not None:
        ax.axvline(float(hist["weighted_mode_mu"]), color="#BB2222", linewidth=2, label="mode median")
    if native_delta is not None:
        ax.axvline(float(native_delta), color="#228833", linewidth=2, linestyle="--", label="native delta")
    ax.set_title(title)
    ax.set_xlabel("signed local log shape ratio")
    ax.set_ylabel("weighted mass")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = args.out_dir / "mode_histogram_previews"
    by_pair = pd.read_csv(args.phase1_dir / "scale_conditioned_pair_by_adjacent.csv")
    anchor_rows = pd.read_csv(args.anchor_rows, usecols=["seq", "prev_chunk", "curr_chunk", "source_path"])
    for frame in (by_pair, anchor_rows):
        frame["seq"] = frame["seq"].astype(str).str.zfill(2)
        frame["prev_chunk"] = frame["prev_chunk"].astype(int)
        frame["curr_chunk"] = frame["curr_chunk"].astype(int)
    source_map = _source_by_pair(anchor_rows)
    native_map = _native_scale_by_pair(args.v87_counterfactual_rows)
    pair_rows: list[dict[str, Any]] = []
    hist_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    preview_count = 0
    edge_path = args.out_dir / "scale_mode_edge_rows.csv"
    with edge_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EDGE_FIELDS)
        writer.writeheader()
        for _, row in by_pair.sort_values(["seq", "prev_chunk", "curr_chunk"]).iterrows():
            seq = _norm_seq(row["seq"])
            prev_chunk = int(row["prev_chunk"])
            curr_chunk = int(row["curr_chunk"])
            key = (seq, prev_chunk, curr_chunk)
            source_path = source_map.get(key, "")
            pair_meta = {
                "seq": seq,
                "prev_chunk": prev_chunk,
                "curr_chunk": curr_chunk,
                "source_path": source_path,
            }
            raw = _load_raw(source_path) if source_path else None
            if raw is None:
                missing.append({"seq": seq, "prev_chunk": prev_chunk, "curr_chunk": curr_chunk, "reason": "missing_or_unreadable_raw", "source_path": source_path})
                edge_data = {"status": "missing_or_unreadable_raw", "edge_count": 0}
            else:
                edge_data = _edge_arrays(raw, source_path, seq, prev_chunk, curr_chunk, args)
            native_delta, native_source = native_map.get(key, (None, ""))
            if native_delta is None and raw is not None and edge_data.get("native_sim3_delta") is not None:
                native_delta = edge_data.get("native_sim3_delta")
                native_source = "native_confidence_weighted_raw_sim3_scale"
            if edge_data.get("status") != "ok":
                hist = {"available": False, "hist_bins": np.linspace(args.hist_bin_min, args.hist_bin_max, args.hist_bin_count + 1), "hist_mass": np.zeros(args.hist_bin_count)}
                valid_edge_count = 0
                weights = np.array([], dtype=np.float64)
                signed = np.array([], dtype=np.float64)
            else:
                _write_edge_rows(writer, pair_meta, edge_data)
                valid_edge_count = int(edge_data["edge_count"])
                signed = edge_data["signed_log_shape_ratio"].astype(np.float64)
                weights = edge_data["edge_weight"].astype(np.float64)
                hist = _histogram_stats(signed, weights, args)
            if not hist.get("available"):
                missing.append({"seq": seq, "prev_chunk": prev_chunk, "curr_chunk": curr_chunk, "reason": edge_data.get("status"), "source_path": source_path})
            mode_mu = hist.get("weighted_mode_mu")
            mode_mad = hist.get("weighted_mode_mad")
            mode_mass = hist.get("mode_mass_top1")
            mode_entropy = hist.get("mode_entropy")
            dominant_left = hist.get("dominant_bin_left")
            dominant_right = hist.get("dominant_bin_right")
            if mode_mu is None:
                mode_confidence = 0.0
            else:
                mode_confidence = float((mode_mass or 0.0) * math.exp(-float(mode_mad or 0.0) / 0.25))
            native_mismatch = None if native_delta is None or mode_mu is None else abs(float(native_delta) - float(mode_mu))
            sign_mismatch = (
                False
                if native_delta is None or mode_mu is None or abs(float(mode_mu)) <= 1e-6
                else np.sign(float(native_delta)) != np.sign(float(mode_mu))
            )
            if hist.get("available") and dominant_left is not None and dominant_right is not None:
                mode_mask = (signed >= float(dominant_left)) & (signed < float(dominant_right)) & np.isfinite(weights) & (weights > 0)
                semantic_static = _mode_mass(~edge_data["dynamic_or_boundary_flag"].astype(bool), weights, mode_mask)
                semantic_dynamic = _mode_mass(edge_data["dynamic_or_boundary_flag"].astype(bool), weights, mode_mask)
                zero_conf_mass = _mode_mass(edge_data["zero_conf_flag"].astype(bool), weights, mode_mask)
            else:
                semantic_static = 0.0
                semantic_dynamic = 0.0
                zero_conf_mass = 0.0
            pair = {
                "seq": seq,
                "prev_chunk": prev_chunk,
                "curr_chunk": curr_chunk,
                "base_case_type": row.get("base_case_type", ""),
                "quality_type": row.get("quality_type", ""),
                "row_count": row.get("pair_row_count", ""),
                "valid_edge_count": valid_edge_count,
                "effective_edge_sample_size": effective_sample_size(weights),
                "weighted_mode_mu": mode_mu,
                "weighted_mode_abs_mu": hist.get("weighted_mode_abs_mu"),
                "weighted_mode_mad": mode_mad,
                "weighted_trimmed_mean": hist.get("weighted_trimmed_mean"),
                "mode_mass_top1": mode_mass,
                "mode_mass_top2": hist.get("mode_mass_top2"),
                "mode_gap_top1_top2": hist.get("mode_gap_top1_top2"),
                "mode_entropy": mode_entropy,
                "mode_bimodality": hist.get("mode_bimodality"),
                "mode_sign": "" if mode_mu is None else int(np.sign(float(mode_mu))),
                "mode_confidence": mode_confidence,
                "native_transition_source": native_source,
                "native_delta_log_scale": native_delta,
                "native_mode_mismatch": native_mismatch,
                "native_mode_sign_mismatch": sign_mismatch,
                "observability_score": row.get("observability_mean", ""),
                "semantic_static_mass_in_mode": semantic_static,
                "semantic_dynamic_or_boundary_mass_in_mode": semantic_dynamic,
                "zero_conf_mass_in_mode": zero_conf_mass,
                "local_shape_proxy_available_ratio": row.get("raw_shape_proxy_available_ratio", ""),
                "scale_label_available": pd.notna(row.get("abs_log_scale_jump_gt")),
                "abs_log_scale_jump_gt": row.get("abs_log_scale_jump_gt", ""),
                "offline_audit_label_only": True,
                "source_path": source_path,
            }
            pair_rows.append(pair)
            for bin_idx, (left, right, mass) in enumerate(zip(hist["hist_bins"][:-1], hist["hist_bins"][1:], hist["hist_mass"])):
                hist_rows.append(
                    {
                        "seq": seq,
                        "prev_chunk": prev_chunk,
                        "curr_chunk": curr_chunk,
                        "bin_index": bin_idx,
                        "bin_left": float(left),
                        "bin_right": float(right),
                        "weighted_mass": float(mass),
                    }
                )
            if preview_count < 8 and hist.get("available"):
                label = f"{seq}_{prev_chunk:03d}_{curr_chunk:03d}_{row.get('base_case_type', '')}_{row.get('quality_type', '')}"
                _plot_histogram(preview_dir / f"{label}.png", hist, label, native_delta)
                preview_count += 1
    write_csv(args.out_dir / "scale_mode_pair_rows.csv", pair_rows)
    write_csv(args.out_dir / "mode_histograms.csv", hist_rows)
    write_csv(args.out_dir / "missing_artifact_rows.csv", missing)
    high_quality = [row for row in pair_rows if row.get("quality_type") == "high_quality"]
    high_valid_counts = [safe_float(row.get("valid_edge_count")) for row in high_quality]
    high_valid_counts = [v for v in high_valid_counts if v is not None]
    high_available = [
        row
        for row in high_quality
        if safe_float(row.get("valid_edge_count")) is not None
        and safe_float(row.get("valid_edge_count")) >= 1000
        and safe_float(row.get("weighted_mode_mu")) is not None
    ]
    native_available = [row for row in pair_rows if safe_float(row.get("native_delta_log_scale")) is not None]
    zero_conf_values = [safe_float(row.get("zero_conf_mass_in_mode")) for row in high_quality]
    zero_conf_values = [v for v in zero_conf_values if v is not None]
    seq01_low_conf = [
        row
        for row in pair_rows
        if row.get("seq") == "01" and row.get("quality_type") != "high_quality"
    ]
    summary = {
        "phase": "Phase1_scale_mode_consensus_universe",
        "config": {
            "knn_k": args.knn_k,
            "patch_radius": args.patch_radius,
            "edge_sampling_cap": args.edge_sampling_cap,
            "min_raw_points": args.min_raw_points,
            "hist_bin_min": args.hist_bin_min,
            "hist_bin_max": args.hist_bin_max,
            "hist_bin_count": args.hist_bin_count,
        },
        "pair_rows": len(pair_rows),
        "sequence_coverage": len({row["seq"] for row in pair_rows}),
        "valid_edge_count_median_high_quality": finite_median(high_valid_counts),
        "local_shape_mode_available_high_quality_ratio": len(high_available) / max(len(high_quality), 1),
        "native_transition_proxy_available_ratio": len(native_available) / max(len(pair_rows), 1),
        "zero_conf_mode_mass_in_high_quality": finite_median(zero_conf_values),
        "seq01_low_conf_rows_retained_as_audit_only": len(seq01_low_conf),
        "histogram_preview_count": preview_count,
        "missing_artifact_count": len(missing),
        "checks": {},
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    zero_conf_mass = summary["zero_conf_mode_mass_in_high_quality"]
    zero_conf_mass_for_gate = 1.0 if zero_conf_mass is None else float(zero_conf_mass)
    checks = {
        "pair_rows_ge_49": summary["pair_rows"] >= 49,
        "sequence_coverage_ge_4": summary["sequence_coverage"] >= 4,
        "valid_edge_count_median_high_quality_ge_1000": (summary["valid_edge_count_median_high_quality"] or 0) >= 1000,
        "local_shape_mode_available_high_quality_ratio_ge_0_80": summary["local_shape_mode_available_high_quality_ratio"] >= 0.80,
        "native_transition_proxy_available_ratio_ge_0_80": summary["native_transition_proxy_available_ratio"] >= 0.80,
        "zero_conf_mode_mass_in_high_quality_le_0_05": zero_conf_mass_for_gate <= 0.05,
        "seq01_low_conf_rows_retained_as_stress_audit_only": len(seq01_low_conf) > 0,
        "histogram_preview_count_ge_8": preview_count >= 8,
    }
    summary["checks"] = checks
    summary["phase1_gate_pass"] = all(checks.values())
    if not summary["phase1_gate_pass"]:
        failed = [name for name, ok in checks.items() if not ok]
        summary["blocker"] = "phase1_gate_failed:" + ",".join(failed)
    else:
        summary["blocker"] = ""
    write_json(args.out_dir / "phase1_gate_summary.json", summary)
    missing_lines = [
        "# Phase1 Missing Artifact Report",
        "",
        f"- missing_artifact_count: `{len(missing)}`",
        f"- phase1_gate_pass: `{summary['phase1_gate_pass']}`",
    ]
    if missing:
        missing_lines += ["", "Examples:", ""]
        for item in missing[:20]:
            missing_lines.append(f"- {item}")
    (args.out_dir / "missing_artifact_report.md").write_text("\n".join(missing_lines) + "\n", encoding="utf-8")
    print(f"phase1_gate_pass={summary['phase1_gate_pass']}")
    print(f"pair_rows={summary['pair_rows']}")
    print(f"sequence_coverage={summary['sequence_coverage']}")
    print(f"valid_edge_count_median_high_quality={summary['valid_edge_count_median_high_quality']}")
    print(f"local_shape_mode_available_high_quality_ratio={summary['local_shape_mode_available_high_quality_ratio']}")
    print(f"native_transition_proxy_available_ratio={summary['native_transition_proxy_available_ratio']}")
    print(f"zero_conf_mode_mass_in_high_quality={summary['zero_conf_mode_mass_in_high_quality']}")
    print(f"histogram_preview_count={summary['histogram_preview_count']}")
    print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
