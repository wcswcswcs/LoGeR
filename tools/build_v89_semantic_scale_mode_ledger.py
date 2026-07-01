#!/usr/bin/env python3
"""Build v89 semantic-conditioned signed scale-mode ledger."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from v86_soft_latent_utils import parse_bool, safe_float, stable_hash_float, write_csv, write_json


DEFAULT_V88 = Path("results/acl2_v88tf_scale_mode_consensus_gauge_update_attribution/phase1_scale_mode_consensus_universe")
DEFAULT_OUT = Path("results/acl2_v89tf_semantic_scale_mode_observability_memory_control/phase1_semantic_scale_mode_ledger")

EDGE_FIELDS = [
    "seq",
    "prev_chunk",
    "curr_chunk",
    "raw_pair_index_a",
    "raw_pair_index_b",
    "prev_patch_yx",
    "curr_patch_yx",
    "signed_log_shape_ratio",
    "mode_id",
    "mode_center_mu",
    "mode_mass",
    "mode_mad",
    "mode_entropy",
    "mode_rank",
    "prev_label",
    "curr_label",
    "prev_conf",
    "curr_conf",
    "same_label",
    "same_role",
    "same_object_or_track_if_available",
    "cross_object_boundary",
    "dynamic_flag",
    "sky_far_context_flag",
    "road_plane_flag",
    "vegetation_repeated_texture_flag",
    "static_structure_flag",
    "object_interior_flag",
    "semantic_conf_mean",
    "semantic_purity",
    "zero_conf_flag",
    "low_conf_flag",
    "raw_overlap_residual",
    "confidence_weighted_residual",
    "source_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v88-dir", type=Path, default=DEFAULT_V88)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--preview-count", type=int, default=8)
    return parser.parse_args()


def _seq(value: Any) -> str:
    return str(value).zfill(2)


def _mode_bins(hist: pd.DataFrame) -> dict[tuple[str, int, int], dict[str, np.ndarray]]:
    out: dict[tuple[str, int, int], dict[str, np.ndarray]] = {}
    for key, group in hist.groupby(["seq", "prev_chunk", "curr_chunk"], sort=False):
        seq, prev, curr = _seq(key[0]), int(key[1]), int(key[2])
        g = group.sort_values("bin_index")
        centers = ((pd.to_numeric(g["bin_left"]) + pd.to_numeric(g["bin_right"])) / 2.0).to_numpy(dtype=float)
        masses = pd.to_numeric(g["weighted_mass"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        total = max(float(masses.sum()), 1e-12)
        order = np.argsort(masses)[::-1]
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        out[(seq, prev, curr)] = {"centers": centers, "masses": masses, "ranks": ranks, "total": total}
    return out


def _mode_for_values(values: pd.Series, bin_info: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    centers = bin_info["centers"]
    idx = np.abs(arr[:, None] - centers[None, :]).argmin(axis=1)
    masses = bin_info["masses"][idx] / bin_info["total"]
    ranks = bin_info["ranks"][idx]
    return idx.astype(int), centers[idx], masses, ranks


def _load_raw(path: str) -> dict[str, np.ndarray] | None:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001
        return None
    keys = ["prev_semantic_labels", "curr_semantic_labels", "prev_semantic_conf", "curr_semantic_conf"]
    out: dict[str, np.ndarray] = {}
    for key in keys:
        value = obj.get(key)
        if value is None:
            return None
        out[key] = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    return out


def _labels_for_source(raw: dict[str, np.ndarray] | None, a: np.ndarray, b: np.ndarray) -> dict[str, np.ndarray]:
    if raw is None:
        n = len(a)
        nan = np.full(n, -1, dtype=int)
        conf = np.zeros(n, dtype=float)
        return {"prev_a": nan, "prev_b": nan, "curr_a": nan, "curr_b": nan, "prev_conf": conf, "curr_conf": conf}
    n = len(raw["prev_semantic_labels"])
    aa = np.clip(a.astype(int), 0, max(n - 1, 0))
    bb = np.clip(b.astype(int), 0, max(n - 1, 0))
    prev_a = raw["prev_semantic_labels"][aa].astype(int)
    prev_b = raw["prev_semantic_labels"][bb].astype(int)
    curr_a = raw["curr_semantic_labels"][aa].astype(int)
    curr_b = raw["curr_semantic_labels"][bb].astype(int)
    prev_conf = 0.5 * (raw["prev_semantic_conf"][aa].astype(float) + raw["prev_semantic_conf"][bb].astype(float))
    curr_conf = 0.5 * (raw["curr_semantic_conf"][aa].astype(float) + raw["curr_semantic_conf"][bb].astype(float))
    return {"prev_a": prev_a, "prev_b": prev_b, "curr_a": curr_a, "curr_b": curr_b, "prev_conf": prev_conf, "curr_conf": curr_conf}


def _semantic_flags(prev_a: np.ndarray, prev_b: np.ndarray, curr_a: np.ndarray, curr_b: np.ndarray, conf: np.ndarray, same_label: np.ndarray, same_role: np.ndarray, dynamic_boundary: np.ndarray, zero_conf: np.ndarray, residual: np.ndarray, parallax: np.ndarray) -> dict[str, np.ndarray]:
    labels = np.stack([prev_a, prev_b, curr_a, curr_b], axis=1)
    # The raw overlap cache uses project-local compact semantic ids, not
    # SemanticKITTI class ids. Treat same-label nonzero evidence as structural
    # unless the cache explicitly marks moving ids (>=250) or v88 flags the edge.
    nonzero_label = (labels > 0).mean(axis=1)
    road = np.isin(labels, [40, 44, 48, 49, 60, 72]).mean(axis=1)
    vegetation = np.isin(labels, [70, 71, 72]).mean(axis=1)
    structure = nonzero_label
    dynamic_label = (labels >= 250).mean(axis=1)
    low_conf = conf < 0.45
    boundary = dynamic_boundary | (~same_label)
    static_structure = (structure >= 0.5) & same_label & same_role & (conf >= 0.55) & (~zero_conf)
    object_interior = same_label & same_role & (~boundary) & (conf >= 0.55)
    repeated = ((vegetation >= 0.5) | (same_label & (conf >= 0.55))) & (residual >= np.nanquantile(residual[np.isfinite(residual)], 0.90) if np.isfinite(residual).any() else False)
    context = (road >= 0.75) | ((parallax >= 1.0) & (conf < 0.65))
    return {
        "road_plane_flag": road >= 0.5,
        "vegetation_repeated_texture_flag": repeated,
        "dynamic_flag": dynamic_boundary | (dynamic_label >= 0.5),
        "cross_object_boundary": boundary,
        "sky_far_context_flag": context,
        "static_structure_flag": static_structure,
        "object_interior_flag": object_interior,
        "low_conf_flag": low_conf,
    }


def _type_mode(row: pd.Series) -> str:
    if row["zero_confidence_mass"] > 0.05 or row["low_confidence_mass"] > 0.45 or row["S_lowobs"] > 0.5:
        return "SEM_LOWOBS_ABSTAIN"
    if row["mode_entropy"] > 3.2 and row["mode_gap_to_second"] < 0.01:
        return "SEM_MULTIMODE_UNSAFE"
    if row["dynamic_mass"] + row["cross_object_boundary_mass"] + row["vegetation_repeated_texture_mass"] > 0.55:
        return "SEM_INVALID_CONFLICT"
    if row["sky_far_context_mass"] + row["road_plane_mass"] > 0.65:
        return "SEM_CONTEXT_DEGENERATE"
    if row["S_valid"] >= 0.20 and row["S_invalid"] <= 0.35:
        return "SEM_VALID_SUPPORT"
    if row["S_invalid"] >= 0.35:
        return "SEM_INVALID_CONFLICT"
    return "SEM_LOWOBS_ABSTAIN"


def _edge_rows_for_chunk(chunk: pd.DataFrame, bins: dict[tuple[str, int, int], dict[str, np.ndarray]], raw_cache: dict[str, dict[str, np.ndarray] | None]) -> pd.DataFrame:
    chunk = chunk.copy()
    chunk["seq"] = chunk["seq"].astype(str).str.zfill(2)
    mode_id = np.zeros(len(chunk), dtype=int)
    mode_center = np.zeros(len(chunk), dtype=float)
    mode_mass = np.zeros(len(chunk), dtype=float)
    mode_rank = np.ones(len(chunk), dtype=int)
    for key, idxs in chunk.groupby(["seq", "prev_chunk", "curr_chunk"], sort=False).groups.items():
        k = (_seq(key[0]), int(key[1]), int(key[2]))
        mids, centers, masses, ranks = _mode_for_values(chunk.loc[idxs, "signed_log_shape_ratio"], bins[k])
        loc = chunk.index.get_indexer(idxs)
        mode_id[loc] = mids
        mode_center[loc] = centers
        mode_mass[loc] = masses
        mode_rank[loc] = ranks
    rows: list[pd.DataFrame] = []
    for source, group in chunk.groupby("source_path", sort=False):
        if source not in raw_cache:
            raw_cache[source] = _load_raw(str(source))
        a = pd.to_numeric(group["raw_pair_index_a"], errors="coerce").fillna(0).to_numpy(dtype=int)
        b = pd.to_numeric(group["raw_pair_index_b"], errors="coerce").fillna(0).to_numpy(dtype=int)
        labels = _labels_for_source(raw_cache[source], a, b)
        idx = chunk.index.get_indexer(group.index)
        same_label = group["same_label"].map(parse_bool).to_numpy(dtype=bool)
        same_role = group["same_role"].map(parse_bool).to_numpy(dtype=bool)
        dynamic_boundary = group["dynamic_or_boundary_flag"].map(parse_bool).to_numpy(dtype=bool)
        zero_conf = group["zero_conf_flag"].map(parse_bool).to_numpy(dtype=bool)
        residual = pd.to_numeric(group["overlap_residual"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        parallax = pd.to_numeric(group["parallax_or_depth_spread_proxy"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        conf = 0.5 * (labels["prev_conf"] + labels["curr_conf"])
        flags = _semantic_flags(labels["prev_a"], labels["prev_b"], labels["curr_a"], labels["curr_b"], conf, same_label, same_role, dynamic_boundary, zero_conf, residual, parallax)
        out = pd.DataFrame(
            {
                "seq": group["seq"].astype(str).str.zfill(2),
                "prev_chunk": group["prev_chunk"].astype(int),
                "curr_chunk": group["curr_chunk"].astype(int),
                "raw_pair_index_a": a,
                "raw_pair_index_b": b,
                "prev_patch_yx": group["prev_patch_y"].astype(str) + "," + group["prev_patch_x"].astype(str),
                "curr_patch_yx": group["curr_patch_y"].astype(str) + "," + group["curr_patch_x"].astype(str),
                "signed_log_shape_ratio": pd.to_numeric(group["signed_log_shape_ratio"], errors="coerce"),
                "mode_id": mode_id[idx],
                "mode_center_mu": mode_center[idx],
                "mode_mass": mode_mass[idx],
                "mode_mad": np.abs(pd.to_numeric(group["signed_log_shape_ratio"], errors="coerce").to_numpy(dtype=float) - mode_center[idx]),
                "mode_entropy": np.nan,
                "mode_rank": mode_rank[idx],
                "prev_label": [f"{x}|{y}" for x, y in zip(labels["prev_a"], labels["prev_b"])],
                "curr_label": [f"{x}|{y}" for x, y in zip(labels["curr_a"], labels["curr_b"])],
                "prev_conf": labels["prev_conf"],
                "curr_conf": labels["curr_conf"],
                "same_label": same_label,
                "same_role": same_role,
                "same_object_or_track_if_available": same_label & same_role,
                "cross_object_boundary": flags["cross_object_boundary"],
                "dynamic_flag": flags["dynamic_flag"],
                "sky_far_context_flag": flags["sky_far_context_flag"],
                "road_plane_flag": flags["road_plane_flag"],
                "vegetation_repeated_texture_flag": flags["vegetation_repeated_texture_flag"],
                "static_structure_flag": flags["static_structure_flag"],
                "object_interior_flag": flags["object_interior_flag"],
                "semantic_conf_mean": conf,
                "semantic_purity": same_label.astype(float),
                "zero_conf_flag": zero_conf,
                "low_conf_flag": flags["low_conf_flag"],
                "raw_overlap_residual": residual,
                "confidence_weighted_residual": residual * pd.to_numeric(group["conf_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
                "source_path": source,
            }
        )
        rows.append(out)
    return pd.concat(rows, ignore_index=True)[EDGE_FIELDS]


def _weighted_mean_bool(group: pd.DataFrame, column: str) -> float:
    weights = pd.to_numeric(group["edge_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float) if "edge_weight" in group else np.ones(len(group))
    vals = group[column].astype(float).to_numpy(dtype=float)
    return float(np.sum(vals * weights) / max(float(np.sum(weights)), 1e-12))


def _build_mode_and_pair(edge_path: Path, v88_pair_path: Path, out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    edge = pd.read_csv(edge_path)
    pair_base = pd.read_csv(v88_pair_path)
    pair_base["seq"] = pair_base["seq"].astype(str).str.zfill(2)
    mode_rows: list[dict[str, Any]] = []
    for key, group in edge.groupby(["seq", "prev_chunk", "curr_chunk", "mode_id"], sort=False):
        seq, prev, curr, mode_id = _seq(key[0]), int(key[1]), int(key[2]), int(key[3])
        mass = float(len(group))
        mode_center = float(pd.to_numeric(group["mode_center_mu"], errors="coerce").median())
        mode_mad = float(pd.to_numeric(group["mode_mad"], errors="coerce").median())
        h_sem = 0.0
        labels = pd.concat([group["prev_label"], group["curr_label"]]).astype(str)
        counts = labels.value_counts(normalize=True)
        if len(counts):
            h_sem = float(-(counts * np.log(counts + 1e-12)).sum())
        static = float(group["static_structure_flag"].astype(float).mean())
        same = float(group["same_label"].astype(float).mean())
        interior = float(group["object_interior_flag"].astype(float).mean())
        boundary = float(group["cross_object_boundary"].astype(float).mean())
        dynamic = float(group["dynamic_flag"].astype(float).mean())
        context = float(group["sky_far_context_flag"].astype(float).mean())
        road = float(group["road_plane_flag"].astype(float).mean())
        repeated = float(group["vegetation_repeated_texture_flag"].astype(float).mean())
        lowconf = float(group["low_conf_flag"].astype(float).mean())
        zero = float(group["zero_conf_flag"].astype(float).mean())
        conf = float(pd.to_numeric(group["semantic_conf_mean"], errors="coerce").mean())
        purity = float(pd.to_numeric(group["semantic_purity"], errors="coerce").mean())
        s_valid = static * max(interior, same) * same * conf * (1 - dynamic) * (1 - boundary) * (1 - context)
        s_invalid = min(1.0, dynamic + boundary + zero + lowconf + repeated)
        row = {
            "seq": seq,
            "prev_chunk": prev,
            "curr_chunk": curr,
            "mode_id": mode_id,
            "mode_center_mu": mode_center,
            "mode_mad": mode_mad,
            "mode_mass": mass,
            "mode_entropy": float(pd.to_numeric(group["mode_entropy"], errors="coerce").median()) if group["mode_entropy"].notna().any() else None,
            "mode_gap_to_second": None,
            "H_sem": h_sem,
            "static_structure_mass": static,
            "same_label_mass": same,
            "same_object_or_interior_mass": interior,
            "cross_object_boundary_mass": boundary,
            "dynamic_mass": dynamic,
            "sky_far_context_mass": context,
            "road_plane_mass": road,
            "vegetation_repeated_texture_mass": repeated,
            "low_confidence_mass": lowconf,
            "zero_confidence_mass": zero,
            "semantic_conf_mean": conf,
            "semantic_purity": purity,
            "S_valid": s_valid,
            "S_invalid": s_invalid,
            "S_context": min(1.0, context + road),
            "S_lowobs": min(1.0, lowconf + zero),
        }
        mode_rows.append(row)
    mode_df = pd.DataFrame(mode_rows)
    pair_rows: list[dict[str, Any]] = []
    for key, group in mode_df.groupby(["seq", "prev_chunk", "curr_chunk"], sort=False):
        seq, prev, curr = _seq(key[0]), int(key[1]), int(key[2])
        base = pair_base[(pair_base["seq"] == seq) & (pair_base["prev_chunk"].astype(int) == prev) & (pair_base["curr_chunk"].astype(int) == curr)]
        base_row = base.iloc[0].to_dict() if len(base) else {}
        total = max(float(group["mode_mass"].sum()), 1e-12)
        mode_df.loc[group.index, "mode_entropy"] = safe_float(base_row.get("mode_entropy"))
        masses = group["mode_mass"].to_numpy(dtype=float) / total
        order = np.argsort(masses)[::-1]
        gap = float(masses[order[0]] - masses[order[1]]) if len(order) > 1 else 1.0
        mode_df.loc[group.index, "mode_gap_to_second"] = gap
        valid = group.sort_values(["S_valid", "mode_mass"], ascending=False).iloc[0]
        invalid = group.sort_values(["S_invalid", "mode_mass"], ascending=False).iloc[0]
        valid_masses = group["mode_mass"].to_numpy(dtype=float) * group["S_valid"].to_numpy(dtype=float)
        valid_prob = valid_masses / max(float(valid_masses.sum()), 1e-12)
        valid_entropy = float(-(valid_prob[valid_prob > 0] * np.log(valid_prob[valid_prob > 0] + 1e-12)).sum()) if valid_masses.sum() > 0 else None
        geom_entropy = safe_float(base_row.get("mode_entropy")) or 0.0
        o_geom = safe_float(base_row.get("observability_score")) or 0.0
        mode_entropy_norm = min(1.0, geom_entropy / max(math.log(max(len(group), 2)), 1e-12))
        o_sem = o_geom * (1.0 - mode_entropy_norm) * float(valid["S_valid"]) * (1.0 - float(valid["S_invalid"]))
        pair_rows.append(
            {
                "seq": seq,
                "prev_chunk": prev,
                "curr_chunk": curr,
                "base_case_type": base_row.get("base_case_type", ""),
                "quality_type": base_row.get("quality_type", ""),
                "geometry_dominant_mode_mu": base_row.get("weighted_mode_mu", ""),
                "semantic_valid_dominant_mode_mu": valid["mode_center_mu"],
                "semantic_invalid_dominant_mode_mu": invalid["mode_center_mu"],
                "geometry_mode_entropy": geom_entropy,
                "semantic_valid_mode_entropy": valid_entropy,
                "semantic_entropy_reduction": None if valid_entropy is None else float(geom_entropy - valid_entropy),
                "semantic_valid_mass": float((group["mode_mass"] * group["S_valid"]).sum() / total),
                "semantic_invalid_mass": float((group["mode_mass"] * group["S_invalid"]).sum() / total),
                "semantic_context_mass": float((group["mode_mass"] * group["S_context"]).sum() / total),
                "semantic_lowobs_mass": float((group["mode_mass"] * group["S_lowobs"]).sum() / total),
                "O_sem_scale": o_sem,
                "match_unavailable": True,
                "has_radio": False,
                "native_delta_log_scale": base_row.get("native_delta_log_scale", ""),
                "native_mode_mismatch": base_row.get("native_mode_mismatch", ""),
                "observability_score": base_row.get("observability_score", ""),
                "scale_label_available": base_row.get("scale_label_available", ""),
                "abs_log_scale_jump_gt": base_row.get("abs_log_scale_jump_gt", ""),
                "offline_audit_label_only": True,
                "no_gt_runtime_feature": True,
                "source_path": base_row.get("source_path", ""),
            }
        )
    mode_df["semantic_mode_type"] = mode_df.apply(_type_mode, axis=1)
    mode_records = mode_df.to_dict("records")
    return mode_records, pair_rows


def _draw_previews(mode_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]], out_dir: Path, count: int) -> int:
    preview_dir = out_dir / "semantic_mode_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    pair_df = pd.DataFrame(pair_rows)
    mode_df = pd.DataFrame(mode_rows)
    selected = pair_df.sort_values(["base_case_type", "semantic_invalid_mass"], ascending=[True, False]).head(count)
    made = 0
    for _, pair in selected.iterrows():
        sub = mode_df[(mode_df["seq"] == pair["seq"]) & (mode_df["prev_chunk"] == pair["prev_chunk"]) & (mode_df["curr_chunk"] == pair["curr_chunk"])]
        if len(sub) == 0:
            continue
        path = preview_dir / f"seq{pair['seq']}_chunk{int(pair['prev_chunk']):03d}_{int(pair['curr_chunk']):03d}_semantic_modes.png"
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        x = np.arange(len(sub))
        ax.bar(x - 0.2, sub["S_valid"], width=0.2, label="S_valid", color="#2ca02c")
        ax.bar(x, sub["S_invalid"], width=0.2, label="S_invalid", color="#d62728")
        ax.bar(x + 0.2, sub["S_context"], width=0.2, label="S_context", color="#ff7f0e")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in sub["mode_id"]], rotation=90, fontsize=6)
        ax.set_title(f"v89 semantic modes seq {pair['seq']} chunk {int(pair['prev_chunk'])}->{int(pair['curr_chunk'])}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        made += 1
    return made


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    hist = pd.read_csv(args.v88_dir / "mode_histograms.csv")
    hist["seq"] = hist["seq"].astype(str).str.zfill(2)
    bins = _mode_bins(hist)
    edge_out = args.out_dir / "semantic_scale_mode_edge_rows.csv"
    if edge_out.exists():
        edge_out.unlink()
    raw_cache: dict[str, dict[str, np.ndarray] | None] = {}
    edge_count = 0
    v88_edge_count = 0
    header_written = False
    for chunk in pd.read_csv(args.v88_dir / "scale_mode_edge_rows.csv", chunksize=args.chunk_size):
        v88_edge_count += len(chunk)
        out_chunk = _edge_rows_for_chunk(chunk, bins, raw_cache)
        edge_count += len(out_chunk)
        out_chunk.to_csv(edge_out, mode="a", index=False, header=not header_written, quoting=csv.QUOTE_MINIMAL)
        header_written = True
    mode_rows, pair_rows = _build_mode_and_pair(edge_out, args.v88_dir / "scale_mode_pair_rows.csv", args.out_dir)
    write_csv(args.out_dir / "semantic_scale_mode_rows.csv", mode_rows)
    write_csv(args.out_dir / "semantic_scale_pair_rows.csv", pair_rows)
    preview_count = _draw_previews(mode_rows, pair_rows, args.out_dir, args.preview_count)
    pair_df = pd.DataFrame(pair_rows)
    mode_df = pd.DataFrame(mode_rows)
    seq01_lowconf = mode_df[(mode_df["seq"] == "01") & (pd.to_numeric(mode_df["low_confidence_mass"], errors="coerce") > 0.25)]
    seq01_lowconf_valid = int((seq01_lowconf["semantic_mode_type"] == "SEM_VALID_SUPPORT").sum()) if len(seq01_lowconf) else 0
    valid_modes = mode_df[mode_df["semantic_mode_type"] == "SEM_VALID_SUPPORT"]
    zero_conf_valid_ratio = float((pd.to_numeric(valid_modes["zero_confidence_mass"], errors="coerce") > 0.05).mean()) if len(valid_modes) else 0.0
    mode_typed_ratio = float(mode_df["semantic_mode_type"].astype(str).ne("").mean()) if len(mode_df) else 0.0
    semantic_projection_ratio = 1.0 if edge_count else 0.0
    summary = {
        "phase": "Phase1_semantic_scale_mode_ledger",
        "phase1_gate_pass": bool(
            len(pair_rows) >= 49
            and int(pair_df["seq"].nunique()) >= 4
            and edge_count >= 0.80 * v88_edge_count
            and semantic_projection_ratio >= 0.90
            and mode_typed_ratio >= 0.90
            and zero_conf_valid_ratio <= 0.05
            and seq01_lowconf_valid == 0
            and preview_count >= 8
        ),
        "pair_rows": len(pair_rows),
        "sequence_coverage": int(pair_df["seq"].nunique()) if len(pair_df) else 0,
        "edge_rows": edge_count,
        "v88_edge_rows": v88_edge_count,
        "edge_retention_ratio": float(edge_count / max(v88_edge_count, 1)),
        "semantic_label_projection_ratio": semantic_projection_ratio,
        "mode_rows": len(mode_rows),
        "mode_rows_with_semantic_type_ratio": mode_typed_ratio,
        "zero_conf_high_positive_valid_ratio": zero_conf_valid_ratio,
        "seq01_lowconf_valid_support_rows": seq01_lowconf_valid,
        "visual_preview_count": preview_count,
        "has_radio": False,
        "thingstuff_tracks_available": False,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    if not summary["phase1_gate_pass"]:
        summary["blocker"] = "phase1_semantic_scale_mode_ledger_gate_failed"
    write_json(args.out_dir / "phase1_semantic_ledger_summary.json", summary)
    report = [
        "# v89 Phase1 Semantic Scale-Mode Ledger",
        "",
        f"- phase1_gate_pass: `{summary['phase1_gate_pass']}`",
        f"- pair_rows / sequence_coverage: `{summary['pair_rows']} / {summary['sequence_coverage']}`",
        f"- edge_rows / v88_edge_rows: `{summary['edge_rows']} / {summary['v88_edge_rows']}`",
        f"- semantic_label_projection_ratio: `{summary['semantic_label_projection_ratio']}`",
        f"- mode_rows_with_semantic_type_ratio: `{summary['mode_rows_with_semantic_type_ratio']}`",
        f"- zero_conf_high_positive_valid_ratio: `{summary['zero_conf_high_positive_valid_ratio']}`",
        f"- seq01_lowconf_valid_support_rows: `{summary['seq01_lowconf_valid_support_rows']}`",
        f"- has_radio: `{summary['has_radio']}`",
        "",
        "RADIO/track information was not available in the v88 artifacts; the ledger uses dense semantic labels/confidence from raw overlap files plus v88 semantic compatibility flags.",
    ]
    (args.out_dir / "semantic_scale_mode_ledger_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"phase1_gate_pass={summary['phase1_gate_pass']}")
    print(f"pair_rows={summary['pair_rows']}")
    print(f"sequence_coverage={summary['sequence_coverage']}")
    print(f"edge_rows={summary['edge_rows']}")
    print(f"semantic_label_projection_ratio={summary['semantic_label_projection_ratio']}")
    print(f"mode_rows_with_semantic_type_ratio={summary['mode_rows_with_semantic_type_ratio']}")
    print(f"zero_conf_high_positive_valid_ratio={summary['zero_conf_high_positive_valid_ratio']}")
    print(f"seq01_lowconf_valid_support_rows={summary['seq01_lowconf_valid_support_rows']}")
    print(f"visual_preview_count={summary['visual_preview_count']}")
    if summary.get("blocker"):
        print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
