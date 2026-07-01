#!/usr/bin/env python3
"""Build v91 cross-frame semantic topology tracklets from v90 topology artifacts."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v86_soft_latent_utils import write_csv, write_json
from v91_semantic_regime_utils import ROOT, V90_FEATURE, V90_LEDGER, V90_SOURCE, bool_series, normalize_pair_columns, nseries, pair_id


DEFAULT_OUT = ROOT / "phase1_semantic_topology_tracklets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v90-source-dir", type=Path, default=V90_SOURCE)
    parser.add_argument("--v90-ledger-dir", type=Path, default=V90_LEDGER)
    parser.add_argument("--v90-feature-dir", type=Path, default=V90_FEATURE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _node_lookup(nodes: pd.DataFrame) -> dict[tuple[str, int, int, str, int, int], dict[str, Any]]:
    out: dict[tuple[str, int, int, str, int, int], dict[str, Any]] = {}
    for _, row in nodes.iterrows():
        key = (
            str(row["seq"]).zfill(2),
            int(row["prev_chunk"]),
            int(row["curr_chunk"]),
            str(row["side"]),
            int(row["label_compact_id"]),
            int(row["component_id"]),
        )
        out[key] = row.to_dict()
    return out


def _tracklet_type(row: dict[str, Any]) -> str:
    same = bool(row["same_label"])
    conf = float(row["semantic_confidence_mean"])
    boundary = float(row["cross_component_boundary_ratio"])
    zero = float(row["zero_conf_ratio"])
    context = float(row["context_lowobs_ratio"])
    split = float(row["split_merge_score"])
    entropy = float(row["mode_entropy"])
    support = int(row["raw_overlap_support_count"]) + int(row["feature_match_support_count"])
    if entropy >= 3.2:
        return "MULTIMODE_UNSAFE"
    if split >= 0.60:
        return "SPLIT_MERGE_UNSTABLE"
    if int(row["prev_label"]) >= 250 or int(row["curr_label"]) >= 250 or (not same and conf < 0.45 and support < 20):
        return "DYNAMIC_TRANSIENT"
    if context >= 0.45 or zero > 0.05 or support <= 0:
        return "CONTEXT_LOWOBS"
    if boundary >= 0.55 or not same:
        return "INVALID_CROSS_BOUNDARY"
    if same and conf >= 0.55 and float(row["interior_mass_prev"]) >= 0.35 and float(row["interior_mass_curr"]) >= 0.35:
        return "VALID_INTERIOR_SUPPORT"
    if same and support > 0:
        return "VALID_BOUNDARY_SUPPORT"
    return "CONTEXT_LOWOBS"


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    nodes = normalize_pair_columns(pd.read_csv(args.v90_source_dir / "topology_nodes.csv"))
    edges = normalize_pair_columns(pd.read_csv(args.v90_source_dir / "topology_edges.csv"))
    pair_rows = normalize_pair_columns(pd.read_csv(args.v90_ledger_dir / "topology_pair_rows.csv"))
    mode_rows = normalize_pair_columns(pd.read_csv(args.v90_ledger_dir / "topology_mode_rows.csv"))
    feature_pairs = normalize_pair_columns(pd.read_csv(args.v90_feature_dir / "feature_match_topology_pair_summary.csv"))
    lookup = _node_lookup(nodes)
    one_to_many_prev = edges.groupby(["pair_id", "prev_label_compact_id", "prev_component_id"])["curr_component_id"].nunique().to_dict()
    many_to_one_curr = edges.groupby(["pair_id", "curr_label_compact_id", "curr_component_id"])["prev_component_id"].nunique().to_dict()
    pair_info = pair_rows.set_index("pair_id").to_dict("index")
    feat_info = feature_pairs.set_index("pair_id").to_dict("index")
    mode_dominant: dict[str, dict[str, Any]] = {}
    for pid, group in mode_rows.groupby("pair_id"):
        g = group.copy()
        g["_mass"] = pd.to_numeric(g["mode_mass"], errors="coerce").fillna(0.0)
        mode_dominant[pid] = g.sort_values("_mass", ascending=False).iloc[0].to_dict()
    rows: list[dict[str, Any]] = []
    for idx, edge in edges.iterrows():
        seq = str(edge["seq"]).zfill(2)
        prev = int(edge["prev_chunk"])
        curr = int(edge["curr_chunk"])
        pid = pair_id(seq, prev, curr)
        prev_label = int(edge["prev_label_compact_id"])
        curr_label = int(edge["curr_label_compact_id"])
        prev_component = int(edge["prev_component_id"])
        curr_component = int(edge["curr_component_id"])
        pn = lookup.get((seq, prev, curr, "prev", prev_label, prev_component), {})
        cn = lookup.get((seq, prev, curr, "curr", curr_label, curr_component), {})
        pair = pair_info.get(pid, {})
        feat = feat_info.get(pid, {})
        dom = mode_dominant.get(pid, {})
        raw_support = int(float(edge.get("raw_overlap_support_count", 0) or 0))
        feature_support = int(float(edge.get("feature_match_support_count", 0) or 0))
        verified = int(float(feat.get("verified_inlier_count", 0) or 0))
        raw_norm = math.log1p(raw_support)
        match_cov = min(1.0, float(feature_support / max(verified, 1))) if verified else 0.0
        area_prev = int(float(pn.get("patch_count", 0) or 0))
        area_curr = int(float(cn.get("patch_count", 0) or 0))
        boundary_prev = float(pn.get("boundary_ratio", edge.get("prev_boundary_ratio", 1.0)) or 0.0)
        boundary_curr = float(cn.get("boundary_ratio", edge.get("curr_boundary_ratio", 1.0)) or 0.0)
        interior_prev = float(pn.get("interior_ratio", max(0.0, 1.0 - boundary_prev)) or 0.0)
        interior_curr = float(cn.get("interior_ratio", max(0.0, 1.0 - boundary_curr)) or 0.0)
        split_prev = one_to_many_prev.get((pid, prev_label, prev_component), 1)
        split_curr = many_to_one_curr.get((pid, curr_label, curr_component), 1)
        split_score = min(1.0, max(split_prev - 1, split_curr - 1) / 4.0)
        local_shape_mu = float(pair.get("topology_valid_dominant_mode_mu", dom.get("mode_center_mu", 0.0)) or 0.0)
        if bool(edge.get("cross_component_boundary", False)) or prev_label != curr_label:
            local_shape_mu = float(pair.get("topology_invalid_dominant_mode_mu", local_shape_mu) or local_shape_mu)
        mode_centers = mode_rows[mode_rows["pair_id"] == pid]
        signed_mode_id = -1
        local_shape_mad = float(dom.get("mode_mad", 0.0) or 0.0)
        if len(mode_centers):
            diffs = (pd.to_numeric(mode_centers["mode_center_mu"], errors="coerce").fillna(0.0) - local_shape_mu).abs()
            nearest = mode_centers.loc[diffs.idxmin()]
            signed_mode_id = int(nearest["mode_id"])
            local_shape_mad = float(nearest.get("mode_mad", local_shape_mad) or local_shape_mad)
        row: dict[str, Any] = {
            "seq": seq,
            "prev_chunk": prev,
            "curr_chunk": curr,
            "pair_id": pid,
            "tracklet_id": f"{pid}_T{idx:05d}",
            "prev_component_id": prev_component,
            "curr_component_id": curr_component,
            "prev_label": prev_label,
            "curr_label": curr_label,
            "same_label": bool(prev_label == curr_label),
            "same_role_proxy": bool(prev_label == curr_label and prev_label > 0),
            "component_area_prev": area_prev,
            "component_area_curr": area_curr,
            "area_ratio": float((area_curr + 1e-6) / (area_prev + 1e-6)),
            "interior_mass_prev": interior_prev,
            "interior_mass_curr": interior_curr,
            "boundary_mass_prev": boundary_prev,
            "boundary_mass_curr": boundary_curr,
            "cross_component_boundary_ratio": float(max(float(edge.get("cross_component_boundary", False) in [True, "true", "True"]), boundary_prev * boundary_curr if prev_label != curr_label else 0.0, float(edge.get("prev_boundary_ratio", 0.0) or 0.0) * float(edge.get("curr_boundary_ratio", 0.0) or 0.0))),
            "raw_overlap_support_count": raw_support,
            "feature_match_support_count": feature_support,
            "verified_inlier_count": verified,
            "match_spatial_coverage": match_cov,
            "local_shape_mu": local_shape_mu,
            "local_shape_mad": local_shape_mad,
            "signed_mode_id": signed_mode_id,
            "mode_entropy": float(pair.get("geometry_mode_entropy", dom.get("mode_entropy", 0.0)) or 0.0),
            "semantic_confidence_mean": float(edge.get("mean_semantic_conf", 0.0) or 0.0),
            "zero_conf_ratio": max(0.0, 1.0 - float(edge.get("mean_semantic_conf", 0.0) or 0.0)),
            "context_lowobs_ratio": float(pair.get("topology_context_mass", 0.0) or 0.0) + float(pair.get("topology_lowobs_mass", 0.0) or 0.0),
            "split_merge_score": split_score,
            "label_mapping_status": "compact_project_local_id_no_class_names",
            "has_radio": False,
            "has_track": False,
            "source_path": edge.get("source_path", pair.get("source_path", "")),
            "abs_log_scale_jump_gt": pair.get("abs_log_scale_jump_gt", ""),
            "base_case_type": pair.get("base_case_type", ""),
            "offline_audit_label_only": True,
            "no_gt_runtime_feature": True,
        }
        row["tracklet_type"] = _tracklet_type(row)
        rows.append(row)
    out = pd.DataFrame(rows)
    write_csv(args.out_dir / "semantic_topology_tracklet_rows.csv", out.to_dict("records"))
    type_counts = out["tracklet_type"].value_counts().reset_index()
    type_counts.columns = ["tracklet_type", "count"]
    write_csv(args.out_dir / "semantic_topology_tracklet_type_counts.csv", type_counts.to_dict("records"))
    pair_summary = (
        out.groupby(["seq", "prev_chunk", "curr_chunk", "pair_id"], as_index=False)
        .agg(
            tracklet_rows=("tracklet_id", "count"),
            raw_overlap_support_count=("raw_overlap_support_count", "sum"),
            feature_match_support_count=("feature_match_support_count", "sum"),
            semantic_confidence_mean=("semantic_confidence_mean", "mean"),
            valid_tracklet_ratio=("tracklet_type", lambda s: float(s.astype(str).str.startswith("VALID").mean())),
            invalid_tracklet_ratio=("tracklet_type", lambda s: float(s.astype(str).isin(["INVALID_CROSS_BOUNDARY", "DYNAMIC_TRANSIENT", "SPLIT_MERGE_UNSTABLE"]).mean())),
            context_lowobs_ratio=("context_lowobs_ratio", "mean"),
            mode_entropy=("mode_entropy", "mean"),
            split_merge_score=("split_merge_score", "mean"),
        )
    )
    base_cols = ["pair_id", "abs_log_scale_jump_gt", "base_case_type", "geometry_dominant_mode_mu", "geometry_mode_entropy", "observability_score", "source_path"]
    pair_summary = pair_summary.merge(pair_rows[base_cols], on="pair_id", how="left")
    write_csv(args.out_dir / "semantic_topology_tracklet_pair_summary.csv", pair_summary.to_dict("records"))
    summary = {
        "phase": "Phase1_semantic_topology_tracklets",
        "pair_rows": int(pair_rows["pair_id"].nunique()),
        "tracklet_rows": int(len(out)),
        "sequence_coverage": int(out["seq"].nunique()) if len(out) else 0,
        "pairs_with_tracklets": int(pair_summary[pair_summary["tracklet_rows"] > 0]["pair_id"].nunique()) if len(pair_summary) else 0,
        "pairs_with_tracklets_ratio": float((pair_summary["tracklet_rows"] > 0).mean()) if len(pair_summary) else 0.0,
        "raw_or_match_backed_tracklet_ratio": float(((out["raw_overlap_support_count"] > 0) | (out["feature_match_support_count"] > 0)).mean()) if len(out) else 0.0,
        "semantic_confidence_available_ratio": float((out["semantic_confidence_mean"] > 0).mean()) if len(out) else 0.0,
        "tracklet_type_coverage_ratio": float(out["tracklet_type"].astype(str).ne("").mean()) if len(out) else 0.0,
        "zero_conf_positive_tracklet_ratio": float(((out["zero_conf_ratio"] > 0.05) & (out["tracklet_type"].astype(str).str.startswith("VALID"))).mean()) if len(out) else 0.0,
        "label_mapping_status": "compact_project_local_id_no_class_names",
        "has_radio": False,
        "has_track": False,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    summary["phase1_tracklet_gate_pass"] = bool(
        summary["pair_rows"] >= 49
        and summary["sequence_coverage"] >= 4
        and summary["pairs_with_tracklets_ratio"] >= 0.90
        and summary["tracklet_rows"] >= summary["pair_rows"] * 5
        and summary["raw_or_match_backed_tracklet_ratio"] >= 0.80
        and summary["semantic_confidence_available_ratio"] >= 0.90
        and summary["tracklet_type_coverage_ratio"] >= 0.90
        and summary["zero_conf_positive_tracklet_ratio"] <= 0.05
    )
    if not summary["phase1_tracklet_gate_pass"]:
        summary["blocker"] = "semantic_topology_tracklet_gate_failed"
    write_json(args.out_dir / "semantic_topology_tracklet_summary.json", summary)
    print(f"phase1_tracklet_gate_pass={summary['phase1_tracklet_gate_pass']}")
    print(f"pair_rows={summary['pair_rows']}")
    print(f"tracklet_rows={summary['tracklet_rows']}")
    print(f"sequence_coverage={summary['sequence_coverage']}")
    print(f"pairs_with_tracklets_ratio={summary['pairs_with_tracklets_ratio']}")
    print(f"raw_or_match_backed_tracklet_ratio={summary['raw_or_match_backed_tracklet_ratio']}")
    print(f"semantic_confidence_available_ratio={summary['semantic_confidence_available_ratio']}")
    print(f"tracklet_type_coverage_ratio={summary['tracklet_type_coverage_ratio']}")
    print(f"zero_conf_positive_tracklet_ratio={summary['zero_conf_positive_tracklet_ratio']}")
    if summary.get("blocker"):
        print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
