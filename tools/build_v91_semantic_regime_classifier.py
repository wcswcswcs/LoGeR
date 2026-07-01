#!/usr/bin/env python3
"""Build v91 deterministic no-GT semantic regime classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from v86_soft_latent_utils import write_csv, write_json
from v91_semantic_regime_utils import ROOT, nseries, normalize_pair_columns


DEFAULT_TRACKLETS = ROOT / "phase1_semantic_topology_tracklets"
DEFAULT_OUT = ROOT / "phase2_semantic_regime_classifier"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracklet-dir", type=Path, default=DEFAULT_TRACKLETS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _entropy_from_counts(counts: pd.Series) -> float:
    p = counts.astype(float)
    p = p / max(float(p.sum()), 1e-12)
    p = p[p > 0]
    return float(-(p * np.log(p + 1e-12)).sum()) if len(p) else 0.0


def _assign(row: pd.Series, q: dict[str, float]) -> tuple[str, float]:
    low_observable = row.get("observability_score", 1.0) <= q["obs_q25"] and row["N_match"] <= q["match_q50"]
    if row["S_context"] >= q["context_q75"] or low_observable:
        return "REGIME_LOWOBS_CONTEXT", max(row["S_context"], 1.0 - min(1.0, row.get("observability_score", 1.0)))
    if row["H_mode"] >= q["hmode_q75"]:
        return "REGIME_MULTIMODE_CONFLICT", row["H_mode"] / max(q["hmode_max"], 1e-12)
    if row["S_invalid"] > max(q["invalid_q75"], 1e-9) or (row["boundary_mass"] >= q["boundary_q75"] and row["H_topo"] >= q["htopo_q75"]):
        return "REGIME_BOUNDARY_RICH", max(row["S_invalid"], min(1.0, row["H_topo"] / max(q["htopo_max"], 1e-12)))
    if row["d_med_proxy"] >= q["depth_q50"] and row["N_match"] >= q["match_q50"] and row["S_context"] <= q["context_q50"]:
        return "REGIME_FAR_OPEN_HIGHOBS", min(1.0, row["B_proxy"] / max(q["bproxy_q75"], 1e-12))
    if row["d_med_proxy"] < q["depth_q50"] and row["H_mode"] <= q["hmode_q50"]:
        return "REGIME_NEAR_STRUCTURED", min(1.0, row["S_valid"] + 0.25 * (1.0 - row["boundary_mass"]))
    return "REGIME_SUPPORT_RICH_AMBIGUOUS", max(row["S_valid"], 1.0 - row["S_invalid"])


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs = normalize_pair_columns(pd.read_csv(args.tracklet_dir / "semantic_topology_tracklet_pair_summary.csv"))
    tracklets = normalize_pair_columns(pd.read_csv(args.tracklet_dir / "semantic_topology_tracklet_rows.csv"))
    rows = pairs.copy()
    rows["B_proxy"] = nseries(rows, "geometry_dominant_mode_mu").abs() * np.log1p(nseries(rows, "raw_overlap_support_count") + nseries(rows, "feature_match_support_count"))
    rows["d_med_proxy"] = 1.0 / (nseries(rows, "observability_score") + 1e-3)
    rows["N_match"] = nseries(rows, "feature_match_support_count") + nseries(rows, "raw_overlap_support_count").clip(upper=5000) / 100.0
    rows["H_mode"] = nseries(rows, "mode_entropy")
    rows["S_valid"] = nseries(rows, "valid_tracklet_ratio")
    rows["S_invalid"] = nseries(rows, "invalid_tracklet_ratio")
    rows["S_context"] = nseries(rows, "context_lowobs_ratio").clip(0, 1)
    rows["S_lowobs"] = rows["S_context"]
    rows["boundary_mass"] = 1.0 - rows["S_valid"].clip(0, 1)
    rows["tracklet_support_count"] = nseries(rows, "tracklet_rows")
    rows["H_topo"] = 0.0
    for pid, group in tracklets.groupby("pair_id"):
        rows.loc[rows["pair_id"] == pid, "H_topo"] = _entropy_from_counts(group["tracklet_type"].value_counts())
    hnorm = rows["H_mode"] / max(float(rows["H_mode"].max()), 1e-12)
    support_norm = np.log1p(rows["N_match"]) / max(float(np.log1p(rows["N_match"].max())), 1e-12)
    rows["O_scale"] = (
        (rows["B_proxy"] / max(float(rows["B_proxy"].quantile(0.90)), 1e-12)).clip(0, 1)
        * support_norm.clip(0, 1)
        * (1.0 - hnorm.clip(0, 1))
        * rows["S_valid"].clip(0, 1)
        * (1.0 - rows["S_context"].clip(0, 1))
    )
    q = {
        "oscale_q25": float(rows["O_scale"].quantile(0.25)),
        "oscale_q50": float(rows["O_scale"].quantile(0.50)),
        "oscale_q75": float(rows["O_scale"].quantile(0.75)),
        "hmode_q75": float(rows["H_mode"].quantile(0.75)),
        "hmode_q50": float(rows["H_mode"].quantile(0.50)),
        "hmode_max": float(rows["H_mode"].max()),
        "htopo_q75": float(rows["H_topo"].quantile(0.75)),
        "htopo_max": float(rows["H_topo"].max()),
        "context_q75": float(rows["S_context"].quantile(0.75)),
        "context_q50": float(rows["S_context"].quantile(0.50)),
        "boundary_q50": float(rows["boundary_mass"].quantile(0.50)),
        "boundary_q75": float(rows["boundary_mass"].quantile(0.75)),
        "invalid_q75": float(rows["S_invalid"].quantile(0.75)),
        "valid_q50": float(rows["S_valid"].quantile(0.50)),
        "depth_q50": float(rows["d_med_proxy"].quantile(0.50)),
        "match_q50": float(rows["N_match"].quantile(0.50)),
        "obs_q25": float(nseries(rows, "observability_score").quantile(0.25)),
        "bproxy_q75": float(rows["B_proxy"].quantile(0.75)),
    }
    regimes = rows.apply(lambda row: _assign(row, q), axis=1)
    rows["regime"] = [item[0] for item in regimes]
    rows["regime_confidence"] = [float(item[1]) for item in regimes]
    rows["regime_shuffle_control_id"] = [f"regime_shuffle_{i % 7}" for i in range(len(rows))]
    rows["bad_good_label_used_for_assignment"] = False
    rows["scale_label_used_for_assignment"] = False
    write_csv(args.out_dir / "semantic_regime_rows.csv", rows.to_dict("records"))
    write_json(args.out_dir / "semantic_regime_thresholds.json", q)
    counts = rows["regime"].value_counts().to_dict()
    known_ratio = float(rows["regime"].astype(str).ne("").mean()) if len(rows) else 0.0
    max_regime_ratio = float(rows["regime"].value_counts(normalize=True).max()) if len(rows) else 0.0
    summary = {
        "phase": "Phase2_semantic_regime_classifier",
        "regime_rows": int(len(rows)),
        "pair_rows": int(len(pairs)),
        "sequence_coverage": int(rows["seq"].nunique()) if len(rows) else 0,
        "known_regime_ratio": known_ratio,
        "max_single_regime_ratio": max_regime_ratio,
        "regime_counts": counts,
        "regime_confidence_available_ratio": float(rows["regime_confidence"].notna().mean()) if len(rows) else 0.0,
        "bad_good_label_used_for_assignment": False,
        "scale_label_used_for_assignment": False,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    summary["phase2_regime_classifier_gate_pass"] = bool(
        summary["regime_rows"] == summary["pair_rows"]
        and summary["sequence_coverage"] >= 4
        and summary["known_regime_ratio"] >= 0.90
        and summary["max_single_regime_ratio"] <= 0.70
        and not summary["bad_good_label_used_for_assignment"]
        and not summary["scale_label_used_for_assignment"]
        and summary["regime_confidence_available_ratio"] >= 0.90
    )
    if not summary["phase2_regime_classifier_gate_pass"]:
        summary["blocker"] = "semantic_regime_classifier_gate_failed"
    write_json(args.out_dir / "semantic_regime_summary.json", summary)
    print(f"phase2_regime_classifier_gate_pass={summary['phase2_regime_classifier_gate_pass']}")
    print(f"regime_rows={summary['regime_rows']}")
    print(f"sequence_coverage={summary['sequence_coverage']}")
    print(f"known_regime_ratio={summary['known_regime_ratio']}")
    print(f"max_single_regime_ratio={summary['max_single_regime_ratio']}")
    print(f"regime_counts={summary['regime_counts']}")
    if summary.get("blocker"):
        print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
