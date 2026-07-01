#!/usr/bin/env python3
"""Audit v91 semantic tracklet mode disambiguation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from v86_soft_latent_utils import spearman_rho, write_csv, write_json
from v91_semantic_regime_utils import ROOT, nseries, stable_shuffle


DEFAULT_TRACKLETS = ROOT / "phase1_semantic_topology_tracklets"
DEFAULT_OUT = ROOT / "phase4_tracklet_mode_disambiguation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracklet-dir", type=Path, default=DEFAULT_TRACKLETS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _entropy(counts: pd.Series) -> float:
    p = counts.astype(float)
    p = p / max(float(p.sum()), 1e-12)
    p = p[p > 0]
    return float(-(p * np.log(p + 1e-12)).sum()) if len(p) else 0.0


def _recall_fpr(labelled: pd.DataFrame, values: pd.Series) -> tuple[float, float]:
    score = pd.to_numeric(values.loc[labelled.index], errors="coerce").fillna(0.0)
    y = pd.to_numeric(labelled["abs_log_scale_jump_gt"], errors="coerce")
    threshold = float(score.quantile(0.75)) if len(score) else 0.0
    flags = score >= threshold
    high = y >= float(y.quantile(0.75)) if len(y) else pd.Series(False, index=labelled.index)
    bad = labelled["base_case_type"].astype(str).eq("bad")
    good_low = labelled["base_case_type"].astype(str).eq("good") & (~high)
    bad_recall = float((flags & (bad | high)).sum() / max(int((bad | high).sum()), 1)) if len(labelled) else 0.0
    good_fpr = float((flags & good_low).sum() / max(int(good_low.sum()), 1)) if len(labelled) else 1.0
    return bad_recall, good_fpr


def _balanced(bad_recall: float, good_fpr: float) -> float:
    return float(0.5 * (bad_recall + 1.0 - good_fpr))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(args.tracklet_dir / "semantic_topology_tracklet_rows.csv")
    pair = pd.read_csv(args.tracklet_dir / "semantic_topology_tracklet_pair_summary.csv")
    rows = []
    for pid, group in tr.groupby("pair_id"):
        base = pair[pair["pair_id"] == pid].iloc[0].to_dict()
        topo_entropy = _entropy(group["tracklet_type"].value_counts())
        geom_entropy = float(base.get("geometry_mode_entropy", group["mode_entropy"].mean()) or 0.0)
        type_counts = group["tracklet_type"].value_counts(normalize=True)
        valid_purity = float(type_counts[[idx for idx in type_counts.index if str(idx).startswith("VALID")]].sum()) if len(type_counts) else 0.0
        invalid_purity = float(type_counts[[idx for idx in type_counts.index if str(idx) in {"INVALID_CROSS_BOUNDARY", "DYNAMIC_TRANSIENT", "SPLIT_MERGE_UNSTABLE"}]].sum()) if len(type_counts) else 0.0
        rows.append(
            {
                "seq": base.get("seq"),
                "prev_chunk": base.get("prev_chunk"),
                "curr_chunk": base.get("curr_chunk"),
                "pair_id": pid,
                "geometry_mode_entropy": geom_entropy,
                "semantic_tracklet_mode_entropy": topo_entropy,
                "entropy_reduction": geom_entropy - topo_entropy,
                "valid_mode_purity": valid_purity,
                "invalid_mode_purity": invalid_purity,
                "context_mode_mass": float(type_counts.get("CONTEXT_LOWOBS", 0.0)),
                "split_merge_mode_mass": float(type_counts.get("SPLIT_MERGE_UNSTABLE", 0.0)),
                "same_component_mode_consensus": float(group["same_label"].astype(str).str.lower().isin(["true", "1"]).mean()),
                "feature_match_mode_consensus": float((pd.to_numeric(group["feature_match_support_count"], errors="coerce").fillna(0) > 0).mean()),
                "tracklet_mode_score": invalid_purity + max(geom_entropy - topo_entropy, 0.0) - 0.25 * valid_purity,
                "abs_log_scale_jump_gt": base.get("abs_log_scale_jump_gt", ""),
                "base_case_type": base.get("base_case_type", ""),
                "offline_audit_label_only": True,
            }
        )
    out = pd.DataFrame(rows)
    labelled = out[pd.to_numeric(out["abs_log_scale_jump_gt"], errors="coerce").notna()].copy()
    score = nseries(labelled, "tracklet_mode_score")
    y = pd.to_numeric(labelled["abs_log_scale_jump_gt"], errors="coerce")
    rho = spearman_rho(score.tolist(), y.tolist()) if len(labelled) else None
    geom_score = nseries(labelled, "geometry_mode_entropy")
    geom_rho = spearman_rho(geom_score.tolist(), y.tolist()) if len(labelled) else None
    bad_recall, good_fpr = _recall_fpr(labelled, out["tracklet_mode_score"])
    entropy_reduction_mean = float(pd.to_numeric(out["entropy_reduction"], errors="coerce").mean()) if len(out) else 0.0
    sem_bad, sem_fpr = _recall_fpr(labelled, stable_shuffle(out["tracklet_mode_score"], "v91_phase4_semantic_shuffle"))
    comp_bad, comp_fpr = _recall_fpr(labelled, stable_shuffle(out["tracklet_mode_score"], "v91_phase4_component_shuffle"))
    sem_margin = _balanced(bad_recall, good_fpr) - _balanced(sem_bad, sem_fpr)
    comp_margin = _balanced(bad_recall, good_fpr) - _balanced(comp_bad, comp_fpr)
    gate = bool(
        entropy_reduction_mean >= 0.10
        and ((rho is not None and geom_rho is not None and rho >= geom_rho + 0.05) or good_fpr <= 0.25)
        and sem_margin >= 0.05
        and comp_margin >= 0.05
        and int(out["seq"].astype(str).str.zfill(2).nunique()) >= 3
    )
    summary = {
        "phase": "Phase4_tracklet_mode_disambiguation",
        "phase4_tracklet_mode_gate_pass": gate,
        "pair_rows": int(len(out)),
        "sequence_coverage": int(out["seq"].astype(str).str.zfill(2).nunique()) if len(out) else 0,
        "mean_entropy_reduction": entropy_reduction_mean,
        "tracklet_mode_rho_abs_scale_jump": rho,
        "geometry_mode_rho": geom_rho,
        "tracklet_mode_bad_recall": bad_recall,
        "tracklet_mode_good_FPR": good_fpr,
        "semantic_shuffle_tracklet_score_margin": sem_margin,
        "component_shuffle_tracklet_score_margin": comp_margin,
        "semantic_shuffle_bad_recall": sem_bad,
        "semantic_shuffle_good_FPR": sem_fpr,
        "component_shuffle_bad_recall": comp_bad,
        "component_shuffle_good_FPR": comp_fpr,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    if not gate:
        summary["blocker"] = "tracklet_mode_disambiguation_gate_failed"
    write_csv(args.out_dir / "tracklet_mode_disambiguation_rows.csv", out.to_dict("records"))
    write_json(args.out_dir / "tracklet_mode_disambiguation_summary.json", summary)
    print(f"phase4_tracklet_mode_gate_pass={summary['phase4_tracklet_mode_gate_pass']}")
    print(f"mean_entropy_reduction={summary['mean_entropy_reduction']}")
    print(f"tracklet_mode_rho_abs_scale_jump={summary['tracklet_mode_rho_abs_scale_jump']}")
    print(f"geometry_mode_rho={summary['geometry_mode_rho']}")
    print(f"tracklet_mode_bad_recall={summary['tracklet_mode_bad_recall']}")
    print(f"tracklet_mode_good_FPR={summary['tracklet_mode_good_FPR']}")
    print(f"semantic_shuffle_tracklet_score_margin={summary['semantic_shuffle_tracklet_score_margin']}")
    print(f"component_shuffle_tracklet_score_margin={summary['component_shuffle_tracklet_score_margin']}")
    if summary.get("blocker"):
        print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
