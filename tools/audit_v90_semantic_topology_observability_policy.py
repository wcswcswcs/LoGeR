#!/usr/bin/env python3
"""Audit v90 topology observability policy against offline labels and controls."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from v86_soft_latent_utils import spearman_rho, write_csv, write_json
from v90_semantic_topology_utils import ROOT, stable_shuffle


DEFAULT_OUT = ROOT / "phase4_semantic_topology_observability_policy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df.get(col, pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)


def _metric(labelled: pd.DataFrame, score: pd.Series, state: pd.Series) -> dict[str, float | None]:
    y = _num(labelled, "abs_log_scale_jump_gt")
    high = y >= float(y.quantile(0.75))
    bad = labelled["base_case_type"].astype(str).eq("bad")
    good_low = labelled["base_case_type"].astype(str).eq("good") & (~high)
    update_like = state.astype(str).isin(["UPDATE", "REJECT", "RESET_RISK"])
    bad_or_high = bad | high
    return {
        "rho": spearman_rho(score.tolist(), y.tolist()),
        "bad_recall": float((update_like & bad_or_high).sum() / max(int(bad_or_high.sum()), 1)),
        "good_fpr": float((update_like & good_low).sum() / max(int(good_low.sum()), 1)),
    }


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.out_dir / "topology_observability_policy_rows.csv")
    df["seq"] = df["seq"].astype(str).str.zfill(2)
    labelled = df[pd.to_numeric(df["abs_log_scale_jump_gt"], errors="coerce").notna()].copy()
    topo = _metric(labelled, _num(labelled, "O_update_topology"), labelled["policy_state"])
    geom_score = _num(labelled, "geometry_dominant_mode_mu").abs()
    y = _num(labelled, "abs_log_scale_jump_gt")
    geom_flags = geom_score >= float(geom_score.quantile(0.75))
    high = y >= float(y.quantile(0.75))
    bad = labelled["base_case_type"].astype(str).eq("bad")
    good_low = labelled["base_case_type"].astype(str).eq("good") & (~high)
    geom_bad_recall = float((geom_flags & (bad | high)).sum() / max(int((bad | high).sum()), 1))
    geom_good_fpr = float((geom_flags & good_low).sum() / max(int(good_low.sum()), 1))
    sem_ctrl_score = stable_shuffle(_num(labelled, "O_update_topology"), "v90_phase4_semantic_shuffle")
    comp_ctrl_score = stable_shuffle(_num(labelled, "O_update_topology"), "v90_phase4_component_shuffle")
    sem_ctrl = _metric(labelled, sem_ctrl_score, labelled["policy_state"])
    comp_ctrl = _metric(labelled, comp_ctrl_score, labelled["policy_state"])
    semantic_margin = float(topo["good_fpr"] - sem_ctrl["good_fpr"]) if topo["good_fpr"] is not None and sem_ctrl["good_fpr"] is not None else None
    component_margin = float(topo["good_fpr"] - comp_ctrl["good_fpr"]) if topo["good_fpr"] is not None and comp_ctrl["good_fpr"] is not None else None
    good_protection_margin = float(geom_good_fpr - topo["good_fpr"])
    # For policy safety, positive margin means lower FPR than shuffled controls.
    semantic_good_margin = float(sem_ctrl["good_fpr"] - topo["good_fpr"])
    component_good_margin = float(comp_ctrl["good_fpr"] - topo["good_fpr"])
    gate = bool(
        topo["bad_recall"] >= 0.55
        and topo["good_fpr"] <= 0.25
        and good_protection_margin >= 0.15
        and semantic_good_margin >= 0.05
        and component_good_margin >= 0.05
        and int(labelled["seq"].nunique()) >= 3
    )
    controls = [
        {"control": "geometry_q75", "bad_recall": geom_bad_recall, "good_fpr": geom_good_fpr},
        {"control": "semantic_shuffle", "bad_recall": sem_ctrl["bad_recall"], "good_fpr": sem_ctrl["good_fpr"], "rho": sem_ctrl["rho"]},
        {"control": "component_shuffle", "bad_recall": comp_ctrl["bad_recall"], "good_fpr": comp_ctrl["good_fpr"], "rho": comp_ctrl["rho"]},
    ]
    summary = {
        "phase": "Phase4_semantic_topology_observability_policy_audit",
        "semantic_topology_observability_policy_gate_pass": gate,
        "O_update_topology_rho_abs_log_scale_jump": topo["rho"],
        "bad_recall": topo["bad_recall"],
        "good_FPR": topo["good_fpr"],
        "geometry_bad_recall": geom_bad_recall,
        "geometry_good_FPR": geom_good_fpr,
        "semantic_good_protection_margin": good_protection_margin,
        "semantic_shuffle_good_margin": semantic_good_margin,
        "component_shuffle_good_margin": component_good_margin,
        "semantic_shuffle_margin": semantic_good_margin,
        "component_shuffle_margin": component_good_margin,
        "state_counts": df["policy_state"].value_counts().to_dict(),
        "sequence_coverage": int(labelled["seq"].nunique()),
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    if not gate:
        summary["blocker"] = "semantic_topology_observability_policy_gate_failed"
    write_csv(args.out_dir / "topology_observability_policy_controls.csv", controls)
    write_json(args.out_dir / "topology_observability_policy_audit_summary.json", summary)
    report = [
        "# v90 Phase4 Semantic Topology Observability Policy",
        "",
        f"- gate_pass: `{gate}`",
        f"- bad_recall: `{topo['bad_recall']}`",
        f"- good_FPR: `{topo['good_fpr']}`",
        f"- semantic_good_protection_margin: `{good_protection_margin}`",
        f"- semantic_shuffle_margin: `{semantic_good_margin}`",
        f"- component_shuffle_margin: `{component_good_margin}`",
    ]
    if summary.get("blocker"):
        report.append(f"- blocker: `{summary['blocker']}`")
    (args.out_dir / "topology_observability_policy_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"semantic_topology_observability_policy_gate_pass={summary['semantic_topology_observability_policy_gate_pass']}")
    print(f"bad_recall={summary['bad_recall']}")
    print(f"good_FPR={summary['good_FPR']}")
    print(f"semantic_good_protection_margin={summary['semantic_good_protection_margin']}")
    print(f"semantic_shuffle_margin={summary['semantic_shuffle_margin']}")
    print(f"component_shuffle_margin={summary['component_shuffle_margin']}")
    if summary.get("blocker"):
        print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
