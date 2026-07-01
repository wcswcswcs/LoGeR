#!/usr/bin/env python3
"""Audit v89 semantic observability/update eligibility policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from v86_soft_latent_utils import spearman_rho, stable_hash_float, write_csv, write_json


DEFAULT_DIR = Path("results/acl2_v89tf_semantic_scale_mode_observability_memory_control/phase4_semantic_observability_policy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_DIR)
    return parser.parse_args()


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _shuffle_bool(flags: pd.Series, salt: str) -> pd.Series:
    arr = flags.astype(bool).to_numpy(copy=True)
    out = arr.copy()
    order = sorted(range(len(arr)), key=lambda i: stable_hash_float(salt, i))
    shuffled = arr[order]
    for dst, value in zip(order, shuffled):
        out[dst] = value
    return pd.Series(out, index=flags.index)


def main() -> None:
    args = parse_args()
    rows = pd.read_csv(args.policy_dir / "semantic_observability_policy_rows.csv")
    labelled = rows[_num(rows["abs_log_scale_jump_gt"]).notna()].copy()
    y = _num(labelled["abs_log_scale_jump_gt"])
    threshold = float(y.quantile(0.75)) if len(y) else 0.0
    high = y >= threshold
    bad = labelled["base_case_type"].astype(str).eq("bad")
    good_low = labelled["base_case_type"].astype(str).eq("good") & (~high)
    unsafe = labelled["unsafe_native_update_flag"].astype(bool)
    geometry_score = _num(labelled["geometry_dominant_mode_mu"]).abs().fillna(0.0)
    geom_flag = geometry_score >= geometry_score.quantile(0.75)
    semantic_bad_recall = float((unsafe & (bad | high)).sum() / max(int((bad | high).sum()), 1))
    semantic_good_fpr = float((unsafe & good_low).sum() / max(int(good_low.sum()), 1))
    geom_bad_recall = float((geom_flag & (bad | high)).sum() / max(int((bad | high).sum()), 1))
    geom_good_fpr = float((geom_flag & good_low).sum() / max(int(good_low.sum()), 1))
    good_margin = float(geom_good_fpr - semantic_good_fpr)
    rho = spearman_rho(_num(labelled["O_update"]).tolist(), y.tolist())
    shuffled = _shuffle_bool(unsafe, "phase4_semantic_shuffle")
    shuffle_recall = float((shuffled & (bad | high)).sum() / max(int((bad | high).sum()), 1))
    shuffle_fpr = float((shuffled & good_low).sum() / max(int(good_low.sum()), 1))
    shuffle_rho = spearman_rho(shuffled.astype(float).tolist(), y.tolist())
    unsafe_rho = spearman_rho(unsafe.astype(float).tolist(), y.tolist())
    margin = None if unsafe_rho is None or shuffle_rho is None else float(unsafe_rho - shuffle_rho)
    gate = bool(
        semantic_bad_recall >= 0.60
        and semantic_good_fpr <= 0.25
        and good_margin >= 0.10
        and margin is not None
        and margin >= 0.05
        and labelled["seq"].astype(str).str.zfill(2).nunique() >= 3
    )
    audit = {
        "phase": "Phase4_semantic_observability_policy_audit",
        "semantic_observability_policy_gate_pass": gate,
        "bad_recall": semantic_bad_recall,
        "good_FPR": semantic_good_fpr,
        "geometry_bad_recall": geom_bad_recall,
        "geometry_good_FPR": geom_good_fpr,
        "semantic_good_protection_margin": good_margin,
        "unsafe_policy_rho_abs_log_scale_jump": unsafe_rho,
        "O_update_rho_abs_log_scale_jump": rho,
        "semantic_shuffle_rho": shuffle_rho,
        "semantic_shuffle_margin": margin,
        "semantic_shuffle_bad_recall": shuffle_recall,
        "semantic_shuffle_good_FPR": shuffle_fpr,
        "sequence_coverage": int(labelled["seq"].astype(str).str.zfill(2).nunique()),
        "state_counts": rows["update_state"].value_counts().to_dict(),
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    if not gate:
        audit["blocker"] = "semantic_observability_policy_gate_failed"
    write_json(args.policy_dir / "semantic_observability_policy_audit_summary.json", audit)
    write_csv(
        args.policy_dir / "semantic_observability_policy_controls.csv",
        [
            {"control": "geometry_q75", "bad_recall": geom_bad_recall, "good_FPR": geom_good_fpr},
            {"control": "semantic_shuffle_same_count", "bad_recall": shuffle_recall, "good_FPR": shuffle_fpr, "rho": shuffle_rho},
        ],
    )
    report = [
        "# v89 Phase4 Semantic Observability Policy Audit",
        "",
        f"- gate_pass: `{audit['semantic_observability_policy_gate_pass']}`",
        f"- bad_recall: `{audit['bad_recall']}`",
        f"- good_FPR: `{audit['good_FPR']}`",
        f"- semantic_good_protection_margin: `{audit['semantic_good_protection_margin']}`",
        f"- semantic_shuffle_margin: `{audit['semantic_shuffle_margin']}`",
        f"- state_counts: `{audit['state_counts']}`",
        f"- blocker: `{audit.get('blocker', '')}`",
    ]
    (args.policy_dir / "semantic_observability_policy_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"semantic_observability_policy_gate_pass={audit['semantic_observability_policy_gate_pass']}")
    print(f"bad_recall={audit['bad_recall']}")
    print(f"good_FPR={audit['good_FPR']}")
    print(f"semantic_good_protection_margin={audit['semantic_good_protection_margin']}")
    print(f"semantic_shuffle_margin={audit['semantic_shuffle_margin']}")
    print(f"state_counts={audit['state_counts']}")
    if audit.get("blocker"):
        print(f"blocker={audit['blocker']}")


if __name__ == "__main__":
    main()
