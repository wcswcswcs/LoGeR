#!/usr/bin/env python3
"""Audit v89 delayed commit policy diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from v86_soft_latent_utils import spearman_rho, stable_hash_float, write_csv, write_json


DEFAULT_DIR = Path("results/acl2_v89tf_semantic_scale_mode_observability_memory_control/phase7_semantic_mode_temporal_consistency")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-dir", type=Path, default=DEFAULT_DIR)
    return parser.parse_args()


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _shuffle(flags: pd.Series) -> pd.Series:
    arr = flags.astype(bool).to_numpy(copy=True)
    out = arr.copy()
    order = sorted(range(len(arr)), key=lambda i: stable_hash_float("phase7_delayed_commit_shuffle", i))
    vals = arr[order]
    for dst, value in zip(order, vals):
        out[dst] = value
    return pd.Series(out, index=flags.index)


def main() -> None:
    args = parse_args()
    rows = pd.read_csv(args.temporal_dir / "semantic_mode_temporal_consistency_rows.csv")
    labelled = rows[_num(rows["abs_log_scale_jump_gt"]).notna()].copy()
    y = _num(labelled["abs_log_scale_jump_gt"])
    high = y >= (float(y.quantile(0.75)) if len(y) else 0.0)
    bad = labelled["base_case_type"].astype(str).eq("bad")
    good_low = labelled["base_case_type"].astype(str).eq("good") & (~high)
    delayed = labelled["update_state"].astype(str).eq("DELAY_COMMIT") | labelled["semantic_valid_mode_persistence"].astype(bool)
    recall = float((delayed & (bad | high)).sum() / max(int((bad | high).sum()), 1))
    fpr = float((delayed & good_low).sum() / max(int(good_low.sum()), 1))
    rho = spearman_rho(delayed.astype(float).tolist(), y.tolist())
    shuf = _shuffle(delayed)
    shuf_rho = spearman_rho(shuf.astype(float).tolist(), y.tolist())
    margin = None if rho is None or shuf_rho is None else float(rho - shuf_rho)
    gate = bool(recall >= 0.60 and fpr <= 0.25 and margin is not None and margin >= 0.05 and labelled["seq"].astype(str).str.zfill(2).nunique() >= 3)
    audit = {
        "phase": "Phase7_delayed_commit_policy_audit",
        "delayed_commit_policy_gate_pass": gate,
        "bad_recall": recall,
        "good_FPR": fpr,
        "semantic_shuffle_rho": shuf_rho,
        "delayed_commit_rho": rho,
        "semantic_shuffle_margin": margin,
        "sequence_coverage": int(labelled["seq"].astype(str).str.zfill(2).nunique()) if len(labelled) else 0,
        "delayed_or_persistent_rows": int(delayed.sum()),
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    if not gate:
        audit["blocker"] = "delayed_commit_policy_gate_failed"
    write_json(args.temporal_dir / "delayed_commit_policy_audit_summary.json", audit)
    write_csv(args.temporal_dir / "delayed_commit_policy_controls.csv", [{"control": "semantic_shuffle", "rho": shuf_rho}])
    print(f"delayed_commit_policy_gate_pass={audit['delayed_commit_policy_gate_pass']}")
    print(f"bad_recall={audit['bad_recall']}")
    print(f"good_FPR={audit['good_FPR']}")
    print(f"semantic_shuffle_margin={audit['semantic_shuffle_margin']}")
    print(f"delayed_or_persistent_rows={audit['delayed_or_persistent_rows']}")
    if audit.get("blocker"):
        print(f"blocker={audit['blocker']}")


if __name__ == "__main__":
    main()
