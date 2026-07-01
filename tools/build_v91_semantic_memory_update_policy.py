#!/usr/bin/env python3
"""Build v91 semantic memory update policy states."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from v86_soft_latent_utils import write_csv, write_json
from v91_semantic_regime_utils import ROOT, nseries


DEFAULT_REGIME = ROOT / "phase2_semantic_regime_classifier"
DEFAULT_OUT = ROOT / "phase5_memory_update_policy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime-dir", type=Path, default=DEFAULT_REGIME)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _state(row: pd.Series, q: dict[str, float]) -> str:
    reject_floor = max(float(q["reject_q75"]), 1e-9)
    reset_floor = max(float(q["reset_q90"]), 1e-9)
    delay_floor = max(float(q["delay_q75"]), 1e-9)
    update_floor = max(float(q["update_q75"]), 1e-9)
    abstain_floor = max(float(q["abstain_q75"]), 1e-9)
    invalid_positive = row["S_invalid"] > max(float(q["invalid_q50"]), 1e-9)
    if invalid_positive and row["P_reset_risk"] >= reset_floor:
        return "RESET_RISK"
    if invalid_positive and row["P_reject"] >= reject_floor:
        return "REJECT"
    if str(row.get("regime", "")) == "REGIME_MULTIMODE_CONFLICT" and row["H_mode"] >= q["hmode_q75"]:
        return "DELAY"
    if str(row.get("regime", "")) == "REGIME_BOUNDARY_RICH" and row.get("H_topo", 0.0) > max(q["htopo_q75"], 1e-9) and row["S_context"] <= q["context_q50"]:
        return "DELAY"
    if str(row.get("regime", "")) == "REGIME_LOWOBS_CONTEXT" and row["S_context"] >= q["context_q75"]:
        return "RESET_RISK"
    if row["P_abstain"] >= abstain_floor:
        return "ABSTAIN"
    if row["P_delay"] >= delay_floor:
        return "DELAY"
    if row["P_update"] >= update_floor and row["S_invalid"] <= q["invalid_q50"] and row["S_context"] <= q["context_q50"]:
        return "UPDATE"
    return "HOLD"


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.regime_dir / "semantic_regime_rows.csv")
    hnorm = nseries(df, "H_mode") / max(float(nseries(df, "H_mode").max()), 1e-12)
    df["P_update"] = nseries(df, "O_scale") * nseries(df, "S_valid") * (1 - nseries(df, "S_invalid").clip(0, 1)) * (1 - hnorm.clip(0, 1))
    df["P_reject"] = nseries(df, "S_invalid") * nseries(df, "boundary_mass") * (1 - nseries(df, "S_valid").clip(0, 1))
    df["P_hold"] = (1 - nseries(df, "O_scale").clip(0, 1)) * (1 - nseries(df, "S_valid").clip(0, 1)) * (1 - nseries(df, "S_invalid").clip(0, 1))
    df["P_delay"] = hnorm.clip(0, 1) * nseries(df, "S_valid") * (1 - nseries(df, "S_invalid").clip(0, 1))
    df["P_abstain"] = nseries(df, "S_context") + nseries(df, "S_lowobs")
    df["P_reset_risk"] = nseries(df, "S_invalid") * nseries(df, "boundary_mass") * hnorm.clip(0, 1)
    q = {
        "update_q75": float(df["P_update"].quantile(0.75)),
        "reject_q75": float(df["P_reject"].quantile(0.75)),
        "delay_q75": float(df["P_delay"].quantile(0.75)),
        "abstain_q75": float(df["P_abstain"].quantile(0.75)),
        "reset_q90": float(df["P_reset_risk"].quantile(0.90)),
        "invalid_q50": float(df["S_invalid"].quantile(0.50)),
        "context_q50": float(df["S_context"].quantile(0.50)),
        "context_q75": float(df["S_context"].quantile(0.75)),
        "hmode_q75": float(df["H_mode"].quantile(0.75)),
        "htopo_q75": float(df["H_topo"].quantile(0.75)),
    }
    df["policy_state"] = df.apply(lambda row: _state(row, q), axis=1)
    df["memory_body_hint"] = df["policy_state"].map({"UPDATE": "merge_gauge", "REJECT": "merge_gauge_veto", "DELAY": "merge_gauge_delayed_commit", "HOLD": "read_context", "ABSTAIN": "none", "RESET_RISK": "merge_gauge_risk"})
    df["READ_action_hint"] = df["policy_state"].map({"UPDATE": "scale_evidence_allowed", "HOLD": "context_only", "ABSTAIN": "context_only"}).fillna("not_positive_scale_evidence")
    df["SWA_action_hint"] = df["policy_state"].map({"UPDATE": "pairwise_qk_candidate"}).fillna("blocked")
    df["merge_gauge_action_hint"] = df["policy_state"].map({"UPDATE": "update_candidate", "REJECT": "reject", "DELAY": "delay_commit", "RESET_RISK": "reset_risk"}).fillna("hold")
    df["TTT_write_hint"] = df["policy_state"].map({"UPDATE": "not_allowed_until_carrier_runtime", "REJECT": "one_hop_transient", "RESET_RISK": "one_hop_transient"}).fillna("neutral_or_no_write")
    df["semantic_shuffle_state"] = df["policy_state"].sample(frac=1.0, random_state=91).to_numpy()
    df["component_shuffle_state"] = df["policy_state"].sample(frac=1.0, random_state=92).to_numpy()
    df["regime_shuffle_state"] = df["policy_state"].sample(frac=1.0, random_state=93).to_numpy()
    write_csv(args.out_dir / "policy_state_rows.csv", df.to_dict("records"))
    summary = {
        "phase": "Phase5_semantic_memory_update_policy_build",
        "policy_rows": int(len(df)),
        "sequence_coverage": int(df["seq"].astype(str).str.zfill(2).nunique()),
        "state_counts": df["policy_state"].value_counts().to_dict(),
        "thresholds": q,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    write_json(args.out_dir / "policy_state_summary.json", summary)
    print(f"policy_rows={summary['policy_rows']}")
    print(f"sequence_coverage={summary['sequence_coverage']}")
    print(f"state_counts={summary['state_counts']}")


if __name__ == "__main__":
    main()
