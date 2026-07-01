#!/usr/bin/env python3
"""Build v90 topology observability and memory-control policy rows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v86_soft_latent_utils import write_csv, write_json
from v90_semantic_topology_utils import ROOT


DEFAULT_LEDGER = ROOT / "phase2_semantic_topology_scale_mode_ledger"
DEFAULT_OUT = ROOT / "phase4_semantic_topology_observability_policy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-dir", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df.get(col, pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)


def _state(row: pd.Series, q: dict[str, float]) -> str:
    valid = float(row["topology_valid_mass"])
    invalid = float(row["topology_invalid_mass"])
    context = float(row["topology_context_mass"])
    lowobs = float(row["topology_lowobs_mass"])
    boundary = float(row["topology_boundary_conflict"])
    entropy = float(row["geometry_mode_entropy"])
    osem = float(row["O_topology_scale"])
    native_mismatch = abs(float(row["native_delta_log_scale"]) - float(row["topology_valid_dominant_mode_mu"]))
    if invalid >= q["invalid_q90"] or (boundary >= q["boundary_q90"] and invalid >= q["invalid_q75"]):
        return "REJECT"
    if lowobs >= q["lowobs_q75"] or context >= q["context_q75"]:
        return "HOLD"
    if valid >= q["valid_q75"] and invalid <= q["invalid_q50"] and boundary <= q["boundary_q50"] and entropy <= q["entropy_q50"] and osem >= q["otopo_q50"]:
        if native_mismatch >= q["mismatch_q75"]:
            return "RESET_RISK"
        return "UPDATE"
    if valid >= q["valid_q50"] and entropy > q["entropy_q50"]:
        return "DELAY"
    return "ABSTAIN"


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.ledger_dir / "topology_pair_rows.csv")
    df["seq"] = df["seq"].astype(str).str.zfill(2)
    for col in [
        "topology_valid_mass",
        "topology_invalid_mass",
        "topology_context_mass",
        "topology_lowobs_mass",
        "topology_boundary_conflict",
        "geometry_mode_entropy",
        "O_topology_scale",
        "native_delta_log_scale",
        "topology_valid_dominant_mode_mu",
    ]:
        df[col] = _num(df, col)
    q = {
        "valid_q50": float(df["topology_valid_mass"].quantile(0.50)),
        "valid_q75": float(df["topology_valid_mass"].quantile(0.75)),
        "invalid_q50": float(df["topology_invalid_mass"].quantile(0.50)),
        "invalid_q75": float(df["topology_invalid_mass"].quantile(0.75)),
        "invalid_q90": float(df["topology_invalid_mass"].quantile(0.90)),
        "context_q75": float(df["topology_context_mass"].quantile(0.75)),
        "lowobs_q75": float(df["topology_lowobs_mass"].quantile(0.75)),
        "boundary_q50": float(df["topology_boundary_conflict"].quantile(0.50)),
        "boundary_q75": float(df["topology_boundary_conflict"].quantile(0.75)),
        "boundary_q90": float(df["topology_boundary_conflict"].quantile(0.90)),
        "entropy_q50": float(df["geometry_mode_entropy"].quantile(0.50)),
        "otopo_q50": float(df["O_topology_scale"].quantile(0.50)),
        "mismatch_q75": float((df["native_delta_log_scale"] - df["topology_valid_dominant_mode_mu"]).abs().quantile(0.75)),
    }
    df["policy_state"] = df.apply(lambda row: _state(row, q), axis=1)
    df["O_update_topology"] = (
        df["O_topology_scale"]
        * (1.0 - df["topology_invalid_mass"].clip(0, 1))
        * (1.0 - df["topology_boundary_conflict"].clip(0, 1))
        * (1.0 - (df["geometry_mode_entropy"] / max(float(df["geometry_mode_entropy"].max()), 1e-12)).clip(0, 1))
    )
    df["unsafe_native_update_flag"] = df["policy_state"].isin(["RESET_RISK", "REJECT"])
    df["hold_delay_or_abstain_flag"] = df["policy_state"].isin(["HOLD", "DELAY", "ABSTAIN"])
    write_csv(args.out_dir / "topology_observability_policy_rows.csv", df.to_dict("records"))
    summary = {
        "phase": "Phase4_semantic_topology_observability_policy_build",
        "policy_rows": int(len(df)),
        "sequence_coverage": int(df["seq"].nunique()),
        "state_counts": df["policy_state"].value_counts().to_dict(),
        "quantiles": q,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    write_json(args.out_dir / "topology_observability_policy_build_summary.json", summary)
    print(f"policy_rows={summary['policy_rows']}")
    print(f"sequence_coverage={summary['sequence_coverage']}")
    print(f"state_counts={summary['state_counts']}")


if __name__ == "__main__":
    main()
