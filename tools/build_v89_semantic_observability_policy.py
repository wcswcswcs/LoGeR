#!/usr/bin/env python3
"""Build v89 semantic observability and update eligibility policy."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v86_soft_latent_utils import write_csv, write_json


DEFAULT_ROOT = Path("results/acl2_v89tf_semantic_scale_mode_observability_memory_control")
DEFAULT_LEDGER = DEFAULT_ROOT / "phase1_semantic_scale_mode_ledger"
DEFAULT_OUT = DEFAULT_ROOT / "phase4_semantic_observability_policy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-dir", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _state(row: pd.Series, q: dict[str, float]) -> str:
    valid = float(row["semantic_valid_mass"])
    invalid = float(row["semantic_invalid_mass"])
    context = float(row["semantic_context_mass"])
    lowobs = float(row["semantic_lowobs_mass"])
    entropy = float(row["geometry_mode_entropy"])
    osem = float(row["O_sem_scale"])
    native_mismatch = abs(float(row["native_delta_log_scale"]) - float(row["semantic_valid_dominant_mode_mu"]))
    if invalid >= q["invalid_q75"]:
        return "REJECT_INVALID"
    if lowobs >= q["lowobs_q75"] or context >= q["context_q75"]:
        return "HOLD_GAUGE"
    if valid >= q["valid_q75"] and invalid <= q["invalid_q50"] and entropy <= q["entropy_q50"] and osem >= q["osem_q50"]:
        if native_mismatch >= q["mismatch_q75"]:
            return "RESET_RISK"
        return "UPDATE_ELIGIBLE"
    if valid >= q["valid_q50"] and entropy > q["entropy_q50"]:
        return "DELAY_COMMIT"
    return "ABSTAIN"


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.ledger_dir / "semantic_scale_pair_rows.csv")
    df["seq"] = df["seq"].astype(str).str.zfill(2)
    for col in ["semantic_valid_mass", "semantic_invalid_mass", "semantic_context_mass", "semantic_lowobs_mass", "geometry_mode_entropy", "O_sem_scale", "native_delta_log_scale", "semantic_valid_dominant_mode_mu"]:
        df[col] = _num(df[col]).fillna(0.0)
    q = {
        "valid_q50": float(df["semantic_valid_mass"].quantile(0.50)),
        "valid_q75": float(df["semantic_valid_mass"].quantile(0.75)),
        "invalid_q50": float(df["semantic_invalid_mass"].quantile(0.50)),
        "invalid_q75": float(df["semantic_invalid_mass"].quantile(0.75)),
        "context_q75": float(df["semantic_context_mass"].quantile(0.75)),
        "lowobs_q75": float(df["semantic_lowobs_mass"].quantile(0.75)),
        "entropy_q50": float(df["geometry_mode_entropy"].quantile(0.50)),
        "osem_q50": float(df["O_sem_scale"].quantile(0.50)),
        "mismatch_q75": float((df["native_delta_log_scale"] - df["semantic_valid_dominant_mode_mu"]).abs().quantile(0.75)),
    }
    df["update_state"] = df.apply(lambda row: _state(row, q), axis=1)
    df["O_update"] = (
        df["O_sem_scale"]
        * (1.0 - df["semantic_invalid_mass"].clip(0, 1))
        * (1.0 - (df["geometry_mode_entropy"] / max(float(df["geometry_mode_entropy"].max()), 1e-12)).clip(0, 1))
    )
    df["unsafe_native_update_flag"] = df["update_state"].isin(["RESET_RISK", "REJECT_INVALID"])
    df["hold_or_delay_flag"] = df["update_state"].isin(["HOLD_GAUGE", "DELAY_COMMIT", "ABSTAIN"])
    write_csv(args.out_dir / "semantic_observability_policy_rows.csv", df.to_dict("records"))
    counts = df["update_state"].value_counts().to_dict()
    summary = {
        "phase": "Phase4_semantic_observability_policy_build",
        "policy_rows": int(len(df)),
        "sequence_coverage": int(df["seq"].nunique()),
        "state_counts": counts,
        "quantiles": q,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    write_json(args.out_dir / "semantic_observability_policy_build_summary.json", summary)
    print(f"policy_rows={summary['policy_rows']}")
    print(f"sequence_coverage={summary['sequence_coverage']}")
    print(f"state_counts={summary['state_counts']}")


if __name__ == "__main__":
    main()
