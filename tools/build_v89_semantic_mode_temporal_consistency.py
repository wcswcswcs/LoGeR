#!/usr/bin/env python3
"""Build v89 semantic mode temporal consistency diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from v86_soft_latent_utils import write_csv, write_json


DEFAULT_ROOT = Path("results/acl2_v89tf_semantic_scale_mode_observability_memory_control")
DEFAULT_LEDGER = DEFAULT_ROOT / "phase1_semantic_scale_mode_ledger"
DEFAULT_POLICY = DEFAULT_ROOT / "phase4_semantic_observability_policy"
DEFAULT_OUT = DEFAULT_ROOT / "phase7_semantic_mode_temporal_consistency"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-dir", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pair = pd.read_csv(args.ledger_dir / "semantic_scale_pair_rows.csv")
    policy = pd.read_csv(args.policy_dir / "semantic_observability_policy_rows.csv")
    pair["seq"] = pair["seq"].astype(str).str.zfill(2)
    policy["seq"] = policy["seq"].astype(str).str.zfill(2)
    merged = pair.merge(policy[["seq", "prev_chunk", "curr_chunk", "update_state"]], on=["seq", "prev_chunk", "curr_chunk"], how="left")
    rows = []
    for seq, group in merged.sort_values(["seq", "prev_chunk"]).groupby("seq"):
        group = group.reset_index(drop=True)
        vals = _num(group["semantic_valid_dominant_mode_mu"]).fillna(0.0)
        for i, row in group.iterrows():
            prev_cons = None
            next_cons = None
            if i > 0:
                prev_cons = abs(float(vals.iloc[i]) - float(vals.iloc[i - 1]))
            if i + 1 < len(group):
                next_cons = abs(float(vals.iloc[i]) - float(vals.iloc[i + 1]))
            cons2 = min([v for v in [prev_cons, next_cons] if v is not None], default=None)
            cons3 = None
            if 0 < i < len(group) - 1:
                cons3 = max(abs(float(vals.iloc[i]) - float(vals.iloc[i - 1])), abs(float(vals.iloc[i]) - float(vals.iloc[i + 1])))
            rows.append(
                {
                    "seq": seq,
                    "prev_chunk": int(row["prev_chunk"]),
                    "curr_chunk": int(row["curr_chunk"]),
                    "semantic_valid_mode_mu": row["semantic_valid_dominant_mode_mu"],
                    "mode_consistency_2edge": cons2,
                    "mode_consistency_3edge": cons3,
                    "semantic_valid_mode_persistence": cons2 is not None and cons2 <= 0.05,
                    "semantic_invalid_conflict_persistence": row.get("update_state") == "REJECT_INVALID",
                    "lowobs_cluster_len": "",
                    "delayed_commit_count": int(row.get("update_state") == "DELAY_COMMIT"),
                    "premature_update_risk": row.get("update_state") in {"RESET_RISK", "UPDATE_ELIGIBLE"} and cons2 is not None and cons2 > 0.05,
                    "commit_delay": int(row.get("update_state") == "DELAY_COMMIT"),
                    "update_state": row.get("update_state", ""),
                    "base_case_type": row.get("base_case_type", ""),
                    "abs_log_scale_jump_gt": row.get("abs_log_scale_jump_gt", ""),
                    "offline_audit_label_only": True,
                }
            )
    out = pd.DataFrame(rows)
    write_csv(args.out_dir / "semantic_mode_temporal_consistency_rows.csv", rows)
    summary = {
        "phase": "Phase7_semantic_mode_temporal_consistency_build",
        "rows": int(len(out)),
        "sequence_coverage": int(out["seq"].nunique()) if len(out) else 0,
        "delayed_commit_count": int(pd.to_numeric(out["delayed_commit_count"], errors="coerce").sum()) if len(out) else 0,
        "persistent_rows": int(out["semantic_valid_mode_persistence"].astype(bool).sum()) if len(out) else 0,
        "premature_update_risk_rows": int(out["premature_update_risk"].astype(bool).sum()) if len(out) else 0,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    write_json(args.out_dir / "semantic_mode_temporal_consistency_summary.json", summary)
    print(f"rows={summary['rows']}")
    print(f"sequence_coverage={summary['sequence_coverage']}")
    print(f"delayed_commit_count={summary['delayed_commit_count']}")
    print(f"persistent_rows={summary['persistent_rows']}")
    print(f"premature_update_risk_rows={summary['premature_update_risk_rows']}")


if __name__ == "__main__":
    main()
