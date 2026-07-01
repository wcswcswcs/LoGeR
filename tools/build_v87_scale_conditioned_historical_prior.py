#!/usr/bin/env python3
"""Build v87 Phase4 no-refresh historical prior decisions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v86_soft_latent_utils import write_csv, write_json


DEFAULT_PHASE1 = Path(
    "results/acl2_v87tf_scale_conditioned_latent_gauge_carrier/phase1_scale_conditioned_pair_universe_k16_r1_median_abs"
)
DEFAULT_PHASE3 = Path("results/acl2_v87tf_scale_conditioned_latent_gauge_carrier/phase3_state_conditioned_latent_transport")
DEFAULT_OUT = Path("results/acl2_v87tf_scale_conditioned_latent_gauge_carrier/phase4_no_refresh_guard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", type=Path, default=DEFAULT_PHASE1)
    parser.add_argument("--phase3-dir", type=Path, default=DEFAULT_PHASE3)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ema-alpha", type=float, default=0.40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    by_pair = pd.read_csv(args.phase1_dir / "scale_conditioned_pair_by_adjacent.csv")
    fit = pd.read_csv(args.phase3_dir / "state_conditioned_c_fit_rows.csv")
    valid_fit_keys = {
        (str(row["seq"]).zfill(2), int(row["prev_chunk"]), int(row["curr_chunk"]))
        for _, row in fit[fit["fit_status"].astype(str) == "ok"].iterrows()
    }
    prior_available_by_seq: dict[str, bool] = {}
    rows: list[dict[str, Any]] = []
    for _, row in by_pair.sort_values(["seq", "prev_chunk", "curr_chunk"]).iterrows():
        seq = str(row["seq"]).zfill(2)
        prev = int(row["prev_chunk"])
        curr = int(row["curr_chunk"])
        prior_available = prior_available_by_seq.get(seq, False)
        support_sufficient = str(row.get("support_sufficient_dim8")).lower() == "true"
        conflict_high = float(row.get("conflict_effective_sample_size") or 0.0) >= 10.0 or str(row.get("state_label")) == "CONFLICT"
        absence_high = float(row.get("absence_score") or 0.0) >= 0.50 or str(row.get("state_label")) == "ABSENCE"
        if support_sufficient and (seq, prev, curr) in valid_fit_keys and not conflict_high:
            decision = "UPDATE"
            reason = "current_support_valid_C"
        elif conflict_high:
            decision = "RESET_RISK"
            reason = "current_conflict_high"
        elif (not support_sufficient or absence_high) and prior_available:
            decision = "HOLD"
            reason = "prior_available_current_support_insufficient"
        else:
            decision = "ABSTAIN"
            reason = "prior_unavailable_or_support_insufficient"
        # No valid C exists in the common v87 path; mismatch is explicit missing, not fabricated.
        prior_mismatch = np.nan
        prior_conflict = np.nan
        rows.append(
            {
                "seq": seq,
                "prev_chunk": prev,
                "curr_chunk": curr,
                "base_case_type": row.get("base_case_type"),
                "quality_type": row.get("quality_type"),
                "state_label": row.get("state_label"),
                "prior_available": prior_available,
                "state_decision": decision,
                "state_decision_reason": reason,
                "prior_mismatch_score": prior_mismatch,
                "prior_conflict_score": prior_conflict,
                "support_effective_sample_size": row.get("support_effective_sample_size"),
                "conflict_effective_sample_size": row.get("conflict_effective_sample_size"),
                "absence_score": row.get("absence_score"),
                "abs_log_scale_jump_gt": row.get("abs_log_scale_jump_gt"),
            }
        )
        if (seq, prev, curr) in valid_fit_keys:
            prior_available_by_seq[seq] = True

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "no_refresh_guard_rows.csv", rows)
    state_decision_counts = {str(k): int(v) for k, v in pd.Series([row["state_decision"] for row in rows]).value_counts().items()}
    summary = {
        "phase": "Phase4_no_refresh_guard_build",
        "rows": len(rows),
        "prior_available_rows": sum(1 for row in rows if row["prior_available"]),
        "state_decision_counts": state_decision_counts,
        "valid_phase3_fit_rows": len(valid_fit_keys),
        "ema_alpha": args.ema_alpha,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    write_json(args.out_dir / "no_refresh_guard_build_summary.json", summary)
    print(f"rows={summary['rows']}")
    print(f"prior_available_rows={summary['prior_available_rows']}")
    print(f"state_decision_counts={summary['state_decision_counts']}")


if __name__ == "__main__":
    main()
