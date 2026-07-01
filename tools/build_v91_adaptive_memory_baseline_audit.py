#!/usr/bin/env python3
"""Build v91 adaptive memory baseline / delayed commit candidates from adjacent edges."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from v86_soft_latent_utils import write_csv, write_json
from v91_semantic_regime_utils import ROOT, nseries


DEFAULT_POLICY = ROOT / "phase5_memory_update_policy"
DEFAULT_OUT = ROOT / "phase6_adaptive_memory_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.policy_dir / "policy_state_rows.csv")
    df["seq"] = df["seq"].astype(str).str.zfill(2)
    df["prev_chunk"] = pd.to_numeric(df["prev_chunk"], errors="coerce").fillna(0).astype(int)
    df["curr_chunk"] = pd.to_numeric(df["curr_chunk"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values(["seq", "prev_chunk", "curr_chunk"]).reset_index(drop=True)
    rows = []
    by_key = {(r.seq, int(r.prev_chunk), int(r.curr_chunk)): r for r in df.itertuples(index=False)}
    for r in df.itertuples(index=False):
        seq, prev, curr = r.seq, int(r.prev_chunk), int(r.curr_chunk)
        prev_edge = by_key.get((seq, prev - 1, prev))
        next_edge = by_key.get((seq, curr, curr + 1))
        states = [r.policy_state]
        mus = [float(getattr(r, "geometry_dominant_mode_mu", 0.0) or 0.0)]
        if prev_edge is not None:
            states.insert(0, prev_edge.policy_state)
            mus.insert(0, float(getattr(prev_edge, "geometry_dominant_mode_mu", 0.0) or 0.0))
        if next_edge is not None:
            states.append(next_edge.policy_state)
            mus.append(float(getattr(next_edge, "geometry_dominant_mode_mu", 0.0) or 0.0))
        mode_consistency = 1.0 / (1.0 + pd.Series(mus).std()) if len(mus) >= 2 else 0.0
        update_votes = sum(s == "UPDATE" for s in states)
        risk_votes = sum(s in {"REJECT", "RESET_RISK", "DELAY"} for s in states)
        if update_votes >= 2 and risk_votes == 0 and mode_consistency >= 0.80:
            delayed_state = "COMMIT_UPDATE"
        elif risk_votes >= 2 and mode_consistency >= 0.75:
            delayed_state = "COMMIT_RISK"
        elif "HOLD" in states:
            delayed_state = "COMMIT_HOLD"
        else:
            delayed_state = "COMMIT_ABSTAIN"
        rows.append(
            {
                "seq": seq,
                "prev_chunk": prev,
                "curr_chunk": curr,
                "pair_id": r.pair_id,
                "single_edge_state": r.policy_state,
                "delayed_commit_state": delayed_state,
                "mode_consistency_2edge": mode_consistency,
                "mode_consistency_3edge": mode_consistency if len(states) >= 3 else "",
                "semantic_state_consistency_2edge": float(update_votes / max(len(states), 1)),
                "semantic_state_consistency_3edge": float(update_votes / max(len(states), 1)) if len(states) >= 3 else "",
                "update_to_delay_transition_count": int(r.policy_state == "UPDATE" and delayed_state != "COMMIT_UPDATE"),
                "premature_update_risk": int(r.policy_state == "UPDATE" and delayed_state != "COMMIT_UPDATE"),
                "commit_delay": max(0, len(states) - 1),
                "abs_log_scale_jump_gt": getattr(r, "abs_log_scale_jump_gt", ""),
                "base_case_type": getattr(r, "base_case_type", ""),
                "offline_audit_label_only": True,
                "non_adjacent_raw_overlap_available": False,
                "non_adjacent_note": "Used chained adjacent edge consensus; no non-adjacent raw pairs fabricated.",
            }
        )
    out = pd.DataFrame(rows)
    write_csv(args.out_dir / "adaptive_memory_baseline_rows.csv", rows)
    summary = {
        "phase": "Phase6_adaptive_memory_baseline_build",
        "rows": int(len(out)),
        "sequence_coverage": int(out["seq"].nunique()) if len(out) else 0,
        "non_adjacent_raw_overlap_available": False,
        "delayed_commit_state_counts": out["delayed_commit_state"].value_counts().to_dict() if len(out) else {},
        "premature_update_risk_count": int(out["premature_update_risk"].sum()) if len(out) else 0,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    write_json(args.out_dir / "adaptive_memory_baseline_summary.json", summary)
    print(f"rows={summary['rows']}")
    print(f"sequence_coverage={summary['sequence_coverage']}")
    print(f"non_adjacent_raw_overlap_available={summary['non_adjacent_raw_overlap_available']}")
    print(f"delayed_commit_state_counts={summary['delayed_commit_state_counts']}")
    print(f"premature_update_risk_count={summary['premature_update_risk_count']}")


if __name__ == "__main__":
    main()
