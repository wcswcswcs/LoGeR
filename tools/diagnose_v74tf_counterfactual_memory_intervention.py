#!/usr/bin/env python3
"""Phase 5 counterfactual semantic-memory intervention audit for ACL2 v74-TF."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from v73_semantic_memory_common import read_csv, safe_float, utc_now, write_csv, write_json, write_text
from v74tf_common import REPORT_ROOT, median


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-family-results", type=Path, default=REPORT_ROOT / "phase4_action_family_oracle" / "action_family_results.csv")
    parser.add_argument("--out-dir", type=Path, default=REPORT_ROOT / "phase5_counterfactual_memory_intervention")
    args = parser.parse_args()

    rows = read_csv(args.action_family_results)
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        out_rows.append(
            {
                "seq": row.get("seq", "01"),
                "chunk_id": row.get("chunk_id"),
                "component_id": "",
                "semantic_role": row.get("fit_semantic_filter"),
                "thingstuff_state": "",
                "radio_component_stability": "",
                "memory_path": "SWA_merge_overlap_pair",
                "intervention_type": row.get("v74tf_action_family"),
                "delta_attention_mass": None,
                "delta_SWA_mass": None,
                "delta_merge_weight": None,
                "delta_TTT_write": None,
                "delta_future_after_overlap": row.get("future_after_overlap_mean_improvement_vs_baseline"),
                "delta_head_to_tail": row.get("head_to_tail_transfer_ratio_mean_improvement_vs_baseline"),
                "delta_scale_cv": row.get("intra_scale_variance_mean_improvement_vs_baseline"),
                "delta_J_scale": row.get("J_scale_improvement_proxy"),
                "matched_random_control_available": row.get("control_kind") not in {"candidate", ""},
                "causal_support_row": bool(str(row.get("positive_by_v74tf_gate")).lower() == "true"),
                "diagnostic_scope": "offline_overlap_pair_counterfactual_proxy_not_online_hook",
            }
        )
    summary_rows: list[dict[str, Any]] = []
    for intervention in sorted({str(row.get("intervention_type", "")) for row in out_rows}):
        inter_rows = [row for row in out_rows if str(row.get("intervention_type", "")) == intervention]
        support_chunks = sorted({int(row.get("chunk_id")) for row in inter_rows if row.get("causal_support_row") and str(row.get("chunk_id", "")).lstrip("-").isdigit()})
        med = median(row.get("delta_J_scale") for row in inter_rows)
        causal_support = bool(len(support_chunks) >= 4 and med is not None and safe_float(med) is not None and float(med) > 0)
        summary_rows.append(
            {
                "intervention_type": intervention,
                "row_count": len(inter_rows),
                "support_chunks": ",".join(str(x) for x in support_chunks),
                "support_chunk_count": len(support_chunks),
                "median_delta_J_scale": med,
                "causal_support": causal_support,
            }
        )
    summary = {
        "schema": "acl2_v74tf_phase5_counterfactual_memory_intervention_v1",
        "created_at": utc_now(),
        "rows": len(out_rows),
        "intervention_summary": summary_rows,
        "phase5_01_gate_pass": any(row.get("causal_support") for row in summary_rows),
        "phase5_09_gate_pass": False,
        "phase5_gate_pass": False,
        "blocked_reason": "No KITTI09 non-reversal check and no online memory-path intervention trace; remains diagnostic-only.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "counterfactual_intervention_rows.csv", out_rows)
    write_csv(args.out_dir / "counterfactual_intervention_summary.csv", summary_rows)
    write_json(args.out_dir / "counterfactual_intervention_summary.json", summary)
    lines = [
        "# v74-TF Phase 5 Counterfactual Memory Intervention",
        "",
        f"- rows: `{len(out_rows)}`",
        f"- phase5_01_gate_pass: `{summary['phase5_01_gate_pass']}`",
        f"- phase5_09_gate_pass: `False`",
        f"- phase5_gate_pass: `False`",
        f"- blocked_reason: `{summary['blocked_reason']}`",
        "",
    ]
    write_text(args.out_dir / "counterfactual_intervention_report.md", "\n".join(lines))
    print({"out_dir": str(args.out_dir), "phase5_gate_pass": summary["phase5_gate_pass"], "blocked_reason": summary["blocked_reason"]})


if __name__ == "__main__":
    main()

