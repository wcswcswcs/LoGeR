#!/usr/bin/env python3
"""Audit v92 Phase2 boundary trace availability before carrier relevance."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from v86_soft_latent_utils import safe_float, write_csv, write_json
from v92_semantic_policy_carrier_utils import ROOT, bool_text


DEFAULT_OUT = ROOT / "phase2_boundary_trace_ledger"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _has_number(value: Any) -> bool:
    return safe_float(value) is not None


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.out_dir / "boundary_trace_rows.csv")
    total = max(int(len(df)), 1)
    row_coverage = float(len(df) / total)
    seq_cov = int(df["seq"].astype(str).str.zfill(2).nunique())
    true_mask = df["true_boundary_trace_available"].map(bool_text)
    proxy_mask = df["auditable_policy_proxy_available"].map(bool_text)
    norm_mask = df["boundary_update_norm"].map(_has_number)
    residual_mask = df["merge_residual_delta"].map(_has_number)
    scale_proxy_mask = df["boundary_scale_proxy"].map(_has_number)
    provenance_mask = (
        df["phase1_row_source"].astype(str).str.len().gt(0)
        & (df["true_trace_path"].astype(str).str.len().gt(0) | df["proxy_trace_path"].astype(str).str.len().gt(0))
    )
    true_ratio = float(true_mask.mean()) if len(df) else 0.0
    true_or_proxy_ratio = float((true_mask | proxy_mask).mean()) if len(df) else 0.0
    norm_ratio = float(norm_mask.mean()) if len(df) else 0.0
    residual_or_scale_ratio = float((residual_mask | scale_proxy_mask).mean()) if len(df) else 0.0
    provenance_ratio = float(provenance_mask.mean()) if len(df) else 0.0
    gate = bool(
        row_coverage >= 0.90
        and seq_cov >= 4
        and true_or_proxy_ratio >= 0.90
        and norm_ratio >= 0.80
        and residual_or_scale_ratio >= 0.80
        and provenance_ratio >= 1.0
    )
    checks = [
        {"check": "row_coverage", "value": row_coverage, "threshold": 0.90, "pass": row_coverage >= 0.90},
        {"check": "sequence_coverage", "value": seq_cov, "threshold": 4, "pass": seq_cov >= 4},
        {
            "check": "true_trace_or_auditable_proxy_ratio",
            "value": true_or_proxy_ratio,
            "threshold": 0.90,
            "pass": true_or_proxy_ratio >= 0.90,
        },
        {"check": "boundary_update_norm_available_ratio", "value": norm_ratio, "threshold": 0.80, "pass": norm_ratio >= 0.80},
        {
            "check": "merge_residual_delta_or_boundary_scale_proxy_available_ratio",
            "value": residual_or_scale_ratio,
            "threshold": 0.80,
            "pass": residual_or_scale_ratio >= 0.80,
        },
        {"check": "provenance_complete_ratio", "value": provenance_ratio, "threshold": 1.0, "pass": provenance_ratio >= 1.0},
    ]
    summary = {
        "phase": "Phase2_boundary_trace_availability_audit",
        "phase2_boundary_trace_availability_gate_pass": gate,
        "row_count": int(len(df)),
        "row_coverage": row_coverage,
        "sequence_coverage": seq_cov,
        "true_trace_ratio": true_ratio,
        "true_trace_rows": int(true_mask.sum()),
        "auditable_policy_proxy_ratio": float(proxy_mask.mean()) if len(df) else 0.0,
        "true_trace_or_auditable_proxy_ratio": true_or_proxy_ratio,
        "boundary_update_norm_available_ratio": norm_ratio,
        "merge_residual_delta_available_ratio": float(residual_mask.mean()) if len(df) else 0.0,
        "boundary_scale_proxy_available_ratio": float(scale_proxy_mask.mean()) if len(df) else 0.0,
        "merge_residual_delta_or_boundary_scale_proxy_available_ratio": residual_or_scale_ratio,
        "provenance_complete_ratio": provenance_ratio,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    if not gate:
        summary["blocker"] = "phase2_true_boundary_update_norm_coverage_insufficient"
    write_csv(args.out_dir / "boundary_trace_availability_checks.csv", checks)
    write_json(args.out_dir / "boundary_trace_availability_audit.json", summary)
    write_json(args.out_dir / "phase2_gate_summary.json", summary)
    print(f"phase2_boundary_trace_availability_gate_pass={summary['phase2_boundary_trace_availability_gate_pass']}")
    print(f"row_count={summary['row_count']}")
    print(f"sequence_coverage={summary['sequence_coverage']}")
    print(f"true_trace_rows={summary['true_trace_rows']}")
    print(f"true_trace_ratio={summary['true_trace_ratio']}")
    print(f"true_trace_or_auditable_proxy_ratio={summary['true_trace_or_auditable_proxy_ratio']}")
    print(f"boundary_update_norm_available_ratio={summary['boundary_update_norm_available_ratio']}")
    print(f"merge_residual_delta_or_boundary_scale_proxy_available_ratio={summary['merge_residual_delta_or_boundary_scale_proxy_available_ratio']}")
    print(f"provenance_complete_ratio={summary['provenance_complete_ratio']}")
    if summary.get("blocker"):
        print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
