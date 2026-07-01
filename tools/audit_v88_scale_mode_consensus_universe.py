#!/usr/bin/env python3
"""Audit v88 Phase1 scale-mode consensus universe outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from v86_soft_latent_utils import finite_median, read_json, safe_float, write_json


DEFAULT_PHASE1 = Path("results/acl2_v88tf_scale_mode_consensus_gauge_update_attribution/phase1_scale_mode_consensus_universe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", type=Path, default=DEFAULT_PHASE1)
    return parser.parse_args()


def _file_nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def main() -> None:
    args = parse_args()
    summary_path = args.phase1_dir / "phase1_gate_summary.json"
    pair_path = args.phase1_dir / "scale_mode_pair_rows.csv"
    edge_path = args.phase1_dir / "scale_mode_edge_rows.csv"
    hist_path = args.phase1_dir / "mode_histograms.csv"
    preview_dir = args.phase1_dir / "mode_histogram_previews"
    summary: dict[str, Any] = read_json(summary_path) if summary_path.exists() else {}
    pair = pd.read_csv(pair_path) if pair_path.exists() else pd.DataFrame()
    high = pair[pair.get("quality_type", pd.Series(dtype=str)) == "high_quality"].copy() if len(pair) else pd.DataFrame()
    preview_count = len(list(preview_dir.glob("*.png"))) if preview_dir.exists() else 0
    high_valid = pd.to_numeric(high.get("valid_edge_count", pd.Series(dtype=float)), errors="coerce") if len(high) else pd.Series(dtype=float)
    native_available = pd.to_numeric(pair.get("native_delta_log_scale", pd.Series(dtype=float)), errors="coerce").notna().sum() if len(pair) else 0
    zero_conf = pd.to_numeric(high.get("zero_conf_mass_in_mode", pd.Series(dtype=float)), errors="coerce") if len(high) else pd.Series(dtype=float)
    audit = {
        "phase": "Phase1_scale_mode_consensus_universe_audit",
        "input_summary_phase1_gate_pass": summary.get("phase1_gate_pass"),
        "files": {
            "phase1_gate_summary_json": _file_nonempty(summary_path),
            "scale_mode_pair_rows_csv": _file_nonempty(pair_path),
            "scale_mode_edge_rows_csv": _file_nonempty(edge_path),
            "mode_histograms_csv": _file_nonempty(hist_path),
            "mode_histogram_previews_dir": preview_dir.exists(),
        },
        "pair_rows": int(len(pair)),
        "sequence_coverage": int(pair["seq"].astype(str).str.zfill(2).nunique()) if len(pair) and "seq" in pair else 0,
        "valid_edge_count_median_high_quality_recomputed": finite_median(high_valid.dropna().tolist()),
        "local_shape_mode_available_high_quality_ratio_recomputed": float(
            (
                (high_valid >= 1000)
                & pd.to_numeric(high.get("weighted_mode_mu", pd.Series(dtype=float)), errors="coerce").notna()
            ).sum()
            / max(len(high), 1)
        )
        if len(high)
        else 0.0,
        "native_transition_proxy_available_ratio_recomputed": float(native_available / max(len(pair), 1)) if len(pair) else 0.0,
        "zero_conf_mode_mass_in_high_quality_recomputed": finite_median(zero_conf.dropna().tolist()),
        "histogram_preview_count_recomputed": preview_count,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    zero_conf_mass = audit["zero_conf_mode_mass_in_high_quality_recomputed"]
    zero_conf_mass_for_gate = 1.0 if zero_conf_mass is None else float(zero_conf_mass)
    checks = {
        "files_nonempty": all(audit["files"].values()),
        "pair_rows_ge_49": audit["pair_rows"] >= 49,
        "sequence_coverage_ge_4": audit["sequence_coverage"] >= 4,
        "valid_edge_count_median_high_quality_ge_1000": (audit["valid_edge_count_median_high_quality_recomputed"] or 0) >= 1000,
        "local_shape_mode_available_high_quality_ratio_ge_0_80": audit["local_shape_mode_available_high_quality_ratio_recomputed"] >= 0.80,
        "native_transition_proxy_available_ratio_ge_0_80": audit["native_transition_proxy_available_ratio_recomputed"] >= 0.80,
        "zero_conf_mode_mass_in_high_quality_le_0_05": zero_conf_mass_for_gate <= 0.05,
        "histogram_preview_count_ge_8": audit["histogram_preview_count_recomputed"] >= 8,
    }
    audit["checks"] = checks
    audit["phase1_audit_gate_pass"] = all(checks.values())
    audit["phase1_summary_consistent"] = bool(audit["phase1_audit_gate_pass"]) == bool(summary.get("phase1_gate_pass"))
    write_json(args.phase1_dir / "phase1_audit_summary.json", audit)
    print(f"phase1_audit_gate_pass={audit['phase1_audit_gate_pass']}")
    print(f"phase1_summary_consistent={audit['phase1_summary_consistent']}")
    print(f"pair_rows={audit['pair_rows']}")
    print(f"sequence_coverage={audit['sequence_coverage']}")
    print(f"histogram_preview_count={preview_count}")


if __name__ == "__main__":
    main()
