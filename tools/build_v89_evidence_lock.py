#!/usr/bin/env python3
"""Build v89 Phase0 evidence lock from v88 final artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from v86_soft_latent_utils import read_json, write_csv, write_json


DEFAULT_ROOT = Path("results/acl2_v89tf_semantic_scale_mode_observability_memory_control")
DEFAULT_V88_ROOT = Path("results/acl2_v88tf_scale_mode_consensus_gauge_update_attribution")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ROOT / "phase0_evidence_lock")
    parser.add_argument("--v88-root", type=Path, default=DEFAULT_V88_ROOT)
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def main() -> None:
    args = parse_args()
    out = args.out_dir
    v88_final_path = args.v88_root / "report_final/final_decision.json"
    v88_visual_path = args.v88_root / "phase7_visual_rediscovery/visual_integrity_audit.json"
    required = [
        Path("docs/ACL2_v88TF_ScaleModeConsensus_GaugeUpdateAttribution_实验结果复盘.md"),
        Path("docs/ACL2_v88TF_ScaleModeConsensus_GaugeUpdateAttribution_执行日志.md"),
        v88_final_path,
        v88_visual_path,
        args.v88_root / "phase1_scale_mode_consensus_universe/scale_mode_pair_rows.csv",
        args.v88_root / "phase1_scale_mode_consensus_universe/scale_mode_edge_rows.csv",
        args.v88_root / "phase1_scale_mode_consensus_universe/mode_histograms.csv",
    ]
    required_rows = [{"path": str(path), "exists": path.exists()} for path in required]
    missing = [row for row in required_rows if not row["exists"]]
    final = _json(v88_final_path)
    visual = _json(v88_visual_path)
    forbidden = [
        "repeat_v88_signed_mode_threshold_entropy_mad_small_sweep_as_success",
        "repeat_native_mismatch_q75_sign_highobs_guard_sweep_as_success",
        "repeat_mode_aware_counterfactual_CF1_CF4_small_sweep_as_success",
        "reuse_v84_source_side_anchor_mask_or_external_source_boost",
        "reuse_v85_hard_anchor_support_sweep",
        "reuse_v86_ridge_feature_dim_generic_latent_C_sweep",
        "run_TTT_without_confirmed_carrier",
        "claim_feature_match_success_when_matcher_unavailable",
        "promote_split_diagnostic_to_runtime_without_no_gt_trigger_and_later_gates",
    ]
    gate = bool(
        not missing
        and final.get("final_status") == "No-Go_before_runtime_action"
        and final.get("runtime_action_allowed") is False
        and final.get("ttt_allowed") is False
        and (visual.get("visual_integrity_gate_pass") is True or final.get("final_no_go_allowed") is True)
        and len(forbidden) > 0
    )
    boundary = {
        "v88_final_status": final.get("final_status"),
        "v88_runtime_action_allowed": final.get("runtime_action_allowed"),
        "v88_ttt_allowed": final.get("ttt_allowed"),
        "v88_final_no_go_allowed": final.get("final_no_go_allowed"),
        "v88_visual_integrity_gate_pass": visual.get("visual_integrity_gate_pass"),
        "v88_key_metrics": final.get("key_metrics", {}),
    }
    summary = {
        "phase": "Phase0_v88_evidence_lock",
        "phase0_gate_pass": gate,
        "missing_required_inputs": len(missing),
        "forbidden_repeat_count": len(forbidden),
        "result_tree_unavailable": False,
        "source": "v88_result_tree_and_docs",
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    if not gate:
        summary["blocker"] = "phase0_evidence_lock_failed"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "evidence_lock.json", {"boundary": boundary, "required_inputs": required_rows, "forbidden_repeats": forbidden})
    write_json(out / "phase0_gate_summary.json", summary)
    write_csv(out / "required_inputs.csv", required_rows)
    write_csv(out / "v89_hypothesis_matrix.csv", [
        {"hypothesis": "H1_semantic_mode_disambiguation", "status": "to_test", "phase": "Phase1_Phase2"},
        {"hypothesis": "H2_semantic_valid_native_mismatch", "status": "to_test", "phase": "Phase2_Phase4"},
        {"hypothesis": "H3_semantic_observability", "status": "to_test", "phase": "Phase4"},
        {"hypothesis": "H4_feature_match_semantic_ruler", "status": "to_test", "phase": "Phase3"},
        {"hypothesis": "H5_delayed_commit", "status": "to_test_if_needed", "phase": "Phase7"},
    ])
    write_csv(out / "forbidden_repeats.csv", [{"forbidden_repeat": item} for item in forbidden])
    (out / "forbidden_repeats.md").write_text("\n".join(["# v89 Forbidden Repeats", "", *[f"- {item}" for item in forbidden]]) + "\n", encoding="utf-8")
    (out / "v88_no_go_boundary.md").write_text(
        "\n".join(
            [
                "# v88 No-Go Boundary",
                "",
                f"- final_status: `{boundary['v88_final_status']}`",
                f"- runtime_action_allowed: `{boundary['v88_runtime_action_allowed']}`",
                f"- ttt_allowed: `{boundary['v88_ttt_allowed']}`",
                f"- final_no_go_allowed: `{boundary['v88_final_no_go_allowed']}`",
                f"- visual_integrity_gate_pass: `{boundary['v88_visual_integrity_gate_pass']}`",
                "",
                "v89 must not convert v88 split diagnostics into runtime eligibility without new semantic-conditioned gates.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"phase0_gate_pass={summary['phase0_gate_pass']}")
    print(f"missing_required_inputs={summary['missing_required_inputs']}")
    print(f"forbidden_repeat_count={summary['forbidden_repeat_count']}")
    print(f"runtime_action_allowed={summary['runtime_action_allowed']}")
    print(f"ttt_allowed={summary['ttt_allowed']}")
    if summary.get("blocker"):
        print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
