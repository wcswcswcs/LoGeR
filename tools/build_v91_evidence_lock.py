#!/usr/bin/env python3
"""Build v91 Phase0 evidence lock from v90 final artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from v86_soft_latent_utils import read_json, write_csv, write_json
from v91_semantic_regime_utils import ROOT, V90_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "phase0_evidence_lock")
    parser.add_argument("--v90-root", type=Path, default=V90_ROOT)
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def main() -> None:
    args = parse_args()
    out = args.out_dir
    final_path = args.v90_root / "report_final/final_decision.json"
    visual_path = args.v90_root / "phase10_visual_rediscovery/visual_integrity_audit.json"
    required = [
        Path("docs/ACL2_v90TF_SemanticObjectTopologyScaleModeMemoryControl_实验结果复盘.md"),
        Path("docs/ACL2_v90TF_SemanticObjectTopologyScaleModeMemoryControl_执行日志.md"),
        final_path,
        visual_path,
        args.v90_root / "phase1_semantic_topology_source/topology_nodes.csv",
        args.v90_root / "phase1_semantic_topology_source/topology_edges.csv",
        args.v90_root / "phase2_semantic_topology_scale_mode_ledger/topology_pair_rows.csv",
        args.v90_root / "phase2_semantic_topology_scale_mode_ledger/topology_mode_rows.csv",
        args.v90_root / "phase3_semantic_topology_relevance/topology_relevance_summary.json",
        args.v90_root / "phase5_feature_match_topology_ruler/feature_match_topology_pair_summary.csv",
    ]
    required_rows = [{"path": str(path), "exists": path.exists()} for path in required]
    missing = [row for row in required_rows if not row["exists"]]
    final = _json(final_path)
    visual = _json(visual_path)
    forbidden = [
        "repeat_v89_compact_semantic_valid_ratio_sweep",
        "repeat_v90_topology_invalid_support_threshold_sweep",
        "promote_v90_highobs_or_far_split_to_global_success",
        "claim_feature_match_abundance_as_scale_ruler_success",
        "claim_geometry_only_regime_success_as_semantic_success",
        "run_runtime_action_before_policy_carrier_counterfactual_visual_gates",
        "run_TTT_before_runtime_or_merge_gauge_confirmation",
    ]
    gate = bool(
        not missing
        and final.get("final_status") == "No-Go_before_runtime_action"
        and final.get("runtime_action_allowed") is False
        and final.get("runtime_action_executed") is False
        and final.get("ttt_allowed") is False
        and visual.get("visual_integrity_gate_pass") is True
        and final.get("phase1_topology_source_gate_pass") is True
        and final.get("phase2_topology_ledger_audit_gate_pass") is True
        and final.get("phase3_topology_relevance_global_gate_pass") is False
        and final.get("phase3_split_diagnostic_only") is True
        and final.get("phase4_topology_policy_gate_pass") is False
        and final.get("phase5_feature_match_topology_ruler_gate_pass") is False
    )
    summary = {
        "phase": "Phase0_v90_evidence_lock",
        "phase0_gate_pass": gate,
        "missing_required_inputs": len(missing),
        "forbidden_repeat_count": len(forbidden),
        "v90_final_status": final.get("final_status"),
        "v90_visual_integrity_gate_pass": visual.get("visual_integrity_gate_pass"),
        "v90_phase3_split_diagnostic_only": final.get("phase3_split_diagnostic_only"),
        "runtime_action_allowed": False,
        "runtime_action_executed": False,
        "ttt_allowed": False,
    }
    if not gate:
        summary["blocker"] = "phase0_v90_evidence_lock_failed"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "evidence_lock.json", {"v90_final": final, "v90_visual": visual, "required_inputs": required_rows, "forbidden_repeats": forbidden})
    write_json(out / "phase0_gate_summary.json", summary)
    write_csv(out / "required_inputs.csv", required_rows)
    write_csv(out / "forbidden_repeats.csv", [{"forbidden_repeat": item} for item in forbidden])
    write_csv(
        out / "v91_hypothesis_matrix.csv",
        [
            {"hypothesis": "H1_regime_mixing", "status": "to_test", "phase": "Phase2_Phase3"},
            {"hypothesis": "H2_cross_frame_topology_tracklets", "status": "to_test", "phase": "Phase1"},
            {"hypothesis": "H3_semantic_explains_geometry_conflict", "status": "to_test", "phase": "Phase3_Phase4"},
            {"hypothesis": "H4_observability_controls_update_hold_delay", "status": "to_test", "phase": "Phase5_Phase6"},
            {"hypothesis": "H5_merge_gauge_carrier_first", "status": "locked_until_policy_pass", "phase": "Phase7"},
        ],
    )
    (out / "forbidden_repeats.md").write_text("\n".join(["# v91 Forbidden Repeats", "", *[f"- {item}" for item in forbidden]]) + "\n", encoding="utf-8")
    (out / "v90_no_go_boundary.md").write_text(
        "\n".join(
            [
                "# v90 No-Go Boundary for v91",
                "",
                f"- final_status: `{final.get('final_status')}`",
                f"- phase1_topology_source_gate_pass: `{final.get('phase1_topology_source_gate_pass')}`",
                f"- phase2_topology_ledger_audit_gate_pass: `{final.get('phase2_topology_ledger_audit_gate_pass')}`",
                f"- phase3_global_gate_pass: `{final.get('phase3_topology_relevance_global_gate_pass')}`",
                f"- phase3_split_diagnostic_only: `{final.get('phase3_split_diagnostic_only')}`",
                f"- phase4_policy_gate_pass: `{final.get('phase4_topology_policy_gate_pass')}`",
                f"- phase5_feature_match_gate_pass: `{final.get('phase5_feature_match_topology_ruler_gate_pass')}`",
                f"- runtime_action_allowed: `{final.get('runtime_action_allowed')}`",
                f"- ttt_allowed: `{final.get('ttt_allowed')}`",
                f"- visual_integrity_gate_pass: `{visual.get('visual_integrity_gate_pass')}`",
                "",
                "v91 may use v90 topology source and split diagnostics as inputs, but must not promote them without a new global no-GT regime policy.",
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
