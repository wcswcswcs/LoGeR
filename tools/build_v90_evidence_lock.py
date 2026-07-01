#!/usr/bin/env python3
"""Build v90 Phase0 evidence lock from v89 final artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from v86_soft_latent_utils import read_json, write_csv, write_json


DEFAULT_ROOT = Path("results/acl2_v90tf_semantic_object_topology_scale_mode_memory_control")
DEFAULT_V89_ROOT = Path("results/acl2_v89tf_semantic_scale_mode_observability_memory_control")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ROOT / "phase0_evidence_lock")
    parser.add_argument("--v89-root", type=Path, default=DEFAULT_V89_ROOT)
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def main() -> None:
    args = parse_args()
    out = args.out_dir
    v89_final_path = args.v89_root / "report_final/final_decision.json"
    v89_visual_path = args.v89_root / "phase10_visual_rediscovery/visual_integrity_audit.json"
    required = [
        Path("docs/ACL2_v89TF_SemanticScaleModeObservabilityMemoryControl_实验结果复盘.md"),
        Path("docs/ACL2_v89TF_SemanticScaleModeObservabilityMemoryControl_执行日志.md"),
        v89_final_path,
        v89_visual_path,
        args.v89_root / "phase1_semantic_scale_mode_ledger/semantic_scale_pair_rows.csv",
        args.v89_root / "phase1_semantic_scale_mode_ledger/semantic_scale_mode_rows.csv",
        args.v89_root / "phase3_feature_match_semantic_ruler/feature_match_semantic_rows.csv",
        args.v89_root / "phase3_feature_match_semantic_ruler/feature_match_pair_summary.csv",
    ]
    required_rows = [{"path": str(path), "exists": path.exists()} for path in required]
    missing = [row for row in required_rows if not row["exists"]]
    final = _json(v89_final_path)
    visual = _json(v89_visual_path)
    forbidden = [
        "repeat_v89_compact_id_same_label_ratio_sweep_as_success",
        "repeat_v89_semantic_valid_native_mismatch_threshold_sweep_as_success",
        "repeat_v89_feature_match_semantic_valid_ratio_as_scale_ruler_success",
        "repeat_v89_observability_or_delayed_commit_threshold_sweep_as_success",
        "reuse_v88_scale_mode_consensus_without_object_topology",
        "reuse_v87_scale_conditioned_latent_gauge_carrier_route",
        "reuse_v86_soft_latent_gauge_transport_route",
        "reuse_v84_memory_ruler_audit_route",
        "run_runtime_action_without_topology_signal_carrier_counterfactual_visual_gates",
        "run_TTT_before_runtime_or_merge_gauge_confirmation",
    ]
    gate = bool(
        not missing
        and final.get("final_status") == "No-Go_before_runtime_action"
        and final.get("runtime_action_allowed") is False
        and final.get("runtime_action_executed") is False
        and final.get("ttt_allowed") is False
        and final.get("carrier_tools_run") is False
        and final.get("counterfactual_tools_run") is False
        and visual.get("visual_integrity_gate_pass") is True
        and len(forbidden) > 0
    )
    boundary = {
        "v89_final_status": final.get("final_status"),
        "v89_final_no_go_allowed": final.get("final_no_go_allowed"),
        "v89_runtime_action_allowed": final.get("runtime_action_allowed"),
        "v89_runtime_action_executed": final.get("runtime_action_executed"),
        "v89_ttt_allowed": final.get("ttt_allowed"),
        "v89_carrier_tools_run": final.get("carrier_tools_run"),
        "v89_counterfactual_tools_run": final.get("counterfactual_tools_run"),
        "v89_blocker": final.get("blocker"),
        "v89_visual_integrity_gate_pass": visual.get("visual_integrity_gate_pass"),
        "v89_key_metrics": final.get("key_metrics", {}),
    }
    summary = {
        "phase": "Phase0_v89_evidence_lock",
        "phase0_gate_pass": gate,
        "missing_required_inputs": len(missing),
        "forbidden_repeat_count": len(forbidden),
        "source": "v89_result_tree_and_docs",
        "runtime_action_allowed": False,
        "runtime_action_executed": False,
        "ttt_allowed": False,
        "carrier_tools_run": False,
        "counterfactual_tools_run": False,
    }
    if not gate:
        summary["blocker"] = "phase0_v89_evidence_lock_failed"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "evidence_lock.json", {"boundary": boundary, "required_inputs": required_rows, "forbidden_repeats": forbidden})
    write_json(out / "phase0_gate_summary.json", summary)
    write_csv(out / "required_inputs.csv", required_rows)
    write_csv(out / "forbidden_repeats.csv", [{"forbidden_repeat": item} for item in forbidden])
    write_csv(
        out / "v90_hypothesis_matrix.csv",
        [
            {"hypothesis": "H1_object_component_topology_source", "status": "to_test", "phase": "Phase1"},
            {"hypothesis": "H2_topology_scale_mode_relevance", "status": "to_test", "phase": "Phase2_Phase3"},
            {"hypothesis": "H3_topology_good_protection", "status": "to_test", "phase": "Phase4"},
            {"hypothesis": "H4_feature_match_topology_ruler", "status": "to_test", "phase": "Phase5"},
            {"hypothesis": "H5_memory_carrier_localization", "status": "locked_until_signal_pass", "phase": "Phase6"},
            {"hypothesis": "H6_runtime_or_TTT", "status": "locked_until_all_gates_pass", "phase": "Phase8_Phase9"},
        ],
    )
    (out / "forbidden_repeats.md").write_text("\n".join(["# v90 Forbidden Repeats", "", *[f"- {item}" for item in forbidden]]) + "\n", encoding="utf-8")
    (out / "v89_no_go_boundary.md").write_text(
        "\n".join(
            [
                "# v89 No-Go Boundary for v90",
                "",
                f"- final_status: `{boundary['v89_final_status']}`",
                f"- final_no_go_allowed: `{boundary['v89_final_no_go_allowed']}`",
                f"- runtime_action_allowed: `{boundary['v89_runtime_action_allowed']}`",
                f"- runtime_action_executed: `{boundary['v89_runtime_action_executed']}`",
                f"- ttt_allowed: `{boundary['v89_ttt_allowed']}`",
                f"- carrier_tools_run: `{boundary['v89_carrier_tools_run']}`",
                f"- counterfactual_tools_run: `{boundary['v89_counterfactual_tools_run']}`",
                f"- visual_integrity_gate_pass: `{boundary['v89_visual_integrity_gate_pass']}`",
                f"- blocker: `{boundary['v89_blocker']}`",
                "",
                "v90 must build new object/component topology evidence and cannot promote v89 compact semantic diagnostics into runtime eligibility.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"phase0_gate_pass={summary['phase0_gate_pass']}")
    print(f"missing_required_inputs={summary['missing_required_inputs']}")
    print(f"forbidden_repeat_count={summary['forbidden_repeat_count']}")
    print(f"runtime_action_allowed={summary['runtime_action_allowed']}")
    print(f"runtime_action_executed={summary['runtime_action_executed']}")
    print(f"ttt_allowed={summary['ttt_allowed']}")
    if summary.get("blocker"):
        print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
