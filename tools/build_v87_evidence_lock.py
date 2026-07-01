#!/usr/bin/env python3
"""Build ACL2 v87 Phase0 evidence lock from v86/v85/v84/v83 artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from v86_soft_latent_utils import read_json, write_csv, write_json


DEFAULT_OUT = Path("results/acl2_v87tf_scale_conditioned_latent_gauge_carrier/phase0_evidence_lock")

REQUIRED_INPUTS = [
    ("v86_execution_log", "docs/ACL2_v86TF_RobustSoftLatentGaugeTransport_执行日志.md"),
    ("v86_recap_log", "docs/ACL2_v86TF_RobustSoftLatentGaugeTransport_实验结果复盘.md"),
    ("v85_recap_log", "docs/ACL2_v85TF_LatentAnchorAlignment_PairwiseMemoryRuler_实验结果复盘.md"),
    ("v84_recap_log", "docs/ACL2_v84TF_MemoryRulerAudit_实验结果复盘.md"),
    ("v83_recap_log", "docs/ACL2_v83TF_ClueSufficiency_vs_ActionMisuse_实验结果复盘.md"),
    ("v86_final_decision", "results/acl2_v86tf_robust_soft_latent_gauge_transport/report_final/final_decision.json"),
    (
        "v86_selected_scale_relevance",
        "results/acl2_v86tf_robust_soft_latent_gauge_transport/phase4_scale_relevance_dim4_ridge10_supportfix_global_prefix/scale_relevance_summary.json",
    ),
    ("v86_phase1_support", "results/acl2_v86tf_robust_soft_latent_gauge_transport/phase1_soft_pair_universe/soft_pair_support_summary.json"),
]

FORBIDDEN_REPEATS = [
    "v84 source-side anchor mask boost",
    "v84 support-map-driven merge/gauge fallback",
    "v85 hard anchor sufficiency threshold sweep",
    "v86 ridge lambda / feature dim micro sweep without new scale proxy",
    "running QK runtime action without Phase2/4/5/6 gates",
    "using pooled Q/K C as per-head route carrier",
    "TTT before SWA or merge/gauge confirmed evidence",
    "per-chunk Sim(3) runtime correction",
    "GT-selected threshold/head/layer",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _load_optional_json(path_text: str) -> Any | None:
    path = Path(path_text)
    if not path.exists() or path.suffix.lower() != ".json":
        return None
    return read_json(path)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    input_rows = [
        {
            "name": name,
            "path": path,
            "exists": Path(path).exists(),
            "required_for": "Phase0",
        }
        for name, path in REQUIRED_INPUTS
    ]
    write_csv(args.out_dir / "required_inputs.csv", input_rows)
    write_csv(args.out_dir / "forbidden_repeats.csv", [{"forbidden_repeat": item} for item in FORBIDDEN_REPEATS])
    (args.out_dir / "forbidden_repeats.md").write_text(
        "# Forbidden Repeats\n\n" + "\n".join(f"- {item}" for item in FORBIDDEN_REPEATS) + "\n",
        encoding="utf-8",
    )

    v86 = _load_optional_json(dict(REQUIRED_INPUTS)["v86_final_decision"]) or {}
    v86_scale = _load_optional_json(dict(REQUIRED_INPUTS)["v86_selected_scale_relevance"]) or {}
    v86_phase1 = _load_optional_json(dict(REQUIRED_INPUTS)["v86_phase1_support"]) or {}
    metrics = v86.get("key_metrics") or {}

    checks = {
        "v86_final_status_no_go_before_runtime": v86.get("final_status") == "No-Go_before_runtime_action",
        "v86_phase1_support_gate_passed": v86.get("phase1_gate_pass") is True
        and (metrics.get("phase1_weighted_support_sufficient_pairs") or 0) >= 10,
        "v86_bad_valid_coverage_insufficient": v86.get("phase2_alignment_gate_pass") is False
        and (metrics.get("phase2_bad_valid_pair_rows") or 0) < 3,
        "v86_scale_relevance_failed": v86.get("phase4_scale_relevance_gate_pass") is False
        and v86_scale.get("phase4_scale_relevance_gate_pass") is False,
        "v86_runtime_and_ttt_blocked": v86.get("runtime_action_allowed") is False and v86.get("ttt_allowed") is False,
        "forbidden_repeats_non_empty": len(FORBIDDEN_REPEATS) > 0,
        "required_inputs_checked": len(input_rows) == len(REQUIRED_INPUTS),
    }
    gate_pass = all(checks.values())

    boundary_lines = [
        "# v86 No-Go Boundary",
        "",
        f"- final_status: `{v86.get('final_status')}`",
        f"- blocker: `{v86.get('blocker')}`",
        f"- active_decision_labels: `{', '.join(v86.get('active_decision_labels') or [])}`",
        f"- v86 Phase1 support gate pass: `{v86.get('phase1_gate_pass')}`",
        f"- weighted_support_sufficient_pairs: `{metrics.get('phase1_weighted_support_sufficient_pairs')}`",
        f"- bad_weighted_support_sufficient_pairs: `{metrics.get('phase1_bad_weighted_support_sufficient_pairs')}`",
        f"- Phase2 alignment gate pass: `{v86.get('phase2_alignment_gate_pass')}`",
        f"- valid_pair_rows / bad_valid_pair_rows: `{metrics.get('phase2_valid_pair_rows')} / {metrics.get('phase2_bad_valid_pair_rows')}`",
        f"- selected median_alignment_gain: `{metrics.get('phase2_median_alignment_gain')}`",
        f"- Phase4 scale relevance gate pass: `{v86.get('phase4_scale_relevance_gate_pass')}`",
        f"- scale_label_rows / sequence_coverage: `{metrics.get('phase4_scale_label_rows')} / {metrics.get('phase4_sequence_coverage')}`",
        f"- runtime_action_allowed: `{v86.get('runtime_action_allowed')}`",
        f"- ttt_allowed: `{v86.get('ttt_allowed')}`",
        "",
        "Locked facts:",
        "",
        "1. v86 final_status is No-Go_before_runtime_action.",
        "2. v86 overcame the v85 hard-anchor support blocker at Phase1, but Phase2 bad valid coverage stayed insufficient.",
        "3. v86 best branches had nontrivial heldout alignment gain, but Phase4 scale relevance failed.",
        "4. Prior / anchor absence / residual / C-distance signals all failed the selected scale relevance gate.",
        "5. Phase5 route audit, Phase7 runtime QK bias, Phase9 merge/gauge, and Phase10 TTT were not allowed.",
        "6. No runtime action and no TTT were run.",
        "7. v87 must be scale-relevance-first, not alignment-gain-first.",
    ]
    (args.out_dir / "v86_no_go_boundary.md").write_text("\n".join(boundary_lines) + "\n", encoding="utf-8")

    hypothesis_rows = [
        {
            "hypothesis_id": "H1",
            "v86_failure": "feature residual gain did not explain offline scale jump",
            "v87_test": "raw overlap local-shape scale proxy relevance",
            "phase": "Phase1/2",
        },
        {
            "hypothesis_id": "H2",
            "v86_failure": "bad handoffs lacked stable positive support",
            "v87_test": "SUPPORT/CONFLICT/ABSENCE/STRESS state separation",
            "phase": "Phase1/2",
        },
        {
            "hypothesis_id": "H3",
            "v86_failure": "generic C alignment was not scale-relevant",
            "v87_test": "state-conditioned C only after no-GT scale proxy passes",
            "phase": "Phase3",
        },
        {
            "hypothesis_id": "H4",
            "v86_failure": "offline pooled Q/K features cannot claim true carrier",
            "v87_test": "per-head/per-layer route or merge/gauge carrier audit",
            "phase": "Phase5/8",
        },
        {
            "hypothesis_id": "H5",
            "v86_failure": "TTT lacked confirmed memory evidence",
            "v87_test": "TTT remains blocked unless SWA or merge/gauge passes",
            "phase": "Phase10",
        },
    ]
    write_csv(args.out_dir / "v87_hypothesis_matrix.csv", hypothesis_rows)

    evidence = {
        "phase": "Phase0_evidence_lock",
        "phase0_gate_pass": gate_pass,
        "checks": checks,
        "required_input_count": len(input_rows),
        "missing_required_inputs": [row for row in input_rows if not row["exists"]],
        "v86_boundary": {
            "final_status": v86.get("final_status"),
            "blocker": v86.get("blocker"),
            "phase1_weighted_support_sufficient_pairs": metrics.get("phase1_weighted_support_sufficient_pairs"),
            "phase1_bad_weighted_support_sufficient_pairs": metrics.get("phase1_bad_weighted_support_sufficient_pairs"),
            "phase2_valid_pair_rows": metrics.get("phase2_valid_pair_rows"),
            "phase2_bad_valid_pair_rows": metrics.get("phase2_bad_valid_pair_rows"),
            "phase2_median_alignment_gain": metrics.get("phase2_median_alignment_gain"),
            "phase4_scale_relevance_gate_pass": v86.get("phase4_scale_relevance_gate_pass"),
            "selected_scale_relevance_gate_pass": v86_scale.get("phase4_scale_relevance_gate_pass"),
            "v86_phase1_summary_gate_pass": v86_phase1.get("phase1_gate_pass"),
        },
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    write_json(args.out_dir / "evidence_lock.json", evidence)
    write_json(args.out_dir / "phase0_gate_summary.json", evidence)

    print(f"phase0_gate_pass={gate_pass}")
    print(f"v86_status={v86.get('final_status')}")
    print(f"missing_required_inputs={len(evidence['missing_required_inputs'])}")
    print("runtime_action_allowed=False")
    print("ttt_allowed=False")


if __name__ == "__main__":
    main()
