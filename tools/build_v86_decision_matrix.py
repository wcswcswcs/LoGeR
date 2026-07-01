#!/usr/bin/env python3
"""Build ACL2 v86 decision matrix from audited phase artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Mapping

from v86_soft_latent_utils import read_json, write_csv, write_json


DEFAULT_ROOT = Path("results/acl2_v86tf_robust_soft_latent_gauge_transport")
DEFAULT_PHASE2 = DEFAULT_ROOT / "phase2_robust_transport_dim4_ridge10_supportfix"
DEFAULT_PHASE3 = DEFAULT_ROOT / "phase3_anchor_absence_signal_dim4_ridge10_supportfix_global_prefix"
DEFAULT_PHASE4 = DEFAULT_ROOT / "phase4_scale_relevance_dim4_ridge10_supportfix_global_prefix"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--phase2-dir", type=Path, default=DEFAULT_PHASE2)
    parser.add_argument("--phase3-dir", type=Path, default=DEFAULT_PHASE3)
    parser.add_argument("--phase4-dir", type=Path, default=DEFAULT_PHASE4)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ROOT / "phase13_decision_matrix")
    return parser.parse_args()


def _write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root
    phase0 = read_json(root / "phase0_evidence_lock/phase0_gate_summary.json")
    phase1 = read_json(root / "phase1_soft_pair_universe/soft_pair_support_summary.json")
    phase2 = read_json(args.phase2_dir / "alignment_gain_gate_summary.json")
    phase3 = read_json(args.phase3_dir / "anchor_absence_signal_summary.json")
    phase4 = read_json(args.phase4_dir / "scale_relevance_summary.json")
    visual = read_json(root / "phase12_visual_rediscovery/visual_integrity_audit.json")

    phase0_pass = bool(phase0.get("phase0_gate_pass", phase0.get("gate_pass", False)))
    phase2_pass = bool(phase2.get("phase2_alignment_gate_pass", False))
    phase3_pass = bool(phase3.get("phase3_anchor_absence_gate_pass", False))
    phase4_pass = bool(phase4.get("phase4_scale_relevance_gate_pass", False))
    visual_pass = bool(visual.get("gate_pass", visual.get("visual_audit_gate_pass", False)))
    runtime_allowed = phase2_pass and phase4_pass
    ttt_allowed = False
    phase2_subreason = (
        f"valid_pair_rows={phase2.get('valid_pair_rows')}; "
        f"bad_valid_pair_rows={phase2.get('bad_valid_pair_rows')}; "
        f"sequence_coverage={phase2.get('sequence_coverage_valid') or phase2.get('sequence_coverage')}; "
        f"median_alignment_gain={phase2.get('median_alignment_gain')}"
    )

    rows: list[dict[str, Any]] = [
        {
            "decision_label": "D1_SOFT_SUPPORT_INSUFFICIENT",
            "active": not bool(phase1.get("phase1_gate_pass", False)),
            "evidence": f"phase1_gate_pass={phase1.get('phase1_gate_pass')}; weighted_support_sufficient_pairs={phase1.get('weighted_support_sufficient_pairs')}; bad_weighted_support_sufficient_pairs={phase1.get('bad_weighted_support_sufficient_pairs')}",
            "blocks_runtime": not bool(phase1.get("phase1_gate_pass", False)),
        },
        {
            "decision_label": "D2_ALIGNMENT_NOT_SPECIFIC",
            "active": not phase2_pass,
            "evidence": "Phase2 robust-C gate failed; subreason=" + phase2_subreason,
            "blocks_runtime": not phase2_pass,
        },
        {
            "decision_label": "D3_ALIGNMENT_NOT_SCALE_RELEVANT",
            "active": not phase4_pass,
            "evidence": f"phase4_scale_relevance_gate_pass={phase4_pass}; scale_label_rows={phase4.get('scale_label_rows')}; sequence_coverage={phase4.get('sequence_coverage')}",
            "blocks_runtime": not phase4_pass,
        },
        {
            "decision_label": "D4_SWA_ROUTE_NOT_CARRIER",
            "active": False,
            "evidence": "Not evaluated because Phase2/3/4 did not pass; route carrier entry condition was not met.",
            "blocks_runtime": False,
        },
        {
            "decision_label": "D5_SWA_ROUTE_NOT_GEOMETRY_ACTUATOR",
            "active": False,
            "evidence": "Not evaluated because Phase5/6/7 did not run.",
            "blocks_runtime": False,
        },
        {
            "decision_label": "D6_ANCHOR_ABSENCE_RISK_SIGNAL",
            "active": phase3_pass and not phase4_pass,
            "evidence": f"phase3_anchor_absence_gate_pass={phase3_pass}; absence_metrics={phase3.get('absence_metrics')}; prior_rho={phase3.get('prior_mismatch_abs_scale_spearman_rho')}",
            "blocks_runtime": False,
        },
        {
            "decision_label": "D7_MERGE_DIRECT_PAIR_WEIGHT_NEEDED",
            "active": False,
            "evidence": "Not entered because scale relevance did not pass.",
            "blocks_runtime": False,
        },
        {
            "decision_label": "D8_TTT_NOT_READY",
            "active": True,
            "evidence": "No Phase7 runtime QK pass and no Phase9 merge/gauge pass; TTT stayed blocked.",
            "blocks_runtime": True,
        },
        {
            "decision_label": "D9_METHOD_CANDIDATE",
            "active": False,
            "evidence": "No route/action/control/official-validation candidate passed.",
            "blocks_runtime": False,
        },
    ]
    active_labels = [row["decision_label"] for row in rows if row["active"]]
    no_go_blocker = "phase2_bad_valid_coverage_insufficient_and_phase4_no_scale_relevance"
    final_no_go_allowed = visual_pass and not runtime_allowed
    payload = {
        "phase": "Phase13_decision_matrix",
        "active_decision_labels": active_labels,
        "phase0_gate_pass": phase0_pass,
        "phase1_gate_pass": bool(phase1.get("phase1_gate_pass", False)),
        "phase2_alignment_gate_pass": phase2_pass,
        "phase3_anchor_absence_gate_pass": phase3_pass,
        "phase4_scale_relevance_gate_pass": phase4_pass,
        "visual_gate_pass": visual_pass,
        "runtime_action_allowed": runtime_allowed,
        "ttt_allowed": ttt_allowed,
        "final_no_go_allowed": final_no_go_allowed,
        "no_go_blocker": no_go_blocker,
        "selected_phase2_dir": str(args.phase2_dir),
        "selected_phase3_dir": str(args.phase3_dir),
        "selected_phase4_dir": str(args.phase4_dir),
        "notes": [
            "Phase1 soft support overcame the v85 hard-anchor blocker at audit-support level.",
            "Phase2 still fails because bad valid current-to-history pair coverage is insufficient after repaired branches.",
            "Phase4 no-scale-relevance gate blocks route, merge/gauge, runtime QK action, and TTT.",
            "Phase12 visual audit passed, so final No-Go is allowed without claiming runtime success.",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "decision_matrix.csv", rows)
    write_json(args.out_dir / "decision_matrix.json", payload)
    _write_md(
        args.out_dir / "next_route_recommendation.md",
        [
            "# ACL2 v86 Next Route Recommendation",
            "",
            "- Do not run runtime QK bias, merge/gauge direct pair weighting, or TTT from current evidence.",
            "- Current blocker: `phase2_bad_valid_coverage_insufficient_and_phase4_no_scale_relevance`.",
            "- Future work should first find a non-GT signal that is scale/gauge relevant, then only after that inspect true per-head/per-layer SWA QK route carrier.",
            "- Do not repeat v84 support-map merge/gauge fallback or v85 hard-anchor-only Phase1 gate as the primary claim.",
            "",
        ],
    )
    print(f"active_decision_labels={','.join(active_labels)}")
    print(f"final_no_go_allowed={str(final_no_go_allowed).lower()}")
    print(f"runtime_action_allowed={str(runtime_allowed).lower()}")
    print(f"ttt_allowed={str(ttt_allowed).lower()}")
    print(f"no_go_blocker={no_go_blocker}")


if __name__ == "__main__":
    main()
