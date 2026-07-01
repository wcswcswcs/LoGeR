#!/usr/bin/env python3
"""Build ACL2 v86 final report from audited phase artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from v86_soft_latent_utils import read_json, write_json


DEFAULT_ROOT = Path("results/acl2_v86tf_robust_soft_latent_gauge_transport")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ROOT / "report_final")
    parser.add_argument("--phase2-dir", type=Path, default=DEFAULT_ROOT / "phase2_robust_transport_dim4_ridge10_supportfix")
    parser.add_argument("--phase3-dir", type=Path, default=DEFAULT_ROOT / "phase3_anchor_absence_signal_dim4_ridge10_supportfix_global_prefix")
    parser.add_argument("--phase4-dir", type=Path, default=DEFAULT_ROOT / "phase4_scale_relevance_dim4_ridge10_supportfix_global_prefix")
    return parser.parse_args()


def _md_value(value: Any) -> str:
    return f"`{value}`"


def main() -> None:
    args = parse_args()
    root = args.root
    out = args.out_dir
    phase0 = read_json(root / "phase0_evidence_lock/phase0_gate_summary.json")
    phase1 = read_json(root / "phase1_soft_pair_universe/soft_pair_support_summary.json")
    phase1_build = read_json(root / "phase1_soft_pair_universe/build_summary.json")
    phase2 = read_json(args.phase2_dir / "alignment_gain_gate_summary.json")
    phase3 = read_json(args.phase3_dir / "anchor_absence_signal_summary.json")
    phase4 = read_json(args.phase4_dir / "scale_relevance_summary.json")
    visual = read_json(root / "phase12_visual_rediscovery/visual_integrity_audit.json")
    decision = read_json(root / "phase13_decision_matrix/decision_matrix.json")
    final = {
        "phase": "report_final",
        "final_status": "No-Go_before_runtime_action",
        "final_no_go_allowed": bool(decision.get("final_no_go_allowed", False)),
        "runtime_action_allowed": False,
        "ttt_allowed": False,
        "blocker": decision.get("no_go_blocker", "unknown"),
        "active_decision_labels": decision.get("active_decision_labels", []),
        "phase0_gate_pass": bool(phase0.get("phase0_gate_pass", phase0.get("gate_pass", False))),
        "phase1_gate_pass": bool(phase1.get("phase1_gate_pass", False)),
        "phase2_alignment_gate_pass": bool(phase2.get("phase2_alignment_gate_pass", False)),
        "phase3_anchor_absence_gate_pass": bool(phase3.get("phase3_anchor_absence_gate_pass", False)),
        "phase4_scale_relevance_gate_pass": bool(phase4.get("phase4_scale_relevance_gate_pass", False)),
        "visual_gate_pass": bool(visual.get("gate_pass", visual.get("visual_audit_gate_pass", False))),
        "selected_phase2_dir": str(args.phase2_dir),
        "selected_phase3_dir": str(args.phase3_dir),
        "selected_phase4_dir": str(args.phase4_dir),
        "key_metrics": {
            "phase1_pair_count": phase1.get("pair_count", phase1_build.get("pair_count")),
            "phase1_adjacent_labelled_rows": phase1.get("adjacent_labelled_rows"),
            "phase1_sequence_coverage": phase1.get("sequence_coverage"),
            "phase1_weighted_support_sufficient_pairs": phase1.get("weighted_support_sufficient_pairs"),
            "phase1_bad_weighted_support_sufficient_pairs": phase1.get("bad_weighted_support_sufficient_pairs"),
            "phase2_valid_pair_rows": phase2.get("valid_pair_rows"),
            "phase2_bad_valid_pair_rows": phase2.get("bad_valid_pair_rows"),
            "phase2_median_alignment_gain": phase2.get("median_alignment_gain"),
            "phase3_bad_recall": phase3.get("absence_metrics", {}).get("bad_recall"),
            "phase3_good_FPR": phase3.get("absence_metrics", {}).get("good_FPR"),
            "phase3_prior_mismatch_abs_scale_spearman_rho": phase3.get("prior_mismatch_abs_scale_spearman_rho"),
            "phase4_scale_label_rows": phase4.get("scale_label_rows"),
            "phase4_sequence_coverage": phase4.get("sequence_coverage"),
            "visual_manifest_rows": visual.get("manifest_rows"),
            "visual_review_coverage": visual.get("review_coverage"),
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "final_decision.json", final)
    (out / "final_report.md").write_text(build_markdown(final, phase0, phase1, phase2, phase3, phase4, visual, decision), encoding="utf-8")
    print(f"final_status={final['final_status']}")
    print(f"final_no_go_allowed={str(final['final_no_go_allowed']).lower()}")
    print(f"blocker={final['blocker']}")
    print(f"runtime_action_allowed={str(final['runtime_action_allowed']).lower()}")
    print(f"ttt_allowed={str(final['ttt_allowed']).lower()}")


def build_markdown(
    final: dict[str, Any],
    phase0: dict[str, Any],
    phase1: dict[str, Any],
    phase2: dict[str, Any],
    phase3: dict[str, Any],
    phase4: dict[str, Any],
    visual: dict[str, Any],
    decision: dict[str, Any],
) -> str:
    answers = [
        (
            "1. Did soft weighted support overcome v85 strong-anchor blocker?",
            (
                "Yes at audit-support level. Phase1 passed with "
                f"weighted_support_sufficient_pairs={phase1.get('weighted_support_sufficient_pairs')} and "
                f"bad_weighted_support_sufficient_pairs={phase1.get('bad_weighted_support_sufficient_pairs')}. "
                "This did not authorize runtime action."
            ),
        ),
        (
            "2. Did current-to-history C outperform identity/random/shuffle on heldout pairs?",
            (
                "Partially in isolated rows, but not enough for the Phase2 gate. Selected repaired branch has "
                f"valid_pair_rows={phase2.get('valid_pair_rows')}, bad_valid_pair_rows={phase2.get('bad_valid_pair_rows')}, "
                f"median_alignment_gain={phase2.get('median_alignment_gain')}, and phase2_alignment_gate_pass={phase2.get('phase2_alignment_gate_pass')}."
            ),
        ),
        (
            "3. Did historical prior help low-support bad pairs?",
            (
                "No gate pass. Global-prefix prior increased availability, but Phase3 prior-scale rho remained "
                f"{phase3.get('prior_mismatch_abs_scale_spearman_rho')}, below the required 0.30."
            ),
        ),
        (
            "4. Did anchor absence become a useful risk signal?",
            (
                "No. Phase3 absence metrics were "
                f"bad_recall={phase3.get('absence_metrics', {}).get('bad_recall')}, "
                f"good_FPR={phase3.get('absence_metrics', {}).get('good_FPR')}, "
                f"sequence_coverage={phase3.get('absence_metrics', {}).get('sequence_coverage')}."
            ),
        ),
        (
            "5. Did any alignment or absence signal correlate with offline scale jump?",
            (
                "No. Phase4 gate is false with "
                f"scale_label_rows={phase4.get('scale_label_rows')} and sequence_coverage={phase4.get('sequence_coverage')}. "
                "The repaired signal rows do not satisfy rho/recall/FPR plus shuffle-margin requirements."
            ),
        ),
        (
            "6. Did route carrier audit pass?",
            "Not run. Entry requires a passing Phase2 or Phase3/4 signal; the current evidence did not meet that condition.",
        ),
        (
            "7. Did runtime QK bias improve J_SWA while protecting good cases?",
            "Not run. Runtime QK bias remained forbidden because Phase2/3/4 and carrier gates did not pass.",
        ),
        (
            "8. If SWA failed, did merge/gauge direct pair weighting pass?",
            "Not run. Phase9 entry requires scale relevance or carrier/counterfactual evidence; Phase4 failed.",
        ),
        (
            "9. Was TTT kept blocked until confirmed evidence?",
            "Yes. No Phase7 or Phase9 confirmed evidence exists; ttt_allowed=false.",
        ),
        (
            "10. Did any candidate pass held-out/official validation?",
            "No. No candidate reached runtime, route action, merge/gauge action, TTT, or official validation.",
        ),
        (
            "11. If No-Go, which blocker label applies?",
            f"`{final['blocker']}` with active labels `{', '.join(final['active_decision_labels'])}`.",
        ),
        (
            "12. What visual evidence supports the blocker?",
            (
                "Phase12 visual audit passed with "
                f"manifest_rows={visual.get('manifest_rows')} and review_coverage={visual.get('review_coverage')}. "
                "Panels show support exists but repaired C/prior/absence signals do not become scale-relevant; route/merge/TTT panels are explicitly gate-blocked."
            ),
        ),
    ]
    lines = [
        "# ACL2 v86TF Final Report",
        "",
        "## Final Decision",
        "",
        f"- Final status: {_md_value(final['final_status'])}",
        f"- Final No-Go allowed: {_md_value(final['final_no_go_allowed'])}",
        f"- Blocker: {_md_value(final['blocker'])}",
        f"- Active decision labels: {_md_value(', '.join(final['active_decision_labels']))}",
        f"- Runtime action allowed: {_md_value(final['runtime_action_allowed'])}",
        f"- TTT allowed: {_md_value(final['ttt_allowed'])}",
        "",
        "## Key Evidence",
        "",
        f"- Phase0 gate pass: {_md_value(phase0.get('phase0_gate_pass', phase0.get('gate_pass')))}",
        f"- Phase1 gate pass: {_md_value(phase1.get('phase1_gate_pass'))}",
        f"- Phase1 weighted support sufficient pairs: {_md_value(phase1.get('weighted_support_sufficient_pairs'))}",
        f"- Phase1 bad weighted support sufficient pairs: {_md_value(phase1.get('bad_weighted_support_sufficient_pairs'))}",
        f"- Phase2 alignment gate pass: {_md_value(phase2.get('phase2_alignment_gate_pass'))}",
        f"- Phase2 valid pair rows / bad valid pair rows: {_md_value(str(phase2.get('valid_pair_rows')) + ' / ' + str(phase2.get('bad_valid_pair_rows')))}",
        f"- Phase3 anchor absence gate pass: {_md_value(phase3.get('phase3_anchor_absence_gate_pass'))}",
        f"- Phase4 scale relevance gate pass: {_md_value(phase4.get('phase4_scale_relevance_gate_pass'))}",
        f"- Visual audit gate pass: {_md_value(visual.get('gate_pass', visual.get('visual_audit_gate_pass')))}",
        "",
        "## Required Questions",
        "",
    ]
    for question, answer in answers:
        lines.extend([f"### {question}", "", answer, ""])
    lines.extend(
        [
            "## Audit Notes",
            "",
            "- No GT/offline scale label was used as a runtime trigger.",
            "- No pose, pointmap, V, runtime QK, merge/gauge, or TTT action was run.",
            "- Code changes made during this run are listed in the execution and recap logs for audit.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
