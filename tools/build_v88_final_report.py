#!/usr/bin/env python3
"""Build v88 final report from Phase8 decision matrix and audited artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from v86_soft_latent_utils import read_json, write_json


DEFAULT_ROOT = Path("results/acl2_v88tf_scale_mode_consensus_gauge_update_attribution")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ROOT / "report_final")
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def _md(value: Any) -> str:
    return f"`{value}`"


def main() -> None:
    args = parse_args()
    root = args.root
    decision = _json(root / "phase8_decision_matrix/decision_matrix.json")
    phase2_all = _json(root / "phase2_scale_mode_relevance/scale_mode_relevance_summary.json")
    phase2_highobs = _json(root / "phase2_scale_mode_relevance_highobs/scale_mode_relevance_summary.json")
    phase2_nonseq01 = _json(root / "phase2_scale_mode_relevance_nonseq01/scale_mode_relevance_summary.json")
    phase2_near = _json(root / "phase2_scale_mode_relevance_near/scale_mode_relevance_summary.json")
    phase3 = _json(root / "phase3_native_gauge_update_attribution/native_gauge_update_attribution_summary.json")
    phase4_swa = _json(root / "phase4_swa_mode_route_audit/swa_mode_route_audit_summary.json")
    phase4_merge = _json(root / "phase4_merge_gauge_mode_carrier/merge_gauge_mode_carrier_summary.json")
    phase5 = _json(root / "phase5_mode_aware_counterfactual/mode_aware_counterfactual_summary.json")
    visual = _json(root / "phase7_visual_rediscovery/visual_integrity_audit.json")
    final = {
        "phase": "report_final",
        "final_status": decision.get("final_status"),
        "final_no_go_allowed": decision.get("final_no_go_allowed"),
        "runtime_action_allowed": decision.get("runtime_action_allowed"),
        "ttt_allowed": decision.get("ttt_allowed"),
        "blocker": decision.get("blocker"),
        "active_decision_labels": decision.get("active_decision_labels", []),
        "key_metrics": decision.get("key_metrics", {}),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "final_decision.json", final)
    (args.out_dir / "final_report.md").write_text(
        build_markdown(final, decision, phase2_all, phase2_highobs, phase2_nonseq01, phase2_near, phase3, phase4_swa, phase4_merge, phase5, visual),
        encoding="utf-8",
    )
    print(f"final_status={final['final_status']}")
    print(f"final_no_go_allowed={final['final_no_go_allowed']}")
    print(f"runtime_action_allowed={final['runtime_action_allowed']}")
    print(f"ttt_allowed={final['ttt_allowed']}")
    print(f"blocker={final['blocker']}")


def _best(summary: dict[str, Any]) -> dict[str, Any]:
    return summary.get("best_signal") or {}


def build_markdown(
    final: dict[str, Any],
    decision: dict[str, Any],
    phase2_all: dict[str, Any],
    phase2_highobs: dict[str, Any],
    phase2_nonseq01: dict[str, Any],
    phase2_near: dict[str, Any],
    phase3: dict[str, Any],
    phase4_swa: dict[str, Any],
    phase4_merge: dict[str, Any],
    phase5: dict[str, Any],
    visual: dict[str, Any],
) -> str:
    metrics = final.get("key_metrics", {})
    best_all = _best(phase2_all)
    best_highobs = _best(phase2_highobs)
    best_nonseq01 = _best(phase2_nonseq01)
    best_near = _best(phase2_near)
    best_variant = phase3.get("best_variant") or {}
    best_family = phase5.get("best_family") or {}
    answers = [
        (
            "1. Did signed scale mode improve over v87 S_overlap/S_shape scalar proxy?",
            (
                "Not globally. The all-row Phase2 gate failed with "
                f"passing_signals={phase2_all.get('passing_signals')}. "
                f"High-observability repair did pass for {phase2_highobs.get('passing_signals')}, "
                f"but the best highobs baseline remained {best_highobs.get('signal')} with rho={best_highobs.get('spearman_rho_abs_log_scale_jump')}. "
                "This is split diagnostic evidence, not runtime eligibility."
            ),
        ),
        (
            "2. Did mode/mismatch explain offline scale jump across >=3 sequences?",
            (
                "Only in restricted diagnostics. Global Phase2 failed over "
                f"{phase2_all.get('sequence_coverage')} sequences; highobs, nonseq01, and near splits had selected pass signals "
                f"({phase2_highobs.get('passing_signals')}, {phase2_nonseq01.get('passing_signals')}, {phase2_near.get('passing_signals')}) "
                "but were not a full route."
            ),
        ),
        (
            "3. Did semantic-aware mode add value over geometry-only mode?",
            (
                "No. semantic-aware pass was false in the global and repaired split summaries; "
                f"highobs semantic_aware_pass={phase2_highobs.get('semantic_aware_pass')}."
            ),
        ),
        (
            "4. Were good cases protected?",
            (
                "Not enough for action. Phase3 best variant had "
                f"MISMATCH_GOOD_FPR={best_variant.get('MISMATCH_GOOD_FPR')}, but recall/rho/control gates failed. "
                f"Phase5 best family worsened good rows by {best_family.get('good_max_scale_error_worsen')}, far above the 0.02 bound."
            ),
        ),
        (
            "5. Did native update mismatch separate bad/good?",
            (
                "No. Phase3 gate failed with "
                f"recall={best_variant.get('MISMATCH_BAD_recall')}, FPR={best_variant.get('MISMATCH_GOOD_FPR')}, "
                f"rho={best_variant.get('native_mode_mismatch_rho_abs_log_scale_jump')}, "
                f"margin={best_variant.get('rho_margin_vs_shape_shuffle')}."
            ),
        ),
        (
            "6. Was bad dominated by MISMATCH_BAD, MULTIMODE_UNSAFE, or LOWOBS_ABSTAIN?",
            (
                f"Best Phase3 class counts were {best_variant.get('class_counts')}. "
                "MISMATCH_BAD existed but did not meet recall/rho/control requirements; multimode and low-observability remained failure explanations rather than safe actions."
            ),
        ),
        (
            "7. Did SWA route attend dominant or outlier mode differently from random?",
            (
                "No carrier evidence was available. "
                f"SWA route carrier pass={phase4_swa.get('swa_route_carrier_gate_pass')}; blocker={phase4_swa.get('blocker')}; "
                f"dominant_mode_route_lift_available={phase4_swa.get('dominant_mode_route_lift_available')}."
            ),
        ),
        (
            "8. Did merge/gauge boundary state align with native-mode mismatch?",
            (
                "No route pass. "
                f"merge_gauge_mode_carrier_gate_pass={phase4_merge.get('merge_gauge_mode_carrier_gate_pass')}; "
                f"blocker={phase4_merge.get('blocker')}; native_mode_mismatch_rho={phase4_merge.get('native_mode_mismatch_rho')}."
            ),
        ),
        (
            "9. Did any counterfactual upper bound pass?",
            (
                "No. "
                f"scale_label_gate_pass={phase5.get('scale_label_gate_pass')}; raw_residual_counterfactual_available={phase5.get('raw_residual_counterfactual_available')}; "
                f"best_family={best_family.get('family')} bad_I={best_family.get('bad_median_I_scale')} good_worsen={best_family.get('good_max_scale_error_worsen')}."
            ),
        ),
        (
            "10. If runtime ran, did it improve J_SWA / boundary / future / overlapScale and beat controls?",
            "Runtime did not run. Phase6 was blocked because no route passed Phase2/3/4/5 together.",
        ),
        (
            "11. Was TTT eligible? If not, why?",
            (
                "No. TTT stayed blocked because no confirmed SWA or merge/gauge carrier exists and Phase5 counterfactual failed. "
                f"ttt_allowed={final.get('ttt_allowed')}."
            ),
        ),
        (
            "12. If No-Go, is blocker no signal, semantic no-add, carrier absent, counterfactual fail, or runtime surface wrong?",
            (
                f"Final blocker: {final.get('blocker')}. Active labels: {final.get('active_decision_labels')}. "
                "The evidence chain is global no-signal for runtime, semantic no-add, native mismatch attribution failure, carrier absence, and counterfactual/good-protection failure."
            ),
        ),
    ]
    lines = [
        "# ACL2 v88TF Final Report",
        "",
        "## Final Decision",
        "",
        f"- Final status: {_md(final.get('final_status'))}",
        f"- Final No-Go allowed: {_md(final.get('final_no_go_allowed'))}",
        f"- Blocker: {_md(final.get('blocker'))}",
        f"- Active decision labels: {_md(', '.join(final.get('active_decision_labels') or []))}",
        f"- Runtime action allowed: {_md(final.get('runtime_action_allowed'))}",
        f"- TTT allowed: {_md(final.get('ttt_allowed'))}",
        "",
        "## Key Evidence",
        "",
        f"- Phase1 pair rows / sequence coverage: {_md(str(metrics.get('phase1_pair_rows')) + ' / ' + str(metrics.get('phase1_sequence_coverage')))}",
        f"- Phase2 global gate pass: {_md(phase2_all.get('phase2_mode_relevance_gate_pass'))}; best={_md(best_all.get('signal'))}; rho={_md(best_all.get('spearman_rho_abs_log_scale_jump'))}",
        f"- Phase2 highobs gate pass: {_md(phase2_highobs.get('phase2_mode_relevance_gate_pass'))}; passing={_md(phase2_highobs.get('passing_signals'))}",
        f"- Phase2 nonseq01 gate pass: {_md(phase2_nonseq01.get('phase2_mode_relevance_gate_pass'))}; best={_md(best_nonseq01.get('signal'))}",
        f"- Phase2 near gate pass: {_md(phase2_near.get('phase2_mode_relevance_gate_pass'))}; best={_md(best_near.get('signal'))}",
        f"- Phase3 native attribution gate pass: {_md(phase3.get('phase3_native_update_attribution_gate_pass'))}; blocker={_md(phase3.get('blocker'))}",
        f"- Phase4 SWA carrier pass: {_md(phase4_swa.get('swa_route_carrier_gate_pass'))}; blocker={_md(phase4_swa.get('blocker'))}",
        f"- Phase4 merge carrier pass: {_md(phase4_merge.get('merge_gauge_mode_carrier_gate_pass'))}; blocker={_md(phase4_merge.get('blocker'))}",
        f"- Phase5 counterfactual scale/raw pass: {_md(str(phase5.get('scale_label_gate_pass')) + ' / ' + str(phase5.get('raw_residual_gate_pass')))}",
        f"- Phase7 visual gate pass: {_md(visual.get('visual_integrity_gate_pass'))}; manifest/question/review={_md(str(visual.get('manifest_rows')) + ' / ' + str(visual.get('question_rows')) + ' / ' + str(visual.get('review_coverage')))}",
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
            "- Offline/GT scale labels were used only for audit metrics and visual labeling.",
            "- No runtime action, QK/TTT intervention, or merge/gauge action was run.",
            "- Split repairs are recorded as diagnostics only because the global route and later gates failed.",
            "- The execution log and recap list code changes, commands, and repaired blockers for audit.",
            "",
            "## Decision Matrix Source",
            "",
            f"- Decision matrix route eligibility: {_md(decision.get('route_eligibility'))}",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
