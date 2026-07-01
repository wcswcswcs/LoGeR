#!/usr/bin/env python3
"""Build v89 final report from the decision matrix and audited artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from v86_soft_latent_utils import read_json, write_json


DEFAULT_ROOT = Path("results/acl2_v89tf_semantic_scale_mode_observability_memory_control")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ROOT / "report_final")
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def _md(value: Any) -> str:
    return f"`{value}`"


def _best(summary: dict[str, Any]) -> dict[str, Any]:
    return summary.get("best_semantic_signal") or {}


def main() -> None:
    args = parse_args()
    root = args.root
    decision = _json(root / "phase10_decision_matrix/decision_matrix.json")
    phase1 = _json(root / "phase1_semantic_scale_mode_ledger/phase1_semantic_ledger_summary.json")
    phase1_audit = _json(root / "phase1_semantic_scale_mode_ledger/phase1_semantic_ledger_audit.json")
    phase2_all = _json(root / "phase2_semantic_mode_relevance/semantic_mode_relevance_summary.json")
    phase2_highobs = _json(root / "phase2_semantic_mode_relevance_highobs/semantic_mode_relevance_summary.json")
    phase2_far = _json(root / "phase2_semantic_mode_relevance_far/semantic_mode_relevance_summary.json")
    phase2_near = _json(root / "phase2_semantic_mode_relevance_near/semantic_mode_relevance_summary.json")
    phase2_lowobs = _json(root / "phase2_semantic_mode_relevance_semantic_lowobs/semantic_mode_relevance_summary.json")
    phase3 = _json(root / "phase3_feature_match_semantic_ruler/feature_match_audit_summary.json")
    phase4 = _json(root / "phase4_semantic_observability_policy/semantic_observability_policy_audit_summary.json")
    phase7 = _json(root / "phase7_semantic_mode_temporal_consistency/delayed_commit_policy_audit_summary.json")
    visual = _json(root / "phase10_visual_rediscovery/visual_integrity_audit.json")
    final = {
        "phase": "report_final",
        "final_status": decision.get("final_status"),
        "final_no_go_allowed": decision.get("final_no_go_allowed"),
        "runtime_action_allowed": decision.get("runtime_action_allowed"),
        "runtime_action_executed": decision.get("runtime_action_executed"),
        "ttt_allowed": decision.get("ttt_allowed"),
        "blocker": decision.get("blocker"),
        "active_decision_labels": decision.get("active_decision_labels", []),
        "carrier_tools_run": decision.get("carrier_tools_run"),
        "counterfactual_tools_run": decision.get("counterfactual_tools_run"),
        "key_metrics": decision.get("key_metrics", {}),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "final_decision.json", final)
    (args.out_dir / "final_report.md").write_text(
        build_markdown(
            final,
            decision,
            phase1,
            phase1_audit,
            phase2_all,
            phase2_highobs,
            phase2_far,
            phase2_near,
            phase2_lowobs,
            phase3,
            phase4,
            phase7,
            visual,
        ),
        encoding="utf-8",
    )
    print(f"final_status={final['final_status']}")
    print(f"final_no_go_allowed={final['final_no_go_allowed']}")
    print(f"runtime_action_allowed={final['runtime_action_allowed']}")
    print(f"runtime_action_executed={final['runtime_action_executed']}")
    print(f"ttt_allowed={final['ttt_allowed']}")
    print(f"blocker={final['blocker']}")


def build_markdown(
    final: dict[str, Any],
    decision: dict[str, Any],
    phase1: dict[str, Any],
    phase1_audit: dict[str, Any],
    phase2_all: dict[str, Any],
    phase2_highobs: dict[str, Any],
    phase2_far: dict[str, Any],
    phase2_near: dict[str, Any],
    phase2_lowobs: dict[str, Any],
    phase3: dict[str, Any],
    phase4: dict[str, Any],
    phase7: dict[str, Any],
    visual: dict[str, Any],
) -> str:
    metrics = final.get("key_metrics", {})
    best_all = _best(phase2_all)
    best_highobs = _best(phase2_highobs)
    best_far = _best(phase2_far)
    best_near = _best(phase2_near)
    best_lowobs = _best(phase2_lowobs)
    answers = [
        (
            "1. Did semantic-conditioned mode improve over geometry-only?",
            (
                "No at the global gate. Global Phase2 had no passing semantic signal: "
                f"passing={phase2_all.get('passing_semantic_signals')}; best={best_all.get('signal')} "
                f"rho={best_all.get('spearman_rho_abs_log_scale_jump')} margin={best_all.get('semantic_shuffle_margin')}. "
                f"The geometry reference was {phase2_all.get('geometry_reference_signal')} rho={phase2_all.get('geometry_reference_rho')}. "
                "Highobs and far splits passed only as diagnostics, not as runtime routes."
            ),
        ),
        (
            "2. Did semantic shuffle break the signal?",
            (
                "Not reliably. Global best semantic margin was "
                f"{best_all.get('semantic_shuffle_margin')}; Phase3 match margin was {phase3.get('semantic_shuffle_match_margin')}; "
                f"Phase4 and Phase7 shuffle margins were {phase4.get('semantic_shuffle_margin')} and {phase7.get('semantic_shuffle_margin')}. "
                "The far split had a positive margin, but it remained split diagnostic evidence."
            ),
        ),
        (
            "3. Did semantic reduce good FPR inside geometry-conflict rows?",
            (
                "Partially as a protection diagnostic, not enough for an action gate. Phase4 good_FPR was "
                f"{phase4.get('good_FPR')} versus geometry_good_FPR={phase4.get('geometry_good_FPR')}, "
                f"giving protection_margin={phase4.get('semantic_good_protection_margin')}. "
                f"However bad_recall={phase4.get('bad_recall')} and shuffle_margin={phase4.get('semantic_shuffle_margin')}, so Phase4 failed."
            ),
        ),
        (
            "4. Did feature-match evidence help, and did semantic filter matches?",
            (
                "Semantic filtering of matches was available, but it did not help scale relevance. "
                f"Matcher={phase3.get('matcher_type')}, inlier_median={phase3.get('verified_inlier_count_median')}, "
                f"match_semantic_valid_ratio_median={phase3.get('match_semantic_valid_ratio_median')}, "
                f"cross_boundary_match_ratio_median={phase3.get('cross_boundary_match_ratio_median')}. "
                f"The match-valid score rho was {phase3.get('match_valid_score_rho_abs_log_scale_jump')}, so the Phase3 gate failed."
            ),
        ),
        (
            "5. Did observability gating separate update/hold/abstain safely?",
            (
                "No. Phase4 produced states "
                f"{phase4.get('state_counts')}, but gate_pass={phase4.get('semantic_observability_policy_gate_pass')}, "
                f"bad_recall={phase4.get('bad_recall')}, good_FPR={phase4.get('good_FPR')}, "
                f"shuffle_margin={phase4.get('semantic_shuffle_margin')}."
            ),
        ),
        (
            "6. Which memory carrier, if any, carried semantic mode evidence?",
            (
                "None was tested as a pass route. The plan allows carrier tools only if Phase2/3/4 pass; they did not pass together. "
                f"carrier_tools_run={final.get('carrier_tools_run')}."
            ),
        ),
        (
            "7. Did counterfactual upper bound pass?",
            (
                "No counterfactual tool was entered, because there was no carrier candidate. "
                f"counterfactual_tools_run={final.get('counterfactual_tools_run')}. "
                "This is recorded as not-entered/no-carrier rather than a fabricated numeric counterfactual failure."
            ),
        ),
        (
            "8. Was runtime action allowed and executed?",
            (
                f"No. runtime_action_allowed={final.get('runtime_action_allowed')} and "
                f"runtime_action_executed={final.get('runtime_action_executed')}. "
                "No explicit scale multiplication, pose correction, Sim3 runtime correction, SWA route bias, merge action, or memory write was run."
            ),
        ),
        (
            "9. Was TTT allowed?",
            (
                f"No. ttt_allowed={final.get('ttt_allowed')}. "
                "TTT remains blocked because no runtime or merge/gauge action passed."
            ),
        ),
        (
            "10. What is the next route if No-Go?",
            (
                "Improve semantic source specificity before action: richer semantic ontology/object topology or better thing/stuff tracks are needed. "
                "Current compact semantic labels can produce valid-support modes after repair, but they do not produce a global, shuffle-robust, carrier-ready memory-control signal."
            ),
        ),
    ]
    lines = [
        "# ACL2 v89TF Final Report",
        "",
        "## Final Decision",
        "",
        f"- Final status: {_md(final.get('final_status'))}",
        f"- Final No-Go allowed: {_md(final.get('final_no_go_allowed'))}",
        f"- Blocker: {_md(final.get('blocker'))}",
        f"- Active decision labels: {_md(', '.join(final.get('active_decision_labels') or []))}",
        f"- Runtime action allowed/executed: {_md(str(final.get('runtime_action_allowed')) + ' / ' + str(final.get('runtime_action_executed')))}",
        f"- TTT allowed: {_md(final.get('ttt_allowed'))}",
        f"- Carrier/counterfactual tools run: {_md(str(final.get('carrier_tools_run')) + ' / ' + str(final.get('counterfactual_tools_run')))}",
        "",
        "## Key Evidence",
        "",
        f"- Phase1 ledger gate/audit: {_md(str(phase1.get('phase1_gate_pass')) + ' / ' + str(phase1_audit.get('phase1_audit_gate_pass')))}; pair/edge/seq={_md(str(phase1.get('pair_rows')) + ' / ' + str(phase1.get('edge_rows')) + ' / ' + str(phase1.get('sequence_coverage')))}",
        f"- Phase1 semantic mode type counts after repair: {_md(phase1_audit.get('semantic_mode_type_counts'))}",
        f"- Phase2 global gate pass: {_md(phase2_all.get('phase2_semantic_mode_relevance_gate_pass'))}; best={_md(best_all.get('signal'))}; rho={_md(best_all.get('spearman_rho_abs_log_scale_jump'))}; margin={_md(best_all.get('semantic_shuffle_margin'))}",
        f"- Phase2 highobs diagnostic: pass={_md(phase2_highobs.get('phase2_semantic_mode_relevance_gate_pass'))}; best={_md(best_highobs.get('signal'))}; rho={_md(best_highobs.get('spearman_rho_abs_log_scale_jump'))}; FPR={_md(best_highobs.get('good_false_positive_rate'))}",
        f"- Phase2 far diagnostic: pass={_md(phase2_far.get('phase2_semantic_mode_relevance_gate_pass'))}; best={_md(best_far.get('signal'))}; rho={_md(best_far.get('spearman_rho_abs_log_scale_jump'))}; FPR={_md(best_far.get('good_false_positive_rate'))}",
        f"- Phase2 near diagnostic: pass={_md(phase2_near.get('phase2_semantic_mode_relevance_gate_pass'))}; best={_md(best_near.get('signal'))}; margin={_md(best_near.get('semantic_shuffle_margin'))}",
        f"- Phase2 lowobs diagnostic: pass={_md(phase2_lowobs.get('phase2_semantic_mode_relevance_gate_pass'))}; best={_md(best_lowobs.get('signal'))}; margin={_md(best_lowobs.get('semantic_shuffle_margin'))}",
        f"- Phase3 feature-match gate pass: {_md(phase3.get('feature_match_semantic_ruler_gate_pass'))}; matcher={_md(phase3.get('matcher_type'))}; valid_ratio={_md(phase3.get('match_semantic_valid_ratio_median'))}; rho={_md(phase3.get('match_valid_score_rho_abs_log_scale_jump'))}",
        f"- Phase4 observability gate pass: {_md(phase4.get('semantic_observability_policy_gate_pass'))}; recall/FPR={_md(str(phase4.get('bad_recall')) + ' / ' + str(phase4.get('good_FPR')))}; protection={_md(phase4.get('semantic_good_protection_margin'))}",
        f"- Phase7 delayed commit gate pass: {_md(phase7.get('delayed_commit_policy_gate_pass'))}; recall/FPR={_md(str(phase7.get('bad_recall')) + ' / ' + str(phase7.get('good_FPR')))}",
        f"- Phase10 visual gate pass: {_md(visual.get('visual_integrity_gate_pass'))}; manifest/question/review={_md(str(visual.get('manifest_rows')) + ' / ' + str(visual.get('question_rows')) + ' / ' + str(visual.get('review_coverage')))}",
        "",
        "## Required Questions",
        "",
    ]
    for question, answer in answers:
        lines.extend([f"### {question}", "", answer, ""])
    lines.extend(
        [
            "## Evidence Discipline",
            "",
            "- Offline/GT scale labels were used only for audit scoring and visual labels.",
            "- Phase2 highobs/far successes are explicitly diagnostic and do not authorize runtime.",
            "- Carrier and counterfactual tools were not run because their plan preconditions failed.",
            "- No runtime action or TTT write was executed.",
            "- The semantic role repair is recorded in the execution log and recap; it changed compact-label interpretation, not any threshold after seeing final pass/fail.",
            "",
            "## Decision Matrix Source",
            "",
            f"- Route eligibility: {_md(decision.get('route_eligibility'))}",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
