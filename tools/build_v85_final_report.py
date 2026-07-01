#!/usr/bin/env python3
"""Build v85 final No-Go report from audited phase artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/acl2_v85tf_latent_anchor_alignment_pairwise_memory_ruler")
DEFAULT_OUT_DIR = DEFAULT_ROOT / "report_final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root
    out = args.out_dir
    phase0 = read_json(root / "phase0_evidence_lock/phase0_gate_summary.json")
    phase1 = read_json(root / "phase1_anchor_pair_universe/anchor_pair_sufficiency_summary.json")
    phase2 = read_json(root / "phase2_qk_feature_bank/feature_sanity_summary.json")
    decision = read_json(root / "phase11_decision_matrix/decision_matrix.json")
    visual = read_json(root / "phase12_visual_rediscovery/visual_integrity_audit.json")
    final = {
        "phase": "report_final",
        "final_status": "No-Go_before_runtime_action",
        "final_no_go_allowed": bool(decision.get("final_no_go_allowed", False)),
        "runtime_action_allowed": False,
        "ttt_allowed": False,
        "blocker": decision.get("no_go_blocker") or "unknown",
        "active_decision_labels": decision.get("active_decision_labels", []),
        "phase0_gate_pass": bool(phase0.get("phase0_gate_pass", False)),
        "phase1_gate_pass": bool(phase1.get("phase1_gate_pass", False)),
        "phase2_feature_gate_pass": bool(phase2.get("phase2_feature_gate_pass", False)),
        "visual_audit_gate_pass": bool(visual.get("visual_audit_gate_pass", False)),
        "key_metrics": {
            "phase1_row_count": phase1.get("row_count"),
            "phase1_pair_row_count": phase1.get("pair_row_count"),
            "adjacent_labelled_rows": phase1.get("adjacent_labelled_rows"),
            "sequence_coverage": phase1.get("sequence_coverage"),
            "strong_bad_pair_rows": phase1.get("strong_bad_pair_rows"),
            "feature_q_availability_ratio": phase1.get("feature_q_availability_ratio"),
            "feature_k_availability_ratio": phase1.get("feature_k_availability_ratio"),
            "phase2_eligible_layer_head_count": phase2.get("eligible_layer_head_count"),
            "visual_manifest_rows": visual.get("manifest_rows"),
            "visual_review_coverage": visual.get("review_coverage"),
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "final_decision.json", final)
    (out / "final_report.md").write_text(build_markdown(final, phase1, phase2, decision, visual), encoding="utf-8")
    print(f"final_status={final['final_status']}")
    print(f"final_no_go_allowed={str(final['final_no_go_allowed']).lower()}")
    print(f"blocker={final['blocker']}")
    print(f"runtime_action_allowed={str(final['runtime_action_allowed']).lower()}")
    print(f"ttt_allowed={str(final['ttt_allowed']).lower()}")


def build_markdown(
    final: dict[str, Any],
    phase1: dict[str, Any],
    phase2: dict[str, Any],
    decision: dict[str, Any],
    visual: dict[str, Any],
) -> str:
    answers = [
        (
            "1. Did reliable anchor pairs exist in labelled bad/good adjacent pairs?",
            (
                "Not sufficiently for labelled bad pairs. Phase1 had "
                f"strong_bad_pair_rows={phase1.get('strong_bad_pair_rows')} with required >=5; "
                f"positive_anchor_rows={phase1.get('positive_anchor_rows')} overall."
            ),
        ),
        (
            "2. Did current-Q/cache-K identity residual separate bad/good?",
            (
                "Not evaluated as a separator because Phase1 failed. Phase2 feature sanity extracted "
                f"{phase2.get('feature_entry_count')} Q/K entries and passed one pooled layer/head, but Phase3 was blocked."
            ),
        ),
        (
            "3. Did low-free C reduce held-out residual beyond random/shuffle controls?",
            "No C was fit. Phase1 failed before alignment, so held-out controls were not run.",
        ),
        (
            "4. Was the best C stable, near-identity/orthogonal, and not overfit?",
            "Not evaluated because no C candidate was fit.",
        ),
        (
            "5. Did alignment relate to offline adjacent log-scale jump?",
            "Not evaluated because alignment did not run.",
        ),
        (
            "6. Did aligned pair score lift true SWA QK route?",
            "Not evaluated. Phase2 has direct PCA Q/cache-K features but no true route mass test.",
        ),
        (
            "7. Was route lift pairwise QK-specific or just source-side mask-like?",
            "Not evaluated because route carrier testing was blocked before Phase5.",
        ),
        (
            "8. Did conservative QK pair bias improve J_SWA or components?",
            "No runtime action was run; runtime_action_allowed=false.",
        ),
        (
            "9. Did semantic compatibility add value over geometry-only pair alignment?",
            "Not evaluated because Phase3/controls did not run.",
        ),
        (
            "10. If SWA route failed, did merge/gauge aligned-pair weighting help?",
            "Not evaluated. Merge/gauge fallback was not triggered because SWA alignment/route/scale evidence never passed.",
        ),
        (
            "11. Did TTT run only after confirmed SWA/merge aligned evidence?",
            "TTT did not run; ttt_allowed=false throughout.",
        ),
        (
            "12. Were good cases protected?",
            "No runtime action ran. Phase12 panels identified good-case positive-anchor risk, so protection remains a required future gate.",
        ),
        (
            "13. Did any candidate pass held-out or official validation?",
            "No candidate reached runtime, held-out, or official validation.",
        ),
        (
            "14. If No-Go, what is the blocker?",
            "No-Go blocker is D1_ANCHOR_PAIR_INSUFFICIENT / strong_bad_support_insufficient, with visual audit complete.",
        ),
    ]
    lines = [
        "# ACL2 v85TF Final Report",
        "",
        "## Final Decision",
        "",
        f"- Final status: `{final['final_status']}`",
        f"- Blocker: `{final['blocker']}`",
        f"- Active decision labels: `{', '.join(final['active_decision_labels'])}`",
        f"- Final No-Go allowed: `{final['final_no_go_allowed']}`",
        f"- Runtime action allowed: `{final['runtime_action_allowed']}`",
        f"- TTT allowed: `{final['ttt_allowed']}`",
        "",
        "## Key Evidence",
        "",
        f"- Phase1 gate pass: `{phase1.get('phase1_gate_pass')}`",
        f"- Phase1 fail reasons: `{phase1.get('fail_reasons')}`",
        f"- Adjacent labelled rows: `{phase1.get('adjacent_labelled_rows')}`",
        f"- Sequence coverage: `{phase1.get('sequence_coverage')}`",
        f"- Anchor count pass ratio: `{phase1.get('anchor_pair_count_pass_ratio')}`",
        f"- Strong bad pair rows: `{phase1.get('strong_bad_pair_rows')}`",
        f"- Q/K availability ratio: `{phase1.get('feature_q_availability_ratio')}`, `{phase1.get('feature_k_availability_ratio')}`",
        f"- Phase2 feature gate pass: `{phase2.get('phase2_feature_gate_pass')}`",
        f"- Phase2 eligible layer/head count: `{phase2.get('eligible_layer_head_count')}`",
        f"- Visual audit gate pass: `{visual.get('visual_audit_gate_pass')}`",
        f"- Visual review coverage: `{visual.get('review_coverage')}`",
        "",
        "## Required Questions",
        "",
    ]
    for question, answer in answers:
        lines.extend([f"### {question}", "", answer, ""])
    lines.extend(
        [
            "## Conclusion",
            "",
            (
                "v85 should stop before runtime action. Direct PCA SWA Q/cache-K features are available and sane, "
                "but reliable strong anchor support is too sparse in labelled bad adjacent pairs. Phase12 visual "
                "audit now supports this No-Go blocker without claiming unrun alignment, route, merge, or TTT evidence."
            ),
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
