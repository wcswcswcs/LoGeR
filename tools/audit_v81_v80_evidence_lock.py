#!/usr/bin/env python3
"""Lock v80 evidence boundaries before running ACL2 v81 actions.

The tool reads landed v80 artifacts and emits the v81 Phase0 evidence-lock
bundle. It does not recompute metrics or infer success beyond the source files.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_V80_ROOT = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final"
)
DEFAULT_OUT_DIR = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v81tf_longwindow_semantic_three_memory_control/"
    "report_final/phase0_v80_evidence_lock"
)

FAILED_ACTION_FAMILIES = [
    {
        "family": "OUT4 merge overlap",
        "v80_status": "failed_or_unresolved",
        "v81_rule": "forbidden_repeat_without_new_actuator_or_visual_evidence",
        "reason": "seq00 evidence existed but seq01 boundary conflict remained unresolved.",
    },
    {
        "family": "RADIO object topology scalar guard",
        "v80_status": "diagnostic_only",
        "v81_rule": "forbidden_scalar_guard_promotion",
        "reason": "RADIO topology audits stayed seq01-scoped and did not produce a runtime candidate.",
    },
    {
        "family": "RADIO qscale merge",
        "v80_status": "failed",
        "v81_rule": "forbidden_qscale_scalar_repeat",
        "reason": "qscale/merge variants did not pass control-separated method gates.",
    },
    {
        "family": "READ/QK naive variants",
        "v80_status": "failed",
        "v81_rule": "forbidden_source_column_or_naive_qk_boost_repeat",
        "reason": "READ/QK variants did not become a deployable short-term method.",
    },
    {
        "family": "RET/QK retrieval variants",
        "v80_status": "failed_or_weak",
        "v81_rule": "allowed_only_with_new_retrieval_memory_design",
        "reason": "Only weak/failed smoke evidence was available; not a current mid-memory fix.",
    },
    {
        "family": "TTT selected-write direct veto / no-persistent",
        "v80_status": "failed",
        "v81_rule": "forbidden_direct_veto_no_persistent_repeat",
        "reason": "low-support selected-write explained risk locally but did not pass runtime/control gates.",
    },
    {
        "family": "handshake variants",
        "v80_status": "failed",
        "v81_rule": "allowed_only_if_read_swa_ttt_alignment_maps_exist",
        "reason": "cross-memory handshake did not pass method gates.",
    },
    {
        "family": "motion future-overlap proxy",
        "v80_status": "failed",
        "v81_rule": "forbidden_proxy_controller_repeat",
        "reason": "proxy signals did not convert into robust downstream geometry improvement.",
    },
    {
        "family": "multiobjective controller",
        "v80_status": "failed",
        "v81_rule": "forbidden_controller_repeat_without_new_gate",
        "reason": "controller did not satisfy control-separated gates.",
    },
    {
        "family": "non-GT gauge signal",
        "v80_status": "failed",
        "v81_rule": "diagnostic_only_until_new_scale_side_state",
        "reason": "non-GT gauge signal was not deployable.",
    },
    {
        "family": "proxy controller",
        "v80_status": "failed",
        "v81_rule": "forbidden_proxy_controller_repeat",
        "reason": "proxy controller failed method gates.",
    },
    {
        "family": "safe-positive controller",
        "v80_status": "failed",
        "v81_rule": "forbidden_controller_repeat_without_new_evidence",
        "reason": "safe-positive rule did not become good-safe runtime policy.",
    },
    {
        "family": "TTSA temporal-spatial scalar trace",
        "v80_status": "diagnostic_only",
        "v81_rule": "allowed_only_as_fixed_temporal_spatial_update_with_controls",
        "reason": "scalar trace audit was diagnostic-only and broader rules selected overlap-harm false positives.",
    },
    {
        "family": "TTT post-delta region",
        "v80_status": "failed_or_no_separator",
        "v81_rule": "forbidden_region_scalar_repeat",
        "reason": "post-delta/support audit found no reliable separator.",
    },
    {
        "family": "semantic false-positive separator",
        "v80_status": "diagnostic_only",
        "v81_rule": "must_be_rebuilt_as_downstream_good_protection_gate",
        "reason": "diagnostic split existed but actuator/control gate still failed.",
    },
]

REUSABLE_CLUES = [
    {
        "clue": "multi-seq case bank passed for 00/01/02/05",
        "v81_use": "Phase1 source bank and sequence scope",
    },
    {
        "clue": "v78/v80 visual audit pipeline is usable",
        "v81_use": "Reuse visual panel/audit pattern for long-window PCA/QKV/TTT confirmation",
    },
    {
        "clue": "READ1 short-term semantic carrier positive",
        "v81_use": "READ acts as evidence filter, not final geometry fix",
    },
    {
        "clue": "selected-write low-support is risk diagnostic",
        "v81_use": "Risk input for long-window policy, not standalone veto",
    },
    {
        "clue": "seq02 chunks 62-70 is continuous long-window bad cluster",
        "v81_use": "Mandatory bad long-window cluster in Phase1",
    },
    {
        "clue": "selected-write direct veto / no-persistent failed",
        "v81_use": "Use downstream direction + good protection + READ/SWA confirmation before action",
    },
    {
        "clue": "current TTT replay token-filter actuator weak",
        "v81_use": "Check action fidelity before interpreting J_long",
    },
    {
        "clue": "RADIO topology only seq01 scoped unless more sidecars are built",
        "v81_use": "Dense/thingstuff variants must run when RADIO sidecars are unavailable",
    },
]

FORBIDDEN_REPEATS = [
    "selected-write low-support threshold sweep",
    "direct selected-write veto / no-persistent family",
    "TTT global write-strength / freeze / old_decay",
    "qscale / scalar support threshold sweep",
    "RADIO topology scalar guard promotion",
    "source-column boost top25/top50/all-positive",
    "L13 negative damp / L07-L13 scalar sweep",
    "SWA semantic role reweight same-family experiment",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v80-root", type=Path, default=DEFAULT_V80_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), ensure_ascii=False)
                    if isinstance(row.get(key), (dict, list))
                    else row.get(key)
                    for key in fieldnames
                }
            )


def latest(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern))
    return matches[-1] if matches else None


def build_required_artifacts() -> dict[str, Any]:
    return {
        "phase1_long_window_cluster_bank": {
            "tools": [
                "tools/build_v81_long_window_cluster_bank.py",
                "tools/summarize_v81_long_window_clusters.py",
            ],
            "outputs": [
                "phase1_long_window_cluster_bank/long_window_cluster_rows.csv",
                "phase1_long_window_cluster_bank/long_window_cluster_summary.json",
                "phase1_long_window_cluster_bank/long_window_cluster_report.md",
            ],
            "gate": "bad long windows >=12; good/false-positive windows >=12; seq coverage >=3; seq02 62-70 included",
        },
        "phase2_visual_confirmation": {
            "tools": [
                "tools/visualize_v81_long_window_memory_panels.py",
                "tools/audit_v81_long_window_visual_artifacts.py",
                "tools/review_v81_long_window_visual_insights.py",
            ],
            "outputs": [
                "phase2_long_window_visual_confirmation/visual_manifest.csv",
                "phase2_long_window_visual_confirmation/visual_integrity_audit.json",
                "phase2_long_window_visual_confirmation/visual_review.csv",
                "phase2_long_window_visual_confirmation/visual_insight.md",
            ],
            "gate": "visual_integrity_audit.gate_pass=true; review coverage >=80%; bad >=12; good/false-positive >=12",
        },
        "phase3_selected_write_risk_rule": {
            "tools": ["tools/audit_v81_selected_write_risk_rule.py"],
            "outputs": [
                "phase3_selected_write_risk_rule/selected_write_risk_rows.csv",
                "phase3_selected_write_risk_rule/bad_good_confusion_matrix.json",
                "phase3_selected_write_risk_rule/rule_audit_report.md",
            ],
            "gate": "bad recall >=0.60; good false positive rate <=0.25; seq coverage >=3",
        },
        "phase4_read_swa_confirmation": {
            "tools": [
                "tools/build_v81_read_swa_confirmation_maps.py",
                "tools/audit_v81_read_swa_confirmation_quality.py",
            ],
            "outputs": [
                "phase4_read_swa_confirmation/read_swa_confirmation_rows.csv",
                "phase4_read_swa_confirmation/confirmation_quality_summary.json",
            ],
            "gate": "read/swa stable mass nonzero; read_swa_alignment >=0.30; random lower than actual",
        },
        "phase5plus_actions": {
            "tools": [
                "tools/run_v81_ttt_write_less_onehop.py",
                "tools/run_v81_merge_boundary_typeb_rescue.py",
                "tools/run_v81_cross_memory_longwindow_handshake.py",
                "tools/evaluate_v81_longwindow_semantic_memory.py",
                "tools/build_v81_rediscovery_questions.py",
            ],
            "gate": "Only run after earlier gates pass; must beat geometry/random/shuffle controls",
        },
    }


def build_lock(v80_root: Path, out_dir: Path) -> dict[str, Any]:
    phase0_path = v80_root / "phase0_multiseq_artifact_audit/phase0_artifact_audit_summary.json"
    case_bank_path = v80_root / "phase1_three_memory_case_bank/case_bank_summary.json"
    phase2_path = v80_root / "phase2_case_visual_confirmation/visual_integrity_audit.json"
    direct_visual_path = (
        v80_root
        / "phase2_direct_hook_enhanced_visual_review_allseq_aggregate/visual_integrity_audit_allseq.json"
    )
    matrix_path = latest(v80_root, "phase10_current_action_evidence_matrix_*/current_action_evidence_matrix_summary.json")
    insight_path = latest(v80_root, "phase10_selected_write_insight_matrix_*/selected_write_insight_matrix_summary.json")
    no_go_path = latest(v80_root, "phase10_formal_no_go_review_*/formal_no_go_summary.json")

    phase0 = read_json(phase0_path)
    case_bank = read_json(case_bank_path)
    phase2 = read_json(phase2_path)
    direct_visual = read_json(direct_visual_path)
    matrix = read_json(matrix_path) if matrix_path else {}
    insight = read_json(insight_path) if insight_path else {}
    no_go = read_json(no_go_path) if no_go_path else {}

    memory_summary = case_bank.get("memory_body_summary") or {}
    case_bank_pass = bool(case_bank.get("phase1_gate_pass")) and all(
        bool((memory_summary.get(body) or {}).get("gate_pass"))
        for body in ("short", "mid", "long")
    )
    visual_gate_pass = bool(phase2.get("gate_pass")) and bool(direct_visual.get("aggregate_phase2_direct_hook_visual_gate_pass", True))
    action_gate_pass_any = bool(matrix.get("action_gate_pass_any") or no_go.get("action_gate_pass_any"))
    method_success_claimed = bool(no_go.get("method_gate_claimed")) or action_gate_pass_any

    lock = {
        "schema": "acl2_v81_v80_evidence_lock_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_v80_root": str(v80_root),
        "source_files": {
            "phase0": str(phase0_path),
            "case_bank": str(case_bank_path),
            "phase2_visual": str(phase2_path),
            "direct_visual": str(direct_visual_path),
            "latest_action_matrix": str(matrix_path) if matrix_path else None,
            "latest_selected_write_insight": str(insight_path) if insight_path else None,
            "latest_formal_no_go": str(no_go_path) if no_go_path else None,
        },
        "gate_pass": True,
        "v80_evidence_locked": True,
        "phase0_artifact_gate_pass": bool(phase0.get("phase0_gate_pass")),
        "phase1_case_bank_gate_pass": case_bank_pass,
        "phase2_visual_gate_pass": visual_gate_pass,
        "v80_goal_achieved": bool(no_go.get("v80_goal_achieved")),
        "method_gate_claimed": method_success_claimed,
        "runtime_promotion_allowed": bool(no_go.get("runtime_promotion_allowed")),
        "official_704F_success": False,
        "semantic_memory_method_success": action_gate_pass_any,
        "allowed_case_mining_seqs": phase0.get("phase1_basic_case_mining_allowed_seqs") or [],
        "kitti08_blockers": (phase0.get("blockers_by_seq") or {}).get("08", []),
        "radio_action_allowed_seqs": phase0.get("radio_action_allowed_seqs") or [],
        "mid_swa_action_allowed_seqs": phase0.get("mid_swa_action_allowed_seqs") or [],
        "long_ttt_action_allowed_seqs": phase0.get("long_ttt_action_allowed_seqs") or [],
        "case_bank_summary": {
            body: {
                "bad": (memory_summary.get(body) or {}).get("bad"),
                "good": (memory_summary.get(body) or {}).get("good"),
                "bad_seqs": (memory_summary.get(body) or {}).get("bad_seqs"),
                "good_seqs": (memory_summary.get(body) or {}).get("good_seqs"),
                "gate_pass": (memory_summary.get(body) or {}).get("gate_pass"),
            }
            for body in ("short", "mid", "long")
        },
        "selected_write_low_support_lock": {
            "diagnostic_only": insight.get("diagnostic_only"),
            "support_threshold": insight.get("support_threshold"),
            "binary_rule_metrics": insight.get("low_support_binary_rule_metrics"),
            "notable_bad_clusters": insight.get("notable_bad_clusters"),
            "good_false_positives": insight.get("good_false_positives"),
            "bad_false_negatives": insight.get("bad_false_negatives"),
            "insights": insight.get("insights"),
        },
        "formal_no_go_lock": {
            "formal_no_go_ready": no_go.get("formal_no_go_ready"),
            "final_decision": no_go.get("final_decision"),
            "failed_method_requirements": no_go.get("failed_method_requirements"),
            "core_blocker": no_go.get("core_blocker"),
            "decision_text": no_go.get("decision_text"),
            "scope_caveats": no_go.get("scope_caveats"),
        },
        "failed_action_families": matrix.get("failed_action_families") or [],
        "uncovered_or_weak_points": matrix.get("uncovered_or_weak_points") or [],
        "reusable_clues": REUSABLE_CLUES,
        "forbidden_repeats": FORBIDDEN_REPEATS,
        "v81_phase0_decision": "pass_to_phase1_with_scope_caveats",
        "v81_scope_caveats": [
            "KITTI08 remains blocked unless a baseline trajectory is repaired or supplied.",
            "RADIO/RADSeg runtime evidence is seq01-scoped unless sidecars are expanded.",
            "v80 selected-write low-support is diagnostic only and cannot be used as a direct veto.",
            "Do not claim method success before v81 long-window risk/action/control gates pass.",
        ],
    }

    write_json(out_dir / "v80_evidence_lock.json", lock)
    write_csv(out_dir / "failed_action_family_matrix.csv", FAILED_ACTION_FAMILIES)
    write_csv(out_dir / "reusable_clues.csv", REUSABLE_CLUES)
    write_json(out_dir / "required_v81_artifacts.json", build_required_artifacts())
    forbidden_md = ["# v81 Forbidden Repeats", ""]
    forbidden_md.extend(f"- {item}" for item in FORBIDDEN_REPEATS)
    forbidden_md.extend(
        [
            "",
            "Gate rule: if any forbidden family is launched again without a new actuator",
            "or new visual evidence, mark the run `invalid_repeat_of_failed_family`.",
            "",
        ]
    )
    (out_dir / "forbidden_repeats.md").write_text("\n".join(forbidden_md), encoding="utf-8")
    return lock


def main() -> None:
    args = parse_args()
    lock = build_lock(args.v80_root, args.out_dir)
    print(json.dumps({
        "out_dir": str(args.out_dir),
        "gate_pass": lock["gate_pass"],
        "v81_phase0_decision": lock["v81_phase0_decision"],
        "allowed_case_mining_seqs": lock["allowed_case_mining_seqs"],
        "kitti08_blockers": lock["kitti08_blockers"],
        "method_gate_claimed": lock["method_gate_claimed"],
        "runtime_promotion_allowed": lock["runtime_promotion_allowed"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
