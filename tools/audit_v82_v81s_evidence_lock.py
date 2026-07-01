#!/usr/bin/env python3
"""Build the ACL2 v82 Phase0 evidence-lock bundle from landed v81S artifacts.

The script is intentionally read-only with respect to source evidence: it reads
existing docs/results and emits a compact Phase0 contract. It does not recompute
model metrics and does not promote prior proxy evidence to method success.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_V81S_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v81s_swa_first_semantic_scale_memory_control/report_final"
)
DEFAULT_OUT_DIR = Path(
    "results/acl2_v82tf_swa_carrier_semantic_scale_handoff/phase0_v81s_evidence_lock"
)

SOURCE_DOCS = [
    Path("docs/ACL2_v81S_Updated_SWAFirst_SemanticScaleMemoryControl_实验结果复盘.md"),
    Path("docs/ACL2_v81S_Updated_SWAFirst_SemanticScaleMemoryControl_执行日志.md"),
    Path("docs/ACL2_v81TF_LongWindow_SemanticThreeMemoryControl_实验结果复盘.md"),
    Path("docs/ACL2_v80TF_MultiSeq_GoodBadCase_SemanticThreeMemoryControl_实验结果复盘.md"),
    Path("docs/ACL2_v78TF_v3_PCA_Grounded_AuditableVisualRediscovery_实验结果复盘.md"),
    Path("docs/ACL2_v67_DenseSemantic_Reconstruction_Emergency_实验结果复盘.md"),
    Path("docs/ACL2_v65_C9_H35_TransitionSwap_MergeGauge_Attribution_实验结果复盘.md"),
]

FORBIDDEN_REPEATS = [
    {
        "family_id": "P9_40_SOURCE_REPLACE_STABLE_V_ALPHA_SWEEP",
        "family": "P9_40 source_replace_stable_v alpha sweep",
        "prior_status": "v81S route/action fidelity present but geometry gate failed",
        "v82_rule": "invalid_repeat_of_failed_family unless a new actuator or new visual carrier evidence is supplied",
        "allowed_v82_form": "not allowed as another source_replace/gate alpha sweep",
        "reason": "bad median improvements were only 0.00x-level and no row passed the metric/control gate.",
    },
    {
        "family_id": "P9_6_SOURCE_GATE_ROLE_NEGATIVE_V_ALPHA_SWEEP",
        "family": "P9_6 source_gate_role_negative_v alpha sweep",
        "prior_status": "v81S route/action fidelity present but geometry gate failed",
        "v82_rule": "invalid_repeat_of_failed_family unless a new actuator or new visual carrier evidence is supplied",
        "allowed_v82_form": "not allowed as another source_replace/gate alpha sweep",
        "reason": "source-gate role-negative V did not satisfy v81S bad improvement thresholds.",
    },
    {
        "family_id": "SOURCE_COLUMN_TOPK_BOOST_DAMP",
        "family": "source-column top-k boost / damp",
        "prior_status": "failed old SWA action family",
        "v82_rule": "invalid_repeat_of_failed_family",
        "allowed_v82_form": "replace with Q/K/V side-specific route or pair-biased carrier after Phase4",
        "reason": "v82 requires query-conditioned and side-specific carrier localization, not broad source-column edits.",
    },
    {
        "family_id": "L13_L07_SCALAR_CONTRAST",
        "family": "L13 negative damp / L07-L13 scalar contrast",
        "prior_status": "failed scalar/reweight family",
        "v82_rule": "invalid_repeat_of_failed_family",
        "allowed_v82_form": "only if Phase4 identifies a specific head/layer carrier and Phase3 visual evidence confirms it",
        "reason": "plan forbids more layer scalar contrast without carrier evidence.",
    },
    {
        "family_id": "SWA_SEMANTIC_ROLE_REWEIGHT",
        "family": "SWA semantic role reweight same-family experiment",
        "prior_status": "failed/unsupported old family",
        "v82_rule": "invalid_repeat_of_failed_family",
        "allowed_v82_form": "only as confidence/scale/QK rule after Phase5 rule audit passes controls",
        "reason": "semantic role weights alone are not proven scale/gauge carriers.",
    },
    {
        "family_id": "SELECTED_WRITE_LOW_SUPPORT_DIRECT_VETO",
        "family": "selected-write low-support direct veto / no-persistent threshold",
        "prior_status": "diagnostic only; failed direct TTT promotion",
        "v82_rule": "invalid_repeat_of_failed_family",
        "allowed_v82_form": "TTT can only write SWA- or merge-confirmed stable evidence after Phase9 entry gate",
        "reason": "v82 explicitly stops direct selected-write low-support -> TTT promotion.",
    },
    {
        "family_id": "TTT_GLOBAL_SCALAR_WRITE",
        "family": "TTT global write-strength / old_decay / freeze",
        "prior_status": "failed scalar family",
        "v82_rule": "invalid_repeat_of_failed_family",
        "allowed_v82_form": "confirmed write-less / one-hop only after SWA or merge/gauge passes",
        "reason": "TTT is downstream handoff in v82, not the first actuator.",
    },
    {
        "family_id": "QSCALE_SUPPORT_RADIO_SCALAR_GUARD",
        "family": "qscale / support threshold / RADIO scalar guard",
        "prior_status": "failed or diagnostic-only scalar guard family",
        "v82_rule": "invalid_repeat_of_failed_family",
        "allowed_v82_form": "RADIO may enter object topology / QK compatibility fields if sidecar is available and marked",
        "reason": "RADIO/RADSeg must not be promoted as an unvalidated scalar runtime guard.",
    },
]

REUSABLE_CLUES = [
    {
        "clue": "00/01/02/05 fixed prefix geometry repair exists",
        "source": "v81S PhaseS1",
        "v82_use": "reuse as overlap artifact input instead of rebuilding from scratch unless Phase1 repair demands it",
    },
    {
        "clue": "default overlap pair materialization produced 21 pair files per seq",
        "source": "v81S PhaseS1",
        "v82_use": "default source for high-confidence overlap audit; seq01 remains sparse under default confidence",
    },
    {
        "clue": "seq01 default median_saved_pairs_per_overlap=871.0",
        "source": "v81S PhaseS1",
        "v82_use": "treat as low-observability/high-sparsity case, not as ordinary high-quality evidence",
    },
    {
        "clue": "seq01 min_conf0 quantity gate passes but either_zero_ratio is high",
        "source": "v81S minconf0 confidence audit",
        "v82_use": "min_conf0 may be low-confidence stress or risk source only, never stable evidence without confidence bins",
    },
    {
        "clue": "S2 adjacent case bank has 24 rows: 12 bad and 12 good over 4 sequences",
        "source": "v81S PhaseS2",
        "v82_use": "rebuild pair bank v2 with carrier fields; do not claim current bank has READ/SWA carrier fields",
    },
    {
        "clue": "S3 visual rows cover 24 pairs but true runtime route masks cover only 8 pair smoke",
        "source": "v81S PhaseS3/S5",
        "v82_use": "Phase3 must fill 24/24 true route/QKV panels before action",
    },
    {
        "clue": "P9_40/P9_6 action fidelity exists but geometry gate fails",
        "source": "v81S PhaseS5",
        "v82_use": "old source_replace/source_gate action family is locked as failed; new actuator must be Q/K/V/head/layer/merge specific",
    },
    {
        "clue": "v65 merge/gauge hook/no-op and donor replay evidence exists",
        "source": "v65 recap",
        "v82_use": "Phase8 fallback route if SWA carrier is not localized or route moves without geometry",
    },
]

CONTRACT_ROWS = [
    {
        "phase": "Phase0",
        "candidate_family": "old source_replace/source_gate",
        "allowed": "false",
        "required_evidence_or_entry_gate": "new actuator or visual carrier evidence; otherwise invalid_repeat_of_failed_family",
        "must_beat_controls": "not_applicable",
        "promotion_scope": "none",
    },
    {
        "phase": "Phase1",
        "candidate_family": "overlap artifact stratification",
        "allowed": "true",
        "required_evidence_or_entry_gate": "confidence bins and seq01 default/minconf0 boundary classification",
        "must_beat_controls": "not_applicable",
        "promotion_scope": "artifact quality only",
    },
    {
        "phase": "Phase2",
        "candidate_family": "SWA pair bank v2",
        "allowed": "true",
        "required_evidence_or_entry_gate": "bad>=12 good/false-positive>=12 coverage>=3 carrier fields>=70% for non-RADIO fields",
        "must_beat_controls": "not_applicable",
        "promotion_scope": "case bank only",
    },
    {
        "phase": "Phase3",
        "candidate_family": "true route / QKV visual confirmation",
        "allowed": "true",
        "required_evidence_or_entry_gate": "24/24 visual rows; true_route_mask_present >=90% reviewed; missing_overlay_count=0",
        "must_beat_controls": "actual_vs_random_difference reviewed",
        "promotion_scope": "visual evidence only",
    },
    {
        "phase": "Phase4",
        "candidate_family": "carrier ledger",
        "allowed": "true",
        "required_evidence_or_entry_gate": "one of K-risk, V-protect, Q-conditioned, context-floor passes with coverage>=3",
        "must_beat_controls": "same-head random and shuffled semantic",
        "promotion_scope": "carrier selection",
    },
    {
        "phase": "Phase5",
        "candidate_family": "semantic-scale rule audit",
        "allowed": "true",
        "required_evidence_or_entry_gate": "bad recall>=0.60 good FPR<=0.25 coverage>=3",
        "must_beat_controls": "same-route-mass random and semantic shuffle",
        "promotion_scope": "action eligibility",
    },
    {
        "phase": "Phase6",
        "candidate_family": "post-route conservative SWA",
        "allowed": "conditional",
        "required_evidence_or_entry_gate": "Phase3/4/5 pass",
        "must_beat_controls": "geometry-only, randoms, semantic shuffle, RADIO shuffle if used",
        "promotion_scope": "SWA adjacent method only",
    },
    {
        "phase": "Phase7",
        "candidate_family": "Q-conditioned / pre-softmax SWA",
        "allowed": "conditional",
        "required_evidence_or_entry_gate": "Phase6 fidelity pass but metric fail/weak and Phase4 supports Q/K carrier",
        "must_beat_controls": "geometry-only QK, same-pair-mass random, semantic shuffle, RADIO shuffle if used",
        "promotion_scope": "SWA adjacent method only",
    },
    {
        "phase": "Phase8",
        "candidate_family": "merge/gauge fallback",
        "allowed": "conditional",
        "required_evidence_or_entry_gate": "SWA carrier fail or SWA route fidelity pass but geometry fail",
        "must_beat_controls": "geometry-only and random/shuffle",
        "promotion_scope": "merge/gauge confirmed stable evidence",
    },
    {
        "phase": "Phase9",
        "candidate_family": "TTT after SWA/merge confirmation",
        "allowed": "conditional",
        "required_evidence_or_entry_gate": "Phase6 or Phase7 SWA pass OR Phase8 merge/gauge pass",
        "must_beat_controls": "geometry-only, same-write-mass random, semantic shuffle",
        "promotion_scope": "long-window only",
    },
]

REQUIRED_NEXT_ARTIFACTS = [
    ("Phase0", "phase0_v81s_evidence_lock/evidence_lock.json"),
    ("Phase0", "phase0_v81s_evidence_lock/reusable_clues.csv"),
    ("Phase0", "phase0_v81s_evidence_lock/forbidden_repeats.md"),
    ("Phase0", "phase0_v81s_evidence_lock/required_next_artifacts.md"),
    ("Phase0", "phase0_v81s_evidence_lock/v82_candidate_family_contract.csv"),
    ("Phase1", "phase1_overlap_quality_stratification/overlap_quality_by_pair.csv"),
    ("Phase1", "phase1_overlap_quality_stratification/overlap_quality_by_seq.json"),
    ("Phase1", "phase1_overlap_quality_stratification/confidence_bin_summary.csv"),
    ("Phase1", "phase1_overlap_quality_stratification/seq01_sparse_support_diagnosis.md"),
    ("Phase2", "phase2_swa_pair_bank_v2/swa_pair_bank_v2.csv"),
    ("Phase2", "phase2_swa_pair_bank_v2/swa_pair_bank_v2_summary.json"),
    ("Phase3", "phase3_swa_true_route_visual_confirmation/visual_manifest.csv"),
    ("Phase3", "phase3_swa_true_route_visual_confirmation/visual_integrity_audit.json"),
    ("Phase3", "phase3_swa_true_route_visual_confirmation/visual_review.csv"),
    ("Phase4", "phase4_swa_carrier_ledger/swa_carrier_ledger.csv"),
    ("Phase4", "phase4_swa_carrier_ledger/swa_carrier_ledger_summary.json"),
    ("Phase5", "phase5_swa_rule_audit/swa_rule_rows.csv"),
    ("Phase5", "phase5_swa_rule_audit/bad_good_confusion_matrix.json"),
    ("Phase6", "phase6_swa_postroute_action/swa_postroute_action_summary.json"),
    ("Phase7", "phase7_swa_qconditioned_action/swa_qconditioned_action_summary.json"),
    ("Phase8", "phase8_merge_gauge_boundary_fallback/merge_gauge_fallback_summary.json"),
    ("Phase9", "phase9_ttt_after_swa_merge/ttt_after_swa_merge_summary.json"),
    ("Phase10", "phase10_long_window_evaluation/long_window_evaluation_summary.json"),
    ("Phase11", "phase11_heldout_official_validation/heldout_official_validation_summary.json"),
    ("Phase12", "phase12_rediscovery/visual_integrity_audit.json"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v81s-root", type=Path, default=DEFAULT_V81S_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), ensure_ascii=False, sort_keys=True)
                    if isinstance(row.get(key), (dict, list, tuple))
                    else row.get(key)
                    for key in fields
                }
            )


def source_status(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.append({"path": str(path), "exists": path.is_file(), "size_bytes": path.stat().st_size if path.is_file() else None})
    return rows


def seq_row(rows: Iterable[Mapping[str, Any]], seq: str) -> dict[str, Any]:
    for row in rows:
        if str(row.get("seq")) == seq:
            return dict(row)
    return {}


def summarize_overlap(root: Path) -> dict[str, Any]:
    default = read_json(root / "phaseS1_multiseq_swa_overlap_repair/phaseS1_overlap_pair_audit_summary.json")
    minconf0 = read_json(root / "phaseS1_multiseq_swa_overlap_repair_minconf0/phaseS1_overlap_pair_audit_summary.json")
    seq01_conf = read_json(root / "phaseS1_multiseq_swa_overlap_repair_minconf0/seq01_minconf0_geometry_confidence_audit.json")
    default_rows = default.get("rows") if isinstance(default.get("rows"), list) else []
    minconf_rows = minconf0.get("rows") if isinstance(minconf0.get("rows"), list) else []
    seq01_default = seq_row(default_rows, "01")
    seq01_minconf0 = seq_row(minconf_rows, "01")
    return {
        "default_source": default.get("root"),
        "minconf0_source": minconf0.get("root"),
        "default_gate": default.get("gate"),
        "minconf0_gate": minconf0.get("gate"),
        "seq01_default": {
            "median_saved_pairs_per_overlap": seq01_default.get("median_saved_pairs_per_overlap"),
            "semantic_label_projected_pair_ratio": seq01_default.get("semantic_label_projected_pair_ratio"),
            "median_raw_residual_rmse": seq01_default.get("median_raw_residual_rmse"),
            "swa_action_allowed": seq01_default.get("swa_action_allowed"),
        },
        "seq01_minconf0": {
            "median_saved_pairs_per_overlap": seq01_minconf0.get("median_saved_pairs_per_overlap"),
            "semantic_label_projected_pair_ratio": seq01_minconf0.get("semantic_label_projected_pair_ratio"),
            "median_raw_residual_rmse": seq01_minconf0.get("median_raw_residual_rmse"),
            "swa_action_allowed": seq01_minconf0.get("swa_action_allowed"),
            "either_zero_ratio": seq01_conf.get("either_zero_ratio"),
            "both_zero_ratio": seq01_conf.get("both_zero_ratio"),
            "min_conf_median": seq01_conf.get("min_conf_median"),
            "min_conf_p90": seq01_conf.get("min_conf_p90"),
        },
        "usable_highconf_seqs": (default.get("gate") or {}).get("swa_action_allowed_seqs") or [],
        "minconf0_allowed_seqs": (minconf0.get("gate") or {}).get("swa_action_allowed_seqs") or [],
    }


def build_required_markdown() -> str:
    lines = ["# v82 Required Next Artifacts", ""]
    current_phase = ""
    for phase, artifact in REQUIRED_NEXT_ARTIFACTS:
        if phase != current_phase:
            lines.extend(["", f"## {phase}"])
            current_phase = phase
        lines.append(f"- `{artifact}`")
    lines.append("")
    return "\n".join(lines)


def build_forbidden_markdown() -> str:
    lines = ["# v82 Forbidden Repeats", ""]
    for row in FORBIDDEN_REPEATS:
        lines.extend(
            [
                f"## {row['family_id']}",
                f"- family: {row['family']}",
                f"- prior_status: {row['prior_status']}",
                f"- v82_rule: {row['v82_rule']}",
                f"- allowed_v82_form: {row['allowed_v82_form']}",
                f"- reason: {row['reason']}",
                "",
            ]
        )
    lines.extend(
        [
            "Gate rule:",
            "",
            "If a forbidden family is launched without the listed new evidence or actuator, mark it `invalid_repeat_of_failed_family` and do not count it as v82 progress.",
            "",
        ]
    )
    return "\n".join(lines)


def build_lock(v81s_root: Path, out_dir: Path) -> dict[str, Any]:
    source_rows = source_status(SOURCE_DOCS)
    overlap = summarize_overlap(v81s_root)
    pair_bank = read_json(v81s_root / "phaseS2_swa_good_bad_pair_bank/swa_good_bad_pair_bank_summary.json")
    visual = read_json(v81s_root / "phaseS3_swa_visual_confirmation/visual_integrity_audit.json")
    action = read_json(v81s_root / "phaseS5_swa_action_route_audit/swa_action_route_audit_summary.json")

    all_source_docs_exist = all(bool(row["exists"]) for row in source_rows)
    forbidden_complete = bool(FORBIDDEN_REPEATS) and all(row.get("v82_rule") for row in FORBIDDEN_REPEATS)
    contract_fields = {"phase", "candidate_family", "allowed", "required_evidence_or_entry_gate", "must_beat_controls", "promotion_scope"}
    contract_complete = all(contract_fields.issubset(row) for row in CONTRACT_ROWS)
    required_artifacts_complete = bool(REQUIRED_NEXT_ARTIFACTS)
    v81s_artifacts_present = all(
        bool(item)
        for item in (
            overlap.get("default_gate"),
            overlap.get("minconf0_gate"),
            pair_bank.get("gate"),
            visual.get("gate"),
            action.get("gate"),
        )
    )

    gate_checks = {
        "source_docs_exist": all_source_docs_exist,
        "v81s_artifacts_present": v81s_artifacts_present,
        "forbidden_repeats_non_empty": bool(FORBIDDEN_REPEATS),
        "forbidden_repeats_mapped": forbidden_complete,
        "candidate_contract_has_required_fields": contract_complete,
        "required_next_artifacts_non_empty": required_artifacts_complete,
    }
    phase0_gate_pass = all(gate_checks.values())

    lock = {
        "schema": "acl2_v82_phase0_v81s_evidence_lock_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_v81s_root": str(v81s_root),
        "out_dir": str(out_dir),
        "phase0_gate_pass": phase0_gate_pass,
        "gate_checks": gate_checks,
        "source_docs": source_rows,
        "overlap_lock": overlap,
        "pair_bank_lock": {
            "rows": pair_bank.get("rows"),
            "case_counts": pair_bank.get("case_counts"),
            "seq_coverage": pair_bank.get("seq_coverage"),
            "artifact_quality_risk_rows": pair_bank.get("artifact_quality_risk_rows"),
            "missing_field_counts": pair_bank.get("missing_field_counts"),
            "gate": pair_bank.get("gate"),
        },
        "visual_lock": {
            "rows": visual.get("rows"),
            "case_counts": visual.get("case_counts"),
            "seq_coverage": visual.get("seq_coverage"),
            "visual_manifest_pair_count": visual.get("visual_manifest_pair_count"),
            "route_smoke_pair_count": visual.get("route_smoke_pair_count"),
            "route_smoke_pairs": visual.get("route_smoke_pairs"),
            "gate": visual.get("gate"),
            "note": visual.get("note"),
        },
        "action_route_lock": {
            "decision": action.get("decision"),
            "seq_coverage": action.get("seq_coverage"),
            "route_file_count": action.get("route_file_count"),
            "route_row_count": action.get("route_row_count"),
            "route_mask_row_count": action.get("route_mask_row_count"),
            "candidate_summaries": action.get("candidate_summaries"),
            "gate": action.get("gate"),
            "notes": action.get("notes"),
        },
        "forbidden_repeats": FORBIDDEN_REPEATS,
        "reusable_clues": REUSABLE_CLUES,
        "candidate_family_contract": CONTRACT_ROWS,
        "required_next_artifacts": [{"phase": phase, "artifact": artifact} for phase, artifact in REQUIRED_NEXT_ARTIFACTS],
        "phase0_decision": "pass_to_phase1" if phase0_gate_pass else "stop_until_phase0_sources_repaired",
        "v82_scope_caveats": [
            "seq01 default high-confidence overlap remains sparse and must be separately classified in Phase1.",
            "min_conf0 overlap can pass quantity gates but is low-confidence stress/risk evidence, not stable evidence.",
            "v81S S3 visual coverage is 24 rows but true runtime route smoke covers only 8 pairs; v82 Phase3 must repair this.",
            "P9_40/P9_6 route/action fidelity does not equal geometry success; old source_replace/source_gate families are forbidden repeats.",
            "TTT cannot run before SWA or merge/gauge confirmation.",
        ],
    }

    write_json(out_dir / "evidence_lock.json", lock)
    write_csv(out_dir / "source_file_status.csv", source_rows)
    write_csv(out_dir / "reusable_clues.csv", REUSABLE_CLUES)
    write_csv(out_dir / "v82_candidate_family_contract.csv", CONTRACT_ROWS)
    write_csv(out_dir / "forbidden_repeats.csv", FORBIDDEN_REPEATS)
    (out_dir / "forbidden_repeats.md").write_text(build_forbidden_markdown(), encoding="utf-8")
    (out_dir / "required_next_artifacts.md").write_text(build_required_markdown(), encoding="utf-8")
    return lock


def main() -> None:
    args = parse_args()
    lock = build_lock(args.v81s_root, args.out_dir)
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "phase0_gate_pass": lock["phase0_gate_pass"],
                "phase0_decision": lock["phase0_decision"],
                "seq01_default_median_saved_pairs_per_overlap": (lock["overlap_lock"].get("seq01_default") or {}).get(
                    "median_saved_pairs_per_overlap"
                ),
                "seq01_minconf0_either_zero_ratio": (lock["overlap_lock"].get("seq01_minconf0") or {}).get(
                    "either_zero_ratio"
                ),
                "visual_true_route_full_gate": ((lock["visual_lock"].get("gate") or {}).get("phaseS3_gate_pass")),
                "action_route_decision": (lock["action_route_lock"].get("decision")),
                "forbidden_repeat_count": len(lock["forbidden_repeats"]),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
