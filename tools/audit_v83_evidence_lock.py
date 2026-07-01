#!/usr/bin/env python3
"""Build ACL2 v83 Phase0 evidence lock artifacts.

This tool is intentionally read-only with respect to prior experiments. It
collects v82 boundary evidence from existing docs/results and writes the v83
Phase0 lock files required before any new runtime action is allowed.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path("results/acl2_v83tf_clue_sufficiency_vs_action_misuse/phase0_evidence_lock")

INPUT_PATTERNS = [
    "docs/ACL2_v82TF_SWA_CarrierDiscovery_SemanticScaleHandoff_实验结果复盘.md",
    "docs/ACL2_v82TF_SWA_CarrierDiscovery_SemanticScaleHandoff_执行日志.md",
    "docs/ACL2_v81S_Updated_SWAFirst_SemanticScaleMemoryControl_实验结果复盘.md",
    "docs/ACL2_v80TF_MultiSeq_GoodBadCase_SemanticThreeMemoryControl_实验结果复盘.md",
    "docs/ACL2_v78TF_v3_PCA_Grounded_AuditableVisualRediscovery_实验结果复盘.md",
    "docs/ACL2_v68_Integrated_CueConstruction_PCA_LayerSelection_*.md",
    "docs/ACL2_v67_DenseSemantic_Reconstruction_Emergency_*.md",
    "docs/ACL2_v65_C9_H35_TransitionSwap_MergeGauge_Attribution_*.md",
]

V82_ROOT = Path("results/acl2_v82tf_swa_carrier_semantic_scale_handoff")
ARTIFACTS = {
    "phase3_visual_integrity": V82_ROOT / "phase3_swa_true_route_visual_confirmation" / "visual_integrity_audit.json",
    "phase4_carrier_ledger": V82_ROOT / "phase4_swa_carrier_ledger" / "swa_carrier_ledger_summary.json",
    "phase8e_merge_gauge": V82_ROOT / "phase8e_projection_tol001_steps64_continuation" / "merge_gauge_fallback_summary.json",
    "phase12_visual_integrity": V82_ROOT / "phase12_rediscovery" / "visual_integrity_audit.json",
    "phase12_merge_oracle": V82_ROOT / "phase12_merge_controller_oracle_audit" / "merge_controller_oracle_summary.json",
    "phase12_per_head": V82_ROOT / "phase12_per_head_route_localization" / "per_head_route_localization_summary.json",
    "phase12_strict_rule": V82_ROOT / "phase12_route_control_rule_refinement" / "route_control_rule_refinement_summary.json",
    "phase12_contextual_rule": V82_ROOT / "phase12_contextual_route_rule_search" / "contextual_route_rule_search_summary.json",
    "phase12_gate_decomp": V82_ROOT / "phase12_route_gate_failure_decomp" / "route_gate_failure_decomp_summary.json",
}

FORBIDDEN_REPEATS = [
    ("P9_40 / P9_6 / source_replace / source_gate alpha sweep", "v82/v81S evidence says route/action fidelity is not geometry success"),
    ("source-column top-k boost / damp", "same source-column family as invalid source action repeats"),
    ("head6/head15 route mass threshold sweep", "v82 Phase12 showed route-mass/head-specific rules do not pass controls"),
    ("samegroup semantic route threshold sweep", "v82 contextual rules over samegroup metadata had zero passing rules"),
    ("L13 negative damp scalar", "old scalar damp family; no new carrier evidence"),
    ("SWA semantic role reweight same family", "label-ratio reweighting without carrier alignment is not valid"),
    ("selected-write low-support direct veto", "TTT cannot self-select from label-only evidence"),
    ("TTT global write-strength / freeze / old_decay", "TTT entry requires SWA/merge confirmed evidence"),
    ("qscale / RADIO scalar guard", "scalar guard is not carrier/action evidence"),
    ("Phase8 overlap_outlier / robust_semoverlap tolerance/steps small sweep", "v82 Phase8d/e already tested this family and failed gates"),
]

HYPOTHESES = [
    ("H1", "geometry_clues_sufficient_semantic_no_add", "Test geometry-only vs geometry+semantic in Phase2 clue sufficiency; require semantic lift over geometry-only."),
    ("H2", "semantic_clues_sufficient_action_wrong", "If combined clues beat geometry-only and shuffle, audit carrier alignment before runtime action."),
    ("H3", "clues_and_carrier_sufficient_action_surface_weak", "If carrier alignment passes but runtime action has <1% geometry gain, test counterfactual/action surface."),
    ("H4", "swa_not_scale_gauge_carrier", "If SWA carrier/action moves route but not geometry, test merge/gauge carrier and counterfactual."),
    ("H5", "ttt_requires_confirmed_evidence", "Run TTT only after SWA or merge/gauge passes and provides confirmed stable evidence."),
]

FAILURE_MATRIX = [
    (
        "v82_route_mass_samegroup_head15_contextual_no_go",
        "9205 contextual rules and route gate decomp had zero passing rules",
        "Phase1/2 unified clue matrix; do not continue same metadata thresholding",
        "H1,H2",
    ),
    (
        "v82_phase4_carrier_not_localized",
        "carrier ledger had limitations: layer aggregate route, no same-head random/shuffle, READ masks not token-aligned",
        "Phase3 carrier alignment with direct READ/SWA dumps and visual panels",
        "H2,H3,H4",
    ),
    (
        "v82_phase8_merge_gauge_family_failed",
        "phase8e merge/gauge summary gate false and oracle found no passing runtime controller",
        "Phase4 counterfactual or Phase6 only if carrier/upper-bound evidence appears",
        "H3,H4",
    ),
    (
        "v82_ttt_not_started",
        "TTT had no SWA/merge confirmed evidence and only not-run availability panel",
        "TTT after confirmed evidence only",
        "H5",
    ),
    (
        "semantic_shuffle_specificity_failed",
        "semantic samegroup all-head/head15 had semantic-shuffle controls but zero shuffle pass rows",
        "Test semantic specificity against geometry-only and expanded clue groups; try RADIO/thingstuff/QK, not label-ratio threshold",
        "H1,H2",
    ),
]


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_inputs() -> list[Path]:
    out: list[Path] = []
    for pattern in INPUT_PATTERNS:
        matches = sorted(Path(p) for p in glob.glob(pattern))
        if not matches and not any(ch in pattern for ch in "*?[]"):
            matches = [Path(pattern)]
        out.extend(matches)
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in out:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _artifact_digest() -> dict[str, Any]:
    digests: dict[str, Any] = {}
    for name, path in ARTIFACTS.items():
        data = _read_json(path)
        digests[name] = {
            "path": str(path),
            "exists": path.exists(),
            "sha256": _sha256(path),
            "schema": data.get("schema", ""),
        }
    return digests


def _v82_fact_lock() -> dict[str, Any]:
    v82_recap = Path("docs/ACL2_v82TF_SWA_CarrierDiscovery_SemanticScaleHandoff_实验结果复盘.md")
    v82_exec = Path("docs/ACL2_v82TF_SWA_CarrierDiscovery_SemanticScaleHandoff_执行日志.md")
    recap_text = _read_text(v82_recap)
    exec_text = _read_text(v82_exec)
    phase3 = _read_json(ARTIFACTS["phase3_visual_integrity"])
    phase4 = _read_json(ARTIFACTS["phase4_carrier_ledger"])
    phase8e = _read_json(ARTIFACTS["phase8e_merge_gauge"])
    phase12_vis = _read_json(ARTIFACTS["phase12_visual_integrity"])
    merge_oracle = _read_json(ARTIFACTS["phase12_merge_oracle"])
    per_head = _read_json(ARTIFACTS["phase12_per_head"])
    strict = _read_json(ARTIFACTS["phase12_strict_rule"])
    contextual = _read_json(ARTIFACTS["phase12_contextual_rule"])
    decomp = _read_json(ARTIFACTS["phase12_gate_decomp"])

    facts = {
        "v82_no_method_success": {
            "value": (
                "目标未达成" in recap_text
                and "No-Go" in recap_text
                and per_head.get("method_promotion_gate_pass") is False
                and phase8e.get("phase8_gate_pass") is False
            ),
            "evidence": [
                str(v82_recap),
                str(ARTIFACTS["phase12_per_head"]),
                str(ARTIFACTS["phase8e_merge_gauge"]),
            ],
            "details": {
                "method_promotion_gate_pass": per_head.get("method_promotion_gate_pass"),
                "phase8e_gate_pass": phase8e.get("phase8_gate_pass"),
                "phase8e_decision": phase8e.get("decision"),
            },
        },
        "phase3_true_route_qkv_visual_complete": {
            "value": bool(phase3.get("gate", {}).get("phase3_gate_pass")),
            "evidence": [str(ARTIFACTS["phase3_visual_integrity"])],
            "details": {
                "rows": phase3.get("rows"),
                "true_route_rows": phase3.get("true_route_rows"),
                "qkv_rows": phase3.get("qkv_rows"),
                "seq_coverage": phase3.get("seq_coverage"),
                "route_dump_counts": phase3.get("route_dump_counts"),
            },
        },
        "head15_join_bug_fixed_but_rules_still_no_go": {
            "value": (
                "head15 join bug" in recap_text
                and "new_missing_head15_rows=0" in recap_text
                and strict.get("phase5_rule_gate_pass") is False
                and contextual.get("phase5_contextual_rule_gate_pass") is False
            ),
            "evidence": [
                str(v82_recap),
                str(ARTIFACTS["phase12_strict_rule"]),
                str(ARTIFACTS["phase12_contextual_rule"]),
            ],
            "details": {
                "strict_passing_rule_count": strict.get("passing_rule_count"),
                "contextual_passing_rule_count": contextual.get("passing_rule_count"),
                "contextual_base_case_missing_rows": contextual.get("base_case_missing_rows"),
            },
        },
        "contextual_9205_rules_no_passing_rule": {
            "value": contextual.get("rule_rows") == 9205 and contextual.get("passing_rule_count") == 0,
            "evidence": [str(ARTIFACTS["phase12_contextual_rule"])],
            "details": {
                "rule_rows": contextual.get("rule_rows"),
                "passing_rule_count": contextual.get("passing_rule_count"),
                "decision": contextual.get("decision"),
            },
        },
        "samegroup_route_mass_head15_confidence_context_rules_insufficient": {
            "value": decomp.get("fully_passing_rule_rows") == 0 and decomp.get("same_mass_near_without_full_gate_rows") == 67,
            "evidence": [str(ARTIFACTS["phase12_gate_decomp"])],
            "details": {
                "fully_passing_rule_rows": decomp.get("fully_passing_rule_rows"),
                "same_mass_near_without_full_gate_rows": decomp.get("same_mass_near_without_full_gate_rows"),
                "near_with_semantic_shuffle_available": decomp.get("near_with_semantic_shuffle_available"),
                "near_without_semantic_shuffle_available": decomp.get("near_without_semantic_shuffle_available"),
            },
        },
        "ttt_not_qualified": {
            "value": (
                "TTT" in exec_text
                and "不能启动" in recap_text
                and phase12_vis.get("ttt_runtime_visual_evidence_available") is False
            ),
            "evidence": [str(v82_recap), str(v82_exec), str(ARTIFACTS["phase12_visual_integrity"])],
            "details": {
                "ttt_runtime_visual_evidence_available": phase12_vis.get("ttt_runtime_visual_evidence_available"),
                "ttt_availability_note": phase12_vis.get("ttt_availability_note"),
            },
        },
        "phase12_rediscovery_visual_audit_pass": {
            "value": phase12_vis.get("visual_audit_gate_pass") is True,
            "evidence": [str(ARTIFACTS["phase12_visual_integrity"])],
            "details": {
                "visual_audit_gate_pass": phase12_vis.get("visual_audit_gate_pass"),
                "visual_review_rows": phase12_vis.get("visual_review_rows"),
                "group_counts": phase12_vis.get("group_counts"),
            },
        },
        "phase4_no_promoted_carrier": {
            "value": bool(phase4.get("limitations")) and phase4.get("visual_rows") == 24,
            "evidence": [str(ARTIFACTS["phase4_carrier_ledger"])],
            "details": {
                "limitations": phase4.get("limitations"),
                "visual_rows": phase4.get("visual_rows"),
                "families": sorted((phase4.get("families") or {}).keys()),
            },
        },
        "phase12_merge_controller_no_runtime_controller": {
            "value": merge_oracle.get("passing_runtime_controllers") == [],
            "evidence": [str(ARTIFACTS["phase12_merge_oracle"])],
            "details": {
                "controller_count": merge_oracle.get("controller_count"),
                "passing_runtime_controllers": merge_oracle.get("passing_runtime_controllers"),
                "diagnostic_gt_oracle_note": merge_oracle.get("diagnostic_gt_oracle_note"),
            },
        },
    }
    return facts


def _write_boundary(out_dir: Path, facts: dict[str, Any]) -> None:
    lines = [
        "# v83 Phase0 v82 No-Go Boundary",
        "",
        "This file records the v82 boundary facts required by the v83 plan. It does not claim method success.",
        "",
    ]
    items = [
        ("v82 没有 method success", "v82_no_method_success"),
        ("Phase3 true route / QKV visual evidence 已补齐", "phase3_true_route_qkv_visual_complete"),
        ("head15 join bug 已修，但 strict/contextual rule search 仍 No-Go", "head15_join_bug_fixed_but_rules_still_no_go"),
        ("9205 contextual rules 无 passing rule", "contextual_9205_rules_no_passing_rule"),
        ("当前 samegroup / route-mass / head15 / confidence-context route rules 不足", "samegroup_route_mass_head15_confidence_context_rules_insufficient"),
        ("TTT 未获启动资格", "ttt_not_qualified"),
        ("Phase12 rediscovery visual audit pass，所以 No-Go 是合规 No-Go", "phase12_rediscovery_visual_audit_pass"),
    ]
    for idx, (title, key) in enumerate(items, 1):
        fact = facts[key]
        lines.append(f"## {idx}. {title}")
        lines.append("")
        lines.append(f"value: `{fact['value']}`")
        lines.append("")
        lines.append("evidence:")
        for ev in fact["evidence"]:
            lines.append(f"- `{ev}`")
        lines.append("")
        lines.append("details:")
        lines.append("```json")
        lines.append(json.dumps(fact["details"], ensure_ascii=False, indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")
    (out_dir / "v82_no_go_boundary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_forbidden(out_dir: Path) -> None:
    lines = ["# v83 Forbidden Repeats", ""]
    for name, reason in FORBIDDEN_REPEATS:
        lines.append(f"- `{name}`: {reason}")
    lines.append("")
    (out_dir / "forbidden_repeats.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = _resolve_inputs()
    input_rows = [
        {
            "path": str(path),
            "exists": path.exists(),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size if path.exists() else None,
        }
        for path in inputs
    ]
    facts = _v82_fact_lock()
    artifact_digest = _artifact_digest()

    hypothesis_rows = [
        {
            "hypothesis_id": hid,
            "hypothesis": name,
            "next_valid_test": test,
            "status": "live",
        }
        for hid, name, test in HYPOTHESES
    ]
    matrix_rows = [
        {
            "old_failure": old,
            "locked_evidence": evidence,
            "next_valid_test": test,
            "hypotheses": hypotheses,
        }
        for old, evidence, test, hypotheses in FAILURE_MATRIX
    ]
    forbidden_rows = [{"forbidden_repeat": item, "reason": reason} for item, reason in FORBIDDEN_REPEATS]

    _write_json(
        out_dir / "evidence_lock.json",
        {
            "schema": "acl2_v83_phase0_evidence_lock_v1",
            "input_documents": input_rows,
            "artifact_digest": artifact_digest,
            "v82_boundary_facts": facts,
            "forbidden_repeat_count": len(FORBIDDEN_REPEATS),
            "hypotheses": hypothesis_rows,
            "failure_matrix": matrix_rows,
            "note": "Phase0 is a lock/audit stage only; no runtime action is launched.",
        },
    )
    _write_boundary(out_dir, facts)
    _write_forbidden(out_dir)
    _write_csv(out_dir / "still_live_hypotheses.csv", hypothesis_rows)
    _write_csv(out_dir / "clue_vs_action_hypothesis_matrix.csv", matrix_rows)
    _write_csv(out_dir / "forbidden_repeats.csv", forbidden_rows)

    gate = {
        "evidence_lock_exists": (out_dir / "evidence_lock.json").exists(),
        "forbidden_repeats_non_empty": len(FORBIDDEN_REPEATS) > 0,
        "still_live_hypotheses_includes_H1_H5": {row["hypothesis_id"] for row in hypothesis_rows} == {"H1", "H2", "H3", "H4", "H5"},
        "hypothesis_matrix_maps_old_failures": len(matrix_rows) >= 5 and all(row["next_valid_test"] for row in matrix_rows),
        "required_v82_boundary_facts_locked": all(fact["value"] for fact in facts.values()),
        "input_documents_all_present": all(row["exists"] for row in input_rows),
        "artifact_json_all_present": all(item["exists"] for item in artifact_digest.values()),
    }
    gate["phase0_gate_pass"] = bool(
        gate["evidence_lock_exists"]
        and gate["forbidden_repeats_non_empty"]
        and gate["still_live_hypotheses_includes_H1_H5"]
        and gate["hypothesis_matrix_maps_old_failures"]
        and gate["required_v82_boundary_facts_locked"]
        and gate["input_documents_all_present"]
        and gate["artifact_json_all_present"]
    )
    summary = {
        "schema": "acl2_v83_phase0_gate_summary_v1",
        "out_dir": str(out_dir),
        "gate": gate,
        "phase0_gate_pass": gate["phase0_gate_pass"],
        "input_document_count": len(input_rows),
        "artifact_count": len(artifact_digest),
        "forbidden_repeat_count": len(FORBIDDEN_REPEATS),
        "hypothesis_count": len(hypothesis_rows),
        "matrix_row_count": len(matrix_rows),
        "blocked_next_action_if_false": "Fix Phase0 lock before any new action.",
    }
    _write_json(out_dir / "phase0_gate_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
