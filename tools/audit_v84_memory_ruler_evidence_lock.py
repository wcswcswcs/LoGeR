#!/usr/bin/env python3
"""Build ACL2 v84 Phase0 Memory Ruler evidence lock artifacts.

This script is intentionally read-only with respect to prior experiment
artifacts. It locks the v83 No-Go boundary, records the v84 hypotheses, checks
required artifact availability, and writes the Phase0 gate summary before any
new runtime action is allowed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_OUT_DIR = Path("results/acl2_v84tf_memory_ruler_audit/phase0_evidence_lock")
V83_ROOT = Path("results/acl2_v83tf_clue_sufficiency_vs_action_misuse")
V82_ROOT = Path("results/acl2_v82tf_swa_carrier_semantic_scale_handoff")

CORE_V83_INPUTS = {
    "v83_plan": Path("docs/ACL2_v83TF_ClueSufficiency_vs_ActionMisuse_ExperimentPlan.md"),
    "v83_execution_log": Path("docs/ACL2_v83TF_ClueSufficiency_vs_ActionMisuse_执行日志.md"),
    "v83_recap_log": Path("docs/ACL2_v83TF_ClueSufficiency_vs_ActionMisuse_实验结果复盘.md"),
    "v83_final_report": V83_ROOT / "phase20_final_report/final_report.json",
    "v83_final_decision": V83_ROOT / "phase10_decision_matrix/final_decision.json",
}

REQUIRED_ARTIFACTS = {
    "v83_final_report": V83_ROOT / "phase20_final_report/final_report.json",
    "v83_final_decision": V83_ROOT / "phase10_decision_matrix/final_decision.json",
    "v83_unified_clue_matrix": V83_ROOT / "phase1_unified_clue_matrix/unified_clue_matrix.csv",
    "v83_clue_sufficiency_summary": V83_ROOT / "phase2_clue_sufficiency/clue_sufficiency_summary.json",
    "v83_carrier_alignment_summary": V83_ROOT / "phase3_carrier_alignment/carrier_alignment_summary.json",
    "v83_counterfactual_summary": V83_ROOT / "phase4_counterfactual_upper_bound/counterfactual_upper_bound_summary.json",
    "v82_true_route_visual_manifest": V82_ROOT / "phase3_swa_true_route_visual_confirmation/visual_manifest.csv",
    "v82_true_route_visual_integrity": V82_ROOT / "phase3_swa_true_route_visual_confirmation/visual_integrity_audit.json",
    "v82_swa_carrier_ledger": V82_ROOT / "phase4_swa_carrier_ledger/swa_carrier_ledger.csv",
    "v82_swa_carrier_ledger_summary": V82_ROOT / "phase4_swa_carrier_ledger/swa_carrier_ledger_summary.json",
    "v82_route_gate_failure_decomp": V82_ROOT / "phase12_route_gate_failure_decomp/route_gate_failure_decomp_summary.json",
    "v82_pair_bank": V82_ROOT / "phase2_swa_pair_bank_v2/swa_pair_bank_v2.csv",
    "v82_overlap_quality": V82_ROOT / "phase1_overlap_quality_stratification/overlap_quality_by_pair.csv",
    "v80_long_window_cases": Path(
        "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
        "report_final/phase1_three_memory_case_bank/long_five_chunk_cases.csv"
    ),
    "v80_mid_adjacent_cases": Path(
        "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
        "report_final/phase1_three_memory_case_bank/mid_adjacent_pair_cases.csv"
    ),
    "v80_short_cases": Path(
        "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
        "report_final/phase1_three_memory_case_bank/short_single_chunk_cases.csv"
    ),
    "v80_direct_hook_audit": V83_ROOT / "phase1_direct_hook_repair_audit/direct_hook_repair_audit.csv",
}

OPTIONAL_GLOB_CHECKS = {
    "semantic_sparse_masklets": "results/kitti_preprocess/*/sparse_masklets_with_semantic.pt",
    "semantic_stage_c_chunks": "results/kitti_preprocess/*/stage_c_cache_semantic_chunks*",
    "v81s_overlap_pair_pts": (
        "results/kitti01_hmc_v2/acl2_v81s_swa_first_semantic_scale_memory_control/"
        "report_final/phaseS1_multiseq_swa_overlap_repair/overlap_pairs/*/chunk_*.pt"
    ),
    "v80_direct_read_dumps": (
        "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
        "report_final/phase2_direct_hook_repair/**/read_cue_patch_dumps/chunk_*_read_cue_patch.pt"
    ),
    "v80_direct_pca_features": (
        "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
        "report_final/phase2_direct_hook_repair/**/pca_features/chunk_*.pt"
    ),
}

FORBIDDEN_REPEATS = [
    ("semantic label ratio -> route mass threshold", "v83/v82 specificity gates failed; label mass is not memory-used ruler evidence."),
    ("samegroup/head15/contextual route rule search", "v82 contextual/per-head route families produced zero fully passing rule rows."),
    ("P9_40/P9_6 source replace/source gate alpha sweep", "old source action family is explicitly forbidden by v84."),
    ("merge overlap_outlier / robust_semoverlap tolerance sweep only", "v82/v83 merge-gauge upper-bound/interface families failed current gates."),
    ("TTT write-strength / freeze / old_decay scalar sweep", "TTT entry requires confirmed ruler carrier evidence."),
    ("selected-write veto from semantic token only", "v84 forbids TTT self-selected semantic tokens before confirmed ruler."),
    ("per-chunk Sim(3) runtime correction", "diagnostic oracle only; runtime explicit scale correction is forbidden."),
    ("chunk-id-specific policy", "violates training-free/no chunk-id policy."),
    ("train selector/classifier/calibrator", "violates training-free constraint."),
]

HYPOTHESES = [
    {
        "hypothesis": "H1",
        "name": "bad_handoff_memory_ruler_discontinuity",
        "test": "RPI/RCI/RCX/ROI separate bad/good, beat geometry-only and controls.",
        "pass_gate": "bad_recall>=0.60; good_FPR<=0.25; AUC>=0.75; coverage>=3; LOSO positive folds>=3.",
        "if_fail": "Phase11 visual rediscovery: candidate scarcity, leverage definition, route dump layer/head, merge/gauge carrier, label mismatch.",
    },
    {
        "hypothesis": "H2",
        "name": "memory_used_anchor_not_semantic_visible_anchor",
        "test": "R5/R8 memory-used ruler beats R1 semantic-visible and semantic shuffle.",
        "pass_gate": "semantic-visible fails or is weaker; memory-used passes; shuffle/random controls do not pass.",
        "if_fail": "Repair READ/SWA true route dump if visible semantic passes but memory-used fails; otherwise extend evidence, not thresholds.",
    },
    {
        "hypothesis": "H3",
        "name": "swa_carrier_relocalization",
        "test": "SWA Q-conditioned/K-risk/V-protect/head-specific families carry ruler evidence.",
        "pass_gate": "actual-vs-same-head-random>=0.05; semantic-shuffle margin>=0.05; bad_recall>=0.60; good_FPR<=0.25.",
        "if_fail": "Check merge/gauge carrier; do not start SWA runtime action.",
    },
    {
        "hypothesis": "H4",
        "name": "ruler_contradiction_not_only_absence",
        "test": "RCX and distance-ratio MAD reveal conflicting ruler clusters in bad pairs.",
        "pass_gate": "contradiction score passes bad/good gate and visual panels show conflicting anchor clusters.",
        "if_fail": "Keep contradiction diagnostic only; no action promotion.",
    },
    {
        "hypothesis": "H5",
        "name": "ttt_only_after_confirmed_ruler",
        "test": "TTT write overlaps confirmed READ+SWA or merge/gauge ruler carrier.",
        "pass_gate": "TTT entry only after SWA or merge/gauge pass; write mass changes and geometry improves versus controls.",
        "if_fail": "Stop TTT variants and return to carrier/merge-gauge interface.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize(row.get(key, "")) for key in fields})


def serialize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def artifact_row(name: str, path: Path, required: bool = True) -> dict[str, Any]:
    return {
        "name": name,
        "path": str(path),
        "required": required,
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else "",
        "sha256": sha256(path),
    }


def build_artifact_rows() -> list[dict[str, Any]]:
    rows = [artifact_row(name, path, True) for name, path in REQUIRED_ARTIFACTS.items()]
    for name, pattern in OPTIONAL_GLOB_CHECKS.items():
        matches = sorted(Path(".").glob(pattern))
        rows.append(
            {
                "name": name,
                "path": pattern,
                "required": False,
                "exists": bool(matches),
                "is_file": "",
                "size_bytes": "",
                "sha256": "",
                "match_count": len(matches),
                "sample_matches": [str(path) for path in matches[:8]],
            }
        )
    return rows


def build_v83_boundary(final_report: dict[str, Any], final_decision: dict[str, Any]) -> str:
    labels = final_report.get("primary_decision_labels") or final_decision.get("primary_decision_labels") or []
    active_labels = final_decision.get("active_decision_labels") or []
    questions = {item.get("key"): item for item in final_report.get("questions", []) if isinstance(item, dict)}
    return "\n".join(
        [
            "# v83 No-Go Boundary Locked for v84",
            "",
            f"- final_status: `{final_report.get('final_status')}`",
            f"- conclusion: {final_report.get('conclusion')}",
            f"- primary_decision_labels: `{labels}`",
            f"- active_decision_labels: `{active_labels}`",
            "",
            "## Required Facts",
            "",
            "1. v83 final_status is `No-Go_before_runtime_action`.",
            f"2. Phase2 geometry-only clue sufficiency: phase2_gate_pass={final_report.get('phase2_gate_pass')}; "
            f"geometry question evidence={questions.get('1_geometry_only_separated_bad_good', {}).get('evidence', {})}.",
            "3. Current semantic/RADIO definitions did not add positive lift over geometry-only.",
            f"4. Phase3 carrier localization failed: phase3_gate_pass={final_report.get('phase3_gate_pass')}.",
            "5. SWA blocker is semantic-shuffle / specificity and carrier-control failure, not missing visual route alone.",
            "6. Merge/gauge separation was stronger in diagnostics but the current interface / upper bound failed.",
            f"7. Phase4 counterfactual upper bound failed: phase4_gate_pass={final_report.get('phase4_gate_pass')}.",
            "8. TTT is not ready and must not run before confirmed memory-ruler carrier evidence.",
            "9. v83 visual rediscovery/audit was a compliant No-Go support artifact, not method success.",
            "",
            "## v84 Consequence",
            "",
            "v84 starts from a new question: whether memory-used semantic-geometric ruler evidence persists across "
            "chunk handoff and can beat geometry-only, random, and shuffle controls. Runtime action remains blocked "
            "until v84 gates allow it.",
            "",
        ]
    )


def build_required_artifacts_md(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# v84 Required Artifact Availability",
        "",
        "This file records availability only. Missing values must not be fabricated in later phases.",
        "",
        "| name | required | exists | detail |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        detail = row.get("path", "")
        if "match_count" in row:
            detail = f"{detail}; match_count={row['match_count']}"
        lines.append(f"| `{row['name']}` | {row['required']} | {row['exists']} | `{detail}` |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    final_report = read_json(CORE_V83_INPUTS["v83_final_report"])
    final_decision = read_json(CORE_V83_INPUTS["v83_final_decision"])
    artifact_rows = build_artifact_rows()
    required_missing = [row for row in artifact_rows if row.get("required") is True and not row.get("exists")]

    core_inputs = [artifact_row(name, path, True) for name, path in CORE_V83_INPUTS.items()]
    v83_boundary_ok = (
        final_report.get("final_status") == "No-Go_before_runtime_action"
        and final_report.get("phase2_gate_pass") is True
        and final_report.get("phase3_gate_pass") is False
        and final_report.get("phase4_gate_pass") is False
    )

    write_json(
        out_dir / "evidence_lock.json",
        {
            "schema": "acl2_v84_phase0_evidence_lock_v1",
            "core_inputs": core_inputs,
            "v83_final_status": final_report.get("final_status"),
            "v83_conclusion": final_report.get("conclusion"),
            "v83_primary_decision_labels": final_report.get("primary_decision_labels", []),
            "v83_active_decision_labels": final_decision.get("active_decision_labels", []),
            "v83_phase2_gate_pass": final_report.get("phase2_gate_pass"),
            "v83_phase3_gate_pass": final_report.get("phase3_gate_pass"),
            "v83_phase4_gate_pass": final_report.get("phase4_gate_pass"),
            "v83_boundary_ok": v83_boundary_ok,
            "runtime_action_allowed_after_phase0": False,
            "notes": [
                "Phase0 only locks prior evidence and forbids old families.",
                "Missing optional artifacts are reported, not backfilled.",
            ],
        },
    )

    (out_dir / "v83_no_go_boundary.md").write_text(
        build_v83_boundary(final_report, final_decision), encoding="utf-8"
    )
    (out_dir / "forbidden_repeats.md").write_text(
        "# Forbidden Repeats for v84\n\n"
        + "\n".join(f"- `{name}`: {reason}" for name, reason in FORBIDDEN_REPEATS)
        + "\n",
        encoding="utf-8",
    )
    write_csv(
        out_dir / "forbidden_repeats.csv",
        [{"family": name, "reason": reason} for name, reason in FORBIDDEN_REPEATS],
    )
    write_csv(out_dir / "memory_ruler_hypothesis_matrix.csv", HYPOTHESES)
    write_csv(out_dir / "required_artifacts.csv", artifact_rows)
    (out_dir / "required_artifacts.md").write_text(build_required_artifacts_md(artifact_rows), encoding="utf-8")

    phase0_gate_pass = (
        (out_dir / "evidence_lock.json").is_file()
        and (out_dir / "forbidden_repeats.md").is_file()
        and len(FORBIDDEN_REPEATS) > 0
        and {row["hypothesis"] for row in HYPOTHESES} == {"H1", "H2", "H3", "H4", "H5"}
        and len(artifact_rows) > 0
        and v83_boundary_ok
        and not required_missing
    )
    write_json(
        out_dir / "phase0_gate_summary.json",
        {
            "schema": "acl2_v84_phase0_gate_summary_v1",
            "phase": "phase0_evidence_lock",
            "phase0_gate_pass": phase0_gate_pass,
            "v83_boundary_ok": v83_boundary_ok,
            "forbidden_repeats_count": len(FORBIDDEN_REPEATS),
            "hypotheses": [row["hypothesis"] for row in HYPOTHESES],
            "required_artifacts_checked": len([row for row in artifact_rows if row.get("required") is True]),
            "required_missing_count": len(required_missing),
            "required_missing": required_missing,
            "runtime_action_allowed": False,
            "next_phase_allowed": bool(phase0_gate_pass),
        },
    )

    print(json.dumps({"out_dir": str(out_dir), "phase0_gate_pass": phase0_gate_pass}, ensure_ascii=False))


if __name__ == "__main__":
    main()

