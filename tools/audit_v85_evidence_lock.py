#!/usr/bin/env python3
"""Build ACL2 v85 Phase0 evidence-lock artifacts.

This stage is read-only with respect to prior experiments. It locks the v83
and v84 No-Go boundaries before v85 starts pairwise QK latent-anchor work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_OUT_DIR = Path("results/acl2_v85tf_latent_anchor_alignment_pairwise_memory_ruler/phase0_evidence_lock")

INPUTS: dict[str, Path] = {
    "v85_plan": Path("docs/ACL2_v85TF_LatentAnchorAlignment_PairwiseMemoryRuler_ExperimentPlan.md"),
    "v85_config": Path("configs/v85tf_latent_anchor_alignment.yaml"),
    "v84_plan": Path("docs/ACL2_v84TF_MemoryRulerAudit_ExperimentPlan.md"),
    "v84_execution_log": Path("docs/ACL2_v84TF_MemoryRulerAudit_执行日志.md"),
    "v84_recap_log": Path("docs/ACL2_v84TF_MemoryRulerAudit_实验结果复盘.md"),
    "v83_execution_log": Path("docs/ACL2_v83TF_ClueSufficiency_vs_ActionMisuse_执行日志.md"),
    "v83_recap_log": Path("docs/ACL2_v83TF_ClueSufficiency_vs_ActionMisuse_实验结果复盘.md"),
    "v83_final_decision": Path(
        "results/acl2_v83tf_clue_sufficiency_vs_action_misuse/phase10_decision_matrix/final_decision.json"
    ),
    "v84_phase12_final_decision": Path(
        "results/acl2_v84tf_memory_ruler_audit/phase12_decision_matrix/final_decision.json"
    ),
    "v84_phase16_continuation_decision": Path(
        "results/acl2_v84tf_memory_ruler_audit/phase16_continuation_decision/continuation_decision.json"
    ),
    "v84_phase15_anchor_positions": Path(
        "results/acl2_v84tf_memory_ruler_audit/phase15_anchor_route_mask_materialization/anchor_route_mask_positions.csv"
    ),
    "v84_phase15_summary": Path(
        "results/acl2_v84tf_memory_ruler_audit/phase15_anchor_route_mask_materialization/"
        "anchor_route_mask_materialization_summary.json"
    ),
    "v84_phase22_external_variant_summary": Path(
        "results/acl2_v84tf_memory_ruler_audit/phase22_v84_external_variant_summary/variant_summary.json"
    ),
    "v84_phase33_merge_gauge_coverage": Path(
        "results/acl2_v84tf_memory_ruler_audit/phase33_v84_merge_gauge_coverage_decision/"
        "merge_gauge_coverage_summary.json"
    ),
}

FORBIDDEN_REPEATS = [
    (
        "v84_source_side_external_anchor_mask_beta_sweep",
        "v84 Phase22 found actual selected/source lift below random same-mass controls for all tested variants.",
    ),
    (
        "current_role_anchor_or_weak_medium_threshold_sweep",
        "v84 Phase14/22 showed broader anchors inflate false positives or remain nonspecific.",
    ),
    (
        "head13_or_head15_source_mask_retest_without_pairwise_qk",
        "v84 head13 external-anchor route also failed same-mass specificity.",
    ),
    (
        "source_column_boost_or_damp",
        "v85 must test current-query to historical-key pair relation, not source-column mass only.",
    ),
    (
        "old_value_protect_or_source_replace_family",
        "v85 first tests QK retrieval; V is diagnostic until QK route is confirmed.",
    ),
    (
        "l13_negative_damp_or_l07_l13_scalar_action",
        "old scalar route families are outside v85 pairwise latent-anchor hypothesis.",
    ),
    (
        "merge_gauge_overlap_outlier_tolerance_small_sweep",
        "v84 Phase33 merge/gauge coverage did not pass the improvement plus beat-controls gate.",
    ),
    (
        "ttt_write_strength_or_selected_write_veto",
        "TTT is ineligible before confirmed SWA or merge/gauge aligned evidence.",
    ),
    (
        "runtime_action_before_anchor_alignment_scale_route_gates",
        "v85 requires Phase1/3/4/5/6 gates before any conservative QK pair-bias action.",
    ),
]

HYPOTHESES = [
    {
        "hypothesis": "H1",
        "v84_failure": "source-side anchor identity was insufficient",
        "v85_test": "Build current-query / historical-key anchor pairs and test identity QK residual by bad/good and scale jump.",
        "required_phase": "Phase1/Phase2/Phase4",
        "pass_condition": "Reliable anchor pairs exist and E_identity relates to bad/high-scale-jump rows.",
        "if_fail": "Classify low_anchor_support or no_bad_good_identity_separation; do not fit runtime action.",
    },
    {
        "hypothesis": "H2",
        "v84_failure": "actual source-side route lift did not beat random same-mass masks",
        "v85_test": "Fit low-free C on anchor pairs and require held-out G_align beyond random/shuffle controls.",
        "required_phase": "Phase3",
        "pass_condition": "G_align_test >= P95(random controls)+0.05 and semantic shuffle margin >=0.05.",
        "if_fail": "Reduce model freedom, feature dim, or tighten anchors; stop if held-out margin remains absent.",
    },
    {
        "hypothesis": "H3",
        "v84_failure": "merge/gauge signal was weak and not a confirmed route carrier",
        "v85_test": "Audit whether pairwise latent alignment explains offline adjacent log-scale jump.",
        "required_phase": "Phase4",
        "pass_condition": "E_identity or G_align is scale-jump relevant with bad recall >=0.60 and good FPR <=0.25.",
        "if_fail": "Treat C as appearance/basis alignment; no SWA action.",
    },
    {
        "hypothesis": "H4",
        "v84_failure": "source-side mass was not pairwise QK-specific",
        "v85_test": "Use true pairwise SWA QK route dump and aligned pair score controls.",
        "required_phase": "Phase5",
        "pass_condition": "Actual pair route lift beats random and semantic/distance shuffles by >=0.05.",
        "if_fail": "Implement/check pair-index dump if missing; stop SWA route if actual remains <= controls.",
    },
    {
        "hypothesis": "H5",
        "v84_failure": "runtime action and TTT were blocked",
        "v85_test": "Only after anchor/alignment/scale/route/visual gates, run conservative QK pair-bias action.",
        "required_phase": "Phase7+",
        "pass_condition": "J_SWA/component geometry improves versus geometry/random/shuffle controls and good cases are protected.",
        "if_fail": "If route fidelity passes but geometry fails, go to merge/gauge aligned-pair weighting; do not beta sweep.",
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


def read_json(path: Path) -> Any:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def serialize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


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
            writer.writerow({field: serialize(row.get(field, "")) for field in fields})


def required_input_rows() -> list[dict[str, Any]]:
    rows = []
    for name, path in INPUTS.items():
        rows.append(
            {
                "name": name,
                "path": str(path),
                "exists": path.exists(),
                "is_file": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else "",
                "sha256": sha256(path),
            }
        )
    return rows


def all_required_inputs_present(rows: list[dict[str, Any]]) -> bool:
    return all(bool(row["exists"]) and bool(row["is_file"]) for row in rows)


def build_v84_boundary(
    v83_final: dict[str, Any],
    v84_phase12: dict[str, Any],
    v84_phase16: dict[str, Any],
    v84_phase22: list[dict[str, Any]],
    v84_phase33: dict[str, Any],
) -> str:
    margins = [
        (
            row.get("variant"),
            row.get("selected_lift_margin"),
            row.get("source_lift_margin"),
            row.get("score_sum"),
        )
        for row in v84_phase22
        if isinstance(row, dict)
    ]
    margin_lines = "\n".join(
        f"- {variant}: selected_margin={selected}, source_margin={source}, score_sum={score}"
        for variant, selected, source, score in margins
    )
    return "\n".join(
        [
            "# v85 Phase0 No-Go Boundary Lock",
            "",
            "## v83 Boundary",
            "",
            f"- active_decision_labels: `{v83_final.get('active_decision_labels', [])}`",
            f"- conclusion: {v83_final.get('conclusion', '')}",
            "- required fact: geometry clues were usable, but semantic definitions were not specific and SWA was not confirmed as scale/gauge carrier.",
            "",
            "## v84 Boundary",
            "",
            f"- phase12 final_status: `{v84_phase12.get('final_status')}`",
            f"- phase12 decision_labels: `{v84_phase12.get('decision_labels', [])}`",
            f"- phase16 final_status_after_continuation: `{v84_phase16.get('final_status_after_continuation')}`",
            f"- phase16 runtime_action_allowed: `{v84_phase16.get('runtime_action_allowed')}`",
            f"- phase16 ttt_allowed: `{v84_phase16.get('ttt_allowed')}`",
            "",
            "### Phase22 External Source-Mask Route Margins",
            "",
            margin_lines or "- missing phase22 variant rows",
            "",
            "### Phase33 Merge/Gauge Coverage",
            "",
            f"- merge_gauge_carrier_pass: `{v84_phase33.get('merge_gauge_carrier_pass')}`",
            f"- head_tail_pass_rows: `{v84_phase33.get('head_tail_pass_rows')}`",
            f"- overlap_future_pass_rows: `{v84_phase33.get('overlap_future_pass_rows')}`",
            f"- reason: {v84_phase33.get('reason')}",
            "",
            "## v85 Consequence",
            "",
            "The old source-side anchor-mask and merge/gauge tolerance routes are locked as No-Go/blocked. "
            "v85 may proceed only with pairwise current-Q to historical/cache-K latent alignment tests, "
            "with GT scale used only as offline audit label.",
            "",
        ]
    )


def build_forbidden_repeats_md(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# v85 Forbidden Repeats",
        "",
        "These paths are prohibited before pairwise QK anchor/alignment/scale/route gates pass.",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row['forbidden_repeat']}`: {row['reason']}")
    lines.append("")
    return "\n".join(lines)


def phase0_gate(v84_phase22: list[dict[str, Any]], v84_phase33: dict[str, Any], inputs_ok: bool) -> dict[str, Any]:
    phase22_rows = [row for row in v84_phase22 if isinstance(row, dict)]
    phase22_all_actual_lte_control = bool(phase22_rows) and all(
        float(row.get("selected_lift_margin", 1.0)) <= 0.0 and float(row.get("source_lift_margin", 1.0)) <= 0.0
        for row in phase22_rows
    )
    checks = {
        "required_inputs_present": inputs_ok,
        "v84_external_source_mask_no_go_locked": phase22_all_actual_lte_control,
        "v84_merge_gauge_no_go_locked": v84_phase33.get("merge_gauge_carrier_pass") is False,
        "forbidden_repeats_non_empty": len(FORBIDDEN_REPEATS) > 0,
        "hypothesis_matrix_maps_to_pairwise_qk": any("current-query" in row["v85_test"] for row in HYPOTHESES),
    }
    return {
        "schema": "acl2_v85_phase0_evidence_lock_summary_v1",
        "phase0_gate_pass": all(checks.values()),
        "checks": checks,
        "required_inputs": {
            "count": len(INPUTS),
            "present": inputs_ok,
        },
        "forbidden_repeat_count": len(FORBIDDEN_REPEATS),
        "hypothesis_count": len(HYPOTHESES),
        "runtime_action_allowed": False,
        "ttt_allowed": False,
        "next_phase": "Phase1_anchor_pair_universe" if all(checks.values()) else "fix_phase0_missing_or_contradictory_evidence",
    }


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    v83_final = read_json(INPUTS["v83_final_decision"])
    v84_phase12 = read_json(INPUTS["v84_phase12_final_decision"])
    v84_phase16 = read_json(INPUTS["v84_phase16_continuation_decision"])
    v84_phase22 = read_json(INPUTS["v84_phase22_external_variant_summary"])
    v84_phase33 = read_json(INPUTS["v84_phase33_merge_gauge_coverage"])
    if not isinstance(v84_phase22, list):
        v84_phase22 = []

    input_rows = required_input_rows()
    forbidden_rows = [
        {"forbidden_repeat": name, "reason": reason, "status": "forbidden_before_v85_pairwise_qk_gates"}
        for name, reason in FORBIDDEN_REPEATS
    ]
    hypothesis_rows = [{**row, "runtime_claim_allowed": False} for row in HYPOTHESES]
    gate = phase0_gate(v84_phase22, v84_phase33, all_required_inputs_present(input_rows))

    evidence_lock = {
        "schema": "acl2_v85_phase0_evidence_lock_v1",
        "phase0_gate_pass": gate["phase0_gate_pass"],
        "prior_boundaries": {
            "v83": {
                "active_decision_labels": v83_final.get("active_decision_labels", []),
                "conclusion": v83_final.get("conclusion", ""),
                "runtime_action_allowed": False,
            },
            "v84": {
                "phase12_final_status": v84_phase12.get("final_status"),
                "phase12_decision_labels": v84_phase12.get("decision_labels", []),
                "phase16_final_status_after_continuation": v84_phase16.get("final_status_after_continuation"),
                "phase16_runtime_action_allowed": v84_phase16.get("runtime_action_allowed"),
                "phase33_merge_gauge_carrier_pass": v84_phase33.get("merge_gauge_carrier_pass"),
            },
        },
        "locked_interpretation": [
            "Do not repeat source-side external anchor mask sweeps.",
            "Do not run merge/gauge tolerance sweeps without pairwise aligned evidence.",
            "Do not run runtime SWA action or TTT before v85 gates pass.",
            "Proceed with pairwise current-Q to historical/cache-K latent ruler alignment.",
        ],
        "evidence_paths": {name: str(path) for name, path in INPUTS.items()},
    }

    write_json(out_dir / "evidence_lock.json", evidence_lock)
    write_csv(out_dir / "required_inputs.csv", input_rows)
    write_csv(out_dir / "forbidden_repeats.csv", forbidden_rows)
    write_csv(out_dir / "v85_hypothesis_matrix.csv", hypothesis_rows)
    write_json(out_dir / "phase0_gate_summary.json", gate)
    (out_dir / "v84_no_go_boundary.md").write_text(
        build_v84_boundary(v83_final, v84_phase12, v84_phase16, v84_phase22, v84_phase33),
        encoding="utf-8",
    )
    (out_dir / "forbidden_repeats.md").write_text(build_forbidden_repeats_md(forbidden_rows), encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
