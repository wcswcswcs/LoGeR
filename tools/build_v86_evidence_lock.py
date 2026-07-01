#!/usr/bin/env python3
"""Build ACL2 v86 Phase0 evidence lock from landed v83/v84/v85 artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from v86_soft_latent_utils import read_json, write_csv, write_json


DEFAULT_OUT = Path("results/acl2_v86tf_robust_soft_latent_gauge_transport/phase0_evidence_lock")


FORBIDDEN_REPEATS = [
    "hard-anchor threshold micro sweep",
    "source-side anchor mask boost",
    "P9_48/P9_49 external source mask route family",
    "head13/head15 source mask rerun",
    "v84 weak_medium_conf_nonzero_fixed continuation",
    "merge support-map-driven overlap_outlier repeat",
    "TTT before confirmed SWA/merge evidence",
]


REQUIRED_INPUTS = [
    ("v85_final_decision", "results/acl2_v85tf_latent_anchor_alignment_pairwise_memory_ruler/report_final/final_decision.json"),
    ("v85_phase1_summary", "results/acl2_v85tf_latent_anchor_alignment_pairwise_memory_ruler/phase1_anchor_pair_universe/anchor_pair_sufficiency_summary.json"),
    ("v85_phase2_feature_sanity", "results/acl2_v85tf_latent_anchor_alignment_pairwise_memory_ruler/phase2_qk_feature_bank/feature_sanity_summary.json"),
    ("v85_visual_audit", "results/acl2_v85tf_latent_anchor_alignment_pairwise_memory_ruler/phase12_visual_rediscovery/visual_integrity_audit.json"),
    ("v84_final_decision", "results/acl2_v84tf_memory_ruler_audit/phase12_decision_matrix/final_decision.json"),
    ("v84_source_mask_variant_summary", "results/acl2_v84tf_memory_ruler_audit/phase22_v84_external_variant_summary/variant_summary.json"),
    ("v84_merge_gauge_coverage", "results/acl2_v84tf_memory_ruler_audit/phase33_v84_merge_gauge_coverage_decision/merge_gauge_coverage_summary.json"),
    ("v83_final_decision", "results/acl2_v83tf_clue_sufficiency_vs_action_misuse/phase10_decision_matrix/final_decision.json"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _load_optional(path: Path) -> Any | None:
    if not path.exists():
        return None
    return read_json(path)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    inputs: dict[str, Any | None] = {name: _load_optional(Path(path)) for name, path in REQUIRED_INPUTS}
    input_rows = [
        {
            "name": name,
            "path": path,
            "exists": Path(path).exists(),
            "required_for": "Phase0",
        }
        for name, path in REQUIRED_INPUTS
    ]
    write_csv(args.out_dir / "required_inputs.csv", input_rows)
    write_csv(args.out_dir / "forbidden_repeats.csv", [{"forbidden_repeat": item} for item in FORBIDDEN_REPEATS])
    (args.out_dir / "forbidden_repeats.md").write_text(
        "# Forbidden Repeats\n\n" + "\n".join(f"- {item}" for item in FORBIDDEN_REPEATS) + "\n",
        encoding="utf-8",
    )

    v85 = inputs["v85_final_decision"] or {}
    v84 = inputs["v84_final_decision"] or {}
    v83 = inputs["v83_final_decision"] or {}
    v84_variant = inputs["v84_source_mask_variant_summary"] or []
    v84_merge = inputs["v84_merge_gauge_coverage"] or {}

    source_mask_no_go_locked = bool(
        v84.get("final_status") == "No-Go_before_runtime_action"
        and isinstance(v84_variant, list)
        and len(v84_variant) >= 1
        and all((row.get("selected_lift_margin", 0.0) <= 0.0 and row.get("source_lift_margin", 0.0) <= 0.0) for row in v84_variant)
    )
    merge_gauge_no_go_locked = bool(
        v84.get("final_status") == "No-Go_before_runtime_action"
        and v84_merge.get("merge_gauge_carrier_pass") is False
        and v84_merge.get("runtime_action_allowed") is False
    )

    checks = {
        "v85_final_status_no_go_before_runtime": v85.get("final_status") == "No-Go_before_runtime_action",
        "v85_blocker_strong_bad_support_insufficient": v85.get("blocker") == "strong_bad_support_insufficient",
        "v85_phase2_feature_gate_pass": v85.get("phase2_feature_gate_pass") is True,
        "v85_visual_audit_gate_pass": v85.get("visual_audit_gate_pass") is True,
        "v84_source_mask_no_go_locked": source_mask_no_go_locked,
        "v84_merge_gauge_no_go_locked": merge_gauge_no_go_locked,
        "v83_no_runtime_action_locked": v83.get("final_status") == "No-Go_before_runtime_action",
        "forbidden_repeats_non_empty": len(FORBIDDEN_REPEATS) > 0,
        "all_required_inputs_exist": all(row["exists"] for row in input_rows),
    }
    gate_pass = all(checks.values())

    evidence = {
        "phase": "Phase0_evidence_lock",
        "gate_pass": gate_pass,
        "checks": checks,
        "v85_boundary": {
            "final_status": v85.get("final_status"),
            "blocker": v85.get("blocker"),
            "strong_bad_pair_rows": (v85.get("key_metrics") or {}).get("strong_bad_pair_rows"),
            "phase2_feature_gate_pass": v85.get("phase2_feature_gate_pass"),
            "visual_audit_gate_pass": v85.get("visual_audit_gate_pass"),
        },
        "v84_boundary": {
            "final_status": v84.get("final_status"),
            "decision_labels": v84.get("decision_labels"),
            "source_mask_no_go_locked": source_mask_no_go_locked,
            "merge_gauge_no_go_locked": merge_gauge_no_go_locked,
            "merge_gauge_reason": v84_merge.get("reason"),
        },
        "v83_boundary": {
            "final_status": v83.get("final_status"),
            "primary_decision_labels": v83.get("primary_decision_labels"),
            "runtime_action_allowed": v83.get("runtime_action_allowed"),
        },
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    write_json(args.out_dir / "evidence_lock.json", evidence)
    write_json(args.out_dir / "phase0_gate_summary.json", evidence)

    (args.out_dir / "v85_no_go_boundary.md").write_text(
        "\n".join(
            [
                "# v85 No-Go Boundary",
                "",
                f"- final_status: `{v85.get('final_status')}`",
                f"- blocker: `{v85.get('blocker')}`",
                f"- phase2_feature_gate_pass: `{v85.get('phase2_feature_gate_pass')}`",
                f"- visual_audit_gate_pass: `{v85.get('visual_audit_gate_pass')}`",
                f"- strong_bad_pair_rows: `{(v85.get('key_metrics') or {}).get('strong_bad_pair_rows')}`",
                "",
                "v86 must not continue hard-anchor threshold micro sweeps. It may test soft weighted robust transport.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    hypothesis_rows = [
        {
            "hypothesis_id": "H1",
            "question": "Can soft weighted reliable support overcome v85 hard-anchor sparsity?",
            "phase": "Phase1",
        },
        {
            "hypothesis_id": "H2",
            "question": "Can low-degree current-to-history C beat identity/random/shuffle heldout controls?",
            "phase": "Phase2",
        },
        {
            "hypothesis_id": "H3",
            "question": "Can anchor absence or historical prior detect bad scale/gauge handoffs?",
            "phase": "Phase3/4",
        },
    ]
    write_csv(args.out_dir / "v86_hypothesis_matrix.csv", hypothesis_rows)

    print(f"phase0_gate_pass={gate_pass}")
    print(f"v85_status={v85.get('final_status')}")
    print(f"v84_source_mask_no_go_locked={source_mask_no_go_locked}")
    print(f"v84_merge_gauge_no_go_locked={merge_gauge_no_go_locked}")


if __name__ == "__main__":
    main()
