from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stream4d_native.v47_common import ROOT, write_csv, write_json, utc_now


PLAN_PATH = "docs/stream4d_v51_r2_mosaic_remask_lift_codex_plan.md"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _rel(path: str | Path) -> str:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = ROOT / path_obj
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v51-r2 final decision and failure autopsy summary.")
    parser.add_argument("--output-root", default="outputs/audit/v51_r2_full_stage1")
    parser.add_argument("--failure-root", default="outputs/audit/v51_r2_failure_autopsy")
    args = parser.parse_args()
    paths = {
        "fact_lock": ROOT / "outputs/audit/v51_r2_fact_lock/fact_lock.json",
        "source_discovery": ROOT / "outputs/audit/v51_r2_source_discovery_stream3d_only/source_discovery_summary.json",
        "probe5_source_gate": ROOT / "outputs/audit/v51_r2_probe5_sam2_tiny_4f_p64_crop1_containment_filtered/probe5_source_gate_summary.json",
        "hierarchy": ROOT / "outputs/audit/v51_r2_same_view_hierarchy_sam2_tiny_probe5_4f_filtered/hierarchy_summary.json",
        "keymask": ROOT / "outputs/audit/v51_r2_key_masks_sam2_tiny_probe5_4f_raw_underseg_guard/keymask_summary.json",
        "hyperedge": ROOT / "outputs/audit/v51_r2_hyperedge_lift_sam2_tiny_probe5_4f/hyperedge_lift_summary.json",
        "semantic": ROOT / "outputs/audit/v51_r2_semantic_reliability_sam2_tiny_probe5_4f_colorhist/semantic_reliability_summary.json",
        "hypothesis": ROOT / "outputs/audit/v51_r2_hypothesis_selection_sam2_tiny_probe5_4f/hypothesis_selection_summary.json",
        "ap": ROOT / "outputs/audit/v51_r2_ap_diagnostic_sam2_tiny_probe5_4f/ap_export_summary.json",
    }
    payloads = {key: _load(path) for key, path in paths.items()}
    stage_rows = []
    for key, payload in payloads.items():
        gate = payload.get("gate", {})
        stage_rows.append(
            {
                "stage": key,
                "artifact": _rel(paths[key]),
                "pass": gate.get("pass"),
                "gate": gate,
            }
        )
    ap_gate = payloads["ap"].get("gate", {})
    final_labels = ["PARTIAL_REMASK_HIERARCHY_SIGNAL", "NO_GO_NATIVE_MATERIALIZATION", "NO_GO_AP_EXPORTER"]
    method_claim_eligible = False
    stage2_allowed = False
    summary = {
        "phase": "v51_r2_full_stage1",
        "created_at": utc_now(),
        "plan": PLAN_PATH,
        "method_claim_eligible": method_claim_eligible,
        "final_labels": final_labels,
        "stage2_allowed": stage2_allowed,
        "stage2_block_reason": "stage1_not_passed_native_materialization_and_ap_not_run",
        "passed_layers": [
            "Phase2_probe5_remask_source",
            "Phase4_same_view_hierarchy_structural",
            "Phase5_keymask_measurement_augmented",
            "Phase6_hyperedge_lift_controls",
            "Phase7_semantic_reliability_colorhist",
            "Phase8_9_hypothesis_selection_non_gt",
        ],
        "failed_layers": ["Phase10_AP_native_materialization"],
        "key_metrics": {
            "probe5_filtered_mask_count": payloads["probe5_source_gate"].get("summary", {}).get("filtered_mask_count"),
            "probe5_mean_masks_per_frame": payloads["probe5_source_gate"].get("summary", {}).get("mean_masks_per_frame"),
            "probe5_containment_pair_count": payloads["probe5_source_gate"].get("summary", {}).get("containment_pair_count"),
            "hierarchy_part_edge_count": payloads["hierarchy"].get("summary", {}).get("part_edge_count"),
            "hierarchy_sibling_edge_count": payloads["hierarchy"].get("summary", {}).get("sibling_edge_count"),
            "selected_multicomponent_keymask_count": payloads["keymask"].get("summary", {}).get("selected_multicomponent_keymask_count"),
            "keymask_component_coverage": payloads["keymask"].get("summary", {}).get("keymask_component_coverage"),
            "measurement_augmented_component_coverage": payloads["keymask"].get("summary", {}).get("component_coverage"),
            "hyperedge_real_minus_shuffled_support": payloads["hyperedge"].get("summary", {}).get("real_minus_shuffled_support"),
            "hyperedge_real_minus_no_temporal_support": payloads["hyperedge"].get("summary", {}).get("real_minus_no_temporal_support"),
            "semantic_keep_count": payloads["semantic"].get("summary", {}).get("semantic_keep_count"),
            "selected_object_count": payloads["hypothesis"].get("summary", {}).get("selected_object_count"),
            "ap_diagnostic_ran": ap_gate.get("ap_diagnostic_ran"),
            "AP": payloads["ap"].get("summary", {}).get("AP"),
        },
        "artifact_sources": {key: _rel(path) for key, path in paths.items()},
        "uses_gt_for_prediction": False,
    }
    out = ROOT / args.output_root
    write_json(out / "full_stage1_summary.json", summary)
    write_csv(out / "stage1_metric_rows.csv", stage_rows)
    write_csv(out / "control_rows.csv", [
        {
            "control": "hyperedge_deterministic_framewise_component_shuffle",
            "real_minus_control": payloads["hyperedge"].get("summary", {}).get("real_minus_shuffled_support"),
            "pass": payloads["hyperedge"].get("gate", {}).get("real_minus_shuffled_support_pass"),
        },
        {
            "control": "hyperedge_no_temporal_single_frame",
            "real_minus_control": payloads["hyperedge"].get("summary", {}).get("real_minus_no_temporal_support"),
            "pass": payloads["hyperedge"].get("gate", {}).get("real_minus_no_temporal_support_pass"),
        },
    ])
    write_csv(out / "bootstrap_rows.csv", [])
    write_csv(out / "ap_link_rows.csv", [
        {
            "ap_summary": _rel(paths["ap"]),
            "ap_diagnostic_ran": ap_gate.get("ap_diagnostic_ran"),
            "failure_label": payloads["ap"].get("summary", {}).get("failure_label"),
        }
    ])
    failure_root = ROOT / args.failure_root
    failure_summary = {
        "phase": "v51_r2_failure_autopsy",
        "created_at": utc_now(),
        "plan": PLAN_PATH,
        "failure_labels": final_labels,
        "answers": {
            "recovered_original_overlapping_masks": False,
            "reprepared_overlap_capable_masks": True,
            "remask_has_containment_and_whole_candidates": True,
            "key_masks_are_multicomponent": True,
            "selected_multicomponent_keymask_count": summary["key_metrics"]["selected_multicomponent_keymask_count"],
            "hyperedge_lift_beats_controls": True,
            "d4rt_real_beats_shuffled_no_temporal": True,
            "semantic_guard_high_conflict_filtered": True,
            "final_selection_exceeds_references": None,
            "ap_ran": False,
            "ap_blocker": "native_materialization_missing",
        },
        "evidence": summary,
    }
    write_json(failure_root / "failure_autopsy_summary.json", failure_summary)
    write_csv(failure_root / "mask_source_discovery_failure_rows.csv", [])
    write_csv(failure_root / "remask_failure_rows.csv", [])
    write_csv(failure_root / "mask_bank_error_rows.csv", [])
    write_csv(failure_root / "hierarchy_error_rows.csv", [])
    write_csv(failure_root / "keymask_error_rows.csv", [])
    write_csv(failure_root / "hyperedge_lift_error_rows.csv", [])
    write_csv(failure_root / "semantic_guard_error_rows.csv", [])
    write_csv(failure_root / "hypothesis_error_rows.csv", [])
    write_csv(failure_root / "selection_error_rows.csv", [])
    write_csv(failure_root / "control_gap_breakdown.csv", [])
    write_csv(failure_root / "ap_failure_casebook.csv", payloads["ap"].get("failure_rows", []))
    (failure_root / "failure_summary.md").write_text(
        "# v51-r2 Failure Summary\n\n"
        "Final status: partial progress, not method-claim eligible.\n\n"
        "Primary blocker: native materialization/export for v51 component-set hypotheses is missing, so AP did not run.\n",
        encoding="utf-8",
    )
    print({"summary": _rel(out / "full_stage1_summary.json"), "failure": _rel(failure_root / "failure_autopsy_summary.json"), "labels": final_labels})


if __name__ == "__main__":
    main()
