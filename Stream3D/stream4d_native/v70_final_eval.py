from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_v65_scene_multiview_ap import _sha256, _write_csv, _write_json  # noqa: E402
from tools.run_v65_soma_pipeline_visualization import _rel  # noqa: E402


def _rooted(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _oracle_metric(summary: dict[str, Any], variant: str, field: str) -> Any:
    rows = summary.get("rows") or {}
    metric_path_text = rows.get("capsule_metric_rows_csv")
    if not metric_path_text:
        return None
    metric_path = _rooted(str(metric_path_text))
    if not metric_path.exists():
        return None
    import csv

    with metric_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("capsule_variant")) == variant or str(row.get("variant")) == variant:
                value = row.get(field)
                if value is None or value == "":
                    return None
                try:
                    return float(value)
                except ValueError:
                    return value
    return None


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _rooted(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    phase0_path = _rooted(args.phase0_summary)
    witness_path = _rooted(args.witness_summary)
    closure_path = _rooted(args.closure_summary)
    tracklet_path = _rooted(args.tracklet_summary)
    tracklet_repair_path = _rooted(args.tracklet_repair_summary)
    capsule_path = _rooted(args.capsule_summary)
    capsule_oracle_path = _rooted(args.capsule_oracle_summary)
    casebook_path = _rooted(args.casebook_summary)
    phase0 = _load_json(phase0_path)
    witness = _load_json(witness_path)
    closure = _load_json(closure_path)
    tracklet = _load_json(tracklet_path)
    tracklet_repair = _load_json(tracklet_repair_path)
    capsule = _load_json(capsule_path)
    capsule_oracle = _load_json(capsule_oracle_path)
    casebook = _load_json(casebook_path)
    closure_best = closure.get("best_closure_variant") or {}
    tracklet_best = tracklet.get("best_tracklet_variant") or {}
    tracklet_repair_best = tracklet_repair.get("best_tracklet_variant") or {}
    capsule_best = capsule.get("best_capsule_variant") or {}
    capsule_phase_decision = capsule.get("decision") or "not_run_blocked_by_tracklet"
    capsule_local_pass = bool((capsule.get("full_gate") or {}).get("pass"))
    casebook_gate_pass = bool((casebook.get("gate") or {}).get("pass", False))
    casebook_visual_note = (
        "A fallback 2D PNG/HTML failure casebook was generated from method artifacts without GT for prediction; Viser live loading remains unverified."
        if casebook_gate_pass
        else "The fallback failure casebook did not satisfy the visualization gate, so no visual success is claimed."
    )
    phase_rows = [
        {"phase": "phase0_fact_code_lock", "decision": phase0.get("decision"), "pass": (phase0.get("gate") or {}).get("pass")},
        {"phase": "phase1_carrier_witness", "decision": witness.get("decision"), "pass": (witness.get("gate") or {}).get("pass")},
        {"phase": "phase2_true_material_closure", "decision": closure.get("decision"), "pass": (closure.get("gate") or {}).get("pass")},
        {"phase": "phase3_masklet_tracklets", "decision": tracklet.get("decision"), "pass": (tracklet.get("gate") or {}).get("pass")},
        {"phase": "phase3_repair1_gap60", "decision": tracklet_repair.get("decision"), "pass": (tracklet_repair.get("gate") or {}).get("pass")},
        {"phase": "phase4_object_capsules", "decision": capsule_phase_decision, "pass": capsule_local_pass},
        {"phase": "phase5_scene_mv_ap", "decision": "not_run_blocked_by_local_gate", "pass": False},
        {"phase": "phase6_local2history", "decision": "blocked_by_local", "pass": False},
        {"phase": "phase7_casebook_viser", "decision": casebook.get("decision") or "not_run_visualization_blocker", "pass": (casebook.get("gate") or {}).get("pass", False)},
    ]
    final_label = "NO_GO_TRUE_MATERIAL_CLOSURE"
    if (witness.get("gate") or {}).get("pass") and not (closure.get("gate") or {}).get("pass"):
        final_label = "NO_GO_TRUE_MATERIAL_CLOSURE"
    if not (tracklet.get("gate") or {}).get("pass"):
        secondary_label = "NO_GO_MASKLET_TRACKLETS"
    else:
        secondary_label = "PARTIAL_TRACKLET_SIGNAL"
    tertiary_label = capsule_phase_decision if capsule else "not_run_blocked_by_tracklet"
    summary = {
        "phase": "v70_final_decision",
        "decision": final_label,
        "secondary_decision": secondary_label,
        "tertiary_decision": tertiary_label,
        "overall_status": "not_achieved",
        "phase_status": {
            "phase0_v69r2_fact_code_lock": phase0.get("decision"),
            "phase1_carrier_witness": witness.get("decision"),
            "phase2_true_material_closure": closure.get("decision"),
            "phase3_masklet_tracklets": tracklet.get("decision"),
            "phase4_object_capsules": capsule_phase_decision,
            "phase5_scene_mv_ap": "not_run_blocked_by_local_gate",
            "phase6_local2history": "blocked_by_local",
            "phase7_casebook_viser": casebook.get("decision") or "not_run_visualization_blocker",
        },
        "required_answers": {
            "v69r2_failed_due_to_proxy_closure": bool((phase0.get("gate") or {}).get("pass")),
            "true_carrier_witness_table_constructed": bool((witness.get("gate") or {}).get("pass")),
            "carrier_closure_beats_component_proxy_and_controls": bool((closure.get("gate") or {}).get("pass")),
            "d4rt_role_after_phase2": "verifier_veto_not_primary_glue",
            "masklet_tracklet_lowers_single_frame_rate": False,
            "object_capsule_forms_stable_local_objects": bool(capsule_local_pass),
            "underseg_shared_not_positive_bridge": "partially_enforced_but_did_not_improve",
            "local_gate_pass": False,
            "failure_layer": "true_carrier_closure_tracklet_generation_and_object_capsule_mask_source_underseg_reliance",
            "local2history_status": "blocked_by_local",
            "legal_all_candidate_oracle_high_but_underseg_dependent": True if capsule_oracle else None,
            "casebook_supports_conclusion": bool((casebook.get("gate") or {}).get("pass")),
        },
        "key_metrics": {
            "phase1_anchor_with_carrier_witness_rate": witness.get("anchor_with_carrier_witness_rate"),
            "phase1_candidate_masks_per_anchor_mean": witness.get("candidate_masks_per_anchor_mean"),
            "phase2_best_variant": closure_best.get("closure_variant"),
            "phase2_best_SF50": closure_best.get("single_anchor_SF50"),
            "phase2_best_AP50": closure_best.get("single_anchor_AP50"),
            "phase2_best_GT_best_IoU": closure_best.get("single_anchor_GT_best_IoU_mean"),
            "phase2_real_minus_no_temporal_SF50": closure.get("real_minus_no_temporal_SF50"),
            "phase2_real_minus_shuffled_SF50": closure.get("real_minus_shuffled_SF50"),
            "v69r2_C10_SF50": (closure.get("v69r2_C10_reproduction") or {}).get("single_anchor_SF50"),
            "phase3_best_variant": tracklet_best.get("tracklet_variant"),
            "phase3_best_SF50": tracklet_best.get("tracklet_SF50"),
            "phase3_tracklet_length_mean": tracklet_best.get("tracklet_length_mean"),
            "phase3_tracklet_single_frame_rate": tracklet_best.get("tracklet_single_frame_rate"),
            "phase3_tracklet_purity_diagnostic": tracklet_best.get("tracklet_purity_diagnostic"),
            "phase3_repair1_best_variant": tracklet_repair_best.get("tracklet_variant"),
            "phase3_repair1_best_SF50": tracklet_repair_best.get("tracklet_SF50"),
            "phase4_best_variant": capsule_best.get("capsule_variant"),
            "phase4_best_SF50": capsule_best.get("local_SF50"),
            "phase4_best_AP50": capsule_best.get("local_AP50"),
            "phase4_best_GT_best_IoU": capsule_best.get("GT_best_IoU_mean"),
            "phase4_best_single_frame_rate": capsule_best.get("single_frame_object_rate"),
            "phase4_best_underseg_false_bridge_rate": capsule_best.get("underseg_false_bridge_rate"),
            "phase4_oracle_all_cannotlink_SF50": _oracle_metric(capsule_oracle, "OC9_oracle_all_candidates_cannotlink_diagnostic", "local_SF50"),
            "phase4_oracle_nonshared_cannotlink_SF50": _oracle_metric(capsule_oracle, "OC10_oracle_nonshared_cannotlink_diagnostic", "local_SF50"),
            "phase7_case_count": casebook.get("case_count"),
            "phase7_bookmark_count": casebook.get("bookmark_count"),
            "phase7_viewer_scene_count": casebook.get("viewer_scene_count"),
            "phase7_method_layers_load_without_GT": casebook.get("method_layers_load_without_GT"),
            "phase7_screenshot_count": casebook.get("screenshot_count"),
            "phase7_viser_live_load_verified": casebook.get("viser_live_load_verified"),
        },
        "blocker_analysis": [
            "Phase 0 confirmed v69-r2 material closure was proxy/component-based rather than true carrier R_in/R_out.",
            "Phase 1 built a non-GT carrier witness table successfully.",
            "Phase 2 true carrier closure did not beat v69-r2 C10 component proxy, failed absolute gates, and matched no-temporal controls.",
            "Phase 2 repair with adjacent-only and underseg-veto variants did not improve the smoke scene.",
            "Phase 3 tracklet construction failed on the smoke scene and gap60/topk repair did not improve it, so object capsule assembly is blocked by tracklet generation.",
            "Continuation implemented a v70 object-capsule repair using v68 frozen-appearance edge evidence as mask evidence and v70 carrier witness as reward/veto; probe5 still failed the capsule local gate.",
            "Diagnostic cannot-link oracle shows all-candidate legal headroom remains high, but nonshared legal oracle is much lower, indicating the upper bound depends heavily on shared/underseg mask source.",
            casebook_visual_note,
        ],
        "source_summaries": {
            "phase0": _rel(phase0_path),
            "witness": _rel(witness_path),
            "closure": _rel(closure_path),
            "tracklet": _rel(tracklet_path),
            "tracklet_repair": _rel(tracklet_repair_path),
            "capsule": _rel(capsule_path),
            "capsule_oracle": _rel(capsule_oracle_path),
            "casebook": _rel(casebook_path),
        },
        "rows": {
            "phase_summary_rows_csv": _rel(output_root / "phase_summary_rows.csv"),
        },
    }
    _write_csv(output_root / "phase_summary_rows.csv", phase_rows)
    _write_json(output_root / "final_decision.json", summary)
    sha_rows = []
    for path in [output_root / "final_decision.json", output_root / "phase_summary_rows.csv"]:
        sha_rows.append({"path": _rel(path), "sha256": _sha256(path), "bytes": int(path.stat().st_size)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream4D v70 final decision builder.")
    parser.add_argument("--output-root", default="outputs/audit/v70_final_decision")
    parser.add_argument("--phase0-summary", default="outputs/audit/v70_phase0_fact_code_lock/fact_code_lock_summary.json")
    parser.add_argument("--witness-summary", default="outputs/audit/v70_carrier_witness/witness_summary.json")
    parser.add_argument("--closure-summary", default="outputs/audit/v70_true_material_closure/closure_summary.json")
    parser.add_argument("--tracklet-summary", default="outputs/audit/v70_masklet_tracklets_smoke_scene0011/tracklet_summary.json")
    parser.add_argument("--tracklet-repair-summary", default="outputs/audit/v70_masklet_tracklets_repair1_gap60_smoke_scene0011/tracklet_summary.json")
    parser.add_argument("--capsule-summary", default="outputs/audit/v70_object_capsules_repair2_probe5/capsule_summary.json")
    parser.add_argument("--capsule-oracle-summary", default="outputs/audit/v70_object_capsules_oracle_cannotlink_smoke_scene0011/capsule_summary.json")
    parser.add_argument("--casebook-summary", default="outputs/audit/v70_casebook/casebook_summary.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
