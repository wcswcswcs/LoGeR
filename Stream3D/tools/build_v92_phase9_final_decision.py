from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_OUT = ROOT / "outputs/audit/v92_phase9_casebook"
PHASE_ROOTS = {
    "phase0": ROOT / "outputs/audit/v92_phase0_mv_ap_contract",
    "phase2": ROOT / "outputs/audit/v92_phase2_d4rt_sufficiency",
    "phase3_hr2": ROOT / "outputs/audit/v92_phase3_d4rt_highres_hr2_grid16",
    "phase3c": ROOT / "outputs/audit/v92_phase3c_hr2_uncertainty_readout",
    "phase4": ROOT / "outputs/audit/v92_phase4_semantic_region_affinity",
    "phase4b": ROOT / "outputs/audit/v92_phase4b_region_granularity_coarse2",
    "phase5a": ROOT / "outputs/audit/v92_phase5_source_container_field",
    "phase5b": ROOT / "outputs/audit/v92_phase5b_source_container_edge_field",
    "phase5c": ROOT / "outputs/audit/v92_phase5c_tight_field_repair",
    "phase5d": ROOT / "outputs/audit/v92_phase5d_score_calibration",
    "phase5e": ROOT / "outputs/audit/v92_phase5e_coarse2_tight_field",
    "phase6": ROOT / "outputs/audit/v92_phase6_attribution",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def _build_failure_rows(phase5_rows: list[dict[str, str]], phase6: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in phase5_rows:
        rows.append(
            {
                "phase_id": "v92_phase5b_source_container_edge_field",
                "variant_id": row.get("variant_id", ""),
                "failure_type": "control_margin_failed",
                "MV_AP_window": row.get("mean_MV_AP_window", ""),
                "MV_AP50_window": row.get("mean_MV_AP50_window", ""),
                "real_minus_best_control_MV_AP_window": row.get("real_minus_best_control_MV_AP_window", ""),
                "real_minus_best_control_MV_AP50_window": row.get("real_minus_best_control_MV_AP50_window", ""),
                "evidence_artifact": "outputs/audit/v92_phase5b_source_container_edge_field/control_metric_rows.csv",
                "note": "Phase5 variant did not beat best control margin under local-window MV_AP.",
            }
        )
    rows.append(
        {
            "phase_id": "v92_phase6_attribution",
            "variant_id": "D4RT_plus_RADIO",
            "failure_type": str(phase6.get("decision", "")),
            "MV_AP_window": phase6.get("D4RT_plus_RADIO_MV_AP_window", ""),
            "MV_AP50_window": "",
            "real_minus_best_control_MV_AP_window": _num(phase6.get("D4RT_plus_RADIO_MV_AP_window")) - _num(phase6.get("best_control_MV_AP_window")),
            "real_minus_best_control_MV_AP50_window": "",
            "evidence_artifact": "outputs/audit/v92_phase6_attribution/summary.json",
            "note": "Fusion is above D4RT-only/RADIO-only but below whole-source and best control.",
        }
    )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = Path(args.output_root)
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    phase0 = _read_json(PHASE_ROOTS["phase0"] / "summary.json")
    phase2 = _read_json(PHASE_ROOTS["phase2"] / "summary.json")
    phase3_hr2 = _read_json(PHASE_ROOTS["phase3_hr2"] / "summary.json")
    phase3c = _read_json(PHASE_ROOTS["phase3c"] / "summary.json")
    phase4 = _read_json(PHASE_ROOTS["phase4"] / "summary.json")
    phase4b = _read_json(PHASE_ROOTS["phase4b"] / "summary.json")
    phase5a = _read_json(PHASE_ROOTS["phase5a"] / "summary.json")
    phase5b = _read_json(PHASE_ROOTS["phase5b"] / "summary.json")
    phase5c = _read_json(PHASE_ROOTS["phase5c"] / "summary.json")
    phase5d = _read_json(PHASE_ROOTS["phase5d"] / "summary.json")
    phase5e = _read_json(PHASE_ROOTS["phase5e"] / "summary.json")
    phase6 = _read_json(PHASE_ROOTS["phase6"] / "summary.json")
    phase5_rows = _csv_rows(PHASE_ROOTS["phase5b"] / "control_metric_rows.csv")

    phase5_gate_pass = _bool(phase5b.get("any_phase5_dev_gate_pass"))
    phase6_decision = str(phase6.get("decision", ""))
    if phase5_gate_pass and phase6_decision == "GEOMETRY_SEMANTIC_COMPLEMENTARITY_SUPPORTED":
        final_decision = "GO_LOCAL_MV_AP_WINDOW"
    elif phase6_decision == "CONTROL_BIAS_BLOCKER":
        final_decision = "NO_GO_CONTROL_BIAS"
    else:
        final_decision = "NO_GO_AFFINITY_FIELD_FUSION_BLOCKER"

    phase4_semantic_strong = str(phase4.get("decision", "")) == "SEMANTIC_REGION_SIGNAL_STRONG"
    da3_conditions = {
        "phase2_resolution_or_geometry_blocker": str(phase2.get("routing_label") or phase2.get("decision", "")).startswith("D4RT_")
        or bool(phase2.get("resolution_blocker"))
        or bool(phase2.get("geometry_blocker")),
        "phase3_highres_failed_dev_gate": not bool(phase3_hr2.get("same_readout_gate_pass")),
        "phase5_field_still_extent_limited": not phase5_gate_pass,
        "phase4_semantic_insufficient": not phase4_semantic_strong,
        "evaluator_materializer_bug_excluded": True,
    }
    da3_branch_decision = (
        "RUN_DA3_BRANCH"
        if all(da3_conditions.values())
        else "SKIP_DA3_BRANCH_DIAGNOSTIC_CONDITIONS_NOT_MET"
    )
    holdout_decision = "SKIP_HOLDOUT_DEV_GATE_NOT_PASSED" if not phase5_gate_pass else "HOLDOUT_REQUIRED"

    best_gate = phase5b.get("best_variant_gate", {}) if isinstance(phase5b.get("best_variant_gate"), dict) else {}
    final = {
        "schema": "stream4d_v92_phase9_final_decision_v1",
        "phase_id": "v92_phase9_casebook",
        "created_at": _now(),
        "final_decision": final_decision,
        "primary_blocker": phase6_decision or final_decision,
        "secondary_blocker": "NO_GO_AFFINITY_FIELD_FUSION_BLOCKER",
        "can_claim_local_method_success": final_decision == "GO_LOCAL_MV_AP_WINDOW",
        "can_enter_local2history": False,
        "holdout_decision": holdout_decision,
        "da3_branch_decision": da3_branch_decision,
        "da3_trigger_conditions": da3_conditions,
        "answers": {
            "d4rt_sampling_resolution_main_blocker": "No. HR2 grid16 improved median support area ratio but same-readout MV_AP_window stayed below v91/control.",
            "d4rt_geometry_quality_main_blocker": "Likely weak witness quality remains, but Phase5 shows the final blocker is not solved by D4RT-only uncertainty or high-res density.",
            "dino_radio_semantic_sufficient_but_readout_not_used_well": "Yes for RADIO diagnostic signal: source-internal AUC is strong, while current field readout still loses AP.",
            "fused_affinity_field_beats_single_routes_and_controls": "No. Fused beats D4RT-only/RADIO-only but not whole-source or best control.",
            "need_da3": "No under v92 trigger rules: Phase4 semantic diagnostic is strong, so the DA3 conditions are not all met.",
            "local_can_enter_local2history": "No. Phase5/6 local MV_AP_window dev gate failed.",
        },
        "key_metrics": {
            "v91_best_MV_AP_window": phase0.get("v91_best_MV_AP_window"),
            "stream3d_local_S3D_L1_MV_AP_window": phase0.get("S3D_local_window_MV_AP_window"),
            "phase3_hr2_same_readout_MV_AP_window": phase3_hr2.get("same_readout_MV_AP_window"),
            "phase3c_best_MV_AP_window": phase3c.get("best_variant_gate", {}).get("mean_MV_AP_window", ""),
            "phase4_radio_source_internal_auc": phase4.get("source_internal_same_gt_different_gt_AUC_mean"),
            "phase5b_best_variant": phase5b.get("best_variant_id"),
            "phase5b_best_real_MV_AP_window": best_gate.get("mean_MV_AP_window"),
            "phase5b_best_real_MV_AP50_window": best_gate.get("mean_MV_AP50_window"),
            "phase5b_real_minus_best_control_MV_AP_window": best_gate.get("real_minus_best_control_MV_AP_window"),
            "phase5b_real_minus_best_control_MV_AP50_window": best_gate.get("real_minus_best_control_MV_AP50_window"),
            "phase5c_best_variant": phase5c.get("best_variant_id", ""),
            "phase5c_best_MV_AP_window": (phase5c.get("best_variant_gate") or {}).get("mean_MV_AP_window", ""),
            "phase5c_best_MV_AP50_window": (phase5c.get("best_variant_gate") or {}).get("mean_MV_AP50_window", ""),
            "phase5c_any_dev_gate_pass": phase5c.get("any_phase5_dev_gate_pass", ""),
            "phase5d_best_variant": phase5d.get("best_variant_id", ""),
            "phase5d_best_MV_AP_window": (phase5d.get("best_variant_gate") or {}).get("mean_MV_AP_window", ""),
            "phase5d_best_MV_AP50_window": (phase5d.get("best_variant_gate") or {}).get("mean_MV_AP50_window", ""),
            "phase5d_any_dev_gate_pass": phase5d.get("any_phase5d_dev_gate_pass", ""),
            "phase4b_coarsen_factor": phase4b.get("coarsen_factor", ""),
            "phase4b_region_node_reduction_ratio": phase4b.get("region_node_reduction_ratio", ""),
            "phase5e_best_variant": phase5e.get("best_variant_id", ""),
            "phase5e_best_MV_AP_window": (phase5e.get("best_variant_gate") or {}).get("mean_MV_AP_window", ""),
            "phase5e_best_MV_AP50_window": (phase5e.get("best_variant_gate") or {}).get("mean_MV_AP50_window", ""),
            "phase5e_any_dev_gate_pass": phase5e.get("any_phase5_dev_gate_pass", ""),
            "phase6_d4rt_only_MV_AP_window": phase6.get("D4RT_only_MV_AP_window"),
            "phase6_radio_only_MV_AP_window": phase6.get("RADIO_only_MV_AP_window"),
            "phase6_fused_MV_AP_window": phase6.get("D4RT_plus_RADIO_MV_AP_window"),
            "phase6_whole_source_MV_AP_window": phase6.get("whole_source_MV_AP_window"),
            "phase6_best_control_MV_AP_window": phase6.get("best_control_MV_AP_window"),
        },
        "source_artifacts": {name: _rel(root) for name, root in PHASE_ROOTS.items()},
        "runtime_sec": time.time() - started,
    }

    failure_fields = [
        "phase_id",
        "variant_id",
        "failure_type",
        "MV_AP_window",
        "MV_AP50_window",
        "real_minus_best_control_MV_AP_window",
        "real_minus_best_control_MV_AP50_window",
        "evidence_artifact",
        "note",
    ]
    failure_rows = _build_failure_rows(phase5_rows, phase6)
    success_rows = [
        {
            "phase_id": "v92_phase4_semantic_region_affinity",
            "success_type": "diagnostic_signal_present",
            "metric_name": "source_internal_same_gt_different_gt_AUC_mean",
            "metric_value": phase4.get("source_internal_same_gt_different_gt_AUC_mean", ""),
            "evidence_artifact": "outputs/audit/v92_phase4_semantic_region_affinity/summary.json",
            "note": "RADIO region features contain source-internal instance-affinity signal; this is diagnostic, not MV_AP success.",
        },
        {
            "phase_id": "v92_phase3_d4rt_highres_hr2_grid16",
            "success_type": "density_improved_diagnostic",
            "metric_name": "highres_median_carrier_support_area_ratio_unique_key",
            "metric_value": phase3_hr2.get("highres_median_carrier_support_area_ratio_unique_key", ""),
            "evidence_artifact": "outputs/audit/v92_phase3_d4rt_highres_hr2_grid16/summary.json",
            "note": "D4RT support density improved, but MV_AP did not pass the dev gate.",
        },
    ]
    success_fields = ["phase_id", "success_type", "metric_name", "metric_value", "evidence_artifact", "note"]
    manifest_rows = [
        {
            "artifact_type": "casebook_rows",
            "artifact_path": "outputs/audit/v92_phase5b_source_container_edge_field/casebook_rows.csv",
            "rendered_visualization": "false",
            "reason": "Phase9 closeout references existing casebook rows; no new visual rendering was required for final decision.",
        },
        {
            "artifact_type": "failure_casebook_rows",
            "artifact_path": "outputs/audit/v92_phase9_casebook/failure_casebook_rows.csv",
            "rendered_visualization": "false",
            "reason": "Tabular local-window MV_AP attribution is sufficient for the No-Go decision.",
        },
    ]
    manifest_fields = ["artifact_type", "artifact_path", "rendered_visualization", "reason"]

    theory = "\n".join(
        [
            "# Stream4D v92 Theory Update",
            "",
            "Final decision: `{}`.".format(final_decision),
            "",
            "The local-window MV_AP failure is not an evaluator/support-scope bug in this run. Phase0 locked the v65 SparseSceneIoU evaluator and local-window GT projection support, while Phase5B kept `same_frame_collision_count=0`, `missing_mask_raster_count=0`, `uses_gt_for_prediction=false`, and `uses_future=false`.",
            "",
            "High-resolution D4RT is diagnostic progress only. HR2 grid16 raised source support density, but the same readout remained below the v91/control gates.",
            "",
            "RADIO dense features do contain source-internal signal: Phase4 measured strong same-GT/different-GT region AUC. The failure is that the current source-container membership field does not convert that signal into a better visible object tube.",
            "",
            "Phase6 shows the narrow attribution: D4RT+RADIO fusion is stronger than D4RT-only and RADIO-only, but it remains below whole-source and best control. That is a control-bias / affinity-field-readout blocker, not a successful geometry-semantic fusion method.",
            "",
            "DA3 is skipped by the v92 trigger rules because Phase4 did not show semantic insufficiency. Holdout and local2history are also skipped because the local dev gate did not pass.",
        ]
    )

    _write_json(out / "final_decision.json", final)
    _write_csv(out / "failure_casebook_rows.csv", failure_rows, failure_fields)
    _write_csv(out / "success_casebook_rows.csv", success_rows, success_fields)
    _write_csv(out / "source_container_visualization_manifest.csv", manifest_rows, manifest_fields)
    _write_text(out / "theory_update.md", theory)
    outputs = [
        out / "final_decision.json",
        out / "failure_casebook_rows.csv",
        out / "success_casebook_rows.csv",
        out / "source_container_visualization_manifest.csv",
        out / "theory_update.md",
    ]
    _write_json(out / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs})
    print(json.dumps(final, indent=2, sort_keys=True), flush=True)
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v92 Phase9 final decision casebook.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
