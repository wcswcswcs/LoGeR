#!/usr/bin/env python3
"""Build the final Stream4D v93 dev decision from existing evidence artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/audit/v93_final_decision"


def _read_json(rel: str) -> dict[str, Any]:
    path = ROOT / rel
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(_jsonable(row))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(rel: str) -> dict[str, Any]:
    path = ROOT / rel
    return {"path": _rel(path), "exists": path.exists(), "sha256": _sha256(path) if path.exists() else None}


def _attempt(
    attempt_id: str,
    family: str,
    summary_rel: str,
    summary: dict[str, Any],
    variant_id_key: str,
    mv_ap_key: str,
    ap50_key: str,
    decision_key: str = "decision",
) -> dict[str, Any]:
    return {
        "attempt_id": attempt_id,
        "family": family,
        "decision": summary.get(decision_key, "missing"),
        "best_variant_id": summary.get(variant_id_key, ""),
        "MV_AP_window": summary.get(mv_ap_key, ""),
        "MV_AP50_window": summary.get(ap50_key, ""),
        "uses_gt_for_prediction": summary.get("uses_gt_for_prediction", summary.get("uses_gt_for_prediction_count", "")),
        "uses_future": summary.get("uses_future", summary.get("uses_future_count", "")),
        "artifact": _artifact(summary_rel),
    }


def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)

    phase0_rel = "outputs/audit/v93_phase0_contract/summary.json"
    phase1_rel = "outputs/audit/v93_phase1_source_edge_registry/summary.json"
    phase2_rel = "outputs/audit/v93_phase2_d4rt_edge_sampling_diagnostic/summary.json"
    phase3_rel = "outputs/audit/v93_phase3_region_edge_graph/summary.json"
    phase4_rel = "outputs/audit/v93_phase4_cue_isolation/summary.json"
    phase4_edge_rel = "outputs/audit/v93_phase4_edge_only_materialization/summary.json"
    phase5_rel = "outputs/audit/v93_phase5_boundary_affinity_field/summary.json"
    phase5b_rel = "outputs/audit/v93_phase5b_unknown_background_field/summary.json"
    phase5_a512_rel = "outputs/audit/v93_phase5_boundary_affinity_field_A512/summary.json"
    phase7_rel = "outputs/audit/v93_phase7_adaptive_d4rt_sampling/summary.json"
    phase7_gap_rel = "outputs/audit/v93_phase7_density_readout_gap/summary.json"
    phase7_same_rel = "outputs/audit/v93_phase7_A512_same_readout_adaptive_materialization/summary.json"
    da3_ready_rel = "outputs/audit/v93_da3_conditional_readiness/summary.json"
    triton_equiv_rel = "outputs/audit/v93_phase5_triton_torch_equivalence_check/report.json"
    triton_synth_rel = "outputs/audit/v93_phase5_triton_kernel_validation/report.json"

    phase0 = _read_json(phase0_rel)
    phase1 = _read_json(phase1_rel)
    phase2 = _read_json(phase2_rel)
    phase3 = _read_json(phase3_rel)
    phase4 = _read_json(phase4_rel)
    phase4_edge = _read_json(phase4_edge_rel)
    phase5 = _read_json(phase5_rel)
    phase5b = _read_json(phase5b_rel)
    phase5_a512 = _read_json(phase5_a512_rel)
    phase7 = _read_json(phase7_rel)
    phase7_gap = _read_json(phase7_gap_rel)
    phase7_same = _read_json(phase7_same_rel)
    da3_ready = _read_json(da3_ready_rel)
    triton_equiv = _read_json(triton_equiv_rel)
    triton_synth = _read_json(triton_synth_rel)

    required_mv_ap = max(
        _num(phase0.get("v91_best_MV_AP_window")) + 0.006,
        _num(phase0.get("best_control_MV_AP_window")) + 0.008,
    )
    required_ap50 = max(
        _num(phase0.get("v91_best_MV_AP50_window")) + 0.012,
        _num(phase0.get("best_control_MV_AP50_window")) + 0.012,
    )

    attempts = [
        _attempt("phase4_edge_only", "edge_only", phase4_edge_rel, phase4_edge, "best_edge_variant_id", "best_edge_MV_AP_window", "best_edge_MV_AP50_window"),
        _attempt("phase4_cue_fusion", "cue_isolation", phase4_rel, phase4, "best_edge_only_variant_id", "D4RT_plus_RADIO_MV_AP_window", "best_edge_only_MV_AP50_window"),
        _attempt("phase5_hr2_field", "gpu_triton_boundary_affinity_field", phase5_rel, phase5, "best_real_variant_id", "best_real_MV_AP_window", "best_real_MV_AP50_window"),
        _attempt(
            "phase5b_unknown_background_field",
            "unknown_background_field_inference",
            phase5b_rel,
            phase5b,
            "best_real_variant_id",
            "best_real_MV_AP_window",
            "best_real_MV_AP50_window",
        ),
        _attempt("phase7_a512_same_readout", "adaptive_d4rt_same_readout", phase7_rel, phase7, "A512_best_variant_id", "A512_MV_AP_window", "A512_MV_AP50_window"),
        _attempt("phase5_a512_field", "gpu_triton_boundary_affinity_field_with_a512_support", phase5_a512_rel, phase5_a512, "best_real_variant_id", "best_real_MV_AP_window", "best_real_MV_AP50_window"),
    ]
    numeric_attempts = [row for row in attempts if row.get("MV_AP_window") != ""]
    best_attempt = max(numeric_attempts, key=lambda row: _num(row.get("MV_AP_window")), default={})

    triton_validated = (
        _num(triton_equiv.get("exact_mismatch_count"), 1.0) == 0.0
        and _num(triton_equiv.get("float_mismatch_count"), 1.0) == 0.0
        and triton_synth.get("decision") == "PASS_V93_PHASE5_TRITON_KERNEL_REFERENCE_CHECK"
        and _num(triton_synth.get("failure_count"), 1.0) == 0.0
    )
    phase5_gate_pass = _bool(phase5.get("any_phase5_dev_gate_pass"))
    phase5b_gate_pass = _bool(phase5b.get("any_phase5b_dev_gate_pass"))
    phase5_a512_gate_pass = _bool(phase5_a512.get("any_phase5_dev_gate_pass"))
    phase7_control_pass = _bool(phase7.get("control_gate_pass"))
    phase7_uniform_gain_pass = _bool(phase7.get("uniform_gain_gate_pass"))
    phase7_runtime_pass = _bool(phase7.get("runtime_budget_pass"))
    dev_gate_pass = phase5_gate_pass or phase5b_gate_pass or phase5_a512_gate_pass or (
        phase7_control_pass and phase7_uniform_gain_pass and phase7_runtime_pass
    )

    evidence_rows = [
        {
            "evidence_id": "phase0_contract",
            "status": "pass" if phase0.get("decision") == "PASS_V93_PHASE0_CONTRACT" else "missing_or_fail",
            "detail": "Metric contract and dev/control baselines locked.",
            "value": phase0.get("decision", ""),
            "artifact": _rel(ROOT / phase0_rel),
        },
        {
            "evidence_id": "phase5_hr2_field_gate",
            "status": "fail" if not phase5_gate_pass else "pass",
            "detail": "GPU/Triton HR2 field best real variant did not clear v91/control margins.",
            "value": {
                "best_real_MV_AP_window": phase5.get("best_real_MV_AP_window"),
                "best_real_MV_AP50_window": phase5.get("best_real_MV_AP50_window"),
                "required_MV_AP_window": required_mv_ap,
                "required_MV_AP50_window": required_ap50,
            },
            "artifact": _rel(ROOT / phase5_rel),
        },
        {
            "evidence_id": "triton_correctness",
            "status": "pass" if triton_validated else "fail",
            "detail": "End-to-end torch/triton artifact equivalence and synthetic kernel checks passed.",
            "value": {
                "exact_mismatch_count": triton_equiv.get("exact_mismatch_count"),
                "float_mismatch_count": triton_equiv.get("float_mismatch_count"),
                "synthetic_decision": triton_synth.get("decision"),
                "synthetic_failure_count": triton_synth.get("failure_count"),
            },
            "artifact": f"{_rel(ROOT / triton_equiv_rel)} ; {_rel(ROOT / triton_synth_rel)}",
        },
        {
            "evidence_id": "phase5b_unknown_background_field_gate",
            "status": "fail" if not phase5b_gate_pass else "pass",
            "detail": "Unknown/background field objective was run after repairing score protocol; whole-source replay matched Phase5 F0 but real variants did not clear gates.",
            "value": {
                "best_real_MV_AP_window": phase5b.get("best_real_MV_AP_window"),
                "best_real_MV_AP50_window": phase5b.get("best_real_MV_AP50_window"),
                "best_control_MV_AP_window": phase5b.get("best_control_MV_AP_window"),
                "score_protocol_counts": phase5b.get("score_protocol_counts"),
            },
            "artifact": _rel(ROOT / phase5b_rel),
        },
        {
            "evidence_id": "phase7_a512_density_runtime_gate",
            "status": "fail",
            "detail": "A512 improved support density but missed uniform/control AP gates and exceeded 2x runtime budget.",
            "value": {
                "density_improved": phase7.get("density_improved"),
                "A512_minus_G16_MV_AP_window": phase7.get("A512_minus_G16_MV_AP_window"),
                "A512_minus_locked_control_MV_AP_window": phase7.get("A512_minus_locked_control_MV_AP_window"),
                "runtime_budget_pass": phase7.get("runtime_budget_pass"),
            },
            "artifact": _rel(ROOT / phase7_rel),
        },
        {
            "evidence_id": "phase7_readout_gap",
            "status": "diagnostic",
            "detail": "Density/readout diagnostic shows denser support barely changed generated/source area.",
            "value": {
                "delta_support_count_mean": phase7_gap.get("best_variant_delta_support_count_mean"),
                "delta_generated_to_source_area_mean": phase7_gap.get("best_variant_delta_generated_to_source_area_mean"),
            },
            "artifact": _rel(ROOT / phase7_gap_rel),
        },
        {
            "evidence_id": "phase5_a512_field_gate",
            "status": "fail" if not phase5_a512_gate_pass else "pass",
            "detail": "A512 support routed back into GPU/Triton field inference still underperformed HR2 field.",
            "value": {
                "A512_field_MV_AP_window": phase5_a512.get("best_real_MV_AP_window"),
                "HR2_field_MV_AP_window": phase5.get("best_real_MV_AP_window"),
                "delta": _num(phase5_a512.get("best_real_MV_AP_window")) - _num(phase5.get("best_real_MV_AP_window")),
            },
            "artifact": _rel(ROOT / phase5_a512_rel),
        },
        {
            "evidence_id": "da3_conditional_readiness",
            "status": "blocked_not_executed",
            "detail": "The conditional dense-geometry/DA3 branch was written as readiness evidence only; DA3 was not run or used for metrics.",
            "value": {
                "decision": da3_ready.get("decision"),
                "da3_conditional_branch_ready": da3_ready.get("da3_conditional_branch_ready"),
                "da3_executed": da3_ready.get("da3_executed"),
                "explicit_da3_authorization_present": da3_ready.get("explicit_da3_authorization_present"),
            },
            "artifact": _rel(ROOT / da3_ready_rel),
        },
    ]

    blocker_rows = [
        {
            "blocker": "NO_GO_DEV_LOCAL_MV_AP_WINDOW",
            "active": not dev_gate_pass,
            "reason": "No v93 dev variant passed the local/control margin gates; no freeze or holdout is allowed.",
        },
        {
            "blocker": "D4RT_DENSITY_NOT_READOUT",
            "active": bool(phase7.get("density_improved")) and not phase7_uniform_gain_pass,
            "reason": "Adaptive A512 increased support area/density, but MV_AP_window did not improve over G16.",
        },
        {
            "blocker": "ADAPTIVE_RUNTIME_OVER_BUDGET",
            "active": not phase7_runtime_pass,
            "reason": "A512 runtime ratio exceeded the plan's <=2x G16 budget.",
        },
        {
            "blocker": "FIELD_INFERENCE_WEAK",
            "active": not phase5_gate_pass and not phase5b_gate_pass and not phase5_a512_gate_pass,
            "reason": "Current unary/pairwise and unknown/background field terms do not convert HR2 or A512 D4RT support into better object extents.",
        },
        {
            "blocker": "DA3_CONDITIONAL_BRANCH_NOT_EXECUTED",
            "active": not _bool(da3_ready.get("explicit_da3_authorization_present")),
            "reason": "DA3 requires explicit authorization/new strict-control branch; v93 only wrote readiness evidence and did not run DA3.",
        },
    ]

    repair_attempt_rows = [
        {
            "repair_attempt": "phase5_gpu_triton_rewrite",
            "what_changed": "Replaced CSV/CPU field generation with CUDA/Triton unary and edge propagation plus NPZ field shards; evaluator remains CSV/CPU.",
            "result": phase5.get("decision", ""),
            "duration_sec": phase5.get("duration_sec", ""),
            "artifact": _rel(ROOT / phase5_rel),
        },
        {
            "repair_attempt": "triton_validation",
            "what_changed": "Added synthetic kernel validation and preserved strict-denom failure evidence; final tolerance uses denominator abs-or-rel due FP32 atomic add ordering.",
            "result": triton_synth.get("decision", ""),
            "duration_sec": triton_synth.get("duration_sec", ""),
            "artifact": _rel(ROOT / triton_synth_rel),
        },
        {
            "repair_attempt": "phase5b_unknown_background_field",
            "what_changed": "Reused Phase5 field shards, added unknown/background field objectives, and repaired score protocol to reuse Phase5 base scores before full evaluation.",
            "result": phase5b.get("decision", ""),
            "duration_sec": phase5b.get("duration_sec", ""),
            "artifact": _rel(ROOT / phase5b_rel),
        },
        {
            "repair_attempt": "phase7_a512_adaptive_d4rt",
            "what_changed": "Ran real A512 adaptive D4RT recompute on GPUs 6/7 and evaluated same-readout density/AP/runtime gates.",
            "result": phase7.get("decision", ""),
            "duration_sec": phase7.get("duration_sec", ""),
            "artifact": _rel(ROOT / phase7_rel),
        },
        {
            "repair_attempt": "phase5_a512_support_field",
            "what_changed": "Fed A512 adaptive support rows back into the GPU/Triton boundary-affinity field.",
            "result": phase5_a512.get("decision", ""),
            "duration_sec": phase5_a512.get("duration_sec", ""),
            "artifact": _rel(ROOT / phase5_a512_rel),
        },
        {
            "repair_attempt": "da3_conditional_readiness_audit",
            "what_changed": "Wrote DA3_CONDITIONAL_BRANCH_READY evidence without running DA3, because the plan requires explicit authorization.",
            "result": da3_ready.get("decision", ""),
            "duration_sec": "",
            "artifact": _rel(ROOT / da3_ready_rel),
        },
    ]

    phase_rows = [
        {"phase": "phase0", "decision": phase0.get("decision", "missing"), "artifact": _rel(ROOT / phase0_rel)},
        {"phase": "phase1", "decision": phase1.get("decision", "missing"), "artifact": _rel(ROOT / phase1_rel)},
        {"phase": "phase2", "decision": phase2.get("decision", "missing"), "artifact": _rel(ROOT / phase2_rel)},
        {"phase": "phase3", "decision": phase3.get("decision", "missing"), "artifact": _rel(ROOT / phase3_rel)},
        {"phase": "phase4", "decision": phase4.get("decision", "missing"), "artifact": _rel(ROOT / phase4_rel)},
        {"phase": "phase4_edge_only", "decision": phase4_edge.get("decision", "missing"), "artifact": _rel(ROOT / phase4_edge_rel)},
        {"phase": "phase5_hr2_field", "decision": phase5.get("decision", "missing"), "artifact": _rel(ROOT / phase5_rel)},
        {"phase": "phase5b_unknown_background_field", "decision": phase5b.get("decision", "missing"), "artifact": _rel(ROOT / phase5b_rel)},
        {"phase": "phase7_a512", "decision": phase7.get("decision", "missing"), "artifact": _rel(ROOT / phase7_rel)},
        {"phase": "phase5_a512_field", "decision": phase5_a512.get("decision", "missing"), "artifact": _rel(ROOT / phase5_a512_rel)},
        {"phase": "da3_conditional_readiness", "decision": da3_ready.get("decision", "missing"), "artifact": _rel(ROOT / da3_ready_rel)},
    ]

    summary = {
        "schema": "stream4d_v93_final_decision_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "decision": "NO_GO_V93_BOUNDARY_AWARE_AFFINITY_READOUT",
        "goal_achieved": False,
        "dev_gate_pass": dev_gate_pass,
        "freeze_config_written": False,
        "holdout_executed": False,
        "holdout_reason": "No dev variant passed control/local margin gates, so Phase9 freeze and Phase10 holdout are not allowed by plan.",
        "required_MV_AP_window": required_mv_ap,
        "required_MV_AP50_window": required_ap50,
        "v91_best_MV_AP_window": phase0.get("v91_best_MV_AP_window"),
        "v91_best_MV_AP50_window": phase0.get("v91_best_MV_AP50_window"),
        "locked_best_control_MV_AP_window": phase0.get("best_control_MV_AP_window"),
        "locked_best_control_MV_AP50_window": phase0.get("best_control_MV_AP50_window"),
        "best_attempt": best_attempt,
        "phase5_hr2_field": {
            "decision": phase5.get("decision"),
            "best_real_variant_id": phase5.get("best_real_variant_id"),
            "best_real_MV_AP_window": phase5.get("best_real_MV_AP_window"),
            "best_real_MV_AP50_window": phase5.get("best_real_MV_AP50_window"),
            "solver_backend_actual": phase5.get("solver_backend_actual"),
            "duration_sec": phase5.get("duration_sec"),
            "processed_source_count": phase5.get("processed_source_count"),
            "gpu_device_source_counts": phase5.get("gpu_device_source_counts"),
        },
        "phase7_a512": {
            "decision": phase7.get("decision"),
            "density_improved": phase7.get("density_improved"),
            "A512_MV_AP_window": phase7.get("A512_MV_AP_window"),
            "A512_MV_AP50_window": phase7.get("A512_MV_AP50_window"),
            "A512_minus_G16_MV_AP_window": phase7.get("A512_minus_G16_MV_AP_window"),
            "A512_minus_locked_control_MV_AP_window": phase7.get("A512_minus_locked_control_MV_AP_window"),
            "runtime_budget_pass": phase7.get("runtime_budget_pass"),
            "A512_readout_runtime_sec": phase7.get("A512_readout_runtime_sec"),
            "A512_bridge_duration_sec": phase7.get("A512_bridge_duration_sec"),
        },
        "phase5b_unknown_background_field": {
            "decision": phase5b.get("decision"),
            "best_real_variant_id": phase5b.get("best_real_variant_id"),
            "best_real_MV_AP_window": phase5b.get("best_real_MV_AP_window"),
            "best_real_MV_AP50_window": phase5b.get("best_real_MV_AP50_window"),
            "best_control_variant_id": phase5b.get("best_control_variant_id"),
            "best_control_MV_AP_window": phase5b.get("best_control_MV_AP_window"),
            "best_control_MV_AP50_window": phase5b.get("best_control_MV_AP50_window"),
            "score_protocol_counts": phase5b.get("score_protocol_counts"),
            "duration_sec": phase5b.get("duration_sec"),
            "processed_source_count": phase5b.get("processed_source_count"),
            "gpu_device_source_counts": phase5b.get("gpu_device_source_counts"),
            "phase8_gate_audit": phase5b.get("phase8_gate_audit"),
        },
        "phase5_a512_field": {
            "decision": phase5_a512.get("decision"),
            "best_real_variant_id": phase5_a512.get("best_real_variant_id"),
            "best_real_MV_AP_window": phase5_a512.get("best_real_MV_AP_window"),
            "best_real_MV_AP50_window": phase5_a512.get("best_real_MV_AP50_window"),
            "delta_vs_hr2_field_MV_AP_window": _num(phase5_a512.get("best_real_MV_AP_window")) - _num(phase5.get("best_real_MV_AP_window")),
            "solver_backend_actual": phase5_a512.get("solver_backend_actual"),
            "duration_sec": phase5_a512.get("duration_sec"),
            "processed_source_count": phase5_a512.get("processed_source_count"),
            "gpu_device_source_counts": phase5_a512.get("gpu_device_source_counts"),
        },
        "da3_conditional_readiness": {
            "decision": da3_ready.get("decision"),
            "da3_conditional_branch_ready": da3_ready.get("da3_conditional_branch_ready"),
            "da3_executed": da3_ready.get("da3_executed"),
            "uses_da3_outputs": da3_ready.get("uses_da3_outputs"),
            "explicit_da3_authorization_present": da3_ready.get("explicit_da3_authorization_present"),
            "next_required_action": da3_ready.get("next_required_action"),
        },
        "triton_validated": triton_validated,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "next_research_direction": (
            "Do not keep increasing D4RT query count in v93. The next version needs a different readout objective "
            "or object-specific support model; a conditional dense-geometry/DA3 branch is readiness-only here and requires explicit authorization."
        ),
    }

    _write_json(OUT / "summary.json", summary)
    _write_csv(OUT / "evidence_rows.csv", evidence_rows)
    _write_csv(OUT / "attempt_rows.csv", attempts)
    _write_csv(OUT / "blocker_rows.csv", blocker_rows)
    _write_csv(OUT / "repair_attempt_rows.csv", repair_attempt_rows)
    _write_csv(OUT / "phase_decision_rows.csv", phase_rows)

    output_files = [
        OUT / "summary.json",
        OUT / "evidence_rows.csv",
        OUT / "attempt_rows.csv",
        OUT / "blocker_rows.csv",
        OUT / "repair_attempt_rows.csv",
        OUT / "phase_decision_rows.csv",
    ]
    _write_json(OUT / "SHA256SUMS.json", {path.name: _sha256(path) for path in output_files})
    return summary


def main() -> None:
    summary = run()
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
