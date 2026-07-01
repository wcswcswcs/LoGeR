#!/usr/bin/env python3
"""Write the v93 DA3 conditional-readiness audit without running DA3."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/audit/v93_da3_conditional_readiness"


def _read_json(rel: str) -> dict[str, Any]:
    path = ROOT / rel
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


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


def _artifact_exists(rel: str) -> bool:
    return (ROOT / rel).exists()


def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)

    phase0_rel = "outputs/audit/v93_phase0_contract/summary.json"
    phase2_rel = "outputs/audit/v93_phase2_d4rt_edge_sampling_diagnostic/summary.json"
    phase5_rel = "outputs/audit/v93_phase5_boundary_affinity_field/summary.json"
    phase5b_rel = "outputs/audit/v93_phase5b_unknown_background_field/summary.json"
    phase5_a512_rel = "outputs/audit/v93_phase5_boundary_affinity_field_A512/summary.json"
    phase7_rel = "outputs/audit/v93_phase7_adaptive_d4rt_sampling/summary.json"
    phase7_gap_rel = "outputs/audit/v93_phase7_density_readout_gap/summary.json"
    final_rel = "outputs/audit/v93_final_decision/summary.json"

    phase0 = _read_json(phase0_rel)
    phase2 = _read_json(phase2_rel)
    phase5 = _read_json(phase5_rel)
    phase5b = _read_json(phase5b_rel)
    phase5_a512 = _read_json(phase5_a512_rel)
    phase7 = _read_json(phase7_rel)
    phase7_gap = _read_json(phase7_gap_rel)
    final = _read_json(final_rel)

    phase5_family_failed = (
        not _bool(phase5.get("any_phase5_dev_gate_pass"))
        and not _bool(phase5b.get("any_phase5b_dev_gate_pass"))
        and not _bool(phase5_a512.get("any_phase5_dev_gate_pass"))
    )
    phase7_failed_after_density = _bool(phase7.get("density_improved")) and not _bool(phase7.get("uniform_gain_gate_pass"))
    phase7_runtime_failed = not _bool(phase7.get("runtime_budget_pass"))
    phase8_not_warranted = not _bool((phase5b.get("phase8_gate_audit") or {}).get("phase8_enter_gate_pass"))
    dev_gate_failed = not _bool(final.get("dev_gate_pass"))
    holdout_blocked = not _bool(final.get("holdout_executed"))

    condition_rows = [
        {
            "condition": "phase2_d4rt_geometry_quality_problem_observed",
            "status": "true" if (_bool(phase2.get("jitter_high")) or _bool(phase2.get("membership_flip_high"))) else "false",
            "value": {
                "projection_jitter_p90": phase2.get("projection_jitter_p90"),
                "jitter_high": phase2.get("jitter_high"),
                "mask_membership_flip_rate_median": phase2.get("mask_membership_flip_rate_median"),
                "membership_flip_high": phase2.get("membership_flip_high"),
            },
            "source_artifact": _rel(ROOT / phase2_rel),
            "interpretation": "D4RT evidence is useful as material witness but not a reliable pixel boundary oracle.",
        },
        {
            "condition": "adaptive_d4rt_sampling_exhausted_for_v93",
            "status": "true" if (phase7_failed_after_density and phase7_runtime_failed) else "false",
            "value": {
                "density_improved": phase7.get("density_improved"),
                "A512_minus_G16_MV_AP_window": phase7.get("A512_minus_G16_MV_AP_window"),
                "A512_minus_G16_MV_AP50_window": phase7.get("A512_minus_G16_MV_AP50_window"),
                "runtime_budget_pass": phase7.get("runtime_budget_pass"),
                "stop_rule": phase7.get("stop_rule"),
            },
            "source_artifact": _rel(ROOT / phase7_rel),
            "interpretation": "A512 improved support density but did not deliver AP gain and exceeded the v93 runtime budget.",
        },
        {
            "condition": "density_readout_gap_localized",
            "status": "true" if _artifact_exists(phase7_gap_rel) else "false",
            "value": {
                "delta_support_density_mean": phase7_gap.get("best_variant_delta_support_density_mean"),
                "delta_generated_to_source_area_mean": phase7_gap.get("best_variant_delta_generated_to_source_area_mean"),
                "interpretation": phase7_gap.get("interpretation"),
            },
            "source_artifact": _rel(ROOT / phase7_gap_rel),
            "interpretation": "More support barely changed generated extent, so the problem is not just missing support rows.",
        },
        {
            "condition": "gpu_triton_field_readout_failed",
            "status": "true" if phase5_family_failed else "false",
            "value": {
                "phase5_hr2_best_MV_AP_window": phase5.get("best_real_MV_AP_window"),
                "phase5_a512_best_MV_AP_window": phase5_a512.get("best_real_MV_AP_window"),
                "phase5b_best_MV_AP_window": phase5b.get("best_real_MV_AP_window"),
                "required_MV_AP_window": final.get("required_MV_AP_window"),
            },
            "source_artifact": f"{_rel(ROOT / phase5_rel)} ; {_rel(ROOT / phase5_a512_rel)} ; {_rel(ROOT / phase5b_rel)}",
            "interpretation": "Current D4RT/RADIO/edge unary-pairwise and unknown/background objectives did not clear dev gates.",
        },
        {
            "condition": "phase8_score_tuning_not_warranted",
            "status": "true" if phase8_not_warranted else "false",
            "value": phase5b.get("phase8_gate_audit"),
            "source_artifact": _rel(ROOT / phase5b_rel),
            "interpretation": "The score-free AP50 gap did not clear the plan threshold for score calibration.",
        },
        {
            "condition": "dev_gate_failed_holdout_blocked",
            "status": "true" if (dev_gate_failed and holdout_blocked) else "false",
            "value": {
                "dev_gate_pass": final.get("dev_gate_pass"),
                "holdout_executed": final.get("holdout_executed"),
                "best_attempt": final.get("best_attempt"),
            },
            "source_artifact": _rel(ROOT / final_rel),
            "interpretation": "No Phase9 freeze or Phase10 holdout is allowed without dev local/control gate pass.",
        },
    ]

    evidence_rows = [
        {
            "evidence_id": "locked_required_gate",
            "metric": "required_MV_AP_window / required_MV_AP50_window",
            "value": {
                "required_MV_AP_window": final.get("required_MV_AP_window"),
                "required_MV_AP50_window": final.get("required_MV_AP50_window"),
                "v91_best_MV_AP_window": phase0.get("v91_best_MV_AP_window"),
                "locked_best_control_MV_AP_window": phase0.get("best_control_MV_AP_window"),
            },
            "source_artifact": _rel(ROOT / phase0_rel),
        },
        {
            "evidence_id": "best_v93_attempt_still_below_gate",
            "metric": "best_attempt",
            "value": final.get("best_attempt"),
            "source_artifact": _rel(ROOT / final_rel),
        },
        {
            "evidence_id": "phase7_density_without_ap",
            "metric": "A512_minus_G16_MV_AP_window",
            "value": phase7.get("A512_minus_G16_MV_AP_window"),
            "source_artifact": _rel(ROOT / phase7_rel),
        },
        {
            "evidence_id": "phase7_runtime_over_budget",
            "metric": "runtime_budget_pass",
            "value": phase7.get("runtime_budget_pass"),
            "source_artifact": _rel(ROOT / phase7_rel),
        },
        {
            "evidence_id": "phase5_a512_field_regression",
            "metric": "phase5_a512_minus_hr2_MV_AP_window",
            "value": _num(phase5_a512.get("best_real_MV_AP_window")) - _num(phase5.get("best_real_MV_AP_window")),
            "source_artifact": _rel(ROOT / phase5_a512_rel),
        },
        {
            "evidence_id": "phase5b_unknown_background_no_gain",
            "metric": "best_real_MV_AP_window",
            "value": phase5b.get("best_real_MV_AP_window"),
            "source_artifact": _rel(ROOT / phase5b_rel),
        },
    ]

    blocked_action_rows = [
        {
            "action": "run_DA3_in_v93",
            "status": "blocked_not_executed",
            "reason": "The v93 plan forbids introducing DA3 as default and requires explicit authorization before running DA3.",
            "required_before_action": "new approved plan or explicit user authorization for DA3 branch",
        },
        {
            "action": "run_A1024_or_G32_more_sampling",
            "status": "stopped_by_plan_rule",
            "reason": "A512 already improved density without AP gain and exceeded the <=2x runtime budget.",
            "required_before_action": "new evidence that query-count escalation can change readout behavior within budget",
        },
        {
            "action": "run_phase8_score_tuning",
            "status": "blocked_by_gate",
            "reason": "ScoreFreeMatch50 - AP50 gap did not reach the plan threshold of 0.10.",
            "required_before_action": "score-free Match50 headroom or rank-correlation evidence",
        },
        {
            "action": "run_holdout",
            "status": "blocked_by_gate",
            "reason": "No dev strong local candidate exists and no frozen config was written.",
            "required_before_action": "dev local/control margin pass and Phase9 frozen config",
        },
    ]

    readiness_conditions_pass = all(row["status"] == "true" for row in condition_rows)
    summary = {
        "schema": "stream4d_v93_da3_conditional_readiness_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "decision": "DA3_CONDITIONAL_BRANCH_READY_NOT_EXECUTED",
        "goal_achieved": False,
        "da3_conditional_branch_ready": readiness_conditions_pass,
        "da3_executed": False,
        "uses_da3_outputs": False,
        "explicit_da3_authorization_present": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "dev_gate_pass": final.get("dev_gate_pass"),
        "holdout_executed": final.get("holdout_executed"),
        "readiness_reason": (
            "v93 exhausted D4RT density, GPU/Triton field, A512-support field, and unknown/background readout repairs "
            "without dev gate pass. The remaining dense-geometry path is conditional only and was not run."
        ),
        "next_required_action": "Do not run DA3 inside v93 without explicit authorization and a strict-control branch plan.",
        "condition_status_counts": {
            "true": sum(1 for row in condition_rows if row["status"] == "true"),
            "false": sum(1 for row in condition_rows if row["status"] == "false"),
        },
    }

    _write_json(OUT / "summary.json", summary)
    _write_csv(OUT / "condition_rows.csv", condition_rows)
    _write_csv(OUT / "evidence_rows.csv", evidence_rows)
    _write_csv(OUT / "blocked_action_rows.csv", blocked_action_rows)

    output_files = [
        OUT / "summary.json",
        OUT / "condition_rows.csv",
        OUT / "evidence_rows.csv",
        OUT / "blocked_action_rows.csv",
    ]
    _write_json(OUT / "SHA256SUMS.json", {path.name: _sha256(path) for path in output_files})
    return summary


def main() -> None:
    summary = run()
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
