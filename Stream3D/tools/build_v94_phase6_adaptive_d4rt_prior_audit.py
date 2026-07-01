#!/usr/bin/env python3
"""Build v94 Phase6 adaptive D4RT prior-evidence audit."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/audit/v94_phase6_adaptive_d4rt_prior_audit"
PHASE_ID = "v94_phase6_adaptive_d4rt_prior_audit"
RUN_ID = "v94_phase6_adaptive_d4rt_prior_audit"

PHASE5 = ROOT / "outputs/audit/v94_phase5_failure_decomposition/blocker_summary.json"
V93_SAMPLING = ROOT / "outputs/audit/v93_phase7_adaptive_d4rt_sampling/summary.json"
V93_GAP = ROOT / "outputs/audit/v93_phase7_density_readout_gap/summary.json"
V93_A512_FIELD = ROOT / "outputs/audit/v93_phase5_boundary_affinity_field_A512/summary.json"
V93_A512_READOUT = ROOT / "outputs/audit/v93_phase7_A512_same_readout_adaptive_materialization/summary.json"
V94_HR2_EDGE_REPAIR = ROOT / "outputs/audit/v94_phase3A_greedy_assignment_edge_repair"
V94_A512_EDGE_REPAIR = ROOT / "outputs/audit/v94_phase6_A512_edge_repair"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _metric_row(root: Path, variant_id: str) -> dict[str, str]:
    for row in _read_csv(root / "variant_metric_rows.csv"):
        if row.get("variant_id") == variant_id:
            return row
    return {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


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


def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    phase5 = _read_json(PHASE5)
    sampling = _read_json(V93_SAMPLING)
    gap = _read_json(V93_GAP)
    a512_field = _read_json(V93_A512_FIELD)
    a512_readout = _read_json(V93_A512_READOUT)
    v94_hr2_summary = _read_json(V94_HR2_EDGE_REPAIR / "summary.json")
    v94_a512_summary = _read_json(V94_A512_EDGE_REPAIR / "summary.json")
    hr2_a10 = _metric_row(V94_HR2_EDGE_REPAIR, "A10_relaxed_no_edge_source_preserve")
    a512_a10 = _metric_row(V94_A512_EDGE_REPAIR, "A10_relaxed_no_edge_source_preserve")
    v94_a512_delta_ap = _num(a512_a10.get("mean_MV_AP_window")) - _num(hr2_a10.get("mean_MV_AP_window"))
    v94_a512_delta_ap50 = _num(a512_a10.get("mean_MV_AP50_window")) - _num(hr2_a10.get("mean_MV_AP50_window"))
    evidence_rows = [
        {
            "schema_version": "stream4d_v94_phase6_adaptive_prior_evidence_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "evidence_id": "v93_phase7_adaptive_d4rt_sampling",
            "artifact": _rel(V93_SAMPLING),
            "density_improved": sampling.get("density_improved", ""),
            "A512_minus_G16_MV_AP_window": sampling.get("A512_minus_G16_MV_AP_window", ""),
            "A512_minus_G16_MV_AP50_window": sampling.get("A512_minus_G16_MV_AP50_window", ""),
            "A512_minus_locked_control_MV_AP_window": sampling.get("A512_minus_locked_control_MV_AP_window", ""),
            "A512_minus_locked_control_MV_AP50_window": sampling.get("A512_minus_locked_control_MV_AP50_window", ""),
            "runtime_budget_pass": sampling.get("runtime_budget_pass", ""),
            "uniform_gain_gate_pass": sampling.get("uniform_gain_gate_pass", ""),
            "control_gate_pass": sampling.get("control_gate_pass", ""),
            "decision": sampling.get("decision", ""),
        },
        {
            "schema_version": "stream4d_v94_phase6_adaptive_prior_evidence_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "evidence_id": "v93_phase7_density_readout_gap",
            "artifact": _rel(V93_GAP),
            "support_density_delta_mean": gap.get("best_variant_delta_support_density_mean", ""),
            "support_count_delta_mean": gap.get("best_variant_delta_support_count_mean", ""),
            "generated_to_source_area_delta_mean": gap.get("best_variant_delta_generated_to_source_area_mean", ""),
            "interpretation": gap.get("interpretation", ""),
            "decision": gap.get("decision", ""),
        },
        {
            "schema_version": "stream4d_v94_phase6_adaptive_prior_evidence_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "evidence_id": "v93_phase5_A512_same_field_readout",
            "artifact": _rel(V93_A512_FIELD),
            "best_real_MV_AP_window": a512_field.get("best_real_MV_AP_window", ""),
            "best_real_MV_AP50_window": a512_field.get("best_real_MV_AP50_window", ""),
            "solver_backend_actual": a512_field.get("solver_backend_actual", ""),
            "any_phase5_dev_gate_pass": a512_field.get("any_phase5_dev_gate_pass", ""),
            "duration_sec": a512_field.get("duration_sec", ""),
            "decision": a512_field.get("decision", ""),
        },
        {
            "schema_version": "stream4d_v94_phase6_adaptive_prior_evidence_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "evidence_id": "v93_A512_same_readout_materialization",
            "artifact": _rel(V93_A512_READOUT),
            "best_MV_AP_window": a512_readout.get("best_variant_gate", {}).get("mean_MV_AP_window", ""),
            "best_MV_AP50_window": a512_readout.get("best_variant_gate", {}).get("mean_MV_AP50_window", ""),
            "runtime_sec": a512_readout.get("runtime_sec", ""),
            "decision": a512_readout.get("decision", ""),
        },
        {
            "schema_version": "stream4d_v94_phase6_adaptive_prior_evidence_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "evidence_id": "v94_A512_edge_repair_readout",
            "artifact": _rel(V94_A512_EDGE_REPAIR / "summary.json"),
            "baseline_artifact": _rel(V94_HR2_EDGE_REPAIR / "summary.json"),
            "variant_id": "A10_relaxed_no_edge_source_preserve",
            "hr2_MV_AP_window": hr2_a10.get("mean_MV_AP_window", ""),
            "hr2_MV_AP50_window": hr2_a10.get("mean_MV_AP50_window", ""),
            "A512_MV_AP_window": a512_a10.get("mean_MV_AP_window", ""),
            "A512_MV_AP50_window": a512_a10.get("mean_MV_AP50_window", ""),
            "A512_minus_HR2_MV_AP_window": v94_a512_delta_ap,
            "A512_minus_HR2_MV_AP50_window": v94_a512_delta_ap50,
            "field_shard_count": v94_a512_summary.get("field_shard_count", ""),
            "processed_source_count": v94_a512_summary.get("processed_source_count", ""),
            "gpu_device_source_counts": json.dumps(v94_a512_summary.get("gpu_device_source_counts", {}), sort_keys=True),
            "duration_sec": v94_a512_summary.get("duration_sec", ""),
            "source_failure_rows": v94_a512_summary.get("row_counts", {}).get("source_failure_rows", ""),
            "decision": v94_a512_summary.get("decision", ""),
        },
    ]
    v94_a512_gain_gate = v94_a512_delta_ap >= 0.005 and v94_a512_delta_ap50 >= 0.010
    summary = {
        "schema": "stream4d_v94_phase6_adaptive_d4rt_prior_audit_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": "NO_GO_V94_PHASE6_ADAPTIVE_D4RT_V94_READOUT_EVIDENCE",
        "phase5_d4rt_blocker": phase5.get("blockers", {}).get("D4RT_BLOCKER", ""),
        "new_adaptive_sampling_executed": False,
        "new_v94_A512_readout_executed": bool(a512_a10),
        "prior_evidence_reused": True,
        "A512_minus_G16_MV_AP_window": sampling.get("A512_minus_G16_MV_AP_window", ""),
        "A512_minus_G16_MV_AP50_window": sampling.get("A512_minus_G16_MV_AP50_window", ""),
        "required_MV_AP_gain": 0.005,
        "required_MV_AP50_gain": 0.010,
        "density_improved": sampling.get("density_improved", ""),
        "runtime_budget_pass": sampling.get("runtime_budget_pass", ""),
        "uniform_gain_gate_pass": sampling.get("uniform_gain_gate_pass", ""),
        "control_gate_pass": sampling.get("control_gate_pass", ""),
        "v94_A512_best_variant_id": "A10_relaxed_no_edge_source_preserve" if a512_a10 else "",
        "v94_A512_MV_AP_window": a512_a10.get("mean_MV_AP_window", ""),
        "v94_A512_MV_AP50_window": a512_a10.get("mean_MV_AP50_window", ""),
        "v94_HR2_reference_MV_AP_window": hr2_a10.get("mean_MV_AP_window", ""),
        "v94_HR2_reference_MV_AP50_window": hr2_a10.get("mean_MV_AP50_window", ""),
        "v94_A512_minus_HR2_MV_AP_window": v94_a512_delta_ap,
        "v94_A512_minus_HR2_MV_AP50_window": v94_a512_delta_ap50,
        "v94_A512_gain_gate_pass": v94_a512_gain_gate,
        "v94_A512_processed_source_count": v94_a512_summary.get("processed_source_count", ""),
        "v94_A512_gpu_device_source_counts": v94_a512_summary.get("gpu_device_source_counts", {}),
        "v94_A512_duration_sec": v94_a512_summary.get("duration_sec", ""),
        "stop_rule": "Same-domain A512 adaptive evidence improves density but fails MV_AP/control/runtime in prior artifacts, and the new v94 A512 edge-repair readout is slightly worse than the HR2 reference. Do not continue D4RT density/sampling inside v94 without a different readout or explicit new dense-geometry plan.",
        "holdout_executed": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_csv(OUT / "adaptive_prior_evidence_rows.csv", evidence_rows)
    _write_json(OUT / "summary.json", summary)
    outputs = [OUT / "adaptive_prior_evidence_rows.csv", OUT / "summary.json"]
    _write_json(OUT / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs if path.exists()})
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run()
