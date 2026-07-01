#!/usr/bin/env python3
"""Build v94 Phase4 control-suite attribution from locked/evaluated artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/audit/v94_phase4_controls"
PHASE_ID = "v94_phase4_controls"
RUN_ID = "v94_phase4_controls"

PHASE3_ROOTS = [
    ("phase3A_main", ROOT / "outputs/audit/v94_phase3A_greedy_assignment"),
    ("phase3A_edge_repair", ROOT / "outputs/audit/v94_phase3A_greedy_assignment_edge_repair"),
    ("phase3B_random_walker", ROOT / "outputs/audit/v94_phase3B_random_walker"),
    ("phase3C_constrained_cut", ROOT / "outputs/audit/v94_phase3C_constrained_cut"),
    ("phase3D_component_pooling", ROOT / "outputs/audit/v94_phase3D_component_pooling"),
]

CONTROL_SPECS = [
    (
        "CTRL0_whole_source",
        [
            (ROOT / "outputs/audit/v94_phase3A_greedy_assignment_edge_repair/variant_metric_rows.csv", "A0_current_whole_source_replay"),
            (ROOT / "outputs/audit/v93_phase4_cue_isolation/variant_metric_rows.csv", "E0_whole_source"),
        ],
        "legal_control",
    ),
    (
        "CTRL1_semantic_only",
        [(ROOT / "outputs/audit/v94_phase0_fact_lock/baseline_metric_rows.csv", "C0_semantic_only_control")],
        "locked_control",
    ),
    (
        "CTRL2_area_semantic_hybrid",
        [(ROOT / "outputs/audit/v94_phase0_fact_lock/baseline_metric_rows.csv", "P3_C0_area_semantic_hybrid_score")],
        "locked_control",
    ),
    (
        "CTRL3_edge_only_outer",
        [(ROOT / "outputs/audit/v93_phase4_edge_only_materialization/variant_metric_rows.csv", "E1_outer_edge_only")],
        "locked_control",
    ),
    (
        "CTRL4_edge_only_nested",
        [(ROOT / "outputs/audit/v93_phase4_edge_only_materialization/variant_metric_rows.csv", "E2_nested_overlap_edge")],
        "locked_control",
    ),
    (
        "CTRL5_random_edge_barrier",
        [(ROOT / "outputs/audit/v93_phase4_edge_only_materialization/variant_metric_rows.csv", "R0_random_edge_control")],
        "locked_control",
    ),
    (
        "CTRL6_shuffled_edge_barrier",
        [(ROOT / "outputs/audit/v93_phase4_edge_only_materialization/variant_metric_rows.csv", "R1_shuffled_edge_control")],
        "locked_control",
    ),
    (
        "CTRL7_D4RT_only",
        [
            (ROOT / "outputs/audit/v93_phase4_cue_isolation/variant_metric_rows.csv", "D0_D4RT_witness_only"),
            (ROOT / "outputs/audit/v92_phase5_source_container_field/control_metric_rows.csv", "V92_C7_d4rt_only_control"),
        ],
        "locked_control",
    ),
    (
        "CTRL8_shuffled_D4RT_witness",
        [(ROOT / "outputs/audit/v94_phase3A_greedy_assignment_d4rt_control/variant_metric_rows.csv", "CTRL8_shuffled_D4RT_witness")],
        "v94_measured_control",
    ),
    (
        "CTRL9_RADIO_only",
        [
            (ROOT / "outputs/audit/v93_phase4_cue_isolation/variant_metric_rows.csv", "S0_RADIO_region_only"),
            (ROOT / "outputs/audit/v92_phase5_source_container_field/control_metric_rows.csv", "V92_C8_radio_only_control"),
        ],
        "locked_control",
    ),
    (
        "CTRL10_random_region_seed",
        [
            (ROOT / "outputs/audit/v93_phase4_cue_isolation/variant_metric_rows.csv", "R0_random_region_seed_control"),
            (ROOT / "outputs/audit/v92_phase5_source_container_field/control_metric_rows.csv", "V92_C5_random_region_seed_control"),
        ],
        "locked_control",
    ),
    (
        "CTRL11_single_largest",
        [(ROOT / "outputs/audit/v90_phase1_variant_resurrection/mv_metric_aggregate_rows.csv", "C4_single_largest_by_scene_control")],
        "historical_control",
    ),
    (
        "CTRL12_oracle_diagnostic_only_not_method",
        [(ROOT / "outputs/audit/v91_source_mask_oracle_upper_bound/summary.json", "OR3_source_gt_intersection_upper")],
        "diagnostic_only_gt_oracle",
    ),
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fieldnames})


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _bool_false(value: Any) -> bool:
    return str(value).strip().lower() in ("", "0", "false", "none")


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _metric(row: dict[str, Any], name: str) -> float:
    candidates = [
        f"mean_{name}",
        name,
        f"{name}_mean",
    ]
    for key in candidates:
        if key in row and str(row.get(key, "")) != "":
            return _num(row.get(key))
    return 0.0


def _variant_id(row: dict[str, Any]) -> str:
    return str(row.get("variant_id") or row.get("control_id") or row.get("method_id") or "")


def _standard_row(
    *,
    schema_version: str,
    row_kind: str,
    source_artifact: Path,
    source_phase: str,
    variant_id: str,
    row: dict[str, Any],
    family: str,
    measurement_status: str,
    created_at: str,
    control_id: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "row_kind": row_kind,
        "control_id": control_id,
        "variant_id": variant_id,
        "family": family,
        "measurement_status": measurement_status,
        "source_phase": source_phase,
        "source_artifact": _rel(source_artifact),
        "created_at": created_at,
        "scene_count": _metric(row, "scene_count"),
        "MV_AP_window": _metric(row, "MV_AP_window"),
        "MV_AP50_window": _metric(row, "MV_AP50_window"),
        "MV_AP25_window": _metric(row, "MV_AP25_window"),
        "ScoreFreeMatch50_window": _metric(row, "score_free_Match50_window")
        or _metric(row, "ScoreFreeMatch50_window"),
        "gt_object_count": _metric(row, "gt_object_count"),
        "pred_object_count": _metric(row, "pred_object_count"),
        "same_frame_collision_count": _metric(row, "same_frame_collision_count"),
        "missing_mask_raster_count": _metric(row, "missing_mask_raster_count"),
        "uses_gt_for_prediction": row.get("uses_gt_for_prediction", row.get("uses_gt_for_prediction_count", False)),
        "uses_future": row.get("uses_future", row.get("uses_future_count", False)),
        "diagnostic_only": family == "oracle_diagnostic",
    }


def _find_csv_variant(path: Path, variant_id: str) -> dict[str, Any]:
    for row in _read_csv(path):
        if _variant_id(row) == variant_id:
            return dict(row)
    return {}


def _find_control(control_id: str, candidates: list[tuple[Path, str]], status: str, created_at: str) -> dict[str, Any]:
    if status == "missing_control":
        return {
            "schema_version": "stream4d_v94_phase4_control_metric_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "row_kind": "control",
            "control_id": control_id,
            "variant_id": "",
            "family": "missing_control",
            "measurement_status": "missing",
            "source_phase": "",
            "source_artifact": "",
            "created_at": created_at,
            "MV_AP_window": "",
            "MV_AP50_window": "",
            "MV_AP25_window": "",
            "ScoreFreeMatch50_window": "",
            "same_frame_collision_count": "",
            "missing_mask_raster_count": "",
            "uses_gt_for_prediction": "",
            "uses_future": "",
            "diagnostic_only": False,
            "missing_reason": "No same-support shuffled D4RT witness control artifact was found; left missing rather than substituting a different random-control family.",
        }
    for path, variant_id in candidates:
        if path.suffix == ".json":
            payload = _read_json(path)
            row = payload.get("best_source_oracle_constant", {}) if variant_id == "OR3_source_gt_intersection_upper" else {}
        else:
            row = _find_csv_variant(path, variant_id)
        if row:
            family = "oracle_diagnostic" if status == "diagnostic_only_gt_oracle" else "control"
            return _standard_row(
                schema_version="stream4d_v94_phase4_control_metric_v1",
                row_kind="control",
                source_artifact=path,
                source_phase=path.parent.name,
                variant_id=variant_id,
                row=row,
                family=family,
                measurement_status=status,
                created_at=created_at,
                control_id=control_id,
            )
    return {
        "schema_version": "stream4d_v94_phase4_control_metric_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "row_kind": "control",
        "control_id": control_id,
        "variant_id": "",
        "family": "missing_control",
        "measurement_status": "missing",
        "source_phase": "",
        "source_artifact": "",
        "created_at": created_at,
        "missing_reason": "Configured source artifacts were absent or did not contain the requested variant_id.",
    }


def _collect_variant_rows(created_at: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase_name, root in PHASE3_ROOTS:
        path = root / "variant_metric_rows.csv"
        for row in _read_csv(path):
            variant_id = _variant_id(row)
            family = "baseline" if variant_id.endswith("whole_source_replay") else "real"
            rows.append(
                _standard_row(
                    schema_version="stream4d_v94_phase4_variant_metric_v1",
                    row_kind="variant",
                    source_artifact=path,
                    source_phase=phase_name,
                    variant_id=variant_id,
                    row=row,
                    family=family,
                    measurement_status="v94_measured",
                    created_at=created_at,
                )
            )
    rows.sort(key=lambda row: (row["family"] == "real", _num(row.get("MV_AP_window")), _num(row.get("MV_AP50_window"))), reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    created_at = _created_at()
    variant_rows = _collect_variant_rows(created_at)
    control_rows = [_find_control(control_id, candidates, status, created_at) for control_id, candidates, status in CONTROL_SPECS]

    real_rows = [row for row in variant_rows if row.get("family") == "real"]
    legal_controls = [
        row
        for row in control_rows
        if row.get("family") == "control" and str(row.get("measurement_status")) != "missing"
    ]
    non_oracle_controls = legal_controls
    best_real = max(real_rows, key=lambda row: (_num(row.get("MV_AP_window")), _num(row.get("MV_AP50_window"))), default={})
    best_control = max(non_oracle_controls, key=lambda row: (_num(row.get("MV_AP_window")), _num(row.get("MV_AP50_window"))), default={})

    def ctrl(control_id: str) -> dict[str, Any]:
        return next((row for row in control_rows if row.get("control_id") == control_id), {})

    random_edge = ctrl("CTRL5_random_edge_barrier")
    shuffled_d4rt = ctrl("CTRL8_shuffled_D4RT_witness")
    d4rt_only = ctrl("CTRL7_D4RT_only")
    radio_only = ctrl("CTRL9_RADIO_only")
    edge_only = ctrl("CTRL3_edge_only_outer")

    shuffled_d4rt_available = str(shuffled_d4rt.get("measurement_status")) != "missing"
    attribution_rows = [
        {
            "schema_version": "stream4d_v94_phase4_attribution_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "comparison": "real_minus_best_control",
            "best_real_variant_id": best_real.get("variant_id", ""),
            "control_id": best_control.get("control_id", ""),
            "control_variant_id": best_control.get("variant_id", ""),
            "delta_MV_AP_window": _num(best_real.get("MV_AP_window")) - _num(best_control.get("MV_AP_window")),
            "delta_MV_AP50_window": _num(best_real.get("MV_AP50_window")) - _num(best_control.get("MV_AP50_window")),
            "gate_MV_AP_window": (_num(best_real.get("MV_AP_window")) - _num(best_control.get("MV_AP_window"))) >= 0.010,
            "gate_MV_AP50_window": (_num(best_real.get("MV_AP50_window")) - _num(best_control.get("MV_AP50_window"))) >= 0.015,
        },
        {
            "schema_version": "stream4d_v94_phase4_attribution_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "comparison": "real_minus_random_edge",
            "best_real_variant_id": best_real.get("variant_id", ""),
            "control_id": random_edge.get("control_id", ""),
            "control_variant_id": random_edge.get("variant_id", ""),
            "delta_MV_AP_window": _num(best_real.get("MV_AP_window")) - _num(random_edge.get("MV_AP_window")),
            "delta_MV_AP50_window": _num(best_real.get("MV_AP50_window")) - _num(random_edge.get("MV_AP50_window")),
            "gate_MV_AP_window": (_num(best_real.get("MV_AP_window")) - _num(random_edge.get("MV_AP_window"))) >= 0.010,
        },
        {
            "schema_version": "stream4d_v94_phase4_attribution_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "comparison": "real_minus_shuffled_D4RT",
            "best_real_variant_id": best_real.get("variant_id", ""),
            "control_id": shuffled_d4rt.get("control_id", ""),
            "control_variant_id": shuffled_d4rt.get("variant_id", ""),
            "delta_MV_AP_window": "" if not shuffled_d4rt_available else _num(best_real.get("MV_AP_window")) - _num(shuffled_d4rt.get("MV_AP_window")),
            "delta_MV_AP50_window": "" if not shuffled_d4rt_available else _num(best_real.get("MV_AP50_window")) - _num(shuffled_d4rt.get("MV_AP50_window")),
            "gate_MV_AP_window": False if not shuffled_d4rt_available else (_num(best_real.get("MV_AP_window")) - _num(shuffled_d4rt.get("MV_AP_window"))) >= 0.010,
            "available": shuffled_d4rt_available,
        },
        {
            "schema_version": "stream4d_v94_phase4_attribution_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "comparison": "real_minus_D4RT_only",
            "best_real_variant_id": best_real.get("variant_id", ""),
            "control_id": d4rt_only.get("control_id", ""),
            "control_variant_id": d4rt_only.get("variant_id", ""),
            "delta_MV_AP_window": _num(best_real.get("MV_AP_window")) - _num(d4rt_only.get("MV_AP_window")),
            "delta_MV_AP50_window": _num(best_real.get("MV_AP50_window")) - _num(d4rt_only.get("MV_AP50_window")),
            "gate_positive": _num(best_real.get("MV_AP_window")) > _num(d4rt_only.get("MV_AP_window")),
        },
        {
            "schema_version": "stream4d_v94_phase4_attribution_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "comparison": "real_minus_RADIO_only",
            "best_real_variant_id": best_real.get("variant_id", ""),
            "control_id": radio_only.get("control_id", ""),
            "control_variant_id": radio_only.get("variant_id", ""),
            "delta_MV_AP_window": _num(best_real.get("MV_AP_window")) - _num(radio_only.get("MV_AP_window")),
            "delta_MV_AP50_window": _num(best_real.get("MV_AP50_window")) - _num(radio_only.get("MV_AP50_window")),
            "gate_positive": _num(best_real.get("MV_AP_window")) > _num(radio_only.get("MV_AP_window")),
        },
        {
            "schema_version": "stream4d_v94_phase4_attribution_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "comparison": "real_minus_edge_only",
            "best_real_variant_id": best_real.get("variant_id", ""),
            "control_id": edge_only.get("control_id", ""),
            "control_variant_id": edge_only.get("variant_id", ""),
            "delta_MV_AP_window": _num(best_real.get("MV_AP_window")) - _num(edge_only.get("MV_AP_window")),
            "delta_MV_AP50_window": _num(best_real.get("MV_AP50_window")) - _num(edge_only.get("MV_AP50_window")),
            "gate_positive": _num(best_real.get("MV_AP_window")) > _num(edge_only.get("MV_AP_window")),
        },
    ]

    failure_rows: list[dict[str, Any]] = []
    if best_control and _num(best_real.get("MV_AP_window")) < _num(best_control.get("MV_AP_window")) + 0.010:
        failure_rows.append(
            {
                "schema_version": "stream4d_v94_phase4_control_failure_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "failure_type": "CONTROL_BIAS_BLOCKER",
                "evidence": "best real does not exceed best legal control by required MV_AP margin",
                "best_real_variant_id": best_real.get("variant_id", ""),
                "best_control_id": best_control.get("control_id", ""),
                "best_control_variant_id": best_control.get("variant_id", ""),
                "real_minus_best_control_MV_AP_window": _num(best_real.get("MV_AP_window")) - _num(best_control.get("MV_AP_window")),
                "required_delta_MV_AP_window": 0.010,
            }
        )
    if edge_only and _num(best_real.get("MV_AP_window")) <= _num(edge_only.get("MV_AP_window")):
        failure_rows.append(
            {
                "schema_version": "stream4d_v94_phase4_control_failure_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "failure_type": "EDGE_CONTROL_STRONGER_THAN_REAL",
                "evidence": "outer edge-only locked control remains stronger than best real field variant",
                "best_real_variant_id": best_real.get("variant_id", ""),
                "edge_control_variant_id": edge_only.get("variant_id", ""),
                "real_minus_edge_only_MV_AP_window": _num(best_real.get("MV_AP_window")) - _num(edge_only.get("MV_AP_window")),
            }
        )
    if not shuffled_d4rt_available:
        failure_rows.append(
            {
                "schema_version": "stream4d_v94_phase4_control_failure_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "failure_type": "MISSING_SHUFFLED_D4RT_CONTROL",
                "evidence": "No same-support shuffled D4RT witness control artifact found; Phase4 gate cannot claim shuffled-D4RT margin.",
            }
        )

    real_minus_best_control_ap = _num(best_real.get("MV_AP_window")) - _num(best_control.get("MV_AP_window"))
    real_minus_best_control_ap50 = _num(best_real.get("MV_AP50_window")) - _num(best_control.get("MV_AP50_window"))
    real_minus_random_edge_ap = _num(best_real.get("MV_AP_window")) - _num(random_edge.get("MV_AP_window"))
    real_minus_shuffled_d4rt_ap = "" if not shuffled_d4rt_available else _num(best_real.get("MV_AP_window")) - _num(shuffled_d4rt.get("MV_AP_window"))
    provenance_pass = bool(
        best_real
        and _num(best_real.get("same_frame_collision_count")) == 0
        and _num(best_real.get("missing_mask_raster_count")) == 0
        and _bool_false(best_real.get("uses_gt_for_prediction"))
        and _bool_false(best_real.get("uses_future"))
    )
    control_gate_pass = bool(
        best_real
        and best_control
        and real_minus_best_control_ap >= 0.010
        and real_minus_best_control_ap50 >= 0.015
        and real_minus_random_edge_ap >= 0.010
        and shuffled_d4rt_available
        and _num(real_minus_shuffled_d4rt_ap) >= 0.010
        and _num(best_real.get("MV_AP_window")) > _num(d4rt_only.get("MV_AP_window"))
        and _num(best_real.get("MV_AP_window")) > _num(radio_only.get("MV_AP_window"))
        and _num(best_real.get("MV_AP_window")) > _num(edge_only.get("MV_AP_window"))
        and provenance_pass
    )
    summary = {
        "schema": "stream4d_v94_phase4_controls_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": created_at,
        "decision": "PASS_V94_PHASE4_CONTROL_GATE" if control_gate_pass else "NO_GO_V94_PHASE4_CONTROL_ATTRIBUTION",
        "best_real_variant_id": best_real.get("variant_id", ""),
        "best_real_phase": best_real.get("source_phase", ""),
        "best_real_MV_AP_window": _num(best_real.get("MV_AP_window")),
        "best_real_MV_AP50_window": _num(best_real.get("MV_AP50_window")),
        "best_control_id": best_control.get("control_id", ""),
        "best_control_variant_id": best_control.get("variant_id", ""),
        "best_control_MV_AP_window": _num(best_control.get("MV_AP_window")),
        "best_control_MV_AP50_window": _num(best_control.get("MV_AP50_window")),
        "real_minus_best_control_MV_AP_window": real_minus_best_control_ap,
        "real_minus_best_control_MV_AP50_window": real_minus_best_control_ap50,
        "real_minus_D4RT_only_MV_AP_window": _num(best_real.get("MV_AP_window")) - _num(d4rt_only.get("MV_AP_window")),
        "real_minus_RADIO_only_MV_AP_window": _num(best_real.get("MV_AP_window")) - _num(radio_only.get("MV_AP_window")),
        "real_minus_edge_only_MV_AP_window": _num(best_real.get("MV_AP_window")) - _num(edge_only.get("MV_AP_window")),
        "real_minus_random_edge_MV_AP_window": real_minus_random_edge_ap,
        "real_minus_shuffled_D4RT_MV_AP_window": real_minus_shuffled_d4rt_ap,
        "shuffled_D4RT_control_available": shuffled_d4rt_available,
        "same_frame_collision_count": _num(best_real.get("same_frame_collision_count")),
        "missing_mask_raster_count": _num(best_real.get("missing_mask_raster_count")),
        "uses_gt_for_prediction": best_real.get("uses_gt_for_prediction", ""),
        "uses_future": best_real.get("uses_future", ""),
        "provenance_gate_pass": provenance_pass,
        "control_gate_pass": control_gate_pass,
        "diagnostic_oracle_control_included": True,
        "control_missing_count": sum(1 for row in control_rows if row.get("measurement_status") == "missing"),
        "missing_control_ids": [row.get("control_id") for row in control_rows if row.get("measurement_status") == "missing"],
        "row_counts": {
            "variant_metric_rows": len(variant_rows),
            "control_metric_rows": len(control_rows),
            "attribution_rows": len(attribution_rows),
            "control_failure_rows": len(failure_rows),
        },
    }
    _write_csv(OUT / "variant_metric_rows.csv", variant_rows)
    _write_csv(OUT / "control_metric_rows.csv", control_rows)
    _write_csv(OUT / "attribution_rows.csv", attribution_rows)
    _write_csv(OUT / "control_failure_rows.csv", failure_rows)
    _write_json(OUT / "summary.json", summary)
    outputs = [
        OUT / "variant_metric_rows.csv",
        OUT / "control_metric_rows.csv",
        OUT / "attribution_rows.csv",
        OUT / "control_failure_rows.csv",
        OUT / "summary.json",
    ]
    _write_json(OUT / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run()
