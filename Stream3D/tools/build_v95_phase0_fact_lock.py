#!/usr/bin/env python3
"""Build Stream4D v95 Phase0 fact lock from v94/v93 evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/audit/v95_phase0_fact_lock"
PHASE_ID = "v95_phase0_fact_lock"
RUN_ID = "v95_phase0_fact_lock"

V65_EVALUATOR = ROOT / "tools/run_v65_scene_multiview_ap.py"
V93_PHASE0 = ROOT / "outputs/audit/v93_phase0_contract/summary.json"
V94_PHASE0 = ROOT / "outputs/audit/v94_phase0_fact_lock/summary.json"
V94_PHASE4 = ROOT / "outputs/audit/v94_phase4_controls/summary.json"
V94_PHASE5 = ROOT / "outputs/audit/v94_phase5_failure_decomposition/blocker_summary.json"
V94_PHASE7C_SMOKE = ROOT / "outputs/audit/v94_phase7c_object_axis_field_smoke/summary.json"
V94_PHASE7C_0011 = ROOT / "outputs/audit/v94_phase7c_object_axis_field_scene0011_full_dev/summary.json"
V94_PHASE7C_0050 = ROOT / "outputs/audit/v94_phase7c_object_axis_field_scene0050_full_dev/summary.json"
V94_PHASE8 = ROOT / "outputs/audit/v94_phase8_dev_decision/summary.json"
V94_VARIANT_RANK = ROOT / "outputs/audit/v94_phase8_dev_decision/variant_rank_rows.csv"
V94_CANONICAL = ROOT / "outputs/audit/v94_phase1_canonical_graph"


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
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
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _provenance_counts(csv_paths: list[Path], summary_paths: list[Path]) -> dict[str, int]:
    counts = {"uses_gt_for_prediction_count": 0, "uses_future_count": 0}
    for path in csv_paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                counts["uses_gt_for_prediction_count"] += int(_bool(row.get("uses_gt_for_prediction")))
                counts["uses_future_count"] += int(_bool(row.get("uses_future")))
    for path in summary_paths:
        summary = _read_json(path)
        if not summary:
            continue
        counts["uses_gt_for_prediction_count"] += int(_bool(summary.get("uses_gt_for_prediction")))
        counts["uses_future_count"] += int(_bool(summary.get("uses_future")))
    return counts


def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    created_at = _created_at()
    v93_phase0 = _read_json(V93_PHASE0)
    v94_phase0 = _read_json(V94_PHASE0)
    v94_phase4 = _read_json(V94_PHASE4)
    v94_phase5 = _read_json(V94_PHASE5)
    v94_phase7c_smoke = _read_json(V94_PHASE7C_SMOKE)
    v94_phase7c_0011 = _read_json(V94_PHASE7C_0011)
    v94_phase7c_0050 = _read_json(V94_PHASE7C_0050)
    v94_phase8 = _read_json(V94_PHASE8)
    evaluator_text = V65_EVALUATOR.read_text(encoding="utf-8") if V65_EVALUATOR.exists() else ""

    formal_metric_source_eq_v65 = bool(
        V65_EVALUATOR.exists()
        and "SparseSceneIoU" in evaluator_text
        and "_summarize_iou" in evaluator_text
        and bool(v94_phase0.get("formal_metric_source_eq_v65", v93_phase0.get("formal_metric_source_eq_v65")))
    )
    local_support_policy = str(v94_phase0.get("local_support_policy", v93_phase0.get("local_support_policy", "")))
    ap_thresholds = v94_phase0.get("AP_thresholds_actual", v93_phase0.get("AP_thresholds_actual", []))
    required_ap = _num(v94_phase8.get("required_MV_AP_window"), _num(v94_phase0.get("required_MV_AP_window")))
    required_ap50 = _num(v94_phase8.get("required_MV_AP50_window"), _num(v94_phase0.get("required_MV_AP50_window")))
    object_axis_full_dev_pass = bool(
        v94_phase7c_0011.get("object_specific_field_input_gate_pass")
        and v94_phase7c_0050.get("object_specific_field_input_gate_pass")
    )
    provenance = _provenance_counts(
        [
            V94_VARIANT_RANK,
            V94_CANONICAL / "container_rows.csv",
            V94_CANONICAL / "object_hypothesis_rows.csv",
            V94_CANONICAL / "container_object_link_rows.csv",
        ],
        [V94_PHASE8, V94_PHASE5, V94_PHASE7C_SMOKE, V94_PHASE7C_0011, V94_PHASE7C_0050],
    )

    gate_rows = [
        {
            "schema_version": "stream4d_v95_phase0_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "gate": "formal_metric_source_eq_v65",
            "pass": formal_metric_source_eq_v65,
            "observed": formal_metric_source_eq_v65,
            "required": True,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v95_phase0_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "gate": "local_support_policy",
            "pass": local_support_policy == "local_window_gt_projection",
            "observed": local_support_policy,
            "required": "local_window_gt_projection",
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v95_phase0_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "gate": "provenance_counts",
            "pass": provenance["uses_gt_for_prediction_count"] == 0 and provenance["uses_future_count"] == 0,
            **provenance,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v95_phase0_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "gate": "required_metrics_available",
            "pass": required_ap > 0 and required_ap50 > 0,
            "required_MV_AP_window": required_ap,
            "required_MV_AP50_window": required_ap50,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
    ]
    phase0_pass = all(bool(row.get("pass")) for row in gate_rows)

    artifact_paths = [
        V65_EVALUATOR,
        V93_PHASE0,
        V94_PHASE0,
        V94_PHASE4,
        V94_PHASE5,
        V94_PHASE7C_SMOKE,
        V94_PHASE7C_0011,
        V94_PHASE7C_0050,
        V94_PHASE8,
        V94_VARIANT_RANK,
    ]
    artifact_rows = [
        {
            "schema_version": "stream4d_v95_phase0_artifact_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "artifact_path": _rel(path),
            "artifact_exists": path.exists(),
            "artifact_sha256": _sha256(path) if path.exists() and path.is_file() else "",
            "created_at": created_at,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for path in artifact_paths
    ]

    summary = {
        "schema": "stream4d_v95_phase0_fact_lock_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": created_at,
        "decision": "PASS_V95_PHASE0_FACT_LOCK" if phase0_pass else "NO_GO_V95_PHASE0_FACT_LOCK",
        "formal_metric_source_eq_v65": formal_metric_source_eq_v65,
        "formal_metric_source": "tools/run_v65_scene_multiview_ap.py SparseSceneIoU/_summarize_iou",
        "local_support_policy": local_support_policy,
        "AP_thresholds_actual": ap_thresholds,
        "B0_MV_AP_window": _num(v93_phase0.get("B0_MV_AP_window")),
        "B0_MV_AP50_window": _num(v93_phase0.get("B0_MV_AP50_window")),
        "best_control_MV_AP_window": _num(v94_phase4.get("best_control_MV_AP_window"), _num(v94_phase0.get("best_control_MV_AP_window"))),
        "best_control_MV_AP50_window": _num(v94_phase4.get("best_control_MV_AP50_window"), _num(v94_phase0.get("best_control_MV_AP50_window"))),
        "v91_best_MV_AP_window": _num(v94_phase0.get("v91_best_MV_AP_window"), _num(v93_phase0.get("v91_best_MV_AP_window"))),
        "v91_best_MV_AP50_window": _num(v94_phase0.get("v91_best_MV_AP50_window"), _num(v93_phase0.get("v91_best_MV_AP50_window"))),
        "v94_best_MV_AP_window": _num(v94_phase8.get("best_real_MV_AP_window"), _num(v94_phase5.get("best_real_MV_AP_window"))),
        "v94_best_MV_AP50_window": _num(v94_phase8.get("best_real_MV_AP50_window"), _num(v94_phase5.get("best_real_MV_AP50_window"))),
        "v94_best_variant_id": v94_phase8.get("best_real_variant_id", v94_phase5.get("best_real_variant_id", "")),
        "v94_final_decision": v94_phase8.get("final_decision", ""),
        "required_MV_AP_window": required_ap,
        "required_MV_AP50_window": required_ap50,
        "object_axis_smoke_available": bool(v94_phase7c_smoke),
        "object_axis_smoke_gate_pass": bool(v94_phase7c_smoke.get("object_specific_field_input_gate_pass", False)),
        "object_axis_full_dev_pass": object_axis_full_dev_pass,
        "object_axis_full_processed_source_count": _num(v94_phase7c_0011.get("processed_source_count")) + _num(v94_phase7c_0050.get("processed_source_count")),
        "object_axis_full_selected_source_count": _num(v94_phase7c_0011.get("selected_source_count")) + _num(v94_phase7c_0050.get("selected_source_count")),
        "object_axis_full_field_unary_count_shard": _num(v94_phase7c_0011.get("field_unary_count_shard")) + _num(v94_phase7c_0050.get("field_unary_count_shard")),
        "object_axis_full_failure_count": _num(v94_phase7c_0011.get("failure_count")) + _num(v94_phase7c_0050.get("failure_count")),
        "D4RT_shuffled_control_evidence_available": bool(v94_phase4.get("shuffled_D4RT_control_available", False)),
        **provenance,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "row_counts": {"gate_rows": len(gate_rows), "artifact_rows": len(artifact_rows)},
    }

    _write_csv(OUT / "phase0_gate_rows.csv", gate_rows)
    _write_csv(OUT / "artifact_rows.csv", artifact_rows)
    _write_json(OUT / "summary.json", summary)
    _write_json(
        OUT / "SHA256SUMS.json",
        {_rel(path): _sha256(path) for path in [OUT / "summary.json", OUT / "phase0_gate_rows.csv", OUT / "artifact_rows.csv"] if path.exists()},
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run()
