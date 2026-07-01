#!/usr/bin/env python3
"""Build v95 dev decision from completed object-core/query/expansion artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/audit/v95_phase8_dev_decision"
PHASE_ID = "v95_phase8_dev_decision"
RUN_ID = "v95_phase8_dev_decision"
PHASE0 = ROOT / "outputs/audit/v95_phase0_fact_lock/summary.json"
PHASE1 = ROOT / "outputs/audit/v95_phase1_physical_source_registry/summary.json"
PHASE2 = ROOT / "outputs/audit/v95_phase2_object_core_discovery_repair1/summary.json"
PHASE3 = ROOT / "outputs/audit/v95_phase3_object_query/summary.json"
PHASE_ROOTS = [
    ("phase4_familyA", ROOT / "outputs/audit/v95_phase4_familyA_core_conditioned_expansion"),
    ("phase4_familyB", ROOT / "outputs/audit/v95_phase4_familyB_seeded_graph_propagation"),
    ("phase4_familyC", ROOT / "outputs/audit/v95_phase4C_object_competition"),
    ("phase4_familyC_support12", ROOT / "outputs/audit/v95_phase4C_object_competition_support12"),
    ("phase4_familyC_support20", ROOT / "outputs/audit/v95_phase4C_object_competition_support20"),
    ("phase4_familyC_support30", ROOT / "outputs/audit/v95_phase4C_object_competition_support30"),
    ("phase4_familyD_support30", ROOT / "outputs/audit/v95_phase4D_component_assignment_support30"),
    ("phase4_familyE_gpu_support30", ROOT / "outputs/audit/v95_phase4E_gpu_object_axis_readout_support30"),
    ("phase4_familyE_gpu_support30_cons06", ROOT / "outputs/audit/v95_phase4E_gpu_query_consistency_support30_cons06"),
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
            writer.writerow({key: row.get(key, "") for key in fields})


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    phase0 = _read_json(PHASE0)
    phase1 = _read_json(PHASE1)
    phase2 = _read_json(PHASE2)
    phase3 = _read_json(PHASE3)
    rows: list[dict[str, Any]] = []
    for phase_name, root in PHASE_ROOTS:
        summary = _read_json(root / "summary.json")
        for row in _read_csv(root / "variant_metric_rows.csv"):
            rows.append(
                {
                    "schema_version": "stream4d_v95_phase8_variant_rank_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "source_phase": phase_name,
                    "source_artifact": _rel(root / "variant_metric_rows.csv"),
                    "source_decision": summary.get("decision", ""),
                    "variant_id": row.get("variant_id", ""),
                    "family": row.get("family", "real"),
                    "MV_AP_window": _num(row.get("mean_MV_AP_window")),
                    "MV_AP50_window": _num(row.get("mean_MV_AP50_window")),
                    "MV_AP25_window": _num(row.get("mean_MV_AP25_window")),
                    "ScoreFreeMatch50_window": _num(row.get("mean_score_free_Match50_window")),
                    "mean_generated_area_ratio": _num(row.get("mean_generated_area_ratio")),
                    "same_frame_collision_count": _num(row.get("same_frame_collision_count")),
                    "missing_mask_raster_count": _num(row.get("missing_mask_raster_count")),
                    "uses_gt_for_prediction": row.get("uses_gt_for_prediction", "False"),
                    "uses_future": row.get("uses_future", "False"),
                }
            )
    rows.sort(key=lambda row: (row["MV_AP_window"], row["MV_AP50_window"], row["ScoreFreeMatch50_window"]), reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    best_real = rows[0] if rows else {}
    required_ap = _num(phase0.get("required_MV_AP_window"))
    required_ap50 = _num(phase0.get("required_MV_AP50_window"))
    candidate_required_ap = max(_num(phase0.get("v91_best_MV_AP_window")) + 0.002, _num(phase0.get("best_control_MV_AP_window")) + 0.005)
    candidate_required_ap50 = max(_num(phase0.get("v91_best_MV_AP50_window")) + 0.004, _num(phase0.get("best_control_MV_AP50_window")) + 0.010)
    dev_progress_gate = bool(best_real and best_real["MV_AP_window"] >= required_ap and best_real["MV_AP50_window"] >= required_ap50)
    phase4_candidate_gate = bool(best_real and best_real["MV_AP_window"] >= candidate_required_ap and best_real["MV_AP50_window"] >= candidate_required_ap50)
    provenance_gate = bool(
        best_real
        and best_real["same_frame_collision_count"] == 0
        and best_real["missing_mask_raster_count"] == 0
        and not _bool(best_real["uses_gt_for_prediction"])
        and not _bool(best_real["uses_future"])
    )
    decision_rows = [
        {
            "schema_version": "stream4d_v95_phase8_decision_matrix_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "criterion": "dev_progress_gate",
            "pass": dev_progress_gate,
            "observed_MV_AP_window": best_real.get("MV_AP_window", ""),
            "observed_MV_AP50_window": best_real.get("MV_AP50_window", ""),
            "required_MV_AP_window": required_ap,
            "required_MV_AP50_window": required_ap50,
        },
        {
            "schema_version": "stream4d_v95_phase8_decision_matrix_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "criterion": "phase4_candidate_gate",
            "pass": phase4_candidate_gate,
            "observed_MV_AP_window": best_real.get("MV_AP_window", ""),
            "observed_MV_AP50_window": best_real.get("MV_AP50_window", ""),
            "candidate_required_MV_AP_window": candidate_required_ap,
            "candidate_required_MV_AP50_window": candidate_required_ap50,
        },
        {
            "schema_version": "stream4d_v95_phase8_decision_matrix_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "criterion": "provenance_materializer_gate",
            "pass": provenance_gate,
            "same_frame_collision_count": best_real.get("same_frame_collision_count", ""),
            "missing_mask_raster_count": best_real.get("missing_mask_raster_count", ""),
            "uses_gt_for_prediction": best_real.get("uses_gt_for_prediction", ""),
            "uses_future": best_real.get("uses_future", ""),
        },
        {
            "schema_version": "stream4d_v95_phase8_decision_matrix_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "criterion": "holdout_allowed",
            "pass": bool(dev_progress_gate and phase4_candidate_gate and provenance_gate),
            "holdout_executed": False,
        },
    ]
    final_pass = bool(dev_progress_gate and phase4_candidate_gate and provenance_gate)
    summary = {
        "schema": "stream4d_v95_phase8_dev_decision_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "final_decision": "PASS_V95_DEV_GATE_READY_FOR_HOLDOUT" if final_pass else "NO_GO_V95_LOCAL_MV_AP_WINDOW",
        "dev_progress_gate_pass": dev_progress_gate,
        "phase4_candidate_gate_pass": phase4_candidate_gate,
        "provenance_materializer_gate_pass": provenance_gate,
        "holdout_executed": False,
        "local2history_blocked": not final_pass,
        "best_real_phase": best_real.get("source_phase", ""),
        "best_real_variant_id": best_real.get("variant_id", ""),
        "best_real_MV_AP_window": best_real.get("MV_AP_window", ""),
        "best_real_MV_AP50_window": best_real.get("MV_AP50_window", ""),
        "best_real_MV_AP25_window": best_real.get("MV_AP25_window", ""),
        "best_real_ScoreFreeMatch50_window": best_real.get("ScoreFreeMatch50_window", ""),
        "required_MV_AP_window": required_ap,
        "required_MV_AP50_window": required_ap50,
        "candidate_required_MV_AP_window": candidate_required_ap,
        "candidate_required_MV_AP50_window": candidate_required_ap50,
        "phase1_decision": phase1.get("decision", ""),
        "phase2_decision": phase2.get("decision", ""),
        "phase3_decision": phase3.get("decision", ""),
        "no_go_reason": (
            "Phase2/Phase3 passed after repairs, but Phase4 A/B/C/D/E and query-support repairs remained far below "
            "local MV_AP_window and AP50 gates; best real variant still has low ScoreFreeMatch50, so score calibration/holdout/local2history stay blocked."
        ),
        "row_counts": {
            "variant_rank_rows": len(rows),
            "decision_matrix_rows": len(decision_rows),
        },
    }
    _write_csv(OUT / "variant_rank_rows.csv", rows)
    _write_csv(OUT / "decision_matrix_rows.csv", decision_rows)
    _write_json(OUT / "summary.json", summary)
    outputs = [OUT / "variant_rank_rows.csv", OUT / "decision_matrix_rows.csv", OUT / "summary.json"]
    _write_json(OUT / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs})
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    run()
