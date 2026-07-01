from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STREAM3D = ROOT / "Stream3D"
AUDIT_ROOT = STREAM3D / "outputs" / "audit"
OUT_DIR = AUDIT_ROOT / "v102_phase0_fact_lock"

PHASE2C_DIR = AUDIT_ROOT / "v100_phase2c_overlap3_local_repair"
V100_FINAL_DIR = AUDIT_ROOT / "v100_phase8e_final_decision_freeze"
V101_FINAL_DIR = AUDIT_ROOT / "v101_final_closeout"
V101_PHASE2_DIR = AUDIT_ROOT / "v101_phase2_geometry_provider_capability"
V101_PHASE2B_DIR = AUDIT_ROOT / "v101_phase2b_false_bridge_repair_probe"
V65_EVALUATOR = STREAM3D / "tools" / "run_v65_scene_multiview_ap.py"
PLAN_DOC = ROOT / "docs" / "stream4d_v102_mask_pair_primitive_bridge_da3_giant_3dgs_plan.md"

EXPECTED_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _git_status_count() -> int:
    proc = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return len([line for line in proc.stdout.splitlines() if line.strip()])


def _load_v65_thresholds() -> list[float]:
    if str(STREAM3D) not in sys.path:
        sys.path.insert(0, str(STREAM3D))
    from tools import run_v65_scene_multiview_ap as v65  # noqa: WPS433

    return [float(x) for x in v65.AP_THRESHOLDS]


def _artifact_rows(paths: list[tuple[Path, str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, role, note in paths:
        exists = path.exists()
        rows.append(
            {
                "schema_version": "stream4d_v102_phase0_artifact_boundary_row_v1",
                "phase_id": "v102_phase0_fact_lock",
                "role": role,
                "path": _rel(path),
                "exists": exists,
                "sha256": _sha256(path) if exists and path.is_file() else "",
                "note": note,
            }
        )
    return rows


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    phase2c_summary_path = PHASE2C_DIR / "summary.json"
    phase2c_metrics_path = PHASE2C_DIR / "variant_metric_rows.csv"
    phase2c_gates_path = PHASE2C_DIR / "variant_gate_rows.csv"
    v100_final_summary_path = V100_FINAL_DIR / "summary.json"
    v101_final_summary_path = V101_FINAL_DIR / "summary.json"
    v101_phase2_summary_path = V101_PHASE2_DIR / "summary.json"
    v101_phase2b_summary_path = V101_PHASE2B_DIR / "summary.json"

    phase2c = _read_json(phase2c_summary_path)
    v100_final = _read_json(v100_final_summary_path)
    v101_final = _read_json(v101_final_summary_path)
    v101_phase2 = _read_json(v101_phase2_summary_path)
    v101_phase2b = _read_json(v101_phase2b_summary_path)
    phase2c_metrics = _read_csv(phase2c_metrics_path)

    thresholds = _load_v65_thresholds()
    formal_metric_source_eq_v65 = (
        V65_EVALUATOR.exists()
        and thresholds == EXPECTED_THRESHOLDS
        and all("v65" in str(row.get("metric_source", "")).lower() for row in phase2c_metrics)
    )

    metric_contract = {
        "schema_version": "stream4d_v102_phase0_evaluator_contract_v1",
        "phase_id": "v102_phase0_fact_lock",
        "canonical_evaluator_path": _rel(V65_EVALUATOR),
        "canonical_evaluator_sha256": _sha256(V65_EVALUATOR),
        "formal_metric_source_eq_v65": bool(formal_metric_source_eq_v65),
        "AP_thresholds_actual": thresholds,
        "AP_thresholds_expected": EXPECTED_THRESHOLDS,
        "ap_thresholds_match": thresholds == EXPECTED_THRESHOLDS,
        "metric_boundary_note": "Phase0 locks existing v100 Phase2c/v101 No-Go facts only; it does not recompute or improve AP.",
    }

    baseline_rows: list[dict[str, Any]] = []
    for row in phase2c_metrics:
        baseline_rows.append(
            {
                "schema_version": "stream4d_v102_phase0_baseline_metric_row_v1",
                "phase_id": "v102_phase0_fact_lock",
                "source_phase_id": row.get("phase_id", ""),
                "source_artifact": _rel(phase2c_metrics_path),
                "dataset_split": row.get("dataset_split", ""),
                "variant_id": row.get("variant_id", ""),
                "MV_AP_window": row.get("MV_AP_window", ""),
                "MV_AP50_window": row.get("MV_AP50_window", ""),
                "MV_AP_scene_fragmented": row.get("MV_AP_scene", ""),
                "MV_AP50_scene_fragmented": row.get("MV_AP50_scene", ""),
                "same_frame_collision_count": row.get("same_frame_collision_count", ""),
                "pixel_collision_rate": row.get("pixel_collision_rate", ""),
                "missing_mask_raster_count": row.get("missing_mask_raster_count", ""),
                "uses_gt_for_prediction": row.get("uses_gt_for_prediction", ""),
                "uses_future": row.get("uses_future", ""),
                "method_chunk_size": row.get("method_chunk_size", ""),
                "frame_stride": row.get("frame_stride", ""),
                "overlap": 3,
                "metric_source": row.get("metric_source", ""),
            }
        )

    by_split = {str(row.get("dataset_split")): row for row in baseline_rows}
    dev = by_split.get("dev", {})
    holdout = by_split.get("holdout", {})

    same_frame_collision_total = sum(int(_num(row.get("same_frame_collision_count"))) for row in baseline_rows)
    missing_mask_total = sum(int(_num(row.get("missing_mask_raster_count"))) for row in baseline_rows)
    max_pixel_collision = max((_num(row.get("pixel_collision_rate")) for row in baseline_rows), default=999.0)
    uses_gt_any = any(_bool(row.get("uses_gt_for_prediction")) for row in baseline_rows)
    uses_future_any = any(_bool(row.get("uses_future")) for row in baseline_rows)
    overlap_pass = int(_num(phase2c.get("min_observed_overlap"), -1)) == 3 and int(
        _num(phase2c.get("max_observed_overlap"), -1)
    ) == 3

    boundary_rows = [
        {
            "schema_version": "stream4d_v102_phase0_prior_boundary_row_v1",
            "phase_id": "v102_phase0_fact_lock",
            "source": "v100_phase8e_final_decision",
            "decision": v100_final.get("decision"),
            "local_claim_allowed": v100_final.get("local_claim_allowed"),
            "scene_claim_allowed": v100_final.get("scene_claim_allowed"),
            "best_scene_attempt_holdout_MV_AP_scene": v100_final.get("best_scene_attempt", {}).get("holdout_MV_AP_scene"),
            "boundary_note": "Only Phase2c local/window F2 is claimable from v100.",
        },
        {
            "schema_version": "stream4d_v102_phase0_prior_boundary_row_v1",
            "phase_id": "v102_phase0_fact_lock",
            "source": "v101_final_closeout",
            "decision": v101_final.get("final_decision"),
            "phase3_fragment_repair_allowed": v101_final.get("phase3_fragment_repair_allowed"),
            "phase3_fragment_repair_run": v101_final.get("phase3_fragment_repair_run"),
            "method_success_claim_allowed": v101_final.get("method_success_claim_allowed"),
            "boundary_note": "v101 blocked fragment repair because provider bridge false-bridge and purity gates did not pass.",
        },
    ]

    gate_rows = [
        {
            "gate_id": "formal_metric_source_eq_v65",
            "pass": bool(formal_metric_source_eq_v65),
            "expected": "v65 evaluator exists, thresholds match, metric_source contains v65",
            "observed": f"thresholds={thresholds}",
            "severity": "required",
        },
        {
            "gate_id": "current_positive_method_locked",
            "pass": bool(dev.get("variant_id") or holdout.get("variant_id")),
            "expected": "F2 Phase2c baseline present",
            "observed": dev.get("variant_id", holdout.get("variant_id", "")),
            "severity": "required",
        },
        {
            "gate_id": "same_frame_collision_count_zero",
            "pass": same_frame_collision_total == 0,
            "expected": 0,
            "observed": same_frame_collision_total,
            "severity": "required",
        },
        {
            "gate_id": "pixel_collision_rate_le_0p02",
            "pass": max_pixel_collision <= 0.02,
            "expected": "<=0.02",
            "observed": max_pixel_collision,
            "severity": "required",
        },
        {
            "gate_id": "missing_mask_raster_count_zero",
            "pass": missing_mask_total == 0,
            "expected": 0,
            "observed": missing_mask_total,
            "severity": "required",
        },
        {
            "gate_id": "uses_gt_for_prediction_false",
            "pass": not uses_gt_any,
            "expected": False,
            "observed": uses_gt_any,
            "severity": "required",
        },
        {
            "gate_id": "uses_future_false",
            "pass": not uses_future_any,
            "expected": False,
            "observed": uses_future_any,
            "severity": "required",
        },
        {
            "gate_id": "chunk32_overlap3_contract",
            "pass": overlap_pass,
            "expected": "chunk_size=32 overlap=3",
            "observed": {
                "method_chunk_size": dev.get("method_chunk_size", holdout.get("method_chunk_size", "")),
                "frame_stride": dev.get("frame_stride", holdout.get("frame_stride", "")),
                "min_observed_overlap": phase2c.get("min_observed_overlap"),
                "max_observed_overlap": phase2c.get("max_observed_overlap"),
            },
            "severity": "required",
        },
        {
            "gate_id": "v101_phase3_not_run",
            "pass": v101_final.get("phase3_fragment_repair_run") is False,
            "expected": False,
            "observed": v101_final.get("phase3_fragment_repair_run"),
            "severity": "context_lock",
        },
        {
            "gate_id": "v101_provider_bridge_not_confirmed",
            "pass": v101_phase2.get("provider_bridge_potential_confirmed") is False
            and v101_phase2b.get("passing_filter_count") == 0,
            "expected": "provider bridge not confirmed; no Phase2b filter passed",
            "observed": {
                "provider_bridge_potential_confirmed": v101_phase2.get("provider_bridge_potential_confirmed"),
                "passing_filter_count": v101_phase2b.get("passing_filter_count"),
            },
            "severity": "context_lock",
        },
    ]
    failure_rows = [
        {
            "schema_version": "stream4d_v102_phase0_failure_row_v1",
            "phase_id": "v102_phase0_fact_lock",
            "gate_id": row["gate_id"],
            "expected": row["expected"],
            "observed": row["observed"],
            "severity": row["severity"],
        }
        for row in gate_rows
        if row["severity"] == "required" and not bool(row["pass"])
    ]

    artifact_rows = _artifact_rows(
        [
            (PLAN_DOC, "plan_doc", "v102 plan being executed"),
            (V65_EVALUATOR, "canonical_evaluator", "v65 SparseSceneIoU/_summarize_iou evaluator"),
            (phase2c_summary_path, "f2_phase2c_summary", "current F2 local baseline summary"),
            (phase2c_metrics_path, "f2_phase2c_metrics", "current F2 dev/holdout local and fragmented scene metrics"),
            (phase2c_gates_path, "f2_phase2c_gates", "current F2 safety and overlap gates"),
            (V100_FINAL_DIR / "summary.json", "v100_final_summary", "v100 local-only/scene-No-Go boundary"),
            (V101_FINAL_DIR / "summary.json", "v101_final_summary", "v101 provider-bridge No-Go boundary"),
            (V101_PHASE2_DIR / "summary.json", "v101_provider_summary", "v101 provider bridge evidence"),
            (V101_PHASE2B_DIR / "summary.json", "v101_false_bridge_repair_summary", "v101 plan-directed false bridge repairs"),
        ]
    )

    phase0_pass = not failure_rows
    decision = "PASS_ENTER_PHASE1" if phase0_pass else "BLOCK_PHASE1_REPAIR_FACT_LOCK"

    evaluator_path = OUT_DIR / "evaluator_contract.json"
    baseline_path = OUT_DIR / "baseline_metric_rows.csv"
    boundary_path = OUT_DIR / "prior_boundary_rows.csv"
    artifact_path = OUT_DIR / "artifact_boundary_rows.csv"
    gate_path = OUT_DIR / "variant_gate_rows.csv"
    failure_path = OUT_DIR / "variant_failure_rows.csv"

    _write_json(evaluator_path, metric_contract)
    _write_csv(baseline_path, baseline_rows)
    _write_csv(boundary_path, boundary_rows)
    _write_csv(artifact_path, artifact_rows)
    _write_csv(gate_path, gate_rows)
    _write_csv(failure_path, failure_rows)

    summary = {
        "schema_version": "stream4d_v102_phase0_fact_lock_summary_v1",
        "phase_id": "v102_phase0_fact_lock",
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "phase0_pass": bool(phase0_pass),
        "failure_count": len(failure_rows),
        "formal_metric_source_eq_v65": bool(formal_metric_source_eq_v65),
        "AP_thresholds": thresholds,
        "current_positive_method": "F2_v100_chunk32_overlap3_surfel_maskview_thr018_p2d2",
        "F2_dev_MV_AP_window": _num(dev.get("MV_AP_window")),
        "F2_dev_MV_AP50_window": _num(dev.get("MV_AP50_window")),
        "F2_dev_fragmented_MV_AP_scene": _num(dev.get("MV_AP_scene_fragmented")),
        "F2_holdout_MV_AP_window": _num(holdout.get("MV_AP_window")),
        "F2_holdout_MV_AP50_window": _num(holdout.get("MV_AP50_window")),
        "F2_holdout_fragmented_MV_AP_scene": _num(holdout.get("MV_AP_scene_fragmented")),
        "same_frame_collision_count": same_frame_collision_total,
        "pixel_collision_rate_max": max_pixel_collision,
        "missing_mask_raster_count": missing_mask_total,
        "uses_gt_for_prediction": uses_gt_any,
        "uses_future": uses_future_any,
        "method_chunk_size": 32,
        "frame_stride": int(_num(dev.get("frame_stride", holdout.get("frame_stride", 0)))),
        "overlap": 3,
        "v101_phase3_fragment_repair_run": v101_final.get("phase3_fragment_repair_run"),
        "v101_provider_bridge_potential_confirmed": v101_phase2.get("provider_bridge_potential_confirmed"),
        "v101_surfel_purity_available_provider_count": v101_phase2.get("surfel_purity_available_provider_count"),
        "v101_phase2b_passing_filter_count": v101_phase2b.get("passing_filter_count"),
        "git_status_short_count_at_run": _git_status_count(),
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "baseline_metric_rows": _rel(baseline_path),
            "evaluator_contract": _rel(evaluator_path),
            "prior_boundary_rows": _rel(boundary_path),
            "artifact_boundary_rows": _rel(artifact_path),
            "variant_gate_rows": _rel(gate_path),
            "variant_failure_rows": _rel(failure_path),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if phase0_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
