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
OUT_DIR = AUDIT_ROOT / "v101_phase0_fact_lock"

PHASE2C_DIR = AUDIT_ROOT / "v100_phase2c_overlap3_local_repair"
PHASE8E_DIR = AUDIT_ROOT / "v100_phase8e_final_decision_freeze"
V65_EVALUATOR = STREAM3D / "tools" / "run_v65_scene_multiview_ap.py"
PLAN_DOC = ROOT / "docs" / "stream4d_v101_geometry_provider_capability_f2_fragment_repair_plan.md"

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
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in keys})


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
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _git_status_count() -> int:
    try:
        proc = subprocess.run(
            ["git", "status", "--short"],
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return len([line for line in proc.stdout.splitlines() if line.strip()])
    except Exception:
        return -1


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
                "schema_version": "stream4d_v101_phase0_artifact_boundary_row_v1",
                "phase_id": "v101_phase0_fact_lock",
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
    phase2c_objects_path = PHASE2C_DIR / "mv_object_rows.parquet"
    phase2c_masks_path = PHASE2C_DIR / "mv_object_frame_mask_rows.parquet"
    phase8e_summary_path = PHASE8E_DIR / "summary.json"
    phase8e_metrics_path = PHASE8E_DIR / "metric_rows.csv"
    phase8e_decisions_path = PHASE8E_DIR / "decision_rows.csv"

    phase2c = _read_json(phase2c_summary_path)
    phase8e = _read_json(phase8e_summary_path)
    phase2c_metrics = _read_csv(phase2c_metrics_path)
    phase8e_metrics = _read_csv(phase8e_metrics_path)

    thresholds = _load_v65_thresholds()
    formal_metric_source_eq_v65 = (
        V65_EVALUATOR.exists()
        and thresholds == EXPECTED_THRESHOLDS
        and all("v65" in str(row.get("metric_source", "")).lower() for row in phase2c_metrics)
    )

    metric_contract = {
        "schema_version": "stream4d_v101_phase0_metric_contract_v1",
        "phase_id": "v101_phase0_fact_lock",
        "canonical_evaluator_path": _rel(V65_EVALUATOR),
        "canonical_evaluator_sha256": _sha256(V65_EVALUATOR),
        "formal_metric_source_eq_v65": bool(formal_metric_source_eq_v65),
        "AP_thresholds_actual": thresholds,
        "AP_thresholds_expected": EXPECTED_THRESHOLDS,
        "ap_thresholds_match": thresholds == EXPECTED_THRESHOLDS,
        "metric_contract_note": "Phase0 only locks existing v100 Phase2c metrics generated by the v65 evaluator; it does not recompute or improve AP.",
    }

    baseline_rows: list[dict[str, Any]] = []
    for row in phase2c_metrics:
        split = row.get("dataset_split", "")
        baseline_rows.append(
            {
                "schema_version": "stream4d_v101_phase0_f2_baseline_metric_row_v1",
                "phase_id": "v101_phase0_fact_lock",
                "source_phase_id": row.get("phase_id", ""),
                "source_artifact": _rel(phase2c_metrics_path),
                "dataset_split": split,
                "variant_id": row.get("variant_id", ""),
                "MV_AP_window": row.get("MV_AP_window", ""),
                "MV_AP50_window": row.get("MV_AP50_window", ""),
                "MV_AP25_window": row.get("MV_AP25_window", ""),
                "ScoreFreeMatch50_window": row.get("ScoreFreeMatch50_window", ""),
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

    scene_attempt = next((r for r in phase8e_metrics if r.get("row_id") == "best_scene_attempt_phase4r"), {})
    scene_status_row = {
        "schema_version": "stream4d_v101_phase0_scene_status_v1",
        "phase_id": "v101_phase0_fact_lock",
        "source_phase_id": "v100_phase8e_final_decision_freeze",
        "source_artifact": _rel(phase8e_metrics_path),
        "decision": phase8e.get("decision"),
        "full_goal_achieved": phase8e.get("full_goal_achieved"),
        "local_claim_allowed": phase8e.get("local_claim_allowed"),
        "scene_claim_allowed": phase8e.get("scene_claim_allowed"),
        "uses_gt_for_prediction": phase8e.get("uses_gt_for_prediction"),
        "best_scene_source_id": scene_attempt.get("source_id", ""),
        "best_scene_variant_id": scene_attempt.get("variant_id", ""),
        "dev_MV_AP_scene": scene_attempt.get("dev_MV_AP_scene", ""),
        "dev_MV_AP50_scene": scene_attempt.get("dev_MV_AP50_scene", ""),
        "holdout_MV_AP_scene": scene_attempt.get("holdout_MV_AP_scene", ""),
        "holdout_MV_AP50_scene": scene_attempt.get("holdout_MV_AP50_scene", ""),
        "dev_scene_gate": scene_attempt.get("dev_scene_gate", ""),
        "dev_ap50_gate": scene_attempt.get("dev_ap50_gate", ""),
        "holdout_scene_gate": scene_attempt.get("holdout_scene_gate", ""),
        "holdout_ap50_gate": scene_attempt.get("holdout_ap50_gate", ""),
    }

    artifact_rows = _artifact_rows(
        [
            (PLAN_DOC, "plan_doc", "v101 plan being executed"),
            (V65_EVALUATOR, "canonical_evaluator", "v65 SparseSceneIoU/_summarize_iou evaluator"),
            (phase2c_summary_path, "f2_phase2c_summary", "current F2 local baseline summary"),
            (phase2c_metrics_path, "f2_phase2c_metrics", "current F2 dev/holdout local and fragmented scene metrics"),
            (phase2c_gates_path, "f2_phase2c_gates", "current F2 safety and overlap gates"),
            (phase2c_objects_path, "f2_phase2c_object_rows", "current F2 object rows for Phase1"),
            (phase2c_masks_path, "f2_phase2c_object_frame_mask_rows", "current F2 object-frame-mask rows for Phase1"),
            (phase8e_summary_path, "v100_final_decision_summary", "v100 final local-only/scene-No-Go boundary"),
            (phase8e_metrics_path, "v100_final_metric_rows", "v100 final best scene attempt and local claim metrics"),
            (phase8e_decisions_path, "v100_final_decision_rows", "v100 final decision row"),
        ]
    )

    by_split = {str(row.get("dataset_split")): row for row in baseline_rows}
    dev = by_split.get("dev", {})
    holdout = by_split.get("holdout", {})
    same_frame_collision_total = sum(int(_num(row.get("same_frame_collision_count"))) for row in baseline_rows)
    missing_mask_total = sum(int(_num(row.get("missing_mask_raster_count"))) for row in baseline_rows)
    max_pixel_collision = max((_num(row.get("pixel_collision_rate")) for row in baseline_rows), default=999.0)
    uses_gt_any = any(_bool(row.get("uses_gt_for_prediction")) for row in baseline_rows) or _bool(phase8e.get("uses_gt_for_prediction"))
    uses_future_any = any(_bool(row.get("uses_future")) for row in baseline_rows)
    overlap_pass = int(_num(phase2c.get("min_observed_overlap"), -1)) == 3 and int(_num(phase2c.get("max_observed_overlap"), -1)) == 3

    gate_rows = [
        {
            "gate_id": "formal_metric_source_eq_v65",
            "pass": bool(formal_metric_source_eq_v65),
            "expected": "v65 evaluator exists, AP thresholds match, metric_source contains v65",
            "observed": f"evaluator_exists={V65_EVALUATOR.exists()} thresholds={thresholds} metric_sources={[row.get('metric_source', '') for row in phase2c_metrics]}",
            "severity": "required",
        },
        {
            "gate_id": "AP_thresholds_actual_match",
            "pass": thresholds == EXPECTED_THRESHOLDS,
            "expected": EXPECTED_THRESHOLDS,
            "observed": thresholds,
            "severity": "required",
        },
        {
            "gate_id": "F2_phase2c_dev_MV_AP_window_available",
            "pass": "MV_AP_window" in dev and str(dev.get("MV_AP_window", "")) != "",
            "expected": "dev MV_AP_window available",
            "observed": dev.get("MV_AP_window", ""),
            "severity": "required",
        },
        {
            "gate_id": "F2_phase2c_holdout_MV_AP_window_available",
            "pass": "MV_AP_window" in holdout and str(holdout.get("MV_AP_window", "")) != "",
            "expected": "holdout MV_AP_window available",
            "observed": holdout.get("MV_AP_window", ""),
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
            "gate_id": "overlap3_materialization_contract",
            "pass": overlap_pass,
            "expected": "min_observed_overlap=max_observed_overlap=3",
            "observed": {
                "overlap_transition_count": phase2c.get("overlap_transition_count"),
                "min_observed_overlap": phase2c.get("min_observed_overlap"),
                "max_observed_overlap": phase2c.get("max_observed_overlap"),
            },
            "severity": "required",
        },
        {
            "gate_id": "v100_scene_claim_blocked",
            "pass": bool(phase8e.get("local_claim_allowed")) and not bool(phase8e.get("scene_claim_allowed")),
            "expected": "local claim allowed and scene claim blocked",
            "observed": {"local_claim_allowed": phase8e.get("local_claim_allowed"), "scene_claim_allowed": phase8e.get("scene_claim_allowed")},
            "severity": "context_lock",
        },
    ]

    failure_rows = [
        {
            "schema_version": "stream4d_v101_phase0_failure_row_v1",
            "phase_id": "v101_phase0_fact_lock",
            "gate_id": row["gate_id"],
            "expected": row["expected"],
            "observed": row["observed"],
            "severity": row["severity"],
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]

    phase0_pass = not failure_rows
    decision = "PASS_ENTER_PHASE1" if phase0_pass else "BLOCK_PHASE1_REPAIR_FACT_LOCK"

    metric_contract_path = OUT_DIR / "metric_contract.json"
    baseline_csv = OUT_DIR / "f2_baseline_metric_rows.csv"
    artifact_csv = OUT_DIR / "artifact_boundary_rows.csv"
    gate_csv = OUT_DIR / "variant_gate_rows.csv"
    failure_csv = OUT_DIR / "variant_failure_rows.csv"
    scene_csv = OUT_DIR / "scene_status_rows.csv"

    _write_json(metric_contract_path, metric_contract)
    _write_csv(baseline_csv, baseline_rows)
    _write_csv(scene_csv, [scene_status_row])
    _write_csv(artifact_csv, artifact_rows)
    _write_csv(gate_csv, gate_rows)
    _write_csv(failure_csv, failure_rows)

    summary = {
        "schema_version": "stream4d_v101_phase0_fact_lock_summary_v1",
        "phase_id": "v101_phase0_fact_lock",
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "phase0_pass": bool(phase0_pass),
        "failure_count": len(failure_rows),
        "formal_metric_source_eq_v65": bool(formal_metric_source_eq_v65),
        "AP_thresholds_actual": thresholds,
        "F2_variant_id": dev.get("variant_id", holdout.get("variant_id", "")),
        "F2_phase2c_dev_MV_AP_window": _num(dev.get("MV_AP_window")),
        "F2_phase2c_dev_MV_AP50_window": _num(dev.get("MV_AP50_window")),
        "F2_phase2c_holdout_MV_AP_window": _num(holdout.get("MV_AP_window")),
        "F2_phase2c_holdout_MV_AP50_window": _num(holdout.get("MV_AP50_window")),
        "F2_phase2c_dev_MV_AP_scene_fragmented": _num(dev.get("MV_AP_scene_fragmented")),
        "F2_phase2c_holdout_MV_AP_scene_fragmented": _num(holdout.get("MV_AP_scene_fragmented")),
        "same_frame_collision_count": same_frame_collision_total,
        "pixel_collision_rate_max": max_pixel_collision,
        "missing_mask_raster_count": missing_mask_total,
        "uses_gt_for_prediction": uses_gt_any,
        "uses_future": uses_future_any,
        "method_chunk_size": int(_num(dev.get("method_chunk_size", holdout.get("method_chunk_size", 0)))),
        "frame_stride": int(_num(dev.get("frame_stride", holdout.get("frame_stride", 0)))),
        "overlap": 3,
        "overlap_transition_count": phase2c.get("overlap_transition_count"),
        "min_observed_overlap": phase2c.get("min_observed_overlap"),
        "max_observed_overlap": phase2c.get("max_observed_overlap"),
        "v100_final_decision": phase8e.get("decision"),
        "v100_local_claim_allowed": phase8e.get("local_claim_allowed"),
        "v100_scene_claim_allowed": phase8e.get("scene_claim_allowed"),
        "v100_best_scene_attempt": scene_status_row,
        "git_status_short_count_at_run": _git_status_count(),
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "metric_contract": _rel(metric_contract_path),
            "f2_baseline_metric_rows": _rel(baseline_csv),
            "scene_status_rows": _rel(scene_csv),
            "artifact_boundary_rows": _rel(artifact_csv),
            "variant_gate_rows": _rel(gate_csv),
            "variant_failure_rows": _rel(failure_csv),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if phase0_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
