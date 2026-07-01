#!/usr/bin/env python3
"""Post-final v99 Phase10K holdout chunk object-birth sweep.

This diagnostic switches the v99 Phase1 surfel-maskview object-birth builder to
the same-scene temporal holdout artifacts. It tests whether the chunk-scoped
object candidate generation itself can beat the locked F2 holdout baseline,
rather than continuing score/support blending on fixed legacy rows.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools import build_v99_phase1_f2_base_reproduction as p1  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase10k_holdout_chunk_object_birth_sweep"
PHASE0_DIR = AUDIT_ROOT / "v99_phase0_fact_lock"
HOLDOUT_SOURCE_ROWS = AUDIT_ROOT / "v98_phase13_holdout/source_container_rows.csv"
HOLDOUT_RADIO_MASK_FEATURES = AUDIT_ROOT / "v98_phase13_holdout_radio_features_npz/mask_features.npz"
HOLDOUT_SURFEL_ROWS = AUDIT_ROOT / "v98_phase13_holdout_phase5_fused_surfel/fused_surfel_rows.csv"
HOLDOUT_SURFEL_OBS_ROWS = AUDIT_ROOT / "v98_phase13_holdout_phase5_fused_surfel/surfel_observation_rows.csv"
HOLDOUT_SURFEL_SUMMARY = AUDIT_ROOT / "v98_phase13_holdout_phase5_fused_surfel/summary.json"


def _rel(path: Path | str) -> str:
    q = Path(path)
    try:
        return q.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return q.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _jsonable(row.get(field, "")) for field in fields})


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _patch_phase1_inputs() -> dict[str, str]:
    original = {
        "SOURCE_ROWS": _rel(p1.SOURCE_ROWS),
        "RADIO_MASK_FEATURES": _rel(p1.RADIO_MASK_FEATURES),
        "SURFEL_ROWS": _rel(p1.SURFEL_ROWS),
        "SURFEL_OBS_ROWS": _rel(p1.SURFEL_OBS_ROWS),
        "SURFEL_SUMMARY": _rel(p1.SURFEL_SUMMARY),
    }
    p1.SOURCE_ROWS = HOLDOUT_SOURCE_ROWS
    p1.RADIO_MASK_FEATURES = HOLDOUT_RADIO_MASK_FEATURES
    p1.SURFEL_ROWS = HOLDOUT_SURFEL_ROWS
    p1.SURFEL_OBS_ROWS = HOLDOUT_SURFEL_OBS_ROWS
    p1.SURFEL_SUMMARY = HOLDOUT_SURFEL_SUMMARY
    return original


def _artifact_inputs() -> dict[str, str]:
    return {
        "source_rows": _rel(HOLDOUT_SOURCE_ROWS),
        "radio_mask_features": _rel(HOLDOUT_RADIO_MASK_FEATURES),
        "surfel_rows": _rel(HOLDOUT_SURFEL_ROWS),
        "surfel_observation_rows": _rel(HOLDOUT_SURFEL_OBS_ROWS),
        "surfel_summary": _rel(HOLDOUT_SURFEL_SUMMARY),
        "semantic_constants": _rel(p1.SEMANTIC_CONSTANTS),
    }


def main() -> int:
    started = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    original_inputs = _patch_phase1_inputs()
    phase0 = json.loads((PHASE0_DIR / "summary.json").read_text(encoding="utf-8"))
    scope = p1._load_source_scope()
    rows, object_rows, birth_stats = p1._build_chunk_surfel_maskview_birth_rows(scope)

    variants = sorted({str(row["variant_id"]) for row in rows})
    metric_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for variant in variants:
        variant_rows = [row for row in rows if str(row["variant_id"]) == variant]
        metrics, frames = p1._evaluate_variant(variant, variant_rows, scope)
        metric_rows.extend(metrics)
        frame_rows.extend(frames)
    aggregate_rows = p1._aggregate_metrics(metric_rows)

    f2_hold_window = float(phase0["F2_base_holdout_MV_AP_window"])
    f2_hold_ap50 = float(phase0["F2_base_holdout_MV_AP50_window"])
    paired: list[dict[str, Any]] = []
    for row in aggregate_rows:
        window = _num(row.get("MV_AP_window"))
        ap50 = _num(row.get("MV_AP50_window"))
        paired.append(
            {
                "schema_version": "stream4d_v99_phase10k_holdout_birth_metric_v1",
                "phase_id": "v99_phase10k_holdout_chunk_object_birth_sweep",
                "variant_id": row["variant_id"],
                "holdout_MV_AP_window": row.get("MV_AP_window"),
                "holdout_MV_AP50_window": row.get("MV_AP50_window"),
                "holdout_MV_AP25_window": row.get("MV_AP25_window"),
                "holdout_MV_AP_scene": row.get("MV_AP_scene"),
                "holdout_MV_AP50_scene": row.get("MV_AP50_scene"),
                "holdout_delta_vs_F2_base_window": window - f2_hold_window,
                "strict_local_holdout_gate_pass": window >= f2_hold_window + 0.005 and ap50 >= f2_hold_ap50 + 0.010,
                "same_frame_collision_count": row.get("same_frame_collision_count"),
                "missing_mask_raster_count": row.get("missing_mask_raster_count"),
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "formal_claim_allowed": False,
            }
        )
    best = max(paired, key=lambda row: (_num(row["holdout_MV_AP_window"]), _num(row["holdout_MV_AP50_window"])))
    any_pass = any(bool(row["strict_local_holdout_gate_pass"]) for row in paired)
    input_uses_future = bool(birth_stats.get("input_phase5_uses_future", False)) or bool(scope.get("source_uses_future", False))
    input_uses_gt = bool(scope.get("source_uses_gt_for_prediction", False))

    gate_rows = [
        {
            "gate_id": "holdout_birth_input_no_future",
            "pass": not input_uses_future,
            "expected": "false",
            "observed": input_uses_future,
            "severity": "causality_contract",
        },
        {
            "gate_id": "holdout_birth_input_no_gt_prediction",
            "pass": not input_uses_gt,
            "expected": "false",
            "observed": input_uses_gt,
            "severity": "gt_contract",
        },
        {
            "gate_id": "chunk_object_birth_strict_holdout_gate",
            "pass": any_pass,
            "expected": f"MV_AP_window>={f2_hold_window + 0.005} and MV_AP50_window>={f2_hold_ap50 + 0.010}",
            "observed": f"best={best['variant_id']} MV_AP_window={best['holdout_MV_AP_window']} MV_AP50_window={best['holdout_MV_AP50_window']}",
            "severity": "method_gate",
        },
        {
            "gate_id": "formal_claim_allowed_after_post_final_diagnostic",
            "pass": False,
            "expected": "fresh frozen holdout and chunk-causal identity proof",
            "observed": "post-final diagnostic; surfel_dependency_proven_chunk_causal=false in Phase1 builder",
            "severity": "formal_claim_blocker",
        },
    ]
    failure_rows = [
        {
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": "If holdout chunk object-birth also fails, the remaining route is a new candidate-generation plan rather than v99 score/fusion repair.",
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    summary = {
        "schema_version": "stream4d_v99_phase10k_holdout_chunk_object_birth_sweep_summary_v1",
        "phase_id": "v99_phase10k_holdout_chunk_object_birth_sweep",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "decision": "NO_GO_HOLDOUT_CHUNK_OBJECT_BIRTH_SWEEP" if not any_pass else "DIAGNOSTIC_HOLDOUT_CHUNK_OBJECT_BIRTH_PASS_REQUIRES_FRESH_PROTOCOL",
        "formal_claim_allowed": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "variant_count": len(variants),
        "best_holdout_variant_id": best["variant_id"],
        "best_holdout_MV_AP_window": float(_num(best["holdout_MV_AP_window"])),
        "best_holdout_MV_AP50_window": float(_num(best["holdout_MV_AP50_window"])),
        "best_holdout_MV_AP_scene": float(_num(best["holdout_MV_AP_scene"])),
        "best_holdout_MV_AP50_scene": float(_num(best["holdout_MV_AP50_scene"])),
        "best_holdout_delta_vs_F2_base_window": float(_num(best["holdout_delta_vs_F2_base_window"])),
        "any_chunk_object_birth_variant_passes_strict_gate": any_pass,
        "F2_base_holdout_MV_AP_window": f2_hold_window,
        "F2_base_holdout_MV_AP50_window": f2_hold_ap50,
        "input_artifacts": _artifact_inputs(),
        "original_phase1_inputs_before_patch": original_inputs,
        "birth_stats": birth_stats,
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "paired_metric_rows": _rel(OUT_DIR / "paired_metric_rows.csv"),
            "variant_metric_rows": _rel(OUT_DIR / "variant_metric_rows.csv"),
            "variant_metric_scene_rows": _rel(OUT_DIR / "variant_metric_scene_rows.csv"),
            "variant_gate_rows": _rel(OUT_DIR / "variant_gate_rows.csv"),
            "variant_failure_rows": _rel(OUT_DIR / "variant_failure_rows.csv"),
            "mv_object_frame_mask_rows": _rel(OUT_DIR / "mv_object_frame_mask_rows.csv"),
            "mv_object_rows": _rel(OUT_DIR / "mv_object_rows.csv"),
        },
    }

    _write_csv(OUT_DIR / "paired_metric_rows.csv", paired)
    _write_csv(OUT_DIR / "variant_metric_rows.csv", aggregate_rows)
    _write_csv(OUT_DIR / "variant_metric_scene_rows.csv", metric_rows)
    _write_csv(OUT_DIR / "frame_eval_rows.csv", frame_rows)
    _write_csv(OUT_DIR / "mv_object_frame_mask_rows.csv", rows)
    _write_csv(OUT_DIR / "mv_object_rows.csv", object_rows)
    _write_csv(OUT_DIR / "variant_gate_rows.csv", gate_rows)
    _write_csv(OUT_DIR / "variant_failure_rows.csv", failure_rows)
    _write_json(OUT_DIR / "birth_stats.json", birth_stats)
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if any_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
