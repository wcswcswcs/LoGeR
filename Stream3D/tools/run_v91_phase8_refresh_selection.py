from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT8 = ROOT / "outputs/audit/v91_phase8_dev_selection"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _base_rows() -> list[dict[str, Any]]:
    rows = _read_csv(OUT8 / "all_variant_metric_rows.csv")
    keep_prefixes = {
        "B0_local_only",
        "C0_semantic_only_control",
        "P3_C0_area_semantic_hybrid_score",
        "S3D_L1_local_merged_masks",
        "W9b_v90_risk_safe",
        "W9a_v90_aggressive",
        "F_W9a_drop_broad_low_h9_5_broad_scene_orig_ge060",
        "F_W9a_drop_broad_low_h9_2_broad_scene_orig_ge060",
    }
    return [dict(row) for row in rows if row.get("variant_id") in keep_prefixes]


def _rows_from_control_csv(path: Path, method_family: str, notes: str, limit_prefix: str | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in _read_csv(path):
        variant_id = row.get("variant_id", "")
        if limit_prefix and not variant_id.startswith(limit_prefix):
            continue
        out.append(
            {
                "variant_id": variant_id,
                "method_family": method_family,
                "source_artifact": _rel(path),
                "mean_MV_AP_window": row.get("mean_MV_AP_window", row.get("MV_AP_window", "")),
                "mean_MV_AP50_window": row.get("mean_MV_AP50_window", row.get("MV_AP50_window", "")),
                "mean_MV_AP25_window": row.get("mean_MV_AP25_window", row.get("MV_AP25_window", "")),
                "mean_score_free_Match50_window": row.get("mean_score_free_Match50_window", ""),
                "same_frame_collision_count": row.get("same_frame_collision_count", "0"),
                "missing_mask_raster_count": row.get("missing_mask_raster_count", "0"),
                "uses_gt_for_prediction": row.get("uses_gt_for_prediction", "False"),
                "uses_future": row.get("uses_future", "False"),
                "risk_penalty_mean": row.get("risk_penalty_mean", ""),
                "risk_safe_vs_B0_proxy": row.get("risk_safe_vs_B0_proxy", ""),
                "notes": notes,
            }
        )
    return out


def _risk_lookup() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for path in [
        ROOT / "outputs/audit/v91_phase4_witness_cover_ap50_control_repair/risk_rows.csv",
        ROOT / "outputs/audit/v91_phase4_witness_cover_radius_sweep/risk_rows.csv",
        ROOT / "outputs/audit/v91_phase4_witness_cover_radius_sweep_fine_radius/risk_rows.csv",
        ROOT / "outputs/audit/v91_phase4_witness_cover_radius_sweep_support_point/risk_rows.csv",
        ROOT / "outputs/audit/v91_phase4_witness_cover_multimask_materialization/risk_rows.csv",
        ROOT / "outputs/audit/v91_phase4_witness_cover_multimask_score_repair/risk_rows.csv",
        ROOT / "outputs/audit/v91_phase4_scene_risk_materialization/risk_rows.csv",
        ROOT / "outputs/audit/v91_phase4_adaptive_uncertainty_materialization/risk_rows.csv",
        ROOT / "outputs/audit/v91_phase4_adaptive_score_repair/risk_rows.csv",
        ROOT / "outputs/audit/v91_phase4_broad_core_precision_repair/risk_rows.csv",
        ROOT / "outputs/audit/v91_phase4_support_wta_repair/risk_rows.csv",
        ROOT / "outputs/audit/v91_phase4_semantic_witness_split_repair/risk_rows.csv",
        ROOT / "outputs/audit/v91_phase4_source_voronoi_partition_repair/risk_rows.csv",
        ROOT / "outputs/audit/v91_phase4_rgb_edge_boundary_repair/risk_rows.csv",
    ]:
        for row in _read_csv(path):
            out[row.get("variant_id", "")] = row
    return out


def _collect_rows() -> list[dict[str, Any]]:
    rows = _base_rows()
    rows.extend(
        _rows_from_control_csv(
            ROOT / "outputs/audit/v91_phase3_carrier_visible_support/control_metric_rows.csv",
            "ours_phase3_visible_support",
            "v91 Phase3 existing full-support audit, no rerun",
            limit_prefix="A",
        )
    )
    rows.extend(
        _rows_from_control_csv(
            ROOT / "outputs/audit/v91_phase4_witness_cover_ap50_control_repair/control_metric_rows.csv",
            "ours_phase4_ap50_control_repair",
            "v91 Phase4 AP50/control risk-residual repair",
        )
    )
    for path, family, notes in [
        (
            ROOT / "outputs/audit/v91_phase4_witness_cover_radius_sweep/control_metric_rows.csv",
            "ours_phase4_radius_sweep_coarse",
            "v91 Phase4 W9a radius coarse sweep after source-row filter fix",
        ),
        (
            ROOT / "outputs/audit/v91_phase4_witness_cover_radius_sweep_fine_radius/control_metric_rows.csv",
            "ours_phase4_radius_sweep_fine",
            "v91 Phase4 W9a fine-radius sweep",
        ),
        (
            ROOT / "outputs/audit/v91_phase4_witness_cover_radius_sweep_support_point/control_metric_rows.csv",
            "ours_phase4_radius_sweep_support_point",
            "v91 Phase4 W9a support-point-radius sweep",
        ),
        (
            ROOT / "outputs/audit/v91_phase4_witness_cover_multimask_materialization/control_metric_rows.csv",
            "ours_phase4_multimask_materialization",
            "v91 Phase4 W8a multi-masklet materialization repair",
        ),
        (
            ROOT / "outputs/audit/v91_phase4_witness_cover_multimask_score_repair/control_metric_rows.csv",
            "ours_phase4_multimask_score_repair",
            "v91 Phase4 D4RT residual/risk guard repair on best multi-mask materialization",
        ),
        (
            ROOT / "outputs/audit/v91_phase4_scene_risk_materialization/control_metric_rows.csv",
            "ours_phase4_scene_risk_materialization",
            "v91 Phase4 GT-free scene-risk-conditioned multi-mask materialization repair",
        ),
        (
            ROOT / "outputs/audit/v91_phase4_adaptive_uncertainty_materialization/control_metric_rows.csv",
            "ours_phase4_adaptive_uncertainty_materialization",
            "v91 Phase4 adaptive D4RT uncertainty dilation materialization repair",
        ),
        (
            ROOT / "outputs/audit/v91_phase4_adaptive_score_repair/control_metric_rows.csv",
            "ours_phase4_adaptive_score_repair",
            "v91 Phase4 D4RT residual/support score repair on best adaptive materialization",
        ),
        (
            ROOT / "outputs/audit/v91_phase4_broad_core_precision_repair/control_metric_rows.csv",
            "ours_phase4_broad_core_precision_repair",
            "v91 continued Phase4 high-risk broad-mask carrier-core precision repair",
        ),
        (
            ROOT / "outputs/audit/v91_phase4_support_wta_repair/control_metric_rows.csv",
            "ours_phase4_support_wta_repair",
            "v91 continued Phase4 same-frame same-mask D4RT support-consistency WTA repair",
        ),
        (
            ROOT / "outputs/audit/v91_phase4_semantic_witness_split_repair/control_metric_rows.csv",
            "ours_phase4_semantic_witness_split_repair",
            "v91 continued Phase4 GT-free geo-semantic witness object-tube split repair",
        ),
        (
            ROOT / "outputs/audit/v91_phase4_source_voronoi_partition_repair/control_metric_rows.csv",
            "ours_phase4_source_voronoi_partition_repair",
            "v91 continued Phase4 source-mask carrier-seed Voronoi partition repair",
        ),
        (
            ROOT / "outputs/audit/v91_phase4_rgb_edge_boundary_repair/control_metric_rows.csv",
            "ours_phase4_rgb_edge_boundary_repair",
            "v91 continued Phase4 legal RGB edge-boundary source-mask readout repair",
        ),
    ]:
        rows.extend(_rows_from_control_csv(path, family, notes))
    risk_by_variant = _risk_lookup()
    for row in rows:
        risk = risk_by_variant.get(str(row.get("variant_id", "")), {})
        if risk:
            row["risk_penalty_mean"] = risk.get("risk_penalty_mean", row.get("risk_penalty_mean", ""))
            row["risk_safe_vs_B0_proxy"] = risk.get("risk_safe_vs_B0_proxy", row.get("risk_safe_vs_B0_proxy", ""))
            row["risk_filter_mode"] = risk.get("risk_filter_mode", "")
            row["score_mode"] = risk.get("score_mode", "")
    dedup: dict[str, dict[str, Any]] = {}
    for row in rows:
        dedup[str(row.get("variant_id", ""))] = row
    return list(dedup.values())


def _is_real_candidate(row: dict[str, Any]) -> bool:
    family = str(row.get("method_family", ""))
    variant = str(row.get("variant_id", ""))
    if family in {"control", "baseline_real", "stream3d_local_diagnostic"}:
        return False
    if variant.startswith("C0_") or variant.startswith("P3_"):
        return False
    if variant == "B0_local_only" or variant == "S3D_L1_local_merged_masks":
        return False
    return True


def _gate(row: dict[str, Any], b0: dict[str, Any], control: dict[str, Any]) -> tuple[dict[str, bool], dict[str, float]]:
    mv_ap = _num(row.get("mean_MV_AP_window"))
    mv_ap50 = _num(row.get("mean_MV_AP50_window"))
    mv_ap25 = _num(row.get("mean_MV_AP25_window"))
    margins = {
        "real_minus_B0_MV_AP_window": mv_ap - _num(b0.get("mean_MV_AP_window")),
        "real_minus_B0_MV_AP50_window": mv_ap50 - _num(b0.get("mean_MV_AP50_window")),
        "real_minus_best_control_MV_AP_window": mv_ap - _num(control.get("mean_MV_AP_window")),
        "real_minus_best_control_MV_AP50_window": mv_ap50 - _num(control.get("mean_MV_AP50_window")),
        "gap_to_required_MV_AP50_window": mv_ap50 - (_num(control.get("mean_MV_AP50_window")) + 0.010),
        "MV_AP25_window": mv_ap25,
    }
    checks = {
        "best_real_MV_AP_window_ge_B0_plus_0p010": mv_ap >= _num(b0.get("mean_MV_AP_window")) + 0.010,
        "best_real_MV_AP50_window_ge_B0_plus_0p020": mv_ap50 >= _num(b0.get("mean_MV_AP50_window")) + 0.020,
        "best_real_MV_AP_window_ge_control_plus_0p005": mv_ap >= _num(control.get("mean_MV_AP_window")) + 0.005,
        "best_real_MV_AP50_window_ge_control_plus_0p010": mv_ap50 >= _num(control.get("mean_MV_AP50_window")) + 0.010,
        "same_frame_collision_count_eq_0": _int(row.get("same_frame_collision_count")) == 0,
        "missing_mask_raster_count_eq_0": _int(row.get("missing_mask_raster_count")) == 0,
        "uses_gt_for_prediction_false": not _bool(row.get("uses_gt_for_prediction")),
        "uses_future_false": not _bool(row.get("uses_future")),
    }
    return checks, margins


def run() -> dict[str, Any]:
    started = time.time()
    rows = _collect_rows()
    b0 = next((row for row in rows if row.get("variant_id") == "B0_local_only"), {})
    control_candidates = [row for row in rows if str(row.get("method_family")) == "control" or str(row.get("variant_id")).startswith("P3_")]
    control = max(control_candidates, key=lambda row: _num(row.get("mean_MV_AP50_window")), default={})
    real_rows = [row for row in rows if _is_real_candidate(row)]
    scored_rows: list[dict[str, Any]] = []
    for row in real_rows:
        checks, margins = _gate(row, b0, control)
        scored_rows.append(
            {
                **row,
                **margins,
                **{f"gate_{key}": value for key, value in checks.items()},
                "v91_phase8_progress_gate_pass": all(checks.values()),
                "dev_gate_min_margin": min(
                    margins["real_minus_B0_MV_AP_window"] - 0.010,
                    margins["real_minus_B0_MV_AP50_window"] - 0.020,
                    margins["real_minus_best_control_MV_AP_window"] - 0.005,
                    margins["real_minus_best_control_MV_AP50_window"] - 0.010,
                ),
            }
        )
    best_real = max(
        scored_rows,
        key=lambda row: (
            _num(row.get("dev_gate_min_margin"), -999.0),
            _num(row.get("mean_MV_AP50_window"), -999.0),
            _num(row.get("mean_MV_AP_window"), -999.0),
        ),
        default={},
    )
    best_checks, best_margins = _gate(best_real, b0, control) if best_real else ({}, {})
    all_rows = []
    for row in rows:
        all_rows.append(scored_rows[[r.get("variant_id") for r in scored_rows].index(row.get("variant_id"))] if row.get("variant_id") in {r.get("variant_id") for r in scored_rows} else row)
    control_comparison = [
        {
            "best_real_variant": best_real.get("variant_id", ""),
            "best_real_MV_AP_window": best_real.get("mean_MV_AP_window", ""),
            "best_real_MV_AP50_window": best_real.get("mean_MV_AP50_window", ""),
            "best_control_variant": control.get("variant_id", ""),
            "best_control_MV_AP_window": control.get("mean_MV_AP_window", ""),
            "best_control_MV_AP50_window": control.get("mean_MV_AP50_window", ""),
            **best_margins,
        }
    ]
    frozen = {
        "frozen": bool(best_real) and all(best_checks.values()),
        "variant_id": best_real.get("variant_id", ""),
        "source_artifact": best_real.get("source_artifact", ""),
        "reason": "dev gate passed" if best_real and all(best_checks.values()) else "dev gate failed; holdout must not run",
        "config_sha256": "",
        "uses_gt_for_prediction": best_real.get("uses_gt_for_prediction", ""),
        "uses_future": best_real.get("uses_future", ""),
    }
    strong_gate = {
        "best_real_MV_AP_window_ge_0p060": _num(best_real.get("mean_MV_AP_window")) >= 0.060,
        "best_real_MV_AP50_window_ge_0p160": _num(best_real.get("mean_MV_AP50_window")) >= 0.160,
        "best_real_MV_AP25_window_ge_0p380": _num(best_real.get("mean_MV_AP25_window")) >= 0.380,
        "best_real_MV_AP_window_ge_control_plus_0p005": best_margins.get("real_minus_best_control_MV_AP_window", -999.0) >= 0.005,
        "risk_safe": _bool(best_real.get("risk_safe_vs_B0_proxy")) or _num(best_real.get("risk_penalty_mean"), 1.0) <= 0.6680042860969729,
    }
    summary = {
        "phase": "v91_phase8_dev_selection",
        "schema": "stream4d_v91_phase8_dev_selection_refresh_v1",
        "refresh_reason": "include v91 Phase3/Phase4 repair variants discovered after initial lock",
        "B0_variant": b0.get("variant_id", ""),
        "best_control_variant": control.get("variant_id", ""),
        "best_control_MV_AP_window": _num(control.get("mean_MV_AP_window")),
        "best_control_MV_AP50_window": _num(control.get("mean_MV_AP50_window")),
        "best_real_variant": best_real.get("variant_id", ""),
        "best_real_MV_AP_window": _num(best_real.get("mean_MV_AP_window")),
        "best_real_MV_AP50_window": _num(best_real.get("mean_MV_AP50_window")),
        "best_real_MV_AP25_window": _num(best_real.get("mean_MV_AP25_window")),
        "dev_progress_gate": best_checks,
        "dev_progress_gate_pass": bool(best_real) and all(best_checks.values()),
        "strong_local_dev_gate": strong_gate,
        "strong_local_dev_gate_pass": all(strong_gate.values()),
        "decision": "DEV_GATE_PASS_READY_FOR_PHASE9_FREEZE" if best_real and all(best_checks.values()) else "DEV_GATE_FAIL_CONTINUE_REPAIR",
        "next_action": "run frozen holdout once" if best_real and all(best_checks.values()) else "continue CONTROL_BIAS/EXTENT repair; no holdout yet",
        "row_counts": {
            "all_variant_metric_rows": len(all_rows),
            "real_candidate_rows": len(scored_rows),
        },
        "runtime_sec": time.time() - started,
    }
    _write_csv(OUT8 / "all_variant_metric_rows.csv", all_rows)
    _write_csv(OUT8 / "control_comparison_rows.csv", control_comparison)
    _write_csv(OUT8 / "casebook_rows.csv", scored_rows[:])
    _write_json(OUT8 / "frozen_candidate_config.json", frozen)
    _write_json(OUT8 / "summary.json", summary)
    outputs = [
        OUT8 / "all_variant_metric_rows.csv",
        OUT8 / "control_comparison_rows.csv",
        OUT8 / "casebook_rows.csv",
        OUT8 / "frozen_candidate_config.json",
        OUT8 / "summary.json",
    ]
    _write_json(OUT8 / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run()
