from __future__ import annotations

import argparse
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_v89_recalc_point_projected_mv_ap as recalc  # noqa: E402
from tools import run_v90_dev_extent_score_cross_audit as phase7d  # noqa: E402
from tools import run_v90_geo_semantic_witness_cover as phase4  # noqa: E402
from tools import run_v90_mv_ap_window_resurrection as phase1  # noqa: E402
from tools import run_v91_phase4_ap50_control_repair as v91repair  # noqa: E402


OUT_BASE = ROOT / "outputs/audit/v91_phase4_witness_cover_radius_sweep"
OUT = OUT_BASE
V90_PHASE4 = ROOT / "outputs/audit/v90_phase4_geo_semantic_witness_cover"
SUPPORT_ROWS = ROOT / "outputs/audit/v90_phase3_v82_full_support/native_carrier_support_rows.csv"
SOURCE_VARIANT = "W8a_risk_balanced_p135_witnesses"


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


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def _variant_specs(sweep: str) -> list[dict[str, Any]]:
    if sweep == "coarse":
        return [
            {"variant_id": "V91_P4R1_W9a_carve_r08_drop5_sceneorig", "radius": 8, "support_point_radius": 3},
            {"variant_id": "V91_P4R2_W9a_carve_r12_drop5_sceneorig", "radius": 12, "support_point_radius": 3},
            {"variant_id": "V91_P4R3_W9a_carve_r16_drop5_sceneorig", "radius": 16, "support_point_radius": 3},
            {"variant_id": "V91_P4R4_W9a_carve_r20_drop5_sceneorig", "radius": 20, "support_point_radius": 3},
            {"variant_id": "V91_P4R5_W9a_carve_r24_drop5_sceneorig", "radius": 24, "support_point_radius": 3},
        ]
    if sweep == "fine_radius":
        return [
            {"variant_id": "V91_P4F1_W9a_carve_r13_drop5_sceneorig", "radius": 13, "support_point_radius": 3},
            {"variant_id": "V91_P4F2_W9a_carve_r14_drop5_sceneorig", "radius": 14, "support_point_radius": 3},
            {"variant_id": "V91_P4F3_W9a_carve_r15_drop5_sceneorig", "radius": 15, "support_point_radius": 3},
            {"variant_id": "V91_P4F4_W9a_carve_r17_drop5_sceneorig", "radius": 17, "support_point_radius": 3},
            {"variant_id": "V91_P4F5_W9a_carve_r18_drop5_sceneorig", "radius": 18, "support_point_radius": 3},
        ]
    if sweep == "support_point":
        return [
            {"variant_id": "V91_P4S1_W9a_carve_r16_spr1_drop5_sceneorig", "radius": 16, "support_point_radius": 1},
            {"variant_id": "V91_P4S2_W9a_carve_r16_spr2_drop5_sceneorig", "radius": 16, "support_point_radius": 2},
            {"variant_id": "V91_P4S3_W9a_carve_r16_spr4_drop5_sceneorig", "radius": 16, "support_point_radius": 4},
            {"variant_id": "V91_P4S4_W9a_carve_r16_spr5_drop5_sceneorig", "radius": 16, "support_point_radius": 5},
            {"variant_id": "V91_P4S5_W9a_carve_r16_spr7_drop5_sceneorig", "radius": 16, "support_point_radius": 7},
        ]
    raise ValueError(sweep)


def _output_root(sweep: str) -> Path:
    return OUT_BASE if sweep == "coarse" else ROOT / f"outputs/audit/v91_phase4_witness_cover_radius_sweep_{sweep}"


def _feature_map(selection_rows: list[dict[str, Any]], generated_rows: list[dict[str, Any]]) -> dict[tuple[str, int, int], dict[str, Any]]:
    generated_by_key = {
        (
            row.get("scene_id", ""),
            _int(row.get("frame_id"), -1),
            _int(row.get("new_mask_id"), -1),
        ): row
        for row in generated_rows
    }
    out: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in selection_rows:
        key = (row.get("scene_id", ""), _int(row.get("frame_id"), -1), _int(row.get("new_mask_id"), -1))
        if not key[0] or key[1] < 0 or key[2] <= 0:
            continue
        gen = generated_by_key.get(key, {})
        feat = dict(row)
        feat.update(
            {
                "support_count": row.get("support_count", gen.get("carrier_support_count", "")),
                "support_area": gen.get("support_area", row.get("support_area", "")),
                "source_mask_area": gen.get("source_mask_area", row.get("source_mask_area", "")),
                "generated_mask_area": gen.get("generated_mask_area", row.get("generated_mask_area", "")),
                "area_ratio": _num(gen.get("generated_mask_area"), _num(row.get("generated_mask_area"))) / 1254528.0,
                "broad_background_risk": row.get("broad_background_risk", False),
            }
        )
        out[key] = feat
    return out


def _evaluate_variant(variant_id: str, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    original_mask_dir = recalc._mask_dir
    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    scope = recalc._frame_scope()
    try:
        recalc._mask_dir = lambda scene, _variant=variant_id: OUT / "generated_masks" / _variant / scene / "mask"
        for scene in ["scene0011_00", "scene0050_00"]:
            scene_rows = [row for row in rows if row.get("scene_id") == scene]
            if not scene_rows:
                continue
            metric, cases, _tops, _window_rows = recalc._evaluate_frame_mask_variant_local_window(
                scene=scene,
                split="dev",
                variant=variant_id,
                frame_ids=scope.get(("dev", scene)),
                rows=scene_rows,
                score_mode="input",
                local_export_root=ROOT / "outputs/audit/v89_recalc_point_projected_mv_ap",
                window_source_step="S3D_L1_local_merged_masks",
            )
            metric_rows.append(
                {
                    **metric,
                    "variant_id": variant_id,
                    "MV_AP_window": metric.get("MV_AP"),
                    "MV_AP50_window": metric.get("MV_AP50"),
                    "MV_AP25_window": metric.get("MV_AP25"),
                    "score_free_Match50_window": phase1._f1(metric.get("SF50_precision"), metric.get("SF50_recall")),
                    "score_free_Match50_precision_window": metric.get("SF50_precision"),
                    "score_free_Match50_recall_window": metric.get("SF50_recall"),
                    "same_frame_collision_count": int(_int(metric.get("duplicate_frame_mask_conflict_count"), 0)),
                    "metric_scope": "local_window_gt_projection",
                }
            )
            case_rows.extend(cases)
    finally:
        recalc._mask_dir = original_mask_dir
    return metric_rows, case_rows


def _support_quality_rows(rows: list[dict[str, Any]], feature_map: dict[tuple[str, int, int], dict[str, Any]], variant_id: str) -> list[dict[str, Any]]:
    return v91repair._support_quality_rows(rows, feature_map, variant_id)


def run(args: argparse.Namespace) -> dict[str, Any]:
    global OUT
    started = time.time()
    OUT = _output_root(args.sweep)
    OUT.mkdir(parents=True, exist_ok=True)
    phase4.OUT = OUT
    mask_dirs = phase4._mask_dir_by_scene()
    semantic_features = phase4._load_semantic_features()
    source_rows = [
        row
        for row in _read_csv(V90_PHASE4 / "witness_cover_selection_rows.csv")
        if row.get("variant_id") == SOURCE_VARIANT
        and not _bool(row.get("conflict_dropped"))
        and not row.get("new_mask_id")
    ]
    source_slots = {(row.get("scene_id", ""), row.get("local_slot_id", "")) for row in source_rows}
    _candidates, support_points = phase4._load_support_candidates(SUPPORT_ROWS, source_slots, semantic_features, mask_dirs)
    baselines = v91repair._phase8_baselines()

    config_rows: list[dict[str, Any]] = []
    generated_rows_all: list[dict[str, Any]] = []
    selected_rows_all: list[dict[str, Any]] = []
    eval_rows_all: list[dict[str, Any]] = []
    scored_rows_all: list[dict[str, Any]] = []
    support_quality_all: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []

    for spec in _variant_specs(args.sweep):
        variant_id = str(spec["variant_id"])
        generated_rows, selected_rows, eval_rows = phase4._generate_carved_masks(
            source_rows,
            support_points,
            mask_dirs,
            radius=int(spec["radius"]),
            support_point_radius=int(spec["support_point_radius"]),
            variant=variant_id,
            source_variant=SOURCE_VARIANT,
        )
        feature_map = _feature_map(selected_rows, generated_rows)
        scored_rows, keep_flags = v91repair._variant_rows(
            eval_rows,
            feature_map,
            variant_id=variant_id,
            risk_mode="drop_broad_low_h9_5",
            score_mode="broad_scene_orig_ge065",
            group_name="radius_sweep",
        )
        support_quality = _support_quality_rows(scored_rows, feature_map, variant_id)
        metrics, cases = _evaluate_variant(variant_id, scored_rows)
        generated_rows_all.extend(generated_rows)
        selected_rows_all.extend(selected_rows)
        eval_rows_all.extend(eval_rows)
        scored_rows_all.extend(scored_rows)
        support_quality_all.extend(support_quality)
        metric_rows.extend(metrics)
        case_rows.extend({**row, "variant_id": variant_id} for row in cases)
        broad_values = [1.0 if _bool(row.get("broad_risk")) else 0.0 for row in support_quality]
        risk_rows.append(
            {
                "variant_id": variant_id,
                "radius": int(spec["radius"]),
                "support_point_radius": int(spec["support_point_radius"]),
                "risk_filter_mode": "drop_broad_low_h9_5",
                "score_mode": "broad_scene_orig_ge065",
                "selected_rows": len(scored_rows),
                "pre_filter_eval_rows": len(eval_rows),
                "dropped_rows": len(eval_rows) - int(sum(1 for flag in keep_flags if flag)),
                "risk_penalty_mean": _mean(broad_values),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        config_rows.append(
            {
                "variant_id": variant_id,
                "parent_source_variant": SOURCE_VARIANT,
                "changed_module": "phase4_witness_cover_connected_component_carving_radius",
                "changed_parameters": f"carving_radius={spec['radius']}; support_point_radius={spec['support_point_radius']}; risk_filter=drop_broad_low_h9_5; score=broad_scene_orig_ge065",
                "reason_for_change": "v91 Phase8 AP50/control failure; test whether W9a extent radius changes improve AP50 while keeping the strongest GT-free score protocol",
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "dev_only_or_holdout": "dev_only",
                "expected_blocker": "CONTROL_BIAS_BLOCKER+EXTENT_BLOCKER",
            }
        )

    aggregate_rows = phase7d._aggregate(metric_rows)
    control_rows = v91repair._add_gate_rows(aggregate_rows, baselines)
    best = max(
        control_rows,
        key=lambda row: (
            _num(row.get("dev_gate_min_margin"), -999.0),
            _num(row.get("mean_MV_AP50_window"), -999.0),
            _num(row.get("mean_MV_AP_window"), -999.0),
        ),
        default={},
    )
    passing = [row for row in control_rows if _bool(row.get("v91_phase8_progress_gate_pass"))]
    summary = {
        "phase": "v91_phase4_witness_cover_radius_sweep",
        "schema": "stream4d_v91_phase4_radius_sweep_v1",
        "sweep": args.sweep,
        "source_variant": SOURCE_VARIANT,
        "variant_count": len(config_rows),
        "best_variant_id": best.get("variant_id", ""),
        "best_variant_gate": best,
        "any_v91_phase8_progress_gate_pass": bool(passing),
        "decision": "DEV_GATE_PASS_READY_FOR_PHASE8_FREEZE" if passing else "DEV_GATE_FAIL_CONTINUE_PHASE3_4_REPAIR",
        "next_action": "freeze candidate in Phase8 before holdout" if passing else "continue extent/materialization repair; do not run Phase9 holdout",
        "row_counts": {
            "variant_config_rows": len(config_rows),
            "generated_mask_rows": len(generated_rows_all),
            "selected_masklet_rows": len(selected_rows_all),
            "pre_filter_eval_rows": len(eval_rows_all),
            "scored_frame_mask_rows": len(scored_rows_all),
            "support_quality_rows": len(support_quality_all),
            "mv_metric_rows": len(metric_rows),
            "control_metric_rows": len(control_rows),
            "casebook_rows": len(case_rows),
        },
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "holdout_policy": "No holdout is used or touched by this dev-only radius sweep.",
        "runtime_sec": time.time() - started,
    }
    _write_csv(OUT / "variant_config_rows.csv", config_rows)
    _write_csv(OUT / "generated_mask_rows.csv", generated_rows_all)
    _write_csv(OUT / "selected_masklet_rows.csv", selected_rows_all)
    _write_csv(OUT / "mv_object_frame_mask_rows.csv", scored_rows_all)
    _write_csv(OUT / "support_quality_rows.csv", support_quality_all)
    _write_csv(OUT / "risk_rows.csv", risk_rows)
    _write_csv(OUT / "mv_metric_rows.csv", metric_rows)
    _write_csv(OUT / "mv_metric_aggregate_rows.csv", aggregate_rows)
    _write_csv(OUT / "control_metric_rows.csv", control_rows)
    _write_csv(OUT / "casebook_rows.csv", case_rows)
    _write_json(OUT / "summary.json", summary)
    outputs = [
        OUT / "variant_config_rows.csv",
        OUT / "generated_mask_rows.csv",
        OUT / "selected_masklet_rows.csv",
        OUT / "mv_object_frame_mask_rows.csv",
        OUT / "support_quality_rows.csv",
        OUT / "risk_rows.csv",
        OUT / "mv_metric_rows.csv",
        OUT / "mv_metric_aggregate_rows.csv",
        OUT / "control_metric_rows.csv",
        OUT / "casebook_rows.csv",
        OUT / "summary.json",
    ]
    _write_json(OUT / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v91 Phase4 W9a carving radius sweep on dev only.")
    parser.add_argument("--sweep", choices=["coarse", "fine_radius", "support_point"], default="coarse")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
