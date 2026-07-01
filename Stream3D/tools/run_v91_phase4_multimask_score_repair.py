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
from tools import run_v90_mv_ap_window_resurrection as phase1  # noqa: E402
from tools import run_v91_phase4_ap50_control_repair as v91repair  # noqa: E402
from tools import run_v91_phase4_radius_sweep as radius_sweep  # noqa: E402


OUT = ROOT / "outputs/audit/v91_phase4_witness_cover_multimask_score_repair"
MATERIALIZATION_ROOT = ROOT / "outputs/audit/v91_phase4_witness_cover_multimask_materialization"
PARENT_VARIANT = "V91_M1_W8a_top2_r16_drop5_sceneorig"


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _variant_specs() -> list[dict[str, str]]:
    return [
        {"variant_id": "V91_MS1_M1_drop2_sceneorig", "risk_mode": "drop_broad_low_h9_2", "score_mode": "broad_scene_orig_ge065"},
        {"variant_id": "V91_MS2_M1_all_sceneorig", "risk_mode": "all", "score_mode": "broad_scene_orig_ge065"},
        {"variant_id": "V91_MS3_M1_drop5_hardneg_residual_scene", "risk_mode": "drop_broad_low_h9_5", "score_mode": "hard_negative_residual_scene"},
        {"variant_id": "V91_MS4_M1_drop5_support_consistency", "risk_mode": "drop_broad_low_h9_5", "score_mode": "support_consistency"},
        {"variant_id": "V91_MS5_M1_drop5_support_residual_scene", "risk_mode": "drop_broad_low_h9_5", "score_mode": "support_consistency_residual_scene"},
    ]


def _base_eval_rows(selected_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in selected_rows:
        if row.get("variant_id") != PARENT_VARIANT:
            continue
        rows.append(
            {
                "split": "dev",
                "scene_id": row.get("scene_id", ""),
                "source_variant": PARENT_VARIANT,
                "variant": PARENT_VARIANT,
                "mv_object_id": row.get("mv_object_id", ""),
                "frame_id": _int(row.get("frame_id"), -1),
                "mask_id": _int(row.get("new_mask_id"), -1),
                "frame_mask_score": _num(row.get("selection_score"), 1.0),
                "object_score": _num(row.get("selection_score"), 1.0),
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "uses_rgbd_pose_mesh": False,
                "materializable": True,
                "selection_reason": "v91_multimask_parent_eval_row_for_score_repair",
            }
        )
    return rows


def _rescore_rows(
    base_rows: list[dict[str, Any]],
    feature_map: dict[tuple[str, int, int], dict[str, Any]],
    *,
    variant_id: str,
    risk_mode: str,
    score_mode: str,
) -> tuple[list[dict[str, Any]], list[bool]]:
    rows, keep_flags = v91repair._variant_rows(
        base_rows,
        feature_map,
        variant_id=variant_id,
        risk_mode=risk_mode,
        score_mode=score_mode,
        group_name="multimask_score_repair",
    )
    for row in rows:
        obj = str(row.get("mv_object_id", ""))
        if obj.startswith(f"{PARENT_VARIANT}:"):
            row["mv_object_id"] = obj.replace(f"{PARENT_VARIANT}:", f"{variant_id}:", 1)
        row["base_extent_variant"] = PARENT_VARIANT
        row["selection_reason"] = f"v91_phase4_multimask_score_repair_{risk_mode}_{score_mode}_from_{PARENT_VARIANT}"
    return rows, keep_flags


def _evaluate_variant(variant_id: str, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    original_mask_dir = recalc._mask_dir
    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    scope = recalc._frame_scope()
    try:
        recalc._mask_dir = lambda scene, _parent=PARENT_VARIANT: MATERIALIZATION_ROOT / "generated_masks" / _parent / scene / "mask"
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
                    "base_extent_variant": PARENT_VARIANT,
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


def run(_args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    selected_parent_rows = _read_csv(MATERIALIZATION_ROOT / "selected_masklet_rows.csv")
    generated_parent_rows = _read_csv(MATERIALIZATION_ROOT / "generated_mask_rows.csv")
    selected_rows_parent_variant = [row for row in selected_parent_rows if row.get("variant_id") == PARENT_VARIANT]
    generated_rows_parent_variant = [row for row in generated_parent_rows if row.get("variant_id") == PARENT_VARIANT]
    feature_map = radius_sweep._feature_map(selected_rows_parent_variant, generated_rows_parent_variant)
    base_rows = _base_eval_rows(selected_rows_parent_variant)
    baselines = v91repair._phase8_baselines()

    config_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []

    for spec in _variant_specs():
        variant_id = spec["variant_id"]
        risk_mode = spec["risk_mode"]
        score_mode = spec["score_mode"]
        rows, keep_flags = _rescore_rows(base_rows, feature_map, variant_id=variant_id, risk_mode=risk_mode, score_mode=score_mode)
        selected_rows.extend(rows)
        support_for_variant = v91repair._support_quality_rows(rows, feature_map, variant_id)
        support_rows.extend(support_for_variant)
        metrics, cases = _evaluate_variant(variant_id, rows)
        metric_rows.extend(metrics)
        case_rows.extend({**row, "variant_id": variant_id} for row in cases)
        broad_values = [1.0 if _bool(row.get("broad_risk")) else 0.0 for row in support_for_variant]
        risk_rows.append(
            {
                "variant_id": variant_id,
                "parent_variant_id": PARENT_VARIANT,
                "risk_filter_mode": risk_mode,
                "score_mode": score_mode,
                "selected_rows": len(rows),
                "parent_pre_filter_rows": len(base_rows),
                "dropped_rows": len(base_rows) - int(sum(1 for flag in keep_flags if flag)),
                "risk_penalty_mean": _mean(broad_values),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        config_rows.append(
            {
                "variant_id": variant_id,
                "parent_variant_id": PARENT_VARIANT,
                "changed_module": "phase4_multimask_d4rt_residual_risk_guard_score",
                "changed_parameters": f"risk_filter_mode={risk_mode}; score_mode={score_mode}",
                "reason_for_change": "v91 Phase8 M1 still misses AP50 control-margin gate; try D4RT residual/risk guard variants on best multi-mask materialization",
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "dev_only_or_holdout": "dev_only",
                "expected_blocker": "CONTROL_BIAS_BLOCKER",
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
        "phase": "v91_phase4_witness_cover_multimask_score_repair",
        "schema": "stream4d_v91_phase4_multimask_score_repair_v1",
        "parent_variant_id": PARENT_VARIANT,
        "variant_count": len(config_rows),
        "best_variant_id": best.get("variant_id", ""),
        "best_variant_gate": best,
        "any_v91_phase8_progress_gate_pass": bool(passing),
        "decision": "DEV_GATE_PASS_READY_FOR_PHASE8_FREEZE" if passing else "DEV_GATE_FAIL_CONTINUE_REPAIR",
        "next_action": "refresh Phase8 and freeze before holdout" if passing else "continue diagnosis/repair; do not run Phase9 holdout",
        "row_counts": {
            "variant_config_rows": len(config_rows),
            "parent_pre_filter_rows": len(base_rows),
            "selected_masklet_rows": len(selected_rows),
            "support_quality_rows": len(support_rows),
            "mv_metric_rows": len(metric_rows),
            "control_metric_rows": len(control_rows),
            "casebook_rows": len(case_rows),
        },
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "holdout_policy": "No holdout is used or touched by this dev-only score/risk repair.",
        "runtime_sec": time.time() - started,
    }
    _write_csv(OUT / "variant_config_rows.csv", config_rows)
    _write_csv(OUT / "selected_masklet_rows.csv", selected_rows)
    _write_csv(OUT / "support_quality_rows.csv", support_rows)
    _write_csv(OUT / "risk_rows.csv", risk_rows)
    _write_csv(OUT / "mv_metric_rows.csv", metric_rows)
    _write_csv(OUT / "mv_metric_aggregate_rows.csv", aggregate_rows)
    _write_csv(OUT / "control_metric_rows.csv", control_rows)
    _write_csv(OUT / "casebook_rows.csv", case_rows)
    _write_json(OUT / "summary.json", summary)
    outputs = [
        OUT / "variant_config_rows.csv",
        OUT / "selected_masklet_rows.csv",
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
    parser = argparse.ArgumentParser(description="Run v91 Phase4 score/risk repair on best multi-mask materialization.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
