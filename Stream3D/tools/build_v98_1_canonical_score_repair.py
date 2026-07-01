#!/usr/bin/env python3
"""Run a small v98.1 canonical score-repair family under the v90 metric."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_v98_1_canonical_mv_metrics as canonical  # noqa: E402


PHASE9 = ROOT / "outputs/audit/v98_phase9_render_snap"
PHASE10 = ROOT / "outputs/audit/v98_phase10_controls"
PHASE11 = ROOT / "outputs/audit/v98_phase11_failure_decomposition"
PHASE12 = ROOT / "outputs/audit/v98_phase12_dev_decision"
INPUT_ROWS = PHASE9 / "mv_object_frame_mask_rows.csv"
RUN_ID = "v98_1_canonical_score_repair"

B0_MV_AP_WINDOW = canonical.B0_MV_AP_WINDOW
B0_MV_AP50_WINDOW = canonical.B0_MV_AP50_WINDOW
BEST_LOCKED_CONTROL_VARIANT = canonical.BEST_LOCKED_CONTROL_VARIANT
BEST_LOCKED_CONTROL_MV_AP_WINDOW = canonical.BEST_LOCKED_CONTROL_MV_AP_WINDOW
BEST_LOCKED_CONTROL_MV_AP50_WINDOW = canonical.BEST_LOCKED_CONTROL_MV_AP50_WINDOW
V91_BEST_MV_AP_WINDOW = canonical.V91_BEST_MV_AP_WINDOW

BASE_VARIANT = "F2_mask_centered_plus_semantic_residual_proxy"
SCORE_POLICIES = [
    "frame_count",
    "frame_count_x_support_iou",
    "frame_count_x_mask_precision",
    "support_iou_x_mask_precision",
    "frame_count_x_support_area_log",
]


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in keys})


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def _load_rows(path: Path, base_variant: str) -> list[dict[str, Any]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("variant_id") == base_variant:
                rows.append(row)
    return rows


def _score_rows(rows: list[dict[str, Any]], base_variant: str, policy: str) -> list[dict[str, Any]]:
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_object[row.get("mv_object_id", row.get("object_id", ""))].append(row)
    stats: dict[str, dict[str, float]] = {}
    max_frame_count = 1.0
    max_area = 1.0
    for oid, vals in by_object.items():
        frames = {(row.get("scene_id", ""), int(_num(row.get("frame_id")))) for row in vals}
        frame_count = float(len(frames))
        support_iou = _mean([_num(row.get("support_iou")) for row in vals])
        mask_precision = _mean([_num(row.get("mask_precision")) for row in vals])
        support_area = _mean([_num(row.get("support_area")) for row in vals])
        original = max(_num(row.get("score"), 1.0) for row in vals)
        max_frame_count = max(max_frame_count, frame_count)
        max_area = max(max_area, support_area)
        stats[oid] = {
            "frame_count": frame_count,
            "support_iou_mean": support_iou,
            "mask_precision_mean": mask_precision,
            "support_area_mean": support_area,
            "original_score_max": original,
        }
    out: list[dict[str, Any]] = []
    variant_id = f"{base_variant}__score_{policy}"
    for row in rows:
        oid = row.get("mv_object_id", row.get("object_id", ""))
        st = stats.get(oid, {})
        frame_norm = st.get("frame_count", 0.0) / max_frame_count
        support_iou = st.get("support_iou_mean", 0.0)
        mask_precision = st.get("mask_precision_mean", 0.0)
        area_log = math.log1p(st.get("support_area_mean", 0.0)) / max(1e-6, math.log1p(max_area))
        if policy == "frame_count":
            score = frame_norm
        elif policy == "frame_count_x_support_iou":
            score = frame_norm * support_iou
        elif policy == "frame_count_x_mask_precision":
            score = frame_norm * mask_precision
        elif policy == "support_iou_x_mask_precision":
            score = support_iou * mask_precision
        elif policy == "frame_count_x_support_area_log":
            score = frame_norm * area_log
        else:
            raise ValueError(f"unknown policy {policy}")
        out.append({**row, "variant_id": variant_id, "mv_object_id": oid, "object_id": oid, "score": float(score), "score_policy": policy})
    return out


def _evaluate_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    tmp = PHASE9 / "canonical_score_repair_input_rows.csv"
    _write_csv(tmp, rows)
    eval_rows = canonical._load_eval_rows(tmp)
    metric_rows, case_rows, top_rows = canonical._evaluate_window(eval_rows)
    aggregate_rows = canonical._aggregate_window(metric_rows)
    return metric_rows, aggregate_rows, top_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-rows", default=str(INPUT_ROWS))
    parser.add_argument("--base-variant", default=BASE_VARIANT)
    args = parser.parse_args()

    started = time.time()
    base_rows = _load_rows(Path(args.input_rows), args.base_variant)
    if not base_rows:
        raise RuntimeError(f"no rows for base variant {args.base_variant}")
    all_metric_rows: list[dict[str, Any]] = []
    all_aggregate_rows: list[dict[str, Any]] = []
    all_top_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    for policy in SCORE_POLICIES:
        scored = _score_rows(base_rows, args.base_variant, policy)
        metric_rows, aggregate_rows, top_rows = _evaluate_rows(scored)
        for row in metric_rows:
            row["score_repair_policy"] = policy
            row["run_id"] = RUN_ID
        for row in aggregate_rows:
            row["score_repair_policy"] = policy
            row["run_id"] = RUN_ID
        all_metric_rows.extend(metric_rows)
        all_aggregate_rows.extend(aggregate_rows)
        all_top_rows.extend({**row, "score_repair_policy": policy, "run_id": RUN_ID} for row in top_rows)
        config_rows.append(
            {
                "variant_id": f"{args.base_variant}__score_{policy}",
                "base_variant": args.base_variant,
                "score_policy": policy,
                "repair_family": "canonical_score_policy_gt_free",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )

    best = max(all_aggregate_rows, key=lambda row: (_num(row.get("mean_MV_AP_window"), -1), _num(row.get("mean_MV_AP50_window"), -1)), default={})
    best_ap = _num(best.get("mean_MV_AP_window"), -1.0)
    best_ap50 = _num(best.get("mean_MV_AP50_window"), -1.0)
    gates = {
        "best_real_MV_AP_window_ge_B0_plus_0p010": best_ap >= B0_MV_AP_WINDOW + 0.010,
        "best_real_MV_AP50_window_ge_B0_plus_0p020": best_ap50 >= B0_MV_AP50_WINDOW + 0.020,
        "best_real_MV_AP_window_ge_best_control_plus_0p005": best_ap >= BEST_LOCKED_CONTROL_MV_AP_WINDOW + 0.005,
        "best_real_MV_AP50_window_ge_best_control_plus_0p010": best_ap50 >= BEST_LOCKED_CONTROL_MV_AP50_WINDOW + 0.010,
        "best_real_MV_AP_window_ge_v91_plus_0p002": best_ap >= V91_BEST_MV_AP_WINDOW + 0.002,
        "same_frame_collision_count_eq_0": int(_num(best.get("same_frame_collision_count"), 1)) == 0,
        "missing_mask_raster_count_eq_0": int(_num(best.get("missing_mask_raster_count"), 1)) == 0,
    }
    dev_gate_pass = all(gates.values())
    decision = "GO_V98_DEV_LOCAL_MV_AP_WINDOW" if dev_gate_pass else "NO_GO_CONTROL_BIAS"
    PHASE9.mkdir(parents=True, exist_ok=True)
    _write_csv(PHASE9 / "canonical_score_repair_config_rows.csv", config_rows)
    _write_csv(PHASE9 / "canonical_score_repair_mv_metric_rows.csv", all_metric_rows)
    _write_csv(PHASE9 / "canonical_score_repair_mv_metric_aggregate_rows.csv", all_aggregate_rows)
    _write_csv(PHASE9 / "canonical_score_repair_top_iou_rows.csv", all_top_rows)
    summary = {
        "schema": "stream4d_v98_1_canonical_score_repair_summary_v1",
        "phase_id": "v98_phase9_render_snap",
        "run_id": RUN_ID,
        "created_at": _created_at(),
        "base_variant": args.base_variant,
        "repair_variant_count": len(SCORE_POLICIES),
        "best_variant": best.get("variant_id", ""),
        "best_score_policy": best.get("score_repair_policy", ""),
        "best_MV_AP_window": best.get("mean_MV_AP_window", ""),
        "best_MV_AP50_window": best.get("mean_MV_AP50_window", ""),
        "best_MV_AP_scene": "",
        "MV_AP_scene_status": "not_computed_by_default_v90_contract_leaves_MV_AP_scene_local2history_pending",
        "best_control_variant": BEST_LOCKED_CONTROL_VARIANT,
        "best_control_MV_AP_window": BEST_LOCKED_CONTROL_MV_AP_WINDOW,
        "best_control_MV_AP50_window": BEST_LOCKED_CONTROL_MV_AP50_WINDOW,
        "dev_gate_pass": dev_gate_pass,
        "decision": decision,
        "gate_results": gates,
        "runtime_sec": float(time.time() - started),
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_json(PHASE9 / "canonical_score_repair_summary.json", summary)
    if dev_gate_pass:
        _write_json(
            PHASE12 / "canonical_final_dev_decision.json",
            {
                "schema": "stream4d_v98_1_canonical_phase12_dev_decision_v1",
                "phase_id": "v98_phase12_dev_decision",
                "run_id": RUN_ID,
                "created_at": _created_at(),
                "decision": "GO_V98_DEV_LOCAL_MV_AP_WINDOW",
                "best_real_variant": best.get("variant_id", ""),
                "best_real_MV_AP_window": best.get("mean_MV_AP_window", ""),
                "best_real_MV_AP50_window": best.get("mean_MV_AP50_window", ""),
                "best_real_MV_AP_scene": "",
                "MV_AP_scene_status": "not_computed_by_default_v90_contract_leaves_MV_AP_scene_local2history_pending",
                "best_control_variant": BEST_LOCKED_CONTROL_VARIANT,
                "best_control_MV_AP_window": BEST_LOCKED_CONTROL_MV_AP_WINDOW,
                "best_control_MV_AP50_window": BEST_LOCKED_CONTROL_MV_AP50_WINDOW,
                "dev_gate_pass": True,
                "holdout_allowed": True,
                "local2history_allowed": False,
                "primary_blocker": "HOLDOUT_NOT_RUN",
                "gate_results": gates,
                "metric_source_window": "run_v89_recalc_point_projected_mv_ap._evaluate_frame_mask_variant_local_window",
                "metric_source_scene": "not_computed_by_default; v90 treats MV_AP_scene/local2history as pending after local-window gate",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            },
        )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
