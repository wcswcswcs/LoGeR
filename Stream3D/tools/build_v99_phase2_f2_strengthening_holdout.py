#!/usr/bin/env python3
"""Run v99 Phase2 fixed-best score policy on the same-scene temporal holdout."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools import build_v98_1_canonical_holdout_metrics as holdout  # noqa: E402
from tools import build_v99_phase1_f2_base_reproduction as p1  # noqa: E402
from tools import build_v99_phase2_f2_strengthening as phase2  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase2_f2_strengthening"
PHASE0_SUMMARY = AUDIT_ROOT / "v99_phase0_fact_lock/summary.json"
DEV_SUMMARY = OUT_DIR / "best_variant_summary.json"
HOLDOUT_SOURCE_ROWS = AUDIT_ROOT / "v98_phase13_holdout/source_container_rows.csv"
HOLDOUT_REAL_ROWS = AUDIT_ROOT / "v98_phase13_holdout_phase9_render_snap/mv_object_frame_mask_rows.csv"
HOLDOUT_FEATURES = AUDIT_ROOT / "v98_phase13_holdout_radio_features_npz/mask_features.npz"
DEV_SEMANTIC_CONSTANTS = AUDIT_ROOT / "v98_phase6_semantic_residual_constants/semantic_constants.json"
BASE_VARIANT = "F2_mask_centered_plus_semantic_residual_proxy"
FIXED_DEV_VARIANT = "P2_D2_frame_count_plus_semantic_tiebreak"
HOLDOUT_VARIANT = "P2_D2_frame_count_plus_semantic_tiebreak__holdout_fixed"
EPS = 1e-4


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


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = []
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _load_holdout_residual_features() -> dict[tuple[str, int, int], np.ndarray]:
    constants = json.loads(DEV_SEMANTIC_CONSTANTS.read_text(encoding="utf-8"))
    mu = np.asarray(np.load(holdout._project(constants["radio_mu_vector_path"])), dtype=np.float32)
    payload = np.load(HOLDOUT_FEATURES, allow_pickle=True)
    features = np.asarray(payload["features"], dtype=np.float32)
    residual = p1._normalize_rows(features - mu[None, :])
    out: dict[tuple[str, int, int], np.ndarray] = {}
    for idx in range(residual.shape[0]):
        out[(str(payload["scene_id"][idx]), int(payload["frame_id"][idx]), int(payload["mask_id"][idx]))] = residual[idx]
    return out


def _semantic_consistency_by_object(rows: list[dict[str, Any]]) -> dict[str, float]:
    features = _load_holdout_residual_features()
    by_object: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        feat = features.get((str(row["scene_id"]), int(row["frame_id"]), int(row["mask_id"])))
        if feat is not None:
            by_object[str(row["mv_object_id"])].append(feat)
    raw: dict[str, float] = {}
    for oid, vals in by_object.items():
        if len(vals) < 2:
            raw[oid] = 0.0
            continue
        stack = np.stack(vals).astype(np.float32)
        centroid = p1._normalize_rows(np.mean(stack, axis=0, keepdims=True))[0]
        raw[oid] = float(np.mean([p1._cosine(row, centroid) for row in stack]))
    return phase2._norm_map(raw)


def _apply_fixed_policy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_object[str(row["mv_object_id"])].append(row)
    max_frame_count = 1.0
    frame_count: dict[str, float] = {}
    for oid, vals in by_object.items():
        frames = {(row["scene_id"], int(row["frame_id"])) for row in vals}
        frame_count[oid] = float(len(frames))
        max_frame_count = max(max_frame_count, frame_count[oid])
    semantic_norm = _semantic_consistency_by_object(rows)
    out: list[dict[str, Any]] = []
    for oid, vals in by_object.items():
        score = frame_count[oid] / max_frame_count + EPS * float(semantic_norm.get(oid, 0.0))
        for row in vals:
            out.append(
                {
                    **row,
                    "variant_id": HOLDOUT_VARIANT,
                    "variant": HOLDOUT_VARIANT,
                    "score": float(score),
                    "score_policy": "holdout_frame_count_norm_plus_1e-4_dev_frozen_semantic_consistency_tiebreak",
                    "fixed_dev_variant_id": FIXED_DEV_VARIANT,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    return out


def _update_phase2_gate_and_summary(holdout_summary: dict[str, Any]) -> dict[str, Any]:
    dev_summary = json.loads(DEV_SUMMARY.read_text(encoding="utf-8"))
    gate_path = OUT_DIR / "variant_gate_rows.csv"
    gate_rows = _read_csv(gate_path)
    for row in gate_rows:
        if row.get("gate_id") == "holdout_not_drop_more_than_0p005_MV_AP_window":
            row["pass"] = bool(holdout_summary["holdout_not_drop_more_than_0p005_MV_AP_window"])
            row["expected"] = f">={holdout_summary['holdout_min_allowed_MV_AP_window']}"
            row["observed"] = holdout_summary["holdout_MV_AP_window"]
    failure_rows = [
        {
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": "fixed best dev variant failed holdout; do not claim Phase2 success",
        }
        for row in gate_rows
        if str(row.get("pass")).strip().lower() not in {"1", "true", "yes"}
    ]
    full_pass = bool(dev_summary.get("phase2_dev_pass")) and not failure_rows
    dev_summary.update(
        {
            "holdout_evaluated": True,
            "holdout_required_before_phase2_success": True,
            "phase2_full_pass": full_pass,
            "decision": "PASS_PHASE2_ENTER_PHASE3" if full_pass else "PASS_DEV_HOLDOUT_FAIL",
            "holdout_variant_id": HOLDOUT_VARIANT,
            "holdout_MV_AP_window": holdout_summary["holdout_MV_AP_window"],
            "holdout_MV_AP50_window": holdout_summary["holdout_MV_AP50_window"],
            "holdout_min_allowed_MV_AP_window": holdout_summary["holdout_min_allowed_MV_AP_window"],
            "holdout_not_drop_more_than_0p005_MV_AP_window": holdout_summary[
                "holdout_not_drop_more_than_0p005_MV_AP_window"
            ],
            "holdout_summary": _rel(OUT_DIR / "holdout_summary.json"),
        }
    )
    _write_csv(gate_path, gate_rows)
    _write_csv(OUT_DIR / "variant_failure_rows.csv", failure_rows)
    _write_json(DEV_SUMMARY, dev_summary)
    return dev_summary


def main() -> int:
    started = datetime.now()
    phase0 = json.loads(PHASE0_SUMMARY.read_text(encoding="utf-8"))
    source_scope = holdout._load_source_scope(HOLDOUT_SOURCE_ROWS)
    base_rows = holdout._load_selected_rows(HOLDOUT_REAL_ROWS, allowed_variants={BASE_VARIANT})
    if not base_rows:
        raise RuntimeError("no holdout F2 base rows found")
    fixed_rows = _apply_fixed_policy(base_rows)
    metric_rows, case_rows, top_rows = holdout._evaluate_variant(HOLDOUT_VARIANT, fixed_rows, source_scope)
    aggregate_rows = holdout._aggregate(metric_rows, family="v99_phase2_fixed_best_holdout")
    fixed = aggregate_rows[0] if aggregate_rows else {}
    holdout_ap = _num(fixed.get("mean_MV_AP_window"), -1.0)
    holdout_ap50 = _num(fixed.get("mean_MV_AP50_window"), -1.0)
    holdout_base = float(phase0["F2_base_holdout_MV_AP_window"])
    min_allowed = holdout_base - 0.005
    holdout_pass = bool(
        holdout_ap >= min_allowed
        and int(_num(fixed.get("same_frame_collision_count"), 1)) == 0
        and int(_num(fixed.get("missing_mask_raster_count"), 1)) == 0
        and not source_scope["uses_future"]
        and not source_scope["uses_gt_for_prediction"]
    )
    summary = {
        "schema_version": "stream4d_v99_phase2_f2_strengthening_holdout_summary_v1",
        "phase_id": "v99_phase2_f2_strengthening_holdout",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "decision": "PASS_HOLDOUT_FOR_FIXED_PHASE2_DEV_VARIANT" if holdout_pass else "NO_GO_HOLDOUT_FOR_FIXED_PHASE2_DEV_VARIANT",
        "fixed_dev_variant_id": FIXED_DEV_VARIANT,
        "holdout_variant_id": HOLDOUT_VARIANT,
        "holdout_MV_AP_window": holdout_ap,
        "holdout_MV_AP50_window": holdout_ap50,
        "holdout_min_allowed_MV_AP_window": min_allowed,
        "F2_base_holdout_MV_AP_window": holdout_base,
        "holdout_not_drop_more_than_0p005_MV_AP_window": holdout_ap >= min_allowed,
        "same_frame_collision_count": int(_num(fixed.get("same_frame_collision_count"), 1)),
        "missing_mask_raster_count": int(_num(fixed.get("missing_mask_raster_count"), 1)),
        "uses_future": False,
        "uses_gt_for_prediction": False,
        "holdout_pass": holdout_pass,
        "metric_source_window": "v98_1 canonical holdout evaluator using run_v65_scene_multiview_ap.SparseSceneIoU/_summarize_iou",
        "semantic_constants_source": _rel(DEV_SEMANTIC_CONSTANTS),
        "outputs": {
            "holdout_metric_rows": _rel(OUT_DIR / "holdout_metric_rows.csv"),
            "holdout_metric_aggregate_rows": _rel(OUT_DIR / "holdout_metric_aggregate_rows.csv"),
            "holdout_case_rows": _rel(OUT_DIR / "holdout_case_rows.csv"),
            "holdout_top_iou_rows": _rel(OUT_DIR / "holdout_top_iou_rows.csv"),
            "holdout_mv_object_frame_mask_rows": _rel(OUT_DIR / "holdout_mv_object_frame_mask_rows.csv"),
            "holdout_summary": _rel(OUT_DIR / "holdout_summary.json"),
        },
    }
    _write_csv(OUT_DIR / "holdout_metric_rows.csv", metric_rows)
    _write_csv(OUT_DIR / "holdout_metric_aggregate_rows.csv", aggregate_rows)
    _write_csv(OUT_DIR / "holdout_case_rows.csv", case_rows)
    _write_csv(OUT_DIR / "holdout_top_iou_rows.csv", top_rows)
    _write_csv(OUT_DIR / "holdout_mv_object_frame_mask_rows.csv", fixed_rows)
    _write_json(OUT_DIR / "holdout_summary.json", summary)
    phase2_summary = _update_phase2_gate_and_summary(summary)
    print(json.dumps(_jsonable({"holdout": summary, "phase2": phase2_summary}), indent=2, sort_keys=True))
    return 0 if bool(phase2_summary.get("phase2_full_pass")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
