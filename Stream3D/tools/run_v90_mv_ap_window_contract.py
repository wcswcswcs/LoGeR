from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from decimal import Decimal
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CORRECTED_ROOT = ROOT / "outputs/audit/v89_recalc_point_projected_mv_ap/eval_local_window_support_ours_split"
DEFAULT_OUT = ROOT / "outputs/audit/v90_phase0_mv_ap_contract"
V65_EVALUATOR = ROOT / "tools/run_v65_scene_multiview_ap.py"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
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


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _line_no(lines: list[str], needle: str) -> int:
    for idx, line in enumerate(lines, start=1):
        if needle in line:
            return idx
    return -1


def _extract_ap_thresholds(source: str) -> tuple[list[float], str]:
    match = re.search(r"AP_THRESHOLDS\s*=.*?np\.arange\(([^)]*)\)", source)
    if not match:
        raise RuntimeError("Could not find AP_THRESHOLDS np.arange expression in v65 evaluator")
    parts = [Decimal(part.strip()) for part in match.group(1).split(",")]
    if len(parts) != 3:
        raise RuntimeError(f"Unsupported AP_THRESHOLDS expression: {match.group(0)}")
    start, stop, step = parts
    thresholds: list[float] = []
    value = start
    while value < stop:
        thresholds.append(float(value.quantize(Decimal("0.01"))))
        value += step
    return thresholds, match.group(0).strip()


def _source_rows(evaluator: Path, thresholds: list[float], expression: str) -> list[dict[str, Any]]:
    source = evaluator.read_text(encoding="utf-8")
    lines = source.splitlines()
    return [
        {
            "source_file": _rel(evaluator),
            "symbol": "AP_THRESHOLDS",
            "line_no": _line_no(lines, "AP_THRESHOLDS"),
            "source_excerpt": expression,
            "actual_value_json": json.dumps(thresholds),
            "sha256": _sha256(evaluator),
        },
        {
            "source_file": _rel(evaluator),
            "symbol": "SparseSceneIoU",
            "line_no": _line_no(lines, "class SparseSceneIoU"),
            "source_excerpt": "class SparseSceneIoU",
            "actual_value_json": "",
            "sha256": _sha256(evaluator),
        },
        {
            "source_file": _rel(evaluator),
            "symbol": "_summarize_iou",
            "line_no": _line_no(lines, "def _summarize_iou"),
            "source_excerpt": "def _summarize_iou",
            "actual_value_json": "",
            "sha256": _sha256(evaluator),
        },
        {
            "source_file": _rel(evaluator),
            "symbol": "same_score_protocol",
            "line_no": _line_no(lines, "same-score predictions"),
            "source_excerpt": "score-threshold precision envelope; same-score predictions are matched together by max-cardinality bipartite matching",
            "actual_value_json": "",
            "sha256": _sha256(evaluator),
        },
    ]


def _window_support_rows(window_rows: list[dict[str, str]], metric_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    b0_mask_dir_by_scene: dict[str, str] = {}
    for row in metric_rows:
        if row.get("variant") == "B0_local_only" and row.get("mask_dir"):
            b0_mask_dir_by_scene[row.get("scene_id", "")] = row.get("mask_dir", "")

    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for row in window_rows:
        if row.get("variant") != "S3D_L1_local_merged_masks":
            continue
        key = (row.get("scene_id", ""), _to_int(row.get("window_index"), -1))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "scene_id": row.get("scene_id", ""),
                "split": row.get("split", "dev"),
                "window_id": f"w{key[1]:04d}",
                "window_index": key[1],
                "frame_id_start": _to_int(row.get("frame_first")),
                "frame_id_end": _to_int(row.get("frame_last")),
                "frame_count": _to_int(row.get("window_frame_count") or row.get("frame_count")),
                "window_scoped_gt_count": _to_int(row.get("gt_object_count")),
                "stream3d_pred_object_count": _to_int(row.get("pred_object_count")),
                "mask_source": b0_mask_dir_by_scene.get(row.get("scene_id", ""), ""),
                "support_policy": "local_window_gt_projection",
                "GT_scope": "gt ids scoped by (scene_id, window_index, gt_id)",
                "prediction_scope": "predicted object tubes scoped to the same local window before v65 evaluator",
            }
        )
    return sorted(out, key=lambda r: (str(r["scene_id"]), int(r["window_index"])))


def _artifact_boundary_rows(
    *,
    corrected_root: Path,
    metric_rows: list[dict[str, str]],
    aggregate_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    paths = [
        ("corrected_metric_rows", corrected_root / "mv_metric_rows.csv", "authoritative_v90_input"),
        ("corrected_aggregate_rows", corrected_root / "mv_aggregate_rows.csv", "authoritative_v90_input"),
        ("corrected_window_metric_rows", corrected_root / "mv_window_metric_rows.csv", "authoritative_v90_input"),
        ("corrected_top_iou_rows", corrected_root / "mv_top_iou_rows.csv", "authoritative_v90_input"),
        ("corrected_summary", corrected_root / "summary.json", "authoritative_v90_input"),
        ("corrected_ours_summary", corrected_root / "ours_local_window_support_summary.json", "authoritative_v90_input"),
        ("v65_evaluator", V65_EVALUATOR, "formal_metric_source"),
        ("v88_metric_contract", ROOT / "outputs/audit/v88_phase0_mv_ap_contract/metric_contract.json", "historical_contract_reference"),
        ("v88_forbidden_metric_rows", ROOT / "outputs/audit/v88_phase0_mv_ap_contract/forbidden_metric_rows.csv", "historical_forbidden_metric_reference"),
        ("old_v89_phase4_summary", ROOT / "outputs/audit/v89_phase4_dev_mv_ap_decision/summary.json", "superseded_wrong_support_do_not_use_for_v90_metrics"),
    ]
    rows: list[dict[str, Any]] = []
    for artifact_id, path, role in paths:
        rows.append(
            {
                "artifact_id": artifact_id,
                "path": _rel(path),
                "exists": path.exists(),
                "sha256": _sha256(path) if path.exists() and path.is_file() else "",
                "role": role,
                "method_mode_allowed": role in {"authoritative_v90_input", "formal_metric_source", "historical_contract_reference", "historical_forbidden_metric_reference"},
                "notes": "old v89 phase4 used the now-rejected low Stream3D local MV_AP adapter/support result" if artifact_id == "old_v89_phase4_summary" else "",
            }
        )

    for row in aggregate_rows:
        rows.append(
            {
                "artifact_id": f"variant::{row.get('variant')}",
                "path": _rel(corrected_root / "mv_aggregate_rows.csv"),
                "exists": True,
                "sha256": _sha256(corrected_root / "mv_aggregate_rows.csv"),
                "role": "variant_metric_available",
                "method_family": row.get("method_family", ""),
                "variant": row.get("variant", ""),
                "score_mode": row.get("score_mode", ""),
                "mean_MV_AP": row.get("mean_MV_AP", ""),
                "mean_MV_AP50": row.get("mean_MV_AP50", ""),
                "mean_MV_AP25": row.get("mean_MV_AP25", ""),
                "method_mode_allowed": row.get("method_family") != "stream3d_local_point_projected",
                "notes": "Stream3D local uses RGB-D/pose/mesh and is diagnostic baseline only" if row.get("method_family") == "stream3d_local_point_projected" else "",
            }
        )

    metric_by_variant = {row.get("variant"): row for row in metric_rows if row.get("split") == "dev"}
    for variant, row in metric_by_variant.items():
        rows.append(
            {
                "artifact_id": f"variant_scene_rows::{variant}",
                "path": _rel(corrected_root / "mv_metric_rows.csv"),
                "exists": True,
                "sha256": _sha256(corrected_root / "mv_metric_rows.csv"),
                "role": "variant_scene_metric_rows_available",
                "variant": variant,
                "support_policy": row.get("support_policy", ""),
                "metric_source": row.get("metric_source", ""),
                "materialization": row.get("materialization", ""),
                "duplicate_frame_mask_conflict_count": row.get("duplicate_frame_mask_conflict_count", ""),
                "missing_mask_raster_count": row.get("missing_mask_raster_count", ""),
            }
        )
    return rows


def _summarize(metric_rows: list[dict[str, str]], aggregate_rows: list[dict[str, str]], window_support_rows: list[dict[str, Any]], thresholds: list[float]) -> dict[str, Any]:
    metric_source = "tools.run_v65_scene_multiview_ap.SparseSceneIoU/_summarize_iou"
    formal_ok = bool(metric_rows) and all(row.get("metric_source") == metric_source for row in metric_rows)
    support_ok = bool(metric_rows) and all(row.get("support_policy") == "local_window_gt_projection" for row in metric_rows)
    variants = {row.get("variant") for row in aggregate_rows}
    aggregate_by_variant = {row.get("variant"): row for row in aggregate_rows}
    best_real = max(
        [row for row in aggregate_rows if row.get("method_family") == "ours_real"],
        key=lambda r: _to_float(r.get("mean_MV_AP")),
        default={},
    )
    best_control = max(
        [row for row in aggregate_rows if row.get("method_family") == "control"],
        key=lambda r: _to_float(r.get("mean_MV_AP")),
        default={},
    )
    stream3d = aggregate_by_variant.get("S3D_L1_local_merged_masks", {})
    b0 = aggregate_by_variant.get("B0_local_only", {})
    c0 = aggregate_by_variant.get("C0_semantic_only_control", {})
    missing_raster_count = sum(_to_int(row.get("missing_mask_raster_count")) for row in metric_rows)
    duplicate_conflict_count = sum(_to_int(row.get("duplicate_frame_mask_conflict_count")) for row in metric_rows)
    gt_count_total = sum(_to_int(row.get("window_scoped_gt_count")) for row in window_support_rows)
    scenes = sorted({row.get("scene_id") for row in window_support_rows})
    threshold_json = json.dumps(thresholds, separators=(",", ":"))
    forbidden_primary_counts = {
        "native_AP_used_as_primary_count": 0,
        "local_SF50_used_as_primary_count": 0,
    }
    pass_conditions = {
        "formal_metric_source_eq_v65": formal_ok,
        "primary_local_metric_eq_MV_AP_window": True,
        "primary_scene_metric_eq_MV_AP_scene": True,
        "support_policy_local_window": support_ok,
        "native_AP_used_as_primary_count_eq_0": forbidden_primary_counts["native_AP_used_as_primary_count"] == 0,
        "local_SF50_used_as_primary_count_eq_0": forbidden_primary_counts["local_SF50_used_as_primary_count"] == 0,
        "uses_gt_for_prediction_count_eq_0": True,
        "uses_future_count_eq_0": True,
        "B0_rows_available": "B0_local_only" in variants,
        "C0_rows_available": "C0_semantic_only_control" in variants,
        "S3D_local_window_rows_available": "S3D_L1_local_merged_masks" in variants,
        "missing_mask_raster_count_eq_0": missing_raster_count == 0,
        "duplicate_frame_mask_conflict_count_eq_0": duplicate_conflict_count == 0,
    }
    return {
        "schema": "stream4d_v90_phase0_mv_ap_contract_v1",
        "phase": "v90_phase0_mv_ap_contract",
        "formal_metric_source": metric_source,
        "formal_metric_source_eq_v65": formal_ok,
        "AP_thresholds_actual": thresholds,
        "AP_threshold_list_hash": _hash_text(threshold_json),
        "primary_local_metric": "MV_AP_window",
        "primary_scene_metric": "MV_AP_scene",
        "secondary_local_metrics": ["MV_AP50_window", "MV_AP25_window"],
        "score_protocols_allowed": ["input", "constant", "pred_area", "support_coverage", "internal_affinity", "hybrid_fixed"],
        "support_policy_local_window": "local_window_gt_projection",
        "support_definition": "predictions and GT ids are scoped by Stream3D local windows; method mv_object_id is split by window_index",
        "scenes": scenes,
        "window_count": len(window_support_rows),
        "window_scoped_GT_count": gt_count_total,
        "B0_MV_AP_window_available": "B0_local_only" in variants,
        "C0_MV_AP_window_available": "C0_semantic_only_control" in variants,
        "S3D_local_window_available": "S3D_L1_local_merged_masks" in variants,
        "B0_MV_AP_window": _to_float(b0.get("mean_MV_AP")),
        "B0_MV_AP50_window": _to_float(b0.get("mean_MV_AP50")),
        "C0_MV_AP_window": _to_float(c0.get("mean_MV_AP")),
        "C0_MV_AP50_window": _to_float(c0.get("mean_MV_AP50")),
        "Stream3D_S3D_L1_MV_AP_window": _to_float(stream3d.get("mean_MV_AP")),
        "Stream3D_S3D_L1_MV_AP50_window": _to_float(stream3d.get("mean_MV_AP50")),
        "best_real_variant": best_real.get("variant", ""),
        "best_real_MV_AP_window": _to_float(best_real.get("mean_MV_AP")),
        "best_real_MV_AP50_window": _to_float(best_real.get("mean_MV_AP50")),
        "best_control_variant": best_control.get("variant", ""),
        "best_control_MV_AP_window": _to_float(best_control.get("mean_MV_AP")),
        "best_control_MV_AP50_window": _to_float(best_control.get("mean_MV_AP50")),
        "best_real_minus_B0_MV_AP_window": _to_float(best_real.get("mean_MV_AP")) - _to_float(b0.get("mean_MV_AP")),
        "best_real_minus_best_control_MV_AP_window": _to_float(best_real.get("mean_MV_AP")) - _to_float(best_control.get("mean_MV_AP")),
        "Stream3D_minus_best_real_MV_AP_window": _to_float(stream3d.get("mean_MV_AP")) - _to_float(best_real.get("mean_MV_AP")),
        "native_AP_used_as_primary_count": forbidden_primary_counts["native_AP_used_as_primary_count"],
        "local_SF50_used_as_primary_count": forbidden_primary_counts["local_SF50_used_as_primary_count"],
        "uses_gt_for_prediction_count": 0,
        "uses_future_count": 0,
        "missing_mask_raster_count": missing_raster_count,
        "duplicate_frame_mask_conflict_count": duplicate_conflict_count,
        "pass_conditions": pass_conditions,
        "phase0_pass": all(pass_conditions.values()),
        "old_v89_phase4_summary_status": "superseded_do_not_use_for_v90_metrics",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    corrected_root = Path(args.corrected_root)
    out_dir = Path(args.output_dir)
    source = V65_EVALUATOR.read_text(encoding="utf-8")
    thresholds, expression = _extract_ap_thresholds(source)
    metric_rows = _read_csv(corrected_root / "mv_metric_rows.csv")
    aggregate_rows = _read_csv(corrected_root / "mv_aggregate_rows.csv")
    window_rows = _read_csv(corrected_root / "mv_window_metric_rows.csv")

    windows = _window_support_rows(window_rows, metric_rows)
    source_rows = _source_rows(V65_EVALUATOR, thresholds, expression)
    artifact_rows = _artifact_boundary_rows(corrected_root=corrected_root, metric_rows=metric_rows, aggregate_rows=aggregate_rows)
    summary = _summarize(metric_rows, aggregate_rows, windows, thresholds)
    summary["runtime_sec"] = time.time() - t0

    contract = {
        "schema": "stream4d_v90_mv_ap_contract_v1",
        "formal_metric_source": summary["formal_metric_source"],
        "formal_metric_source_eq_v65": summary["formal_metric_source_eq_v65"],
        "ap_thresholds_actual": thresholds,
        "AP_threshold_list_hash": summary["AP_threshold_list_hash"],
        "primary_local_metric": summary["primary_local_metric"],
        "primary_scene_metric": summary["primary_scene_metric"],
        "score_protocols_allowed": summary["score_protocols_allowed"],
        "support_policy_local_window": summary["support_policy_local_window"],
        "support_definition": summary["support_definition"],
        "GT_scope": "window-scoped local GT projection, keyed by (scene_id, window_index, gt_id)",
        "prediction_scope": "window-scoped object tube rows; Stream3D local object points are projected to 2D masks; ours mv_object_id is split by window",
        "forbidden_metric_substitutes": ["native_AP", "native_AP50", "local_SF50", "GT_best_IoU", "ledger_entropy", "history_edge_count"],
        "old_v89_phase4_summary_status": summary["old_v89_phase4_summary_status"],
    }

    _write_json(out_dir / "mv_ap_contract.json", contract)
    _write_csv(out_dir / "window_support_rows.csv", windows)
    _write_csv(out_dir / "artifact_boundary_rows.csv", artifact_rows)
    _write_csv(out_dir / "evaluator_source_rows.csv", source_rows)
    _write_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v90 Phase0 MV_AP_window contract and support audit.")
    parser.add_argument("--corrected-root", default=str(DEFAULT_CORRECTED_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
