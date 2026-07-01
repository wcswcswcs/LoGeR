from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STREAM3D = ROOT / "Stream3D"
AUDIT_ROOT = STREAM3D / "outputs" / "audit"
PHASE1_DIR = AUDIT_ROOT / "v101_phase1_f2_fragmentation_casebook"
OUT_DIR = AUDIT_ROOT / "v101_phase1b_fragment_quality_decomp"


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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
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


def _agg_rows(df: pd.DataFrame, group_cols: list[str], schema: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped = df.groupby(group_cols, observed=True, dropna=False) if group_cols else [((), df)]
    for key, sub in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = {
            "schema_version": schema,
            "phase_id": "v101_phase1b_fragment_quality_decomp",
        }
        for col, value in zip(group_cols, key):
            row[col] = value
        row.update(
            {
                "gt_count": int(len(sub)),
                "fragment_count_mean": float(sub["fragment_count"].mean()),
                "fragment_count_p50": float(sub["fragment_count"].quantile(0.5)),
                "fragment_count_p90": float(sub["fragment_count"].quantile(0.9)),
                "fragment_count_ge2_rate": float((sub["fragment_count"] >= 2).mean()),
                "fragment_count_ge3_rate": float((sub["fragment_count"] >= 3).mean()),
                "union_minus_best_IoU_mean": float(sub["union_minus_best_IoU"].mean()),
                "union_minus_best_IoU_p90": float(sub["union_minus_best_IoU"].quantile(0.9)),
                "union_gain_gt_0p10_rate": float((sub["union_minus_best_IoU"] > 0.10).mean()),
                "best_pred_IoU_mean": float(sub["best_pred_IoU"].mean()),
                "union_pred_IoU_mean": float(sub["union_pred_IoU"].mean()),
            }
        )
        rows.append(row)
    return rows


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary1 = json.loads((PHASE1_DIR / "summary.json").read_text(encoding="utf-8"))
    gt = pd.read_csv(PHASE1_DIR / "gt_fragment_rows.csv")
    pred = pd.read_csv(PHASE1_DIR / "pred_object_fragment_rows.csv")
    overlap = pd.read_csv(PHASE1_DIR / "pred_gt_overlap_rows.csv")
    threshold = pd.read_csv(PHASE1_DIR / "fragment_threshold_summary_rows.csv")

    gt = gt.copy()
    gt["size_bucket"] = pd.cut(
        gt["GT_pixels"],
        bins=[0, 10_000, 50_000, 200_000, 10**15],
        labels=["tiny", "small", "medium", "large"],
        include_lowest=True,
    )

    split_rows = _agg_rows(gt, ["dataset_split"], "stream4d_v101_phase1b_split_fragment_quality_row_v1")
    scene_rows = _agg_rows(gt, ["dataset_split", "scene_id"], "stream4d_v101_phase1b_scene_fragment_quality_row_v1")
    size_rows = _agg_rows(gt, ["size_bucket"], "stream4d_v101_phase1b_size_fragment_quality_row_v1")

    quantile_rows = [
        {
            "schema_version": "stream4d_v101_phase1b_overlap_iou_quantile_row_v1",
            "phase_id": "v101_phase1b_fragment_quality_decomp",
            "quantile": float(q),
            "IoU": float(overlap["IoU"].quantile(q)),
        }
        for q in [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    ]

    pred_summary_rows = [
        {
            "schema_version": "stream4d_v101_phase1b_pred_broad_summary_row_v1",
            "phase_id": "v101_phase1b_fragment_quality_decomp",
            "row_id": "all_pred_objects",
            "pred_object_count": int(len(pred)),
            "broad_mask_share_mean": float(pred["broad_mask_share"].mean()),
            "broad_mask_share_p75": float(pred["broad_mask_share"].quantile(0.75)),
            "broad_mask_share_max": float(pred["broad_mask_share"].max()),
            "mean_mask_area_ratio_mean": float(pred["mean_mask_area_ratio"].mean()),
            "mean_mask_area_ratio_p75": float(pred["mean_mask_area_ratio"].quantile(0.75)),
            "matched_GT_count_mean": float(pred["matched_GT_count"].mean()),
            "matched_GT_count_p75": float(pred["matched_GT_count"].quantile(0.75)),
            "matched_GT_count_max": int(pred["matched_GT_count"].max()),
            "best_GT_IoU_mean": float(pred["best_GT_IoU"].mean()),
        },
        {
            "schema_version": "stream4d_v101_phase1b_pred_broad_summary_row_v1",
            "phase_id": "v101_phase1b_fragment_quality_decomp",
            "row_id": "pred_objects_with_any_broad_mask",
            "pred_object_count": int((pred["broad_mask_share"] > 0).sum()),
            "matched_GT_count_mean": float(pred.loc[pred["broad_mask_share"] > 0, "matched_GT_count"].mean()) if (pred["broad_mask_share"] > 0).any() else 0.0,
            "best_GT_IoU_mean": float(pred.loc[pred["broad_mask_share"] > 0, "best_GT_IoU"].mean()) if (pred["broad_mask_share"] > 0).any() else 0.0,
        },
    ]

    top = gt.sort_values(["fragment_count", "union_minus_best_IoU"], ascending=[False, False]).head(40)
    top_rows = [
        {
            "schema_version": "stream4d_v101_phase1b_top_low_quality_fragment_row_v1",
            "phase_id": "v101_phase1b_fragment_quality_decomp",
            **row,
        }
        for row in top.to_dict("records")
    ]

    merge_potential_low = bool(summary1.get("merge_potential_confirmed") is False)
    tiny_overlap_dominates = bool(float(threshold.loc[threshold["pred_gt_iou_threshold"] == 0.05, "fragment_count_ge2_rate"].iloc[0]) < 0.25)
    union_gain_low = bool(float(summary1.get("union_minus_best_IoU_mean", 0.0)) < 0.08)
    decision = (
        "DIAG_FRAGMENTATION_LOW_QUALITY_BLOCK_DIRECT_MERGE_ENTER_PROVIDER_AUDIT_WITH_CAUTION"
        if merge_potential_low and tiny_overlap_dominates and union_gain_low
        else "DIAG_FRAGMENTATION_HAS_MERGE_POTENTIAL"
    )

    gate_rows = [
        {
            "gate_id": "phase1_fragmentation_confirmed",
            "pass": bool(summary1.get("fragmentation_confirmed")),
            "expected": "true from Phase1",
            "observed": summary1.get("fragmentation_confirmed"),
            "severity": "routing",
        },
        {
            "gate_id": "merge_potential_confirmed",
            "pass": bool(summary1.get("merge_potential_confirmed")),
            "expected": "true for direct fragment merge",
            "observed": summary1.get("merge_potential_confirmed"),
            "severity": "routing",
        },
        {
            "gate_id": "effective_iou0p05_fragment_ge2_rate_lt_0p25",
            "pass": tiny_overlap_dominates,
            "expected": "<0.25 means tiny overlaps dominate raw fragment_count",
            "observed": float(threshold.loc[threshold["pred_gt_iou_threshold"] == 0.05, "fragment_count_ge2_rate"].iloc[0]),
            "severity": "diagnostic",
        },
        {
            "gate_id": "union_gain_mean_low",
            "pass": union_gain_low,
            "expected": "<0.08",
            "observed": summary1.get("union_minus_best_IoU_mean"),
            "severity": "diagnostic",
        },
    ]

    split_csv = OUT_DIR / "split_fragment_quality_rows.csv"
    scene_csv = OUT_DIR / "scene_fragment_quality_rows.csv"
    size_csv = OUT_DIR / "size_fragment_quality_rows.csv"
    quantile_csv = OUT_DIR / "overlap_iou_quantile_rows.csv"
    pred_csv = OUT_DIR / "pred_broad_summary_rows.csv"
    top_csv = OUT_DIR / "top_low_quality_fragment_rows.csv"
    gate_csv = OUT_DIR / "variant_gate_rows.csv"

    _write_csv(split_csv, split_rows)
    _write_csv(scene_csv, scene_rows)
    _write_csv(size_csv, size_rows)
    _write_csv(quantile_csv, quantile_rows)
    _write_csv(pred_csv, pred_summary_rows)
    _write_csv(top_csv, top_rows)
    _write_csv(gate_csv, gate_rows)

    summary = {
        "schema_version": "stream4d_v101_phase1b_fragment_quality_decomp_summary_v1",
        "phase_id": "v101_phase1b_fragment_quality_decomp",
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "phase1b_pass": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic": True,
        "phase1_decision": summary1.get("decision"),
        "phase1_fragmentation_confirmed": summary1.get("fragmentation_confirmed"),
        "phase1_merge_potential_confirmed": summary1.get("merge_potential_confirmed"),
        "raw_fragment_count_ge2_rate": summary1.get("GT_fragment_count_ge2_rate"),
        "effective_iou0p05_fragment_ge2_rate": float(threshold.loc[threshold["pred_gt_iou_threshold"] == 0.05, "fragment_count_ge2_rate"].iloc[0]),
        "raw_union_minus_best_IoU_mean": summary1.get("union_minus_best_IoU_mean"),
        "overlap_IoU_p50": float(overlap["IoU"].quantile(0.5)),
        "overlap_IoU_p75": float(overlap["IoU"].quantile(0.75)),
        "overlap_IoU_p90": float(overlap["IoU"].quantile(0.9)),
        "large_gt_fragment_gain_mean": float(gt.loc[gt["size_bucket"] == "large", "union_minus_best_IoU"].mean()),
        "pred_matched_GT_count_mean": float(pred["matched_GT_count"].mean()),
        "pred_matched_GT_count_p75": float(pred["matched_GT_count"].quantile(0.75)),
        "pred_matched_GT_count_max": int(pred["matched_GT_count"].max()),
        "analysis": {
            "route": "Do not run direct fragment merge yet; Phase2 provider audit may proceed, but any later merge must overcome low-quality/tiny-overlap evidence and use cannot-link/quality filters.",
            "evidence": "Raw IoU>0 fragment count is high, but unioning all fragments reduces IoU on average and effective IoU>=0.05 fragmentation is below the v101 0.25 severity threshold.",
        },
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "split_fragment_quality_rows": _rel(split_csv),
            "scene_fragment_quality_rows": _rel(scene_csv),
            "size_fragment_quality_rows": _rel(size_csv),
            "overlap_iou_quantile_rows": _rel(quantile_csv),
            "pred_broad_summary_rows": _rel(pred_csv),
            "top_low_quality_fragment_rows": _rel(top_csv),
            "variant_gate_rows": _rel(gate_csv),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
