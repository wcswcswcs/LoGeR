from __future__ import annotations

import csv
import json
import math
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STREAM3D = ROOT / "Stream3D"
AUDIT_ROOT = STREAM3D / "outputs" / "audit"
OUT_DIR = AUDIT_ROOT / "v102_phase1_fragment_casebook"

PHASE0_DIR = AUDIT_ROOT / "v102_phase0_fact_lock"
V101_PHASE1_DIR = AUDIT_ROOT / "v101_phase1_f2_fragmentation_casebook"
V101_PHASE1B_DIR = AUDIT_ROOT / "v101_phase1b_fragment_quality_decomp"
PHASE2C_DIR = AUDIT_ROOT / "v100_phase2c_overlap3_local_repair"
PLAN_DOC = ROOT / "docs" / "stream4d_v102_mask_pair_primitive_bridge_da3_giant_3dgs_plan.md"

EFFECTIVE_IOU_TAU = 0.05
BROAD_PAIR_SCORE_TAU = 0.50
OBJECT_LIKE_MIN_AREA_RATIO = 0.005
BROAD_MASK_AREA_RATIO = 0.20


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
    if isinstance(value, (list, tuple, set)):
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


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "" or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _object_stats(pred_rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for row in pred_rows.to_dict("records"):
        oid = str(row["mv_object_id"])
        stats[oid] = {
            "matched_GT_count": int(_num(row.get("matched_GT_count"))),
            "best_GT_IoU": _num(row.get("best_GT_IoU")),
            "second_best_GT_IoU": _num(row.get("second_best_GT_IoU")),
            "mean_mask_area_ratio": _num(row.get("mean_mask_area_ratio")),
            "broad_mask_share": _num(row.get("broad_mask_share")),
            "object_frame_count": int(_num(row.get("object_frame_count"))),
            "mask_count": int(_num(row.get("mask_count"))),
            "semantic_residual_coherence": row.get("semantic_residual_coherence", ""),
        }
    return stats


def _gt_stats(gt_rows: pd.DataFrame) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    stats: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in gt_rows.to_dict("records"):
        key = (
            str(row["dataset_split"]),
            str(row["scene_id"]),
            str(row["chunk_id"]),
            int(row["gt_window_id"]),
        )
        stats[key] = row
    return stats


def _frame_membership(mask_rows: pd.DataFrame) -> dict[str, dict[int, set[int]]]:
    membership: dict[str, dict[int, set[int]]] = {}
    for row in mask_rows[["mv_object_id", "frame_id", "selected_mask_id"]].to_dict("records"):
        oid = str(row["mv_object_id"])
        membership.setdefault(oid, {}).setdefault(int(row["frame_id"]), set()).add(int(row["selected_mask_id"]))
    return membership


def _same_frame_collision_if_merged(obj_i: str, obj_j: str, membership: dict[str, dict[int, set[int]]]) -> tuple[bool, int, int]:
    a = membership.get(obj_i, {})
    b = membership.get(obj_j, {})
    shared = sorted(set(a) & set(b))
    collision_frames = 0
    same_mask_shared = 0
    for frame_id in shared:
        masks_a = a.get(frame_id, set())
        masks_b = b.get(frame_id, set())
        if masks_a & masks_b:
            same_mask_shared += 1
        if masks_a and masks_b and not (masks_a & masks_b):
            collision_frames += 1
    return collision_frames > 0, len(shared), same_mask_shared


def _copy_rows(df: pd.DataFrame, schema: str, phase_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in df.to_dict("records"):
        new = dict(row)
        new["schema_version"] = schema
        new["phase_id"] = phase_id
        new["source_phase_id"] = row.get("phase_id", "")
        rows.append(new)
    return rows


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    phase0 = _read_json(PHASE0_DIR / "summary.json")
    if not bool(phase0.get("phase0_pass")):
        raise RuntimeError("Phase0 did not pass; refusing Phase1 casebook.")

    v101_phase1 = _read_json(V101_PHASE1_DIR / "summary.json")
    v101_phase1b = _read_json(V101_PHASE1B_DIR / "summary.json")
    gt_df = pd.read_csv(V101_PHASE1_DIR / "gt_fragment_rows.csv")
    pred_df = pd.read_csv(V101_PHASE1_DIR / "pred_object_fragment_rows.csv")
    pair_df = pd.read_csv(V101_PHASE1_DIR / "fragment_pair_rows.csv")
    threshold_df = pd.read_csv(V101_PHASE1_DIR / "fragment_threshold_summary_rows.csv")
    mask_df = pd.read_parquet(PHASE2C_DIR / "mv_object_frame_mask_rows.parquet")

    obj_stats = _object_stats(pred_df)
    gt_by_key = _gt_stats(gt_df)
    membership = _frame_membership(mask_df)

    fragment_pair_rows: list[dict[str, Any]] = []
    repair_candidate_rows: list[dict[str, Any]] = []
    broad_rows: list[dict[str, Any]] = []
    effective_pair_count = 0
    repair_candidate_count = 0
    repair_candidate_gt: set[tuple[str, str, str, int]] = set()
    union_gain_values: list[float] = []
    candidate_union_gain_values: list[float] = []
    same_frame_collision_if_all_merged_count = 0
    broad_contaminated_pair_count = 0

    for row in pair_df.to_dict("records"):
        split = str(row["dataset_split"])
        scene = str(row["scene_id"])
        chunk = str(row["chunk_id"])
        gid = int(row["gt_window_id"])
        obj_i = str(row["obj_i"])
        obj_j = str(row["obj_j"])
        i_iou = _num(row.get("obj_i_iou_to_gt"))
        j_iou = _num(row.get("obj_j_iou_to_gt"))
        effective = bool(i_iou >= EFFECTIVE_IOU_TAU and j_iou >= EFFECTIVE_IOU_TAU)
        obj_i_stats = obj_stats.get(obj_i, {})
        obj_j_stats = obj_stats.get(obj_j, {})
        broad_score = max(_num(obj_i_stats.get("broad_mask_share")), _num(obj_j_stats.get("broad_mask_share")))
        mean_area_score = max(_num(obj_i_stats.get("mean_mask_area_ratio")), _num(obj_j_stats.get("mean_mask_area_ratio")))
        broad_risk = bool(broad_score > 0 or mean_area_score >= BROAD_MASK_AREA_RATIO)
        broad_contaminated_pair_count += int(broad_risk)
        gt_row = gt_by_key.get((split, scene, chunk, gid), {})
        union_gain = _num(gt_row.get("union_minus_best_IoU"))
        union_gain_values.append(union_gain)
        collision, shared_frame_count, same_mask_shared = _same_frame_collision_if_merged(obj_i, obj_j, membership)
        same_frame_collision_if_all_merged_count += int(collision)
        repair_candidate = bool(effective and union_gain > 0.0 and not collision and broad_score <= BROAD_PAIR_SCORE_TAU)
        effective_pair_count += int(effective)
        repair_candidate_count += int(repair_candidate)
        if repair_candidate:
            repair_candidate_gt.add((split, scene, chunk, gid))
            candidate_union_gain_values.append(union_gain)

        out = {
            "schema_version": "stream4d_v102_phase1_fragment_pair_row_v1",
            "phase_id": "v102_phase1_fragment_casebook",
            "source_phase_id": row.get("phase_id", ""),
            "dataset_split": split,
            "scene_id": scene,
            "chunk_id": chunk,
            "window_id": row.get("window_id", chunk),
            "gt_window_id": gid,
            "raw_gt_object_id": row.get("raw_gt_object_id", ""),
            "obj_i": obj_i,
            "obj_j": obj_j,
            "obj_i_iou_to_gt": i_iou,
            "obj_j_iou_to_gt": j_iou,
            "effective_fragment_pair": effective,
            "repair_candidate_pair": repair_candidate,
            "union_minus_best_IoU_for_GT": union_gain,
            "same_frame_collision_if_merged": collision,
            "shared_frame_count": shared_frame_count,
            "same_mask_shared_frame_count": same_mask_shared,
            "obj_i_broad_mask_share": obj_i_stats.get("broad_mask_share", ""),
            "obj_j_broad_mask_share": obj_j_stats.get("broad_mask_share", ""),
            "obj_i_mean_mask_area_ratio": obj_i_stats.get("mean_mask_area_ratio", ""),
            "obj_j_mean_mask_area_ratio": obj_j_stats.get("mean_mask_area_ratio", ""),
            "broad_contamination_score": broad_score,
            "broad_area_score": mean_area_score,
            "broad_contamination_risk": broad_risk,
            "uses_gt_for_label": True,
            "uses_gt_for_prediction": False,
        }
        fragment_pair_rows.append(out)
        if repair_candidate:
            repair_candidate_rows.append(out)

    for (split, scene, chunk), sub in pd.DataFrame(fragment_pair_rows).groupby(
        ["dataset_split", "scene_id", "chunk_id"], sort=True
    ):
        broad_rows.append(
            {
                "schema_version": "stream4d_v102_phase1_broad_contamination_row_v1",
                "phase_id": "v102_phase1_fragment_casebook",
                "dataset_split": split,
                "scene_id": scene,
                "chunk_id": chunk,
                "fragment_pair_count": int(len(sub)),
                "effective_fragment_pair_count": int(sub["effective_fragment_pair"].sum()),
                "repair_candidate_pair_count": int(sub["repair_candidate_pair"].sum()),
                "broad_contamination_pair_count": int(sub["broad_contamination_risk"].sum()),
                "broad_contamination_rate": float(sub["broad_contamination_risk"].mean()) if len(sub) else 0.0,
                "same_frame_collision_if_all_merged_count": int(sub["same_frame_collision_if_merged"].sum()),
                "union_minus_best_IoU_mean": float(sub["union_minus_best_IoU_for_GT"].mean()) if len(sub) else 0.0,
            }
        )

    raw_ge2_rate = _num(v101_phase1.get("GT_fragment_count_ge2_rate"))
    effective_tau_row = threshold_df.loc[threshold_df["pred_gt_iou_threshold"] == EFFECTIVE_IOU_TAU]
    effective_ge2_rate = float(effective_tau_row["fragment_count_ge2_rate"].iloc[0]) if len(effective_tau_row) else 0.0
    broad_contamination_rate = float(broad_contaminated_pair_count / max(1, len(fragment_pair_rows)))
    union_minus_best_positive_rate = float(sum(1 for v in union_gain_values if v > 0.0) / max(1, len(union_gain_values)))
    candidate_union_minus_best_mean = float(np.mean(candidate_union_gain_values)) if candidate_union_gain_values else 0.0

    phase1_route = (
        "DIAG_REPAIR_SPACE_INSUFFICIENT_BRIDGE_DIAGNOSTIC_ONLY"
        if repair_candidate_count < 30
        else "BLOCK_DIRECT_MERGE_BROAD_CONTAMINATION_HIGH_ENTER_BRIDGE_PURIFICATION"
        if broad_contamination_rate > 0.5
        else "PASS_REPAIR_CANDIDATES_ENTER_PROVIDER_BRIDGE"
    )

    gate_rows = [
        {
            "gate_id": "repair_candidate_pair_count_ge_30",
            "pass": repair_candidate_count >= 30,
            "expected": ">=30 for later AP repair",
            "observed": repair_candidate_count,
            "severity": "phase_route",
        },
        {
            "gate_id": "broad_contamination_rate_le_0p5",
            "pass": broad_contamination_rate <= 0.5,
            "expected": "<=0.5 before direct merge",
            "observed": broad_contamination_rate,
            "severity": "phase_route",
        },
        {
            "gate_id": "missing_phase0_inputs",
            "pass": True,
            "expected": "v102 Phase0 passed and v101 casebook artifacts present",
            "observed": "present",
            "severity": "required",
        },
        {
            "gate_id": "uses_gt_only_for_diagnostic",
            "pass": True,
            "expected": "GT only labels diagnostic same-object candidates; no method threshold is promoted from GT",
            "observed": "script emits diagnostic target universe only",
            "severity": "required",
        },
    ]
    failure_rows = [
        {
            "schema_version": "stream4d_v102_phase1_failure_row_v1",
            "phase_id": "v102_phase1_fragment_casebook",
            "gate_id": row["gate_id"],
            "expected": row["expected"],
            "observed": row["observed"],
            "severity": row["severity"],
        }
        for row in gate_rows
        if row["severity"] == "required" and not bool(row["pass"])
    ]

    gt_csv = OUT_DIR / "gt_fragment_rows.csv"
    pair_csv = OUT_DIR / "fragment_pair_rows.csv"
    candidate_csv = OUT_DIR / "repair_candidate_pair_rows.csv"
    broad_csv = OUT_DIR / "broad_contamination_rows.csv"
    threshold_csv = OUT_DIR / "fragment_threshold_summary_rows.csv"
    gate_csv = OUT_DIR / "variant_gate_rows.csv"
    failure_csv = OUT_DIR / "variant_failure_rows.csv"

    _write_csv(gt_csv, _copy_rows(gt_df, "stream4d_v102_phase1_gt_fragment_row_v1", "v102_phase1_fragment_casebook"))
    _write_csv(pair_csv, fragment_pair_rows)
    _write_csv(candidate_csv, repair_candidate_rows)
    _write_csv(broad_csv, broad_rows)
    _write_csv(
        threshold_csv,
        _copy_rows(
            threshold_df,
            "stream4d_v102_phase1_fragment_threshold_summary_row_v1",
            "v102_phase1_fragment_casebook",
        ),
    )
    _write_csv(gate_csv, gate_rows)
    _write_csv(failure_csv, failure_rows)

    summary = {
        "schema_version": "stream4d_v102_phase1_fragment_casebook_summary_v1",
        "phase_id": "v102_phase1_fragment_casebook",
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": phase1_route if not failure_rows else "BLOCK_PHASE2_REPAIR_PHASE1_INPUTS",
        "phase1_pass": not failure_rows,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic": True,
        "raw_GT_fragment_count_ge2_rate": raw_ge2_rate,
        "effective_iou0p05_fragment_ge2_rate": effective_ge2_rate,
        "repair_candidate_pair_count": repair_candidate_count,
        "repair_candidate_GT_count": len(repair_candidate_gt),
        "fragment_pair_count": len(fragment_pair_rows),
        "effective_fragment_pair_count": effective_pair_count,
        "union_minus_best_IoU_mean": _num(v101_phase1.get("union_minus_best_IoU_mean")),
        "union_minus_best_IoU_positive_rate_among_fragment_pairs": union_minus_best_positive_rate,
        "repair_candidate_union_minus_best_IoU_mean": candidate_union_minus_best_mean,
        "pred_matched_GT_count_mean": _num(v101_phase1b.get("pred_matched_GT_count_mean")),
        "pred_matched_GT_count_p75": _num(v101_phase1b.get("pred_matched_GT_count_p75")),
        "broad_contamination_pair_count": broad_contaminated_pair_count,
        "broad_contamination_rate": broad_contamination_rate,
        "same_frame_collision_if_all_merged_count": same_frame_collision_if_all_merged_count,
        "phase1_route_note": (
            "Phase1 builds a diagnostic repair target universe. It does not authorize Phase6 AP repair; "
            "Phase5 bridge gates must pass first."
        ),
        "repair_candidate_policy": {
            "effective_pair": f"both obj_i/object_j IoU to diagnostic GT >= {EFFECTIVE_IOU_TAU}",
            "repair_candidate": "effective pair, GT union_minus_best_IoU > 0, no same-frame competing collision, broad_contamination_score <= 0.50",
            "broad_mask_policy": {
                "object_like_min_area_ratio": OBJECT_LIKE_MIN_AREA_RATIO,
                "broad_mask_area_ratio": BROAD_MASK_AREA_RATIO,
            },
        },
        "source_context": {
            "v101_phase1_decision": v101_phase1.get("decision"),
            "v101_phase1b_decision": v101_phase1b.get("decision"),
            "plan_doc": _rel(PLAN_DOC),
        },
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "gt_fragment_rows": _rel(gt_csv),
            "fragment_pair_rows": _rel(pair_csv),
            "repair_candidate_pair_rows": _rel(candidate_csv),
            "broad_contamination_rows": _rel(broad_csv),
            "fragment_threshold_summary_rows": _rel(threshold_csv),
            "variant_gate_rows": _rel(gate_csv),
            "variant_failure_rows": _rel(failure_csv),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if not failure_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
