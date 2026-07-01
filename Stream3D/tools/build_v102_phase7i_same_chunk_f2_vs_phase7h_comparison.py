#!/usr/bin/env python3
"""Same-chunk local comparison between v100 F2 overlap3 baseline and v102 Phase7h."""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools.run_v65_scene_multiview_ap import SparseSceneIoU, _load_gt_2d, _summarize_iou  # noqa: E402
from tools import build_v102_phase7d_phase7c_materialized_ap_diagnostic as p7d  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
OUT_DIR = AUDIT_ROOT / "v102_phase7i_same_chunk_f2_vs_phase7h_comparison"
PLAN_DOC = REPO_ROOT / "docs" / "stream4d_v102_mask_pair_primitive_bridge_da3_giant_3dgs_plan.md"
F2_ROWS = AUDIT_ROOT / "v100_phase2c_overlap3_local_repair" / "mv_object_frame_mask_rows.parquet"
PHASE7H_SUMMARY = AUDIT_ROOT / "v102_phase7h_chunk32_primitive_support_shape_diagnostic" / "summary.json"
PHASE7H_ROWS = AUDIT_ROOT / "v102_phase7h_chunk32_primitive_support_shape_diagnostic" / "materialized_expanded_rows.csv"

PHASE_ID = "v102_phase7i_same_chunk_f2_vs_phase7h_comparison"
F2_VARIANT = "F2_v100_chunk32_overlap3_surfel_maskview_thr018_p2d2"


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _filter_same_chunk(rows: pd.DataFrame, variant_id: str) -> pd.DataFrame:
    frames = set(p7d._frame_universe())
    out = rows[
        (rows["scene_id"].astype(str) == p7d.SCENE_ID)
        & (rows["frame_id"].astype(int).isin(frames))
        & (rows["variant_id"].astype(str) == variant_id)
    ].copy()
    if "dataset_split" in out.columns:
        split = out["dataset_split"].fillna("").astype(str).str.lower()
        out = out[split.isin(["dev", "", "nan"])]
    return out.reset_index(drop=True)


def _evaluate(rows: pd.DataFrame, variant_id: str, source_label: str) -> dict[str, Any]:
    record_rows = rows.to_dict(orient="records")
    object_ids = sorted({str(row["mv_object_id"]) for row in record_rows})
    object_index = {oid: idx + 1 for idx, oid in enumerate(object_ids)}
    object_scores: dict[str, float] = defaultdict(float)
    for row in record_rows:
        oid = str(row["mv_object_id"])
        object_scores[oid] = max(float(object_scores[oid]), _num(row.get("score"), 1.0))
    rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in record_rows:
        rows_by_frame[int(row["frame_id"])].append(row)

    mask_path_by_frame, mask_source = p7d._mask_path_lookup()
    acc = SparseSceneIoU()
    pixel_collision_count = 0
    missing_mask_frame_count = 0
    selected_mask_missing_count = 0
    total_pred_positive = 0
    total_gt_positive = 0
    for frame in p7d._frame_universe():
        mask_path = mask_path_by_frame.get((p7d.SCENE_ID, int(frame)))
        if mask_path is None or not mask_path.exists():
            missing_mask_frame_count += 1
            continue
        label = p7d._read_label(mask_path)
        gt = _load_gt_2d(p7d.SCENE_ID, int(frame), label.shape)
        pred = np.zeros(label.shape, dtype=np.int64)
        selected_rows = sorted(
            rows_by_frame.get(int(frame), []),
            key=lambda r: (-object_scores[str(r["mv_object_id"])], str(r["mv_object_id"])),
        )
        for row in selected_rows:
            mask = label == int(_num(row.get("selected_mask_id")))
            if int(np.count_nonzero(mask)) <= 0:
                selected_mask_missing_count += 1
                continue
            occupied = (pred > 0) & mask
            pixel_collision_count += int(np.count_nonzero(occupied))
            pred[(pred == 0) & mask] = object_index[str(row["mv_object_id"])]
        acc.add(pred, gt)
        total_pred_positive += int(np.count_nonzero(pred > 0))
        total_gt_positive += int(np.count_nonzero(gt > 0))

    input_scores = np.ones((len(object_ids),), dtype=np.float32)
    for oid, idx in object_index.items():
        input_scores[idx - 1] = float(object_scores.get(oid, 1.0))
    summary, _iou, _pred_ids, _gt_ids = _summarize_iou(
        accumulator=acc,
        min_pred_pixels=64,
        min_gt_pixels=64,
        score_mode="input",
        input_scores=input_scores,
    )
    duplicate_keys = Counter((int(row["frame_id"]), int(row["selected_mask_id"])) for row in record_rows)
    object_frame_keys = Counter((str(row["mv_object_id"]), int(row["frame_id"])) for row in record_rows)
    return {
        "schema_version": "stream4d_v102_phase7i_same_chunk_metric_v1",
        "phase_id": PHASE_ID,
        "source_label": source_label,
        "variant_id": variant_id,
        "metric_scope": "scene0050_00_c0000_frames_0_155_stride5_local_diagnostic",
        "MV_AP_window": summary.get("ap"),
        "MV_AP50_window": summary.get("ap50"),
        "MV_AP25_window": summary.get("ap25"),
        "MV_AP_scene": None,
        "MV_AP50_scene": None,
        "scene_metric_computed": False,
        "scene_metric_not_computed_reason": "Phase7i evaluates only scene0050_00/c0000 chunk32 frames 0..155 stride5; full-scene/local2history scene metric is not computed.",
        "ScoreFreeMatch50_window": (summary.get("score_free_match_at_050") or {}).get("recall"),
        "ScoreFreeMatch25_window": (summary.get("score_free_match_at_025") or {}).get("recall"),
        "object_count": int(len(object_ids)),
        "frame_mask_count": int(len(record_rows)),
        "eval_frame_count": int(acc.frame_count),
        "same_frame_duplicate_mask_count": int(sum(max(0, c - 1) for c in duplicate_keys.values())),
        "object_frame_duplicate_count": int(sum(max(0, c - 1) for c in object_frame_keys.values())),
        "pixel_collision_count": int(pixel_collision_count),
        "missing_mask_frame_count": int(missing_mask_frame_count),
        "selected_mask_missing_count": int(selected_mask_missing_count),
        "total_pred_positive_pixels": int(total_pred_positive),
        "total_gt_positive_pixels": int(total_gt_positive),
        "uses_gt_for_prediction_any": bool(any(_bool(row.get("uses_gt_for_prediction")) for row in record_rows)),
        "uses_future_any": bool(any(_bool(row.get("uses_future")) for row in record_rows)),
        "mask_source": mask_source,
    }


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase7h = _read_json(PHASE7H_SUMMARY)
    phase7h_variant = f"P2_v102_phase7h_{phase7h.get('best_variant_id')}"

    f2_rows = _filter_same_chunk(pd.read_parquet(F2_ROWS), F2_VARIANT)
    p7h_rows = _filter_same_chunk(pd.read_csv(PHASE7H_ROWS), phase7h_variant)
    metric_rows = [
        _evaluate(f2_rows, F2_VARIANT, "v100_phase2c_f2_overlap3_same_chunk"),
        _evaluate(p7h_rows, phase7h_variant, "v102_phase7h_best_same_chunk"),
    ]
    f2 = metric_rows[0]
    p7h = metric_rows[1]
    delta_ap = _num(p7h.get("MV_AP_window")) - _num(f2.get("MV_AP_window"))
    delta_ap50 = _num(p7h.get("MV_AP50_window")) - _num(f2.get("MV_AP50_window"))
    delta_sf50 = _num(p7h.get("ScoreFreeMatch50_window")) - _num(f2.get("ScoreFreeMatch50_window"))
    integrity_pass = bool(
        p7h["same_frame_duplicate_mask_count"] == 0
        and p7h["object_frame_duplicate_count"] == 0
        and p7h["pixel_collision_count"] == 0
        and not p7h["uses_gt_for_prediction_any"]
        and not p7h["uses_future_any"]
    )
    local_beats_f2 = bool(delta_ap > 1e-12 and delta_ap50 > 1e-12 and delta_sf50 >= -1e-12)
    decision = (
        "PASS_PHASE7I_PHASE7H_BEATS_SAME_CHUNK_F2_LOCAL_DIAGNOSTIC__FORMAL_TARGET_NOT_CLAIMED"
        if local_beats_f2 and integrity_pass
        else "NO_GO_PHASE7I_PHASE7H_DOES_NOT_BEAT_SAME_CHUNK_F2_LOCAL_DIAGNOSTIC"
    )
    gate_rows = [
        {
            "schema_version": "stream4d_v102_phase7i_gate_v1",
            "phase_id": PHASE_ID,
            "gate": "phase7h_beats_same_chunk_f2_ap_and_ap50",
            "pass": bool(local_beats_f2),
            "observed": f"delta_ap={delta_ap}; delta_ap50={delta_ap50}; delta_sf50={delta_sf50}",
            "required": "MV_AP_window and MV_AP50_window improve; ScoreFreeMatch50 does not regress",
        },
        {
            "schema_version": "stream4d_v102_phase7i_gate_v1",
            "phase_id": PHASE_ID,
            "gate": "phase7h_same_chunk_integrity",
            "pass": bool(integrity_pass),
            "observed": (
                f"dup_mask={p7h['same_frame_duplicate_mask_count']}; "
                f"object_frame_dup={p7h['object_frame_duplicate_count']}; "
                f"pixel_collision={p7h['pixel_collision_count']}; "
                f"uses_gt={p7h['uses_gt_for_prediction_any']}; uses_future={p7h['uses_future_any']}"
            ),
            "required": "no duplicates/collisions and no GT/future prediction use",
        },
        {
            "schema_version": "stream4d_v102_phase7i_gate_v1",
            "phase_id": PHASE_ID,
            "gate": "formal_v102_target_achieved",
            "pass": False,
            "observed": "same-chunk local diagnostic only; no full-dev/holdout Phase6 AP repair",
            "required": "full-dev/holdout formal AP repair gate",
        },
    ]

    _write_csv(OUT_DIR / "same_chunk_metric_rows.csv", metric_rows)
    _write_csv(OUT_DIR / "same_chunk_gate_rows.csv", gate_rows)
    summary = {
        "schema_version": "stream4d_v102_phase7i_same_chunk_f2_vs_phase7h_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "metric_scope": "scene0050_00_c0000_frames_0_155_stride5_local_diagnostic",
        "f2_variant_id": F2_VARIANT,
        "phase7h_variant_id": phase7h_variant,
        "f2_MV_AP_window": f2.get("MV_AP_window"),
        "f2_MV_AP50_window": f2.get("MV_AP50_window"),
        "f2_MV_AP_scene": None,
        "f2_MV_AP50_scene": None,
        "f2_ScoreFreeMatch50_window": f2.get("ScoreFreeMatch50_window"),
        "phase7h_MV_AP_window": p7h.get("MV_AP_window"),
        "phase7h_MV_AP50_window": p7h.get("MV_AP50_window"),
        "phase7h_MV_AP_scene": None,
        "phase7h_MV_AP50_scene": None,
        "phase7h_ScoreFreeMatch50_window": p7h.get("ScoreFreeMatch50_window"),
        "scene_metric_computed": False,
        "scene_metric_not_computed_reason": "Phase7i same-chunk comparison evaluates only scene0050_00/c0000 frames; MV_AP_scene/MV_AP50_scene are not computed.",
        "delta_MV_AP_window_vs_same_chunk_f2": delta_ap,
        "delta_MV_AP50_window_vs_same_chunk_f2": delta_ap50,
        "delta_ScoreFreeMatch50_window_vs_same_chunk_f2": delta_sf50,
        "local_diagnostic_beats_same_chunk_f2": local_beats_f2,
        "integrity_pass": integrity_pass,
        "formal_v102_target_achieved": False,
        "formal_target_blocker": "Phase7i is same-chunk local diagnostic only; Phase6 full AP repair remains blocked by Phase1b repair-space.",
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
        "truthfulness_note": (
            "Both rows are fixed before evaluation. GT is used only by SparseSceneIoU for same-scope local diagnostic readout."
        ),
        "plan_doc": _rel(PLAN_DOC),
        "inputs": {
            "f2_rows": _rel(F2_ROWS),
            "phase7h_summary": _rel(PHASE7H_SUMMARY),
            "phase7h_rows": _rel(PHASE7H_ROWS),
        },
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "same_chunk_metric_rows": _rel(OUT_DIR / "same_chunk_metric_rows.csv"),
            "same_chunk_gate_rows": _rel(OUT_DIR / "same_chunk_gate_rows.csv"),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
