#!/usr/bin/env python3
"""GT-free score calibration diagnostic for v102 Phase7d components."""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import defaultdict
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
OUT_DIR = AUDIT_ROOT / "v102_phase7e_gtfree_score_calibration_diagnostic"
PHASE7D_DIR = AUDIT_ROOT / "v102_phase7d_phase7c_materialized_ap_diagnostic"
PHASE7D_SUMMARY = PHASE7D_DIR / "summary.json"
NODE_ROWS = PHASE7D_DIR / "mv_object_frame_mask_rows.parquet"
OBJECT_ROWS = PHASE7D_DIR / "mv_object_rows.parquet"
COMPONENT_ROWS = PHASE7D_DIR / "materialized_component_rows.csv"

PHASE_ID = "v102_phase7e_gtfree_score_calibration_diagnostic"
VARIANT_PREFIX = "P2_v102_phase7e"


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


def _norm(values: dict[str, float], *, invert: bool = False) -> dict[str, float]:
    finite = [float(v) for v in values.values() if math.isfinite(float(v))]
    if not finite:
        return {key: 0.0 for key in values}
    lo = min(finite)
    hi = max(finite)
    if hi - lo <= 1e-12:
        base = {key: 0.5 for key in values}
    else:
        base = {key: (float(value) - lo) / (hi - lo) for key, value in values.items()}
    if invert:
        return {key: 1.0 - value for key, value in base.items()}
    return base


def _render_accumulator(
    node_rows: pd.DataFrame,
    object_ids: list[str],
) -> tuple[SparseSceneIoU, list[int], dict[str, Any]]:
    object_index = {oid: idx + 1 for idx, oid in enumerate(object_ids)}
    rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in node_rows.to_dict(orient="records"):
        rows_by_frame[int(row["frame_id"])].append(row)
    mask_path_by_frame, mask_source = p7d._mask_path_lookup()
    acc = SparseSceneIoU()
    missing_mask_frame_count = 0
    selected_mask_missing_count = 0
    pixel_collision_count = 0
    pred_positive_total = 0
    gt_positive_total = 0
    eval_frames = p7d._frame_universe()
    for frame_id in eval_frames:
        mask_path = mask_path_by_frame.get((p7d.SCENE_ID, int(frame_id)))
        if mask_path is None or not mask_path.exists():
            missing_mask_frame_count += 1
            continue
        label = p7d._read_label(mask_path)
        pred = np.zeros(label.shape, dtype=np.int64)
        for row in sorted(
            rows_by_frame.get(int(frame_id), []),
            key=lambda r: (-float(_num(r.get("object_score"))), str(r.get("mv_object_id"))),
        ):
            mask = label == int(row["selected_mask_id"])
            if int(np.count_nonzero(mask)) <= 0:
                selected_mask_missing_count += 1
                continue
            pred_id = int(object_index[str(row["mv_object_id"])])
            occupied = (pred > 0) & mask
            pixel_collision_count += int(np.count_nonzero(occupied))
            pred[(pred == 0) & mask] = pred_id
        gt = _load_gt_2d(p7d.SCENE_ID, int(frame_id), label.shape)
        acc.add(pred, gt)
        pred_positive_total += int(np.count_nonzero(pred > 0))
        gt_positive_total += int(np.count_nonzero(gt > 0))
    diag = {
        "eval_frame_count": int(acc.frame_count),
        "expected_frame_count": len(eval_frames),
        "missing_mask_frame_count": int(missing_mask_frame_count),
        "selected_mask_missing_count": int(selected_mask_missing_count),
        "pixel_collision_count": int(pixel_collision_count),
        "pred_positive_total": int(pred_positive_total),
        "gt_positive_total": int(gt_positive_total),
        "mask_source": mask_source,
    }
    return acc, eval_frames, diag


def _object_feature_rows(
    node_rows: pd.DataFrame,
    object_rows: pd.DataFrame,
    component_rows: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    comp = component_rows.set_index("mv_object_id").to_dict(orient="index")
    out_rows: list[dict[str, Any]] = []
    raw: dict[str, dict[str, float]] = {}
    for obj in object_rows.sort_values("mv_object_id").to_dict(orient="records"):
        oid = str(obj["mv_object_id"])
        rows = node_rows[node_rows["mv_object_id"].astype(str) == oid]
        used = pd.to_numeric(rows["used_pixel_count"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        entropy = pd.to_numeric(rows["semantic_entropy"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        margin = pd.to_numeric(rows["semantic_prototype_margin"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        broad = rows["broad_background_risk"].astype(str).str.lower().isin(["1", "true", "yes", "y"]).to_numpy()
        meta = comp.get(oid, {})
        vals = {
            "phase7d_score": float(_num(obj.get("object_score"))),
            "frame_count": float(_num(obj.get("object_frame_count"))),
            "node_count": float(len(rows)),
            "area_sum": float(np.sum(used)) if used.size else 0.0,
            "area_mean": float(np.mean(used)) if used.size else 0.0,
            "area_max": float(np.max(used)) if used.size else 0.0,
            "semantic_entropy_mean": float(np.mean(entropy)) if entropy.size else _num(meta.get("semantic_entropy_mean")),
            "semantic_margin_mean": float(np.mean(margin)) if margin.size else _num(meta.get("semantic_margin_mean")),
            "broad_fraction": float(np.mean(broad)) if broad.size else 0.0,
        }
        raw[oid] = vals
        out_rows.append(
            {
                "schema_version": "stream4d_v102_phase7e_object_feature_v1",
                "phase_id": PHASE_ID,
                "mv_object_id": oid,
                **vals,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return out_rows, raw


def _score_variants(raw: dict[str, dict[str, float]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    ids = sorted(raw)
    frame = _norm({oid: raw[oid]["frame_count"] for oid in ids})
    node = _norm({oid: raw[oid]["node_count"] for oid in ids})
    area_sum = _norm({oid: math.log1p(raw[oid]["area_sum"]) for oid in ids})
    area_mean = _norm({oid: math.log1p(raw[oid]["area_mean"]) for oid in ids})
    inv_area_mean = _norm({oid: math.log1p(raw[oid]["area_mean"]) for oid in ids}, invert=True)
    entropy_good = _norm({oid: raw[oid]["semantic_entropy_mean"] for oid in ids}, invert=True)
    margin = _norm({oid: raw[oid]["semantic_margin_mean"] for oid in ids})
    broad_good = {oid: 1.0 - min(1.0, max(0.0, raw[oid]["broad_fraction"])) for oid in ids}
    phase7d = {oid: raw[oid]["phase7d_score"] for oid in ids}
    sem_quality = {oid: 0.60 * margin[oid] + 0.35 * entropy_good[oid] + 0.05 * broad_good[oid] for oid in ids}
    variants = {
        "S0_phase7d_frame_count_semantic_tiebreak": (
            phase7d,
            "original Phase7d component_frame_count_over_32_plus_semantic_quality_tiebreak",
        ),
        "S1_node_count": (node, "minmax(node_count)"),
        "S2_log_area_sum": (area_sum, "minmax(log1p(sum_used_pixel_count))"),
        "S3_semantic_quality": (
            sem_quality,
            "0.60*margin_norm + 0.35*inverse_entropy_norm + 0.05*(1-broad_fraction)",
        ),
        "S4_frame_x_semantic_quality": (
            {oid: frame[oid] * (0.25 + 0.75 * sem_quality[oid]) for oid in ids},
            "frame_norm * (0.25 + 0.75*semantic_quality)",
        ),
        "S5_frame_x_inverse_mean_area": (
            {oid: frame[oid] * (0.25 + 0.75 * inv_area_mean[oid]) for oid in ids},
            "frame_norm * (0.25 + 0.75*inverse_mean_area_norm)",
        ),
        "S6_balanced_frame_semantic_area": (
            {oid: 0.50 * frame[oid] + 0.30 * sem_quality[oid] + 0.20 * area_sum[oid] for oid in ids},
            "0.50*frame_norm + 0.30*semantic_quality + 0.20*area_sum_norm",
        ),
        "S7_small_area_semantic": (
            {oid: 0.55 * sem_quality[oid] + 0.45 * inv_area_mean[oid] for oid in ids},
            "0.55*semantic_quality + 0.45*inverse_mean_area_norm",
        ),
        "S8_node_area_semantic": (
            {oid: 0.45 * node[oid] + 0.35 * area_sum[oid] + 0.20 * sem_quality[oid] for oid in ids},
            "0.45*node_norm + 0.35*area_sum_norm + 0.20*semantic_quality",
        ),
        "S9_margin_only": (margin, "minmax(semantic_prototype_margin_mean)"),
    }
    config_rows = [
        {
            "schema_version": "stream4d_v102_phase7e_score_variant_config_v1",
            "phase_id": PHASE_ID,
            "score_variant_id": key,
            "score_formula": formula,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for key, (_scores, formula) in variants.items()
    ]
    return config_rows, {key: scores for key, (scores, _formula) in variants.items()}


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase7d = _read_json(PHASE7D_SUMMARY)
    node_rows = pd.read_parquet(NODE_ROWS)
    object_rows = pd.read_parquet(OBJECT_ROWS)
    component_rows = pd.read_csv(COMPONENT_ROWS)
    object_ids = sorted(str(v) for v in object_rows["mv_object_id"].tolist())
    object_index = {oid: idx + 1 for idx, oid in enumerate(object_ids)}
    acc, eval_frames, render_diag = _render_accumulator(node_rows, object_ids)
    feature_rows, raw_features = _object_feature_rows(node_rows, object_rows, component_rows)
    config_rows, score_maps = _score_variants(raw_features)

    metric_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for score_variant_id, scores_by_object in score_maps.items():
        input_scores = np.ones((len(object_ids),), dtype=np.float32)
        for oid, idx in object_index.items():
            input_scores[idx - 1] = float(scores_by_object.get(oid, 0.0))
            score_rows.append(
                {
                    "schema_version": "stream4d_v102_phase7e_object_score_v1",
                    "phase_id": PHASE_ID,
                    "score_variant_id": score_variant_id,
                    "mv_object_id": oid,
                    "score": float(scores_by_object.get(oid, 0.0)),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
        summary, _iou, _pred_ids, _gt_ids = _summarize_iou(
            accumulator=acc,
            min_pred_pixels=64,
            min_gt_pixels=64,
            score_mode="input",
            input_scores=input_scores,
        )
        metric_rows.append(
            {
                "schema_version": "stream4d_v102_phase7e_score_metric_v1",
                "phase_id": PHASE_ID,
                "score_variant_id": score_variant_id,
                "metric_scope": "chunk32_scene0050_local_diagnostic_not_full_dev",
                "MV_AP_window": summary.get("ap"),
                "MV_AP50_window": summary.get("ap50"),
                "MV_AP25_window": summary.get("ap25"),
                "MV_AP_scene": summary.get("ap"),
                "MV_AP50_scene": summary.get("ap50"),
                "ScoreFreeMatch50_window": (summary.get("score_free_match_at_050") or {}).get("recall"),
                "ScoreFreeMatch25_window": (summary.get("score_free_match_at_025") or {}).get("recall"),
                "score_unique_count": summary.get("score_unique_count"),
                "evaluated_pred_count": summary.get("evaluated_pred_count"),
                "evaluated_gt_count": summary.get("evaluated_gt_count"),
                "uses_gt_for_prediction": False,
                "uses_gt_for_eval": True,
                "uses_future": False,
            }
        )

    base = next(row for row in metric_rows if row["score_variant_id"] == "S0_phase7d_frame_count_semantic_tiebreak")
    best = max(metric_rows, key=lambda row: (_num(row.get("MV_AP50_window")), _num(row.get("MV_AP_window"))))
    best_delta_ap50 = _num(best.get("MV_AP50_window")) - _num(base.get("MV_AP50_window"))
    best_delta_ap = _num(best.get("MV_AP_window")) - _num(base.get("MV_AP_window"))
    local_improves_ap50 = best_delta_ap50 > 1e-12
    decision = (
        "PASS_PHASE7E_GT_FREE_SCORE_CALIBRATION_LOCAL_AP50_IMPROVES__FORMAL_TARGET_NOT_CLAIMED"
        if local_improves_ap50
        else "NO_GO_PHASE7E_GT_FREE_SCORE_CALIBRATION_NO_LOCAL_AP50_GAIN"
    )
    gate_rows = [
        {
            "schema_version": "stream4d_v102_phase7e_gate_v1",
            "phase_id": PHASE_ID,
            "gate": "phase7d_integrity_pass",
            "pass": bool(phase7d.get("integrity_pass")),
            "observed": phase7d.get("decision"),
            "required": "Phase7d local materialization integrity",
        },
        {
            "schema_version": "stream4d_v102_phase7e_gate_v1",
            "phase_id": PHASE_ID,
            "gate": "render_integrity_pass",
            "pass": bool(
                render_diag["missing_mask_frame_count"] == 0
                and render_diag["selected_mask_missing_count"] == 0
                and render_diag["pixel_collision_count"] == 0
                and render_diag["eval_frame_count"] == render_diag["expected_frame_count"]
            ),
            "observed": json.dumps(render_diag, sort_keys=True),
            "required": "no missing masks/collisions and all 32 frames evaluated",
        },
        {
            "schema_version": "stream4d_v102_phase7e_gate_v1",
            "phase_id": PHASE_ID,
            "gate": "best_gtfree_score_ap50_improves_over_phase7d",
            "pass": bool(local_improves_ap50),
            "observed": best_delta_ap50,
            "required": ">0 AP50 delta in local diagnostic",
        },
        {
            "schema_version": "stream4d_v102_phase7e_gate_v1",
            "phase_id": PHASE_ID,
            "gate": "formal_v102_target_achieved",
            "pass": False,
            "observed": "not claimed from score calibration diagnostic",
            "required": "full-dev/holdout formal AP repair gate",
        },
    ]

    _write_csv(OUT_DIR / "score_variant_config_rows.csv", config_rows)
    _write_csv(OUT_DIR / "object_feature_rows.csv", feature_rows)
    _write_csv(OUT_DIR / "object_score_rows.csv", score_rows)
    _write_csv(OUT_DIR / "score_metric_rows.csv", metric_rows)
    _write_csv(OUT_DIR / "score_gate_rows.csv", gate_rows)
    summary = {
        "schema_version": "stream4d_v102_phase7e_score_calibration_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "metric_scope": "chunk32_scene0050_local_diagnostic_not_full_dev",
        "score_variant_count": len(score_maps),
        "base_score_variant_id": base["score_variant_id"],
        "base_MV_AP_window": base.get("MV_AP_window"),
        "base_MV_AP50_window": base.get("MV_AP50_window"),
        "best_score_variant_id": best["score_variant_id"],
        "best_MV_AP_window": best.get("MV_AP_window"),
        "best_MV_AP50_window": best.get("MV_AP50_window"),
        "best_MV_AP25_window": best.get("MV_AP25_window"),
        "best_MV_AP_scene": best.get("MV_AP_scene"),
        "best_MV_AP50_scene": best.get("MV_AP50_scene"),
        "best_delta_MV_AP50_window_vs_phase7d": best_delta_ap50,
        "best_delta_MV_AP_window_vs_phase7d": best_delta_ap,
        "best_ScoreFreeMatch50_window": best.get("ScoreFreeMatch50_window"),
        "best_ScoreFreeMatch25_window": best.get("ScoreFreeMatch25_window"),
        "local_gtfree_score_ap50_improves_over_phase7d": local_improves_ap50,
        "formal_v102_target_achieved": False,
        "formal_target_blocker": "Phase7e changes only GT-free score ordering for a local chunk32 diagnostic; Phase6 full repair remains blocked by Phase1b.",
        "render_diag": render_diag,
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
        "truthfulness_note": (
            "Score formulas use only GT-free Phase7d/component features. GT is used only to diagnose AP differences. "
            "This is local chunk32 score calibration, not a full-dev/holdout AP improvement claim."
        ),
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "score_variant_config_rows": _rel(OUT_DIR / "score_variant_config_rows.csv"),
            "object_feature_rows": _rel(OUT_DIR / "object_feature_rows.csv"),
            "object_score_rows": _rel(OUT_DIR / "object_score_rows.csv"),
            "score_metric_rows": _rel(OUT_DIR / "score_metric_rows.csv"),
            "score_gate_rows": _rel(OUT_DIR / "score_gate_rows.csv"),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
