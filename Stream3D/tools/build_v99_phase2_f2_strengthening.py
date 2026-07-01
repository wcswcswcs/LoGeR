#!/usr/bin/env python3
"""Build v99 Phase2 F2 strengthening audit on top of the Phase1 rows."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools import build_v99_phase1_f2_base_reproduction as p1  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase2_f2_strengthening"
PHASE1_DIR = AUDIT_ROOT / "v99_phase1_f2_base_reproduction"
PHASE0_SUMMARY = AUDIT_ROOT / "v99_phase0_fact_lock/summary.json"
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


def _rankdata(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty_like(arr, dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | str:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return ""
    rx = _rankdata(xs)
    ry = _rankdata(ys)
    sx = float(np.std(rx))
    sy = float(np.std(ry))
    if sx <= 1e-12 or sy <= 1e-12:
        return ""
    return float(np.corrcoef(rx, ry)[0, 1])


def _norm_map(values: dict[str, float]) -> dict[str, float]:
    vals = [float(v) for v in values.values() if math.isfinite(float(v))]
    if not vals:
        return {key: 0.0 for key in values}
    lo = min(vals)
    hi = max(vals)
    if hi - lo <= 1e-12:
        return {key: 0.0 for key in values}
    return {key: float((float(val) - lo) / (hi - lo)) for key, val in values.items()}


def _load_phase1_main_rows() -> tuple[str, list[dict[str, Any]]]:
    summary = json.loads((PHASE1_DIR / "summary.json").read_text(encoding="utf-8"))
    main_variant = str(summary["main_variant"])
    rows = [
        dict(row)
        for row in _read_csv(PHASE1_DIR / "mv_object_frame_mask_rows.csv")
        if row.get("variant_id") == main_variant
    ]
    if not rows:
        raise RuntimeError(f"no Phase1 rows found for main_variant={main_variant}")
    return main_variant, rows


def _object_stats(rows: list[dict[str, Any]], scope: dict[str, Any]) -> dict[str, dict[str, float]]:
    features, _tau = p1._load_radio_residual_features()
    area_by_key: dict[tuple[str, int, int], float] = {}
    for row in scope["source_rows"]:
        scene = row.get("scene_id", "")
        frame = p1._int(row.get("frame_id"), -1)
        mask_id = p1._int(row.get("source_mask_id"), -1)
        if scene and frame >= 0 and mask_id > 0:
            area_by_key[(scene, frame, mask_id)] = _num(row.get("mask_area_ratio"))

    by_obj: dict[str, dict[str, Any]] = defaultdict(lambda: {"frames": set(), "support": [], "broad": [], "features": []})
    for row in rows:
        oid = str(row["mv_object_id"])
        scene = str(row["scene_id"])
        frame = int(row["frame_id"])
        mask_id = int(row["selected_mask_id"])
        by_obj[oid]["frames"].add((scene, frame))
        by_obj[oid]["support"].append(_num(row.get("support_surfel_count")))
        area = area_by_key.get((scene, frame, mask_id), 0.0)
        by_obj[oid]["broad"].append(p1._mask_scale_weights(area)[3])
        feat = features.get((scene, frame, mask_id))
        if feat is not None:
            by_obj[oid]["features"].append(feat)

    frame_count = {oid: float(len(vals["frames"])) for oid, vals in by_obj.items()}
    support = {oid: float(np.mean(vals["support"])) if vals["support"] else 0.0 for oid, vals in by_obj.items()}
    broad = {oid: float(np.mean(vals["broad"])) if vals["broad"] else 0.0 for oid, vals in by_obj.items()}
    semantic_consistency: dict[str, float] = {}
    for oid, vals in by_obj.items():
        feats = vals["features"]
        if len(feats) < 2:
            semantic_consistency[oid] = 0.0
            continue
        stack = np.stack(feats).astype(np.float32)
        centroid = p1._normalize_rows(np.mean(stack, axis=0, keepdims=True))[0]
        cos = [p1._cosine(row, centroid) for row in stack]
        semantic_consistency[oid] = float(np.mean(cos))

    by_scene_chunk: dict[tuple[str, str], list[str]] = defaultdict(list)
    obj_scene_chunk: dict[str, tuple[str, str]] = {}
    frames_only: dict[str, set[int]] = {}
    for oid, vals in by_obj.items():
        sample = next(row for row in rows if str(row["mv_object_id"]) == oid)
        key = (str(sample["scene_id"]), str(sample["chunk_id"]))
        by_scene_chunk[key].append(oid)
        obj_scene_chunk[oid] = key
        frames_only[oid] = {frame for _scene, frame in vals["frames"]}

    duplicate_penalty = {oid: 0.0 for oid in by_obj}
    duplicate_pair_count = 0
    for _key, oids in by_scene_chunk.items():
        for i, lhs in enumerate(oids[:-1]):
            lhs_frames = frames_only[lhs]
            for rhs in oids[i + 1 :]:
                rhs_frames = frames_only[rhs]
                denom = max(1, min(len(lhs_frames), len(rhs_frames)))
                overlap = len(lhs_frames & rhs_frames) / float(denom)
                if overlap >= 0.8:
                    duplicate_pair_count += 1
                duplicate_penalty[lhs] = max(duplicate_penalty[lhs], overlap)
                duplicate_penalty[rhs] = max(duplicate_penalty[rhs], overlap)

    support_n = _norm_map(support)
    semantic_n = _norm_map(semantic_consistency)
    broad_n = _norm_map(broad)
    duplicate_n = _norm_map(duplicate_penalty)
    out: dict[str, dict[str, float]] = {}
    for oid in by_obj:
        out[oid] = {
            "frame_count": frame_count[oid],
            "score_frame_count_over_32": frame_count[oid] / float(p1.CHUNK_SIZE),
            "support_norm": support_n[oid],
            "semantic_consistency": semantic_consistency[oid],
            "semantic_norm": semantic_n[oid],
            "broad_risk": broad[oid],
            "broad_norm": broad_n[oid],
            "duplicate_penalty": duplicate_penalty[oid],
            "duplicate_norm": duplicate_n[oid],
            "duplicate_pair_count_global": float(duplicate_pair_count),
        }
    return out


def _make_variant_rows(
    rows: list[dict[str, Any]],
    variant_id: str,
    score_policy: str,
    score_fn: Callable[[str, dict[str, float]], float],
    stats: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        oid = str(row["mv_object_id"])
        new = dict(row)
        new["variant_id"] = variant_id
        new["score"] = float(score_fn(oid, stats[oid]))
        new["score_policy"] = score_policy
        new["score_scope"] = "current_chunk"
        new["phase2_parent_variant_id"] = row["variant_id"]
        new["uses_future"] = False
        out.append(new)
    return out


def _score_vs_best_iou_spearman(variant_id: str, rows: list[dict[str, Any]], scope: dict[str, Any]) -> float | str:
    rows_by_frame_mask: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_frame_mask[(str(row["scene_id"]), str(row["chunk_id"]), int(row["frame_id"]), int(row["selected_mask_id"]))].append(row)

    all_scores: list[float] = []
    all_best_iou: list[float] = []
    for scene in sorted(scope["frames_by_scene"]):
        object_to_idx: dict[str, int] = {}
        scores: dict[str, float] = {}
        acc = p1.SparseSceneIoU()
        gt_id_map: dict[tuple[str, int], int] = {}
        for (row_scene, chunk_id), frames in sorted(scope["chunks"].items()):
            if row_scene != scene:
                continue
            for frame in frames:
                mask_path = scope["mask_path_by_frame"].get((scene, int(frame)))
                if mask_path is not None and mask_path.exists():
                    label = p1._read_label(mask_path)
                    shape_hw = tuple(int(v) for v in label.shape[:2])
                else:
                    shape_hw = (968, 1296)
                    label = np.zeros(shape_hw, dtype=np.int64)
                gt = p1._load_gt_2d(scene, int(frame), shape_hw)
                pred = np.zeros(shape_hw, dtype=np.int64)
                selected_rows: list[dict[str, Any]] = []
                for mask_id in sorted({key[3] for key in rows_by_frame_mask if key[:3] == (scene, chunk_id, int(frame))}):
                    vals = rows_by_frame_mask.get((scene, chunk_id, int(frame), int(mask_id)), [])
                    if not vals:
                        continue
                    selected_rows.append(sorted(vals, key=lambda r: (_num(r.get("score")), str(r.get("mv_object_id"))), reverse=True)[0])
                for row in sorted(selected_rows, key=lambda r: (-_num(r.get("score")), str(r.get("mv_object_id")))):
                    oid = str(row["mv_object_id"])
                    if oid not in object_to_idx:
                        object_to_idx[oid] = len(object_to_idx) + 1
                    scores[oid] = max(float(scores.get(oid, 0.0)), _num(row.get("score"), 1.0))
                    mask = label == int(row["selected_mask_id"])
                    pred[(pred == 0) & mask] = object_to_idx[oid]
                gt_local = p1._window_scoped_gt(gt, f"{scene}|{chunk_id}", gt_id_map)
                acc.add(pred, gt_local)
        _summary, iou, pred_ids, _gt_ids = p1._summarize_iou(
            accumulator=acc,
            min_pred_pixels=1,
            min_gt_pixels=1,
            score_mode="input",
            input_scores=p1._score_array(object_to_idx, scores),
        )
        if not len(pred_ids):
            continue
        best = np.max(iou, axis=1) if iou.shape[1] else np.zeros((iou.shape[0],), dtype=np.float32)
        score_arr = p1._score_array(object_to_idx, scores)
        for idx, pred_id in enumerate(pred_ids):
            source_idx = int(pred_id) - 1
            if 0 <= source_idx < score_arr.shape[0]:
                all_scores.append(float(score_arr[source_idx]))
                all_best_iou.append(float(best[idx]))
    return _spearman(all_scores, all_best_iou)


def main() -> int:
    started = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase0 = json.loads(PHASE0_SUMMARY.read_text(encoding="utf-8"))
    phase1_summary = json.loads((PHASE1_DIR / "summary.json").read_text(encoding="utf-8"))
    if not bool(phase1_summary.get("phase1_pass")):
        raise RuntimeError("Phase2 requires a passing Phase1 summary")

    scope = p1._load_source_scope()
    parent_variant, parent_rows = _load_phase1_main_rows()
    stats = _object_stats(parent_rows, scope)
    duplicate_tube_count = int(max([row["duplicate_pair_count_global"] for row in stats.values()] or [0]))
    broad_mask_selected_rate = float(np.mean([row["broad_risk"] > 0.0 for row in stats.values()])) if stats else 0.0
    semantic_margin_mean = float(np.mean([row["semantic_consistency"] for row in stats.values()])) if stats else 0.0

    configs = [
        {
            "variant_id": "P2_B0_phase1_main_score_replay",
            "family": "baseline",
            "score_policy": "current_chunk_frame_count_over_32_post_nms",
            "description": "Phase1 main rows replay.",
            "fn": lambda _oid, s: s["score_frame_count_over_32"],
        },
        {
            "variant_id": "P2_D1_frame_count_plus_support_tiebreak",
            "family": "F2-D_score_policy",
            "score_policy": "current_chunk_frame_count_over_32_plus_1e-4_surfel_support_tiebreak",
            "description": "Frame-count primary score with tiny support reliability tie-break.",
            "fn": lambda _oid, s: s["score_frame_count_over_32"] + EPS * s["support_norm"],
        },
        {
            "variant_id": "P2_D2_frame_count_plus_semantic_tiebreak",
            "family": "F2-D_score_policy",
            "score_policy": "current_chunk_frame_count_over_32_plus_1e-4_semantic_consistency_tiebreak",
            "description": "Frame-count primary score with tiny semantic consistency tie-break.",
            "fn": lambda _oid, s: s["score_frame_count_over_32"] + EPS * s["semantic_norm"],
        },
        {
            "variant_id": "P2_D3_frame_count_minus_broad_risk_tiebreak",
            "family": "F2-A_mask_reliability",
            "score_policy": "current_chunk_frame_count_over_32_minus_1e-4_broad_risk_tiebreak",
            "description": "Frame-count primary score with tiny broad-risk penalty.",
            "fn": lambda _oid, s: s["score_frame_count_over_32"] - EPS * s["broad_norm"],
        },
        {
            "variant_id": "P2_D4_frame_count_plus_support_semantic_tiebreak",
            "family": "F2-A_F2-D_reliability_score",
            "score_policy": "current_chunk_frame_count_over_32_plus_1e-4_support_semantic_tiebreak",
            "description": "Frame-count primary score with combined support and semantic tie-break.",
            "fn": lambda _oid, s: s["score_frame_count_over_32"] + EPS * (0.5 * s["support_norm"] + 0.5 * s["semantic_norm"]),
        },
        {
            "variant_id": "P2_D5_frame_count_minus_duplicate_tiebreak",
            "family": "F2-C_duplicate_control",
            "score_policy": "current_chunk_frame_count_over_32_minus_1e-4_duplicate_frame_overlap_tiebreak",
            "description": "Frame-count primary score with tiny duplicate frame-overlap penalty.",
            "fn": lambda _oid, s: s["score_frame_count_over_32"] - EPS * s["duplicate_norm"],
        },
    ]

    variant_config_rows: list[dict[str, Any]] = []
    all_metric_scene_rows: list[dict[str, Any]] = []
    all_frame_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for config in configs:
        variant_id = str(config["variant_id"])
        rows = _make_variant_rows(parent_rows, variant_id, str(config["score_policy"]), config["fn"], stats)
        all_rows.extend(rows)
        metric_rows, frame_rows = p1._evaluate_variant(variant_id, rows, scope)
        all_metric_scene_rows.extend(metric_rows)
        all_frame_rows.extend(frame_rows)
        variant_config_rows.append(
            {
                "schema_version": "stream4d_v99_phase2_variant_config_v1",
                "phase_id": "v99_phase2_f2_strengthening",
                "variant_id": variant_id,
                "family": config["family"],
                "parent_phase1_variant_id": parent_variant,
                "score_policy": config["score_policy"],
                "description": config["description"],
                "score_scope": "current_chunk",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )

    aggregate_rows = p1._aggregate_metrics(all_metric_scene_rows)
    spearman_by_variant: dict[str, float | str] = {}
    for config in configs:
        variant_id = str(config["variant_id"])
        rows = [row for row in all_rows if row["variant_id"] == variant_id]
        spearman_by_variant[variant_id] = _score_vs_best_iou_spearman(variant_id, rows, scope)

    config_by_variant = {str(row["variant_id"]): row for row in variant_config_rows}
    for row in aggregate_rows:
        variant_id = str(row["variant_id"])
        row["family"] = config_by_variant[variant_id]["family"]
        row["duplicate_tube_count"] = duplicate_tube_count
        row["broad_mask_selected_rate"] = broad_mask_selected_rate
        row["semantic_residual_margin_mean"] = semantic_margin_mean
        row["tube_nms_suppressed_count"] = 0
        row["score_vs_best_iou_spearman"] = spearman_by_variant.get(variant_id, "")
        row["score_scope"] = "current_chunk"
        row["uses_future"] = False

    baseline = next(row for row in aggregate_rows if row["variant_id"] == "P2_B0_phase1_main_score_replay")
    candidates = [row for row in aggregate_rows if row["variant_id"] != "P2_B0_phase1_main_score_replay"]
    best = max(candidates, key=lambda row: (float(row["MV_AP_window"]), float(row["MV_AP50_window"]), float(row["MV_AP_scene"])))

    base_window = float(phase0["F2_base_full_dev_MV_AP_window"])
    base_scene = float(phase0["F2_base_full_dev_MV_AP_scene"])
    best_window = float(best["MV_AP_window"])
    best_scene = float(best["MV_AP_scene"])
    phase2_dev_pass = bool(best_window >= base_window + 0.003 or best_scene >= base_scene + 0.008)
    safety_pass = (
        int(best["same_frame_collision_count"]) == 0
        and float(best["pixel_collision_rate"]) <= 0.02
        and int(best["missing_mask_raster_count"]) == 0
    )

    gate_rows = [
        {
            "gate_id": "dev_MV_AP_window_ge_F2_base_plus_0p003_or_scene_ge_plus_0p008",
            "pass": phase2_dev_pass,
            "expected": f"MV_AP_window>={base_window + 0.003} or MV_AP_scene>={base_scene + 0.008}",
            "observed": f"MV_AP_window={best_window}; MV_AP_scene={best_scene}",
            "severity": "required_dev",
        },
        {
            "gate_id": "same_frame_collision_count_eq_0",
            "pass": int(best["same_frame_collision_count"]) == 0,
            "expected": "0",
            "observed": best["same_frame_collision_count"],
            "severity": "required",
        },
        {
            "gate_id": "pixel_collision_rate_le_0p02",
            "pass": float(best["pixel_collision_rate"]) <= 0.02,
            "expected": "<=0.02",
            "observed": best["pixel_collision_rate"],
            "severity": "required",
        },
        {
            "gate_id": "missing_mask_raster_count_eq_0",
            "pass": int(best["missing_mask_raster_count"]) == 0,
            "expected": "0",
            "observed": best["missing_mask_raster_count"],
            "severity": "required",
        },
        {
            "gate_id": "holdout_not_drop_more_than_0p005_MV_AP_window",
            "pass": False,
            "expected": "holdout evaluated for fixed best dev variant",
            "observed": "not_run_in_this_dev_strengthening_script",
            "severity": "required_holdout",
        },
    ]
    failure_rows = [
        {
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": "run holdout for fixed best Phase2 dev variant before claiming Phase2 success"
            if row["gate_id"].startswith("holdout")
            else "stop or change Phase2 repair family according to v99 plan",
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]

    decision = (
        "PASS_DEV_NEEDS_HOLDOUT"
        if phase2_dev_pass and safety_pass
        else "NO_GO_PHASE2_DEV_STRENGTHENING"
    )

    casebook_rows = [
        {
            "schema_version": "stream4d_v99_phase2_casebook_v1",
            "phase_id": "v99_phase2_f2_strengthening",
            "case_id": f"rank_{idx:02d}",
            "variant_id": row["variant_id"],
            "family": row["family"],
            "MV_AP_window": row["MV_AP_window"],
            "MV_AP50_window": row["MV_AP50_window"],
            "MV_AP_scene": row["MV_AP_scene"],
            "MV_AP50_scene": row["MV_AP50_scene"],
            "delta_vs_phase1_main_window": float(row["MV_AP_window"]) - float(baseline["MV_AP_window"]),
            "delta_vs_F2_base_window": float(row["MV_AP_window"]) - base_window,
            "score_vs_best_iou_spearman": row["score_vs_best_iou_spearman"],
        }
        for idx, row in enumerate(sorted(aggregate_rows, key=lambda r: float(r["MV_AP_window"]), reverse=True), start=1)
    ]

    summary = {
        "schema_version": "stream4d_v99_phase2_f2_strengthening_summary_v1",
        "phase_id": "v99_phase2_f2_strengthening",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "decision": decision,
        "phase2_dev_pass": phase2_dev_pass,
        "phase2_full_pass": False,
        "holdout_evaluated": False,
        "holdout_required_before_phase2_success": True,
        "parent_phase1_variant_id": parent_variant,
        "baseline_variant_id": baseline["variant_id"],
        "best_variant_id": best["variant_id"],
        "best_family": best["family"],
        "best_MV_AP_window": float(best["MV_AP_window"]),
        "best_MV_AP50_window": float(best["MV_AP50_window"]),
        "best_MV_AP_scene": float(best["MV_AP_scene"]),
        "best_MV_AP50_scene": float(best["MV_AP50_scene"]),
        "phase1_main_MV_AP_window": float(baseline["MV_AP_window"]),
        "phase1_main_MV_AP50_window": float(baseline["MV_AP50_window"]),
        "F2_base_full_dev_MV_AP_window": base_window,
        "F2_base_full_dev_MV_AP_scene": base_scene,
        "delta_best_vs_phase1_main_window": float(best["MV_AP_window"]) - float(baseline["MV_AP_window"]),
        "delta_best_vs_F2_base_window": float(best["MV_AP_window"]) - base_window,
        "same_frame_collision_count": int(best["same_frame_collision_count"]),
        "pixel_collision_rate": float(best["pixel_collision_rate"]),
        "missing_mask_raster_count": int(best["missing_mask_raster_count"]),
        "duplicate_tube_count": duplicate_tube_count,
        "broad_mask_selected_rate": broad_mask_selected_rate,
        "semantic_residual_margin_mean": semantic_margin_mean,
        "variant_count": len(aggregate_rows),
        "outputs": {
            "variant_config_rows": _rel(OUT_DIR / "variant_config_rows.csv"),
            "variant_metric_rows": _rel(OUT_DIR / "variant_metric_rows.csv"),
            "variant_metric_scene_rows": _rel(OUT_DIR / "variant_metric_scene_rows.csv"),
            "variant_gate_rows": _rel(OUT_DIR / "variant_gate_rows.csv"),
            "variant_failure_rows": _rel(OUT_DIR / "variant_failure_rows.csv"),
            "best_variant_summary": _rel(OUT_DIR / "best_variant_summary.json"),
            "casebook_rows": _rel(OUT_DIR / "casebook_rows.csv"),
            "mv_object_frame_mask_rows": _rel(OUT_DIR / "mv_object_frame_mask_rows.csv"),
        },
    }

    _write_csv(OUT_DIR / "variant_config_rows.csv", variant_config_rows)
    _write_csv(OUT_DIR / "variant_metric_scene_rows.csv", all_metric_scene_rows)
    _write_csv(OUT_DIR / "variant_metric_rows.csv", aggregate_rows)
    _write_csv(OUT_DIR / "variant_gate_rows.csv", gate_rows)
    _write_csv(OUT_DIR / "variant_failure_rows.csv", failure_rows)
    _write_csv(OUT_DIR / "casebook_rows.csv", casebook_rows)
    _write_csv(OUT_DIR / "mv_object_frame_mask_rows.csv", all_rows)
    _write_json(OUT_DIR / "best_variant_summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if decision == "PASS_DEV_NEEDS_HOLDOUT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
