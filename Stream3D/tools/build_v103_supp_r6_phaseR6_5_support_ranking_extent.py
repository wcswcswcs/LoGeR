#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.stats import spearmanr


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools.run_v65_scene_multiview_ap import SparseSceneIoU, _load_gt_2d, _summarize_iou  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
PHASE_ID = "v103_supp_r6_phaseR6_5_support_ranking_extent"
DEFAULT_OUT = AUDIT_ROOT / "v103_supp_r6_phase5_support_ranking_extent"
DEFAULT_FACT_LOCK_ROOT = AUDIT_ROOT / "v103_supp_r6_phase0_fact_lock"
DEFAULT_CURRENT_PHASE6D_ROOT = AUDIT_ROOT / "v103_phase6d_f2_skeleton_affinity_merge_phase9n_suppS1_d4rt48mix_s5repair_r4_directpair_guard"
DEFAULT_R6_LOCAL_AP_ROOT = AUDIT_ROOT / "v103_supp_r6_phase2_support_conditioned_local_ap"
DEFAULT_R6_DIAG_ROOT = AUDIT_ROOT / "v103_supp_r6_phase6_gt_coverage_inconsistency"
DEFAULT_D4RT_ROOT_BY_SCENE = {
    "scene0011_00": AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0011_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8",
    "scene0050_00": AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0050_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8",
}
D9_VARIANT = "D9_affinity_merge_tau065_top1_broad_support_veto"
D0_VARIANT = "D0_f2_original_replay"

R6SR_VARIANTS = [
    {
        "variant_id": "R6SR1_support_coverage_score",
        "description": "Score_base + beta * object support-density coverage",
        "term": "support_density_norm",
        "beta": 0.05,
        "risk_beta": 0.0,
    },
    {
        "variant_id": "R6SR2_support_consistency_score",
        "description": "Score_base + beta * mean log-support consistency",
        "term": "support_mean_log_norm",
        "beta": 0.05,
        "risk_beta": 0.0,
    },
    {
        "variant_id": "R6SR3_support_consistency_minus_veto_risk",
        "description": "Score_base + beta * support consistency - beta_v * support veto-risk proxy",
        "term": "support_mean_log_norm",
        "beta": 0.05,
        "risk_beta": 0.05,
    },
    {
        "variant_id": "R6SR4_anchor_supported_support_score",
        "description": "Score_base + beta * support consistency gated by base object confidence",
        "term": "anchor_supported_support_norm",
        "beta": 0.04,
        "risk_beta": 0.0,
        "anchor_supported_proxy": True,
    },
    {
        "variant_id": "R6SR5_temporal_support_persistence_score",
        "description": "Score_base + beta * temporal support persistence",
        "term": "support_persistence_norm",
        "beta": 0.05,
        "risk_beta": 0.0,
    },
]


def _project(path: Path | str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return p.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(path: Path, variant_id: str | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if variant_id is not None:
        rows = [r for r in rows if str(r.get("variant_id")) == str(variant_id)]
    return rows


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        if not np.isfinite(out):
            return default
        return out
    except Exception:
        return default


def _load_label_png(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(str(path))
    if label.ndim == 3:
        label = label[:, :, 0]
    return np.asarray(label, dtype=np.int32)


def _object_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("scene_id", "")), str(row.get("mv_object_id", ""))


def _mask_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row.get("scene_id", "")),
        int(_num(row.get("frame_local_index"), -1)),
        int(_num(row.get("selected_mask_id"), -1)),
    )


def _support_lookup(local_ap_root: Path, feature_variant: str) -> dict[tuple[str, int, int], float]:
    rows = _read_rows(local_ap_root / "phase6d_runs" / feature_variant / "merge_selected_rows.csv", D9_VARIANT)
    return {_mask_key(r): _num(r.get("support_count"), 0.0) for r in rows}


def _risk_lookup(local_ap_root: Path) -> dict[tuple[str, int, int], float]:
    raw = _support_lookup(local_ap_root, "R6F1_support005_specificity")
    veto = _support_lookup(local_ap_root, "R6F3_support010_specificity_semantic_vetoatten")
    out: dict[tuple[str, int, int], float] = {}
    for key, raw_value in raw.items():
        if raw_value <= 0:
            out[key] = 0.0
        else:
            out[key] = float(np.clip((raw_value - veto.get(key, 0.0)) / max(raw_value, 1.0), 0.0, 1.0))
    return out


def _normalise_by_scene(stats: dict[tuple[str, str], dict[str, Any]], source_key: str, target_key: str) -> None:
    by_scene: dict[str, list[float]] = defaultdict(list)
    for (scene, _oid), item in stats.items():
        by_scene[scene].append(float(item.get(source_key, 0.0)))
    scale = {
        scene: float(np.percentile(np.asarray(values, dtype=np.float64), 95)) if values else 0.0
        for scene, values in by_scene.items()
    }
    for (scene, _oid), item in stats.items():
        denom = max(scale.get(scene, 0.0), 1e-12)
        item[target_key] = float(np.clip(float(item.get(source_key, 0.0)) / denom, 0.0, 1.0))


def _build_object_stats(rows: list[dict[str, Any]], risk_by_mask: dict[tuple[str, int, int], float]) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_object_key(row)].append(row)
    stats: dict[tuple[str, str], dict[str, Any]] = {}
    for key, items in grouped.items():
        support_values = np.asarray([_num(r.get("support_count"), 0.0) for r in items], dtype=np.float64)
        area_values = np.asarray([max(_num(r.get("selected_mask_area"), 0.0), 1.0) for r in items], dtype=np.float64)
        base_scores = np.asarray([_num(r.get("object_score"), 0.0) for r in items], dtype=np.float64)
        risks = np.asarray([risk_by_mask.get(_mask_key(r), 0.0) for r in items], dtype=np.float64)
        frames = {int(_num(r.get("frame_local_index"), -1)) for r in items}
        support_frames = {int(_num(r.get("frame_local_index"), -1)) for r in items if _num(r.get("support_count"), 0.0) > 0}
        stats[key] = {
            "scene_id": key[0],
            "mv_object_id": key[1],
            "mask_count": len(items),
            "frame_count": len(frames),
            "support_frame_count": len(support_frames),
            "base_score": float(np.max(base_scores)) if base_scores.size else 0.0,
            "support_sum": float(np.sum(support_values)),
            "support_density": float(np.sum(support_values) / max(float(np.sum(area_values)), 1.0)),
            "support_mean_log": float(np.mean(np.log1p(support_values))) if support_values.size else 0.0,
            "support_persistence": float(len(support_frames) / max(len(frames), 1)),
            "veto_risk_proxy": float(np.mean(risks)) if risks.size else 0.0,
        }
    _normalise_by_scene(stats, "support_density", "support_density_norm")
    _normalise_by_scene(stats, "support_mean_log", "support_mean_log_norm")
    _normalise_by_scene(stats, "base_score", "base_score_norm")
    for item in stats.values():
        item["support_persistence_norm"] = float(np.clip(item.get("support_persistence", 0.0), 0.0, 1.0))
        item["anchor_supported_support_norm"] = float(item["support_mean_log_norm"] * item["base_score_norm"])
    return stats


def _make_term_map(
    *,
    stats: dict[tuple[str, str], dict[str, Any]],
    term_key: str,
    mode: str,
    seed: int,
) -> dict[tuple[str, str], float]:
    real = {key: float(item.get(term_key, 0.0)) for key, item in stats.items()}
    if mode == "real":
        return real
    rng = random.Random(seed)
    out: dict[tuple[str, str], float] = {}
    for scene in sorted({key[0] for key in stats}):
        keys = [key for key in sorted(stats) if key[0] == scene]
        values = [real[key] for key in keys]
        if mode == "shuffled":
            rng.shuffle(values)
        elif mode == "stale_proxy":
            values = values[1:] + values[:1] if values else values
        else:
            raise ValueError(f"unknown term map mode={mode}")
        for key, value in zip(keys, values):
            out[key] = float(value)
    return out


def _apply_score_variant(
    rows: list[dict[str, Any]],
    *,
    variant_id: str,
    stats: dict[tuple[str, str], dict[str, Any]],
    term_map: dict[tuple[str, str], float],
    term_key: str,
    beta: float,
    risk_beta: float,
    score_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    emitted_objects: set[tuple[str, str]] = set()
    for row in rows:
        key = _object_key(row)
        item = stats[key]
        base_score = float(item["base_score"])
        support_component = float(term_map.get(key, 0.0))
        risk_component = float(item.get("veto_risk_proxy", 0.0))
        new_score = float(base_score + float(beta) * support_component - float(risk_beta) * risk_component)
        new = dict(row)
        new["variant_id"] = variant_id
        new["object_score"] = new_score
        new["score"] = new_score
        new["base_object_score"] = base_score
        new["support_score_component"] = support_component
        new["support_risk_component"] = risk_component
        new["score_policy"] = score_mode
        out.append(new)
        if key not in emitted_objects:
            emitted_objects.add(key)
            score_rows.append(
                {
                    "schema_version": "stream4d_v103_supp_r6_phaseR6_5_object_score_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": variant_id,
                    "score_mode": score_mode,
                    "scene_id": key[0],
                    "mv_object_id": key[1],
                    "base_object_score": base_score,
                    "rescored_object_score": new_score,
                    "support_score_component": support_component,
                    "support_risk_component": risk_component,
                    "support_sum": item["support_sum"],
                    "support_density_norm": item["support_density_norm"],
                    "support_mean_log_norm": item["support_mean_log_norm"],
                    "support_persistence_norm": item["support_persistence_norm"],
                    "anchor_supported_support_norm": item["anchor_supported_support_norm"],
                    "veto_risk_proxy": item["veto_risk_proxy"],
                    "mask_count": item["mask_count"],
                    "frame_count": item["frame_count"],
                    "support_frame_count": item["support_frame_count"],
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    return out, score_rows


def _evaluate_rows(
    *,
    variant_id: str,
    rows: list[dict[str, Any]],
    phase2_summaries: dict[str, dict[str, Any]],
    min_pred_pixels: int,
    min_gt_pixels: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    per_scene_metrics: list[dict[str, Any]] = []
    pred_diag_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    pixel_collision_count = 0
    missing_mask_raster_count = 0
    pred_positive_pixels = 0
    for scene, summary in sorted(phase2_summaries.items()):
        frame_ids = [int(v) for v in summary["frame_ids"]]
        mask_root = _project(summary["mask_root"])
        scene_rows = [r for r in rows if str(r.get("scene_id")) == scene]
        by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in scene_rows:
            local = int(_num(row.get("frame_local_index"), -1))
            if 0 <= local < len(frame_ids):
                new = dict(row)
                new["frame_id"] = int(frame_ids[local])
                by_frame[int(frame_ids[local])].append(new)
        acc = SparseSceneIoU()
        object_index: dict[str, int] = {}
        object_scores: dict[str, float] = {}
        for frame_id in frame_ids:
            mask_path = mask_root / f"{int(frame_id)}.png"
            if not mask_path.exists():
                missing_mask_raster_count += 1
                gt = _load_gt_2d(scene, frame_id, (968, 1296))
                acc.add(np.zeros(gt.shape, dtype=np.int64), gt)
                continue
            label = _load_label_png(mask_path)
            gt = _load_gt_2d(scene, frame_id, label.shape)
            pred = np.zeros(label.shape, dtype=np.int64)
            for row in sorted(by_frame.get(int(frame_id), []), key=lambda r: (-float(r.get("object_score", 0.0)), str(r.get("mv_object_id", "")))):
                oid = str(row.get("mv_object_id", ""))
                if oid not in object_index:
                    object_index[oid] = len(object_index) + 1
                object_scores[oid] = max(object_scores.get(oid, -1e9), float(row.get("object_score", 0.0)))
                mid = int(_num(row.get("selected_mask_id"), -1))
                pixels = label == mid
                if not np.any(pixels):
                    missing_mask_raster_count += 1
                    continue
                overlap = pixels & (pred > 0)
                pixel_collision_count += int(np.count_nonzero(overlap))
                pred[(pred == 0) & pixels] = int(object_index[oid])
                selected = dict(row)
                selected["frame_id"] = int(frame_id)
                selected["selected_mask_area"] = int(np.count_nonzero(pixels))
                selected_rows.append(selected)
            pred_positive_pixels += int(np.count_nonzero(pred > 0))
            acc.add(pred, gt)
        input_scores = np.ones((len(object_index),), dtype=np.float32)
        oid_by_pred_id = {idx: oid for oid, idx in object_index.items()}
        for oid, idx in object_index.items():
            input_scores[int(idx) - 1] = float(object_scores.get(oid, 1.0))
        metric, iou, pred_ids, _gt_ids = _summarize_iou(
            accumulator=acc,
            min_pred_pixels=int(min_pred_pixels),
            min_gt_pixels=int(min_gt_pixels),
            score_mode="input",
            input_scores=input_scores,
        )
        per_scene_metrics.append(
            {
                "scene_id": scene,
                "MV_AP_window": metric.get("ap") or 0.0,
                "MV_AP50_window": metric.get("ap50") or 0.0,
                "MV_AP25_window": metric.get("ap25") or 0.0,
                "ScoreFreeMatch25_window": metric.get("score_free_match_at_025", {}).get("f1", 0.0),
                "ScoreFreeMatch50_window": metric.get("score_free_match_at_050", {}).get("f1", 0.0),
                "evaluated_pred_count": metric.get("evaluated_pred_count", 0),
                "evaluated_gt_count": metric.get("evaluated_gt_count", 0),
            }
        )
        pred_best = np.max(iou, axis=1) if iou.size and iou.shape[1] else np.zeros((len(pred_ids),), dtype=np.float32)
        for idx, pred_id in enumerate(pred_ids):
            oid = oid_by_pred_id.get(int(pred_id), "")
            score = float(input_scores[int(pred_id) - 1]) if int(pred_id) - 1 < input_scores.shape[0] else 1.0
            best_iou = float(pred_best[idx]) if idx < pred_best.shape[0] else 0.0
            pred_diag_rows.append(
                {
                    "schema_version": "stream4d_v103_supp_r6_phaseR6_5_pred_score_diag_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": variant_id,
                    "scene_id": scene,
                    "mv_object_id": oid,
                    "pred_id": int(pred_id),
                    "object_score": score,
                    "diagnostic_best_gt_iou": best_iou,
                    "diagnostic_tp50": int(best_iou >= 0.50),
                    "diagnostic_fp25": int(best_iou < 0.25),
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_eval": True,
                    "uses_future": False,
                }
            )
    aggregate: dict[str, Any] = {
        "schema_version": "stream4d_v103_supp_r6_phaseR6_5_variant_metric_row_v1",
        "phase_id": PHASE_ID,
        "variant_id": variant_id,
        "scene_count": len(per_scene_metrics),
        "metric_scope": "dev c0001 current subset; score-only ranking/extent diagnostic",
        "iou_backend": "v65_sparse_scene_iou_cpu",
        "same_frame_collision_count": 0,
        "pixel_collision_count": int(pixel_collision_count),
        "pixel_collision_rate": float(pixel_collision_count / max(pred_positive_pixels, 1)),
        "missing_mask_raster_count": int(missing_mask_raster_count),
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
    }
    for key in ["MV_AP_window", "MV_AP50_window", "MV_AP25_window", "ScoreFreeMatch25_window", "ScoreFreeMatch50_window"]:
        vals = [float(row[key]) for row in per_scene_metrics]
        aggregate[key] = float(np.mean(vals)) if vals else 0.0
    return aggregate, per_scene_metrics, pred_diag_rows, selected_rows


def _score_diag(metric: dict[str, Any], pred_rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = np.asarray([_num(r.get("object_score"), 0.0) for r in pred_rows], dtype=np.float64)
    tp_scores = np.asarray([_num(r.get("object_score"), 0.0) for r in pred_rows if int(_num(r.get("diagnostic_tp50"), 0))], dtype=np.float64)
    fp_scores = np.asarray([_num(r.get("object_score"), 0.0) for r in pred_rows if int(_num(r.get("diagnostic_fp25"), 0))], dtype=np.float64)
    top = sorted(pred_rows, key=lambda r: -_num(r.get("object_score"), 0.0))[:20]
    top_tp = [_num(r.get("object_score"), 0.0) for r in top if int(_num(r.get("diagnostic_tp50"), 0))]
    top_fp = [_num(r.get("object_score"), 0.0) for r in top if int(_num(r.get("diagnostic_fp25"), 0))]
    return {
        "true_positive_score_mean_diagnostic": float(np.mean(tp_scores)) if tp_scores.size else 0.0,
        "false_positive_score_mean_diagnostic": float(np.mean(fp_scores)) if fp_scores.size else 0.0,
        "top20_true_positive_support_score_mean": float(np.mean(top_tp)) if top_tp else 0.0,
        "top20_false_positive_support_score_mean": float(np.mean(top_fp)) if top_fp else 0.0,
        "pred_score_mean": float(np.mean(scores)) if scores.size else 0.0,
        "evaluated_pred_count": len(pred_rows),
    }


def _spearman(base_rows: list[dict[str, Any]], score_rows: list[dict[str, Any]]) -> float:
    base = {(_object_key(r)): _num(r.get("object_score"), 0.0) for r in base_rows}
    new = {(str(r.get("scene_id")), str(r.get("mv_object_id"))): _num(r.get("rescored_object_score"), 0.0) for r in score_rows}
    keys = sorted(set(base) & set(new))
    if len(keys) < 2:
        return 1.0
    corr = spearmanr([base[k] for k in keys], [new[k] for k in keys]).correlation
    return float(corr) if np.isfinite(corr) else 0.0


def build(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _project(args.output_root)
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"output root already exists and is non-empty; pass --force: {_rel(out)}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")

    fact = _read_json(_project(args.fact_lock_root) / "summary.json")
    replay_ap = _num(fact.get("current_replay_MV_AP_window"), 0.0)
    replay_ap50 = _num(fact.get("current_replay_MV_AP50_window"), 0.0)
    current_root = _project(args.current_phase6d_root)
    local_ap_root = _project(args.r6_local_ap_root)
    phase2_summaries = {
        "scene0011_00": _read_json(_project(args.scene0011_d4rt_root) / "summary.json"),
        "scene0050_00": _read_json(_project(args.scene0050_d4rt_root) / "summary.json"),
    }
    base_rows = _read_rows(current_root / "merge_selected_rows.csv", D9_VARIANT)
    if not base_rows:
        raise RuntimeError(f"missing current D9 selected rows: {current_root / 'merge_selected_rows.csv'}")
    risk_by_mask = _risk_lookup(local_ap_root)
    stats = _build_object_stats(base_rows, risk_by_mask)

    metric_rows: list[dict[str, Any]] = []
    scene_metric_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    object_score_rows: list[dict[str, Any]] = []
    pred_score_diag_rows: list[dict[str, Any]] = []
    selected_rows_all: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    control_compare_rows: list[dict[str, Any]] = []

    base_metric, base_scene_rows, base_pred_diag, base_selected = _evaluate_rows(
        variant_id="R6SR0_current_d9_score_replay",
        rows=[dict(r, variant_id="R6SR0_current_d9_score_replay") for r in base_rows],
        phase2_summaries=phase2_summaries,
        min_pred_pixels=int(args.min_pred_pixels),
        min_gt_pixels=int(args.min_gt_pixels),
    )
    base_metric.update(
        {
            "support_ranking_variant": "base_current_d9",
            "score_policy": "base_object_score",
            "object_score_spearman_change": 1.0,
            **_score_diag(base_metric, base_pred_diag),
        }
    )
    metric_rows.append(base_metric)
    scene_metric_rows.extend([r | {"variant_id": "R6SR0_current_d9_score_replay"} for r in base_scene_rows])
    pred_score_diag_rows.extend(base_pred_diag)
    selected_rows_all.extend(base_selected)
    base_scorefree50 = float(base_metric["ScoreFreeMatch50_window"])

    for idx, spec in enumerate(R6SR_VARIANTS):
        variant_id = str(spec["variant_id"])
        real_terms = _make_term_map(stats=stats, term_key=str(spec["term"]), mode="real", seed=int(args.random_seed) + idx)
        shuffled_terms = _make_term_map(stats=stats, term_key=str(spec["term"]), mode="shuffled", seed=int(args.random_seed) + idx)
        stale_terms = _make_term_map(stats=stats, term_key=str(spec["term"]), mode="stale_proxy", seed=int(args.random_seed) + idx)
        runs = [
            ("real", variant_id, real_terms, False),
            ("shuffled_support_control", f"{variant_id}__shuffled_support_control", shuffled_terms, True),
            ("stale_support_proxy_control", f"{variant_id}__stale_support_proxy_control", stale_terms, True),
        ]
        run_metrics: dict[str, dict[str, Any]] = {}
        for control_id, run_id, terms, is_control in runs:
            rescored_rows, score_rows = _apply_score_variant(
                base_rows,
                variant_id=run_id,
                stats=stats,
                term_map=terms,
                term_key=str(spec["term"]),
                beta=float(spec["beta"]),
                risk_beta=float(spec.get("risk_beta", 0.0)),
                score_mode=control_id,
            )
            metric, per_scene, pred_diag, selected = _evaluate_rows(
                variant_id=run_id,
                rows=rescored_rows,
                phase2_summaries=phase2_summaries,
                min_pred_pixels=int(args.min_pred_pixels),
                min_gt_pixels=int(args.min_gt_pixels),
            )
            diag = _score_diag(metric, pred_diag)
            metric.update(
                {
                    "support_ranking_variant": variant_id,
                    "score_policy": spec["description"],
                    "control_id": control_id,
                    "is_control": bool(is_control),
                    "term_key": spec["term"],
                    "beta": float(spec["beta"]),
                    "risk_beta": float(spec.get("risk_beta", 0.0)),
                    "anchor_supported_proxy": bool(spec.get("anchor_supported_proxy", False)),
                    "object_score_spearman_change": _spearman(base_rows, score_rows),
                    **diag,
                }
            )
            if is_control:
                control_rows.append(metric)
            else:
                metric_rows.append(metric)
            scene_metric_rows.extend([r | {"variant_id": run_id, "control_id": control_id, "support_ranking_variant": variant_id} for r in per_scene])
            object_score_rows.extend(score_rows)
            pred_score_diag_rows.extend(pred_diag)
            selected_rows_all.extend(selected)
            run_metrics[control_id] = metric
        real = run_metrics["real"]
        shuffled = run_metrics["shuffled_support_control"]
        stale = run_metrics["stale_support_proxy_control"]
        real_minus_shuffled = float(real["MV_AP_window"]) - float(shuffled["MV_AP_window"])
        real_minus_stale_proxy = float(real["MV_AP_window"]) - float(stale["MV_AP_window"])
        real_minus_replay = float(real["MV_AP_window"]) - replay_ap
        real_ap50_minus_replay = float(real["MV_AP50_window"]) - replay_ap50
        compare = {
            "schema_version": "stream4d_v103_supp_r6_phaseR6_5_control_comparison_row_v1",
            "phase_id": PHASE_ID,
            "variant_id": variant_id,
            "real_MV_AP_window": real["MV_AP_window"],
            "real_MV_AP50_window": real["MV_AP50_window"],
            "real_ScoreFreeMatch50_window": real["ScoreFreeMatch50_window"],
            "shuffled_MV_AP_window": shuffled["MV_AP_window"],
            "stale_proxy_MV_AP_window": stale["MV_AP_window"],
            "real_minus_shuffled_score_repair": real_minus_shuffled,
            "real_minus_stale_proxy_MV_AP_window": real_minus_stale_proxy,
            "real_minus_replay_MV_AP_window": real_minus_replay,
            "real_minus_replay_MV_AP50_window": real_ap50_minus_replay,
            "stale_support_control_exact_available": False,
            "uses_gt_for_prediction": False,
            "uses_gt_for_eval": True,
            "uses_future": False,
        }
        control_compare_rows.append(compare)
        gate_specs = [
            ("MV_AP_window_ge_replay_plus_0p003", float(real["MV_AP_window"]), replay_ap + 0.003, ">="),
            ("MV_AP50_window_ge_replay_plus_0p006", float(real["MV_AP50_window"]), replay_ap50 + 0.006, ">="),
            ("real_minus_shuffled_score_repair_ge_0p003", real_minus_shuffled, 0.003, ">="),
            ("ScoreFreeMatch50_window_not_drop_gt_0p002", float(real["ScoreFreeMatch50_window"]), base_scorefree50 - 0.002, ">="),
            ("same_frame_collision_count_eq_0", int(real["same_frame_collision_count"]), 0, "=="),
            ("missing_mask_raster_count_eq_0", int(real["missing_mask_raster_count"]), 0, "=="),
            ("stale_support_control_exact_available", 0, 1, "=="),
        ]
        for gate_id, observed, required, op in gate_specs:
            passed = observed == required if op == "==" else observed >= required
            gate_rows.append(
                {
                    "schema_version": "stream4d_v103_supp_r6_phaseR6_5_gate_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": variant_id,
                    "gate_id": gate_id,
                    "pass": bool(passed),
                    "observed": observed,
                    "required": required,
                    "operator": op,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
            if not passed:
                failure_rows.append(
                    {
                        "schema_version": "stream4d_v103_supp_r6_phaseR6_5_failure_row_v1",
                        "phase_id": PHASE_ID,
                        "variant_id": variant_id,
                        "blocker": gate_id,
                        "detail": f"observed={observed} required={required}",
                        "repair_direction": "Do not promote R6-5; support ranking must beat shuffled and exact stale controls without AP50/ScoreFree safety loss.",
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )

    fully_passing = []
    for spec in R6SR_VARIANTS:
        vid = str(spec["variant_id"])
        sub = [row for row in gate_rows if row["variant_id"] == vid]
        if sub and all(bool(row["pass"]) for row in sub):
            fully_passing.append(vid)
    best_real = max(
        [row for row in metric_rows if str(row.get("variant_id")) != "R6SR0_current_d9_score_replay"],
        key=lambda r: float(r.get("MV_AP_window", 0.0)),
        default={},
    )
    artifact_rows = [
        {
            "schema_version": "stream4d_v103_supp_r6_phaseR6_5_artifact_row_v1",
            "phase_id": PHASE_ID,
            "artifact_role": role,
            "path": _rel(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for role, path in [
            ("summary", out / "summary.json"),
            ("variant_metric_rows", out / "variant_metric_rows.csv"),
            ("control_metric_rows", out / "control_metric_rows.csv"),
            ("control_comparison_rows", out / "control_comparison_rows.csv"),
            ("object_score_rows", out / "object_score_rows.csv"),
            ("pred_score_diagnostic_rows", out / "pred_score_diagnostic_rows.csv"),
            ("rescored_selected_rows", out / "rescored_selected_rows.csv"),
            ("gate_rows", out / "gate_rows.csv"),
            ("failure_rows", out / "failure_rows.csv"),
            ("last_command", out / "last_command.txt"),
        ]
    ]

    _write_csv(out / "variant_metric_rows.csv", metric_rows)
    _write_csv(out / "control_metric_rows.csv", control_rows)
    _write_csv(out / "scene_metric_rows.csv", scene_metric_rows)
    _write_csv(out / "control_comparison_rows.csv", control_compare_rows)
    _write_csv(out / "object_score_rows.csv", object_score_rows)
    _write_csv(out / "pred_score_diagnostic_rows.csv", pred_score_diag_rows)
    _write_csv(out / "rescored_selected_rows.csv", selected_rows_all)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    _write_csv(out / "artifact_rows.csv", artifact_rows)

    phase_pass = bool(fully_passing)
    summary = {
        "schema_version": "stream4d_v103_supp_r6_phaseR6_5_support_ranking_extent_summary_v1",
        "phase_id": PHASE_ID,
        "decision": "PASS_R6_5_SUPPORT_RANKING_EXTENT_SIGNAL" if phase_pass else "NO_GO_R6_5_SUPPORT_RANKING_EXTENT",
        "phase_r6_5_pass": bool(phase_pass),
        "failure_count": len(failure_rows),
        "tested_r6sr_variants": [str(v["variant_id"]) for v in R6SR_VARIANTS],
        "fully_passing_r6sr_variants": fully_passing,
        "best_real_variant_id": best_real.get("variant_id", ""),
        "best_real_MV_AP_window": best_real.get("MV_AP_window", ""),
        "best_real_MV_AP50_window": best_real.get("MV_AP50_window", ""),
        "base_current_d9_MV_AP_window": base_metric["MV_AP_window"],
        "base_current_d9_MV_AP50_window": base_metric["MV_AP50_window"],
        "replay_MV_AP_window": replay_ap,
        "replay_MV_AP50_window": replay_ap50,
        "stale_support_control_exact_available": False,
        "stale_support_control_proxy_available": True,
        "runs_AP": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
        "runtime_sec": time.time() - t0,
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "variant_metric_rows": _rel(out / "variant_metric_rows.csv"),
            "control_metric_rows": _rel(out / "control_metric_rows.csv"),
            "control_comparison_rows": _rel(out / "control_comparison_rows.csv"),
            "object_score_rows": _rel(out / "object_score_rows.csv"),
            "pred_score_diagnostic_rows": _rel(out / "pred_score_diagnostic_rows.csv"),
            "rescored_selected_rows": _rel(out / "rescored_selected_rows.csv"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "artifact_rows": _rel(out / "artifact_rows.csv"),
        },
        "truthfulness_note": (
            "R6-5 reuses current locked D9 membership and changes only object scores. "
            "It has exact shuffled controls, but no exact previous/non-adjacent stale support artifact for both selected scenes; "
            "therefore stale support is reported as a proxy and blocks method promotion."
        ),
    }
    _write_json(out / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream4D v103 R6-5 support ranking/extent diagnostic.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--fact-lock-root", default=str(DEFAULT_FACT_LOCK_ROOT))
    parser.add_argument("--current-phase6d-root", default=str(DEFAULT_CURRENT_PHASE6D_ROOT))
    parser.add_argument("--r6-local-ap-root", default=str(DEFAULT_R6_LOCAL_AP_ROOT))
    parser.add_argument("--r6-diag-root", default=str(DEFAULT_R6_DIAG_ROOT))
    parser.add_argument("--scene0011-d4rt-root", default=str(DEFAULT_D4RT_ROOT_BY_SCENE["scene0011_00"]))
    parser.add_argument("--scene0050-d4rt-root", default=str(DEFAULT_D4RT_ROOT_BY_SCENE["scene0050_00"]))
    parser.add_argument("--min-pred-pixels", type=int, default=20)
    parser.add_argument("--min-gt-pixels", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=1036)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = build(args)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if summary["phase_r6_5_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
