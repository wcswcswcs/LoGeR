#!/usr/bin/env python3
"""Decompose v96 Phase6 failures with scene-safe MV IoU diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_v65_scene_multiview_ap import SparseSceneIoU, _load_gt_2d, _summarize_iou  # noqa: E402


PHASE_ID = "v96_phase9_error_decomposition"
RUN_ID = "v96_phase9_error_decomposition"
DEFAULT_PHASE6 = ROOT / "outputs/audit/v96_phase6_render_snap_w0020_segmented_r4_D3_sceneoffset_ranked_suppressed_more"
DEFAULT_SOURCE_ROWS = ROOT / "outputs/audit/v95_phase1_physical_source_registry/source_container_rows.csv"
DEFAULT_OUT = ROOT / "outputs/audit/v96_phase9_error_decomposition_w0020_segmented_r4_D3"


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in {"Stream3D", "Open-d4rt", "docs"}:
        return REPO_ROOT / p
    return ROOT / p


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return path.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return _rel(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_label(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to read label image: {path}")
    if image.ndim == 3:
        image = image[..., 0]
    return np.asarray(image, dtype=np.int64)


def _mask_path_lookup(source_rows: Path) -> dict[tuple[str, str, int], Path]:
    out: dict[tuple[str, str, int], Path] = {}
    for row in _read_csv(source_rows):
        key = (row.get("scene_id", ""), row.get("window_id", ""), int(_num(row.get("frame_id"))))
        raw = row.get("mask_path", "")
        if raw and key not in out:
            out[key] = _project(raw)
    return out


def _best_variant(phase6_root: Path, requested: str) -> str:
    if requested:
        return requested
    summary = _load_json(phase6_root / "summary.json")
    return str((summary.get("best_variant") or {}).get("readout_variant") or "R9_snap_min6_frames")


def _top_rows(
    iou: np.ndarray,
    row_ids: list[int],
    col_ids: list[int],
    *,
    row_name: str,
    col_name: str,
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if iou.size == 0:
        return rows
    flat = np.argsort(iou.reshape(-1))[::-1]
    for flat_idx in flat[: int(top_k)]:
        ridx = int(flat_idx // max(1, iou.shape[1]))
        cidx = int(flat_idx % max(1, iou.shape[1]))
        value = float(iou[ridx, cidx])
        if value <= 0:
            break
        rows.append({row_name: int(row_ids[ridx]), col_name: int(col_ids[cidx]), "iou": value})
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    phase6_root = _project(args.phase6_root)
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    variant = _best_variant(phase6_root, args.variant)
    mask_lookup = _mask_path_lookup(_project(args.source_rows))
    rows = [row for row in _read_csv(phase6_root / "mv_object_frame_mask_rows.csv") if row.get("readout_variant") == variant]
    if not rows:
        raise ValueError(f"no mv_object_frame_mask_rows for variant={variant}")
    eval_frame_keys = {
        (row.get("scene_id", ""), row.get("window_id", ""), int(_num(row.get("frame_id"))))
        for row in _read_csv(phase6_root / "rendered_support_rows.csv")
        if row.get("scene_id", "") and row.get("window_id", "")
    }

    by_frame: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    object_index: dict[str, int] = {}
    index_to_object: dict[int, str] = {}
    for row in rows:
        oid = row.get("object_id", "")
        if oid not in object_index:
            object_index[oid] = len(object_index) + 1
            index_to_object[object_index[oid]] = oid
        by_frame[(row.get("scene_id", ""), row.get("window_id", ""), int(_num(row.get("frame_id"))))].append(row)

    scene_gt_offsets = {scene: (idx + 1) * 1_000_000 for idx, scene in enumerate(sorted({key[0] for key in eval_frame_keys}))}
    acc = SparseSceneIoU()
    frame_rows: list[dict[str, Any]] = []
    for key in sorted(set(eval_frame_keys) | set(by_frame)):
        frame_rows_in = by_frame.get(key, [])
        scene, window, frame_id = key
        mask_path = mask_lookup.get(key)
        if mask_path is None or not mask_path.exists():
            frame_rows.append({"scene_id": scene, "window_id": window, "frame_id": frame_id, "status": "missing_mask"})
            continue
        label = _load_label(mask_path)
        pred = np.zeros(label.shape, dtype=np.int64)
        for row in frame_rows_in:
            pred[label == int(_num(row.get("selected_mask_id")))] = object_index[row.get("object_id", "")]
        gt = _load_gt_2d(scene, frame_id, label.shape)
        gt = np.where(gt > 0, gt + int(scene_gt_offsets.get(scene, 0)), 0).astype(np.int64, copy=False)
        acc.add(pred, gt)
        frame_rows.append(
            {
                "scene_id": scene,
                "window_id": window,
                "frame_id": frame_id,
                "status": "evaluated",
                "object_count": len(frame_rows_in),
                "eval_frame_scope": "phase6_rendered_support_frame_keys_even_if_no_prediction",
                "pred_positive_pixels": int(np.count_nonzero(pred > 0)),
                "gt_positive_pixels": int(np.count_nonzero(gt > 0)),
                "gt_scene_offset": int(scene_gt_offsets.get(scene, 0)),
            }
        )

    built = acc.build(min_pred_pixels=int(args.min_pred_pixels), min_gt_pixels=int(args.min_gt_pixels))
    iou = np.asarray(built["iou"], dtype=np.float32)
    pred_ids = list(built["pred_ids"])
    gt_ids = list(built["gt_ids"])
    summary, _iou2, _pred_ids2, _gt_ids2 = _summarize_iou(
        accumulator=acc,
        min_pred_pixels=int(args.min_pred_pixels),
        min_gt_pixels=int(args.min_gt_pixels),
        score_mode="constant",
        input_scores=None,
    )
    gt_best = np.max(iou, axis=0) if iou.shape[0] and iou.shape[1] else np.asarray([], dtype=np.float32)
    pred_best = np.max(iou, axis=1) if iou.shape[0] and iou.shape[1] else np.asarray([], dtype=np.float32)

    gt_rows = []
    for idx, gt_id in enumerate(gt_ids):
        best_pred_idx = int(np.argmax(iou[:, idx])) if iou.shape[0] else -1
        gt_rows.append(
            {
                "schema_version": "stream4d_v96_gt_top_iou_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "readout_variant": variant,
                "gt_id_scene_offset": int(gt_id),
                "best_pred_id": int(pred_ids[best_pred_idx]) if best_pred_idx >= 0 else 0,
                "best_pred_object_id": index_to_object.get(int(pred_ids[best_pred_idx]), "") if best_pred_idx >= 0 else "",
                "best_iou": float(gt_best[idx]) if gt_best.size else 0.0,
            }
        )
    pred_rows = []
    for idx, pred_id in enumerate(pred_ids):
        best_gt_idx = int(np.argmax(iou[idx])) if iou.shape[1] else -1
        pred_rows.append(
            {
                "schema_version": "stream4d_v96_pred_top_iou_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "readout_variant": variant,
                "pred_id": int(pred_id),
                "pred_object_id": index_to_object.get(int(pred_id), ""),
                "best_gt_id_scene_offset": int(gt_ids[best_gt_idx]) if best_gt_idx >= 0 else 0,
                "best_iou": float(pred_best[idx]) if pred_best.size else 0.0,
            }
        )

    metric_rows = _read_csv(phase6_root / "render_variant_metric_rows.csv")
    best_metric = next((row for row in metric_rows if row.get("readout_variant") == variant), {})
    support_rows_all = _read_csv(phase6_root / "support_iou_rows.csv")
    emitted_keys = {(row.get("object_id", ""), row.get("scene_id", ""), row.get("window_id", ""), int(_num(row.get("frame_id"))), int(_num(row.get("selected_mask_id")))) for row in rows}
    support_values = []
    visible_values = []
    for row in support_rows_all:
        key = (row.get("object_id", ""), row.get("scene_id", ""), row.get("window_id", ""), int(_num(row.get("frame_id"))), int(_num(row.get("selected_mask_id"))))
        if key in emitted_keys:
            support_values.append(_num(row.get("support_iou")))
            visible_values.append(_num(row.get("visible_micro_count")))

    grouping_rows = [
        {
            "schema_version": "stream4d_v96_grouping_error_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "readout_variant": variant,
            "evaluated_pred_count": int(len(pred_ids)),
            "evaluated_gt_count": int(len(gt_ids)),
            "gt_best_iou_mean": float(np.mean(gt_best)) if gt_best.size else 0.0,
            "gt_best_iou_p50": float(np.percentile(gt_best, 50)) if gt_best.size else 0.0,
            "gt_best_iou_p90": float(np.percentile(gt_best, 90)) if gt_best.size else 0.0,
            "gt_best_iou_max": float(np.max(gt_best)) if gt_best.size else 0.0,
            "gt_recall_best_iou_ge_025": float(np.mean(gt_best >= 0.25)) if gt_best.size else 0.0,
            "gt_recall_best_iou_ge_050": float(np.mean(gt_best >= 0.50)) if gt_best.size else 0.0,
            "pred_best_iou_mean": float(np.mean(pred_best)) if pred_best.size else 0.0,
            "pred_best_iou_p50": float(np.percentile(pred_best, 50)) if pred_best.size else 0.0,
            "pred_best_iou_max": float(np.max(pred_best)) if pred_best.size else 0.0,
        }
    ]
    ranking_rows = [
        {
            "schema_version": "stream4d_v96_ranking_error_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "readout_variant": variant,
            "MV_AP_window": _num(best_metric.get("MV_AP_window")),
            "MV_AP50_window": _num(best_metric.get("MV_AP50_window")),
            "ScoreFreeMatch50_window": _num(best_metric.get("ScoreFreeMatch50_window")),
            "scorefree_minus_ap50": _num(best_metric.get("ScoreFreeMatch50_window")) - _num(best_metric.get("MV_AP50_window")),
            "emitted_object_count": int(_num(best_metric.get("emitted_object_count"))),
            "emitted_object_frame_count": int(_num(best_metric.get("emitted_object_frame_count"))),
            "no_emit_rate": _num(best_metric.get("no_emit_rate")),
        }
    ]
    geometry_rows = [
        {
            "schema_version": "stream4d_v96_geometry_projection_casebook_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "readout_variant": variant,
            "support_iou_mean": float(np.mean(support_values)) if support_values else 0.0,
            "support_iou_p10": float(np.percentile(support_values, 10)) if support_values else 0.0,
            "support_iou_p50": float(np.percentile(support_values, 50)) if support_values else 0.0,
            "support_iou_p90": float(np.percentile(support_values, 90)) if support_values else 0.0,
            "support_iou_lt_0p25_rate": float(np.mean(np.asarray(support_values) < 0.25)) if support_values else 0.0,
        }
    ]
    density_rows = [
        {
            "schema_version": "stream4d_v96_primitive_density_casebook_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "readout_variant": variant,
            "visible_micro_count_mean": float(np.mean(visible_values)) if visible_values else 0.0,
            "visible_micro_count_p10": float(np.percentile(visible_values, 10)) if visible_values else 0.0,
            "visible_micro_count_p50": float(np.percentile(visible_values, 50)) if visible_values else 0.0,
            "visible_micro_count_p90": float(np.percentile(visible_values, 90)) if visible_values else 0.0,
        }
    ]
    blocker_labels: list[str] = []
    if grouping_rows[0]["gt_recall_best_iou_ge_050"] < 0.2:
        blocker_labels.append("OBJECT_BIRTH_GROUPING_BLOCKER")
    if geometry_rows[0]["support_iou_mean"] < 0.5:
        blocker_labels.append("RENDER_SUPPORT_ALIGNMENT_BLOCKER")
    if ranking_rows[0]["scorefree_minus_ap50"] > 0.03:
        blocker_labels.append("RANKING_BLOCKER")
    if not blocker_labels:
        blocker_labels.append("UNRESOLVED_PHASE6_AP_BLOCKER")

    _write_csv(output_root / "gt_top_iou_rows.csv", gt_rows)
    _write_csv(output_root / "pred_top_iou_rows.csv", pred_rows)
    _write_csv(output_root / "top_iou_pair_rows.csv", _top_rows(iou, pred_ids, gt_ids, row_name="pred_id", col_name="gt_id_scene_offset", top_k=int(args.top_k)))
    _write_csv(output_root / "grouping_error_rows.csv", grouping_rows)
    _write_csv(output_root / "ranking_error_rows.csv", ranking_rows)
    _write_csv(output_root / "geometry_projection_casebook.csv", geometry_rows)
    _write_csv(output_root / "primitive_density_casebook.csv", density_rows)
    _write_csv(output_root / "frame_eval_rows.csv", frame_rows)
    summary_payload = {
        "schema": "stream4d_v96_phase9_error_decomposition_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": _created_at(),
        "decision": "NO_GO_V96_PHASE6_DIAGNOSED",
        "phase6_root": _rel(phase6_root),
        "output_root": _rel(output_root),
        "readout_variant": variant,
        "metric_summary": summary,
        "grouping_error": grouping_rows[0],
        "ranking_error": ranking_rows[0],
        "geometry_projection": geometry_rows[0],
        "primitive_density": density_rows[0],
        "blocker_labels": blocker_labels,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_eval": True,
        "uses_future": False,
        "runtime_total_sec": float(time.time() - started),
    }
    _write_json(output_root / "summary.json", summary_payload)
    print(json.dumps({"decision": summary_payload["decision"], "blocker_labels": blocker_labels, "output_root": _rel(output_root)}, sort_keys=True))
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v96 Phase9 error decomposition.")
    parser.add_argument("--phase6-root", default=str(DEFAULT_PHASE6))
    parser.add_argument("--source-rows", default=str(DEFAULT_SOURCE_ROWS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--variant", default="")
    parser.add_argument("--min-pred-pixels", type=int, default=64)
    parser.add_argument("--min-gt-pixels", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
