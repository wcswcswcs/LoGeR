#!/usr/bin/env python3
"""Diagnose foreground/coverage parity for v106 Phase9 H4 chain artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Stream3D.stream4d_v106.artifacts import read_json, sha256_file, write_json  # noqa: E402


def _resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _rel(path: str | Path) -> str:
    path = _resolve(path)
    try:
        return str(path.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _load_label(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.int64, copy=False)


def _label_map(summary: dict[str, Any]) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for row in summary.get("records", []):
        if "frame_id" not in row or "label_path" not in row:
            continue
        out[int(row["frame_id"])] = _resolve(str(row["label_path"]))
    return out


def _visible_ids(label: np.ndarray, *, min_object_area: int = 0) -> list[int]:
    ids: list[int] = []
    for value in np.unique(label).tolist():
        value = int(value)
        if value <= 0:
            continue
        if min_object_area > 0 and int(np.count_nonzero(label == value)) < min_object_area:
            continue
        ids.append(value)
    return ids


def _foreground_mask(
    label: np.ndarray,
    *,
    min_object_area: int,
    min_component_area: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    fg = np.zeros(label.shape, dtype=bool)
    raw_ids = [int(v) for v in np.unique(label).tolist() if int(v) > 0]
    kept_ids: list[int] = []
    dropped_small_object_ids: list[int] = []
    dropped_small_object_pixels = 0
    for value in raw_ids:
        mask = label == value
        area = int(np.count_nonzero(mask))
        if min_object_area > 0 and area < min_object_area:
            dropped_small_object_ids.append(value)
            dropped_small_object_pixels += area
            continue
        kept_ids.append(value)
        fg |= mask

    dropped_small_component_pixels = 0
    dropped_small_component_count = 0
    if min_component_area > 0 and np.any(fg):
        component_count, components, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
        filtered = np.zeros_like(fg)
        for idx in range(1, component_count):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < min_component_area:
                dropped_small_component_pixels += area
                dropped_small_component_count += 1
                continue
            filtered |= components == idx
        fg = filtered

    meta = {
        "raw_visible_id_count": len(raw_ids),
        "filtered_visible_id_count": len(kept_ids),
        "dropped_small_object_id_count": len(dropped_small_object_ids),
        "dropped_small_object_pixels": int(dropped_small_object_pixels),
        "dropped_small_component_count": int(dropped_small_component_count),
        "dropped_small_component_pixels": int(dropped_small_component_pixels),
    }
    return fg, meta


def _write_diff_image(ref_fg: np.ndarray, pred_fg: np.ndarray, path: Path) -> None:
    image = np.zeros((*ref_fg.shape, 3), dtype=np.uint8)
    image[ref_fg & pred_fg] = (220, 220, 220)
    image[ref_fg & ~pred_fg] = (0, 0, 255)
    image[pred_fg & ~ref_fg] = (0, 255, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"failed to write diff image: {path}")


def _frame_record(
    *,
    scene_id: str,
    boundary: str,
    replay_summary_path: Path,
    reference_summary_path: Path,
    frame_id: int,
    pred_path: Path,
    ref_path: Path,
    diff_path: Path | None,
    min_object_area: int,
    min_component_area: int,
) -> dict[str, Any]:
    pred = _load_label(pred_path)
    ref = _load_label(ref_path)
    if pred.shape != ref.shape:
        return {
            "scene_id": scene_id,
            "boundary": boundary,
            "frame_id": int(frame_id),
            "shape_mismatch": True,
            "pred_shape": list(pred.shape),
            "ref_shape": list(ref.shape),
            "pred_label_path": _rel(pred_path),
            "reference_label_path": _rel(ref_path),
        }

    pred_fg, pred_filter_meta = _foreground_mask(
        pred,
        min_object_area=min_object_area,
        min_component_area=min_component_area,
    )
    ref_fg, ref_filter_meta = _foreground_mask(
        ref,
        min_object_area=min_object_area,
        min_component_area=min_component_area,
    )
    intersection = pred_fg & ref_fg
    union = pred_fg | ref_fg
    ref_count = int(np.count_nonzero(ref_fg))
    pred_count = int(np.count_nonzero(pred_fg))
    intersection_count = int(np.count_nonzero(intersection))
    union_count = int(np.count_nonzero(union))
    missing_count = int(np.count_nonzero(ref_fg & ~pred_fg))
    extra_count = int(np.count_nonzero(pred_fg & ~ref_fg))
    fg_iou = 1.0 if union_count == 0 else float(intersection_count / union_count)
    coverage_recall = 1.0 if ref_count == 0 else float(intersection_count / ref_count)
    pred_precision = 1.0 if pred_count == 0 else float(intersection_count / pred_count)
    if diff_path is not None:
        _write_diff_image(ref_fg, pred_fg, diff_path)

    return {
        "scene_id": scene_id,
        "boundary": boundary,
        "frame_id": int(frame_id),
        "shape_mismatch": False,
        "foreground_union_iou": fg_iou,
        "coverage_recall_vs_reference": coverage_recall,
        "pred_foreground_precision_vs_reference": pred_precision,
        "reference_foreground_pixels": ref_count,
        "pred_foreground_pixels": pred_count,
        "intersection_foreground_pixels": intersection_count,
        "union_foreground_pixels": union_count,
        "missing_reference_foreground_pixels": missing_count,
        "extra_pred_foreground_pixels": extra_count,
        "missing_reference_foreground_fraction": 0.0 if ref_count == 0 else float(missing_count / ref_count),
        "extra_pred_foreground_fraction_vs_ref": 0.0 if ref_count == 0 else float(extra_count / ref_count),
        "pred_to_reference_foreground_ratio": None if ref_count == 0 else float(pred_count / ref_count),
        "reference_visible_id_count": len(_visible_ids(ref)),
        "pred_visible_id_count": len(_visible_ids(pred)),
        "filtered_reference_visible_id_count": len(_visible_ids(ref, min_object_area=min_object_area)),
        "filtered_pred_visible_id_count": len(_visible_ids(pred, min_object_area=min_object_area)),
        "coverage_filter": {
            "min_object_area": int(min_object_area),
            "min_component_area": int(min_component_area),
            "pred": pred_filter_meta,
            "reference": ref_filter_meta,
        },
        "pred_label_path": _rel(pred_path),
        "reference_label_path": _rel(ref_path),
        "replay_summary": _rel(replay_summary_path),
        "reference_summary": _rel(reference_summary_path),
        "diff_image_path": _rel(diff_path) if diff_path is not None else "",
        "pred_label_sha256": sha256_file(pred_path),
        "reference_label_sha256": sha256_file(ref_path),
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in records if not row.get("shape_mismatch")]
    if not usable:
        return {"frame_count": len(records), "compared_frame_count": 0, "has_metrics": False}

    def values(key: str) -> list[float]:
        return [float(row[key]) for row in usable if row.get(key) is not None]

    fg_iou = values("foreground_union_iou")
    recall = values("coverage_recall_vs_reference")
    precision = values("pred_foreground_precision_vs_reference")
    missing = values("missing_reference_foreground_fraction")
    ratio = values("pred_to_reference_foreground_ratio")
    ref_pixels = sum(int(row["reference_foreground_pixels"]) for row in usable)
    pred_pixels = sum(int(row["pred_foreground_pixels"]) for row in usable)
    intersection_pixels = sum(int(row["intersection_foreground_pixels"]) for row in usable)
    return {
        "frame_count": len(records),
        "compared_frame_count": len(usable),
        "has_metrics": True,
        "min_foreground_union_iou": min(fg_iou) if fg_iou else None,
        "mean_foreground_union_iou": float(np.mean(fg_iou)) if fg_iou else None,
        "min_coverage_recall_vs_reference": min(recall) if recall else None,
        "mean_coverage_recall_vs_reference": float(np.mean(recall)) if recall else None,
        "min_pred_foreground_precision_vs_reference": min(precision) if precision else None,
        "mean_pred_foreground_precision_vs_reference": float(np.mean(precision)) if precision else None,
        "max_missing_reference_foreground_fraction": max(missing) if missing else None,
        "mean_missing_reference_foreground_fraction": float(np.mean(missing)) if missing else None,
        "min_pred_to_reference_foreground_ratio": min(ratio) if ratio else None,
        "mean_pred_to_reference_foreground_ratio": float(np.mean(ratio)) if ratio else None,
        "reference_foreground_pixels_total": int(ref_pixels),
        "pred_foreground_pixels_total": int(pred_pixels),
        "intersection_foreground_pixels_total": int(intersection_pixels),
        "global_coverage_recall_vs_reference": 1.0 if ref_pixels == 0 else float(intersection_pixels / ref_pixels),
        "global_pred_foreground_precision_vs_reference": 1.0 if pred_pixels == 0 else float(intersection_pixels / pred_pixels),
    }


def diagnose_chain(
    chain_path: Path,
    output_root: Path,
    worst_k: int,
    *,
    min_object_area: int,
    min_component_area: int,
) -> dict[str, Any]:
    chain = read_json(chain_path)
    scene_id = str(chain.get("scene_id") or "")
    frame_records: list[dict[str, Any]] = []
    boundary_records: list[dict[str, Any]] = []

    for boundary in chain.get("boundaries", []):
        if not isinstance(boundary, dict):
            continue
        boundary_name = str(boundary.get("boundary") or "")
        replay_summary_path = _resolve(str(boundary.get("replay_summary") or ""))
        replay_summary = read_json(replay_summary_path)
        reference_summary_path = _resolve(str(replay_summary.get("reference_summary_path") or ""))
        reference_summary = read_json(reference_summary_path)
        pred_labels = _label_map(replay_summary)
        ref_labels = _label_map(reference_summary)
        common_frame_ids = sorted(set(pred_labels) & set(ref_labels))
        boundary_frame_records = []
        for frame_id in common_frame_ids:
            diff_path = output_root / "diff_images" / boundary_name / f"frame_{int(frame_id):06d}_diff.png"
            record = _frame_record(
                scene_id=scene_id,
                boundary=boundary_name,
                replay_summary_path=replay_summary_path,
                reference_summary_path=reference_summary_path,
                frame_id=int(frame_id),
                pred_path=pred_labels[frame_id],
                ref_path=ref_labels[frame_id],
                diff_path=diff_path,
                min_object_area=min_object_area,
                min_component_area=min_component_area,
            )
            boundary_frame_records.append(record)
            frame_records.append(record)
        aggregate = _aggregate(boundary_frame_records)
        boundary_records.append(
            {
                "scene_id": scene_id,
                "boundary": boundary_name,
                "replay_summary": _rel(replay_summary_path),
                "replay_summary_sha256": sha256_file(replay_summary_path),
                "reference_summary": _rel(reference_summary_path),
                "reference_summary_sha256": sha256_file(reference_summary_path),
                "input_chain_boundary_passes": bool(boundary.get("passes", False)),
                "input_chain_min_CCOC": boundary.get("min_CCOC"),
                "input_chain_min_HIR": boundary.get("min_HIR"),
                "input_chain_min_HCR": boundary.get("min_HCR"),
                "input_chain_min_OPC": boundary.get("min_OPC"),
                "input_chain_max_CFR": boundary.get("max_CFR"),
                "input_chain_max_CMR": boundary.get("max_CMR"),
                "input_chain_max_BFMR": boundary.get("max_BFMR"),
                "runtime_ratio_vs_reference": boundary.get("runtime_ratio_vs_reference"),
                "replay_peak_cuda_memory_mb": boundary.get("replay_peak_cuda_memory_mb"),
                "aggregate": aggregate,
                "missing_pred_frame_ids": sorted(set(ref_labels) - set(pred_labels)),
                "extra_pred_frame_ids": sorted(set(pred_labels) - set(ref_labels)),
            }
        )

    worst_cases = sorted(
        [row for row in frame_records if not row.get("shape_mismatch")],
        key=lambda row: (
            float(row.get("foreground_union_iou", 1.0)),
            float(row.get("coverage_recall_vs_reference", 1.0)),
            -int(row.get("missing_reference_foreground_pixels", 0)),
        ),
    )[: int(worst_k)]
    summary = {
        "schema_version": "stream4d_v106_phase9_chain_coverage_diagnostic_v2",
        "scene_id": scene_id,
        "coverage_filter": {
            "min_object_area": int(min_object_area),
            "min_component_area": int(min_component_area),
            "note": "metrics are computed after dropping small per-object masks and small connected foreground components; use zero values for strict v1-equivalent foreground.",
        },
        "chain_verification": _rel(chain_path),
        "chain_verification_sha256": sha256_file(chain_path),
        "boundary_count": len(boundary_records),
        "frame_record_count": len(frame_records),
        "all_input_boundaries_pass": all(bool(row["input_chain_boundary_passes"]) for row in boundary_records),
        "boundary_records_json": _rel(output_root / "boundary_coverage_records.json"),
        "frame_records_json": _rel(output_root / "frame_coverage_records.json"),
        "foreground_loss_casebook_json": _rel(output_root / "foreground_loss_casebook.json"),
        "boundaries": boundary_records,
        "worst_case_count": len(worst_cases),
        "worst_cases": worst_cases,
        "passes_foreground_union_iou_098": bool(boundary_records)
        and all(
            float((row.get("aggregate") or {}).get("min_foreground_union_iou") or 0.0) >= 0.98
            for row in boundary_records
        ),
        "passes_coverage_recall_099": bool(boundary_records)
        and all(
            float((row.get("aggregate") or {}).get("min_coverage_recall_vs_reference") or 0.0) >= 0.99
            for row in boundary_records
        ),
    }
    write_json(output_root / "boundary_coverage_records.json", boundary_records)
    write_json(output_root / "frame_coverage_records.json", frame_records)
    write_json(output_root / "foreground_loss_casebook.json", worst_cases)
    write_json(output_root / "coverage_diagnostic_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain", action="append", required=True, help="Path to three_chunk_chain_verification.json")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--worst-k", type=int, default=8)
    parser.add_argument("--min-object-area", type=int, default=0)
    parser.add_argument("--min-component-area", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    scene_summaries = []
    for raw_chain in args.chain:
        chain_path = _resolve(raw_chain)
        scene_output_root = output_root / str(read_json(chain_path).get("scene_id") or chain_path.parent.name)
        scene_summaries.append(
            diagnose_chain(
                chain_path,
                scene_output_root,
                args.worst_k,
                min_object_area=max(0, int(args.min_object_area)),
                min_component_area=max(0, int(args.min_component_area)),
            )
        )
    aggregate = {
        "schema_version": "stream4d_v106_phase9_chain_coverage_diagnostic_aggregate_v2",
        "coverage_filter": {
            "min_object_area": max(0, int(args.min_object_area)),
            "min_component_area": max(0, int(args.min_component_area)),
        },
        "scene_count": len(scene_summaries),
        "scenes": [row.get("scene_id") for row in scene_summaries],
        "all_input_boundaries_pass": all(bool(row.get("all_input_boundaries_pass")) for row in scene_summaries),
        "all_pass_foreground_union_iou_098": all(bool(row.get("passes_foreground_union_iou_098")) for row in scene_summaries),
        "all_pass_coverage_recall_099": all(bool(row.get("passes_coverage_recall_099")) for row in scene_summaries),
        "scene_summaries": [
            {
                "scene_id": row.get("scene_id"),
                "coverage_diagnostic_summary": _rel(output_root / str(row.get("scene_id")) / "coverage_diagnostic_summary.json"),
                "passes_foreground_union_iou_098": row.get("passes_foreground_union_iou_098"),
                "passes_coverage_recall_099": row.get("passes_coverage_recall_099"),
                "boundary_count": row.get("boundary_count"),
                "frame_record_count": row.get("frame_record_count"),
                "worst_cases": row.get("worst_cases", [])[:3],
            }
            for row in scene_summaries
        ],
    }
    write_json(output_root / "coverage_diagnostic_aggregate.json", aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0 if aggregate["all_pass_foreground_union_iou_098"] and aggregate["all_pass_coverage_recall_099"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
