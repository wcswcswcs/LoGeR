from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from stream4d_native.v47_common import ROOT, utc_now, write_csv, write_json
from stream4d_native.v51_remask_source_discovery import _frame_id


PLAN_PATH = "docs/stream4d_v51_r2_mosaic_remask_lift_codex_plan.md"


def _rel(path: str | Path) -> str:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = ROOT / path_obj
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def _load_npz(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _containment_sets(masks: np.ndarray, contain_threshold: float, area_ratio_threshold: float) -> tuple[set[int], set[int], int]:
    if masks.dtype != bool:
        masks = masks != 0
    n = int(masks.shape[0])
    areas = masks.reshape(n, -1).sum(axis=1).astype(np.float64)
    parents: set[int] = set()
    parts: set[int] = set()
    pair_count = 0
    for i in range(n):
        if areas[i] <= 0:
            continue
        left = masks[i]
        for j in range(i + 1, n):
            if areas[j] <= 0:
                continue
            inter = int(np.count_nonzero(left & masks[j]))
            if inter <= 0:
                continue
            ci = inter / float(areas[i])
            cj = inter / float(areas[j])
            ratio = max(float(areas[i]), float(areas[j])) / max(min(float(areas[i]), float(areas[j])), 1.0)
            if max(ci, cj) >= contain_threshold and ratio >= area_ratio_threshold:
                pair_count += 1
                if ci >= cj:
                    parts.add(i)
                    parents.add(j)
                else:
                    parts.add(j)
                    parents.add(i)
    return parents, parts, pair_count


def filter_mask_bank_by_containment(
    input_root: str | Path,
    output_root: str | Path,
    contain_threshold: float = 0.85,
    area_ratio_threshold: float = 1.30,
    min_masks_per_frame: int = 10,
) -> dict[str, Any]:
    input_root = ROOT / input_root if not Path(input_root).is_absolute() else Path(input_root)
    output_root = ROOT / output_root if not Path(output_root).is_absolute() else Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    files = sorted(input_root.glob("*_masks.npz"), key=lambda p: _frame_id(p) if _frame_id(p) is not None else 10**12)
    if not files:
        files = sorted(input_root.glob("*.npz"), key=lambda p: _frame_id(p) if _frame_id(p) is not None else 10**12)
    rows: list[dict[str, Any]] = []
    total_input = 0
    total_output = 0
    total_pairs = 0
    for path in files:
        payload = _load_npz(path)
        masks = np.asarray(payload["masks"])
        if masks.dtype != bool:
            masks = masks != 0
        parents, parts, pair_count = _containment_sets(masks, contain_threshold, area_ratio_threshold)
        selected = set(parents) | set(parts)
        areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
        if len(selected) < min_masks_per_frame and masks.shape[0] > 0:
            largest = np.argsort(areas)[-min(int(min_masks_per_frame), int(masks.shape[0])) :]
            selected.update(int(idx) for idx in largest.tolist())
        selected_ids = np.asarray(sorted(selected), dtype=np.int64)
        out_payload: dict[str, Any] = {}
        for key, value in payload.items():
            arr = np.asarray(value)
            if key in {"masks", "scores", "stability_scores", "boxes", "areas"} and arr.shape[:1] == (masks.shape[0],):
                out_payload[key] = arr[selected_ids]
            else:
                out_payload[key] = value
        metadata = {
            "source": "containment_filtered_sam2",
            "input_path": _rel(path),
            "contain_threshold": float(contain_threshold),
            "area_ratio_threshold": float(area_ratio_threshold),
            "min_masks_per_frame": int(min_masks_per_frame),
            "selected_by_gt": False,
            "uses_gt_for_prediction": False,
        }
        out_payload["filter_metadata"] = np.asarray(json.dumps(metadata, sort_keys=True), dtype=object)
        out_path = output_root / path.name
        np.savez_compressed(out_path, **out_payload)
        total_input += int(masks.shape[0])
        total_output += int(selected_ids.shape[0])
        total_pairs += int(pair_count)
        rows.append(
            {
                "input_path": _rel(path),
                "output_path": _rel(out_path),
                "frame_id": _frame_id(path),
                "input_mask_count": int(masks.shape[0]),
                "output_mask_count": int(selected_ids.shape[0]),
                "parent_candidate_count": int(len(parents)),
                "part_candidate_count": int(len(parts)),
                "containment_pair_count_before_filter": int(pair_count),
                "selected_ids": selected_ids.tolist(),
                "uses_gt_for_prediction": False,
            }
        )
    summary = {
        "phase": "v51_r2_mask_bank_filter",
        "created_at": utc_now(),
        "plan": PLAN_PATH,
        "input_root": _rel(input_root),
        "output_root": _rel(output_root),
        "frame_count": len(rows),
        "input_mask_count": total_input,
        "output_mask_count": total_output,
        "containment_pair_count_before_filter": total_pairs,
        "mean_output_masks_per_frame": total_output / max(len(rows), 1),
        "contain_threshold": float(contain_threshold),
        "area_ratio_threshold": float(area_ratio_threshold),
        "min_masks_per_frame": int(min_masks_per_frame),
        "uses_gt_for_prediction": False,
        "rows": rows,
    }
    write_json(output_root / "mask_bank_filter_summary.json", summary)
    write_csv(output_root / "mask_bank_filter_rows.csv", rows)
    return summary
