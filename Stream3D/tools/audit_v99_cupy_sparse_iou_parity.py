#!/usr/bin/env python3
"""Audit CPU SparseSceneIoU vs CuPySparseSceneIoU parity for v99 AP speedups."""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools.run_v65_scene_multiview_ap import SparseSceneIoU, _summarize_iou  # noqa: E402
from tools.v99_cupy_sparse_iou import CuPySparseSceneIoU  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_cupy_sparse_iou_parity"


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
            writer.writerow({field: _jsonable(row.get(field, "")) for field in fields})


def _max_abs_summary_diff(a: dict[str, Any], b: dict[str, Any]) -> float:
    max_diff = 0.0
    for key, value in a.items():
        if key not in b:
            continue
        av = value
        bv = b[key]
        if av is None or bv is None:
            if av != bv:
                return float("inf")
            continue
        if isinstance(av, (int, float, np.integer, np.floating)) and isinstance(bv, (int, float, np.integer, np.floating)):
            if math.isfinite(float(av)) and math.isfinite(float(bv)):
                max_diff = max(max_diff, abs(float(av) - float(bv)))
    return max_diff


def _one_case(case_id: str, frames: list[tuple[np.ndarray, np.ndarray]], *, device_id: int) -> dict[str, Any]:
    cpu = SparseSceneIoU()
    gpu = CuPySparseSceneIoU(device_id=device_id)
    t0 = time.perf_counter()
    for pred, gt in frames:
        cpu.add(pred, gt)
    cpu_add_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    for pred, gt in frames:
        gpu.add(pred, gt)
    gpu_add_sec = time.perf_counter() - t0
    cpu_build = cpu.build(min_pred_pixels=1, min_gt_pixels=1)
    gpu_build = gpu.build(min_pred_pixels=1, min_gt_pixels=1)
    cpu_summary, cpu_iou, cpu_pred_ids, cpu_gt_ids = _summarize_iou(
        accumulator=cpu,
        min_pred_pixels=1,
        min_gt_pixels=1,
        score_mode="constant",
        input_scores=None,
    )
    gpu_summary, gpu_iou, gpu_pred_ids, gpu_gt_ids = _summarize_iou(
        accumulator=gpu,
        min_pred_pixels=1,
        min_gt_pixels=1,
        score_mode="constant",
        input_scores=None,
    )
    pred_ids_match = cpu_build["pred_ids"] == gpu_build["pred_ids"] == cpu_pred_ids == gpu_pred_ids
    gt_ids_match = cpu_build["gt_ids"] == gpu_build["gt_ids"] == cpu_gt_ids == gpu_gt_ids
    iou_max_abs_diff = float(np.max(np.abs(cpu_iou - gpu_iou))) if cpu_iou.shape == gpu_iou.shape and cpu_iou.size else 0.0
    summary_max_abs_diff = _max_abs_summary_diff(cpu_summary, gpu_summary)
    pass_gate = bool(pred_ids_match and gt_ids_match and iou_max_abs_diff <= 1e-12 and summary_max_abs_diff <= 1e-12)
    return {
        "schema_version": "stream4d_v99_cupy_sparse_iou_parity_case_v1",
        "case_id": case_id,
        "frame_count": len(frames),
        "shape_h": int(frames[0][0].shape[0]),
        "shape_w": int(frames[0][0].shape[1]),
        "cpu_add_sec": cpu_add_sec,
        "cupy_add_sec": gpu_add_sec,
        "speedup_cpu_add_over_cupy_add": float(cpu_add_sec / gpu_add_sec) if gpu_add_sec > 0 else 0.0,
        "pred_ids_match": pred_ids_match,
        "gt_ids_match": gt_ids_match,
        "iou_shape_match": tuple(cpu_iou.shape) == tuple(gpu_iou.shape),
        "iou_max_abs_diff": iou_max_abs_diff,
        "summary_max_abs_diff": summary_max_abs_diff,
        "cpu_ap": cpu_summary.get("ap"),
        "cupy_ap": gpu_summary.get("ap"),
        "cpu_ap50": cpu_summary.get("ap50"),
        "cupy_ap50": gpu_summary.get("ap50"),
        "pass": pass_gate,
    }


def _synthetic_frames(seed: int, *, frame_count: int, shape_hw: tuple[int, int], pred_classes: int, gt_classes: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    h, w = shape_hw
    frames: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(frame_count):
        pred = rng.integers(0, pred_classes + 1, size=(h, w), dtype=np.int64)
        gt = rng.integers(0, gt_classes + 1, size=(h, w), dtype=np.int64)
        pred[rng.random((h, w)) < 0.55] = 0
        gt[rng.random((h, w)) < 0.50] = 0
        frames.append((pred, gt))
    return frames


def main() -> int:
    started = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import cupy as cp

    device_count = int(cp.cuda.runtime.getDeviceCount())
    device_id = 0
    cases = [
        ("synthetic_256x384_f16", _synthetic_frames(9901, frame_count=16, shape_hw=(256, 384), pred_classes=80, gt_classes=70)),
        ("synthetic_968x1296_f4", _synthetic_frames(9902, frame_count=4, shape_hw=(968, 1296), pred_classes=160, gt_classes=120)),
    ]
    rows = [_one_case(case_id, frames, device_id=device_id) for case_id, frames in cases]
    parity_pass = all(bool(row["pass"]) for row in rows)
    summary = {
        "schema_version": "stream4d_v99_cupy_sparse_iou_parity_summary_v1",
        "phase_id": "v99_cupy_sparse_iou_parity",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "decision": "PASS_CUPY_SPARSE_IOU_PARITY_SMOKE" if parity_pass else "NO_GO_CUPY_SPARSE_IOU_PARITY",
        "parity_pass": parity_pass,
        "cupy_version": cp.__version__,
        "cupy_device_count": device_count,
        "cupy_device_id": device_id,
        "case_count": len(rows),
        "case_rows": rows,
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "case_rows": _rel(OUT_DIR / "case_rows.csv"),
        },
    }
    _write_csv(OUT_DIR / "case_rows.csv", rows)
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if parity_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
