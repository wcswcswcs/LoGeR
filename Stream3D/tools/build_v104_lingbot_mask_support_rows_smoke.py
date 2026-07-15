#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
PHASE_ID = "v104_lingbot_map_only_phase3_mask_support_rows_smoke"
DEFAULT_OUT = AUDIT_ROOT / PHASE_ID
DEFAULT_BSS_ROOT = (
    REPO_ROOT
    / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
    / "stage1_lingbot_baseline/workspace/kitti_v105_00_01_02_05/00/lingbot_map_stream_default"
)


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
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_frames(value: str) -> list[int]:
    frames: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if part:
            frames.append(int(part))
    return frames


def _finite_points_and_xy(points: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    n = min(points.shape[0], xy.shape[0])
    points = points[:n]
    xy = xy[:n]
    finite = np.isfinite(points).all(axis=1) & np.isfinite(xy).all(axis=1)
    return points[finite], xy[finite]


def _build_two_mask_image(xy: np.ndarray, image_shape: tuple[int, int], first_mask_id: int, second_mask_id: int) -> np.ndarray:
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.int32)
    if xy.shape[0] == 0:
        return mask
    split = max(xy.shape[0] // 2, 1)
    mx = np.rint(xy[:, 0]).astype(np.int64).clip(0, max(w - 1, 0))
    my = np.rint(xy[:, 1]).astype(np.int64).clip(0, max(h - 1, 0))
    mask[my[:split], mx[:split]] = int(first_mask_id)
    mask[my[split:], mx[split:]] = int(second_mask_id)
    return mask


def build(args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(STREAM3D_ROOT))
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    from geometry_provider.lingbot_map_provider import LingBotMapGeometryProvider

    t0 = time.time()
    out = Path(args.output_root)
    if not out.is_absolute():
        out = STREAM3D_ROOT / out
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"output root already exists and is non-empty; pass --force: {_rel(out)}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")

    frames = _parse_frames(args.frames)
    provider = LingBotMapGeometryProvider(
        geometry_root=args.lingbot_root,
        nn_radius=float(args.nn_radius),
        max_points_per_frame=int(args.max_points_per_frame),
        min_confidence=args.min_confidence,
    )

    support_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for frame_id in frames:
        samples = provider.load_frame_samples(frame_id)
        if samples.xy is None or samples.image_shape is None:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v104_lingbot_mask_support_failure_row_v1",
                    "phase_id": PHASE_ID,
                    "frame_id": frame_id,
                    "failure_id": "NO_PIXEL_SAMPLES",
                    "source": samples.source,
                    "evidence": "LingBot frame samples did not expose xy/image_shape",
                    "uses_d4rt_for_prediction": False,
                    "uses_da3_for_prediction": False,
                    "uses_gt_for_prediction": False,
                }
            )
            continue

        points, xy = _finite_points_and_xy(samples.points, samples.xy)
        if points.shape[0] < 2:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v104_lingbot_mask_support_failure_row_v1",
                    "phase_id": PHASE_ID,
                    "frame_id": frame_id,
                    "failure_id": "TOO_FEW_FINITE_SAMPLES",
                    "source": samples.source,
                    "num_finite_samples": int(points.shape[0]),
                    "uses_d4rt_for_prediction": False,
                    "uses_da3_for_prediction": False,
                    "uses_gt_for_prediction": False,
                }
            )
            continue

        first_mask_id = frame_id * 10 + 1
        second_mask_id = frame_id * 10 + 2
        mask_image = _build_two_mask_image(xy, samples.image_shape, first_mask_id, second_mask_id)
        projection = provider.project_frame_masks(
            dataset=object(),
            scene_points=points,
            mask_image=mask_image,
            frame_id=frame_id,
            depth_max_pre=0.0,
        )
        projection_rows.append(
            {
                "schema_version": "stream4d_v104_lingbot_mask_support_projection_row_v1",
                "phase_id": PHASE_ID,
                "frame_id": frame_id,
                "source_frame_id": samples.source_frame_id if samples.source_frame_id is not None else frame_id,
                "lingbot_root": _rel(Path(args.lingbot_root)),
                "source": samples.source,
                "num_scene_points": int(points.shape[0]),
                "image_shape": list(samples.image_shape),
                "frame_point_ids": ";".join(str(v) for v in projection.frame_point_ids),
                "mask_ids": ";".join(str(v) for v in sorted(projection.mask_info)),
                **projection.diagnostics,
                "uses_d4rt_for_prediction": False,
                "uses_da3_for_prediction": False,
                "uses_gt_for_prediction": False,
            }
        )
        for mask_id, point_ids in sorted(projection.mask_info.items()):
            support_rows.append(
                {
                    "schema_version": "stream4d_v104_lingbot_mask_support_row_v1",
                    "phase_id": PHASE_ID,
                    "frame_id": frame_id,
                    "source_frame_id": samples.source_frame_id if samples.source_frame_id is not None else frame_id,
                    "mask_id": int(mask_id),
                    "support_scope": "synthetic_two_mask_on_actual_lingbot_bss_samples",
                    "support_point_count": int(len(point_ids)),
                    "support_point_ids": ";".join(str(v) for v in sorted(point_ids)),
                    "lingbot_root": _rel(Path(args.lingbot_root)),
                    "geometry_source": samples.source,
                    "has_pixel_samples": True,
                    "image_shape": list(samples.image_shape),
                    "uses_d4rt_for_prediction": False,
                    "uses_da3_for_prediction": False,
                    "uses_gt_for_prediction": False,
                    "stream4d_metric_ready": False,
                }
            )

    masks_per_frame: dict[int, int] = {}
    for row in support_rows:
        frame_id = int(row["frame_id"])
        masks_per_frame[frame_id] = masks_per_frame.get(frame_id, 0) + 1
    support_rows_pass = (
        bool(frames)
        and not failure_rows
        and len(support_rows) >= 2 * len(frames)
        and all(int(row["support_point_count"]) > 0 for row in support_rows)
        and all(masks_per_frame.get(frame_id, 0) >= 2 for frame_id in frames)
    )
    gate_rows = [
        {
            "schema_version": "stream4d_v104_lingbot_mask_support_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_name": "actual_bss_pixel_samples_available",
            "pass": not any(row.get("failure_id") == "NO_PIXEL_SAMPLES" for row in failure_rows),
            "observed_failure_count": sum(row.get("failure_id") == "NO_PIXEL_SAMPLES" for row in failure_rows),
            "required": "xy/image_shape for every requested LingBot BSS frame",
            "uses_d4rt_for_prediction": False,
            "uses_da3_for_prediction": False,
            "uses_gt_for_prediction": False,
        },
        {
            "schema_version": "stream4d_v104_lingbot_mask_support_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_name": "two_positive_mask_support_rows_per_frame",
            "pass": support_rows_pass,
            "observed_masks_per_frame": masks_per_frame,
            "required": "at least two non-empty support rows per requested frame",
            "uses_d4rt_for_prediction": False,
            "uses_da3_for_prediction": False,
            "uses_gt_for_prediction": False,
        },
    ]
    summary = {
        "schema_version": "stream4d_v104_lingbot_mask_support_rows_summary_v1",
        "phase_id": PHASE_ID,
        "support_rows_pass": support_rows_pass,
        "taxonomy": "LINGBOT_BSS_SYNTHETIC_MASK_SUPPORT_ROWS_PASS" if support_rows_pass else "LINGBOT_BSS_SYNTHETIC_MASK_SUPPORT_ROWS_FAIL",
        "lingbot_root": _rel(Path(args.lingbot_root)),
        "frame_ids": frames,
        "num_support_rows": len(support_rows),
        "num_projection_rows": len(projection_rows),
        "num_failure_rows": len(failure_rows),
        "masks_per_frame": masks_per_frame,
        "stream4d_metric_ready": False,
        "stream4d_metric_note": "Mask support rows are generated from actual LingBot BSS samples with synthetic masks; no real Stream4D AP/MV_AP metric is produced.",
        "outputs": {
            "mask_support_rows": _rel(out / "mask_support_rows.csv"),
            "projection_diagnostics_rows": _rel(out / "projection_diagnostics_rows.csv"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "summary": _rel(out / "summary.json"),
        },
        "runtime_sec": round(time.time() - t0, 3),
    }
    _write_csv(out / "mask_support_rows.csv", support_rows)
    _write_csv(out / "projection_diagnostics_rows.csv", projection_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LingBot BSS synthetic mask support rows for the Stream4D v104 audit.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--lingbot-root", default=str(DEFAULT_BSS_ROOT))
    parser.add_argument("--frames", default="0,1")
    parser.add_argument("--max-points-per-frame", type=int, default=32)
    parser.add_argument("--nn-radius", type=float, default=0.05)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
