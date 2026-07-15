#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
DEFAULT_OUT = AUDIT_ROOT / "v104_lingbot_map_only_phase1_provider_smoke"
PHASE_ID = "v104_lingbot_map_only_phase1_provider_smoke"
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

    with tempfile.TemporaryDirectory(prefix="stream4d_v104_lingbot_smoke_") as d:
        root = Path(d)
        (root / "points").mkdir(parents=True)
        np.save(
            root / "points" / "000000.npy",
            np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [9.0, 9.0, 9.0],
                ],
                dtype=np.float32,
            ),
        )
        provider = LingBotMapGeometryProvider(geometry_root=root, nn_radius=0.2)
        scene_points = np.asarray(
            [
                [0.02, 0.0, 0.0],
                [1.01, 0.0, 0.0],
                [5.0, 5.0, 5.0],
            ],
            dtype=np.float32,
        )
        mask_image = np.ones((4, 4), dtype=np.int32)
        result = provider.project_frame_masks(
            dataset=object(),
            scene_points=scene_points,
            mask_image=mask_image,
            frame_id=0,
            depth_max_pre=0.0,
        )

    expected_ids = [0, 1]
    expected_mask = {1: {0, 1}}
    smoke_pass = result.frame_point_ids == expected_ids and result.mask_info == expected_mask
    projection_rows = [
        {
            "schema_version": "stream4d_v104_lingbot_provider_smoke_projection_row_v1",
            "phase_id": PHASE_ID,
            "smoke_role": "synthetic_npy_projection",
            "frame_id": 0,
            "frame_point_ids": ";".join(str(v) for v in result.frame_point_ids),
            "mask_ids": ";".join(str(v) for v in sorted(result.mask_info)),
            "mask_1_points": ";".join(str(v) for v in sorted(result.mask_info.get(1, set()))),
            **result.diagnostics,
        }
    ]
    actual_bss_pass = False
    actual_bss_error = ""
    actual_bss_points_shape = ""
    actual_bss_source = ""
    actual_multimask_pass = False
    actual_multimask_observed: dict[int, list[int]] = {}
    actual_root = Path(args.lingbot_root)
    try:
        if actual_root.exists():
            actual_provider = LingBotMapGeometryProvider(
                geometry_root=actual_root,
                max_points_per_frame=int(args.actual_max_points),
            )
            actual_samples = actual_provider.load_frame_samples(int(args.actual_frame_id))
            actual_points = actual_samples.points
            actual_bss_source = actual_samples.source
            actual_bss_points_shape = "x".join(str(v) for v in actual_points.shape)
            actual_bss_pass = bool(actual_points.size and np.isfinite(actual_points).all())
            projection_rows.append(
                {
                    "schema_version": "stream4d_v104_lingbot_provider_smoke_projection_row_v1",
                    "phase_id": PHASE_ID,
                    "smoke_role": "actual_v105_bss_depth_pose_intrinsics_read",
                    "frame_id": int(args.actual_frame_id),
                    "lingbot_root": _rel(actual_root),
                    "source": actual_bss_source,
                    "points_shape": actual_bss_points_shape,
                    "finite_points": actual_bss_pass,
                    "has_pixel_samples": bool(actual_samples.xy is not None and actual_samples.xy.shape[0] > 0),
                    "image_shape": list(actual_samples.image_shape) if actual_samples.image_shape is not None else [],
                    "first_point": actual_points[0].tolist() if actual_points.size else [],
                }
            )
            if actual_bss_pass and actual_samples.xy is not None and actual_samples.image_shape is not None:
                h, w = actual_samples.image_shape
                mask = np.zeros((h, w), dtype=np.int32)
                xy = np.asarray(actual_samples.xy, dtype=np.float32)
                split = max(xy.shape[0] // 2, 1)
                mx = np.rint(xy[:, 0]).astype(np.int64).clip(0, max(w - 1, 0))
                my = np.rint(xy[:, 1]).astype(np.int64).clip(0, max(h - 1, 0))
                mask[my[:split], mx[:split]] = 1
                mask[my[split:], mx[split:]] = 2
                projected = actual_provider.project_frame_masks(
                    dataset=object(),
                    scene_points=np.asarray(actual_points, dtype=np.float32),
                    mask_image=mask,
                    frame_id=int(args.actual_frame_id),
                    depth_max_pre=0.0,
                )
                actual_multimask_observed = {int(k): sorted(int(v) for v in values) for k, values in projected.mask_info.items()}
                actual_multimask_pass = set(actual_multimask_observed) == {1, 2} and all(
                    len(values) > 0 for values in actual_multimask_observed.values()
                )
                projection_rows.append(
                    {
                        "schema_version": "stream4d_v104_lingbot_provider_smoke_projection_row_v1",
                        "phase_id": PHASE_ID,
                        "smoke_role": "actual_v105_bss_synthetic_two_mask_support",
                        "frame_id": int(args.actual_frame_id),
                        "lingbot_root": _rel(actual_root),
                        "source": actual_bss_source,
                        "mask_ids": ";".join(str(v) for v in sorted(actual_multimask_observed)),
                        "mask_1_points": ";".join(str(v) for v in actual_multimask_observed.get(1, [])),
                        "mask_2_points": ";".join(str(v) for v in actual_multimask_observed.get(2, [])),
                        **projected.diagnostics,
                    }
                )
    except Exception as exc:
        actual_bss_error = repr(exc)
        projection_rows.append(
            {
                "schema_version": "stream4d_v104_lingbot_provider_smoke_projection_row_v1",
                "phase_id": PHASE_ID,
                "smoke_role": "actual_v105_bss_depth_pose_intrinsics_read",
                "frame_id": int(args.actual_frame_id),
                "lingbot_root": _rel(actual_root),
                "source": actual_bss_source,
                "points_shape": actual_bss_points_shape,
                "finite_points": False,
                "error": actual_bss_error,
            }
        )
    gate_rows = [
        {
            "schema_version": "stream4d_v104_lingbot_provider_smoke_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_name": "synthetic_npy_points_project_to_scene_points",
            "pass": smoke_pass,
            "observed_frame_point_ids": result.frame_point_ids,
            "required_frame_point_ids": expected_ids,
            "observed_mask_info": {k: sorted(v) for k, v in result.mask_info.items()},
            "required_mask_info": {1: [0, 1]},
            "uses_d4rt_for_prediction": False,
            "uses_da3_for_prediction": False,
            "uses_gt_for_prediction": False,
        },
        {
            "schema_version": "stream4d_v104_lingbot_provider_smoke_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_name": "actual_v105_bss_depth_pose_intrinsics_read",
            "pass": actual_bss_pass,
            "observed": {
                "lingbot_root": _rel(actual_root),
                "source": actual_bss_source,
                "points_shape": actual_bss_points_shape,
                "error": actual_bss_error,
            },
            "required": "finite LingBot frame points from saved depth/traj/intrinsics",
            "uses_d4rt_for_prediction": False,
            "uses_da3_for_prediction": False,
            "uses_gt_for_prediction": False,
        },
        {
            "schema_version": "stream4d_v104_lingbot_provider_smoke_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_name": "actual_v105_bss_synthetic_two_mask_support",
            "pass": actual_multimask_pass,
            "observed_mask_info": actual_multimask_observed,
            "required": "two positive mask ids receive LingBot support points",
            "uses_d4rt_for_prediction": False,
            "uses_da3_for_prediction": False,
            "uses_gt_for_prediction": False,
        },
    ]
    summary = {
        "schema_version": "stream4d_v104_lingbot_provider_smoke_summary_v1",
        "phase_id": PHASE_ID,
        "smoke_pass": smoke_pass and actual_bss_pass and actual_multimask_pass,
        "synthetic_smoke_pass": smoke_pass,
        "actual_bss_depth_pose_intrinsics_pass": actual_bss_pass,
        "actual_bss_synthetic_two_mask_support_pass": actual_multimask_pass,
        "taxonomy": "PROVIDER_SYNTHETIC_ACTUAL_BSS_AND_MULTIMASK_SMOKE_PASS" if smoke_pass and actual_bss_pass and actual_multimask_pass else "PROVIDER_SMOKE_PARTIAL_OR_FAIL",
        "stream4d_metric_ready": False,
        "stream4d_metric_note": "Provider smoke only; no Stream4D AP/MV_AP metric is produced.",
        "actual_lingbot_root": _rel(actual_root),
        "outputs": {
            "projection_rows": _rel(out / "projection_rows.csv"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "summary": _rel(out / "summary.json"),
        },
        "runtime_sec": round(time.time() - t0, 3),
    }
    _write_csv(out / "projection_rows.csv", projection_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic smoke for the LingBot Stream4D provider.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--lingbot-root", default=str(DEFAULT_BSS_ROOT))
    parser.add_argument("--actual-frame-id", type=int, default=0)
    parser.add_argument("--actual-max-points", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
