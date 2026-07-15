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
PHASE_ID = "v104_lingbot_map_only_phase6_same_scene_bss_smoke"
PHASE5_ROOT = AUDIT_ROOT / "v104_lingbot_map_only_phase5_general_config_readiness"
DEFAULT_SCENE = "scene0011_00"
DEFAULT_DATASET = f"stream4d_{DEFAULT_SCENE}_general_smoke8"
DEFAULT_METHOD = "lingbot_map_stream4d_general"
DEFAULT_LINGBOT_ROOT = PHASE5_ROOT / "lingbot_workspace_smoke8" / DEFAULT_DATASET / DEFAULT_SCENE / DEFAULT_METHOD
DEFAULT_OUT = AUDIT_ROOT / PHASE_ID


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return p.as_posix()


def _project(path: Path | str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_frames(value: str, default_count: int) -> list[int]:
    if not value.strip():
        return list(range(default_count))
    frames: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if part:
            frames.append(int(part))
    return frames


def _sampling_frames(sampling_json: Path) -> list[int]:
    payload = _read_json(sampling_json)
    frames = payload.get("frames", [])
    out: list[int] = []
    for value in frames:
        try:
            out.append(int(value))
        except Exception:
            continue
    return out


def _count_files(root: Path, subdir: str, suffix: str) -> int:
    path = root / subdir
    if not path.exists():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() == suffix)


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip() and not line.lstrip().startswith("#"))


def build(args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(STREAM3D_ROOT))
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    from geometry_provider.lingbot_map_provider import LingBotMapGeometryProvider

    t0 = time.time()
    out = _project(args.output_root)
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"output root already exists and is non-empty; pass --force: {_rel(out)}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")

    lingbot_root = _project(args.lingbot_root)
    gt_root = lingbot_root.parent / "gt"
    gt_complete = _read_json(gt_root / ".complete.json")
    pred_complete = _read_json(lingbot_root / ".complete.json")
    sampling_json = gt_root / "sampling.json"
    source_frames = _sampling_frames(sampling_json)
    default_count = int(pred_complete.get("metadata", {}).get("num_frames", len(source_frames) or 0))
    frames = _parse_frames(args.frames, default_count)

    provider = LingBotMapGeometryProvider(
        geometry_root=lingbot_root,
        max_points_per_frame=int(args.max_points_per_frame),
        min_confidence=args.min_confidence,
    )
    frame_rows: list[dict[str, Any]] = []
    for frame_id in frames:
        samples = provider.load_frame_samples(frame_id)
        points = np.asarray(samples.points, dtype=np.float32).reshape(-1, 3)
        xy = samples.xy
        finite = np.isfinite(points).all(axis=1)
        finite_points = points[finite]
        expected_source_frame_id = source_frames[frame_id] if 0 <= frame_id < len(source_frames) else frame_id
        frame_rows.append(
            {
                "schema_version": "stream4d_v104_lingbot_same_scene_bss_frame_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": args.scene_id,
                "bss_frame_id": int(frame_id),
                "source_frame_id": samples.source_frame_id if samples.source_frame_id is not None else frame_id,
                "expected_source_frame_id": expected_source_frame_id,
                "source_frame_id_matches_sampling": (samples.source_frame_id if samples.source_frame_id is not None else frame_id) == expected_source_frame_id,
                "source": samples.source,
                "num_points": int(finite_points.shape[0]),
                "has_pixel_samples": bool(xy is not None and xy.shape[0] > 0),
                "num_pixel_samples": int(xy.shape[0]) if xy is not None else 0,
                "image_shape": list(samples.image_shape) if samples.image_shape is not None else [],
                "bbox_min": finite_points.min(axis=0).tolist() if finite_points.shape[0] else [],
                "bbox_max": finite_points.max(axis=0).tolist() if finite_points.shape[0] else [],
                "uses_d4rt_for_prediction": False,
                "uses_da3_for_prediction": False,
                "uses_gt_for_prediction": False,
            }
        )

    source_diffs = [b - a for a, b in zip(source_frames[:-1], source_frames[1:])]
    expected_stride = int(args.expected_stride)
    stride_aligned = bool(source_frames) and all(diff == expected_stride for diff in source_diffs)
    provider_read_pass = bool(frame_rows) and all(int(row["num_points"]) > 0 and bool(row["has_pixel_samples"]) for row in frame_rows)
    source_mapping_pass = bool(frame_rows) and all(bool(row["source_frame_id_matches_sampling"]) for row in frame_rows)
    depth_exr_count = _count_files(lingbot_root, "depth", ".exr")
    confidence_exr_count = _count_files(lingbot_root, "confidence", ".exr")
    required_count = len(frames)
    file_schema_pass = (
        bool(gt_complete)
        and bool(pred_complete)
        and sampling_json.exists()
        and (lingbot_root / "traj.txt").exists()
        and (lingbot_root / "intrinsics.txt").exists()
        and depth_exr_count >= required_count
        and confidence_exr_count >= required_count
    )
    smoke_pass = file_schema_pass and stride_aligned and provider_read_pass and source_mapping_pass
    gate_rows = [
        {
            "schema_version": "stream4d_v104_lingbot_same_scene_bss_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_name": "bss_file_schema_complete",
            "pass": file_schema_pass,
            "observed": {
                "gt_complete": bool(gt_complete),
                "pred_complete": bool(pred_complete),
                "depth_exr_count": depth_exr_count,
                "confidence_exr_count": confidence_exr_count,
                "traj_noncomment_rows": _line_count(lingbot_root / "traj.txt"),
                "intrinsics_noncomment_rows": _line_count(lingbot_root / "intrinsics.txt"),
            },
            "required": "complete metadata plus depth/confidence/traj/intrinsics for requested frames",
            "uses_d4rt_for_prediction": False,
            "uses_da3_for_prediction": False,
            "uses_gt_for_prediction": False,
        },
        {
            "schema_version": "stream4d_v104_lingbot_same_scene_bss_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_name": "sampling_stride5_aligned",
            "pass": stride_aligned,
            "observed": source_frames,
            "required": f"source frame ids increase by {expected_stride}",
            "uses_d4rt_for_prediction": False,
            "uses_da3_for_prediction": False,
            "uses_gt_for_prediction": False,
        },
        {
            "schema_version": "stream4d_v104_lingbot_same_scene_bss_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_name": "provider_reads_depth_pose_intrinsics",
            "pass": provider_read_pass,
            "observed": {row["bss_frame_id"]: row["num_points"] for row in frame_rows},
            "required": "provider returns finite points and pixel samples for every requested BSS frame",
            "uses_d4rt_for_prediction": False,
            "uses_da3_for_prediction": False,
            "uses_gt_for_prediction": False,
        },
        {
            "schema_version": "stream4d_v104_lingbot_same_scene_bss_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_name": "bss_to_source_frame_mapping_preserved",
            "pass": source_mapping_pass,
            "observed": {row["bss_frame_id"]: row["source_frame_id"] for row in frame_rows},
            "required": "provider source_frame_id matches gt/sampling.json frames",
            "uses_d4rt_for_prediction": False,
            "uses_da3_for_prediction": False,
            "uses_gt_for_prediction": False,
        },
    ]
    summary = {
        "schema_version": "stream4d_v104_lingbot_same_scene_bss_smoke_summary_v1",
        "phase_id": PHASE_ID,
        "scene_id": args.scene_id,
        "lingbot_root": _rel(lingbot_root),
        "gt_root": _rel(gt_root),
        "sampling_json": _rel(sampling_json),
        "smoke_pass": smoke_pass,
        "taxonomy": "LINGBOT_SAME_SCENE_STRIDE5_BSS_SMOKE_PASS" if smoke_pass else "LINGBOT_SAME_SCENE_STRIDE5_BSS_SMOKE_FAIL",
        "blocker": "" if smoke_pass else "SAME_SCENE_LINGBOT_BSS_SMOKE_FAILED",
        "num_frames_checked": len(frame_rows),
        "source_frames": source_frames,
        "expected_stride": expected_stride,
        "stride_aligned": stride_aligned,
        "provider_read_pass": provider_read_pass,
        "source_mapping_pass": source_mapping_pass,
        "file_schema_pass": file_schema_pass,
        "stream4d_metric_ready": False,
        "stream4d_metric_note": "Same-scene LingBot BSS smoke only; no real mask support rows or AP/MV_AP metric is produced.",
        "outputs": {
            "frame_rows": _rel(out / "frame_rows.csv"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "summary": _rel(out / "summary.json"),
        },
        "runtime_sec": round(time.time() - t0, 3),
    }
    _write_csv(out / "frame_rows.csv", frame_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a same-scene stride-5 LingBot BSS smoke output.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--lingbot-root", default=str(DEFAULT_LINGBOT_ROOT))
    parser.add_argument("--scene-id", default=DEFAULT_SCENE)
    parser.add_argument("--frames", default="")
    parser.add_argument("--expected-stride", type=int, default=5)
    parser.add_argument("--max-points-per-frame", type=int, default=64)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
