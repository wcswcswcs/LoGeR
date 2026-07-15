#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
PHASE_ID = "v104_lingbot_map_only_phase10c_temporal_track_full_scene"
DEFAULT_OUT = AUDIT_ROOT / PHASE_ID
DEFAULT_SUPPORT_ROOT = AUDIT_ROOT / "v104_lingbot_map_only_phase7_real_mask_support_rows"
DEFAULT_SELECTED_ROWS = AUDIT_ROOT / "v87_phase1_mv_input_generation/frame_mask_selected_rows.csv"
DEFAULT_BASELINE_ROWS = AUDIT_ROOT / "v103_phase0_contract/baseline_metric_rows.csv"

if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from geometry_provider.lingbot_map_provider import LingBotMapGeometryProvider  # noqa: E402
from tools import run_v104_lingbot_temporal_track_local_mv_ap as base  # noqa: E402


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _full_variants(frame_denominator: int) -> list[dict[str, object]]:
    return [
        {
            "variant_id": "F0_full_2dshape_control_s2d008_thr050_gap1",
            "sigma_3d": 1.0,
            "sigma_2d": 0.08,
            "sigma_log_area": 0.85,
            "w3d": 0.0,
            "w2d": 0.70,
            "warea": 0.30,
            "threshold": 0.50,
            "max_gap": 1,
            "min_frames": 2,
            "non_broad_only": False,
            "score_frame_denominator": frame_denominator,
        },
        {
            "variant_id": "F1_full_lingbot_w3d0p2_s3d0p6_s2d0p08_area0p30_thr055_gap1",
            "sigma_3d": 0.60,
            "sigma_2d": 0.08,
            "sigma_log_area": 0.85,
            "w3d": 0.20,
            "w2d": 0.56,
            "warea": 0.24,
            "threshold": 0.55,
            "max_gap": 1,
            "min_frames": 2,
            "non_broad_only": False,
            "score_frame_denominator": frame_denominator,
        },
    ]


def _selected_meta(path: Path) -> dict[str, dict[str, str]]:
    return {row.get("candidate_row_id", ""): row for row in _read_csv(path)}


def _provider(
    cache: dict[str, LingBotMapGeometryProvider],
    root: Path,
    *,
    max_points_per_frame: int,
    min_confidence: float | None,
) -> LingBotMapGeometryProvider:
    key = root.as_posix()
    if key not in cache:
        cache[key] = LingBotMapGeometryProvider(
            geometry_root=root,
            max_points_per_frame=max_points_per_frame,
            min_confidence=min_confidence,
        )
    return cache[key]


def _build_full_feature_inputs(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    support_root = _project(args.support_root)
    support_rows = _read_csv(support_root / "real_mask_support_rows.csv")
    selected_meta = _selected_meta(_project(args.selected_rows))
    feature_root = out / "full_scene_feature_inputs"
    phase2_root = out / "full_scene_phase2_summaries"
    provider_cache: dict[str, LingBotMapGeometryProvider] = {}
    frame_sample_cache: dict[tuple[str, int], Any] = {}

    scene_frames: dict[str, set[int]] = defaultdict(set)
    scene_mask_roots: dict[str, str] = {}
    for row in support_rows:
        scene = row.get("scene_id", "")
        if not scene:
            continue
        source_frame = int(row.get("source_frame_id", "0"))
        scene_frames[scene].add(source_frame)
        mask_path = Path(row.get("mask_path", ""))
        if mask_path.name:
            scene_mask_roots[scene] = mask_path.parent.as_posix()

    frame_index = {scene: {frame: idx for idx, frame in enumerate(sorted(frames))} for scene, frames in scene_frames.items()}
    feature_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for row in support_rows:
        scene = row.get("scene_id", "")
        candidate_row_id = row.get("candidate_row_id", "")
        source_frame = int(row.get("source_frame_id", "0"))
        bss_frame = int(row.get("bss_frame_id", "0"))
        lingbot_root = _project(row.get("lingbot_root", ""))
        support_ids_path = _project(row.get("support_point_ids_path", ""))
        meta = selected_meta.get(candidate_row_id, {})
        try:
            ids = np.asarray(np.load(support_ids_path), dtype=np.int64)
        except Exception as exc:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v104_lingbot_full_scene_feature_failure_row_v1",
                    "phase_id": PHASE_ID,
                    "candidate_row_id": candidate_row_id,
                    "scene_id": scene,
                    "failure_id": "SUPPORT_POINT_IDS_LOAD_FAILED",
                    "error": str(exc),
                }
            )
            continue
        prov = _provider(
            provider_cache,
            lingbot_root,
            max_points_per_frame=int(args.max_points_per_frame),
            min_confidence=args.min_confidence,
        )
        sample_key = (lingbot_root.as_posix(), bss_frame)
        if sample_key not in frame_sample_cache:
            frame_sample_cache[sample_key] = prov.load_frame_samples(bss_frame)
        samples = frame_sample_cache[sample_key]
        points = np.asarray(samples.points, dtype=np.float32).reshape(-1, 3)
        valid_ids = ids[(ids >= 0) & (ids < points.shape[0])]
        if valid_ids.size == 0:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v104_lingbot_full_scene_feature_failure_row_v1",
                    "phase_id": PHASE_ID,
                    "candidate_row_id": candidate_row_id,
                    "scene_id": scene,
                    "source_frame_id": source_frame,
                    "bss_frame_id": bss_frame,
                    "failure_id": "EMPTY_VALID_SUPPORT_POINT_IDS",
                    "support_point_count": int(ids.size),
                    "frame_point_count": int(points.shape[0]),
                }
            )
            continue
        centroid = np.mean(points[valid_ids], axis=0)
        object_like = _as_bool(meta.get("object_mask_ownership_allowed", "True")) and _as_bool(
            meta.get("adapter_candidate_valid", "True")
        )
        feature_rows.append(
            {
                "schema_version": "stream4d_v104_lingbot_full_scene_mask_observation_row_v1",
                "phase_id": PHASE_ID,
                "candidate_row_id": candidate_row_id,
                "scene_id": scene,
                "chunk_id": row.get("chunk_id", ""),
                "source_frame_id": source_frame,
                "frame_local_index": frame_index.get(scene, {}).get(source_frame, -1),
                "bss_frame_id": bss_frame,
                "mask_id": row.get("mask_id", ""),
                "history_id": row.get("history_id", ""),
                "support_point_count": int(valid_ids.size),
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "centroid_z": float(centroid[2]),
                "mask_is_broad": _as_bool(meta.get("broad_mask_flag", "False")),
                "mask_is_object_like": object_like,
                "support_scope": "full_scene_phase7_real_mask_support_rows",
                "uses_d4rt_for_prediction": False,
                "uses_da3_for_prediction": False,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )

    feature_rows.sort(key=lambda r: (str(r["scene_id"]), int(r["frame_local_index"]), int(r["mask_id"]), int(r["candidate_row_id"])))
    _write_csv(feature_root / "mask_observation_rows.csv", feature_rows)
    _write_csv(feature_root / "feature_failure_rows.csv", failure_rows)

    phase2_paths: dict[str, str] = {}
    for scene, frames in sorted(scene_frames.items()):
        scene_phase2_root = phase2_root / scene
        summary = {
            "schema_version": "stream4d_v104_lingbot_full_scene_phase2_summary_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene,
            "frame_ids": sorted(int(v) for v in frames),
            "frame_count": len(frames),
            "mask_root": scene_mask_roots.get(scene, ""),
            "source": _rel(support_root / "real_mask_support_rows.csv"),
            "metric_scope": "full_selected_scene_from_phase7_rows",
            "uses_d4rt_for_prediction": False,
            "uses_da3_for_prediction": False,
            "uses_gt_for_prediction": False,
        }
        _write_json(scene_phase2_root / "summary.json", summary)
        phase2_paths[scene] = _rel(scene_phase2_root)

    summary = {
        "schema_version": "stream4d_v104_lingbot_full_scene_feature_input_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix": time.time(),
        "support_root": _rel(support_root),
        "selected_rows": _rel(_project(args.selected_rows)),
        "feature_root": _rel(feature_root),
        "feature_row_count": len(feature_rows),
        "failure_row_count": len(failure_rows),
        "scene_frame_counts": {scene: len(frames) for scene, frames in sorted(scene_frames.items())},
        "scene_mask_roots": scene_mask_roots,
        "phase2_paths": phase2_paths,
        "uses_d4rt_for_prediction": False,
        "uses_da3_for_prediction": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_json(feature_root / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v104 LingBot temporal tracker on the full selected scene rows.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--support-root", default=str(DEFAULT_SUPPORT_ROOT))
    parser.add_argument("--selected-rows", default=str(DEFAULT_SELECTED_ROWS))
    parser.add_argument("--baseline-rows", default=str(DEFAULT_BASELINE_ROWS))
    parser.add_argument("--max-points-per-frame", type=int, default=20000)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--min-pred-pixels", type=int, default=20)
    parser.add_argument("--min-gt-pixels", type=int, default=20)
    parser.add_argument("--cupy-device-id", type=int, default=0)
    parser.add_argument("--disable-cupy-iou", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    out = _project(args.output_root)
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"output root already exists and is non-empty; pass --force: {_rel(out)}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command_full_scene.txt").write_text(
        " ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n",
        encoding="utf-8",
    )

    input_summary = _build_full_feature_inputs(args, out)
    phase2_paths = input_summary["phase2_paths"]
    max_frame_count = max((int(v) for v in input_summary["scene_frame_counts"].values()), default=32)
    base.PHASE_ID = PHASE_ID
    base.VARIANTS = _full_variants(max_frame_count)
    forwarded = argparse.Namespace(
        output_root=str(out),
        feature_root=str(out / "full_scene_feature_inputs"),
        scene0011_phase2_root=str(_project(phase2_paths["scene0011_00"])),
        scene0050_phase2_root=str(_project(phase2_paths["scene0050_00"])),
        baseline_rows=str(_project(args.baseline_rows)),
        variants="",
        metric_scope_label="full_selected_scene_window_mean_from_phase7_rows; not full-dev beyond selected rows",
        scene_metric_scope_label="full_selected_scene_raw_gt_from_phase7_rows",
        scene_aggregate_scope_label="full_selected_scene_raw_gt_mean_from_phase7_rows",
        min_pred_pixels=args.min_pred_pixels,
        min_gt_pixels=args.min_gt_pixels,
        cupy_device_id=args.cupy_device_id,
        disable_cupy_iou=args.disable_cupy_iou,
        force=True,
    )
    summary = base.build(forwarded)
    summary.update(
        {
            "full_scene_eval_completed": True,
            "full_scene_input_summary": input_summary,
            "truthfulness_note": (
                "This is a full selected-row scene evaluation over Phase7 rows, not the prior first32 subset. "
                "It still evaluates only selected Stream4D mask rows, not every possible mask proposal."
            ),
        }
    )
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
