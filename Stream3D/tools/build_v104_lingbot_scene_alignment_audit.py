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
PHASE_ID = "v104_lingbot_map_only_phase4_scene_alignment_audit"
DEFAULT_OUT = AUDIT_ROOT / PHASE_ID
DEFAULT_SELECTED_ROWS = STREAM3D_ROOT / "outputs/audit/v87_phase1_mv_input_generation/frame_mask_selected_rows.csv"
DEFAULT_PIPELINE_SUMMARIES = {
    "scene0011_00": STREAM3D_ROOT / "outputs/audit/v66_soma_fullscene_pipeline_scene0011_00_stride5_conf02_integrated_d4rt/pipeline_summary.json",
    "scene0050_00": STREAM3D_ROOT / "outputs/audit/v65_soma_fullscene_pipeline_scene0050_stride5_conf02_integrated_d4rt/pipeline_summary.json",
}


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
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(float(str(value)))
    except Exception:
        return default


def _mask_dir_for_scene(scene_id: str) -> Path:
    summary_path = DEFAULT_PIPELINE_SUMMARIES.get(scene_id)
    if summary_path is None:
        return Path("__missing_mask_dir__")
    summary = _read_json(summary_path)
    raw = (
        summary.get("resolved_mask_dir")
        or summary.get("mask_frame_coverage", {}).get("mask_dir")
        or summary.get("mask_materialization_summary", {}).get("merge_summary", {}).get("final_mask_dir")
    )
    return _project(str(raw)) if raw else Path("__missing_mask_dir__")


def _has_lingbot_geometry(root: Path) -> bool:
    depth_dir = root / "depth"
    points_dir = root / "points"
    has_depth = depth_dir.exists() and any(depth_dir.glob("*.exr")) or depth_dir.exists() and any(depth_dir.glob("*.npy"))
    has_points = points_dir.exists() and any(points_dir.glob("*.npy"))
    return (root / "traj.txt").exists() and (root / "intrinsics.txt").exists() and (has_depth or has_points)


def _lingbot_sampling_frames(root: Path) -> list[int]:
    candidates = [root.parent / "gt" / "sampling.json", root / "sampling.json"]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        frames = []
        for value in payload.get("frames", []):
            try:
                frames.append(int(value))
            except Exception:
                continue
        return frames
    return []


def _complete_num_frames(root: Path) -> int:
    payload = _read_json(root / ".complete.json")
    try:
        return int(payload.get("metadata", {}).get("num_frames", 0))
    except Exception:
        return 0


def _scan_lingbot_roots(search_roots: list[Path], target_scenes: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for search_root in search_roots:
        root = _project(search_root)
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if "traj.txt" not in filenames or "intrinsics.txt" not in filenames:
                continue
            current = Path(dirpath)
            text = current.as_posix().lower()
            if "lingbot" not in text:
                continue
            for scene_id in target_scenes:
                same_scene = scene_id.lower() in text
                if not same_scene:
                    continue
                sampling_frames = _lingbot_sampling_frames(current)
                rows.append(
                    {
                        "schema_version": "stream4d_v104_lingbot_scene_alignment_candidate_root_v1",
                        "phase_id": PHASE_ID,
                        "scene_id": scene_id,
                        "lingbot_root": _rel(current),
                        "has_traj": (current / "traj.txt").exists(),
                        "has_intrinsics": (current / "intrinsics.txt").exists(),
                        "has_depth_or_points": _has_lingbot_geometry(current),
                        "complete_num_frames": _complete_num_frames(current),
                        "sampling_source_frame_count": len(sampling_frames),
                        "sampling_source_frames": sampling_frames,
                        "sampling_source_frames_preview": ";".join(str(v) for v in sampling_frames[:16]),
                        "same_scene_path_match": True,
                        "uses_d4rt_for_prediction": False,
                        "uses_da3_for_prediction": False,
                        "uses_gt_for_prediction": False,
                    }
                )
    return rows


def _sample_real_mask_rows(selected_rows: list[dict[str, str]], scene_id: str, sample_limit: int) -> list[dict[str, Any]]:
    import cv2

    mask_dir = _mask_dir_for_scene(scene_id)
    scene_rows = [row for row in selected_rows if row.get("scene_id") == scene_id]
    samples: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for row in scene_rows:
        frame_id = _as_int(row.get("frame_id"))
        mask_id = _as_int(row.get("mask_id") or row.get("selected_mask_id") or row.get("mask_id_or_generated_id"))
        key = (frame_id, mask_id)
        if frame_id < 0 or mask_id <= 0 or key in seen:
            continue
        seen.add(key)
        mask_path = mask_dir / f"{frame_id}.png"
        exists = mask_path.exists()
        pixel_count = 0
        shape: list[int] = []
        if exists:
            image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if image is not None:
                if image.ndim == 3:
                    image = image[..., 0]
                image = np.asarray(image, dtype=np.int64)
                shape = [int(v) for v in image.shape[:2]]
                pixel_count = int(np.count_nonzero(image == mask_id))
        samples.append(
            {
                "schema_version": "stream4d_v104_lingbot_scene_alignment_mask_sample_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene_id,
                "frame_id": frame_id,
                "mask_id": mask_id,
                "mask_path": _rel(mask_path),
                "mask_raster_exists": exists,
                "mask_id_pixel_count": pixel_count,
                "mask_image_shape": shape,
                "uses_d4rt_for_prediction": False,
                "uses_da3_for_prediction": False,
                "uses_gt_for_prediction": False,
            }
        )
        if len(samples) >= sample_limit:
            break
    return samples


def build(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _project(args.output_root)
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"output root already exists and is non-empty; pass --force: {_rel(out)}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")

    target_scenes = [part.strip() for part in str(args.target_scenes).split(",") if part.strip()]
    search_roots = [Path(part.strip()) for part in str(args.search_roots).split(",") if part.strip()]
    selected_rows = _read_csv(_project(args.selected_rows))
    candidate_rows = _scan_lingbot_roots(search_roots, target_scenes)

    mask_sample_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    for scene_id in target_scenes:
        scene_selected = [row for row in selected_rows if row.get("scene_id") == scene_id]
        selected_frame_ids = sorted({_as_int(row.get("frame_id")) for row in scene_selected if _as_int(row.get("frame_id")) >= 0})
        scene_candidates = [row for row in candidate_rows if row.get("scene_id") == scene_id and row.get("has_depth_or_points")]
        candidate_source_frames: set[int] = set()
        for row in scene_candidates:
            for value in row.get("sampling_source_frames", []):
                try:
                    candidate_source_frames.add(int(value))
                except Exception:
                    continue
        covered_selected_frames = sorted(set(selected_frame_ids) & candidate_source_frames)
        samples = _sample_real_mask_rows(selected_rows, scene_id, int(args.sample_rows_per_scene))
        mask_sample_rows.extend(samples)
        mask_ready = bool(scene_selected) and bool(samples) and all(
            bool(row.get("mask_raster_exists")) and int(row.get("mask_id_pixel_count", 0)) > 0 for row in samples
        )
        lingbot_ready = bool(scene_candidates)
        selected_frame_coverage_complete = bool(selected_frame_ids) and len(covered_selected_frames) == len(selected_frame_ids)
        selected_frame_coverage_ratio = (
            float(len(covered_selected_frames)) / float(len(selected_frame_ids)) if selected_frame_ids else 0.0
        )
        if not lingbot_ready:
            blocker = "MISSING_SAME_SCENE_LINGBOT_BSS_FOR_STREAM4D_MASKS"
        elif not selected_frame_coverage_complete:
            blocker = "SAME_SCENE_LINGBOT_BSS_SELECTED_FRAME_COVERAGE_INCOMPLETE"
        else:
            blocker = ""
        scene_rows.append(
            {
                "schema_version": "stream4d_v104_lingbot_scene_alignment_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene_id,
                "selected_frame_mask_row_count": len(scene_selected),
                "selected_unique_frame_count": len(selected_frame_ids),
                "selected_frame_min": selected_frame_ids[0] if selected_frame_ids else "",
                "selected_frame_max": selected_frame_ids[-1] if selected_frame_ids else "",
                "mask_dir": _rel(_mask_dir_for_scene(scene_id)),
                "sample_mask_rows_checked": len(samples),
                "sample_mask_raster_present_count": sum(bool(row.get("mask_raster_exists")) for row in samples),
                "sample_mask_id_present_count": sum(int(row.get("mask_id_pixel_count", 0)) > 0 for row in samples),
                "same_scene_lingbot_bss_root_count": len(scene_candidates),
                "lingbot_sampled_source_frame_count": len(candidate_source_frames),
                "lingbot_sampled_source_frames_preview": ";".join(str(v) for v in sorted(candidate_source_frames)[:16]),
                "covered_selected_unique_frame_count": len(covered_selected_frames),
                "selected_frame_coverage_ratio": selected_frame_coverage_ratio,
                "selected_frame_coverage_complete": selected_frame_coverage_complete,
                "mask_raster_ready": mask_ready,
                "same_scene_lingbot_bss_ready": lingbot_ready,
                "alignment_ready": mask_ready and lingbot_ready and selected_frame_coverage_complete,
                "blocker": blocker,
                "uses_d4rt_for_prediction": False,
                "uses_da3_for_prediction": False,
                "uses_gt_for_prediction": False,
            }
        )

    real_masks_available = bool(scene_rows) and all(bool(row["mask_raster_ready"]) for row in scene_rows)
    same_scene_lingbot_available = bool(scene_rows) and all(bool(row["same_scene_lingbot_bss_ready"]) for row in scene_rows)
    selected_frame_coverage_complete = bool(scene_rows) and all(bool(row["selected_frame_coverage_complete"]) for row in scene_rows)
    alignment_pass = real_masks_available and same_scene_lingbot_available and selected_frame_coverage_complete
    if alignment_pass:
        taxonomy = "READY_FOR_REAL_MASK_SUPPORT_ROWS"
        blocker = ""
    elif real_masks_available and same_scene_lingbot_available:
        taxonomy = "PARTIAL_SAME_SCENE_LINGBOT_BSS_SMOKE_ONLY_SELECTED_FRAME_COVERAGE_INCOMPLETE"
        blocker = "SAME_SCENE_LINGBOT_BSS_SELECTED_FRAME_COVERAGE_INCOMPLETE"
    else:
        taxonomy = "BLOCKED_MISSING_SAME_SCENE_LINGBOT_BSS_FOR_STREAM4D_MASKS"
        blocker = "MISSING_SAME_SCENE_LINGBOT_BSS_FOR_STREAM4D_MASKS"
    gate_rows = [
        {
            "schema_version": "stream4d_v104_lingbot_scene_alignment_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_name": "real_stream4d_mask_rasters_available",
            "pass": real_masks_available,
            "observed": {row["scene_id"]: row["selected_frame_mask_row_count"] for row in scene_rows},
            "required": "selected frame-mask rows and positive mask pixels for every target scene",
            "repair_direction": "restore v87 selected rows or CropFormer mask rasters",
            "uses_d4rt_for_prediction": False,
            "uses_da3_for_prediction": False,
            "uses_gt_for_prediction": False,
        },
        {
            "schema_version": "stream4d_v104_lingbot_scene_alignment_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_name": "same_scene_lingbot_bss_available",
            "pass": same_scene_lingbot_available,
            "observed": {row["scene_id"]: row["same_scene_lingbot_bss_root_count"] for row in scene_rows},
            "required": "LingBot BSS depth/points plus traj/intrinsics for every target Stream4D scene",
            "repair_direction": "run or export LingBot-Map BSS for scene0011_00/scene0050_00 before real mask support rows",
            "uses_d4rt_for_prediction": False,
            "uses_da3_for_prediction": False,
            "uses_gt_for_prediction": False,
        },
        {
            "schema_version": "stream4d_v104_lingbot_scene_alignment_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_name": "same_scene_lingbot_bss_covers_selected_frames",
            "pass": selected_frame_coverage_complete,
            "observed": {
                row["scene_id"]: {
                    "covered": row["covered_selected_unique_frame_count"],
                    "selected_unique": row["selected_unique_frame_count"],
                    "coverage_ratio": row["selected_frame_coverage_ratio"],
                }
                for row in scene_rows
            },
            "required": "LingBot BSS source frame ids cover every selected Stream4D mask frame",
            "repair_direction": "run full stride-5 LingBot configs, not only smoke8 configs, before full real-mask support rows",
            "uses_d4rt_for_prediction": False,
            "uses_da3_for_prediction": False,
            "uses_gt_for_prediction": False,
        },
    ]
    summary = {
        "schema_version": "stream4d_v104_lingbot_scene_alignment_summary_v1",
        "phase_id": PHASE_ID,
        "alignment_audit_pass": alignment_pass,
        "taxonomy": taxonomy,
        "blocker": blocker,
        "target_scenes": target_scenes,
        "real_masks_available": real_masks_available,
        "same_scene_lingbot_bss_available": same_scene_lingbot_available,
        "same_scene_lingbot_selected_frame_coverage_complete": selected_frame_coverage_complete,
        "stream4d_metric_ready": False,
        "stream4d_metric_note": "This audit checks scene alignment only; no AP/MV_AP metric is produced.",
        "outputs": {
            "scene_alignment_rows": _rel(out / "scene_alignment_rows.csv"),
            "real_mask_sample_rows": _rel(out / "real_mask_sample_rows.csv"),
            "lingbot_candidate_roots": _rel(out / "lingbot_candidate_roots.csv"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "summary": _rel(out / "summary.json"),
        },
        "runtime_sec": round(time.time() - t0, 3),
    }
    _write_csv(out / "scene_alignment_rows.csv", scene_rows)
    _write_csv(out / "real_mask_sample_rows.csv", mask_sample_rows)
    _write_csv(
        out / "lingbot_candidate_roots.csv",
        candidate_rows,
        fields=[
            "schema_version",
            "phase_id",
            "scene_id",
            "lingbot_root",
            "has_traj",
            "has_intrinsics",
            "has_depth_or_points",
            "complete_num_frames",
            "sampling_source_frame_count",
            "sampling_source_frames_preview",
            "same_scene_path_match",
            "uses_d4rt_for_prediction",
            "uses_da3_for_prediction",
            "uses_gt_for_prediction",
        ],
    )
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether LingBot BSS geometry and Stream4D masks are in the same target scene.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--selected-rows", default=str(DEFAULT_SELECTED_ROWS))
    parser.add_argument("--target-scenes", default="scene0011_00,scene0050_00")
    parser.add_argument("--search-roots", default="results,Stream3D/outputs")
    parser.add_argument("--sample-rows-per-scene", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
