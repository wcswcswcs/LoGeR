#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
STREAM3D_ROOT = REPO_ROOT / "Stream3D"
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools.run_v65_scene_multiview_ap import run as run_v65  # type: ignore


def _project(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
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


def _write_records_json(path: Path, rows: list[dict[str, Any]], *, schema_version: str) -> None:
    _write_json(path, {"schema_version": schema_version, "row_count": len(rows), "rows": rows})


def _sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_list(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _numeric_ids(path: Path, suffix: str) -> list[int]:
    if not path.exists():
        return []
    ids: list[int] = []
    for item in path.glob(f"*{suffix}"):
        try:
            ids.append(int(item.stem))
        except ValueError:
            continue
    return sorted(set(ids))


def _label_image_ids(path: Path) -> list[int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return []
    if image.ndim == 3:
        image = image[..., 0]
    values = np.unique(image)
    return [int(value) for value in values if int(value) > 0]


def _expected_stride_ids(scene_id: str, stride: int) -> list[int]:
    color_dir = STREAM3D_ROOT / "data" / "scannet" / "processed" / scene_id / "color"
    return [frame_id for frame_id in _numeric_ids(color_dir, ".jpg") if frame_id % int(stride) == 0]


def _write_support_root(
    *,
    pipeline_root: Path,
    variant_id: str,
    mask_root: Path,
    scenes: list[str],
    stride: int,
) -> dict[str, Any]:
    objectlet_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    frame_ids_by_scene: dict[str, list[int]] = {}
    mask_frame_count_by_scene: dict[str, int] = {}
    mask_id_count_by_scene: dict[str, int] = {}
    support_pair_count_by_scene: dict[str, int] = {}
    for scene_id in scenes:
        mask_dir = mask_root / scene_id / "mask"
        mask_frame_ids = _numeric_ids(mask_dir, ".png")
        frame_ids_by_scene[scene_id] = mask_frame_ids
        mask_frame_count_by_scene[scene_id] = len(mask_frame_ids)
        scene_mask_ids: set[int] = set()
        scene_support_pairs = 0
        for frame_id in mask_frame_ids:
            mask_path = mask_dir / f"{int(frame_id)}.png"
            for mask_id in _label_image_ids(mask_path):
                scene_mask_ids.add(int(mask_id))
                objectlet_id = f"{variant_id}:{scene_id}:track{int(mask_id):04d}"
                candidate_id = f"{objectlet_id}:candidate"
                objectlet_rows.append(
                    {
                        "scene": scene_id,
                        "variant": variant_id,
                        "objectlet_id": objectlet_id,
                        "candidate_id": candidate_id,
                    }
                )
                ledger_rows.append(
                    {
                        "scene": scene_id,
                        "candidate_id": candidate_id,
                        "reprojection_success": True,
                        "best_mask_observation_id": f"{scene_id}:{int(frame_id)}:{int(mask_id)}",
                        "mask_path": _rel(mask_path),
                    }
                )
                scene_support_pairs += 1
        mask_id_count_by_scene[scene_id] = len(scene_mask_ids)
        support_pair_count_by_scene[scene_id] = scene_support_pairs

    local_dir = pipeline_root / "local_objectlets"
    ledger_dir = pipeline_root / "reprojection_ledger"
    _write_records_json(local_dir / "objectlet_records.json", objectlet_rows, schema_version="stream4d_v105_mvdiag_objectlet_manifest_v1")
    _write_records_json(ledger_dir / "reprojection_ledger_records.json", ledger_rows, schema_version="stream4d_v105_mvdiag_reprojection_manifest_v1")
    _write_json(
        local_dir / "local_objectlet_summary.json",
        {
            "schema_version": "stream4d_v105_mvdiag_local_objectlet_summary_v1",
            "best_real_variant": variant_id,
            "variant_id": variant_id,
            "object_id_policy": "mask_id_is_track",
            "record_format": "json_manifest",
            "objectlet_records": _rel(local_dir / "objectlet_records.json"),
            "reprojection_ledger_records": _rel(ledger_dir / "reprojection_ledger_records.json"),
            "objectlet_row_count": len(objectlet_rows),
            "ledger_row_count": len(ledger_rows),
            "support_pair_count": len(ledger_rows),
            "mask_frame_count_by_scene": mask_frame_count_by_scene,
            "mask_id_count_by_scene": mask_id_count_by_scene,
        },
    )
    _write_json(
        pipeline_root / "pipeline_summary.json",
        {
            "schema_version": "stream4d_v105_mvdiag_pipeline_summary_v1",
            "variant_id": variant_id,
            "mask_root": _rel(mask_root),
            "mask_frame_coverage": {},
            "object_id_policy": "mask_id_is_track",
            "frame_ids_by_scene": frame_ids_by_scene,
            "stride": int(stride),
            "uses_gt_for_prediction": False,
            "uses_future": False,
            "diagnostic_only": True,
            "note": "Generated only to let the v65 MV-AP diagnostic read existing v105 label PNGs. This is not a method-stage output.",
        },
    )
    return {
        "pipeline_root": _rel(pipeline_root),
        "mask_root": _rel(mask_root),
        "objectlet_records_sha256": _sha256(local_dir / "objectlet_records.json"),
        "reprojection_ledger_records_sha256": _sha256(ledger_dir / "reprojection_ledger_records.json"),
        "pipeline_summary_sha256": _sha256(pipeline_root / "pipeline_summary.json"),
        "objectlet_row_count": len(objectlet_rows),
        "ledger_row_count": len(ledger_rows),
        "mask_frame_count_by_scene": mask_frame_count_by_scene,
        "mask_id_count_by_scene": mask_id_count_by_scene,
        "support_pair_count_by_scene": support_pair_count_by_scene,
    }


def _run_eval(
    *,
    scene_id: str,
    pipeline_root: Path,
    output_root: Path,
    stride: int,
    max_frames: int,
) -> dict[str, Any]:
    ns = argparse.Namespace(
        scene=scene_id,
        methods="soma",
        strides=str(int(stride)),
        pipeline_root=str(pipeline_root),
        stream3d_config="scannet",
        output_root=str(output_root),
        score_mode="constant",
        min_pred_pixels=1,
        min_gt_pixels=1,
        vertex_nn_radius=0.08,
        vertex_cache_root=str(output_root / "vertex_cache_unused_for_soma"),
        use_cache=0,
        max_frames=int(max_frames),
    )
    payload = run_v65(ns)
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    if not rows:
        raise RuntimeError(f"v65 returned no rows for scene={scene_id} max_frames={max_frames}")
    return rows[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v105 final/local2history MV_AP window and scene diagnostics from existing label PNGs.")
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--variant-id", required=True)
    parser.add_argument("--mask-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--window-max-frames", type=int, default=64)
    args = parser.parse_args()

    scenes = _parse_list(args.scenes)
    variant_id = str(args.variant_id)
    mask_root = _project(args.mask_root)
    output_root = _project(args.output_root)
    pipeline_root = output_root / "pipeline_root" / variant_id
    support = _write_support_root(
        pipeline_root=pipeline_root,
        variant_id=variant_id,
        mask_root=mask_root,
        scenes=scenes,
        stride=int(args.stride),
    )

    rows: list[dict[str, Any]] = []
    for scene_id in scenes:
        expected_scene_frames = _expected_stride_ids(scene_id, int(args.stride))
        available_mask_frames = _numeric_ids(mask_root / scene_id / "mask", ".png")
        available_set = set(available_mask_frames)
        expected_set = set(expected_scene_frames)
        window_row = _run_eval(
            scene_id=scene_id,
            pipeline_root=pipeline_root,
            output_root=output_root / "evaluation" / "window64" / scene_id,
            stride=int(args.stride),
            max_frames=int(args.window_max_frames),
        )
        scene_row = _run_eval(
            scene_id=scene_id,
            pipeline_root=pipeline_root,
            output_root=output_root / "evaluation" / "scene_padded_missing_predictions_zero" / scene_id,
            stride=int(args.stride),
            max_frames=0,
        )
        common = {
            "schema_version": "stream4d_v105_mv_scene_window_metric_row_v1",
            "split_name": str(args.split_name),
            "variant_id": variant_id,
            "scene_id": scene_id,
            "stride": int(args.stride),
            "mask_root": _rel(mask_root),
            "pipeline_root": _rel(pipeline_root),
            "available_mask_frame_count": len(available_mask_frames),
            "available_mask_first": int(available_mask_frames[0]) if available_mask_frames else None,
            "available_mask_last": int(available_mask_frames[-1]) if available_mask_frames else None,
            "expected_stride_frame_count": len(expected_scene_frames),
            "expected_stride_first": int(expected_scene_frames[0]) if expected_scene_frames else None,
            "expected_stride_last": int(expected_scene_frames[-1]) if expected_scene_frames else None,
            "missing_expected_prediction_frame_count": len([frame_id for frame_id in expected_scene_frames if frame_id not in available_set]),
            "extra_prediction_frame_count": len([frame_id for frame_id in available_mask_frames if frame_id not in expected_set]),
        }
        rows.append(
            {
                **common,
                "metric_scope": "MV_AP_window_current_64frame_stride5_v65_soma",
                "frame_count": window_row.get("frame_count"),
                "MV_AP_window": window_row.get("AP"),
                "MV_AP50_window": window_row.get("AP50"),
                "MV_AP25_window": window_row.get("AP25"),
                "evaluated_pred_count_window": window_row.get("evaluated_pred_count"),
                "evaluated_gt_count_window": window_row.get("evaluated_gt_count"),
                "gt_best_iou_mean_window": window_row.get("gt_best_iou_mean"),
                "summary_json": window_row.get("summary_json"),
                "full_scene_prediction_complete": False,
                "diagnostic_note": "Window diagnostic over the same first 64 stride-5 frames used by the v105 artifacts.",
            }
        )
        rows.append(
            {
                **common,
                "metric_scope": "MV_AP_scene_stride5_missing_predictions_zero_padded_v65_soma",
                "frame_count": scene_row.get("frame_count"),
                "MV_AP_scene": scene_row.get("AP"),
                "MV_AP50_scene": scene_row.get("AP50"),
                "MV_AP25_scene": scene_row.get("AP25"),
                "evaluated_pred_count_scene": scene_row.get("evaluated_pred_count"),
                "evaluated_gt_count_scene": scene_row.get("evaluated_gt_count"),
                "gt_best_iou_mean_scene": scene_row.get("gt_best_iou_mean"),
                "summary_json": scene_row.get("summary_json"),
                "full_scene_prediction_complete": len(expected_scene_frames) > 0 and expected_set.issubset(available_set),
                "diagnostic_note": "Scene diagnostic over all stride-5 RGB frames. Missing method predictions are read by v65 as empty masks, so this is not a valid full-scene method claim unless full_scene_prediction_complete is true.",
            }
        )

    _write_records_json(output_root / "mv_scene_window_metric_records.json", rows, schema_version="stream4d_v105_mv_scene_window_metric_records_v1")
    summary = {
        "schema_version": "stream4d_v105_mv_scene_window_summary_v1",
        "split_name": str(args.split_name),
        "variant_id": variant_id,
        "scenes": scenes,
        "stride": int(args.stride),
        "window_max_frames": int(args.window_max_frames),
        "diagnostic_only": True,
        "gate_usage": "MV_AP_window and MV_AP_scene are reported per v105 plan but are not stage gates.",
        "scene_metric_boundary": "MV_AP_scene rows are padded with empty predictions after available mask frames when full_scene_prediction_complete=false.",
        "support": support,
        "metric_records": _rel(output_root / "mv_scene_window_metric_records.json"),
        "metric_records_sha256": _sha256(output_root / "mv_scene_window_metric_records.json"),
        "rows": rows,
    }
    _write_json(output_root / "mv_scene_window_summary.json", summary)
    summary["summary_sha256"] = _sha256(output_root / "mv_scene_window_summary.json")
    _write_json(output_root / "mv_scene_window_summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
