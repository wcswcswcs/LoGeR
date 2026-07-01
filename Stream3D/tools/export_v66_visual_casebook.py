from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from tools.run_v65_scene_multiview_ap import _load_gt_2d, _read_label_png, _sha256, _write_csv, _write_json  # noqa: E402
from tools.run_v66_ledger_audit import _best_variant, _load_support  # noqa: E402
from tools.run_v66_scene_mv_ap_probe5 import DEFAULT_SCENES, _discover_pipeline_root, _mask_dir_from_pipeline, _parse_csv_list, _rel  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _label_colors(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    colors = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for value in np.unique(labels):
        value = int(value)
        if value <= 0:
            continue
        color = np.asarray(
            [
                (value * 37 + 59) % 255,
                (value * 67 + 101) % 255,
                (value * 97 + 149) % 255,
            ],
            dtype=np.uint8,
        )
        colors[labels == value] = color
    return colors


def _overlay(rgb: np.ndarray, labels: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    out = rgb.copy()
    colors = _label_colors(labels)
    mask = labels > 0
    out[mask] = ((1.0 - alpha) * out[mask].astype(np.float32) + alpha * colors[mask].astype(np.float32)).astype(np.uint8)
    return out


def _panel_title(image: np.ndarray, title: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), thickness=-1)
    cv2.putText(out, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _resize_rgb(image: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    if image.shape[:2] == shape_hw:
        return image
    return cv2.resize(image, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_LINEAR)


def _load_mask(mask_dir: Path, frame_id: int, shape_hw: tuple[int, int]) -> np.ndarray:
    path = mask_dir / f"{int(frame_id)}.png"
    if not path.exists():
        return np.zeros(shape_hw, dtype=np.int64)
    return _read_label_png(path, shape_hw)


def _majority_gt_iou(gt: np.ndarray, mask_binary: np.ndarray) -> tuple[int, float]:
    if not np.any(mask_binary):
        return 0, 0.0
    gt_values = gt[mask_binary]
    ids, counts = np.unique(gt_values[gt_values > 0], return_counts=True)
    if ids.size == 0:
        return 0, 0.0
    best_idx = int(np.argmax(counts))
    gt_id = int(ids[best_idx])
    inter = int(np.count_nonzero(mask_binary & (gt == gt_id)))
    union = int(np.count_nonzero(mask_binary | (gt == gt_id)))
    return gt_id, float(inter / max(1, union))


def _make_case_image(
    *,
    scene: str,
    frame_id: int,
    mask_id: int,
    mask_dir: Path,
    mapping: dict[tuple[int, int], int],
    output_path: Path,
) -> dict[str, Any]:
    stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
    shape_hw = tuple(int(value) for value in stream.load_depth(frame_id).shape)
    rgb = _resize_rgb(stream.load_rgb(frame_id), shape_hw)
    gt = _load_gt_2d(scene, frame_id, shape_hw)
    mask = _load_mask(mask_dir, frame_id, shape_hw)
    pred = np.zeros(shape_hw, dtype=np.int64)
    for current_mask_id in np.unique(mask):
        current_mask_id = int(current_mask_id)
        if current_mask_id <= 0:
            continue
        object_idx = int(mapping.get((frame_id, current_mask_id), 0))
        if object_idx > 0:
            pred[mask == current_mask_id] = object_idx
    selected = (mask == int(mask_id)).astype(np.int64)
    gt_id, mask_gt_iou = _majority_gt_iou(gt, selected > 0)
    panels = [
        _panel_title(rgb, "RGB"),
        _panel_title(_overlay(rgb, gt), "GT instance diagnostic"),
        _panel_title(_overlay(rgb, pred), "SOMA current objects"),
        _panel_title(_overlay(rgb, selected), f"Conflict mask {mask_id}"),
    ]
    thumb_h = 240
    resized = []
    for panel in panels:
        scale = thumb_h / panel.shape[0]
        resized.append(cv2.resize(panel, (int(panel.shape[1] * scale), thumb_h), interpolation=cv2.INTER_AREA))
    canvas = np.concatenate(resized, axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return {"gt_id": gt_id, "mask_gt_iou": mask_gt_iou}


def _manifest_refs(pipeline_root: Path) -> str:
    manifests = [
        pipeline_root / "reprojection_ledger/visualization_manifest.json",
        pipeline_root / "local_objectlets/visualization_manifest.json",
        pipeline_root / "mask_component_support/visualization_manifest.json",
        pipeline_root / "chunk_universe/visualization_manifest.json",
    ]
    return ";".join(_rel(path) for path in manifests if path.exists())


def run(args: argparse.Namespace) -> dict[str, Any]:
    scenes = _parse_csv_list(args.scenes)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    conflict_rows = _read_csv(ROOT / args.conflict_rows)
    rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for scene in scenes:
        pipeline_root = _discover_pipeline_root(scene)
        if pipeline_root is None:
            missing_rows.append({"scene_id": scene, "missing": "soma_fullscene_pipeline_root"})
            continue
        variant = _best_variant(pipeline_root)
        support = _load_support(pipeline_root=pipeline_root, scene=scene, variant=variant)
        mapping = support["current_mapping"]
        mask_dir = _mask_dir_from_pipeline(pipeline_root)
        scene_conflicts = [row for row in conflict_rows if row.get("scene_id") == scene]
        scene_conflicts = sorted(
            scene_conflicts,
            key=lambda row: (int(float(row.get("object_count") or 0)), int(float(row.get("total_vote_weight") or 0))),
            reverse=True,
        )
        for index, conflict in enumerate(scene_conflicts[: int(args.cases_per_scene)]):
            frame_id = int(float(conflict.get("frame_id") or 0))
            mask_id = int(float(conflict.get("mask_id") or 0))
            image_path = output_root / "images" / scene / f"{index:03d}_frame{frame_id:06d}_mask{mask_id:04d}.png"
            image_diag = _make_case_image(
                scene=scene,
                frame_id=frame_id,
                mask_id=mask_id,
                mask_dir=mask_dir,
                mapping=mapping,
                output_path=image_path,
            )
            rows.append(
                {
                    "case_id": f"{scene}_duplicate_conflict_{index:03d}",
                    "scene_id": scene,
                    "frame_id": frame_id,
                    "gt_id": image_diag["gt_id"],
                    "soma_object_ids": conflict.get("object_indices", ""),
                    "stream3d_object_id": "",
                    "mask_id": mask_id,
                    "mask_gt_iou": image_diag["mask_gt_iou"],
                    "failure_type": "duplicate_frame_mask_conflict",
                    "screenshot_path": _rel(image_path),
                    "viewer_bookmark": _manifest_refs(pipeline_root),
                    "recommended_fix": "local one-mask-one-owner conflict-aware objectlet formation; WTA alone already tested in Phase 2",
                    "diagnostic_only": True,
                    "uses_gt_for_prediction": False,
                }
            )
    _write_csv(output_root / "casebook_rows.csv", rows)
    _write_csv(output_root / "missing_input_rows.csv", missing_rows)
    sha_rows = []
    image_paths = [output_root / Path(str(row["screenshot_path"])).relative_to("outputs/audit/v66_visual_casebook") for row in rows if str(row["screenshot_path"]).startswith("outputs/audit/v66_visual_casebook")]
    for path in [output_root / "casebook_rows.csv", output_root / "missing_input_rows.csv", *image_paths[:20]]:
        if path.exists():
            sha_rows.append({"path": _rel(path), "sha256": _sha256(path), "bytes": int(path.stat().st_size)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    scenes_with_cases = sorted({row["scene_id"] for row in rows})
    gate = {
        "at_least_5_probe_scenes_visualized": len(scenes_with_cases) >= 5,
        "at_least_50_cases_in_casebook": len(rows) >= 50,
        "has_2d_png_screenshots": bool(rows) and all((ROOT / row["screenshot_path"]).exists() for row in rows),
        "has_3d_manifest_references": bool(rows) and all(str(row["viewer_bookmark"]) for row in rows),
        "live_viewer_load_verified": False,
    }
    gate["pass"] = (
        gate["at_least_5_probe_scenes_visualized"]
        and gate["at_least_50_cases_in_casebook"]
        and gate["has_2d_png_screenshots"]
        and gate["has_3d_manifest_references"]
    )
    summary = {
        "phase": "v66_visual_casebook",
        "diagnostic_only": True,
        "gate": gate,
        "case_count": len(rows),
        "scenes_with_cases": scenes_with_cases,
        "casebook_rows_csv": _rel(output_root / "casebook_rows.csv"),
        "missing_input_rows_csv": _rel(output_root / "missing_input_rows.csv"),
        "notes": [
            "Casebook images are 2D RGB/GT/SOMA/mask panels for duplicate frame-mask conflicts.",
            "viewer_bookmark contains existing pipeline visualization manifests; live Viser loading was not verified by this exporter.",
            "GT panels are diagnostic-only and not used for method prediction.",
        ],
    }
    _write_json(output_root / "visual_casebook_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export v66 visual failure casebook PNGs and CSV rows.")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--conflict-rows", default="outputs/audit/v66_ledger_audit_full/frame_mask_conflict_rows.csv")
    parser.add_argument("--cases-per-scene", type=int, default=10)
    parser.add_argument("--output-root", default="outputs/audit/v66_visual_casebook")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
