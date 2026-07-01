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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_v65_scene_multiview_ap import SparseSceneIoU, _sha256, _summarize_iou, _write_csv, _write_json


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_label(path: Path, shape_hw: tuple[int, int] | None = None) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to read label image: {path}")
    if image.ndim == 3:
        image = image[..., 0]
    image = np.asarray(image, dtype=np.int64)
    if shape_hw is not None and image.shape[:2] != tuple(shape_hw):
        image = cv2.resize(image, (int(shape_hw[1]), int(shape_hw[0])), interpolation=cv2.INTER_NEAREST)
        image = np.asarray(image, dtype=np.int64)
    return image


def _selected_maps(pipeline_root: Path, scene: str, variant: str) -> dict[str, Any]:
    selected: dict[str, tuple[str, int]] = {}
    object_to_idx: dict[str, int] = {}
    with (pipeline_root / "local_objectlets" / "objectlet_rows.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("scene") != scene or row.get("variant") != variant:
                continue
            object_id = str(row.get("objectlet_id") or "")
            candidate_id = str(row.get("candidate_id") or "")
            if not object_id or not candidate_id:
                continue
            object_to_idx.setdefault(object_id, len(object_to_idx) + 1)
            selected[candidate_id] = (object_id, object_to_idx[object_id])
    objectlet_map: dict[tuple[int, int], int] = {}
    gt_oracle_map: dict[tuple[int, int], int] = {}
    objectlet_conflicts = 0
    gt_oracle_conflicts = 0
    with (pipeline_root / "reprojection_ledger" / "reprojection_ledger_rows.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            selected_item = selected.get(str(row.get("candidate_id") or ""))
            if selected_item is None or str(row.get("reprojection_success")) != "True":
                continue
            parts = str(row.get("best_mask_observation_id") or "").split(":")
            if len(parts) != 3 or parts[0] != scene:
                continue
            frame_id = int(parts[1])
            mask_id = int(parts[2])
            object_idx = int(selected_item[1])
            key = (frame_id, mask_id)
            if key in objectlet_map and objectlet_map[key] != object_idx:
                objectlet_conflicts += 1
                object_idx = min(objectlet_map[key], object_idx)
            objectlet_map[key] = object_idx
            gt_id = int(float(row.get("diagnostic_best_gt") or 0))
            if gt_id > 0:
                if key in gt_oracle_map and gt_oracle_map[key] != gt_id:
                    gt_oracle_conflicts += 1
                gt_oracle_map[key] = gt_id
    return {
        "object_count": len(object_to_idx),
        "objectlet_map": objectlet_map,
        "gt_oracle_map": gt_oracle_map,
        "objectlet_conflicts": objectlet_conflicts,
        "gt_oracle_conflicts": gt_oracle_conflicts,
    }


def _frame_ids(scene: str, stride: int) -> list[int]:
    color_dir = ROOT / "data" / "scannet" / "processed" / scene / "color"
    return sorted(int(path.stem) for path in color_dir.glob("*.jpg") if int(path.stem) % int(stride) == 0)


def _run_case(
    *,
    scene: str,
    stride: int,
    shape_mode: str,
    pred_kind: str,
    mask_dir: Path,
    mapping: dict[tuple[int, int], int],
) -> dict[str, Any]:
    acc = SparseSceneIoU()
    frame_ids = _frame_ids(scene, stride)
    missing_masks = 0
    raw_mask_pixels = 0
    mapped_pixels = 0
    for frame_id in frame_ids:
        gt_path = ROOT / "data" / "scannet" / "processed" / scene / "instance" / "instance" / f"{frame_id}.png"
        if shape_mode == "rgb":
            gt = _load_label(gt_path)
            shape_hw = gt.shape[:2]
        elif shape_mode == "depth":
            depth_path = ROOT / "data" / "scannet" / "processed" / scene / "depth" / f"{frame_id}.png"
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise FileNotFoundError(f"failed to read depth image: {depth_path}")
            shape_hw = depth.shape[:2]
            gt = _load_label(gt_path, shape_hw)
        else:
            raise ValueError(f"unsupported shape_mode={shape_mode}")
        pred = np.zeros(shape_hw, dtype=np.int64)
        mask_path = mask_dir / f"{frame_id}.png"
        if not mask_path.exists():
            missing_masks += 1
        else:
            mask = _load_label(mask_path, shape_hw)
            raw_mask_pixels += int(np.count_nonzero(mask > 0))
            for mask_id in np.unique(mask):
                mask_id = int(mask_id)
                if mask_id <= 0:
                    continue
                label = int(mapping.get((int(frame_id), mask_id), 0))
                if label > 0:
                    pred[mask == mask_id] = label
            mapped_pixels += int(np.count_nonzero(pred > 0))
        acc.add(pred, gt)
    summary, _iou, _pred_ids, _gt_ids = _summarize_iou(
        accumulator=acc,
        min_pred_pixels=1,
        min_gt_pixels=1,
        score_mode="constant",
        input_scores=None,
    )
    return {
        "scene": scene,
        "stride": int(stride),
        "shape_mode": shape_mode,
        "pred_kind": pred_kind,
        "frame_count": len(frame_ids),
        "missing_masks": int(missing_masks),
        "raw_mask_pixels": int(raw_mask_pixels),
        "mapped_pixels": int(mapped_pixels),
        "AP": summary["ap"],
        "AP50": summary["ap50"],
        "AP25": summary["ap25"],
        "pred_count": summary["evaluated_pred_count"],
        "gt_count": summary["evaluated_gt_count"],
        "gt_best_iou_mean": summary["gt_best_iou_mean"],
        "gt_recall_best_iou_ge_025": summary["gt_recall_best_iou_ge_025"],
        "gt_recall_best_iou_ge_050": summary["gt_recall_best_iou_ge_050"],
        "score_free_match_at_025": summary["score_free_match_at_025"],
        "score_free_match_at_050": summary["score_free_match_at_050"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    pipeline_root = ROOT / args.pipeline_root if not Path(args.pipeline_root).is_absolute() else Path(args.pipeline_root)
    output_root = ROOT / args.output_root if not Path(args.output_root).is_absolute() else Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pipeline_summary = _read_json(pipeline_root / "pipeline_summary.json")
    objectlet_summary = _read_json(pipeline_root / "local_objectlets" / "local_objectlet_summary.json")
    variant = str(objectlet_summary["best_real_variant"])
    mask_dir = ROOT / str(pipeline_summary["mask_frame_coverage"]["mask_dir"])
    maps = _selected_maps(pipeline_root, args.scene, variant)
    rows: list[dict[str, Any]] = []
    for stride in [int(value) for value in str(args.strides).split(",") if value.strip()]:
        rows.append(
            _run_case(
                scene=args.scene,
                stride=stride,
                shape_mode="depth",
                pred_kind="objectlet",
                mask_dir=mask_dir,
                mapping=maps["objectlet_map"],
            )
        )
        rows.append(
            _run_case(
                scene=args.scene,
                stride=stride,
                shape_mode="rgb",
                pred_kind="objectlet",
                mask_dir=mask_dir,
                mapping=maps["objectlet_map"],
            )
        )
        rows.append(
            _run_case(
                scene=args.scene,
                stride=stride,
                shape_mode="rgb",
                pred_kind="gt_oracle_selected_masks",
                mask_dir=mask_dir,
                mapping=maps["gt_oracle_map"],
            )
        )
    _write_csv(output_root / "sanity_rows.csv", rows)
    payload = {
        "phase": "v65_scene_multiview_ap_sanity",
        "scene": args.scene,
        "pipeline_root": str(pipeline_root.relative_to(ROOT)),
        "mask_dir": str(mask_dir.relative_to(ROOT)),
        "objectlet_variant": variant,
        "object_count": int(maps["object_count"]),
        "objectlet_map_pair_count": int(len(maps["objectlet_map"])),
        "gt_oracle_map_pair_count": int(len(maps["gt_oracle_map"])),
        "objectlet_conflicts": int(maps["objectlet_conflicts"]),
        "gt_oracle_conflicts": int(maps["gt_oracle_conflicts"]),
        "rows": rows,
        "notes": [
            "depth/objectlet should reproduce the main SOMA constant-score output.",
            "rgb/objectlet tests whether nearest resize to depth grid caused the low score.",
            "rgb/gt_oracle_selected_masks groups the same selected ledger masks by diagnostic_best_gt; this uses GT and is diagnostic-only.",
        ],
        "outputs": {
            "summary_json": str((output_root / "summary.json").relative_to(ROOT)),
            "sanity_rows_csv": str((output_root / "sanity_rows.csv").relative_to(ROOT)),
        },
    }
    _write_json(output_root / "summary.json", payload)
    sha_rows = [
        {"path": str((output_root / "summary.json").relative_to(ROOT)), "sha256": _sha256(output_root / "summary.json")},
        {"path": str((output_root / "sanity_rows.csv").relative_to(ROOT)), "sha256": _sha256(output_root / "sanity_rows.csv")},
    ]
    _write_csv(output_root / "SHA256SUMS.csv", sha_rows)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit v65 scene-level multi-view AP sanity checks.")
    parser.add_argument("--scene", default="scene0050_00")
    parser.add_argument("--strides", default="5,10")
    parser.add_argument("--pipeline-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
