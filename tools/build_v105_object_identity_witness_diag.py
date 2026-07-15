#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_label(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim != 2:
        raise ValueError(f"expected 2D label image: {path}")
    return label


def _read_rgb(path: Path) -> np.ndarray:
    rgb = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(path)
    return rgb


def _parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _parse_crop(value: str) -> tuple[int, int, int, int]:
    vals = _parse_ints(value)
    if len(vals) != 4:
        raise ValueError(f"expected crop x1,y1,x2,y2, got {value!r}")
    x1, y1, x2, y2 = vals
    if x2 < x1 or y2 < y1:
        raise ValueError(f"invalid crop coordinates: {value!r}")
    return x1, y1, x2, y2


def _color_for_id(label_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(int(label_id) * 1009 + 17)
    return tuple(int(v) for v in rng.integers(40, 255, size=3))


def _id_stats(label: np.ndarray, ids: list[int], crop: tuple[int, int, int, int]) -> list[dict[str, Any]]:
    x1, y1, x2, y2 = crop
    rows: list[dict[str, Any]] = []
    for label_id in ids:
        mask = label == int(label_id)
        area = int(mask.sum())
        if area <= 0:
            rows.append({"id": int(label_id), "area": 0, "present": False})
            continue
        ys, xs = np.where(mask)
        rows.append(
            {
                "id": int(label_id),
                "area": area,
                "present": True,
                "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "crop_area": int(mask[y1 : y2 + 1, x1 : x2 + 1].sum()),
            }
        )
    return rows


def _draw_crop(
    *,
    rgb: np.ndarray,
    label: np.ndarray,
    ids: list[int],
    crop: tuple[int, int, int, int],
    title: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    stats = _id_stats(label, ids, crop)
    panel = rgb.copy()
    overlay = np.zeros_like(panel)
    for row in stats:
        if not row.get("present"):
            continue
        label_id = int(row["id"])
        color = _color_for_id(label_id)
        mask = label == label_id
        overlay[mask] = color
        x1, y1, x2, y2 = [int(v) for v in row["bbox"]]
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            panel,
            f"id={label_id} a={int(row['area'])}",
            (max(0, x1), max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    panel = cv2.addWeighted(panel, 0.62, overlay, 0.38, 0)
    x1, y1, x2, y2 = crop
    cv2.rectangle(panel, (x1, y1), (x2, y2), (255, 255, 255), 3)
    crop_img = panel[y1 : y2 + 1, x1 : x2 + 1].copy()
    cv2.putText(crop_img, title, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return crop_img, stats


def _variant_paths(scene_id: str) -> dict[str, Path]:
    return {
        "v6_global": REPO_ROOT
        / "Stream3D/outputs/audit/v105_fullscene_l2h_tinylock_repair_20260711/local2history_reappearance/masks/P8_l2h_reappearance_weakoverride_tinylock_v6"
        / scene_id
        / "mask",
        "v9_global": REPO_ROOT
        / "Stream3D/outputs/audit/v105_fullscene_l2h_vertexremap_v9_scene0591_23_45_20260711/local2history_reappearance/masks/P8_l2h_reappearance_weakoverride_vertexremap_v9_scene0591_23_45"
        / scene_id
        / "mask",
        "v10_global": REPO_ROOT
        / "Stream3D/outputs/audit/v105_fullscene_l2h_objectcomplete_v10_scene0591_localreplace_20260711/local2history_reappearance/masks/P8_l2h_reappearance_weakoverride_objectcomplete_v10_scene0591_localreplace"
        / scene_id
        / "mask",
        "v10_local": REPO_ROOT
        / "Stream3D/outputs/audit/v105_fullscene_multichunk_v10_objectcomplete_scene0591_640_960_1120_20260711/assembled_scene_videos/sgq_local/masks/P6_v10_objectcomplete_scene0591_640_960_1120_localreplace_diagnostic"
        / scene_id
        / "mask",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build v105 object identity witness crop diagnostics.")
    parser.add_argument("--scene-id", default="scene0591_00")
    parser.add_argument("--boundary", default="002->003")
    parser.add_argument("--frames", default="635,640,645,650")
    parser.add_argument("--crop", default="1160,160,1295,880")
    parser.add_argument("--global-focus-ids", default="98,104,110,111,118,129")
    parser.add_argument("--local-focus-ids", default="3,6,7,10,11,33")
    parser.add_argument(
        "--output-root",
        default="Stream3D/outputs/audit/v105_v11_object_identity_witness_diag_20260711",
    )
    args = parser.parse_args()

    scene_id = str(args.scene_id)
    frames = _parse_ints(str(args.frames))
    crop = _parse_crop(str(args.crop))
    global_ids = _parse_ints(str(args.global_focus_ids))
    local_ids = _parse_ints(str(args.local_focus_ids))
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    rgb_dir = REPO_ROOT / "Stream3D/data/scannet/processed" / scene_id / "color"
    variants = _variant_paths(scene_id)
    records: list[dict[str, Any]] = []

    for frame_id in frames:
        rgb = _read_rgb(rgb_dir / f"{int(frame_id)}.jpg")
        for variant_name, mask_dir in variants.items():
            label = _read_label(mask_dir / f"{int(frame_id)}.png")
            focus_ids = local_ids if variant_name.endswith("_local") else global_ids
            crop_img, stats = _draw_crop(
                rgb=rgb,
                label=label,
                ids=focus_ids,
                crop=crop,
                title=f"{scene_id} frame={int(frame_id)} {variant_name}",
            )
            out_path = output_root / f"{scene_id}_boundary{str(args.boundary).replace('->', '')}_frame{int(frame_id)}_{variant_name}_rightcrop_ids.jpg"
            cv2.imwrite(str(out_path), crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            records.append(
                {
                    "scene_id": scene_id,
                    "boundary": str(args.boundary),
                    "frame_id": int(frame_id),
                    "variant": variant_name,
                    "mask_path": str(mask_dir / f"{int(frame_id)}.png"),
                    "rgb_path": str(rgb_dir / f"{int(frame_id)}.jpg"),
                    "crop_xyxy": list(crop),
                    "focus_ids": [int(v) for v in focus_ids],
                    "id_stats": stats,
                    "crop_debug_path": str(out_path),
                }
            )

    for variant_name in variants:
        imgs = []
        for frame_id in frames:
            p = output_root / f"{scene_id}_boundary{str(args.boundary).replace('->', '')}_frame{int(frame_id)}_{variant_name}_rightcrop_ids.jpg"
            imgs.append(_read_rgb(p))
        h = max(img.shape[0] for img in imgs)
        w = max(img.shape[1] for img in imgs)
        resized = [cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA) if img.shape[:2] != (h, w) else img for img in imgs]
        cv2.imwrite(
            str(output_root / f"{scene_id}_boundary{str(args.boundary).replace('->', '')}_{variant_name}_rightcrop_sheet.jpg"),
            np.concatenate(resized, axis=1),
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        )

    rows = []
    for frame_id in frames:
        imgs = []
        for variant_name in ["v6_global", "v9_global", "v10_global", "v10_local"]:
            p = output_root / f"{scene_id}_boundary{str(args.boundary).replace('->', '')}_frame{int(frame_id)}_{variant_name}_rightcrop_ids.jpg"
            imgs.append(_read_rgb(p))
        rows.append(np.concatenate(imgs, axis=1))
    comparison = np.concatenate(rows, axis=0)
    comparison_path = output_root / f"{scene_id}_boundary{str(args.boundary).replace('->', '')}_rightcrop_comparison_v6_v9_v10_local.jpg"
    cv2.imwrite(str(comparison_path), comparison, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    summary = {
        "schema_version": "stream4d_v105_object_identity_witness_diag_v1",
        "scene_id": scene_id,
        "boundary": str(args.boundary),
        "frames": [int(v) for v in frames],
        "crop_xyxy": list(crop),
        "records": records,
        "comparison_sheet": str(comparison_path),
        "claim_boundary": "Numeric/crop diagnostic only. It does not claim visual pass or method success.",
    }
    summary_path = output_root / "object_identity_witness_diag_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["schema_version", "scene_id", "boundary", "frames", "crop_xyxy", "comparison_sheet"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
