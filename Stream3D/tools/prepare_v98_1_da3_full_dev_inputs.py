#!/usr/bin/env python3
"""Prepare DA3-Streaming full-dev image inputs from the v95 source universe."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
DEFAULT_SOURCE_ROWS = ROOT / "outputs/audit/v95_phase1_physical_source_registry/source_container_rows.csv"
DEFAULT_OUT_BASE = ROOT / "outputs/audit/v98_phase1_provider_contract"


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in {"Stream3D", "docs", "Open-d4rt"}:
        return REPO_ROOT / p
    return ROOT / p


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return path.as_posix()


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_frame_ids(source_rows: Path, scene_id: str, split: str) -> list[int]:
    frames: set[int] = set()
    with source_rows.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("scene_id") != scene_id:
                continue
            if split and row.get("split") != split:
                continue
            try:
                frames.add(int(row.get("frame_id", "")))
            except ValueError:
                continue
    return sorted(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--source-rows", default=str(DEFAULT_SOURCE_ROWS))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--rgb-root", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_rows = _project(args.source_rows)
    if args.output_dir:
        output_dir = _project(args.output_dir)
    else:
        output_dir = DEFAULT_OUT_BASE / f"da3_streaming_full_{args.scene_id}_input"
    rgb_root = _project(args.rgb_root) if args.rgb_root else ROOT / "data/scannet/processed" / args.scene_id / "color"

    frame_ids = _load_frame_ids(source_rows, args.scene_id, args.split)
    if not frame_ids:
        raise RuntimeError(f"no frames found for scene_id={args.scene_id!r}, split={args.split!r} in {source_rows}")
    if not rgb_root.exists():
        raise FileNotFoundError(f"rgb root does not exist: {rgb_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_images = sorted(output_dir.glob("*.jpg")) + sorted(output_dir.glob("*.png"))
    if existing_images and not args.overwrite:
        raise RuntimeError(f"output dir already contains images: {output_dir}; pass --overwrite")
    for item in existing_images:
        if item.is_symlink() or item.is_file():
            item.unlink()

    rows: list[dict[str, Any]] = []
    missing: list[int] = []
    for da3_frame_index, frame_id in enumerate(frame_ids):
        src = rgb_root / f"{frame_id}.jpg"
        if not src.exists():
            missing.append(frame_id)
            continue
        dst = output_dir / f"{da3_frame_index:06d}_fid{frame_id:06d}.jpg"
        dst.symlink_to(src.resolve())
        rows.append(
            {
                "scene_id": args.scene_id,
                "split": args.split,
                "da3_frame_index": da3_frame_index,
                "frame_id": frame_id,
                "image_name": dst.name,
                "source_rgb": _rel(src),
                "symlink_path": _rel(dst),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )

    if missing:
        raise FileNotFoundError(f"missing {len(missing)} RGB frames for {args.scene_id}: {missing[:10]}")

    manifest = output_dir / "frame_manifest_rows.csv"
    _write_csv(
        manifest,
        rows,
        [
            "scene_id",
            "split",
            "da3_frame_index",
            "frame_id",
            "image_name",
            "source_rgb",
            "symlink_path",
            "uses_gt_for_prediction",
            "uses_future",
        ],
    )
    summary = {
        "created_at": _created_at(),
        "scene_id": args.scene_id,
        "split": args.split,
        "frame_count": len(rows),
        "frame_id_min": min(frame_ids),
        "frame_id_max": max(frame_ids),
        "source_rows": _rel(source_rows),
        "rgb_root": _rel(rgb_root),
        "input_dir": _rel(output_dir),
        "frame_manifest_rows": _rel(manifest),
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
