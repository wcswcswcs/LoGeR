#!/usr/bin/env python3
"""Build v98.1 same-scene temporal holdout source-container rows."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_v89_recalc_point_projected_mv_ap as recalc  # noqa: E402


DEFAULT_ADAPTER_ROWS = ROOT / "outputs/audit/v84_holdout_replay_v82_local_shadow/phase1_adapter_v84_holdout_replay/adapter_rows.csv"
DEFAULT_OUT = ROOT / "outputs/audit/v98_phase13_holdout"
DEV_LAST_FRAME = {"scene0011_00": 880, "scene0050_00": 590}


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _int(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _load_feature_keys(path: Path) -> set[tuple[str, int, int]]:
    if not path.exists():
        return set()
    payload = np.load(path, allow_pickle=True)
    return {
        (str(scene), int(frame), int(mask))
        for scene, frame, mask in zip(payload["scene_id"], payload["frame_id"], payload["mask_id"])
    }


def _mask_area_ratio(mask_path: Path, mask_id: int) -> tuple[float, int, int]:
    image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return 0.0, 0, 0
    if image.ndim == 3:
        image = image[..., 0]
    area = int(np.count_nonzero(image == int(mask_id)))
    total = int(image.shape[0] * image.shape[1])
    return (float(area / max(1, total)), area, total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-rows", default=str(DEFAULT_ADAPTER_ROWS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--feature-store-npz", default="")
    args = parser.parse_args()

    adapter_rows = _project(args.adapter_rows)
    out = _project(args.output_root)
    feature_keys = _load_feature_keys(_project(args.feature_store_npz)) if args.feature_store_npz else set()
    seen: set[tuple[str, int, int, int]] = set()
    rows: list[dict[str, Any]] = []
    missing_mask_rows: list[dict[str, Any]] = []
    with adapter_rows.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scene = row.get("scene_id", "")
            frame = _int(row.get("frame_id"), -1)
            mask_id = _int(row.get("mask_id"), -1)
            chunk_id = _int(row.get("chunk_id"), -1)
            if scene not in DEV_LAST_FRAME or frame <= DEV_LAST_FRAME[scene] or mask_id <= 0 or chunk_id < 0:
                continue
            key = (scene, chunk_id, frame, mask_id)
            if key in seen:
                continue
            seen.add(key)
            mask_path = recalc._mask_dir(scene) / f"{frame}.png"
            if not mask_path.exists():
                missing_mask_rows.append({"scene_id": scene, "frame_id": frame, "mask_id": mask_id, "mask_path": _rel(mask_path)})
                continue
            area_ratio, area_pixels, total_pixels = _mask_area_ratio(mask_path, mask_id)
            rows.append(
                {
                    "schema_version": "stream4d_v98_1_holdout_source_container_v1",
                    "split": "holdout",
                    "scene_id": scene,
                    "window_id": f"h_c{chunk_id:02d}",
                    "chunk_id": chunk_id,
                    "frame_id": frame,
                    "source_mask_id": mask_id,
                    "mask_path": _rel(mask_path),
                    "mask_area_ratio": area_ratio,
                    "mask_area_pixels": area_pixels,
                    "mask_total_pixels": total_pixels,
                    "has_region_feature": (scene, frame, mask_id) in feature_keys,
                    "source_universe": "v84_holdout_replay_v82_local_shadow_adapter_rows",
                    "support_policy": "same_scene_temporal_holdout_later_windows",
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )

    by_scene: dict[str, set[int]] = {}
    for row in rows:
        by_scene.setdefault(str(row["scene_id"]), set()).add(int(row["frame_id"]))
    summary = {
        "schema": "stream4d_v98_1_holdout_source_registry_summary_v1",
        "created_at": _created_at(),
        "adapter_rows": _rel(adapter_rows),
        "output_root": _rel(out),
        "source_container_row_count": len(rows),
        "missing_mask_row_count": len(missing_mask_rows),
        "feature_store_npz": _rel(_project(args.feature_store_npz)) if args.feature_store_npz else "",
        "feature_marked_row_count": int(sum(1 for row in rows if row["has_region_feature"])),
        "scene_frame_counts": {scene: len(frames) for scene, frames in sorted(by_scene.items())},
        "scene_frame_minmax": {scene: [min(frames), max(frames)] for scene, frames in sorted(by_scene.items()) if frames},
        "dev_last_frame": DEV_LAST_FRAME,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_csv(out / "source_container_rows.csv", rows)
    _write_csv(out / "missing_mask_rows.csv", missing_mask_rows)
    _write_json(out / "source_registry_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
