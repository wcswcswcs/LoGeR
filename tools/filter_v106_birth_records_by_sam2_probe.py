#!/usr/bin/env python3
"""Filter v106 birth records with a short non-oracle SAM2 persistence probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
STREAM3D_ROOT = REPO_ROOT / "Stream3D"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "v105" / "baseline_chunk_table" / "baseline_x_gapadaptive_sam2.generated.yaml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(1, str(STREAM3D_ROOT))

from tools.audit_v105_baseline_x_sam2_twostage_tracking import (  # noqa: E402
    load_config,
    make_args,
    propagate_new_masks_chunked,
    setup_models,
)
from tools.audit_v105_4dpm_largest_tracking_baseline import make_numeric_frame_dir  # noqa: E402
from tools.audit_v105_4dpm_style_per_frame_segmentors import parse_frame_ids, read_rgb  # noqa: E402


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(raw_path: str | Path, *, base: Path = REPO_ROOT) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return base / path


def _load_mask(path: Path, h: int, w: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    mask = image > 0
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask.astype(bool)


def _make_baseline_args(config_path: Path, cli: argparse.Namespace) -> SimpleNamespace:
    config = load_config(config_path)
    baseline_cli = SimpleNamespace(
        config=str(config_path),
        scene_id=cli.scene_id,
        rgb_root=cli.rgb_root,
        frame_start=cli.frame_start,
        frame_stride=cli.frame_stride,
        frame_count=cli.frame_count,
        frame_ids=cli.frame_ids,
        output_root=cli.output_root,
        seed=cli.seed,
        birth_dump_dir="",
    )
    args = make_args(config, baseline_cli)
    args.output_root = str(cli.output_root)
    return args


def _row_considered(row: Dict[str, Any], cli: argparse.Namespace) -> bool:
    if str(row.get("phase5_role")) != "birth_new":
        return False
    if int(row.get("chunk_frame_index", -1)) != int(cli.anchor_chunk_index):
        return False
    if cli.protect_frame0_parent_original and str(row.get("frame0_child_split_role")) in {
        "parent_original",
        "parent_original_no_child_fallback",
    }:
        return False
    return True


def _load_frame_ids(args: SimpleNamespace) -> List[int]:
    return parse_frame_ids(
        str(args.frame_ids),
        int(args.frame_start),
        int(args.frame_stride),
        int(args.frame_count),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a short SAM2 propagation probe on selected birth_new masks and "
            "drop masks that do not persist for enough predicted frames. This "
            "does not use reference labels or a full preliminary replay."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--birth-records", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--rgb-root", default=str(STREAM3D_ROOT / "data" / "scannet" / "processed"))
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--frame-count", type=int, default=32)
    parser.add_argument("--frame-ids", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--anchor-chunk-index", type=int, default=0)
    parser.add_argument("--probe-frame-count", type=int, default=8)
    parser.add_argument("--min-present-frames", type=int, required=True)
    parser.add_argument("--min-probe-mask-area", type=int, default=1)
    parser.add_argument("--protect-frame0-parent-original", action="store_true", default=False)
    parser.add_argument("--propagation-chunk-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    import torch

    output_root = _resolve(cli.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = _resolve(cli.config)
    birth_path = _resolve(cli.birth_records)
    output_path = _resolve(cli.output)

    args = _make_baseline_args(config_path, cli)
    if cli.propagation_chunk_size is not None:
        args.propagation_chunk_size = int(cli.propagation_chunk_size)
    frame_ids = _load_frame_ids(args)
    if int(cli.anchor_chunk_index) < 0 or int(cli.anchor_chunk_index) >= len(frame_ids):
        raise ValueError(f"anchor_chunk_index out of range: {cli.anchor_chunk_index}")
    probe_end = min(len(frame_ids), int(cli.anchor_chunk_index) + int(cli.probe_frame_count))
    probe_frame_ids = frame_ids[int(cli.anchor_chunk_index):probe_end]
    if len(probe_frame_ids) < 2:
        raise ValueError("probe_frame_count must include at least one future frame")

    rgb_root = _resolve(args.rgb_root) / str(args.scene_id) / "color"
    probe_frame_paths = [rgb_root / f"{int(frame_id)}.jpg" for frame_id in probe_frame_ids]
    missing = [str(path) for path in probe_frame_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing[:5])
    first_rgb = read_rgb(probe_frame_paths[0])
    h, w = first_rgb.shape[:2]

    birth = _read_json(birth_path)
    payload_frame_ids = [int(v) for v in birth.get("frame_ids", [])]
    if payload_frame_ids and payload_frame_ids != [int(v) for v in frame_ids]:
        raise ValueError("birth record frame_ids do not match requested frame_ids")

    rows = [dict(row) for row in birth.get("rows", [])]
    candidates = [row for row in rows if _row_considered(row, cli)]

    probe_records: List[Dict[str, Any]] = []
    dropped_ids: set[int] = set()
    probe_runtime_sec = 0.0
    setup_sec = 0.0
    peak_cuda_memory_mb = 0.0
    group_record: Dict[str, Any] = {}
    if candidates:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        setup_t0 = time.time()
        models = setup_models(args)
        setup_sec = time.time() - setup_t0
        tracker_model = models["tracker_model"]
        video_dir = make_numeric_frame_dir(probe_frame_paths, output_root / "probe_video")
        obj_ids = np.asarray([int(row["obj_id"]) for row in candidates], dtype=np.int64)
        masks = np.stack([
            _load_mask(_resolve(row["mask_path"]), h, w)
            for row in candidates
        ], axis=0).astype(bool)
        for obj_id, mask, row in zip(obj_ids.tolist(), masks, candidates, strict=False):
            area = int(np.count_nonzero(mask))
            probe_records.append(
                {
                    "obj_id": int(obj_id),
                    "frame_id": int(row.get("frame_id", -1)),
                    "chunk_frame_index": int(row.get("chunk_frame_index", -1)),
                    "frame0_child_split_role": row.get("frame0_child_split_role"),
                    "anchor_mask_area": int(area),
                    "present_frame_ids": [int(probe_frame_ids[0])] if area >= int(cli.min_probe_mask_area) else [],
                    "areas": [int(area)] if area >= int(cli.min_probe_mask_area) else [],
                }
            )

        chunk_runtime_records: List[Dict[str, Any]] = []
        probe_t0 = time.time()
        propagated = propagate_new_masks_chunked(
            tracker_model,
            tracker=str(args.tracker_backend),
            video_dir=video_dir,
            seed_frame=0,
            obj_ids=obj_ids,
            masks=masks,
            total_frames=len(probe_frame_ids),
            offload_video_to_cpu=bool(args.offload_video_to_cpu),
            offload_state_to_cpu=bool(args.offload_state_to_cpu),
            chunk_size=int(args.propagation_chunk_size),
            chunk_runtime_records=chunk_runtime_records,
        )
        probe_runtime_sec = time.time() - probe_t0
        peak_cuda_memory_mb = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))

        by_obj = {int(record["obj_id"]): record for record in probe_records}
        for local_frame_idx, frame_outputs in propagated.items():
            if int(local_frame_idx) <= 0:
                continue
            if int(local_frame_idx) >= len(probe_frame_ids):
                continue
            frame_id = int(probe_frame_ids[int(local_frame_idx)])
            for obj_id, mask in frame_outputs.items():
                area = int(np.count_nonzero(mask))
                if area < int(cli.min_probe_mask_area):
                    continue
                record = by_obj.get(int(obj_id))
                if record is None:
                    continue
                record["present_frame_ids"].append(frame_id)
                record["areas"].append(int(area))

        for record in probe_records:
            areas = [int(v) for v in record.get("areas", [])]
            present = [int(v) for v in record.get("present_frame_ids", [])]
            record["present_frame_count"] = int(len(present))
            record["first_present_frame"] = int(present[0]) if present else None
            record["last_present_frame"] = int(present[-1]) if present else None
            record["mean_area"] = float(np.mean(areas)) if areas else 0.0
            record["max_area"] = int(max(areas)) if areas else 0
            record["dropped"] = bool(int(record["present_frame_count"]) < int(cli.min_present_frames))
            record["reason"] = "present_frame_count_lt_min" if record["dropped"] else "kept"
            if record["dropped"]:
                dropped_ids.add(int(record["obj_id"]))

        group_record = {
            "anchor_chunk_index": int(cli.anchor_chunk_index),
            "anchor_frame_id": int(probe_frame_ids[0]),
            "candidate_count": int(len(candidates)),
            "probe_frame_count": int(len(probe_frame_ids)),
            "probe_frame_ids": [int(v) for v in probe_frame_ids],
            "setup_sec": float(setup_sec),
            "probe_runtime_sec": float(probe_runtime_sec),
            "peak_cuda_memory_mb": float(peak_cuda_memory_mb),
            "chunk_runtime_records": chunk_runtime_records,
        }

    probe_by_obj = {int(record["obj_id"]): record for record in probe_records}
    filter_records = []
    kept_rows = []
    for row in rows:
        obj_id = int(row["obj_id"])
        considered = _row_considered(row, cli)
        probe = probe_by_obj.get(obj_id)
        drop = bool(considered and obj_id in dropped_ids)
        filter_records.append(
            {
                "obj_id": int(obj_id),
                "frame_id": int(row.get("frame_id", -1)),
                "chunk_frame_index": int(row.get("chunk_frame_index", -1)),
                "phase5_role": str(row.get("phase5_role")),
                "frame0_child_split_role": row.get("frame0_child_split_role"),
                "considered": bool(considered),
                "dropped": bool(drop),
                "probe": probe,
            }
        )
        if not drop:
            kept_rows.append(row)

    out = dict(birth)
    out["schema_version"] = str(birth.get("schema_version", "unknown")) + "+sam2_probe_filter_v1"
    out["source_birth_records"] = str(birth_path)
    out["source_birth_records_sha256"] = _sha256_file(birth_path)
    out["sam2_probe_filter_policy"] = {
        "uses_reference_labels": False,
        "uses_preliminary_full_replay": False,
        "anchor_chunk_index": int(cli.anchor_chunk_index),
        "probe_frame_count": int(cli.probe_frame_count),
        "probe_frame_ids": [int(v) for v in probe_frame_ids],
        "min_present_frames": int(cli.min_present_frames),
        "min_probe_mask_area": int(cli.min_probe_mask_area),
        "protect_frame0_parent_original": bool(cli.protect_frame0_parent_original),
        "config": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "note": "Short online-style SAM2 persistence probe for tentative/defer control.",
    }
    out["sam2_probe_filter_runtime"] = group_record
    out["sam2_probe_filter_records"] = filter_records
    out["sam2_probe_filter_dropped_obj_ids"] = [int(v) for v in sorted(dropped_ids)]
    out["sam2_probe_filter_dropped_count"] = int(len(dropped_ids))
    out["filtered_from_row_count"] = int(len(rows))
    out["row_count"] = int(len(kept_rows))
    out["rows"] = kept_rows

    _write_json(output_path, out)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "row_count_before": int(len(rows)),
                "row_count_after": int(len(kept_rows)),
                "dropped_obj_ids": [int(v) for v in sorted(dropped_ids)],
                "setup_sec": float(setup_sec),
                "probe_runtime_sec": float(probe_runtime_sec),
                "peak_cuda_memory_mb": float(peak_cuda_memory_mb),
                "output_sha256": _sha256_file(output_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
