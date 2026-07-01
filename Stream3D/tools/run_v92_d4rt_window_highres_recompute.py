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

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.d4rt_adapter import D4RTAdapter
from stream4d.scannet_stream import ScanNetStream
from stream4d_native.sim3 import Sim3Transform
from tools.export_d4rt_grid_surfel_field_v8 import _grid_sources
from tools.run_v65_d4rt_stride_overlap_geometry import (
    ChunkRecord,
    load_window_without_masks,
    project,
    raw_valid_mask,
    resolve_repo,
    save_chunk_npz,
    write_csv,
    write_json,
)


DEFAULT_OUTPUT_ROOT = "outputs/audit/v92_phase3_d4rt_highres_recompute/HR1_grid12_local_window_safe"
DEFAULT_WINDOW_ROWS = "outputs/audit/v91_phase0_mv_ap_contract/window_support_rows.csv"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _window_frame_ids(row: dict[str, str], stride: int) -> list[int]:
    start = _int(row.get("frame_id_start"))
    end = _int(row.get("frame_id_end"))
    if end < start:
        return []
    return list(range(start, end + 1, max(1, int(stride))))


def _load_windows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = _read_csv(project(args.window_support_rows))
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("scene_id") != args.scene:
            continue
        if row.get("split", args.split) != args.split:
            continue
        frame_ids = _window_frame_ids(row, int(args.stride))
        if not frame_ids:
            continue
        out.append(
            {
                "scene_id": row.get("scene_id", ""),
                "split": row.get("split", args.split),
                "window_id": row.get("window_id", ""),
                "window_index": row.get("window_index", ""),
                "frame_id_start": int(frame_ids[0]),
                "frame_id_end": int(frame_ids[-1]),
                "frame_ids": frame_ids,
                "window_scoped_gt_count": row.get("window_scoped_gt_count", ""),
                "stream3d_pred_object_count": row.get("stream3d_pred_object_count", ""),
            }
        )
    out.sort(key=lambda item: (int(item["frame_id_start"]), int(item["frame_id_end"]), str(item["window_id"])))
    if int(args.max_windows) > 0:
        out = out[: int(args.max_windows)]
    return out


def _run_window(
    *,
    args: argparse.Namespace,
    adapter: D4RTAdapter,
    stream: ScanNetStream,
    window: dict[str, Any],
    window_index: int,
    out_root: Path,
) -> dict[str, Any]:
    started = time.time()
    frame_ids = [int(v) for v in window["frame_ids"]]
    model_frame_ids = frame_ids if len(frame_ids) >= 2 else [frame_ids[0], frame_ids[0]]
    data = load_window_without_masks(stream, model_frame_ids)
    sources, _source_diag = _grid_sources(
        masks=np.asarray(data["mask"]),
        frame_ids=model_frame_ids,
        grid_size=int(args.grid_size),
        grid_margin_ratio=float(args.grid_margin_ratio),
        mask_aware_min_points_per_mask=0,
        min_mask_area=1,
    )
    batch = adapter.infer_carriers(
        video_rgb_uint8=np.asarray(data["rgb"]),
        src_uv_norm=sources.src_uv,
        src_frame_local=sources.src_frame,
        carrier_id=sources.carrier_id,
        src_frame_global=sources.src_frame_global,
        src_xy=sources.src_xy,
        src_mask_id=sources.src_mask_id,
        query_chunk_size=int(args.query_chunk_size),
    )
    chunk = ChunkRecord(
        chunk_index=window_index,
        frame_ids=frame_ids,
        xyz=np.asarray(batch.xyz_ref, dtype=np.float32)[: len(frame_ids)],
        uv=np.asarray(batch.uv_pred, dtype=np.float32)[: len(frame_ids)],
        valid=np.asarray(batch.valid, dtype=bool)[: len(frame_ids)],
        visibility=np.asarray(batch.visibility_prob, dtype=np.float32)[: len(frame_ids)],
        confidence=np.asarray(batch.confidence_prob, dtype=np.float32)[: len(frame_ids)],
        carrier_id=np.asarray(batch.carrier_id, dtype=np.int64),
        src_frame_global=np.asarray(batch.src_frame_global, dtype=np.int64),
        src_xy=np.asarray(batch.src_xy, dtype=np.int64),
        transform_to_scene=Sim3Transform(scale=1.0, rot=np.eye(3, dtype=np.float64), trans=np.zeros(3, dtype=np.float64)),
    )
    window_dir = out_root / "windows"
    npz_path = window_dir / f"window_{str(window['window_id']).replace('/', '_')}.npz"
    save_chunk_npz(npz_path, chunk)
    valid_observation_count = int(np.count_nonzero(raw_valid_mask(chunk, args)))
    return {
        "scene": args.scene,
        "split": args.split,
        "stride": int(args.stride),
        "window_id": window["window_id"],
        "window_index": window["window_index"],
        "frame_start": int(frame_ids[0]),
        "frame_end": int(frame_ids[-1]),
        "frame_ids": ",".join(str(v) for v in frame_ids),
        "model_frame_ids": ",".join(str(v) for v in model_frame_ids),
        "num_frames": int(len(frame_ids)),
        "num_model_frames": int(len(model_frame_ids)),
        "grid_size": int(args.grid_size),
        "grid_points_per_frame": int(args.grid_size) * int(args.grid_size),
        "source_count": int(sources.carrier_id.shape[0]),
        "valid_observation_count": valid_observation_count,
        "min_visibility": float(args.min_visibility),
        "min_confidence": float(args.min_confidence),
        "window_npz": _rel(npz_path),
        "seconds": float(time.time() - started),
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "uses_rgbd_pose_mesh": False,
        "protocol_note": "D4RT forward input is restricted to the listed local-window frame_ids.",
        "single_frame_window_padding": len(frame_ids) == 1,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out_root = project(args.output_root) / args.scene / f"stride_{int(args.stride)}"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "last_command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8"
    )
    windows = _load_windows(args)
    if not windows:
        raise RuntimeError(f"No windows found for scene={args.scene} split={args.split}")
    stream = ScanNetStream(seq_name=args.scene, root=resolve_repo(args.scannet_root))
    errors = stream.validate(require_masks=False)
    if errors:
        raise RuntimeError("; ".join(errors))
    adapter = D4RTAdapter(
        d4rt_root=resolve_repo(args.d4rt_root),
        model_config=resolve_repo(args.d4rt_config),
        ckpt_path=resolve_repo(args.d4rt_ckpt),
        device=args.device,
    )
    rows: list[dict[str, Any]] = []
    for idx, window in enumerate(windows):
        print(
            f"[v92-window-hr] scene={args.scene} window={window['window_id']} "
            f"frames={window['frame_id_start']}..{window['frame_id_end']} "
            f"n={len(window['frame_ids'])} grid={args.grid_size}",
            flush=True,
        )
        rows.append(_run_window(args=args, adapter=adapter, stream=stream, window=window, window_index=idx, out_root=out_root))
    write_csv(out_root / "window_rows.csv", rows)
    summary = {
        "schema": "stream4d_v92_phase3_window_safe_d4rt_highres_recompute_v1",
        "phase": "v92_phase3_window_safe_d4rt_highres_recompute",
        "scene": args.scene,
        "split": args.split,
        "stride": int(args.stride),
        "grid_size": int(args.grid_size),
        "grid_points_per_frame": int(args.grid_size) * int(args.grid_size),
        "window_count": len(rows),
        "frame_count": len({frame for row in windows for frame in row["frame_ids"]}),
        "valid_observation_count": int(sum(_int(row.get("valid_observation_count")) for row in rows)),
        "window_support_rows": _rel(project(args.window_support_rows)),
        "d4rt_config": _rel(resolve_repo(args.d4rt_config)),
        "d4rt_config_sha256": _sha256(resolve_repo(args.d4rt_config)) if resolve_repo(args.d4rt_config).exists() else "",
        "d4rt_ckpt": _rel(resolve_repo(args.d4rt_ckpt)),
        "d4rt_ckpt_size_bytes": resolve_repo(args.d4rt_ckpt).stat().st_size if resolve_repo(args.d4rt_ckpt).exists() else 0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device": args.device,
        "query_chunk_size": int(args.query_chunk_size),
        "min_visibility": float(args.min_visibility),
        "min_confidence": float(args.min_confidence),
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "uses_rgbd_pose_mesh": False,
        "protocol_note": "Each D4RT forward call is scoped to one local MV_AP window from window_support_rows.",
        "duration_sec": float(time.time() - started),
        "outputs": {
            "window_rows_csv": _rel(out_root / "window_rows.csv"),
            "windows_dir": _rel(out_root / "windows"),
        },
    }
    write_json(out_root / "summary.json", summary)
    write_json(
        out_root / "SHA256SUMS.json",
        {
            _rel(path): _sha256(path)
            for path in sorted([out_root / "summary.json", out_root / "window_rows.csv", out_root / "last_command.txt"])
            if path.exists()
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run D4RT high-res recompute with one forward pass per local MV_AP window.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--window-support-rows", default=DEFAULT_WINDOW_ROWS)
    parser.add_argument("--d4rt-root", default="Open-d4rt")
    parser.add_argument("--d4rt-config", default="Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml")
    parser.add_argument("--d4rt-ckpt", default="Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scannet-root", default="Stream3D/data/scannet/processed")
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--grid-margin-ratio", type=float, default=0.02)
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--uv-radius", type=float, default=0.002)
    parser.add_argument("--max-windows", type=int, default=0)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
