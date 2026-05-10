#!/usr/bin/env python3
"""Convert standalone sparse masklet output into Pipeline v2 Stage C cache.

The standalone ``run_video_masklet_front_end.py`` runner writes a compact
full-sequence ``sparse_masklets_v1`` file. Pipeline v2 expects per-chunk
``chunk_xxx/masklet.pt`` cache entries with dense ``MaskletOutput`` fields.
This utility bridges those formats so a verified frontend run can be reused
without re-running SAM / detector / panoptic inference.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch


def _split_into_chunks(total_frames: int, chunk_size: int, overlap: int) -> List[Tuple[int, int]]:
    if chunk_size <= 0 or chunk_size >= total_frames:
        return [(0, total_frames)]
    step = max(int(chunk_size) - int(overlap), 1)
    chunks: List[Tuple[int, int]] = []
    for start in range(0, total_frames, step):
        end = min(start + int(chunk_size), total_frames)
        chunks.append((start, end))
        if end == total_frames:
            break
    return chunks


def _unpack_mask(packed: Any, height: int, width: int) -> torch.Tensor:
    if isinstance(packed, torch.Tensor):
        packed_np = packed.detach().cpu().numpy().astype(np.uint8, copy=False)
    else:
        packed_np = np.asarray(packed, dtype=np.uint8)
    flat = np.unpackbits(packed_np.reshape(-1), count=int(height) * int(width))
    return torch.from_numpy(flat.reshape(int(height), int(width)).astype(np.bool_))


def _normalise_track(track: Dict[str, Any]) -> Dict[str, Any]:
    frames = [int(x) for x in track.get("frames", [])]
    boxes = track.get("boxes", torch.zeros((0, 4), dtype=torch.float32))
    scores = track.get("scores", torch.zeros((0,), dtype=torch.float32))
    area = track.get("area_ratio", torch.zeros((0,), dtype=torch.float32))
    packed_masks = list(track.get("packed_masks", []))
    by_frame: Dict[int, Dict[str, Any]] = {}
    for idx, frame in enumerate(frames):
        by_frame[int(frame)] = {
            "packed_mask": packed_masks[idx],
            "box": boxes[idx].float() if idx < int(boxes.shape[0]) else torch.zeros(4, dtype=torch.float32),
            "score": float(scores[idx].item()) if idx < int(scores.shape[0]) else 1.0,
            "area": float(area[idx].item()) if idx < int(area.shape[0]) else 0.0,
        }
    out = dict(track)
    out["_by_frame"] = by_frame
    return out


def _chunk_cache_dict(
    tracks: List[Dict[str, Any]],
    *,
    start: int,
    end: int,
    chunk_idx: int,
    height: int,
    width: int,
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    active: List[Dict[str, Any]] = []
    for global_idx, track in enumerate(tracks):
        by_frame = track.get("_by_frame", {})
        if any(int(t) in by_frame for t in range(int(start), int(end))):
            item = dict(track)
            item["_global_track_idx"] = int(global_idx)
            active.append(item)

    J = len(active)
    T = int(end) - int(start)
    M_mask = torch.zeros((J, T, height, width), dtype=torch.bool)
    V_mask = torch.zeros((J, T), dtype=torch.bool)
    B_mask = torch.zeros((J, T, 4), dtype=torch.float32)
    Q_mask = torch.zeros((J, T), dtype=torch.float32)
    A_ratio = torch.zeros((J, T), dtype=torch.float32)
    L_sem: List[str] = []
    G_sem = torch.zeros((J,), dtype=torch.long)
    W_sem = torch.zeros((J,), dtype=torch.float32)
    source_type: List[str] = []
    birth_frame: List[int] = []
    seed_global_track_idx: List[int] = []

    for j, track in enumerate(active):
        L_sem.append(str(track.get("L_sem", "unknown")))
        G_sem[j] = int(track.get("G_sem", 4))
        W_sem[j] = float(track.get("W_sem", 0.15))
        source_type.append(str(track.get("source_type", "stuff_static")))
        global_birth = int(track.get("birth_frame", start))
        birth_frame.append(max(0, min(T - 1, global_birth - int(start))) if T > 0 else 0)
        seed_global_track_idx.append(int(track.get("_global_track_idx", j)))
        by_frame = track.get("_by_frame", {})
        for global_t in range(int(start), int(end)):
            rec = by_frame.get(int(global_t))
            if rec is None:
                continue
            local_t = int(global_t) - int(start)
            mask = _unpack_mask(rec["packed_mask"], height, width)
            if not bool(mask.any().item()):
                continue
            M_mask[j, local_t] = mask
            V_mask[j, local_t] = True
            B_mask[j, local_t] = rec["box"].float()
            Q_mask[j, local_t] = float(rec["score"])
            A_ratio[j, local_t] = float(rec["area"])

    return {
        "schema_version": 1,
        "manifest": manifest,
        "M_mask": M_mask,
        "V_mask": V_mask,
        "B_mask": B_mask,
        "Q_mask": Q_mask,
        "L_sem": L_sem,
        "G_sem": G_sem,
        "W_sem": W_sem,
        "A_ratio": A_ratio,
        "num_masklets": int(J),
        "num_frames": int(T),
        "frame_height": int(height),
        "frame_width": int(width),
        "source_type": source_type,
        "birth_frame": birth_frame,
        "seed_global_track_idx": seed_global_track_idx,
        "debug": {
            "converted_from_sparse_masklets_v1": True,
            "source_global_track_indices": seed_global_track_idx,
            "visible_masklet_frames": int(V_mask.sum().item()),
        },
    }


def _write_chunk(cache_root: Path, chunk_name: str, payload: Dict[str, Any], manifest: Dict[str, Any]) -> None:
    chunk_dir = cache_root / chunk_name
    tmp_dir = cache_root / f"{chunk_name}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, tmp_dir / "masklet.pt")
    (tmp_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    tmp_dir.rename(chunk_dir)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--chunk_overlap", type=int, default=3)
    parser.add_argument("--stage_c_mode", default="reference")
    parser.add_argument("--sam_backend", default="sam31_multiplex")
    parser.add_argument("--detector", default="yoloe")
    parser.add_argument("--tag", default="mask2former_cityscapes_full")
    parser.add_argument("--overwrite", type=int, default=0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    input_pt = Path(args.input_pt)
    cache_root = Path(args.cache_dir)
    if cache_root.exists() and any(cache_root.iterdir()) and not bool(args.overwrite):
        raise SystemExit(f"Refusing to overwrite non-empty cache dir: {cache_root}")
    cache_root.mkdir(parents=True, exist_ok=True)

    data = torch.load(input_pt, map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or data.get("format") != "sparse_masklets_v1":
        raise SystemExit(f"Expected sparse_masklets_v1 dict, got {type(data)} format={data.get('format') if isinstance(data, dict) else None}")

    height = int(data["frame_height"])
    width = int(data["frame_width"])
    total_frames = int(data["num_frames"])
    tracks = [_normalise_track(dict(track)) for track in data.get("tracks", [])]
    chunks = _split_into_chunks(total_frames, int(args.chunk_size), int(args.chunk_overlap))

    index_rows: List[Dict[str, Any]] = []
    for chunk_idx, (start, end) in enumerate(chunks):
        manifest = {
            "schema_version": 1,
            "chunk_idx": int(chunk_idx),
            "start_frame": int(start),
            "end_frame": int(end),
            "chunk_shape": [int(end) - int(start), 3, height, width],
            "stage_c_mode": str(args.stage_c_mode),
            "sam_backend": str(args.sam_backend),
            "detector": str(args.detector),
            "source_sparse_pt": str(input_pt),
            "source_format": "sparse_masklets_v1",
            "conversion_tag": str(args.tag),
        }
        payload = _chunk_cache_dict(
            tracks,
            start=start,
            end=end,
            chunk_idx=chunk_idx,
            height=height,
            width=width,
            manifest=manifest,
        )
        chunk_name = f"chunk_{chunk_idx:03d}_{int(start):06d}_{int(end):06d}"
        _write_chunk(cache_root, chunk_name, payload, manifest)
        row = {
            "chunk": chunk_name,
            "chunk_idx": int(chunk_idx),
            "start_frame": int(start),
            "end_frame": int(end),
            "num_masklets": int(payload["num_masklets"]),
            "visible_masklet_frames": int(payload["debug"]["visible_masklet_frames"]),
        }
        index_rows.append(row)
        print(
            f"{chunk_name}: J={row['num_masklets']} "
            f"visible={row['visible_masklet_frames']}",
            flush=True,
        )

    with (cache_root / "cache_index.jsonl").open("w", encoding="utf-8") as f:
        for row in index_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "input_pt": str(input_pt),
        "cache_dir": str(cache_root),
        "num_chunks": len(chunks),
        "num_frames": total_frames,
        "frame_height": height,
        "frame_width": width,
        "source_num_masklets": int(data.get("num_masklets", len(tracks))),
        "chunk_size": int(args.chunk_size),
        "chunk_overlap": int(args.chunk_overlap),
    }
    (cache_root / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote {len(chunks)} chunks to {cache_root}")


if __name__ == "__main__":
    main()
