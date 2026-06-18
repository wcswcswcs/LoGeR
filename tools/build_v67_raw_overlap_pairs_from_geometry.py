#!/usr/bin/env python3
"""Materialize ACL2 v67 raw overlap point-pair artifacts from per-chunk geometry.

The pipeline currently saves per-chunk duplicate pointmaps but not the requested
``overlap_pairs/chunk_PREV_CURR.pt`` files. This posthoc materializer creates
auditable sampled point pairs from adjacent chunk overlap frames and projects the
v67 dense semantic label/confidence maps onto the saved geometry resolution.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def _load_pt(path: Path) -> Dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected dict payload, got {type(obj).__name__}")
    return obj


def _project_nearest(x: torch.Tensor, size: Tuple[int, int], *, is_label: bool) -> torch.Tensor:
    y = F.interpolate(x[:, None].float(), size=size, mode="nearest").squeeze(1)
    return y.long() if is_label else y.float().clamp(0.0, 1.0)


def _frame_semantics(
    label_maps: torch.Tensor,
    conf_maps: Optional[torch.Tensor],
    frame_ids: Sequence[int],
    size: Tuple[int, int],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    frames = torch.as_tensor([int(f) for f in frame_ids], dtype=torch.long)
    if frames.numel() == 0:
        return torch.empty((0, *size), dtype=torch.long), None
    if int(frames.min()) < 0 or int(frames.max()) >= int(label_maps.shape[0]):
        raise IndexError(
            f"semantic frame range [{int(frames.min())}, {int(frames.max())}] outside label_maps T={label_maps.shape[0]}"
        )
    labels = _project_nearest(label_maps[frames].cpu(), size, is_label=True)
    conf = None
    if conf_maps is not None:
        conf = _project_nearest(conf_maps[frames].cpu(), size, is_label=False)
    return labels, conf


def _chunk_paths(geometry_dir: Path) -> List[Path]:
    return sorted(geometry_dir.glob("chunk_*.pt"))


def _chunk_id(path: Path, payload: Dict[str, Any]) -> int:
    if "chunk_idx" in payload:
        return int(payload["chunk_idx"])
    return int(path.stem.split("_")[-1])


def _tensor(payload: Dict[str, Any], *keys: str) -> Optional[torch.Tensor]:
    for key in keys:
        value = payload.get(key)
        if torch.is_tensor(value):
            return value.detach().cpu()
    return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _median(values: Iterable[Any]) -> Optional[float]:
    xs = sorted(float(v) for v in values if _safe_float(v) is not None)
    if not xs:
        return None
    mid = len(xs) // 2
    if len(xs) % 2:
        return xs[mid]
    return float((xs[mid - 1] + xs[mid]) / 2.0)


def _sample_frame_pairs(
    *,
    prev_points: torch.Tensor,
    curr_points: torch.Tensor,
    prev_local: Optional[torch.Tensor],
    curr_local: Optional[torch.Tensor],
    prev_conf: torch.Tensor,
    curr_conf: torch.Tensor,
    labels: torch.Tensor,
    sem_conf: Optional[torch.Tensor],
    frame_id: int,
    max_pairs: int,
    min_conf: float,
    sample_policy: str,
) -> Dict[str, torch.Tensor]:
    height, width = int(prev_conf.shape[-2]), int(prev_conf.shape[-1])
    valid = (
        torch.isfinite(prev_points).all(dim=-1)
        & torch.isfinite(curr_points).all(dim=-1)
        & torch.isfinite(prev_conf)
        & torch.isfinite(curr_conf)
        & (prev_conf >= float(min_conf))
        & (curr_conf >= float(min_conf))
    )
    flat_valid = valid.reshape(-1)
    valid_count = int(flat_valid.sum().item())
    if valid_count <= 0:
        empty_i = torch.empty((0,), dtype=torch.long)
        empty_p = torch.empty((0, 3), dtype=torch.float32)
        return {
            "indices": empty_i,
            "pixel_coords": torch.empty((0, 2), dtype=torch.int32),
            "frame_ids": empty_i,
            "prev_points": empty_p,
            "curr_points": empty_p,
            "prev_local_points": empty_p,
            "curr_local_points": empty_p,
            "prev_conf": torch.empty((0,), dtype=torch.float32),
            "curr_conf": torch.empty((0,), dtype=torch.float32),
            "labels": empty_i,
            "semantic_conf": torch.empty((0,), dtype=torch.float32),
            "valid_count": torch.tensor(0, dtype=torch.long),
        }
    conf_score = torch.minimum(prev_conf, curr_conf).reshape(-1)
    residual_score = torch.linalg.norm(
        (prev_points - curr_points).reshape(-1, 3),
        dim=1,
    )
    if sample_policy == "top_conf":
        score = conf_score
    elif sample_policy == "top_residual":
        score = residual_score
    elif sample_policy == "top_residual_conf_product":
        score = residual_score * conf_score
    else:
        raise ValueError(f"unknown sample_policy={sample_policy!r}")
    score = torch.where(flat_valid, score, torch.full_like(score, -1.0))
    k = min(int(max_pairs), valid_count) if int(max_pairs) > 0 else valid_count
    _, idx = torch.topk(score, k=k, largest=True, sorted=False)
    y = torch.div(idx, width, rounding_mode="floor")
    x = idx - y * width
    coords = torch.stack([y, x], dim=1).to(torch.int32)
    prev_flat = prev_points.reshape(-1, 3)
    curr_flat = curr_points.reshape(-1, 3)
    prev_local_flat = prev_local.reshape(-1, 3) if prev_local is not None else prev_flat
    curr_local_flat = curr_local.reshape(-1, 3) if curr_local is not None else curr_flat
    labels_flat = labels.reshape(-1)
    if sem_conf is not None:
        sem_conf_flat = sem_conf.reshape(-1).float()
        sem_conf_out = sem_conf_flat[idx].to(torch.float32)
    else:
        sem_conf_out = torch.empty((0,), dtype=torch.float32)
    return {
        "indices": idx.to(torch.long),
        "pixel_coords": coords,
        "frame_ids": torch.full((idx.numel(),), int(frame_id), dtype=torch.long),
        "prev_points": prev_flat[idx].to(torch.float32),
        "curr_points": curr_flat[idx].to(torch.float32),
        "prev_local_points": prev_local_flat[idx].to(torch.float32),
        "curr_local_points": curr_local_flat[idx].to(torch.float32),
        "prev_conf": prev_conf.reshape(-1)[idx].to(torch.float32),
        "curr_conf": curr_conf.reshape(-1)[idx].to(torch.float32),
        "labels": labels_flat[idx].to(torch.long),
        "semantic_conf": sem_conf_out,
        "valid_count": torch.tensor(valid_count, dtype=torch.long),
    }


def _concat(parts: Sequence[Dict[str, torch.Tensor]], key: str, *, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    xs = [part[key] for part in parts if part[key].numel() > 0]
    if xs:
        return torch.cat(xs, dim=0)
    return torch.empty(shape, dtype=dtype)


def _write_pair_file(
    *,
    out_path: Path,
    prev_chunk: int,
    curr_chunk: int,
    prev_payload: Dict[str, Any],
    curr_payload: Dict[str, Any],
    label_maps: torch.Tensor,
    conf_maps: Optional[torch.Tensor],
    label_names: Sequence[str],
    max_pairs_per_frame: int,
    min_conf: float,
    sample_policy: str,
    semantic_full_pt: Path,
    geometry_dir: Path,
) -> Dict[str, Any]:
    prev_points = _tensor(prev_payload, "points", "world_points")
    curr_points = _tensor(curr_payload, "points", "world_points")
    prev_local = _tensor(prev_payload, "local_points")
    curr_local = _tensor(curr_payload, "local_points")
    prev_conf = _tensor(prev_payload, "conf", "confidence")
    curr_conf = _tensor(curr_payload, "conf", "confidence")
    if prev_points is None or curr_points is None or prev_conf is None or curr_conf is None:
        raise ValueError(f"chunks {prev_chunk}->{curr_chunk}: missing points/conf tensors")
    prev_points = prev_points.float()
    curr_points = curr_points.float()
    prev_local = prev_local.float() if prev_local is not None else None
    curr_local = curr_local.float() if curr_local is not None else None
    prev_conf = prev_conf.float().clamp(0.0, 1.0)
    curr_conf = curr_conf.float().clamp(0.0, 1.0)

    prev_start, prev_end = int(prev_payload["start_frame"]), int(prev_payload["end_frame"])
    curr_start, curr_end = int(curr_payload["start_frame"]), int(curr_payload["end_frame"])
    overlap_start = max(prev_start, curr_start)
    overlap_end = min(prev_end, curr_end)
    if overlap_end <= overlap_start:
        return {
            "prev_chunk": int(prev_chunk),
            "curr_chunk": int(curr_chunk),
            "overlap_frame_count": 0,
            "saved_pair_count": 0,
            "valid_pair_count": 0,
            "written": False,
            "reason": "no_frame_overlap",
        }
    height, width = int(prev_points.shape[-3]), int(prev_points.shape[-2])
    if tuple(curr_points.shape[-3:-1]) != (height, width):
        raise ValueError(
            f"chunks {prev_chunk}->{curr_chunk}: geometry size mismatch "
            f"{tuple(prev_points.shape)} vs {tuple(curr_points.shape)}"
        )
    frame_ids = list(range(overlap_start, overlap_end))
    labels_proj, sem_conf_proj = _frame_semantics(label_maps, conf_maps, frame_ids, (height, width))
    parts: List[Dict[str, torch.Tensor]] = []
    valid_counts: List[int] = []
    for local_i, frame_id in enumerate(frame_ids):
        prev_i = int(frame_id - prev_start)
        curr_i = int(frame_id - curr_start)
        part = _sample_frame_pairs(
            prev_points=prev_points[prev_i],
            curr_points=curr_points[curr_i],
            prev_local=prev_local[prev_i] if prev_local is not None else None,
            curr_local=curr_local[curr_i] if curr_local is not None else None,
            prev_conf=prev_conf[prev_i],
            curr_conf=curr_conf[curr_i],
            labels=labels_proj[local_i],
            sem_conf=sem_conf_proj[local_i] if sem_conf_proj is not None else None,
            frame_id=int(frame_id),
            max_pairs=int(max_pairs_per_frame),
            min_conf=float(min_conf),
            sample_policy=str(sample_policy),
        )
        valid_counts.append(int(part["valid_count"].item()))
        parts.append(part)

    prev_overlap_points = _concat(parts, "prev_points", shape=(0, 3), dtype=torch.float32)
    curr_overlap_points = _concat(parts, "curr_points", shape=(0, 3), dtype=torch.float32)
    labels = _concat(parts, "labels", shape=(0,), dtype=torch.long)
    sem_conf_out = _concat(parts, "semantic_conf", shape=(0,), dtype=torch.float32)
    saved = int(prev_overlap_points.shape[0])
    label_valid = (labels >= 0) & (labels < int(len(label_names))) if labels.numel() else torch.empty((0,), dtype=torch.bool)
    nonvoid = labels != 0 if labels.numel() else torch.empty((0,), dtype=torch.bool)
    residual = torch.linalg.norm(prev_overlap_points - curr_overlap_points, dim=1) if saved else torch.empty((0,), dtype=torch.float32)
    payload: Dict[str, Any] = {
        "schema": "acl2_v67_raw_overlap_pairs_posthoc_from_per_chunk_geometry_v1",
        "created_by": "tools/build_v67_raw_overlap_pairs_from_geometry.py",
        "semantic_full_pt": str(semantic_full_pt),
        "geometry_dir": str(geometry_dir),
        "prev_chunk": int(prev_chunk),
        "curr_chunk": int(curr_chunk),
        "prev_start_frame": int(prev_start),
        "prev_end_frame": int(prev_end),
        "curr_start_frame": int(curr_start),
        "curr_end_frame": int(curr_end),
        "overlap_start_frame": int(overlap_start),
        "overlap_end_frame": int(overlap_end),
        "overlap_frame_count": int(len(frame_ids)),
        "sample_policy": str(sample_policy),
        "max_pairs_per_frame": int(max_pairs_per_frame),
        "min_conf": float(min_conf),
        "valid_pair_count": int(sum(valid_counts)),
        "saved_pair_count": int(saved),
        "semantic_label_projected_count": int(label_valid.sum().item()),
        "semantic_label_projected_ratio": float(label_valid.float().mean().item()) if saved else 0.0,
        "semantic_nonvoid_count": int(nonvoid.sum().item()),
        "semantic_nonvoid_ratio": float(nonvoid.float().mean().item()) if saved else 0.0,
        "raw_residual_mean": float(residual.mean().item()) if residual.numel() else None,
        "raw_residual_rmse": float(torch.sqrt((residual * residual).mean()).item()) if residual.numel() else None,
        "prev_overlap_points": prev_overlap_points,
        "curr_overlap_points": curr_overlap_points,
        "prev_overlap_local_points": _concat(parts, "prev_local_points", shape=(0, 3), dtype=torch.float32),
        "curr_overlap_local_points": _concat(parts, "curr_local_points", shape=(0, 3), dtype=torch.float32),
        "prev_conf": _concat(parts, "prev_conf", shape=(0,), dtype=torch.float32),
        "curr_conf": _concat(parts, "curr_conf", shape=(0,), dtype=torch.float32),
        "prev_frame_ids": _concat(parts, "frame_ids", shape=(0,), dtype=torch.long),
        "curr_frame_ids": _concat(parts, "frame_ids", shape=(0,), dtype=torch.long),
        "prev_pixel_coords": _concat(parts, "pixel_coords", shape=(0, 2), dtype=torch.int32),
        "curr_pixel_coords": _concat(parts, "pixel_coords", shape=(0, 2), dtype=torch.int32),
        "prev_semantic_labels": labels,
        "curr_semantic_labels": labels.clone(),
        "prev_semantic_conf": sem_conf_out,
        "curr_semantic_conf": sem_conf_out.clone(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return {
        "prev_chunk": int(prev_chunk),
        "curr_chunk": int(curr_chunk),
        "out_file": str(out_path),
        "overlap_start_frame": int(overlap_start),
        "overlap_end_frame": int(overlap_end),
        "overlap_frame_count": int(len(frame_ids)),
        "valid_pair_count": int(sum(valid_counts)),
        "saved_pair_count": int(saved),
        "semantic_label_projected_count": int(label_valid.sum().item()),
        "semantic_label_projected_ratio": float(label_valid.float().mean().item()) if saved else 0.0,
        "semantic_nonvoid_ratio": float(nonvoid.float().mean().item()) if saved else 0.0,
        "raw_residual_mean": float(residual.mean().item()) if residual.numel() else None,
        "raw_residual_rmse": float(torch.sqrt((residual * residual).mean()).item()) if residual.numel() else None,
        "written": bool(saved > 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-dir", type=Path, required=True)
    parser.add_argument("--semantic-full-pt", type=Path, default=Path("results/kitti_preprocess/01/sparse_masklets_with_semantic.pt"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--max-pairs-per-frame", type=int, default=20000)
    parser.add_argument("--min-conf", type=float, default=0.05)
    parser.add_argument(
        "--sample-policy",
        choices=("top_conf", "top_residual", "top_residual_conf_product"),
        default="top_conf",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    chunk_files = _chunk_paths(args.geometry_dir)
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.pt files in {args.geometry_dir}")
    sem_payload = torch.load(args.semantic_full_pt, map_location="cpu", weights_only=False)
    sem = sem_payload.get("semantic_segmentation", sem_payload) if isinstance(sem_payload, dict) else {}
    if not isinstance(sem, dict) or "label_maps" not in sem:
        raise ValueError(f"{args.semantic_full_pt}: missing semantic_segmentation.label_maps")
    label_maps = sem["label_maps"].detach().cpu() if torch.is_tensor(sem["label_maps"]) else torch.as_tensor(sem["label_maps"])
    conf_maps = sem.get("confidence_maps")
    conf_maps = conf_maps.detach().cpu() if torch.is_tensor(conf_maps) else (torch.as_tensor(conf_maps) if conf_maps is not None else None)
    label_names = sem.get("label_names", [])
    if isinstance(label_names, dict):
        label_names = [str(label_names[k]) for k in sorted(label_names, key=lambda x: int(x))]
    else:
        label_names = [str(x) for x in label_names]

    chunks: Dict[int, Tuple[Path, Dict[str, Any]]] = {}
    for path in chunk_files:
        payload = _load_pt(path)
        chunks[_chunk_id(path, payload)] = (path, payload)

    if args.out_dir.exists() and any(args.out_dir.glob("*.pt")) and not args.overwrite:
        raise FileExistsError(f"{args.out_dir} already has .pt files; pass --overwrite to replace")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for prev_chunk, curr_chunk in zip(sorted(chunks)[:-1], sorted(chunks)[1:]):
        if curr_chunk != prev_chunk + 1:
            continue
        out_file = args.out_dir / f"chunk_{prev_chunk:03d}_{curr_chunk:03d}.pt"
        rows.append(_write_pair_file(
            out_path=out_file,
            prev_chunk=prev_chunk,
            curr_chunk=curr_chunk,
            prev_payload=chunks[prev_chunk][1],
            curr_payload=chunks[curr_chunk][1],
            label_maps=label_maps,
            conf_maps=conf_maps,
            label_names=label_names,
            max_pairs_per_frame=int(args.max_pairs_per_frame),
            min_conf=float(args.min_conf),
            sample_policy=str(args.sample_policy),
            semantic_full_pt=args.semantic_full_pt,
            geometry_dir=args.geometry_dir,
        ))

    written = [row for row in rows if row.get("written")]
    total_saved = sum(int(row.get("saved_pair_count", 0)) for row in written)
    total_projected = sum(int(row.get("semantic_label_projected_count", 0)) for row in written)
    summary = {
        "schema": "acl2_v67_raw_overlap_pairs_posthoc_summary_v1",
        "created_by": "tools/build_v67_raw_overlap_pairs_from_geometry.py",
        "geometry_dir": str(args.geometry_dir),
        "semantic_full_pt": str(args.semantic_full_pt),
        "out_dir": str(args.out_dir),
        "sample_policy": str(args.sample_policy),
        "chunk_geometry_files": len(chunk_files),
        "candidate_adjacent_pairs": len(rows),
        "overlap_pair_files_written": len(written),
        "total_saved_pairs": int(total_saved),
        "total_semantic_label_projected": int(total_projected),
        "semantic_label_projected_pair_ratio": (
            float(total_projected) / float(total_saved) if total_saved else 0.0
        ),
        "median_saved_pairs_per_overlap": _median(row.get("saved_pair_count") for row in written),
        "median_valid_pairs_per_overlap": _median(row.get("valid_pair_count") for row in written),
        "median_raw_residual_rmse": _median(row.get("raw_residual_rmse") for row in written),
        "gate": {
            "overlap_pairs_rows_ge_30": len(written) >= 30,
            "semantic_labels_projected_ge_90pct": (
                (float(total_projected) / float(total_saved)) >= 0.90 if total_saved else False
            ),
            "phaseO2_overlap_pair_gate_pass": bool(
                len(written) >= 30
                and total_saved > 0
                and (float(total_projected) / float(total_saved)) >= 0.90
            ),
        },
        "rows": rows,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
