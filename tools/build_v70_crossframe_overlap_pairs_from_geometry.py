#!/usr/bin/env python3
"""Build diagnostic cross-frame overlap pairs for ACL2 v70 RADIO MERGE checks.

The older v67 overlap-pair materializer compares duplicate observations from
the same frame/pixel under adjacent chunk gauges.  That is useful for gauge
diagnostics, but it makes RADIO feature cosine trivially 1.0.  This tool builds
an explicitly diagnostic alternative: adjacent cross-frame pairs around each
chunk boundary, matched by nearest 3D points after confidence filtering.

The output schema keeps the field names consumed by
``diagnose_v70_radio_merge_oracle.py`` but marks the provenance as cross-frame
nearest-neighbor.  It is not ground-truth optical flow.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


DEFAULT_TARGET_CHUNKS = "6,7,8,10,12,19,20,29,30,31,32"
DEFAULT_GEOMETRY_DIR = Path(
    "results/kitti01_hmc_v2/acl2_v67_dense_semantic_reconstruction/"
    "phaseO2_h35_trace_geom_merge_full/rollouts/"
    "V67S_H35_TRACE_GEOM_MERGE_FULL_H35_PARITY/per_chunk_geometry"
)
DEFAULT_SEMANTIC_FULL = Path("results/kitti_preprocess/01/sparse_masklets_with_semantic.pt")


def _load_pt(path: Path) -> Dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected dict payload, got {type(obj).__name__}")
    return obj


def _tensor(payload: Dict[str, Any], *keys: str) -> Optional[torch.Tensor]:
    for key in keys:
        value = payload.get(key)
        if torch.is_tensor(value):
            return value.detach().cpu()
    return None


def _required_tensor(payload: Dict[str, Any], *keys: str) -> torch.Tensor:
    value = _tensor(payload, *keys)
    if value is None:
        raise ValueError(f"missing tensor field; tried {keys}")
    return value


def _parse_chunks(text: str) -> List[int]:
    out: List[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


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


def _percentile(values: torch.Tensor, q: float) -> Optional[float]:
    if values.numel() == 0:
        return None
    return float(torch.quantile(values.float(), float(q)).item())


def _chunk_paths(geometry_dir: Path) -> List[Path]:
    return sorted(geometry_dir.glob("chunk_*.pt"))


def _chunk_id(path: Path, payload: Dict[str, Any]) -> int:
    if "chunk_idx" in payload:
        return int(payload["chunk_idx"])
    return int(path.stem.split("_")[-1])


def _load_semantics(path: Path) -> Tuple[torch.Tensor, Optional[torch.Tensor], Sequence[str]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    sem = payload.get("semantic_segmentation", payload) if isinstance(payload, dict) else {}
    if not isinstance(sem, dict) or "label_maps" not in sem:
        raise ValueError(f"{path}: missing semantic_segmentation.label_maps")
    labels = sem["label_maps"].detach().cpu() if torch.is_tensor(sem["label_maps"]) else torch.as_tensor(sem["label_maps"])
    conf = sem.get("confidence_maps")
    conf = conf.detach().cpu() if torch.is_tensor(conf) else (torch.as_tensor(conf) if conf is not None else None)
    label_names = sem.get("label_names", [])
    if isinstance(label_names, dict):
        label_names = [str(label_names[k]) for k in sorted(label_names, key=lambda x: int(x))]
    else:
        label_names = [str(x) for x in label_names]
    return labels, conf, label_names


def _grid_coords(indices: torch.Tensor, height: int, width: int) -> torch.Tensor:
    y = torch.div(indices, int(width), rounding_mode="floor")
    x = indices - y * int(width)
    return torch.stack([y, x], dim=1).to(torch.float32)


def _scale_coords(coords_yx: torch.Tensor, src_size: Tuple[int, int], dst_size: Tuple[int, int]) -> torch.Tensor:
    src_h, src_w = int(src_size[0]), int(src_size[1])
    dst_h, dst_w = int(dst_size[0]), int(dst_size[1])
    y = (coords_yx[:, 0] + 0.5) / max(float(src_h), 1.0) * float(dst_h)
    x = (coords_yx[:, 1] + 0.5) / max(float(src_w), 1.0) * float(dst_w)
    y = torch.clamp(y, 0.0, max(float(dst_h - 1), 0.0))
    x = torch.clamp(x, 0.0, max(float(dst_w - 1), 0.0))
    return torch.stack([y, x], dim=1)


def _sample_semantic(
    label_maps: torch.Tensor,
    conf_maps: Optional[torch.Tensor],
    frame_id: int,
    raw_coords_yx: torch.Tensor,
    *,
    raw_size: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    sem_h, sem_w = int(label_maps.shape[-2]), int(label_maps.shape[-1])
    coords = _scale_coords(raw_coords_yx, raw_size, (sem_h, sem_w)).round().long()
    yy = torch.clamp(coords[:, 0], 0, sem_h - 1)
    xx = torch.clamp(coords[:, 1], 0, sem_w - 1)
    labels = label_maps[int(frame_id), yy, xx].long()
    if conf_maps is None:
        conf = torch.ones_like(labels, dtype=torch.float32)
    else:
        conf = conf_maps[int(frame_id), yy, xx].float().clamp(0.0, 1.0)
    return labels, conf


def _top_conf_indices(
    points: torch.Tensor,
    conf: torch.Tensor,
    *,
    min_conf: float,
    max_candidates: int,
) -> torch.Tensor:
    valid = (
        torch.isfinite(points).all(dim=-1)
        & torch.isfinite(conf)
        & (conf >= float(min_conf))
    ).reshape(-1)
    idx = torch.where(valid)[0]
    if idx.numel() == 0:
        return idx
    scores = conf.reshape(-1)[idx].float()
    k = min(int(max_candidates), int(idx.numel())) if int(max_candidates) > 0 else int(idx.numel())
    order = torch.argsort(scores, descending=True, stable=True)[:k]
    return idx[order].long()


def _match_frame_pair(
    *,
    prev_payload: Dict[str, Any],
    curr_payload: Dict[str, Any],
    prev_frame_id: int,
    curr_frame_id: int,
    raw_size: Tuple[int, int],
    label_maps: torch.Tensor,
    conf_maps: Optional[torch.Tensor],
    min_conf: float,
    max_candidates: int,
    max_pairs: int,
    max_nn_dist_m: float,
    mutual: bool,
    match_space: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    prev_points_all = _required_tensor(prev_payload, "points", "world_points").float()
    curr_points_all = _required_tensor(curr_payload, "points", "world_points").float()
    prev_local_all = _tensor(prev_payload, "local_points")
    curr_local_all = _tensor(curr_payload, "local_points")
    prev_conf_all = _required_tensor(prev_payload, "conf", "confidence").float().clamp(0.0, 1.0)
    curr_conf_all = _required_tensor(curr_payload, "conf", "confidence").float().clamp(0.0, 1.0)
    if prev_points_all.numel() == 0 or curr_points_all.numel() == 0 or prev_conf_all.numel() == 0 or curr_conf_all.numel() == 0:
        raise ValueError("missing points/conf tensors")

    prev_start = int(prev_payload["start_frame"])
    curr_start = int(curr_payload["start_frame"])
    prev_i = int(prev_frame_id - prev_start)
    curr_i = int(curr_frame_id - curr_start)
    if prev_i < 0 or prev_i >= int(prev_points_all.shape[0]):
        raise IndexError(f"prev_frame_id {prev_frame_id} outside chunk range")
    if curr_i < 0 or curr_i >= int(curr_points_all.shape[0]):
        raise IndexError(f"curr_frame_id {curr_frame_id} outside chunk range")

    prev_points = prev_points_all[prev_i]
    curr_points = curr_points_all[curr_i]
    prev_conf = prev_conf_all[prev_i]
    curr_conf = curr_conf_all[curr_i]
    prev_local = prev_local_all[prev_i] if torch.is_tensor(prev_local_all) else None
    curr_local = curr_local_all[curr_i] if torch.is_tensor(curr_local_all) else None
    height, width = int(prev_points.shape[-3]), int(prev_points.shape[-2])
    if tuple(curr_points.shape[-3:-1]) != (height, width):
        raise ValueError(f"geometry size mismatch: {tuple(prev_points.shape)} vs {tuple(curr_points.shape)}")

    prev_idx = _top_conf_indices(prev_points, prev_conf, min_conf=min_conf, max_candidates=max_candidates)
    curr_idx = _top_conf_indices(curr_points, curr_conf, min_conf=min_conf, max_candidates=max_candidates)
    if prev_idx.numel() == 0 or curr_idx.numel() == 0:
        empty_i = torch.empty((0,), dtype=torch.long)
        empty_p = torch.empty((0, 3), dtype=torch.float32)
        return {
            "indices": empty_i,
            "prev_match_indices": empty_i,
            "curr_match_indices": empty_i,
            "prev_points": empty_p,
            "curr_points": empty_p,
            "prev_local_points": empty_p,
            "curr_local_points": empty_p,
            "prev_conf": torch.empty((0,), dtype=torch.float32),
            "curr_conf": torch.empty((0,), dtype=torch.float32),
            "prev_pixel_coords": torch.empty((0, 2), dtype=torch.float32),
            "curr_pixel_coords": torch.empty((0, 2), dtype=torch.float32),
            "prev_semantic_labels": empty_i,
            "curr_semantic_labels": empty_i,
            "prev_semantic_conf": torch.empty((0,), dtype=torch.float32),
            "curr_semantic_conf": torch.empty((0,), dtype=torch.float32),
            "nn_dist_m": torch.empty((0,), dtype=torch.float32),
        }, {
            "prev_candidate_count": int(prev_idx.numel()),
            "curr_candidate_count": int(curr_idx.numel()),
            "saved_pair_count": 0,
            "reason": "no_candidates",
            "match_space": str(match_space),
        }

    prev_flat = prev_points.reshape(-1, 3)[prev_idx].float()
    curr_flat = curr_points.reshape(-1, 3)[curr_idx].float()
    if match_space == "local_points":
        if prev_local is None or curr_local is None:
            raise ValueError("match_space=local_points requires local_points tensors")
        prev_query = prev_local.reshape(-1, 3)[prev_idx].float()
        curr_query = curr_local.reshape(-1, 3)[curr_idx].float()
    elif match_space == "median_centered_points":
        prev_query = prev_flat - torch.median(prev_flat, dim=0).values
        curr_query = curr_flat - torch.median(curr_flat, dim=0).values
    elif match_space == "points":
        prev_query = prev_flat
        curr_query = curr_flat
    else:
        raise ValueError(f"unsupported match_space={match_space!r}")
    dists = torch.cdist(prev_query, curr_query, p=2.0)
    nn_dist, nn_j = torch.min(dists, dim=1)

    keep = torch.isfinite(nn_dist) & (nn_dist <= float(max_nn_dist_m))
    if mutual and keep.any():
        _, rev_i = torch.min(dists, dim=0)
        keep = keep & (rev_i[nn_j] == torch.arange(nn_j.numel()))
    if not keep.any():
        empty_i = torch.empty((0,), dtype=torch.long)
        empty_p = torch.empty((0, 3), dtype=torch.float32)
        return {
            "indices": empty_i,
            "prev_match_indices": empty_i,
            "curr_match_indices": empty_i,
            "prev_points": empty_p,
            "curr_points": empty_p,
            "prev_local_points": empty_p,
            "curr_local_points": empty_p,
            "prev_conf": torch.empty((0,), dtype=torch.float32),
            "curr_conf": torch.empty((0,), dtype=torch.float32),
            "prev_pixel_coords": torch.empty((0, 2), dtype=torch.float32),
            "curr_pixel_coords": torch.empty((0, 2), dtype=torch.float32),
            "prev_semantic_labels": empty_i,
            "curr_semantic_labels": empty_i,
            "prev_semantic_conf": torch.empty((0,), dtype=torch.float32),
            "curr_semantic_conf": torch.empty((0,), dtype=torch.float32),
            "nn_dist_m": torch.empty((0,), dtype=torch.float32),
        }, {
            "prev_candidate_count": int(prev_idx.numel()),
            "curr_candidate_count": int(curr_idx.numel()),
            "saved_pair_count": 0,
            "reason": "no_matches_under_distance",
            "match_space": str(match_space),
        }

    kept_prev_pos = torch.where(keep)[0]
    kept_curr_pos = nn_j[keep]
    kept_dist = nn_dist[keep]
    prev_conf_kept = prev_conf.reshape(-1)[prev_idx[kept_prev_pos]].float()
    curr_conf_kept = curr_conf.reshape(-1)[curr_idx[kept_curr_pos]].float()
    score = torch.minimum(prev_conf_kept, curr_conf_kept) / (1.0 + kept_dist.float())
    order = torch.argsort(score, descending=True, stable=True)
    if int(max_pairs) > 0:
        order = order[: int(max_pairs)]

    prev_match_idx = prev_idx[kept_prev_pos[order]].long()
    curr_match_idx = curr_idx[kept_curr_pos[order]].long()
    nn_dist_out = kept_dist[order].float()

    prev_geom_coords = _grid_coords(prev_match_idx, height, width)
    curr_geom_coords = _grid_coords(curr_match_idx, height, width)
    prev_raw_coords = _scale_coords(prev_geom_coords, (height, width), raw_size)
    curr_raw_coords = _scale_coords(curr_geom_coords, (height, width), raw_size)
    prev_labels, prev_sem_conf = _sample_semantic(label_maps, conf_maps, prev_frame_id, prev_raw_coords, raw_size=raw_size)
    curr_labels, curr_sem_conf = _sample_semantic(label_maps, conf_maps, curr_frame_id, curr_raw_coords, raw_size=raw_size)

    prev_local_points = (
        prev_local_all[prev_i].reshape(-1, 3)[prev_match_idx].float()
        if torch.is_tensor(prev_local_all)
        else prev_points.reshape(-1, 3)[prev_match_idx].float()
    )
    curr_local_points = (
        curr_local_all[curr_i].reshape(-1, 3)[curr_match_idx].float()
        if torch.is_tensor(curr_local_all)
        else curr_points.reshape(-1, 3)[curr_match_idx].float()
    )
    payload = {
        "indices": torch.arange(prev_match_idx.numel(), dtype=torch.long),
        "prev_match_indices": prev_match_idx,
        "curr_match_indices": curr_match_idx,
        "prev_points": prev_points.reshape(-1, 3)[prev_match_idx].float(),
        "curr_points": curr_points.reshape(-1, 3)[curr_match_idx].float(),
        "prev_local_points": prev_local_points,
        "curr_local_points": curr_local_points,
        "prev_conf": prev_conf.reshape(-1)[prev_match_idx].float(),
        "curr_conf": curr_conf.reshape(-1)[curr_match_idx].float(),
        "prev_pixel_coords": prev_raw_coords.float(),
        "curr_pixel_coords": curr_raw_coords.float(),
        "prev_semantic_labels": prev_labels.long(),
        "curr_semantic_labels": curr_labels.long(),
        "prev_semantic_conf": prev_sem_conf.float(),
        "curr_semantic_conf": curr_sem_conf.float(),
        "nn_dist_m": nn_dist_out,
    }
    meta = {
        "prev_candidate_count": int(prev_idx.numel()),
        "curr_candidate_count": int(curr_idx.numel()),
        "saved_pair_count": int(prev_match_idx.numel()),
        "match_space": str(match_space),
        "nn_dist_m_median": _percentile(nn_dist_out, 0.5),
        "nn_dist_m_p90": _percentile(nn_dist_out, 0.9),
        "nn_dist_m_max": float(nn_dist_out.max().item()) if nn_dist_out.numel() else None,
        "same_semantic_label_ratio": (
            float((prev_labels == curr_labels).float().mean().item()) if prev_labels.numel() else None
        ),
        "prev_semantic_conf_median": _percentile(prev_sem_conf, 0.5),
        "curr_semantic_conf_median": _percentile(curr_sem_conf, 0.5),
    }
    return payload, meta


def _write_pair(
    *,
    out_path: Path,
    prev_chunk: int,
    curr_chunk: int,
    prev_payload: Dict[str, Any],
    curr_payload: Dict[str, Any],
    label_maps: torch.Tensor,
    conf_maps: Optional[torch.Tensor],
    label_names: Sequence[str],
    raw_size: Tuple[int, int],
    prev_frame_offset: int,
    curr_frame_offset: int,
    min_conf: float,
    max_candidates: int,
    max_pairs: int,
    max_nn_dist_m: float,
    mutual: bool,
    match_space: str,
    geometry_dir: Path,
    semantic_full_pt: Path,
) -> Dict[str, Any]:
    curr_start = int(curr_payload["start_frame"])
    prev_frame_id = curr_start + int(prev_frame_offset)
    curr_frame_id = curr_start + int(curr_frame_offset)
    part, meta = _match_frame_pair(
        prev_payload=prev_payload,
        curr_payload=curr_payload,
        prev_frame_id=prev_frame_id,
        curr_frame_id=curr_frame_id,
        raw_size=raw_size,
        label_maps=label_maps,
        conf_maps=conf_maps,
        min_conf=float(min_conf),
        max_candidates=int(max_candidates),
        max_pairs=int(max_pairs),
        max_nn_dist_m=float(max_nn_dist_m),
        mutual=bool(mutual),
        match_space=str(match_space),
    )
    saved = int(part["prev_points"].shape[0])
    residual = torch.linalg.norm(part["prev_points"] - part["curr_points"], dim=1) if saved else torch.empty((0,))
    pix_l2 = torch.linalg.norm(part["prev_pixel_coords"] - part["curr_pixel_coords"], dim=1) if saved else torch.empty((0,))
    payload: Dict[str, Any] = {
        "schema": "acl2_v70_crossframe_overlap_pairs_from_geometry_v1",
        "created_by": "tools/build_v70_crossframe_overlap_pairs_from_geometry.py",
        "diagnostic_only": True,
        "correspondence_source": "cross_frame_3d_nearest_neighbor_not_ground_truth_flow",
        "geometry_dir": str(geometry_dir),
        "semantic_full_pt": str(semantic_full_pt),
        "prev_chunk": int(prev_chunk),
        "curr_chunk": int(curr_chunk),
        "prev_start_frame": int(prev_payload["start_frame"]),
        "prev_end_frame": int(prev_payload["end_frame"]),
        "curr_start_frame": int(curr_payload["start_frame"]),
        "curr_end_frame": int(curr_payload["end_frame"]),
        "prev_frame_id": int(prev_frame_id),
        "curr_frame_id": int(curr_frame_id),
        "prev_frame_offset_from_curr_start": int(prev_frame_offset),
        "curr_frame_offset_from_curr_start": int(curr_frame_offset),
        "frame_delta": int(curr_frame_id - prev_frame_id),
        "raw_frame_height": int(raw_size[0]),
        "raw_frame_width": int(raw_size[1]),
        "pixel_coords_are_raw_image_yx": True,
        "geometry_pixel_coords_scaled_to_raw_image": True,
        "min_conf": float(min_conf),
        "max_candidates": int(max_candidates),
        "max_pairs": int(max_pairs),
        "max_nn_dist_m": float(max_nn_dist_m),
        "mutual_nn": bool(mutual),
        "match_space": str(match_space),
        "label_names": list(label_names),
        "valid_pair_count": int(meta.get("saved_pair_count", 0)),
        "saved_pair_count": int(saved),
        "raw_residual_mean": float(residual.mean().item()) if residual.numel() else None,
        "raw_residual_rmse": float(torch.sqrt((residual * residual).mean()).item()) if residual.numel() else None,
        "pixel_l2_median": _percentile(pix_l2, 0.5),
        "pixel_l2_p90": _percentile(pix_l2, 0.9),
        **part,
        "prev_frame_ids": torch.full((saved,), int(prev_frame_id), dtype=torch.long),
        "curr_frame_ids": torch.full((saved,), int(curr_frame_id), dtype=torch.long),
        "prev_overlap_points": part["prev_points"],
        "curr_overlap_points": part["curr_points"],
        "prev_overlap_local_points": part["prev_local_points"],
        "curr_overlap_local_points": part["curr_local_points"],
        "prev_semantic_labels": part["prev_semantic_labels"],
        "curr_semantic_labels": part["curr_semantic_labels"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    row = {
        "prev_chunk": int(prev_chunk),
        "curr_chunk": int(curr_chunk),
        "out_file": str(out_path),
        "written": bool(saved > 0),
        "saved_pair_count": int(saved),
        "prev_frame_id": int(prev_frame_id),
        "curr_frame_id": int(curr_frame_id),
        "frame_delta": int(curr_frame_id - prev_frame_id),
        "raw_residual_rmse": payload["raw_residual_rmse"],
        "pixel_l2_median": payload["pixel_l2_median"],
        "pixel_l2_p90": payload["pixel_l2_p90"],
        **meta,
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-dir", type=Path, default=DEFAULT_GEOMETRY_DIR)
    parser.add_argument("--semantic-full-pt", type=Path, default=DEFAULT_SEMANTIC_FULL)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--target-chunks", default=DEFAULT_TARGET_CHUNKS)
    parser.add_argument("--raw-frame-height", type=int, default=376)
    parser.add_argument("--raw-frame-width", type=int, default=1241)
    parser.add_argument("--prev-frame-offset", type=int, default=-1)
    parser.add_argument("--curr-frame-offset", type=int, default=0)
    parser.add_argument("--min-conf", type=float, default=0.05)
    parser.add_argument("--max-candidates-per-frame", type=int, default=4096)
    parser.add_argument("--max-pairs-per-boundary", type=int, default=4096)
    parser.add_argument("--max-nn-dist-m", type=float, default=2.0)
    parser.add_argument(
        "--match-space",
        choices=["points", "local_points", "median_centered_points"],
        default="points",
        help="Coordinate space used only for nearest-neighbor correspondence search; output points remain original chunk points.",
    )
    parser.add_argument("--mutual-nn", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    chunk_files = _chunk_paths(args.geometry_dir)
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.pt files in {args.geometry_dir}")
    if args.out_dir.exists() and any(args.out_dir.glob("*.pt")) and not args.overwrite:
        raise FileExistsError(f"{args.out_dir} already has .pt files; pass --overwrite to replace")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    label_maps, conf_maps, label_names = _load_semantics(args.semantic_full_pt)
    chunks: Dict[int, Dict[str, Any]] = {}
    for path in chunk_files:
        payload = _load_pt(path)
        chunks[_chunk_id(path, payload)] = payload

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for curr_chunk in _parse_chunks(args.target_chunks):
        prev_chunk = int(curr_chunk) - 1
        if prev_chunk not in chunks or curr_chunk not in chunks:
            failures.append({"prev_chunk": prev_chunk, "curr_chunk": curr_chunk, "reason": "missing_chunk_geometry"})
            continue
        try:
            row = _write_pair(
                out_path=args.out_dir / f"chunk_{prev_chunk:03d}_{curr_chunk:03d}.pt",
                prev_chunk=prev_chunk,
                curr_chunk=curr_chunk,
                prev_payload=chunks[prev_chunk],
                curr_payload=chunks[curr_chunk],
                label_maps=label_maps,
                conf_maps=conf_maps,
                label_names=label_names,
                raw_size=(int(args.raw_frame_height), int(args.raw_frame_width)),
                prev_frame_offset=int(args.prev_frame_offset),
                curr_frame_offset=int(args.curr_frame_offset),
                min_conf=float(args.min_conf),
                max_candidates=int(args.max_candidates_per_frame),
                max_pairs=int(args.max_pairs_per_boundary),
                max_nn_dist_m=float(args.max_nn_dist_m),
                mutual=bool(args.mutual_nn),
                match_space=str(args.match_space),
                geometry_dir=args.geometry_dir,
                semantic_full_pt=args.semantic_full_pt,
            )
            rows.append(row)
        except Exception as exc:  # noqa: BLE001 - diagnostic artifact records failures.
            failures.append({
                "prev_chunk": prev_chunk,
                "curr_chunk": curr_chunk,
                "reason": f"{type(exc).__name__}:{exc}",
            })

    written = [row for row in rows if row.get("written")]
    summary = {
        "schema": "acl2_v70_crossframe_overlap_pairs_summary_v1",
        "created_by": "tools/build_v70_crossframe_overlap_pairs_from_geometry.py",
        "diagnostic_only": True,
        "correspondence_source": "cross_frame_3d_nearest_neighbor_not_ground_truth_flow",
        "geometry_dir": str(args.geometry_dir),
        "semantic_full_pt": str(args.semantic_full_pt),
        "out_dir": str(args.out_dir),
        "target_chunks": _parse_chunks(args.target_chunks),
        "raw_frame_height": int(args.raw_frame_height),
        "raw_frame_width": int(args.raw_frame_width),
        "prev_frame_offset": int(args.prev_frame_offset),
        "curr_frame_offset": int(args.curr_frame_offset),
        "min_conf": float(args.min_conf),
        "max_candidates_per_frame": int(args.max_candidates_per_frame),
        "max_pairs_per_boundary": int(args.max_pairs_per_boundary),
        "max_nn_dist_m": float(args.max_nn_dist_m),
        "match_space": str(args.match_space),
        "mutual_nn": bool(args.mutual_nn),
        "pair_files_written": len(written),
        "total_saved_pairs": int(sum(int(row.get("saved_pair_count", 0)) for row in written)),
        "median_saved_pairs_per_boundary": _median(row.get("saved_pair_count") for row in written),
        "median_nn_dist_m": _median(row.get("nn_dist_m_median") for row in written),
        "median_pixel_l2": _median(row.get("pixel_l2_median") for row in written),
        "failures": failures,
        "gate": {
            "all_target_pairs_written": len(written) == len(_parse_chunks(args.target_chunks)),
            "nonzero_pairs": all(int(row.get("saved_pair_count", 0)) > 0 for row in written) if written else False,
            "cross_frame_pair_artifact_gate_pass": (
                len(written) == len(_parse_chunks(args.target_chunks))
                and all(int(row.get("saved_pair_count", 0)) > 0 for row in written)
                and not failures
            ),
        },
        "rows": rows,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
