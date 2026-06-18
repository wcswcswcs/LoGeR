#!/usr/bin/env python3
"""Build forced merge-state traces from v66B per-chunk geometry.

The output JSONL is meant for run_pipeline_abc_v2.py --load_merge_state_path.
Each row stores rotation, translation, and transform_scale_value separately so
the loader can construct a Sim(3) matrix without losing scale.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tools.diagnose_v66b_dense_semantic_scale as diag


def _read_audit_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _parse_chunks(text: str, default_max: int) -> List[int]:
    text = str(text or "").strip()
    if not text or text == "all":
        return list(range(default_max + 1))
    out: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _sample_fit(
    *,
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
    source_slice: slice,
    target_slice: slice,
    max_points: int,
    seed: int,
) -> Optional[Tuple[float, torch.Tensor, torch.Tensor, float, int, int, float]]:
    mask = (weights[source_slice] > 0) & valid[source_slice]
    valid_count = int(mask.sum().item())
    if valid_count < 30:
        return None
    idx = diag._sample_flat(mask, int(max_points), int(seed))
    x = source_points[source_slice].reshape(-1, 3)[idx]
    y = target_points[target_slice].reshape(-1, 3)[idx]
    flat_w = weights[source_slice].reshape(-1)[idx]
    fit = diag._weighted_umeyama(x, y, flat_w)
    if fit is None:
        return None
    scale, rot, trans, condition = fit
    pred = diag._apply_sim3(x, fit)
    err = torch.linalg.norm(pred - y.float(), dim=1)
    residual = float(torch.sqrt((err * err).mean()).item()) if err.numel() else float("nan")
    return float(scale), rot.detach().cpu().float(), trans.detach().cpu().float(), float(condition), valid_count, int(idx.numel()), residual


def _apply_sim3_volume(
    points: torch.Tensor,
    fit: Tuple[float, torch.Tensor, torch.Tensor, float],
) -> torch.Tensor:
    shape = points.shape
    flat = points.reshape(-1, 3)
    aligned = diag._apply_sim3(flat, fit)
    return aligned.reshape(shape).to(torch.float32)


def _row_hash_payload(row: Mapping[str, Any]) -> str:
    import hashlib

    payload = json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _identity_row(cid: int, audit: Mapping[str, Any], *, strategy: str) -> Dict[str, Any]:
    row = {
        "schema": "acl2_v66b_phase20_geometry_merge_trace_v1",
        "chunk_idx": int(cid),
        "local_chunk_idx": int(cid),
        "window_idx": int(cid),
        "start_frame": int(audit["start_frame"]),
        "end_frame": int(audit["end_frame"]),
        "strategy": strategy,
        "fit_success": True,
        "fit_reason": "identity_first_window",
        "rotation": torch.eye(3).tolist(),
        "translation": [0.0, 0.0, 0.0],
        "transform_scale_value": 1.0,
        "transform_kind": "sim3_geometry_fit",
        "transform_reason": "identity_first_window",
        "fit_point_count": 0,
        "valid_point_count": 0,
        "overlap_residual": None,
        "sim3_condition_score": None,
    }
    row["state_hash"] = _row_hash_payload(row)
    row["transform_hash"] = row["state_hash"]
    return row


def build_trace(
    *,
    audit_rows: Sequence[Mapping[str, Any]],
    stage_c_cache: Path,
    geometry_dir: Path,
    strategy: str,
    target_chunks: Sequence[int],
    max_points: int,
    random_seed: int,
    fit_mode: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_id = {int(row["chunk_id"]): row for row in audit_rows}
    _, first_sem = diag._load_chunk_semantic(stage_c_cache, audit_rows[0])
    group_ids = diag._label_group_ids([str(x) for x in first_sem.get("label_names", [])])
    rows: List[Dict[str, Any]] = []
    fit_failures: List[int] = []
    aligned_world_by_chunk: Dict[int, torch.Tensor] = {}

    for cid in target_chunks:
        if cid not in by_id:
            continue
        audit = by_id[cid]
        if cid == min(target_chunks):
            rows.append(_identity_row(cid, audit, strategy=strategy))
            first_geo = diag._load_geometry(geometry_dir / f"chunk_{cid:03d}.pt")
            first_world = first_geo.get("points", first_geo.get("world_points")) if first_geo else None
            if torch.is_tensor(first_world):
                aligned_world_by_chunk[cid] = first_world.detach().cpu().float()
            continue
        if (cid - 1) not in by_id:
            fit_failures.append(cid)
            continue

        cur_geo = diag._load_geometry(geometry_dir / f"chunk_{cid:03d}.pt")
        prev_geo = diag._load_geometry(geometry_dir / f"chunk_{cid - 1:03d}.pt")
        if not cur_geo or not prev_geo:
            fit_failures.append(cid)
            continue
        cur_local = cur_geo.get("local_points")
        cur_world = cur_geo.get("points", cur_geo.get("world_points"))
        prev_world = prev_geo.get("points", prev_geo.get("world_points"))
        conf = cur_geo.get("conf", cur_geo.get("confidence"))
        if not (
            torch.is_tensor(cur_local)
            and torch.is_tensor(cur_world)
            and torch.is_tensor(prev_world)
            and torch.is_tensor(conf)
        ):
            fit_failures.append(cid)
            continue

        cur_local = cur_local.detach().cpu().float()
        cur_world = cur_world.detach().cpu().float()
        prev_world = prev_world.detach().cpu().float()
        conf = conf.detach().cpu().float().clamp(0.0, 1.0)

        start = int(audit["start_frame"])
        end = int(audit["end_frame"])
        prev_audit = by_id[cid - 1]
        prev_start = int(prev_audit["start_frame"])
        prev_end = int(prev_audit["end_frame"])
        overlap_start = max(start, prev_start)
        overlap_end = min(end, prev_end)
        if overlap_end <= overlap_start:
            fit_failures.append(cid)
            continue
        cur_sl = slice(overlap_start - start, overlap_end - start)
        prev_sl = slice(overlap_start - prev_start, overlap_end - prev_start)

        label_maps, _ = diag._load_chunk_semantic(stage_c_cache, audit)
        label_point = diag._project_label_maps(label_maps[: int(conf.shape[0])], (int(conf.shape[-2]), int(conf.shape[-1])))
        masks = diag._chunk_masks(label_point, group_ids)
        if fit_mode == "current_local_to_previous_world":
            source_points = cur_local
            target_points = prev_world
            fit_reason = "weighted_overlap_umeyama_current_local_to_prev_world"
        elif fit_mode == "current_world_to_aligned_previous":
            prev_aligned = aligned_world_by_chunk.get(cid - 1)
            if prev_aligned is None:
                fit_failures.append(cid)
                continue
            source_points = cur_world
            target_points = prev_aligned
            fit_reason = "weighted_overlap_umeyama_current_world_to_aligned_prev_world"
        else:
            raise ValueError(f"Unsupported fit_mode={fit_mode}")

        valid = (
            torch.isfinite(source_points).all(dim=-1)
            & torch.isfinite(target_points[: source_points.shape[0]]).all(dim=-1)
            & (conf > 0.05)
        )
        weights, info = diag._strategy_weights(
            strategy,
            valid,
            conf,
            masks,
            chunk_id=cid,
            label_point=label_point,
            random_seed=random_seed,
        )
        fit = _sample_fit(
            source_points=source_points,
            target_points=target_points,
            weights=weights,
            valid=valid,
            source_slice=cur_sl,
            target_slice=prev_sl,
            max_points=max_points,
            seed=random_seed + cid * 1009 + len(strategy),
        )
        if fit is None:
            fit_failures.append(cid)
            continue
        scale, rot, trans, condition, valid_count, fit_count, residual = fit
        row = {
            "schema": "acl2_v66b_phase20_geometry_merge_trace_v1",
            "chunk_idx": int(cid),
            "local_chunk_idx": int(cid),
            "window_idx": int(cid),
            "start_frame": start,
            "end_frame": end,
            "strategy": strategy,
            "fit_success": True,
            "fit_reason": fit_reason,
            "fit_mode": fit_mode,
            "overlap_start_frame": int(overlap_start),
            "overlap_end_frame": int(overlap_end),
            "overlap_frame_count": int(overlap_end - overlap_start),
            "rotation": rot.tolist(),
            "translation": trans.reshape(-1).tolist(),
            "transform_scale_value": float(scale),
            "transform_kind": "sim3_geometry_fit",
            "transform_reason": "phase20_geometry_overlap_fit",
            "fit_point_count": int(fit_count),
            "valid_point_count": int(valid_count),
            "overlap_residual": float(residual),
            "sim3_condition_score": float(condition),
            "removed_dynamic_mass": info.get("removed_dynamic_mass"),
            "removed_sky_mass": info.get("removed_sky_mass"),
            "removed_vegetation_mass": info.get("removed_vegetation_mass"),
            "kept_vertical_static_mass": info.get("kept_vertical_static_mass"),
            "kept_ground_mass": info.get("kept_ground_mass"),
            "remaining_valid_mass": info.get("remaining_valid_mass"),
            "remaining_valid_ratio": info.get("remaining_valid_ratio"),
            "control_type": info.get("control_type"),
        }
        row["state_hash"] = _row_hash_payload(row)
        row["transform_hash"] = row["state_hash"]
        rows.append(row)
        if fit_mode == "current_world_to_aligned_previous":
            aligned_world_by_chunk[cid] = _apply_sim3_volume(
                cur_world,
                (scale, rot, trans, condition),
            )

    summary = {
        "strategy": strategy,
        "fit_mode": fit_mode,
        "geometry_dir": str(geometry_dir),
        "stage_c_cache": str(stage_c_cache),
        "target_chunks": list(map(int, target_chunks)),
        "row_count": len(rows),
        "fit_success_count": sum(1 for row in rows if row.get("fit_success")),
        "fit_failures": fit_failures,
        "scale_min": min((float(row["transform_scale_value"]) for row in rows), default=None),
        "scale_max": max((float(row["transform_scale_value"]) for row in rows), default=None),
        "scale_mean": (
            sum(float(row["transform_scale_value"]) for row in rows) / len(rows)
            if rows
            else None
        ),
        "median_overlap_residual": diag._median(row.get("overlap_residual") for row in rows),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit_csv", type=Path, default=Path("results/kitti01_hmc_v2/acl2_v66b_dense_semantic_scale/report_final/phase19_full704_geometry_diagnostic/phase0_cache_audit/semantic_cache_audit.csv"))
    parser.add_argument("--stage_c_cache", type=Path, default=Path("results/kitti_preprocess/01/stage_c_cache_semantic_chunks"))
    parser.add_argument("--geometry_dir", type=Path, default=Path("results/kitti01_hmc_v2/acl2_v66b_artifacts/H35_FULL/per_chunk_geometry"))
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--target_chunks", default="0-24")
    parser.add_argument("--max_points_per_fit", type=int, default=12000)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument(
        "--fit_mode",
        choices=("current_local_to_previous_world", "current_world_to_aligned_previous"),
        default="current_local_to_previous_world",
        help=(
            "current_local_to_previous_world reproduces the initial pairwise diagnostic trace; "
            "current_world_to_aligned_previous composes absolute replay transforms into one gauge."
        ),
    )
    parser.add_argument("--out_jsonl", type=Path, required=True)
    parser.add_argument("--out_summary", type=Path, default=None)
    args = parser.parse_args()

    audit_rows = _read_audit_rows(args.audit_csv)
    target_chunks = _parse_chunks(args.target_chunks, max(int(row["chunk_id"]) for row in audit_rows))
    rows, summary = build_trace(
        audit_rows=audit_rows,
        stage_c_cache=args.stage_c_cache,
        geometry_dir=args.geometry_dir,
        strategy=args.strategy,
        target_chunks=target_chunks,
        max_points=int(args.max_points_per_fit),
        random_seed=int(args.random_seed),
        fit_mode=str(args.fit_mode),
    )
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary_path = args.out_summary or args.out_jsonl.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
