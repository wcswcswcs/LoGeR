#!/usr/bin/env python3
"""Build bad-window tables for v78 memory-family targeting.

This is diagnostic-only: it reads existing TUM trajectories and KITTI GT poses,
then ranks single chunks, adjacent chunk pairs, and contiguous 5-chunk windows.
It does not rerun LoGeR or claim method success.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.kitti_trajectory_diagnostics import (  # noqa: E402
    _apply_alignment,
    _load_kitti_gt,
    _load_tum_prediction,
    _rmse,
    _umeyama_sim3,
)


DEFAULT_GT_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        action="append",
        required=True,
        help="NAME=SEQ:PATH, e.g. kitti00_c3=00:results/.../C3_raw.tum.txt",
    )
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _finite(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _parse_trajectory_spec(spec: str) -> tuple[str, str, Path]:
    if "=" not in spec or ":" not in spec.split("=", 1)[1]:
        raise ValueError(f"Expected NAME=SEQ:PATH, got {spec!r}")
    name, rest = spec.split("=", 1)
    seq, path = rest.split(":", 1)
    name = name.strip()
    seq = seq.strip().zfill(2)
    if not name or not seq:
        raise ValueError(f"Bad trajectory spec: {spec!r}")
    return name, seq, Path(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), ensure_ascii=False)
                    if isinstance(row.get(key), (list, dict))
                    else row.get(key)
                    for key in fieldnames
                }
            )


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _frames_in_range(frames: np.ndarray, start: int, end: int) -> np.ndarray:
    mask = (frames >= int(start)) & (frames < int(end))
    return np.where(mask)[0].astype(np.int64)


def _coverage(frames: np.ndarray, start: int, end: int) -> float:
    expected = max(0, int(end) - int(start))
    if expected <= 0:
        return 0.0
    return float(_frames_in_range(frames, start, end).size / expected)


def _fit_eval(
    *,
    frames: np.ndarray,
    raw_pos: np.ndarray,
    gt_pos_all: np.ndarray,
    fit_start: int,
    fit_end: int,
    eval_start: int,
    eval_end: int,
) -> dict[str, Any]:
    fit_idx = _frames_in_range(frames, fit_start, fit_end)
    eval_idx = _frames_in_range(frames, eval_start, eval_end)
    out: dict[str, Any] = {
        "fit_start": int(fit_start),
        "fit_end": int(fit_end),
        "eval_start": int(eval_start),
        "eval_end": int(eval_end),
        "fit_n": int(fit_idx.size),
        "eval_n": int(eval_idx.size),
        "rmse_m": None,
        "mean_m": None,
        "p90_m": None,
        "sim3_scale": None,
        "valid": False,
    }
    if int(fit_idx.size) < 3 or int(eval_idx.size) < 1:
        out["reason"] = "insufficient_frames"
        return out
    try:
        scale, rot, trans = _umeyama_sim3(raw_pos[fit_idx], gt_pos_all[frames[fit_idx]], with_scale=True)
    except Exception as exc:
        out["reason"] = f"sim3_fit_failed:{type(exc).__name__}"
        return out
    aligned = (scale * (rot @ raw_pos[eval_idx].T)).T + trans
    err = np.linalg.norm(aligned - gt_pos_all[frames[eval_idx]], axis=1)
    out.update(
        {
            "rmse_m": float(_rmse(err)),
            "mean_m": float(np.mean(err)),
            "p90_m": float(np.percentile(err, 90)),
            "sim3_scale": float(scale),
            "valid": True,
        }
    )
    return out


def _local_sim3_window(
    *,
    frames: np.ndarray,
    raw_pos: np.ndarray,
    gt_pos_all: np.ndarray,
    start: int,
    end: int,
) -> dict[str, Any]:
    return _fit_eval(
        frames=frames,
        raw_pos=raw_pos,
        gt_pos_all=gt_pos_all,
        fit_start=start,
        fit_end=end,
        eval_start=start,
        eval_end=end,
    )


def _global_rmse(aligned_pos: np.ndarray, frames: np.ndarray, gt_pos_all: np.ndarray, start: int, end: int) -> float | None:
    idx = _frames_in_range(frames, start, end)
    if int(idx.size) < 1:
        return None
    err = np.linalg.norm(aligned_pos[idx] - gt_pos_all[frames[idx]], axis=1)
    return float(_rmse(err))


def _boundary_step_error(
    aligned_pos: np.ndarray,
    frames: np.ndarray,
    gt_pos_all: np.ndarray,
    boundary_frame: int,
) -> float | None:
    prev_f = int(boundary_frame) - 1
    curr_f = int(boundary_frame)
    prev_idx = np.where(frames == prev_f)[0]
    curr_idx = np.where(frames == curr_f)[0]
    if int(prev_idx.size) < 1 or int(curr_idx.size) < 1:
        return None
    pred_step = aligned_pos[int(curr_idx[0])] - aligned_pos[int(prev_idx[0])]
    gt_step = gt_pos_all[curr_f] - gt_pos_all[prev_f]
    return float(np.linalg.norm(pred_step - gt_step))


def _chunk_starts(frames: np.ndarray, chunk_size: int, stride: int, window_chunks: int) -> list[int]:
    if int(frames.size) == 0:
        return []
    min_frame = int(frames.min())
    max_frame_excl = int(frames.max()) + 1
    first = (min_frame // stride) * stride
    starts: list[int] = []
    start = first
    total_len = int(chunk_size) + max(0, int(window_chunks) - 1) * int(stride)
    while start + total_len <= max_frame_excl:
        starts.append(int(start))
        start += int(stride)
    return starts


def _scale_cv(values: Iterable[Any]) -> float | None:
    vals = [_finite(v) for v in values]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    arr = np.asarray(vals, dtype=np.float64)
    denom = max(abs(float(np.mean(arr))), 1e-12)
    return float(np.std(arr) / denom)


def _rank_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: _finite(row.get(key)) if _finite(row.get(key)) is not None else -math.inf, reverse=True)


def _evaluate_run(
    *,
    name: str,
    seq: str,
    path: Path,
    gt_root: Path,
    chunk_size: int,
    overlap: int,
    min_coverage: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    gt_path = gt_root / f"{seq}.txt"
    if not gt_path.is_file():
        raise FileNotFoundError(gt_path)
    _, gt_poses, gt_pos = _load_kitti_gt(gt_path)
    frames, raw_poses, raw_pos = _load_tum_prediction(path, gt_pos.shape[0])
    scale, rot, trans = _umeyama_sim3(raw_pos, gt_pos[frames], with_scale=True)
    aligned_poses = _apply_alignment(raw_poses, scale, rot, trans)
    aligned_pos = aligned_poses[:, :3, 3]
    stride = int(chunk_size) - int(overlap)
    if stride <= 0:
        raise ValueError(f"chunk_size must be larger than overlap, got {chunk_size=} {overlap=}")

    common = {
        "run": name,
        "sequence": seq,
        "trajectory": str(path),
        "gt": str(gt_path),
        "frame_start": int(frames.min()),
        "frame_end_exclusive": int(frames.max()) + 1,
        "frame_count": int(frames.size),
        "global_sim3_scale": float(scale),
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(overlap),
        "chunk_stride": int(stride),
    }
    all_err = np.linalg.norm(aligned_pos - gt_pos[frames], axis=1)
    summary = dict(common)
    summary.update({"global_all_rmse_m": float(_rmse(all_err))})

    single_rows: list[dict[str, Any]] = []
    for start in _chunk_starts(frames, chunk_size, stride, 1):
        end = start + int(chunk_size)
        cov = _coverage(frames, start, end)
        if cov < float(min_coverage):
            continue
        local = _local_sim3_window(frames=frames, raw_pos=raw_pos, gt_pos_all=gt_pos, start=start, end=end)
        row = dict(common)
        row.update(
            {
                "chunk_id": int(round(start / stride)),
                "chunk_start_frame": int(start),
                "chunk_end_frame": int(end),
                "coverage": cov,
                "local_sim3_rmse_m": local.get("rmse_m"),
                "local_sim3_mean_m": local.get("mean_m"),
                "local_sim3_p90_m": local.get("p90_m"),
                "local_sim3_scale": local.get("sim3_scale"),
                "global_sim3_chunk_rmse_m": _global_rmse(aligned_pos, frames, gt_pos, start, end),
            }
        )
        single_rows.append(row)

    pair_rows: list[dict[str, Any]] = []
    for start in _chunk_starts(frames, chunk_size, stride, 2):
        next_start = start + stride
        end = start + int(chunk_size) + stride
        cov = _coverage(frames, start, end)
        if cov < float(min_coverage):
            continue
        joint = _local_sim3_window(frames=frames, raw_pos=raw_pos, gt_pos_all=gt_pos, start=start, end=end)
        tail_head = _fit_eval(
            frames=frames,
            raw_pos=raw_pos,
            gt_pos_all=gt_pos,
            fit_start=next_start - int(overlap),
            fit_end=next_start,
            eval_start=next_start,
            eval_end=next_start + int(overlap),
        )
        tail_future = _fit_eval(
            frames=frames,
            raw_pos=raw_pos,
            gt_pos_all=gt_pos,
            fit_start=next_start - int(overlap),
            fit_end=next_start,
            eval_start=next_start,
            eval_end=end,
        )
        row = dict(common)
        row.update(
            {
                "chunk_pair": f"{int(round(start / stride))}-{int(round(next_start / stride))}",
                "start_chunk_id": int(round(start / stride)),
                "end_chunk_id": int(round(next_start / stride)),
                "pair_start_frame": int(start),
                "pair_end_frame": int(end),
                "boundary_frame": int(next_start),
                "coverage": cov,
                "pair_joint_sim3_rmse_m": joint.get("rmse_m"),
                "pair_joint_sim3_scale": joint.get("sim3_scale"),
                "global_sim3_pair_rmse_m": _global_rmse(aligned_pos, frames, gt_pos, start, end),
                "boundary_step_error_global_sim3_m": _boundary_step_error(aligned_pos, frames, gt_pos, next_start),
                "tail3_to_head3_sim3_rmse_m": tail_head.get("rmse_m"),
                "tail3_to_future_from_boundary_sim3_rmse_m": tail_future.get("rmse_m"),
                "overlap_fit_n": tail_head.get("fit_n"),
                "overlap_eval_n": tail_head.get("eval_n"),
            }
        )
        pair_rows.append(row)

    window5_rows: list[dict[str, Any]] = []
    for start in _chunk_starts(frames, chunk_size, stride, 5):
        end = start + int(chunk_size) + 4 * stride
        cov = _coverage(frames, start, end)
        if cov < float(min_coverage):
            continue
        joint = _local_sim3_window(frames=frames, raw_pos=raw_pos, gt_pos_all=gt_pos, start=start, end=end)
        scales: list[float | None] = []
        for offset in range(5):
            cstart = start + offset * stride
            cend = cstart + int(chunk_size)
            cmetric = _local_sim3_window(frames=frames, raw_pos=raw_pos, gt_pos_all=gt_pos, start=cstart, end=cend)
            scales.append(_finite(cmetric.get("sim3_scale")))
        row = dict(common)
        row.update(
            {
                "window_chunks": "-".join(str(int(round((start + i * stride) / stride))) for i in range(5)),
                "start_chunk_id": int(round(start / stride)),
                "end_chunk_id": int(round((start + 4 * stride) / stride)),
                "window_start_frame": int(start),
                "window_end_frame": int(end),
                "coverage": cov,
                "window5_joint_sim3_rmse_m": joint.get("rmse_m"),
                "window5_joint_sim3_scale": joint.get("sim3_scale"),
                "global_sim3_window5_rmse_m": _global_rmse(aligned_pos, frames, gt_pos, start, end),
                "window5_subchunk_scale_values": scales,
                "window5_subchunk_scale_cv": _scale_cv(scales),
            }
        )
        window5_rows.append(row)

    return single_rows, pair_rows, window5_rows, summary


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_single: list[dict[str, Any]] = []
    all_pairs: list[dict[str, Any]] = []
    all_window5: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    specs = [_parse_trajectory_spec(spec) for spec in args.trajectory]
    for name, seq, path in specs:
        single, pairs, window5, summary = _evaluate_run(
            name=name,
            seq=seq,
            path=path,
            gt_root=args.gt_root,
            chunk_size=int(args.chunk_size),
            overlap=int(args.chunk_overlap),
            min_coverage=float(args.min_coverage),
        )
        all_single.extend(single)
        all_pairs.extend(pairs)
        all_window5.extend(window5)
        summaries.append(summary)

    single_ranked = _rank_rows(all_single, "local_sim3_rmse_m")
    pair_ranked = _rank_rows(all_pairs, "tail3_to_future_from_boundary_sim3_rmse_m")
    window5_ranked = _rank_rows(all_window5, "window5_joint_sim3_rmse_m")

    _write_csv(args.out_dir / "bad_single_chunk_table.csv", single_ranked)
    _write_csv(args.out_dir / "bad_adjacent_chunk_pair_table.csv", pair_ranked)
    _write_csv(args.out_dir / "bad_5chunk_window_table.csv", window5_ranked)
    _write_json(
        args.out_dir / "bad_window_summary.json",
        {
            "schema": "v78_bad_window_tables_v1",
            "diagnostic_only": True,
            "inputs": [
                {"name": name, "sequence": seq, "trajectory": str(path)}
                for name, seq, path in specs
            ],
            "chunk_size": int(args.chunk_size),
            "chunk_overlap": int(args.chunk_overlap),
            "min_coverage": float(args.min_coverage),
            "summaries": summaries,
            "top_single_chunks": single_ranked[: int(args.top_k)],
            "top_adjacent_chunk_pairs": pair_ranked[: int(args.top_k)],
            "top_5chunk_windows": window5_ranked[: int(args.top_k)],
            "ranking_keys": {
                "bad_single_chunk_table.csv": "local_sim3_rmse_m desc",
                "bad_adjacent_chunk_pair_table.csv": "tail3_to_future_from_boundary_sim3_rmse_m desc",
                "bad_5chunk_window_table.csv": "window5_joint_sim3_rmse_m desc",
            },
        },
    )

    print(f"wrote_csv={args.out_dir / 'bad_single_chunk_table.csv'} rows={len(single_ranked)}")
    print(f"wrote_csv={args.out_dir / 'bad_adjacent_chunk_pair_table.csv'} rows={len(pair_ranked)}")
    print(f"wrote_csv={args.out_dir / 'bad_5chunk_window_table.csv'} rows={len(window5_ranked)}")
    print(f"wrote_json={args.out_dir / 'bad_window_summary.json'}")


if __name__ == "__main__":
    main()
