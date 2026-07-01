#!/usr/bin/env python3
"""Build chunk-prefix-causal DA3-D4RT Sim3 alignment scores for v99."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STREAM3D_ROOT = ROOT / "Stream3D"
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))
if str(STREAM3D_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT / "tools"))

from geometry_provider.common import fit_transform  # noqa: E402
from stream4d_native.sim3 import apply_sim3_to_xyz  # noqa: E402
from tools.build_v99_phase10aa_da3_d4rt_sim3_alignment_visual import (  # noqa: E402
    DA3FrameCache,
    _load_manifest,
    _residual_stats,
)


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase10ag_prefix_da3_d4rt_sim3_alignment"
SCENES = {
    "scene0011_00": {
        "da3_root": AUDIT_ROOT / "v99_phase10s_da3_holdout_chunk32o3_scene0011_base",
        "da3_manifest": AUDIT_ROOT / "v99_phase10s_da3_holdout_chunk32o3_scene0011_input/frame_manifest_rows.csv",
        "d4rt_root": AUDIT_ROOT / "v99_phase10ad_d4rt_da3grid_stitched_scene0011",
    },
    "scene0050_00": {
        "da3_root": AUDIT_ROOT / "v99_phase10s_da3_holdout_chunk32o3_scene0050_base",
        "da3_manifest": AUDIT_ROOT / "v99_phase10s_da3_holdout_chunk32o3_scene0050_input/frame_manifest_rows.csv",
        "d4rt_root": AUDIT_ROOT / "v99_phase10ad_d4rt_da3grid_stitched_scene0050",
    },
}


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _jsonable(row.get(field, "")) for field in fields})


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _window_index(window_id: str) -> int:
    match = re.search(r"(\d+)$", str(window_id))
    return int(match.group(1)) if match else 0


def _read_window_frames(d4rt_root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    path = d4rt_root / "micro_track_quality_rows.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frames = [int(float(tok)) for tok in str(row.get("frame_ids", "")).replace(" ", ",").split(",") if tok.strip()]
            out[row.get("window_id", "")] = {
                "frame_min": min(frames) if frames else "",
                "frame_max": max(frames) if frames else "",
                "frame_count": len(frames),
                "frame_ids": frames,
            }
    return out


def _append_reservoir(
    reservoir: dict[str, list[np.ndarray]],
    window_id: str,
    arrays: list[np.ndarray],
    cap: int,
    rng: np.random.Generator,
) -> None:
    if arrays[0].shape[0] == 0:
        return
    existing = reservoir.get(window_id)
    merged = [np.asarray(value) for value in arrays] if existing is None else [np.concatenate([old, new], axis=0) for old, new in zip(existing, arrays)]
    if int(cap) > 0 and merged[0].shape[0] > int(cap):
        keep = rng.choice(merged[0].shape[0], size=int(cap), replace=False)
        keep.sort()
        merged = [value[keep] for value in merged]
    reservoir[window_id] = merged


def _collect_scene_samples(
    *,
    scene_id: str,
    d4rt_root: Path,
    da3_cache: DA3FrameCache,
    frame_to_da3: dict[int, int],
    args: argparse.Namespace,
) -> tuple[dict[str, list[np.ndarray]], dict[str, Any]]:
    usecols = [
        "scene_id",
        "window_id",
        "target_frame_id",
        "u_tgt",
        "v_tgt",
        "x_3d",
        "y_3d",
        "z_3d",
        "visibility",
        "confidence",
        "uv_in01",
        "overlap_stitch_applied",
        "geometry_coordinate_mode",
        "d4rt_output_width",
        "d4rt_output_height",
    ]
    reservoir: dict[str, list[np.ndarray]] = {}
    rng = np.random.default_rng(int(args.seed) + _window_index(scene_id))
    rows_seen = 0
    rows_candidate = 0
    rows_backprojected = 0
    rows_da3_valid = 0
    frame_set = set(int(v) for v in frame_to_da3)
    for chunk in pd.read_csv(d4rt_root / "micro_track_rows.csv", usecols=usecols, chunksize=int(args.chunksize)):
        rows_seen += int(chunk.shape[0])
        chunk = chunk[chunk["scene_id"].astype(str) == str(scene_id)].copy()
        if chunk.empty:
            continue
        numeric = chunk[["target_frame_id", "u_tgt", "v_tgt", "x_3d", "y_3d", "z_3d", "visibility", "confidence"]].apply(pd.to_numeric, errors="coerce")
        finite = np.isfinite(numeric.to_numpy(dtype=np.float64)).all(axis=1)
        keep = (
            finite
            & chunk["uv_in01"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
            & chunk["overlap_stitch_applied"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
            & chunk["geometry_coordinate_mode"].astype(str).eq("d4rt_overlap_self_stitched_no_final_gt_sim3").to_numpy()
            & numeric["target_frame_id"].astype(np.int64).isin(frame_set).to_numpy()
            & (numeric["visibility"].to_numpy(dtype=np.float64) >= float(args.min_visibility))
            & (numeric["confidence"].to_numpy(dtype=np.float64) >= float(args.min_confidence))
        )
        if not np.any(keep):
            continue
        kept = chunk.loc[keep, ["window_id"]].reset_index(drop=True)
        vals = numeric.loc[keep].reset_index(drop=True)
        rows_candidate += int(vals.shape[0])
        for (window_id, frame_id), idx in kept.assign(target_frame_id=vals["target_frame_id"].astype(np.int64)).groupby(["window_id", "target_frame_id"]).groups.items():
            idx_arr = np.asarray(list(idx), dtype=np.int64)
            if idx_arr.size > int(args.max_pairs_per_frame_per_chunk):
                idx_arr = rng.choice(idx_arr, size=int(args.max_pairs_per_frame_per_chunk), replace=False)
                idx_arr.sort()
            frame_vals = vals.iloc[idx_arr]
            xy = frame_vals[["u_tgt", "v_tgt"]].to_numpy(dtype=np.float64)
            rows_backprojected += int(xy.shape[0])
            da3_world, _colors, da3_ok = da3_cache.backproject(int(frame_id), xy, conf_min=float(args.da3_conf_min))
            if not np.any(da3_ok):
                continue
            d4rt_xyz = frame_vals[["x_3d", "y_3d", "z_3d"]].to_numpy(dtype=np.float32)[da3_ok]
            da3_xyz = da3_world[da3_ok].astype(np.float32)
            frames = np.full((d4rt_xyz.shape[0],), int(frame_id), dtype=np.int32)
            rows_da3_valid += int(d4rt_xyz.shape[0])
            _append_reservoir(
                reservoir,
                str(window_id),
                [d4rt_xyz, da3_xyz, frames],
                cap=int(args.max_pairs_per_window),
                rng=rng,
            )
    return reservoir, {
        "rows_seen": rows_seen,
        "rows_candidate_after_filters": rows_candidate,
        "rows_backprojected_sampled": rows_backprojected,
        "rows_with_valid_da3_depth": rows_da3_valid,
        "sampled_window_count": len(reservoir),
    }


def _sample_prefix(arrays: list[np.ndarray], cap: int, seed: int) -> list[np.ndarray]:
    if not arrays or arrays[0].shape[0] <= int(cap) or int(cap) <= 0:
        return arrays
    rng = np.random.default_rng(int(seed))
    keep = rng.choice(arrays[0].shape[0], size=int(cap), replace=False)
    keep.sort()
    return [value[keep] for value in arrays]


def _fit_scene_prefix(
    *,
    scene_id: str,
    samples: dict[str, list[np.ndarray]],
    window_frames: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prefix_parts: list[list[np.ndarray]] = []
    for window_id in sorted(window_frames, key=_window_index):
        arrays = samples.get(window_id)
        if arrays is not None:
            prefix_parts.append(arrays)
        info = window_frames[window_id]
        if not prefix_parts:
            rows.append(_alignment_row(scene_id, window_id, info, None, None, None, args, reason="no_prefix_samples"))
            continue
        prefix = [np.concatenate([part[i] for part in prefix_parts], axis=0) for i in range(3)]
        prefix = _sample_prefix(prefix, int(args.max_prefix_fit_pairs), int(args.seed) + _window_index(window_id))
        current = samples.get(window_id)
        if prefix[0].shape[0] < int(args.min_fit_pairs) or current is None or current[0].shape[0] < int(args.min_fit_pairs):
            rows.append(_alignment_row(scene_id, window_id, info, None, prefix, current, args, reason="insufficient_pairs"))
            continue
        fit = fit_transform(
            prefix[0],
            prefix[1],
            robust_trim_percentile=float(args.robust_trim_percentile),
        )
        if fit is None:
            rows.append(_alignment_row(scene_id, window_id, info, None, prefix, current, args, reason="fit_failed"))
            continue
        rows.append(_alignment_row(scene_id, window_id, info, fit, prefix, current, args, reason="ok"))
    return rows


def _alignment_row(
    scene_id: str,
    window_id: str,
    info: dict[str, Any],
    fit: Any,
    prefix: list[np.ndarray] | None,
    current: list[np.ndarray] | None,
    args: argparse.Namespace,
    *,
    reason: str,
) -> dict[str, Any]:
    prefix_count = int(prefix[0].shape[0]) if prefix is not None else 0
    current_count = int(current[0].shape[0]) if current is not None else 0
    score = 0.0
    aligned_stats = _residual_stats(np.asarray([], dtype=np.float64))
    fit_stats = _residual_stats(np.asarray([], dtype=np.float64))
    scale = ""
    det = ""
    trans_norm = ""
    if fit is not None and current is not None and current[0].shape[0] > 0:
        aligned = apply_sim3_to_xyz(current[0], transform=fit)
        residual = np.linalg.norm(aligned.astype(np.float64) - current[1].astype(np.float64), axis=1)
        aligned_stats = _residual_stats(residual)
        fit_residual = np.asarray(fit.get("residual", []), dtype=np.float64)
        fit_stats = _residual_stats(fit_residual)
        scale = float(fit["scale"])
        det = float(np.linalg.det(np.asarray(fit["rot"], dtype=np.float64)))
        trans_norm = float(np.linalg.norm(np.asarray(fit["trans"], dtype=np.float64)))
        p90 = float(aligned_stats["p90"])
        if np.isfinite(p90):
            score = float(math.exp(-p90 / max(1e-12, float(args.alignment_sigma_m))))
    frame_max = int(info["frame_max"]) if info.get("frame_max") != "" else -1
    prefix_frame_max = int(np.max(prefix[2])) if prefix is not None and prefix[2].shape[0] else -1
    return {
        "schema_version": "stream4d_v99_phase10ag_prefix_da3_d4rt_sim3_alignment_row_v1",
        "phase_id": "v99_phase10ag_prefix_da3_d4rt_sim3_alignment",
        "scene_id": scene_id,
        "window_id": window_id,
        "window_index": _window_index(window_id),
        "chunk_frame_min": info.get("frame_min", ""),
        "chunk_frame_max": info.get("frame_max", ""),
        "chunk_frame_count": info.get("frame_count", ""),
        "prefix_anchor_count": prefix_count,
        "current_window_anchor_count": current_count,
        "cross_model_sim3_anchor_frame_max": prefix_frame_max,
        "cross_model_sim3_anchor_frame_max_le_chunk_frame_max": bool(prefix_frame_max <= frame_max if frame_max >= 0 and prefix_frame_max >= 0 else False),
        "sim3_fit_status": reason,
        "sim3_scale_d4rt_to_da3": scale,
        "sim3_rotation_det": det,
        "sim3_translation_norm": trans_norm,
        "fit_residual_p50_m": fit_stats["p50"],
        "fit_residual_p90_m": fit_stats["p90"],
        "aligned_residual_p50_m": aligned_stats["p50"],
        "aligned_residual_p90_m": aligned_stats["p90"],
        "aligned_residual_p95_m": aligned_stats["p95"],
        "alignment_sigma_m": float(args.alignment_sigma_m),
        "alignment_score": score,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    scene_summaries: list[dict[str, Any]] = []
    for scene_id, spec in SCENES.items():
        manifest = _load_manifest(Path(spec["da3_manifest"]), scene_id=scene_id)
        frame_to_da3 = {int(row.frame_id): int(row.da3_frame_index) for row in manifest.itertuples(index=False)}
        da3_cache = DA3FrameCache(Path(spec["da3_root"]), manifest)
        window_frames = _read_window_frames(Path(spec["d4rt_root"]))
        samples, sample_info = _collect_scene_samples(
            scene_id=scene_id,
            d4rt_root=Path(spec["d4rt_root"]),
            da3_cache=da3_cache,
            frame_to_da3=frame_to_da3,
            args=args,
        )
        rows = _fit_scene_prefix(scene_id=scene_id, samples=samples, window_frames=window_frames, args=args)
        all_rows.extend(rows)
        scene_summaries.append(
            {
                "scene_id": scene_id,
                "da3_root": _rel(spec["da3_root"]),
                "da3_manifest": _rel(spec["da3_manifest"]),
                "d4rt_root": _rel(spec["d4rt_root"]),
                "window_count": len(window_frames),
                "fit_ok_count": sum(1 for row in rows if row["sim3_fit_status"] == "ok"),
                "min_alignment_score": min((float(row["alignment_score"]) for row in rows), default=0.0),
                "mean_alignment_score": float(np.mean([float(row["alignment_score"]) for row in rows])) if rows else 0.0,
                "max_alignment_score": max((float(row["alignment_score"]) for row in rows), default=0.0),
                **sample_info,
            }
        )
    _write_csv(output_root / "window_alignment_rows.csv", all_rows)
    gate_pass = (
        bool(all_rows)
        and all(row["sim3_fit_status"] == "ok" for row in all_rows)
        and all(bool(row["cross_model_sim3_anchor_frame_max_le_chunk_frame_max"]) for row in all_rows)
    )
    summary = {
        "schema_version": "stream4d_v99_phase10ag_prefix_da3_d4rt_sim3_alignment_summary_v1",
        "phase_id": "v99_phase10ag_prefix_da3_d4rt_sim3_alignment",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": "PASS_PREFIX_CAUSAL_DA3_D4RT_SIM3_ALIGNMENT_SCORES" if gate_pass else "NO_GO_PREFIX_CAUSAL_DA3_D4RT_SIM3_ALIGNMENT_SCORES",
        "prefix_causal_alignment_gate_pass": bool(gate_pass),
        "chunk_size": 32,
        "overlap": 3,
        "alignment_score_definition": "exp(-aligned_residual_p90_m / alignment_sigma_m)",
        "alignment_sigma_m": float(args.alignment_sigma_m),
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "scene_summaries": scene_summaries,
        "outputs": {
            "summary": _rel(output_root / "summary.json"),
            "window_alignment_rows": _rel(output_root / "window_alignment_rows.csv"),
        },
        "runtime_sec": float(time.time() - t0),
    }
    _write_json(output_root / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUT_DIR))
    parser.add_argument("--chunksize", type=int, default=200000)
    parser.add_argument("--max-pairs-per-frame-per-chunk", type=int, default=256)
    parser.add_argument("--max-pairs-per-window", type=int, default=8000)
    parser.add_argument("--max-prefix-fit-pairs", type=int, default=120000)
    parser.add_argument("--min-fit-pairs", type=int, default=128)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--da3-conf-min", type=float, default=0.0)
    parser.add_argument("--robust-trim-percentile", type=float, default=90.0)
    parser.add_argument("--alignment-sigma-m", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=9917)
    args = parser.parse_args()
    summary = build(args)
    return 0 if bool(summary.get("prefix_causal_alignment_gate_pass")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
