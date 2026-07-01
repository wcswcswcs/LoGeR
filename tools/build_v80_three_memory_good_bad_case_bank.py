#!/usr/bin/env python3
"""Build ACL2 v80 three-memory good/bad case bank.

This tool is diagnostic and audit-oriented. It recomputes real trajectory/GT
metrics for short, mid, and long memory units, enriches selected cases with
semantic/RADIO evidence when present, and leaves unavailable fields empty rather
than synthesizing metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from kitti_trajectory_diagnostics import (  # noqa: E402
    _apply_alignment,
    _load_kitti_gt,
    _load_tum_prediction,
    _rmse,
    _umeyama_sim3,
)


DEFAULT_PHASE0_DIR = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final/phase0_multiseq_artifact_audit"
)
DEFAULT_OUT_DIR = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final/phase1_three_memory_case_bank"
)
DEFAULT_KITTI_POSES_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses")
DEFAULT_PREPROCESS_ROOT = Path("results/kitti_preprocess")
DEFAULT_RESULTS_ROOT = Path("results")

STABLE_WORDS = (
    "building",
    "house",
    "wall",
    "fence",
    "handrail_or_fence",
    "pole",
    "traffic sign",
    "traffic light",
    "bridge",
    "construction",
    "billboard",
    "pillar",
    "stair",
)
DYNAMIC_WORDS = ("car", "person", "rider", "bicycle", "motorcycle", "bus", "truck", "train", "dog")
LOWTRUST_WORDS = ("tree", "grass", "vegetation", "mountain", "terrain", "void", "unknown", "plant")
CONTEXT_WORDS = ("sky", "road", "ground", "sidewalk", "path", "crosswalk")

SHORT_FIELDS = (
    "seq",
    "chunk_id",
    "frame_start",
    "frame_end",
    "case_type",
    "local_sim3_ate",
    "head_to_tail",
    "scale_cv",
    "intra_scale_variance",
    "stable_mass",
    "harm_mass",
    "context_mass",
    "thing_moving_ratio",
    "thing_static_ratio",
    "stuff_stable_ratio",
    "lowtrust_stuff_ratio",
    "RADIO_boundary_ratio",
    "RADIO_temporal_stability",
    "READ_attention_entropy",
    "global_attn_layer_candidate",
    "case_reason",
)
MID_FIELDS = (
    "seq",
    "prev_chunk",
    "curr_chunk",
    "frame_start",
    "frame_end",
    "case_type",
    "future_after_overlap",
    "boundary_jump",
    "raw_overlap_residual",
    "overlap_semantic_agreement",
    "stable_overlap_mass",
    "harm_overlap_mass",
    "context_overlap_mass",
    "same_object_overlap_ratio",
    "cross_object_boundary_ratio",
    "V_alignment_delta",
    "K_risk_delta",
    "SWA_gate_mass",
    "SWA_replace_mass",
    "case_reason",
)
LONG_FIELDS = (
    "seq",
    "chunk_start",
    "chunk_end",
    "frame_start",
    "frame_end",
    "case_type",
    "window5_joint_sim3_rmse",
    "window5_subchunk_scale_cv",
    "downstream_future_consistency",
    "low_observability_score",
    "regime_shift_score",
    "shadow_exposure_change",
    "road_edge_continuity",
    "corridor_stability",
    "TTT_update_conflict",
    "post_zp_delta",
    "case_reason",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def parse_seqs(text: str | Sequence[str]) -> list[str]:
    if isinstance(text, str):
        return [part.strip().zfill(2) for part in text.split(",") if part.strip()]
    return [str(part).zfill(2) for part in text]


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = list(fieldnames or [])
    if not keys:
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), ensure_ascii=False)
                    if isinstance(row.get(key), (list, dict))
                    else row.get(key)
                    for key in keys
                }
            )


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def count_tum_rows(path: Path) -> tuple[int, int | None, int | None]:
    frames: list[int] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue
                frames.append(int(round(float(parts[0]))))
    except OSError:
        return 0, None, None
    if not frames:
        return 0, None, None
    return len(frames), min(frames), max(frames)


def preference_score(path: Path) -> int:
    text = str(path).lower()
    score = 0
    for word, weight in (
        ("geom", 60),
        ("native", 55),
        ("h35", 40),
        ("base", 35),
        ("cross_sequence_full", 30),
        ("full", 25),
        ("c3", 12),
    ):
        if word in text:
            score += weight
    for word, penalty in (
        ("random", 80),
        ("shuffled", 80),
        ("semconf", 30),
        ("semantic", 10),
    ):
        if word in text:
            score -= penalty
    return score


def discover_trajectories(results_root: Path, seq: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in results_root.rglob(f"{seq}.txt"):
        if "code_audit_pack" in path.parts or not path.is_file():
            continue
        rows, frame_min, frame_max = count_tum_rows(path)
        if rows < 3:
            continue
        out.append(
            {
                "seq": seq,
                "trajectory": str(path),
                "frame_count": rows,
                "frame_min": frame_min,
                "frame_max": frame_max,
                "preference_score": preference_score(path),
                "mtime": path.stat().st_mtime,
            }
        )
    return sorted(
        out,
        key=lambda row: (
            int(row["frame_count"]),
            int(row["preference_score"]),
            float(row["mtime"]),
        ),
        reverse=True,
    )


def frames_in_range(frames: np.ndarray, start: int, end: int) -> np.ndarray:
    return np.where((frames >= int(start)) & (frames < int(end)))[0].astype(np.int64)


def coverage(frames: np.ndarray, start: int, end: int) -> float:
    expected = max(0, int(end) - int(start))
    if expected <= 0:
        return 0.0
    return float(frames_in_range(frames, start, end).size / expected)


def fit_eval(
    frames: np.ndarray,
    raw_pos: np.ndarray,
    gt_pos: np.ndarray,
    fit_start: int,
    fit_end: int,
    eval_start: int,
    eval_end: int,
) -> dict[str, Any]:
    fit_idx = frames_in_range(frames, fit_start, fit_end)
    eval_idx = frames_in_range(frames, eval_start, eval_end)
    out: dict[str, Any] = {
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
        scale, rot, trans = _umeyama_sim3(raw_pos[fit_idx], gt_pos[frames[fit_idx]], with_scale=True)
    except Exception as exc:  # pragma: no cover - diagnostic path
        out["reason"] = f"sim3_fit_failed:{type(exc).__name__}"
        return out
    aligned = (scale * (rot @ raw_pos[eval_idx].T)).T + trans
    err = np.linalg.norm(aligned - gt_pos[frames[eval_idx]], axis=1)
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


def global_rmse(aligned_pos: np.ndarray, frames: np.ndarray, gt_pos: np.ndarray, start: int, end: int) -> float | None:
    idx = frames_in_range(frames, start, end)
    if int(idx.size) < 1:
        return None
    err = np.linalg.norm(aligned_pos[idx] - gt_pos[frames[idx]], axis=1)
    return float(_rmse(err))


def boundary_step_error(aligned_pos: np.ndarray, frames: np.ndarray, gt_pos: np.ndarray, boundary_frame: int) -> float | None:
    prev_idx = np.where(frames == int(boundary_frame) - 1)[0]
    curr_idx = np.where(frames == int(boundary_frame))[0]
    if int(prev_idx.size) < 1 or int(curr_idx.size) < 1:
        return None
    pred_step = aligned_pos[int(curr_idx[0])] - aligned_pos[int(prev_idx[0])]
    gt_step = gt_pos[int(boundary_frame)] - gt_pos[int(boundary_frame) - 1]
    return float(np.linalg.norm(pred_step - gt_step))


def scale_cv(values: Iterable[Any]) -> float | None:
    vals = [finite(value) for value in values]
    vals = [value for value in vals if value is not None]
    if not vals:
        return None
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.std(arr) / max(abs(float(np.mean(arr))), 1e-12))


def scale_variance(values: Iterable[Any]) -> float | None:
    vals = [finite(value) for value in values]
    vals = [value for value in vals if value is not None]
    return float(np.var(np.asarray(vals, dtype=np.float64))) if vals else None


def chunk_starts(frames: np.ndarray, chunk_size: int, stride: int, window_chunks: int) -> list[int]:
    if int(frames.size) == 0:
        return []
    first = (int(frames.min()) // stride) * stride
    max_frame_excl = int(frames.max()) + 1
    total_len = int(chunk_size) + max(0, int(window_chunks) - 1) * int(stride)
    starts: list[int] = []
    start = first
    while start + total_len <= max_frame_excl:
        starts.append(int(start))
        start += int(stride)
    return starts


def p75_norm(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    vals = [finite(row.get(key)) for row in rows]
    vals = [value for value in vals if value is not None]
    if not vals:
        return None
    denom = float(np.percentile(np.asarray(vals, dtype=np.float64), 75))
    return denom if abs(denom) > 1e-12 and math.isfinite(denom) else None


def add_weighted_score(
    rows: list[dict[str, Any]],
    score_key: str,
    terms: Sequence[tuple[str, float]],
) -> dict[str, Any]:
    denoms = {key: p75_norm(rows, key) for key, _weight in terms}
    for row in rows:
        parts: list[float] = []
        missing: list[str] = []
        for key, weight in terms:
            val = finite(row.get(key))
            denom = denoms.get(key)
            if val is None or denom is None:
                missing.append(key)
                continue
            row[f"{key}_p75_norm"] = float(val / denom)
            parts.append(float(weight) * float(val / denom))
        if missing:
            row[score_key] = None
            row[f"{score_key}_status"] = "missing:" + ",".join(missing)
        else:
            row[score_key] = float(sum(parts))
            row[f"{score_key}_status"] = "computed_from_global_p75_normalized_terms"
    return {"denominators": denoms, "terms": [{"field": key, "weight": weight} for key, weight in terms]}


def evaluate_trajectory(
    name: str,
    seq: str,
    path: Path,
    gt_root: Path,
    chunk_size: int,
    overlap: int,
    min_coverage: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    gt_path = gt_root / f"{seq}.txt"
    _, _gt_poses, gt_pos = _load_kitti_gt(gt_path)
    frames, raw_poses, raw_pos = _load_tum_prediction(path, gt_pos.shape[0])
    scale, rot, trans = _umeyama_sim3(raw_pos, gt_pos[frames], with_scale=True)
    aligned_poses = _apply_alignment(raw_poses, scale, rot, trans)
    aligned_pos = aligned_poses[:, :3, 3]
    stride = int(chunk_size) - int(overlap)
    if stride <= 0:
        raise ValueError(f"chunk_size must be larger than overlap, got {chunk_size=} {overlap=}")
    common = {
        "seq": seq,
        "run_name": name,
        "trajectory": str(path),
        "gt_path": str(gt_path),
        "trajectory_frame_min": int(frames.min()),
        "trajectory_frame_max": int(frames.max()),
        "trajectory_frame_count": int(frames.size),
        "global_sim3_scale": float(scale),
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(overlap),
        "chunk_stride": int(stride),
        "metric_source": "trajectory_gt_recomputed_sim3",
    }
    all_err = np.linalg.norm(aligned_pos - gt_pos[frames], axis=1)
    summary = {**common, "global_all_rmse_m": float(_rmse(all_err))}

    short_rows: list[dict[str, Any]] = []
    chunk_scale_by_id: dict[int, float | None] = {}
    head_len = min(10, int(chunk_size))
    third = max(3, int(chunk_size) // 3)
    for start in chunk_starts(frames, chunk_size, stride, 1):
        end = start + int(chunk_size)
        cov = coverage(frames, start, end)
        if cov < float(min_coverage):
            continue
        chunk_id = int(round(start / stride))
        whole = fit_eval(frames, raw_pos, gt_pos, start, end, start, end)
        head_to_tail = fit_eval(frames, raw_pos, gt_pos, start, start + head_len, end - head_len, end)
        scales = []
        for offset in (0, third, 2 * third):
            part_start = min(start + offset, end - 3)
            part_end = min(part_start + third, end)
            part = fit_eval(frames, raw_pos, gt_pos, part_start, part_end, part_start, part_end)
            scales.append(part.get("sim3_scale"))
        chunk_scale_by_id[chunk_id] = finite(whole.get("sim3_scale"))
        short_rows.append(
            {
                **common,
                "chunk_id": chunk_id,
                "frame_start": int(start),
                "frame_end": int(end),
                "coverage": cov,
                "local_sim3_ate": whole.get("rmse_m"),
                "local_sim3_mean": whole.get("mean_m"),
                "local_sim3_p90": whole.get("p90_m"),
                "head_to_tail": head_to_tail.get("rmse_m"),
                "scale_cv": scale_cv(scales),
                "intra_scale_variance": scale_variance(scales),
                "global_chunk_ate": global_rmse(aligned_pos, frames, gt_pos, start, end),
            }
        )

    mid_rows: list[dict[str, Any]] = []
    for start in chunk_starts(frames, chunk_size, stride, 2):
        curr_start = start + stride
        end = start + int(chunk_size) + stride
        cov = coverage(frames, start, end)
        if cov < float(min_coverage):
            continue
        prev_chunk = int(round(start / stride))
        curr_chunk = int(round(curr_start / stride))
        tail_future = fit_eval(frames, raw_pos, gt_pos, curr_start - int(overlap), curr_start, curr_start, end)
        tail_head = fit_eval(
            frames,
            raw_pos,
            gt_pos,
            curr_start - int(overlap),
            curr_start,
            curr_start,
            curr_start + int(overlap),
        )
        mid_rows.append(
            {
                **common,
                "prev_chunk": prev_chunk,
                "curr_chunk": curr_chunk,
                "frame_start": int(start),
                "frame_end": int(end),
                "boundary_frame": int(curr_start),
                "coverage": cov,
                "future_after_overlap": tail_future.get("rmse_m"),
                "boundary_jump": boundary_step_error(aligned_pos, frames, gt_pos, curr_start),
                "raw_overlap_residual": tail_head.get("rmse_m"),
                "raw_overlap_residual_source": "tail_overlap_to_head_overlap_sim3_proxy",
                "scale_cv": scale_cv([chunk_scale_by_id.get(prev_chunk), chunk_scale_by_id.get(curr_chunk)]),
                "global_pair_ate": global_rmse(aligned_pos, frames, gt_pos, start, end),
            }
        )

    long_rows: list[dict[str, Any]] = []
    for start in chunk_starts(frames, chunk_size, stride, 5):
        end = start + int(chunk_size) + 4 * stride
        cov = coverage(frames, start, end)
        if cov < float(min_coverage):
            continue
        chunk_start = int(round(start / stride))
        chunk_end = int(round((start + 4 * stride) / stride))
        joint = fit_eval(frames, raw_pos, gt_pos, start, end, start, end)
        downstream = fit_eval(frames, raw_pos, gt_pos, start, end, end, end + stride)
        scales = [
            chunk_scale_by_id.get(int(round((start + offset * stride) / stride)))
            for offset in range(5)
        ]
        long_rows.append(
            {
                **common,
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "frame_start": int(start),
                "frame_end": int(end),
                "coverage": cov,
                "window5_joint_sim3_rmse": joint.get("rmse_m"),
                "window5_subchunk_scale_cv": scale_cv(scales),
                "window5_subchunk_scale_values": scales,
                "downstream_future_consistency": downstream.get("rmse_m"),
                "downstream_future_eval_n": downstream.get("eval_n"),
                "global_window5_ate": global_rmse(aligned_pos, frames, gt_pos, start, end),
            }
        )
    return short_rows, mid_rows, long_rows, summary


def find_chunk_dir(root: Path, chunk_id: int) -> Path | None:
    if not root.is_dir():
        return None
    matches = sorted(root.glob(f"chunk_{int(chunk_id):03d}_*"))
    return matches[0] if matches else None


def label_ids(label_names: Sequence[Any], words: Sequence[str]) -> set[int]:
    out: set[int] = set()
    lowered = [str(name).lower() for name in label_names]
    for idx, name in enumerate(lowered):
        if any(word in name for word in words):
            out.add(idx)
    return out


def semantic_features(preprocess_root: Path, seq: str, chunk_id: int) -> dict[str, Any]:
    stage_dir = preprocess_root / seq / "stage_c_cache_semantic_chunks"
    chunk_dir = find_chunk_dir(stage_dir, int(chunk_id))
    out: dict[str, Any] = {
        "semantic_available": False,
        "semantic_role_source": "stage_c_label_map_word_groups_v80",
        "stable_mass": None,
        "harm_mass": None,
        "context_mass": None,
        "thing_moving_ratio": None,
        "thing_static_ratio": None,
        "stuff_stable_ratio": None,
        "lowtrust_stuff_ratio": None,
        "semantic_confidence_mean": None,
        "semantic_histogram": {},
    }
    if chunk_dir is None:
        out["semantic_error"] = "missing_stage_c_chunk_dir"
        return out
    path = chunk_dir / "masklet.pt"
    if not path.is_file():
        out["semantic_error"] = "missing_masklet_pt"
        return out
    try:
        payload = torch.load(path, map_location="cpu")
        sem = payload.get("semantic_segmentation") if isinstance(payload, dict) else None
    except Exception as exc:  # pragma: no cover - diagnostic path
        out["semantic_error"] = type(exc).__name__
        return out
    if not isinstance(sem, dict) or not hasattr(sem.get("label_maps"), "detach"):
        out["semantic_error"] = "missing_label_maps"
        return out
    labels = sem["label_maps"].detach().cpu().numpy()
    label_names = sem.get("label_names", [])
    if not isinstance(label_names, list):
        label_names = []
    total = max(int(labels.size), 1)
    stable_ids = label_ids(label_names, STABLE_WORDS)
    dynamic_ids = label_ids(label_names, DYNAMIC_WORDS)
    lowtrust_ids = label_ids(label_names, LOWTRUST_WORDS)
    context_ids = label_ids(label_names, CONTEXT_WORDS)
    void_ids = {0}
    all_special = stable_ids | dynamic_ids | lowtrust_ids | context_ids | void_ids
    present = set(int(x) for x in np.unique(labels))
    static_thing_ids = {idx for idx in present if idx not in all_special}

    def ratio(ids: set[int]) -> float:
        if not ids:
            return 0.0
        return float(np.isin(labels, np.asarray(sorted(ids), dtype=labels.dtype)).sum() / total)

    stable = ratio(stable_ids)
    dynamic = ratio(dynamic_ids)
    lowtrust = ratio(lowtrust_ids)
    context = ratio(context_ids)
    static_thing = ratio(static_thing_ids)
    conf = sem.get("confidence_maps")
    out.update(
        {
            "semantic_available": True,
            "stable_mass": stable,
            "harm_mass": float(dynamic + lowtrust),
            "context_mass": context,
            "thing_moving_ratio": dynamic,
            "thing_static_ratio": static_thing,
            "stuff_stable_ratio": stable,
            "lowtrust_stuff_ratio": lowtrust,
            "semantic_confidence_mean": float(conf.detach().float().mean().cpu().item()) if hasattr(conf, "detach") else None,
            "semantic_histogram": {
                str(idx): int(count)
                for idx, count in enumerate(np.bincount(labels.reshape(-1).astype(np.int64)))
                if int(count) > 0
            },
        }
    )
    return out


def radio_features(preprocess_root: Path, seq: str, chunk_id: int) -> dict[str, Any]:
    seq_root = preprocess_root / seq
    candidates: list[Path] = []
    for pattern in ("radseg_sidecar_chunks*", "radio_sidecar_chunks*"):
        candidates.extend(path for path in seq_root.glob(pattern) if path.is_dir())
    out = {
        "radio_available": False,
        "RADIO_boundary_ratio": None,
        "RADIO_temporal_stability": None,
        "radio_static_mean": None,
        "radio_dynamic_mean": None,
        "radio_lowtrust_mean": None,
    }
    for radio_dir in sorted(candidates):
        chunk_dir = find_chunk_dir(radio_dir, int(chunk_id))
        if chunk_dir is None:
            continue
        path = chunk_dir / "radio_sidecar.pt"
        if not path.is_file():
            continue
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception as exc:  # pragma: no cover - diagnostic path
            out["radio_error"] = type(exc).__name__
            return out
        if not isinstance(payload, dict):
            continue

        def mean_tensor(key: str) -> float | None:
            value = payload.get(key)
            return float(value.float().mean().cpu().item()) if hasattr(value, "float") else None

        out.update(
            {
                "radio_available": True,
                "RADIO_boundary_ratio": mean_tensor("object_boundary_score"),
                "RADIO_temporal_stability": mean_tensor("temporal_stability"),
                "radio_static_mean": mean_tensor("radio_static_score"),
                "radio_dynamic_mean": mean_tensor("radio_dynamic_score"),
                "radio_lowtrust_mean": mean_tensor("radio_lowtrust_score"),
            }
        )
        return out
    return out


def hist_cosine(a: dict[str, Any], b: dict[str, Any]) -> float | None:
    keys = sorted(set(a) | set(b), key=lambda x: int(x))
    if not keys:
        return None
    av = np.asarray([float(a.get(key, 0.0)) for key in keys], dtype=np.float64)
    bv = np.asarray([float(b.get(key, 0.0)) for key in keys], dtype=np.float64)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-12:
        return None
    return float(np.dot(av, bv) / denom)


def average_feature(features: Sequence[dict[str, Any]], key: str) -> float | None:
    vals = [finite(item.get(key)) for item in features]
    vals = [value for value in vals if value is not None]
    return float(np.mean(vals)) if vals else None


def load_hmc_ttt_by_chunk(run_dir: Path) -> dict[int, float]:
    path = run_dir / "hmc_state_hash.jsonl"
    out: dict[int, float] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            chunk = row.get("chunk_idx")
            val = finite(row.get("memory_ttt_max_rel_diff"))
            if chunk is None or val is None:
                continue
            idx = int(chunk)
            out[idx] = max(out.get(idx, 0.0), float(val))
    return out


def case_key(row: dict[str, Any], memory: str) -> tuple[Any, ...]:
    if memory == "short":
        return (row.get("seq"), row.get("chunk_id"))
    if memory == "mid":
        return (row.get("seq"), row.get("prev_chunk"), row.get("curr_chunk"))
    return (row.get("seq"), row.get("chunk_start"), row.get("chunk_end"))


def select_cases(
    rows: Sequence[dict[str, Any]],
    memory: str,
    score_key: str,
    target: int,
    case_type: str,
    exclude: set[tuple[Any, ...]] | None = None,
) -> list[dict[str, Any]]:
    exclude = exclude or set()
    reverse = case_type == "bad"
    eligible = [row for row in rows if finite(row.get(score_key)) is not None and case_key(row, memory) not in exclude]
    by_seq: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_seq[str(row["seq"])].append(row)
    for seq_rows in by_seq.values():
        seq_rows.sort(key=lambda row: float(row[score_key]), reverse=reverse)
    seqs = sorted(by_seq, key=lambda seq: float(by_seq[seq][0][score_key]) if by_seq[seq] else -math.inf, reverse=reverse)
    quota = max(1, math.ceil(int(target) / max(len(seqs), 1)))
    selected: list[dict[str, Any]] = []
    used: set[tuple[Any, ...]] = set()
    for seq in seqs:
        for row in by_seq[seq][:quota]:
            key = case_key(row, memory)
            if key in used:
                continue
            selected.append(dict(row))
            used.add(key)
            if len(selected) >= int(target):
                break
        if len(selected) >= int(target):
            break
    if len(selected) < int(target):
        ranked = sorted(eligible, key=lambda row: float(row[score_key]), reverse=reverse)
        for row in ranked:
            key = case_key(row, memory)
            if key in used:
                continue
            selected.append(dict(row))
            used.add(key)
            if len(selected) >= int(target):
                break
    for idx, row in enumerate(selected, start=1):
        row["case_type"] = case_type
        row["case_rank"] = idx
    return selected


def missing_fields(row: dict[str, Any], fields: Sequence[str]) -> list[str]:
    out = []
    for field in fields:
        value = row.get(field)
        if value is None or value == "":
            out.append(field)
    return out


def enrich_short(rows: list[dict[str, Any]], preprocess_root: Path) -> list[dict[str, Any]]:
    sem_cache: dict[tuple[str, int], dict[str, Any]] = {}
    radio_cache: dict[tuple[str, int], dict[str, Any]] = {}
    out: list[dict[str, Any]] = []
    for row in rows:
        seq = str(row["seq"])
        chunk = int(row["chunk_id"])
        sem = sem_cache.setdefault((seq, chunk), semantic_features(preprocess_root, seq, chunk))
        radio = radio_cache.setdefault((seq, chunk), radio_features(preprocess_root, seq, chunk))
        enriched = {
            **row,
            **{key: sem.get(key) for key in (
                "stable_mass",
                "harm_mass",
                "context_mass",
                "thing_moving_ratio",
                "thing_static_ratio",
                "stuff_stable_ratio",
                "lowtrust_stuff_ratio",
                "semantic_available",
                "semantic_confidence_mean",
                "semantic_role_source",
            )},
            **{key: radio.get(key) for key in ("RADIO_boundary_ratio", "RADIO_temporal_stability", "radio_available")},
            "READ_attention_entropy": None,
            "global_attn_layer_candidate": None,
        }
        enriched["case_reason"] = (
            f"{enriched['case_type']}_short_by_J_short_rank; "
            f"J_short={enriched.get('J_short')}; semantic_available={enriched.get('semantic_available')}; "
            f"radio_available={enriched.get('radio_available')}"
        )
        enriched["missing_fields"] = ";".join(missing_fields(enriched, SHORT_FIELDS))
        out.append(enriched)
    return out


def enrich_mid(rows: list[dict[str, Any]], preprocess_root: Path) -> list[dict[str, Any]]:
    sem_cache: dict[tuple[str, int], dict[str, Any]] = {}
    out: list[dict[str, Any]] = []
    for row in rows:
        seq = str(row["seq"])
        prev_chunk = int(row["prev_chunk"])
        curr_chunk = int(row["curr_chunk"])
        prev_sem = sem_cache.setdefault((seq, prev_chunk), semantic_features(preprocess_root, seq, prev_chunk))
        curr_sem = sem_cache.setdefault((seq, curr_chunk), semantic_features(preprocess_root, seq, curr_chunk))
        feats = [prev_sem, curr_sem]
        enriched = {
            **row,
            "overlap_semantic_agreement": hist_cosine(
                prev_sem.get("semantic_histogram", {}),
                curr_sem.get("semantic_histogram", {}),
            ),
            "stable_overlap_mass": average_feature(feats, "stable_mass"),
            "harm_overlap_mass": average_feature(feats, "harm_mass"),
            "context_overlap_mass": average_feature(feats, "context_mass"),
            "same_object_overlap_ratio": None,
            "cross_object_boundary_ratio": None,
            "V_alignment_delta": None,
            "K_risk_delta": None,
            "SWA_gate_mass": None,
            "SWA_replace_mass": None,
            "semantic_available": bool(prev_sem.get("semantic_available")) and bool(curr_sem.get("semantic_available")),
        }
        enriched["case_reason"] = (
            f"{enriched['case_type']}_mid_by_J_mid_rank; J_mid={enriched.get('J_mid')}; "
            f"semantic_pair_available={enriched.get('semantic_available')}; "
            f"raw_overlap_residual_source={enriched.get('raw_overlap_residual_source')}"
        )
        enriched["missing_fields"] = ";".join(missing_fields(enriched, MID_FIELDS))
        out.append(enriched)
    return out


def enrich_long(rows: list[dict[str, Any]], preprocess_root: Path) -> list[dict[str, Any]]:
    sem_cache: dict[tuple[str, int], dict[str, Any]] = {}
    hmc_cache: dict[Path, dict[int, float]] = {}
    out: list[dict[str, Any]] = []
    for row in rows:
        seq = str(row["seq"])
        chunks = list(range(int(row["chunk_start"]), int(row["chunk_end"]) + 1))
        feats = [sem_cache.setdefault((seq, chunk), semantic_features(preprocess_root, seq, chunk)) for chunk in chunks]
        hist_pairs = [
            hist_cosine(feats[i].get("semantic_histogram", {}), feats[i + 1].get("semantic_histogram", {}))
            for i in range(max(0, len(feats) - 1))
        ]
        finite_agreement = [value for value in hist_pairs if value is not None]
        regime_shift = float(1.0 - min(finite_agreement)) if finite_agreement else None
        lowobs = None
        context = average_feature(feats, "context_mass")
        lowtrust = average_feature(feats, "lowtrust_stuff_ratio")
        stable = average_feature(feats, "stable_mass")
        if any(value is not None for value in (context, lowtrust, stable)):
            lowobs = float((context or 0.0) + (lowtrust or 0.0) - (stable or 0.0))
        run_dir = Path(str(row["trajectory"])).parent
        hmc = hmc_cache.setdefault(run_dir, load_hmc_ttt_by_chunk(run_dir))
        ttt_vals = [hmc.get(chunk) for chunk in chunks if hmc.get(chunk) is not None]
        enriched = {
            **row,
            "low_observability_score": lowobs,
            "regime_shift_score": regime_shift,
            "shadow_exposure_change": None,
            "road_edge_continuity": average_feature(feats, "context_mass"),
            "corridor_stability": average_feature(feats, "stable_mass"),
            "TTT_update_conflict": float(max(ttt_vals)) if ttt_vals else None,
            "post_zp_delta": None,
            "semantic_available": all(bool(feat.get("semantic_available")) for feat in feats),
        }
        enriched["case_reason"] = (
            f"{enriched['case_type']}_long_by_J_long_rank; J_long={enriched.get('J_long')}; "
            f"semantic_window_available={enriched.get('semantic_available')}; "
            f"ttt_conflict_source=hmc_state_hash_if_present"
        )
        enriched["missing_fields"] = ";".join(missing_fields(enriched, LONG_FIELDS))
        out.append(enriched)
    return out


def field_order(base: Sequence[str], extra_rows: Sequence[dict[str, Any]]) -> list[str]:
    keys = list(base)
    for row in extra_rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    return keys


def coverage_counts(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_type = Counter(str(row.get("case_type")) for row in rows)
    seqs_by_type: dict[str, list[str]] = {}
    for case_type in ("good", "bad"):
        seqs_by_type[case_type] = sorted({str(row.get("seq")) for row in rows if row.get("case_type") == case_type})
    semantic_complete = sum(bool(row.get("semantic_available")) for row in rows)
    return {
        "rows": len(rows),
        "good": by_type.get("good", 0),
        "bad": by_type.get("bad", 0),
        "good_seqs": seqs_by_type["good"],
        "bad_seqs": seqs_by_type["bad"],
        "semantic_available_rows": semantic_complete,
        "missing_field_counts": dict(Counter(field for row in rows for field in str(row.get("missing_fields", "")).split(";") if field)),
    }


def write_balance_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ACL2 v80 Phase1 Good/Bad Case Balance",
        "",
        "This is a Phase1 case-bank audit summary. It does not claim later action success.",
        "",
        "| memory | rows | good | bad | good_seqs | bad_seqs | semantic_rows | gate_pass |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: | --- |",
    ]
    for memory in ("short", "mid", "long"):
        item = summary["memory_body_summary"][memory]
        lines.append(
            "| {memory} | {rows} | {good} | {bad} | {good_seqs} | {bad_seqs} | {semantic_rows} | {gate} |".format(
                memory=memory,
                rows=item["rows"],
                good=item["good"],
                bad=item["bad"],
                good_seqs=",".join(item["good_seqs"]),
                bad_seqs=",".join(item["bad_seqs"]),
                semantic_rows=item["semantic_available_rows"],
                gate=item["gate_pass"],
            )
        )
    lines += [
        "",
        f"Phase1 gate pass: `{summary['phase1_gate_pass']}`",
        "",
        "Missing plan fields are recorded per CSV row in `missing_fields`; unavailable action-specific fields remain blank.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seqs", default="")
    parser.add_argument("--phase0-dir", type=Path, default=DEFAULT_PHASE0_DIR)
    parser.add_argument("--preprocess-root", type=Path, default=DEFAULT_PREPROCESS_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--kitti-poses-root", type=Path, default=DEFAULT_KITTI_POSES_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--cases-per-type", type=int, default=12)
    args = parser.parse_args()

    phase0_summary = read_json(args.phase0_dir / "phase0_artifact_audit_summary.json")
    if args.seqs:
        seqs = parse_seqs(args.seqs)
    else:
        seqs = parse_seqs(phase0_summary.get("phase1_basic_case_mining_allowed_seqs", []))
    if not seqs:
        raise SystemExit("No Phase1-allowed sequences found. Pass --seqs or rerun Phase0 audit.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trajectory_selection: list[dict[str, Any]] = []
    short_candidates: list[dict[str, Any]] = []
    mid_candidates: list[dict[str, Any]] = []
    long_candidates: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []

    for seq in seqs:
        candidates = discover_trajectories(args.results_root, seq)
        if not candidates:
            trajectory_selection.append({"seq": seq, "selected": False, "reason": "no_trajectory_txt_found"})
            continue
        selected = candidates[0]
        selected.update({"selected": True, "candidate_count": len(candidates), "selection_policy": "max_frame_count_then_geometry_native_preference"})
        trajectory_selection.append(selected)
        short, mid, long, summary = evaluate_trajectory(
            name=Path(str(selected["trajectory"])).parent.name,
            seq=seq,
            path=Path(str(selected["trajectory"])),
            gt_root=args.kitti_poses_root,
            chunk_size=int(args.chunk_size),
            overlap=int(args.chunk_overlap),
            min_coverage=float(args.min_coverage),
        )
        short_candidates.extend(short)
        mid_candidates.extend(mid)
        long_candidates.extend(long)
        run_summaries.append(summary)

    short_score = add_weighted_score(
        short_candidates,
        "J_short",
        (("local_sim3_ate", 0.30), ("head_to_tail", 0.35), ("scale_cv", 0.35)),
    )
    mid_score = add_weighted_score(
        mid_candidates,
        "J_mid",
        (("future_after_overlap", 0.40), ("boundary_jump", 0.25), ("scale_cv", 0.20), ("raw_overlap_residual", 0.15)),
    )
    long_score = add_weighted_score(
        long_candidates,
        "J_long",
        (("window5_joint_sim3_rmse", 0.45), ("window5_subchunk_scale_cv", 0.35), ("downstream_future_consistency", 0.20)),
    )

    short_bad = select_cases(short_candidates, "short", "J_short", args.cases_per_type, "bad")
    short_good = select_cases(short_candidates, "short", "J_short", args.cases_per_type, "good", {case_key(row, "short") for row in short_bad})
    mid_bad = select_cases(mid_candidates, "mid", "J_mid", args.cases_per_type, "bad")
    mid_good = select_cases(mid_candidates, "mid", "J_mid", args.cases_per_type, "good", {case_key(row, "mid") for row in mid_bad})
    long_bad = select_cases(long_candidates, "long", "J_long", args.cases_per_type, "bad")
    long_good = select_cases(long_candidates, "long", "J_long", args.cases_per_type, "good", {case_key(row, "long") for row in long_bad})

    short_cases = enrich_short(short_bad + short_good, args.preprocess_root)
    mid_cases = enrich_mid(mid_bad + mid_good, args.preprocess_root)
    long_cases = enrich_long(long_bad + long_good, args.preprocess_root)

    write_csv(args.out_dir / "trajectory_selection.csv", trajectory_selection)
    write_csv(args.out_dir / "short_single_chunk_candidate_metrics.csv", short_candidates)
    write_csv(args.out_dir / "mid_adjacent_pair_candidate_metrics.csv", mid_candidates)
    write_csv(args.out_dir / "long_five_chunk_candidate_metrics.csv", long_candidates)
    write_csv(args.out_dir / "short_single_chunk_cases.csv", short_cases, field_order(SHORT_FIELDS, short_cases))
    write_csv(args.out_dir / "mid_adjacent_pair_cases.csv", mid_cases, field_order(MID_FIELDS, mid_cases))
    write_csv(args.out_dir / "long_five_chunk_cases.csv", long_cases, field_order(LONG_FIELDS, long_cases))

    memory_summary = {
        "short": coverage_counts(short_cases),
        "mid": coverage_counts(mid_cases),
        "long": coverage_counts(long_cases),
    }
    for memory, item in memory_summary.items():
        item["balance_gate_pass"] = (
            item["good"] >= int(args.cases_per_type)
            and item["bad"] >= int(args.cases_per_type)
            and len(item["good_seqs"]) >= 3
            and len(item["bad_seqs"]) >= 3
        )
        item["semantic_diagnosis_gate_pass"] = item["semantic_available_rows"] == item["rows"]
        item["gate_pass"] = bool(item["balance_gate_pass"] and item["semantic_diagnosis_gate_pass"])
    summary = {
        "schema": "acl2_v80tf_phase1_three_memory_case_bank_v1",
        "created_at_utc": utc_now(),
        "phase0_dir": str(args.phase0_dir),
        "seqs": seqs,
        "chunk_size": int(args.chunk_size),
        "chunk_overlap": int(args.chunk_overlap),
        "min_coverage": float(args.min_coverage),
        "cases_per_type_target": int(args.cases_per_type),
        "trajectory_selection_policy": "max_frame_count_then_geometry_native_preference",
        "candidate_counts": {
            "short": len(short_candidates),
            "mid": len(mid_candidates),
            "long": len(long_candidates),
        },
        "score_definitions": {
            "J_short": short_score,
            "J_mid": mid_score,
            "J_long": long_score,
        },
        "run_summaries": run_summaries,
        "memory_body_summary": memory_summary,
        "phase1_balance_gate_pass": all(item["balance_gate_pass"] for item in memory_summary.values()),
        "semantic_diagnosis_gate_pass": all(item["semantic_diagnosis_gate_pass"] for item in memory_summary.values()),
        "phase1_gate_pass": all(item["gate_pass"] for item in memory_summary.values()),
        "notes": [
            "READ_attention_entropy/global_attn_layer_candidate are left empty; they require Phase2 visual/QKV artifacts.",
            "SWA K/V fields are left empty unless overlap/action artifacts provide them in later phases.",
            "post_zp_delta is left empty unless post-zp delta artifacts are explicitly available in later phases.",
        ],
    }
    write_json(args.out_dir / "case_bank_summary.json", summary)
    write_balance_md(args.out_dir / "good_bad_case_balance.md", summary)

    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "phase1_gate_pass": summary["phase1_gate_pass"],
                "candidate_counts": summary["candidate_counts"],
                "memory_body_summary": {
                    key: {
                        "good": value["good"],
                        "bad": value["bad"],
                        "good_seqs": value["good_seqs"],
                        "bad_seqs": value["bad_seqs"],
                        "gate_pass": value["gate_pass"],
                    }
                    for key, value in memory_summary.items()
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
