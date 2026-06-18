#!/usr/bin/env python3
"""ACL2 v64 TTT-scale mechanism attribution diagnostics.

This generator only reports metrics that can be recomputed from landed
artifacts. Missing tensor/head/token/merge-gauge evidence is written as
unavailable or not_run; no placeholder numeric evidence is fabricated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GT_PATH = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
RESULT_ROOT = ROOT / "results/kitti01_hmc_v2/acl2_v64_ttt_scale_mechanism_attribution"
OUT_DEFAULT = RESULT_ROOT / "report_final"

V43_ROOT = ROOT / "results/kitti01_hmc_v2/acl2_v43_c9_dechunk_attribution_semanticread_target30"
V63_ROOT = ROOT / "results/kitti01_hmc_v2/acl2_v63_ttt_scale_causal_diagnostic"

COMPONENT_RUNS = [
    {
        "plan_key": "C9_P0_R2",
        "component_family": "base_c9",
        "run_name": "V43_P0_F0_C9_LOCKED_REPEAT",
        "path": V43_ROOT / "phase0_c9_repeat/rollouts/V43_P0_F0_C9_LOCKED_REPEAT",
    },
    {
        "plan_key": "C9_MINUS_READ_MAP_TO_FLAT",
        "component_family": "read_map",
        "run_name": "V43_P2_ATTR_01_C9_MINUS_READ_MAP_TO_FLAT",
        "path": V43_ROOT / "phase2_attribution/rollouts/V43_P2_ATTR_01_C9_MINUS_READ_MAP_TO_FLAT",
    },
    {
        "plan_key": "C9_MINUS_TRI_CHUNKMAP_TO_FLAT",
        "component_family": "ttt_tri_gamma_chunk_map",
        "run_name": "V43_P2_ATTR_02_C9_MINUS_TRI_CHUNKMAP_TO_FLAT",
        "path": V43_ROOT / "phase2_attribution/rollouts/V43_P2_ATTR_02_C9_MINUS_TRI_CHUNKMAP_TO_FLAT",
    },
    {
        "plan_key": "C9_MINUS_COMMIT_EMA",
        "component_family": "ttt_commit_ema",
        "run_name": "V43_P2_ATTR_03_C9_MINUS_COMMIT_EMA",
        "path": V43_ROOT / "phase2_attribution/rollouts/V43_P2_ATTR_03_C9_MINUS_COMMIT_EMA",
    },
    {
        "plan_key": "C9_MINUS_SWA_OVERLAP_REPLACE",
        "component_family": "swa_overlap_replace",
        "run_name": "V43_P2_ATTR_04_C9_MINUS_SWA_OVERLAP_REPLACE",
        "path": V43_ROOT / "phase2_attribution/rollouts/V43_P2_ATTR_04_C9_MINUS_SWA_OVERLAP_REPLACE",
    },
    {
        "plan_key": "C9_MINUS_TTT_TRI_REPLAY",
        "component_family": "ttt_tri_replay",
        "run_name": "V43_P2_ATTR_05_C9_MINUS_TTT_TRI_REPLAY",
        "path": V43_ROOT / "phase2_attribution/rollouts/V43_P2_ATTR_05_C9_MINUS_TTT_TRI_REPLAY",
    },
    {
        "plan_key": "C9_MINUS_NATIVE_MIX",
        "component_family": "native_mix",
        "run_name": "V43_P2_ATTR_06_C9_MINUS_NATIVE_MIX",
        "path": V43_ROOT / "phase2_attribution/rollouts/V43_P2_ATTR_06_C9_MINUS_NATIVE_MIX",
    },
    {
        "plan_key": "C9_NO_CHUNK_ID_ALL__FLAT_01",
        "component_family": "no_chunk_id_all_flat",
        "run_name": "V43_P1_FLAT_01",
        "path": V43_ROOT / "phase1_flat/rollouts/V43_P1_FLAT_01",
    },
]

V63_H35_FULL = V63_ROOT / "phase0_full_trace/rollouts/V63_P0_H35_POSTZP_TRACE_FULL"
V63_C9_FULL = V63_ROOT / "phase0_full_trace/rollouts/V63_P0_C9_POSTZP_TRACE_FULL"
V63_FORK_ROOT = V63_ROOT / "phase3_ttt_causal_fork_basic/rollouts"
V63_FORKS = {
    "F0_BASE": V63_FORK_ROOT / "V63_P3_H35_C19_BASE_E730",
    "F1_NO_TTT_WRITE": V63_FORK_ROOT / "V63_P3_H35_C19_FREEZE_E730",
    "F2_NATIVE_COMMIT_ONLY": V63_FORK_ROOT / "V63_P3_H35_C19_NATIVE0_E730",
    "F4_HALF_DELTA_COMMIT": V63_FORK_ROOT / "V63_P3_H35_C19_HALF_E730",
    "F5_DOUBLE_DELTA_COMMIT": V63_FORK_ROOT / "V63_P3_H35_C19_DOUBLE_E730",
}
V63_FORK_CHUNK = 19
V64_PHASE3_ROLLOUT_ROOT = RESULT_ROOT / "phase3_within_reset_ttt_causal_fork/rollouts"

EPS = 1e-12


@dataclass
class PoseData:
    centers: np.ndarray
    rotations: np.ndarray
    count: int


@dataclass
class RunData:
    key: str
    plan_key: str
    component_family: str
    run_name: str
    path: Path
    poses: PoseData
    gt: PoseData
    global_scale: float
    global_rot: np.ndarray
    global_trans: np.ndarray
    global_aligned: np.ndarray
    global_residual: np.ndarray
    hmc_rows: List[Dict[str, Any]]
    probe_rows: List[Dict[str, Any]]
    pose_trace_rows: List[Dict[str, Any]]
    layer_rows: List[Dict[str, Any]]
    registry: Dict[str, Any]


def _clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            out: Dict[str, Any] = {}
            for key in fields:
                value = _clean(row.get(key))
                if value is None:
                    out[key] = ""
                elif isinstance(value, (dict, list)):
                    out[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
                else:
                    out[key] = value
            writer.writerow(out)


def _safe_float(value: Any) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return val if math.isfinite(val) else float("nan")


def _finite(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        val = _safe_float(value)
        if math.isfinite(val):
            out.append(val)
    return out


def _mean(values: Iterable[Any]) -> Optional[float]:
    vals = _finite(values)
    return float(np.mean(vals)) if vals else None


def _median(values: Iterable[Any]) -> Optional[float]:
    vals = _finite(values)
    return float(np.median(vals)) if vals else None


def _max(values: Iterable[Any]) -> Optional[float]:
    vals = _finite(values)
    return float(np.max(vals)) if vals else None


def _fmt(value: Any, digits: int = 6) -> str:
    val = _safe_float(value)
    return f"{val:.{digits}f}" if math.isfinite(val) else "NA"


def _pct(value: Any) -> str:
    val = _safe_float(value)
    return f"{100.0 * val:.2f}%" if math.isfinite(val) else "NA"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    q = q / (np.linalg.norm(q) + EPS)
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _read_poses(path: Path, limit: Optional[int] = None) -> PoseData:
    centers: List[List[float]] = []
    rotations: List[np.ndarray] = []
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split()
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue
            if len(vals) == 12:
                mat = np.asarray(vals, dtype=np.float64).reshape(3, 4)
                centers.append(mat[:, 3].tolist())
                rotations.append(mat[:, :3])
            elif len(vals) >= 8:
                centers.append(vals[1:4])
                rotations.append(_quat_to_rot(vals[4], vals[5], vals[6], vals[7]))
            if limit is not None and len(centers) >= limit:
                break
    if not centers:
        raise ValueError(f"No poses parsed from {path}")
    arr = np.asarray(centers, dtype=np.float64)
    rots = np.asarray(rotations, dtype=np.float64)
    return PoseData(arr, rots, int(arr.shape[0]))


def _fit_sim3(src: np.ndarray, dst: np.ndarray) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = min(src.shape[0], dst.shape[0])
    src = src[:n]
    dst = dst[:n]
    mask = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    src = src[mask]
    dst = dst[mask]
    if src.shape[0] < 3:
        return None
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    xs = src - mu_src
    yd = dst - mu_dst
    var_src = float(np.sum(xs * xs) / src.shape[0])
    if var_src <= EPS:
        return None
    cov = (yd.T @ xs) / src.shape[0]
    u, d, vt = np.linalg.svd(cov)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1.0
    rot = u @ np.diag(sign) @ vt
    scale = float(np.sum(d * sign) / var_src)
    if not math.isfinite(scale) or abs(scale) <= EPS:
        return None
    trans = mu_dst - scale * (rot @ mu_src)
    return scale, rot, trans


def _apply_sim3(points: np.ndarray, fit: Tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    scale, rot, trans = fit
    return scale * (points @ rot.T) + trans[None, :]


def _residual_norms(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    n = min(pred.shape[0], gt.shape[0])
    if n <= 0:
        return np.zeros((0,), dtype=np.float64)
    pred = pred[:n]
    gt = gt[:n]
    mask = np.isfinite(pred).all(axis=1) & np.isfinite(gt).all(axis=1)
    if not np.any(mask):
        return np.zeros((0,), dtype=np.float64)
    return np.linalg.norm(pred[mask] - gt[mask], axis=1)


def _rmse_from_delta(delta: np.ndarray) -> Optional[float]:
    arr = np.asarray(delta, dtype=np.float64)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        arr = arr[np.isfinite(arr)]
        return float(np.sqrt(np.mean(arr * arr))) if arr.size else None
    mask = np.isfinite(arr).all(axis=1)
    if not np.any(mask):
        return None
    vals = np.sum(arr[mask] * arr[mask], axis=1)
    return float(np.sqrt(np.mean(vals)))


def _rmse_points(pred: np.ndarray, gt: np.ndarray) -> Optional[float]:
    norms = _residual_norms(pred, gt)
    return float(np.sqrt(np.mean(norms * norms))) if norms.size else None


def _step_lengths(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    return np.linalg.norm(np.diff(points, axis=0), axis=1)


def _step_ratio(pred: np.ndarray, gt: np.ndarray) -> Optional[float]:
    pred_steps = _step_lengths(pred)
    gt_steps = _step_lengths(gt)
    n = min(pred_steps.size, gt_steps.size)
    if n <= 0:
        return None
    mask = np.isfinite(pred_steps[:n]) & np.isfinite(gt_steps[:n]) & (gt_steps[:n] > EPS)
    if not np.any(mask):
        return None
    return float(np.median(pred_steps[:n][mask] / gt_steps[:n][mask]))


def _rot_angle_deg(rot: np.ndarray) -> Optional[float]:
    if rot is None:
        return None
    val = float(np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.degrees(math.acos(val)))


def _window_slices(start: int, end: int) -> Dict[str, Tuple[int, int]]:
    length = max(0, end - start)
    third = max(3, length // 3)
    head = (start, min(end, start + third))
    mid_start = start + max(0, (length - third) // 2)
    mid = (mid_start, min(end, mid_start + third))
    tail = (max(start, end - third), end)
    return {"head": head, "mid": mid, "tail": tail}


def _slice(run: RunData, start: int, end: int) -> Tuple[np.ndarray, np.ndarray]:
    start = max(0, min(int(start), run.poses.count))
    end = max(start, min(int(end), run.poses.count))
    return run.poses.centers[start:end], run.gt.centers[start:end]


def _row_by_chunk(rows: Sequence[Mapping[str, Any]]) -> Dict[int, Mapping[str, Any]]:
    out: Dict[int, Mapping[str, Any]] = {}
    for idx, row in enumerate(rows):
        chunk = row.get("chunk_idx", row.get("chunk_id", idx))
        try:
            out[int(chunk)] = row
        except (TypeError, ValueError):
            continue
    return out


def _unique_chunk_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return len(_row_by_chunk(rows))


def _chunk_ids_from_rows(rows: Sequence[Mapping[str, Any]]) -> set[int]:
    out: set[int] = set()
    for row in rows:
        val = _safe_float(row.get("chunk_id"))
        if math.isfinite(val):
            out.add(int(val))
    return out


def _chunk_range(run: RunData, chunk_id: int) -> Tuple[int, int]:
    hmc = _row_by_chunk(run.hmc_rows).get(chunk_id, {})
    probe = _row_by_chunk(run.probe_rows).get(chunk_id, {})
    start = hmc.get("start_frame", probe.get("start_frame"))
    end = hmc.get("end_frame", probe.get("end_frame"))
    if start is None or end is None:
        chunk_size = int(_safe_float(hmc.get("chunk_size")) or 32)
        overlap = int(_safe_float(hmc.get("chunk_overlap")) or 3)
        stride = max(1, chunk_size - overlap)
        start = int(chunk_id) * stride
        end = int(start) + chunk_size
    return int(start), int(end)


def _load_registry_rows() -> Dict[str, Dict[str, Any]]:
    paths = [
        V43_ROOT / "phase0_c9_repeat/report_R1/full_metrics/full_online_registry.csv",
        V43_ROOT / "phase1_flat/report_R1/full_online_registry.csv",
        V43_ROOT / "phase2_attribution/report_R1/full_online_registry.csv",
    ]
    out: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        for row in _read_csv_rows(path):
            run_name = row.get("run_name")
            if run_name:
                out[str(run_name)] = row
    return out


def _load_layer_rows(path: Path) -> List[Dict[str, Any]]:
    rows = _read_csv_rows(path / "per_layer_branch_update_heatmap.csv")
    out: List[Dict[str, Any]] = []
    for row in rows:
        item: Dict[str, Any] = dict(row)
        for key in (
            "chunk_idx",
            "layer",
            "post_delta_norm_mean",
            "pre_delta_norm_mean",
            "pre_post_cos_mean",
            "native_mix_scale",
            "scale_mean",
            "scale_min",
            "cos_mean",
            "cos_min",
        ):
            if key in item:
                val = _safe_float(item[key])
                item[key] = val if math.isfinite(val) else None
        out.append(item)
    return out


def _layer_aggs(run: RunData, chunk_id: int) -> Dict[str, Any]:
    rows = [r for r in run.layer_rows if int(_safe_float(r.get("chunk_idx"))) == chunk_id]
    out: Dict[str, Any] = {
        "layer_branch_rows": len(rows),
        "post_zp_delta_norm_total": _mean(r.get("post_delta_norm_mean") for r in rows),
        "post_zp_delta_norm_max": _max(r.get("post_delta_norm_mean") for r in rows),
        "native_mix_scale_mean": _mean(r.get("native_mix_scale") for r in rows),
    }
    for branch in ("w0", "w1", "w2"):
        sub = [r for r in rows if str(r.get("branch")) == branch]
        out[f"{branch}_post_zp_delta_norm_mean"] = _mean(r.get("post_delta_norm_mean") for r in sub)
        out[f"{branch}_native_mix_scale_mean"] = _mean(r.get("native_mix_scale") for r in sub)
    for layer in (0, 8, 17):
        sub = [r for r in rows if int(_safe_float(r.get("layer"))) == layer]
        out[f"layer{layer}_post_zp_delta_norm_mean"] = _mean(r.get("post_delta_norm_mean") for r in sub)
    return out


def _load_run(spec: Mapping[str, Any], gt_all: PoseData, registry_by_run: Mapping[str, Dict[str, Any]]) -> Optional[RunData]:
    path = Path(spec["path"])
    pred_path = path / "01.txt"
    if not pred_path.is_file():
        return None
    pred = _read_poses(pred_path)
    n = min(pred.count, gt_all.count)
    pred = PoseData(pred.centers[:n], pred.rotations[:n], n)
    gt = PoseData(gt_all.centers[:n], gt_all.rotations[:n], n)
    fit = _fit_sim3(pred.centers, gt.centers)
    if fit is None:
        scale, rot, trans = 1.0, np.eye(3), np.zeros(3)
        aligned = pred.centers.copy()
    else:
        scale, rot, trans = fit
        aligned = _apply_sim3(pred.centers, fit)
    return RunData(
        key=str(spec.get("key") or spec.get("plan_key") or spec.get("run_name")),
        plan_key=str(spec.get("plan_key") or spec.get("run_name")),
        component_family=str(spec.get("component_family") or ""),
        run_name=str(spec.get("run_name") or path.name),
        path=path,
        poses=pred,
        gt=gt,
        global_scale=float(scale),
        global_rot=rot,
        global_trans=trans,
        global_aligned=aligned,
        global_residual=aligned - gt.centers,
        hmc_rows=_read_jsonl(path / "hmc_state_hash.jsonl"),
        probe_rows=_read_jsonl(path / "hmc_probe_summary.jsonl"),
        pose_trace_rows=_read_jsonl(path / "per_chunk_pose_trace.jsonl"),
        layer_rows=_load_layer_rows(path),
        registry=dict(registry_by_run.get(str(spec.get("run_name") or path.name), {})),
    )


def _run_segment_rmse(run: RunData, start: int, end: int) -> Optional[float]:
    start = max(0, min(start, run.poses.count))
    end = max(start, min(end, run.poses.count))
    return _rmse_from_delta(run.global_residual[start:end])


def _chunk_metrics(run: RunData) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    chunk_ids: List[int] = []
    for idx, row in enumerate(run.hmc_rows or run.probe_rows):
        chunk = row.get("chunk_idx", row.get("chunk_id", idx))
        try:
            chunk_ids.append(int(chunk))
        except (TypeError, ValueError):
            continue
    if not chunk_ids:
        chunk_ids = list(range(max(0, (run.poses.count - 3) // 29)))
    full_pred_step_median = _median(_step_lengths(run.poses.centers))
    prev_log_scale: Optional[float] = None
    for chunk_id in sorted(set(chunk_ids)):
        start, end = _chunk_range(run, chunk_id)
        pred, gt = _slice(run, start, end)
        fit = _fit_sim3(pred, gt)
        if fit is not None:
            local = _apply_sim3(pred, fit)
            local_ate = _rmse_points(local, gt)
            scale, rot, trans = fit
            log_scale = math.log(abs(scale) + EPS)
            rot_deg = _rot_angle_deg(rot)
            trans_norm = float(np.linalg.norm(trans))
            final_err = float(_residual_norms(local, gt)[-1]) if _residual_norms(local, gt).size else None
        else:
            local_ate = None
            scale = None
            log_scale = None
            rot_deg = None
            trans_norm = None
            final_err = None
        global_ate = _rmse_from_delta(run.global_residual[start:end])
        ratio = (
            float(local_ate / max(global_ate, EPS))
            if local_ate is not None and global_ate is not None and math.isfinite(global_ate)
            else None
        )
        sub_scales: List[float] = []
        windows = _window_slices(start, end)
        for s, e in windows.values():
            p, g = _slice(run, s, e)
            sub_fit = _fit_sim3(p, g)
            if sub_fit is not None:
                sub_scales.append(math.log(abs(sub_fit[0]) + EPS))
        head_pred, head_gt = _slice(run, *windows["head"])
        tail_pred, tail_gt = _slice(run, *windows["tail"])
        head_fit = _fit_sim3(head_pred, head_gt)
        if head_fit is not None and tail_pred.shape[0] >= 3:
            head_tail = _rmse_points(_apply_sim3(tail_pred, head_fit), tail_gt)
        else:
            head_tail = None
        pred_step_median = _median(_step_lengths(pred))
        scale_jump = (log_scale - prev_log_scale) if log_scale is not None and prev_log_scale is not None else None
        if log_scale is not None:
            prev_log_scale = log_scale
        hmc = _row_by_chunk(run.hmc_rows).get(chunk_id, {})
        row = {
            "run_key": run.key,
            "plan_key": run.plan_key,
            "component_family": run.component_family,
            "run_name": run.run_name,
            "chunk_id": chunk_id,
            "frame_start": start,
            "frame_end": end,
            "reset_group_id": chunk_id // 5,
            "reset_relative_idx": chunk_id % 5,
            "global_chunk_ate": global_ate,
            "local_sim3_chunk_ate": local_ate,
            "local_to_global_ratio": ratio,
            "scale_fit_pred_to_gt": scale,
            "log_scale_residual": log_scale,
            "abs_log_scale_residual": abs(log_scale) if log_scale is not None else None,
            "scale_jump_vs_prev": scale_jump,
            "abs_scale_jump_vs_prev": abs(scale_jump) if scale_jump is not None else None,
            "step_length_ratio": _step_ratio(pred, gt),
            "step_length_ratio_nogt": (
                float(pred_step_median / max(full_pred_step_median or EPS, EPS))
                if pred_step_median is not None and full_pred_step_median is not None
                else None
            ),
            "head_to_tail_transfer_error": head_tail,
            "head_to_tail_transfer_ratio": (
                float(head_tail / max(local_ate, EPS))
                if head_tail is not None and local_ate is not None and math.isfinite(local_ate)
                else None
            ),
            "intra_scale_variance": float(np.var(np.asarray(sub_scales))) if len(sub_scales) >= 2 else None,
            "rolling100": _rmse_from_delta(run.global_residual[max(0, start - 50) : min(run.poses.count, end + 50)]),
            "future_200_300": _run_segment_rmse(run, 200, 300),
            "future_400_600": _run_segment_rmse(run, 400, 600),
            "Rot": rot_deg,
            "FinalErr": final_err,
            "local_sim3_translation_norm": trans_norm,
            "probe_ttt_write_tri_delta_norm_mean": hmc.get("probe_ttt_write_tri_delta_norm_mean"),
            "probe_ttt_write_tri_pos_mass_mean": hmc.get("probe_ttt_write_tri_pos_mass_mean", hmc.get("auxgeo_tri_replay_pos_mass_mean")),
            "probe_ttt_write_tri_neu_mass_mean": hmc.get("probe_ttt_write_tri_neu_mass_mean", hmc.get("auxgeo_tri_replay_neu_mass_mean")),
            "probe_ttt_write_tri_neg_mass_mean": hmc.get("probe_ttt_write_tri_neg_mass_mean", hmc.get("auxgeo_tri_replay_neg_mass_mean")),
            "probe_ttt_write_state_hash": hmc.get("probe_ttt_write_state_hash"),
            "commit_source_state_hash": hmc.get("commit_source_state_hash"),
            "ttt_semantic_scaled_state_hash": hmc.get("ttt_semantic_scaled_state_hash"),
            "ttt_write_commit_frozen_at_chunk": hmc.get("ttt_write_commit_frozen_at_chunk"),
            "ttt_semantic_write_scale_at_chunk": hmc.get("ttt_semantic_write_scale_at_chunk"),
            **_layer_aggs(run, chunk_id),
        }
        rows.append(row)
    by_chunk = {int(r["chunk_id"]): r for r in rows}
    for row in rows:
        cid = int(row["chunk_id"])
        for horizon in (1, 3, 5):
            fut = [by_chunk.get(cid + j, {}) for j in range(1, horizon + 1)]
            row[f"future_h{horizon}_scale_residual"] = _mean(r.get("abs_log_scale_residual") for r in fut)
            row[f"future_h{horizon}_rolling100"] = _mean(r.get("rolling100") for r in fut)
            row[f"future_h{horizon}_global_chunk_ate"] = _mean(r.get("global_chunk_ate") for r in fut)
    return rows


def _taxonomy(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    abs_scale_vals = _finite(r.get("abs_log_scale_residual") for r in rows)
    future_vals = _finite(r.get("future_h3_scale_residual") for r in rows)
    global_vals = _finite(r.get("global_chunk_ate") for r in rows)
    scale_hi = float(np.percentile(abs_scale_vals, 75)) if abs_scale_vals else float("nan")
    future_hi = float(np.percentile(future_vals, 75)) if future_vals else float("nan")
    global_hi = float(np.percentile(global_vals, 75)) if global_vals else float("nan")
    out: List[Dict[str, Any]] = []
    for row in rows:
        local_ratio = _safe_float(row.get("local_to_global_ratio"))
        abs_scale = _safe_float(row.get("abs_log_scale_residual"))
        future = _safe_float(row.get("future_h3_scale_residual"))
        global_ate = _safe_float(row.get("global_chunk_ate"))
        if math.isfinite(local_ratio) and local_ratio <= 0.35 and math.isfinite(abs_scale) and abs_scale >= scale_hi:
            typ = "TYPE_B_cross_chunk_scale_gauge"
        elif math.isfinite(future) and future >= future_hi and math.isfinite(abs_scale) and abs_scale < scale_hi:
            typ = "TYPE_D_overlap_future_transfer"
        elif math.isfinite(global_ate) and global_ate >= global_hi and math.isfinite(local_ratio) and local_ratio <= 0.5:
            typ = "TYPE_E_global_placement"
        elif math.isfinite(abs_scale) and abs_scale >= scale_hi:
            typ = "TYPE_A_chunk_scale_residual"
        else:
            typ = "TYPE_C_lower_scale_risk"
        out.append(
            {
                "run_key": row.get("run_key"),
                "plan_key": row.get("plan_key"),
                "run_name": row.get("run_name"),
                "chunk_id": row.get("chunk_id"),
                "primary_error_type": typ,
                "threshold_abs_log_scale_p75": scale_hi,
                "threshold_future_h3_scale_p75": future_hi,
                "threshold_global_chunk_ate_p75": global_hi,
                "evidence_abs_log_scale": row.get("abs_log_scale_residual"),
                "evidence_future_h3_scale": row.get("future_h3_scale_residual"),
                "evidence_local_to_global_ratio": row.get("local_to_global_ratio"),
                "evidence_global_chunk_ate": row.get("global_chunk_ate"),
            }
        )
    return out


def _summarize_run(run: RunData, rows: Sequence[Mapping[str, Any]], tax_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for row in tax_rows:
        counts[str(row.get("primary_error_type"))] = counts.get(str(row.get("primary_error_type")), 0) + 1
    return {
        "plan_key": run.plan_key,
        "component_family": run.component_family,
        "run_name": run.run_name,
        "path": str(run.path),
        "trajectory_available": (run.path / "01.txt").is_file(),
        "frames": run.poses.count,
        "hmc_rows": len(run.hmc_rows),
        "probe_rows": len(run.probe_rows),
        "pose_trace_rows": len(run.pose_trace_rows),
        "layer_branch_rows": len(run.layer_rows),
        "registry_ATE_full": run.registry.get("ATE_full"),
        "registry_Rot_full": run.registry.get("Rot_full"),
        "registry_FinalErr_full": run.registry.get("FinalErr_full"),
        "computed_global_sim3_rmse": _rmse_from_delta(run.global_residual),
        "computed_global_sim3_scale": run.global_scale,
        "local_sim3_chunk_ate_mean": _mean(r.get("local_sim3_chunk_ate") for r in rows),
        "global_chunk_ate_mean": _mean(r.get("global_chunk_ate") for r in rows),
        "local_to_global_ratio_median": _median(r.get("local_to_global_ratio") for r in rows),
        "abs_log_scale_residual_mean": _mean(r.get("abs_log_scale_residual") for r in rows),
        "abs_scale_jump_vs_prev_mean": _mean(r.get("abs_scale_jump_vs_prev") for r in rows),
        "intra_scale_variance_mean": _mean(r.get("intra_scale_variance") for r in rows),
        "head_to_tail_transfer_ratio_mean": _mean(r.get("head_to_tail_transfer_ratio") for r in rows),
        "future_h3_scale_residual_mean": _mean(r.get("future_h3_scale_residual") for r in rows),
        "future_h3_rolling100_mean": _mean(r.get("future_h3_rolling100") for r in rows),
        "taxonomy_counts": counts,
    }


def _component_deltas(summary: Sequence[Mapping[str, Any]], tax_by_plan: Mapping[str, Dict[str, int]]) -> List[Dict[str, Any]]:
    base = next((r for r in summary if r.get("plan_key") == "C9_P0_R2"), None)
    if base is None:
        return []
    metric_names = [
        "computed_global_sim3_rmse",
        "local_sim3_chunk_ate_mean",
        "abs_log_scale_residual_mean",
        "abs_scale_jump_vs_prev_mean",
        "intra_scale_variance_mean",
        "head_to_tail_transfer_ratio_mean",
        "future_h3_scale_residual_mean",
        "future_h3_rolling100_mean",
    ]
    rows: List[Dict[str, Any]] = []
    base_type_total = sum(tax_by_plan.get("C9_P0_R2", {}).values())
    base_high = sum(
        count
        for typ, count in tax_by_plan.get("C9_P0_R2", {}).items()
        if typ in {"TYPE_B_cross_chunk_scale_gauge", "TYPE_D_overlap_future_transfer", "TYPE_E_global_placement"}
    )
    base_high_frac = base_high / base_type_total if base_type_total else float("nan")
    for row in summary:
        if row.get("plan_key") == "C9_P0_R2":
            continue
        out = {
            "plan_key": row.get("plan_key"),
            "component_family": row.get("component_family"),
            "run_name": row.get("run_name"),
        }
        regressions: List[float] = []
        for metric in metric_names:
            b = _safe_float(base.get(metric))
            v = _safe_float(row.get(metric))
            delta = v - b if math.isfinite(v) and math.isfinite(b) else float("nan")
            rel = delta / abs(b) if math.isfinite(delta) and abs(b) > EPS else float("nan")
            out[f"{metric}_delta_vs_c9"] = delta
            out[f"{metric}_delta_percent_vs_c9"] = rel
            if metric in {
                "abs_log_scale_residual_mean",
                "abs_scale_jump_vs_prev_mean",
                "intra_scale_variance_mean",
                "head_to_tail_transfer_ratio_mean",
                "future_h3_scale_residual_mean",
            } and math.isfinite(rel):
                regressions.append(rel)
        total = sum(tax_by_plan.get(str(row.get("plan_key")), {}).values())
        high = sum(
            count
            for typ, count in tax_by_plan.get(str(row.get("plan_key")), {}).items()
            if typ in {"TYPE_B_cross_chunk_scale_gauge", "TYPE_D_overlap_future_transfer", "TYPE_E_global_placement"}
        )
        high_frac = high / total if total else float("nan")
        out["high_risk_taxonomy_fraction"] = high_frac
        out["high_risk_taxonomy_fraction_delta_vs_c9"] = high_frac - base_high_frac if math.isfinite(high_frac) and math.isfinite(base_high_frac) else None
        out["max_scale_metric_regression_percent"] = max(regressions) if regressions else None
        rows.append(out)
    return rows


def _phase0(out_dir: Path, component_runs: Sequence[RunData]) -> Dict[str, Any]:
    phase = out_dir / "phase0_c9_component_scale_ledger"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    all_chunk_rows: List[Dict[str, Any]] = []
    tax_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    tax_by_plan: Dict[str, Dict[str, int]] = {}
    for run in component_runs:
        rows = _chunk_metrics(run)
        taxes = _taxonomy(rows)
        all_chunk_rows.extend(rows)
        tax_rows.extend(taxes)
        counts: Dict[str, int] = {}
        for tax in taxes:
            counts[str(tax["primary_error_type"])] = counts.get(str(tax["primary_error_type"]), 0) + 1
        tax_by_plan[run.plan_key] = counts
        summaries.append(_summarize_run(run, rows, taxes))
    delta_rows = _component_deltas(summaries, tax_by_plan)
    inventory = [
        {
            "plan_key": run.plan_key,
            "component_family": run.component_family,
            "run_name": run.run_name,
            "path": str(run.path),
            "trajectory_available": (run.path / "01.txt").is_file(),
            "hmc_state_hash_available": (run.path / "hmc_state_hash.jsonl").is_file(),
            "layer_branch_csv_available": (run.path / "per_layer_branch_update_heatmap.csv").is_file(),
            "registry_ATE_full": run.registry.get("ATE_full"),
            "frames": run.poses.count,
        }
        for run in component_runs
    ]
    _write_csv(phase / "component_inventory.csv", inventory)
    _write_csv(phase / "component_chunk_scale_metrics.csv", all_chunk_rows)
    _write_csv(phase / "component_scale_summary.csv", summaries)
    _write_csv(phase / "component_scale_delta_vs_c9.csv", delta_rows)
    _write_csv(phase / "component_taxonomy_shift.csv", tax_rows)
    _write_json(phase / "component_taxonomy_counts.json", tax_by_plan)
    _plot_component_ledger(fig_dir / "c9_component_scale_ledger.png", delta_rows)
    _plot_reset_profile(fig_dir / "reset_relative_ttt_scale_profile.png", all_chunk_rows)
    gate = _phase0_gate(delta_rows)
    report = [
        "# Phase 0 C9 Component Scale Ledger",
        "",
        "Source artifacts are landed v43 C9 component trajectories. Scale metrics were recomputed from `01.txt` trajectories against KITTI01 GT; registry ATE is reported only as inventory context.",
        "",
        "| plan key | ATE | abs log-scale mean | future h3 scale mean | max scale regression | high-risk taxonomy delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    by_summary = {str(r["plan_key"]): r for r in summaries}
    by_delta = {str(r["plan_key"]): r for r in delta_rows}
    for row in summaries:
        delta = by_delta.get(str(row["plan_key"]), {})
        report.append(
            f"| `{row['plan_key']}` | {_fmt(row.get('registry_ATE_full'))} | {_fmt(row.get('abs_log_scale_residual_mean'))} | "
            f"{_fmt(row.get('future_h3_scale_residual_mean'))} | {_pct(delta.get('max_scale_metric_regression_percent'))} | "
            f"{_pct(delta.get('high_risk_taxonomy_fraction_delta_vs_c9'))} |"
        )
    report.extend(
        [
            "",
            f"Gate decision: `{gate['decision']}`.",
            f"TTT-family max regression: `{_pct(gate.get('ttt_family_max_regression'))}`; non-TTT-family max regression: `{_pct(gate.get('non_ttt_family_max_regression'))}`.",
            f"TTT-family max taxonomy shift: `{_pct(gate.get('ttt_family_max_taxonomy_shift'))}`; non-TTT-family max taxonomy shift: `{_pct(gate.get('non_ttt_family_max_taxonomy_shift'))}`.",
            "",
            "Evidence boundary: taxonomy is a v64 pose-only proxy over GT-local Sim(3), future scale residual, and global placement; duplicate raw overlap points are not available in these landed v43 artifacts.",
        ]
    )
    _write_text(phase / "c9_component_scale_ledger_report.md", report)
    return {
        "summaries": summaries,
        "delta_rows": delta_rows,
        "taxonomy_counts": tax_by_plan,
        "gate": gate,
        "by_summary": by_summary,
    }


def _phase0_gate(delta_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ttt_families = {"ttt_tri_gamma_chunk_map", "ttt_commit_ema", "ttt_tri_replay"}
    ttt = [r for r in delta_rows if r.get("component_family") in ttt_families]
    non = [r for r in delta_rows if r.get("component_family") not in ttt_families]
    ttt_reg = _max(r.get("max_scale_metric_regression_percent") for r in ttt)
    non_reg = _max(r.get("max_scale_metric_regression_percent") for r in non)
    ttt_tax = _max(r.get("high_risk_taxonomy_fraction_delta_vs_c9") for r in ttt)
    non_tax = _max(r.get("high_risk_taxonomy_fraction_delta_vs_c9") for r in non)
    reg_margin = (
        ttt_reg - non_reg
        if ttt_reg is not None and non_reg is not None and math.isfinite(ttt_reg) and math.isfinite(non_reg)
        else float("nan")
    )
    tax_margin = (
        ttt_tax - non_tax
        if ttt_tax is not None and non_tax is not None and math.isfinite(ttt_tax) and math.isfinite(non_tax)
        else float("nan")
    )
    positive = (
        (math.isfinite(reg_margin) and reg_margin >= 0.10)
        or (math.isfinite(tax_margin) and tax_margin >= 0.20)
    )
    return {
        "decision": "C9_TTT_SCALE_POSITIVE_BY_LANDED_LEDGER" if positive else "C9_TTT_SCALE_NOT_PROVEN_BY_PHASE0_LEDGER",
        "ttt_family_max_regression": ttt_reg,
        "non_ttt_family_max_regression": non_reg,
        "ttt_family_max_taxonomy_shift": ttt_tax,
        "non_ttt_family_max_taxonomy_shift": non_tax,
        "regression_margin": reg_margin,
        "taxonomy_shift_margin": tax_margin,
    }


def _state_hash(row: Mapping[str, Any]) -> Optional[str]:
    for key in ("hash_H_next", "hash_H_m_after_commit", "probe_ttt_write_state_hash", "ttt_semantic_scaled_state_hash", "commit_source_state_hash"):
        val = row.get(key)
        if val not in (None, ""):
            return str(val)
    return None


def _phase1(out_dir: Path, h35_full: Optional[RunData], c9_full: Optional[RunData], fork_runs: Mapping[str, RunData]) -> Dict[str, Any]:
    phase = out_dir / "phase1_state_propagation_audit"
    fig_dir = out_dir / "figures"
    lifecycle_rows: List[Dict[str, Any]] = []
    availability_rows: List[Dict[str, Any]] = []
    for run in [r for r in (h35_full, c9_full) if r is not None]:
        hmc_by = _row_by_chunk(run.hmc_rows)
        probe_by = _row_by_chunk(run.probe_rows)
        for chunk_id in sorted(set(hmc_by) | set(probe_by)):
            start, end = _chunk_range(run, chunk_id)
            probe = probe_by.get(chunk_id, {})
            hmc = hmc_by.get(chunk_id, {})
            next_probe = probe_by.get(chunk_id + 1, {})
            commit_hash = _state_hash(hmc)
            next_loaded = next_probe.get("hash_H_m_before_probe")
            lifecycle_rows.append(
                {
                    "run_key": run.key,
                    "run_name": run.run_name,
                    "chunk_id": chunk_id,
                    "frame_start": start,
                    "frame_end": end,
                    "reset_group_id": chunk_id // 5,
                    "reset_relative_idx": chunk_id % 5,
                    "reset_boundary_after_chunk_inferred": (chunk_id % 5 == 4),
                    "reset_reason": "inferred_every_5_chunks_from_v63_plan_and_prior_artifacts" if chunk_id % 5 == 4 else "",
                    "ttt_hash_before_probe": probe.get("hash_H_m_before_probe"),
                    "ttt_hash_after_probe": probe.get("hash_H_m_after_probe"),
                    "ttt_hash_after_commit": commit_hash,
                    "ttt_hash_loaded_next_probe": next_loaded,
                    "ttt_commit_to_next_probe_equal": (commit_hash == next_loaded) if commit_hash and next_loaded else None,
                    "swa_state_hash_before": None,
                    "swa_state_hash_after": None,
                    "merge_gauge_state_hash_before": None,
                    "merge_gauge_state_hash_after": None,
                    "frame_bias_state_hash_before": None,
                    "frame_bias_state_hash_after": None,
                    "component_hash_note": "TTT hash available; SWA/merge-gauge/frame-bias component hashes were not emitted by these runs.",
                }
            )
        availability_rows.append(
            {
                "run_key": run.key,
                "run_name": run.run_name,
                "ttt_component_hash_available": bool(run.probe_rows or run.hmc_rows),
                "swa_component_hash_available": False,
                "merge_gauge_component_hash_available": False,
                "frame_bias_component_hash_available": False,
                "supplement_status": "partial_ttt_only_from_existing_hashes",
            }
        )
    fork_rows = _fork_state_rows(fork_runs)
    _write_csv(phase / "state_lifecycle_by_chunk.csv", lifecycle_rows)
    _write_csv(phase / "component_hash_availability.csv", availability_rows)
    _write_csv(phase / "v63_chunk19_fork_state_propagation.csv", fork_rows)
    _plot_state_timeline(fig_dir / "state_propagation_timeline.png", lifecycle_rows)
    chunk19 = [r for r in fork_rows if r.get("fork_type") != "F0_BASE"]
    commit_changed = any(r.get("current_commit_hash_changed_vs_base") is True for r in chunk19)
    next_changed = any(r.get("next_probe_ttt_state_hash_changed_vs_base") is True for r in chunk19)
    report = [
        "# Phase 1 State Propagation Audit",
        "",
        "The audit supplements component-wise lifecycle rows from existing v63 hashes. Only the TTT fast-weight hash is available in these artifacts; SWA, merge/gauge, and frame-bias hashes are marked unavailable.",
        "",
        f"v63 chunk19 fork result: commit hash changed vs base = `{commit_changed}`, next-probe TTT hash changed vs base = `{next_changed}`.",
        f"chunk19 reset-relative index is `{V63_FORK_CHUNK % 5}`, so chunk20 is an inferred reset boundary.",
        "",
        "Conclusion: v63 chunk19 remains propagation-boundary evidence, not negative causal evidence against TTT scale. v64 causal forks must use reset-relative 1/2/3 chunks or explicitly prove next-probe hash changed.",
    ]
    _write_text(phase / "state_propagation_audit_report.md", report)
    return {"lifecycle_rows": lifecycle_rows, "availability_rows": availability_rows, "fork_rows": fork_rows}


def _pose_trace_by_chunk(run: RunData) -> Dict[int, Mapping[str, Any]]:
    out: Dict[int, Mapping[str, Any]] = {}
    for row in run.pose_trace_rows:
        if "chunk_idx" in row:
            out[int(row["chunk_idx"])] = row
    return out


def _current_output_pose_max_diff(base: RunData, other: RunData, chunk_id: int) -> Optional[float]:
    b = _pose_trace_by_chunk(base).get(chunk_id)
    o = _pose_trace_by_chunk(other).get(chunk_id)
    if not b or not o or "camera_poses" not in b or "camera_poses" not in o:
        return None
    ba = np.asarray(b["camera_poses"], dtype=np.float64)
    oa = np.asarray(o["camera_poses"], dtype=np.float64)
    n = min(ba.shape[0], oa.shape[0])
    if n <= 0:
        return None
    return float(np.max(np.abs(ba[:n] - oa[:n])))


def _fork_state_rows(fork_runs: Mapping[str, RunData]) -> List[Dict[str, Any]]:
    base = fork_runs.get("F0_BASE")
    if base is None:
        return []
    base_hmc = _row_by_chunk(base.hmc_rows)
    base_probe = _row_by_chunk(base.probe_rows)
    rows: List[Dict[str, Any]] = []
    for fork_type, run in fork_runs.items():
        hmc = _row_by_chunk(run.hmc_rows)
        probe = _row_by_chunk(run.probe_rows)
        cur = hmc.get(V63_FORK_CHUNK, {})
        bcur = base_hmc.get(V63_FORK_CHUNK, {})
        next_probe = probe.get(V63_FORK_CHUNK + 1, {})
        base_next_probe = base_probe.get(V63_FORK_CHUNK + 1, {})
        cur_hash = _state_hash(cur)
        base_hash = _state_hash(bcur)
        next_hash = next_probe.get("hash_H_m_before_probe")
        base_next_hash = base_next_probe.get("hash_H_m_before_probe")
        rows.append(
            {
                "fork_type": fork_type,
                "run_name": run.run_name,
                "chunk_id": V63_FORK_CHUNK,
                "reset_relative_idx": V63_FORK_CHUNK % 5,
                "reset_boundary_after_chunk_inferred": V63_FORK_CHUNK % 5 == 4,
                "current_output_pose_max_diff_vs_base": (
                    0.0 if fork_type == "F0_BASE" else _current_output_pose_max_diff(base, run, V63_FORK_CHUNK)
                ),
                "current_commit_hash": cur_hash,
                "base_current_commit_hash": base_hash,
                "current_commit_hash_changed_vs_base": (cur_hash != base_hash) if fork_type != "F0_BASE" and cur_hash and base_hash else False,
                "next_probe_ttt_hash": next_hash,
                "base_next_probe_ttt_hash": base_next_hash,
                "next_probe_ttt_state_hash_changed_vs_base": (
                    next_hash != base_next_hash
                    if fork_type != "F0_BASE" and next_hash is not None and base_next_hash is not None
                    else False
                ),
                "evidence_status": "propagation_boundary_if_commit_changed_but_next_probe_same",
            }
        )
    return rows


def _phase2(out_dir: Path, h35_full: Optional[RunData], c9_base: Optional[RunData]) -> Dict[str, Any]:
    phase = out_dir / "phase2_chunk_selection"
    if h35_full is None:
        _write_csv(phase / "selected_chunks.csv", [])
        return {"selected": []}
    h35_rows = _chunk_metrics(h35_full)
    c9_rows = _chunk_metrics(c9_base) if c9_base is not None else []
    c9_by = {int(r["chunk_id"]): r for r in c9_rows}
    candidates: List[Dict[str, Any]] = []
    for row in h35_rows:
        rel = int(row["reset_relative_idx"])
        cid = int(row["chunk_id"])
        if rel not in (1, 2, 3):
            continue
        c9 = c9_by.get(cid, {})
        post = _safe_float(row.get("post_zp_delta_norm_total"))
        tri = _safe_float(row.get("probe_ttt_write_tri_delta_norm_mean"))
        neg = _safe_float(row.get("probe_ttt_write_tri_neg_mass_mean"))
        future = _safe_float(row.get("future_h3_scale_residual"))
        c9_future = _safe_float(c9.get("future_h3_scale_residual"))
        gap = future - c9_future if math.isfinite(future) and math.isfinite(c9_future) else float("nan")
        runtime_score = sum(v for v in (post, tri, neg) if math.isfinite(v))
        oracle_score = sum(v for v in (future, gap) if math.isfinite(v))
        candidates.append(
            {
                "chunk_id": cid,
                "frame_start": row["frame_start"],
                "frame_end": row["frame_end"],
                "reset_group_id": row["reset_group_id"],
                "reset_relative_idx": rel,
                "runtime_score": runtime_score,
                "oracle_score": oracle_score,
                "h35_future_h3_scale_residual": row.get("future_h3_scale_residual"),
                "c9_future_h3_scale_residual": c9.get("future_h3_scale_residual"),
                "h35_minus_c9_future_h3_scale_residual": gap if math.isfinite(gap) else None,
                "h35_abs_log_scale_residual": row.get("abs_log_scale_residual"),
                "h35_global_chunk_ate": row.get("global_chunk_ate"),
                "post_zp_delta_norm_total": row.get("post_zp_delta_norm_total"),
                "tri_delta_norm_mean": row.get("probe_ttt_write_tri_delta_norm_mean"),
                "tri_neg_mass_mean": row.get("probe_ttt_write_tri_neg_mass_mean"),
            }
        )
    runtime = sorted(candidates, key=lambda r: _safe_float(r.get("runtime_score")), reverse=True)[:3]
    selected: List[Dict[str, Any]] = []
    for row in runtime:
        selected.append({**row, "selection_bucket": "runtime_style", "selection_reason": "top runtime TTT/post-zp proxy among reset-relative 1/2/3 chunks"})
    selected_ids = {int(r["chunk_id"]) for r in selected}
    for row in sorted(candidates, key=lambda r: _safe_float(r.get("oracle_score")), reverse=True):
        if int(row["chunk_id"]) in selected_ids:
            continue
        selected.append({**row, "selection_bucket": "oracle_diagnostic", "selection_reason": "top future scale / H35-C9 gap among reset-relative 1/2/3 chunks"})
        selected_ids.add(int(row["chunk_id"]))
        if len([r for r in selected if r["selection_bucket"] == "oracle_diagnostic"]) >= 3:
            break
    diagnostic_rel4 = [
        {
            "chunk_id": int(r["chunk_id"]),
            "reset_relative_idx": int(r["reset_relative_idx"]),
            "future_h3_scale_residual": r.get("future_h3_scale_residual"),
            "note": "rel4 is reset-boundary diagnostic only, not main fork selection",
        }
        for r in h35_rows
        if int(r["reset_relative_idx"]) == 4
    ]
    _write_csv(phase / "chunk_selection_candidates.csv", candidates)
    _write_csv(phase / "selected_chunks.csv", selected)
    _write_csv(phase / "reset_relative_4_diagnostic_chunks.csv", diagnostic_rel4)
    lines = [
        "# Phase 2 Chunk Selection",
        "",
        "Main selections avoid reset-relative 4. Runtime selections use H35 TTT/post-zp proxies; oracle selections use GT-derived future scale residual and H35-C9 future scale gap.",
        "",
        "| bucket | chunk | reset rel | frame range | runtime score | oracle score | reason |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    for row in selected:
        lines.append(
            f"| `{row['selection_bucket']}` | {row['chunk_id']} | {row['reset_relative_idx']} | "
            f"{row['frame_start']}..{row['frame_end']} | {_fmt(row.get('runtime_score'))} | {_fmt(row.get('oracle_score'))} | {row['selection_reason']} |"
        )
    _write_text(phase / "chunk_selection_report.md", lines)
    return {"selected": selected, "candidates": candidates}


def _read_kitti_ate(path: Path) -> Dict[str, Any]:
    log = path / "kitti_benchmark.log"
    if not log.is_file():
        return {}
    vals: List[str] = []
    with log.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            parts = raw.strip().split()
            if len(parts) >= 3 and parts[0] == "01":
                vals = parts
    if len(vals) >= 3:
        return {"ATE": _safe_float(vals[1]), "Rot": _safe_float(vals[2])}
    return {}


def _load_wall_summary(path: Path) -> Dict[str, Any]:
    p = path / "wall_time_summary.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _discover_v64_phase3_runs(gt: PoseData, registry: Mapping[str, Dict[str, Any]]) -> Dict[int, Dict[str, RunData]]:
    groups: Dict[int, Dict[str, RunData]] = {}
    if not V64_PHASE3_ROLLOUT_ROOT.is_dir():
        return groups
    pattern = re.compile(r"^V64_P3_C(?P<chunk>\d+)_(?P<fork>F\d+_[A-Z0-9_]+)_E(?P<end>\d+)$")
    for path in sorted(V64_PHASE3_ROLLOUT_ROOT.iterdir()):
        if not path.is_dir():
            continue
        m = pattern.match(path.name)
        if not m:
            continue
        chunk = int(m.group("chunk"))
        fork = m.group("fork")
        run = _load_run(
            {
                "key": fork,
                "plan_key": fork,
                "component_family": "v64_phase3_fork",
                "run_name": path.name,
                "path": path,
            },
            gt,
            registry,
        )
        if run is not None:
            groups.setdefault(chunk, {})[fork] = run
    return groups


def _fork_scale_label(fork: str) -> Optional[float]:
    if fork.startswith("F2_"):
        return 0.0
    if fork.startswith("F4_"):
        return 0.5
    if fork.startswith("F5_"):
        return 2.0
    if fork.startswith("F6_"):
        return -1.0
    return None


def _future_until_reset(row_by_chunk: Mapping[int, Mapping[str, Any]], chunk_id: int) -> Optional[float]:
    rel = chunk_id % 5
    horizon = max(0, 4 - rel)
    fut = [row_by_chunk.get(chunk_id + j, {}) for j in range(1, horizon + 1)]
    return _mean(r.get("abs_log_scale_residual") for r in fut)


def _future_cross_reset(row_by_chunk: Mapping[int, Mapping[str, Any]], chunk_id: int) -> Optional[float]:
    rel = chunk_id % 5
    next_reset_start = chunk_id + (5 - rel)
    fut = [row_by_chunk.get(next_reset_start + j, {}) for j in range(0, 2)]
    return _mean(r.get("abs_log_scale_residual") for r in fut)


def _phase3(out_dir: Path, phase3_runs: Mapping[int, Mapping[str, RunData]]) -> Dict[str, Any]:
    phase = out_dir / "phase3_within_reset_ttt_causal_fork"
    fig_dir = out_dir / "figures"
    registry_rows: List[Dict[str, Any]] = []
    future_rows: List[Dict[str, Any]] = []
    traj_rows: List[Dict[str, Any]] = []
    current_rows: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    for chunk_id, runs in sorted(phase3_runs.items()):
        base = runs.get("F0_BASE")
        if base is None:
            continue
        base_metrics = _chunk_metrics(base)
        base_by = {int(r["chunk_id"]): r for r in base_metrics}
        base_hmc = _row_by_chunk(base.hmc_rows)
        base_probe = _row_by_chunk(base.probe_rows)
        for fork, run in sorted(runs.items()):
            wall = _load_wall_summary(run.path)
            kitti = _read_kitti_ate(run.path)
            status = (run.path / "run_status.txt").read_text(encoding="utf-8", errors="replace") if (run.path / "run_status.txt").is_file() else ""
            registry_rows.append(
                {
                    "chunk_id": chunk_id,
                    "fork_type": fork,
                    "run_name": run.run_name,
                    "path": str(run.path),
                    "status_done": "DONE" in status,
                    "frames": run.poses.count,
                    "hmc_rows_raw": len(run.hmc_rows),
                    "hmc_rows_unique_chunks": _unique_chunk_count(run.hmc_rows),
                    "probe_rows_raw": len(run.probe_rows),
                    "probe_rows_unique_chunks": _unique_chunk_count(run.probe_rows),
                    "pose_trace_rows_raw": len(run.pose_trace_rows),
                    "pose_trace_rows_unique_chunks": _unique_chunk_count(run.pose_trace_rows),
                    "duplicate_trace_note": (
                        "raw_rows_exceed_unique_chunks; same run directory was rerun after an interrupted attempt"
                        if len(run.pose_trace_rows) > _unique_chunk_count(run.pose_trace_rows)
                        else ""
                    ),
                    "wall_seconds": wall.get("wall_seconds"),
                    "gpu": wall.get("gpu"),
                    "ATE": kitti.get("ATE"),
                    "Rot": kitti.get("Rot"),
                    "semantic_delta_scale": _fork_scale_label(fork),
                }
            )
            metrics = _chunk_metrics(run)
            by = {int(r["chunk_id"]): r for r in metrics}
            cur = by.get(chunk_id, {})
            bcur = base_by.get(chunk_id, {})
            hmc = _row_by_chunk(run.hmc_rows)
            probe = _row_by_chunk(run.probe_rows)
            cur_hash = _state_hash(hmc.get(chunk_id, {}))
            base_hash = _state_hash(base_hmc.get(chunk_id, {}))
            next_hash = probe.get(chunk_id + 1, {}).get("hash_H_m_before_probe")
            base_next_hash = base_probe.get(chunk_id + 1, {}).get("hash_H_m_before_probe")
            current_output_diff = 0.0 if fork == "F0_BASE" else _current_output_pose_max_diff(base, run, chunk_id)
            current_rows.append(
                {
                    "chunk_id": chunk_id,
                    "fork_type": fork,
                    "run_name": run.run_name,
                    "current_output_pose_max_diff_vs_base": current_output_diff,
                    "current_output_small_le_1e_minus_6": (
                        abs(_safe_float(current_output_diff)) <= 1e-6
                        if current_output_diff is not None
                        else None
                    ),
                    "current_commit_hash": cur_hash,
                    "base_current_commit_hash": base_hash,
                    "current_commit_hash_changed_vs_base": (cur_hash != base_hash) if fork != "F0_BASE" and cur_hash and base_hash else False,
                    "next_probe_ttt_hash": next_hash,
                    "base_next_probe_ttt_hash": base_next_hash,
                    "next_probe_ttt_state_hash_changed_vs_base": (
                        next_hash != base_next_hash
                        if fork != "F0_BASE" and next_hash is not None and base_next_hash is not None
                        else False
                    ),
                    "reset_boundary_after_chunk": chunk_id % 5 == 4,
                }
            )
            for horizon in (1, 2, 3, 5):
                if horizon in (1, 3, 5):
                    base_scale = _safe_float(bcur.get(f"future_h{horizon}_scale_residual"))
                    fork_scale = _safe_float(cur.get(f"future_h{horizon}_scale_residual"))
                else:
                    base_scale = _mean(base_by.get(chunk_id + j, {}).get("abs_log_scale_residual") for j in range(1, horizon + 1))
                    fork_scale = _mean(by.get(chunk_id + j, {}).get("abs_log_scale_residual") for j in range(1, horizon + 1))
                    base_scale = _safe_float(base_scale)
                    fork_scale = _safe_float(fork_scale)
                delta = fork_scale - base_scale if math.isfinite(fork_scale) and math.isfinite(base_scale) else float("nan")
                pct = delta / abs(base_scale) if math.isfinite(delta) and abs(base_scale) > EPS else float("nan")
                future_rows.append(
                    {
                        "chunk_id": chunk_id,
                        "fork_type": fork,
                        "run_name": run.run_name,
                        "horizon": f"h{horizon}",
                        "future_scale_residual_base": base_scale if math.isfinite(base_scale) else None,
                        "future_scale_residual_fork": fork_scale if math.isfinite(fork_scale) else None,
                        "future_scale_residual_delta": delta if math.isfinite(delta) else None,
                        "future_scale_residual_delta_percent": pct if math.isfinite(pct) else None,
                        "current_output_pose_max_diff_vs_base": current_output_diff,
                        "next_probe_ttt_state_hash_changed_vs_base": (
                            next_hash != base_next_hash
                            if fork != "F0_BASE" and next_hash is not None and base_next_hash is not None
                            else False
                        ),
                    }
                )
            for label, base_val, fork_val in (
                ("until_reset", _future_until_reset(base_by, chunk_id), _future_until_reset(by, chunk_id)),
                ("cross_reset_next2", _future_cross_reset(base_by, chunk_id), _future_cross_reset(by, chunk_id)),
            ):
                b = _safe_float(base_val)
                f = _safe_float(fork_val)
                d = f - b if math.isfinite(f) and math.isfinite(b) else float("nan")
                p = d / abs(b) if math.isfinite(d) and abs(b) > EPS else float("nan")
                future_rows.append(
                    {
                        "chunk_id": chunk_id,
                        "fork_type": fork,
                        "run_name": run.run_name,
                        "horizon": label,
                        "future_scale_residual_base": b if math.isfinite(b) else None,
                        "future_scale_residual_fork": f if math.isfinite(f) else None,
                        "future_scale_residual_delta": d if math.isfinite(d) else None,
                        "future_scale_residual_delta_percent": p if math.isfinite(p) else None,
                        "current_output_pose_max_diff_vs_base": current_output_diff,
                        "next_probe_ttt_state_hash_changed_vs_base": (
                            next_hash != base_next_hash
                            if fork != "F0_BASE" and next_hash is not None and base_next_hash is not None
                            else False
                        ),
                    }
                )
            for cid, row in by.items():
                brow = base_by.get(cid, {})
                traj_rows.append(
                    {
                        "target_chunk_id": chunk_id,
                        "fork_type": fork,
                        "chunk_id": cid,
                        "abs_log_scale_base": brow.get("abs_log_scale_residual"),
                        "abs_log_scale_fork": row.get("abs_log_scale_residual"),
                        "abs_log_scale_delta": _safe_float(row.get("abs_log_scale_residual")) - _safe_float(brow.get("abs_log_scale_residual")),
                        "global_chunk_ate_base": brow.get("global_chunk_ate"),
                        "global_chunk_ate_fork": row.get("global_chunk_ate"),
                    }
                )
        chunk_future = [r for r in future_rows if int(r.get("chunk_id", -1)) == chunk_id and r.get("horizon") == "h3" and r.get("fork_type") != "F0_BASE"]
        chunk_current = [r for r in current_rows if int(r.get("chunk_id", -1)) == chunk_id and r.get("fork_type") != "F0_BASE"]
        max_abs_h3 = _max(abs(r.get("future_scale_residual_delta_percent")) for r in chunk_future)
        current_small = all(r.get("current_output_small_le_1e_minus_6") is not False for r in chunk_current)
        any_next_changed = any(r.get("next_probe_ttt_state_hash_changed_vs_base") is True for r in chunk_current)
        all_next_changed = all(r.get("next_probe_ttt_state_hash_changed_vs_base") is True for r in chunk_current) if chunk_current else False
        positive = bool(max_abs_h3 is not None and max_abs_h3 >= 0.10 and current_small and any_next_changed)
        neg_forks = [r for r in chunk_future if str(r.get("fork_type")).startswith(("F1_", "F2_", "F3_", "F4_", "F5_", "F6_"))]
        negative = bool(
            neg_forks
            and all(abs(_safe_float(r.get("future_scale_residual_delta_percent"))) < 0.03 for r in neg_forks)
            and all_next_changed
        )
        if positive:
            decision = "TTT_SCALE_CAUSAL_POSITIVE"
        elif negative:
            decision = "TTT_SCALE_CAUSAL_NEGATIVE_FOR_PROPAGATED_FORKS"
        elif not any_next_changed:
            decision = "PROPAGATION_BLOCKER_NEXT_PROBE_UNCHANGED"
        else:
            decision = "TTT_SCALE_CAUSAL_INCONCLUSIVE"
        decisions.append(
            {
                "chunk_id": chunk_id,
                "decision": decision,
                "max_abs_h3_future_scale_delta_percent": max_abs_h3,
                "current_output_small_all": current_small,
                "any_next_probe_ttt_state_hash_changed": any_next_changed,
                "all_next_probe_ttt_state_hash_changed": all_next_changed,
                "f3_zero_post_zp_status": "not_run_no_independent_hook; F2 scale=0 is native/zero semantic delta",
            }
        )
    _write_csv(phase / "fork_run_registry.csv", registry_rows)
    _write_csv(phase / "fork_current_output_and_state_gate.csv", current_rows)
    _write_csv(phase / "fork_future_scale_delta_by_chunk.csv", future_rows)
    _write_csv(phase / "fork_future_trajectory_delta.csv", traj_rows)
    _write_csv(phase / "phase3_decision_by_chunk.csv", decisions)
    if registry_rows:
        _plot_fork_delta(fig_dir / "fork_future_scale_delta_by_chunk.png", future_rows)
        _plot_strength_curve(fig_dir / "fork_strength_response_curve.png", future_rows)
        lines = ["# Phase 3 Within-Reset TTT Causal Fork", ""]
        for row in decisions:
            lines.append(
                f"- chunk `{row['chunk_id']}` decision `{row['decision']}`; max abs h3 delta "
                f"`{_pct(row.get('max_abs_h3_future_scale_delta_percent'))}`, current output small `{row['current_output_small_all']}`, "
                f"any next-probe changed `{row['any_next_probe_ttt_state_hash_changed']}`."
            )
        lines.append("")
        lines.append("F3 zero-post-zp remains not run because no independent zero-post-zp-only hook exists; F2 `scale=0` is native/zero semantic delta.")
        _write_text(phase / "phase3_report.md", lines)
    return {"registry_rows": registry_rows, "future_rows": future_rows, "current_rows": current_rows, "decisions": decisions}


def _phase4_shape_audit(out_dir: Path, runs: Sequence[RunData]) -> Dict[str, Any]:
    phase = out_dir / "phase4_layer_branch_head_attribution"
    rows: List[Dict[str, Any]] = []
    layers = sorted({int(_safe_float(r.get("layer"))) for run in runs for r in run.layer_rows if math.isfinite(_safe_float(r.get("layer")))})
    branches = sorted({str(r.get("branch")) for run in runs for r in run.layer_rows if r.get("branch")})
    rows.append(
        {
            "source": "per_layer_branch_update_heatmap.csv",
            "layer_count": len(layers),
            "layers": layers,
            "branches": branches,
            "head_dimension_available": False,
            "head_dimension_unavailable": True,
            "using_channel_group_proxy": True,
            "intervention_status": "not_run_pending_positive_phase3_chunk_fork",
        }
    )
    _write_csv(phase / "layer_branch_head_shape_audit.csv", rows)
    _write_text(
        phase / "phase4_layer_branch_head_shape_audit.md",
        [
            "# Phase 4 Layer/Branch/Head Shape Audit",
            "",
            f"Observed layers in landed/v63 CSV artifacts: `{layers}`.",
            f"Observed branches: `{branches}`.",
            "",
            "`head_dimension_unavailable=true`; `using_channel_group_proxy=true` because the available summaries expose layer and branch rows but no explicit attention-head axis.",
            "Selective layer/branch/head interventions are not claimed here; they require a Phase3-positive chunk with verified next-probe TTT hash propagation.",
        ],
    )
    return {"rows": rows}


def _phase7_ttt_vs_merge_gauge(out_dir: Path, h35_full: Optional[RunData], phase3: Mapping[str, Any]) -> Dict[str, Any]:
    phase = out_dir / "phase7_ttt_vs_merge_gauge"
    fig_dir = out_dir / "figures"
    rows: List[Dict[str, Any]] = []
    if h35_full is not None:
        global_rmse = _rmse_from_delta(h35_full.global_residual)
        rows.extend(_output_alignment_oracle_rows(h35_full, global_rmse))
    decisions = phase3.get("decisions", [])
    max_ttt = _max(r.get("max_abs_h3_future_scale_delta_percent") for r in decisions)
    direct_rows = [
        {
            "path": "ttt_memory_only_phase3",
            "status": "measured",
            "metric": "max_abs_h3_future_scale_delta_percent",
            "value": max_ttt,
            "note": "max over propagated within-reset v64 Phase3 fork chunks",
        },
        {
            "path": "merge_gauge_only_direct_fork",
            "status": "blocked",
            "metric": "",
            "value": None,
            "note": "wrapper exposes SAVE_MERGE_STATES/LOAD_MERGE_STATE_AT_CHUNK but run_pipeline_abc_v2.py has no argparse or implementation for these flags",
        },
        {
            "path": "output_only_gt_sim3_oracle",
            "status": "diagnostic_only_measured" if rows else "no_data",
            "metric": "rmse_after_alignment",
            "value": min((_safe_float(r.get("rmse")) for r in rows), default=float("nan")),
            "note": "GT output alignment oracle; not deployable, used only to locate scale/gauge carrier outside TTT write",
        },
    ]
    _write_csv(phase / "ttt_vs_merge_gauge_registry.csv", direct_rows)
    _write_csv(phase / "output_alignment_oracle.csv", rows)
    _plot_ttt_vs_merge(fig_dir / "ttt_vs_merge_gauge_comparison.png", rows, max_ttt)
    best = min(rows, key=lambda r: _safe_float(r.get("rmse"))) if rows else {}
    if rows and (max_ttt is None or _safe_float(max_ttt) < 0.03):
        decision = "TTT_NOT_PRIMARY_SCALE_CARRIER_CURRENT_INTERFACE__NEXT_TARGET_OUTPUT_MERGE_GAUGE"
    else:
        decision = "TTT_VS_MERGE_GAUGE_INCONCLUSIVE"
    lines = [
        "# Phase 7 TTT vs Merge/Gauge",
        "",
        f"Decision: `{decision}`.",
        f"Measured TTT Phase3 max h3 scale effect: `{_pct(max_ttt)}`.",
        f"Best output alignment oracle: `{best.get('alignment_unit', 'NA')}` RMSE `{_fmt(best.get('rmse'))}`.",
        "",
        "Direct merge-gauge-only fork remains blocked: `tools/run_attention_cue_experiment.sh` passes merge-state flags, but `run_pipeline_abc_v2.py` currently has no corresponding argparse/implementation. This was audited in v64 and is not treated as a completed causal fork.",
        "The output oracle uses GT and is diagnostic-only; it shows whether scale/gauge is controllable after trajectory output, not a deployable method.",
    ]
    _write_text(phase / "ttt_vs_merge_gauge_report.md", lines)
    return {"rows": rows, "registry": direct_rows, "decision": decision, "max_ttt": max_ttt, "best": best}


def _output_alignment_oracle_rows(run: RunData, global_rmse: Optional[float]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if global_rmse is None:
        return rows
    rows.append(
        {
            "run_name": run.run_name,
            "alignment_unit": "global_sim3",
            "rmse": global_rmse,
            "improvement_vs_global_percent": 0.0,
            "diagnostic_only": True,
        }
    )
    chunk_rows = _chunk_metrics(run)
    by_group: Dict[int, List[Mapping[str, Any]]] = {}
    for row in chunk_rows:
        by_group.setdefault(int(row["reset_group_id"]), []).append(row)
    aligned_reset = np.full_like(run.poses.centers, np.nan)
    for group_rows in by_group.values():
        start = min(int(r["frame_start"]) for r in group_rows)
        end = max(int(r["frame_end"]) for r in group_rows)
        pred, gt = _slice(run, start, end)
        fit = _fit_sim3(pred, gt)
        if fit is not None:
            aligned_reset[start:end] = _apply_sim3(pred, fit)
    reset_rmse = _rmse_points(aligned_reset, run.gt.centers)
    rows.append(
        {
            "run_name": run.run_name,
            "alignment_unit": "per_reset_group_sim3_gt_oracle",
            "rmse": reset_rmse,
            "improvement_vs_global_percent": (global_rmse - reset_rmse) / global_rmse if reset_rmse is not None and global_rmse > EPS else None,
            "diagnostic_only": True,
        }
    )
    chunk_sq = 0.0
    chunk_count = 0
    chunk_scale_only_sq = 0.0
    chunk_scale_only_count = 0
    for row in chunk_rows:
        start = int(row["frame_start"])
        end = int(row["frame_end"])
        pred, gt = _slice(run, start, end)
        fit = _fit_sim3(pred, gt)
        if fit is not None:
            aligned = _apply_sim3(pred, fit)
            vals = _residual_norms(aligned, gt)
            chunk_sq += float(np.sum(vals * vals))
            chunk_count += int(vals.size)
            scale = fit[0]
            pred_c = pred - pred.mean(axis=0, keepdims=True)
            gt_c = gt - gt.mean(axis=0, keepdims=True)
            aligned_scale_only = pred_c * scale + gt.mean(axis=0, keepdims=True)
            vals2 = _residual_norms(aligned_scale_only, gt_c + gt.mean(axis=0, keepdims=True))
            chunk_scale_only_sq += float(np.sum(vals2 * vals2))
            chunk_scale_only_count += int(vals2.size)
    chunk_rmse = math.sqrt(chunk_sq / chunk_count) if chunk_count else None
    scale_only_rmse = math.sqrt(chunk_scale_only_sq / chunk_scale_only_count) if chunk_scale_only_count else None
    rows.append(
        {
            "run_name": run.run_name,
            "alignment_unit": "per_chunk_sim3_gt_oracle",
            "rmse": chunk_rmse,
            "improvement_vs_global_percent": (global_rmse - chunk_rmse) / global_rmse if chunk_rmse is not None and global_rmse > EPS else None,
            "diagnostic_only": True,
        }
    )
    rows.append(
        {
            "run_name": run.run_name,
            "alignment_unit": "per_chunk_scale_translation_only_gt_oracle",
            "rmse": scale_only_rmse,
            "improvement_vs_global_percent": (global_rmse - scale_only_rmse) / global_rmse if scale_only_rmse is not None and global_rmse > EPS else None,
            "diagnostic_only": True,
        }
    )
    return rows


def _write_required_no_data(out_dir: Path, *, phase3_has_data: bool = False, phase7_has_data: bool = False) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    placeholders = {
        "layer_scale_effect_heatmap.png": ("Layer intervention no-data", "Requires Phase3-positive propagated TTT fork."),
        "branch_scale_effect_bar.png": ("Branch intervention no-data", "Requires branch-selective commit intervention runs."),
        "head_or_channel_group_scale_effect.png": ("Head/channel proxy no-data", "No explicit head axis in current artifacts; channel proxy not run."),
        "token_group_scale_effect_bar.png": ("Token group intervention no-data", "Requires token-scope commit mask runs."),
        "image_region_scale_effect_heatmap.png": ("Image region intervention no-data", "No region-group mask fork completed."),
    }
    if not phase7_has_data:
        placeholders["ttt_vs_merge_gauge_comparison.png"] = (
            "TTT vs merge/gauge no-data",
            "Merge/gauge-only and output-only forks not completed.",
        )
    if not phase3_has_data:
        placeholders.update(
            {
                "fork_future_scale_delta_by_chunk.png": ("Phase3 forks not run in v64 yet", "Chunk selections are prepared; within-reset fork jobs still pending."),
                "fork_strength_response_curve.png": ("Phase3 strength curve not run in v64 yet", "Requires F0-F7 fork matrix on selected chunks."),
            }
        )
    for name, (title, note) in placeholders.items():
        _plot_no_data(fig_dir / name, title, note)
    if not phase3_has_data:
        phase3 = out_dir / "phase3_within_reset_ttt_causal_fork"
        _write_csv(phase3 / "fork_run_registry.csv", [])
        _write_csv(phase3 / "fork_future_scale_delta_by_chunk.csv", [])
        _write_text(
            phase3 / "phase3_status.md",
            [
                "# Phase 3 Status",
                "",
                "No v64 within-reset F0-F7 fork matrix has been executed yet. v63 chunk19 rows are audited in Phase1 and are not treated as negative TTT evidence because next-probe TTT state did not change across the reset boundary.",
            ],
        )
    phase5 = out_dir / "phase5_token_region_attribution"
    _write_csv(phase5 / "token_group_intervention_registry.csv", [])
    _write_text(
        phase5 / "phase5_status.md",
        [
            "# Phase 5 Status",
            "",
            "Token/region group interventions are not run yet. No group is credited or ruled out.",
        ],
    )
    phase6 = out_dir / "phase6_c9_style_positive_injection"
    _write_csv(phase6 / "c9_style_injection_registry.csv", [])


def _write_final_report(
    out_dir: Path,
    phase0: Mapping[str, Any],
    phase1: Mapping[str, Any],
    phase2: Mapping[str, Any],
    phase3: Mapping[str, Any],
    phase7: Mapping[str, Any],
    shape: Mapping[str, Any],
) -> None:
    summaries = phase0.get("summaries", [])
    delta_rows = phase0.get("delta_rows", [])
    gate = phase0.get("gate", {})
    selected = phase2.get("selected", [])
    p3_decisions = phase3.get("decisions", [])
    selected_ids = _chunk_ids_from_rows(selected)
    decision_ids = _chunk_ids_from_rows(p3_decisions)
    p3_selected_done = bool(selected_ids) and selected_ids.issubset(decision_ids)
    phase7_decision = phase7.get("decision")
    top_ttt = [r for r in delta_rows if r.get("component_family") in {"ttt_tri_gamma_chunk_map", "ttt_commit_ema", "ttt_tri_replay"}]
    top_ttt = sorted(top_ttt, key=lambda r: _safe_float(r.get("max_scale_metric_regression_percent")), reverse=True)
    fork_rows = phase1.get("fork_rows", [])
    commit_changed = any(r.get("current_commit_hash_changed_vs_base") is True for r in fork_rows if r.get("fork_type") != "F0_BASE")
    next_changed = any(r.get("next_probe_ttt_state_hash_changed_vs_base") is True for r in fork_rows if r.get("fork_type") != "F0_BASE")
    lines = [
        "# ACL2 v64 TTT-Scale Mechanism Attribution Report",
        "",
        "Diagnostic-only report. Numeric scale metrics are recomputed from trajectories; missing intervention data is marked no-data instead of inferred.",
        "",
        "## Current Decision",
        "",
        f"- Phase0 landed C9 component gate: `{gate.get('decision')}`.",
        f"- v63 chunk19 propagation audit: commit changed `{commit_changed}`, next-probe changed `{next_changed}`; chunk19 remains reset-boundary evidence, not negative TTT evidence.",
        f"- v64 selected main chunks: `{[r.get('chunk_id') for r in selected]}`.",
        f"- v64 Phase3 selected chunks covered by F0/F1/F2/F4/F5/F6: `{p3_selected_done}`; decisions: `{p3_decisions}`.",
        f"- Phase7 TTT vs output/merge-gauge decision: `{phase7_decision}`.",
        "",
        "## Eight Plan Questions",
        "",
        f"1. C9 TTT components and scale/gauge: Phase0 ledger decision `{gate.get('decision')}`. Strongest TTT-family row is `{top_ttt[0].get('plan_key') if top_ttt else 'NA'}` with max scale metric regression `{_pct(top_ttt[0].get('max_scale_metric_regression_percent') if top_ttt else None)}`.",
        f"2. Single-chunk TTT write future scale: Phase3 v64 selected chunks are covered by F0/F1/F2/F4/F5/F6 = `{p3_selected_done}`; decisions `{p3_decisions if p3_decisions else 'not_run'}`. v63 chunk19 is invalid as negative evidence because next-probe TTT hash did not change. F3 zero-post-zp and F7 C9-style injection are not claimed because no independent safe hooks were available in this run.",
        "3. Layer/branch/head: shape audit is complete; explicit head dimension is unavailable, so future head work must use channel/group proxy. No selective intervention is claimed yet.",
        "4. Token/region: not run yet; no token group is credited or ruled out.",
        f"5. If TTT is not carrier: Phase7 decision `{phase7_decision}`. Direct merge-gauge-only fork remains blocked by missing CLI implementation; output GT oracle is diagnostic-only evidence for output/merge-gauge as next target.",
        "6. C9-style positive injection: not run yet; no injection result is claimed.",
        "7. State propagation: TTT component hash is available; SWA/merge-gauge/frame-bias component hashes are missing in current artifacts and must be added if Phase7 becomes decisive.",
        "8. Next action: implement/audit a real merge-gauge state save/load or merge-transform trace hook, because Phase3 propagated TTT forks are negative and direct merge-gauge-only fork is currently blocked.",
        "",
        "## Evidence Files",
        "",
        "- `phase0_c9_component_scale_ledger/component_scale_summary.csv`",
        "- `phase0_c9_component_scale_ledger/component_scale_delta_vs_c9.csv`",
        "- `phase1_state_propagation_audit/state_lifecycle_by_chunk.csv`",
        "- `phase1_state_propagation_audit/v63_chunk19_fork_state_propagation.csv`",
        "- `phase2_chunk_selection/selected_chunks.csv`",
        "- `phase4_layer_branch_head_attribution/phase4_layer_branch_head_shape_audit.md`",
    ]
    _write_text(out_dir / "v64_ttt_scale_mechanism_attribution_report.md", lines)
    _write_json(
        out_dir / "v64_summary.json",
        {
            "phase0_gate": gate,
            "selected_chunks": selected,
            "phase3_decisions": p3_decisions,
            "phase7_decision": phase7_decision,
            "phase7": phase7,
            "phase1_fork_rows": fork_rows,
            "shape_audit": shape,
            "completion_status": (
                "diagnostics_phase0_phase1_phase2_complete_phase3_selected_f0_f1_f2_f4_f5_f6_complete_phase7_merge_gauge_blocked"
                if p3_selected_done
                else "partial_diagnostics_phase0_phase1_phase2_complete_phase3_selected_pending"
            ),
        },
    )


def _plot_component_ledger(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(r.get("plan_key")).replace("C9_MINUS_", "").replace("_", "\n") for r in rows]
    vals = [_safe_float(r.get("max_scale_metric_regression_percent")) * 100.0 for r in rows]
    if not rows:
        _plot_no_data(path, "C9 component scale ledger no-data", "No component rows loaded.")
        return
    fig, ax = plt.subplots(figsize=(12, 4.5))
    colors = ["#c44e52" if "TTT" in str(r.get("plan_key")) or "TRI" in str(r.get("plan_key")) or "EMA" in str(r.get("plan_key")) else "#4c72b0" for r in rows]
    ax.bar(range(len(rows)), vals, color=colors)
    ax.axhline(10.0, color="#222222", linestyle="--", linewidth=1, label="10% gate")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.set_ylabel("max scale metric regression vs C9 (%)")
    ax.set_title("C9 landed component scale ledger (trajectory recomputed)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_reset_profile(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        _plot_no_data(path, "Reset-relative profile no-data", "No chunk rows.")
        return
    plans = []
    for row in rows:
        plan = str(row.get("plan_key"))
        if plan not in plans:
            plans.append(plan)
    fig, ax = plt.subplots(figsize=(10, 5))
    for plan in plans[:8]:
        sub = [r for r in rows if str(r.get("plan_key")) == plan]
        xs = sorted({int(r["reset_relative_idx"]) for r in sub})
        ys = [_mean(r.get("future_h3_scale_residual") for r in sub if int(r["reset_relative_idx"]) == x) for x in xs]
        ax.plot(xs, ys, marker="o", linewidth=1.2, label=plan.replace("C9_MINUS_", "MINUS_"))
    ax.set_xlabel("reset-relative idx")
    ax.set_ylabel("mean future h3 abs log-scale residual")
    ax.set_title("Reset-relative TTT/scale profile (pose-only)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_state_timeline(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sub = [r for r in rows if str(r.get("run_key")) == "H35_FULL_TRACE"]
    if not sub:
        _plot_no_data(path, "State propagation timeline no-data", "No H35 full trace lifecycle rows.")
        return
    xs = [int(r["chunk_id"]) for r in sub]
    ys = [1.0 if r.get("ttt_commit_to_next_probe_equal") is True else 0.0 if r.get("ttt_commit_to_next_probe_equal") is False else np.nan for r in sub]
    rel = [int(r["reset_relative_idx"]) for r in sub]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(xs, ys, marker="o", linewidth=1.0)
    for x, rr in zip(xs, rel):
        if rr == 4:
            ax.axvline(x + 0.5, color="#c44e52", alpha=0.25, linewidth=1)
    ax.set_ylim(-0.15, 1.15)
    ax.set_xlabel("chunk")
    ax.set_ylabel("commit hash == next probe hash")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["false", "true"])
    ax.set_title("TTT state propagation timeline; red bands mark inferred reset boundary")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_fork_delta(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    sub = [r for r in rows if r.get("horizon") == "h3" and r.get("fork_type") != "F0_BASE"]
    if not sub:
        _plot_no_data(path, "Phase3 h3 fork deltas no-data", "No h3 fork rows.")
        return
    labels = [f"C{r.get('chunk_id')} {str(r.get('fork_type')).replace('_', ' ')}" for r in sub]
    vals = [_safe_float(r.get("future_scale_residual_delta_percent")) * 100.0 for r in sub]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.4), 4.5))
    ax.bar(range(len(labels)), vals, color="#4c72b0")
    ax.axhline(10.0, color="#c44e52", linestyle="--", linewidth=1)
    ax.axhline(-10.0, color="#c44e52", linestyle="--", linewidth=1)
    ax.axhline(3.0, color="#222222", linestyle=":", linewidth=1)
    ax.axhline(-3.0, color="#222222", linestyle=":", linewidth=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("h3 future scale delta (%)")
    ax.set_title("Phase3 future scale delta by chunk/fork")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_strength_curve(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    sub = [
        r
        for r in rows
        if r.get("horizon") == "h3" and _fork_scale_label(str(r.get("fork_type"))) is not None
    ]
    if not sub:
        _plot_no_data(path, "Phase3 strength curve no-data", "No scale-labeled fork rows.")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    chunks = sorted({int(r["chunk_id"]) for r in sub})
    for chunk in chunks:
        cs = [r for r in sub if int(r["chunk_id"]) == chunk]
        pairs = sorted(
            [
                (_fork_scale_label(str(r.get("fork_type"))), _safe_float(r.get("future_scale_residual_fork")))
                for r in cs
                if _fork_scale_label(str(r.get("fork_type"))) is not None
            ],
            key=lambda x: x[0],
        )
        ax.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", label=f"chunk {chunk}")
    ax.set_xlabel("semantic TTT delta scale")
    ax.set_ylabel("h3 future abs log-scale residual")
    ax.set_title("Phase3 TTT write strength response")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_ttt_vs_merge(path: Path, rows: Sequence[Mapping[str, Any]], max_ttt: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["TTT max h3 delta (%)"]
    vals = [_safe_float(max_ttt) * 100.0]
    for row in rows:
        labels.append(str(row.get("alignment_unit")).replace("_", "\n"))
        vals.append(_safe_float(row.get("rmse")))
    if len(labels) == 1:
        _plot_no_data(path, "TTT vs merge/gauge no-data", "No Phase7 rows.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].bar([0], [vals[0]], color="#c44e52")
    axes[0].axhline(3.0, color="#222222", linestyle=":", linewidth=1)
    axes[0].set_xticks([0])
    axes[0].set_xticklabels(["TTT\nPhase3"], fontsize=8)
    axes[0].set_ylabel("max h3 scale delta (%)")
    axes[0].set_title("TTT memory-only effect")
    axes[1].bar(range(len(rows)), [vals[i + 1] for i in range(len(rows))], color="#4c72b0")
    axes[1].set_xticks(range(len(rows)))
    axes[1].set_xticklabels([str(r.get("alignment_unit")).replace("_", "\n") for r in rows], fontsize=8)
    axes[1].set_ylabel("RMSE after output alignment")
    axes[1].set_title("Output alignment oracle (GT diagnostic-only)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_no_data(path: Path, title: str, note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=14, weight="bold", wrap=True)
    ax.text(0.5, 0.38, note, ha="center", va="center", fontsize=10, wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-poses", type=Path, default=GT_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = _read_poses(args.gt_poses)
    registry = _load_registry_rows()
    component_runs = [r for spec in COMPONENT_RUNS if (r := _load_run(spec, gt, registry)) is not None]
    if not component_runs:
        raise RuntimeError("No component runs loaded; cannot build v64 ledger.")
    h35_full = _load_run(
        {"key": "H35_FULL_TRACE", "plan_key": "H35_FULL_TRACE", "component_family": "h35_trace", "run_name": V63_H35_FULL.name, "path": V63_H35_FULL},
        gt,
        registry,
    )
    c9_full = _load_run(
        {"key": "C9_FULL_TRACE", "plan_key": "C9_FULL_TRACE", "component_family": "c9_trace", "run_name": V63_C9_FULL.name, "path": V63_C9_FULL},
        gt,
        registry,
    )
    fork_runs = {
        key: run
        for key, path in V63_FORKS.items()
        if (
            run := _load_run(
                {"key": key, "plan_key": key, "component_family": "v63_fork", "run_name": path.name, "path": path},
                gt,
                registry,
            )
        )
        is not None
    }
    phase0 = _phase0(out_dir, component_runs)
    phase1 = _phase1(out_dir, h35_full, c9_full, fork_runs)
    c9_base = next((r for r in component_runs if r.plan_key == "C9_P0_R2"), None)
    phase2 = _phase2(out_dir, h35_full, c9_base)
    phase3_runs = _discover_v64_phase3_runs(gt, registry)
    phase3 = _phase3(out_dir, phase3_runs)
    shape = _phase4_shape_audit(out_dir, [*component_runs, *([h35_full] if h35_full else [])])
    phase7 = _phase7_ttt_vs_merge_gauge(out_dir, h35_full, phase3)
    _write_required_no_data(
        out_dir,
        phase3_has_data=bool(phase3.get("registry_rows")),
        phase7_has_data=bool(phase7.get("rows")),
    )
    _write_final_report(out_dir, phase0, phase1, phase2, phase3, phase7, shape)


if __name__ == "__main__":
    main()
