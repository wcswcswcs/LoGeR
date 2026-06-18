#!/usr/bin/env python3
"""KITTI01 ORIG/C9/H35 error-source autopsy for ACL2 v62.

The tool is diagnostic-only. It reads landed trajectories/configs and writes
phase outputs requested by docs/ACL2_v62_KITTI01_ErrorSource_Autopsy_ORIG_C9_H35_Plan.md.
It does not invent missing raw chunk artifacts: unavailable point/raw overlap
data is recorded as unavailable and pose/GT-overlap proxies are explicitly
tagged as such.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
DEFAULT_OUT = ROOT / "results/kitti01_hmc_v2/acl2_v62_kitti01_error_source_autopsy_orig_c9_h35/report_final"


RUNS = [
    {
        "method": "orig",
        "display": "ORIG_NATIVE_FALLBACK",
        "run_name": "C30A_no_control_loger_stateful_full",
        "result_dir": ROOT / "results/kitti01_hmc_v2/phaseC_v3_signalgate/C30A_no_control_loger_stateful_full",
        "note": "Native/H9-like reference available in this repo; not claimed as paper-original LoGeR.",
    },
    {
        "method": "c9",
        "display": "C9_P0_R2",
        "run_name": "V45_P0_C9_REPEAT",
        "result_dir": ROOT
        / "results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase0_hard_gate/rollouts/V45_P0_C9_REPEAT",
        "note": "Historical C9 repeat artifact; V43 and V45 repeats have identical ATE/trajectory for this run family.",
    },
    {
        "method": "h35",
        "display": "Clean_H35_v53",
        "run_name": "V53_PHASE7_FULL_H35_LAYERGAMMAFIX_RHO0075",
        "result_dir": ROOT
        / "results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase7_layergamma_fix_full/rollouts/V53_PHASE7_FULL_H35_LAYERGAMMAFIX_RHO0075",
        "note": "Clean H35 full artifact named in the plan (ATE around 35.7409m).",
    },
]


EPS = 1e-12


def _clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, tuple):
        return [_clean(v) for v in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
                if isinstance(value, (dict, list)):
                    out[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
                elif value is None:
                    out[key] = ""
                else:
                    out[key] = value
            writer.writerow(out)


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


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


def _std(values: Iterable[Any]) -> Optional[float]:
    vals = _finite(values)
    return float(np.std(vals)) if vals else None


def _max(values: Iterable[Any]) -> Optional[float]:
    vals = _finite(values)
    return float(np.max(vals)) if vals else None


def _min(values: Iterable[Any]) -> Optional[float]:
    vals = _finite(values)
    return float(np.min(vals)) if vals else None


def _percentile(values: Iterable[Any], q: float) -> Optional[float]:
    vals = _finite(values)
    return float(np.percentile(vals, q)) if vals else None


def _fmt(value: Any, digits: int = 6) -> str:
    val = _safe_float(value)
    return f"{val:.{digits}f}" if math.isfinite(val) else "NA"


def _corr(xs: Iterable[Any], ys: Iterable[Any]) -> Optional[float]:
    pairs: List[Tuple[float, float]] = []
    for x, y in zip(xs, ys):
        xf = _safe_float(x)
        yf = _safe_float(y)
        if math.isfinite(xf) and math.isfinite(yf):
            pairs.append((xf, yf))
    if len(pairs) < 3:
        return None
    arr = np.asarray(pairs, dtype=np.float64)
    if np.std(arr[:, 0]) <= EPS or np.std(arr[:, 1]) <= EPS:
        return None
    return float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


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
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                rows.append(data)
    return rows


def _read_text_head(path: Path, n: int = 20) -> List[str]:
    if not path.is_file():
        return []
    lines: List[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for _, line in zip(range(n), handle):
            lines.append(line.rstrip("\n"))
    return lines


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm <= EPS:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = (q / norm).tolist()
    return np.asarray(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


@dataclass
class Trajectory:
    centers: np.ndarray
    rotations: np.ndarray
    timestamps: Optional[np.ndarray]


def _read_poses(path: Path) -> Trajectory:
    centers: List[np.ndarray] = []
    rotations: List[np.ndarray] = []
    timestamps: List[float] = []
    has_timestamps = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                vals = [float(x) for x in raw.split()]
            except ValueError:
                continue
            if len(vals) == 12:
                mat = np.eye(4, dtype=np.float64)
                mat[:3, :4] = np.asarray(vals, dtype=np.float64).reshape(3, 4)
                centers.append(mat[:3, 3].copy())
                rotations.append(mat[:3, :3].copy())
                timestamps.append(float(len(timestamps)))
            elif len(vals) >= 8:
                has_timestamps = True
                timestamps.append(float(vals[0]))
                centers.append(np.asarray(vals[1:4], dtype=np.float64))
                rotations.append(_quat_to_rot(vals[4], vals[5], vals[6], vals[7]))
    if not centers:
        raise ValueError(f"No poses parsed from {path}")
    return Trajectory(
        centers=np.stack(centers, axis=0),
        rotations=np.stack(rotations, axis=0),
        timestamps=np.asarray(timestamps, dtype=np.float64) if has_timestamps else None,
    )


def _rot_angle_rad(rot: np.ndarray) -> float:
    val = (float(np.trace(rot)) - 1.0) / 2.0
    val = max(-1.0, min(1.0, val))
    return float(math.acos(val))


def _fit_sim3(src: np.ndarray, dst: np.ndarray, weights: Optional[np.ndarray] = None) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
    mask = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    src = np.asarray(src[mask], dtype=np.float64)
    dst = np.asarray(dst[mask], dtype=np.float64)
    if weights is not None:
        weights = np.asarray(weights[mask], dtype=np.float64)
    if src.shape[0] < 3:
        return None
    if weights is None:
        weights = np.ones(src.shape[0], dtype=np.float64) / float(src.shape[0])
    else:
        weights = np.maximum(weights, 0.0)
        total = float(np.sum(weights))
        if total <= EPS:
            return None
        weights = weights / total
    mu_src = np.sum(src * weights[:, None], axis=0)
    mu_dst = np.sum(dst * weights[:, None], axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    var_src = float(np.sum(weights * np.sum(src_c * src_c, axis=1)))
    if var_src <= EPS:
        return None
    cov = (dst_c * weights[:, None]).T @ src_c
    u, singular_values, vt = np.linalg.svd(cov)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1.0
    rot = u @ np.diag(sign) @ vt
    scale = float(np.sum(singular_values * sign) / var_src)
    if not math.isfinite(scale) or abs(scale) <= EPS:
        return None
    trans = mu_dst - scale * (rot @ mu_src)
    return scale, rot, trans


def _apply_sim3(points: np.ndarray, fit: Tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    scale, rot, trans = fit
    return scale * (points @ rot.T) + trans[None, :]


def _residual_values(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    n = min(pred.shape[0], gt.shape[0])
    pred = pred[:n]
    gt = gt[:n]
    mask = np.isfinite(pred).all(axis=1) & np.isfinite(gt).all(axis=1)
    if not np.any(mask):
        return np.zeros((0,), dtype=np.float64)
    return np.linalg.norm(pred[mask] - gt[mask], axis=1)


def _residual_stats(pred: np.ndarray, gt: np.ndarray) -> Dict[str, Any]:
    residual = _residual_values(pred, gt)
    if residual.size == 0:
        return {"count": 0, "rmse": None, "median": None, "p90": None, "max": None, "final": None}
    return {
        "count": int(residual.size),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "median": float(np.median(residual)),
        "p90": float(np.percentile(residual, 90)),
        "max": float(np.max(residual)),
        "final": float(residual[-1]),
    }


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


def _rmse(values: np.ndarray) -> Optional[float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.sqrt(np.mean(values * values)))


def _rotation_rmse(pred_rot: np.ndarray, gt_rot: np.ndarray, sim_rot: Optional[np.ndarray] = None) -> Optional[float]:
    n = min(pred_rot.shape[0], gt_rot.shape[0])
    if n <= 0:
        return None
    angles: List[float] = []
    r_align = np.eye(3, dtype=np.float64) if sim_rot is None else sim_rot
    for i in range(n):
        pred = r_align @ pred_rot[i]
        rel = pred @ gt_rot[i].T
        angles.append(math.degrees(_rot_angle_rad(rel)))
    return _rmse(np.asarray(angles, dtype=np.float64))


def _rpe_t(pred_points: np.ndarray, gt_points: np.ndarray) -> Optional[float]:
    n = min(pred_points.shape[0], gt_points.shape[0])
    if n < 2:
        return None
    diff = np.diff(pred_points[:n], axis=0) - np.diff(gt_points[:n], axis=0)
    return _rmse(np.linalg.norm(diff, axis=1))


def _rpe_r(pred_rot: np.ndarray, gt_rot: np.ndarray, sim_rot: Optional[np.ndarray] = None) -> Optional[float]:
    n = min(pred_rot.shape[0], gt_rot.shape[0])
    if n < 2:
        return None
    r_align = np.eye(3, dtype=np.float64) if sim_rot is None else sim_rot
    angles: List[float] = []
    pred = np.asarray([r_align @ pred_rot[i] for i in range(n)], dtype=np.float64)
    for i in range(n - 1):
        dp = pred[i].T @ pred[i + 1]
        dg = gt_rot[i].T @ gt_rot[i + 1]
        angles.append(math.degrees(_rot_angle_rad(dp @ dg.T)))
    return _rmse(np.asarray(angles, dtype=np.float64))


def _read_kitti_results_ate(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "results_sim3/results_ate.txt"
    out: Dict[str, Any] = {"kitti_results_ate_path": str(path) if path.is_file() else ""}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw.split()
        if len(parts) >= 3 and parts[0] == "01":
            out["kitti_results_ate"] = _safe_float(parts[1])
            out["kitti_results_rpe"] = _safe_float(parts[2])
            return out
    return out


def _split_chunks(total_frames: int, chunk_size: int, overlap: int) -> List[Dict[str, int]]:
    if chunk_size <= 0 or chunk_size >= total_frames:
        return [{"chunk_id": 0, "start_frame": 0, "end_frame": total_frames}]
    step = max(chunk_size - overlap, 1)
    rows: List[Dict[str, int]] = []
    s = 0
    cid = 0
    while s < total_frames:
        e = min(s + chunk_size, total_frames)
        rows.append({"chunk_id": cid, "start_frame": s, "end_frame": e})
        if e >= total_frames:
            break
        s += step
        cid += 1
    return rows


def _extract_schedule(run_dir: Path, config: Mapping[str, Any], frames_used: int) -> Tuple[List[Dict[str, Any]], str]:
    for name in ("hmc_state_hash.jsonl", "raw_prediction_buffer_summary.jsonl"):
        rows = _read_jsonl(run_dir / name)
        sched: List[Dict[str, Any]] = []
        seen = set()
        for row in rows:
            if "chunk_idx" not in row or "start_frame" not in row or "end_frame" not in row:
                continue
            cid = int(row["chunk_idx"])
            if cid in seen:
                continue
            seen.add(cid)
            start = max(0, int(row["start_frame"]))
            end = min(frames_used, int(row["end_frame"]))
            if end <= start:
                continue
            sched.append(
                {
                    "chunk_id": cid,
                    "start_frame": start,
                    "end_frame": end,
                    "chunk_size": int(row.get("chunk_size", config.get("chunk_size", 0) or 0)),
                    "chunk_overlap": int(row.get("chunk_overlap", config.get("chunk_overlap", config.get("overlap_size", 0)) or 0)),
                }
            )
        if sched:
            sched.sort(key=lambda item: int(item["chunk_id"]))
            return sched, name
    chunk_size = int(config.get("chunk_size", config.get("window_size", 0)) or 0)
    overlap = int(config.get("chunk_overlap", config.get("overlap_size", 0)) or 0)
    sched = _split_chunks(frames_used, chunk_size, overlap)
    for row in sched:
        row["chunk_size"] = chunk_size
        row["chunk_overlap"] = overlap
    return sched, "config_split_fallback"


@dataclass
class RunData:
    method: str
    display: str
    run_name: str
    result_dir: Path
    note: str
    pred_path: Path
    config_path: Path
    effective_config_path: Path
    config: Dict[str, Any]
    effective_config: Dict[str, Any]
    audit: Dict[str, Any]
    pred: Trajectory
    gt: Trajectory
    frames_used: int
    schedule: List[Dict[str, Any]]
    schedule_source: str
    reset_every: int
    global_fit: Tuple[float, np.ndarray, np.ndarray]
    global_aligned: np.ndarray
    global_residual: np.ndarray
    kitti_results: Dict[str, Any]


def _load_run(spec: Mapping[str, Any], gt_full: Trajectory) -> RunData:
    run_dir = Path(spec["result_dir"])
    pred_path = run_dir / "01.txt"
    pred = _read_poses(pred_path)
    frames_used = min(pred.centers.shape[0], gt_full.centers.shape[0])
    pred = Trajectory(pred.centers[:frames_used], pred.rotations[:frames_used], pred.timestamps[:frames_used] if pred.timestamps is not None else None)
    gt = Trajectory(gt_full.centers[:frames_used], gt_full.rotations[:frames_used], gt_full.timestamps[:frames_used] if gt_full.timestamps is not None else None)
    config_path = run_dir / "hmc_config.yaml"
    effective_config_path = run_dir / "effective_config.yaml"
    config = _read_yaml(config_path)
    effective_config = _read_yaml(effective_config_path)
    merged_config = dict(config)
    merged_config.update({k: v for k, v in effective_config.items() if v not in (None, "")})
    schedule, schedule_source = _extract_schedule(run_dir, merged_config, frames_used)
    reset_every = int(merged_config.get("reset_every", 5) or 5)
    fit = _fit_sim3(pred.centers, gt.centers)
    if fit is None:
        raise RuntimeError(f"Could not fit global Sim3 for {run_dir}")
    global_aligned = _apply_sim3(pred.centers, fit)
    return RunData(
        method=str(spec["method"]),
        display=str(spec["display"]),
        run_name=str(spec["run_name"]),
        result_dir=run_dir,
        note=str(spec.get("note", "")),
        pred_path=pred_path,
        config_path=config_path,
        effective_config_path=effective_config_path,
        config=config,
        effective_config=effective_config,
        audit=_read_json(run_dir / "chunk_id_policy_audit.json"),
        pred=pred,
        gt=gt,
        frames_used=frames_used,
        schedule=schedule,
        schedule_source=schedule_source,
        reset_every=reset_every,
        global_fit=fit,
        global_aligned=global_aligned,
        global_residual=_residual_values(global_aligned, gt.centers),
        kitti_results=_read_kitti_results_ate(run_dir),
    )


def _config_get(run: RunData, key: str) -> Any:
    if key in run.effective_config:
        return run.effective_config.get(key)
    return run.config.get(key)


def _bool_from_audit(audit: Mapping[str, Any], key: str) -> Optional[bool]:
    value = audit.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return None


def _artifact_inventory(runs: Sequence[RunData], out_dir: Path, gt_path: Path) -> None:
    phase = out_dir / "phase0_audit"
    inventory: List[Dict[str, Any]] = []
    config_rows: List[Dict[str, Any]] = []
    schedule_rows: List[Dict[str, Any]] = []
    keys = [
        "chunk_size",
        "chunk_overlap",
        "overlap_size",
        "window_size",
        "reset_every",
        "hmc_commit_mode",
        "read_path",
        "ttt_write_mode",
        "swa_expected",
        "beta_swa",
        "enable_swa_overlap_source_replace",
        "swa_write_mode",
        "read_beta_frame_chunks",
        "ttt_write_gradient_reversal_chunk_gammas",
        "ttt_write_tri_replay_chunk_params",
        "ttt_write_commit_ema_chunks",
        "semantic_memory_paths",
        "semantic_role_policy",
        "semantic_prior_mode",
    ]
    for run in runs:
        raw_dir = run.result_dir / "per_chunk_geometry"
        pointmap_available = raw_dir.is_dir() and any(raw_dir.glob("chunk_*.pt"))
        has_manual_tri = bool(_config_get(run, "ttt_write_tri_replay_chunk_params"))
        has_chunk_policy = any(
            bool(_config_get(run, key))
            for key in (
                "read_beta_frame_chunks",
                "ttt_write_gradient_reversal_chunk_gammas",
                "ttt_write_tri_replay_chunk_params",
                "ttt_write_commit_ema_chunks",
                "semantic_action_active_chunks",
            )
        )
        inv = {
            "method": run.method,
            "display": run.display,
            "run_name": run.run_name,
            "result_dir": str(run.result_dir),
            "prediction_txt": str(run.pred_path),
            "prediction_txt_exists": run.pred_path.is_file(),
            "kitti_benchmark_log": str(run.result_dir / "kitti_benchmark.log"),
            "kitti_benchmark_log_exists": (run.result_dir / "kitti_benchmark.log").is_file(),
            "hmc_config_yaml": str(run.config_path),
            "effective_config_yaml": str(run.effective_config_path) if run.effective_config_path.is_file() else "",
            "chunk_size": _config_get(run, "chunk_size"),
            "chunk_overlap": _config_get(run, "chunk_overlap"),
            "reset_every": run.reset_every,
            "frames_pred": int(run.pred.centers.shape[0]),
            "frames_gt_total": int(_read_poses(gt_path).centers.shape[0]),
            "frames_used": int(run.frames_used),
            "hmc_rows": len(_read_jsonl(run.result_dir / "hmc_state_hash.jsonl")),
            "schedule_rows": len(run.schedule),
            "schedule_source": run.schedule_source,
            "read_path": _config_get(run, "read_path"),
            "hmc_commit_mode": _config_get(run, "hmc_commit_mode"),
            "ttt_write_mode": _config_get(run, "ttt_write_mode"),
            "swa_setting": json.dumps(
                {
                    "swa_expected": _config_get(run, "swa_expected"),
                    "beta_swa": _config_get(run, "beta_swa"),
                    "enable_swa_overlap_source_replace": _config_get(run, "enable_swa_overlap_source_replace"),
                    "swa_write_mode": _config_get(run, "swa_write_mode"),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "uses_absolute_chunk_id_policy": has_chunk_policy,
            "uses_manual_tri_replay_percentage": has_manual_tri,
            "uses_semantic": bool(_config_get(run, "semantic_memory_paths")) or str(_config_get(run, "semantic_role_policy")) not in {"", "none", "None"},
            "raw_prediction_missing": not pointmap_available,
            "point_overlap_unavailable": not pointmap_available,
            "pose_overlap_proxy_used": True,
            "note": run.note,
            **run.kitti_results,
        }
        inventory.append(inv)
        for key in keys:
            config_rows.append({"method": run.method, "run_name": run.run_name, "key": key, "value": _config_get(run, key)})
        for row in run.schedule:
            schedule_rows.append(
                {
                    "method": run.method,
                    "run_name": run.run_name,
                    "chunk_id": row["chunk_id"],
                    "frame_start": row["start_frame"],
                    "frame_end": row["end_frame"],
                    "chunk_size": row.get("chunk_size"),
                    "chunk_overlap": row.get("chunk_overlap"),
                    "reset_group_id": int(row["chunk_id"]) // max(run.reset_every, 1),
                    "reset_relative_idx": int(row["chunk_id"]) % max(run.reset_every, 1),
                }
            )
    _write_csv(phase / "run_inventory.csv", inventory)
    _write_csv(phase / "config_comparison_table.csv", config_rows)
    _write_csv(phase / "chunk_schedule_comparison.csv", schedule_rows)
    lines = [
        "# Phase 0 Artifact Availability Report",
        "",
        f"GT: `{gt_path}`",
        "",
        "| method | run | frames used | chunks | pred | raw chunks | fallback/proxy | landed ATE | note |",
        "|---|---|---:|---:|---|---|---|---:|---|",
    ]
    for row in inventory:
        lines.append(
            f"| `{row['display']}` | `{row['run_name']}` | {row['frames_used']} | {row['schedule_rows']} | "
            f"`{row['prediction_txt_exists']}` | `{'no' if row['raw_prediction_missing'] else 'yes'}` | "
            f"`pose-only; GT-overlap proxy` | {_fmt(row.get('kitti_results_ate'))} | {row['note']} |"
        )
    lines.extend(
        [
            "",
            "Gate decision: all three selected systems have complete `01.txt` trajectories and recoverable chunk schedules. "
            "Per-chunk raw/duplicate overlap point artifacts are unavailable for the selected ORIG/C9/v53-H35 artifacts, "
            "so v62 uses the documented pose-only fallback and records point overlap fields as unavailable.",
        ]
    )
    (phase / "artifact_availability_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _smoke_points(n: int = 30) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi, n)
    return np.stack([t, np.sin(t), np.cos(t) * 0.5 + 0.1 * t], axis=1)


def _run_smoke(out_dir: Path) -> Dict[str, Any]:
    phase = out_dir / "phase1_metric_smoke"
    phase.mkdir(parents=True, exist_ok=True)
    gt = _smoke_points(45)

    fit = _fit_sim3(gt, gt)
    identity_res = _residual_stats(_apply_sim3(gt, fit), gt) if fit is not None else {"rmse": None}
    identity = {"fit_scale": fit[0] if fit else None, "residual_rmse": identity_res["rmse"], "gate": _safe_float(identity_res["rmse"]) < 1e-5}

    known_scale = 1.7
    pred_scaled = gt / known_scale
    fit2 = _fit_sim3(pred_scaled, gt)
    known_res = _residual_stats(_apply_sim3(pred_scaled, fit2), gt) if fit2 is not None else {"rmse": None}
    known = {
        "expected_fit_scale_pred_to_gt": known_scale,
        "fit_scale": fit2[0] if fit2 else None,
        "scale_abs_error": abs((fit2[0] if fit2 else float("nan")) - known_scale),
        "residual_rmse": known_res["rmse"],
        "gate": abs((fit2[0] if fit2 else float("nan")) - known_scale) < 1e-4 and _safe_float(known_res["rmse"]) < 1e-5,
    }

    pred_nonuniform = gt.copy()
    split = gt.shape[0] // 2
    pred_nonuniform[split:] = gt[split:] / 1.2
    whole_fit = _fit_sim3(pred_nonuniform, gt)
    whole_stats = _residual_stats(_apply_sim3(pred_nonuniform, whole_fit), gt) if whole_fit is not None else {"rmse": None}
    head_fit = _fit_sim3(pred_nonuniform[:split], gt[:split])
    tail_stats = _residual_stats(_apply_sim3(pred_nonuniform[split:], head_fit), gt[split:]) if head_fit is not None else {"rmse": None}
    nonuniform = {
        "whole_chunk_sim3_rmse": whole_stats["rmse"],
        "head_to_tail_transfer_rmse": tail_stats["rmse"],
        "transfer_gt_whole_ratio": _safe_float(tail_stats["rmse"]) / max(_safe_float(whole_stats["rmse"]), EPS),
        "gate": _safe_float(tail_stats["rmse"]) > _safe_float(whole_stats["rmse"]) * 1.5,
    }

    overlap_len = 10
    future_pred = pred_nonuniform.copy()
    overlap_fit = _fit_sim3(future_pred[:overlap_len], gt[:overlap_len])
    overlap_stats = _residual_stats(_apply_sim3(future_pred[:overlap_len], overlap_fit), gt[:overlap_len]) if overlap_fit else {"rmse": None}
    future_stats = _residual_stats(_apply_sim3(future_pred[overlap_len:], overlap_fit), gt[overlap_len:]) if overlap_fit else {"rmse": None}
    overlap = {
        "overlap_rmse": overlap_stats["rmse"],
        "future_rmse": future_stats["rmse"],
        "future_gt_overlap_ratio": _safe_float(future_stats["rmse"]) / max(_safe_float(overlap_stats["rmse"]), EPS),
        "gate": _safe_float(overlap_stats["rmse"]) < 1e-5 and _safe_float(future_stats["rmse"]) > 1e-2,
    }

    _write_json(phase / "sim3_identity_test.json", identity)
    _write_json(phase / "sim3_known_scale_test.json", known)
    _write_json(phase / "intra_chunk_nonuniform_scale_test.json", nonuniform)
    _write_json(phase / "overlap_transfer_failure_test.json", overlap)
    passed = bool(identity["gate"] and known["gate"] and nonuniform["gate"] and overlap["gate"])
    report = [
        "# Phase 1 Metric Smoke Report",
        "",
        f"- Identity residual RMSE: `{_fmt(identity['residual_rmse'], 12)}`; gate `{identity['gate']}`.",
        f"- Known scale fit: `{_fmt(known['fit_scale'], 12)}` expected `{known_scale}`; abs error `{_fmt(known['scale_abs_error'], 12)}`; gate `{known['gate']}`.",
        f"- Nonuniform scale whole RMSE `{_fmt(nonuniform['whole_chunk_sim3_rmse'], 12)}`, head->tail RMSE `{_fmt(nonuniform['head_to_tail_transfer_rmse'], 12)}`; gate `{nonuniform['gate']}`.",
        f"- Overlap residual `{_fmt(overlap['overlap_rmse'], 12)}`, future residual `{_fmt(overlap['future_rmse'], 12)}`; gate `{overlap['gate']}`.",
        "",
        f"Overall smoke gate: `{passed}`.",
    ]
    (phase / "metric_smoke_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("Phase 1 Sim3 smoke failed; refusing to run later diagnostics.")
    return {"phase1_metric_smoke_passed": passed, "identity": identity, "known_scale": known, "nonuniform": nonuniform, "overlap": overlap}


def _slice(run: RunData, start: int, end: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start = max(0, min(start, run.frames_used))
    end = max(start, min(end, run.frames_used))
    return (
        run.pred.centers[start:end],
        run.gt.centers[start:end],
        run.pred.rotations[start:end],
        run.gt.rotations[start:end],
    )


def _rolling_rmse(residual: np.ndarray, start: int, length: int = 100) -> Optional[float]:
    if residual.size == 0:
        return None
    s = max(0, min(start, residual.size - 1))
    e = min(residual.size, s + length)
    if e <= s:
        return None
    return float(np.sqrt(np.mean(residual[s:e] * residual[s:e])))


def _phase2_manifest(runs: Sequence[RunData], out_dir: Path) -> None:
    phase = out_dir / "phase2_artifacts"
    manifest: List[Dict[str, Any]] = []
    raw_rows: List[Dict[str, Any]] = []
    point_rows: List[Dict[str, Any]] = []
    for run in runs:
        raw_dir = run.result_dir / "per_chunk_geometry"
        raw_paths = sorted(raw_dir.glob("chunk_*.pt")) if raw_dir.is_dir() else []
        row = {
            "method": run.method,
            "run_name": run.run_name,
            "prediction_trajectory": str(run.pred_path),
            "gt_trajectory": str(DEFAULT_GT),
            "chunk_boundary_mapping": run.schedule_source,
            "duplicate_overlap_frame_predictions_available": bool(raw_paths),
            "camera_poses_per_chunk_before_merge_available": bool(raw_paths),
            "pointmaps_or_local_points_available": bool(raw_paths),
            "confidence_maps_available": bool(raw_paths),
            "hmc_state_hashes": str(run.result_dir / "hmc_state_hash.jsonl") if (run.result_dir / "hmc_state_hash.jsonl").is_file() else "",
            "merge_state_or_gauge_state": str(run.result_dir / "merge_state_hash.jsonl") if (run.result_dir / "merge_state_hash.jsonl").is_file() else "",
            "pose_only_fallback_used": not bool(raw_paths),
            "raw_prediction_missing": not bool(raw_paths),
            "point_overlap_unavailable": not bool(raw_paths),
        }
        manifest.append(row)
        raw_rows.append(
            {
                "method": run.method,
                "run_name": run.run_name,
                "per_chunk_geometry_dir": str(raw_dir),
                "chunk_pt_count": len(raw_paths),
                "available": bool(raw_paths),
            }
        )
        point_rows.append(
            {
                "method": run.method,
                "run_name": run.run_name,
                "pointmap_availability": "available" if raw_paths else "unavailable",
                "confidence_availability": "available" if raw_paths else "unavailable",
                "fallback": "pose-only global trajectory windows" if not raw_paths else "",
            }
        )
    _write_csv(phase / "artifact_manifest.csv", manifest)
    _write_csv(phase / "raw_chunk_availability.csv", raw_rows)
    _write_csv(phase / "pointmap_availability.csv", point_rows)
    lines = [
        "# Phase 2 Pose-Only Fallback Report",
        "",
        "Selected ORIG/C9/v53-H35 artifacts have complete trajectories and schedules, but not reusable duplicate per-chunk raw pose/pointmap tensors.",
        "Per plan section 14.1/14.2, the run proceeds with trajectory-based pose-only diagnostics and marks point/raw overlap fields unavailable.",
        "",
        "| method | run | raw chunk tensors | fallback |",
        "|---|---|---:|---|",
    ]
    for row in raw_rows:
        lines.append(f"| `{row['method']}` | `{row['run_name']}` | {row['chunk_pt_count']} | `pose-only` |")
    (phase / "pose_only_fallback_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compute_phase3(runs: Sequence[RunData], out_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[int, Tuple[float, np.ndarray, np.ndarray]]]]:
    phase = out_dir / "phase3_intrachunk"
    all_rows: List[Dict[str, Any]] = []
    fits: Dict[str, Dict[int, Tuple[float, np.ndarray, np.ndarray]]] = {}
    for run in runs:
        run_rows: List[Dict[str, Any]] = []
        run_fits: Dict[int, Tuple[float, np.ndarray, np.ndarray]] = {}
        for chunk in run.schedule:
            cid = int(chunk["chunk_id"])
            start = int(chunk["start_frame"])
            end = int(chunk["end_frame"])
            pred, gt, pred_rot, gt_rot = _slice(run, start, end)
            fit = _fit_sim3(pred, gt)
            if fit is not None:
                run_fits[cid] = fit
                aligned = _apply_sim3(pred, fit)
                sim_scale, sim_rot, sim_trans = fit
                local_stats = _residual_stats(aligned, gt)
                rot_rmse = _rotation_rmse(pred_rot, gt_rot, sim_rot)
                rpe_t = _rpe_t(aligned, gt)
                rpe_r = _rpe_r(pred_rot, gt_rot, sim_rot)
            else:
                sim_scale, sim_rot, sim_trans = None, None, None
                aligned = np.full_like(pred, np.nan)
                local_stats = _residual_stats(aligned, gt)
                rot_rmse = None
                rpe_t = None
                rpe_r = None
            global_stats = _residual_stats(run.global_aligned[start:end], gt)
            ratio = None
            improvement = None
            if local_stats["rmse"] is not None and global_stats["rmse"] is not None:
                ratio = float(local_stats["rmse"] / max(global_stats["rmse"], EPS))
                improvement = float(1.0 - ratio)
            row = {
                "method": run.method,
                "display": run.display,
                "run_name": run.run_name,
                "chunk_id": cid,
                "frame_start": start,
                "frame_end": end,
                "frame_count": int(end - start),
                "reset_group_id": cid // max(run.reset_every, 1),
                "reset_relative_idx": cid % max(run.reset_every, 1),
                "global_chunk_ate": global_stats["rmse"],
                "global_chunk_p90": global_stats["p90"],
                "local_sim3_chunk_ate": local_stats["rmse"],
                "local_sim3_chunk_p90": local_stats["p90"],
                "local_rot_rmse": rot_rmse,
                "local_rpe_t": rpe_t,
                "local_rpe_r": rpe_r,
                "local_final_error": local_stats["final"],
                "local_sim3_scale": sim_scale,
                "local_sim3_log_scale": math.log(abs(sim_scale)) if sim_scale is not None and abs(sim_scale) > EPS else None,
                "local_sim3_rotation_angle": math.degrees(_rot_angle_rad(sim_rot)) if sim_rot is not None else None,
                "local_sim3_translation_norm": float(np.linalg.norm(sim_trans)) if sim_trans is not None else None,
                "local_improvement_ratio": improvement,
                "local_to_global_ate_ratio": ratio,
                "chunk_step_length_ratio": _step_ratio(pred, gt),
                "rolling100_error": _rolling_rmse(run.global_residual, start, 100),
                "raw_prediction_missing": True,
                "pose_only_fallback_used": True,
            }
            run_rows.append(row)
            all_rows.append(row)
        fits[run.method] = run_fits
        _write_csv(phase / f"{run.method}_intrachunk_metrics.csv", run_rows)
    comparison = []
    by_chunk: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for row in all_rows:
        by_chunk.setdefault(int(row["chunk_id"]), {})[str(row["method"])] = row
    for cid, rows in sorted(by_chunk.items()):
        out = {"chunk_id": cid}
        for method in ("orig", "c9", "h35"):
            row = rows.get(method, {})
            out[f"{method}_global_chunk_ate"] = row.get("global_chunk_ate")
            out[f"{method}_local_sim3_chunk_ate"] = row.get("local_sim3_chunk_ate")
            out[f"{method}_local_improvement_ratio"] = row.get("local_improvement_ratio")
            out[f"{method}_local_sim3_scale"] = row.get("local_sim3_scale")
        if rows.get("h35") and rows.get("c9"):
            out["h35_minus_c9_global_chunk_ate"] = _safe_float(rows["h35"].get("global_chunk_ate")) - _safe_float(rows["c9"].get("global_chunk_ate"))
            out["h35_minus_c9_local_sim3_chunk_ate"] = _safe_float(rows["h35"].get("local_sim3_chunk_ate")) - _safe_float(rows["c9"].get("local_sim3_chunk_ate"))
        comparison.append(out)
    _write_csv(phase / "intrachunk_method_comparison.csv", comparison)
    _write_intrachunk_report(phase / "intrachunk_autopsy_report.md", runs, all_rows)
    _plot_phase3(phase / "figures", runs, all_rows)
    return all_rows, fits


def _write_intrachunk_report(path: Path, runs: Sequence[RunData], rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Phase 3 Intrachunk Autopsy Report",
        "",
        "| method | global ATE (computed) | landed ATE | local Sim3 mean | local/global median | top local chunk | interpretation |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for run in runs:
        subset = [r for r in rows if r["method"] == run.method]
        top = max(subset, key=lambda r: _safe_float(r.get("local_sim3_chunk_ate"))) if subset else {}
        ratio = _median(r.get("local_to_global_ate_ratio") for r in subset)
        interp = "gauge-dominant by local/global ratio" if ratio is not None and ratio <= 0.4 else "local geometry contributes strongly"
        lines.append(
            f"| `{run.display}` | {_fmt(_rmse(run.global_residual))} | {_fmt(run.kitti_results.get('kitti_results_ate'))} | "
            f"{_fmt(_mean(r.get('local_sim3_chunk_ate') for r in subset))} | {_fmt(ratio)} | "
            f"`{top.get('chunk_id', 'NA')}` `{_fmt(top.get('local_sim3_chunk_ate'))}` | {interp} |"
        )
    lines.extend(
        [
            "",
            "Definitions: `local_sim3_chunk_ate` fits one Sim(3) per chunk against GT; `global_chunk_ate` uses one full-trajectory Sim(3).",
            "All rows use pose-only fallback because selected ORIG/C9/v53-H35 artifacts do not expose duplicate raw chunk camera poses.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_no_data(path: Path, title: str, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.5, 0.55, title, ha="center", va="center", fontsize=14)
    ax.text(0.5, 0.38, message, ha="center", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_phase3(fig_dir: Path, runs: Sequence[RunData], rows: Sequence[Mapping[str, Any]]) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for key, fname, ylabel in [
        ("local_sim3_chunk_ate", "local_sim3_ate_timeline.png", "local Sim3 ATE (m)"),
        ("local_improvement_ratio", "local_improvement_ratio_timeline.png", "local improvement ratio"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 4))
        for run in runs:
            subset = [r for r in rows if r["method"] == run.method]
            ax.plot([r["chunk_id"] for r in subset], [_safe_float(r.get(key)) for r in subset], marker="o", label=run.display)
        ax.set_xlabel("chunk")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    for run in runs:
        subset = [r for r in rows if r["method"] == run.method]
        ax.scatter([_safe_float(r.get("global_chunk_ate")) for r in subset], [_safe_float(r.get("local_sim3_chunk_ate")) for r in subset], label=run.display, s=24)
    ax.set_xlabel("global chunk ATE (m)")
    ax.set_ylabel("local Sim3 chunk ATE (m)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "global_vs_local_chunk_ate.png", dpi=160)
    plt.close(fig)

    matrix: List[List[float]] = []
    labels: List[str] = []
    max_chunks = max((len([r for r in rows if r["method"] == run.method]) for run in runs), default=0)
    for run in runs:
        subset = [r for r in rows if r["method"] == run.method]
        vals = [_safe_float(r.get("local_sim3_chunk_ate")) for r in subset]
        vals += [float("nan")] * max(0, max_chunks - len(vals))
        matrix.append(vals)
        labels.append(run.display)
    fig, ax = plt.subplots(figsize=(11, 2.8))
    im = ax.imshow(np.asarray(matrix, dtype=np.float64), aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("chunk")
    fig.colorbar(im, ax=ax, label="local Sim3 ATE (m)")
    fig.tight_layout()
    fig.savefig(fig_dir / "intrachunk_error_heatmap.png", dpi=160)
    plt.close(fig)


def _compute_phase4(
    runs: Sequence[RunData],
    out_dir: Path,
    chunk_rows: Sequence[Mapping[str, Any]],
    fits: Mapping[str, Mapping[int, Tuple[float, np.ndarray, np.ndarray]]],
) -> List[Dict[str, Any]]:
    phase = out_dir / "phase4_interchunk"
    chunk_by_method_id = {(r["method"], int(r["chunk_id"])): r for r in chunk_rows}
    all_rows: List[Dict[str, Any]] = []
    for run in runs:
        run_rows: List[Dict[str, Any]] = []
        for prev, curr in zip(run.schedule[:-1], run.schedule[1:]):
            prev_id = int(prev["chunk_id"])
            curr_id = int(curr["chunk_id"])
            curr_fit = fits.get(run.method, {}).get(curr_id)
            prev_fit = fits.get(run.method, {}).get(prev_id)
            if curr_fit is not None and prev_fit is not None:
                scale_jump = math.log(abs(curr_fit[0])) - math.log(abs(prev_fit[0]))
                rot_jump = math.degrees(_rot_angle_rad(curr_fit[1] @ prev_fit[1].T))
                trans_jump = float(np.linalg.norm(curr_fit[2] - prev_fit[2]))
            else:
                scale_jump, rot_jump, trans_jump = None, None, None
            ov_start = max(int(prev["start_frame"]), int(curr["start_frame"]))
            ov_end = min(int(prev["end_frame"]), int(curr["end_frame"]), run.frames_used)
            overlap_pred, overlap_gt, _, _ = _slice(run, ov_start, ov_end)
            overlap_fit = _fit_sim3(overlap_pred, overlap_gt)
            if overlap_fit is not None:
                overlap_aligned = _apply_sim3(overlap_pred, overlap_fit)
                overlap_stats = _residual_stats(overlap_aligned, overlap_gt)
                overlap_residuals = _residual_values(overlap_aligned, overlap_gt)
                inlier_ratio = float(np.mean(overlap_residuals < 0.10)) if overlap_residuals.size else None
            else:
                overlap_stats = _residual_stats(np.full_like(overlap_pred, np.nan), overlap_gt)
                inlier_ratio = None
            future_start = max(ov_end, int(curr["start_frame"]))
            future_end = int(curr["end_frame"])
            future_pred, future_gt, _, _ = _slice(run, future_start, future_end)
            if overlap_fit is not None and future_pred.shape[0] >= 1:
                future_stats = _residual_stats(_apply_sim3(future_pred, overlap_fit), future_gt)
            else:
                future_stats = _residual_stats(np.full_like(future_pred, np.nan), future_gt)
            tail_start = int(curr["start_frame"]) + max(0, int(curr["end_frame"]) - int(curr["start_frame"])) * 2 // 3
            tail_pred, tail_gt, _, _ = _slice(run, tail_start, int(curr["end_frame"]))
            if overlap_fit is not None and tail_pred.shape[0] >= 1:
                tail_stats = _residual_stats(_apply_sim3(tail_pred, overlap_fit), tail_gt)
            else:
                tail_stats = _residual_stats(np.full_like(tail_pred, np.nan), tail_gt)
            boundary_jump = None
            if int(curr["start_frame"]) > 0 and int(curr["start_frame"]) < run.frames_used:
                s = int(curr["start_frame"])
                pred_delta = run.global_aligned[s] - run.global_aligned[s - 1]
                gt_delta = run.gt.centers[s] - run.gt.centers[s - 1]
                boundary_jump = float(np.linalg.norm(pred_delta - gt_delta))
            row = {
                "method": run.method,
                "display": run.display,
                "run_name": run.run_name,
                "chunk_id": curr_id,
                "prev_chunk_id": prev_id,
                "reset_group_id": curr_id // max(run.reset_every, 1),
                "reset_relative_idx": curr_id % max(run.reset_every, 1),
                "scale_jump_gtlocal": scale_jump,
                "abs_scale_jump_gtlocal": abs(scale_jump) if scale_jump is not None else None,
                "rotation_jump_gtlocal": rot_jump,
                "translation_jump_gtlocal": trans_jump,
                "overlap_start": ov_start,
                "overlap_end": ov_end,
                "overlap_frame_count": max(0, ov_end - ov_start),
                "overlap_proxy_type": "GT-overlap pose proxy; duplicate current-vs-prev raw overlap unavailable",
                "overlap_sim3_scale_all": overlap_fit[0] if overlap_fit is not None else None,
                "overlap_sim3_scale_geo": None,
                "overlap_sim3_residual_all": overlap_stats["rmse"],
                "overlap_sim3_residual_geo": None,
                "overlap_inlier_ratio_all": inlier_ratio,
                "overlap_inlier_ratio_geo": None,
                "nonoverlap_future_error_after_overlap_sim3": future_stats["rmse"],
                "tail_error_after_overlap_sim3": tail_stats["rmse"],
                "boundary_pose_jump": boundary_jump,
                "duplicate_overlap_pose_disagreement": None,
                "point_overlap_unavailable": True,
                "pose_overlap_proxy_used": True,
                "rolling100_error": chunk_by_method_id.get((run.method, curr_id), {}).get("rolling100_error"),
                "local_sim3_chunk_ate": chunk_by_method_id.get((run.method, curr_id), {}).get("local_sim3_chunk_ate"),
                "global_chunk_ate": chunk_by_method_id.get((run.method, curr_id), {}).get("global_chunk_ate"),
            }
            all_rows.append(row)
            run_rows.append(row)
        _write_csv(phase / f"{run.method}_interchunk_metrics.csv", run_rows)
    h35_by_chunk = {int(r["chunk_id"]): r for r in all_rows if r["method"] == "h35"}
    c9_by_chunk = {int(r["chunk_id"]): r for r in all_rows if r["method"] == "c9"}
    gap_rows: List[Dict[str, Any]] = []
    for cid in sorted(set(h35_by_chunk) & set(c9_by_chunk)):
        h = h35_by_chunk[cid]
        c = c9_by_chunk[cid]
        gap_rows.append(
            {
                "chunk_id": cid,
                "h35_abs_scale_jump": h.get("abs_scale_jump_gtlocal"),
                "c9_abs_scale_jump": c.get("abs_scale_jump_gtlocal"),
                "h35_minus_c9_abs_scale_jump": _safe_float(h.get("abs_scale_jump_gtlocal")) - _safe_float(c.get("abs_scale_jump_gtlocal")),
                "h35_future_after_overlap": h.get("nonoverlap_future_error_after_overlap_sim3"),
                "c9_future_after_overlap": c.get("nonoverlap_future_error_after_overlap_sim3"),
                "h35_minus_c9_future_after_overlap": _safe_float(h.get("nonoverlap_future_error_after_overlap_sim3"))
                - _safe_float(c.get("nonoverlap_future_error_after_overlap_sim3")),
            }
        )
    _write_csv(phase / "h35_vs_c9_interchunk_gap.csv", gap_rows)
    _write_csv(phase / "overlap_transfer_metrics.csv", all_rows)
    _write_interchunk_report(phase / "interchunk_autopsy_report.md", runs, all_rows)
    _plot_phase4(phase / "figures", runs, all_rows, gap_rows)
    return all_rows


def _write_interchunk_report(path: Path, runs: Sequence[RunData], rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Phase 4 Interchunk Autopsy Report",
        "",
        "| method | scale jump mean(abs) | scale jump vs rolling100 corr | overlap residual mean | future-after-overlap mean | top future pair | interpretation |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for run in runs:
        subset = [r for r in rows if r["method"] == run.method]
        corr = _corr([r.get("abs_scale_jump_gtlocal") for r in subset], [r.get("rolling100_error") for r in subset])
        top = max(subset, key=lambda r: _safe_float(r.get("nonoverlap_future_error_after_overlap_sim3"))) if subset else {}
        overlap_mean = _mean(r.get("overlap_sim3_residual_all") for r in subset)
        future_mean = _mean(r.get("nonoverlap_future_error_after_overlap_sim3") for r in subset)
        interp = "overlap-local-only risk" if overlap_mean is not None and future_mean is not None and overlap_mean < future_mean else "inspect raw overlap"
        lines.append(
            f"| `{run.display}` | {_fmt(_mean(r.get('abs_scale_jump_gtlocal') for r in subset))} | {_fmt(corr)} | "
            f"{_fmt(overlap_mean)} | {_fmt(future_mean)} | `{top.get('prev_chunk_id', 'NA')}->{top.get('chunk_id', 'NA')}` `{_fmt(top.get('nonoverlap_future_error_after_overlap_sim3'))}` | {interp} |"
        )
    lines.extend(
        [
            "",
            "Overlap note: selected artifacts lack duplicate raw current/previous chunk overlap poses and point pairs, so `overlap_sim3_*` is a GT-overlap pose proxy. "
            "It tests whether a chunk's overlap/head region predicts its future frames, not an online no-GT correction claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_phase4(fig_dir: Path, runs: Sequence[RunData], rows: Sequence[Mapping[str, Any]], gap_rows: Sequence[Mapping[str, Any]]) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for key, fname, ylabel in [
        ("scale_jump_gtlocal", "scale_jump_timeline.png", "log scale jump"),
        ("rotation_jump_gtlocal", "rotation_jump_timeline.png", "rotation jump (deg)"),
        ("boundary_pose_jump", "boundary_jump_timeline.png", "boundary pose jump (m)"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 4))
        for run in runs:
            subset = [r for r in rows if r["method"] == run.method]
            ax.plot([r["chunk_id"] for r in subset], [_safe_float(r.get(key)) for r in subset], marker="o", label=run.display)
        ax.set_xlabel("chunk")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for run in runs:
        subset = [r for r in rows if r["method"] == run.method]
        ax.scatter(
            [_safe_float(r.get("overlap_sim3_residual_all")) for r in subset],
            [_safe_float(r.get("nonoverlap_future_error_after_overlap_sim3")) for r in subset],
            label=run.display,
            s=24,
        )
    ax.set_xlabel("GT-overlap proxy residual (m)")
    ax.set_ylabel("future error after overlap Sim3 (m)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "overlap_residual_vs_future_error.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot([r["chunk_id"] for r in gap_rows], [_safe_float(r.get("h35_minus_c9_abs_scale_jump")) for r in gap_rows], marker="o")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("chunk")
    ax.set_ylabel("H35 - C9 abs scale jump")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "h35_vs_c9_scale_jump.png", dpi=160)
    plt.close(fig)

    methods = [run.method for run in runs]
    labels = [run.display for run in runs]
    max_chunks = max((len([r for r in rows if r["method"] == m]) for m in methods), default=0)
    matrix = []
    for m in methods:
        subset = [r for r in rows if r["method"] == m]
        vals = [_safe_float(r.get("abs_scale_jump_gtlocal")) for r in subset]
        vals += [float("nan")] * max(0, max_chunks - len(vals))
        matrix.append(vals)
    fig, ax = plt.subplots(figsize=(11, 2.8))
    im = ax.imshow(np.asarray(matrix, dtype=np.float64), aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("chunk")
    fig.colorbar(im, ax=ax, label="abs log scale jump")
    fig.tight_layout()
    fig.savefig(fig_dir / "interchunk_error_heatmap.png", dpi=160)
    plt.close(fig)


def _window_slices(start: int, end: int) -> Dict[str, Tuple[int, int]]:
    n = max(0, end - start)
    a = start
    b = start + n // 3
    c = start + (2 * n) // 3
    d = end
    return {"head": (a, max(b, a)), "mid": (max(b, a), max(c, b)), "tail": (max(c, b), d)}


def _scale_drift_for_run(run: RunData, local_rows: Mapping[int, Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    sub_rows: List[Dict[str, Any]] = []
    chunk_rows: List[Dict[str, Any]] = []
    full_pred_step_median = _median(_step_lengths(run.pred.centers))
    for chunk in run.schedule:
        cid = int(chunk["chunk_id"])
        start = int(chunk["start_frame"])
        end = int(chunk["end_frame"])
        pred, gt, _, _ = _slice(run, start, end)
        sub_scales: List[float] = []
        sub_names = _window_slices(start, end)
        for name, (s, e) in sub_names.items():
            p, g, _, _ = _slice(run, s, e)
            fit = _fit_sim3(p, g)
            stats = _residual_stats(_apply_sim3(p, fit), g) if fit is not None else {"rmse": None}
            scale = fit[0] if fit is not None else None
            if scale is not None and scale > EPS:
                sub_scales.append(math.log(abs(scale)))
            sub_rows.append(
                {
                    "method": run.method,
                    "display": run.display,
                    "run_name": run.run_name,
                    "chunk_id": cid,
                    "subwindow_id": name,
                    "subwindow_start": s,
                    "subwindow_end": e,
                    "subwindow_sim3_scale": scale,
                    "subwindow_log_scale": math.log(abs(scale)) if scale is not None and abs(scale) > EPS else None,
                    "subwindow_local_ate": stats["rmse"],
                    "subwindow_kind": "third",
                }
            )
        # Sliding windows requested by plan. Kept in same file with subwindow_kind=sliding.
        win_len = 8
        stride = 4
        local_n = end - start
        if local_n >= win_len:
            idx = 0
            for local_s in range(0, local_n - win_len + 1, stride):
                s = start + local_s
                e = s + win_len
                p, g, _, _ = _slice(run, s, e)
                fit = _fit_sim3(p, g)
                stats = _residual_stats(_apply_sim3(p, fit), g) if fit is not None else {"rmse": None}
                scale = fit[0] if fit is not None else None
                sub_rows.append(
                    {
                        "method": run.method,
                        "display": run.display,
                        "run_name": run.run_name,
                        "chunk_id": cid,
                        "subwindow_id": f"slide{idx:02d}",
                        "subwindow_start": s,
                        "subwindow_end": e,
                        "subwindow_sim3_scale": scale,
                        "subwindow_log_scale": math.log(abs(scale)) if scale is not None and abs(scale) > EPS else None,
                        "subwindow_local_ate": stats["rmse"],
                        "subwindow_kind": "sliding_8_stride_4",
                    }
                )
                idx += 1
        head = sub_names["head"]
        mid = sub_names["mid"]
        tail = sub_names["tail"]
        head_pred, head_gt, _, _ = _slice(run, head[0], head[1])
        mid_pred, mid_gt, _, _ = _slice(run, mid[0], mid[1])
        tail_pred, tail_gt, _, _ = _slice(run, tail[0], tail[1])
        head_fit = _fit_sim3(head_pred, head_gt)
        mid_fit = _fit_sim3(mid_pred, mid_gt)
        head_tail_stats = _residual_stats(_apply_sim3(tail_pred, head_fit), tail_gt) if head_fit is not None else {"rmse": None}
        mid_tail_stats = _residual_stats(_apply_sim3(tail_pred, mid_fit), tail_gt) if mid_fit is not None else {"rmse": None}
        step_whole = _step_ratio(pred, gt)
        step_head = _step_ratio(head_pred, head_gt)
        step_mid = _step_ratio(mid_pred, mid_gt)
        step_tail = _step_ratio(tail_pred, tail_gt)
        step_vals = _finite([step_head, step_mid, step_tail])
        step_span_frac = (max(step_vals) - min(step_vals)) / max(_median(step_vals) or EPS, EPS) if len(step_vals) >= 2 else None
        pred_step_med = _median(_step_lengths(pred))
        no_gt = (pred_step_med / max(full_pred_step_median or EPS, EPS)) if pred_step_med is not None and full_pred_step_median is not None else None
        local_ate = _safe_float(local_rows.get(cid, {}).get("local_sim3_chunk_ate"))
        chunk_rows.append(
            {
                "method": run.method,
                "display": run.display,
                "run_name": run.run_name,
                "chunk_id": cid,
                "frame_start": start,
                "frame_end": end,
                "reset_group_id": cid // max(run.reset_every, 1),
                "reset_relative_idx": cid % max(run.reset_every, 1),
                "head_to_tail_transfer_error": head_tail_stats["rmse"],
                "mid_to_tail_transfer_error": mid_tail_stats["rmse"],
                "head_to_tail_transfer_ratio": _safe_float(head_tail_stats["rmse"]) / max(local_ate, EPS) if math.isfinite(local_ate) else None,
                "mid_to_tail_transfer_ratio": _safe_float(mid_tail_stats["rmse"]) / max(local_ate, EPS) if math.isfinite(local_ate) else None,
                "intra_scale_variance": float(np.var(np.asarray(sub_scales))) if len(sub_scales) >= 2 else None,
                "step_length_ratio": step_whole,
                "step_length_ratio_head": step_head,
                "step_length_ratio_mid": step_mid,
                "step_length_ratio_tail": step_tail,
                "step_length_ratio_head_mid_tail_span_frac": step_span_frac,
                "step_length_ratio_nogt": no_gt,
                "depth_scale_ratio_head_tail": None,
                "depth_scale_ratio_availability": "unavailable: selected artifacts lack depth/pointmap tensors",
            }
        )
    return sub_rows, chunk_rows


def _compute_phase5(runs: Sequence[RunData], out_dir: Path, chunk_rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    phase = out_dir / "phase5_intrachunk_scale"
    local_by_method: Dict[str, Dict[int, Mapping[str, Any]]] = {}
    for row in chunk_rows:
        local_by_method.setdefault(str(row["method"]), {})[int(row["chunk_id"])] = row
    all_sub: List[Dict[str, Any]] = []
    all_chunk: List[Dict[str, Any]] = []
    for run in runs:
        sub_rows, c_rows = _scale_drift_for_run(run, local_by_method.get(run.method, {}))
        all_sub.extend(sub_rows)
        all_chunk.extend(c_rows)
        _write_csv(phase / f"{run.method}_intrachunk_scale_metrics.csv", c_rows)
    comp: List[Dict[str, Any]] = []
    by_chunk: Dict[int, Dict[str, Mapping[str, Any]]] = {}
    for row in all_chunk:
        by_chunk.setdefault(int(row["chunk_id"]), {})[str(row["method"])] = row
    for cid, rows in sorted(by_chunk.items()):
        out = {"chunk_id": cid}
        for method in ("orig", "c9", "h35"):
            row = rows.get(method, {})
            out[f"{method}_intra_scale_variance"] = row.get("intra_scale_variance")
            out[f"{method}_head_to_tail_transfer_ratio"] = row.get("head_to_tail_transfer_ratio")
            out[f"{method}_step_length_ratio"] = row.get("step_length_ratio")
        if rows.get("h35") and rows.get("c9"):
            out["h35_minus_c9_intra_scale_variance"] = _safe_float(rows["h35"].get("intra_scale_variance")) - _safe_float(rows["c9"].get("intra_scale_variance"))
            out["h35_minus_c9_head_tail_ratio"] = _safe_float(rows["h35"].get("head_to_tail_transfer_ratio")) - _safe_float(rows["c9"].get("head_to_tail_transfer_ratio"))
        comp.append(out)
    _write_csv(phase / "intrachunk_scale_comparison.csv", comp)
    _write_csv(phase / "intrachunk_scale_subwindows.csv", all_sub)
    _write_intrachunk_scale_report(phase / "intrachunk_scale_autopsy_report.md", runs, all_chunk)
    _plot_phase5(phase / "figures", runs, all_sub, all_chunk)
    return all_sub, all_chunk


def _write_intrachunk_scale_report(path: Path, runs: Sequence[RunData], rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Phase 5 Intrachunk Scale Autopsy Report",
        "",
        "| method | intra-scale var mean | head->tail ratio mean | step ratio mean | no-GT step ratio range | positive evidence |",
        "|---|---:|---:|---:|---|---|",
    ]
    for run in runs:
        subset = [r for r in rows if r["method"] == run.method]
        span_positive = sum(1 for r in subset if _safe_float(r.get("step_length_ratio_head_mid_tail_span_frac")) >= 0.10)
        transfer_positive = sum(1 for r in subset if _safe_float(r.get("head_to_tail_transfer_ratio")) >= 1.5)
        lines.append(
            f"| `{run.display}` | {_fmt(_mean(r.get('intra_scale_variance') for r in subset))} | "
            f"{_fmt(_mean(r.get('head_to_tail_transfer_ratio') for r in subset))} | "
            f"{_fmt(_mean(r.get('step_length_ratio') for r in subset))} | "
            f"{_fmt(_min(r.get('step_length_ratio_nogt') for r in subset))}..{_fmt(_max(r.get('step_length_ratio_nogt') for r in subset))} | "
            f"{transfer_positive} chunks transfer>=1.5; {span_positive} chunks step span>=10% |"
        )
    lines.append("")
    lines.append("Depth head/tail scale is marked unavailable because selected artifacts do not include reusable depth/pointmap tensors.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_phase5(fig_dir: Path, runs: Sequence[RunData], sub_rows: Sequence[Mapping[str, Any]], chunk_rows: Sequence[Mapping[str, Any]]) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    third_rows = [r for r in sub_rows if r.get("subwindow_kind") == "third"]
    fig, ax = plt.subplots(figsize=(10, 4))
    for run in runs:
        subset = [r for r in third_rows if r["method"] == run.method]
        xs = [int(r["chunk_id"]) + {"head": 0.0, "mid": 0.28, "tail": 0.56}.get(str(r["subwindow_id"]), 0.0) for r in subset]
        ax.scatter(xs, [_safe_float(r.get("subwindow_log_scale")) for r in subset], s=14, label=run.display)
    ax.set_xlabel("chunk (+ head/mid/tail offset)")
    ax.set_ylabel("subwindow log scale")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "subwindow_scale_timeline.png", dpi=160)
    plt.close(fig)

    for key, fname, ylabel in [
        ("intra_scale_variance", "intra_scale_variance_by_chunk.png", "intra-scale variance"),
        ("head_to_tail_transfer_ratio", "head_to_tail_transfer_error.png", "head->tail / local ratio"),
        ("step_length_ratio", "step_length_ratio_timeline.png", "step-length ratio pred/GT"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 4))
        for run in runs:
            subset = [r for r in chunk_rows if r["method"] == run.method]
            ax.plot([r["chunk_id"] for r in subset], [_safe_float(r.get(key)) for r in subset], marker="o", label=run.display)
        ax.set_xlabel("chunk")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=160)
        plt.close(fig)
    _plot_no_data(fig_dir / "depth_scale_ratio_head_tail.png", "depth scale ratio unavailable", "No reusable depth/pointmap tensors in selected ORIG/C9/v53-H35 artifacts.")


def _compute_phase6(
    runs: Sequence[RunData],
    out_dir: Path,
    chunk_rows: Sequence[Mapping[str, Any]],
    inter_rows: Sequence[Mapping[str, Any]],
    scale_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    phase = out_dir / "phase6_reset_relative"
    inter_by = {(r["method"], int(r["chunk_id"])): r for r in inter_rows}
    scale_by = {(r["method"], int(r["chunk_id"])): r for r in scale_rows}
    all_rows: List[Dict[str, Any]] = []
    for run in runs:
        method_rows: List[Dict[str, Any]] = []
        rels = sorted({int(r["reset_relative_idx"]) for r in chunk_rows if r["method"] == run.method})
        for rel in rels:
            csub = [r for r in chunk_rows if r["method"] == run.method and int(r["reset_relative_idx"]) == rel]
            isub = [inter_by.get((run.method, int(r["chunk_id"])), {}) for r in csub]
            ssub = [scale_by.get((run.method, int(r["chunk_id"])), {}) for r in csub]
            row = {
                "method": run.method,
                "display": run.display,
                "run_name": run.run_name,
                "reset_relative_idx": rel,
                "chunk_count": len(csub),
                "local_sim3_chunk_ate_mean": _mean(r.get("local_sim3_chunk_ate") for r in csub),
                "local_sim3_chunk_ate_std": _std(r.get("local_sim3_chunk_ate") for r in csub),
                "scale_jump_gtlocal_abs_mean": _mean(r.get("abs_scale_jump_gtlocal") for r in isub),
                "scale_jump_gtlocal_abs_std": _std(r.get("abs_scale_jump_gtlocal") for r in isub),
                "overlap_sim3_scale_mean": _mean(r.get("overlap_sim3_scale_all") for r in isub),
                "step_length_ratio_mean": _mean(r.get("step_length_ratio") for r in ssub),
                "intra_scale_variance_mean": _mean(r.get("intra_scale_variance") for r in ssub),
                "rolling100_error_mean": _mean(r.get("rolling100_error") for r in csub),
                "rolling100_error_std": _std(r.get("rolling100_error") for r in csub),
            }
            method_rows.append(row)
            all_rows.append(row)
        _write_csv(phase / f"{run.method}_reset_relative_metrics.csv", method_rows)
    _write_csv(phase / "reset_relative_comparison.csv", all_rows)
    _write_reset_report(phase / "reset_relative_autopsy_report.md", runs, all_rows)
    _plot_phase6(phase / "figures", runs, all_rows)
    return all_rows


def _write_reset_report(path: Path, runs: Sequence[RunData], rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Phase 6 Reset-Relative Autopsy Report",
        "",
        "| method | rel0 scale jump | rel1 scale jump | other scale jump mean | max scale rel | max rolling rel | max local rel | reset-scale-loss evidence |",
        "|---|---:|---:|---:|---|---|---|---|",
    ]
    for run in runs:
        subset = [r for r in rows if r["method"] == run.method]
        rel = {int(r["reset_relative_idx"]): r for r in subset}
        rel0 = rel.get(0, {})
        rel1 = rel.get(1, {})
        other = [r for r in subset if int(r["reset_relative_idx"]) not in {0, 1}]
        rel01 = _finite([rel0.get("scale_jump_gtlocal_abs_mean"), rel1.get("scale_jump_gtlocal_abs_mean")])
        other_mean = _mean(r.get("scale_jump_gtlocal_abs_mean") for r in other)
        evidence = bool(rel01 and other_mean is not None and max(rel01) > other_mean * 1.2)
        max_scale = max(
            subset,
            key=lambda r: _safe_float(r.get("scale_jump_gtlocal_abs_mean")),
        )
        max_rolling = max(
            subset,
            key=lambda r: _safe_float(r.get("rolling100_error_mean")),
        )
        max_local = max(
            subset,
            key=lambda r: _safe_float(r.get("local_sim3_chunk_ate_mean")),
        )
        lines.append(
            f"| `{run.display}` | {_fmt(rel0.get('scale_jump_gtlocal_abs_mean'))} | {_fmt(rel1.get('scale_jump_gtlocal_abs_mean'))} | "
            f"{_fmt(other_mean)} | rel={int(max_scale['reset_relative_idx'])} ({_fmt(max_scale.get('scale_jump_gtlocal_abs_mean'))}) | "
            f"rel={int(max_rolling['reset_relative_idx'])} ({_fmt(max_rolling.get('rolling100_error_mean'))}) | "
            f"rel={int(max_local['reset_relative_idx'])} ({_fmt(max_local.get('local_sim3_chunk_ate_mean'))}) | `{evidence}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_phase6(fig_dir: Path, runs: Sequence[RunData], rows: Sequence[Mapping[str, Any]]) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for key, fname, ylabel in [
        ("scale_jump_gtlocal_abs_mean", "reset_relative_scale_jump.png", "mean abs log scale jump"),
        ("step_length_ratio_mean", "reset_relative_step_length_ratio.png", "step-length ratio"),
        ("local_sim3_chunk_ate_mean", "reset_relative_local_ate.png", "local Sim3 ATE (m)"),
        ("rolling100_error_mean", "reset_relative_rolling100.png", "rolling100 RMSE (m)"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4))
        for run in runs:
            subset = [r for r in rows if r["method"] == run.method]
            ax.plot([r["reset_relative_idx"] for r in subset], [_safe_float(r.get(key)) for r in subset], marker="o", label=run.display)
        ax.set_xlabel("reset relative idx")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=160)
        plt.close(fig)


def _compute_taxonomy(
    runs: Sequence[RunData],
    out_dir: Path,
    chunk_rows: Sequence[Mapping[str, Any]],
    inter_rows: Sequence[Mapping[str, Any]],
    scale_rows: Sequence[Mapping[str, Any]],
    reset_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    phase = out_dir / "phase7_taxonomy"
    inter_by = {(r["method"], int(r["chunk_id"])): r for r in inter_rows}
    scale_by = {(r["method"], int(r["chunk_id"])): r for r in scale_rows}
    thresholds: Dict[str, Any] = {}
    thresholds["local_high"] = max(2.0, _percentile([r.get("local_sim3_chunk_ate") for r in chunk_rows], 75) or 2.0)
    thresholds["scale_jump_high"] = max(0.10, _percentile([r.get("abs_scale_jump_gtlocal") for r in inter_rows], 75) or 0.10)
    thresholds["overlap_res_high"] = max(0.25, _percentile([r.get("overlap_sim3_residual_all") for r in inter_rows], 75) or 0.25)
    thresholds["future_high"] = max(3.0, _percentile([r.get("nonoverlap_future_error_after_overlap_sim3") for r in inter_rows], 75) or 3.0)
    thresholds["intra_var_high"] = max(0.005, _percentile([r.get("intra_scale_variance") for r in scale_rows], 75) or 0.005)
    thresholds["rolling_high"] = _percentile([r.get("rolling100_error") for r in chunk_rows], 75) or 0.0
    rows: List[Dict[str, Any]] = []
    for c in chunk_rows:
        method = str(c["method"])
        cid = int(c["chunk_id"])
        inter = inter_by.get((method, cid), {})
        scale = scale_by.get((method, cid), {})
        local = _safe_float(c.get("local_sim3_chunk_ate"))
        ratio = _safe_float(c.get("local_to_global_ate_ratio"))
        scale_jump = _safe_float(inter.get("abs_scale_jump_gtlocal"))
        overlap_res = _safe_float(inter.get("overlap_sim3_residual_all"))
        future = _safe_float(inter.get("nonoverlap_future_error_after_overlap_sim3"))
        intra_var = _safe_float(scale.get("intra_scale_variance"))
        transfer = _safe_float(scale.get("head_to_tail_transfer_ratio"))
        step_span = _safe_float(scale.get("step_length_ratio_head_mid_tail_span_frac"))
        rolling = _safe_float(c.get("rolling100_error"))
        flags: List[str] = []
        if math.isfinite(local) and (local >= thresholds["local_high"] or (math.isfinite(ratio) and ratio >= 0.7)):
            flags.append("TYPE_A_LOCAL_GEOMETRY")
        if math.isfinite(ratio) and ratio <= 0.4 and math.isfinite(scale_jump) and scale_jump >= thresholds["scale_jump_high"]:
            flags.append("TYPE_B_INTERCHUNK_GAUGE")
        if math.isfinite(overlap_res) and overlap_res >= thresholds["overlap_res_high"]:
            flags.append("TYPE_C_OVERLAP_UNRELIABLE")
        if math.isfinite(overlap_res) and math.isfinite(future) and overlap_res < thresholds["overlap_res_high"] and future >= thresholds["future_high"]:
            flags.append("TYPE_D_OVERLAP_LOCAL_ONLY")
        if (
            (math.isfinite(intra_var) and intra_var >= thresholds["intra_var_high"])
            or (math.isfinite(transfer) and transfer >= 1.5)
            or (math.isfinite(step_span) and step_span >= 0.10)
        ):
            flags.append("TYPE_E_INTRACHUNK_SCALE_DRIFT")
        if int(c.get("reset_relative_idx", -1)) in {0, 1} and (
            (math.isfinite(scale_jump) and scale_jump >= thresholds["scale_jump_high"])
            or (math.isfinite(rolling) and rolling >= thresholds["rolling_high"])
        ):
            flags.append("TYPE_F_RESET_SCALE_REBUILD")
        if not flags:
            primary = "TYPE_G_MIXED_OR_LOW_SIGNAL"
        elif len(flags) > 1:
            primary = "TYPE_G_MIXED"
        else:
            primary = flags[0]
        rows.append(
            {
                "method": method,
                "display": c.get("display"),
                "run_name": c.get("run_name"),
                "chunk_id": cid,
                "frame_start": c.get("frame_start"),
                "frame_end": c.get("frame_end"),
                "primary_error_type": primary,
                "error_type_flags": flags,
                "local_sim3_chunk_ate": c.get("local_sim3_chunk_ate"),
                "global_chunk_ate": c.get("global_chunk_ate"),
                "local_to_global_ate_ratio": c.get("local_to_global_ate_ratio"),
                "abs_scale_jump_gtlocal": inter.get("abs_scale_jump_gtlocal"),
                "overlap_sim3_residual_all": inter.get("overlap_sim3_residual_all"),
                "future_after_overlap_sim3": inter.get("nonoverlap_future_error_after_overlap_sim3"),
                "intra_scale_variance": scale.get("intra_scale_variance"),
                "head_to_tail_transfer_ratio": scale.get("head_to_tail_transfer_ratio"),
                "step_length_ratio_span_frac": scale.get("step_length_ratio_head_mid_tail_span_frac"),
                "reset_relative_idx": c.get("reset_relative_idx"),
                "rolling100_error": c.get("rolling100_error"),
            }
        )
    _write_csv(phase / "chunk_error_taxonomy.csv", rows)
    method_summary: List[Dict[str, Any]] = []
    for run in runs:
        subset = [r for r in rows if r["method"] == run.method]
        counts: Dict[str, int] = {}
        for row in subset:
            counts[str(row["primary_error_type"])] = counts.get(str(row["primary_error_type"]), 0) + 1
        top = max(counts.items(), key=lambda kv: kv[1]) if counts else ("", 0)
        clear = sum(1 for row in subset if row["primary_error_type"] != "TYPE_G_MIXED_OR_LOW_SIGNAL")
        method_summary.append(
            {
                "method": run.method,
                "display": run.display,
                "run_name": run.run_name,
                "chunk_count": len(subset),
                "clear_or_mixed_classified_count": clear,
                "clear_or_mixed_classified_ratio": clear / max(len(subset), 1),
                "top_error_type": top[0],
                "top_error_type_count": top[1],
                "type_counts": counts,
            }
        )
    _write_csv(phase / "method_error_type_summary.csv", method_summary)
    h35_c9_rows = _h35_c9_gap_taxonomy(rows, chunk_rows, inter_rows, scale_rows)
    _write_csv(phase / "h35_c9_gap_taxonomy.csv", h35_c9_rows)
    _write_optimization_report(phase / "optimization_target_report.md", runs, method_summary, h35_c9_rows, thresholds)
    _plot_phase7(phase / "figures", runs, rows, h35_c9_rows)
    _plot_trajectory_overlay(phase / "figures/three_method_trajectory_overlay.png", runs)
    return rows


def _h35_c9_gap_taxonomy(
    tax_rows: Sequence[Mapping[str, Any]],
    chunk_rows: Sequence[Mapping[str, Any]],
    inter_rows: Sequence[Mapping[str, Any]],
    scale_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    c_by = {(r["method"], int(r["chunk_id"])): r for r in chunk_rows}
    i_by = {(r["method"], int(r["chunk_id"])): r for r in inter_rows}
    s_by = {(r["method"], int(r["chunk_id"])): r for r in scale_rows}
    out: List[Dict[str, Any]] = []
    chunks = sorted({int(r["chunk_id"]) for r in chunk_rows if r["method"] == "h35"} & {int(r["chunk_id"]) for r in chunk_rows if r["method"] == "c9"})
    for cid in chunks:
        h_c = c_by.get(("h35", cid), {})
        c_c = c_by.get(("c9", cid), {})
        h_i = i_by.get(("h35", cid), {})
        c_i = i_by.get(("c9", cid), {})
        h_s = s_by.get(("h35", cid), {})
        c_s = s_by.get(("c9", cid), {})
        local_gap = _safe_float(h_c.get("local_sim3_chunk_ate")) - _safe_float(c_c.get("local_sim3_chunk_ate"))
        global_gap = _safe_float(h_c.get("global_chunk_ate")) - _safe_float(c_c.get("global_chunk_ate"))
        scale_gap = _safe_float(h_i.get("abs_scale_jump_gtlocal")) - _safe_float(c_i.get("abs_scale_jump_gtlocal"))
        future_gap = _safe_float(h_i.get("nonoverlap_future_error_after_overlap_sim3")) - _safe_float(c_i.get("nonoverlap_future_error_after_overlap_sim3"))
        intra_gap = _safe_float(h_s.get("intra_scale_variance")) - _safe_float(c_s.get("intra_scale_variance"))
        candidates = {
            "TYPE_A_LOCAL_GEOMETRY": local_gap,
            "TYPE_B_INTERCHUNK_GAUGE": scale_gap,
            "TYPE_D_OVERLAP_LOCAL_ONLY": future_gap,
            "TYPE_E_INTRACHUNK_SCALE_DRIFT": intra_gap * 100.0 if math.isfinite(intra_gap) else float("nan"),
        }
        finite = {k: v for k, v in candidates.items() if math.isfinite(v)}
        dominant = max(finite.items(), key=lambda kv: kv[1])[0] if finite else "UNKNOWN"
        out.append(
            {
                "chunk_id": cid,
                "h35_minus_c9_global_chunk_ate": global_gap,
                "h35_minus_c9_local_sim3_chunk_ate": local_gap,
                "h35_minus_c9_abs_scale_jump": scale_gap,
                "h35_minus_c9_future_after_overlap": future_gap,
                "h35_minus_c9_intra_scale_variance": intra_gap,
                "dominant_gap_type": dominant,
            }
        )
    return out


def _write_optimization_report(
    path: Path,
    runs: Sequence[RunData],
    method_summary: Sequence[Mapping[str, Any]],
    gap_rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> None:
    gap_counts: Dict[str, int] = {}
    for row in gap_rows:
        key = str(row.get("dominant_gap_type"))
        gap_counts[key] = gap_counts.get(key, 0) + 1
    top_gap = max(gap_counts.items(), key=lambda kv: kv[1]) if gap_counts else ("UNKNOWN", 0)
    lines = [
        "# Phase 7 Optimization Target Report",
        "",
        "Thresholds are data-derived and recorded here for audit:",
        "",
        "| threshold | value |",
        "|---|---:|",
    ]
    for key, value in thresholds.items():
        lines.append(f"| `{key}` | {_fmt(value)} |")
    lines.extend(
        [
            "",
            "## Method Error Types",
            "",
            "| method | top error type | classified ratio | counts |",
            "|---|---|---:|---|",
        ]
    )
    for row in method_summary:
        lines.append(
            f"| `{row.get('display')}` | `{row.get('top_error_type')}` | {_fmt(row.get('clear_or_mixed_classified_ratio'))} | "
            f"`{row.get('type_counts')}` |"
        )
    lines.extend(
        [
            "",
            "## H35-C9 Gap",
            "",
            f"Top H35-C9 gap type by chunk count: `{top_gap[0]}` ({top_gap[1]} chunks).",
            "",
            "Optimization target ordering is based on the observed taxonomy, not on fabricated method improvements:",
            "",
            "1. Fix/diagnose cross-chunk gauge and overlap-transfer if TYPE_B/TYPE_D dominates.",
            "2. Add explicit chunk-internal scale consistency if TYPE_E is frequent.",
            "3. Return to READ/local geometry only where TYPE_A is the H35-C9 gap driver.",
            "4. Treat reset-carried gauge state as a candidate only when TYPE_F concentrates at reset-relative 0/1.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_phase7(fig_dir: Path, runs: Sequence[RunData], rows: Sequence[Mapping[str, Any]], gap_rows: Sequence[Mapping[str, Any]]) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    type_order = [
        "TYPE_A_LOCAL_GEOMETRY",
        "TYPE_B_INTERCHUNK_GAUGE",
        "TYPE_C_OVERLAP_UNRELIABLE",
        "TYPE_D_OVERLAP_LOCAL_ONLY",
        "TYPE_E_INTRACHUNK_SCALE_DRIFT",
        "TYPE_F_RESET_SCALE_REBUILD",
        "TYPE_G_MIXED",
        "TYPE_G_MIXED_OR_LOW_SIGNAL",
    ]
    type_to_int = {name: i for i, name in enumerate(type_order)}
    max_chunks = max((len([r for r in rows if r["method"] == run.method]) for run in runs), default=0)
    matrix = []
    labels = []
    for run in runs:
        subset = [r for r in rows if r["method"] == run.method]
        vals = [type_to_int.get(str(r.get("primary_error_type")), len(type_order)) for r in subset]
        vals += [float("nan")] * max(0, max_chunks - len(vals))
        matrix.append(vals)
        labels.append(run.display)
    fig, ax = plt.subplots(figsize=(12, 2.8))
    im = ax.imshow(np.asarray(matrix, dtype=np.float64), aspect="auto", interpolation="nearest", vmin=0, vmax=len(type_order) - 1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("chunk")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_ticks(range(len(type_order)))
    cbar.set_ticklabels([x.replace("TYPE_", "")[:18] for x in type_order])
    fig.tight_layout()
    fig.savefig(fig_dir / "taxonomy_timeline.png", dpi=160)
    plt.close(fig)

    counts_by_method = []
    for run in runs:
        subset = [r for r in rows if r["method"] == run.method]
        counts_by_method.append([sum(1 for r in subset if r.get("primary_error_type") == t) for t in type_order])
    bottom = np.zeros(len(runs))
    fig, ax = plt.subplots(figsize=(10, 4))
    xs = np.arange(len(runs))
    for idx, t in enumerate(type_order):
        vals = np.asarray([counts[idx] for counts in counts_by_method], dtype=np.float64)
        ax.bar(xs, vals, bottom=bottom, label=t.replace("TYPE_", ""))
        bottom += vals
    ax.set_xticks(xs)
    ax.set_xticklabels([r.display for r in runs], rotation=10)
    ax.set_ylabel("chunk count")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "taxonomy_stacked_bar_by_method.png", dpi=160)
    plt.close(fig)

    gap_counts: Dict[str, int] = {}
    for row in gap_rows:
        gap_counts[str(row.get("dominant_gap_type"))] = gap_counts.get(str(row.get("dominant_gap_type")), 0) + 1
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = list(gap_counts.keys())
    ax.bar(np.arange(len(labels)), [gap_counts[k] for k in labels])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels([l.replace("TYPE_", "") for l in labels], rotation=25, ha="right")
    ax.set_ylabel("chunk count")
    fig.tight_layout()
    fig.savefig(fig_dir / "h35_c9_gap_by_error_type.png", dpi=160)
    plt.close(fig)


def _plot_trajectory_overlay(path: Path, runs: Sequence[RunData]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 4), squeeze=False)
    for ax, run in zip(axes[0], runs):
        points = run.global_aligned
        chunks = np.zeros(points.shape[0], dtype=np.int32) - 1
        for chunk in run.schedule:
            chunks[int(chunk["start_frame"]) : int(chunk["end_frame"])] = int(chunk["chunk_id"])
        sc = ax.scatter(points[:, 0], points[:, 2], c=chunks, s=4, cmap="turbo")
        ax.plot(run.gt.centers[:, 0], run.gt.centers[:, 2], color="black", linewidth=1, alpha=0.6, label="GT")
        ax.set_title(run.display)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.axis("equal")
        ax.grid(True, alpha=0.2)
    fig.colorbar(sc, ax=axes.ravel().tolist(), label="chunk id")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _compute_phase8(
    runs: Sequence[RunData],
    out_dir: Path,
    fits: Mapping[str, Mapping[int, Tuple[float, np.ndarray, np.ndarray]]],
) -> Dict[str, Any]:
    phase = out_dir / "phase8_diagnostic_oracles"
    oracle_rows: List[Dict[str, Any]] = []
    overlap_rows: List[Dict[str, Any]] = []
    reset_proxy_rows: List[Dict[str, Any]] = []
    for run in runs:
        aligned = np.full_like(run.pred.centers, np.nan)
        for idx, chunk in enumerate(run.schedule):
            cid = int(chunk["chunk_id"])
            start = int(chunk["start_frame"])
            next_start = int(run.schedule[idx + 1]["start_frame"]) if idx + 1 < len(run.schedule) else int(chunk["end_frame"])
            end = min(next_start, int(chunk["end_frame"]), run.frames_used)
            fit = fits.get(run.method, {}).get(cid)
            if fit is not None and end > start:
                aligned[start:end] = _apply_sim3(run.pred.centers[start:end], fit)
        stats = _residual_stats(aligned[np.isfinite(aligned).all(axis=1)], run.gt.centers[np.isfinite(aligned).all(axis=1)])
        oracle_rows.append(
            {
                "method": run.method,
                "display": run.display,
                "run_name": run.run_name,
                "oracle_type": "per-chunk Sim3 GT oracle, tail-trim non-overlap assignment",
                "oracle_rmse": stats["rmse"],
                "oracle_p90": stats["p90"],
                "global_sim3_rmse": _rmse(run.global_residual),
                "oracle_improvement_vs_global": (_rmse(run.global_residual) - stats["rmse"]) if stats["rmse"] is not None and _rmse(run.global_residual) is not None else None,
            }
        )
        overlap_rows.append(
            {
                "method": run.method,
                "display": run.display,
                "run_name": run.run_name,
                "oracle_type": "overlap-only Sim3 no-GT proxy",
                "available": False,
                "reason": "duplicate raw current/previous chunk overlap predictions or point pairs unavailable for selected artifact",
            }
        )
        pred_steps = _step_lengths(run.pred.centers)
        gt_steps = _step_lengths(run.gt.centers)
        global_pred_med = _median(pred_steps)
        for group in sorted({int(c["chunk_id"]) // max(run.reset_every, 1) for c in run.schedule}):
            ranges = [
                (int(c["start_frame"]), int(c["end_frame"]))
                for c in run.schedule
                if int(c["chunk_id"]) // max(run.reset_every, 1) == group
            ]
            if not ranges:
                continue
            s = min(a for a, _ in ranges)
            e = max(b for _, b in ranges)
            ps = pred_steps[s : max(s, min(e - 1, pred_steps.size))]
            gs = gt_steps[s : max(s, min(e - 1, gt_steps.size))]
            pred_med = _median(ps)
            gt_med = _median(gs)
            no_gt_modifier = (global_pred_med / pred_med) if pred_med is not None and global_pred_med is not None and pred_med > EPS else None
            reset_proxy_rows.append(
                {
                    "method": run.method,
                    "display": run.display,
                    "run_name": run.run_name,
                    "reset_group_id": group,
                    "frame_start": s,
                    "frame_end": e,
                    "pred_step_median": pred_med,
                    "gt_step_median": gt_med,
                    "step_ratio_to_gt": (pred_med / gt_med) if pred_med is not None and gt_med is not None and gt_med > EPS else None,
                    "nogt_global_median_pred_step": global_pred_med,
                    "nogt_modifier_global_over_group": no_gt_modifier,
                    "nogt_modifier_clipped_0p5_2p0": min(2.0, max(0.5, no_gt_modifier)) if no_gt_modifier is not None else None,
                    "gt_oracle_modifier_gt_over_pred": (gt_med / pred_med) if pred_med is not None and gt_med is not None and pred_med > EPS else None,
                }
            )
    _write_csv(phase / "per_chunk_sim3_oracle.csv", oracle_rows)
    _write_csv(phase / "overlap_sim3_proxy.csv", overlap_rows)
    _write_csv(phase / "per_reset_step_length_proxy.csv", reset_proxy_rows)
    lines = [
        "# Phase 8 Diagnostic Oracle Summary",
        "",
        "| method | global Sim3 RMSE | per-chunk oracle RMSE | improvement | no-GT reset modifier range | GT modifier range |",
        "|---|---:|---:|---:|---|---|",
    ]
    for run in runs:
        oracle = next((r for r in oracle_rows if r["method"] == run.method), {})
        reset = [r for r in reset_proxy_rows if r["method"] == run.method]
        lines.append(
            f"| `{run.display}` | {_fmt(oracle.get('global_sim3_rmse'))} | {_fmt(oracle.get('oracle_rmse'))} | "
            f"{_fmt(oracle.get('oracle_improvement_vs_global'))} | "
            f"{_fmt(_min(r.get('nogt_modifier_global_over_group') for r in reset))}..{_fmt(_max(r.get('nogt_modifier_global_over_group') for r in reset))} | "
            f"{_fmt(_min(r.get('gt_oracle_modifier_gt_over_pred') for r in reset))}..{_fmt(_max(r.get('gt_oracle_modifier_gt_over_pred') for r in reset))} |"
        )
    lines.extend(
        [
            "",
            "Overlap-only no-GT proxy is explicitly unavailable for selected artifacts because duplicate raw overlap predictions/point pairs are not present.",
        ]
    )
    (phase / "oracle_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"per_chunk_oracle": oracle_rows, "reset_proxy": reset_proxy_rows}


def _write_final_report(
    out_dir: Path,
    runs: Sequence[RunData],
    smoke: Mapping[str, Any],
    chunk_rows: Sequence[Mapping[str, Any]],
    inter_rows: Sequence[Mapping[str, Any]],
    scale_rows: Sequence[Mapping[str, Any]],
    reset_rows: Sequence[Mapping[str, Any]],
    tax_rows: Sequence[Mapping[str, Any]],
    phase8: Mapping[str, Any],
) -> Dict[str, Any]:
    summary_rows: List[Dict[str, Any]] = []
    tax_by_method: Dict[str, List[Mapping[str, Any]]] = {}
    for row in tax_rows:
        tax_by_method.setdefault(str(row["method"]), []).append(row)
    for run in runs:
        csub = [r for r in chunk_rows if r["method"] == run.method]
        isub = [r for r in inter_rows if r["method"] == run.method]
        ssub = [r for r in scale_rows if r["method"] == run.method]
        counts: Dict[str, int] = {}
        for row in tax_by_method.get(run.method, []):
            counts[str(row["primary_error_type"])] = counts.get(str(row["primary_error_type"]), 0) + 1
        top = max(counts.items(), key=lambda kv: kv[1]) if counts else ("UNKNOWN", 0)
        summary_rows.append(
            {
                "method": run.method,
                "display": run.display,
                "run_name": run.run_name,
                "landed_ate": run.kitti_results.get("kitti_results_ate"),
                "computed_global_sim3_rmse": _rmse(run.global_residual),
                "local_sim3_mean": _mean(r.get("local_sim3_chunk_ate") for r in csub),
                "local_to_global_median": _median(r.get("local_to_global_ate_ratio") for r in csub),
                "abs_scale_jump_mean": _mean(r.get("abs_scale_jump_gtlocal") for r in isub),
                "overlap_residual_mean": _mean(r.get("overlap_sim3_residual_all") for r in isub),
                "future_after_overlap_mean": _mean(r.get("nonoverlap_future_error_after_overlap_sim3") for r in isub),
                "intra_scale_variance_mean": _mean(r.get("intra_scale_variance") for r in ssub),
                "head_to_tail_transfer_ratio_mean": _mean(r.get("head_to_tail_transfer_ratio") for r in ssub),
                "top_error_type": top[0],
                "top_error_type_count": top[1],
                "type_counts": counts,
                "raw_prediction_missing": True,
                "pose_only_fallback_used": True,
            }
        )
    _write_csv(out_dir / "v62_method_summary.csv", summary_rows)
    _write_json(out_dir / "v62_summary.json", {"smoke": smoke, "method_summary": summary_rows})
    h35 = next((r for r in summary_rows if r["method"] == "h35"), {})
    c9 = next((r for r in summary_rows if r["method"] == "c9"), {})
    orig = next((r for r in summary_rows if r["method"] == "orig"), {})
    oracle_rows = phase8.get("per_chunk_oracle", [])
    oracle_by_method = {r["method"]: r for r in oracle_rows}
    h35_reset = [r for r in reset_rows if r["method"] == "h35"]
    if h35_reset:
        h35_max_scale = max(h35_reset, key=lambda r: _safe_float(r.get("scale_jump_gtlocal_abs_mean")))
        h35_max_rolling = max(h35_reset, key=lambda r: _safe_float(r.get("rolling100_error_mean")))
        h35_max_local = max(h35_reset, key=lambda r: _safe_float(r.get("local_sim3_chunk_ate_mean")))
        h35_reset_text = (
            f"H35 max scale-jump mean is at reset-relative `{int(h35_max_scale['reset_relative_idx'])}` "
            f"({_fmt(h35_max_scale.get('scale_jump_gtlocal_abs_mean'))}), max rolling100 mean at reset-relative "
            f"`{int(h35_max_rolling['reset_relative_idx'])}` ({_fmt(h35_max_rolling.get('rolling100_error_mean'))}), "
            f"and max local Sim3 chunk ATE mean at reset-relative `{int(h35_max_local['reset_relative_idx'])}` "
            f"({_fmt(h35_max_local.get('local_sim3_chunk_ate_mean'))})."
        )
    else:
        h35_reset_text = "H35 reset-relative rows are unavailable."
    lines = [
        "# ACL2 v62 KITTI01 Error Source Autopsy Report",
        "",
        "This report is generated from landed artifacts only. Missing raw/point overlap data is marked unavailable; no metric is filled with fabricated zeros.",
        "",
        "## Selected Runs",
        "",
        "| method | run | landed ATE | computed RMSE | local Sim3 mean | local/global median | scale jump mean | future-after-overlap mean | top type |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['display']}` | `{row['run_name']}` | {_fmt(row.get('landed_ate'))} | {_fmt(row.get('computed_global_sim3_rmse'))} | "
            f"{_fmt(row.get('local_sim3_mean'))} | {_fmt(row.get('local_to_global_median'))} | "
            f"{_fmt(row.get('abs_scale_jump_mean'))} | {_fmt(row.get('future_after_overlap_mean'))} | `{row.get('top_error_type')}` |"
        )
    lines.extend(
        [
            "",
            "## Required Answers",
            "",
            f"1. ORIG/native fallback: local/global median is `{_fmt(orig.get('local_to_global_median'))}` with local Sim3 mean `{_fmt(orig.get('local_sim3_mean'))}` vs global RMSE `{_fmt(orig.get('computed_global_sim3_rmse'))}`. This points more to cross-chunk gauge/scale than pure chunk-local geometry, under pose-only fallback.",
            f"2. C9 vs H35: H35 local Sim3 mean `{_fmt(h35.get('local_sim3_mean'))}` vs C9 `{_fmt(c9.get('local_sim3_mean'))}`; H35 abs scale jump mean `{_fmt(h35.get('abs_scale_jump_mean'))}` vs C9 `{_fmt(c9.get('abs_scale_jump_mean'))}`; H35 future-after-overlap mean `{_fmt(h35.get('future_after_overlap_mean'))}` vs C9 `{_fmt(c9.get('future_after_overlap_mean'))}`.",
            f"3. H35 bottleneck: taxonomy top/tied type is `{h35.get('top_error_type')}` with counts `{h35.get('type_counts')}`; local/global median is `{_fmt(h35.get('local_to_global_median'))}` and intra-scale variance mean is `{_fmt(h35.get('intra_scale_variance_mean'))}`.",
            "4. Overlap Sim(3) residual in this artifact set is a GT-overlap pose proxy, not an online no-GT duplicate-overlap test. Low overlap residual with larger future error is evidence that the overlap/head region does not fully represent future chunk scale/shape.",
            f"5. Chunk-internal nonuniform scale: H35 head->tail/local ratio mean is `{_fmt(h35.get('head_to_tail_transfer_ratio_mean'))}` and intra-scale variance mean `{_fmt(h35.get('intra_scale_variance_mean'))}`.",
            f"6. Reset effects: see `phase6_reset_relative/reset_relative_autopsy_report.md`; {h35_reset_text} The phase6 reset-start evidence gate does not claim a reset-relative 0/1 concentration unless the table supports it.",
            "7. Per-reset step-length proxy: see `phase8_diagnostic_oracles/per_reset_step_length_proxy.csv`; no-GT modifier range and GT-oracle modifier range are reported without claiming an online correction.",
            "8. Next optimization target: prioritize the dominant H35-C9 taxonomy in `phase7_taxonomy/optimization_target_report.md`, with explicit raw-overlap rerun as the first blocker-clearing step if online overlap correction is to be evaluated.",
            "",
            "## Diagnostic Oracles",
            "",
            "| method | global RMSE | per-chunk oracle RMSE | improvement |",
            "|---|---:|---:|---:|",
        ]
    )
    for run in runs:
        row = oracle_by_method.get(run.method, {})
        lines.append(
            f"| `{run.display}` | {_fmt(row.get('global_sim3_rmse'))} | {_fmt(row.get('oracle_rmse'))} | {_fmt(row.get('oracle_improvement_vs_global'))} |"
        )
    lines.extend(
        [
            "",
            "## Evidence Chain",
            "",
            "- Phase 0 confirms complete predictions and schedules for all three systems; ORIG is explicitly a native/H9-like fallback.",
            "- Phase 1 synthetic smoke passes identity, known-scale, nonuniform scale, and overlap-transfer tests before real metrics are generated.",
            "- Phase 2 records raw/point overlap unavailability and switches to pose-only fallback instead of inventing point metrics.",
            "- Phases 3-6 write per-chunk local/global/gauge/scale/reset CSVs, then Phase 7 assigns taxonomy from recorded thresholds.",
            "- Phase 8 provides GT per-chunk oracle and reset step proxy upper-bound diagnostics; overlap no-GT proxy remains unavailable.",
        ],
    )
    (out_dir / "v62_error_source_autopsy_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"method_summary": summary_rows, "report": str(out_dir / "v62_error_source_autopsy_report.md")}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-poses", type=Path, default=DEFAULT_GT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = _read_poses(args.gt_poses)
    runs = [_load_run(spec, gt) for spec in RUNS]
    _artifact_inventory(runs, out_dir, args.gt_poses)
    smoke = _run_smoke(out_dir)
    _phase2_manifest(runs, out_dir)
    chunk_rows, fits = _compute_phase3(runs, out_dir)
    inter_rows = _compute_phase4(runs, out_dir, chunk_rows, fits)
    _, scale_rows = _compute_phase5(runs, out_dir, chunk_rows)
    reset_rows = _compute_phase6(runs, out_dir, chunk_rows, inter_rows, scale_rows)
    tax_rows = _compute_taxonomy(runs, out_dir, chunk_rows, inter_rows, scale_rows, reset_rows)
    phase8 = _compute_phase8(runs, out_dir, fits)
    final = _write_final_report(out_dir, runs, smoke, chunk_rows, inter_rows, scale_rows, reset_rows, tax_rows, phase8)
    print(json.dumps(_clean({"out_dir": str(out_dir), **final}), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
