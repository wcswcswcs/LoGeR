#!/usr/bin/env python3
"""ACL2 v63 TTT-scale causal diagnostic report generator.

This tool combines landed H35/C9 artifacts with v63 fork rollouts. It is
diagnostic-only: unavailable post-zp tensors, duplicate point overlap, and
unrun component forks are written as unavailable/no-data instead of fabricated
metrics.
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
GT_PATH = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
RESULT_ROOT = ROOT / "results/kitti01_hmc_v2/acl2_v63_ttt_scale_causal_diagnostic"
OUT_DEFAULT = RESULT_ROOT / "report_final"

H35_LANDED = ROOT / (
    "results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/"
    "phase7_layergamma_fix_full/rollouts/V53_PHASE7_FULL_H35_LAYERGAMMAFIX_RHO0075"
)
C9_LANDED = ROOT / (
    "results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/"
    "phase0_hard_gate/rollouts/V45_P0_C9_REPEAT"
)
SMOKE_RUN = RESULT_ROOT / "phase1_instrumentation_smoke/rollouts/V63_P1_H35_TRACE_SMOKE_E96"
PATCH_SMOKE_RUN = RESULT_ROOT / "phase0_trace_patch_smoke/rollouts/V63_P0_H35_POSTZP_LOG_SMOKE_E96"
H35_FULL_TRACE = RESULT_ROOT / "phase0_full_trace/rollouts/V63_P0_H35_POSTZP_TRACE_FULL"
C9_FULL_TRACE = RESULT_ROOT / "phase0_full_trace/rollouts/V63_P0_C9_POSTZP_TRACE_FULL"
FORK_BASE = RESULT_ROOT / "phase3_ttt_causal_fork_basic/rollouts"
FORK_RUNS = {
    "F0_BASE": "V63_P3_H35_C19_BASE_E730",
    "F1_NO_TTT_WRITE": "V63_P3_H35_C19_FREEZE_E730",
    "F2_NATIVE_COMMIT_ONLY": "V63_P3_H35_C19_NATIVE0_E730",
    "F4_HALF_DELTA_COMMIT": "V63_P3_H35_C19_HALF_E730",
    "F5_DOUBLE_DELTA_COMMIT": "V63_P3_H35_C19_DOUBLE_E730",
}
FORK_CHUNK_ID = 19


EPS = 1e-12


@dataclass
class PoseData:
    centers: np.ndarray
    rotations: np.ndarray
    count: int


@dataclass
class RunData:
    key: str
    display: str
    path: Path
    poses: PoseData
    hmc_rows: List[Dict[str, Any]]
    pose_trace_rows: List[Dict[str, Any]]
    global_aligned: np.ndarray
    global_residual: np.ndarray
    prefix_gt: PoseData


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
                elif isinstance(value, (list, dict)):
                    out[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
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
    vals = []
    for value in values:
        val = _safe_float(value)
        if math.isfinite(val):
            vals.append(val)
    return vals


def _mean(values: Iterable[Any]) -> Optional[float]:
    vals = _finite(values)
    return float(np.mean(vals)) if vals else None


def _std(values: Iterable[Any]) -> Optional[float]:
    vals = _finite(values)
    return float(np.std(vals)) if vals else None


def _max(values: Iterable[Any]) -> Optional[float]:
    vals = _finite(values)
    return float(np.max(vals)) if vals else None


def _fmt(value: Any, digits: int = 6) -> str:
    val = _safe_float(value)
    return f"{val:.{digits}f}" if math.isfinite(val) else "NA"


def _corr(xs: Iterable[Any], ys: Iterable[Any]) -> Optional[float]:
    pairs = []
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


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


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
            if len(parts) == 12:
                mat = np.asarray([float(x) for x in parts], dtype=np.float64).reshape(3, 4)
                rotations.append(mat[:, :3])
                centers.append(mat[:, 3].tolist())
            elif len(parts) >= 8:
                tx, ty, tz = map(float, parts[1:4])
                qx, qy, qz, qw = map(float, parts[4:8])
                rotations.append(_quat_to_rot(qx, qy, qz, qw))
                centers.append([tx, ty, tz])
            if limit is not None and len(centers) >= limit:
                break
    arr = np.asarray(centers, dtype=np.float64)
    rots = np.asarray(rotations, dtype=np.float64)
    return PoseData(arr, rots, int(arr.shape[0]))


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


def _fit_sim3(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = min(len(src), len(dst))
    src = src[:n]
    dst = dst[:n]
    if n < 3:
        return 1.0, np.eye(3), np.zeros(3)
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    xs = src - mu_src
    yd = dst - mu_dst
    cov = (yd.T @ xs) / n
    u, d, vt = np.linalg.svd(cov)
    s_mat = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        s_mat[-1, -1] = -1
    rot = u @ s_mat @ vt
    var_src = float(np.sum(xs * xs) / n)
    scale = float(np.trace(np.diag(d) @ s_mat) / max(var_src, EPS))
    trans = mu_dst - scale * (rot @ mu_src)
    return scale, rot, trans


def _apply_sim3(points: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return scale * (points @ rot.T) + trans[None, :]


def _rmse(vec: np.ndarray) -> Optional[float]:
    arr = np.asarray(vec, dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.sqrt(np.mean(np.sum(arr * arr, axis=-1)))) if arr.ndim > 1 else float(np.sqrt(np.mean(arr * arr)))


def _load_run(key: str, display: str, path: Path, gt_all: PoseData) -> Optional[RunData]:
    pred_path = path / "01.txt"
    if not pred_path.is_file():
        return None
    pred = _read_poses(pred_path)
    n = min(pred.count, gt_all.count)
    pred = PoseData(pred.centers[:n], pred.rotations[:n], n)
    gt = PoseData(gt_all.centers[:n], gt_all.rotations[:n], n)
    scale, rot, trans = _fit_sim3(pred.centers, gt.centers)
    aligned = _apply_sim3(pred.centers, scale, rot, trans)
    return RunData(
        key=key,
        display=display,
        path=path,
        poses=pred,
        hmc_rows=_read_jsonl(path / "hmc_state_hash.jsonl"),
        pose_trace_rows=_read_jsonl(path / "per_chunk_pose_trace.jsonl"),
        global_aligned=aligned,
        global_residual=aligned - gt.centers,
        prefix_gt=gt,
    )


def _chunk_range(row: Mapping[str, Any], fallback_chunk: int) -> Tuple[int, int]:
    start = row.get("start_frame")
    end = row.get("end_frame")
    if start is None or end is None:
        start = int(fallback_chunk) * 29
        end = start + 32
    return int(start), int(end)


def _chunk_metrics(run: RunData) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, hmc in enumerate(run.hmc_rows):
        chunk_id = int(hmc.get("chunk_idx", idx))
        start, end = _chunk_range(hmc, chunk_id)
        start = max(0, min(start, run.poses.count))
        end = max(start, min(end, run.poses.count))
        pred = run.poses.centers[start:end]
        gt = run.prefix_gt.centers[start:end]
        if len(pred) >= 3:
            scale, rot, trans = _fit_sim3(pred, gt)
            local = _apply_sim3(pred, scale, rot, trans)
            local_rmse = _rmse(local - gt)
            log_scale = float(math.log(abs(scale) + EPS))
        else:
            scale, local_rmse, log_scale = None, None, None
        glob = run.global_aligned[start:end]
        gres = glob - gt if len(gt) else np.zeros((0, 3))
        layer_branch = _layer_branch_aggregates(hmc.get("probe_ttt_write_layer_branch_summary"))
        row = {
            "run_key": run.key,
            "display": run.display,
            "run_name": run.path.name,
            "chunk_id": chunk_id,
            "frame_start": start,
            "frame_end": end,
            "reset_group_id": chunk_id // 5,
            "reset_relative_idx": chunk_id % 5,
            "current_chunk_global_ate": _rmse(gres),
            "current_chunk_local_sim3_ate": local_rmse,
            "scale_fit_pred_to_gt": scale,
            "log_scale_residual": log_scale,
            "abs_log_scale_residual": abs(log_scale) if log_scale is not None else None,
            "rolling100": _rmse(run.global_residual[max(0, start - 50) : min(run.poses.count, end + 50)]),
            "probe_ttt_write_tri_delta_norm_mean": hmc.get("probe_ttt_write_tri_delta_norm_mean"),
            "probe_ttt_write_tri_delta_norm_max": hmc.get("probe_ttt_write_tri_delta_norm_max"),
            "probe_ttt_write_tri_pos_mass_mean": hmc.get("probe_ttt_write_tri_pos_mass_mean", hmc.get("auxgeo_tri_replay_pos_mass_mean")),
            "probe_ttt_write_tri_neu_mass_mean": hmc.get("probe_ttt_write_tri_neu_mass_mean", hmc.get("auxgeo_tri_replay_neu_mass_mean")),
            "probe_ttt_write_tri_neg_mass_mean": hmc.get("probe_ttt_write_tri_neg_mass_mean", hmc.get("auxgeo_tri_replay_neg_mass_mean")),
            "memory_ttt_mean_rel_diff": hmc.get("memory_ttt_mean_rel_diff"),
            "memory_ttt_w0_mean_rel_diff": hmc.get("memory_ttt_w0_mean_rel_diff"),
            "probe_ttt_write_post_delta_norm_mean": hmc.get("probe_ttt_write_post_delta_norm_mean"),
            "probe_ttt_write_native_delta_norm_mean": hmc.get("probe_ttt_write_native_delta_norm_mean"),
            "probe_ttt_write_action_delta_norm_mean": hmc.get("probe_ttt_write_action_delta_norm_mean"),
            "probe_ttt_write_native_cosine_mean": hmc.get("probe_ttt_write_native_cosine_mean"),
            "probe_ttt_write_action_native_cosine_mean": hmc.get("probe_ttt_write_action_native_cosine_mean"),
            "probe_committed_state_hash": hmc.get("probe_committed_state_hash"),
            "controlled_input_state_hash": hmc.get("controlled_input_state_hash"),
            "controlled_output_state_hash": hmc.get("controlled_output_state_hash"),
            "commit_source_state_hash": hmc.get("commit_source_state_hash"),
            "probe_ttt_write_state_hash": hmc.get("probe_ttt_write_state_hash"),
            "frozen_ttt_state_hash": hmc.get("frozen_ttt_state_hash"),
            "ttt_semantic_scaled_state_hash": hmc.get("ttt_semantic_scaled_state_hash"),
            "ttt_write_commit_frozen_at_chunk": hmc.get("ttt_write_commit_frozen_at_chunk"),
            "ttt_semantic_write_scale_at_chunk": hmc.get("ttt_semantic_write_scale_at_chunk"),
            **layer_branch,
        }
        rows.append(row)
    by_chunk = {int(r["chunk_id"]): r for r in rows}
    for row in rows:
        cid = int(row["chunk_id"])
        for h in (1, 3, 5):
            fut = [by_chunk.get(cid + j, {}) for j in range(1, h + 1)]
            row[f"future_h{h}_scale_residual"] = _mean(r.get("abs_log_scale_residual") for r in fut)
            row[f"future_h{h}_global_ate"] = _mean(r.get("current_chunk_global_ate") for r in fut)
            row[f"future_h{h}_rolling100"] = _mean(r.get("rolling100") for r in fut)
    cumulative = 0.0
    cumulative_post_zp = 0.0
    current_reset = None
    for row in rows:
        reset = int(row["reset_group_id"])
        if reset != current_reset:
            current_reset = reset
            cumulative = 0.0
            cumulative_post_zp = 0.0
        delta = _safe_float(row.get("probe_ttt_write_tri_delta_norm_mean"))
        if math.isfinite(delta):
            cumulative += delta
        post_delta = _safe_float(row.get("probe_ttt_write_post_delta_norm_mean"))
        if math.isfinite(post_delta):
            cumulative_post_zp += post_delta
        row["cumulative_tri_delta_norm_since_reset"] = cumulative
        row["cumulative_post_zp_delta_norm_since_reset"] = cumulative_post_zp
    return rows


def _layer_branch_rows(run: RunData) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, hmc in enumerate(run.hmc_rows):
        chunk_id = int(hmc.get("chunk_idx", idx))
        summary = hmc.get("probe_ttt_write_layer_branch_summary") or []
        if not isinstance(summary, list):
            continue
        for item in summary:
            if isinstance(item, dict):
                out.append({"run_key": run.key, "run_name": run.path.name, "chunk_id": chunk_id, **item})
    return out


def _pose_trace_by_chunk(run: RunData) -> Dict[int, Dict[str, Any]]:
    return {int(r.get("chunk_idx")): r for r in run.pose_trace_rows if "chunk_idx" in r}


def _pose_max_abs_delta(base: RunData, other: RunData, chunk_id: int) -> Optional[float]:
    b = _pose_trace_by_chunk(base).get(chunk_id)
    o = _pose_trace_by_chunk(other).get(chunk_id)
    if not b or not o or "camera_poses" not in b or "camera_poses" not in o:
        return None
    ba = np.asarray(b["camera_poses"], dtype=np.float64)
    oa = np.asarray(o["camera_poses"], dtype=np.float64)
    n = min(len(ba), len(oa))
    if n == 0:
        return None
    return float(np.max(np.abs(ba[:n] - oa[:n])))


def _rotation_delta_stats(ref: PoseData, run: PoseData) -> Dict[str, Optional[float]]:
    n = min(ref.count, run.count, len(ref.rotations), len(run.rotations))
    if n == 0:
        return {"raw_rotation_mean_deg_diff": None, "raw_rotation_max_deg_diff": None}
    vals: List[float] = []
    for a, b in zip(ref.rotations[:n], run.rotations[:n]):
        rel = np.asarray(a, dtype=np.float64).T @ np.asarray(b, dtype=np.float64)
        cos = float(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))
        vals.append(float(math.degrees(math.acos(cos))))
    return {
        "raw_rotation_mean_deg_diff": float(np.mean(vals)),
        "raw_rotation_max_deg_diff": float(np.max(vals)),
    }


def _state_hash_sequence(run: RunData) -> List[str]:
    hashes: List[str] = []
    for row in run.hmc_rows:
        value = row.get("hash_H_next") or row.get("hash_H_m_after_commit") or row.get("commit_source_state_hash")
        if value is not None:
            hashes.append(str(value))
    return hashes


def _state_hash_parity(ref: RunData, run: RunData) -> Optional[bool]:
    ref_hashes = _state_hash_sequence(ref)
    run_hashes = _state_hash_sequence(run)
    n = min(len(ref_hashes), len(run_hashes))
    if n == 0:
        return None
    return ref_hashes[:n] == run_hashes[:n]


def _summary_mean(
    summary: Any,
    key: str,
    *,
    layer: Optional[int] = None,
    branch: Optional[str] = None,
) -> Optional[float]:
    if not isinstance(summary, list):
        return None
    vals: List[float] = []
    for item in summary:
        if not isinstance(item, dict):
            continue
        if layer is not None and int(item.get("layer", -1)) != layer:
            continue
        if branch is not None and str(item.get("branch")) != branch:
            continue
        val = _safe_float(item.get(key))
        if math.isfinite(val):
            vals.append(val)
    return float(np.mean(vals)) if vals else None


def _layer_branch_aggregates(summary: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(summary, list):
        out["layer_branch_row_count"] = 0
        return out
    out["layer_branch_row_count"] = len([r for r in summary if isinstance(r, dict)])
    for layer in (0, 8, 17):
        out[f"post_zp_delta_norm_layer{layer}"] = _summary_mean(
            summary, "post_zp_committed_delta_norm", layer=layer
        )
        out[f"post_zp_action_delta_norm_layer{layer}"] = _summary_mean(
            summary, "post_zp_action_delta_norm", layer=layer
        )
        out[f"candidate_native_cosine_layer{layer}"] = _summary_mean(
            summary, "candidate_native_cosine", layer=layer
        )
    for branch in ("w0", "w1", "w2"):
        out[f"post_zp_delta_norm_{branch}"] = _summary_mean(
            summary, "post_zp_committed_delta_norm", branch=branch
        )
        out[f"post_zp_action_delta_norm_{branch}"] = _summary_mean(
            summary, "post_zp_action_delta_norm", branch=branch
        )
        out[f"candidate_native_cosine_{branch}"] = _summary_mean(
            summary, "candidate_native_cosine", branch=branch
        )
    out["gamma_effective_mean"] = _summary_mean(summary, "gamma")
    out["rho_effective_mean"] = _summary_mean(summary, "rho")
    out["configured_rho_mean"] = _summary_mean(summary, "configured_rho")
    return out


def _phase0(out_dir: Path, runs: Sequence[RunData]) -> None:
    phase = out_dir / "phase0_trace_audit"
    rows = []
    for run in runs:
        cfg = _read_yaml(run.path / "effective_config.yaml")
        post_zp_available = any(
            math.isfinite(_safe_float(r.get("probe_ttt_write_post_delta_norm_mean")))
            for r in run.hmc_rows
        )
        tri_mass_proxy_available = any(
            math.isfinite(
                _safe_float(
                    r.get("probe_ttt_write_tri_neg_mass_mean", r.get("auxgeo_tri_replay_neg_mass_mean"))
                )
            )
            for r in run.hmc_rows
        )
        rows.append(
            {
                "run_key": run.key,
                "run_name": run.path.name,
                "path": str(run.path),
                "trajectory_available": (run.path / "01.txt").is_file(),
                "frames": run.poses.count,
                "hmc_rows": len(run.hmc_rows),
                "pose_trace_rows": len(run.pose_trace_rows),
                "pose_trace_available": bool(run.pose_trace_rows),
                "post_zp_delta_summary_available": post_zp_available,
                "tri_delta_proxy_available": any(
                    math.isfinite(_safe_float(r.get("probe_ttt_write_tri_delta_norm_mean"))) for r in run.hmc_rows
                ),
                "layer_branch_proxy_available": any(r.get("probe_ttt_write_layer_branch_summary") for r in run.hmc_rows),
                "tri_mass_proxy_available": tri_mass_proxy_available,
                "candidate_native_cosine_available": any(
                    math.isfinite(_safe_float(r.get("probe_ttt_write_native_cosine_mean"))) for r in run.hmc_rows
                ),
                "duplicate_overlap_raw_pose_available": bool(run.pose_trace_rows),
                "point_overlap_available": False,
                "read_path": cfg.get("read_path"),
                "hmc_commit_mode": cfg.get("hmc_commit_mode"),
            }
        )
    _write_csv(phase / "trace_artifact_manifest.csv", rows)
    _write_csv(
        phase / "raw_overlap_availability.csv",
        [
            {
                "run_key": row["run_key"],
                "run_name": row["run_name"],
                "frames": row["frames"],
                "pose_trace_rows": row["pose_trace_rows"],
                "duplicate_overlap_raw_pose_available": row["duplicate_overlap_raw_pose_available"],
                "point_overlap_available": row["point_overlap_available"],
                "overlap_evidence": "per_chunk_pose_trace_jsonl" if row["pose_trace_available"] else "unavailable",
            }
            for row in rows
        ],
    )
    _write_csv(
        phase / "ttt_update_trace_availability.csv",
        [
            {
                "run_key": row["run_key"],
                "run_name": row["run_name"],
                "hmc_rows": row["hmc_rows"],
                "post_zp_delta_summary_available": row["post_zp_delta_summary_available"],
                "tri_delta_proxy_available": row["tri_delta_proxy_available"],
                "tri_mass_proxy_available": row["tri_mass_proxy_available"],
                "layer_branch_proxy_available": row["layer_branch_proxy_available"],
                "candidate_native_cosine_available": row["candidate_native_cosine_available"],
            }
            for row in rows
        ],
    )
    _write_csv(
        phase / "merge_gauge_trace_availability.csv",
        [
            {
                "run_key": row["run_key"],
                "run_name": row["run_name"],
                "trajectory_available": row["trajectory_available"],
                "merge_gauge_causal_fork_available": False,
                "merge_gauge_trace_boundary": "trajectory only; no merge/gauge intervention run",
            }
            for row in rows
        ],
    )
    for key in ("h35", "c9"):
        subset = [r for r in rows if r["run_key"].startswith(key)]
        lines = [f"# {key.upper()} Trace Availability", ""]
        if not subset:
            lines.append("No run loaded.")
        for row in subset:
            lines.append(
                f"- `{row['run_name']}`: frames={row['frames']}, hmc_rows={row['hmc_rows']}, "
                f"pose_trace_rows={row['pose_trace_rows']}, tri_delta_proxy={row['tri_delta_proxy_available']}, "
                f"post_zp={row['post_zp_delta_summary_available']}, point_overlap={row['point_overlap_available']}."
            )
        _write_text(phase / f"{key}_trace_availability.md", lines)


def _phase1(
    out_dir: Path,
    h35_landed: RunData,
    c9_landed: RunData,
    *,
    smoke: Optional[RunData],
    base: Optional[RunData],
    h35_full: Optional[RunData],
    c9_full: Optional[RunData],
) -> Dict[str, Any]:
    phase = out_dir / "phase1_instrumentation_parity"
    rows: List[Dict[str, Any]] = []
    comparisons = (
        ("h35_smoke_e96", smoke, h35_landed, "short smoke vs H35 landed prefix"),
        ("h35_fork_base_e730", base, h35_landed, "C19 fork base vs H35 landed prefix"),
        ("h35_full_trace", h35_full, h35_landed, "full trace rerun vs H35 landed"),
        ("c9_full_trace", c9_full, c9_landed, "full trace rerun vs C9 landed"),
    )
    for label, run, landed, note in comparisons:
        if run is None:
            continue
        n = min(run.poses.count, landed.poses.count)
        ref = PoseData(landed.poses.centers[:n], landed.poses.rotations[:n], n)
        scale, rot, trans = _fit_sim3(ref.centers, run.prefix_gt.centers[:n])
        ref_aligned = _apply_sim3(ref.centers, scale, rot, trans)
        ref_rmse = _rmse(ref_aligned - run.prefix_gt.centers[:n])
        run_rmse = _rmse(run.global_residual[:n])
        pose_max = float(np.max(np.abs(run.poses.centers[:n] - ref.centers[:n]))) if n else None
        rot_stats = _rotation_delta_stats(ref, run.poses)
        hmc_hash_same = _state_hash_parity(landed, run)
        ate_drift = (run_rmse - ref_rmse) if run_rmse is not None and ref_rmse is not None else None
        row = {
            "label": label,
            "run_name": run.path.name,
            "reference_run_name": landed.path.name,
            "comparison_note": note,
            "frames": n,
            "trace_rmse": run_rmse,
            "landed_prefix_rmse": ref_rmse,
            "ate_drift": ate_drift,
            "rot_drift_deg": None,
            "finalerr_drift": None,
            "translation_max_abs_diff_raw_pred_vs_landed": pose_max,
            **rot_stats,
            "hmc_rows": len(run.hmc_rows),
            "pose_trace_rows": len(run.pose_trace_rows),
            "reference_hmc_rows": len(landed.hmc_rows),
            "hmc_state_hash_prefix_parity": hmc_hash_same,
            "parity_gate_ate_0p03m": abs(_safe_float(ate_drift)) <= 0.03,
            "parity_gate_raw_pose_1e_minus_4": (
                abs(_safe_float(pose_max)) <= 1e-4 if pose_max is not None else None
            ),
            "rot_finalerr_metric_source": "not emitted by current results_sim3 evaluator; raw rotation diff is reported separately",
        }
        rows.append(row)
    _write_csv(phase / "instrumentation_parity.csv", rows)
    lines = ["# Phase 1 Instrumentation No-op Parity", ""]
    for row in rows:
        lines.append(
            f"- `{row['label']}` `{row['run_name']}`: frames={row['frames']}, "
            f"ATE drift={_fmt(row['ate_drift'])}, raw translation max diff={_fmt(row['translation_max_abs_diff_raw_pred_vs_landed'])}, "
            f"raw rotation max diff={_fmt(row['raw_rotation_max_deg_diff'])} deg, "
            f"hmc_hash_prefix_parity=`{row['hmc_state_hash_prefix_parity']}`, "
            f"ATE gate={row['parity_gate_ate_0p03m']}, raw pose gate={row['parity_gate_raw_pose_1e_minus_4']}."
        )
    _write_text(phase / "instrumentation_parity_report.md", lines)
    return {
        "rows": rows,
        "gate_by_label": {
            str(row["label"]): bool(
                row.get("parity_gate_ate_0p03m")
                and row.get("parity_gate_raw_pose_1e_minus_4") is not False
            )
            for row in rows
        },
    }


def _phase2(out_dir: Path, h35_rows: List[Dict[str, Any]], c9_rows: List[Dict[str, Any]], layer_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    phase = out_dir / "phase2_passive_ttt_scale_correlation"
    _write_csv(phase / "h35_ttt_scale_chunk_table.csv", h35_rows)
    _write_csv(phase / "c9_ttt_scale_chunk_table.csv", c9_rows)
    gap_rows = []
    c9_by = {int(r["chunk_id"]): r for r in c9_rows}
    for h in h35_rows:
        c = c9_by.get(int(h["chunk_id"]), {})
        gap_rows.append(
            {
                "chunk_id": h["chunk_id"],
                "h35_future_h3_scale_residual": h.get("future_h3_scale_residual"),
                "c9_future_h3_scale_residual": c.get("future_h3_scale_residual"),
                "h35_minus_c9_future_h3_scale_residual": _safe_float(h.get("future_h3_scale_residual"))
                - _safe_float(c.get("future_h3_scale_residual")),
                "h35_tri_delta_norm": h.get("probe_ttt_write_tri_delta_norm_mean"),
                "c9_tri_delta_norm": c.get("probe_ttt_write_tri_delta_norm_mean"),
            }
        )
    _write_csv(phase / "h35_c9_ttt_scale_gap_table.csv", gap_rows)
    corr_rows = []
    for run_key, rows in (("h35", h35_rows), ("c9", c9_rows)):
        for col in (
            "probe_ttt_write_tri_delta_norm_mean",
            "probe_ttt_write_tri_neg_mass_mean",
            "memory_ttt_mean_rel_diff",
            "memory_ttt_w0_mean_rel_diff",
            "probe_ttt_write_post_delta_norm_mean",
            "probe_ttt_write_action_delta_norm_mean",
            "probe_ttt_write_native_cosine_mean",
            "probe_ttt_write_action_native_cosine_mean",
            "cumulative_post_zp_delta_norm_since_reset",
            "cumulative_tri_delta_norm_since_reset",
            "reset_relative_idx",
        ):
            corr_rows.append(
                {
                    "run_key": run_key,
                    "metric": col,
                    "target": "future_h3_scale_residual",
                    "corr": _corr((r.get(col) for r in rows), (r.get("future_h3_scale_residual") for r in rows)),
                    "n": len(rows),
                }
            )
    h35_layer_key = h35_rows[0].get("run_key") if h35_rows else "h35"
    for layer in sorted({int(r.get("layer", -1)) for r in layer_rows if r.get("run_key") == h35_layer_key}):
        sub = [r for r in layer_rows if r.get("run_key") == h35_layer_key and int(r.get("layer", -1)) == layer]
        by_chunk = {int(r["chunk_id"]): r for r in sub}
        xs, ys = [], []
        for row in h35_rows:
            lb = by_chunk.get(int(row["chunk_id"]))
            if lb:
                vals = [_safe_float(lb.get(k)) for k in ("post_zp_committed_delta_norm", "pos_delta_norm_mean", "neu_delta_norm_mean", "neg_delta_norm_mean")]
                xs.append(np.nanmean([v for v in vals if math.isfinite(v)]) if any(math.isfinite(v) for v in vals) else float("nan"))
                ys.append(row.get("future_h3_scale_residual"))
        corr_rows.append({"run_key": h35_layer_key, "metric": f"layer_{layer}_post_zp_or_tri_delta", "target": "future_h3_scale_residual", "corr": _corr(xs, ys), "n": len(sub)})
    _write_csv(phase / "layer_branch_scale_correlation.csv", corr_rows)
    reset_rows = []
    for run_key, rows in (("h35", h35_rows), ("c9", c9_rows)):
        for rel in sorted({int(r["reset_relative_idx"]) for r in rows}):
            sub = [r for r in rows if int(r["reset_relative_idx"]) == rel]
            reset_rows.append(
                {
                    "run_key": run_key,
                    "reset_relative_idx": rel,
                    "chunk_count": len(sub),
                    "future_h3_scale_residual_mean": _mean(r.get("future_h3_scale_residual") for r in sub),
                    "tri_delta_norm_mean": _mean(r.get("probe_ttt_write_tri_delta_norm_mean") for r in sub),
                    "post_zp_delta_norm_mean": _mean(r.get("probe_ttt_write_post_delta_norm_mean") for r in sub),
                    "cumulative_post_zp_delta_norm_mean": _mean(r.get("cumulative_post_zp_delta_norm_since_reset") for r in sub),
                    "cumulative_tri_delta_norm_mean": _mean(r.get("cumulative_tri_delta_norm_since_reset") for r in sub),
                    "rolling100_mean": _mean(r.get("rolling100") for r in sub),
                }
            )
    _write_csv(phase / "reset_relative_ttt_scale_profile.csv", reset_rows)
    risk_rows = []
    for reason, key in (
        ("future_scale_residual_h3", "future_h3_scale_residual"),
        ("post_tri_delta_norm", "probe_ttt_write_tri_delta_norm_mean"),
        ("ttt_state_drift_from_reset_proxy", "cumulative_tri_delta_norm_since_reset"),
    ):
        for row in sorted(h35_rows, key=lambda r: _safe_float(r.get(key)), reverse=True)[:5]:
            risk_rows.append({"selection_reason": reason, **row})
    _write_csv(phase / "top_scale_risk_chunks.csv", risk_rows)
    post_key = "probe_ttt_write_post_delta_norm_mean" if any(math.isfinite(_safe_float(r.get("probe_ttt_write_post_delta_norm_mean"))) for r in h35_rows) else "probe_ttt_write_tri_delta_norm_mean"
    post_title = "post-zp committed delta vs future scale\ndiagnostic-only" if post_key == "probe_ttt_write_post_delta_norm_mean" else "tri delta proxy vs future scale\npost-zp unavailable; diagnostic-only"
    native_cos_available = any(math.isfinite(_safe_float(r.get("probe_ttt_write_native_cosine_mean"))) for r in h35_rows)
    native_title = "candidate-native cosine vs future scale\ndiagnostic-only" if native_cos_available else "candidate-native cosine unavailable\nno-data"
    _plot_scatter(phase / "figures/post_zp_delta_vs_future_scale.png", h35_rows, post_key, "future_h3_scale_residual", post_title)
    _plot_scatter(phase / "figures/candidate_native_cosine_vs_future_scale.png", h35_rows, "probe_ttt_write_native_cosine_mean", "future_h3_scale_residual", native_title)
    _plot_scatter(phase / "figures/ttt_state_drift_vs_scale_residual.png", h35_rows, "cumulative_tri_delta_norm_since_reset", "future_h3_scale_residual", "cumulative tri delta proxy vs future scale\ndiagnostic-only")
    _plot_reset_profile(phase / "figures/reset_relative_ttt_state_drift.png", reset_rows)
    _plot_layer_heatmap(phase / "figures/layer_branch_corr_heatmap.png", corr_rows)
    max_corr = _max(abs(r["corr"]) for r in corr_rows if r.get("corr") is not None)
    max_corr_row = max(
        (r for r in corr_rows if r.get("corr") is not None),
        key=lambda r: abs(float(r["corr"])),
        default=None,
    )
    corr_drift = next((r.get("corr") for r in corr_rows if r["run_key"] == "h35" and r["metric"] == "cumulative_tri_delta_norm_since_reset"), None)
    jaccard = _top_jaccard(h35_rows, "probe_ttt_write_tri_delta_norm_mean", "future_h3_scale_residual")
    positive = bool((max_corr is not None and max_corr >= 0.35) or (corr_drift is not None and corr_drift >= 0.35) or jaccard >= 0.25)
    lines = [
        "# Phase 2 Passive TTT-Scale Correlation Report",
        "",
        (
            "Post-zp committed-delta and candidate-native cosine summaries are available where the v63 rerun logging is present; "
            "older landed runs still use tri-replay/memory proxies only."
            if any(math.isfinite(_safe_float(r.get("probe_ttt_write_post_delta_norm_mean"))) for r in h35_rows)
            else "Post-zp tensors and candidate-native cosine are unavailable in current H35 v63 trace; tri-replay delta norms and memory drift are used as recorded proxies, not as post-zp replacements."
        ),
        "",
        f"- max abs corr across available passive metrics: `{_fmt(max_corr)}`.",
        f"- max corr source: `{max_corr_row['run_key']}/{max_corr_row['metric']}` corr=`{_fmt(max_corr_row['corr'])}`." if max_corr_row else "- max corr source: `unavailable`.",
        f"- corr(cumulative tri delta since reset, future_h3_scale_residual): `{_fmt(corr_drift)}`.",
        f"- top20 Jaccard(tri_delta, future_h3_scale): `{_fmt(jaccard)}`.",
        f"- passive signal positive by v63 threshold: `{positive}`.",
    ]
    _write_text(phase / "passive_ttt_scale_correlation_report.md", lines)
    return {"corr_rows": corr_rows, "passive_positive": positive, "max_abs_corr": max_corr, "drift_corr": corr_drift, "jaccard": jaccard}


def _top_jaccard(rows: Sequence[Mapping[str, Any]], a: str, b: str) -> float:
    clean_a = [r for r in rows if math.isfinite(_safe_float(r.get(a)))]
    clean_b = [r for r in rows if math.isfinite(_safe_float(r.get(b)))]
    n = max(1, int(math.ceil(0.2 * min(len(clean_a), len(clean_b)))))
    top_a = {int(r["chunk_id"]) for r in sorted(clean_a, key=lambda r: _safe_float(r.get(a)), reverse=True)[:n]}
    top_b = {int(r["chunk_id"]) for r in sorted(clean_b, key=lambda r: _safe_float(r.get(b)), reverse=True)[:n]}
    if not top_a or not top_b:
        return 0.0
    return len(top_a & top_b) / len(top_a | top_b)


def _phase3(out_dir: Path, base: Optional[RunData], forks: Mapping[str, Optional[RunData]], metrics: Mapping[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    phase = out_dir / "phase3_ttt_causal_fork_basic"
    selection = [
        {
            "fork_chunk_id": FORK_CHUNK_ID,
            "selection_type": "oracle_diagnostic",
            "selection_uses_GT": True,
            "selection_uses_ATE": True,
            "selection_reason": "v62 H35-C9 gap TYPE_E high-risk chunk; end_frame=730 covers h5",
        }
    ]
    _write_csv(phase / "fork_chunk_selection.csv", selection)
    registry = []
    current_rows = []
    future_rows = []
    traj_rows = []
    state_rows = []
    causal_rows = []
    if base is None:
        _write_text(phase / "ttt_causal_fork_basic_report.md", ["# Phase 3 TTT Causal Fork", "", "Base fork run unavailable."])
        return {"decision": "unavailable", "rows": []}
    base_metrics = {int(r["chunk_id"]): r for r in metrics.get("F0_BASE", [])}
    for fork_type, run in forks.items():
        if run is None:
            registry.append({"fork_type": fork_type, "run_available": False})
            continue
        rows_by = {int(r["chunk_id"]): r for r in metrics.get(fork_type, [])}
        registry.append(
            {
                "fork_type": fork_type,
                "run_name": run.path.name,
                "run_path": str(run.path),
                "run_available": True,
                "frames": run.poses.count,
                "hmc_rows": len(run.hmc_rows),
                "pose_trace_rows": len(run.pose_trace_rows),
            }
        )
        cid = FORK_CHUNK_ID
        bcur = base_metrics.get(cid, {})
        ocur = rows_by.get(cid, {})
        bnext = base_metrics.get(cid + 1, {})
        onext = rows_by.get(cid + 1, {})
        pose_delta = _pose_max_abs_delta(base, run, cid)
        commit_changed = _hash_changed(bcur.get("commit_source_state_hash"), ocur.get("commit_source_state_hash"))
        next_probe_changed = _hash_changed(bnext.get("probe_committed_state_hash"), onext.get("probe_committed_state_hash"))
        next_commit_changed = _hash_changed(bnext.get("commit_source_state_hash"), onext.get("commit_source_state_hash"))
        current_rows.append(
            {
                "fork_type": fork_type,
                "current_chunk_id": cid,
                "current_output_pose_max_diff": pose_delta,
                "current_chunk_global_ate_delta": _safe_float(ocur.get("current_chunk_global_ate")) - _safe_float(bcur.get("current_chunk_global_ate")),
                "current_chunk_local_sim3_delta": _safe_float(ocur.get("current_chunk_local_sim3_ate")) - _safe_float(bcur.get("current_chunk_local_sim3_ate")),
                "current_chunk_scale_residual_delta": _safe_float(ocur.get("abs_log_scale_residual")) - _safe_float(bcur.get("abs_log_scale_residual")),
            }
        )
        state_rows.append(
            {
                "fork_type": fork_type,
                "chunk_id": cid,
                "base_probe_ttt_write_state_hash": bcur.get("probe_ttt_write_state_hash"),
                "fork_probe_ttt_write_state_hash": ocur.get("probe_ttt_write_state_hash"),
                "fork_frozen_ttt_state_hash": ocur.get("frozen_ttt_state_hash"),
                "fork_scaled_ttt_state_hash": ocur.get("ttt_semantic_scaled_state_hash"),
                "base_commit_state_hash": bcur.get("commit_source_state_hash"),
                "fork_commit_state_hash": ocur.get("commit_source_state_hash"),
                "commit_state_hash_changed": commit_changed,
                "base_next_probe_state_hash": bnext.get("probe_committed_state_hash"),
                "fork_next_probe_state_hash": onext.get("probe_committed_state_hash"),
                "next_probe_state_hash_changed": next_probe_changed,
                "base_next_commit_state_hash": bnext.get("commit_source_state_hash"),
                "fork_next_commit_state_hash": onext.get("commit_source_state_hash"),
                "next_commit_state_hash_changed": next_commit_changed,
                "ttt_state_hash_changed": _state_changed(bcur, ocur, fallback=commit_changed),
                "post_zp_delta_norm_changed": None,
                "commit_delta_norm": ocur.get("memory_ttt_mean_rel_diff"),
            }
        )
        for h in (1, 3, 5):
            base_scale = _safe_float(bcur.get(f"future_h{h}_scale_residual"))
            fork_scale = _safe_float(ocur.get(f"future_h{h}_scale_residual"))
            scale_delta = fork_scale - base_scale
            percent = scale_delta / base_scale if math.isfinite(base_scale) and abs(base_scale) > EPS else float("nan")
            frow = {
                "fork_type": fork_type,
                "fork_chunk_id": cid,
                "horizon": h,
                "future_scale_residual_base": base_scale,
                "future_scale_residual_fork": fork_scale,
                "future_scale_residual_delta": scale_delta,
                "future_scale_residual_delta_percent": percent,
                "future_rolling100_delta": _safe_float(ocur.get(f"future_h{h}_rolling100")) - _safe_float(bcur.get(f"future_h{h}_rolling100")),
                "future_global_ate_delta": _safe_float(ocur.get(f"future_h{h}_global_ate")) - _safe_float(bcur.get(f"future_h{h}_global_ate")),
            }
            future_rows.append(frow)
            causal_rows.append(
                {
                    "fork_chunk_id": cid,
                    "fork_type": fork_type,
                    "selection_type": "oracle_diagnostic",
                    "selection_uses_GT": True,
                    "selection_uses_ATE": True,
                    "horizon": h,
                    "current_output_delta": pose_delta,
                    "future_scale_delta": scale_delta,
                    "future_scale_delta_percent": percent,
                    "future_rolling100_delta": frow["future_rolling100_delta"],
                    "future_global_ate_delta": frow["future_global_ate_delta"],
                    "future_local_sim3_delta": None,
                    "future_head_to_tail_ratio_delta": None,
                    "post_zp_delta_change": None,
                    "effect_per_delta_norm": None,
                    "causal_decision": _causal_label(percent, frow["future_rolling100_delta"]),
                }
            )
        for j in range(0, 6):
            b = base_metrics.get(cid + j, {})
            o = rows_by.get(cid + j, {})
            traj_rows.append(
                {
                    "fork_type": fork_type,
                    "chunk_id": cid + j,
                    "global_ate_base": b.get("current_chunk_global_ate"),
                    "global_ate_fork": o.get("current_chunk_global_ate"),
                    "global_ate_delta": _safe_float(o.get("current_chunk_global_ate")) - _safe_float(b.get("current_chunk_global_ate")),
                    "abs_log_scale_base": b.get("abs_log_scale_residual"),
                    "abs_log_scale_fork": o.get("abs_log_scale_residual"),
                    "abs_log_scale_delta": _safe_float(o.get("abs_log_scale_residual")) - _safe_float(b.get("abs_log_scale_residual")),
                }
            )
    _write_csv(phase / "fork_run_registry.csv", registry)
    _write_csv(phase / "fork_current_chunk_delta.csv", current_rows)
    _write_csv(phase / "fork_future_scale_delta.csv", future_rows)
    _write_csv(phase / "fork_future_trajectory_delta.csv", traj_rows)
    _write_csv(phase / "fork_state_delta.csv", state_rows)
    _write_csv(out_dir / "causal_effect_metrics.csv", causal_rows)
    h3 = [r for r in future_rows if int(r["horizon"]) == 3 and r["fork_type"] != "F0_BASE"]
    max_abs_pct = _max(abs(r.get("future_scale_residual_delta_percent")) for r in h3)
    all_lt_3 = all(abs(_safe_float(r.get("future_scale_residual_delta_percent"))) < 0.03 for r in h3)
    any_ge_10 = any(abs(_safe_float(r.get("future_scale_residual_delta_percent"))) >= 0.10 for r in h3)
    decision = "TTT_SCALE_CAUSAL_NEGATIVE_WEAK_OR_NONE" if all_lt_3 else ("TTT_SCALE_CAUSAL_POSITIVE" if any_ge_10 else "TTT_SCALE_CAUSAL_WEAK")
    _plot_fork_bar(phase / "figures/fork_future_scale_delta_bar.png", future_rows)
    _plot_current_future(phase / "figures/fork_current_vs_future_delta.png", current_rows, future_rows)
    _plot_strength_curve(phase / "figures/fork_delta_strength_curve.png", future_rows)
    _plot_fork_by_reset(phase / "figures/fork_by_reset_relative_idx.png", future_rows)
    lines = [
        "# Phase 3 TTT Causal Fork Basic Report",
        "",
        f"Fork chunk: `{FORK_CHUNK_ID}`. Selection is oracle/diagnostic and uses v62 GT-derived scale/gap evidence.",
        f"Decision: `{decision}`.",
        f"Max abs h3 future scale residual percent change across non-base forks: `{_fmt(max_abs_pct)}`.",
        "",
        "| fork | h3 base | h3 fork | delta | delta % | rolling100 delta | current pose max diff | commit changed | next probe changed |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    cur_by = {r["fork_type"]: r for r in current_rows}
    st_by = {r["fork_type"]: r for r in state_rows}
    for row in h3:
        cur = cur_by.get(row["fork_type"], {})
        st = st_by.get(row["fork_type"], {})
        lines.append(
            f"| `{row['fork_type']}` | {_fmt(row.get('future_scale_residual_base'))} | {_fmt(row.get('future_scale_residual_fork'))} | "
            f"{_fmt(row.get('future_scale_residual_delta'))} | {_fmt(row.get('future_scale_residual_delta_percent'))} | "
            f"{_fmt(row.get('future_rolling100_delta'))} | {_fmt(cur.get('current_output_pose_max_diff'))} | "
            f"`{st.get('commit_state_hash_changed')}` | `{st.get('next_probe_state_hash_changed')}` |"
        )
    _write_text(phase / "ttt_causal_fork_basic_report.md", lines)
    return {"decision": decision, "rows": future_rows, "max_abs_pct": max_abs_pct}


def _state_changed(base: Mapping[str, Any], other: Mapping[str, Any], fallback: Optional[bool] = None) -> Optional[bool]:
    base_hash = base.get("probe_ttt_write_state_hash")
    for key in ("probe_ttt_write_state_hash", "frozen_ttt_state_hash", "ttt_semantic_scaled_state_hash"):
        val = other.get(key)
        if val:
            return val != base_hash
    return fallback


def _hash_changed(base_hash: Any, other_hash: Any) -> Optional[bool]:
    if not base_hash or not other_hash:
        return None
    return str(base_hash) != str(other_hash)


def _causal_label(percent: Any, rolling_delta: Any) -> str:
    pct = _safe_float(percent)
    roll = _safe_float(rolling_delta)
    if not math.isfinite(pct):
        return "UNAVAILABLE"
    if abs(pct) < 0.03:
        return "WEAK_OR_NONE_LT_3PCT"
    if abs(pct) >= 0.10:
        if math.isfinite(roll) and roll > 0.5:
            return "SCALE_CHANGED_WITH_ROLLING_REGRESSION"
        return "CAUSAL_POSITIVE_GE_10PCT"
    return "WEAK_3_TO_10PCT"


def _phase4_5_7_no_data(out_dir: Path, phase3: Mapping[str, Any], passive: Mapping[str, Any]) -> None:
    phase4 = out_dir / "phase4_layer_branch_attribution"
    _write_csv(phase4 / "layer_branch_fork_registry.csv", [])
    _write_csv(phase4 / "layer_effect_table.csv", [])
    _write_csv(phase4 / "branch_effect_table.csv", [])
    _write_csv(phase4 / "layer_branch_effect_per_delta_norm.csv", [])
    _plot_no_data(phase4 / "figures/layer_effect_on_future_scale.png", "layer fork not run", "Phase 4 requires positive Phase 3 causal effect and layer intervention hooks.")
    _plot_no_data(phase4 / "figures/branch_effect_on_future_scale.png", "branch fork not run", "No branch-specific commit intervention hook in this v63 run.")
    _plot_no_data(phase4 / "figures/layer_branch_effect_heatmap.png", "layer/branch causal no-data", "Passive layer/branch proxies are in Phase 2; causal attribution unavailable.")
    _write_text(
        phase4 / "layer_branch_attribution_report.md",
        [
            "# Phase 4 Layer / Branch Attribution",
            "",
            "No layer/branch causal fork was run. The current runner exposes chunk-level freeze and semantic-delta scaling, but not layer/branch selective commit intervention.",
            f"Phase 3 decision was `{phase3.get('decision')}`; passive max abs corr was `{_fmt(passive.get('max_abs_corr'))}`.",
        ],
    )
    phase5 = out_dir / "phase5_c9_component_scale_attribution"
    _write_csv(phase5 / "h35_plus_c9_component_results.csv", [])
    _write_csv(phase5 / "c9_minus_component_results.csv", [])
    _write_csv(phase5 / "component_scale_effect_table.csv", [])
    _write_csv(phase5 / "component_taxonomy_shift.csv", [])
    _plot_no_data(phase5 / "figures/h35_plus_c9_component_scale_bar.png", "C9 component forks not run", "No safe component injection matrix completed in this run.")
    _plot_no_data(phase5 / "figures/c9_minus_component_scale_bar.png", "C9 ablation forks not run", "No C9 ablation matrix completed in this run.")
    _plot_no_data(phase5 / "figures/component_taxonomy_shift.png", "component taxonomy no-data", "No component fork outputs.")
    _write_text(
        phase5 / "c9_component_scale_attribution_report.md",
        [
            "# Phase 5 C9 Component Scale Attribution",
            "",
            "C9 component injection/ablation was not executed. This is recorded as no-data; no C9 TTT component is credited with a causal scale effect.",
            "Available evidence is limited to config audit and passive C9/H35 comparisons.",
        ],
    )
    phase7 = out_dir / "phase7_ttt_vs_merge_gauge"
    _write_csv(
        phase7 / "ttt_vs_merge_gauge_results.csv",
        [
            {
                "path": "MEMORY_ONLY_TTT_CHANGE",
                "evidence": phase3.get("decision"),
                "merge_gauge_intervention_run": False,
                "decision": "insufficient_for_merge_gauge_causal_claim",
            }
        ],
    )
    _plot_no_data(phase7 / "figures/output_memory_merge_comparison.png", "merge/gauge fork unavailable", "Only TTT memory fork was executed.")
    _plot_no_data(phase7 / "figures/future_scale_persistence_by_path.png", "path comparison unavailable", "Output-only/merge-gauge-only forks were not run.")
    _write_text(
        phase7 / "ttt_vs_merge_gauge_decision.md",
        [
            "# Phase 7 TTT vs Merge/Gauge Decision",
            "",
            f"TTT memory fork decision: `{phase3.get('decision')}`.",
            "No output-only or merge-gauge-only intervention was executed, so the report cannot make a direct causal merge/gauge claim.",
            "If TTT fork effect is weak, the next diagnostic target should be explicit merge/gauge state intervention.",
        ],
    )


def _phase6(out_dir: Path, rows: Sequence[Dict[str, Any]], phase3: Mapping[str, Any]) -> Dict[str, Any]:
    phase = out_dir / "phase6_reset_group_ttt_accumulation"
    state_rows = []
    for row in rows:
        state_rows.append(
            {
                "reset_group_id": row.get("reset_group_id"),
                "reset_relative_idx": row.get("reset_relative_idx"),
                "chunk_id": row.get("chunk_id"),
                "cumulative_post_zp_delta_norm_since_reset": row.get("cumulative_post_zp_delta_norm_since_reset"),
                "cumulative_commit_delta_norm_since_reset": row.get("cumulative_tri_delta_norm_since_reset"),
                "ttt_state_drift_from_reset": row.get("cumulative_tri_delta_norm_since_reset"),
                "candidate_native_cosine": row.get("probe_ttt_write_native_cosine_mean"),
                "future_scale_residual": row.get("future_h3_scale_residual"),
                "rolling100_error": row.get("rolling100"),
                "head_to_tail_ratio": None,
            }
        )
    _write_csv(phase / "reset_group_ttt_state_table.csv", state_rows)
    rel_rows = []
    for rel in sorted({int(r["reset_relative_idx"]) for r in rows}):
        sub = [r for r in rows if int(r["reset_relative_idx"]) == rel]
        rel_rows.append(
            {
                "reset_relative_idx": rel,
                "chunk_count": len(sub),
                "future_scale_residual_mean": _mean(r.get("future_h3_scale_residual") for r in sub),
                "cumulative_post_zp_delta_norm_mean": _mean(r.get("cumulative_post_zp_delta_norm_since_reset") for r in sub),
                "ttt_state_drift_from_reset_mean": _mean(r.get("cumulative_tri_delta_norm_since_reset") for r in sub),
                "rolling100_mean": _mean(r.get("rolling100") for r in sub),
            }
        )
    _write_csv(phase / "reset_relative_ttt_accumulation.csv", rel_rows)
    _write_csv(phase / "reset_late_fork_results.csv", [r for r in phase3.get("rows", []) if r.get("fork_type") != "F0_BASE"])
    corr_cum = _corr((r.get("cumulative_tri_delta_norm_since_reset") for r in rows), (r.get("future_h3_scale_residual") for r in rows))
    corr_rel = _corr((r.get("reset_relative_idx") for r in rows), (r.get("future_h3_scale_residual") for r in rows))
    positive = bool(corr_cum is not None and corr_cum >= 0.35)
    _plot_reset_profile(phase / "figures/cumulative_ttt_delta_by_reset_relative.png", [{"run_key": "h35", **r, "cumulative_tri_delta_norm_mean": r.get("ttt_state_drift_from_reset_mean")} for r in rel_rows])
    _plot_scatter(phase / "figures/ttt_state_drift_vs_future_scale.png", rows, "cumulative_tri_delta_norm_since_reset", "future_h3_scale_residual", "reset cumulative tri delta proxy vs future scale")
    _plot_fork_bar(phase / "figures/reset_late_fork_scale_delta.png", phase3.get("rows", []))
    _write_text(
        phase / "reset_group_ttt_accumulation_report.md",
        [
            "# Phase 6 Reset-Group TTT Accumulation Diagnostic",
            "",
            "Cumulative post-zp norm is available where v63 trace logging is present; cumulative tri delta norm remains a role-specific proxy.",
            f"- corr(cumulative tri delta since reset, future_h3_scale_residual): `{_fmt(corr_cum)}`.",
            f"- corr(reset_relative_idx, future_h3_scale_residual): `{_fmt(corr_rel)}`.",
            f"- reset accumulation positive by threshold: `{positive}`.",
        ],
    )
    return {"corr_cum": corr_cum, "corr_rel": corr_rel, "positive": positive}


def _write_unified_and_final(
    out_dir: Path,
    runs: Mapping[str, RunData],
    metrics: Mapping[str, List[Dict[str, Any]]],
    phase1: Mapping[str, Any],
    passive_sources: Mapping[str, str],
    passive: Mapping[str, Any],
    phase3: Mapping[str, Any],
    phase6: Mapping[str, Any],
) -> None:
    traj_rows = []
    ttt_rows = []
    for key, rows in metrics.items():
        for row in rows:
            traj_rows.append(
                {
                    "run_name": row.get("run_name"),
                    "method_family": key,
                    "chunk_id": row.get("chunk_id"),
                    "frame_start": row.get("frame_start"),
                    "frame_end": row.get("frame_end"),
                    "reset_group_id": row.get("reset_group_id"),
                    "reset_relative_idx": row.get("reset_relative_idx"),
                    "current_chunk_global_ate": row.get("current_chunk_global_ate"),
                    "current_chunk_local_sim3_ate": row.get("current_chunk_local_sim3_ate"),
                    "future_h1_global_ate": row.get("future_h1_global_ate"),
                    "future_h3_global_ate": row.get("future_h3_global_ate"),
                    "future_h5_global_ate": row.get("future_h5_global_ate"),
                    "future_h1_scale_residual": row.get("future_h1_scale_residual"),
                    "future_h3_scale_residual": row.get("future_h3_scale_residual"),
                    "future_h5_scale_residual": row.get("future_h5_scale_residual"),
                    "future_h1_step_length_ratio": None,
                    "future_h3_step_length_ratio": None,
                    "future_h5_step_length_ratio": None,
                    "head_to_tail_transfer_ratio": None,
                    "rolling50": None,
                    "rolling100": row.get("rolling100"),
                    "rolling200": None,
                    "FinalErr": None,
                    "Rot": None,
                }
            )
            ttt_rows.append(
                {
                    "run_name": row.get("run_name"),
                    "chunk_id": row.get("chunk_id"),
                    "fork_type": key,
                    "post_zp_delta_norm_total": row.get("probe_ttt_write_post_delta_norm_mean"),
                    "post_zp_delta_norm_layer0": row.get("post_zp_delta_norm_layer0"),
                    "post_zp_delta_norm_layer8": row.get("post_zp_delta_norm_layer8"),
                    "post_zp_delta_norm_layer17": row.get("post_zp_delta_norm_layer17"),
                    "post_zp_delta_norm_w0": row.get("post_zp_delta_norm_w0"),
                    "post_zp_delta_norm_w1": row.get("post_zp_delta_norm_w1"),
                    "post_zp_delta_norm_w2": row.get("post_zp_delta_norm_w2"),
                    "post_zp_action_delta_norm_w0": row.get("post_zp_action_delta_norm_w0"),
                    "post_zp_action_delta_norm_w1": row.get("post_zp_action_delta_norm_w1"),
                    "post_zp_action_delta_norm_w2": row.get("post_zp_action_delta_norm_w2"),
                    "positive_mass": row.get("probe_ttt_write_tri_pos_mass_mean"),
                    "neutral_mass": row.get("probe_ttt_write_tri_neu_mass_mean"),
                    "negative_mass": row.get("probe_ttt_write_tri_neg_mass_mean"),
                    "candidate_native_cosine": row.get("probe_ttt_write_native_cosine_mean"),
                    "candidate_native_cosine_w0": row.get("candidate_native_cosine_w0"),
                    "candidate_native_cosine_w1": row.get("candidate_native_cosine_w1"),
                    "candidate_native_cosine_w2": row.get("candidate_native_cosine_w2"),
                    "commit_delta_norm": row.get("memory_ttt_mean_rel_diff"),
                    "commit_ema_strength": None,
                    "native_mix_strength": None,
                    "gamma_effective": row.get("gamma_effective_mean"),
                    "tri_replay_params": {
                        "positive_mass": row.get("probe_ttt_write_tri_pos_mass_mean"),
                        "neutral_mass": row.get("probe_ttt_write_tri_neu_mass_mean"),
                        "negative_mass": row.get("probe_ttt_write_tri_neg_mass_mean"),
                        "rho_effective_mean": row.get("rho_effective_mean"),
                        "configured_rho_mean": row.get("configured_rho_mean"),
                        "layer_branch_row_count": row.get("layer_branch_row_count"),
                    },
                    "ttt_state_hash_before": row.get("commit_source_state_hash"),
                    "ttt_state_hash_after": row.get("probe_ttt_write_state_hash"),
                    "ttt_state_drift_from_reset": row.get("cumulative_tri_delta_norm_since_reset"),
                    "post_zp_state_drift_from_reset": row.get("cumulative_post_zp_delta_norm_since_reset"),
                }
            )
    _write_csv(out_dir / "trajectory_scale_metrics.csv", traj_rows)
    _write_csv(out_dir / "ttt_update_metrics.csv", ttt_rows)
    _write_json(
        out_dir / "v63_summary.json",
        {
            "passive": passive,
            "phase1": phase1,
            "passive_sources": dict(passive_sources),
            "phase3": phase3,
            "phase6": phase6,
            "loaded_runs": {k: str(v.path) for k, v in runs.items()},
        },
    )
    _plot_no_data(out_dir / "figures/chunk_selection_map.png", "chunk selection map", "Selected chunk 19 from v62 H35-C9 TYPE_E high-risk evidence.")
    _plot_fork_bar(out_dir / "figures/future_scale_delta_by_fork.png", phase3.get("rows", []))
    _plot_current_future(out_dir / "figures/current_vs_future_delta_scatter.png", [], phase3.get("rows", []))
    _plot_strength_curve(out_dir / "figures/ttt_delta_strength_curve.png", phase3.get("rows", []))
    _plot_no_data(out_dir / "figures/layer_branch_scale_effect_heatmap.png", "layer/branch causal no-data", "No layer/branch fork completed.")
    _plot_no_data(out_dir / "figures/c9_component_scale_effect_bar.png", "C9 component no-data", "No C9 component fork completed.")
    _plot_no_data(out_dir / "figures/output_vs_memory_vs_merge_comparison.png", "merge/gauge no-data", "Only TTT memory fork completed.")
    h35_trace_rows = metrics.get("h35_full_trace") or metrics.get("F0_BASE", [])
    post_available = any(math.isfinite(_safe_float(r.get("probe_ttt_write_post_delta_norm_mean"))) for r in h35_trace_rows)
    native_cos_available = any(math.isfinite(_safe_float(r.get("probe_ttt_write_native_cosine_mean"))) for r in h35_trace_rows)
    passive_source = (
        f"H35 `{passive_sources.get('h35')}` "
        f"({'post-zp/native summaries' if post_available else 'tri-delta proxy'}); "
        f"C9 `{passive_sources.get('c9')}` "
        f"({'post-zp/native summaries' if passive_sources.get('c9') == 'c9_full_trace' else 'landed tri/memory proxies because full-trace parity failed'})"
    )
    gates = phase1.get("gate_by_label", {}) if isinstance(phase1, Mapping) else {}
    c9_full_gate = gates.get("c9_full_trace")
    h35_full_gate = gates.get("h35_full_trace")
    lines = [
        "# ACL2 v63 TTT Scale Causal Diagnostic Report",
        "",
        "This report is diagnostic-only. Missing point overlap, C9 component forks, and merge/gauge forks are explicitly marked unavailable/no-data.",
        "",
        "## Required Answers",
        "",
        f"1. Changing chunk `{FORK_CHUNK_ID}` TTT write: Phase 3 decision `{phase3.get('decision')}`; see `phase3_ttt_causal_fork_basic/fork_future_scale_delta.csv`.",
        f"2. Strength of TTT future-scale effect: max abs h3 percent change `{_fmt(phase3.get('max_abs_pct'))}`.",
        f"3. Layer/branch: causal attribution unavailable; passive max abs corr `{_fmt(passive.get('max_abs_corr'))}` from {passive_source}.",
        "4. C9 gamma/tri replay/commit EMA attribution: no component fork completed; no component is credited with causal scale effect.",
        f"5. Reset rel=3/4 accumulation: corr cumulative tri-delta proxy vs future scale `{_fmt(phase6.get('corr_cum'))}`, threshold-positive `{phase6.get('positive')}`.",
        "6. Merge/gauge: no direct merge/gauge fork completed; weak/negative TTT evidence would support prioritizing this next, but direct causal proof is not claimed.",
        "7. Scale improves but ATE worsens: see fork tables for rolling/global ATE deltas; no deployable policy claim is made.",
        "8. Next target is decided from the causal fork plus missing-data boundaries in the retrospective log.",
        "",
        "## Key Boundaries",
        "",
        f"- `post_zp_delta_norm_mean` available in H35 v63 full trace: `{post_available}`.",
        f"- `candidate_native_cosine_mean` available in H35 v63 full trace: `{native_cos_available}`.",
        f"- H35 full-trace parity gate: `{h35_full_gate}`; passive H35 source: `{passive_sources.get('h35')}`.",
        f"- C9 full-trace parity gate: `{c9_full_gate}`; passive C9 source: `{passive_sources.get('c9')}`.",
        "- duplicate overlap raw pose is available from lightweight pose trace; point overlap is unavailable.",
        "- C9 component and merge/gauge interventions are no-data in this run.",
    ]
    _write_text(out_dir / "v63_ttt_scale_causal_diagnostic_report.md", lines)


def _plot_scatter(path: Path, rows: Sequence[Mapping[str, Any]], xkey: str, ykey: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xs = [_safe_float(r.get(xkey)) for r in rows]
    ys = [_safe_float(r.get(ykey)) for r in rows]
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    fig, ax = plt.subplots(figsize=(7, 4))
    if pairs:
        arr = np.asarray(pairs)
        ax.scatter(arr[:, 0], arr[:, 1], s=28)
        ax.set_xlabel(xkey)
        ax.set_ylabel(ykey)
    else:
        ax.text(0.5, 0.5, "no-data / unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_reset_profile(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for run_key in sorted({str(r.get("run_key", "h35")) for r in rows}):
        sub = [r for r in rows if str(r.get("run_key", "h35")) == run_key]
        xs = [int(r["reset_relative_idx"]) for r in sub]
        ys = [_safe_float(r.get("cumulative_tri_delta_norm_mean", r.get("ttt_state_drift_from_reset_mean"))) for r in sub]
        if xs and any(math.isfinite(y) for y in ys):
            ax.plot(xs, ys, marker="o", label=run_key)
    if not ax.lines:
        ax.text(0.5, 0.5, "no-data / unavailable", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("reset_relative_idx")
    ax.set_ylabel("cumulative tri delta proxy")
    ax.set_title("reset-relative TTT state drift proxy\npost-zp unavailable; diagnostic-only")
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_layer_heatmap(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    layer_rows = [r for r in rows if str(r.get("metric", "")).startswith("layer_")]
    fig, ax = plt.subplots(figsize=(8, 3))
    if layer_rows:
        layers = [int(str(r["metric"]).split("_")[1]) for r in layer_rows]
        vals = [_safe_float(r.get("corr")) for r in layer_rows]
        ax.bar([str(x) for x in layers], vals)
        ax.set_xlabel("layer")
        ax.set_ylabel("corr")
    else:
        ax.text(0.5, 0.5, "no-data / unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    ax.set_title("layer tri-delta proxy correlation with future scale")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_fork_bar(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sub = [r for r in rows if int(_safe_float(r.get("horizon"))) == 3 and r.get("fork_type") != "F0_BASE"]
    fig, ax = plt.subplots(figsize=(8, 4))
    if sub:
        ax.bar([str(r["fork_type"]).replace("_", "\n") for r in sub], [_safe_float(r.get("future_scale_residual_delta_percent")) for r in sub])
        ax.axhline(0.10, color="r", linestyle="--", linewidth=1)
        ax.axhline(-0.10, color="r", linestyle="--", linewidth=1)
        ax.set_ylabel("h3 future scale delta percent")
    else:
        ax.text(0.5, 0.5, "no-data / unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    ax.set_title("diagnostic-only future scale delta by fork")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_current_future(path: Path, current_rows: Sequence[Mapping[str, Any]], future_rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cur_by = {r.get("fork_type"): r for r in current_rows}
    sub = [r for r in future_rows if int(_safe_float(r.get("horizon"))) == 3 and r.get("fork_type") != "F0_BASE"]
    fig, ax = plt.subplots(figsize=(6, 4))
    pairs = []
    for row in sub:
        cur = cur_by.get(row.get("fork_type"), {})
        pairs.append((_safe_float(cur.get("current_output_pose_max_diff")), _safe_float(row.get("future_scale_residual_delta_percent"))))
    pairs = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
    if pairs:
        arr = np.asarray(pairs)
        ax.scatter(arr[:, 0], arr[:, 1])
        ax.set_xlabel("current pose max abs diff")
        ax.set_ylabel("future h3 scale delta percent")
    else:
        ax.text(0.5, 0.5, "no-data / unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    ax.set_title("current-vs-future fork delta")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_strength_curve(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    order = {
        "F1_NO_TTT_WRITE": 0.0,
        "F2_NATIVE_COMMIT_ONLY": 0.0,
        "F4_HALF_DELTA_COMMIT": 0.5,
        "F0_BASE": 1.0,
        "F5_DOUBLE_DELTA_COMMIT": 2.0,
    }
    sub = [r for r in rows if int(_safe_float(r.get("horizon"))) == 3 and r.get("fork_type") in order]
    fig, ax = plt.subplots(figsize=(7, 4))
    if sub:
        sub = sorted(sub, key=lambda r: order[str(r["fork_type"])])
        ax.plot([order[str(r["fork_type"])] for r in sub], [_safe_float(r.get("future_scale_residual_fork")) for r in sub], marker="o")
        ax.set_xlabel("semantic TTT delta scale (freeze/native both shown at 0)")
        ax.set_ylabel("future h3 scale residual")
    else:
        ax.text(0.5, 0.5, "no-data / unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    ax.set_title("TTT delta strength curve, diagnostic-only")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_fork_by_reset(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _plot_fork_bar(path, rows)


def _plot_no_data(path: Path, title: str, subtitle: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.text(0.5, 0.55, "no-data / unavailable", ha="center", va="center", fontsize=14, transform=ax.transAxes)
    ax.text(0.5, 0.42, subtitle, ha="center", va="center", fontsize=9, transform=ax.transAxes)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--gt-poses", type=Path, default=GT_PATH)
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_all = _read_poses(args.gt_poses)
    runs: Dict[str, RunData] = {}
    for key, display, path in [
        ("h35_landed", "Clean_H35_v53_landed", H35_LANDED),
        ("c9_landed", "C9_P0_R2_landed", C9_LANDED),
        ("h35_smoke_e96", "V63_H35_trace_smoke_E96", SMOKE_RUN),
        ("h35_patch_smoke_e96", "V63_H35_postzp_log_smoke_E96", PATCH_SMOKE_RUN),
        ("h35_full_trace", "V63_H35_postzp_trace_full", H35_FULL_TRACE),
        ("c9_full_trace", "V63_C9_postzp_trace_full", C9_FULL_TRACE),
    ]:
        run = _load_run(key, display, path, gt_all)
        if run is not None:
            runs[key] = run
    for fork_type, name in FORK_RUNS.items():
        path = FORK_BASE / name
        run = _load_run(fork_type, fork_type, path, gt_all)
        if run is not None:
            runs[fork_type] = run
    _phase0(out_dir, list(runs.values()))
    base = runs.get("F0_BASE")
    phase1 = _phase1(
        out_dir,
        runs["h35_landed"],
        runs["c9_landed"],
        smoke=runs.get("h35_smoke_e96"),
        base=base,
        h35_full=runs.get("h35_full_trace"),
        c9_full=runs.get("c9_full_trace"),
    )
    metrics = {key: _chunk_metrics(run) for key, run in runs.items()}
    layer_rows = []
    for key, run in runs.items():
        layer_rows.extend(_layer_branch_rows(run))
    gates = phase1.get("gate_by_label", {})
    h35_source = "h35_full_trace" if gates.get("h35_full_trace") and metrics.get("h35_full_trace") else "F0_BASE"
    if h35_source == "F0_BASE" and not metrics.get("F0_BASE"):
        h35_source = "h35_landed"
    c9_source = "c9_full_trace" if gates.get("c9_full_trace") and metrics.get("c9_full_trace") else "c9_landed"
    h35_passive = metrics.get(h35_source, [])
    c9_passive = metrics.get(c9_source, [])
    passive_sources = {"h35": h35_source, "c9": c9_source}
    passive = _phase2(out_dir, h35_passive, c9_passive, layer_rows)
    forks = {key: runs.get(key) for key in FORK_RUNS}
    phase3 = _phase3(out_dir, base, forks, metrics)
    phase6 = _phase6(out_dir, h35_passive, phase3)
    _phase4_5_7_no_data(out_dir, phase3, passive)
    _write_unified_and_final(out_dir, runs, metrics, phase1, passive_sources, passive, phase3, phase6)
    print(json.dumps(_clean({"out_dir": str(out_dir), "phase3": phase3, "passive": passive, "phase6": phase6}), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
