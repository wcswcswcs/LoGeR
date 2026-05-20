#!/usr/bin/env python3
"""v12 registry, projection-action autopsy, and offline selector audit.

This tool is intentionally read-only with respect to experiment runs.  It uses
existing v11 artifacts to decide whether v12 gates allow new full runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.kitti_trajectory_diagnostics import (  # noqa: E402
    _angle_diff_deg,
    _apply_alignment,
    _load_kitti_gt,
    _load_tum_prediction,
    _rmse,
    _umeyama_sim3,
    _yaw_from_pose,
)


DEFAULT_GT = "/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt"
DEFAULT_V11_BASE = "results/kitti01_hmc_v2/acl2_v11_no_postprocess_ttt_scaleaware"
DEFAULT_V12_BASE = "results/kitti01_hmc_v2/acl2_v12_ttt_write_windowscale_mpc_target25"

CHUNK_SIZE = 32
CHUNK_OVERLAP = 3
CHUNK_STRIDE = CHUNK_SIZE - CHUNK_OVERLAP
EPS = 1e-12

SEGMENTS = (
    (0, 100),
    (100, 200),
    (200, 300),
    (200, 400),
    (300, 400),
    (400, 500),
    (400, 600),
    (600, 800),
)


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    path: Path
    family: str
    output_from_online_hmc: bool
    no_postprocess: bool
    counts_as_ttt_write: bool
    uses_gt_runtime_action: bool
    uses_offline_trajectory_rewrite: bool
    gate_status: str
    include_phase1: bool = False
    include_selector: bool = False


@dataclass
class TrajectoryStats:
    run_id: str
    frames: np.ndarray
    raw_poses: np.ndarray
    raw_pos: np.ndarray
    aligned_poses: np.ndarray
    aligned_pos: np.ndarray
    gt_poses: np.ndarray
    gt_pos: np.ndarray
    err_norm: np.ndarray
    yaw_err: np.ndarray
    sim3_scale: float
    global_ate: float
    final_err: float
    yaw_rmse: float
    chunk_rows: Dict[int, Dict[str, float]]
    segment_rows: Dict[Tuple[int, int], Dict[str, float]]


def _nan() -> float:
    return float("nan")


def _is_finite(x: object) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _float_or_nan(x: object) -> float:
    try:
        val = float(x)
    except (TypeError, ValueError):
        return _nan()
    return val if math.isfinite(val) else _nan()


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        keys: List[str] = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fields = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _chunks(n_frames: int) -> List[Tuple[int, int, int]]:
    chunks = []
    idx = 0
    for start in range(0, n_frames, CHUNK_STRIDE):
        end = min(start + CHUNK_SIZE, n_frames)
        if end - start < max(5, CHUNK_SIZE // 2):
            break
        chunks.append((idx, start, end))
        idx += 1
        if end >= n_frames:
            break
    return chunks


def _parse_kitti_log(path: Path) -> Dict[str, float]:
    out = {"rpe_t": _nan(), "rpe_r": _nan(), "ate_log": _nan(), "rot_log": _nan()}
    if not path.exists():
        return out
    mode = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("RPE stats"):
            mode = "rpe"
            continue
        if line.startswith("ATE RMSE stats"):
            mode = "ate"
            continue
        match = re.match(r"^01\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)", line.strip())
        if not match:
            continue
        a, b = float(match.group(1)), float(match.group(2))
        if mode == "rpe":
            out["rpe_t"], out["rpe_r"] = a, b
        elif mode == "ate":
            out["ate_log"], out["rot_log"] = a, b
    return out


def _hmc_state_summary(run_dir: Path) -> Dict[str, object]:
    path = run_dir / "hmc_state_hash.jsonl"
    out: Dict[str, object] = {
        "hmc_commit_mode": "",
        "state_changed_count": 0,
        "hmc_state_rows": 0,
        "commit_mode_ok": False,
        "probe_no_commit_hash_equal_all": "",
    }
    rows = _read_jsonl(path)
    if not rows:
        return out
    modes = [str(row.get("hmc_commit_mode", "")) for row in rows]
    changed = 0
    probe_equal: List[bool] = []
    for row in rows:
        before = row.get("controlled_input_state_hash")
        after = row.get("hash_H_m_after_commit")
        if before is not None and after is not None and before != after:
            changed += 1
        if row.get("probe_no_commit_hash_equal") is not None:
            probe_equal.append(bool(row.get("probe_no_commit_hash_equal")))
    out["hmc_commit_mode"] = modes[0] if modes else ""
    out["state_changed_count"] = changed
    out["hmc_state_rows"] = len(rows)
    out["commit_mode_ok"] = bool(modes) and all(mode == "probe_ttt_write" for mode in modes)
    out["probe_no_commit_hash_equal_all"] = all(probe_equal) if probe_equal else ""
    return out


def _step_lengths(pos: np.ndarray) -> np.ndarray:
    if pos.shape[0] < 2:
        return np.asarray([], dtype=np.float64)
    return np.linalg.norm(np.diff(pos, axis=0), axis=1)


def _chunk_step_log_gt_over_pred(
    frames: np.ndarray,
    raw_pos: np.ndarray,
    gt_pos_by_frame: np.ndarray,
    start: int,
    end: int,
) -> float:
    mask = (frames >= start) & (frames < end)
    idx = np.nonzero(mask)[0]
    if idx.size < 3:
        return _nan()
    pred_steps = _step_lengths(raw_pos[idx])
    gt_steps = _step_lengths(gt_pos_by_frame[frames[idx]])
    pred_med = float(np.nanmedian(pred_steps)) if pred_steps.size else _nan()
    gt_med = float(np.nanmedian(gt_steps)) if gt_steps.size else _nan()
    if not (_is_finite(pred_med) and _is_finite(gt_med)) or pred_med <= 0.0 or gt_med <= 0.0:
        return _nan()
    return float(math.log(gt_med / pred_med))


def _chunk_yaw_delta(raw_poses: np.ndarray, indices: np.ndarray) -> float:
    if indices.size < 2:
        return _nan()
    yaw = _yaw_from_pose(raw_poses[indices], "xz")
    return float(_angle_diff_deg(np.asarray([yaw[-1]]), np.asarray([yaw[0]]))[0])


def _load_trajectory_stats(run_id: str, run_path: Path, gt_poses_all: np.ndarray, gt_pos_all: np.ndarray) -> TrajectoryStats:
    path = run_path / "01.txt" if run_path.is_dir() else run_path
    frames, raw_poses, raw_pos = _load_tum_prediction(path, gt_pos_all.shape[0])
    frames = frames.astype(np.int64)
    matched_gt = gt_pos_all[frames]
    scale, R, t = _umeyama_sim3(raw_pos, matched_gt, with_scale=True)
    aligned_poses = _apply_alignment(raw_poses, scale, R, t)
    aligned_pos = aligned_poses[:, :3, 3]
    err = aligned_pos - matched_gt
    err_norm = np.linalg.norm(err, axis=1)
    gt_poses = gt_poses_all[frames]
    yaw_err = _angle_diff_deg(_yaw_from_pose(aligned_poses, "xz"), _yaw_from_pose(gt_poses, "xz"))

    chunk_rows: Dict[int, Dict[str, float]] = {}
    for chunk_idx, start, end in _chunks(gt_pos_all.shape[0]):
        mask = (frames >= start) & (frames < end)
        idx = np.nonzero(mask)[0]
        if idx.size < 3:
            continue
        chunk_rows[chunk_idx] = {
            "chunk_idx": float(chunk_idx),
            "start_frame": float(start),
            "end_frame": float(end),
            "chunk_ate": _rmse(err_norm[idx]),
            "chunk_yaw_rmse": _rmse(yaw_err[idx]),
            "step_scale_log_gt_over_pred": _chunk_step_log_gt_over_pred(frames, raw_pos, gt_pos_all, start, end),
            "raw_yaw_delta_deg": _chunk_yaw_delta(raw_poses, idx),
        }

    segment_rows: Dict[Tuple[int, int], Dict[str, float]] = {}
    for start, end in SEGMENTS:
        mask = (frames >= start) & (frames < end)
        idx = np.nonzero(mask)[0]
        if idx.size < 3:
            segment_rows[(start, end)] = {"ATE": _nan(), "YawRMSE": _nan()}
            continue
        segment_rows[(start, end)] = {
            "ATE": _rmse(err_norm[idx]),
            "YawRMSE": _rmse(yaw_err[idx]),
        }

    return TrajectoryStats(
        run_id=run_id,
        frames=frames,
        raw_poses=raw_poses,
        raw_pos=raw_pos,
        aligned_poses=aligned_poses,
        aligned_pos=aligned_pos,
        gt_poses=gt_poses,
        gt_pos=matched_gt,
        err_norm=err_norm,
        yaw_err=yaw_err,
        sim3_scale=float(scale),
        global_ate=_rmse(err_norm),
        final_err=float(err_norm[-1]),
        yaw_rmse=_rmse(yaw_err),
        chunk_rows=chunk_rows,
        segment_rows=segment_rows,
    )


def _default_run_specs(v11_base: Path) -> List[RunSpec]:
    phase0 = v11_base / "phase0_repeats"
    phase2 = v11_base / "phase2_oracle_ttt"
    nogt = v11_base / "phase0_nogt_pose_proxy" / "NOGTPOSE_27_reset_global_clip35_body600_t105.tum.txt"
    return [
        RunSpec(
            "H9_REPEAT",
            phase0 / "V11_P0_REPEAT_H9_READBETA2_03_body485_exit425_c16_425_SWKS3",
            "online_hmc_ttt_write_baseline",
            True,
            True,
            True,
            False,
            False,
            "baseline_pass",
            include_phase1=True,
            include_selector=True,
        ),
        RunSpec(
            "C16ROLE_REPEAT",
            phase0 / "V11_P0_REPEAT_C16ROLE_01_SWKS3",
            "online_hmc_ttt_write_reference",
            True,
            True,
            True,
            False,
            False,
            "reference_only",
            include_selector=True,
        ),
        RunSpec(
            "WINGAM_REPEAT",
            phase0 / "V11_P0_REPEAT_WINGAM_03_SWKS3",
            "online_hmc_ttt_write_reference",
            True,
            True,
            True,
            False,
            False,
            "reference_only",
            include_selector=True,
        ),
        RunSpec(
            "ORACLE_TTT_01",
            phase2 / "V11_ORACLE_TTT_01_body_gt_scale_projection_w0_all_SWKS3",
            "gt_runtime_projection_oracle_upper_bound",
            True,
            True,
            False,
            True,
            False,
            "oracle_only_not_deployable",
            include_phase1=True,
            include_selector=True,
        ),
        RunSpec(
            "ORACLE_TTT_02",
            phase2 / "V11_ORACLE_TTT_02_body_handoff_gt_scale_projection_w0_all_SWKS3",
            "gt_runtime_projection_oracle_upper_bound",
            True,
            True,
            False,
            True,
            False,
            "oracle_only_not_deployable",
            include_phase1=True,
            include_selector=True,
        ),
        RunSpec(
            "ORACLE_TTT_03",
            phase2 / "V11_ORACLE_TTT_03_scale_projection_c5_16_w0_all_SWKS3",
            "gt_runtime_projection_oracle_upper_bound",
            True,
            True,
            False,
            True,
            False,
            "oracle_only_not_deployable",
            include_phase1=True,
            include_selector=True,
        ),
        RunSpec(
            "ORACLE_TTT_04",
            phase2 / "V11_ORACLE_TTT_04_toplayer_gt_scale_projection_w0_SWKS3",
            "gt_runtime_projection_oracle_upper_bound",
            True,
            True,
            False,
            True,
            False,
            "oracle_only_not_deployable",
            include_selector=True,
        ),
        RunSpec(
            "ORACLE_TTT_05_INV",
            phase2 / "V11_ORACLE_TTT_05_body_gt_scale_projection_inverse_w0_all_SWKS3",
            "gt_runtime_projection_oracle_upper_bound",
            True,
            True,
            False,
            True,
            False,
            "oracle_only_not_deployable",
            include_phase1=True,
            include_selector=True,
        ),
        RunSpec(
            "ORACLE_TTT_06_INV",
            phase2 / "V11_ORACLE_TTT_06_scale_projection_inverse_c5_16_w0_all_SWKS3",
            "gt_runtime_projection_oracle_upper_bound",
            True,
            True,
            False,
            True,
            False,
            "oracle_only_not_deployable",
            include_phase1=True,
            include_selector=True,
        ),
        RunSpec(
            "NOGTPOSE_27",
            nogt,
            "offline_nogt_pose_proxy_diagnostic",
            False,
            False,
            False,
            False,
            True,
            "diagnostic_only_not_ttt_write",
            include_phase1=True,
            include_selector=False,
        ),
    ]


def _build_registry(specs: Sequence[RunSpec], stats: Mapping[str, TrajectoryStats]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for spec in specs:
        st = stats.get(spec.run_id)
        run_dir = spec.path if spec.path.is_dir() else spec.path.parent
        hmc = _hmc_state_summary(run_dir) if spec.output_from_online_hmc else {}
        log = _parse_kitti_log(run_dir / "kitti_benchmark.log") if spec.output_from_online_hmc else {}
        row: Dict[str, object] = {
            "run_id": spec.run_id,
            "output_from_online_hmc": spec.output_from_online_hmc,
            "no_postprocess": spec.no_postprocess,
            "counts_as_ttt_write": spec.counts_as_ttt_write,
            "uses_gt_runtime_action": spec.uses_gt_runtime_action,
            "uses_offline_trajectory_rewrite": spec.uses_offline_trajectory_rewrite,
            "hmc_commit_mode": hmc.get("hmc_commit_mode", ""),
            "state_changed_count": hmc.get("state_changed_count", 0 if spec.output_from_online_hmc else ""),
            "candidate_family": spec.family,
            "gate_status": spec.gate_status,
            "deployable_success": bool(spec.counts_as_ttt_write and not spec.uses_gt_runtime_action and not spec.uses_offline_trajectory_rewrite),
            "hmc_state_rows": hmc.get("hmc_state_rows", ""),
            "commit_mode_ok": hmc.get("commit_mode_ok", ""),
            "source_path": str(spec.path),
            "ATE": st.global_ate if st else _nan(),
            "Rot": log.get("rot_log", _nan()),
            "RPE_t": log.get("rpe_t", _nan()),
            "RPE_r": log.get("rpe_r", _nan()),
            "FinalErr": st.final_err if st else _nan(),
            "YawRMSE": st.yaw_rmse if st else _nan(),
            "Sim3Scale": st.sim3_scale if st else _nan(),
        }
        if st:
            for start, end in SEGMENTS:
                seg = st.segment_rows.get((start, end), {})
                row[f"seg_{start}_{end}"] = seg.get("ATE", _nan())
        rows.append(row)
    return rows


def _mean(vals: Iterable[object]) -> float:
    good = [_float_or_nan(v) for v in vals]
    good = [v for v in good if math.isfinite(v)]
    return float(np.mean(good)) if good else _nan()


def _percentile(vals: Iterable[object], q: float) -> float:
    good = [_float_or_nan(v) for v in vals]
    good = [v for v in good if math.isfinite(v)]
    return float(np.percentile(np.asarray(good, dtype=np.float64), q)) if good else _nan()


def _rankdata_average(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    ranks = np.empty(arr.shape[0], dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    i = 0
    while i < order.size:
        j = i + 1
        while j < order.size and arr[order[j]] == arr[order[i]]:
            j += 1
        avg = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg
        i = j
    return ranks


def _spearman(x: Sequence[object], y: Sequence[object]) -> Tuple[float, int]:
    pairs = []
    for a, b in zip(x, y):
        aa = _float_or_nan(a)
        bb = _float_or_nan(b)
        if math.isfinite(aa) and math.isfinite(bb):
            pairs.append((aa, bb))
    if len(pairs) < 3:
        return _nan(), len(pairs)
    xx = _rankdata_average([p[0] for p in pairs])
    yy = _rankdata_average([p[1] for p in pairs])
    if float(np.std(xx)) <= EPS or float(np.std(yy)) <= EPS:
        return _nan(), len(pairs)
    rho = float(np.corrcoef(xx, yy)[0, 1])
    return rho, len(pairs)


def _trace_rows(run_dir: Path) -> List[Dict[str, object]]:
    return _read_jsonl(run_dir / "v11_projection_trace" / "v11_trace_summary.jsonl")


def _risk_rows(run_dir: Path) -> List[Dict[str, object]]:
    return _read_jsonl(run_dir / "v11_projection_trace" / "ttt_update_conflict_energy.jsonl")


def _active_chunk_set(run_dir: Path) -> set[int]:
    active = set()
    for row in _trace_rows(run_dir):
        if bool(row.get("projection_action_active", False)):
            try:
                active.add(int(row.get("chunk_idx")))
            except (TypeError, ValueError):
                pass
    return active


def _build_chunk_autopsy(specs: Sequence[RunSpec], stats: Mapping[str, TrajectoryStats]) -> List[Dict[str, object]]:
    h9 = stats["H9_REPEAT"]
    rows: List[Dict[str, object]] = []
    for spec in specs:
        if not spec.include_phase1 or spec.run_id == "H9_REPEAT" or not spec.output_from_online_hmc:
            continue
        run_dir = spec.path if spec.path.is_dir() else spec.path.parent
        st = stats[spec.run_id]
        traces = {int(row.get("chunk_idx")): row for row in _trace_rows(run_dir) if row.get("chunk_idx") is not None}
        risks_by_chunk: Dict[int, List[Dict[str, object]]] = defaultdict(list)
        for risk in _risk_rows(run_dir):
            try:
                risks_by_chunk[int(risk.get("chunk_idx"))].append(risk)
            except (TypeError, ValueError):
                continue
        for chunk_idx, trace in sorted(traces.items()):
            if not bool(trace.get("projection_action_active", False)):
                continue
            c = st.chunk_rows.get(chunk_idx)
            b = h9.chunk_rows.get(chunk_idx)
            if not c or not b:
                continue
            risk_rows = risks_by_chunk.get(chunk_idx, [])
            role = trace.get("projection_role_mass") if isinstance(trace.get("projection_role_mass"), dict) else {}
            delta_scale = c["step_scale_log_gt_over_pred"] - b["step_scale_log_gt_over_pred"]
            delta_abs_scale = abs(c["step_scale_log_gt_over_pred"]) - abs(b["step_scale_log_gt_over_pred"])
            delta_ate = c["chunk_ate"] - b["chunk_ate"]
            delta_yaw = c["chunk_yaw_rmse"] - b["chunk_yaw_rmse"]
            row = {
                "run_id": spec.run_id,
                "candidate_family": spec.family,
                "chunk_idx": chunk_idx,
                "start_frame": int(trace.get("start_frame", c["start_frame"])),
                "end_frame": int(trace.get("end_frame", c["end_frame"])),
                "risk_mean": _mean([r.get("risk_mean", r.get("risk_source_mean")) for r in risk_rows]),
                "risk_p90": _mean([r.get("risk_p90", r.get("risk_source_p90")) for r in risk_rows]),
                "risk_mean_p90_over_layers": _percentile([r.get("risk_mean", r.get("risk_source_mean")) for r in risk_rows], 90),
                "projection_helpful_energy": _float_or_nan(trace.get("projection_helpful_energy")),
                "projection_harmful_energy": _float_or_nan(trace.get("projection_harmful_energy")),
                "projection_role_pos_mass": _float_or_nan(role.get("pos_mass")),
                "projection_role_neutral_mass": _float_or_nan(role.get("neutral_mass")),
                "projection_role_neg_mass": _float_or_nan(role.get("neg_mass")),
                "update_cos_to_window_drift": _float_or_nan(trace.get("update_cos_to_window_drift")),
                "h9_step_scale_log_gt_over_pred": b["step_scale_log_gt_over_pred"],
                "candidate_step_scale_log_gt_over_pred": c["step_scale_log_gt_over_pred"],
                "actual_delta_step_scale_after_action": delta_scale,
                "actual_delta_abs_step_scale_error": delta_abs_scale,
                "actual_delta_yaw_after_action": delta_yaw,
                "actual_delta_segment_ATE": delta_ate,
                "actual_scale_improvement": -delta_abs_scale,
                "actual_segment_ATE_improvement": -delta_ate,
                "h9_chunk_ATE": b["chunk_ate"],
                "candidate_chunk_ATE": c["chunk_ate"],
                "h9_chunk_yaw_rmse": b["chunk_yaw_rmse"],
                "candidate_chunk_yaw_rmse": c["chunk_yaw_rmse"],
            }
            rows.append(row)
    return rows


def _correlation_summary(chunk_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in chunk_rows:
        groups["ALL"].append(row)
        groups[str(row.get("run_id"))].append(row)
        run_id = str(row.get("run_id"))
        if run_id.endswith("_INV"):
            groups["INVERSE_ONLY"].append(row)
        else:
            groups["NORMAL_ONLY"].append(row)

    out: List[Dict[str, object]] = []
    for name, rows in sorted(groups.items()):
        helpful = [row.get("projection_helpful_energy") for row in rows]
        harmonic = [row.get("projection_harmful_energy") for row in rows]
        risk = [row.get("risk_mean") for row in rows]
        scale_imp = [row.get("actual_scale_improvement") for row in rows]
        ate_imp = [row.get("actual_segment_ATE_improvement") for row in rows]
        helpful_scale, n1 = _spearman(helpful, scale_imp)
        helpful_ate, n2 = _spearman(helpful, ate_imp)
        harmful_scale, n3 = _spearman(harmonic, scale_imp)
        harmful_ate, n4 = _spearman(harmonic, ate_imp)
        risk_ate, n5 = _spearman(risk, ate_imp)
        out.append(
            {
                "group": name,
                "rows": len(rows),
                "spearman_helpful_vs_scale_improvement": helpful_scale,
                "spearman_helpful_vs_ATE_improvement": helpful_ate,
                "spearman_harmful_vs_scale_improvement": harmful_scale,
                "spearman_harmful_vs_ATE_improvement": harmful_ate,
                "spearman_risk_mean_vs_ATE_improvement": risk_ate,
                "n_helpful_scale": n1,
                "n_helpful_ate": n2,
                "n_harmful_scale": n3,
                "n_harmful_ate": n4,
                "n_risk_ate": n5,
            }
        )
    return out


def _layer_branch_summary(specs: Sequence[RunSpec]) -> List[Dict[str, object]]:
    accum: Dict[Tuple[str, int, str], Dict[str, object]] = {}
    for spec in specs:
        if not spec.include_phase1 or spec.run_id == "H9_REPEAT" or not spec.output_from_online_hmc:
            continue
        run_dir = spec.path if spec.path.is_dir() else spec.path.parent
        active = _active_chunk_set(run_dir)
        rows = _read_csv(run_dir / "v11_projection_trace" / "update_to_drift_projection.csv")
        for rec in rows:
            try:
                chunk_idx = int(rec.get("chunk_idx", ""))
                layer = int(rec.get("layer", ""))
            except ValueError:
                continue
            if chunk_idx not in active:
                continue
            branch = str(rec.get("branch", ""))
            key = (spec.run_id, layer, branch)
            if key not in accum:
                accum[key] = {
                    "run_id": spec.run_id,
                    "layer_id": layer,
                    "branch_id": branch,
                    "active_chunk_rows": 0,
                    "native_update_norm": _nan(),
                    "candidate_update_norm_values": [],
                    "update_cos_to_native": _nan(),
                    "update_cos_to_window_drift_values": [],
                    "candidate_minus_native_norm": _nan(),
                    "projection_helpful_values": [],
                    "projection_harmful_values": [],
                    "native_comparison_available": False,
                    "native_comparison_note": "H9 does not have matching v11 per-layer projection trace; native columns intentionally left NaN.",
                }
            item = accum[key]
            item["active_chunk_rows"] = int(item["active_chunk_rows"]) + 1
            item["candidate_update_norm_values"].append(_float_or_nan(rec.get("delta_norm")))  # type: ignore[index]
            item["update_cos_to_window_drift_values"].append(_float_or_nan(rec.get("update_cos_to_window_drift")))  # type: ignore[index]
            item["projection_helpful_values"].append(_float_or_nan(rec.get("projection_helpful_energy")))  # type: ignore[index]
            item["projection_harmful_values"].append(_float_or_nan(rec.get("projection_harmful_energy")))  # type: ignore[index]
    out: List[Dict[str, object]] = []
    for item in accum.values():
        out.append(
            {
                "run_id": item["run_id"],
                "layer_id": item["layer_id"],
                "branch_id": item["branch_id"],
                "active_chunk_rows": item["active_chunk_rows"],
                "native_update_norm": item["native_update_norm"],
                "candidate_update_norm": _mean(item["candidate_update_norm_values"]),  # type: ignore[arg-type]
                "update_cos_to_native": item["update_cos_to_native"],
                "update_cos_to_window_drift": _mean(item["update_cos_to_window_drift_values"]),  # type: ignore[arg-type]
                "candidate_minus_native_norm": item["candidate_minus_native_norm"],
                "projection_helpful_energy_sum": float(np.nansum(np.asarray(item["projection_helpful_values"], dtype=np.float64))),  # type: ignore[arg-type]
                "projection_harmful_energy_sum": float(np.nansum(np.asarray(item["projection_harmful_values"], dtype=np.float64))),  # type: ignore[arg-type]
                "native_comparison_available": item["native_comparison_available"],
                "native_comparison_note": item["native_comparison_note"],
            }
        )
    return sorted(out, key=lambda r: (str(r["run_id"]), int(r["layer_id"]), str(r["branch_id"])))


def _proxy_chunk_features(stats: TrajectoryStats) -> Dict[int, Dict[str, float]]:
    chunk_ids = sorted(stats.chunk_rows)
    log_steps = {idx: stats.chunk_rows[idx]["step_scale_log_gt_over_pred"] for idx in chunk_ids}
    # A true no-GT selector cannot know GT/pred scale. For audit scoring below,
    # recompute the predicted step log in its own coordinate system.
    raw_step_logs: Dict[int, float] = {}
    for idx in chunk_ids:
        start = int(stats.chunk_rows[idx]["start_frame"])
        end = int(stats.chunk_rows[idx]["end_frame"])
        mask = (stats.frames >= start) & (stats.frames < end)
        indices = np.nonzero(mask)[0]
        pred_steps = _step_lengths(stats.raw_pos[indices]) if indices.size >= 2 else np.asarray([])
        med = float(np.nanmedian(pred_steps)) if pred_steps.size else _nan()
        raw_step_logs[idx] = float(math.log(max(med, EPS))) if _is_finite(med) and med > 0 else _nan()

    out: Dict[int, Dict[str, float]] = {}
    ema: Optional[float] = None
    yaw_hist: List[float] = []
    for idx in chunk_ids:
        raw_log = raw_step_logs[idx]
        yaw_delta = stats.chunk_rows[idx]["raw_yaw_delta_deg"]
        if ema is None or not _is_finite(raw_log):
            scale_proxy = 0.0
            ema = raw_log if _is_finite(raw_log) else ema
        else:
            scale_proxy = abs(raw_log - ema)
            ema = 0.8 * ema + 0.2 * raw_log
        if yaw_hist and _is_finite(yaw_delta):
            yaw_proxy = abs(yaw_delta - float(np.nanmedian(np.asarray(yaw_hist[-5:], dtype=np.float64))))
        else:
            yaw_proxy = 0.0
        if _is_finite(yaw_delta):
            yaw_hist.append(yaw_delta)
        # Jitter: local variation of raw step lengths inside the chunk, no GT.
        start = int(stats.chunk_rows[idx]["start_frame"])
        end = int(stats.chunk_rows[idx]["end_frame"])
        mask = (stats.frames >= start) & (stats.frames < end)
        indices = np.nonzero(mask)[0]
        pred_steps = _step_lengths(stats.raw_pos[indices]) if indices.size >= 2 else np.asarray([])
        if pred_steps.size >= 3 and float(np.nanmedian(pred_steps)) > 0:
            jitter = float(np.nanstd(pred_steps) / max(np.nanmedian(pred_steps), EPS))
        else:
            jitter = 0.0
        out[idx] = {
            "chunk_idx": float(idx),
            "J_scale": float(scale_proxy),
            "J_yaw": float(yaw_proxy),
            "J_jitter": float(jitter),
            "gt_over_pred_scale_log_for_diagnostics_only": log_steps[idx],
        }
    return out


def _selector_audit(specs: Sequence[RunSpec], stats: Mapping[str, TrajectoryStats]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    candidate_ids = [spec.run_id for spec in specs if spec.include_selector and spec.run_id in stats]
    h9 = stats["H9_REPEAT"]
    per_run_rows: List[Dict[str, object]] = []
    per_chunk_rows: List[Dict[str, object]] = []
    chunk_feature_by_run = {rid: _proxy_chunk_features(stats[rid]) for rid in candidate_ids}

    for rid in candidate_ids:
        feats = chunk_feature_by_run[rid]
        active_ids = [idx for idx in sorted(feats) if 5 <= idx <= 16]
        j_scale = _mean(feats[idx]["J_scale"] for idx in active_ids)
        j_yaw = _mean(feats[idx]["J_yaw"] for idx in active_ids)
        j_jitter = _mean(feats[idx]["J_jitter"] for idx in active_ids)
        score = j_scale + 0.01 * j_yaw + 0.10 * j_jitter
        st = stats[rid]
        per_run_rows.append(
            {
                "run_id": rid,
                "proxy_score_balanced": score,
                "proxy_scale_mean": j_scale,
                "proxy_yaw_mean": j_yaw,
                "proxy_jitter_mean": j_jitter,
                "ATE": st.global_ate,
                "delta_ATE_vs_H9": st.global_ate - h9.global_ate,
                "is_oracle_gt_runtime": rid.startswith("ORACLE_TTT"),
                "selector_rank_proxy": _nan(),
                "actual_rank_ATE": _nan(),
            }
        )

    score_ranks = _rankdata_average([float(r["proxy_score_balanced"]) for r in per_run_rows])
    ate_ranks = _rankdata_average([float(r["ATE"]) for r in per_run_rows])
    for row, sr, ar in zip(per_run_rows, score_ranks, ate_ranks):
        row["selector_rank_proxy"] = sr
        row["actual_rank_ATE"] = ar

    selected_good = 0
    selected_total = 0
    selected_oracle_worse = 0
    chunk_range = sorted(idx for idx in h9.chunk_rows if 5 <= idx <= 16)
    for idx in chunk_range:
        choices = []
        for rid in candidate_ids:
            feats = chunk_feature_by_run[rid].get(idx)
            actual = stats[rid].chunk_rows.get(idx)
            h9_actual = h9.chunk_rows.get(idx)
            if not feats or not actual or not h9_actual:
                continue
            score = feats["J_scale"] + 0.01 * feats["J_yaw"] + 0.10 * feats["J_jitter"]
            choices.append((score, rid, feats, actual, h9_actual))
        if not choices:
            continue
        score, rid, feats, actual, h9_actual = sorted(choices, key=lambda x: x[0])[0]
        better_or_equal_h9 = actual["chunk_ate"] <= h9_actual["chunk_ate"] + 1e-9
        selected_good += int(better_or_equal_h9)
        selected_total += 1
        if rid.startswith("ORACLE_TTT") and stats[rid].global_ate > h9.global_ate:
            selected_oracle_worse += 1
        per_chunk_rows.append(
            {
                "chunk_idx": idx,
                "selected_run_id": rid,
                "selected_proxy_score": score,
                "selected_proxy_scale": feats["J_scale"],
                "selected_proxy_yaw": feats["J_yaw"],
                "selected_proxy_jitter": feats["J_jitter"],
                "selected_chunk_ATE": actual["chunk_ate"],
                "H9_chunk_ATE": h9_actual["chunk_ate"],
                "selected_delta_ATE_vs_H9": actual["chunk_ate"] - h9_actual["chunk_ate"],
                "selected_is_H9_or_better_by_chunk_ATE": better_or_equal_h9,
                "selected_is_oracle_with_worse_global_ATE": bool(rid.startswith("ORACLE_TTT") and stats[rid].global_ate > h9.global_ate),
            }
        )

    rho_score_ate, rho_n = _spearman(
        [row["proxy_score_balanced"] for row in per_run_rows],
        [row["ATE"] for row in per_run_rows],
    )
    best_proxy = min(per_run_rows, key=lambda row: float(row["proxy_score_balanced"])) if per_run_rows else {}
    best_proxy_is_h9_or_better = bool(best_proxy) and float(best_proxy["ATE"]) <= h9.global_ate + 1e-9
    choose_good_ratio = (selected_good / selected_total) if selected_total else _nan()
    gate_pass = (
        _is_finite(rho_score_ate)
        and rho_score_ate >= 0.5
        and best_proxy_is_h9_or_better
        and _is_finite(choose_good_ratio)
        and choose_good_ratio >= 0.70
        and selected_oracle_worse == 0
    )
    gate = {
        "phase": "Phase 2A offline selector audit",
        "proxy_score_definition": "J_scale + 0.01*J_yaw + 0.10*J_jitter, all no-GT trajectory-internal terms",
        "candidate_count": len(candidate_ids),
        "chunk_selection_total": selected_total,
        "chunk_selected_H9_or_better_ratio": choose_good_ratio,
        "selected_oracle_with_worse_global_ATE_count": selected_oracle_worse,
        "spearman_proxy_score_vs_ATE": rho_score_ate,
        "spearman_n": rho_n,
        "best_proxy_run_id": best_proxy.get("run_id", ""),
        "best_proxy_run_ATE": best_proxy.get("ATE", _nan()),
        "best_proxy_is_H9_or_better": best_proxy_is_h9_or_better,
        "gate_pass": gate_pass,
        "phase2B_full_or_smoke_allowed": gate_pass,
        "reason": "pass" if gate_pass else "offline selector proxy did not meet v12 Phase 2A gate; do not run SELECT_TTT full/smoke from this selector",
    }
    return per_run_rows, per_chunk_rows, gate


def _phase1_gate(
    registry_rows: Sequence[Mapping[str, object]],
    corr_rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    by_group = {str(row.get("group")): row for row in corr_rows}
    all_row = by_group.get("ALL", {})
    rho_scale = _float_or_nan(all_row.get("spearman_helpful_vs_scale_improvement"))
    rho_ate = _float_or_nan(all_row.get("spearman_helpful_vs_ATE_improvement"))
    h9_ate = next(float(row["ATE"]) for row in registry_rows if row.get("run_id") == "H9_REPEAT")
    oracle_rows = [row for row in registry_rows if str(row.get("run_id", "")).startswith("ORACLE_TTT")]
    normal_worse = [
        row for row in oracle_rows if not str(row.get("run_id")).endswith("_INV") and _float_or_nan(row.get("ATE")) > h9_ate
    ]
    inverse_worse = [
        row for row in oracle_rows if str(row.get("run_id")).endswith("_INV") and _float_or_nan(row.get("ATE")) > h9_ate
    ]
    low_corr = (
        (not _is_finite(rho_scale) or rho_scale <= 0.2)
        or (not _is_finite(rho_ate) or rho_ate <= 0.2)
    )
    normal_and_inverse_worse = bool(normal_worse) and bool(inverse_worse)
    failed_coordinate = bool(low_corr or normal_and_inverse_worse)
    return {
        "phase": "Phase 1 projection action autopsy",
        "spearman_helpful_vs_scale_improvement": rho_scale,
        "spearman_helpful_vs_ATE_improvement": rho_ate,
        "low_correlation_gate": low_corr,
        "normal_and_inverse_oracles_worse_than_H9": normal_and_inverse_worse,
        "normal_oracle_worse_count": len(normal_worse),
        "inverse_oracle_worse_count": len(inverse_worse),
        "projection_action_coordinate_failed": failed_coordinate,
        "phase2_projection_selector_allowed": not failed_coordinate,
        "recommended_next": (
            "stop v11_gt_scale_projection risk source; use Phase 3/4 only with a new non-GT proxy or auxiliary objective"
            if failed_coordinate
            else "projection cue may be used in candidate selector, but still requires Phase 2A gate"
        ),
    }


def _try_plots(out_dir: Path, chunk_rows: Sequence[Mapping[str, object]], layer_rows: Sequence[Mapping[str, object]], selector_rows: Sequence[Mapping[str, object]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    if chunk_rows:
        fig, ax = plt.subplots(figsize=(6, 4))
        x = [_float_or_nan(row.get("projection_helpful_energy")) for row in chunk_rows]
        y = [_float_or_nan(row.get("actual_segment_ATE_improvement")) for row in chunk_rows]
        labels = [str(row.get("run_id")) for row in chunk_rows]
        colors = ["#1f77b4" if not label.endswith("_INV") else "#d62728" for label in labels]
        ax.scatter(x, y, c=colors, s=24, alpha=0.85)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("projection_helpful_energy")
        ax.set_ylabel("actual segment ATE improvement vs H9 (m)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "projection_helpful_vs_actual_delta_scatter.png", dpi=160)
        plt.close(fig)

    if layer_rows:
        # One compact heatmap for candidate_update_norm averaged over all oracle runs.
        layers = sorted({int(row["layer_id"]) for row in layer_rows})
        branches = sorted({str(row["branch_id"]) for row in layer_rows})
        data = np.full((len(layers), len(branches)), np.nan, dtype=np.float64)
        for i, layer in enumerate(layers):
            for j, branch in enumerate(branches):
                vals = [
                    _float_or_nan(row.get("candidate_update_norm"))
                    for row in layer_rows
                    if int(row["layer_id"]) == layer and str(row["branch_id"]) == branch
                ]
                vals = [v for v in vals if math.isfinite(v)]
                if vals:
                    data[i, j] = float(np.mean(vals))
        fig, ax = plt.subplots(figsize=(5, 7))
        im = ax.imshow(data, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(branches)), branches)
        ax.set_yticks(range(len(layers)), layers)
        ax.set_xlabel("branch")
        ax.set_ylabel("layer")
        ax.set_title("candidate update norm, active chunks")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / "layer_branch_update_norm_heatmap.png", dpi=160)
        plt.close(fig)

        cos_data = np.full((len(layers), len(branches)), np.nan, dtype=np.float64)
        for i, layer in enumerate(layers):
            for j, branch in enumerate(branches):
                vals = [
                    _float_or_nan(row.get("update_cos_to_window_drift"))
                    for row in layer_rows
                    if int(row["layer_id"]) == layer and str(row["branch_id"]) == branch
                ]
                vals = [v for v in vals if math.isfinite(v)]
                if vals:
                    cos_data[i, j] = float(np.mean(vals))
        fig, ax = plt.subplots(figsize=(5, 7))
        im = ax.imshow(cos_data, aspect="auto", interpolation="nearest", vmin=-0.05, vmax=0.05, cmap="coolwarm")
        ax.set_xticks(range(len(branches)), branches)
        ax.set_yticks(range(len(layers)), layers)
        ax.set_xlabel("branch")
        ax.set_ylabel("layer")
        ax.set_title("update cosine to drift, active chunks")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / "update_to_drift_projection_heatmap.png", dpi=160)
        plt.close(fig)

    if selector_rows:
        names = [str(row["run_id"]) for row in selector_rows]
        deltas = [_float_or_nan(row.get("delta_ATE_vs_H9")) for row in selector_rows]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(names, deltas, color=["#2ca02c" if v <= 0 else "#d62728" for v in deltas])
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("ATE delta vs H9 (m)")
        ax.tick_params(axis="x", labelrotation=45)
        fig.tight_layout()
        fig.savefig(out_dir / "candidate_minus_H9_global_ATE_bar.png", dpi=160)
        plt.close(fig)


def _write_report(
    path: Path,
    v12_base: Path,
    registry_rows: Sequence[Mapping[str, object]],
    phase1_gate: Mapping[str, object],
    corr_rows: Sequence[Mapping[str, object]],
    selector_gate: Mapping[str, object],
) -> None:
    h9 = next(row for row in registry_rows if row.get("run_id") == "H9_REPEAT")
    nogt = next(row for row in registry_rows if row.get("run_id") == "NOGTPOSE_27")
    oracle_rows = [row for row in registry_rows if str(row.get("run_id", "")).startswith("ORACLE_TTT")]
    best_oracle = min(oracle_rows, key=lambda row: _float_or_nan(row.get("ATE")))
    all_corr = next((row for row in corr_rows if row.get("group") == "ALL"), {})
    lines = [
        "# ACL2 v12 实验复盘：TTT Write WindowScale/MPC Target25",
        "",
        "日期：2026-05-18",
        "",
        "本文件由 `tools/v12_ttt_write_autopsy.py` 从已有落盘 artifact 生成；没有把缺失数据补写成实验结果。",
        "",
        "## 0. 固定边界",
        "",
        f"- Best online HMC / TTT-write baseline: `H9_REPEAT`, ATE `{_float_or_nan(h9.get('ATE')):.4f}`, Rot `{_float_or_nan(h9.get('Rot')):.4f}`, counts_as_ttt_write=`{h9.get('counts_as_ttt_write')}`。",
        f"- Best diagnostic-only no-GT pose proxy: `NOGTPOSE_27`, ATE `{_float_or_nan(nogt.get('ATE')):.4f}`, counts_as_ttt_write=`{nogt.get('counts_as_ttt_write')}`, uses_offline_trajectory_rewrite=`{nogt.get('uses_offline_trajectory_rewrite')}`。",
        f"- Best v11 projection oracle: `{best_oracle.get('run_id')}`, ATE `{_float_or_nan(best_oracle.get('ATE')):.4f}`, uses_gt_runtime_action=`{best_oracle.get('uses_gt_runtime_action')}`, deployable_success=`{best_oracle.get('deployable_success')}`。",
        "",
        "## 1. Phase 0 Registry",
        "",
        "输出：",
        "",
        "```text",
        str(v12_base / "v12_result_registry.csv"),
        "```",
        "",
        "Phase 0 gate 结论：H9 是 deployable online baseline；NOGTPOSE_27 是 offline diagnostic；ORACLE_TTT_* 是 GT runtime oracle-only，不允许写成 deployable TTT success。",
        "",
        "## 2. Phase 1 Projection Autopsy",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Spearman helpful vs scale improvement | `{_float_or_nan(all_corr.get('spearman_helpful_vs_scale_improvement')):.4f}` |",
        f"| Spearman helpful vs segment ATE improvement | `{_float_or_nan(all_corr.get('spearman_helpful_vs_ATE_improvement')):.4f}` |",
        f"| normal+inverse oracle worse than H9 | `{phase1_gate.get('normal_and_inverse_oracles_worse_than_H9')}` |",
        f"| projection action coordinate failed | `{phase1_gate.get('projection_action_coordinate_failed')}` |",
        "",
        "输出：",
        "",
        "```text",
        str(v12_base / "phase1_projection_autopsy" / "projection_chunk_autopsy.csv"),
        str(v12_base / "phase1_projection_autopsy" / "projection_correlation_summary.csv"),
        str(v12_base / "phase1_projection_autopsy" / "layer_branch_projection_summary.csv"),
        "```",
        "",
        "Native-vs-candidate layer delta 对比没有被伪造：H9 没有匹配的 v11 per-layer projection trace，因此 `native_update_norm/update_cos_to_native/candidate_minus_native_norm` 在 layer summary 中保留为 NaN，并明确标注 unavailable。",
        "",
        "## 3. Phase 2A Offline Selector Audit",
        "",
        "| Check | Value |",
        "|---|---:|",
        f"| candidate_count | `{selector_gate.get('candidate_count')}` |",
        f"| chunk_selected_H9_or_better_ratio | `{_float_or_nan(selector_gate.get('chunk_selected_H9_or_better_ratio')):.4f}` |",
        f"| selected_oracle_with_worse_global_ATE_count | `{selector_gate.get('selected_oracle_with_worse_global_ATE_count')}` |",
        f"| Spearman proxy score vs ATE | `{_float_or_nan(selector_gate.get('spearman_proxy_score_vs_ATE')):.4f}` |",
        f"| best proxy run | `{selector_gate.get('best_proxy_run_id')}` |",
        f"| gate_pass | `{selector_gate.get('gate_pass')}` |",
        "",
        "输出：",
        "",
        "```text",
        str(v12_base / "phase2_selector_offline_audit" / "offline_selector_run_scores.csv"),
        str(v12_base / "phase2_selector_offline_audit" / "offline_selector_chunk_choices.csv"),
        str(v12_base / "phase2_selector_offline_audit" / "offline_selector_gate.json"),
        "```",
        "",
        "## 4. v12 当前结论",
        "",
    ]
    if bool(phase1_gate.get("projection_action_coordinate_failed")):
        lines.append("1. `v11_gt_scale_projection` 不再允许作为 v12 selector 的有效 projection coordinate；当前数据支持停止这条 risk source。")
    else:
        lines.append("1. Projection coordinate 没有被 Phase 1 gate 否决，但后续仍必须经过 Phase 2A selector gate。")
    if not bool(selector_gate.get("gate_pass")):
        lines.append("2. Phase 2A offline selector gate 失败，因此没有启动 `SELECT_TTT_*` smoke/full run。")
    else:
        lines.append("2. Phase 2A offline selector gate 通过，可以进入 `END_FRAME=256` selector smoke。")
    lines.extend(
        [
            "3. 截至本文件生成时，没有新的 deployable online TTT write candidate，也没有 online target-25。",
            "4. `NOGTPOSE_27` 仍只证明 window-level pose/scale state 是 target-25 杠杆，不能计入 TTT write success。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", default=DEFAULT_GT)
    parser.add_argument("--v11-base", default=DEFAULT_V11_BASE)
    parser.add_argument("--v12-base", default=DEFAULT_V12_BASE)
    parser.add_argument("--report", default="docs/ACL2_v12_实验复盘.md")
    args = parser.parse_args()

    v11_base = Path(args.v11_base)
    v12_base = Path(args.v12_base)
    phase1_dir = v12_base / "phase1_projection_autopsy"
    selector_dir = v12_base / "phase2_selector_offline_audit"
    phase1_dir.mkdir(parents=True, exist_ok=True)
    selector_dir.mkdir(parents=True, exist_ok=True)

    _gt_frames, gt_poses_all, gt_pos_all = _load_kitti_gt(Path(args.gt))
    specs = _default_run_specs(v11_base)

    stats: Dict[str, TrajectoryStats] = {}
    missing: List[str] = []
    for spec in specs:
        path = spec.path / "01.txt" if spec.path.is_dir() else spec.path
        if not path.exists():
            missing.append(f"{spec.run_id}: {path}")
            continue
        stats[spec.run_id] = _load_trajectory_stats(spec.run_id, spec.path, gt_poses_all, gt_pos_all)
    if missing:
        raise SystemExit("Missing required run artifacts:\n" + "\n".join(missing))

    registry_rows = _build_registry(specs, stats)
    registry_fields = [
        "run_id",
        "output_from_online_hmc",
        "no_postprocess",
        "counts_as_ttt_write",
        "uses_gt_runtime_action",
        "uses_offline_trajectory_rewrite",
        "hmc_commit_mode",
        "state_changed_count",
        "candidate_family",
        "gate_status",
        "deployable_success",
        "hmc_state_rows",
        "commit_mode_ok",
        "ATE",
        "Rot",
        "RPE_t",
        "RPE_r",
        "FinalErr",
        "YawRMSE",
        "Sim3Scale",
        "seg_200_300",
        "seg_200_400",
        "seg_400_600",
        "source_path",
    ]
    _write_csv(v12_base / "v12_result_registry.csv", registry_rows, registry_fields)

    chunk_rows = _build_chunk_autopsy(specs, stats)
    chunk_fields = [
        "run_id",
        "candidate_family",
        "chunk_idx",
        "start_frame",
        "end_frame",
        "risk_mean",
        "risk_p90",
        "risk_mean_p90_over_layers",
        "projection_helpful_energy",
        "projection_harmful_energy",
        "projection_role_pos_mass",
        "projection_role_neutral_mass",
        "projection_role_neg_mass",
        "update_cos_to_window_drift",
        "h9_step_scale_log_gt_over_pred",
        "candidate_step_scale_log_gt_over_pred",
        "actual_delta_step_scale_after_action",
        "actual_delta_abs_step_scale_error",
        "actual_delta_yaw_after_action",
        "actual_delta_segment_ATE",
        "actual_scale_improvement",
        "actual_segment_ATE_improvement",
        "h9_chunk_ATE",
        "candidate_chunk_ATE",
        "h9_chunk_yaw_rmse",
        "candidate_chunk_yaw_rmse",
    ]
    _write_csv(phase1_dir / "projection_chunk_autopsy.csv", chunk_rows, chunk_fields)

    corr_rows = _correlation_summary(chunk_rows)
    _write_csv(phase1_dir / "projection_correlation_summary.csv", corr_rows)

    layer_rows = _layer_branch_summary(specs)
    _write_csv(phase1_dir / "layer_branch_projection_summary.csv", layer_rows)

    phase1_gate = _phase1_gate(registry_rows, corr_rows)
    _write_json(phase1_dir / "phase1_projection_gate.json", phase1_gate)

    selector_run_rows, selector_chunk_rows, selector_gate = _selector_audit(specs, stats)
    _write_csv(selector_dir / "offline_selector_run_scores.csv", selector_run_rows)
    _write_csv(selector_dir / "offline_selector_chunk_choices.csv", selector_chunk_rows)
    _write_json(selector_dir / "offline_selector_gate.json", selector_gate)

    _try_plots(phase1_dir, chunk_rows, layer_rows, selector_run_rows)
    _write_report(Path(args.report), v12_base, registry_rows, phase1_gate, corr_rows, selector_gate)

    print(f"Wrote {v12_base / 'v12_result_registry.csv'}")
    print(f"Wrote {phase1_dir}")
    print(f"Wrote {selector_dir}")
    print(f"Wrote {args.report}")
    print(json.dumps({"phase1_gate": phase1_gate, "selector_gate": selector_gate}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
