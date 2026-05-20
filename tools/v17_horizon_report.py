#!/usr/bin/env python3
"""Aggregate ACL2 v17 h5/h8/h10 causal-fork horizon rollouts.

The delta convention is intentionally strict: every candidate is compared
against H9 recomputed on the exact same prediction frame intersection. This
avoids mixing full-run or older h3 table baselines with longer-horizon rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

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


FAMILY_BY_CANDIDATE = {
    "K1_H9": "baseline",
    "K11_ORTHO_SUPPRESS_W0": "ortho_suppress_w0",
    "K21_ORTHO_SUPPRESS_ALL": "ortho_suppress_all",
    "K13_ORTHO_RHO025_W0": "ortho_suppress_rho025",
    "K14_TTGR_ZERO_ORTHO_W0": "ttgr_zero_ortho_suppress",
    "K25_ONLINE_SCALE_OVERLAP_STEP": "online_scale_state_overlap_step",
    "K26_ONLINE_SCALE_OVERLAP_TIGHT": "online_scale_state_overlap_tight",
    "BASIS_01_PROXY_HARM_W0": "basis_proxy_harm_w0",
    "BASIS_02_PROXY_HARM_W0_EMA090": "basis_proxy_harm_w0_ema090",
    "BASIS_03_PROXY_RHO025_W0": "basis_proxy_rho025_w0",
    "BASIS_04_PROXY_CONFLICT_COMMIT_W0": "basis_proxy_conflict_commit_w0",
    "BASIS_05_PROXY_TTGR_ZERO_POSTZP_W0": "basis_proxy_ttgr_zero_postzp_w0",
    "BASIS_06_PROXY_BRANCHSEP_W0W2": "basis_proxy_branchsep_w0w2",
    "AUXGEO_01_PROXY_OVERLAP_V_W0": "auxgeo_proxy_overlap_v_w0",
    "AUXGEO_02_PROXY_OVERLAP_KV_W0": "auxgeo_proxy_overlap_kv_w0",
    "AUXGEO_03_PROXY_STRUCT_KV_W0W2": "auxgeo_proxy_struct_kv_w0w2",
    "AUXGEO_04_PROXY_BODY_WEAK_V_W0": "auxgeo_proxy_body_weak_v_w0",
    "AUXGEO_05_PROXY_EXIT_STRONG_KV_W0": "auxgeo_proxy_exit_strong_kv_w0",
    "AUXGEO_06_PROXY_STATIC_TOPK_KV_W0": "auxgeo_proxy_static_topk_kv_w0",
    "DLBANK_01_SHORT_CONFLICT_TAU2_W0": "dual_bank_short_conflict_tau2_w0",
    "DLBANK_02_SHORT_CONFLICT_TAU3_W0": "dual_bank_short_conflict_tau3_w0",
    "DLBANK_03_STRUCTURE_LONG_SHORT_REST": "dual_bank_structure_long_short_rest",
}

CHUNK_START_FRAME = {
    5: 145,
    6: 174,
    9: 261,
    10: 290,
    12: 348,
    16: 464,
}


def _uses_gt_runtime_action(candidate_id: str) -> bool:
    return False


def _is_deployable_ttt_write_candidate(candidate_id: str) -> bool:
    return not (candidate_id.startswith("K25_") or candidate_id.startswith("K26_"))


def _to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _serialize_value(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, rows: List[Dict[str, object]]) -> None:
    serial = [
        {key: _serialize_value(value) for key, value in row.items()}
        for row in rows
    ]
    path.write_text(json.dumps(serial, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _align_metrics(frames: np.ndarray, raw_poses: np.ndarray, gt_poses: np.ndarray, gt_pos: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    frames = frames.astype(np.int64)
    raw_pos = raw_poses[:, :3, 3]
    matched_gt_pos = gt_pos[frames]
    scale, rot, trans = _umeyama_sim3(raw_pos, matched_gt_pos, with_scale=True)
    aligned = _apply_alignment(raw_poses, scale, rot, trans)
    err = aligned[:, :3, 3] - matched_gt_pos
    err_norm = np.linalg.norm(err, axis=1)
    yaw = _yaw_from_pose(aligned)
    gt_yaw = _yaw_from_pose(gt_poses[frames])
    yaw_err = _angle_diff_deg(yaw, gt_yaw)
    metrics = {
        "ATE_horizon": _rmse(err_norm),
        "Rot_horizon": _rmse(yaw_err),
        "FinalErr_horizon": float(err_norm[-1]) if err_norm.shape[0] else float("nan"),
        "alignment_scale": float(scale),
    }
    return aligned, metrics


def _segment_ate(frames: np.ndarray, aligned: np.ndarray, gt_pos: np.ndarray, start: int, end: int) -> float:
    mask = (frames >= start) & (frames < end)
    if int(mask.sum()) < 3:
        return float("nan")
    err = aligned[mask, :3, 3] - gt_pos[frames[mask]]
    return _rmse(np.linalg.norm(err, axis=1))


def _pose_lookup(frames: np.ndarray, poses: np.ndarray) -> Dict[int, np.ndarray]:
    return {int(frame): pose for frame, pose in zip(frames, poses)}


def _raw_diff(frames: np.ndarray, poses: np.ndarray, reference: Mapping[int, np.ndarray]) -> Tuple[float, float, bool]:
    max_abs = 0.0
    max_trans = 0.0
    matched = 0
    for frame, pose in zip(frames, poses):
        ref = reference.get(int(frame))
        if ref is None:
            return float("nan"), float("nan"), False
        diff = pose - ref
        max_abs = max(max_abs, float(np.max(np.abs(diff))))
        max_trans = max(max_trans, float(np.linalg.norm(diff[:3, 3])))
        matched += 1
    return max_abs, max_trans, matched == int(frames.shape[0])


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _hash_map(path: Path, *, value_key: str, include_kind: bool = False) -> Dict[Tuple[int, str], str]:
    out: Dict[Tuple[int, str], str] = {}
    for row in _read_jsonl(path):
        if "chunk_idx" not in row:
            continue
        kind = str(row.get("kind", "")) if include_kind else ""
        value = row.get(value_key)
        if value is None:
            continue
        out[(int(row["chunk_idx"]), kind)] = str(value)
    return out


def _hash_mismatch_count(run_map: Mapping[Tuple[int, str], str], ref_map: Mapping[Tuple[int, str], str]) -> Tuple[int, int]:
    compared = 0
    mismatches = 0
    for key, value in run_map.items():
        ref = ref_map.get(key)
        if ref is None:
            continue
        compared += 1
        if value != ref:
            mismatches += 1
    return mismatches, compared


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _runtime_sec(run_dir: Path) -> float:
    path = run_dir / "run_status.txt"
    if not path.exists():
        return float("nan")
    starts: List[datetime] = []
    dones: List[datetime] = []
    pattern = re.compile(r"^\[(?P<ts>[^\]]+)\]\s+(?P<kind>START|DONE)\b")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        stamp = datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S")
        if match.group("kind") == "START":
            starts.append(stamp)
        else:
            dones.append(stamp)
    if not starts or not dones:
        return float("nan")
    return float((max(dones) - min(starts)).total_seconds())


def _run_done(run_dir: Path) -> bool:
    status = run_dir / "run_status.txt"
    if not status.exists():
        return False
    return "DONE" in status.read_text(encoding="utf-8", errors="replace")


def _run_dir(rollout_root: Path, run_prefix: str, candidate_id: str, chunk_id: int, horizon: int) -> Path:
    name = f"{run_prefix}_{candidate_id}_chunk{chunk_id}_h{horizon}_globalgate_H9parent_SWKS3"
    return rollout_root / name


def _eval_window(chunk_id: int, horizon: int) -> Tuple[int, int]:
    """Return [start, end) for the requested horizon, excluding plus-one flush frames."""
    start = CHUNK_START_FRAME[chunk_id]
    return start, start + 32 + (horizon - 1) * 29


def _best_rows(rows: Iterable[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[int, int], List[Mapping[str, object]]] = {}
    for row in rows:
        if row.get("candidate_id") == "K1_H9":
            continue
        grouped.setdefault((int(row["chunk_id"]), int(row["horizon"])), []).append(row)
    out: List[Dict[str, object]] = []
    for (chunk_id, horizon), group in sorted(grouped.items()):
        finite = [row for row in group if math.isfinite(_to_float(row.get("ATE_delta_vs_H9")))]
        if not finite:
            continue
        best = min(finite, key=lambda row: _to_float(row.get("ATE_delta_vs_H9")))
        best_seg = min(
            finite,
            key=lambda row: _to_float(row.get("intersection_200_300_delta_vs_H9"))
            if math.isfinite(_to_float(row.get("intersection_200_300_delta_vs_H9")))
            else float("inf"),
        )
        out.append({
            "chunk_id": chunk_id,
            "horizon": horizon,
            "best_ATE_candidate": best.get("candidate_id"),
            "best_ATE_delta_vs_H9": best.get("ATE_delta_vs_H9"),
            "best_ATE_horizon": best.get("ATE_horizon"),
            "best_200_300_candidate": best_seg.get("candidate_id"),
            "best_200_300_delta_vs_H9": best_seg.get("intersection_200_300_delta_vs_H9"),
            "best_downstream_proxy_delta_vs_H9": best_seg.get("intersection_400_600_delta_vs_H9"),
        })
    return out


def _gate_summary(rows: List[Dict[str, object]]) -> Dict[str, object]:
    candidates = [row for row in rows if row.get("candidate_id") != "K1_H9" and int(row.get("horizon", 0)) in {8, 10}]
    best_ate = min(
        candidates,
        key=lambda row: _to_float(row.get("ATE_delta_vs_H9")) if math.isfinite(_to_float(row.get("ATE_delta_vs_H9"))) else float("inf"),
        default=None,
    )
    finite_seg = [
        row for row in candidates
        if math.isfinite(_to_float(row.get("intersection_200_300_delta_vs_H9")))
    ]
    best_seg = min(
        finite_seg,
        key=lambda row: _to_float(row.get("intersection_200_300_delta_vs_H9")),
        default=None,
    )

    def no_downstream_regression(row: Optional[Mapping[str, object]], limit: float = 1.0) -> bool:
        if row is None:
            return False
        downstream = _to_float(row.get("intersection_400_600_delta_vs_H9"))
        return (not math.isfinite(downstream)) or downstream <= limit

    pass_gate = (
        best_ate is not None
        and _to_float(best_ate.get("ATE_delta_vs_H9")) <= -2.5
        and no_downstream_regression(best_ate)
    ) or (
        best_seg is not None
        and _to_float(best_seg.get("intersection_200_300_delta_vs_H9")) <= -5.0
        and no_downstream_regression(best_seg)
    )
    weak_gate = (
        not pass_gate
        and best_ate is not None
        and _to_float(best_ate.get("ATE_delta_vs_H9")) <= -1.5
        and no_downstream_regression(best_ate)
    )
    status = "pass" if pass_gate else ("weak" if weak_gate else "fail")
    return {
        "phase": "Phase 1 horizon expansion",
        "status": status,
        "best_h8_h10_ATE_candidate": best_ate.get("candidate_id") if best_ate else "",
        "best_h8_h10_ATE_chunk": best_ate.get("chunk_id") if best_ate else "",
        "best_h8_h10_ATE_horizon": best_ate.get("horizon") if best_ate else "",
        "best_h8_h10_ATE_delta_vs_H9": best_ate.get("ATE_delta_vs_H9") if best_ate else float("nan"),
        "best_200_300_candidate": best_seg.get("candidate_id") if best_seg else "",
        "best_200_300_chunk": best_seg.get("chunk_id") if best_seg else "",
        "best_200_300_horizon": best_seg.get("horizon") if best_seg else "",
        "best_200_300_delta_vs_H9": best_seg.get("intersection_200_300_delta_vs_H9") if best_seg else float("nan"),
        "selector_allowed": False,
        "full_online_validation_allowed": False,
        "gate_rule": "pass if h8/h10 ATE delta <= -2.5m or [200,300) delta <= -5m with downstream proxy <= +1m",
    }


def _write_markdown(path: Path, rows: List[Dict[str, object]], summary: Mapping[str, object], best_rows: List[Dict[str, object]]) -> None:
    lines = [
        "# ACL2 v17 Horizon Expansion Report",
        "",
        "All deltas are recomputed against H9 on the same candidate frame intersection.",
        "",
        "## Gate",
        "",
        f"Status: `{summary.get('status')}`",
        "",
        f"Best h8/h10 ATE delta: `{summary.get('best_h8_h10_ATE_delta_vs_H9')}` "
        f"({summary.get('best_h8_h10_ATE_candidate')} chunk {summary.get('best_h8_h10_ATE_chunk')} h{summary.get('best_h8_h10_ATE_horizon')})",
        "",
        f"Best [200,300) delta: `{summary.get('best_200_300_delta_vs_H9')}` "
        f"({summary.get('best_200_300_candidate')} chunk {summary.get('best_200_300_chunk')} h{summary.get('best_200_300_horizon')})",
        "",
        "Selector allowed: `false` unless this report is later superseded by Phase 2/3/4 oracle gate.",
        "",
        "## Best By Chunk/Horizon",
        "",
        "| Chunk | Horizon | Best ATE candidate | ATE delta | Best [200,300) candidate | [200,300) delta | Downstream proxy |",
        "|---:|---:|---|---:|---|---:|---:|",
    ]
    for row in best_rows:
        lines.append(
            f"| {row.get('chunk_id')} | {row.get('horizon')} | `{row.get('best_ATE_candidate')}` | "
            f"{_to_float(row.get('best_ATE_delta_vs_H9')):.6f} | `{row.get('best_200_300_candidate')}` | "
            f"{_to_float(row.get('best_200_300_delta_vs_H9')):.6f} | "
            f"{_to_float(row.get('best_downstream_proxy_delta_vs_H9')):.6f} |"
        )
    lines += [
        "",
        "## Boundaries",
        "",
        "- `K25/K26` are online scale-state diagnostics and are not counted as TTT write success.",
        "- All rows are short-rollout oracle diagnostics; no row counts as full online deployable success.",
        "- No no-GT selector or full online validation is authorized by this report unless the gate is pass.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--gt", default="/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--chunk", action="append", type=int, default=[])
    parser.add_argument("--horizon", action="append", type=int, default=[])
    parser.add_argument(
        "--run-prefix",
        default="V17_P1R2",
        help="Run directory prefix. R1 used an invalid horizon end-frame convention; default to corrected R2.",
    )
    args = parser.parse_args()

    rollout_root = Path(args.rollout_root)
    out_dir = Path(args.out_dir)
    ref_run = Path(args.reference_run)
    candidates = args.candidate or sorted(FAMILY_BY_CANDIDATE)
    chunks = args.chunk or [5, 6, 10, 16]
    horizons = args.horizon or [5, 8, 10]

    gt_frames, gt_poses, gt_pos = _load_kitti_gt(Path(args.gt))
    if gt_frames.size != gt_pos.shape[0]:
        raise ValueError("Unexpected GT frame layout")

    h9_frames, h9_poses, _h9_pos = _load_tum_prediction(ref_run / "01.txt", gt_pos.shape[0])
    h9_lookup = _pose_lookup(h9_frames.astype(np.int64), h9_poses)
    h9_hmc_hash = _hash_map(ref_run / "hmc_state_hash.jsonl", value_key="hash_H_next")
    h9_merge_hash = _hash_map(ref_run / "merge_state_hash.jsonl", value_key="state_hash", include_kind=True)

    rows: List[Dict[str, object]] = []
    segment_rows: List[Dict[str, object]] = []
    downstream_rows: List[Dict[str, object]] = []
    manifest_rows: List[Dict[str, object]] = []

    for chunk_id in chunks:
        for horizon in horizons:
            for candidate_id in candidates:
                run_dir = _run_dir(rollout_root, args.run_prefix, candidate_id, chunk_id, horizon)
                traj = run_dir / "01.txt"
                if not traj.exists():
                    continue
                frames, poses, _pos = _load_tum_prediction(traj, gt_pos.shape[0])
                frames = frames.astype(np.int64)
                rollout_frame_start = int(frames[0])
                rollout_frame_end_inclusive = int(frames[-1])
                rollout_frame_count = int(frames.shape[0])
                eval_start, eval_end = _eval_window(chunk_id, horizon)
                eval_mask = (frames >= eval_start) & (frames < eval_end)
                if int(eval_mask.sum()) < 3:
                    continue
                frames = frames[eval_mask]
                poses = poses[eval_mask]
                h9_subset_poses = []
                for frame in frames:
                    ref_pose = h9_lookup.get(int(frame))
                    if ref_pose is None:
                        raise KeyError(f"H9 reference lacks frame {int(frame)} for {run_dir}")
                    h9_subset_poses.append(ref_pose)
                h9_subset = np.stack(h9_subset_poses, axis=0)

                aligned, metrics = _align_metrics(frames, poses, gt_poses, gt_pos)
                h9_aligned, h9_metrics = _align_metrics(frames, h9_subset, gt_poses, gt_pos)
                raw_max_abs, raw_max_trans, timestamp_equal = _raw_diff(frames, poses, h9_lookup)

                segs = {
                    "200_300": (200, 300),
                    "200_400": (200, 400),
                    "400_600": (400, 600),
                }
                seg_metrics: Dict[str, float] = {}
                for name, (start, end) in segs.items():
                    cand_ate = _segment_ate(frames, aligned, gt_pos, start, end)
                    h9_ate = _segment_ate(frames, h9_aligned, gt_pos, start, end)
                    delta = cand_ate - h9_ate if math.isfinite(cand_ate) and math.isfinite(h9_ate) else float("nan")
                    seg_metrics[f"intersection_{name}_ATE"] = cand_ate
                    seg_metrics[f"intersection_{name}_H9_ATE"] = h9_ate
                    seg_metrics[f"intersection_{name}_delta_vs_H9"] = delta
                    segment_rows.append({
                        "candidate_id": candidate_id,
                        "chunk_id": chunk_id,
                        "horizon": horizon,
                        "segment": f"[{start},{end})",
                        "candidate_ATE": cand_ate,
                        "H9_ATE": h9_ate,
                        "delta_vs_H9": delta,
                    })

                hmc_map = _hash_map(run_dir / "hmc_state_hash.jsonl", value_key="hash_H_next")
                merge_map = _hash_map(run_dir / "merge_state_hash.jsonl", value_key="state_hash", include_kind=True)
                hmc_mismatch, hmc_compared = _hash_mismatch_count(hmc_map, h9_hmc_hash)
                merge_mismatch, merge_compared = _hash_mismatch_count(merge_map, h9_merge_hash)

                row: Dict[str, object] = {
                    "candidate_id": candidate_id,
                    "family": FAMILY_BY_CANDIDATE.get(candidate_id, "unknown"),
                    "chunk_id": chunk_id,
                    "horizon": horizon,
                    "rollout_frame_start": rollout_frame_start,
                    "rollout_frame_end_inclusive": rollout_frame_end_inclusive,
                    "rollout_frame_count": rollout_frame_count,
                    "eval_frame_start": int(frames[0]),
                    "eval_frame_end_inclusive": int(frames[-1]),
                    "eval_frame_end_exclusive": eval_end,
                    "eval_frame_count": int(frames.shape[0]),
                    "ATE_horizon": metrics["ATE_horizon"],
                    "ATE_H9_intersection": h9_metrics["ATE_horizon"],
                    "ATE_delta_vs_H9": metrics["ATE_horizon"] - h9_metrics["ATE_horizon"],
                    "alignment_scale": metrics["alignment_scale"],
                    "alignment_scale_H9_intersection": h9_metrics["alignment_scale"],
                    "alignment_scale_delta_vs_H9": metrics["alignment_scale"] - h9_metrics["alignment_scale"],
                    "Rot_horizon": metrics["Rot_horizon"],
                    "Rot_H9_intersection": h9_metrics["Rot_horizon"],
                    "FinalErr_horizon": metrics["FinalErr_horizon"],
                    "FinalErr_H9_intersection": h9_metrics["FinalErr_horizon"],
                    **seg_metrics,
                    "raw_pose_max_abs_diff": raw_max_abs,
                    "raw_trans_max_diff": raw_max_trans,
                    "timestamp_mapping_equal": timestamp_equal,
                    "hmc_rows": _line_count(run_dir / "hmc_state_hash.jsonl"),
                    "hmc_hash_mismatch": hmc_mismatch,
                    "hmc_hash_rows_compared": hmc_compared,
                    "merge_rows": _line_count(run_dir / "merge_state_hash.jsonl"),
                    "merge_hash_mismatch": merge_mismatch,
                    "merge_hash_rows_compared": merge_compared,
                    "runtime_sec": _runtime_sec(run_dir),
                    "run_done": _run_done(run_dir),
                    "diagnostic_only_short_rollout": True,
                    "uses_gt_runtime_action": _uses_gt_runtime_action(candidate_id),
                    "counts_as_deployable_if_selected": _is_deployable_ttt_write_candidate(candidate_id),
                    "counts_as_ttt_write_success": False,
                    "run_dir": str(run_dir),
                }
                rows.append(row)
                downstream_rows.append({
                    "candidate_id": candidate_id,
                    "chunk_id": chunk_id,
                    "horizon": horizon,
                    "downstream_proxy_segment": "[400,600)",
                    "downstream_proxy_ATE": row["intersection_400_600_ATE"],
                    "downstream_proxy_H9_ATE": row["intersection_400_600_H9_ATE"],
                    "downstream_proxy_delta_vs_H9": row["intersection_400_600_delta_vs_H9"],
                    "downstream_proxy_regression_gt_1m": (
                        math.isfinite(_to_float(row["intersection_400_600_delta_vs_H9"]))
                        and _to_float(row["intersection_400_600_delta_vs_H9"]) > 1.0
                    ),
                })
                manifest_rows.append({
                    "candidate_id": candidate_id,
                    "chunk_id": chunk_id,
                    "horizon": horizon,
                    "parent_run_id": "H9_P0_V16_R2",
                    "family": FAMILY_BY_CANDIDATE.get(candidate_id, "unknown"),
                    "diagnostic_only_short_rollout": True,
                    "uses_gt_runtime_action": _uses_gt_runtime_action(candidate_id),
                    "counts_as_deployable_if_selected": _is_deployable_ttt_write_candidate(candidate_id),
                    "counts_as_ttt_write_success": False,
                    "run_dir": str(run_dir),
                })

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "short_rollout_metrics_h5_h8_h10.csv", rows)
    _write_json(out_dir / "short_rollout_metrics_h5_h8_h10.json", rows)
    delta_rows = [
        {
            "candidate_id": row["candidate_id"],
            "chunk_id": row["chunk_id"],
            "horizon": row["horizon"],
            "ATE_delta_vs_H9": row["ATE_delta_vs_H9"],
            "intersection_200_300_delta_vs_H9": row["intersection_200_300_delta_vs_H9"],
            "intersection_400_600_delta_vs_H9": row["intersection_400_600_delta_vs_H9"],
            "raw_trans_max_diff": row["raw_trans_max_diff"],
            "run_dir": row["run_dir"],
        }
        for row in rows
    ]
    _write_csv(out_dir / "candidate_vs_H9_delta_by_horizon.csv", delta_rows)
    _write_json(out_dir / "candidate_vs_H9_delta_by_horizon.json", delta_rows)
    _write_csv(out_dir / "candidate_window_segment_intersection.csv", segment_rows)
    _write_json(out_dir / "candidate_window_segment_intersection.json", segment_rows)
    _write_csv(out_dir / "candidate_downstream_proxy.csv", downstream_rows)
    _write_json(out_dir / "candidate_downstream_proxy.json", downstream_rows)
    _write_csv(out_dir / "candidate_manifest_horizon.csv", manifest_rows)
    _write_json(out_dir / "candidate_manifest_horizon.json", manifest_rows)

    heat_rows = [
        {
            "candidate_id": row["candidate_id"],
            "chunk_id": row["chunk_id"],
            "horizon": row["horizon"],
            "value": row["ATE_delta_vs_H9"],
        }
        for row in rows
    ]
    _write_csv(out_dir / "horizon_gain_heatmap.csv", heat_rows)
    seg_heat_rows = [
        {
            "candidate_id": row["candidate_id"],
            "chunk_id": row["chunk_id"],
            "horizon": row["horizon"],
            "segment": row["segment"],
            "value": row["delta_vs_H9"],
        }
        for row in segment_rows
    ]
    _write_csv(out_dir / "segment_gain_heatmap.csv", seg_heat_rows)

    best_rows = _best_rows(rows)
    _write_csv(out_dir / "best_by_chunk_horizon.csv", best_rows)
    _write_json(out_dir / "best_by_chunk_horizon.json", best_rows)
    summary = [_gate_summary(rows)]
    _write_csv(out_dir / "phase1_horizon_gate_summary.csv", summary)
    _write_json(out_dir / "phase1_horizon_gate_summary.json", summary)
    _write_markdown(out_dir / "acl2_v17_horizon_expansion_report.md", rows, summary[0], best_rows)


if __name__ == "__main__":
    main()
