#!/usr/bin/env python3
"""Aggregate ACL2 v16 candidate-bank short-rollout oracle rows."""

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
    _apply_alignment,
    _load_kitti_gt,
    _load_tum_prediction,
    _rmse,
    _umeyama_sim3,
)


FAMILY_BY_CANDIDATE = {
    "K13_ORTHO_RHO025_W0": "ortho_suppress_rho025",
    "K14_TTGR_ZERO_ORTHO_W0": "ttgr_zero_ortho_suppress",
    "K15_COSCAP_W0_025_050": "cosine_cap",
    "K16_COMMIT_CONFLICT_NATIVE025": "commit_filter_conflict_native025",
    "K17_COMMIT_CONFLICT_NATIVE050_Q90": "commit_filter_conflict_native050_q90",
    "K18_OVERLAP_PSEUDO_V050": "overlap_pseudo_v050",
    "K19_ORTHO_SUPPRESS_W1": "ortho_suppress_w1",
    "K20_ORTHO_SUPPRESS_W2": "ortho_suppress_w2",
    "K21_ORTHO_SUPPRESS_ALL": "ortho_suppress_all",
    "K22_GT_SCALE_PROJ_BASE": "gt_scale_projection_base",
    "K23_GT_SCALE_PROJ_STR2": "gt_scale_projection_strength2",
    "K24_GT_SCALE_PROJ_DOUBLE": "gt_scale_projection_double_gamma",
    "K25_ONLINE_SCALE_OVERLAP_STEP": "online_scale_state_overlap_step",
    "K26_ONLINE_SCALE_OVERLAP_TIGHT": "online_scale_state_overlap_tight",
    "K27_ONLINE_SCALE_INV_STEP": "online_scale_state_overlap_inverse",
    "K28_ONLINE_SCALE_INV_TIGHT": "online_scale_state_overlap_inverse_tight",
}


def _uses_gt_runtime_action(candidate_id: str) -> bool:
    return candidate_id.startswith("K22_") or candidate_id.startswith("K23_") or candidate_id.startswith("K24_")


def _is_ttt_write_candidate(candidate_id: str) -> bool:
    return not (
        _uses_gt_runtime_action(candidate_id)
        or candidate_id.startswith("K25_")
        or candidate_id.startswith("K26_")
        or candidate_id.startswith("K27_")
        or candidate_id.startswith("K28_")
    )


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _fixed_segment(frames: np.ndarray, aligned: np.ndarray, gt_pos_by_frame: np.ndarray, start: int, end: int) -> float:
    mask = (frames >= start) & (frames < end)
    if int(mask.sum()) < 3:
        return float("nan")
    err = aligned[mask, :3, 3] - gt_pos_by_frame[frames[mask]]
    return _rmse(np.linalg.norm(err, axis=1))


def _metrics_for_raw(frames: np.ndarray, raw_poses: np.ndarray, raw_pos: np.ndarray, gt_pos: np.ndarray) -> Dict[str, float]:
    frames = frames.astype(np.int64)
    matched_gt = gt_pos[frames]
    scale, rot, trans = _umeyama_sim3(raw_pos, matched_gt, with_scale=True)
    aligned = _apply_alignment(raw_poses, scale, rot, trans)
    err = aligned[:, :3, 3] - matched_gt
    norm = np.linalg.norm(err, axis=1)
    metrics = {
        "future_h3_ATE": _rmse(norm),
        "seg_200_300": _fixed_segment(frames, aligned, gt_pos, 200, 300),
        "seg_400_600": _fixed_segment(frames, aligned, gt_pos, 400, 600),
    }
    return metrics


def _load_aligned_metrics(run_dir: Path, gt_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    frames, raw_poses, raw_pos = _load_tum_prediction(run_dir / "01.txt", gt_pos.shape[0])
    metrics = _metrics_for_raw(frames, raw_poses, raw_pos, gt_pos)
    return frames, raw_poses, metrics


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


def _hmc_rows(run_dir: Path) -> int:
    path = run_dir / "hmc_state_hash.jsonl"
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


def _candidate_run_dir(rollout_root: Path, candidate_id: str, chunk_id: int) -> Path:
    name = f"V16_P2_{candidate_id}_chunk{chunk_id}_h3_globalgate_H9parent_SWKS3"
    return rollout_root / name


def _baseline_by_chunk(rows: Iterable[Mapping[str, str]]) -> Dict[int, Mapping[str, str]]:
    out: Dict[int, Mapping[str, str]] = {}
    for row in rows:
        if row.get("candidate_id") == "K1_H9":
            out[int(row["chunk_id"])] = row
    return out


def _serialize_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        rec: Dict[str, object] = {}
        for key, value in row.items():
            if isinstance(value, float) and math.isnan(value):
                rec[key] = None
            else:
                rec[key] = value
        out.append(rec)
    return out


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
    path.write_text(
        json.dumps(_serialize_rows(rows), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _manifest_row(candidate_id: str, chunk_id: int, run_dir: Path) -> Dict[str, object]:
    uses_gt = _uses_gt_runtime_action(candidate_id)
    return {
        "candidate_id": candidate_id,
        "chunk_id": chunk_id,
        "parent_run_id": "H9_P0_V16_R2",
        "action_family": FAMILY_BY_CANDIDATE.get(candidate_id, "unknown"),
        "run_dir": str(run_dir),
        "counts_as_deployable_if_selected": _is_ttt_write_candidate(candidate_id),
        "uses_gt_runtime_action": uses_gt,
        "diagnostic_only_short_rollout": True,
        "counts_as_ttt_write_success": False,
    }


def _summary_by_chunk(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    chunks = sorted({int(row["chunk_id"]) for row in rows})
    out: List[Dict[str, object]] = []
    for chunk_id in chunks:
        chunk_rows = [
            row for row in rows
            if int(row["chunk_id"]) == chunk_id and row.get("candidate_id") != "K1_H9"
        ]
        finite = [
            row for row in chunk_rows
            if math.isfinite(_to_float(row.get("future_h3_ATE")))
        ]
        if not finite:
            continue
        best = min(finite, key=lambda row: _to_float(row.get("future_h3_ATE")))
        out.append({
            "chunk_id": chunk_id,
            "best_candidate": best.get("candidate_id"),
            "best_ate": best.get("future_h3_ATE"),
            "best_ate_delta_vs_H9": best.get("future_h3_ATE_delta_vs_H9"),
            "best_seg_200_300_delta_vs_H9": best.get("future_h3_seg_200_300_delta_vs_H9"),
            "best_seg_400_600_delta_vs_H9": best.get("future_h3_seg_400_600_delta_vs_H9"),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-table", required=True)
    parser.add_argument("--rollout-root", required=True)
    parser.add_argument("--gt", default="/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--chunk", action="append", type=int, default=[])
    parser.add_argument("--out-table", required=True)
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--out-manifest", required=True)
    args = parser.parse_args()

    base_table = Path(args.base_table)
    rollout_root = Path(args.rollout_root)
    gt_path = Path(args.gt)
    _gt_frames, _gt_poses, gt_pos = _load_kitti_gt(gt_path)
    base_rows_str = _read_csv(base_table)
    baselines = _baseline_by_chunk(base_rows_str)
    baseline_run_dir = Path(next(iter(baselines.values()))["run_dir"])
    h9_frames, h9_poses, h9_pos = _load_tum_prediction(baseline_run_dir / "01.txt", gt_pos.shape[0])
    h9_frames = h9_frames.astype(np.int64)
    h9_lookup = _pose_lookup(h9_frames, h9_poses)
    h9_pos_lookup = {int(frame): pos for frame, pos in zip(h9_frames, h9_pos)}

    rows: List[Dict[str, object]] = [dict(row) for row in base_rows_str]
    manifest: List[Dict[str, object]] = []

    for chunk_id in args.chunk:
        if chunk_id not in baselines:
            raise KeyError(f"Missing K1_H9 baseline for chunk {chunk_id} in {base_table}")
        base = baselines[chunk_id]
        for candidate_id in args.candidate:
            run_dir = _candidate_run_dir(rollout_root, candidate_id, chunk_id)
            if not (run_dir / "01.txt").exists():
                continue
            frames, poses, metrics = _load_aligned_metrics(run_dir, gt_pos)
            base_raw_poses = []
            base_raw_pos = []
            for frame in frames:
                pose = h9_lookup.get(int(frame))
                pos = h9_pos_lookup.get(int(frame))
                if pose is None or pos is None:
                    raise KeyError(f"H9 baseline is missing frame {int(frame)} for chunk {chunk_id}")
                base_raw_poses.append(pose)
                base_raw_pos.append(pos)
            base_metrics = _metrics_for_raw(
                frames,
                np.stack(base_raw_poses, axis=0),
                np.stack(base_raw_pos, axis=0),
                gt_pos,
            )
            base_ate = base_metrics["future_h3_ATE"]
            base_seg_200_300 = base_metrics["seg_200_300"]
            base_seg_400_600 = base_metrics["seg_400_600"]
            raw_max_abs, raw_max_trans, timestamp_equal = _raw_diff(frames, poses, h9_lookup)
            ate = metrics["future_h3_ATE"]
            seg_200_300 = metrics["seg_200_300"]
            seg_400_600 = metrics["seg_400_600"]
            row: Dict[str, object] = {
                "candidate_id": candidate_id,
                "chunk_id": chunk_id,
                "family": FAMILY_BY_CANDIDATE.get(candidate_id, "unknown"),
                "future_h3_ATE": ate,
                "future_h3_ATE_delta_vs_H9": ate - base_ate,
                "future_h3_seg_200_300_delta_vs_H9": (
                    seg_200_300 - base_seg_200_300
                    if math.isfinite(seg_200_300) and math.isfinite(base_seg_200_300)
                    else float("nan")
                ),
                "future_h3_seg_400_600_delta_vs_H9": (
                    seg_400_600 - base_seg_400_600
                    if math.isfinite(seg_400_600) and math.isfinite(base_seg_400_600)
                    else float("nan")
                ),
                "raw_pose_max_abs_diff_vs_H9": raw_max_abs,
                "raw_pose_max_trans_diff_vs_H9": raw_max_trans,
                "timestamp_mapping_equal": timestamp_equal,
                "hmc_rows": _hmc_rows(run_dir),
                "runtime_sec": _runtime_sec(run_dir),
                "diagnostic_only_short_rollout": True,
                "counts_as_ttt_write_success": False,
                "uses_gt_runtime_action": _uses_gt_runtime_action(candidate_id),
                "run_dir": str(run_dir),
            }
            rows.append(row)
            manifest.append(_manifest_row(candidate_id, chunk_id, run_dir))

    _write_csv(Path(args.out_table), rows)
    _write_json(Path(args.out_table).with_suffix(".json"), rows)
    summary = _summary_by_chunk(rows)
    _write_csv(Path(args.out_summary), summary)
    _write_json(Path(args.out_summary).with_suffix(".json"), summary)
    _write_csv(Path(args.out_manifest), manifest)
    _write_json(Path(args.out_manifest).with_suffix(".json"), manifest)


if __name__ == "__main__":
    main()
