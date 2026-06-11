#!/usr/bin/env python3
"""Full-sequence drift autopsy for ACL2 v53 clean adaptive TTT runs.

This script only reads landed rollout artifacts. It does not fill missing
measurements; missing fields are emitted as empty values in the reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v18_true_action_report import _align_metrics, _load_kitti_gt, _load_tum_prediction  # noqa: E402
from tools.v42_full_online_report import _as_positions, _ate, _rolling_windows  # noqa: E402
from tools.v47_adaptive_ttt_writer_report import _debug_stats  # noqa: E402


C9_P0_ATE = 33.76294210291885
DEFAULT_RESULT_ROOT = REPO_ROOT / "results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9"
DEFAULT_GT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
DEFAULT_C9_RUN = (
    REPO_ROOT
    / "results/kitti01_hmc_v2/acl2_v15_ttt_repro_causal_sandbox_target25/"
    "phase0_repro/V15_P0_A2_C9_REPEAT_no_state_save_SWKS3"
)


DEFAULT_RUNS: Sequence[Tuple[str, Path]] = (
    ("C9_REF", DEFAULT_C9_RUN),
    (
        "V53_A_BEST",
        DEFAULT_RESULT_ROOT / "phase4_full/rollouts/V53_FULL_A_SCGAMMASPLIT_AW110",
    ),
    (
        "V53_H8_SCREEN_384",
        DEFAULT_RESULT_ROOT
        / "phase5_clean_continuation_screen/rollouts/"
        "V53_CONT_SCREEN_H8_SCALESTATE_ALPHA3_SAMPLE2048_AW110_384F",
    ),
    (
        "V53_H8_FULL",
        DEFAULT_RESULT_ROOT
        / "phase5_clean_continuation_full/rollouts/"
        "V53_CONT_FULL_H8_SCALESTATE_ALPHA3_SAMPLE2048_AW110",
    ),
    (
        "V53_H13_FULL",
        DEFAULT_RESULT_ROOT
        / "phase5_clean_continuation_full/rollouts/"
        "V53_CONT_FULL_H13_SCALESTATE_ALPHA4_ONLINESCALE_OVERLAP_AW110",
    ),
)


def _clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _mean(values: Iterable[Any]) -> Optional[float]:
    vals = [_safe_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def _p90(values: Iterable[Any]) -> Optional[float]:
    vals = [_safe_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.percentile(vals, 90)) if vals else None


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


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
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 6) -> str:
    number = _safe_float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _load_run_poses(run_dir: Path, gt_poses: np.ndarray, gt_pos: np.ndarray) -> Dict[str, Any]:
    pred_path = run_dir / "01.txt"
    if not pred_path.is_file():
        return {"pose_status": "missing_prediction", "frames": np.asarray([], dtype=np.int64)}
    frames, raw_poses, _ = _load_tum_prediction(pred_path, gt_pos.shape[0])
    if frames.size == 0:
        return {"pose_status": "empty_prediction", "frames": frames}
    aligned, metrics = _align_metrics(frames.astype(np.int64), raw_poses, gt_poses, gt_pos)
    return {
        "pose_status": "done",
        "frames": frames.astype(np.int64),
        "raw_poses": raw_poses,
        "aligned": aligned,
        "aligned_pos": _as_positions(aligned),
        "metrics": metrics,
    }


def _subset_self_aligned_ate(
    frames: np.ndarray,
    raw_poses: np.ndarray,
    gt_poses: np.ndarray,
    gt_pos: np.ndarray,
    frame_min: int,
    frame_max_exclusive: int,
) -> Optional[float]:
    mask = (frames >= frame_min) & (frames < frame_max_exclusive)
    if int(mask.sum()) < 3:
        return None
    _aligned, metrics = _align_metrics(frames[mask].astype(np.int64), raw_poses[mask], gt_poses, gt_pos)
    return _safe_float(metrics.get("ATE_horizon"), default=float("nan"))


def _segment_error(
    frames: np.ndarray,
    aligned_pos: np.ndarray,
    gt_pos: np.ndarray,
    frame_min: int,
    frame_max_exclusive: int,
) -> Dict[str, Any]:
    mask = (frames >= frame_min) & (frames < frame_max_exclusive)
    count = int(mask.sum())
    out: Dict[str, Any] = {
        "frame_min": frame_min,
        "frame_max_exclusive": frame_max_exclusive,
        "frame_count": count,
    }
    if count <= 0:
        out.update({"rmse": None, "mean": None, "p90": None, "max": None})
        return out
    err = np.linalg.norm(aligned_pos[mask] - gt_pos[frames[mask].astype(np.int64)], axis=1)
    out.update(
        {
            "rmse": float(np.sqrt(np.mean(err * err))),
            "mean": float(np.mean(err)),
            "p90": float(np.percentile(err, 90)),
            "max": float(np.max(err)),
        }
    )
    return out


def _top_rolling_windows(
    run_key: str,
    frames: np.ndarray,
    aligned_pos: np.ndarray,
    gt_pos: np.ndarray,
    width: int,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    values = _rolling_windows(frames.astype(np.int64), aligned_pos, gt_pos, width)
    rows: List[Dict[str, Any]] = []
    for start, ate in sorted(values.items(), key=lambda item: item[1], reverse=True)[:limit]:
        rows.append(
            {
                "run_key": run_key,
                "width": width,
                "start_frame": int(start),
                "end_frame": int(start + width),
                "rolling_ate": float(ate),
            }
        )
    return rows


def _per_chunk_rows(
    run_key: str,
    run_dir: Path,
    frames: np.ndarray,
    aligned_pos: np.ndarray,
    gt_pos: np.ndarray,
) -> List[Dict[str, Any]]:
    state_rows = {int(row.get("chunk_idx")): row for row in _read_jsonl(run_dir / "hmc_state_hash.jsonl") if row.get("chunk_idx") is not None}
    raw_rows = {int(row.get("chunk_idx")): row for row in _read_jsonl(run_dir / "raw_prediction_buffer_summary.jsonl") if row.get("chunk_idx") is not None}
    timing = _read_json(run_dir / "timing_summary.json")
    timing_rows = {
        int(row.get("chunk_idx")): row
        for row in timing.get("chunks", [])
        if isinstance(row, dict) and row.get("chunk_idx") is not None
    }
    chunk_ids = sorted(set(state_rows) | set(raw_rows) | set(timing_rows))
    rows: List[Dict[str, Any]] = []
    for chunk_idx in chunk_ids:
        state = state_rows.get(chunk_idx, {})
        raw = raw_rows.get(chunk_idx, {})
        time = timing_rows.get(chunk_idx, {})
        start = int(state.get("start_frame", raw.get("global_start", time.get("start_frame", -1))) or -1)
        end = int(state.get("end_frame", raw.get("global_end", time.get("end_frame", -1))) or -1)
        row: Dict[str, Any] = {
            "run_key": run_key,
            "chunk_idx": chunk_idx,
            "start_frame": start if start >= 0 else None,
            "end_frame": end if end >= 0 else None,
        }
        if start >= 0 and end >= 0 and frames.size and aligned_pos.size:
            seg = _segment_error(frames, aligned_pos, gt_pos, start, end + 1)
            row.update({f"global_aligned_{k}": v for k, v in seg.items() if k not in ("frame_min", "frame_max_exclusive")})
        for key in (
            "chunk_total_seconds",
            "pass1_probe_seconds",
            "stage_b_seconds",
            "stage_d_seconds",
            "pass2_control_seconds",
            "probe_ttt_write_seconds",
        ):
            row[key] = time.get(key)
        for key in (
            "auxgeo_tri_replay_pos_mass_mean",
            "auxgeo_tri_replay_neu_mass_mean",
            "auxgeo_tri_replay_neg_mass_mean",
            "auxgeo_tri_replay_applied_layer_count",
            "memory_ttt_mean_rel_diff",
            "memory_ttt_w0_mean_rel_diff",
            "memory_ttt_w1_mean_rel_diff",
            "memory_ttt_w2_mean_rel_diff",
            "pass1_pass2_pose_t_mean",
            "pass1_pass2_pose_t_max",
            "pass1_pass2_pose_r_deg_mean",
            "pass1_pass2_world_points_l1_mean",
            "prior_mean_D_tok",
            "prior_q90_D_tok",
            "prior_anchor_collision",
            "prior_fragmentation",
            "prior_cue_quality_pass",
        ):
            row[key] = state.get(key)
        for key in (
            "online_scale_aligned_step_median",
            "online_scale_ema_step_median_after",
            "online_scale_modifier",
            "online_scale_raw_log_ratio",
            "online_scale_state_mode",
            "transform_reason",
            "transform_scale_value",
            "transform_trans_norm",
            "transform_rot_trace",
        ):
            row[key] = raw.get(key)
        rows.append(row)
    return rows


def _prefix_raw_pose_diff(
    left_key: str,
    left: Mapping[str, Any],
    right_key: str,
    right: Mapping[str, Any],
    frame_max_exclusive: int,
) -> Dict[str, Any]:
    lf = left.get("frames")
    rf = right.get("frames")
    lp = left.get("raw_poses")
    rp = right.get("raw_poses")
    if not isinstance(lf, np.ndarray) or not isinstance(rf, np.ndarray) or not isinstance(lp, np.ndarray) or not isinstance(rp, np.ndarray):
        return {"left_key": left_key, "right_key": right_key, "status": "missing_pose"}
    left_idx = {int(frame): i for i, frame in enumerate(lf) if int(frame) < frame_max_exclusive}
    right_idx = {int(frame): i for i, frame in enumerate(rf) if int(frame) < frame_max_exclusive}
    common = sorted(set(left_idx) & set(right_idx))
    if not common:
        return {"left_key": left_key, "right_key": right_key, "status": "no_common_frames"}
    trans_diffs = []
    rot_fro = []
    for frame in common:
        lpose = lp[left_idx[frame]]
        rpose = rp[right_idx[frame]]
        trans_diffs.append(float(np.linalg.norm(lpose[:3, 3] - rpose[:3, 3])))
        rot_fro.append(float(np.linalg.norm(lpose[:3, :3] - rpose[:3, :3])))
    return {
        "left_key": left_key,
        "right_key": right_key,
        "status": "done",
        "common_frames": len(common),
        "frame_max_exclusive": frame_max_exclusive,
        "translation_rmse": float(np.sqrt(np.mean(np.square(trans_diffs)))),
        "translation_max": float(np.max(trans_diffs)),
        "rotation_fro_mean": float(np.mean(rot_fro)),
        "rotation_fro_max": float(np.max(rot_fro)),
    }


def _run_summary(
    run_key: str,
    run_dir: Path,
    pose: Mapping[str, Any],
    gt_poses: np.ndarray,
    gt_pos: np.ndarray,
) -> Dict[str, Any]:
    debug = _debug_stats(run_dir) if run_dir.is_dir() else {}
    timing = _read_json(run_dir / "timing_summary.json")
    timing_chunks = timing.get("chunks", []) if isinstance(timing.get("chunks"), list) else []
    wall = _read_json(run_dir / "wall_time_summary.json")
    frames = pose.get("frames")
    raw_poses = pose.get("raw_poses")
    metrics = pose.get("metrics") if isinstance(pose.get("metrics"), dict) else {}
    out: Dict[str, Any] = {
        "run_key": run_key,
        "run_dir": str(run_dir),
        "pose_status": pose.get("pose_status"),
        "frames": int(frames.size) if isinstance(frames, np.ndarray) else 0,
        "ATE": metrics.get("ATE_horizon"),
        "Rot": metrics.get("Rot_horizon"),
        "FinalErr": metrics.get("FinalErr_horizon"),
        "alignment_scale": metrics.get("alignment_scale"),
        "delta_vs_C9_reference": (_safe_float(metrics.get("ATE_horizon")) - C9_P0_ATE if math.isfinite(_safe_float(metrics.get("ATE_horizon"))) else None),
        "wall_time_min": (_safe_float(wall.get("wall_seconds")) / 60.0 if math.isfinite(_safe_float(wall.get("wall_seconds"))) else None),
        "timing_chunks": len(timing_chunks),
        "chunk_total_seconds_mean": _mean(row.get("chunk_total_seconds") for row in timing_chunks if isinstance(row, dict)),
        "chunk_total_seconds_p90": _p90(row.get("chunk_total_seconds") for row in timing_chunks if isinstance(row, dict)),
        "probe_ttt_write_seconds_mean": _mean(row.get("probe_ttt_write_seconds") for row in timing_chunks if isinstance(row, dict)),
        "probe_ttt_write_seconds_p90": _p90(row.get("probe_ttt_write_seconds") for row in timing_chunks if isinstance(row, dict)),
    }
    out.update({f"debug_{k}": v for k, v in debug.items()})
    if isinstance(frames, np.ndarray) and isinstance(raw_poses, np.ndarray) and frames.size:
        out["prefix384_self_aligned_ATE"] = _subset_self_aligned_ate(frames, raw_poses, gt_poses, gt_pos, 0, 384)
        out["prefix410_self_aligned_ATE"] = _subset_self_aligned_ate(frames, raw_poses, gt_poses, gt_pos, 0, 410)
    if isinstance(frames, np.ndarray) and isinstance(pose.get("aligned_pos"), np.ndarray) and frames.size:
        aligned_pos = pose["aligned_pos"]
        for name, start, end in (
            ("seg0_000_384", 0, 384),
            ("seg1_384_700", 384, 700),
            ("seg2_700_end", 700, 20000),
        ):
            seg = _segment_error(frames, aligned_pos, gt_pos, start, end)
            for key in ("frame_count", "rmse", "mean", "p90", "max"):
                out[f"{name}_{key}"] = seg.get(key)
    return out


def _plot_chunk_metric(
    out_path: Path,
    chunk_rows: Sequence[Mapping[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in chunk_rows:
        if row.get(metric) is None:
            continue
        grouped.setdefault(str(row.get("run_key")), []).append(row)
    if not grouped:
        return
    plt.figure(figsize=(12, 6))
    for run_key, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda r: int(r.get("chunk_idx") or 0))
        xs = [int(r.get("chunk_idx") or 0) for r in rows]
        ys = [_safe_float(r.get(metric)) for r in rows]
        plt.plot(xs, ys, marker="o", linewidth=1.5, label=run_key)
    plt.xlabel("chunk_idx")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _write_markdown(
    path: Path,
    run_rows: Sequence[Mapping[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
    top_windows: Sequence[Mapping[str, Any]],
    prefix_diffs: Sequence[Mapping[str, Any]],
) -> None:
    lines: List[str] = []
    lines.append("# ACL2 v53 full-sequence drift autopsy")
    lines.append("")
    lines.append("Generated from landed artifacts only. Missing fields are left as NA; no measurements are fabricated.")
    lines.append("")
    lines.append("## Run Summary")
    lines.append("")
    lines.append("| run | frames | ATE | prefix384 self ATE | seg0 rmse | seg1 rmse | seg2 rmse | wall min | chunk mean | TTT mean | pos/neut/neg mass | gamma | lambda |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|")
    for row in run_rows:
        masses = (
            f"{_fmt(row.get('debug_ttt_positive_mass_mean'), 4)}/"
            f"{_fmt(row.get('debug_ttt_neutral_mass_mean'), 4)}/"
            f"{_fmt(row.get('debug_ttt_negative_mass_mean'), 4)}"
        )
        lines.append(
            f"| `{row.get('run_key')}` | {row.get('frames')} | {_fmt(row.get('ATE'))} | "
            f"{_fmt(row.get('prefix384_self_aligned_ATE'))} | {_fmt(row.get('seg0_000_384_rmse'))} | "
            f"{_fmt(row.get('seg1_384_700_rmse'))} | {_fmt(row.get('seg2_700_end_rmse'))} | "
            f"{_fmt(row.get('wall_time_min'))} | {_fmt(row.get('chunk_total_seconds_mean'))} | "
            f"{_fmt(row.get('probe_ttt_write_seconds_mean'))} | {masses} | "
            f"{_fmt(row.get('debug_adaptive_gamma_mean'), 6)} | {_fmt(row.get('debug_adaptive_neutral_lambda_mean'), 6)} |"
        )
    lines.append("")
    lines.append("## Screen/Full Prefix Consistency")
    lines.append("")
    lines.append("| left | right | status | common frames | trans rmse | trans max | rot fro mean |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for row in prefix_diffs:
        lines.append(
            f"| `{row.get('left_key')}` | `{row.get('right_key')}` | {row.get('status')} | "
            f"{row.get('common_frames', 'NA')} | {_fmt(row.get('translation_rmse'))} | "
            f"{_fmt(row.get('translation_max'))} | {_fmt(row.get('rotation_fro_mean'))} |"
        )
    lines.append("")
    lines.append("## Worst Rolling Windows")
    lines.append("")
    lines.append("| run | width | start | end | rolling ATE |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in top_windows[:30]:
        lines.append(
            f"| `{row.get('run_key')}` | {row.get('width')} | {row.get('start_frame')} | "
            f"{row.get('end_frame')} | {_fmt(row.get('rolling_ate'))} |"
        )
    lines.append("")
    lines.append("## Segment Table")
    lines.append("")
    lines.append("| run | segment | frames | rmse | mean | p90 | max |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in segment_rows:
        lines.append(
            f"| `{row.get('run_key')}` | `{row.get('segment')}` | {row.get('frame_count')} | "
            f"{_fmt(row.get('rmse'))} | {_fmt(row.get('mean'))} | {_fmt(row.get('p90'))} | "
            f"{_fmt(row.get('max'))} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_runs(values: Sequence[str]) -> List[Tuple[str, Path]]:
    if not values:
        return list(DEFAULT_RUNS)
    runs: List[Tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--run expects KEY=PATH, got {value!r}")
        key, path = value.split("=", 1)
        runs.append((key, Path(path)))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_RESULT_ROOT / "phase6_full_sequence_autopsy" / "report_R1"))
    parser.add_argument("--gt", default=str(DEFAULT_GT))
    parser.add_argument("--run", action="append", default=[], help="Run mapping KEY=PATH. Defaults to v53 key runs plus C9 ref.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    _gt_frames, gt_poses, gt_pos = _load_kitti_gt(Path(args.gt))
    run_specs = _parse_runs(args.run)

    pose_by_key: Dict[str, Dict[str, Any]] = {}
    run_rows: List[Dict[str, Any]] = []
    chunk_rows: List[Dict[str, Any]] = []
    segment_rows: List[Dict[str, Any]] = []
    top_windows: List[Dict[str, Any]] = []

    for run_key, run_dir in run_specs:
        pose = _load_run_poses(run_dir, gt_poses, gt_pos)
        pose_by_key[run_key] = pose
        run_rows.append(_run_summary(run_key, run_dir, pose, gt_poses, gt_pos))
        frames = pose.get("frames")
        aligned_pos = pose.get("aligned_pos")
        if isinstance(frames, np.ndarray) and isinstance(aligned_pos, np.ndarray) and frames.size:
            for segment, start, end in (
                ("000_384", 0, 384),
                ("384_700", 384, 700),
                ("700_end", 700, 20000),
            ):
                seg = _segment_error(frames, aligned_pos, gt_pos, start, end)
                segment_rows.append({"run_key": run_key, "segment": segment, **seg})
            for width in (50, 100, 200):
                top_windows.extend(_top_rolling_windows(run_key, frames, aligned_pos, gt_pos, width, limit=5))
            chunk_rows.extend(_per_chunk_rows(run_key, run_dir, frames, aligned_pos, gt_pos))

    prefix_diffs: List[Dict[str, Any]] = []
    if "V53_H8_SCREEN_384" in pose_by_key and "V53_H8_FULL" in pose_by_key:
        prefix_diffs.append(
            _prefix_raw_pose_diff(
                "V53_H8_SCREEN_384",
                pose_by_key["V53_H8_SCREEN_384"],
                "V53_H8_FULL",
                pose_by_key["V53_H8_FULL"],
                384,
            )
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "v53_full_sequence_drift_autopsy_by_run.csv", run_rows)
    _write_csv(out_dir / "v53_full_sequence_drift_autopsy_by_chunk.csv", chunk_rows)
    _write_csv(out_dir / "v53_full_sequence_drift_autopsy_segments.csv", segment_rows)
    _write_csv(out_dir / "v53_full_sequence_drift_autopsy_top_rolling_windows.csv", top_windows)
    _write_csv(out_dir / "v53_full_sequence_drift_autopsy_prefix_diff.csv", prefix_diffs)
    _write_json(
        out_dir / "v53_full_sequence_drift_autopsy_summary.json",
        {
            "runs": run_rows,
            "segments": segment_rows,
            "prefix_diffs": prefix_diffs,
            "top_windows": top_windows,
            "sources": [{"run_key": key, "run_dir": str(path)} for key, path in run_specs],
        },
    )
    _write_markdown(
        out_dir / "v53_full_sequence_drift_autopsy.md",
        run_rows,
        segment_rows,
        top_windows,
        prefix_diffs,
    )
    _plot_chunk_metric(
        out_dir / "v53_chunk_global_error_rmse_timeline.png",
        chunk_rows,
        "global_aligned_rmse",
        "global-aligned chunk RMSE",
        "Chunk error timeline",
    )
    _plot_chunk_metric(
        out_dir / "v53_chunk_role_positive_mass_timeline.png",
        chunk_rows,
        "auxgeo_tri_replay_pos_mass_mean",
        "positive role mass",
        "Positive role mass timeline",
    )
    _plot_chunk_metric(
        out_dir / "v53_chunk_ttt_write_seconds_timeline.png",
        chunk_rows,
        "probe_ttt_write_seconds",
        "probe TTT write seconds",
        "Probe TTT write runtime timeline",
    )
    _plot_chunk_metric(
        out_dir / "v53_chunk_pose_delta_timeline.png",
        chunk_rows,
        "pass1_pass2_pose_t_mean",
        "pass1/pass2 pose t mean",
        "Controlled pose delta timeline",
    )
    print(f"Wrote v53 full-sequence drift autopsy to {out_dir}")


if __name__ == "__main__":
    main()
