#!/usr/bin/env python3
"""Generate ACL2 v55 C9 schedule autopsy and fail-forward reports.

This report only reads landed rollout artifacts.  Missing measurements are left
empty; no metric is synthesized from a plan expectation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
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

from tools.v47_adaptive_ttt_writer_report import _debug_stats, _walk  # noqa: E402
from tools.v53_experiment_report import (  # noqa: E402
    C9_P0_ATE,
    V52_AUTOPSY_DIR,
    _fmt,
    _iter_run_dirs,
    _mean,
    _plot_no_data,
    _read_csv,
    _read_jsonl,
    _safe_float,
    _summarize_runs,
    _write_csv,
    _write_json,
)
from tools.v53_full_sequence_drift_autopsy import (  # noqa: E402
    _load_kitti_gt,
    _load_run_poses,
    _segment_error,
)


DEFAULT_RESULT_ROOT = REPO_ROOT / "results/kitti01_hmc_v2/acl2_v55_c9schedule_autopsy_failforward_adaptivettt_clean"
DEFAULT_GT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
DEFAULT_C9 = (
    REPO_ROOT
    / "results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/"
    "phase0_hard_gate/rollouts/V45_P0_C9_REPEAT"
)
DEFAULT_H35_704 = (
    REPO_ROOT
    / "results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/"
    "phase7_layergamma_refine_screen/rollouts/V53_PHASE7_SCREEN_H35_LAYERGAMMAFIX_RHO0075_704F"
)
DEFAULT_H35_FULL = (
    REPO_ROOT
    / "results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/"
    "phase7_layergamma_fix_full/rollouts/V53_PHASE7_FULL_H35_LAYERGAMMAFIX_RHO0075"
)
DEFAULT_V50_FULL = (
    REPO_ROOT
    / "results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/"
    "phase2_adaptive_failure_audit/rollouts/V52_TRACE_V50_SPLIT_RESXDG_AW111"
)

SEGMENTS: Sequence[Tuple[str, int, int]] = (
    ("seg0_000_384", 0, 384),
    ("seg1_384_700", 384, 700),
    ("seg2_700_end", 700, 20000),
    ("window_200_300", 200, 300),
    ("window_400_600", 400, 600),
)

REQUIRED_FILES = (
    "v55_phase0_salvage_report.md",
    "v55_phase1_c9_h35_m1_autopsy_report.md",
    "v55_failure_type_summary.json",
    "v55_candidate_design_decision.md",
    "v55_smoke_registry.csv",
    "v55_704f_registry.csv",
    "v55_full_registry.csv",
    "v55_runtime_audit.csv",
    "v55_no_chunk_manual_percentage_audit.csv",
    "v55_role_mass_timeline.csv",
    "v55_gamma_timeline.csv",
    "v55_commit_alpha_timeline.csv",
    "v55_post_zp_delta_timeline.csv",
    "v55_layer_branch_heatmap.png",
    "v55_segment_error_timeline.png",
    "v55_final_report.md",
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


def _write_text(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _write_json_file(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _is_704(row: Mapping[str, Any]) -> bool:
    frames = int(row.get("frames") or 0)
    name = str(row.get("run_name") or "").upper()
    return "704" in name or (650 <= frames <= 750)


def _is_smoke(row: Mapping[str, Any]) -> bool:
    frames = int(row.get("frames") or 0)
    name = str(row.get("run_name") or "").upper()
    return "96" in name or (1 <= frames <= 140)


def _is_full(row: Mapping[str, Any]) -> bool:
    return bool(row.get("full_kitti01")) or int(row.get("frames") or 0) >= 1000


def _augment_segments(rows: Sequence[Dict[str, Any]], gt_path: Path) -> None:
    _gt_frames, gt_poses, gt_pos = _load_kitti_gt(gt_path)
    for row in rows:
        run_dir = Path(str(row.get("run_dir") or ""))
        pose = _load_run_poses(run_dir, gt_poses, gt_pos)
        if pose.get("pose_status") != "done":
            row["segment_pose_status"] = pose.get("pose_status")
            continue
        frames = pose["frames"]
        aligned_pos = pose["aligned_pos"]
        for key, start, end in SEGMENTS:
            seg = _segment_error(frames, aligned_pos, gt_pos, start, end)
            for metric_key, value in seg.items():
                row[f"{key}_{metric_key}"] = value


def _chunk_ranges(run_dir: Path, frame_count: int) -> List[Tuple[int, int, int]]:
    rows = _read_jsonl(run_dir / "hmc_state_hash.jsonl")
    out: List[Tuple[int, int, int]] = []
    for idx, row in enumerate(rows):
        start = row.get("start_frame")
        end = row.get("end_frame")
        if start is None or end is None:
            continue
        try:
            out.append((int(row.get("chunk_idx", idx)), int(start), int(end)))
        except (TypeError, ValueError):
            continue
    if out:
        return out
    start = 0
    chunk = 0
    while start < frame_count:
        end = min(start + 32, frame_count)
        out.append((chunk, start, end))
        chunk += 1
        start = end
    return out


def _segment_id(start: int, end: int) -> str:
    mid = 0.5 * (start + end)
    if mid < 384:
        return "seg0"
    if mid < 700:
        return "seg1"
    return "seg2"


def _per_chunk_error_rows(
    run_key: str,
    run_dir: Path,
    gt_poses: np.ndarray,
    gt_pos: np.ndarray,
    ranges: Sequence[Tuple[int, int, int]],
) -> Dict[int, Dict[str, Any]]:
    pose = _load_run_poses(run_dir, gt_poses, gt_pos)
    if pose.get("pose_status") != "done":
        return {}
    frames = pose["frames"]
    aligned_pos = pose["aligned_pos"]
    rows: Dict[int, Dict[str, Any]] = {}
    for chunk_idx, start, end in ranges:
        seg = _segment_error(frames, aligned_pos, gt_pos, start, end)
        rows[int(chunk_idx)] = {
            "chunk_idx": int(chunk_idx),
            "frame_start": int(start),
            "frame_end": int(end),
            "segment": _segment_id(start, end),
            f"{run_key}_rmse": seg.get("rmse"),
            f"{run_key}_mean": seg.get("mean"),
            f"{run_key}_p90": seg.get("p90"),
            f"{run_key}_frame_count": seg.get("frame_count"),
        }
    return rows


def _build_chunk_gap_table(
    out_dir: Path,
    gt_path: Path,
    run_map: Mapping[str, Path],
) -> List[Dict[str, Any]]:
    _gt_frames, gt_poses, gt_pos = _load_kitti_gt(gt_path)
    base_dir = next((p for p in run_map.values() if p.is_dir()), Path(""))
    frame_count = gt_pos.shape[0]
    ranges = _chunk_ranges(base_dir, frame_count)
    by_chunk: Dict[int, Dict[str, Any]] = {
        int(idx): {
            "chunk_idx": int(idx),
            "frame_start": int(start),
            "frame_end": int(end),
            "segment": _segment_id(start, end),
            "reset_age": int(idx % 5),
            "reset_phase": int(idx % 5),
        }
        for idx, start, end in ranges
    }
    for key, run_dir in run_map.items():
        if not run_dir.is_dir():
            continue
        rows = _per_chunk_error_rows(key, run_dir, gt_poses, gt_pos, ranges)
        for idx, values in rows.items():
            by_chunk.setdefault(idx, {"chunk_idx": idx}).update(values)
    for row in by_chunk.values():
        c9 = _safe_float(row.get("C9_rmse"))
        h35 = _safe_float(row.get("H35_rmse"))
        m1 = _safe_float(row.get("S0_M1_rmse"))
        v50 = _safe_float(row.get("V50_rmse"))
        if math.isfinite(c9):
            if math.isfinite(h35):
                row["H35_minus_C9_rmse"] = h35 - c9
            if math.isfinite(m1):
                row["S0_M1_minus_C9_rmse"] = m1 - c9
            if math.isfinite(v50):
                row["V50_minus_C9_rmse"] = v50 - c9
            if math.isfinite(m1) and math.isfinite(h35):
                row["S0_M1_minus_H35_rmse"] = m1 - h35
    rows = [by_chunk[idx] for idx in sorted(by_chunk)]
    _write_csv(out_dir / "phase1_trace_autopsy/c9_h35_m1_chunk_gap_table.csv", rows)
    return rows


def _segment_gap_table(rows: Sequence[Mapping[str, Any]], out_dir: Path) -> List[Dict[str, Any]]:
    by_name = {str(row.get("reference") or row.get("run_name")): row for row in rows}
    c9 = by_name.get("C9_REF") or by_name.get("C9")
    out: List[Dict[str, Any]] = []
    for row in rows:
        name = str(row.get("reference") or row.get("run_name"))
        for seg_name, _start, _end in SEGMENTS:
            rmse = _safe_float(row.get(f"{seg_name}_rmse"))
            c9_rmse = _safe_float(c9.get(f"{seg_name}_rmse") if c9 else None)
            out.append({
                "run": name,
                "segment": seg_name,
                "rmse": rmse if math.isfinite(rmse) else None,
                "c9_rmse": c9_rmse if math.isfinite(c9_rmse) else None,
                "delta_vs_C9": (rmse - c9_rmse) if math.isfinite(rmse) and math.isfinite(c9_rmse) else None,
            })
    _write_csv(out_dir / "phase1_trace_autopsy/c9_h35_m1_segment_gap_table.csv", out)
    return out


def _state_variable_table(out_dir: Path, run_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    selected = [row for row in run_rows if row.get("run_dir")]
    out: List[Dict[str, Any]] = []
    keys = (
        "prior_mean_D_tok",
        "prior_q80_D_tok",
        "prior_q90_D_tok",
        "prior_high_D_mass",
        "prior_stage_d_mean",
        "prior_stage_d_low_mass",
        "prior_hmc_write_score_mean",
        "pass1_pass2_pose_t_mean",
        "pass1_pass2_pose_r_deg_mean",
        "memory_ttt_mean_rel_diff",
        "memory_ttt_w0_mean_rel_diff",
        "memory_ttt_w1_mean_rel_diff",
        "memory_ttt_w2_mean_rel_diff",
    )
    for run in selected:
        run_name = str(run.get("reference") or run.get("run_name"))
        run_dir = Path(str(run.get("run_dir")))
        for row in _read_jsonl(run_dir / "hmc_state_hash.jsonl"):
            rec: Dict[str, Any] = {
                "run": run_name,
                "chunk_idx": row.get("chunk_idx"),
                "frame_start": row.get("start_frame"),
                "frame_end": row.get("end_frame"),
                "reset_age": int(row.get("chunk_idx") or 0) % 5 if row.get("chunk_idx") is not None else None,
            }
            for key in keys:
                rec[key] = row.get(key)
            out.append(rec)
    _write_csv(out_dir / "phase1_trace_autopsy/c9_h35_m1_state_variable_table.csv", out)
    return out


def _extract_debug_timelines(run_rows: Sequence[Mapping[str, Any]], out_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    role_rows: List[Dict[str, Any]] = []
    gamma_rows: List[Dict[str, Any]] = []
    commit_rows: List[Dict[str, Any]] = []

    def add_from_node(run_name: str, chunk_idx: Any, node: Mapping[str, Any], source: str) -> None:
        role_keys = {
            "positive_mass": "ttt_tri_replay_pos_mass",
            "neutral_mass": "ttt_tri_replay_neu_mass",
            "negative_mass": "ttt_tri_replay_neg_mass",
            "neutral_lambda": "ttt_tri_replay_adaptive_neutral_lambda",
        }
        if any(k in node for k in role_keys.values()):
            role_rows.append({
                "run": run_name,
                "chunk_idx": chunk_idx,
                "source": source,
                **{out_key: node.get(in_key) for out_key, in_key in role_keys.items()},
                "role_source": node.get("ttt_tri_replay_role_source"),
                "role_collapsed": node.get("ttt_tri_replay_role_collapsed"),
            })
        gamma = None
        for key in (
            "ttt_tri_replay_state_energy_gamma_mean",
            "ttt_tri_replay_sc_gamma_gamma_mean",
            "ttt_tri_replay_energy_matched_gamma_mean",
            "ttt_tri_replay_adaptive_gamma",
        ):
            if key in node:
                gamma = node.get(key)
                break
        if gamma is not None:
            gamma_rows.append({
                "run": run_name,
                "chunk_idx": chunk_idx,
                "source": source,
                "gamma_mean": gamma,
                "neutral_lambda_mean": node.get("ttt_tri_replay_state_energy_neutral_lambda_mean", node.get("ttt_tri_replay_adaptive_neutral_lambda")),
            })
        if "ttt_write_commit_filter_scale_mean" in node or "ttt_write_commit_filter_activation_rate" in node:
            commit_rows.append({
                "run": run_name,
                "chunk_idx": chunk_idx,
                "source": source,
                "commit_alpha_mean": node.get("ttt_write_commit_filter_scale_mean"),
                "activation_rate": node.get("ttt_write_commit_filter_activation_rate"),
                "mode": node.get("ttt_write_commit_filter_mode"),
                "num_tensors": node.get("ttt_write_commit_filter_num_tensors"),
            })

    for run in run_rows:
        run_name = str(run.get("reference") or run.get("run_name"))
        run_dir = Path(str(run.get("run_dir") or ""))
        if not run_dir.is_dir():
            continue
        for idx, outer in enumerate(_read_jsonl(run_dir / "hmc_state_hash.jsonl")):
            chunk_idx = outer.get("chunk_idx", idx)
            for node in _walk(outer):
                add_from_node(run_name, chunk_idx, node, "hmc_state_hash.jsonl")
        log_path = run_dir / "01.log"
        if log_path.is_file():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            matches = list(re.finditer(r"# V2 Chunk\s+(\d+)/\d+", text))
            for pos, match in enumerate(matches):
                chunk_idx = int(match.group(1))
                start = match.start()
                end = matches[pos + 1].start() if pos + 1 < len(matches) else len(text)
                segment = text[start:end]
                for key in (
                    "ttt_tri_replay_state_energy_gamma_mean",
                    "ttt_tri_replay_sc_gamma_gamma_mean",
                    "ttt_tri_replay_energy_matched_gamma_mean",
                    "ttt_tri_replay_adaptive_gamma",
                ):
                    vals = [_safe_float(v) for v in re.findall(rf"['\"]{key}['\"]:\s*([-+0-9.eE]+)", segment)]
                    vals = [v for v in vals if math.isfinite(v)]
                    if vals:
                        gamma_rows.append({
                            "run": run_name,
                            "chunk_idx": chunk_idx,
                            "source": "01.log",
                            "gamma_mean": float(np.mean(vals)),
                            "gamma_key": key,
                        })
                        break
                cvals = [_safe_float(v) for v in re.findall(r"['\"]ttt_write_commit_filter_scale_mean['\"]:\s*([-+0-9.eE]+)", segment)]
                cvals = [v for v in cvals if math.isfinite(v)]
                avals = [_safe_float(v) for v in re.findall(r"['\"]ttt_write_commit_filter_activation_rate['\"]:\s*([-+0-9.eE]+)", segment)]
                avals = [v for v in avals if math.isfinite(v)]
                if cvals or avals:
                    commit_rows.append({
                        "run": run_name,
                        "chunk_idx": chunk_idx,
                        "source": "01.log",
                        "commit_alpha_mean": float(np.mean(cvals)) if cvals else None,
                        "activation_rate": float(np.mean(avals)) if avals else None,
                    })

    v52_role = _read_csv(V52_AUTOPSY_DIR / "role_mass_timeline.csv")
    for row in v52_role:
        role_rows.append({
            "run": row.get("run"),
            "chunk_idx": row.get("chunk_idx"),
            "source": "v52_role_mass_timeline.csv",
            "positive_mass": row.get("positive_mass_mean"),
            "neutral_mass": row.get("neutral_mass_mean"),
            "negative_mass": row.get("negative_mass_mean"),
            "neutral_lambda": row.get("neutral_lambda_mean"),
        })
        gamma_rows.append({
            "run": row.get("run"),
            "chunk_idx": row.get("chunk_idx"),
            "source": "v52_role_mass_timeline.csv",
            "gamma_mean": row.get("w0_gamma_mean"),
            "neutral_lambda_mean": row.get("neutral_lambda_mean"),
        })

    _write_csv(out_dir / "v55_role_mass_timeline.csv", role_rows)
    _write_csv(out_dir / "v55_gamma_timeline.csv", gamma_rows)
    _write_csv(out_dir / "v55_commit_alpha_timeline.csv", commit_rows)
    _write_csv(out_dir / "phase1_trace_autopsy/c9_h35_m1_role_mass_table.csv", role_rows)
    _write_csv(out_dir / "phase1_trace_autopsy/c9_h35_m1_commit_behavior_table.csv", commit_rows)
    return role_rows, gamma_rows, commit_rows


def _copy_or_plot_v52_artifacts(out_dir: Path) -> None:
    src_post = V52_AUTOPSY_DIR / "post_zp_delta_ratio_by_chunk.csv"
    dst_post = out_dir / "v55_post_zp_delta_timeline.csv"
    if src_post.is_file():
        shutil.copy2(src_post, dst_post)
        shutil.copy2(src_post, out_dir / "phase1_trace_autopsy/c9_h35_m1_post_zp_delta_table.csv")
    else:
        _write_csv(dst_post, [])
        _write_csv(out_dir / "phase1_trace_autopsy/c9_h35_m1_post_zp_delta_table.csv", [])

    src_layer = V52_AUTOPSY_DIR / "delta_norm_ratio_by_layer.csv"
    if src_layer.is_file():
        shutil.copy2(src_layer, out_dir / "phase1_trace_autopsy/c9_h35_m1_layer_branch_delta_table.csv")
    else:
        _write_csv(out_dir / "phase1_trace_autopsy/c9_h35_m1_layer_branch_delta_table.csv", [])

    if (V52_AUTOPSY_DIR / "post_zp_delta_ratio_by_chunk.png").is_file():
        shutil.copy2(V52_AUTOPSY_DIR / "post_zp_delta_ratio_by_chunk.png", out_dir / "teacher_student_post_zp_delta_timeline.png")
    else:
        _plot_no_data(out_dir / "teacher_student_post_zp_delta_timeline.png", "post-zp delta timeline", "missing v52 source plot")

    if (V52_AUTOPSY_DIR / "teacher_student_role_mass_timeline.png").is_file():
        shutil.copy2(V52_AUTOPSY_DIR / "teacher_student_role_mass_timeline.png", out_dir / "teacher_student_role_mass_timeline.png")
    else:
        _plot_no_data(out_dir / "teacher_student_role_mass_timeline.png", "role mass timeline", "missing v52 source plot")

    if (V52_AUTOPSY_DIR / "delta_norm_ratio_by_layer.png").is_file():
        shutil.copy2(V52_AUTOPSY_DIR / "delta_norm_ratio_by_layer.png", out_dir / "v55_layer_branch_heatmap.png")
        shutil.copy2(V52_AUTOPSY_DIR / "delta_norm_ratio_by_layer.png", out_dir / "teacher_student_layer_branch_heatmap.png")
    else:
        _plot_no_data(out_dir / "v55_layer_branch_heatmap.png", "layer branch heatmap", "missing v52 source plot")
        _plot_no_data(out_dir / "teacher_student_layer_branch_heatmap.png", "layer branch heatmap", "missing v52 source plot")


def _plot_lines(rows: Sequence[Mapping[str, Any]], out: Path, y_key: str, title: str, ylabel: str) -> None:
    pts_by_run: Dict[str, List[Tuple[int, float]]] = {}
    for row in rows:
        x = _safe_float(row.get("chunk_idx"))
        y = _safe_float(row.get(y_key))
        run = str(row.get("run") or "")
        if math.isfinite(x) and math.isfinite(y) and run:
            pts_by_run.setdefault(run, []).append((int(x), y))
    if not pts_by_run:
        _plot_no_data(out, title, f"no finite {y_key} rows")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    for run, pts in sorted(pts_by_run.items()):
        pts = sorted(pts)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", linewidth=1.0, label=run)
    ax.set_xlabel("chunk index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _plot_segment_gap(segment_rows: Sequence[Mapping[str, Any]], out_dir: Path) -> None:
    runs = sorted({str(r.get("run")) for r in segment_rows})
    segs = [name for name, _start, _end in SEGMENTS[:3]]
    data = np.full((len(runs), len(segs)), np.nan)
    for row in segment_rows:
        try:
            i = runs.index(str(row.get("run")))
            j = segs.index(str(row.get("segment")))
        except ValueError:
            continue
        data[i, j] = _safe_float(row.get("rmse"))
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.35 * len(runs) + 2)))
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(segs)))
    ax.set_xticklabels(segs)
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels(runs, fontsize=7)
    ax.set_title("v55 segment RMSE timeline")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "v55_segment_error_timeline.png", dpi=160)
    plt.close(fig)


def _plot_chunk_gap(chunk_rows: Sequence[Mapping[str, Any]], out_dir: Path) -> None:
    xs = [_safe_float(r.get("chunk_idx")) for r in chunk_rows]
    ys = [_safe_float(r.get("H35_minus_C9_rmse")) for r in chunk_rows]
    pts = [(int(x), y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if not pts:
        _plot_no_data(out_dir / "c9_minus_h35_gap_by_chunk.png", "C9-H35 gap by chunk", "no finite chunk gap rows")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", linewidth=1.0)
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xlabel("chunk index")
    ax.set_ylabel("H35 RMSE - C9 RMSE")
    ax.set_title("C9 vs H35 chunk RMSE gap")
    fig.tight_layout()
    fig.savefig(out_dir / "c9_minus_h35_gap_by_chunk.png", dpi=160)
    plt.close(fig)


def _plot_scatter(state_rows: Sequence[Mapping[str, Any]], gamma_rows: Sequence[Mapping[str, Any]], commit_rows: Sequence[Mapping[str, Any]], out_dir: Path) -> None:
    state_by_key = {
        (str(r.get("run")), str(r.get("chunk_idx"))): _safe_float(r.get("memory_ttt_w0_mean_rel_diff"))
        for r in state_rows
    }

    def scatter(rows: Sequence[Mapping[str, Any]], y_key: str, out: Path, title: str) -> None:
        pts: List[Tuple[float, float]] = []
        for row in rows:
            key = (str(row.get("run")), str(row.get("chunk_idx")))
            x = state_by_key.get(key, float("nan"))
            y = _safe_float(row.get(y_key))
            if math.isfinite(x) and math.isfinite(y):
                pts.append((x, y))
        if not pts:
            _plot_no_data(out, title, "no shared finite state/debug rows")
            return
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=18)
        ax.set_xlabel("memory_ttt_w0_mean_rel_diff")
        ax.set_ylabel(y_key)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        plt.close(fig)

    scatter(gamma_rows, "gamma_mean", out_dir / "state_variable_vs_teacher_gamma_scatter.png", "state variable vs gamma")
    scatter(commit_rows, "commit_alpha_mean", out_dir / "state_variable_vs_commit_alpha_scatter.png", "state variable vs commit alpha")


def _gap_scores(segment_rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for row in segment_rows:
        if str(row.get("run")) != "H35_FULL":
            continue
        seg = str(row.get("segment"))
        if not seg.startswith("seg"):
            continue
        delta = _safe_float(row.get("delta_vs_C9"))
        if math.isfinite(delta):
            scores[seg] = delta
    return scores


def _mean_abs_log_ratio(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    vals = []
    for row in rows:
        val = _safe_float(row.get(key))
        if math.isfinite(val) and val > 0:
            vals.append(abs(math.log(val)))
    return float(np.mean(vals)) if vals else None


def _classify_failure(
    segment_rows: Sequence[Mapping[str, Any]],
    role_rows: Sequence[Mapping[str, Any]],
    commit_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    seg_scores = _gap_scores(segment_rows)
    largest_seg = max(seg_scores.items(), key=lambda item: item[1])[0] if seg_scores else "unknown"
    post_rows = _read_csv(V52_AUTOPSY_DIR / "post_zp_delta_ratio_by_chunk.csv")
    layer_rows = _read_csv(V52_AUTOPSY_DIR / "delta_norm_ratio_by_layer.csv")
    committed_gap = _mean_abs_log_ratio(post_rows, "student_over_c9_committed_delta_norm")
    native_gap = _mean_abs_log_ratio(post_rows, "student_over_c9_native_delta_norm")
    layer_gap = _mean_abs_log_ratio(layer_rows, "student_over_c9_delta_norm")

    role_summary = _read_csv(V52_AUTOPSY_DIR / "run_overview.csv")
    c9_role = next((r for r in role_summary if r.get("run") == "C9_exact_teacher"), {})
    v50_role = next((r for r in role_summary if r.get("run") == "V50_split_resxdg_student"), {})
    role_gap = sum(
        abs(_safe_float(v50_role.get(key)) - _safe_float(c9_role.get(key)))
        for key in ("positive_mass_mean", "neutral_mass_mean", "negative_mass_mean")
        if math.isfinite(_safe_float(v50_role.get(key))) and math.isfinite(_safe_float(c9_role.get(key)))
    )

    commit_activation = _mean(r.get("activation_rate") for r in commit_rows if str(r.get("run")) == "S0_M1_FULL")
    if commit_activation is None:
        commit_activation = 0.0

    candidates = {
        "TYPE_A_ROLE_MASS_GAP": float(role_gap),
        "TYPE_B_GAMMA_ENERGY_GAP": float(committed_gap or 0.0),
        "TYPE_C_COMMIT_SCHEDULE_GAP": float(commit_activation or 0.0),
        "TYPE_D_LAYER_BRANCH_ACTION_GAP": float(layer_gap or 0.0),
        "TYPE_E_SEG2_STATE_GAP": float(seg_scores.get("seg2_700_end", 0.0)),
    }

    if largest_seg == "seg2_700_end" and candidates["TYPE_E_SEG2_STATE_GAP"] > 0:
        failure_type = "TYPE_E_SEG2_STATE_GAP"
        reason = "H35 full has its largest finite C9 segment RMSE gap in seg2."
    else:
        failure_type = max(candidates.items(), key=lambda item: item[1])[0]
        reason = "Selected by largest normalized landed evidence score; segment largest was not seg2."

    return {
        "failure_type": failure_type,
        "reason": reason,
        "scores": candidates,
        "largest_segment_gap": largest_seg,
        "segment_delta_vs_C9": seg_scores,
        "role_mass_l1_gap_v50_vs_c9": role_gap,
        "post_zp_committed_abs_log_ratio_mean": committed_gap,
        "post_zp_native_abs_log_ratio_mean": native_gap,
        "layer_branch_abs_log_ratio_mean": layer_gap,
        "evidence_sources": {
            "segments": "computed from 01.txt with GT alignment",
            "role_post_zp_layer": str(V52_AUTOPSY_DIR),
            "commit": "v55/v54 hmc_state_hash.jsonl and 01.log when present",
        },
    }


def _candidate_decision(failure_type: str) -> Dict[str, Any]:
    mapping = {
        "TYPE_A_ROLE_MASS_GAP": [
            ("A1_RoleSplitV3", "adaptive_writer_role_split_v3"),
            ("A2_RoleSplitV3_StateEnergy", "adaptive_writer_role_split_v3_state_energy"),
        ],
        "TYPE_B_GAMMA_ENERGY_GAP": [
            ("B1_TeacherEnvelopeGammaV2", "adaptive_writer_teacher_envelope_gamma_v2"),
            ("B2_TeacherEnvelopeGammaV2_SelectiveCommit", "adaptive_writer_teacher_envelope_gamma_v2_selective_commit"),
        ],
        "TYPE_C_COMMIT_SCHEDULE_GAP": [
            ("C1_SelectiveCommitEMA", "adaptive_writer_state_energy_selective_commit_split"),
            ("C2_SelectiveCommitEMA_Loose", "adaptive_writer_state_energy_selective_commit_loose_split"),
        ],
        "TYPE_D_LAYER_BRANCH_ACTION_GAP": [
            ("D1_LayerBranchEnergyRouter", "adaptive_writer_layer_branch_energy_router"),
            ("D2_LayerBranchEnergyRouter_SelectiveCommit", "adaptive_writer_layer_branch_energy_router_selective_commit"),
        ],
        "TYPE_E_SEG2_STATE_GAP": [
            ("E1_TailStateContinuityGuard", "adaptive_writer_tail_state_continuity_guard"),
            ("E2_TailStateContinuityGuard_SelectiveCommit", "adaptive_writer_tail_state_continuity_guard_selective_commit"),
        ],
    }
    return {
        "failure_type": failure_type,
        "candidates": [
            {"candidate": name, "role_mode": role_mode}
            for name, role_mode in mapping.get(failure_type, mapping["TYPE_B_GAMMA_ENERGY_GAP"])
        ],
    }


def _screen_decision(row: Mapping[str, Any], h35_704: Optional[Mapping[str, Any]]) -> str:
    if row.get("status") != "done":
        return "not_done"
    if row.get("no_chunk_policy_pass") is not True:
        return "audit_fail_no_chunk"
    if row.get("manual_percentage_audit_pass") is not True:
        return "audit_fail_manual_percentage"
    if _safe_float(row.get("role_collapse_rate"), 1.0) > 0.05:
        return "role_collapse_fail"
    if _safe_float(row.get("chunk_total_seconds_mean"), 999.0) > 42.0:
        return "runtime_fail_chunk_mean"
    probe_ttt = _safe_float(row.get("probe_ttt_write_seconds_mean"))
    if math.isfinite(probe_ttt) and probe_ttt > 8.0:
        return "runtime_fail_probe_ttt"
    if h35_704 is None:
        return "missing_h35_704"
    ate = _safe_float(row.get("ATE"), 999.0)
    h35 = _safe_float(h35_704.get("ATE"), 999.0)
    projected = _safe_float(row.get("projected_full_wall_time_min"), 999.0)
    if projected > 28.0:
        return "reject_projected_runtime_gt_28"
    if ate <= h35 + 0.10:
        return "promote_full"
    if ate <= h35 + 0.35:
        return "borderline_diagnostic_full"
    return "reject_ate_gt_h35_plus_0.35"


def _write_phase0_report(out_dir: Path, s0: Optional[Mapping[str, Any]], h35_full: Optional[Mapping[str, Any]]) -> None:
    lines = [
        "# ACL2 v55 Phase 0 salvage report",
        "",
        "S0 uses the v54 M1 state-energy matched split as a full diagnostic, not as an expected success.",
        "",
        "| run | status | frames | ATE | delta vs H35 full | Rot | FinalErr | wall min | chunk mean | TTT mean | no-chunk | manual % | role collapse rate | diagnostic decision |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|",
    ]
    if s0 is None:
        lines.append("| S0_M1_FULL_DIAGNOSTIC | missing | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | not available |")
    else:
        h35_ate = _safe_float(h35_full.get("ATE") if h35_full else None)
        delta = _safe_float(s0.get("ATE")) - h35_ate if math.isfinite(h35_ate) else float("nan")
        decision = "borderline_action_space_evidence" if math.isfinite(delta) and delta <= 0.30 else "m1_invalid_no_extension"
        lines.append(
            f"| `{s0.get('run_name')}` | `{s0.get('status')}` | {s0.get('frames')} | {_fmt(s0.get('ATE'))} | "
            f"{_fmt(delta)} | {_fmt(s0.get('Rot'))} | {_fmt(s0.get('FinalErr'))} | {_fmt(s0.get('wall_time_min'))} | "
            f"{_fmt(s0.get('chunk_total_seconds_mean'))} | {_fmt(s0.get('probe_ttt_write_seconds_mean'))} | "
            f"{s0.get('no_chunk_policy_pass')} | {s0.get('manual_percentage_audit_pass')} | {_fmt(s0.get('role_collapse_rate'))} | `{decision}` |"
        )
    _write_text(out_dir / "v55_phase0_salvage_report.md", lines)


def _write_phase1_report(
    out_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
    failure: Mapping[str, Any],
) -> None:
    overview = _read_csv(V52_AUTOPSY_DIR / "run_overview.csv")
    lines = [
        "# ACL2 v55 Phase 1 C9-H35-M1 trace autopsy report",
        "",
        "This report combines landed v52 teacher/student trace artifacts with v53 H35 and v55 S0 trajectory artifacts. Missing fields are left as NA.",
        "",
        f"v52 trace source: `{V52_AUTOPSY_DIR}`",
        "",
        "## Run Metrics",
        "",
        "| run | source | status | frames | ATE | Rot | FinalErr | wall min | chunk mean | TTT mean |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row.get('reference') or row.get('run_name')}` | `{row.get('source')}` | `{row.get('status')}` | "
            f"{row.get('frames')} | {_fmt(row.get('ATE'))} | {_fmt(row.get('Rot'))} | {_fmt(row.get('FinalErr'))} | "
            f"{_fmt(row.get('wall_time_min'))} | {_fmt(row.get('chunk_total_seconds_mean'))} | {_fmt(row.get('probe_ttt_write_seconds_mean'))} |"
        )
    lines.extend([
        "",
        "## Segment RMSE",
        "",
        "| run | segment | rmse | C9 rmse | delta vs C9 |",
        "|---|---|---:|---:|---:|",
    ])
    for row in segment_rows:
        if str(row.get("segment")).startswith("window"):
            continue
        lines.append(
            f"| `{row.get('run')}` | `{row.get('segment')}` | {_fmt(row.get('rmse'))} | {_fmt(row.get('c9_rmse'))} | {_fmt(row.get('delta_vs_C9'))} |"
        )
    lines.extend([
        "",
        "## v52 Teacher/Student Trace Overview",
        "",
        "| run | ATE | role source | pos mass | neu mass | neg mass | gamma mean | risk source |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ])
    for row in overview:
        lines.append(
            f"| `{row.get('run')}` | {_fmt(row.get('ATE_full'))} | `{row.get('role_sources_seen')}` | "
            f"{_fmt(row.get('positive_mass_mean'))} | {_fmt(row.get('neutral_mass_mean'))} | {_fmt(row.get('negative_mass_mean'))} | "
            f"{_fmt(row.get('w0_gamma_mean'))} | `{row.get('risk_source_seen')}` |"
        )
    lines.extend([
        "",
        "## Required Questions",
        "",
        f"Q1: 最大 H35-C9 segment gap: `{failure.get('largest_segment_gap')}`; finite deltas: `{failure.get('segment_delta_vs_C9')}`.",
        "Q2: H35 role mass has no exact C9 trace in the v52 artifact; v50 vs C9 landed role-mass L1 gap is "
        f"`{_fmt(failure.get('role_mass_l1_gap_v50_vs_c9'))}` and is not filled with H35 synthetic values.",
        "Q3: post-zp committed energy gap from v52 student/C9 abs log ratio mean is "
        f"`{_fmt(failure.get('post_zp_committed_abs_log_ratio_mean'))}`.",
        "Q4: layer/branch delta pattern gap from v52 layer table abs log ratio mean is "
        f"`{_fmt(failure.get('layer_branch_abs_log_ratio_mean'))}`.",
        "Q5: C9 commit EMA selectivity cannot be fully recovered from H35/M1 artifacts; commit fields are reported in `c9_h35_m1_commit_behavior_table.csv` when present.",
        "Q6: reset_age/state variables are exported in `c9_h35_m1_state_variable_table.csv` and scatter plots; no learned selector is inferred from them.",
        "Q7: M1 fixes state-energy gamma timing, but Phase 0 and segment deltas decide whether it fixed the largest gap; see `v55_phase0_salvage_report.md`.",
        "",
        "## Failure Type",
        "",
        f"Selected failure type: `{failure.get('failure_type')}`.",
        f"Reason: {failure.get('reason')}",
    ])
    _write_text(out_dir / "v55_phase1_c9_h35_m1_autopsy_report.md", lines)
    _write_text(out_dir / "phase1_trace_autopsy/c9_h35_m1_autopsy_report.md", lines)


def _write_candidate_decision(out_dir: Path, decision: Mapping[str, Any]) -> None:
    lines = [
        "# ACL2 v55 candidate design decision",
        "",
        f"Failure type: `{decision.get('failure_type')}`.",
        "",
        "| candidate | role mode / implementation handle |",
        "|---|---|",
    ]
    for row in decision.get("candidates", []):
        lines.append(f"| `{row.get('candidate')}` | `{row.get('role_mode')}` |")
    lines.extend([
        "",
        "The selected candidates are the only Phase 2 candidates allowed by this report. They must keep no-chunk and no-manual-percentage audits true.",
    ])
    _write_text(out_dir / "v55_candidate_design_decision.md", lines)


def _write_registries(
    out_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    h35_704: Optional[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    smoke = [dict(row) for row in rows if _is_smoke(row)]
    screen = [dict(row) for row in rows if _is_704(row)]
    full = [dict(row) for row in rows if _is_full(row)]
    for row in smoke:
        row["screen_decision"] = _screen_decision(row, h35_704)
    for row in screen:
        row["screen_decision"] = _screen_decision(row, h35_704)
    for row in full:
        ate = _safe_float(row.get("ATE"), 999.0)
        row["progress_pass"] = ate <= 35.30
        row["soft_pass"] = ate <= 34.60
        row["close_to_c9_pass"] = ate <= 34.30
        row["excellent_pass"] = ate <= 34.06
    _write_csv(out_dir / "v55_smoke_registry.csv", smoke)
    _write_csv(out_dir / "v55_704f_registry.csv", screen)
    _write_csv(out_dir / "v55_full_registry.csv", full)
    _write_csv(out_dir / "v55_all_registry.csv", rows)
    _write_json(out_dir / "v55_all_registry.json", list(rows))
    _write_csv(out_dir / "v55_runtime_audit.csv", [
        {
            "run_name": row.get("run_name"),
            "phase": row.get("phase"),
            "status": row.get("status"),
            "frames": row.get("frames"),
            "wall_time_min": row.get("wall_time_min"),
            "projected_full_wall_time_min": row.get("projected_full_wall_time_min"),
            "chunk_total_seconds_mean": row.get("chunk_total_seconds_mean"),
            "probe_ttt_write_seconds_mean": row.get("probe_ttt_write_seconds_mean"),
            "probe_ttt_write_seconds_missing": not math.isfinite(_safe_float(row.get("probe_ttt_write_seconds_mean"))),
            "smoke_runtime_gate_pass": row.get("smoke_runtime_gate_pass"),
            "full_runtime_gate_pass": row.get("full_runtime_gate_pass"),
            "v55_runtime_gate_allow_probe_missing": (
                row.get("status") == "done"
                and _safe_float(row.get("chunk_total_seconds_mean"), 999.0) <= 42.0
                and (
                    not math.isfinite(_safe_float(row.get("probe_ttt_write_seconds_mean")))
                    or _safe_float(row.get("probe_ttt_write_seconds_mean"), 999.0) <= 8.0
                )
                and (
                    (not bool(row.get("full_kitti01")))
                    or _safe_float(row.get("wall_time_min"), 999.0) <= 28.0
                )
            ),
        }
        for row in rows
    ])
    _write_csv(out_dir / "v55_no_chunk_manual_percentage_audit.csv", [
        {
            "run_name": row.get("run_name"),
            "phase": row.get("phase"),
            "no_chunk_policy_pass": row.get("no_chunk_policy_pass"),
            "manual_percentage_audit_pass": row.get("manual_percentage_audit_pass"),
            "role_mode_config": row.get("role_mode_config"),
            "risk_source_config": row.get("risk_source_config"),
            "commit_filter_mode_config": row.get("commit_filter_mode_config"),
            "role_collapse_rate": row.get("role_collapse_rate"),
            "run_dir": row.get("run_dir"),
        }
        for row in rows
    ])
    return smoke, screen, full


def _write_final_report(
    out_dir: Path,
    failure: Mapping[str, Any],
    decision: Mapping[str, Any],
    full: Sequence[Mapping[str, Any]],
    screen: Sequence[Mapping[str, Any]],
    s0: Optional[Mapping[str, Any]],
    h35_full: Optional[Mapping[str, Any]],
) -> None:
    best_full = None
    for row in full:
        if best_full is None or _safe_float(row.get("ATE"), 999.0) < _safe_float(best_full.get("ATE"), 999.0):
            best_full = row
    s0_delta = float("nan")
    if s0 is not None and h35_full is not None:
        s0_delta = _safe_float(s0.get("ATE")) - _safe_float(h35_full.get("ATE"))
    lines = [
        "# ACL2 v55 final report",
        "",
        "Generated from landed artifacts only. Missing measurements are reported as NA.",
        "",
        "## Required Answers",
        "",
        f"1. v54 M1 full 是否应该被 704F gate 误杀: S0 delta vs H35 full = `{_fmt(s0_delta)}`; see Phase 0 for the exact decision.",
        f"2. C9-H35 最大差距: `{failure.get('largest_segment_gap')}`; selected failure type `{failure.get('failure_type')}`.",
        f"3. 新候选修的目标: `{decision.get('failure_type')}` with candidates `{decision.get('candidates')}`.",
        f"4. no chunk / manual percentage / runtime gate: see `v55_no_chunk_manual_percentage_audit.csv` and `v55_runtime_audit.csv`.",
    ]
    if best_full is not None:
        lines.append(
            f"5. Best v55 full: `{best_full.get('run_name')}` ATE `{_fmt(best_full.get('ATE'))}`, progress pass `{best_full.get('progress_pass')}`."
        )
        next_step = "freeze_or_refine_formula" if bool(best_full.get("progress_pass")) else "action_space_redesign"
    else:
        lines.append("5. Best v55 full: `none yet`; no full candidate artifact beyond S0/reference was found.")
        next_step = "continue_plan_if_candidates_pending"
    lines.append(f"6. 如果不接近 C9，下一步路由: `{next_step}`.")
    lines.extend([
        "",
        "## Full Runs",
        "",
        "| run | ATE | delta vs C9 | frames | wall min | progress | soft | close | excellent |",
        "|---|---:|---:|---:|---:|---|---|---|---|",
    ])
    for row in full:
        lines.append(
            f"| `{row.get('run_name')}` | {_fmt(row.get('ATE'))} | {_fmt(row.get('delta_vs_C9_P0'))} | {row.get('frames')} | "
            f"{_fmt(row.get('wall_time_min'))} | {row.get('progress_pass')} | {row.get('soft_pass')} | "
            f"{row.get('close_to_c9_pass')} | {row.get('excellent_pass')} |"
        )
    if not full:
        lines.append("| no v55 full candidate | NA | NA | NA | NA | False | False | False | False |")
    lines.extend([
        "",
        "## 704F Screens",
        "",
        "| run | ATE | projected full min | decision |",
        "|---|---:|---:|---|",
    ])
    for row in screen:
        lines.append(
            f"| `{row.get('run_name')}` | {_fmt(row.get('ATE'))} | {_fmt(row.get('projected_full_wall_time_min'))} | `{row.get('screen_decision')}` |"
        )
    if not screen:
        lines.append("| no 704F candidate | NA | NA | no_data |")
    if best_full is not None and _safe_float(best_full.get("ATE"), 999.0) > 35.30:
        lines.extend([
            "",
            "## Action-space Redesign Conclusion",
            "",
            "- Current adaptive split/gamma/commit patching did not produce a progress-pass full candidate.",
            "- The next round should change the TTT action space if candidates that specifically target the selected failure type also fail.",
        ])
    _write_text(out_dir / "v55_final_report.md", lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--gt", default=str(DEFAULT_GT))
    parser.add_argument("--c9", default=str(DEFAULT_C9))
    parser.add_argument("--h35-704", default=str(DEFAULT_H35_704))
    parser.add_argument("--h35-full", default=str(DEFAULT_H35_FULL))
    parser.add_argument("--v50-full", default=str(DEFAULT_V50_FULL))
    args = parser.parse_args()

    result_root = Path(args.result_root)
    out_dir = Path(args.out_dir) if args.out_dir else result_root / "report_final"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase1_trace_autopsy").mkdir(parents=True, exist_ok=True)

    rollout_roots = sorted(result_root.glob("phase*/rollouts"))
    run_dirs = _iter_run_dirs(rollout_roots)
    v55_rows = _summarize_runs(run_dirs, Path(args.gt)) if run_dirs else []
    _augment_segments(v55_rows, Path(args.gt))

    ref_specs = [
        ("C9_REF", "c9_reference", Path(args.c9)),
        ("H35_704", "v53_reference", Path(args.h35_704)),
        ("H35_FULL", "v53_reference", Path(args.h35_full)),
        ("V50_FULL", "v52_reference", Path(args.v50_full)),
    ]
    ref_dirs = [path for _name, _source, path in ref_specs if path.is_dir()]
    ref_rows = _summarize_runs(ref_dirs, Path(args.gt)) if ref_dirs else []
    _augment_segments(ref_rows, Path(args.gt))
    by_dir = {str(Path(str(row.get("run_dir"))).resolve()): row for row in ref_rows}
    refs: List[Dict[str, Any]] = []
    for name, source, path in ref_specs:
        row = dict(by_dir.get(str(path.resolve()), {}))
        row["reference"] = name
        row["source"] = source
        row["run_dir"] = str(path)
        refs.append(row)
    for row in v55_rows:
        row["source"] = "v55_run"

    all_for_autopsy = refs + [dict(r, reference=r.get("run_name")) for r in v55_rows]
    h35_704 = next((r for r in refs if r.get("reference") == "H35_704"), None)
    h35_full = next((r for r in refs if r.get("reference") == "H35_FULL"), None)
    s0 = next((r for r in v55_rows if str(r.get("run_name")) == "S0_M1_FULL_DIAGNOSTIC"), None)

    _write_phase0_report(out_dir, s0, h35_full)

    run_map = {
        "C9": Path(args.c9),
        "H35": Path(args.h35_full),
        "V50": Path(args.v50_full),
    }
    if s0 is not None:
        run_map["S0_M1"] = Path(str(s0.get("run_dir")))
    chunk_rows = _build_chunk_gap_table(out_dir, Path(args.gt), run_map)
    segment_rows = _segment_gap_table(all_for_autopsy, out_dir)
    state_rows = _state_variable_table(out_dir, all_for_autopsy)
    role_rows, gamma_rows, commit_rows = _extract_debug_timelines(all_for_autopsy, out_dir)
    _copy_or_plot_v52_artifacts(out_dir)
    _plot_lines(gamma_rows, out_dir / "teacher_student_gamma_timeline.png", "gamma_mean", "teacher/student gamma timeline", "gamma")
    _plot_lines(commit_rows, out_dir / "teacher_student_commit_alpha_timeline.png", "commit_alpha_mean", "teacher/student commit alpha timeline", "commit alpha")
    _plot_segment_gap(segment_rows, out_dir)
    _plot_chunk_gap(chunk_rows, out_dir)
    _plot_scatter(state_rows, gamma_rows, commit_rows, out_dir)

    failure = _classify_failure(segment_rows, role_rows, commit_rows)
    _write_json_file(out_dir / "v55_failure_type_summary.json", failure)
    decision = _candidate_decision(str(failure.get("failure_type")))
    _write_candidate_decision(out_dir, decision)
    smoke, screen, full = _write_registries(out_dir, v55_rows, h35_704)
    _write_phase1_report(out_dir, all_for_autopsy, segment_rows, failure)
    _write_final_report(out_dir, failure, decision, full, screen, s0, h35_full)

    missing = [name for name in REQUIRED_FILES if not (out_dir / name).exists()]
    if missing:
        raise RuntimeError(f"v55 report missing required files: {missing}")
    print(f"Wrote v55 report with {len(v55_rows)} v55 rows to {out_dir}")


if __name__ == "__main__":
    main()
