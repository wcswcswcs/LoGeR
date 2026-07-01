#!/usr/bin/env python3
"""Bridge TTT trajectory geometry errors to visual/semantic inspection targets.

The script does not run a new method. It reads completed accelerated TTT
rollouts, aligns each trajectory to KITTI GT, writes per-frame/per-chunk error
maps, and selects the frames/chunks where a candidate is worse than the TTT
baseline and paired random control. Those selected rows are the inputs for
targeted TTT visual probes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional in headless envs.
    plt = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.kitti_trajectory_diagnostics import (  # noqa: E402
    _apply_alignment,
    _load_kitti_gt,
    _load_tum_prediction,
    _umeyama_sim3,
)


DEFAULT_ROLLOUTS_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final/phase5_ttt_long_case_accelerated_smoke/rollouts"
)
DEFAULT_OUT_DIR = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final/phase5_ttt_geometry_error_visual_bridge"
)
DEFAULT_DATA_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset")
DEFAULT_CANDIDATES = (
    "LW23_TTT_OVERLAP_ROLECAL_TTL_B0_PRIOR100,"
    "LW25_TTT_OVERLAP_ROLECAL_TTL_B0_PRIOR108"
)
PAIRED_RANDOM_CONTROLS = {
    "LW23_TTT_OVERLAP_ROLECAL_TTL_B0_PRIOR100": "LW24_TTT_OVERLAP_ROLECAL_TTL_RANDOM_ROLE_B0_PRIOR100",
    "LW25_TTT_OVERLAP_ROLECAL_TTL_B0_PRIOR108": "LW26_TTT_OVERLAP_ROLECAL_TTL_RANDOM_ROLE_B0_PRIOR108",
}
WINDOW_RE = re.compile(
    r"^seq(?P<seq>\d+)_chunks(?P<start>\d+)_(?P<end>\d+)_(?P<case_type>bad|good)_rank(?P<rank>\d+)$"
)


def _finite(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean(values: list[Any]) -> float | None:
    vals = [_finite(v) for v in values]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _max(values: list[Any]) -> float | None:
    vals = [_finite(v) for v in values]
    vals = [v for v in vals if v is not None]
    return float(np.max(vals)) if vals else None


def _parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), ensure_ascii=False)
                    if isinstance(row.get(key), (dict, list))
                    else row.get(key)
                    for key in fieldnames
                }
            )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _window_info(path: Path) -> dict[str, Any] | None:
    match = WINDOW_RE.match(path.name)
    if not match:
        return None
    info = match.groupdict()
    return {
        "window_id": path.name,
        "seq": str(info["seq"]).zfill(2),
        "chunk_start": int(info["start"]),
        "chunk_end": int(info["end"]),
        "case_type": info["case_type"],
        "case_rank": int(info["rank"]),
        "path": path,
    }


def _discover_windows(root: Path) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for path in sorted(root.iterdir()) if root.exists() else []:
        if not path.is_dir():
            continue
        info = _window_info(path)
        if info is not None:
            windows.append(info)
    return windows


def _primary_chunk_for_frame(frame: int, chunk_size: int, overlap: int) -> int:
    stride = int(chunk_size) - int(overlap)
    return int(math.floor(int(frame) / max(1, stride)))


def _load_aligned_run(traj: Path, gt_path: Path) -> dict[str, Any]:
    gt_frames, gt_poses, gt_pos = _load_kitti_gt(gt_path)
    pred_frames, pred_poses, pred_pos = _load_tum_prediction(traj, n_gt=len(gt_frames))
    if len(pred_frames) < 3:
        raise ValueError(f"need at least 3 prediction frames for alignment: {traj}")
    gt_for_pred = gt_pos[pred_frames]
    scale, rot, trans = _umeyama_sim3(pred_pos, gt_for_pred, with_scale=True)
    aligned_poses = _apply_alignment(pred_poses, scale, rot, trans)
    aligned_pos = aligned_poses[:, :3, 3]
    err_vec = aligned_pos - gt_for_pred
    err_m = np.linalg.norm(err_vec, axis=1)
    return {
        "frames": pred_frames,
        "raw_pos": pred_pos,
        "aligned_pos": aligned_pos,
        "gt_pos": gt_for_pred,
        "err_m": err_m,
        "scale": scale,
        "rot": rot,
        "trans": trans,
    }


def _run_rows(
    *,
    window: dict[str, Any],
    case: str,
    run_dir: Path,
    aligned: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frames = aligned["frames"]
    for idx, frame in enumerate(frames):
        chunk_id = _primary_chunk_for_frame(int(frame), int(args.chunk_size), int(args.chunk_overlap))
        rows.append(
            {
                "window_id": window["window_id"],
                "seq": window["seq"],
                "case_type": window["case_type"],
                "case_rank": window["case_rank"],
                "window_chunk_start": window["chunk_start"],
                "window_chunk_end": window["chunk_end"],
                "case": case,
                "frame": int(frame),
                "primary_chunk_id": int(chunk_id),
                "local_frame_in_primary_chunk": int(frame - chunk_id * (int(args.chunk_size) - int(args.chunk_overlap))),
                "aligned_error_m": float(aligned["err_m"][idx]),
                "aligned_x": float(aligned["aligned_pos"][idx, 0]),
                "aligned_y": float(aligned["aligned_pos"][idx, 1]),
                "aligned_z": float(aligned["aligned_pos"][idx, 2]),
                "gt_x": float(aligned["gt_pos"][idx, 0]),
                "gt_y": float(aligned["gt_pos"][idx, 1]),
                "gt_z": float(aligned["gt_pos"][idx, 2]),
                "sim3_scale_to_gt": float(aligned["scale"]),
                "trajectory": str(run_dir / f"{window['seq']}.txt"),
                "run_dir": str(run_dir),
            }
        )
    return rows


def _hmc_by_global_chunk(run_dir: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in _read_jsonl(run_dir / "hmc_state_hash.jsonl"):
        chunk = row.get("prior_semantic_action_chunk_idx")
        if chunk is None:
            start = row.get("start_frame")
            if start is not None:
                chunk = _primary_chunk_for_frame(int(start), int(row.get("chunk_size") or 32), int(row.get("chunk_overlap") or 3))
        if chunk is not None:
            out[int(chunk)] = row
    return out


def _extract_hmc_summary(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {}
    keys = [
        "memory_ttt_mean_rel_diff",
        "memory_ttt_max_rel_diff",
        "memory_ttt_w0_mean_rel_diff",
        "memory_ttt_w0_max_rel_diff",
        "prior_semantic_role_consumed_any",
        "prior_semantic_role_control_applied",
        "prior_semantic_role_control_mode",
        "prior_semantic_role_control_changed_fraction",
        "prior_semantic_role_counts",
        "prior_R_ttt_role_counts",
        "prior_ttt_write_mean",
        "prior_ttt_write_present",
        "prior_ttt_semantic_write_role_stats",
        "ttt_replay_token_filter_applied",
        "ttt_replay_token_filter_keep_mass",
        "ttt_replay_token_filter_tokens_before",
        "ttt_replay_token_filter_tokens_after",
        "ttt_replay_token_filter_modes",
        "ttt_replay_token_filter_scopes",
        "ttt_replay_token_filter_scope_tokens",
        "ttt_replay_token_filter_scope_mass",
        "ttt_replay_token_filter_semantic_harm_tokens",
        "ttt_replay_token_filter_semantic_harm_scope_tokens",
        "ttt_replay_token_filter_semantic_harm_veto_tokens",
        "ttt_replay_token_filter_semantic_role_missing_true_count",
        "ttt_role_alignment_modes",
        "ttt_role_alignment_available_true_count",
        "ttt_role_alignment_cache_tokens",
        "ttt_role_alignment_full_tokens",
        "ttt_role_alignment_patch_tokens",
        "ttt_role_alignment_special_tokens",
        "ttt_transient_delta_prev_present",
        "ttt_transient_delta_prev_subtract_applied",
        "ttt_transient_delta_stored",
        "ttt_transient_delta_w0_norm_mean",
        "ttt_write_commit_filter_mode",
        "ttt_write_commit_filter_applied",
    ]
    return {key: row.get(key) for key in keys if key in row}


def _chunk_rows_from_frame_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["window_id"],
            row["seq"],
            row["case_type"],
            row["case"],
            row["primary_chunk_id"],
        )
        grouped.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key, vals in sorted(grouped.items()):
        window_id, seq, case_type, case, chunk = key
        out.append(
            {
                "window_id": window_id,
                "seq": seq,
                "case_type": case_type,
                "case": case,
                "primary_chunk_id": chunk,
                "frame_start": min(int(v["frame"]) for v in vals),
                "frame_end": max(int(v["frame"]) for v in vals),
                "frame_count": len(vals),
                "aligned_error_mean_m": _mean([v.get("aligned_error_m") for v in vals]),
                "aligned_error_max_m": _max([v.get("aligned_error_m") for v in vals]),
            }
        )
    return out


def _index_frame_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    return {(str(row["window_id"]), str(row["case"]), int(row["frame"])): row for row in rows}


def _plot_compare(out_dir: Path, delta_rows: list[dict[str, Any]], title: str) -> dict[str, str]:
    if plt is None or not delta_rows:
        return {}
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = np.array([int(row["frame"]) for row in delta_rows], dtype=np.int64)
    cand_err = np.array([float(row["candidate_error_m"]) for row in delta_rows], dtype=np.float64)
    base_err = np.array([float(row["baseline_error_m"]) for row in delta_rows], dtype=np.float64)
    delta = np.array([float(row["delta_error_vs_baseline_m"]) for row in delta_rows], dtype=np.float64)
    gt_x = np.array([float(row["gt_x"]) for row in delta_rows], dtype=np.float64)
    gt_z = np.array([float(row["gt_z"]) for row in delta_rows], dtype=np.float64)
    cand_x = np.array([float(row["candidate_aligned_x"]) for row in delta_rows], dtype=np.float64)
    cand_z = np.array([float(row["candidate_aligned_z"]) for row in delta_rows], dtype=np.float64)
    base_x = np.array([float(row["baseline_aligned_x"]) for row in delta_rows], dtype=np.float64)
    base_z = np.array([float(row["baseline_aligned_z"]) for row in delta_rows], dtype=np.float64)

    paths: dict[str, str] = {}
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frames, base_err, label="baseline error", linewidth=1.4)
    ax.plot(frames, cand_err, label="candidate error", linewidth=1.4)
    ax.bar(frames, delta, label="candidate-baseline", alpha=0.35)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("frame")
    ax.set_ylabel("aligned position error / delta (m)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path = out_dir / "error_over_frame.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["error_over_frame_png"] = str(path)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(gt_x, gt_z, color="black", label="GT", linewidth=1.2)
    ax.plot(base_x, base_z, color="#377eb8", label="baseline aligned", linewidth=1.0, alpha=0.75)
    sc = ax.scatter(cand_x, cand_z, c=delta, cmap="coolwarm", s=18, label="candidate delta")
    fig.colorbar(sc, ax=ax, label="candidate error - baseline error (m)")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.axis("equal")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path = out_dir / "trajectory_error_map_xz.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["trajectory_error_map_xz_png"] = str(path)
    return paths


def _compare_rows(
    *,
    frame_rows: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    index = _index_frame_rows(frame_rows)
    candidate_cases = _parse_csv_list(args.candidates)
    delta_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    plot_rows: list[dict[str, Any]] = []
    compare_summaries: list[dict[str, Any]] = []
    for window in windows:
        for candidate in candidate_cases:
            control = PAIRED_RANDOM_CONTROLS.get(candidate)
            frames = sorted(
                {
                    frame
                    for win_id, case, frame in index
                    if win_id == window["window_id"] and case == candidate
                }
            )
            if not frames:
                compare_summaries.append(
                    {
                        "window_id": window["window_id"],
                        "candidate": candidate,
                        "status": "missing_candidate_trajectory",
                    }
                )
                continue
            rows_for_plot: list[dict[str, Any]] = []
            for frame in frames:
                cand = index.get((window["window_id"], candidate, frame))
                base = index.get((window["window_id"], args.baseline, frame))
                if cand is None or base is None:
                    continue
                ctrl = index.get((window["window_id"], control, frame)) if control else None
                row = {
                    "window_id": window["window_id"],
                    "seq": window["seq"],
                    "case_type": window["case_type"],
                    "case_rank": window["case_rank"],
                    "candidate": candidate,
                    "baseline": args.baseline,
                    "paired_random_control": control,
                    "frame": int(frame),
                    "primary_chunk_id": int(cand["primary_chunk_id"]),
                    "candidate_error_m": cand["aligned_error_m"],
                    "baseline_error_m": base["aligned_error_m"],
                    "delta_error_vs_baseline_m": float(cand["aligned_error_m"]) - float(base["aligned_error_m"]),
                    "candidate_aligned_x": cand["aligned_x"],
                    "candidate_aligned_y": cand["aligned_y"],
                    "candidate_aligned_z": cand["aligned_z"],
                    "baseline_aligned_x": base["aligned_x"],
                    "baseline_aligned_y": base["aligned_y"],
                    "baseline_aligned_z": base["aligned_z"],
                    "gt_x": cand["gt_x"],
                    "gt_y": cand["gt_y"],
                    "gt_z": cand["gt_z"],
                    "candidate_run_dir": cand["run_dir"],
                    "baseline_run_dir": base["run_dir"],
                }
                if ctrl is not None:
                    row["paired_random_control_error_m"] = ctrl["aligned_error_m"]
                    row["delta_error_vs_paired_random_control_m"] = float(cand["aligned_error_m"]) - float(ctrl["aligned_error_m"])
                    row["paired_random_control_run_dir"] = ctrl["run_dir"]
                delta_rows.append(row)
                rows_for_plot.append(row)
            if not rows_for_plot:
                compare_summaries.append(
                    {
                        "window_id": window["window_id"],
                        "candidate": candidate,
                        "status": "missing_shared_frames_with_baseline",
                    }
                )
                continue
            ranked = sorted(rows_for_plot, key=lambda r: float(r["delta_error_vs_baseline_m"]), reverse=True)
            chosen = [r for r in ranked if float(r["delta_error_vs_baseline_m"]) > float(args.min_bad_delta_m)]
            if not chosen:
                chosen = ranked[: int(args.top_k_frames)]
            else:
                chosen = chosen[: int(args.top_k_frames)]
            hmc_map = _hmc_by_global_chunk(Path(str(chosen[0]["candidate_run_dir"]))) if chosen else {}
            for rank, row in enumerate(chosen, start=1):
                hmc = _extract_hmc_summary(hmc_map.get(int(row["primary_chunk_id"])))
                selected = dict(row)
                selected.update(
                    {
                        "selection_rank": rank,
                        "selection_reason": "largest_positive_candidate_minus_baseline_frame_error",
                        "stage_c_cache_dir": str(Path(f"results/kitti_preprocess/{window['seq']}/stage_c_cache_semantic_chunks")),
                        "rgb_dir": str(args.data_root / "sequences" / window["seq"] / "image_2"),
                        "visual_probe_chunk": int(row["primary_chunk_id"]),
                        "visual_probe_frames_hint": int(row["frame"]),
                        "hmc_candidate_chunk_summary": hmc,
                    }
                )
                selected_rows.append(selected)
            plot_dir = args.out_dir / "plots" / window["window_id"] / candidate
            plot_paths = _plot_compare(plot_dir, rows_for_plot, f"{window['window_id']} {candidate}")
            plot_rows.append(
                {
                    "window_id": window["window_id"],
                    "candidate": candidate,
                    "shared_frame_count": len(rows_for_plot),
                    **plot_paths,
                }
            )
            compare_summaries.append(
                {
                    "window_id": window["window_id"],
                    "candidate": candidate,
                    "status": "compared",
                    "shared_frame_count": len(rows_for_plot),
                    "mean_delta_error_vs_baseline_m": _mean([r["delta_error_vs_baseline_m"] for r in rows_for_plot]),
                    "max_delta_error_vs_baseline_m": _max([r["delta_error_vs_baseline_m"] for r in rows_for_plot]),
                    "selected_frame_count": len(chosen),
                    **plot_paths,
                }
            )
    return delta_rows, selected_rows, plot_rows, compare_summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts-root", type=Path, default=DEFAULT_ROLLOUTS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--baseline", default="LW1_TTT_SEMANTIC_BASE")
    parser.add_argument("--native-baseline", default="LW0_READPATH_NATIVE")
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--seqs", default="00,01,02,05")
    parser.add_argument("--case-types", default="bad,good")
    parser.add_argument("--max-windows", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--top-k-frames", type=int, default=6)
    parser.add_argument("--min-bad-delta-m", type=float, default=0.0)
    args = parser.parse_args()

    seqs = {seq.zfill(2) for seq in _parse_csv_list(args.seqs)}
    case_types = set(_parse_csv_list(args.case_types))
    windows = [
        window
        for window in _discover_windows(args.rollouts_root)
        if window["seq"] in seqs and window["case_type"] in case_types
    ]
    windows = sorted(windows, key=lambda w: (w["seq"], w["case_type"], w["case_rank"], w["chunk_start"]))
    if int(args.max_windows) > 0:
        windows = windows[: int(args.max_windows)]
    cases_to_load = [args.native_baseline, args.baseline] + _parse_csv_list(args.candidates)
    for candidate in _parse_csv_list(args.candidates):
        control = PAIRED_RANDOM_CONTROLS.get(candidate)
        if control:
            cases_to_load.append(control)
    cases_to_load = list(dict.fromkeys(cases_to_load))

    frame_rows: list[dict[str, Any]] = []
    load_rows: list[dict[str, Any]] = []
    for window in windows:
        gt_path = args.data_root / "poses" / f"{window['seq']}.txt"
        for case in cases_to_load:
            run_dir = Path(window["path"]) / case
            traj = run_dir / f"{window['seq']}.txt"
            load_row = {
                "window_id": window["window_id"],
                "seq": window["seq"],
                "case_type": window["case_type"],
                "case": case,
                "trajectory": str(traj),
                "trajectory_exists": traj.exists(),
            }
            if traj.exists():
                try:
                    aligned = _load_aligned_run(traj, gt_path)
                    rows = _run_rows(window=window, case=case, run_dir=run_dir, aligned=aligned, args=args)
                    frame_rows.extend(rows)
                    load_row.update(
                        {
                            "status": "loaded",
                            "frame_count": len(rows),
                            "sim3_scale_to_gt": float(aligned["scale"]),
                            "aligned_error_mean_m": _mean([r["aligned_error_m"] for r in rows]),
                            "aligned_error_max_m": _max([r["aligned_error_m"] for r in rows]),
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - diagnostics must record exact failure.
                    load_row.update({"status": "failed", "error": repr(exc)})
            else:
                load_row["status"] = "missing"
            load_rows.append(load_row)

    chunk_rows = _chunk_rows_from_frame_rows(frame_rows)
    delta_rows, selected_rows, plot_rows, compare_summaries = _compare_rows(
        frame_rows=frame_rows,
        windows=windows,
        args=args,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "trajectory_load_status.csv", load_rows)
    _write_csv(args.out_dir / "per_frame_error.csv", frame_rows)
    _write_csv(args.out_dir / "per_chunk_error.csv", chunk_rows)
    _write_csv(args.out_dir / "per_frame_candidate_delta.csv", delta_rows)
    _write_csv(args.out_dir / "selected_visual_targets.csv", selected_rows)
    _write_csv(args.out_dir / "plot_manifest.csv", plot_rows)
    summary = {
        "schema": "acl2_v80_ttt_geometry_error_visual_bridge_v1",
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "args": vars(args),
        "window_count": len(windows),
        "cases_loaded": cases_to_load,
        "trajectory_loaded_count": sum(1 for row in load_rows if row.get("status") == "loaded"),
        "trajectory_missing_count": sum(1 for row in load_rows if row.get("status") == "missing"),
        "frame_error_rows": len(frame_rows),
        "candidate_delta_rows": len(delta_rows),
        "selected_visual_targets": len(selected_rows),
        "compare_summaries": compare_summaries,
        "outputs": {
            "trajectory_load_status_csv": str(args.out_dir / "trajectory_load_status.csv"),
            "per_frame_error_csv": str(args.out_dir / "per_frame_error.csv"),
            "per_chunk_error_csv": str(args.out_dir / "per_chunk_error.csv"),
            "per_frame_candidate_delta_csv": str(args.out_dir / "per_frame_candidate_delta.csv"),
            "selected_visual_targets_csv": str(args.out_dir / "selected_visual_targets.csv"),
            "plot_manifest_csv": str(args.out_dir / "plot_manifest.csv"),
        },
    }
    _write_json(args.out_dir / "geometry_error_visual_bridge_summary.json", summary)
    print(
        json.dumps(
            _jsonable(
                {
                    "trajectory_loaded_count": summary["trajectory_loaded_count"],
                    "trajectory_missing_count": summary["trajectory_missing_count"],
                    "candidate_delta_rows": len(delta_rows),
                    "selected_visual_targets": len(selected_rows),
                    "summary": args.out_dir / "geometry_error_visual_bridge_summary.json",
                }
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
