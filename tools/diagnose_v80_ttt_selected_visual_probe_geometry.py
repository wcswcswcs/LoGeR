#!/usr/bin/env python3
"""Diagnose geometry errors from selected TTT visual microprobes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnose_v80_ttt_geometry_error_visual_bridge import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    _extract_hmc_summary,
    _load_aligned_run,
    _mean,
    _max,
    _plot_compare,
    _primary_chunk_for_frame,
    _write_csv,
    _write_json,
)


DEFAULT_VISUAL_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final/phase5_ttt_geometry_selected_visual_probe"
)
ROLE_NAMES = {"0": "void_or_unset", "1": "positive_stable", "2": "neutral", "3": "negative_harm", "4": "swa_protect"}


def _finite(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _torch_load(path: Path) -> Any:
    if torch is None:
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_label_names(stage_c_dir: Path, chunk: int) -> list[str]:
    matches = sorted(stage_c_dir.glob(f"chunk_{int(chunk):03d}_*/masklet.pt"))
    if not matches:
        return []
    payload = _torch_load(matches[0])
    sem = payload.get("semantic_segmentation") if isinstance(payload, dict) else None
    if not isinstance(sem, dict):
        return []
    return [str(x) for x in sem.get("label_names", [])]


def _role_label_summary(hmc_row: dict[str, Any], label_names: list[str], top_k: int = 12) -> list[dict[str, Any]]:
    counts = hmc_row.get("prior_fine_label_path_role_counts")
    if not isinstance(counts, dict):
        return []
    ttt = counts.get("ttt")
    if not isinstance(ttt, dict):
        return []
    rows: list[dict[str, Any]] = []
    for label_id, role_counts in ttt.items():
        if not isinstance(role_counts, dict):
            continue
        label_idx = int(label_id)
        label_name = label_names[label_idx] if 0 <= label_idx < len(label_names) else str(label_id)
        total = sum(int(v) for v in role_counts.values())
        row = {
            "label_id": label_idx,
            "label_name": label_name,
            "total": int(total),
        }
        for role_id, value in role_counts.items():
            row[f"role_{role_id}_{ROLE_NAMES.get(str(role_id), 'unknown')}"] = int(value)
        row["negative_harm_count"] = int(role_counts.get("3", 0))
        row["positive_stable_count"] = int(role_counts.get("1", 0))
        row["neutral_count"] = int(role_counts.get("2", 0))
        rows.append(row)
    return sorted(rows, key=lambda r: (int(r.get("negative_harm_count", 0)), int(r.get("positive_stable_count", 0)), int(r.get("total", 0))), reverse=True)[:top_k]


def _post_delta_stats(case_dir: Path, chunk: int) -> dict[str, Any]:
    path = case_dir / "ttt_spatial_post_delta_maps" / f"chunk_{int(chunk):03d}_ttt_spatial_post_delta_map.pt"
    payload = _torch_load(path)
    if not isinstance(payload, dict):
        return {"post_delta_path": str(path), "post_delta_loaded": False}
    stats = payload.get("stats") if isinstance(payload.get("stats"), dict) else {}
    out = {"post_delta_path": str(path), "post_delta_loaded": True}
    for key in (
        "ttt_write_prior_patch",
        "committed_post_delta_norm_projection_patch",
        "native_delta_norm_projection_patch",
        "action_delta_norm_projection_patch",
        "D_tok_patch",
    ):
        if key in stats:
            out[f"stats_{key}"] = stats[key]
    return out


def _frame_rows_for_job(job: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    seq = str(job["seq"]).zfill(2)
    case = str(job["case"])
    case_dir = Path(str(job["case_dir"]))
    traj = case_dir / f"{seq}.txt"
    aligned = _load_aligned_run(traj, args.data_root / "poses" / f"{seq}.txt")
    rows: list[dict[str, Any]] = []
    for idx, frame in enumerate(aligned["frames"]):
        chunk = _primary_chunk_for_frame(int(frame), int(args.chunk_size), int(args.chunk_overlap))
        rows.append(
            {
                "target_id": job["target_id"],
                "seq": seq,
                "case": case,
                "frame": int(frame),
                "primary_chunk_id": int(chunk),
                "aligned_error_m": float(aligned["err_m"][idx]),
                "aligned_x": float(aligned["aligned_pos"][idx, 0]),
                "aligned_y": float(aligned["aligned_pos"][idx, 1]),
                "aligned_z": float(aligned["aligned_pos"][idx, 2]),
                "gt_x": float(aligned["gt_pos"][idx, 0]),
                "gt_y": float(aligned["gt_pos"][idx, 1]),
                "gt_z": float(aligned["gt_pos"][idx, 2]),
                "sim3_scale_to_gt": float(aligned["scale"]),
                "trajectory": str(traj),
                "run_dir": str(case_dir),
            }
        )
    return rows


def _job_hmc_summary(job: dict[str, Any], chunk: int, args: argparse.Namespace) -> dict[str, Any]:
    case_dir = Path(str(job["case_dir"]))
    rows = _read_jsonl(case_dir / "hmc_state_hash.jsonl")
    hmc_row = None
    for row in rows:
        if int(row.get("prior_semantic_action_chunk_idx") or -1) == int(chunk):
            hmc_row = row
            break
    if hmc_row is None and rows:
        hmc_row = rows[-1]
    stage_c_dir = Path(str(job.get("source_selected_target", {}).get("stage_c_cache_dir") or f"results/kitti_preprocess/{job['seq']}/stage_c_cache_semantic_chunks"))
    label_names = _load_label_names(stage_c_dir, int(chunk))
    return {
        "target_id": job["target_id"],
        "seq": job["seq"],
        "case": job["case"],
        "chunk": int(chunk),
        "hmc_rows": len(rows),
        **_extract_hmc_summary(hmc_row),
        "top_ttt_role_labels": _role_label_summary(hmc_row or {}, label_names),
        **_post_delta_stats(case_dir, int(chunk)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-root", type=Path, default=DEFAULT_VISUAL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--top-k-frames", type=int, default=8)
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = args.visual_root / "geometry_error_microdiagnostic"
    summary = _read_json(args.visual_root / "selected_geometry_visual_probe_summary.json")
    jobs = summary.get("jobs", []) if isinstance(summary, dict) else []
    if not jobs:
        raise FileNotFoundError(f"no jobs found in {args.visual_root / 'selected_geometry_visual_probe_summary.json'}")

    frame_rows: list[dict[str, Any]] = []
    load_rows: list[dict[str, Any]] = []
    for job in jobs:
        try:
            rows = _frame_rows_for_job(job, args)
            frame_rows.extend(rows)
            load_rows.append(
                {
                    "target_id": job["target_id"],
                    "case": job["case"],
                    "status": "loaded",
                    "frame_count": len(rows),
                    "aligned_error_mean_m": _mean([r["aligned_error_m"] for r in rows]),
                    "aligned_error_max_m": _max([r["aligned_error_m"] for r in rows]),
                }
            )
        except Exception as exc:  # noqa: BLE001
            load_rows.append({"target_id": job.get("target_id"), "case": job.get("case"), "status": "failed", "error": repr(exc)})

    by_target: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        by_target.setdefault(str(job["target_id"]), []).append(job)
    frame_index = {(r["target_id"], r["case"], int(r["frame"])): r for r in frame_rows}
    delta_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    hmc_rows: list[dict[str, Any]] = []
    compare_rows: list[dict[str, Any]] = []
    combined_rows = _read_csv(args.visual_root / "selected_geometry_visual_probe_combined.csv")
    for target_id, target_jobs in sorted(by_target.items()):
        source = target_jobs[0].get("source_selected_target", {})
        baseline = str(source.get("baseline") or "LW1_TTT_SEMANTIC_BASE")
        candidate = str(source.get("candidate") or "")
        control = str(source.get("paired_random_control") or "")
        if not candidate:
            continue
        candidate_frames = sorted(
            int(frame)
            for tid, case, frame in frame_index
            if tid == target_id and case == candidate
        )
        rows_for_plot: list[dict[str, Any]] = []
        for frame in candidate_frames:
            cand = frame_index.get((target_id, candidate, frame))
            base = frame_index.get((target_id, baseline, frame))
            if cand is None or base is None:
                continue
            ctrl = frame_index.get((target_id, control, frame)) if control else None
            row = {
                "target_id": target_id,
                "seq": cand["seq"],
                "candidate": candidate,
                "baseline": baseline,
                "paired_random_control": control,
                "frame": frame,
                "primary_chunk_id": cand["primary_chunk_id"],
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
            }
            if ctrl is not None:
                row["paired_random_control_error_m"] = ctrl["aligned_error_m"]
                row["delta_error_vs_paired_random_control_m"] = float(cand["aligned_error_m"]) - float(ctrl["aligned_error_m"])
            delta_rows.append(row)
            rows_for_plot.append(row)
        ranked = sorted(rows_for_plot, key=lambda r: float(r["delta_error_vs_baseline_m"]), reverse=True)
        selected_rows.extend({**row, "selection_rank": idx + 1} for idx, row in enumerate(ranked[: int(args.top_k_frames)]))
        plot_dir = args.out_dir / "plots" / target_id / candidate
        plot_paths = _plot_compare(plot_dir, rows_for_plot, f"{target_id} {candidate}") if rows_for_plot else {}
        compare_rows.append(
            {
                "target_id": target_id,
                "candidate": candidate,
                "baseline": baseline,
                "paired_random_control": control,
                "shared_frames": len(rows_for_plot),
                "mean_delta_error_vs_baseline_m": _mean([r["delta_error_vs_baseline_m"] for r in rows_for_plot]),
                "max_delta_error_vs_baseline_m": _max([r["delta_error_vs_baseline_m"] for r in rows_for_plot]),
                "mean_delta_error_vs_paired_random_control_m": _mean([r.get("delta_error_vs_paired_random_control_m") for r in rows_for_plot]),
                "max_delta_error_vs_paired_random_control_m": _max([r.get("delta_error_vs_paired_random_control_m") for r in rows_for_plot]),
                **plot_paths,
            }
        )
        chunk = int(source.get("visual_probe_chunk") or ranked[0]["primary_chunk_id"] if ranked else 0)
        for job in target_jobs:
            hmc_rows.append(_job_hmc_summary(job, chunk, args))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "microprobe_trajectory_load_status.csv", load_rows)
    _write_csv(args.out_dir / "microprobe_per_frame_error.csv", frame_rows)
    _write_csv(args.out_dir / "microprobe_candidate_delta.csv", delta_rows)
    _write_csv(args.out_dir / "microprobe_selected_bad_frames.csv", selected_rows)
    _write_csv(args.out_dir / "microprobe_compare_summary.csv", compare_rows)
    _write_csv(args.out_dir / "microprobe_hmc_semantic_summary.csv", hmc_rows)
    _write_csv(args.out_dir / "microprobe_combined_visuals.csv", combined_rows)
    out = {
        "schema": "acl2_v80_ttt_selected_visual_probe_geometry_v1",
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "visual_root": str(args.visual_root),
        "jobs": len(jobs),
        "frame_error_rows": len(frame_rows),
        "candidate_delta_rows": len(delta_rows),
        "selected_bad_frames": selected_rows[: int(args.top_k_frames)],
        "compare_summary": compare_rows,
        "hmc_semantic_summary": hmc_rows,
        "combined_visuals": combined_rows,
        "outputs": {
            "load_status_csv": str(args.out_dir / "microprobe_trajectory_load_status.csv"),
            "per_frame_error_csv": str(args.out_dir / "microprobe_per_frame_error.csv"),
            "candidate_delta_csv": str(args.out_dir / "microprobe_candidate_delta.csv"),
            "selected_bad_frames_csv": str(args.out_dir / "microprobe_selected_bad_frames.csv"),
            "compare_summary_csv": str(args.out_dir / "microprobe_compare_summary.csv"),
            "hmc_semantic_summary_csv": str(args.out_dir / "microprobe_hmc_semantic_summary.csv"),
            "combined_visuals_csv": str(args.out_dir / "microprobe_combined_visuals.csv"),
        },
    }
    _write_json(args.out_dir / "microprobe_geometry_summary.json", out)
    print(
        json.dumps(
            {
                "candidate_delta_rows": len(delta_rows),
                "selected_bad_frames": len(selected_rows),
                "compare_summary": compare_rows,
                "summary": str(args.out_dir / "microprobe_geometry_summary.json"),
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
