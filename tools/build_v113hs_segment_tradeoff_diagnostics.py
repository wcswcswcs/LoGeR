#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from build_v113hs_baseline_metric_summary import read_w2c_txt, similarity_align


ROLE_NAMES = {
    0: "dynamic",
    1: "boundary_lowpurity",
    2: "weak_context",
    3: "stable_landmark",
    4: "vegetation_repetitive",
    5: "sky_lowobs",
    6: "unknown_lowtrust",
}


def load_xyz(output_root: Path, seq: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose_dir = output_root / seq / "02" / "poses"
    pred_frames, pred_w2c = read_w2c_txt(pose_dir / "abs_pose.txt")
    gt_frames, gt_w2c = read_w2c_txt(pose_dir / "gt_abs_pose.txt")
    if pred_frames != gt_frames:
        raise ValueError(f"frame mismatch for {seq}: pred={len(pred_frames)} gt={len(gt_frames)}")
    pred_c2w = np.linalg.inv(pred_w2c)
    gt_c2w = np.linalg.inv(gt_w2c)
    return np.asarray(pred_frames, dtype=np.int64), pred_c2w[:, :3, 3], gt_c2w[:, :3, 3]


def aligned_errors(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> np.ndarray:
    scale, rot, trans = similarity_align(pred_xyz, gt_xyz, with_scale=True)
    aligned = (scale * (rot @ pred_xyz.T)).T + trans[None]
    return np.linalg.norm(aligned - gt_xyz, axis=1)


def semantic_stats(sem_root: Path, seq: str, start: int, end: int) -> dict[str, float]:
    roles = np.load(sem_root / f"seq{seq}_role_ids.npy", mmap_mode="r")
    risk = np.load(sem_root / f"seq{seq}_risk.npy", mmap_mode="r")
    stable = np.load(sem_root / f"seq{seq}_stable.npy", mmap_mode="r")
    start = max(0, int(start))
    end = min(int(end), int(roles.shape[0]))
    out: dict[str, float] = {}
    if end <= start:
        for name in ROLE_NAMES.values():
            out[f"semantic_{name}_mass"] = float("nan")
        out["semantic_risk_mean"] = float("nan")
        out["semantic_stable_value_mean"] = float("nan")
        return out
    role_slice = np.asarray(roles[start:end])
    denom = float(max(role_slice.size, 1))
    for role_id, name in ROLE_NAMES.items():
        out[f"semantic_{name}_mass"] = float(np.count_nonzero(role_slice == role_id) / denom)
    out["semantic_risk_mean"] = float(np.mean(risk[start:end]))
    out["semantic_stable_value_mean"] = float(np.mean(stable[start:end]))
    out["semantic_stable_lowrisk_product"] = float(np.mean(stable[start:end] * np.clip(1.0 - risk[start:end], 0.0, 1.0)))
    return out


def corr(xs: list[float], ys: list[float]) -> float | None:
    arr_x = np.asarray(xs, dtype=np.float64)
    arr_y = np.asarray(ys, dtype=np.float64)
    valid = np.isfinite(arr_x) & np.isfinite(arr_y)
    if int(np.count_nonzero(valid)) < 3:
        return None
    if float(np.std(arr_x[valid])) <= 1e-12 or float(np.std(arr_y[valid])) <= 1e-12:
        return None
    return float(np.corrcoef(arr_x[valid], arr_y[valid])[0, 1])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose v113-HS segment-scale tradeoffs.")
    parser.add_argument("--results-root", default="results/acl2_v113hs_horizonstream_semantic_aware_geometric_evidence_influence")
    parser.add_argument("--seqs", default="00,02")
    parser.add_argument("--baseline-template", default="outputs/baseline_kitti_{seq}")
    parser.add_argument("--baseline-template-by-seq-json", default="")
    parser.add_argument("--candidate-template", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--semantic-root", default="")
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--stride", type=int, default=50)
    args = parser.parse_args()

    root = Path(args.results_root)
    sem_root = Path(args.semantic_root) if str(args.semantic_root).strip() else root / "semantic_projection"
    baseline_template_by_seq = json.loads(args.baseline_template_by_seq_json) if args.baseline_template_by_seq_json.strip() else {}
    rows: list[dict[str, Any]] = []
    for seq in [s.strip() for s in args.seqs.split(",") if s.strip()]:
        baseline_template = baseline_template_by_seq.get(seq, args.baseline_template)
        base_frames, base_pred_xyz, gt_xyz = load_xyz(root / baseline_template.format(seq=seq), seq)
        cand_frames, cand_pred_xyz, cand_gt_xyz = load_xyz(root / args.candidate_template.format(seq=seq), seq)
        if not np.array_equal(base_frames, cand_frames) or not np.allclose(gt_xyz, cand_gt_xyz):
            raise ValueError(f"frame/GT mismatch for {seq}")
        base_errors = aligned_errors(base_pred_xyz, gt_xyz)
        cand_errors = aligned_errors(cand_pred_xyz, gt_xyz)
        for start in range(0, len(gt_xyz) - args.window, args.stride):
            end = start + args.window
            gt_dist = float(np.linalg.norm(gt_xyz[end] - gt_xyz[start]))
            base_dist = float(np.linalg.norm(base_pred_xyz[end] - base_pred_xyz[start]))
            cand_dist = float(np.linalg.norm(cand_pred_xyz[end] - cand_pred_xyz[start]))
            if gt_dist <= 1e-9 or base_dist <= 1e-9 or cand_dist <= 1e-9:
                continue
            base_log = math.log(base_dist / gt_dist)
            cand_log = math.log(cand_dist / gt_dist)
            row: dict[str, Any] = {
                "seq": seq,
                "candidate_name": args.candidate_name,
                "start": int(start),
                "end": int(end),
                "gt_dist": gt_dist,
                "baseline_pred_dist": base_dist,
                "candidate_pred_dist": cand_dist,
                "baseline_segment_scale_log_error": base_log,
                "candidate_segment_scale_log_error": cand_log,
                "baseline_segment_scale_abs_log_error": abs(base_log),
                "candidate_segment_scale_abs_log_error": abs(cand_log),
                "segment_scale_abs_delta_candidate_minus_baseline": abs(cand_log) - abs(base_log),
                "segment_scale_abs_rel_improvement": (abs(base_log) - abs(cand_log)) / abs(base_log) if abs(base_log) > 1e-12 else None,
                "baseline_segment_ate_rmse": float(np.sqrt(np.mean(base_errors[start:end] ** 2))),
                "candidate_segment_ate_rmse": float(np.sqrt(np.mean(cand_errors[start:end] ** 2))),
            }
            b_ate = row["baseline_segment_ate_rmse"]
            c_ate = row["candidate_segment_ate_rmse"]
            row["segment_ate_rel_improvement"] = (b_ate - c_ate) / b_ate if b_ate > 1e-12 else None
            row.update(semantic_stats(sem_root, seq, start, end))
            rows.append(row)

    diag = root / "diagnostics"
    write_csv(diag / f"{args.output_prefix}_segment_tradeoff_rows.csv", rows)
    deltas = [float(r["segment_scale_abs_delta_candidate_minus_baseline"]) for r in rows]
    ate_impr = [float(r["segment_ate_rel_improvement"]) for r in rows if r["segment_ate_rel_improvement"] is not None]
    summary = {
        "candidate_name": args.candidate_name,
        "seqs": [s.strip() for s in args.seqs.split(",") if s.strip()],
        "window": args.window,
        "stride": args.stride,
        "row_count": len(rows),
        "mean_segment_scale_abs_delta_candidate_minus_baseline": float(np.mean(deltas)) if deltas else None,
        "median_segment_scale_abs_delta_candidate_minus_baseline": float(np.median(deltas)) if deltas else None,
        "worse_segment_count": int(sum(d > 0 for d in deltas)),
        "better_segment_count": int(sum(d < 0 for d in deltas)),
        "median_segment_ate_rel_improvement": float(np.median(ate_impr)) if ate_impr else None,
        "correlations_with_scale_abs_delta": {
            "semantic_stable_value_mean": corr(deltas, [float(r["semantic_stable_value_mean"]) for r in rows]),
            "semantic_risk_mean": corr(deltas, [float(r["semantic_risk_mean"]) for r in rows]),
            "semantic_stable_lowrisk_product": corr(deltas, [float(r["semantic_stable_lowrisk_product"]) for r in rows]),
            "semantic_weak_context_mass": corr(deltas, [float(r["semantic_weak_context_mass"]) for r in rows]),
            "semantic_sky_lowobs_mass": corr(deltas, [float(r["semantic_sky_lowobs_mass"]) for r in rows]),
        },
        "top_worse_segments": sorted(rows, key=lambda r: float(r["segment_scale_abs_delta_candidate_minus_baseline"]), reverse=True)[:10],
        "top_better_segments": sorted(rows, key=lambda r: float(r["segment_scale_abs_delta_candidate_minus_baseline"]))[:10],
    }
    (diag / f"{args.output_prefix}_segment_tradeoff_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: summary[k] for k in summary if k not in {"top_worse_segments", "top_better_segments"}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
