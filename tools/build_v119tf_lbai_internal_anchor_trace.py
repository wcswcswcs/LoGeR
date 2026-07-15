#!/usr/bin/env python3
"""Build current-code LB-AI anchor internal utility traces for v119."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_representation_repair"
SOURCE_RUN_NAME = os.environ.get(
    "ACL2_V119_LBAI_INTERNAL_TRACE_SOURCE_RUN",
    "stage2_lbai_form2_anchor_token_routing_value_scaling_sdpa_repair",
).strip()
SOURCE_ROOT = RESULT_ROOT / SOURCE_RUN_NAME
OUT_ROOT = RESULT_ROOT / "stage2_lbai_internal_anchor_trace"
SEQ_LENGTHS = {"00": 4541, "02": 4661}
FRAME_LIMIT = int(os.environ.get("ACL2_V119_LBAI_INTERNAL_TRACE_FRAME_LIMIT", "32"))


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_traj(path: Path) -> dict[int, np.ndarray]:
    poses: dict[int, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 13:
                continue
            frame = int(parts[0])
            vals = [float(value) for value in parts[1:13]]
            poses[frame] = np.array([vals[3], vals[7], vals[11]], dtype=np.float64)
    return poses


def confidence_stats(path: Path) -> dict[str, float]:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0
    return {
        "confidence_mean": float(arr.mean()),
        "confidence_std": float(arr.std()),
        "confidence_p10": float(np.quantile(arr, 0.10)),
        "confidence_p90": float(np.quantile(arr, 0.90)),
    }


def normalize(values: dict[int, float]) -> dict[int, float]:
    lo = min(values.values())
    hi = max(values.values())
    if abs(hi - lo) < 1e-12:
        return {key: 0.5 for key in values}
    return {key: (value - lo) / (hi - lo) for key, value in values.items()}


def quantile_bucket(values: dict[int, float], frame: int, bucket_count: int = 4) -> str:
    sorted_values = sorted(values.values())
    value = values[frame]
    rank = sum(1 for item in sorted_values if item <= value) - 1
    denom = max(1, len(sorted_values) - 1)
    bucket = min(bucket_count - 1, int(math.floor((rank / denom) * bucket_count)))
    return f"q{bucket + 1}_of_{bucket_count}"


def no_action_root(seq: str) -> Path:
    return (
        SOURCE_ROOT
        / "workspace"
        / f"kitti_v119_lbai_form2_seq{seq}"
        / seq
        / f"lingbot_map_v119_lbai_f2_ai0_default_no_action_{seq}"
    )


def build_rows_for_seq(seq: str) -> list[dict[str, Any]]:
    root = no_action_root(seq)
    traj_path = root / "traj.txt"
    if not traj_path.exists():
        raise FileNotFoundError(traj_path)
    poses = read_traj(traj_path)
    frames = list(range(min(FRAME_LIMIT, SEQ_LENGTHS[seq])))
    missing = [frame for frame in frames if frame not in poses]
    if missing:
        raise RuntimeError(f"missing pose rows for seq{seq}: {missing[:10]}")

    step_delta: dict[int, float] = {}
    baseline_delta: dict[int, float] = {}
    min_prior_delta: dict[int, float] = {}
    conf_mean: dict[int, float] = {}
    raw_rows: dict[int, dict[str, Any]] = {}
    origin = poses[frames[0]]
    for frame in frames:
        conf_path = root / "confidence" / f"{frame:06d}.jpg"
        if not conf_path.exists():
            raise FileNotFoundError(conf_path)
        stats = confidence_stats(conf_path)
        conf_mean[frame] = stats["confidence_mean"]
        previous = poses[frame - 1] if frame > frames[0] and (frame - 1) in poses else poses[frame]
        step_delta[frame] = float(np.linalg.norm(poses[frame] - previous))
        baseline_delta[frame] = float(np.linalg.norm(poses[frame] - origin))
        if frame == frames[0]:
            min_prior_delta[frame] = 0.0
        else:
            min_prior_delta[frame] = float(
                min(np.linalg.norm(poses[frame] - poses[prior]) for prior in frames if prior < frame)
            )
        raw_rows[frame] = {
            "schema": "acl2_v119tf_lbai_internal_anchor_trace_row_v1",
            "seq": seq,
            "frame_id": frame,
            "source_run_name": SOURCE_RUN_NAME,
            "source_method": f"lingbot_map_v119_lbai_f2_ai0_default_no_action_{seq}",
            "source_no_action_root": rel(root),
            **stats,
            "pose_step_delta": step_delta[frame],
            "pose_baseline_delta": baseline_delta[frame],
            "pose_min_prior_delta": min_prior_delta[frame],
        }

    step_n = normalize(step_delta)
    baseline_n = normalize(baseline_delta)
    prior_n = normalize(min_prior_delta)
    conf_n = normalize(conf_mean)
    scores: dict[int, float] = {}
    rows: list[dict[str, Any]] = []
    for frame in frames:
        redundancy_penalty = 1.0 - prior_n[frame]
        score = (
            0.35 * step_n[frame]
            + 0.25 * baseline_n[frame]
            + 0.30 * conf_n[frame]
            + 0.10 * prior_n[frame]
            - 0.20 * redundancy_penalty
        )
        scores[frame] = max(0.0, min(1.0, float(score)))
    for frame in frames:
        rows.append(
            {
                **raw_rows[frame],
                "pose_step_delta_norm": step_n[frame],
                "pose_baseline_delta_norm": baseline_n[frame],
                "pose_min_prior_delta_norm": prior_n[frame],
                "confidence_mean_norm": conf_n[frame],
                "redundancy_penalty": 1.0 - prior_n[frame],
                "internal_anchor_utility_score": scores[frame],
                "internal_anchor_bucket_q4": quantile_bucket(scores, frame, bucket_count=4),
                "internal_anchor_bucket_q2": quantile_bucket(scores, frame, bucket_count=2),
                "internal_score_version": "v119_lbai_noaction_pose_confidence_viewnovelty_redundancy_v1",
                "runtime_cue_boundary": (
                    "current-code no-action outputs only: predicted C2W translation, model confidence JPG, "
                    "no GT, no external depth, no SLAM, no external geometry model"
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    rows: list[dict[str, Any]] = []
    for seq in sorted(SEQ_LENGTHS):
        rows.extend(build_rows_for_seq(seq))

    rows_csv = OUT_ROOT / "lbai_internal_anchor_utility_rows.csv"
    write_csv(rows_csv, rows)
    trace_hash = sha256_file(rows_csv)
    summary = {
        "schema": "acl2_v119tf_lbai_internal_anchor_trace_summary_v1",
        "source_run_name": SOURCE_RUN_NAME,
        "source_run_root": rel(SOURCE_ROOT),
        "rows_csv": rel(rows_csv),
        "trace_hash": trace_hash,
        "row_count": len(rows),
        "sequences": sorted(SEQ_LENGTHS),
        "frame_limit": FRAME_LIMIT,
        "internal_score_version": "v119_lbai_noaction_pose_confidence_viewnovelty_redundancy_v1",
        "formula": (
            "clip(0.35*step_delta_norm + 0.25*baseline_delta_norm + "
            "0.30*confidence_mean_norm + 0.10*min_prior_delta_norm - "
            "0.20*(1-min_prior_delta_norm), 0, 1)"
        ),
        "truthfulness_boundary": (
            "This is a current-code internal utility trace from LingBot no-action outputs only. "
            "It is not semantic and not an external/GT geometry cue."
        ),
    }
    summary_path = OUT_ROOT / "lbai_internal_anchor_utility_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
