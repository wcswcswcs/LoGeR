#!/usr/bin/env python3
"""Summarize ACL2 v119-TF Stage1 LB-SCHED parity artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import argparse
from pathlib import Path
from typing import Any

import numpy as np


os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_representation_repair"
RUN_ROOT = RESULT_ROOT / "stage1_lbsched_parity"
WORKSPACE = RUN_ROOT / "workspace"
SEQ = "00"
NUM_FRAMES = 4541
SEQ_NUM_FRAMES = {
    "00": 4541,
    "02": 4661,
}
SCALE_FRAMES = 8
AUTO_KEYFRAME_THRESHOLD = 320
VARIANTS: dict[str, str] = {}
DATASET = "kitti_v119_stage1_lbsched_seq00"
PARITY_TOL = 1e-6


def configure_seq(seq: str) -> None:
    global SEQ, NUM_FRAMES, DATASET, VARIANTS
    if seq not in SEQ_NUM_FRAMES:
        raise SystemExit(f"unsupported seq={seq!r}; expected one of {sorted(SEQ_NUM_FRAMES)}")
    SEQ = seq
    NUM_FRAMES = SEQ_NUM_FRAMES[seq]
    DATASET = f"kitti_v119_stage1_lbsched_seq{seq}"
    VARIANTS = {
        variant: f"lingbot_map_v119_lbsched_{variant}_{seq}"
        for variant in ("legacy_default", "frozen_default_a", "frozen_default_b")
    }


def output_path(stem: str, suffix: str) -> Path:
    seq_suffix = "" if SEQ == "00" else f"_seq{SEQ}"
    return RUN_ROOT / f"{stem}{seq_suffix}.{suffix}"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def default_frozen_indices(num_frames: int) -> list[int]:
    interval = math.ceil(num_frames / AUTO_KEYFRAME_THRESHOLD)
    stream = [idx for idx in range(num_frames) if idx >= SCALE_FRAMES]
    return [idx for pos, idx in enumerate(stream) if interval <= 1 or pos % interval == 0]


def frozen_hash(indices: list[int]) -> str:
    payload = ",".join(str(idx) for idx in sorted(indices))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def method_root(method: str) -> Path:
    return WORKSPACE / DATASET / SEQ / method


def finite_max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    if diff.size == 0:
        return 0.0
    return float(np.nanmax(diff))


def compare_txt(ref: Path, cand: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ref_exists": ref.exists(),
        "cand_exists": cand.exists(),
        "max_abs_diff": float("inf"),
        "shape_match": False,
    }
    if not (ref.exists() and cand.exists()):
        return out
    ref_arr = np.loadtxt(ref)
    cand_arr = np.loadtxt(cand)
    out["shape_match"] = tuple(ref_arr.shape) == tuple(cand_arr.shape)
    out["max_abs_diff"] = finite_max_abs_diff(ref_arr, cand_arr)
    return out


def read_exr(path: Path) -> np.ndarray:
    import cv2

    arr = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"failed to read EXR: {path}")
    return arr


def compare_exr_dir(ref_dir: Path, cand_dir: Path) -> dict[str, Any]:
    ref_files = sorted(path.relative_to(ref_dir) for path in ref_dir.glob("*.exr"))
    cand_files = sorted(path.relative_to(cand_dir) for path in cand_dir.glob("*.exr"))
    missing = sorted(str(path) for path in set(ref_files) - set(cand_files))
    extra = sorted(str(path) for path in set(cand_files) - set(ref_files))
    common = sorted(set(ref_files) & set(cand_files))
    max_abs_diff = 0.0
    max_file = ""
    compared = 0
    for rel_path in common:
        diff = finite_max_abs_diff(read_exr(ref_dir / rel_path), read_exr(cand_dir / rel_path))
        compared += 1
        if diff > max_abs_diff:
            max_abs_diff = diff
            max_file = str(rel_path)
    return {
        "ref_count": len(ref_files),
        "cand_count": len(cand_files),
        "common_count": len(common),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "missing_sample": ";".join(missing[:10]),
        "extra_sample": ";".join(extra[:10]),
        "compared_count": compared,
        "max_abs_diff": float(max_abs_diff),
        "max_abs_diff_file": max_file,
    }


def file_set_counts(root: Path) -> dict[str, int]:
    return {
        "complete_exists": int((root / ".complete.json").exists()),
        "traj_exists": int((root / "traj.txt").exists()),
        "intrinsics_exists": int((root / "intrinsics.txt").exists()),
        "depth_exr_count": len(list((root / "depth").glob("*.exr"))),
        "confidence_exr_count": len(list((root / "confidence").glob("*.exr"))),
        "rgb_jpg_count": len(list((root / "rgb").glob("*.jpg"))),
    }


def summarize_action(variant: str, method: str, expected_set: set[int], expected_hash: str) -> dict[str, Any]:
    action_file = RUN_ROOT / "raw_action" / f"{DATASET}_{SEQ}_{method}.jsonl"
    rows = read_jsonl(action_file)
    sample_positions = [int(row["sample_position"]) for row in rows]
    base_keyframes = sorted(int(row["sample_position"]) for row in rows if bool(row.get("base_is_keyframe")))
    final_keyframes = sorted(int(row["sample_position"]) for row in rows if bool(row.get("final_is_keyframe")))
    modes = sorted({str(row.get("keyframe_schedule_mode", "")) for row in rows})
    hashes = sorted({str(row.get("frozen_keyframe_indices_hash", "")) for row in rows})
    counts = sorted({str(row.get("frozen_keyframe_count", "")) for row in rows})
    duplicate_positions = len(sample_positions) - len(set(sample_positions))
    expected_rows = set(range(SCALE_FRAMES, NUM_FRAMES))
    row_set = set(sample_positions)
    missing_rows = sorted(expected_rows - row_set)
    extra_rows = sorted(row_set - expected_rows)
    missing_keyframes = sorted(expected_set - set(final_keyframes))
    extra_keyframes = sorted(set(final_keyframes) - expected_set)
    return {
        "variant": variant,
        "method": method,
        "action_file": rel(action_file),
        "row_count": len(rows),
        "expected_row_count": NUM_FRAMES - SCALE_FRAMES,
        "duplicate_position_count": duplicate_positions,
        "missing_row_count": len(missing_rows),
        "extra_row_count": len(extra_rows),
        "keyframe_schedule_modes": ";".join(modes),
        "frozen_keyframe_hashes": ";".join(hashes),
        "frozen_keyframe_counts": ";".join(counts),
        "expected_frozen_hash": expected_hash if variant.startswith("frozen_") else "",
        "expected_keyframe_count": len(expected_set),
        "base_keyframe_count": len(base_keyframes),
        "final_keyframe_count": len(final_keyframes),
        "missing_keyframe_count": len(missing_keyframes),
        "extra_keyframe_count": len(extra_keyframes),
        "missing_keyframe_sample": ";".join(str(idx) for idx in missing_keyframes[:10]),
        "extra_keyframe_sample": ";".join(str(idx) for idx in extra_keyframes[:10]),
        "schedule_pass": (
            len(rows) == NUM_FRAMES - SCALE_FRAMES
            and duplicate_positions == 0
            and not missing_rows
            and not extra_rows
            and not missing_keyframes
            and not extra_keyframes
        ),
    }


def summarize_pair(reference: str, candidate: str) -> dict[str, Any]:
    ref_method = VARIANTS[reference]
    cand_method = VARIANTS[candidate]
    ref_root = method_root(ref_method)
    cand_root = method_root(cand_method)
    traj = compare_txt(ref_root / "traj.txt", cand_root / "traj.txt")
    intrinsics = compare_txt(ref_root / "intrinsics.txt", cand_root / "intrinsics.txt")
    depth = compare_exr_dir(ref_root / "depth", cand_root / "depth")
    confidence = compare_exr_dir(ref_root / "confidence", cand_root / "confidence")
    complete_hash_match = (
        sha256_file(ref_root / ".complete.json") == sha256_file(cand_root / ".complete.json")
        if (ref_root / ".complete.json").exists() and (cand_root / ".complete.json").exists()
        else False
    )
    return {
        "reference": reference,
        "candidate": candidate,
        "reference_method": ref_method,
        "candidate_method": cand_method,
        "traj_max_abs_diff": traj["max_abs_diff"],
        "traj_shape_match": traj["shape_match"],
        "intrinsics_max_abs_diff": intrinsics["max_abs_diff"],
        "intrinsics_shape_match": intrinsics["shape_match"],
        "depth_max_abs_diff": depth["max_abs_diff"],
        "depth_max_abs_diff_file": depth["max_abs_diff_file"],
        "depth_ref_count": depth["ref_count"],
        "depth_candidate_count": depth["cand_count"],
        "depth_missing_count": depth["missing_count"],
        "depth_extra_count": depth["extra_count"],
        "confidence_max_abs_diff": confidence["max_abs_diff"],
        "confidence_max_abs_diff_file": confidence["max_abs_diff_file"],
        "confidence_ref_count": confidence["ref_count"],
        "confidence_candidate_count": confidence["cand_count"],
        "confidence_missing_count": confidence["missing_count"],
        "confidence_extra_count": confidence["extra_count"],
        "complete_json_hash_match": complete_hash_match,
        "numeric_parity_pass": (
            bool(traj["shape_match"])
            and bool(intrinsics["shape_match"])
            and float(traj["max_abs_diff"]) <= PARITY_TOL
            and float(intrinsics["max_abs_diff"]) <= PARITY_TOL
            and depth["missing_count"] == 0
            and depth["extra_count"] == 0
            and float(depth["max_abs_diff"]) <= PARITY_TOL
            and confidence["missing_count"] == 0
            and confidence["extra_count"] == 0
            and float(confidence["max_abs_diff"]) <= PARITY_TOL
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", default="00", choices=sorted(SEQ_NUM_FRAMES))
    args = parser.parse_args()
    configure_seq(args.seq)

    expected_indices = default_frozen_indices(NUM_FRAMES)
    expected_set = set(expected_indices)
    expected_hash = frozen_hash(expected_indices)
    action_rows = [
        summarize_action(variant, method, expected_set, expected_hash)
        for variant, method in VARIANTS.items()
    ]
    completeness_rows = []
    for variant, method in VARIANTS.items():
        row = {"variant": variant, "method": method, "root": rel(method_root(method))}
        row.update(file_set_counts(method_root(method)))
        completeness_rows.append(row)
    pair_rows = [
        summarize_pair("legacy_default", "frozen_default_a"),
        summarize_pair("legacy_default", "frozen_default_b"),
        summarize_pair("frozen_default_a", "frozen_default_b"),
    ]
    group_pass = (
        all(row["schedule_pass"] for row in action_rows)
        and all(row["numeric_parity_pass"] for row in pair_rows)
        and all(
            row["complete_exists"]
            and row["traj_exists"]
            and row["intrinsics_exists"]
            and row["depth_exr_count"] == NUM_FRAMES
            and row["confidence_exr_count"] == NUM_FRAMES
            for row in completeness_rows
        )
    )
    summary = {
        "schema": "acl2_v119tf_stage1_lbsched_default_parity_summary_v1",
        "seq": SEQ,
        "num_frames": NUM_FRAMES,
        "scale_frames": SCALE_FRAMES,
        "auto_keyframe_threshold": AUTO_KEYFRAME_THRESHOLD,
        "expected_keyframe_count": len(expected_indices),
        "expected_frozen_hash": expected_hash,
        "parity_tolerance": PARITY_TOL,
        "variants": list(VARIANTS),
        "action_summary_csv": rel(output_path("lbsched_default_action_summary", "csv")),
        "completeness_csv": rel(output_path("lbsched_default_completeness", "csv")),
        "pair_parity_csv": rel(output_path("lbsched_default_pair_parity", "csv")),
        "default_parity_group_pass": bool(group_pass),
        "scope": (
            f"seq{SEQ} default first-n parity only: legacy_default vs frozen_default_a, "
            "legacy_default vs frozen_default_b, and frozen repeat a vs b. "
            "Random-anchor observed schedule match and other sequences are not claimed here."
        ),
    }
    write_csv(output_path("lbsched_default_action_summary", "csv"), action_rows)
    write_csv(output_path("lbsched_default_completeness", "csv"), completeness_rows)
    write_csv(output_path("lbsched_default_pair_parity", "csv"), pair_rows)
    output_path("lbsched_default_parity_summary", "json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
