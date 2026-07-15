#!/usr/bin/env python3
"""Summarize ACL2 v119-TF Stage1 LB-SCHED random-anchor schedule exactness."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import argparse
from pathlib import Path
from typing import Any


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
DATASET = "kitti_v119_stage1_lbsched_seq00"
VARIANTS: dict[str, str] = {}


def configure_seq(seq: str) -> None:
    global SEQ, NUM_FRAMES, DATASET, VARIANTS
    if seq not in SEQ_NUM_FRAMES:
        raise SystemExit(f"unsupported seq={seq!r}; expected one of {sorted(SEQ_NUM_FRAMES)}")
    SEQ = seq
    NUM_FRAMES = SEQ_NUM_FRAMES[seq]
    DATASET = f"kitti_v119_stage1_lbsched_seq{seq}"
    VARIANTS = {
        variant: f"lingbot_map_v119_lbsched_{variant}_{seq}"
        for variant in ("frozen_delayed_b1_like", "frozen_spread_seed0")
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


def method_root(method: str) -> Path:
    return WORKSPACE / DATASET / SEQ / method


def parse_scale_indices(rows: list[dict[str, Any]]) -> list[int]:
    scale_rows = [row for row in rows if bool(row.get("anchor_scale_frame"))]
    observed = sorted({int(row["sample_position"]) for row in scale_rows})
    declared_values: set[int] = set()
    for row in scale_rows:
        payload = str(row.get("scale_frame_indices", ""))
        for part in payload.split(","):
            part = part.strip()
            if part:
                declared_values.add(int(part))
    if declared_values and sorted(declared_values) != observed:
        raise ValueError(f"declared scale_frame_indices mismatch: declared={sorted(declared_values)} observed={observed}")
    return observed


def summarize_variant(variant: str, method: str, frozen: set[int], expected_hash: str) -> dict[str, Any]:
    action_file = RUN_ROOT / "raw_action" / f"{DATASET}_{SEQ}_{method}.jsonl"
    rows = read_jsonl(action_file)
    scale_indices = parse_scale_indices(rows)
    scale_set = set(scale_indices)
    stream_rows = [row for row in rows if not bool(row.get("anchor_scale_frame"))]
    stream_positions = [int(row["sample_position"]) for row in stream_rows]
    final_keyframes = sorted(
        int(row["sample_position"])
        for row in stream_rows
        if bool(row.get("final_is_keyframe"))
    )
    base_keyframes = sorted(
        int(row["sample_position"])
        for row in stream_rows
        if bool(row.get("base_is_keyframe"))
    )
    expected_downstream = sorted(frozen - scale_set)
    expected_stream_positions = set(range(NUM_FRAMES)) - scale_set
    stream_set = set(stream_positions)
    duplicate_stream_count = len(stream_positions) - len(stream_set)
    missing_stream = sorted(expected_stream_positions - stream_set)
    extra_stream = sorted(stream_set - expected_stream_positions)
    missing_keyframes = sorted(set(expected_downstream) - set(final_keyframes))
    extra_keyframes = sorted(set(final_keyframes) - set(expected_downstream))
    modes = sorted({str(row.get("keyframe_schedule_mode", "")) for row in rows})
    hashes = sorted({str(row.get("frozen_keyframe_indices_hash", "")) for row in rows})
    counts = sorted({str(row.get("frozen_keyframe_count", "")) for row in rows})
    root = method_root(method)
    schedule_pass = (
        len(rows) == NUM_FRAMES
        and len(scale_indices) == SCALE_FRAMES
        and duplicate_stream_count == 0
        and not missing_stream
        and not extra_stream
        and final_keyframes == expected_downstream
        and base_keyframes == expected_downstream
        and modes == ["global_frozen"]
        and hashes == [expected_hash]
        and counts == [str(len(frozen))]
    )
    return {
        "variant": variant,
        "method": method,
        "action_file": rel(action_file),
        "row_count": len(rows),
        "expected_row_count": NUM_FRAMES,
        "scale_anchor_count": len(scale_indices),
        "scale_anchor_indices": ";".join(str(idx) for idx in scale_indices),
        "scale_anchor_frozen_overlap_count": len(scale_set & frozen),
        "stream_row_count": len(stream_rows),
        "expected_stream_row_count": NUM_FRAMES - SCALE_FRAMES,
        "duplicate_stream_count": duplicate_stream_count,
        "missing_stream_count": len(missing_stream),
        "extra_stream_count": len(extra_stream),
        "keyframe_schedule_modes": ";".join(modes),
        "frozen_keyframe_hashes": ";".join(hashes),
        "expected_frozen_hash": expected_hash,
        "frozen_keyframe_counts": ";".join(counts),
        "frozen_keyframe_count": len(frozen),
        "expected_downstream_keyframe_count": len(expected_downstream),
        "base_keyframe_count": len(base_keyframes),
        "final_keyframe_count": len(final_keyframes),
        "missing_keyframe_count": len(missing_keyframes),
        "extra_keyframe_count": len(extra_keyframes),
        "missing_keyframe_sample": ";".join(str(idx) for idx in missing_keyframes[:10]),
        "extra_keyframe_sample": ";".join(str(idx) for idx in extra_keyframes[:10]),
        "complete_exists": int((root / ".complete.json").exists()),
        "traj_exists": int((root / "traj.txt").exists()),
        "intrinsics_exists": int((root / "intrinsics.txt").exists()),
        "depth_exr_count": len(list((root / "depth").glob("*.exr"))),
        "confidence_exr_count": len(list((root / "confidence").glob("*.exr"))),
        "random_schedule_exact_pass": bool(schedule_pass),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", default="00", choices=sorted(SEQ_NUM_FRAMES))
    args = parser.parse_args()
    configure_seq(args.seq)

    frozen_indices = default_frozen_indices(NUM_FRAMES)
    frozen = set(frozen_indices)
    expected_hash = frozen_hash(frozen_indices)
    rows = [
        summarize_variant(variant, method, frozen, expected_hash)
        for variant, method in VARIANTS.items()
    ]
    all_pass = all(row["random_schedule_exact_pass"] for row in rows)
    summary = {
        "schema": "acl2_v119tf_stage1_lbsched_random_schedule_summary_v1",
        "seq": SEQ,
        "num_frames": NUM_FRAMES,
        "scale_frames": SCALE_FRAMES,
        "expected_frozen_keyframe_count": len(frozen_indices),
        "expected_frozen_hash": expected_hash,
        "variants": list(VARIANTS),
        "random_schedule_exact_pass": bool(all_pass),
        "rows_csv": rel(output_path("lbsched_random_schedule_summary", "csv")),
        "scope": (
            f"seq{SEQ} random-anchor observed downstream schedule exactness only. "
            "This does not claim default numeric parity, other sequences, or global v119 success."
        ),
    }
    write_csv(output_path("lbsched_random_schedule_summary", "csv"), rows)
    output_path("lbsched_random_schedule_summary", "json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
