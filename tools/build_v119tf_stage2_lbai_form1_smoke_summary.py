#!/usr/bin/env python3
"""Summarize ACL2 v119-TF Stage2 LB-AI Form1 smoke runtime artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_representation_repair"
RUN_ROOT = RESULT_ROOT / "stage2_lbai_form1_anchor_initialization"
WORKSPACE = RUN_ROOT / "workspace"
SEQ_NUM_FRAMES = {
    "00": 4541,
    "02": 4661,
}
SCALE_FRAMES = 8
AUTO_KEYFRAME_THRESHOLD = 320
SMOKE_VARIANTS = [
    "ai0_default_firstn",
    "ai2_semantic_only",
    "ai5_reverse_semantic",
    "ai8_latency_control",
    "ai6_random_seed00",
]


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


def finite_metric(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


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


def log_success(path: Path, required_markers: list[str]) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return all(marker in text for marker in required_markers)


def method_name(seq: str, variant: str) -> str:
    return f"lingbot_map_v119_lbai_f1_{variant}_{seq}"


def dataset_name(seq: str) -> str:
    return f"kitti_v119_lbai_form1_seq{seq}"


def method_root(seq: str, variant: str) -> Path:
    return WORKSPACE / dataset_name(seq) / seq / method_name(seq, variant)


def action_file(seq: str, variant: str) -> Path:
    method = method_name(seq, variant)
    return RUN_ROOT / "raw_action" / f"{dataset_name(seq)}_{seq}_{method}.jsonl"


def worker_log(seq: str, variant: str) -> Path:
    matches = sorted((RUN_ROOT / "logs").glob(f"run_{variant}_seq{seq}_gpu*.log"))
    if not matches:
        return RUN_ROOT / "logs" / f"run_{variant}_seq{seq}.log"
    return matches[0]


def evaluate_log(seq: str, variant: str) -> Path:
    return RUN_ROOT / "logs" / f"evaluate_{variant}_seq{seq}.log"


def summarize_variant(seq: str, variant: str, frozen: set[int], expected_hash: str, num_frames: int) -> dict[str, Any]:
    method = method_name(seq, variant)
    action_path = action_file(seq, variant)
    rows = read_jsonl(action_path)
    scale_indices = parse_scale_indices(rows)
    scale_set = set(scale_indices)
    stream_rows = [row for row in rows if not bool(row.get("anchor_scale_frame"))]
    stream_positions = [int(row["sample_position"]) for row in stream_rows]
    stream_set = set(stream_positions)
    expected_stream_positions = set(range(num_frames)) - scale_set
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
    duplicate_stream_count = len(stream_positions) - len(stream_set)
    missing_stream = sorted(expected_stream_positions - stream_set)
    extra_stream = sorted(stream_set - expected_stream_positions)
    missing_keyframes = sorted(set(expected_downstream) - set(final_keyframes))
    extra_keyframes = sorted(set(final_keyframes) - set(expected_downstream))
    modes = sorted({str(row.get("keyframe_schedule_mode", "")) for row in rows})
    hashes = sorted({str(row.get("frozen_keyframe_indices_hash", "")) for row in rows})
    counts = sorted({str(row.get("frozen_keyframe_count", "")) for row in rows})
    root = method_root(seq, variant)
    traj_path = root / "eval/traj.json"
    traj = json.loads(traj_path.read_text(encoding="utf-8")) if traj_path.exists() else {}
    metrics_present = all(finite_metric(traj.get(key)) for key in ("ate", "rpe_rot", "rpe_trans"))
    worker_ok = log_success(worker_log(seq, variant), ["Completed successfully", "Worker done: 1/1 scenes succeeded"])
    eval_ok = log_success(evaluate_log(seq, variant), ["Total successful: 1", "Total failed: 0"])
    schedule_pass = (
        len(rows) == num_frames
        and len(scale_indices) == SCALE_FRAMES
        and duplicate_stream_count == 0
        and not missing_stream
        and not extra_stream
        and base_keyframes == expected_downstream
        and final_keyframes == expected_downstream
        and modes == ["global_frozen"]
        and hashes == [expected_hash]
        and counts == [str(len(frozen))]
    )
    return {
        "seq": seq,
        "variant": variant,
        "method": method,
        "action_file": rel(action_path),
        "worker_log": rel(worker_log(seq, variant)),
        "evaluate_log": rel(evaluate_log(seq, variant)),
        "row_count": len(rows),
        "expected_row_count": num_frames,
        "scale_anchor_count": len(scale_indices),
        "scale_anchor_indices": ";".join(str(idx) for idx in scale_indices),
        "scale_anchor_frozen_overlap_count": len(scale_set & frozen),
        "stream_row_count": len(stream_rows),
        "expected_stream_row_count": num_frames - SCALE_FRAMES,
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
        "combined_keyframe_count_including_anchor_overlap": len(final_keyframes) + len(scale_set & frozen),
        "missing_keyframe_count": len(missing_keyframes),
        "extra_keyframe_count": len(extra_keyframes),
        "complete_exists": int((root / ".complete.json").exists()),
        "traj_exists": int((root / "traj.txt").exists()),
        "intrinsics_exists": int((root / "intrinsics.txt").exists()),
        "depth_exr_count": len(list((root / "depth").glob("*.exr"))),
        "confidence_exr_count": len(list((root / "confidence").glob("*.exr"))),
        "traj_json": rel(traj_path),
        "ate": traj.get("ate", ""),
        "rpe_rot": traj.get("rpe_rot", ""),
        "rpe_trans": traj.get("rpe_trans", ""),
        "metrics_present": bool(metrics_present),
        "worker_log_success_markers": bool(worker_ok),
        "evaluate_log_success_markers": bool(eval_ok),
        "action_schedule_exact_pass": bool(schedule_pass),
        "runtime_smoke_pass": bool(worker_ok and eval_ok and metrics_present and schedule_pass),
    }


def output_path(seq: str, stem: str, suffix: str) -> Path:
    return RUN_ROOT / f"{stem}_seq{seq}.{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", default="00", choices=sorted(SEQ_NUM_FRAMES))
    args = parser.parse_args()

    seq = args.seq
    num_frames = SEQ_NUM_FRAMES[seq]
    frozen_indices = default_frozen_indices(num_frames)
    frozen = set(frozen_indices)
    expected_hash = frozen_hash(frozen_indices)
    rows = [summarize_variant(seq, variant, frozen, expected_hash, num_frames) for variant in SMOKE_VARIANTS]
    baseline = next((row for row in rows if row["variant"] == "ai0_default_firstn"), None)
    if baseline:
        for row in rows:
            for metric in ("ate", "rpe_rot", "rpe_trans"):
                base_value = baseline.get(metric)
                value = row.get(metric)
                row[f"{metric}_delta_vs_ai0"] = (
                    float(value) - float(base_value)
                    if finite_metric(value) and finite_metric(base_value)
                    else ""
                )
    all_pass = all(row["runtime_smoke_pass"] for row in rows)
    summary = {
        "schema": "acl2_v119tf_stage2_lbai_form1_smoke_summary_v1",
        "seq": seq,
        "num_frames": num_frames,
        "scale_frames": SCALE_FRAMES,
        "expected_frozen_keyframe_count": len(frozen_indices),
        "expected_frozen_hash": expected_hash,
        "variants": SMOKE_VARIANTS,
        "smoke_variant_count": len(rows),
        "runtime_smoke_pass": bool(all_pass),
        "rows_csv": rel(output_path(seq, "lbai_form1_smoke_summary", "csv")),
        "truthfulness_boundary": (
            f"seq{seq} Form1 smoke subset only. This does not claim seq02, all AI6 seeds, "
            "AI1/AI3/AI4/AI7, Form2, or full v119 success."
        ),
    }
    write_csv(output_path(seq, "lbai_form1_smoke_summary", "csv"), rows)
    output_path(seq, "lbai_form1_smoke_summary", "json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
