#!/usr/bin/env python3
"""Audit actionable fresh-validation inputs for ACL2 v118 Stage4 LB-AR.

R47 produced a safety-gated source-value policy, but it is not a blind success:
the rule was assembled after seeing the reused 01/05 stress result.  This
builder checks which untouched KITTI odometry sequences can support a real
fresh validation chain and which upstream artifacts still need to be built.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
STAGE = RESULT_ROOT / "stage4_r48_lingbot_ar_fresh_validation_readiness"
SUMMARY_DIR = STAGE / "summary"

RAW_ROOT = ROOT / "data/kitti/dataset"
POSE_ROOT = RAW_ROOT / "poses"
SEM_ROOT = ROOT / "results/kitti_preprocess"
V105_STAGE1 = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility/stage1_lingbot_baseline"
V105_WORKSPACE = V105_STAGE1 / "workspace"
V105_DATASET = "kitti_v105_00_01_02_05"
DEFAULT_METHOD = "lingbot_map_stream_default"
FLASHINFER_METHOD = "lingbot_map_stream_flashinfer_v118_r15_full"

V108_STAGE2 = ROOT / "results/acl2_v108tf_lingbot_memory_operation_action_surface_search/stage2_semantic_cue_bank"
V108_SUMMARY = V108_STAGE2 / "stage2_summary.json"
V108_TOKEN_ROWS = V108_STAGE2 / "token_semantic_rows.csv"
R20_SUPPORT = RESULT_ROOT / "stage4_r20_lingbot_semantic_bridge_audit/summary/stage4_r20_frame_semantic_support_rows.csv"
R27_SUPPORT = RESULT_ROOT / "stage4_r27_holdout_cue_prep/summary/stage4_r27_holdout_frame_semantic_support_rows.csv"
TRACE_DIR = RESULT_ROOT / "stage3_r14_lingbot_flashinfer_internal_signal_probe/runtime_full"
R45_TOKEN_ROOT = ROOT / "results/acl2_v116tf_fast_semantic_causal_memory_influence/task2_l2t/token_semantics"
R42_TOKEN_ROOT = RESULT_ROOT / "stage4_r42_lingbot_ar_token_gated_oriented_source_value_holdout/token_semantics"

SEQUENCES = tuple(f"{idx:02d}" for idx in range(11))
TRAINED_SEQUENCES = {"00", "01", "02", "05"}
FRESH_CANDIDATES = tuple(seq for seq in SEQUENCES if seq not in TRAINED_SEQUENCES)
CHUNK_RE = re.compile(r"_(\d{6})_(\d{6})$")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, tuple):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(data), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def count_image_frames(seq: str) -> int:
    image_dir = RAW_ROOT / "sequences" / seq / "image_2"
    if not image_dir.is_dir():
        return 0
    return sum(1 for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"})


def count_pose_rows(seq: str) -> int:
    path = POSE_ROOT / f"{seq}.txt"
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def chunk_stats(seq: str) -> dict[str, Any]:
    root = SEM_ROOT / seq / "stage_c_cache_semantic_chunks"
    intervals: list[tuple[int, int]] = []
    for path in sorted(root.glob("chunk_*/masklet.pt")):
        match = CHUNK_RE.search(path.parent.name)
        if not match:
            continue
        intervals.append((int(match.group(1)), int(match.group(2))))
    covered: set[int] = set()
    for start, end in intervals:
        covered.update(range(start, end))
    return {
        "semantic_chunk_root_exists": root.is_dir(),
        "semantic_chunk_masklet_count": len(intervals),
        "semantic_chunk_frame_coverage": len(covered),
        "semantic_chunk_first_frame": min(covered) if covered else "",
        "semantic_chunk_last_frame": max(covered) if covered else "",
    }


def count_csv_by_seq(path: Path, seq_key: str = "seq") -> Counter[str]:
    counts: Counter[str] = Counter()
    if not path.is_file():
        return counts
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        key = seq_key if seq_key in (reader.fieldnames or []) else "seq_id"
        for row in reader:
            seq = str(row.get(key, "")).zfill(2)
            if seq:
                counts[seq] += 1
    return counts


def token_tensor_ready(seq: str) -> tuple[bool, str]:
    for root in (R45_TOKEN_ROOT, R42_TOKEN_ROOT):
        required = [root / f"seq{seq}_{name}.npy" for name in ("dynamic", "boundary", "lowtrust", "stable", "filled")]
        if all(path.is_file() for path in required):
            return True, rel(root)
    return False, ""


def trace_line_count(seq: str) -> int:
    path = TRACE_DIR / f"seq{seq}_flashinfer_trace.jsonl"
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def complete_exists(seq: str, method: str) -> bool:
    return (V105_WORKSPACE / V105_DATASET / seq / method / ".complete.json").is_file()


def eval_exists(seq: str, method: str) -> bool:
    return (V105_WORKSPACE / V105_DATASET / seq / method / "eval/traj.json").is_file()


def main() -> int:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    stage2_summary = read_json(V108_SUMMARY)
    stage2_frame_universe = {str(k).zfill(2): int(v) for k, v in dict(stage2_summary.get("frame_universe_by_seq", {})).items()}
    token_rows_by_seq = count_csv_by_seq(V108_TOKEN_ROWS, "seq_id")
    support_r20_by_seq = count_csv_by_seq(R20_SUPPORT, "seq")
    support_r27_by_seq = count_csv_by_seq(R27_SUPPORT, "seq")

    rows: list[dict[str, Any]] = []
    for seq in SEQUENCES:
        image_frames = count_image_frames(seq)
        pose_rows = count_pose_rows(seq)
        chunks = chunk_stats(seq)
        token_ready, token_root = token_tensor_ready(seq)
        support_rows = support_r20_by_seq.get(seq, 0) + support_r27_by_seq.get(seq, 0)
        trace_rows = trace_line_count(seq)
        raw_pose_ready = image_frames > 0 and pose_rows > 0 and image_frames == pose_rows
        semantic_ready = bool(chunks["semantic_chunk_root_exists"] and chunks["semantic_chunk_masklet_count"] > 0)
        baseline_ready = complete_exists(seq, DEFAULT_METHOD) and eval_exists(seq, DEFAULT_METHOD)
        trace_ready = trace_rows > 0
        existing_token_rows = int(token_rows_by_seq.get(seq, 0))
        cue_bank_ready = seq in stage2_frame_universe and existing_token_rows > 0
        fresh_input_ready = raw_pose_ready and semantic_ready
        direct_r47_ready = bool(fresh_input_ready and baseline_ready and trace_ready and token_ready and support_rows)
        rows.append(
            {
                "schema": "acl2_v118tf_stage4_r48_fresh_validation_readiness_row_v1",
                "seq": seq,
                "was_used_in_r45_r46_reused_set": seq in TRAINED_SEQUENCES,
                "fresh_candidate": seq in FRESH_CANDIDATES,
                "raw_image_frames": image_frames,
                "pose_rows": pose_rows,
                "raw_pose_ready": raw_pose_ready,
                **chunks,
                "fresh_input_ready": fresh_input_ready,
                "v105_default_complete": complete_exists(seq, DEFAULT_METHOD),
                "v105_default_eval": eval_exists(seq, DEFAULT_METHOD),
                "r15_flashinfer_complete": complete_exists(seq, FLASHINFER_METHOD),
                "r15_flashinfer_eval": eval_exists(seq, FLASHINFER_METHOD),
                "baseline_ready_for_eval": baseline_ready,
                "internal_trace_rows": trace_rows,
                "internal_trace_ready": trace_ready,
                "v108_stage2_frame_universe": stage2_frame_universe.get(seq, 0),
                "v108_token_semantic_rows": existing_token_rows,
                "cue_bank_ready": cue_bank_ready,
                "frame_semantic_support_rows": support_rows,
                "frame_semantic_support_ready": support_rows > 0,
                "token_tensor_ready": token_ready,
                "token_tensor_root": token_root,
                "direct_r47_fresh_validation_ready": direct_r47_ready,
            }
        )

    fresh_rows = [row for row in rows if row["fresh_candidate"]]
    buildable = [
        row for row in fresh_rows
        if row["fresh_input_ready"] and not row["direct_r47_fresh_validation_ready"]
    ]
    direct_ready = [row for row in fresh_rows if row["direct_r47_fresh_validation_ready"]]
    recommended = sorted(buildable, key=lambda row: (int(row["raw_image_frames"]), row["seq"]))[:2]
    missing_categories: Counter[str] = Counter()
    for row in buildable:
        if not row["baseline_ready_for_eval"]:
            missing_categories["baseline_workspace"] += 1
        if not row["internal_trace_ready"]:
            missing_categories["internal_trace"] += 1
        if not row["cue_bank_ready"]:
            missing_categories["semantic_cue_bank"] += 1
        if not row["frame_semantic_support_ready"]:
            missing_categories["frame_semantic_support"] += 1
        if not row["token_tensor_ready"]:
            missing_categories["token_tensor"] += 1

    if direct_ready:
        decision = "FRESH_VALIDATION_READY_TO_RUN_EXISTING_ARTIFACTS"
    elif buildable:
        decision = "FRESH_VALIDATION_PREP_REQUIRED_BASELINE_TRACE_TOKEN_BANK_MISSING"
    else:
        decision = "NO_FRESH_SEQUENCE_INPUTS_READY"

    summary = {
        "schema": "acl2_v118tf_stage4_r48_lingbot_ar_fresh_validation_readiness_summary_v1",
        "stage4_r48_decision": decision,
        "global_goal_achieved": False,
        "boundary": (
            "R48 does not evaluate geometry and does not promote R47. It checks whether "
            "untouched KITTI sequences have the upstream artifacts needed for a blind "
            "R47 validation."
        ),
        "trained_or_reused_sequences": sorted(TRAINED_SEQUENCES),
        "fresh_candidate_sequences": list(FRESH_CANDIDATES),
        "direct_ready_fresh_sequences": [row["seq"] for row in direct_ready],
        "buildable_fresh_sequences": [row["seq"] for row in buildable],
        "recommended_next_fresh_sequences": [row["seq"] for row in recommended],
        "missing_categories_for_buildable_fresh_sequences": dict(sorted(missing_categories.items())),
        "outputs": {
            "rows": rel(SUMMARY_DIR / "stage4_r48_fresh_validation_readiness_rows.csv"),
            "summary": rel(SUMMARY_DIR / "stage4_r48_fresh_validation_readiness_summary.json"),
            "report": rel(SUMMARY_DIR / "STAGE4_R48_FRESH_VALIDATION_READINESS_REPORT.md"),
        },
        "next_steps": [
            "Build fresh dataset/configs and default/trace baseline for recommended sequences.",
            "Build fresh semantic cue bank rows, frame support rows, and token tensors from existing semantic chunks.",
            "Pre-register the R47 rule before running candidate/opposite/random controls on those fresh sequences.",
        ],
    }

    write_csv(SUMMARY_DIR / "stage4_r48_fresh_validation_readiness_rows.csv", rows)
    write_json(SUMMARY_DIR / "stage4_r48_fresh_validation_readiness_summary.json", summary)

    report_lines = [
        "# ACL2 v118 Stage4-R48 Fresh Validation Readiness",
        "",
        f"- decision: `{decision}`",
        f"- global_goal_achieved: `{summary['global_goal_achieved']}`",
        f"- recommended_next_fresh_sequences: `{','.join(summary['recommended_next_fresh_sequences'])}`",
        "",
        "| seq | fresh | raw/pose | sem chunks | baseline | trace | cue bank | support | token tensor | direct ready |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report_lines.append(
            f"| {row['seq']} | {row['fresh_candidate']} | {row['raw_pose_ready']} | "
            f"{row['semantic_chunk_masklet_count']} | {row['baseline_ready_for_eval']} | "
            f"{row['internal_trace_ready']} | {row['cue_bank_ready']} | "
            f"{row['frame_semantic_support_ready']} | {row['token_tensor_ready']} | "
            f"{row['direct_r47_fresh_validation_ready']} |"
        )
    report_lines += [
        "",
        "## Boundary",
        "",
        summary["boundary"],
        "",
        "## Recommended Next Work",
        "",
    ]
    for step in summary["next_steps"]:
        report_lines.append(f"- {step}")
    (SUMMARY_DIR / "STAGE4_R48_FRESH_VALIDATION_READINESS_REPORT.md").write_text(
        "\n".join(report_lines).rstrip() + "\n",
        encoding="utf-8",
    )

    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
