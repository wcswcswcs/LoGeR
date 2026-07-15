#!/usr/bin/env python3
"""Summarize ACL2 v119-TF LB-AI Form2 anchor-token routing artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_representation_repair"
RUN_NAME = os.environ.get("ACL2_V119_LBAI_FORM2_RUN_NAME", "stage2_lbai_form2_anchor_token_routing").strip()
RUN_ROOT = RESULT_ROOT / RUN_NAME
WORKSPACE = RUN_ROOT / "workspace"
CONFIG_ROWS = RUN_ROOT / "config_rows.csv"
SEQ_NUM_FRAMES = {"00": 4541, "02": 4661}
SCALE_FRAMES = 8
AUTO_KEYFRAME_THRESHOLD = 320
PRIMARY_VARIANTS = [
    "ai0_default_no_action",
    "ai1_internal_anchor_utility",
    "ai2_semantic_anchor_role",
    "ai3_internal_semantic_anchor_role",
    "ai5_reverse_semantic_anchor_role",
    "ai7_same_internal_bucket_shuffle",
    "ai8_uniform_hook_noop",
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
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


def log_success(path: Path, required_markers: list[str]) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return all(marker in text for marker in required_markers)


def method_root(seq: str, dataset: str, method: str) -> Path:
    return WORKSPACE / dataset / seq / method


def config_by_variant(seq: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in read_csv(CONFIG_ROWS):
        if str(row["seq"]).zfill(2) == seq:
            out[row["variant"]] = row
    return out


def routing_row_changed_count(row: dict[str, Any]) -> int:
    if row.get("row_type") == "anchor_source_attention_weight":
        return int(row.get("changed_key_count") or 0)
    if row.get("row_type") == "anchor_source_value_scaling":
        return int(row.get("changed_value_token_count") or 0)
    return 0


def summarize_variant(seq: str, row: dict[str, str], frozen: set[int], expected_hash: str) -> dict[str, Any]:
    num_frames = SEQ_NUM_FRAMES[seq]
    variant = row["variant"]
    method = row["method"]
    dataset = row["dataset"]
    action_path = ROOT / row["action_file"]
    rows = read_jsonl(action_path)
    schedule_rows = [
        item
        for item in rows
        if item.get("schema") == "acl2_v105_lingbot_stage4_action_row_v1"
        and not str(item.get("row_type", "") or "")
        and "sample_position" in item
        and not bool(item.get("anchor_scale_frame"))
    ]
    routing_rows = [
        item
        for item in rows
        if item.get("row_type") in {"anchor_source_attention_weight", "anchor_source_value_scaling"}
    ]
    stream_positions = [int(item["sample_position"]) for item in schedule_rows]
    stream_set = set(stream_positions)
    expected_stream = set(range(SCALE_FRAMES, num_frames))
    expected_downstream = sorted(frozen - set(range(SCALE_FRAMES)))
    final_keyframes = sorted(
        int(item["sample_position"])
        for item in schedule_rows
        if bool(item.get("final_is_keyframe"))
    )
    base_keyframes = sorted(
        int(item["sample_position"])
        for item in schedule_rows
        if bool(item.get("base_is_keyframe"))
    )
    modes = sorted({str(item.get("keyframe_schedule_mode", "")) for item in schedule_rows})
    hashes = sorted({str(item.get("frozen_keyframe_indices_hash", "")) for item in schedule_rows})
    counts = sorted({str(item.get("frozen_keyframe_count", "")) for item in schedule_rows})
    schedule_pass = (
        len(schedule_rows) == num_frames - SCALE_FRAMES
        and len(stream_positions) == len(stream_set)
        and stream_set == expected_stream
        and base_keyframes == expected_downstream
        and final_keyframes == expected_downstream
        and modes == ["global_frozen"]
        and hashes == [expected_hash]
        and counts == [str(len(frozen))]
    )
    expected_changed = int(float(row.get("changed_source_frame_count") or 0))
    routing_changed_total = sum(routing_row_changed_count(item) for item in routing_rows)
    routing_target_total = sum(
        int(item.get("target_key_count") or item.get("target_value_token_count") or 0)
        for item in routing_rows
    )
    sidecar_hashes = sorted({str(item.get("semantic_sidecar_hash", "")) for item in routing_rows})
    carrier_forms = sorted({str(item.get("carrier_form", "")) for item in routing_rows})
    token_roles = sorted({str(item.get("token_roles", "")) for item in routing_rows})
    source_context_roles = sorted({str(item.get("source_context_role", "")) for item in routing_rows})
    if expected_changed > 0:
        routing_pass = (
            bool(routing_rows)
            and routing_changed_total > 0
            and routing_target_total > 0
            and sidecar_hashes == [row["semantic_sidecar_hash"]]
        )
    elif row.get("action_mode") in {"anchor_source_attention_weight", "anchor_source_value_scaling"} and row.get("weight_mode") == "uniform_noop":
        routing_pass = bool(routing_rows) and routing_changed_total == 0 and routing_target_total > 0
    else:
        routing_pass = len(routing_rows) == 0

    root = method_root(seq, dataset, method)
    traj_path = root / "eval/traj.json"
    traj = json.loads(traj_path.read_text(encoding="utf-8")) if traj_path.exists() else {}
    metrics_present = all(finite_metric(traj.get(key)) for key in ("ate", "rpe_rot", "rpe_trans"))
    worker_matches = sorted((RUN_ROOT / "logs").glob(f"run_{variant}_seq{seq}_gpu*.log"))
    worker_log = worker_matches[0] if worker_matches else RUN_ROOT / "logs" / f"run_{variant}_seq{seq}.log"
    evaluate_log = RUN_ROOT / "logs" / f"evaluate_{variant}_seq{seq}.log"
    worker_ok = log_success(worker_log, ["Completed successfully", "Worker done: 1/1 scenes succeeded"])
    eval_ok = log_success(evaluate_log, ["Total successful: 1", "Total failed: 0"])
    return {
        "seq": seq,
        "variant": variant,
        "method": method,
        "policy": row.get("policy", ""),
        "role": row.get("role", ""),
        "action_mode": row.get("action_mode", ""),
        "weight_mode": row.get("weight_mode", ""),
        "token_roles_expected": row.get("token_roles", ""),
        "query_roles_expected": row.get("query_roles", ""),
        "source_context_roles_expected": row.get("source_context_roles", ""),
        "action_file": rel(action_path),
        "worker_log": rel(worker_log),
        "evaluate_log": rel(evaluate_log),
        "row_count": len(rows),
        "schedule_row_count": len(schedule_rows),
        "expected_schedule_row_count": num_frames - SCALE_FRAMES,
        "schedule_exact_pass": bool(schedule_pass),
        "routing_row_count": len(routing_rows),
        "routing_target_total": int(routing_target_total),
        "routing_changed_total": int(routing_changed_total),
        "routing_fidelity_pass": bool(routing_pass),
        "routing_sidecar_hashes": ";".join(sidecar_hashes),
        "routing_carrier_forms": ";".join(carrier_forms),
        "routing_token_roles": ";".join(token_roles),
        "routing_source_context_roles": ";".join(source_context_roles),
        "expected_sidecar_hash": row.get("semantic_sidecar_hash", ""),
        "weight_hash": row.get("weight_hash", ""),
        "changed_source_frame_count": expected_changed,
        "weight_min": row.get("weight_min", ""),
        "weight_max": row.get("weight_max", ""),
        "weight_mean": row.get("weight_mean", ""),
        "complete_exists": int((root / ".complete.json").exists()),
        "traj_exists": int((root / "traj.txt").exists()),
        "intrinsics_exists": int((root / "intrinsics.txt").exists()),
        "traj_json": rel(traj_path),
        "ate": traj.get("ate", ""),
        "rpe_rot": traj.get("rpe_rot", ""),
        "rpe_trans": traj.get("rpe_trans", ""),
        "metrics_present": bool(metrics_present),
        "worker_log_success_markers": bool(worker_ok),
        "evaluate_log_success_markers": bool(eval_ok),
        "runtime_pass": bool(worker_ok and eval_ok and metrics_present and schedule_pass and routing_pass),
    }


def output_path(seq: str, stem: str, suffix: str) -> Path:
    return RUN_ROOT / f"{stem}_seq{seq}.{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", default="00", choices=sorted(SEQ_NUM_FRAMES))
    parser.add_argument(
        "--variants",
        default=",".join(PRIMARY_VARIANTS),
        help="Comma-separated variant ids to summarize; default is primary smoke subset.",
    )
    args = parser.parse_args()
    seq = args.seq
    requested = [part.strip() for part in args.variants.split(",") if part.strip()]
    by_variant = config_by_variant(seq)
    missing = [variant for variant in requested if variant not in by_variant]
    if missing:
        raise RuntimeError(f"variants not in config_rows for seq{seq}: {missing}")
    frozen_indices = default_frozen_indices(SEQ_NUM_FRAMES[seq])
    frozen = set(frozen_indices)
    expected_hash = frozen_hash(frozen_indices)
    rows = [summarize_variant(seq, by_variant[variant], frozen, expected_hash) for variant in requested]
    baseline = next((item for item in rows if item["variant"] == "ai0_default_no_action"), None)
    if baseline:
        for item in rows:
            for metric in ("ate", "rpe_rot", "rpe_trans"):
                base_value = baseline.get(metric)
                value = item.get(metric)
                item[f"{metric}_delta_vs_ai0"] = (
                    float(value) - float(base_value)
                    if finite_metric(value) and finite_metric(base_value)
                    else ""
                )
            if finite_metric(item.get("ate")) and finite_metric(baseline.get("ate")):
                item["ate_improvement_pct_vs_ai0"] = (
                    (float(baseline["ate"]) - float(item["ate"])) / float(baseline["ate"]) * 100.0
                )
            else:
                item["ate_improvement_pct_vs_ai0"] = ""

    semantic_candidate = next((item for item in rows if item["variant"] == "ai2_semantic_anchor_role"), None)
    combined_candidate = next((item for item in rows if item["variant"] == "ai3_internal_semantic_anchor_role"), None)
    controls = [
        item
        for item in rows
        if item["variant"]
        in {
            "ai1_internal_anchor_utility",
            "ai5_reverse_semantic_anchor_role",
            "ai7_same_internal_bucket_shuffle",
            "ai8_uniform_hook_noop",
        }
        or item["variant"].startswith("ai6_random_seed")
    ]

    def beats_primary_controls(candidate: dict[str, Any] | None) -> bool:
        if not (
            candidate
            and finite_metric(candidate.get("ate"))
            and controls
            and all(finite_metric(item.get("ate")) for item in controls)
        ):
            return False
        return all(float(candidate["ate"]) < float(item["ate"]) for item in controls)

    summary = {
        "schema": "acl2_v119tf_stage2_lbai_form2_anchor_token_routing_summary_v1",
        "seq": seq,
        "num_frames": SEQ_NUM_FRAMES[seq],
        "scale_frames": SCALE_FRAMES,
        "expected_frozen_keyframe_count": len(frozen_indices),
        "expected_frozen_hash": expected_hash,
        "variants": requested,
        "variant_count": len(rows),
        "runtime_smoke_pass": bool(all(item["runtime_pass"] for item in rows)),
        "candidate_beats_primary_controls_by_ate": bool(beats_primary_controls(semantic_candidate)),
        "candidate_ate_improvement_pct_vs_ai0": (
            semantic_candidate.get("ate_improvement_pct_vs_ai0", "") if semantic_candidate else ""
        ),
        "combined_candidate_beats_primary_controls_by_ate": bool(beats_primary_controls(combined_candidate)),
        "combined_candidate_ate_improvement_pct_vs_ai0": (
            combined_candidate.get("ate_improvement_pct_vs_ai0", "") if combined_candidate else ""
        ),
        "rows_csv": rel(output_path(seq, "lbai_form2_anchor_token_routing_summary", "csv")),
        "truthfulness_boundary": (
            f"seq{seq} Form2 summarized variants only. This does not claim all AI6 seeds, AI1/AI3/AI7, "
            "holdout 01/05, or full v119 success."
        ),
    }
    write_csv(output_path(seq, "lbai_form2_anchor_token_routing_summary", "csv"), rows)
    output_path(seq, "lbai_form2_anchor_token_routing_summary", "json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
