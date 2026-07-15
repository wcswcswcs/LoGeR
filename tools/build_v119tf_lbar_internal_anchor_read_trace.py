#!/usr/bin/env python3
"""Build ACL2 v119-TF LB-AR-FIX internal anchor-read trace rows."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_representation_repair"
TRACE_ROOT = RESULT_ROOT / "stage2_lbar_internal_anchor_read_trace"
RAW_TRACE = TRACE_ROOT / "raw_trace"
ROWS_CSV = TRACE_ROOT / "lbar_internal_anchor_read_rows.csv"
SUMMARY_JSON = TRACE_ROOT / "lbar_internal_anchor_read_summary.json"
SEQ_LENGTHS = {"00": 4541, "02": 4661}
SCALE_FRAMES = 8


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def norm(values: dict[int, float], default: float = 0.0) -> dict[int, float]:
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if abs(hi - lo) < 1e-12:
        return {key: default for key in values}
    return {key: (value - lo) / (hi - lo) for key, value in values.items()}


def bucket_labels(scores: dict[int, float]) -> tuple[dict[int, str], dict[int, str]]:
    ordered = sorted(scores, key=lambda frame: (scores[frame], frame))
    n = max(1, len(ordered))
    q4: dict[int, str] = {}
    q2: dict[int, str] = {}
    for rank, frame in enumerate(ordered):
        q4[frame] = f"q{min(4, int(rank * 4 / n) + 1)}_of_4"
        q2[frame] = f"q{min(2, int(rank * 2 / n) + 1)}_of_2"
    return q4, q2


def raw_trace_path(seq: str) -> Path:
    return RAW_TRACE / f"seq{seq}_ai0_global23_head0_top32_special.jsonl"


def build_seq(seq: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = raw_trace_path(seq)
    if not path.exists():
        raise FileNotFoundError(path)

    stats: dict[int, dict[str, Any]] = {
        frame: {
            "attention_weight_sum": 0.0,
            "topk_hit_count": 0,
            "entropy_sum": 0.0,
            "topk_mass_sum": 0.0,
            "query_roles": set(),
            "current_frames": set(),
        }
        for frame in range(SCALE_FRAMES)
    }
    total_rows = 0
    raw_gca_context_topk_rows = 0
    used_rows = 0
    raw_max_current_frame = -1
    anchor_read_max_current_frame = -1
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total_rows += 1
            row = json.loads(line)
            if row.get("row_type") != "gca_context_topk":
                continue
            raw_gca_context_topk_rows += 1
            try:
                raw_current_frame = int(row.get("last_read_time", -1))
            except (TypeError, ValueError):
                raw_current_frame = -1
            if raw_current_frame >= 0:
                raw_max_current_frame = max(raw_max_current_frame, raw_current_frame)
            if row.get("key_context_role") != "scale_reference_context":
                continue
            try:
                source_frame = int(row.get("source_frame_id", row.get("key_frame_offset", -1)))
            except (TypeError, ValueError):
                continue
            if source_frame not in stats:
                continue
            query_role = str(row.get("query_token_role", ""))
            if query_role not in {"camera_special", "register_special", "scale_special"}:
                continue
            try:
                attention_weight = float(row.get("attention_weight", 0.0))
            except (TypeError, ValueError):
                attention_weight = 0.0
            try:
                entropy = float(row.get("attention_entropy", 0.0))
            except (TypeError, ValueError):
                entropy = 0.0
            try:
                topk_mass = float(row.get("attention_topk_mass", 0.0))
            except (TypeError, ValueError):
                topk_mass = 0.0
            current_frame = raw_current_frame
            target = stats[source_frame]
            target["attention_weight_sum"] += attention_weight
            target["topk_hit_count"] += 1
            target["entropy_sum"] += entropy
            target["topk_mass_sum"] += topk_mass
            target["query_roles"].add(query_role)
            if current_frame >= 0:
                target["current_frames"].add(current_frame)
                anchor_read_max_current_frame = max(anchor_read_max_current_frame, current_frame)
            used_rows += 1

    mass = {frame: float(stats[frame]["attention_weight_sum"]) for frame in stats}
    hits = {frame: float(stats[frame]["topk_hit_count"]) for frame in stats}
    entropy_mean = {
        frame: (
            float(stats[frame]["entropy_sum"]) / max(1, int(stats[frame]["topk_hit_count"]))
        )
        for frame in stats
    }
    mass_norm = norm(mass, default=0.5)
    hits_norm = norm(hits, default=0.5)
    entropy_norm = norm(entropy_mean, default=0.5)

    score: dict[int, float] = {}
    for frame in stats:
        entropy_focus = 1.0 - entropy_norm[frame]
        score[frame] = max(
            0.0,
            min(1.0, 0.55 * mass_norm[frame] + 0.25 * hits_norm[frame] + 0.20 * entropy_focus),
        )
    q4, q2 = bucket_labels(score)

    rows: list[dict[str, Any]] = []
    for frame in range(SCALE_FRAMES):
        item = stats[frame]
        hit_count = int(item["topk_hit_count"])
        current_frame_count = len(item["current_frames"])
        rows.append(
            {
                "schema": "acl2_v119tf_lbar_internal_anchor_read_row_v1",
                "seq": seq,
                "source_frame": frame,
                "raw_trace": rel(path),
                "raw_trace_hash": sha256_file(path),
                "internal_read_score_version": "v119_lbar_noaction_qk_topk_entropy_global23_head0_special_v1",
                "attention_weight_sum": float(item["attention_weight_sum"]),
                "topk_hit_count": hit_count,
                "current_frame_count": current_frame_count,
                "mean_attention_weight_per_hit": float(item["attention_weight_sum"]) / max(1, hit_count),
                "mean_attention_weight_per_current_frame": float(item["attention_weight_sum"]) / max(1, current_frame_count),
                "mean_attention_entropy": float(item["entropy_sum"]) / max(1, hit_count),
                "mean_topk_mass": float(item["topk_mass_sum"]) / max(1, hit_count),
                "attention_mass_norm": mass_norm[frame],
                "topk_hit_norm": hits_norm[frame],
                "entropy_focus_norm": 1.0 - entropy_norm[frame],
                "internal_anchor_read_score": score[frame],
                "internal_anchor_read_bucket_q4": q4[frame],
                "internal_anchor_read_bucket_q2": q2[frame],
                "query_roles_observed": ",".join(sorted(item["query_roles"])),
                "trace_global_idxs": "23",
                "trace_head_idxs": "0",
                "trace_topk": 32,
                "trace_query_roles": "camera_special,register_special,scale_special",
                "truthfulness_boundary": (
                    "current-code no-action SDPA QK top-k trace only; no GT, no external depth, no SLAM, "
                    "no post-hoc ATE feedback"
                ),
            }
        )
    summary = {
        "seq": seq,
        "raw_trace": rel(path),
        "raw_trace_hash": sha256_file(path),
        "raw_trace_total_rows": total_rows,
        "raw_gca_context_topk_rows": raw_gca_context_topk_rows,
        "used_anchor_read_rows": used_rows,
        "raw_max_current_frame": raw_max_current_frame,
        "anchor_read_max_current_frame": anchor_read_max_current_frame,
        "expected_last_frame": SEQ_LENGTHS[seq] - 1,
        "full_sequence_trace_coverage": raw_max_current_frame >= SEQ_LENGTHS[seq] - 1,
        "anchor_scale_reference_topk_observed_to_sequence_end": (
            anchor_read_max_current_frame >= SEQ_LENGTHS[seq] - 1
        ),
    }
    return rows, summary


def main() -> None:
    all_rows: list[dict[str, Any]] = []
    seq_summaries = []
    for seq in sorted(SEQ_LENGTHS):
        rows, summary = build_seq(seq)
        all_rows.extend(rows)
        seq_summaries.append(summary)

    write_csv(ROWS_CSV, all_rows)
    payload = {
        "schema": "acl2_v119tf_lbar_internal_anchor_read_summary_v1",
        "rows_csv": rel(ROWS_CSV),
        "row_count": len(all_rows),
        "sequences": sorted(SEQ_LENGTHS),
        "internal_read_score_version": "v119_lbar_noaction_qk_topk_entropy_global23_head0_special_v1",
        "seq_summaries": seq_summaries,
        "truthfulness_boundary": (
            "Internal read score is derived from no-action SDPA QK top-k trace with fixed global_idx/head/query roles. "
            "It is not semantic and does not use GT/ATE."
        ),
    }
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
