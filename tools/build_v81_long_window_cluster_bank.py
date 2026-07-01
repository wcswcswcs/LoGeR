#!/usr/bin/env python3
"""Build ACL2 v81 five-chunk long-window cluster bank.

This uses v80 trajectory/semantic artifacts as inputs and adds the v81
long-window selected-write/support fields. Missing RADIO or selected-write
evidence is recorded explicitly instead of synthesized.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from build_v80_three_memory_good_bad_case_bank import (  # noqa: E402
    average_feature,
    radio_features,
    semantic_features,
)


DEFAULT_V80_PHASE1 = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final/phase1_three_memory_case_bank"
)
DEFAULT_V80_REPORT = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final"
)
DEFAULT_PHASE0_LOCK = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v81tf_longwindow_semantic_three_memory_control/"
    "report_final/phase0_v80_evidence_lock/v80_evidence_lock.json"
)
DEFAULT_OUT_DIR = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v81tf_longwindow_semantic_three_memory_control/"
    "report_final/phase1_long_window_cluster_bank"
)
DEFAULT_PREPROCESS_ROOT = Path("results/kitti_preprocess")

OUTPUT_FIELDS = [
    "seq",
    "window_id",
    "chunk_start",
    "chunk_end",
    "center_chunk",
    "case_type",
    "window5_joint_sim3_rmse",
    "window5_subchunk_scale_cv",
    "downstream_future_consistency",
    "selected_low_support_ratio",
    "selected_low_support_mass",
    "selected_runtime_mass",
    "continuous_low_support_cluster_len",
    "selected_minus_control_downstream_direction",
    "baseline_abs_error_mean",
    "baseline_abs_error_p90",
    "stable_mass",
    "harm_mass",
    "context_mass",
    "thing_moving_ratio",
    "thing_static_ratio",
    "lowtrust_stuff_ratio",
    "radio_lowtrust_mean",
    "radio_boundary_mean",
    "radio_temporal_stability_mean",
    "has_radio",
    "has_ttt_post_delta",
    "target_reason",
    "selected_chunk_evidence_count",
    "selected_missing_chunks",
    "semantic_available_chunks",
    "radio_available_chunks",
    "source_v80_candidate",
]

BAD_FORCED_WINDOWS = [
    ("02", 62, "mandatory_seq02_62_70_cluster"),
    ("02", 63, "mandatory_seq02_62_70_cluster"),
    ("02", 64, "mandatory_seq02_62_70_cluster"),
    ("02", 65, "mandatory_seq02_62_70_cluster"),
    ("02", 66, "mandatory_seq02_62_70_cluster"),
    ("00", 140, "mandatory_seq00_chunk142_centered"),
    ("01", 6, "mandatory_seq01_chunk08_centered"),
    ("05", 81, "mandatory_seq05_chunk83_centered"),
]

FALSE_POSITIVE_FORCED_WINDOWS = [
    ("02", 23, "mandatory_seq02_chunks26_27_false_positive"),
    ("02", 24, "mandatory_seq02_chunk26_false_positive_centered"),
    ("02", 25, "mandatory_seq02_chunk27_false_positive_centered"),
    ("02", 26, "mandatory_seq02_chunks26_27_false_positive"),
    ("02", 42, "mandatory_seq02_chunk44_false_positive_centered"),
    ("05", 6, "mandatory_seq05_chunk08_false_positive_centered"),
    ("05", 21, "mandatory_seq05_chunk23_false_positive_centered"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v80-phase1-dir", type=Path, default=DEFAULT_V80_PHASE1)
    parser.add_argument("--v80-report-root", type=Path, default=DEFAULT_V80_REPORT)
    parser.add_argument("--phase0-lock", type=Path, default=DEFAULT_PHASE0_LOCK)
    parser.add_argument("--preprocess-root", type=Path, default=DEFAULT_PREPROCESS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-per-group", type=int, default=12)
    return parser.parse_args()


def finite(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields or [])
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


def key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row["seq"]).zfill(2), int(row["chunk_start"]))


def load_candidates(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = read_csv(path)
    return {(str(row["seq"]).zfill(2), int(row["chunk_start"])): row for row in rows}


def seq_from_summary_path(path: Path, data: dict[str, Any]) -> str | None:
    source = str(data.get("source_stage_c_masklet") or "")
    match = re.search(r"results/kitti_preprocess/(\d\d)/", source)
    if match:
        return match.group(1)
    match = re.search(r"seq(\d\d)", str(path))
    return match.group(1) if match else None


def scan_selected_write_summaries(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for path in root.glob("**/chunk_*selected_write_support_map_summary.json"):
        data = read_json(path)
        seq = seq_from_summary_path(path, data)
        chunk = data.get("chunk")
        if seq is None or chunk is None:
            continue
        threshold = finite(data.get("support_threshold"))
        if threshold is not None and abs(threshold - 0.5) > 1e-9:
            continue
        item = {
            "seq": seq,
            "chunk": int(chunk),
            "path": str(path),
            "selected_low_support_ratio": finite(data.get("selected_low_support_given_selected_runtime")),
            "selected_low_support_mass": finite(data.get("selected_low_support_mass")),
            "selected_runtime_mass": finite(data.get("selected_runtime_mass")),
            "runtime_low_support_mass": finite(data.get("runtime_low_support_mass")),
            "score_mean": finite(data.get("score_mean")),
            "support_threshold": threshold,
            "has_ttt_post_delta": bool(data.get("source_post_delta_pt")),
        }
        old = out.get((seq, int(chunk)))
        if old is None:
            out[(seq, int(chunk))] = item
            continue
        old_mass = finite(old.get("selected_runtime_mass")) or 0.0
        new_mass = finite(item.get("selected_runtime_mass")) or 0.0
        if new_mass >= old_mass:
            out[(seq, int(chunk))] = item
    return out


def load_downstream_rows(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    paths = sorted(root.glob("phase10_selected_write_insight_matrix_*/selected_write_downstream_join_rows.csv"))
    if not paths:
        return {}
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in read_csv(paths[-1]):
        try:
            out[(str(row["seq"]).zfill(2), int(row["chunk"]))] = row
        except (KeyError, ValueError):
            continue
    return out


def longest_run(flags: list[bool]) -> int:
    best = cur = 0
    for flag in flags:
        if flag:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def aggregate_selected(
    seq: str,
    chunks: list[int],
    selected_by_chunk: dict[tuple[str, int], dict[str, Any]],
    downstream_by_chunk: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    selected = [selected_by_chunk.get((seq, chunk)) for chunk in chunks]
    selected = [item for item in selected if item]
    low_mass = sum(float(item.get("selected_low_support_mass") or 0.0) for item in selected)
    runtime_mass = sum(float(item.get("selected_runtime_mass") or 0.0) for item in selected)
    ratio = (low_mass / runtime_mass) if runtime_mass > 0 else None
    known_ratios = {
        int(item["chunk"]): finite(item.get("selected_low_support_ratio"))
        for item in selected
    }
    flags = [(known_ratios.get(chunk) or 0.0) >= 0.5 for chunk in chunks]

    downstream_vals: list[float] = []
    baseline_vals: list[float] = []
    downstream_labels: list[str] = []
    for chunk in chunks:
        row = downstream_by_chunk.get((seq, chunk))
        if not row:
            continue
        diff = finite(row.get("selected_minus_control_downstream_max"))
        if diff is not None:
            downstream_vals.append(diff)
            downstream_labels.append("harmful" if diff > 0 else "not_harmful")
        base = finite(row.get("baseline_abs_error_mean_m_phase2"))
        if base is not None:
            baseline_vals.append(base)
    if any(label == "harmful" for label in downstream_labels):
        direction = "harmful"
    elif downstream_labels and all(label == "not_harmful" for label in downstream_labels):
        direction = "not_harmful"
    else:
        direction = "unknown"

    return {
        "selected_low_support_ratio": ratio,
        "selected_low_support_mass": low_mass if selected else None,
        "selected_runtime_mass": runtime_mass if selected else None,
        "continuous_low_support_cluster_len": longest_run(flags),
        "selected_minus_control_downstream_direction": direction,
        "selected_chunk_evidence_count": len(selected),
        "selected_missing_chunks": ",".join(str(chunk) for chunk in chunks if (seq, chunk) not in selected_by_chunk),
        "has_ttt_post_delta": any(bool(item.get("has_ttt_post_delta")) for item in selected),
        "baseline_abs_error_mean": float(np.mean(baseline_vals)) if baseline_vals else None,
        "baseline_abs_error_p90": float(np.percentile(baseline_vals, 90)) if baseline_vals else None,
        "selected_chunk_sources": [item["path"] for item in selected],
        "selected_minus_control_downstream_values": downstream_vals,
    }


def aggregate_semantic(preprocess_root: Path, seq: str, chunks: list[int]) -> dict[str, Any]:
    sem = [semantic_features(preprocess_root, seq, chunk) for chunk in chunks]
    radio = [radio_features(preprocess_root, seq, chunk) for chunk in chunks]
    radio_present = [item for item in radio if item.get("radio_available")]
    return {
        "stable_mass": average_feature(sem, "stable_mass"),
        "harm_mass": average_feature(sem, "harm_mass"),
        "context_mass": average_feature(sem, "context_mass"),
        "thing_moving_ratio": average_feature(sem, "thing_moving_ratio"),
        "thing_static_ratio": average_feature(sem, "thing_static_ratio"),
        "lowtrust_stuff_ratio": average_feature(sem, "lowtrust_stuff_ratio"),
        "semantic_available_chunks": sum(bool(item.get("semantic_available")) for item in sem),
        "has_radio": bool(radio_present),
        "radio_available_chunks": len(radio_present),
        "radio_lowtrust_mean": average_feature(radio_present, "radio_lowtrust_mean") if radio_present else None,
        "radio_boundary_mean": average_feature(radio_present, "RADIO_boundary_ratio") if radio_present else None,
        "radio_temporal_stability_mean": average_feature(radio_present, "RADIO_temporal_stability") if radio_present else None,
    }


def build_row(
    row: dict[str, Any],
    case_type: str,
    reason: str,
    selected_by_chunk: dict[tuple[str, int], dict[str, Any]],
    downstream_by_chunk: dict[tuple[str, int], dict[str, Any]],
    preprocess_root: Path,
) -> dict[str, Any]:
    seq = str(row["seq"]).zfill(2)
    start = int(row["chunk_start"])
    end = int(row["chunk_end"])
    chunks = list(range(start, end + 1))
    selected = aggregate_selected(seq, chunks, selected_by_chunk, downstream_by_chunk)
    semantic = aggregate_semantic(preprocess_root, seq, chunks)
    out = {
        "seq": seq,
        "window_id": f"seq{seq}_chunks{start:03d}_{end:03d}",
        "chunk_start": start,
        "chunk_end": end,
        "center_chunk": start + ((end - start) // 2),
        "case_type": case_type,
        "window5_joint_sim3_rmse": finite(row.get("window5_joint_sim3_rmse")),
        "window5_subchunk_scale_cv": finite(row.get("window5_subchunk_scale_cv")),
        "downstream_future_consistency": finite(row.get("downstream_future_consistency")),
        "target_reason": reason,
        "source_v80_candidate": str(DEFAULT_V80_PHASE1 / "long_five_chunk_candidate_metrics.csv"),
        "J_long": finite(row.get("J_long")),
        "frame_start": row.get("frame_start"),
        "frame_end": row.get("frame_end"),
        "trajectory": row.get("trajectory"),
        "gt_path": row.get("gt_path"),
    }
    out.update(selected)
    out.update(semantic)
    return out


def sorted_by_metric(rows: Iterable[dict[str, Any]], reverse: bool) -> list[dict[str, Any]]:
    return sorted(
        [row for row in rows if finite(row.get("J_long")) is not None],
        key=lambda row: float(row["J_long"]),
        reverse=reverse,
    )


def has_selected_evidence(row: dict[str, Any], selected_by_chunk: dict[tuple[str, int], dict[str, Any]]) -> bool:
    seq = str(row["seq"]).zfill(2)
    return any((seq, chunk) in selected_by_chunk for chunk in range(int(row["chunk_start"]), int(row["chunk_end"]) + 1))


def choose_rows(
    candidates: dict[tuple[str, int], dict[str, Any]],
    selected_by_chunk: dict[tuple[str, int], dict[str, Any]],
    target_per_group: int,
) -> list[tuple[dict[str, Any], str, str]]:
    chosen: list[tuple[dict[str, Any], str, str]] = []
    used: set[tuple[str, int]] = set()

    def add(seq: str, start: int, case_type: str, reason: str) -> None:
        item = candidates.get((seq, start))
        if item is None or (seq, start) in used:
            return
        chosen.append((item, case_type, reason))
        used.add((seq, start))

    for seq, start, reason in BAD_FORCED_WINDOWS:
        add(seq, start, "bad", reason)
    for row in sorted_by_metric(candidates.values(), reverse=True):
        if sum(1 for _, case_type, _ in chosen if case_type == "bad") >= target_per_group:
            break
        if key(row) in used or not has_selected_evidence(row, selected_by_chunk):
            continue
        add(str(row["seq"]).zfill(2), int(row["chunk_start"]), "bad", "bad_long_by_J_long_rank_with_selected_write_evidence")

    for seq, start, reason in FALSE_POSITIVE_FORCED_WINDOWS:
        add(seq, start, "false_positive", reason)
    for row in sorted_by_metric(candidates.values(), reverse=False):
        if sum(1 for _, case_type, _ in chosen if case_type in {"good", "false_positive"}) >= target_per_group:
            break
        if key(row) in used or not has_selected_evidence(row, selected_by_chunk):
            continue
        add(str(row["seq"]).zfill(2), int(row["chunk_start"]), "good", "good_long_by_low_J_long_with_selected_write_evidence")

    return chosen


def summarize(rows: list[dict[str, Any]], phase0_lock: dict[str, Any]) -> dict[str, Any]:
    bad_rows = [row for row in rows if row["case_type"] == "bad"]
    good_rows = [row for row in rows if row["case_type"] in {"good", "false_positive"}]
    seq02_cluster = [
        row for row in rows
        if row["seq"] == "02" and int(row["chunk_start"]) <= 66 and int(row["chunk_end"]) >= 62
    ]
    missing_selected = [row["window_id"] for row in rows if int(row.get("selected_chunk_evidence_count") or 0) <= 0]
    missing_sem = [row["window_id"] for row in rows if int(row.get("semantic_available_chunks") or 0) <= 0]
    gate = (
        len(bad_rows) >= 12
        and len(good_rows) >= 12
        and len({row["seq"] for row in rows}) >= 3
        and bool(seq02_cluster)
        and not missing_selected
        and not missing_sem
    )
    return {
        "schema": "acl2_v81_phase1_long_window_cluster_bank_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase0_lock": str(DEFAULT_PHASE0_LOCK),
        "phase0_allowed_case_mining_seqs": phase0_lock.get("allowed_case_mining_seqs"),
        "kitti08_status": "blocked" if phase0_lock.get("kitti08_blockers") else "not_requested",
        "kitti08_blockers": phase0_lock.get("kitti08_blockers") or [],
        "row_count": len(rows),
        "bad_long_windows": len(bad_rows),
        "good_or_false_positive_windows": len(good_rows),
        "case_type_counts": dict(Counter(str(row["case_type"]) for row in rows)),
        "seqs_covered": sorted({row["seq"] for row in rows}),
        "bad_seqs": sorted({row["seq"] for row in bad_rows}),
        "good_false_positive_seqs": sorted({row["seq"] for row in good_rows}),
        "seq02_62_70_cluster_rows": [row["window_id"] for row in seq02_cluster],
        "selected_evidence_missing_rows": missing_selected,
        "semantic_missing_rows": missing_sem,
        "radio_available_rows": sum(bool(row.get("has_radio")) for row in rows),
        "gate_pass": gate,
        "gate_requirements": {
            "bad_long_windows_min": 12,
            "good_or_false_positive_windows_min": 12,
            "seq_coverage_min": 3,
            "seq02_62_70_cluster_included": bool(seq02_cluster),
            "each_row_has_selected_low_support_evidence": not missing_selected,
            "each_row_has_semantic_support_fields": not missing_sem,
        },
        "outputs": {
            "rows_csv": str(DEFAULT_OUT_DIR / "long_window_cluster_rows.csv"),
            "summary_json": str(DEFAULT_OUT_DIR / "long_window_cluster_summary.json"),
            "report_md": str(DEFAULT_OUT_DIR / "long_window_cluster_report.md"),
        },
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ACL2 v81 Phase1 Long-Window Cluster Bank",
        "",
        f"Gate pass: `{summary['gate_pass']}`",
        "",
        f"- rows: {summary['row_count']}",
        f"- bad_long_windows: {summary['bad_long_windows']}",
        f"- good_or_false_positive_windows: {summary['good_or_false_positive_windows']}",
        f"- seqs_covered: {','.join(summary['seqs_covered'])}",
        f"- seq02_62_70_cluster_rows: {','.join(summary['seq02_62_70_cluster_rows'])}",
        f"- radio_available_rows: {summary['radio_available_rows']}",
        f"- KITTI08: {summary['kitti08_status']} {summary['kitti08_blockers']}",
        "",
        "Missing selected evidence rows:",
        "",
        ", ".join(summary["selected_evidence_missing_rows"]) or "none",
        "",
        "Missing semantic rows:",
        "",
        ", ".join(summary["semantic_missing_rows"]) or "none",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    phase0_lock = read_json(args.phase0_lock)
    candidates = load_candidates(args.v80_phase1_dir / "long_five_chunk_candidate_metrics.csv")
    selected_by_chunk = scan_selected_write_summaries(args.v80_report_root)
    downstream_by_chunk = load_downstream_rows(args.v80_report_root)
    chosen = choose_rows(candidates, selected_by_chunk, int(args.target_per_group))
    rows = [
        build_row(row, case_type, reason, selected_by_chunk, downstream_by_chunk, args.preprocess_root)
        for row, case_type, reason in chosen
    ]
    rows.sort(key=lambda row: (0 if row["case_type"] == "bad" else 1, row["seq"], int(row["chunk_start"])))
    summary = summarize(rows, phase0_lock)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "long_window_cluster_rows.csv", rows, OUTPUT_FIELDS)
    write_json(args.out_dir / "long_window_cluster_summary.json", summary)
    write_report(args.out_dir / "long_window_cluster_report.md", summary)
    print(json.dumps({
        "out_dir": str(args.out_dir),
        "gate_pass": summary["gate_pass"],
        "row_count": summary["row_count"],
        "bad_long_windows": summary["bad_long_windows"],
        "good_or_false_positive_windows": summary["good_or_false_positive_windows"],
        "seqs_covered": summary["seqs_covered"],
        "seq02_62_70_cluster_rows": summary["seq02_62_70_cluster_rows"],
        "radio_available_rows": summary["radio_available_rows"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
