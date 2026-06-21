from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from .merge_geometry_guard import summarize_guard_rows


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def scale_alignment_guard_audit(
    *,
    ratio_rows: list[dict[str, Any]],
    window_rows: list[dict[str, Any]],
    block_outside_10pct: bool = True,
) -> dict[str, Any]:
    guarded_ratio_rows: list[dict[str, Any]] = []
    outside_10pct = 0
    blocked_outside = 0
    for row in ratio_rows:
        ratio = float(row.get("scale_next_over_prev") or "nan")
        ok10 = bool(str(row.get("scale_aligned_within_10pct", "")).lower() == "true")
        outside = bool(not ok10 or not (math.isfinite(ratio) and 0.90 <= ratio <= 1.10))
        outside_10pct += int(outside)
        blocked = bool(outside and block_outside_10pct)
        blocked_outside += int(blocked)
        guarded_ratio_rows.append(
            {
                **row,
                "outside_10pct_scale": outside,
                "alignment_pass": not outside,
                "allow_metric_merge": not blocked,
                "weak_alignment_reason": "outside_10pct_scale_pair_blocked" if blocked else "",
            }
        )
    guard_rows = [
        {
            "source_frame": "method_canonical" if str(row.get("allow_metric_merge", "")).lower() == "true" else "chunk_local",
            "target_frame": "method_canonical",
            "allow_metric_merge": str(row.get("allow_metric_merge", "")).lower() == "true",
            "weak_alignment": str(row.get("weak_alignment", "")).lower() == "true",
            "uses_gt_alignment": False,
        }
        for row in window_rows
    ]
    guard = summarize_guard_rows(guard_rows)
    gate = {
        "cross_chunk_local_metric_reads_eq_0": guard["cross_chunk_local_metric_reads"] == 0,
        "cross_chunk_eval_reads_eq_0": guard["cross_chunk_eval_reads"] == 0,
        "scale_sensitive_metric_reads_eq_0_or_blocked": (
            guard["scale_sensitive_metric_reads"] == 0 or blocked_outside == outside_10pct
        ),
        "outside_10pct_pairs_zero_or_blocked": outside_10pct == 0 or blocked_outside == outside_10pct,
    }
    gate["pass"] = bool(all(gate.values()))
    return {
        "phase": "v45_scale_alignment_guard",
        "is_method_result": False,
        "uses_gt_for_prediction": False,
        "uses_gt_depth_pose_for_scale_diagnostic": True,
        "scale_guard_rows": guarded_ratio_rows,
        "window_row_count": int(len(window_rows)),
        "alignment_pair_count": int(len(ratio_rows)),
        "outside_10pct_scale_pair_count": int(outside_10pct),
        "blocked_outside_10pct_pair_count": int(blocked_outside),
        **guard,
        "gate": gate,
    }

