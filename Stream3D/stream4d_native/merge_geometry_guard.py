from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GeometryRead:
    source_frame: str
    target_frame: str
    relation: str = "metric_merge"
    allow_metric_merge: bool = True
    weak_alignment: bool = False
    uses_gt_alignment: bool = False


def metric_merge_allowed(read: GeometryRead | dict[str, Any]) -> dict[str, Any]:
    item = read if isinstance(read, dict) else read.__dict__
    source = str(item.get("source_frame", ""))
    target = str(item.get("target_frame", ""))
    weak = bool(item.get("weak_alignment", False))
    allowed_flag = bool(item.get("allow_metric_merge", True))
    uses_gt = bool(item.get("uses_gt_alignment", False))
    local_cross_chunk = source == "chunk_local" and target in {"chunk_local", "method_canonical"}
    eval_in_method = source == "eval_aligned_gt" or target == "eval_aligned_gt" or uses_gt
    allowed = bool(allowed_flag and not weak and not local_cross_chunk and not eval_in_method)
    reasons: list[str] = []
    if weak:
        reasons.append("weak_alignment")
    if not allowed_flag:
        reasons.append("allow_metric_merge_false")
    if local_cross_chunk:
        reasons.append("cross_chunk_local_metric_read")
    if eval_in_method:
        reasons.append("eval_aligned_or_gt_geometry_in_method")
    return {
        "allowed": allowed,
        "guard_reason": "pass" if allowed else ";".join(reasons),
        "cross_chunk_local_metric_read": bool(local_cross_chunk),
        "cross_chunk_eval_aligned_read": bool(eval_in_method),
        "scale_sensitive_metric_read": bool(local_cross_chunk or weak or not allowed_flag),
    }


def summarize_guard_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    checked = [metric_merge_allowed(row) for row in rows]
    return {
        "guard_checked_count": int(len(checked)),
        "cross_chunk_local_metric_reads": int(sum(item["cross_chunk_local_metric_read"] for item in checked)),
        "cross_chunk_eval_reads": int(sum(item["cross_chunk_eval_aligned_read"] for item in checked)),
        "scale_sensitive_metric_reads": int(sum(item["scale_sensitive_metric_read"] for item in checked)),
        "blocked_read_count": int(sum(not item["allowed"] for item in checked)),
    }

