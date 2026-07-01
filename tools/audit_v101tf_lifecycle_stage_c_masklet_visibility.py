#!/usr/bin/env python3
"""Audit current-chunk Stage-C masklet visibility for lifecycle seed ids.

This is a diagnostic-only Track U repair attempt.  It checks whether lifecycle
``source_stage_c_seed_global_track_idx_mode`` ids are visible in the current
Stage-C masklet cache for each boundary.  It does not rewrite Track U rows or
authorize runtime action.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
FINAL = ROOT / "final_decision"
PREPROCESS_ROOT = Path("results/kitti_preprocess")

LIFECYCLE_ROWS = FINAL / "anchor_seed_lifecycle_expanded_rows.csv"
ROWS_OUT = FINAL / "anchor_seed_lifecycle_stage_c_masklet_visibility_rows.csv"
SUMMARY_OUT = FINAL / "anchor_seed_lifecycle_stage_c_masklet_visibility_summary.json"
REPORT_OUT = FINAL / "anchor_seed_lifecycle_stage_c_masklet_visibility_report.md"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    if not keys:
        keys = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in keys})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def as_float(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def mean(values: list[Any]) -> float:
    finite = [as_float(value) for value in values if math.isfinite(as_float(value))]
    return sum(finite) / len(finite) if finite else math.nan


def parse_case(case_id: str) -> tuple[str, int | None, int | None]:
    parts = case_id.split("_")
    if len(parts) != 3:
        return parts[0] if parts else "", None, None
    try:
        return parts[0], int(parts[1]), int(parts[2])
    except ValueError:
        return parts[0], None, None


def seed_text(value: Any) -> str:
    number = as_float(value)
    return str(int(number)) if math.isfinite(number) else ""


def limited_join(values: list[Any], *, limit: int = 16) -> str:
    text_values = sorted({str(value) for value in values if value not in (None, "")})
    if len(text_values) > limit:
        return ";".join(text_values[:limit]) + f";...(+{len(text_values) - limit})"
    return ";".join(text_values)


def read_cache_index(seq: str) -> dict[int, dict[str, Any]]:
    path = PREPROCESS_ROOT / seq / "stage_c_cache_semantic_chunks/cache_index.jsonl"
    out: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                out[int(row.get("chunk_idx"))] = row
            except Exception:
                continue
    return out


def masklet_path(seq: str, chunk_idx: int, cache_by_seq: dict[str, dict[int, dict[str, Any]]]) -> Path:
    if seq not in cache_by_seq:
        cache_by_seq[seq] = read_cache_index(seq)
    chunk_name = cache_by_seq.get(seq, {}).get(chunk_idx, {}).get("chunk", f"chunk_{chunk_idx:03d}")
    return PREPROCESS_ROOT / seq / "stage_c_cache_semantic_chunks" / str(chunk_name) / "masklet.pt"


def load_masklet_seed_summary(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return {}
    seeds = [str(item) for item in payload.get("seed_global_track_idx", [])]
    labels = payload.get("L_sem", [])
    source_types = payload.get("source_type", [])
    g_sem = payload.get("G_sem")
    v_mask = payload.get("V_mask")
    b_mask = payload.get("B_mask")
    a_ratio = payload.get("A_ratio")
    q_mask = payload.get("Q_mask")
    out: dict[str, dict[str, Any]] = {}
    for idx, seed in enumerate(seeds):
        visible_frames = 0
        area_mean = math.nan
        area_max = math.nan
        quality_mean = math.nan
        bbox_center_span_px = math.nan
        bbox_area_px_mean = math.nan
        bbox_area_px_cv = math.nan
        area_ratio_std = math.nan
        if torch.is_tensor(v_mask) and idx < int(v_mask.shape[0]):
            visible_mask = v_mask[idx].detach().bool()
            visible_frames = int(visible_mask.sum().item())
        else:
            visible_mask = None
        if torch.is_tensor(a_ratio) and idx < int(a_ratio.shape[0]):
            vals = a_ratio[idx].detach().float()
            finite = vals[torch.isfinite(vals)]
            if int(finite.numel()) > 0:
                area_mean = float(finite.mean().item())
                area_max = float(finite.max().item())
                area_ratio_std = float(finite.std(unbiased=False).item())
        if torch.is_tensor(b_mask) and idx < int(b_mask.shape[0]):
            boxes = b_mask[idx].detach().float()
            if visible_mask is not None and int(visible_mask.numel()) == int(boxes.shape[0]):
                boxes = boxes[visible_mask]
            if int(boxes.numel()) > 0 and boxes.ndim == 2 and int(boxes.shape[-1]) >= 4:
                x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                width = (x2 - x1).clamp_min(0.0)
                height = (y2 - y1).clamp_min(0.0)
                area_px = width * height
                center_x = (x1 + x2) * 0.5
                center_y = (y1 + y2) * 0.5
                if int(center_x.numel()) > 0:
                    bbox_center_span_px = float(
                        torch.sqrt(
                            (center_x.max() - center_x.min()).pow(2)
                            + (center_y.max() - center_y.min()).pow(2)
                        ).item()
                    )
                if int(area_px.numel()) > 0:
                    bbox_area_px_mean = float(area_px.mean().item())
                    denom = float(area_px.mean().abs().clamp_min(1.0e-6).item())
                    bbox_area_px_cv = float(area_px.std(unbiased=False).item() / denom)
        if torch.is_tensor(q_mask) and idx < int(q_mask.shape[0]):
            vals = q_mask[idx].detach().float()
            finite = vals[torch.isfinite(vals)]
            if int(finite.numel()) > 0:
                quality_mean = float(finite.mean().item())
        out[seed] = {
            "masklet_path": str(path),
            "seed_visible": visible_frames > 0,
            "visible_frame_count": visible_frames,
            "visible_frame_frac": visible_frames / max(int(payload.get("num_frames", 0) or 0), 1),
            "area_ratio_mean": area_mean,
            "area_ratio_max": area_max,
            "area_ratio_std": area_ratio_std,
            "bbox_center_span_px": bbox_center_span_px,
            "bbox_area_px_mean": bbox_area_px_mean,
            "bbox_area_px_cv": bbox_area_px_cv,
            "quality_mean": quality_mean,
            "semantic_label_name": labels[idx] if idx < len(labels) else "",
            "source_type": source_types[idx] if idx < len(source_types) else "",
            "G_sem": int(g_sem[idx].item()) if torch.is_tensor(g_sem) and idx < int(g_sem.shape[0]) else "",
        }
    return out


def lifecycle_seed_groups() -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_rows(LIFECYCLE_ROWS):
        case_id = row.get("case_id", "")
        seed = seed_text(row.get("source_stage_c_seed_global_track_idx_mode"))
        if case_id and seed:
            grouped[(case_id, seed)].append(row)
    return grouped


def main() -> None:
    grouped = lifecycle_seed_groups()
    cache_by_seq: dict[str, dict[int, dict[str, Any]]] = {}
    masklet_cache: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    load_error_rows: list[dict[str, Any]] = []

    for (case_id, seed), parts in sorted(grouped.items()):
        seq, left, right = parse_case(case_id)
        source_chunks = [int(as_float(row.get("source_chunk_idx"))) for row in parts if math.isfinite(as_float(row.get("source_chunk_idx")))]
        source_chunk = Counter(source_chunks).most_common(1)[0][0] if source_chunks else left
        current_summary: dict[str, Any] = {}
        source_summary: dict[str, Any] = {}
        for label, chunk_idx in (("current", right), ("source", source_chunk)):
            if chunk_idx is None:
                continue
            cache_key = (seq, int(chunk_idx))
            if cache_key not in masklet_cache:
                path = masklet_path(seq, int(chunk_idx), cache_by_seq)
                try:
                    masklet_cache[cache_key] = load_masklet_seed_summary(path)
                except Exception as exc:  # noqa: BLE001
                    masklet_cache[cache_key] = {}
                    load_error_rows.append(
                        {
                            "seq": seq,
                            "chunk_idx": int(chunk_idx),
                            "masklet_path": str(path),
                            "error": f"{type(exc).__name__}:{exc}",
                        }
                    )
            if label == "current":
                current_summary = masklet_cache[cache_key].get(seed, {})
            else:
                source_summary = masklet_cache[cache_key].get(seed, {})

        rows.append(
            {
                "case_id": case_id,
                "seq": seq,
                "boundary_left_chunk": left if left is not None else "",
                "boundary_right_chunk": right if right is not None else "",
                "source_chunk_idx_mode": source_chunk if source_chunk is not None else "",
                "stage_c_seed_global_track_idx": seed,
                "lifecycle_row_count": len(parts),
                "lifecycle_anchor_id_count": len({row.get("anchor_id", "") for row in parts if row.get("anchor_id", "")}),
                "lifecycle_anchor_ids_sample": limited_join([row.get("anchor_id", "") for row in parts], limit=12),
                "source_label_modes": limited_join([row.get("source_label_mode", "") for row in parts]),
                "current_chunk_seed_visible": bool(current_summary.get("seed_visible", False)),
                "current_chunk_visible_frame_count": current_summary.get("visible_frame_count", ""),
                "current_chunk_visible_frame_frac": current_summary.get("visible_frame_frac", ""),
                "current_chunk_area_ratio_mean": current_summary.get("area_ratio_mean", ""),
                "current_chunk_area_ratio_max": current_summary.get("area_ratio_max", ""),
                "current_chunk_area_ratio_std": current_summary.get("area_ratio_std", ""),
                "current_chunk_bbox_center_span_px": current_summary.get("bbox_center_span_px", ""),
                "current_chunk_bbox_area_px_mean": current_summary.get("bbox_area_px_mean", ""),
                "current_chunk_bbox_area_px_cv": current_summary.get("bbox_area_px_cv", ""),
                "current_chunk_quality_mean": current_summary.get("quality_mean", ""),
                "current_chunk_semantic_label_name": current_summary.get("semantic_label_name", ""),
                "current_chunk_source_type": current_summary.get("source_type", ""),
                "current_chunk_G_sem": current_summary.get("G_sem", ""),
                "source_chunk_seed_visible": bool(source_summary.get("seed_visible", False)),
                "source_chunk_visible_frame_count": source_summary.get("visible_frame_count", ""),
                "source_chunk_visible_frame_frac": source_summary.get("visible_frame_frac", ""),
                "source_chunk_area_ratio_max": source_summary.get("area_ratio_max", ""),
                "source_chunk_bbox_center_span_px": source_summary.get("bbox_center_span_px", ""),
                "source_chunk_bbox_area_px_cv": source_summary.get("bbox_area_px_cv", ""),
                "source_chunk_semantic_label_name": source_summary.get("semantic_label_name", ""),
                "source_chunk_source_type": source_summary.get("source_type", ""),
                "component_current_visibility_materialized": bool(current_summary),
                "claim_level": "diagnostic_stage_c_masklet_current_visibility_no_action",
            }
        )

    write_rows(ROWS_OUT, rows)
    write_rows(FINAL / "anchor_seed_lifecycle_stage_c_masklet_visibility_load_errors.csv", load_error_rows)

    current_visible = [row for row in rows if row.get("current_chunk_seed_visible") is True]
    source_visible = [row for row in rows if row.get("source_chunk_seed_visible") is True]
    current_source_visible = [
        row
        for row in rows
        if row.get("current_chunk_seed_visible") is True and row.get("source_chunk_seed_visible") is True
    ]
    summary = {
        "schema": "acl2_v101_lifecycle_stage_c_masklet_visibility_v1",
        "diagnostic_only": True,
        "runtime_action_allowed": False,
        "method_goal_achieved": False,
        "lifecycle_unique_case_seed_count": len(rows),
        "masklet_chunk_load_error_count": len(load_error_rows),
        "current_chunk_visible_unique_case_seed_count": len(current_visible),
        "source_chunk_visible_unique_case_seed_count": len(source_visible),
        "current_and_source_visible_unique_case_seed_count": len(current_source_visible),
        "current_chunk_visibility_coverage": len(current_visible) / len(rows) if rows else 0.0,
        "source_chunk_visibility_coverage": len(source_visible) / len(rows) if rows else 0.0,
        "current_and_source_visibility_coverage": len(current_source_visible) / len(rows) if rows else 0.0,
        "current_visible_frame_count_mean": mean([row.get("current_chunk_visible_frame_count") for row in current_visible]),
        "current_visible_frame_frac_mean": mean([row.get("current_chunk_visible_frame_frac") for row in current_visible]),
        "current_area_ratio_max_mean": mean([row.get("current_chunk_area_ratio_max") for row in current_visible]),
        "current_area_ratio_std_mean": mean([row.get("current_chunk_area_ratio_std") for row in current_visible]),
        "current_bbox_center_span_px_mean": mean([row.get("current_chunk_bbox_center_span_px") for row in current_visible]),
        "current_bbox_area_px_cv_mean": mean([row.get("current_chunk_bbox_area_px_cv") for row in current_visible]),
        "current_semantic_label_name_counts": dict(Counter(row.get("current_chunk_semantic_label_name", "") for row in current_visible)),
        "current_source_type_counts": dict(Counter(row.get("current_chunk_source_type", "") for row in current_visible)),
        "component_current_visibility_materialized": len(current_visible) > 0,
        "trackU_component_current_visibility_repair_candidate": len(current_visible) > 0,
        "masklet_2d_observability_proxy_materialized": len(current_visible) > 0,
        "scale_observability_proxy_only": True,
        "true_scale_observability_pass": False,
        "true_current_support_strict_pass": False,
        "strict_current_support_blocker": (
            "Stage-C masklet current visibility is component-level diagnostic evidence, but it is not yet integrated "
            "with anchor-level geometry/scale observability controls, JL4 instance identity, Q2 true-stage admission, or M4."
        ),
        "rows_path": str(ROWS_OUT),
    }
    write_json(SUMMARY_OUT, summary)
    report_lines = [
        "# ACL2 v101 Lifecycle Stage-C Masklet Visibility Audit",
        "",
        "This report is diagnostic-only. It checks whether lifecycle Stage-C seed ids are visible in current/source Stage-C masklet chunks.",
        "",
        "## Key Counts",
        "",
        f"- lifecycle_unique_case_seed_count: {summary['lifecycle_unique_case_seed_count']}",
        f"- masklet_chunk_load_error_count: {summary['masklet_chunk_load_error_count']}",
        f"- current_chunk_visible_unique_case_seed_count: {summary['current_chunk_visible_unique_case_seed_count']}",
        f"- source_chunk_visible_unique_case_seed_count: {summary['source_chunk_visible_unique_case_seed_count']}",
        f"- current_and_source_visible_unique_case_seed_count: {summary['current_and_source_visible_unique_case_seed_count']}",
        f"- current_chunk_visibility_coverage: {summary['current_chunk_visibility_coverage']}",
        f"- source_chunk_visibility_coverage: {summary['source_chunk_visibility_coverage']}",
        f"- current_and_source_visibility_coverage: {summary['current_and_source_visibility_coverage']}",
        f"- current_bbox_center_span_px_mean: {summary['current_bbox_center_span_px_mean']}",
        f"- current_bbox_area_px_cv_mean: {summary['current_bbox_area_px_cv_mean']}",
        "",
        "## Decision",
        "",
        summary["strict_current_support_blocker"],
        "",
        "## Artifacts",
        "",
        f"- `{ROWS_OUT}`",
        f"- `{SUMMARY_OUT}`",
    ]
    REPORT_OUT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "schema": summary["schema"],
                "lifecycle_unique_case_seed_count": summary["lifecycle_unique_case_seed_count"],
                "current_chunk_visibility_coverage": summary["current_chunk_visibility_coverage"],
                "current_chunk_visible_unique_case_seed_count": summary[
                    "current_chunk_visible_unique_case_seed_count"
                ],
                "true_current_support_strict_pass": summary["true_current_support_strict_pass"],
                "true_scale_observability_pass": summary["true_scale_observability_pass"],
                "runtime_action_allowed": summary["runtime_action_allowed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
