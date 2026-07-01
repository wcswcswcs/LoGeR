#!/usr/bin/env python3
"""Join lifecycle anchors to Track V geometry observability sidecars.

This is an audit-only Track V follow-up.  It checks whether the existing
per-anchor geometry observability repair can be used for the lifecycle
anchor/seed universe.  It does not authorize Q2, M4, runtime, or full
validation.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
FINAL = ROOT / "final_decision"
TRACK_V = ROOT / "trackV_anchor_scale_observability"

LIFECYCLE_ROWS = FINAL / "anchor_seed_lifecycle_expanded_rows.csv"
GEOMETRY_ROWS = TRACK_V / "per_anchor_geometry_observability_rows.csv"
GEOMETRY_SUMMARY = TRACK_V / "per_anchor_geometry_observability_summary.json"
POINTMAP_DEPTH_ROWS = TRACK_V / "pointmap_depth_support_rows.csv"
SCALE_MODE_ROWS = TRACK_V / "scale_mode_rows.csv"
TARGET_ROWS = ROOT / "trackT_drift_target_relabel/target_universe_v101.csv"
RAW_GEOMETRY_EDGE_ROWS = Path(
    "results/acl2_v100tf_same_space_semantic_anchor_latent_state_memory_control/"
    "trackL2_anchor_scale_observability/geometry_edge_rows.csv"
)

JOIN_ROWS_OUT = FINAL / "anchor_seed_lifecycle_geometry_observability_join_rows.csv"
CASE_ROWS_OUT = FINAL / "anchor_seed_lifecycle_geometry_observability_case_rows.csv"
SUMMARY_OUT = FINAL / "anchor_seed_lifecycle_geometry_observability_summary.json"
REPORT_OUT = FINAL / "anchor_seed_lifecycle_geometry_observability_report.md"

POS_TAX = "HANDOFF_SCALE_GAUGE_TARGET"
SAFE_TAX = "SAFE_GOOD"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"read_error": f"{type(exc).__name__}:{exc}"}
    return payload if isinstance(payload, dict) else {"value": payload}


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


def f(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def b(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def norm_id(value: Any) -> str:
    number = f(value)
    if math.isfinite(number):
        return str(int(number))
    text = str(value).strip()
    if not text:
        return ""
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def mean(values: list[Any]) -> float:
    finite = [f(value) for value in values if math.isfinite(f(value))]
    return sum(finite) / len(finite) if finite else math.nan


def frac(numer: int, denom: int) -> float:
    return float(numer) / float(denom) if denom else math.nan


def row_count(path: Path) -> int:
    rows = read_rows(path)
    return len(rows)


def key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("case_id", "")), norm_id(row.get("anchor_id", ""))


def target_by_case() -> dict[str, dict[str, str]]:
    return {row.get("case_id", ""): row for row in read_rows(TARGET_ROWS)}


def add_stat(stats: dict[str, Any], name: str, value: Any) -> None:
    number = f(value)
    if not math.isfinite(number):
        return
    stats[f"{name}_sum"] = stats.get(f"{name}_sum", 0.0) + number
    stats[f"{name}_count"] = stats.get(f"{name}_count", 0) + 1


def stat_mean(stats: dict[str, Any], name: str) -> float:
    count = int(stats.get(f"{name}_count", 0) or 0)
    return float(stats.get(f"{name}_sum", 0.0)) / count if count else math.nan


def load_raw_geometry_edge_stats() -> tuple[dict[tuple[str, str], dict[str, Any]], int]:
    stats_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    raw_row_count = 0
    if not RAW_GEOMETRY_EDGE_ROWS.is_file():
        return stats_by_key, raw_row_count
    with RAW_GEOMETRY_EDGE_ROWS.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_row_count += 1
            k = key(row)
            if not k[0] or not k[1]:
                continue
            stats = stats_by_key.setdefault(
                k,
                {
                    "raw_geometry_edge_count": 0,
                    "raw_frame_min": math.inf,
                    "raw_frame_max": -math.inf,
                },
            )
            stats["raw_geometry_edge_count"] += 1
            for name in [
                "query_depth",
                "cache_depth",
                "query_conf",
                "cache_conf",
                "camera_translation_baseline",
                "world_pair_distance",
                "abs_log_depth_ratio",
                "abs_depth_diff",
            ]:
                add_stat(stats, name, row.get(name))
            for name in ["query_frame", "cache_frame"]:
                frame = f(row.get(name))
                if math.isfinite(frame):
                    stats["raw_frame_min"] = min(float(stats["raw_frame_min"]), frame)
                    stats["raw_frame_max"] = max(float(stats["raw_frame_max"]), frame)
    for stats in stats_by_key.values():
        frame_min = float(stats.get("raw_frame_min", math.inf))
        frame_max = float(stats.get("raw_frame_max", -math.inf))
        stats["raw_anchor_frame_span"] = frame_max - frame_min if math.isfinite(frame_min) and math.isfinite(frame_max) else math.nan
        for name in [
            "query_depth",
            "cache_depth",
            "query_conf",
            "cache_conf",
            "camera_translation_baseline",
            "world_pair_distance",
            "abs_log_depth_ratio",
            "abs_depth_diff",
        ]:
            stats[f"raw_{name}_mean"] = stat_mean(stats, name)
    return stats_by_key, raw_row_count


def main() -> None:
    lifecycle_rows = read_rows(LIFECYCLE_ROWS)
    geometry_rows = read_rows(GEOMETRY_ROWS)
    geometry_summary = read_json(GEOMETRY_SUMMARY)
    target = target_by_case()
    raw_edge_by_key, raw_geometry_edge_row_count = load_raw_geometry_edge_stats()

    geometry_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in geometry_rows:
        k = key(row)
        if k[0] and k[1] and k not in geometry_by_key:
            geometry_by_key[k] = row

    lifecycle_by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    lifecycle_row_with_anchor_count = 0
    for row in lifecycle_rows:
        k = key(row)
        if k[0] and k[1]:
            lifecycle_by_key[k].append(row)
            lifecycle_row_with_anchor_count += 1

    join_rows: list[dict[str, Any]] = []
    case_parts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    joined_keys: set[tuple[str, str]] = set()
    raw_edge_joined_keys: set[tuple[str, str]] = set()
    true_geometry_joined_keys: set[tuple[str, str]] = set()
    temporal_proxy_joined_keys: set[tuple[str, str]] = set()
    handoff_keys: set[tuple[str, str]] = set()
    safe_keys: set[tuple[str, str]] = set()
    handoff_joined_keys: set[tuple[str, str]] = set()
    safe_joined_keys: set[tuple[str, str]] = set()

    for k, rows in sorted(lifecycle_by_key.items()):
        case_id, anchor_id = k
        geom = geometry_by_key.get(k)
        raw_edge = raw_edge_by_key.get(k)
        tax = (geom or {}).get("target_taxonomy") or target.get(case_id, {}).get("target_taxonomy", "")
        joined = geom is not None
        raw_edge_joined = raw_edge is not None
        if tax == POS_TAX:
            handoff_keys.add(k)
        if tax == SAFE_TAX:
            safe_keys.add(k)
        if raw_edge_joined:
            raw_edge_joined_keys.add(k)
        if joined:
            joined_keys.add(k)
            if tax == POS_TAX:
                handoff_joined_keys.add(k)
            if tax == SAFE_TAX:
                safe_joined_keys.add(k)
            if b(geom.get("true_geometry_source_available")):
                true_geometry_joined_keys.add(k)
            if b(geom.get("temporal_proxy_only")):
                temporal_proxy_joined_keys.add(k)

        source_stage_c_seeds = sorted(
            {
                norm_id(row.get("source_stage_c_seed_global_track_idx_mode"))
                for row in rows
                if norm_id(row.get("source_stage_c_seed_global_track_idx_mode"))
            }
        )
        out_row = {
            "case_id": case_id,
            "anchor_id": anchor_id,
            "target_taxonomy": tax,
            "lifecycle_row_count": len(rows),
            "source_stage_c_seed_count": len(source_stage_c_seeds),
            "source_stage_c_seed_sample": ";".join(source_stage_c_seeds[:8]),
            "geometry_joined": joined,
            "raw_geometry_edge_joined": raw_edge_joined,
            "geometry_source_level": (geom or {}).get("geometry_source_level", ""),
            "true_geometry_source_available": (geom or {}).get("true_geometry_source_available", ""),
            "temporal_proxy_only": (geom or {}).get("temporal_proxy_only", ""),
            "geometry_edge_count": (geom or {}).get("geometry_edge_count", ""),
            "raw_geometry_edge_count": (raw_edge or {}).get("raw_geometry_edge_count", ""),
            "anchor_point_count": (geom or {}).get("anchor_point_count", ""),
            "anchor_depth_mean": (geom or {}).get("anchor_depth_mean", ""),
            "anchor_depth_std": (geom or {}).get("anchor_depth_std", ""),
            "anchor_inverse_depth_std": (geom or {}).get("anchor_inverse_depth_std", ""),
            "anchor_frame_span": (geom or {}).get("anchor_frame_span", ""),
            "anchor_cross_chunk_pixel_motion_proxy": (geom or {}).get("anchor_cross_chunk_pixel_motion_proxy", ""),
            "O_scale_repaired": (geom or {}).get("O_scale_repaired", ""),
            "raw_query_depth_mean": (raw_edge or {}).get("raw_query_depth_mean", ""),
            "raw_cache_depth_mean": (raw_edge or {}).get("raw_cache_depth_mean", ""),
            "raw_anchor_frame_span": (raw_edge or {}).get("raw_anchor_frame_span", ""),
            "raw_world_pair_distance_mean": (raw_edge or {}).get("raw_world_pair_distance_mean", ""),
            "raw_abs_log_depth_ratio_mean": (raw_edge or {}).get("raw_abs_log_depth_ratio_mean", ""),
            "raw_abs_depth_diff_mean": (raw_edge or {}).get("raw_abs_depth_diff_mean", ""),
            "claim_level": "lifecycle_geometry_observability_join_diagnostic_no_action",
        }
        join_rows.append(out_row)
        case_parts[case_id].append(out_row)

    case_rows: list[dict[str, Any]] = []
    for case_id, parts in sorted(case_parts.items()):
        joined_parts = [row for row in parts if row.get("geometry_joined")]
        raw_joined_parts = [row for row in parts if row.get("raw_geometry_edge_joined")]
        combined_parts = [
            row
            for row in parts
            if row.get("geometry_joined") or row.get("raw_geometry_edge_joined")
        ]
        true_geom_parts = [row for row in joined_parts if b(row.get("true_geometry_source_available"))]
        tax = target.get(case_id, {}).get("target_taxonomy", parts[0].get("target_taxonomy", "") if parts else "")
        case_rows.append(
            {
                "case_id": case_id,
                "target_taxonomy": tax,
                "lifecycle_unique_case_anchor_count": len(parts),
                "geometry_joined_unique_case_anchor_count": len(joined_parts),
                "geometry_join_unique_coverage": frac(len(joined_parts), len(parts)),
                "raw_geometry_edge_joined_unique_case_anchor_count": len(raw_joined_parts),
                "raw_geometry_edge_unique_coverage": frac(len(raw_joined_parts), len(parts)),
                "combined_geometry_joined_unique_case_anchor_count": len(combined_parts),
                "combined_geometry_unique_coverage": frac(len(combined_parts), len(parts)),
                "true_geometry_source_available_unique_case_anchor_count": len(true_geom_parts),
                "true_geometry_source_available_frac": frac(len(true_geom_parts), len(joined_parts)),
                "temporal_proxy_only_unique_case_anchor_count": sum(1 for row in joined_parts if b(row.get("temporal_proxy_only"))),
                "O_scale_repaired_mean": mean([row.get("O_scale_repaired") for row in joined_parts]),
                "anchor_point_count_mean": mean([row.get("anchor_point_count") for row in joined_parts]),
                "raw_geometry_edge_count_mean": mean([row.get("raw_geometry_edge_count") for row in raw_joined_parts]),
                "raw_abs_log_depth_ratio_mean": mean([row.get("raw_abs_log_depth_ratio_mean") for row in raw_joined_parts]),
            }
        )

    lifecycle_unique_case_anchor_count = len(lifecycle_by_key)
    joined_unique_count = len(joined_keys)
    raw_edge_joined_unique_count = len(raw_edge_joined_keys)
    combined_joined_keys = joined_keys | raw_edge_joined_keys
    combined_joined_unique_count = len(combined_joined_keys)
    joined_row_count = sum(len(lifecycle_by_key[k]) for k in joined_keys)
    raw_edge_joined_row_count = sum(len(lifecycle_by_key[k]) for k in raw_edge_joined_keys)
    combined_joined_row_count = sum(len(lifecycle_by_key[k]) for k in combined_joined_keys)
    true_geometry_count = len(true_geometry_joined_keys)
    trackv_gate_pass = geometry_summary.get("gate_pass") is True
    trackv_materialized = geometry_summary.get("geometry_materialization_pass") is True
    scale_mode_row_count = row_count(SCALE_MODE_ROWS)
    pointmap_depth_row_count = row_count(POINTMAP_DEPTH_ROWS)
    lifecycle_geometry_unique_coverage = frac(joined_unique_count, lifecycle_unique_case_anchor_count)
    lifecycle_raw_edge_unique_coverage = frac(raw_edge_joined_unique_count, lifecycle_unique_case_anchor_count)
    lifecycle_combined_unique_coverage = frac(combined_joined_unique_count, lifecycle_unique_case_anchor_count)
    true_geometry_joined_frac = frac(true_geometry_count, joined_unique_count)
    lifecycle_true_stage_pass = (
        trackv_gate_pass
        and lifecycle_combined_unique_coverage >= 0.8
        and true_geometry_joined_frac >= 0.8
        and pointmap_depth_row_count > 0
    )

    summary = {
        "schema": "acl2_v101_lifecycle_geometry_observability_join_v1",
        "diagnostic_only": True,
        "lifecycle_expanded_row_count": len(lifecycle_rows),
        "lifecycle_row_with_anchor_count": lifecycle_row_with_anchor_count,
        "lifecycle_unique_case_anchor_count": lifecycle_unique_case_anchor_count,
        "geometry_row_count": len(geometry_rows),
        "geometry_unique_case_anchor_count": len(geometry_by_key),
        "raw_geometry_edge_row_count": raw_geometry_edge_row_count,
        "raw_geometry_edge_unique_case_anchor_count": len(raw_edge_by_key),
        "pointmap_depth_support_row_count": pointmap_depth_row_count,
        "scale_mode_row_count": scale_mode_row_count,
        "trackv_geometry_materialization_pass": trackv_materialized,
        "trackv_gate_pass": trackv_gate_pass,
        "trackv_blockers": geometry_summary.get("blockers", []),
        "trackv_handoff_target_case_count": geometry_summary.get("handoff_target_case_count", ""),
        "trackv_safe_good_case_count": geometry_summary.get("safe_good_case_count", ""),
        "lifecycle_geometry_joined_row_count": joined_row_count,
        "lifecycle_geometry_joined_unique_case_anchor_count": joined_unique_count,
        "lifecycle_geometry_row_coverage": frac(joined_row_count, lifecycle_row_with_anchor_count),
        "lifecycle_geometry_unique_coverage": lifecycle_geometry_unique_coverage,
        "lifecycle_raw_geometry_edge_joined_row_count": raw_edge_joined_row_count,
        "lifecycle_raw_geometry_edge_joined_unique_case_anchor_count": raw_edge_joined_unique_count,
        "lifecycle_raw_geometry_edge_row_coverage": frac(raw_edge_joined_row_count, lifecycle_row_with_anchor_count),
        "lifecycle_raw_geometry_edge_unique_coverage": lifecycle_raw_edge_unique_coverage,
        "lifecycle_combined_geometry_joined_row_count": combined_joined_row_count,
        "lifecycle_combined_geometry_joined_unique_case_anchor_count": combined_joined_unique_count,
        "lifecycle_combined_geometry_row_coverage": frac(combined_joined_row_count, lifecycle_row_with_anchor_count),
        "lifecycle_combined_geometry_unique_coverage": lifecycle_combined_unique_coverage,
        "lifecycle_true_geometry_joined_unique_case_anchor_count": true_geometry_count,
        "lifecycle_true_geometry_joined_frac": true_geometry_joined_frac,
        "lifecycle_temporal_proxy_only_joined_unique_case_anchor_count": len(temporal_proxy_joined_keys),
        "lifecycle_handoff_unique_case_anchor_count": len(handoff_keys),
        "lifecycle_handoff_geometry_joined_unique_case_anchor_count": len(handoff_joined_keys),
        "lifecycle_handoff_geometry_join_coverage": frac(len(handoff_joined_keys), len(handoff_keys)),
        "lifecycle_safe_good_unique_case_anchor_count": len(safe_keys),
        "lifecycle_safe_good_geometry_joined_unique_case_anchor_count": len(safe_joined_keys),
        "lifecycle_safe_good_geometry_join_coverage": frac(len(safe_joined_keys), len(safe_keys)),
        "lifecycle_joined_O_scale_repaired_mean": mean([row.get("O_scale_repaired") for row in join_rows if row.get("geometry_joined")]),
        "lifecycle_geometry_join_materialized": joined_unique_count > 0,
        "lifecycle_raw_geometry_edge_join_materialized": raw_edge_joined_unique_count > 0,
        "lifecycle_combined_geometry_join_materialized": combined_joined_unique_count > 0,
        "lifecycle_true_geometry_source_available": true_geometry_count > 0,
        "lifecycle_scale_mode_materialized": scale_mode_row_count > 0,
        "lifecycle_scale_observability_true_stage_pass": lifecycle_true_stage_pass,
        "q2_true_stage_ready": False,
        "runtime_action_allowed": False,
        "method_goal_achieved": False,
        "claim": (
            "Track V geometry sidecar can be joined to a small lifecycle anchor subset; "
            "strict true-stage observability remains blocked by coverage and Track V gate failures."
        ),
    }

    write_rows(JOIN_ROWS_OUT, join_rows)
    write_rows(CASE_ROWS_OUT, case_rows)
    write_json(SUMMARY_OUT, summary)
    REPORT_OUT.write_text(
        "\n".join(
            [
                "# ACL2 v101 Lifecycle Geometry Observability Join",
                "",
                "This audit joins lifecycle anchor rows to the existing Track V per-anchor geometry observability sidecar. It is no-action.",
                "",
                "## Summary",
                "",
                f"- lifecycle_unique_case_anchor_count: {summary['lifecycle_unique_case_anchor_count']}",
                f"- lifecycle_geometry_joined_unique_case_anchor_count: {summary['lifecycle_geometry_joined_unique_case_anchor_count']}",
                f"- lifecycle_geometry_unique_coverage: {summary['lifecycle_geometry_unique_coverage']}",
                f"- lifecycle_raw_geometry_edge_joined_unique_case_anchor_count: {summary['lifecycle_raw_geometry_edge_joined_unique_case_anchor_count']}",
                f"- lifecycle_raw_geometry_edge_unique_coverage: {summary['lifecycle_raw_geometry_edge_unique_coverage']}",
                f"- lifecycle_combined_geometry_joined_unique_case_anchor_count: {summary['lifecycle_combined_geometry_joined_unique_case_anchor_count']}",
                f"- lifecycle_combined_geometry_unique_coverage: {summary['lifecycle_combined_geometry_unique_coverage']}",
                f"- lifecycle_true_geometry_joined_frac: {summary['lifecycle_true_geometry_joined_frac']}",
                f"- pointmap_depth_support_row_count: {summary['pointmap_depth_support_row_count']}",
                f"- scale_mode_row_count: {summary['scale_mode_row_count']}",
                f"- trackv_geometry_materialization_pass: {summary['trackv_geometry_materialization_pass']}",
                f"- trackv_gate_pass: {summary['trackv_gate_pass']}",
                f"- lifecycle_scale_observability_true_stage_pass: {summary['lifecycle_scale_observability_true_stage_pass']}",
                "",
                "## Blocker",
                "",
                "The existing geometry sidecar is materialized, but lifecycle join coverage is small and Track V strict gate is still false. This cannot authorize Q2 true-stage, M4, runtime, or full validation.",
                "",
                "## Artifacts",
                "",
                f"- `{JOIN_ROWS_OUT}`",
                f"- `{CASE_ROWS_OUT}`",
                f"- `{SUMMARY_OUT}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema": summary["schema"],
                "lifecycle_geometry_joined_unique_case_anchor_count": joined_unique_count,
                "lifecycle_geometry_unique_coverage": lifecycle_geometry_unique_coverage,
                "lifecycle_raw_geometry_edge_joined_unique_case_anchor_count": raw_edge_joined_unique_count,
                "lifecycle_combined_geometry_joined_unique_case_anchor_count": combined_joined_unique_count,
                "lifecycle_combined_geometry_unique_coverage": lifecycle_combined_unique_coverage,
                "lifecycle_true_geometry_joined_frac": true_geometry_joined_frac,
                "lifecycle_scale_observability_true_stage_pass": lifecycle_true_stage_pass,
                "runtime_action_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
