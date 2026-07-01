#!/usr/bin/env python3
"""Materialize v101 per-anchor geometry observability from v100 L2 sidecars.

The output repairs the earlier case-level-only Track V diagnostic by joining
v101 S-B support anchors to v100 per-anchor geometry edge rows.  It remains a
diagnostic materialization: no runtime action is authorized here.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
TRACK_U = ROOT / "trackU_true_current_support"
TRACK_T = ROOT / "trackT_drift_target_relabel"
TRACK_V = ROOT / "trackV_anchor_scale_observability"
V100_L2 = Path("results/acl2_v100tf_same_space_semantic_anchor_latent_state_memory_control/trackL2_anchor_scale_observability")

SUPPORT_ROWS = TRACK_U / "anchor_current_support_rows.csv"
TARGET_ROWS = TRACK_T / "target_universe_v101.csv"
GEOMETRY_EDGE_ROWS = V100_L2 / "geometry_edge_rows.csv"

EPS = 1.0e-9


def f(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


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
            clean: dict[str, Any] = {}
            for key in keys:
                value = row.get(key, "")
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, ensure_ascii=False, sort_keys=True)
                clean[key] = value
            writer.writerow(clean)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def mean(values: list[Any]) -> float:
    vals = [f(v) for v in values if math.isfinite(f(v))]
    return sum(vals) / len(vals) if vals else math.nan


def std(values: list[Any]) -> float:
    vals = [f(v) for v in values if math.isfinite(f(v))]
    if len(vals) < 2:
        return 0.0 if vals else math.nan
    mu = sum(vals) / len(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))


def median(values: list[Any]) -> float:
    vals = sorted(f(v) for v in values if math.isfinite(f(v)))
    return statistics.median(vals) if vals else math.nan


def quantile(values: list[Any], q: float) -> float:
    vals = sorted(f(v) for v in values if math.isfinite(f(v)))
    if not vals:
        return math.nan
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def norm01(value: Any, values: list[Any], *, invert: bool = False) -> float:
    vals = [f(v) for v in values if math.isfinite(f(v))]
    fv = f(value)
    if not vals or not math.isfinite(fv):
        return math.nan
    lo = min(vals)
    hi = max(vals)
    if hi - lo <= EPS:
        out = 0.5
    else:
        out = (fv - lo) / (hi - lo)
    out = max(0.0, min(1.0, out))
    return 1.0 - out if invert else out


def bounds(values: list[Any]) -> tuple[float, float]:
    vals = [f(v) for v in values if math.isfinite(f(v))]
    return (min(vals), max(vals)) if vals else (math.nan, math.nan)


def norm01_bounds(value: Any, lo: float, hi: float, *, invert: bool = False) -> float:
    fv = f(value)
    if not all(math.isfinite(v) for v in [fv, lo, hi]):
        return math.nan
    if hi - lo <= EPS:
        out = 0.5
    else:
        out = (fv - lo) / (hi - lo)
    out = max(0.0, min(1.0, out))
    return 1.0 - out if invert else out


def pearson(xs: list[Any], ys: list[Any]) -> float:
    pairs: list[tuple[float, float]] = []
    for x, y in zip(xs, ys):
        fx = f(x)
        fy = f(y)
        if math.isfinite(fx) and math.isfinite(fy):
            pairs.append((fx, fy))
    if len(pairs) < 2:
        return math.nan
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 1.0e-12 or vy <= 1.0e-12:
        return math.nan
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def first_by_anchor(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row.get("case_id", ""), row.get("anchor_id", ""))
        if key not in out:
            out[key] = row
    return out


def finite_fraction(values: list[Any]) -> float:
    if not values:
        return 0.0
    return sum(1.0 for value in values if math.isfinite(f(value))) / len(values)


def main() -> None:
    support_rows = read_rows(SUPPORT_ROWS)
    target_rows = {row.get("case_id", ""): row for row in read_rows(TARGET_ROWS)}
    support_by_anchor = first_by_anchor(support_rows)
    geom_by_anchor: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_rows(GEOMETRY_EDGE_ROWS):
        geom_by_anchor[(row.get("case_id", ""), row.get("anchor_id", ""))].append(row)

    support_keys = set(support_by_anchor)
    geom_keys = set(geom_by_anchor)
    overlap_keys = support_keys & geom_keys
    all_query_depths = [f(row.get("query_depth")) for rows in geom_by_anchor.values() for row in rows]
    far_depth_threshold = quantile(all_query_depths, 0.75)

    prelim: list[dict[str, Any]] = []
    for key, support in support_by_anchor.items():
        case_id, anchor_id = key
        geom_rows = geom_by_anchor.get(key, [])
        query_depths = [f(row.get("query_depth")) for row in geom_rows]
        cache_depths = [f(row.get("cache_depth")) for row in geom_rows]
        inv_depths = [1.0 / max(f(row.get("query_depth")), EPS) for row in geom_rows if math.isfinite(f(row.get("query_depth")))]
        query_frames = [f(row.get("query_frame")) for row in geom_rows]
        cache_frames = [f(row.get("cache_frame")) for row in geom_rows]
        frame_deltas = [abs(f(row.get("query_frame")) - f(row.get("cache_frame"))) for row in geom_rows]
        pixel_motions = []
        for row in geom_rows:
            qr = f(row.get("query_patch_row"))
            qc = f(row.get("query_patch_col"))
            cr = f(row.get("cache_patch_row"))
            cc = f(row.get("cache_patch_col"))
            if all(math.isfinite(v) for v in [qr, qc, cr, cc]):
                pixel_motions.append(math.hypot(qr - cr, qc - cc))
        baselines = [f(row.get("camera_translation_baseline")) for row in geom_rows]
        world_distances = [f(row.get("world_pair_distance")) for row in geom_rows]
        abs_log_depth_ratios = [f(row.get("abs_log_depth_ratio")) for row in geom_rows]
        abs_depth_diffs = [f(row.get("abs_depth_diff")) for row in geom_rows]
        query_conf = [f(row.get("query_conf")) for row in geom_rows]
        cache_conf = [f(row.get("cache_conf")) for row in geom_rows]
        depth_mean = mean(query_depths)
        baseline_over_depth = mean([v / max(depth_mean, EPS) for v in baselines if math.isfinite(v) and math.isfinite(depth_mean)])
        target = target_rows.get(case_id, {})
        prelim.append(
            {
                "case_id": case_id,
                "anchor_id": anchor_id,
                "semantic_label": support.get("semantic_label", ""),
                "target_taxonomy": target.get("target_taxonomy", support.get("target_taxonomy", "")),
                "L3_handoff_transfer_penalty_proxy": target.get("L3_handoff_transfer_penalty_proxy", support.get("L3_handoff_transfer_penalty_proxy", "")),
                "geometry_edge_count": len(geom_rows),
                "geometry_source_level": "per_anchor_geometry_sidecar_depth" if geom_rows else "missing",
                "true_geometry_source_available": bool(geom_rows) and finite_fraction(query_depths) > 0.0,
                "temporal_proxy_only": False if geom_rows else True,
                "anchor_depth_mean": depth_mean,
                "anchor_depth_std": std(query_depths),
                "anchor_cache_depth_mean": mean(cache_depths),
                "anchor_inverse_depth_std": std(inv_depths),
                "anchor_point_count": len([v for v in query_depths if math.isfinite(v)]),
                "far_depth_fraction": mean([1.0 if math.isfinite(v) and math.isfinite(far_depth_threshold) and v >= far_depth_threshold else 0.0 for v in query_depths]),
                "anchor_frame_span": (max(query_frames) - min(query_frames)) if any(math.isfinite(v) for v in query_frames) else math.nan,
                "anchor_top1_frame_delta": mean(frame_deltas),
                "anchor_cross_chunk_pixel_motion_proxy": mean(pixel_motions),
                "anchor_world_pair_distance_mean": mean(world_distances),
                "anchor_abs_log_depth_ratio_mean": mean(abs_log_depth_ratios),
                "anchor_abs_depth_diff_mean": mean(abs_depth_diffs),
                "anchor_baseline_mean": mean(baselines),
                "anchor_baseline_over_depth": baseline_over_depth,
                "anchor_query_conf_mean": mean(query_conf),
                "anchor_cache_conf_mean": mean(cache_conf),
                "support_source_flags": support.get("support_source_flags", ""),
                "support_quality": support.get("support_quality", ""),
            }
        )

    point_counts = [row["anchor_point_count"] for row in prelim]
    inv_stds = [row["anchor_inverse_depth_std"] for row in prelim]
    pixel_motions = [row["anchor_cross_chunk_pixel_motion_proxy"] for row in prelim]
    world_distances = [row["anchor_world_pair_distance_mean"] for row in prelim]
    baseline_over_depths = [row["anchor_baseline_over_depth"] for row in prelim]
    conf_means = [mean([row["anchor_query_conf_mean"], row["anchor_cache_conf_mean"]]) for row in prelim]
    depth_ratio_means = [row["anchor_abs_log_depth_ratio_mean"] for row in prelim]
    point_bounds = bounds(point_counts)
    inv_std_bounds = bounds(inv_stds)
    pixel_motion_bounds = bounds(pixel_motions)
    world_distance_bounds = bounds(world_distances)
    baseline_over_depth_bounds = bounds(baseline_over_depths)
    conf_bounds = bounds(conf_means)
    depth_ratio_bounds = bounds(depth_ratio_means)

    rows: list[dict[str, Any]] = []
    for row in prelim:
        depth_spread_score = norm01_bounds(row["anchor_inverse_depth_std"], *inv_std_bounds)
        parallax_score = mean(
            [
                norm01_bounds(row["anchor_cross_chunk_pixel_motion_proxy"], *pixel_motion_bounds),
                norm01_bounds(row["anchor_world_pair_distance_mean"], *world_distance_bounds),
                norm01_bounds(row["anchor_baseline_over_depth"], *baseline_over_depth_bounds),
            ]
        )
        geometry_condition_score = mean(
            [
                norm01_bounds(row["anchor_point_count"], *point_bounds),
                norm01_bounds(mean([row["anchor_query_conf_mean"], row["anchor_cache_conf_mean"]]), *conf_bounds),
                norm01_bounds(row["anchor_abs_log_depth_ratio_mean"], *depth_ratio_bounds, invert=True),
            ]
        )
        risk_penalty = 0.0
        if row["target_taxonomy"] in {"MULTIMODE_LOWOBS_ABSTAIN", "GOOD_HIGH_L3_CONTAMINATED"}:
            risk_penalty = 0.35
        elif "LOW_OBSERVABILITY" in str(row["target_taxonomy"]):
            risk_penalty = 0.20
        o_raw = mean([depth_spread_score, parallax_score, geometry_condition_score])
        o_scale = max(0.0, min(1.0, o_raw * (1.0 - risk_penalty))) if math.isfinite(o_raw) else math.nan
        rows.append(
            {
                **row,
                "D_spread_score": depth_spread_score,
                "P_parallax_score": parallax_score,
                "G_condition_score": geometry_condition_score,
                "semantic_risk_penalty": risk_penalty,
                "O_scale_repaired": o_scale,
                "claim_level": "per_anchor_geometry_sidecar_diagnostic_no_action",
            }
        )

    case_means: dict[str, dict[str, Any]] = {}
    for case_id in sorted({row["case_id"] for row in rows}):
        case_rows = [row for row in rows if row["case_id"] == case_id]
        target = target_rows.get(case_id, {})
        case_means[case_id] = {
            "case_id": case_id,
            "target_taxonomy": target.get("target_taxonomy", ""),
            "case_label": target.get("case_label", ""),
            "failure_type": target.get("failure_type", ""),
            "L3_handoff_transfer_penalty_proxy": target.get("L3_handoff_transfer_penalty_proxy", ""),
            "anchor_count": len(case_rows),
            "geometry_available_anchor_frac": mean([1.0 if row["true_geometry_source_available"] else 0.0 for row in case_rows]),
            "O_scale_repaired_mean": mean([row["O_scale_repaired"] for row in case_rows]),
            "O_scale_repaired_p25": quantile([row["O_scale_repaired"] for row in case_rows], 0.25),
            "O_scale_repaired_p75": quantile([row["O_scale_repaired"] for row in case_rows], 0.75),
        }

    case_rows = list(case_means.values())
    safe_cases = [row for row in case_rows if row["target_taxonomy"] == "SAFE_GOOD"]
    handoff_cases = [row for row in case_rows if row["target_taxonomy"] == "HANDOFF_SCALE_GAUGE_TARGET"]
    low_o_threshold = quantile([row["O_scale_repaired_mean"] for row in case_rows], 0.25)
    safe_good_low_o_fpr = (
        sum(1 for row in safe_cases if f(row["O_scale_repaired_mean"]) <= low_o_threshold) / len(safe_cases)
        if safe_cases
        else math.nan
    )
    corr_all = pearson([row["O_scale_repaired_mean"] for row in case_rows], [row["L3_handoff_transfer_penalty_proxy"] for row in case_rows])
    corr_non_ambig = pearson(
        [
            row["O_scale_repaired_mean"]
            for row in case_rows
            if row["target_taxonomy"] in {"HANDOFF_SCALE_GAUGE_TARGET", "SAFE_GOOD", "LOCAL_BAD_NOT_HANDOFF"}
        ],
        [
            row["L3_handoff_transfer_penalty_proxy"]
            for row in case_rows
            if row["target_taxonomy"] in {"HANDOFF_SCALE_GAUGE_TARGET", "SAFE_GOOD", "LOCAL_BAD_NOT_HANDOFF"}
        ],
    )
    target_anchor_rows = [row for row in rows if row["target_taxonomy"] == "HANDOFF_SCALE_GAUGE_TARGET"]
    target_available_frac = mean([1.0 if row["true_geometry_source_available"] else 0.0 for row in target_anchor_rows]) if target_anchor_rows else 0.0
    geometry_materialization_pass = (
        bool(rows)
        and len(overlap_keys) / max(len(support_keys), 1) >= 0.95
        and target_available_frac >= 0.80
        and mean([1.0 if row["true_geometry_source_available"] else 0.0 for row in rows]) >= 0.80
    )
    strict_gate_pass = (
        geometry_materialization_pass
        and len(handoff_cases) >= 8
        and safe_good_low_o_fpr <= 0.30
        and math.isfinite(corr_non_ambig)
        and corr_non_ambig < 0.0
    )
    blockers = []
    if len(handoff_cases) < 8:
        blockers.append("Track T provides fewer than 8 HANDOFF_SCALE_GAUGE_TARGET cases.")
    if not math.isfinite(corr_non_ambig) or corr_non_ambig >= 0.0:
        blockers.append("Repaired O_scale does not show required negative case-level L3 relation on non-ambiguous target/control cases.")
    if math.isfinite(safe_good_low_o_fpr) and safe_good_low_o_fpr > 0.30:
        blockers.append("SAFE_GOOD low-observability FPR is above 0.30.")
    blockers.append("Anchor-id/semantic query-head control reruns are not materialized in this repair script.")

    summary = {
        "schema": "acl2_v101_per_anchor_geometry_observability_repair_v1",
        "support_unique_anchor_count": len(support_keys),
        "geometry_unique_anchor_count": len(geom_keys),
        "overlap_anchor_count": len(overlap_keys),
        "overlap_fraction_of_support": len(overlap_keys) / max(len(support_keys), 1),
        "missing_support_anchor_count": len(support_keys - geom_keys),
        "case_coverage": len({key[0] for key in support_keys}),
        "true_geometry_source_available_frac": mean([1.0 if row["true_geometry_source_available"] else 0.0 for row in rows]),
        "handoff_target_anchor_count": len(target_anchor_rows),
        "handoff_target_geometry_available_frac": target_available_frac,
        "safe_good_case_count": len(safe_cases),
        "handoff_target_case_count": len(handoff_cases),
        "safe_good_low_observability_fpr": safe_good_low_o_fpr,
        "case_mean_O_scale_corr_L3_all": corr_all,
        "case_mean_O_scale_corr_L3_non_ambiguous": corr_non_ambig,
        "geometry_materialization_pass": geometry_materialization_pass,
        "gate_pass": strict_gate_pass,
        "runtime_action_allowed": False,
        "blockers": blockers,
        "claim": "Per-anchor depth/parallax geometry sidecar was materialized for diagnostics; strict Track V remains blocked by Track T size/correlation/control evidence.",
    }

    write_rows(TRACK_V / "per_anchor_geometry_observability_rows.csv", rows)
    write_rows(TRACK_V / "per_anchor_geometry_case_summary.csv", case_rows)
    write_json(TRACK_V / "per_anchor_geometry_observability_summary.json", summary)
    write_text(
        TRACK_V / "per_anchor_geometry_observability_report.md",
        "# Per-Anchor Geometry Observability Repair\n\n"
        f"- Support unique anchors: {summary['support_unique_anchor_count']}\n"
        f"- Geometry overlap anchors: {summary['overlap_anchor_count']}\n"
        f"- Overlap fraction of support: {summary['overlap_fraction_of_support']}\n"
        f"- True geometry source available fraction: {summary['true_geometry_source_available_frac']}\n"
        f"- HANDOFF target anchors with geometry: {summary['handoff_target_geometry_available_frac']}\n"
        f"- SAFE_GOOD low-observability FPR: {summary['safe_good_low_observability_fpr']}\n"
        f"- Corr(mean O_scale, L3) all: {summary['case_mean_O_scale_corr_L3_all']}\n"
        f"- Corr(mean O_scale, L3) non-ambiguous: {summary['case_mean_O_scale_corr_L3_non_ambiguous']}\n"
        f"- Geometry materialization pass: {summary['geometry_materialization_pass']}\n"
        f"- Strict Track V gate pass: {summary['gate_pass']}\n\n"
        "Blockers:\n"
        + "\n".join(f"- {item}" for item in blockers)
        + "\n",
    )

    support_repair_summary = {
        "schema": "acl2_v101_trackU_per_anchor_geometry_support_repair_v1",
        "support_unique_anchor_count": len(support_keys),
        "geometry_overlap_anchor_count": len(overlap_keys),
        "overlap_fraction_of_support": len(overlap_keys) / max(len(support_keys), 1),
        "case_coverage": len({key[0] for key in support_keys}),
        "support_repair_source": str(GEOMETRY_EDGE_ROWS),
        "strict_instance_semantic_support": False,
        "runtime_action_allowed": False,
        "claim": "Per-anchor geometry support source is available for almost all S-B support anchors; semantic instance-level support remains unresolved.",
    }
    write_json(TRACK_U / "per_anchor_geometry_support_repair_summary.json", support_repair_summary)
    write_text(
        TRACK_U / "per_anchor_geometry_support_repair_report.md",
        "# Per-Anchor Geometry Support Repair\n\n"
        f"- Support unique anchors: {support_repair_summary['support_unique_anchor_count']}\n"
        f"- Geometry overlap anchors: {support_repair_summary['geometry_overlap_anchor_count']}\n"
        f"- Overlap fraction: {support_repair_summary['overlap_fraction_of_support']}\n"
        f"- Case coverage: {support_repair_summary['case_coverage']}\n\n"
        "This repairs the non-feature geometry-support availability evidence, but it does not create semantic instance ids and does not authorize runtime action.\n",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
