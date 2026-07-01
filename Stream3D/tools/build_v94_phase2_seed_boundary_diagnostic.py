#!/usr/bin/env python3
"""Build v94 Phase2 seed/boundary diagnostics before inference variants."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PHASE_ID = "v94_phase2_seed_boundary_diagnostic"
RUN_ID = "v94_phase2_seed_boundary_diagnostic"
OUT = ROOT / "outputs/audit/v94_phase2_seed_boundary_diagnostic"

V94_PHASE1 = ROOT / "outputs/audit/v94_phase1_canonical_graph"
V93_PHASE1 = ROOT / "outputs/audit/v93_phase1_source_edge_registry"
V93_PHASE2 = ROOT / "outputs/audit/v93_phase2_d4rt_edge_sampling_diagnostic"
V93_PHASE3 = ROOT / "outputs/audit/v93_phase3_region_edge_graph"


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_jsonable(row))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _p10(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return float(values[max(0, int(math.floor(0.1 * (len(values) - 1))))])


def _load_container_object_counts(path: Path) -> Counter[tuple[str, str, str, str, str]]:
    counts: Counter[tuple[str, str, str, str, str]] = Counter()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                str(row.get("variant_id", "")),
                str(row.get("scene_id", "")),
                str(row.get("window_id", "")),
                str(row.get("frame_id", "")),
                str(row.get("source_mask_id", "")),
            )
            counts[key] += 1
    return counts


def _build_object_seed_rows(created_at: str, link_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    container_counts = _load_container_object_counts(link_path)
    rows: list[dict[str, Any]] = []
    carrier_masses: list[float] = []
    score_values: list[float] = []
    selected_count = 0
    conflict_count = 0
    with link_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                str(row.get("variant_id", "")),
                str(row.get("scene_id", "")),
                str(row.get("window_id", "")),
                str(row.get("frame_id", "")),
                str(row.get("source_mask_id", "")),
            )
            carrier_mass = _num(row.get("carrier_support_mass"))
            score = _num(row.get("adapter_score_raw", row.get("mask_selected_score", "")))
            is_selected = _bool(row.get("mask_selected_by_variant"))
            object_count = container_counts[key]
            has_seed = carrier_mass > 0.0 or score > 0.0
            conflict = object_count > 1
            carrier_masses.append(carrier_mass)
            score_values.append(score)
            selected_count += int(is_selected)
            conflict_count += int(conflict)
            rows.append(
                {
                    "schema_version": "stream4d_v94_phase2_object_seed_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "variant_id": row.get("variant_id", ""),
                    "scene_id": row.get("scene_id", ""),
                    "split": row.get("split", "dev"),
                    "window_id": row.get("window_id", ""),
                    "frame_id": row.get("frame_id", ""),
                    "source_mask_id": row.get("source_mask_id", ""),
                    "object_hypothesis_id": row.get("object_hypothesis_id", ""),
                    "seed_source": "v93_object_container_link_carrier_support_proxy",
                    "carrier_support_mass": carrier_mass,
                    "adapter_score_raw": score,
                    "mask_area_px": _num(row.get("mask_area_px")),
                    "object_count_in_variant_container": object_count,
                    "seed_available_proxy": has_seed,
                    "seed_conflict_proxy": conflict,
                    "seed_area_ratio_proxy": carrier_mass,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                    "created_at": created_at,
                }
            )
    stats = {
        "object_seed_row_count": len(rows),
        "seed_coverage_rate": float(sum(1 for row in rows if row["seed_available_proxy"]) / len(rows)) if rows else 0.0,
        "seed_area_ratio_mean": "",
        "seed_area_ratio_p10": "",
        "seed_area_ratio_available": False,
        "seed_area_ratio_note": "Not directly available in v93 object-container links; carrier_support_mass is recorded separately as a seed proxy.",
        "seed_area_ratio_proxy_mean": _mean([_num(row["seed_area_ratio_proxy"]) for row in rows]),
        "seed_area_ratio_proxy_p10": _p10([_num(row["seed_area_ratio_proxy"]) for row in rows]),
        "seed_conflict_rate": float(conflict_count / len(rows)) if rows else 0.0,
        "D4RT_positive_mass_mean": _mean(carrier_masses),
        "D4RT_negative_mass_mean": 0.0,
        "adapter_score_mean": _mean(score_values),
        "mask_selected_rate": float(selected_count / len(rows)) if rows else 0.0,
    }
    return rows, stats


def _build_d4rt_quality_rows(created_at: str, source_support_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "support": [], "carrier": [], "confidence": [], "boundary": []})
    with source_support_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            variant = str(row.get("variant_id", ""))
            item = grouped[variant]
            item["count"] += 1
            item["support"].append(_num(row.get("carrier_support_area_ratio")))
            item["carrier"].append(_num(row.get("carrier_count_inside_source")))
            item["confidence"].append(_num(row.get("carrier_confidence_mean")))
            item["boundary"].append(_num(row.get("mask_boundary_carrier_distance_mean")))

    rows: list[dict[str, Any]] = []
    for variant, item in sorted(grouped.items()):
        rows.append(
            {
                "schema_version": "stream4d_v94_phase2_d4rt_witness_quality_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": variant,
                "source_support_row_count": item["count"],
                "carrier_count_inside_source_mean": _mean(item["carrier"]),
                "carrier_support_area_ratio_mean": _mean(item["support"]),
                "carrier_support_area_ratio_p10": _p10(item["support"]),
                "carrier_confidence_mean": _mean(item["confidence"]),
                "mask_boundary_carrier_distance_mean": _mean(item["boundary"]),
                "row_mode": "source_support_proxy_not_per_carrier_witness",
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "created_at": created_at,
            }
        )
    all_support = [value for item in grouped.values() for value in item["support"]]
    all_carrier = [value for item in grouped.values() for value in item["carrier"]]
    stats = {
        "d4rt_source_support_variant_count": len(grouped),
        "d4rt_source_support_row_count": sum(int(item["count"]) for item in grouped.values()),
        "D4RT_source_support_area_ratio_mean": _mean(all_support),
        "D4RT_source_support_area_ratio_p10": _p10(all_support),
        "D4RT_carrier_count_mean": _mean(all_carrier),
    }
    return rows, stats


def _copy_edge_diagnostic_rows(created_at: str, src: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    nested_counts = []
    competing_counts = []
    source_counts = []
    with src.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source_count = _num(row.get("source_container_count"))
            nested = _num(row.get("nested_overlap_edge_count"))
            competing = _num(row.get("competing_edge_count"))
            source_counts.append(source_count)
            nested_counts.append(nested)
            competing_counts.append(competing)
            out = dict(row)
            out["schema_version"] = "stream4d_v94_phase2_edge_diagnostic_v1"
            out["phase_id"] = PHASE_ID
            out["run_id"] = RUN_ID
            out["nested_edge_per_source"] = nested / source_count if source_count else 0.0
            out["competing_edge_per_source"] = competing / source_count if source_count else 0.0
            out["created_at"] = created_at
            rows.append(out)
    stats = {
        "nested_edge_available_rate": 1.0 if sum(nested_counts) > 0 else 0.0,
        "competing_edge_available_rate": 1.0 if sum(competing_counts) > 0 else 0.0,
        "nested_edge_per_source_mean": (sum(nested_counts) / sum(source_counts)) if sum(source_counts) else 0.0,
        "competing_edge_per_source_mean": (sum(competing_counts) / sum(source_counts)) if sum(source_counts) else 0.0,
    }
    return rows, stats


def _semantic_quality_rows(created_at: str, src: Path) -> list[dict[str, Any]]:
    rows = []
    with src.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out = dict(row)
            out["schema_version"] = "stream4d_v94_phase2_semantic_region_quality_v1"
            out["phase_id"] = PHASE_ID
            out["run_id"] = RUN_ID
            out["created_at"] = created_at
            rows.append(out)
    return rows


def run() -> dict[str, Any]:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    created_at = _created_at()

    phase1 = _read_json(V94_PHASE1 / "summary.json")
    v93_phase2 = _read_json(V93_PHASE2 / "summary.json")
    v93_phase3 = _read_json(V93_PHASE3 / "summary.json")

    object_seed_rows, seed_stats = _build_object_seed_rows(created_at, V94_PHASE1 / "container_object_link_rows.csv")
    d4rt_quality_rows, d4rt_stats = _build_d4rt_quality_rows(created_at, V94_PHASE1 / "d4rt_witness_rows.csv")
    edge_rows, edge_stats = _copy_edge_diagnostic_rows(created_at, V93_PHASE1 / "edge_registry_quality_rows.csv")
    semantic_rows = _semantic_quality_rows(created_at, V93_PHASE3 / "region_diagnostic_auc_rows.csv")

    d4rt_boundary_band_support_ratio = _num(v93_phase2.get("boundary_band_support_ratio"))
    d4rt_jitter_high = _bool(v93_phase2.get("jitter_high"))
    d4rt_flip_high = _bool(v93_phase2.get("membership_flip_high"))
    radio_auc = _num(v93_phase3.get("source_internal_same_gt_different_gt_AUC_diagnostic"))
    edge_barrier_density = _num(v93_phase3.get("edge_barrier_density"))
    multi_object_container_rate = _num(phase1.get("multi_object_container_rate_variant_key"))
    single_object_container_rate = 1.0 - multi_object_container_rate

    route_seed_weak = seed_stats["seed_coverage_rate"] < 0.20 or seed_stats["seed_conflict_rate"] > 0.30
    route_d4rt_weak = d4rt_boundary_band_support_ratio < 0.01 and d4rt_jitter_high
    route_radio_weak = radio_auc < 0.58
    route_edge_weak = edge_barrier_density < 0.05 or (edge_stats["nested_edge_available_rate"] == 0 and edge_stats["competing_edge_available_rate"] == 0)
    usable_signal_count = int(not route_d4rt_weak) + int(not route_radio_weak) + int(not route_edge_weak)
    route_ready = (not (route_seed_weak or route_d4rt_weak or route_radio_weak or route_edge_weak)) or usable_signal_count >= 2
    route_multi_object_sparse = multi_object_container_rate < 0.05

    route_rows = [
        {
            "schema_version": "stream4d_v94_phase2_route_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "route": "ROUTE_SEED_WEAK",
            "triggered": route_seed_weak,
            "evidence": {"seed_coverage_rate": seed_stats["seed_coverage_rate"], "seed_conflict_rate": seed_stats["seed_conflict_rate"]},
            "repair_direction": "Try probabilistic D4RT seed and object prototype seed; do not hard-threshold seeds.",
            "created_at": created_at,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v94_phase2_route_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "route": "ROUTE_D4RT_WEAK",
            "triggered": route_d4rt_weak,
            "evidence": {"D4RT_boundary_band_support_ratio": d4rt_boundary_band_support_ratio, "D4RT_projection_jitter_p90": v93_phase2.get("projection_jitter_p90"), "jitter_high": d4rt_jitter_high},
            "repair_direction": "Prepare adaptive edge/conflict sampling plan, but only run after field inference proves D4RT is the blocker.",
            "created_at": created_at,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v94_phase2_route_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "route": "ROUTE_RADIO_WEAK",
            "triggered": route_radio_weak,
            "evidence": {"RADIO_source_internal_auc_diagnostic": radio_auc},
            "repair_direction": "Repair region granularity / feature normalization; do not fall back to mask-level cosine.",
            "created_at": created_at,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v94_phase2_route_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "route": "ROUTE_EDGE_WEAK",
            "triggered": route_edge_weak,
            "evidence": {"edge_barrier_density": edge_barrier_density, **edge_stats},
            "repair_direction": "Repair nested/competing/repeated edge diagnostics before edge-dependent method claims.",
            "created_at": created_at,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v94_phase2_route_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "route": "ROUTE_MULTI_OBJECT_SPARSE",
            "triggered": route_multi_object_sparse,
            "evidence": {"multi_object_container_rate_variant_key": multi_object_container_rate, "single_object_container_rate": single_object_container_rate},
            "repair_direction": "Run Phase3A anyway, but record that current artifact universe barely stress-tests true multi-object competition unless source registry is rebuilt.",
            "created_at": created_at,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v94_phase2_route_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "route": "ROUTE_READY_FOR_MULTI_OBJECT_FIELD",
            "triggered": route_ready,
            "evidence": {"usable_signal_count_nonweak_among_D4RT_RADIO_EDGE": usable_signal_count},
            "repair_direction": "Proceed to Phase3A greedy assignment before any adaptive D4RT or DA3 branch.",
            "created_at": created_at,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
    ]

    decision = "PASS_V94_PHASE2_READY_FOR_PHASE3A" if route_ready else "BLOCK_V94_PHASE2_SIGNAL_DIAGNOSTIC"
    summary = {
        "schema": "stream4d_v94_phase2_seed_boundary_diagnostic_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": created_at,
        "decision": decision,
        "phase2_pass": route_ready,
        **seed_stats,
        **d4rt_stats,
        "D4RT_boundary_band_support_ratio": d4rt_boundary_band_support_ratio,
        "D4RT_projection_jitter_p90": v93_phase2.get("projection_jitter_p90"),
        "D4RT_membership_flip_median": v93_phase2.get("mask_membership_flip_rate_median"),
        "D4RT_jitter_high": d4rt_jitter_high,
        "D4RT_membership_flip_high": d4rt_flip_high,
        "RADIO_source_internal_auc_diagnostic": radio_auc,
        "RADIO_region_graph_LCC_ratio": v93_phase3.get("edge_graph_lcc_ratio_mean"),
        "edge_barrier_density": edge_barrier_density,
        **edge_stats,
        "multi_object_container_rate": multi_object_container_rate,
        "single_object_container_rate": single_object_container_rate,
        "route_seed_weak": route_seed_weak,
        "route_d4rt_weak": route_d4rt_weak,
        "route_radio_weak": route_radio_weak,
        "route_edge_weak": route_edge_weak,
        "route_multi_object_sparse": route_multi_object_sparse,
        "route_ready_for_multi_object_field": route_ready,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "diagnostic_gt_used_for_thresholds": False,
        "row_counts": {
            "object_seed_rows": len(object_seed_rows),
            "edge_diagnostic_rows": len(edge_rows),
            "d4rt_witness_quality_rows": len(d4rt_quality_rows),
            "semantic_region_quality_rows": len(semantic_rows),
            "route_rows": len(route_rows),
        },
        "runtime_sec": time.time() - started,
    }

    _write_csv(OUT / "object_seed_rows.csv", object_seed_rows)
    _write_csv(OUT / "edge_diagnostic_rows.csv", edge_rows)
    _write_csv(OUT / "d4rt_witness_quality_rows.csv", d4rt_quality_rows)
    _write_csv(OUT / "semantic_region_quality_rows.csv", semantic_rows)
    _write_csv(OUT / "route_rows.csv", route_rows)
    _write_json(OUT / "summary.json", summary)

    output_files = [
        OUT / "summary.json",
        OUT / "object_seed_rows.csv",
        OUT / "edge_diagnostic_rows.csv",
        OUT / "d4rt_witness_quality_rows.csv",
        OUT / "semantic_region_quality_rows.csv",
        OUT / "route_rows.csv",
    ]
    _write_json(OUT / "SHA256SUMS.json", {path.name: _sha256(path) for path in output_files})
    return summary


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
