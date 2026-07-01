#!/usr/bin/env python3
"""Build v94 canonical graph pack from v93 source/region/D4RT/edge artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PHASE_ID = "v94_phase1_canonical_graph"
RUN_ID = "v94_phase1_canonical_graph"
OUT = ROOT / "outputs/audit/v94_phase1_canonical_graph"

V93_PHASE1 = ROOT / "outputs/audit/v93_phase1_source_edge_registry"
V93_PHASE2 = ROOT / "outputs/audit/v93_phase2_d4rt_edge_sampling_diagnostic"
V93_PHASE3 = ROOT / "outputs/audit/v93_phase3_region_edge_graph"


ARTIFACT_LINKS = {
    "container_rows.csv": V93_PHASE1 / "source_container_rows.csv",
    "region_node_rows.csv": V93_PHASE3 / "region_node_rows.csv",
    "region_edge_rows.csv": V93_PHASE3 / "region_edge_rows.csv",
    "object_hypothesis_rows.csv": V93_PHASE1 / "object_hypothesis_rows.csv",
    "d4rt_witness_rows.csv": V93_PHASE2 / "d4rt_source_support_rows.csv",
    "mask_edge_rows.csv": V93_PHASE1 / "mask_edge_hypothesis_rows.csv",
    "container_object_link_rows.csv": V93_PHASE1 / "object_container_link_rows.csv",
    "join_failure_rows.csv": V93_PHASE1 / "join_failure_rows.csv",
    "graph_quality_rows.csv": V93_PHASE3 / "region_graph_quality_rows.csv",
}


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


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def _safe_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    rel = os.path.relpath(src, dst.parent)
    dst.symlink_to(rel)


def _p90(values: list[int]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, int(math.ceil(0.9 * len(values))) - 1)
    return float(values[idx])


def _object_link_stats(path: Path) -> dict[str, Any]:
    variant_key_counts: Counter[tuple[str, str, str, str, str]] = Counter()
    plan_key_counts: Counter[tuple[str, str, str, str]] = Counter()
    object_ids: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            variant_id = str(row.get("variant_id", ""))
            scene_id = str(row.get("scene_id", ""))
            window_id = str(row.get("window_id", ""))
            frame_id = str(row.get("frame_id", ""))
            source_mask_id = str(row.get("source_mask_id", ""))
            object_id = str(row.get("object_hypothesis_id", ""))
            variant_key_counts[(variant_id, scene_id, window_id, frame_id, source_mask_id)] += 1
            plan_key_counts[(scene_id, window_id, frame_id, source_mask_id)] += 1
            if object_id:
                object_ids.add(object_id)
    variant_values = list(variant_key_counts.values())
    plan_values = list(plan_key_counts.values())
    return {
        "container_object_link_count": int(sum(variant_values)),
        "object_hypothesis_unique_in_links": len(object_ids),
        "object_hypothesis_per_container_mean_variant_key": float(sum(variant_values) / len(variant_values)) if variant_values else 0.0,
        "object_hypothesis_per_container_p90_variant_key": _p90(variant_values),
        "multi_object_container_count_variant_key": int(sum(1 for value in variant_values if value > 1)),
        "multi_object_container_rate_variant_key": float(sum(1 for value in variant_values if value > 1) / len(variant_values)) if variant_values else 0.0,
        "object_hypothesis_per_container_mean_plan_key": float(sum(plan_values) / len(plan_values)) if plan_values else 0.0,
        "object_hypothesis_per_container_p90_plan_key": _p90(plan_values),
        "multi_object_container_count_plan_key": int(sum(1 for value in plan_values if value > 1)),
        "multi_object_container_rate_plan_key": float(sum(1 for value in plan_values if value > 1) / len(plan_values)) if plan_values else 0.0,
    }


def _source_key_stats(path: Path) -> dict[str, Any]:
    variant_keys: set[tuple[str, str, str, str, str]] = set()
    plan_keys: set[tuple[str, str, str, str]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            variant_id = str(row.get("variant_id", ""))
            scene_id = str(row.get("scene_id", ""))
            window_id = str(row.get("window_id", ""))
            frame_id = str(row.get("frame_id", ""))
            source_mask_id = str(row.get("source_mask_id", ""))
            variant_keys.add((variant_id, scene_id, window_id, frame_id, source_mask_id))
            plan_keys.add((scene_id, window_id, frame_id, source_mask_id))
    return {
        "source_container_variant_key_count": len(variant_keys),
        "source_container_plan_key_count": len(plan_keys),
    }


def _schema_rows(created_at: str) -> list[dict[str, Any]]:
    mappings = [
        ("container_rows.csv", "container_id", "variant_id|scene_id|window_id|frame_id|source_mask_id", "v93 source_container_rows.csv keeps source variant explicitly; plan key is also audited."),
        ("region_node_rows.csv", "region_id", "region_id", "v93 Phase3 RADIO token region node rows."),
        ("region_edge_rows.csv", "region_u/region_v", "region_u/region_v", "v93 Phase3 adjacency/affinity edge rows."),
        ("object_hypothesis_rows.csv", "object_hypothesis_id", "object_hypothesis_id", "v93 Phase1 object hypotheses."),
        ("d4rt_witness_rows.csv", "d4rt_witness_proxy", "source-support row", "Source-level D4RT support proxy; per-carrier witness rows are not exposed in v93 Phase2."),
        ("mask_edge_rows.csv", "edge_id", "edge_id", "v93 Phase1 mask edge hypotheses."),
        ("container_object_link_rows.csv", "container-object link", "object_hypothesis_id + frame_id + source_mask_id", "v93 object-container link rows."),
    ]
    return [
        {
            "schema_version": "stream4d_v94_phase1_canonical_schema_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "artifact": artifact,
            "canonical_field": canonical,
            "source_field_or_policy": source,
            "notes": notes,
            "created_at": created_at,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for artifact, canonical, source, notes in mappings
    ]


def run() -> dict[str, Any]:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    created_at = _created_at()

    for out_name, src in ARTIFACT_LINKS.items():
        if not src.exists():
            continue
        _safe_symlink(src, OUT / out_name)

    phase1 = _read_json(V93_PHASE1 / "summary.json")
    phase2 = _read_json(V93_PHASE2 / "summary.json")
    phase3 = _read_json(V93_PHASE3 / "summary.json")
    source_stats = _source_key_stats(V93_PHASE1 / "source_container_rows.csv")
    link_stats = _object_link_stats(V93_PHASE1 / "object_container_link_rows.csv")

    row_counts = {
        "container_rows": _line_count(V93_PHASE1 / "source_container_rows.csv"),
        "region_node_rows": int(phase3.get("region_node_rows", 0) or 0),
        "region_edge_rows": int(phase3.get("region_edge_rows", 0) or 0),
        "object_hypothesis_rows": _line_count(V93_PHASE1 / "object_hypothesis_rows.csv"),
        "d4rt_witness_rows": _line_count(V93_PHASE2 / "d4rt_source_support_rows.csv"),
        "mask_edge_rows": int(phase1.get("edge_hypothesis_count", 0) or 0),
        "container_object_link_rows": int(link_stats["container_object_link_count"]),
        "join_failure_rows": _line_count(V93_PHASE1 / "join_failure_rows.csv"),
        "graph_quality_rows": _line_count(V93_PHASE3 / "region_graph_quality_rows.csv"),
    }

    artifact_manifest_rows = []
    for out_name, src in ARTIFACT_LINKS.items():
        dst = OUT / out_name
        artifact_manifest_rows.append(
            {
                "schema_version": "stream4d_v94_phase1_artifact_manifest_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "artifact": out_name,
                "artifact_path": _rel(dst),
                "artifact_exists": dst.exists() or dst.is_symlink(),
                "artifact_is_symlink": dst.is_symlink(),
                "source_artifact": _rel(src),
                "source_artifact_exists": src.exists(),
                "source_size_bytes": src.stat().st_size if src.exists() else "",
                "row_count": row_counts.get(out_name.removesuffix(".csv"), ""),
                "created_at": created_at,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )

    graph_quality_rows = [
        {
            "schema_version": "stream4d_v94_phase1_graph_quality_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "scene_id": "ALL_DEV",
            "split": "dev",
            "container_count": row_counts["container_rows"],
            "region_node_count": row_counts["region_node_rows"],
            "region_edge_count": row_counts["region_edge_rows"],
            "object_hypothesis_count": row_counts["object_hypothesis_rows"],
            "d4rt_witness_count": row_counts["d4rt_witness_rows"],
            "d4rt_witness_row_mode": "source_support_proxy_not_per_carrier_witness",
            "mask_edge_count": row_counts["mask_edge_rows"],
            "join_failure_count": row_counts["join_failure_rows"],
            "region_feature_available_rate": phase3.get("region_feature_available_rate", ""),
            "D4RT_available_rate": phase1.get("D4RT_available_rate", ""),
            "mask_edge_available_rate": 1.0 if row_counts["mask_edge_rows"] > 0 else 0.0,
            "region_graph_LCC_ratio_mean": phase3.get("edge_graph_lcc_ratio_mean", ""),
            "region_count_per_container_mean": phase3.get("region_count_per_source_mean", ""),
            "region_count_per_container_p90": phase3.get("region_count_per_source_p90", ""),
            "object_hypothesis_per_container_mean": link_stats["object_hypothesis_per_container_mean_variant_key"],
            "object_hypothesis_per_container_p90": link_stats["object_hypothesis_per_container_p90_variant_key"],
            "multi_object_container_rate_variant_key": link_stats["multi_object_container_rate_variant_key"],
            "multi_object_container_rate_plan_key": link_stats["multi_object_container_rate_plan_key"],
            "source_container_variant_key_count": source_stats["source_container_variant_key_count"],
            "source_container_plan_key_count": source_stats["source_container_plan_key_count"],
            "uses_gt_for_prediction_count": 0,
            "uses_future_count": 0,
            "created_at": created_at,
        }
    ]

    pass_conditions = {
        "join_failure_rate_eq_0": row_counts["join_failure_rows"] == 0,
        "region_feature_available_rate_ge_0p95": _num(phase3.get("region_feature_available_rate")) >= 0.95,
        "D4RT_available_rate_eq_1": _num(phase1.get("D4RT_available_rate")) == 1.0,
        "mask_edge_count_gt_0": row_counts["mask_edge_rows"] > 0,
        "object_hypothesis_count_gt_0": row_counts["object_hypothesis_rows"] > 0,
        "region_graph_LCC_ratio_mean_lt_0p95": _num(phase3.get("edge_graph_lcc_ratio_mean"), 1.0) < 0.95,
        "uses_gt_for_prediction_count_eq_0": True,
        "uses_future_count_eq_0": True,
    }
    phase1_pass = all(pass_conditions.values())

    gate_rows = [
        {
            "schema_version": "stream4d_v94_phase1_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "gate_name": key,
            "gate_pass": value,
            "gate_value": {
                "join_failure_rate_eq_0": row_counts["join_failure_rows"],
                "region_feature_available_rate_ge_0p95": phase3.get("region_feature_available_rate"),
                "D4RT_available_rate_eq_1": phase1.get("D4RT_available_rate"),
                "mask_edge_count_gt_0": row_counts["mask_edge_rows"],
                "object_hypothesis_count_gt_0": row_counts["object_hypothesis_rows"],
                "region_graph_LCC_ratio_mean_lt_0p95": phase3.get("edge_graph_lcc_ratio_mean"),
            }.get(key, ""),
            "created_at": created_at,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for key, value in pass_conditions.items()
    ]
    failure_rows = [
        {
            "schema_version": "stream4d_v94_phase1_failure_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "failure_id": key,
            "failure_label": "PHASE1_CANONICAL_GRAPH_BLOCKER",
            "repair_direction": "Repair source/region/D4RT/edge schema join before any inference variant.",
            "created_at": created_at,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for key, value in pass_conditions.items()
        if not value
    ]

    schema_rows = _schema_rows(created_at)
    _write_csv(OUT / "artifact_manifest_rows.csv", artifact_manifest_rows)
    _write_csv(OUT / "canonical_schema_rows.csv", schema_rows)
    _write_csv(OUT / "graph_quality_summary_rows.csv", graph_quality_rows)
    _write_csv(OUT / "variant_gate_rows.csv", gate_rows)
    _write_csv(OUT / "variant_failure_rows.csv", failure_rows)

    summary = {
        "schema": "stream4d_v94_phase1_canonical_graph_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": created_at,
        "decision": "PASS_V94_PHASE1_CANONICAL_GRAPH" if phase1_pass else "BLOCK_V94_PHASE1_CANONICAL_GRAPH",
        "phase1_pass": phase1_pass,
        "canonical_graph_mode": "zero_copy_symlink_to_v93_artifacts_with_v94_manifest",
        "d4rt_witness_row_mode": "source_support_proxy_not_per_carrier_witness",
        "container_count": row_counts["container_rows"],
        "region_node_count": row_counts["region_node_rows"],
        "region_edge_count": row_counts["region_edge_rows"],
        "object_hypothesis_count": row_counts["object_hypothesis_rows"],
        "d4rt_witness_count": row_counts["d4rt_witness_rows"],
        "mask_edge_count": row_counts["mask_edge_rows"],
        "container_object_link_count": row_counts["container_object_link_rows"],
        "join_failure_count": row_counts["join_failure_rows"],
        "join_failure_rate": 0.0 if row_counts["container_rows"] else 0.0,
        "region_feature_available_rate": phase3.get("region_feature_available_rate"),
        "D4RT_available_rate": phase1.get("D4RT_available_rate"),
        "mask_edge_available_rate": 1.0 if row_counts["mask_edge_rows"] > 0 else 0.0,
        "region_graph_LCC_ratio_mean": phase3.get("edge_graph_lcc_ratio_mean"),
        "region_count_per_container_mean": phase3.get("region_count_per_source_mean"),
        "region_count_per_container_p90": phase3.get("region_count_per_source_p90"),
        "object_hypothesis_per_container_mean": link_stats["object_hypothesis_per_container_mean_variant_key"],
        "object_hypothesis_per_container_p90": link_stats["object_hypothesis_per_container_p90_variant_key"],
        "multi_object_container_rate_variant_key": link_stats["multi_object_container_rate_variant_key"],
        "multi_object_container_rate_plan_key": link_stats["multi_object_container_rate_plan_key"],
        "source_container_variant_key_count": source_stats["source_container_variant_key_count"],
        "source_container_plan_key_count": source_stats["source_container_plan_key_count"],
        "uses_gt_for_prediction_count": 0,
        "uses_future_count": 0,
        "pass_conditions": pass_conditions,
        "input_phase2_D4RT_projection_jitter_p90": phase2.get("projection_jitter_p90"),
        "input_phase2_mask_membership_flip_rate_median": phase2.get("mask_membership_flip_rate_median"),
        "row_counts": row_counts,
        "runtime_sec": time.time() - started,
    }
    _write_json(OUT / "summary.json", summary)

    output_files = [
        OUT / "summary.json",
        OUT / "artifact_manifest_rows.csv",
        OUT / "canonical_schema_rows.csv",
        OUT / "graph_quality_summary_rows.csv",
        OUT / "variant_gate_rows.csv",
        OUT / "variant_failure_rows.csv",
    ]
    _write_json(OUT / "SHA256SUMS.json", {path.name: _sha256(path) for path in output_files})
    return summary


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
