#!/usr/bin/env python3
"""Audit whether v94 can safely exercise true multi-object competition."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PHASE_ID = "v94_phase7_competition_registry_audit"
RUN_ID = "v94_phase7_competition_registry_audit"
OUT = ROOT / "outputs/audit/v94_phase7_competition_registry_audit"

PHASE1 = ROOT / "outputs/audit/v94_phase1_canonical_graph"
V93_FIELD = ROOT / "outputs/audit/v93_phase5_boundary_affinity_field"


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


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
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fieldnames})


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def _mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else 0.0


def _p90(values: list[float]) -> float:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return 0.0
    idx = min(len(finite) - 1, int(math.ceil(0.9 * len(finite))) - 1)
    return float(finite[idx])


def _parse_int(value: Any, default: int = -1) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except Exception:
        return default


def _phase1_variant_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("variant_id", "")),
        str(row.get("scene_id", "")),
        str(row.get("window_id", "")),
        str(row.get("frame_id", "")),
        str(row.get("source_mask_id", "")),
    )


def _phase1_plan_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("scene_id", "")),
        str(row.get("window_id", "")),
        str(row.get("frame_id", "")),
        str(row.get("source_mask_id", "")),
    )


def _field_source_key(scene: str, frame_id: str, source_mask_id: str) -> str:
    return f"{scene}|{_parse_int(frame_id)}|{_parse_int(source_mask_id)}"


def _short_join(values: set[str], limit: int = 8) -> str:
    ordered = sorted(str(value) for value in values if str(value) != "")
    if len(ordered) <= limit:
        return ";".join(ordered)
    return ";".join(ordered[:limit]) + f";...(+{len(ordered) - limit})"


_CLUSTER_RE = re.compile(r"(c\d+:cluster\d+)")


def _canonical_object_key(object_id: str, meta: dict[str, str] | None, scene_id: str) -> tuple[str, str]:
    if meta:
        local_cluster = str(meta.get("local_cluster_id", ""))
        if local_cluster:
            return "local_cluster", f"{scene_id}|{local_cluster}"
        history_id = str(meta.get("history_id", ""))
        if history_id:
            return "history", f"{scene_id}|{history_id}"
    match = _CLUSTER_RE.search(str(object_id))
    if match:
        return "parsed_cluster", f"{scene_id}|V82_local:{match.group(1)}"
    return "raw_object_id", f"{scene_id}|{object_id}"


def _load_object_meta(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            object_id = str(row.get("object_hypothesis_id", ""))
            if object_id:
                out[object_id] = dict(row)
    return out


def _load_container_registry(path: Path) -> tuple[dict[tuple[str, str, str, str], dict[str, Any]], dict[str, set[tuple[str, str, str, str]]]]:
    plan_groups: dict[tuple[str, str, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "variant_ids": set(),
            "source_variants": set(),
            "mask_paths": set(),
            "mask_areas": set(),
            "bbox_signatures": set(),
            "container_rows": 0,
        }
    )
    field_key_to_plan_keys: dict[str, set[tuple[str, str, str, str]]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            plan_key = _phase1_plan_key(row)
            item = plan_groups[plan_key]
            item["variant_ids"].add(str(row.get("variant_id", "")))
            item["source_variants"].add(str(row.get("source_variant", "")))
            item["mask_paths"].add(str(row.get("mask_path", "")))
            item["mask_areas"].add(str(row.get("mask_area_px", "")))
            item["bbox_signatures"].add(
                "|".join(
                    [
                        str(row.get("mask_bbox_x0", "")),
                        str(row.get("mask_bbox_y0", "")),
                        str(row.get("mask_bbox_x1", "")),
                        str(row.get("mask_bbox_y1", "")),
                    ]
                )
            )
            item["container_rows"] += 1
            field_key_to_plan_keys[_field_source_key(row.get("scene_id", ""), row.get("frame_id", ""), row.get("source_mask_id", ""))].add(plan_key)
    return dict(plan_groups), dict(field_key_to_plan_keys)


def _load_links(
    path: Path,
    object_meta: dict[str, dict[str, str]],
) -> tuple[
    dict[tuple[str, str, str, str], dict[str, Any]],
    Counter[tuple[str, str, str, str, str]],
    list[dict[str, Any]],
    set[str],
]:
    plan_groups: dict[tuple[str, str, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "raw_objects": set(),
            "canonical_objects": set(),
            "canonical_modes": Counter(),
            "canonical_to_raw": defaultdict(set),
            "canonical_to_variants": defaultdict(set),
            "canonical_to_source_variants": defaultdict(set),
            "variant_ids": set(),
            "source_variants": set(),
            "local_clusters": set(),
            "history_ids": set(),
            "link_count": 0,
        }
    )
    variant_key_counts: Counter[tuple[str, str, str, str, str]] = Counter()
    canonical_rows: list[dict[str, Any]] = []
    all_canonical_objects: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            object_id = str(row.get("object_hypothesis_id", ""))
            meta = object_meta.get(object_id)
            scene_id = str(row.get("scene_id", ""))
            mode, canonical_key = _canonical_object_key(object_id, meta, scene_id)
            plan_key = _phase1_plan_key(row)
            variant_key = _phase1_variant_key(row)
            source_variant = str((meta or {}).get("source_variant", row.get("variant_id", "")))
            local_cluster = str((meta or {}).get("local_cluster_id", ""))
            history_id = str((meta or {}).get("history_id", ""))
            item = plan_groups[plan_key]
            item["raw_objects"].add(object_id)
            item["canonical_objects"].add(canonical_key)
            item["canonical_modes"][mode] += 1
            item["canonical_to_raw"][canonical_key].add(object_id)
            item["canonical_to_variants"][canonical_key].add(str(row.get("variant_id", "")))
            item["canonical_to_source_variants"][canonical_key].add(source_variant)
            item["variant_ids"].add(str(row.get("variant_id", "")))
            item["source_variants"].add(source_variant)
            if local_cluster:
                item["local_clusters"].add(local_cluster)
            if history_id:
                item["history_ids"].add(history_id)
            item["link_count"] += 1
            variant_key_counts[variant_key] += 1
            all_canonical_objects.add(canonical_key)
            canonical_rows.append(
                {
                    "schema_version": "stream4d_v94_phase7_canonical_object_link_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "variant_id": row.get("variant_id", ""),
                    "scene_id": scene_id,
                    "window_id": row.get("window_id", ""),
                    "frame_id": row.get("frame_id", ""),
                    "source_mask_id": row.get("source_mask_id", ""),
                    "raw_object_hypothesis_id": object_id,
                    "canonical_object_mode": mode,
                    "canonical_object_key": canonical_key,
                    "source_variant": source_variant,
                    "local_cluster_id": local_cluster,
                    "history_id": history_id,
                    "carrier_support_mass": row.get("carrier_support_mass", ""),
                    "mask_selected_score": row.get("mask_selected_score", row.get("adapter_score_raw", "")),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    return dict(plan_groups), variant_key_counts, canonical_rows, all_canonical_objects


def _load_field_shard_coverage(
    shard_root: Path,
    object_meta: dict[str, dict[str, str]],
    field_key_to_plan_keys: dict[str, set[tuple[str, str, str, str]]],
) -> tuple[dict[tuple[str, str, str, str], set[str]], dict[str, Any], list[dict[str, Any]]]:
    shard_paths = sorted(shard_root.glob("field_shard_*.npz"))
    field_objects_by_plan_key: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    field_rows: list[dict[str, Any]] = []
    direct_object_ids: set[str] = set()
    canonical_objects: set[str] = set()
    source_keys_seen: set[str] = set()
    ambiguous_source_key_count = 0
    unmatched_source_key_count = 0
    direct_match_count = 0
    parsed_cluster_match_count = 0
    source_record_count = 0

    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as data:
            source_keys = [str(value) for value in data["source_keys"].tolist()]
            object_ids = [str(value) for value in data["source_object_ids"].tolist()]
            for source_key, object_id in zip(source_keys, object_ids, strict=False):
                source_record_count += 1
                source_keys_seen.add(source_key)
                scene_id = source_key.split("|", 1)[0] if "|" in source_key else ""
                meta = object_meta.get(object_id)
                mode, canonical_key = _canonical_object_key(object_id, meta, scene_id)
                direct_objects = object_id in object_meta
                direct_match_count += int(direct_objects)
                parsed_cluster_match_count += int(mode == "parsed_cluster")
                direct_object_ids.add(object_id)
                canonical_objects.add(canonical_key)
                plan_keys = field_key_to_plan_keys.get(source_key, set())
                if not plan_keys:
                    unmatched_source_key_count += 1
                if len(plan_keys) > 1:
                    ambiguous_source_key_count += 1
                for plan_key in plan_keys:
                    field_objects_by_plan_key[plan_key].add(canonical_key)
                field_rows.append(
                    {
                        "schema_version": "stream4d_v94_phase7_field_shard_object_coverage_v1",
                        "phase_id": PHASE_ID,
                        "run_id": RUN_ID,
                        "field_shard": _rel(shard_path),
                        "field_source_key": source_key,
                        "raw_field_source_object_id": object_id,
                        "direct_phase1_object_id_match": direct_objects,
                        "canonical_object_mode": mode,
                        "canonical_object_key": canonical_key,
                        "matched_plan_key_count": len(plan_keys),
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )

    coverage = {
        "field_shard_count": len(shard_paths),
        "field_source_record_count": source_record_count,
        "field_unique_source_key_count": len(source_keys_seen),
        "field_unique_raw_object_id_count": len(direct_object_ids),
        "field_unique_canonical_object_count": len(canonical_objects),
        "field_direct_phase1_object_id_match_count": direct_match_count,
        "field_direct_phase1_object_id_match_rate": float(direct_match_count / source_record_count) if source_record_count else 0.0,
        "field_parsed_cluster_fallback_count": parsed_cluster_match_count,
        "field_unmatched_source_key_count": unmatched_source_key_count,
        "field_ambiguous_window_source_key_count": ambiguous_source_key_count,
    }
    return dict(field_objects_by_plan_key), coverage, field_rows


def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    created_at = _created_at()
    phase1 = _read_json(PHASE1 / "summary.json")
    field_summary = _read_json(V93_FIELD / "summary.json")

    object_meta = _load_object_meta(PHASE1 / "object_hypothesis_rows.csv")
    container_groups, field_key_to_plan_keys = _load_container_registry(PHASE1 / "container_rows.csv")
    link_groups, variant_key_counts, canonical_link_rows, all_canonical_objects = _load_links(PHASE1 / "container_object_link_rows.csv", object_meta)
    field_objects_by_plan_key, field_coverage, field_coverage_rows = _load_field_shard_coverage(
        V93_FIELD / "field_shards",
        object_meta,
        field_key_to_plan_keys,
    )

    plan_key_rows: list[dict[str, Any]] = []
    casebook_rows: list[dict[str, Any]] = []
    raw_counts: list[float] = []
    canonical_counts: list[float] = []
    variant_counts: list[float] = []
    field_counts: list[float] = []
    raw_multi_count = 0
    canonical_multi_count = 0
    variant_collapse_count = 0
    field_multi_count = 0
    canonical_multi_with_field_multi = 0
    duplicate_drop_total = 0
    raw_unique_total = 0
    canonical_total = 0
    same_physical_mask_count = 0

    all_plan_keys = set(container_groups) | set(link_groups)
    for plan_key in sorted(all_plan_keys):
        container = container_groups.get(plan_key, {})
        links = link_groups.get(plan_key, {})
        raw_objects = set(links.get("raw_objects", set()))
        canonical_objects = set(links.get("canonical_objects", set()))
        link_variants = set(links.get("variant_ids", set()))
        source_variants = set(links.get("source_variants", set()))
        container_variants = set(container.get("variant_ids", set()))
        field_objects = set(field_objects_by_plan_key.get(plan_key, set()))
        raw_count = len(raw_objects)
        canonical_count = len(canonical_objects)
        variant_count = len(container_variants or link_variants)
        field_count = len(field_objects)
        duplicate_drop = max(0, raw_count - canonical_count)
        raw_unique_total += raw_count
        canonical_total += canonical_count
        duplicate_drop_total += duplicate_drop
        raw_multi = raw_count > 1
        canonical_multi = canonical_count > 1
        variant_collapse = variant_count > 1 or len(link_variants) > 1 or len(source_variants) > 1
        field_multi = field_count > 1
        raw_multi_count += int(raw_multi)
        canonical_multi_count += int(canonical_multi)
        variant_collapse_count += int(variant_collapse)
        field_multi_count += int(field_multi)
        canonical_multi_with_field_multi += int(canonical_multi and field_multi)
        mask_path_count = len(set(container.get("mask_paths", set())))
        mask_area_count = len(set(container.get("mask_areas", set())))
        bbox_count = len(set(container.get("bbox_signatures", set())))
        same_physical_mask = mask_path_count <= 1 and mask_area_count <= 1 and bbox_count <= 1
        same_physical_mask_count += int(same_physical_mask)
        raw_counts.append(float(raw_count))
        canonical_counts.append(float(canonical_count))
        variant_counts.append(float(variant_count))
        field_counts.append(float(field_count))
        row = {
            "schema_version": "stream4d_v94_phase7_registry_plan_key_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "scene_id": plan_key[0],
            "window_id": plan_key[1],
            "frame_id": plan_key[2],
            "source_mask_id": plan_key[3],
            "container_variant_count": len(container_variants),
            "link_variant_count": len(link_variants),
            "source_variant_count": len(source_variants),
            "raw_object_count": raw_count,
            "canonical_object_count": canonical_count,
            "field_canonical_object_count": field_count,
            "raw_multi_object": raw_multi,
            "canonical_multi_object": canonical_multi,
            "field_multi_object_supported": field_multi,
            "variant_collapse": variant_collapse,
            "duplicate_drop_count": duplicate_drop,
            "canonical_per_raw_ratio": float(canonical_count / raw_count) if raw_count else 0.0,
            "same_physical_mask_signature": same_physical_mask,
            "mask_path_count": mask_path_count,
            "mask_area_count": mask_area_count,
            "bbox_signature_count": bbox_count,
            "sample_canonical_object_keys": _short_join(canonical_objects, 5),
            "sample_field_canonical_object_keys": _short_join(field_objects, 5),
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        plan_key_rows.append(row)
        if raw_multi or canonical_multi or variant_collapse:
            casebook_rows.append(row)

    casebook_rows.sort(
        key=lambda row: (
            int(row.get("canonical_object_count", 0)),
            int(row.get("raw_object_count", 0)),
            int(row.get("source_variant_count", 0)),
        ),
        reverse=True,
    )
    casebook_rows = casebook_rows[:500]

    variant_values = list(variant_key_counts.values())
    plan_count = len(plan_key_rows)
    canonical_multi_rate = float(canonical_multi_count / plan_count) if plan_count else 0.0
    raw_multi_rate = float(raw_multi_count / plan_count) if plan_count else 0.0
    variant_key_multi_rate = float(sum(1 for value in variant_values if value > 1) / len(variant_values)) if variant_values else 0.0
    variant_collapse_rate = float(variant_collapse_count / plan_count) if plan_count else 0.0
    duplicate_drop_rate = float(duplicate_drop_total / raw_unique_total) if raw_unique_total else 0.0
    field_multi_rate = float(field_multi_count / plan_count) if plan_count else 0.0
    canonical_multi_field_supported_rate = float(canonical_multi_with_field_multi / canonical_multi_count) if canonical_multi_count else 0.0
    same_physical_mask_rate = float(same_physical_mask_count / plan_count) if plan_count else 0.0

    registry_has_real_multi_candidates = canonical_multi_rate >= 0.20
    current_field_exercises_multi_object = field_multi_rate >= 0.05
    duplicate_collapse_severe = duplicate_drop_rate >= 0.50
    field_object_axis_missing = registry_has_real_multi_candidates and not current_field_exercises_multi_object
    competition_registry_gate_pass = bool(registry_has_real_multi_candidates and current_field_exercises_multi_object and not duplicate_collapse_severe)
    if field_object_axis_missing:
        decision = "NO_GO_V94_COMPETITION_REGISTRY_FIELD_REBUILD_REQUIRED"
        blocker = "object_specific_field_axis_missing_despite_canonical_multi_object_registry"
        repair = "Rebuild source-object registry/field shards with object-specific unary/prototype rows before materializing multi-object competition."
    elif not registry_has_real_multi_candidates:
        decision = "NO_GO_V94_COMPETITION_REGISTRY_CANONICAL_MULTI_OBJECT_SPARSE"
        blocker = "canonical_multi_object_rate_sparse_after_dedup"
        repair = "Rebuild or import a denser object registry before further competition readout work."
    elif duplicate_collapse_severe:
        decision = "NO_GO_V94_COMPETITION_REGISTRY_DUPLICATE_COLLAPSE_SEVERE"
        blocker = "variant_collapse_duplicates_dominate_registry"
        repair = "Deduplicate source variants into stable object identities and audit object-specific evidence before materialization."
    else:
        decision = "PASS_V94_COMPETITION_REGISTRY_REPAIR_READY"
        blocker = "competition_registry_ready_for_object_specific_materialization"
        repair = "Proceed to object-specific multi-label materialization with the audited canonical registry."

    _write_csv(OUT / "registry_plan_key_rows.csv", plan_key_rows)
    _write_csv(OUT / "canonical_object_link_rows.csv", canonical_link_rows)
    _write_csv(OUT / "field_shard_object_coverage_rows.csv", field_coverage_rows)
    _write_csv(OUT / "competition_casebook_rows.csv", casebook_rows)

    summary = {
        "schema": "stream4d_v94_phase7_competition_registry_audit_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": created_at,
        "decision": decision,
        "competition_registry_gate_pass": competition_registry_gate_pass,
        "competition_blocker": blocker,
        "recommended_repair_direction": repair,
        "safe_to_materialize_current_v94": competition_registry_gate_pass,
        "materialization_skip_reason": "" if competition_registry_gate_pass else repair,
        "phase1_decision": phase1.get("decision", ""),
        "phase1_multi_object_container_rate_variant_key": phase1.get("multi_object_container_rate_variant_key", ""),
        "phase1_multi_object_container_rate_plan_key": phase1.get("multi_object_container_rate_plan_key", ""),
        "v93_field_decision": field_summary.get("decision", ""),
        "v93_field_multi_label_exercised": field_summary.get("multi_label_exercised", ""),
        "v93_field_multi_object_source_count": field_summary.get("multi_object_source_count", ""),
        "v93_field_solver_backend_actual": field_summary.get("solver_backend_actual", ""),
        "v93_field_triton_available": field_summary.get("triton_available", ""),
        "v93_field_gpu_device_source_counts": field_summary.get("gpu_device_source_counts", {}),
        "plan_key_count": plan_count,
        "variant_key_count": len(variant_values),
        "raw_multi_object_plan_key_count": raw_multi_count,
        "raw_multi_object_rate_plan_key": raw_multi_rate,
        "canonical_multi_object_plan_key_count": canonical_multi_count,
        "canonical_multi_object_rate_plan_key": canonical_multi_rate,
        "variant_key_multi_object_rate": variant_key_multi_rate,
        "variant_collapse_plan_key_count": variant_collapse_count,
        "variant_collapse_plan_key_rate": variant_collapse_rate,
        "same_physical_mask_signature_rate": same_physical_mask_rate,
        "raw_object_per_plan_key_mean": _mean(raw_counts),
        "raw_object_per_plan_key_p90": _p90(raw_counts),
        "canonical_object_per_plan_key_mean": _mean(canonical_counts),
        "canonical_object_per_plan_key_p90": _p90(canonical_counts),
        "source_variant_per_plan_key_mean": _mean(variant_counts),
        "source_variant_per_plan_key_p90": _p90(variant_counts),
        "raw_unique_object_occurrence_total": raw_unique_total,
        "canonical_object_occurrence_total": canonical_total,
        "phase1_unique_canonical_object_count": len(all_canonical_objects),
        "duplicate_drop_total": duplicate_drop_total,
        "duplicate_drop_rate_raw_to_canonical": duplicate_drop_rate,
        "field_multi_object_plan_key_count": field_multi_count,
        "field_multi_object_plan_key_rate": field_multi_rate,
        "canonical_multi_plan_key_with_field_multi_count": canonical_multi_with_field_multi,
        "canonical_multi_plan_key_with_field_multi_rate": canonical_multi_field_supported_rate,
        "field_object_per_plan_key_mean": _mean(field_counts),
        "field_object_per_plan_key_p90": _p90(field_counts),
        **field_coverage,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "row_counts": {
            "registry_plan_key_rows": len(plan_key_rows),
            "canonical_object_link_rows": len(canonical_link_rows),
            "field_shard_object_coverage_rows": len(field_coverage_rows),
            "competition_casebook_rows": len(casebook_rows),
        },
        "input_artifacts": {
            _rel(path): _sha256(path)
            for path in [
                PHASE1 / "container_rows.csv",
                PHASE1 / "container_object_link_rows.csv",
                PHASE1 / "object_hypothesis_rows.csv",
                V93_FIELD / "summary.json",
                V93_FIELD / "field_artifact_manifest.json",
            ]
            if path.exists()
        },
    }
    _write_json(OUT / "summary.json", summary)
    outputs = [
        OUT / "registry_plan_key_rows.csv",
        OUT / "canonical_object_link_rows.csv",
        OUT / "field_shard_object_coverage_rows.csv",
        OUT / "competition_casebook_rows.csv",
        OUT / "summary.json",
    ]
    _write_json(OUT / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run()
