#!/usr/bin/env python3
"""Build v95 physical source registry and true object axis."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/audit/v95_phase1_physical_source_registry"
PHASE_ID = "v95_phase1_physical_source_registry"
RUN_ID = "v95_phase1_physical_source_registry"

PHASE0 = ROOT / "outputs/audit/v95_phase0_fact_lock/summary.json"
V94_CANONICAL = ROOT / "outputs/audit/v94_phase1_canonical_graph"

CONTAINER_ROWS = V94_CANONICAL / "container_rows.csv"
OBJECT_ROWS = V94_CANONICAL / "object_hypothesis_rows.csv"
LINK_ROWS = V94_CANONICAL / "container_object_link_rows.csv"
REGION_NODE_ROWS = V94_CANONICAL / "region_node_rows.csv"
REGION_EDGE_ROWS = V94_CANONICAL / "region_edge_rows.csv"
D4RT_ROWS = V94_CANONICAL / "d4rt_witness_rows.csv"
MASK_EDGE_ROWS = V94_CANONICAL / "mask_edge_rows.csv"

CLUSTER_RE = re.compile(r"(c\d+:cluster\d+)")


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _safe_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(os.path.relpath(src, dst.parent))


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _int(value: Any, default: int = -1) -> int:
    try:
        if value in (None, ""):
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _percentile(values: list[float], q: float) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return 0.0
    idx = min(len(vals) - 1, int(math.ceil(float(q) / 100.0 * len(vals))) - 1)
    return float(vals[idx])


def _key(row: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(row.get("scene_id", "")),
        str(row.get("window_id", "")),
        _int(row.get("frame_id"), -1),
        _int(row.get("source_mask_id"), -1),
    )


def _variant_key(row: dict[str, Any]) -> tuple[str, str, str, int, int]:
    scene, window, frame_id, mask_id = _key(row)
    return (str(row.get("variant_id", row.get("source_variant", ""))), scene, window, frame_id, mask_id)


def _key_text(key: tuple[str, str, int, int]) -> str:
    scene, window, frame_id, mask_id = key
    return f"{scene}|{window}|{frame_id}|{mask_id}"


def _variant_key_text(key: tuple[str, str, str, int, int]) -> str:
    variant, scene, window, frame_id, mask_id = key
    return f"{variant}|{scene}|{window}|{frame_id}|{mask_id}"


def _canonical_object(object_id: str, meta: dict[str, str] | None, scene: str) -> tuple[str, str]:
    if meta:
        local_cluster = str(meta.get("local_cluster_id", ""))
        if local_cluster:
            return "local_cluster", f"{scene}|{local_cluster}"
        history_id = str(meta.get("history_id", ""))
        if history_id:
            return "history", f"{scene}|{history_id}"
    match = CLUSTER_RE.search(str(object_id))
    if match:
        return "parsed_cluster", f"{scene}|V82_local:{match.group(1)}"
    return "raw_object_id", f"{scene}|{object_id}"


def _scan_container_rows() -> tuple[dict[tuple[str, str, int, int], dict[str, Any]], dict[tuple[str, str, int, int], set[str]], dict[str, int]]:
    best: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    variants: dict[tuple[str, str, int, int], set[str]] = defaultdict(set)
    counts = Counter()
    with CONTAINER_ROWS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("split", "dev")) != "dev":
                continue
            if _bool(row.get("uses_gt_for_prediction")) or _bool(row.get("uses_future")):
                counts["provenance_filtered_container_rows"] += 1
                continue
            key = _key(row)
            variant = str(row.get("variant_id", row.get("source_variant", "")))
            variants[key].add(variant)
            existing = best.get(key)
            if existing is None or variant == "B0_local_only":
                best[key] = dict(row)
            counts["container_rows"] += 1
    counts["physical_source_count_raw"] = len(best)
    counts["variant_source_count_raw"] = sum(len(v) for v in variants.values())
    return best, variants, dict(counts)


def _scan_count_by_key(path: Path, count_field: str | None = None) -> tuple[Counter[tuple[str, str, int, int]], int, int]:
    counts: Counter[tuple[str, str, int, int]] = Counter()
    rows = 0
    provenance_filtered = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("split", "dev")) != "dev":
                continue
            if _bool(row.get("uses_gt_for_prediction")) or _bool(row.get("uses_future")):
                provenance_filtered += 1
                continue
            rows += 1
            increment = 1
            if count_field:
                increment = max(0, int(_num(row.get(count_field), 0.0)))
            counts[_key(row)] += increment
    return counts, rows, provenance_filtered


def _load_object_meta() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with OBJECT_ROWS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            object_id = str(row.get("object_hypothesis_id", ""))
            if object_id:
                out[object_id] = dict(row)
    return out


def _scan_links(object_meta: dict[str, dict[str, str]]) -> tuple[dict[tuple[str, str, int, int], dict[str, dict[str, Any]]], dict[str, Any]]:
    grouped: dict[tuple[str, str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    raw_rows = 0
    provenance_filtered = 0
    raw_object_pairs: Counter[tuple[tuple[str, str, int, int], str]] = Counter()
    with LINK_ROWS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("split", "dev")) != "dev":
                continue
            if _bool(row.get("uses_gt_for_prediction")) or _bool(row.get("uses_future")):
                provenance_filtered += 1
                continue
            key = _key(row)
            scene = key[0]
            raw_object_id = str(row.get("object_hypothesis_id", ""))
            source_variant = str(row.get("variant_id", ""))
            mode, object_id = _canonical_object(raw_object_id, object_meta.get(raw_object_id), scene)
            meta = object_meta.get(raw_object_id, {})
            item = grouped[key].setdefault(
                object_id,
                {
                    "object_id": object_id,
                    "object_source_type": mode,
                    "raw_object_ids": set(),
                    "source_variant_ids": set(),
                    "local_slot_ids": set(),
                    "cluster_ids": set(),
                    "history_ids": set(),
                    "candidate_confidences": [],
                    "carrier_support_masses": [],
                    "risk_scores": [],
                    "semantic_descriptor_hashes": set(),
                    "appearance_descriptor_hashes": set(),
                },
            )
            item["raw_object_ids"].add(raw_object_id)
            item["source_variant_ids"].add(source_variant)
            local_cluster = str(meta.get("local_cluster_id", ""))
            if local_cluster:
                item["local_slot_ids"].add(local_cluster)
                item["cluster_ids"].add(local_cluster)
            history_id = str(meta.get("history_id", ""))
            if history_id:
                item["history_ids"].add(history_id)
            item["candidate_confidences"].append(_num(row.get("mask_selected_score"), _num(meta.get("object_score"))))
            item["carrier_support_masses"].append(_num(row.get("carrier_support_mass"), _num(meta.get("carrier_count"))))
            item["risk_scores"].append(_num(meta.get("risk_score"), _num(meta.get("hard_negative_density"))))
            if str(meta.get("semantic_proto", "")):
                item["semantic_descriptor_hashes"].add(str(meta.get("semantic_proto", "")))
            if str(meta.get("appearance_feature_hash", "")):
                item["appearance_descriptor_hashes"].add(str(meta.get("appearance_feature_hash", "")))
            raw_object_pairs[(key, raw_object_id)] += 1
            raw_rows += 1
    duplicate_candidate_rows = sum(1 for objects in grouped.values() for item in objects.values() if len(item["raw_object_ids"]) > 1 or len(item["source_variant_ids"]) > 1)
    object_counts = [len(objects) for objects in grouped.values()]
    stats = {
        "object_link_rows": raw_rows,
        "object_link_rows_provenance_filtered": provenance_filtered,
        "object_candidate_physical_source_count": len(grouped),
        "object_candidate_count": int(sum(object_counts)),
        "mean_object_count_per_physical_source": _mean([float(v) for v in object_counts]),
        "p90_object_count_per_physical_source": _percentile([float(v) for v in object_counts], 90),
        "multi_object_physical_source_rate": float(sum(1 for v in object_counts if v > 1) / len(object_counts)) if object_counts else 0.0,
        "single_object_physical_source_rate": float(sum(1 for v in object_counts if v == 1) / len(object_counts)) if object_counts else 0.0,
        "duplicate_candidate_count": duplicate_candidate_rows,
        "duplicate_candidate_rate": float(duplicate_candidate_rows / max(1, sum(object_counts))),
        "unsafe_bare_key_merge_count": int(sum(1 for (key, _raw), count in raw_object_pairs.items() if count > 1)),
    }
    return grouped, stats


def run() -> dict[str, Any]:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    created_at = _created_at()
    phase0 = _read_json(PHASE0)
    if phase0.get("decision") != "PASS_V95_PHASE0_FACT_LOCK":
        raise RuntimeError("v95 Phase0 must pass before Phase1")

    object_meta = _load_object_meta()
    container_best, container_variants, container_stats = _scan_container_rows()
    region_counts, region_rows, region_prov = _scan_count_by_key(REGION_NODE_ROWS)
    region_edge_counts, edge_rows, edge_prov = _scan_count_by_key(REGION_EDGE_ROWS)
    d4rt_counts, d4rt_rows, d4rt_prov = _scan_count_by_key(D4RT_ROWS, "carrier_count_inside_source")
    mask_edge_counts, mask_edge_rows, mask_edge_prov = _scan_count_by_key(MASK_EDGE_ROWS)
    object_grouped, object_stats = _scan_links(object_meta)

    raw_keys = set(container_best) | set(object_grouped) | set(region_counts) | set(d4rt_counts) | set(mask_edge_counts)
    method_keys = {
        key
        for key in raw_keys
        if key in container_best
        and key in object_grouped
        and region_counts.get(key, 0) > 0
        and d4rt_counts.get(key, 0) > 0
        and mask_edge_counts.get(key, 0) > 0
    }
    join_failure_rows: list[dict[str, Any]] = []
    for key in sorted(raw_keys):
        failures: list[str] = []
        if key not in container_best:
            failures.append("missing_source_container")
        if key not in object_grouped:
            failures.append("missing_object_candidates")
        if region_counts.get(key, 0) <= 0:
            failures.append("missing_region_nodes")
        if d4rt_counts.get(key, 0) <= 0:
            failures.append("missing_d4rt_witness_proxy")
        if mask_edge_counts.get(key, 0) <= 0:
            failures.append("missing_mask_edge_rows")
        if failures:
            scene, window, frame_id, mask_id = key
            join_failure_rows.append(
                {
                    "schema_version": "stream4d_v95_phase1_registry_join_failure_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "scene_id": scene,
                    "window_id": window,
                    "frame_id": frame_id,
                    "source_mask_id": mask_id,
                    "failure_types": "|".join(failures),
                    "repair_action": "excluded_from_method_safe_registry_universe; keep raw counts for audit",
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )

    source_rows: list[dict[str, Any]] = []
    object_candidate_rows: list[dict[str, Any]] = []
    registry_rows: list[dict[str, Any]] = []
    object_count_values: list[float] = []
    for key in sorted(method_keys):
        scene, window, frame_id, mask_id = key
        row = container_best[key]
        variants = sorted(container_variants.get(key, set()))
        objects = object_grouped.get(key, {})
        object_count_values.append(float(len(objects)))
        source_rows.append(
            {
                "schema_version": "stream4d_v95_phase1_source_container_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "physical_source_key": _key_text(key),
                "scene_id": scene,
                "split": row.get("split", "dev"),
                "window_id": window,
                "frame_id": frame_id,
                "source_mask_id": mask_id,
                "variant_source_key_count": len(variants),
                "variant_source_keys": "|".join(_variant_key_text((variant, scene, window, frame_id, mask_id)) for variant in variants),
                "mask_path": row.get("mask_path", ""),
                "mask_area_px": row.get("mask_area_px", ""),
                "image_area_px": row.get("image_area_px", ""),
                "mask_area_ratio": row.get("mask_area_ratio", ""),
                "has_d4rt_support": d4rt_counts.get(key, 0) > 0,
                "has_region_feature": region_counts.get(key, 0) > 0,
                "has_mask_edge": mask_edge_counts.get(key, 0) > 0,
                "region_count_in_source": region_counts.get(key, 0),
                "edge_count_in_source": region_edge_counts.get(key, 0),
                "object_candidate_count": len(objects),
                "method_source_allowed": True,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        for object_id, item in sorted(objects.items()):
            raw_ids = sorted(item["raw_object_ids"])
            source_variants = sorted(item["source_variant_ids"])
            is_duplicate = len(raw_ids) > 1 or len(source_variants) > 1
            base = {
                "schema_version": "stream4d_v95_phase1_object_candidate_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "scene_id": scene,
                "split": row.get("split", "dev"),
                "window_id": window,
                "frame_id": frame_id,
                "source_mask_id": mask_id,
                "physical_source_key": _key_text(key),
                "object_id": object_id,
                "object_source_type": item["object_source_type"],
                "local_slot_id": "|".join(sorted(item["local_slot_ids"])),
                "cluster_id": "|".join(sorted(item["cluster_ids"])),
                "history_id": "|".join(sorted(item["history_ids"])),
                "candidate_confidence": max(item["candidate_confidences"]) if item["candidate_confidences"] else 0.0,
                "candidate_ambiguity": 1.0 / max(1, len(source_variants)),
                "support_source_variant": "|".join(source_variants),
                "carrier_support_mass": max(item["carrier_support_masses"]) if item["carrier_support_masses"] else 0.0,
                "semantic_descriptor_hash": "|".join(sorted(item["semantic_descriptor_hashes"])),
                "appearance_descriptor_hash": "|".join(sorted(item["appearance_descriptor_hashes"])),
                "risk_score": max(item["risk_scores"]) if item["risk_scores"] else 0.0,
                "raw_object_ids": "|".join(raw_ids),
                "is_method_candidate": True,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
            object_candidate_rows.append(base)
            registry_rows.append(
                {
                    "schema_version": "stream4d_v95_phase1_source_object_registry_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    **{k: base[k] for k in ["scene_id", "window_id", "frame_id", "source_mask_id", "object_id", "object_source_type"]},
                    "source_variant_id": base["support_source_variant"],
                    "local_slot_id": base["local_slot_id"],
                    "cluster_id": base["cluster_id"],
                    "history_id": base["history_id"],
                    "candidate_confidence": base["candidate_confidence"],
                    "candidate_risk": base["risk_score"],
                    "carrier_support_mass": base["carrier_support_mass"],
                    "radio_feature_available": True,
                    "region_count_in_source": region_counts.get(key, 0),
                    "edge_count_in_source": region_edge_counts.get(key, 0),
                    "is_duplicate_candidate": is_duplicate,
                    "duplicate_of_object_id": "",
                    "method_candidate_allowed": True,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )

    raw_physical_count = len(raw_keys)
    method_physical_count = len(method_keys)
    raw_region_feature_rate = len([key for key in raw_keys if region_counts.get(key, 0) > 0]) / max(1, raw_physical_count)
    raw_d4rt_rate = len([key for key in raw_keys if d4rt_counts.get(key, 0) > 0]) / max(1, raw_physical_count)
    raw_mask_edge_rate = len([key for key in raw_keys if mask_edge_counts.get(key, 0) > 0]) / max(1, raw_physical_count)
    raw_join_failure_rate = len(join_failure_rows) / max(1, raw_physical_count)
    method_join_failure_rate = 0.0
    quality_rows = [
        {
            "schema_version": "stream4d_v95_phase1_registry_quality_summary_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "scope": "raw_dev_physical_sources",
            "physical_source_count": raw_physical_count,
            "region_feature_available_rate": raw_region_feature_rate,
            "D4RT_available_rate": raw_d4rt_rate,
            "mask_edge_available_rate": raw_mask_edge_rate,
            "join_failure_rate": raw_join_failure_rate,
            "join_failure_count": len(join_failure_rows),
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v95_phase1_registry_quality_summary_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "scope": "method_safe_registry_universe",
            "physical_source_count": method_physical_count,
            "region_feature_available_rate": 1.0 if method_physical_count else 0.0,
            "D4RT_available_rate": 1.0 if method_physical_count else 0.0,
            "mask_edge_available_rate": 1.0 if method_physical_count else 0.0,
            "join_failure_rate": method_join_failure_rate,
            "join_failure_count": 0,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
    ]
    multi_rate = float(sum(1 for value in object_count_values if value > 1) / max(1, len(object_count_values)))
    single_rate = float(sum(1 for value in object_count_values if value == 1) / max(1, len(object_count_values)))
    sparse_multi = multi_rate < 0.05
    phase1_pass = bool(
        method_join_failure_rate == 0.0
        and method_physical_count > 0
        and len(object_candidate_rows) > 0
        and raw_region_feature_rate >= 0.95
        and raw_d4rt_rate >= 0.95
        and raw_mask_edge_rate >= 0.95
    )

    for name, src in {
        "region_node_rows.csv": REGION_NODE_ROWS,
        "region_edge_rows.csv": REGION_EDGE_ROWS,
        "d4rt_witness_rows.csv": D4RT_ROWS,
        "mask_edge_rows.csv": MASK_EDGE_ROWS,
    }.items():
        _safe_symlink(src, OUT / name)

    summary = {
        "schema": "stream4d_v95_phase1_physical_source_registry_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": created_at,
        "decision": "PASS_V95_PHASE1_REGISTRY_READY" if phase1_pass else "NO_GO_V95_PHASE1_REGISTRY",
        "physical_source_count": method_physical_count,
        "raw_physical_source_count": raw_physical_count,
        "variant_source_count": sum(len(container_variants.get(key, set())) for key in method_keys),
        "raw_variant_source_count": container_stats.get("variant_source_count_raw", 0),
        "object_candidate_count": len(object_candidate_rows),
        "mean_object_count_per_physical_source": _mean(object_count_values),
        "p90_object_count_per_physical_source": _percentile(object_count_values, 90),
        "multi_object_physical_source_rate": multi_rate,
        "single_object_physical_source_rate": single_rate,
        "region_feature_available_rate": 1.0 if method_physical_count else 0.0,
        "D4RT_available_rate": 1.0 if method_physical_count else 0.0,
        "mask_edge_available_rate": 1.0 if method_physical_count else 0.0,
        "raw_region_feature_available_rate": raw_region_feature_rate,
        "raw_D4RT_available_rate": raw_d4rt_rate,
        "raw_mask_edge_available_rate": raw_mask_edge_rate,
        "join_failure_rate": method_join_failure_rate,
        "raw_join_failure_rate": raw_join_failure_rate,
        "registry_join_failure_count": len(join_failure_rows),
        "source_excluded_by_registry_repair_count": raw_physical_count - method_physical_count,
        "duplicate_candidate_count": object_stats.get("duplicate_candidate_count", 0),
        "duplicate_candidate_rate": object_stats.get("duplicate_candidate_rate", 0.0),
        "unsafe_bare_key_merge_count": object_stats.get("unsafe_bare_key_merge_count", 0),
        "MULTI_OBJECT_COMPETITION_SPARSE": sparse_multi,
        "input_row_counts": {
            **container_stats,
            "region_node_rows": region_rows,
            "region_edge_rows": edge_rows,
            "d4rt_rows": d4rt_rows,
            "mask_edge_rows": mask_edge_rows,
            "object_link_rows": object_stats.get("object_link_rows", 0),
            "provenance_filtered_rows": region_prov + edge_prov + d4rt_prov + mask_edge_prov + object_stats.get("object_link_rows_provenance_filtered", 0),
        },
        "uses_gt_for_prediction_count": 0,
        "uses_future_count": 0,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "duration_sec": time.time() - started,
        "row_counts": {
            "source_container_rows": len(source_rows),
            "object_candidate_rows": len(object_candidate_rows),
            "source_object_registry_rows": len(registry_rows),
            "registry_join_failure_rows": len(join_failure_rows),
            "registry_quality_summary_rows": len(quality_rows),
        },
    }

    _write_csv(OUT / "source_container_rows.csv", source_rows)
    _write_csv(OUT / "object_candidate_rows.csv", object_candidate_rows)
    _write_csv(OUT / "source_object_registry_rows.csv", registry_rows)
    _write_csv(OUT / "registry_join_failure_rows.csv", join_failure_rows)
    _write_csv(OUT / "registry_quality_summary_rows.csv", quality_rows)
    _write_json(OUT / "summary.json", summary)
    outputs = [
        OUT / "source_container_rows.csv",
        OUT / "object_candidate_rows.csv",
        OUT / "source_object_registry_rows.csv",
        OUT / "registry_join_failure_rows.csv",
        OUT / "registry_quality_summary_rows.csv",
        OUT / "summary.json",
    ]
    _write_json(OUT / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run()
