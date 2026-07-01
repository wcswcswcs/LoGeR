from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PHASE_ID = "v93_phase1_source_edge_registry"
RUN_ID = "v93_phase1_source_edge_registry"
OUT = ROOT / "outputs/audit/v93_phase1_source_edge_registry"

V92_PHASE1 = ROOT / "outputs/audit/v92_phase1_source_container_registry"

COMMON_FIELDS = [
    "schema_version",
    "phase_id",
    "run_id",
    "variant_id",
    "scene_id",
    "split",
    "window_id",
    "chunk_id",
    "uses_gt_for_prediction",
    "uses_future",
    "uses_rgbd_pose_mesh",
    "source_artifact",
    "source_artifact_sha256",
    "created_at",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        try:
            return str(path.relative_to(ROOT.parent))
        except ValueError:
            return str(path)


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    if (ROOT / path).exists():
        return ROOT / path
    if (ROOT.parent / path).exists():
        return ROOT.parent / path
    return ROOT / path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = []
        for key in COMMON_FIELDS:
            if any(key in row for row in rows):
                fieldnames.append(key)
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


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "allow"}


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _float(value: Any, default: float | str = "") -> float | str:
    try:
        if value is None or value == "":
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _safe_div(num: float | int, den: float | int) -> float:
    den_f = float(den)
    return 0.0 if den_f == 0.0 else float(num) / den_f


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _rewrite_phase_row(row: dict[str, Any], schema_suffix: str, created_at: str) -> dict[str, Any]:
    out = dict(row)
    out["schema_version"] = f"stream4d_v93_phase1_{schema_suffix}_v1"
    out["phase_id"] = PHASE_ID
    out["run_id"] = RUN_ID
    out["created_at"] = created_at
    return out


def _load_source_rows(created_at: str) -> list[dict[str, Any]]:
    rows = []
    for row in _read_csv(V92_PHASE1 / "source_container_rows.csv"):
        out = _rewrite_phase_row(row, "source_container", created_at)
        rows.append(out)
    return rows


def _copy_transformed_csv(src: Path, dst: Path, schema_suffix: str, created_at: str) -> int:
    rows = [_rewrite_phase_row(row, schema_suffix, created_at) for row in _read_csv(src)]
    _write_csv(dst, rows)
    return len(rows)


def _frame_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("scene_id", "")), str(row.get("frame_id", "")), str(row.get("mask_path", "")))


def _load_label_map(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.int64, copy=False)


def _frame_stats(arr: np.ndarray) -> tuple[dict[int, dict[str, int]], dict[int, int], dict[int, list[tuple[int, int]]]]:
    labels = [int(x) for x in np.unique(arr) if int(x) != 0]
    stats: dict[int, dict[str, int]] = {}
    for label in labels:
        ys, xs = np.nonzero(arr == label)
        if ys.size == 0:
            continue
        stats[label] = {
            "area": int(ys.size),
            "x0": int(xs.min()),
            "y0": int(ys.min()),
            "x1": int(xs.max()) + 1,
            "y1": int(ys.max()) + 1,
        }

    boundary_counts: Counter[int] = Counter()
    adjacency_counts: Counter[tuple[int, int]] = Counter()

    def accumulate(a: np.ndarray, b: np.ndarray) -> None:
        diff = a != b
        if not np.any(diff):
            return
        aa = a[diff].reshape(-1)
        bb = b[diff].reshape(-1)
        for left, right in zip(aa.tolist(), bb.tolist()):
            left_i = int(left)
            right_i = int(right)
            if left_i > 0:
                boundary_counts[left_i] += 1
            if right_i > 0:
                boundary_counts[right_i] += 1
            if left_i > 0 and right_i > 0 and left_i != right_i:
                pair = (left_i, right_i) if left_i < right_i else (right_i, left_i)
                adjacency_counts[pair] += 1

    accumulate(arr[:, :-1], arr[:, 1:])
    accumulate(arr[:-1, :], arr[1:, :])
    for side in [arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]]:
        for label in side.tolist():
            label_i = int(label)
            if label_i > 0:
                boundary_counts[label_i] += 1

    neighbors: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (a, b), count in adjacency_counts.items():
        neighbors[a].append((b, int(count)))
        neighbors[b].append((a, int(count)))
    for label in neighbors:
        neighbors[label].sort(key=lambda item: item[1], reverse=True)

    return stats, dict(boundary_counts), neighbors


def _contained_labels(source_label: int, stats: dict[int, dict[str, int]], limit: int = 3) -> list[tuple[int, float]]:
    src = stats.get(source_label)
    if not src:
        return []
    candidates: list[tuple[int, float]] = []
    src_area = max(1, src["area"])
    for label, st in stats.items():
        if label == source_label:
            continue
        if st["area"] >= 0.8 * src_area:
            continue
        inside_bbox = st["x0"] >= src["x0"] and st["y0"] >= src["y0"] and st["x1"] <= src["x1"] and st["y1"] <= src["y1"]
        if not inside_bbox:
            continue
        score = _safe_div(st["area"], src_area)
        candidates.append((label, score))
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[:limit]


def _edge_common(row: dict[str, Any], edge_id: str, edge_type: str, created_at: str) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v93_phase1_mask_edge_hypothesis_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "variant_id": row.get("variant_id", ""),
        "scene_id": row.get("scene_id", ""),
        "split": row.get("split", "dev"),
        "window_id": row.get("window_id", ""),
        "chunk_id": row.get("chunk_id", ""),
        "frame_id": row.get("frame_id", ""),
        "source_mask_id": row.get("source_mask_id", ""),
        "edge_id": edge_id,
        "edge_type": edge_type,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "uses_rgbd_pose_mesh": False,
        "source_artifact": row.get("source_artifact", ""),
        "source_artifact_sha256": row.get("source_artifact_sha256", ""),
        "created_at": created_at,
    }


def _build_edge_rows(source_rows: list[dict[str, Any]], created_at: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    by_frame: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        by_frame[_frame_key(row)].append(row)

    edge_rows: list[dict[str, Any]] = []
    join_failures: list[dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()
    frame_cache: dict[tuple[str, str, str], tuple[dict[int, dict[str, int]], dict[int, int], dict[int, list[tuple[int, int]]]]] = {}

    for frame_key, rows in sorted(by_frame.items()):
        _, _, path_str = frame_key
        mask_path = _resolve(path_str)
        arr = _load_label_map(mask_path)
        if arr is None:
            for row in rows:
                failure_counts["missing_mask_raster_path"] += 1
                join_failures.append(
                    {
                        **_edge_common(row, f"missing_raster:{row.get('source_mask_id')}", "join_failure", created_at),
                        "failure_type": "missing_mask_raster_path",
                        "join_key": path_str,
                        "expected_count": 1,
                        "observed_count": 0,
                    }
                )
            continue
        if frame_key not in frame_cache:
            frame_cache[frame_key] = _frame_stats(arr)
        stats, boundary_counts, neighbors = frame_cache[frame_key]

        for row in rows:
            source_label = _int(row.get("source_mask_id"))
            source_stats = stats.get(source_label)
            if not source_stats:
                failure_counts["missing_mask_label_in_raster"] += 1
                join_failures.append(
                    {
                        **_edge_common(row, f"missing_label:{source_label}", "join_failure", created_at),
                        "failure_type": "missing_mask_label_in_raster",
                        "join_key": f"{path_str}:{source_label}",
                        "expected_count": 1,
                        "observed_count": 0,
                    }
                )
                continue

            source_area = max(1, int(source_stats["area"]))
            boundary_px = int(boundary_counts.get(source_label, 0))
            outer_id = f"{row.get('variant_id')}:{row.get('scene_id')}:{row.get('frame_id')}:{source_label}:outer"
            edge_rows.append(
                {
                    **_edge_common(row, outer_id, "outer", created_at),
                    "edge_mask_id_a": source_label,
                    "edge_mask_id_b": "",
                    "edge_pixel_count": boundary_px,
                    "edge_band_area": min(source_area, max(1, boundary_px * 4)),
                    "edge_confidence": 0.35,
                    "edge_repeated_count": 1,
                    "edge_source_area_ratio": _safe_div(boundary_px, source_area),
                    "near_d4rt_witness_mass": _float(row.get("d4rt_unique_carrier_count"), 0.0),
                    "near_hard_negative_mass": 0.0,
                }
            )

            for other_label, contact_px in neighbors.get(source_label, [])[:6]:
                other_stats = stats.get(other_label, {})
                other_area = max(1, int(other_stats.get("area", 1)))
                confidence = min(1.0, _safe_div(contact_px, math.sqrt(source_area * other_area)) * 8.0)
                edge_id = f"{row.get('variant_id')}:{row.get('scene_id')}:{row.get('frame_id')}:{source_label}:competing:{other_label}"
                edge_rows.append(
                    {
                        **_edge_common(row, edge_id, "competing", created_at),
                        "edge_mask_id_a": source_label,
                        "edge_mask_id_b": other_label,
                        "edge_pixel_count": int(contact_px),
                        "edge_band_area": min(source_area, max(1, int(contact_px) * 4)),
                        "edge_confidence": confidence,
                        "edge_repeated_count": 1,
                        "edge_source_area_ratio": _safe_div(contact_px, source_area),
                        "near_d4rt_witness_mass": _float(row.get("d4rt_unique_carrier_count"), 0.0),
                        "near_hard_negative_mass": 0.0,
                    }
                )

            for nested_label, score in _contained_labels(source_label, stats):
                nested_stats = stats.get(nested_label, {})
                nested_area = int(nested_stats.get("area", 0))
                edge_id = f"{row.get('variant_id')}:{row.get('scene_id')}:{row.get('frame_id')}:{source_label}:nested:{nested_label}"
                edge_rows.append(
                    {
                        **_edge_common(row, edge_id, "nested_overlap", created_at),
                        "edge_mask_id_a": source_label,
                        "edge_mask_id_b": nested_label,
                        "edge_pixel_count": 0,
                        "edge_band_area": nested_area,
                        "edge_confidence": min(1.0, score * 4.0),
                        "edge_repeated_count": 1,
                        "edge_source_area_ratio": _safe_div(nested_area, source_area),
                        "near_d4rt_witness_mass": _float(row.get("d4rt_unique_carrier_count"), 0.0),
                        "near_hard_negative_mass": 0.0,
                    }
                )

    return edge_rows, join_failures, failure_counts


def run() -> dict[str, Any]:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    created_at = _created_at()

    source_rows = _load_source_rows(created_at)
    source_path = V92_PHASE1 / "source_container_rows.csv"
    _write_csv(OUT / "source_container_rows.csv", source_rows)
    object_count = _copy_transformed_csv(V92_PHASE1 / "object_hypothesis_rows.csv", OUT / "object_hypothesis_rows.csv", "object_hypothesis", created_at)
    link_count = _copy_transformed_csv(V92_PHASE1 / "object_container_link_rows.csv", OUT / "object_container_link_rows.csv", "object_container_link", created_at)
    feature_rows = [_rewrite_phase_row(row, "feature_availability", created_at) for row in _read_csv(V92_PHASE1 / "container_feature_availability_rows.csv")]
    _write_csv(OUT / "feature_availability_rows.csv", feature_rows)

    inherited_join_rows = [_rewrite_phase_row(row, "join_failure", created_at) for row in _read_csv(V92_PHASE1 / "join_failure_rows.csv")]
    edge_rows, edge_join_rows, edge_failure_counts = _build_edge_rows(source_rows, created_at)
    join_failure_rows = inherited_join_rows + edge_join_rows
    for row in join_failure_rows:
        row.setdefault("failure_rank", "")
    for rank, row in enumerate(join_failure_rows[:100], 1):
        row["failure_rank"] = rank

    edge_fieldnames = COMMON_FIELDS + [
        "frame_id",
        "source_mask_id",
        "edge_id",
        "edge_type",
        "edge_mask_id_a",
        "edge_mask_id_b",
        "edge_pixel_count",
        "edge_band_area",
        "edge_confidence",
        "edge_repeated_count",
        "edge_source_area_ratio",
        "near_d4rt_witness_mass",
        "near_hard_negative_mass",
    ]
    _write_csv(OUT / "mask_edge_hypothesis_rows.csv", edge_rows, edge_fieldnames)
    _write_csv(OUT / "join_failure_rows.csv", join_failure_rows)

    edge_counter = Counter(row["edge_type"] for row in edge_rows)
    variant_counter = Counter(row.get("variant_id", "") for row in source_rows)
    edge_variant_counter: dict[str, Counter[str]] = defaultdict(Counter)
    for row in edge_rows:
        edge_variant_counter[str(row.get("variant_id", ""))][str(row.get("edge_type", ""))] += 1

    feature_variant_counter: dict[str, Counter[str]] = defaultdict(Counter)
    for row in feature_rows:
        variant_id = str(row.get("variant_id", ""))
        for field in ["has_d4rt_support", "has_dino_feature", "has_radio_feature", "has_region_feature"]:
            if _bool(row.get(field)):
                feature_variant_counter[variant_id][field] += 1

    quality_rows: list[dict[str, Any]] = []
    for variant_id, container_count in sorted(variant_counter.items()):
        ecounts = edge_variant_counter.get(variant_id, Counter())
        fcounts = feature_variant_counter.get(variant_id, Counter())
        total_edges = sum(ecounts.values())
        quality_rows.append(
            {
                "schema_version": "stream4d_v93_phase1_edge_registry_quality_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": variant_id,
                "scene_id": "ALL_DEV",
                "split": "dev",
                "window_id": "ALL_WINDOWS",
                "source_container_count": container_count,
                "edge_hypothesis_count": total_edges,
                "outer_edge_count": ecounts.get("outer", 0),
                "nested_overlap_edge_count": ecounts.get("nested_overlap", 0),
                "competing_edge_count": ecounts.get("competing", 0),
                "repeated_edge_count": ecounts.get("repeated_multiview", 0),
                "edge_band_area_ratio_mean": "",
                "edge_join_failure_rate": _safe_div(len(edge_join_rows), max(1, len(source_rows))),
                "feature_availability_rate": _safe_div(fcounts.get("has_dino_feature", 0) + fcounts.get("has_radio_feature", 0), 2 * max(1, container_count)),
                "D4RT_available_rate": _safe_div(fcounts.get("has_d4rt_support", 0), max(1, container_count)),
                "DINO_or_RADIO_feature_available_rate": _safe_div(
                    sum(1 for row in feature_rows if row.get("variant_id") == variant_id and (_bool(row.get("has_dino_feature")) or _bool(row.get("has_radio_feature")))),
                    max(1, container_count),
                ),
                "region_feature_availability_rate": _safe_div(fcounts.get("has_region_feature", 0), max(1, container_count)),
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "created_at": created_at,
            }
        )
    _write_csv(OUT / "edge_registry_quality_rows.csv", quality_rows)

    join_failure_rate = _safe_div(len(join_failure_rows), max(1, len(source_rows)))
    mask_raster_missing_count = edge_failure_counts.get("missing_mask_raster_path", 0)
    feature_rows_count = len(feature_rows)
    d4rt_available_count = sum(1 for row in feature_rows if _bool(row.get("has_d4rt_support")))
    dino_or_radio_available_count = sum(1 for row in feature_rows if _bool(row.get("has_dino_feature")) or _bool(row.get("has_radio_feature")))
    region_feature_available_count = sum(1 for row in feature_rows if _bool(row.get("has_region_feature")))
    uses_gt_count = sum(1 for row in source_rows if _bool(row.get("uses_gt_for_prediction")))
    uses_future_count = sum(1 for row in source_rows if _bool(row.get("uses_future")))
    phase1_pass_conditions = {
        "join_failure_rate_eq_0": join_failure_rate == 0.0,
        "mask_raster_missing_count_eq_0": mask_raster_missing_count == 0,
        "edge_hypothesis_count_gt_0": len(edge_rows) > 0,
        "nested_overlap_plus_competing_edge_count_gt_0": edge_counter.get("nested_overlap", 0) + edge_counter.get("competing", 0) > 0,
        "D4RT_available_rate_eq_1": d4rt_available_count == feature_rows_count and feature_rows_count > 0,
        "DINO_or_RADIO_feature_available_rate_eq_1": dino_or_radio_available_count == feature_rows_count and feature_rows_count > 0,
        "uses_gt_for_prediction_count_eq_0": uses_gt_count == 0,
        "uses_future_count_eq_0": uses_future_count == 0,
    }
    phase1_pass = all(phase1_pass_conditions.values())

    gate_rows = []
    for gate_name, gate_pass in phase1_pass_conditions.items():
        gate_rows.append(
            {
                "schema_version": "stream4d_v93_phase1_variant_gate_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": "ALL",
                "scene_id": "ALL_DEV",
                "split": "dev",
                "window_id": "ALL_WINDOWS",
                "gate_name": gate_name,
                "gate_pass": bool(gate_pass),
                "gate_value": {
                    "join_failure_rate_eq_0": join_failure_rate,
                    "mask_raster_missing_count_eq_0": mask_raster_missing_count,
                    "edge_hypothesis_count_gt_0": len(edge_rows),
                    "nested_overlap_plus_competing_edge_count_gt_0": edge_counter.get("nested_overlap", 0) + edge_counter.get("competing", 0),
                    "D4RT_available_rate_eq_1": _safe_div(d4rt_available_count, max(1, feature_rows_count)),
                    "DINO_or_RADIO_feature_available_rate_eq_1": _safe_div(dino_or_radio_available_count, max(1, feature_rows_count)),
                    "uses_gt_for_prediction_count_eq_0": uses_gt_count,
                    "uses_future_count_eq_0": uses_future_count,
                }.get(gate_name, ""),
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "created_at": created_at,
            }
        )
    _write_csv(OUT / "variant_gate_rows.csv", gate_rows)

    failure_repairs = {
        "join_failure_rate_eq_0": "fix key schema before any method phase",
        "mask_raster_missing_count_eq_0": "repair mask raster paths before method phase",
        "edge_hypothesis_count_gt_0": "extract same-frame mask overlap/adjoining relations",
        "nested_overlap_plus_competing_edge_count_gt_0": "increase edge band width from 2 px to 4 px and derive same-frame adjacency",
        "D4RT_available_rate_eq_1": "repair D4RT support join before method phase",
        "DINO_or_RADIO_feature_available_rate_eq_1": "repair feature join before method phase",
    }
    failure_rows = [
        {
            "schema_version": "stream4d_v93_phase1_variant_failure_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "variant_id": "ALL",
            "scene_id": "ALL_DEV",
            "split": "dev",
            "window_id": "ALL_WINDOWS",
            "failure_type": row["gate_name"],
            "failure_count": row["gate_value"],
            "repair_direction": failure_repairs.get(row["gate_name"], "repair source artifact and rerun Phase1"),
            "uses_gt_for_prediction": False,
            "uses_future": False,
            "created_at": created_at,
        }
        for row in gate_rows
        if not _bool(row.get("gate_pass"))
    ]
    if region_feature_available_count == 0:
        failure_rows.append(
            {
                "schema_version": "stream4d_v93_phase1_variant_failure_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": "ALL",
                "scene_id": "ALL_DEV",
                "split": "dev",
                "window_id": "ALL_WINDOWS",
                "failure_type": "REGION_FEATURE_MISSING",
                "failure_count": region_feature_available_count,
                "repair_direction": "trigger Phase3 region feature build; do not fake region features with mask-level vectors",
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "created_at": created_at,
            }
        )
    _write_csv(OUT / "variant_failure_rows.csv", failure_rows)

    summary = {
        "schema": "stream4d_v93_phase1_source_edge_registry_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "decision": "PASS_V93_PHASE1_EDGE_REGISTRY" if phase1_pass else "BLOCK_V93_PHASE1_EDGE_REGISTRY",
        "phase1_pass": phase1_pass,
        "phase1_pass_conditions": phase1_pass_conditions,
        "source_container_count": len(source_rows),
        "object_hypothesis_count": object_count,
        "object_container_link_count": link_count,
        "edge_hypothesis_count": len(edge_rows),
        "outer_edge_count": edge_counter.get("outer", 0),
        "nested_overlap_edge_count": edge_counter.get("nested_overlap", 0),
        "competing_edge_count": edge_counter.get("competing", 0),
        "repeated_edge_count": edge_counter.get("repeated_multiview", 0),
        "edge_join_failure_rate": join_failure_rate,
        "join_failure_count": len(join_failure_rows),
        "edge_failure_type_counts": dict(edge_failure_counts),
        "mask_raster_missing_count": mask_raster_missing_count,
        "feature_availability_rows": feature_rows_count,
        "D4RT_available_rate": _safe_div(d4rt_available_count, max(1, feature_rows_count)),
        "DINO_or_RADIO_feature_available_rate": _safe_div(dino_or_radio_available_count, max(1, feature_rows_count)),
        "region_feature_availability_rate": _safe_div(region_feature_available_count, max(1, feature_rows_count)),
        "region_feature_status": "REGION_FEATURE_MISSING; v92 registry exposes mask-level DINO/RADIO only, Phase3 must build explicit region features",
        "uses_gt_for_prediction_count": uses_gt_count,
        "uses_future_count": uses_future_count,
        "input_artifacts": {
            _rel(path): _sha256(path)
            for path in [
                V92_PHASE1 / "summary.json",
                V92_PHASE1 / "source_container_rows.csv",
                V92_PHASE1 / "object_hypothesis_rows.csv",
                V92_PHASE1 / "object_container_link_rows.csv",
                V92_PHASE1 / "container_feature_availability_rows.csv",
                V92_PHASE1 / "join_failure_rows.csv",
            ]
            if path.exists()
        },
        "duration_sec": time.time() - started,
        "created_at": created_at,
    }
    _write_json(OUT / "summary.json", summary)
    sha_rows = {path.name: _sha256(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "SHA256SUMS.json"}
    _write_json(OUT / "SHA256SUMS.json", sha_rows)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run()
