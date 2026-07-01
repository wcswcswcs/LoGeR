#!/usr/bin/env python3
"""Build a method-safe v94 object-axis RADIO region-unary smoke artifact.

This script repairs the Phase7b finding that v94 lacks persisted
object-specific region vector pairs.  It does not run the MV evaluator and does
not claim a dev gate pass; it proves whether current artifacts can produce
source x object x region unary tensors without GT/future inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from stream4d_native.frozen_feature_adapter import FrozenFeatureAdapter, locate_default_radio_checkpoint  # noqa: E402


PHASE_ID = "v94_phase7c_object_axis_field_smoke"
RUN_ID = "v94_phase7c_object_axis_field_smoke"
OUT = ROOT / "outputs/audit/v94_phase7c_object_axis_field_smoke"

PHASE1 = ROOT / "outputs/audit/v94_phase1_canonical_graph"
V91_MASK_FEATURE_STORE = ROOT / "outputs/audit/v91_radio_mask_features_npz"
REGION_NODE_ROWS = PHASE1 / "region_node_rows.csv"

_CLUSTER_RE = re.compile(r"(c\d+:cluster\d+)")


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_vec(vec: np.ndarray) -> str:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(WORKSPACE_ROOT).as_posix()
        except ValueError:
            return path.as_posix()


def _resolve_workspace_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ROOT.name:
        return WORKSPACE_ROOT / path
    return ROOT / path


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _int(value: Any, default: int = -1) -> int:
    try:
        if value in (None, ""):
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _mean(values: list[float]) -> float:
    vals = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(np.asarray(vals, dtype=np.float64))) if vals else 0.0


def _percentile(values: list[float], q: float) -> float:
    vals = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.percentile(np.asarray(vals, dtype=np.float64), float(q))) if vals else 0.0


def _normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 1e-8 else arr


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


def _plan_key(row: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(row.get("scene_id", "")),
        str(row.get("window_id", "")),
        _int(row.get("frame_id"), -1),
        _int(row.get("source_mask_id"), -1),
    )


def _phys_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (str(row.get("scene_id", "")), _int(row.get("frame_id"), -1), _int(row.get("source_mask_id"), -1))


def _source_key_text(key: tuple[str, str, int, int]) -> str:
    scene, window, frame_id, mask_id = key
    return f"{scene}|{window}|{frame_id}|{mask_id}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_object_meta(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in _read_csv(path):
        object_id = str(row.get("object_hypothesis_id", ""))
        if object_id:
            out[object_id] = dict(row)
    return out


def _load_mask_features(npz_path: Path) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / np.maximum(norms, 1e-8)
        scenes = [str(value) for value in data["scene_id"].tolist()]
        frames = [int(value) for value in data["frame_id"].tolist()]
        masks = [int(value) for value in data["mask_id"].tolist()]
        feature_sha = [str(value) for value in data["feature_sha256"].tolist()] if "feature_sha256" in data.files else ["" for _ in scenes]
        backend = str(data["backend"].item()) if "backend" in data.files and data["backend"].shape == () else ""
        layer = str(data["layer"].item()) if "layer" in data.files and data["layer"].shape == () else ""
    key_to_index = {(scene, frame, mask): index for index, (scene, frame, mask) in enumerate(zip(scenes, frames, masks, strict=True))}
    return {
        "features": features,
        "feature_sha256": feature_sha,
        "key_to_index": key_to_index,
        "backend": backend,
        "layer": layer,
        "feature_dim": int(features.shape[1]) if features.ndim == 2 else 0,
        "vector_count": int(features.shape[0]),
    }


def _load_links(
    link_path: Path,
    object_meta: dict[str, dict[str, str]],
    mask_feature_keys: set[tuple[str, int, int]],
    scenes: set[str] | None,
) -> dict[str, Any]:
    plan_objects: dict[tuple[str, str, int, int], set[str]] = defaultdict(set)
    plan_raw_objects: dict[tuple[str, str, int, int], set[str]] = defaultdict(set)
    plan_variants: dict[tuple[str, str, int, int], set[str]] = defaultdict(set)
    object_obs: dict[str, set[tuple[str, int, int]]] = defaultdict(set)
    object_modes: dict[str, Counter[str]] = defaultdict(Counter)
    object_raw_ids: dict[str, set[str]] = defaultdict(set)
    link_rows = 0
    covered_link_rows = 0
    physical_sources: set[tuple[str, int, int]] = set()
    covered_physical_sources: set[tuple[str, int, int]] = set()
    skipped_rows = Counter()

    with link_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("split", "dev")) != "dev":
                skipped_rows["non_dev"] += 1
                continue
            if _bool(row.get("uses_gt_for_prediction")) or _bool(row.get("uses_future")):
                skipped_rows["gt_or_future"] += 1
                continue
            scene = str(row.get("scene_id", ""))
            if scenes and scene not in scenes:
                skipped_rows["scene_filter"] += 1
                continue
            frame_id = _int(row.get("frame_id"), -1)
            mask_id = _int(row.get("source_mask_id"), -1)
            if not scene or frame_id < 0 or mask_id <= 0:
                skipped_rows["bad_key"] += 1
                continue
            object_id = str(row.get("object_hypothesis_id", ""))
            mode, canonical_key = _canonical_object_key(object_id, object_meta.get(object_id), scene)
            plan_key = _plan_key(row)
            phys_key = (scene, frame_id, mask_id)
            link_rows += 1
            physical_sources.add(phys_key)
            plan_objects[plan_key].add(canonical_key)
            plan_raw_objects[plan_key].add(object_id)
            plan_variants[plan_key].add(str(row.get("variant_id", "")))
            object_obs[canonical_key].add(phys_key)
            object_modes[canonical_key][mode] += 1
            object_raw_ids[canonical_key].add(object_id)
            if phys_key in mask_feature_keys:
                covered_link_rows += 1
                covered_physical_sources.add(phys_key)

    proto_available = {
        canonical_key: any(phys in mask_feature_keys for phys in observations)
        for canonical_key, observations in object_obs.items()
    }
    proto_obs_counts = {
        canonical_key: sum(1 for phys in observations if phys in mask_feature_keys)
        for canonical_key, observations in object_obs.items()
    }
    plan_count = len(plan_objects)
    multi_plan_keys = [key for key, objects in plan_objects.items() if len(objects) > 1]
    full_proto_plan_count = sum(1 for objects in plan_objects.values() if objects and all(proto_available.get(obj, False) for obj in objects))
    full_proto_multi_count = sum(1 for key in multi_plan_keys if all(proto_available.get(obj, False) for obj in plan_objects[key]))

    return {
        "plan_objects": dict(plan_objects),
        "plan_raw_objects": dict(plan_raw_objects),
        "plan_variants": dict(plan_variants),
        "object_obs": dict(object_obs),
        "object_modes": dict(object_modes),
        "object_raw_ids": dict(object_raw_ids),
        "proto_available": proto_available,
        "proto_obs_counts": proto_obs_counts,
        "stats": {
            "dev_link_rows": int(link_rows),
            "covered_link_rows": int(covered_link_rows),
            "covered_link_row_rate": float(covered_link_rows / link_rows) if link_rows else 0.0,
            "physical_source_count": len(physical_sources),
            "covered_physical_source_count": len(covered_physical_sources),
            "covered_physical_source_rate": float(len(covered_physical_sources) / len(physical_sources)) if physical_sources else 0.0,
            "plan_key_count": int(plan_count),
            "multi_object_plan_key_count": int(len(multi_plan_keys)),
            "multi_object_plan_key_rate": float(len(multi_plan_keys) / plan_count) if plan_count else 0.0,
            "canonical_object_count": len(object_obs),
            "prototype_available_object_count": sum(1 for value in proto_available.values() if value),
            "prototype_available_object_rate": float(sum(1 for value in proto_available.values() if value) / len(object_obs)) if object_obs else 0.0,
            "plan_key_all_objects_have_proto_count": int(full_proto_plan_count),
            "plan_key_all_objects_have_proto_rate": float(full_proto_plan_count / plan_count) if plan_count else 0.0,
            "multi_plan_all_objects_have_proto_count": int(full_proto_multi_count),
            "multi_plan_all_objects_have_proto_rate": float(full_proto_multi_count / len(multi_plan_keys)) if multi_plan_keys else 0.0,
            "skipped_link_rows": dict(skipped_rows),
        },
    }


def _load_container_rows(path: Path, selected_keys: set[tuple[str, str, int, int]]) -> dict[tuple[str, str, int, int], dict[str, str]]:
    out: dict[tuple[str, str, int, int], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = _plan_key(row)
            if key not in selected_keys:
                continue
            if _bool(row.get("uses_gt_for_prediction")) or _bool(row.get("uses_future")):
                continue
            existing = out.get(key)
            if existing is None or str(row.get("variant_id", "")) == "B0_local_only":
                out[key] = dict(row)
    return out


def _select_sources(args: argparse.Namespace, link_pack: dict[str, Any], mask_feature_keys: set[tuple[str, int, int]]) -> tuple[list[tuple[str, str, int, int]], list[dict[str, Any]]]:
    plan_objects: dict[tuple[str, str, int, int], set[str]] = link_pack["plan_objects"]
    proto_available: dict[str, bool] = link_pack["proto_available"]
    rows: list[dict[str, Any]] = []
    selected: list[tuple[str, str, int, int]] = []
    frame_keys: set[tuple[str, int]] = set()
    candidate_keys = sorted(plan_objects, key=lambda key: (key[0], key[2], key[3], key[1]))
    for key in candidate_keys:
        scene, window_id, frame_id, mask_id = key
        objects = sorted(plan_objects[key])
        phys = (scene, frame_id, mask_id)
        is_multi = len(objects) > 1
        source_feature_available = phys in mask_feature_keys
        all_proto = bool(objects) and all(proto_available.get(obj, False) for obj in objects)
        select_reason = ""
        if args.require_multi_object and not is_multi:
            select_reason = "skip_single_object"
        elif not source_feature_available:
            select_reason = "skip_source_mask_feature_missing"
        elif not all_proto:
            select_reason = "skip_object_proto_missing"
        elif int(args.max_frames) > 0 and (scene, frame_id) not in frame_keys and len(frame_keys) >= int(args.max_frames):
            select_reason = "skip_max_frames"
        elif int(args.max_sources) > 0 and len(selected) >= int(args.max_sources):
            select_reason = "skip_max_sources"
        else:
            selected.append(key)
            frame_keys.add((scene, frame_id))
            select_reason = "selected"
        if select_reason == "selected" or len(rows) < int(args.casebook_limit):
            rows.append(
                {
                    "schema_version": "stream4d_v94_phase7c_source_selection_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "scene_id": scene,
                    "window_id": window_id,
                    "frame_id": frame_id,
                    "source_mask_id": mask_id,
                    "canonical_object_count": len(objects),
                    "source_mask_feature_available": source_feature_available,
                    "all_object_prototypes_available": all_proto,
                    "is_multi_object": is_multi,
                    "selection_status": select_reason,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    return selected, rows


def _load_region_nodes(
    path: Path,
    selected_keys: set[tuple[str, str, int, int]],
    max_regions_per_source: int,
) -> tuple[dict[tuple[str, str, int, int], list[dict[str, str]]], dict[str, Any]]:
    by_source: dict[tuple[str, str, int, int], list[dict[str, str]]] = defaultdict(list)
    scanned = 0
    matched = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scanned += 1
            key = _plan_key(row)
            if key not in selected_keys:
                continue
            if _bool(row.get("uses_gt_for_prediction")) or _bool(row.get("uses_future")):
                continue
            matched += 1
            by_source[key].append(dict(row))
    truncated_count = 0
    if int(max_regions_per_source) > 0:
        for key, rows in list(by_source.items()):
            if len(rows) <= int(max_regions_per_source):
                continue
            rows.sort(key=lambda row: (_num(row.get("area_px", row.get("pixel_count", 0.0))), _int(row.get("region_index"), 0)), reverse=True)
            by_source[key] = sorted(rows[: int(max_regions_per_source)], key=lambda row: _int(row.get("region_index"), 0))
            truncated_count += 1
    return dict(by_source), {
        "region_node_rows_scanned": int(scanned),
        "selected_region_node_rows_matched": int(matched),
        "source_region_rows_loaded": int(sum(len(rows) for rows in by_source.values())),
        "source_region_truncated_count": int(truncated_count),
    }


def _prototype_for_object(
    canonical_object_key: str,
    current_phys: tuple[str, int, int],
    object_obs: dict[str, set[tuple[str, int, int]]],
    mask_pack: dict[str, Any],
    leave_one_source_out: bool,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    key_to_index: dict[tuple[str, int, int], int] = mask_pack["key_to_index"]
    features: np.ndarray = mask_pack["features"]
    observations = sorted(phys for phys in object_obs.get(canonical_object_key, set()) if phys in key_to_index)
    used = [phys for phys in observations if not (leave_one_source_out and phys == current_phys)]
    self_excluded = leave_one_source_out and current_phys in observations and bool(used)
    self_included = current_phys in used
    fallback_self_included = False
    if not used and observations:
        used = observations
        self_included = current_phys in used
        fallback_self_included = self_included and leave_one_source_out
    if not used:
        return None, {
            "prototype_available": False,
            "prototype_obs_count": 0,
            "prototype_self_excluded": False,
            "prototype_self_included": False,
            "prototype_fallback_self_included": False,
        }
    vecs = np.stack([features[key_to_index[phys]] for phys in used], axis=0).astype(np.float32)
    proto = _normalize(vecs.mean(axis=0))
    return proto, {
        "prototype_available": True,
        "prototype_obs_count": int(len(used)),
        "prototype_total_obs_count": int(len(observations)),
        "prototype_self_excluded": bool(self_excluded),
        "prototype_self_included": bool(self_included),
        "prototype_fallback_self_included": bool(fallback_self_included),
        "prototype_sha256": _sha256_vec(proto),
    }


def _torch_cosine_matrix(proto_matrix: np.ndarray, region_matrix: np.ndarray, device: str) -> tuple[np.ndarray, str]:
    try:
        import torch

        actual_device = device
        if device.startswith("cuda") and not torch.cuda.is_available():
            actual_device = "cpu"
        protos = torch.from_numpy(np.asarray(proto_matrix, dtype=np.float32)).to(actual_device)
        regions = torch.from_numpy(np.asarray(region_matrix, dtype=np.float32)).to(actual_device)
        with torch.inference_mode():
            out = protos @ regions.transpose(0, 1)
        return out.detach().cpu().numpy().astype(np.float32), f"torch_matmul:{actual_device}"
    except Exception:
        return (np.asarray(proto_matrix, dtype=np.float32) @ np.asarray(region_matrix, dtype=np.float32).T).astype(np.float32), "numpy_matmul_cpu_fallback"


def _cpu_parity_max_abs(proto_matrix: np.ndarray, region_matrix: np.ndarray, scores: np.ndarray) -> float:
    expected = (np.asarray(proto_matrix, dtype=np.float32) @ np.asarray(region_matrix, dtype=np.float32).T).astype(np.float32)
    if expected.shape != scores.shape:
        return float("inf")
    return float(np.max(np.abs(expected - np.asarray(scores, dtype=np.float32)))) if expected.size else 0.0


def _build_field(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = _resolve_workspace_path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "field_shards").mkdir(parents=True, exist_ok=True)
    for stale_shard in (out / "field_shards").glob("object_axis_unary_shard_*.npz"):
        stale_shard.unlink()
    created_at = _created_at()
    scenes = {item.strip() for item in str(args.scenes).split(",") if item.strip()}

    mask_pack = _load_mask_features(_resolve_workspace_path(args.mask_features_npz))
    mask_feature_keys = set(mask_pack["key_to_index"])
    object_meta = _load_object_meta(_resolve_workspace_path(args.object_rows))
    link_pack = _load_links(_resolve_workspace_path(args.link_rows), object_meta, mask_feature_keys, scenes or None)
    selected_sources, source_selection_rows = _select_sources(args, link_pack, mask_feature_keys)
    selected_set = set(selected_sources)
    container_rows = _load_container_rows(_resolve_workspace_path(args.container_rows), selected_set)
    region_nodes, region_stats = _load_region_nodes(_resolve_workspace_path(args.region_node_rows), selected_set, int(args.max_regions_per_source))

    checkpoint = str(args.checkpoint).strip() or str(locate_default_radio_checkpoint() or "")
    adapter = FrozenFeatureAdapter(
        backend="radio_radseg",
        device=str(args.device),
        checkpoint=checkpoint,
        radio_lang_model=str(args.radio_lang_model),
        radio_lang_align=bool(args.radio_lang_align),
        radio_slide_crop=int(args.radio_slide_crop),
        radio_slide_stride=int(args.radio_slide_stride),
    )

    source_object_rows: list[dict[str, Any]] = []
    prototype_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    unary_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    source_summary_rows: list[dict[str, Any]] = []

    shard_source_keys: list[str] = []
    shard_object_keys: list[str] = []
    shard_object_source_index: list[int] = []
    shard_object_local_index: list[int] = []
    shard_region_source_index: list[int] = []
    shard_region_ids: list[str] = []
    shard_region_indices: list[int] = []
    shard_region_feature_yx: list[tuple[int, int]] = []
    shard_unary_source_index: list[int] = []
    shard_unary_object_local_index: list[int] = []
    shard_unary_region_local_index: list[int] = []
    shard_unary_cosine: list[float] = []
    shard_paths: list[Path] = []
    shard_index = 0
    total_unary_count = 0
    shard_max_unary_count = int(args.shard_max_unary_count)

    def _flush_field_shard() -> None:
        nonlocal shard_index, total_unary_count
        nonlocal shard_source_keys, shard_object_keys, shard_object_source_index, shard_object_local_index
        nonlocal shard_region_source_index, shard_region_ids, shard_region_indices, shard_region_feature_yx
        nonlocal shard_unary_source_index, shard_unary_object_local_index, shard_unary_region_local_index, shard_unary_cosine
        if not shard_source_keys:
            return
        shard_path = out / "field_shards" / f"object_axis_unary_shard_{shard_index:04d}.npz"
        np.savez_compressed(
            shard_path,
            source_keys=np.asarray(shard_source_keys, dtype="U128"),
            object_keys=np.asarray(shard_object_keys, dtype="U256"),
            object_source_index=np.asarray(shard_object_source_index, dtype=np.int32),
            object_local_index=np.asarray(shard_object_local_index, dtype=np.int32),
            region_source_index=np.asarray(shard_region_source_index, dtype=np.int32),
            region_ids=np.asarray(shard_region_ids, dtype="U128"),
            region_indices=np.asarray(shard_region_indices, dtype=np.int32),
            region_feature_yx=np.asarray(shard_region_feature_yx, dtype=np.int32).reshape((-1, 2)) if shard_region_feature_yx else np.zeros((0, 2), dtype=np.int32),
            unary_source_index=np.asarray(shard_unary_source_index, dtype=np.int32),
            unary_object_local_index=np.asarray(shard_unary_object_local_index, dtype=np.int32),
            unary_region_local_index=np.asarray(shard_unary_region_local_index, dtype=np.int32),
            unary_cosine=np.asarray(shard_unary_cosine, dtype=np.float32),
            schema_version=np.asarray("stream4d_v94_phase7c_object_axis_unary_shard_v1"),
            phase_id=np.asarray(PHASE_ID),
            run_id=np.asarray(RUN_ID),
            uses_gt_for_prediction=np.asarray(False),
            uses_future=np.asarray(False),
        )
        shard_paths.append(shard_path)
        total_unary_count += len(shard_unary_cosine)
        shard_index += 1
        shard_source_keys = []
        shard_object_keys = []
        shard_object_source_index = []
        shard_object_local_index = []
        shard_region_source_index = []
        shard_region_ids = []
        shard_region_indices = []
        shard_region_feature_yx = []
        shard_unary_source_index = []
        shard_unary_object_local_index = []
        shard_unary_region_local_index = []
        shard_unary_cosine = []

    streams: dict[str, ScanNetStream] = {}
    feature_cache: dict[tuple[str, int], np.ndarray] = {}
    frame_feature_shapes: dict[str, tuple[int, int, int]] = {}
    cosine_backends = Counter()
    parity_values: list[float] = []
    region_counts: list[int] = []
    object_counts: list[int] = []
    processed_source_count = 0
    extracted_frame_count = 0
    unary_csv_limit = int(args.unary_csv_limit)

    for source_index, key in enumerate(selected_sources):
        scene, window_id, frame_id, mask_id = key
        source_key = _source_key_text(key)
        nodes = region_nodes.get(key, [])
        objects = sorted(link_pack["plan_objects"].get(key, set()))
        if not nodes:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v94_phase7c_failure_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "scene_id": scene,
                    "window_id": window_id,
                    "frame_id": frame_id,
                    "source_mask_id": mask_id,
                    "failure_type": "region_nodes_missing_for_selected_source",
                    "repair_direction": "rebuild v93/v94 region_node_rows for this source before Phase3A materialization",
                    "detail": source_key,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
            continue
        if key not in container_rows:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v94_phase7c_failure_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "scene_id": scene,
                    "window_id": window_id,
                    "frame_id": frame_id,
                    "source_mask_id": mask_id,
                    "failure_type": "container_row_missing_for_selected_source",
                    "repair_direction": "repair Phase1 container symlink/schema before field smoke",
                    "detail": source_key,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
            continue
        frame_key = (scene, frame_id)
        try:
            if frame_key not in feature_cache:
                streams.setdefault(scene, ScanNetStream(scene, root=ROOT / "data/scannet/processed"))
                rgb = streams[scene].load_rgb(int(frame_id))
                feature_map = adapter.extract_dense_features(rgb)
                features = np.asarray(feature_map.features, dtype=np.float32)
                feature_cache[frame_key] = features
                frame_feature_shapes[f"{scene}|{frame_id}"] = tuple(int(v) for v in features.shape)
                extracted_frame_count += 1
            features = feature_cache[frame_key]
        except Exception as exc:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v94_phase7c_failure_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "scene_id": scene,
                    "window_id": window_id,
                    "frame_id": frame_id,
                    "source_mask_id": mask_id,
                    "failure_type": f"radio_dense_feature_extract_failed:{type(exc).__name__}",
                    "repair_direction": "check RADIO checkpoint/import/GPU/RGB path and rerun; do not fall back to mask-level cosine as region feature",
                    "detail": str(exc),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
            continue

        region_vecs: list[np.ndarray] = []
        valid_nodes: list[dict[str, str]] = []
        feature_h, feature_w, feature_dim = features.shape
        for node in nodes:
            fy = _int(node.get("feature_y"), -1)
            fx = _int(node.get("feature_x"), -1)
            if fy < 0 or fx < 0 or fy >= feature_h or fx >= feature_w:
                continue
            region_vecs.append(_normalize(features[fy, fx]))
            valid_nodes.append(node)
        if not region_vecs:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v94_phase7c_failure_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "scene_id": scene,
                    "window_id": window_id,
                    "frame_id": frame_id,
                    "source_mask_id": mask_id,
                    "failure_type": "no_region_vectors_after_feature_grid_alignment",
                    "repair_direction": "verify v93 region feature_y/x were generated with the same RADIO grid",
                    "detail": f"feature_shape={features.shape} node_count={len(nodes)}",
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
            continue

        proto_vecs: list[np.ndarray] = []
        object_meta_rows: list[dict[str, Any]] = []
        current_phys = (scene, frame_id, mask_id)
        for object_local_index, canonical_object_key in enumerate(objects):
            proto, proto_meta = _prototype_for_object(
                canonical_object_key,
                current_phys,
                link_pack["object_obs"],
                mask_pack,
                bool(args.leave_one_source_out),
            )
            if proto is None:
                continue
            proto_vecs.append(proto)
            row = {
                "schema_version": "stream4d_v94_phase7c_object_prototype_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "source_index": source_index,
                "scene_id": scene,
                "window_id": window_id,
                "frame_id": frame_id,
                "source_mask_id": mask_id,
                "canonical_object_key": canonical_object_key,
                "object_local_index": object_local_index,
                "raw_object_count_for_canonical": len(link_pack["object_raw_ids"].get(canonical_object_key, set())),
                "canonical_mode_histogram": dict(link_pack["object_modes"].get(canonical_object_key, Counter())),
                "mask_feature_backend": mask_pack["backend"],
                "mask_feature_layer": mask_pack["layer"],
                **proto_meta,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
            object_meta_rows.append(row)
            prototype_rows.append(row)

        if not proto_vecs:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v94_phase7c_failure_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "scene_id": scene,
                    "window_id": window_id,
                    "frame_id": frame_id,
                    "source_mask_id": mask_id,
                    "failure_type": "no_object_prototypes_for_selected_source",
                    "repair_direction": "rebuild method-safe object prototype store keyed by canonical object id",
                    "detail": source_key,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
            continue

        proto_matrix = np.stack(proto_vecs, axis=0).astype(np.float32)
        region_matrix = np.stack(region_vecs, axis=0).astype(np.float32)
        scores, backend = _torch_cosine_matrix(proto_matrix, region_matrix, str(args.cosine_device))
        cosine_backends[backend] += 1
        parity = _cpu_parity_max_abs(proto_matrix, region_matrix, scores)
        parity_values.append(parity)
        top_object = np.argmax(scores, axis=0)
        sorted_scores = np.sort(scores, axis=0)
        top_scores = sorted_scores[-1, :]
        second_scores = sorted_scores[-2, :] if scores.shape[0] >= 2 else np.zeros_like(top_scores)
        margins = top_scores - second_scores

        source_unary_count = int(scores.shape[0] * scores.shape[1])
        source_region_count = int(scores.shape[1])
        source_object_count = int(scores.shape[0])
        object_counts.append(source_object_count)
        region_counts.append(source_region_count)
        shard_source_keys.append(source_key)
        source_shard_index = len(shard_source_keys) - 1
        object_key_by_local = [row["canonical_object_key"] for row in object_meta_rows]

        for object_local_index, canonical_object_key in enumerate(object_key_by_local):
            values = scores[object_local_index, :]
            source_object_rows.append(
                {
                    "schema_version": "stream4d_v94_phase7c_source_object_axis_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "source_index": source_index,
                    "scene_id": scene,
                    "window_id": window_id,
                    "frame_id": frame_id,
                    "source_mask_id": mask_id,
                    "canonical_object_key": canonical_object_key,
                    "object_local_index": object_local_index,
                    "region_count": source_region_count,
                    "unary_cosine_mean": float(values.mean()) if values.size else 0.0,
                    "unary_cosine_max": float(values.max()) if values.size else 0.0,
                    "unary_cosine_p90": float(np.percentile(values, 90)) if values.size else 0.0,
                    "assigned_region_count": int(np.count_nonzero(top_object == object_local_index)),
                    "assigned_region_rate": float(np.count_nonzero(top_object == object_local_index) / max(1, source_region_count)),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
            shard_object_keys.append(canonical_object_key)
            shard_object_source_index.append(source_shard_index)
            shard_object_local_index.append(object_local_index)

        for region_local_index, node in enumerate(valid_nodes):
            assigned_idx = int(top_object[region_local_index])
            assigned_object_key = object_key_by_local[assigned_idx]
            top_score = float(top_scores[region_local_index])
            margin = float(margins[region_local_index])
            assignment_rows.append(
                {
                    "schema_version": "stream4d_v94_phase7c_region_assignment_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "source_index": source_index,
                    "scene_id": scene,
                    "window_id": window_id,
                    "frame_id": frame_id,
                    "source_mask_id": mask_id,
                    "region_id": node.get("region_id", ""),
                    "region_index": _int(node.get("region_index"), region_local_index),
                    "feature_y": _int(node.get("feature_y"), -1),
                    "feature_x": _int(node.get("feature_x"), -1),
                    "assigned_object_local_index": assigned_idx,
                    "assigned_canonical_object_key": assigned_object_key,
                    "top_unary_cosine": top_score,
                    "unary_margin": margin,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
            shard_region_source_index.append(source_shard_index)
            shard_region_ids.append(str(node.get("region_id", "")))
            shard_region_indices.append(_int(node.get("region_index"), region_local_index))
            shard_region_feature_yx.append((_int(node.get("feature_y"), -1), _int(node.get("feature_x"), -1)))

        for object_local_index in range(scores.shape[0]):
            for region_local_index in range(scores.shape[1]):
                value = float(scores[object_local_index, region_local_index])
                shard_unary_source_index.append(source_shard_index)
                shard_unary_object_local_index.append(object_local_index)
                shard_unary_region_local_index.append(region_local_index)
                shard_unary_cosine.append(value)
                if unary_csv_limit <= 0 or len(unary_rows) < unary_csv_limit:
                    node = valid_nodes[region_local_index]
                    unary_rows.append(
                        {
                            "schema_version": "stream4d_v94_phase7c_field_unary_v1",
                            "phase_id": PHASE_ID,
                            "run_id": RUN_ID,
                            "source_index": source_index,
                            "scene_id": scene,
                            "window_id": window_id,
                            "frame_id": frame_id,
                            "source_mask_id": mask_id,
                            "canonical_object_key": object_key_by_local[object_local_index],
                            "object_local_index": object_local_index,
                            "region_id": node.get("region_id", ""),
                            "region_index": _int(node.get("region_index"), region_local_index),
                            "unary_kind": "radio_region_to_canonical_object_proto_cosine",
                            "unary_cosine": value,
                            "cosine_backend": backend,
                            "uses_gt_for_prediction": False,
                            "uses_future": False,
                        }
                    )

        source_summary_rows.append(
            {
                "schema_version": "stream4d_v94_phase7c_source_summary_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "source_index": source_index,
                "scene_id": scene,
                "window_id": window_id,
                "frame_id": frame_id,
                "source_mask_id": mask_id,
                "canonical_object_count": len(objects),
                "prototype_object_count": source_object_count,
                "region_count": source_region_count,
                "unary_count": source_unary_count,
                "cosine_backend": backend,
                "cpu_parity_max_abs_diff": parity,
                "mean_top_unary_cosine": float(top_scores.mean()) if top_scores.size else 0.0,
                "mean_unary_margin": float(margins.mean()) if margins.size else 0.0,
                "p10_unary_margin": float(np.percentile(margins, 10)) if margins.size else 0.0,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        processed_source_count += 1
        if shard_max_unary_count > 0 and len(shard_unary_cosine) >= shard_max_unary_count:
            _flush_field_shard()

    _flush_field_shard()

    _write_csv(out / "source_selection_rows.csv", source_selection_rows)
    _write_csv(out / "object_prototype_rows.csv", prototype_rows)
    _write_csv(out / "source_object_axis_rows.csv", source_object_rows)
    _write_csv(out / "field_unary_rows.csv", unary_rows)
    _write_csv(out / "region_assignment_rows.csv", assignment_rows)
    _write_csv(out / "source_summary_rows.csv", source_summary_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)

    parity_max = max(parity_values) if parity_values else float("inf")
    parity_gate_pass = bool(parity_values and parity_max <= float(args.parity_tolerance))
    processed_source_rate = float(processed_source_count / max(1, len(selected_sources)))
    object_specific_field_input_gate_pass = bool(
        processed_source_count > 0
        and extracted_frame_count > 0
        and total_unary_count > 0
        and parity_gate_pass
        and processed_source_rate >= float(args.min_processed_source_rate)
    )
    if object_specific_field_input_gate_pass and not failure_rows:
        decision = "PASS_V94_OBJECT_AXIS_FIELD_SMOKE_READY_FOR_PHASE3A"
        blocker = ""
        repair = "Promote this object-axis unary builder into Phase3A materialization/evaluation variants."
    elif object_specific_field_input_gate_pass:
        decision = "PASS_V94_OBJECT_AXIS_FIELD_SMOKE_READY_FOR_PHASE3A_WITH_SOURCE_SKIPS"
        blocker = "nonfatal_source_level_skips_present"
        repair = "Promote to Phase3A with explicit source-skip accounting; inspect failure_rows before full-dev materialization."
    elif processed_source_count == 0:
        decision = "NO_GO_V94_OBJECT_AXIS_FIELD_NO_SOURCE_PROCESSED"
        blocker = "no_method_safe_source_object_region_vector_pair_built"
        repair = "Repair selection, region_node, RADIO checkpoint, or prototype coverage before Phase3A."
    elif not parity_gate_pass:
        decision = "NO_GO_V94_OBJECT_AXIS_FIELD_GPU_PARITY_FAILED"
        blocker = "gpu_cosine_parity_failed"
        repair = "Debug torch/GPU cosine computation before using object-axis unary scores."
    else:
        decision = "NO_GO_V94_OBJECT_AXIS_FIELD_PARTIAL_FAILURES"
        blocker = "source_level_failures_present"
        repair = "Inspect failure_rows.csv and repair missing region/RADIO/prototype inputs before Phase3A."

    gate_rows.append(
        {
            "schema_version": "stream4d_v94_phase7c_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "gate_name": "object_specific_field_input_gate",
            "gate_pass": object_specific_field_input_gate_pass,
            "processed_source_count": processed_source_count,
            "selected_source_count": len(selected_sources),
            "processed_source_rate": processed_source_rate,
            "min_processed_source_rate": float(args.min_processed_source_rate),
            "extracted_frame_count": extracted_frame_count,
            "unary_count": total_unary_count,
            "field_shard_count": len(shard_paths),
            "failure_count": len(failure_rows),
            "cpu_parity_max_abs_diff": "" if not parity_values else parity_max,
            "parity_tolerance": float(args.parity_tolerance),
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
    )
    _write_csv(out / "variant_gate_rows.csv", gate_rows)

    artifact_paths = [
        out / "source_selection_rows.csv",
        out / "object_prototype_rows.csv",
        out / "source_object_axis_rows.csv",
        out / "field_unary_rows.csv",
        out / "region_assignment_rows.csv",
        out / "source_summary_rows.csv",
        out / "failure_rows.csv",
        out / "variant_gate_rows.csv",
        *shard_paths,
    ]

    summary = {
        "schema": "stream4d_v94_phase7c_object_axis_field_smoke_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": created_at,
        "decision": decision,
        "object_specific_field_input_gate_pass": object_specific_field_input_gate_pass,
        "blocker": blocker,
        "recommended_repair_direction": repair,
        "materialized_predictions": False,
        "dev_progress_gate_pass": False,
        "holdout_executed": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "scenes": sorted(scenes) if scenes else "ALL",
        "max_sources": int(args.max_sources),
        "max_frames": int(args.max_frames),
        "require_multi_object": bool(args.require_multi_object),
        "leave_one_source_out": bool(args.leave_one_source_out),
        "radio_checkpoint": checkpoint,
        "radio_device_requested": str(args.device),
        "cosine_device_requested": str(args.cosine_device),
        "cosine_backend_counts": dict(cosine_backends),
        "frame_feature_shapes": frame_feature_shapes,
        "v91_mask_feature_vector_count": mask_pack["vector_count"],
        "v91_mask_feature_dim": mask_pack["feature_dim"],
        "v91_mask_feature_backend": mask_pack["backend"],
        "v91_mask_feature_layer": mask_pack["layer"],
        **link_pack["stats"],
        **region_stats,
        "selected_source_count": len(selected_sources),
        "processed_source_count": processed_source_count,
        "processed_source_rate": processed_source_rate,
        "min_processed_source_rate": float(args.min_processed_source_rate),
        "extracted_frame_count": extracted_frame_count,
        "object_prototype_rows": len(prototype_rows),
        "source_object_axis_rows": len(source_object_rows),
        "field_unary_rows_csv": len(unary_rows),
        "field_unary_rows_csv_truncated": bool(unary_csv_limit > 0 and len(unary_rows) < total_unary_count),
        "field_unary_count_shard": total_unary_count,
        "field_shard_count": len(shard_paths),
        "shard_max_unary_count": shard_max_unary_count,
        "region_assignment_rows": len(assignment_rows),
        "failure_count": len(failure_rows),
        "source_region_count_mean": _mean([float(v) for v in region_counts]),
        "source_region_count_p90": _percentile([float(v) for v in region_counts], 90),
        "source_object_count_mean": _mean([float(v) for v in object_counts]),
        "source_object_count_p90": _percentile([float(v) for v in object_counts], 90),
        "cpu_parity_checked_source_count": len(parity_values),
        "cpu_parity_max_abs_diff": "" if not parity_values else parity_max,
        "parity_gate_pass": parity_gate_pass,
        "runtime_sec": float(time.time() - started),
        "artifacts": [_rel(path) for path in artifact_paths],
    }

    _write_json(out / "summary.json", summary)
    artifact_paths.append(out / "summary.json")
    _write_json(out / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in artifact_paths if path.exists()})
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUT))
    parser.add_argument("--container-rows", default=str(PHASE1 / "container_rows.csv"))
    parser.add_argument("--link-rows", default=str(PHASE1 / "container_object_link_rows.csv"))
    parser.add_argument("--object-rows", default=str(PHASE1 / "object_hypothesis_rows.csv"))
    parser.add_argument("--region-node-rows", default=str(REGION_NODE_ROWS))
    parser.add_argument("--mask-features-npz", default=str(V91_MASK_FEATURE_STORE / "mask_features.npz"))
    parser.add_argument("--scenes", default="scene0011_00")
    parser.add_argument("--max-sources", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=2)
    parser.add_argument("--max-regions-per-source", type=int, default=0)
    parser.add_argument("--min-processed-source-rate", type=float, default=0.95)
    parser.add_argument("--casebook-limit", type=int, default=200)
    parser.add_argument("--unary-csv-limit", type=int, default=250000)
    parser.add_argument("--shard-max-unary-count", type=int, default=2_000_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cosine-device", default="cuda:0")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--radio-lang-model", default="siglip2")
    parser.add_argument("--radio-lang-align", action="store_true")
    parser.add_argument("--radio-slide-crop", type=int, default=0)
    parser.add_argument("--radio-slide-stride", type=int, default=224)
    parser.add_argument("--parity-tolerance", type=float, default=1e-4)
    parser.add_argument("--require-multi-object", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--leave-one-source-out", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    summary = _build_field(parse_args())
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
