#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_v92_phase2_d4rt_sufficiency as phase2
from tools import run_v90_carrier_supported_carving as v90_carve


PHASE_ID = "v92_phase3_d4rt_highres"
RUN_ID = "v92_phase3b_hr1_local_window_bridge"
DEFAULT_OUT = ROOT / "outputs/audit/v92_phase3_d4rt_highres"
DEFAULT_RECOMPUTE_ROOT = ROOT / "outputs/audit/v92_phase3_d4rt_highres_recompute/HR1_grid12_local_window_safe"
DEFAULT_SAME_READOUT_ROOT = ROOT / "outputs/audit/v92_phase3_hr1_same_readout_adaptive_materialization"
PHASE0_SUMMARY = ROOT / "outputs/audit/v92_phase0_mv_ap_contract/summary.json"
PHASE2_SUMMARY = ROOT / "outputs/audit/v92_phase2_d4rt_sufficiency/summary.json"
SOURCE_CONTAINER_ROWS = ROOT / "outputs/audit/v92_phase1_source_container_registry/source_container_rows.csv"
V90_ADAPTER_ROWS = ROOT / "outputs/audit/v90_phase1_variant_resurrection/adapter_input_frame_mask_rows.csv"
WINDOW_ROWS = ROOT / "outputs/audit/v91_phase0_mv_ap_contract/window_support_rows.csv"
CONFIG_KNOB_ROWS = DEFAULT_OUT / "config_knob_audit_rows.csv"
METHOD_SOURCE_VARIANT = "R10_v82_local_B0_object_slot_config"
SUPPORT_RADIUS_PX = 3

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


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_sha(path: Path, cache: dict[Path, str]) -> str:
    if path not in cache:
        cache[path] = _sha256(path) if path.exists() else ""
    return cache[path]


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _float(value: Any, default: float | str = "") -> float | str:
    try:
        if value is None or value == "":
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _percentile(values: list[float], q: float) -> float | str:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return ""
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q))


def _median(values: list[float]) -> float | str:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.median(vals)) if vals else ""


def _mean(values: list[float]) -> float | str:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else ""


def _key(scene_id: str, frame_id: str | int, mask_id: str | int) -> tuple[str, str, str]:
    return (str(scene_id), str(int(float(frame_id))), str(int(float(mask_id))))


def _path_from_rel(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    if (ROOT / path).exists():
        return ROOT / path
    if (REPO_ROOT / path).exists():
        return REPO_ROOT / path
    return ROOT / path


def _common(
    *,
    schema_version: str,
    variant_id: str,
    scene_id: str,
    split: str,
    window_id: str,
    chunk_id: str,
    uses_gt_for_prediction: bool,
    uses_future: bool,
    uses_rgbd_pose_mesh: bool,
    source_artifact: Path,
    source_artifact_sha256: str,
    created_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "variant_id": variant_id,
        "scene_id": scene_id,
        "split": split,
        "window_id": window_id,
        "chunk_id": chunk_id,
        "uses_gt_for_prediction": bool(uses_gt_for_prediction),
        "uses_future": bool(uses_future),
        "uses_rgbd_pose_mesh": bool(uses_rgbd_pose_mesh),
        "source_artifact": _rel(source_artifact),
        "source_artifact_sha256": source_artifact_sha256,
        "created_at": created_at,
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_same_readout(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not root.exists():
        return {}, {}
    return _load_json(root / "best_variant_summary.json"), _load_json(root / "v92_wrapper_summary.json")


def _load_label(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr


def _load_label_lru(path: Path, cache: dict[Path, np.ndarray], order: list[Path], max_items: int = 32) -> np.ndarray:
    if path in cache:
        return cache[path]
    arr = _load_label(path)
    cache[path] = arr
    order.append(path)
    while len(order) > max_items:
        old = order.pop(0)
        cache.pop(old, None)
    return arr


def _load_source_maps() -> tuple[list[dict[str, str]], dict[tuple[str, str, str], dict[str, str]], dict[tuple[str, int], Path], set[tuple[str, str, str]], dict[tuple[str, str, str], int]]:
    rows = [row for row in _read_csv(SOURCE_CONTAINER_ROWS) if row.get("split", "dev") == "dev"]
    unique: dict[tuple[str, str, str], dict[str, str]] = {}
    frame_mask_path: dict[tuple[str, int], Path] = {}
    mask_area: dict[tuple[str, str, str], int] = {}
    for row in rows:
        try:
            key = _key(row.get("scene_id", ""), row.get("frame_id", ""), row.get("source_mask_id", ""))
        except Exception:
            continue
        unique.setdefault(key, row)
        scene_id, frame_text, _mask_text = key
        frame_mask_path.setdefault((scene_id, int(frame_text)), _path_from_rel(row.get("mask_path", "")))
        mask_area[key] = _int(row.get("mask_area_px"), 0)
    return rows, unique, frame_mask_path, set(unique), mask_area


def _load_slot_map() -> dict[tuple[str, str, str], list[str]]:
    slots: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in _read_csv(V90_ADAPTER_ROWS):
        if row.get("variant") != METHOD_SOURCE_VARIANT:
            continue
        if _bool(row.get("uses_gt_for_prediction")) or _bool(row.get("uses_future")):
            continue
        slot = v90_carve._local_slot_from_row(row)
        if not slot:
            continue
        try:
            key = _key(row.get("scene_id", ""), row.get("frame_id", ""), row.get("mask_id", ""))
        except Exception:
            continue
        slots[key].add(slot)
    return {key: sorted(values) for key, values in slots.items()}


def _load_windows() -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    by_id: dict[tuple[str, str], dict[str, Any]] = {}
    by_frame: dict[tuple[str, int], dict[str, Any]] = {}
    for row in _read_csv(WINDOW_ROWS):
        if row.get("split", "dev") != "dev":
            continue
        scene_id = row.get("scene_id", "")
        window_id = row.get("window_id", "")
        start = _int(row.get("frame_id_start"))
        end = _int(row.get("frame_id_end"))
        item = {
            "scene_id": scene_id,
            "split": "dev",
            "window_id": window_id,
            "window_index": row.get("window_index", ""),
            "frame_id_start": start,
            "frame_id_end": end,
            "chunk_id": row.get("chunk_id", ""),
        }
        by_id[(scene_id, window_id)] = item
        for frame_id in range(start, end + 1, 5):
            by_frame[(scene_id, frame_id)] = item
    return by_id, by_frame


def _window_npz_paths(recompute_root: Path) -> list[Path]:
    return sorted(recompute_root.glob("*/stride_5/windows/window_*.npz"))


def _parse_scene_window(path: Path) -> tuple[str, str]:
    scene = path.parents[2].name
    name = path.stem
    window_id = name[len("window_") :] if name.startswith("window_") else name
    return scene, window_id


def _quality_rows(
    quality_obs: dict[tuple[str, str, str], list[dict[str, Any]]],
    artifact_hashes: dict[Path, str],
    source_artifact: Path,
    created_at: str,
    variant_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (scene_id, window_id, carrier_id), obs in sorted(quality_obs.items()):
        by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in obs:
            by_frame[int(item["frame_id"])].append(item)
        frame_ids = sorted(by_frame)
        uv_by_frame: list[tuple[int, float, float]] = []
        mask_tuple_by_frame: list[tuple[int, tuple[str, ...]]] = []
        confs: list[float] = []
        contradiction_frames = 0
        for frame_id in frame_ids:
            items = by_frame[frame_id]
            confs.extend(float(item["confidence"]) for item in items)
            uv_by_frame.append((frame_id, float(np.mean([float(item["x"]) for item in items])), float(np.mean([float(item["y"]) for item in items]))))
            mask_ids = tuple(sorted({str(item["mask_id"]) for item in items}))
            mask_tuple_by_frame.append((frame_id, mask_ids))
            if len(mask_ids) > 1:
                contradiction_frames += 1
        jitters: list[float] = []
        flips = 0
        gaps = 0
        for idx in range(1, len(uv_by_frame)):
            prev = uv_by_frame[idx - 1]
            cur = uv_by_frame[idx]
            jitters.append(float(math.hypot(cur[1] - prev[1], cur[2] - prev[2])))
            if cur[0] - prev[0] > 5:
                gaps += 1
            if mask_tuple_by_frame[idx][1] != mask_tuple_by_frame[idx - 1][1]:
                flips += 1
        rows.append(
            {
                **_common(
                    schema_version="stream4d_v92_phase3_highres_quality_proxy_v1",
                    variant_id=variant_id,
                    scene_id=scene_id,
                    split="dev",
                    window_id=window_id,
                    chunk_id="",
                    uses_gt_for_prediction=False,
                    uses_future=False,
                    uses_rgbd_pose_mesh=False,
                    source_artifact=source_artifact,
                    source_artifact_sha256=_artifact_sha(source_artifact, artifact_hashes),
                    created_at=created_at,
                ),
                "carrier_id": carrier_id,
                "visible_frame_count": len(frame_ids),
                "confidence_mean": _mean(confs),
                "confidence_p10": _percentile(confs, 10),
                "projection_jitter_mean_px": _mean(jitters),
                "projection_jitter_p90_px": _percentile(jitters, 90),
                "mask_membership_flip_rate": (float(flips) / float(max(1, len(mask_tuple_by_frame) - 1))) if len(mask_tuple_by_frame) > 1 else "",
                "source_container_contradiction_rate": (float(contradiction_frames) / float(max(1, len(frame_ids)))) if frame_ids else "",
                "same_track_visibility_gap_count": gaps,
            }
        )
    return rows


def _write_observation_headers(obs_path: Path, support_path: Path) -> tuple[csv.DictWriter, Any, csv.DictWriter, Any]:
    obs_fields = COMMON_FIELDS + [
        "frame_id",
        "source_mask_id",
        "hr_carrier_id",
        "carrier_uv_x_norm",
        "carrier_uv_y_norm",
        "carrier_x_px",
        "carrier_y_px",
        "confidence",
        "visibility_prob",
        "src_frame_global",
        "src_x_px",
        "src_y_px",
        "source_slot_count",
        "source_local_slots",
    ]
    support_fields = [
        "scene_id",
        "chunk_id",
        "frame_id",
        "mask_id",
        "history_id",
        "local_slot_id",
        "cluster_id",
        "native_carrier_global_id",
        "carrier_uv_x",
        "carrier_uv_y",
        "confidence",
        "visibility_prob",
        "observed_mask_support_density",
        "source_observation_table",
        "native_support_kind",
        "native_support_allowed",
        "is_scannet_ap_export",
        "uses_gt_for_prediction",
        "uses_rgbd_pose_mesh_for_export",
        "method_uses_gt",
        "uses_future",
    ]
    obs_path.parent.mkdir(parents=True, exist_ok=True)
    support_path.parent.mkdir(parents=True, exist_ok=True)
    obs_handle = obs_path.open("w", newline="", encoding="utf-8")
    support_handle = support_path.open("w", newline="", encoding="utf-8")
    obs_writer = csv.DictWriter(obs_handle, fieldnames=obs_fields)
    support_writer = csv.DictWriter(support_handle, fieldnames=support_fields)
    obs_writer.writeheader()
    support_writer.writeheader()
    return obs_writer, obs_handle, support_writer, support_handle


def _build_bridge(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    created_at = _created_at()
    variant_id = args.variant_id
    artifact_hashes: dict[Path, str] = {}
    phase0 = _load_json(PHASE0_SUMMARY)
    phase2_summary = _load_json(PHASE2_SUMMARY)
    source_rows, unique_sources, frame_mask_path, source_keys, mask_area_by_key = _load_source_maps()
    slots_by_key = _load_slot_map()
    _windows_by_id, windows_by_frame = _load_windows()
    recompute_root = args.recompute_root
    npz_paths = _window_npz_paths(recompute_root)
    if not npz_paths:
        raise RuntimeError(f"No window npz files found under {recompute_root}")

    obs_path = out / "highres_carrier_observation_rows.csv"
    native_support_path = out / "highres_native_carrier_support_rows.csv"
    obs_writer, obs_handle, support_writer, support_handle = _write_observation_headers(obs_path, native_support_path)

    label_cache: dict[Path, np.ndarray] = {}
    label_order: list[Path] = []
    support_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    quality_obs: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    carrier_window_frames: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    counts = Counter()
    observed_keys: set[tuple[str, str, str]] = set()

    try:
        for npz_path in npz_paths:
            scene_id, window_id = _parse_scene_window(npz_path)
            z = np.load(npz_path)
            frame_ids = np.asarray(z["frame_ids"], dtype=np.int64)
            uv = np.asarray(z["uv"], dtype=np.float32)
            valid = np.asarray(z["valid"], dtype=bool)
            visibility = np.asarray(z["visibility"], dtype=np.float32)
            confidence = np.asarray(z["confidence"], dtype=np.float32)
            carrier_ids = np.asarray(z["carrier_id"], dtype=np.int64)
            src_frame_global = np.asarray(z["src_frame_global"], dtype=np.int64)
            src_xy = np.asarray(z["src_xy"], dtype=np.int64)
            counts["window_npz_count"] += 1
            counts["raw_observation_count"] += int(np.prod(valid.shape))
            for local_i, frame_id_value in enumerate(frame_ids.tolist()):
                frame_id = int(frame_id_value)
                window = windows_by_frame.get((scene_id, frame_id), {})
                label_path = frame_mask_path.get((scene_id, frame_id))
                if label_path is None or not label_path.exists():
                    counts["missing_label_path_frame_count"] += 1
                    continue
                label = _load_label_lru(label_path, label_cache, label_order, max_items=32)
                h, w = label.shape[:2]
                uv_i = uv[local_i]
                ok = (
                    valid[local_i]
                    & np.isfinite(uv_i).all(axis=1)
                    & (uv_i[:, 0] >= 0.0)
                    & (uv_i[:, 0] <= 1.0)
                    & (uv_i[:, 1] >= 0.0)
                    & (uv_i[:, 1] <= 1.0)
                    & (visibility[local_i] >= float(args.min_visibility))
                    & (confidence[local_i] >= float(args.min_confidence))
                )
                counts["inbounds_visibility_confidence_pass_count"] += int(np.count_nonzero(ok))
                if not np.any(ok):
                    continue
                idxs = np.flatnonzero(ok)
                xs = np.rint(uv_i[idxs, 0] * float(max(1, w - 1))).astype(np.int64)
                ys = np.rint(uv_i[idxs, 1] * float(max(1, h - 1))).astype(np.int64)
                xs = np.clip(xs, 0, w - 1)
                ys = np.clip(ys, 0, h - 1)
                mask_ids = label[ys, xs].astype(np.int64)
                for idx_pos, carrier_idx in enumerate(idxs.tolist()):
                    mask_id = int(mask_ids[idx_pos])
                    if mask_id <= 0:
                        counts["background_projected_observation_count"] += 1
                        continue
                    try:
                        key = _key(scene_id, frame_id, mask_id)
                    except Exception:
                        counts["bad_key_count"] += 1
                        continue
                    if key not in source_keys:
                        counts["positive_mask_without_source_key_count"] += 1
                        continue
                    observed_keys.add(key)
                    x_px = int(xs[idx_pos])
                    y_px = int(ys[idx_pos])
                    carrier_uid = f"{scene_id}:{args.support_label}_window:{window_id}:carrier{int(carrier_ids[carrier_idx])}"
                    confidence_value = float(confidence[local_i, carrier_idx])
                    visibility_value = float(visibility[local_i, carrier_idx])
                    item = {
                        "scene_id": scene_id,
                        "window_id": window.get("window_id", window_id),
                        "frame_id": frame_id,
                        "mask_id": str(mask_id),
                        "carrier_id": carrier_uid,
                        "x": float(x_px),
                        "y": float(y_px),
                        "confidence": confidence_value,
                        "visibility": visibility_value,
                    }
                    support_by_key[key].append(item)
                    quality_obs[(scene_id, window.get("window_id", window_id), carrier_uid)].append(item)
                    if visibility_value > 0.0:
                        carrier_window_frames[(scene_id, window.get("window_id", window_id), carrier_uid)].add(frame_id)
                    slots = slots_by_key.get(key, [])
                    source_artifact_sha = _artifact_sha(npz_path, artifact_hashes)
                    obs_writer.writerow(
                        _jsonable(
                            {
                                **_common(
                                    schema_version="stream4d_v92_phase3_highres_carrier_observation_v1",
                                    variant_id=variant_id,
                                    scene_id=scene_id,
                                    split="dev",
                                    window_id=window.get("window_id", window_id),
                                    chunk_id=window_id,
                                    uses_gt_for_prediction=False,
                                    uses_future=False,
                                    uses_rgbd_pose_mesh=False,
                                    source_artifact=npz_path,
                                    source_artifact_sha256=source_artifact_sha,
                                    created_at=created_at,
                                ),
                                "frame_id": frame_id,
                                "source_mask_id": mask_id,
                                "hr_carrier_id": carrier_uid,
                                "carrier_uv_x_norm": float(uv_i[carrier_idx, 0]),
                                "carrier_uv_y_norm": float(uv_i[carrier_idx, 1]),
                                "carrier_x_px": x_px,
                                "carrier_y_px": y_px,
                                "confidence": confidence_value,
                                "visibility_prob": visibility_value,
                                "src_frame_global": int(src_frame_global[carrier_idx]),
                                "src_x_px": int(src_xy[carrier_idx, 0]),
                                "src_y_px": int(src_xy[carrier_idx, 1]),
                                "source_slot_count": len(slots),
                                "source_local_slots": "|".join(slots),
                            }
                        )
                    )
                    counts["matched_source_observation_count"] += 1
                    if not slots:
                        counts["matched_source_observation_without_readout_slot_count"] += 1
                        continue
                    density = 1.0 / float(max(1, mask_area_by_key.get(key, 0)))
                    cluster_id = ""
                    for slot in slots:
                        if ":cluster" in slot:
                            cluster_id = slot.split(":cluster", 1)[1].split(":", 1)[0]
                        support_writer.writerow(
                            {
                                "scene_id": scene_id,
                                "chunk_id": window.get("chunk_id", ""),
                                "frame_id": frame_id,
                                "mask_id": mask_id,
                                "history_id": "",
                                "local_slot_id": slot,
                                "cluster_id": cluster_id,
                                "native_carrier_global_id": carrier_uid,
                                "carrier_uv_x": float(uv_i[carrier_idx, 0]),
                                "carrier_uv_y": float(uv_i[carrier_idx, 1]),
                                "confidence": confidence_value,
                                "visibility_prob": visibility_value,
                                "observed_mask_support_density": density,
                                "source_observation_table": f"v92_{args.support_label}_window_safe_d4rt_projected_to_source_mask",
                                "native_support_kind": f"v92_{args.support_label}_window_safe_source_mask_to_existing_local_slot",
                                "native_support_allowed": True,
                                "is_scannet_ap_export": False,
                                "uses_gt_for_prediction": False,
                                "uses_rgbd_pose_mesh_for_export": False,
                                "method_uses_gt": False,
                                "uses_future": False,
                            }
                        )
                        counts["readout_native_support_row_count"] += 1
    finally:
        obs_handle.close()
        support_handle.close()

    label_cache_stats: dict[Path, np.ndarray] = {}
    support_stats_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    boundary_by_quality: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for key, source_row in sorted(unique_sources.items()):
        stats, boundary_by_carrier, _source_mask, _dilated = phase2._support_stats(
            source_row=source_row,
            support_rows=support_by_key.get(key, []),
            carrier_window_frames=carrier_window_frames,
            label_cache=label_cache_stats,
        )
        support_stats_by_key[key] = stats
        scene_id = source_row.get("scene_id", "")
        window_id = source_row.get("window_id", "")
        for carrier_id, distances in boundary_by_carrier.items():
            boundary_by_quality[(scene_id, window_id, carrier_id)].extend(distances)

    incidence_rows: list[dict[str, Any]] = []
    for row in source_rows:
        key = _key(row.get("scene_id", ""), row.get("frame_id", ""), row.get("source_mask_id", ""))
        stats = support_stats_by_key.get(key, {})
        slots = slots_by_key.get(key, [])
        incidence_rows.append(
            {
                **_common(
                    schema_version="stream4d_v92_phase3_highres_incidence_v1",
                    variant_id=variant_id,
                    scene_id=row.get("scene_id", ""),
                    split=row.get("split", "dev"),
                    window_id=row.get("window_id", ""),
                    chunk_id=row.get("chunk_id", ""),
                    uses_gt_for_prediction=False,
                    uses_future=False,
                    uses_rgbd_pose_mesh=False,
                    source_artifact=native_support_path,
                    source_artifact_sha256=_artifact_sha(native_support_path, artifact_hashes),
                    created_at=created_at,
                ),
                "frame_id": row.get("frame_id", ""),
                "source_mask_id": row.get("source_mask_id", ""),
                "source_variant": row.get("source_variant", row.get("variant_id", "")),
                "carrier_count_inside_source": stats.get("carrier_count_inside_source", 0),
                "carrier_mass_inside_source": stats.get("carrier_mass_inside_source", 0.0),
                "visible_carrier_count_inside_source": stats.get("visible_carrier_count_inside_source", 0),
                "carrier_confidence_mean": stats.get("carrier_confidence_mean", ""),
                "carrier_confidence_p10": stats.get("carrier_confidence_p10", ""),
                "carrier_visibility_frame_count_mean": stats.get("carrier_visibility_frame_count_mean", ""),
                "carrier_support_area_ratio": stats.get("carrier_support_area_ratio", ""),
                "carrier_point_density": stats.get("carrier_point_density", ""),
                "carrier_support_connected_component_count": stats.get("carrier_support_connected_component_count", ""),
                "carrier_support_hole_count": stats.get("carrier_support_hole_count", ""),
                "carrier_support_hole_area_ratio": stats.get("carrier_support_hole_area_ratio", ""),
                "carrier_nearest_neighbor_distance_mean_px": stats.get("carrier_nearest_neighbor_distance_mean_px", ""),
                "carrier_nearest_neighbor_distance_p90_px": stats.get("carrier_nearest_neighbor_distance_p90_px", ""),
                "mask_boundary_carrier_distance_mean": stats.get("mask_boundary_carrier_distance_mean", ""),
                "mask_boundary_carrier_distance_p10": stats.get("mask_boundary_carrier_distance_p10", ""),
                "mask_boundary_carrier_distance_p90": stats.get("mask_boundary_carrier_distance_p90", ""),
                "readout_local_slot_count": len(slots),
                "readout_local_slots": "|".join(slots),
                "support_footprint_radius_px": SUPPORT_RADIUS_PX,
            }
        )

    quality_rows = _quality_rows(quality_obs, artifact_hashes, native_support_path, created_at, variant_id)
    density_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in incidence_rows:
        density_groups[(row["scene_id"], row["split"], row["window_id"], row["variant_id"])].append(row)

    density_rows: list[dict[str, Any]] = []
    lowres_row = {
        **_common(
            schema_version="stream4d_v92_phase3_highres_density_v1",
            variant_id="LOWRES_AD4_baseline",
            scene_id="ALL_DEV",
            split="dev",
            window_id="ALL_WINDOWS",
            chunk_id="",
            uses_gt_for_prediction=False,
            uses_future=False,
            uses_rgbd_pose_mesh=False,
            source_artifact=PHASE2_SUMMARY,
            source_artifact_sha256=_artifact_sha(PHASE2_SUMMARY, artifact_hashes),
            created_at=created_at,
        ),
        "carrier_count_multiplier": 1.0,
        "median_carrier_count_inside_source": phase2_summary.get("median_carrier_count_inside_source_unique_key", ""),
        "median_carrier_support_area_ratio": phase2_summary.get("median_carrier_support_area_ratio_unique_key", ""),
        "carrier_support_area_ratio_p10": phase2_summary.get("carrier_support_area_ratio_p10_unique_key", ""),
        "projection_jitter_p90_px": phase2_summary.get("projection_jitter_p90_global", ""),
        "mask_membership_flip_rate_median": phase2_summary.get("mask_membership_flip_rate_median", ""),
        "source_container_count": phase2_summary.get("source_container_count", ""),
        "run_status": "completed_existing_lowres",
        "notes": "Phase2 low-res D4RT sufficiency diagnostic",
    }
    density_rows.append(lowres_row)
    for (scene_id, split, window_id, group_variant), rows in sorted(density_groups.items()):
        counts_by_key = [float(row.get("carrier_count_inside_source") or 0.0) for row in rows]
        ratios_by_key = [float(row.get("carrier_support_area_ratio") or 0.0) for row in rows if row.get("carrier_support_area_ratio") != ""]
        density_rows.append(
            {
                **_common(
                    schema_version="stream4d_v92_phase3_highres_density_v1",
                    variant_id=group_variant,
                    scene_id=scene_id,
                    split=split,
                    window_id=window_id,
                    chunk_id="",
                    uses_gt_for_prediction=False,
                    uses_future=False,
                    uses_rgbd_pose_mesh=False,
                    source_artifact=native_support_path,
                    source_artifact_sha256=_artifact_sha(native_support_path, artifact_hashes),
                    created_at=created_at,
                ),
                "carrier_count_multiplier": float(args.carrier_count_multiplier),
                "median_carrier_count_inside_source": _median(counts_by_key),
                "median_carrier_support_area_ratio": _median(ratios_by_key),
                "carrier_support_area_ratio_p10": _percentile(ratios_by_key, 10),
                "projection_jitter_p90_px": "",
                "mask_membership_flip_rate_median": "",
                "source_container_count": len(rows),
                "run_status": "completed_hr1_window_safe_bridge",
                "notes": "window-level density row; global quality proxy is in highres_quality_proxy_rows.csv",
            }
        )

    unique_stats = list(support_stats_by_key.values())
    unique_counts = [float(row.get("carrier_count_inside_source") or 0.0) for row in unique_stats]
    unique_ratios = [float(row.get("carrier_support_area_ratio") or 0.0) for row in unique_stats if row.get("carrier_support_area_ratio") != ""]
    quality_jitter_p90_values = [float(row.get("projection_jitter_p90_px") or 0.0) for row in quality_rows if row.get("projection_jitter_p90_px") != ""]
    flip_rates = [float(row.get("mask_membership_flip_rate") or 0.0) for row in quality_rows if row.get("mask_membership_flip_rate") != ""]
    hr1_global_density = {
        **_common(
            schema_version="stream4d_v92_phase3_highres_density_v1",
            variant_id=variant_id,
            scene_id="ALL_DEV",
            split="dev",
            window_id="ALL_WINDOWS",
            chunk_id="",
            uses_gt_for_prediction=False,
            uses_future=False,
            uses_rgbd_pose_mesh=False,
            source_artifact=native_support_path,
            source_artifact_sha256=_artifact_sha(native_support_path, artifact_hashes),
            created_at=created_at,
        ),
        "carrier_count_multiplier": float(args.carrier_count_multiplier),
        "median_carrier_count_inside_source": _median(unique_counts),
        "median_carrier_support_area_ratio": _median(unique_ratios),
        "carrier_support_area_ratio_p10": _percentile(unique_ratios, 10),
        "projection_jitter_p90_px": _percentile(quality_jitter_p90_values, 90),
        "mask_membership_flip_rate_median": _median(flip_rates),
        "source_container_count": len(incidence_rows),
        "run_status": f"completed_{args.support_label}_window_safe_bridge",
        "notes": f"global {args.support_label} source-container density bridge; MV_AP requires same-readout materialization",
    }
    density_rows.insert(1, hr1_global_density)

    best_control_mv = phase0.get("best_control_MV_AP_window", "")
    best_control_ap50 = phase0.get("best_control_MV_AP50_window", "")
    same_readout_best, same_readout_wrapper = _load_same_readout(args.same_readout_root)
    same_readout_completed = bool(same_readout_best)
    same_mv_ap = same_readout_best.get("mean_MV_AP_window", "") if same_readout_completed else ""
    same_ap50 = same_readout_best.get("mean_MV_AP50_window", "") if same_readout_completed else ""
    same_ap25 = same_readout_best.get("mean_MV_AP25_window", "") if same_readout_completed else ""
    same_sf50 = same_readout_best.get("mean_score_free_Match50_window", "") if same_readout_completed else ""
    same_gate = _bool(same_readout_best.get("v91_phase8_progress_gate_pass")) if same_readout_completed else False
    mv_metric_rows = [
        {
            **_common(
                schema_version="stream4d_v92_phase3_highres_mv_metric_v1",
                variant_id="LOWRES_AD4_baseline",
                scene_id="ALL_DEV",
                split="dev",
                window_id="ALL_WINDOWS",
                chunk_id="",
                uses_gt_for_prediction=False,
                uses_future=False,
                uses_rgbd_pose_mesh=False,
                source_artifact=PHASE0_SUMMARY,
                source_artifact_sha256=_artifact_sha(PHASE0_SUMMARY, artifact_hashes),
                created_at=created_at,
            ),
            "carrier_count_multiplier": 1.0,
            "MV_AP_window": phase0.get("v91_best_MV_AP_window", ""),
            "MV_AP50_window": phase0.get("v91_best_MV_AP50_window", ""),
            "MV_AP25_window": phase0.get("v91_best_MV_AP25_window", ""),
            "ScoreFreeMatch50_window": "",
            "best_control_MV_AP_window": best_control_mv,
            "best_control_MV_AP50_window": best_control_ap50,
            "real_minus_best_control_MV_AP_window": "",
            "real_minus_best_control_MV_AP50_window": "",
            "same_frame_collision_count": 0,
            "missing_mask_raster_count": 0,
            "runtime_gpu_hours": "",
            "run_status": "completed_existing_lowres",
        },
        {
            **_common(
                schema_version="stream4d_v92_phase3_highres_mv_metric_v1",
                variant_id=variant_id,
                scene_id="ALL_DEV",
                split="dev",
                window_id="ALL_WINDOWS",
                chunk_id="",
                uses_gt_for_prediction=False,
                uses_future=False,
                uses_rgbd_pose_mesh=False,
                source_artifact=native_support_path,
                source_artifact_sha256=_artifact_sha(native_support_path, artifact_hashes),
                created_at=created_at,
            ),
            "carrier_count_multiplier": float(args.carrier_count_multiplier),
            "MV_AP_window": same_mv_ap,
            "MV_AP50_window": same_ap50,
            "MV_AP25_window": same_ap25,
            "ScoreFreeMatch50_window": same_sf50,
            "best_control_MV_AP_window": best_control_mv,
            "best_control_MV_AP50_window": best_control_ap50,
            "real_minus_best_control_MV_AP_window": same_readout_best.get("real_minus_best_control_MV_AP_window", "") if same_readout_completed else "",
            "real_minus_best_control_MV_AP50_window": same_readout_best.get("real_minus_best_control_MV_AP50_window", "") if same_readout_completed else "",
            "same_frame_collision_count": same_readout_best.get("same_frame_collision_count", "") if same_readout_completed else "",
            "missing_mask_raster_count": same_readout_best.get("missing_mask_raster_count", "") if same_readout_completed else "",
            "runtime_gpu_hours": "",
            "run_status": "completed_same_readout" if same_readout_completed else "support_bridge_completed_mv_ap_not_run_yet",
        },
    ]

    failure_rows = []
    if not counts["readout_native_support_row_count"]:
        failure_rows.append(
            {
                **_common(
                    schema_version="stream4d_v92_phase3_highres_failure_v1",
                    variant_id=variant_id,
                    scene_id="ALL_DEV",
                    split="dev",
                    window_id="ALL_WINDOWS",
                    chunk_id="",
                    uses_gt_for_prediction=False,
                    uses_future=False,
                    uses_rgbd_pose_mesh=False,
                    source_artifact=native_support_path,
                    source_artifact_sha256=_artifact_sha(native_support_path, artifact_hashes),
                    created_at=created_at,
                ),
                "failure_type": f"no_{args.support_label}_readout_support_rows",
                "repair_direction": "check source-mask to R10 local_slot join before same-readout MV_AP",
            }
        )
    if same_readout_completed and not same_gate:
        failure_rows.append(
            {
                **_common(
                    schema_version="stream4d_v92_phase3_highres_failure_v1",
                    variant_id=variant_id,
                    scene_id="ALL_DEV",
                    split="dev",
                    window_id="ALL_WINDOWS",
                    chunk_id="",
                    uses_gt_for_prediction=False,
                    uses_future=False,
                    uses_rgbd_pose_mesh=False,
                    source_artifact=args.same_readout_root / "best_variant_summary.json",
                    source_artifact_sha256=_artifact_sha(args.same_readout_root / "best_variant_summary.json", artifact_hashes),
                    created_at=created_at,
                ),
                "failure_type": f"{args.support_label}_same_readout_no_mv_ap_gain",
                "repair_direction": "per v92 Phase3, try HR2 or route to Phase3C/Phase4 after high-res variants fail",
                "MV_AP_window": same_mv_ap,
                "MV_AP50_window": same_ap50,
                "v91_best_MV_AP_window": phase0.get("v91_best_MV_AP_window", ""),
                "v91_best_MV_AP50_window": phase0.get("v91_best_MV_AP50_window", ""),
            }
        )

    variant_config_rows = [
        {
            **_common(
                schema_version="stream4d_v92_phase3_variant_config_v1",
                variant_id=variant_id,
                scene_id="ALL_DEV",
                split="dev",
                window_id="ALL_WINDOWS",
                chunk_id="",
                uses_gt_for_prediction=False,
                uses_future=False,
                uses_rgbd_pose_mesh=False,
                source_artifact=native_support_path,
                source_artifact_sha256=_artifact_sha(native_support_path, artifact_hashes),
                created_at=created_at,
            ),
            "grid_size": int(args.grid_size),
            "stride": 5,
            "support_footprint_radius_px": SUPPORT_RADIUS_PX,
            "readout_slot_source": f"{_rel(V90_ADAPTER_ROWS)}::{METHOD_SOURCE_VARIANT}",
            "recompute_protocol": "one D4RT forward per local MV_AP dev window",
        }
    ]
    variant_metric_rows = [hr1_global_density]
    gate_values = {
        "window_npz_count_gt_0": counts["window_npz_count"] > 0,
        "matched_source_observation_count_gt_0": counts["matched_source_observation_count"] > 0,
        "readout_native_support_row_count_gt_0": counts["readout_native_support_row_count"] > 0,
        "uses_gt_for_prediction_false": True,
        "uses_future_false": True,
    }
    variant_gate_rows = [
        {
            **_common(
                schema_version="stream4d_v92_phase3_variant_gate_v1",
                variant_id=variant_id,
                scene_id="ALL_DEV",
                split="dev",
                window_id="ALL_WINDOWS",
                chunk_id="",
                uses_gt_for_prediction=False,
                uses_future=False,
                uses_rgbd_pose_mesh=False,
                source_artifact=native_support_path,
                source_artifact_sha256=_artifact_sha(native_support_path, artifact_hashes),
                created_at=created_at,
            ),
            "gate_name": name,
            "gate_pass": bool(value),
        }
        for name, value in gate_values.items()
    ]
    variant_failure_rows = list(failure_rows)
    for name, value in gate_values.items():
        if not value:
            variant_failure_rows.append(
                {
                    **_common(
                        schema_version="stream4d_v92_phase3_variant_failure_v1",
                        variant_id=variant_id,
                        scene_id="ALL_DEV",
                        split="dev",
                        window_id="ALL_WINDOWS",
                        chunk_id="",
                        uses_gt_for_prediction=False,
                        uses_future=False,
                        uses_rgbd_pose_mesh=False,
                        source_artifact=native_support_path,
                        source_artifact_sha256=_artifact_sha(native_support_path, artifact_hashes),
                        created_at=created_at,
                    ),
                    "failure_type": name,
                    "repair_direction": f"repair {args.support_label} bridge before MV_AP materialization",
                }
            )
    casebook_rows = [
        {
            **_common(
                schema_version="stream4d_v92_phase3_casebook_v1",
                variant_id=variant_id,
                scene_id="ALL_DEV",
                split="dev",
                window_id="ALL_WINDOWS",
                chunk_id="",
                uses_gt_for_prediction=False,
                uses_future=False,
                uses_rgbd_pose_mesh=False,
                source_artifact=native_support_path,
                source_artifact_sha256=_artifact_sha(native_support_path, artifact_hashes),
                created_at=created_at,
            ),
            "case_type": f"{args.support_label}_window_safe_bridge",
            "evidence": (
                f"observations={counts['matched_source_observation_count']}; "
                f"readout_support_rows={counts['readout_native_support_row_count']}; "
                f"median_support_area_ratio={hr1_global_density['median_carrier_support_area_ratio']}"
            ),
        }
    ]

    control_rows = _read_csv(out / "highres_control_rows.csv")
    if not control_rows:
        control_rows = []

    _write_csv(out / "highres_incidence_rows.csv", incidence_rows)
    _write_csv(out / "highres_density_rows.csv", density_rows)
    _write_csv(out / "highres_quality_proxy_rows.csv", quality_rows)
    _write_csv(out / "highres_mv_metric_rows.csv", mv_metric_rows)
    _write_csv(out / "highres_failure_rows.csv", failure_rows)
    if control_rows:
        _write_csv(out / "highres_control_rows.csv", control_rows)
    _write_csv(out / "variant_config_rows.csv", variant_config_rows)
    _write_csv(out / "variant_metric_rows.csv", variant_metric_rows)
    _write_csv(out / "variant_gate_rows.csv", variant_gate_rows)
    _write_csv(out / "variant_failure_rows.csv", variant_failure_rows)
    _write_csv(out / "casebook_rows.csv", casebook_rows)

    if CONFIG_KNOB_ROWS.exists():
        # Preserve Phase3A knob audit rows; they are not regenerated by this bridge.
        pass

    recompute_summaries = sorted(args.recompute_root.glob("*/stride_5/summary.json"))
    gpu_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    for summary_path in recompute_summaries:
        summary = _load_json(summary_path)
        scene_id = summary.get("scene", "")
        gpu_rows.append(
            {
                **_common(
                    schema_version="stream4d_v92_phase3_gpu_usage_v1",
                    variant_id=variant_id,
                    scene_id=scene_id,
                    split="dev",
                    window_id="ALL_WINDOWS",
                    chunk_id="",
                    uses_gt_for_prediction=False,
                    uses_future=False,
                    uses_rgbd_pose_mesh=False,
                    source_artifact=summary_path,
                    source_artifact_sha256=_artifact_sha(summary_path, artifact_hashes),
                    created_at=created_at,
                ),
                "device_id": summary.get("cuda_visible_devices", ""),
                "runtime_sec": summary.get("duration_sec", ""),
                "memory_peak_mb": "",
                "memory_peak_source": "not_recorded_by_recompute_script",
            }
        )
        model_rows.append(
            {
                **_common(
                    schema_version="stream4d_v92_phase3_model_forward_v1",
                    variant_id=variant_id,
                    scene_id=scene_id,
                    split="dev",
                    window_id="ALL_WINDOWS",
                    chunk_id="",
                    uses_gt_for_prediction=False,
                    uses_future=False,
                    uses_rgbd_pose_mesh=False,
                    source_artifact=summary_path,
                    source_artifact_sha256=_artifact_sha(summary_path, artifact_hashes),
                    created_at=created_at,
                ),
                "model_name": "OpenD4RT",
                "model_checkpoint_or_version": summary.get("d4rt_ckpt", ""),
                "device_id": summary.get("cuda_visible_devices", ""),
                "num_images_or_masks": summary.get("frame_count", ""),
                "runtime_sec": summary.get("duration_sec", ""),
                "memory_peak_mb": "",
                "grid_size": summary.get("grid_size", ""),
                "window_count": summary.get("window_count", ""),
            }
        )
    _write_csv(out / "gpu_usage_rows.csv", gpu_rows)
    _write_csv(out / "model_forward_rows.csv", model_rows)

    density_improved = (
        hr1_global_density["median_carrier_support_area_ratio"] != ""
        and phase2_summary.get("median_carrier_support_area_ratio_unique_key", "") != ""
        and float(hr1_global_density["median_carrier_support_area_ratio"])
        > float(phase2_summary.get("median_carrier_support_area_ratio_unique_key"))
    )
    if not all(gate_values.values()):
        decision = f"NO_GO_V92_PHASE3B_{args.support_label}_SUPPORT_BRIDGE"
    elif same_readout_completed and not same_gate:
        decision = f"NO_GO_V92_PHASE3B_{args.support_label}_SAME_READOUT_NO_AP_GAIN"
    elif same_readout_completed and same_gate:
        decision = f"PASS_V92_PHASE3B_{args.support_label}_SAME_READOUT_GATE"
    else:
        decision = f"PASS_V92_PHASE3B_{args.support_label}_SUPPORT_BRIDGE_MV_AP_PENDING"

    summary = {
        "schema": "stream4d_v92_phase3b_highres_bridge_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "decision": decision,
        "variant_id": variant_id,
        "support_label": args.support_label,
        "grid_size": int(args.grid_size),
        "carrier_count_multiplier": float(args.carrier_count_multiplier),
        "recompute_root": _rel(args.recompute_root),
        "window_npz_count": counts["window_npz_count"],
        "raw_observation_count": counts["raw_observation_count"],
        "inbounds_visibility_confidence_pass_count": counts["inbounds_visibility_confidence_pass_count"],
        "matched_source_observation_count": counts["matched_source_observation_count"],
        "readout_native_support_row_count": counts["readout_native_support_row_count"],
        "matched_source_observation_without_readout_slot_count": counts["matched_source_observation_without_readout_slot_count"],
        "positive_mask_without_source_key_count": counts["positive_mask_without_source_key_count"],
        "source_container_rows": len(source_rows),
        "unique_scene_frame_mask_count": len(unique_sources),
        "observed_scene_frame_mask_count": len(observed_keys),
        "highres_incidence_rows": len(incidence_rows),
        "highres_quality_proxy_rows": len(quality_rows),
        "highres_median_carrier_count_inside_source_unique_key": hr1_global_density["median_carrier_count_inside_source"],
        "highres_median_carrier_support_area_ratio_unique_key": hr1_global_density["median_carrier_support_area_ratio"],
        "highres_carrier_support_area_ratio_p10": hr1_global_density["carrier_support_area_ratio_p10"],
        "highres_projection_jitter_p90_global": hr1_global_density["projection_jitter_p90_px"],
        "highres_mask_membership_flip_rate_median": hr1_global_density["mask_membership_flip_rate_median"],
        "hr1_median_carrier_count_inside_source_unique_key": hr1_global_density["median_carrier_count_inside_source"],
        "hr1_median_carrier_support_area_ratio_unique_key": hr1_global_density["median_carrier_support_area_ratio"],
        "hr1_carrier_support_area_ratio_p10": hr1_global_density["carrier_support_area_ratio_p10"],
        "hr1_projection_jitter_p90_global": hr1_global_density["projection_jitter_p90_px"],
        "hr1_mask_membership_flip_rate_median": hr1_global_density["mask_membership_flip_rate_median"],
        "lowres_median_carrier_support_area_ratio_unique_key": phase2_summary.get("median_carrier_support_area_ratio_unique_key", ""),
        "density_improved_vs_lowres": bool(density_improved),
        "MV_AP_window_status": "completed_same_readout" if same_readout_completed else "not_run_yet_same_readout_materialization_required",
        "same_readout_root": _rel(args.same_readout_root) if args.same_readout_root.exists() else "",
        "same_readout_best_variant_id": same_readout_best.get("variant_id", ""),
        "same_readout_MV_AP_window": same_mv_ap,
        "same_readout_MV_AP50_window": same_ap50,
        "same_readout_MV_AP25_window": same_ap25,
        "same_readout_gate_pass": bool(same_gate),
        "same_readout_wrapper": same_readout_wrapper,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "uses_rgbd_pose_mesh": False,
        "phase3b_gates": gate_values,
        "input_artifacts": {
            _rel(path): _artifact_sha(path, artifact_hashes)
            for path in [PHASE0_SUMMARY, PHASE2_SUMMARY, SOURCE_CONTAINER_ROWS, V90_ADAPTER_ROWS, WINDOW_ROWS]
            if path.exists()
        },
        "output_artifacts": {
            "highres_carrier_observation_rows": _rel(obs_path),
            "highres_native_carrier_support_rows": _rel(native_support_path),
            "highres_incidence_rows": _rel(out / "highres_incidence_rows.csv"),
            "highres_density_rows": _rel(out / "highres_density_rows.csv"),
            "highres_quality_proxy_rows": _rel(out / "highres_quality_proxy_rows.csv"),
            "highres_mv_metric_rows": _rel(out / "highres_mv_metric_rows.csv"),
        },
        "duration_sec": time.time() - started,
        "created_at": created_at,
    }
    _write_json(out / "summary.json", summary)
    sha_rows = {path.name: _sha256(path) for path in sorted(out.iterdir()) if path.is_file() and path.name != "SHA256SUMS.json"}
    _write_json(out / "SHA256SUMS.json", sha_rows)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge v92 high-res D4RT window-safe chunks into source-container support rows.")
    parser.add_argument("--recompute-root", type=Path, default=DEFAULT_RECOMPUTE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--variant-id", default="HR1_grid12_local_window_safe_same_readout")
    parser.add_argument("--support-label", default="HR1_grid12")
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--carrier-count-multiplier", type=float, default=2.25)
    parser.add_argument("--same-readout-root", type=Path, default=DEFAULT_SAME_READOUT_ROOT)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    return parser


def main() -> None:
    _build_bridge(build_parser().parse_args())


if __name__ == "__main__":
    main()
