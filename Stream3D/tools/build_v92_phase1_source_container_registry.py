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


PHASE_ID = "v92_phase1_source_container_registry"
RUN_ID = "v92_phase1_registry"
OUT = ROOT / "outputs/audit/v92_phase1_source_container_registry"

ADAPTER_ROWS = ROOT / "outputs/audit/v90_phase1_variant_resurrection/adapter_input_frame_mask_rows.csv"
V90_MV_OBJECT_ROWS = ROOT / "outputs/audit/v90_phase1_variant_resurrection/mv_object_rows.csv"
V90_MV_FRAME_ROWS = ROOT / "outputs/audit/v90_phase1_variant_resurrection/mv_object_frame_mask_rows.csv"
WINDOW_ROWS = ROOT / "outputs/audit/v91_phase0_mv_ap_contract/window_support_rows.csv"
D4RT_SUPPORT_ROWS = ROOT / "outputs/audit/v90_phase3_v82_full_support/native_carrier_support_rows.csv"
DINO_MASK_ROWS = ROOT / "outputs/audit/v81_dino_feature_json_scene0011_scene0050/mask_feature_rows.csv"
RADIO_INDEX_ROWS = ROOT / "outputs/audit/v91_radio_mask_features_npz/mask_feature_index.csv"
RADIO_SCENE_ROWS = [
    ROOT / "outputs/audit/v91_radio_mask_features_npz_scene0011/mask_feature_rows.csv",
    ROOT / "outputs/audit/v91_radio_mask_features_npz_scene0050/mask_feature_rows.csv",
]
V91_SOURCE_SELECTIONS = [
    ROOT / "outputs/audit/v91_phase4_adaptive_uncertainty_materialization/source_selection_rows.csv",
    ROOT / "outputs/audit/v91_phase4_radio_affinity_readout_npz/source_selection_rows.csv",
    ROOT / "outputs/audit/v91_phase4_affinity_semantic_consensus_repair/source_selection_rows.csv",
    ROOT / "outputs/audit/v91_phase4_scene_risk_materialization/source_selection_rows.csv",
]

V91_BEST_VARIANT = "V91_AD4_sr2_adapt_sig8_b05_j075_r12"
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
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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


def _safe_div(num: float | int, den: float | int) -> float | str:
    try:
        den_f = float(den)
        if den_f == 0:
            return ""
        return float(num) / den_f
    except Exception:
        return ""


def _key(scene_id: str, frame_id: str | int, mask_id: str | int) -> tuple[str, str, str]:
    return (str(scene_id), str(int(float(frame_id))), str(int(float(mask_id))))


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _is_v91_best_variant(variant_id: str) -> bool:
    return variant_id == V91_BEST_VARIANT or variant_id == f"{V91_BEST_VARIANT}_source"


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


def _load_windows() -> dict[str, list[dict[str, Any]]]:
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_csv(WINDOW_ROWS):
        by_scene[row["scene_id"]].append(
            {
                "split": row.get("split", "dev"),
                "window_id": row.get("window_id", ""),
                "window_index": row.get("window_index", ""),
                "start": _int(row.get("frame_id_start")),
                "end": _int(row.get("frame_id_end")),
                "chunk_id": row.get("chunk_id", ""),
                "mask_source": row.get("mask_source", ""),
            }
        )
    for rows in by_scene.values():
        rows.sort(key=lambda item: (item["start"], item["end"]))
    return by_scene


def _find_window(
    windows_by_scene: dict[str, list[dict[str, Any]]], cache: dict[tuple[str, int], dict[str, Any]], scene_id: str, frame_id: int
) -> dict[str, Any]:
    cache_key = (scene_id, frame_id)
    if cache_key in cache:
        return cache[cache_key]
    for row in windows_by_scene.get(scene_id, []):
        if row["start"] <= frame_id <= row["end"]:
            cache[cache_key] = row
            return row
    cache[cache_key] = {}
    return {}


def _resolve_mask_path(mask_source: str, frame_id: int, row_path: str = "") -> Path:
    if row_path:
        path = Path(row_path)
        if path.is_absolute():
            return path
        if (ROOT / path).exists():
            return ROOT / path
        if (ROOT.parent / path).exists():
            return ROOT.parent / path
        return ROOT / path
    source = Path(mask_source)
    candidate = source / f"{frame_id}.png"
    if candidate.is_absolute():
        return candidate
    if (ROOT / candidate).exists():
        return ROOT / candidate
    if (ROOT.parent / candidate).exists():
        return ROOT.parent / candidate
    return ROOT / candidate


def _label_stats_for_frame(mask_path: Path, cache: dict[Path, dict[int, dict[str, int]]]) -> dict[int, dict[str, int]]:
    if mask_path in cache:
        return cache[mask_path]
    stats: dict[int, dict[str, int]] = {}
    if not mask_path.exists():
        cache[mask_path] = stats
        return stats
    arr = np.asarray(Image.open(mask_path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    height, width = int(arr.shape[0]), int(arr.shape[1])
    labels = np.unique(arr)
    for raw_label in labels:
        label = int(raw_label)
        if label == 0:
            continue
        ys, xs = np.nonzero(arr == raw_label)
        if ys.size == 0:
            continue
        stats[label] = {
            "mask_area_px": int(ys.size),
            "image_area_px": height * width,
            "mask_bbox_x0": int(xs.min()),
            "mask_bbox_y0": int(ys.min()),
            "mask_bbox_x1": int(xs.max()) + 1,
            "mask_bbox_y1": int(ys.max()) + 1,
        }
    cache[mask_path] = stats
    return stats


def _load_feature_rows(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    out: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in _read_csv(path):
        if "feature_available" in row and not _bool(row.get("feature_available")):
            continue
        try:
            out[_key(row.get("scene_id", ""), row.get("frame_id", ""), row.get("mask_id", ""))] = row
        except Exception:
            continue
    return out


def _load_radio_rows() -> dict[tuple[str, str, str], dict[str, str]]:
    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in _read_csv(RADIO_INDEX_ROWS):
        try:
            rows[_key(row.get("scene_id", ""), row.get("frame_id", ""), row.get("mask_id", ""))] = row
        except Exception:
            continue
    for path in RADIO_SCENE_ROWS:
        for row in _read_csv(path):
            try:
                key = _key(row.get("scene_id", ""), row.get("frame_id", ""), row.get("mask_id", ""))
            except Exception:
                continue
            if key in rows:
                rows[key].update(row)
            elif _bool(row.get("feature_available", "true")):
                rows[key] = row
    return rows


def _load_d4rt_support() -> dict[tuple[str, str, str], dict[str, Any]]:
    support: dict[tuple[str, str, str], dict[str, Any]] = {}
    carrier_sets: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    confidence_sum: Counter[tuple[str, str, str]] = Counter()
    visibility_sum: Counter[tuple[str, str, str]] = Counter()
    row_count: Counter[tuple[str, str, str]] = Counter()
    density_sum: Counter[tuple[str, str, str]] = Counter()
    for row in _read_csv(D4RT_SUPPORT_ROWS):
        try:
            key = _key(row.get("scene_id", ""), row.get("frame_id", ""), row.get("mask_id", ""))
        except Exception:
            continue
        carrier_sets[key].add(row.get("native_carrier_global_id", ""))
        confidence_sum[key] += float(_float(row.get("confidence"), 0.0) or 0.0)
        visibility_sum[key] += float(_float(row.get("visibility_prob"), 0.0) or 0.0)
        density_sum[key] += float(_float(row.get("observed_mask_support_density"), 0.0) or 0.0)
        row_count[key] += 1
    for key, count in row_count.items():
        support[key] = {
            "d4rt_support_row_count": int(count),
            "d4rt_unique_carrier_count": len([item for item in carrier_sets[key] if item]),
            "d4rt_confidence_mean": confidence_sum[key] / count if count else "",
            "d4rt_visibility_mean": visibility_sum[key] / count if count else "",
            "d4rt_density_mean": density_sum[key] / count if count else "",
        }
    return support


def _merge_container(
    containers: dict[tuple[str, str, str, str, str, str], dict[str, Any]],
    container_key: tuple[str, str, str, str, str, str],
    row: dict[str, Any],
) -> None:
    existing = containers.get(container_key)
    if existing is None:
        containers[container_key] = row
        return
    existing["source_artifact_set"].update(row.get("source_artifact_set", set()))
    existing["source_row_count"] = int(existing.get("source_row_count", 1)) + int(row.get("source_row_count", 1))
    for field in ["source_mask_score", "cropformer_confidence", "background_risk_score", "underseg_risk_score"]:
        current = _float(existing.get(field), "")
        incoming = _float(row.get(field), "")
        if current == "" and incoming != "":
            existing[field] = incoming
        elif current != "" and incoming != "":
            existing[field] = max(float(current), float(incoming))
    existing["broad_mask_flag"] = bool(_bool(existing.get("broad_mask_flag")) or _bool(row.get("broad_mask_flag")))


def _object_state() -> dict[str, Any]:
    return {
        "frames": set(),
        "containers": set(),
        "carrier_count": 0,
        "object_score": "",
        "risk_score": "",
        "local_cluster_id": "",
        "history_id": "",
        "tracklet_id": "",
        "object_scale": "",
        "semantic_proto": "",
        "source_artifact_set": set(),
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "uses_rgbd_pose_mesh": False,
    }


def _update_object(
    objects: dict[tuple[str, str, str, str, str], dict[str, Any]],
    object_key: tuple[str, str, str, str, str],
    *,
    frame_id: str,
    container_key: tuple[str, str, str, str, str, str],
    row: dict[str, str],
    source_artifact: Path,
) -> None:
    state = objects.setdefault(object_key, _object_state())
    state["frames"].add(str(frame_id))
    state["containers"].add(container_key)
    state["source_artifact_set"].add(source_artifact)
    state["carrier_count"] += _int(row.get("native_carrier_support_count", row.get("support_count", 0)))
    for dst, src in [
        ("object_score", "object_score"),
        ("risk_score", "broad_background_risk"),
        ("local_cluster_id", "local_slot_id"),
        ("history_id", "history_id"),
        ("tracklet_id", "tracklet_id"),
        ("object_scale", "object_state"),
        ("semantic_proto", "semantic_proto_id"),
    ]:
        value = row.get(src, "")
        if value and not state.get(dst):
            state[dst] = value
    score = _float(row.get("object_score"), "")
    if score != "":
        current = _float(state.get("object_score"), "")
        if current == "" or float(score) > float(current):
            state["object_score"] = score
    state["uses_gt_for_prediction"] = bool(state["uses_gt_for_prediction"] or _bool(row.get("uses_gt_for_prediction")) or _bool(row.get("method_uses_gt")))
    state["uses_future"] = bool(state["uses_future"] or _bool(row.get("uses_future")))
    state["uses_rgbd_pose_mesh"] = bool(state["uses_rgbd_pose_mesh"] or _bool(row.get("uses_rgbd_pose_mesh")))


def _build_source_row(
    *,
    row: dict[str, str],
    source_artifact: Path,
    artifact_hashes: dict[Path, str],
    windows_by_scene: dict[str, list[dict[str, Any]]],
    window_cache: dict[tuple[str, int], dict[str, Any]],
    d4rt_support: dict[tuple[str, str, str], dict[str, Any]],
    dino_rows: dict[tuple[str, str, str], dict[str, str]],
    radio_rows: dict[tuple[str, str, str], dict[str, str]],
    mask_stats_cache: dict[Path, dict[int, dict[str, int]]],
    source_variant: str,
    object_hypothesis_id: str,
    link_role: str,
) -> tuple[tuple[str, str, str, str, str, str], dict[str, Any], tuple[str, str, str, str, str], dict[str, Any], list[dict[str, Any]]]:
    created_at = _created_at()
    scene_id = row.get("scene_id", "")
    split = row.get("split", "dev") or "dev"
    frame_id_i = _int(row.get("frame_id"))
    mask_id_i = _int(row.get("mask_id"))
    source_mask_id = str(mask_id_i)
    key3 = _key(scene_id, frame_id_i, mask_id_i)
    window = _find_window(windows_by_scene, window_cache, scene_id, frame_id_i)
    window_id = row.get("window_id") or window.get("window_id", "")
    chunk_id = row.get("chunk_id") or row.get("v80_chunk_id") or window.get("chunk_id", "")
    mask_path = _resolve_mask_path(window.get("mask_source", ""), frame_id_i, row.get("mask_raster_path", ""))
    label_stats = _label_stats_for_frame(mask_path, mask_stats_cache).get(mask_id_i, {})
    row_area = _int(row.get("mask_area", row.get("source_mask_area", "")), 0)
    mask_area_px = int(label_stats.get("mask_area_px", row_area))
    image_area_px = int(label_stats.get("image_area_px", _int(row.get("image_area"), 0)))
    mask_area_ratio = _safe_div(mask_area_px, image_area_px) if image_area_px else _float(row.get("area_ratio"), "")
    d4rt = d4rt_support.get(key3, {})
    dino = dino_rows.get(key3, {})
    radio = radio_rows.get(key3, {})
    source_expected_support = _int(row.get("native_carrier_support_count", row.get("support_count", 0)))
    has_d4rt_support = bool(d4rt.get("d4rt_support_row_count", 0) or source_expected_support > 0)
    has_dino_feature = bool(dino)
    has_radio_feature = bool(radio)
    has_region_feature = False
    broad_mask_flag = _bool(row.get("broad_mask_flag")) or _bool(row.get("broad_background_risk"))
    background_risk_score = _float(row.get("broad_background_risk"), "")
    if background_risk_score == "":
        background_risk_score = 1.0 if broad_mask_flag else 0.0
    underseg_risk_score = _float(row.get("broad_leak_risk"), "")
    if underseg_risk_score == "":
        underseg_risk_score = float(mask_area_ratio) if broad_mask_flag and mask_area_ratio != "" else 0.0
    common = _common(
        schema_version="stream4d_v92_phase1_source_container_v1",
        variant_id=source_variant,
        scene_id=scene_id,
        split=split,
        window_id=window_id,
        chunk_id=chunk_id,
        uses_gt_for_prediction=_bool(row.get("uses_gt_for_prediction")) or _bool(row.get("method_uses_gt")),
        uses_future=_bool(row.get("uses_future")),
        uses_rgbd_pose_mesh=_bool(row.get("uses_rgbd_pose_mesh")),
        source_artifact=source_artifact,
        source_artifact_sha256=_artifact_sha(source_artifact, artifact_hashes),
        created_at=created_at,
    )
    container_key = (scene_id, split, window_id, str(frame_id_i), source_mask_id, source_variant)
    container_row = {
        **common,
        "frame_id": str(frame_id_i),
        "source_mask_id": source_mask_id,
        "source_variant": source_variant,
        "mask_path": _rel(mask_path),
        "mask_path_exists": mask_path.exists(),
        "mask_area_px": mask_area_px,
        "image_area_px": image_area_px,
        "mask_area_ratio": mask_area_ratio,
        "mask_bbox_x0": label_stats.get("mask_bbox_x0", ""),
        "mask_bbox_y0": label_stats.get("mask_bbox_y0", ""),
        "mask_bbox_x1": label_stats.get("mask_bbox_x1", ""),
        "mask_bbox_y1": label_stats.get("mask_bbox_y1", ""),
        "source_mask_score": _float(row.get("frame_mask_score", row.get("selection_score", "")), ""),
        "cropformer_confidence": _float(row.get("frame_mask_score", ""), ""),
        "broad_mask_flag": broad_mask_flag,
        "background_risk_score": background_risk_score,
        "underseg_risk_score": underseg_risk_score,
        "available_in_B0": False,
        "available_in_v91_best": _is_v91_best_variant(source_variant),
        "has_d4rt_support": has_d4rt_support,
        "has_dino_feature": has_dino_feature,
        "has_radio_feature": has_radio_feature,
        "has_region_feature": has_region_feature,
        "d4rt_support_row_count": d4rt.get("d4rt_support_row_count", source_expected_support),
        "d4rt_unique_carrier_count": d4rt.get("d4rt_unique_carrier_count", source_expected_support),
        "d4rt_confidence_mean": d4rt.get("d4rt_confidence_mean", row.get("confidence_mean", "")),
        "d4rt_visibility_mean": d4rt.get("d4rt_visibility_mean", row.get("visibility_mean", "")),
        "d4rt_density_mean": d4rt.get("d4rt_density_mean", row.get("observed_density_mean", "")),
        "dino_feature_sha256": dino.get("feature_sha256", ""),
        "dino_feature_backend": dino.get("semantic_backend", ""),
        "radio_feature_sha256": radio.get("feature_sha256", ""),
        "radio_feature_backend": radio.get("semantic_backend", "radio_npz_index" if radio else ""),
        "source_row_count": 1,
        "source_artifact_set": {source_artifact},
    }
    object_key = (scene_id, split, window_id, object_hypothesis_id, source_variant)
    link_common = _common(
        schema_version="stream4d_v92_phase1_object_container_link_v1",
        variant_id=source_variant,
        scene_id=scene_id,
        split=split,
        window_id=window_id,
        chunk_id=chunk_id,
        uses_gt_for_prediction=container_row["uses_gt_for_prediction"],
        uses_future=container_row["uses_future"],
        uses_rgbd_pose_mesh=container_row["uses_rgbd_pose_mesh"],
        source_artifact=source_artifact,
        source_artifact_sha256=_artifact_sha(source_artifact, artifact_hashes),
        created_at=created_at,
    )
    link_row = {
        **link_common,
        "object_hypothesis_id": object_hypothesis_id,
        "frame_id": str(frame_id_i),
        "source_mask_id": source_mask_id,
        "link_role": link_role,
        "adapter_precision": "",
        "adapter_recall": "",
        "adapter_f1": "",
        "adapter_score_raw": _float(row.get("adapter_score"), ""),
        "carrier_support_mass": _float(row.get("support_score", row.get("support_count", "")), ""),
        "carrier_visible_mass": _float(row.get("visibility_mean"), ""),
        "mask_area_px": mask_area_px,
        "mask_selected_by_variant": _bool(row.get("selected_flag", "true")) or row.get("selection_stage", "") != "",
        "mask_selected_score": _float(row.get("selection_score", row.get("frame_mask_score", "")), ""),
        "selection_reason": row.get("selection_reason", ""),
    }
    failures: list[dict[str, Any]] = []
    failure_common = _common(
        schema_version="stream4d_v92_phase1_join_failure_v1",
        variant_id=source_variant,
        scene_id=scene_id,
        split=split,
        window_id=window_id,
        chunk_id=chunk_id,
        uses_gt_for_prediction=container_row["uses_gt_for_prediction"],
        uses_future=container_row["uses_future"],
        uses_rgbd_pose_mesh=container_row["uses_rgbd_pose_mesh"],
        source_artifact=source_artifact,
        source_artifact_sha256=_artifact_sha(source_artifact, artifact_hashes),
        created_at=created_at,
    )
    if not window:
        failures.append({**failure_common, "failure_type": "missing_window_key", "frame_id": str(frame_id_i), "source_mask_id": source_mask_id, "join_key": "|".join(key3), "expected_count": "", "observed_count": 0})
    if source_expected_support > 0 and not d4rt:
        failures.append({**failure_common, "failure_type": "missing_d4rt_support_key", "frame_id": str(frame_id_i), "source_mask_id": source_mask_id, "join_key": "|".join(key3), "expected_count": source_expected_support, "observed_count": 0})
    if not has_dino_feature:
        failures.append({**failure_common, "failure_type": "missing_dino_feature_key", "frame_id": str(frame_id_i), "source_mask_id": source_mask_id, "join_key": "|".join(key3), "expected_count": 1, "observed_count": 0})
    if not has_radio_feature:
        failures.append({**failure_common, "failure_type": "missing_radio_feature_key", "frame_id": str(frame_id_i), "source_mask_id": source_mask_id, "join_key": "|".join(key3), "expected_count": 1, "observed_count": 0})
    if not mask_path.exists():
        failures.append({**failure_common, "failure_type": "missing_mask_raster_path", "frame_id": str(frame_id_i), "source_mask_id": source_mask_id, "join_key": _rel(mask_path), "expected_count": 1, "observed_count": 0})
    elif mask_area_px == 0 and row_area > 0:
        failures.append({**failure_common, "failure_type": "missing_mask_label_in_raster", "frame_id": str(frame_id_i), "source_mask_id": source_mask_id, "join_key": _rel(mask_path), "expected_count": row_area, "observed_count": 0})
    return container_key, container_row, object_key, link_row, failures


def run() -> dict[str, Any]:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    created_at = _created_at()
    artifact_hashes: dict[Path, str] = {}
    windows_by_scene = _load_windows()
    window_cache: dict[tuple[str, int], dict[str, Any]] = {}
    d4rt_support = _load_d4rt_support()
    dino_rows = _load_feature_rows(DINO_MASK_ROWS)
    radio_rows = _load_radio_rows()
    mask_stats_cache: dict[Path, dict[int, dict[str, int]]] = {}
    containers: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    objects: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    link_rows: list[dict[str, Any]] = []
    join_failure_rows: list[dict[str, Any]] = []
    source_artifact_row_counts: Counter[str] = Counter()

    def ingest(row: dict[str, str], source_artifact: Path, source_variant: str, object_id: str, role: str) -> None:
        if not row.get("scene_id") or row.get("frame_id", "") == "" or row.get("mask_id", "") == "":
            return
        container_key, container_row, object_key, link_row, failures = _build_source_row(
            row=row,
            source_artifact=source_artifact,
            artifact_hashes=artifact_hashes,
            windows_by_scene=windows_by_scene,
            window_cache=window_cache,
            d4rt_support=d4rt_support,
            dino_rows=dino_rows,
            radio_rows=radio_rows,
            mask_stats_cache=mask_stats_cache,
            source_variant=source_variant,
            object_hypothesis_id=object_id,
            link_role=role,
        )
        _merge_container(containers, container_key, container_row)
        _update_object(objects, object_key, frame_id=container_key[3], container_key=container_key, row=row, source_artifact=source_artifact)
        link_rows.append(link_row)
        join_failure_rows.extend(failures)
        source_artifact_row_counts[_rel(source_artifact)] += 1

    for row in _read_csv(ADAPTER_ROWS):
        source_variant = row.get("source_variant") or row.get("variant") or "unknown_adapter_variant"
        object_id = row.get("mv_object_id") or f"{source_variant}:{row.get('scene_id')}:{row.get('history_id')}"
        ingest(row, ADAPTER_ROWS, source_variant, object_id, "v90_adapter_source_container")

    for path in V91_SOURCE_SELECTIONS:
        for row in _read_csv(path):
            variant_id = row.get("variant_id") or row.get("variant") or row.get("source_variant") or path.parent.name
            object_id = row.get("mv_object_id") or f"{variant_id}:{row.get('scene_id')}:{row.get('local_slot_id')}"
            row = dict(row)
            row.setdefault("split", "dev")
            ingest(row, path, variant_id, object_id, "v91_phase4_source_selection")

    base_has_b0: set[tuple[str, str, str, str, str]] = set()
    base_has_v91: set[tuple[str, str, str, str, str]] = set()
    for key, row in containers.items():
        scene_id, split, window_id, frame_id, source_mask_id, source_variant = key
        base_key = (scene_id, split, window_id, frame_id, source_mask_id)
        if source_variant == "B0_local_only":
            base_has_b0.add(base_key)
        if _is_v91_best_variant(source_variant):
            base_has_v91.add(base_key)
    source_container_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    for key, row in sorted(containers.items()):
        scene_id, split, window_id, frame_id, source_mask_id, source_variant = key
        base_key = (scene_id, split, window_id, frame_id, source_mask_id)
        artifacts = sorted(_rel(path) for path in row.pop("source_artifact_set", set()))
        row["source_artifacts_all"] = ";".join(artifacts)
        row["available_in_B0"] = base_key in base_has_b0
        row["available_in_v91_best"] = base_key in base_has_v91
        source_container_rows.append(row)
        feature_common = dict(row)
        feature_common["schema_version"] = "stream4d_v92_phase1_container_feature_availability_v1"
        feature_rows.append(
            {
                **{field: feature_common.get(field, "") for field in COMMON_FIELDS},
                "frame_id": frame_id,
                "source_mask_id": source_mask_id,
                "source_variant": source_variant,
                "has_d4rt_support": row.get("has_d4rt_support", False),
                "d4rt_support_row_count": row.get("d4rt_support_row_count", ""),
                "d4rt_unique_carrier_count": row.get("d4rt_unique_carrier_count", ""),
                "d4rt_confidence_mean": row.get("d4rt_confidence_mean", ""),
                "d4rt_visibility_mean": row.get("d4rt_visibility_mean", ""),
                "d4rt_density_mean": row.get("d4rt_density_mean", ""),
                "has_dino_feature": row.get("has_dino_feature", False),
                "dino_feature_sha256": row.get("dino_feature_sha256", ""),
                "dino_feature_backend": row.get("dino_feature_backend", ""),
                "has_radio_feature": row.get("has_radio_feature", False),
                "radio_feature_sha256": row.get("radio_feature_sha256", ""),
                "radio_feature_backend": row.get("radio_feature_backend", ""),
                "has_region_feature": row.get("has_region_feature", False),
                "region_feature_status": "missing_region_store_mask_level_only",
            }
        )
        risk_common = dict(row)
        risk_common["schema_version"] = "stream4d_v92_phase1_container_risk_v1"
        risk_rows.append(
            {
                **{field: risk_common.get(field, "") for field in COMMON_FIELDS},
                "frame_id": frame_id,
                "source_mask_id": source_mask_id,
                "source_variant": source_variant,
                "broad_mask_flag": row.get("broad_mask_flag", False),
                "background_risk_score": row.get("background_risk_score", ""),
                "underseg_risk_score": row.get("underseg_risk_score", ""),
                "mask_area_px": row.get("mask_area_px", ""),
                "mask_area_ratio": row.get("mask_area_ratio", ""),
                "has_region_feature": row.get("has_region_feature", False),
                "risk_source": "source_broad_flag_and_area_proxy",
            }
        )

    object_rows: list[dict[str, Any]] = []
    for key, state in sorted(objects.items()):
        scene_id, split, window_id, object_id, source_variant = key
        artifacts = sorted(_rel(path) for path in state["source_artifact_set"])
        source_artifact = Path(artifacts[0]) if artifacts else ADAPTER_ROWS
        object_rows.append(
            {
                **_common(
                    schema_version="stream4d_v92_phase1_object_hypothesis_v1",
                    variant_id=source_variant,
                    scene_id=scene_id,
                    split=split,
                    window_id=window_id,
                    chunk_id="",
                    uses_gt_for_prediction=state["uses_gt_for_prediction"],
                    uses_future=state["uses_future"],
                    uses_rgbd_pose_mesh=state["uses_rgbd_pose_mesh"],
                    source_artifact=ROOT / source_artifact,
                    source_artifact_sha256=";".join(_artifact_sha(ROOT / path, artifact_hashes) for path in artifacts if (ROOT / path).exists()),
                    created_at=created_at,
                ),
                "object_hypothesis_id": object_id,
                "source_variant": source_variant,
                "local_cluster_id": state.get("local_cluster_id", ""),
                "history_id": state.get("history_id", ""),
                "tracklet_id": state.get("tracklet_id", ""),
                "object_score": state.get("object_score", ""),
                "object_scale": state.get("object_scale", ""),
                "object_family": "v91_best_family" if _is_v91_best_variant(source_variant) else "source_registry_family",
                "carrier_count": state.get("carrier_count", 0),
                "frame_support_count": len(state["frames"]),
                "source_container_count": len(state["containers"]),
                "risk_score": state.get("risk_score", ""),
                "hard_negative_density": "",
                "semantic_proto": state.get("semantic_proto", ""),
                "appearance_feature_hash": "",
                "source_artifacts_all": ";".join(artifacts),
            }
        )

    serious_join_failures = [
        row
        for row in join_failure_rows
        if row.get("failure_type") in {"missing_window_key", "missing_d4rt_support_key", "missing_dino_feature_key", "missing_radio_feature_key"}
    ]
    join_failure_rate = len(serious_join_failures) / max(1, len(source_container_rows))
    failure_counter = Counter(row.get("failure_type", "") for row in join_failure_rows)
    variant_counter = Counter(row.get("variant_id", "") for row in source_container_rows)
    feature_variant_counter: dict[str, Counter[str]] = defaultdict(Counter)
    for row in feature_rows:
        variant_id = str(row.get("variant_id", ""))
        for field in ["has_d4rt_support", "has_dino_feature", "has_radio_feature", "has_region_feature"]:
            if _bool(row.get(field)):
                feature_variant_counter[variant_id][field] += 1
    uses_gt_count = sum(1 for row in source_container_rows if _bool(row.get("uses_gt_for_prediction")))
    uses_future_count = sum(1 for row in source_container_rows if _bool(row.get("uses_future")))
    phase1_pass_conditions = {
        "source_container_count_gt_0": len(source_container_rows) > 0,
        "object_hypothesis_count_gt_0": len(object_rows) > 0,
        "container_feature_availability_rows_have_d4rt_dino_radio_fields": bool(feature_rows)
        and all(field in feature_rows[0] for field in ["has_d4rt_support", "has_dino_feature", "has_radio_feature"]),
        "join_failure_rate_lte_0p02": join_failure_rate <= 0.02,
        "uses_gt_for_prediction_count_eq_0": uses_gt_count == 0,
        "uses_future_count_eq_0": uses_future_count == 0,
    }
    phase1_pass = all(phase1_pass_conditions.values())

    variant_config_rows: list[dict[str, Any]] = []
    variant_metric_rows: list[dict[str, Any]] = []
    variant_gate_rows: list[dict[str, Any]] = []
    for variant_id, count in sorted(variant_counter.items()):
        feature_counts = feature_variant_counter.get(variant_id, Counter())
        common = _common(
            schema_version="stream4d_v92_phase1_variant_config_v1",
            variant_id=variant_id,
            scene_id="ALL_DEV",
            split="dev",
            window_id="ALL_WINDOWS",
            chunk_id="",
            uses_gt_for_prediction=False,
            uses_future=False,
            uses_rgbd_pose_mesh=False,
            source_artifact=ADAPTER_ROWS,
            source_artifact_sha256=_artifact_sha(ADAPTER_ROWS, artifact_hashes),
            created_at=created_at,
        )
        variant_config_rows.append(
            {
                **common,
                "source_container_registry_mode": "diagnostic_join_only",
                "source_container_count": count,
                "uses_generated_masks_as_source_containers": False,
                "region_feature_store_available": False,
            }
        )
        metric_common = dict(common)
        metric_common["schema_version"] = "stream4d_v92_phase1_variant_metric_v1"
        variant_metric_rows.append(
            {
                **metric_common,
                "source_container_count": count,
                "d4rt_available_count": feature_counts["has_d4rt_support"],
                "dino_available_count": feature_counts["has_dino_feature"],
                "radio_available_count": feature_counts["has_radio_feature"],
                "region_feature_available_count": feature_counts["has_region_feature"],
                "d4rt_available_rate": _safe_div(feature_counts["has_d4rt_support"], count),
                "dino_available_rate": _safe_div(feature_counts["has_dino_feature"], count),
                "radio_available_rate": _safe_div(feature_counts["has_radio_feature"], count),
                "region_feature_available_rate": _safe_div(feature_counts["has_region_feature"], count),
            }
        )
    for gate, passed in phase1_pass_conditions.items():
        variant_gate_rows.append(
            {
                **_common(
                    schema_version="stream4d_v92_phase1_variant_gate_v1",
                    variant_id="ALL",
                    scene_id="ALL_DEV",
                    split="dev",
                    window_id="ALL_WINDOWS",
                    chunk_id="",
                    uses_gt_for_prediction=False,
                    uses_future=False,
                    uses_rgbd_pose_mesh=False,
                    source_artifact=ADAPTER_ROWS,
                    source_artifact_sha256=_artifact_sha(ADAPTER_ROWS, artifact_hashes),
                    created_at=created_at,
                ),
                "gate_name": gate,
                "gate_pass": bool(passed),
                "gate_value": {
                    "join_failure_rate_lte_0p02": join_failure_rate,
                    "uses_gt_for_prediction_count_eq_0": uses_gt_count,
                    "uses_future_count_eq_0": uses_future_count,
                }.get(gate, ""),
            }
        )

    variant_failure_rows = []
    if not phase1_pass:
        for failure_type, count in failure_counter.most_common():
            variant_failure_rows.append(
                {
                    **_common(
                        schema_version="stream4d_v92_phase1_variant_failure_v1",
                        variant_id="ALL",
                        scene_id="ALL_DEV",
                        split="dev",
                        window_id="ALL_WINDOWS",
                        chunk_id="",
                        uses_gt_for_prediction=False,
                        uses_future=False,
                        uses_rgbd_pose_mesh=False,
                        source_artifact=ADAPTER_ROWS,
                        source_artifact_sha256=_artifact_sha(ADAPTER_ROWS, artifact_hashes),
                        created_at=created_at,
                    ),
                    "failure_type": failure_type,
                    "failure_count": count,
                    "repair_direction": "check key naming and rebuild feature store if DINO/RADIO join failures persist",
                }
            )
    casebook_rows = [
        {
            **_common(
                schema_version="stream4d_v92_phase1_casebook_v1",
                variant_id="ALL",
                scene_id="ALL_DEV",
                split="dev",
                window_id="ALL_WINDOWS",
                chunk_id="",
                uses_gt_for_prediction=False,
                uses_future=False,
                uses_rgbd_pose_mesh=False,
                source_artifact=ADAPTER_ROWS,
                source_artifact_sha256=_artifact_sha(ADAPTER_ROWS, artifact_hashes),
                created_at=created_at,
            ),
            "case_type": "registry_boundary",
            "evidence": "source containers are observation containers; v91 generated masks are not reinterpreted as source containers",
            "region_feature_status": "mask_level_DINO_RADIO_only_no_source_internal_region_store",
        }
    ]

    for row in join_failure_rows:
        row["failure_rank"] = ""
    for rank, row in enumerate(join_failure_rows[:100], 1):
        row["failure_rank"] = rank

    output_files = {
        "source_container_rows.csv": source_container_rows,
        "object_hypothesis_rows.csv": object_rows,
        "object_container_link_rows.csv": link_rows,
        "container_feature_availability_rows.csv": feature_rows,
        "container_risk_rows.csv": risk_rows,
        "join_failure_rows.csv": join_failure_rows,
        "variant_config_rows.csv": variant_config_rows,
        "variant_metric_rows.csv": variant_metric_rows,
        "variant_gate_rows.csv": variant_gate_rows,
        "variant_failure_rows.csv": variant_failure_rows,
        "casebook_rows.csv": casebook_rows,
    }
    field_order = {
        "source_container_rows.csv": COMMON_FIELDS
        + [
            "frame_id",
            "source_mask_id",
            "source_variant",
            "mask_path",
            "mask_path_exists",
            "mask_area_px",
            "image_area_px",
            "mask_area_ratio",
            "mask_bbox_x0",
            "mask_bbox_y0",
            "mask_bbox_x1",
            "mask_bbox_y1",
            "source_mask_score",
            "cropformer_confidence",
            "broad_mask_flag",
            "background_risk_score",
            "underseg_risk_score",
            "available_in_B0",
            "available_in_v91_best",
            "has_d4rt_support",
            "has_dino_feature",
            "has_radio_feature",
            "has_region_feature",
            "d4rt_support_row_count",
            "d4rt_unique_carrier_count",
            "d4rt_confidence_mean",
            "d4rt_visibility_mean",
            "d4rt_density_mean",
            "dino_feature_sha256",
            "dino_feature_backend",
            "radio_feature_sha256",
            "radio_feature_backend",
            "source_row_count",
            "source_artifacts_all",
        ],
        "object_hypothesis_rows.csv": COMMON_FIELDS
        + [
            "object_hypothesis_id",
            "source_variant",
            "local_cluster_id",
            "history_id",
            "tracklet_id",
            "object_score",
            "object_scale",
            "object_family",
            "carrier_count",
            "frame_support_count",
            "source_container_count",
            "risk_score",
            "hard_negative_density",
            "semantic_proto",
            "appearance_feature_hash",
            "source_artifacts_all",
        ],
        "object_container_link_rows.csv": COMMON_FIELDS
        + [
            "object_hypothesis_id",
            "frame_id",
            "source_mask_id",
            "link_role",
            "adapter_precision",
            "adapter_recall",
            "adapter_f1",
            "adapter_score_raw",
            "carrier_support_mass",
            "carrier_visible_mass",
            "mask_area_px",
            "mask_selected_by_variant",
            "mask_selected_score",
            "selection_reason",
        ],
        "join_failure_rows.csv": COMMON_FIELDS
        + [
            "failure_rank",
            "failure_type",
            "frame_id",
            "source_mask_id",
            "join_key",
            "expected_count",
            "observed_count",
        ],
    }
    for filename, rows in output_files.items():
        _write_csv(OUT / filename, rows, field_order.get(filename))

    summary = {
        "schema": "stream4d_v92_phase1_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "decision": "PASS_V92_PHASE1_REGISTRY" if phase1_pass else "BLOCK_V92_PHASE1_REGISTRY_JOIN",
        "phase1_pass": phase1_pass,
        "phase1_pass_conditions": phase1_pass_conditions,
        "source_container_count": len(source_container_rows),
        "object_hypothesis_count": len(object_rows),
        "object_container_link_count": len(link_rows),
        "unique_scene_frame_mask_count": len({(row["scene_id"], row["frame_id"], row["source_mask_id"]) for row in source_container_rows}),
        "d4rt_available_container_count": sum(1 for row in feature_rows if _bool(row.get("has_d4rt_support"))),
        "dino_available_container_count": sum(1 for row in feature_rows if _bool(row.get("has_dino_feature"))),
        "radio_available_container_count": sum(1 for row in feature_rows if _bool(row.get("has_radio_feature"))),
        "region_feature_available_container_count": sum(1 for row in feature_rows if _bool(row.get("has_region_feature"))),
        "region_feature_store_available": False,
        "region_feature_status": "current artifacts expose mask-level DINO/RADIO features only; source-internal region feature branch remains unavailable until rebuilt",
        "join_failure_count": len(serious_join_failures),
        "join_failure_rate": join_failure_rate,
        "all_join_failure_rows_count": len(join_failure_rows),
        "join_failure_type_counts": dict(failure_counter),
        "mask_raster_missing_count": failure_counter.get("missing_mask_raster_path", 0),
        "mask_label_missing_count": failure_counter.get("missing_mask_label_in_raster", 0),
        "uses_gt_for_prediction_count": uses_gt_count,
        "uses_future_count": uses_future_count,
        "uses_rgbd_pose_mesh_count": sum(1 for row in source_container_rows if _bool(row.get("uses_rgbd_pose_mesh"))),
        "source_artifact_row_counts": dict(source_artifact_row_counts),
        "input_artifacts": {
            _rel(path): _artifact_sha(path, artifact_hashes)
            for path in [ADAPTER_ROWS, V90_MV_OBJECT_ROWS, V90_MV_FRAME_ROWS, WINDOW_ROWS, D4RT_SUPPORT_ROWS, DINO_MASK_ROWS, RADIO_INDEX_ROWS, *RADIO_SCENE_ROWS, *V91_SOURCE_SELECTIONS]
            if path.exists()
        },
        "duration_sec": time.time() - started,
        "created_at": created_at,
    }
    _write_json(OUT / "summary.json", summary)
    sha_rows = {path.name: _sha256(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "SHA256SUMS.json"}
    _write_json(OUT / "SHA256SUMS.json", sha_rows)
    return summary


if __name__ == "__main__":
    result = run()
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
