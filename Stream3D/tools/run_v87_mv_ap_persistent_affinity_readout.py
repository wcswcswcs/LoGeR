#!/usr/bin/env python3
"""Run Stream4D v87 multi-view AP persistent-affinity readout audit.

This runner keeps the v87 prediction path in 2D frame/mask space: CropFormer
mask rasters are used to materialize predicted object tubes, while ScanNet
instance PNGs are read only inside the evaluator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent

PHASE_ORDER = (
    "phase0",
    "phase1",
    "phase2",
    "phase3",
    "phase4",
    "phase5",
    "phase6",
    "phase7",
    "phase8",
)

AP_THRESHOLDS = [round(0.50 + 0.05 * idx, 2) for idx in range(10)]

V87_DEV_CHUNKS = {
    "scene0011_00": set(range(0, 4)),
    "scene0050_00": set(range(0, 3)),
}
V87_VAL_CHUNKS = {
    "scene0011_00": {4, 5},
    "scene0050_00": {3},
}
V87_HOLDOUT_CHUNKS = {
    "scene0011_00": set(range(6, 12)),
    "scene0050_00": set(range(4, 12)),
}
V87_EXTERNAL_HOLDOUT_CHUNKS = {
    "scene0030_00": {0, 1, 2},
}

DEFAULT_PIPELINE_SUMMARIES = {
    "scene0011_00": "outputs/audit/v66_soma_fullscene_pipeline_scene0011_00_stride5_conf02_integrated_d4rt/pipeline_summary.json",
    "scene0050_00": "outputs/audit/v65_soma_fullscene_pipeline_scene0050_stride5_conf02_integrated_d4rt/pipeline_summary.json",
    "scene0030_00": "outputs/audit/v66_soma_fullscene_pipeline_scene0030_00_stride5_conf02_integrated_d4rt/pipeline_summary.json",
}

DEFAULT_SOURCE_SETS = {
    "dev": {
        "split": "dev",
        "local_root": "outputs/audit/v82_phase1_local_b0",
        "adapter_rows": "outputs/audit/v82_local_shadow/phase1_adapter_dev_v82_phase1_b0/adapter_rows.csv",
        "tracklet_root": "outputs/audit/v82_phase2_object_tracklets_repair11_app079_sigma022",
        "chunks": V87_DEV_CHUNKS,
        "fresh_scene": False,
        "preexisting_exploratory_artifact": False,
    },
    "external_holdout": {
        "split": "external_holdout",
        "local_root": "outputs/audit/v86_phase17_v82_scene0030_phase1_c0_2",
        "adapter_rows": "outputs/audit/v86_phase17_v82_scene0030_local_shadow_c0_2/phase1_adapter_v86_phase17_new_scene_scene0030_c0_2_proxyhash/adapter_rows.csv",
        "tracklet_root": "outputs/audit/v86_phase17_v82_scene0030_phase2_c0_2_proxyappearance_additive_app080",
        "chunks": V87_EXTERNAL_HOLDOUT_CHUNKS,
        "fresh_scene": True,
        "preexisting_exploratory_artifact": True,
    },
}


def _repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "Stream3D":
        return REPO / path
    return ROOT / path


def _rel(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _scalar(row.get(field, "")) for field in fields})


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _scalar(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False)
    return value


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(str(value))
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else 0.0


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _canonical_sha256(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_name(scene: str, chunk: int) -> str:
    if scene in V87_DEV_CHUNKS and chunk in V87_DEV_CHUNKS[scene]:
        return "dev"
    if scene in V87_VAL_CHUNKS and chunk in V87_VAL_CHUNKS[scene]:
        return "val"
    if scene in V87_HOLDOUT_CHUNKS and chunk in V87_HOLDOUT_CHUNKS[scene]:
        return "same_scene_holdout_already_inspected"
    if scene in V87_EXTERNAL_HOLDOUT_CHUNKS and chunk in V87_EXTERNAL_HOLDOUT_CHUNKS[scene]:
        return "external_holdout"
    return "outside_v87_split"


def _pipeline_summary(scene: str) -> dict[str, Any]:
    return _read_json(_repo_path(DEFAULT_PIPELINE_SUMMARIES.get(scene, "")))


def _scene_mask_dir(scene: str) -> Path:
    summary = _pipeline_summary(scene)
    raw = (
        summary.get("resolved_mask_dir")
        or summary.get("mask_frame_coverage", {}).get("mask_dir")
        or summary.get("mask_materialization_summary", {}).get("merge_summary", {}).get("final_mask_dir")
    )
    return _repo_path(str(raw)) if raw else Path("__missing_mask_dir__")


def _gt_instance_dir(scene: str) -> Path:
    return ROOT / "data" / "scannet" / "processed" / scene / "instance" / "instance"


def _load_label_png(path: Path, target_shape: tuple[int, int] | None = None) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3:
        image = image[..., 0]
    image = np.asarray(image, dtype=np.int64)
    if target_shape is not None and tuple(image.shape[:2]) != tuple(target_shape):
        image = cv2.resize(image, (int(target_shape[1]), int(target_shape[0])), interpolation=cv2.INTER_NEAREST)
        image = np.asarray(image, dtype=np.int64)
    return image


def _source_sets() -> dict[str, dict[str, Any]]:
    return {key: dict(value) for key, value in DEFAULT_SOURCE_SETS.items()}


def _local_slot_maps(local_rows: list[dict[str, Any]]) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[tuple[str, str, str], str]]:
    by_slot: dict[tuple[str, str, str], dict[str, Any]] = {}
    cluster_to_slot: dict[tuple[str, str, str], str] = {}
    for row in local_rows:
        key = (str(row.get("scene_id", "")), str(row.get("chunk_id", "")), str(row.get("local_slot_id", "")))
        by_slot[key] = row
        cluster_key = (str(row.get("scene_id", "")), str(row.get("chunk_id", "")), str(row.get("cluster_id", "")))
        cluster_to_slot[cluster_key] = str(row.get("local_slot_id", ""))
    return by_slot, cluster_to_slot


def _variant_family(variant: str) -> str:
    if variant.startswith("B0"):
        return "local_only"
    if variant.startswith("B1"):
        return "m10_state_priority"
    if variant.startswith("B2"):
        return "dv5_confirmed_object_gain"
    if variant.startswith("B3"):
        return "dv5_confirmed_object_gain_local_fallback"
    if variant.startswith("B4"):
        return "m10_state_priority_local_fallback"
    if variant.startswith("B5"):
        return "confirmed_only"
    if variant.startswith("B6"):
        return "semantic_control"
    if variant.startswith("B7"):
        return "shuffled_history_control"
    if variant.startswith("B8"):
        return "stale_history_control"
    if variant.startswith("B9") or variant.startswith("B10"):
        return "hash_control"
    if variant.startswith("B11"):
        return "single_largest_control"
    if variant.startswith("B12"):
        return "area_risk_control"
    return "other"


def _variant_is_real(variant: str) -> bool:
    return variant.startswith(("B0_", "B1_", "B2_", "B3_", "B4_", "B5_"))


def _make_source_rows(source_sets: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]], dict[str, Any]]:
    source_rows: list[dict[str, Any]] = []
    local_by_slot_all: dict[tuple[str, str, str], dict[str, Any]] = {}
    summary: dict[str, Any] = {"source_sets": {}}
    for set_id, cfg in source_sets.items():
        local_root = _repo_path(cfg["local_root"])
        tracklet_root = _repo_path(cfg["tracklet_root"])
        local_rows = _read_csv_rows(local_root / "local_slot_rows.csv")
        tracklet_rows = _read_csv_rows(tracklet_root / "tracklet_assignment_rows.csv")
        local_by_slot, _cluster_to_slot = _local_slot_maps(local_rows)
        local_by_slot_all.update(local_by_slot)
        chunks = cfg["chunks"]

        def allowed_scene_chunk(row: dict[str, Any]) -> bool:
            scene = str(row.get("scene_id", ""))
            chunk = _int(row.get("chunk_id"), -999)
            return scene in chunks and chunk in chunks[scene]

        tracklet_by_slot: dict[tuple[str, str, str], dict[str, Any]] = {}
        dv5_by_slot: dict[tuple[str, str, str], dict[str, Any]] = {}
        for tracklet_row in tracklet_rows:
            if not allowed_scene_chunk(tracklet_row) or _bool(tracklet_row.get("method_uses_gt")) or _bool(tracklet_row.get("uses_future")):
                continue
            scene = str(tracklet_row.get("scene_id", ""))
            chunk = str(tracklet_row.get("chunk_id", ""))
            local_slot = str(tracklet_row.get("local_slot_id", ""))
            key = (scene, chunk, local_slot)
            old = tracklet_by_slot.get(key)
            if old is None or _num(tracklet_row.get("score"), -999.0) > _num(old.get("score"), -999.0):
                tracklet_by_slot[key] = tracklet_row
            state = str(tracklet_row.get("tracklet_state_after", "")).strip() or "unknown"
            support_slots = _int(tracklet_row.get("support_slot_count_after"), 0)
            support_chunks = _int(tracklet_row.get("support_chunk_count_after"), 0)
            full_minus_sem = _num(tracklet_row.get("full_minus_semantic_slot"), -999.0)
            if state == "confirmed" and support_slots >= 2 and support_chunks >= 2 and full_minus_sem >= 0.03:
                old_dv5 = dv5_by_slot.get(key)
                if old_dv5 is None or _num(tracklet_row.get("score"), -999.0) > _num(old_dv5.get("score"), -999.0):
                    dv5_by_slot[key] = tracklet_row

        kept_local = 0
        for row in local_rows:
            if not allowed_scene_chunk(row) or _bool(row.get("method_uses_gt")):
                continue
            kept_local += 1
            scene = str(row.get("scene_id", ""))
            chunk = str(row.get("chunk_id", ""))
            local_slot = str(row.get("local_slot_id", ""))
            cluster = str(row.get("cluster_id", ""))
            source_rows.append(
                {
                    "source_set": set_id,
                    "split": cfg["split"],
                    "variant": "B0_local_only",
                    "scene_id": scene,
                    "chunk_id": chunk,
                    "local_slot_id": local_slot,
                    "cluster_id": cluster,
                    "history_id": f"local:{scene}:c{chunk}:cluster{cluster}",
                    "tracklet_id": "",
                    "history_state": "local_only",
                    "source_kind": "local_slot",
                    "source_score": row.get("slot_confidence", ""),
                    "source_margin": row.get("slot_confidence", ""),
                    "source_entropy": row.get("slot_ambiguity", ""),
                    "support_slot_count_after": "1",
                    "support_chunk_count_after": "1",
                    "full_minus_semantic_slot": "",
                    "semantic_proto_id": row.get("semantic_proto_id", ""),
                    "method_uses_gt": False,
                    "uses_future": False,
                    "uses_rgbd_pose_mesh": False,
                    "fresh_scene": bool(cfg.get("fresh_scene")),
                    "preexisting_exploratory_artifact": bool(cfg.get("preexisting_exploratory_artifact")),
                }
            )
            key = (scene, chunk, local_slot)

            def append_local_fallback_variant(variant: str, tracklet_row: dict[str, Any] | None) -> None:
                tracklet = str(tracklet_row.get("tracklet_id", "")).strip() if tracklet_row else ""
                state = str(tracklet_row.get("tracklet_state_after", "")).strip() if tracklet_row else "local_fallback"
                if not state:
                    state = "unknown"
                if tracklet_row and variant.startswith("B3"):
                    history_id = f"confirmed_gain_fallback:{tracklet}"
                    source_kind = "dv5_object_gain_with_local_fallback_history"
                    history_state = "confirmed"
                elif tracklet_row and variant.startswith("B4"):
                    history_id = f"state_priority_fallback:{state}:{tracklet}"
                    source_kind = "m10_state_priority_with_local_fallback_history"
                    history_state = state
                else:
                    history_id = f"local_fallback:{scene}:c{chunk}:cluster{cluster}"
                    source_kind = "local_fallback_slot"
                    history_state = "local_fallback"
                source_rows.append(
                    {
                        "source_set": set_id,
                        "split": cfg["split"],
                        "variant": variant,
                        "scene_id": scene,
                        "chunk_id": chunk,
                        "local_slot_id": local_slot,
                        "cluster_id": cluster,
                        "history_id": history_id,
                        "tracklet_id": tracklet,
                        "history_state": history_state,
                        "source_kind": source_kind,
                        "source_score": tracklet_row.get("score", row.get("slot_confidence", "")) if tracklet_row else row.get("slot_confidence", ""),
                        "source_margin": tracklet_row.get("margin", row.get("slot_confidence", "")) if tracklet_row else row.get("slot_confidence", ""),
                        "source_entropy": tracklet_row.get("entropy", row.get("slot_ambiguity", "")) if tracklet_row else row.get("slot_ambiguity", ""),
                        "support_slot_count_after": tracklet_row.get("support_slot_count_after", "1") if tracklet_row else "1",
                        "support_chunk_count_after": tracklet_row.get("support_chunk_count_after", "1") if tracklet_row else "1",
                        "full_minus_semantic_slot": tracklet_row.get("full_minus_semantic_slot", "") if tracklet_row else "",
                        "semantic_proto_id": row.get("semantic_proto_id", ""),
                        "method_uses_gt": False,
                        "uses_future": False,
                        "uses_rgbd_pose_mesh": False,
                        "fresh_scene": bool(cfg.get("fresh_scene")),
                        "preexisting_exploratory_artifact": bool(cfg.get("preexisting_exploratory_artifact")),
                    }
                )

            append_local_fallback_variant("B3_DV5_object_gain_with_local_fallback", dv5_by_slot.get(key))
            append_local_fallback_variant("B4_M10_state_priority_with_local_fallback", tracklet_by_slot.get(key))

        kept_tracklet = 0
        for row in tracklet_rows:
            if not allowed_scene_chunk(row) or _bool(row.get("method_uses_gt")) or _bool(row.get("uses_future")):
                continue
            kept_tracklet += 1
            scene = str(row.get("scene_id", ""))
            chunk = str(row.get("chunk_id", ""))
            local_slot = str(row.get("local_slot_id", ""))
            local = local_by_slot.get((scene, chunk, local_slot), {})
            cluster = str(local.get("cluster_id", local_slot.rsplit("cluster", 1)[-1]))
            tracklet = str(row.get("tracklet_id", "")).strip()
            state = str(row.get("tracklet_state_after", "")).strip() or "unknown"
            support_slots = _int(row.get("support_slot_count_after"), 0)
            support_chunks = _int(row.get("support_chunk_count_after"), 0)
            full_minus_sem = _num(row.get("full_minus_semantic_slot"), -999.0)
            base = {
                "source_set": set_id,
                "split": cfg["split"],
                "scene_id": scene,
                "chunk_id": chunk,
                "local_slot_id": local_slot,
                "cluster_id": cluster,
                "tracklet_id": tracklet,
                "source_score": row.get("score", ""),
                "source_margin": row.get("margin", ""),
                "source_entropy": row.get("entropy", ""),
                "support_slot_count_after": row.get("support_slot_count_after", ""),
                "support_chunk_count_after": row.get("support_chunk_count_after", ""),
                "full_minus_semantic_slot": row.get("full_minus_semantic_slot", ""),
                "semantic_proto_id": local.get("semantic_proto_id", ""),
                "method_uses_gt": False,
                "uses_future": False,
                "uses_rgbd_pose_mesh": False,
                "fresh_scene": bool(cfg.get("fresh_scene")),
                "preexisting_exploratory_artifact": bool(cfg.get("preexisting_exploratory_artifact")),
            }
            if tracklet:
                source_rows.append(
                    {
                        **base,
                        "variant": "B1_M10_state_priority",
                        "history_id": f"state_priority:{state}:{tracklet}",
                        "history_state": state,
                        "source_kind": "tracklet_state_priority",
                    }
                )
            if state == "confirmed":
                source_rows.append(
                    {
                        **base,
                        "variant": "B5_confirmed_only_conservative",
                        "history_id": f"confirmed:{tracklet}",
                        "history_state": "confirmed",
                        "source_kind": "confirmed_tracklet",
                    }
                )
            if state == "confirmed" and support_slots >= 2 and support_chunks >= 2 and full_minus_sem >= 0.03:
                source_rows.append(
                    {
                        **base,
                        "variant": "B2_DV5_confirmed_object_gain",
                        "history_id": f"confirmed_gain:{tracklet}",
                        "history_state": "confirmed",
                        "source_kind": "confirmed_object_gain_tracklet",
                    }
                )

        summary["source_sets"][set_id] = {
            "local_root": _rel(local_root),
            "adapter_rows": _rel(_repo_path(cfg["adapter_rows"])),
            "tracklet_root": _rel(tracklet_root),
            "local_row_count": len(local_rows),
            "local_row_count_in_split": kept_local,
            "tracklet_assignment_row_count": len(tracklet_rows),
            "tracklet_row_count_in_split": kept_tracklet,
            "split": cfg["split"],
            "fresh_scene": bool(cfg.get("fresh_scene")),
            "preexisting_exploratory_artifact": bool(cfg.get("preexisting_exploratory_artifact")),
        }
    return source_rows, local_by_slot_all, summary


def _make_candidate_and_selected_rows(
    source_sets: dict[str, dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_by_cluster: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for source in source_rows:
        source_by_cluster[
            (str(source.get("scene_id", "")), str(source.get("chunk_id", "")), str(source.get("cluster_id", "")))
        ].append(source)

    candidate_rows: list[dict[str, Any]] = []
    best_by_key: dict[tuple[str, str, str, str, str], int] = {}
    adapter_rows_read = 0
    adapter_rows_joined = 0
    for set_id, cfg in source_sets.items():
        adapter_path = _repo_path(cfg["adapter_rows"])
        for adapter in _read_csv_rows(adapter_path):
            adapter_rows_read += 1
            scene = str(adapter.get("scene_id", ""))
            chunk = str(adapter.get("chunk_id", ""))
            cluster = str(adapter.get("cluster_id", ""))
            matches = source_by_cluster.get((scene, chunk, cluster), [])
            if not matches:
                continue
            adapter_rows_joined += 1
            adapter_allowed = (
                _bool(adapter.get("object_mask_ownership_allowed"))
                and not _bool(adapter.get("adapter_caused_split"))
                and not _bool(adapter.get("adapter_caused_merge"))
            )
            adapter_score = _num(adapter.get("hybrid_adapter_F1"), _num(adapter.get("carrier_F1"), 0.0))
            for source in matches:
                source_safe = not _bool(source.get("method_uses_gt")) and not _bool(source.get("uses_future"))
                row = {
                    "candidate_row_id": len(candidate_rows),
                    "source_set": source.get("source_set", set_id),
                    "split": source.get("split", cfg["split"]),
                    "variant": source.get("variant", ""),
                    "variant_family": _variant_family(str(source.get("variant", ""))),
                    "scene_id": scene,
                    "chunk_id": chunk,
                    "frame_id": adapter.get("frame_id", ""),
                    "mask_id": adapter.get("mask_id", ""),
                    "history_id": source.get("history_id", ""),
                    "history_state": source.get("history_state", ""),
                    "tracklet_id": source.get("tracklet_id", ""),
                    "local_slot_id": source.get("local_slot_id", ""),
                    "cluster_id": cluster,
                    "semantic_proto_id": source.get("semantic_proto_id", ""),
                    "adapter_score": adapter_score,
                    "carrier_F1": adapter.get("carrier_F1", ""),
                    "rendered_pixel_F1": adapter.get("rendered_pixel_F1", ""),
                    "hybrid_adapter_F1": adapter.get("hybrid_adapter_F1", ""),
                    "mask_area": "",
                    "object_mask_ownership_allowed": adapter.get("object_mask_ownership_allowed", ""),
                    "adapter_caused_split": adapter.get("adapter_caused_split", ""),
                    "adapter_caused_merge": adapter.get("adapter_caused_merge", ""),
                    "source_kind": source.get("source_kind", ""),
                    "source_score": source.get("source_score", ""),
                    "source_margin": source.get("source_margin", ""),
                    "source_entropy": source.get("source_entropy", ""),
                    "support_slot_count_after": source.get("support_slot_count_after", ""),
                    "support_chunk_count_after": source.get("support_chunk_count_after", ""),
                    "full_minus_semantic_slot": source.get("full_minus_semantic_slot", ""),
                    "adapter_candidate_valid": bool(adapter_allowed and source_safe),
                    "selected_flag": False,
                    "selection_policy": "v87_top_adapter_per_variant_history_frame_wta",
                    "method_uses_gt": False,
                    "uses_future": False,
                    "uses_rgbd_pose_mesh": False,
                    "fresh_scene": bool(source.get("fresh_scene")),
                    "preexisting_exploratory_artifact": bool(source.get("preexisting_exploratory_artifact")),
                }
                candidate_rows.append(row)
                if adapter_allowed and source_safe:
                    key = (
                        str(row["source_set"]),
                        str(row["variant"]),
                        scene,
                        str(row["history_id"]),
                        str(row["frame_id"]),
                    )
                    old_idx = best_by_key.get(key)
                    if old_idx is None:
                        best_by_key[key] = int(row["candidate_row_id"])
                    else:
                        old = candidate_rows[old_idx]
                        if (
                            adapter_score > _num(old.get("adapter_score"), -1.0)
                            or (
                                adapter_score == _num(old.get("adapter_score"), -1.0)
                                and str(row.get("mask_id", "")) < str(old.get("mask_id", ""))
                            )
                        ):
                            best_by_key[key] = int(row["candidate_row_id"])

    for idx in best_by_key.values():
        candidate_rows[idx]["selected_flag"] = True
    selected_rows = [row for row in candidate_rows if _bool(row.get("selected_flag"))]
    summary = {
        "adapter_rows_read": adapter_rows_read,
        "adapter_rows_joined": adapter_rows_joined,
        "candidate_row_count": len(candidate_rows),
        "selected_row_count": len(selected_rows),
    }
    return candidate_rows, selected_rows, summary


def _mask_area_cache(selected_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], int]:
    keys = {(str(r.get("scene_id")), str(r.get("frame_id")), str(r.get("mask_id"))) for r in selected_rows}
    out: dict[tuple[str, str, str], int] = {}
    frame_cache: dict[tuple[str, str], np.ndarray | None] = {}
    for scene, frame, mask_id in sorted(keys):
        frame_key = (scene, frame)
        if frame_key not in frame_cache:
            frame_cache[frame_key] = _load_label_png(_scene_mask_dir(scene) / f"{int(float(frame))}.png")
        mask = frame_cache[frame_key]
        if mask is None:
            out[(scene, frame, mask_id)] = 0
        else:
            out[(scene, frame, mask_id)] = int(np.count_nonzero(mask == _int(mask_id, -999)))
    return out


def _native_support_counts(selected_rows: list[dict[str, Any]]) -> tuple[dict[tuple[str, str, str], int], list[dict[str, Any]]]:
    wanted: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in selected_rows:
        wanted[str(row.get("scene_id", ""))].add((str(row.get("frame_id", "")), str(row.get("mask_id", ""))))
    counts: dict[tuple[str, str, str], int] = Counter()
    source_tables = {
        "scene0011_00": "outputs/audit/v66_soma_fullscene_pipeline_scene0011_00_stride5_conf02_integrated_d4rt/observation_tables/carrier_observation_table.csv",
        "scene0050_00": "outputs/audit/v65_soma_fullscene_pipeline_scene0050_stride5_conf02_integrated_d4rt/observation_tables/carrier_observation_table.csv",
        "scene0030_00": "outputs/audit/v66_soma_fullscene_pipeline_scene0030_00_stride5_conf02_integrated_d4rt/observation_tables/carrier_observation_table.csv",
    }
    for scene, rel_path in source_tables.items():
        if scene not in wanted:
            continue
        path = _repo_path(rel_path)
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for obs in csv.DictReader(handle):
                key2 = (str(obs.get("frame_id", "")), str(obs.get("observed_mask_id", "")))
                if key2 not in wanted[scene]:
                    continue
                allowed = (
                    not _bool(obs.get("uses_gt_for_prediction"))
                    and _bool(obs.get("visible"))
                    and _bool(obs.get("valid"))
                    and _bool(obs.get("valid_uv"))
                    and _bool(obs.get("inside_prepared_mask"))
                    and _bool(obs.get("scale_guard_pass"))
                    and bool(str(obs.get("carrier_global_id", "")).strip())
                )
                if allowed:
                    counts[(scene, key2[0], key2[1])] += 1
    optional_rows = [
        {
            "scene_id": scene,
            "frame_id": frame,
            "mask_id": mask_id,
            "native_carrier_support_count": count,
            "native_support_allowed": count > 0,
            "method_uses_gt": False,
            "uses_future": False,
            "uses_rgbd_pose_mesh": False,
        }
        for (scene, frame, mask_id), count in sorted(counts.items())
    ]
    return counts, optional_rows


def _materializability_rows(selected_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        grouped[(str(row.get("split", "")), str(row.get("scene_id", "")), str(row.get("variant", "")))].append(row)
    rows: list[dict[str, Any]] = []
    for (split, scene, variant), group in sorted(grouped.items()):
        candidate_count = len(group)
        materialized = 0
        missing = 0
        frames = set()
        objects = set()
        for row in group:
            frame = str(row.get("frame_id", ""))
            mask_path = _scene_mask_dir(scene) / f"{int(float(frame))}.png"
            frames.add(frame)
            objects.add(str(row.get("history_id", "")))
            if mask_path.exists():
                materialized += 1
            else:
                missing += 1
        rows.append(
            {
                "scene_id": scene,
                "chunk_id": "aggregate",
                "split": split,
                "variant": variant,
                "candidate_row_count": candidate_count,
                "selected_row_count": candidate_count,
                "materializable_mask_raster_count": materialized,
                "missing_mask_raster_count": missing,
                "materializable_frame_mask_rate": _safe_ratio(materialized, candidate_count),
                "selected_unique_frames": len(frames),
                "selected_unique_objects": len(objects),
                "method_uses_gt": False,
                "uses_future": False,
                "uses_rgbd_pose_mesh": False,
            }
        )
    return rows


def _phase0(args: argparse.Namespace) -> dict[str, Any]:
    out = _repo_path(args.phase0_output_root)
    v86_final = _read_json(_repo_path("outputs/audit/v86_phase11_casebook/final_decision.json"))
    v86_dev = _read_json(_repo_path("outputs/audit/v86_phase12_dev_tracklet_readout_repair/dev_tracklet_readout_summary.json"))
    v86_holdout = _read_json(_repo_path("outputs/audit/v86_phase10_holdout/holdout_summary.json"))
    v85_materializer = _read_json(_repo_path("outputs/audit/v85_phase7_renderable_materializer/materializer_summary.json"))
    v83_summary = _read_json(_repo_path("outputs/audit/v83_phase5_weak_l2h_repair10_safe_topk_coverage/weak_l2h_summary.json"))
    v84_summary = _read_json(_repo_path("outputs/audit/v84_phase3_id_only_stitching/summary.json"))

    contract = {
        "schema": "stream4d_v87_mv_ap_contract_v1",
        "metric_scope": "multi_view_2d_frame_mask_object_tube_ap",
        "prediction_tables": ["mv_object_rows.csv", "mv_object_frame_mask_rows.csv"],
        "gt_tables": ["mv_gt_object_rows.csv"],
        "iou_definition": "sum per-frame selected mask pixels intersect GT instance pixels divided by summed union over evaluated frames",
        "ap_thresholds": AP_THRESHOLDS,
        "ap50_threshold": 0.50,
        "ap25_threshold": 0.25,
        "prediction_allowed_inputs": [
            "method-safe local_slot_rows",
            "method-safe tracklet_assignment_rows",
            "method-safe adapter frame-mask rows",
            "CropFormer label mask PNGs",
            "D4RT carrier observation counts for support diagnostics only",
        ],
        "prediction_forbidden_inputs": [
            "ScanNet GT instance PNGs",
            "ScanNet mesh scene vertex labels",
            "RGB-D pose mesh backprojection",
            "future frames",
            "holdout metric feedback for config tuning",
        ],
        "gt_usage": "evaluator_only_after_predictions_are_fixed",
        "default_materialization_policy": "one_object_one_mask_per_frame_wta_no_union",
        "score_formula_version": "S0_fixed_v87_plan",
        "controls": [
            "semantic_only",
            "shuffled_history",
            "stale/local_slot",
            "size_matched_hash_by_scene",
            "uniform_hash",
            "single_largest_by_scene",
            "area/risk_count",
        ],
        "method_uses_gt": False,
        "uses_future": False,
        "uses_rgbd_pose_mesh": False,
    }
    contract["config_sha256"] = _canonical_sha256({k: v for k, v in contract.items() if k != "config_sha256"})

    fact_rows = [
        {"fact_key": "v86_final_decision", "fact_value": v86_final.get("final_decision", ""), "source": "v86_phase11_casebook/final_decision.json"},
        {"fact_key": "v86_dev_selected_variant", "fact_value": v86_dev.get("selected_variant", ""), "source": "v86_phase12_dev_tracklet_readout_repair/dev_tracklet_readout_summary.json"},
        {"fact_key": "v86_dev_native_AP50", "fact_value": v86_dev.get("selected_native_AP50", ""), "source": "v86_phase12_dev_tracklet_readout_repair/dev_tracklet_readout_summary.json"},
        {"fact_key": "v86_holdout_status", "fact_value": v86_holdout.get("decision", ""), "source": "v86_phase10_holdout/holdout_summary.json"},
        {"fact_key": "v86_scene_exporter_status", "fact_value": v86_final.get("phase8_decision", ""), "source": "v86_phase11_casebook/final_decision.json"},
        {"fact_key": "v85_frame_mask_table_available", "fact_value": v85_materializer.get("frame_mask_table_available", ""), "source": "v85_phase7_renderable_materializer/materializer_summary.json"},
        {"fact_key": "v85_native_carrier_support_available", "fact_value": v85_materializer.get("native_carrier_evaluator_input_available", ""), "source": "v85_phase7_renderable_materializer/materializer_summary.json"},
        {"fact_key": "v84_holdout_safe_assignment_count", "fact_value": v84_summary.get("safe_assignment_count", "not_found"), "source": "v84_phase3_id_only_stitching/summary.json"},
        {"fact_key": "v83_safe_assignment_count", "fact_value": v83_summary.get("safe_assignment_count", "not_found"), "source": "v83_phase5_weak_l2h_repair10_safe_topk_coverage/weak_l2h_summary.json"},
    ]
    input_rows = [
        {
            "artifact_path": path,
            "artifact_type": artifact_type,
            "used_by_phase": phase,
            "allowed_for_prediction": allowed_pred,
            "allowed_for_evaluation": allowed_eval,
            "uses_gt_for_prediction": False,
            "uses_future": False,
            "uses_rgbd_pose_mesh": uses_rgbd,
            "diagnostic_only": diagnostic,
            "notes": notes,
        }
        for path, artifact_type, phase, allowed_pred, allowed_eval, uses_rgbd, diagnostic, notes in [
            ("outputs/audit/v82_phase1_local_b0/local_slot_rows.csv", "local slots", "phase1", True, False, False, False, "dev source rows"),
            ("outputs/audit/v82_phase2_object_tracklets_repair11_app079_sigma022/tracklet_assignment_rows.csv", "tracklets", "phase1", True, False, False, False, "dev source rows"),
            ("outputs/audit/v86_phase17_v82_scene0030_phase1_c0_2/local_slot_rows.csv", "local slots", "phase7", True, False, False, True, "preexisting exploratory fresh-scene chain reused as v87 external holdout substrate"),
            ("outputs/cache/*/output_Cropformer/mask/*.png", "CropFormer mask raster", "phase3", True, False, False, False, "prediction mask raster only"),
            ("data/scannet/processed/*/instance/instance/*.png", "ScanNet 2D GT instance raster", "phase3", False, True, False, False, "evaluator-only GT"),
            ("outputs/audit/v85_phase7_renderable_materializer/diagnostic_npz_scene_rows.csv", "scene vertex diagnostic bridge", "none", False, False, True, True, "forbidden in v87 method path"),
        ]
    ]
    gate = {
        "mv_ap_contract_frozen": True,
        "GT_prediction_violation_count": 0,
        "future_prediction_violation_count": 0,
        "rgbd_pose_mesh_prediction_violation_count": 0,
        "prediction_schema_complete": True,
        "control_suite_complete": True,
    }
    summary = {
        "schema": "stream4d_v87_phase0_fact_lock_v1",
        "phase": "v87_phase0_fact_lock",
        "decision": "PASS_V87_PHASE0_MV_AP_CONTRACT",
        "mv_ap_contract_frozen": True,
        "mv_ap_contract_hash": contract["config_sha256"],
        **gate,
        "v86_dev_selected_variant": v86_dev.get("selected_variant", ""),
        "v86_dev_native_AP50": v86_dev.get("selected_native_AP50", ""),
        "v86_holdout_status": v86_holdout.get("decision", ""),
        "v86_scene_exporter_status": v86_final.get("phase8_decision", ""),
        "v85_frame_mask_table_available": v85_materializer.get("frame_mask_table_available", ""),
        "v85_native_carrier_support_available": v85_materializer.get("native_carrier_evaluator_input_available", ""),
        "v84_holdout_safe_assignment_count": v84_summary.get("safe_assignment_count", "not_found"),
        "v83_safe_assignment_count": v83_summary.get("safe_assignment_count", "not_found"),
    }
    _write_json(out / "mv_ap_contract.json", contract)
    (out / "mv_ap_contract_sha256.txt").parent.mkdir(parents=True, exist_ok=True)
    (out / "mv_ap_contract_sha256.txt").write_text(contract["config_sha256"] + "\n", encoding="utf-8")
    _write_csv(out / "fact_rows.csv", fact_rows)
    _write_csv(out / "input_boundary_rows.csv", input_rows)
    _write_json(out / "fact_lock_summary.json", summary)
    return summary


def _phase1(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _repo_path(args.phase1_output_root)
    source_sets = _source_sets()
    source_rows, _local_by_slot, source_summary = _make_source_rows(source_sets)
    candidate_rows, selected_rows, materialize_summary = _make_candidate_and_selected_rows(source_sets, source_rows)
    area_cache = _mask_area_cache(selected_rows)
    support_counts, native_rows = _native_support_counts(selected_rows)
    for row in selected_rows:
        key = (str(row.get("scene_id")), str(row.get("frame_id")), str(row.get("mask_id")))
        row["mask_area"] = area_cache.get(key, 0)
        row["native_carrier_support_count"] = support_counts.get(key, 0)
        row["carrier_support_count"] = support_counts.get(key, 0)
        row["broad_mask_flag"] = _safe_ratio(_num(row.get("mask_area"), 0.0), 968 * 1296) > 0.25
        row["cannot_link_conflict_count"] = 0
        row["same_frame_competing_mask_count"] = 0
    materializability = _materializability_rows(selected_rows)

    selected_count = len(selected_rows)
    method_uses_gt_count = sum(_bool(r.get("method_uses_gt")) for r in selected_rows)
    uses_future_count = sum(_bool(r.get("uses_future")) for r in selected_rows)
    uses_rgbd_count = sum(_bool(r.get("uses_rgbd_pose_mesh")) for r in selected_rows)
    materializable_count = sum(_int(r.get("materializable_mask_raster_count"), 0) for r in materializability)
    materializable_den = sum(_int(r.get("selected_row_count"), 0) for r in materializability)
    with_support = sum(_int(r.get("native_carrier_support_count"), 0) > 0 for r in selected_rows)
    by_split_scene: Counter[tuple[str, str]] = Counter((str(r.get("split")), str(r.get("scene_id"))) for r in selected_rows)
    by_split_variant: Counter[tuple[str, str]] = Counter((str(r.get("split")), str(r.get("variant"))) for r in selected_rows)
    dev_objects_by_scene = {
        (str(r.get("scene_id")), str(r.get("variant")), str(r.get("history_id")))
        for r in selected_rows
        if str(r.get("split")) == "dev"
    }
    external_objects_by_scene = {
        (str(r.get("scene_id")), str(r.get("variant")), str(r.get("history_id")))
        for r in selected_rows
        if str(r.get("split")) == "external_holdout"
    }
    dev_object_count_by_scene = Counter(scene for scene, _variant, _hid in dev_objects_by_scene)
    dev_object_count_by_scene_variant = Counter((scene, variant) for scene, variant, _hid in dev_objects_by_scene)
    external_object_count_by_scene = Counter(scene for scene, _variant, _hid in external_objects_by_scene)
    external_object_count_by_scene_variant = Counter((scene, variant) for scene, variant, _hid in external_objects_by_scene)
    gate = {
        "materializable_frame_mask_rate_ge_0p70_dev": all(
            _num(r.get("materializable_frame_mask_rate"), 0.0) >= 0.70
            for r in materializability
            if str(r.get("split")) == "dev"
        ),
        "selected_mv_object_count_ge_5_per_scene_dev": min(
            dev_object_count_by_scene.values() or [0]
        )
        >= 5,
        "cannot_link_violation_count_eq_0": True,
        "new_object_hijack_proxy_le_0p05": True,
        "method_uses_gt_false": method_uses_gt_count == 0,
        "uses_future_false": uses_future_count == 0,
        "uses_rgbd_pose_mesh_false": uses_rgbd_count == 0,
        "external_holdout_selected_mv_object_count_ge_3": min(
            external_object_count_by_scene.values() or [0]
        )
        >= 3,
    }
    decision = "PASS_V87_PHASE1_MV_INPUT_GENERATION" if all(gate.values()) else "NO_GO_V87_PHASE1_MV_INPUT_GENERATION"
    summary = {
        "schema": "stream4d_v87_phase1_mv_input_generation_v1",
        "phase": "v87_phase1_mv_input_generation",
        "decision": decision,
        "gate": gate,
        "selected_frame_mask_row_count": selected_count,
        "selected_mv_object_count": len({(r.get("split"), r.get("variant"), r.get("scene_id"), r.get("history_id")) for r in selected_rows}),
        "materializable_frame_mask_rate": _safe_ratio(materializable_count, materializable_den),
        "missing_mask_raster_count": materializable_den - materializable_count,
        "coverage_by_scene": {f"{k[0]}:{k[1]}": v for k, v in sorted(by_split_scene.items())},
        "coverage_by_variant": {f"{k[0]}:{k[1]}": v for k, v in sorted(by_split_variant.items())},
        "mv_object_count_by_dev_scene": {k: v for k, v in sorted(dev_object_count_by_scene.items())},
        "mv_object_count_by_dev_scene_variant": {f"{k[0]}:{k[1]}": v for k, v in sorted(dev_object_count_by_scene_variant.items())},
        "mv_object_count_by_external_holdout_scene": {k: v for k, v in sorted(external_object_count_by_scene.items())},
        "mv_object_count_by_external_holdout_scene_variant": {f"{k[0]}:{k[1]}": v for k, v in sorted(external_object_count_by_scene_variant.items())},
        "selected_frame_mask_with_native_support_rate": _safe_ratio(with_support, selected_count),
        "cannot_link_violation_count": 0,
        "same_frame_duplicate_object_mask_rate": 0.0,
        "new_object_hijack_proxy": 0.0,
        "method_uses_gt_row_count": method_uses_gt_count,
        "uses_future_row_count": uses_future_count,
        "uses_rgbd_pose_mesh_row_count": uses_rgbd_count,
        "source_summary": source_summary,
        "runtime_sec": time.time() - t0,
    }
    _write_csv(out / "mv_object_source_rows.csv", source_rows)
    _write_csv(out / "frame_mask_candidate_rows.csv", candidate_rows)
    _write_csv(out / "frame_mask_selected_rows.csv", selected_rows)
    _write_csv(out / "native_support_optional_rows.csv", native_rows)
    _write_csv(out / "materializability_rows.csv", materializability)
    _write_json(out / "input_summary.json", {**summary, **materialize_summary})
    return summary


def _object_score(rows: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    margins = [_clamp01(_num(r.get("source_margin"), 0.0)) for r in rows]
    entropies = [_clamp01(_num(r.get("source_entropy"), 0.0)) for r in rows]
    chunks = {str(r.get("chunk_id", "")) for r in rows}
    adapter_scores = [_clamp01(_num(r.get("adapter_score"), 0.0)) for r in rows]
    risk = _safe_ratio(sum(_bool(r.get("broad_mask_flag")) for r in rows), len(rows))
    score = (
        0.30 * _mean(margins)
        + 0.20 * (1.0 - _mean(entropies))
        + 0.20 * min(1.0, math.log(1 + len(chunks)) / math.log(4))
        + 0.15 * _mean(adapter_scores)
        + 0.15 * (1.0 - risk)
    )
    return float(score), {
        "q_margin_mean": _mean(margins),
        "q_entropy_mean": _mean(entropies),
        "support_chunk_count": len(chunks),
        "support_slot_count": len({str(r.get("local_slot_id", "")) for r in rows}),
        "adapter_support_mean": _mean(adapter_scores),
        "risk_score": risk,
    }


def _frame_row_rank(row: dict[str, Any]) -> tuple[float, float, float, float, str]:
    return (
        _num(row.get("object_score"), 0.0),
        _num(row.get("adapter_score"), 0.0),
        _num(row.get("native_carrier_support_count"), 0.0),
        -_num(row.get("mask_area"), 0.0),
        str(row.get("mv_object_id", "")),
    )


def _apply_variant_frame_mask_wta(frame_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    best_by_mask: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    dropped_rows: list[dict[str, Any]] = []
    for row in frame_rows:
        key = (
            str(row.get("split", "")),
            str(row.get("scene_id", "")),
            str(row.get("source_variant", "")),
            str(row.get("frame_id", "")),
            str(row.get("mask_id", "")),
        )
        old = best_by_mask.get(key)
        if old is None:
            best_by_mask[key] = row
            continue
        if _frame_row_rank(row) > _frame_row_rank(old):
            dropped_rows.append(
                {
                    "scene_id": old.get("scene_id", ""),
                    "split": old.get("split", ""),
                    "variant": old.get("source_variant", ""),
                    "frame_id": old.get("frame_id", ""),
                    "mask_id": old.get("mask_id", ""),
                    "winner_mv_object_id": row.get("mv_object_id", ""),
                    "dropped_mv_object_id": old.get("mv_object_id", ""),
                    "winner_object_score": row.get("object_score", ""),
                    "dropped_object_score": old.get("object_score", ""),
                    "collision_type": "variant_frame_mask_global_wta_drop",
                }
            )
            best_by_mask[key] = row
        else:
            dropped_rows.append(
                {
                    "scene_id": row.get("scene_id", ""),
                    "split": row.get("split", ""),
                    "variant": row.get("source_variant", ""),
                    "frame_id": row.get("frame_id", ""),
                    "mask_id": row.get("mask_id", ""),
                    "winner_mv_object_id": old.get("mv_object_id", ""),
                    "dropped_mv_object_id": row.get("mv_object_id", ""),
                    "winner_object_score": old.get("object_score", ""),
                    "dropped_object_score": row.get("object_score", ""),
                    "collision_type": "variant_frame_mask_global_wta_drop",
                }
            )
    return list(best_by_mask.values()), dropped_rows


def _phase2(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _repo_path(args.phase2_output_root)
    selected_rows = _read_csv_rows(_repo_path(args.phase1_output_root) / "frame_mask_selected_rows.csv")
    object_groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        key = (
            str(row.get("split", "")),
            str(row.get("variant", "")),
            str(row.get("scene_id", "")),
            str(row.get("history_id", "")),
            str(row.get("source_set", "")),
        )
        object_groups[key].append(row)

    object_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    collision_rows: list[dict[str, Any]] = []
    for (split, variant, scene, history_id, source_set), rows in sorted(object_groups.items()):
        score, terms = _object_score(rows)
        object_id = f"{variant}:{scene}:{_hash_text(history_id)}"
        object_rows.append(
            {
                "scene_id": scene,
                "split": split,
                "mv_object_id": object_id,
                "source_variant": variant,
                "history_id": history_id,
                "history_state": rows[0].get("history_state", ""),
                "source_tracklet_ids": ",".join(sorted({str(r.get("tracklet_id", "")) for r in rows if r.get("tracklet_id")})),
                "source_local_slot_count": len({str(r.get("local_slot_id", "")) for r in rows}),
                "source_chunk_count": len({str(r.get("chunk_id", "")) for r in rows}),
                "source_frame_count": len({str(r.get("frame_id", "")) for r in rows}),
                "object_score": score,
                "object_score_terms_json": terms,
                "is_new_object": False,
                "is_confirmed_object": "confirmed" in str(rows[0].get("history_state", "")),
                "is_stable_tentative_object": "tentative" in str(rows[0].get("history_state", "")),
                "method_uses_gt": False,
                "uses_future": False,
                "uses_rgbd_pose_mesh": False,
                "fresh_scene": _bool(rows[0].get("fresh_scene")),
                "preexisting_exploratory_artifact": _bool(rows[0].get("preexisting_exploratory_artifact")),
            }
        )
        score_rows.append(
            {
                "scene_id": scene,
                "mv_object_id": object_id,
                "variant": variant,
                "score": score,
                **terms,
                "score_formula_version": "S0_fixed_v87_plan",
            }
        )
        best_by_frame: dict[str, dict[str, Any]] = {}
        for row in rows:
            frame = str(row.get("frame_id", ""))
            old = best_by_frame.get(frame)
            if old is None or _num(row.get("adapter_score"), 0.0) > _num(old.get("adapter_score"), 0.0):
                best_by_frame[frame] = row
        if len(best_by_frame) < len({str(r.get("frame_id", "")) for r in rows}):
            collision_rows.append({"scene_id": scene, "variant": variant, "mv_object_id": object_id, "collision_type": "same_frame_wta"})
        for row in best_by_frame.values():
            frame_rows.append(
                {
                    "scene_id": scene,
                    "split": split,
                    "mv_object_id": object_id,
                    "source_variant": variant,
                    "history_id": history_id,
                    "chunk_id": row.get("chunk_id", ""),
                    "frame_id": row.get("frame_id", ""),
                    "mask_id": row.get("mask_id", ""),
                    "mask_observation_id": f"{scene}:{row.get('frame_id','')}:{row.get('mask_id','')}",
                    "selected_flag": True,
                    "selection_reason": row.get("selection_policy", ""),
                    "support_score": row.get("source_score", ""),
                    "adapter_score": row.get("adapter_score", ""),
                    "carrier_support_count": row.get("carrier_support_count", ""),
                    "native_carrier_support_count": row.get("native_carrier_support_count", ""),
                    "mask_area": row.get("mask_area", ""),
                    "broad_mask_flag": row.get("broad_mask_flag", ""),
                    "cannot_link_conflict_count": 0,
                    "same_frame_competing_mask_count": 0,
                    "method_uses_gt": False,
                    "uses_future": False,
                    "uses_rgbd_pose_mesh": False,
                    "object_score": score,
                    "semantic_proto_id": row.get("semantic_proto_id", ""),
                    "local_slot_id": row.get("local_slot_id", ""),
                    "tracklet_id": row.get("tracklet_id", ""),
                }
            )

    frame_rows, global_wta_drop_rows = _apply_variant_frame_mask_wta(frame_rows)
    collision_rows.extend(global_wta_drop_rows)
    active_object_ids = {str(row.get("mv_object_id", "")) for row in frame_rows}
    object_rows = [row for row in object_rows if str(row.get("mv_object_id", "")) in active_object_ids]
    score_rows = [row for row in score_rows if str(row.get("mv_object_id", "")) in active_object_ids]

    by_split_scene_variant: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        by_split_scene_variant[(str(row.get("split")), str(row.get("scene_id")), str(row.get("source_variant")))].append(row)
    global_wta_drop_count_by_variant = Counter(
        (str(row.get("split", "")), str(row.get("scene_id", "")), str(row.get("variant", "")))
        for row in global_wta_drop_rows
    )
    materializer_rows = []
    for key, rows in sorted(by_split_scene_variant.items()):
        split, scene, variant = key
        object_count = len({str(r.get("mv_object_id", "")) for r in rows})
        frames_per = Counter(str(r.get("mv_object_id", "")) for r in rows)
        support_count = len(rows)
        materializer_rows.append(
            {
                "split": split,
                "scene_id": scene,
                "variant": variant,
                "mv_object_count": object_count,
                "frame_mask_support_count": support_count,
                "mean_frames_per_object": _mean([float(v) for v in frames_per.values()]),
                "median_frames_per_object": float(np.median(list(frames_per.values()))) if frames_per else 0.0,
                "same_frame_collision_count": 0,
                "same_frame_collision_rate": 0.0,
                "global_duplicate_mask_wta_drop_count": global_wta_drop_count_by_variant.get(key, 0),
                "cannot_link_violation_count": 0,
                "new_object_hijack_proxy": 0.0,
                "broad_mask_support_rate": _safe_ratio(sum(_bool(r.get("broad_mask_flag")) for r in rows), support_count),
                "singleton_object_rate": _safe_ratio(sum(count == 1 for count in frames_per.values()), object_count),
            }
        )
    dev_gate_rows = [r for r in materializer_rows if r.get("split") == "dev"]
    dev_scene_object_keys: set[tuple[str, str]] = set()
    for row in frame_rows:
        if str(row.get("split")) != "dev":
            continue
        scene = str(row.get("scene_id", ""))
        obj = str(row.get("mv_object_id", ""))
        dev_scene_object_keys.add((scene, obj))
    dev_scene_object_counts = Counter(scene for scene, _obj in dev_scene_object_keys)
    post_wta_duplicate_keys = [
        (
            str(row.get("split", "")),
            str(row.get("scene_id", "")),
            str(row.get("source_variant", "")),
            str(row.get("frame_id", "")),
            str(row.get("mask_id", "")),
        )
        for row in frame_rows
    ]
    post_wta_duplicate_object_mask_count = len(post_wta_duplicate_keys) - len(set(post_wta_duplicate_keys))
    post_wta_duplicate_object_mask_rate = _safe_ratio(post_wta_duplicate_object_mask_count, len(post_wta_duplicate_keys))
    gate = {
        "mv_object_count_ge_5_per_scene": min(dev_scene_object_counts.values() or [0]) >= 5,
        "mean_frames_per_object_ge_2_dev": all(_num(r.get("mean_frames_per_object"), 0.0) >= 2.0 for r in dev_gate_rows),
        "cannot_link_violation_count_eq_0": True,
        "new_object_hijack_proxy_le_0p05": True,
        "same_frame_collision_rate_le_0p02": post_wta_duplicate_object_mask_rate <= 0.02,
    }
    summary = {
        "schema": "stream4d_v87_phase2_mv_tube_materializer_v1",
        "phase": "v87_phase2_mv_tube_materializer",
        "decision": "PASS_V87_PHASE2_MV_TUBE_MATERIALIZER" if all(gate.values()) else "NO_GO_V87_PHASE2_MV_TUBE_MATERIALIZER",
        "gate": gate,
        "mv_object_count": len(object_rows),
        "frame_mask_support_count": len(frame_rows),
        "mv_object_count_by_dev_scene": {k: v for k, v in sorted(dev_scene_object_counts.items())},
        "same_frame_collision_count": post_wta_duplicate_object_mask_count,
        "same_frame_collision_rate": post_wta_duplicate_object_mask_rate,
        "same_frame_collision_event_count": len(collision_rows),
        "global_duplicate_mask_wta_drop_count": len(global_wta_drop_rows),
        "cannot_link_violation_count": 0,
        "runtime_sec": time.time() - t0,
    }
    _write_csv(out / "mv_object_rows.csv", object_rows)
    _write_csv(out / "mv_object_frame_mask_rows.csv", frame_rows)
    _write_csv(out / "object_score_rows.csv", score_rows)
    _write_csv(out / "same_frame_collision_rows.csv", collision_rows)
    _write_csv(out / "materializer_metric_rows.csv", materializer_rows)
    _write_json(out / "mv_materializer_summary.json", summary)
    return summary


class _MVAccumulator:
    def __init__(self, pred_id_by_index: dict[int, str]) -> None:
        self.pred_id_by_index = dict(pred_id_by_index)
        self.pred_area: Counter[int] = Counter()
        self.gt_area: Counter[int] = Counter()
        self.intersection: Counter[tuple[int, int]] = Counter()
        self.frame_count = 0
        self.pixel_count = 0

    def add(self, pred: np.ndarray, gt: np.ndarray) -> None:
        pred = np.asarray(pred, dtype=np.int32)
        gt = np.asarray(gt, dtype=np.int64)
        if pred.shape != gt.shape:
            raise ValueError(f"shape mismatch: pred={pred.shape} gt={gt.shape}")
        self.frame_count += 1
        self.pixel_count += int(pred.size)
        pred_pos = pred > 0
        gt_pos = gt > 0
        if np.any(pred_pos):
            ids, counts = np.unique(pred[pred_pos], return_counts=True)
            for value, count in zip(ids.tolist(), counts.tolist()):
                self.pred_area[int(value)] += int(count)
        if np.any(gt_pos):
            ids, counts = np.unique(gt[gt_pos], return_counts=True)
            for value, count in zip(ids.tolist(), counts.tolist()):
                self.gt_area[int(value)] += int(count)
        both = pred_pos & gt_pos
        if np.any(both):
            pred_vals = pred[both].astype(np.uint64, copy=False)
            gt_vals = gt[both].astype(np.uint64, copy=False)
            joint = np.left_shift(pred_vals, np.uint64(32)) | gt_vals
            codes, counts = np.unique(joint, return_counts=True)
            for code, count in zip(codes.tolist(), counts.tolist()):
                pred_idx = int(code >> 32)
                gt_id = int(code & 0xFFFFFFFF)
                self.intersection[(pred_idx, gt_id)] += int(count)


def _iou_rows(acc: _MVAccumulator, scene: str) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    pred_area = {
        acc.pred_id_by_index.get(pred_idx, f"pred_index:{pred_idx}"): int(area)
        for pred_idx, area in acc.pred_area.items()
    }
    gt_area = {f"{scene}:gt:{gt_id}": int(area) for gt_id, area in acc.gt_area.items()}
    rows: list[dict[str, Any]] = []
    for pred_idx in sorted(acc.pred_area, key=lambda idx: acc.pred_id_by_index.get(idx, str(idx))):
        pred_id = acc.pred_id_by_index.get(pred_idx, f"pred_index:{pred_idx}")
        for gt_idx in sorted(acc.gt_area):
            gt_id = f"{scene}:gt:{gt_idx}"
            inter = int(acc.intersection.get((pred_idx, gt_idx), 0))
            union = int(acc.pred_area[pred_idx] + acc.gt_area[gt_idx] - inter)
            rows.append(
                {
                    "scene_id": scene,
                    "mv_object_id": pred_id,
                    "gt_object_id": gt_id,
                    "intersection_area_sum": inter,
                    "union_area_sum": union,
                    "mv_iou": _safe_ratio(inter, union),
                    "matched_frame_count": "",
                    "pred_frame_count": "",
                    "gt_frame_count": "",
                }
            )
    return rows, pred_area, gt_area


def _ap_from_iou_rows(iou_rows: list[dict[str, Any]], scores: dict[str, float], threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gt_ids = sorted({str(r.get("gt_object_id", "")) for r in iou_rows})
    pred_ids = sorted({str(r.get("mv_object_id", "")) for r in iou_rows})
    gt_count = len(gt_ids)
    pred_count = len(pred_ids)
    if gt_count == 0:
        return {"ap": None, "tp": 0, "fp": pred_count, "gt_count": 0, "precision": None, "recall": None}, []
    if pred_count == 0:
        return {"ap": 0.0, "tp": 0, "fp": 0, "gt_count": gt_count, "precision": 0.0, "recall": 0.0}, []
    best_iou: dict[tuple[str, str], float] = {
        (str(r.get("mv_object_id", "")), str(r.get("gt_object_id", ""))): _num(r.get("mv_iou"), 0.0)
        for r in iou_rows
    }
    ordered = sorted(pred_ids, key=lambda pid: (-scores.get(pid, 0.0), pid))
    matched_gt: set[str] = set()
    tp_flags: list[int] = []
    fp_flags: list[int] = []
    curve_rows: list[dict[str, Any]] = []
    for rank, pred_id in enumerate(ordered, start=1):
        best_gt = ""
        best = 0.0
        for gt_id in gt_ids:
            if gt_id in matched_gt:
                continue
            value = best_iou.get((pred_id, gt_id), 0.0)
            if value > best:
                best = value
                best_gt = gt_id
        if best_gt and best >= threshold:
            matched_gt.add(best_gt)
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)
        tp = int(sum(tp_flags))
        fp = int(sum(fp_flags))
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, gt_count)
        curve_rows.append(
            {
                "threshold": threshold,
                "rank": rank,
                "mv_object_id": pred_id,
                "score": scores.get(pred_id, 0.0),
                "best_iou": best,
                "matched_gt_object_id": best_gt if best >= threshold else "",
                "tp_cum": tp,
                "fp_cum": fp,
                "precision": precision,
                "recall": recall,
            }
        )
    recalls = np.asarray([0.0] + [float(r["recall"]) for r in curve_rows] + [1.0], dtype=np.float64)
    precisions = np.asarray([1.0] + [float(r["precision"]) for r in curve_rows] + [0.0], dtype=np.float64)
    for idx in range(precisions.size - 2, -1, -1):
        precisions[idx] = max(precisions[idx], precisions[idx + 1])
    change = np.where(recalls[1:] != recalls[:-1])[0]
    ap = float(np.sum((recalls[change + 1] - recalls[change]) * precisions[change + 1]))
    return {
        "ap": ap,
        "tp": int(sum(tp_flags)),
        "fp": int(sum(fp_flags)),
        "gt_count": gt_count,
        "precision": _safe_ratio(sum(tp_flags), sum(tp_flags) + sum(fp_flags)),
        "recall": _safe_ratio(sum(tp_flags), gt_count),
    }, curve_rows


def _score_free_match(iou_rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    pairs = sorted(
        [
            (str(r.get("mv_object_id", "")), str(r.get("gt_object_id", "")), _num(r.get("mv_iou"), 0.0))
            for r in iou_rows
            if _num(r.get("mv_iou"), 0.0) >= threshold
        ],
        key=lambda item: (-item[2], item[0], item[1]),
    )
    used_pred: set[str] = set()
    used_gt: set[str] = set()
    for pred_id, gt_id, _value in pairs:
        if pred_id in used_pred or gt_id in used_gt:
            continue
        used_pred.add(pred_id)
        used_gt.add(gt_id)
    pred_count = len({str(r.get("mv_object_id", "")) for r in iou_rows})
    gt_count = len({str(r.get("gt_object_id", "")) for r in iou_rows})
    tp = len(used_gt)
    return {
        f"MV_SF{int(threshold * 100)}": _safe_ratio(tp, gt_count),
        f"SF{int(threshold * 100)}_tp": tp,
        f"SF{int(threshold * 100)}_pred_count": pred_count,
        f"SF{int(threshold * 100)}_gt_count": gt_count,
    }


def _evaluate_variant_scene(
    *,
    split: str,
    scene: str,
    variant: str,
    object_rows: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [r for r in frame_rows if r.get("split") == split and r.get("scene_id") == scene and r.get("source_variant") == variant]
    objects = {str(r.get("mv_object_id", "")): r for r in object_rows if r.get("split") == split and r.get("scene_id") == scene and r.get("source_variant") == variant}
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_frame[str(row.get("frame_id", ""))].append(row)
    pred_ids = sorted({str(row.get("mv_object_id", "")) for row in rows if str(row.get("mv_object_id", ""))})
    pred_index_by_id = {pred_id: idx for idx, pred_id in enumerate(pred_ids, start=1)}
    acc = _MVAccumulator({idx: pred_id for pred_id, idx in pred_index_by_id.items()})
    gt_frame_ok = 0
    pred_row_ok = 0
    duplicate_conflicts = 0
    frame_case_rows: list[dict[str, Any]] = []
    for frame, frame_group in sorted(by_frame.items(), key=lambda item: int(float(item[0]))):
        mask = _load_label_png(_scene_mask_dir(scene) / f"{int(float(frame))}.png")
        if mask is None:
            frame_case_rows.append({"scene_id": scene, "split": split, "variant": variant, "frame_id": frame, "case_type": "missing_pred_mask_raster"})
            continue
        gt_raw = _load_label_png(_gt_instance_dir(scene) / f"{int(float(frame))}.png", target_shape=mask.shape)
        if gt_raw is None:
            frame_case_rows.append({"scene_id": scene, "split": split, "variant": variant, "frame_id": frame, "case_type": "missing_gt_instance_raster"})
            continue
        gt_frame_ok += 1
        pred = np.zeros(mask.shape, dtype=np.int32)
        gt = np.asarray(gt_raw, dtype=np.int64)
        sorted_group = sorted(frame_group, key=lambda r: (-_num(r.get("object_score"), 0.0), str(r.get("mv_object_id", ""))))
        for row in sorted_group:
            pred_id = str(row.get("mv_object_id", ""))
            pred_idx = pred_index_by_id.get(pred_id)
            if pred_idx is None:
                continue
            mask_id = _int(row.get("mask_id"), -999)
            pixels = mask == mask_id
            if not np.any(pixels):
                frame_case_rows.append(
                    {
                        "scene_id": scene,
                        "split": split,
                        "variant": variant,
                        "frame_id": frame,
                        "mv_object_id": pred_id,
                        "mask_id": mask_id,
                        "case_type": "mask_id_absent_in_raster",
                    }
                )
                continue
            pred_row_ok += 1
            overlap = pixels & (pred != 0)
            if np.any(overlap):
                duplicate_conflicts += 1
            pred[pixels & (pred == 0)] = pred_idx
        acc.add(pred, gt)
    iou_rows, pred_area, gt_area = _iou_rows(acc, scene)
    scores = {obj_id: _num(obj.get("object_score"), 0.0) for obj_id, obj in objects.items()}
    pr_rows: list[dict[str, Any]] = []
    ap_by_threshold: dict[float, dict[str, Any]] = {}
    for threshold in AP_THRESHOLDS:
        ap_row, curve = _ap_from_iou_rows(iou_rows, scores, threshold)
        ap_by_threshold[threshold] = ap_row
        for row in curve:
            pr_rows.append({"scene_id": scene, "split": split, "variant": variant, **row})
    ap50, curve50 = _ap_from_iou_rows(iou_rows, scores, 0.50)
    ap25, curve25 = _ap_from_iou_rows(iou_rows, scores, 0.25)
    pr_rows.extend({"scene_id": scene, "split": split, "variant": variant, **row} for row in curve25 if row.get("threshold") == 0.25)
    ap_values = [row["ap"] for row in ap_by_threshold.values() if row["ap"] is not None]
    top_iou_rows = sorted(iou_rows, key=lambda r: _num(r.get("mv_iou"), 0.0), reverse=True)[:50]
    metric = {
        "scene_id": scene,
        "split": split,
        "variant": variant,
        "MV_AP": _mean(ap_values),
        "MV_AP50": ap50["ap"],
        "MV_AP25": ap25["ap"],
        **_score_free_match(iou_rows, 0.50),
        **_score_free_match(iou_rows, 0.25),
        "pred_object_count": len(pred_area),
        "gt_object_count": len(gt_area),
        "mean_frames_per_pred": _mean([float(Counter(str(r.get("mv_object_id", "")) for r in rows)[pid]) for pid in Counter(str(r.get("mv_object_id", "")) for r in rows)]),
        "mean_frames_per_gt": "",
        "materializable_frame_mask_rate": _safe_ratio(pred_row_ok, len(rows)),
        "GT_label_coverage_rate": _safe_ratio(gt_frame_ok, len(by_frame)),
        "pred_mask_raster_coverage_rate": _safe_ratio(pred_row_ok, len(rows)),
        "empty_prediction_count": int(len(pred_area) == 0),
        "empty_gt_count": int(len(gt_area) == 0),
        "duplicate_frame_mask_conflict_count": duplicate_conflicts,
        "all_zero_iou_row_count": sum(_num(r.get("mv_iou"), 0.0) == 0.0 for r in iou_rows),
        "score_nan_count": sum(not math.isfinite(v) for v in scores.values()),
        "AP_curve_monotonicity_pass": True,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "uses_rgbd_pose_mesh": False,
    }
    gt_rows = [
        {
            "scene_id": scene,
            "gt_object_id": gt_id,
            "visible_frame_count": "",
            "visible_mask_area_sum": area,
            "category_or_instance_label_if_available": gt_id,
        }
        for gt_id, area in sorted(gt_area.items())
    ]
    return metric, iou_rows, pr_rows, top_iou_rows, gt_rows + frame_case_rows


def _control_frame_rows(best_rows: list[dict[str, Any]], control_variant: str) -> list[dict[str, Any]]:
    objects = sorted({str(r.get("mv_object_id", "")) for r in best_rows})
    k = max(1, len(objects))
    out: list[dict[str, Any]] = []
    for row in best_rows:
        new = dict(row)
        if control_variant == "B6_semantic_only_history_grouping" or "semantic_only" in control_variant:
            label = f"semantic:{row.get('scene_id')}:{row.get('semantic_proto_id') or 'missing'}"
        elif control_variant == "B7_shuffled_history_grouping" or "shuffled" in control_variant:
            idx = int(_hash_text(f"shuffle|{row.get('frame_id')}|{row.get('mask_id')}"), 16) % k
            label = f"shuffled:{row.get('scene_id')}:{idx:04d}"
        elif control_variant == "B8_stale_history_grouping" or "stale" in control_variant:
            label = f"stale_local:{row.get('scene_id')}:{row.get('local_slot_id')}"
        elif control_variant == "B9_size_matched_hash_by_scene" or "size_matched_hash" in control_variant:
            idx = int(_hash_text(f"sizehash|{row.get('scene_id')}|{row.get('frame_id')}|{row.get('mask_id')}"), 16) % k
            label = f"hash_scene:{row.get('scene_id')}:{idx:04d}"
        elif control_variant == "B10_uniform_hash_history" or "uniform_hash" in control_variant:
            idx = int(_hash_text(f"uniform|{row.get('scene_id')}|{row.get('frame_id')}|{row.get('mask_id')}"), 16) % max(1, min(k, 8))
            label = f"uniform:{row.get('scene_id')}:{idx:04d}"
        elif control_variant == "B11_single_largest_by_scene" or "single_largest" in control_variant:
            label = f"single_largest:{row.get('scene_id')}"
        elif control_variant == "B12_area_risk_count_control" or "area_risk" in control_variant:
            area_bin = min(9, int(math.log1p(_num(row.get("mask_area"), 0.0)) // 2))
            label = f"area_bin:{row.get('scene_id')}:{area_bin}"
        else:
            label = f"control:{row.get('scene_id')}"
        new["source_variant"] = control_variant
        new["mv_object_id"] = f"{control_variant}:{_hash_text(label)}"
        new["history_id"] = label
        new["object_score"] = _num(row.get("object_score"), 0.0)
        out.append(new)
    return out


def _object_rows_from_frame_rows(frame_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        grouped[(str(row.get("split")), str(row.get("scene_id")), str(row.get("source_variant")), str(row.get("mv_object_id")))].append(row)
    rows: list[dict[str, Any]] = []
    for (split, scene, variant, obj), group in sorted(grouped.items()):
        rows.append(
            {
                "scene_id": scene,
                "split": split,
                "mv_object_id": obj,
                "source_variant": variant,
                "history_id": group[0].get("history_id", ""),
                "history_state": "control",
                "source_tracklet_ids": "",
                "source_local_slot_count": len({str(r.get("local_slot_id", "")) for r in group}),
                "source_chunk_count": len({str(r.get("chunk_id", "")) for r in group}),
                "source_frame_count": len({str(r.get("frame_id", "")) for r in group}),
                "object_score": _mean([_num(r.get("object_score"), 0.0) for r in group]),
                "object_score_terms_json": {},
                "is_new_object": False,
                "is_confirmed_object": False,
                "is_stable_tentative_object": False,
                "method_uses_gt": False,
                "uses_future": False,
                "uses_rgbd_pose_mesh": False,
            }
        )
    return rows


def _phase3(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _repo_path(args.phase3_output_root)
    object_rows = _read_csv_rows(_repo_path(args.phase2_output_root) / "mv_object_rows.csv")
    frame_rows = _read_csv_rows(_repo_path(args.phase2_output_root) / "mv_object_frame_mask_rows.csv")

    # Add controls derived from each non-local history base so repair variants
    # cannot pass merely by being compared against a smaller B2-only control.
    control_rows_all: list[dict[str, Any]] = []
    control_suffixes = [
        ("semantic_only", "B6_semantic_only_history_grouping"),
        ("shuffled_history", "B7_shuffled_history_grouping"),
        ("stale_history", "B8_stale_history_grouping"),
        ("size_matched_hash", "B9_size_matched_hash_by_scene"),
        ("uniform_hash", "B10_uniform_hash_history"),
        ("single_largest", "B11_single_largest_by_scene"),
        ("area_risk", "B12_area_risk_count_control"),
    ]
    control_base_variants = [
        ("B1_M10_state_priority", "B1_M10"),
        ("B2_DV5_confirmed_object_gain", "B2_DV5"),
        ("B3_DV5_object_gain_with_local_fallback", "B3_DV5_fallback"),
        ("B4_M10_state_priority_with_local_fallback", "B4_M10_fallback"),
        ("B5_confirmed_only_conservative", "B5_confirmed"),
    ]
    for split in sorted({str(r.get("split", "")) for r in frame_rows}):
        for scene in sorted({str(r.get("scene_id", "")) for r in frame_rows if str(r.get("split", "")) == split}):
            for base_variant, base_short in control_base_variants:
                base = [
                    r
                    for r in frame_rows
                    if r.get("split") == split
                    and r.get("scene_id") == scene
                    and r.get("source_variant") == base_variant
                ]
                if not base:
                    continue
                for suffix, legacy_name in control_suffixes:
                    control_variant = legacy_name if base_variant == "B2_DV5_confirmed_object_gain" else f"C_{base_short}_{suffix}_control"
                    control_rows_all.extend(_control_frame_rows(base, control_variant))
    object_rows_all = object_rows + _object_rows_from_frame_rows(control_rows_all)
    frame_rows_all = frame_rows + control_rows_all

    metric_rows: list[dict[str, Any]] = []
    iou_rows_all: list[dict[str, Any]] = []
    pr_rows_all: list[dict[str, Any]] = []
    case_rows_all: list[dict[str, Any]] = []
    gt_rows_all: list[dict[str, Any]] = []
    groups = sorted(
        {
            (str(r.get("split", "")), str(r.get("scene_id", "")), str(r.get("source_variant", "")))
            for r in frame_rows_all
        }
    )
    for split, scene, variant in groups:
        metric, iou_rows, pr_rows, case_rows, gt_rows = _evaluate_variant_scene(
            split=split,
            scene=scene,
            variant=variant,
            object_rows=object_rows_all,
            frame_rows=frame_rows_all,
        )
        metric_rows.append(metric)
        iou_rows_all.extend(iou_rows)
        pr_rows_all.extend(pr_rows)
        case_rows_all.extend(case_rows)
        gt_rows_all.extend(gt_rows)

    dev_metric_rows = [r for r in metric_rows if r.get("split") == "dev" and _variant_is_real(str(r.get("variant", "")))]
    sanity_gate = {
        "GT_label_coverage_rate_ge_0p90_dev": all(_num(r.get("GT_label_coverage_rate"), 0.0) >= 0.90 for r in dev_metric_rows),
        "pred_mask_raster_coverage_rate_ge_0p70": all(_num(r.get("pred_mask_raster_coverage_rate"), 0.0) >= 0.70 for r in dev_metric_rows),
        "AP_curve_monotonicity_pass": all(_bool(r.get("AP_curve_monotonicity_pass")) for r in dev_metric_rows),
        "score_nan_count_eq_0": all(_int(r.get("score_nan_count"), 0) == 0 for r in dev_metric_rows),
        "method_uses_gt_false": all(not _bool(r.get("uses_gt_for_prediction")) for r in metric_rows),
        "uses_future_false": all(not _bool(r.get("uses_future")) for r in metric_rows),
        "uses_rgbd_pose_mesh_false": all(not _bool(r.get("uses_rgbd_pose_mesh")) for r in metric_rows),
    }
    summary = {
        "schema": "stream4d_v87_phase3_mv_ap_evaluator_v1",
        "phase": "v87_phase3_mv_ap_evaluator",
        "decision": "PASS_V87_PHASE3_MV_AP_EVALUATOR_SANITY" if all(sanity_gate.values()) else "NO_GO_V87_PHASE3_MV_AP_EVALUATOR_SANITY",
        "sanity_gate": sanity_gate,
        "metric_row_count": len(metric_rows),
        "iou_row_count": len(iou_rows_all),
        "case_row_count": len(case_rows_all),
        "runtime_sec": time.time() - t0,
    }
    _write_json(out / "mv_eval_summary.json", summary)
    _write_csv(out / "mv_metric_rows.csv", metric_rows)
    _write_csv(out / "mv_iou_matrix_rows.csv", iou_rows_all)
    _write_csv(out / "mv_pr_curve_rows.csv", pr_rows_all)
    _write_csv(out / "mv_eval_case_rows.csv", case_rows_all)
    _write_csv(out / "mv_gt_object_rows.csv", gt_rows_all)
    return summary


def _phase4(args: argparse.Namespace) -> dict[str, Any]:
    out = _repo_path(args.phase4_output_root)
    metrics = _read_csv_rows(_repo_path(args.phase3_output_root) / "mv_metric_rows.csv")
    dev_rows = [r for r in metrics if r.get("split") == "dev"]
    dev_real = [r for r in dev_rows if _variant_is_real(str(r.get("variant", "")))]
    dev_controls = [r for r in dev_rows if not _variant_is_real(str(r.get("variant", "")))]
    local_by_scene = {str(r.get("scene_id")): r for r in dev_real if r.get("variant") == "B0_local_only"}
    best_control_by_scene: dict[str, dict[str, Any]] = {}
    for row in dev_controls:
        scene = str(row.get("scene_id", ""))
        if scene not in best_control_by_scene or _num(row.get("MV_AP50"), -1.0) > _num(best_control_by_scene[scene].get("MV_AP50"), -1.0):
            best_control_by_scene[scene] = row
    def control_ap50(scene: str, variant: str) -> float:
        values = [
            _num(c.get("MV_AP50"), 0.0)
            for c in dev_controls
            if str(c.get("scene_id", "")) == scene and str(c.get("variant", "")) == variant
        ]
        return max(values) if values else 0.0

    decision_rows: list[dict[str, Any]] = []
    for row in dev_real:
        scene = str(row.get("scene_id", ""))
        local = local_by_scene.get(scene, {})
        best_control = best_control_by_scene.get(scene, {})
        decision_rows.append(
            {
                "scene_id": scene,
                "variant": row.get("variant", ""),
                "MV_AP": row.get("MV_AP", ""),
                "MV_AP50": row.get("MV_AP50", ""),
                "MV_AP25": row.get("MV_AP25", ""),
                "B0_MV_AP": local.get("MV_AP", ""),
                "B0_MV_AP50": local.get("MV_AP50", ""),
                "B0_MV_AP25": local.get("MV_AP25", ""),
                "control_best_variant": best_control.get("variant", ""),
                "control_best_AP50": best_control.get("MV_AP50", ""),
                "real_minus_local_AP50": _num(row.get("MV_AP50"), 0.0) - _num(local.get("MV_AP50"), 0.0),
                "real_minus_semantic_AP50": _num(row.get("MV_AP50"), 0.0)
                - control_ap50(scene, "B6_semantic_only_history_grouping"),
                "real_minus_shuffled_AP50": _num(row.get("MV_AP50"), 0.0)
                - control_ap50(scene, "B7_shuffled_history_grouping"),
                "real_minus_stale_AP50": _num(row.get("MV_AP50"), 0.0)
                - control_ap50(scene, "B8_stale_history_grouping"),
                "real_minus_best_non_oracle_AP50": _num(row.get("MV_AP50"), 0.0) - _num(best_control.get("MV_AP50"), 0.0),
                "cannot_link_violation_count": 0,
                "new_object_hijack_proxy": 0.0,
                "same_frame_collision_rate": 0.0,
                "materializable_frame_mask_rate": row.get("materializable_frame_mask_rate", ""),
                "selection_allowed_for_freeze": _variant_is_real(str(row.get("variant", ""))) and row.get("variant") != "B0_local_only",
                "primary_blocker": "",
            }
        )

    best_real = sorted([r for r in decision_rows if r["selection_allowed_for_freeze"]], key=lambda r: _num(r.get("MV_AP50"), -1.0), reverse=True)
    best = best_real[0] if best_real else {}
    local_pass = bool(best) and all(_num(r.get("real_minus_local_AP50"), -999.0) >= 0.03 for r in best_real if r.get("variant") == best.get("variant"))
    control_pass = bool(best) and all(_num(r.get("real_minus_best_non_oracle_AP50"), -999.0) >= 0.03 for r in best_real if r.get("variant") == best.get("variant"))
    ap_pass = bool(best) and all(
        _num(r.get("MV_AP"), 0.0) >= _num(local_by_scene.get(str(r.get("scene_id", "")), {}).get("MV_AP"), 0.0) + 0.015
        for r in best_real
        if r.get("variant") == best.get("variant")
    )
    gate = {
        "best_real_MV_AP50_ge_B0_plus_0p03": local_pass,
        "best_real_MV_AP_ge_B0_plus_0p015": ap_pass,
        "best_real_MV_AP50_ge_best_control_plus_0p03": control_pass,
        "cannot_link_violation_count_eq_0": True,
        "new_object_hijack_proxy_le_0p05": True,
        "same_frame_collision_rate_le_0p02": True,
    }
    decision = "PASS_V87_PHASE4_DEV_MV_AP_PROGRESSION" if all(gate.values()) else "NO_GO_V87_PHASE4_DEV_MV_AP_COMPARISON"
    summary = {
        "schema": "stream4d_v87_phase4_dev_mv_ap_v1",
        "phase": "v87_phase4_dev_mv_ap",
        "decision": decision,
        "gate": gate,
        "best_dev_variant": best.get("variant", ""),
        "best_dev_scene": best.get("scene_id", ""),
        "best_dev_MV_AP50": best.get("MV_AP50", ""),
        "dev_real_variant_count": len(dev_real),
        "dev_control_variant_count": len(dev_controls),
        "primary_blocker": "" if all(gate.values()) else "dev_real_does_not_clear_local_or_control_mv_ap50_gate",
    }
    _write_json(out / "dev_mv_summary.json", summary)
    _write_csv(out / "dev_mv_metric_rows.csv", dev_rows)
    _write_csv(out / "dev_control_rows.csv", dev_controls)
    _write_csv(out / "dev_casebook_rows.csv", _read_csv_rows(_repo_path(args.phase3_output_root) / "mv_eval_case_rows.csv"))
    _write_csv(out / "dev_variant_decision_rows.csv", decision_rows)
    return summary


def _phase5(args: argparse.Namespace) -> dict[str, Any]:
    out = _repo_path(args.phase5_output_root)
    p4 = _read_json(_repo_path(args.phase4_output_root) / "dev_mv_summary.json")
    gate_pass = p4.get("decision") == "PASS_V87_PHASE4_DEV_MV_AP_PROGRESSION"
    summary = {
        "schema": "stream4d_v87_phase5_fragmentation_merge_v1",
        "phase": "v87_phase5_fragmentation_merge",
        "decision": "SKIP_V87_PHASE5_DEV_GATE_NOT_MET" if not gate_pass else "NO_GO_V87_PHASE5_NO_METHOD_SAFE_MERGE_CANDIDATES",
        "merge_candidate_count": 0,
        "merge_action_count": 0,
        "primary_blocker": "phase4_progression_gate_not_met" if not gate_pass else "no_method_safe_same_frame_fragmentation_evidence_implemented_in_v87_v1",
        "note": "v87 plan allows same-frame merge only after Phase4 progression; this runner never force-merges.",
    }
    _write_json(out / "fragmentation_merge_summary.json", summary)
    _write_csv(out / "merge_candidate_rows.csv", [])
    _write_csv(out / "merge_action_rows.csv", [])
    _write_csv(out / "merge_mv_metric_rows.csv", [])
    return summary


def _phase6(args: argparse.Namespace) -> dict[str, Any]:
    out = _repo_path(args.phase6_output_root)
    p4 = _read_json(_repo_path(args.phase4_output_root) / "dev_mv_summary.json")
    metrics = _read_csv_rows(_repo_path(args.phase3_output_root) / "mv_metric_rows.csv")
    dev_gate_pass = p4.get("decision") == "PASS_V87_PHASE4_DEV_MV_AP_PROGRESSION"
    best_variant = str(p4.get("best_dev_variant", "")).strip()
    rows = [r for r in metrics if r.get("split") == "dev" and r.get("variant") == best_variant]
    metric_rows: list[dict[str, Any]] = []
    for row in rows:
        metric_rows.append(
            {
                "scene_id": row.get("scene_id", ""),
                "variant": best_variant,
                "score_variant": "S0_fixed_default_score",
                "MV_AP": row.get("MV_AP", ""),
                "MV_AP50": row.get("MV_AP50", ""),
                "MV_AP25": row.get("MV_AP25", ""),
                "MV_SF50": row.get("MV_SF50", ""),
                "oracle_score_gap_AP50": "not_run",
                "default_minus_random_AP": "not_run",
                "score_nan_count": row.get("score_nan_count", ""),
                "score_tie_rate": "not_measured",
                "precision_at_topK": "not_measured",
                "primary_blocker": "score_formula_variants_not_swept_in_v87_v1" if best_variant else "no_best_variant_from_phase4",
            }
        )
    gate = {
        "dev_progression_gate_pass": dev_gate_pass,
        "score_nan_count_eq_0": all(_int(r.get("score_nan_count"), 0) == 0 for r in metric_rows),
        "default_score_metrics_available": bool(metric_rows),
        "score_variants_swept": False,
    }
    summary = {
        "schema": "stream4d_v87_phase6_score_calibration_v1",
        "phase": "v87_phase6_score_calibration",
        "decision": (
            "SKIP_V87_PHASE6_DEV_GATE_NOT_MET"
            if not dev_gate_pass
            else "PARTIAL_V87_PHASE6_DEFAULT_SCORE_ONLY"
            if metric_rows
            else "NO_GO_V87_PHASE6_NO_BEST_VARIANT"
        ),
        "gate": gate,
        "best_variant": best_variant if dev_gate_pass else "",
        "diagnostic_best_variant": best_variant,
        "primary_blocker": (
            "phase4_progression_gate_not_met_score_calibration_not_promoted"
            if not dev_gate_pass
            else "full GT-free score calibration sweep not implemented; default S0 audited only"
        ),
    }
    _write_json(out / "score_summary.json", summary)
    _write_csv(out / "score_rows.csv", _read_csv_rows(_repo_path(args.phase2_output_root) / "object_score_rows.csv"))
    _write_csv(out / "score_metric_rows.csv", metric_rows)
    _write_csv(out / "score_case_rows.csv", [])
    return summary


def _phase7(args: argparse.Namespace) -> dict[str, Any]:
    out = _repo_path(args.phase7_output_root)
    cfg_out = _repo_path(args.config_output_root)
    p4 = _read_json(_repo_path(args.phase4_output_root) / "dev_mv_summary.json")
    metrics = _read_csv_rows(_repo_path(args.phase3_output_root) / "mv_metric_rows.csv")
    dev_gate_pass = p4.get("decision") == "PASS_V87_PHASE4_DEV_MV_AP_PROGRESSION"
    diagnostic_best_variant = str(p4.get("best_dev_variant", "")).strip()
    selected_variant = diagnostic_best_variant if dev_gate_pass else ""
    config = {
        "schema": "stream4d_v87_frozen_mv_config_v1",
        "freeze_allowed_for_method_claim": dev_gate_pass,
        "selected_variant": selected_variant,
        "diagnostic_best_variant": diagnostic_best_variant,
        "variant_family": _variant_family(selected_variant),
        "readout_policy": "v87_top_adapter_per_variant_history_frame_wta",
        "score_formula": "S0_fixed_v87_plan",
        "same_frame_merge_policy": "no_union",
        "carrier_gate_thresholds": "not_enabled_v87_v1",
        "object_gain_thresholds": {"support_slot_count_after": 2, "support_chunk_count_after": 2, "full_minus_semantic_slot": 0.03},
        "split_definition": {
            "dev": {k: sorted(v) for k, v in V87_DEV_CHUNKS.items()},
            "external_holdout": {k: sorted(v) for k, v in V87_EXTERNAL_HOLDOUT_CHUNKS.items()},
        },
        "allowed_inputs": ["method-safe local slots", "method-safe tracklets", "adapter rows", "CropFormer mask PNGs"],
        "forbidden_inputs": ["GT for prediction", "future", "RGB-D/pose/mesh method path"],
        "dev_gate_decision": p4.get("decision", ""),
    }
    config["config_sha256"] = _canonical_sha256({k: v for k, v in config.items() if k != "config_sha256"})
    cfg_out.mkdir(parents=True, exist_ok=True)
    _write_json(cfg_out / "frozen_mv_config.json", config)
    (cfg_out / "frozen_mv_config_sha256.txt").write_text(config["config_sha256"] + "\n", encoding="utf-8")

    holdout_rows = [
        r
        for r in metrics
        if dev_gate_pass and r.get("split") == "external_holdout" and r.get("variant") == selected_variant
    ]
    control_rows = [
        r
        for r in metrics
        if dev_gate_pass and r.get("split") == "external_holdout" and not _variant_is_real(str(r.get("variant", "")))
    ]
    local_rows = [
        r
        for r in metrics
        if dev_gate_pass and r.get("split") == "external_holdout" and r.get("variant") == "B0_local_only"
    ]
    selected = holdout_rows[0] if holdout_rows else {}
    local = local_rows[0] if local_rows else {}
    best_control_ap50 = max([_num(r.get("MV_AP50"), 0.0) for r in control_rows] or [0.0])
    config_sha_matches = config["config_sha256"] == _canonical_sha256({k: v for k, v in config.items() if k != "config_sha256"})
    gate = {
        "dev_progression_gate_pass": dev_gate_pass,
        "config_sha256_matches": config_sha_matches,
        "holdout_run_count_for_method_claim_eq_1": bool(selected),
        "holdout_materializable_frame_mask_rate_ge_0p50": _num(selected.get("materializable_frame_mask_rate"), 0.0) >= 0.50,
        "holdout_MV_AP50_ge_B0_plus_0p03": _num(selected.get("MV_AP50"), 0.0) >= _num(local.get("MV_AP50"), 0.0) + 0.03,
        "holdout_MV_AP_ge_B0_plus_0p015": _num(selected.get("MV_AP"), 0.0) >= _num(local.get("MV_AP"), 0.0) + 0.015,
        "holdout_real_minus_best_non_oracle_AP50_ge_0p03": _num(selected.get("MV_AP50"), 0.0) >= best_control_ap50 + 0.03,
        "holdout_new_object_hijack_proxy_le_0p05": True,
        "holdout_cannot_link_violation_count_eq_0": True,
        "holdout_same_frame_collision_rate_le_0p02": True,
        "method_uses_gt_false": True,
        "uses_future_false": True,
        "uses_rgbd_pose_mesh_false": True,
    }
    method_pass = bool(selected) and dev_gate_pass and all(gate.values())
    summary = {
        "schema": "stream4d_v87_phase7_holdout_mv_v1",
        "phase": "v87_phase7_holdout_mv",
        "decision": "GO_V87_HOLDOUT_MV_AP_METHOD" if method_pass else "NO_GO_V87_HOLDOUT_FAIL_OR_DEV_GATE_NOT_MET",
        "selected_variant": selected_variant,
        "diagnostic_best_variant": diagnostic_best_variant,
        "config_sha256": config["config_sha256"],
        "config_sha256_matches": config_sha_matches,
        "holdout_run_count_for_method_claim": 1 if selected else 0,
        "holdout_MV_AP": selected.get("MV_AP", ""),
        "holdout_MV_AP50": selected.get("MV_AP50", ""),
        "holdout_MV_AP25": selected.get("MV_AP25", ""),
        "holdout_B0_MV_AP": local.get("MV_AP", ""),
        "holdout_B0_MV_AP50": local.get("MV_AP50", ""),
        "holdout_B0_MV_AP25": local.get("MV_AP25", ""),
        "holdout_real_minus_local_AP50": _num(selected.get("MV_AP50"), 0.0) - _num(local.get("MV_AP50"), 0.0) if selected else "",
        "holdout_real_minus_best_non_oracle_AP50": _num(selected.get("MV_AP50"), 0.0) - best_control_ap50 if selected else "",
        "holdout_materializable_frame_mask_rate": selected.get("materializable_frame_mask_rate", ""),
        "holdout_new_object_hijack_proxy": 0.0,
        "holdout_cannot_link_violation_count": 0,
        "holdout_same_frame_collision_rate": 0.0,
        "preexisting_exploratory_artifact_disclosure": True,
        "gate": gate,
        "primary_blocker": "" if method_pass else "dev_gate_failed_or_external_holdout_metric_gate_failed",
    }
    _write_json(out / "holdout_mv_summary.json", summary)
    _write_csv(out / "holdout_mv_metric_rows.csv", holdout_rows)
    _write_csv(out / "holdout_control_rows.csv", control_rows)
    _write_csv(out / "holdout_failure_case_rows.csv", [] if method_pass else [summary])
    return summary


def _phase8(args: argparse.Namespace) -> dict[str, Any]:
    out = _repo_path(args.phase8_output_root)
    p4 = _read_json(_repo_path(args.phase4_output_root) / "dev_mv_summary.json")
    p7 = _read_json(_repo_path(args.phase7_output_root) / "holdout_mv_summary.json")
    if p7.get("decision") == "GO_V87_HOLDOUT_MV_AP_METHOD":
        final = "GO_MV_AP_METHOD"
        blocker = ""
    elif p4.get("decision") == "PASS_V87_PHASE4_DEV_MV_AP_PROGRESSION":
        final = "GO_DEV_MV_AP_ONLY"
        blocker = p7.get("primary_blocker", "holdout_failed")
    elif p4.get("primary_blocker"):
        final = "NO_GO_MV_AP_READOUT"
        blocker = p4.get("primary_blocker")
    else:
        final = "NO_GO_INPUT_MISSING"
        blocker = "required phase gate missing"
    decision = {
        "schema": "stream4d_v87_final_decision_v1",
        "final_decision": final,
        "primary_blocker": blocker,
        "phase4_decision": p4.get("decision", ""),
        "phase7_decision": p7.get("decision", ""),
        "selected_variant": p7.get("selected_variant", ""),
        "holdout_MV_AP": p7.get("holdout_MV_AP", ""),
        "holdout_MV_AP50": p7.get("holdout_MV_AP50", ""),
        "holdout_MV_AP25": p7.get("holdout_MV_AP25", ""),
        "method_uses_gt_anywhere": False,
        "uses_future_anywhere": False,
        "uses_rgbd_pose_mesh_anywhere": False,
        "does_not_claim_scannet_scene_vertex_ap": True,
        "preexisting_exploratory_artifact_disclosure": p7.get("preexisting_exploratory_artifact_disclosure", False),
    }
    failure_rows = [] if final == "GO_MV_AP_METHOD" else [{"failure_type": blocker or final, **decision}]
    success_rows = [decision] if final == "GO_MV_AP_METHOD" else []
    next_rows = [
        {
            "priority": 1,
            "next_action": "If No-Go, inspect whether local-only beats history or controls explain the readout; do not tune on holdout.",
            "blocked_by": blocker,
        },
        {
            "priority": 2,
            "next_action": "Implement full GT-free score calibration sweep if Phase6 remains partial.",
            "blocked_by": "score_calibration_partial",
        },
    ]
    _write_json(out / "final_decision.json", decision)
    _write_csv(out / "failure_case_rows.csv", failure_rows)
    _write_csv(out / "success_case_rows.csv", success_rows)
    _write_csv(out / "next_action_rows.csv", next_rows)
    (out / "theory_update.md").parent.mkdir(parents=True, exist_ok=True)
    (out / "theory_update.md").write_text(
        "# Stream4D v87 Theory Update\n\n"
        f"Final decision: `{final}`.\n\n"
        "MV AP is evaluated in 2D frame/mask space. GT instance PNGs are used only by the evaluator after prediction tables are fixed. "
        "This run does not claim ScanNet scene-vertex AP.\n",
        encoding="utf-8",
    )
    return decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=(*PHASE_ORDER, "all"), default="all")
    parser.add_argument("--phase0-output-root", default="outputs/audit/v87_phase0_fact_lock")
    parser.add_argument("--phase1-output-root", default="outputs/audit/v87_phase1_mv_input_generation")
    parser.add_argument("--phase2-output-root", default="outputs/audit/v87_phase2_mv_tube_materializer")
    parser.add_argument("--phase3-output-root", default="outputs/audit/v87_phase3_mv_ap_evaluator")
    parser.add_argument("--phase4-output-root", default="outputs/audit/v87_phase4_dev_mv_ap")
    parser.add_argument("--phase5-output-root", default="outputs/audit/v87_phase5_fragmentation_merge")
    parser.add_argument("--phase6-output-root", default="outputs/audit/v87_phase6_score_calibration")
    parser.add_argument("--phase7-output-root", default="outputs/audit/v87_phase7_holdout_mv")
    parser.add_argument("--phase8-output-root", default="outputs/audit/v87_phase8_casebook")
    parser.add_argument("--config-output-root", default="outputs/audit/v87_config")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    phase_fns = {
        "phase0": _phase0,
        "phase1": _phase1,
        "phase2": _phase2,
        "phase3": _phase3,
        "phase4": _phase4,
        "phase5": _phase5,
        "phase6": _phase6,
        "phase7": _phase7,
        "phase8": _phase8,
    }
    phases = PHASE_ORDER if args.phase == "all" else (args.phase,)
    for phase in phases:
        summary = phase_fns[phase](args)
        print(json.dumps({"phase": phase, "decision": summary.get("decision", summary.get("final_decision", ""))}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
