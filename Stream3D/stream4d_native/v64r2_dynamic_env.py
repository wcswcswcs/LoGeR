from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .v47_common import ROOT, utc_now, write_csv, write_json


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def _load_annotations(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    if not path.exists():
        return [], "annotation_file_missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, list):
        return [], f"annotation_root_is_{type(payload).__name__}"
    return [item for item in payload if isinstance(item, dict)], None


def _nested_path(item: dict[str, Any], key: str) -> str:
    value = item.get(key)
    if isinstance(value, dict):
        raw = value.get("path")
        return str(raw or "")
    return ""


def _annotation_rel_path(item: dict[str, Any], key: str) -> str:
    if key == "instance_id_map_path":
        return str(item.get("instance_id_map_path") or "")
    return _nested_path(item, key)


def _exists(split_dir: Path, rel: str) -> bool:
    return bool(rel and (split_dir / rel).exists())


def _camera_ok(item: dict[str, Any]) -> bool:
    viewpoint = item.get("viewpoint")
    return isinstance(viewpoint, dict) and all(key in viewpoint for key in ["R", "T", "focal_length", "principal_point"])


def _split_level(flags: dict[str, Any]) -> tuple[int, str, list[str]]:
    blockers: list[str] = []
    if not flags["rgb_frames_exist"]:
        return 0, "DYN_LEVEL_0", ["no actual RGB frames found"]
    if flags["depth_frames_exist"] and flags["camera_metadata_exist"] and (
        flags["instance_masks_exist"] or flags["object_id_maps_exist"] or flags["object_ids_declared"]
    ) and flags["trajectories_exist"]:
        return 4, "DYN_LEVEL_4", []
    if (flags["instance_masks_exist"] or flags["object_id_maps_exist"] or flags["object_ids_declared"]) and flags[
        "rgb_frames_exist"
    ]:
        if flags["instance_masks_exist"] or flags["object_id_maps_exist"]:
            return 3, "DYN_LEVEL_3", []
        blockers.append("object ids are declared in annotations but actual instance/object-id mask files are missing")
    if flags["rgb_frames_exist"] and flags["depth_frames_exist"] and flags["camera_metadata_exist"]:
        return 2, "DYN_LEVEL_2", blockers
    if flags["rgb_frames_exist"]:
        blockers.extend(
            [
                "actual depth files missing",
                "actual instance masks/object-id maps missing",
                "official IDF1/IDSW/4D metrics not allowed",
            ]
        )
        return 1, "DYN_LEVEL_1", blockers
    return 0, "DYN_LEVEL_0", blockers


def build_v64r2_dynamic_env(
    *,
    data_root: str | Path = "data/dynamic-replica/v2",
    split: str = "valid",
    max_annotation_rows: int = 0,
) -> dict[str, Any]:
    root = _project(data_root)
    split_dir = root / split
    annotation_path = split_dir / f"frame_annotations_{split}.json"
    annotations, annotation_error = _load_annotations(annotation_path)
    scan_annotations = annotations[: int(max_annotation_rows)] if max_annotation_rows and max_annotation_rows > 0 else annotations
    scene_acc: dict[str, Counter[str]] = defaultdict(Counter)
    file_rows: list[dict[str, Any]] = []
    global_counts = Counter()
    for item in scan_annotations:
        scene = str(item.get("sequence_name") or "")
        camera = str(item.get("camera_name") or "")
        frame = item.get("frame_number")
        scene_key = f"{scene}_source_{camera}" if camera and not scene.endswith(f"_source_{camera}") else scene
        rels = {
            "image": _annotation_rel_path(item, "image"),
            "depth": _annotation_rel_path(item, "depth"),
            "mask": _annotation_rel_path(item, "mask"),
            "flow_forward": _annotation_rel_path(item, "flow_forward"),
            "flow_forward_mask": _annotation_rel_path(item, "flow_forward_mask"),
            "flow_backward": _annotation_rel_path(item, "flow_backward"),
            "flow_backward_mask": _annotation_rel_path(item, "flow_backward_mask"),
            "trajectories": _annotation_rel_path(item, "trajectories"),
            "instance_id_map": _annotation_rel_path(item, "instance_id_map_path"),
        }
        exists = {key: _exists(split_dir, rel) for key, rel in rels.items()}
        camera_ok = _camera_ok(item)
        object_ids_declared = bool(item.get("instance_ids"))
        for key, flag in exists.items():
            if flag:
                global_counts[f"{key}_exists"] += 1
                scene_acc[scene_key][f"{key}_exists"] += 1
        if camera_ok:
            global_counts["camera_metadata_ok"] += 1
            scene_acc[scene_key]["camera_metadata_ok"] += 1
        if object_ids_declared:
            global_counts["object_ids_declared"] += 1
            scene_acc[scene_key]["object_ids_declared"] += 1
        scene_acc[scene_key]["annotation_rows"] += 1
        if len(file_rows) < 200:
            file_rows.append(
                {
                    "split": split,
                    "scene_id": scene_key,
                    "frame_number": frame,
                    "image_path": rels["image"],
                    "image_exists": exists["image"],
                    "depth_path": rels["depth"],
                    "depth_exists": exists["depth"],
                    "mask_path": rels["mask"],
                    "mask_exists": exists["mask"],
                    "trajectory_path": rels["trajectories"],
                    "trajectory_exists": exists["trajectories"],
                    "instance_id_map_path": rels["instance_id_map"],
                    "instance_id_map_exists": exists["instance_id_map"],
                    "object_ids_declared": object_ids_declared,
                    "camera_metadata_ok": camera_ok,
                }
            )
    scene_rows: list[dict[str, Any]] = []
    for scene_id, counts in sorted(scene_acc.items()):
        annotation_rows = int(counts["annotation_rows"])
        scene_rows.append(
            {
                "split": split,
                "scene_id": scene_id,
                "annotation_rows": annotation_rows,
                "image_exists_count": int(counts["image_exists"]),
                "depth_exists_count": int(counts["depth_exists"]),
                "mask_exists_count": int(counts["mask_exists"]),
                "trajectory_exists_count": int(counts["trajectories_exists"]),
                "instance_id_map_exists_count": int(counts["instance_id_map_exists"]),
                "object_ids_declared_count": int(counts["object_ids_declared"]),
                "camera_metadata_ok_count": int(counts["camera_metadata_ok"]),
                "rgb_frames_exist": int(counts["image_exists"]) > 0,
                "depth_frames_exist": int(counts["depth_exists"]) > 0,
                "instance_masks_exist": int(counts["mask_exists"]) > 0,
                "object_id_maps_exist": int(counts["instance_id_map_exists"]) > 0,
                "trajectories_exist": int(counts["trajectories_exists"]) > 0,
            }
        )
    flags = {
        "rgb_frames_exist": global_counts["image_exists"] > 0,
        "depth_frames_exist": global_counts["depth_exists"] > 0,
        "camera_metadata_exist": global_counts["camera_metadata_ok"] > 0,
        "instance_masks_exist": global_counts["mask_exists"] > 0,
        "object_ids_declared": global_counts["object_ids_declared"] > 0,
        "object_id_maps_exist": global_counts["instance_id_map_exists"] > 0,
        "semantic_labels_exist": False,
        "trajectories_exist": global_counts["trajectories_exists"] > 0,
        "frame_annotations_valid": bool(annotations and annotation_error is None),
        "split_files_exist": split_dir.exists() and annotation_path.exists(),
    }
    dyn_level, dyn_level_label, blockers = _split_level(flags)
    summary = {
        "phase": "v64r2_dynamic_env",
        "created_at": utc_now(),
        "data_root": str(root),
        "data_root_rel": _rel(root),
        "split": split,
        "split_dir": str(split_dir),
        "annotation_path": str(annotation_path),
        "data_root_exists": root.exists(),
        "split_dir_exists": split_dir.exists(),
        "annotation_exists": annotation_path.exists(),
        "annotation_error": annotation_error,
        "annotation_items_total": len(annotations),
        "annotation_items_scanned": len(scan_annotations),
        "scene_count": len(scene_rows),
        "rgb_frames_exist": flags["rgb_frames_exist"],
        "depth_frames_exist": flags["depth_frames_exist"],
        "camera_metadata_exist": flags["camera_metadata_exist"],
        "instance_masks_exist": flags["instance_masks_exist"],
        "object_ids_exist": bool(flags["object_ids_declared"] and flags["object_id_maps_exist"]),
        "object_ids_declared_in_annotations": flags["object_ids_declared"],
        "semantic_labels_exist": flags["semantic_labels_exist"],
        "trajectories_exist": flags["trajectories_exist"],
        "frame_annotations_valid": flags["frame_annotations_valid"],
        "split_files_exist": flags["split_files_exist"],
        "actual_file_counts": dict(global_counts),
        "mean_image_frames_per_scene": float(mean([row["image_exists_count"] for row in scene_rows])) if scene_rows else 0.0,
        "dyn_level": dyn_level,
        "dyn_level_label": dyn_level_label,
        "dynamic_env_check_complete": True,
        "can_run_any_dynamic_experiment": dyn_level >= 1,
        "can_report_official_object_tracking": dyn_level >= 3 and flags["object_id_maps_exist"],
        "can_report_3d_4d_trajectory_metrics": dyn_level >= 4,
        "can_report_semantic_4d": dyn_level >= 5,
        "blocked_official_metric_reasons": blockers,
    }
    summary["gate"] = {
        "dynamic_env_check_complete": True,
        "dyn_level_ge_1": dyn_level >= 1,
        "dyn_level_ge_3_for_official_tracking": dyn_level >= 3,
        "dyn_level_ge_4_for_3d4d": dyn_level >= 4,
    }
    return {
        "summary": summary,
        "dataset_file_rows": file_rows,
        "split_rows": scene_rows,
    }


def write_v64r2_dynamic_env(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = _project(output_root)
    write_json(out / "dynamic_env_summary.json", payload["summary"])
    write_csv(out / "dataset_file_rows.csv", payload["dataset_file_rows"])
    write_csv(out / "split_rows.csv", payload["split_rows"])
