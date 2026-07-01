from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .v47_common import ROOT, utc_now, write_csv, write_json


DEFAULT_OBJECT_BANK_ROWS = "outputs/audit/v65_soma_object_bank/soma_object_bank_rows.csv"
DEFAULT_OBJECT_SUPPORT_ROWS = "outputs/audit/v65_soma_object_bank/soma_object_support_rows.csv"


@dataclass(frozen=True)
class V65SOMAEvalAdapterConfig:
    object_bank_rows_path: str | Path = DEFAULT_OBJECT_BANK_ROWS
    object_support_rows_path: str | Path = DEFAULT_OBJECT_SUPPORT_ROWS
    output_config: str = "v65_soma_object_bank_eval_bridge"
    output_root: str | Path = "outputs/audit/v65_soma_eval_adapter"
    split_path: str | Path = "splits/scannet_v6_probe5.txt"
    export_nn_radius: float = 0.05
    export_mask_sample_stride: int = 2
    export_mask_max_pixels: int = 50000
    export_min_points_per_object: int = 1
    export_score_mode: str = "area"


def build_scene_object_dicts(
    object_bank_rows: Iterable[dict[str, Any]],
    object_support_rows: Iterable[dict[str, Any]],
) -> dict[str, dict[int, dict[str, Any]]]:
    """Build ScanNetExporter object dictionaries from method-safe object-bank view-mask support.

    The returned object dict is still a 2D-support carrier. Turning it into AP
    point masks requires a diagnostic/evaluation bridge that may use ScanNet
    depth/pose/mesh; this function itself does not read GT geometry.
    """

    object_meta: dict[tuple[str, str], dict[str, Any]] = {}
    for row in object_bank_rows:
        scene = _scene(row)
        history_id = str(row.get("history_id", ""))
        if scene and history_id:
            object_meta[(scene, history_id)] = row

    grouped: dict[tuple[str, str], dict[tuple[int, int], float]] = defaultdict(dict)
    for row in object_support_rows:
        scene = _scene(row)
        history_id = str(row.get("history_id", ""))
        frame_id = _int_or_none(row.get("frame_id"))
        mask_id = _int_or_none(row.get("observed_mask_id"))
        if mask_id is None:
            parsed = _parse_mask_observation_id(str(row.get("support_mask_observation_id", "")))
            if parsed:
                _obs_scene, frame_id_from_obs, mask_id_from_obs = parsed
                frame_id = frame_id if frame_id is not None else frame_id_from_obs
                mask_id = mask_id_from_obs
        if not scene or not history_id or frame_id is None or mask_id is None or mask_id <= 0:
            continue
        key = (int(frame_id), int(mask_id))
        weight = _float_or_default(row.get("confidence"), 1.0)
        grouped[(scene, history_id)][key] = max(weight, grouped[(scene, history_id)].get(key, 0.0))

    scene_dicts: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for scene, history_id in sorted(grouped):
        object_index = len(scene_dicts[scene])
        mask_list = [
            (int(frame_id), int(mask_id), float(weight))
            for (frame_id, mask_id), weight in sorted(grouped[(scene, history_id)].items())
        ]
        meta = object_meta.get((scene, history_id), {})
        scene_dicts[scene][object_index] = {
            "mask_list": mask_list,
            "carrier_ids": np.empty((0,), dtype=np.int64),
            "history_id": history_id,
            "object_id": str(meta.get("object_id", history_id.rsplit("|", 1)[-1])),
            "score": _float_or_default(meta.get("confidence"), float(len(mask_list))),
            "area_score": float(len(mask_list)),
            "source_kind": "soma_object_bank_view_mask_support",
        }
    return dict(scene_dicts)


def summarize_scene_object_dicts(scene_dicts: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    scene_object_counts = {scene: len(objects) for scene, objects in sorted(scene_dicts.items())}
    scene_mask_counts = {
        scene: int(sum(len(obj.get("mask_list", [])) for obj in objects.values()))
        for scene, objects in sorted(scene_dicts.items())
    }
    return {
        "scene_count": len(scene_dicts),
        "object_count": int(sum(scene_object_counts.values())),
        "mask_observation_count": int(sum(scene_mask_counts.values())),
        "scene_object_counts": scene_object_counts,
        "scene_mask_observation_counts": scene_mask_counts,
    }


def build_eval_adapter_summary(
    *,
    cfg: V65SOMAEvalAdapterConfig,
    scene_dicts: dict[str, dict[int, dict[str, Any]]],
    scene_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    base = summarize_scene_object_dicts(scene_dicts)
    ok_rows = [row for row in scene_rows if row.get("ok")]
    support_kind_counts = Counter()
    support_rows = _read_csv_large(_project(cfg.object_support_rows_path))
    for row in support_rows:
        support_kind_counts[str(row.get("support_kind", ""))] += 1
    summary = {
        "phase": "v65_soma_eval_adapter",
        "created_at": utc_now(),
        "output_config": cfg.output_config,
        "adapter_contract": "evaluation-only: SOMA object-bank view-mask support backprojected through ScanNet depth/pose/mesh",
        **base,
        "scene_rows": scene_rows,
        "ok_scene_count": len(ok_rows),
        "num_exported_objects_total": int(sum(int(float(row.get("num_exported_objects", 0))) for row in ok_rows)),
        "num_exported_points_total": int(sum(int(float(row.get("num_exported_points", 0))) for row in ok_rows)),
        "support_kind_counts": dict(sorted(support_kind_counts.items())),
        "uses_gt_for_prediction": False,
        "uses_gt_geometry_for_inference": False,
        "uses_rgbd_for_evaluation_support": True,
        "uses_rgbd_pose_mesh_for_export": True,
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
        "method_ap_available": False,
        "diagnostic_ap_export_available": bool(ok_rows),
        "input_paths": {
            "object_bank_rows": _rel(cfg.object_bank_rows_path),
            "object_support_rows": _rel(cfg.object_support_rows_path),
            "split_path": _rel(cfg.split_path),
        },
        "gate": {
            "object_bank_view_mask_support_available": base["mask_observation_count"] > 0,
            "no_gt_geometry_for_inference": True,
            "diagnostic_only": True,
            "forbidden_for_method_table": True,
            "exported_prediction_files": len(ok_rows) > 0,
        },
    }
    summary["gate"]["pass"] = bool(all(summary["gate"].values()))
    return summary


def write_v65_soma_eval_adapter(output_root: str | Path, payload: dict[str, Any]) -> dict[str, str]:
    root = _project(output_root)
    paths = {
        "summary": root / "soma_eval_adapter_summary.json",
        "scene_rows": root / "soma_eval_adapter_scene_rows.csv",
    }
    write_json(paths["summary"], payload["summary"])
    write_csv(paths["scene_rows"], payload["scene_rows"])
    return {name: _rel(path) for name, path in paths.items()}


def read_inputs(cfg: V65SOMAEvalAdapterConfig) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    return _read_csv_large(_project(cfg.object_bank_rows_path)), _read_csv_large(_project(cfg.object_support_rows_path))


def read_split(path: str | Path) -> list[str]:
    path_obj = _project(path)
    return [line.strip() for line in path_obj.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_csv_large(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def _scene(row: dict[str, Any]) -> str:
    return str(row.get("scene_id") or row.get("scene") or "")


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _float_or_default(value: Any, default: float) -> float:
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _parse_mask_observation_id(obs_id: str) -> tuple[str, int, int] | None:
    parts = str(obs_id).split(":")
    if len(parts) == 4 and parts[0] == "m":
        scene, frame, mask = parts[1], parts[2], parts[3]
    elif len(parts) == 3:
        scene, frame, mask = parts
    else:
        return None
    try:
        return scene, int(float(frame)), int(float(mask))
    except ValueError:
        return None


__all__ = [
    "DEFAULT_OBJECT_BANK_ROWS",
    "DEFAULT_OBJECT_SUPPORT_ROWS",
    "V65SOMAEvalAdapterConfig",
    "build_eval_adapter_summary",
    "build_scene_object_dicts",
    "read_inputs",
    "read_split",
    "summarize_scene_object_dicts",
    "write_v65_soma_eval_adapter",
]
