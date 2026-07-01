from __future__ import annotations

from pathlib import Path
from typing import Any

from .v47_common import write_csv, write_json
from .v64r2_dynamic_env import build_v64r2_dynamic_env
from .v65_common import project, sha256_file


DYNAMIC_ROOT = "outputs/audit/v65_dynamic_data"


def build_v65_dynamic_data(
    *,
    data_root: str | Path = "data/dynamic-replica/v2",
    split: str = "valid",
    max_annotation_rows: int = 0,
) -> dict[str, Any]:
    env_payload = build_v64r2_dynamic_env(data_root=data_root, split=split, max_annotation_rows=max_annotation_rows)
    env = env_payload["summary"]
    actual = env.get("actual_file_counts", {})
    total = int(env.get("annotation_items_scanned") or 0)
    summary = {
        "phase": "v65_dynamic_data",
        "data_root": env.get("data_root"),
        "split": split,
        "annotation_items_total": env.get("annotation_items_total"),
        "annotation_items_scanned": total,
        "actual_images_count": int(actual.get("image_exists", 0)),
        "actual_depth_count": int(actual.get("depth_exists", 0)),
        "actual_mask_count": int(actual.get("mask_exists", 0)),
        "actual_instance_id_map_count": int(actual.get("instance_id_map_exists", 0)),
        "actual_object_id_count": int(actual.get("object_ids_declared", 0)) if env.get("object_ids_exist") else 0,
        "annotation_declared_object_id_count": int(actual.get("object_ids_declared", 0)),
        "actual_trajectory_count": int(actual.get("trajectories_exists", 0)),
        "actual_file_missing_rate": _missing_rate(total, actual),
        "dyn_level": env.get("dyn_level"),
        "dyn_level_label": env.get("dyn_level_label"),
        "blocked_official_metric_reasons": env.get("blocked_official_metric_reasons", []),
        "can_report_IDF1": bool(env.get("can_report_official_object_tracking")),
        "can_report_IDSW": bool(env.get("can_report_official_object_tracking")),
        "can_report_4D_IoU": bool(env.get("can_report_3d_4d_trajectory_metrics")),
        "can_report_semantic_AP": bool(env.get("can_report_semantic_4d")),
        "dynamic_status": "NO_GO_DYNAMIC_DATA"
        if not env.get("can_report_official_object_tracking")
        else "GO_DYNAMIC_OFFICIAL_TRACKING_DATA",
        "source_scanner": "stream4d_native.v64r2_dynamic_env.build_v64r2_dynamic_env",
        "gate": {
            "dynamic_env_check_complete": bool(env.get("dynamic_env_check_complete")),
            "official_dynamic_metrics_only_if_dyn_level_ge_3": not env.get("can_report_official_object_tracking")
            or int(env.get("dyn_level") or 0) >= 3,
            "trajectory_metrics_only_if_dyn_level_ge_4": not env.get("can_report_3d_4d_trajectory_metrics")
            or int(env.get("dyn_level") or 0) >= 4,
            "semantic_metrics_only_if_dyn_level_ge_5": not env.get("can_report_semantic_4d")
            or int(env.get("dyn_level") or 0) >= 5,
        },
    }
    summary["gate"]["pass"] = bool(all(summary["gate"].values()))
    return {
        "summary": summary,
        "file_existence_rows": env_payload["dataset_file_rows"],
        "split_rows": env_payload["split_rows"],
        "metric_permission_rows": _metric_permission_rows(summary),
    }


def write_v65_dynamic_data(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = project(output_root)
    write_json(out / "dynamic_data_summary.json", payload["summary"])
    write_csv(out / "file_existence_rows.csv", payload["file_existence_rows"])
    write_csv(out / "split_rows.csv", payload["split_rows"])
    write_csv(out / "metric_permission_rows.csv", payload["metric_permission_rows"])


def _missing_rate(total: int, actual: dict[str, Any]) -> float | None:
    if total <= 0:
        return None
    required_actual = [
        int(actual.get("image_exists", 0)),
        int(actual.get("depth_exists", 0)),
        int(actual.get("mask_exists", 0)),
        int(actual.get("instance_id_map_exists", 0)),
        int(actual.get("trajectories_exists", 0)),
    ]
    present = sum(min(total, count) for count in required_actual)
    return 1.0 - present / float(total * len(required_actual))


def _metric_permission_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    dyn_level = int(summary.get("dyn_level") or 0)
    rows = [
        ("qualitative_self_consistency", 1, True),
        ("geometry_diagnostic", 2, False),
        ("IDF1", 3, False),
        ("IDSW", 3, False),
        ("4D_IoU", 4, False),
        ("semantic_AP", 5, False),
    ]
    out = []
    for metric, required, allowed_when_missing_gt in rows:
        allowed = dyn_level >= required
        out.append(
            {
                "metric_name": metric,
                "required_dyn_level": required,
                "current_dyn_level": dyn_level,
                "allowed": allowed,
                "allowed_when_missing_gt": allowed_when_missing_gt,
                "status": "allowed" if allowed else "blocked_data_level",
                "blocked_reason": ""
                if allowed
                else "; ".join(summary.get("blocked_official_metric_reasons", [])),
            }
        )
    return out


def output_hashes(output_root: str | Path = DYNAMIC_ROOT) -> dict[str, str]:
    out = project(output_root)
    return {
        name: sha256_file(out / name)
        for name in [
            "dynamic_data_summary.json",
            "file_existence_rows.csv",
            "split_rows.csv",
            "metric_permission_rows.csv",
        ]
    }
