from __future__ import annotations

from pathlib import Path
from typing import Any

from .v47_common import ROOT, read_json, utc_now, write_csv, write_json


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def _load_dict(path: str | Path) -> dict[str, Any]:
    path_obj = _project(path)
    if not path_obj.exists():
        return {}
    payload = read_json(path_obj)
    return payload if isinstance(payload, dict) else {}


def build_v64r2_dynamic_metrics(
    *,
    dynamic_env_summary_path: str | Path = "outputs/audit/v64r2_dynamic_env/dynamic_env_summary.json",
    adapter_summary_path: str | Path = "outputs/audit/v64r2_dynamic_adapter/adapter_summary.json",
    split_name: str = "small",
) -> dict[str, Any]:
    env = _load_dict(dynamic_env_summary_path)
    adapter = _load_dict(adapter_summary_path)
    dyn_level = int(env.get("dyn_level") or 0)
    adapter_pass = bool(adapter.get("gate", {}).get("pass")) if isinstance(adapter.get("gate"), dict) else False
    metric_rows: list[dict[str, Any]] = []
    official_tracking_allowed = bool(env.get("can_report_official_object_tracking"))
    official_4d_allowed = bool(env.get("can_report_3d_4d_trajectory_metrics"))
    for metric_name in [
        "IDF1",
        "ID_switches",
        "fragmentation",
        "track_purity",
        "track_coverage",
        "reactivation_precision",
        "4D_IoU_over_time",
        "trajectory_EPE",
        "object_trajectory_consistency",
    ]:
        official_required = metric_name in {"IDF1", "ID_switches", "fragmentation", "track_purity", "track_coverage", "reactivation_precision"}
        metric_rows.append(
            {
                "split": split_name,
                "metric": metric_name,
                "value": None,
                "baseline_value": None,
                "status": "blocked_gt_level_or_adapter_not_available",
                "official_metric": official_required or metric_name.startswith("4D") or "trajectory" in metric_name,
                "allowed_by_data_level": (official_tracking_allowed if official_required else official_4d_allowed),
                "adapter_pass": adapter_pass,
                "reason": (
                    "actual instance/object-id masks are missing, so official tracking metrics cannot be reported"
                    if official_required and not official_tracking_allowed
                    else (
                        "actual depth/camera/object-id trajectory GT level is insufficient for 3D/4D metrics"
                        if not official_required and not official_4d_allowed
                        else "dynamic adapter did not produce tracks"
                    )
                ),
            }
        )
    gate = {
        "metrics_finite": False,
        "visualization_generated": False,
        "no_gt_leakage": True,
        "official_metrics_blocked_when_gt_missing": not official_tracking_allowed and not official_4d_allowed,
    }
    summary = {
        "phase": f"v64r2_dynamic_{split_name}",
        "created_at": utc_now(),
        "input_paths": {
            "dynamic_env_summary": _rel(dynamic_env_summary_path),
            "adapter_summary": _rel(adapter_summary_path),
        },
        "split": split_name,
        "dyn_level": dyn_level,
        "dyn_level_label": env.get("dyn_level_label"),
        "dynamic_metric_status": "blocked_gt_level_or_adapter_not_available",
        "can_report_official_object_tracking": official_tracking_allowed,
        "can_report_3d_4d_trajectory_metrics": official_4d_allowed,
        "adapter_pass": adapter_pass,
        "gate": gate,
        "blocked_reason": adapter.get("blocked_reason")
        or "Dynamic data level does not allow official metrics and adapter produced no tracks.",
    }
    gate["pass"] = bool(gate["metrics_finite"] and gate["visualization_generated"] and gate["no_gt_leakage"])
    return {"summary": summary, "dynamic_metric_rows": metric_rows}


def write_v64r2_dynamic_metrics(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = _project(output_root)
    prefix = "dynamic_small" if payload["summary"]["split"] == "small" else "dynamic_full"
    write_json(out / f"{prefix}_summary.json", payload["summary"])
    write_csv(out / "dynamic_metric_rows.csv", payload["dynamic_metric_rows"])
