from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent

PHASE_ID = "v93_phase7_adaptive_d4rt_sampling"
RUN_ID = "v93_phase7_adaptive_d4rt_sampling_summary"
OUT = ROOT / "outputs/audit/v93_phase7_adaptive_d4rt_sampling"

PHASE0 = ROOT / "outputs/audit/v93_phase0_contract/summary.json"
PHASE2 = ROOT / "outputs/audit/v93_phase2_d4rt_edge_sampling_diagnostic"
HR2_RECOMPUTE = ROOT / "outputs/audit/v92_phase3_d4rt_highres_recompute/HR2_grid16_local_window_safe"
HR2_BRIDGE = ROOT / "outputs/audit/v92_phase3_d4rt_highres_hr2_grid16"
HR2_READOUT = ROOT / "outputs/audit/v92_phase3_hr2_same_readout_adaptive_materialization"
A512_RECOMPUTE = ROOT / "outputs/audit/v93_phase7_adaptive_d4rt_recompute/A512_adaptive_edge_conflict"
A512_BRIDGE = ROOT / "outputs/audit/v93_phase7_adaptive_d4rt_sampling/A512_adaptive_edge_conflict_bridge"
A512_READOUT = ROOT / "outputs/audit/v93_phase7_A512_same_readout_adaptive_materialization"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
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


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)


def _resolve(path: Path | str) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ROOT.name:
        return REPO_ROOT / path
    return ROOT / path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _scene_runtime_rows(root: Path, sampling_plan_id: str, method_family: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(root.glob("*/stride_5/summary.json")):
        summary = _read_json(summary_path)
        scene = str(summary.get("scene", summary.get("scene_id", summary_path.parents[1].name)))
        rows.append(
            {
                "schema_version": "stream4d_v93_phase7_runtime_memory_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "sampling_plan_id": sampling_plan_id,
                "method_family": method_family,
                "scene_id": scene,
                "window_id": "ALL_WINDOWS",
                "window_count": summary.get("window_count", ""),
                "frame_count": summary.get("frame_count", ""),
                "query_budget_per_frame": summary.get("query_budget_per_frame", summary.get("grid_points_per_frame", "")),
                "source_count": summary.get("source_count", ""),
                "valid_observation_count": summary.get("valid_observation_count", ""),
                "runtime_sec_total": summary.get("runtime_sec_total", summary.get("duration_sec", "")),
                "duration_sec": summary.get("duration_sec", ""),
                "peak_memory_gb": summary.get("peak_memory_gb_max", ""),
                "cuda_visible_devices": summary.get("cuda_visible_devices", ""),
                "device": summary.get("device", ""),
                "source_artifact": _rel(summary_path),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return rows


def _aggregate_a512_strata(root: Path) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*/stride_5/sampling_stratum_rows.csv")):
        for row in _read_csv(path):
            name = str(row.get("stratum_name", ""))
            item = groups.setdefault(
                name,
                {
                    "schema_version": "stream4d_v93_phase7_sampling_stratum_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "sampling_plan_id": "A512_adaptive_edge_conflict",
                    "stratum_name": name,
                    "query_budget_per_frame": _int(row.get("query_budget_per_frame"), 0),
                    "expected_query_budget": 0,
                    "actual_query_count": 0,
                    "scene_count": set(),
                    "normalization_weight": row.get("normalization_weight", ""),
                    "uses_gt_for_routing": False,
                    "uses_future": False,
                },
            )
            item["expected_query_budget"] += _int(row.get("expected_query_budget"), 0)
            item["actual_query_count"] += _int(row.get("actual_query_count"), 0)
            item["scene_count"].add(str(row.get("scene_id", "")))
    out: list[dict[str, Any]] = []
    for item in groups.values():
        item = dict(item)
        scenes = item.pop("scene_count")
        item["scene_count"] = len({scene for scene in scenes if scene})
        item["actual_to_expected_ratio"] = float(item["actual_query_count"] / max(1, item["expected_query_budget"]))
        out.append(item)
    return sorted(out, key=lambda row: str(row.get("stratum_name", "")))


def _symlink_or_manifest(out_path: Path, target: Path) -> dict[str, Any]:
    if out_path.exists() or out_path.is_symlink():
        out_path.unlink()
    rel_target = os.path.relpath(target, out_path.parent)
    out_path.symlink_to(rel_target)
    return {
        "path": _rel(out_path),
        "target": _rel(target),
        "target_sha256": _sha256(target) if target.exists() else "",
        "is_symlink": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = _resolve(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    created_at = _created_at()
    phase0 = _read_json(PHASE0)
    phase2 = _read_json(PHASE2 / "summary.json")
    hr2_summary = _read_json(HR2_BRIDGE / "summary.json")
    hr2_best = _read_json(HR2_READOUT / "best_variant_summary.json")
    a512_bridge = _read_json(A512_BRIDGE / "summary.json")
    a512_best = _read_json(A512_READOUT / "best_variant_summary.json")
    a512_readout_summary = _read_json(A512_READOUT / "summary.json")
    locked_control_mv = _num(phase0.get("best_control_MV_AP_window"))
    locked_control_ap50 = _num(phase0.get("best_control_MV_AP50_window"))
    best_uniform_mv = _num(hr2_best.get("mean_MV_AP_window"))
    best_uniform_ap50 = _num(hr2_best.get("mean_MV_AP50_window"))
    a512_mv = _num(a512_best.get("mean_MV_AP_window"))
    a512_ap50 = _num(a512_best.get("mean_MV_AP50_window"))
    a512_support = _num(a512_bridge.get("highres_median_carrier_support_area_ratio_unique_key"))
    hr2_support = _num(hr2_summary.get("highres_median_carrier_support_area_ratio_unique_key"))
    lowres_support = _num(phase2.get("median_carrier_support_area_ratio"))
    if not lowres_support:
        lowres_support = _num(phase2.get("median_carrier_support_area_ratio_unique_key"))

    sampling_config_rows = [
        {
            "schema_version": "stream4d_v93_phase7_sampling_config_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "sampling_plan_id": "G16_uniform_existing",
            "method_family": "uniform_existing",
            "query_budget_per_frame": 256,
            "status": "completed_existing_v92_hr2_grid16",
            "source_artifact": _rel(HR2_BRIDGE / "summary.json"),
            "uses_gt_for_routing": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v93_phase7_sampling_config_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "sampling_plan_id": "A512_adaptive_edge_conflict",
            "method_family": "adaptive_edge_conflict",
            "query_budget_per_frame": 512,
            "status": "completed_recompute_bridge_readout",
            "source_artifact": _rel(A512_RECOMPUTE),
            "uses_gt_for_routing": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v93_phase7_sampling_config_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "sampling_plan_id": "A1024_adaptive_edge_conflict_uncertainty",
            "method_family": "adaptive_edge_conflict_uncertainty",
            "query_budget_per_frame": 1024,
            "status": "not_run_stop_rule_after_A512_density_not_readout_and_runtime_over_budget",
            "source_artifact": _rel(PHASE2 / "sampling_stratum_rows.csv"),
            "uses_gt_for_routing": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v93_phase7_sampling_config_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "sampling_plan_id": "G32_uniform_subset_if_affordable",
            "method_family": "uniform_subset",
            "query_budget_per_frame": 1024,
            "status": "not_run_stop_rule_after_A512_runtime_over_budget",
            "source_artifact": "",
            "uses_gt_for_routing": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v93_phase7_sampling_config_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "sampling_plan_id": "A512_boundary_only/A512_conflict_only/A512_interior_only",
            "method_family": "ablation_controls",
            "query_budget_per_frame": 512,
            "status": "not_run_stop_rule_after_primary_A512_no_uniform_gain",
            "source_artifact": "",
            "uses_gt_for_routing": False,
            "uses_future": False,
        },
    ]
    sampling_stratum_rows = _aggregate_a512_strata(A512_RECOMPUTE)

    runtime_rows = _scene_runtime_rows(HR2_RECOMPUTE, "G16_uniform_existing", "uniform_existing")
    runtime_rows.extend(_scene_runtime_rows(A512_RECOMPUTE, "A512_adaptive_edge_conflict", "adaptive_edge_conflict"))
    hr2_runtime_by_scene = {row["scene_id"]: _num(row.get("runtime_sec_total")) for row in runtime_rows if row["sampling_plan_id"] == "G16_uniform_existing"}
    for row in runtime_rows:
        if row["sampling_plan_id"] != "A512_adaptive_edge_conflict":
            continue
        base = hr2_runtime_by_scene.get(row["scene_id"], 0.0)
        row["runtime_ratio_vs_G16"] = _num(row.get("runtime_sec_total")) / max(1e-6, base)
        row["runtime_budget_pass_vs_G16_2x"] = row["runtime_ratio_vs_G16"] <= 2.0

    readout_metric_rows = [
        {
            "schema_version": "stream4d_v93_phase7_readout_metric_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "sampling_plan_id": "G16_uniform_existing",
            "variant_id": hr2_best.get("variant_id", ""),
            "scene_id": "ALL_DEV",
            "MV_AP_window": best_uniform_mv,
            "MV_AP50_window": best_uniform_ap50,
            "MV_AP25_window": _num(hr2_best.get("mean_MV_AP25_window")),
            "median_carrier_support_area_ratio": hr2_support,
            "projection_jitter_p90": hr2_summary.get("highres_projection_jitter_p90_global", ""),
            "mask_membership_flip_rate": hr2_summary.get("highres_mask_membership_flip_rate_median", ""),
            "best_control_MV_AP_window": locked_control_mv,
            "best_control_MV_AP50_window": locked_control_ap50,
            "source_artifact": _rel(HR2_READOUT / "best_variant_summary.json"),
            "status": "completed_existing_uniform",
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v93_phase7_readout_metric_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "sampling_plan_id": "A512_adaptive_edge_conflict",
            "variant_id": a512_best.get("variant_id", ""),
            "scene_id": "ALL_DEV",
            "MV_AP_window": a512_mv,
            "MV_AP50_window": a512_ap50,
            "MV_AP25_window": _num(a512_best.get("mean_MV_AP25_window")),
            "median_carrier_support_area_ratio": a512_support,
            "projection_jitter_p90": a512_bridge.get("highres_projection_jitter_p90_global", ""),
            "mask_membership_flip_rate": a512_bridge.get("highres_mask_membership_flip_rate_median", ""),
            "best_uniform_MV_AP_window": best_uniform_mv,
            "best_uniform_MV_AP50_window": best_uniform_ap50,
            "adaptive_minus_uniform_MV_AP_window": a512_mv - best_uniform_mv,
            "adaptive_minus_uniform_MV_AP50_window": a512_ap50 - best_uniform_ap50,
            "best_control_MV_AP_window": locked_control_mv,
            "best_control_MV_AP50_window": locked_control_ap50,
            "adaptive_minus_control_MV_AP_window": a512_mv - locked_control_mv,
            "adaptive_minus_control_MV_AP50_window": a512_ap50 - locked_control_ap50,
            "uniform_gain_gate_pass": a512_mv >= best_uniform_mv + 0.004 and a512_ap50 >= best_uniform_ap50 + 0.008,
            "control_gate_pass": a512_mv >= locked_control_mv + 0.008,
            "source_artifact": _rel(A512_READOUT / "best_variant_summary.json"),
            "status": "completed_adaptive_no_gate",
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
    ]
    runtime_fail = any(
        row.get("sampling_plan_id") == "A512_adaptive_edge_conflict" and not bool(row.get("runtime_budget_pass_vs_G16_2x"))
        for row in runtime_rows
    )
    density_improved = a512_support > hr2_support and a512_support > lowres_support
    uniform_gain_pass = a512_mv >= best_uniform_mv + 0.004 and a512_ap50 >= best_uniform_ap50 + 0.008
    control_gate_pass = a512_mv >= locked_control_mv + 0.008
    decision = (
        "PASS_V93_PHASE7_ADAPTIVE_D4RT"
        if uniform_gain_pass and control_gate_pass and not runtime_fail
        else "NO_GO_V93_PHASE7_D4RT_DENSITY_NOT_READOUT_RUNTIME_OVER_BUDGET"
    )
    failure_rows = [
        {
            "schema_version": "stream4d_v93_phase7_failure_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "sampling_plan_id": "A512_adaptive_edge_conflict",
            "failure_type": "D4RT_DENSITY_NOT_READOUT",
            "density_improved_vs_G16": density_improved,
            "median_support_area_ratio_G16": hr2_support,
            "median_support_area_ratio_A512": a512_support,
            "adaptive_minus_uniform_MV_AP_window": a512_mv - best_uniform_mv,
            "adaptive_minus_uniform_MV_AP50_window": a512_ap50 - best_uniform_ap50,
            "repair_direction": "Do not increase query count blindly; return to uncertainty-aware readout/field inference because support density improved without AP gain.",
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v93_phase7_failure_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "sampling_plan_id": "A512_adaptive_edge_conflict",
            "failure_type": "ADAPTIVE_RUNTIME_OVER_BUDGET",
            "max_runtime_ratio_vs_G16": max([_num(row.get("runtime_ratio_vs_G16"), 0.0) for row in runtime_rows], default=0.0),
            "runtime_budget": "A512 runtime must be <= 2x G16 runtime",
            "repair_direction": "Avoid A1024/G32 escalation unless readout changes justify another recompute.",
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
    ]

    _write_csv(out / "sampling_config_rows.csv", sampling_config_rows)
    _write_csv(out / "sampling_stratum_rows.csv", sampling_stratum_rows)
    _write_csv(out / "readout_metric_rows.csv", readout_metric_rows)
    _write_csv(out / "runtime_memory_rows.csv", runtime_rows)
    _write_csv(out / "variant_failure_rows.csv", failure_rows)
    symlinks = {
        "d4rt_observation_rows": _symlink_or_manifest(out / "d4rt_observation_rows.csv", A512_BRIDGE / "highres_carrier_observation_rows.csv"),
        "d4rt_support_rows": _symlink_or_manifest(out / "d4rt_support_rows.csv", A512_BRIDGE / "highres_native_carrier_support_rows.csv"),
    }
    summary = {
        "schema": "stream4d_v93_phase7_adaptive_d4rt_sampling_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "decision": decision,
        "created_at": created_at,
        "duration_sec": "",
        "G16_uniform_existing_MV_AP_window": best_uniform_mv,
        "G16_uniform_existing_MV_AP50_window": best_uniform_ap50,
        "A512_MV_AP_window": a512_mv,
        "A512_MV_AP50_window": a512_ap50,
        "A512_minus_G16_MV_AP_window": a512_mv - best_uniform_mv,
        "A512_minus_G16_MV_AP50_window": a512_ap50 - best_uniform_ap50,
        "locked_best_control_MV_AP_window": locked_control_mv,
        "locked_best_control_MV_AP50_window": locked_control_ap50,
        "A512_minus_locked_control_MV_AP_window": a512_mv - locked_control_mv,
        "A512_minus_locked_control_MV_AP50_window": a512_ap50 - locked_control_ap50,
        "lowres_median_support_area_ratio": lowres_support,
        "G16_median_support_area_ratio": hr2_support,
        "A512_median_support_area_ratio": a512_support,
        "density_improved": density_improved,
        "uniform_gain_gate_pass": uniform_gain_pass,
        "control_gate_pass": control_gate_pass,
        "runtime_budget_pass": not runtime_fail,
        "A512_readout_runtime_sec": a512_readout_summary.get("runtime_sec", ""),
        "A512_bridge_duration_sec": a512_bridge.get("duration_sec", ""),
        "A512_readout_row_counts": a512_readout_summary.get("row_counts", {}),
        "stop_rule": "D4RT_DENSITY_NOT_READOUT; A1024/G32/ablation controls not run after primary A512 improved support density but failed AP and runtime gates.",
        "symlinked_large_artifacts": symlinks,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_json(out / "summary.json", summary)
    sha_paths = [
        out / "summary.json",
        out / "sampling_config_rows.csv",
        out / "sampling_stratum_rows.csv",
        out / "readout_metric_rows.csv",
        out / "runtime_memory_rows.csv",
        out / "variant_failure_rows.csv",
    ]
    _write_json(out / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in sha_paths if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v93 Phase7 adaptive D4RT sampling summary from completed A512/G16 artifacts.")
    parser.add_argument("--output-root", default=str(OUT))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
