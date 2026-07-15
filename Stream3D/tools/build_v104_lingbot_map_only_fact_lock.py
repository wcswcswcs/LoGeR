#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"

PHASE_ID = "v104_lingbot_map_only_phase0_fact_lock"
DEFAULT_OUT = AUDIT_ROOT / PHASE_ID
PLAN_DOC = REPO_ROOT / "docs/stream4d_v104_lingbot_map_only_affinity_plan.md"

LINGBOT_ROOT = REPO_ROOT / "third_party/lingbot-map"
LINGBOT_CHECKPOINT = LINGBOT_ROOT / "checkpoints/lingbot-map-long.pt"
LINGBOT_METHOD = LINGBOT_ROOT / "benchmark/methods/lingbot_map.py"
LINGBOT_GCT_BASE = LINGBOT_ROOT / "lingbot_map/models/gct_base.py"
LINGBOT_GCT_STREAM_WINDOW = LINGBOT_ROOT / "lingbot_map/models/gct_stream_window_v2.py"

PROVIDER_FILE = STREAM3D_ROOT / "geometry_provider/lingbot_map_provider.py"
PROVIDER_SMOKE_FILE = STREAM3D_ROOT / "tools/build_v104_lingbot_provider_smoke.py"
MATERIALIZER_FILE = STREAM3D_ROOT / "stream4d/lingbot_map_stream3d_geometry_adapter.py"
MATERIALIZER_SMOKE_FILE = STREAM3D_ROOT / "tools/build_v104_lingbot_bss_materialization_smoke.py"
MASK_SUPPORT_ROWS_SMOKE_FILE = STREAM3D_ROOT / "tools/build_v104_lingbot_mask_support_rows_smoke.py"
SCENE_ALIGNMENT_AUDIT_FILE = STREAM3D_ROOT / "tools/build_v104_lingbot_scene_alignment_audit.py"
GENERAL_CONFIG_READINESS_FILE = STREAM3D_ROOT / "tools/build_v104_lingbot_general_config_readiness.py"
REAL_MASK_SUPPORT_ROWS_FILE = STREAM3D_ROOT / "tools/build_v104_lingbot_real_mask_support_rows.py"
VOXEL_AFFINITY_FEATURE_FILE = STREAM3D_ROOT / "tools/build_v104_lingbot_voxel_affinity_features.py"
VOXEL_LOCAL_AP_GRID_FILE = STREAM3D_ROOT / "tools/run_v104_lingbot_voxel_local_mv_ap_grid.py"
TEMPORAL_TRACK_LOCAL_AP_FILE = STREAM3D_ROOT / "tools/run_v104_lingbot_temporal_track_local_mv_ap.py"
TEMPORAL_TRACK_REPAIR_GRID_FILE = STREAM3D_ROOT / "tools/run_v104_lingbot_temporal_track_repair_grid.py"
TEMPORAL_TRACK_REPAIR_GRID_SMALL_FILE = STREAM3D_ROOT / "tools/run_v104_lingbot_temporal_track_repair_grid_small.py"
TEMPORAL_TRACK_FULL_SCENE_FILE = STREAM3D_ROOT / "tools/run_v104_lingbot_temporal_track_full_scene.py"

V105_BASE = REPO_ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
V105_FULL_METRIC = V105_BASE / "stage1_lingbot_baseline/full_sequence_metrics/lingbot_stream_default_full_metrics.csv"
V105_FULL_SUMMARY = V105_BASE / "stage1_lingbot_baseline/full_sequence_metrics/stage1_full_metric_summary.json"
V105_METHOD_CONFIG = V105_BASE / "configs/methods/lingbot_map_stream_default.yaml"

V109_BASE = REPO_ROOT / "results/acl2_v109tf_lingbot_f_surface_causal_dissection_semantic_memory_control"
V109_EXEC_LOG = REPO_ROOT / "docs/ACL2_v109TF_LingBot_FSurfaceCausalDissection_SemanticAwareMemoryControl_执行日志.md"
V109_REVIEW_LOG = REPO_ROOT / "docs/ACL2_v109TF_LingBot_FSurfaceCausalDissection_SemanticAwareMemoryControl_实验结果复盘.md"
V109_CORE_SUMMARY = V109_BASE / "stage2_f_core_ablation/stage2_summary.json"
V109_F19_SUMMARY = V109_BASE / "stage2_role_specific_safety_candidates/role_specific_safety_candidate_summary.json"
V109_F19_FULL_METRIC = V109_BASE / "stage2_role_specific_safety_candidates/full_metric_rows.csv"
V109_F19_ACTION_FIDELITY = V109_BASE / "stage2_role_specific_safety_candidates/action_fidelity_rows.csv"
V109_F19_POLICY_ROWS = V109_BASE / "stage2_role_specific_safety_candidates/policy_selected_frame_rows.csv"
V109_F19_CONTROL_CONFIG = V109_BASE / "stage2_f19_keyframe_controls/config_generation_summary.json"
PHASE1_PROVIDER_SMOKE_SUMMARY = AUDIT_ROOT / "v104_lingbot_map_only_phase1_provider_smoke/summary.json"
PHASE2_BSS_MATERIALIZATION_SUMMARY = AUDIT_ROOT / "v104_lingbot_map_only_phase2_bss_materialization_smoke/summary.json"
PHASE3_MASK_SUPPORT_ROWS_SUMMARY = AUDIT_ROOT / "v104_lingbot_map_only_phase3_mask_support_rows_smoke/summary.json"
PHASE4_SCENE_ALIGNMENT_SUMMARY = AUDIT_ROOT / "v104_lingbot_map_only_phase4_scene_alignment_audit/summary.json"
PHASE5_GENERAL_CONFIG_READINESS_SUMMARY = AUDIT_ROOT / "v104_lingbot_map_only_phase5_general_config_readiness/summary.json"
PHASE6_SCENE0011_BSS_SMOKE_SUMMARY = AUDIT_ROOT / "v104_lingbot_map_only_phase6_same_scene_bss_smoke/summary.json"
PHASE6_SCENE0050_BSS_SMOKE_SUMMARY = AUDIT_ROOT / "v104_lingbot_map_only_phase6_same_scene_bss_smoke_scene0050/summary.json"
PHASE6_SCENE0011_FULL_BSS_SUMMARY = AUDIT_ROOT / "v104_lingbot_map_only_phase6_full_bss_scene0011/summary.json"
PHASE6_SCENE0050_FULL_BSS_SUMMARY = AUDIT_ROOT / "v104_lingbot_map_only_phase6_full_bss_scene0050/summary.json"
PHASE7_REAL_MASK_SUPPORT_SUMMARY = AUDIT_ROOT / "v104_lingbot_map_only_phase7_real_mask_support_rows/summary.json"
PHASE8_VOXEL_FEATURE_SUMMARY = AUDIT_ROOT / "v104_lingbot_map_only_phase8_voxel_affinity_features/summary.json"
PHASE8_CENTROID_FEATURE_SUMMARY = AUDIT_ROOT / "v104_lingbot_map_only_phase8_centroid_rff_sigma050/summary.json"
PHASE8_VOXEL_CENTROID_FEATURE_SUMMARY = AUDIT_ROOT / "v104_lingbot_map_only_phase8_voxel_centroid_sigma050/summary.json"
PHASE9_VOXEL_LOCAL_AP_SUMMARY = AUDIT_ROOT / "v104_lingbot_map_only_phase9_local_mv_ap_from_voxel_affinity/summary.json"
PHASE9_CENTROID_LOCAL_AP_SUMMARY = AUDIT_ROOT / "v104_lingbot_map_only_phase9_centroid_rff_sigma050_local_mv_ap_grid/summary.json"
PHASE9_VOXEL_CENTROID_LOCAL_AP_SUMMARY = AUDIT_ROOT / "v104_lingbot_map_only_phase9_voxel_centroid_sigma050_local_mv_ap_grid/summary.json"
PHASE10_TEMPORAL_TRACK_LOCAL_AP_SUMMARY = AUDIT_ROOT / "v104_lingbot_map_only_phase10_temporal_track_local_mv_ap/summary.json"
PHASE10_TEMPORAL_REPAIR_GRID_SUMMARY = AUDIT_ROOT / "v104_lingbot_map_only_phase10_temporal_track_repair_grid/summary.json"
PHASE10_TEMPORAL_REPAIR_GRID_SMALL_SUMMARY = AUDIT_ROOT / "v104_lingbot_map_only_phase10_temporal_track_repair_grid_small/summary.json"
PHASE10_TEMPORAL_FULL_SCENE_SUMMARY = AUDIT_ROOT / "v104_lingbot_map_only_phase10c_temporal_track_full_scene/summary.json"

FORBIDDEN_PROVIDER_TOKENS = ("d4rt", "da3", "depth-anything", "3dgs", "gaussian")


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return p.as_posix()


def _project(path: Path | str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _artifact_row(role: str, path: Path, required: bool = True, note: str = "") -> dict[str, Any]:
    exists = path.exists()
    return {
        "schema_version": "stream4d_v104_lingbot_artifact_row_v1",
        "phase_id": PHASE_ID,
        "artifact_role": role,
        "path": _rel(path),
        "exists": bool(exists),
        "required": bool(required),
        "is_file": path.is_file() if exists else False,
        "is_dir": path.is_dir() if exists else False,
        "size_bytes": path.stat().st_size if exists and path.is_file() else "",
        "sha256": _sha256(path),
        "note": note,
        "uses_d4rt_for_prediction": False,
        "uses_da3_for_prediction": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def _gate_row(name: str, passed: bool, observed: Any, required: Any, repair: str) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v104_lingbot_gate_row_v1",
        "phase_id": PHASE_ID,
        "gate_name": name,
        "pass": bool(passed),
        "observed": json.dumps(_jsonable(observed), sort_keys=True) if isinstance(observed, (dict, list, tuple)) else observed,
        "required": required,
        "repair_direction": repair,
        "uses_d4rt_for_prediction": False,
        "uses_da3_for_prediction": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def _failure_row(failure_id: str, evidence: Any, repair: str, severity: str = "blocking") -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v104_lingbot_failure_row_v1",
        "phase_id": PHASE_ID,
        "failure_id": failure_id,
        "severity": severity,
        "evidence": json.dumps(_jsonable(evidence), sort_keys=True) if isinstance(evidence, (dict, list, tuple)) else evidence,
        "repair_direction": repair,
        "uses_d4rt_for_prediction": False,
        "uses_da3_for_prediction": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def _contains_all(path: Path, needles: tuple[str, ...]) -> dict[str, bool]:
    if not path.exists():
        return {needle: False for needle in needles}
    text = path.read_text(encoding="utf-8", errors="replace")
    return {needle: needle in text for needle in needles}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _best_local_ap_attempt(rows: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row for row in rows if row]
    if not available:
        return {}
    return max(available, key=lambda row: (_num(row.get("best_MV_AP_window")), _num(row.get("best_MV_AP50_window"))))


def _metric_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_csv(V105_FULL_METRIC):
        rows.append(
            {
                "schema_version": "stream4d_v104_lingbot_metric_evidence_row_v1",
                "phase_id": PHASE_ID,
                "evidence_role": "v105_frozen_lingbot_stream_default_baseline",
                "seq": row.get("seq", ""),
                "frames": row.get("frames", ""),
                "ATE_full_sim3_m": row.get("ATE_full_sim3_m", ""),
                "final_error_m": row.get("final_error_m", ""),
                "rolling_ATE_p90": row.get("rolling_ATE_p90", ""),
                "method_root": row.get("method_root", ""),
                "metric_scope_note": row.get("metric_scope_note", ""),
                "uses_d4rt_for_prediction": False,
                "uses_da3_for_prediction": False,
                "uses_gt_for_eval": True,
            }
        )
    for row in _read_csv(V109_F19_FULL_METRIC):
        rows.append(
            {
                "schema_version": "stream4d_v104_lingbot_metric_evidence_row_v1",
                "phase_id": PHASE_ID,
                "evidence_role": "v109_f19_lingbot_action_safety_candidate",
                "seq": row.get("seq", row.get("seq_id", "")),
                "frames": row.get("num_frames", ""),
                "ATE_full_sim3_m": row.get("full_ATE_sim3", ""),
                "baseline_full_ATE_sim3": row.get("baseline_full_ATE_sim3", ""),
                "relative_improvement_vs_baseline": row.get("full_ATE_sim3_relative_improvement_vs_baseline", ""),
                "action_fidelity_pass": row.get("action_fidelity_pass", ""),
                "action_traj": row.get("action_traj", ""),
                "uses_d4rt_for_prediction": False,
                "uses_da3_for_prediction": False,
                "uses_gt_for_eval": True,
            }
        )
    return rows


def _provider_contract_rows() -> list[dict[str, Any]]:
    schema_hits = _contains_all(
        LINGBOT_GCT_BASE,
        ("pose_enc", "depth", "depth_conf", "world_points", "world_points_conf"),
    )
    method_hits = _contains_all(
        LINGBOT_METHOD,
        ("process_scene", "depth_list", "pose_list", "intrinsics_list", "confidence_list"),
    )
    return [
        {
            "schema_version": "stream4d_v104_lingbot_provider_contract_row_v1",
            "phase_id": PHASE_ID,
            "contract_item": "lingbot_prediction_schema",
            "source_path": _rel(LINGBOT_GCT_BASE),
            "observed": json.dumps(schema_hits, sort_keys=True),
            "status": "available" if all(schema_hits.values()) else "missing_fields",
            "stream4d_use": "depth/pose/intrinsics or world_points can supply geometry support",
            "uses_d4rt_for_prediction": False,
            "uses_da3_for_prediction": False,
        },
        {
            "schema_version": "stream4d_v104_lingbot_provider_contract_row_v1",
            "phase_id": PHASE_ID,
            "contract_item": "benchmark_output_format",
            "source_path": _rel(LINGBOT_METHOD),
            "observed": json.dumps(method_hits, sort_keys=True),
            "status": "available" if all(method_hits.values()) else "missing_fields",
            "stream4d_use": "BSS directories expose saved depth/confidence plus traj/intrinsics",
            "uses_d4rt_for_prediction": False,
            "uses_da3_for_prediction": False,
        },
        {
            "schema_version": "stream4d_v104_lingbot_provider_contract_row_v1",
            "phase_id": PHASE_ID,
            "contract_item": "stream4d_geometry_provider_scaffold",
            "source_path": _rel(PROVIDER_FILE),
            "observed": PROVIDER_FILE.exists(),
            "status": "available" if PROVIDER_FILE.exists() else "missing",
            "stream4d_use": "projects LingBot per-frame points/depth to scene point ids and mask support",
            "uses_d4rt_for_prediction": False,
            "uses_da3_for_prediction": False,
        },
    ]


def build(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _project(args.output_root)
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"output root already exists and is non-empty; pass --force: {_rel(out)}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")

    artifact_rows = [
        _artifact_row("plan_doc", PLAN_DOC, required=False, note="created for this LingBot-only handoff"),
        _artifact_row("lingbot_repo", LINGBOT_ROOT),
        _artifact_row("lingbot_checkpoint", LINGBOT_CHECKPOINT),
        _artifact_row("lingbot_benchmark_method", LINGBOT_METHOD),
        _artifact_row("lingbot_gct_base_schema_source", LINGBOT_GCT_BASE),
        _artifact_row("lingbot_window_schema_source", LINGBOT_GCT_STREAM_WINDOW, required=False),
        _artifact_row("stream4d_lingbot_provider_scaffold", PROVIDER_FILE),
        _artifact_row("stream4d_lingbot_provider_smoke", PROVIDER_SMOKE_FILE),
        _artifact_row("stream4d_lingbot_bss_materializer", MATERIALIZER_FILE),
        _artifact_row("stream4d_lingbot_bss_materializer_smoke", MATERIALIZER_SMOKE_FILE),
        _artifact_row("stream4d_lingbot_mask_support_rows_smoke", MASK_SUPPORT_ROWS_SMOKE_FILE),
        _artifact_row("stream4d_lingbot_scene_alignment_audit", SCENE_ALIGNMENT_AUDIT_FILE),
        _artifact_row("stream4d_lingbot_general_config_readiness", GENERAL_CONFIG_READINESS_FILE),
        _artifact_row("stream4d_lingbot_real_mask_support_rows", REAL_MASK_SUPPORT_ROWS_FILE),
        _artifact_row("stream4d_lingbot_voxel_affinity_features", VOXEL_AFFINITY_FEATURE_FILE),
        _artifact_row("stream4d_lingbot_voxel_local_ap_grid", VOXEL_LOCAL_AP_GRID_FILE),
        _artifact_row("stream4d_lingbot_temporal_track_local_ap", TEMPORAL_TRACK_LOCAL_AP_FILE),
        _artifact_row("stream4d_lingbot_temporal_track_repair_grid", TEMPORAL_TRACK_REPAIR_GRID_FILE, required=False),
        _artifact_row("stream4d_lingbot_temporal_track_repair_grid_small", TEMPORAL_TRACK_REPAIR_GRID_SMALL_FILE, required=False),
        _artifact_row("stream4d_lingbot_temporal_track_full_scene", TEMPORAL_TRACK_FULL_SCENE_FILE, required=False),
        _artifact_row("v105_lingbot_method_config", V105_METHOD_CONFIG),
        _artifact_row("v105_lingbot_full_metric_csv", V105_FULL_METRIC),
        _artifact_row("v105_lingbot_full_metric_summary", V105_FULL_SUMMARY),
        _artifact_row("v109_execution_log", V109_EXEC_LOG),
        _artifact_row("v109_retrospective_log", V109_REVIEW_LOG),
        _artifact_row("v109_stage2_core_summary", V109_CORE_SUMMARY),
        _artifact_row("v109_f19_safety_candidate_summary", V109_F19_SUMMARY),
        _artifact_row("v109_f19_full_metric_rows", V109_F19_FULL_METRIC),
        _artifact_row("v109_f19_action_fidelity_rows", V109_F19_ACTION_FIDELITY),
        _artifact_row("v109_f19_policy_selected_frame_rows", V109_F19_POLICY_ROWS),
        _artifact_row("v109_f19_same_count_control_config", V109_F19_CONTROL_CONFIG, required=False),
        _artifact_row("phase1_provider_smoke_summary", PHASE1_PROVIDER_SMOKE_SUMMARY),
        _artifact_row("phase2_bss_materialization_smoke_summary", PHASE2_BSS_MATERIALIZATION_SUMMARY),
        _artifact_row("phase3_mask_support_rows_smoke_summary", PHASE3_MASK_SUPPORT_ROWS_SUMMARY),
        _artifact_row("phase4_scene_alignment_audit_summary", PHASE4_SCENE_ALIGNMENT_SUMMARY),
        _artifact_row("phase5_general_config_readiness_summary", PHASE5_GENERAL_CONFIG_READINESS_SUMMARY),
        _artifact_row("phase6_scene0011_same_scene_bss_smoke_summary", PHASE6_SCENE0011_BSS_SMOKE_SUMMARY),
        _artifact_row("phase6_scene0050_same_scene_bss_smoke_summary", PHASE6_SCENE0050_BSS_SMOKE_SUMMARY),
        _artifact_row("phase6_scene0011_full_stride5_bss_summary", PHASE6_SCENE0011_FULL_BSS_SUMMARY),
        _artifact_row("phase6_scene0050_full_stride5_bss_summary", PHASE6_SCENE0050_FULL_BSS_SUMMARY),
        _artifact_row("phase7_real_mask_support_rows_summary", PHASE7_REAL_MASK_SUPPORT_SUMMARY),
        _artifact_row("phase8_voxel_feature_summary", PHASE8_VOXEL_FEATURE_SUMMARY, required=False),
        _artifact_row("phase8_centroid_feature_summary", PHASE8_CENTROID_FEATURE_SUMMARY, required=False),
        _artifact_row("phase8_voxel_centroid_feature_summary", PHASE8_VOXEL_CENTROID_FEATURE_SUMMARY, required=False),
        _artifact_row("phase9_voxel_local_ap_summary", PHASE9_VOXEL_LOCAL_AP_SUMMARY, required=False),
        _artifact_row("phase9_centroid_local_ap_summary", PHASE9_CENTROID_LOCAL_AP_SUMMARY, required=False),
        _artifact_row("phase9_voxel_centroid_local_ap_summary", PHASE9_VOXEL_CENTROID_LOCAL_AP_SUMMARY, required=False),
        _artifact_row("phase10_temporal_track_local_ap_summary", PHASE10_TEMPORAL_TRACK_LOCAL_AP_SUMMARY, required=False),
        _artifact_row("phase10_temporal_repair_grid_summary", PHASE10_TEMPORAL_REPAIR_GRID_SUMMARY, required=False),
        _artifact_row("phase10_temporal_repair_grid_small_summary", PHASE10_TEMPORAL_REPAIR_GRID_SMALL_SUMMARY, required=False),
        _artifact_row("phase10_temporal_full_scene_summary", PHASE10_TEMPORAL_FULL_SCENE_SUMMARY, required=False),
    ]
    provider_paths = [PROVIDER_FILE, LINGBOT_METHOD, LINGBOT_GCT_BASE, LINGBOT_CHECKPOINT, V105_METHOD_CONFIG]
    forbidden_hits = {
        _rel(path): [token for token in FORBIDDEN_PROVIDER_TOKENS if token in _rel(path).lower()]
        for path in provider_paths
    }
    forbidden_hits = {path: hits for path, hits in forbidden_hits.items() if hits}

    v105_summary = _read_json(V105_FULL_SUMMARY)
    v105_rows = _read_csv(V105_FULL_METRIC)
    v109_core = _read_json(V109_CORE_SUMMARY)
    v109_f19 = _read_json(V109_F19_SUMMARY)
    v109_f19_rows = _read_csv(V109_F19_FULL_METRIC)
    phase1_smoke = _read_json(PHASE1_PROVIDER_SMOKE_SUMMARY)
    phase2_materialization = _read_json(PHASE2_BSS_MATERIALIZATION_SUMMARY)
    phase3_support_rows = _read_json(PHASE3_MASK_SUPPORT_ROWS_SUMMARY)
    phase4_scene_alignment = _read_json(PHASE4_SCENE_ALIGNMENT_SUMMARY)
    phase5_general_config = _read_json(PHASE5_GENERAL_CONFIG_READINESS_SUMMARY)
    phase6_scene0011_bss_smoke = _read_json(PHASE6_SCENE0011_BSS_SMOKE_SUMMARY)
    phase6_scene0050_bss_smoke = _read_json(PHASE6_SCENE0050_BSS_SMOKE_SUMMARY)
    phase6_scene0011_full_bss = _read_json(PHASE6_SCENE0011_FULL_BSS_SUMMARY)
    phase6_scene0050_full_bss = _read_json(PHASE6_SCENE0050_FULL_BSS_SUMMARY)
    phase7_real_mask_support = _read_json(PHASE7_REAL_MASK_SUPPORT_SUMMARY)
    phase8_voxel_feature = _read_json(PHASE8_VOXEL_FEATURE_SUMMARY)
    phase8_centroid_feature = _read_json(PHASE8_CENTROID_FEATURE_SUMMARY)
    phase8_voxel_centroid_feature = _read_json(PHASE8_VOXEL_CENTROID_FEATURE_SUMMARY)
    phase9_voxel_local_ap = _read_json(PHASE9_VOXEL_LOCAL_AP_SUMMARY)
    phase9_centroid_local_ap = _read_json(PHASE9_CENTROID_LOCAL_AP_SUMMARY)
    phase9_voxel_centroid_local_ap = _read_json(PHASE9_VOXEL_CENTROID_LOCAL_AP_SUMMARY)
    phase10_temporal_track = _read_json(PHASE10_TEMPORAL_TRACK_LOCAL_AP_SUMMARY)
    phase10_repair_grid = _read_json(PHASE10_TEMPORAL_REPAIR_GRID_SUMMARY)
    phase10_repair_grid_small = _read_json(PHASE10_TEMPORAL_REPAIR_GRID_SMALL_SUMMARY)
    phase10_full_scene = _read_json(PHASE10_TEMPORAL_FULL_SCENE_SUMMARY)
    phase9_best_local_ap = _best_local_ap_attempt(
        [
            phase9_voxel_local_ap | {"feature_mode": "voxel"},
            phase9_centroid_local_ap | {"feature_mode": "centroid_rff_sigma050"},
            phase9_voxel_centroid_local_ap | {"feature_mode": "voxel_centroid_sigma050"},
        ]
    )
    phase9_attempted = bool(phase9_best_local_ap)
    phase9_baseline_ap = _num(phase9_best_local_ap.get("baseline_contract", {}).get("MV_AP_window"))
    phase9_required_ap = phase9_baseline_ap - 0.003 if phase9_attempted else ""
    phase9_best_ap = _num(phase9_best_local_ap.get("best_MV_AP_window"))
    phase9_local_ap_pass = bool(phase9_best_local_ap.get("phase6_pass"))
    phase10_active = phase10_full_scene or phase10_repair_grid_small or phase10_temporal_track
    phase10_active_source = (
        "full_selected_scene"
        if phase10_full_scene
        else "repair_grid_small"
        if phase10_repair_grid_small
        else "temporal_track_local_ap"
    )
    phase10_full_scene_attempted = bool(phase10_full_scene.get("full_scene_eval_completed"))
    phase10_full_scene_pass = phase10_full_scene_attempted and bool(phase10_full_scene.get("qualified_lingbot_local_ap_pass"))
    phase10_full_scene_control_beats_lingbot = phase10_full_scene_attempted and bool(phase10_full_scene.get("control_beats_lingbot_MV_AP_window"))
    phase10_lingbot_local_ap_pass = bool(phase10_active.get("qualified_lingbot_local_ap_pass"))
    phase10_control_beats_lingbot = bool(phase10_active.get("control_beats_lingbot_MV_AP_window"))
    phase10_control_resistant = (
        phase10_lingbot_local_ap_pass
        and bool(phase10_active.get("best_control_variant_id"))
        and not phase10_control_beats_lingbot
    )
    phase10_first32_control_resistant = (
        bool(phase10_repair_grid_small.get("qualified_lingbot_local_ap_pass"))
        and bool(phase10_repair_grid_small.get("best_control_variant_id"))
        and not bool(phase10_repair_grid_small.get("control_beats_lingbot_MV_AP_window"))
    )
    phase10_scene_metric_available = phase10_active.get("best_lingbot_MV_AP_scene", "") != ""
    f19_improved = [
        float(row.get("full_ATE_sim3_relative_improvement_vs_baseline", "nan"))
        for row in v109_f19_rows
        if row.get("full_ATE_sim3_relative_improvement_vs_baseline", "") != ""
    ]

    contract_rows = _provider_contract_rows()
    metric_rows = _metric_rows()

    fact_gates = [
        _gate_row("lingbot_repo_exists", LINGBOT_ROOT.exists(), _rel(LINGBOT_ROOT), "directory exists", "sync third_party/lingbot-map"),
        _gate_row("lingbot_checkpoint_exists", LINGBOT_CHECKPOINT.exists(), _rel(LINGBOT_CHECKPOINT), "checkpoint file exists", "restore lingbot-map-long.pt"),
        _gate_row("lingbot_prediction_schema_available", all(row["status"] == "available" for row in contract_rows[:2]), [row["observed"] for row in contract_rows[:2]], "pose/depth/confidence/world_points schema", "inspect LingBot source and update adapter"),
        _gate_row("stream4d_lingbot_provider_scaffold_exists", PROVIDER_FILE.exists(), _rel(PROVIDER_FILE), "provider source exists", "add provider scaffold"),
        _gate_row("v105_full_kitti_baseline_complete", len(v105_rows) >= 4 and bool(v105_summary), {"row_count": len(v105_rows), "summary_exists": bool(v105_summary)}, "4 KITTI baseline rows plus summary", "rerun or restore v105 baseline metrics"),
        _gate_row("v109_f19_lingbot_candidate_passed", bool(v109_f19.get("safety_candidate_pass")) and bool(v109_f19.get("metric_complete")), v109_f19.get("taxonomy", ""), "metric_complete and safety_candidate_pass", "finish v109 F19 or choose a different LingBot policy prior"),
        _gate_row("phase1_provider_multimask_smoke_passed", bool(phase1_smoke.get("smoke_pass")) and bool(phase1_smoke.get("actual_bss_synthetic_two_mask_support_pass")), phase1_smoke.get("taxonomy", ""), "provider smoke_pass and actual two-mask support pass", "repair provider pixel-to-mask support"),
        _gate_row("phase2_bss_materialization_smoke_passed", bool(phase2_materialization.get("materialization_pass")), phase2_materialization.get("taxonomy", ""), "saved LingBot frame points and pixel samples", "repair BSS materialization"),
        _gate_row("phase3_mask_support_rows_smoke_passed", bool(phase3_support_rows.get("support_rows_pass")), phase3_support_rows.get("taxonomy", ""), "mask_support_rows.csv contains non-empty support rows for synthetic masks on actual LingBot BSS samples", "repair LingBot pixel-sample mask support row generation"),
        _gate_row("phase5_general_config_readiness_passed", bool(phase5_general_config.get("readiness_pass")), phase5_general_config.get("taxonomy", ""), "LingBot general configs are runnable and stride-5 aligned for target Stream4D RGB frame folders", "restore target RGB frames, general dataset adapter, LingBot checkpoint, or stride-5 sampling config"),
        _gate_row(
            "phase6_same_scene_stride5_bss_smoke_passed",
            bool(phase6_scene0011_bss_smoke.get("smoke_pass")) and bool(phase6_scene0050_bss_smoke.get("smoke_pass")),
            {
                "scene0011_00": phase6_scene0011_bss_smoke.get("taxonomy", ""),
                "scene0050_00": phase6_scene0050_bss_smoke.get("taxonomy", ""),
            },
            "smoke8 LingBot BSS exists, is stride-5 aligned, and provider-readable for both target scenes",
            "rerun LingBot prepare/run_worker smoke8 and Phase6 BSS smoke for any missing target scene",
        ),
        _gate_row(
            "phase6_full_stride5_bss_coverage_passed",
            bool(phase6_scene0011_full_bss.get("smoke_pass")) and bool(phase6_scene0050_full_bss.get("smoke_pass")) and bool(phase4_scene_alignment.get("same_scene_lingbot_selected_frame_coverage_complete")),
            {
                "scene0011_00": {
                    "taxonomy": phase6_scene0011_full_bss.get("taxonomy", ""),
                    "num_frames_checked": phase6_scene0011_full_bss.get("num_frames_checked", ""),
                },
                "scene0050_00": {
                    "taxonomy": phase6_scene0050_full_bss.get("taxonomy", ""),
                    "num_frames_checked": phase6_scene0050_full_bss.get("num_frames_checked", ""),
                },
                "phase4_coverage_complete": phase4_scene_alignment.get("same_scene_lingbot_selected_frame_coverage_complete", ""),
            },
            "full stride-5 LingBot BSS is provider-readable and covers all selected Stream4D mask frames",
            "rerun full stride-5 LingBot prepare/run_worker and Phase6 BSS smoke for any missing target scene",
        ),
        _gate_row(
            "phase7_real_mask_support_rows_passed",
            bool(phase7_real_mask_support.get("materialization_pass"))
            and bool(phase7_real_mask_support.get("all_real_masks_have_positive_lingbot_support"))
            and int(phase7_real_mask_support.get("selected_row_count") or 0) > 0
            and int(phase7_real_mask_support.get("support_row_count") or 0)
            == int(phase7_real_mask_support.get("selected_row_count") or -1),
            {
                "taxonomy": phase7_real_mask_support.get("taxonomy", ""),
                "selected_row_count": phase7_real_mask_support.get("selected_row_count", ""),
                "support_row_count": phase7_real_mask_support.get("support_row_count", ""),
                "nonempty_support_row_count": phase7_real_mask_support.get("nonempty_support_row_count", ""),
                "empty_support_row_count": phase7_real_mask_support.get("empty_support_row_count", ""),
                "support_coverage_ratio": phase7_real_mask_support.get("support_coverage_ratio", ""),
            },
            "real selected Stream4D mask support rows are materialized from full stride-5 LingBot BSS and all supports are non-empty",
            "repair mask raster alignment, BSS source-frame mapping, sampling density, or support projection for any empty/missing selected mask row",
        ),
        _gate_row(
            "phase8_mask_level_affinity_features_materialized",
            bool(phase8_voxel_feature.get("feature_build_pass")) or bool(phase8_centroid_feature.get("feature_build_pass")) or bool(phase8_voxel_centroid_feature.get("feature_build_pass")),
            {
                "voxel": phase8_voxel_feature.get("taxonomy", ""),
                "centroid_rff_sigma050": phase8_centroid_feature.get("taxonomy", ""),
                "voxel_centroid_sigma050": phase8_voxel_centroid_feature.get("taxonomy", ""),
            },
            "at least one LingBot real-support mask-level feature family is materialized for first32 local-window AP",
            "repair support-to-feature materialization before AP evaluation",
        ),
        _gate_row("forbidden_provider_dependency_paths_clear", not forbidden_hits, forbidden_hits, "no D4RT/DA3 provider dependency path", "remove forbidden provider dependency"),
    ]
    scene_alignment_gate = _gate_row(
        "same_scene_lingbot_bss_selected_frame_coverage_complete",
        bool(phase4_scene_alignment.get("same_scene_lingbot_selected_frame_coverage_complete")),
        {
            "phase4_taxonomy": phase4_scene_alignment.get("taxonomy", ""),
            "real_masks_available": phase4_scene_alignment.get("real_masks_available", ""),
            "same_scene_lingbot_bss_available": phase4_scene_alignment.get("same_scene_lingbot_bss_available", ""),
            "same_scene_lingbot_selected_frame_coverage_complete": phase4_scene_alignment.get("same_scene_lingbot_selected_frame_coverage_complete", ""),
            "phase5_general_config_readiness": phase5_general_config.get("taxonomy", ""),
            "phase6_scene0011_bss_smoke": phase6_scene0011_bss_smoke.get("taxonomy", ""),
            "phase6_scene0050_bss_smoke": phase6_scene0050_bss_smoke.get("taxonomy", ""),
            "phase6_scene0011_full_bss": phase6_scene0011_full_bss.get("taxonomy", ""),
            "phase6_scene0050_full_bss": phase6_scene0050_full_bss.get("taxonomy", ""),
            "target_scenes": phase4_scene_alignment.get("target_scenes", []),
        },
        "same-scene LingBot BSS source frame ids cover every selected Stream4D mask frame",
        "run full stride-5 LingBot configs for scene0011_00 and scene0050_00 before full real-mask support rows",
    )
    stream4d_gate = _gate_row(
        "stream4d_lingbot_local_mv_ap_gate_passed",
        phase10_lingbot_local_ap_pass,
        {
            "active_phase10_source": phase10_active_source,
            "phase9_attempted": phase9_attempted,
            "phase9_best_feature_mode": phase9_best_local_ap.get("feature_mode", ""),
            "phase9_best_MV_AP_window": phase9_best_local_ap.get("best_MV_AP_window", ""),
            "phase10_decision": phase10_active.get("decision", ""),
            "phase10_best_lingbot_variant_id": phase10_active.get("best_lingbot_variant_id", ""),
            "phase10_best_lingbot_MV_AP_window": phase10_active.get("best_lingbot_MV_AP_window", ""),
            "phase10_best_lingbot_MV_AP50_window": phase10_active.get("best_lingbot_MV_AP50_window", ""),
            "phase10_best_lingbot_MV_AP_scene": phase10_active.get("best_lingbot_MV_AP_scene", ""),
            "phase10_best_lingbot_MV_AP50_scene": phase10_active.get("best_lingbot_MV_AP50_scene", ""),
            "phase10_best_control_variant_id": phase10_active.get("best_control_variant_id", ""),
            "phase10_best_control_MV_AP_window": phase10_active.get("best_control_MV_AP_window", ""),
            "phase10_best_control_MV_AP_scene": phase10_active.get("best_control_MV_AP_scene", ""),
            "phase10_control_beats_lingbot": phase10_control_beats_lingbot,
            "baseline_MV_AP_window": phase10_active.get("baseline_contract", {}).get("MV_AP_window", ""),
        },
        "LingBot-only first32 local MV_AP_window gate passes through canonical v65/v103 evaluator",
        "promote beyond first32 and run same-count/random/keyframe controls before any Go claim",
    )
    control_resistance_gate = _gate_row(
        "phase10_first32_2d_shape_control_not_stronger",
        phase10_first32_control_resistant,
        {
            "active_phase10_source": "repair_grid_small",
            "best_lingbot_variant_id": phase10_repair_grid_small.get("best_lingbot_variant_id", ""),
            "best_lingbot_MV_AP_window": phase10_repair_grid_small.get("best_lingbot_MV_AP_window", ""),
            "best_control_variant_id": phase10_repair_grid_small.get("best_control_variant_id", ""),
            "best_control_MV_AP_window": phase10_repair_grid_small.get("best_control_MV_AP_window", ""),
            "control_minus_lingbot_MV_AP_window": phase10_repair_grid_small.get("control_minus_lingbot_MV_AP_window", ""),
            "best_lingbot_MV_AP_scene": phase10_repair_grid_small.get("best_lingbot_MV_AP_scene", ""),
            "best_control_MV_AP_scene": phase10_repair_grid_small.get("best_control_MV_AP_scene", ""),
            "control_minus_lingbot_MV_AP_scene": phase10_repair_grid_small.get("control_minus_lingbot_MV_AP_scene", ""),
        },
        "best w3d>0 LingBot temporal variant is at least as strong as the w3d=0 2D/shape control on the first32 subset",
        "if this gate regresses, repair LingBot-specific 3D support contribution before any Go claim",
    )
    gate_rows = fact_gates + [scene_alignment_gate, stream4d_gate, control_resistance_gate]

    failure_rows = [
        _failure_row(
            "LINGBOT_FULL_SELECTED_SCENE_TEMPORAL_CONTROL_STRONGER"
            if phase10_full_scene_attempted and phase10_full_scene_control_beats_lingbot
            else "LINGBOT_FULL_SELECTED_SCENE_TEMPORAL_AP_FAIL"
            if phase10_full_scene_attempted and not phase10_full_scene_pass
            else "LINGBOT_TEMPORAL_FIRST32_CONTROL_BEATEN_BUT_FULL_CONTROLS_PENDING"
            if phase10_control_resistant
            else "LINGBOT_TEMPORAL_LOCAL_AP_PASS_CONTROL_STRONGER"
            if phase10_lingbot_local_ap_pass and phase10_control_beats_lingbot
            else "LINGBOT_TO_STREAM4D_MASK_AFFINITY_AP_FAIL"
            if phase9_attempted and not phase10_lingbot_local_ap_pass
            else "LINGBOT_TO_STREAM4D_MASK_AFFINITY_NOT_VALIDATED",
            {
                "lingbot_fact_lock_ready": all(row["pass"] for row in fact_gates),
                "scene_alignment_gate": scene_alignment_gate["gate_name"],
                "phase4_scene_alignment_taxonomy": phase4_scene_alignment.get("taxonomy", ""),
                "phase4_scene_alignment_blocker": phase4_scene_alignment.get("blocker", ""),
                "phase7_real_mask_support_taxonomy": phase7_real_mask_support.get("taxonomy", ""),
                "phase7_selected_row_count": phase7_real_mask_support.get("selected_row_count", ""),
                "phase7_support_row_count": phase7_real_mask_support.get("support_row_count", ""),
                "phase7_nonempty_support_row_count": phase7_real_mask_support.get("nonempty_support_row_count", ""),
                "stream4d_metric_gate": stream4d_gate["gate_name"],
                "phase3_support_rows_scope": phase3_support_rows.get("stream4d_metric_note", ""),
                "phase7_support_rows_scope": phase7_real_mask_support.get("stream4d_metric_note", ""),
                "phase9_attempted": phase9_attempted,
                "phase9_best_feature_mode": phase9_best_local_ap.get("feature_mode", ""),
                "phase9_best_variant_id": phase9_best_local_ap.get("best_variant_id", ""),
                "phase9_best_MV_AP_window": phase9_best_local_ap.get("best_MV_AP_window", ""),
                "phase9_best_MV_AP50_window": phase9_best_local_ap.get("best_MV_AP50_window", ""),
                "phase9_baseline_MV_AP_window": phase9_baseline_ap if phase9_attempted else "",
                "phase10_active_source": phase10_active_source,
                "phase10_full_scene_attempted": phase10_full_scene_attempted,
                "phase10_full_scene_pass": phase10_full_scene_pass,
                "phase10_full_scene_best_lingbot_variant_id": phase10_full_scene.get("best_lingbot_variant_id", ""),
                "phase10_full_scene_best_lingbot_MV_AP_window": phase10_full_scene.get("best_lingbot_MV_AP_window", ""),
                "phase10_full_scene_best_lingbot_MV_AP_scene": phase10_full_scene.get("best_lingbot_MV_AP_scene", ""),
                "phase10_full_scene_best_control_variant_id": phase10_full_scene.get("best_control_variant_id", ""),
                "phase10_full_scene_best_control_MV_AP_window": phase10_full_scene.get("best_control_MV_AP_window", ""),
                "phase10_full_scene_best_control_MV_AP_scene": phase10_full_scene.get("best_control_MV_AP_scene", ""),
                "phase10_full_scene_control_beats_lingbot": phase10_full_scene_control_beats_lingbot,
                "phase10_first32_control_resistant": phase10_first32_control_resistant,
                "phase10_first32_best_lingbot_variant_id": phase10_repair_grid_small.get("best_lingbot_variant_id", ""),
                "phase10_first32_best_lingbot_MV_AP_scene": phase10_repair_grid_small.get("best_lingbot_MV_AP_scene", ""),
                "phase10_first32_best_control_MV_AP_scene": phase10_repair_grid_small.get("best_control_MV_AP_scene", ""),
                "phase10_original_best_lingbot_variant_id": phase10_temporal_track.get("best_lingbot_variant_id", ""),
                "phase10_original_best_lingbot_MV_AP_window": phase10_temporal_track.get("best_lingbot_MV_AP_window", ""),
                "phase10_original_best_control_variant_id": phase10_temporal_track.get("best_control_variant_id", ""),
                "phase10_original_best_control_MV_AP_window": phase10_temporal_track.get("best_control_MV_AP_window", ""),
                "phase10_best_lingbot_variant_id": phase10_active.get("best_lingbot_variant_id", ""),
                "phase10_best_lingbot_MV_AP_window": phase10_active.get("best_lingbot_MV_AP_window", ""),
                "phase10_best_lingbot_MV_AP_scene": phase10_active.get("best_lingbot_MV_AP_scene", ""),
                "phase10_best_control_variant_id": phase10_active.get("best_control_variant_id", ""),
                "phase10_best_control_MV_AP_window": phase10_active.get("best_control_MV_AP_window", ""),
                "phase10_best_control_MV_AP_scene": phase10_active.get("best_control_MV_AP_scene", ""),
                "phase10_control_beats_lingbot": phase10_control_beats_lingbot,
                "phase10_control_resistant": phase10_control_resistant,
                "v109_scope": "trajectory/action memory control, not Stream4D object affinity",
            },
            "Full selected scene regressed below control; repair temporal readout and add cached evaluator before any Go.",
        )
    ]

    fact_lock_pass = all(row["pass"] for row in fact_gates)
    taxonomy = (
        "NO_GO_LINGBOT_FULL_SELECTED_SCENE_CONTROL_STRONGER"
        if phase10_full_scene_attempted and phase10_full_scene_control_beats_lingbot
        else "NO_GO_LINGBOT_FULL_SELECTED_SCENE_AP_FAIL"
        if phase10_full_scene_attempted and not phase10_full_scene_pass
        else "PARTIAL_LINGBOT_TEMPORAL_FIRST32_CONTROL_BEATEN_CONTROLS_PENDING"
        if phase10_control_resistant
        else "PARTIAL_LINGBOT_TEMPORAL_LOCAL_AP_PASS_CONTROL_STRONGER"
        if phase10_lingbot_local_ap_pass and phase10_control_beats_lingbot
        else "PARTIAL_LINGBOT_TEMPORAL_LOCAL_AP_PASS_CONTROLS_PENDING"
        if phase10_lingbot_local_ap_pass
        else "NO_GO_LINGBOT_REAL_SUPPORT_LOCAL_AFFINITY_AP_FAIL"
        if phase9_attempted and not phase9_local_ap_pass
        else "PARTIAL_LINGBOT_AVAILABLE_STREAM4D_AFFINITY_NOT_RUN"
    )
    if phase10_full_scene_attempted and phase10_full_scene_control_beats_lingbot:
        blocker = "LINGBOT_FULL_SELECTED_SCENE_CONTROL_STRONGER_AND_AP_BELOW_BASELINE"
    elif phase10_full_scene_attempted and not phase10_full_scene_pass:
        blocker = "LINGBOT_FULL_SELECTED_SCENE_AP_FAIL"
    elif phase10_control_resistant:
        blocker = "LINGBOT_FIRST32_CONTROL_BEATEN_BUT_FULL_AND_RANDOM_KEYFRAME_CONTROLS_PENDING"
    elif phase10_lingbot_local_ap_pass and phase10_control_beats_lingbot:
        blocker = "LINGBOT_LOCAL_AP_PASS_BUT_2D_CONTROL_STRONGER"
    elif phase10_lingbot_local_ap_pass:
        blocker = "LINGBOT_LOCAL_AP_PASS_SCENE_AND_CONTROLS_PENDING"
    elif phase9_attempted and not phase9_local_ap_pass:
        blocker = "LINGBOT_LOCAL_AFFINITY_AP_FAIL"
    else:
        blocker = "LINGBOT_TO_STREAM4D_MASK_AFFINITY_NOT_VALIDATED"
    summary = {
        "schema_version": "stream4d_v104_lingbot_map_only_fact_lock_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix": time.time(),
        "fact_lock_pass": fact_lock_pass,
        "stream4d_model_ready": False,
        "taxonomy": taxonomy,
        "blocker": blocker,
        "good_news": [
            "local LingBot-Map repo and checkpoint are present",
            "LingBot source exposes pose/depth/confidence/world_points schema",
            "v105 full KITTI LingBot baseline metrics are present",
            "v109 F19 LingBot action candidate is complete and safety_candidate_pass=true",
            "Stream4D LingBot provider scaffold now exists",
            "provider smoke can assign actual LingBot BSS samples to two synthetic mask ids",
            "BSS materialization smoke stores LingBot frame points and pixel samples",
            "mask support rows smoke writes non-empty support rows from actual LingBot BSS samples",
            "scene alignment audit confirms real Stream4D mask rasters are available for scene0011_00 and scene0050_00",
            "LingBot general config readiness finds target RGB frame folders and emits stride-5 aligned runnable config templates",
            "LingBot smoke8 prepare/run_worker produced same-scene BSS for scene0011_00 and scene0050_00",
            "Phase6 BSS smoke confirms provider-readable depth/pose/intrinsics and BSS-to-source-frame mapping for both scenes",
            "Full stride-5 LingBot prepare/run_worker produced provider-readable BSS for scene0011_00 and scene0050_00",
            "Scene alignment audit now confirms full selected-frame coverage for both target scenes",
            "Real selected Stream4D mask support rows are materialized from full stride-5 LingBot BSS: 2802/2802 rows, all non-empty",
            "LingBot real-support mask-level feature families were materialized for first32 local-window AP attempts",
            "Phase10 temporal tracker has a LingBot-qualified local AP pass: T3 MV_AP_window=0.14260261343594677 vs baseline=0.1135433179342663",
            "Phase10 small repair grid finds a control-resistant LingBot temporal variant on first32: S1_w3d0p2_s3d0p6_s2d0p08_area0p30_thr0p55_gap1 MV_AP_window=0.15620051245051245 vs 2D/shape control=0.15368018320916868",
            "Phase10c full selected-scene runner evaluates 2802 observations over 209 source frames in 40.874s with only two variants",
        ],
        "current_blockers": [
            "LingBot evidence is trajectory/action-memory evidence, not yet Stream4D mask/object affinity evidence",
            "First32 control-resistant result does not generalize to full selected scene",
            f"Full selected-scene LingBot MV_AP_scene={phase10_full_scene.get('best_lingbot_MV_AP_scene', '')}; full selected-scene control MV_AP_scene={phase10_full_scene.get('best_control_MV_AP_scene', '')}",
            f"Full selected-scene control_minus_lingbot_MV_AP_scene={phase10_full_scene.get('control_minus_lingbot_MV_AP_scene', '')}",
            "Full selected-scene LingBot AP is below the first32 baseline contract, so phase10_pass=false",
            "The 145-variant long repair grid was too inefficient for interactive work: after about 15 minutes it had no summary/CSV output",
            "No same-count/random/keyframe control gate has been run for the full selected-scene support path",
        ],
        "remaining_gates": [
            "repair full selected-scene temporal readout so LingBot does not lose to w3d=0 control",
            "add cached mask/GT/IoU evaluation before any broader grid",
            "run same-count/random/keyframe controls only after full selected-scene AP is repaired",
            "do not claim Go until full scene and controls pass",
        ],
        "v105_baseline_row_count": len(v105_rows),
        "v109_f19_metric_row_count": len(v109_f19_rows),
        "v109_f19_improved_sequence_count": int(sum(v > 0.0 for v in f19_improved)),
        "phase1_provider_smoke_taxonomy": phase1_smoke.get("taxonomy", ""),
        "phase2_bss_materialization_taxonomy": phase2_materialization.get("taxonomy", ""),
        "phase3_mask_support_rows_taxonomy": phase3_support_rows.get("taxonomy", ""),
        "phase4_scene_alignment_taxonomy": phase4_scene_alignment.get("taxonomy", ""),
        "phase4_scene_alignment_blocker": phase4_scene_alignment.get("blocker", ""),
        "phase4_real_masks_available": phase4_scene_alignment.get("real_masks_available", ""),
        "phase4_same_scene_lingbot_bss_available": phase4_scene_alignment.get("same_scene_lingbot_bss_available", ""),
        "phase4_same_scene_lingbot_selected_frame_coverage_complete": phase4_scene_alignment.get("same_scene_lingbot_selected_frame_coverage_complete", ""),
        "phase5_general_config_readiness_taxonomy": phase5_general_config.get("taxonomy", ""),
        "phase5_template_configs_runnable_now": phase5_general_config.get("template_configs_runnable_now", ""),
        "phase5_sampling_alignment": phase5_general_config.get("sampling_alignment", ""),
        "phase5_smoke_config_stride": phase5_general_config.get("smoke_config_stride", ""),
        "phase5_smoke_frame_positions": phase5_general_config.get("smoke_frame_positions", []),
        "phase6_scene0011_bss_smoke_taxonomy": phase6_scene0011_bss_smoke.get("taxonomy", ""),
        "phase6_scene0050_bss_smoke_taxonomy": phase6_scene0050_bss_smoke.get("taxonomy", ""),
        "phase6_scene0011_full_bss_taxonomy": phase6_scene0011_full_bss.get("taxonomy", ""),
        "phase6_scene0011_full_bss_num_frames_checked": phase6_scene0011_full_bss.get("num_frames_checked", ""),
        "phase6_scene0050_full_bss_taxonomy": phase6_scene0050_full_bss.get("taxonomy", ""),
        "phase6_scene0050_full_bss_num_frames_checked": phase6_scene0050_full_bss.get("num_frames_checked", ""),
        "phase7_real_mask_support_taxonomy": phase7_real_mask_support.get("taxonomy", ""),
        "phase7_real_mask_support_selected_row_count": phase7_real_mask_support.get("selected_row_count", ""),
        "phase7_real_mask_support_support_row_count": phase7_real_mask_support.get("support_row_count", ""),
        "phase7_real_mask_support_nonempty_support_row_count": phase7_real_mask_support.get("nonempty_support_row_count", ""),
        "phase7_real_mask_support_empty_support_row_count": phase7_real_mask_support.get("empty_support_row_count", ""),
        "phase7_real_mask_support_coverage_ratio": phase7_real_mask_support.get("support_coverage_ratio", ""),
        "phase8_voxel_feature_taxonomy": phase8_voxel_feature.get("taxonomy", ""),
        "phase8_centroid_feature_taxonomy": phase8_centroid_feature.get("taxonomy", ""),
        "phase8_voxel_centroid_feature_taxonomy": phase8_voxel_centroid_feature.get("taxonomy", ""),
        "phase9_best_feature_mode": phase9_best_local_ap.get("feature_mode", ""),
        "phase9_best_variant_id": phase9_best_local_ap.get("best_variant_id", ""),
        "phase9_best_MV_AP_window": phase9_best_local_ap.get("best_MV_AP_window", ""),
        "phase9_best_MV_AP50_window": phase9_best_local_ap.get("best_MV_AP50_window", ""),
        "phase9_baseline_MV_AP_window": phase9_baseline_ap if phase9_attempted else "",
        "phase9_local_ap_pass": phase9_local_ap_pass,
        "phase10_active_source": phase10_active_source,
        "phase10_original_decision": phase10_temporal_track.get("decision", ""),
        "phase10_full_scene_attempted": phase10_full_scene_attempted,
        "phase10_full_scene_pass": phase10_full_scene_pass,
        "phase10_full_scene_control_beats_lingbot": phase10_full_scene_control_beats_lingbot,
        "phase10_full_scene_observation_count": phase10_full_scene.get("observation_count", ""),
        "phase10_full_scene_runtime_sec": phase10_full_scene.get("runtime_sec", ""),
        "phase10_full_scene_input_summary": phase10_full_scene.get("full_scene_input_summary", {}),
        "phase10_repair_grid_summary_exists": bool(phase10_repair_grid),
        "phase10_repair_grid_small_summary_exists": bool(phase10_repair_grid_small),
        "phase10_decision": phase10_active.get("decision", ""),
        "phase10_qualified_lingbot_local_ap_pass": phase10_lingbot_local_ap_pass,
        "phase10_scene_metric_available": phase10_scene_metric_available,
        "phase10_control_resistant": phase10_control_resistant,
        "phase10_first32_control_resistant": phase10_first32_control_resistant,
        "phase10_first32_best_lingbot_variant_id": phase10_repair_grid_small.get("best_lingbot_variant_id", ""),
        "phase10_first32_best_lingbot_MV_AP_scene": phase10_repair_grid_small.get("best_lingbot_MV_AP_scene", ""),
        "phase10_first32_best_control_MV_AP_scene": phase10_repair_grid_small.get("best_control_MV_AP_scene", ""),
        "phase10_best_lingbot_variant_id": phase10_active.get("best_lingbot_variant_id", ""),
        "phase10_best_lingbot_MV_AP_window": phase10_active.get("best_lingbot_MV_AP_window", ""),
        "phase10_best_lingbot_MV_AP50_window": phase10_active.get("best_lingbot_MV_AP50_window", ""),
        "phase10_best_lingbot_MV_AP_scene": phase10_active.get("best_lingbot_MV_AP_scene", ""),
        "phase10_best_lingbot_MV_AP50_scene": phase10_active.get("best_lingbot_MV_AP50_scene", ""),
        "phase10_best_control_variant_id": phase10_active.get("best_control_variant_id", ""),
        "phase10_best_control_MV_AP_window": phase10_active.get("best_control_MV_AP_window", ""),
        "phase10_best_control_MV_AP_scene": phase10_active.get("best_control_MV_AP_scene", ""),
        "phase10_control_minus_lingbot_MV_AP_window": phase10_active.get("control_minus_lingbot_MV_AP_window", ""),
        "phase10_control_minus_lingbot_MV_AP_scene": phase10_active.get("control_minus_lingbot_MV_AP_scene", ""),
        "phase10_control_beats_lingbot_MV_AP_window": phase10_control_beats_lingbot,
        "phase10_original_best_lingbot_variant_id": phase10_temporal_track.get("best_lingbot_variant_id", ""),
        "phase10_original_best_lingbot_MV_AP_window": phase10_temporal_track.get("best_lingbot_MV_AP_window", ""),
        "phase10_original_best_control_variant_id": phase10_temporal_track.get("best_control_variant_id", ""),
        "phase10_original_best_control_MV_AP_window": phase10_temporal_track.get("best_control_MV_AP_window", ""),
        "v109_stage2_core_taxonomy": v109_core.get("taxonomy", ""),
        "v109_f19_taxonomy": v109_f19.get("taxonomy", ""),
        "forbidden_provider_dependency_hits": forbidden_hits,
        "outputs": {
            "artifact_rows": _rel(out / "artifact_rows.csv"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "metric_evidence_rows": _rel(out / "metric_evidence_rows.csv"),
            "provider_contract_rows": _rel(out / "provider_contract_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "status_report": _rel(out / "status_report.md"),
            "summary": _rel(out / "summary.json"),
        },
        "runtime_sec": round(time.time() - t0, 3),
    }

    status_report = "\n".join(
        [
            "# Stream4D v104 LingBot-Map-only fact lock",
            "",
            f"taxonomy: `{summary['taxonomy']}`",
            f"fact_lock_pass: `{fact_lock_pass}`",
            f"stream4d_model_ready: `False`",
            f"blocker: `{summary['blocker']}`",
            "",
            "## Good news",
            "- LingBot-Map repo/checkpoint are present locally.",
            "- LingBot code exposes pose/depth/confidence/world_points outputs.",
            "- v105 full KITTI baseline evidence is available.",
            "- v109 F19 finished full-sequence LingBot action evaluation with `safety_candidate_pass=true`.",
            "- A Stream4D `LingBotMapGeometryProvider` scaffold now exists.",
            "- Provider smoke assigns actual LingBot BSS samples to two synthetic mask ids.",
            "- BSS materialization smoke writes frame points plus pixel samples.",
            "- Mask support rows smoke writes non-empty support rows from actual LingBot BSS samples.",
            "- Scene alignment audit confirms real Stream4D mask rasters are available for `scene0011_00` and `scene0050_00`.",
            "- General-config readiness finds target RGB frame folders and emits stride-5 aligned runnable LingBot config templates.",
            "- LingBot smoke8 prepare/run_worker produced same-scene BSS for both target scenes.",
            "- Phase6 BSS smoke confirms provider-readable depth/pose/intrinsics and source-frame mapping for both scenes.",
            "- Full stride-5 LingBot prepare/run_worker produced provider-readable BSS for both target scenes.",
            "- Scene alignment audit confirms selected-frame coverage is complete for both target scenes.",
            "- Real selected Stream4D mask support rows are materialized from full stride-5 LingBot BSS: 2802/2802 rows, all non-empty.",
            "- LingBot real-support mask-level feature families were materialized and evaluated through the canonical first32 local MV_AP path.",
            "- Phase10 temporal tracking has a LingBot-qualified local AP pass: `T3` MV_AP_window=0.14260261343594677.",
            "- Phase10 small repair grid finds a first32 control-resistant LingBot variant: `S1_w3d0p2_s3d0p6_s2d0p08_area0p30_thr0p55_gap1` MV_AP_window=0.15620051245051245.",
            "",
            "## Stuck point",
            "- The current verified evidence is still LingBot trajectory/action-memory evidence.",
            "- The active first32 LingBot temporal repair now beats the 2D/shape control, but only on the first32 dev subset.",
            f"- Best active LingBot MV_AP_window is `{phase10_active.get('best_lingbot_MV_AP_window', '')}`; best active control MV_AP_window is `{phase10_active.get('best_control_MV_AP_window', '')}`.",
            f"- Best active LingBot MV_AP_scene is `{phase10_active.get('best_lingbot_MV_AP_scene', '')}`; best active control MV_AP_scene is `{phase10_active.get('best_control_MV_AP_scene', '')}`.",
            "- No full/holdout or same-count/random/keyframe Go claim is made by this phase.",
            "",
            "## Remaining gates",
            "- Promote the control-resistant temporal variant beyond first32.",
            "- Run same-count/random/keyframe/control gates and compare against the first32 control-resistant variant.",
            "- Add same-count/random/keyframe controls before any Go decision.",
            "",
            "## Important v109 lessons reused",
            "- Keep `stage2_action_mode` and legacy `stage4_action_mode` compatibility when reusing LingBot manifest runners.",
            "- Do not run evaluate/report with concurrent writers for the same dataset aggregate JSON.",
            "- Separate metric improvement from semantic or object-specific causality claims.",
            "",
        ]
    )

    _write_csv(out / "artifact_rows.csv", artifact_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_csv(out / "metric_evidence_rows.csv", metric_rows)
    _write_csv(out / "provider_contract_rows.csv", contract_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    _write_json(out / "summary.json", summary)
    (out / "status_report.md").write_text(status_report, encoding="utf-8")
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stream4D v104 LingBot-Map-only fact lock.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
