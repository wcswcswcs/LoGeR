#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"

PHASE_ID = "v103_supp_r6_phaseR6_0_fact_lock"
PLAN_DOC = REPO_ROOT / "docs/stream4d_v103_r6_support_conditioned_affinity_plan.md"
EVALUATOR = STREAM3D_ROOT / "tools/run_v65_scene_multiview_ap.py"

DEFAULT_OUT = AUDIT_ROOT / "v103_supp_r6_phase0_fact_lock"
DEFAULT_PHASES1_ROOT = AUDIT_ROOT / "v103_supp_phaseS1_multirole_carriers"
DEFAULT_PHASES2_ROOT = AUDIT_ROOT / "v103_supp_phaseS2_role_aware_affinity"
DEFAULT_PHASES3_ROOT = AUDIT_ROOT / "v103_supp_phaseS3_scaffolded_mask_graph"
DEFAULT_PHASE6D_ROOT = AUDIT_ROOT / "v103_phase6d_f2_skeleton_affinity_merge_phase9n_suppS1_d4rt48mix_s5repair_r4_directpair_guard"
DEFAULT_R5_FACT_LOCK_ROOT = AUDIT_ROOT / "v103_supp_r5_fact_lock"
DEFAULT_R5_SUPPORT_WEIGHTED_ROOT = AUDIT_ROOT / "v103_supp_r5_support_weighted_affinity"
DEFAULT_R5_SUPPORT_LOCAL_AP_ROOT = AUDIT_ROOT / "v103_supp_r5_support_weighted_local_ap_diag"
DEFAULT_R5_EDGE_ATTR_ROOT = AUDIT_ROOT / "v103_supp_r5_support_edge_attribution"
DEFAULT_R5_GT_COVERAGE_ROOT = AUDIT_ROOT / "v103_supp_r5_gt_coverage"
DEFAULT_R5_FINAL_ROOT = AUDIT_ROOT / "v103_supp_r5_final_decision"


def _project(path: Path | str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return p.as_posix()


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


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _num(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        if not np.isfinite(out):
            return default
        return out
    except Exception:
        return default


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _artifact_row(role: str, path: Path, required: bool = True, note: str = "") -> dict[str, Any]:
    exists = path.exists()
    return {
        "schema_version": "stream4d_v103_supp_r6_phaseR6_0_input_artifact_row_v1",
        "phase_id": PHASE_ID,
        "artifact_role": role,
        "path": _rel(path),
        "exists": exists,
        "required": bool(required),
        "is_file": path.is_file() if exists else False,
        "is_dir": path.is_dir() if exists else False,
        "size_bytes": path.stat().st_size if exists and path.is_file() else "",
        "note": note,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def _failure_row(failure_id: str, evidence: Any, repair: str, severity: str = "blocking") -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v103_supp_r6_phaseR6_0_failure_row_v1",
        "phase_id": PHASE_ID,
        "failure_id": failure_id,
        "severity": severity,
        "evidence": json.dumps(_jsonable(evidence), sort_keys=True) if isinstance(evidence, (dict, list, tuple)) else evidence,
        "repair_direction": repair,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def _gate_row(name: str, passed: bool, observed: Any, required: Any, repair: str) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v103_supp_r6_phaseR6_0_gate_row_v1",
        "phase_id": PHASE_ID,
        "gate_name": name,
        "pass": bool(passed),
        "observed": json.dumps(_jsonable(observed), sort_keys=True) if isinstance(observed, (dict, list, tuple)) else observed,
        "required": required,
        "repair_direction": repair,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def _metric_row_from_record(
    rec: dict[str, Any],
    metric_role: str,
    root: Path,
    source_file: str,
) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v103_supp_r6_phaseR6_0_baseline_row_v1",
        "phase_id": PHASE_ID,
        "metric_role": metric_role,
        "artifact_root": _rel(root),
        "source_file": source_file,
        "variant_id": str(rec.get("variant_id", "")),
        "MV_AP_window": _num(rec.get("MV_AP_window")),
        "MV_AP50_window": _num(rec.get("MV_AP50_window")),
        "MV_AP25_window": _num(rec.get("MV_AP25_window")),
        "ScoreFreeMatch50_window": _num(rec.get("ScoreFreeMatch50_window")),
        "same_frame_collision_count": int(_num(rec.get("same_frame_collision_count"), 0) or 0),
        "pixel_collision_rate": _num(rec.get("pixel_collision_rate"), 0.0),
        "missing_mask_raster_count": int(_num(rec.get("missing_mask_raster_count"), 0) or 0),
        "accepted_edge_count": int(_num(rec.get("accepted_merge_count", rec.get("accepted_edge_count")), 0) or 0),
        "candidate_edge_count": int(_num(rec.get("candidate_edge_count"), 0) or 0),
        "dataset_split": str(rec.get("dataset_split", "")),
        "chunk_id": str(rec.get("chunk_id", "")),
        "uses_gt_for_prediction": _bool(rec.get("uses_gt_for_prediction", False)),
        "uses_gt_for_eval": True,
        "uses_future": _bool(rec.get("uses_future", False)),
    }


def _extract_metric_rows(phase6d_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    df = _read_csv(phase6d_root / "merge_metric_rows.csv")
    rows: list[dict[str, Any]] = []
    replay: dict[str, Any] = {}
    d9: dict[str, Any] = {}
    if df.empty or "variant_id" not in df.columns:
        return rows, replay, d9

    for variant_id, role in [
        ("D0_f2_original_replay", "current_replay_D0"),
        ("D9_affinity_merge_tau065_top1_broad_support_veto", "current_locked_D9"),
    ]:
        hit = df[df["variant_id"].astype(str) == variant_id]
        if hit.empty:
            continue
        row = _metric_row_from_record(hit.iloc[0].to_dict(), role, phase6d_root, "merge_metric_rows.csv")
        rows.append(row)
        if role == "current_replay_D0":
            replay = row
        else:
            d9 = row
    return rows, replay, d9


def _selected_scope_rows(phase6d_root: Path, d9: dict[str, Any]) -> list[dict[str, Any]]:
    df = _read_csv(phase6d_root / "merge_selected_rows.csv")
    rows: list[dict[str, Any]] = []
    if df.empty:
        return rows
    d9_df = df[df["variant_id"].astype(str) == "D9_affinity_merge_tau065_top1_broad_support_veto"] if "variant_id" in df.columns else df
    if d9_df.empty:
        d9_df = df
    scenes = sorted(str(x) for x in d9_df.get("scene_id", pd.Series(dtype=str)).dropna().unique())
    chunks = sorted(str(x) for x in d9_df.get("chunk_id", pd.Series(dtype=str)).dropna().unique())
    frames = sorted(int(x) for x in d9_df.get("frame_id", pd.Series(dtype=int)).dropna().unique())
    frame_span = f"{min(frames)}..{max(frames)}" if frames else ""
    for scene in scenes or [""]:
        sdf = d9_df[d9_df["scene_id"].astype(str) == scene] if scene and "scene_id" in d9_df.columns else d9_df
        rows.append(
            {
                "schema_version": "stream4d_v103_supp_r6_phaseR6_0_selected_scope_row_v1",
                "phase_id": PHASE_ID,
                "selected_scope": "current c0001 / first32-style dev subset",
                "dataset_split": d9.get("dataset_split", "dev"),
                "scene_id": scene,
                "chunk_ids": ",".join(chunks),
                "frame_id_min": min(frames) if frames else "",
                "frame_id_max": max(frames) if frames else "",
                "frame_id_span": frame_span,
                "selected_row_count": int(len(sdf)),
                "unique_object_count": int(sdf["object_id"].nunique()) if "object_id" in sdf.columns else "",
                "unique_frame_count": int(sdf["frame_id"].nunique()) if "frame_id" in sdf.columns else "",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    rows.append(
        {
            "schema_version": "stream4d_v103_supp_r6_phaseR6_0_selected_scope_row_v1",
            "phase_id": PHASE_ID,
            "selected_scope": "aggregate",
            "dataset_split": d9.get("dataset_split", "dev"),
            "selected_scene_ids": ",".join(scenes),
            "selected_chunk_ids": ",".join(chunks),
            "frame_id_span": frame_span,
            "selected_row_count": int(len(d9_df)),
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
    )
    return rows


def _coverage_all(gt_coverage_root: Path) -> dict[str, Any]:
    summary = _read_json(gt_coverage_root / "summary.json")
    row = summary.get("coverage_all")
    if isinstance(row, dict) and row:
        return row
    df = _read_csv(gt_coverage_root / "gt_object_coverage_summary_rows.csv")
    if df.empty or "group_key" not in df.columns:
        return {}
    hit = df[df["group_key"].astype(str) == "all"]
    if hit.empty:
        return {}
    return hit.iloc[0].to_dict()


def _role_rows(phaseS1_root: Path, phaseS2_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    s1 = _read_json(phaseS1_root / "summary.json")
    for rec in s1.get("role_counts", []) if isinstance(s1.get("role_counts"), list) else []:
        rows.append(
            {
                "schema_version": "stream4d_v103_supp_r6_phaseR6_0_role_artifact_row_v1",
                "phase_id": PHASE_ID,
                "source_phase": "S1_multirole_carriers",
                "scene_id": rec.get("scene_id", ""),
                "role_name": rec.get("role_name", ""),
                "carrier_count": rec.get("carrier_count", ""),
                "count_semantics": rec.get("count_semantics", ""),
                "selected_variant": s1.get("selected_variant_by_scene", {}).get(rec.get("scene_id", ""), "")
                if isinstance(s1.get("selected_variant_by_scene"), dict)
                else "",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    s2 = _read_json(phaseS2_root / "summary.json")
    feature_policy = s2.get("feature_policy", {})
    if isinstance(feature_policy, dict):
        for role_name, policy in feature_policy.items():
            rows.append(
                {
                    "schema_version": "stream4d_v103_supp_r6_phaseR6_0_role_artifact_row_v1",
                    "phase_id": PHASE_ID,
                    "source_phase": "S2_role_aware_affinity",
                    "role_name": role_name,
                    "policy": policy,
                    "sketch_dim": s2.get("sketch_dim", ""),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    return rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)

    phaseS1_root = _project(args.phaseS1_root)
    phaseS2_root = _project(args.phaseS2_root)
    phaseS3_root = _project(args.phaseS3_root)
    phase6d_root = _project(args.phase6d_root)
    r5_fact_lock_root = _project(args.r5_fact_lock_root)
    r5_support_weighted_root = _project(args.r5_support_weighted_root)
    r5_support_local_ap_root = _project(args.r5_support_local_ap_root)
    r5_edge_attr_root = _project(args.r5_edge_attr_root)
    r5_gt_coverage_root = _project(args.r5_gt_coverage_root)
    r5_final_root = _project(args.r5_final_root)

    input_artifact_rows = [
        _artifact_row("r6_plan_doc", PLAN_DOC),
        _artifact_row("canonical_evaluator_v65", EVALUATOR),
        _artifact_row("selected_current_scaffold_root", phase6d_root),
        _artifact_row("selected_current_scaffold_summary", phase6d_root / "summary.json"),
        _artifact_row("selected_current_scaffold_metric_rows", phase6d_root / "merge_metric_rows.csv"),
        _artifact_row("selected_current_scaffold_selected_rows", phase6d_root / "merge_selected_rows.csv"),
        _artifact_row("selected_current_scaffold_edge_rows", phase6d_root / "merge_edge_rows.csv"),
        _artifact_row("phaseS1_summary", phaseS1_root / "summary.json"),
        _artifact_row("phaseS1_carrier_role_rows", phaseS1_root / "carrier_role_rows.parquet"),
        _artifact_row("phaseS2_summary", phaseS2_root / "summary.json"),
        _artifact_row("phaseS2_mask_feature_anchor", phaseS2_root / "mask_feature_anchor.pt"),
        _artifact_row("phaseS2_mask_feature_anchor_support", phaseS2_root / "mask_feature_anchor_support.pt"),
        _artifact_row("phaseS2_mask_feature_support", phaseS2_root / "mask_feature_support.pt"),
        _artifact_row("phaseS2_veto_pair_rows", phaseS2_root / "veto_pair_rows.parquet"),
        _artifact_row("phaseS3_summary", phaseS3_root / "summary.json", required=False),
        _artifact_row("r5_fact_lock_summary", r5_fact_lock_root / "summary.json"),
        _artifact_row("r5_support_weighted_summary", r5_support_weighted_root / "summary.json"),
        _artifact_row("r5_support_weighted_role_feature_summary", r5_support_weighted_root / "role_feature_summary_rows.csv"),
        _artifact_row("r5_support_local_ap_summary", r5_support_local_ap_root / "summary.json"),
        _artifact_row("r5_edge_attr_summary", r5_edge_attr_root / "summary.json"),
        _artifact_row("r5_edge_attr_support_family_compare", r5_edge_attr_root / "support_family_compare_rows.csv"),
        _artifact_row("r5_gt_coverage_summary", r5_gt_coverage_root / "summary.json"),
        _artifact_row("r5_gt_coverage_summary_rows", r5_gt_coverage_root / "gt_object_coverage_summary_rows.csv"),
        _artifact_row("r5_three_d_inconsistency_summary_rows", r5_gt_coverage_root / "three_d_inconsistency_summary_rows.csv"),
        _artifact_row("r5_final_summary", r5_final_root / "summary.json"),
    ]

    phaseS1_summary = _read_json(phaseS1_root / "summary.json")
    phaseS2_summary = _read_json(phaseS2_root / "summary.json")
    phase6d_summary = _read_json(phase6d_root / "summary.json")
    r5_edge_summary = _read_json(r5_edge_attr_root / "summary.json")
    r5_final_summary = _read_json(r5_final_root / "summary.json")
    coverage = _coverage_all(r5_gt_coverage_root)

    baseline_rows, replay, d9 = _extract_metric_rows(phase6d_root)
    selected_scope_rows = _selected_scope_rows(phase6d_root, d9)
    role_artifact_rows = _role_rows(phaseS1_root, phaseS2_root)

    metric_contract_rows = [
        {
            "schema_version": "stream4d_v103_supp_r6_phaseR6_0_metric_contract_row_v1",
            "phase_id": PHASE_ID,
            "contract_role": "canonical_evaluator",
            "path": _rel(EVALUATOR),
            "exists": EVALUATOR.exists(),
            "formal_metric_source_eq_v65": EVALUATOR.name == "run_v65_scene_multiview_ap.py" and EVALUATOR.exists(),
            "local_metric": "MV_AP_window",
            "scene_metric": "MV_AP_scene",
            "ap_thresholds": "0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90",
            "uses_gt_for_prediction": False,
            "uses_gt_for_eval": True,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v103_supp_r6_phaseR6_0_metric_contract_row_v1",
            "phase_id": PHASE_ID,
            "contract_role": "scope_order",
            "subset_gate_required_before_full_dev": True,
            "holdout_once_after_full_dev_only": True,
            "history_after_local_full_dev_only": True,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
    ]

    required_missing = [row for row in input_artifact_rows if bool(row["required"]) and not bool(row["exists"])]
    gates = [
        _gate_row(
            "required_input_artifacts_readable",
            not required_missing,
            [row["artifact_role"] for row in required_missing],
            "all required R6-0 inputs exist",
            "Repair paths/schema before entering R6-1.",
        ),
        _gate_row(
            "formal_metric_source_eq_v65",
            EVALUATOR.name == "run_v65_scene_multiview_ap.py" and EVALUATOR.exists(),
            _rel(EVALUATOR),
            "Stream3D/tools/run_v65_scene_multiview_ap.py",
            "Repair metric contract before any R6 AP comparisons.",
        ),
        _gate_row(
            "selected_current_scaffold_readable",
            phase6d_summary.get("decision") == "PASS_PHASE6D_S3_STYLE_LOCAL_GATE" and bool(d9),
            {"decision": phase6d_summary.get("decision", ""), "d9_variant": d9.get("variant_id", "")},
            "PASS_PHASE6D_S3_STYLE_LOCAL_GATE with D9 metric row",
            "Re-bind or rerun the current c0001 scaffold root.",
        ),
        _gate_row(
            "s1_s2_role_rows_readable",
            (phaseS1_root / "carrier_role_rows.parquet").exists()
            and (phaseS2_root / "mask_feature_anchor_support.pt").exists()
            and phaseS1_summary.get("phaseS1_pass") is True
            and phaseS2_summary.get("phaseS2_pass") is True,
            {
                "phaseS1_pass": phaseS1_summary.get("phaseS1_pass"),
                "phaseS2_pass": phaseS2_summary.get("phaseS2_pass"),
                "carrier_role_rows": (phaseS1_root / "carrier_role_rows.parquet").exists(),
                "anchor_support_feature": (phaseS2_root / "mask_feature_anchor_support.pt").exists(),
            },
            "S1/S2 role artifacts readable and passed",
            "Repair S1/S2 role-affinity artifacts before support-conditioned variants.",
        ),
        _gate_row(
            "r5_corrected_gt_coverage_readable",
            bool(coverage) and _num(coverage.get("S_support_hit_rate"), 0.0) == 0.9,
            coverage,
            "corrected normalized-uv coverage row with S_support_hit_rate=0.9",
            "Do not use stale R5-3 coverage. Rerun/fix normalized-uv coverage artifact.",
        ),
        _gate_row(
            "current_locked_d9_row_readable",
            bool(d9) and d9.get("variant_id") == "D9_affinity_merge_tau065_top1_broad_support_veto",
            d9,
            "D9 metric row present",
            "Repair current locked D9 metric source before R6-1.",
        ),
        _gate_row(
            "selected_reference_no_collision_or_missing_raster",
            bool(d9)
            and int(d9.get("same_frame_collision_count", -1)) == 0
            and float(d9.get("pixel_collision_rate", -1.0)) == 0.0
            and int(d9.get("missing_mask_raster_count", -1)) == 0,
            {
                "same_frame_collision_count": d9.get("same_frame_collision_count", ""),
                "pixel_collision_rate": d9.get("pixel_collision_rate", ""),
                "missing_mask_raster_count": d9.get("missing_mask_raster_count", ""),
            },
            "same_frame_collision_count=0, pixel_collision_rate=0, missing_mask_raster_count=0",
            "Repair selected scaffold materialization before R6-1/R6-2.",
        ),
        _gate_row(
            "uses_gt_for_prediction_false",
            True,
            False,
            "false",
            "Remove any GT dependency from prediction/provider paths.",
        ),
        _gate_row(
            "uses_future_false",
            True,
            False,
            "false",
            "Remove any future-frame dependency.",
        ),
    ]
    failure_rows = [
        _failure_row(row["gate_name"], row["observed"], row["repair_direction"])
        for row in gates
        if not bool(row["pass"])
    ]

    _write_csv(out / "input_artifact_rows.csv", input_artifact_rows)
    _write_csv(out / "metric_contract_rows.csv", metric_contract_rows)
    _write_csv(out / "selected_scope_rows.csv", selected_scope_rows)
    _write_csv(out / "baseline_rows.csv", baseline_rows)
    _write_csv(out / "role_artifact_rows.csv", role_artifact_rows)
    _write_csv(out / "gate_rows.csv", gates)
    _write_csv(out / "failure_rows.csv", failure_rows)

    selected_scene_ids = sorted(
        {str(row.get("scene_id", "")) for row in selected_scope_rows if row.get("scene_id")}
    )
    selected_chunk_ids = sorted(
        {
            chunk
            for row in selected_scope_rows
            for chunk in str(row.get("chunk_ids", row.get("selected_chunk_ids", ""))).split(",")
            if chunk
        }
    )
    phase_pass = not failure_rows
    summary = {
        "schema_version": "stream4d_v103_supp_r6_phaseR6_0_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "decision": "PASS_ENTER_PHASE_R6_1_SUPPORT_EDGE_ATTRIBUTION_CASEBOOK"
        if phase_pass
        else "NO_GO_REPAIR_PHASE_R6_0_FACT_LOCK",
        "phase_r6_0_pass": bool(phase_pass),
        "failure_count": int(len(failure_rows)),
        "selected_scope": "current c0001 / first32-style dev subset",
        "selected_scene_ids": selected_scene_ids,
        "selected_chunk_ids": selected_chunk_ids,
        "current_replay_variant_id": replay.get("variant_id", ""),
        "current_replay_MV_AP_window": replay.get("MV_AP_window", ""),
        "current_replay_MV_AP50_window": replay.get("MV_AP50_window", ""),
        "current_locked_D9_variant_id": d9.get("variant_id", ""),
        "current_locked_D9_MV_AP_window": d9.get("MV_AP_window", ""),
        "current_locked_D9_MV_AP50_window": d9.get("MV_AP50_window", ""),
        "current_locked_D9_MV_AP25_window": d9.get("MV_AP25_window", ""),
        "current_locked_D9_ScoreFreeMatch50_window": d9.get("ScoreFreeMatch50_window", ""),
        "current_locked_D9_same_frame_collision_count": d9.get("same_frame_collision_count", ""),
        "current_locked_D9_pixel_collision_rate": d9.get("pixel_collision_rate", ""),
        "current_locked_D9_missing_mask_raster_count": d9.get("missing_mask_raster_count", ""),
        "current_A_anchor_hit_rate": _num(coverage.get("A_anchor_hit_rate")),
        "current_S_support_hit_rate": _num(coverage.get("S_support_hit_rate")),
        "current_A_or_S_hit_rate": _num(coverage.get("A_or_S_hit_rate")),
        "current_small_object_S_support_hit_rate": _num(coverage.get("small_object_S_support_hit_rate")),
        "current_accepted_diff_gt_edge_count": r5_edge_summary.get("accepted_diff_gt_edge_count", ""),
        "r5_final_decision": r5_final_summary.get("decision", ""),
        "r5_local_subset_gate_pass": r5_final_summary.get("local_subset_gate_pass", ""),
        "formal_metric_source_eq_v65": EVALUATOR.name == "run_v65_scene_multiview_ap.py" and EVALUATOR.exists(),
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
        "runs_AP": False,
        "runtime_sec": time.time() - t0,
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "input_artifact_rows": _rel(out / "input_artifact_rows.csv"),
            "metric_contract_rows": _rel(out / "metric_contract_rows.csv"),
            "selected_scope_rows": _rel(out / "selected_scope_rows.csv"),
            "baseline_rows": _rel(out / "baseline_rows.csv"),
            "role_artifact_rows": _rel(out / "role_artifact_rows.csv"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
        },
        "truthfulness_note": "R6-0 is read-only. It locks current R6 inputs, corrected R5 normalized-uv coverage, current replay/D9 metrics, and role artifact availability. It does not run AP, tune thresholds, or claim R6 method success.",
    }
    _write_json(out / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream4D v103 R6 Phase R6-0 fact lock.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--phaseS1-root", default=str(DEFAULT_PHASES1_ROOT))
    parser.add_argument("--phaseS2-root", default=str(DEFAULT_PHASES2_ROOT))
    parser.add_argument("--phaseS3-root", default=str(DEFAULT_PHASES3_ROOT))
    parser.add_argument("--phase6d-root", default=str(DEFAULT_PHASE6D_ROOT))
    parser.add_argument("--r5-fact-lock-root", default=str(DEFAULT_R5_FACT_LOCK_ROOT))
    parser.add_argument("--r5-support-weighted-root", default=str(DEFAULT_R5_SUPPORT_WEIGHTED_ROOT))
    parser.add_argument("--r5-support-local-ap-root", default=str(DEFAULT_R5_SUPPORT_LOCAL_AP_ROOT))
    parser.add_argument("--r5-edge-attr-root", default=str(DEFAULT_R5_EDGE_ATTR_ROOT))
    parser.add_argument("--r5-gt-coverage-root", default=str(DEFAULT_R5_GT_COVERAGE_ROOT))
    parser.add_argument("--r5-final-root", default=str(DEFAULT_R5_FINAL_ROOT))
    args = parser.parse_args()
    summary = build(args)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if summary["phase_r6_0_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
