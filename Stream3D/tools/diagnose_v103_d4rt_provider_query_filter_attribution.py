#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
PHASE_ID = "v103_d4rt_provider_query_filter_attribution"
DEFAULT_OUT = AUDIT_ROOT / "v103_supp_d4rt_provider_query_filter_attribution_r1"


PHASE2_PROVIDER_ROOTS = [
    {
        "provider_label": "d4rt32_q5c",
        "scene_id": "scene0011_00",
        "root": AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0011_c0001_cap24576_qchunk16384",
    },
    {
        "provider_label": "d4rt32_q5c",
        "scene_id": "scene0050_00",
        "root": AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0050_c0001_cap24576_qchunk16384",
    },
    {
        "provider_label": "d4rt48mix_q5c",
        "scene_id": "scene0011_00",
        "root": AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0011_c0001_cap24576_qchunk16384_d4rt48mix",
    },
    {
        "provider_label": "d4rt48mix_q5c_maskbalanced8",
        "scene_id": "scene0011_00",
        "root": AUDIT_ROOT
        / "v103_phase2_stratified_q5c_objlike16384_scene0011_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8",
    },
    {
        "provider_label": "d4rt48mix_q5c_maskbalanced8",
        "scene_id": "scene0050_00",
        "root": AUDIT_ROOT
        / "v103_phase2_stratified_q5c_objlike16384_scene0050_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8",
    },
]


PHASE3_FILTER_ROOTS = [
    {
        "filter_label": "d4rt32_q5c_competing_repair5",
        "root": AUDIT_ROOT / "v103_phase3_carrier_reliability_filter_q5c_objlike16384_competing_repair5",
    },
    {
        "filter_label": "d4rt48mix_q5c_scene0011_competing_repair5",
        "root": AUDIT_ROOT / "v103_phase3_carrier_reliability_filter_scene0011_c0001_d4rt48mix_competing_repair5",
    },
    {
        "filter_label": "d4rt48mix_q5c_scene0011_semantic_veto_repair6",
        "root": AUDIT_ROOT / "v103_phase3_carrier_reliability_filter_scene0011_c0001_d4rt48mix_semantic_veto_repair6",
    },
    {
        "filter_label": "d4rt48mix_maskbalanced8_competing_repair5",
        "root": AUDIT_ROOT / "v103_phase3_carrier_reliability_filter_all_d4rt48mix_maskbalanced8_competing_repair5",
    },
    {
        "filter_label": "d4rt48mix_maskbalanced8_scene0011_semantic_veto_repair6",
        "root": AUDIT_ROOT / "v103_phase3_carrier_reliability_filter_scene0011_c0001_d4rt48mix_maskbalanced8_semantic_veto_repair6",
    },
    {
        "filter_label": "d4rt48mix_maskbalanced8_scene0050_semantic_veto_repair6",
        "root": AUDIT_ROOT / "v103_phase3_carrier_reliability_filter_scene0050_c0001_d4rt48mix_maskbalanced8_semantic_veto_repair6",
    },
]


STAGE_SUMMARY_ROOTS = [
    ("phaseS0_fact_lock", AUDIT_ROOT / "v103_supp_phaseS0_fact_lock"),
    ("phaseS1_multirole_carriers", AUDIT_ROOT / "v103_supp_phaseS1_multirole_carriers"),
    ("phaseS2_role_aware_affinity", AUDIT_ROOT / "v103_supp_phaseS2_role_aware_affinity"),
    ("phaseS3_scaffolded_mask_graph", AUDIT_ROOT / "v103_supp_phaseS3_scaffolded_mask_graph"),
    ("phaseS4_post_birth_history_inheritance", AUDIT_ROOT / "v103_supp_phaseS4_post_birth_history_inheritance"),
    ("phaseS5_trigger_audit", AUDIT_ROOT / "v103_supp_phaseS5_trigger_audit"),
    ("phaseS5_da3_subchunk_provider_repair", AUDIT_ROOT / "v103_supp_phaseS5_da3_subchunk_provider_repair_summary_r1"),
    ("phaseS5_da3_preregistered_barrier", AUDIT_ROOT / "v103_supp_phaseS5_da3_preregistered_barrier_r1"),
    ("phase3_filter_decision_audit", AUDIT_ROOT / "v103_phase3_carrier_filter_decision_audit_r1"),
    ("phase4_affinity_correctness", AUDIT_ROOT / "v103_phase4_affinity_correctness_d4rt48mix_maskbalanced8_e5_r1"),
]


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _rel(path: str | Path) -> str:
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
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
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


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except Exception:
        return default


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _provider_name_from_summary(summary: dict[str, Any]) -> str:
    diag = summary.get("d4rt_infer_diagnostics", {})
    clip_frames = diag.get("clip_frames", "")
    size = diag.get("checkpoint_size_bytes", "")
    if str(clip_frames) == "48" or str(size) == "13950737434":
        return "OpenD4RT_48CLIP_9Mix_NoCropAUG"
    if str(clip_frames) == "32" or str(size) == "13950006682":
        return "OpenD4RT_32CLIP_9Dataset_NoAUG"
    return "unknown"


def _phase2_provider_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in PHASE2_PROVIDER_ROOTS:
        root = _project(spec["root"])
        summary_path = root / "summary.json"
        summary = _read_json(summary_path)
        if not summary:
            rows.append(
                {
                    "schema_version": "stream4d_v103_d4rt_provider_query_row_v1",
                    "phase_id": PHASE_ID,
                    "provider_label": spec["provider_label"],
                    "scene_id": spec["scene_id"],
                    "root": _rel(root),
                    "status": "missing_summary",
                }
            )
            continue
        diag = summary.get("d4rt_infer_diagnostics", {})
        cache = summary.get("carrier_batch_cache", {})
        policy = summary.get("query_generation_policy", {})
        metrics = summary.get("metrics", {})
        rows.append(
            {
                "schema_version": "stream4d_v103_d4rt_provider_query_row_v1",
                "phase_id": PHASE_ID,
                "provider_label": spec["provider_label"],
                "scene_id": spec["scene_id"],
                "provider_name": _provider_name_from_summary(summary),
                "root": _rel(root),
                "summary_path": _rel(summary_path),
                "frame_offset": summary.get("frame_offset"),
                "frame_count": summary.get("frame_count"),
                "query_count_per_frame": summary.get("query_count_per_frame"),
                "source_count": summary.get("source_count"),
                "max_query_count_per_frame_cap": summary.get("max_query_count_per_frame_cap"),
                "min_query_count_per_frame_required": summary.get("min_query_count_per_frame_required"),
                "fresh_d4rt_decode": cache.get("fresh_d4rt_decode", False),
                "carrier_batch_cache_used": cache.get("carrier_batch_cache_used", False),
                "source_parity_pass": cache.get("source_parity_pass", ""),
                "checkpoint_size_bytes": diag.get("checkpoint_size_bytes"),
                "clip_frames": diag.get("clip_frames"),
                "num_carriers": diag.get("num_carriers"),
                "seconds_d4rt_decode": diag.get("seconds_d4rt_decode"),
                "query_chunk_size": diag.get("query_chunk_size"),
                "mask_balanced_view_probe_points_per_mask": policy.get("mask_balanced_view_probe_points_per_mask", 0),
                "mask_balanced_reserved_under_cap": policy.get("mask_balanced_view_probe_reserved_under_cap", False),
                "carrier_track_in_image_rate_p10": metrics.get("carrier_track_in_image_rate_p10"),
                "carrier_track_in_image_rate_p50": metrics.get("carrier_track_in_image_rate_p50"),
                "carrier_track_in_image_rate_p90": metrics.get("carrier_track_in_image_rate_p90"),
                "uses_gt_for_query_selection": summary.get("uses_gt_for_query_selection", ""),
                "all_required_query_strata_present": summary.get("all_required_query_strata_present", ""),
            }
        )
    return rows


def _selected_rows_for_filter(root: Path, label: str) -> list[dict[str, Any]]:
    summary = _read_json(root / "summary.json")
    if not summary:
        return [
            {
                "schema_version": "stream4d_v103_d4rt_filter_attribution_row_v1",
                "phase_id": PHASE_ID,
                "filter_label": label,
                "root": _rel(root),
                "status": "missing_summary",
            }
        ]
    metric_path = root / "carrier_filter_metric_rows.csv"
    if not metric_path.exists():
        return [
            {
                "schema_version": "stream4d_v103_d4rt_filter_attribution_row_v1",
                "phase_id": PHASE_ID,
                "filter_label": label,
                "root": _rel(root),
                "decision": summary.get("decision"),
                "phase3_pass": summary.get("phase3_pass"),
                "status": "missing_metric_rows",
            }
        ]
    df = pd.read_csv(metric_path)
    selected_by_scene = {str(k): str(v) for k, v in dict(summary.get("selected_variant_by_scene", {})).items()}
    if not selected_by_scene:
        for scene in sorted(df["scene_id"].astype(str).unique().tolist()):
            scene_df = df[df["scene_id"].astype(str) == scene]
            if not scene_df.empty:
                selected_by_scene[scene] = str(scene_df.iloc[0]["variant_id"])
    rows: list[dict[str, Any]] = []
    for scene, variant in selected_by_scene.items():
        sub = df[(df["scene_id"].astype(str) == scene) & (df["variant_id"].astype(str) == variant)]
        if sub.empty:
            rows.append(
                {
                    "schema_version": "stream4d_v103_d4rt_filter_attribution_row_v1",
                    "phase_id": PHASE_ID,
                    "filter_label": label,
                    "root": _rel(root),
                    "decision": summary.get("decision"),
                    "phase3_pass": summary.get("phase3_pass"),
                    "scene_id": scene,
                    "selected_variant_id": variant,
                    "status": "selected_variant_missing_metric_row",
                }
            )
            continue
        row = sub.iloc[0].to_dict()
        rows.append(
            {
                "schema_version": "stream4d_v103_d4rt_filter_attribution_row_v1",
                "phase_id": PHASE_ID,
                "filter_label": label,
                "root": _rel(root),
                "decision": summary.get("decision"),
                "phase3_pass": summary.get("phase3_pass"),
                "failure_count": summary.get("failure_count"),
                "variant_family": summary.get("variant_family"),
                "scene_id": scene,
                "selected_variant_id": variant,
                "retained_carrier_rate": row.get("retained_carrier_rate"),
                "retained_carrier_count": row.get("retained_carrier_count"),
                "total_carrier_count": row.get("total_carrier_count"),
                "object_like_mask_support_p10": row.get("object_like_mask_support_p10"),
                "boundary_band_support_p10": row.get("boundary_band_support_p10"),
                "mask_support_coverage_after_filter": row.get("mask_support_coverage_after_filter"),
                "broad_mask_participation_rate": row.get("broad_mask_participation_rate"),
                "broad_relative_reduction": row.get("broad_relative_reduction"),
                "semantic_contradiction_rate": row.get("semantic_contradiction_rate"),
                "semantic_relative_reduction": row.get("semantic_relative_reduction"),
                "normalized_jitter_p90": row.get("normalized_jitter_p90"),
                "jitter_relative_reduction": row.get("jitter_relative_reduction"),
                "competing_mask_conflict_rate": row.get("competing_mask_conflict_rate"),
                "competing_mask_conflict_relative_reduction": row.get("competing_mask_conflict_relative_reduction"),
                "source_risk_score_mean": row.get("source_risk_score_mean"),
                "source_risk_relative_reduction": row.get("source_risk_relative_reduction"),
                "uses_gt_for_threshold": row.get("uses_gt_for_threshold"),
                "uses_future": row.get("uses_future"),
            }
        )
    return rows


def _phase3_filter_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in PHASE3_FILTER_ROOTS:
        rows.extend(_selected_rows_for_filter(_project(spec["root"]), spec["filter_label"]))
    return rows


def _stage_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, root in STAGE_SUMMARY_ROOTS:
        root = _project(root)
        summary = _read_json(root / "summary.json")
        if not summary:
            rows.append(
                {
                    "schema_version": "stream4d_v103_stage_status_row_v1",
                    "phase_id": PHASE_ID,
                    "stage_label": label,
                    "root": _rel(root),
                    "status": "missing_summary",
                }
            )
            continue
        row = {
            "schema_version": "stream4d_v103_stage_status_row_v1",
            "phase_id": PHASE_ID,
            "stage_label": label,
            "root": _rel(root),
            "decision": summary.get("decision"),
            "failure_count": summary.get("failure_count"),
            "phaseS0_pass": summary.get("phaseS0_pass"),
            "phaseS1_pass": summary.get("phaseS1_pass"),
            "phaseS2_pass": summary.get("phaseS2_pass"),
            "phaseS3_pass": summary.get("phaseS3_pass"),
            "phaseS4_pass": summary.get("phaseS4_pass"),
            "phase3_pass": summary.get("phase3_pass"),
            "best_MV_AP_window": summary.get("best_MV_AP_window"),
            "best_minus_baseline_MV_AP_window": summary.get("best_minus_baseline_MV_AP_window"),
            "best_MV_AP_scene": summary.get("best_MV_AP_scene"),
            "real_minus_shuffled_MV_AP_scene": summary.get("real_minus_shuffled_MV_AP_scene"),
            "stable_all_root_variant_ids": summary.get("stable_all_root_variant_ids"),
            "stable_subchunk16_variant_ids": summary.get("stable_subchunk16_variant_ids"),
        }
        rows.append(row)
    return rows


def _find_filter(rows: list[dict[str, Any]], label: str, scene: str) -> dict[str, Any]:
    for row in rows:
        if row.get("filter_label") == label and row.get("scene_id") == scene:
            return row
    return {}


def _attribution_rows(provider_rows: list[dict[str, Any]], filter_rows: list[dict[str, Any]], stage_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    d4rt48_fresh = [
        row
        for row in provider_rows
        if row.get("provider_name") == "OpenD4RT_48CLIP_9Mix_NoCropAUG" and _boolish(row.get("fresh_d4rt_decode"))
    ]
    d32_s0011 = _find_filter(filter_rows, "d4rt32_q5c_competing_repair5", "scene0011_00")
    d48_s0011 = _find_filter(filter_rows, "d4rt48mix_q5c_scene0011_competing_repair5", "scene0011_00")
    d48_mb_s0011 = _find_filter(filter_rows, "d4rt48mix_maskbalanced8_competing_repair5", "scene0011_00")
    d48_mb_s0050 = _find_filter(filter_rows, "d4rt48mix_maskbalanced8_competing_repair5", "scene0050_00")
    d48_sem_s0011 = _find_filter(filter_rows, "d4rt48mix_maskbalanced8_scene0011_semantic_veto_repair6", "scene0011_00")
    d48_sem_s0050 = _find_filter(filter_rows, "d4rt48mix_maskbalanced8_scene0050_semantic_veto_repair6", "scene0050_00")
    stage = {str(row.get("stage_label")): row for row in stage_rows}

    def f(row: dict[str, Any], key: str) -> float:
        return _float(row.get(key))

    rows = [
        {
            "schema_version": "stream4d_v103_d4rt_attribution_row_v1",
            "phase_id": PHASE_ID,
            "question": "has_new_d4rt_weight_been_tried",
            "answer": "yes" if len(d4rt48_fresh) >= 2 else "partial_or_missing",
            "evidence": f"fresh OpenD4RT_48CLIP_9Mix_NoCropAUG decode roots={len(d4rt48_fresh)}",
            "method_implication": "The local new weight is already represented by d4rt48mix artifacts; do not describe v103 as only tested on 32CLIP.",
        },
        {
            "schema_version": "stream4d_v103_d4rt_attribution_row_v1",
            "phase_id": PHASE_ID,
            "question": "is_phase3_clean_carrier_failure_only_filter_algorithm",
            "answer": "no",
            "evidence": (
                "scene0011 q5c 32CLIP selected object_like_p10="
                f"{f(d32_s0011, 'object_like_mask_support_p10')}, broad={f(d32_s0011, 'broad_mask_participation_rate')}; "
                "same query with 48CLIP selected object_like_p10="
                f"{f(d48_s0011, 'object_like_mask_support_p10')}, broad={f(d48_s0011, 'broad_mask_participation_rate')}"
            ),
            "method_implication": "Provider quality mattered for scene0011 Phase3; this is not explained by thresholding/filter code alone.",
        },
        {
            "schema_version": "stream4d_v103_d4rt_attribution_row_v1",
            "phase_id": PHASE_ID,
            "question": "did_mask_balanced_query_fix_phase3_by_itself",
            "answer": "not_the_primary_scene0011_factor",
            "evidence": (
                "48CLIP q5c scene0011 already passes Phase3; maskbalanced8 keeps the same selected E5 gate with "
                f"object_like_p10 {f(d48_s0011, 'object_like_mask_support_p10')} -> {f(d48_mb_s0011, 'object_like_mask_support_p10')} "
                f"and broad {f(d48_s0011, 'broad_mask_participation_rate')} -> {f(d48_mb_s0011, 'broad_mask_participation_rate')}"
            ),
            "method_implication": "Mask-balanced probes are useful for query fairness/cache, but current evidence does not show they alone solved the decisive Phase3 gate.",
        },
        {
            "schema_version": "stream4d_v103_d4rt_attribution_row_v1",
            "phase_id": PHASE_ID,
            "question": "can_48mix_maskbalanced_phase3_provide_reliable_carriers",
            "answer": "yes_for_phase3_gate_not_full_method",
            "evidence": (
                "competing_repair5 passes all-scene roots; semantic_veto_repair6 passes scene0011 and scene0050. "
                f"scene0011 V1 object_like_p10={f(d48_sem_s0011, 'object_like_mask_support_p10')}; "
                f"scene0050 V1 object_like_p10={f(d48_sem_s0050, 'object_like_mask_support_p10')}"
            ),
            "method_implication": "The current remaining blocker is downstream use of carriers, not absence of any Phase3 reliable carrier artifact.",
        },
        {
            "schema_version": "stream4d_v103_d4rt_attribution_row_v1",
            "phase_id": PHASE_ID,
            "question": "is_there_evidence_geometry_affinity_calculation_bug",
            "answer": "no_current_evidence",
            "evidence": (
                "phase3 filter decision audit says "
                f"{stage.get('phase3_filter_decision_audit', {}).get('decision')}; "
                "phase4 correctness root is present with decision "
                f"{stage.get('phase4_affinity_correctness', {}).get('decision')}"
            ),
            "method_implication": "Existing evidence points to carrier role/use and downstream graph/history gates, not a known arithmetic bug in geometric affinity.",
        },
        {
            "schema_version": "stream4d_v103_d4rt_attribution_row_v1",
            "phase_id": PHASE_ID,
            "question": "where_is_the_current_real_blocker",
            "answer": "phaseS3_phaseS4_and_phaseS5_still_no_go",
            "evidence": (
                f"S3={stage.get('phaseS3_scaffolded_mask_graph', {}).get('decision')}, "
                f"S4={stage.get('phaseS4_post_birth_history_inheritance', {}).get('decision')}, "
                f"S5 barrier={stage.get('phaseS5_da3_preregistered_barrier', {}).get('decision')}"
            ),
            "method_implication": "Do not claim v103 success; next repair should target scaffolded edge intervention/history inheritance rather than more single-set carrier filtering.",
        },
    ]
    return rows


def _gate_rows(provider_rows: list[dict[str, Any]], filter_rows: list[dict[str, Any]], stage_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stage = {str(row.get("stage_label")): row for row in stage_rows}
    d4rt48_fresh_scenes = {
        str(row.get("scene_id"))
        for row in provider_rows
        if row.get("provider_name") == "OpenD4RT_48CLIP_9Mix_NoCropAUG" and _boolish(row.get("fresh_d4rt_decode"))
    }
    phase3_new_weight_pass = all(
        _find_filter(filter_rows, label, scene).get("phase3_pass") is True
        or str(_find_filter(filter_rows, label, scene).get("phase3_pass")).lower() == "true"
        for label, scene in [
            ("d4rt48mix_maskbalanced8_scene0011_semantic_veto_repair6", "scene0011_00"),
            ("d4rt48mix_maskbalanced8_scene0050_semantic_veto_repair6", "scene0050_00"),
        ]
    )
    rows = [
        {
            "schema_version": "stream4d_v103_d4rt_attribution_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_name": "new_d4rt48mix_weight_available_and_fresh_decoded",
            "passed": {"scene0011_00", "scene0050_00"}.issubset(d4rt48_fresh_scenes),
            "observed": sorted(d4rt48_fresh_scenes),
            "required": ["scene0011_00", "scene0050_00"],
        },
        {
            "schema_version": "stream4d_v103_d4rt_attribution_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_name": "phase3_new_weight_filter_gate_pass",
            "passed": phase3_new_weight_pass,
            "observed": "semantic_veto_repair6 scene0011/scene0050 phase3_pass",
            "required": True,
        },
        {
            "schema_version": "stream4d_v103_d4rt_attribution_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_name": "phaseS3_scaffolded_mask_graph_pass",
            "passed": _boolish(stage.get("phaseS3_scaffolded_mask_graph", {}).get("phaseS3_pass")),
            "observed": stage.get("phaseS3_scaffolded_mask_graph", {}).get("decision"),
            "required": "PASS",
        },
        {
            "schema_version": "stream4d_v103_d4rt_attribution_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_name": "phaseS4_post_birth_history_pass",
            "passed": _boolish(stage.get("phaseS4_post_birth_history_inheritance", {}).get("phaseS4_pass")),
            "observed": stage.get("phaseS4_post_birth_history_inheritance", {}).get("decision"),
            "required": "PASS",
        },
        {
            "schema_version": "stream4d_v103_d4rt_attribution_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_name": "phaseS5_preregistered_barrier_stable_variant_exists",
            "passed": bool(stage.get("phaseS5_da3_preregistered_barrier", {}).get("stable_all_root_variant_ids")),
            "observed": stage.get("phaseS5_da3_preregistered_barrier", {}).get("stable_all_root_variant_ids"),
            "required": "non_empty",
        },
    ]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    provider_rows = _phase2_provider_rows()
    filter_rows = _phase3_filter_rows()
    stage_rows = _stage_rows()
    attribution_rows = _attribution_rows(provider_rows, filter_rows, stage_rows)
    gate_rows = _gate_rows(provider_rows, filter_rows, stage_rows)
    failure_rows = [
        {
            **row,
            "repair_direction": (
                "Continue plan-backed S3/S4 repair. Do not claim D4RT filtering or DA3 provider success as full v103 success."
            ),
        }
        for row in gate_rows
        if not _boolish(row.get("passed"))
    ]

    decision = (
        "D4RT48MIX_PROVIDER_TRIED_PHASE3_REPAIRED__DOWNSTREAM_S3_S4_S5_STILL_NO_GO"
        if failure_rows
        else "PASS_D4RT_PROVIDER_QUERY_FILTER_ATTRIBUTION"
    )
    summary = {
        "schema_version": "stream4d_v103_d4rt_provider_query_filter_attribution_summary_v1",
        "phase_id": PHASE_ID,
        "decision": decision,
        "provider_row_count": len(provider_rows),
        "filter_row_count": len(filter_rows),
        "stage_row_count": len(stage_rows),
        "attribution_row_count": len(attribution_rows),
        "failure_count": len(failure_rows),
        "uses_gt_for_prediction": False,
        "uses_gt_for_threshold": False,
        "uses_future": False,
        "truthfulness_note": (
            "This script only summarizes existing artifacts. It does not perform threshold selection from GT diagnostics "
            "and does not create object predictions."
        ),
        "outputs": {
            "provider_rows": output_root / "provider_query_rows.csv",
            "filter_rows": output_root / "filter_attribution_rows.csv",
            "stage_rows": output_root / "stage_status_rows.csv",
            "attribution_rows": output_root / "attribution_rows.csv",
            "gate_rows": output_root / "gate_rows.csv",
            "failure_rows": output_root / "failure_rows.csv",
        },
    }

    _write_csv(output_root / "provider_query_rows.csv", provider_rows)
    _write_csv(output_root / "filter_attribution_rows.csv", filter_rows)
    _write_csv(output_root / "stage_status_rows.csv", stage_rows)
    _write_csv(output_root / "attribution_rows.csv", attribution_rows)
    _write_csv(output_root / "gate_rows.csv", gate_rows)
    _write_csv(output_root / "failure_rows.csv", failure_rows)
    _write_json(output_root / "summary.json", summary)

    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 2 if failure_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
