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


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"

PHASE_ID = "v103_r2_phase0_fact_lock"
DEFAULT_OUT = AUDIT_ROOT / PHASE_ID

DEFAULT_ROOTS = {
    "baseline_current_local": AUDIT_ROOT / "v100_phase2c_overlap3_local_repair",
    "phase0_contract": AUDIT_ROOT / "v103_phase0_contract",
    "phase2_scene0011": AUDIT_ROOT
    / "v103_phase2_stratified_q5c_objlike16384_scene0011_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8",
    "phase2_scene0050": AUDIT_ROOT
    / "v103_phase2_stratified_q5c_objlike16384_scene0050_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8",
    "phase6d": AUDIT_ROOT
    / "v103_phase6d_f2_skeleton_affinity_merge_phase9n_suppS1_d4rt48mix_s5repair_r3_allclean_gate_r1",
    "phase6d_directpair_guard": AUDIT_ROOT
    / "v103_phase6d_f2_skeleton_affinity_merge_phase9n_suppS1_d4rt48mix_s5repair_r4_directpair_guard",
    "phaseS4": AUDIT_ROOT / "v103_supp_phaseS4_post_birth_history_inheritance_phase6d_s5repair_r3_gate_r1",
    "phaseS5_provider_bridge": AUDIT_ROOT / "v103_supp_phaseS5_da3_provider_bridge_failure_r1",
    "phaseS5_preregistered": AUDIT_ROOT / "v103_supp_phaseS5_da3_preregistered_barrier_r1",
    "phase9e_expected_c0001": AUDIT_ROOT / "v103_phase9e_d4rt_anchor_da3_induced_carriers_suppS1_d4rt48mix_s5repair_r1",
    "phase9j": AUDIT_ROOT / "v103_phase9j_broad_mask_threshold_relaxation_r1",
    "phase9k": AUDIT_ROOT / "v103_phase9k_da3_semantic_soft_seed_growth_r2_coverage",
}

DOCS = {
    "method_r2": REPO_ROOT / "docs/stream4d_v103_method_thinking_training_free_primitive_affinity_field_adjusted_r2.md",
    "supplement_r2": REPO_ROOT / "docs/stream4d_v103_supplement_r2_semantic_soft_candidate_plan.md",
    "execution_log": REPO_ROOT / "docs/stream4d_v103_执行日志.md",
    "retrospective_log": REPO_ROOT / "docs/stream4d_v103_实验结果复盘.md",
    "v65_evaluator": STREAM3D_ROOT / "tools/run_v65_scene_multiview_ap.py",
}

REQUIRED_FILES = {
    "baseline_current_local": ["summary.json", "variant_metric_rows.csv"],
    "phase0_contract": ["summary.json", "metric_contract.json", "baseline_metric_rows.csv"],
    "phase2_scene0011": ["summary.json", "gate_rows.csv"],
    "phase2_scene0050": ["summary.json", "gate_rows.csv"],
    "phase6d": ["summary.json", "gate_rows.csv", "merge_selected_rows.csv", "failure_rows.csv"],
    "phase6d_directpair_guard": ["summary.json", "gate_rows.csv", "failure_rows.csv"],
    "phaseS4": ["summary.json", "gate_rows.csv", "failure_rows.csv", "scene_metric_rows.csv", "control_metric_rows.csv"],
    "phaseS5_provider_bridge": ["summary.json", "gate_rows.csv", "failure_rows.csv"],
    "phaseS5_preregistered": ["summary.json", "gate_rows.csv", "failure_rows.csv"],
    "phase9e_expected_c0001": ["summary.json", "scene_summary_rows.csv"],
    "phase9j": ["summary.json", "scene_summary_rows.csv", "failure_rows.csv"],
    "phase9k": ["summary.json", "scene_summary_rows.csv", "failure_rows.csv"],
}


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
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def _sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _floatish(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _path_from_summary(summary: dict[str, Any], *keys: str) -> str:
    cur: Any = summary
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return ""
        cur = cur[key]
    return str(cur)


def _artifact_rows(roots: dict[str, Path], summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact_id, root in roots.items():
        summary = summaries.get(artifact_id, {})
        for rel_name in REQUIRED_FILES.get(artifact_id, ["summary.json"]):
            path = root / rel_name
            rows.append(
                {
                    "schema_version": "stream4d_v103_r2_phase0_input_artifact_row_v1",
                    "phase_id": PHASE_ID,
                    "artifact_id": artifact_id,
                    "root": _rel(root),
                    "file_role": rel_name,
                    "path": _rel(path),
                    "exists": path.exists(),
                    "size_bytes": path.stat().st_size if path.exists() and path.is_file() else "",
                    "sha256": _sha256(path) if path.exists() and path.is_file() else "",
                    "source_phase_id": summary.get("phase_id", ""),
                    "decision": summary.get("decision", summary.get("phase_decision", "")),
                    "uses_gt_for_prediction": summary.get("uses_gt_for_prediction", ""),
                    "uses_future": summary.get("uses_future", ""),
                }
            )
    for doc_id, path in DOCS.items():
        rows.append(
            {
                "schema_version": "stream4d_v103_r2_phase0_input_artifact_row_v1",
                "phase_id": PHASE_ID,
                "artifact_id": doc_id,
                "root": _rel(path.parent),
                "file_role": path.name,
                "path": _rel(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() and path.is_file() else "",
                "sha256": _sha256(path) if path.exists() and path.is_file() else "",
                "source_phase_id": "doc_or_evaluator",
                "decision": "",
                "uses_gt_for_prediction": "",
                "uses_future": "",
            }
        )
    return rows


def _metric_contract_rows(roots: dict[str, Path], summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    contract_path = roots["phase0_contract"] / "metric_contract.json"
    contract = _read_json(contract_path)
    rows.append(
        {
            "schema_version": "stream4d_v103_r2_phase0_metric_contract_row_v1",
            "phase_id": PHASE_ID,
            "row_type": "canonical_evaluator",
            "contract_path": _rel(contract_path),
            "canonical_evaluator_path": contract.get("canonical_evaluator_path", ""),
            "canonical_evaluator_exists": DOCS["v65_evaluator"].exists(),
            "canonical_evaluator_sha256_from_contract": contract.get("canonical_evaluator_sha256", ""),
            "canonical_evaluator_sha256_current": _sha256(DOCS["v65_evaluator"]),
            "formal_metric_source_eq_v65": contract.get("formal_metric_source_eq_v65", ""),
            "has_sparse_scene_iou": contract.get("has_sparse_scene_iou", ""),
            "has_summarize_iou": contract.get("has_summarize_iou", ""),
            "local_metric": "MV_AP_window",
            "scene_metric": "MV_AP_scene",
            "method_chunk_size": contract.get("method_chunk_size", ""),
            "frame_stride": contract.get("frame_stride", ""),
            "chunk_overlap": contract.get("chunk_overlap", ""),
        }
    )
    for row in _read_csv(roots["phase0_contract"] / "baseline_metric_rows.csv"):
        if row.get("baseline_role") != "current_strong_local_baseline":
            continue
        rows.append(
            {
                "schema_version": "stream4d_v103_r2_phase0_metric_contract_row_v1",
                "phase_id": PHASE_ID,
                "row_type": "baseline_metric",
                "dataset_split": row.get("dataset_split", ""),
                "source_phase_id": row.get("source_phase_id", ""),
                "source_artifact": row.get("source_artifact", ""),
                "variant_id": row.get("variant_id", ""),
                "MV_AP_window": row.get("MV_AP_window", ""),
                "MV_AP50_window": row.get("MV_AP50_window", ""),
                "fragmented_MV_AP_scene": row.get("fragmented_MV_AP_scene", ""),
                "fragmented_MV_AP50_scene": row.get("fragmented_MV_AP50_scene", ""),
                "uses_gt_for_prediction": row.get("uses_gt_for_prediction", ""),
                "uses_future": row.get("uses_future", ""),
                "metric_source": row.get("metric_source", ""),
            }
        )
    phase6d = summaries["phase6d"]
    phase_s4 = summaries["phaseS4"]
    phase9k_rows = _read_csv(roots["phase9k"] / "scene_summary_rows.csv")
    for scene_row in phase9k_rows:
        rows.append(
            {
                "schema_version": "stream4d_v103_r2_phase0_metric_contract_row_v1",
                "phase_id": PHASE_ID,
                "row_type": "phase9k_scene_fact",
                "scene_id": scene_row.get("scene_id", ""),
                "best_variant_id": scene_row.get("best_variant_id", ""),
                "clean_component_count": scene_row.get("best_clean_component_count", ""),
                "clean_induced_object_like_obs_count": scene_row.get("best_clean_induced_object_like_obs_count", ""),
                "clean_induced_broad_rate": scene_row.get("best_clean_induced_broad_obs_rate", ""),
                "gate_pass": scene_row.get("phase9k_semantic_soft_seed_growth_gate_pass", ""),
                "blocker": scene_row.get("blocker", ""),
            }
        )
    rows.append(
        {
            "schema_version": "stream4d_v103_r2_phase0_metric_contract_row_v1",
            "phase_id": PHASE_ID,
            "row_type": "phase6d_local_gate_fact",
            "root": _rel(roots["phase6d"]),
            "decision": phase6d.get("decision", ""),
            "best_real_variant_id": phase6d.get("best_real_variant_id", ""),
            "best_real_MV_AP_window": phase6d.get("best_real_MV_AP_window", ""),
            "best_real_MV_AP50_window": phase6d.get("best_real_MV_AP50_window", ""),
            "best_real_minus_replay_MV_AP_window": phase6d.get("best_real_minus_replay_MV_AP_window", ""),
            "best_real_minus_best_shuffled_MV_AP_window": phase6d.get("best_real_minus_best_shuffled_MV_AP_window", ""),
            "accepted_specific_conflict_count_best_real": phase6d.get("accepted_specific_conflict_count_best_real", ""),
        }
    )
    rows.append(
        {
            "schema_version": "stream4d_v103_r2_phase0_metric_contract_row_v1",
            "phase_id": PHASE_ID,
            "row_type": "phaseS4_history_fact",
            "root": _rel(roots["phaseS4"]),
            "decision": phase_s4.get("decision", ""),
            "current_object_source": phase_s4.get("current_object_source", ""),
            "phase6d_variant_id": phase_s4.get("phase6d_variant_id", ""),
            "best_MV_AP_scene": phase_s4.get("best_MV_AP_scene", ""),
            "real_minus_shuffled_MV_AP_scene": phase_s4.get("real_minus_shuffled_MV_AP_scene", ""),
            "real_minus_stale_MV_AP_scene": phase_s4.get("real_minus_stale_MV_AP_scene", ""),
            "real_minus_semantic_MV_AP_scene": phase_s4.get("real_minus_semantic_MV_AP_scene", ""),
            "best_local_MV_AP_window_after_history": phase_s4.get("best_local_MV_AP_window_after_history", ""),
        }
    )
    return rows


def _frame_ids(summary: dict[str, Any]) -> list[int]:
    values = summary.get("frame_ids", [])
    if isinstance(values, list):
        return [int(v) for v in values]
    return []


def _s4_scene_rows(summary: dict[str, Any], support_source: str = "h2_real") -> list[dict[str, Any]]:
    rows = summary.get("scene_meta", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict) and row.get("support_source") == support_source]


def _provenance_rows(roots: dict[str, Path], summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p2_11 = summaries["phase2_scene0011"]
    p2_50 = summaries["phase2_scene0050"]
    phase_s4 = summaries["phaseS4"]
    frame11 = _frame_ids(p2_11)
    frame50 = _frame_ids(p2_50)
    for artifact_id, summary in [("phase2_scene0011", p2_11), ("phase2_scene0050", p2_50)]:
        frames = _frame_ids(summary)
        rows.append(
            {
                "schema_version": "stream4d_v103_r2_phase0_provenance_alignment_row_v1",
                "phase_id": PHASE_ID,
                "alignment_id": f"{artifact_id}_c0001_d4rt48mix_maskbalanced8",
                "artifact_id": artifact_id,
                "path": _rel(roots[artifact_id]),
                "scene_id": summary.get("scene_id", ""),
                "chunk_id": "c0001",
                "chunk_index": summary.get("chunk_index", ""),
                "frame_min": min(frames) if frames else "",
                "frame_max": max(frames) if frames else "",
                "frame_count": len(frames),
                "mask_registry_root": summary.get("mask_root", ""),
                "d4rt_carrier_provider_id": "D4RT48Mix_maskbalanced8",
                "query_count_per_frame": summary.get("query_count_per_frame", ""),
                "query_generation_policy": summary.get("query_generation_policy", ""),
                "carrier_batch_cache": summary.get("carrier_batch_cache", ""),
                "pass": bool(summary.get("chunk_index") == 1 and len(frames) == 32),
                "note": "Phase2 current c0001 root used by S4 real scene_meta.",
            }
        )
    rows.append(
        {
            "schema_version": "stream4d_v103_r2_phase0_provenance_alignment_row_v1",
            "phase_id": PHASE_ID,
            "alignment_id": "phase2_scene_frame_universe_match",
            "artifact_id": "phase2_scene0011_vs_scene0050",
            "path": f"{_rel(roots['phase2_scene0011'])};{_rel(roots['phase2_scene0050'])}",
            "scene_id": "scene0011_00;scene0050_00",
            "chunk_id": "c0001",
            "frame_min": min(frame11) if frame11 else "",
            "frame_max": max(frame11) if frame11 else "",
            "frame_count": len(frame11),
            "pass": frame11 == frame50 and len(frame11) == 32,
            "note": "Both dev scenes should share the same c0001 stride-5 frame ids.",
        }
    )
    phase_s4_phase6d_root = str(phase_s4.get("phase6d_root", ""))
    selected_phase6d_root = _rel(roots["phase6d"])
    rows.append(
        {
            "schema_version": "stream4d_v103_r2_phase0_provenance_alignment_row_v1",
            "phase_id": PHASE_ID,
            "alignment_id": "phaseS4_uses_selected_phase6d_root",
            "artifact_id": "phaseS4",
            "path": _rel(roots["phaseS4"]),
            "scene_id": "all",
            "chunk_id": "c0001",
            "phaseS4_phase6d_root": phase_s4_phase6d_root,
            "selected_phase6d_root": selected_phase6d_root,
            "phase6d_variant_id": phase_s4.get("phase6d_variant_id", ""),
            "pass": phase_s4_phase6d_root == selected_phase6d_root,
            "note": "History fact lock binds to the Phase6d root actually consumed by S4.",
        }
    )
    s4_real_rows = _s4_scene_rows(phase_s4)
    selected_phase2 = {
        "scene0011_00": _rel(roots["phase2_scene0011"]),
        "scene0050_00": _rel(roots["phase2_scene0050"]),
    }
    for row in s4_real_rows:
        scene_id = str(row.get("scene_id", ""))
        frames = row.get("frame_ids", [])
        frames = [int(v) for v in frames] if isinstance(frames, list) else []
        observed_phase2 = str(row.get("phase2_root", ""))
        rows.append(
            {
                "schema_version": "stream4d_v103_r2_phase0_provenance_alignment_row_v1",
                "phase_id": PHASE_ID,
                "alignment_id": f"phaseS4_real_scene_phase2_alignment_{scene_id}",
                "artifact_id": "phaseS4",
                "path": _rel(roots["phaseS4"]),
                "scene_id": scene_id,
                "chunk_id": "c0001",
                "frame_min": min(frames) if frames else "",
                "frame_max": max(frames) if frames else "",
                "frame_count": len(frames),
                "phaseS4_scene_phase2_root": observed_phase2,
                "selected_phase2_root": selected_phase2.get(scene_id, ""),
                "object_with_valid_carrier_hit_rate": row.get("object_with_valid_carrier_hit_rate", ""),
                "projection_backend": row.get("projection_backend", ""),
                "pass": observed_phase2 == selected_phase2.get(scene_id, "") and len(frames) == 32,
                "note": "S4 real rows must use the selected D4RT48Mix c0001 Phase2 roots.",
            }
        )
    for artifact_id in ["phase9j", "phase9k"]:
        scene_rows = _read_csv(roots[artifact_id] / "scene_summary_rows.csv")
        for row in scene_rows:
            manifest = row.get("da3_projection_manifest", "")
            observed_phase9e_root = str(row.get("phase9e_root", ""))
            expected_phase9e_root = _rel(roots["phase9e_expected_c0001"])
            root_matches = observed_phase9e_root == expected_phase9e_root
            rows.append(
                {
                    "schema_version": "stream4d_v103_r2_phase0_provenance_alignment_row_v1",
                    "phase_id": PHASE_ID,
                    "alignment_id": f"{artifact_id}_da3_provider_manifest_{row.get('scene_id', '')}",
                    "artifact_id": artifact_id,
                    "path": _rel(roots[artifact_id]),
                    "scene_id": row.get("scene_id", ""),
                    "chunk_id": "chunk32_process252",
                    "da3_provider_id": "DA3/DA3-GIANT projection cache from manifest",
                    "phase9e_root": observed_phase9e_root,
                    "expected_phase9e_root": expected_phase9e_root,
                    "da3_projection_manifest": manifest,
                    "uses_gt_for_prediction": row.get("uses_gt_for_prediction", ""),
                    "uses_gt_for_gate": row.get("uses_gt_for_gate", ""),
                    "pass": root_matches
                    and row.get("uses_gt_for_prediction", "False") == "False"
                    and row.get("uses_gt_for_gate", "False") == "False",
                    "note": (
                        "Phase9j/k must be regenerated on the c0001 Phase9e/Phase2 universe before R2-1; "
                        "first32 provider diagnostics are not method-ready inputs."
                    ),
                }
            )
    for artifact_id in ["phase9j"]:
        for scene_id, phase2_key in [
            ("scene0011_00", "phase2_scene0011"),
            ("scene0050_00", "phase2_scene0050"),
        ]:
            path = roots[artifact_id] / scene_id / "extra_candidate_rows.csv"
            if not path.exists():
                continue
            rows_in = _read_csv(path)
            frame_values = [int(row["frame_id"]) for row in rows_in if str(row.get("frame_id", "")).strip()]
            phase2_frames = _frame_ids(summaries[phase2_key])
            expected_min = min(phase2_frames) if phase2_frames else ""
            expected_max = max(phase2_frames) if phase2_frames else ""
            observed_min = min(frame_values) if frame_values else ""
            observed_max = max(frame_values) if frame_values else ""
            rows.append(
                {
                    "schema_version": "stream4d_v103_r2_phase0_provenance_alignment_row_v1",
                    "phase_id": PHASE_ID,
                    "alignment_id": f"{artifact_id}_candidate_frame_universe_{scene_id}",
                    "artifact_id": artifact_id,
                    "path": _rel(path),
                    "scene_id": scene_id,
                    "chunk_id": "c0001",
                    "frame_min": observed_min,
                    "frame_max": observed_max,
                    "expected_frame_min": expected_min,
                    "expected_frame_max": expected_max,
                    "pass": observed_min == expected_min and observed_max == expected_max,
                    "note": "Candidate rows must use global c0001 frame ids, not first32 local frame ids.",
                }
            )
    return rows


def _gate_row(gate_id: str, passed: bool, observed: Any, required: Any, repair_direction: str = "") -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v103_r2_phase0_gate_row_v1",
        "phase_id": PHASE_ID,
        "gate_id": gate_id,
        "pass": bool(passed),
        "observed": observed,
        "required": required,
        "repair_direction": repair_direction,
    }


def _decision(summary: dict[str, Any]) -> str:
    return str(summary.get("decision", summary.get("phase_decision", "")))


def _contains(path: Path, needle: str) -> bool:
    if not path.exists() or not path.is_file():
        return False
    return needle in path.read_text(encoding="utf-8", errors="ignore")


def _gates_and_failures(
    roots: dict[str, Path],
    summaries: dict[str, dict[str, Any]],
    input_rows: list[dict[str, Any]],
    provenance_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gates: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    missing_inputs = [row["path"] for row in input_rows if row.get("exists") is False]
    gates.append(
        _gate_row(
            "all_required_input_artifacts_exist",
            not missing_inputs,
            missing_inputs,
            "[]",
            "Explicitly bind existing c0001 roots or regenerate missing aligned artifacts.",
        )
    )

    contract = _read_json(roots["phase0_contract"] / "metric_contract.json")
    evaluator_sha_match = contract.get("canonical_evaluator_sha256") == _sha256(DOCS["v65_evaluator"])
    gates.append(
        _gate_row(
            "v65_evaluator_contract_readable_and_hash_matches",
            bool(
                contract.get("formal_metric_source_eq_v65")
                and contract.get("has_sparse_scene_iou")
                and contract.get("has_summarize_iou")
                and evaluator_sha_match
            ),
            {
                "formal_metric_source_eq_v65": contract.get("formal_metric_source_eq_v65"),
                "has_sparse_scene_iou": contract.get("has_sparse_scene_iou"),
                "has_summarize_iou": contract.get("has_summarize_iou"),
                "evaluator_sha_match": evaluator_sha_match,
            },
            "all true",
            "Re-run v103 Phase0 contract before R2 if evaluator contract drifted.",
        )
    )

    decision_expectations = {
        "phase6d": "PASS_PHASE6D_S3_STYLE_LOCAL_GATE",
        "phaseS4": "NO_GO_REPAIR_PHASES4_POST_BIRTH_HISTORY_INHERITANCE",
        "phaseS5_provider_bridge": "NO_GO_PHASES5_DA3_PROVIDER_BRIDGE_FRONTIER_DIAGNOSTIC",
        "phaseS5_preregistered": "NO_GO_PHASES5_DA3_PREREGISTERED_BARRIER_STABLE_PROVIDER_DIAGNOSTIC",
        "phase9j": "PASS_PHASE9J_RELAXED_BROAD_THRESHOLD_HAS_ACTIONABLE_CANDIDATES",
        "phase9k": "NO_GO_PHASE9K_DA3_SEMANTIC_SOFT_SEED_GROWTH",
    }
    observed_decisions = {key: _decision(summaries[key]) for key in decision_expectations}
    gates.append(
        _gate_row(
            "key_artifact_decisions_match_current_r2_boundary",
            observed_decisions == decision_expectations,
            observed_decisions,
            decision_expectations,
            "Inspect root binding; do not mix older first32/c0000 artifacts with c0001 R2.",
        )
    )

    bad_provenance = [row for row in provenance_rows if row.get("pass") is False]
    gates.append(
        _gate_row(
            "provenance_alignment_passes",
            not bad_provenance,
            [row.get("alignment_id", "") for row in bad_provenance],
            "[]",
            "Follow R2-0 repair ladder: explicit phase2 roots, c0001, pair source/filter; regenerate c0001 if needed.",
        )
    )

    method_doc_ok = _contains(DOCS["method_r2"], "semantic broad / background proxy")
    plan_doc_ok = _contains(DOCS["supplement_r2"], "Phase R2-0")
    gates.append(
        _gate_row(
            "r2_docs_readable_and_match_requested_plan",
            method_doc_ok and plan_doc_ok,
            {"method_doc_has_semantic_broad_boundary": method_doc_ok, "supplement_has_phase_r2_0": plan_doc_ok},
            "both true",
            "Use the exact R2 docs supplied by the user before proceeding.",
        )
    )

    no_gt_prediction = True
    no_future = True
    truth_rows: list[dict[str, Any]] = []
    for key, summary in summaries.items():
        uses_gt = _boolish(summary.get("uses_gt_for_prediction", False))
        uses_future = _boolish(summary.get("uses_future", False))
        if uses_gt:
            no_gt_prediction = False
        if uses_future:
            no_future = False
        truth_rows.append({"artifact_id": key, "uses_gt_for_prediction": uses_gt, "uses_future": uses_future})
    gates.append(
        _gate_row(
            "method_path_uses_no_gt_for_prediction",
            no_gt_prediction,
            truth_rows,
            "all uses_gt_for_prediction false or absent",
            "Remove GT-dependent method-path artifact before R2-1/R2-2.",
        )
    )
    gates.append(
        _gate_row(
            "method_path_uses_no_future",
            no_future,
            truth_rows,
            "all uses_future false or absent",
            "Regenerate any artifact that peeks beyond current chunk.",
        )
    )

    phase9k_rows = _read_csv(roots["phase9k"] / "scene_summary_rows.csv")
    phase9k_obs = {
        row.get("scene_id", ""): {
            "clean_component_count": row.get("best_clean_component_count", ""),
            "clean_induced_object_like_obs_count": row.get("best_clean_induced_object_like_obs_count", ""),
            "clean_induced_broad_rate": row.get("best_clean_induced_broad_obs_rate", ""),
            "gate_pass": row.get("phase9k_semantic_soft_seed_growth_gate_pass", ""),
        }
        for row in phase9k_rows
    }
    gates.append(
        _gate_row(
            "phase9k_recorded_as_provider_diagnostic_not_method_success",
            _decision(summaries["phase9k"]) == "NO_GO_PHASE9K_DA3_SEMANTIC_SOFT_SEED_GROWTH",
            phase9k_obs,
            "No-Go diagnostic unless clean_component_count>=5 and object_like_obs>=30 per scene",
            "Proceed to R2-1/R2-2 repair; do not claim provider-ready success from Phase9k.",
        )
    )

    for gate in gates:
        if not gate["pass"]:
            failures.append(
                {
                    "schema_version": "stream4d_v103_r2_phase0_failure_row_v1",
                    "phase_id": PHASE_ID,
                    "failure_id": gate["gate_id"],
                    "severity": "blocker",
                    "observed": gate["observed"],
                    "expected": gate["required"],
                    "repair_direction": gate["repair_direction"],
                }
            )
    return gates, failures


def build(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    output_root = _project(args.output_root)
    roots = {key: _project(value) for key, value in DEFAULT_ROOTS.items()}

    summaries = {key: _read_json(root / "summary.json") for key, root in roots.items()}
    input_rows = _artifact_rows(roots, summaries)
    metric_rows = _metric_contract_rows(roots, summaries)
    provenance_rows = _provenance_rows(roots, summaries)
    gate_rows, failure_rows = _gates_and_failures(roots, summaries, input_rows, provenance_rows)

    phase2_11 = summaries["phase2_scene0011"]
    phase2_50 = summaries["phase2_scene0050"]
    frames = _frame_ids(phase2_11)
    phase9k_rows = _read_csv(roots["phase9k"] / "scene_summary_rows.csv")
    phase9k_min_clean_components = min(
        [_floatish(row.get("best_clean_component_count")) for row in phase9k_rows] or [0.0]
    )
    phase9k_min_induced_objlike = min(
        [_floatish(row.get("best_clean_induced_object_like_obs_count")) for row in phase9k_rows] or [0.0]
    )
    phase9k_max_broad_rate = max(
        [_floatish(row.get("best_clean_induced_broad_obs_rate")) for row in phase9k_rows] or [0.0]
    )
    pass_all = all(row["pass"] for row in gate_rows)

    summary = {
        "schema_version": "stream4d_v103_r2_phase0_summary_v1",
        "phase_id": PHASE_ID,
        "decision": "PASS_R2_PHASE0_FACT_LOCK_READY_FOR_R2_1"
        if pass_all
        else "NO_GO_R2_PHASE0_PROVENANCE_ALIGNMENT_BLOCKER",
        "output_root": _rel(output_root),
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "baseline_current_local_root": _rel(roots["baseline_current_local"]),
        "phase6d_root": _rel(roots["phase6d"]),
        "phase6d_variant_id": summaries["phase6d"].get("best_real_variant_id", ""),
        "phase6d_directpair_guard_root": _rel(roots["phase6d_directpair_guard"]),
        "phaseS4_root": _rel(roots["phaseS4"]),
        "phaseS4_source_mode": summaries["phaseS4"].get("current_object_source", ""),
        "phaseS4_decision": _decision(summaries["phaseS4"]),
        "phaseS5_provider_bridge_root": _rel(roots["phaseS5_provider_bridge"]),
        "phaseS5_preregistered_root": _rel(roots["phaseS5_preregistered"]),
        "phase9e_expected_c0001_root": _rel(roots["phase9e_expected_c0001"]),
        "phase9j_root": _rel(roots["phase9j"]),
        "phase9k_root": _rel(roots["phase9k"]),
        "phase2_scene0011_root": _rel(roots["phase2_scene0011"]),
        "phase2_scene0050_root": _rel(roots["phase2_scene0050"]),
        "scene_ids": [phase2_11.get("scene_id", ""), phase2_50.get("scene_id", "")],
        "chunk_id": "c0001",
        "frame_range": [min(frames), max(frames)] if frames else [],
        "frame_count": len(frames),
        "mask_registry_roots": {
            "scene0011_00": phase2_11.get("mask_root", ""),
            "scene0050_00": phase2_50.get("mask_root", ""),
        },
        "d4rt_carrier_provider_id": "D4RT48Mix_maskbalanced8",
        "da3_provider_id": "DA3/DA3-GIANT projection cache from Phase9j/k manifests",
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "phase9k_min_clean_component_count": phase9k_min_clean_components,
        "phase9k_min_clean_induced_object_like_obs_count": phase9k_min_induced_objlike,
        "phase9k_max_clean_induced_broad_rate": phase9k_max_broad_rate,
        "r2_next_phase": "R2-1 semantic-soft candidate universe reconstruction",
        "truthfulness_note": "R2-0 is a fact/provenance lock only. It does not claim v103 success or provider readiness.",
        "outputs": {
            "input_artifact_rows": _rel(output_root / "input_artifact_rows.csv"),
            "metric_contract_rows": _rel(output_root / "metric_contract_rows.csv"),
            "provenance_alignment_rows": _rel(output_root / "provenance_alignment_rows.csv"),
            "gate_rows": _rel(output_root / "gate_rows.csv"),
            "failure_rows": _rel(output_root / "failure_rows.csv"),
            "summary": _rel(output_root / "summary.json"),
        },
    }

    _write_csv(output_root / "input_artifact_rows.csv", input_rows)
    _write_csv(output_root / "metric_contract_rows.csv", metric_rows)
    _write_csv(output_root / "provenance_alignment_rows.csv", provenance_rows)
    _write_csv(output_root / "gate_rows.csv", gate_rows)
    _write_csv(output_root / "failure_rows.csv", failure_rows)
    _write_json(output_root / "summary.json", summary)
    (output_root / "last_command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stream4D v103 R2 Phase0 fact/provenance lock.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    return parser.parse_args()


def main() -> int:
    summary = build(parse_args())
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if summary["decision"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
