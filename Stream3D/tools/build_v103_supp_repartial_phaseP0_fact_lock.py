#!/usr/bin/env python3
"""Build v103 supplement R2 Phase P0 fact-lock artifacts.

This phase is read-only. It binds the current c0001 D4RT48Mix carrier roots,
CropFormer mask registries, Phase6d D9 local-gate artifact, canonical v65
metric contract, and currently usable semantic evidence roots before any
partial-carrier or semantic-feature construction is allowed to run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"

PHASE_ID = "v103_supp_r2_phaseP0_fact_lock"
DEFAULT_OUT = AUDIT_ROOT / PHASE_ID

PLAN_DOC = REPO_ROOT / "docs" / "stream4d_v103_supplement_repartial_carrier_semantic_evidence_plan.md"
METHOD_DOC = REPO_ROOT / "docs" / "stream4d_v103_method_thinking_training_free_primitive_affinity_field.md"
BASE_PLAN_DOC = REPO_ROOT / "docs" / "stream4d_v103_training_free_primitive_affinity_field_experiment_plan.md"
EXEC_LOG = REPO_ROOT / "docs" / "stream4d_v103_执行日志.md"
RETRO_LOG = REPO_ROOT / "docs" / "stream4d_v103_实验结果复盘.md"
V65_EVALUATOR = STREAM3D_ROOT / "tools" / "run_v65_scene_multiview_ap.py"

D4RT_48MIX_ROOT = REPO_ROOT / "Open-d4rt" / "checkpoints" / "OpenD4RT_48CLIP_9Mix_NoCropAUG"
D4RT_48MIX_CONFIG = D4RT_48MIX_ROOT / "model.yaml"
D4RT_48MIX_CKPT = D4RT_48MIX_ROOT / "opend4rt.ckpt"

ROOTS = {
    "phase0_contract": AUDIT_ROOT / "v103_phase0_contract",
    "current_strong_local_baseline": AUDIT_ROOT / "v100_phase2c_overlap3_local_repair",
    "phase6d_d9_directpair_guard": AUDIT_ROOT
    / "v103_phase6d_f2_skeleton_affinity_merge_phase9n_suppS1_d4rt48mix_s5repair_r4_directpair_guard",
    "d4rt_phase2_scene0011": AUDIT_ROOT
    / "v103_phase2_stratified_q5c_objlike16384_scene0011_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8",
    "d4rt_phase2_scene0050": AUDIT_ROOT
    / "v103_phase2_stratified_q5c_objlike16384_scene0050_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8",
    "radio_mask_features_scene0011": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0011",
    "radio_mask_features_scene0050": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0050",
    "dino_mask_features_candidate": AUDIT_ROOT / "v81_dino_feature_json_scene0011_scene0050",
}


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
    return value


def _csv_value(value: Any) -> Any:
    value = _jsonable(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
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
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


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
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fields})


def _sha256_first(path: Path, max_bytes: int = 64 * 1024 * 1024) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    remaining = int(max_bytes)
    with path.open("rb") as f:
        while remaining > 0:
            chunk = f.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _floatish(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _decision(summary: dict[str, Any]) -> str:
    return str(summary.get("decision", summary.get("phase_decision", "")))


def _frame_ids(summary: dict[str, Any]) -> list[int]:
    values = summary.get("frame_ids", [])
    if not isinstance(values, list):
        return []
    out: list[int] = []
    for value in values:
        try:
            out.append(int(value))
        except Exception:
            continue
    return out


def _count_files(path: Path) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    return sum(1 for child in path.iterdir() if child.is_file())


def _artifact_row(
    artifact_id: str,
    file_role: str,
    path: Path,
    required: bool,
    source_phase_id: str = "",
    note: str = "",
    checksum_policy: str = "sha256_first64m_for_files",
) -> dict[str, Any]:
    exists = path.exists()
    return {
        "schema_version": "stream4d_v103_supp_r2_phaseP0_source_artifact_row_v1",
        "phase_id": PHASE_ID,
        "artifact_id": artifact_id,
        "file_role": file_role,
        "path": _rel(path),
        "required_for_phaseP0": bool(required),
        "exists": bool(exists),
        "is_file": bool(path.is_file()),
        "is_dir": bool(path.is_dir()),
        "size_bytes": path.stat().st_size if exists and path.is_file() else "",
        "file_count": _count_files(path) if exists and path.is_dir() else "",
        "sha256_first64m": _sha256_first(path) if exists and path.is_file() else "",
        "checksum_policy": checksum_policy,
        "source_phase_id": source_phase_id,
        "note": note,
    }


def _source_artifact_rows(summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    phase0_summary = summaries["phase0_contract"]
    phase6d_summary = summaries["phase6d_d9_directpair_guard"]
    phase2_11 = summaries["d4rt_phase2_scene0011"]
    phase2_50 = summaries["d4rt_phase2_scene0050"]

    docs = [
        ("supplement_plan_doc", PLAN_DOC, True, "new R2 partial-carrier semantic evidence plan"),
        ("method_doc", METHOD_DOC, True, "base v103 method-thinking doc"),
        ("base_experiment_plan_doc", BASE_PLAN_DOC, True, "base v103 experiment plan"),
        ("canonical_execution_log", EXEC_LOG, True, "canonical execution log"),
        ("canonical_retrospective_log", RETRO_LOG, True, "canonical result retrospective"),
        ("v65_evaluator", V65_EVALUATOR, True, "formal AP source: SparseSceneIoU/_summarize_iou"),
    ]
    for artifact_id, path, required, note in docs:
        rows.append(_artifact_row(artifact_id, path.name, path, required, "doc_or_evaluator", note))

    rows.extend(
        [
            _artifact_row(
                "phase0_contract",
                "summary.json",
                ROOTS["phase0_contract"] / "summary.json",
                True,
                phase0_summary.get("phase_id", ""),
                "locks v65 metric contract and current strong baseline",
            ),
            _artifact_row(
                "phase0_contract",
                "metric_contract.json",
                ROOTS["phase0_contract"] / "metric_contract.json",
                True,
                phase0_summary.get("phase_id", ""),
                "formal_metric_source_eq_v65 must remain true",
            ),
            _artifact_row(
                "phase0_contract",
                "baseline_metric_rows.csv",
                ROOTS["phase0_contract"] / "baseline_metric_rows.csv",
                True,
                phase0_summary.get("phase_id", ""),
                "current strong local baseline dev/holdout rows",
            ),
            _artifact_row(
                "current_strong_local_baseline",
                "variant_metric_rows.csv",
                ROOTS["current_strong_local_baseline"] / "variant_metric_rows.csv",
                True,
                "v100_phase2c_overlap3_local_repair",
                "baseline metric source referenced by v103 Phase0",
            ),
            _artifact_row(
                "phase6d_d9_directpair_guard",
                "summary.json",
                ROOTS["phase6d_d9_directpair_guard"] / "summary.json",
                True,
                phase6d_summary.get("phase_id", ""),
                "current Phase6d D9 local positive signal root",
            ),
            _artifact_row(
                "phase6d_d9_directpair_guard",
                "merge_metric_rows.csv",
                ROOTS["phase6d_d9_directpair_guard"] / "merge_metric_rows.csv",
                True,
                phase6d_summary.get("phase_id", ""),
                "contains D9/replay/shuffled local metric rows",
            ),
            _artifact_row(
                "phase6d_d9_directpair_guard",
                "gate_rows.csv",
                ROOTS["phase6d_d9_directpair_guard"] / "gate_rows.csv",
                True,
                phase6d_summary.get("phase_id", ""),
                "local gate rows only; not local2history success",
            ),
            _artifact_row(
                "d4rt_provider_48mix",
                "model.yaml",
                D4RT_48MIX_CONFIG,
                True,
                "OpenD4RT_48CLIP_9Mix_NoCropAUG",
                "D4RT 48Mix config used by selected Phase2 logs",
            ),
            _artifact_row(
                "d4rt_provider_48mix",
                "opend4rt.ckpt",
                D4RT_48MIX_CKPT,
                True,
                "OpenD4RT_48CLIP_9Mix_NoCropAUG",
                "D4RT 48Mix checkpoint; huge file locked by size/first64m hash",
            ),
        ]
    )

    for key, summary in [("d4rt_phase2_scene0011", phase2_11), ("d4rt_phase2_scene0050", phase2_50)]:
        root = ROOTS[key]
        outputs = summary.get("outputs", {}) if isinstance(summary.get("outputs", {}), dict) else {}
        rows.extend(
            [
                _artifact_row(
                    key,
                    "summary.json",
                    root / "summary.json",
                    True,
                    summary.get("phase_id", ""),
                    "selected c0001 D4RT48Mix mask-balanced Phase2 root",
                ),
                _artifact_row(
                    key,
                    "carrier_batch.npz",
                    _project(outputs.get("carrier_batch", root / "carrier_batch.npz")),
                    True,
                    summary.get("phase_id", ""),
                    "D4RT carrier batch consumed by partial-carrier phases",
                ),
                _artifact_row(
                    key,
                    "carrier_sources.npz",
                    _project(outputs.get("carrier_sources", root / "carrier_sources.npz")),
                    True,
                    summary.get("phase_id", ""),
                    "query-source provenance for mask-balanced D4RT carriers",
                ),
                _artifact_row(
                    key,
                    "query_source_count_rows.csv",
                    root / "query_source_count_rows.csv",
                    True,
                    summary.get("phase_id", ""),
                    "query source mix, including mask_balanced_view_probe",
                ),
                _artifact_row(
                    f"cropformer_masks_{summary.get('scene_id', key)}",
                    "mask_root",
                    _project(summary.get("mask_root", "")),
                    True,
                    "cropformer_mask_registry",
                    "CropFormer mask registry used by selected D4RT root",
                ),
            ]
        )

    for key in ["radio_mask_features_scene0011", "radio_mask_features_scene0050"]:
        root = ROOTS[key]
        summary = _read_json(root / "semantic_summary.json")
        rows.extend(
            [
                _artifact_row(key, "semantic_summary.json", root / "semantic_summary.json", True, summary.get("phase", ""), "RADIO pooled mask feature summary"),
                _artifact_row(key, "feature_store_manifest.json", root / "feature_store_manifest.json", True, summary.get("phase", ""), "RADIO feature store manifest"),
                _artifact_row(key, "mask_features.npz", root / "mask_features.npz", True, summary.get("phase", ""), "RADIO mask-pooled sparse feature store"),
                _artifact_row(key, "mask_feature_index.csv", root / "mask_feature_index.csv", True, summary.get("phase", ""), "RADIO mask id lookup table"),
                _artifact_row(key, "mask_feature_rows.csv", root / "mask_feature_rows.csv", True, summary.get("phase", ""), "RADIO mask feature row metadata"),
            ]
        )

    dino_root = ROOTS["dino_mask_features_candidate"]
    dino_summary = _read_json(dino_root / "semantic_summary.json")
    rows.extend(
        [
            _artifact_row(
                "dino_mask_features_candidate",
                "semantic_summary.json",
                dino_root / "semantic_summary.json",
                False,
                dino_summary.get("phase", ""),
                "historical DINO CSV candidate; P1 must adapt/compact before formal E_pool_dino use",
            ),
            _artifact_row(
                "dino_mask_features_candidate",
                "mask_feature_rows.csv",
                dino_root / "mask_feature_rows.csv",
                False,
                dino_summary.get("phase", ""),
                "large JSON-in-CSV DINO feature rows; not the preferred P1 compact store",
            ),
        ]
    )
    rows.append(
        _artifact_row(
            "clip_crop_features",
            "existing_feature_root",
            AUDIT_ROOT / "v103_supp_r2_phaseP1_clip_crop_features",
            False,
            "",
            "expected absent at P0; P1 may build sparse mask-level CLIP or low-resolution compact maps",
        )
    )
    return rows


def _baseline_metric_rows(summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_csv(ROOTS["phase0_contract"] / "baseline_metric_rows.csv"):
        if row.get("baseline_role") != "current_strong_local_baseline":
            continue
        rows.append(
            {
                "schema_version": "stream4d_v103_supp_r2_phaseP0_baseline_metric_row_v1",
                "phase_id": PHASE_ID,
                "metric_row_type": "current_strong_local_baseline",
                "source_phase_id": row.get("source_phase_id", ""),
                "source_artifact": row.get("source_artifact", ""),
                "variant_id": row.get("variant_id", ""),
                "dataset_split": row.get("dataset_split", ""),
                "metric_scope": "local_window_and_fragmented_scene_baseline",
                "MV_AP_window": row.get("MV_AP_window", ""),
                "MV_AP50_window": row.get("MV_AP50_window", ""),
                "MV_AP25_window": row.get("MV_AP25_window", ""),
                "ScoreFreeMatch50_window": row.get("ScoreFreeMatch50_window", ""),
                "MV_AP_scene": row.get("fragmented_MV_AP_scene", ""),
                "MV_AP50_scene": row.get("fragmented_MV_AP50_scene", ""),
                "uses_gt_for_prediction": row.get("uses_gt_for_prediction", ""),
                "uses_future": row.get("uses_future", ""),
                "metric_source": row.get("metric_source", ""),
                "note": "Copied from v103 Phase0 contract; local metric lock remains MV_AP_window.",
            }
        )

    phase6d = summaries["phase6d_d9_directpair_guard"]
    metric_rows = _read_csv(ROOTS["phase6d_d9_directpair_guard"] / "merge_metric_rows.csv")
    wanted = {
        phase6d.get("replay_variant_id", "D0_f2_original_replay"): "phase6d_replay_baseline",
        phase6d.get("best_real_variant_id", "D9_affinity_merge_tau065_top1_broad_support_veto"): "phase6d_current_d9_local_signal",
        phase6d.get("best_shuffled_variant_id", ""): "phase6d_best_shuffled_control",
    }
    for row in metric_rows:
        variant_id = row.get("variant_id", "")
        if variant_id not in wanted:
            continue
        rows.append(
            {
                "schema_version": "stream4d_v103_supp_r2_phaseP0_baseline_metric_row_v1",
                "phase_id": PHASE_ID,
                "metric_row_type": wanted[variant_id],
                "source_phase_id": row.get("phase_id", ""),
                "source_artifact": _rel(ROOTS["phase6d_d9_directpair_guard"] / "merge_metric_rows.csv"),
                "variant_id": variant_id,
                "dataset_split": row.get("dataset_split", ""),
                "chunk_id": row.get("chunk_id", ""),
                "metric_scope": row.get("metric_scope", ""),
                "MV_AP_window": row.get("MV_AP_window", ""),
                "MV_AP50_window": row.get("MV_AP50_window", ""),
                "MV_AP25_window": row.get("MV_AP25_window", ""),
                "ScoreFreeMatch50_window": row.get("ScoreFreeMatch50_window", ""),
                "same_frame_collision_count": row.get("same_frame_collision_count", ""),
                "pixel_collision_rate": row.get("pixel_collision_rate", ""),
                "missing_mask_raster_count": row.get("missing_mask_raster_count", ""),
                "uses_gt_for_prediction": row.get("uses_gt_for_prediction", ""),
                "uses_future": row.get("uses_future", ""),
                "metric_source": row.get("iou_backend", ""),
                "note": "Phase6d local c0001 fact only; not a scene/local2history success claim.",
            }
        )
    return rows


def _scope_contract_rows(summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    contract = _read_json(ROOTS["phase0_contract"] / "metric_contract.json")
    phase2_11 = summaries["d4rt_phase2_scene0011"]
    phase2_50 = summaries["d4rt_phase2_scene0050"]
    frames11 = _frame_ids(phase2_11)
    frames50 = _frame_ids(phase2_50)
    phase6d = summaries["phase6d_d9_directpair_guard"]
    rows = [
        {
            "schema_version": "stream4d_v103_supp_r2_phaseP0_scope_contract_row_v1",
            "phase_id": PHASE_ID,
            "contract_id": "formal_metric_contract",
            "pass": bool(contract.get("formal_metric_source_eq_v65")),
            "observed": {
                "canonical_evaluator_path": contract.get("canonical_evaluator_path", ""),
                "formal_metric_source_eq_v65": contract.get("formal_metric_source_eq_v65", ""),
                "has_sparse_scene_iou": contract.get("has_sparse_scene_iou", ""),
                "has_summarize_iou": contract.get("has_summarize_iou", ""),
                "AP_thresholds_actual": contract.get("AP_thresholds_actual", ""),
            },
            "required": "v65 SparseSceneIoU/_summarize_iou; MV_AP_window local, MV_AP_scene scene/local2history",
            "note": "No alternative AP source is allowed for formal claims.",
        },
        {
            "schema_version": "stream4d_v103_supp_r2_phaseP0_scope_contract_row_v1",
            "phase_id": PHASE_ID,
            "contract_id": "current_chunk_universe",
            "pass": frames11 == frames50 and len(frames11) == 32,
            "observed": {
                "scene0011_frame_min": min(frames11) if frames11 else "",
                "scene0011_frame_max": max(frames11) if frames11 else "",
                "scene0050_frame_min": min(frames50) if frames50 else "",
                "scene0050_frame_max": max(frames50) if frames50 else "",
                "frame_count_scene0011": len(frames11),
                "frame_count_scene0050": len(frames50),
            },
            "required": "c0001, chunk_index=1, same 32 stride-5 frame ids across dev scenes",
            "note": "Current supplement scope is the c0001 dev subset, not full dev/holdout.",
        },
        {
            "schema_version": "stream4d_v103_supp_r2_phaseP0_scope_contract_row_v1",
            "phase_id": PHASE_ID,
            "contract_id": "method_scope_parameters",
            "pass": True,
            "observed": {
                "current_chunk_id": "c0001",
                "method_chunk_size": contract.get("method_chunk_size", 32),
                "frame_stride": contract.get("frame_stride", 5),
                "overlap": contract.get("chunk_overlap", 3),
                "dataset_split": "dev",
                "scene_ids": ["scene0011_00", "scene0050_00"],
            },
            "required": "method_chunk_size=32, frame_stride=5, overlap=3",
            "note": "P0 only locks the scope; it does not generate predictions.",
        },
        {
            "schema_version": "stream4d_v103_supp_r2_phaseP0_scope_contract_row_v1",
            "phase_id": PHASE_ID,
            "contract_id": "d4rt_query_variant",
            "pass": True,
            "observed": {
                "d4rt_provider_id": "OpenD4RT_48CLIP_9Mix_NoCropAUG",
                "query_variant_id": "q5c_objlike16384_cap24576_qchunk16384_d4rt48mix_maskbalanced8",
                "query_chunk_size_scene0011": phase2_11.get("d4rt_infer_diagnostics", {}).get("query_chunk_size", ""),
                "query_chunk_size_scene0050": phase2_50.get("d4rt_infer_diagnostics", {}).get("query_chunk_size", ""),
                "query_count_per_frame_scene0011": phase2_11.get("query_count_per_frame", ""),
                "query_count_per_frame_scene0050": phase2_50.get("query_count_per_frame", ""),
                "mask_balanced_points_per_mask": phase2_11.get("query_generation_policy", {}).get("mask_balanced_view_probe_points_per_mask", ""),
                "max_query_count_per_frame_cap": phase2_11.get("max_query_count_per_frame_cap", ""),
            },
            "required": "D4RT48Mix, qchunk16384, cap24576, mask_balanced_view_probe_points_per_mask=8",
            "note": "This records the user's 16384/frame direction and mask-balanced query branch.",
        },
        {
            "schema_version": "stream4d_v103_supp_r2_phaseP0_scope_contract_row_v1",
            "phase_id": PHASE_ID,
            "contract_id": "phase6d_local_signal_boundary",
            "pass": _decision(phase6d) == "PASS_PHASE6D_S3_STYLE_LOCAL_GATE",
            "observed": {
                "phase6d_root": _rel(ROOTS["phase6d_d9_directpair_guard"]),
                "decision": _decision(phase6d),
                "best_real_variant_id": phase6d.get("best_real_variant_id", ""),
                "best_real_MV_AP_window": phase6d.get("best_real_MV_AP_window", ""),
                "best_real_minus_replay_MV_AP_window": phase6d.get("best_real_minus_replay_MV_AP_window", ""),
                "best_real_minus_best_shuffled_MV_AP_window": phase6d.get("best_real_minus_best_shuffled_MV_AP_window", ""),
            },
            "required": "Phase6d D9 local gate fact may seed local work; no local2history/full-dev claim",
            "note": "This is a local positive signal boundary, not a completed v103 success.",
        },
        {
            "schema_version": "stream4d_v103_supp_r2_phaseP0_scope_contract_row_v1",
            "phase_id": PHASE_ID,
            "contract_id": "clip_backfill_storage_policy",
            "pass": True,
            "observed": {
                "sparse_mask_level_clip_required": True,
                "high_dim_dense_pixel_clip_map_allowed": False,
                "low_resolution_clip_feature_map_allowed_by_user_override": True,
                "default_compact_dim": 64,
                "max_default_compact_dim_without_new_calibration": 128,
            },
            "required": "No dense HxWx512/768 CLIP maps. Low-resolution compact maps may be tested if resolution stays small.",
            "note": "Records the user's relaxation: low-res feature maps are allowed, high-resolution dense CLIP maps are not.",
        },
        {
            "schema_version": "stream4d_v103_supp_r2_phaseP0_scope_contract_row_v1",
            "phase_id": PHASE_ID,
            "contract_id": "causality_contract",
            "pass": True,
            "observed": {"uses_gt_for_prediction": False, "uses_future": False, "uses_gt_for_eval_allowed": True},
            "required": "no GT/future at prediction/query/feature construction time; v65 may use GT only for evaluation",
            "note": "P0 and P1 remain read-only or feature-construction phases, not AP-tuned prediction phases.",
        },
    ]
    return rows


def _semantic_source_availability_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene_id, key in [
        ("scene0011_00", "radio_mask_features_scene0011"),
        ("scene0050_00", "radio_mask_features_scene0050"),
    ]:
        root = ROOTS[key]
        manifest = _read_json(root / "feature_store_manifest.json")
        summary = _read_json(root / "semantic_summary.json")
        store = root / "mask_features.npz"
        index = root / "mask_feature_index.csv"
        ready = (
            store.exists()
            and index.exists()
            and bool(summary.get("radio_available", False))
            and _floatish(summary.get("semantic_feature_success_rate", 0.0)) >= 0.95
            and _floatish(summary.get("semantic_nan_rate", 1.0), 1.0) == 0.0
        )
        rows.append(
            {
                "schema_version": "stream4d_v103_supp_r2_phaseP0_semantic_source_availability_row_v1",
                "phase_id": PHASE_ID,
                "semantic_source_id": "E_pool_radio",
                "scene_id": scene_id,
                "artifact_root": _rel(root),
                "available_for_phaseP1": bool(ready),
                "feature_storage": "mask_features.npz + mask_feature_index.csv",
                "feature_dim": manifest.get("feature_dim", ""),
                "feature_dtype": manifest.get("feature_dtype", ""),
                "row_count": manifest.get("row_count", ""),
                "feature_success_rate": summary.get("semantic_feature_success_rate", ""),
                "semantic_nan_rate": summary.get("semantic_nan_rate", ""),
                "uses_gt_for_prediction": manifest.get("metadata", {}).get("uses_gt_for_prediction", summary.get("uses_gt_for_prediction", "")),
                "uses_future": manifest.get("metadata", {}).get("uses_future", summary.get("uses_future", "")),
                "availability_class": "ready_non_clip_sparse_mask_pooled",
                "note": "Formal non-CLIP semantic source for P1.",
            }
        )
        rows.append(
            {
                "schema_version": "stream4d_v103_supp_r2_phaseP0_semantic_source_availability_row_v1",
                "phase_id": PHASE_ID,
                "semantic_source_id": "E_raw_radio",
                "scene_id": scene_id,
                "artifact_root": _rel(root),
                "available_for_phaseP1": False,
                "feature_storage": "not_found_as_raw_point_or_dense_feature_map_for_current_P0",
                "feature_dim": "",
                "feature_dtype": "",
                "row_count": "",
                "feature_success_rate": "",
                "semantic_nan_rate": "",
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "availability_class": "missing_current_scope",
                "note": "Existing RADIO artifact is mask-pooled; raw point-sampling source must be built or explicitly skipped in P1.",
            }
        )

    dino_root = ROOTS["dino_mask_features_candidate"]
    dino_summary = _read_json(dino_root / "semantic_summary.json")
    rows.append(
        {
            "schema_version": "stream4d_v103_supp_r2_phaseP0_semantic_source_availability_row_v1",
            "phase_id": PHASE_ID,
            "semantic_source_id": "E_pool_dino",
            "scene_id": "scene0011_00;scene0050_00",
            "artifact_root": _rel(dino_root),
            "available_for_phaseP1": False,
            "candidate_artifact_exists": bool((dino_root / "mask_feature_rows.csv").exists()),
            "feature_storage": "large mask_feature_rows.csv with embedded JSON feature vectors",
            "feature_dim": dino_summary.get("key_metrics", {}).get("feature_dim", 384),
            "feature_dtype": "csv_json_float",
            "row_count": dino_summary.get("key_metrics", {}).get("mask_feature_row_count", ""),
            "feature_success_rate": dino_summary.get("key_metrics", {}).get("semantic_feature_success_rate", ""),
            "semantic_nan_rate": dino_summary.get("key_metrics", {}).get("semantic_nan_rate", ""),
            "uses_gt_for_prediction": False,
            "uses_future": False,
            "availability_class": "candidate_needs_p1_compaction",
            "note": "Historical DINO is useful evidence but not yet the efficient sparse/compact P1 store.",
        }
    )
    for source_id in ["E_raw_dino", "E_clip_crop_b16", "E_consensus"]:
        rows.append(
            {
                "schema_version": "stream4d_v103_supp_r2_phaseP0_semantic_source_availability_row_v1",
                "phase_id": PHASE_ID,
                "semantic_source_id": source_id,
                "scene_id": "scene0011_00;scene0050_00",
                "artifact_root": "",
                "available_for_phaseP1": False,
                "candidate_artifact_exists": False,
                "feature_storage": "not_built_at_P0",
                "feature_dim": "",
                "feature_dtype": "",
                "row_count": "",
                "feature_success_rate": "",
                "semantic_nan_rate": "",
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "availability_class": "to_build_in_phaseP1",
                "note": "Allowed missing at P0; P1 decides whether to build/enable it.",
            }
        )
    return rows


def _gate_row(gate_id: str, passed: bool, observed: Any, required: Any, repair_direction: str) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v103_supp_r2_phaseP0_failure_row_v1",
        "phase_id": PHASE_ID,
        "gate_id": gate_id,
        "pass": bool(passed),
        "observed": observed,
        "required": required,
        "repair_direction": repair_direction,
    }


def _build_gates(
    summaries: dict[str, dict[str, Any]],
    source_rows: list[dict[str, Any]],
    semantic_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    contract = _read_json(ROOTS["phase0_contract"] / "metric_contract.json")
    phase6d = summaries["phase6d_d9_directpair_guard"]
    phase2_11 = summaries["d4rt_phase2_scene0011"]
    phase2_50 = summaries["d4rt_phase2_scene0050"]

    phase6d_artifact_available = (
        (ROOTS["phase6d_d9_directpair_guard"] / "summary.json").exists()
        and (ROOTS["phase6d_d9_directpair_guard"] / "merge_metric_rows.csv").exists()
        and _decision(phase6d) == "PASS_PHASE6D_S3_STYLE_LOCAL_GATE"
        and phase6d.get("best_real_variant_id", "") == "D9_affinity_merge_tau065_top1_broad_support_veto"
    )
    d4rt_carrier_batch_available = all(
        (ROOTS[key] / "carrier_batch.npz").exists()
        and (ROOTS[key] / "carrier_sources.npz").exists()
        and int(summaries[key].get("failure_count", 999)) == 0
        for key in ["d4rt_phase2_scene0011", "d4rt_phase2_scene0050"]
    )
    cropformer_masks_available = bool(_project(phase2_11.get("mask_root", "")).exists() and _project(phase2_50.get("mask_root", "")).exists())
    non_clip_ready = [
        row
        for row in semantic_rows
        if row.get("semantic_source_id", "").lower() != "e_clip_crop_b16" and row.get("available_for_phaseP1") is True
    ]
    required_missing = [
        row.get("path", "")
        for row in source_rows
        if row.get("required_for_phaseP0") is True and row.get("exists") is not True
    ]
    truth_summaries = [summaries["phase0_contract"], phase6d, phase2_11, phase2_50]
    uses_gt_for_prediction = any(_boolish(summary.get("uses_gt_for_prediction", False)) for summary in truth_summaries)
    uses_future = any(_boolish(summary.get("uses_future", False)) for summary in truth_summaries)
    uses_gt_for_prediction = uses_gt_for_prediction or any(_boolish(row.get("uses_gt_for_prediction", False)) for row in semantic_rows)
    uses_future = uses_future or any(_boolish(row.get("uses_future", False)) for row in semantic_rows)

    return [
        _gate_row(
            "all_required_source_artifacts_exist",
            not required_missing,
            required_missing,
            "[]",
            "Fix artifact roots or rerun fact-lock only; do not generate method predictions to bypass P0.",
        ),
        _gate_row(
            "formal_metric_source_eq_v65",
            bool(contract.get("formal_metric_source_eq_v65")),
            contract.get("formal_metric_source_eq_v65", ""),
            "true",
            "Repair v103 Phase0 metric contract before P1.",
        ),
        _gate_row(
            "phase6d_artifact_available",
            phase6d_artifact_available,
            {"decision": _decision(phase6d), "best_real_variant_id": phase6d.get("best_real_variant_id", "")},
            "PASS_PHASE6D_S3_STYLE_LOCAL_GATE with D9 variant",
            "Re-bind the current Phase6d direct-pair guard root or regenerate only the missing fact artifact.",
        ),
        _gate_row(
            "d4rt_carrier_batch_available",
            d4rt_carrier_batch_available,
            {
                "scene0011_failure_count": phase2_11.get("failure_count", ""),
                "scene0050_failure_count": phase2_50.get("failure_count", ""),
                "scene0011_carrier_batch": _rel(ROOTS["d4rt_phase2_scene0011"] / "carrier_batch.npz"),
                "scene0050_carrier_batch": _rel(ROOTS["d4rt_phase2_scene0050"] / "carrier_batch.npz"),
            },
            "both selected D4RT48Mix scene roots have carrier_batch/carrier_sources and failure_count=0",
            "Repair selected D4RT root/cache binding before partial-carrier work.",
        ),
        _gate_row(
            "cropformer_masks_available",
            cropformer_masks_available,
            {"scene0011_mask_root": phase2_11.get("mask_root", ""), "scene0050_mask_root": phase2_50.get("mask_root", "")},
            "both selected CropFormer mask roots exist",
            "Repair mask registry root before building semantic/mask incidence features.",
        ),
        _gate_row(
            "at_least_one_non_clip_semantic_source_available",
            bool(non_clip_ready),
            [row.get("semantic_source_id", "") + ":" + row.get("scene_id", "") for row in non_clip_ready],
            "one or more non-CLIP semantic sources ready for P1",
            "Build or re-bind RADIO/DINO sparse semantic feature artifacts before P1.",
        ),
        _gate_row(
            "uses_gt_for_prediction_false",
            not uses_gt_for_prediction,
            uses_gt_for_prediction,
            "false",
            "Remove prediction/query/semantic feature GT dependency before continuing.",
        ),
        _gate_row(
            "uses_future_false",
            not uses_future,
            uses_future,
            "false",
            "Remove future-frame dependency before continuing.",
        ),
    ]


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _project(args.output_root)
    summaries = {key: _read_json(root / "summary.json") for key, root in ROOTS.items()}
    summaries["radio_mask_features_scene0011"] = _read_json(ROOTS["radio_mask_features_scene0011"] / "semantic_summary.json")
    summaries["radio_mask_features_scene0050"] = _read_json(ROOTS["radio_mask_features_scene0050"] / "semantic_summary.json")

    source_rows = _source_artifact_rows(summaries)
    baseline_rows = _baseline_metric_rows(summaries)
    scope_rows = _scope_contract_rows(summaries)
    semantic_rows = _semantic_source_availability_rows()
    gates = _build_gates(summaries, source_rows, semantic_rows)
    failure_rows = [row for row in gates if not row["pass"]]
    phaseP0_pass = len(failure_rows) == 0

    output_root.mkdir(parents=True, exist_ok=True)
    source_csv = output_root / "source_artifact_rows.csv"
    baseline_csv = output_root / "baseline_metric_rows.csv"
    scope_csv = output_root / "scope_contract_rows.csv"
    semantic_csv = output_root / "semantic_source_availability_rows.csv"
    failure_csv = output_root / "failure_rows.csv"
    summary_path = output_root / "summary.json"

    _write_csv(source_csv, source_rows)
    _write_csv(baseline_csv, baseline_rows)
    _write_csv(scope_csv, scope_rows)
    _write_csv(semantic_csv, semantic_rows)
    _write_csv(failure_csv, failure_rows)

    contract = _read_json(ROOTS["phase0_contract"] / "metric_contract.json")
    phase6d = summaries["phase6d_d9_directpair_guard"]
    phase2_11 = summaries["d4rt_phase2_scene0011"]
    phase2_50 = summaries["d4rt_phase2_scene0050"]
    ready_sources = sorted(
        {row["semantic_source_id"] for row in semantic_rows if row.get("available_for_phaseP1") is True}
    )
    summary = {
        "schema_version": "stream4d_v103_supp_r2_phaseP0_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "decision": "PASS_ENTER_PHASEP1_SEMANTIC_FEATURE_CONSTRUCTION" if phaseP0_pass else "NO_GO_REPAIR_PHASEP0_FACT_LOCK",
        "phaseP0_pass": bool(phaseP0_pass),
        "failure_count": len(failure_rows),
        "gate_rows_inline": gates,
        "formal_metric_source_eq_v65": bool(contract.get("formal_metric_source_eq_v65")),
        "phase6d_artifact_available": any(row["gate_id"] == "phase6d_artifact_available" and row["pass"] for row in gates),
        "d4rt_carrier_batch_available": any(row["gate_id"] == "d4rt_carrier_batch_available" and row["pass"] for row in gates),
        "cropformer_masks_available": any(row["gate_id"] == "cropformer_masks_available" and row["pass"] for row in gates),
        "at_least_one_non_clip_semantic_source_available": any(
            row["gate_id"] == "at_least_one_non_clip_semantic_source_available" and row["pass"] for row in gates
        ),
        "uses_gt_for_prediction": not any(row["gate_id"] == "uses_gt_for_prediction_false" and row["pass"] for row in gates),
        "uses_future": not any(row["gate_id"] == "uses_future_false" and row["pass"] for row in gates),
        "current_baseline_artifact_root": _rel(ROOTS["current_strong_local_baseline"]),
        "current_phase6d_d9_artifact_root": _rel(ROOTS["phase6d_d9_directpair_guard"]),
        "current_phase6d_d9_variant_id": phase6d.get("best_real_variant_id", ""),
        "current_phase6d_d9_MV_AP_window": phase6d.get("best_real_MV_AP_window", ""),
        "current_phase6d_d9_MV_AP50_window": phase6d.get("best_real_MV_AP50_window", ""),
        "d4rt_provider_checkpoint": _rel(D4RT_48MIX_CKPT),
        "d4rt_provider_config": _rel(D4RT_48MIX_CONFIG),
        "d4rt_query_variant_id": "q5c_objlike16384_cap24576_qchunk16384_d4rt48mix_maskbalanced8",
        "d4rt_scene0011_carrier_batch": _rel(ROOTS["d4rt_phase2_scene0011"] / "carrier_batch.npz"),
        "d4rt_scene0050_carrier_batch": _rel(ROOTS["d4rt_phase2_scene0050"] / "carrier_batch.npz"),
        "cropformer_mask_roots": {
            "scene0011_00": phase2_11.get("mask_root", ""),
            "scene0050_00": phase2_50.get("mask_root", ""),
        },
        "semantic_sources_ready_for_P1": ready_sources,
        "semantic_sources_candidate_needing_P1_adaptation": ["E_pool_dino"],
        "semantic_sources_to_build_in_P1": ["E_raw_radio", "E_raw_dino", "E_clip_crop_b16", "E_consensus"],
        "clip_storage_policy": {
            "mask_level_sparse_table_default": True,
            "high_dim_dense_pixel_map_allowed": False,
            "low_resolution_compact_feature_map_allowed": True,
            "compact_dim_default": 64,
            "compact_dim_max_without_new_calibration": 128,
        },
        "current_chunk_id": "c0001",
        "frame_stride": contract.get("frame_stride", 5),
        "method_chunk_size": contract.get("method_chunk_size", 32),
        "overlap": contract.get("chunk_overlap", 3),
        "metric_evaluator_path": _rel(V65_EVALUATOR),
        "outputs": {
            "summary": _rel(summary_path),
            "source_artifact_rows": _rel(source_csv),
            "baseline_metric_rows": _rel(baseline_csv),
            "scope_contract_rows": _rel(scope_csv),
            "semantic_source_availability_rows": _rel(semantic_csv),
            "failure_rows": _rel(failure_csv),
        },
        "truthfulness_note": (
            "Phase P0 is read-only and does not generate predictions, tune thresholds, compute new AP, "
            "or claim v103 completion. It only decides whether P1 semantic feature construction may start."
        ),
    }
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    summary = build(args)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True, ensure_ascii=False))
    raise SystemExit(0 if summary["phaseP0_pass"] else 2)


if __name__ == "__main__":
    main()
