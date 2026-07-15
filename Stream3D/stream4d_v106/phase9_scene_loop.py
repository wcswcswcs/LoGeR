from __future__ import annotations

import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from .artifacts import sha256_file, write_json
from .config import Phase3HandoffConfig, Phase5RepairBirthDeferConfig, Phase9SceneLoopSmokeConfig
from .phase5_repair_birth_defer import (
    _build_variant_birth_records,
    _copy_or_merge_row,
    _load_label,
    _load_mask,
    _mask_for_obj,
    _read_json,
    _summary_label_map,
)
from .phase3_handoff import run_phase3_handoff_smoke


PYTHON = Path("/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python")
H2_VARIANT = "H2_best_plus_one_correction"
H4_VARIANT = "H4_endpoint_drift_correction"
SUPPORTED_REPLAY_VARIANTS = {H2_VARIANT, H4_VARIANT}
SUPPORTED_PHASE5_VARIANTS = {"R1_repair_vs_birth", "R2_repair_birth_defer"}


def _resolve(repo_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _rel(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _handoff_config(
    config: Phase9SceneLoopSmokeConfig,
    replay_summary: str = "",
    replay_variant: str | None = None,
) -> Phase3HandoffConfig:
    variant = replay_variant or str(config.handoff_replay_variant)
    h2_replay_summary = replay_summary if variant == H2_VARIANT else config.h2_replay_summary
    h4_replay_summary = replay_summary if variant == H4_VARIANT else config.h4_replay_summary
    return Phase3HandoffConfig(
        enabled=True,
        scene_id=config.scene_id,
        c0_summary=config.c0_summary,
        c0_labels_dir=config.c0_labels_dir,
        c0_chunk_index=int(config.c0_chunk_index),
        c1_chunk_index=int(config.c1_chunk_index),
        c0_frame_start=int(config.c0_frame_start),
        c1_frame_start=int(config.c1_frame_start),
        frame_stride=int(config.frame_stride),
        frame_count=int(config.frame_count),
        overlap=int(config.overlap),
        reference_c1_summary=config.reference_c1_summary,
        h2_replay_summary=h2_replay_summary,
        h4_replay_summary=h4_replay_summary,
        min_ccoc=float(config.min_ccoc),
        min_hir=float(config.min_hir),
        min_hcr=float(config.min_hcr),
        max_cfr=float(config.max_cfr),
        max_cmr=float(config.max_cmr),
        max_bfmr=float(config.max_bfmr),
        fragment_overlap_fraction_threshold=float(config.fragment_overlap_fraction_threshold),
        merge_overlap_fraction_threshold=float(config.merge_overlap_fraction_threshold),
        endpoint_drift_area_ratio_min=float(config.endpoint_drift_area_ratio_min),
        endpoint_drift_area_ratio_max=float(config.endpoint_drift_area_ratio_max),
    )


def _replay_variant(config: Phase9SceneLoopSmokeConfig) -> str:
    variant = str(config.handoff_replay_variant or H4_VARIANT)
    if variant not in SUPPORTED_REPLAY_VARIANTS:
        raise ValueError(f"unsupported handoff_replay_variant={variant}; expected one of {sorted(SUPPORTED_REPLAY_VARIANTS)}")
    return variant


def _boundary_name(config: Phase9SceneLoopSmokeConfig) -> str:
    return f"boundary_c{int(config.c0_chunk_index)}_c{int(config.c1_chunk_index)}"


def _boundary_dir(config: Phase9SceneLoopSmokeConfig, output_dir: Path) -> Path:
    return output_dir / _boundary_name(config)


def _birth_records_path(config: Phase9SceneLoopSmokeConfig, output_dir: Path, variant: str) -> Path:
    return _boundary_dir(config, output_dir) / variant / "birth_records.json"


def _replay_output_root(config: Phase9SceneLoopSmokeConfig, output_dir: Path, variant: str) -> Path:
    if config.replay_output_root:
        return Path(config.replay_output_root)
    return _boundary_dir(config, output_dir) / f"{variant}_phase5_replay"


def _replay_summary_path(output_root: Path) -> Path:
    return output_root / "phase5_frozen_birth_replay_summary.json"


def _phase5_variant(config: Phase9SceneLoopSmokeConfig) -> str:
    variant = str(config.phase5_variant or "R1_repair_vs_birth")
    if variant not in SUPPORTED_PHASE5_VARIANTS:
        raise ValueError(f"unsupported phase5_variant={variant}; expected one of {sorted(SUPPORTED_PHASE5_VARIANTS)}")
    return variant


def _phase5_reference_summary(config: Phase9SceneLoopSmokeConfig) -> str:
    return str(config.phase5_reference_summary or config.reference_c1_summary)


def _build_phase5_schedule(
    *,
    repo_root: Path,
    config: Phase9SceneLoopSmokeConfig,
    output_dir: Path,
    inherited_birth_records: Path,
    inherited_replay_summary: Path,
) -> Dict[str, Any]:
    variant = _phase5_variant(config)
    reference_birth_records = _resolve(repo_root, config.phase5_reference_birth_records)
    reference_summary = _resolve(repo_root, _phase5_reference_summary(config))
    phase5_dir = output_dir / f"{_replay_variant(config)}_{variant}_schedule"
    phase5_dir.mkdir(parents=True, exist_ok=True)
    phase5_config = Phase5RepairBirthDeferConfig(
        enabled=True,
        scene_id=config.scene_id,
        frame_start=int(config.c1_frame_start),
        frame_stride=int(config.frame_stride),
        frame_count=int(config.frame_count),
        selected_chunk_indices=str(config.phase5_selected_chunk_indices),
        inherited_birth_records=_rel(repo_root, inherited_birth_records),
        inherited_replay_summary=_rel(repo_root, inherited_replay_summary),
        reference_birth_records=_rel(repo_root, reference_birth_records),
        reference_summary=_rel(repo_root, reference_summary),
        min_area=int(config.phase5_min_area),
        repair_overlap_coeff=float(config.phase5_repair_overlap_coeff),
        duplicate_suppress_overlap_coeff=float(config.phase5_duplicate_suppress_overlap_coeff),
        duplicate_suppress_area_ratio_min=float(config.phase5_duplicate_suppress_area_ratio_min),
        birth_max_overlap_coeff=float(config.phase5_birth_max_overlap_coeff),
        min_persistence_frames=int(config.phase5_min_persistence_frames),
        large_persistent_area=int(config.phase5_large_persistent_area),
        duplicate_window_frames=int(config.duplicate_window_frames),
        duplicate_overlap_threshold=float(config.duplicate_overlap_threshold),
        fragment_overlap_fraction_threshold=float(config.fragment_overlap_fraction_threshold),
        merge_overlap_fraction_threshold=float(config.merge_overlap_fraction_threshold),
    )
    inherited_payload = _read_json(inherited_birth_records)
    reference_payload = _read_json(reference_birth_records)
    inherited_labels = _summary_label_map(repo_root, inherited_replay_summary)
    reference_labels = _summary_label_map(repo_root, reference_summary)
    audit = _build_variant_birth_records(
        repo_root=repo_root,
        output_dir=phase5_dir,
        config=phase5_config,
        variant=variant,
        inherited_payload=inherited_payload,
        reference_payload=reference_payload,
        inherited_labels=inherited_labels,
        reference_labels=reference_labels,
    )
    if bool(config.phase5_defer_new_births_until_non_overlap):
        audit = _defer_phase5_overlap_births(
            repo_root=repo_root,
            config=config,
            phase5_dir=phase5_dir,
            variant=variant,
            audit=audit,
            reference_labels=reference_labels,
        )
    if bool(config.phase5_preserve_overlap_inherited_masks):
        audit = _guard_phase5_overlap_inherited_masks(
            repo_root=repo_root,
            config=config,
            phase5_dir=phase5_dir,
            variant=variant,
            audit=audit,
            inherited_birth_records=inherited_birth_records,
        )
    summary = {
        "schema_version": "stream4d_v106_phase9_scene_loop_phase5_schedule_v1",
        "enabled": True,
        "variant": variant,
        "handoff_variant": _replay_variant(config),
        "inherited_birth_records": _rel(repo_root, inherited_birth_records),
        "inherited_birth_records_sha256": sha256_file(inherited_birth_records),
        "inherited_replay_summary": _rel(repo_root, inherited_replay_summary),
        "inherited_replay_summary_sha256": sha256_file(inherited_replay_summary),
        "reference_birth_records": _rel(repo_root, reference_birth_records),
        "reference_birth_records_sha256": sha256_file(reference_birth_records),
        "reference_summary": _rel(repo_root, reference_summary),
        "reference_summary_sha256": sha256_file(reference_summary),
        "selected_chunk_indices": str(config.phase5_selected_chunk_indices or ""),
        "schedule_audit": audit,
        "control_limitation": (
            "Uses frozen SAM2 baseline-x reference birth records as a schedule proxy; "
            "valid for diagnosing Phase9 missing gap-birth integration, not final cold-run exact gap detection."
        ),
    }
    summary_path = phase5_dir / "phase5_schedule_summary.json"
    write_json(summary_path, summary)
    summary["summary_path"] = _rel(repo_root, summary_path)
    summary["summary_sha256"] = sha256_file(summary_path)
    return summary


def _source_area_by_obj(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    by_obj: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        obj_id = int(row["obj_id"])
        area = int(row.get("mask_area", 0) or 0)
        entry = by_obj.setdefault(obj_id, {"source_max_area": 0, "source_rows": []})
        entry["source_max_area"] = max(int(entry["source_max_area"]), area)
        entry["source_rows"].append(
            {
                "frame_id": int(row.get("frame_id", -1)),
                "chunk_frame_index": int(row.get("chunk_frame_index", -1)),
                "source_overlap_index": row.get("source_overlap_index"),
                "mask_area": int(area),
            }
        )
    return by_obj


def _filter_handoff_birth_records_by_drift(
    *,
    repo_root: Path,
    config: Phase9SceneLoopSmokeConfig,
    output_dir: Path,
    birth_records_path: Path,
    handoff_replay_summary: Path,
) -> Dict[str, Any]:
    filter_dir = output_dir / f"{_replay_variant(config)}_handoff_drift_filter"
    labels_dir = filter_dir / "synthetic_filtered_labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    birth_payload = _read_json(birth_records_path)
    replay_payload = _read_json(handoff_replay_summary)
    rows = [dict(row) for row in birth_payload.get("rows", [])]
    frame_ids = [int(v) for v in birth_payload.get("frame_ids", [])]
    if not frame_ids:
        frame_ids = [int(v) for v in replay_payload.get("frame_ids", [])]
    probe_start = int(config.overlap)
    probe_end = probe_start + int(config.handoff_drift_probe_frame_count)
    probe_frame_ids = frame_ids[probe_start:probe_end]
    if not probe_frame_ids:
        raise ValueError("handoff drift filter selected no probe frames")

    label_paths = _summary_label_map(repo_root, handoff_replay_summary)
    missing_probe = [int(frame_id) for frame_id in probe_frame_ids if int(frame_id) not in label_paths]
    if missing_probe:
        raise FileNotFoundError(f"handoff drift filter missing probe labels: {missing_probe}")

    source_by_obj = _source_area_by_obj(rows)
    probe_labels = {int(frame_id): _load_label(label_paths[int(frame_id)]) for frame_id in probe_frame_ids}
    drift_records: List[Dict[str, Any]] = []
    dropped_obj_ids: set[int] = set()
    for obj_id, source_entry in sorted(source_by_obj.items()):
        source_max_area = int(source_entry["source_max_area"])
        probe_areas = {
            int(frame_id): int(np.count_nonzero(label == int(obj_id) + 1))
            for frame_id, label in probe_labels.items()
        }
        max_probe_area = max(probe_areas.values()) if probe_areas else 0
        growth_ratio = float(max_probe_area / max(1, source_max_area))
        dropped = bool(
            max_probe_area >= int(config.handoff_drift_min_probe_area)
            and growth_ratio >= float(config.handoff_drift_growth_threshold)
        )
        if dropped:
            dropped_obj_ids.add(int(obj_id))
        drift_records.append(
            {
                "obj_id": int(obj_id),
                "source_max_area": int(source_max_area),
                "source_rows": source_entry["source_rows"],
                "probe_frame_ids": [int(v) for v in probe_frame_ids],
                "probe_areas": {str(k): int(v) for k, v in sorted(probe_areas.items())},
                "max_probe_area": int(max_probe_area),
                "growth_ratio_vs_source_max": float(growth_ratio),
                "dropped": bool(dropped),
                "reason": "early_nonoverlap_area_growth_exceeds_threshold" if dropped else "kept",
            }
        )

    kept_rows = [row for row in rows if int(row["obj_id"]) not in dropped_obj_ids]
    filtered_payload = dict(birth_payload)
    filtered_payload["schema_version"] = str(
        birth_payload.get("schema_version", "unknown")
    ) + "+handoff_drift_filter_v1"
    filtered_payload["source_birth_records"] = _rel(repo_root, birth_records_path)
    filtered_payload["source_birth_records_sha256"] = sha256_file(birth_records_path)
    filtered_payload["source_handoff_replay_summary"] = _rel(repo_root, handoff_replay_summary)
    filtered_payload["source_handoff_replay_summary_sha256"] = sha256_file(handoff_replay_summary)
    filtered_payload["handoff_drift_filter_policy"] = {
        "overlap": int(config.overlap),
        "probe_frame_count": int(config.handoff_drift_probe_frame_count),
        "probe_frame_ids": [int(v) for v in probe_frame_ids],
        "growth_threshold": float(config.handoff_drift_growth_threshold),
        "min_probe_area": int(config.handoff_drift_min_probe_area),
        "uses_reference_labels": False,
        "uses_ground_truth": False,
        "uses_predicted_handoff_replay_labels": True,
    }
    filtered_payload["handoff_drift_filter_records"] = drift_records
    filtered_payload["handoff_drift_filter_dropped_obj_ids"] = [int(v) for v in sorted(dropped_obj_ids)]
    filtered_payload["handoff_drift_filter_dropped_count"] = int(len(dropped_obj_ids))
    filtered_payload["filtered_from_row_count"] = int(len(rows))
    filtered_payload["row_count"] = int(len(kept_rows))
    filtered_payload["rows"] = kept_rows

    filtered_birth_records_path = filter_dir / "filtered_handoff_birth_records.json"
    write_json(filtered_birth_records_path, filtered_payload)

    synthetic_records = []
    for record in replay_payload.get("records", []):
        frame_id = int(record["frame_id"])
        label = _load_label(_resolve(repo_root, str(record["label_path"]))).copy()
        for obj_id in dropped_obj_ids:
            label[label == int(obj_id) + 1] = 0
        dst = labels_dir / f"frame_{frame_id:06d}.png"
        if not cv2.imwrite(str(dst), label):
            raise IOError(f"failed to write synthetic filtered label: {dst}")
        copied = dict(record)
        copied["label_path"] = _rel(repo_root, dst)
        copied["synthetic_dropped_obj_ids"] = [int(v) for v in sorted(dropped_obj_ids)]
        synthetic_records.append(copied)

    synthetic_summary = dict(replay_payload)
    synthetic_summary["schema_version"] = str(
        replay_payload.get("schema_version", "unknown")
    ) + "+synthetic_handoff_drift_filter_v1"
    synthetic_summary["source_handoff_replay_summary"] = _rel(repo_root, handoff_replay_summary)
    synthetic_summary["source_handoff_replay_summary_sha256"] = sha256_file(handoff_replay_summary)
    synthetic_summary["filtered_handoff_birth_records"] = _rel(repo_root, filtered_birth_records_path)
    synthetic_summary["filtered_handoff_birth_records_sha256"] = sha256_file(filtered_birth_records_path)
    synthetic_summary["synthetic_filtered_label_note"] = (
        "Dropped handoff obj_id labels are set to background for Phase5 overlap "
        "classification only; final outputs must come from a fresh replay."
    )
    synthetic_summary["synthetic_dropped_obj_ids"] = [int(v) for v in sorted(dropped_obj_ids)]
    synthetic_summary["records"] = synthetic_records
    synthetic_summary_path = filter_dir / "synthetic_filtered_handoff_summary.json"
    write_json(synthetic_summary_path, synthetic_summary)

    audit = {
        "enabled": True,
        "birth_records": _rel(repo_root, birth_records_path),
        "birth_records_sha256": sha256_file(birth_records_path),
        "handoff_replay_summary": _rel(repo_root, handoff_replay_summary),
        "handoff_replay_summary_sha256": sha256_file(handoff_replay_summary),
        "filtered_birth_records": _rel(repo_root, filtered_birth_records_path),
        "filtered_birth_records_sha256": sha256_file(filtered_birth_records_path),
        "synthetic_filtered_handoff_summary": _rel(repo_root, synthetic_summary_path),
        "synthetic_filtered_handoff_summary_sha256": sha256_file(synthetic_summary_path),
        "policy": filtered_payload["handoff_drift_filter_policy"],
        "row_count_before": int(len(rows)),
        "row_count_after": int(len(kept_rows)),
        "dropped_obj_ids": [int(v) for v in sorted(dropped_obj_ids)],
        "records": drift_records,
        "uses_reference_labels": False,
        "uses_ground_truth": False,
        "uses_extra_replay": False,
    }
    audit_path = filter_dir / "handoff_drift_filter_audit.json"
    write_json(audit_path, audit)
    audit["audit_path"] = _rel(repo_root, audit_path)
    audit["audit_sha256"] = sha256_file(audit_path)
    return audit


def _defer_phase5_overlap_births(
    *,
    repo_root: Path,
    config: Phase9SceneLoopSmokeConfig,
    phase5_dir: Path,
    variant: str,
    audit: Dict[str, Any],
    reference_labels: Dict[int, Path],
) -> Dict[str, Any]:
    birth_records_path = _resolve(repo_root, audit["birth_records_path"])
    payload = _read_json(birth_records_path)
    frame_ids = [int(v) for v in payload.get("frame_ids", [])]
    non_overlap_idx = int(config.overlap)
    if non_overlap_idx < 0 or non_overlap_idx >= len(frame_ids):
        raise ValueError(f"invalid overlap={config.overlap} for frame_count={len(frame_ids)}")
    first_non_overlap_frame_id = int(frame_ids[non_overlap_idx])
    target_label_path = reference_labels.get(first_non_overlap_frame_id)
    if target_label_path is None:
        raise FileNotFoundError(f"missing reference label for non-overlap frame {first_non_overlap_frame_id}")
    target_label = _load_label(target_label_path)
    shape = target_label.shape[:2]

    out_dir = phase5_dir / f"{variant}_overlap_deferred"
    rows_by_key: Dict[tuple[int, int], Dict[str, Any]] = {}
    mask_by_key: Dict[tuple[int, int], Any] = {}
    records = []
    moved_count = 0
    dropped_count = 0
    kept_count = 0
    kept_overlap_birth_count = 0
    keep_min_area = int(getattr(config, "phase5_overlap_birth_keep_min_area", 0) or 0)
    for row in payload.get("rows", []):
        role = str(row.get("phase5_role", ""))
        chunk_idx = int(row.get("chunk_frame_index", -1))
        if role == "birth_new" and chunk_idx < non_overlap_idx:
            source_ref_obj_id = row.get("source_ref_obj_id")
            row_area = int(row.get("mask_area", 0) or 0)
            record = {
                "obj_id": int(row.get("obj_id", -1)),
                "source_ref_obj_id": source_ref_obj_id,
                "mask_area": int(row_area),
                "from_chunk_frame_index": int(chunk_idx),
                "from_frame_id": int(row.get("frame_id", -1)),
                "first_non_overlap_chunk_frame_index": int(non_overlap_idx),
                "first_non_overlap_frame_id": int(first_non_overlap_frame_id),
                "action": "move_birth_new_to_earliest_valid_non_overlap",
                "keep_min_area": int(keep_min_area),
            }
            if keep_min_area > 0 and row_area >= keep_min_area:
                record["action"] = "keep_overlap_birth_above_min_area"
                kept_overlap_birth_count += 1
                mask = _load_mask(_resolve(repo_root, str(row["mask_path"])), shape)
                kept = dict(row)
                kept["source"] = f"{row.get('source')}_phase9_overlap_birth_kept"
                kept["phase9_overlap_birth_kept"] = True
                kept["phase9_overlap_birth_keep_min_area"] = int(keep_min_area)
                _copy_or_merge_row(
                    repo_root=repo_root,
                    rows_by_key=rows_by_key,
                    mask_by_key=mask_by_key,
                    variant_dir=out_dir,
                    row=kept,
                    mask=mask,
                )
                kept_count += 1
                records.append(record)
                continue
            if source_ref_obj_id is None:
                record["action"] = "drop_overlap_birth_without_source_ref_obj_id"
                dropped_count += 1
                records.append(record)
                continue
            selected_idx = None
            selected_frame_id = None
            selected_mask = None
            selected_area = 0
            for candidate_idx in range(non_overlap_idx, len(frame_ids)):
                candidate_frame_id = int(frame_ids[candidate_idx])
                candidate_label_path = reference_labels.get(candidate_frame_id)
                if candidate_label_path is None:
                    continue
                candidate_label = _load_label(candidate_label_path)
                candidate_mask = _mask_for_obj(candidate_label, int(source_ref_obj_id))
                candidate_area = int(candidate_mask.sum())
                if candidate_area >= int(config.phase5_min_area):
                    selected_idx = int(candidate_idx)
                    selected_frame_id = int(candidate_frame_id)
                    selected_mask = candidate_mask
                    selected_area = int(candidate_area)
                    break
            record["to_chunk_frame_index"] = selected_idx
            record["to_frame_id"] = selected_frame_id
            record["target_mask_area"] = int(selected_area)
            if selected_mask is None or selected_idx is None or selected_frame_id is None:
                record["action"] = "drop_overlap_birth_missing_non_overlap_mask"
                dropped_count += 1
                records.append(record)
                continue
            moved = dict(row)
            moved["chunk_frame_index"] = int(selected_idx)
            moved["frame_id"] = int(selected_frame_id)
            moved["source"] = f"{row.get('source')}_phase9_overlap_birth_deferred"
            moved["phase9_overlap_birth_deferred"] = True
            moved["phase9_overlap_birth_original_chunk_frame_index"] = int(chunk_idx)
            moved["phase9_overlap_birth_original_frame_id"] = int(row.get("frame_id", -1))
            _copy_or_merge_row(
                repo_root=repo_root,
                rows_by_key=rows_by_key,
                mask_by_key=mask_by_key,
                variant_dir=out_dir,
                row=moved,
                mask=selected_mask,
            )
            moved_count += 1
            records.append(record)
            continue

        mask = _load_mask(_resolve(repo_root, str(row["mask_path"])), shape)
        _copy_or_merge_row(
            repo_root=repo_root,
            rows_by_key=rows_by_key,
            mask_by_key=mask_by_key,
            variant_dir=out_dir,
            row=dict(row),
            mask=mask,
        )
        kept_count += 1

    rows = sorted(rows_by_key.values(), key=lambda item: (int(item["chunk_frame_index"]), int(item["obj_id"])))
    out_payload = {
        "schema_version": "stream4d_v106_phase9_phase5_overlap_deferred_birth_records_v1",
        "scene_id": config.scene_id,
        "variant": f"{variant}_overlap_deferred",
        "base_variant": variant,
        "frame_ids": frame_ids,
        "row_count": int(len(rows)),
        "rows": rows,
        "overlap_defer_policy": {
            "enabled": True,
            "overlap": int(config.overlap),
            "first_non_overlap_chunk_frame_index": int(non_overlap_idx),
            "first_non_overlap_frame_id": int(first_non_overlap_frame_id),
            "moved_birth_new_count": int(moved_count),
            "dropped_overlap_birth_count": int(dropped_count),
            "kept_overlap_birth_count": int(kept_overlap_birth_count),
            "keep_min_area": int(keep_min_area),
            "kept_row_count": int(kept_count),
        },
    }
    out_birth_records = out_dir / "birth_records.json"
    transition_records = out_dir / "overlap_birth_defer_records.json"
    write_json(out_birth_records, out_payload)
    write_json(transition_records, records)
    updated = dict(audit)
    updated.update(
        {
            "base_birth_records_path": audit["birth_records_path"],
            "base_birth_records_sha256": audit["birth_records_sha256"],
            "birth_records_path": _rel(repo_root, out_birth_records),
            "birth_records_sha256": sha256_file(out_birth_records),
            "schedule_row_count": int(len(rows)),
            "overlap_birth_defer": {
                "enabled": True,
                "records_path": _rel(repo_root, transition_records),
                "records_sha256": sha256_file(transition_records),
                "moved_birth_new_count": int(moved_count),
                "dropped_overlap_birth_count": int(dropped_count),
                "kept_overlap_birth_count": int(kept_overlap_birth_count),
                "keep_min_area": int(keep_min_area),
                "kept_row_count": int(kept_count),
                "first_non_overlap_chunk_frame_index": int(non_overlap_idx),
                "first_non_overlap_frame_id": int(first_non_overlap_frame_id),
            },
        }
    )
    return updated


def _load_mask_native(path: Path) -> Any:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img > 0


def _guard_phase5_overlap_inherited_masks(
    *,
    repo_root: Path,
    config: Phase9SceneLoopSmokeConfig,
    phase5_dir: Path,
    variant: str,
    audit: Dict[str, Any],
    inherited_birth_records: Path,
) -> Dict[str, Any]:
    current_birth_records = _resolve(repo_root, audit["birth_records_path"])
    current_payload = _read_json(current_birth_records)
    inherited_payload = _read_json(inherited_birth_records)
    frame_ids = [int(v) for v in current_payload.get("frame_ids", [])]
    if not frame_ids:
        frame_ids = [int(v) for v in inherited_payload.get("frame_ids", [])]

    inherited_by_key: Dict[tuple[int, int], Dict[str, Any]] = {}
    for row in inherited_payload.get("rows", []):
        key = (int(row["chunk_frame_index"]), int(row["obj_id"]))
        inherited_by_key.setdefault(key, dict(row))

    out_dir = phase5_dir / f"{current_birth_records.parent.name}_handoff_overlap_guarded"
    rows_by_key: Dict[tuple[int, int], Dict[str, Any]] = {}
    mask_by_key: Dict[tuple[int, int], Any] = {}
    guard_records: List[Dict[str, Any]] = []
    preserved_count = 0
    skipped_count = 0
    overlap = int(config.overlap)

    for row in current_payload.get("rows", []):
        copied = dict(row)
        key = (int(copied["chunk_frame_index"]), int(copied["obj_id"]))
        current_mask_path = _resolve(repo_root, str(copied["mask_path"]))
        current_mask = _load_mask_native(current_mask_path)
        roles = {str(v) for v in copied.get("phase5_merged_roles", []) if str(v)}
        merged_source_count = int(copied.get("phase5_merged_source_count", 1) or 1)
        inherited_row = inherited_by_key.get(key)
        original_mask = None
        original_mask_path = None
        original_area = 0
        current_area = int(current_mask.sum())
        if inherited_row is not None:
            original_mask_path = _resolve(repo_root, str(inherited_row["mask_path"]))
            original_mask = _load_mask(original_mask_path, current_mask.shape[:2])
            original_area = int(original_mask.sum())
        should_guard = (
            int(copied["chunk_frame_index"]) < overlap
            and "inherited" in roles
            and original_mask is not None
            and current_area > original_area
        )
        if should_guard:
            copied["source"] = f"{copied.get('source')}_phase9_handoff_overlap_inherited_mask_guarded"
            copied["phase9_handoff_overlap_inherited_mask_guarded"] = True
            copied["phase9_handoff_overlap_guard_reason"] = (
                "overlap inherited row had repair/birth/defer union sources; "
                "preserve original handoff objectlet mask for overlap identity"
            )
            copied["phase9_handoff_overlap_guard_original_mask_path"] = _rel(repo_root, original_mask_path)
            copied["phase9_handoff_overlap_guard_original_mask_area"] = int(original_area)
            copied["phase9_handoff_overlap_guard_pre_guard_mask_path"] = _rel(repo_root, current_mask_path)
            copied["phase9_handoff_overlap_guard_pre_guard_mask_area"] = int(current_area)
            copied["phase9_handoff_overlap_guard_removed_pixels_estimate"] = int(
                max(0, current_area - original_area)
            )
            mask_to_write = original_mask
            preserved_count += 1
            guard_records.append(
                {
                    "chunk_frame_index": int(copied["chunk_frame_index"]),
                    "frame_id": int(copied["frame_id"]),
                    "obj_id": int(copied["obj_id"]),
                    "action": "preserve_original_inherited_handoff_mask",
                    "phase5_merged_roles": sorted(roles),
                    "phase5_merged_source_count": int(merged_source_count),
                    "pre_guard_mask_area": int(current_area),
                    "original_mask_area": int(original_area),
                    "removed_pixels_estimate": int(max(0, current_area - original_area)),
                    "pre_guard_mask_path": _rel(repo_root, current_mask_path),
                    "original_mask_path": _rel(repo_root, original_mask_path),
                }
            )
        else:
            mask_to_write = current_mask
            if int(copied["chunk_frame_index"]) < overlap and "inherited" in roles:
                skipped_count += 1

        _copy_or_merge_row(
            repo_root=repo_root,
            rows_by_key=rows_by_key,
            mask_by_key=mask_by_key,
            variant_dir=out_dir,
            row=copied,
            mask=mask_to_write,
        )

    rows = sorted(rows_by_key.values(), key=lambda item: (int(item["chunk_frame_index"]), int(item["obj_id"])))
    out_payload = {
        "schema_version": "stream4d_v106_phase9_phase5_handoff_overlap_guarded_birth_records_v1",
        "scene_id": config.scene_id,
        "variant": f"{current_birth_records.parent.name}_handoff_overlap_guarded",
        "base_variant": variant,
        "frame_ids": frame_ids,
        "row_count": int(len(rows)),
        "rows": rows,
        "handoff_overlap_inherited_mask_guard": {
            "enabled": True,
            "overlap": int(overlap),
            "preserved_overlap_inherited_mask_count": int(preserved_count),
            "skipped_overlap_inherited_mask_count": int(skipped_count),
            "policy": "preserve original inherited handoff masks for overlap rows widened by repair unions",
        },
    }
    out_birth_records = out_dir / "birth_records.json"
    guard_records_path = out_dir / "handoff_overlap_inherited_mask_guard_records.json"
    write_json(out_birth_records, out_payload)
    write_json(guard_records_path, guard_records)
    updated = dict(audit)
    updated.update(
        {
            "pre_handoff_overlap_guard_birth_records_path": audit["birth_records_path"],
            "pre_handoff_overlap_guard_birth_records_sha256": audit["birth_records_sha256"],
            "birth_records_path": _rel(repo_root, out_birth_records),
            "birth_records_sha256": sha256_file(out_birth_records),
            "schedule_row_count": int(len(rows)),
            "handoff_overlap_inherited_mask_guard": {
                "enabled": True,
                "records_path": _rel(repo_root, guard_records_path),
                "records_sha256": sha256_file(guard_records_path),
                "preserved_overlap_inherited_mask_count": int(preserved_count),
                "skipped_overlap_inherited_mask_count": int(skipped_count),
                "overlap": int(overlap),
                "policy": "preserve original inherited handoff masks for overlap rows widened by repair unions",
            },
        }
    )
    return updated


def _configured_replay_summary(config: Phase9SceneLoopSmokeConfig, variant: str) -> str:
    if variant == H2_VARIANT:
        return config.h2_replay_summary
    if variant == H4_VARIANT:
        return config.h4_replay_summary
    raise ValueError(f"unsupported handoff_replay_variant={variant}")


def _execute_replay_requested(config: Phase9SceneLoopSmokeConfig, variant: str) -> bool:
    if bool(config.execute_handoff_replay):
        return True
    return bool(variant == H2_VARIANT and config.execute_h2_replay)


def _build_replay_command(
    *,
    repo_root: Path,
    config: Phase9SceneLoopSmokeConfig,
    birth_records_path: Path,
    output_root: Path,
) -> List[str]:
    cmd = [
        str(PYTHON),
        "tools/build_v105_phase5_frozen_birth_replay.py",
        "--config",
        config.sam2_replay_config,
        "--scene-id",
        config.scene_id,
        "--birth-records",
        _rel(repo_root, birth_records_path),
        "--reference-summary",
        config.reference_c1_summary,
        "--output-root",
        _rel(repo_root, output_root),
        "--frame-start",
        str(int(config.c1_frame_start)),
        "--frame-stride",
        str(int(config.frame_stride)),
        "--frame-count",
        str(int(config.frame_count)),
        "--anchor-birth-priority",
        "--use-video-feature-bank",
        "--video-feature-bank-storage-device",
        str(config.video_feature_bank_storage_device),
        "--video-gpu-hot-window",
        str(int(config.video_gpu_hot_window)),
        "--duplicate-window-frames",
        str(int(config.duplicate_window_frames)),
        "--duplicate-overlap-threshold",
        str(float(config.duplicate_overlap_threshold)),
        "--min-birth-mask-area",
        str(int(config.phase5_min_birth_mask_area)),
        "--min-output-mask-area",
        str(int(config.phase5_min_output_mask_area)),
        "--min-output-component-area",
        str(int(config.phase5_min_output_component_area)),
    ]
    if bool(config.reuse_video_state_template):
        cmd.append("--reuse-video-state-template")
    return cmd


def _build_residual_repair_command(
    *,
    repo_root: Path,
    config: Phase9SceneLoopSmokeConfig,
    inherited_birth_records: Path,
    handoff_replay_summary: Path,
    output_root: Path,
) -> List[str]:
    cmd = [
        str(PYTHON),
        "tools/build_v106_phase9_residual_gap_birth_bank.py",
        "--config",
        str(config.sam2_replay_config),
        "--scene-id",
        str(config.scene_id),
        "--inherited-birth-records",
        _rel(repo_root, inherited_birth_records),
        "--handoff-replay-summary",
        _rel(repo_root, handoff_replay_summary),
        "--output-root",
        _rel(repo_root, output_root),
        "--seed",
        "106",
        "--model-dtype",
        str(config.residual_model_dtype),
        "--residual-input-role",
        str(config.residual_input_role),
        "--residual-mode",
        str(config.residual_mode),
        "--min-uncovered-ratio",
        str(float(config.residual_min_uncovered_ratio)),
        "--start-chunk-index",
        str(int(config.residual_start_chunk_index)),
        "--frame-step",
        str(int(config.residual_frame_step)),
        "--max-residual-frames",
        str(int(config.residual_max_residual_frames)),
        "--max-birth-mask-area",
        str(int(config.residual_max_birth_mask_area)),
        "--max-birth-mask-area-ratio",
        str(float(config.residual_max_birth_mask_area_ratio)),
        "--max-birth-mask-uncovered-ratio",
        str(float(config.residual_max_birth_mask_uncovered_ratio)),
        "--repair-birth-defer-mode",
        str(config.residual_repair_birth_defer_mode),
        "--repair-overlap-coeff",
        str(float(config.residual_repair_overlap_coeff)),
        "--duplicate-suppress-overlap-coeff",
        str(float(config.residual_duplicate_suppress_overlap_coeff)),
        "--birth-max-overlap-coeff",
        str(float(config.residual_birth_max_overlap_coeff)),
        "--repair-birth-defer-min-area",
        str(int(config.residual_repair_birth_defer_min_area)),
        "--ambiguous-overlap-action",
        str(config.residual_ambiguous_overlap_action),
        "--temporal-residual-repair-mode",
        str(config.residual_temporal_repair_mode),
        "--temporal-residual-min-overlap",
        str(float(config.residual_temporal_min_overlap)),
        "--temporal-residual-min-area",
        str(int(config.residual_temporal_min_area)),
        "--temporal-residual-max-area-ratio",
        str(float(config.residual_temporal_max_area_ratio)),
        "--temporal-residual-window-chunks",
        str(int(config.residual_temporal_window_chunks)),
        "--temporal-residual-min-target-age-chunks",
        str(int(config.residual_temporal_min_target_age_chunks)),
        "--temporal-residual-young-match-action",
        str(config.residual_temporal_young_match_action),
        "--source",
        str(config.residual_source),
    ]
    selected = str(config.residual_selected_chunk_indices or "").strip()
    if selected:
        cmd.extend(["--selected-chunk-indices", selected])
    return cmd


def _run_logged(
    *,
    repo_root: Path,
    cmd: List[str],
    log_path: Path,
    gpu: str,
) -> Dict[str, Any]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTHONUNBUFFERED": "1",
        }
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "$ CUDA_VISIBLE_DEVICES="
            + str(gpu)
            + " PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 "
            + " ".join(cmd)
            + "\n\n"
        )
        handle.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "returncode": int(proc.returncode),
        "runtime_sec": float(time.time() - started),
        "log_path": _rel(repo_root, log_path),
    }


def run_phase9_scene_loop_smoke(
    repo_root: Path,
    config: Phase9SceneLoopSmokeConfig,
    output_dir: Path,
) -> Dict[str, Any]:
    if os.environ.get("STREAM4D_V106_ALLOW_LEGACY_FILE_CHAIN_SMOKE", "").strip() != "1":
        raise RuntimeError(
            "v106 Phase9 file-level scene-loop smoke is disabled for the main path. "
            "It chains chunks through summary/labels/birth-record files instead of an "
            "in-memory SAM2 scene state. Use tools/run_v106_stateful_sam2_scene_stream.py "
            "for visual-first stateful streaming. Set "
            "STREAM4D_V106_ALLOW_LEGACY_FILE_CHAIN_SMOKE=1 only for legacy audit reproduction."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    boundary_dir = _boundary_dir(config, output_dir)
    boundary_name = _boundary_name(config)
    replay_variant = _replay_variant(config)

    prep_summary = run_phase3_handoff_smoke(repo_root, _handoff_config(config), boundary_dir)
    birth_records_path = _birth_records_path(config, output_dir, replay_variant)
    replay_output_root = _replay_output_root(config, output_dir, replay_variant)
    replay_summary_path = _replay_summary_path(replay_output_root)
    configured_summary_text = _configured_replay_summary(config, replay_variant)
    configured_replay_summary = _resolve(repo_root, configured_summary_text) if configured_summary_text else None

    reference_path = _resolve(repo_root, config.reference_c1_summary) if config.reference_c1_summary else None
    replay_requested = _execute_replay_requested(config, replay_variant)
    can_execute_replay = bool(
        replay_requested
        and birth_records_path.exists()
        and reference_path is not None
        and reference_path.exists()
    )
    replay_command = _build_replay_command(
        repo_root=repo_root,
        config=config,
        birth_records_path=birth_records_path,
        output_root=replay_output_root,
    )
    replay_record: Dict[str, Any] = {
        "variant": replay_variant,
        "requested": bool(replay_requested),
        "executed": False,
        "can_execute": bool(can_execute_replay),
        "gpu": str(config.replay_gpu),
        "command": [
            f"CUDA_VISIBLE_DEVICES={config.replay_gpu}",
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            "PYTHONUNBUFFERED=1",
            *replay_command,
        ],
        "reference_summary_exists": bool(reference_path and reference_path.exists()),
        "birth_records_exists": bool(birth_records_path.exists()),
        "output_root": _rel(repo_root, replay_output_root),
    }
    if can_execute_replay:
        run_record = _run_logged(
            repo_root=repo_root,
            cmd=replay_command,
            log_path=output_dir / f"{replay_variant}_replay_command.log",
            gpu=str(config.replay_gpu),
        )
        replay_record.update(run_record)
        replay_record["executed"] = True

    selected_replay_summary = None
    if replay_summary_path.exists():
        selected_replay_summary = replay_summary_path
    elif configured_replay_summary is not None and configured_replay_summary.exists():
        selected_replay_summary = configured_replay_summary

    active_birth_records_path = birth_records_path
    handoff_drift_filter_record: Dict[str, Any] = {
        "enabled": bool(config.handoff_drift_filter_enabled),
    }
    if bool(config.handoff_drift_filter_enabled) and not bool(config.phase5_repair_birth_defer_enabled):
        handoff_drift_filter_record.update(
            {
                "skipped": True,
                "skip_reason": "handoff_drift_filter currently feeds the Phase5 schedule; phase5_repair_birth_defer is disabled.",
            }
        )
    phase5_record: Dict[str, Any] = {"enabled": bool(config.phase5_repair_birth_defer_enabled)}
    if bool(config.phase5_repair_birth_defer_enabled):
        if not config.phase5_reference_birth_records:
            raise ValueError("phase5_reference_birth_records is required when phase5_repair_birth_defer_enabled=true")
        inherited_summary_text = str(config.phase5_inherited_replay_summary or "")
        inherited_replay_summary = _resolve(repo_root, inherited_summary_text) if inherited_summary_text else selected_replay_summary
        if inherited_replay_summary is None or not inherited_replay_summary.exists():
            raise FileNotFoundError(
                "Phase5 schedule needs an inherited handoff replay summary; "
                "set phase5_inherited_replay_summary or enable/execute the handoff replay first."
            )
        phase5_inherited_birth_records = birth_records_path
        phase5_inherited_replay_summary = inherited_replay_summary
        if bool(config.handoff_drift_filter_enabled):
            handoff_drift_filter_record = _filter_handoff_birth_records_by_drift(
                repo_root=repo_root,
                config=config,
                output_dir=boundary_dir,
                birth_records_path=birth_records_path,
                handoff_replay_summary=inherited_replay_summary,
            )
            phase5_inherited_birth_records = _resolve(
                repo_root,
                handoff_drift_filter_record["filtered_birth_records"],
            )
            phase5_inherited_replay_summary = _resolve(
                repo_root,
                handoff_drift_filter_record["synthetic_filtered_handoff_summary"],
            )
        phase5_record = _build_phase5_schedule(
            repo_root=repo_root,
            config=config,
            output_dir=boundary_dir,
            inherited_birth_records=phase5_inherited_birth_records,
            inherited_replay_summary=phase5_inherited_replay_summary,
        )
        if bool(config.handoff_drift_filter_enabled):
            phase5_record["handoff_drift_filter_applied"] = True
            phase5_record["handoff_drift_filter"] = handoff_drift_filter_record
        phase5_birth_records = _resolve(repo_root, phase5_record["schedule_audit"]["birth_records_path"])
        phase5_replay_variant = f"{replay_variant}_{phase5_record['variant']}"
        phase5_replay_output_root = boundary_dir / f"{phase5_replay_variant}_phase5_replay"
        phase5_replay_summary_path = _replay_summary_path(phase5_replay_output_root)
        phase5_replay_command = _build_replay_command(
            repo_root=repo_root,
            config=config,
            birth_records_path=phase5_birth_records,
            output_root=phase5_replay_output_root,
        )
        phase5_replay_record: Dict[str, Any] = {
            "variant": phase5_replay_variant,
            "requested": bool(replay_requested),
            "executed": False,
            "can_execute": bool(replay_requested),
            "gpu": str(config.replay_gpu),
            "command": [
                f"CUDA_VISIBLE_DEVICES={config.replay_gpu}",
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                "PYTHONUNBUFFERED=1",
                *phase5_replay_command,
            ],
            "birth_records": _rel(repo_root, phase5_birth_records),
            "output_root": _rel(repo_root, phase5_replay_output_root),
        }
        if replay_requested:
            run_record = _run_logged(
                repo_root=repo_root,
                cmd=phase5_replay_command,
                log_path=output_dir / f"{phase5_replay_variant}_replay_command.log",
                gpu=str(config.replay_gpu),
            )
            phase5_replay_record.update(run_record)
            phase5_replay_record["executed"] = True
        phase5_record["replay"] = phase5_replay_record
        if phase5_replay_summary_path.exists():
            selected_replay_summary = phase5_replay_summary_path
            active_birth_records_path = phase5_birth_records

    residual_record: Dict[str, Any] = {"enabled": bool(config.residual_repair_enabled)}
    residual_required_failed = False
    if bool(config.residual_repair_enabled):
        if selected_replay_summary is None or not selected_replay_summary.exists():
            raise FileNotFoundError(
                "Residual repair needs a replay summary; enable/execute handoff or Phase5 replay first."
            )
        if not active_birth_records_path.exists():
            raise FileNotFoundError(f"Residual repair inherited birth records missing: {active_birth_records_path}")
        residual_variant_parts = [replay_variant]
        if bool(config.phase5_repair_birth_defer_enabled):
            residual_variant_parts.append(str(phase5_record.get("variant") or _phase5_variant(config)))
        residual_variant_parts.append("residual_repair")
        residual_variant = "_".join(residual_variant_parts)
        residual_bank_output_root = boundary_dir / f"{residual_variant}_birth_bank"
        residual_birth_records_path = residual_bank_output_root / "residual_gap_birth_records.json"
        residual_bank_summary_path = residual_bank_output_root / "residual_gap_birth_bank_summary.json"
        residual_command = _build_residual_repair_command(
            repo_root=repo_root,
            config=config,
            inherited_birth_records=active_birth_records_path,
            handoff_replay_summary=selected_replay_summary,
            output_root=residual_bank_output_root,
        )
        residual_record = {
            "enabled": True,
            "variant": residual_variant,
            "birth_bank": {
                "gpu": str(config.residual_gpu),
                "command": [
                    f"CUDA_VISIBLE_DEVICES={config.residual_gpu}",
                    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                    "PYTHONUNBUFFERED=1",
                    *residual_command,
                ],
                "inherited_birth_records": _rel(repo_root, active_birth_records_path),
                "inherited_birth_records_sha256": sha256_file(active_birth_records_path),
                "handoff_replay_summary": _rel(repo_root, selected_replay_summary),
                "handoff_replay_summary_sha256": sha256_file(selected_replay_summary),
                "output_root": _rel(repo_root, residual_bank_output_root),
            },
        }
        residual_run_record = _run_logged(
            repo_root=repo_root,
            cmd=residual_command,
            log_path=output_dir / f"{residual_variant}_birth_bank_command.log",
            gpu=str(config.residual_gpu),
        )
        residual_record["birth_bank"].update(residual_run_record)
        residual_record["birth_bank"]["executed"] = True
        residual_record["birth_records"] = _rel(repo_root, residual_birth_records_path)
        residual_record["birth_records_exists"] = bool(residual_birth_records_path.exists())
        residual_record["birth_records_sha256"] = (
            sha256_file(residual_birth_records_path) if residual_birth_records_path.exists() else None
        )
        residual_record["birth_bank_summary"] = _rel(repo_root, residual_bank_summary_path)
        residual_record["birth_bank_summary_exists"] = bool(residual_bank_summary_path.exists())
        residual_record["birth_bank_summary_sha256"] = (
            sha256_file(residual_bank_summary_path) if residual_bank_summary_path.exists() else None
        )

        residual_replay_output_root = boundary_dir / f"{residual_variant}_phase5_replay"
        residual_replay_summary_path = _replay_summary_path(residual_replay_output_root)
        residual_replay_command = _build_replay_command(
            repo_root=repo_root,
            config=config,
            birth_records_path=residual_birth_records_path,
            output_root=residual_replay_output_root,
        )
        residual_replay_record: Dict[str, Any] = {
            "variant": residual_variant,
            "requested": True,
            "executed": False,
            "can_execute": bool(residual_birth_records_path.exists()),
            "gpu": str(config.replay_gpu),
            "command": [
                f"CUDA_VISIBLE_DEVICES={config.replay_gpu}",
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                "PYTHONUNBUFFERED=1",
                *residual_replay_command,
            ],
            "birth_records": _rel(repo_root, residual_birth_records_path),
            "output_root": _rel(repo_root, residual_replay_output_root),
        }
        if residual_birth_records_path.exists():
            residual_replay_run = _run_logged(
                repo_root=repo_root,
                cmd=residual_replay_command,
                log_path=output_dir / f"{residual_variant}_replay_command.log",
                gpu=str(config.replay_gpu),
            )
            residual_replay_record.update(residual_replay_run)
            residual_replay_record["executed"] = True
        residual_replay_record["summary"] = _rel(repo_root, residual_replay_summary_path)
        residual_replay_record["summary_exists"] = bool(residual_replay_summary_path.exists())
        residual_replay_record["summary_sha256"] = (
            sha256_file(residual_replay_summary_path) if residual_replay_summary_path.exists() else None
        )
        residual_record["replay"] = residual_replay_record
        if residual_replay_summary_path.exists():
            selected_replay_summary = residual_replay_summary_path
            active_birth_records_path = residual_birth_records_path
        else:
            residual_required_failed = True

    final_summary: Dict[str, Any] | None = None
    if selected_replay_summary is not None:
        final_summary = run_phase3_handoff_smoke(
            repo_root,
            _handoff_config(config, _rel(repo_root, selected_replay_summary), replay_variant),
            boundary_dir,
        )

    replay_metrics = {}
    if isinstance(final_summary, dict):
        replay_metrics = final_summary.get("variant_metrics", {}).get(replay_variant, {})

    summary = {
        "schema_version": "stream4d_v106_phase9_scene_loop_smoke_v1",
        "scope": "two_chunk_scene_loop_smoke",
        "full_dev_evidence": False,
        "scene_id": config.scene_id,
        "boundary_name": boundary_name,
        "from_chunk_index": int(config.c0_chunk_index),
        "to_chunk_index": int(config.c1_chunk_index),
        "same_scene_chunks_sequential": True,
        "stage12_full_initialization_used_by_handoff": False,
        "config": asdict(config),
        "prep_summary_json": _rel(repo_root, boundary_dir / "phase3_gate_summary.json"),
        "prep_initial_passes": bool(prep_summary.get("passes", False)),
        "prep_artifacts_ready": bool(birth_records_path.exists()),
        "handoff_replay_variant": replay_variant,
        "handoff_birth_records": _rel(repo_root, birth_records_path),
        "handoff_birth_records_exists": bool(birth_records_path.exists()),
        "handoff_birth_records_sha256": sha256_file(birth_records_path) if birth_records_path.exists() else None,
        "active_birth_records": _rel(repo_root, active_birth_records_path),
        "active_birth_records_exists": bool(active_birth_records_path.exists()),
        "active_birth_records_sha256": sha256_file(active_birth_records_path) if active_birth_records_path.exists() else None,
        "handoff_replay": replay_record,
        "handoff_drift_filter": handoff_drift_filter_record,
        "phase5_repair_birth_defer": phase5_record,
        "residual_repair": residual_record,
        "handoff_replay_summary": _rel(repo_root, selected_replay_summary) if selected_replay_summary else "",
        "handoff_replay_summary_sha256": sha256_file(selected_replay_summary) if selected_replay_summary else None,
        "handoff_overlap_metrics": replay_metrics,
        "h2_birth_records": _rel(repo_root, boundary_dir / H2_VARIANT / "birth_records.json"),
        "h2_replay": replay_record if replay_variant == H2_VARIANT else {},
        "h2_overlap_metrics": final_summary.get("variant_metrics", {}).get(H2_VARIANT, {}) if isinstance(final_summary, dict) else {},
        "passes": bool(replay_metrics.get("passes", False)) and not bool(residual_required_failed),
        "reason_if_not_pass": (
            "" if bool(replay_metrics.get("passes", False)) and not bool(residual_required_failed) else
            (
                "Residual repair was enabled but residual replay summary was missing; this remains smoke evidence, "
                "not full-dev evidence."
                if bool(residual_required_failed)
                else f"{replay_variant} replay summary is missing or overlap gate did not pass; this remains smoke evidence, not full-dev evidence."
            )
        ),
    }
    write_json(output_dir / "scene_loop_smoke_summary.json", summary)
    return summary
