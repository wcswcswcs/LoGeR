from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .artifacts import sha256_file, write_json
from .config import Phase5RepairBirthDeferConfig
from .phase3_handoff import _real_label_metrics


VARIANTS = ("R0_all_gap_birth", "R1_repair_vs_birth", "R2_repair_birth_defer")


def _resolve(repo_root: Path, path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_chunk_index_set(text: str | None) -> set[int] | None:
    stripped = str(text or "").strip()
    if not stripped:
        return None
    values: set[int] = set()
    for part in stripped.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            values.add(int(token))
        except ValueError as exc:
            raise ValueError(f"invalid selected_chunk_indices token {token!r} in {text!r}") from exc
    return values if values else None


def _load_label(path: str | Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.uint16, copy=False)


def _load_mask(path: str | Path, shape: Tuple[int, int]) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    mask = img > 0
    if mask.shape[:2] != shape:
        mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return mask.astype(bool)


def _write_mask(mask: np.ndarray, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
    if not ok:
        raise IOError(f"failed to write mask: {path}")
    return int(np.count_nonzero(mask))


def _summary_label_map(repo_root: Path, summary_path: Path) -> Dict[int, Path]:
    summary = _read_json(summary_path)
    out = {}
    for row in summary.get("records", []):
        out[int(row["frame_id"])] = _resolve(repo_root, str(row["label_path"]))
    return out


def _visible_ids(label: np.ndarray) -> List[int]:
    return [int(v) - 1 for v in np.unique(label.astype(np.int64)).tolist() if int(v) > 0]


def _mask_for_obj(label: np.ndarray, obj_id: int) -> np.ndarray:
    return (label == int(obj_id) + 1).astype(bool)


def _rel(repo_root: Path, path: Path) -> str:
    return str(path.relative_to(repo_root) if path.is_relative_to(repo_root) else path)


def _overlap_with_inherited(mask: np.ndarray, inherited_label: np.ndarray | None) -> Dict[str, Any]:
    if inherited_label is None:
        return {
            "best_inherited_obj_id": None,
            "best_overlap_coeff": 0.0,
            "best_intersection_pixels": 0,
            "best_inherited_area": 0,
        }
    area = int(np.count_nonzero(mask))
    best = {
        "best_inherited_obj_id": None,
        "best_overlap_coeff": 0.0,
        "best_intersection_pixels": 0,
        "best_inherited_area": 0,
    }
    for obj_id in _visible_ids(inherited_label):
        inherited = _mask_for_obj(inherited_label, obj_id)
        inherited_area = int(np.count_nonzero(inherited))
        inter = int(np.count_nonzero(mask & inherited))
        coeff = float(inter / max(1, min(area, inherited_area)))
        if coeff > float(best["best_overlap_coeff"]):
            best = {
                "best_inherited_obj_id": int(obj_id),
                "best_overlap_coeff": float(coeff),
                "best_intersection_pixels": int(inter),
                "best_inherited_area": int(inherited_area),
            }
    return best


def _persistence_frames(ref_labels: Dict[int, Path], frame_ids: List[int], start_idx: int, obj_id: int) -> int:
    count = 0
    for idx in range(int(start_idx), len(frame_ids)):
        label = _load_label(ref_labels[int(frame_ids[idx])])
        if np.any(label == int(obj_id) + 1):
            count += 1
    return int(count)


def _copy_or_merge_row(
    *,
    repo_root: Path,
    rows_by_key: Dict[Tuple[int, int], Dict[str, Any]],
    mask_by_key: Dict[Tuple[int, int], np.ndarray],
    variant_dir: Path,
    row: Dict[str, Any],
    mask: np.ndarray,
) -> None:
    key = (int(row["chunk_frame_index"]), int(row["obj_id"]))
    current = mask_by_key.get(key)
    merged = mask.astype(bool) if current is None else (current | mask.astype(bool))
    mask_by_key[key] = merged
    mask_path = variant_dir / "masks" / f"frame_{int(row['frame_id']):06d}_obj_{int(row['obj_id']):06d}.png"
    area = _write_mask(merged, mask_path)
    rel_mask = _rel(repo_root, mask_path)
    if key not in rows_by_key:
        copied = dict(row)
        copied["mask_path"] = rel_mask
        copied["mask_area"] = int(area)
        copied["phase5_merged_roles"] = [str(row.get("phase5_role", ""))]
        copied["phase5_merged_source_count"] = 1
        rows_by_key[key] = copied
    else:
        existing = rows_by_key[key]
        roles = set(existing.get("phase5_merged_roles", []))
        roles.add(str(row.get("phase5_role", "")))
        existing["phase5_merged_roles"] = sorted(v for v in roles if v)
        existing["phase5_merged_source_count"] = int(existing.get("phase5_merged_source_count", 1)) + 1
        existing["mask_path"] = rel_mask
        existing["mask_area"] = int(area)


def _build_variant_birth_records(
    *,
    repo_root: Path,
    output_dir: Path,
    config: Phase5RepairBirthDeferConfig,
    variant: str,
    inherited_payload: Dict[str, Any],
    reference_payload: Dict[str, Any],
    inherited_labels: Dict[int, Path],
    reference_labels: Dict[int, Path],
) -> Dict[str, Any]:
    variant_dir = output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    frame_ids = [int(v) for v in reference_payload.get("frame_ids", [])]
    if not frame_ids:
        frame_ids = [int(config.frame_start + i * config.frame_stride) for i in range(config.frame_count)]
    first_label = _load_label(reference_labels[frame_ids[0]])
    shape = first_label.shape[:2]

    inherited_rows = inherited_payload.get("rows", [])
    reference_rows = reference_payload.get("rows", [])
    selected_chunk_indices = _parse_chunk_index_set(config.selected_chunk_indices)
    selected_chunk_index_list = sorted(selected_chunk_indices) if selected_chunk_indices is not None else None
    max_existing_obj = max(
        [int(row["obj_id"]) for row in inherited_rows + reference_rows] + [0]
    )
    next_new_obj_id = max_existing_obj + 1000

    rows_by_key: Dict[Tuple[int, int], Dict[str, Any]] = {}
    mask_by_key: Dict[Tuple[int, int], np.ndarray] = {}
    classification_records: List[Dict[str, Any]] = []

    for row in inherited_rows:
        src = _resolve(repo_root, str(row["mask_path"]))
        mask = _load_mask(src, shape)
        copied = dict(row)
        copied["source"] = f"v106_phase5_{variant}_inherited_handoff"
        copied["phase5_role"] = "inherited"
        copied["phase5_variant"] = variant
        _copy_or_merge_row(
            repo_root=repo_root,
            rows_by_key=rows_by_key,
            mask_by_key=mask_by_key,
            variant_dir=variant_dir,
            row=copied,
            mask=mask,
        )

    for idx, row in enumerate(reference_rows):
        chunk_idx = int(row["chunk_frame_index"])
        frame_id = int(row["frame_id"])
        source_ref_obj_id = int(row["obj_id"])
        if selected_chunk_indices is not None and chunk_idx not in selected_chunk_indices:
            classification_records.append(
                {
                    "candidate_index": int(idx),
                    "variant": variant,
                    "source_ref_obj_id": int(source_ref_obj_id),
                    "source_frame_id": int(frame_id),
                    "source_chunk_frame_index": int(chunk_idx),
                    "source_mask_area": None,
                    "action": "skip_unselected_frame",
                    "reason": "chunk_frame_index_not_in_phase5_selected_chunk_indices",
                    "scheduled_obj_id": None,
                    "scheduled_frame_id": None,
                    "scheduled_chunk_frame_index": None,
                    "deferred_to_frame_id": None,
                    "persistence_frames_from_anchor": None,
                    "feature_similarity_available": False,
                    "candidate_inherited_area_ratio": None,
                    "selected_chunk_indices": selected_chunk_index_list,
                }
            )
            continue
        mask = _load_mask(_resolve(repo_root, str(row["mask_path"])), shape)
        area = int(np.count_nonzero(mask))
        inherited_label = _load_label(inherited_labels[frame_id]) if frame_id in inherited_labels else None
        overlap = _overlap_with_inherited(mask, inherited_label)
        persistence = _persistence_frames(reference_labels, frame_ids, chunk_idx, source_ref_obj_id)
        inherited_area = int(overlap.get("best_inherited_area", 0) or 0)
        area_ratio = (
            float(min(area, inherited_area) / max(1, max(area, inherited_area)))
            if inherited_area > 0
            else 0.0
        )
        action = "birth_new"
        target_obj_id: int | None = None
        deferred_to_frame_id: int | None = None
        scheduled_mask = mask
        scheduled_chunk_idx = chunk_idx
        scheduled_frame_id = frame_id
        reason = "r0_all_candidates_are_births"

        if variant == "R1_repair_vs_birth":
            if (
                float(overlap["best_overlap_coeff"]) >= float(config.duplicate_suppress_overlap_coeff)
                and area_ratio >= float(config.duplicate_suppress_area_ratio_min)
            ):
                action = "duplicate_suppress_existing"
                target_obj_id = int(overlap["best_inherited_obj_id"])
                reason = "high_overlap_duplicate_suppressed_without_mask_update"
            elif float(overlap["best_overlap_coeff"]) >= float(config.repair_overlap_coeff):
                action = "repair_existing"
                target_obj_id = int(overlap["best_inherited_obj_id"])
                reason = "overlap_coeff_above_repair_threshold"
            elif area < int(config.min_area) or persistence < int(config.min_persistence_frames):
                action = "noise"
                reason = "too_small_or_single_frame_unstable"
            else:
                action = "birth_new"
                reason = "not_repair_and_no_defer_in_R1"
        elif variant == "R2_repair_birth_defer":
            if (
                float(overlap["best_overlap_coeff"]) >= float(config.duplicate_suppress_overlap_coeff)
                and area_ratio >= float(config.duplicate_suppress_area_ratio_min)
            ):
                action = "duplicate_suppress_existing"
                target_obj_id = int(overlap["best_inherited_obj_id"])
                reason = "high_overlap_duplicate_suppressed_without_mask_update"
            elif float(overlap["best_overlap_coeff"]) >= float(config.repair_overlap_coeff):
                action = "repair_existing"
                target_obj_id = int(overlap["best_inherited_obj_id"])
                reason = "overlap_coeff_above_repair_threshold"
            elif area < int(config.min_area) or persistence < int(config.min_persistence_frames):
                action = "noise"
                reason = "too_small_or_single_frame_unstable"
            elif (
                float(overlap["best_overlap_coeff"]) <= float(config.birth_max_overlap_coeff)
                or area >= int(config.large_persistent_area)
            ):
                action = "birth_new"
                reason = "low_overlap_or_large_persistent_gap"
            else:
                action = "defer"
                reason = "ambiguous_overlap_one_frame_defer"
                next_idx = min(chunk_idx + 1, len(frame_ids) - 1)
                next_frame_id = int(frame_ids[next_idx])
                next_label = _load_label(reference_labels[next_frame_id])
                next_mask = _mask_for_obj(next_label, source_ref_obj_id)
                if next_idx > chunk_idx and int(np.count_nonzero(next_mask)) >= int(config.min_area):
                    action = "birth_new"
                    scheduled_chunk_idx = int(next_idx)
                    scheduled_frame_id = int(next_frame_id)
                    scheduled_mask = next_mask
                    deferred_to_frame_id = int(next_frame_id)
                    reason = "ambiguous_deferred_one_frame_then_birth_from_frozen_reference_mask"
                else:
                    action = "noise"
                    reason = "ambiguous_defer_without_valid_next_frame_mask"

        scheduled_obj_id: int | None = None
        if action == "duplicate_suppress_existing":
            scheduled_obj_id = None
        elif action == "repair_existing":
            scheduled_obj_id = int(target_obj_id)
        elif action == "birth_new":
            scheduled_obj_id = int(next_new_obj_id)
            next_new_obj_id += 1

        record = {
            "candidate_index": int(idx),
            "variant": variant,
            "source_ref_obj_id": int(source_ref_obj_id),
            "source_frame_id": int(frame_id),
            "source_chunk_frame_index": int(chunk_idx),
            "source_mask_area": int(area),
            "action": action,
            "reason": reason,
            "scheduled_obj_id": scheduled_obj_id,
            "scheduled_frame_id": scheduled_frame_id if scheduled_obj_id is not None else None,
            "scheduled_chunk_frame_index": scheduled_chunk_idx if scheduled_obj_id is not None else None,
            "deferred_to_frame_id": deferred_to_frame_id,
            "persistence_frames_from_anchor": int(persistence),
            "feature_similarity_available": False,
            "candidate_inherited_area_ratio": float(area_ratio),
            **overlap,
        }
        classification_records.append(record)

        if scheduled_obj_id is None:
            continue
        schedule_row = {
            "scene_id": config.scene_id,
            "chunk_frame_index": int(scheduled_chunk_idx),
            "frame_id": int(scheduled_frame_id),
            "obj_id": int(scheduled_obj_id),
            "global_id": int(scheduled_obj_id) + 1,
            "source": f"v106_phase5_{variant}_{action}",
            "source_ref_obj_id": int(source_ref_obj_id),
            "source_ref_birth_frame_id": int(frame_id),
            "phase5_role": "repair_existing" if action == "repair_existing" else "birth_new",
            "phase5_variant": variant,
            "prompt_kind": "binary_mask",
            "logits_available": False,
        }
        _copy_or_merge_row(
            repo_root=repo_root,
            rows_by_key=rows_by_key,
            mask_by_key=mask_by_key,
            variant_dir=variant_dir,
            row=schedule_row,
            mask=scheduled_mask,
        )

    rows = sorted(rows_by_key.values(), key=lambda item: (int(item["chunk_frame_index"]), int(item["obj_id"])))
    payload = {
        "schema_version": "stream4d_v106_phase5_repair_birth_defer_birth_records_v1",
        "scene_id": config.scene_id,
        "variant": variant,
        "frame_ids": frame_ids,
        "row_count": int(len(rows)),
        "rows": rows,
    }
    birth_records_path = variant_dir / "birth_records.json"
    write_json(birth_records_path, payload)
    write_json(variant_dir / "classification_records.json", classification_records)
    counts = defaultdict(int)
    for row in classification_records:
        counts[str(row["action"])] += 1
    audit = {
        "variant": variant,
        "birth_records_path": str(birth_records_path),
        "birth_records_sha256": sha256_file(birth_records_path),
        "classification_records_path": str(variant_dir / "classification_records.json"),
        "candidate_count": int(len(classification_records)),
        "reference_row_count": int(len(reference_rows)),
        "selected_chunk_indices": selected_chunk_index_list,
        "skipped_unselected_reference_row_count": int(counts.get("skip_unselected_frame", 0)),
        "schedule_row_count": int(len(rows)),
        "action_counts": {k: int(v) for k, v in sorted(counts.items())},
        "inherited_row_count": int(len(inherited_rows)),
        "feature_similarity_available": False,
        "defer_mask_source_note": "R2 one-frame deferred births use frozen reference labels as a replay proxy when available.",
    }
    write_json(variant_dir / "schedule_audit.json", audit)
    return audit


def _metrics_against_reference(
    *,
    repo_root: Path,
    replay_summary_path: Path,
    reference_summary_path: Path,
    config: Phase5RepairBirthDeferConfig,
) -> Dict[str, Any]:
    replay = _read_json(replay_summary_path)
    reference = _read_json(reference_summary_path)
    replay_labels = _summary_label_map(repo_root, replay_summary_path)
    reference_labels = _summary_label_map(repo_root, reference_summary_path)
    frame_ids = sorted(set(replay_labels) & set(reference_labels))
    asa_num = 0
    asa_den = 0
    coverage_num = 0
    coverage_den = 0
    ttp_overlap: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    ttp_den_by_pred: Dict[int, int] = defaultdict(int)
    metric_records = []
    blank_frames = []

    for frame_id in frame_ids:
        pred = _load_label(replay_labels[frame_id])
        ref = _load_label(reference_labels[frame_id])
        pred_fg = pred > 0
        ref_fg = ref > 0
        if not np.any(pred_fg):
            blank_frames.append(int(frame_id))
        coverage_num += int(np.count_nonzero(pred_fg & ref_fg))
        coverage_den += int(np.count_nonzero(ref_fg))
        ref_ids = _visible_ids(ref)
        for pred_id in _visible_ids(pred):
            pred_mask = _mask_for_obj(pred, pred_id)
            pred_area = int(np.count_nonzero(pred_mask))
            asa_den += pred_area
            ttp_den_by_pred[int(pred_id)] += pred_area
            best = 0
            for ref_id in ref_ids:
                overlap = int(np.count_nonzero(pred_mask & _mask_for_obj(ref, ref_id)))
                best = max(best, overlap)
                ttp_overlap[int(pred_id)][int(ref_id)] += overlap
            asa_num += int(best)
        metrics = _real_label_metrics(
            ref,
            pred,
            fragment_overlap_fraction_threshold=float(config.fragment_overlap_fraction_threshold),
            merge_overlap_fraction_threshold=float(config.merge_overlap_fraction_threshold),
        )
        metrics.update({"frame_id": int(frame_id), "pred_label_path": str(replay_labels[frame_id])})
        metric_records.append(metrics)

    ttp_num = sum(max(overlaps.values()) if overlaps else 0 for overlaps in ttp_overlap.values())
    ttp_den = sum(ttp_den_by_pred.values())
    drr = replay.get("duplicate_rebirth_audit", {})
    values = {
        "frame_count": int(len(frame_ids)),
        "coverage": float(coverage_num / coverage_den) if coverage_den > 0 else 1.0,
        "asa": float(asa_num / asa_den) if asa_den > 0 else 1.0,
        "ttp": float(ttp_num / ttp_den) if ttp_den > 0 else 1.0,
        "coverage_numerator": int(coverage_num),
        "coverage_denominator": int(coverage_den),
        "asa_numerator": int(asa_num),
        "asa_denominator": int(asa_den),
        "ttp_numerator": int(ttp_num),
        "ttp_denominator": int(ttp_den),
        "max_cfr": float(max([row["CFR"] for row in metric_records], default=0.0)),
        "max_cmr": float(max([row["CMR"] for row in metric_records], default=0.0)),
        "mean_foreground_union_iou": float(np.mean([row["foreground_union_iou"] for row in metric_records]))
        if metric_records else 1.0,
        "blank_frame_count": int(len(blank_frames)),
        "blank_frame_ids": blank_frames,
        "drr_area": float(drr.get("drr_area", 0.0)),
        "drr_count": float(drr.get("drr_count", 0.0)),
        "duplicate_birth_id_count": int(drr.get("duplicate_birth_id_count", 0) or 0),
        "new_birth_id_count": int(drr.get("new_birth_id_count", 0) or 0),
        "duplicate_rebirth_audit_available": bool(drr),
        "feature_similarity_used_for_drr": bool(drr.get("feature_similarity_used", False)) if drr else False,
        "total_runtime_sec": replay.get("total_runtime_sec"),
        "total_tracking_runtime_sec": replay.get("total_tracking_runtime_sec"),
        "peak_cuda_memory_mb": replay.get("peak_cuda_memory_mb"),
        "video_path": replay.get("video_path"),
    }
    return {"metrics": values, "frame_records": metric_records}


def _replay_command(
    *,
    config: Phase5RepairBirthDeferConfig,
    variant: str,
    birth_records_path: str,
    output_root: Path,
) -> List[str]:
    return [
        f"CUDA_VISIBLE_DEVICES=<6-or-7>",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "PYTHONUNBUFFERED=1",
        "/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python",
        "tools/build_v105_phase5_frozen_birth_replay.py",
        "--config",
        "configs/v105/baseline_chunk_table/baseline_x_gapadaptive_sam2.generated.yaml",
        "--scene-id",
        config.scene_id,
        "--birth-records",
        birth_records_path,
        "--reference-summary",
        config.reference_summary,
        "--output-root",
        str(output_root / f"{variant}_phase5_replay"),
        "--frame-start",
        str(config.frame_start),
        "--frame-stride",
        str(config.frame_stride),
        "--frame-count",
        str(config.frame_count),
        "--anchor-birth-priority",
        "--use-video-feature-bank",
        "--video-feature-bank-storage-device",
        "cuda",
        "--video-gpu-hot-window",
        "0",
        "--reuse-video-state-template",
        "--duplicate-window-frames",
        str(config.duplicate_window_frames),
        "--duplicate-overlap-threshold",
        str(config.duplicate_overlap_threshold),
    ]


def run_phase5_repair_birth_defer(
    repo_root: Path,
    config: Phase5RepairBirthDeferConfig,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "inherited_birth_records": _resolve(repo_root, config.inherited_birth_records),
        "inherited_replay_summary": _resolve(repo_root, config.inherited_replay_summary),
        "reference_birth_records": _resolve(repo_root, config.reference_birth_records),
        "reference_summary": _resolve(repo_root, config.reference_summary),
    }
    missing = {name: str(path) for name, path in paths.items() if not path.exists()}
    if missing:
        summary = {
            "schema_version": "stream4d_v106_phase5_repair_birth_defer_summary_v1",
            "passes": False,
            "missing_paths": missing,
        }
        write_json(output_dir / "gate_records.json", summary)
        write_json(output_dir / "failure_records.json", [{"failure": "missing_required_artifact", "paths": missing}])
        return summary

    inherited_payload = _read_json(paths["inherited_birth_records"])
    reference_payload = _read_json(paths["reference_birth_records"])
    inherited_labels = _summary_label_map(repo_root, paths["inherited_replay_summary"])
    reference_labels = _summary_label_map(repo_root, paths["reference_summary"])

    schedule_audits = {}
    commands = {}
    for variant in VARIANTS:
        audit = _build_variant_birth_records(
            repo_root=repo_root,
            output_dir=output_dir,
            config=config,
            variant=variant,
            inherited_payload=inherited_payload,
            reference_payload=reference_payload,
            inherited_labels=inherited_labels,
            reference_labels=reference_labels,
        )
        schedule_audits[variant] = audit
        commands[variant] = _replay_command(
            config=config,
            variant=variant,
            birth_records_path=audit["birth_records_path"],
            output_root=output_dir,
        )

    replay_paths = {
        "R0_all_gap_birth": config.r0_replay_summary,
        "R1_repair_vs_birth": config.r1_replay_summary,
        "R2_repair_birth_defer": config.r2_replay_summary,
    }
    variant_metrics = {}
    for variant, path_text in replay_paths.items():
        if not path_text:
            variant_metrics[variant] = {"has_replay": False, "passes_metric_eval": False}
            continue
        replay_path = _resolve(repo_root, path_text)
        if not replay_path.exists():
            variant_metrics[variant] = {
                "has_replay": False,
                "passes_metric_eval": False,
                "missing_replay_summary": str(replay_path),
            }
            continue
        metrics = _metrics_against_reference(
            repo_root=repo_root,
            replay_summary_path=replay_path,
            reference_summary_path=paths["reference_summary"],
            config=config,
        )
        variant_dir = output_dir / variant
        write_json(variant_dir / "metric_records_vs_reference.json", metrics["frame_records"])
        write_json(variant_dir / "metric_summary_vs_reference.json", metrics["metrics"])
        variant_metrics[variant] = {
            "has_replay": True,
            "replay_summary_path": str(replay_path),
            "replay_summary_sha256": sha256_file(replay_path),
            **metrics["metrics"],
        }

    checks: List[Dict[str, Any]] = [
        {
            "name": "all_variant_schedules_written",
            "passes": all(Path(schedule_audits[v]["birth_records_path"]).exists() for v in VARIANTS),
            "actual": {v: schedule_audits[v]["birth_records_path"] for v in VARIANTS},
            "expected": "R0/R1/R2 birth_records exist",
        }
    ]
    r0 = variant_metrics.get("R0_all_gap_birth", {})
    passing_variants = []
    if not r0.get("has_replay"):
        checks.append(
            {
                "name": "r0_replay_available",
                "passes": False,
                "actual": replay_paths["R0_all_gap_birth"],
                "expected": "run R0 replay first",
            }
        )
    else:
        for variant in ("R1_repair_vs_birth", "R2_repair_birth_defer"):
            cur = variant_metrics.get(variant, {})
            if not cur.get("has_replay"):
                checks.append(
                    {
                        "name": f"{variant}_replay_available",
                        "passes": False,
                        "actual": replay_paths[variant],
                        "expected": f"run {variant} replay",
                    }
                )
                continue
            r0_drr = float(r0.get("drr_area", 0.0))
            cur_drr = float(cur.get("drr_area", 0.0))
            if r0_drr > 0.0:
                drr_reduction = (r0_drr - cur_drr) / r0_drr
                drr_pass = drr_reduction >= float(config.required_drr_relative_reduction)
            else:
                drr_reduction = 0.0 if cur_drr <= 0.0 else -1.0
                drr_pass = cur_drr <= 0.0
            variant_checks = [
                {
                    "name": f"{variant}_drr_relative_reduction",
                    "passes": bool(drr_pass),
                    "actual": {
                        "r0_drr_area": r0_drr,
                        "variant_drr_area": cur_drr,
                        "relative_reduction": float(drr_reduction),
                    },
                    "expected_min": config.required_drr_relative_reduction,
                },
                {
                    "name": f"{variant}_coverage_not_down",
                    "passes": float(cur.get("coverage", 0.0)) + config.metric_epsilon >= float(r0.get("coverage", 0.0)),
                    "actual": {"r0": r0.get("coverage"), "variant": cur.get("coverage")},
                    "expected": "variant >= R0",
                },
                {
                    "name": f"{variant}_asa_not_down",
                    "passes": float(cur.get("asa", 0.0)) + config.metric_epsilon >= float(r0.get("asa", 0.0)),
                    "actual": {"r0": r0.get("asa"), "variant": cur.get("asa")},
                    "expected": "variant >= R0",
                },
                {
                    "name": f"{variant}_ttp_not_down",
                    "passes": float(cur.get("ttp", 0.0)) + config.metric_epsilon >= float(r0.get("ttp", 0.0)),
                    "actual": {"r0": r0.get("ttp"), "variant": cur.get("ttp")},
                    "expected": "variant >= R0",
                },
                {
                    "name": f"{variant}_cfr_not_worse",
                    "passes": float(cur.get("max_cfr", 1.0)) <= float(r0.get("max_cfr", 1.0)) + config.metric_epsilon,
                    "actual": {"r0": r0.get("max_cfr"), "variant": cur.get("max_cfr")},
                    "expected": "variant <= R0",
                },
                {
                    "name": f"{variant}_cmr_not_worse",
                    "passes": float(cur.get("max_cmr", 1.0)) <= float(r0.get("max_cmr", 1.0)) + config.metric_epsilon,
                    "actual": {"r0": r0.get("max_cmr"), "variant": cur.get("max_cmr")},
                    "expected": "variant <= R0",
                },
            ]
            checks.extend(variant_checks)
            if all(check["passes"] for check in variant_checks):
                passing_variants.append(variant)

    failure_records = [
        {
            "name": check["name"],
            "actual": check.get("actual"),
            "expected": check.get("expected", check.get("expected_min")),
        }
        for check in checks
        if not check["passes"]
    ]
    summary = {
        "schema_version": "stream4d_v106_phase5_repair_birth_defer_summary_v1",
        "scene_id": config.scene_id,
        "passes": bool(not failure_records and bool(passing_variants)),
        "passing_variants": passing_variants,
        "gate_checks": checks,
        "failure_count": int(len(failure_records)),
        "schedule_audits": schedule_audits,
        "variant_metrics": variant_metrics,
        "replay_commands": commands,
        "notes": [
            "R0/R1/R2 schedules are built from inherited handoff rows plus frozen C2 reference candidate masks.",
            "R2 one-frame defer uses frozen reference labels as a replay proxy when a next-frame candidate mask is available.",
            "DRR is computed by the replay helper from pre-disjoin masks; SAM2 feature similarity is not available in this audit.",
        ],
    }
    write_json(output_dir / "gate_records.json", summary)
    write_json(output_dir / "failure_records.json", failure_records)
    write_json(output_dir / "replay_commands.json", commands)
    return summary
