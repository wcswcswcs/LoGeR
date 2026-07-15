from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2
import numpy as np

from .artifacts import sha256_file, write_json
from .config import Phase3HandoffConfig


VARIANT_ORDER = (
    "H0_latest_overlap",
    "H1_best_quality_overlap",
    "H2_best_plus_one_correction",
    "H4_endpoint_drift_correction",
)


@dataclass(frozen=True)
class ObjectMaskCandidate:
    obj_id: int
    global_id: int
    source_frame_id: int
    source_overlap_index: int
    target_chunk_frame_index: int
    mask_area: int
    mask_path: Path


def _resolve(repo_root: Path, path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _label_path_from_summary(summary: Dict[str, Any], frame_id: int, fallback_dir: Path) -> Path:
    for row in summary.get("records", []):
        if int(row.get("frame_id")) == int(frame_id):
            return _resolve(Path.cwd(), str(row["label_path"]))
    return fallback_dir / f"frame_{int(frame_id):06d}.png"


def _load_label(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.uint16, copy=False)


def _visible_obj_ids(label: np.ndarray) -> List[int]:
    values = np.unique(label.astype(np.int64))
    return [int(v) - 1 for v in values.tolist() if int(v) > 0]


def _write_mask(label: np.ndarray, obj_id: int, path: Path) -> int:
    mask = (label == int(obj_id) + 1).astype(np.uint8) * 255
    area = int(np.count_nonzero(mask))
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), mask)
    if not ok:
        raise IOError(f"failed to write mask: {path}")
    return area


def _summary_label_map(repo_root: Path, summary_path: str) -> Dict[int, Path]:
    if not summary_path:
        return {}
    summary = _read_json(_resolve(repo_root, summary_path))
    out: Dict[int, Path] = {}
    for row in summary.get("records", []):
        frame_id = int(row["frame_id"])
        out[frame_id] = _resolve(repo_root, str(row["label_path"]))
    return out


def _pair_partition_consistency(previous: np.ndarray, current: np.ndarray, union: np.ndarray) -> float:
    prev = previous[union].astype(np.int64)
    cur = current[union].astype(np.int64)
    n = int(prev.size)
    if n < 2:
        return 1.0
    total_pairs = comb(n, 2)
    prev_pos = prev > 0
    cur_pos = cur > 0
    same_prev_pairs = sum(comb(int(count), 2) for count in np.bincount(prev[prev_pos]).tolist() if count >= 2)
    same_cur_pairs = sum(comb(int(count), 2) for count in np.bincount(cur[cur_pos]).tolist() if count >= 2)
    if np.any(prev_pos & cur_pos):
        max_cur = int(cur.max()) + 1
        joint = prev[prev_pos & cur_pos] * max_cur + cur[prev_pos & cur_pos]
        both_same = sum(comb(int(count), 2) for count in np.bincount(joint).tolist() if count >= 2)
    else:
        both_same = 0
    both_diff = total_pairs - same_prev_pairs - same_cur_pairs + both_same
    return float((both_same + both_diff) / total_pairs)


def _real_label_metrics(
    source: np.ndarray,
    target: np.ndarray,
    *,
    fragment_overlap_fraction_threshold: float,
    merge_overlap_fraction_threshold: float,
) -> Dict[str, Any]:
    if source.shape != target.shape:
        raise ValueError(f"label shape mismatch: source={source.shape} target={target.shape}")
    source_fg = source > 0
    target_fg = target > 0
    union = source_fg | target_fg
    intersection = source_fg & target_fg
    union_count = int(np.count_nonzero(union))
    source_count = int(np.count_nonzero(source_fg))
    target_count = int(np.count_nonzero(target_fg))
    same_label_on_union = int(np.count_nonzero((source == target) & union))

    source_ids = set(_visible_obj_ids(source))
    target_ids = set(_visible_obj_ids(target))
    id_intersection = source_ids & target_ids

    fragmented = 0
    raw_fragmented = 0
    source_area_total = 0
    fragmented_source_area = 0
    raw_fragmented_source_area = 0
    source_overlap_ids: Dict[int, List[int]] = {}
    source_overlap_ids_thresholded: Dict[int, List[int]] = {}
    for obj_id in sorted(source_ids):
        source_mask = source == int(obj_id) + 1
        source_area = max(1, int(np.count_nonzero(source_mask)))
        source_area_total += source_area
        vals, counts = np.unique(target[source_mask], return_counts=True)
        cur_pairs = [
            (int(v) - 1, int(c), float(int(c) / source_area))
            for v, c in zip(vals, counts, strict=False)
            if int(v) > 0
        ]
        cur_ids = sorted({item[0] for item in cur_pairs})
        threshold_ids = sorted(
            {item[0] for item in cur_pairs if item[2] >= float(fragment_overlap_fraction_threshold)}
        )
        source_overlap_ids[int(obj_id)] = cur_ids
        source_overlap_ids_thresholded[int(obj_id)] = threshold_ids
        if len(cur_ids) > 1:
            raw_fragmented += 1
            raw_fragmented_source_area += source_area
        if len(threshold_ids) > 1:
            fragmented += 1
            fragmented_source_area += source_area

    merged = 0
    raw_merged = 0
    target_area_total = 0
    merged_target_area = 0
    raw_merged_target_area = 0
    false_merge_minor_pixels = 0
    raw_false_merge_minor_pixels = 0
    target_overlap_ids: Dict[int, List[int]] = {}
    target_overlap_ids_thresholded: Dict[int, List[int]] = {}
    raw_max_minor_overlap_fraction = 0.0
    for obj_id in sorted(target_ids):
        target_mask = target == int(obj_id) + 1
        target_area = max(1, int(np.count_nonzero(target_mask)))
        target_area_total += target_area
        vals, counts = np.unique(source[target_mask], return_counts=True)
        prev_pairs = sorted(
            [
                (int(v) - 1, int(c), float(int(c) / target_area))
                for v, c in zip(vals, counts, strict=False)
                if int(v) > 0
            ],
            key=lambda item: item[1],
            reverse=True,
        )
        prev_ids = sorted({item[0] for item in prev_pairs})
        threshold_pairs = [
            item for item in prev_pairs if item[2] >= float(merge_overlap_fraction_threshold)
        ]
        threshold_ids = sorted({item[0] for item in threshold_pairs})
        target_overlap_ids[int(obj_id)] = prev_ids
        target_overlap_ids_thresholded[int(obj_id)] = threshold_ids
        if len(prev_pairs) > 1:
            raw_merged += 1
            raw_merged_target_area += target_area
            raw_minor = sum(item[1] for item in prev_pairs[1:])
            raw_false_merge_minor_pixels += raw_minor
            raw_max_minor_overlap_fraction = max(raw_max_minor_overlap_fraction, float(raw_minor / target_area))
        if len(threshold_pairs) > 1:
            merged += 1
            merged_target_area += target_area
            false_merge_minor_pixels += sum(item[1] for item in threshold_pairs[1:])

    cfr_count = 0.0 if not source_ids else float(fragmented / len(source_ids))
    cmr_count = 0.0 if not target_ids else float(merged / len(target_ids))
    raw_contact_cfr_count = 0.0 if not source_ids else float(raw_fragmented / len(source_ids))
    raw_contact_cmr_count = 0.0 if not target_ids else float(raw_merged / len(target_ids))
    cfr_area = 0.0 if source_area_total == 0 else float(fragmented_source_area / source_area_total)
    cmr_area = 0.0 if target_area_total == 0 else float(merged_target_area / target_area_total)
    raw_contact_cfr_area = (
        0.0 if source_area_total == 0 else float(raw_fragmented_source_area / source_area_total)
    )
    raw_contact_cmr_area = (
        0.0 if target_area_total == 0 else float(raw_merged_target_area / target_area_total)
    )

    return {
        "CCOC": 1.0 if union_count == 0 else float(same_label_on_union / union_count),
        "HIR": 1.0 if not source_ids else float(len(id_intersection) / len(source_ids)),
        "HCR": 1.0 if source_count == 0 else float(np.count_nonzero(intersection) / source_count),
        "OPC": _pair_partition_consistency(source, target, union) if union_count else 1.0,
        "CFR": cfr_area,
        "CMR": cmr_area,
        "BFMR": 0.0 if target_count == 0 else float(false_merge_minor_pixels / target_count),
        "CFR_count": cfr_count,
        "CMR_count": cmr_count,
        "raw_contact_CFR": raw_contact_cfr_area,
        "raw_contact_CMR": raw_contact_cmr_area,
        "raw_contact_CFR_count": raw_contact_cfr_count,
        "raw_contact_CMR_count": raw_contact_cmr_count,
        "raw_contact_CFR_area": raw_contact_cfr_area,
        "raw_contact_CMR_area": raw_contact_cmr_area,
        "raw_contact_BFMR_minor_pixel_ratio": (
            0.0 if target_count == 0 else float(raw_false_merge_minor_pixels / target_count)
        ),
        "raw_max_minor_overlap_fraction": float(raw_max_minor_overlap_fraction),
        "source_area_total": int(source_area_total),
        "target_area_total": int(target_area_total),
        "fragmented_source_id_count": int(fragmented),
        "merged_target_id_count": int(merged),
        "raw_fragmented_source_id_count": int(raw_fragmented),
        "raw_merged_target_id_count": int(raw_merged),
        "fragmented_source_area": int(fragmented_source_area),
        "merged_target_area": int(merged_target_area),
        "raw_fragmented_source_area": int(raw_fragmented_source_area),
        "raw_merged_target_area": int(raw_merged_target_area),
        "fragment_overlap_fraction_threshold": float(fragment_overlap_fraction_threshold),
        "merge_overlap_fraction_threshold": float(merge_overlap_fraction_threshold),
        "foreground_union_iou": 1.0 if union_count == 0 else float(np.count_nonzero(intersection) / union_count),
        "exact_equal": bool(np.array_equal(source, target)),
        "source_visible_id_count": int(len(source_ids)),
        "target_visible_id_count": int(len(target_ids)),
        "shared_visible_id_count": int(len(id_intersection)),
        "missing_source_ids": [int(v) for v in sorted(source_ids - target_ids)],
        "extra_target_ids": [int(v) for v in sorted(target_ids - source_ids)],
        "source_overlap_ids": source_overlap_ids,
        "source_overlap_ids_thresholded": source_overlap_ids_thresholded,
        "target_overlap_ids": target_overlap_ids,
        "target_overlap_ids_thresholded": target_overlap_ids_thresholded,
    }


def _aggregate_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {
            "has_metrics": False,
            "frame_count": 0,
        }
    metric_names = (
        "CCOC",
        "HIR",
        "HCR",
        "OPC",
        "CFR",
        "CMR",
        "BFMR",
        "foreground_union_iou",
        "CFR_count",
        "CMR_count",
        "raw_contact_CFR",
        "raw_contact_CMR",
        "raw_contact_CFR_count",
        "raw_contact_CMR_count",
        "raw_contact_CFR_area",
        "raw_contact_CMR_area",
    )
    out: Dict[str, Any] = {"has_metrics": True, "frame_count": int(len(records))}
    for name in metric_names:
        values = [float(row[name]) for row in records if name in row]
        if not values:
            continue
        out[f"mean_{name}"] = float(np.mean(values))
        out[f"min_{name}"] = float(min(values))
        out[f"max_{name}"] = float(max(values))
    out["exact_frame_count"] = int(sum(1 for row in records if row["exact_equal"]))
    out["source_visible_id_count_mean"] = float(np.mean([row["source_visible_id_count"] for row in records]))
    out["target_visible_id_count_mean"] = float(np.mean([row["target_visible_id_count"] for row in records]))
    return out


def _select_candidates(
    variant: str,
    candidates_by_obj: Dict[int, List[ObjectMaskCandidate]],
    config: Phase3HandoffConfig,
) -> List[ObjectMaskCandidate]:
    selected: List[ObjectMaskCandidate] = []
    for obj_id in sorted(candidates_by_obj):
        rows = sorted(
            candidates_by_obj[obj_id],
            key=lambda row: (row.source_overlap_index, row.mask_area),
        )
        best = max(rows, key=lambda row: (row.mask_area, row.source_overlap_index))
        earliest_index = min(r.source_overlap_index for r in rows)
        earliest = [row for row in rows if row.source_overlap_index == earliest_index][-1]
        latest_index = max(r.source_overlap_index for r in rows)
        latest = [row for row in rows if row.source_overlap_index == latest_index][-1]
        if variant == "H0_latest_overlap":
            selected.append(latest)
        elif variant == "H1_best_quality_overlap":
            selected.append(best)
        elif variant == "H2_best_plus_one_correction":
            selected.append(best)
            if earliest.source_overlap_index != best.source_overlap_index:
                selected.append(earliest)
        elif variant == "H4_endpoint_drift_correction":
            by_overlap = {int(best.source_overlap_index): best, int(earliest.source_overlap_index): earliest}
            if latest.source_overlap_index not in by_overlap:
                best_area = max(1, int(best.mask_area))
                latest_area_ratio = float(int(latest.mask_area) / best_area)
                if (
                    latest_area_ratio <= float(config.endpoint_drift_area_ratio_min)
                    or latest_area_ratio >= float(config.endpoint_drift_area_ratio_max)
                ):
                    by_overlap[int(latest.source_overlap_index)] = latest
            selected.extend(by_overlap[index] for index in sorted(by_overlap))
        else:
            raise ValueError(f"unknown Phase3 handoff variant: {variant}")
    selected.sort(key=lambda row: (row.target_chunk_frame_index, row.obj_id, row.mask_area))
    return selected


def _write_variant_artifacts(
    *,
    repo_root: Path,
    output_dir: Path,
    config: Phase3HandoffConfig,
    variant: str,
    candidates_by_obj: Dict[int, List[ObjectMaskCandidate]],
    c1_frame_ids: List[int],
) -> Dict[str, Any]:
    variant_dir = output_dir / variant
    masks_dir = variant_dir / "handoff_masks"
    selected = _select_candidates(variant, candidates_by_obj, config)

    rows = []
    objects = []
    for row in selected:
        variant_mask_path = masks_dir / row.mask_path.name
        variant_mask_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row.mask_path, variant_mask_path)
        rel_mask = (
            variant_mask_path.relative_to(repo_root)
            if variant_mask_path.is_relative_to(repo_root)
            else variant_mask_path
        )
        rows.append(
            {
                "scene_id": config.scene_id,
                "chunk_frame_index": int(row.target_chunk_frame_index),
                "frame_id": int(row.source_frame_id),
                "obj_id": int(row.obj_id),
                "global_id": int(row.global_id),
                "source": f"v106_phase3_{variant}",
                "mask_path": str(rel_mask),
                "mask_area": int(row.mask_area),
                "source_chunk_index": int(config.c0_chunk_index),
                "target_chunk_index": int(config.c1_chunk_index),
                "source_overlap_index": int(row.source_overlap_index),
                "prompt_kind": "binary_mask",
                "logits_available": False,
            }
        )
        objects.append(
            {
                "global_id": int(row.global_id),
                "runtime_local_id": int(row.obj_id),
                "source_chunk_index": int(config.c0_chunk_index),
                "target_chunk_index": int(config.c1_chunk_index),
                "source_frame_id": int(row.source_frame_id),
                "target_chunk_frame_index": int(row.target_chunk_frame_index),
                "prompt_kind": "binary_mask",
                "quality_score": float(row.mask_area),
                "mask_path": str(rel_mask),
                "logits_path": None,
                "logits_available": False,
            }
        )

    birth_payload = {
        "schema_version": "stream4d_v106_phase3_handoff_birth_records_v1",
        "scene_id": config.scene_id,
        "frame_ids": [int(v) for v in c1_frame_ids],
        "variant": variant,
        "row_count": int(len(rows)),
        "rows": rows,
    }
    birth_path = variant_dir / "birth_records.json"
    write_json(birth_path, birth_payload)

    runtime_map = {str(obj["runtime_local_id"]): int(obj["global_id"]) for obj in objects}
    package = {
        "schema_version": "stream4d_v106_handoff_package_v1",
        "scene_id": config.scene_id,
        "from_chunk_index": int(config.c0_chunk_index),
        "to_chunk_index": int(config.c1_chunk_index),
        "from_history_version": int(config.c0_chunk_index + 1),
        "to_history_version": int(config.c1_chunk_index + 1),
        "variant": variant,
        "object_count": int(len({obj["global_id"] for obj in objects})),
        "prompt_record_count": int(len(objects)),
        "runtime_local_to_global": runtime_map,
        "objects": objects,
        "forbidden_stage12_full_initialization": True,
        "stage12_full_initialization_used_by_handoff": False,
        "fake_logits_written": False,
        "handoff_masks_dir": str(masks_dir.relative_to(repo_root) if masks_dir.is_relative_to(repo_root) else masks_dir),
        "birth_records_path": str(birth_path.relative_to(repo_root) if birth_path.is_relative_to(repo_root) else birth_path),
    }
    package_path = variant_dir / "handoff_package.json"
    write_json(package_path, package)

    mask_missing = [row["mask_path"] for row in rows if not _resolve(repo_root, row["mask_path"]).exists()]
    audit = {
        "variant": variant,
        "handoff_package_path": str(package_path),
        "birth_records_path": str(birth_path),
        "birth_records_sha256": sha256_file(birth_path),
        "object_count": int(package["object_count"]),
        "prompt_record_count": int(package["prompt_record_count"]),
        "mask_missing_count": int(len(mask_missing)),
        "mask_missing": mask_missing[:16],
        "runtime_local_to_global_count": int(len(runtime_map)),
        "logits_available": False,
        "fake_logits_written": False,
        "stage12_full_initialization_used_by_handoff": False,
    }
    write_json(variant_dir / "handoff_audit.json", audit)
    return audit


def _build_candidates(
    *,
    repo_root: Path,
    output_dir: Path,
    config: Phase3HandoffConfig,
    c0_summary: Dict[str, Any],
    c0_labels_dir: Path,
    c0_overlap_frame_ids: List[int],
    c1_frame_ids: List[int],
) -> Dict[int, List[ObjectMaskCandidate]]:
    candidates_by_obj: Dict[int, List[ObjectMaskCandidate]] = {}
    mask_root = output_dir / "source_overlap_masks"
    for overlap_index, frame_id in enumerate(c0_overlap_frame_ids):
        label_path = _label_path_from_summary(c0_summary, frame_id, c0_labels_dir)
        if not label_path.is_absolute():
            label_path = repo_root / label_path
        label = _load_label(label_path)
        for obj_id in _visible_obj_ids(label):
            mask_path = mask_root / f"frame_{int(frame_id):06d}_obj_{int(obj_id):06d}.png"
            area = _write_mask(label, obj_id, mask_path)
            if area <= 0:
                continue
            candidates_by_obj.setdefault(int(obj_id), []).append(
                ObjectMaskCandidate(
                    obj_id=int(obj_id),
                    global_id=int(obj_id) + 1,
                    source_frame_id=int(frame_id),
                    source_overlap_index=int(overlap_index),
                    target_chunk_frame_index=int(c1_frame_ids.index(frame_id)),
                    mask_area=int(area),
                    mask_path=mask_path,
                )
            )
    return candidates_by_obj


def _evaluate_variant(
    *,
    repo_root: Path,
    output_dir: Path,
    config: Phase3HandoffConfig,
    variant: str,
    c0_labels: Dict[int, np.ndarray],
    replay_summary_path: str,
) -> Dict[str, Any]:
    variant_dir = output_dir / variant
    if not replay_summary_path:
        result = {
            "variant": variant,
            "has_replay": False,
            "passes": False,
            "reason": "missing replay summary; run the exported birth_records through Phase5 frozen replay",
        }
        write_json(variant_dir / "overlap_metric_summary.json", result)
        return result

    replay_summary_abs = _resolve(repo_root, replay_summary_path)
    label_map = _summary_label_map(repo_root, str(replay_summary_abs))
    records = []
    missing_frames = []
    for frame_id, source_label in sorted(c0_labels.items()):
        target_path = label_map.get(int(frame_id))
        if target_path is None or not target_path.exists():
            missing_frames.append(int(frame_id))
            continue
        target_label = _load_label(target_path)
        metrics = _real_label_metrics(
            source_label,
            target_label,
            fragment_overlap_fraction_threshold=float(config.fragment_overlap_fraction_threshold),
            merge_overlap_fraction_threshold=float(config.merge_overlap_fraction_threshold),
        )
        metrics.update(
            {
                "variant": variant,
                "frame_id": int(frame_id),
                "source_label_shape": list(source_label.shape),
                "target_label_path": str(target_path),
            }
        )
        records.append(metrics)
    aggregate = _aggregate_metrics(records)
    checks = [
        {"name": "replay_summary_exists", "passes": replay_summary_abs.exists(), "actual": str(replay_summary_abs)},
        {"name": "all_overlap_frames_present", "passes": not missing_frames, "actual": missing_frames, "expected": []},
        {"name": "min_CCOC", "passes": float(aggregate.get("min_CCOC", 0.0)) >= float(config.min_ccoc), "actual": aggregate.get("min_CCOC"), "expected_min": config.min_ccoc},
        {"name": "min_HIR", "passes": float(aggregate.get("min_HIR", 0.0)) >= float(config.min_hir), "actual": aggregate.get("min_HIR"), "expected_min": config.min_hir},
        {"name": "min_HCR", "passes": float(aggregate.get("min_HCR", 0.0)) >= float(config.min_hcr), "actual": aggregate.get("min_HCR"), "expected_min": config.min_hcr},
        {"name": "max_CFR", "passes": float(aggregate.get("max_CFR", 1.0)) <= float(config.max_cfr), "actual": aggregate.get("max_CFR"), "expected_max": config.max_cfr},
        {"name": "max_CMR", "passes": float(aggregate.get("max_CMR", 1.0)) <= float(config.max_cmr), "actual": aggregate.get("max_CMR"), "expected_max": config.max_cmr},
        {"name": "max_BFMR", "passes": float(aggregate.get("max_BFMR", 1.0)) <= float(config.max_bfmr), "actual": aggregate.get("max_BFMR"), "expected_max": config.max_bfmr},
    ]
    result = {
        "variant": variant,
        "has_replay": True,
        "replay_summary_path": str(replay_summary_abs),
        "replay_summary_sha256": sha256_file(replay_summary_abs),
        "missing_overlap_frames": missing_frames,
        "aggregate": aggregate,
        "checks": checks,
        "passes": all(bool(check["passes"]) for check in checks),
    }
    write_json(variant_dir / "overlap_frame_metrics.json", records)
    write_json(variant_dir / "overlap_metric_summary.json", result)
    return result


def _phase5_command(
    *,
    config: Phase3HandoffConfig,
    variant: str,
    birth_records_path: Path,
    reference_summary: str,
    output_root: Path,
) -> List[str]:
    return [
        "CUDA_VISIBLE_DEVICES=<6-or-7>",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "PYTHONUNBUFFERED=1",
        "/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python",
        "tools/build_v105_phase5_frozen_birth_replay.py",
        "--config",
        "configs/v105/baseline_chunk_table/baseline_x_gapadaptive_sam2.generated.yaml",
        "--scene-id",
        config.scene_id,
        "--birth-records",
        str(birth_records_path),
        "--reference-summary",
        reference_summary or "<C1-reference-summary.json>",
        "--output-root",
        str(output_root / f"{variant}_phase5_replay"),
        "--frame-start",
        str(config.c1_frame_start),
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
    ]


def run_phase3_handoff_smoke(repo_root: Path, config: Phase3HandoffConfig, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    c0_summary_path = _resolve(repo_root, config.c0_summary)
    c0_labels_dir = _resolve(repo_root, config.c0_labels_dir)
    c0_summary = _read_json(c0_summary_path)
    c0_frame_ids = [int(v) for v in c0_summary.get("frame_ids", [])]
    if not c0_frame_ids:
        c0_frame_ids = [int(config.c0_frame_start + i * config.frame_stride) for i in range(config.frame_count)]
    c1_frame_ids = [int(config.c1_frame_start + i * config.frame_stride) for i in range(config.frame_count)]
    c0_overlap_frame_ids = c0_frame_ids[-int(config.overlap) :]
    expected_overlap = c1_frame_ids[: int(config.overlap)]
    alignment_checks = [
        {
            "name": "overlap_frame_ids_align",
            "passes": c0_overlap_frame_ids == expected_overlap,
            "actual": {"c0_tail": c0_overlap_frame_ids, "c1_head": expected_overlap},
            "expected": "C0 tail equals C1 head",
        },
        {
            "name": "c1_forbids_stage12_full_initialization",
            "passes": True,
            "actual": {"handoff_birth_records_source": "C0 overlap labels only"},
            "expected": "no Stage1/Stage2 masks used by v106 C1 handoff",
        },
    ]
    if not all(check["passes"] for check in alignment_checks):
        summary = {
            "schema_version": "stream4d_v106_phase3_handoff_smoke_summary_v1",
            "passes": False,
            "alignment_checks": alignment_checks,
            "repair_ladder_position": "frame index / preprocess alignment audit",
        }
        write_json(output_dir / "phase3_gate_summary.json", summary)
        return summary

    c0_labels = {
        int(frame_id): _load_label(_label_path_from_summary(c0_summary, int(frame_id), c0_labels_dir))
        for frame_id in c0_overlap_frame_ids
    }
    candidates_by_obj = _build_candidates(
        repo_root=repo_root,
        output_dir=output_dir,
        config=config,
        c0_summary=c0_summary,
        c0_labels_dir=c0_labels_dir,
        c0_overlap_frame_ids=c0_overlap_frame_ids,
        c1_frame_ids=c1_frame_ids,
    )

    inventory = {
        "schema_version": "stream4d_v106_phase3_overlap_inventory_v1",
        "scene_id": config.scene_id,
        "c0_frame_ids": c0_frame_ids,
        "c1_frame_ids": c1_frame_ids,
        "c0_overlap_frame_ids": c0_overlap_frame_ids,
        "candidate_object_count": int(len(candidates_by_obj)),
        "candidate_prompt_count": int(sum(len(v) for v in candidates_by_obj.values())),
        "objects": {
            str(obj_id): [
                {
                    "source_frame_id": int(row.source_frame_id),
                    "source_overlap_index": int(row.source_overlap_index),
                    "target_chunk_frame_index": int(row.target_chunk_frame_index),
                    "mask_area": int(row.mask_area),
                    "mask_path": str(row.mask_path),
                }
                for row in rows
            ]
            for obj_id, rows in sorted(candidates_by_obj.items())
        },
    }
    write_json(output_dir / "handoff_overlap_inventory.json", inventory)

    variant_audits = {}
    replay_paths = {
        "H0_latest_overlap": config.h0_replay_summary,
        "H1_best_quality_overlap": config.h1_replay_summary,
        "H2_best_plus_one_correction": config.h2_replay_summary,
        "H4_endpoint_drift_correction": config.h4_replay_summary,
    }
    variant_metrics = {}
    phase5_commands = {}
    for variant in VARIANT_ORDER:
        audit = _write_variant_artifacts(
            repo_root=repo_root,
            output_dir=output_dir,
            config=config,
            variant=variant,
            candidates_by_obj=candidates_by_obj,
            c1_frame_ids=c1_frame_ids,
        )
        variant_audits[variant] = audit
        variant_metrics[variant] = _evaluate_variant(
            repo_root=repo_root,
            output_dir=output_dir,
            config=config,
            variant=variant,
            c0_labels=c0_labels,
            replay_summary_path=replay_paths[variant],
        )
        phase5_commands[variant] = _phase5_command(
            config=config,
            variant=variant,
            birth_records_path=Path(audit["birth_records_path"]),
            reference_summary=config.reference_c1_summary,
            output_root=output_dir,
        )

    passing_variants = [name for name, payload in variant_metrics.items() if payload.get("passes")]
    repair_ladder = []
    if not passing_variants:
        repair_ladder = [
            "1. frame index / preprocess alignment audit",
            "2. latest -> best quality overlap",
            "3. use logits instead of binary mask when SAM2 exposes real logits",
            "4. add one corrective overlap prompt",
            "5. overlap 3 -> 5 as the only structural variant",
        ]

    summary = {
        "schema_version": "stream4d_v106_phase3_handoff_smoke_summary_v1",
        "scene_id": config.scene_id,
        "passes": bool(passing_variants),
        "passing_variants": passing_variants,
        "alignment_checks": alignment_checks,
        "c1_stage12_full_initialization_used_by_handoff": False,
        "c0_summary_path": str(c0_summary_path),
        "c0_summary_sha256": sha256_file(c0_summary_path),
        "reference_c1_summary": str(_resolve(repo_root, config.reference_c1_summary)) if config.reference_c1_summary else "",
        "variant_audits": variant_audits,
        "variant_metrics": variant_metrics,
        "phase5_replay_commands": phase5_commands,
        "repair_ladder_if_failed": repair_ladder,
        "artifact_files": {
            "handoff_overlap_inventory": str(output_dir / "handoff_overlap_inventory.json"),
            "phase3_gate_summary": str(output_dir / "phase3_gate_summary.json"),
        },
    }
    write_json(output_dir / "phase5_replay_commands.json", phase5_commands)
    write_json(output_dir / "phase3_gate_summary.json", summary)
    return summary
