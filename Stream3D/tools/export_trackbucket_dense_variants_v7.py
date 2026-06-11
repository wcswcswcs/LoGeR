from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_TRACK_CONFIG = "stream4d_v7_c8p_posttrack_owned_wta_probe5"
COMPACT_PRIMARY = "stream4d_v6_e4_probe5_objcomp_m670_g101_compact_only_preserve"
SCOREUNIQUE_PRIMARY = "stream4d_v6_e4_probe5_objcomp_m670_g101_score_unique_compact_preserve"


@dataclass(frozen=True)
class Variant:
    output_config: str
    primary_config: str
    implementation: str
    assign_primary_ioc_min: float
    overlap_mode: str
    overlap_threshold: float
    notes: str


VARIANTS: dict[str, list[Variant]] = {
    "c11": [
        Variant("stream4d_v7_c11a_compact_trackbucket_minioc090_a050", COMPACT_PRIMARY, "sparse_primary_union_overlap", 0.50, "min_ioc", 0.90, "Dense masks/TMP inherited from primary; C8P carrier track only buckets dense candidates for within-track duplicate NMS. Overlap is computed on primary dense union only. No C8P masks are output. No GT used."),
        Variant("stream4d_v7_c11b_compact_trackbucket_minioc075_a050", COMPACT_PRIMARY, "sparse_primary_union_overlap", 0.50, "min_ioc", 0.75, "Dense masks/TMP inherited from primary; C8P carrier track only buckets dense candidates for within-track duplicate NMS. Overlap is computed on primary dense union only. No C8P masks are output. No GT used."),
        Variant("stream4d_v7_c11c_compact_trackbucket_iou050_a050", COMPACT_PRIMARY, "sparse_primary_union_overlap", 0.50, "iou", 0.50, "Dense masks/TMP inherited from primary; C8P carrier track only buckets dense candidates for within-track duplicate NMS. Overlap is computed on primary dense union only. No C8P masks are output. No GT used."),
        Variant("stream4d_v7_c11d_scoreunique_trackbucket_minioc090_a050", SCOREUNIQUE_PRIMARY, "sparse_primary_union_overlap", 0.50, "min_ioc", 0.90, "Dense masks/TMP inherited from primary; C8P carrier track only buckets dense candidates for within-track duplicate NMS. Overlap is computed on primary dense union only. No C8P masks are output. No GT used."),
        Variant("stream4d_v7_c11e_scoreunique_trackbucket_minioc075_a050", SCOREUNIQUE_PRIMARY, "sparse_primary_union_overlap", 0.50, "min_ioc", 0.75, "Dense masks/TMP inherited from primary; C8P carrier track only buckets dense candidates for within-track duplicate NMS. Overlap is computed on primary dense union only. No C8P masks are output. No GT used."),
    ],
    "c12": [
        Variant("stream4d_v7_c12a_compact_trackbucket_minioc099_a090", COMPACT_PRIMARY, "sparse_primary_union_overlap_conservative", 0.90, "min_ioc", 0.99, "Dense masks/TMP inherited from primary; conservative C8P track bucket duplicate NMS with high assignment/overlap thresholds. No C8P masks are output. No GT used."),
        Variant("stream4d_v7_c12b_compact_trackbucket_iou085_a090", COMPACT_PRIMARY, "sparse_primary_union_overlap_conservative", 0.90, "iou", 0.85, "Dense masks/TMP inherited from primary; conservative C8P track bucket duplicate NMS with high assignment/overlap thresholds. No C8P masks are output. No GT used."),
        Variant("stream4d_v7_c12c_compact_trackbucket_minioc099_a095", COMPACT_PRIMARY, "sparse_primary_union_overlap_conservative", 0.95, "min_ioc", 0.99, "Dense masks/TMP inherited from primary; conservative C8P track bucket duplicate NMS with high assignment/overlap thresholds. No C8P masks are output. No GT used."),
        Variant("stream4d_v7_c12d_scoreunique_trackbucket_minioc099_a090", SCOREUNIQUE_PRIMARY, "sparse_primary_union_overlap_conservative", 0.90, "min_ioc", 0.99, "Dense masks/TMP inherited from primary; conservative C8P track bucket duplicate NMS with high assignment/overlap thresholds. No C8P masks are output. No GT used."),
        Variant("stream4d_v7_c12e_scoreunique_trackbucket_iou085_a090", SCOREUNIQUE_PRIMARY, "sparse_primary_union_overlap_conservative", 0.90, "iou", 0.85, "Dense masks/TMP inherited from primary; conservative C8P track bucket duplicate NMS with high assignment/overlap thresholds. No C8P masks are output. No GT used."),
    ],
    "c13": [
        Variant("stream4d_v7_c13a_compact_trackbucket_iou090_a090", COMPACT_PRIMARY, "sparse_primary_union_overlap_conservative_iou_grid", 0.90, "iou", 0.90, "Dense masks/TMP inherited from primary; C8P track only buckets dense candidates for conservative within-track IoU duplicate suppression. No C8P masks are output. No GT used."),
        Variant("stream4d_v7_c13b_compact_trackbucket_iou095_a090", COMPACT_PRIMARY, "sparse_primary_union_overlap_conservative_iou_grid", 0.90, "iou", 0.95, "Dense masks/TMP inherited from primary; C8P track only buckets dense candidates for conservative within-track IoU duplicate suppression. No C8P masks are output. No GT used."),
        Variant("stream4d_v7_c13c_compact_trackbucket_iou090_a095", COMPACT_PRIMARY, "sparse_primary_union_overlap_conservative_iou_grid", 0.95, "iou", 0.90, "Dense masks/TMP inherited from primary; C8P track only buckets dense candidates for conservative within-track IoU duplicate suppression. No C8P masks are output. No GT used."),
        Variant("stream4d_v7_c13d_compact_trackbucket_iou095_a095", COMPACT_PRIMARY, "sparse_primary_union_overlap_conservative_iou_grid", 0.95, "iou", 0.95, "Dense masks/TMP inherited from primary; C8P track only buckets dense candidates for conservative within-track IoU duplicate suppression. No C8P masks are output. No GT used."),
        Variant("stream4d_v7_c13e_scoreunique_trackbucket_iou090_a090", SCOREUNIQUE_PRIMARY, "sparse_primary_union_overlap_conservative_iou_grid", 0.90, "iou", 0.90, "Dense masks/TMP inherited from primary; C8P track only buckets dense candidates for conservative within-track IoU duplicate suppression. No C8P masks are output. No GT used."),
        Variant("stream4d_v7_c13f_scoreunique_trackbucket_iou095_a090", SCOREUNIQUE_PRIMARY, "sparse_primary_union_overlap_conservative_iou_grid", 0.90, "iou", 0.95, "Dense masks/TMP inherited from primary; C8P track only buckets dense candidates for conservative within-track IoU duplicate suppression. No C8P masks are output. No GT used."),
    ],
}


def _config_pred_dir(root: Path, config: str) -> Path:
    return root / "data" / "prediction" / f"{config}_class_agnostic"


def _copy_tmp(root: Path, src_config: str, dst_config: str, seq_names: list[str]) -> None:
    src_root = root / "data" / "TMP" / src_config
    dst_root = root / "data" / "TMP" / dst_config
    dst_root.mkdir(parents=True, exist_ok=True)
    for seq_name in seq_names:
        src = src_root / f"{seq_name}_pre_points.npy"
        dst = dst_root / f"{seq_name}_pre_points.npy"
        if src.exists():
            shutil.copy2(src, dst)


def _read_seq_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_masks(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return (
            np.asarray(data["pred_masks"], dtype=bool),
            np.asarray(data["pred_score"]),
            np.asarray(data["pred_classes"]),
        )


def _overlap_value(inter: int, area_a: int, area_b: int, mode: str) -> float:
    if inter <= 0:
        return 0.0
    if mode == "min_ioc":
        denom = max(1, min(area_a, area_b))
    elif mode == "iou":
        denom = max(1, area_a + area_b - inter)
    else:
        raise ValueError(f"unsupported overlap mode: {mode}")
    return float(inter / denom)


def _nms_within_buckets(
    masks: np.ndarray,
    scores: np.ndarray,
    buckets: dict[int, list[int]],
    *,
    overlap_mode: str,
    overlap_threshold: float,
) -> tuple[list[int], list[dict[str, Any]]]:
    areas = np.asarray(masks.sum(axis=0), dtype=np.int64)
    keep = np.ones(masks.shape[1], dtype=bool)
    suppressed_preview: list[dict[str, Any]] = []

    for bucket_id, indices in sorted(buckets.items()):
        order = sorted(indices, key=lambda idx: (-float(scores[idx]), -int(areas[idx]), int(idx)))
        kept_in_bucket: list[int] = []
        for idx in order:
            suppress = False
            suppress_overlap = 0.0
            for kept_idx in kept_in_bucket:
                inter = int(np.logical_and(masks[:, idx], masks[:, kept_idx]).sum())
                overlap = _overlap_value(inter, int(areas[idx]), int(areas[kept_idx]), overlap_mode)
                if overlap >= overlap_threshold:
                    suppress = True
                    suppress_overlap = overlap
                    break
            if suppress:
                keep[idx] = False
                if len(suppressed_preview) < 10:
                    suppressed_preview.append(
                        {
                            "idx": int(idx),
                            "bucket": int(bucket_id),
                            "score": float(scores[idx]),
                            "area": float(areas[idx]),
                            "overlap": float(suppress_overlap),
                        }
                    )
            else:
                kept_in_bucket.append(idx)

    kept = [int(idx) for idx in np.flatnonzero(keep)]
    return kept, suppressed_preview


def _process_scene(root: Path, variant: Variant, track_config: str, seq_name: str) -> dict[str, Any]:
    primary_path = _config_pred_dir(root, variant.primary_config) / f"{seq_name}.npz"
    track_path = _config_pred_dir(root, track_config) / f"{seq_name}.npz"
    masks, scores, classes = _load_masks(primary_path)
    track_masks, _, _ = _load_masks(track_path)

    primary_areas = np.asarray(masks.sum(axis=0), dtype=np.float32)
    track_areas = np.asarray(track_masks.sum(axis=0), dtype=np.float32)
    inter = masks.T.astype(np.uint8) @ track_masks.astype(np.uint8)
    best_track = np.asarray(inter.argmax(axis=1), dtype=np.int64) if track_masks.shape[1] else np.full(masks.shape[1], -1)
    best_inter = np.asarray(inter.max(axis=1), dtype=np.float32) if track_masks.shape[1] else np.zeros(masks.shape[1], dtype=np.float32)
    best_primary_ioc = best_inter / np.maximum(primary_areas, 1.0)

    buckets: dict[int, list[int]] = {}
    for idx, (track_idx, value) in enumerate(zip(best_track, best_primary_ioc)):
        if track_idx >= 0 and float(value) >= variant.assign_primary_ioc_min:
            buckets.setdefault(int(track_idx), []).append(int(idx))

    kept, suppressed_preview = _nms_within_buckets(
        masks,
        scores,
        buckets,
        overlap_mode=variant.overlap_mode,
        overlap_threshold=variant.overlap_threshold,
    )
    kept_arr = np.asarray(kept, dtype=np.int64)
    out_masks = masks[:, kept_arr]
    out_scores = scores[kept_arr]
    out_classes = classes[kept_arr]

    out_dir = _config_pred_dir(root, variant.output_config)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{seq_name}.npz", pred_masks=out_masks, pred_score=out_scores, pred_classes=out_classes)

    bucket_sizes = [len(v) for v in buckets.values()]
    return {
        "seq_name": seq_name,
        "primary_config": variant.primary_config,
        "track_config": track_config,
        "assign_primary_ioc_min": variant.assign_primary_ioc_min,
        "overlap_mode": variant.overlap_mode,
        "overlap_threshold": variant.overlap_threshold,
        "num_instances_before": int(masks.shape[1]),
        "num_instances_after": int(out_masks.shape[1]),
        "num_suppressed": int(masks.shape[1] - out_masks.shape[1]),
        "num_assigned_to_track": int(sum(bucket_sizes)),
        "num_buckets": int(len(bucket_sizes)),
        "bucket_size_mean": float(np.mean(bucket_sizes)) if bucket_sizes else 0.0,
        "bucket_size_max": int(max(bucket_sizes)) if bucket_sizes else 0,
        "best_primary_ioc_mean": float(np.mean(best_primary_ioc)) if best_primary_ioc.size else 0.0,
        "best_primary_ioc_p50": float(np.percentile(best_primary_ioc, 50)) if best_primary_ioc.size else 0.0,
        "best_primary_ioc_p90": float(np.percentile(best_primary_ioc, 90)) if best_primary_ioc.size else 0.0,
        "primary_union_count": int(np.count_nonzero(masks.any(axis=1))) if masks.shape[1] else 0,
        "union_count_before": int(np.count_nonzero(masks.any(axis=1))) if masks.shape[1] else 0,
        "union_count_after": int(np.count_nonzero(out_masks.any(axis=1))) if out_masks.shape[1] else 0,
        "suppressed_preview": suppressed_preview,
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    return float(np.mean(values)) if values else 0.0


def _write_manifest(root: Path, variant: Variant, track_config: str) -> None:
    manifest = {
        "schema_version": "stream4d_prediction_manifest_v1",
        "output_config": variant.output_config,
        "source_configs": [variant.primary_config, track_config],
        "primary_config": variant.primary_config,
        "track_config": track_config,
        "implementation": variant.implementation,
        "support_policy": f"{variant.implementation}:assign_primary_ioc>={variant.assign_primary_ioc_min}:"
        f"{variant.overlap_mode}@{variant.overlap_threshold}",
        "pre_points_policy": "inherit_primary_tmp",
        "assign_primary_ioc_min": variant.assign_primary_ioc_min,
        "overlap_mode": variant.overlap_mode,
        "overlap_threshold": variant.overlap_threshold,
        "uses_gt": False,
        "gt_usage": "none",
        "is_method_result": True,
        "is_diagnostic_only": False,
        "notes": variant.notes,
        "command": "python -m tools.export_trackbucket_dense_variants_v7",
        "cwd": str(root),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path = _config_pred_dir(root, variant.output_config) / "config_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_group(root: Path, seq_names: list[str], group: str, track_config: str) -> None:
    summary_root = root / "outputs" / {
        "c11": "stream4d_v7_c11_track_bucket_nms",
        "c12": "stream4d_v7_c12_conservative_track_bucket",
        "c13": "stream4d_v7_c13_conservative_iou_grid",
    }[group]
    summary_root.mkdir(parents=True, exist_ok=True)

    for variant in VARIANTS[group]:
        rows = [_process_scene(root, variant, track_config, seq_name) for seq_name in seq_names]
        _copy_tmp(root, variant.primary_config, variant.output_config, seq_names)
        _write_manifest(root, variant, track_config)

        aggregate_keys = [
            "assign_primary_ioc_min",
            "best_primary_ioc_mean",
            "best_primary_ioc_p50",
            "best_primary_ioc_p90",
            "bucket_size_max",
            "bucket_size_mean",
            "num_assigned_to_track",
            "num_buckets",
            "num_instances_after",
            "num_instances_before",
            "num_suppressed",
            "overlap_threshold",
            "primary_union_count",
            "union_count_after",
            "union_count_before",
        ]
        summary = {
            "aggregate": {f"mean_{key}": _mean(rows, key) for key in aggregate_keys},
            "output_config": variant.output_config,
            "primary_config": variant.primary_config,
            "track_config": track_config,
            "implementation": variant.implementation,
            "assign_primary_ioc_min": variant.assign_primary_ioc_min,
            "overlap_mode": variant.overlap_mode,
            "overlap_threshold": variant.overlap_threshold,
            "rows": rows,
        }
        (summary_root / f"{variant.output_config}_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[trackbucket-generate] wrote {variant.output_config}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate C11/C12/C13 dense-primary track-bucket suppression variants.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--seq-list", type=Path, default=Path("splits/scannet_v6_probe5.txt"))
    parser.add_argument("--groups", nargs="+", choices=sorted(VARIANTS), default=sorted(VARIANTS))
    parser.add_argument("--track-config", default=DEFAULT_TRACK_CONFIG)
    args = parser.parse_args()

    root = args.root.resolve()
    seq_list = args.seq_list if args.seq_list.is_absolute() else root / args.seq_list
    seq_names = _read_seq_list(seq_list)
    for group in args.groups:
        run_group(root, seq_names, group, args.track_config)


if __name__ == "__main__":
    main()
