from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from tools.export_trackbucket_dense_variants_v7 import VARIANTS, Variant, _overlap_value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _read_seq_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _prediction_dir(root: Path, config: str) -> Path:
    return root / "data" / "prediction" / f"{config}_class_agnostic"


def _tmp_path(root: Path, config: str, scene_id: str) -> Path:
    candidates = [
        root / "data" / "TMP" / config / f"{scene_id}_pre_points.npy",
        root / "TMP" / config / f"{scene_id}_pre_points.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _load_full_masks(root: Path, config: str, scene_id: str, scene_vertices: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_path = _prediction_dir(root, config) / f"{scene_id}.npz"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    with np.load(pred_path) as pred:
        masks = np.asarray(pred["pred_masks"], dtype=bool)
        scores = np.asarray(pred["pred_score"], dtype=np.float32)
        classes = np.asarray(pred["pred_classes"], dtype=np.int32)
    if masks.shape[0] == scene_vertices:
        return masks, scores, classes
    tmp_path = _tmp_path(root, config, scene_id)
    if not tmp_path.exists():
        raise FileNotFoundError(tmp_path)
    pre_points = np.load(tmp_path).astype(np.int64)
    if masks.shape[0] != pre_points.shape[0]:
        raise ValueError(
            f"{config}/{scene_id}: pred mask first dim {masks.shape[0]} does not match "
            f"scene vertices {scene_vertices} or TMP length {pre_points.shape[0]}"
        )
    full = np.zeros((scene_vertices, masks.shape[1]), dtype=bool)
    full[pre_points, :] = masks
    return full, scores, classes


def _best_gt_for_masks(masks: np.ndarray, gt_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray], dict[int, int]]:
    pred_areas = masks.sum(axis=0).astype(np.int64)
    gt_instance_ids = [int(v) for v in np.unique(gt_ids[gt_ids >= 1000]).tolist()]
    gt_masks = {instance_id: gt_ids == int(instance_id) for instance_id in gt_instance_ids}
    gt_areas = {instance_id: int(np.count_nonzero(mask)) for instance_id, mask in gt_masks.items()}
    best_iou = np.zeros((masks.shape[1],), dtype=np.float32)
    best_gt = np.full((masks.shape[1],), -1, dtype=np.int64)
    iou_by_gt: dict[int, np.ndarray] = {}
    for instance_id, gt_mask in gt_masks.items():
        gt_area = gt_areas[instance_id]
        intersections = masks[gt_mask].sum(axis=0).astype(np.int64) if masks.shape[1] else np.zeros((0,), dtype=np.int64)
        unions = gt_area + pred_areas - intersections
        ious = intersections / np.maximum(unions, 1)
        iou_by_gt[int(instance_id)] = ious.astype(np.float32)
        update = ious > best_iou
        best_iou[update] = ious[update]
        best_gt[update] = int(instance_id)
    return best_iou, best_gt, iou_by_gt, gt_areas


def _suppression_records(
    masks: np.ndarray,
    scores: np.ndarray,
    variant: Variant,
    track_masks: np.ndarray,
    gt_ids: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    primary_areas = np.asarray(masks.sum(axis=0), dtype=np.int64)
    track_areas = np.asarray(track_masks.sum(axis=0), dtype=np.int64)
    inter = masks.T.astype(np.uint8) @ track_masks.astype(np.uint8)
    best_track = np.asarray(inter.argmax(axis=1), dtype=np.int64) if track_masks.shape[1] else np.full(masks.shape[1], -1)
    best_inter = np.asarray(inter.max(axis=1), dtype=np.float32) if track_masks.shape[1] else np.zeros(masks.shape[1], dtype=np.float32)
    best_primary_ioc = best_inter / np.maximum(primary_areas.astype(np.float32), 1.0)
    best_track_ioc = best_inter / np.maximum(track_areas[np.maximum(best_track, 0)].astype(np.float32), 1.0) if track_masks.shape[1] else np.zeros_like(best_primary_ioc)

    buckets: dict[int, list[int]] = {}
    for idx, (track_idx, value) in enumerate(zip(best_track, best_primary_ioc)):
        if track_idx >= 0 and float(value) >= variant.assign_primary_ioc_min:
            buckets.setdefault(int(track_idx), []).append(int(idx))

    best_gt_iou, best_gt, iou_by_gt, _ = _best_gt_for_masks(masks, gt_ids)
    records: list[dict[str, Any]] = []
    keep = np.ones(masks.shape[1], dtype=bool)

    for bucket_id, indices in sorted(buckets.items()):
        order = sorted(indices, key=lambda idx: (-float(scores[idx]), -int(primary_areas[idx]), int(idx)))
        kept_in_bucket: list[int] = []
        for idx in order:
            suppressor = -1
            suppress_overlap = 0.0
            for kept_idx in kept_in_bucket:
                pair_inter = int(np.logical_and(masks[:, idx], masks[:, kept_idx]).sum())
                overlap = _overlap_value(pair_inter, int(primary_areas[idx]), int(primary_areas[kept_idx]), variant.overlap_mode)
                if overlap >= variant.overlap_threshold:
                    suppressor = int(kept_idx)
                    suppress_overlap = float(overlap)
                    break
            if suppressor >= 0:
                keep[idx] = False
                target_gt = int(best_gt[idx])
                suppressor_iou_to_suppressed_gt = float(iou_by_gt.get(target_gt, np.zeros_like(best_gt_iou))[suppressor]) if target_gt >= 0 else 0.0
                suppressed_best = float(best_gt_iou[idx])
                suppressor_best = float(best_gt_iou[suppressor])
                same_best_gt = bool(target_gt >= 0 and target_gt == int(best_gt[suppressor]))
                harmful_iou25 = bool(suppressed_best >= 0.25 and suppressor_iou_to_suppressed_gt < 0.25)
                harmful_iou50 = bool(suppressed_best >= 0.50 and suppressor_iou_to_suppressed_gt < 0.50)
                safe_duplicate_like = bool(
                    suppressed_best < 0.25
                    or (same_best_gt and suppressor_iou_to_suppressed_gt >= max(0.25, suppressed_best - 0.05))
                )
                records.append(
                    {
                        "suppressed_idx": int(idx),
                        "suppressor_idx": int(suppressor),
                        "bucket": int(bucket_id),
                        "score": float(scores[idx]),
                        "suppressor_score": float(scores[suppressor]),
                        "area": int(primary_areas[idx]),
                        "suppressor_area": int(primary_areas[suppressor]),
                        "bucket_overlap": float(suppress_overlap),
                        "best_primary_ioc": float(best_primary_ioc[idx]),
                        "best_track_ioc": float(best_track_ioc[idx]),
                        "best_gt_id": target_gt,
                        "best_gt_iou": suppressed_best,
                        "suppressor_best_gt_id": int(best_gt[suppressor]),
                        "suppressor_best_gt_iou": suppressor_best,
                        "suppressor_iou_to_suppressed_best_gt": suppressor_iou_to_suppressed_gt,
                        "same_best_gt": same_best_gt,
                        "harmful_iou25": harmful_iou25,
                        "harmful_iou50": harmful_iou50,
                        "safe_duplicate_like": safe_duplicate_like,
                    }
                )
            else:
                kept_in_bucket.append(int(idx))

    meta = {
        "num_instances_before": int(masks.shape[1]),
        "num_instances_after": int(np.count_nonzero(keep)),
        "num_suppressed": int(len(records)),
        "num_assigned_to_track": int(sum(len(v) for v in buckets.values())),
        "num_buckets": int(len(buckets)),
        "bucket_size_mean": float(mean([len(v) for v in buckets.values()])) if buckets else 0.0,
        "bucket_size_max": int(max([len(v) for v in buckets.values()])) if buckets else 0,
    }
    return records, meta


def _aggregate_variant(output_config: str, rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(key: str, default: float = 0.0) -> float:
        values = [float(row[key]) for row in rows if key in row]
        return float(mean(values)) if values else default

    def frac(key: str) -> float:
        return float(sum(1 for record in records if bool(record.get(key))) / max(len(records), 1))

    best_ious = [float(record["best_gt_iou"]) for record in records]
    return {
        "output_config": output_config,
        "num_scenes": int(len(rows)),
        "num_suppressed_total": int(len(records)),
        "num_suppressed_mean": avg("num_suppressed"),
        "num_assigned_to_track_mean": avg("num_assigned_to_track"),
        "num_buckets_mean": avg("num_buckets"),
        "bucket_size_mean": avg("bucket_size_mean"),
        "bucket_size_max_mean": avg("bucket_size_max"),
        "suppressed_best_gt_iou_mean": float(mean(best_ious)) if best_ious else 0.0,
        "suppressed_best_gt_iou_ge_0p25": float(sum(v >= 0.25 for v in best_ious) / max(len(best_ious), 1)),
        "suppressed_best_gt_iou_ge_0p50": float(sum(v >= 0.50 for v in best_ious) / max(len(best_ious), 1)),
        "same_best_gt_fraction": frac("same_best_gt"),
        "harmful_iou25_fraction": frac("harmful_iou25"),
        "harmful_iou50_fraction": frac("harmful_iou50"),
        "safe_duplicate_like_fraction": frac("safe_duplicate_like"),
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Track-Bucket Suppression Diagnostic",
        "",
        "- diagnostic_only: `True`",
        "- uses_gt: `True`",
        "- purpose: error analysis only; not a method result and not a model-selection table.",
        "",
        "## Summary",
        "",
        "| Config | suppressed total/mean | assigned mean | suppressed GT IoU mean | GT IoU >=25/50 | same GT | harmful@25/50 | safe duplicate-like |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregates"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["output_config"],
                    f"{row['num_suppressed_total']} / {row['num_suppressed_mean']:.2f}",
                    f"{row['num_assigned_to_track_mean']:.2f}",
                    f"{row['suppressed_best_gt_iou_mean']:.4f}",
                    f"{row['suppressed_best_gt_iou_ge_0p25']:.4f}/{row['suppressed_best_gt_iou_ge_0p50']:.4f}",
                    f"{row['same_best_gt_fraction']:.4f}",
                    f"{row['harmful_iou25_fraction']:.4f}/{row['harmful_iou50_fraction']:.4f}",
                    f"{row['safe_duplicate_like_fraction']:.4f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guard",
            "",
            "This diagnostic reads GT ids to classify suppression mistakes. It must not enter reportable method tables.",
            "A harmful@25 record means the suppressed mask had best GT IoU >= 0.25 while its suppressor had IoU < 0.25 to that same GT.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--groups", nargs="+", choices=sorted(VARIANTS), default=sorted(VARIANTS))
    parser.add_argument("--configs", default="", help="Optional comma-separated output_config allowlist.")
    parser.add_argument("--track-config", default="stream4d_v7_c8p_posttrack_owned_wta_probe5")
    parser.add_argument("--output-prefix", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    seq_names = _read_seq_list(root / args.seq_list)
    all_scene_rows: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    allowlist = {item.strip() for item in str(args.configs).split(",") if item.strip()}

    for group in args.groups:
        for variant in VARIANTS[group]:
            if allowlist and variant.output_config not in allowlist:
                continue
            print(f"[trackbucket-diagnostic] processing {variant.output_config}", flush=True)
            variant_rows: list[dict[str, Any]] = []
            variant_records: list[dict[str, Any]] = []
            for scene_id in seq_names:
                gt_path = root / "data" / "scannet" / "gt" / f"{scene_id}.txt"
                gt_ids = np.loadtxt(gt_path).astype(np.int64)
                masks, scores, _ = _load_full_masks(root, variant.primary_config, scene_id, int(gt_ids.shape[0]))
                track_masks, _, _ = _load_full_masks(root, args.track_config, scene_id, int(gt_ids.shape[0]))
                records, meta = _suppression_records(masks, scores, variant, track_masks, gt_ids)
                row = {
                    "group": group,
                    "output_config": variant.output_config,
                    "scene_id": scene_id,
                    **meta,
                }
                variant_rows.append(row)
                for record in records:
                    variant_records.append({"group": group, "output_config": variant.output_config, "scene_id": scene_id, **record})
            all_scene_rows.extend(variant_rows)
            all_records.extend(variant_records)
            aggregates.append(_aggregate_variant(variant.output_config, variant_rows, variant_records))

    payload = {
        "diagnostic_only": True,
        "uses_gt": True,
        "args": vars(args),
        "aggregates": aggregates,
        "scene_rows": all_scene_rows,
        "suppression_records": all_records,
    }
    prefix = root / args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in all_records for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    _write_markdown(prefix.with_suffix(".md"), payload)
    print(f"[trackbucket-diagnostic] wrote {prefix.with_suffix('.json')}")
    print(f"[trackbucket-diagnostic] wrote {prefix.with_suffix('.md')}")
    print(f"[trackbucket-diagnostic] wrote {prefix.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
