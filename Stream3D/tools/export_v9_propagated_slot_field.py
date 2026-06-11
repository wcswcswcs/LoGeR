from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.export_scannet import ScanNetExporter
from stream4d.scannet_stream import ScanNetStream
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


def _read_seq_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip() and not line.strip().startswith("#")]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _carrier_path(root: Path, carrier_run: str, scene_id: str, window_idx: int) -> Path:
    return (
        root
        / "outputs"
        / "v8_d4rt_grid_surfel_field"
        / carrier_run
        / scene_id
        / f"carriers_window{int(window_idx):03d}.npz"
    )


def _frame_ids(root: Path, carrier_run: str, scene_id: str, num_frames: int, window_idx: int) -> list[int]:
    summary_path = root / "outputs" / "v8_d4rt_grid_surfel_field" / carrier_run / "summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in payload.get("windows", []):
            if row.get("scene") == scene_id and int(row.get("window_index", -1)) == int(window_idx):
                frame_ids = [int(v) for v in row.get("frame_ids", [])]
                if len(frame_ids) == int(num_frames):
                    return frame_ids
    return list(range(int(num_frames)))


def _slot_key(src_frame_global: np.ndarray, src_mask_id: np.ndarray, keep: np.ndarray) -> list[tuple[int, int]]:
    slots = np.stack([src_frame_global[keep], src_mask_id[keep]], axis=1)
    unique = np.unique(slots, axis=0)
    return [(int(frame_id), int(mask_id)) for frame_id, mask_id in unique.tolist() if int(mask_id) > 0]


def _score(point_count: int, observed_frames: int, source_carriers: int, mode: str) -> float:
    if mode == "one":
        return 1.0
    if mode == "points":
        return float(point_count)
    if mode == "observations":
        return float(observed_frames)
    if mode == "propagated_quality":
        return float(observed_frames * math.sqrt(max(point_count, 1)) * math.sqrt(max(source_carriers, 1)))
    raise ValueError(f"Unsupported score mode: {mode}")


def export_scene(args: argparse.Namespace, root: Path, scene_id: str) -> dict[str, Any]:
    carrier_path = _carrier_path(root, args.carrier_run, scene_id, args.window_idx)
    if not carrier_path.exists():
        raise FileNotFoundError(carrier_path)
    stream = ScanNetStream(seq_name=scene_id, backbone=args.backbone)
    errors = stream.validate(require_masks=False)
    if errors:
        raise RuntimeError("; ".join(errors))
    exporter = ScanNetExporter(
        stream,
        output_config=args.output_config,
        export_nn_radius=float(args.export_nn_radius),
        export_support_mode="reuse_point_ids",
        export_min_points_per_object=int(args.min_points_per_object),
        export_score_mode="one",
    )

    with np.load(carrier_path) as data:
        carrier_id = data["carrier_id"].astype(np.int64)
        src_frame_global = data["src_frame_global"].astype(np.int64)
        src_mask_id = data["src_mask_id"].astype(np.int64)
        uv_pred = data["uv_pred"].astype(np.float32)
        valid = data["valid"].astype(bool)
        visibility_prob = data["visibility_prob"].astype(np.float32)
        confidence_prob = data["confidence_prob"].astype(np.float32)

    num_frames, num_carriers = valid.shape
    frame_ids = _frame_ids(root, args.carrier_run, scene_id, num_frames, args.window_idx)
    positive = src_mask_id > 0
    slots = _slot_key(src_frame_global, src_mask_id, positive)
    slot_to_carrier_indices: dict[tuple[int, int], np.ndarray] = {}
    for slot in slots:
        slot_keep = (src_frame_global == slot[0]) & (src_mask_id == slot[1])
        if int(np.count_nonzero(slot_keep)) >= int(args.min_source_carriers):
            slot_to_carrier_indices[slot] = np.flatnonzero(slot_keep)

    object_records: list[dict[str, Any]] = []
    object_dict: dict[int, dict[str, Any]] = {}
    point_owner_counts = np.zeros((exporter.scene_points.shape[0],), dtype=np.uint16)
    total_backproject_queries = 0
    total_backproject_hits = 0
    dropped_small_points = 0
    dropped_low_frames = 0
    observed_frame_counts: list[int] = []
    point_counts: list[int] = []

    for slot_idx, (slot, carrier_indices) in enumerate(sorted(slot_to_carrier_indices.items())):
        point_ids: set[int] = set()
        observed_frames = 0
        frame_pixel_counts: list[int] = []
        for local_idx, frame_id in enumerate(frame_ids):
            keep = np.zeros((num_carriers,), dtype=bool)
            keep[carrier_indices] = True
            keep &= valid[local_idx]
            keep &= visibility_prob[local_idx] >= float(args.visibility_threshold)
            keep &= confidence_prob[local_idx] >= float(args.confidence_threshold)
            uv = uv_pred[local_idx, keep]
            if uv.size == 0:
                continue
            in_bounds = (uv[:, 0] >= 0.0) & (uv[:, 0] <= 1.0) & (uv[:, 1] >= 0.0) & (uv[:, 1] <= 1.0)
            uv = uv[in_bounds]
            if uv.shape[0] < int(args.min_pixels_per_frame):
                continue
            depth = stream.load_depth(int(frame_id))
            height, width = depth.shape[:2]
            xs = np.clip(np.rint(uv[:, 0] * float(max(width - 1, 1))).astype(np.int64), 0, width - 1)
            ys = np.clip(np.rint(uv[:, 1] * float(max(height - 1, 1))).astype(np.int64), 0, height - 1)
            xy = np.unique(np.stack([xs, ys], axis=1), axis=0)
            if args.max_pixels_per_frame > 0 and xy.shape[0] > int(args.max_pixels_per_frame):
                keep_idx = np.linspace(0, xy.shape[0] - 1, num=int(args.max_pixels_per_frame), dtype=np.int64)
                xy = xy[keep_idx]
            if xy.shape[0] < int(args.min_pixels_per_frame):
                continue
            hit_ids, _ = exporter._backproject_xy(int(frame_id), xy, nn_radius=float(args.export_nn_radius))
            total_backproject_queries += int(xy.shape[0])
            total_backproject_hits += int(hit_ids.shape[0])
            if hit_ids.size == 0:
                continue
            observed_frames += 1
            frame_pixel_counts.append(int(xy.shape[0]))
            point_ids.update(int(v) for v in hit_ids.tolist())
        if observed_frames < int(args.min_observed_frames):
            dropped_low_frames += 1
            continue
        if len(point_ids) < int(args.min_points_per_object):
            dropped_small_points += 1
            continue
        object_id = len(object_records)
        ids = np.fromiter(point_ids, dtype=np.int64)
        point_owner_counts[ids] += 1
        observed_frame_counts.append(int(observed_frames))
        point_counts.append(int(len(point_ids)))
        score = _score(len(point_ids), observed_frames, int(carrier_indices.shape[0]), args.score_mode)
        object_records.append(
            {
                "object_id": object_id,
                "point_ids": set(int(v) for v in point_ids),
                "score": float(score),
                "area_score": float(len(point_ids)),
                "observations": float(observed_frames),
                "carrier_count": float(carrier_indices.shape[0]),
                "reliability": float(score),
            }
        )
        object_dict[object_id] = {
            "point_ids": np.asarray(sorted(point_ids), dtype=np.int64),
            "mask_list": [(int(slot[0]), int(slot[1]), 1.0)],
            "repre_mask_list": [(int(slot[0]), int(slot[1]), 1.0)],
            "carrier_ids": carrier_id[carrier_indices],
            "source_slot": {"frame_id": int(slot[0]), "mask_id": int(slot[1])},
            "observed_frames": int(observed_frames),
            "frame_pixel_count_mean": float(np.mean(frame_pixel_counts)) if frame_pixel_counts else 0.0,
        }

    kept_records = [record for record in object_records if len(record["point_ids"]) >= int(args.min_points_per_object)]
    masks = np.zeros((exporter.scene_points.shape[0], len(kept_records)), dtype=bool)
    scores = np.zeros((len(kept_records),), dtype=np.float32)
    classes = np.zeros((len(kept_records),), dtype=np.int32)
    kept_object_dict: dict[int, dict[str, Any]] = {}
    for out_idx, record in enumerate(kept_records):
        ids = np.fromiter(record["point_ids"], dtype=np.int64)
        masks[ids, out_idx] = True
        scores[out_idx] = float(record["score"])
        kept_object_dict[out_idx] = object_dict[int(record["object_id"])]

    pred_dir = root / "data" / "prediction" / f"{args.output_config}_class_agnostic"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_dir / f"{scene_id}.npz",
        pred_masks=masks,
        pred_score=scores,
        pred_classes=classes,
    )
    tmp_dir = root / "data" / "TMP" / args.output_config
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pre_points = np.flatnonzero(masks.any(axis=1)).astype(np.int64) if masks.shape[1] else np.zeros((0,), dtype=np.int64)
    np.save(tmp_dir / f"{scene_id}_pre_points.npy", pre_points)
    object_dir = stream.object_dir / args.output_config
    object_dir.mkdir(parents=True, exist_ok=True)
    np.save(object_dir / "object_dict.npy", kept_object_dict, allow_pickle=True)

    conflict_points = int(np.count_nonzero(masks.sum(axis=1) > 1)) if masks.shape[1] else 0
    return {
        "seq_name": scene_id,
        "carrier_path": str(carrier_path),
        "num_frames": int(num_frames),
        "num_carriers": int(num_carriers),
        "num_positive_slots": int(len(slots)),
        "num_candidate_slots": int(len(slot_to_carrier_indices)),
        "num_exported_objects": int(len(kept_records)),
        "num_exported_points": int(pre_points.shape[0]),
        "num_scene_points": int(exporter.scene_points.shape[0]),
        "pre_points_ratio": float(pre_points.shape[0] / max(exporter.scene_points.shape[0], 1)),
        "export_conflict_rate": float(conflict_points / max(pre_points.shape[0], 1)),
        "backproject_queries": int(total_backproject_queries),
        "backproject_hits": int(total_backproject_hits),
        "backproject_hit_rate": float(total_backproject_hits / max(total_backproject_queries, 1)),
        "dropped_small_points": int(dropped_small_points),
        "dropped_low_frames": int(dropped_low_frames),
        "observed_frames_mean": float(np.mean(observed_frame_counts)) if observed_frame_counts else 0.0,
        "observed_frames_min": int(min(observed_frame_counts)) if observed_frame_counts else 0,
        "observed_frames_max": int(max(observed_frame_counts)) if observed_frame_counts else 0,
        "points_per_object_mean": float(np.mean(point_counts)) if point_counts else 0.0,
    }


def _aggregate(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    numeric_keys = sorted({k for row in rows for k, v in row.items() if isinstance(v, (int, float, np.generic))})
    means = {
        f"{key}_mean": float(np.mean([float(row[key]) for row in rows if key in row]))
        for key in numeric_keys
        if any(key in row for row in rows)
    }
    return {
        "algorithm": "v9_propagated_slot_field",
        "uses_gt": False,
        "is_method_result": True,
        "output_config": args.output_config,
        "carrier_run": args.carrier_run,
        "num_scenes": len(rows),
        **means,
    }


def _write_summary(root: Path, args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    out_dir = root / args.summary_root
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate = _aggregate(rows, args)
    payload = {"args": vars(args), "aggregate": aggregate, "rows": rows}
    prefix = out_dir / f"{args.output_config}_summary"
    prefix.with_suffix(".json").write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    fieldnames = ["seq_name"] + sorted({k for row in rows for k in row if k != "seq_name"})
    with prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    lines = [
        f"# {args.output_config}",
        "",
        "| scene | objects | points | pre% | conflict | obs frames | backproject hit | slots |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seq_name']} | {row['num_exported_objects']} | {row['num_exported_points']} | "
            f"{row['pre_points_ratio']:.6f} | {row['export_conflict_rate']:.6f} | "
            f"{row['observed_frames_mean']:.3f} | {row['backproject_hit_rate']:.6f} | "
            f"{row['num_candidate_slots']} |"
        )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = build_prediction_manifest(
        root=root,
        output_config=args.output_config,
        is_method_result=True,
        is_diagnostic_only=False,
        uses_gt=False,
        gt_usage="none",
        source_configs=[args.carrier_run],
        pre_points_policy="recompute",
        support_policy="d4rt_propagated_slot_pixels_backproject",
        notes=(
            "v9 O5 prototype: source 2D mask slots are propagated through D4RT carrier uv tracks, "
            "then propagated pixels are backprojected to ScanNet mesh. No GT is read."
        ),
        extra={
            "algorithm": "v9_propagated_slot_field",
            "eval_policy": args.eval_policy,
            "carrier_run": args.carrier_run,
            "summary_path": str(prefix.with_suffix(".json")),
            "visibility_threshold": float(args.visibility_threshold),
            "confidence_threshold": float(args.confidence_threshold),
            "min_observed_frames": int(args.min_observed_frames),
        },
    )
    write_prediction_manifest(args.output_config, manifest, root=root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a D4RT propagated source-mask slot field.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--carrier-run", required=True)
    parser.add_argument("--window-idx", type=int, default=0)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--visibility-threshold", type=float, default=0.5)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--min-source-carriers", type=int, default=8)
    parser.add_argument("--min-pixels-per-frame", type=int, default=4)
    parser.add_argument("--min-observed-frames", type=int, default=2)
    parser.add_argument("--min-points-per-object", type=int, default=20)
    parser.add_argument("--max-pixels-per-frame", type=int, default=20000)
    parser.add_argument("--export-nn-radius", type=float, default=0.05)
    parser.add_argument(
        "--score-mode",
        default="propagated_quality",
        choices=["one", "points", "observations", "propagated_quality"],
    )
    parser.add_argument("--summary-root", default="outputs/v9_propagated_slot_field")
    parser.add_argument("--eval-policy", default="own_recompute_d4rt_propagated_slot_field")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    rows = [export_scene(args, root, scene_id) for scene_id in _read_seq_list(root / args.seq_list)]
    _write_summary(root, args, rows)
    print(json.dumps(_json_safe({"output_config": args.output_config, "rows": rows}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
