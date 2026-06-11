from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.export_scannet import ScanNetExporter
from stream4d.scannet_stream import ScanNetStream
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_unique_observations(debug_root: Path, seq_name: str) -> list[tuple[int, int, float]]:
    seq_dir = debug_root / seq_name
    if not seq_dir.exists():
        raise FileNotFoundError(seq_dir)
    best: dict[tuple[int, int], float] = {}
    raw = 0
    for window_path in sorted(seq_dir.glob("local_props_window*.json")):
        with window_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for prop in payload.get("proposals", []):
            for item in prop.get("mask_observations", []):
                raw += 1
                key = (int(item["frame_id"]), int(item["mask_id"]))
                best[key] = max(float(item.get("coverage", 0.0)), best.get(key, 0.0))
    observations = [(frame_id, mask_id, coverage) for (frame_id, mask_id), coverage in best.items()]
    observations.sort(key=lambda item: (item[2], -item[0], -item[1]), reverse=True)
    return observations


def main() -> None:
    parser = argparse.ArgumentParser(description="Export unique 2D mask-observation bank from local_props debug files.")
    parser.add_argument("--debug-root", required=True)
    parser.add_argument("--seq-name", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--min-coverage", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--export-nn-radius", type=float, default=0.05)
    parser.add_argument("--min-points-per-mask", type=int, default=100)
    parser.add_argument("--summary-root", default="outputs/mask_observation_bank")
    args = parser.parse_args()

    observations_all = _load_unique_observations(Path(args.debug_root), args.seq_name)
    observations = [item for item in observations_all if float(item[2]) >= float(args.min_coverage)]
    if int(args.top_k) > 0:
        observations = observations[: int(args.top_k)]

    stream = ScanNetStream(seq_name=args.seq_name, backbone=args.backbone)
    errors = stream.validate(require_masks=True)
    if errors:
        raise RuntimeError("; ".join(errors))
    exporter = ScanNetExporter(
        stream,
        output_config=args.output_config,
        export_nn_radius=args.export_nn_radius,
        export_support_mode="mask_backproject",
    )

    object_masks: list[np.ndarray] = []
    scores: list[float] = []
    rows: list[dict[str, float | int]] = []
    total_queries = 0
    total_hits = 0
    dropped_small = 0
    for frame_id, mask_id, coverage in observations:
        point_ids, query_count = exporter._backproject_mask(  # diagnostic tool, keep implementation local.
            int(frame_id),
            int(mask_id),
            nn_radius=float(args.export_nn_radius),
        )
        total_queries += int(query_count)
        total_hits += int(point_ids.shape[0])
        if point_ids.shape[0] < int(args.min_points_per_mask):
            dropped_small += 1
            continue
        mask = np.zeros((exporter.scene_points.shape[0],), dtype=bool)
        mask[point_ids.astype(np.int64)] = True
        object_masks.append(mask)
        scores.append(float(coverage))
        rows.append(
            {
                "frame_id": int(frame_id),
                "mask_id": int(mask_id),
                "coverage": float(coverage),
                "point_count": int(point_ids.shape[0]),
                "query_count": int(query_count),
            }
        )

    if object_masks:
        pred_masks = np.stack(object_masks, axis=1).astype(bool, copy=False)
        pred_score = np.asarray(scores, dtype=np.float32)
    else:
        pred_masks = np.zeros((exporter.scene_points.shape[0], 0), dtype=bool)
        pred_score = np.zeros((0,), dtype=np.float32)
    pred_classes = np.zeros((pred_score.shape[0],), dtype=np.int32)

    pred_dir = Path("data/prediction") / f"{args.output_config}_class_agnostic"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_dir / f"{args.seq_name}.npz",
        pred_masks=pred_masks,
        pred_score=pred_score,
        pred_classes=pred_classes,
    )
    tmp_dir = Path("data/TMP") / args.output_config
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pre_points = np.flatnonzero(pred_masks.any(axis=1)).astype(np.int64)
    np.save(tmp_dir / f"{args.seq_name}_pre_points.npy", pre_points)
    manifest = build_prediction_manifest(
        output_config=args.output_config,
        is_method_result=True,
        is_diagnostic_only=False,
        uses_gt=False,
        gt_usage="none",
        source_configs=[args.debug_root],
        pre_points_policy="recompute",
        support_policy="mask_observation_bank",
        notes="Generated by tools.export_mask_observation_bank from Stream4D local proposal debug files.",
    )
    write_prediction_manifest(args.output_config, manifest)

    summary = {
        "args": vars(args),
        "raw_unique_observations": len(observations_all),
        "filtered_observations": len(observations),
        "exported_observations": int(pred_score.shape[0]),
        "dropped_small": int(dropped_small),
        "union_points": int(pre_points.shape[0]),
        "total_queries": int(total_queries),
        "total_hits": int(total_hits),
        "hit_rate": float(total_hits / max(total_queries, 1)),
        "score_min": float(np.min(pred_score)) if pred_score.size else 0.0,
        "score_mean": float(np.mean(pred_score)) if pred_score.size else 0.0,
        "score_max": float(np.max(pred_score)) if pred_score.size else 0.0,
        "rows": rows[:50],
    }
    out_dir = Path(args.summary_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{args.output_config}_{args.seq_name}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, indent=2, ensure_ascii=False)
    with (out_dir / f"{args.output_config}_latest_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, indent=2, ensure_ascii=False)
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
