from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest
from tools.v11_candidate_pool_oracle import (
    _conflict_rate,
    _dedup_candidates,
    _json_safe,
    _load_prediction,
    _read_seq_list,
)


def _aggregate(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float, np.generic)) and value is not None and not isinstance(value, bool)
        }
    )
    return {
        "algorithm": "v13_unsupervised_candidate_pool",
        "pool_name": args.pool_name or args.output_config,
        "output_config": args.output_config,
        "pool_configs": [item.strip() for item in args.pool_configs.split(",") if item.strip()],
        "uses_gt": False,
        "is_method_result": True,
        "num_scenes": int(len(rows)),
        "numeric_mean": {
            key: float(np.mean([float(row[key]) for row in rows if row.get(key) is not None]))
            for key in numeric_keys
            if any(row.get(key) is not None for row in rows)
        },
        "scenes": rows,
    }


def _write_summary(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    summary_root = Path(args.summary_root)
    summary_root.mkdir(parents=True, exist_ok=True)
    prefix = summary_root / f"{args.output_config}_summary"
    prefix.with_suffix(".json").write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    rows = payload["scenes"]
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
    lines = [
        f"# {args.output_config}",
        "",
        "This unsupervised candidate-pool baseline does not read GT. It concatenates existing predictions, filters tiny candidates, and deduplicates by overlap.",
        "",
        "| scene | before | after area | after dedup | union% | conflict | sources |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["scene"]),
                    str(row["num_candidates_before_area_filter"]),
                    str(row["num_candidates_after_area_filter"]),
                    str(row["num_candidates_after_dedup"]),
                    f"{float(row['candidate_union_ratio']) * 100.0:.4f}",
                    f"{float(row['candidate_conflict_rate']) * 100.0:.4f}",
                    str(row["source_counts_after_dedup"]),
                ]
            )
            + " |"
        )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--pool-name", default="")
    parser.add_argument("--pool-configs", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--pred-suffix", default="class_agnostic")
    parser.add_argument("--summary-root", default="outputs/v13_candidate_unsupervised")
    parser.add_argument("--min-candidate-points", type=int, default=100)
    parser.add_argument("--dedup-threshold", type=float, default=0.95)
    parser.add_argument("--dedup-overlap-mode", choices=["iou", "min_ioc", "candidate_ioc"], default="min_ioc")
    parser.add_argument("--max-candidates", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    configs = [item.strip() for item in args.pool_configs.split(",") if item.strip()]
    scenes = _read_seq_list((root / args.seq_list).resolve())
    pred_dir = root / "data" / "prediction" / f"{args.output_config}_class_agnostic"
    tmp_dir = root / "data" / "TMP" / args.output_config
    pred_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        loaded = [_load_prediction(root, config, args.pred_suffix, scene) for config in configs]
        vertex_counts = {int(item["pred_masks"].shape[0]) for item in loaded}
        if len(vertex_counts) != 1:
            raise ValueError(f"{scene}: prediction vertex count mismatch: {sorted(vertex_counts)}")
        all_masks = np.concatenate([item["pred_masks"] for item in loaded], axis=1)
        all_scores = np.concatenate([item["pred_score"] for item in loaded], axis=0)
        all_classes = np.concatenate([item["pred_classes"] for item in loaded], axis=0)
        all_sources: list[str] = []
        for config, item in zip(configs, loaded):
            all_sources.extend([config] * int(item["pred_masks"].shape[1]))
        raw_union = np.flatnonzero(np.any(all_masks, axis=1)).astype(np.int64)
        pool_masks, pool_scores, pool_classes, pool_sources, dedup_diag = _dedup_candidates(
            all_masks,
            all_scores,
            all_classes,
            all_sources,
            support=raw_union,
            min_area=int(args.min_candidate_points),
            threshold=float(args.dedup_threshold),
            mode=args.dedup_overlap_mode,
        )
        if int(args.max_candidates) > 0 and pool_masks.shape[1] > int(args.max_candidates):
            areas = pool_masks[raw_union, :].sum(axis=0)
            order = sorted(
                range(pool_masks.shape[1]),
                key=lambda idx: (float(pool_scores[idx]), int(areas[idx]), -int(idx)),
                reverse=True,
            )[: int(args.max_candidates)]
            keep = np.asarray(order, dtype=np.int64)
            pool_masks = pool_masks[:, keep]
            pool_scores = pool_scores[keep]
            pool_classes = pool_classes[keep]
            pool_sources = [pool_sources[int(idx)] for idx in keep.tolist()]
        support = np.flatnonzero(np.any(pool_masks, axis=1)).astype(np.int64)
        np.savez_compressed(
            pred_dir / f"{scene}.npz",
            pred_masks=pool_masks,
            pred_score=pool_scores,
            pred_classes=pool_classes,
        )
        np.save(tmp_dir / f"{scene}_pre_points.npy", support)
        source_counts: dict[str, int] = {}
        for source in pool_sources:
            source_counts[source] = source_counts.get(source, 0) + 1
        rows.append(
            {
                "scene": scene,
                **dedup_diag,
                "num_candidates_after_max": int(pool_masks.shape[1]),
                "candidate_union_points": int(support.shape[0]),
                "candidate_union_ratio": float(support.shape[0] / max(int(pool_masks.shape[0]), 1)),
                "candidate_conflict_rate": _conflict_rate(pool_masks[support, :]) if support.size else 0.0,
                "source_counts_after_dedup": source_counts,
            }
        )
    payload = _aggregate(rows, args)
    _write_summary(args, payload)
    manifest = build_prediction_manifest(
        root=root,
        output_config=args.output_config,
        is_method_result=True,
        is_diagnostic_only=False,
        uses_gt=False,
        gt_usage="none",
        source_configs=configs,
        pre_points_policy="recompute",
        support_policy="v13_unsupervised_candidate_pool",
        notes="Unsupervised no-GT candidate-pool baseline for v13 failure attribution.",
        extra={
            "algorithm": "v13_unsupervised_candidate_pool",
            "eval_policy": "own_recompute_paper_style",
            "prediction_config": args.output_config,
            "pre_points_config": args.output_config,
            "support_source": "own",
            "geometry_source": "mixed_existing_predictions",
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic": False,
            "is_method_result": True,
            "is_diagnostic_only": False,
            "summary_path": str(Path(args.summary_root) / f"{args.output_config}_summary.json"),
        },
    )
    write_prediction_manifest(args.output_config, manifest, root=root)
    print(json.dumps(_json_safe(payload["numeric_mean"]), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
