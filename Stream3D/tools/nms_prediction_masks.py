from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def _read_seq_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _tmp_path(root: Path, config: str, seq_name: str) -> Path:
    return root / "data" / "TMP" / config / f"{seq_name}_pre_points.npy"


def _overlap_with_kept(
    kept_masks: np.ndarray,
    kept_counts: np.ndarray,
    mask: np.ndarray,
    mask_count: float,
    mode: str,
) -> float:
    if kept_masks.shape[1] == 0 or mask_count <= 0.0:
        return 0.0
    intersections = np.logical_and(kept_masks, mask[:, None]).sum(axis=0).astype(np.float64)
    if mode == "iou":
        denom = kept_counts + mask_count - intersections
    elif mode == "candidate_ioc":
        denom = np.full_like(intersections, mask_count, dtype=np.float64)
    elif mode == "min_ioc":
        denom = np.minimum(kept_counts, mask_count)
    else:
        raise ValueError(f"Unsupported overlap mode: {mode}")
    overlaps = intersections / np.maximum(denom, 1.0)
    return float(np.max(overlaps)) if overlaps.size else 0.0


def _order_indices(scores: np.ndarray, areas: np.ndarray, tie_breaker: str) -> list[int]:
    indices = list(range(scores.shape[0]))
    if tie_breaker == "original":
        return sorted(indices, key=lambda idx: (-float(scores[idx]), idx))
    if tie_breaker == "area_desc":
        return sorted(indices, key=lambda idx: (-float(scores[idx]), -float(areas[idx]), idx))
    if tie_breaker == "area_asc":
        return sorted(indices, key=lambda idx: (-float(scores[idx]), float(areas[idx]), idx))
    raise ValueError(f"Unsupported tie breaker: {tie_breaker}")


def nms_sequence(args: argparse.Namespace, seq_name: str) -> dict[str, float | str]:
    root = Path(args.root)
    pred_in = root / "data" / "prediction" / f"{args.input_config}{args.pred_suffix}" / f"{seq_name}.npz"
    if not pred_in.exists():
        raise FileNotFoundError(f"Missing prediction: {pred_in}")
    with np.load(pred_in) as data:
        masks = data["pred_masks"].astype(bool, copy=False)
        scores = data["pred_score"].astype(np.float32, copy=False)
        classes = data["pred_classes"].astype(np.int32, copy=False)

    if masks.ndim != 2:
        raise ValueError(f"{seq_name}: pred_masks must be 2D, got {masks.shape}")
    areas = masks.sum(axis=0).astype(np.float64)
    kept: list[int] = []
    suppressed: list[dict[str, float | int]] = []
    for idx in _order_indices(scores, areas, args.tie_breaker):
        area = float(areas[idx])
        if int(args.min_area) > 0 and area < float(args.min_area):
            suppressed.append({"idx": int(idx), "reason": "min_area", "area": area, "score": float(scores[idx])})
            continue
        if int(args.max_area) > 0 and area > float(args.max_area):
            suppressed.append({"idx": int(idx), "reason": "max_area", "area": area, "score": float(scores[idx])})
            continue
        if kept:
            kept_masks = masks[:, np.asarray(kept, dtype=np.int64)]
            kept_counts = areas[np.asarray(kept, dtype=np.int64)]
            overlap = _overlap_with_kept(
                kept_masks=kept_masks,
                kept_counts=kept_counts,
                mask=masks[:, idx],
                mask_count=area,
                mode=args.overlap_mode,
            )
            if overlap >= float(args.overlap_threshold):
                suppressed.append(
                    {
                        "idx": int(idx),
                        "reason": "overlap",
                        "overlap": float(overlap),
                        "area": area,
                        "score": float(scores[idx]),
                    }
                )
                continue
        kept.append(idx)
        if int(args.max_instances) > 0 and len(kept) >= int(args.max_instances):
            break

    kept_arr = np.asarray(kept, dtype=np.int64)
    out_masks = masks[:, kept_arr]
    out_scores = scores[kept_arr]
    out_classes = classes[kept_arr]

    pred_out_dir = root / "data" / "prediction" / f"{args.output_config}{args.pred_suffix}"
    pred_out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_out_dir / f"{seq_name}.npz",
        pred_masks=out_masks,
        pred_score=out_scores,
        pred_classes=out_classes,
    )

    tmp_in = _tmp_path(root, args.input_config, seq_name)
    tmp_out_dir = root / "data" / "TMP" / args.output_config
    tmp_out_dir.mkdir(parents=True, exist_ok=True)
    tmp_out = tmp_out_dir / f"{seq_name}_pre_points.npy"
    if tmp_in.exists():
        shutil.copy2(tmp_in, tmp_out)
        tmp_mode = "copied_input_tmp"
    else:
        pre_points = np.flatnonzero(np.any(out_masks, axis=1)).astype(np.int64)
        np.save(tmp_out, pre_points)
        tmp_mode = "recomputed_missing_input_tmp"

    return {
        "seq_name": seq_name,
        "input_config": args.input_config,
        "output_config": args.output_config,
        "num_instances_before": float(masks.shape[1]),
        "num_instances_after": float(out_masks.shape[1]),
        "num_suppressed": float(len(suppressed)),
        "overlap_mode": args.overlap_mode,
        "overlap_threshold": float(args.overlap_threshold),
        "tie_breaker": args.tie_breaker,
        "min_area": int(args.min_area),
        "max_area": int(args.max_area),
        "tmp_mode": tmp_mode,
        "output_union_count": float(np.count_nonzero(np.any(out_masks, axis=1))),
        "suppressed_preview": suppressed[:50],
    }


def aggregate(rows: list[dict[str, float | str]], args: argparse.Namespace) -> dict[str, float | str]:
    numeric_keys = sorted(
        key
        for row in rows
        for key, value in row.items()
        if isinstance(value, (int, float))
    )
    means = {}
    for key in numeric_keys:
        vals = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if vals:
            means[f"mean_{key}"] = float(np.mean(vals))
    return {
        "input_config": args.input_config,
        "output_config": args.output_config,
        "overlap_mode": args.overlap_mode,
        "overlap_threshold": float(args.overlap_threshold),
        "tie_breaker": args.tie_breaker,
        "scenes": len(rows),
        **means,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--input-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--pred-suffix", default="_class_agnostic")
    parser.add_argument("--overlap-mode", default="min_ioc", choices=["iou", "candidate_ioc", "min_ioc"])
    parser.add_argument("--overlap-threshold", type=float, default=0.9)
    parser.add_argument("--tie-breaker", default="original", choices=["original", "area_desc", "area_asc"])
    parser.add_argument("--min-area", type=int, default=0)
    parser.add_argument("--max-area", type=int, default=0)
    parser.add_argument("--max-instances", type=int, default=0)
    parser.add_argument("--summary-root", default="outputs/stream4d_mask_nms_v4_1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root)
    rows = [nms_sequence(args, seq_name) for seq_name in _read_seq_list(Path(args.seq_list))]
    out_dir = root / args.summary_root
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"args": vars(args), "aggregate": aggregate(rows, args), "rows": rows}
    out_path = out_dir / f"{args.output_config}_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[nms-prediction-masks] wrote {out_path}")


if __name__ == "__main__":
    main()
